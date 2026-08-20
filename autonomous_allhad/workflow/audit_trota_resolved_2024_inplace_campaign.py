#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a 2024/2025 in-place TROTA campaign.")
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--target-year", type=int, choices=(2024, 2025), default=2024)
    args = parser.parse_args()
    schema_version = f"trota_topresolved_{args.target_year}_inplace_campaign_v1"
    campaign = args.campaign_dir.absolute()
    manifest = json.loads((campaign / "input_manifest.json").read_text())
    inputs = manifest["inputs"]
    timestamp = now()

    complete: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    total_events = 0
    total_candidates = 0
    total_selected = 0
    total_bytes_added = 0
    for item in inputs:
        metadata_path = Path(item["job_metadata"])
        if not metadata_path.is_file():
            pending.append({"name": item["name"], "input": item["input_root"]})
            continue
        try:
            payload = json.loads(metadata_path.read_text())
        except Exception as exc:
            failed.append(
                {
                    "name": item["name"],
                    "input": item["input_root"],
                    "failure_stage": "metadata_read",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "first_failure_time": timestamp,
                    "last_failure_time": timestamp,
                    "alternate_access_attempted": False,
                    "permanently_skipped": False,
                }
            )
            continue
        if payload.get("status") not in (
            "complete",
            "already_complete",
            "recovered_complete",
        ):
            failed.append(
                {
                    "name": item["name"],
                    "input": item["input_root"],
                    "failure_stage": "worker",
                    "exception_type": payload.get("error_type", "WorkerFailure"),
                    "error": str(payload.get("error", "non-complete worker metadata"))[:1000],
                    "first_failure_time": payload.get("failed_at", timestamp),
                    "last_failure_time": payload.get("failed_at", timestamp),
                    "alternate_access_attempted": False,
                    "permanently_skipped": False,
                }
            )
            continue
        complete.append({"name": item["name"], "input": item["input_root"], "metadata": str(metadata_path)})
        counts = payload.get("counts", payload.get("marker", {}))
        total_events += int(counts.get("events", counts.get("events_entries", 0)) or 0)
        total_candidates += int(counts.get("candidates_evaluated", 0) or 0)
        total_selected += int(counts.get("selected_candidates", 0) or 0)
        total_bytes_added += int(payload.get("storage", {}).get("bytes_added", 0) or 0)

    if failed:
        status = "failed_outputs_present"
    elif pending:
        status = "running_or_pending"
    else:
        status = "complete"
    state = {
        "schema_version": schema_version,
        "application_year": args.target_year,
        "updated_at": timestamp,
        "status": status,
        "input_files": len(inputs),
        "complete": len(complete),
        "failed": len(failed),
        "pending": len(pending),
        "events": total_events,
        "candidates_evaluated": total_candidates,
        "selected_candidates": total_selected,
        "bytes_added": total_bytes_added,
        "input_digest": manifest["input_digest"],
    }
    previous_state = {}
    state_path = campaign / "state.json"
    if state_path.is_file():
        try:
            previous_state = json.loads(state_path.read_text())
        except Exception:
            previous_state = {}
    write_json(state_path, {**previous_state, **state})
    write_json(
        campaign / "file_validation_summary.json",
        {
            **state,
            "valid": len(complete),
            "invalid": len(failed),
            "pending_files": pending,
        },
    )
    write_json(
        campaign / "bad_files.json",
        {"schema_version": schema_version, "application_year": args.target_year, "files": failed},
    )
    atomic_write(campaign / "bad_files.txt", "".join(f"{item['input']}\n" for item in failed))
    with (campaign / "history.jsonl").open("a") as history:
        history.write(json.dumps({"time": timestamp, "event": "campaign_audit", **state}, sort_keys=True) + "\n")
    atomic_write(
        campaign / "latest_summary.md",
        "\n".join(
            [
                f"# {args.target_year} TROTA TopResolved in-place campaign",
                "",
                f"Status: {status}",
                "",
                f"- Complete: {len(complete):,} / {len(inputs):,}",
                f"- Failed: {len(failed):,}",
                f"- Pending: {len(pending):,}",
                f"- Events recorded complete: {total_events:,}",
                f"- Candidates evaluated: {total_candidates:,}",
                f"- 1% WP candidates stored: {total_selected:,}",
                f"- Bytes appended: {total_bytes_added:,}",
                "- Original `Events` trees are preserved and validated per worker.",
                "",
            ]
        ),
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
