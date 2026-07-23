#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_NAMES = {
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring_output",
    7: "suspended",
}
OK_METADATA_STATUSES = {"complete", "complete_with_bad_files"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_names(arguments_path: Path) -> list[str]:
    names = [
        line.split()[0]
        for line in arguments_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate job names in {arguments_path}")
    return names


def query_queue(cluster_id: int, schedd: str) -> dict[str, Any]:
    command = [
        "condor_q",
        "-name",
        schedd,
        "-constraint",
        f"ClusterId == {cluster_id}",
        "-af",
        "ProcId",
        "JobStatus",
        "HoldReason",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "counts": {},
            "active_jobs": 0,
            "held_preview": [],
            "malformed_preview": [],
            "stderr_tail": f"condor_q timeout: {exc}",
        }
    counts: Counter[str] = Counter()
    held: list[dict[str, Any]] = []
    malformed: list[str] = []
    for raw in result.stdout.splitlines():
        parts = raw.split(maxsplit=2)
        if len(parts) < 2:
            if raw.strip():
                malformed.append(raw)
            continue
        try:
            proc_id = int(parts[0])
            status_id = int(parts[1])
        except ValueError:
            malformed.append(raw)
            continue
        status = STATUS_NAMES.get(status_id, f"unknown_{status_id}")
        counts[status] += 1
        if status == "held" and len(held) < 20:
            held.append(
                {
                    "proc_id": proc_id,
                    "reason": parts[2] if len(parts) == 3 else "undefined",
                }
            )
    return {
        "command": command,
        "returncode": result.returncode,
        "counts": dict(sorted(counts.items())),
        "active_jobs": sum(counts.values()),
        "held_preview": held,
        "malformed_preview": malformed[:20],
        "stderr_tail": result.stderr[-2000:],
    }


def scan_outputs(campaign: Path, expected: list[str]) -> dict[str, Any]:
    output_dir = campaign / "outputs"
    roots = {path.stem: path for path in output_dir.glob("*.root")}
    metadata = {path.stem: path for path in output_dir.glob("*.json")}
    expected_set = set(expected)
    valid: set[str] = set()
    invalid: list[dict[str, str]] = []
    events_read = 0
    events_written = 0
    bad_file_records = 0
    for name, meta_path in metadata.items():
        if name not in expected_set:
            continue
        try:
            payload = json.loads(meta_path.read_text())
        except Exception as exc:
            invalid.append({"name": name, "reason": f"metadata_read_failed: {exc}"})
            continue
        events_read += int(payload.get("events_read") or 0)
        events_written += int(payload.get("events_written") or 0)
        bad_file_records += len(payload.get("bad_files") or [])
        reason = None
        if payload.get("status") not in OK_METADATA_STATUSES:
            reason = f"bad_status:{payload.get('status')}"
        elif payload.get("files_processed") != payload.get("files_attempted"):
            reason = (
                f"file_count_mismatch:{payload.get('files_processed')}/"
                f"{payload.get('files_attempted')}"
            )
        elif name not in roots:
            reason = "root_missing"
        else:
            try:
                if roots[name].stat().st_size <= 0:
                    reason = "root_empty"
            except OSError as exc:
                reason = f"root_stat_failed:{exc}"
        if reason:
            invalid.append({"name": name, "reason": reason})
        else:
            valid.add(name)
    missing = expected_set - valid
    return {
        "expected": len(expected),
        "root_files": len(roots),
        "metadata_files": len(metadata),
        "valid_pairs": len(valid),
        "invalid_pairs": len(invalid),
        "invalid_preview": invalid[:20],
        "missing_pairs": len(missing),
        "missing_preview": sorted(missing)[:20],
        "events_read": events_read,
        "events_written": events_written,
        "bad_file_records": bad_file_records,
    }


def campaign_status(queue: dict[str, Any], outputs: dict[str, Any]) -> str:
    if outputs["valid_pairs"] == outputs["expected"]:
        return "complete"
    if queue["returncode"] != 0:
        return "queue_query_failed"
    if queue["counts"].get("held", 0):
        return "running_with_held_jobs"
    if queue["active_jobs"]:
        return "running"
    return "incomplete_no_active_jobs"


def render_summary(state: dict[str, Any]) -> str:
    queue = state["queue"]
    outputs = state["outputs"]
    counts = queue["counts"]
    return (
        "# Condor Campaign Monitor\n\n"
        f"- Updated: {state['updated_at']}\n"
        f"- Status: {state['status']}\n"
        f"- Cluster: {state['cluster_id']} on {state['schedd']}\n"
        f"- Queue: idle={counts.get('idle', 0)}, running={counts.get('running', 0)}, "
        f"held={counts.get('held', 0)}, removed={counts.get('removed', 0)}\n"
        f"- Valid ROOT/JSON pairs: {outputs['valid_pairs']}/{outputs['expected']}\n"
        f"- Invalid pairs: {outputs['invalid_pairs']}\n"
        f"- Events read/written: {outputs['events_read']}/{outputs['events_written']}\n"
        f"- Bad-file records: {outputs['bad_file_records']}\n"
        f"- Analysis paths written: no\n"
    )


def snapshot(campaign: Path, cluster_id: int, schedd: str) -> dict[str, Any]:
    expected = expected_names(campaign / "condor" / "arguments.txt")
    queue = query_queue(cluster_id, schedd)
    outputs = scan_outputs(campaign, expected)
    state = {
        "schema_version": "condor_campaign_monitor_v1",
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "campaign": str(campaign),
        "cluster_id": cluster_id,
        "schedd": schedd,
        "status": campaign_status(queue, outputs),
        "queue": queue,
        "outputs": outputs,
        "analysis_paths_written": False,
    }
    atomic_write(campaign / "monitor_state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    atomic_write(campaign / "latest_summary.md", render_summary(state))
    with (campaign / "monitor_history.jsonl").open("a") as handle:
        handle.write(json.dumps(state, sort_keys=True) + "\n")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a Condor campaign without modifying analysis outputs.")
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--cluster-id", required=True, type=int)
    parser.add_argument("--schedd", default="bigbird24")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    if not str(campaign).startswith("/eos/"):
        raise ValueError(f"Campaign must be on EOS: {campaign}")
    while True:
        try:
            state = snapshot(campaign, args.cluster_id, args.schedd)
        except Exception as exc:
            state = {
                "schema_version": "condor_campaign_monitor_v1",
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "campaign": str(campaign),
                "cluster_id": args.cluster_id,
                "schedd": args.schedd,
                "status": "monitor_snapshot_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "analysis_paths_written": False,
            }
            atomic_write(
                campaign / "monitor_state.json",
                json.dumps(state, indent=2, sort_keys=True) + "\n",
            )
            with (campaign / "monitor_history.jsonl").open("a") as handle:
                handle.write(json.dumps(state, sort_keys=True) + "\n")
        print(json.dumps(state, sort_keys=True), flush=True)
        if args.once or state["status"] in {"complete", "incomplete_no_active_jobs"}:
            return 0 if state["status"] == "complete" or args.once else 2
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    raise SystemExit(main())
