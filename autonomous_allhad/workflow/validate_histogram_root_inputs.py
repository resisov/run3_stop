#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uproot

from build_flat_boosted_recoil_hists import READ_BRANCHES


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_root(path: Path, step_size: int) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    metadata_path = path.with_suffix(".json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    datasets = sorted((metadata.get("datasets") or {}).keys())
    result: dict[str, Any] = {
        "path": str(path),
        "metadata": str(metadata_path),
        "metadata_events_written": int(metadata.get("events_written") or 0),
        "dataset": ",".join(datasets),
        "datasets": datasets,
        "checked_at": checked_at,
    }
    try:
        with uproot.open(path) as root_file:
            tree = root_file["Events"]
            tree_entries = int(tree.num_entries)
            result["tree_entries"] = tree_entries
            if tree_entries == 0:
                result["status"] = "zero_entries"
                return result
            present = set(tree.keys())
            branches = [branch for branch in READ_BRANCHES if branch in present]
            entries_read = 0
            for chunk in tree.iterate(branches, step_size=step_size, library="ak"):
                entries_read += len(chunk["dataset_id"])
            result["entries_read"] = entries_read
            if entries_read != tree_entries or entries_read != result["metadata_events_written"]:
                raise ValueError(
                    f"entry mismatch: read={entries_read}, tree={tree_entries}, metadata={result['metadata_events_written']}"
                )
            result["status"] = "valid"
    except Exception as exc:
        result.update(
            {
                "status": "bad",
                "failure_stage": "histogram_branch_iteration",
                "exception_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "first_failure_time": checked_at,
                "last_failure_time": checked_at,
                "alternate_access_attempted": False,
                "permanently_skipped": False,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fully iterate histogram branches in flat ROOT inputs.")
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bad-text-output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=50000)
    args = parser.parse_args()

    roots = [Path(line.strip()) for line in args.input_list.read_text().splitlines() if line.strip()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(validate_root, path, args.step_size) for path in roots]
        results = [future.result() for future in futures]
    bad = [result for result in results if result["status"] == "bad"]
    zero = [result for result in results if result["status"] == "zero_entries"]
    valid = [result for result in results if result["status"] == "valid"]
    payload = {
        "status": "complete_with_bad_files" if bad else "complete",
        "input_count": len(roots),
        "valid_count": len(valid),
        "zero_entry_count": len(zero),
        "bad_count": len(bad),
        "valid_entries": sum(int(result.get("entries_read") or 0) for result in valid),
        "bad_tree_entries": sum(int(result.get("tree_entries") or 0) for result in bad),
        "zero_entry_roots": [result["path"] for result in zero],
        "bad_files": bad,
    }
    write_json(args.output, payload)
    args.bad_text_output.parent.mkdir(parents=True, exist_ok=True)
    args.bad_text_output.write_text("".join(f"{result['path']}\n" for result in bad))
    print(json.dumps({key: payload[key] for key in ("status", "input_count", "valid_count", "zero_entry_count", "bad_count", "valid_entries", "bad_tree_entries")}, sort_keys=True))
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
