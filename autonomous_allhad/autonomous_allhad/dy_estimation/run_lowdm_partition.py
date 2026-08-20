#!/usr/bin/env python3
"""Run one exact Low-dM sparse-recovery partition."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

from .lowdm_recovery import (
    merge_tree,
    process_source,
    read_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args(argv)

    manifest = read_json(args.manifest)
    if args.output.is_file():
        existing = read_json(args.output)
        existing_summary = existing.get("summary") or {}
        manifest_summary = manifest.get("summary") or {}
        if (
            existing.get("status") == "complete"
            and existing.get("stem") == manifest.get("stem")
            and int(existing_summary.get("candidate_files", -1))
            == int(manifest_summary.get("candidate_files", -2))
            and int(existing_summary.get("candidate_events", -1))
            == int(manifest_summary.get("candidate_events", -2))
            and int(existing_summary.get("matched_events", -1))
            == int(manifest_summary.get("candidate_events", -2))
        ):
            print(json.dumps({"status": "complete", "reused": str(args.output)}))
            return 0
    tasks = []
    for record in manifest.get("tasks") or []:
        task = dict(record)
        task["repo"] = str(args.repo)
        tasks.append(task)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]
    merged: dict[str, Any] = {
        "schema_version": "dy_estimation_lowdm_partition_2024_v1",
        "status": "running",
        "stem": manifest.get("stem"),
        "raw": {},
        "mll": {},
        "summary": {
            "candidate_files": len(tasks),
            "candidate_events": sum(len(task["candidates"]) for task in tasks),
            "completed_files": 0,
            "matched_events": 0,
            "selected_events": 0,
            "read_windows": 0,
            "failures": [],
        },
    }
    def consume(task: dict[str, Any], result: dict[str, Any]) -> None:
        summary = result["summary"]
        merged["summary"]["completed_files"] += 1
        merged["summary"]["matched_events"] += int(summary["matched"])
        merged["summary"]["selected_events"] += int(summary["selected"])
        merged["summary"]["read_windows"] += int(summary["windows"])
        merge_tree(merged["raw"], result["raw"])
        merge_tree(merged["mll"], result["mll"])

    if args.jobs <= 1:
        # XRootD's native client is not reliably fork-safe on every worker
        # node.  The single-process path is the production default; Condor
        # provides file-level concurrency across partitions.
        for task in tasks:
            try:
                consume(task, process_source(task))
            except Exception as exc:
                merged["summary"]["failures"].append(
                    {
                        "file_id": task["file_id"],
                        "file_path": task["source"]["file_path"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_source, task): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    consume(task, future.result())
                except Exception as exc:
                    merged["summary"]["failures"].append(
                        {
                            "file_id": task["file_id"],
                            "file_path": task["source"]["file_path"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    complete = (
        not merged["summary"]["failures"]
        and merged["summary"]["completed_files"] == merged["summary"]["candidate_files"]
        and merged["summary"]["matched_events"] == merged["summary"]["candidate_events"]
    )
    merged["status"] = "complete" if complete else "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(merged, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, args.output)
    print(json.dumps({"status": merged["status"], "output": str(args.output), "summary": merged["summary"]}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
