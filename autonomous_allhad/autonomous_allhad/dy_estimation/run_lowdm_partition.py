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


def run_partition(
    repo: Path,
    manifest_path: Path,
    output_path: Path,
    jobs: int,
    max_tasks: int | None,
) -> dict[str, Any]:
    """Run or reuse one partition and return a machine-readable result."""

    manifest = read_json(manifest_path)
    if output_path.is_file():
        existing = read_json(output_path)
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
            return {"status": "complete", "reused": str(output_path)}
    tasks = []
    for record in manifest.get("tasks") or []:
        task = dict(record)
        task["repo"] = str(repo)
        tasks.append(task)
    if max_tasks is not None:
        tasks = tasks[: max(0, max_tasks)]
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

    if jobs <= 1:
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(merged, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, output_path)
    return {
        "status": merged["status"],
        "output": str(output_path),
        "summary": merged["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected",
        type=Path,
        help="Run every partition listed by one expected.json locally.",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--partition-workers",
        type=int,
        default=12,
        help="Concurrent single-process partitions used with --expected.",
    )
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args(argv)

    if args.expected:
        if args.manifest or args.output or args.max_tasks is not None:
            parser.error(
                "--expected cannot be combined with --manifest, --output, or --max-tasks"
            )
        expected = read_json(args.expected)
        records = list(expected.get("manifests") or [])
        results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.partition_workers)
        ) as executor:
            futures = {
                executor.submit(
                    run_partition,
                    args.repo,
                    Path(record["manifest"]),
                    Path(record["output"]),
                    1,
                    None,
                ): str(record["stem"])
                for record in records
            }
            for future in concurrent.futures.as_completed(futures):
                stem = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "incomplete",
                        "stem": stem,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                result.setdefault("stem", stem)
                results.append(result)
                if len(results) % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "completed_partitions": len(results),
                                "expected_partitions": len(records),
                                "failures": sum(
                                    item.get("status") != "complete"
                                    for item in results
                                ),
                            }
                        ),
                        flush=True,
                    )
        failures = [item for item in results if item.get("status") != "complete"]
        summary = {
            "schema_version": "dy_estimation_local_refinement_run_v1",
            "status": "complete" if not failures else "incomplete",
            "expected": str(args.expected),
            "expected_partitions": len(records),
            "completed_partitions": len(results) - len(failures),
            "failures": failures,
        }
        summary_path = args.expected.parent / "local_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({**summary, "output": str(summary_path)}))
        return 0 if not failures else 2

    if not args.manifest or not args.output:
        parser.error("provide --expected or both --manifest and --output")
    result = run_partition(
        args.repo,
        args.manifest,
        args.output,
        args.jobs,
        args.max_tasks,
    )
    print(json.dumps(result))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
