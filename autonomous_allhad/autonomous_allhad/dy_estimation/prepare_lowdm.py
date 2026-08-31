#!/usr/bin/env python3
"""Prepare auditable Condor partitions for exact Low-dM DY sparse recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .lowdm_recovery import read_json, source_map


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def lexical_absolute(path: Path) -> Path:
    """Make paths absolute without rewriting the /eos/user namespace."""
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--combined",
        type=Path,
        help="Single feature-stage artifact containing both DY2E and DY2M.",
    )
    parser.add_argument("--ee", type=Path)
    parser.add_argument("--mumu", type=Path)
    parser.add_argument("--shard-bundle", type=Path, required=True)
    parser.add_argument("--source-list-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files-per-job", type=int, default=5)
    parser.add_argument("--max-span", type=int, default=50000)
    parser.add_argument("--max-gap", type=int, default=5000)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python"),
    )
    parser.add_argument("--memory-mb", type=int, default=6000)
    args = parser.parse_args(argv)

    if args.combined:
        if args.ee or args.mumu:
            parser.error("--combined cannot be used with --ee/--mumu")
        feature_inputs = [("combined", read_json(args.combined))]
    else:
        if not args.ee or not args.mumu:
            parser.error("provide --combined or both --ee and --mumu")
        feature_inputs = [
            ("DY2E", read_json(args.ee)),
            ("DY2M", read_json(args.mumu)),
        ]
    candidates: dict[str, list[dict[str, Any]]] = {}
    for expected_channel, feature in feature_inputs:
        if feature.get("status") != "feature_stage_complete":
            raise SystemExit(f"{expected_channel}: incomplete feature input")
        for file_id, records in (feature.get("sparse_low_candidates") or {}).items():
            target = candidates.setdefault(str(file_id), [])
            for record in records:
                copied = dict(record)
                record_channel = copied.get("channel")
                if record_channel not in ("DY2E", "DY2M"):
                    raise RuntimeError(f"{file_id}: channel mismatch")
                if (
                    expected_channel != "combined"
                    and record_channel != expected_channel
                ):
                    raise RuntimeError(f"{file_id}: channel mismatch")
                target.append(copied)

    source_roots: list[str] = []
    source_lists: list[str] = []
    for directory in args.source_list_dir:
        for source_list in sorted(directory.glob("*.txt")):
            source_lists.append(str(source_list))
            source_roots.extend(
                line.strip()
                for line in source_list.read_text().splitlines()
                if line.strip()
            )
    wanted_ids = {int(value) for value in candidates}
    mapping = source_map(args.shard_bundle, source_roots, wanted_ids)
    unresolved = sorted(wanted_ids - set(mapping))
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} candidate source IDs unresolved: {unresolved[:20]}"
        )

    output_dir = args.output_dir
    manifests_dir = output_dir / "manifests"
    outputs_dir = output_dir / "outputs"
    logs_dir = output_dir / "logs"
    for directory in (manifests_dir, outputs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "file_id": int(file_id),
            "source": mapping[int(file_id)],
            "candidates": records,
            "max_span": args.max_span,
            "max_gap": args.max_gap,
        }
        for file_id, records in sorted(candidates.items(), key=lambda item: int(item[0]))
    ]
    files_per_job = max(1, int(args.files_per_job))
    queue_lines: list[str] = []
    manifests: list[dict[str, Any]] = []
    for index in range(0, len(tasks), files_per_job):
        partition = tasks[index : index + files_per_job]
        stem = f"part_{index // files_per_job:04d}"
        manifest_path = manifests_dir / f"{stem}.json"
        output_path = outputs_dir / f"{stem}.json"
        manifest = {
            "schema_version": "dy_estimation_lowdm_manifest_2024_v1",
            "stem": stem,
            "tasks": partition,
            "summary": {
                "candidate_files": len(partition),
                "candidate_events": sum(len(task["candidates"]) for task in partition),
            },
        }
        write_json(manifest_path, manifest)
        queue_lines.append(f"{manifest_path}\t{output_path}\t{stem}")
        manifests.append(
            {
                "stem": stem,
                "manifest": str(manifest_path),
                "output": str(output_path),
                **manifest["summary"],
            }
        )

    (output_dir / "queue.tsv").write_text("\n".join(queue_lines) + "\n")
    repo = lexical_absolute(args.repo)
    run_directory = repo / "autonomous_allhad"
    submit_path = output_dir / "submit.sub"
    submit_path.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {args.python}",
                (
                    "arguments = -m autonomous_allhad.dy_estimation "
                    "run-lowdm-partition "
                    f"--repo {repo} "
                    "--manifest $(manifest_path) --output $(output_path) --jobs 1"
                ),
                f"initialdir = {run_directory}",
                f'environment = "PYTHONPATH={run_directory};PYTHONUNBUFFERED=1"',
                "should_transfer_files = NO",
                "request_cpus = 1",
                f"request_memory = {args.memory_mb}",
                '+JobFlavour = "tomorrow"',
                '+JobBatchName = "DY_RZ_lowdm_exact"',
                f"output = {logs_dir}/$(stem).out",
                f"error = {logs_dir}/$(stem).err",
                f"log = {logs_dir}/$(stem).log",
                "on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)",
                (
                    "queue manifest_path, output_path, stem from "
                    f"{output_dir / 'queue.tsv'}"
                ),
                "",
            ]
        )
    )
    expected = {
        "schema_version": "dy_estimation_lowdm_expected_2024_v1",
        "status": "prepared",
        "inputs": (
            {"combined": str(args.combined)}
            if args.combined
            else {"ee": str(args.ee), "mumu": str(args.mumu)}
        ),
        "candidate_files": len(tasks),
        "candidate_events": sum(len(task["candidates"]) for task in tasks),
        "partitions": len(manifests),
        "files_per_job": files_per_job,
        "source_lists": source_lists,
        "submit": str(submit_path),
        "manifests": manifests,
    }
    write_json(output_dir / "expected.json", expected)
    print(json.dumps({key: expected[key] for key in ("candidate_files", "candidate_events", "partitions")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
