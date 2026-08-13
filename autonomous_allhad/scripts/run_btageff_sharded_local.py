#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from coffea import processor
from coffea.util import load, save


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)


def output_key(dataset: str) -> str:
    parts = dataset.strip("/").split("/")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}-{parts[1]}"


def validate(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"valid": False, "error": "missing_or_empty_output"}
    try:
        payload = load(path)
        histogram = payload.get("UParTAK4")
        if histogram is None:
            return {"valid": False, "error": "UParTAK4_missing"}
        values = np.asarray(histogram.values(flow=True), dtype=float)
        total = float(np.sum(values[np.isfinite(values)]))
        if not np.isfinite(total) or total <= 0:
            return {"valid": False, "error": f"invalid_histogram_sum:{total}"}
        return {
            "valid": True,
            "histogram_sum": total,
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}:{exc}"}


def run_shard(
    repo: Path,
    campaign: Path,
    python: Path,
    shard: dict[str, Any],
) -> dict[str, Any]:
    shard_id = shard["shard_id"]
    output = campaign / "shard_outputs" / f"{shard_id}.futures"
    existing = validate(output)
    if existing["valid"]:
        return {**shard, "action": "reused", **existing}
    log = campaign / "shard_logs" / f"{shard_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "run.py",
        "-p",
        "btageff2024",
        "-m",
        "unused",
        "--metadata-path",
        shard["metadata_path"],
        "-d",
        shard["key"],
        "-w",
        "1",
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment["X509_USER_PROXY"] = str(
        repo / "autonomous_allhad/x509up_u147757"
    )
    started = time.time()
    with log.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=repo / "analysis",
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return {
        **shard,
        "action": "processed",
        "returncode": result.returncode,
        "wall_time_s": time.time() - started,
        "log": str(log),
        **validate(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--metadata-name", required=True)
    parser.add_argument("--files-per-shard", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=12)
    args = parser.parse_args()

    repo = Path(args.repo)
    campaign = Path(args.campaign)
    python = Path(args.python)
    metadata_path = repo / "analysis/metadata" / f"{args.metadata_name}.json.gz"
    with gzip.open(metadata_path, "rt") as handle:
        metadata = json.load(handle)

    shards: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for dataset in sorted(metadata):
        info = metadata[dataset]
        files = list(info["files"])
        key = output_key(dataset)
        for shard_index, start in enumerate(range(0, len(files), args.files_per_shard)):
            shard_files = files[start : start + args.files_per_shard]
            duplicate = seen_files.intersection(shard_files)
            if duplicate:
                raise RuntimeError(f"duplicate source files: {sorted(duplicate)[:3]}")
            seen_files.update(shard_files)
            shard_id = f"{key}__{shard_index:05d}"
            shard_metadata = campaign / "shard_metadata" / f"{shard_id}.json.gz"
            shard_info = dict(info)
            shard_info["files"] = shard_files
            if not shard_metadata.exists():
                write_gzip_json(shard_metadata, {key: shard_info})
            shards.append(
                {
                    "dataset": dataset,
                    "key": key,
                    "shard_id": shard_id,
                    "shard_index": shard_index,
                    "source_files": shard_files,
                    "source_file_count": len(shard_files),
                    "metadata_path": str(shard_metadata),
                }
            )

    manifest = {
        "schema": "btageff_sharded_local_manifest_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata": str(metadata_path),
        "files_per_shard": args.files_per_shard,
        "max_parallel": args.max_parallel,
        "dataset_count": len(metadata),
        "source_file_count": len(seen_files),
        "shard_count": len(shards),
        "shards": shards,
    }
    write_json(campaign / "shard_manifest.json", manifest)

    state_path = campaign / "btageff_state.json"
    state: dict[str, Any] = {
        "schema": "btageff_sharded_local_state_v1",
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_datasets": len(metadata),
        "expected_source_files": len(seen_files),
        "expected_shards": len(shards),
        "completed_valid": 0,
        "failed_or_invalid": 0,
        "results": {},
    }
    write_json(state_path, state)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.max_parallel)
    ) as pool:
        futures = {
            pool.submit(run_shard, repo, campaign, python, shard): shard
            for shard in shards
        }
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **shard,
                    "valid": False,
                    "error": f"runner:{type(exc).__name__}:{exc}",
                }
            state["results"][shard["shard_id"]] = result
            state["completed_valid"] = sum(
                bool(item.get("valid")) for item in state["results"].values()
            )
            state["failed_or_invalid"] = sum(
                not bool(item.get("valid")) for item in state["results"].values()
            )
            state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(state_path, state)

    if state["completed_valid"] != len(shards) or state["failed_or_invalid"]:
        state["status"] = "incomplete"
        state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(state_path, state)
        return 2

    merged_results: dict[str, Any] = {}
    for dataset in sorted(metadata):
        key = output_key(dataset)
        paths = [
            campaign / "shard_outputs" / f"{item['shard_id']}.futures"
            for item in shards
            if item["dataset"] == dataset
        ]
        merged = processor.accumulate(load(path) for path in paths)
        campaign_output = campaign / "merged" / f"{key}.futures"
        campaign_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = campaign_output.with_suffix(".futures.tmp")
        save(merged, temporary)
        os.replace(temporary, campaign_output)
        merged_validation = validate(campaign_output)
        if not merged_validation["valid"]:
            raise RuntimeError(f"invalid merged output for {dataset}: {merged_validation}")
        final_output = repo / "analysis/hists/btageff2024" / f"{key}.futures"
        final_output.parent.mkdir(parents=True, exist_ok=True)
        final_temporary = final_output.with_suffix(".futures.tmp")
        shutil.copy2(campaign_output, final_temporary)
        os.replace(final_temporary, final_output)
        merged_results[dataset] = {
            "key": key,
            "shard_count": len(paths),
            "output": str(final_output),
            **merged_validation,
        }

    state["status"] = "complete"
    state["merged_results"] = merged_results
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print(json.dumps({
        "status": state["status"],
        "completed_valid": state["completed_valid"],
        "expected_shards": len(shards),
        "merged_datasets": len(merged_results),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
