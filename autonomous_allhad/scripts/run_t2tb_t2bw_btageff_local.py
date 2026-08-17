#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from coffea.util import load


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def output_key(dataset: str) -> str:
    parts = dataset.strip("/").split("/")
    if len(parts) == 1:
        return parts[0]
    primary, production = parts[:2]
    return f"{primary}-{production}"


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


def run_one(
    repo: Path,
    campaign: Path,
    python: Path,
    metadata_name: str,
    dataset: str,
    workers: int,
) -> dict[str, Any]:
    key = output_key(dataset)
    output = repo / "analysis/hists/btageff2024" / f"{key}.futures"
    existing = validate(output)
    if existing["valid"]:
        return {"dataset": dataset, "key": key, "action": "reused", **existing}
    log = campaign / "btageff_logs" / f"{key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "run.py",
        "-p",
        "btageff2024",
        "-m",
        metadata_name,
        "-d",
        key,
        "-w",
        str(workers),
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
        "dataset": dataset,
        "key": key,
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
    parser.add_argument("--metadata-name", default="KNU_2024_t2tb_t2bw")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--workers-per-dataset", type=int, default=1)
    args = parser.parse_args()
    repo = Path(args.repo)
    campaign = Path(args.campaign)
    python = Path(args.python)
    metadata_path = repo / "analysis/metadata" / f"{args.metadata_name}.json.gz"
    with gzip.open(metadata_path, "rt") as handle:
        metadata = json.load(handle)
    datasets = sorted(metadata)
    state_path = campaign / "btageff_state.json"
    state: dict[str, Any] = {
        "schema": "t2tb_t2bw_btageff_local_state_v1",
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_datasets": len(datasets),
        "results": {},
    }
    write_json(state_path, state)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.max_parallel)
    ) as pool:
        futures = {
            pool.submit(
                run_one,
                repo,
                campaign,
                python,
                args.metadata_name,
                dataset,
                max(1, args.workers_per_dataset),
            ): dataset
            for dataset in datasets
        }
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "dataset": dataset,
                    "key": output_key(dataset),
                    "valid": False,
                    "error": f"runner:{type(exc).__name__}:{exc}",
                }
            state["results"][dataset] = result
            state["completed_valid"] = sum(
                bool(item.get("valid")) for item in state["results"].values()
            )
            state["failed_or_invalid"] = sum(
                not bool(item.get("valid")) for item in state["results"].values()
            )
            state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(state_path, state)
    state["status"] = (
        "complete"
        if state.get("completed_valid") == len(datasets)
        and state.get("failed_or_invalid") == 0
        else "incomplete"
    )
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "completed_valid": state.get("completed_valid", 0),
                "failed_or_invalid": state.get("failed_or_invalid", 0),
                "expected_datasets": len(datasets),
            },
            sort_keys=True,
        )
    )
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
