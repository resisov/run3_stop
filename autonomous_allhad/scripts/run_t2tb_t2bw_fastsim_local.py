#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import uproot


TOPOLOGY_ID = {"T2tb": 2, "T2bW": 3}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate(shard: dict[str, Any]) -> dict[str, Any]:
    root_path = Path(shard["root"])
    metadata_path = Path(shard["json"])
    errors: list[str] = []
    if not root_path.exists() or root_path.stat().st_size <= 0:
        errors.append("missing_or_empty_root")
    if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
        errors.append("missing_or_empty_metadata")
    if errors:
        return {"valid": False, "errors": errors}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        return {"valid": False, "errors": [f"metadata:{type(exc).__name__}:{exc}"]}
    expected_files = int(shard["records"])
    if metadata.get("status") != "complete":
        errors.append(f"status={metadata.get('status')}")
    if int(metadata.get("files_attempted") or 0) != expected_files:
        errors.append("files_attempted_mismatch")
    if int(metadata.get("files_processed") or 0) != expected_files:
        errors.append("files_processed_mismatch")
    if metadata.get("bad_files"):
        errors.append("bad_files_nonempty")
    topology = str(shard["topology"])
    prefix = f"GenModel_{topology}_"
    sumw_prefix = f"Runs.genEventSumw_{topology}_"
    for file_record in metadata.get("files") or []:
        if not file_record.get("fastsim_trigger_bypass"):
            errors.append("fastsim_trigger_bypass_missing")
        if file_record.get("read_status") != "success":
            errors.append("file_read_not_success")
        if file_record.get("processing_status") != "processed_full_file":
            errors.append("file_not_processed_full")
    dataset_records = list((metadata.get("datasets") or {}).values())
    signal_sumw: dict[str, Any] = {}
    if len(dataset_records) != 1:
        errors.append("dataset_record_count")
    else:
        dataset_record = dataset_records[0]
        signal_sumw = dataset_record.get("signal_sumw_by_genmodel") or {}
        if not signal_sumw or any(
            not str(key).startswith(prefix) for key in signal_sumw
        ):
            errors.append("wrong_or_missing_runs_mass_sumw")
        source_counts = dataset_record.get("signal_runs_sumw_source_counts") or {}
        if not source_counts:
            errors.append("runs_sumw_source_count_missing")
    try:
        with uproot.open(root_path) as root_file:
            tree = root_file["Events"]
            entries = int(tree.num_entries)
            branches = set(tree.keys())
            if "signal_topology_id" not in branches:
                errors.append("signal_topology_id_branch_missing")
            else:
                values = np.asarray(
                    tree["signal_topology_id"].array(library="np"),
                    dtype=np.int32,
                )
                if len(values) and set(values.tolist()) != {TOPOLOGY_ID[topology]}:
                    errors.append("signal_topology_id_mismatch")
            if "mStop" not in branches or "mLSP" not in branches:
                errors.append("mass_branches_missing")
            else:
                mass_arrays = tree.arrays(["mStop", "mLSP"], library="np")
                selected_mass_keys = {
                    f"{prefix}{int(mstop)}_{int(mlsp)}"
                    for mstop, mlsp in zip(
                        mass_arrays["mStop"],
                        mass_arrays["mLSP"],
                    )
                }
                if not selected_mass_keys.issubset(set(signal_sumw)):
                    errors.append("selected_mass_point_missing_runs_sumw")
            if entries != int(metadata.get("events_written") or -1):
                errors.append("root_entries_events_written_mismatch")
            cutflow_keys = [
                str(key)
                for key in root_file.keys(recursive=True, cycle=False)
                if str(key).startswith("signal_cutflow/")
            ]
            if not cutflow_keys:
                errors.append("signal_cutflow_histograms_missing")
    except Exception as exc:
        errors.append(f"root:{type(exc).__name__}:{exc}")
        entries = -1
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "entries": entries,
        "events_read": int(metadata.get("events_read") or 0),
        "events_written": int(metadata.get("events_written") or 0),
        "files": expected_files,
        "sumw_source": sumw_prefix + "<mStop>_<mLSP>",
    }


def run_one(
    repo: Path,
    campaign: Path,
    python: Path,
    shard: dict[str, Any],
    record_workers: int,
) -> dict[str, Any]:
    name = shard["name"]
    output = Path(shard["root"])
    metadata = Path(shard["json"])
    log = campaign / "logs" / f"{name}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "autonomous_allhad.intermediate_2024_worker",
        "--repo",
        str(repo),
        "--shard",
        str(shard["shard"]),
        "--output",
        str(output),
        "--metadata-output",
        str(metadata),
        "--chunk-size",
        "50000",
        "--shift",
        "nominal",
        "--skim-flag",
        "feature_flat_preselection",
        "--record-workers",
        str(record_workers),
    ]
    started = time.time()
    environment = dict(os.environ)
    environment["X509_USER_PROXY"] = str(
        repo / "autonomous_allhad/x509up_u147757"
    )
    with log.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=repo / "autonomous_allhad",
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    validation = validate(shard)
    return {
        "name": name,
        "topology": shard["topology"],
        "dataset": shard["dataset"],
        "returncode": result.returncode,
        "wall_time_s": time.time() - started,
        "log": str(log),
        **validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--record-workers", type=int, default=4)
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--only-shard",
        action="append",
        default=[],
        help="Run only the named manifest shard; repeat for multiple shards.",
    )
    args = parser.parse_args()
    repo = Path(args.repo)
    campaign = Path(args.campaign)
    python = Path(args.python)
    manifest = json.loads((campaign / "manifest.json").read_text())
    shards = list(manifest["shards"])
    if args.only_shard:
        requested = set(args.only_shard)
        shards = [shard for shard in shards if shard["name"] in requested]
        found = {shard["name"] for shard in shards}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"unknown shard(s): {','.join(missing)}")
    state_path = Path(args.state) if args.state else campaign / "processing_state.json"
    results: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    for shard in shards:
        existing = validate(shard)
        if existing["valid"]:
            results[shard["name"]] = {"action": "reused", **existing}
        else:
            pending.append(shard)
    state = {
        "schema": "t2tb_t2bw_fastsim_local_state_v1",
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_shards": len(shards),
        "completed_valid": sum(bool(x.get("valid")) for x in results.values()),
        "failed_or_invalid": 0,
        "results": results,
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
                shard,
                max(1, args.record_workers),
            ): shard
            for shard in pending
        }
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "name": shard["name"],
                    "topology": shard["topology"],
                    "dataset": shard["dataset"],
                    "valid": False,
                    "errors": [f"runner:{type(exc).__name__}:{exc}"],
                }
            state["results"][shard["name"]] = result
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
        if state["completed_valid"] == len(shards)
        and state["failed_or_invalid"] == 0
        else "incomplete"
    )
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "completed_valid": state["completed_valid"],
                "failed_or_invalid": state["failed_or_invalid"],
                "expected_shards": len(shards),
            },
            sort_keys=True,
        )
    )
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
