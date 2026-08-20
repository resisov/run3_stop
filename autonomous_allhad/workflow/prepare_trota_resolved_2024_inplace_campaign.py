#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHARD_PATTERN = re.compile(r"^(data|mc|signal)_shard_(\d+)\.root$")


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
    parser = argparse.ArgumentParser(
        description="Prepare a sparse in-place 2024/2025 TROTA TopResolved campaign."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--base-script", required=True, type=Path)
    parser.add_argument("--inplace-script", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-materialize", type=int, default=100)
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--target-year", type=int, choices=(2024, 2025), default=2024)
    args = parser.parse_args()

    schema_version = f"trota_topresolved_{args.target_year}_inplace_campaign_v1"

    source_dir = args.source_dir.absolute()
    campaign = args.campaign_dir.absolute()
    metadata_dir = campaign / "metadata"
    logs_dir = campaign / "logs"
    condor_dir = campaign / "condor"
    for directory in (campaign, metadata_dir, logs_dir, condor_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inputs: list[dict[str, Any]] = []
    missing_source_metadata: list[str] = []
    for path in sorted(source_dir.glob("*_shard_*.root")):
        match = SHARD_PATTERN.match(path.name)
        if match is None:
            continue
        kind, index_text = match.groups()
        source_metadata = path.with_suffix(".json")
        if not source_metadata.is_file():
            missing_source_metadata.append(str(source_metadata))
        name = path.stem
        inputs.append(
            {
                "name": name,
                "kind": kind,
                "shard_index": int(index_text),
                "input_root": str(path),
                "input_bytes_before": path.stat().st_size,
                "source_metadata": str(source_metadata),
                "source_metadata_exists": source_metadata.is_file(),
                "job_metadata": str(metadata_dir / f"{name}.json"),
            }
        )
    inputs.sort(key=lambda item: ({"data": 0, "mc": 1, "signal": 2}[item["kind"]], item["shard_index"]))
    if not inputs:
        raise RuntimeError(
            f"no {args.target_year} intermediate ROOT files found under {source_dir}"
        )
    if missing_source_metadata:
        raise RuntimeError(
            f"{len(missing_source_metadata)} input ROOT files lack source metadata; "
            f"first={missing_source_metadata[0]}"
        )

    for required in (args.base_script, args.inplace_script, args.model):
        if not required.is_file():
            raise FileNotFoundError(required)

    input_digest = hashlib.sha256(
        "\n".join(f"{item['input_root']} {item['input_bytes_before']}" for item in inputs).encode()
    ).hexdigest()
    counts = {
        kind: sum(item["kind"] == kind for item in inputs)
        for kind in ("data", "mc", "signal")
    }
    total_input_bytes = sum(item["input_bytes_before"] for item in inputs)
    created_at = now()
    manifest = {
        "schema_version": schema_version,
        "application_year": args.target_year,
        "model_release_year": 2024,
        "created_at": created_at,
        "status": "prepared",
        "source_dir": str(source_dir),
        "campaign_dir": str(campaign),
        "input_files": len(inputs),
        "input_file_counts": counts,
        "input_bytes": total_input_bytes,
        "input_digest": input_digest,
        "storage_policy": (
            "append a flat TROTA tree containing only 1% WP candidates to each original ROOT"
        ),
        "original_events_policy": "the Events tree is never rewritten",
        "working_point": {
            "name": "1pct_qcd_mistag",
            "threshold": 0.9433798789978027,
            "discriminant": "TTScore / (TTScore + QCDScore)",
        },
        "float_policy": (
            "features, model outputs, derived discriminant, and stored candidate "
            "kinematics/scores are float32; p4 accumulation is float64"
        ),
        "inputs": inputs,
    }
    write_json(campaign / "input_manifest.json", manifest)
    write_json(campaign / "job_manifest.json", manifest)

    arguments = "\n".join(
        f"{item['name']} {item['input_root']} {item['job_metadata']}" for item in inputs
    ) + "\n"
    arguments_path = condor_dir / "arguments.txt"
    atomic_write(arguments_path, arguments)

    wrapper = condor_dir / f"run_trota_resolved_{args.target_year}_inplace.sh"
    wrapper_text = """#!/usr/bin/env bash
set -eo pipefail

name="$1"
input_root="$2"
metadata_dest="$3"
metadata_name="${name}.json"

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export NUMBA_NUM_THREADS=1

worker_status=0
python3 trota_resolved_2024_inplace.py \
    --input "$input_root" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$metadata_name" \
    --target-year __TARGET_YEAR__ \
    --chunk-events 20000 \
    --batch-size 8192 || worker_status=$?

metadata_url="root://eosuser.cern.ch/${metadata_dest}"
if [ -s "$metadata_name" ]; then
    staged=0
    for attempt in 1 2 3 4 5; do
        if xrdcp -f --nopbar "$metadata_name" "$metadata_url"; then
            staged=1
            break
        fi
        sleep "$((attempt * 10))"
    done
    if [ "$staged" -ne 1 ]; then
        exit 74
    fi
fi
exit "$worker_status"
""".replace("__TARGET_YEAR__", str(args.target_year))
    atomic_write(wrapper, wrapper_text)
    wrapper.chmod(0o755)

    transfer_inputs = ", ".join(
        str(path.absolute())
        for path in (args.base_script, args.inplace_script, args.model)
    )
    submit = f"""universe = vanilla
initialdir = {condor_dir}
executable = {wrapper}
arguments = $(name) $(input_root) $(metadata_dest)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {transfer_inputs}
transfer_output_files = ""
output = {logs_dir}/$(name).out
error = {logs_dir}/$(name).err
log = {logs_dir}/campaign.log
request_cpus = 2
request_memory = {args.request_memory_mb}MB
request_disk = 2000MB
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour = "tomorrow"
max_materialize = {args.max_materialize}
max_idle = {args.max_materialize}
queue name,input_root,metadata_dest from {arguments_path}
"""
    submit_path = condor_dir / f"trota_topresolved_{args.target_year}_inplace.sub"
    atomic_write(submit_path, submit)

    state = {
        "schema_version": schema_version,
        "application_year": args.target_year,
        "updated_at": created_at,
        "status": "prepared",
        "input_files": len(inputs),
        "complete": 0,
        "failed": 0,
        "pending": len(inputs),
        "cluster_id": None,
        "input_digest": input_digest,
    }
    write_json(campaign / "state.json", state)
    write_json(
        campaign / "file_validation_summary.json",
        {
            "schema_version": schema_version,
            "application_year": args.target_year,
            "updated_at": created_at,
            "status": "not_started",
            "input_files": len(inputs),
            "valid": 0,
            "invalid": 0,
            "pending": len(inputs),
        },
    )
    write_json(
        campaign / "bad_files.json",
        {"schema_version": schema_version, "application_year": args.target_year, "files": []},
    )
    atomic_write(campaign / "bad_files.txt", "")
    atomic_write(
        campaign / "history.jsonl",
        json.dumps(
            {
                "time": created_at,
                "event": "campaign_prepared",
                "input_files": len(inputs),
                "input_digest": input_digest,
            },
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write(
        campaign / "latest_summary.md",
        "\n".join(
            [
                f"# {args.target_year} TROTA TopResolved in-place campaign",
                "",
                "Status: prepared",
                "",
                f"- Input ROOT files: {len(inputs):,}",
                f"- Data / MC / signal: {counts['data']:,} / {counts['mc']:,} / {counts['signal']:,}",
                f"- Input size: {total_input_bytes / 1024**3:.3f} GiB",
                "- Output policy: sparse 1% WP candidates appended as `TROTA` tree",
                "- Original `Events` tree: preserved",
                "",
                f"Submit with: `condor_submit {submit_path}`",
                "",
            ]
        ),
    )
    print(json.dumps({key: manifest[key] for key in ("status", "input_files", "input_file_counts", "input_bytes", "input_digest")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
