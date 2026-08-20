#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare validation-first recovery for unmarked TROTA trees."
    )
    parser.add_argument("--scan-json", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--base-script", required=True, type=Path)
    parser.add_argument("--inplace-script", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-materialize", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = json.loads(args.scan_json.read_text())
    partial = sorted(
        (row for row in rows if row.get("state") == "partial"),
        key=lambda row: int(row["proc"]),
    )
    partial = partial[args.skip :]
    if args.limit is not None:
        partial = partial[: args.limit]
    if not partial:
        raise RuntimeError("scan contains no partial TROTA files")
    for required in (args.base_script, args.inplace_script, args.model):
        if not required.is_file():
            raise FileNotFoundError(required)

    campaign = args.campaign_dir.absolute()
    condor_dir = campaign / "condor"
    logs_dir = campaign / "logs"
    for directory in (campaign, condor_dir, logs_dir, args.metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    arguments_path = condor_dir / "arguments.txt"
    atomic_write(
        arguments_path,
        "".join(
            f"{row['name']} {row['path']} {args.metadata_dir / (row['name'] + '.json')}\n"
            for row in partial
        ),
    )

    wrapper = condor_dir / "run_trota_resolved_2024_recovery.sh"
    atomic_write(
        wrapper,
        """#!/usr/bin/env bash
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
    --chunk-events 20000 \
    --batch-size 8192 \
    --recover-partial || worker_status=$?

if [ -s "$metadata_name" ]; then
    metadata_url="root://eosuser.cern.ch/${metadata_dest}"
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
""",
    )
    wrapper.chmod(0o755)

    transfer_inputs = ", ".join(
        str(path.absolute())
        for path in (args.base_script, args.inplace_script, args.model)
    )
    submit_path = condor_dir / "trota_topresolved_2024_recovery.sub"
    atomic_write(
        submit_path,
        f"""universe = vanilla
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
request_memory = 4000MB
request_disk = 2000MB
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour = "tomorrow"
max_materialize = {args.max_materialize}
max_idle = {args.max_materialize}
queue name,input_root,metadata_dest from {arguments_path}
""",
    )
    atomic_write(
        campaign / "recovery_manifest.json",
        json.dumps(
            {
                "schema_version": "trota_topresolved_2024_inplace_recovery_v1",
                "partial_files": len(partial),
                "source_scan": str(args.scan_json.absolute()),
                "policy": (
                    "fresh inference must match every persisted candidate identity and "
                    "float32 value before only the completion marker is appended"
                ),
                "inputs": partial,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(submit_path)
    print(f"partial files: {len(partial)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
