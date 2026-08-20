#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


COMPLETE_STATUSES = {"complete", "already_complete", "recovered_complete"}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def metadata_is_complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("status") in COMPLETE_STATUSES
    except (OSError, ValueError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an idempotent bounded-concurrency TROTA resume campaign."
    )
    parser.add_argument("--source-campaign", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--base-script", required=True, type=Path)
    parser.add_argument("--inplace-script", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-materialize", type=int, default=50)
    parser.add_argument(
        "--exclude-partial-scan",
        action="append",
        default=[],
        type=Path,
        help="exclude state=partial inputs already assigned to validation-first recovery",
    )
    args = parser.parse_args()

    manifest = json.loads((args.source_campaign / "input_manifest.json").read_text())
    recovery_names: set[str] = set()
    for scan_path in args.exclude_partial_scan:
        recovery_names.update(
            str(row["name"])
            for row in json.loads(scan_path.read_text())
            if row.get("state") == "partial"
        )
    pending = [
        item
        for item in manifest["inputs"]
        if item["name"] not in recovery_names
        and not metadata_is_complete(Path(item["job_metadata"]))
    ]
    if not pending:
        raise RuntimeError("all manifest inputs already have complete metadata")
    for required in (args.base_script, args.inplace_script, args.model):
        if not required.is_file():
            raise FileNotFoundError(required)

    campaign = args.campaign_dir.absolute()
    condor_dir = campaign / "condor"
    logs_dir = campaign / "logs"
    for directory in (campaign, condor_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    arguments_path = condor_dir / "arguments.txt"
    atomic_write(
        arguments_path,
        "".join(
            f"{item['name']} {item['input_root']} {item['job_metadata']}\n"
            for item in pending
        ),
    )
    wrapper = condor_dir / "run_trota_resolved_2024_resume.sh"
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export NUMBA_NUM_THREADS=1

worker_status=0
python3 trota_resolved_2024_inplace.py \
    --input "$input_root" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$metadata_name" \
    --chunk-events 20000 \
    --batch-size 8192 || worker_status=$?

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
    submit_path = condor_dir / "trota_topresolved_2024_resume.sub"
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
request_cpus = 1
request_memory = 3000MB
request_disk = 2000MB
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour = "tomorrow"
max_materialize = {args.max_materialize}
max_idle = {args.max_materialize}
queue name,input_root,metadata_dest from {arguments_path}
""",
    )
    atomic_write(
        campaign / "resume_manifest.json",
        json.dumps(
            {
                "schema_version": "trota_topresolved_2024_inplace_resume_v1",
                "source_campaign": str(args.source_campaign.absolute()),
                "input_files": manifest["input_files"],
                "already_complete_metadata": manifest["input_files"] - len(pending),
                "pending_files": len(pending),
                "max_materialize": args.max_materialize,
                "excluded_recovery_files": len(recovery_names),
                "inputs": pending,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(submit_path)
    print(f"pending files: {len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
