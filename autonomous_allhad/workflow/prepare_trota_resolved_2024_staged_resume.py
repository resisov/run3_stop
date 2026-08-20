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
        description=(
            "Prepare local-stage, validate, EOS atomic-replacement TROTA production."
        )
    )
    parser.add_argument("--source-campaign", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--base-script", required=True, type=Path)
    parser.add_argument("--inplace-script", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--max-materialize",
        type=int,
        default=0,
        help="optional factory limit; zero omits materialization and idle caps",
    )
    parser.add_argument("--target-year", type=int, choices=(2024, 2025), default=2024)
    parser.add_argument("--request-memory-mb", type=int, default=4000)
    parser.add_argument("--request-disk-mb", type=int, default=6000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-name")
    args = parser.parse_args()

    manifest = json.loads((args.source_campaign / "input_manifest.json").read_text())
    pending = [
        item
        for item in manifest["inputs"]
        if not metadata_is_complete(Path(item["job_metadata"]))
    ]
    if args.only_name is not None:
        pending = [item for item in pending if item["name"] == args.only_name]
    if args.limit is not None:
        pending = pending[: args.limit]
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
    wrapper = condor_dir / f"run_trota_resolved_{args.target_year}_staged.sh"
    atomic_write(
        wrapper,
        """#!/usr/bin/env bash
set -o pipefail

name="$1"
input_root="$2"
metadata_dest="$3"
cluster_id="$4"
proc_id="$5"
metadata_name="${name}.json"
verify_name="${name}.verify.json"
local_root="${name}.root"
input_url="root://eosuser.cern.ch/${input_root}"
remote_temp="${input_root}.trota_tmp_${cluster_id}_${proc_id}"
remote_backup="${input_root}.trota_backup_${cluster_id}_${proc_id}"
temp_url="root://eosuser.cern.ch/${remote_temp}"

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export NUMBA_NUM_THREADS=1

stage_metadata() {
    local metadata_url="root://eosuser.cern.ch/${metadata_dest}"
    local staged=0
    if [ ! -s "$metadata_name" ]; then
        return 0
    fi
    for attempt in 1 2 3 4 5; do
        if xrdcp -f --nopbar "$metadata_name" "$metadata_url"; then
            staged=1
            break
        fi
        sleep "$((attempt * 10))"
    done
    [ "$staged" -eq 1 ]
}

if xrdfs eosuser.cern.ch stat "$remote_temp" >/dev/null 2>&1; then
    echo "Refusing pre-existing remote temp: $remote_temp" >&2
    exit 78
fi
if xrdfs eosuser.cern.ch stat "$remote_backup" >/dev/null 2>&1; then
    echo "Refusing pre-existing remote backup: $remote_backup" >&2
    exit 78
fi

if ! xrdcp -f --nopbar "$input_url" "$local_root"; then
    exit 74
fi

worker_status=0
python3 trota_resolved_2024_inplace.py \
    --input "$local_root" \
    --input-label "$input_root" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$metadata_name" \
    --target-year __TARGET_YEAR__ \
    --chunk-events 20000 \
    --batch-size 8192 \
    --recover-partial \
    --allow-hadd-repair || worker_status=$?
if [ "$worker_status" -ne 0 ]; then
    stage_metadata || exit 74
    exit "$worker_status"
fi

status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$metadata_name")"
if [ "$status" = "already_complete" ]; then
    stage_metadata || exit 74
    exit 0
fi
if [ "$status" != "complete" ] && [ "$status" != "recovered_complete" ]; then
    echo "Unexpected local processing status: $status" >&2
    exit 70
fi

python3 trota_resolved_2024_inplace.py \
    --input "$local_root" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$verify_name" \
    --target-year __TARGET_YEAR__ \
    --verify-only || exit 71

if ! xrdcp -f --nopbar "$local_root" "$temp_url"; then
    xrdfs eosuser.cern.ch rm "$remote_temp" >/dev/null 2>&1 || true
    exit 74
fi
if ! python3 trota_resolved_2024_inplace.py \
    --input "$remote_temp" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$verify_name" \
    --target-year __TARGET_YEAR__ \
    --verify-only; then
    xrdfs eosuser.cern.ch rm "$remote_temp" >/dev/null 2>&1 || true
    exit 71
fi

if ! xrdfs eosuser.cern.ch mv "$input_root" "$remote_backup"; then
    xrdfs eosuser.cern.ch rm "$remote_temp" >/dev/null 2>&1 || true
    exit 75
fi
if ! xrdfs eosuser.cern.ch mv "$remote_temp" "$input_root"; then
    xrdfs eosuser.cern.ch mv "$remote_backup" "$input_root" || true
    exit 75
fi
if ! python3 trota_resolved_2024_inplace.py \
    --input "$input_root" \
    --model model_TopResolved_2024_TROTA2D_ptcut.h5 \
    --metadata-output "$verify_name" \
    --target-year __TARGET_YEAR__ \
    --verify-only; then
    xrdfs eosuser.cern.ch mv "$input_root" "$remote_temp" || true
    xrdfs eosuser.cern.ch mv "$remote_backup" "$input_root" || true
    exit 71
fi
if ! xrdfs eosuser.cern.ch rm "$remote_backup"; then
    echo "Verified replacement is complete but backup cleanup failed: $remote_backup" >&2
    exit 77
fi

stage_metadata || exit 74
exit 0
""".replace("__TARGET_YEAR__", str(args.target_year)),
    )
    wrapper.chmod(0o755)

    transfer_inputs = ", ".join(
        str(path.absolute())
        for path in (args.base_script, args.inplace_script, args.model)
    )
    submit_path = condor_dir / f"trota_topresolved_{args.target_year}_staged.sub"
    materialize_lines = ""
    if args.max_materialize > 0:
        materialize_lines = (
            f"max_materialize = {args.max_materialize}\n"
            f"max_idle = {args.max_materialize}\n"
        )
    atomic_write(
        submit_path,
        f"""universe = vanilla
initialdir = {condor_dir}
executable = {wrapper}
arguments = $(name) $(input_root) $(metadata_dest) $(ClusterId) $(ProcId)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {transfer_inputs}
transfer_output_files = ""
output = {logs_dir}/$(name).out
error = {logs_dir}/$(name).err
log = {logs_dir}/campaign.log
request_cpus = 1
request_memory = {args.request_memory_mb}MB
request_disk = {args.request_disk_mb}MB
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour = "tomorrow"
{materialize_lines}
queue name,input_root,metadata_dest from {arguments_path}
""",
    )
    atomic_write(
        campaign / "staged_resume_manifest.json",
        json.dumps(
            {
                "schema_version": (
                    f"trota_topresolved_{args.target_year}_staged_resume_v1"
                ),
                "application_year": args.target_year,
                "model_release_year": 2024,
                "request_memory_mb": args.request_memory_mb,
                "request_disk_mb": args.request_disk_mb,
                "source_campaign": str(args.source_campaign.absolute()),
                "input_files": manifest["input_files"],
                "already_complete_metadata": manifest["input_files"] - len(pending),
                "pending_files": len(pending),
                "max_materialize": (
                    args.max_materialize if args.max_materialize > 0 else None
                ),
                "write_policy": (
                    "copy to worker local disk, update and deep-validate locally, upload to "
                    "a unique EOS temp, deep-validate temp, atomically rename original to "
                    "backup and verified temp to original, revalidate, then remove backup"
                ),
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
