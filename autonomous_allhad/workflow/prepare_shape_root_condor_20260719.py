#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SHIFTS = (
    "jesTotalUp",
    "jesTotalDown",
    "metUnclusteredUp",
    "metUnclusteredDown",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]


def records_from_sidecar(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"non-complete source sidecar: {path}: {payload.get('status')}")
    if payload.get("bad_files"):
        raise RuntimeError(f"source sidecar contains bad files: {path}")

    xsecs = {
        str(item.get("dataset")): item.get("xsec_pb")
        for item in (payload.get("datasets") or {}).values()
        if isinstance(item, dict)
    }
    records: list[dict[str, Any]] = []
    files = payload.get("files") or []
    if len(files) != int(payload.get("files_attempted") or -1):
        raise RuntimeError(f"file accounting mismatch in {path}")
    for index, item in enumerate(files):
        if item.get("read_status") != "success":
            raise RuntimeError(f"non-success file record in {path}: {item.get('file_path')}")
        dataset = str(item.get("dataset") or "")
        process = str(item.get("process") or "")
        file_path = str(item.get("file_path") or "")
        if not dataset or not process or not file_path:
            raise RuntimeError(f"incomplete file record in {path}: index {index}")
        is_signal = process == "SMS"
        record = {
            "dataset": dataset,
            "process_group": process,
            "file_path": file_path,
            "file_index": index,
            "year": "2024",
            "is_data": False,
            "is_signal": is_signal,
            "is_background": not is_signal,
            "xsec_pb": xsecs.get(dataset),
        }
        if is_signal and item.get("fastsim_trigger_bypass") is True:
            record["simulation_type"] = "FastSim signal dataset"
        records.append(record)
    return records


def wrapper_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

NAME="${1:?missing job name}"
SHIFT="${2:?missing shift}"
SHARD_NAME="${3:?missing shard name}"
ROOT_DEST="${4:?missing ROOT destination}"
WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$WORKDIR"

case "$ROOT_DEST" in
  /eos/user/t/taiwoo/*) ;;
  *)
    echo "refusing non-EOS-user destination: $ROOT_DEST" >&2
    exit 64
    ;;
esac

mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig fragments
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export MPLCONFIGDIR="$WORKDIR/runtime_mplconfig"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export AUTONOMOUS_ALLHAD_ANALYSIS_CACHE_DIR="$WORKDIR/runtime_cache/analysis"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$WORKDIR/runtime_cache/xrd"
export AUTONOMOUS_ALLHAD_FRAGMENT_DIR="$WORKDIR/fragments"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4
export XRD_REQUESTTIMEOUT=180
export XRD_REDIRECTLIMIT=10
export X509_USER_PROXY="$WORKDIR/x509up_u147757"
chmod 600 "$X509_USER_PROXY"

tar -xzf py38.tgz
tar -xzf analysis.tgz
tar -xzf flat_worker_20260719.tgz

PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
test -x "$PY"
test -d "$WORKDIR/analysis"
test -f "$WORKDIR/autonomous_allhad/autonomous_allhad/flat_ntuple_worker.py"
export PATH="$WORKDIR/bin:$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR/autonomous_allhad"

XRDCOPY="$WORKDIR/bin/xrdcp"
XRDFS="$WORKDIR/bin/xrdfs"
test -x "$XRDCOPY"
test -x "$XRDFS"

set +e
"$PY" -u -m autonomous_allhad.flat_ntuple_worker \
  --repo "$WORKDIR" \
  --shard "$WORKDIR/$SHARD_NAME" \
  --output "$WORKDIR/out.root" \
  --metadata-output "$WORKDIR/out.json" \
  --shift "$SHIFT" \
  --record-workers 4
WORKER_STATUS=$?
set -e

export ROOT_DEST WORKER_STATUS
if [ "$WORKER_STATUS" -ne 0 ]; then
  test -s out.json
  FAILED_META_DEST="${ROOT_DEST%.root}.failed.json"
  FAILED_ROOT_DEST="${ROOT_DEST%.root}.failed.root"
  export FAILED_META_DEST FAILED_ROOT_DEST
  "$PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); d["root_file"]=os.environ["FAILED_ROOT_DEST"] if pathlib.Path("out.root").is_file() else None; d["expected_root_file"]=os.environ["ROOT_DEST"]; d["condor_worker_exit_status"]=int(os.environ["WORKER_STATUS"]); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
  failed_staged=0
  for attempt in 1 2 3 4 5; do
    if "$XRDCOPY" -f --nopbar out.json "root://eosuser.cern.ch/${FAILED_META_DEST}" &&
       "$XRDFS" eosuser.cern.ch stat "$FAILED_META_DEST" >/dev/null; then
      if [ ! -s out.root ] || {
        "$XRDCOPY" -f --nopbar --streams 4 out.root "root://eosuser.cern.ch/${FAILED_ROOT_DEST}" &&
        "$XRDFS" eosuser.cern.ch stat "$FAILED_ROOT_DEST" >/dev/null;
      }; then
        failed_staged=1
        break
      fi
    fi
    echo "failed-output stage-out attempt $attempt failed for $ROOT_DEST" >&2
    sleep "$((attempt * 10))"
  done
  test "$failed_staged" -eq 1
  echo "worker failed for $NAME $SHIFT; diagnostic output staged to EOS" >&2
  exit "$WORKER_STATUS"
fi

"$PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); d["root_file"]=os.environ["ROOT_DEST"]; p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
test -s out.root
test -s out.json

META_DEST="${ROOT_DEST%.root}.json"
ROOT_URL="root://eosuser.cern.ch/${ROOT_DEST}"
META_URL="root://eosuser.cern.ch/${META_DEST}"

staged=0
for attempt in 1 2 3 4 5; do
  if "$XRDCOPY" -f --nopbar --streams 4 out.root "$ROOT_URL" &&
     "$XRDCOPY" -f --nopbar out.json "$META_URL" &&
     "$XRDFS" eosuser.cern.ch stat "$ROOT_DEST" >/dev/null &&
     "$XRDFS" eosuser.cern.ch stat "$META_DEST" >/dev/null; then
    staged=1
    break
  fi
  echo "stage-out attempt $attempt failed for $ROOT_DEST" >&2
  sleep "$((attempt * 10))"
done
test "$staged" -eq 1
echo "completed $NAME $SHIFT and staged to EOS"
"""


def submit_text(
    wrapper: Path,
    args_file: Path,
    log_dir: Path,
    py38: Path,
    analysis_bundle: Path,
    worker_bundle: Path,
    proxy: Path,
) -> str:
    return f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shift) $(shard_name) $(root_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {analysis_bundle}, {worker_bundle}, {proxy}, $(shard)
transfer_output_files = ""
output = {log_dir}/$(name).out
error = {log_dir}/$(name).err
log = {log_dir}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 12000MB
+JobFlavour = \"workday\"
queue name,shift,shard,shard_name,root_out,meta_out from {args_file}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare scratch-local HTCondor intermediate ROOT campaigns for JES Total and MET unclustered shifts.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-root-list", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--worker-bundle", required=True)
    parser.add_argument("--analysis-bundle")
    parser.add_argument(
        "--record-scope",
        default="all validated 2024 background MC NanoAOD records recovered from nominal flat ROOT sidecars",
    )
    parser.add_argument("--pilot-size", type=int, default=1)
    args = parser.parse_args()

    repo = Path(args.repo)
    source_list = Path(args.source_root_list)
    campaign = Path(args.campaign_dir)
    worker_bundle = Path(args.worker_bundle)
    py38 = repo / "condor/py38.tgz"
    analysis_bundle = Path(args.analysis_bundle) if args.analysis_bundle else repo / "condor/analysis.tgz"
    proxy = repo / f"analysis/proxy/x509up_u147757"
    required = [source_list, worker_bundle, py38, analysis_bundle, proxy]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing required input: " + ", ".join(missing))
    if args.pilot_size < 1:
        raise RuntimeError("pilot-size must be positive")

    roots = [Path(line.strip()) for line in source_list.read_text().splitlines() if line.strip()]
    if not roots:
        raise RuntimeError("source ROOT list is empty")
    campaign.mkdir(parents=True, exist_ok=True)
    shard_dir = campaign / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    wrapper = campaign / "run_shape_root_worker.sh"
    wrapper.write_text(wrapper_text())
    wrapper.chmod(0o755)

    source_manifest: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    total_records = 0
    for index, root in enumerate(roots):
        sidecar = root.with_suffix(".json")
        if not root.is_file() or not sidecar.is_file():
            raise RuntimeError(f"missing source ROOT or sidecar: {root}")
        records = records_from_sidecar(sidecar)
        duplicates = sorted({str(item["file_path"]) for item in records} & seen_files)
        if duplicates:
            raise RuntimeError(f"duplicate NanoAOD file across source sidecars: {duplicates[0]}")
        seen_files.update(str(item["file_path"]) for item in records)
        shard_name = f"shard_{index:05d}.json"
        shard_path = shard_dir / shard_name
        payload = {
            "schema_version": "full_production_shard_spec_v2_boosted",
            "shard_id": f"shard_{index:05d}",
            "record_digest": record_digest(records),
            "record_group": "mc",
            "records_per_shard": len(records),
            "records": records,
        }
        write_json(shard_path, payload)
        source_manifest.append(
            {
                "name": f"shard_{index:05d}",
                "shard": str(shard_path),
                "shard_name": shard_name,
                "source_root": str(root),
                "source_sidecar": str(sidecar),
                "records": len(records),
                "record_digest": payload["record_digest"],
            }
        )
        total_records += len(records)

    shift_manifests: dict[str, Any] = {}
    for shift in SHIFTS:
        shift_dir = campaign / shift
        output_dir = shift_dir / "outputs"
        log_dir = shift_dir / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in source_manifest:
            name = item["name"]
            rows.append(
                " ".join(
                    [
                        name,
                        shift,
                        item["shard"],
                        item["shard_name"],
                        str(output_dir / f"{name}.root"),
                        str(output_dir / f"{name}.json"),
                    ]
                )
            )
        all_args = shift_dir / "args_all.txt"
        pilot_args = shift_dir / "args_pilot.txt"
        remaining_args = shift_dir / "args_remaining.txt"
        all_args.write_text("\n".join(rows) + "\n")
        pilot_args.write_text("\n".join(rows[: args.pilot_size]) + "\n")
        remaining_args.write_text("\n".join(rows[args.pilot_size :]) + ("\n" if rows[args.pilot_size :] else ""))
        pilot_submit = shift_dir / "pilot.sub"
        remaining_submit = shift_dir / "remaining.sub"
        pilot_submit.write_text(submit_text(wrapper, pilot_args, log_dir, py38, analysis_bundle, worker_bundle, proxy))
        remaining_submit.write_text(submit_text(wrapper, remaining_args, log_dir, py38, analysis_bundle, worker_bundle, proxy))
        shift_manifests[shift] = {
            "jobs": len(rows),
            "pilot_jobs": min(args.pilot_size, len(rows)),
            "remaining_jobs": max(0, len(rows) - args.pilot_size),
            "output_dir": str(output_dir),
            "args_all": str(all_args),
            "pilot_submit": str(pilot_submit),
            "remaining_submit": str(remaining_submit),
        }

    manifest = {
        "schema_version": "shape_intermediate_root_condor_v1",
        "status": "prepared",
        "year": 2024,
        "record_scope": args.record_scope,
        "source_flat_roots": len(roots),
        "source_nanoaod_records": total_records,
        "unique_nanoaod_files": len(seen_files),
        "shifts": shift_manifests,
        "execution_policy": {
            "worker_code": "transferred flat_worker_20260719.tgz",
            "analysis_code_and_corrections": "transferred analysis.tgz",
            "python_environment": "transferred py38.tgz",
            "job_execution": "Condor scratch only",
            "eos_fuse_execution": False,
            "afs_execution": False,
            "temporary_storage": "Condor scratch runtime_tmp and fragments directories",
            "stageout": "worker xrdcp to root://eosuser.cern.ch//eos/user/t/taiwoo with xrdfs verification",
        },
        "inputs": {
            "source_root_list": str(source_list),
            "worker_bundle": str(worker_bundle),
            "analysis_bundle": str(analysis_bundle),
            "python_bundle": str(py38),
            "proxy": str(proxy),
        },
    }
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
