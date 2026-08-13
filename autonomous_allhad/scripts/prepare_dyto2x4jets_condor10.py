#!/usr/bin/env python3
"""Prepare and optionally submit the full 2024 DYto2X-4Jets production."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Any


DATASETS = {
    "dyto2e": {
        "das": (
            "/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/"
            "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/"
            "NANOAODSIM"
        ),
    },
    "dyto2mu": {
        "das": (
            "/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/"
            "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/"
            "NANOAODSIM"
        ),
    },
    "dyto2tau": {
        "das": (
            "/DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/"
            "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v5/"
            "NANOAODSIM"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dataset_name(das: str) -> str:
    primary, processing, _tier = das.strip("/").split("/")
    return f"{primary}-{processing}"


def das_files(dasgoclient: Path, das: str) -> list[str]:
    result = subprocess.run(
        [str(dasgoclient), "--query", f"file dataset={das}"],
        check=True,
        text=True,
        capture_output=True,
    )
    files = sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().endswith(".root")
        }
    )
    if not files:
        raise RuntimeError(f"DAS returned no ROOT files for {das}")
    return files


def add_python_tree(bundle: tarfile.TarFile, source: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        bundle.add(
            path,
            arcname=str(Path("autonomous_allhad") / path.relative_to(source)),
            recursive=False,
        )
        count += 1
    return count


VALIDATOR = r'''#!/usr/bin/env python3
import json
import pathlib
import sys

import uproot

metadata_path = pathlib.Path(sys.argv[1])
root_path = pathlib.Path(sys.argv[2])
metadata = json.loads(metadata_path.read_text())
errors = []
if metadata.get("status") != "complete":
    errors.append(f"status={metadata.get('status')}")
attempted = int(metadata.get("files_attempted") or 0)
processed = int(metadata.get("files_processed") or 0)
if attempted <= 0 or attempted != processed:
    errors.append(f"files={processed}/{attempted}")
if metadata.get("bad_files"):
    errors.append("bad_files_nonempty")
for record in metadata.get("files") or []:
    if record.get("read_status") != "success":
        errors.append("file_read_not_success")
    if record.get("processing_status") != "processed_full_file":
        errors.append("file_not_processed_full")
with uproot.open(root_path) as root_file:
    events = root_file["Events"]
    entries = int(events.num_entries)
    branches = set(events.keys())
    if entries != int(metadata.get("events_written") or -1):
        errors.append("entries_events_written_mismatch")
    required = {
        "dataset_id", "gen_weight", "met", "ht", "nb_medium",
        "lowdm_search_bin", "lowdm_search_bin_LLCR",
        "lowdm_search_bin_QCDCR", "lowdm_search_bin_GCR",
        "lowdm_search_bin_DY2E", "lowdm_search_bin_DY2M",
        "lowdm_search_bin_SR",
    }
    missing = sorted(required - branches)
    if missing:
        errors.append("missing_branches:" + ",".join(missing))
dataset_records = list((metadata.get("datasets") or {}).values())
if len(dataset_records) != 1:
    errors.append("dataset_record_count")
else:
    source_counts = dataset_records[0].get("sumw_source_counts") or {}
    if int(source_counts.get("Runs.genEventSumw") or 0) != attempted:
        errors.append("runs_sumw_coverage")
if errors:
    raise SystemExit(";".join(sorted(set(errors))))
print(json.dumps({
    "status": "valid",
    "entries": entries,
    "files": attempted,
}))
'''


WRAPPER = r'''#!/bin/bash
set -euo pipefail
NAME="${1:?missing name}"
SHARD="${2:?missing shard}"
DEST="${3:?missing EOS destination}"
WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$WORKDIR"
case "$DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS destination: $DEST" >&2; exit 64 ;;
esac
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig runtime_xrd fragments
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export MPLCONFIGDIR="$WORKDIR/runtime_mplconfig"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$WORKDIR/runtime_xrd"
export AUTONOMOUS_ALLHAD_FRAGMENT_DIR="$WORKDIR/fragments"
export AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA=0
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
tar -xzf dyto2x4jets_worker.tgz
tar -xzf objectcorr_2024_payloads.tgz
tar -xzf "$4"
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR"
"$PY" -u -m autonomous_allhad.intermediate_2024_worker \
  --repo "$WORKDIR" \
  --shard "$WORKDIR/shards/$SHARD" \
  --output "$WORKDIR/out.root" \
  --metadata-output "$WORKDIR/out.json" \
  --chunk-size 50000 \
  --shift nominal \
  --skim-flag feature_flat_preselection \
  --record-workers 4
test -s out.root
test -s out.json
"$PY" validate_dyto2x4jets.py out.json out.root
export DEST
"$PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); d["root_file"]=os.environ["DEST"]; p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")'
ROOT_URL="root://eosuser.cern.ch/$DEST"
META_DEST="${DEST%.root}.json"
META_URL="root://eosuser.cern.ch/$META_DEST"
staged=0
for attempt in 1 2 3 4 5; do
  if xrdcp -f --nopbar --streams 4 out.root "$ROOT_URL" &&
     xrdcp -f --nopbar out.json "$META_URL" &&
     xrdfs eosuser.cern.ch stat "$DEST" >/dev/null &&
     xrdfs eosuser.cern.ch stat "$META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "$((attempt * 10))"
done
test "$staged" -eq 1
echo "completed $NAME"
'''


def make_submit(
    *,
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py38: Path,
    worker: Path,
    payloads: Path,
    shard_bundle: Path,
    proxy: Path,
    validator: Path,
    fingerprint: str,
    process: str,
) -> str:
    return f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shard_name) $(root_out) {shard_bundle.name}
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker}, {payloads}, {shard_bundle}, {proxy}, {validator}
transfer_output_files = ""
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 14000MB
+JobFlavour = "workday"
+JobBatchName = "2024-dyto2x4jets-{process}-10files"
+CampaignFingerprint = "{fingerprint}"
+DYProcess = "{process}"
queue name,shard_name,root_out from {arguments}
"""


def submit(submit_file: Path, dryrun_file: Path) -> dict[str, Any]:
    dry = subprocess.run(
        [
            "condor_submit",
            "-name",
            "bigbird24",
            "-dry-run",
            str(dryrun_file),
            str(submit_file),
        ],
        text=True,
        capture_output=True,
    )
    if dry.returncode:
        raise RuntimeError(dry.stderr or dry.stdout)
    result = subprocess.run(
        ["condor_submit", "-name", "bigbird24", "-terse", str(submit_file)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    match = re.search(r"(\d+)\.", result.stdout + "\n" + result.stderr)
    if not match:
        raise RuntimeError(f"cannot parse cluster: {result.stdout} {result.stderr}")
    return {
        "cluster_id": int(match.group(1)),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--files-per-job", type=int, default=10)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    if args.files_per_job != 10:
        raise RuntimeError("this campaign is frozen to exactly 10 files per job")
    if not str(repo).startswith("/eos/user/t/taiwoo/"):
        raise RuntimeError("repo must be on the authorized EOS path")
    workflow_root = repo / "autonomous_allhad/workflow"
    if not str(campaign).startswith(str(workflow_root)):
        raise RuntimeError("campaign must be inside the workflow directory")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / "autonomous_allhad/x509up_u147757"
    payloads = (
        workflow_root
        / "intermediate_2024_fullselection_v3_lowdm_relaxed_20260724"
        / "bundles/objectcorr_2024_payloads.tgz"
    )
    runtime = (
        workflow_root
        / "dyto2x4jets_subset_validation_20260729"
        / "runtime/autonomous_allhad"
    )
    dasgoclient = Path("/cvmfs/cms.cern.ch/common/dasgoclient")
    for path in (py38, proxy, payloads, dasgoclient):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required input missing: {path}")
    if not runtime.is_dir():
        raise RuntimeError(f"validated runtime snapshot missing: {runtime}")

    if args.submit and (campaign / "submission_receipts.json").exists():
        raise RuntimeError("this exact campaign already has submission receipts")

    for directory in ("bundles", "condor", "logs", "shards", "outputs"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)

    worker = campaign / "bundles/dyto2x4jets_worker.tgz"
    with tarfile.open(worker.with_suffix(".tgz.tmp"), "w:gz") as bundle:
        worker_file_count = add_python_tree(bundle, runtime)
    os.replace(worker.with_suffix(".tgz.tmp"), worker)

    validator = campaign / "condor/validate_dyto2x4jets.py"
    validator.write_text(VALIDATOR)
    validator.chmod(0o755)
    wrapper = campaign / "condor/run_dyto2x4jets.sh"
    wrapper.write_text(WRAPPER)
    wrapper.chmod(0o755)

    processes: dict[str, Any] = {}
    for process, specification in DATASETS.items():
        das = specification["das"]
        name = dataset_name(das)
        dataset_id = stable_id(name)
        files = das_files(dasgoclient, das)
        process_shards = campaign / "shards" / process
        process_shards.mkdir(parents=True, exist_ok=True)
        shard_names = []
        for shard_index, start in enumerate(range(0, len(files), 10)):
            selected = files[start : start + 10]
            shard_name = f"{process}_4jets_{shard_index:04d}"
            records = [
                {
                    "dataset": name,
                    "file_index": start + offset,
                    "file_path": f"root://cms-xrd-global.cern.ch/{file_path}",
                    "is_background": True,
                    "is_data": False,
                    "is_signal": False,
                    "process_group": "DY",
                    "sample_name": name,
                    "sumw_source": (
                        "Runs.genEventSumw preferred; Events.genWeight fallback"
                    ),
                    "xsec_pb": 1.0,
                    "year": "2024",
                }
                for offset, file_path in enumerate(selected)
            ]
            write_json(
                process_shards / f"{shard_name}.json",
                {
                    "record_digest": hashlib.sha256(
                        json.dumps(records, sort_keys=True).encode()
                    ).hexdigest()[:16],
                    "record_group": "mc",
                    "records": records,
                    "records_per_shard": len(records),
                    "schema_version": (
                        "full_production_shard_spec_v5_"
                        "fullselection_2024_dyto2x4jets_condor10"
                    ),
                    "shard_id": shard_name,
                },
            )
            shard_names.append(shard_name)

        shard_bundle = campaign / f"bundles/{process}_shards.tgz"
        with tarfile.open(shard_bundle.with_suffix(".tgz.tmp"), "w:gz") as bundle:
            for shard_name in shard_names:
                source = process_shards / f"{shard_name}.json"
                bundle.add(source, arcname=f"shards/{source.name}", recursive=False)
        os.replace(shard_bundle.with_suffix(".tgz.tmp"), shard_bundle)

        output = campaign / "outputs" / process / "nominal"
        output.mkdir(parents=True, exist_ok=True)
        logs = campaign / "logs" / process
        logs.mkdir(parents=True, exist_ok=True)
        arguments = campaign / f"condor/{process}_arguments.txt"
        arguments.write_text(
            "\n".join(
                f"{shard_name} {shard_name}.json {output / f'{shard_name}.root'}"
                for shard_name in shard_names
            )
            + "\n"
        )
        processes[process] = {
            "das": das,
            "dataset": name,
            "dataset_id": dataset_id,
            "file_count": len(files),
            "job_count": len(shard_names),
            "last_job_files": len(files) % 10 or 10,
            "shard_bundle": str(shard_bundle),
            "shard_bundle_sha256": sha256(shard_bundle),
            "arguments": str(arguments),
            "output": str(output),
        }

    fingerprint_components = {
        "worker": sha256(worker),
        "payloads": sha256(payloads),
        "py38": sha256(py38),
        "process_shards": {
            process: record["shard_bundle_sha256"]
            for process, record in processes.items()
        },
        "files_per_job": 10,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_components, sort_keys=True).encode()
    ).hexdigest()

    for process, record in processes.items():
        submit_file = campaign / f"condor/submit_{process}.sub"
        submit_file.write_text(
            make_submit(
                wrapper=wrapper,
                arguments=Path(record["arguments"]),
                logs=campaign / "logs" / process,
                py38=py38,
                worker=worker,
                payloads=payloads,
                shard_bundle=Path(record["shard_bundle"]),
                proxy=proxy,
                validator=validator,
                fingerprint=fingerprint,
                process=process,
            )
        )
        record["submit_file"] = str(submit_file)

    manifest = {
        "schema": "dyto2x4jets_condor10_campaign_v1",
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "files_per_job": 10,
        "processes": processes,
        "totals": {
            "files": sum(item["file_count"] for item in processes.values()),
            "jobs": sum(item["job_count"] for item in processes.values()),
        },
        "normalization": {
            "status": "pending_authoritative_xsdb_cross_sections",
            "intermediate_root_weight": "raw gen_weight",
            "shard_xsec_pb": (
                "1.0 metadata placeholder only; forbidden for physical yields"
            ),
            "rerun_needed_after_xsec": False,
        },
        "validated_runtime_source": str(runtime),
        "worker_bundle": {
            "path": str(worker),
            "sha256": sha256(worker),
            "python_files": worker_file_count,
        },
        "payload_bundle": str(payloads),
        "python_bundle": str(py38),
        "proxy": str(proxy),
        "subset_policy": (
            "The prior 12-file diagnostic subset is preserved but excluded "
            "from final coverage; this campaign independently covers every "
            "DAS file exactly once."
        ),
    }
    write_json(campaign / "manifest.json", manifest)
    if not args.submit:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    receipts = {}
    for process, record in processes.items():
        receipts[process] = submit(
            Path(record["submit_file"]),
            campaign / f"condor/{process}.dryrun",
        )
    receipt = {
        "schema": "dyto2x4jets_condor10_submission_receipts_v1",
        "status": "submitted",
        "fingerprint": fingerprint,
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "receipts": receipts,
    }
    write_json(campaign / "submission_receipts.json", receipt)
    manifest["status"] = "submitted"
    manifest["submitted_at"] = receipt["submitted_at"]
    manifest["cluster_ids"] = {
        process: record["cluster_id"] for process, record in receipts.items()
    }
    manifest["submission_receipts"] = str(
        campaign / "submission_receipts.json"
    )
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
