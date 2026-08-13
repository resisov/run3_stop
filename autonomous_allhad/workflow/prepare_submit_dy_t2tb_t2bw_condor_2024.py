#!/usr/bin/env python3
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def add_tree(bundle: tarfile.TarFile, source: Path, prefix: str) -> int:
    count = 0
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source)
        bundle.add(path, arcname=str(Path(prefix) / relative), recursive=False)
        count += 1
    return count


def make_worker_bundle(repo: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        files = add_tree(
            bundle,
            repo / "autonomous_allhad/autonomous_allhad",
            "autonomous_allhad",
        )
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "files": files,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def make_shard_bundle(source: Path, destination: Path) -> dict[str, Any]:
    shards = sorted(source.glob("*.json"))
    if not shards:
        raise RuntimeError(f"no JSON shards in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for shard in shards:
            bundle.add(shard, arcname=f"shards/{shard.name}", recursive=False)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "shards": len(shards),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "names": [shard.stem for shard in shards],
    }


VALIDATOR = r'''#!/usr/bin/env python3
import json
import pathlib
import sys

import numpy as np
import uproot

metadata_path = pathlib.Path(sys.argv[1])
root_path = pathlib.Path(sys.argv[2])
kind = sys.argv[3]
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
        "met", "ht", "nb_medium", "lowdm_search_bin",
        "lowdm_search_bin_LLCR", "lowdm_search_bin_QCDCR",
        "lowdm_search_bin_GCR", "lowdm_search_bin_DY2E",
        "lowdm_search_bin_DY2M", "lowdm_search_bin_SR",
    }
    missing = sorted(required - branches)
    if missing:
        errors.append("missing_branches:" + ",".join(missing))
    if kind == "signal":
        signal_required = {"signal_topology_id", "mStop", "mLSP"}
        missing_signal = sorted(signal_required - branches)
        if missing_signal:
            errors.append("missing_signal_branches:" + ",".join(missing_signal))
        else:
            topology_ids = set(
                np.asarray(
                    events["signal_topology_id"].array(library="np"),
                    dtype=np.int32,
                ).tolist()
            )
            if topology_ids not in ({2}, {3}):
                errors.append(f"wrong_topology_ids:{sorted(topology_ids)}")
        cutflow = [
            str(key)
            for key in root_file.keys(recursive=True, cycle=False)
            if str(key).startswith("signal_cutflow/")
        ]
        if not cutflow:
            errors.append("signal_cutflow_missing")
if kind == "signal":
    if any(
        not bool(record.get("fastsim_trigger_bypass"))
        for record in metadata.get("files") or []
    ):
        errors.append("fastsim_trigger_bypass_missing")
    dataset_records = list((metadata.get("datasets") or {}).values())
    if len(dataset_records) != 1:
        errors.append("signal_dataset_record_count")
    else:
        sumw = dataset_records[0].get("signal_sumw_by_genmodel") or {}
        if not sumw:
            errors.append("signal_runs_sumw_missing")
elif kind == "dy":
    stitch_records = [
        chunk.get("dy_ptll_gen_stitching")
        for record in metadata.get("files") or []
        for chunk in record.get("chunk_summaries") or []
    ]
    if not stitch_records or any(
        not isinstance(item, dict)
        or item.get("status") != "applied"
        or item.get("branch") != "LHE_Vpt"
        for item in stitch_records
    ):
        errors.append("dy_lhe_vpt_stitching_missing_or_incomplete")
if errors:
    raise SystemExit(";".join(sorted(set(errors))))
print(json.dumps({"status": "valid", "kind": kind, "entries": entries}))
'''


def wrapper_text(
    *,
    worker_bundle: str,
    payload_bundle: str,
    shard_bundle: str,
    proxy: str,
    validator: str,
) -> str:
    return f"""#!/bin/bash
set -euo pipefail
NAME="${{1:?missing name}}"
KIND="${{2:?missing kind}}"
SHARD="${{3:?missing shard}}"
DEST="${{4:?missing EOS destination}}"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
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
export X509_USER_PROXY="$WORKDIR/{proxy}"
chmod 600 "$X509_USER_PROXY"
tar -xzf py38.tgz
tar -xzf {worker_bundle}
tar -xzf {payload_bundle}
tar -xzf {shard_bundle}
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
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
"$PY" {validator} out.json out.root "$KIND"
export DEST
"$PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); d["root_file"]=os.environ["DEST"]; p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
ROOT_URL="root://eosuser.cern.ch/$DEST"
META_DEST="${{DEST%.root}}.json"
META_URL="root://eosuser.cern.ch/$META_DEST"
XRDCOPY="$(command -v xrdcp)"
XRDFS="$(command -v xrdfs)"
staged=0
for attempt in 1 2 3 4 5; do
  if "$XRDCOPY" -f --nopbar --streams 4 out.root "$ROOT_URL" &&
     "$XRDCOPY" -f --nopbar out.json "$META_URL" &&
     "$XRDFS" eosuser.cern.ch stat "$DEST" >/dev/null &&
     "$XRDFS" eosuser.cern.ch stat "$META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "$((attempt * 10))"
done
test "$staged" -eq 1
echo "completed $KIND $NAME"
"""


def submit_text(
    *,
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py38: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    shard_bundle: Path,
    proxy: Path,
    validator: Path,
    fingerprint: str,
    batch_name: str,
) -> str:
    return f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(kind) $(shard_name) $(root_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker_bundle}, {payload_bundle}, {shard_bundle}, {proxy}, {validator}
transfer_output_files = ""
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 14000MB
+JobFlavour = "workday"
+JobBatchName = "{batch_name}"
+CampaignFingerprint = "{fingerprint}"
queue name,kind,shard_name,root_out from {arguments}
"""


def submit_one(submit: Path, dryrun: Path) -> dict[str, Any]:
    dry = subprocess.run(
        ["condor_submit", "-name", "bigbird24", "-dry-run", str(dryrun), str(submit)],
        text=True,
        capture_output=True,
    )
    if dry.returncode:
        raise RuntimeError(f"dry-run failed: {dry.stderr or dry.stdout}")
    result = subprocess.run(
        ["condor_submit", "-name", "bigbird24", "-terse", str(submit)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"submit failed: {result.stderr or result.stdout}")
    match = re.search(r"(\d+)\.", result.stdout + "\n" + result.stderr)
    if match is None:
        raise RuntimeError(f"cannot parse cluster: {result.stdout} {result.stderr}")
    return {
        "cluster_id": int(match.group(1)),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "submit_file": str(submit),
        "dryrun_file": str(dryrun),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    if not str(repo).startswith("/eos/user/t/taiwoo/"):
        raise RuntimeError("repo must be under the authorized EOS path")
    if not str(campaign).startswith(str(repo / "autonomous_allhad/workflow/")):
        raise RuntimeError("campaign must be under autonomous_allhad/workflow")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / "autonomous_allhad/x509up_u147757"
    payload_bundle = (
        repo
        / "autonomous_allhad/workflow/intermediate_2024_fullselection_v3_lowdm_relaxed_20260724/bundles/objectcorr_2024_payloads.tgz"
    )
    dy_source = (
        repo
        / "autonomous_allhad/workflow/intermediate_2024_dy_exclusive_lhevpt_20260728/shards_exclusive"
    )
    signal_source = (
        repo
        / "autonomous_allhad/workflow/intermediate_2024_t2tb_t2bw_fastsim_local_20260728/shards"
    )
    for path in (py38, proxy, payload_bundle):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required input missing: {path}")

    for directory in ("bundles", "condor", "logs/dy", "logs/signals"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)
    worker = make_worker_bundle(
        repo,
        campaign / "bundles/objectcorr_2024_dy_t2tb_t2bw_worker.tgz",
    )
    dy = make_shard_bundle(
        dy_source,
        campaign / "bundles/dy_exclusive_shards.tgz",
    )
    signals = make_shard_bundle(
        signal_source,
        campaign / "bundles/t2tb_t2bw_shards.tgz",
    )
    validator = campaign / "condor/validate_intermediate.py"
    validator.write_text(VALIDATOR)
    validator.chmod(0o755)

    components = {
        "worker": worker["sha256"],
        "dy_shards": dy["sha256"],
        "signal_shards": signals["sha256"],
        "py38": sha256(py38),
        "payloads": sha256(payload_bundle),
    }
    fingerprint = hashlib.sha256(
        json.dumps(components, sort_keys=True).encode()
    ).hexdigest()
    jobs: dict[str, dict[str, Any]] = {}
    for kind, info in (("dy", dy), ("signal", signals)):
        output = campaign / "outputs_condor" / kind / "nominal"
        output.mkdir(parents=True, exist_ok=True)
        arguments = campaign / f"condor/{kind}_arguments.txt"
        arguments.write_text(
            "\n".join(
                f"{name} {kind} {name}.json {output / f'{name}.root'}"
                for name in info["names"]
            )
            + "\n"
        )
        shard_bundle = Path(info["path"])
        wrapper = campaign / f"condor/run_{kind}.sh"
        wrapper.write_text(
            wrapper_text(
                worker_bundle=Path(worker["path"]).name,
                payload_bundle=payload_bundle.name,
                shard_bundle=shard_bundle.name,
                proxy=proxy.name,
                validator=validator.name,
            )
        )
        wrapper.chmod(0o755)
        submit = campaign / f"condor/submit_{kind}.sub"
        submit.write_text(
            submit_text(
                wrapper=wrapper,
                arguments=arguments,
                logs=campaign / f"logs/{kind}",
                py38=py38,
                worker_bundle=Path(worker["path"]),
                payload_bundle=payload_bundle,
                shard_bundle=shard_bundle,
                proxy=proxy,
                validator=validator,
                fingerprint=fingerprint,
                batch_name=f"2024-{kind}-exclusive-newsignals",
            )
        )
        jobs[kind] = {
            "count": info["shards"],
            "arguments": str(arguments),
            "submit": str(submit),
            "output": str(output),
        }

    receipt_path = campaign / "submission_receipt.json"
    if args.submit and receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing.get("status") == "submitted":
            raise RuntimeError(
                f"equivalent campaign already submitted: {receipt_path}"
            )
    manifest = {
        "schema": "dy_t2tb_t2bw_condor_campaign_v1",
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "components": components,
        "jobs": jobs,
        "worker_bundle": worker,
        "dy_shard_bundle": {k: v for k, v in dy.items() if k != "names"},
        "signal_shard_bundle": {
            k: v for k, v in signals.items() if k != "names"
        },
        "proxy": str(proxy),
        "python_bundle": str(py38),
        "payload_bundle": str(payload_bundle),
        "output_collision_policy": (
            "Condor writes only outputs_condor; active local outputs remain untouched."
        ),
    }
    write_json(campaign / "manifest.json", manifest)
    if not args.submit:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    receipts = {
        kind: submit_one(
            Path(job["submit"]),
            campaign / f"condor/{kind}.dryrun",
        )
        for kind, job in jobs.items()
    }
    receipt = {
        "schema": "dy_t2tb_t2bw_condor_submission_receipt_v1",
        "status": "submitted",
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "receipts": receipts,
    }
    write_json(receipt_path, receipt)
    manifest["status"] = "submitted"
    manifest["submission_receipt"] = str(receipt_path)
    manifest["cluster_ids"] = {
        kind: item["cluster_id"] for kind, item in receipts.items()
    }
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
