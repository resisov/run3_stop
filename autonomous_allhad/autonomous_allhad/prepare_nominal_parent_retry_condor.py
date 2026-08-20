from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

import uproot


WORKER_FILES = (
    "__init__.py",
    "flat_ntuple_worker.py",
    "full_production_worker.py",
    "intermediate_2024_worker.py",
    "object_corrections_2024.py",
    "real_subset_worker.py",
    "toptag_eff_worker.py",
)
VALID_STATUSES = {"complete", "complete_with_bad_files"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(text)
    os.replace(temporary, path)
    if executable:
        path.chmod(0o755)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_argument(line: str) -> dict[str, str]:
    fields = line.split()
    if len(fields) != 5:
        raise RuntimeError(f"unexpected argument line with {len(fields)} fields: {line!r}")
    name, shift, shard, shard_name, root_out = fields
    return {
        "name": name,
        "shift": shift,
        "shard": shard,
        "shard_name": shard_name,
        "root_out": root_out,
        "meta_out": str(Path(root_out).with_suffix(".json")),
    }


def validate_output(job: dict[str, str]) -> tuple[bool, str]:
    root_path = Path(job["root_out"])
    metadata_path = Path(job["meta_out"])
    shard_path = Path(job["shard"])
    if not root_path.is_file() or root_path.stat().st_size <= 0:
        return False, "missing_or_empty_root"
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        return False, "missing_or_empty_metadata"
    try:
        shard = json.loads(shard_path.read_text())
        expected_files = len(shard.get("records") or [])
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") not in VALID_STATUSES:
            return False, f"invalid_status:{metadata.get('status')}"
        if metadata.get("shard_id") != job["name"]:
            return False, "shard_id_mismatch"
        if metadata.get("shape_shift") != job["shift"]:
            return False, "shift_mismatch"
        if int(metadata.get("files_attempted", -1)) != expected_files:
            return False, "files_attempted_mismatch"
        if int(metadata.get("files_processed", -1)) != expected_files:
            return False, "files_processed_mismatch"
        if metadata.get("bad_files"):
            return False, "nonempty_bad_files"
        if int(metadata.get("events_written", -1)) < 0:
            return False, "invalid_events_written"
    except Exception as exc:
        return False, f"validation_failed:{type(exc).__name__}"
    return True, "valid"


def build_worker_bundle(package: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".new")
    file_hashes: dict[str, str] = {}
    with tarfile.open(temporary, "w:gz") as archive:
        for name in WORKER_FILES:
            path = package / name
            if not path.is_file():
                raise FileNotFoundError(path)
            file_hashes[name] = sha256(path)
            archive.add(path, arcname=f"autonomous_allhad/{name}", recursive=False)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "size": destination.stat().st_size,
        "source_sha256": file_hashes,
    }


def wrapper_text(proxy_name: str, worker_sha256: str, retry_tag: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
NAME="${{1:?missing name}}"
SHIFT="${{2:?missing shift}}"
SHARD="${{3:?missing shard basename}}"
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
export X509_USER_PROXY="$WORKDIR/{proxy_name}"
chmod 600 "$X509_USER_PROXY"
tar -xzf py38.tgz
tar -xzf objectcorr_2024_worker_optimized_parent_retry.tgz
tar -xzf objectcorr_2024_payloads.tgz
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORKDIR"
"$PY" -u -m autonomous_allhad.intermediate_2024_worker \
  --repo "$WORKDIR" --shard "$WORKDIR/$SHARD" \
  --output "$WORKDIR/out.root" --metadata-output "$WORKDIR/out.json" \
  --shift "$SHIFT" --record-workers 4
test -s out.root
test -s out.json
ROOT_URL="root://eosuser.cern.ch/$DEST"
META_DEST="${{DEST%.root}}.json"
META_URL="root://eosuser.cern.ch/$META_DEST"
XRDCOPY="$(command -v xrdcp)"
XRDFS="$(command -v xrdfs)"
export DEST NAME SHARD
"$PY" - <<'PY'
import json
import os
import pathlib
import uproot

metadata_path = pathlib.Path("out.json")
metadata = json.loads(metadata_path.read_text())
shard = json.loads(pathlib.Path(os.environ["SHARD"]).read_text())
expected_files = len(shard.get("records") or [])
if metadata.get("status") not in {{"complete", "complete_with_bad_files"}}:
    raise SystemExit(f"invalid worker status: {{metadata.get('status')}}")
if metadata.get("shard_id") != os.environ["NAME"]:
    raise SystemExit(f"shard mismatch: {{metadata.get('shard_id')}}")
if int(metadata.get("files_attempted", -1)) != expected_files:
    raise SystemExit("files_attempted mismatch")
if int(metadata.get("files_processed", -1)) != expected_files:
    raise SystemExit("files_processed mismatch")
if metadata.get("bad_files"):
    raise SystemExit("bad_files is not empty")
with uproot.open("out.root") as root_file:
    entries = int(root_file["Events"].num_entries)
if entries != int(metadata.get("events_written", -1)):
    raise SystemExit("ROOT/metadata event count mismatch")
metadata["root_file"] = os.environ["DEST"]
metadata["retry_campaign"] = "{retry_tag}"
metadata["retry_source_cluster"] = 957385
metadata["materialization_policy"] = "feature_flat_preselection_before_python_rows"
metadata["worker_bundle_sha256"] = "{worker_sha256}"
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n")
PY
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
echo "completed optimized parent retry $NAME $SHIFT"
'''


def submit_text(
    wrapper: Path,
    arguments: Path,
    logs: Path,
    python_bundle: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    proxy: Path,
) -> str:
    return f'''universe = vanilla
initialdir = {wrapper.parent}
executable = {wrapper}
arguments = $(name) $(shift) $(shard_name) $(root_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {python_bundle}, {worker_bundle}, {payload_bundle}, {proxy}, $(shard)
transfer_output_files = ""
output = {logs}/$(name)_$(shift).out
error = {logs}/$(name)_$(shift).err
log = {logs}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 14000MB
+JobFlavour = "workday"
+CampaignName = "nominal_2024_parent_retry_optimized_v1"
+RetrySourceCluster = 957385
+MaterializationPolicy = "feature_flat_preselection_before_python_rows"
queue name,shift,shard,shard_name,root_out from {arguments}
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare optimized parent-shard retry for missing 2024 nominal outputs.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--retry-tag", default="retry_optimized_parent_20260722")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    campaign = args.campaign.resolve()
    retry = campaign / args.retry_tag
    if not str(repo).startswith("/eos/") or not str(campaign).startswith("/eos/"):
        raise RuntimeError("repo and campaign must be EOS paths")
    if retry.exists():
        raise FileExistsError(f"refusing to overwrite existing retry campaign: {retry}")

    original_arguments = campaign / "condor/arguments.txt"
    jobs = [parse_argument(line) for line in original_arguments.read_text().splitlines() if line.strip()]
    if len(jobs) != 4283:
        raise RuntimeError(f"expected 4283 nominal jobs, found {len(jobs)}")

    pending: list[dict[str, str]] = []
    valid_jobs: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for index, job in enumerate(jobs, start=1):
        valid, reason = validate_output(job)
        reasons[reason] += 1
        if not valid:
            item = dict(job)
            item["validation_reason"] = reason
            pending.append(item)
        else:
            valid_jobs.append(job)
        if index % 500 == 0 or index == len(jobs):
            print(f"validated {index}/{len(jobs)}; retry={len(pending)}", flush=True)

    if not pending:
        raise RuntimeError("no invalid outputs found; refusing to create empty retry")

    root_sample: list[dict[str, Any]] = []
    stride = max(1, len(valid_jobs) // 20)
    for job in valid_jobs[::stride][:20]:
        metadata = json.loads(Path(job["meta_out"]).read_text())
        with uproot.open(job["root_out"]) as root_file:
            if "Events" not in root_file:
                raise RuntimeError(f"sampled ROOT is missing Events: {job['root_out']}")
            entries = int(root_file["Events"].num_entries)
        if entries != int(metadata.get("events_written", -1)):
            raise RuntimeError(f"sampled ROOT/metadata count mismatch: {job['name']}")
        root_sample.append({"name": job["name"], "entries": entries})

    condor = retry / "condor"
    logs = retry / "logs"
    bundles = retry / "bundles"
    logs.mkdir(parents=True)
    arguments = condor / "arguments.txt"
    wrapper = condor / "run_intermediate_2024_retry.sh"
    submit = condor / "submit.sub"
    worker_bundle = bundles / "objectcorr_2024_worker_optimized_parent_retry.tgz"
    payload_bundle = campaign / "bundles/objectcorr_2024_payloads.tgz"
    python_bundle = repo / "condor/py38.tgz"
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    for required in (payload_bundle, python_bundle, proxy):
        if not required.is_file():
            raise FileNotFoundError(required)

    worker = build_worker_bundle(repo / "autonomous_allhad/autonomous_allhad", worker_bundle)
    lines = [
        " ".join((job["name"], job["shift"], job["shard"], job["shard_name"], job["root_out"]))
        for job in pending
    ]
    write_text(arguments, "\n".join(lines) + "\n")
    write_text(wrapper, wrapper_text(proxy.name, worker["sha256"], args.retry_tag), executable=True)
    write_text(submit, submit_text(wrapper, arguments, logs, python_bundle, worker_bundle, payload_bundle, proxy))

    by_kind = Counter("data" if job["name"].startswith("data_") else "background" for job in pending)
    manifest = {
        "schema_version": "nominal_2024_parent_retry_optimized_v1",
        "status": "prepared",
        "created_at": now(),
        "source_campaign": str(campaign),
        "source_cluster": 957385,
        "retry_campaign": str(retry),
        "selection_policy": "fresh validation of all 4283 nominal ROOT/JSON pairs",
        "validation": {
            "total": len(jobs),
            "valid": len(jobs) - len(pending),
            "retry": len(pending),
            "reasons": dict(sorted(reasons.items())),
            "by_kind": dict(sorted(by_kind.items())),
            "root_content_sample": root_sample,
        },
        "runtime_policy": {
            "schedd_pool": "eossubmit",
            "schedd": "bigbird24.cern.ch",
            "record_workers": 4,
            "request_cpus": 4,
            "request_memory_mb": 12000,
            "request_disk_mb": 14000,
            "job_flavour": "workday",
            "materialization": "feature_flat_preselection_before_python_rows",
            "worker_execution": "transferred bundle in Condor scratch",
            "persistent_path_policy": "EOS only; no AFS",
        },
        "worker_bundle": worker,
        "payload_bundle": {"path": str(payload_bundle), "sha256": sha256(payload_bundle)},
        "python_bundle": {"path": str(python_bundle), "sha256": sha256(python_bundle)},
        "proxy": str(proxy),
        "files": {
            "arguments": str(arguments),
            "wrapper": str(wrapper),
            "submit": str(submit),
        },
        "retry_jobs": pending,
        "submission_command": f"source /etc/profile.d/modules.sh && module load lxbatch/eossubmit && condor_submit {submit}",
    }
    write_json(retry / "manifest.json", manifest)

    for path in (arguments, wrapper, submit, retry / "manifest.json"):
        if "/afs" in path.read_text().lower():
            raise RuntimeError(f"AFS reference found in {path}")
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True), flush=True)
    print(f"worker_bundle_sha256={worker['sha256']}", flush=True)
    print(f"prepared={retry}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
