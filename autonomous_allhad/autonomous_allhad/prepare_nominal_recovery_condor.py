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


VALID_STATUSES = {"complete", "complete_with_bad_files"}
WORKER_FILES = (
    "__init__.py",
    "cli.py",
    "flat_ntuple_worker.py",
    "full_production_worker.py",
    "intermediate_2024_worker.py",
    "object_corrections_2024.py",
    "pipeline.py",
    "real_subset_worker.py",
    "report_pages.py",
    "toptag_eff_worker.py",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.new")
    temporary.write_text(text)
    os.replace(temporary, path)
    if executable:
        path.chmod(0o755)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_output(task: dict[str, Any]) -> tuple[bool, str]:
    root_path = Path(task["root_output"])
    metadata_path = Path(task["metadata_output"])
    if not root_path.is_file() or root_path.stat().st_size <= 0:
        return False, "missing_or_empty_root"
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        return False, "missing_or_empty_metadata"
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") not in VALID_STATUSES:
            return False, f"invalid_status:{metadata.get('status')}"
        if metadata.get("shard_id") != task["task_id"]:
            return False, f"shard_id_mismatch:{metadata.get('shard_id')}"
        if int(metadata.get("files_attempted", -1)) != 1:
            return False, "files_attempted_mismatch"
        if int(metadata.get("files_processed", -1)) != 1:
            return False, "files_processed_mismatch"
        with uproot.open(str(root_path)) as root_file:
            if "Events" not in root_file:
                return False, "missing_Events_tree"
            entries = int(root_file["Events"].num_entries)
        if entries != int(metadata.get("events_written", -1)):
            return False, "event_count_mismatch"
    except Exception as exc:
        return False, f"validation_failed:{type(exc).__name__}:{exc}"
    return True, "valid"


def build_worker_bundle(repo: Path, destination: Path) -> None:
    source = repo / "autonomous_allhad/autonomous_allhad"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.new")
    with tarfile.open(temporary, "w:gz") as archive:
        for name in WORKER_FILES:
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=f"autonomous_allhad/{name}", recursive=False)
    os.replace(temporary, destination)


def wrapper_text(proxy_name: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
NAME="${{1:?missing name}}"
SHARD="${{2:?missing shard basename}}"
ROOT_DEST="${{3:?missing ROOT EOS destination}}"
META_DEST="${{4:?missing metadata EOS destination}}"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
cd "$WORKDIR"
case "$ROOT_DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS ROOT destination: $ROOT_DEST" >&2; exit 64 ;;
esac
case "$META_DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS metadata destination: $META_DEST" >&2; exit 64 ;;
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
tar -xzf objectcorr_2024_worker_optimized.tgz
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
  --shift nominal --record-workers 1
test -s out.root
test -s out.json
export NAME ROOT_DEST
"$PY" - <<'PY'
import json
import os
import pathlib
import uproot

metadata_path = pathlib.Path("out.json")
metadata = json.loads(metadata_path.read_text())
if metadata.get("status") not in {{"complete", "complete_with_bad_files"}}:
    raise SystemExit(f"invalid worker status: {{metadata.get('status')}}")
if metadata.get("shard_id") != os.environ["NAME"]:
    raise SystemExit(f"shard mismatch: {{metadata.get('shard_id')}}")
if int(metadata.get("files_attempted", -1)) != 1 or int(metadata.get("files_processed", -1)) != 1:
    raise SystemExit("singleton file count validation failed")
with uproot.open("out.root") as root_file:
    entries = int(root_file["Events"].num_entries)
if entries != int(metadata.get("events_written", -1)):
    raise SystemExit("ROOT/metadata event count mismatch")
metadata["root_file"] = os.environ["ROOT_DEST"]
metadata["condor_singleton_recovery"] = True
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
ROOT_URL="root://eosuser.cern.ch/$ROOT_DEST"
META_URL="root://eosuser.cern.ch/$META_DEST"
XRDCOPY="$(command -v xrdcp)"
XRDFS="$(command -v xrdfs)"
staged=0
for attempt in 1 2 3 4 5; do
  if "$XRDCOPY" -f --nopbar --streams 4 out.root "$ROOT_URL" &&
     "$XRDCOPY" -f --nopbar out.json "$META_URL" &&
     "$XRDFS" eosuser.cern.ch stat "$ROOT_DEST" >/dev/null &&
     "$XRDFS" eosuser.cern.ch stat "$META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "$((attempt * 10))"
done
test "$staged" -eq 1
echo "completed $NAME nominal singleton"
"""


def submit_text(
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py_bundle: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    proxy: Path,
) -> str:
    return f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shard_name) $(root_out) $(meta_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py_bundle}, {worker_bundle}, {payload_bundle}, {proxy}, $(shard)
transfer_output_files = ""
output = {logs}/$(parent)/$(name).out
error = {logs}/$(parent)/$(name).err
log = {logs}/campaign.log
request_cpus = 1
request_memory = 4000MB
request_disk = 6000MB
+JobFlavour = "workday"
+CampaignName = "nominal_2024_singleton_recovery_v2"
+RecoveryParent = "$(parent)"
queue name,parent,shard,shard_name,root_out,meta_out from {arguments}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare missing-only 2024 nominal singleton HTCondor recovery.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    recovery = Path(args.recovery).resolve()
    proxy = Path(args.proxy).resolve()
    manifest_path = recovery / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = list((manifest.get("tasks") or {}).values())
    if not tasks:
        raise RuntimeError("recovery manifest has no singleton tasks")
    if not proxy.is_file():
        raise FileNotFoundError(proxy)

    pending = []
    skipped = []
    validation_reasons: Counter[str] = Counter()
    for task in tasks:
        valid, reason = valid_output(task)
        validation_reasons[reason] += 1
        (skipped if valid else pending).append(task)

    condor = recovery / "condor"
    bundles = recovery / "bundles"
    logs = recovery / "logs_condor"
    arguments = condor / "arguments.txt"
    wrapper = condor / "run_nominal_singleton.sh"
    submit = condor / "submit.sub"
    worker_bundle = bundles / "objectcorr_2024_worker_optimized.tgz"
    source_campaign = repo / "autonomous_allhad/workflow/intermediate_2024_objectcorr_v1_20260721"
    payload_bundle = source_campaign / "bundles/objectcorr_2024_payloads.tgz"
    py_bundle = repo / "condor/py38.tgz"
    for required in (payload_bundle, py_bundle):
        if not required.is_file():
            raise FileNotFoundError(required)

    build_worker_bundle(repo, worker_bundle)
    lines = []
    by_kind: Counter[str] = Counter()
    by_process: Counter[str] = Counter()
    for task in sorted(pending, key=lambda item: item["task_id"]):
        parent = str(task["parent_shard_id"])
        Path(task["root_output"]).parent.mkdir(parents=True, exist_ok=True)
        (logs / parent).mkdir(parents=True, exist_ok=True)
        lines.append(
            " ".join(
                (
                    str(task["task_id"]),
                    parent,
                    str(task["singleton_shard"]),
                    Path(task["singleton_shard"]).name,
                    str(task["root_output"]),
                    str(task["metadata_output"]),
                )
            )
        )
        by_kind[str(task.get("kind") or "unknown")] += 1
        by_process[str(task.get("process") or "unknown")] += 1
    write_text(arguments, "\n".join(lines) + ("\n" if lines else ""))
    write_text(wrapper, wrapper_text(proxy.name), executable=True)
    write_text(
        submit,
        submit_text(wrapper, arguments, logs, py_bundle, worker_bundle, payload_bundle, proxy),
    )

    provenance = {
        "schema_version": "nominal_2024_singleton_condor_recovery_v2",
        "created_at": now(),
        "source_manifest": str(manifest_path),
        "source_manifest_updated_at": manifest.get("updated_at"),
        "worker_policy": {
            "source_files_per_job": 1,
            "record_workers": 1,
            "materialization": "selection_before_python_rows",
            "runtime_directory": "_CONDOR_SCRATCH_DIR",
            "stageout": "xrdcp_to_eos_with_xrdfs_verification",
            "forbidden_persistent_paths": ["/afs", "/tmp"],
        },
        "tasks": {
            "manifest": len(tasks),
            "already_valid": len(skipped),
            "submitted": len(pending),
            "by_kind": dict(sorted(by_kind.items())),
            "by_process": dict(sorted(by_process.items())),
            "validation_reasons": dict(sorted(validation_reasons.items())),
        },
        "files": {
            "arguments": str(arguments),
            "wrapper": str(wrapper),
            "submit": str(submit),
            "worker_bundle": str(worker_bundle),
            "worker_bundle_sha256": sha256(worker_bundle),
            "payload_bundle": str(payload_bundle),
            "payload_bundle_sha256": sha256(payload_bundle),
            "python_bundle": str(py_bundle),
            "python_bundle_sha256": sha256(py_bundle),
            "proxy": str(proxy),
        },
        "submission_command": f"module load lxbatch/eossubmit && condor_submit {submit}",
    }
    write_json(recovery / "condor_recovery_manifest.json", provenance)
    print(json.dumps(provenance["tasks"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
