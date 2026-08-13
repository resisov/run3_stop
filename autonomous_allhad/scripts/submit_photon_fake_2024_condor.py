#!/usr/bin/env python3
"""Move the pending photon-fake sidecar shards from local execution to HTCondor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/photon_fake_2024_local2400k_v3_20260726"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_local_runner(repo: Path) -> Any:
    path = repo / "autonomous_allhad/scripts/run_photon_fake_2024_local.py"
    spec = importlib.util.spec_from_file_location("photon_fake_local_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import local runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrapper_text(
    proxy_name: str,
    local_jobs_name: str,
    python_bundle_name: str,
    worker_bundle_name: str,
    payload_bundle_name: str,
) -> str:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
JOB_INDEX="@D@{{1:?missing job index}}"
NAME="@D@{{2:?missing name}}"
SHARD_BASENAME="@D@{{3:?missing shard basename}}"
HIST_DEST="@D@{{4:?missing histogram destination}}"
META_DEST="@D@{{5:?missing metadata destination}}"
WORKDIR="@D@{{_CONDOR_SCRATCH_DIR:-@D@PWD}}"
cd "@D@WORKDIR"
case "@D@HIST_DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS histogram destination: @D@HIST_DEST" >&2; exit 64 ;;
esac
case "@D@META_DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS metadata destination: @D@META_DEST" >&2; exit 64 ;;
esac
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mpl runtime_xrd
export HOME="@D@WORKDIR/runtime_home"
export TMPDIR="@D@WORKDIR/runtime_tmp"
export TMP="@D@TMPDIR"
export TEMP="@D@TMPDIR"
export XDG_CACHE_HOME="@D@WORKDIR/runtime_cache"
export NUMBA_CACHE_DIR="@D@WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="@D@WORKDIR/runtime_cache/pycache"
export MPLCONFIGDIR="@D@WORKDIR/runtime_mpl"
export AUTONOMOUS_ALLHAD_XRD_CACHE="@D@WORKDIR/runtime_xrd"
export AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=1
export AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE=0
export AUTONOMOUS_ALLHAD_XRD_STREAMS=4
export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=900
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
export X509_USER_PROXY="@D@WORKDIR/{proxy_name}"
chmod 600 "@D@X509_USER_PROXY"
tar -xzf {python_bundle_name}
tar -xzf {worker_bundle_name}
tar -xzf {payload_bundle_name}
PY="@D@WORKDIR/bin/python3"
[ -x "@D@PY" ] || PY="@D@WORKDIR/bin/python"
[ -x "@D@PY" ] || PY="@D@WORKDIR/py38/bin/python"
test -x "@D@PY"
export PATH="@D@(dirname "@D@PY"):@D@PATH"
export LD_LIBRARY_PATH="@D@WORKDIR/lib:@D@WORKDIR/py38/lib:@D@{{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="@D@WORKDIR"
export JOB_INDEX NAME SHARD_BASENAME
"@D@PY" - <<'PY'
import gzip
import json
import os
from pathlib import Path

with gzip.open("{local_jobs_name}", "rt", encoding="utf-8") as handle:
    bundle = json.load(handle)
jobs = list(bundle.get("jobs") or [])
index = int(os.environ["JOB_INDEX"])
if index < 0 or index >= len(jobs):
    raise SystemExit(f"job index out of range: {{index}}")
job = jobs[index]
if str(job.get("name")) != os.environ["NAME"]:
    raise SystemExit("job name differs from indexed local-job manifest")
if str(job.get("shard_basename")) != os.environ["SHARD_BASENAME"]:
    raise SystemExit("shard basename differs from indexed local-job manifest")
Path(os.environ["SHARD_BASENAME"]).write_text(
    json.dumps(job["shard"], sort_keys=True, allow_nan=False) + "\\n"
)
PY
"@D@PY" -u -m autonomous_allhad.photon_fake_2024_worker \
  --shard "@D@WORKDIR/@D@SHARD_BASENAME" \
  --output "@D@WORKDIR/out.json.gz" \
  --metadata-output "@D@WORKDIR/out.meta.json" \
  --chunk-size 200000 \
  --prefilter-block-size 5000 \
  --record-workers 2
test -s out.json.gz
test -s out.meta.json
export SHARD_BASENAME
"@D@PY" - <<'PY'
import gzip
import hashlib
import json
import os
from pathlib import Path

metadata = json.loads(Path("out.meta.json").read_text())
shard = json.loads(Path(os.environ["SHARD_BASENAME"]).read_text())
with gzip.open("out.json.gz", "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
summary = metadata.get("summary") or {{}}
records = list(shard.get("records") or [])
expected_paths = sorted(str(record["file_path"]) for record in records)
observed_paths = sorted(
    str(record["file_path"]) for record in summary.get("file_records") or []
)
errors = []
if metadata.get("status") != "complete" or payload.get("status") != "complete":
    errors.append("worker output is not complete")
if str(metadata.get("source_record_digest") or "") != str(
    shard.get("record_digest") or ""
):
    errors.append("source record digest differs from shard")
if int(summary.get("files_attempted") or 0) != len(records):
    errors.append("attempted file count differs from shard")
if int(summary.get("files_processed") or 0) != len(records):
    errors.append("not every input file was processed")
if int(summary.get("events_read") or 0) != int(shard["expected_events"]):
    errors.append("processed event count differs from shard")
if observed_paths != expected_paths:
    errors.append("processed ROOT-file coverage differs from shard")
if summary.get("bad_files"):
    errors.append(f"bad_files is non-empty: {{len(summary['bad_files'])}}")
if int(summary.get("target_cutbased_mismatch_objects") or 0) != 0:
    errors.append("bitmap target does not match Photon_cutBased>=2")
actual = hashlib.sha256(Path("out.json.gz").read_bytes()).hexdigest()
if actual != metadata.get("histogram_sha256"):
    errors.append("histogram checksum mismatch")
if errors:
    raise SystemExit("; ".join(errors))
PY
HIST_URL="root://eosuser.cern.ch/@D@HIST_DEST"
META_URL="root://eosuser.cern.ch/@D@META_DEST"
XRDCOPY="@D@(command -v xrdcp)"
XRDFS="@D@(command -v xrdfs)"
staged=0
for attempt in 1 2 3 4 5; do
  if "@D@XRDCOPY" -f --nopbar out.json.gz "@D@HIST_URL" &&
     "@D@XRDCOPY" -f --nopbar out.meta.json "@D@META_URL" &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@HIST_DEST" >/dev/null &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "@D@((attempt * 10))"
done
test "@D@staged" -eq 1
echo "completed @D@NAME photon-fake N-1 sidecar"
"""
    return text.replace("@D@", "$")


def submit_text(
    campaign: Path,
    control_dir: Path,
    python_bundle: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    proxy: Path,
    local_jobs: Path,
    request_memory_mb: int,
    request_disk_mb: int,
    max_materialize: int,
    job_flavour: str,
) -> str:
    materialize_line = (
        f"max_materialize = {max_materialize}\n" if max_materialize > 0 else ""
    )
    return f"""universe = vanilla
initialdir = {control_dir}
executable = {control_dir / 'run_photon_fake_2024_pending.sh'}
arguments = $(job_index) $(name) $(shard_basename) $(hist_out) $(meta_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {python_bundle}, {worker_bundle}, {payload_bundle}, {proxy}, {local_jobs}
transfer_output_files = ""
output = $(log_dir)/$(name).condor.out
error = $(log_dir)/$(name).condor.err
log = {control_dir / 'campaign.log'}
request_cpus = 2
request_memory = {request_memory_mb}MB
request_disk = {request_disk_mb}MB
+JobFlavour = "{job_flavour}"
+CampaignName = "{campaign.name}_pending"
+PhysicsTask = "photon_fake_measurement"
+NominalIntermediateMutation = False
{materialize_line}\
queue job_index,name,shard_basename,hist_out,meta_out,log_dir from {control_dir / 'pending_arguments.txt'}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--request-memory-mb", type=int, default=5_000)
    parser.add_argument("--request-disk-mb", type=int, default=15_000)
    parser.add_argument("--max-materialize", type=int, default=0)
    parser.add_argument("--job-flavour", default="workday")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    if not str(repo).startswith("/eos/user/") or not str(campaign).startswith(
        "/eos/user/"
    ):
        raise ValueError("repository and campaign must be under /eos/user")
    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    local_status_path = campaign / "local_status.json"
    if local_status_path.is_file():
        status = read_json(local_status_path)
        if status.get("status") != "stopped":
            raise RuntimeError(
                f"local campaign is not stopped: {status.get('status')}"
            )
        controller_pid = int(status.get("controller_pid") or 0)
        if controller_pid > 0:
            probe = subprocess.run(
                ["ps", "-p", str(controller_pid), "-o", "cmd="],
                text=True,
                capture_output=True,
            )
            if "run_photon_fake_2024_local.py" in probe.stdout:
                raise RuntimeError("local photon-fake controller is still running")
    elif manifest.get("status") != "prepared_local":
        raise RuntimeError(
            "campaign has neither a stopped local controller nor a "
            f"prepared_local manifest: {manifest.get('status')}"
        )
    if manifest.get("nominal_intermediate_mutation") is not False:
        raise RuntimeError("campaign does not explicitly preserve nominal outputs")
    if not (manifest.get("local_job_bundle") or {}).get("path"):
        raise RuntimeError("campaign has no local job bundle")
    local_runner = load_local_runner(repo)
    jobs = local_runner.read_jobs(campaign, manifest)
    valid_jobs: list[dict[str, Any]] = []
    pending_jobs: list[tuple[int, dict[str, Any], str]] = []
    for index, job in enumerate(jobs):
        valid, reason = local_runner.validate_output(job)
        if valid:
            valid_jobs.append(job)
        else:
            pending_jobs.append((index, job, reason))
    if not pending_jobs:
        raise RuntimeError("all photon-fake jobs are already valid")

    python_bundle = repo / "condor/py38.tgz"
    worker_bundle = Path(manifest["bundles"]["worker"]["path"])
    payload_bundle = Path(manifest["bundles"]["payload"]["path"])
    proxy = Path(manifest["bundles"]["proxy"]["path"])
    local_jobs = Path(manifest["local_job_bundle"]["path"])
    for path in (
        python_bundle,
        worker_bundle,
        payload_bundle,
        proxy,
        local_jobs,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if sha256(worker_bundle) != str(manifest["bundles"]["worker"]["sha256"]):
        raise RuntimeError("worker bundle checksum mismatch")
    if sha256(payload_bundle) != str(manifest["bundles"]["payload"]["sha256"]):
        raise RuntimeError("payload bundle checksum mismatch")
    if sha256(local_jobs) != str(manifest["local_job_bundle"]["sha256"]):
        raise RuntimeError("local job bundle checksum mismatch")

    condor_dir = campaign / "condor"
    if condor_dir.exists() and any(condor_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty Condor directory: {condor_dir}")
    condor_dir.mkdir(parents=True, exist_ok=True)

    arguments = condor_dir / "pending_arguments.txt"
    lines: list[str] = []
    for index, job, _reason in pending_jobs:
        log_dir = condor_dir / "logs" / str(job["process"])
        log_dir.mkdir(parents=True, exist_ok=True)
        lines.append(
            " ".join(
                [
                    str(index),
                    str(job["name"]),
                    str(job["shard_basename"]),
                    str(job["histogram"]),
                    str(job["metadata"]),
                    str(log_dir),
                ]
            )
        )
    arguments.write_text("\n".join(lines) + "\n")
    wrapper = condor_dir / "run_photon_fake_2024_pending.sh"
    wrapper.write_text(
        wrapper_text(
            proxy.name,
            local_jobs.name,
            python_bundle.name,
            worker_bundle.name,
            payload_bundle.name,
        )
    )
    wrapper.chmod(0o755)
    submit = condor_dir / "pending.sub"
    submit.write_text(
        submit_text(
            campaign,
            condor_dir,
            python_bundle,
            worker_bundle,
            payload_bundle,
            proxy,
            local_jobs,
            args.request_memory_mb,
            args.request_disk_mb,
            args.max_materialize,
            args.job_flavour,
        )
    )
    pending_manifest = {
        "schema_version": "photon_fake_2024_pending_condor_v1",
        "status": "prepared",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "schedd_pool": "eossubmit",
        "schedd": "bigbird24.cern.ch",
        "persistent_path_policy": "EOS only; no AFS",
        "valid_jobs_preserved": len(valid_jobs),
        "pending_jobs": len(pending_jobs),
        "total_jobs": len(jobs),
        "pending_by_process": {
            process: sum(
                1 for _, job, _ in pending_jobs if job["process"] == process
            )
            for process in sorted({job["process"] for _, job, _ in pending_jobs})
        },
        "pending_validation_reasons": {
            reason: sum(1 for _, _, item in pending_jobs if item == reason)
            for reason in sorted({item for _, _, item in pending_jobs})
        },
        "files": {
            "eos_inputs": {
                source.name: {
                    "path": str(source),
                    "size": source.stat().st_size,
                    "sha256": sha256(source),
                }
                for source in (
                    python_bundle,
                    worker_bundle,
                    payload_bundle,
                    proxy,
                    local_jobs,
                )
            },
            "arguments": {
                "path": str(arguments),
                "size": arguments.stat().st_size,
                "sha256": sha256(arguments),
            },
            "wrapper": {
                "path": str(wrapper),
                "size": wrapper.stat().st_size,
                "sha256": sha256(wrapper),
            },
            "submit": {
                "path": str(submit),
                "size": submit.stat().st_size,
                "sha256": sha256(submit),
            },
        },
        "resources": {
            "request_cpus": 2,
            "request_memory_mb": args.request_memory_mb,
            "request_disk_mb": args.request_disk_mb,
            "max_materialize": args.max_materialize,
            "job_flavour": args.job_flavour,
            "xrd_prefer_cache": True,
            "chunk_size": 200_000,
            "prefilter_block_size": 5_000,
        },
    }
    pending_manifest_path = condor_dir / "pending_manifest.json"
    write_json(pending_manifest_path, pending_manifest)

    if args.submit:
        result = subprocess.run(
            [
                "condor_submit",
                "-name",
                "bigbird24.cern.ch",
                str(submit),
            ],
            text=True,
            capture_output=True,
        )
        cluster_match = re.search(
            r"cluster\s+(\d+)",
            result.stdout,
            re.IGNORECASE,
        )
        pending_manifest["submission"] = {
            "attempted": True,
            "exit_status": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cluster_id": (
                int(cluster_match.group(1)) if cluster_match is not None else None
            ),
            "submitted_jobs": len(pending_jobs) if result.returncode == 0 else 0,
            "submitted_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        pending_manifest["status"] = (
            "submitted" if result.returncode == 0 else "submit_failed"
        )
        write_json(pending_manifest_path, pending_manifest)
        manifest["status"] = (
            "submitted_condor_pending"
            if result.returncode == 0
            else "condor_pending_submit_failed"
        )
        manifest["condor_migration"] = {
            "pending_manifest": str(pending_manifest_path),
            "cluster_id": pending_manifest["submission"]["cluster_id"],
            "valid_jobs_preserved": len(valid_jobs),
            "pending_jobs_submitted": (
                len(pending_jobs) if result.returncode == 0 else 0
            ),
            "migrated_at": pending_manifest["submission"]["submitted_at"],
        }
        write_json(manifest_path, manifest)
        if result.returncode != 0:
            raise RuntimeError(
                f"condor_submit failed: {result.stderr or result.stdout}"
            )

    print(
        json.dumps(
            {
                "status": pending_manifest["status"],
                "valid_jobs_preserved": len(valid_jobs),
                "pending_jobs": len(pending_jobs),
                "pending_by_process": pending_manifest["pending_by_process"],
                "cluster_id": (
                    (pending_manifest.get("submission") or {}).get("cluster_id")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
