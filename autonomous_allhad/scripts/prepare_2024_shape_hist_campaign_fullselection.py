#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from autonomous_allhad.object_corrections_2024 import PAYLOADS, validate_payloads
from autonomous_allhad.shape_histogram_2024_worker import (
    FINAL_JES_CORRECTION_SOURCES,
    FINAL_JES_PUBLIC_SOURCES,
    FINAL_SHAPE_NUISANCES,
    FINAL_SHAPE_VARIATIONS,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_NOMINAL_CAMPAIGN = DEFAULT_REPO / "autonomous_allhad/workflow/intermediate_2024_fullselection_v2_20260723"
DEFAULT_CAMPAIGN = DEFAULT_REPO / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v4_20260723"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_eos(path: Path, label: str) -> None:
    text = str(path.absolute())
    if not text.startswith("/eos/user/"):
        raise ValueError(f"{label} must be under /eos/user: {text}")
    if "/afs/" in text or text.startswith("/tmp"):
        raise ValueError(f"{label} violates path policy: {text}")


def build_worker_bundle(repo: Path, destination: Path) -> dict[str, Any]:
    package = repo / "autonomous_allhad/autonomous_allhad"
    histogram_builder = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    files = sorted(path for path in package.glob("*.py") if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"autonomous_allhad/{path.name}", recursive=False)
        archive.add(
            histogram_builder,
            arcname="workflow/build_flat_boosted_recoil_hists.py",
            recursive=False,
        )
    return {
        "path": str(destination),
        "files": [path.name for path in files] + ["workflow/build_flat_boosted_recoil_hists.py"],
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def build_payload_bundle(repo: Path, destination: Path, btag_path: Path) -> dict[str, Any]:
    analysis_data = repo / "analysis/data"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(analysis_data, arcname="analysis/data", recursive=True)
        archive.add(btag_path, arcname="analysis/hists/btageff2024.merged", recursive=False)
        archive.add(btag_path, arcname="payloads/BTV/btageff2024.merged", recursive=False)
        for payload in PAYLOADS:
            archive.add(payload.source, arcname=str(payload.relative), recursive=False)
            if payload.filename in {"jetid.json.gz", "jetvetomaps.json.gz"}:
                archive.add(
                    payload.source,
                    arcname=f"analysis/data/JMESF/2024/{payload.filename}",
                    recursive=False,
                )
    return {
        "path": str(destination),
        "analysis_data_source": str(analysis_data),
        "btag_efficiency_source": str(btag_path),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def wrapper_text(proxy_name: str) -> str:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
NAME="@D@{{1:?missing name}}"
SHARD="@D@{{2:?missing shard basename}}"
HIST_DEST="@D@{{3:?missing histogram EOS destination}}"
META_DEST="@D@{{4:?missing metadata EOS destination}}"
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
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig runtime_xrd
export HOME="@D@WORKDIR/runtime_home"
export TMPDIR="@D@WORKDIR/runtime_tmp"
export TMP="@D@TMPDIR"
export TEMP="@D@TMPDIR"
export XDG_CACHE_HOME="@D@WORKDIR/runtime_cache"
export MPLCONFIGDIR="@D@WORKDIR/runtime_mplconfig"
export NUMBA_CACHE_DIR="@D@WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="@D@WORKDIR/runtime_cache/pycache"
export AUTONOMOUS_ALLHAD_XRD_CACHE="@D@WORKDIR/runtime_xrd"
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
tar -xzf py38.tgz
tar -xzf shape_hist_2024_worker.tgz
tar -xzf shape_hist_2024_payloads.tgz
tar -xzf shape_input_shards.tgz
PY="@D@WORKDIR/bin/python3"
[ -x "@D@PY" ] || PY="@D@WORKDIR/bin/python"
[ -x "@D@PY" ] || PY="@D@WORKDIR/py38/bin/python"
test -x "@D@PY"
export PATH="@D@(dirname "@D@PY"):@D@PATH"
export LD_LIBRARY_PATH="@D@WORKDIR/lib:@D@WORKDIR/py38/lib:@D@{{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="@D@WORKDIR"
set +e
"@D@PY" -u -m autonomous_allhad.shape_histogram_2024_worker   --shard "@D@WORKDIR/shards/@D@SHARD"   --output "@D@WORKDIR/out.json.gz"   --metadata-output "@D@WORKDIR/out.meta.json"   --variation-group all --chunk-size 5000 --record-workers "@D@{{REQUEST_CPUS:-1}}"
WORKER_STATUS="@D@?"
set -e
test -s out.meta.json
test -s out.json.gz
export WORKER_STATUS
"@D@PY" - <<'PY'
import gzip
import hashlib
import json
import os
from pathlib import Path

worker_status = int(os.environ["WORKER_STATUS"])
metadata = json.loads(Path("out.meta.json").read_text())
with gzip.open("out.json.gz", "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
summary = metadata.get("summary") or {{}}
errors = []
if worker_status != 0:
    errors.append(f"worker exit status is {{worker_status}}")
if metadata.get("status") != "complete" or payload.get("status") != "complete":
    errors.append(f"non-complete status metadata={{metadata.get('status')}} payload={{payload.get('status')}}")
if int(summary.get("files_processed") or 0) != int(summary.get("files_attempted") or 0):
    errors.append("not every input file was processed")
if summary.get("bad_files"):
    errors.append(f"bad_files is non-empty: {{len(summary['bad_files'])}}")
if int(metadata.get("variation_count") or 0) != 40 or len(payload.get("variations") or []) != 40:
    errors.append("variation count is not 40")
expected_sections = {{
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
}}
if set((payload.get("output_policy") or {{}}).get("sections") or []) != expected_sections:
    errors.append("histogram sections do not match the adopted nominal schema")
actual_sha256 = hashlib.sha256(Path("out.json.gz").read_bytes()).hexdigest()
if actual_sha256 != metadata.get("histogram_sha256"):
    errors.append("histogram checksum mismatch")
if errors:
    raise SystemExit("; ".join(errors))
PY
export HIST_DEST META_DEST WORKER_STATUS
"@D@PY" -c 'import json,os,pathlib; p=pathlib.Path("out.meta.json"); d=json.loads(p.read_text()); d["histogram_file"]=os.environ["HIST_DEST"]; d["condor_worker_exit_status"]=int(os.environ["WORKER_STATUS"]); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
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
test "@D@WORKER_STATUS" -eq 0
echo "completed @D@NAME final_shape_variations=40"
"""
    return text.replace("@D@", "$")


def submit_text(
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py38: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    shard_bundle: Path,
    proxy: Path,
    initialdir: Path,
    request_memory_mb: int,
    request_cpus: int,
    job_flavour: str,
    campaign_name: str,
    source_files_per_job: int,
) -> str:
    return f"""universe = vanilla
initialdir = {initialdir}
executable = {wrapper}
arguments = $(name) $(shard_name) $(hist_out) $(meta_out)
getenv = False
environment = "REQUEST_CPUS={request_cpus}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker_bundle}, {payload_bundle}, {shard_bundle}, {proxy}
transfer_output_files = ""
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/campaign.log
request_cpus = {request_cpus}
request_memory = {request_memory_mb}MB
request_disk = 10000MB
+JobFlavour = "{job_flavour}"
+CampaignName = "{campaign_name}"
+SourceFilesPerJob = {source_files_per_job}
+VariationCount = 40
+NanoAODReadPolicy = "once_per_chunk_all_40"
+MaterializationPolicy = "feature_flat_preselection_before_python_rows"
queue name,shard_name,hist_out,meta_out from {arguments}
"""


def build_mc_shards(
    source_bundle: Path,
    campaign: Path,
    files_per_job: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    bundle_path = campaign / "bundles/shape_input_shards.tgz"
    shards: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    parent_shards = 0
    all_records: list[dict[str, Any]] = []
    with tarfile.open(source_bundle, "r:gz") as source:
        members = sorted(
            (
                member
                for member in source.getmembers()
                if member.isfile() and Path(member.name).name.startswith("mc_shard_")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"unable to read parent shard {member.name}")
            parent_payload = json.load(extracted)
            records = list(parent_payload.get("records") or [])
            if not records:
                raise RuntimeError(f"empty parent MC shard: {member.name}")
            parent_shards += 1
            for record in records:
                if not record.get("is_background"):
                    raise RuntimeError(f"non-background record in {member.name}")
                file_path = str(record.get("file_path") or "")
                if not file_path:
                    raise RuntimeError(f"record without file_path in {member.name}")
                if file_path in seen_files:
                    raise RuntimeError(f"duplicate MC file across parent shards: {file_path}")
                seen_files.add(file_path)
                all_records.append(record)

    with tempfile.TemporaryDirectory(prefix="shape_input_shards_") as temporary:
        destination = Path(temporary) / "shards"
        destination.mkdir(parents=True)
        for start in range(0, len(all_records), files_per_job):
            child_records = all_records[start : start + files_per_job]
            child_index = len(shards)
            child_id = f"shape_input_{child_index:05d}"
            canonical = json.dumps(
                child_records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            digest = hashlib.sha256(canonical).hexdigest()
            child_payload = {
                "schema_version": "shape_histogram_2024_input_shard_v3",
                "shard_id": child_id,
                "record_group": "mc_background",
                "record_digest": digest,
                "records_per_shard": len(child_records),
                "records": child_records,
            }
            target = destination / f"{child_id}.json"
            write_json(target, child_payload)
            shards.append(
                {
                    "name": child_id,
                    "basename": target.name,
                    "records": len(child_records),
                    "record_digest": digest,
                }
            )
        with tarfile.open(bundle_path, "w:gz") as archive:
            archive.add(destination, arcname="shards", recursive=True)

    if not shards:
        raise RuntimeError(f"no MC shards found in {source_bundle}")
    return shards, {
        "parent_shards": parent_shards,
        "source_records": len(all_records),
        "child_shards": len(shards),
        "unique_files": len(seen_files),
    }, {
        "path": str(bundle_path),
        "size": bundle_path.stat().st_size,
        "sha256": sha256(bundle_path),
        "files": len(shards),
        "layout": "shards/shape_input_<index>.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare compact 2024 shape-histogram HTCondor campaign with final 11-source JES."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--nominal-campaign", type=Path, default=DEFAULT_NOMINAL_CAMPAIGN)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--files-per-job", type=int, default=4)
    parser.add_argument("--job-flavour", default="nextweek")
    parser.add_argument("--request-memory-mb", type=int, default=16000)
    parser.add_argument("--request-cpus", type=int, default=4)
    args = parser.parse_args(argv)

    if args.files_per_job <= 0:
        raise ValueError("--files-per-job must be positive")
    if args.request_memory_mb <= 0:
        raise ValueError("--request-memory-mb must be positive")
    if args.request_cpus <= 0:
        raise ValueError("--request-cpus must be positive")

    repo = args.repo.absolute()
    nominal_campaign = args.nominal_campaign.absolute()
    campaign = args.campaign.absolute()
    for label, path in (("repo", repo), ("nominal campaign", nominal_campaign), ("campaign", campaign)):
        require_eos(path, label)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty campaign: {campaign}")

    nominal_manifest_path = nominal_campaign / "manifest.json"
    nominal_manifest = read_json(nominal_manifest_path)
    if nominal_manifest.get("year") != 2024:
        raise RuntimeError("nominal campaign is not 2024")
    if nominal_manifest.get("requested_shifts") != ["nominal"]:
        raise RuntimeError("shape campaign must be based on a nominal-only source campaign")
    if int((nominal_manifest.get("shards") or {}).get("background") or 0) <= 0:
        raise RuntimeError("nominal campaign has no background MC shards")
    source_bundle = Path((nominal_manifest.get("shard_bundle") or {}).get("path") or "")
    if not source_bundle.is_file():
        raise FileNotFoundError(f"nominal input shard bundle is missing: {source_bundle}")
    expected_source_sha256 = str((nominal_manifest.get("shard_bundle") or {}).get("sha256") or "")
    if sha256(source_bundle) != expected_source_sha256:
        raise RuntimeError("nominal input shard bundle checksum mismatch")

    payload_status = validate_payloads(repo)
    if payload_status.get("status") != "valid":
        raise RuntimeError(f"2024 correction payload validation failed: {payload_status.get('errors')}")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    btag = repo / "analysis/hists/btageff2024.merged"
    histogram_builder = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    for label, path in (
        ("Python bundle", py38),
        ("proxy", proxy),
        ("b-tag efficiency", btag),
        ("histogram builder", histogram_builder),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    for directory in ("bundles", "condor", "logs", "outputs", "reports"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)

    shards, shard_stats, shard_bundle = build_mc_shards(
        source_bundle,
        campaign,
        args.files_per_job,
    )
    worker_bundle_path = campaign / "bundles/shape_hist_2024_worker.tgz"
    payload_bundle_path = campaign / "bundles/shape_hist_2024_payloads.tgz"
    worker_bundle = build_worker_bundle(repo, worker_bundle_path)
    payload_bundle = build_payload_bundle(repo, payload_bundle_path, btag)

    wrapper = campaign / "condor/run_shape_hist_2024.sh"
    wrapper.write_text(wrapper_text(proxy.name))
    wrapper.chmod(0o755)

    rows = []
    for index, shard in enumerate(shards):
        name = f"shape_mc_{index:05d}"
        hist_out = campaign / f"outputs/{name}.json.gz"
        meta_out = campaign / f"outputs/{name}.meta.json"
        rows.append(f"{name} {shard['basename']} {hist_out} {meta_out}")

    arguments = campaign / "condor/arguments.txt"
    arguments.write_text("\n".join(rows) + "\n")

    all_submit = campaign / "condor/submit_all.sub"
    submit_common = {
        "wrapper": wrapper,
        "logs": campaign / "logs",
        "py38": py38,
        "worker_bundle": worker_bundle_path,
        "payload_bundle": payload_bundle_path,
        "shard_bundle": Path(shard_bundle["path"]),
        "proxy": proxy,
        "initialdir": campaign / "condor",
        "request_memory_mb": args.request_memory_mb,
        "request_cpus": args.request_cpus,
        "job_flavour": args.job_flavour,
        "campaign_name": campaign.name,
        "source_files_per_job": args.files_per_job,
    }
    all_submit.write_text(submit_text(arguments=arguments, **submit_common))

    manifest = {
        "schema_version": "shape_histogram_2024_campaign_v2",
        "status": "prepared_not_submitted",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "year": 2024,
        "campaign": str(campaign),
        "source_nominal_campaign": str(nominal_campaign),
        "source_nominal_manifest": str(nominal_manifest_path),
        "source_policy": {
            "background_mc_only": True,
            "data_shape_jobs": 0,
            "fastsim_signal": "deferred_by_user",
            "nano_chunk_read": "once per chunk; nominal/unaffected corrections cached; shifted chunks streamed one at a time",
            "parallelism": "independent source files processed concurrently inside each multi-core job",
            "histogram_schema": "nominal regions, variables, and bins; legacy 17-bin and 42-bin search schemes excluded",
        },
        "jes_source_policy": {
            "status": "adopted_final_11_sources",
            "public_sources": list(FINAL_JES_PUBLIC_SOURCES),
            "correctionlib_sources": list(FINAL_JES_CORRECTION_SOURCES),
            "excluded_validation_envelopes": ["jesTotal", "jesRegroupedTotal"],
        },
        "shape_nuisances": list(FINAL_SHAPE_NUISANCES),
        "shape_nuisance_count": len(FINAL_SHAPE_NUISANCES),
        "variations": list(FINAL_SHAPE_VARIATIONS),
        "variation_count": len(FINAL_SHAPE_VARIATIONS),
        "jobs": {
            "total": len(shards),
            "condor_pilot": 0,
            "local_pilot_replacement": "required and completed before submission",
            "source_files_per_job": args.files_per_job,
            "source_parent_shards": shard_stats["parent_shards"],
            "source_records": shard_stats["source_records"],
            "unique_source_files": shard_stats["unique_files"],
            "output_files_per_job": 2,
        },
        "output_policy": {
            "shifted_root_files": 0,
            "event_rows": 0,
            "histogram_files": len(shards),
            "metadata_files": len(shards),
            "histogram_format": "gzip-compressed JSON",
            "histogram_definition_source": str(histogram_builder),
            "search_bin_schemes": [
                "boosted_an17_selected_recoil6_with_nt0_wsplit_SR",
                "cat2_LLCR_lowDeltaM",
                "cat3_QCDCR_lowDeltaM",
                "cat4_GCR_lowDeltaM",
                "cat5_DY2E_lowDeltaM",
                "cat6_DY2M_lowDeltaM",
                "cat7_SR_lowDeltaM",
            ],
            "histogram_sections": [
                "histograms",
                "search_bin_histograms",
                "lowdm_variable_histograms",
                "highdm_variable_histograms",
            ],
            "search_bins": "54-bin high-dM plus 53-bin low-dM per region",
            "normalization": "deferred to nominal campaign metadata merge",
        },
        "runtime_policy": {
            "execution_directory": "_CONDOR_SCRATCH_DIR",
            "python_environment": "transferred py38.tgz",
            "worker_environment": "transferred shape_hist_2024_worker.tgz",
            "correction_environment": "transferred shape_hist_2024_payloads.tgz",
            "stageout": "xrdcp to root://eosuser.cern.ch//eos/user/t/taiwoo with xrdfs verification",
            "persistent_path_policy": "EOS-only; transient files live in the Condor scratch directory",
            "job_flavour": args.job_flavour,
            "request_cpus": args.request_cpus,
            "request_memory_mb": args.request_memory_mb,
        },
        "payload_validation": payload_status,
        "btag_efficiency": {
            "path": str(btag),
            "sha256": sha256(btag),
            "size": btag.stat().st_size,
            "hard_failure_if_not_applied": True,
        },
        "python_bundle": {"path": str(py38), "size": py38.stat().st_size, "sha256": sha256(py38)},
        "worker_bundle": worker_bundle,
        "payload_bundle": payload_bundle,
        "shard_bundle": shard_bundle,
        "submission": {
            "pool": "eossubmit",
            "all": f"module load lxbatch/eossubmit && condor_submit {all_submit}",
            "all_submit_file": str(all_submit),
            "pilot_policy": "no Condor pilot; replaced by validated lxplus local all-40 tests",
        },
        "downstream": {
            "nominal_normalization_command": (
                f"{repo}/autonomous_allhad/workflow/merge_flat_ntuple_metadata.py "
                f"--inputs {nominal_campaign}/outputs/nominal "
                f"--output {nominal_campaign}/normalization.json"
            ),
            "shape_merge_module": "autonomous_allhad.merge_shape_histogram_2024",
            "plotting_and_combine_policy": "attach variation leaves to the existing flat histogram payload; no new plotting code",
        },
    }
    write_json(campaign / "manifest.json", manifest)

    readiness = f"""# 2024 shape-histogram campaign readiness

Status: prepared, not submitted.

- MC shape jobs: {len(shards)}
- Source NanoAOD files: {shard_stats['source_records']} ({args.files_per_job} per job maximum)
- Shifted intermediate ROOT files: 0
- Compact histogram outputs: {len(shards)} gzip JSON files plus metadata
- Shape nuisance pairs: {len(FINAL_SHAPE_NUISANCES)}
- Directional variations: {len(FINAL_SHAPE_VARIATIONS)}
- Final JES sources: {len(FINAL_JES_PUBLIC_SOURCES)}
- JES Total and Regrouped Total: excluded
- Search bins: final 54-bin high-dM SR plus 53-bin Low-dM categories in all six regions
- Detailed distributions: nominal CR/VR/SR region, variable, and bin definitions
- Legacy 17-bin and 42-bin schemes: excluded
- Runtime: transferred tgz bundles under Condor scratch; {args.request_cpus} files processed concurrently per job
- Stageout: EOS through xrdcp
- Pilot: no Condor pilot; replaced by local lxplus all-40 validation
- Submission: full campaign only after local smoke, checksum validation, and AFS-free Condor dry-run
"""
    (campaign / "reports/readiness.md").write_text(readiness)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
