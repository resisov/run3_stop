#!/usr/bin/env python3
"""Prepare the EOS-schedd campaign for the photon template-fit event worker."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_CAMPAIGN = DEFAULT_REPO / "autonomous_allhad/workflow/photon_fake_template_run2method_20260810"
DEFAULT_METADATA = DEFAULT_REPO / "analysis/metadata/KNU_2024_v4.json.gz"
DEFAULT_NORMALIZATION = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/intermediate_2024_fullselection_v3_lowdm_relaxed_20260724/"
    "final_nominal_inputs_20260725/normalization.json"
)
DEFAULT_RUNTIME_PAYLOADS = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/intermediate_2024_fullselection_v3_lowdm_relaxed_20260724/"
    "recovery/badfile_18683_local_actual_once_20260725/runtime_bundle"
)
PROCESSES = ("QCD", "GJ", "EGamma", "DY", "TT", "WtoLNu", "ST", "VV", "Zto2Nu")
ALLOWED_DY_PREFIXES = (
    "DYto2E-4Jets_Bin-MLL-50_",
    "DYto2Mu-4Jets_Bin-MLL-50_",
    "DYto2Tau-4Jets_Bin-MLL-50_",
)


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_records(
    metadata: dict[str, Any],
    normalization: dict[str, Any],
    dataset_regex: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_name = {str(record["dataset"]): record for record in (normalization.get("datasets") or {}).values()}
    records: dict[str, list[dict[str, Any]]] = {process: [] for process in PROCESSES}
    missing: list[str] = []
    rejected_legacy_dy: list[str] = []
    pattern = re.compile(dataset_regex) if dataset_regex else None
    for dataset, metadata_record in metadata.items():
        if pattern is not None and pattern.search(str(dataset)) is None:
            continue
        normalized = by_name.get(str(dataset))
        if normalized is None:
            missing.append(str(dataset))
            continue
        process = str(normalized.get("process") or "unknown")
        if process not in records:
            continue
        if process == "DY" and not str(dataset).startswith(ALLOWED_DY_PREFIXES):
            rejected_legacy_dy.append(str(dataset))
            continue
        files = list(metadata_record.get("files") or [])
        if not files:
            continue
        total_events = int(normalized.get("events_read") or 0)
        base, remainder = divmod(total_events, len(files))
        for index, file_path in enumerate(files):
            expected = base + (1 if index < remainder else 0)
            records[process].append(
                {
                    "dataset": str(dataset),
                    "file_path": str(file_path),
                    "process_group": process,
                    "year": "2024",
                    "is_data": bool(normalized.get("is_data")),
                    "is_background": bool(normalized.get("is_background")),
                    "is_signal": False,
                    "xsec_pb": normalized.get("xsec_pb"),
                    "expected_events": expected,
                    "event_count_basis": "dataset-average estimate for packing only",
                }
            )
    for process in records:
        records[process].sort(key=lambda item: (str(item["dataset"]), str(item["file_path"])))
    return records, {
        "dataset_regex": dataset_regex,
        "metadata_datasets_without_normalization": len(missing),
        "examples": missing[:20],
        "dy_policy": "use only DYto2E/Mu/Tau-4Jets; reject legacy PTLL-binned DY",
        "rejected_legacy_dy_datasets": sorted(rejected_legacy_dy),
    }


def pack_process(records: list[dict[str, Any]], target_events: int, max_files: int) -> list[list[dict[str, Any]]]:
    bins: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (-int(item["expected_events"]), str(item["file_path"]))):
        candidates = [
            slot
            for slot in bins
            if len(slot["records"]) < max_files
            and int(slot["events"]) + int(record["expected_events"]) <= target_events
        ]
        if candidates:
            target = max(candidates, key=lambda item: int(item["events"]))
        else:
            target = {"events": 0, "records": []}
            bins.append(target)
        target["records"].append(record)
        target["events"] += int(record["expected_events"])
    return [slot["records"] for slot in bins]


def build_worker_bundle(repo: Path, destination: Path) -> dict[str, Any]:
    package = repo / "autonomous_allhad/autonomous_allhad"
    builder = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    required = package / "photon_fake_template_2024_worker.py"
    if not required.is_file():
        raise FileNotFoundError(required)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(package.glob("*.py")):
            archive.add(path, arcname=f"autonomous_allhad/{path.name}", recursive=False)
        archive.add(builder, arcname="workflow/build_flat_boosted_recoil_hists.py", recursive=False)
    return {"path": str(destination), "size": destination.stat().st_size, "sha256": sha256(destination)}


def build_payload_bundle(
    source: Path,
    destination: Path,
    current_btag_efficiency: Path,
    current_analysis_data: Path,
) -> dict[str, Any]:
    required = (
        source / "analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json",
        source / "analysis/hists/btageff2024.merged",
        source / "payloads/BTV/btageff2024.merged",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not current_btag_efficiency.is_file():
        raise FileNotFoundError(current_btag_efficiency)
    if not current_analysis_data.is_dir():
        raise FileNotFoundError(current_analysis_data)
    analysis_data_files = sorted(path for path in current_analysis_data.rglob("*") if path.is_file())
    if not analysis_data_files:
        raise RuntimeError(f"no files found below {current_analysis_data}")
    analysis_data_members = {
        f"analysis/data/{path.relative_to(current_analysis_data)}"
        for path in analysis_data_files
    }
    replaced = {
        "analysis/hists/btageff2024.merged",
        "payloads/BTV/btageff2024.merged",
    } | analysis_data_members
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            relative = str(path.relative_to(source))
            if path.is_file() and relative not in replaced:
                archive.add(path, arcname=relative, recursive=False)
        for path in analysis_data_files:
            relative = f"analysis/data/{path.relative_to(current_analysis_data)}"
            archive.add(path, arcname=relative, recursive=False)
        for relative in sorted(replaced):
            if relative in analysis_data_members:
                continue
            archive.add(current_btag_efficiency, arcname=relative, recursive=False)
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "btag_efficiency_source": str(current_btag_efficiency),
        "btag_efficiency_sha256": sha256(current_btag_efficiency),
        "btag_efficiency_size": current_btag_efficiency.stat().st_size,
        "btag_efficiency_archive_members": [
            "analysis/hists/btageff2024.merged",
            "payloads/BTV/btageff2024.merged",
        ],
        "analysis_data_source": str(current_analysis_data),
        "analysis_data_file_count": len(analysis_data_files),
        "analysis_data_sha256": canonical_digest(
            {str(path.relative_to(current_analysis_data)): sha256(path) for path in analysis_data_files}
        ),
    }


def wrapper_text(proxy_name: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
NAME="${{1:?missing name}}"
SHARD="${{2:?missing shard}}"
OUTPUT="${{3:?missing output}}"
METADATA="${{4:?missing metadata}}"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
cd "$WORKDIR"
case "$OUTPUT" in /eos/user/t/taiwoo/*) ;; *) exit 64;; esac
case "$METADATA" in /eos/user/t/taiwoo/*) ;; *) exit 64;; esac
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mpl runtime_xrd
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export MPLCONFIGDIR="$WORKDIR/runtime_mpl"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$WORKDIR/runtime_xrd"
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
tar -xzf photon_fake_template_worker.tgz
tar -xzf photon_fake_template_payloads.tgz
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORKDIR"
"$PY" -u -m autonomous_allhad.photon_fake_template_2024_worker \
  --shard "$WORKDIR/$(basename "$SHARD")" \
  --output "$WORKDIR/out.json.gz" \
  --metadata-output "$WORKDIR/out.meta.json" \
  --chunk-size 20000 \
  --prefilter-block-size 512 \
  --record-workers "${{REQUEST_CPUS:-1}}"
"$PY" - <<'PY'
import gzip, hashlib, json
from pathlib import Path
metadata=json.loads(Path('out.meta.json').read_text())
with gzip.open('out.json.gz','rt') as handle: payload=json.load(handle)
summary=metadata.get('summary') or {{}}
errors=[]
if metadata.get('status')!='complete' or payload.get('status')!='complete': errors.append('output incomplete')
if int(summary.get('files_processed') or 0)!=int(summary.get('files_attempted') or 0): errors.append('file coverage incomplete')
if summary.get('bad_files'): errors.append('bad_files non-empty')
if hashlib.sha256(Path('out.json.gz').read_bytes()).hexdigest()!=metadata.get('event_file_sha256'): errors.append('checksum mismatch')
if errors:
    import sys
    print(json.dumps({{"validation_errors": errors, "bad_files": summary.get("bad_files") or []}}, sort_keys=True), file=sys.stderr)
    raise SystemExit('; '.join(errors))
PY
for attempt in 1 2 3 4 5; do
  if xrdcp -f --nopbar out.json.gz "root://eosuser.cern.ch/$OUTPUT" &&
     xrdcp -f --nopbar out.meta.json "root://eosuser.cern.ch/$METADATA"; then exit 0; fi
  sleep $((attempt * 10))
done
exit 74
"""


def submit_text(campaign: Path, python: Path, worker: Path, payloads: Path, proxy: Path, cpus: int) -> str:
    initialdir = campaign / "condor/initialdir"
    return f"""universe = vanilla
initialdir = {initialdir}
executable = {campaign / 'condor/run.sh'}
arguments = $(name) $(shard_path) $(hist_dest) $(meta_dest)
getenv = False
environment = "REQUEST_CPUS={cpus}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {python}, {worker}, {payloads}, {proxy}, $(shard_path)
transfer_output_files = ""
output = $(log_dir)/$(name).out
error = $(log_dir)/$(name).err
log = {campaign / 'condor/campaign.log'}
request_cpus = {cpus}
request_memory = 32000MB
request_disk = 16000MB
+JobFlavour = "workday"
+CampaignName = "{campaign.name}"
+PhysicsTask = "photon_fake_template_measurement"
+NominalIntermediateMutation = False
max_materialize = 1000
queue name,shard_path,hist_dest,meta_dest,log_dir from {campaign / 'condor/arguments.txt'}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--runtime-payloads", type=Path, default=DEFAULT_RUNTIME_PAYLOADS)
    parser.add_argument(
        "--current-btag-efficiency",
        type=Path,
        default=DEFAULT_REPO / "analysis/hists/btageff2024.merged",
    )
    parser.add_argument(
        "--current-analysis-data",
        type=Path,
        default=DEFAULT_REPO / "analysis/data",
    )
    parser.add_argument("--refresh-payload-bundle", action="store_true")
    parser.add_argument("--target-events", type=int, default=2_400_000)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--record-workers", type=int, default=8)
    parser.add_argument(
        "--dataset-regex",
        help="optional regular expression selecting metadata dataset names",
    )
    args = parser.parse_args()
    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    if args.refresh_payload_bundle:
        destination = campaign / "bundles/photon_fake_template_payloads.tgz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        info = build_payload_bundle(
            args.runtime_payloads,
            destination,
            args.current_btag_efficiency,
            args.current_analysis_data,
        )
        manifest_path = campaign / "campaign_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            previous = manifest.get("payload_bundle")
            history = manifest.setdefault("payload_refresh_history", [])
            history.append(
                {
                    "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "previous": previous,
                    "replacement": info,
                }
            )
            manifest["payload_bundle"] = info
            write_json(manifest_path, manifest)
        print(json.dumps({"status": "payload_refreshed", "payload_bundle": info}, sort_keys=True))
        return 0
    if campaign.exists() and any(campaign.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty campaign: {campaign}")
    metadata = read_json(args.metadata)
    normalization = read_json(args.normalization)
    if normalization.get("status") != "complete":
        raise RuntimeError("normalization is incomplete")
    records, input_audit = build_records(metadata, normalization, args.dataset_regex)
    for directory in ("bundles", "shards", "outputs", "metadata", "logs", "condor/initialdir"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)
    worker = campaign / "bundles/photon_fake_template_worker.tgz"
    payloads = campaign / "bundles/photon_fake_template_payloads.tgz"
    worker_info = build_worker_bundle(repo, worker)
    payload_info = build_payload_bundle(
        args.runtime_payloads,
        payloads,
        args.current_btag_efficiency,
        args.current_analysis_data,
    )
    arguments: list[str] = []
    shard_counts = Counter()
    file_counts = Counter()
    estimated_events = Counter()
    for process in PROCESSES:
        packed = pack_process(records[process], args.target_events, args.max_files)
        for index, shard_records in enumerate(packed):
            name = f"{process.lower()}_{index:05d}"
            shard_path = campaign / "shards" / f"{name}.json"
            payload = {
                "schema_version": "photon_fake_template_balanced_shard_v1",
                "shard_id": name,
                "process": process,
                "record_digest": canonical_digest(shard_records),
                "records": shard_records,
            }
            write_json(shard_path, payload)
            output = campaign / "outputs" / process / f"{name}.json.gz"
            metadata_path = campaign / "metadata" / process / f"{name}.json"
            log_dir = campaign / "logs" / process
            output.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            arguments.append(f"template_{name} {shard_path} {output} {metadata_path} {log_dir}")
            shard_counts[process] += 1
            file_counts[process] += len(shard_records)
            estimated_events[process] += sum(int(item["expected_events"]) for item in shard_records)
    arguments_path = campaign / "condor/arguments.txt"
    arguments_path.write_text("\n".join(arguments) + "\n")
    proxy = repo / "analysis/proxy/x509up_u147757"
    python = repo / "condor/py38.tgz"
    for path in (proxy, python):
        if not path.is_file():
            raise FileNotFoundError(path)
    wrapper = campaign / "condor/run.sh"
    wrapper.write_text(wrapper_text(proxy.name))
    wrapper.chmod(0o755)
    (campaign / "condor/submit.sub").write_text(
        submit_text(campaign, python, worker, payloads, proxy, args.record_workers)
    )
    manifest = {
        "schema_version": "photon_fake_template_campaign_v1",
        "status": "prepared",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "worker_module": "autonomous_allhad.photon_fake_template_2024_worker",
        "selection_source": "real_subset_worker.py via intermediate_2024_worker.py",
        "nominal_intermediate_mutation": False,
        "source_metadata": str(args.metadata),
        "normalization": str(args.normalization),
        "input_audit": input_audit,
        "target_events_per_shard_estimate": args.target_events,
        "maximum_files_per_shard": args.max_files,
        "record_workers": args.record_workers,
        "jobs": len(arguments),
        "shard_counts": dict(shard_counts),
        "file_counts": dict(file_counts),
        "estimated_events": dict(estimated_events),
        "worker_bundle": worker_info,
        "payload_bundle": payload_info,
        "submission": {"schedd": "bigbird24.cern.ch", "attempted": False},
    }
    write_json(campaign / "campaign_manifest.json", manifest)
    print(json.dumps({"status": "prepared", "campaign": str(campaign), "jobs": len(arguments), "shards": dict(shard_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
