#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_SOURCE_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/"
    "intermediate_2024_fullselection_v3_lowdm_relaxed_20260724"
)
DEFAULT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/photon_fake_2024_local2400k_v3_20260726"
)
DEFAULT_MC_EVENT_INDEX = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/"
    "nominal_plots_2024_fullselection_v3_20260725/"
    "variation_prep/nanoaod_event_index_20260725.json.gz"
)
DEFAULT_DATA_SOURCE_INDEX = (
    DEFAULT_SOURCE_CAMPAIGN
    / "merged/data_balanced20/source_index.json"
)
DEFAULT_CONDOR_INITIALDIR = Path("/afs/cern.ch/user/t/taiwoo")
DEFAULT_PROCESSES = (
    "EGamma",
    "GJ",
    "QCD",
    "DY",
    "TT",
    "WtoLNu",
    "ST",
    "VV",
    "Zto2Nu",
)


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
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


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_eos(path: Path, label: str) -> None:
    text = str(path.absolute())
    if not text.startswith("/eos/user/"):
        raise ValueError(f"{label} must be under /eos/user: {text}")
    if "/../" in text:
        raise ValueError(f"{label} contains parent traversal: {text}")


def build_worker_bundle(
    repo: Path,
    destination: Path,
    worker_module: str = "photon_fake_2024_worker",
) -> dict[str, Any]:
    package = repo / "autonomous_allhad/autonomous_allhad"
    builder = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    files = sorted(path for path in package.glob("*.py") if path.is_file())
    required = package / f"{worker_module}.py"
    if required not in files:
        raise FileNotFoundError(required)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(
                path,
                arcname=f"autonomous_allhad/{path.name}",
                recursive=False,
            )
        archive.add(
            builder,
            arcname="workflow/build_flat_boosted_recoil_hists.py",
            recursive=False,
        )
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "files": [path.name for path in files]
        + ["workflow/build_flat_boosted_recoil_hists.py"],
    }


def load_filtered_records(
    source_bundle: Path,
    processes: set[str],
    mc_event_index: Path,
    data_source_index: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    mc_index_payload = read_json(mc_event_index)
    if mc_index_payload.get("status") != "complete":
        raise RuntimeError("MC NanoAOD event index is not complete")
    mc_index: dict[str, dict[str, Any]] = {}
    for item in mc_index_payload.get("records") or []:
        file_path = str(item.get("file_path") or "")
        if not file_path or file_path in mc_index:
            raise RuntimeError(f"invalid or duplicate MC event-index path: {file_path}")
        mc_index[file_path] = item

    data_index_payload = read_json(data_source_index)
    data_index: dict[str, dict[str, Any]] = {}
    for item in data_index_payload.get("sources") or []:
        shard = str(item.get("shard") or "")
        if not shard or shard in data_index:
            raise RuntimeError(f"invalid or duplicate data source-index shard: {shard}")
        data_index[shard] = item

    selected: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with tarfile.open(source_bundle, "r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".json")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot extract source shard {member.name}")
            shard = json.load(extracted)
            records = list(shard.get("records") or [])
            if not records:
                continue
            source_shard = str(shard.get("shard_id") or Path(member.name).stem)
            for record in records:
                process = str(record.get("process_group") or "unknown")
                if process not in processes:
                    continue
                if process == "EGamma":
                    if not bool(record.get("is_data")):
                        raise RuntimeError(
                            f"EGamma record is not data: {member.name}"
                        )
                elif (
                    not bool(record.get("is_background"))
                    or bool(record.get("is_data"))
                ):
                    raise RuntimeError(
                        f"selected MC fake record is not background MC: "
                        f"{member.name}"
                    )
                file_path = str(record.get("file_path") or "")
                if not file_path:
                    raise RuntimeError(f"source record has no file path: {member.name}")
                if file_path in seen_paths:
                    raise RuntimeError(
                        f"duplicate selected physical ROOT file: {file_path}"
                    )
                source_record_digest = canonical_digest(
                    {
                        "dataset": record.get("dataset"),
                        "file_path": file_path,
                        "process_group": process,
                        "source_shard": source_shard,
                    }
                )
                if bool(record.get("is_data")):
                    index_record = data_index.get(source_shard)
                    if index_record is None:
                        excluded.append(
                            {
                                "process": process,
                                "dataset": record.get("dataset"),
                                "file_path": file_path,
                                "source_shard": source_shard,
                                "source_record_digest": source_record_digest,
                                "event_index_status": (
                                    "missing_after_nominal_read_failure"
                                ),
                                "events": None,
                            }
                        )
                        continue
                    events = int(index_record.get("events_read") or 0)
                    index_status = "ready"
                else:
                    index_record = mc_index.get(file_path)
                    if index_record is None:
                        raise RuntimeError(
                            f"MC event count missing for {file_path}"
                        )
                    events = int(index_record.get("events") or 0)
                    index_status = str(index_record.get("status") or "missing")
                if index_status != "ready":
                    excluded.append(
                        {
                            "process": process,
                            "dataset": record.get("dataset"),
                            "file_path": file_path,
                            "source_shard": source_shard,
                            "source_record_digest": source_record_digest,
                            "event_index_status": index_status,
                            "events": events,
                        }
                    )
                    continue
                if events < 0:
                    raise RuntimeError(f"negative event count for {file_path}")
                augmented = dict(record)
                augmented.update(
                    {
                        "expected_events": events,
                        "source_shard": source_shard,
                        "source_record_digest": source_record_digest,
                    }
                )
                selected.setdefault(process, []).append(augmented)
                seen_paths.add(file_path)
    return selected, excluded


def pack_records(
    process: str,
    records: list[dict[str, Any]],
    target_events: int,
    max_files: int,
) -> list[list[dict[str, Any]]]:
    bins: list[dict[str, Any]] = []
    ordered = sorted(
        records,
        key=lambda record: (
            -int(record["expected_events"]),
            str(record["file_path"]),
        ),
    )
    for record in ordered:
        events = int(record["expected_events"])
        best_index: int | None = None
        best_key: tuple[int, int, int] | None = None
        for index, target in enumerate(bins):
            if len(target["records"]) >= max_files:
                continue
            combined = int(target["events"]) + events
            if combined > target_events:
                continue
            key = (
                target_events - combined,
                max_files - len(target["records"]) - 1,
                index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index is None:
            bins.append({"events": events, "records": [record]})
        else:
            bins[best_index]["events"] += events
            bins[best_index]["records"].append(record)
    packed = [item["records"] for item in bins]
    if sum(len(item) for item in packed) != len(records):
        raise RuntimeError(f"{process} packing lost source records")
    return packed


def segment_records(
    process: str,
    records: list[dict[str, Any]],
    target_events: int,
) -> list[dict[str, Any]]:
    """Split physical files into deterministic entry ranges no larger than target."""
    segmented: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["file_path"])):
        events = int(record["expected_events"])
        segment_count = max(1, (events + target_events - 1) // target_events)
        for segment_index in range(segment_count):
            entry_start = segment_index * target_events
            entry_stop = min(entry_start + target_events, events)
            segment_events = entry_stop - entry_start
            original_digest = str(record["source_record_digest"])
            segment = dict(record)
            segment.update(
                {
                    "entry_start": entry_start,
                    "entry_stop": entry_stop,
                    "segment_events": segment_events,
                    "expected_events": segment_events,
                    "physical_source_record_digest": original_digest,
                    "source_record_digest": canonical_digest(
                        {
                            "physical_source_record_digest": original_digest,
                            "entry_start": entry_start,
                            "entry_stop": entry_stop,
                        }
                    ),
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                }
            )
            segmented.append(segment)
    expected = sum(int(record["expected_events"]) for record in records)
    observed = sum(int(record["expected_events"]) for record in segmented)
    if observed != expected:
        raise RuntimeError(
            f"{process} entry segmentation changed event coverage: "
            f"{observed} vs {expected}"
        )
    if any(int(record["expected_events"]) > target_events for record in segmented):
        raise RuntimeError(f"{process} contains an oversized entry segment")
    return segmented


def write_balanced_shards(
    destination: Path,
    by_process: dict[str, list[dict[str, Any]]],
    target_events: int,
    max_files: int,
    materialize_files: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if materialize_files:
        destination.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    process_summary: dict[str, Any] = {}
    seen_digests: set[str] = set()
    for process in sorted(by_process):
        records = by_process[process]
        packed = pack_records(process, records, target_events, max_files)
        physical_files = {str(record["file_path"]) for record in records}
        process_events = 0
        process_segments = 0
        shard_events: list[int] = []
        shard_files: list[int] = []
        for index, shard_records in enumerate(packed):
            name = f"{process}_balanced_{index:05d}"
            expected_events = sum(
                int(record["expected_events"]) for record in shard_records
            )
            source_digests = [
                str(record["source_record_digest"]) for record in shard_records
            ]
            duplicates = seen_digests.intersection(source_digests)
            if duplicates:
                raise RuntimeError(
                    f"source records occur in multiple balanced shards: "
                    f"{sorted(duplicates)[:3]}"
                )
            seen_digests.update(source_digests)
            payload = {
                "schema_version": "photon_fake_2024_balanced_shard_v1",
                "shard_id": name,
                "process_group": process,
                "record_group": "data" if process == "EGamma" else "mc_background",
                "record_digest": canonical_digest(source_digests),
                "source_record_digests": source_digests,
                "expected_events": expected_events,
                "expected_files": len(shard_records),
                "records": shard_records,
            }
            target = destination / process / f"{name}.json"
            item = {
                "name": name,
                "process": process,
                "path": str(target) if materialize_files else None,
                "record_digest": payload["record_digest"],
                "records": len(shard_records),
                "expected_events": expected_events,
                "payload": payload,
            }
            if materialize_files:
                write_json(target, payload)
                item["bytes"] = target.stat().st_size
                item["sha256"] = sha256(target)
            selected.append(item)
            process_events += expected_events
            process_segments += len(shard_records)
            shard_events.append(expected_events)
            shard_files.append(len(shard_records))
        expected_events = sum(int(record["expected_events"]) for record in records)
        if process_events != expected_events or process_segments != len(records):
            raise RuntimeError(f"{process} balanced-shard coverage mismatch")
        process_summary[process] = {
            "source_files": len(physical_files),
            "entry_segments": process_segments,
            "events": process_events,
            "shards": len(packed),
            "min_events_per_shard": min(shard_events),
            "max_events_per_shard": max(shard_events),
            "min_files_per_shard": min(shard_files),
            "max_files_per_shard": max(shard_files),
        }
    return selected, process_summary


def write_local_job_bundle(
    destination: Path,
    selected: list[dict[str, Any]],
    campaign: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
            handle.write(
                '{"schema_version":"photon_fake_2024_local_jobs_v1","jobs":['
            )
            for index, item in enumerate(selected):
                if index:
                    handle.write(",")
                process = str(item["process"])
                name = str(item["name"])
                job = {
                    "name": f"fake_{name}",
                    "process": process,
                    "shard_basename": f"{name}.json",
                    "histogram": str(
                        campaign / "outputs" / process / f"{name}.json.gz"
                    ),
                    "metadata": str(
                        campaign / "metadata" / process / f"{name}.json"
                    ),
                    "log_dir": str(campaign / "logs" / process),
                    "shard": item["payload"],
                }
                json.dump(
                    job,
                    handle,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            handle.write("]}\n")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "jobs": len(selected),
    }


def wrapper_text(
    proxy_name: str,
    worker_module: str = "photon_fake_2024_worker",
) -> str:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
NAME="@D@{{1:?missing name}}"
SHARD_BASENAME="@D@{{2:?missing shard basename}}"
HIST_DEST="@D@{{3:?missing histogram destination}}"
META_DEST="@D@{{4:?missing metadata destination}}"
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
tar -xzf photon_fake_2024_worker.tgz
tar -xzf photon_fake_2024_payloads.tgz
PY="@D@WORKDIR/bin/python3"
[ -x "@D@PY" ] || PY="@D@WORKDIR/bin/python"
[ -x "@D@PY" ] || PY="@D@WORKDIR/py38/bin/python"
test -x "@D@PY"
export PATH="@D@(dirname "@D@PY"):@D@PATH"
export LD_LIBRARY_PATH="@D@WORKDIR/lib:@D@WORKDIR/py38/lib:@D@{{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="@D@WORKDIR"
"@D@PY" -u -m autonomous_allhad.{worker_module} \
  --shard "@D@WORKDIR/@D@SHARD_BASENAME" \
  --output "@D@WORKDIR/out.json.gz" \
  --metadata-output "@D@WORKDIR/out.meta.json" \
  --chunk-size 20000 \
  --record-workers "@D@{{REQUEST_CPUS:-1}}"
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
shard = json.loads(Path(os.environ.get("SHARD_BASENAME", "")).read_text())
with gzip.open("out.json.gz", "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
summary = metadata.get("summary") or {{}}
records = list(shard.get("records") or [])
expected_paths = sorted(str(record["file_path"]) for record in records)
observed_paths = sorted(
    str(record["file_path"]) for record in summary.get("file_records") or []
)
expected_events = sum(int(record["expected_events"]) for record in records)
errors = []
if metadata.get("status") != "complete" or payload.get("status") != "complete":
    errors.append("worker output is not complete")
if int(summary.get("files_attempted") or 0) != len(records):
    errors.append("attempted file count differs from balanced shard")
if int(summary.get("files_processed") or 0) != len(records):
    errors.append("not every input file was processed")
if int(summary.get("events_read") or 0) != expected_events:
    errors.append("processed event count differs from indexed balanced-shard total")
if observed_paths != expected_paths:
    errors.append("processed ROOT-file coverage differs from balanced shard")
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
    python_bundle: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    proxy: Path,
    request_cpus: int,
    request_memory_mb: int,
    request_disk_mb: int,
    job_flavour: str,
) -> str:
    return f"""universe = vanilla
initialdir = {DEFAULT_CONDOR_INITIALDIR}
executable = {campaign / 'condor/run_photon_fake_2024.sh'}
arguments = $(name) $(shard_basename) $(hist_out) $(meta_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {python_bundle}, {worker_bundle}, {payload_bundle}, {proxy}, $(shard_path)
transfer_output_files = ""
output = $(log_dir)/$(name).out
error = $(log_dir)/$(name).err
log = {campaign / 'condor/campaign.log'}
request_cpus = {request_cpus}
request_memory = {request_memory_mb}MB
request_disk = {request_disk_mb}MB
+JobFlavour = "{job_flavour}"
+CampaignName = "{campaign.name}"
+PhysicsTask = "photon_fake_measurement"
+NominalIntermediateMutation = False
max_materialize = 1000
queue name,shard_path,shard_basename,hist_out,meta_out,log_dir from {campaign / 'condor/arguments.txt'}
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the non-destructive 2024 photon fake sidecar campaign."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=DEFAULT_SOURCE_CAMPAIGN,
    )
    parser.add_argument(
        "--mc-event-index",
        type=Path,
        default=DEFAULT_MC_EVENT_INDEX,
    )
    parser.add_argument(
        "--data-source-index",
        type=Path,
        default=DEFAULT_DATA_SOURCE_INDEX,
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--worker-module",
        default="photon_fake_2024_worker",
        choices=("photon_fake_2024_worker", "photon_fake_template_2024_worker"),
    )
    parser.add_argument(
        "--processes",
        nargs="+",
        default=list(DEFAULT_PROCESSES),
    )
    parser.add_argument("--target-events-per-shard", type=int, default=2_400_000)
    parser.add_argument("--max-files-per-shard", type=int, default=128)
    parser.add_argument("--record-workers", type=int, default=8)
    parser.add_argument("--request-memory-mb", type=int, default=40_000)
    parser.add_argument("--request-disk-mb", type=int, default=20_000)
    parser.add_argument("--job-flavour", default="workday")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Store all logical shard definitions in one compressed manifest.",
    )
    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument("--submit", action="store_true")
    submit_group.add_argument("--submit-existing", action="store_true")
    submit_group.add_argument("--repair-held-initialdir", action="store_true")
    args = parser.parse_args()
    if args.target_events_per_shard <= 0:
        parser.error("--target-events-per-shard must be positive")
    if args.max_files_per_shard <= 1:
        parser.error("--max-files-per-shard must exceed one")
    if args.record_workers <= 1:
        parser.error("--record-workers must exceed one for a balanced campaign")
    if args.request_memory_mb <= 0 or args.request_disk_mb <= 0:
        parser.error("memory and disk requests must be positive")
    if args.local_only and (
        args.submit or args.submit_existing or args.repair_held_initialdir
    ):
        parser.error("--local-only cannot be combined with Condor actions")

    repo = args.repo.absolute()
    source_campaign = args.source_campaign.absolute()
    mc_event_index = args.mc_event_index.absolute()
    data_source_index = args.data_source_index.absolute()
    campaign = args.campaign.absolute()
    for path, label in (
        (repo, "repository"),
        (source_campaign, "source campaign"),
        (mc_event_index, "MC event index"),
        (data_source_index, "data source index"),
        (campaign, "campaign"),
    ):
        require_eos(path, label)
    if args.repair_held_initialdir:
        manifest_path = campaign / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        submission = manifest.get("submission") or {}
        cluster_id = int(submission.get("cluster_id") or 0)
        expected_jobs = int(manifest.get("jobs") or 0)
        if cluster_id <= 0 or expected_jobs <= 0:
            raise RuntimeError("campaign manifest has no submitted Condor cluster")
        queue = subprocess.run(
            [
                "condor_q",
                str(cluster_id),
                "-af",
                "ProcId",
                "JobStatus",
                "HoldReasonCode",
                "HoldReasonSubCode",
                "HoldReason",
            ],
            text=True,
            capture_output=True,
            timeout=180,
        )
        if queue.returncode != 0:
            raise RuntimeError(f"condor_q failed: {queue.stderr}")
        rows = [line.split(maxsplit=4) for line in queue.stdout.splitlines()]
        invalid = [
            row
            for row in rows
            if len(row) < 4
            or row[1] != "5"
            or row[2] != "14"
            or row[3] != "2"
        ]
        if len(rows) != expected_jobs or invalid:
            raise RuntimeError(
                "refusing Iwd repair because the cluster is not uniformly held "
                f"for inaccessible initialdir: rows={len(rows)}, "
                f"expected={expected_jobs}, invalid={invalid[:3]}"
            )
        if not DEFAULT_CONDOR_INITIALDIR.is_dir():
            raise FileNotFoundError(DEFAULT_CONDOR_INITIALDIR)
        submit_path = Path(str((manifest.get("condor") or {}).get("submit") or ""))
        submit_body = submit_path.read_text()
        repaired_body, replacements = re.subn(
            r"(?m)^initialdir\s*=.*$",
            f"initialdir = {DEFAULT_CONDOR_INITIALDIR}",
            submit_body,
            count=1,
        )
        if replacements != 1:
            raise RuntimeError("prepared submit file has no unique initialdir line")
        partial = submit_path.with_name(f"{submit_path.name}.partial.{os.getpid()}")
        try:
            partial.write_text(repaired_body)
            os.replace(partial, submit_path)
        finally:
            partial.unlink(missing_ok=True)
        qedit = subprocess.run(
            [
                "condor_qedit",
                str(cluster_id),
                "Iwd",
                f'"{DEFAULT_CONDOR_INITIALDIR}"',
            ],
            text=True,
            capture_output=True,
        )
        if qedit.returncode != 0:
            raise RuntimeError(
                f"condor_qedit failed: {qedit.stderr or qedit.stdout}"
            )
        release = subprocess.run(
            ["condor_release", str(cluster_id)],
            text=True,
            capture_output=True,
        )
        recovery = {
            "type": "uniform_initialdir_hold_repair",
            "cluster_id": cluster_id,
            "held_jobs_verified": len(rows),
            "old_initialdir": str(campaign / "condor"),
            "new_initialdir": str(DEFAULT_CONDOR_INITIALDIR),
            "qedit": {
                "exit_status": qedit.returncode,
                "stdout": qedit.stdout,
                "stderr": qedit.stderr,
            },
            "release": {
                "exit_status": release.returncode,
                "stdout": release.stdout,
                "stderr": release.stderr,
            },
            "repaired_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        manifest.setdefault("recoveries", []).append(recovery)
        manifest["condor"]["initialdir"] = str(DEFAULT_CONDOR_INITIALDIR)
        manifest["status"] = (
            "submitted" if release.returncode == 0 else "release_failed"
        )
        write_json(manifest_path, manifest)
        if release.returncode != 0:
            raise RuntimeError(
                f"condor_release failed: {release.stderr or release.stdout}"
            )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "cluster_id": cluster_id,
                    "released_jobs": len(rows),
                    "initialdir": str(DEFAULT_CONDOR_INITIALDIR),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.submit_existing:
        manifest_path = campaign / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        if manifest.get("status") != "prepared":
            raise RuntimeError(
                f"campaign is not in prepared state: {manifest.get('status')}"
            )
        if manifest.get("nominal_intermediate_mutation") is not False:
            raise RuntimeError("campaign does not explicitly preserve nominal outputs")
        submit_path = Path(str((manifest.get("condor") or {}).get("submit") or ""))
        arguments_path = Path(
            str((manifest.get("condor") or {}).get("arguments") or "")
        )
        if not submit_path.is_file() or not arguments_path.is_file():
            raise FileNotFoundError("prepared Condor submit or arguments file is missing")
        argument_count = sum(
            1 for line in arguments_path.read_text().splitlines() if line.strip()
        )
        if argument_count != int(manifest.get("jobs") or -1):
            raise RuntimeError(
                f"Condor argument count mismatch: {argument_count} vs "
                f"{manifest.get('jobs')}"
            )
        result = subprocess.run(
            ["condor_submit", str(submit_path)],
            text=True,
            capture_output=True,
        )
        cluster_match = re.search(r"cluster\s+(\d+)", result.stdout, re.IGNORECASE)
        manifest["submission"] = {
            "attempted": True,
            "exit_status": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cluster_id": (
                int(cluster_match.group(1)) if cluster_match is not None else None
            ),
            "submitted_jobs": argument_count if result.returncode == 0 else 0,
            "submitted_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        manifest["status"] = (
            "submitted" if result.returncode == 0 else "submit_failed"
        )
        write_json(manifest_path, manifest)
        if result.returncode != 0:
            raise RuntimeError(
                f"condor_submit failed: {result.stderr or result.stdout}"
            )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "campaign": str(campaign),
                    "cluster_id": manifest["submission"]["cluster_id"],
                    "submitted_jobs": argument_count,
                },
                sort_keys=True,
            )
        )
        return 0
    if campaign.exists() and any(campaign.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty campaign: {campaign}")

    source_bundle = source_campaign / "bundles/fullselection_shards.tgz"
    python_bundle = repo / "condor/py38.tgz"
    proxy = repo / "analysis/proxy/x509up_u147757"
    payload_bundle = (
        repo
        / "autonomous_allhad/workflow/"
        "shape_hists_2024_fullselection_v7_condorpairs_20260725/"
        "bundles/shape_hist_2024_payloads.tgz"
    )
    for path in (
        source_bundle,
        mc_event_index,
        data_source_index,
        python_bundle,
        proxy,
        payload_bundle,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    for directory in (
        campaign / "bundles",
        campaign / "outputs",
        campaign / "metadata",
        campaign / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.local_only:
        for directory in (campaign / "shards", campaign / "condor"):
            directory.mkdir(parents=True, exist_ok=True)
    worker_bundle = campaign / "bundles/photon_fake_2024_worker.tgz"
    worker_info = build_worker_bundle(repo, worker_bundle, args.worker_module)
    by_process, excluded = load_filtered_records(
        source_bundle,
        set(args.processes),
        mc_event_index,
        data_source_index,
    )
    record_counts = {
        process: len(records) for process, records in by_process.items()
    }
    segmented_by_process = {
        process: segment_records(
            process,
            records,
            args.target_events_per_shard,
        )
        for process, records in by_process.items()
    }
    selected, process_summary = write_balanced_shards(
        campaign / "shards",
        segmented_by_process,
        args.target_events_per_shard,
        args.max_files_per_shard,
        materialize_files=not args.local_only,
    )
    if not selected:
        raise RuntimeError("no source shards matched the requested processes")
    segment_counts = {
        process: int(summary["entry_segments"])
        for process, summary in process_summary.items()
    }

    shard_counts: Counter[str] = Counter()
    for item in selected:
        shard_counts[item["process"]] += 1
    for process in sorted(by_process):
        (campaign / "outputs" / process).mkdir(parents=True, exist_ok=True)
        (campaign / "metadata" / process).mkdir(parents=True, exist_ok=True)
        (campaign / "logs" / process).mkdir(parents=True, exist_ok=True)
    local_job_bundle = None
    arguments = None
    wrapper = None
    submit = None
    if args.local_only:
        local_job_bundle = write_local_job_bundle(
            campaign / "bundles/photon_fake_2024_local_jobs.json.gz",
            selected,
            campaign,
        )
    else:
        argument_lines: list[str] = []
        for item in selected:
            process = item["process"]
            name = item["name"]
            histogram = campaign / "outputs" / process / f"{name}.json.gz"
            metadata = campaign / "metadata" / process / f"{name}.json"
            log_dir = campaign / "logs" / process
            argument_lines.append(
                " ".join(
                    [
                        f"fake_{name}",
                        str(item["path"]),
                        Path(str(item["path"])).name,
                        str(histogram),
                        str(metadata),
                        str(log_dir),
                    ]
                )
            )
        arguments = campaign / "condor/arguments.txt"
        arguments.write_text("\n".join(argument_lines) + "\n")
        wrapper = campaign / "condor/run_photon_fake_2024.sh"
        wrapper.write_text(wrapper_text(proxy.name, args.worker_module))
        wrapper.chmod(0o755)
        submit = campaign / "condor/submit.sub"
        submit.write_text(
            submit_text(
                campaign,
                python_bundle,
                worker_bundle,
                payload_bundle,
                proxy,
                args.record_workers,
                args.request_memory_mb,
                args.request_disk_mb,
                args.job_flavour,
            )
        )

    manifest = {
        "schema_version": "photon_fake_2024_balanced_campaign_v2",
        "status": "prepared_local" if args.local_only else "prepared",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "repo": str(repo),
        "source_campaign": str(source_campaign),
        "source_shard_bundle": {
            "path": str(source_bundle),
            "size": source_bundle.stat().st_size,
            "sha256": sha256(source_bundle),
        },
        "event_indexes": {
            "mc": {
                "path": str(mc_event_index),
                "size": mc_event_index.stat().st_size,
                "sha256": sha256(mc_event_index),
            },
            "data": {
                "path": str(data_source_index),
                "size": data_source_index.stat().st_size,
                "sha256": sha256(data_source_index),
            },
        },
        "selection_source": "real_subset_worker.py via intermediate_2024_worker.py",
        "nominal_intermediate_mutation": False,
        "worker_module": args.worker_module,
        "requested_processes": list(args.processes),
        "shard_counts": dict(sorted(shard_counts.items())),
        "record_counts": record_counts,
        "entry_segment_counts": segment_counts,
        "process_summary": process_summary,
        "excluded_indexed_records": excluded,
        "jobs": len(selected),
        "local_job_bundle": local_job_bundle,
        "packing": {
            "basis": "indexed Events entries",
            "algorithm": (
                "deterministic physical-file entry segmentation followed by "
                "process-pure best-fit decreasing"
            ),
            "target_events_per_shard": args.target_events_per_shard,
            "max_files_per_shard": args.max_files_per_shard,
            "record_workers_per_job": args.record_workers,
            "one_root_file_per_shard": False,
            "physical_file_splitting": True,
            "entry_range_segment_max_events": args.target_events_per_shard,
        },
        "bundles": {
            "python": {
                "path": str(python_bundle),
                "size": python_bundle.stat().st_size,
                "sha256": sha256(python_bundle),
            },
            "worker": worker_info,
            "payload": {
                "path": str(payload_bundle),
                "size": payload_bundle.stat().st_size,
                "sha256": sha256(payload_bundle),
            },
            "proxy": {
                "path": str(proxy),
                "size": proxy.stat().st_size,
            },
        },
        "condor": (
            None
            if args.local_only
            else {
                "arguments": str(arguments),
                "wrapper": str(wrapper),
                "submit": str(submit),
                "job_flavour": args.job_flavour,
                "initialdir": str(DEFAULT_CONDOR_INITIALDIR),
                "request_cpus": args.record_workers,
                "request_memory_mb": args.request_memory_mb,
                "request_disk_mb": args.request_disk_mb,
                "max_materialize": 1000,
            }
        ),
        "output_policy": {
            "histograms": str(campaign / "outputs/<process>/<shard>.json.gz"),
            "metadata": str(campaign / "metadata/<process>/<shard>.json"),
            "nominal_outputs_touched": False,
        },
    }
    write_json(campaign / "manifest.json", manifest)
    if args.submit:
        result = subprocess.run(
            ["condor_submit", str(submit)],
            text=True,
            capture_output=True,
        )
        manifest["submission"] = {
            "attempted": True,
            "exit_status": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "submitted_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        manifest["status"] = "submitted" if result.returncode == 0 else "submit_failed"
        write_json(campaign / "manifest.json", manifest)
        if result.returncode != 0:
            raise RuntimeError(
                f"condor_submit failed: {result.stderr or result.stdout}"
            )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "campaign": str(campaign),
                "jobs": len(selected),
                "shard_counts": manifest["shard_counts"],
                "record_counts": record_counts,
                "entry_segment_counts": segment_counts,
                "submitted": bool(args.submit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
