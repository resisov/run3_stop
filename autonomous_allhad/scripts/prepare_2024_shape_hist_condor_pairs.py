#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from autonomous_allhad.shape_histogram_2024_worker import FINAL_SHAPE_NUISANCES

from prepare_2024_shape_hist_campaign_fullselection import (
    build_payload_bundle,
    build_worker_bundle,
    require_eos,
    sha256,
    write_json,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_SOURCE_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v5_local1_20260725"
)
DEFAULT_LOCAL_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v6_localpairs_20260725"
)
DEFAULT_EVENT_INDEX = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/nominal_plots_2024_fullselection_v3_20260725"
    / "variation_prep/nanoaod_event_index_20260725.json.gz"
)
DEFAULT_BENCHMARK = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/nominal_plots_2024_fullselection_v3_20260725"
    / "variation_prep/local_pair_benchmark_20260725.json"
)
DEFAULT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v8_condorpairs_20260725"
)


def read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source_records(source_campaign: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_payload(source_campaign / "manifest.json")
    bundle = Path((manifest.get("shard_bundle") or {}).get("path") or "")
    if not bundle.is_file():
        raise FileNotFoundError(f"source shard bundle is missing: {bundle}")
    if sha256(bundle) != str((manifest.get("shard_bundle") or {}).get("sha256") or ""):
        raise RuntimeError("source shard bundle checksum mismatch")
    records: dict[str, dict[str, Any]] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile()
                and Path(member.name).name.startswith("shape_input_")
                and Path(member.name).suffix == ".json"
            ),
            key=lambda member: member.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot extract {member.name}")
            payload = json.load(extracted)
            rows = list(payload.get("records") or [])
            if len(rows) != 1:
                raise RuntimeError(f"expected one record in {member.name}")
            digest = str(payload.get("record_digest") or "")
            if not digest or digest in records:
                raise RuntimeError(f"invalid or duplicate source digest in {member.name}")
            records[digest] = rows[0]
    return manifest, records


def freeze_local_seed(local_campaign: Path, destination: Path) -> dict[str, Any]:
    status_path = local_campaign / "local_status.json"
    checkpoint_path = local_campaign / "checkpoint.json"
    status = read_payload(status_path)
    if status.get("status") != "stopped":
        raise RuntimeError(
            "local pair campaign must be stopped cleanly before Condor preparation: "
            f"{status.get('status')}"
        )
    checkpoint = read_payload(checkpoint_path)
    generation = str(checkpoint.get("generation") or "")
    source_dir = local_campaign / "checkpoints" / generation
    destination.mkdir(parents=True, exist_ok=False)
    coverage: set[str] | None = None
    files: list[dict[str, Any]] = []
    for nuisance in FINAL_SHAPE_NUISANCES:
        histogram = source_dir / f"{nuisance}.json.gz"
        metadata = source_dir / f"{nuisance}.meta.json"
        payload = read_payload(histogram)
        sidecar = read_payload(metadata)
        observed = set((payload.get("summary") or {}).get("source_record_digests") or [])
        if coverage is None:
            coverage = observed
        elif coverage != observed:
            raise RuntimeError(f"local checkpoint coverage differs for {nuisance}")
        if payload.get("nuisance") != nuisance:
            raise RuntimeError(f"local checkpoint nuisance mismatch: {nuisance}")
        if sha256(histogram) != sidecar.get("histogram_sha256"):
            raise RuntimeError(f"local checkpoint checksum mismatch: {nuisance}")
        target_histogram = destination / histogram.name
        target_metadata = destination / metadata.name
        shutil.copy2(histogram, target_histogram)
        shutil.copy2(metadata, target_metadata)
        files.append(
            {
                "nuisance": nuisance,
                "histogram": str(target_histogram),
                "metadata": str(target_metadata),
                "histogram_sha256": sha256(target_histogram),
                "metadata_sha256": sha256(target_metadata),
            }
        )
    coverage = coverage or set()
    completed = int(checkpoint.get("completed_sources") or 0)
    if len(coverage) != completed:
        raise RuntimeError(
            f"local checkpoint coverage mismatch: {len(coverage)} vs {completed}"
        )
    return {
        "source_campaign": str(local_campaign),
        "source_status": status,
        "source_checkpoint": checkpoint,
        "completed_source_records": completed,
        "source_record_digests": sorted(coverage),
        "coverage_sha256": canonical_digest(sorted(coverage)),
        "files": files,
    }


def split_record(
    record: dict[str, Any],
    index_record: dict[str, Any],
    segment_events: int,
) -> list[dict[str, Any]]:
    events = int(index_record.get("events") or 0)
    if events < 0:
        raise RuntimeError("negative indexed event count")
    pieces = max(1, int(math.ceil(events / segment_events)))
    if pieces > 8:
        raise RuntimeError(
            f"source requires {pieces} segments, exceeding the validated per-job limit"
        )
    output = []
    start = 0
    for index in range(pieces):
        stop = min(events, start + segment_events)
        segment = dict(record)
        segment.update(
            {
                "source_record_digest": str(index_record["record_digest"]),
                "source_name": str(index_record["source_name"]),
                "entry_start": start,
                "entry_stop": stop,
                "segment_events": stop - start,
                "segment_index": index,
                "segment_count": pieces,
                "segment_id": (
                    f"{index_record['record_digest']}:{start}:{stop}"
                ),
            }
        )
        output.append(segment)
        start = stop
    if start != events:
        raise RuntimeError("entry-range split does not cover the indexed event count")
    return output


def pack_process_sources(
    process: str,
    source_groups: list[list[dict[str, Any]]],
    records_per_job: int,
    target_events_per_job: int,
) -> list[list[dict[str, Any]]]:
    bins: list[dict[str, Any]] = []
    ordered = sorted(
        source_groups,
        key=lambda group: (
            -len(group),
            -sum(int(item["segment_events"]) for item in group),
            str(group[0]["source_record_digest"]),
        ),
    )
    for group in ordered:
        slots = len(group)
        events = sum(int(item["segment_events"]) for item in group)
        if slots > records_per_job or events > target_events_per_job:
            raise RuntimeError(
                f"source group cannot fit one {process} partition: "
                f"slots={slots}, events={events}"
            )
        best_index = None
        best_key = None
        for index, target in enumerate(bins):
            if target["slots"] + slots > records_per_job:
                continue
            if target["events"] + events > target_events_per_job:
                continue
            key = (
                target_events_per_job - target["events"] - events,
                records_per_job - target["slots"] - slots,
                index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index is None:
            bins.append(
                {
                    "slots": slots,
                    "events": events,
                    "records": list(group),
                }
            )
        else:
            target = bins[best_index]
            target["slots"] += slots
            target["events"] += events
            target["records"].extend(group)
    return [item["records"] for item in bins]


def wrapper_text(proxy_name: str, expected_btag_sha256: str) -> str:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
NAME="@D@{{1:?missing name}}"
SHARD_EOS="@D@{{2:?missing shard EOS path}}"
SHARD_SHA256="@D@{{3:?missing shard checksum}}"
NUISANCE="@D@{{4:?missing nuisance}}"
HIST_DEST="@D@{{5:?missing histogram EOS destination}}"
META_DEST="@D@{{6:?missing metadata EOS destination}}"
EXPECTED_EVENTS="@D@{{7:?missing expected events}}"
EXPECTED_SEGMENTS="@D@{{8:?missing expected segments}}"
WORKDIR="@D@{{_CONDOR_SCRATCH_DIR:-@D@PWD}}"
cd "@D@WORKDIR"
for path in "@D@SHARD_EOS" "@D@HIST_DEST" "@D@META_DEST"; do
  case "@D@path" in
    /eos/user/t/taiwoo/*) ;;
    *) echo "refusing non-EOS path: @D@path" >&2; exit 64 ;;
  esac
done
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
export AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE=1
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
PY="@D@WORKDIR/bin/python3"
[ -x "@D@PY" ] || PY="@D@WORKDIR/bin/python"
[ -x "@D@PY" ] || PY="@D@WORKDIR/py38/bin/python"
test -x "@D@PY"
export PATH="@D@(dirname "@D@PY"):@D@PATH"
export LD_LIBRARY_PATH="@D@WORKDIR/lib:@D@WORKDIR/py38/lib:@D@{{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="@D@WORKDIR"
XRDCOPY="@D@(command -v xrdcp)"
XRDFS="@D@(command -v xrdfs)"
"@D@XRDCOPY" -f --nopbar "root://eosuser.cern.ch/@D@SHARD_EOS" shard.json
export SHARD_SHA256 EXPECTED_EVENTS EXPECTED_SEGMENTS
"@D@PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from autonomous_allhad.real_subset_worker import open_root_with_xrd_fallback

path = Path("shard.json")
if hashlib.sha256(path.read_bytes()).hexdigest() != os.environ["SHARD_SHA256"]:
    raise SystemExit("partition shard checksum mismatch")
shard = json.loads(path.read_text())
records = list(shard.get("records") or [])
if len(records) != int(os.environ["EXPECTED_SEGMENTS"]):
    raise SystemExit("partition segment count mismatch")
if sum(int(record.get("segment_events") or 0) for record in records) != int(
    os.environ["EXPECTED_EVENTS"]
):
    raise SystemExit("partition expected-event count mismatch")
for file_path in dict.fromkeys(str(record["file_path"]) for record in records):
    root, access = open_root_with_xrd_fallback(file_path, timeout=60)
    root.close()
    cache = Path(str(access.get("cache_path") or ""))
    if not cache.is_file() or cache.stat().st_size == 0:
        raise SystemExit(f"prefetched cache is invalid: {{file_path}}")
PY
BTagFile="@D@WORKDIR/analysis/hists/btageff2024.merged"
test -s "@D@BTagFile"
test "@D@(sha256sum "@D@BTagFile" | awk '{{print @D@1}}')" = "{expected_btag_sha256}"
set +e
"@D@PY" -u -m autonomous_allhad.shape_histogram_2024_worker \
  --shard "@D@WORKDIR/shard.json" \
  --output "@D@WORKDIR/out.json.gz" \
  --metadata-output "@D@WORKDIR/out.meta.json" \
  --variation-group "@D@NUISANCE" \
  --chunk-size 5000 \
  --record-workers "@D@{{REQUEST_CPUS:-1}}"
WORKER_STATUS="@D@?"
set -e
test -s out.meta.json
test -s out.json.gz
export WORKER_STATUS NUISANCE HIST_DEST META_DEST
"@D@PY" - <<'PY'
import gzip
import hashlib
import json
import os
from pathlib import Path

worker_status = int(os.environ["WORKER_STATUS"])
metadata = json.loads(Path("out.meta.json").read_text())
shard = json.loads(Path("shard.json").read_text())
with gzip.open("out.json.gz", "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
summary = metadata.get("summary") or {{}}
nuisance = os.environ["NUISANCE"]
pair = [f"{{nuisance}}Up", f"{{nuisance}}Down"]
records = list(shard.get("records") or [])
expected_ranges = sorted(
    (
        str(record["file_path"]),
        int(record["entry_start"]),
        int(record["entry_stop"]),
        str(record["segment_id"]),
    )
    for record in records
)
observed_ranges = sorted(
    (
        str(record["file_path"]),
        int(record["entry_start"]),
        int(record["entry_stop"]),
        str(record["segment_id"]),
    )
    for record in summary.get("file_records") or []
)
errors = []
if worker_status != 0:
    errors.append(f"worker exit status {{worker_status}}")
if metadata.get("status") != "complete" or payload.get("status") != "complete":
    errors.append("non-complete output status")
if list(payload.get("variations") or []) != pair:
    errors.append("payload does not contain the exact Up/Down pair")
if list(summary.get("variations") or []) != pair:
    errors.append("summary does not contain the exact Up/Down pair")
if int(metadata.get("variation_count") or 0) != 2:
    errors.append("variation count is not two")
if int(summary.get("files_attempted") or 0) != len(records):
    errors.append("attempted segment count mismatch")
if int(summary.get("files_processed") or 0) != len(records):
    errors.append("not every attempted segment was processed")
if int(summary.get("events_read") or 0) != int(os.environ["EXPECTED_EVENTS"]):
    errors.append("processed event count mismatch")
if summary.get("bad_files"):
    errors.append("bad_files is non-empty")
if expected_ranges != observed_ranges:
    errors.append("processed entry-range coverage mismatch")
expected_sections = {{
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
}}
if set((payload.get("output_policy") or {{}}).get("sections") or []) != expected_sections:
    errors.append("histogram section mismatch")
actual_sha256 = hashlib.sha256(Path("out.json.gz").read_bytes()).hexdigest()
if actual_sha256 != metadata.get("histogram_sha256"):
    errors.append("histogram checksum mismatch")
btag = summary.get("btag_sf_status") or {{}}
if set(btag) != set(pair) or any(not (item or {{}}).get("applied") for item in btag.values()):
    errors.append("btag efficiency was not applied for both directions")
if errors:
    raise SystemExit("; ".join(errors))
metadata.update(
    {{
        "nuisance": nuisance,
        "partition_id": shard.get("partition_id"),
        "partition_digest": shard.get("record_digest"),
        "partition_shard_sha256": os.environ["SHARD_SHA256"],
        "source_record_digests": shard.get("source_record_digests"),
        "expected_events": int(os.environ["EXPECTED_EVENTS"]),
        "expected_segments": int(os.environ["EXPECTED_SEGMENTS"]),
        "btag_efficiency_sha256": "{expected_btag_sha256}",
        "histogram_file": os.environ["HIST_DEST"],
    }}
)
Path("out.meta.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\\n"
)
PY
staged=0
for attempt in 1 2 3 4 5; do
  if "@D@XRDCOPY" -f --nopbar out.json.gz "root://eosuser.cern.ch/@D@HIST_DEST" &&
     "@D@XRDCOPY" -f --nopbar out.meta.json "root://eosuser.cern.ch/@D@META_DEST" &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@HIST_DEST" >/dev/null &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "@D@((attempt * 10))"
done
test "@D@staged" -eq 1
test "@D@WORKER_STATUS" -eq 0
echo "completed @D@NAME nuisance=@D@NUISANCE events=@D@EXPECTED_EVENTS segments=@D@EXPECTED_SEGMENTS"
"""
    return text.replace("@D@", "$")


def submit_text(
    *,
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py38: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    proxy: Path,
    initialdir: Path,
    request_memory_mb: int,
    request_disk_mb: int,
    request_cpus: int,
    job_flavour: str,
    campaign_name: str,
    campaign_fingerprint: str,
    segment_events: int,
) -> str:
    return f"""universe = vanilla
initialdir = {initialdir}
executable = {wrapper}
arguments = $(name) $(shard_path) $(shard_sha256) $(nuisance) $(hist_out) $(meta_out) $(expected_events) $(expected_segments)
getenv = False
environment = "REQUEST_CPUS={request_cpus}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker_bundle}, {payload_bundle}, {proxy}
transfer_output_files = ""
output = $(log_dir)/$(name).out
error = $(log_dir)/$(name).err
log = $(log_dir)/campaign.log
request_cpus = {request_cpus}
request_memory = {request_memory_mb}MB
request_disk = {request_disk_mb}MB
+JobFlavour = "{job_flavour}"
+CampaignName = "{campaign_name}"
+CampaignFingerprint = "{campaign_fingerprint}"
+NuisancePair = "$(nuisance)"
+ExpectedEvents = $(expected_events)
+ExpectedSegments = $(expected_segments)
+TargetWalltimeSeconds = 3600
+MaxSegmentEvents = {segment_events}
+NanoAODReadPolicy = "once_per_physical_file_per_nuisance_pair"
+OutputPolicy = "one_Up_Down_pair"
queue name,shard_path,shard_sha256,nuisance,hist_out,meta_out,log_dir,expected_events,expected_segments from {arguments}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare process-pure, event-balanced 2024 object-variation "
            "HTCondor jobs with one Up/Down nuisance pair per output."
        )
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--source-campaign", type=Path, default=DEFAULT_SOURCE_CAMPAIGN)
    parser.add_argument("--local-campaign", type=Path, default=DEFAULT_LOCAL_CAMPAIGN)
    parser.add_argument("--event-index", type=Path, default=DEFAULT_EVENT_INDEX)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--segment-events", type=int, default=250_000)
    parser.add_argument("--records-per-job", type=int, default=16)
    parser.add_argument("--request-cpus", type=int, default=8)
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--request-disk-mb", type=int, default=20000)
    parser.add_argument("--job-flavour", default="longlunch")
    args = parser.parse_args(argv)
    if args.segment_events <= 0:
        raise ValueError("--segment-events must be positive")
    if args.records_per_job <= 0 or args.records_per_job > 16:
        raise ValueError("--records-per-job must be between one and sixteen")
    if args.request_cpus <= 0 or args.request_cpus > 8:
        raise ValueError("--request-cpus must be between one and eight")
    if not (
        args.request_cpus <= args.records_per_job <= 2 * args.request_cpus
    ):
        raise ValueError(
            "records per job must occupy one or two complete worker waves"
        )
    if min(args.request_memory_mb, args.request_disk_mb) <= 0:
        raise ValueError("memory and disk requests must be positive")

    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    for label, path in (
        ("repo", repo),
        ("source campaign", args.source_campaign.absolute()),
        ("local campaign", args.local_campaign.absolute()),
        ("event index", args.event_index.absolute()),
        ("benchmark", args.benchmark.absolute()),
        ("campaign", campaign),
    ):
        require_eos(path, label)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty campaign: {campaign}")

    source_manifest, source_records = load_source_records(args.source_campaign)
    event_index = read_payload(args.event_index)
    benchmark = read_payload(args.benchmark)
    if event_index.get("status") != "complete":
        raise RuntimeError("event index is incomplete")
    if benchmark.get("status") != "complete_observed_dy_sample":
        raise RuntimeError("local pair benchmark is not complete")
    if list(FINAL_SHAPE_NUISANCES) != list(
        benchmark.get("physics_policy", {}).get("nuisance_sets_list")
        or FINAL_SHAPE_NUISANCES
    ):
        raise RuntimeError("benchmark nuisance policy mismatch")

    for directory in (
        "bundles",
        "condor",
        "inputs",
        "logs",
        "outputs",
        "reports",
        "seed",
    ):
        (campaign / directory).mkdir(parents=True, exist_ok=True)
    seed = freeze_local_seed(args.local_campaign, campaign / "seed/local_pairs")
    completed_digests = set(seed["source_record_digests"])

    indexed = {
        str(record["record_digest"]): record
        for record in event_index.get("records") or []
    }
    if set(indexed) != set(source_records):
        raise RuntimeError("event index and source shard bundle coverage differ")
    ready_digests = {
        digest for digest, record in indexed.items() if record.get("status") == "ready"
    }
    excluded = {
        digest: record
        for digest, record in indexed.items()
        if record.get("status") != "ready"
    }
    if not completed_digests <= ready_digests:
        raise RuntimeError("local checkpoint contains non-ready or unknown sources")
    remaining_digests = sorted(ready_digests - completed_digests)

    by_process: dict[str, list[list[dict[str, Any]]]] = {}
    for digest in remaining_digests:
        index_record = indexed[digest]
        source_record = source_records[digest]
        process = str(index_record.get("process_group") or "")
        if process != str(source_record.get("process_group") or ""):
            raise RuntimeError(f"process mismatch for source {digest}")
        by_process.setdefault(process, []).append(
            split_record(source_record, index_record, args.segment_events)
        )

    target_events = args.segment_events * args.records_per_job
    partitions: list[dict[str, Any]] = []
    process_summary: dict[str, Any] = {}
    for process in sorted(by_process):
        packed = pack_process_sources(
            process,
            by_process[process],
            args.records_per_job,
            target_events,
        )
        process_events = 0
        process_segments = 0
        process_sources: set[str] = set()
        for local_index, records in enumerate(packed):
            partition_id = f"{process}_pairpart_{local_index:05d}"
            source_digests = sorted(
                {str(record["source_record_digest"]) for record in records}
            )
            expected_events = sum(int(record["segment_events"]) for record in records)
            digest = canonical_digest(records)
            payload = {
                "schema_version": "shape_histogram_2024_pair_partition_v1",
                "partition_id": partition_id,
                "record_group": "mc_background",
                "process_group": process,
                "record_digest": digest,
                "source_record_digests": source_digests,
                "expected_events": expected_events,
                "expected_segments": len(records),
                "physical_files": sorted(
                    {str(record["file_path"]) for record in records}
                ),
                "records": records,
            }
            target = campaign / "inputs" / process / f"{partition_id}.json"
            write_json(target, payload)
            partition = {
                "partition_id": partition_id,
                "process_group": process,
                "path": str(target),
                "sha256": sha256(target),
                "record_digest": digest,
                "expected_events": expected_events,
                "expected_segments": len(records),
                "physical_files": len(payload["physical_files"]),
                "source_records": len(source_digests),
            }
            partitions.append(partition)
            process_events += expected_events
            process_segments += len(records)
            process_sources.update(source_digests)
        expected_process_events = sum(
            int(indexed[digest]["events"])
            for digest in remaining_digests
            if indexed[digest]["process_group"] == process
        )
        if process_events != expected_process_events:
            raise RuntimeError(f"partition event coverage mismatch for {process}")
        process_summary[process] = {
            "source_records": len(process_sources),
            "segments": process_segments,
            "partitions": len(packed),
            "events": process_events,
        }

    covered_source_digests: set[str] = set()
    segment_ids: set[str] = set()
    for partition in partitions:
        payload = read_payload(Path(partition["path"]))
        for record in payload["records"]:
            digest = str(record["source_record_digest"])
            segment_id = str(record["segment_id"])
            covered_source_digests.add(digest)
            if segment_id in segment_ids:
                raise RuntimeError(f"duplicate segment: {segment_id}")
            segment_ids.add(segment_id)
    if covered_source_digests != set(remaining_digests):
        raise RuntimeError("partition source coverage is not exact")
    if covered_source_digests & completed_digests:
        raise RuntimeError("Condor partitions overlap the frozen local checkpoint")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    btag = repo / "analysis/hists/btageff2024.merged"
    for label, path in (
        ("Python archive", py38),
        ("proxy", proxy),
        ("btag efficiency", btag),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    worker_bundle_path = campaign / "bundles/shape_hist_2024_worker.tgz"
    payload_bundle_path = campaign / "bundles/shape_hist_2024_payloads.tgz"
    worker_bundle = build_worker_bundle(repo, worker_bundle_path)
    payload_bundle = build_payload_bundle(repo, payload_bundle_path, btag)
    btag_sha256 = sha256(btag)

    config = {
        "segment_events": args.segment_events,
        "records_per_job": args.records_per_job,
        "target_events_per_job": target_events,
        "request_cpus": args.request_cpus,
        "worker_waves_per_job": int(
            math.ceil(args.records_per_job / args.request_cpus)
        ),
        "request_memory_mb": args.request_memory_mb,
        "request_disk_mb": args.request_disk_mb,
        "job_flavour": args.job_flavour,
        "target_walltime_seconds": 3600,
    }
    fingerprint_payload = {
        "event_index_sha256": sha256(args.event_index),
        "source_manifest_sha256": sha256(args.source_campaign / "manifest.json"),
        "local_seed_coverage_sha256": seed["coverage_sha256"],
        "partition_digests": [item["record_digest"] for item in partitions],
        "nuisances": list(FINAL_SHAPE_NUISANCES),
        "worker_bundle_sha256": worker_bundle["sha256"],
        "payload_bundle_sha256": payload_bundle["sha256"],
        "python_bundle_sha256": sha256(py38),
        "btag_efficiency_sha256": btag_sha256,
        "config": config,
    }
    campaign_fingerprint = canonical_digest(fingerprint_payload)

    wrapper = campaign / "condor/run_shape_hist_pair_2024.sh"
    wrapper.write_text(wrapper_text(proxy.name, btag_sha256))
    wrapper.chmod(0o755)

    submit_files: list[dict[str, Any]] = []
    total_jobs = 0
    for nuisance in FINAL_SHAPE_NUISANCES:
        rows = []
        for partition in partitions:
            process = partition["process_group"]
            name = f"{nuisance}_{partition['partition_id']}"
            output_dir = campaign / "outputs" / nuisance / process
            log_dir = campaign / "logs" / nuisance / process
            output_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            hist_out = output_dir / f"{partition['partition_id']}.json.gz"
            meta_out = output_dir / f"{partition['partition_id']}.meta.json"
            rows.append(
                " ".join(
                    [
                        name,
                        partition["path"],
                        partition["sha256"],
                        nuisance,
                        str(hist_out),
                        str(meta_out),
                        str(log_dir),
                        str(partition["expected_events"]),
                        str(partition["expected_segments"]),
                    ]
                )
            )
        arguments = campaign / "condor" / f"arguments_{nuisance}.txt"
        arguments.write_text("\n".join(rows) + "\n")
        submit = campaign / "condor" / f"submit_{nuisance}.sub"
        submit.write_text(
            submit_text(
                wrapper=wrapper,
                arguments=arguments,
                logs=campaign / "logs",
                py38=py38,
                worker_bundle=worker_bundle_path,
                payload_bundle=payload_bundle_path,
                proxy=proxy,
                initialdir=campaign / "condor",
                request_memory_mb=args.request_memory_mb,
                request_disk_mb=args.request_disk_mb,
                request_cpus=args.request_cpus,
                job_flavour=args.job_flavour,
                campaign_name=campaign.name,
                campaign_fingerprint=campaign_fingerprint,
                segment_events=args.segment_events,
            )
        )
        submit_files.append(
            {
                "nuisance": nuisance,
                "submit_file": str(submit),
                "arguments_file": str(arguments),
                "jobs": len(rows),
                "submit_file_sha256": sha256(submit),
                "arguments_file_sha256": sha256(arguments),
            }
        )
        total_jobs += len(rows)

    manifest = {
        "schema_version": "shape_histogram_2024_condor_pair_campaign_v1",
        "status": "prepared_not_submitted",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "campaign_fingerprint": campaign_fingerprint,
        "year": 2024,
        "physics_policy": {
            "bjet_pt_min_gev": 30.0,
            "object_variations_are_weight_only": False,
            "selection_category_search_bin_and_xaxis_migration": True,
            "nuisance_sets": list(FINAL_SHAPE_NUISANCES),
            "nuisance_set_count": len(FINAL_SHAPE_NUISANCES),
            "directional_variations": 40,
            "output_per_job": "one exact Up/Down nuisance pair",
            "histogram_sections": [
                "histograms",
                "search_bin_histograms",
                "lowdm_variable_histograms",
                "highdm_variable_histograms",
            ],
            "btag_efficiency_sha256": btag_sha256,
        },
        "input_policy": {
            "mode": "minimum_required_NanoAOD_reread",
            "branch_audit": str(
                args.event_index.parent
                / "branch_audit_final_nominal_intermediate_20260725.json"
            ),
            "process_pure": True,
            "event_balanced": True,
            "physical_file_cache": (
                "one xrdcp cache per physical file per nuisance-pair job; "
                "entry segments share that cache"
            ),
            "source_records_total": len(source_records),
            "ready_source_records": len(ready_digests),
            "frozen_local_completed_sources": len(completed_digests),
            "condor_source_records": len(remaining_digests),
            "excluded_sources": list(excluded.values()),
            "condor_events": sum(
                int(indexed[digest]["events"]) for digest in remaining_digests
            ),
            "source_event_index": str(args.event_index),
            "source_event_index_sha256": sha256(args.event_index),
        },
        "local_seed": seed,
        "partition_policy": {
            **config,
            "partitions": len(partitions),
            "segments": len(segment_ids),
            "process_summary": process_summary,
            "partition_manifest": "inputs/<process>/<partition_id>.json",
            "large_source_policy": (
                "entry ranges remain in one partition and run concurrently; "
                "their physical ROOT file is prefetched once"
            ),
        },
        "jobs": {
            "total": total_jobs,
            "jobs_per_nuisance": len(partitions),
            "submit_clusters_expected": len(FINAL_SHAPE_NUISANCES),
            "condor_pilot": 0,
            "local_tests_substitute_for_pilot": True,
        },
        "runtime_policy": {
            "pool": "bigbird24 via module load lxbatch/eossubmit",
            "initialdir": str(campaign / "condor"),
            "persistent_paths": "EOS only",
            "afs_paths": 0,
            "python_environment": str(py38),
            "python_environment_sha256": sha256(py38),
            "worker_bundle": worker_bundle,
            "payload_bundle": payload_bundle,
            "request_cpus": args.request_cpus,
            "request_memory_mb": args.request_memory_mb,
            "request_disk_mb": args.request_disk_mb,
            "job_flavour": args.job_flavour,
            "target_walltime_seconds": 3600,
        },
        "benchmark": {
            "path": str(args.benchmark),
            "sha256": sha256(args.benchmark),
            "observed": benchmark.get("observed_sample"),
            "sizing_basis": (
                "250k-event entry segments run in at most two eight-worker "
                "waves; the exact worst-case layout is benchmarked below the "
                "one-hour envelope before submission"
            ),
            "representative_condor_layout_test": "required before submission",
            "representative_condor_layout_report": str(
                campaign / "benchmarks/submission_qualification/report.json"
            ),
        },
        "bundles": {
            "worker": worker_bundle,
            "payload": payload_bundle,
            "python": {
                "path": str(py38),
                "size": py38.stat().st_size,
                "sha256": sha256(py38),
            },
        },
        "submit_files": submit_files,
        "submission": {
            "status": "not_submitted",
            "cluster_ids": [],
            "submitted_at": None,
            "equivalent_campaign_check": None,
            "proxy_check": None,
        },
        "fingerprint_payload": fingerprint_payload,
    }
    write_json(campaign / "manifest.json", manifest)
    write_json(
        campaign / "partitions.json",
        {
            "schema_version": "shape_histogram_2024_pair_partitions_v1",
            "status": "complete",
            "campaign_fingerprint": campaign_fingerprint,
            "partitions": partitions,
        },
    )
    (campaign / "reports/readiness.md").write_text(
        "\n".join(
            [
                "# 2024 object-variation Condor pair campaign",
                "",
                "Status: prepared, not submitted.",
                "",
                f"- Frozen local sources reused: {len(completed_digests)}",
                f"- Condor source records: {len(remaining_digests)}",
                f"- Process-pure event-balanced partitions: {len(partitions)}",
                f"- Entry-range segments: {len(segment_ids)}",
                f"- Nuisance-pair jobs: {total_jobs}",
                f"- Requested CPUs/job: {args.request_cpus}",
                f"- Target walltime/job: <= 3600 s",
                f"- JobFlavour safety envelope: {args.job_flavour}",
                "- Output: one exact Up/Down pair with four adopted histogram sections",
                "- Condor pilot: none; representative lxplus tests are required",
                "",
            ]
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "campaign": str(campaign),
                "campaign_fingerprint": campaign_fingerprint,
                "local_seed_sources": len(completed_digests),
                "condor_sources": len(remaining_digests),
                "partitions": len(partitions),
                "jobs": total_jobs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
