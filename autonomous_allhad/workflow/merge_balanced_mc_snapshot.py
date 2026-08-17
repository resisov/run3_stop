#!/usr/bin/env python3
"""Merge a frozen snapshot of completed MC shards and remove exact sources.

Groups are process-pure, contain at most 20 completed shard ROOT files, and
are balanced by ``events_written``. Detailed source sidecars are retained in
compressed provenance payloads so the main merged JSON remains small.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uproot


SHARD_RE = re.compile(r"^mc_shard_\d+$")
COUNT_FIELDS = ("files_attempted", "files_processed", "events_read", "events_written")
FLOAT_FIELDS = ("sumw", "sumw2")
COUNT_MAP_FIELDS = ("sumw_source_counts", "signal_runs_sumw_source_counts")
FLOAT_MAP_FIELDS = ("signal_sumw_by_genmodel", "signal_event_genweight_sum_by_genmodel")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "process"
    return slug[:48]


@dataclass(frozen=True)
class Source:
    shard: str
    process: str
    root_path: str
    json_path: str
    events_read: int
    events_written: int
    files_attempted: int
    files_processed: int
    root_size: int
    record_digest: str
    schema_version: str
    branch_schema_digest: str


@dataclass
class Group:
    process: str
    index: int
    sources: list[Source]

    @property
    def name(self) -> str:
        process_hash = hashlib.sha256(self.process.encode()).hexdigest()[:8]
        return f"mc_{safe_slug(self.process)}_{process_hash}_balanced20_{self.index:04d}"

    @property
    def events_written(self) -> int:
        return sum(item.events_written for item in self.sources)

    @property
    def events_read(self) -> int:
        return sum(item.events_read for item in self.sources)

    @property
    def root_size(self) -> int:
        return sum(item.root_size for item in self.sources)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "algorithm": "process_pure_lpt_events_written_capacity_v1",
                "process": self.process,
                "sources": [
                    {
                        "shard": item.shard,
                        "record_digest": item.record_digest,
                        "events_written": item.events_written,
                        "root_size": item.root_size,
                        "branch_schema_digest": item.branch_schema_digest,
                    }
                    for item in sorted(self.sources, key=lambda source: source.shard)
                ],
            }
        )


def expected_mc_shards(arguments_path: Path) -> set[str]:
    expected: set[str] = set()
    with arguments_path.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            matches = [token for token in shlex.split(line) if SHARD_RE.fullmatch(token)]
            if len(matches) == 1:
                expected.add(matches[0])
    if not expected:
        raise RuntimeError(f"no MC shard names found in {arguments_path}")
    return expected


def source_from_sidecar(path: Path) -> Source:
    payload = read_json(path)
    shard = path.stem
    if not SHARD_RE.fullmatch(shard):
        raise ValueError(f"{path}: invalid MC shard filename")
    payload_shard = payload.get("shard") or payload.get("shard_id")
    if payload_shard and Path(str(payload_shard)).stem != shard:
        raise ValueError(f"{path}: payload shard does not match filename")
    if payload.get("status") != "complete":
        raise ValueError(f"{path}: status={payload.get('status')!r}")
    attempted = int(payload.get("files_attempted") or 0)
    processed = int(payload.get("files_processed") or 0)
    if attempted <= 0 or processed != attempted:
        raise ValueError(f"{path}: processed={processed}, attempted={attempted}")
    if payload.get("bad_files"):
        raise ValueError(f"{path}: bad_files is nonempty")
    datasets = payload.get("datasets") or {}
    processes = {str(item.get("process")) for item in datasets.values() if item.get("process")}
    if len(processes) != 1:
        raise ValueError(f"{path}: expected one process, found {sorted(processes)}")
    if any(bool(item.get("is_data")) or bool(item.get("is_signal")) for item in datasets.values()):
        raise ValueError(f"{path}: non-background dataset found in MC shard")
    process = next(iter(processes))
    root_path = Path(str(payload.get("root_file") or path.with_suffix(".root")))
    if not root_path.exists() or root_path.stat().st_size <= 0:
        raise ValueError(f"{path}: paired ROOT missing or empty")
    events_written = int(payload.get("events_written") or 0)
    if events_written < 0:
        raise ValueError(f"{path}: negative events_written")
    branch_schema = payload.get("branch_schema")
    if not isinstance(branch_schema, dict) or not branch_schema:
        raise ValueError(f"{path}: branch_schema missing")
    return Source(
        shard=shard,
        process=process,
        root_path=str(root_path),
        json_path=str(path),
        events_read=int(payload.get("events_read") or 0),
        events_written=events_written,
        files_attempted=attempted,
        files_processed=processed,
        root_size=root_path.stat().st_size,
        record_digest=str(payload.get("record_digest") or ""),
        schema_version=str(payload.get("schema_version") or ""),
        branch_schema_digest=canonical_digest(branch_schema),
    )


def scan_sources(paths: list[Path], expected: set[str], workers: int) -> tuple[list[Source], dict[str, str]]:
    def load(path: Path) -> tuple[Path, Source | None, str | None]:
        try:
            return path, source_from_sidecar(path), None
        except Exception as exc:
            return path, None, f"{type(exc).__name__}: {exc}"

    sources: list[Source] = []
    rejected: dict[str, str] = {}
    seen: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for path, source, error in pool.map(load, paths):
            if error is not None or source is None:
                rejected[path.stem] = error or "unknown scan error"
                continue
            if source.shard not in expected:
                rejected[source.shard] = "unexpected shard not present in arguments file"
                continue
            if source.shard in seen:
                rejected[source.shard] = "duplicate sidecar"
                continue
            seen.add(source.shard)
            sources.append(source)
    return sources, rejected


def balance_process(sources: list[Source], max_sources: int) -> list[Group]:
    if not sources:
        return []
    count = math.ceil(len(sources) / max_sources)
    groups = [Group(process=sources[0].process, index=index, sources=[]) for index in range(count)]
    ordered = sorted(sources, key=lambda item: (-item.events_written, -item.root_size, item.shard))
    for source in ordered:
        eligible = [group for group in groups if len(group.sources) < max_sources]
        target = min(
            eligible,
            key=lambda group: (
                group.events_written,
                group.root_size,
                len(group.sources),
                group.index,
            ),
        )
        target.sources.append(source)
    return groups


def merge_count_map(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)


def merge_float_map(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = float(target.get(str(key), 0.0)) + float(value or 0.0)


def merge_record(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in COUNT_FIELDS:
        target[key] = int(target.get(key, 0)) + int(source.get(key) or 0)
    for key in FLOAT_FIELDS:
        target[key] = float(target.get(key, 0.0)) + float(source.get(key) or 0.0)
    for key in COUNT_MAP_FIELDS:
        merge_count_map(target.setdefault(key, {}), source.get(key) or {})
    for key in FLOAT_MAP_FIELDS:
        merge_float_map(target.setdefault(key, {}), source.get(key) or {})


def merge_keyed_records(payloads: list[dict[str, Any]], key: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for record_key, source in (payload.get(key) or {}).items():
            if record_key not in merged:
                merged[record_key] = copy.deepcopy(source)
            else:
                merge_record(merged[record_key], source)
                if key == "physical_datasets":
                    values = set(str(item) for item in merged[record_key].get("split_dataset_ids") or [])
                    values.update(str(item) for item in source.get("split_dataset_ids") or [])
                    merged[record_key]["split_dataset_ids"] = sorted(values)
    return merged


def root_entries_and_branches(path: Path) -> tuple[int, list[str]]:
    with uproot.open(path) as root_file:
        if "Events" not in root_file:
            raise ValueError(f"{path}: Events tree missing")
        tree = root_file["Events"]
        return int(tree.num_entries), sorted(str(key) for key in tree.keys())


def write_provenance(path: Path, payloads: list[dict[str, Any]], sources: list[Source]) -> str:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(
                {
                    "schema_version": "flat_ntuple_merged_mc_provenance_v1",
                    "created_at": utc_now(),
                    "source_shards": [item.shard for item in sources],
                    "source_sidecars": payloads,
                },
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return file_sha256(path)


def existing_valid(group: Group, output_dir: Path) -> bool:
    root_path = output_dir / f"{group.name}.root"
    json_path = output_dir / f"{group.name}.json"
    provenance_path = output_dir / f"{group.name}.provenance.json.gz"
    if not all(path.exists() and path.stat().st_size > 0 for path in (root_path, json_path, provenance_path)):
        return False
    try:
        sidecar = read_json(json_path)
        entries, _ = root_entries_and_branches(root_path)
        return (
            sidecar.get("status") == "complete"
            and sidecar.get("source_fingerprint") == group.fingerprint
            and int(sidecar.get("events_written") or -1) == entries == group.events_written
            and sidecar.get("provenance_sha256") == file_sha256(provenance_path)
        )
    except Exception:
        return False


def merge_group(group: Group, output_dir: Path, hadd: str) -> dict[str, Any]:
    started = time.time()
    root_path = output_dir / f"{group.name}.root"
    json_path = output_dir / f"{group.name}.json"
    provenance_path = output_dir / f"{group.name}.provenance.json.gz"
    if existing_valid(group, output_dir):
        return {
            "group": group.name,
            "process": group.process,
            "status": "reused",
            "source_count": len(group.sources),
            "events_written": group.events_written,
        }

    payloads = [read_json(Path(source.json_path)) for source in group.sources]
    branch_schemas = {canonical_digest(payload.get("branch_schema")) for payload in payloads}
    schema_versions = {str(payload.get("schema_version") or "") for payload in payloads}
    if len(branch_schemas) != 1 or len(schema_versions) != 1:
        raise ValueError(f"{group.name}: source schemas differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = output_dir / f".{group.name}.tmp.{os.getpid()}.root"
    try:
        proc = subprocess.run(
            [hadd, "-f", str(tmp_root)] + [source.root_path for source in group.sources],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"hadd rc={proc.returncode}: {proc.stdout[-4000:]}")
        entries, branches = root_entries_and_branches(tmp_root)
        if entries != group.events_written:
            raise ValueError(f"{group.name}: entries={entries}, expected={group.events_written}")
        first_entries, first_branches = root_entries_and_branches(Path(group.sources[0].root_path))
        if first_entries != group.sources[0].events_written or branches != first_branches:
            raise ValueError(f"{group.name}: ROOT schema/source entry validation failed")
        os.replace(tmp_root, root_path)
    finally:
        if tmp_root.exists():
            tmp_root.unlink()

    provenance_sha256 = write_provenance(provenance_path, payloads, group.sources)
    datasets = merge_keyed_records(payloads, "datasets")
    physical_datasets = merge_keyed_records(payloads, "physical_datasets")
    for record in physical_datasets.values():
        record["normalization_factor"] = None
        record["normalization_status"] = (
            "requires_campaign_level_reaggregation_across_all_merged_and_pending_sources"
        )
    sidecar = {
        "schema_version": "flat_ntuple_merged_mc_balanced20_v1",
        "source_schema_version": next(iter(schema_versions)),
        "status": "complete",
        "created_at": utc_now(),
        "tree": "Events",
        "process": group.process,
        "group": group.name,
        "root_file": str(root_path),
        "provenance_file": str(provenance_path),
        "provenance_sha256": provenance_sha256,
        "source_fingerprint": group.fingerprint,
        "grouping_algorithm": "process-pure LPT events_written with hard source-count capacity",
        "max_sources_per_group": 20,
        "source_count": len(group.sources),
        "source_shards": sorted(source.shard for source in group.sources),
        "source_roots": [source.root_path for source in sorted(group.sources, key=lambda item: item.shard)],
        "source_jsons": [source.json_path for source in sorted(group.sources, key=lambda item: item.shard)],
        "source_record_digests": {
            source.shard: source.record_digest for source in sorted(group.sources, key=lambda item: item.shard)
        },
        "events_read": sum(int(payload.get("events_read") or 0) for payload in payloads),
        "events_written": group.events_written,
        "files_attempted": sum(int(payload.get("files_attempted") or 0) for payload in payloads),
        "files_processed": sum(int(payload.get("files_processed") or 0) for payload in payloads),
        "bad_files": [],
        "branch_schema": payloads[0].get("branch_schema"),
        "datasets": datasets,
        "physical_datasets": physical_datasets,
        "physical_dataset_split_counts": {
            name: len(set(str(item) for item in record.get("split_dataset_ids") or []))
            for name, record in physical_datasets.items()
        },
        "validation": {
            "merged_root_nonempty": True,
            "expected_entries": group.events_written,
            "observed_entries": group.events_written,
            "branch_schema_matches_sources": True,
            "all_sources_complete": True,
            "all_attempted_files_processed": True,
            "no_bad_files": True,
            "provenance_checksum_verified": True,
        },
        "merge_wall_time_s": time.time() - started,
    }
    write_json_atomic(json_path, sidecar)
    return {
        "group": group.name,
        "process": group.process,
        "status": "created",
        "source_count": len(group.sources),
        "events_written": group.events_written,
        "root_bytes": root_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
        "provenance_bytes": provenance_path.stat().st_size,
    }


def group_summary(groups: list[Group]) -> dict[str, Any]:
    values = [group.events_written for group in groups]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    counts = [len(group.sources) for group in groups]
    return {
        "source_count": sum(counts),
        "group_count": len(groups),
        "sources_per_group_min": min(counts) if counts else 0,
        "sources_per_group_max": max(counts) if counts else 0,
        "events_per_group_min": min(values) if values else 0,
        "events_per_group_max": max(values) if values else 0,
        "events_per_group_mean": mean,
        "events_per_group_cv": math.sqrt(variance) / mean if mean else 0.0,
    }


def validate_outputs(groups: list[Group], output_dir: Path, workers: int) -> None:
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(existing_valid, group, output_dir): group.name for group in groups}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                if not future.result():
                    failures.append(f"{name}: validation returned false")
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("output validation failed:\n" + "\n".join(sorted(failures)[:100]))


def cleanup_sources(
    process: str,
    sources: list[Source],
    groups: list[Group],
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    replacement: dict[str, str] = {}
    for group in groups:
        for source in group.sources:
            if source.shard in replacement:
                raise ValueError(f"duplicate source in groups: {source.shard}")
            replacement[source.shard] = group.name
    if set(replacement) != {source.shard for source in sources}:
        raise ValueError("group/source union mismatch before cleanup")
    process_key = f"{safe_slug(process)}_{hashlib.sha256(process.encode()).hexdigest()[:8]}"
    cleanup_path = output_dir / f"cleanup_{process_key}.json"
    if cleanup_path.exists():
        cleanup = read_json(cleanup_path)
        if cleanup.get("process") != process:
            raise ValueError(f"{cleanup_path}: process mismatch")
        if cleanup.get("status") == "complete":
            validate_outputs(groups, output_dir, workers)
            return cleanup
        if cleanup.get("status") != "deleting":
            raise ValueError(f"{cleanup_path}: unsupported resume status={cleanup.get('status')!r}")
        records = cleanup.get("records") or []
        if {str(item.get("shard")) for item in records} != set(replacement):
            raise ValueError(f"{cleanup_path}: cleanup/source union mismatch")
    else:
        records: list[dict[str, Any]] = []
        for source in sorted(sources, key=lambda item: item.shard):
            root_path = Path(source.root_path)
            json_path = Path(source.json_path)
            if root_path.name != f"{source.shard}.root" or json_path.name != f"{source.shard}.json":
                raise ValueError(f"{source.shard}: exact source filename mismatch")
            if not root_path.exists() or not json_path.exists():
                raise FileNotFoundError(f"{source.shard}: exact source pair missing before cleanup")
            records.append(
                {
                    "shard": source.shard,
                    "root_path": str(root_path),
                    "json_path": str(json_path),
                    "root_bytes": root_path.stat().st_size,
                    "json_bytes": json_path.stat().st_size,
                    "replacement_group": replacement[source.shard],
                }
            )
        cleanup = {
            "schema_version": "balanced_mc_snapshot_process_cleanup_v1",
            "status": "deleting",
            "process": process,
            "started_at": utc_now(),
            "source_shards": len(records),
            "source_root_bytes": sum(item["root_bytes"] for item in records),
            "source_json_bytes": sum(item["json_bytes"] for item in records),
            "records": records,
        }
        write_json_atomic(cleanup_path, cleanup)
    for index, item in enumerate(records, start=1):
        root_path = Path(item["root_path"])
        json_path = Path(item["json_path"])
        if root_path.exists():
            root_path.unlink()
        if json_path.exists():
            json_path.unlink()
        if index % 1000 == 0:
            print(f"{process}: deleted MC source pairs {index}/{len(records)}", flush=True)
    remaining = [
        path
        for item in records
        for path in (item["root_path"], item["json_path"])
        if Path(path).exists()
    ]
    if remaining:
        raise RuntimeError(f"{len(remaining)} exact MC source files remain")
    validate_outputs(groups, output_dir, workers)
    cleanup.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "deleted_root_files": len(records),
            "deleted_json_files": len(records),
            "remaining_source_files": 0,
            "post_delete_validated_groups": len(groups),
            "logical_bytes_reclaimed": cleanup["source_root_bytes"] + cleanup["source_json_bytes"],
            "recoverability": "exact source files were unlinked; merged ROOT/JSON/provenance replacements remain",
        }
    )
    write_json_atomic(cleanup_path, cleanup)
    return cleanup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sources", type=int, default=20)
    parser.add_argument("--scan-workers", type=int, default=12)
    parser.add_argument("--merge-workers", type=int, default=4)
    parser.add_argument("--delete-sources", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.max_sources, args.scan_workers, args.merge_workers) <= 0:
        parser.error("worker and source limits must be positive")

    campaign = args.campaign
    input_dir = campaign / "outputs" / "nominal"
    arguments_path = campaign / "condor" / "arguments.txt"
    expected = expected_mc_shards(arguments_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_index_path = args.output_dir / "source_index.json"
    if source_index_path.exists():
        source_index = read_json(source_index_path)
        if int(source_index.get("expected_mc_shards") or 0) != len(expected):
            raise ValueError("existing source index expected-MC count mismatch")
        sources = [Source(**item) for item in source_index.get("sources") or []]
        rejected = dict(source_index.get("rejected_sidecars") or {})
        pending_or_missing = [str(item) for item in source_index.get("pending_or_missing_shards") or []]
        print(f"reusing frozen source index with {len(sources)} completed MC sources", flush=True)
    else:
        # Freeze the completed sidecar snapshot before parsing. Outputs staged
        # by still-running Condor jobs after this line are excluded.
        frozen_paths = sorted(input_dir.glob("mc_shard_*.json"))
        sources, rejected = scan_sources(frozen_paths, expected, args.scan_workers)
        source_names = {source.shard for source in sources}
        pending_or_missing = sorted(expected - source_names)
        source_index = {
            "schema_version": "balanced_mc_snapshot_source_index_v1",
            "created_at": utc_now(),
            "frozen_sidecar_count": len(frozen_paths),
            "expected_mc_shards": len(expected),
            "valid_completed_sources": len(sources),
            "pending_or_missing_shards": pending_or_missing,
            "rejected_sidecars": rejected,
            "sources": [asdict(source) for source in sources],
        }
        write_json_atomic(source_index_path, source_index)

    by_process: dict[str, list[Source]] = {}
    for source in sources:
        by_process.setdefault(source.process, []).append(source)
    groups: list[Group] = []
    for process in sorted(by_process):
        groups.extend(balance_process(by_process[process], args.max_sources))
    process_summaries = {
        process: group_summary([group for group in groups if group.process == process])
        for process in sorted(by_process)
    }
    process_order = sorted(
        by_process,
        key=lambda process: (
            sum(source.root_size for source in by_process[process]),
            process,
        ),
    )
    plan = {
        "schema_version": "balanced_mc_snapshot_merge_plan_v1",
        "created_at": utc_now(),
        "campaign": str(campaign),
        "input_dir": str(input_dir),
        "output_dir": str(args.output_dir),
        "snapshot_policy": "only sidecars present at initial directory snapshot",
        "grouping_algorithm": "process-pure LPT events_written with hard source-count capacity",
        "max_sources_per_group": args.max_sources,
        "expected_mc_shards": len(expected),
        "valid_completed_sources": len(sources),
        "pending_or_missing_count": len(pending_or_missing),
        "rejected_sidecars": rejected,
        "execution_policy": "one process at a time: merge, validate, delete exact sources, revalidate",
        "process_order": process_order,
        "processes": process_summaries,
        "groups": [
            {
                "group": group.name,
                "process": group.process,
                "source_count": len(group.sources),
                "events_written": group.events_written,
                "input_root_bytes": group.root_size,
                "source_fingerprint": group.fingerprint,
                "source_shards": sorted(source.shard for source in group.sources),
            }
            for group in groups
        ],
    }
    write_json_atomic(args.output_dir / "merge_plan.json", plan)
    print(
        json.dumps(
            {
                "expected_mc_shards": len(expected),
                "valid_completed_sources": len(sources),
                "pending_or_missing_count": len(pending_or_missing),
                "rejected_sidecars": len(rejected),
                "process_count": len(by_process),
                "group_count": len(groups),
                "process_order": process_order,
                "processes": process_summaries,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.plan_only:
        return 0

    hadd = shutil.which("hadd")
    if not hadd:
        raise RuntimeError("hadd executable not found")
    manifest = {
        "schema_version": "balanced_mc_snapshot_manifest_v1",
        "started_at": utc_now(),
        "status": "running_process_by_process",
        "expected_mc_shards": len(expected),
        "snapshot_valid_sources": len(sources),
        "pending_or_missing_count": len(pending_or_missing),
        "pending_or_missing_shards": pending_or_missing,
        "rejected_sidecars": rejected,
        "group_count_expected": len(groups),
        "group_count_valid": 0,
        "failures": [],
        "source_files_deleted": False,
        "execution_policy": "one process at a time: merge, validate, delete exact sources, revalidate",
        "process_order": process_order,
        "completed_processes": [],
        "process_results": {},
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)

    for process_index, process in enumerate(process_order, start=1):
        process_sources = by_process[process]
        process_groups = [group for group in groups if group.process == process]
        process_key = f"{safe_slug(process)}_{hashlib.sha256(process.encode()).hexdigest()[:8]}"
        process_manifest_path = args.output_dir / f"process_{process_key}.manifest.json"
        cleanup_path = args.output_dir / f"cleanup_{process_key}.json"

        if cleanup_path.exists() and read_json(cleanup_path).get("status") == "complete":
            validate_outputs(process_groups, args.output_dir, args.merge_workers)
            cleanup = read_json(cleanup_path)
            manifest["process_results"][process] = {
                "status": "complete",
                "source_count": len(process_sources),
                "group_count": len(process_groups),
                "source_files_deleted": True,
                "cleanup_manifest": str(cleanup_path),
                "logical_bytes_reclaimed": cleanup.get("logical_bytes_reclaimed"),
                "resumed": True,
            }
            manifest["completed_processes"].append(process)
            manifest["group_count_valid"] = sum(
                int(item.get("group_count") or 0)
                for item in manifest["process_results"].values()
                if item.get("status") == "complete"
            )
            write_json_atomic(args.output_dir / "manifest.json", manifest)
            print(
                f"process {process_index}/{len(process_order)} {process}: reused completed cleanup",
                flush=True,
            )
            continue

        print(
            f"process {process_index}/{len(process_order)} {process}: "
            f"sources={len(process_sources)} groups={len(process_groups)} starting",
            flush=True,
        )
        process_results: list[dict[str, Any]] = []
        process_failures: list[dict[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.merge_workers) as pool:
            future_map = {
                pool.submit(merge_group, group, args.output_dir, hadd): group
                for group in process_groups
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                group = future_map[future]
                try:
                    process_results.append(future.result())
                except Exception as exc:
                    process_failures.append(
                        {
                            "group": group.name,
                            "process": process,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                if completed % 10 == 0 or completed == len(process_groups):
                    print(
                        f"{process}: groups complete={completed}/{len(process_groups)} "
                        f"failures={len(process_failures)}",
                        flush=True,
                    )

        process_manifest = {
            "schema_version": "balanced_mc_snapshot_process_manifest_v1",
            "process": process,
            "status": "failed" if process_failures else "merged",
            "source_count": len(process_sources),
            "group_count_expected": len(process_groups),
            "group_count_valid": len(process_results),
            "failures": process_failures,
            "results": sorted(process_results, key=lambda item: item["group"]),
        }
        write_json_atomic(process_manifest_path, process_manifest)
        if process_failures or len(process_results) != len(process_groups):
            manifest["status"] = "failed"
            manifest["failures"].extend(process_failures)
            manifest["process_results"][process] = {
                "status": "failed",
                "source_count": len(process_sources),
                "group_count": len(process_groups),
                "source_files_deleted": False,
                "process_manifest": str(process_manifest_path),
            }
            write_json_atomic(args.output_dir / "manifest.json", manifest)
            print(json.dumps(manifest, indent=2), flush=True)
            return 1

        validate_outputs(process_groups, args.output_dir, args.merge_workers)
        cleanup: dict[str, Any] | None = None
        if args.delete_sources:
            cleanup = cleanup_sources(
                process,
                process_sources,
                process_groups,
                args.output_dir,
                args.merge_workers,
            )
        process_manifest.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "source_files_deleted": bool(cleanup and cleanup.get("status") == "complete"),
                "cleanup_manifest": str(cleanup_path) if cleanup else None,
                "logical_bytes_reclaimed": cleanup.get("logical_bytes_reclaimed") if cleanup else None,
            }
        )
        write_json_atomic(process_manifest_path, process_manifest)
        manifest["process_results"][process] = {
            "status": "complete",
            "source_count": len(process_sources),
            "group_count": len(process_groups),
            "source_files_deleted": process_manifest["source_files_deleted"],
            "process_manifest": str(process_manifest_path),
            "cleanup_manifest": process_manifest["cleanup_manifest"],
            "logical_bytes_reclaimed": process_manifest["logical_bytes_reclaimed"],
        }
        manifest["completed_processes"].append(process)
        manifest["group_count_valid"] = sum(
            int(item.get("group_count") or 0)
            for item in manifest["process_results"].values()
            if item.get("status") == "complete"
        )
        write_json_atomic(args.output_dir / "manifest.json", manifest)
        print(
            f"process {process_index}/{len(process_order)} {process}: complete, "
            f"sources_deleted={process_manifest['source_files_deleted']}",
            flush=True,
        )

    validate_outputs(groups, args.output_dir, args.merge_workers)
    manifest.update(
        {
            "status": "snapshot_complete_with_pending_inputs",
            "completed_at": utc_now(),
            "group_count_valid": len(groups),
            "source_files_deleted": args.delete_sources
            and all(
                item.get("source_files_deleted") is True
                for item in manifest["process_results"].values()
            ),
            "logical_bytes_reclaimed": sum(
                int(item.get("logical_bytes_reclaimed") or 0)
                for item in manifest["process_results"].values()
            ),
        }
    )
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "snapshot_valid_sources": manifest["snapshot_valid_sources"],
                "pending_or_missing_count": manifest["pending_or_missing_count"],
                "group_count_valid": manifest["group_count_valid"],
                "failures": manifest["failures"],
                "source_files_deleted": manifest["source_files_deleted"],
                "completed_processes": manifest["completed_processes"],
                "logical_bytes_reclaimed": manifest["logical_bytes_reclaimed"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
