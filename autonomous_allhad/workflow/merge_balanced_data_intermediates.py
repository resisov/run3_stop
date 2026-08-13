#!/usr/bin/env python3
"""Merge completed data flat-ntuples into event-balanced, stream-pure groups.

The source ROOT/JSON pairs are never modified or deleted.  Groups are formed
independently for JetMET, EGamma, and Muon with a longest-processing-time
greedy assignment using ``events_written`` as the load and a hard maximum on
the number of source shards per group.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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


STREAMS = ("JetMET", "EGamma", "Muon")
COUNT_FIELDS = ("files_attempted", "files_processed", "events_read", "events_written")
FLOAT_FIELDS = ("sumw", "sumw2")
COUNT_MAP_FIELDS = ("sumw_source_counts", "signal_runs_sumw_source_counts")
FLOAT_MAP_FIELDS = ("signal_sumw_by_genmodel", "signal_event_genweight_sum_by_genmodel")
SHARD_RE = re.compile(r"^data_shard_\d+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
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


@dataclass(frozen=True)
class Source:
    shard: str
    stream: str
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
    stream: str
    index: int
    sources: list[Source]

    @property
    def name(self) -> str:
        return f"{self.stream}_balanced20_{self.index:04d}"

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
                "algorithm": "lpt_events_written_capacity_v1",
                "stream": self.stream,
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


def expected_data_shards(arguments_path: Path) -> set[str]:
    expected: set[str] = set()
    with arguments_path.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            tokens = shlex.split(line)
            matches = [token for token in tokens if SHARD_RE.fullmatch(token)]
            if len(matches) == 1:
                expected.add(matches[0])
    if not expected:
        raise RuntimeError(f"no data shard names found in {arguments_path}")
    return expected


def source_from_sidecar(path: Path) -> Source:
    payload = read_json(path)
    shard = path.stem
    if not SHARD_RE.fullmatch(shard):
        raise ValueError(f"{path}: invalid data shard name {shard!r}")
    payload_shard = payload.get("shard") or payload.get("shard_id")
    if payload_shard and Path(str(payload_shard)).stem != shard:
        raise ValueError(f"{path}: payload shard {payload_shard!r} does not match filename")
    if payload.get("status") != "complete":
        raise ValueError(f"{path}: status is {payload.get('status')!r}, not 'complete'")
    attempted = int(payload.get("files_attempted") or 0)
    processed = int(payload.get("files_processed") or 0)
    if attempted <= 0 or processed != attempted:
        raise ValueError(f"{path}: files_processed={processed} does not match files_attempted={attempted}")
    if payload.get("bad_files"):
        raise ValueError(f"{path}: bad_files is nonempty")
    datasets = payload.get("datasets") or {}
    streams = {str(item.get("process")) for item in datasets.values() if item.get("process")}
    if len(streams) != 1:
        raise ValueError(f"{path}: expected exactly one stream, found {sorted(streams)}")
    stream = next(iter(streams))
    if stream not in STREAMS:
        raise ValueError(f"{path}: unsupported data stream {stream!r}")
    root_path = Path(str(payload.get("root_file") or path.with_suffix(".root")))
    if not root_path.exists() or root_path.stat().st_size <= 0:
        raise ValueError(f"{path}: paired ROOT file is missing or empty: {root_path}")
    events_written = int(payload.get("events_written") or 0)
    if events_written < 0:
        raise ValueError(f"{path}: negative events_written={events_written}")
    branch_schema = payload.get("branch_schema")
    if not isinstance(branch_schema, dict) or not branch_schema:
        raise ValueError(f"{path}: missing branch_schema")
    return Source(
        shard=shard,
        stream=stream,
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


def scan_sources(
    input_dir: Path,
    expected: set[str],
    workers: int,
) -> tuple[list[Source], dict[str, str]]:
    sources: list[Source] = []
    rejected: dict[str, str] = {}
    seen: set[str] = set()
    paths = sorted(input_dir.glob("data_shard_*.json"))

    def load(path: Path) -> tuple[Path, Source | None, str | None]:
        try:
            source = source_from_sidecar(path)
        except Exception as exc:
            return path, None, f"{type(exc).__name__}: {exc}"
        return path, source, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for path, source, error in pool.map(load, paths):
            if error is not None or source is None:
                rejected[path.stem] = error or "unknown source scan error"
                continue
            if source.shard not in expected:
                rejected[path.stem] = "unexpected data shard not present in arguments file"
                continue
            if source.shard in seen:
                rejected[source.shard] = "duplicate sidecar"
                continue
            seen.add(source.shard)
            sources.append(source)
    return sources, rejected


def balance_stream(sources: list[Source], max_sources: int) -> list[Group]:
    if not sources:
        return []
    group_count = math.ceil(len(sources) / max_sources)
    groups = [Group(stream=sources[0].stream, index=index, sources=[]) for index in range(group_count)]
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
    groups.sort(key=lambda group: group.index)
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
                    prior = set(str(item) for item in merged[record_key].get("split_dataset_ids") or [])
                    prior.update(str(item) for item in source.get("split_dataset_ids") or [])
                    merged[record_key]["split_dataset_ids"] = sorted(prior)
    return merged


def root_entries_and_branches(path: Path) -> tuple[int, list[str]]:
    with uproot.open(path) as root_file:
        if "Events" not in root_file:
            raise ValueError(f"{path}: Events tree is missing")
        tree = root_file["Events"]
        return int(tree.num_entries), sorted(str(key) for key in tree.keys())


def existing_group_valid(root_path: Path, json_path: Path, group: Group) -> bool:
    if not root_path.exists() or root_path.stat().st_size <= 0 or not json_path.exists():
        return False
    try:
        payload = read_json(json_path)
        if payload.get("status") != "complete":
            return False
        if payload.get("source_fingerprint") != group.fingerprint:
            return False
        entries, _branches = root_entries_and_branches(root_path)
        return entries == group.events_written == int(payload.get("events_written") or -1)
    except Exception:
        return False


def merge_group(group: Group, output_dir: Path, hadd: str) -> dict[str, Any]:
    started = time.time()
    root_path = output_dir / f"{group.name}.root"
    json_path = output_dir / f"{group.name}.json"
    if existing_group_valid(root_path, json_path, group):
        return {
            "group": group.name,
            "stream": group.stream,
            "status": "reused",
            "source_count": len(group.sources),
            "events_written": group.events_written,
            "root_size": root_path.stat().st_size,
            "wall_time_s": time.time() - started,
        }

    payloads = [read_json(Path(source.json_path)) for source in group.sources]
    branch_schemas = {canonical_digest(payload.get("branch_schema")) for payload in payloads}
    schema_versions = {str(payload.get("schema_version") or "") for payload in payloads}
    if len(branch_schemas) != 1:
        raise ValueError(f"{group.name}: source branch schemas do not match")
    if len(schema_versions) != 1:
        raise ValueError(f"{group.name}: source schema versions do not match: {sorted(schema_versions)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = output_dir / f".{group.name}.tmp.{os.getpid()}.root"
    try:
        cmd = [hadd, "-f", str(tmp_root)] + [source.root_path for source in group.sources]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{group.name}: hadd rc={proc.returncode}: {proc.stdout[-4000:]}")
        observed_entries, merged_branches = root_entries_and_branches(tmp_root)
        if observed_entries != group.events_written:
            raise ValueError(
                f"{group.name}: merged entries={observed_entries}, expected={group.events_written}"
            )
        source_entries, source_branches = root_entries_and_branches(Path(group.sources[0].root_path))
        if source_entries != group.sources[0].events_written:
            raise ValueError(
                f"{group.name}: reference source entries={source_entries}, "
                f"sidecar={group.sources[0].events_written}"
            )
        if merged_branches != source_branches:
            raise ValueError(f"{group.name}: merged ROOT branches differ from source branches")
        os.replace(tmp_root, root_path)
    finally:
        if tmp_root.exists():
            tmp_root.unlink()

    datasets = merge_keyed_records(payloads, "datasets")
    physical_datasets = merge_keyed_records(payloads, "physical_datasets")
    files: list[Any] = []
    for payload in payloads:
        files.extend(payload.get("files") or [])
    sidecar = {
        "schema_version": "flat_ntuple_merged_data_balanced20_v1",
        "source_schema_version": next(iter(schema_versions)),
        "status": "complete",
        "created_at": utc_now(),
        "tree": "Events",
        "stream": group.stream,
        "group": group.name,
        "root_file": str(root_path),
        "source_fingerprint": group.fingerprint,
        "grouping_algorithm": "LPT descending events_written with hard source-count capacity",
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
        "files": files,
        "validation": {
            "merged_root_nonempty": root_path.stat().st_size > 0,
            "expected_entries": group.events_written,
            "observed_entries": group.events_written,
            "branch_schema_matches_sources": True,
            "all_sources_complete": True,
            "all_attempted_files_processed": True,
            "no_bad_files": True,
        },
        "merge_wall_time_s": time.time() - started,
    }
    write_json_atomic(json_path, sidecar)
    return {
        "group": group.name,
        "stream": group.stream,
        "status": "created",
        "source_count": len(group.sources),
        "events_written": group.events_written,
        "root_size": root_path.stat().st_size,
        "wall_time_s": time.time() - started,
    }


def group_summary(groups: list[Group]) -> dict[str, Any]:
    events = [group.events_written for group in groups]
    sizes = [group.root_size for group in groups]
    counts = [len(group.sources) for group in groups]
    mean_events = sum(events) / len(events) if events else 0.0
    variance = sum((value - mean_events) ** 2 for value in events) / len(events) if events else 0.0
    return {
        "source_count": sum(counts),
        "group_count": len(groups),
        "source_count_per_group_min": min(counts) if counts else 0,
        "source_count_per_group_max": max(counts) if counts else 0,
        "events_per_group_min": min(events) if events else 0,
        "events_per_group_max": max(events) if events else 0,
        "events_per_group_mean": mean_events,
        "events_per_group_cv": math.sqrt(variance) / mean_events if mean_events else 0.0,
        "input_bytes_per_group_min": min(sizes) if sizes else 0,
        "input_bytes_per_group_max": max(sizes) if sizes else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sources", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scan-workers", type=int, default=12)
    parser.add_argument(
        "--reuse-source-index",
        action="store_true",
        help="Reuse output-dir/source_index.json after verifying that the set of source sidecars is unchanged",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_sources <= 0 or args.workers <= 0 or args.scan_workers <= 0:
        parser.error("--max-sources, --workers, and --scan-workers must be positive")

    arguments_path = args.campaign / "condor" / "arguments.txt"
    input_dir = args.campaign / "outputs" / "nominal"
    expected = expected_data_shards(arguments_path)
    source_index_path = args.output_dir / "source_index.json"
    if args.reuse_source_index:
        source_index = read_json(source_index_path)
        if int(source_index.get("expected_data_shards") or 0) != len(expected):
            raise RuntimeError("source index is stale: expected shard count changed")
        sources = [Source(**item) for item in source_index.get("sources") or []]
        rejected = dict(source_index.get("rejected_sidecars") or {})
        indexed_names = {source.shard for source in sources}
        current_names = {
            path.stem
            for path in input_dir.glob("data_shard_*.json")
            if path.stem in expected
        }
        if current_names != indexed_names:
            added = sorted(current_names - indexed_names)
            removed = sorted(indexed_names - current_names)
            raise RuntimeError(
                f"source index is stale: added={added[:20]} removed={removed[:20]}"
            )
    else:
        sources, rejected = scan_sources(input_dir, expected, args.scan_workers)
        source_index = {
            "schema_version": "balanced_data_source_index_v1",
            "created_at": utc_now(),
            "campaign": str(args.campaign),
            "expected_data_shards": len(expected),
            "rejected_sidecars": rejected,
            "sources": [asdict(source) for source in sources],
        }
        write_json_atomic(source_index_path, source_index)
    valid_names = {source.shard for source in sources}
    missing = sorted(expected - valid_names)

    by_stream: dict[str, list[Source]] = {stream: [] for stream in STREAMS}
    for source in sources:
        by_stream[source.stream].append(source)
    groups: list[Group] = []
    for stream in STREAMS:
        groups.extend(balance_stream(by_stream[stream], args.max_sources))

    plan = {
        "schema_version": "balanced_data_merge_plan_v1",
        "created_at": utc_now(),
        "campaign": str(args.campaign),
        "input_dir": str(input_dir),
        "output_dir": str(args.output_dir),
        "grouping_algorithm": "LPT descending events_written with hard source-count capacity",
        "balance_metric": "events_written",
        "max_sources_per_group": args.max_sources,
        "expected_data_shards": len(expected),
        "valid_source_shards": len(sources),
        "missing_or_invalid_shards": missing,
        "rejected_sidecars": rejected,
        "streams": {
            stream: group_summary([group for group in groups if group.stream == stream])
            for stream in STREAMS
        },
        "groups": [
            {
                "group": group.name,
                "stream": group.stream,
                "source_count": len(group.sources),
                "events_read": group.events_read,
                "events_written": group.events_written,
                "input_root_bytes": group.root_size,
                "source_fingerprint": group.fingerprint,
                "source_shards": sorted(source.shard for source in group.sources),
            }
            for group in groups
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "merge_plan.json", plan)
    print(
        json.dumps(
            {
                "expected": len(expected),
                "valid": len(sources),
                "missing": len(missing),
                "groups": len(groups),
                "streams": plan["streams"],
                "plan": str(args.output_dir / "merge_plan.json"),
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
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(merge_group, group, args.output_dir, hadd): group for group in groups}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            group = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "group": group.name,
                        "stream": group.stream,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % 10 == 0 or completed == len(groups):
                print(
                    f"[{utc_now()}] groups complete={completed}/{len(groups)} failures={len(failures)}",
                    flush=True,
                )

    result_names = {result["group"] for result in results}
    status = "complete" if not failures and len(result_names) == len(groups) and not missing else (
        "complete_with_missing_inputs" if not failures and len(result_names) == len(groups) else "failed"
    )
    manifest = {
        "schema_version": "balanced_data_merge_manifest_v1",
        "completed_at": utc_now(),
        "status": status,
        "campaign": str(args.campaign),
        "output_dir": str(args.output_dir),
        "expected_data_shards": len(expected),
        "valid_source_shards": len(sources),
        "missing_or_invalid_shards": missing,
        "rejected_sidecars": rejected,
        "source_files_deleted": False,
        "group_count_expected": len(groups),
        "group_count_valid": len(result_names),
        "groups_created": sum(result["status"] == "created" for result in results),
        "groups_reused": sum(result["status"] == "reused" for result in results),
        "failures": sorted(failures, key=lambda item: item["group"]),
        "streams": plan["streams"],
        "results": sorted(results, key=lambda item: item["group"]),
        "plan_digest": canonical_digest(plan),
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status",
        "expected_data_shards",
        "valid_source_shards",
        "missing_or_invalid_shards",
        "group_count_expected",
        "group_count_valid",
        "groups_created",
        "groups_reused",
        "failures",
    )}, indent=2), flush=True)
    return 0 if status in {"complete", "complete_with_missing_inputs"} else 1


if __name__ == "__main__":
    sys.exit(main())
