#!/usr/bin/env python3
"""Validate balanced data merges and delete only their exact source pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uproot


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


def extract_value_from_bytes(data: bytes, key: str) -> Any:
    # The sidecars are emitted with indent=2. Restrict matches to exactly the
    # top-level indentation so similarly named fields inside datasets/files
    # cannot be mistaken for group-level audit fields.
    marker = b"\n  " + json.dumps(key).encode() + b":"
    position = data.find(marker)
    if position < 0:
        raise KeyError(key)
    position += len(marker)
    while position < len(data) and data[position] in b" \t\r\n":
        position += 1
    if position >= len(data):
        raise ValueError(f"truncated value for {key}")
    start = position
    first = data[position]
    if first in (ord("["), ord("{")):
        opening, closing = (first, ord("]") if first == ord("[") else ord("}"))
        depth = 0
        in_string = False
        escaped = False
        while position < len(data):
            char = data[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == ord("\\"):
                    escaped = True
                elif char == ord('"'):
                    in_string = False
            else:
                if char == ord('"'):
                    in_string = True
                elif char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        position += 1
                        return json.loads(data[start:position])
            position += 1
        raise ValueError(f"unterminated container for {key}")
    if first == ord('"'):
        position += 1
        escaped = False
        while position < len(data):
            char = data[position]
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord('"'):
                position += 1
                return json.loads(data[start:position])
            position += 1
        raise ValueError(f"unterminated string for {key}")
    while position < len(data) and data[position] not in b",}\r\n":
        position += 1
    return json.loads(data[start:position])


def sparse_sidecar(path: Path) -> dict[str, Any]:
    """Read only small audit fields, skipping the very large ``files`` value."""
    size = path.stat().st_size
    window = 4 * 1024 * 1024
    with path.open("rb") as handle:
        head = handle.read(min(window, size))
        if size > window:
            handle.seek(max(0, size - window))
            tail = handle.read(window)
        else:
            tail = head
    keys = (
        "bad_files",
        "events_written",
        "files_attempted",
        "files_processed",
        "group",
        "source_count",
        "source_fingerprint",
        "source_shards",
        "status",
        "stream",
        "validation",
    )
    result: dict[str, Any] = {}
    for key in keys:
        try:
            result[key] = extract_value_from_bytes(head, key)
        except KeyError:
            try:
                result[key] = extract_value_from_bytes(tail, key)
            except KeyError:
                # This fallback is not expected for the sorted sidecars, but
                # preserves correctness if their formatting changes.
                result[key] = extract_value_from_bytes(path.read_bytes(), key)
    return result


def root_entries(path: Path) -> int:
    with uproot.open(path) as root_file:
        if "Events" not in root_file:
            raise ValueError(f"{path}: Events tree missing")
        return int(root_file["Events"].num_entries)


def validate_group(
    group_record: dict[str, Any],
    merged_dir: Path,
) -> dict[str, Any]:
    name = str(group_record["group"])
    root_path = merged_dir / f"{name}.root"
    json_path = merged_dir / f"{name}.json"
    if not root_path.exists() or root_path.stat().st_size <= 0:
        raise ValueError(f"{name}: ROOT missing or empty")
    if not json_path.exists() or json_path.stat().st_size <= 0:
        raise ValueError(f"{name}: JSON missing or empty")
    sidecar = sparse_sidecar(json_path)
    if sidecar["group"] != name:
        raise ValueError(f"{name}: sidecar group mismatch")
    if sidecar["status"] != "complete":
        raise ValueError(f"{name}: status={sidecar['status']!r}")
    if sidecar["source_fingerprint"] != group_record["source_fingerprint"]:
        raise ValueError(f"{name}: source fingerprint mismatch")
    source_shards = [str(item) for item in sidecar["source_shards"]]
    if sorted(source_shards) != sorted(str(item) for item in group_record["source_shards"]):
        raise ValueError(f"{name}: source shard list differs from merge plan")
    source_count = int(sidecar["source_count"])
    if source_count != len(source_shards) or source_count not in (19, 20):
        raise ValueError(f"{name}: invalid source_count={source_count}")
    if sidecar["bad_files"]:
        raise ValueError(f"{name}: bad_files is nonempty")
    if int(sidecar["files_attempted"]) != int(sidecar["files_processed"]):
        raise ValueError(f"{name}: not all attempted files were processed")
    validation = sidecar["validation"]
    required_flags = (
        "merged_root_nonempty",
        "branch_schema_matches_sources",
        "all_sources_complete",
        "all_attempted_files_processed",
        "no_bad_files",
    )
    if any(validation.get(key) is not True for key in required_flags):
        raise ValueError(f"{name}: validation flag failed: {validation}")
    entries = root_entries(root_path)
    events_written = int(sidecar["events_written"])
    if entries != events_written:
        raise ValueError(f"{name}: entries={entries}, events_written={events_written}")
    if entries != int(validation.get("observed_entries", -1)):
        raise ValueError(f"{name}: observed_entries validation mismatch")
    if entries != int(validation.get("expected_entries", -1)):
        raise ValueError(f"{name}: expected_entries validation mismatch")
    return {
        "group": name,
        "stream": sidecar["stream"],
        "source_shards": source_shards,
        "source_count": source_count,
        "events_written": entries,
        "root_path": str(root_path),
        "json_path": str(json_path),
        "root_bytes": root_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
    }


def validate_all_groups(
    plan: dict[str, Any],
    merged_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    groups = plan.get("groups") or []
    if len(groups) != 680:
        raise ValueError(f"merge plan has {len(groups)} groups, expected 680")
    failures: list[str] = []
    validated: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(validate_group, group, merged_dir): str(group["group"])
            for group in groups
        }
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                validated.append(future.result())
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("merged validation failed:\n" + "\n".join(sorted(failures)[:100]))
    return sorted(validated, key=lambda item: item["group"])


def absolute_source_path(raw_path: str, repository: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repository / path


def build_cleanup_records(
    source_index: dict[str, Any],
    validated_groups: list[dict[str, Any]],
    repository: Path,
) -> list[dict[str, Any]]:
    replacement: dict[str, str] = {}
    duplicates: list[str] = []
    for group in validated_groups:
        for shard in group["source_shards"]:
            if shard in replacement:
                duplicates.append(shard)
            replacement[shard] = group["group"]
    if duplicates:
        raise ValueError(f"source shards occur in multiple groups: {sorted(set(duplicates))[:20]}")

    indexed = {str(item["shard"]): item for item in source_index.get("sources") or []}
    if set(replacement) != set(indexed):
        missing = sorted(set(indexed) - set(replacement))
        extra = sorted(set(replacement) - set(indexed))
        raise ValueError(f"source union mismatch: missing={missing[:20]} extra={extra[:20]}")
    if len(indexed) != 13_574:
        raise ValueError(f"source index has {len(indexed)} sources, expected 13574")

    records: list[dict[str, Any]] = []
    for shard in sorted(indexed):
        item = indexed[shard]
        root_path = absolute_source_path(str(item["root_path"]), repository).absolute()
        json_path = absolute_source_path(str(item["json_path"]), repository).absolute()
        if root_path.name != f"{shard}.root" or json_path.name != f"{shard}.json":
            raise ValueError(f"{shard}: source filename mismatch")
        if not root_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"{shard}: source ROOT/JSON pair is not present")
        records.append(
            {
                "shard": shard,
                "root_path": str(root_path),
                "json_path": str(json_path),
                "root_bytes": root_path.stat().st_size,
                "json_bytes": json_path.stat().st_size,
                "replacement_group": replacement[shard],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--merged-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    # Keep the user-facing /eos/user/... spelling. Path.resolve() rewrites the
    # EOS symlink to /eos/home-<letter>/..., while source_index.json correctly
    # retains /eos/user/...; mixing the two would trip the exact-parent guard.
    campaign = args.campaign.absolute()
    merged_dir = args.merged_dir.absolute()
    repository = campaign.parent.parent.absolute()
    manifest_path = merged_dir / "manifest.json"
    plan_path = merged_dir / "merge_plan.json"
    source_index_path = merged_dir / "source_index.json"
    cleanup_path = merged_dir / "cleanup_manifest.json"

    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete_with_missing_inputs":
        raise ValueError(f"merge manifest status={manifest.get('status')!r}")
    if int(manifest.get("group_count_valid") or 0) != 680:
        raise ValueError("merge manifest does not have 680 valid groups")
    missing = sorted(str(item) for item in manifest.get("missing_or_invalid_shards") or [])
    if missing != ["data_shard_04737", "data_shard_06758"]:
        raise ValueError(f"unexpected missing data shards: {missing}")
    if manifest.get("failures"):
        raise ValueError(f"merge manifest contains failures: {manifest['failures']}")

    plan = read_json(plan_path)
    source_index = read_json(source_index_path)
    validated_groups = validate_all_groups(plan, merged_dir, args.workers)
    cleanup_records = build_cleanup_records(
        source_index,
        validated_groups,
        repository,
    )

    cleanup_manifest = {
        "schema_version": "balanced_data_source_cleanup_v1",
        "status": "ready_for_deletion",
        "validated_at": utc_now(),
        "campaign": str(campaign),
        "merged_dir": str(merged_dir),
        "irrecoverable_warning": (
            "The exact source files listed below are deleted after --delete. "
            "The 680 validated merged ROOT/JSON replacements are retained."
        ),
        "missing_failed_shards_not_deleted": missing,
        "validated_merged_groups": len(validated_groups),
        "unique_source_shards": len(cleanup_records),
        "source_root_files": len(cleanup_records),
        "source_json_files": len(cleanup_records),
        "source_root_bytes": sum(item["root_bytes"] for item in cleanup_records),
        "source_json_bytes": sum(item["json_bytes"] for item in cleanup_records),
        "merged_root_bytes": sum(item["root_bytes"] for item in validated_groups),
        "merged_json_bytes": sum(item["json_bytes"] for item in validated_groups),
        "records": cleanup_records,
    }
    write_json_atomic(cleanup_path, cleanup_manifest)
    print(
        json.dumps(
            {
                key: cleanup_manifest[key]
                for key in (
                    "status",
                    "validated_merged_groups",
                    "unique_source_shards",
                    "source_root_files",
                    "source_json_files",
                    "source_root_bytes",
                    "source_json_bytes",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.delete:
        return 0

    cleanup_manifest["status"] = "deleting"
    cleanup_manifest["deletion_started_at"] = utc_now()
    write_json_atomic(cleanup_path, cleanup_manifest)
    deleted_root = 0
    deleted_json = 0
    for index, item in enumerate(cleanup_records, start=1):
        root_path = Path(item["root_path"])
        json_path = Path(item["json_path"])
        root_path.unlink()
        deleted_root += 1
        json_path.unlink()
        deleted_json += 1
        if index % 1000 == 0:
            print(f"deleted source pairs {index}/{len(cleanup_records)}", flush=True)

    remaining = [
        path
        for item in cleanup_records
        for path in (item["root_path"], item["json_path"])
        if Path(path).exists()
    ]
    if remaining:
        raise RuntimeError(f"{len(remaining)} exact source files remain after deletion")
    post_delete_groups = validate_all_groups(plan, merged_dir, args.workers)
    cleanup_manifest.update(
        {
            "status": "complete",
            "deletion_completed_at": utc_now(),
            "deleted_root_files": deleted_root,
            "deleted_json_files": deleted_json,
            "verified_source_files_remaining": 0,
            "post_delete_validated_merged_groups": len(post_delete_groups),
            "logical_bytes_reclaimed": cleanup_manifest["source_root_bytes"]
            + cleanup_manifest["source_json_bytes"],
            "recoverability": "source files were unlinked from EOS and are not retained by this workflow",
        }
    )
    write_json_atomic(cleanup_path, cleanup_manifest)
    print(
        json.dumps(
            {
                key: cleanup_manifest[key]
                for key in (
                    "status",
                    "deleted_root_files",
                    "deleted_json_files",
                    "verified_source_files_remaining",
                    "post_delete_validated_merged_groups",
                    "logical_bytes_reclaimed",
                    "recoverability",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
