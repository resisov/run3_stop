#!/usr/bin/env python3
"""Audit photon-template campaign outputs against the submitted shard manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "photon_fake_template_events_2024_v1"


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--exclude-process",
        action="append",
        default=[],
        help="process group to exclude from this audit; may be repeated",
    )
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    manifest = read_json(campaign / "campaign_manifest.json")
    excluded_processes = {str(item) for item in args.exclude_process}
    shard_paths = []
    for path in sorted((campaign / "shards").glob("*.json")):
        shard = read_json(path)
        if str(shard.get("process")) not in excluded_processes:
            shard_paths.append(path)
    expected = {path.stem: path for path in shard_paths}

    by_process: dict[str, Counter[str]] = {}
    category_counts: Counter[str] = Counter()
    source_digests: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    valid_names: set[str] = set()
    total_events_read = 0
    total_selected_events = 0
    total_files_attempted = 0
    total_files_processed = 0
    for metadata_path in sorted((campaign / "metadata").rglob("*.json")):
        process = metadata_path.parent.name
        if process in excluded_processes:
            continue
        name = metadata_path.stem
        counts = by_process.setdefault(process, Counter())
        counts["metadata"] += 1
        output_path = campaign / "outputs" / process / f"{name}.json.gz"
        errors: list[str] = []
        try:
            metadata = read_json(metadata_path)
            summary = metadata.get("summary") or {}
            if metadata.get("status") != "complete":
                errors.append(f"metadata status={metadata.get('status')}")
            if not output_path.is_file():
                errors.append("event output missing")
            else:
                counts["outputs"] += 1
                if args.verify_checksums:
                    if sha256(output_path) != metadata.get("event_file_sha256"):
                        errors.append("event output checksum mismatch")
                    payload = read_json(output_path)
                    if payload.get("schema_version") != EXPECTED_SCHEMA:
                        errors.append(f"event schema={payload.get('schema_version')}")
                    if payload.get("status") != "complete":
                        errors.append(f"event status={payload.get('status')}")
            attempted = int(summary.get("files_attempted") or 0)
            processed = int(summary.get("files_processed") or 0)
            if attempted <= 0 or processed != attempted:
                errors.append(f"file coverage={processed}/{attempted}")
            if summary.get("bad_files"):
                errors.append(f"bad_files={len(summary['bad_files'])}")
            expected_shard_path = expected.get(name)
            if expected_shard_path is None:
                errors.append("metadata has no expected shard")
            else:
                expected_shard = read_json(expected_shard_path)
                if str(summary.get("source_record_digest") or "") != str(expected_shard.get("record_digest") or ""):
                    errors.append("source record digest mismatch")
            digest = str(summary.get("source_record_digest") or "")
            if digest:
                source_digests[digest] += 1
            total_events_read += int(summary.get("events_read") or 0)
            total_selected_events += int(summary.get("selected_events") or 0)
            total_files_attempted += attempted
            total_files_processed += processed
            category_counts.update(
                {key: int(value) for key, value in (summary.get("category_counts") or {}).items()}
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
        if errors:
            counts["invalid"] += 1
            invalid.append({"name": name, "process": process, "metadata": str(metadata_path), "errors": errors})
        else:
            counts["valid"] += 1
            valid_names.add(name)

    missing = sorted(set(expected) - valid_names)
    duplicate_digests = sorted(key for key, count in source_digests.items() if count > 1)
    expected_jobs = len(expected) if excluded_processes else int(manifest.get("jobs") or len(expected))
    complete = (
        len(expected) == expected_jobs
        and not missing
        and not invalid
        and not duplicate_digests
        and len(valid_names) == expected_jobs
    )
    result = {
        "schema_version": "photon_fake_template_campaign_audit_2024_v1",
        "status": "complete" if complete else "incomplete",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "expected_jobs": expected_jobs,
        "expected_shards": len(expected),
        "valid_outputs": len(valid_names),
        "missing_outputs": len(missing),
        "missing_output_names": missing,
        "invalid_outputs": len(invalid),
        "invalid_output_records": invalid,
        "duplicate_source_record_digests": duplicate_digests,
        "files_attempted": total_files_attempted,
        "files_processed": total_files_processed,
        "events_read": total_events_read,
        "selected_events": total_selected_events,
        "category_counts": dict(sorted(category_counts.items())),
        "by_process": {key: dict(sorted(value.items())) for key, value in sorted(by_process.items())},
        "checksum_verification": bool(args.verify_checksums),
        "excluded_processes": sorted(excluded_processes),
        "manifest_payload_sha256": (manifest.get("payload_bundle") or {}).get("sha256"),
        "manifest_worker_sha256": (manifest.get("worker_bundle") or {}).get("sha256"),
    }
    output = args.output or campaign / "latest_audit.json"
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ("status", "expected_jobs", "valid_outputs", "missing_outputs", "invalid_outputs", "events_read", "selected_events")}, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
