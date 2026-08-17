#!/usr/bin/env python3
"""Validate and freeze the accepted jesFlavorQCD Condor partition outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


NUISANCE = "jesFlavorQCD"
VARIATIONS = {"jesFlavorQCDUp", "jesFlavorQCDDown"}
SECTIONS = (
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
)


def load(path: Path) -> dict:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    destination = args.output.resolve()
    frozen = destination / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)

    manifest = load(campaign / "manifest.json")
    btag_sha256 = manifest["fingerprint_payload"]["btag_efficiency_sha256"]
    exclusions = set(args.exclude)
    shard_paths = sorted((campaign / "inputs").glob("*/*_pairpart_*.json"))
    if len(shard_paths) != 6479:
        raise RuntimeError(f"expected 6479 partition shards, found {len(shard_paths)}")

    accepted: list[dict] = []
    excluded: list[dict] = []
    accepted_source_digests: set[str] = set()
    expected_source_digests: set[str] = set()
    accepted_segment_ids: set[str] = set()
    expected_events = 0
    accepted_events = 0
    expected_segments = 0
    accepted_segments = 0

    for shard_path in shard_paths:
        shard = load(shard_path)
        partition_id = str(shard["partition_id"])
        process = str(shard["process_group"])
        records = shard["records"]
        source_digests = set(shard["source_record_digests"])
        segment_ids = {str(record["segment_id"]) for record in records}
        expected_source_digests.update(source_digests)
        expected_events += int(shard["expected_events"])
        expected_segments += int(shard["expected_segments"])
        if partition_id in exclusions:
            excluded.append(
                {
                    "partition_id": partition_id,
                    "process": process,
                    "expected_events": int(shard["expected_events"]),
                    "expected_segments": int(shard["expected_segments"]),
                    "source_record_digests": sorted(source_digests),
                    "segment_ids": sorted(segment_ids),
                    "reason": "user_accepted_terminal_input_io_failure_no_recovery",
                }
            )
            continue

        histogram = campaign / "outputs" / NUISANCE / process / f"{partition_id}.json.gz"
        metadata_path = histogram.with_name(f"{partition_id}.meta.json")
        if not histogram.is_file() or histogram.stat().st_size <= 0:
            raise FileNotFoundError(histogram)
        if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
            raise FileNotFoundError(metadata_path)
        metadata = load(metadata_path)
        payload = load(histogram)
        summary = metadata.get("summary") or {}
        payload_summary = payload.get("summary") or {}

        checks = {
            "metadata_status": metadata.get("status") == "complete",
            "payload_status": payload.get("status") == "complete",
            "nuisance": metadata.get("nuisance") == NUISANCE,
            "variation_count": int(metadata.get("variation_count") or 0) == 2,
            "variation_pair": set(payload.get("variations") or []) == VARIATIONS,
            "histogram_checksum": sha256(histogram) == metadata.get("histogram_sha256"),
            "partition_digest": metadata.get("partition_digest") == shard.get("record_digest"),
            "partition_shard_checksum": (
                metadata.get("partition_shard_sha256") == sha256(shard_path)
            ),
            "source_digest_coverage": (
                set(metadata.get("source_record_digests") or []) == source_digests
            ),
            "expected_events": (
                int(metadata.get("expected_events") or -1) == int(shard["expected_events"])
            ),
            "events_read": (
                int(summary.get("events_read") or -1) == int(shard["expected_events"])
                and int(payload_summary.get("events_read") or -1)
                == int(shard["expected_events"])
            ),
            "expected_segments": (
                int(metadata.get("expected_segments") or -1)
                == int(shard["expected_segments"])
            ),
            "processed_segments": (
                len(summary.get("file_records") or []) == int(shard["expected_segments"])
            ),
            "no_bad_files": not (summary.get("bad_files") or [])
            and not (payload_summary.get("bad_files") or []),
            "btag_checksum": metadata.get("btag_efficiency_sha256") == btag_sha256,
            "four_sections": all(
                all(section in dataset for section in SECTIONS)
                for dataset in (payload.get("datasets") or {}).values()
            ),
        }
        file_records = summary.get("file_records") or []
        actual_segments = {
            (
                str(record.get("source_record_digest")),
                int(record.get("entry_start")),
                int(record.get("entry_stop")),
            )
            for record in file_records
        }
        expected_ranges = {
            (
                str(record["source_record_digest"]),
                int(record["entry_start"]),
                int(record["entry_stop"]),
            )
            for record in records
        }
        checks["exact_entry_ranges"] = actual_segments == expected_ranges
        checks["all_segments_complete"] = all(
            record.get("status") == "complete" for record in file_records
        )
        checks["segment_event_sum"] = (
            sum(int(record.get("events_read") or 0) for record in file_records)
            == int(shard["expected_events"])
        )
        checks["btag_applied"] = set((summary.get("btag_sf_status") or {})) == VARIATIONS and all(
            record.get("applied") is True
            for record in (summary.get("btag_sf_status") or {}).values()
        )
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise RuntimeError(f"{partition_id} failed checks: {failed}")
        if accepted_source_digests & source_digests:
            raise RuntimeError(f"duplicate source digest in {partition_id}")
        if accepted_segment_ids & segment_ids:
            raise RuntimeError(f"duplicate segment id in {partition_id}")
        accepted_source_digests.update(source_digests)
        accepted_segment_ids.update(segment_ids)
        accepted_events += int(shard["expected_events"])
        accepted_segments += int(shard["expected_segments"])

        link_dir = frozen / process
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / histogram.name
        if link.is_symlink():
            if link.resolve() != histogram.resolve():
                raise RuntimeError(f"frozen link target mismatch: {link}")
        elif link.exists():
            raise RuntimeError(f"frozen input is not a symlink: {link}")
        else:
            os.symlink(histogram, link)
        metadata_link = link_dir / metadata_path.name
        if metadata_link.is_symlink():
            if metadata_link.resolve() != metadata_path.resolve():
                raise RuntimeError(f"frozen metadata link target mismatch: {metadata_link}")
        elif metadata_link.exists():
            raise RuntimeError(f"frozen metadata input is not a symlink: {metadata_link}")
        else:
            os.symlink(metadata_path, metadata_link)
        accepted.append(
            {
                "partition_id": partition_id,
                "process": process,
                "histogram": str(histogram),
                "histogram_sha256": metadata["histogram_sha256"],
                "expected_events": int(shard["expected_events"]),
                "expected_segments": int(shard["expected_segments"]),
                "source_record_digests": sorted(source_digests),
            }
        )
        if len(accepted) % 500 == 0:
            print(
                json.dumps(
                    {
                        "stage": "validation_progress",
                        "accepted_partitions": len(accepted),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    unexpected_exclusions = exclusions - {record["partition_id"] for record in excluded}
    if unexpected_exclusions:
        raise RuntimeError(f"unknown exclusions: {sorted(unexpected_exclusions)}")
    missing_digests = expected_source_digests - accepted_source_digests
    excluded_digests = {
        digest for record in excluded for digest in record["source_record_digests"]
    }
    if missing_digests != excluded_digests:
        raise RuntimeError("source coverage gap does not exactly match exclusions")

    result = {
        "schema_version": "jesFlavorQCD_accepted_coverage_v1",
        "status": "complete_with_explicit_exclusions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "campaign_fingerprint": manifest["campaign_fingerprint"],
        "nuisance": NUISANCE,
        "variations": sorted(VARIATIONS),
        "btag_efficiency_sha256": btag_sha256,
        "expected_partition_count": len(shard_paths),
        "accepted_partition_count": len(accepted),
        "excluded_partition_count": len(excluded),
        "expected_events": expected_events,
        "accepted_events": accepted_events,
        "excluded_events": expected_events - accepted_events,
        "expected_segments": expected_segments,
        "accepted_segments": accepted_segments,
        "excluded_segments": expected_segments - accepted_segments,
        "expected_source_record_count": len(expected_source_digests),
        "accepted_source_record_count": len(accepted_source_digests),
        "excluded_source_record_count": len(missing_digests),
        "frozen_input_directory": str(frozen),
        "accepted": accepted,
        "excluded": excluded,
    }
    output = destination / "coverage_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "expected_partition_count",
                    "accepted_partition_count",
                    "excluded_partition_count",
                    "accepted_events",
                    "excluded_events",
                    "accepted_segments",
                    "excluded_segments",
                    "accepted_source_record_count",
                    "excluded_source_record_count",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
