#!/usr/bin/env python3
"""Freeze a complete photon-fake snapshot from corrected and reusable sidecars."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SIDECAR_SCHEMA = "photon_fake_2024_sidecar_shard_v1"
SNAPSHOT_SCHEMA = "photon_fake_2024_hybrid_snapshot_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def read_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    bundle = Path(str((manifest.get("local_job_bundle") or {}).get("path") or ""))
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    expected_sha = str((manifest.get("local_job_bundle") or {}).get("sha256") or "")
    if not expected_sha or sha256(bundle) != expected_sha:
        raise RuntimeError(f"local job bundle checksum mismatch: {bundle}")
    payload = read_gzip_json(bundle)
    jobs = list(payload.get("jobs") or [])
    if len(jobs) != int(manifest.get("jobs") or -1):
        raise RuntimeError(f"job count mismatch in {bundle}")
    return jobs


def job_key(job: dict[str, Any]) -> tuple[str, str]:
    return str(job["process"]), str(job["name"])


def validate_sidecar(
    campaign: Path,
    job: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    process = str(job["process"])
    shard_name = str(job["shard_basename"])[:-5]
    metadata_path = campaign / "metadata" / process / f"{shard_name}.json"
    output_path = campaign / "outputs" / process / f"{shard_name}.json.gz"
    reasons: list[str] = []
    metadata: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    try:
        metadata = read_json(metadata_path)
    except Exception as exc:
        reasons.append(f"metadata_read:{type(exc).__name__}:{exc}")
    if metadata and str(metadata.get("status") or "") != "complete":
        reasons.append(f"metadata_status:{metadata.get('status')}")
    if not output_path.is_file():
        reasons.append("output_missing")
    else:
        expected_size = metadata.get("histogram_size")
        if expected_size is not None and output_path.stat().st_size != int(expected_size):
            reasons.append(
                f"size_mismatch:{output_path.stat().st_size}!={int(expected_size)}"
            )
        actual_sha = sha256(output_path)
        expected_sha = str(metadata.get("histogram_sha256") or "")
        if not expected_sha:
            reasons.append("metadata_sha256_missing")
        elif actual_sha != expected_sha:
            reasons.append(f"sha256_mismatch:{actual_sha}!={expected_sha}")
        try:
            payload = read_gzip_json(output_path)
        except Exception as exc:
            reasons.append(f"sidecar_read:{type(exc).__name__}:{exc}")
    expected_digest = str((job.get("shard") or {}).get("record_digest") or "")
    metadata_digest = str(metadata.get("source_record_digest") or "")
    payload_digest = str((payload.get("summary") or {}).get("source_record_digest") or "")
    if payload:
        if payload.get("schema_version") != SIDECAR_SCHEMA:
            reasons.append(f"sidecar_schema:{payload.get('schema_version')}")
        if str(payload.get("status") or "") != "complete":
            reasons.append(f"sidecar_status:{payload.get('status')}")
    if not expected_digest:
        reasons.append("expected_source_digest_missing")
    if metadata_digest != expected_digest:
        reasons.append(
            f"metadata_source_digest:{metadata_digest}!={expected_digest}"
        )
    if payload_digest != expected_digest:
        reasons.append(f"payload_source_digest:{payload_digest}!={expected_digest}")
    if reasons:
        return None, reasons
    return {
        "process": process,
        "name": str(job["name"]),
        "metadata": str(metadata_path),
        "source": str(output_path),
        "size": output_path.stat().st_size,
        "sha256": str(metadata["histogram_sha256"]),
        "source_record_digest": expected_digest,
    }, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-campaign", required=True, type=Path)
    parser.add_argument("--reusable-campaign", required=True, type=Path)
    parser.add_argument(
        "--corrected-processes",
        nargs="+",
        default=["EGamma", "GJ", "QCD"],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    corrected_campaign = args.corrected_campaign.absolute()
    reusable_campaign = args.reusable_campaign.absolute()
    output_dir = args.output_dir.absolute()
    inputs_dir = output_dir / "inputs"
    snapshot_path = output_dir / "snapshot_manifest.json"
    measurement_manifest_path = output_dir / "measurement_campaign_manifest.json"
    if output_dir.exists():
        raise RuntimeError(f"snapshot destination already exists: {output_dir}")

    corrected_manifest = read_json(corrected_campaign / "manifest.json")
    reusable_manifest = read_json(reusable_campaign / "manifest.json")
    invariant_fields = (
        "source_campaign",
        "record_counts",
        "entry_segment_counts",
        "shard_counts",
        "requested_processes",
    )
    for field in invariant_fields:
        if corrected_manifest.get(field) != reusable_manifest.get(field):
            raise RuntimeError(f"campaign invariant differs: {field}")
    if (
        (corrected_manifest.get("source_shard_bundle") or {}).get("sha256")
        != (reusable_manifest.get("source_shard_bundle") or {}).get("sha256")
    ):
        raise RuntimeError("source shard bundle differs between campaigns")

    corrected_jobs = {job_key(job): job for job in read_jobs(corrected_manifest)}
    reusable_jobs = {job_key(job): job for job in read_jobs(reusable_manifest)}
    if set(corrected_jobs) != set(reusable_jobs):
        raise RuntimeError("logical job sets differ between campaigns")
    for key in corrected_jobs:
        left = str((corrected_jobs[key].get("shard") or {}).get("record_digest") or "")
        right = str((reusable_jobs[key].get("shard") or {}).get("record_digest") or "")
        if left != right:
            raise RuntimeError(f"logical source digest differs for {key}")

    corrected_processes = set(args.corrected_processes)
    selected: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for key in sorted(corrected_jobs):
        process, _name = key
        campaign = (
            corrected_campaign if process in corrected_processes else reusable_campaign
        )
        record, reasons = validate_sidecar(campaign, corrected_jobs[key])
        if reasons:
            invalid.append(
                {
                    "process": process,
                    "name": key[1],
                    "campaign": str(campaign),
                    "reasons": reasons,
                }
            )
            continue
        assert record is not None
        destination = inputs_dir / process / Path(record["source"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(record["source"])
        record["snapshot"] = str(destination)
        record["source_campaign"] = str(campaign)
        selected.append(record)
        counts[process] += 1
        source_counts[campaign.name] += 1

    complete = not invalid and len(selected) == len(corrected_jobs)
    status = "complete" if complete else "partial"
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "corrected_campaign": str(corrected_campaign),
        "reusable_campaign": str(reusable_campaign),
        "corrected_processes": sorted(corrected_processes),
        "selection_source": "real_subset_worker.py",
        "probe_policy": {
            "A_target": (
                "corrected: exactly one medium target; additional non-medium "
                "candidates do not veto the nominal GCR event"
            ),
            "B_C_D": (
                "unchanged: no target and exactly one candidate in the "
                "application/measurement-pass/measurement-fail union"
            ),
        },
        "expected_sidecars": len(corrected_jobs),
        "valid_sidecars": len(selected),
        "valid_counts_by_process": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "invalid_sidecars": len(invalid),
        "invalid": invalid,
        "sidecars": selected,
    }
    write_json(snapshot_path, snapshot)

    measurement_manifest = {
        "schema_version": "photon_fake_2024_hybrid_measurement_campaign_v1",
        "status": status,
        "created_at": snapshot["created_at"],
        "campaign": str(output_dir),
        "source_campaign": corrected_manifest.get("source_campaign"),
        "selection_source": "real_subset_worker.py",
        "nominal_intermediate_mutation": False,
        "jobs": len(corrected_jobs),
        "record_counts": corrected_manifest["record_counts"],
        "entry_segment_counts": corrected_manifest["entry_segment_counts"],
        "shard_counts": corrected_manifest["shard_counts"],
        "requested_processes": corrected_manifest["requested_processes"],
        "hybrid_snapshot": str(snapshot_path),
        "corrected_campaign": str(corrected_campaign),
        "reusable_campaign": str(reusable_campaign),
        "corrected_processes": sorted(corrected_processes),
        "probe_policy": snapshot["probe_policy"],
    }
    write_json(measurement_manifest_path, measurement_manifest)
    print(
        json.dumps(
            {
                "status": status,
                "valid_sidecars": len(selected),
                "expected_sidecars": len(corrected_jobs),
                "valid_counts_by_process": dict(sorted(counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "invalid_sidecars": len(invalid),
                "snapshot_manifest": str(snapshot_path),
                "measurement_campaign_manifest": str(measurement_manifest_path),
            },
            sort_keys=True,
        )
    )
    if not complete and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
