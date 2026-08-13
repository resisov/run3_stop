#!/usr/bin/env python3
"""Stage physics-equivalent chunks after an append-only b-tag payload update."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from coffea.util import load


STRICT_WARNING_KEYS = (
    "weight_failures",
    "missing_input_roots",
    "missing_sidecars",
    "zero_entry_roots",
    "weight_rejections",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_signature(histogram: Any) -> list[str]:
    return [str(axis) for axis in histogram.axes]


def histograms_identical(left: Any, right: Any) -> bool:
    if axis_signature(left) != axis_signature(right):
        return False
    left_values = np.asarray(left.values(flow=True))
    right_values = np.asarray(right.values(flow=True))
    if not np.array_equal(left_values, right_values, equal_nan=True):
        return False
    left_variances = left.variances(flow=True)
    right_variances = right.variances(flow=True)
    if left_variances is None or right_variances is None:
        return left_variances is None and right_variances is None
    return bool(
        np.array_equal(
            np.asarray(left_variances),
            np.asarray(right_variances),
            equal_nan=True,
        )
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-payload", required=True, type=Path)
    parser.add_argument("--new-payload", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--source-work-dir", required=True, type=Path)
    parser.add_argument("--destination-work-dir", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    with gzip.open(args.metadata, "rt") as handle:
        metadata = json.load(handle)
    added_keys = sorted(metadata)
    if not added_keys:
        raise RuntimeError("metadata contains no added b-tag dataset keys")

    old = load(args.old_payload)
    new = load(args.new_payload)
    old_map = old.get("UParTAK4") or {}
    new_map = new.get("UParTAK4") or {}
    old_keys = set(old_map)
    new_keys = set(new_map)
    if new_keys - old_keys != set(added_keys):
        raise RuntimeError(
            "new b-tag keys do not exactly match metadata: "
            f"payload_only={sorted(new_keys - old_keys)}, metadata={added_keys}"
        )
    if old_keys - new_keys:
        raise RuntimeError(f"old b-tag keys disappeared: {sorted(old_keys - new_keys)[:5]}")
    changed_old_keys = [
        key for key in sorted(old_keys) if not histograms_identical(old_map[key], new_map[key])
    ]
    if changed_old_keys:
        raise RuntimeError(
            f"existing b-tag histograms changed: {changed_old_keys[:5]}"
        )

    old_sha = sha256(args.old_payload)
    new_sha = sha256(args.new_payload)
    source_chunks = args.source_work_dir / "hist_chunks"
    destination_chunks = args.destination_work_dir / "hist_chunks"
    destination_chunks.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: dict[str, str] = {}
    for source in sorted(source_chunks.glob("chunk_*.json")):
        payload = json.loads(source.read_text())
        summary = payload.get("summary") or {}
        reason = ""
        if payload.get("status") != "complete":
            reason = "status_not_complete"
        elif any(summary.get(key) for key in STRICT_WARNING_KEYS):
            reason = "strict_warning_present"
        else:
            contract = ((summary.get("build_options") or {}).get("btag_efficiency") or {})
            if contract.get("sha256") != old_sha:
                reason = "unexpected_old_btag_sha256"
            elif any(
                "intermediate_2024_dyto2x4jets_condor10_20260729" in str(root)
                for root in summary.get("input_roots") or []
            ):
                reason = "contains_new_dy_input"
            elif any(
                key in json.dumps(summary.get("scale_factor_status_audit") or {})
                for key in added_keys
            ):
                reason = "uses_added_btag_key"
        if reason:
            skipped[source.name] = reason
            continue

        staged = json.loads(json.dumps(payload))
        staged_summary = staged["summary"]
        staged_summary["build_options"]["btag_efficiency"] = {
            "path": "analysis/hists/btageff2024.merged",
            "exists": True,
            "sha256": new_sha,
            "expected_sha256": new_sha,
            "matches_expected": True,
        }
        staged_summary["append_only_btag_reuse"] = {
            "status": "validated_physics_equivalent",
            "old_payload_sha256": old_sha,
            "new_payload_sha256": new_sha,
            "added_keys": added_keys,
            "unchanged_existing_keys": len(old_keys),
            "chunk_uses_added_keys": False,
        }
        destination = destination_chunks / source.name
        write_json(destination, staged)
        copied.append(source.name)

    audit = {
        "status": "complete",
        "old_payload": str(args.old_payload),
        "new_payload": str(args.new_payload),
        "old_payload_sha256": old_sha,
        "new_payload_sha256": new_sha,
        "added_keys": added_keys,
        "unchanged_existing_keys": len(old_keys),
        "changed_existing_keys": changed_old_keys,
        "source_work_dir": str(args.source_work_dir),
        "destination_work_dir": str(args.destination_work_dir),
        "copied_chunks": copied,
        "copied_chunk_count": len(copied),
        "skipped_chunks": skipped,
    }
    write_json(args.audit, audit)
    print(
        json.dumps(
            {
                "status": "complete",
                "copied_chunks": len(copied),
                "skipped_chunks": len(skipped),
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "audit": str(args.audit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
