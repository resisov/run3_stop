#!/usr/bin/env python3
"""Append validated per-dataset 2024 b-tag efficiency histograms."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from coffea.util import load, save


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram_sum(histogram: Any) -> float:
    values = np.asarray(histogram.values(flow=True), dtype=float)
    return float(np.sum(values[np.isfinite(values)]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--metadata-name", required=True)
    parser.add_argument("--audit-name", default="btageff_append_audit.json")
    args = parser.parse_args()

    metadata_path = args.repo / "analysis/metadata" / f"{args.metadata_name}.json.gz"
    payload_path = args.repo / "analysis/hists/btageff2024.merged"
    output_dir = args.repo / "analysis/hists/btageff2024"
    with gzip.open(metadata_path, "rt") as handle:
        metadata = json.load(handle)
    expected_keys = sorted(metadata)
    if not expected_keys:
        raise RuntimeError("b-tag metadata contains no datasets")

    missing_outputs = [
        key for key in expected_keys if not (output_dir / f"{key}.futures").exists()
    ]
    if missing_outputs:
        raise RuntimeError(
            f"missing {len(missing_outputs)} b-tag outputs: {missing_outputs[:5]}"
        )

    payload = load(payload_path)
    target = payload.setdefault("UParTAK4", {})
    collisions = sorted(set(expected_keys) & set(target))
    if collisions:
        raise RuntimeError(
            "refusing to overwrite existing b-tag efficiency keys: "
            + ", ".join(collisions[:5])
        )

    sums: dict[str, float] = {}
    axis_signatures: dict[str, list[str]] = {}
    for key in expected_keys:
        item = load(output_dir / f"{key}.futures")
        histogram = item.get("UParTAK4")
        if histogram is None:
            raise RuntimeError(f"{key}: UParTAK4 histogram missing")
        total = histogram_sum(histogram)
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"{key}: invalid histogram sum {total}")
        signature = [str(axis) for axis in histogram.axes]
        if axis_signatures and signature != next(iter(axis_signatures.values())):
            raise RuntimeError(f"{key}: histogram axes differ from the first dataset")
        target[key] = histogram
        sums[key] = total
        axis_signatures[key] = signature

    before = sha256(payload_path)
    args.campaign.mkdir(parents=True, exist_ok=True)
    backup = args.campaign / "btageff2024.merged.before_append"
    if not backup.exists():
        shutil.copy2(payload_path, backup)
    temporary = payload_path.with_name(f"{payload_path.name}.tmp.append.{os.getpid()}")
    save(payload, temporary)
    os.replace(temporary, payload_path)

    after = sha256(payload_path)
    reloaded = load(payload_path)
    missing_after = sorted(
        set(expected_keys) - set((reloaded.get("UParTAK4") or {}).keys())
    )
    if missing_after:
        raise RuntimeError(f"post-write keys missing: {missing_after}")

    audit = {
        "schema": "btageff2024_dataset_append_audit_v1",
        "status": "complete",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata": str(metadata_path),
        "metadata_dataset_count": len(expected_keys),
        "metadata_file_count": sum(
            len((metadata[key] or {}).get("files") or []) for key in expected_keys
        ),
        "payload": str(payload_path),
        "backup": str(backup),
        "sha256_before": before,
        "sha256_after": after,
        "added_keys": expected_keys,
        "histogram_sums": sums,
        "axis_signatures": axis_signatures,
        "missing_keys_after": missing_after,
    }
    audit_path = args.campaign / args.audit_name
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "added_keys": len(expected_keys),
                "sha256_after": after,
                "audit": str(audit_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
