#!/usr/bin/env python3
"""Deterministically merge disjoint DY feature-stage partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import CHANNELS, finalize_rz, merge_tree


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    merged: dict[str, Any] = {
        "schema_version": "dy_estimation_feature_2024_v1",
        "status": "running",
        "rz_high_raw": {},
        "rz_low_feature_raw": {},
        "mll_high": {},
        "mll_low_feature": {},
        "sparse_low_candidates": {},
        "summary": {
            "input_roots": 0,
            "completed_roots": 0,
            "missing_roots": [],
            "events_scanned": 0,
            "candidate_events": 0,
            "datasets": {},
        },
        "provenance": {
            "merge_method": "exact additive merge of disjoint ROOT partitions",
            "inputs": [],
            "channels": [],
        },
    }
    seen_roots = 0
    for path in args.inputs:
        payload = read_json(path)
        if not str(payload.get("status", "")).startswith("feature_stage_complete"):
            raise RuntimeError(f"incomplete partition {path}: {payload.get('status')}")
        provenance = payload.get("provenance") or {}
        merged["provenance"]["inputs"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "normalization_sha256": provenance.get("normalization_sha256"),
                "dy_dataset_policy": provenance.get(
                    "dy_dataset_policy",
                    "legacy_input_policy="
                    + str(provenance.get("dy_ptll_policy")),
                ),
            }
        )
        for channel in provenance.get("channels") or CHANNELS:
            if channel not in merged["provenance"]["channels"]:
                merged["provenance"]["channels"].append(channel)
        summary = payload.get("summary") or {}
        input_roots = int(summary.get("input_roots", 0))
        seen_roots += input_roots
        merged["summary"]["input_roots"] += input_roots
        merged["summary"]["completed_roots"] += int(
            summary.get("completed_roots", 0)
        )
        merged["summary"]["events_scanned"] += int(
            summary.get("events_scanned", 0)
        )
        merged["summary"]["candidate_events"] += int(
            summary.get("candidate_events", 0)
        )
        merged["summary"]["missing_roots"].extend(
            summary.get("missing_roots") or []
        )
        for dataset, count in (summary.get("datasets") or {}).items():
            merged["summary"]["datasets"][dataset] = (
                int(merged["summary"]["datasets"].get(dataset, 0))
                + int(count)
            )
        for key in (
            "rz_high_raw",
            "rz_low_feature_raw",
            "mll_high",
            "mll_low_feature",
        ):
            merge_tree(merged[key], payload.get(key) or {})
        for file_id, records in (
            payload.get("sparse_low_candidates") or {}
        ).items():
            merged["sparse_low_candidates"].setdefault(file_id, []).extend(
                records
            )
    if seen_roots != merged["summary"]["completed_roots"]:
        if not merged["summary"]["missing_roots"]:
            raise RuntimeError(
                "partition input/completed ROOT accounting does not close"
            )
    for records in merged["sparse_low_candidates"].values():
        records.sort(key=lambda item: (item["entry"], item["channel"]))
    merged["rz_high"] = finalize_rz(merged["rz_high_raw"])
    merged["rz_low_feature"] = finalize_rz(
        merged["rz_low_feature_raw"]
    )
    merged["status"] = (
        "feature_stage_complete"
        if not merged["summary"]["missing_roots"]
        else "feature_stage_complete_with_missing_roots"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "status": merged["status"],
                "inputs": len(args.inputs),
                "input_roots": merged["summary"]["input_roots"],
                "completed_roots": merged["summary"]["completed_roots"],
                "events_scanned": merged["summary"]["events_scanned"],
                "candidate_events": merged["summary"]["candidate_events"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
