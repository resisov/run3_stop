#!/usr/bin/env python3
"""Merge exact Low-dM sparse partitions with the lossless feature baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import finalize_rz, merge_tree


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee", type=Path, required=True)
    parser.add_argument("--mumu", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = read_json(args.expected)
    bases = [read_json(args.ee), read_json(args.mumu)]
    merged: dict[str, Any] = {
        "schema_version": "dy_estimation_lowdm_merged_2024_v1",
        "status": "running",
        "rz_low_raw": {},
        "mll_low": {},
        "summary": {
            "expected_partitions": int(expected["partitions"]),
            "completed_partitions": 0,
            "candidate_files": 0,
            "candidate_events": 0,
            "matched_events": 0,
            "selected_events": 0,
            "read_windows": 0,
            "failures": [],
        },
        "provenance": {
            "ee": str(args.ee),
            "mumu": str(args.mumu),
            "expected": str(args.expected),
            "method": "exact canonical sparse NanoAOD recovery added once to the exact feature baseline",
        },
    }
    for base in bases:
        if base.get("status") != "feature_stage_complete":
            raise SystemExit("feature baseline is incomplete")
        merge_tree(merged["rz_low_raw"], base.get("rz_low_feature_raw") or {})
        merge_tree(merged["mll_low"], base.get("mll_low_feature") or {})

    seen: set[str] = set()
    for record in expected.get("manifests") or []:
        stem = str(record["stem"])
        if stem in seen:
            raise RuntimeError(f"duplicate expected partition {stem}")
        seen.add(stem)
        output_path = Path(record["output"])
        if not output_path.is_file():
            merged["summary"]["failures"].append({"stem": stem, "error": "missing output"})
            continue
        payload = read_json(output_path)
        if payload.get("status") != "complete" or payload.get("stem") != stem:
            merged["summary"]["failures"].append({"stem": stem, "error": "incomplete or mismatched output"})
            continue
        summary = payload["summary"]
        merged["summary"]["completed_partitions"] += 1
        for key in ("candidate_files", "candidate_events", "matched_events", "selected_events", "read_windows"):
            merged["summary"][key] += int(summary[key])
        merge_tree(merged["rz_low_raw"], payload.get("raw") or {})
        merge_tree(merged["mll_low"], payload.get("mll") or {})

    summary = merged["summary"]
    complete = (
        not summary["failures"]
        and summary["completed_partitions"] == expected["partitions"]
        and summary["candidate_files"] == expected["candidate_files"]
        and summary["candidate_events"] == expected["candidate_events"]
        and summary["matched_events"] == expected["candidate_events"]
    )
    merged["rz_low"] = finalize_rz(merged["rz_low_raw"])
    merged["status"] = "complete" if complete else "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, sort_keys=True, separators=(",", ":")))
    print(json.dumps({"status": merged["status"], "summary": summary, "combined": merged["rz_low"]["combined"]}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
