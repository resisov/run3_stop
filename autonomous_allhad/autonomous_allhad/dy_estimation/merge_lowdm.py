#!/usr/bin/env python3
"""Build one DY measurement from the feature baseline and exact refinements.

High- and Low-dM are two views of the same on/off-Z measurement.  The feature
stage already fills both views.  Only the topology-ambiguous Low-dM subset
needs a sparse NanoAOD refinement, so this merger adds that refinement and
writes a single artifact containing both regimes.
"""

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
    parser.add_argument(
        "--combined",
        type=Path,
        help="Single feature-stage artifact containing both DY2E and DY2M.",
    )
    parser.add_argument("--ee", type=Path)
    parser.add_argument("--mumu", type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = read_json(args.expected)
    if args.combined:
        if args.ee or args.mumu:
            parser.error("--combined cannot be used with --ee/--mumu")
        bases = [read_json(args.combined)]
    else:
        if not args.ee or not args.mumu:
            parser.error("provide --combined or both --ee and --mumu")
        bases = [read_json(args.ee), read_json(args.mumu)]
    merged: dict[str, Any] = {
        "schema_version": "dy_estimation_measurement_v2",
        "status": "running",
        "rz_high_raw": {},
        "mll_high": {},
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
            "combined": str(args.combined) if args.combined else None,
            "ee": str(args.ee) if args.ee else None,
            "mumu": str(args.mumu) if args.mumu else None,
            "expected": str(args.expected),
            "method": (
                "single High/Low-dM measurement: common feature baseline plus "
                "one exact canonical sparse NanoAOD refinement for only the "
                "topology-ambiguous Low-dM subset"
            ),
        },
    }
    for base in bases:
        if base.get("status") != "feature_stage_complete":
            raise SystemExit("feature baseline is incomplete")
        merge_tree(merged["rz_high_raw"], base.get("rz_high_raw") or {})
        merge_tree(merged["mll_high"], base.get("mll_high") or {})
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
    merged["rz_high"] = finalize_rz(merged["rz_high_raw"])
    merged["rz_low"] = finalize_rz(merged["rz_low_raw"])
    merged["status"] = "complete" if complete else "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, sort_keys=True, separators=(",", ":")))
    print(
        json.dumps(
            {
                "status": merged["status"],
                "summary": summary,
                "combined": {
                    "highdm": merged["rz_high"]["combined"],
                    "lowdm": merged["rz_low"]["combined"],
                },
            }
        )
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
