#!/usr/bin/env python3
"""Validate frozen diagonal-v3 SR/CR templates before datacard creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CATEGORIES = (
    "Nb1_NISR0", "Nb1_NISR1plus", "Nb2plus_NISR0", "Nb2plus_NISR1plus"
)
CR_REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
BACKGROUNDS = {"ST", "TT", "WtoLNu", "Zto2Nu", "QCD", "DY", "GJ", "VV"}
ALLOWED = BACKGROUNDS | {"data_obs"}


def check_leaf(record: dict[str, Any], label: str, failures: list[str]) -> None:
    score = record.get("gnn_score") or {}
    for name in ("sumw", "sumw2", "entries"):
        values = np.asarray(score.get(name, []), dtype=float)
        if len(values) != 5 or not np.all(np.isfinite(values)):
            failures.append(f"{label}: invalid {name}")
    if np.any(np.asarray(score.get("sumw2", []), dtype=float) < 0.0):
        failures.append(f"{label}: negative sumw2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr", required=True, type=Path)
    parser.add_argument("--cr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sr = json.loads(args.sr.read_text())
    cr = json.loads(args.cr.read_text())
    failures: list[str] = []
    if sr.get("status") != "complete" or sr.get("bad_files"):
        failures.append("SR is incomplete or has bad files")
    if cr.get("status") != "complete" or cr.get("bad_files"):
        failures.append("CR is incomplete or has bad files")
    if sr.get("score_edges") != cr.get("score_edges"):
        failures.append("SR/CR score edges differ")
    if sr.get("checkpoint") != cr.get("checkpoint"):
        failures.append("SR/CR checkpoints differ")
    if cr.get("input_files_requested") != cr.get("input_files_valid"):
        failures.append("CR lost requested input files")
    if int(cr.get("input_files_valid", -1)) != 5437:
        failures.append("CR does not contain all 5437 data+MC intermediate files")

    sr_records = sr.get("histograms", {}).get("nominal", {}).get("SR", {})
    if set(sr_records) != set(CATEGORIES):
        failures.append("SR category set differs from frozen four categories")
    sr_backgrounds: set[str] = set()
    signal_points: set[str] = set()
    for category, by_sample in sr_records.items():
        for sample, record in by_sample.items():
            check_leaf(record, f"SR/{category}/{sample}", failures)
            if sample.startswith("T2"):
                signal_points.add(sample)
            else:
                sr_backgrounds.add(sample)
    if sr_backgrounds != BACKGROUNDS:
        failures.append(f"SR background set is {sorted(sr_backgrounds)}")

    nominal = cr.get("histograms", {}).get("nominal", {})
    cr_samples: dict[str, list[str]] = {}
    for region in CR_REGIONS:
        records = nominal.get(region, {})
        if set(records) != set(CATEGORIES):
            failures.append(f"{region}: category set differs from frozen definition")
        present: set[str] = set()
        for category, by_sample in records.items():
            unexpected = set(by_sample) - ALLOWED
            if unexpected:
                failures.append(f"{region}/{category}: Other samples {sorted(unexpected)}")
            for sample, record in by_sample.items():
                present.add(sample)
                check_leaf(record, f"{region}/{category}/{sample}", failures)
        cr_samples[region] = sorted(present)
        if "data_obs" not in present:
            failures.append(f"{region}: data_obs is absent")

    rz = cr.get("histograms", {}).get("nominal_rz", {})
    for region in ("DY2E", "DY2M"):
        if not any("DY" in samples for samples in rz.get(region, {}).values()):
            failures.append(f"{region}: RZ-weighted DY template is absent")

    payload = {
        "schema_version": "diagonal_v3_srcr_validation_v1",
        "status": "complete" if not failures else "failed",
        "failures": failures,
        "checkpoint": sr.get("checkpoint"),
        "score_edges": sr.get("score_edges"),
        "sr": {
            "test_events": sr.get("test_events"),
            "signal_mass_points": len(signal_points),
            "backgrounds": sorted(sr_backgrounds),
        },
        "cr": {
            "input_files": cr.get("input_files_valid"),
            "samples_by_region": cr_samples,
        },
        "other_present": False if not any("Other samples" in item for item in failures) else True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
