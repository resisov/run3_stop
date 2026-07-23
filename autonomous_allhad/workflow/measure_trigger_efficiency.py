#!/usr/bin/env python3
"""Reduce trigger numerator/denominator counts into efficiencies and scale factors.

The event-processing stage must write the JSON schema documented in
reports/trigger_efficiency_measurement.md. This reducer is deterministic and
does not silently turn invalid weighted bins into efficiencies.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


def _beta_interval(passed: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0 or passed < 0 or passed > total:
        return (math.nan, math.nan)
    try:
        from scipy.special import betaincinv
    except ImportError as exc:
        raise RuntimeError("scipy is required for Clopper-Pearson intervals") from exc
    alpha = 1.0 - confidence
    low = 0.0 if passed == 0 else float(betaincinv(passed, total - passed + 1, alpha / 2))
    high = 1.0 if passed == total else float(betaincinv(passed + 1, total - passed, 1 - alpha / 2))
    return low, high


def _wilson(efficiency: float, entries: float, confidence: float) -> tuple[float, float]:
    if entries <= 0 or not 0.0 <= efficiency <= 1.0:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / entries
    centre = (efficiency + z * z / (2 * entries)) / denominator
    half = z * math.sqrt(efficiency * (1 - efficiency) / entries + z * z / (4 * entries**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _data_bin(item: dict[str, Any], confidence: float) -> dict[str, Any]:
    total, passed = int(item["total"]), int(item["passed"])
    low, high = _beta_interval(passed, total, confidence)
    efficiency = passed / total if total else math.nan
    return {"total": total, "passed": passed, "efficiency": efficiency, "interval": [low, high], "valid": total > 0}


def _mc_bin(item: dict[str, Any], confidence: float) -> dict[str, Any]:
    swt, sw2t = float(item["sumw_total"]), float(item["sumw2_total"])
    swp = float(item["sumw_passed"])
    valid = swt > 0 and sw2t > 0 and 0 <= swp <= swt
    efficiency = swp / swt if valid else math.nan
    neff = swt * swt / sw2t if valid else 0.0
    low, high = _wilson(efficiency, neff, confidence)
    return {"sumw_total": swt, "sumw2_total": sw2t, "sumw_passed": swp, "effective_entries": neff,
            "efficiency": efficiency, "interval": [low, high], "valid": valid}


def reduce_counts(payload: dict[str, Any], confidence: float = 0.682689492) -> dict[str, Any]:
    edges = payload["bin_edges_gev"]
    data, mc = payload["data"], payload["mc"]
    if len(edges) != len(data) + 1 or len(data) != len(mc):
        raise ValueError("bin_edges_gev must have one more entry than equally-sized data and mc arrays")
    bins = []
    for index, (d_raw, m_raw) in enumerate(zip(data, mc)):
        d, m = _data_bin(d_raw, confidence), _mc_bin(m_raw, confidence)
        sf_valid = d["valid"] and m["valid"] and m["efficiency"] > 0
        sf = d["efficiency"] / m["efficiency"] if sf_valid else math.nan
        if sf_valid:
            d_sigma = max(d["efficiency"] - d["interval"][0], d["interval"][1] - d["efficiency"])
            m_sigma = max(m["efficiency"] - m["interval"][0], m["interval"][1] - m["efficiency"])
            sf_unc = sf * math.sqrt((d_sigma / d["efficiency"]) ** 2 + (m_sigma / m["efficiency"]) ** 2) if d["efficiency"] > 0 else math.nan
        else:
            sf_unc = math.nan
        bins.append({"low_gev": edges[index], "high_gev": edges[index + 1], "data": d, "mc": m,
                     "scale_factor": sf, "scale_factor_uncertainty": sf_unc, "scale_factor_valid": sf_valid})
    return {"schema_version": 1, "measurement": payload.get("measurement", "unnamed"),
            "confidence_level": confidence, "bins": bins,
            "warnings": ["MC intervals use denominator effective entries; validate negative-weight coverage separately."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.682689492)
    args = parser.parse_args()
    payload = json.loads(args.counts.read_text())
    result = reduce_counts(payload, args.confidence_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
