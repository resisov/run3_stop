#!/usr/bin/env python3
"""Validate and summarize the Low-dM selection, plot, and limit update."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty artifact: {path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def limit_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete limit result: {path}: {payload.get('status')}")
    requested = int(payload.get("requested_point_count") or 0)
    collected = int(payload.get("collected_point_count") or 0)
    if requested <= 0 or collected != requested or payload.get("missing_points"):
        raise ValueError(f"invalid limit point accounting: {path}")
    return {
        "status": "complete",
        "requested_points": requested,
        "collected_points": collected,
        "missing_points": 0,
        "path": f"{path.parent.name}/{path.name}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", required=True, type=Path)
    parser.add_argument("--lowdm-limits", required=True, type=Path)
    parser.add_argument("--combined-limits", required=True, type=Path)
    parser.add_argument("--category-png", required=True, type=Path)
    parser.add_argument("--category-pdf", required=True, type=Path)
    parser.add_argument("--lowdm-contour-png", required=True, type=Path)
    parser.add_argument("--lowdm-contour-pdf", required=True, type=Path)
    parser.add_argument("--combined-contour-png", required=True, type=Path)
    parser.add_argument("--combined-contour-pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hists = read_json(args.hists)
    update = hists.get("lowdm_selection_update") or {}
    if update.get("status") != "complete" or update.get("processed_nominal_roots") != 9484:
        raise ValueError("merged histogram payload did not validate all 9,484 nominal ROOT inputs")
    scheme = ((hists.get("search_bin_histograms") or {}).get("cat7_SR_lowDeltaM") or {})
    if not scheme:
        raise ValueError("merged histogram payload has no Low-dM SR")
    for sample, variations in scheme.items():
        for variation, leaf in variations.items():
            if any(len(leaf.get(key) or []) != 42 for key in ("sumw", "sumw2", "entries")):
                raise ValueError(f"invalid 42-bin histogram: {sample}/{variation}")

    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": 2024,
        "luminosity_fb": 109.82,
        "classification": "adopted physics proposal",
        "selection_change": [
            "ISR-subjet b veto removed from Low-dM selection",
            "mTb < 175 GeV requirement removed from Low-dM selection and category assignment",
        ],
        "histograms": {
            "status": "complete",
            "processed_nominal_roots": 9484,
            "sr_bins": 42,
            "yield_comparison": update.get("nominal_yield_comparison"),
            "path": args.hists.name,
        },
        "limits": {
            "lowdm_only": limit_summary(args.lowdm_limits),
            "highdm54_lowdm42": limit_summary(args.combined_limits),
            "data_mode": "asimov",
            "max_mstop_gev": 1600,
            "run2_overlay": "SUS-19-010 observed and expected",
        },
        "artifacts": {
            "category_png": artifact(args.category_png),
            "category_pdf": artifact(args.category_pdf),
            "lowdm_contour_png": artifact(args.lowdm_contour_png),
            "lowdm_contour_pdf": artifact(args.lowdm_contour_pdf),
            "combined_contour_png": artifact(args.combined_contour_png),
            "combined_contour_pdf": artifact(args.combined_contour_pdf),
        },
        "known_limitation": (
            "The SR is rebuilt exactly from broad nominal-intermediate primitives. "
            "The five Low-dM CR 42-bin templates retain their current nominal definitions "
            "because region-specific CR primitives were not stored in that intermediate. "
            "The expected-limit model includes the weight-shape nuisances carried by the "
            "nominal histogram payload, luminosity, and autoMCStats; object-shape "
            "variations are not included in this result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
