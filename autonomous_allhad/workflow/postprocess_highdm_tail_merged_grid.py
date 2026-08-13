#!/usr/bin/env python3
"""Collect and plot a complete High-dM tail-bin-merge limit grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_combine_inputs_from_preview import collect_limits, plot_contour
from build_free_background_combine_inputs_2024 import DECAY_LABELS
from build_boosted_an17_combine_inputs import write_json
from validate_highdm_tail_merged_grid import load_excluded_points


RUN2_CONTOURS = {
    "T2tt": Path("/eos/user/t/taiwoo/run2_sus19010_contours.json"),
    "T2tb": Path("/eos/user/t/taiwoo/run2_sus19010_contours_t2tb.json"),
    "T2bW": Path("/eos/user/t/taiwoo/run2_sus19010_contours_t2bw.json"),
}


def comparison(
    merged: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    ratios = []
    absolute = []
    for mass_key, point in merged.get("points", {}).items():
        reference = baseline.get("points", {}).get(mass_key)
        if not reference:
            continue
        new_value = float(point["expected"])
        old_value = float(reference["expected"])
        if old_value <= 0.0:
            continue
        ratios.append(new_value / old_value)
        absolute.append(new_value - old_value)
    if not ratios:
        return {"matched_points": 0}
    ratio_array = np.asarray(ratios, dtype=float)
    absolute_array = np.asarray(absolute, dtype=float)
    return {
        "matched_points": int(ratio_array.size),
        "expected_limit_ratio": {
            "minimum": float(np.min(ratio_array)),
            "median": float(np.median(ratio_array)),
            "maximum": float(np.max(ratio_array)),
        },
        "expected_limit_difference": {
            "minimum": float(np.min(absolute_array)),
            "median": float(np.median(absolute_array)),
            "maximum": float(np.max(absolute_array)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-limits", type=Path, required=True)
    parser.add_argument(
        "--topology",
        choices=("T2tt", "T2bW", "T2tb"),
        required=True,
    )
    parser.add_argument("--max-mstop", type=float, default=1800.0)
    parser.add_argument("--highdm-bins", type=int, default=50)
    parser.add_argument("--min-mlsp", type=float, default=1.0)
    parser.add_argument("--max-mlsp", type=float, default=1200.0)
    args = parser.parse_args()

    build_manifest = json.loads(
        (args.input_dir / "manifest.json").read_text()
    )
    excluded_points = load_excluded_points(args.input_dir)
    original_mass_keys = list(build_manifest["mass_keys"])
    mass_keys = [
        mass_key for mass_key in original_mass_keys
        if mass_key not in excluded_points
    ]
    limits = collect_limits(
        args.input_dir / "limits",
        mass_keys,
        args.input_dir / "expected_limits.json",
    )
    baseline = json.loads(args.baseline_limits.read_text())
    output_png = args.input_dir / (
        f"expected_limit_{args.topology.lower()}_"
        f"highdm{args.highdm_bins}_tailmerged_lowdm34_"
        "free_background_x1800.png"
    )
    contour_complete = False
    if limits["status"] in {"complete", "partial"}:
        contour_complete = plot_contour(
            limits,
            output_png,
            run2_contours=RUN2_CONTOURS[args.topology],
            luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
            analysis_label=None,
            x_max=args.max_mstop,
            y_min=args.min_mlsp,
            y_max=args.max_mlsp,
            decay_label=DECAY_LABELS[args.topology],
        )

    result = {
        "status": (
            "complete"
            if contour_complete and limits["status"] == "complete"
            else "partial"
            if contour_complete
            else "failed"
        ),
        "schema_version": (
            f"highdm{args.highdm_bins}_tailmerged_lowdm34_limit_v1"
        ),
        "topology": args.topology,
        "highdm_bins": args.highdm_bins,
        "lowdm_bins": 34,
        "merge_pairs_1based": build_manifest["merge_pairs_1based"],
        "original_mass_point_count": len(original_mass_keys),
        "mass_point_count": len(mass_keys),
        "excluded_point_count": len(excluded_points),
        "excluded_points": sorted(excluded_points),
        "limits": limits,
        "baseline_limits": str(args.baseline_limits),
        "baseline_comparison": comparison(limits, baseline),
        "contour_png": str(output_png) if contour_complete else None,
        "contour_pdf": (
            str(output_png.with_suffix(".pdf"))
            if contour_complete
            else None
        ),
        "run2_overlay": True,
        "data_mode": "asimov",
    }
    write_json(args.input_dir / "limit_manifest.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "topology": args.topology,
                "collected": limits["collected_point_count"],
                "missing": len(limits["missing_points"]),
                "comparison": result["baseline_comparison"],
                "contour_png": result["contour_png"],
            },
            sort_keys=True,
        )
    )
    return 0 if contour_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
