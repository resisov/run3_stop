#!/usr/bin/env python3
"""Validate, collect, and plot the diagonal-v3 Low-dM-only expected limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = THIS_DIR.parent.parent / "workflow"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from postprocess_limits import DECAY_LABELS, collect_limits, plot_contour, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--topology", choices=("T2tt", "T2bW", "T2tb"), required=True)
    parser.add_argument("--max-mstop", type=int, default=1800)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    if manifest.get("topology") != args.topology:
        raise SystemExit("topology does not match input manifest")
    masses = list(manifest["mass_points"])
    limits = collect_limits(
        args.input_dir / "limits", masses, args.input_dir / "expected_limits.json"
    )
    output_png = args.input_dir / f"expected_limit_{args.topology.lower()}_lowdm_diagonal_v3.png"
    contour = False
    if limits["status"] in {"complete", "partial"}:
        contour = plot_contour(
            limits,
            output_png,
            run2_contours=None,
            luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
            analysis_label=(
                r"Low-$\Delta m$ diagonal-v3 GNN30, nominal+CR+MC stat. (preliminary)"
            ),
            x_max=float(args.max_mstop),
            y_min=200.0,
            y_max=float(args.max_mstop - 100),
            decay_label=DECAY_LABELS[args.topology],
        )
    result = {
        "status": "complete" if contour and limits["status"] == "complete" else "partial",
        "schema_version": "preliminary_lowdm_diagonal_v3_limit_contour_v1",
        "topology": args.topology,
        "model_manifest": str(args.input_dir / "manifest.json"),
        "limits": limits,
        "contour_png": str(output_png) if contour else None,
        "contour_pdf": str(output_png.with_suffix(".pdf")) if contour else None,
        "interpretation": (
            "preliminary expected sensitivity diagnostic; nominal templates, simultaneous "
            "LLCR/QCDCR/GCR constraints, signal luminosity, and MC statistics only"
        ),
    }
    write_json(args.input_dir / "limit_manifest.json", result)
    print(json.dumps({
        "status": result["status"], "topology": args.topology,
        "collected": limits["collected_point_count"],
        "missing": len(limits["missing_points"]), "contour": result["contour_png"],
    }, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
