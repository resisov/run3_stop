#!/usr/bin/env python3
"""Collect and plot the orthogonal 2024 High-dM(79-6) + Low-dM(30) limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKFLOW = THIS_DIR.parent.parent / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from postprocess_limits import DECAY_LABELS, collect_limits, plot_contour  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--extra-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run2-dir", type=Path, required=True)
    parser.add_argument("--tag", default="highdm79minus6_lowdm30_allsignal")
    parser.add_argument("--x-max", type=float, default=1800.0)
    parser.add_argument("--y-min", type=float, default=1.0)
    parser.add_argument("--y-max-onshell", type=float, default=1200.0)
    parser.add_argument("--y-max-offshell", type=float, default=1700.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.campaign / "campaign_manifest.json").read_text())
    extra_manifest = json.loads(
        (args.extra_campaign / "campaign_manifest.json").read_text()
    )
    summary = {
        "schema_version": "highdm79minus6_lowdm30_limit_results_v1",
        "status": "complete",
        "campaign": str(args.campaign),
        "extra_campaign": str(args.extra_campaign),
        "model": "High-dM 79 bins with overlapping first 6 replaced by Low-dM 30 bins",
        "combined_sr_bin_count": 103,
        "topologies": {},
    }
    for topology in ("T2tt", "T2bW", "T2tb"):
        masses = manifest["topologies"][topology]["mass_points"]
        target = args.campaign / topology
        combined = collect_limits(
            target / "limits",
            masses,
            args.output / topology / "combined_expected_limits.json",
        )
        extra_masses = extra_manifest["topologies"][topology]["mass_points"]
        extra_target = args.extra_campaign / topology
        extra = collect_limits(
            extra_target / "limits",
            extra_masses,
            args.output / topology / "lowdm_only_extra_expected_limits.json",
        )
        all_masses = list(dict.fromkeys([*masses, *extra_masses]))
        points = dict(extra["points"])
        points.update(combined["points"])
        missing = [mass for mass in all_masses if mass not in points]
        result = {
            "status": "complete" if not missing else "partial",
            "points": points,
            "requested_point_count": len(all_masses),
            "collected_point_count": len(points),
            "missing_points": missing,
            "source_priority": "orthogonal High-dM(79-6)+Low-dM30 when available; Low-dM30-only otherwise",
            "combined_point_count": len(combined["points"]),
            "lowdm_only_extra_point_count": len(extra["points"]),
        }
        write_json(args.output / topology / "expected_limits.json", result)
        if result["status"] != "complete":
            summary["status"] = "partial"
        plots = {}
        for suffix, mask_offshell, y_max in (
            ("", False, args.y_max_offshell),
            ("_onshell", True, args.y_max_onshell),
        ):
            output = args.output / (
                f"expected_limit_{topology.lower()}_{args.tag}{suffix}.png"
            )
            ok = plot_contour(
                result,
                output,
                run2_contours=args.run2_dir / f"run2_{topology.lower()}.json",
                luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
                analysis_label=None,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=y_max,
                decay_label=DECAY_LABELS[topology],
                mask_offshell=mask_offshell,
            )
            plots["offshell" if not mask_offshell else "onshell"] = (
                str(output) if ok else None
            )
        summary["topologies"][topology] = {
            "status": result["status"],
            "requested_point_count": result["requested_point_count"],
            "collected_point_count": result["collected_point_count"],
            "missing_point_count": len(result["missing_points"]),
            "combined_point_count": len(combined["points"]),
            "lowdm_only_extra_point_count": len(extra["points"]),
            "plots": plots,
        }
    write_json(args.output / "combined_limit_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
