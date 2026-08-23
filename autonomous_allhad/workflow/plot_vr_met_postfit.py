#!/usr/bin/env python3
"""Render CR-only postfit High-dM VR MET templates with the main plot style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from background_process_groups import BACKGROUND_DISPLAY_LABELS
from build_combine_inputs import BACKGROUND_PROCESS_ORDER
from plot_control_search_bins_style import GROUP_ORDER, draw_flat_blocks


REGION_LABELS = {
    "HighDMVR_Nb1": "High-$\\Delta m$ VR, $N_{b}=1$\nCR-only postfit",
    "HighDMVR_Nb2": "High-$\\Delta m$ VR, $N_{b}=2$\nCR-only postfit",
    "HighDMVR_Nb3plus": "High-$\\Delta m$ VR, $N_{b}\\geq3$\nCR-only postfit",
}
LUMINOSITY = {"2024": 109.82, "2025": 110.84, "combined": 220.66}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--page-summary", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    if payload.get("status") != "complete":
        raise SystemExit("VR postfit summary is not complete")
    if payload.get("template_observable") != "met":
        raise SystemExit("only the canonical VR MET template may be plotted")
    if payload.get("vr_observation_in_likelihood") is not False:
        raise SystemExit("VR observation entered the likelihood")
    if payload.get("sr_observation_in_likelihood") is not False:
        raise SystemExit("SR observation entered the likelihood")

    edges = np.asarray(payload["edges"], dtype=float)
    plots = []
    scopes = {
        "2024": payload["years"]["2024"],
        "2025": payload["years"]["2025"],
        "combined": payload["combined"],
    }
    for scope, regions in scopes.items():
        for region_name, record in regions.items():
            groups = {group: np.zeros(len(edges) - 1) for group in GROUP_ORDER}
            for process in BACKGROUND_PROCESS_ORDER:
                group = BACKGROUND_DISPLAY_LABELS[process]
                groups[group] += np.asarray(record["processes"][process], dtype=float)
            total = np.asarray(record["total"], dtype=float)
            group_total = sum(groups.values(), np.zeros_like(total))
            if not np.allclose(group_total, total, rtol=1.0e-10, atol=1.0e-10):
                raise SystemExit(f"process stack does not close for {scope}/{region_name}")
            data = np.asarray(record["data"], dtype=float)
            data_sumw2 = np.asarray(record["data_sumw2"], dtype=float)
            block = {
                "groups": groups,
                "background": total,
                "background_unc": np.asarray(record["uncertainty"], dtype=float),
                "background_stat_unc": np.sqrt(
                    np.maximum(
                        np.asarray(record.get("mc_stat_variance") or np.zeros_like(total)),
                        0.0,
                    )
                ),
                "data": data,
                "data_unc": np.sqrt(np.maximum(data_sumw2, 0.0)),
                "signals": {},
                "label": REGION_LABELS[region_name],
                "annotation": REGION_LABELS[region_name],
                "nbin": len(total),
                "edges": edges.tolist(),
                "xlabels": [],
                "blind_data": False,
                "reference_style": True,
                "label_box": False,
                "show_annotation": True,
                "annotation_x": 0.68,
                "annotation_y": 0.68,
                "group_labels": {},
            }
            outbase = args.output_dir / scope / f"{region_name}_met_postfit"
            plot = draw_flat_blocks(
                [block],
                outbase,
                xlabel=r"$p_{T}^{miss}$ (GeV)",
                reference_style=True,
                show_yields=True,
                ratio_ylabel="Data/Pred.",
                uncertainty_label_override="Total postfit unc.",
                luminosity_fb=LUMINOSITY[scope],
            )
            plot.update(
                {
                    "scope": scope,
                    "region": region_name,
                    "template_observable": "met",
                    "fit": "2024+2025 CR-only background-only",
                }
            )
            plots.append(plot)

    if len(plots) != 9:
        raise SystemExit(f"expected exactly 9 VR MET plots, produced {len(plots)}")

    result = {
        "status": "complete",
        "source": str(args.summary),
        "plot_count": len(plots),
        "plots": plots,
        "template_observable": "met",
        "non_template_variables_excluded": ["njet", "nb", "ntop", "nw"],
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
    }
    args.page_summary.parent.mkdir(parents=True, exist_ok=True)
    args.page_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "plots": len(plots)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
