#!/usr/bin/env python3
"""Plot one 2024 CR-to-SR transfer path per figure versus recoil."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")


PROCESS_COMPONENTS = {
    "Top": ("ST", "TT"),
    "WtoLNu": ("WtoLNu",),
    "QCD": ("QCD",),
    "Zto2Nu": ("Zto2Nu",),
    "PhotonJet": ("GJ",),
    "DY": ("DY",),
}
PATHS = (
    {
        "key": "top_llcr",
        "title": "Top lost-lepton transfer factor",
        "label": r"Top: SR / LLCR",
        "numerator_process": "Top",
        "denominator_region": "LLCR",
        "denominator_process": "Top",
        "color": "#D55E00",
    },
    {
        "key": "w_llcr",
        "title": r"$W\!\to\!\ell\nu$ lost-lepton transfer factor",
        "label": r"$W\!\to\!\ell\nu$: SR / LLCR",
        "numerator_process": "WtoLNu",
        "denominator_region": "LLCR",
        "denominator_process": "WtoLNu",
        "color": "#0072B2",
    },
    {
        "key": "qcd_qcdcr",
        "title": "QCD transfer factor",
        "label": r"QCD: SR / QCDCR",
        "numerator_process": "QCD",
        "denominator_region": "QCDCR",
        "denominator_process": "QCD",
        "color": "#CC79A7",
    },
    {
        "key": "zinv_gcr",
        "title": r"$Z(\nu\nu)$ from the photon control region",
        "label": r"$Z(\nu\nu)$ SR / Photon+jet GCR",
        "numerator_process": "Zto2Nu",
        "denominator_region": "GCR",
        "denominator_process": "PhotonJet",
        "color": "#009E73",
    },
    {
        "key": "zinv_dy2e",
        "title": r"$Z(\nu\nu)$ from the dielectron control region",
        "label": r"$Z(\nu\nu)$ SR / DY(ee) CR",
        "numerator_process": "Zto2Nu",
        "denominator_region": "DY2E",
        "denominator_process": "DY",
        "color": "#E69F00",
    },
    {
        "key": "zinv_dy2m",
        "title": r"$Z(\nu\nu)$ from the dimuon control region",
        "label": r"$Z(\nu\nu)$ SR / DY($\mu\mu$) CR",
        "numerator_process": "Zto2Nu",
        "denominator_region": "DY2M",
        "denominator_process": "DY",
        "color": "#56B4E9",
    },
)
LINE_STYLE = {
    "Nb1": "-",
    "Nb2": "--",
    "Nb2plus": "--",
    "Nb3plus": "-.",
}
DISPLAY_GROUP = {
    "Nb1": r"$N_b=1$",
    "Nb2": r"$N_b=2$",
    "Nb2plus": r"$N_b\geq2$",
    "Nb3plus": r"$N_b\geq3$",
}


def leaf(
    recoil: dict[str, Any],
    region: str,
    group: str,
    process: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = None
    variances = None
    for sample in PROCESS_COMPONENTS[process]:
        record = (
            (((recoil.get(region) or {}).get(group) or {}).get(sample) or {})
            .get("nominal")
            or {}
        )
        current = np.asarray(record.get("sumw") or [], dtype=float)
        current_var = np.asarray(record.get("sumw2") or [], dtype=float)
        if values is None:
            values = np.zeros(len(current), dtype=float)
            variances = np.zeros(len(current_var), dtype=float)
        if len(current) != len(values) or len(current_var) != len(variances):
            raise ValueError(f"inconsistent bin count for {region}/{group}/{sample}")
        values += current
        variances += current_var
    if values is None or variances is None:
        raise ValueError(f"missing process {process} in {region}/{group}")
    return values, variances


def calculate_ratio(
    numerator: tuple[np.ndarray, np.ndarray],
    denominator: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    num, num_var = numerator
    den, den_var = denominator
    valid = (num > 0.0) & (den > 0.0)
    ratio = np.full(len(num), np.nan, dtype=float)
    variance = np.full(len(num), np.nan, dtype=float)
    ratio[valid] = num[valid] / den[valid]
    variance[valid] = (
        num_var[valid] / np.square(den[valid])
        + np.square(num[valid]) * den_var[valid] / np.power(den[valid], 4)
    )
    uncertainty = np.sqrt(np.maximum(variance, 0.0))
    return {
        "numerator": num.tolist(),
        "numerator_sumw2": num_var.tolist(),
        "denominator": den.tolist(),
        "denominator_sumw2": den_var.tolist(),
        "valid": valid.tolist(),
        "transfer_factor": [
            float(value) if math.isfinite(value) else None for value in ratio
        ],
        "mcstat": [
            float(value) if math.isfinite(value) else None for value in uncertainty
        ],
    }


def step_values(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[-1]]


def plot_path(
    output_dir: Path,
    path: dict[str, Any],
    regimes: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8), constrained_layout=True)
    for axis, regime_name in zip(axes, ("highdm", "lowdm")):
        regime = regimes[regime_name]
        physical_edges = np.asarray(regime["edges"], dtype=float)
        plot_edges = np.arange(len(physical_edges), dtype=float)
        centers = plot_edges[:-1] + 0.5
        panel_finite: list[float] = []
        for group in regime["groups"]:
            record = regime["records"][path["key"]][group]
            values = np.asarray(
                [
                    value if value is not None else np.nan
                    for value in record["transfer_factor"]
                ],
                dtype=float,
            )
            uncertainty = np.asarray(
                [
                    value if value is not None else np.nan
                    for value in record["mcstat"]
                ],
                dtype=float,
            )
            finite = np.isfinite(values)
            panel_finite.extend(values[finite].tolist())
            axis.step(
                plot_edges,
                step_values(values),
                where="post",
                color=path["color"],
                linestyle=LINE_STYLE[group],
                linewidth=2.8,
                label=DISPLAY_GROUP[group],
            )
            axis.errorbar(
                centers[finite],
                values[finite],
                yerr=uncertainty[finite],
                fmt="none",
                ecolor=path["color"],
                elinewidth=1.6,
                capsize=3.0,
                capthick=1.6,
                alpha=0.8,
            )
        axis.set_xlim(plot_edges[0], plot_edges[-1])
        axis.set_xticks(plot_edges)
        axis.set_xticklabels(
            [f"{edge:g}" for edge in physical_edges[:-1]] + [r"$\infty$"]
        )
        axis.tick_params(axis="x", labelsize=14)
        axis.set_xlabel(r"$p_{T}^{miss}$ / hadronic recoil $U_T$ (GeV)")
        axis.set_ylabel(r"TF = $N_{\mathrm{SR}}/N_{\mathrm{CR}}$")
        axis.grid(alpha=0.2)
        axis.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.91),
        )
        hep.cms.label(
            "Work in progress",
            data=True,
            year="2024",
            com=13.6,
            fontsize=18,
            loc=0,
            ax=axis,
        )
        if panel_finite:
            axis.set_ylim(0.0, max(panel_finite) * 1.35)
    stem = output_dir / f"transfer_factor_{path['key']}_vs_recoil"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    if source.get("status") != "complete":
        raise SystemExit(f"input is not complete: {source.get('status')}")
    regimes: dict[str, Any] = {}
    for regime_name in ("highdm", "lowdm"):
        regime_source = source[regime_name]
        groups = regime_source["nb_groups"]
        recoil = regime_source["recoil"]
        regime_records = {}
        for path in PATHS:
            group_records = {}
            for group in groups:
                group_records[group] = calculate_ratio(
                    leaf(recoil, "SR", group, path["numerator_process"]),
                    leaf(
                        recoil,
                        path["denominator_region"],
                        group,
                        path["denominator_process"],
                    ),
                )
            regime_records[path["key"]] = group_records
        regimes[regime_name] = {
            "edges": regime_source["recoil_edges"],
            "groups": groups,
            "records": regime_records,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in PATHS:
        plot_path(args.output_dir, path, regimes)
    output = {
        "schema_version": "recoil_transfer_factors_2024_v1",
        "status": "complete",
        "definition": "nominal simulated SR target-process yield divided by nominal simulated CR control-process yield in the same Nb and recoil bin",
        "ratio_orientation": "SR_over_CR",
        "uncertainty": "numerator and denominator MC statistical uncertainties propagated in quadrature",
        "paths": [
            {
                key: value
                for key, value in path.items()
                if key != "color"
            }
            for path in PATHS
        ],
        "regimes": regimes,
        "input": args.input.name,
    }
    output_path = args.output_dir / "transfer_factors_2024_nb_recoil.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
