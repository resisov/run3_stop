#!/usr/bin/env python3
"""Calculate and plot 2024 Top, W, and QCD CR-to-SR transfer factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")

CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}
PROCESS_COMPONENTS = {
    "Top": ("ST", "TT"),
    "WtoLNu": ("WtoLNu",),
    "QCD": ("QCD",),
}
PATHS = (
    {
        "key": "top_llcr",
        "display": "Top",
        "numerator_process": "Top",
        "denominator_region": "LLCR",
        "denominator_process": "Top",
        "ratio_label": r"Top: SR / LLCR",
    },
    {
        "key": "w_llcr",
        "display": r"$W\!\to\!\ell\nu$",
        "numerator_process": "WtoLNu",
        "denominator_region": "LLCR",
        "denominator_process": "WtoLNu",
        "ratio_label": r"$W\!\to\!\ell\nu$: SR / LLCR",
    },
    {
        "key": "qcd_qcdcr",
        "display": "QCD",
        "numerator_process": "QCD",
        "denominator_region": "QCDCR",
        "denominator_process": "QCD",
        "ratio_label": "QCD: SR / QCDCR",
    },
)
HIGH_STYLES = {
    "inclusive": ("o", "#E41A1C", "Inclusive"),
    "Nb1": ("o", "#E41A1C", r"$N_b=1$"),
    "Nb2plus": ("s", "#0057FF", r"$N_b\geq2$"),
    "Nb2": ("s", "#0057FF", r"$N_b=2$"),
    "Nb3plus": ("^", "#168B38", r"$N_b\geq3$"),
}
LOW_PANEL_SPECS = (
    (
        r"$N_b=1$" "\n" r"$300\leq p_T^{ISR}<500$",
        "PISR300to500",
        (
            ("Nb1_PISR300to500_PTb20to40", r"$20<p_T^b<40$", "o", "#E41A1C"),
            ("Nb1_PISR300to500_PTb40to70", r"$40<p_T^b<70$", "s", "#0057FF"),
        ),
    ),
    (
        r"$N_b=1$" "\n" r"$p_T^{ISR}\geq500$",
        "PISR500plus",
        (
            ("Nb1_PISR500plus_PTb20to40", r"$20<p_T^b<40$", "o", "#E41A1C"),
            ("Nb1_PISR500plus_PTb40to70", r"$40<p_T^b<70$", "s", "#0057FF"),
        ),
    ),
    (
        r"$N_b\geq2$" "\n" r"$300\leq p_T^{ISR}<500$",
        "PISR300to500",
        (
            ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", r"$40<p_T^b<80,\ N_j\geq2$", "o", "#E41A1C"),
            ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", r"$80<p_T^b<140,\ N_j\geq2$", "s", "#0057FF"),
        ),
    ),
    (
        r"$N_b\geq2$" "\n" r"$p_T^{ISR}\geq500$",
        "PISR500plus",
        (
            ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", r"$40<p_T^b<80,\ N_j\geq2$", "o", "#E41A1C"),
            ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", r"$80<p_T^b<140,\ N_j\geq2$", "s", "#0057FF"),
        ),
    ),
    (
        r"$N_b\geq2$" "\n" r"$300\leq p_T^{ISR}<500$",
        "PISR300to500",
        (
            ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", r"$p_T^b>140,\ N_j\geq7$", "^", "#168B38"),
        ),
    ),
    (
        r"$N_b\geq2$" "\n" r"$p_T^{ISR}\geq500$",
        "PISR500plus",
        (
            ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", r"$p_T^b>140,\ N_j\geq7$", "^", "#168B38"),
        ),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nominal_leaf(
    source: dict[str, Any],
    region: str,
    group: str,
    process: str,
    nbin: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(nbin, dtype=float)
    variances = np.zeros(nbin, dtype=float)
    by_sample = ((source.get(region) or {}).get(group) or {})
    for sample in PROCESS_COMPONENTS[process]:
        nominal = ((by_sample.get(sample) or {}).get("nominal") or {})
        current = np.asarray(nominal.get("sumw") or [0.0] * nbin, dtype=float)
        current_var = np.asarray(
            nominal.get("sumw2") or [0.0] * nbin,
            dtype=float,
        )
        if len(current) != nbin or len(current_var) != nbin:
            raise ValueError(
                f"inconsistent bin count for {region}/{group}/{sample}: "
                f"{len(current)}/{len(current_var)} != {nbin}"
            )
        values += current
        variances += current_var
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
    residual = np.full(len(num), np.nan, dtype=float)
    residual[valid] = (
        ratio[valid] * den[valid] - num[valid]
    ) / num[valid]
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
            float(value) if math.isfinite(value) else None
            for value in uncertainty
        ],
        "mechanical_relative_residual": [
            float(value) if math.isfinite(value) else None for value in residual
        ],
    }


def low_family(label: str) -> str:
    return re.sub(r"_recoil_[0-9]+$", "", label)


def low_geometry(
    isr_group: str,
    nbin: int,
    overflow_cap: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray]:
    lower = 300.0 if isr_group == "PISR300to500" else 450.0
    edges = lower + 100.0 * np.arange(nbin + 1, dtype=float)
    if overflow_cap <= edges[-2]:
        raise ValueError("overflow cap must exceed the final finite Low-dM edge")
    edges[-1] = overflow_cap
    return 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)


def finite_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(
        [np.nan if value is None else value for value in record["transfer_factor"]],
        dtype=float,
    )
    errors = np.asarray(
        [np.nan if value is None else value for value in record["mcstat"]],
        dtype=float,
    )
    return values, errors, np.isfinite(values) & np.isfinite(errors)


def set_tf_ylim(axis: plt.Axes, records: list[dict[str, Any]]) -> None:
    upper: list[float] = []
    for record in records:
        values, errors, valid = finite_arrays(record)
        upper.extend((values[valid] + errors[valid]).tolist())
    ymax = max(upper, default=1.0)
    axis.set_ylim(0.0, max(1.2 * ymax, 0.05))


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 180}),
        (".pdf", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_highdm(
    output_dir: Path,
    path: dict[str, Any],
    edges: np.ndarray,
    records: dict[str, Any],
    regime: str = "highdm",
) -> list[str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    fig, axis = plt.subplots(figsize=(10.2, 10.2))
    used: list[dict[str, Any]] = []
    group_order = [
        group
        for group in ("inclusive", "Nb1", "Nb2plus", "Nb2", "Nb3plus")
        if group in records
    ]
    for group in group_order:
        record = records[group]
        values, errors, valid = finite_arrays(record)
        marker, color, label = HIGH_STYLES[group]
        axis.errorbar(
            centers[valid],
            values[valid],
            xerr=widths[valid],
            yerr=errors[valid],
            fmt=marker,
            ls="none",
            color=color,
            lw=2.5,
            ms=9.5,
            mew=1.5,
            capsize=3.2,
            label=label,
        )
        used.append(record)
    set_tf_ylim(axis, used)
    axis.set_xlim(float(edges[0]), float(edges[-1]))
    axis.set_xmargin(0)
    axis.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
    axis.set_ylabel(r"Transfer factor $N_{\mathrm{SR}}/N_{\mathrm{CR}}$", fontsize=29)
    axis.tick_params(labelsize=24)
    axis.grid(alpha=0.16)
    axis.text(
        0.04,
        0.73 if path["key"] == "qcd_qcdcr" else 0.07,
        ("High-" if regime == "highdm" else "Low-")
        + r"$\Delta m$"
        + "\n"
        + path["ratio_label"],
        transform=axis.transAxes,
        fontsize=22,
    )
    axis.legend(
        frameon=False,
        fontsize=24,
        markerscale=1.25,
        handlelength=1.8,
        labelspacing=0.7,
    )
    hep.cms.label(**CMS_LABEL, ax=axis)
    return save_figure(
        fig,
        output_dir / f"transfer_factor_{path['key']}_{regime}",
    )


def plot_lowdm(
    output_dir: Path,
    path: dict[str, Any],
    families: dict[str, Any],
) -> list[str]:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0))
    for panel_index, (axis, (annotation, isr_group, series)) in enumerate(
        zip(axes.flat, LOW_PANEL_SPECS)
    ):
        panel_records: list[dict[str, Any]] = []
        for family, label, marker, color in series:
            record = families[family]
            values, errors, valid = finite_arrays(record)
            centers, widths = low_geometry(isr_group, len(values))
            axis.errorbar(
                centers[valid],
                values[valid],
                xerr=widths[valid],
                yerr=errors[valid],
                fmt=marker,
                ls="none",
                color=color,
                lw=2.0,
                ms=7.8,
                mew=1.3,
                capsize=2.5,
                label=label,
            )
            panel_records.append(record)
        set_tf_ylim(axis, panel_records)
        axis.set_xlim(300.0 if isr_group == "PISR300to500" else 450.0, 1500.0)
        axis.set_xmargin(0)
        axis.tick_params(labelsize=14)
        axis.grid(alpha=0.16)
        axis.text(
            0.04,
            0.06,
            annotation,
            transform=axis.transAxes,
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )
        if panel_index == 0:
            axis.text(
                0.04,
                0.88,
                "Low-" + r"$\Delta m$" + "\n" + path["ratio_label"],
                transform=axis.transAxes,
                fontsize=13,
                va="top",
            )
        axis.legend(frameon=False, fontsize=12.5, loc="upper right")
    fig.supxlabel(r"$U_T$ (GeV)", fontsize=25, x=0.57, y=0.045)
    fig.supylabel(
        r"Transfer factor $N_{\mathrm{SR}}/N_{\mathrm{CR}}$",
        fontsize=24,
        x=0.035,
        y=0.52,
    )
    label_axis = fig.add_subplot(111, frameon=False)
    label_axis.set_xticks([])
    label_axis.set_yticks([])
    label_axis.patch.set_visible(False)
    label_axis.set_zorder(-1)
    hep.cms.label(**CMS_LABEL, ax=label_axis)
    return save_figure(
        fig,
        output_dir / f"transfer_factor_{path['key']}_lowdm",
    )


def build_factors(source: dict[str, Any]) -> dict[str, Any]:
    high_source = source["highdm"]["recoil"]
    high_edges = np.asarray(source["highdm"]["recoil_edges"], dtype=float)
    high_nbin = len(high_edges) - 1
    high_records: dict[str, Any] = {}
    for path in PATHS:
        by_group = {}
        for group in source["highdm"]["nb_groups"]:
            by_group[group] = calculate_ratio(
                nominal_leaf(
                    high_source,
                    "SR",
                    group,
                    path["numerator_process"],
                    high_nbin,
                ),
                nominal_leaf(
                    high_source,
                    path["denominator_region"],
                    group,
                    path["denominator_process"],
                    high_nbin,
                ),
            )
        high_records[path["key"]] = by_group

    low_payload = source["lowdm"]
    low_records: dict[str, Any] = {}
    low_kind = "search_bins"
    if "recoil" in low_payload:
        low_kind = "nb_recoil"
        low_source = low_payload["recoil"]
        low_edges = np.asarray(low_payload["recoil_edges"], dtype=float)
        low_nbin = len(low_edges) - 1
        for path in PATHS:
            low_records[path["key"]] = {
                group: calculate_ratio(
                    nominal_leaf(
                        low_source,
                        "SR",
                        group,
                        path["numerator_process"],
                        low_nbin,
                    ),
                    nominal_leaf(
                        low_source,
                        path["denominator_region"],
                        group,
                        path["denominator_process"],
                        low_nbin,
                    ),
                )
                for group in low_payload["nb_groups"]
            }
    else:
        labels = list(low_payload["search_bin_labels"])
        low_source = low_payload["search_components"]
        low_nbin = len(labels)
        family_indices: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            family_indices.setdefault(low_family(label), []).append(index)
        for path in PATHS:
            by_family = {}
            for family, indices in family_indices.items():
                group = "Nb1" if family.startswith("Nb1_") else "Nb2plus"
                full = calculate_ratio(
                    nominal_leaf(
                        low_source,
                        "SR",
                        group,
                        path["numerator_process"],
                        low_nbin,
                    ),
                    nominal_leaf(
                        low_source,
                        path["denominator_region"],
                        group,
                        path["denominator_process"],
                        low_nbin,
                    ),
                )
                by_family[family] = {
                    key: [values[index] for index in indices]
                    for key, values in full.items()
                }
            low_records[path["key"]] = by_family
    return {
        "highdm": {"edges": high_edges.tolist(), "records": high_records},
        "lowdm": {
            "kind": low_kind,
            **(
                {"edges": low_edges.tolist()}
                if low_kind == "nb_recoil"
                else {"search_bin_labels": labels}
            ),
            "records": low_records,
        },
    }


def validate_input(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("status") != "complete":
        raise ValueError(f"input is not complete: {source.get('status')}")
    provenance = source.get("provenance") or {}
    regions = set(provenance.get("regions") or [])
    required = {"SR", "LLCR", "QCDCR"}
    if not required.issubset(regions):
        raise ValueError(f"required regions absent: {sorted(required - regions)}")
    if provenance.get("include_data"):
        raise ValueError("TF input must be MC-only; SR data must remain blinded")
    datasets = sorted((source.get("summary") or {}).get("datasets") or {})
    data_records = [name for name in datasets if "Run2024" in name]
    if data_records:
        raise ValueError(
            "TF input summary contains data despite MC-only provenance: "
            f"{data_records[:3]}"
        )
    forbidden_ptll = [name for name in datasets if "PTLL" in name.upper()]
    if forbidden_ptll:
        raise ValueError(f"forbidden PTLL DY inputs found: {forbidden_ptll[:3]}")
    qcd = [name for name in datasets if name.startswith("QCD")]
    histogram_derived = bool(provenance.get("histogram_derived"))
    if datasets and (not qcd or any(not name.startswith("QCD-4Jets_Bin-HT-") for name in qcd)):
        raise ValueError("QCD inputs are not exclusively QCD-4Jets HT bins")
    required_dy = {
        "DYto2E-4Jets": any("DYto2E-4Jets" in name for name in datasets),
        "DYto2Mu-4Jets": any("DYto2Mu-4Jets" in name for name in datasets),
        "DYto2Tau-4Jets": any("DYto2Tau-4Jets" in name for name in datasets),
    }
    missing_dy = [name for name, present in required_dy.items() if not present]
    if datasets and missing_dy:
        raise ValueError(f"required DY2x samples absent: {missing_dy}")
    if histogram_derived:
        policy = provenance.get("sample_policy") or {}
        if "HT" not in str(policy.get("qcd", "")):
            raise ValueError("histogram-derived input does not record the QCD HT policy")
        if "PTLL excluded" not in str(policy.get("dy", "")):
            raise ValueError("histogram-derived input does not record PTLL exclusion")
    return {
        "dataset_records": len(datasets),
        "forbidden_ptll_count": len(forbidden_ptll),
        "qcd_family": "QCD-4Jets HT only",
        "required_dy2x_present": required_dy if datasets else "recorded in histogram provenance",
        "histogram_derived": histogram_derived,
        "sr_data_recorded": False,
        "data_dataset_records": len(data_records),
        "top_definition": "TT + ST",
    }


def max_mechanical_residual(factors: dict[str, Any]) -> float:
    residuals = []
    for regime in factors.values():
        records = regime.get("records") or {}
        for by_category in records.values():
            for record in by_category.values():
                residuals.extend(
                    abs(value)
                    for value in record["mechanical_relative_residual"]
                    if value is not None
                )
    return max(residuals, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-year",
        choices=("2024", "2025"),
        default="2024",
        help="Campaign year used in labels and output metadata.",
    )
    parser.add_argument(
        "--regime",
        choices=("all", "highdm", "lowdm"),
        default="all",
        help="Limit plot regeneration while retaining both regimes in the JSON.",
    )
    args = parser.parse_args()
    CMS_LABEL["rlabel"] = f"{args.campaign_year} (13.6 TeV)"

    source = json.loads(args.input.read_text())
    sample_check = validate_input(source)
    factors = build_factors(source)
    max_residual = max_mechanical_residual(factors)
    if max_residual > 1.0e-12:
        raise ValueError(f"TF mechanical residual too large: {max_residual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    high_edges = np.asarray(factors["highdm"]["edges"], dtype=float)
    for path in PATHS:
        if args.regime in {"all", "highdm"}:
            plot_paths.extend(
                plot_highdm(
                    args.output_dir,
                    path,
                    high_edges,
                    factors["highdm"]["records"][path["key"]],
                )
            )
        if args.regime in {"all", "lowdm"}:
            if factors["lowdm"].get("kind") == "nb_recoil":
                plot_paths.extend(
                    plot_highdm(
                        args.output_dir,
                        path,
                        np.asarray(factors["lowdm"]["edges"], dtype=float),
                        factors["lowdm"]["records"][path["key"]],
                        regime="lowdm",
                    )
                )
            else:
                plot_paths.extend(
                    plot_lowdm(
                        args.output_dir,
                        path,
                        factors["lowdm"]["records"][path["key"]],
                    )
                )

    output = {
        "schema_version": f"recoil_transfer_factors_{args.campaign_year}_v2",
        "status": "complete",
        "definition": (
            "nominal simulated SR target-process yield divided by nominal "
            "simulated CR target-process yield in the same category and U_T bin"
        ),
        "ratio_orientation": "SR_over_CR",
        "uncertainty": (
            "numerator and denominator MC statistical uncertainties propagated "
            "in quadrature"
        ),
        "paths": list(PATHS),
        "factors": factors,
        "provenance": {
            "input": str(args.input),
            "input_sha256": file_sha256(args.input),
            "input_provenance": source.get("provenance"),
            "sample_check": sample_check,
            "lowdm_mode": "all 34 adopted search bins; no category aggregation",
            "lowdm_plot_overflow_cap_gev": 1500.0,
            "plot_regime": args.regime,
            "campaign_year": args.campaign_year,
        },
        "mechanical_checks": {
            "max_relative_residual": max_residual,
        },
        "plots": plot_paths,
    }
    output_path = (
        args.output_dir
        / f"transfer_factors_{args.campaign_year}_nb_recoil.json"
    )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path),
                "plots": plot_paths,
                "max_mechanical_residual": max_residual,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
