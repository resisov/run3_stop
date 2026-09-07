#!/usr/bin/env python3
"""Plot High-dM recoil and Low-dM GNN transfer factors from histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from gnn_background_histograms import require_gnn_mapping


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
GNN_PATHS = PATHS + ({
    "key": "zinv_gcr",
    "ratio_label": r"Raw $Z\!\to\!\nu\nu$: SR / $\gamma$ CR",
},)

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
    *,
    xlabel: str = r"$U_T$ (GeV)",
    annotation: str | None = None,
    output_suffix: str | None = None,
    ylabel: str = r"Transfer factor $N_{\mathrm{SR}}/N_{\mathrm{CR}}$",
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
    axis.set_xlabel(xlabel, fontsize=30)
    axis.set_ylabel(ylabel, fontsize=29)
    axis.tick_params(labelsize=24)
    axis.grid(alpha=0.16)
    axis.text(
        0.04,
        0.73 if path["key"] == "qcd_qcdcr" else 0.07,
        annotation if annotation is not None else ("High-" if regime == "highdm" else "Low-")
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
        output_dir / f"transfer_factor_{path['key']}_{output_suffix or regime}",
    )




def build_factors(source: dict[str, Any]) -> dict[str, Any]:
    mapping = require_gnn_mapping(source)
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

    return {
        "highdm": {"edges": high_edges.tolist(), "records": high_records},
        "lowdm": {
            "kind": "gnn", "axis": "GNN output",
            "sr_binning": mapping["sr_binning"], "cr_binning": mapping["cr_binning"],
            "records": mapping["transfer_factors"]["nominal"],
            "ut_fallback_allowed": False,
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


    output = {
        "schema_version": f"template_transfer_factors_{args.campaign_year}_v3",
        "status": "complete",
        "definition": (
            "High-dM: SR/CR in the same category and U_T bin. "
            "Low-dM: SR GNN score bin / mapped GNN CR parent integral. "
            "The raw Zinv/GJ coefficient does not yet contain RZ or Sgamma."
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
            "lowdm_mode": "GNN30 final templates only; no UT or legacy-search-bin fallback",
            "plot_regime": args.regime,
            "campaign_year": args.campaign_year,
        },
        "mechanical_checks": {
            "max_relative_residual": max_residual,
        },
        "plots": plot_paths,
    }
    mapping = require_gnn_mapping(source)
    gnn_plots = []
    if args.regime in {"all", "lowdm"}:
        (args.output_dir / "gnn").mkdir(parents=True, exist_ok=True)
        for path in GNN_PATHS:
            for category, record in factors["lowdm"]["records"][path["key"]].items():
                gnn_plots.extend(plot_highdm(
                    args.output_dir / "gnn", path,
                    np.asarray(record["score_edges"]), {record["nb_group"]: record},
                    regime="lowdm", xlabel="GNN output",
                    annotation="Low-" + r"$\Delta m$" + "\n" + category.replace("_", " ") + "\n" + path["ratio_label"],
                    output_suffix=f"lowdm_gnn_{category}",
                    ylabel=r"Transfer factor $N_{\mathrm{SR,bin}}/N_{\mathrm{CR,parent}}$",
                ))
    output["lowdm_gnn"] = {
        key: mapping[key] for key in ("schema_version", "axis", "sr_binning", "cr_binning", "definition", "mechanical_checks", "provenance")
    }
    output["lowdm_gnn"]["transfer_factors"] = mapping["transfer_factors"]["nominal"]
    output["lowdm_gnn"]["plots"] = gnn_plots
    plot_paths.extend(gnn_plots)

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
