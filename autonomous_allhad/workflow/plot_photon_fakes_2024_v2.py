#!/usr/bin/env python3
"""Make the AN and prefit-validation plots for photon-fake measurement v2."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v2"
EVALUATION_SCHEMA = "photon_fake_2024_datamc_evaluation_v2"
PT_EDGES = np.asarray([220.0, 300.0, 400.0, 600.0, 800.0])
PT_CENTERS = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])
PT_XERR = 0.5 * (PT_EDGES[1:] - PT_EDGES[:-1])
PROCESS_ORDER = ("VV", "ST", "TT", "DY", "WtoLNu", "Zto2Nu", "GJ", "QCD")
PROCESS_LABELS = {
    "VV": "VV",
    "ST": "Single top",
    "TT": r"$t\bar{t}$",
    "DY": "DY",
    "WtoLNu": r"$W\to\ell\nu$",
    "Zto2Nu": r"$Z\to\nu\nu$",
    "GJ": r"$\gamma$+jets nonfake",
    "QCD": "QCD nonfake",
}
COLORS = {
    "VV": "#64734b",
    "ST": "#8d78bf",
    "TT": "#8eb9aa",
    "DY": "#37bdc4",
    "WtoLNu": "#d9c2a9",
    "Zto2Nu": "#efbd7c",
    "GJ": "#7f007f",
    "QCD": "#d38e9a",
    "fake": "#e45756",
}
hep.style.use("CMS")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def arrays(leaf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(leaf["sumw"], dtype=float),
        np.asarray(leaf["sumw2"], dtype=float),
    )


def save(
    fig: Any,
    output_dir: Path,
    stem: str,
    records: list[dict[str, str]],
    description: str,
) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.88, right=0.96)
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=fig.axes[0],
    )
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)
    records.append(
        {
            "stem": stem,
            "png": png.name,
            "pdf": pdf.name,
            "description": description,
        }
    )


def residual_component(
    audit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(audit["data"], dtype=float)
    prompt = np.asarray(audit["prompt"], dtype=float)
    electron = np.asarray(audit["electron"], dtype=float)
    variance = (
        np.asarray(audit["data_variance"], dtype=float)
        + np.asarray(audit["prompt_variance"], dtype=float)
        + np.asarray(audit["electron_variance"], dtype=float)
    )
    return data - prompt - electron, variance


def plot_abcd_inputs(
    measurement: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, str]],
) -> None:
    audit = measurement["kappa"]["component_audit"]
    definitions = (
        ("A", "pass shower shape, pass charged isolation", "#000000"),
        ("B", "pass shower shape, fail-loose charged isolation", "#4C78A8"),
        ("C", "fail-loose shower shape, pass charged isolation", "#F58518"),
        ("D", "fail-loose shower shape, fail-loose charged isolation", "#54A24B"),
    )
    for eta_index, eta in enumerate(("EB", "EE")):
        slc = slice(eta_index * 4, (eta_index + 1) * 4)
        fig, ax = plt.subplots(figsize=(10, 10))
        for label, description, color in definitions:
            value, variance = residual_component(audit[label])
            ax.errorbar(
                PT_CENTERS,
                value[slc],
                xerr=PT_XERR,
                yerr=np.sqrt(np.maximum(variance[slc], 0.0)),
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=3,
                color=color,
                label=f"{label}: {description}",
            )
        ax.axhline(0.0, color="0.4", linewidth=1)
        ax.set_xlim(PT_EDGES[0], PT_EDGES[-1])
        ax.margins(x=0.0)
        ax.set_xlabel(r"Photon $p_{\mathrm{T}}$ (GeV)", loc="right")
        ax.set_ylabel("Data − prompt photon − electron")
        ax.text(
            0.03,
            0.04,
            f"{eta}\nκ source: 0.30 ≤ min Δφ < 0.50",
            transform=ax.transAxes,
            fontsize=18,
            va="bottom",
        )
        ax.legend(fontsize=13, frameon=False, loc="best")
        save(
            fig,
            output_dir,
            f"abcd_inputs_{eta.lower()}_photon_pt",
            records,
            f"{eta} contamination-subtracted ABCD inputs versus photon pT.",
        )


def plot_factor(
    factor_records: list[dict[str, Any]],
    ylabel: str,
    stem: str,
    output_dir: Path,
    records: list[dict[str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    for eta_index, (eta, color, marker) in enumerate(
        (("EB", "#4C78A8", "o"), ("EE", "#F58518", "s"))
    ):
        selected = factor_records[eta_index * 4 : (eta_index + 1) * 4]
        values = np.asarray([item["factor"] for item in selected], dtype=float)
        errors = np.asarray(
            [item["factor_uncertainty"] for item in selected],
            dtype=float,
        )
        ax.errorbar(
            PT_CENTERS,
            values,
            xerr=PT_XERR,
            yerr=errors,
            marker=marker,
            markersize=8,
            linewidth=2,
            capsize=3,
            color=color,
            label=eta,
        )
    ax.axhline(1.0 if "κ" in ylabel else 0.0, color="0.4", linewidth=1)
    ax.set_xlim(PT_EDGES[0], PT_EDGES[-1])
    ax.margins(x=0.0)
    ax.set_xlabel(r"Photon $p_{\mathrm{T}}$ (GeV)", loc="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    save(
        fig,
        output_dir,
        stem,
        records,
        f"{ylabel} versus photon pT in EB and EE.",
    )


def plot_closure(
    measurement: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, str]],
) -> None:
    closure = measurement["closure"]
    target = closure["target_fake_residual_histogram"]
    prediction = closure["prediction_histogram"]
    target_value, target_variance = arrays(target)
    predicted, predicted_variance = arrays(prediction)
    edges = np.asarray(target["bin_edges"], dtype=float)
    center = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    ax.stairs(
        predicted,
        edges,
        color="#E45756",
        linewidth=2.5,
        label="PLJ prediction",
    )
    ax.fill_between(
        edges,
        np.r_[
            predicted - np.sqrt(np.maximum(predicted_variance, 0.0)),
            (predicted - np.sqrt(np.maximum(predicted_variance, 0.0)))[-1],
        ],
        np.r_[
            predicted + np.sqrt(np.maximum(predicted_variance, 0.0)),
            (predicted + np.sqrt(np.maximum(predicted_variance, 0.0)))[-1],
        ],
        step="post",
        color="#E45756",
        alpha=0.25,
        linewidth=0,
    )
    ax.errorbar(
        center,
        target_value,
        xerr=xerr,
        yerr=np.sqrt(np.maximum(target_variance, 0.0)),
        fmt="o",
        color="black",
        markersize=7,
        capsize=2,
        label="Data − prompt photon − electron",
    )
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    positive = np.r_[target_value[target_value > 0], predicted[predicted > 0]]
    if len(positive):
        ax.set_ylim(max(0.05, float(np.min(positive)) * 0.25), None)
    ax.text(
        0.03,
        0.04,
        "Independent validation region\n0.10 ≤ min Δφ < 0.30",
        transform=ax.transAxes,
        fontsize=18,
        va="bottom",
    )
    ax.legend(frameon=False, fontsize=15)
    ratio = np.divide(
        target_value,
        predicted,
        out=np.full_like(target_value, np.nan),
        where=predicted > 0.0,
    )
    ratio_variance = np.divide(
        target_variance,
        np.square(predicted),
        out=np.zeros_like(target_variance),
        where=predicted > 0.0,
    )
    ratio_variance += np.divide(
        np.square(target_value) * predicted_variance,
        np.power(predicted, 4),
        out=np.zeros_like(target_variance),
        where=predicted > 0.0,
    )
    ratio_ax.errorbar(
        center,
        ratio,
        xerr=xerr,
        yerr=np.sqrt(np.maximum(ratio_variance, 0.0)),
        fmt="o",
        color="black",
        markersize=6,
        capsize=2,
    )
    ratio_ax.axhline(1.0, color="#E45756", linewidth=1.5)
    ratio_ax.set_ylabel("Target/Pred.")
    ratio_ax.set_xlabel(r"$U_{T}$ (GeV)", loc="right")
    ratio_ax.set_xlim(edges[0], edges[-1])
    ratio_ax.margins(x=0.0)
    ratio_ax.set_ylim(0.0, 2.2)
    save(
        fig,
        output_dir,
        "independent_validation_closure_ut",
        records,
        "Independent low-delta-phi closure in U_T.",
    )


def xlabel(
    nominal: dict[str, Any],
    variable: str,
) -> str:
    if variable in {"recoil", "ut"}:
        return r"$U_{T}$ (GeV)"
    if variable == "met":
        return r"$p^{miss}_{T}$ (GeV)"
    spec = (nominal.get("highdm_distribution_variable_specs") or {}).get(
        variable
    )
    if spec is None or not spec.get("xlabel"):
        raise RuntimeError(f"no approved CR/SR xlabel for {variable}")
    return str(spec["xlabel"])


def nominal_leaf(
    nominal: dict[str, Any],
    region: str,
    variable: str,
    process: str,
) -> dict[str, Any]:
    if variable == "recoil":
        return nominal["histograms"][region][process]["nominal"]
    return nominal["highdm_variable_histograms"][region][variable][process][
        "nominal"
    ]


def plot_datamc(
    nominal: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, str]],
) -> None:
    region = str(result["region"])
    variable = str(result["variable"])
    data_leaf = nominal_leaf(nominal, region, variable, "data_obs")
    data, data_variance = arrays(data_leaf)
    edges = np.asarray(result["bin_edges"], dtype=float)
    center = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    candidate = np.asarray(result["candidate_prediction"], dtype=float)
    candidate_variance = np.asarray(
        result["candidate_prediction_variance"], dtype=float
    )
    nominal_prediction = np.asarray(result["nominal_prediction"], dtype=float)
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    bottom = np.zeros_like(data)
    for process in PROCESS_ORDER:
        values = np.asarray(
            result["process_origin_audits"][process][
                "retained_prompt_plus_electron_mc"
            ],
            dtype=float,
        )
        top = bottom + values
        ax.stairs(
            top,
            edges,
            baseline=bottom,
            fill=True,
            color=COLORS[process],
            edgecolor="black",
            linewidth=0.8,
            label=PROCESS_LABELS[process],
        )
        bottom = top
    fake = np.asarray(result["data_driven_fake"], dtype=float)
    ax.stairs(
        bottom + fake,
        edges,
        baseline=bottom,
        fill=True,
        color=COLORS["fake"],
        edgecolor="black",
        linewidth=0.8,
        label="Data-driven fake photon",
    )
    uncertainty = np.sqrt(np.maximum(candidate_variance, 0.0))
    ax.fill_between(
        edges,
        np.r_[
            np.maximum(candidate - uncertainty, 0.0),
            np.maximum(candidate - uncertainty, 0.0)[-1],
        ],
        np.r_[candidate + uncertainty, (candidate + uncertainty)[-1]],
        step="post",
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        label="Pred. unc.",
    )
    ax.stairs(
        nominal_prediction,
        edges,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Nominal MC total",
    )
    ax.errorbar(
        center,
        data,
        xerr=xerr,
        yerr=np.sqrt(np.maximum(data_variance, 0.0)),
        fmt="o",
        color="black",
        markersize=6,
        capsize=2,
        label="Data",
    )
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    positive = np.r_[data[data > 0], candidate[candidate > 0]]
    if len(positive):
        ax.set_ylim(max(0.08, float(np.min(positive)) * 0.25), None)
    ax.text(0.03, 0.04, region, transform=ax.transAxes, fontsize=18, va="bottom")
    ax.legend(
        frameon=False,
        fontsize=11,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
    )
    ratio = np.divide(
        data,
        candidate,
        out=np.full_like(data, np.nan),
        where=candidate > 0.0,
    )
    ratio_error = np.divide(
        np.sqrt(np.maximum(data_variance, 0.0)),
        candidate,
        out=np.zeros_like(data),
        where=candidate > 0.0,
    )
    relative_uncertainty = np.divide(
        uncertainty,
        candidate,
        out=np.zeros_like(candidate),
        where=candidate > 0.0,
    )
    ratio_ax.fill_between(
        edges,
        np.r_[
            1.0 - relative_uncertainty,
            (1.0 - relative_uncertainty)[-1],
        ],
        np.r_[
            1.0 + relative_uncertainty,
            (1.0 + relative_uncertainty)[-1],
        ],
        step="post",
        color="0.65",
        alpha=0.45,
        linewidth=0,
    )
    ratio_ax.errorbar(
        center,
        ratio,
        xerr=xerr,
        yerr=ratio_error,
        fmt="o",
        color="black",
        markersize=6,
        capsize=2,
    )
    ratio_ax.axhline(1.0, color="black", linewidth=1)
    ratio_ax.set_ylabel("Data/Pred.")
    ratio_ax.set_xlabel(xlabel(nominal, variable), loc="right")
    ratio_ax.set_xlim(edges[0], edges[-1])
    ratio_ax.margins(x=0.0)
    ratio_ax.set_ylim(0.0, 2.2)
    save(
        fig,
        output_dir,
        f"{region.lower()}_{variable}_prefit_truth_fake_replacement",
        records,
        (
            f"{region} {variable}: nominal MC compared with the prefit "
            "all-process truth-fake replacement."
        ),
    )


def write_index(
    output_dir: Path,
    records: list[dict[str, str]],
    measurement: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    cards = "\n".join(
        (
            "<section><h2>"
            + html.escape(record["description"])
            + "</h2><a href=\""
            + html.escape(record["png"])
            + "\"><img src=\""
            + html.escape(record["png"])
            + "\" alt=\""
            + html.escape(record["description"])
            + "\"></a><p><a href=\""
            + html.escape(record["pdf"])
            + "\">PDF</a></p></section>"
        )
        for record in records
    )
    closure = measurement.get("closure") or {}
    decision = evaluation.get("decision") or {}
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>2024 photon-fake measurement v2</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
img{{width:100%;max-width:900px;border:1px solid #ddd}}
section{{margin:2.5rem 0}} code{{background:#eee;padding:.15rem .3rem}}
</style></head><body>
<h1>2024 photon-fake measurement v2</h1>
<p>Status: <code>{html.escape(str(measurement.get("status")))}</code>.
Closure: <code>{html.escape(str(closure.get("status")))}</code>,
target/prediction = <code>{html.escape(str(closure.get("target_over_prediction")))}</code>.
Prefit decision: <code>{html.escape(str(decision.get("replace_all_mc_truth_fake")))}</code>.</p>
{cards}
</body></html>
"""
    (output_dir / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    measurement = read_json(args.measurement)
    evaluation = read_json(args.evaluation)
    nominal = read_json(args.nominal)
    if measurement.get("schema_version") != MEASUREMENT_SCHEMA:
        raise RuntimeError("not a photon-fake v2 measurement")
    if evaluation.get("schema_version") != EVALUATION_SCHEMA:
        raise RuntimeError("not a photon-fake v2 Data/MC evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    plot_abcd_inputs(measurement, args.output_dir, records)
    plot_factor(
        measurement["kappa"]["records"],
        r"ABCD correlation correction $\kappa$",
        "kappa_vs_photon_pt",
        args.output_dir,
        records,
    )
    plot_factor(
        measurement["measurement"]["central_plj_factors"],
        "PLJ extrapolation factor",
        "plj_factor_vs_photon_pt",
        args.output_dir,
        records,
    )
    plot_closure(measurement, args.output_dir, records)
    for key in sorted(evaluation["results"]):
        plot_datamc(
            nominal,
            evaluation["results"][key],
            args.output_dir,
            records,
        )
    write_index(args.output_dir, records, measurement, evaluation)
    write_json(
        args.output_dir / "plot_manifest.json",
        {
            "schema_version": "photon_fake_2024_v2_plot_manifest",
            "measurement": str(args.measurement),
            "evaluation": str(args.evaluation),
            "nominal": str(args.nominal),
            "plots": records,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "plots": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
