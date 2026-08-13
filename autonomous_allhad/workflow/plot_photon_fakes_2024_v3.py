#!/usr/bin/env python3
"""Plot the cross-validated photon-fake v3 measurement."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")
PT_CENTERS = np.asarray([260.0, 350.0, 500.0, 700.0])
PT_XERR = np.asarray([40.0, 50.0, 100.0, 100.0])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def cms_label(ax: Any) -> None:
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=ax,
    )


def save(fig: Any, output: Path, stem: str) -> dict[str, str]:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        pass
    fig.subplots_adjust(top=0.82, right=0.96)
    cms_label(fig.axes[0])
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"stem": stem, "png": png.name, "pdf": pdf.name}


def plot_kappa(measurement: dict[str, Any], output: Path) -> dict[str, str]:
    records = measurement["kappa"]["records"]
    fig, ax = plt.subplots(figsize=(10, 10))
    for eta_index, (label, color, marker) in enumerate(
        (("EB", "#4C78A8", "o"), ("EE", "#F58518", "s"))
    ):
        selected = records[eta_index * 4 : (eta_index + 1) * 4]
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
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlim(220.0, 800.0)
    ax.margins(x=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Photon $p_{T}$ (GeV)")
    ax.set_ylabel(r"ABCD correlation correction $\kappa$")
    ax.legend(frameon=False, loc="upper left", fontsize=18)
    ax.text(
        0.03,
        0.04,
        "truth-fake QCD, full sample\n"
        "validation uses opposite-fold prediction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
    )
    return save(fig, output, "kappa_vs_photon_pt")


def plot_closure(
    measurement: dict[str, Any],
    output: Path,
) -> dict[str, str]:
    closure = measurement["closure"]
    edges = np.asarray(
        closure["folds"][0]["target_histogram"]["bin_edges"],
        dtype=float,
    )
    target = np.asarray(closure["recoil_target"], dtype=float)
    prediction = np.asarray(closure["recoil_prediction"], dtype=float)
    target_variance = sum(
        (
            np.asarray(fold["target_histogram"]["sumw2"], dtype=float)
            for fold in closure["folds"]
        ),
        start=np.zeros_like(target),
    )
    prediction_variance = sum(
        (
            np.asarray(fold["prediction_histogram"]["sumw2"], dtype=float)
            for fold in closure["folds"]
        ),
        start=np.zeros_like(prediction),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    ax.stairs(
        prediction,
        edges,
        color="#4C78A8",
        linewidth=2,
        label="Opposite-fold ABCD prediction",
    )
    ax.fill_between(
        edges,
        np.r_[
            np.maximum(0.0, prediction - np.sqrt(prediction_variance)),
            max(0.0, prediction[-1] - np.sqrt(prediction_variance[-1])),
        ],
        np.r_[
            prediction + np.sqrt(prediction_variance),
            prediction[-1] + np.sqrt(prediction_variance[-1]),
        ],
        step="post",
        color="#4C78A8",
        alpha=0.22,
        linewidth=0,
    )
    ax.errorbar(
        centers,
        target,
        xerr=xerr,
        yerr=np.sqrt(target_variance),
        fmt="o",
        color="black",
        label="Truth-fake QCD target",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, loc="upper right", fontsize=18)
    ratio = np.divide(
        target,
        prediction,
        out=np.full_like(target, np.nan),
        where=prediction > 0.0,
    )
    ratio_uncertainty = np.sqrt(
        np.divide(
            target_variance,
            np.square(prediction),
            out=np.zeros_like(target),
            where=prediction > 0.0,
        )
        + np.divide(
            np.square(target) * prediction_variance,
            np.power(prediction, 4),
            out=np.zeros_like(target),
            where=prediction > 0.0,
        )
    )
    ratio_ax.errorbar(
        centers,
        ratio,
        xerr=xerr,
        yerr=ratio_uncertainty,
        fmt="o",
        color="black",
    )
    ratio_ax.axhline(1.0, color="black", linewidth=1)
    ratio_ax.set_ylim(0.0, 3.0)
    ratio_ax.set_ylabel("Target/Pred.")
    ratio_ax.set_xlabel(r"$U_{T}$ (GeV)")
    for axis in (ax, ratio_ax):
        axis.set_xlim(edges[0], edges[-1])
        axis.margins(x=0.0)
    ratio_value = closure["global_target_over_prediction"]
    ax.text(
        0.03,
        0.05,
        f"two-fold closure: target/pred. = {ratio_value:.2f}\n"
        f"assigned nonclosure = {100*closure['assigned_relative_nonclosure']:.0f}%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
    )
    return save(fig, output, "qcd_twofold_closure_ut")


def plot_datamc(
    evaluation: dict[str, Any],
    output: Path,
) -> dict[str, str]:
    record = evaluation["results"]["GCR/ut"]
    edges = np.asarray(record["bin_edges"], dtype=float)
    data = np.asarray(record["data"], dtype=float)
    nominal = np.asarray(record["nominal_prediction"], dtype=float)
    candidate = np.asarray(record["candidate_prediction"], dtype=float)
    candidate_variance = np.asarray(
        record["candidate_prediction_variance"],
        dtype=float,
    )
    fake = np.asarray(record["data_driven_fake"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    ax.stairs(
        nominal,
        edges,
        color="#777777",
        linestyle="--",
        linewidth=2,
        label="Nominal MC",
    )
    ax.stairs(
        candidate,
        edges,
        color="#4C78A8",
        linewidth=2,
        label="MC with measured fake",
    )
    ax.stairs(
        fake,
        edges,
        fill=True,
        color="#E45756",
        alpha=0.35,
        label="Data-driven fake photon",
    )
    ax.fill_between(
        edges,
        np.r_[
            np.maximum(0.0, candidate - np.sqrt(candidate_variance)),
            max(0.0, candidate[-1] - np.sqrt(candidate_variance[-1])),
        ],
        np.r_[
            candidate + np.sqrt(candidate_variance),
            candidate[-1] + np.sqrt(candidate_variance[-1]),
        ],
        step="post",
        color="#4C78A8",
        alpha=0.18,
        linewidth=0,
    )
    ax.errorbar(
        centers,
        data,
        xerr=xerr,
        yerr=np.sqrt(data),
        fmt="o",
        color="black",
        label="Data",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=18)
    for values, color, marker, label in (
        (nominal, "#777777", "s", "Nominal"),
        (candidate, "#4C78A8", "o", "Measured fake"),
    ):
        ratio = np.divide(
            data,
            values,
            out=np.full_like(data, np.nan),
            where=values > 0.0,
        )
        ratio_ax.errorbar(
            centers,
            ratio,
            xerr=xerr,
            fmt=marker,
            color=color,
            label=label,
        )
    ratio_ax.axhline(1.0, color="black", linewidth=1)
    ratio_ax.set_ylim(0.5, 2.0)
    ratio_ax.set_ylabel("Data/MC")
    ratio_ax.set_xlabel(r"$U_{T}$ (GeV)")
    ratio_ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=16)
    for axis in (ax, ratio_ax):
        axis.set_xlim(edges[0], edges[-1])
        axis.margins(x=0.0)
    nominal_ratio = record["metrics"]["nominal"][
        "integral_data_over_prediction"
    ]
    candidate_ratio = record["metrics"]["replace_all_mc_truth_fake"][
        "integral_data_over_prediction"
    ]
    ax.text(
        0.03,
        0.05,
        f"prefit integral Data/MC\n"
        f"nominal {nominal_ratio:.3f}  →  measured fake {candidate_ratio:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
    )
    return save(fig, output, "gcr_ut_prefit_fake_replacement")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    measurement = read_json(args.measurement)
    evaluation = read_json(args.evaluation)
    if measurement.get("schema_version") != "photon_fake_2024_measurement_v3":
        raise RuntimeError("unexpected measurement schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        plot_kappa(measurement, args.output_dir),
        plot_closure(measurement, args.output_dir),
        plot_datamc(evaluation, args.output_dir),
    ]
    closure = measurement["closure"]
    validation = measurement["target_validation"]
    primary = evaluation["results"]["GCR/ut"]["metrics"]
    page = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Photon fake v3 measurement</title>",
        "<style>body{font-family:system-ui;max-width:1050px;margin:2rem auto;"
        "padding:0 1rem}img{max-width:100%;height:auto}code{background:#eee;"
        "padding:.15rem .3rem}</style>",
        "<h1>Photon fake v3: cross-validated correlation-corrected ABCD</h1>",
        "<p>Complete medium-pass/fail sidecar campaign; nominal intermediates "
        "were read only.</p>",
        "<ul>",
        f"<li>Predicted fake yield: <code>{validation['predicted_fake']:.3f}</code></li>",
        f"<li>Two-fold QCD closure target/prediction: "
        f"<code>{closure['global_target_over_prediction']:.3f}</code></li>",
        f"<li>Assigned closure uncertainty: "
        f"<code>{100*closure['assigned_relative_nonclosure']:.0f}%</code></li>",
        f"<li>Prefit GCR U<sub>T</sub> integral Data/MC: "
        f"<code>{primary['nominal']['integral_data_over_prediction']:.3f}</code> "
        f"→ <code>{primary['replace_all_mc_truth_fake']['integral_data_over_prediction']:.3f}</code></li>",
        "</ul>",
    ]
    for record in plots:
        page.extend(
            [
                f"<h2>{html.escape(record['stem'])}</h2>",
                f"<a href='{html.escape(record['pdf'])}'>PDF</a>",
                f"<img src='{html.escape(record['png'])}'>",
            ]
        )
    (args.output_dir / "index.html").write_text("\n".join(page) + "\n")
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "photon_fake_2024_v3_plot_manifest",
                "plots": plots,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "plots": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
