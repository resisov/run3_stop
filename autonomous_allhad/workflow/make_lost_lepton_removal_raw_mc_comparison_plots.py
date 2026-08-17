#!/usr/bin/env python3
"""Compare event-removal predictions with raw Top+W MC and data residuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import analyze_lost_lepton_removal_closure_2024 as removal_analysis  # noqa: E402


COLORS = {
    "raw": "#2f6fae",
    "removal": "#c43c39",
    "data": "#111111",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def setup_style() -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 9.0),
            "axes.labelsize": 21,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "savefig.bbox": None,
            "figure.dpi": 120,
        }
    )


def cms_label(axis: Any) -> None:
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        ax=axis,
    )


def display_labels(labels: list[str]) -> list[str]:
    rendered = []
    for label in labels:
        if label.endswith("plus") and label[:-4].isdigit():
            rendered.append(rf"$\geq {label[:-4]}$")
        else:
            rendered.append(label.replace("-", "\N{EN DASH}"))
    return rendered


def step_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(len(values) + 1, dtype=float)
    return edges, np.r_[values, values[-1] if len(values) else 0.0]


def ratio_and_error(
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.divide(
        prediction,
        observation,
        out=np.full_like(prediction, np.nan),
        where=observation != 0.0,
    )
    variance = np.divide(
        prediction_variance,
        np.square(observation),
        out=np.zeros_like(prediction),
        where=observation != 0.0,
    )
    variance += np.divide(
        np.square(prediction) * observation_variance,
        np.power(observation, 4),
        out=np.zeros_like(prediction),
        where=observation != 0.0,
    )
    return ratio, np.sqrt(np.clip(variance, 0.0, None))


def integrated_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
) -> float:
    return float(np.sum(prediction) / np.sum(observation))


def raw_mc_histogram(
    merged: dict[str, Any],
    scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    top, top_variance = removal_analysis.full_hist(
        merged,
        "Top",
        scheme,
        "target_inclusive",
    )
    wjets, wjets_variance = removal_analysis.full_hist(
        merged,
        "W",
        scheme,
        "target_inclusive",
    )
    return top + wjets, top_variance + wjets_variance


def comparison_record(
    merged: dict[str, Any],
    results: dict[str, Any],
    scheme: str,
) -> dict[str, Any]:
    record = results["data_validation"][scheme]
    raw, raw_variance = raw_mc_histogram(merged, scheme)
    observation = np.asarray(record["target"], dtype=float)
    observation_variance = np.asarray(record["target_variance"], dtype=float)
    removal = np.asarray(record["prediction"], dtype=float)
    removal_variance = np.asarray(record["prediction_variance"], dtype=float)
    raw_ratio, raw_ratio_error = ratio_and_error(
        raw,
        raw_variance,
        observation,
        observation_variance,
    )
    raw_pull_variance = raw_variance + observation_variance
    raw_pull = np.divide(
        raw - observation,
        np.sqrt(raw_pull_variance),
        out=np.full_like(raw, np.nan),
        where=raw_pull_variance > 0.0,
    )
    return {
        "scheme": scheme,
        "labels": merged["schemes"][scheme]["labels"],
        "raw_mc": raw.tolist(),
        "raw_mc_variance": raw_variance.tolist(),
        "raw_mc_ratio": raw_ratio.tolist(),
        "raw_mc_ratio_error": raw_ratio_error.tolist(),
        "raw_mc_integrated_ratio": integrated_ratio(raw, observation),
        "raw_mc_maximum_absolute_pull": float(np.nanmax(np.abs(raw_pull))),
        "removal_prediction": removal.tolist(),
        "removal_prediction_variance": removal_variance.tolist(),
        "removal_integrated_ratio": integrated_ratio(removal, observation),
        "removal_maximum_absolute_pull": record["maximum_absolute_pull"],
        "data_residual": observation.tolist(),
        "data_residual_variance": observation_variance.tolist(),
    }


def plot_comparison(
    record: dict[str, Any],
    xlabel: str,
    output_dir: Path,
    stem: str,
    log_scale: bool,
) -> None:
    raw = np.asarray(record["raw_mc"], dtype=float)
    raw_variance = np.asarray(record["raw_mc_variance"], dtype=float)
    removal = np.asarray(record["removal_prediction"], dtype=float)
    removal_variance = np.asarray(
        record["removal_prediction_variance"],
        dtype=float,
    )
    observation = np.asarray(record["data_residual"], dtype=float)
    observation_variance = np.asarray(
        record["data_residual_variance"],
        dtype=float,
    )
    labels = display_labels(record["labels"])
    centers = np.arange(len(raw), dtype=float) + 0.5

    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 9.0),
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.16,
        top=0.88,
        hspace=0.05,
    )
    for values, variance, color, linestyle, label, band_label in (
        (
            raw,
            raw_variance,
            COLORS["raw"],
            "-",
            "Raw Top + W MC",
            "Raw MC stat. unc.",
        ),
        (
            removal,
            removal_variance,
            COLORS["removal"],
            "--",
            "Event-removal prediction",
            "Removal prediction unc.",
        ),
    ):
        edges, stepped = step_values(values)
        _, low = step_values(
            np.clip(values - np.sqrt(np.clip(variance, 0.0, None)), 0.0, None)
        )
        _, high = step_values(values + np.sqrt(np.clip(variance, 0.0, None)))
        axis.step(
            edges,
            stepped,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.3,
            label=label,
        )
        axis.fill_between(
            edges,
            low,
            high,
            step="post",
            color=color,
            alpha=0.16,
            linewidth=0,
            label=band_label,
        )
    axis.errorbar(
        centers,
        observation,
        yerr=np.sqrt(np.clip(observation_variance, 0.0, None)),
        fmt="o",
        color=COLORS["data"],
        markersize=4.7,
        label="Data - Other MC",
    )
    axis.set_xlim(0.0, float(len(raw)))
    axis.set_ylabel("Events")
    if log_scale:
        axis.set_yscale("log")
        positive = observation[observation > 0.0]
        axis.set_ylim(
            bottom=max(float(np.min(positive)) * 0.2, 1e-2),
            top=max(float(np.max(observation)), float(np.max(raw))) * 8.0,
        )
    else:
        axis.set_ylim(
            0.0,
            1.45
            * max(
                float(np.max(observation)),
                float(np.max(raw)),
                float(np.max(removal)),
            ),
        )
    axis.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.92)
    axis.text(
        0.03,
        0.06,
        (
            "Integrated prediction / data residual\n"
            f"Raw MC = {record['raw_mc_integrated_ratio']:.3f}\n"
            f"Event removal = {record['removal_integrated_ratio']:.3f}"
        ),
        transform=axis.transAxes,
        fontsize=13,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
            "pad": 2.0,
        },
    )
    cms_label(axis)

    finite_maximum = 1.0
    for values, variance, color, marker, label in (
        (raw, raw_variance, COLORS["raw"], "o", "Raw MC"),
        (
            removal,
            removal_variance,
            COLORS["removal"],
            "s",
            "Event removal",
        ),
    ):
        ratio, ratio_error = ratio_and_error(
            values,
            variance,
            observation,
            observation_variance,
        )
        finite = np.isfinite(ratio) & np.isfinite(ratio_error)
        if np.any(finite):
            finite_maximum = max(
                finite_maximum,
                float(np.max(ratio[finite] + ratio_error[finite])),
            )
        offset = -0.045 if label == "Raw MC" else 0.045
        ratio_axis.errorbar(
            centers + offset,
            ratio,
            yerr=ratio_error,
            fmt=marker,
            color=color,
            markersize=4.3,
            linewidth=1.0,
            label=label,
        )
    ratio_axis.axhline(1.0, color="black", linewidth=1.0)
    ratio_axis.set_ylabel("Pred./data")
    ratio_axis.set_ylim(0.0, max(1.25, min(5.8, 1.08 * finite_maximum)))
    ratio_axis.set_xlabel(xlabel)
    ratio_axis.set_xticks(centers)
    ratio_axis.set_xticklabels(labels, rotation=35, ha="right")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{suffix}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    merged = load_json(args.merged)
    results = load_json(args.results)
    setup_style()
    records = {
        scheme: comparison_record(merged, results, scheme)
        for scheme in results["data_validation"]
    }
    plot_comparison(
        records["highdm_vr_njet3to4_nb1plus"],
        r"$U_{T}$ (GeV)",
        args.output_dir,
        "removal_highdm_top_enriched_data_closure",
        log_scale=True,
    )
    plot_comparison(
        records["lowdm_vr_met250to300"],
        r"$p_{T}^{miss}$ (GeV)",
        args.output_dir,
        "removal_lowdm_met_data_closure",
        log_scale=False,
    )
    args.summary_output.write_text(
        json.dumps(
            {
                "schema_version": "removal_raw_mc_comparison_v1",
                "status": "complete",
                "records": records,
                "provenance": {
                    "merged": str(args.merged.resolve()),
                    "results": str(args.results.resolve()),
                    "nominal_inputs_modified": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "plots": 2,
                "summary": str(args.summary_output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
