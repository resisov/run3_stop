#!/usr/bin/env python3
"""Make report plots that directly compare raw MC and two TF estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


COLORS = {
    "raw": "#2f6fae",
    "combined": "#e68613",
    "split": "#c43c39",
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


def apply_cms_label(axis: Any) -> None:
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


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{suffix}")
    plt.close(fig)


def step(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(len(values) + 1, dtype=float)
    stepped = np.r_[values, values[-1] if len(values) else 0.0]
    return edges, stepped


def integrated_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
) -> float:
    return float(np.sum(prediction) / np.sum(observation))


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


def plot_tf_comparison(
    record: dict[str, Any],
    xlabel: str,
    output_dir: Path,
    stem: str,
    log_scale: bool,
) -> None:
    observation = np.asarray(record["target_residual"], dtype=float)
    observation_variance = np.asarray(
        record["target_residual_variance"],
        dtype=float,
    )
    raw = np.asarray(record["top_target"], dtype=float) + np.asarray(
        record["w_target"],
        dtype=float,
    )
    raw_variance = np.asarray(
        record["top_target_variance"],
        dtype=float,
    ) + np.asarray(record["w_target_variance"], dtype=float)
    combined = np.asarray(
        record["combined_baseline"]["prediction"],
        dtype=float,
    )
    combined_variance = np.asarray(
        record["combined_baseline"]["prediction_variance"],
        dtype=float,
    )
    split = np.asarray(record["split_top_w"]["prediction"], dtype=float)
    split_variance = np.asarray(
        record["split_top_w"]["prediction_variance"],
        dtype=float,
    )
    labels = display_labels(record["labels"])
    centers = np.arange(len(observation), dtype=float) + 0.5

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
    edges, raw_step = step(raw)
    _, raw_low = step(np.clip(raw - np.sqrt(raw_variance), 0.0, None))
    _, raw_high = step(raw + np.sqrt(raw_variance))
    axis.step(
        edges,
        raw_step,
        where="post",
        color=COLORS["raw"],
        linewidth=2.3,
        label="Raw Top + W MC",
    )
    axis.fill_between(
        edges,
        raw_low,
        raw_high,
        step="post",
        color=COLORS["raw"],
        alpha=0.18,
        linewidth=0,
        label="Raw MC stat. unc.",
    )
    for values, color, linestyle, label in (
        (
            combined,
            COLORS["combined"],
            "--",
            "Combined-TF prediction",
        ),
        (
            split,
            COLORS["split"],
            "-.",
            "Top/W-split prediction",
        ),
    ):
        plot_edges, stepped = step(values)
        axis.step(
            plot_edges,
            stepped,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
        )
    axis.errorbar(
        centers,
        observation,
        yerr=np.sqrt(np.clip(observation_variance, 0.0, None)),
        fmt="o",
        color=COLORS["data"],
        markersize=4.7,
        label="Data − Other MC",
    )
    axis.set_xlim(0.0, float(len(observation)))
    axis.set_ylabel("Events")
    if log_scale:
        axis.set_yscale("log")
        positive = observation[observation > 0.0]
        axis.set_ylim(
            bottom=max(float(np.min(positive)) * 0.25, 1e-2),
            top=float(np.max(observation)) * 8.0,
        )
    else:
        linear_maximum = max(
            float(np.max(observation)),
            float(np.max(raw)),
            float(np.max(combined)),
            float(np.max(split)),
        )
        axis.set_ylim(0.0, 1.45 * linear_maximum)
    axis.legend(
        loc="upper right",
        ncol=2 if not log_scale else 1,
        frameon=True,
        framealpha=0.92,
    )
    raw_integral = integrated_ratio(raw, observation)
    combined_integral = integrated_ratio(combined, observation)
    split_integral = integrated_ratio(split, observation)
    axis.text(
        0.03,
        0.06,
        (
            "Integrated prediction / data residual\n"
            f"Raw MC = {raw_integral:.3f}\n"
            f"Combined TF = {combined_integral:.3f}\n"
            f"Top/W split = {split_integral:.3f}"
        ),
        transform=axis.transAxes,
        fontsize=13,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.0,
        },
    )
    apply_cms_label(axis)

    series = (
        (raw, raw_variance, COLORS["raw"], "o", "Raw MC"),
        (
            combined,
            combined_variance,
            COLORS["combined"],
            "s",
            "Combined TF",
        ),
        (
            split,
            split_variance,
            COLORS["split"],
            "^",
            "Top/W split",
        ),
    )
    offsets = (-0.08, 0.0, 0.08)
    for offset, (values, variance, color, marker, label) in zip(
        offsets,
        series,
    ):
        ratio, ratio_error = ratio_and_error(
            values,
            variance,
            observation,
            observation_variance,
        )
        ratio_axis.errorbar(
            centers + offset,
            ratio,
            yerr=ratio_error,
            fmt=marker,
            color=color,
            markersize=4.2,
            linewidth=1.0,
            label=label,
        )
    ratio_axis.axhline(1.0, color="black", linewidth=1.0)
    ratio_axis.set_ylabel("Pred./data")
    ratio_axis.set_ylim(0.0, 2.1 if log_scale else 1.2)
    ratio_axis.set_xlabel(xlabel)
    ratio_axis.set_xticks(centers)
    ratio_axis.set_xticklabels(labels, rotation=35, ha="right")
    save_figure(fig, output_dir, stem)


def plot_held_out_nb1(
    record: dict[str, Any],
    output_dir: Path,
) -> None:
    index = 1
    prediction = float(record["prediction"][index])
    prediction_error = float(
        np.sqrt(max(record["prediction_variance"][index], 0.0))
    )
    observation = float(record["observation"][index])
    observation_error = float(
        np.sqrt(max(record["observation_variance"][index], 0.0))
    )
    ratio = float(record["ratio"][index])
    ratio_error = float(np.sqrt(max(record["ratio_variance"][index], 0.0)))
    pull = float(record["pull"][index])

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
    axis.errorbar(
        [0.45],
        [observation],
        yerr=[observation_error],
        fmt="o",
        color=COLORS["data"],
        markersize=7,
        linewidth=1.5,
        label="Data − Other MC",
    )
    axis.errorbar(
        [0.55],
        [prediction],
        yerr=[prediction_error],
        fmt="s",
        color=COLORS["split"],
        markersize=7,
        linewidth=1.5,
        label="Top/W prediction",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, max(observation, prediction) * 1.35)
    axis.set_ylabel("Events")
    axis.legend(loc="upper right")
    deficit = 100.0 * (1.0 - ratio)
    axis.text(
        0.04,
        0.07,
        (
            r"$N_b=0$ and $N_b\geq2$ are fit inputs, not closure tests."
            "\nOnly the held-out $N_b=1$ category is shown."
            f"\nPrediction/data = {ratio:.3f} ± {ratio_error:.3f}"
            f"\nDeficit = {deficit:.1f}%; pull = {pull:.2f}"
        ),
        transform=axis.transAxes,
        fontsize=14,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 3.0,
        },
    )
    apply_cms_label(axis)

    ratio_axis.errorbar(
        [0.5],
        [ratio],
        yerr=[ratio_error],
        fmt="s",
        color=COLORS["split"],
        markersize=7,
        linewidth=1.5,
    )
    ratio_axis.axhline(1.0, color="black", linewidth=1.0)
    ratio_axis.axhspan(0.9, 1.1, color="#cfd8dc", alpha=0.28, linewidth=0)
    ratio_axis.set_ylim(0.65, 1.1)
    ratio_axis.set_ylabel("Pred./data")
    ratio_axis.set_xlabel("Held-out validation category")
    ratio_axis.set_xticks([0.5])
    ratio_axis.set_xticklabels(
        [r"$N_b=1$ (excluded from fit)"],
        fontsize=15,
    )
    save_figure(
        fig,
        output_dir,
        "tf_lowdm_nb1_control_validation",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-w", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-held-out", action="store_true")
    args = parser.parse_args()

    result = load_json(args.top_w)
    setup_style()
    validation = result["validation_regions"]
    plot_tf_comparison(
        validation["highdm_njet3to4_nb1plus"],
        r"$U_{T}$ (GeV)",
        args.output_dir,
        "tf_highdm_top_enriched_data_closure",
        log_scale=True,
    )
    plot_tf_comparison(
        validation["lowdm_met250to300"],
        r"$p_{T}^{miss}$ (GeV)",
        args.output_dir,
        "tf_lowdm_met_data_closure",
        log_scale=False,
    )
    if not args.skip_held_out:
        plot_held_out_nb1(
            result["control_validations"]["lowdm_nb_groups"],
            args.output_dir,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "plots": 2 if args.skip_held_out else 3,
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
