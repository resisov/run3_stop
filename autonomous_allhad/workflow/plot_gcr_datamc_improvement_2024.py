#!/usr/bin/env python3
"""Plot the read-only 2024 GCR Data/MC improvement study.

This script deliberately consumes only derived audit JSON files.  It never
opens or mutates the nominal histogram payload.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import chi2 as chi2_distribution


hep.style.use("CMS")

UT_EDGES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0]
)
PHOTON_PT_EDGES = np.asarray([220.0, 300.0, 400.0, 600.0, 800.0])
CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}
COLORS = {
    "data": "black",
    "nominal": "#8c8c8c",
    "candidate": "#0072B2",
    "gj": "#6A3D9A",
    "qcd_prompt": "#B15928",
    "fake": "#33A02C",
    "other": "#A6CEE3",
    "eb": "#0072B2",
    "ee": "#D55E00",
}
PRETTY_DISTRIBUTIONS = {
    "GCR/recoil": r"$U_{T}$ (coarse)",
    "GCR/ut": r"$U_{T}$",
    "GCR_Nt0/recoil": r"$U_{T}$, $N_{\mathrm{top}}=0$",
    "GCR_Nt1/recoil": r"$U_{T}$, $N_{\mathrm{top}}\geq1$",
    "GCR/bjet_pt": r"Leading b-jet $p_{\mathrm{T}}$",
    "GCR/fatjet_pt": r"Leading fatjet $p_{\mathrm{T}}$",
    "GCR/ht": r"$H_{\mathrm{T}}$",
    "GCR/jet_pt": r"Leading jet $p_{\mathrm{T}}$",
    "GCR/nb": r"$N_b$",
    "GCR/nfatjet": r"$N_{\mathrm{fj}}$",
    "GCR/njet": r"$N_j$",
    "GCR/ntop": r"$N_{\mathrm{top}}$",
    "GCR/nw": r"$N_W$",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def widths(edges: np.ndarray) -> np.ndarray:
    return edges[1:] - edges[:-1]


def step_values(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[-1]]


def finish_figure(fig: Any, axis: Any, stem: Path) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.87, right=0.96)
    hep.cms.label(ax=axis, loc=0, **CMS_LABEL)
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def exact_x_range(axis: Any, low: float, high: float) -> None:
    axis.set_xlim(low, high)
    axis.margins(x=0)


def shape_only_test(
    data: np.ndarray,
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
) -> dict[str, float | int]:
    mask = (
        np.isfinite(data)
        & np.isfinite(prediction)
        & np.isfinite(prediction_variance)
        & (data >= 0.0)
        & (prediction > 0.0)
        & (prediction_variance >= 0.0)
        & ((data > 0.0) | (prediction > 0.0))
    )
    selected_data = data[mask]
    selected_prediction = prediction[mask]
    selected_variance = prediction_variance[mask]
    scale = float(np.sum(selected_data) / np.sum(selected_prediction))
    scaled_prediction = scale * selected_prediction
    variance = selected_data + scale * scale * selected_variance
    chi2 = float(
        np.sum(
            np.divide(
                np.square(selected_data - scaled_prediction),
                variance,
                out=np.zeros_like(selected_data),
                where=variance > 0.0,
            )
        )
    )
    dof = int(np.sum(mask) - 1)
    return {
        "scale": scale,
        "chi2": chi2,
        "dof": dof,
        "pvalue": float(chi2_distribution.sf(chi2, dof)),
    }


def plot_ut(study: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = study["results"]["GCR/ut"]
    data = np.asarray(result["data"], dtype=float)
    nominal = result["predictions"]["nominal"]
    candidate = result["predictions"]["fit_prompt_pool_with_data_fake"]
    nominal_values = np.asarray(nominal["prediction"], dtype=float)
    candidate_values = np.asarray(candidate["prediction"], dtype=float)
    candidate_variance = np.asarray(candidate["prediction_variance"], dtype=float)
    components = {
        key: np.asarray(result["components"][key], dtype=float)
        for key in ("other", "fake", "gj", "qcd_nonfake")
    }
    alpha = float(candidate["alpha_from_primary_gcr_ut"])
    stack = [
        ("Other MC", components["other"], COLORS["other"]),
        ("Data-driven fake", components["fake"], COLORS["fake"]),
        (
            rf"$\gamma$+jets prompt pool $\times\,{alpha:.3f}$",
            alpha * components["gj"],
            COLORS["gj"],
        ),
        (
            rf"QCD prompt pool $\times\,{alpha:.3f}$",
            alpha * components["qcd_nonfake"],
            COLORS["qcd_prompt"],
        ),
    ]
    x = centers(UT_EDGES)
    w = widths(UT_EDGES)

    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    bottom = np.zeros_like(data)
    for label, values, color in stack:
        axis.bar(
            x,
            values,
            width=w,
            bottom=bottom,
            align="center",
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )
        bottom += values
    axis.step(
        UT_EDGES,
        step_values(nominal_values),
        where="post",
        color=COLORS["nominal"],
        linewidth=2.2,
        linestyle="--",
        label="Nominal MC",
    )
    candidate_sigma = np.sqrt(np.maximum(candidate_variance, 0.0))
    axis.fill_between(
        UT_EDGES,
        step_values(np.maximum(candidate_values - candidate_sigma, 1.0e-3)),
        step_values(candidate_values + candidate_sigma),
        step="post",
        facecolor="none",
        edgecolor="0.25",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
        label="Candidate template stat.",
    )
    axis.errorbar(
        x,
        data,
        yerr=np.sqrt(data),
        color=COLORS["data"],
        marker="o",
        linestyle="none",
        label="Data",
        zorder=10,
    )
    axis.set_yscale("log")
    axis.set_ylim(0.5, 2.8e4)
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, fontsize=8, loc="upper right")
    axis.text(
        0.025,
        0.045,
        "GCR\nPrompt pool constrained in GCR",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )

    nominal_ratio = data / nominal_values
    candidate_ratio = data / candidate_values
    nominal_err = np.sqrt(data) / nominal_values
    candidate_err = np.sqrt(data) / candidate_values
    offset = 0.035 * w
    lower.errorbar(
        x - offset,
        nominal_ratio,
        yerr=nominal_err,
        color=COLORS["nominal"],
        marker="s",
        markerfacecolor="white",
        linestyle="none",
        label="Nominal",
    )
    lower.errorbar(
        x + offset,
        candidate_ratio,
        yerr=candidate_err,
        color=COLORS["candidate"],
        marker="o",
        linestyle="none",
        label="Constrained R&D",
    )
    rel_stat = np.divide(
        candidate_sigma,
        candidate_values,
        out=np.zeros_like(candidate_sigma),
        where=candidate_values > 0,
    )
    lower.fill_between(
        UT_EDGES,
        step_values(1.0 - rel_stat),
        step_values(1.0 + rel_stat),
        step="post",
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
    )
    lower.axhline(1.0, color="black", linewidth=0.9)
    lower.set_ylim(0.55, 1.85)
    lower.set_ylabel("Data/pred.")
    lower.set_xlabel(r"$U_{T}$ (GeV)")
    lower.grid(axis="y", alpha=0.22)
    lower.legend(ncol=2, fontsize=9, loc="upper right")
    exact_x_range(lower, UT_EDGES[0], UT_EDGES[-1])
    finish_figure(fig, axis, output_dir / "gcr-ut-nominal-vs-prompt-constraint")

    return {
        "alpha": alpha,
        "data": float(np.sum(data)),
        "nominal": float(np.sum(nominal_values)),
        "candidate": float(np.sum(candidate_values)),
        "nominal_data_mc": float(np.sum(data) / np.sum(nominal_values)),
        "candidate_data_mc": float(np.sum(data) / np.sum(candidate_values)),
        "nominal_log_ratio_rms": float(
            nominal["metrics"]["log_ratio_rms"]
        ),
        "candidate_log_ratio_rms": float(
            candidate["metrics"]["log_ratio_rms"]
        ),
        "nominal_deviance": float(nominal["metrics"]["poisson_deviance"]),
        "candidate_deviance": float(
            candidate["metrics"]["poisson_deviance"]
        ),
    }


def plot_ut_fake_only(
    study: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    result = study["results"]["GCR/ut"]
    data = np.asarray(result["data"], dtype=float)
    nominal = result["predictions"]["nominal"]
    fake_only = result["predictions"]["replace_truth_fake_only"]
    nominal_values = np.asarray(nominal["prediction"], dtype=float)
    nominal_variance = np.asarray(
        nominal["prediction_variance"], dtype=float
    )
    fake_only_values = np.asarray(fake_only["prediction"], dtype=float)
    fake_only_variance = np.asarray(
        fake_only["prediction_variance"], dtype=float
    )
    components = {
        key: np.asarray(result["components"][key], dtype=float)
        for key in ("other", "fake", "gj", "qcd_nonfake")
    }
    stack = [
        ("Other MC", components["other"], COLORS["other"]),
        ("Data-driven fake", components["fake"], COLORS["fake"]),
        (r"$\gamma$+jets", components["gj"], COLORS["gj"]),
        (
            "QCD prompt/electron",
            components["qcd_nonfake"],
            COLORS["qcd_prompt"],
        ),
    ]
    x = centers(UT_EDGES)
    w = widths(UT_EDGES)

    nominal_shape = shape_only_test(
        data, nominal_values, nominal_variance
    )
    fake_only_shape = shape_only_test(
        data, fake_only_values, fake_only_variance
    )

    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    bottom = np.zeros_like(data)
    for label, values, color in stack:
        axis.bar(
            x,
            values,
            width=w,
            bottom=bottom,
            align="center",
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )
        bottom += values
    axis.step(
        UT_EDGES,
        step_values(nominal_values),
        where="post",
        color=COLORS["nominal"],
        linewidth=2.2,
        linestyle="--",
        label="Nominal MC",
    )
    fake_only_sigma = np.sqrt(np.maximum(fake_only_variance, 0.0))
    axis.fill_between(
        UT_EDGES,
        step_values(
            np.maximum(fake_only_values - fake_only_sigma, 1.0e-3)
        ),
        step_values(fake_only_values + fake_only_sigma),
        step="post",
        facecolor="none",
        edgecolor="0.25",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
        label="Available template stat.",
    )
    axis.errorbar(
        x,
        data,
        yerr=np.sqrt(data),
        color=COLORS["data"],
        marker="o",
        linestyle="none",
        label="Data",
        zorder=10,
    )
    axis.set_yscale("log")
    axis.set_ylim(0.5, 2.8e4)
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, fontsize=8, loc="upper right")
    axis.text(
        0.025,
        0.045,
        "GCR\nPrefit: truth-fake component replaced only",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    axis.text(
        0.975,
        0.045,
        (
            rf"Data/pred. $={fake_only['metrics']['integral_data_over_prediction']:.3f}$"
            "\n"
            rf"shape-only $p={fake_only_shape['pvalue']:.3f}$"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
    )

    nominal_ratio = data / nominal_values
    fake_only_ratio = data / fake_only_values
    nominal_err = np.sqrt(data) / nominal_values
    fake_only_err = np.sqrt(data) / fake_only_values
    offset = 0.035 * w
    lower.errorbar(
        x - offset,
        nominal_ratio,
        yerr=nominal_err,
        color=COLORS["nominal"],
        marker="s",
        markerfacecolor="white",
        linestyle="none",
        label=rf"Nominal ({nominal['metrics']['integral_data_over_prediction']:.3f})",
    )
    lower.errorbar(
        x + offset,
        fake_only_ratio,
        yerr=fake_only_err,
        color=COLORS["candidate"],
        marker="o",
        linestyle="none",
        label=rf"Fake-only ({fake_only['metrics']['integral_data_over_prediction']:.3f})",
    )
    relative_stat = np.divide(
        fake_only_sigma,
        fake_only_values,
        out=np.zeros_like(fake_only_sigma),
        where=fake_only_values > 0,
    )
    lower.fill_between(
        UT_EDGES,
        step_values(1.0 - relative_stat),
        step_values(1.0 + relative_stat),
        step="post",
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
    )
    lower.axhline(1.0, color="black", linewidth=0.9)
    lower.set_ylim(0.9, 1.75)
    lower.set_ylabel("Data/pred.")
    lower.set_xlabel(r"$U_{T}$ (GeV)")
    lower.grid(axis="y", alpha=0.22)
    lower.legend(ncol=2, fontsize=9, loc="upper right")
    exact_x_range(lower, UT_EDGES[0], UT_EDGES[-1])
    finish_figure(fig, axis, output_dir / "gcr-ut-prefit-fake-only")

    return {
        "data": float(np.sum(data)),
        "nominal": float(np.sum(nominal_values)),
        "fake_only": float(np.sum(fake_only_values)),
        "nominal_data_mc": float(np.sum(data) / np.sum(nominal_values)),
        "fake_only_data_mc": float(
            np.sum(data) / np.sum(fake_only_values)
        ),
        "nominal_shape_only": nominal_shape,
        "fake_only_shape_only": fake_only_shape,
        "binwise_nominal_data_mc": nominal_ratio.tolist(),
        "binwise_fake_only_data_mc": fake_only_ratio.tolist(),
    }


def plot_ut_alpha_stability(
    study: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    fit = study["fits"]["fit_prompt_pool_with_data_fake"]
    per_bin = np.asarray(fit["per_bin"]["alpha"], dtype=float)
    per_bin_err = np.asarray(fit["per_bin"]["data_stat_sigma"], dtype=float)
    inclusive = fit["inclusive"]
    low = fit["low_ut_bins_0_to_3"]
    high = fit["high_ut_bins_4_to_7"]
    x = centers(UT_EDGES)

    fig, axis = plt.subplots(figsize=(10, 10))
    axis.errorbar(
        x,
        per_bin,
        xerr=0.5 * widths(UT_EDGES),
        yerr=per_bin_err,
        color=COLORS["candidate"],
        marker="o",
        linestyle="none",
        capsize=2,
        label=r"Per-$U_T$ bin",
    )
    axis.fill_between(
        [UT_EDGES[0], UT_EDGES[4]],
        [
            low["alpha"] - low["data_stat_sigma"],
            low["alpha"] - low["data_stat_sigma"],
        ],
        [
            low["alpha"] + low["data_stat_sigma"],
            low["alpha"] + low["data_stat_sigma"],
        ],
        color="#56B4E9",
        alpha=0.25,
        label=rf"Low $U_T$: {low['alpha']:.3f}",
    )
    axis.fill_between(
        [UT_EDGES[4], UT_EDGES[-1]],
        [
            high["alpha"] - high["data_stat_sigma"],
            high["alpha"] - high["data_stat_sigma"],
        ],
        [
            high["alpha"] + high["data_stat_sigma"],
            high["alpha"] + high["data_stat_sigma"],
        ],
        color="#E69F00",
        alpha=0.25,
        label=rf"High $U_T$: {high['alpha']:.3f}",
    )
    axis.axhline(
        inclusive["alpha"],
        color="black",
        linewidth=1.8,
        label=rf"Inclusive: {inclusive['alpha']:.3f}",
    )
    axis.axhspan(
        inclusive["alpha"] - inclusive["data_stat_sigma"],
        inclusive["alpha"] + inclusive["data_stat_sigma"],
        color="black",
        alpha=0.08,
    )
    axis.set_ylim(0.95, 1.80)
    axis.set_xlabel(r"$U_{T}$ (GeV)")
    axis.set_ylabel(r"Prompt-pool normalization $\alpha$")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper right", fontsize=10)
    axis.text(
        0.025,
        0.045,
        "GCR\nData-driven fake fixed",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )
    exact_x_range(axis, UT_EDGES[0], UT_EDGES[-1])
    finish_figure(fig, axis, output_dir / "gcr-prompt-scale-vs-ut")

    return {
        "inclusive": inclusive,
        "low_ut": low,
        "high_ut": high,
        "relative_low_high_difference": float(
            abs(high["alpha"] - low["alpha"]) / inclusive["alpha"]
        ),
    }


def plot_photon_pt_alpha(
    strata: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    diagnostic = strata["prompt_pool_diagnostic"]
    alpha = np.asarray(diagnostic["per_stratum_alpha"], dtype=float)
    data = np.asarray(strata["data"]["sumw"], dtype=float)
    prompt_pool = np.asarray(diagnostic["prompt_pool"], dtype=float)
    alpha_err = np.divide(
        np.sqrt(data),
        prompt_pool,
        out=np.zeros_like(data),
        where=prompt_pool > 0,
    )
    x = centers(PHOTON_PT_EDGES)
    xerr = 0.5 * widths(PHOTON_PT_EDGES)
    global_alpha = float(diagnostic["global_alpha_integral"])

    fig, axis = plt.subplots(figsize=(10, 10))
    axis.errorbar(
        x,
        alpha[:4],
        xerr=xerr,
        yerr=alpha_err[:4],
        color=COLORS["eb"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        capsize=2,
        label="EB",
    )
    axis.errorbar(
        x,
        alpha[4:],
        xerr=xerr,
        yerr=alpha_err[4:],
        color=COLORS["ee"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        capsize=2,
        label="EE",
    )
    axis.axhline(
        global_alpha,
        color="black",
        linewidth=1.6,
        label=rf"Global sidecar fit: {global_alpha:.3f}",
    )
    axis.set_xlabel(r"Photon $p_{\mathrm{T}}$ (GeV)")
    axis.set_ylabel(r"Prompt-pool normalization $\alpha$")
    axis.set_ylim(0.85, 2.35)
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", fontsize=10)
    axis.text(
        0.025,
        0.045,
        "GCR target sidecars",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )
    exact_x_range(axis, PHOTON_PT_EDGES[0], PHOTON_PT_EDGES[-1])
    finish_figure(fig, axis, output_dir / "gcr-prompt-scale-vs-photon-pt")

    return {
        "global_alpha_sidecar": global_alpha,
        "eb_alpha": diagnostic["eta_integral_fits"]["EB"],
        "ee_alpha": diagnostic["eta_integral_fits"]["EE"],
        "per_stratum_alpha": alpha.tolist(),
        "per_stratum_data_stat_sigma": alpha_err.tolist(),
    }


def plot_distribution_metric(
    study: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    rows: list[tuple[str, float, float, bool]] = []
    for name, result in study["results"].items():
        nominal = float(
            result["predictions"]["nominal"]["metrics"]["log_ratio_rms"]
        )
        candidate = float(
            result["predictions"]["fit_prompt_pool_with_data_fake"]["metrics"][
                "log_ratio_rms"
            ]
        )
        all_improve = bool(
            result["comparisons_to_nominal"][
                "fit_prompt_pool_with_data_fake"
            ]["all_primary_metrics_improve"]
        )
        rows.append((name, nominal, candidate, all_improve))
    rows.sort(key=lambda row: row[1])

    fig, axis = plt.subplots(figsize=(10, 10))
    y = np.arange(len(rows), dtype=float)
    nominal_values = np.asarray([row[1] for row in rows])
    candidate_values = np.asarray([row[2] for row in rows])
    for yi, before, after, improves in zip(
        y,
        nominal_values,
        candidate_values,
        [row[3] for row in rows],
    ):
        axis.plot(
            [before, after],
            [yi, yi],
            color="0.72",
            linewidth=1.3,
            zorder=1,
        )
        if not improves:
            axis.scatter(
                [after],
                [yi],
                marker="x",
                s=95,
                linewidths=2.0,
                color="#CC3311",
                zorder=5,
            )
    axis.scatter(
        nominal_values,
        y,
        marker="s",
        facecolors="white",
        edgecolors=COLORS["nominal"],
        s=55,
        linewidths=1.6,
        label="Nominal",
        zorder=3,
    )
    axis.scatter(
        candidate_values,
        y,
        marker="o",
        color=COLORS["candidate"],
        s=55,
        label="Constrained R&D",
        zorder=4,
    )
    axis.set_yticks(y)
    axis.set_yticklabels(
        [PRETTY_DISTRIBUTIONS[row[0]] for row in rows],
        fontsize=15,
    )
    axis.set_ylim(-0.5, len(rows) - 0.5)
    axis.set_xlim(0.0, max(1.2, 1.08 * float(np.max(nominal_values))))
    axis.margins(x=0)
    axis.set_xlabel(
        r"RMS of $\log(\mathrm{Data}/\mathrm{prediction})$",
        fontsize=20,
    )
    axis.tick_params(axis="x", labelsize=15)
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(loc="lower right", fontsize=10)
    axis.text(
        0.975,
        0.955,
        r"$\times$: at least one primary metric worsens",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    finish_figure(fig, axis, output_dir / "gcr-all-distributions-metric")

    return {
        name: {
            "nominal_log_ratio_rms": before,
            "candidate_log_ratio_rms": after,
            "all_primary_metrics_improve": improves,
        }
        for name, before, after, improves in rows
    }


def plot_overlap_radius(
    overlap: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    radii = np.asarray(overlap["definitions"]["radii"], dtype=float)
    process_scan = overlap["scan"]["process"]

    def survival(process: str, field: str) -> np.ndarray:
        return np.asarray(
            [
                process_scan[process][f"{radius:.2f}"]["inclusive"][field]
                for radius in radii
            ],
            dtype=float,
        )

    gj_abs = survival("GJ", "abs_weighted_survival")
    qcd_abs = survival("QCD", "abs_weighted_survival")
    qcd_unweighted = survival("QCD", "unweighted_survival")

    fig, axis = plt.subplots(figsize=(10, 10))
    axis.plot(
        radii,
        gj_abs,
        color=COLORS["gj"],
        marker="s",
        linewidth=1.8,
        label=r"$\gamma$+jets direct keep, abs.-weighted",
    )
    axis.plot(
        radii,
        qcd_abs,
        color=COLORS["qcd_prompt"],
        marker="o",
        linewidth=1.8,
        label="QCD fragmentation keep, abs.-weighted",
    )
    axis.plot(
        radii,
        qcd_unweighted,
        color=COLORS["qcd_prompt"],
        marker="^",
        markerfacecolor="white",
        linestyle="--",
        linewidth=1.5,
        label="QCD fragmentation keep, unweighted",
    )
    axis.set_xlabel(r"Hard-parton matching radius $R$")
    axis.set_ylabel("Representative survival fraction")
    axis.set_ylim(0.0, 1.15)
    axis.grid(alpha=0.22)
    axis.legend(loc="center left", fontsize=10)
    exact_x_range(axis, float(radii[0]), float(radii[-1]))
    finish_figure(fig, axis, output_dir / "gcr-hardparton-dr-radius-scan")

    return {
        "status": overlap["status"],
        "artifact_status": overlap.get("artifact_status"),
        "sampled_exact_gcr": overlap["event_counts"]["sampled_exact_gcr"],
        "eligible_primary_dr": overlap["event_counts"][
            "eligible_primary_dr"
        ],
        "radii": radii.tolist(),
        "gj_direct_abs_weighted_survival": gj_abs.tolist(),
        "qcd_fragmentation_abs_weighted_survival": qcd_abs.tolist(),
        "qcd_fragmentation_unweighted_survival": qcd_unweighted.tolist(),
        "interpretation": (
            "The QCD weighted survival is dominated by rare large weights and "
            "is too radius-sensitive for an adoption decision."
        ),
    }


def build_index(
    output_dir: Path,
    summary: dict[str, Any],
    study: dict[str, Any],
) -> None:
    primary = summary["gcr_ut"]
    candidate_summary = study["candidate_summary"]
    rows = []
    for name, values in candidate_summary.items():
        alpha = values["alpha_from_primary_gcr_ut"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{'—' if alpha is None else f'{alpha:.4f}'}</td>"
            f"<td>{values['all_metrics_improved_distributions']}"
            f"/{values['valid_distributions']}</td>"
            "</tr>"
        )
    page = f"""
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2024 GCR Data/MC improvement study</title>
<style>
:root {{ color-scheme: light; --ink:#18212b; --muted:#5c6978;
  --line:#dce3e9; --blue:#0072b2; --warn:#a35c00; --bg:#f5f7f9; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:28px; }}
h1 {{ margin:0 0 8px; font-size:2rem; }}
h2 {{ margin-top:34px; }}
p,li {{ line-height:1.55; }}
.lede {{ color:var(--muted); max-width:900px; }}
.status {{ border-left:5px solid var(--warn); background:#fff8ec;
  padding:14px 18px; margin:24px 0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:14px; margin:22px 0; }}
.card,.plot,table {{ background:white; border:1px solid var(--line);
  border-radius:10px; }}
.card {{ padding:16px; }}
.value {{ font-size:1.75rem; font-weight:700; color:var(--blue); }}
.label {{ color:var(--muted); font-size:.88rem; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(450px,1fr));
  gap:18px; }}
.plot {{ padding:12px; }}
.plot img {{ width:100%; height:auto; display:block; }}
.plot p {{ margin:8px 8px 2px; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line);
  text-align:left; }}
th {{ background:#eef3f6; }}
code {{ overflow-wrap:anywhere; }}
@media(max-width:600px) {{ main {{ padding:16px; }} .plots {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>2024 high-dM GCR Data/MC improvement study</h1>
<p class="lede">Selection source: <code>real_subset_worker.py</code>.
Nominal histograms are read-only. Legacy processors and IDs are not used.</p>
<div class="status"><strong>Conditional R&amp;D result — not adopted.</strong>
The prompt-pool constraint improves the primary GCR U<sub>T</sub> agreement,
but generator-level γ+jets/QCD overlap and the residual H<sub>T</sub> shape must be
resolved before this becomes a physics correction.</div>
<div class="cards">
 <div class="card"><div class="value">{primary['nominal_data_mc']:.3f}</div>
  <div class="label">Nominal GCR Data/MC</div></div>
 <div class="card"><div class="value">{summary['gcr_ut_fake_only']['fake_only_data_mc']:.3f}</div>
  <div class="label">Prefit fake-only Data/prediction</div></div>
 <div class="card"><div class="value">{primary['candidate_data_mc']:.3f}</div>
  <div class="label">Conditional candidate Data/prediction</div></div>
 <div class="card"><div class="value">{primary['alpha']:.3f}</div>
  <div class="label">Prompt-pool GCR constraint</div></div>
 <div class="card"><div class="value">{primary['candidate_log_ratio_rms']:.3f}</div>
  <div class="label">Candidate U<sub>T</sub> log-ratio RMS</div></div>
</div>
<h2>Primary evidence</h2>
<div class="plots">
 <div class="plot"><a href="gcr-ut-prefit-fake-only.pdf">
  <img src="gcr-ut-prefit-fake-only.png"></a>
  <p>Pure prefit comparison: only the truth-fake photon component is replaced
  by the data-driven estimate. No prompt-photon rate constraint is applied.</p></div>
 <div class="plot"><a href="gcr-ut-nominal-vs-prompt-constraint.pdf">
  <img src="gcr-ut-nominal-vs-prompt-constraint.png"></a>
  <p>The conditional model retains other MC, inserts only the data-driven
  fake component, and constrains the γ+jets + QCD-prompt pool together.</p></div>
 <div class="plot"><a href="gcr-prompt-scale-vs-ut.pdf">
  <img src="gcr-prompt-scale-vs-ut.png"></a>
  <p>Low-U<sub>T</sub> and high-U<sub>T</sub> fits are compatible at the few-percent level;
  no per-bin reweighting is used.</p></div>
 <div class="plot"><a href="gcr-prompt-scale-vs-photon-pt.pdf">
  <img src="gcr-prompt-scale-vs-photon-pt.png"></a>
  <p>EB/EE and photon-p<sub>T</sub> sidecar diagnostic. The final open bin is shown
  with a finite display endpoint only; it is not used to define a correction.</p></div>
 <div class="plot"><a href="gcr-all-distributions-metric.pdf">
  <img src="gcr-all-distributions-metric.png"></a>
  <p>A single prompt normalization improves all four primary metrics in 12 of
  13 valid distributions. H<sub>T</sub> remains a failed validation gate.</p></div>
 <div class="plot"><a href="gcr-hardparton-dr-radius-scan.pdf">
  <img src="gcr-hardparton-dr-radius-scan.png"></a>
  <p>Representative exact-GCR hard-parton scan. The QCD weighted result jumps
  with radius because a few large-weight events dominate; no radius is adopted
  from this diagnostic.</p></div>
</div>
<h2>Candidate comparison</h2>
<table><thead><tr><th>Policy</th><th>\u03b1 from GCR U<sub>T</sub></th>
<th>Distributions improving all metrics</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Decision</h2>
<ul>
 <li>Reject replacing all QCD with the fake estimate: it worsens GCR
 Data/MC from 1.416 to 2.137.</li>
 <li>Retain the data-driven estimate only for the truth-fake component.</li>
 <li>Treat \u03b1≈1.39 as a GCR-constrained nuisance/transfer input, not as
 a universal prefit rescaling.</li>
 <li>Do not adopt until γ+jets/QCD direct-vs-fragmentation overlap is made
 explicit and H<sub>T</sub>, N<sub>b</sub>, and jet-related closure gates are passed.</li>
</ul>
<p class="lede">Machine-readable provenance:
<a href="summary.json"><code>summary.json</code></a>,
<a href="audits/gcr-nominal-data-mc-audit.json">nominal shape/data audit</a>,
<a href="audits/gcr-hardparton-dr-representative.json">representative overlap audit</a>,
and <a href="../../validation/gcr_normalization_corrections_audit_20260726.json">
normalization/correction audit</a>.</p>
</main></body></html>
"""
    (output_dir / "index.html").write_text(page)


def write_artifact_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact-manifest.json"
    records = []
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or path.name.startswith(".")
        ):
            continue
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "gcr_datamc_report_artifacts_v1",
                "status": "complete",
                "files": records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--strata-audit", type=Path, required=True)
    parser.add_argument("--normalization-audit", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    study = read_json(args.study)
    strata = read_json(args.strata_audit)
    normalization = read_json(args.normalization_audit)
    overlap = read_json(args.overlap_audit)
    if study.get("status") != "complete":
        raise RuntimeError("GCR improvement study is not complete")
    if strata.get("status") != "complete":
        raise RuntimeError("photon-strata audit is not complete")
    if overlap.get("status") != "complete":
        raise RuntimeError("representative overlap audit is not complete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "gcr_datamc_improvement_report_v1",
        "status": "conditional_r_and_d_not_adopted",
        "selection_source": study["selection_source"],
        "nominal_mutated": False,
        "inputs": {
            "study": {
                "path": str(args.study),
                "sha256": sha256(args.study),
            },
            "strata_audit": {
                "path": str(args.strata_audit),
                "sha256": sha256(args.strata_audit),
            },
            "normalization_audit": {
                "path": str(args.normalization_audit),
                "sha256": sha256(args.normalization_audit),
            },
            "overlap_audit": {
                "path": str(args.overlap_audit),
                "sha256": sha256(args.overlap_audit),
            },
        },
        "normalization_audit_status": normalization.get("status"),
        "gcr_ut": plot_ut(study, args.output_dir),
        "gcr_ut_fake_only": plot_ut_fake_only(study, args.output_dir),
        "ut_alpha_stability": plot_ut_alpha_stability(
            study, args.output_dir
        ),
        "photon_pt_alpha": plot_photon_pt_alpha(strata, args.output_dir),
        "distribution_metrics": plot_distribution_metric(
            study, args.output_dir
        ),
        "representative_overlap_scan": plot_overlap_radius(
            overlap, args.output_dir
        ),
        "guardrails": [
            "Do not replace all QCD with the data-driven fake estimate.",
            "Do not apply the fitted factor globally outside the GCR model.",
            "Resolve generator-level GJ/QCD overlap before adoption.",
            "Require HT and jet-related closure before adoption.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    build_index(args.output_dir, summary, study)
    write_artifact_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "gcr_ut": summary["gcr_ut"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
