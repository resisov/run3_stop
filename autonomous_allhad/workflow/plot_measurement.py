#!/usr/bin/env python3
"""Draw all trigger and low-pT scale-factor measurement plots with one CMS style."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


CMS_LABEL_FONT_SIZE = 17
FIGURE_SIZE = (8.0, 8.0)
COLORBAR_FIGURE_SIZE = (12.0, 10.0)
PNG_DPI = 180
DATA_STYLE = {
    "fmt": "o",
    "color": "black",
    "markersize": 6,
    "capsize": 2,
    "linewidth": 1.2,
}
SIMULATION_STYLE = {
    "fmt": "s",
    "color": "#e31a1c",
    "markerfacecolor": "white",
    "markersize": 6,
    "capsize": 2,
    "linewidth": 1.2,
}


def _apply_style() -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "axes.linewidth": 1.5,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def _cms_rlabel(year: str) -> str:
    return f"{year} (13.6 TeV)"


def _measurement_year(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        if payload.get("year"):
            return str(payload["year"])
    for payload in payloads:
        measurement = str(payload.get("measurement", ""))
        for year in range(2022, 2030):
            if str(year) in measurement:
                return str(year)
    return "Run-3"


def _cms_label(ax: plt.Axes, year: str) -> None:
    hep.cms.label(
        llabel="Work in progress",
        rlabel=_cms_rlabel(year),
        loc=0,
        ax=ax,
        fontsize=CMS_LABEL_FONT_SIZE,
    )


def _figure_header(
    fig: plt.Figure,
    year: str,
    *,
    y: float = 0.90,
    left: float = 0.08,
    right: float = 0.92,
) -> None:
    header_ax = fig.add_axes((left, y, right - left, 0.001), frameon=False)
    header_ax.set_axis_off()
    _cms_label(header_ax, year)


def _save(fig: plt.Figure, output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=PNG_DPI if suffix == ".png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_interval_errors(bins: list[dict[str, Any]], prefix: str) -> np.ndarray:
    values = np.asarray([item[f"{prefix}_efficiency"] for item in bins], dtype=float)
    intervals = np.asarray([item[f"{prefix}_interval"] for item in bins], dtype=float)
    # Closed intervals at exactly zero or one can differ from the stored
    # efficiency by a few ulps after JSON round-tripping.  Matplotlib rejects
    # even those numerically negative error lengths.
    return np.maximum(
        np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
        0.0,
    )


def _scale_factor_ylim(values: np.ndarray, uncertainties: np.ndarray) -> tuple[float, float]:
    """Include every SF error bar while retaining unity as the visual reference."""
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    lower = min(float(np.nanmin(values - uncertainties)), 1.0)
    upper = max(float(np.nanmax(values + uncertainties)), 1.0)
    span = max(upper - lower, 0.02)
    padding = 0.08 * span
    return lower - padding, upper + padding


def _signal_bin_average(
    masses: np.ndarray,
    *,
    bin_width: float,
    mean: float,
    sigma: float,
    model: str,
    parameters: dict[str, float],
) -> np.ndarray:
    """Evaluate the fitted signal PDF integrated over a sliding mass bin."""
    masses = np.asarray(masses, dtype=float)
    offsets = ((np.arange(24, dtype=float) + 0.5) / 24.0 - 0.5) * bin_width
    points = masses[:, None] + offsets[None, :]
    t = (points - mean) / sigma
    values = np.exp(-0.5 * t**2)
    if model == "double_crystal_ball":
        alpha_left = max(float(parameters["alpha_left"]), 1.0e-3)
        n_left = max(float(parameters["n_left"]), 1.001)
        alpha_right = max(float(parameters["alpha_right"]), 1.0e-3)
        n_right = max(float(parameters["n_right"]), 1.001)
        left = t < -alpha_left
        right = t > alpha_right
        b_left = n_left / alpha_left - alpha_left
        b_right = n_right / alpha_right - alpha_right
        log_a_left = n_left * np.log(n_left / alpha_left) - 0.5 * alpha_left**2
        log_a_right = n_right * np.log(n_right / alpha_right) - 0.5 * alpha_right**2
        values[left] = np.exp(
            np.clip(log_a_left - n_left * np.log(np.maximum(b_left - t[left], 1.0e-12)), -700.0, 700.0)
        )
        values[right] = np.exp(
            np.clip(log_a_right - n_right * np.log(np.maximum(b_right + t[right], 1.0e-12)), -700.0, 700.0)
        )
    elif model == "gaussian_exponential":
        transition_left = max(float(parameters["transition_left"]), 1.0e-3)
        transition_right = max(float(parameters["transition_right"]), 1.0e-3)
        left = t < -transition_left
        right = t > transition_right
        values[left] = np.exp(0.5 * transition_left**2 + transition_left * t[left])
        values[right] = np.exp(0.5 * transition_right**2 - transition_right * t[right])
    elif model == "crystal_ball":
        alpha = max(float(parameters["alpha"]), 1.0e-3)
        n_value = max(float(parameters["n"]), 1.001)
        tail = t <= -alpha
        b_value = n_value / alpha - alpha
        log_a = n_value * np.log(n_value / alpha) - 0.5 * alpha**2
        values[tail] = np.exp(
            np.clip(log_a - n_value * np.log(np.maximum(b_value - t[tail], 1.0e-12)), -700.0, 700.0)
        )
    elif model == "double_gaussian":
        values = 0.8 * values + 0.2 * np.exp(-0.5 * ((points - mean) / (2.0 * sigma)) ** 2)
    elif model != "gaussian":
        raise ValueError(f"unsupported plotted signal model: {model}")
    return np.clip(np.mean(values, axis=1), 1.0e-12, None)


def _background_values(
    masses: np.ndarray,
    *,
    reference_centres: np.ndarray,
    model: str,
    first_order: float,
    second_order: float = 0.0,
) -> np.ndarray:
    """Evaluate a background shape using the normalization convention of the fit."""
    shifted = np.asarray(masses, dtype=float) - float(np.mean(reference_centres))
    if model == "exponential":
        values = np.exp(np.clip(first_order * shifted, -30.0, 30.0))
    elif model == "chebyshev":
        reference_shifted = reference_centres - float(np.mean(reference_centres))
        scale = max(float(np.max(np.abs(reference_shifted))), 1.0e-12)
        scaled = shifted / scale
        values = (
            1.0
            + np.clip(first_order, -0.95, 0.95) * scaled
            + np.clip(second_order, -0.95, 0.95) * (2.0 * scaled**2 - 1.0)
        )
    elif model == "linear":
        span = max(float(np.ptp(reference_centres)), 1.0e-6)
        values = 1.0 + np.clip(first_order, -0.95, 0.95) * shifted / (span / 2.0)
    else:
        raise ValueError(f"unsupported plotted background model: {model}")
    return np.clip(values, 1.0e-12, None)


def _continuous_fit_curves(
    nominal: dict[str, Any],
    category: str,
    mass_centres: np.ndarray,
    dense_mass: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-evaluate the analytic likelihood model instead of joining bin centres."""
    signal_parameters = nominal.get("signal_shape_parameters", {})
    mean_key = "mean" if category == "pass" else "fail_mean"
    sigma_key = "sigma" if category == "pass" else "fail_sigma"
    signal_at_centres = _signal_bin_average(
        mass_centres,
        bin_width=bin_width,
        mean=float(nominal[mean_key]),
        sigma=float(nominal[sigma_key]),
        model=str(nominal["signal_model"]),
        parameters=signal_parameters,
    )
    signal_dense = _signal_bin_average(
        dense_mass,
        bin_width=bin_width,
        mean=float(nominal[mean_key]),
        sigma=float(nominal[sigma_key]),
        model=str(nominal["signal_model"]),
        parameters=signal_parameters,
    )
    signal_dense /= float(np.sum(signal_at_centres))
    signal_yield = float(np.sum(np.asarray(nominal[f"{category}_signal_model"], dtype=float)))

    background_parameters = nominal.get("background_shape_parameters") or {}
    prefix = "pass" if category == "pass" else "fail"
    stored_background = np.asarray(nominal[f"{category}_background_model"], dtype=float)
    first_order_key = f"{prefix}_first_order"
    if first_order_key not in background_parameters:
        # Older fit results stored the evaluated background component but not
        # its coefficients.  Recover the same analytic family without
        # changing or refitting those published measurement numbers.
        if str(nominal["background_model"]) == "exponential":
            coefficients = np.polyfit(mass_centres, np.log(np.clip(stored_background, 1.0e-12, None)), 1)
            background_curve = np.exp(np.polyval(coefficients, dense_mass))
        else:
            degree = 2 if str(nominal["background_model"]) == "chebyshev" else 1
            coefficients = np.polyfit(mass_centres, stored_background, degree)
            background_curve = np.clip(np.polyval(coefficients, dense_mass), 1.0e-12, None)
        return signal_yield * signal_dense + background_curve, background_curve
    background_at_centres = _background_values(
        mass_centres,
        reference_centres=mass_centres,
        model=str(nominal["background_model"]),
        first_order=float(background_parameters[first_order_key]),
        second_order=float(background_parameters.get(f"{prefix}_second_order", 0.0)),
    )
    background_dense = _background_values(
        dense_mass,
        reference_centres=mass_centres,
        model=str(nominal["background_model"]),
        first_order=float(background_parameters[first_order_key]),
        second_order=float(background_parameters.get(f"{prefix}_second_order", 0.0)),
    )
    background_dense /= float(np.sum(background_at_centres))
    background_yield = float(
        np.sum(stored_background)
    )
    background_curve = background_yield * background_dense
    return signal_yield * signal_dense + background_curve, background_curve


def plot_trigger_result(result_path: Path, output_dir: Path, measurement: str) -> dict[str, Any]:
    """Draw MET or photon trigger efficiency and scale-factor plots."""
    _apply_style()
    result = json.loads(result_path.read_text())
    if result.get("measurement_type") != measurement:
        raise ValueError(f"measurement mismatch in {result_path}")
    bins = result.get("bins") or []
    if not bins:
        raise ValueError(f"no bins in {result_path}")
    year = _measurement_year(result)

    outputs: list[str] = []
    captions: dict[str, str] = {}
    if measurement == "met_genuine":
        edges = np.asarray(result["bin_edges_gev"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * (edges[1:] - edges[:-1])
        data_eff = np.asarray([item["data_efficiency"] for item in bins], dtype=float)
        mc_eff = np.asarray([item["mc_efficiency"] for item in bins], dtype=float)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        ax.errorbar(
            centers,
            data_eff,
            xerr=xerr,
            yerr=_plot_interval_errors(bins, "data"),
            label="Data (EGamma)",
            zorder=4,
            **DATA_STYLE,
        )
        ax.errorbar(
            centers,
            mc_eff,
            xerr=xerr,
            yerr=_plot_interval_errors(bins, "mc"),
            label=r"Simulation ($t\bar{t}$ semileptonic)",
            zorder=3,
            **SIMULATION_STYLE,
        )
        ax.set(
            xlim=(edges[0], edges[-1]),
            ylim=(0.0, 1.08),
            xlabel=r"$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)",
            ylabel="Trigger efficiency",
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        ax.legend(loc="lower right")
        _figure_header(fig, year, y=0.88, left=0.16, right=0.96)
        outputs += _save(fig, output_dir / f"met_trigger_efficiency_{year}")
        captions[f"met_trigger_efficiency_{year}"] = (
            "MET-trigger efficiencies in EGamma data and semileptonic ttbar simulation."
        )

        sf = np.asarray([item["scale_factor"] for item in bins], dtype=float)
        sf_unc = np.asarray([item["scale_factor_uncertainty"] for item in bins], dtype=float)
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        ax.axhline(1.0, color="#e31a1c", linewidth=1.5, linestyle="--", zorder=1)
        ax.errorbar(
            centers,
            sf,
            xerr=xerr,
            yerr=sf_unc,
            zorder=3,
            **DATA_STYLE,
        )
        ax.set(
            xlim=(edges[0], edges[-1]),
            ylim=_scale_factor_ylim(sf, sf_unc),
            xlabel=r"$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)",
            ylabel="Trigger scale factor",
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        _figure_header(fig, year, y=0.88, left=0.16, right=0.96)
        outputs += _save(fig, output_dir / f"met_trigger_scale_factor_{year}")
        captions[f"met_trigger_scale_factor_{year}"] = (
            "Data-to-simulation MET-trigger scale factors with total uncertainties."
        )
    elif measurement == "photon":
        pt_edges = np.asarray(result["pt_edges_gev"], dtype=float)
        eta_edges = np.asarray(result["abseta_edges"], dtype=float)
        centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
        xerr = 0.5 * (pt_edges[1:] - pt_edges[:-1])
        colors = ("black", "#1f78b4", "#e31a1c", "#33a02c")

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        eta_handles: list[Any] = []
        for eta_index, color in enumerate(colors):
            group = bins[eta_index * len(centers) : (eta_index + 1) * len(centers)]
            label = rf"${eta_edges[eta_index]:g} \leq |\eta^\gamma| < {eta_edges[eta_index + 1]:g}$"
            eta_handles.append(
                Line2D([0], [0], color=color, marker="o", linewidth=1.4, markersize=6, label=label)
            )
            offset = (eta_index - 1.5) * 3.0
            ax.errorbar(
                centers + offset,
                np.asarray([item["data_efficiency"] for item in group], dtype=float),
                xerr=xerr,
                yerr=_plot_interval_errors(group, "data"),
                fmt="o",
                color=color,
                markersize=5.5,
                capsize=2,
                linewidth=1.0,
                zorder=4,
            )
            ax.errorbar(
                centers + offset,
                np.asarray([item["mc_efficiency"] for item in group], dtype=float),
                xerr=xerr,
                yerr=_plot_interval_errors(group, "mc"),
                fmt="s",
                color=color,
                markerfacecolor="white",
                markersize=5.5,
                capsize=2,
                linewidth=1.0,
                zorder=3,
            )
        type_handles = [
            Line2D([0], [0], color="0.25", marker="o", linestyle="none", markersize=6, label="Data (JetMET)"),
            Line2D(
                [0],
                [0],
                color="0.25",
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                markersize=6,
                label=r"Simulation ($\gamma$+jets)",
            ),
        ]
        first_legend = ax.legend(handles=type_handles, loc="lower right")
        ax.add_artist(first_legend)
        ax.legend(handles=eta_handles, loc="lower left")
        ax.set(
            xlim=(pt_edges[0], pt_edges[-1]),
            ylim=(0.988, 1.0025),
            xlabel=r"Photon $p_{\mathrm{T}}$ (GeV)",
            ylabel="Trigger efficiency",
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        _figure_header(fig, year, y=0.88, left=0.16, right=0.96)
        outputs += _save(fig, output_dir / f"photon_trigger_efficiency_{year}")
        captions[f"photon_trigger_efficiency_{year}"] = (
            "Photon-trigger efficiencies in JetMET data and gamma+jets simulation."
        )

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        photon_sf = np.asarray([item["scale_factor"] for item in bins], dtype=float)
        photon_sf_unc = np.asarray(
            [item["scale_factor_uncertainty"] for item in bins],
            dtype=float,
        )
        for eta_index, color in enumerate(colors):
            group = bins[eta_index * len(centers) : (eta_index + 1) * len(centers)]
            offset = (eta_index - 1.5) * 3.0
            ax.errorbar(
                centers + offset,
                np.asarray([item["scale_factor"] for item in group], dtype=float),
                xerr=xerr,
                yerr=np.asarray([item["scale_factor_uncertainty"] for item in group], dtype=float),
                fmt="o",
                color=color,
                markersize=5.5,
                capsize=2,
                linewidth=1.1,
                label=rf"${eta_edges[eta_index]:g} \leq |\eta^\gamma| < {eta_edges[eta_index + 1]:g}$",
                zorder=3,
            )
        ax.axhline(1.0, color="#e31a1c", linewidth=1.5, linestyle="--", zorder=1)
        ax.set(
            xlim=(pt_edges[0], pt_edges[-1]),
            ylim=_scale_factor_ylim(photon_sf, photon_sf_unc),
            xlabel=r"Photon $p_{\mathrm{T}}$ (GeV)",
            ylabel="Trigger scale factor",
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.34, 0.015),
            ncol=2,
        )
        _figure_header(fig, year, y=0.88, left=0.16, right=0.96)
        outputs += _save(fig, output_dir / f"photon_trigger_scale_factor_{year}")
        captions[f"photon_trigger_scale_factor_{year}"] = (
            "Data-to-simulation photon-trigger scale factors with total uncertainties."
        )
    else:
        raise ValueError(measurement)

    manifest = {
        "schema_version": 1,
        "measurement": measurement,
        "style": {
            "cms_llabel": "Work in progress",
            "cms_rlabel": _cms_rlabel(year),
            "cms_label_fontsize": CMS_LABEL_FONT_SIZE,
            "standard_figure_inches": list(FIGURE_SIZE),
            "colorbar_figure_inches": list(COLORBAR_FIGURE_SIZE),
            "titles": False,
        },
        "files": outputs,
        "captions": captions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def plot_tnp_result(result_path: Path, histograms_path: Path, output_dir: Path) -> dict[str, Any]:
    """Draw low-pT electron or muon tag-and-probe plots."""
    _apply_style()
    result = json.loads(result_path.read_text())
    histograms = json.loads(histograms_path.read_text())
    year = _measurement_year(result, histograms)
    measurement_kind = str(histograms.get("kind") or result.get("kind") or "")
    if measurement_kind == "electron":
        mass_axis_label = r"$m_{ee}$ (GeV)"
        object_pt_axis_label = r"Electron $p_{\mathrm{T}}$ (GeV)"
        object_eta_axis_label = r"Electron $|\eta|$"
    elif measurement_kind == "muon":
        mass_axis_label = r"$m_{\mu\mu}$ (GeV)"
        object_pt_axis_label = r"Muon $p_{\mathrm{T}}$ (GeV)"
        object_eta_axis_label = r"Muon $|\eta|$"
    else:
        mass_axis_label = r"$m_{\ell\ell}$ (GeV)"
        object_pt_axis_label = r"Lepton $p_{\mathrm{T}}$ (GeV)"
        object_eta_axis_label = r"Lepton $|\eta|$"
    output_dir.mkdir(parents=True, exist_ok=True)
    eta_edges = np.asarray(result["probe_abseta_edges"], dtype=float)
    pt_edges = np.asarray(result["probe_pt_edges_gev"], dtype=float)
    n_eta = len(eta_edges) - 1
    n_pt = len(pt_edges) - 1
    bins = result["bins"]
    sf = np.asarray([item.get("scale_factor", np.nan) for item in bins], dtype=float).reshape(n_eta, n_pt)
    uncertainty = np.asarray(
        [item.get("scale_factor_uncertainty", np.nan) for item in bins], dtype=float
    ).reshape(n_eta, n_pt)
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    pt_errors = 0.5 * (pt_edges[1:] - pt_edges[:-1])
    outputs: list[str] = []
    captions: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
    if n_pt == 1:
        eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        eta_errors = 0.5 * (eta_edges[1:] - eta_edges[:-1])
        ax.errorbar(eta_centers, sf[:, 0], xerr=eta_errors, yerr=uncertainty[:, 0], zorder=3, **DATA_STYLE)
        ax.set_xlabel(object_eta_axis_label)
        ax.set_xlim(eta_edges[0], eta_edges[-1])
    else:
        colors = ("black", "#e31a1c", "#377eb8", "#4daf4a", "#984ea3")
        markers = ("o", "s", "^", "D", "v")
        for eta_index in range(n_eta):
            ax.errorbar(
                pt_centers,
                sf[eta_index],
                xerr=pt_errors,
                yerr=uncertainty[eta_index],
                fmt=markers[eta_index % len(markers)],
                color=colors[eta_index % len(colors)],
                markerfacecolor="white" if eta_index else "black",
                markersize=6,
                linewidth=1.2,
                capsize=2,
                label=rf"${eta_edges[eta_index]:g}<|\eta|<{eta_edges[eta_index + 1]:g}$",
            )
        ax.set_xlabel(object_pt_axis_label)
        ax.set_xlim(pt_edges[0], pt_edges[-1])
    ax.axhline(1.0, color="#e31a1c", linestyle="--", linewidth=1.3, zorder=1)
    ax.set_ylabel("Data/MC scale factor")
    finite = np.concatenate(
        (sf.ravel() - uncertainty.ravel(), sf.ravel() + uncertainty.ravel(), np.asarray([1.0]))
    )
    finite = finite[np.isfinite(finite)]
    if finite.size:
        padding = max(0.002, 0.15 * float(np.ptp(finite)))
        ax.set_ylim(float(np.min(finite)) - padding, float(np.max(finite)) + padding)
    if n_pt > 1:
        ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
    _cms_label(ax, year)
    outputs += _save(fig, output_dir / "scale_factor")
    captions["scale_factor"] = (
        "Data-to-simulation scale factors with total uncertainties; bins without a valid fit are omitted. "
        + (
            rf"The horizontal axis is probe |eta| for the single {pt_edges[0]:g}-{pt_edges[-1]:g} GeV pT bin."
            if n_pt == 1
            else "The marker series correspond to the probe |eta| bins listed in the legend."
        )
    )

    fig, ax = plt.subplots(figsize=COLORBAR_FIGURE_SIZE)
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.88)
    image = ax.pcolormesh(pt_edges, eta_edges, sf, shading="flat", cmap="viridis")
    for eta_index in range(n_eta):
        for pt_index in range(n_pt):
            if not np.isfinite(sf[eta_index, pt_index]):
                continue
            ax.text(
                pt_centers[pt_index],
                0.5 * (eta_edges[eta_index] + eta_edges[eta_index + 1]),
                f"{sf[eta_index, pt_index]:.3f}\n$\\pm${uncertainty[eta_index, pt_index]:.3f}",
                ha="center",
                va="center",
                color="white" if sf[eta_index, pt_index] < np.nanmedian(sf) else "black",
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, label="Data/MC scale factor")
    ax.set_xlabel(object_pt_axis_label)
    ax.set_ylabel(object_eta_axis_label)
    _cms_label(ax, year)
    outputs += _save(fig, output_dir / "scale_factor_heatmap")
    captions["scale_factor_heatmap"] = (
        "Two-dimensional scale-factor map; each cell shows the central value and total uncertainty."
    )

    data_eff_all = np.asarray(
        [item["fits"]["nominal"]["data"]["efficiency"] for item in bins], dtype=float
    )
    data_unc_all = np.asarray(
        [item["fits"]["nominal"]["data"]["efficiency_stat_uncertainty"] for item in bins],
        dtype=float,
    )
    mc_eff_all = np.asarray(
        [item["fits"]["nominal"]["mc"]["efficiency"] for item in bins], dtype=float
    )
    mc_unc_all = np.asarray(
        [item["fits"]["nominal"]["mc"]["efficiency_stat_uncertainty"] for item in bins],
        dtype=float,
    )
    efficiency_extent = np.concatenate(
        (
            data_eff_all - data_unc_all,
            data_eff_all + data_unc_all,
            mc_eff_all - mc_unc_all,
            mc_eff_all + mc_unc_all,
            np.asarray([1.0]),
        )
    )
    efficiency_extent = efficiency_extent[np.isfinite(efficiency_extent)]
    efficiency_padding = max(0.005, 0.10 * float(np.ptp(efficiency_extent)))
    efficiency_ylim = (
        max(0.0, float(np.min(efficiency_extent)) - efficiency_padding),
        min(1.08, float(np.max(efficiency_extent)) + efficiency_padding),
    )

    if n_pt == 1:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        eta_errors = 0.5 * (eta_edges[1:] - eta_edges[:-1])
        data_eff = np.asarray([bins[i]["fits"]["nominal"]["data"]["efficiency"] for i in range(n_eta)])
        data_unc = np.asarray(
            [bins[i]["fits"]["nominal"]["data"]["efficiency_stat_uncertainty"] for i in range(n_eta)]
        )
        mc_eff = np.asarray([bins[i]["fits"]["nominal"]["mc"]["efficiency"] for i in range(n_eta)])
        mc_unc = np.asarray(
            [bins[i]["fits"]["nominal"]["mc"]["efficiency_stat_uncertainty"] for i in range(n_eta)]
        )
        ax.errorbar(eta_centers, data_eff, xerr=eta_errors, yerr=data_unc, label="Data", zorder=4, **DATA_STYLE)
        ax.errorbar(
            eta_centers,
            mc_eff,
            xerr=eta_errors,
            yerr=mc_unc,
            label="Simulation",
            zorder=3,
            **SIMULATION_STYLE,
        )
        ax.set(
            xlabel=object_eta_axis_label,
            ylabel="Efficiency",
            xlim=(eta_edges[0], eta_edges[-1]),
            ylim=efficiency_ylim,
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        ax.legend(loc="best")
        _cms_label(ax, year)
        outputs += _save(fig, output_dir / "efficiency")
    else:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
        eta_colors = ("#000000", "#0000ff", "#e31a1c", "#009e73", "#984ea3")
        eta_handles: list[Any] = []
        for eta_index in range(n_eta):
            group = bins[eta_index * n_pt : (eta_index + 1) * n_pt]
            color = eta_colors[eta_index % len(eta_colors)]
            offset = (eta_index - 0.5 * (n_eta - 1)) * 0.08
            ax.errorbar(
                pt_centers + offset - 0.018,
                np.asarray([item["fits"]["nominal"]["data"]["efficiency"] for item in group]),
                xerr=pt_errors,
                yerr=np.asarray(
                    [item["fits"]["nominal"]["data"]["efficiency_stat_uncertainty"] for item in group]
                ),
                fmt="o",
                color=color,
                markerfacecolor=color,
                markersize=6,
                capsize=2,
                linewidth=1.2,
                zorder=4,
            )
            ax.errorbar(
                pt_centers + offset + 0.018,
                np.asarray([item["fits"]["nominal"]["mc"]["efficiency"] for item in group]),
                xerr=pt_errors,
                yerr=np.asarray(
                    [item["fits"]["nominal"]["mc"]["efficiency_stat_uncertainty"] for item in group]
                ),
                fmt="s",
                color=color,
                markerfacecolor="white",
                markersize=6,
                capsize=2,
                linewidth=1.2,
                zorder=3,
            )
            eta_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.0,
                    label=rf"${eta_edges[eta_index]:g}<|\eta|<{eta_edges[eta_index + 1]:g}$",
                )
            )
        type_handles = [
            Line2D([0], [0], color="0.2", marker="o", linestyle="none", markersize=6, label="Data"),
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                markersize=6,
                label="Simulation",
            ),
        ]
        type_legend = ax.legend(handles=type_handles, loc="upper left", fontsize=12)
        ax.add_artist(type_legend)
        ax.legend(handles=eta_handles, loc="upper right", fontsize=11)
        ax.set(
            xlabel=object_pt_axis_label,
            ylabel="Efficiency",
            xlim=(pt_edges[0], pt_edges[-1]),
            ylim=efficiency_ylim,
        )
        ax.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
        _cms_label(ax, year)
        outputs += _save(fig, output_dir / "efficiency")
    captions["efficiency"] = (
        "Tag-and-probe efficiencies in data and simulation; error bars are statistical. "
        + (
            rf"The horizontal axis is probe |eta| for {pt_edges[0]:g} < pT < {pt_edges[-1]:g} GeV."
            if n_pt == 1
            else "The colored series correspond to the probe |eta| bins: "
            + ", ".join(
                rf"{eta_edges[index]:g} < |eta| < {eta_edges[index + 1]:g}"
                for index in range(n_eta)
            )
            + "."
        )
    )

    mass_edges = np.asarray(histograms["mass_edges_gev"], dtype=float)
    for flat_index, item in enumerate(bins):
        if not item.get("fits", {}).get("nominal"):
            continue
        nominal_rebin_factor = int(item["fits"]["nominal"]["data"].get("rebin_factor", 1))
        rebin_starts = np.arange(0, len(mass_edges) - 1, nominal_rebin_factor)
        rebinned_mass_edges = np.concatenate([mass_edges[rebin_starts], mass_edges[-1:]])
        mass_centers_all = 0.5 * (rebinned_mass_edges[:-1] + rebinned_mass_edges[1:])
        fit_window = item["fits"]["nominal"]["data"]["fit_window_gev"]
        selected_mass = (mass_centers_all >= fit_window[0]) & (mass_centers_all <= fit_window[1])
        mass_centers = mass_centers_all[selected_mass]
        dense_mass = np.linspace(float(fit_window[0]), float(fit_window[1]), 1200)
        bin_width_gev = float(np.median(np.diff(rebinned_mass_edges)))
        bin_width_mev = 1000.0 * float(np.median(np.diff(rebinned_mass_edges)))
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True)
        for row, sample in enumerate(("data", "mc")):
            nominal = item["fits"]["nominal"][sample]
            display_sample = "Simulation" if sample == "mc" else "Data"
            for column, category in enumerate(("pass", "fail")):
                ax = axes[row, column]
                observed = np.add.reduceat(
                    np.asarray(histograms["samples"][sample][f"{category}_sumw"][flat_index], dtype=float),
                    rebin_starts,
                )[selected_mass]
                variance = np.add.reduceat(
                    np.asarray(histograms["samples"][sample][f"{category}_sumw2"][flat_index], dtype=float),
                    rebin_starts,
                )[selected_mass]
                model, background_model = _continuous_fit_curves(
                    nominal,
                    category,
                    mass_centers,
                    dense_mass,
                    bin_width_gev,
                )
                ax.errorbar(
                    mass_centers,
                    observed,
                    yerr=np.sqrt(np.maximum(variance, 0.0)),
                    fmt="o",
                    color="black",
                    markersize=3,
                    linewidth=1.0,
                    label=display_sample,
                )
                ax.plot(
                    dense_mass,
                    model,
                    color="#e31a1c",
                    linewidth=1.5,
                    label="Sig.+bkg. fit",
                )
                ax.plot(
                    dense_mass,
                    background_model,
                    color="#0000ff",
                    linewidth=1.2,
                    label="Background",
                )
                probe_category = "Passing probes" if category == "pass" else "Failing probes"
                ax.text(
                    0.035,
                    0.955,
                    f"{probe_category}\n{display_sample}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    linespacing=1.05,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
                    zorder=8,
                )
                if max(float(np.nanmax(observed)), float(np.nanmax(model))) >= 1.0e4:
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((0, 0))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.yaxis.get_offset_text().set_fontsize(12)
                ax.tick_params(axis="both", labelsize=13)
                ax.set_box_aspect(1)
                ax.grid(axis="y", linestyle=":", color="0.82", linewidth=0.8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        order = [labels.index(label) for label in ("Data", "Sig.+bkg. fit", "Background")]
        fig.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            loc="center left",
            bbox_to_anchor=(0.84, 0.52),
            fontsize=10,
        )
        # Place the four square panels explicitly.  Automatic tight/constrained
        # layout leaves excessive white space once a shared external legend is
        # present, especially for a square 8x8 canvas.
        panel_positions = (
            (0.12, 0.49, 0.31, 0.31),
            (0.51, 0.49, 0.31, 0.31),
            (0.12, 0.15, 0.31, 0.31),
            (0.51, 0.15, 0.31, 0.31),
        )
        for ax, position in zip(axes.flat, panel_positions):
            ax.set_position(position)
        fig.supxlabel(mass_axis_label, x=0.47, y=0.035, fontsize=22)
        fig.supylabel(rf"Events / {bin_width_mev:g} MeV", x=0.018, fontsize=22)
        eta_index, pt_index = divmod(flat_index, n_pt)
        _figure_header(fig, year, y=0.825)
        outputs += _save(fig, output_dir / f"mass_fit_bin_{flat_index:02d}")
        scale_factor_caption = (
            rf"The resulting scale factor is {item['scale_factor']:.4f} +/- {item['scale_factor_uncertainty']:.4f}."
            if item.get("valid")
            else "The scale-factor extraction in this bin is invalid and is not exported."
        )
        captions[f"mass_fit_bin_{flat_index:02d}"] = (
            rf"Probe bin {eta_edges[eta_index]:g} < |eta| < {eta_edges[eta_index + 1]:g}, "
            rf"{pt_edges[pt_index]:g} < pT < {pt_edges[pt_index + 1]:g} GeV. "
            "Panels are data pass, data fail, simulation pass, and simulation fail in reading order. "
            rf"The nominal data fit has chi2/ndf={item['fits']['nominal']['data']['chi2_ndf']:.3f}; "
            + scale_factor_caption
        )

    manifest = {
        "schema_version": 1,
        "measurement": result["measurement"],
        "style": {
            "cms_llabel": "Work in progress",
            "cms_rlabel": _cms_rlabel(year),
            "cms_label_fontsize": CMS_LABEL_FONT_SIZE,
            "standard_figure_inches": list(FIGURE_SIZE),
            "colorbar_figure_inches": list(COLORBAR_FIGURE_SIZE),
            "titles": False,
        },
        "files": outputs,
        "captions": captions,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trigger = commands.add_parser("trigger", help="plot a MET or photon trigger result")
    trigger.add_argument("result", type=Path)
    trigger.add_argument("--measurement", choices=("met_genuine", "photon"), required=True)
    trigger.add_argument("--output-dir", type=Path, required=True)
    tnp = commands.add_parser("tnp", help="plot a low-pT electron or muon result")
    tnp.add_argument("result", type=Path)
    tnp.add_argument("histograms", type=Path)
    tnp.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "trigger":
        payload = plot_trigger_result(args.result, args.output_dir, args.measurement)
    else:
        payload = plot_tnp_result(args.result, args.histograms, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
