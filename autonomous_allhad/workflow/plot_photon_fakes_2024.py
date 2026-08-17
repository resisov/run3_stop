#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


EXPECTED_SCHEMA = "photon_fake_2024_measurement_v1"
EXPECTED_GCR_AUDIT_SCHEMA = "photon_fake_gcr_event_key_audit_v1"
hep.style.use("CMS")
COLORS = {
    "data": "black",
    "prompt": "#4C78A8",
    "electron": "#F58518",
    "fake": "#54A24B",
    "residual": "#B279A2",
    "prediction": "#E45756",
}
PHOTON_PT_DISPLAY_EDGES = np.asarray([220.0, 300.0, 400.0, 600.0, 800.0])
PHOTON_PT_CENTERS = 0.5 * (
    PHOTON_PT_DISPLAY_EDGES[:-1] + PHOTON_PT_DISPLAY_EDGES[1:]
)
PHOTON_PT_XERR = 0.5 * (
    PHOTON_PT_DISPLAY_EDGES[1:] - PHOTON_PT_DISPLAY_EDGES[:-1]
)
PHOTON_PT_XLABEL = r"Photon $p_{\mathrm{T}}$ (GeV)"
VARIABLE_XLABELS = {
    "recoil": r"$U_{T}$ (GeV)",
    "ut": r"$U_{T}$ (GeV)",
    "met": r"$p^{\mathrm{miss}}_{T}$ (GeV)",
    "ht": r"$H_{T}$ (GeV)",
    "nb": r"$N_{b}$",
    "njet": r"$N_{j}$",
    "nfatjet": r"$N_{fj}$",
    "ntop": r"$N_{top}$",
    "nw": r"$N_{W}$",
    "jet_pt": r"Leading Jet $p_{T}$ (GeV)",
    "fatjet_pt": r"Leading FatJet $p_{T}$ (GeV)",
    "bjet_pt": r"Leading b-jet $p_{T}$ (GeV)",
}
GCR_UT_NON_QCD_PROCESSES = (
    "DY",
    "GJ",
    "ST",
    "TT",
    "VV",
    "WtoLNu",
    "Zto2Nu",
)
MAX_NOMINAL_TARGET_SUBSET_LOSS_FRACTION = 0.005


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def arrays(leaf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(leaf["sumw"], dtype=float),
        np.asarray(leaf["sumw2"], dtype=float),
        np.asarray(leaf.get("entries") or [0] * len(leaf["sumw"]), dtype=int),
    )


def edges_for(leaf: dict[str, Any]) -> np.ndarray:
    return np.asarray(leaf["bin_edges"], dtype=float)


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def widths(edges: np.ndarray) -> np.ndarray:
    return edges[1:] - edges[:-1]


def step_values(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return np.r_[values, values[-1]]


def style_physical_xaxis(
    axis: Any,
    xlabel: str,
    xmin: float,
    xmax: float,
) -> None:
    axis.set_xlim(float(xmin), float(xmax))
    axis.margins(x=0.0)
    axis.set_xlabel(xlabel, fontsize=30, loc="right")


def variable_xlabel(variable: str) -> str:
    try:
        return VARIABLE_XLABELS[variable]
    except KeyError as exc:
        raise RuntimeError(f"no approved CR/SR x-axis label for {variable}") from exc


def histogram_is_empty(leaf: dict[str, Any]) -> bool:
    values = np.asarray(leaf["sumw"], dtype=float)
    return bool(np.all(values == 0.0))


def diagnostic_distribution_is_empty(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> bool:
    diagnostic = measurement["diagnostic_histograms"][region][variable]
    leaves = [
        diagnostic["application"]["data"],
        diagnostic["application"]["prompt"],
        diagnostic["application"]["electron"],
        diagnostic["target"]["data"],
        diagnostic["target"]["prompt"],
        diagnostic["target"]["electron"],
        fake_variations(measurement, region, variable)["nominal"],
    ]
    return all(histogram_is_empty(leaf) for leaf in leaves)


def save_figure(
    fig: Any,
    output_dir: Path,
    stem: str,
    records: list[dict[str, Any]],
    category: str,
    title: str,
) -> None:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    # The CMS style enables an automatic layout pass at draw time.  Disable it
    # after the explicit tight-layout pass so saving cannot move the axes back
    # over the llabel/rlabel area for plots with wide y-axis labels.
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.86, right=0.95)
    if fig.axes:
        hep.cms.label(
            llabel="Work in progress",
            rlabel="2024 (13.6 TeV)",
            loc=0,
            ax=fig.axes[0],
        )
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    records.append(
        {
            "category": category,
            "title": title,
            "png": png.name,
            "pdf": pdf.name,
        }
    )


def fake_variations(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    fake = measurement["fake_prediction"]
    if variable == "recoil":
        return fake["histograms"][region]
    return fake["highdm_variable_histograms"][region][variable]


def nominal_target_leaf(
    nominal: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    if variable == "recoil":
        return nominal["histograms"][region]["data_obs"]["nominal"]
    return nominal["highdm_variable_histograms"][region][variable]["data_obs"][
        "nominal"
    ]


def nominal_target_subset_is_safe(
    audit: dict[str, Any] | None,
) -> tuple[bool, float | None]:
    if audit is None:
        return False, None
    if audit.get("status") == "equal":
        return True, 0.0
    comparison = audit.get("comparison") or {}
    nominal_only = int(comparison.get("nominal_only_event_keys") or 0)
    fake_only = int(comparison.get("fake_only_event_keys") or 0)
    differing_ut = int(comparison.get("differing_ut_event_keys") or 0)
    nominal_events = int((audit.get("nominal") or {}).get("unique_event_keys") or 0)
    fraction = nominal_only / nominal_events if nominal_events > 0 else None
    safe = (
        fake_only == 0
        and differing_ut == 0
        and fraction is not None
        and fraction <= MAX_NOMINAL_TARGET_SUBSET_LOSS_FRACTION
    )
    return safe, fraction


def plot_transfer_inputs(
    measurement: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    factors = measurement["measurement"]["central_transfer_factors"]
    for eta_region in ("EB", "EE"):
        selected = [
            record
            for record in factors
            if str(record["label"]).startswith(f"{eta_region}_")
        ]
        if len(selected) != len(PHOTON_PT_CENTERS):
            raise RuntimeError(
                f"{eta_region} has {len(selected)} transfer-factor bins; "
                f"expected {len(PHOTON_PT_CENTERS)}"
            )
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        for axis, suffix, title in (
            (
                axes[0],
                "pass",
                "Measurement pass: "
                r"fail $\sigma_{i\eta i\eta}$,"
                "\npass charged isolation",
            ),
            (
                axes[1],
                "fail",
                "Measurement fail: "
                r"fail $\sigma_{i\eta i\eta}$,"
                "\nfail charged isolation",
            ),
        ):
            data = np.asarray(
                [record[f"data_{suffix}"] for record in selected],
                dtype=float,
            )
            prompt = np.asarray(
                [record[f"prompt_{suffix}"] for record in selected],
                dtype=float,
            )
            electron = np.asarray(
                [record[f"electron_{suffix}"] for record in selected],
                dtype=float,
            )
            residual = np.asarray(
                [record[f"fake_{suffix}"] for record in selected],
                dtype=float,
            )
            axis.errorbar(
                PHOTON_PT_CENTERS,
                data,
                xerr=PHOTON_PT_XERR,
                yerr=np.sqrt(np.maximum(data, 0.0)),
                color=COLORS["data"],
                marker="o",
                linestyle="none",
                capsize=3,
                label="EGamma data",
                zorder=5,
            )
            axis.stairs(
                prompt,
                PHOTON_PT_DISPLAY_EDGES,
                color=COLORS["prompt"],
                label="prompt MC",
            )
            axis.stairs(
                electron,
                PHOTON_PT_DISPLAY_EDGES,
                color=COLORS["electron"],
                label="electron MC",
            )
            axis.stairs(
                residual,
                PHOTON_PT_DISPLAY_EDGES,
                color=COLORS["residual"],
                linestyle="--",
                label="data − prompt − electron",
            )
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.set_ylabel("Events / bin")
            axis.text(
                0.04,
                0.08,
                title,
                transform=axis.transAxes,
                va="bottom",
                fontsize=16,
            )
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(ncol=2, fontsize=9)
        style_physical_xaxis(
            axes[1],
            PHOTON_PT_XLABEL,
            PHOTON_PT_DISPLAY_EDGES[0],
            PHOTON_PT_DISPLAY_EDGES[-1],
        )
        axes[1].set_xticks(PHOTON_PT_DISPLAY_EDGES)
        axes[1].set_xticklabels(["220", "300", "400", "600", ""])
        save_figure(
            fig,
            output_dir,
            f"01_transfer_factor_inputs_{eta_region}",
            records,
            "transfer factor",
            f"{eta_region} measurement pass/fail inputs vs photon pT",
        )


def plot_transfer_factors(
    measurement: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    fit = measurement["measurement"]
    central_records = fit["central_transfer_factors"]
    variation_specs = (
        ("prompt_up_transfer_factors", "prompt +30%", "#4C78A8", "--"),
        ("prompt_down_transfer_factors", "prompt −30%", "#4C78A8", ":"),
        ("electron_up_transfer_factors", "electron +50%", "#F58518", "--"),
        ("electron_down_transfer_factors", "electron −50%", "#F58518", ":"),
    )
    for eta_region in ("EB", "EE"):
        indices = [
            index
            for index, record in enumerate(central_records)
            if str(record["label"]).startswith(f"{eta_region}_")
        ]
        if len(indices) != len(PHOTON_PT_CENTERS):
            raise RuntimeError(
                f"{eta_region} has {len(indices)} transfer-factor bins; "
                f"expected {len(PHOTON_PT_CENTERS)}"
            )
        selected = [central_records[index] for index in indices]
        central = np.asarray(
            [record["factor"] for record in selected],
            dtype=float,
        )
        error = np.asarray(
            [record["factor_uncertainty"] for record in selected],
            dtype=float,
        )
        sources = [str(record["source"]) for record in selected]
        fig, (axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        axis.errorbar(
            PHOTON_PT_CENTERS,
            central,
            xerr=PHOTON_PT_XERR,
            yerr=error,
            color="black",
            marker="o",
            linestyle="none",
            capsize=3,
            label="central ± TF statistical",
        )
        for key, label, color, style in variation_specs:
            values = np.asarray(
                [fit[key][index]["factor"] for index in indices],
                dtype=float,
            )
            axis.stairs(
                values,
                PHOTON_PT_DISPLAY_EDGES,
                color=color,
                linestyle=style,
                label=label,
            )
            ratio = np.divide(
                values,
                central,
                out=np.full_like(values, np.nan),
                where=central > 0.0,
            )
            ratio_axis.stairs(
                ratio,
                PHOTON_PT_DISPLAY_EDGES,
                color=color,
                linestyle=style,
            )
        for index, source in enumerate(sources):
            axis.annotate(
                source.replace("_fallback", ""),
                (PHOTON_PT_CENTERS[index], central[index]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=25,
            )
        axis.set_ylabel("Fake factor")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
        ratio_axis.axhline(1.0, color="black", linewidth=0.8)
        ratio_axis.set_ylabel("Var./nom.")
        ratio_axis.set_ylim(0.0, 2.0)
        ratio_axis.grid(alpha=0.25)
        style_physical_xaxis(
            ratio_axis,
            PHOTON_PT_XLABEL,
            PHOTON_PT_DISPLAY_EDGES[0],
            PHOTON_PT_DISPLAY_EDGES[-1],
        )
        ratio_axis.set_xticks(PHOTON_PT_DISPLAY_EDGES)
        ratio_axis.set_xticklabels(["220", "300", "400", "600", ""])
        save_figure(
            fig,
            output_dir,
            f"02_transfer_factors_{eta_region}",
            records,
            "transfer factor",
            f"{eta_region} fake factors vs photon pT",
        )


def plot_application(
    measurement: dict[str, Any],
    region: str,
    variable: str,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    components = measurement["diagnostic_histograms"][region][variable]["application"]
    data, data_var, _ = arrays(components["data"])
    prompt, prompt_var, _ = arrays(components["prompt"])
    electron, electron_var, _ = arrays(components["electron"])
    residual, residual_var, _ = arrays(components["data_minus_prompt_electron"])
    edges = edges_for(components["data"])
    x = centers(edges)
    w = widths(edges)
    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.bar(
        x,
        prompt,
        width=w,
        color=COLORS["prompt"],
        alpha=0.75,
        label="prompt MC",
    )
    axis.bar(
        x,
        electron,
        width=w,
        bottom=prompt,
        color=COLORS["electron"],
        alpha=0.75,
        label="electron MC",
    )
    axis.errorbar(
        x,
        data,
        yerr=np.sqrt(np.maximum(data_var, 0.0)),
        color=COLORS["data"],
        marker="o",
        linestyle="none",
        label="EGamma data",
    )
    lower.bar(
        x,
        residual,
        width=w,
        color=COLORS["residual"],
        alpha=0.55,
        label="data − prompt − electron",
    )
    lower.errorbar(
        x,
        residual,
        yerr=np.sqrt(np.maximum(residual_var, 0.0)),
        color=COLORS["residual"],
        marker=".",
        linestyle="none",
    )
    lower.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Application yield")
    lower.set_ylabel("Residual")
    style_physical_xaxis(
        lower,
        variable_xlabel(variable),
        edges[0],
        edges[-1],
    )
    axis.grid(axis="y", alpha=0.25)
    lower.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    save_figure(
        fig,
        output_dir,
        f"{safe_name(region)}__{safe_name(variable)}__application",
        records,
        "application",
        f"{region}/{variable} application composition",
    )


def plot_target_validation(
    measurement: dict[str, Any],
    region: str,
    variable: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    nominal: dict[str, Any] | None = None,
) -> None:
    components = measurement["diagnostic_histograms"][region][variable]["target"]
    target_data = (
        components["data"]
        if nominal is None
        else nominal_target_leaf(nominal, region, variable)
    )
    data, data_var, _ = arrays(target_data)
    prompt, prompt_var, _ = arrays(components["prompt"])
    electron, electron_var, _ = arrays(components["electron"])
    fake_leaf = fake_variations(measurement, region, variable)["nominal"]
    fake, fake_var, _ = arrays(fake_leaf)
    edges = edges_for(components["data"])
    x = centers(edges)
    w = widths(edges)
    prediction = prompt + electron + fake
    prediction_var = prompt_var + electron_var + fake_var
    ratio = np.divide(
        data,
        prediction,
        out=np.full_like(data, np.nan),
        where=prediction > 0.0,
    )
    ratio_err = np.divide(
        np.sqrt(np.maximum(data_var, 0.0)),
        prediction,
        out=np.zeros_like(data),
        where=prediction > 0.0,
    )
    pred_rel = np.divide(
        np.sqrt(np.maximum(prediction_var, 0.0)),
        prediction,
        out=np.zeros_like(prediction),
        where=prediction > 0.0,
    )
    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.bar(x, prompt, width=w, color=COLORS["prompt"], label="prompt MC")
    axis.bar(
        x,
        electron,
        width=w,
        bottom=prompt,
        color=COLORS["electron"],
        label="electron MC",
    )
    axis.bar(
        x,
        fake,
        width=w,
        bottom=prompt + electron,
        color=COLORS["fake"],
        label="data-driven fake",
    )
    axis.fill_between(
        edges,
        step_values(prediction - np.sqrt(np.maximum(prediction_var, 0.0))),
        step_values(prediction + np.sqrt(np.maximum(prediction_var, 0.0))),
        step="post",
        color="0.35",
        alpha=0.25,
        label="prediction stat.",
    )
    axis.errorbar(
        x,
        data,
        yerr=np.sqrt(np.maximum(data_var, 0.0)),
        color=COLORS["data"],
        marker="o",
        linestyle="none",
        label="target data (validation only)",
    )
    lower.errorbar(
        x,
        ratio,
        yerr=ratio_err,
        color="black",
        marker="o",
        linestyle="none",
    )
    lower.fill_between(
        edges,
        step_values(1.0 - pred_rel),
        step_values(1.0 + pred_rel),
        step="post",
        color="0.35",
        alpha=0.25,
    )
    lower.axhline(1.0, color="black", linewidth=0.8)
    finite_ratio = ratio[np.isfinite(ratio)]
    finite_error = ratio_err[np.isfinite(ratio)]
    ratio_upper = 2.0
    if len(finite_ratio):
        ratio_upper = max(
            2.0,
            min(20.0, 1.15 * float(np.max(finite_ratio + finite_error))),
        )
    lower.set_ylim(0.0, ratio_upper)
    axis.set_ylabel("Target yield")
    lower.set_ylabel("Data/pred.")
    style_physical_xaxis(
        lower,
        variable_xlabel(variable),
        edges[0],
        edges[-1],
    )
    axis.grid(axis="y", alpha=0.25)
    lower.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    save_figure(
        fig,
        output_dir,
        f"{safe_name(region)}__{safe_name(variable)}__target_validation",
        records,
        "target validation",
        f"{region}/{variable} target validation",
    )


def plot_systematics(
    measurement: dict[str, Any],
    region: str,
    variable: str,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    variations = fake_variations(measurement, region, variable)
    nominal, nominal_var, _ = arrays(variations["nominal"])
    edges = edges_for(variations["nominal"])
    x = centers(edges)
    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.step(edges, step_values(nominal), where="post", color="black", label="nominal")
    axis.fill_between(
        edges,
        step_values(np.maximum(0.0, nominal - np.sqrt(np.maximum(nominal_var, 0.0)))),
        step_values(nominal + np.sqrt(np.maximum(nominal_var, 0.0))),
        step="post",
        color="0.4",
        alpha=0.2,
        label="nominal stat.",
    )
    style = {
        "photonFakeTFUp": ("#9467BD", "--"),
        "photonFakeTFDown": ("#9467BD", ":"),
        "photonFakePromptUp": ("#4C78A8", "--"),
        "photonFakePromptDown": ("#4C78A8", ":"),
        "photonFakeElectronUp": ("#F58518", "--"),
        "photonFakeElectronDown": ("#F58518", ":"),
        "photonFakeClosureUp": ("#E45756", "--"),
        "photonFakeClosureDown": ("#E45756", ":"),
    }
    for name, (color, line_style) in style.items():
        values, _, _ = arrays(variations[name])
        axis.step(
            edges,
            step_values(values),
            where="post",
            color=color,
            linestyle=line_style,
            label=name,
        )
        ratio = np.divide(
            values,
            nominal,
            out=np.full_like(values, np.nan),
            where=nominal > 0.0,
        )
        lower.step(
            edges,
            step_values(ratio),
            where="post",
            color=color,
            linestyle=line_style,
        )
    axis.set_ylabel("Predicted fake")
    lower.set_ylabel("Var./nom.")
    style_physical_xaxis(
        lower,
        variable_xlabel(variable),
        edges[0],
        edges[-1],
    )
    lower.axhline(1.0, color="black", linewidth=0.8)
    lower.set_ylim(0.0, 2.0)
    axis.grid(axis="y", alpha=0.25)
    lower.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=7)
    save_figure(
        fig,
        output_dir,
        f"{safe_name(region)}__{safe_name(variable)}__systematics",
        records,
        "systematics",
        f"{region}/{variable} systematic variations",
    )


def plot_gcr_ut_qcd_replacement(
    measurement: dict[str, Any],
    nominal: dict[str, Any],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    nominal_ut = nominal["highdm_variable_histograms"]["GCR"]["ut"]
    nominal_edges = np.asarray(
        nominal["highdm_distribution_variable_specs"]["ut"]["bins"],
        dtype=float,
    )
    fake_leaf = measurement["fake_prediction"]["highdm_variable_histograms"][
        "GCR"
    ]["ut"]["nominal"]
    fake, fake_var, _ = arrays(fake_leaf)
    fake_edges = edges_for(fake_leaf)
    if not np.array_equal(nominal_edges, fake_edges):
        raise RuntimeError(
            "GCR U_T bin mismatch between nominal and photon-fake measurement: "
            f"{nominal_edges.tolist()} != {fake_edges.tolist()}"
        )
    qcd_origins = measurement["qcd_target_origin_histograms"]["GCR"]["ut"]
    qcd_sidecar_all, _, _ = arrays(qcd_origins["all"])
    qcd_sidecar_prompt, _, _ = arrays(qcd_origins["prompt"])
    qcd_sidecar_electron, _, _ = arrays(qcd_origins["electron"])
    qcd_sidecar_fake, _, _ = arrays(qcd_origins["fake"])

    data, data_var, _ = arrays(nominal_ut["data_obs"]["nominal"])
    qcd_mc, qcd_mc_var, _ = arrays(nominal_ut["QCD"]["nominal"])
    non_qcd = np.zeros_like(data)
    non_qcd_var = np.zeros_like(data)
    for process in GCR_UT_NON_QCD_PROCESSES:
        values, variances, _ = arrays(nominal_ut[process]["nominal"])
        non_qcd += values
        non_qcd_var += variances

    qcd_nonfake_fraction = np.divide(
        qcd_sidecar_prompt + qcd_sidecar_electron,
        qcd_sidecar_all,
        out=np.zeros_like(qcd_sidecar_all),
        where=qcd_sidecar_all != 0.0,
    )
    qcd_nonfake_fraction = np.clip(qcd_nonfake_fraction, 0.0, 1.0)
    retained_qcd_nonfake = qcd_mc * qcd_nonfake_fraction
    retained_qcd_nonfake_var = qcd_mc_var * np.square(qcd_nonfake_fraction)
    nominal_total = non_qcd + qcd_mc
    nominal_total_var = non_qcd_var + qcd_mc_var
    full_replaced_total = non_qcd + fake
    full_replaced_total_var = non_qcd_var + fake_var
    origin_replaced_total = non_qcd + retained_qcd_nonfake + fake
    origin_replaced_total_var = (
        non_qcd_var + retained_qcd_nonfake_var + fake_var
    )
    nominal_ratio = np.divide(
        data,
        nominal_total,
        out=np.full_like(data, np.nan),
        where=nominal_total > 0.0,
    )
    full_replaced_ratio = np.divide(
        data,
        full_replaced_total,
        out=np.full_like(data, np.nan),
        where=full_replaced_total > 0.0,
    )
    origin_replaced_ratio = np.divide(
        data,
        origin_replaced_total,
        out=np.full_like(data, np.nan),
        where=origin_replaced_total > 0.0,
    )
    nominal_ratio_error = np.divide(
        np.sqrt(np.maximum(data_var, 0.0)),
        nominal_total,
        out=np.zeros_like(data),
        where=nominal_total > 0.0,
    )
    full_replaced_ratio_error = np.divide(
        np.sqrt(np.maximum(data_var, 0.0)),
        full_replaced_total,
        out=np.zeros_like(data),
        where=full_replaced_total > 0.0,
    )
    origin_replaced_ratio_error = np.divide(
        np.sqrt(np.maximum(data_var, 0.0)),
        origin_replaced_total,
        out=np.zeros_like(data),
        where=origin_replaced_total > 0.0,
    )
    nominal_rel_stat = np.divide(
        np.sqrt(np.maximum(nominal_total_var, 0.0)),
        nominal_total,
        out=np.zeros_like(data),
        where=nominal_total > 0.0,
    )
    origin_replaced_rel_stat = np.divide(
        np.sqrt(np.maximum(origin_replaced_total_var, 0.0)),
        origin_replaced_total,
        out=np.zeros_like(data),
        where=origin_replaced_total > 0.0,
    )

    x = centers(nominal_edges)
    w = widths(nominal_edges)
    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.bar(
        x,
        non_qcd,
        width=w,
        color=COLORS["prompt"],
        label="non-QCD MC",
    )
    axis.bar(
        x,
        retained_qcd_nonfake,
        width=w,
        bottom=non_qcd,
        color=COLORS["residual"],
        label="QCD truth-prompt/electron MC",
    )
    axis.bar(
        x,
        fake,
        width=w,
        bottom=non_qcd + retained_qcd_nonfake,
        color=COLORS["fake"],
        label="data-driven fake",
    )
    axis.step(
        nominal_edges,
        step_values(nominal_total),
        where="post",
        color=COLORS["residual"],
        linewidth=2.0,
        linestyle="--",
        label="nominal total",
    )
    axis.step(
        nominal_edges,
        step_values(full_replaced_total),
        where="post",
        color=COLORS["prediction"],
        linewidth=1.8,
        linestyle=":",
        label="replace entire QCD (rejected)",
    )
    axis.fill_between(
        nominal_edges,
        step_values(
            np.maximum(
                1.0e-9,
                origin_replaced_total
                - np.sqrt(np.maximum(origin_replaced_total_var, 0.0)),
            )
        ),
        step_values(
            origin_replaced_total
            + np.sqrt(np.maximum(origin_replaced_total_var, 0.0))
        ),
        step="post",
        color="0.35",
        alpha=0.25,
        label="truth-fake-only candidate stat.",
    )
    axis.errorbar(
        x,
        data,
        yerr=np.sqrt(np.maximum(data_var, 0.0)),
        color=COLORS["data"],
        marker="o",
        linestyle="none",
        label="data",
        zorder=5,
    )
    axis.set_yscale("log")
    axis.set_ylim(bottom=0.1)
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)

    offset = 0.04 * w
    lower.errorbar(
        x - offset,
        nominal_ratio,
        yerr=nominal_ratio_error,
        color=COLORS["residual"],
        marker="s",
        markerfacecolor="none",
        linestyle="none",
        label="nominal",
    )
    lower.errorbar(
        x,
        full_replaced_ratio,
        yerr=full_replaced_ratio_error,
        color=COLORS["prediction"],
        marker="^",
        markerfacecolor="none",
        linestyle="none",
        label="entire QCD (rejected)",
    )
    lower.errorbar(
        x + offset,
        origin_replaced_ratio,
        yerr=origin_replaced_ratio_error,
        color=COLORS["fake"],
        marker="o",
        linestyle="none",
        label="truth-fake only",
    )
    lower.fill_between(
        nominal_edges,
        step_values(1.0 - nominal_rel_stat),
        step_values(1.0 + nominal_rel_stat),
        step="post",
        facecolor="none",
        edgecolor=COLORS["residual"],
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
    )
    lower.fill_between(
        nominal_edges,
        step_values(1.0 - origin_replaced_rel_stat),
        step_values(1.0 + origin_replaced_rel_stat),
        step="post",
        color="0.35",
        alpha=0.20,
    )
    lower.axhline(1.0, color="black", linewidth=0.8)
    finite = np.r_[
        nominal_ratio[np.isfinite(nominal_ratio)],
        full_replaced_ratio[np.isfinite(full_replaced_ratio)],
        origin_replaced_ratio[np.isfinite(origin_replaced_ratio)],
    ]
    lower.set_ylim(0.0, max(2.0, 1.18 * float(np.max(finite))))
    lower.set_ylabel("Data/pred.")
    lower.grid(axis="y", alpha=0.25)
    lower.legend(ncol=3, fontsize=7)
    style_physical_xaxis(
        lower,
        variable_xlabel("ut"),
        nominal_edges[0],
        nominal_edges[-1],
    )
    save_figure(
        fig,
        output_dir,
        "GCR__ut__qcd_replacement_comparison",
        records,
        "QCD replacement",
        "GCR U_T: nominal, entire-QCD, and truth-fake-only policies",
    )

    write_json(
        output_dir / "GCR__ut__qcd_replacement_summary.json",
        {
            "schema_version": "photon_fake_gcr_ut_replacement_v1",
            "bin_edges": nominal_edges.tolist(),
            "data": data.tolist(),
            "non_qcd_mc": non_qcd.tolist(),
            "qcd_mc": qcd_mc.tolist(),
            "qcd_sidecar_origins": {
                "all": qcd_sidecar_all.tolist(),
                "prompt": qcd_sidecar_prompt.tolist(),
                "electron": qcd_sidecar_electron.tolist(),
                "fake": qcd_sidecar_fake.tolist(),
            },
            "qcd_nonfake_fraction": qcd_nonfake_fraction.tolist(),
            "retained_qcd_nonfake": retained_qcd_nonfake.tolist(),
            "data_driven_fake": fake.tolist(),
            "nominal_total": nominal_total.tolist(),
            "entire_qcd_replaced_total": full_replaced_total.tolist(),
            "truth_fake_only_replaced_total": origin_replaced_total.tolist(),
            "data_over_nominal_total": nominal_ratio.tolist(),
            "data_over_entire_qcd_replaced_total": full_replaced_ratio.tolist(),
            "data_over_truth_fake_only_replaced_total": (
                origin_replaced_ratio.tolist()
            ),
            "integrals": {
                "data": float(np.sum(data)),
                "non_qcd_mc": float(np.sum(non_qcd)),
                "qcd_mc": float(np.sum(qcd_mc)),
                "retained_qcd_nonfake": float(np.sum(retained_qcd_nonfake)),
                "data_driven_fake": float(np.sum(fake)),
                "nominal_total": float(np.sum(nominal_total)),
                "entire_qcd_replaced_total": float(
                    np.sum(full_replaced_total)
                ),
                "truth_fake_only_replaced_total": float(
                    np.sum(origin_replaced_total)
                ),
                "data_over_nominal_total": float(
                    np.sum(data) / np.sum(nominal_total)
                ),
                "data_over_entire_qcd_replaced_total": float(
                    np.sum(data) / np.sum(full_replaced_total)
                ),
                "data_over_truth_fake_only_replaced_total": float(
                    np.sum(data) / np.sum(origin_replaced_total)
                ),
            },
            "replacement_policy": (
                "The entire-QCD replacement is shown and rejected. The "
                "truth-fake-only candidate retains the nominal QCD yield "
                "multiplied by the sidecar truth-prompt/electron fraction and "
                "replaces only the truth-fake fraction with the data-driven "
                "prediction; all non-QCD processes are retained."
            ),
            "measurement_status": measurement.get("status"),
        },
    )


def plot_closure(
    measurement: dict[str, Any],
    key: str,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    closure = measurement["closure"]["distributions"][key]
    target, target_var, _ = arrays(closure["target_histogram"])
    predicted, predicted_var, _ = arrays(closure["predicted_histogram"])
    edges = edges_for(closure["target_histogram"])
    x = centers(edges)
    ratio = np.divide(
        target,
        predicted,
        out=np.full_like(target, np.nan),
        where=predicted > 0.0,
    )
    ratio_var = np.zeros_like(target)
    valid = predicted > 0.0
    ratio_var[valid] = (
        target_var[valid] / np.square(predicted[valid])
        + np.square(target[valid])
        * predicted_var[valid]
        / np.power(predicted[valid], 4)
    )
    region, variable = key.split("/", 1)
    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.errorbar(
        x,
        target,
        yerr=np.sqrt(np.maximum(target_var, 0.0)),
        color="black",
        marker="o",
        linestyle="none",
        label="QCD truth-fake target",
    )
    axis.step(
        edges,
        step_values(predicted),
        where="post",
        color=COLORS["prediction"],
        label="predicted from QCD application",
    )
    axis.fill_between(
        edges,
        step_values(np.maximum(0.0, predicted - np.sqrt(np.maximum(predicted_var, 0.0)))),
        step_values(predicted + np.sqrt(np.maximum(predicted_var, 0.0))),
        step="post",
        color=COLORS["prediction"],
        alpha=0.2,
    )
    lower.errorbar(
        x,
        ratio,
        yerr=np.sqrt(np.maximum(ratio_var, 0.0)),
        color="black",
        marker="o",
        linestyle="none",
    )
    lower.axhline(1.0, color="black", linewidth=0.8)
    lower.set_ylim(0.0, 2.0)
    axis.set_ylabel("QCD fake yield")
    lower.set_ylabel("Target/pred.")
    style_physical_xaxis(
        lower,
        variable_xlabel(variable),
        edges[0],
        edges[-1],
    )
    axis.grid(axis="y", alpha=0.25)
    lower.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=9)
    ratio_text = closure.get("target_over_prediction")
    if ratio_text is not None:
        axis.text(
            0.02,
            0.88,
            f"Integral target/prediction = {float(ratio_text):.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=12,
        )
    save_figure(
        fig,
        output_dir,
        f"{safe_name(region)}__{safe_name(variable)}__qcd_closure",
        records,
        "QCD closure",
        f"{key} QCD truth-fake closure",
    )


def write_index(
    output_dir: Path,
    measurement: dict[str, Any],
    records: list[dict[str, Any]],
    gcr_audit: dict[str, Any] | None,
    nominal_target_authoritative: bool,
    excluded_distributions: list[dict[str, Any]],
    evaluation: dict[str, Any] | None,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["category"], []).append(record)
    sections = []
    for category, items in grouped.items():
        cards = []
        for item in items:
            cards.append(
                "<article><h3>{title}</h3>"
                "<a href='{png}'><img src='{png}' loading='lazy'></a>"
                "<p><a href='{pdf}'>PDF</a></p></article>".format(
                    title=html.escape(item["title"]),
                    png=html.escape(item["png"]),
                    pdf=html.escape(item["pdf"]),
                )
            )
        sections.append(
            f"<h2>{html.escape(category)}</h2><div class='grid'>"
            + "".join(cards)
            + "</div>"
        )
    coverage = measurement.get("coverage") or {}
    measurement_complete = measurement.get("status") == "complete"
    audit_equal = gcr_audit is not None and gcr_audit.get("status") == "equal"
    safe_subset, subset_fraction = nominal_target_subset_is_safe(gcr_audit)
    validated = measurement_complete and (
        audit_equal or (nominal_target_authoritative and safe_subset)
    )
    if validated:
        document_title = "2024 photon-fake background estimation"
        heading = "2024 photon-fake background estimation"
        note_class = "note complete"
        if audit_equal:
            note = (
                "Complete campaign measurement with exact run–lumi–event and "
                "U_T agreement between the nominal and photon-fake GCR data."
            )
        else:
            note = (
                "Complete B/C/D sideband measurement. The trusted nominal "
                "real_subset_worker.py target data are used for A-region "
                "validation; the sidecar target is a strict subset differing "
                f"by {100.0 * float(subset_fraction):.3f}%."
            )
    else:
        document_title = "Photon fake incomplete diagnostic"
        heading = "2024 photon-fake workflow diagnostic"
        note_class = "note"
        note = (
            "This result is not validated for physics use. A complete campaign "
            "and an equal nominal/fake GCR event-key audit are both required."
        )
    audit_status = "not supplied" if gcr_audit is None else str(
        gcr_audit.get("status")
    )
    excluded_html = ""
    if excluded_distributions:
        items = "".join(
            "<li><strong>{distribution}</strong>: {reason}</li>".format(
                distribution=html.escape(str(item["distribution"])),
                reason=html.escape(str(item["reason"])),
            )
            for item in excluded_distributions
        )
        excluded_html = (
            "<section class='note'><h2>Excluded empty distributions</h2>"
            "<p>These are configuration/binning failures, not zero-yield "
            "physics results.</p><ul>"
            + items
            + "</ul></section>"
        )
    decision_html = ""
    if evaluation is not None:
        decision = evaluation["decision"]
        metrics = evaluation["results"]["GCR/ut"]["metrics"]
        rows = []
        labels = {
            "nominal": "Nominal",
            "replace_entire_qcd": "Replace entire QCD",
            "replace_truth_fake_only": "Replace truth-fake only",
        }
        for key in (
            "nominal",
            "replace_entire_qcd",
            "replace_truth_fake_only",
        ):
            record = metrics[key]
            rows.append(
                "<tr><td>{label}</td><td>{prediction:.2f}</td>"
                "<td>{ratio:.4f}</td><td>{deviance:.1f}</td>"
                "<td>{chi2:.1f}</td></tr>".format(
                    label=html.escape(labels[key]),
                    prediction=float(record["integral_prediction"]),
                    ratio=float(record["integral_data_over_prediction"]),
                    deviance=float(record["poisson_deviance"]),
                    chi2=float(record["chi2_data_plus_mcstat"]),
                )
            )
        decision_html = (
            "<section class='note'><h2>Adoption decision</h2>"
            "<p><strong>Entire-QCD replacement: {full}</strong><br>"
            "<strong>Truth-fake-only replacement: {origin}</strong></p>"
            "<p>The full-QCD replacement is not adopted. The truth-fake-only "
            "result is diagnostic until the QCD/G+jets prompt-photon overlap "
            "is resolved.</p>"
            "<table><thead><tr><th>GCR U<sub>T</sub> prediction</th>"
            "<th>Integral</th><th>Data/MC</th><th>Poisson deviance</th>"
            "<th>&chi;<sup>2</sup></th></tr></thead><tbody>{rows}</tbody>"
            "</table></section>"
        ).format(
            full=html.escape(str(decision["entire_qcd_replacement"])),
            origin=html.escape(str(decision["truth_fake_only_replacement"])),
            rows="".join(rows),
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>{document_title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f5f6f7;color:#1b1f23}}
.note{{padding:14px;background:#fff4ce;border:1px solid #e2b93b}}
.note.complete{{background:#e8f4ee;border-color:#58a276}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
article{{background:white;padding:12px;border:1px solid #d8dee4;border-radius:6px}}
img{{width:100%;height:auto}} h2{{margin-top:34px}} h3{{font-size:15px}}
table{{border-collapse:collapse;background:white}} th,td{{padding:7px 11px;border:1px solid #b8c0c8;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
</style></head><body>
<h1>{heading}</h1>
<div class="{note_class}"><strong>Measurement: {status}</strong><br>
<strong>GCR event-key audit: {audit_status}</strong><br>
{note}<br>Observed sidecars: {observed} / expected {expected}.</div>
{decision_html}
{excluded_html}
{sections}
</body></html>
""".format(
        document_title=html.escape(document_title),
        heading=html.escape(heading),
        note_class=note_class,
        status=html.escape(str(measurement.get("status"))),
        audit_status=html.escape(audit_status),
        note=html.escape(note),
        observed=html.escape(str(coverage.get("observed_sidecars"))),
        expected=html.escape(str(coverage.get("expected_sidecars"))),
        decision_html=decision_html,
        excluded_html=excluded_html,
        sections="".join(sections),
    )
    (output_dir / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot intermediate diagnostics for the 2024 photon-fake measurement."
    )
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument(
        "--nominal",
        type=Path,
        help=(
            "Optional nominal histogram payload. When provided, add the key "
            "GCR U_T plot comparing QCD MC with its data-driven replacement."
        ),
    )
    parser.add_argument(
        "--gcr-audit",
        type=Path,
        help=(
            "Nominal/fake GCR run-lumi-event audit. A strict sub-0.5% sidecar "
            "target subset is allowed only when the nominal target is used."
        ),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="Optional origin-aware Data/MC adoption evaluation JSON.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    measurement_path = args.measurement.absolute()
    output_dir = args.output_dir.absolute()
    measurement = read_json(measurement_path)
    if measurement.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"unexpected measurement schema: {measurement.get('schema_version')}"
        )
    if "diagnostic_histograms" not in measurement:
        raise RuntimeError("measurement has no diagnostic_histograms")
    gcr_audit = None
    if args.gcr_audit is not None:
        gcr_audit = read_json(args.gcr_audit.absolute())
        if gcr_audit.get("schema_version") != EXPECTED_GCR_AUDIT_SCHEMA:
            raise RuntimeError(
                f"unexpected GCR audit schema: {gcr_audit.get('schema_version')}"
            )
    nominal_payload = (
        None if args.nominal is None else read_json(args.nominal.absolute())
    )
    evaluation = (
        None
        if args.evaluation is None
        else read_json(args.evaluation.absolute())
    )
    safe_subset, subset_fraction = nominal_target_subset_is_safe(gcr_audit)
    if measurement.get("status") == "complete":
        if gcr_audit is None:
            raise RuntimeError(
                "refusing to plot a complete measurement without a GCR audit"
            )
        if gcr_audit.get("status") != "equal" and (
            nominal_payload is None or not safe_subset
        ):
            raise RuntimeError(
                "a non-equal target audit is allowed only for a safe strict "
                "subset when authoritative nominal target data are supplied"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    excluded_distributions: list[dict[str, Any]] = []
    plot_transfer_inputs(measurement, output_dir, records)
    plot_transfer_factors(measurement, output_dir, records)
    for region, variables in measurement["diagnostic_histograms"].items():
        for variable in variables:
            if diagnostic_distribution_is_empty(measurement, region, variable):
                excluded_distributions.append(
                    {
                        "distribution": f"{region}/{variable}",
                        "reason": (
                            "all target, application, contamination, and fake "
                            "histograms are zero; real_subset_worker.py selects "
                            "GCR pTmiss < 250 GeV while the supplied histogram "
                            "bins start at 250 GeV"
                            if region == "GCR" and variable == "met"
                            else "all supplied histogram components are zero"
                        ),
                    }
                )
                continue
            plot_application(
                measurement,
                region,
                variable,
                output_dir,
                records,
            )
            plot_target_validation(
                measurement,
                region,
                variable,
                output_dir,
                records,
                nominal_payload,
            )
            plot_systematics(
                measurement,
                region,
                variable,
                output_dir,
                records,
            )
    for key in sorted((measurement.get("closure") or {}).get("distributions") or {}):
        closure = measurement["closure"]["distributions"][key]
        if histogram_is_empty(
            closure["target_histogram"]
        ) and histogram_is_empty(closure["predicted_histogram"]):
            continue
        plot_closure(measurement, key, output_dir, records)
    if args.nominal is not None:
        plot_gcr_ut_qcd_replacement(
            measurement,
            nominal_payload,
            output_dir,
            records,
        )
    write_index(
        output_dir,
        measurement,
        records,
        gcr_audit,
        nominal_target_authoritative=nominal_payload is not None,
        excluded_distributions=excluded_distributions,
        evaluation=evaluation,
    )
    summary = {
        "schema_version": "photon_fake_2024_plot_summary_v1",
        "status": "complete",
        "measurement": str(measurement_path),
        "measurement_sha256": sha256(measurement_path),
        "measurement_status": measurement.get("status"),
        "plot_count": len(records),
        "png_count": len(list(output_dir.glob("*.png"))),
        "pdf_count": len(list(output_dir.glob("*.pdf"))),
        "index": str(output_dir / "index.html"),
        "plots": records,
        "closure_status": (measurement.get("closure") or {}).get("status"),
        "closure_uncertainty": (measurement.get("closure") or {}).get(
            "assigned_relative_nonclosure"
        ),
        "gcr_event_key_audit": (
            None if args.gcr_audit is None else str(args.gcr_audit.absolute())
        ),
        "gcr_event_key_audit_status": (
            None if gcr_audit is None else gcr_audit.get("status")
        ),
        "nominal_target_authoritative": nominal_payload is not None,
        "nominal_target_subset_loss_fraction": subset_fraction,
        "excluded_distributions": excluded_distributions,
        "evaluation": (
            None if args.evaluation is None else str(args.evaluation.absolute())
        ),
        "adoption_decision": (
            None if evaluation is None else evaluation.get("decision")
        ),
    }
    write_json(output_dir / "plot_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
