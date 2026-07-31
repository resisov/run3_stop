#!/usr/bin/env python3
"""Plot a UT-binned on/off-Z matrix measurement and its DYCR movement."""

from __future__ import annotations

import argparse
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


hep.style.use("CMS")
CMS_LABEL = {"llabel": "Work in progress", "rlabel": "2024 (13.6 TeV)"}
COLORS = {"dy": "#35B6B4", "other": "#6A625F", "data": "black"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def geometry(measurement: dict[str, Any], regime: str, channel: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if regime == "highdm":
        source = measurement["rz_high_ut"]
    else:
        source = measurement["rz_low_ut"]
    edges = np.asarray(source["edges"], dtype=float)
    records = list(source["channels"][channel]["bins"])
    if len(records) != len(edges) - 1:
        raise ValueError("UT geometry and matrix-bin count differ")
    return edges, records


def leaf(record: dict[str, Any], key: str) -> tuple[float, float]:
    source = (record.get("inputs") or {}).get(key) or {}
    return float(source.get("sumw", 0.0)), float(source.get("sumw2", 0.0))


def factor_arrays(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [float(record[key]) if record.get("status") == "complete" else np.nan for record in records]
    )
    errors = np.asarray(
        [float(record[f"{key}_stat"]) if record.get("status") == "complete" else np.nan for record in records]
    )
    return values, errors


def regime_label(regime: str) -> str:
    return r"High-$\Delta m$" if regime == "highdm" else r"Low-$\Delta m$"


def channel_label(channel: str) -> str:
    return "DY(ee)" if channel == "DY2E" else r"DY($\mu\mu$)"


def style_axis(axis: plt.Axes, edges: np.ndarray) -> None:
    axis.set_xlim(float(edges[0]), float(edges[-1]))
    axis.set_xmargin(0)
    axis.tick_params(which="major", direction="in", top=True, right=True, labelsize=23, length=9)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=5)
    axis.minorticks_on()


def plot_factor(
    edges: np.ndarray,
    records: list[dict[str, Any]],
    key: str,
    regime: str,
    channel: str,
    output_dir: Path,
) -> list[str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    values, errors = factor_arrays(records, key)
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    ax.errorbar(
        centers,
        values,
        xerr=widths,
        yerr=errors,
        fmt="o",
        color="#D62728" if key == "RZ" else "#1F77B4",
        ms=7,
        lw=2.4,
        capsize=4,
        label="Statistical uncertainty",
    )
    ax.axhline(1.0, color="0.35", ls="--", lw=1.7)
    ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30, loc="right")
    ax.set_ylabel(r"$R_Z$" if key == "RZ" else r"$R_{\mathrm{non-DY}}$", fontsize=32)
    style_axis(ax, edges)
    finite = np.isfinite(values) & np.isfinite(errors)
    if np.any(finite):
        low = float(np.nanmin(values[finite] - errors[finite]))
        high = float(np.nanmax(values[finite] + errors[finite]))
        span = max(high - min(low, 0.0), 0.3)
        ax.set_ylim(max(0.0, low - 0.12 * span), high + 0.25 * span)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    ax.text(
        0.035,
        0.075,
        f"{regime_label(regime)} {channel_label(channel)}\n$N_b \\geq 1$ inclusive",
        transform=ax.transAxes,
        fontsize=22,
        va="bottom",
    )
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"{regime}_{channel.lower()}_{key.lower()}_ut")


def window_arrays(records: list[dict[str, Any]], window: str, stage: str) -> dict[str, np.ndarray]:
    data = np.asarray([leaf(record, f"data_{window}")[0] for record in records])
    data2 = np.asarray([leaf(record, f"data_{window}")[1] for record in records])
    z = np.asarray([leaf(record, f"zll_{window}")[0] for record in records])
    z2 = np.asarray([leaf(record, f"zll_{window}")[1] for record in records])
    other = np.asarray([leaf(record, f"other_{window}")[0] for record in records])
    other2 = np.asarray([leaf(record, f"other_{window}")[1] for record in records])
    if stage == "prefit":
        z_plot = z
        other_plot = other
        variance = z2 + other2
    else:
        rz, _ = factor_arrays(records, "RZ")
        rt, _ = factor_arrays(records, "RT")
        template_index = (0, 2) if window == "on" else (1, 3)
        profiled = [record.get("profiled_templates") for record in records]
        z_template = np.asarray(
            [
                float(item[template_index[0]]) if item is not None else z[index]
                for index, item in enumerate(profiled)
            ]
        )
        other_template = np.asarray(
            [
                float(item[template_index[1]]) if item is not None else other[index]
                for index, item in enumerate(profiled)
            ]
        )
        z_plot = rz * z_template
        other_plot = rt * other_template
        variance = np.zeros_like(z_plot)
        for index, record in enumerate(records):
            if record.get("status") != "complete":
                variance[index] = np.nan
                continue
            covariance = np.asarray(record["covariance"], dtype=float)
            template = np.asarray([z[index], other[index]], dtype=float)
            variance[index] = float(template @ covariance @ template)
    return {
        "data": data,
        "data2": data2,
        "z": z_plot,
        "other": other_plot,
        "total": z_plot + other_plot,
        "variance": np.maximum(variance, 0.0),
    }


def plot_dycr(
    edges: np.ndarray,
    records: list[dict[str, Any]],
    window: str,
    stage: str,
    regime: str,
    channel: str,
    output_dir: Path,
) -> list[str]:
    arrays = window_arrays(records, window, stage)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    total_error = np.sqrt(arrays["variance"])
    data_error = np.sqrt(np.maximum(arrays["data2"], 0.0))
    valid = np.isfinite(arrays["total"]) & (arrays["total"] > 0.0)
    ratio = np.full_like(arrays["data"], np.nan)
    ratio_error = np.full_like(arrays["data"], np.nan)
    relative = np.full_like(arrays["total"], np.nan)
    ratio[valid] = arrays["data"][valid] / arrays["total"][valid]
    ratio_error[valid] = data_error[valid] / arrays["total"][valid]
    relative[valid] = total_error[valid] / arrays["total"][valid]

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(10.2, 10.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
    )
    ax.stairs(
        arrays["z"], edges, baseline=0.0, fill=True, color=COLORS["dy"],
        edgecolor="black", linewidth=0.7, label="DY",
    )
    ax.stairs(
        arrays["total"], edges, baseline=arrays["z"], fill=True, color=COLORS["other"],
        edgecolor="black", linewidth=0.7, label="Others",
    )
    lower = np.maximum(arrays["total"] - total_error, 1.0e-9)
    ax.stairs(
        arrays["total"] + total_error,
        edges,
        baseline=lower,
        fill=True,
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        label="Stat. unc.",
    )
    ax.errorbar(
        centers, arrays["data"], xerr=widths, yerr=data_error, fmt="o", color="black",
        ms=6, lw=2.0, capsize=2, label="Data",
    )
    rax.stairs(
        1.0 + np.nan_to_num(relative), edges,
        baseline=1.0 - np.nan_to_num(relative), fill=True,
        color="0.72", alpha=0.55, linewidth=0.0,
    )
    rax.errorbar(
        centers[valid], ratio[valid], xerr=widths[valid], yerr=ratio_error[valid],
        fmt="o", color="black", ms=6, lw=2.0, capsize=2,
    )
    rax.axhline(1.0, color="black", lw=1.4)
    ax.set_ylabel("Events / bin", fontsize=30)
    rax.set_ylabel("Data/MC", fontsize=25)
    rax.set_xlabel(r"$U_T$ (GeV)", fontsize=30, loc="right")
    ax.set_yscale("log")
    positive = np.concatenate([arrays["data"][arrays["data"] > 0.0], arrays["total"][arrays["total"] > 0.0]])
    if len(positive):
        ax.set_ylim(max(float(np.min(positive)) * 0.25, 0.05), float(np.max(positive)) * 8.0)
    rax.set_ylim(0.0, 2.0)
    for axis in (ax, rax):
        style_axis(axis, edges)
    handles, labels = ax.get_legend_handles_labels()
    order = ["Stat. unc.", "Others", "DY", "Data"]
    ordered = [(handles[labels.index(label)], label) for label in order if label in labels]
    ax.legend(
        [item[0] for item in ordered], [item[1] for item in ordered],
        frameon=False, fontsize=17, ncol=2, loc="upper right",
    )
    stage_label = "Prefit" if stage == "prefit" else "Profile-fit reconstruction"
    window_label = "on-$Z$" if window == "on" else "off-$Z$"
    ax.text(
        0.035,
        0.055,
        f"{regime_label(regime)} {channel_label(channel)}\n$N_b \\geq 1$, {window_label}, {stage_label}",
        transform=ax.transAxes,
        fontsize=19,
        va="bottom",
    )
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"{regime}_{channel.lower()}_{window}z_{stage}_ut")


def plot_movement(
    edges: np.ndarray,
    records: list[dict[str, Any]],
    regime: str,
    channel: str,
    output_dir: Path,
) -> list[str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    styles = {
        ("on", "prefit"): ("#D62728", "o", "none", "on-$Z$ prefit"),
        ("on", "matrix"): ("#D62728", "o", "#D62728", "on-$Z$ profile-fit reconstruction"),
        ("off", "prefit"): ("#1F77B4", "s", "none", "off-$Z$ prefit"),
        ("off", "matrix"): ("#1F77B4", "s", "#1F77B4", "off-$Z$ profile-fit reconstruction"),
    }
    for window in ("on", "off"):
        for stage in ("prefit", "matrix"):
            arrays = window_arrays(records, window, stage)
            valid = np.isfinite(arrays["total"]) & (arrays["total"] > 0.0)
            ratio = np.full_like(arrays["data"], np.nan)
            error = np.full_like(arrays["data"], np.nan)
            ratio[valid] = arrays["data"][valid] / arrays["total"][valid]
            error[valid] = np.sqrt(np.maximum(arrays["data2"][valid], 0.0)) / arrays["total"][valid]
            color, marker, markerface, label = styles[(window, stage)]
            ax.errorbar(
                centers[valid], ratio[valid], xerr=widths[valid], yerr=error[valid],
                fmt=marker, ls="none", color=color, markerfacecolor=markerface,
                markeredgewidth=1.7, ms=7, lw=2.0, capsize=3, label=label,
            )
    ax.axhline(1.0, color="0.25", lw=1.5)
    ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30, loc="right")
    ax.set_ylabel("Data / prediction", fontsize=30)
    ax.set_ylim(0.0, 2.0)
    style_axis(ax, edges)
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=17, loc="upper right")
    ax.text(
        0.035,
        0.075,
        f"{regime_label(regime)} {channel_label(channel)}\n$N_b \\geq 1$ inclusive",
        transform=ax.transAxes,
        fontsize=22,
        va="bottom",
    )
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"{regime}_{channel.lower()}_dycr_ratio_movement_ut")


def sum_histograms(source: dict[str, Any], channel: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    nodes = [((source.get(channel) or {}).get(group) or {}) for group in ("Nb1", "Nb2plus")]
    first = next((payload for node in nodes for payload in node.values() if payload), None)
    if first is None:
        raise ValueError("no mll histogram inputs")
    edges = np.asarray(first["edges"], dtype=float)
    size = len(edges) - 1
    output: dict[str, np.ndarray] = {}
    for component in ("data", "zll", "other"):
        output[component] = np.sum(
            [np.asarray((node.get(component) or {}).get("sumw", [0.0] * size), dtype=float) for node in nodes],
            axis=0,
        )
        output[f"{component}2"] = np.sum(
            [np.asarray((node.get(component) or {}).get("sumw2", [0.0] * size), dtype=float) for node in nodes],
            axis=0,
        )
    return edges, output


def plot_mll_confirmation(
    measurement: dict[str, Any], regime: str, channel: str, output_dir: Path
) -> list[str]:
    source = measurement["mll_high" if regime == "highdm" else "mll_low"]
    edges, arrays = sum_histograms(source, channel)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = arrays["zll"] + arrays["other"]
    total_error = np.sqrt(np.maximum(arrays["zll2"] + arrays["other2"], 0.0))
    data_error = np.sqrt(np.maximum(arrays["data2"], 0.0))
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10.2, 10.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
    )
    ax.stairs(arrays["zll"], edges, baseline=0.0, fill=True, color=COLORS["dy"], edgecolor="black", linewidth=0.7, label="DY")
    ax.stairs(total, edges, baseline=arrays["zll"], fill=True, color=COLORS["other"], edgecolor="black", linewidth=0.7, label="Others")
    ax.stairs(total + total_error, edges, baseline=np.maximum(total - total_error, 1e-9), fill=True, facecolor="none", edgecolor="0.35", hatch="////", linewidth=0.0, label="Stat. unc.")
    ax.errorbar(centers, arrays["data"], yerr=data_error, fmt="o", color="black", ms=6, lw=2.0, capsize=2, label="Data")
    valid = total > 0.0
    ratio = np.full_like(total, np.nan)
    ratio_error = np.full_like(total, np.nan)
    ratio[valid] = arrays["data"][valid] / total[valid]
    ratio_error[valid] = data_error[valid] / total[valid]
    rax.errorbar(centers[valid], ratio[valid], yerr=ratio_error[valid], fmt="o", color="black", ms=6, lw=2.0, capsize=2)
    rax.axhline(1.0, color="black", lw=1.4)
    for axis in (ax, rax):
        axis.axvline(81.0, color="#D6278B", ls="--", lw=1.6)
        axis.axvline(101.0, color="#D6278B", ls="--", lw=1.6)
        style_axis(axis, edges)
    ax.set_yscale("log")
    positive = np.concatenate([arrays["data"][arrays["data"] > 0.0], total[total > 0.0]])
    if len(positive):
        ax.set_ylim(max(float(np.min(positive)) * 0.25, 0.05), float(np.max(positive)) * 8.0)
    rax.set_ylim(0.0, 2.0)
    ax.set_ylabel("Events / bin", fontsize=30)
    rax.set_ylabel("Data/MC", fontsize=25)
    rax.set_xlabel(r"$m_{\ell\ell}$ (GeV)", fontsize=30, loc="right")
    handles, labels = ax.get_legend_handles_labels()
    order = ["Stat. unc.", "Others", "DY", "Data"]
    ordered = [(handles[labels.index(label)], label) for label in order if label in labels]
    ax.legend([item[0] for item in ordered], [item[1] for item in ordered], frameon=False, fontsize=17, ncol=2, loc="upper right")
    ax.text(0.035, 0.055, f"{regime_label(regime)} {channel_label(channel)}\n$N_b \\geq 1$ inclusive", transform=ax.transAxes, fontsize=20, va="bottom")
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"{regime}_{channel.lower()}_mll_onoff_confirmation")


def plot_matrix_diagnostics(
    edges: np.ndarray,
    records: list[dict[str, Any]],
    regime: str,
    channel: str,
    output_dir: Path,
) -> list[str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    purity_on, purity_off, correlation = [], [], []
    for record in records:
        zon = leaf(record, "zll_on")[0]
        oon = leaf(record, "other_on")[0]
        zoff = leaf(record, "zll_off")[0]
        ooff = leaf(record, "other_off")[0]
        purity_on.append(zon / (zon + oon) if zon + oon > 0.0 else np.nan)
        purity_off.append(zoff / (zoff + ooff) if zoff + ooff > 0.0 else np.nan)
        correlation.append(float(record.get("correlation", np.nan)))
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10.2, 10.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.06},
    )
    ax.errorbar(centers, purity_on, xerr=widths, fmt="o", color="#D62728", lw=2.0, capsize=3, label="on-$Z$")
    ax.errorbar(centers, purity_off, xerr=widths, fmt="s", color="#1F77B4", lw=2.0, capsize=3, label="off-$Z$")
    ax.set_ylabel("DY fraction", fontsize=28)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=18, loc="best")
    rax.errorbar(centers, correlation, xerr=widths, fmt="o", color="black", lw=2.0, capsize=3)
    rax.axhline(0.0, color="0.45", ls="--", lw=1.4)
    rax.set_ylabel(r"corr$(R_Z,R_T)$", fontsize=25)
    rax.set_ylim(-1.05, 1.05)
    rax.set_xlabel(r"$U_T$ (GeV)", fontsize=30, loc="right")
    for axis in (ax, rax):
        style_axis(axis, edges)
        axis.grid(alpha=0.16)
    ax.text(0.035, 0.075, f"{regime_label(regime)} {channel_label(channel)}\n$N_b \\geq 1$ inclusive", transform=ax.transAxes, fontsize=20, va="bottom")
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"{regime}_{channel.lower()}_matrix_diagnostics_ut")


def html_page(output_dir: Path, regime: str, channel: str, summary: dict[str, Any]) -> None:
    images = [
        f"{regime}_{channel.lower()}_rz_ut.png",
        f"{regime}_{channel.lower()}_rt_ut.png",
        f"{regime}_{channel.lower()}_mll_onoff_confirmation.png",
        f"{regime}_{channel.lower()}_matrix_diagnostics_ut.png",
        f"{regime}_{channel.lower()}_dycr_ratio_movement_ut.png",
        f"{regime}_{channel.lower()}_onz_prefit_ut.png",
        f"{regime}_{channel.lower()}_onz_matrix_ut.png",
        f"{regime}_{channel.lower()}_offz_prefit_ut.png",
        f"{regime}_{channel.lower()}_offz_matrix_ut.png",
    ]
    rows = []
    def value_with_error(record: dict[str, Any], key: str) -> str:
        value = record.get(key)
        error = record.get(f"{key}_stat")
        if value is None or error is None:
            return "—"
        suffix = " (boundary)" if (record.get("boundary") or {}).get(key) else ""
        return f"{float(value):.3f} ± {float(error):.3f}{suffix}"

    def value_or_dash(record: dict[str, Any], key: str) -> str:
        value = record.get(key)
        return "—" if value is None else f"{float(value):.3f}"

    for record in summary["bins"]:
        rows.append(
            "<tr>"
            f"<td>{record['low']:.0f}–{record['high']:.0f}</td>"
            f"<td>{value_with_error(record, 'RZ')}</td>"
            f"<td>{value_with_error(record, 'RT')}</td>"
            f"<td>{value_or_dash(record, 'correlation')}</td>"
            "</tr>"
        )
    cards = "\n".join(
        f'<section><h2>{html.escape(path.replace("_", " ").replace(".png", ""))}</h2><a href="{path}"><img src="{path}" loading="lazy"></a></section>'
        for path in images
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024 UT-binned RZ measurement</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:28px auto;padding:0 18px;color:#202124}}h1,h2{{font-weight:600}}.notice{{padding:14px 18px;background:#fff4ce;border-left:5px solid #d79b00}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}}section{{border:1px solid #ddd;border-radius:10px;padding:14px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#f2f2f2;padding:2px 5px}}</style></head>
<body><h1>2024 {regime_label(regime)} {channel_label(channel)}: UT-binned R<sub>Z</sub></h1>
<p>Method: in every <i>U</i><sub>T</sub> bin, fit the on/off-Z 2×2 system simultaneously for non-negative <i>R</i><sub>Z</sub> and <i>R</i><sub>other</sub>. Data use Poisson likelihoods and weighted-MC statistics enter as Gaussian template constraints. on-Z is <code>81 &lt; mll &lt; 101 GeV</code>; off-Z is <code>50 &lt; mll &lt; 81 GeV or mll &gt; 101 GeV</code>.</p>
<p class="notice"><b>Interpretation:</b> the profile-fit reconstruction plots are fit diagnostics, not independent closure tests. Channel compatibility provides the first independent validation after both ee and μμ are complete.</p>
<table><thead><tr><th>U<sub>T</sub> (GeV)</th><th>R<sub>Z</sub></th><th>R<sub>non-DY</sub></th><th>corr(R<sub>Z</sub>,R<sub>non-DY</sub>)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid">{cards}</div></body></html>"""
    (output_dir / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--regime", choices=("highdm", "lowdm"), required=True)
    parser.add_argument("--channel", choices=("DY2E", "DY2M"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    measurement = read_json(args.measurement)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges, records = geometry(measurement, args.regime, args.channel)
    plot_factor(edges, records, "RZ", args.regime, args.channel, args.output_dir)
    plot_factor(edges, records, "RT", args.regime, args.channel, args.output_dir)
    plot_mll_confirmation(measurement, args.regime, args.channel, args.output_dir)
    plot_matrix_diagnostics(edges, records, args.regime, args.channel, args.output_dir)
    plot_movement(edges, records, args.regime, args.channel, args.output_dir)
    for window in ("on", "off"):
        for stage in ("prefit", "matrix"):
            plot_dycr(edges, records, window, stage, args.regime, args.channel, args.output_dir)
    summary = {
        "schema": "an_zinv_rz_ut_plot_summary_2024_v1",
        "regime": args.regime,
        "channel": args.channel,
        "method": "on/off-Z non-negative profile likelihood solved independently in each UT bin",
        "edges": edges.tolist(),
        "bins": records,
        "matrix_reconstruction_is_closure": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    html_page(args.output_dir, args.regime, args.channel, summary)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "plots": 9}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
