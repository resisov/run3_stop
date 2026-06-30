#!/usr/bin/env python3
"""Draw a CMS-style control/search-bin summary from a partial preview."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


REGION_ORDER = [
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
    "cat7_SR_highDeltaM",
]

REGION_LABELS = {
    "cat2_LLCR_highDeltaM": "LLCR",
    "cat3_QCDCR_highDeltaM": "QCDCR",
    "cat4_GCR_highDeltaM": "GCR",
    "cat5_DY2E_highDeltaM": "DY2e CR",
    "cat6_DY2M_highDeltaM": "DY2mu CR",
    "cat7_SR_highDeltaM": "SR - BLIND",
}

GROUP_ORDER = [
    "QCD Multijet",
    "DY",
    "VV",
    "Single Top",
    "W -> lv",
    "Z -> vv",
    "Gamma + Jets",
    "ttbar",
]

GROUP_COLORS = {
    "QCD Multijet": "#6d625f",
    "DY": "#9b4fa8",
    "VV": "#7f7f7f",
    "Single Top": "#18d75f",
    "W -> lv": "#d9f000",
    "Z -> vv": "#2ac7c9",
    "Gamma + Jets": "#ffb000",
    "ttbar": "#1597e5",
}

SIGNAL_STYLES = {
    "mStop1000_mLSP1": ("#00d020", "--"),
    "mStop1200_mLSP1": ("#ff0000", "--"),
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def as_array(values: list[float] | None, nbin: int) -> np.ndarray:
    out = np.zeros(nbin, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float)
    out[: min(nbin, arr.size)] = arr[:nbin]
    return out


def poisson_unc(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(data, 0.0))


def flatten_templates(fit: dict) -> dict:
    templates = fit.get("templates") or {}
    records = []
    boundaries = [0]
    labels = []
    variables = []
    blinded = []
    for region in REGION_ORDER:
        rec = templates.get(region) or {}
        values = rec.get("background_total") or []
        nbin = len(values)
        if rec.get("status") != "complete" or nbin == 0:
            continue
        records.append((region, rec, nbin))
        boundaries.append(boundaries[-1] + nbin)
        labels.append(REGION_LABELS.get(region, rec.get("region_short") or region))
        variables.append(rec.get("variable") or "")
        blinded.append(bool(rec.get("data_blinded_in_plots")))

    nbin_total = boundaries[-1]
    groups = {group: np.zeros(nbin_total, dtype=float) for group in GROUP_ORDER}
    bkg_total = np.zeros(nbin_total, dtype=float)
    bkg_unc = np.zeros(nbin_total, dtype=float)
    data = np.zeros(nbin_total, dtype=float)
    data_unc = np.zeros(nbin_total, dtype=float)
    data_mask = np.zeros(nbin_total, dtype=bool)

    offset = 0
    for region, rec, nbin in records:
        slc = slice(offset, offset + nbin)
        bkg_total[slc] = as_array(rec.get("background_total"), nbin)
        bkg_unc[slc] = as_array(rec.get("background_total_unc"), nbin)
        data[slc] = as_array(rec.get("data"), nbin)
        data_unc[slc] = as_array(rec.get("data_stat_unc"), nbin)
        if not rec.get("data_blinded_in_plots"):
            data_mask[slc] = True
        for group in GROUP_ORDER:
            group_rec = (rec.get("background_by_group") or {}).get(group) or {}
            groups[group][slc] = as_array(group_rec.get("values"), nbin)
        offset += nbin

    return {
        "records": records,
        "boundaries": boundaries,
        "labels": labels,
        "variables": variables,
        "blinded": blinded,
        "groups": groups,
        "background": bkg_total,
        "background_unc": bkg_unc,
        "data": data,
        "data_unc": data_unc,
        "data_mask": data_mask,
    }


def flatten_signals(payload: dict, flattened: dict) -> dict[str, np.ndarray]:
    signals = {}
    hists = payload.get("signal_overlay_hists") or {}
    nbin_total = flattened["boundaries"][-1]
    for key in SIGNAL_STYLES:
        signals[key] = np.zeros(nbin_total, dtype=float)
    offset = 0
    for region, rec, nbin in flattened["records"]:
        variable = rec.get("variable")
        by_region = ((hists.get(variable) or {}).get(region) or {})
        for key in SIGNAL_STYLES:
            hist = by_region.get(key)
            if hist:
                signals[key][offset : offset + nbin] = as_array(hist.get("values"), nbin)
        offset += nbin
    return {key: vals for key, vals in signals.items() if np.any(vals > 0)}


def signal_label(key: str) -> str:
    if key == "mStop1000_mLSP1":
        return r"T2tt $m_{\tilde{t}}=1000$ GeV, $m_{\tilde{\chi}^{0}_{1}}=1$ GeV"
    if key == "mStop1200_mLSP1":
        return r"T2tt $m_{\tilde{t}}=1200$ GeV, $m_{\tilde{\chi}^{0}_{1}}=1$ GeV"
    return key


def draw(fit_path: Path, payload_path: Path, outbase: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import mplhep as hep

    hep.style.use("CMS")
    fit = load_json(fit_path)
    payload = load_json(payload_path) if payload_path.exists() else {}
    flat = flatten_templates(fit)
    signals = flatten_signals(payload, flat)

    nbin = flat["boundaries"][-1]
    if nbin <= 0:
        raise RuntimeError("No complete bins found in fit template summary.")

    centers = np.arange(1, nbin + 1, dtype=float)
    edges = np.arange(0.5, nbin + 1.5, 1.0)
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(18, 8.7),
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
        sharex=True,
    )

    bottom = np.zeros(nbin, dtype=float)
    bar_handles = []
    for group in GROUP_ORDER:
        vals = flat["groups"][group]
        if not np.any(vals > 0):
            continue
        handle = ax.bar(
            centers,
            vals,
            bottom=bottom,
            width=1.0,
            align="center",
            color=GROUP_COLORS.get(group, "0.7"),
            edgecolor="black",
            linewidth=0.45,
            label=group,
        )
        bar_handles.append(handle)
        bottom += vals

    bkg = flat["background"]
    unc = flat["background_unc"]
    valid = bkg > 0
    lower = np.maximum(bkg - unc, 1.0e-8)
    upper = np.maximum(bkg + unc, 1.0e-8)
    ax.fill_between(
        edges,
        np.r_[lower, lower[-1]],
        np.r_[upper, upper[-1]],
        step="post",
        facecolor="0.85",
        edgecolor="0.45",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
        label="MC stat. + syst.",
    )

    for key, vals in signals.items():
        color, ls = SIGNAL_STYLES[key]
        ax.hist(
            centers,
            bins=edges,
            weights=vals,
            histtype="step",
            linewidth=2.0,
            linestyle=ls,
            color=color,
            label=signal_label(key),
        )

    data = flat["data"]
    data_unc = flat["data_unc"]
    mask = flat["data_mask"] & (data > 0)
    ax.errorbar(
        centers[mask],
        data[mask],
        yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])),
        fmt="o",
        color="black",
        markersize=4.0,
        linewidth=1.0,
        label="Data 2024",
        zorder=10,
    )

    ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    ratio_unc = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    rmask = np.isfinite(ratio)
    rax.errorbar(centers[rmask], ratio[rmask], yerr=ratio_unc[rmask], fmt="o", color="black", markersize=3.5)
    rel_unc = np.divide(unc, bkg, out=np.full_like(unc, np.nan), where=valid)
    rel_unc = np.nan_to_num(rel_unc, nan=0.0, posinf=0.0, neginf=0.0)
    rax.fill_between(
        edges,
        np.r_[1.0 - rel_unc, 1.0 - rel_unc[-1]],
        np.r_[1.0 + rel_unc, 1.0 + rel_unc[-1]],
        step="post",
        facecolor="0.80",
        edgecolor="none",
        alpha=0.7,
    )
    rax.axhline(1.0, color="0.45", linewidth=1.0)

    for axis in (ax, rax):
        for boundary in flat["boundaries"][1:-1]:
            axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
        axis.set_xlim(0.5, nbin + 0.5)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.minorticks_on()

    for boundary in range(1, nbin):
        if boundary not in flat["boundaries"]:
            ax.axvline(boundary + 0.5, color="0.72", linestyle=":", linewidth=0.7, zorder=0)
            rax.axvline(boundary + 0.5, color="0.72", linestyle=":", linewidth=0.7, zorder=0)

    for start, end, label, variable in zip(flat["boundaries"][:-1], flat["boundaries"][1:], flat["labels"], flat["variables"]):
        center = 0.5 * (start + end) + 0.5
        ax.text(center, 0.96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=18, fontweight="bold")
        ax.text(center, 0.905, variable, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, color="0.25")

    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *(signals.values())]:
        positive.extend(np.asarray(arr)[np.asarray(arr) > 0].tolist())
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(0.05, min(positive) * 0.08), max(max(positive) * 70.0, 10.0))
    ax.set_ylabel("Events / bin")
    rax.set_ylabel("Data/MC")
    rax.set_ylim(0.0, 2.0)
    rax.set_xlabel("Control/search bin number")
    rax.set_xticks(centers)
    rax.set_xticklabels([str(i) for i in range(1, nbin + 1)], fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    custom = [Patch(facecolor="0.85", edgecolor="0.45", hatch="////", label="MC stat. + syst.")]
    legend_items = []
    seen = set()
    for handle, label in list(zip(handles, labels)) + [(custom[0], "MC stat. + syst.")]:
        if label in seen:
            continue
        seen.add(label)
        legend_items.append((handle, label))
    ax.legend(
        [h for h, _ in legend_items],
        [l for _, l in legend_items],
        fontsize=11,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.2,
        handlelength=1.7,
    )
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)

    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "complete",
        "name": outbase.name,
        "png": str(outbase.with_suffix(".png")),
        "pdf": str(outbase.with_suffix(".pdf")),
        "bins": nbin,
        "regions": [r for r, _, _ in flat["records"]],
        "signals": list(signals),
    }


def add_to_index(docs_dir: Path, plot_name: str) -> None:
    index = docs_dir / "index.html"
    if not index.exists():
        return
    html = index.read_text()
    stem = f"plots/{plot_name}.png"
    if stem in html:
        return
    token = "</div>"
    card = (
        f"<a class='plot' href='plots/{plot_name}.png'>"
        f"<img src='plots/{plot_name}.png' loading='lazy'>"
        f"<span>{plot_name}</span></a>"
    )
    html = html.replace(token, card + token, 1)
    index.write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--name", default="partial_control_search_bins_style")
    args = parser.parse_args()

    fit = args.preview_dir / "fit_template_summary.json"
    payload = args.preview_dir / "partial_normalized_yields.json"
    outbase = args.preview_dir / "plots" / args.name
    summary = draw(fit, payload, outbase)
    if args.docs_dir:
        plot_dst = args.docs_dir / "plots"
        plot_dst.mkdir(parents=True, exist_ok=True)
        for suffix in [".png", ".pdf"]:
            shutil.copy2(outbase.with_suffix(suffix), plot_dst / outbase.with_suffix(suffix).name)
        add_to_index(args.docs_dir, args.name)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
