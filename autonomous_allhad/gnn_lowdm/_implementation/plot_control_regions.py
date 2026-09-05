#!/usr/bin/env python3
"""Plot inclusive-parent Low-dM CR GNN-output templates in CMS style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.4
import matplotlib.pyplot as plt


LUMINOSITY_FB = 109.82
CATEGORIES = ("Nb1_NISR0", "Nb1_NISR1plus", "Nb2plus_NISR0", "Nb2plus_NISR1plus")
CATEGORY_LABELS = {
    "Nb1_NISR0": r"$N_b=1$, $N_{\mathrm{ISR}}=0$",
    "Nb1_NISR1plus": r"$N_b=1$, $N_{\mathrm{ISR}}\geq1$",
    "Nb2plus_NISR0": r"$N_b\geq2$, $N_{\mathrm{ISR}}=0$",
    "Nb2plus_NISR1plus": r"$N_b\geq2$, $N_{\mathrm{ISR}}\geq1$",
}
REGION_COMPONENTS = {
    "LLCR": ("LLCR",),
    "QCDCR": ("QCDCR",),
    "GCR": ("GCR",),
    "DYCR": ("DY2E", "DY2M"),
}
REGION_LABELS = {
    "LLCR": r"Low-$\Delta m$ LLCR",
    "QCDCR": r"Low-$\Delta m$ QCDCR",
    "GCR": r"Low-$\Delta m$ GCR (unit area)",
    "DYCR": r"Low-$\Delta m$ DYCR ($ee+\mu\mu$, $R_Z$ applied)",
}
GROUP_ORDER = ("VV+VVV", "Top", "DY", "Photon+jet", "W -> lv", "Z -> vv", "QCD Multijet")
GROUP_COLORS = {
    "VV+VVV": "#6F7661", "Top": "#7A9FC2", "DY": "#35B6B4",
    "Photon+jet": "#8E3B9E", "W -> lv": "#D9C6A5",
    "Z -> vv": "#E6A84F", "QCD Multijet": "#C995A2",
}
GROUP_LABELS = {
    "VV+VVV": "VV+VVV", "Top": "Top", "DY": "DY", "Photon+jet": "Photon+jet",
    "W -> lv": r"$W\to\ell\nu$", "Z -> vv": r"$Z\to\nu\nu$", "QCD Multijet": "QCD Multijet",
}
RAW_GROUPS = {
    "VV+VVV": ("VV",), "Top": ("ST", "TT"), "DY": ("DY",),
    "Photon+jet": ("GJ",), "W -> lv": ("WtoLNu",),
    "Z -> vv": ("Zto2Nu",), "QCD Multijet": ("QCD",),
}
ALLOWED_SAMPLES = {"DY", "GJ", "QCD", "ST", "TT", "VV", "WtoLNu", "Zto2Nu", "data_obs"}


def score_edges(payload: dict, category: str) -> np.ndarray:
    value = payload["score_edges"]
    if isinstance(value, dict) and "CR" in value:
        value = value["CR"]
    if isinstance(value, dict):
        key = "Nb1" if category.startswith("Nb1_") else "Nb2plus"
        value = value[key]
    return np.asarray(value, dtype=float)


def component(payload: dict, region: str, category: str, sample: str) -> tuple[np.ndarray, np.ndarray]:
    variation = "nominal_rz" if region in {"DY2E", "DY2M"} and sample == "DY" else "nominal"
    record = (payload["histograms"].get(variation, {}).get(region, {}).get(category, {}).get(sample, {}).get("gnn_score"))
    size = len(score_edges(payload, category)) - 1
    if record is None:
        return np.zeros(size), np.zeros(size)
    return np.asarray(record["sumw"], dtype=float), np.asarray(record["sumw2"], dtype=float)


def combined_component(payload: dict, region: str, category: str, sample: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [component(payload, source, category, sample) for source in REGION_COMPONENTS[region]]
    return np.sum([part[0] for part in parts], axis=0), np.sum([part[1] for part in parts], axis=0)


def signed_stack(axis, values, edges, labels, colors) -> None:
    positive_bottom = np.zeros(len(edges) - 1)
    negative_bottom = np.zeros(len(edges) - 1)
    for current, label, color in zip(values, labels, colors):
        positive = np.clip(current, 0.0, None)
        negative = np.clip(current, None, 0.0)
        axis.stairs(positive_bottom + positive, edges, baseline=positive_bottom, fill=True,
                    color=color, edgecolor="black", linewidth=0.65, label=label)
        axis.stairs(negative_bottom + negative, edges, baseline=negative_bottom, fill=True,
                    color=color, edgecolor="black", linewidth=0.65)
        positive_bottom += positive
        negative_bottom += negative


def flattened(payload: dict, region: str) -> dict:
    source_regions = REGION_COMPONENTS[region]
    present = {
        sample
        for source in source_regions
        for category in CATEGORIES
        for sample in payload["histograms"].get("nominal", {}).get(source, {}).get(category, {})
    }
    unexpected = sorted(present - ALLOWED_SAMPLES)
    if unexpected:
        raise RuntimeError(f"unexpected/Other samples in {region}: {unexpected}")
    raw, raw2 = {}, {}
    for sample in sorted(ALLOWED_SAMPLES):
        parts = [combined_component(payload, region, category, sample) for category in CATEGORIES]
        raw[sample] = np.concatenate([part[0] for part in parts])
        raw2[sample] = np.concatenate([part[1] for part in parts])
    groups = {group: np.sum([raw[sample] for sample in samples], axis=0) for group, samples in RAW_GROUPS.items()}
    groups2 = {group: np.sum([raw2[sample] for sample in samples], axis=0) for group, samples in RAW_GROUPS.items()}
    background = np.sum([groups[group] for group in GROUP_ORDER], axis=0)
    background2 = np.sum([groups2[group] for group in GROUP_ORDER], axis=0)
    data, data2 = raw["data_obs"], raw2["data_obs"]
    raw_integrals = {"data": float(data.sum()), "background": float(background.sum())}
    if region == "GCR":
        data_norm, mc_norm = float(data.sum()), float(background.sum())
        if data_norm <= 0 or mc_norm <= 0:
            raise RuntimeError("cannot normalize empty GCR")
        data, data2 = data / data_norm, data2 / data_norm**2
        groups = {name: values / mc_norm for name, values in groups.items()}
        groups2 = {name: values / mc_norm**2 for name, values in groups2.items()}
        background, background2 = background / mc_norm, background2 / mc_norm**2
    return {"groups": groups, "groups2": groups2, "background": background,
            "background2": background2, "data": data, "data2": data2,
            "raw_integrals": raw_integrals, "samples": sorted(present)}


def interval_label(low: float, high: float, last: bool) -> str:
    return f"[{low:.3g}, {high:.3g}{']' if last else ')'}"


def draw(payload: dict, region: str, output: Path, luminosity_fb: float) -> dict:
    import mplhep as hep
    hep.style.use("CMS")
    content = flattened(payload, region)
    bins_per_category = 5
    total_bins = bins_per_category * len(CATEGORIES)
    edges = np.arange(0.5, total_bins + 1.5)
    centers = np.arange(1.0, total_bins + 1.0)
    fig, (axis, ratio_axis) = plt.subplots(2, 1, figsize=(18.4, 8.4),
        gridspec_kw={"height_ratios": [3.35, 1.05], "hspace": 0.04}, sharex=True)
    signed_stack(axis, [content["groups"][g] for g in GROUP_ORDER], edges,
                 [GROUP_LABELS[g] for g in GROUP_ORDER], [GROUP_COLORS[g] for g in GROUP_ORDER])
    background = content["background"]
    uncertainty = np.sqrt(np.clip(content["background2"], 0.0, None))
    lower, upper = np.maximum(background - uncertainty, 1e-12), np.maximum(background + uncertainty, 1e-12)
    axis.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post",
                      facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0,
                      alpha=0.65, label="MC stat. unc.", zorder=6)
    axis.stairs(background, edges, color="black", linewidth=1.4, zorder=7)
    data = content["data"]
    data_unc = np.sqrt(np.clip(content["data2"], 0.0, None))
    axis.errorbar(centers, data, xerr=0.5, yerr=data_unc, fmt="o", color="black",
                  markersize=5.6, capsize=3.2, capthick=1.25, elinewidth=1.25, label="DATA", zorder=10)
    ratio = np.divide(data, background, out=np.full_like(data, np.nan), where=background > 0)
    ratio_unc = np.divide(data_unc, background, out=np.full_like(data, np.nan), where=background > 0)
    rel_mc = np.divide(uncertainty, background, out=np.zeros_like(background), where=background > 0)
    ratio_axis.fill_between(edges, np.r_[1-rel_mc, 1-rel_mc[-1]], np.r_[1+rel_mc, 1+rel_mc[-1]],
                            step="post", facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0, alpha=0.65)
    ratio_axis.errorbar(centers, ratio, xerr=0.5, yerr=ratio_unc, fmt="o", color="black",
                        markersize=5.6, capsize=3.2, capthick=1.2, elinewidth=1.2)
    ratio_axis.axhline(1.0, color="black", linewidth=1.0)
    labels = []
    for category_index, category in enumerate(CATEGORIES):
        start, stop = category_index * bins_per_category, (category_index + 1) * bins_per_category
        if category_index:
            axis.axvline(start + 0.5, color="black", linewidth=1.2)
            ratio_axis.axvline(start + 0.5, color="black", linewidth=1.2)
        axis.text((start + stop + 1.0) / 2.0, 0.76, CATEGORY_LABELS[category],
                  transform=axis.get_xaxis_transform(), ha="center", va="center", fontsize=14.5,
                  bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.96}, zorder=20)
        local_edges = score_edges(payload, category)
        labels.extend(interval_label(lo, hi, i == bins_per_category - 1)
                      for i, (lo, hi) in enumerate(zip(local_edges[:-1], local_edges[1:])))
    positive = np.r_[data + data_unc, background + uncertainty]
    positive = positive[positive > 0]
    if region == "GCR":
        axis.set_ylim(1e-6, max(float(positive.max()) * 120.0, 1.0))
        axis.set_ylabel("Normalized events / bin", fontsize=27)
    else:
        ymax = 10.0 ** np.ceil(np.log10(max(float(positive.max()) * 80.0, 1.0)))
        axis.set_ylim(1e-2, ymax)
        axis.set_ylabel("Events / bin", fontsize=30)
    axis.set_yscale("log")
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.set_ylabel("DATA / MC", fontsize=24)
    ratio_axis.set_xticks(centers)
    ratio_axis.set_xticklabels(labels, rotation=90, fontsize=12)
    ratio_axis.set_xlabel("GNN output", fontsize=28)
    for current in (axis, ratio_axis):
        current.set_xlim(0.5, total_bins + 0.5)
        current.tick_params(which="major", direction="in", top=True, right=True, labelsize=17, length=8)
        current.tick_params(which="minor", direction="in", top=True, right=True, length=4)
        current.minorticks_on()
    hep.cms.label(
        "Work in progress", data=True, lumi=luminosity_fb, com=13.6,
        ax=axis, fontsize=22,
    )
    axis.text(0.015, 0.91, REGION_LABELS[region], transform=axis.transAxes, fontsize=20, va="top")
    handles, legend_labels = axis.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    desired = ["DATA", "MC stat. unc.", *[GROUP_LABELS[g] for g in reversed(GROUP_ORDER)]]
    axis.legend([by_label[x] for x in desired if x in by_label], [x for x in desired if x in by_label],
                loc="upper right", ncol=3, fontsize=14, frameon=False, columnspacing=1.2, handlelength=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"region": region, "normalization": "unit_area_data_and_total_mc" if region == "GCR" else "absolute_expected_yield",
            "raw_integrals": content["raw_integrals"], "samples": content["samples"],
            "png": str(output.with_suffix(".png").resolve()), "pdf": str(output.with_suffix(".pdf").resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--luminosity-fb", type=float)
    opts = parser.parse_args()
    payload = json.loads(opts.input.read_text())
    if payload.get("status") != "complete" or payload.get("bad_files"):
        raise RuntimeError("merged CR NN-out input is incomplete")
    if payload["input_files_valid"] != payload["input_files_requested"]:
        raise RuntimeError("merged CR NN-out input lost requested files")
    luminosity_fb = opts.luminosity_fb
    if luminosity_fb is None:
        luminosity_fb = {2024: LUMINOSITY_FB, 2025: 110.84}.get(
            int(payload.get("year", 2024)), LUMINOSITY_FB
        )
    records = [
        draw(
            payload, region,
            opts.output / f"lowdm_{region.lower()}_gnn_out_inclusive",
            luminosity_fb,
        )
        for region in REGION_COMPONENTS
    ]
    summary = {"schema_version": "lowdm_cr_nnout_inclusive_plot_v1", "status": "complete",
               "source": str(opts.input.resolve()), "input_files": payload["input_files_valid"],
               "luminosity_fb": luminosity_fb,
               "categories": list(CATEGORIES), "regions": list(REGION_COMPONENTS), "other_samples": [], "plots": records}
    (opts.output / "lowdm_cr_nnout_inclusive_plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
