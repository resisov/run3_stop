#!/usr/bin/env python3
"""Draw the CR fit-template bins plus boosted AN17 SR search bins."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


REGION_ORDER = [
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
]
SR_REGION = "cat7_SR_highDeltaM"
SEARCH_BIN_SCHEME = "boosted_an_17"

REGION_LABELS = {
    "cat2_LLCR_highDeltaM": "LLCR",
    "cat3_QCDCR_highDeltaM": "QCDCR",
    "cat4_GCR_highDeltaM": "GCR",
    "cat5_DY2E_highDeltaM": "DY2E",
    "cat6_DY2M_highDeltaM": "DY2M",
}

# Keep this process order fixed across control-region and search-bin plots.
GROUP_ORDER = ["VV", "Single Top", "TT", "DY", "Gamma + Jets", "W -> lv", "Z -> vv", "QCD Multijet", "others"]
GROUP_COLORS = {
    "VV": "#6f7661",
    "Single Top": "#8f7cc2",
    "TT": "#9ec5b8",
    "DY": "#23c9c8",
    "Gamma + Jets": "#800080",
    "W -> lv": "#eadac8",
    "Z -> vv": "#f2c58f",
    "QCD Multijet": "#d798a5",
    "others": "#6a625f",
}
SIGNAL_OVERLAYS = [
    {"key": "mStop1000_mLSP1", "label": '$m_{\\tilde{t}}=1000$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=1$ GeV', "color": "#ff0000"},
    {"key": "mStop1200_mLSP1", "label": '$m_{\\tilde{t}}=1200$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=1$ GeV', "color": "#00ff00"},
]
LOWDM_SIGNAL_OVERLAYS = [
    {"key": "mStop600_mLSP400", "label": '$m_{\\tilde{t}}=600$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=400$ GeV', "color": "#0066FF"},
    {"key": "mStop900_mLSP700", "label": '$m_{\\tilde{t}}=900$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=700$ GeV', "color": "#FF7A00"},
]
LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES = [
    ("Nb0_Nj2to5_PISR500plus", 4),
    ("Nb0_Nj6plus_PISR500plus", 4),
    ("Nb1_PISR300to500_PTb20to40", 4),
    ("Nb1_PISR300to500_PTb40to70", 4),
    ("Nb1_PISR500plus_PTb20to40", 4),
    ("Nb1_PISR500plus_PTb40to70", 4),
    ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", 3),
    ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", 3),
]
LOWDM_NSV_INCLUSIVE_CATEGORY_LABELS = {
    "Nb0_Nj2to5_PISR500plus": '$N_{b}=0$\n$2\\leq N_{j}\\leq5$',
    "Nb0_Nj6plus_PISR500plus": '$N_{b}=0$\n$N_{j}\\geq6$',
    "Nb1_PISR300to500_PTb20to40": '$N_{b}=1$\n$300\\leq p_{T}^{ISR}<500$\n$30<p_{T}^{b}<40$',
    "Nb1_PISR300to500_PTb40to70": '$N_{b}=1$\n$300\\leq p_{T}^{ISR}<500$\n$40<p_{T}^{b}<70$',
    "Nb1_PISR500plus_PTb20to40": '$N_{b}=1$\n$p_{T}^{ISR}\\geq500$\n$30<p_{T}^{b}<40$',
    "Nb1_PISR500plus_PTb40to70": '$N_{b}=1$\n$p_{T}^{ISR}\\geq500$\n$40<p_{T}^{b}<70$',
    "Nb2plus_PISR300to500_PTb40to80_Nj2plus": '$N_{b}\\geq2$\n$300\\leq p_{T}^{ISR}<500$\n$40<p_{T}^{b}<80$',
    "Nb2plus_PISR300to500_PTb80to140_Nj2plus": '$N_{b}\\geq2$\n$300\\leq p_{T}^{ISR}<500$\n$80<p_{T}^{b}<140$',
    "Nb2plus_PISR300to500_PTb140plus_Nj7plus": '$N_{b}\\geq2$, $N_{j}\\geq7$\n$300\\leq p_{T}^{ISR}<500$\n$p_{T}^{b}>140$',
    "Nb2plus_PISR500plus_PTb40to80_Nj2plus": '$N_{b}\\geq2$\n$p_{T}^{ISR}\\geq500$\n$40<p_{T}^{b}<80$',
    "Nb2plus_PISR500plus_PTb80to140_Nj2plus": '$N_{b}\\geq2$\n$p_{T}^{ISR}\\geq500$\n$80<p_{T}^{b}<140$',
    "Nb2plus_PISR500plus_PTb140plus_Nj7plus": '$N_{b}\\geq2$, $N_{j}\\geq7$\n$p_{T}^{ISR}\\geq500$\n$p_{T}^{b}>140$',
}
PARTIAL_AN17_SPLIT_BINS = [4, 5, 8, 9, 14, 15, 16]
SELECTED_AN17_RECOIL_SCHEME = "boosted_an17_selected_recoil6_SR"
LATEST_AN17_RECOIL_SCHEME = "boosted_an17_selected_recoil6_with_nt0_wsplit_SR"
EXTENDED_AN17_RECOIL_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"
RECOIL6_LABELS = ["250-300", "300-350", "350-400", "400-500", "500-800", "800-1500"]
LUMINOSITY_FB = 109.82
LUMINOSITY_RELATIVE_UNCERTAINTY = 0.016
PLOT_SYSTEMATIC_SOURCES = [
    "pileup",
    "electron_id",
    "electron_hlt",
    "muon_id",
    "muon_hlt",
    "photon_id",
    "btagSF_bc_correlated",
    "btagSF_bc_uncorrelated",
    "btagSF_light_correlated",
    "btagSF_light_uncorrelated",
    "jesTotal",
    "metUnclustered",
]
SELECTED_AN17_CATEGORY_LABELS = {
    'Nb1plus_T0_W0': '$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$',
    'Nb1plus_T0_W1plus': '$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}\\geq1$',
    'Nb1_T1plus_W0': '$N_{b}=1$, $N_{t}\\geq1$\n$N_{W}=0$',
    'Nb1_T1plus_W1plus': '$N_{b}=1$, $N_{t}\\geq1$\n$N_{W}\\geq1$',
    'Nb2_T1_W0': '$N_{b}=2$, $N_{t}=1$\n$N_{W}=0$',
    'Nb2_T1_W1': '$N_{b}=2$, $N_{t}=1$\n$N_{W}=1$',
    'Nb3plus_T1_W0': '$N_{b}\\geq3$, $N_{t}=1$\n$N_{W}=0$',
    'Nb3plus_T1_W1': '$N_{b}\\geq3$, $N_{t}=1$\n$N_{W}=1$',
    'Nb3plus_T2_W0': '$N_{b}\\geq3$, $N_{t}=2$\n$N_{W}=0$',
    'Nb2_Nt2plus_W0': '$N_{b}=2$, $N_{t}\\geq2$\n$N_{W}=0$',
}



def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def as_array(values: list[float] | None, nbin: int) -> np.ndarray:
    out = np.zeros(nbin, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float)
    out[: min(nbin, arr.size)] = arr[:nbin]
    return out


def process_to_group(process: str) -> str:
    if process == "VV":
        return "VV"
    if process == "ST":
        return "Single Top"
    if process == "TT":
        return "TT"
    if process == "DY":
        return "DY"
    if process == "GJ":
        return "Gamma + Jets"
    if process == "WtoLNu":
        return "W -> lv"
    if process == "Zto2Nu":
        return "Z -> vv"
    if process == "QCD":
        return "QCD Multijet"
    return "others"


def poisson_unc(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(data, 0.0))


def recoil_record_from_payload(payload: dict, region: str) -> tuple[dict, int] | None:
    raw_bkg = (((payload.get("histograms") or {}).get("background") or {}).get("recoil_pt") or {}).get(region) or {}
    raw_data = (((payload.get("histograms") or {}).get("data") or {}).get("recoil_pt") or {}).get(region) or {}
    ref = next(iter(raw_bkg.values()), None) or next(iter(raw_data.values()), None)
    if not ref:
        return None
    nbin = max(0, len(ref.get("bin_edges") or []) - 1)
    if nbin <= 0:
        return None
    groups = {group: {"values": [0.0] * nbin, "sumw2": [0.0] * nbin} for group in GROUP_ORDER}
    bkg_total = np.zeros(nbin, dtype=float)
    bkg_stat2 = np.zeros(nbin, dtype=float)
    for proc, hist in raw_bkg.items():
        group = process_to_group(proc)
        vals = as_array(hist.get("values"), nbin)
        s2 = as_array(hist.get("sumw2"), nbin)
        bkg_total += vals
        bkg_stat2 += s2
        groups[group]["values"] = (np.asarray(groups[group]["values"], dtype=float) + vals).tolist()
        groups[group]["sumw2"] = (np.asarray(groups[group]["sumw2"], dtype=float) + s2).tolist()
    data = np.zeros(nbin, dtype=float)
    data_s2 = np.zeros(nbin, dtype=float)
    for hist in raw_data.values():
        data += as_array(hist.get("values"), nbin)
        data_s2 += as_array(hist.get("sumw2"), nbin)
    syst2 = np.zeros(nbin, dtype=float)
    variations = ((((payload.get("histogram_systematic_variations") or {}).get("background") or {}).get("recoil_pt") or {}).get(region) or {})
    for var in variations.values():
        up = as_array(var.get("up_delta"), nbin)
        down = as_array(var.get("down_delta"), nbin)
        syst2 += np.maximum(np.abs(up), np.abs(down)) ** 2
    syst2 += (0.016 * bkg_total) ** 2
    rec = {
        "status": "complete",
        "variable": "recoil_pt",
        "region_short": REGION_LABELS.get(region, region),
        "plot_bin_edges": ref.get("bin_edges") or [],
        "physics_bin_edges": ref.get("bin_edges") or [],
        "background_total": bkg_total.tolist(),
        "background_stat_unc": np.sqrt(bkg_stat2).tolist(),
        "background_syst_unc": np.sqrt(syst2).tolist(),
        "background_total_unc": np.sqrt(bkg_stat2 + syst2).tolist(),
        "background_by_group": {k: v for k, v in groups.items() if any(abs(x) > 0 for x in v["values"])},
        "data": data.tolist(),
        "data_stat_unc": np.sqrt(data_s2).tolist(),
        "data_blinded_in_plots": False,
    }
    return rec, nbin


def flatten_cr_templates(fit: dict, payload: dict) -> dict:
    templates = fit.get("templates") or {}
    records = []
    boundaries = [0]
    labels = []
    for region in REGION_ORDER:
        if region in {"cat2_LLCR_highDeltaM", "cat3_QCDCR_highDeltaM"}:
            built = recoil_record_from_payload(payload, region)
            if not built:
                continue
            rec, nbin = built
        else:
            rec = templates.get(region) or {}
            values = rec.get("background_total") or []
            nbin = len(values)
            if rec.get("status") != "complete" or nbin == 0:
                continue
        records.append((region, rec, nbin))
        boundaries.append(boundaries[-1] + nbin)
        labels.append(REGION_LABELS.get(region, rec.get("region_short") or region))

    nbin_total = boundaries[-1]
    groups = {group: np.zeros(nbin_total, dtype=float) for group in GROUP_ORDER}
    bkg_total = np.zeros(nbin_total, dtype=float)
    bkg_unc = np.zeros(nbin_total, dtype=float)
    data = np.zeros(nbin_total, dtype=float)
    data_unc = np.zeros(nbin_total, dtype=float)
    offset = 0
    for _, rec, nbin in records:
        slc = slice(offset, offset + nbin)
        bkg_total[slc] = as_array(rec.get("background_total"), nbin)
        bkg_unc[slc] = as_array(rec.get("background_total_unc"), nbin)
        data[slc] = as_array(rec.get("data"), nbin)
        data_unc[slc] = as_array(rec.get("data_stat_unc"), nbin)
        for group in GROUP_ORDER:
            group_rec = (rec.get("background_by_group") or {}).get(group) or {}
            groups[group][slc] = as_array(group_rec.get("values"), nbin)
        offset += nbin

    return {
        "records": records,
        "boundaries": boundaries,
        "labels": labels,
        "groups": groups,
        "background": bkg_total,
        "background_unc": bkg_unc,
        "data": data,
        "data_unc": data_unc,
    }


def boosted_search_bins(payload: dict, signal_payload: dict) -> dict:
    bins = (payload.get("search_bins") or {}).get(SEARCH_BIN_SCHEME) or {}
    signal_bins = (signal_payload.get("yields") or {}).get(SEARCH_BIN_SCHEME) or {}
    names = list(bins)
    nbin = len(names)
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_unc2 = np.zeros(nbin, dtype=float)
    for idx, name in enumerate(names):
        for proc, rec in (bins.get(name) or {}).items():
            val = float(rec.get("normalized_weighted") or 0.0)
            s2 = float(rec.get("normalized_sumw2") or 0.0)
            if rec.get("kind") == "data":
                data[idx] += val
                data_unc2[idx] += s2
            elif rec.get("kind") == "background":
                group = process_to_group(proc)
                groups[group][idx] += val
                stat2[idx] += s2
    signals = {}
    for spec in SIGNAL_OVERLAYS:
        vals = np.zeros(nbin, dtype=float)
        for idx, name in enumerate(names):
            vals[idx] = float(((signal_bins.get(name) or {}).get(spec["key"]) or {}).get("normalized_weighted") or 0.0)
        if np.any(vals > 0):
            signals[spec["key"]] = vals
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    lumi_unc = 0.016 * bkg
    # The current preview stores nominal boosted search-bin yields, but not per-search-bin shape variations.
    # Use MC stat plus lumi here; fit-template CR bins retain the full stored stat+syst band.
    unc = np.sqrt(stat2 + lumi_unc * lumi_unc)
    return {
        "names": names,
        "groups": groups,
        "background": bkg,
        "background_unc": unc,
        "data": data,
        "data_unc": np.sqrt(data_unc2),
        "signals": signals,
        "uncertainty_note": "SR boosted_an_17 search-bin band uses MC stat + Lumi_2024 only; search-bin shape variations are not stored in this partial preview payload.",
    }


def concat(cr: dict, sr: dict) -> dict:
    n_cr = len(cr["background"])
    n_sr = len(sr["background"])
    groups = {group: np.r_[cr["groups"].get(group, np.zeros(n_cr)), sr["groups"].get(group, np.zeros(n_sr))] for group in GROUP_ORDER}
    data_mask = np.r_[np.ones(n_cr, dtype=bool), np.zeros(n_sr, dtype=bool)]
    signals = {}
    for spec in SIGNAL_OVERLAYS:
        key = spec["key"]
        sr_vals = sr["signals"].get(key)
        if sr_vals is not None:
            signals[key] = np.r_[np.zeros(n_cr, dtype=float), sr_vals]
    return {
        "groups": groups,
        "background": np.r_[cr["background"], sr["background"]],
        "background_unc": np.r_[cr["background_unc"], sr["background_unc"]],
        "data": np.r_[cr["data"], sr["data"]],
        "data_unc": np.r_[cr["data_unc"], sr["data_unc"]],
        "data_mask": data_mask,
        "signals": signals,
        "boundaries": cr["boundaries"] + [cr["boundaries"][-1] + n_sr],
        "labels": cr["labels"] + ["SR - BLIND"],
        "sr_search_bins": sr["names"],
        "sr_uncertainty_note": sr["uncertainty_note"],
    }


def draw(fit_path: Path, payload_path: Path, signal_searchbin_path: Path, outbase: Path) -> dict:
    reference_style = False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    fit = load_json(fit_path)
    payload = load_json(payload_path)
    signal_payload = load_json(signal_searchbin_path) if signal_searchbin_path.exists() else {}
    cr = flatten_cr_templates(fit, payload)
    sr = boosted_search_bins(payload, signal_payload)
    flat = concat(cr, sr)

    nbin = len(flat["background"])
    if nbin <= 0:
        raise RuntimeError("No complete bins found.")
    centers = np.arange(1, nbin + 1, dtype=float)
    edges = np.arange(0.5, nbin + 1.5, 1.0)

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(18, 8.7), gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04}, sharex=True)

    stack_inputs = []
    stack_weights = []
    stack_colors = []
    stack_labels = []
    for group in GROUP_ORDER:
        vals = flat["groups"].get(group)
        if vals is None or not np.any(vals > 0):
            continue
        stack_inputs.append(centers.copy())
        stack_weights.append(vals)
        stack_colors.append(GROUP_COLORS.get(group, "0.7"))
        stack_labels.append(group)
    if stack_inputs:
        ax.hist(stack_inputs, bins=edges, weights=stack_weights, stacked=True, histtype="stepfilled", color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.7)

    bkg = flat["background"]
    unc = flat["background_unc"]
    lower = np.maximum(bkg - unc, 1.0e-12)
    upper = np.maximum(bkg + unc, 1.0e-12)
    if np.any(bkg > 0):
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", facecolor="0.85", edgecolor="0.35", hatch="////", linewidth=0.0, alpha=0.35, label="MC stat+syst unc.")

    for spec in SIGNAL_OVERLAYS:
        vals = flat["signals"].get(spec["key"])
        if vals is None:
            continue
        ax.hist(centers, bins=edges, weights=vals, histtype="step", linewidth=2.0, color=spec["color"], label=spec["label"])

    data = flat["data"]
    data_unc = flat["data_unc"]
    mask = flat["data_mask"] & (data > 0)
    if np.any(mask):
        ax.errorbar(centers[mask], data[mask], yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])), fmt="o", color="black", markersize=4, label="Data" if reference_style else "DATA", zorder=10)

    ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    ratio_err = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    rmask = np.isfinite(ratio)
    rax.errorbar(centers[rmask], ratio[rmask], yerr=ratio_err[rmask], fmt="o", color="black", markersize=3)
    rel = np.divide(unc, bkg, out=np.full_like(unc, np.nan), where=bkg > 0)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    rax.fill_between(edges, np.r_[1.0 - rel, 1.0 - rel[-1]], np.r_[1.0 + rel, 1.0 + rel[-1]], step="post", facecolor="0.85", edgecolor="none", alpha=0.6)
    rax.axhline(1.0, color="0.45", linewidth=1)

    for axis in (ax, rax):
        for boundary in flat["boundaries"][1:-1]:
            axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
        for boundary in range(1, nbin):
            if boundary not in flat["boundaries"]:
                axis.axvline(boundary + 0.5, color="0.65", linestyle=":", linewidth=0.8, zorder=0)
        axis.set_xlim(0.5, nbin + 0.5)
        axis.tick_params(which="major", direction="in", top=True, right=True, labelsize=20, length=9)
        axis.tick_params(which="minor", direction="in", top=True, right=True, length=5)
        axis.minorticks_on()


    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *flat["signals"].values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    else:
        ax.set_ylim(0.03, 1.0)
    ax.set_ylabel("Events / bin", fontsize=30)
    rax.set_ylabel("Data/MC", fontsize=26)
    rax.set_ylim(0, 2)
    rax.set_xlabel("Control/search bin number", fontsize=30, loc="right")
    rax.set_xticks(centers)
    rax.set_xticklabels([str(i) for i in range(1, nbin + 1)], fontsize=13)
    hep.cms.label(llabel="Work in progress", rlabel=rf"{LUMINOSITY_FB:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=12, ncol=4, frameon=False, columnspacing=1.05, handlelength=2.0, loc="upper center", bbox_to_anchor=(0.5, 0.995))

    outbase.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if reference_style else {"bbox_inches": "tight"}
    fig.savefig(outbase.with_suffix(".png"), dpi=180, **save_kwargs)
    fig.savefig(outbase.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
    return {
        "status": "complete",
        "name": outbase.name,
        "png": str(outbase.with_suffix(".png")),
        "pdf": str(outbase.with_suffix(".pdf")),
        "bins": nbin,
        "control_bins": int(cr["boundaries"][-1]),
        "sr_search_bins": len(sr["names"]),
        "sr_search_bin_names": sr["names"],
        "signals": list(flat["signals"]),
        "sr_uncertainty_note": flat["sr_uncertainty_note"],
    }



FLAT_REGION_LABELS = {
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
    "GCR": "GCR",
    "DY2E": "DY2E",
    "DY2M": "DY2M",
    "HighDMVR_Nb1": "High-dM VR\n" + r"$N_{b}=1$",
    "HighDMVR_Nb2": "High-dM VR\n" + r"$N_{b}=2$",
    "HighDMVR_Nb3plus": "High-dM VR\n" + r"$N_{b}\geq3$",
    "SR": "SR",
    "LLCR_Nt0": r"LLCR\n$N_{t}=0$",
    "QCDCR_Nt0": r"QCDCR\n$N_{t}=0$",
    "GCR_Nt0": r"GCR\n$N_{t}=0$",
    "DY2E_Nt0": r"DY2E\n$N_{t}=0$",
    "DY2M_Nt0": r"DY2M\n$N_{t}=0$",
    "SR_Nt0": r"SR\n$N_{t}=0$",
    "LLCR_Nt1": r"LLCR\n$N_{t}\geq1$",
    "QCDCR_Nt1": r"QCDCR\n$N_{t}\geq1$",
    "GCR_Nt1": r"GCR\n$N_{t}\geq1$",
    "DY2E_Nt1": r"DY2E\n$N_{t}\geq1$",
    "DY2M_Nt1": r"DY2M\n$N_{t}\geq1$",
    "SR_Nt1": r"SR\n$N_{t}\geq1$",
}


def is_signal_sample(sample: str) -> bool:
    return sample.startswith("T2tt")


def flat_values(rec: dict, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    nominal = rec.get("nominal") or rec
    return as_array(nominal.get("sumw"), nbin), as_array(nominal.get("sumw2"), nbin)


def background_systematic_variance(raw: dict, nbin: int) -> np.ndarray:
    syst2 = np.zeros(nbin, dtype=float)
    for source in PLOT_SYSTEMATIC_SOURCES:
        up_total = np.zeros(nbin, dtype=float)
        down_total = np.zeros(nbin, dtype=float)
        have = False
        for sample, rec in raw.items():
            if sample == "data_obs" or is_signal_sample(sample):
                continue
            nominal, _ = flat_values(rec, nbin)
            up_rec = rec.get(source + "Up")
            down_rec = rec.get(source + "Down")
            if up_rec:
                up_total += as_array(up_rec.get("sumw"), nbin) - nominal
                have = True
            if down_rec:
                down_total += as_array(down_rec.get("sumw"), nbin) - nominal
                have = True
        if have:
            syst2 += np.maximum(np.abs(up_total), np.abs(down_total)) ** 2
    return syst2


def flat_hist_record(payload: dict, region: str, allow_signal: bool) -> dict | None:
    raw = (payload.get("histograms") or {}).get(region) or {}
    if not raw:
        return None
    nbin = 0
    for rec in raw.values():
        nominal = rec.get("nominal") or rec
        nbin = max(nbin, len(nominal.get("sumw") or []))
    if nbin <= 0:
        return None
    recoil_edges = payload.get("recoil_pt_bins") or []
    if len(recoil_edges) != nbin + 1:
        recoil_edges = []
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in SIGNAL_OVERLAYS:
                    flat_key = "T2tt_" + spec["key"].replace("mStop", "mStop").replace("_mLSP", "_mLSP")
                    if sample == flat_key:
                        signals[spec["key"]] += vals
        else:
            group = process_to_group(sample)
            groups[group] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    return {
        "groups": groups,
        "background": bkg,
        "background_unc": np.sqrt(stat2 + syst2),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "label": FLAT_REGION_LABELS.get(region, region),
        "nbin": nbin,
        "edges": recoil_edges,
    }


def flat_search_record(payload: dict, scheme: str, label: str, allow_signal: bool, signal_overlays: list[dict] | None = None) -> dict | None:
    raw = (payload.get("search_bin_histograms") or {}).get(scheme) or {}
    if not raw:
        return None
    nbin = 0
    for rec in raw.values():
        nominal = rec.get("nominal") or rec
        nbin = max(nbin, len(nominal.get("sumw") or []))
    if nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signal_overlays = signal_overlays or SIGNAL_OVERLAYS
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in signal_overlays}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in signal_overlays:
                    if sample == "T2tt_" + spec["key"]:
                        signals[spec["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    unc = np.sqrt(stat2 + syst2 + (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2)
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    return {"groups": groups, "background": bkg, "background_unc": unc, "data": data, "data_unc": np.sqrt(data2), "signals": signals, "label": label, "nbin": nbin}



VARIABLE_XLABELS = {
    "met": r"$\not\!E_{T}$ (GeV)",
    "recoil_gcr": r"$\not\!U_{T}$ (GeV)",
    "recoil_dy2e": r"$\not\!U_{T}$ (GeV)",
    "recoil_dy2m": r"$\not\!U_{T}$ (GeV)",
    "lowdm_met_sqrt_ht": r"$\not\!E_{T}/\sqrt{H_{T}}$",
}


def lowdm_variable_record(payload: dict, scheme: str, variable: str, label: str, allow_signal: bool) -> dict | None:
    raw = (((payload.get("lowdm_variable_histograms") or {}).get(scheme) or {}).get(variable) or {})
    if not raw:
        return None
    spec = ((payload.get("lowdm_variable_specs") or {}).get(variable) or {})
    edges = spec.get("bins") or []
    nbin = max(0, len(edges) - 1)
    if nbin <= 0:
        for rec in raw.values():
            nominal = rec.get("nominal") or rec
            nbin = max(nbin, len(nominal.get("sumw") or []))
        edges = []
    if nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {sig["key"]: np.zeros(nbin, dtype=float) for sig in SIGNAL_OVERLAYS}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for sig in SIGNAL_OVERLAYS:
                    if sample == "T2tt_" + sig["key"]:
                        signals[sig["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    return {
        "groups": groups,
        "background": bkg,
        "background_unc": np.sqrt(stat2 + syst2),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "label": label,
        "nbin": nbin,
        "edges": edges,
        "xlabel": VARIABLE_XLABELS.get(variable, spec.get("xlabel") or variable),
        "variable": variable,
    }



HIGHDM_DISTRIBUTION_REGION_LABELS = {
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
    "GCR": "GCR",
    "DY2E": "DY2E",
    "DY2M": "DY2M",
    "HighDMVR_Nb1": r"High-$\Delta m$ VR, $N_{b}=1$",
    "HighDMVR_Nb2": r"High-$\Delta m$ VR, $N_{b}=2$",
    "HighDMVR_Nb3plus": r"High-$\Delta m$ VR, $N_{b}\geq3$",
    "SR_Nb1plus_T0_W0": r"SR, $N_{b}\geq1$, $N_{top}=0$, $N_{W}=0$",
    "SR_Nb1plus_T0_W1plus": r"SR, $N_{b}\geq1$, $N_{top}=0$, $N_{W}\geq1$",
    "SR_Nb1_T1plus_W0": r"SR, $N_{b}=1$, $N_{top}\geq1$, $N_{W}=0$",
    "SR_Nb1_T1plus_W1plus": r"SR, $N_{b}=1$, $N_{top}\geq1$, $N_{W}\geq1$",
    "SR_Nb2_T1_W0": r"SR, $N_{b}=2$, $N_{top}=1$, $N_{W}=0$",
    "SR_Nb2_T1_W1": r"SR, $N_{b}=2$, $N_{top}=1$, $N_{W}=1$",
    "SR_Nb3plus_T1_W0": r"SR, $N_{b}\geq3$, $N_{top}=1$, $N_{W}=0$",
    "SR_Nb3plus_T1_W1": r"SR, $N_{b}\geq3$, $N_{top}=1$, $N_{W}=1$",
    "SR_Nb3plus_T2_W0": r"SR, $N_{b}\geq3$, $N_{top}=2$, $N_{W}=0$",
}

GROUP_DISPLAY_LABELS = {
    "VV": "VV",
    "Single Top": "Single Top",
    "TT": r"$\mathrm{t\bar{t}}$",
    "DY": "DY",
    "Gamma + Jets": "Gamma + Jets",
    "W -> lv": r"$\mathrm{W}\to\ell\nu$",
    "Z -> vv": r"$\mathrm{Z}\to\nu\bar{\nu}$",
    "QCD Multijet": "QCD Multijet",
    "others": "Others",
}


HIGHDM_MULTIPLICITY_PLOT_BINS = {
    "nb": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "groups": [[1], [2], [3, 4, 5]],
        "edges": [0.5, 1.5, 2.5, 3.5],
        "labels": ["1", "2", r"$\geq3$"],
    },
    "njet": {
        "source_edges": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 10.5, 12.5, 16.5],
        "groups": [[4], [5], [6], [7, 8, 9]],
        "edges": [4.5, 5.5, 6.5, 7.5, 8.5],
        "labels": ["5", "6", "7-8", r"$\geq9$"],
    },
    "nfatjet": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "groups": [[0], [1], [2], [3, 4, 5]],
        "edges": [-0.5, 0.5, 1.5, 2.5, 3.5],
        "labels": ["0", "1", "2", r"$\geq3$"],
    },
    "ntop": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5],
        "groups": [[0], [1], [2, 3, 4]],
        "edges": [-0.5, 0.5, 1.5, 2.5],
        "labels": ["0", "1", r"$\geq2$"],
    },
    "nw": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5],
        "groups": [[0], [1], [2, 3, 4]],
        "edges": [-0.5, 0.5, 1.5, 2.5],
        "labels": ["0", "1", r"$\geq2$"],
    },
}


def rebin_highdm_multiplicity(
    raw: dict, variable: str, source_edges: list[float]
) -> tuple[dict, list[float], list[str]]:
    config = HIGHDM_MULTIPLICITY_PLOT_BINS.get(variable)
    if config is None:
        return raw, source_edges, []
    expected_edges = np.asarray(config["source_edges"], dtype=float)
    if len(source_edges) != len(expected_edges) or not np.allclose(source_edges, expected_edges):
        raise ValueError(f"unexpected source bins for {variable}: {source_edges}")

    def rebin_leaf(leaf: dict) -> dict:
        result = dict(leaf)
        for key in ("entries", "sumw", "sumw2"):
            if key not in leaf:
                continue
            values = as_array(leaf.get(key), len(expected_edges) - 1)
            result[key] = [float(np.sum(values[group])) for group in config["groups"]]
        return result

    rebinned = {}
    for sample, record in raw.items():
        if "sumw" in record:
            rebinned[sample] = rebin_leaf(record)
            continue
        rebinned[sample] = {
            name: rebin_leaf(value) if isinstance(value, dict) and "sumw" in value else value
            for name, value in record.items()
        }
    return rebinned, list(config["edges"]), list(config["labels"])


def highdm_variable_record(payload: dict, region: str, variable: str) -> dict | None:
    raw = ((((payload.get("highdm_variable_histograms") or {}).get(region) or {}).get(variable)) or {})
    spec = ((payload.get("highdm_distribution_variable_specs") or {}).get(variable) or {})
    source_edges = spec.get("bins") or []
    raw, edges, xlabels = rebin_highdm_multiplicity(raw, variable, source_edges)
    nbin = max(0, len(edges) - 1)
    if not raw or nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif not is_signal_sample(sample):
            groups[process_to_group(sample)] += vals
            stat2 += s2
    background = sum(groups.values(), np.zeros(nbin, dtype=float))
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * background) ** 2
    if not np.any(background > 0) and not np.any(data > 0):
        return None
    visible = background if region.startswith("SR_") else background + data
    first_visible_bin = next((index for index, value in enumerate(visible) if value > 0), 0)
    xlim_left = float(edges[0] if xlabels else edges[first_visible_bin])
    return {
        "groups": groups,
        "background": background,
        "background_unc": np.sqrt(stat2 + syst2),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": {},
        "label": HIGHDM_DISTRIBUTION_REGION_LABELS.get(region, region),
        "nbin": nbin,
        "edges": edges,
        "xlabels": xlabels,
        "xlim_left": xlim_left,
        "xlabel": spec.get("xlabel") or variable,
        "variable": variable,
        "region": region,
        "blind_data": region.startswith("SR_"),
        "annotation": HIGHDM_DISTRIBUTION_REGION_LABELS.get(region, region),
        "reference_style": True,
    }


def partial_an17_search_record(payload: dict, label: str, split_bins: list[int], allow_signal: bool) -> dict | None:
    raw_inc = (payload.get("search_bin_histograms") or {}).get("boosted_an_17_SR") or {}
    raw_nt1 = (payload.get("search_bin_histograms") or {}).get("boosted_an_17_SR_Nt1") or {}
    if not raw_inc or not raw_nt1:
        return None
    labels = (((payload.get("search_bin_schemes") or {}).get("boosted_an_17_SR") or {}).get("bin_labels") or [])
    nbin_in = len(labels)
    if nbin_in <= 0:
        for rec in raw_inc.values():
            nominal = rec.get("nominal") or rec
            nbin_in = max(nbin_in, len(nominal.get("sumw") or []))
    if nbin_in <= 0:
        return None
    split = set(split_bins)
    nbin = nbin_in + sum(1 for idx in range(1, nbin_in + 1) if idx in split)

    def expanded(sample: str) -> tuple[np.ndarray, np.ndarray]:
        inc_vals, inc_s2 = flat_values(raw_inc.get(sample) or {}, nbin_in)
        nt1_vals, nt1_s2 = flat_values(raw_nt1.get(sample) or {}, nbin_in)
        nt0_vals = np.maximum(inc_vals - nt1_vals, 0.0)
        nt0_s2 = np.maximum(inc_s2 - nt1_s2, 0.0)
        vals = []
        s2 = []
        for idx in range(nbin_in):
            if idx + 1 in split:
                vals.extend([float(nt0_vals[idx]), float(nt1_vals[idx])])
                s2.extend([float(nt0_s2[idx]), float(nt1_s2[idx])])
            else:
                vals.append(float(inc_vals[idx]))
                s2.append(float(inc_s2[idx]))
        return np.asarray(vals, dtype=float), np.asarray(s2, dtype=float)

    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
    for sample in sorted(raw_inc):
        vals, s2 = expanded(sample)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in SIGNAL_OVERLAYS:
                    if sample == "T2tt_" + spec["key"]:
                        signals[spec["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    unc = np.sqrt(stat2 + (0.016 * bkg) ** 2)
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    xlabels = []
    for idx in range(1, nbin_in + 1):
        if idx in split:
            xlabels.extend([rf"{idx}\n$N_{{t}}=0$", rf"{idx}\n$N_{{t}}\geq1$"])
        else:
            xlabels.append(str(idx))
    return {
        "groups": groups,
        "background": bkg,
        "background_unc": unc,
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "label": label,
        "nbin": nbin,
        "split_bins_1based": split_bins,
        "xlabels": xlabels,
    }


def selected_an17_recoil_blocks(payload: dict, scheme_name: str) -> list[dict]:
    rec = flat_search_record(payload, scheme_name, "selected AN17 recoil", allow_signal=True)
    if not rec:
        return []
    scheme = (payload.get("search_bin_schemes") or {}).get(scheme_name) or {}
    raw_labels = scheme.get("bin_labels") or []
    blocks = []
    n_recoil = len(RECOIL6_LABELS)
    category_count = int(rec["nbin"]) // n_recoil
    for pos in range(category_count):
        slc = slice(pos * n_recoil, (pos + 1) * n_recoil)
        if slc.stop > int(rec["nbin"]):
            continue
        category = ""
        if raw_labels and pos * n_recoil < len(raw_labels):
            raw = raw_labels[pos * n_recoil]
            category = raw.split("_recoil_")[0]
            if category.startswith("NT0_"):
                category = category[len("NT0_"):]
            elif category.startswith("AN17_"):
                category = category.split("_", 2)[2]
        label = SELECTED_AN17_CATEGORY_LABELS.get(category, category) if category else f"category {pos + 1}"
        block = {
            "groups": {group: vals[slc] for group, vals in rec["groups"].items()},
            "background": rec["background"][slc],
            "background_unc": rec["background_unc"][slc],
            "data": rec["data"][slc],
            "data_unc": rec["data_unc"][slc],
            "signals": {key: vals[slc] for key, vals in rec.get("signals", {}).items()},
            "label": label,
            "nbin": n_recoil,
            "xlabels": [],
            "blind_data": True,
            "label_box": True,
        }
        blocks.append(block)
    return blocks


def lowdm_nsv_inclusive_blocks(payload: dict, scheme_name: str) -> list[dict]:
    rec = flat_search_record(
        payload,
        scheme_name,
        "Low-dM SR",
        allow_signal=True,
        signal_overlays=LOWDM_SIGNAL_OVERLAYS,
    )
    if not rec:
        return []
    scheme = (payload.get("search_bin_schemes") or {}).get(scheme_name) or {}
    category_sizes = scheme.get("category_sizes") or LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES
    normalized_sizes = [(str(category), int(size)) for category, size in category_sizes]
    if sum(size for _, size in normalized_sizes) != int(rec["nbin"]):
        return []
    blocks = []
    offset = 0
    for category, size in normalized_sizes:
        slc = slice(offset, offset + size)
        blocks.append({
            "groups": {group: vals[slc] for group, vals in rec["groups"].items()},
            "background": rec["background"][slc],
            "background_unc": rec["background_unc"][slc],
            "data": rec["data"][slc],
            "data_unc": rec["data_unc"][slc],
            "signals": {key: vals[slc] for key, vals in rec.get("signals", {}).items()},
            "signal_specs": LOWDM_SIGNAL_OVERLAYS,
            "label": LOWDM_NSV_INCLUSIVE_CATEGORY_LABELS.get(category, category),
            "nbin": size,
            "xlabels": [],
            "blind_data": True,
            "label_box": True,
            "label_fontsize": 8.2,
            "figure_width": 14.04,
        })
        offset += size
    return blocks


def draw_flat_blocks(blocks: list[dict], outbase: Path, xlabel: str = "Recoil/search bin number", reference_style: bool = False) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["hatch.linewidth"] = 1.4
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    reference_style = reference_style or any(bool(block.get("reference_style")) for block in blocks)
    nbin = sum(int(block["nbin"]) for block in blocks)
    physical_edges = None
    if len(blocks) == 1:
        candidate_edges = np.asarray(blocks[0].get("edges") or [], dtype=float)
        if len(candidate_edges) == int(blocks[0]["nbin"]) + 1 and np.all(np.diff(candidate_edges) > 0):
            physical_edges = candidate_edges
    if physical_edges is not None:
        edges = physical_edges
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        centers = np.arange(1, nbin + 1, dtype=float)
        edges = np.arange(0.5, nbin + 1.5, 1.0)
    xerr = 0.5 * np.diff(edges)
    boundaries = [0]
    labels = []
    for block in blocks:
        boundaries.append(boundaries[-1] + int(block["nbin"]))
        labels.append(block["label"])
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    bkg = np.zeros(nbin, dtype=float)
    unc = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_unc = np.zeros(nbin, dtype=float)
    data_mask = np.ones(nbin, dtype=bool)
    signal_specs = []
    seen_signal_keys = set()
    for block in blocks:
        for spec in block.get("signal_specs") or SIGNAL_OVERLAYS:
            if spec["key"] not in seen_signal_keys:
                signal_specs.append(spec)
                seen_signal_keys.add(spec["key"])
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in signal_specs}
    offset = 0
    for block in blocks:
        n = int(block["nbin"])
        slc = slice(offset, offset + n)
        for group in GROUP_ORDER:
            groups[group][slc] = block["groups"].get(group, np.zeros(n))
        bkg[slc] = block["background"]
        unc[slc] = block["background_unc"]
        data[slc] = block["data"]
        data_unc[slc] = block["data_unc"]
        if block.get("blind_data"):
            data_mask[slc] = False
        for key, vals in block.get("signals", {}).items():
            signals[key][slc] = vals
        offset += n
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}

    requested_width = max((float(block.get("figure_width", 0.0)) for block in blocks), default=0.0)
    figure_size = (12.0, 10.0) if reference_style else (max(12.0, nbin * 0.26, requested_width), 8.4)
    fig, (ax, rax) = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={"height_ratios": [3.2, 1.05 if reference_style else 1.1], "hspace": 0.04}, sharex=True)
    stack_inputs = []
    stack_weights = []
    stack_colors = []
    stack_labels = []
    for group in GROUP_ORDER:
        vals = groups[group]
        if np.any(vals > 0):
            stack_inputs.append(centers.copy())
            stack_weights.append(vals)
            stack_colors.append(GROUP_COLORS.get(group, "0.7"))
            stack_labels.append(GROUP_DISPLAY_LABELS.get(group, group) if reference_style else group)
    if stack_inputs:
        ax.hist(stack_inputs, bins=edges, weights=stack_weights, stacked=True, histtype="stepfilled", color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.7)
    lower = np.maximum(bkg - unc, 1.0e-12)
    upper = np.maximum(bkg + unc, 1.0e-12)
    if np.any(bkg > 0):
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0.0, alpha=0.65, label="Stat. syst. unc" if reference_style else "MC stat+syst unc.")
    if reference_style and np.any(bkg > 0):
        ax.stairs(bkg, edges, color="black", linewidth=1.4, zorder=6)
    for spec in signal_specs:
        vals = signals.get(spec["key"])
        if vals is not None:
            ax.hist(centers, bins=edges, weights=vals, histtype="step", linewidth=2.8, linestyle="--", color=spec["color"], label=spec["label"])
    mask = data_mask & (data > 0)
    ax.errorbar(
        centers[mask],
        data[mask],
        xerr=xerr[mask],
        yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])),
        fmt="o",
        color="black",
        markersize=8.0 if reference_style else 5.5,
        capsize=4.5,
        capthick=1.4,
        elinewidth=1.4,
        label="Data" if reference_style else "DATA",
        zorder=10,
    )
    ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
    ratio_err = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
    rmask = np.isfinite(ratio)
    rax.errorbar(
        centers[rmask],
        ratio[rmask],
        xerr=xerr[rmask],
        yerr=ratio_err[rmask],
        fmt="o",
        color="black",
        markersize=7.0 if reference_style else 4.5,
        capsize=4.5,
        capthick=1.4,
        elinewidth=1.4,
    )
    rel = np.divide(unc, bkg, out=np.zeros_like(unc), where=bkg > 0)
    rax.fill_between(edges, np.r_[1.0 - rel, 1.0 - rel[-1]], np.r_[1.0 + rel, 1.0 + rel[-1]], step="post", facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0.0, alpha=0.65)
    rax.axhline(1.0, color="0.45", linewidth=1)
    for axis in (ax, rax):
        axis.set_xmargin(0)
        if physical_edges is None:
            for boundary in boundaries[1:-1]:
                axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
            axis.set_xlim(0.5, nbin + 0.5)
        else:
            requested_left = blocks[0].get("xlim_left") if len(blocks) == 1 else None
            axis.set_xlim(float(edges[0] if requested_left is None else requested_left), float(edges[-1]))
        axis.tick_params(which="major", direction="in", top=True, right=True, labelsize=22 if reference_style else 20, length=9)
        axis.tick_params(which="minor", direction="in", top=True, right=True, length=5)
        axis.minorticks_on()
    for start, end, label, block in zip(boundaries[:-1], boundaries[1:], labels, blocks):
        center = 0.5 * (start + end) + 0.5 if physical_edges is None else 0.5 * (float(edges[0]) + float(edges[-1]))
        if block.get("label_box"):
            rax.text(
                center,
                0.5,
                label,
                transform=rax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=float(block.get("label_fontsize", 15)),
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.96},
                zorder=20,
            )
    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *signals.values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    if positive:
        if reference_style:
            ymax = 10.0 ** np.ceil(np.log10(max(max(positive) * 60.0, 1.0)))
            ax.set_ylim(1.0e-1, ymax)
        else:
            ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    else:
        ax.set_ylim(0.03, 1.0)
    ax.set_ylabel("Events" if reference_style else "Events / bin", fontsize=32 if reference_style else 30)
    rax.set_ylabel("Data/MC", fontsize=30 if reference_style else 26)
    rax.set_ylim(0, 2)
    rax.set_xlabel(xlabel, fontsize=32 if reference_style else 30, loc="right")
    annotations = [str(block.get("annotation") or "") for block in blocks if block.get("annotation")]
    if len(annotations) == 1 and not reference_style:
        ax.text(0.035, 0.72, annotations[0], transform=ax.transAxes, ha="left", va="top", fontsize=20)
    if physical_edges is not None and len(blocks) == 1:
        xlabels = blocks[0].get("xlabels") or []
        if len(xlabels) == nbin:
            rax.set_xticks(centers)
            rax.set_xticklabels(xlabels, fontsize=22 if reference_style else 16)
    elif physical_edges is None:
        xlabels = []
        for block in blocks:
            block_labels = block.get("xlabels") or []
            if len(block_labels) == int(block["nbin"]):
                xlabels.extend(block_labels)
            else:
                start = len(xlabels) + 1
                xlabels.extend(str(i) for i in range(start, start + int(block["nbin"])))
        rax.set_xticks(centers)
        label_fontsize = 12 if any("\n" in lab for lab in xlabels) else (13 if nbin > 24 else 16)
        rax.set_xticklabels(xlabels, fontsize=label_fontsize)
    hep.cms.label(llabel="Work in progress", rlabel=rf"{LUMINOSITY_FB:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=ax)
    if reference_style:
        handles, legend_labels = ax.get_legend_handles_labels()
        desired = ["Stat. syst. unc", "VV", "Single Top", r"$\mathrm{t\bar{t}}$", "DY", "Gamma + Jets", r"$\mathrm{W}\to\ell\nu$", r"$\mathrm{Z}\to\nu\bar{\nu}$", "QCD Multijet", "Others", "Data"]
        ordered = [(handles[legend_labels.index(label)], label) for label in desired if label in legend_labels]
        if ordered:
            ax.legend([item[0] for item in ordered], [item[1] for item in ordered], fontsize=15, ncol=3, frameon=False, columnspacing=1.2, handlelength=1.8, loc="upper center", bbox_to_anchor=(0.52, 0.995))
    else:
        ax.legend(fontsize=12, ncol=4, frameon=False, columnspacing=1.05, handlelength=2.0, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    outbase.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if reference_style else {"bbox_inches": "tight"}
    fig.savefig(outbase.with_suffix(".png"), dpi=180, **save_kwargs)
    fig.savefig(outbase.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
    return {"status": "complete", "name": outbase.name, "png": str(outbase.with_suffix(".png")), "pdf": str(outbase.with_suffix(".pdf")), "bins": nbin, "labels": labels, "signals": list(signals)}


def draw_highdm_distribution_report(payload_path: Path, output_dir: Path, year: str) -> dict:
    payload = load_json(payload_path)
    region_groups = payload.get("highdm_distribution_regions") or {}
    variable_specs = payload.get("highdm_distribution_variable_specs") or {}
    plots = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, metadata_key in [("CR", "control"), ("VR", "validation"), ("SR", "signal_categories")]:
        for region in region_groups.get(metadata_key) or []:
            for variable in variable_specs:
                record = highdm_variable_record(payload, region, variable)
                if not record:
                    continue
                slug = region.lower().replace("highdm", "highdm_").replace("__", "_")
                name = f"{kind.lower()}_{slug}_{variable}"
                plot = draw_flat_blocks(
                    [record],
                    output_dir / name,
                    xlabel=record["xlabel"],
                    reference_style=True,
                )
                plot.update({
                    "year": year,
                    "kind": kind,
                    "region": region,
                    "region_label": record["label"],
                    "variable": variable,
                    "xlabel": record["xlabel"],
                    "blind_data": bool(record.get("blind_data")),
                    "figure_size_inches": [12.0, 10.0],
                })
                plots.append(plot)
    summary = {
        "status": "complete",
        "year": year,
        "luminosity_fb": LUMINOSITY_FB,
        "luminosity_relative_uncertainty": LUMINOSITY_RELATIVE_UNCERTAINTY,
        "background_systematic_sources": list(PLOT_SYSTEMATIC_SOURCES),
        "uncertainty_model": "MC stat plus luminosity plus per-source max(abs(Up-Nominal), abs(Down-Nominal)) envelopes in quadrature",
        "source": str(payload_path),
        "output_dir": str(output_dir),
        "plot_count": len(plots),
        "plots": plots,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_highdm_distribution_webpage(
    summary_2024: Path,
    summary_2025: Path,
    docs_dir: Path,
    impact_png: Path | None = None,
    impact_pdf: Path | None = None,
    impact_json: Path | None = None,
) -> dict:
    import html

    summaries = [load_json(summary_2024), load_json(summary_2025)]
    docs_dir.mkdir(parents=True, exist_ok=True)
    variable_names = {
        "nb": "Nb", "njet": "Nj", "nfatjet": "Nfj", "ntop": "Ntop", "nw": "NW",
        "ht": "HT", "ut": "UT", "met": "pTmiss", "jet_pt": "Jet pT",
        "fatjet_pt": "FatJet pT", "bjet_pt": "b-jet pT",
    }
    options = ["<option value='all'>All variables</option>"]
    variables = []
    cards = []
    for summary in summaries:
        year = str(summary["year"])
        for plot in summary.get("plots") or []:
            variable = str(plot["variable"])
            if variable not in variables:
                variables.append(variable)
            name = str(plot["name"])
            kind = str(plot["kind"])
            region = str(plot["region"])
            title = f"{year} · {kind} · {region} · {variable_names.get(variable, variable)}"
            cards.append(
                f"<a class='plot' data-year='{html.escape(year)}' data-kind='{html.escape(kind)}' "
                f"data-variable='{html.escape(variable)}' href='plots/{html.escape(year)}/{html.escape(name)}.pdf'>"
                f"<img src='plots/{html.escape(year)}/{html.escape(name)}.png' loading='lazy' alt='{html.escape(title)}'>"
                f"<span>{html.escape(title)}</span></a>"
            )
    impact_record = None
    if impact_png or impact_pdf or impact_json:
        if not impact_png or not impact_pdf or not impact_json:
            raise ValueError("impact PNG, PDF and JSON must be supplied together")
        impact_dir = docs_dir / "impacts"
        impact_dir.mkdir(parents=True, exist_ok=True)
        for source in (impact_png, impact_pdf, impact_json):
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, impact_dir / source.name)
        title = "2024+2025 · Asimov r=1 impacts · mStop 1250 GeV, mLSP 500 GeV"
        cards.append(
            f"<a class='plot' data-year='Combined' data-kind='Impact' data-variable='impact' "
            f"href='impacts/{html.escape(impact_pdf.name)}'>"
            f"<img src='impacts/{html.escape(impact_png.name)}' loading='lazy' alt='{html.escape(title)}'>"
            f"<span>{html.escape(title)}</span></a>"
        )
        impact_record = {
            "status": "complete",
            "benchmark": "mStop1250_mLSP500",
            "asimov_expect_signal": 1,
            "png": f"impacts/{impact_png.name}",
            "pdf": f"impacts/{impact_pdf.name}",
            "json": f"impacts/{impact_json.name}",
        }
    for variable in variables:
        options.append(f"<option value='{html.escape(variable)}'>{html.escape(variable_names.get(variable, variable))}</option>")
    page = """<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Run-3 CR, SR and VR distributions</title>
<style>
:root{--ink:#171b1d;--muted:#5f686d;--line:#d6dcdf;--bg:#f3f5f6;--panel:#fff;--accent:#087f5b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:22px 24px}header div,main{max-width:1500px;margin:0 auto}
h1{font-size:26px;margin:0;letter-spacing:0}.toolbar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:11px 18px}
.toolbar-inner{max-width:1500px;margin:0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.segments{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
button,select{font:inherit;background:#fff;color:var(--ink);border:0;min-height:38px;padding:7px 12px}button{border-right:1px solid var(--line);cursor:pointer}button:last-child{border-right:0}
button.active{background:var(--ink);color:#fff}select{border:1px solid var(--line);border-radius:6px;min-width:150px}
main{padding:18px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:13px}.plot{display:block;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;color:var(--ink);text-decoration:none}
.plot[hidden]{display:none}.plot img{display:block;width:100%;height:auto}.plot span{display:block;padding:8px 10px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
@media(max-width:540px){header{padding:18px}h1{font-size:22px}.plots{grid-template-columns:1fr}.toolbar{padding:9px}.toolbar-inner{gap:7px}button{padding:6px 9px}}
</style></head><body>
<header><div><h1>Run-3 CR, SR and VR distributions</h1></div></header>
<div class='toolbar'><div class='toolbar-inner'>
<div class='segments' id='years'><button data-value='2024' class='active'>2024</button><button data-value='2025'>2025</button>""" + ("<button data-value='Combined'>Combined</button>" if impact_record else "") + """</div>
<div class='segments' id='kinds'><button data-value='all' class='active'>All</button><button data-value='CR'>CR</button><button data-value='SR'>SR</button><button data-value='VR'>VR</button>""" + ("<button data-value='Impact'>Impact</button>" if impact_record else "") + """</div>
<select id='variables'>""" + "".join(options) + """</select>
</div></div><main><div class='plots'>""" + "".join(cards) + """</div></main>
<script>
let year='2024',kind='all',variable='all';
function apply(){document.querySelectorAll('.plot').forEach(card=>{card.hidden=!(card.dataset.year===year&&(kind==='all'||card.dataset.kind===kind)&&(variable==='all'||card.dataset.variable===variable));});}
function bind(id,setter){document.querySelectorAll('#'+id+' button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.remove('active'));button.classList.add('active');setter(button.dataset.value);apply();}));}
bind('years',value=>year=value);bind('kinds',value=>kind=value);document.getElementById('variables').addEventListener('change',event=>{variable=event.target.value;apply();});apply();
</script></body></html>"""
    (docs_dir / "index.html").write_text(page)
    result = {
        "status": "complete",
        "page": str(docs_dir / "index.html"),
        "plot_count": len(cards),
        "years": [str(summary["year"]) for summary in summaries],
        "impact": impact_record,
    }
    (docs_dir / "page_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def draw_flat_report(flat_hists: Path, output_dir: Path) -> dict:
    payload = load_json(flat_hists)
    plots = []
    output_dir.mkdir(parents=True, exist_ok=True)
    cr_regions = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"]

    cr_inclusive = []
    for base in cr_regions:
        rec = flat_hist_record(payload, base, allow_signal=False)
        if rec:
            cr_inclusive.append(rec)
            plots.append(draw_flat_blocks([rec], output_dir / f"highdm_cr_{base.lower()}_recoil", xlabel=r"$\not\!U_{T}$ (GeV)"))
    if cr_inclusive:
        plots.append(draw_flat_blocks(cr_inclusive, output_dir / "highdm_cr_recoil_inclusive", xlabel=r"High-dM CR $\not\!U_{T}$ bin number"))

    cr_split = []
    for base in cr_regions:
        split_blocks = []
        for suffix in ["Nt0", "Nt1"]:
            rec = flat_hist_record(payload, f"{base}_{suffix}", allow_signal=False)
            if rec:
                split_blocks.append(rec)
                cr_split.append(rec)
        if split_blocks:
            plots.append(draw_flat_blocks(split_blocks, output_dir / f"highdm_cr_{base.lower()}_recoil_ntop_split", xlabel=fr"High-dM {base} $\not\!U_{{T}}$ bin number"))
    if cr_split:
        plots.append(draw_flat_blocks(cr_split, output_dir / "highdm_cr_recoil_ntop_split"))
    for region, slug in [
        ("HighDMVR_Nb1", "nb1"),
        ("HighDMVR_Nb2", "nb2"),
        ("HighDMVR_Nb3plus", "nb3plus"),
    ]:
        highdm_vr = flat_hist_record(payload, region, allow_signal=False)
        if highdm_vr:
            highdm_vr["annotation"] = highdm_vr["label"]
            plots.append(
                draw_flat_blocks(
                    [highdm_vr],
                    output_dir / f"highdm_vr_{slug}_met",
                    xlabel=r"$p_{T}^{miss}$ (GeV)",
                )
            )
    sr_inc = flat_hist_record(payload, "SR", allow_signal=True)
    if sr_inc:
        sr_inc["blind_data"] = True
        plots.append(draw_flat_blocks([sr_inc], output_dir / "highdm_sr_recoil_inclusive", xlabel=r"$\not\!U_{T}$ (GeV)"))
    sr_split = []
    for region in ["SR_Nt0", "SR_Nt1"]:
        rec = flat_hist_record(payload, region, allow_signal=True)
        if rec:
            rec["blind_data"] = True
            sr_split.append(rec)
    if sr_split:
        plots.append(draw_flat_blocks(sr_split, output_dir / "highdm_sr_recoil_ntop_split", xlabel=r"High-dM SR $\not\!U_{T}$ bin number"))
    an17 = flat_search_record(payload, "boosted_an_17_SR", "SR", allow_signal=True)
    if an17:
        an17["blind_data"] = True
        plots.append(draw_flat_blocks([an17], output_dir / "highdm_sr_an17_search_bins", xlabel="High-dM SR search bin number"))
    an17_nt1 = flat_search_record(payload, "boosted_an_17_SR_Nt1", r"SR\n$N_{t}\geq1$", allow_signal=True)
    if an17_nt1:
        an17_nt1["blind_data"] = True
        plots.append(draw_flat_blocks([an17_nt1], output_dir / "highdm_sr_nt1_an17_search_bins", xlabel="High-dM SR, $N_{t}\geq1$ search bin number"))
    available_schemes = payload.get("search_bin_schemes") or {}
    if EXTENDED_AN17_RECOIL_SCHEME in available_schemes:
        selected_scheme = EXTENDED_AN17_RECOIL_SCHEME
    elif LATEST_AN17_RECOIL_SCHEME in available_schemes:
        selected_scheme = LATEST_AN17_RECOIL_SCHEME
    else:
        selected_scheme = SELECTED_AN17_RECOIL_SCHEME
    selected_recoil_blocks = selected_an17_recoil_blocks(payload, selected_scheme)
    if selected_recoil_blocks:
        if selected_scheme == EXTENDED_AN17_RECOIL_SCHEME:
            selected_name = "highdm_sr_selected_recoil60_nb2_nt2plus_w0_bins"
        elif selected_scheme == LATEST_AN17_RECOIL_SCHEME:
            selected_name = "highdm_sr_selected_recoil54_nt0_wsplit_bins"
        else:
            selected_name = "highdm_sr_selected_recoil6_bins"
        plots.append(draw_flat_blocks(selected_recoil_blocks, output_dir / selected_name, xlabel="Search bin number"))
    low_cr_blocks = []
    low_blocks = []
    low_map = [
        ("cat2_LLCR_lowDeltaM", "LLCR low $\Delta m$", False, "LLCR"),
        ("cat3_QCDCR_lowDeltaM", "QCDCR low $\Delta m$", False, "QCDCR"),
        ("cat4_GCR_lowDeltaM", "GCR low $\Delta m$", False, "GCR"),
        ("cat5_DY2E_lowDeltaM", "DY2E low $\Delta m$", False, "DY2E"),
        ("cat6_DY2M_lowDeltaM", "DY2M low $\Delta m$", False, "DY2M"),
        ("cat7_SR_lowDeltaM", "SR low $\Delta m$", True, "SR"),
    ]
    for scheme, label, is_sr, base_region in low_map:
        rec = flat_search_record(payload, scheme, label, allow_signal=False)
        if rec:
            rec["blind_data"] = is_sr
            if not is_sr:
                short = scheme.replace("_lowDeltaM", "").split("_", 1)[1].lower()
                clean_label = label.replace(" low $\Delta m$", "")
                low_cr_blocks.append(rec)
                plots.append(draw_flat_blocks([rec], output_dir / f"lowdm_cr_{short}_onebin", xlabel=f"Low-dM {clean_label} bin number"))
            low_blocks.append(rec)
    lowdm_variable_plots = []
    lowdm_region_variables = payload.get("lowdm_region_variables") or {}
    for scheme, label, is_sr, base_region in low_map:
        available = ((payload.get("lowdm_variable_histograms") or {}).get(scheme) or {})
        variables = lowdm_region_variables.get(base_region) or sorted(available)
        short = scheme.replace("_lowDeltaM", "").split("_", 1)[1].lower()
        kind = "sr" if is_sr else "cr"
        for variable in variables:
            rec = lowdm_variable_record(payload, scheme, variable, label, allow_signal=is_sr)
            if not rec:
                continue
            rec["blind_data"] = is_sr
            outname = f"lowdm_{kind}_{short}_{variable}"
            plot = draw_flat_blocks([rec], output_dir / outname, xlabel=rec.get("xlabel", variable))
            plot["variable"] = variable
            plot["region"] = base_region
            lowdm_variable_plots.append(plot)
            plots.append(plot)
    if low_cr_blocks:
        plots.append(draw_flat_blocks(low_cr_blocks, output_dir / "lowdm_cr_onebin", xlabel="Low-dM CR region bin number"))
    if low_blocks:
        plots.append(draw_flat_blocks(low_blocks, output_dir / "lowdm_cr_sr_onebin", xlabel="Low-dM region bin number"))
    low_sr_blocks = lowdm_nsv_inclusive_blocks(payload, "cat7_SR_lowDeltaM")
    if low_sr_blocks:
        plots.append(draw_flat_blocks(low_sr_blocks, output_dir / "lowdm_sr_onebin", xlabel="Search bin number"))
    summary = {"status": "complete", "source": str(flat_hists), "output_dir": str(output_dir), "plots": plots, "lowdm_variable_plot_count": len([p for p in plots if str(p.get("name", "")).startswith("lowdm_") and p.get("variable")]), "signal_policy": "Signals are drawn only in SR plots; CR blocks exclude T2tt overlays.", "cr_plot_policy": "High-dM and low-dM CRs are drawn both as combined overview plots and as individual region plots.", "ntop_order": "N_t = 0 blocks are placed left of N_t >= 1 blocks.", "luminosity_fb": LUMINOSITY_FB, "luminosity_relative_uncertainty": LUMINOSITY_RELATIVE_UNCERTAINTY, "background_systematic_sources": PLOT_SYSTEMATIC_SOURCES}
    (output_dir / "flat_plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary

def add_to_index(docs_dir: Path, plot_name: str) -> None:
    index = docs_dir / "index.html"
    if not index.exists():
        return
    html = index.read_text()
    stem = f"plots/{plot_name}.png"
    if stem in html:
        return
    token = "</div>"
    card = f"<a class='plot' href='plots/{plot_name}.png'><img src='plots/{plot_name}.png' loading='lazy'><span>{plot_name}</span></a>"
    html = html.replace(token, card + token, 1)
    index.write_text(html)


def main() -> int:
    global LUMINOSITY_FB, LUMINOSITY_RELATIVE_UNCERTAINTY
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", required=False, type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--signal-searchbin-yields", default="docs/data/signal_searchbin_yields.json", type=Path)
    parser.add_argument("--name", default="partial_control_search_bins_style")
    parser.add_argument("--highdm-distributions", type=Path)
    parser.add_argument("--year", choices=["2024", "2025"])
    parser.add_argument("--summary-2024", type=Path)
    parser.add_argument("--summary-2025", type=Path)
    parser.add_argument("--web-dir", type=Path)
    parser.add_argument("--impact-png", type=Path)
    parser.add_argument("--impact-pdf", type=Path)
    parser.add_argument("--impact-json", type=Path)
    parser.add_argument("--flat-hists", type=Path)
    parser.add_argument("--flat-output-dir", type=Path)
    parser.add_argument("--luminosity-fb", type=float, default=LUMINOSITY_FB)
    parser.add_argument("--luminosity-relative-uncertainty", type=float, default=LUMINOSITY_RELATIVE_UNCERTAINTY)
    args = parser.parse_args()
    LUMINOSITY_FB = args.luminosity_fb
    LUMINOSITY_RELATIVE_UNCERTAINTY = args.luminosity_relative_uncertainty

    if args.summary_2024 and args.summary_2025:
        if not args.web_dir:
            parser.error("--web-dir is required with year summaries")
        print(json.dumps(write_highdm_distribution_webpage(
            args.summary_2024,
            args.summary_2025,
            args.web_dir,
            impact_png=args.impact_png,
            impact_pdf=args.impact_pdf,
            impact_json=args.impact_json,
        ), sort_keys=True))
        return 0

    if args.highdm_distributions:
        if not args.year:
            parser.error("--year is required with --highdm-distributions")
        outdir = args.flat_output_dir or Path(args.docs_dir or ".") / "plots" / args.year
        print(json.dumps(draw_highdm_distribution_report(args.highdm_distributions, outdir, args.year), sort_keys=True))
        return 0

    if args.flat_hists:
        outdir = args.flat_output_dir or Path(args.docs_dir or ".") / "plots"
        print(json.dumps(draw_flat_report(args.flat_hists, outdir), sort_keys=True))
        return 0

    if not args.preview_dir:
        parser.error("--preview-dir is required for the legacy preview plot")
    fit = args.preview_dir / "fit_template_summary.json"
    payload = args.preview_dir / "partial_normalized_yields.json"
    outbase = args.preview_dir / "plots" / args.name
    summary = draw(fit, payload, args.signal_searchbin_yields, outbase)
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
