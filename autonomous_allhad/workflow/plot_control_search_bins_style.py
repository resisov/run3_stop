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

# This is intentionally identical to build_partial_merge_preview.py.
GROUP_ORDER = ["VV", "Single Top", "ttbar", "DY", "Gamma + Jets", "W -> lv", "Z -> vv", "QCD Multijet", "others"]
GROUP_COLORS = {
    "VV": "#6f7661",
    "Single Top": "#8f7cc2",
    "ttbar": "#9ec5b8",
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
PARTIAL_AN17_SPLIT_BINS = [4, 5, 8, 9, 14, 15, 16]
SELECTED_AN17_RECOIL_SCHEME = "boosted_an17_selected_recoil6_SR"
SELECTED_AN17_RECOIL_BINS = [4, 5, 8, 9, 14, 15, 16]
RECOIL6_LABELS = ["250-300", "300-350", "350-400", "400-500", "500-800", "800-1500"]
SELECTED_AN17_CATEGORY_LABELS = {
    'Nb1_T1plus_W0': '$N_{b}=1$, $N_{t}\\geq1$\n$N_{W}=0$',
    'Nb1_T1plus_W1plus': '$N_{b}=1$, $N_{t}\\geq1$\n$N_{W}\\geq1$',
    'Nb2_T1_W0': '$N_{b}=2$, $N_{t}=1$\n$N_{W}=0$',
    'Nb2_T1_W1': '$N_{b}=2$, $N_{t}=1$\n$N_{W}=1$',
    'Nb3plus_T1_W0': '$N_{b}\\geq3$, $N_{t}=1$\n$N_{W}=0$',
    'Nb3plus_T1_W1': '$N_{b}\\geq3$, $N_{t}=1$\n$N_{W}=1$',
    'Nb3plus_T2_W0': '$N_{b}\\geq3$, $N_{t}=2$\n$N_{W}=0$',
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
        return "ttbar"
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
    ax.errorbar(centers[mask], data[mask], yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])), fmt="o", color="black", markersize=4, label="Data 2024", zorder=10)

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
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.minorticks_on()

    for start, end, label in zip(flat["boundaries"][:-1], flat["boundaries"][1:], flat["labels"]):
        center = 0.5 * (start + end) + 0.5
        ax.text(center, 0.965, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=17, fontweight="bold")

    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *flat["signals"].values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    ax.set_ylabel("Events / bin")
    rax.set_ylabel("Data/MC")
    rax.set_ylim(0, 2)
    rax.set_xlabel("Control/search bin number")
    rax.set_xticks(centers)
    rax.set_xticklabels([str(i) for i in range(1, nbin + 1)], fontsize=8)
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=10, ncol=4, frameon=False, columnspacing=1.0, handlelength=1.6, loc="upper center", bbox_to_anchor=(0.5, 0.995))

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
    syst2 = np.zeros(nbin, dtype=float)
    for source in ["pileup", "electron_id", "electron_hlt", "muon_id", "muon_hlt", "photon_id"]:
        up_total = np.zeros(nbin, dtype=float)
        down_total = np.zeros(nbin, dtype=float)
        have = False
        for sample, rec in raw.items():
            if sample == "data_obs" or is_signal_sample(sample):
                continue
            nom, _ = flat_values(rec, nbin)
            up_rec = rec.get(source + "Up")
            down_rec = rec.get(source + "Down")
            if up_rec:
                up_total += as_array(up_rec.get("sumw"), nbin) - nom
                have = True
            if down_rec:
                down_total += as_array(down_rec.get("sumw"), nbin) - nom
                have = True
        if have:
            syst2 += np.maximum(np.abs(up_total), np.abs(down_total)) ** 2
    syst2 += (0.016 * bkg) ** 2
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
    }


def flat_search_record(payload: dict, scheme: str, label: str, allow_signal: bool) -> dict | None:
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
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
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
    return {"groups": groups, "background": bkg, "background_unc": unc, "data": data, "data_unc": np.sqrt(data2), "signals": signals, "label": label, "nbin": nbin}




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


def selected_an17_recoil_blocks(payload: dict) -> list[dict]:
    rec = flat_search_record(payload, SELECTED_AN17_RECOIL_SCHEME, "selected AN17 recoil", allow_signal=True)
    if not rec:
        return []
    scheme = (payload.get("search_bin_schemes") or {}).get(SELECTED_AN17_RECOIL_SCHEME) or {}
    raw_labels = scheme.get("bin_labels") or []
    blocks = []
    n_recoil = len(RECOIL6_LABELS)
    for pos, bin_number in enumerate(SELECTED_AN17_RECOIL_BINS):
        slc = slice(pos * n_recoil, (pos + 1) * n_recoil)
        if slc.stop > int(rec["nbin"]):
            continue
        category = ""
        if raw_labels and pos * n_recoil < len(raw_labels):
            raw = raw_labels[pos * n_recoil]
            marker = f"AN17_{bin_number}_"
            if raw.startswith(marker):
                category = raw[len(marker):].split("_recoil_")[0]
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

def draw_flat_blocks(blocks: list[dict], outbase: Path, xlabel: str = "Recoil/search bin number") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    nbin = sum(int(block["nbin"]) for block in blocks)
    centers = np.arange(1, nbin + 1, dtype=float)
    edges = np.arange(0.5, nbin + 1.5, 1.0)
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
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
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

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(max(12.0, nbin * 0.26), 8.4), gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04}, sharex=True)
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
            stack_labels.append(group)
    if stack_inputs:
        ax.hist(stack_inputs, bins=edges, weights=stack_weights, stacked=True, histtype="stepfilled", color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.7)
    lower = np.maximum(bkg - unc, 1.0e-12)
    upper = np.maximum(bkg + unc, 1.0e-12)
    if np.any(bkg > 0):
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", facecolor="0.85", edgecolor="0.35", hatch="////", linewidth=0.0, alpha=0.35, label="MC stat+syst unc.")
    for spec in SIGNAL_OVERLAYS:
        vals = signals.get(spec["key"])
        if vals is not None:
            ax.hist(centers, bins=edges, weights=vals, histtype="step", linewidth=2.8, linestyle="--", color=spec["color"], label=spec["label"])
    mask = data_mask & (data > 0)
    ax.errorbar(centers[mask], data[mask], yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])), fmt="o", color="black", markersize=4, label="Data 2024", zorder=10)
    ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
    ratio_err = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
    rmask = np.isfinite(ratio)
    rax.errorbar(centers[rmask], ratio[rmask], yerr=ratio_err[rmask], fmt="o", color="black", markersize=3)
    rel = np.divide(unc, bkg, out=np.zeros_like(unc), where=bkg > 0)
    rax.fill_between(edges, np.r_[1.0 - rel, 1.0 - rel[-1]], np.r_[1.0 + rel, 1.0 + rel[-1]], step="post", facecolor="0.85", edgecolor="none", alpha=0.6)
    rax.axhline(1.0, color="0.45", linewidth=1)
    for axis in (ax, rax):
        for boundary in boundaries[1:-1]:
            axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
        axis.set_xlim(0.5, nbin + 0.5)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.minorticks_on()
    for start, end, label, block in zip(boundaries[:-1], boundaries[1:], labels, blocks):
        center = 0.5 * (start + end) + 0.5
        if block.get("label_box"):
            ax.text(
                center,
                0.15,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=13,
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.96},
            )
        else:
            ax.text(center, 0.965, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=15, fontweight="bold")
    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *signals.values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    ax.set_ylabel("Events / bin")
    rax.set_ylabel("Data/MC")
    rax.set_ylim(0, 2)
    rax.set_xlabel(xlabel)
    xlabels = []
    for block in blocks:
        block_labels = block.get("xlabels") or []
        if len(block_labels) == int(block["nbin"]):
            xlabels.extend(block_labels)
        else:
            start = len(xlabels) + 1
            xlabels.extend(str(i) for i in range(start, start + int(block["nbin"])))
    rax.set_xticks(centers)
    rax.set_xticklabels(xlabels, fontsize=7 if any("\n" in lab for lab in xlabels) else (8 if nbin > 24 else 10))
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=10, ncol=4, frameon=False, columnspacing=1.0, handlelength=1.6, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"status": "complete", "name": outbase.name, "png": str(outbase.with_suffix(".png")), "pdf": str(outbase.with_suffix(".pdf")), "bins": nbin, "labels": labels, "signals": list(signals)}


def draw_flat_report(flat_hists: Path, output_dir: Path) -> dict:
    payload = load_json(flat_hists)
    plots = []
    output_dir.mkdir(parents=True, exist_ok=True)
    cr_regions = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"]
    cr_split = []
    for base in cr_regions:
        for suffix in ["Nt0", "Nt1"]:
            rec = flat_hist_record(payload, f"{base}_{suffix}", allow_signal=False)
            if rec:
                cr_split.append(rec)
    if cr_split:
        plots.append(draw_flat_blocks(cr_split, output_dir / "highdm_cr_recoil_ntop_split"))
    sr_inc = flat_hist_record(payload, "SR", allow_signal=True)
    if sr_inc:
        sr_inc["blind_data"] = True
        plots.append(draw_flat_blocks([sr_inc], output_dir / "highdm_sr_recoil_inclusive"))
    sr_split = []
    for region in ["SR_Nt0", "SR_Nt1"]:
        rec = flat_hist_record(payload, region, allow_signal=True)
        if rec:
            rec["blind_data"] = True
            sr_split.append(rec)
    if sr_split:
        plots.append(draw_flat_blocks(sr_split, output_dir / "highdm_sr_recoil_ntop_split"))
    an17 = flat_search_record(payload, "boosted_an_17_SR", "SR", allow_signal=True)
    if an17:
        an17["blind_data"] = True
        plots.append(draw_flat_blocks([an17], output_dir / "highdm_sr_an17_search_bins", xlabel="High-dM SR search bin number"))
    an17_nt1 = flat_search_record(payload, "boosted_an_17_SR_Nt1", r"SR\n$N_{t}\geq1$", allow_signal=True)
    if an17_nt1:
        an17_nt1["blind_data"] = True
        plots.append(draw_flat_blocks([an17_nt1], output_dir / "highdm_sr_nt1_an17_search_bins", xlabel="High-dM SR, $N_{t}\geq1$ search bin number"))
    selected_recoil_blocks = selected_an17_recoil_blocks(payload)
    if selected_recoil_blocks:
        plots.append(draw_flat_blocks(selected_recoil_blocks, output_dir / "highdm_sr_selected_recoil6_bins", xlabel="Search bin number"))
    low_blocks = []
    low_map = [
        ("cat2_LLCR_lowDeltaM", "LLCR low $\Delta m$", False),
        ("cat3_QCDCR_lowDeltaM", "QCDCR low $\Delta m$", False),
        ("cat4_GCR_lowDeltaM", "GCR low $\Delta m$", False),
        ("cat5_DY2E_lowDeltaM", "DY2E low $\Delta m$", False),
        ("cat6_DY2M_lowDeltaM", "DY2M low $\Delta m$", False),
        ("cat7_SR_lowDeltaM", "SR low $\Delta m$", True),
    ]
    for scheme, label, is_sr in low_map:
        rec = flat_search_record(payload, scheme, label, allow_signal=is_sr)
        if rec:
            rec["blind_data"] = is_sr
            low_blocks.append(rec)
    if low_blocks:
        plots.append(draw_flat_blocks(low_blocks, output_dir / "lowdm_cr_sr_onebin", xlabel="Low-dM region bin number"))
    low_sr = flat_search_record(payload, "cat7_SR_lowDeltaM", "SR low $\Delta m$", allow_signal=True)
    if low_sr:
        low_sr["blind_data"] = True
        plots.append(draw_flat_blocks([low_sr], output_dir / "lowdm_sr_onebin", xlabel="Low-dM SR bin number"))
    summary = {"status": "complete", "source": str(flat_hists), "output_dir": str(output_dir), "plots": plots, "signal_policy": "Signals are drawn only in SR plots; CR blocks exclude T2tt overlays.", "ntop_order": "N_t = 0 blocks are placed left of N_t >= 1 blocks."}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--signal-searchbin-yields", default="docs/data/signal_searchbin_yields.json", type=Path)
    parser.add_argument("--name", default="partial_control_search_bins_style")
    parser.add_argument("--flat-hists", type=Path)
    parser.add_argument("--flat-output-dir", type=Path)
    args = parser.parse_args()

    if args.flat_hists:
        outdir = args.flat_output_dir or Path(args.docs_dir or ".") / "plots"
        print(json.dumps(draw_flat_report(args.flat_hists, outdir), sort_keys=True))
        return 0

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
