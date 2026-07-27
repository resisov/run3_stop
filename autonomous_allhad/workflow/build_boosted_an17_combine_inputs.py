#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEME = "boosted_an_17"
BACKGROUND_NAME = "background"
LUMI_NAME = "Lumi_2024"
LUMI_LNN = 1.016
MIN_BIN = 1.0e-9
CONTROL_REGIONS = [
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
]
SR_CHANNEL = "cat7_SR_boosted_an_17"
DATA_PROCESSES = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}
SHAPE_DIRS = {
    "jesTotalUp": "jesTotal",
    "jesTotalDown": "jesTotal",
    "metUnclusteredUp": "metUnclustered",
    "metUnclusteredDown": "metUnclustered",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sanitize(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return out or "unnamed"


def signal_process_name(mass_key: str) -> str:
    return "sig_" + sanitize(mass_key)


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.match(r"mStop(\d+)_mLSP(\d+)$", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def stable_path(path: Path) -> str:
    return str(path.resolve()).replace("/eos/home-t/taiwoo", "/eos/user/t/taiwoo")


def finite_edges(edges: list[Any]) -> np.ndarray:
    vals: list[float] = []
    for item in edges:
        if item == "inf":
            vals.append(float("inf"))
        else:
            vals.append(float(item))
    finite: list[float] = []
    for val in vals:
        if math.isinf(val):
            finite.append(finite[-1] + (finite[-1] - finite[-2]) if len(finite) >= 2 else (finite[-1] + 1.0 if finite else 1.0))
        else:
            finite.append(val)
    return np.asarray(finite, dtype=float)


def unit_edges(nbin: int) -> np.ndarray:
    return np.asarray([float(i) for i in range(nbin + 1)], dtype=float)


def add_arrays(target: np.ndarray, source: Any, scale: float = 1.0) -> None:
    vals = np.asarray(source or [], dtype=float)
    n = min(len(target), len(vals))
    if n:
        target[:n] += vals[:n] * float(scale)


def make_hist(name: str, values: np.ndarray, sumw2: np.ndarray, edges: np.ndarray):
    import ROOT

    hist = ROOT.TH1D(name, name, len(values), array("d", [float(x) for x in edges]))
    hist.Sumw2()
    for idx, value in enumerate(values, start=1):
        hist.SetBinContent(idx, float(max(value, MIN_BIN)))
        err2 = float(sumw2[idx - 1]) if idx - 1 < len(sumw2) else 0.0
        hist.SetBinError(idx, math.sqrt(max(err2, 0.0)))
    return hist


def write_hist(directory, name: str, values: np.ndarray, sumw2: np.ndarray, edges: np.ndarray) -> None:
    directory.cd()
    hist = make_hist(name, values, sumw2, edges)
    hist.Write(name, 1)


def cr_channel_from_preview(preview: dict[str, Any], region: str, variable: str = "recoil_pt") -> dict[str, Any]:
    bkg_rec = (((preview.get("histograms") or {}).get("background") or {}).get(variable) or {}).get(region) or {}
    data_rec = (((preview.get("histograms") or {}).get("data") or {}).get(variable) or {}).get(region) or {}
    if not bkg_rec:
        raise ValueError(f"missing background histogram for {variable}/{region}")
    first = next(iter(bkg_rec.values()))
    edges = finite_edges(first.get("bin_edges") or [])
    nbin = len(edges) - 1
    bkg = np.zeros(nbin, dtype=float)
    stat2 = np.zeros(nbin, dtype=float)
    for proc, hist in bkg_rec.items():
        add_arrays(bkg, hist.get("values"))
        add_arrays(stat2, hist.get("sumw2"))
    data = np.zeros(nbin, dtype=float)
    data_stat2 = np.zeros(nbin, dtype=float)
    for hist in data_rec.values():
        add_arrays(data, hist.get("values"))
        add_arrays(data_stat2, hist.get("sumw2"))
    variations: dict[str, dict[str, np.ndarray]] = {}
    hvars = ((((preview.get("histogram_systematic_variations") or {}).get("background") or {}).get(variable) or {}).get(region) or {})
    for syst_name, syst_rec in hvars.items():
        up = bkg + np.asarray(syst_rec.get("up_delta") or [0.0] * nbin, dtype=float)
        down = bkg + np.asarray(syst_rec.get("down_delta") or [0.0] * nbin, dtype=float)
        variations[syst_name] = {"up": up, "down": down}
    return {
        "name": region,
        "source_region": region,
        "kind": "control_recoil_6bin",
        "edges": edges,
        "background": bkg,
        "background_sumw2": stat2,
        "data": data,
        "data_sumw2": data_stat2,
        "variations": variations,
        "variable": variable,
    }


def search_bin_order(preview: dict[str, Any]) -> list[str]:
    bins = list(((preview.get("search_bins") or {}).get(SCHEME) or {}).keys())
    if not bins:
        raise ValueError(f"missing search bin scheme {SCHEME}")
    return bins


def sr_nominal_from_preview(preview: dict[str, Any], bins: list[str]) -> dict[str, Any]:
    sb = (preview.get("search_bins") or {}).get(SCHEME) or {}
    nbin = len(bins)
    bkg = np.zeros(nbin, dtype=float)
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_stat2 = np.zeros(nbin, dtype=float)
    by_process: dict[str, list[float]] = defaultdict(lambda: [0.0] * nbin)
    for idx, bin_name in enumerate(bins):
        rec = sb.get(bin_name) or {}
        for proc, prec in rec.items():
            val = float(prec.get("normalized_weighted") or 0.0)
            s2 = float(prec.get("normalized_sumw2") or 0.0)
            if proc in DATA_PROCESSES or prec.get("kind") == "data":
                data[idx] += val
                data_stat2[idx] += s2
            else:
                bkg[idx] += val
                stat2[idx] += s2
                by_process[proc][idx] += val
    return {
        "name": SR_CHANNEL,
        "kind": "signal_region_boosted_an17_17bin",
        "edges": unit_edges(nbin),
        "background": bkg,
        "background_sumw2": stat2,
        "data": data,
        "data_sumw2": data_stat2,
        "variations": {},
        "bin_labels": bins,
        "background_by_process": {k: v for k, v in sorted(by_process.items())},
    }


def normalization_factor_map(preview: dict[str, Any]) -> dict[str, float]:
    factors: dict[str, float] = {}
    for ds, rec in (preview.get("normalization_factors") or {}).items():
        if rec.get("is_data"):
            continue
        factors[ds] = float(rec.get("normalization_factor") or 0.0)
    return factors


def raw_search_arr(rec: dict[str, Any], bins: list[str], source_key: str = "search_bins") -> tuple[np.ndarray, np.ndarray]:
    nbin = len(bins)
    vals = np.zeros(nbin, dtype=float)
    s2 = np.zeros(nbin, dtype=float)
    sb = ((rec.get(source_key) or {}).get(SCHEME) or {})
    for idx, bin_name in enumerate(bins):
        b = sb.get(bin_name) or {}
        vals[idx] = float(b.get("raw_weighted") or 0.0)
        s2[idx] = float(b.get("raw_sumw2") or 0.0)
    return vals, s2


def physical_nominal_dataset_name(ds: str) -> str:
    return ds


def aggregate_nominal_search_datasets(nominal_dir: Path, factors: dict[str, float], bins: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    nominal_by_dataset: dict[str, np.ndarray] = {}
    process_by_dataset: dict[str, str] = {}
    meta = {"files": 0, "datasets": 0, "missing_factor": 0}
    for path in sorted(nominal_dir.glob("mc_shard_*.json")):
        meta["files"] += 1
        payload = read_json(path)
        for ds, rec in (payload.get("datasets") or {}).items():
            proc = str(rec.get("process") or "")
            if rec.get("is_data") or proc in DATA_PROCESSES:
                continue
            factor = factors.get(ds)
            if factor is None:
                meta["missing_factor"] += 1
                continue
            vals, _ = raw_search_arr(rec, bins)
            nominal_by_dataset[ds] = vals * factor
            process_by_dataset[ds] = proc
            meta["datasets"] += 1
    return nominal_by_dataset, process_by_dataset, meta


def aggregate_weight_variation_deltas(nominal_dir: Path, factors: dict[str, float], bins: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    nbin = len(bins)
    deltas: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(nbin, dtype=float))
    meta = {"files": 0, "datasets": 0, "variation_names": set(), "missing_factor": 0}
    for path in sorted(nominal_dir.glob("mc_shard_*.json")):
        meta["files"] += 1
        payload = read_json(path)
        for ds, rec in (payload.get("datasets") or {}).items():
            proc = str(rec.get("process") or "")
            if rec.get("is_data") or proc in DATA_PROCESSES:
                continue
            factor = factors.get(ds)
            if factor is None:
                meta["missing_factor"] += 1
                continue
            sbv = ((rec.get("search_bin_variations") or {}).get(SCHEME) or {})
            nominal = np.zeros(nbin, dtype=float)
            for idx, bin_name in enumerate(bins):
                bin_vars = sbv.get(bin_name) or {}
                nrec = bin_vars.get("nominal")
                if nrec is not None:
                    nominal[idx] = float(nrec.get("raw_weighted") or 0.0) * factor
                else:
                    nominal[idx] = raw_search_arr(rec, bins)[0][idx] * factor
            var_names = sorted({name for bin_rec in sbv.values() for name in (bin_rec or {}) if name != "nominal"})
            for var_name in var_names:
                arr = np.zeros(nbin, dtype=float)
                for idx, bin_name in enumerate(bins):
                    vrec = (sbv.get(bin_name) or {}).get(var_name)
                    arr[idx] = float(vrec.get("raw_weighted") or 0.0) * factor if vrec is not None else nominal[idx]
                deltas[var_name] += arr - nominal
                meta["variation_names"].add(var_name)
            meta["datasets"] += 1
    meta["variation_names"] = sorted(meta["variation_names"])
    return dict(deltas), meta



def aggregate_nominal_and_weight_variation_deltas(nominal_dir: Path, factors: dict[str, float], bins: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    nbin = len(bins)
    nominal_by_dataset: dict[str, np.ndarray] = {}
    deltas: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(nbin, dtype=float))
    meta = {"files": 0, "datasets": 0, "variation_names": set(), "missing_factor": 0}
    for path in sorted(nominal_dir.glob("mc_shard_*.json")):
        meta["files"] += 1
        payload = read_json(path)
        for ds, rec in (payload.get("datasets") or {}).items():
            proc = str(rec.get("process") or "")
            if rec.get("is_data") or proc in DATA_PROCESSES:
                continue
            factor = factors.get(ds)
            if factor is None:
                meta["missing_factor"] += 1
                continue
            raw_nominal, _ = raw_search_arr(rec, bins)
            nominal = raw_nominal * factor
            nominal_by_dataset[ds] = nominal
            sbv = ((rec.get("search_bin_variations") or {}).get(SCHEME) or {})
            var_names = sorted({name for bin_rec in sbv.values() for name in (bin_rec or {}) if name != "nominal"})
            for var_name in var_names:
                arr = nominal.copy()
                for idx, bin_name in enumerate(bins):
                    vrec = (sbv.get(bin_name) or {}).get(var_name)
                    if vrec is not None:
                        arr[idx] = float(vrec.get("raw_weighted") or 0.0) * factor
                deltas[var_name] += arr - nominal
                meta["variation_names"].add(var_name)
            meta["datasets"] += 1
    meta["variation_names"] = sorted(meta["variation_names"])
    return nominal_by_dataset, dict(deltas), meta


def serialize_variations(variations: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {name: {"up": [float(x) for x in pair["up"]], "down": [float(x) for x in pair["down"]]} for name, pair in variations.items()}


def deserialize_variations(payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {name: {"up": np.asarray(pair.get("up") or [], dtype=float), "down": np.asarray(pair.get("down") or [], dtype=float)} for name, pair in payload.items()}

def aggregate_shape_shift_delta(shape_dir: Path, factors: dict[str, float], bins: list[str], nominal_by_dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    nbin = len(bins)
    delta = np.zeros(nbin, dtype=float)
    meta = {"files": 0, "datasets": 0, "missing_nominal_dataset": 0, "missing_factor": 0}
    for path in sorted(shape_dir.glob("shard_*.json")):
        meta["files"] += 1
        payload = read_json(path)
        for ds, rec in (payload.get("datasets") or {}).items():
            proc = str(rec.get("process") or "")
            if rec.get("is_data") or proc in DATA_PROCESSES:
                continue
            factor = factors.get(ds)
            if factor is None:
                meta["missing_factor"] += 1
                continue
            nominal = nominal_by_dataset.get(ds)
            if nominal is None:
                meta["missing_nominal_dataset"] += 1
                continue
            vals, _ = raw_search_arr(rec, bins)
            delta += vals * factor - nominal
            meta["datasets"] += 1
    return delta, meta


def pair_variations(nominal: np.ndarray, variation_values: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    paired: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in variation_values.items():
        if name.endswith("Up"):
            base = name[:-2]
            paired.setdefault(base, {})["up"] = arr
        elif name.endswith("Down"):
            base = name[:-4]
            paired.setdefault(base, {})["down"] = arr
    complete: dict[str, dict[str, np.ndarray]] = {}
    for base, pair in paired.items():
        if "up" in pair or "down" in pair:
            complete[base] = {
                "up": pair.get("up", nominal.copy()),
                "down": pair.get("down", nominal.copy()),
            }
    return complete


def add_sr_systematics(sr: dict[str, Any], preview: dict[str, Any], nominal_dir: Path, shape_base: Path | None, shape_prefix: str, bins: list[str], cache_path: Path | None = None) -> dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        cache = read_json(cache_path)
        sr["variations"] = deserialize_variations(cache.get("variations") or {})
        meta = cache.get("meta") or {}
        meta["cache"] = {"status": "loaded", "path": str(cache_path)}
        meta["sr_shape_nuisances"] = sorted(sr["variations"].keys())
        return meta

    factors = normalization_factor_map(preview)
    nominal_by_dataset, weight_deltas, weight_meta = aggregate_nominal_and_weight_variation_deltas(nominal_dir, factors, bins)
    nominal = np.asarray(sr["background"], dtype=float)
    variation_values: dict[str, np.ndarray] = {name: nominal + delta for name, delta in weight_deltas.items()}
    shape_meta: dict[str, Any] = {}
    if shape_base is not None:
        for suffix in SHAPE_DIRS:
            shape_dir = shape_base / f"{shape_prefix}_{suffix}"
            if not shape_dir.exists():
                shape_meta[suffix] = {"status": "missing", "path": str(shape_dir)}
                continue
            delta, meta = aggregate_shape_shift_delta(shape_dir, factors, bins, nominal_by_dataset)
            variation_values[suffix] = nominal + delta
            meta["path"] = str(shape_dir)
            shape_meta[suffix] = meta
    sr["variations"] = pair_variations(nominal, variation_values)
    meta = {
        "normalization_factor_count": len(factors),
        "nominal_and_weight_aggregation": weight_meta,
        "shape_variation_aggregation": shape_meta,
        "sr_shape_nuisances": sorted(sr["variations"].keys()),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_path, {"meta": meta, "variations": serialize_variations(sr["variations"])})
        meta["cache"] = {"status": "written", "path": str(cache_path)}
    return meta


def mass_points_from_signal(signal_searchbins: dict[str, Any], max_points: int | None, only: list[str] | None) -> list[str]:
    scheme = signal_searchbins.get("yields", {}).get(SCHEME) or {}
    keys: set[str] = set()
    for bin_rec in scheme.values():
        keys.update(bin_rec.keys())
    selected = []
    for key in keys:
        if only and key not in only:
            continue
        try:
            mstop, mlsp = parse_mass_key(key)
        except ValueError:
            continue
        if mlsp >= mstop:
            continue
        selected.append((mstop, mlsp, key))
    selected.sort()
    out = [key for _, _, key in selected]
    if max_points is not None:
        out = out[:max_points]
    return out


def signal_array(signal_searchbins: dict[str, Any], bins: list[str], mass_key: str) -> tuple[np.ndarray, np.ndarray]:
    scheme = signal_searchbins.get("yields", {}).get(SCHEME) or {}
    vals = np.zeros(len(bins), dtype=float)
    s2 = np.zeros(len(bins), dtype=float)
    for idx, bin_name in enumerate(bins):
        rec = (scheme.get(bin_name) or {}).get(mass_key) or {}
        vals[idx] = float(rec.get("normalized_weighted") or 0.0)
        s2[idx] = float(rec.get("normalized_sumw2") or 0.0)
    return vals, s2


def build_root(channels: list[dict[str, Any]], signal_searchbins: dict[str, Any], mass_keys: list[str], output_root: Path, data_mode: str) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {"channels": {}, "signals": {}, "background_shape_nuisances": sorted({name for ch in channels for name in ch.get("variations", {})})}
    try:
        for channel in channels:
            name = channel["name"]
            directory = root_file.mkdir(name)
            edges = np.asarray(channel["edges"], dtype=float)
            bkg = np.asarray(channel["background"], dtype=float)
            bkg_s2 = np.asarray(channel["background_sumw2"], dtype=float)
            data = bkg if data_mode == "asimov" else np.asarray(channel["data"], dtype=float)
            write_hist(directory, "data_obs", data, np.maximum(data, 0.0), edges)
            write_hist(directory, BACKGROUND_NAME, bkg, bkg_s2, edges)
            for syst_name, pair in (channel.get("variations") or {}).items():
                up = np.asarray(pair.get("up", bkg), dtype=float)
                down = np.asarray(pair.get("down", bkg), dtype=float)
                if len(up) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Up", up, bkg_s2, edges)
                if len(down) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Down", down, bkg_s2, edges)
            summary["channels"][name] = {
                "kind": channel.get("kind"),
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted((channel.get("variations") or {}).keys()),
            }
            if channel.get("bin_labels"):
                summary["channels"][name]["bin_labels"] = channel.get("bin_labels")
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                if name == SR_CHANNEL:
                    sig, sig_s2 = signal_array(signal_searchbins, channel["bin_labels"], mass_key)
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    root_summary: dict[str, Any],
    auto_mc_stats: int,
    lumi_name: str = LUMI_NAME,
    lumi_lnn: float = LUMI_LNN,
) -> str:
    channel_names = [ch["name"] for ch in channels]
    proc = signal_process_name(mass_key)
    columns: list[tuple[str, str, int]] = []
    grouping = root_summary.get("background_grouping_contract") or {}
    grouped_process_order = list(grouping.get("process_order") or [])
    grouped_model = bool(grouped_process_order)
    grouped_process_ids = {
        process: index + 1 for index, process in enumerate(grouped_process_order)
    }
    for channel in channel_names:
        sig_yield = float((((root_summary.get("signals") or {}).get(mass_key) or {}).get("channels") or {}).get(channel, 0.0))
        if sig_yield > 0.0:
            columns.append((channel, proc, 0))
        if grouped_model:
            available = (
                ((root_summary.get("channels") or {}).get(channel) or {})
                .get("background_processes")
                or {}
            )
            for background_process in grouped_process_order:
                if background_process in available:
                    columns.append(
                        (
                            channel,
                            background_process,
                            grouped_process_ids[background_process],
                        )
                    )
        else:
            columns.append((channel, BACKGROUND_NAME, 1))
    signal_systs = set(
        (((root_summary.get("signals") or {}).get(mass_key) or {}).get("shape_nuisances") or {}).keys()
    )
    all_systs = sorted({s for ch in channels for s in (ch.get("variations") or {})} | signal_systs)
    lines = [
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {stable_path(template_root)} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
        "------------",
        "bin " + " ".join(channel_names),
        "observation " + " ".join(["-1"] * len(channel_names)),
        "------------",
        "bin " + " ".join(c[0] for c in columns),
        "process " + " ".join(c[1] for c in columns),
        "process " + " ".join(str(c[2]) for c in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
    ]
    for syst_name in all_systs:
        mask = []
        for channel, process, _ in columns:
            channel_summary = (root_summary.get("channels") or {}).get(channel) or {}
            if grouped_model and process != proc:
                has = syst_name in (
                    ((channel_summary.get("background_processes") or {}).get(process) or {})
                    .get("shape_nuisances")
                    or []
                )
            else:
                has = syst_name in (
                    channel_summary.get("background_shape_nuisances") or []
                )
            signal_channels = (
                (((root_summary.get("signals") or {}).get(mass_key) or {}).get("shape_nuisances") or {}).get(syst_name)
                or []
            )
            mask.append(
                "1"
                if (
                    (process != proc and has)
                    or (process == proc and channel in signal_channels)
                )
                else "-"
            )
        lines.append(syst_name + " shape " + " ".join(mask))
    lines.append(lumi_name + " lnN " + " ".join(f"{lumi_lnn:.3f}" for _ in columns))
    lines.extend([
        f"* autoMCStats {auto_mc_stats}",
        (
            "# Background grouping: Top=TT+ST; VV_VVV is displayed as VV+VVV; "
            "PhotonJet is displayed as Photon+jet."
            if grouped_model
            else "# Backgrounds use the legacy single-template model."
        ),
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.",
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.",
    ])
    return "\n".join(lines) + "\n"


def write_datacards(channels: list[dict[str, Any]], mass_keys: list[str], template_root: Path, root_summary: dict[str, Any], output_dir: Path, auto_mc_stats: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def write_limit_runner(cards: dict[str, str], output_dir: Path, runner: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "COMBINE=${COMBINE:-combine}", f"OUTDIR={stable_path(output_dir)}", "mkdir -p \"$OUTDIR\""]
    for mass_key, card in sorted(cards.items()):
        lines.append(f"echo '[combine] {mass_key}'")
        lines.append(f"(cd \"$OUTDIR\" && \"$COMBINE\" -M AsymptoticLimits --run blind -n _{mass_key} \"{stable_path(Path(card))}\") | tee \"$OUTDIR/log_{mass_key}.txt\"")
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("\n".join(lines) + "\n")
    runner.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", default="docs/partial_merge_preview_boosted_an17_20260629_current/partial_normalized_yields.json")
    parser.add_argument("--signal-searchbins", default="docs/data/signal_searchbin_yields.json")
    parser.add_argument("--nominal-output-dir", default="autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_nominal")
    parser.add_argument("--shape-base-dir", default="autonomous_allhad/workflow")
    parser.add_argument("--shape-prefix", default="production_outputs_boosted_an17_20260629")
    parser.add_argument("--output-dir", default="analysis/combine/boosted_an17_20260630")
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--sr-systematics-cache", default=None)
    args = parser.parse_args()

    preview_path = Path(args.preview)
    signal_path = Path(args.signal_searchbins)
    preview = read_json(preview_path)
    signal_searchbins = read_json(signal_path)
    bins = search_bin_order(preview)
    channels = [cr_channel_from_preview(preview, region) for region in CONTROL_REGIONS]
    sr = sr_nominal_from_preview(preview, bins)
    cache_path = Path(args.sr_systematics_cache) if args.sr_systematics_cache else None
    sr_syst_meta = add_sr_systematics(sr, preview, Path(args.nominal_output_dir), Path(args.shape_base_dir), args.shape_prefix, bins, cache_path)
    channels.append(sr)
    mass_keys = mass_points_from_signal(signal_searchbins, args.max_points, args.only)
    if not mass_keys:
        raise SystemExit("no mass points selected from signal search-bin yields")

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_boosted_an17.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root(channels, signal_searchbins, mass_keys, template_root, args.data_mode)
    cards = write_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_limit_runner(cards, limit_dir, runner)
    manifest = {
        "status": "combine_inputs_ready",
        "schema": "boosted_an17_searchbin_v1",
        "preview": str(preview_path),
        "signal_searchbins": str(signal_path),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "search_bin_scheme": SCHEME,
        "search_bin_count": len(bins),
        "search_bin_order": bins,
        "change_from_previous_builder": {
            "old_sr": "cat7_SR_highDeltaM metpt 9-bin template from fit_template_summary.json",
            "new_sr": "cat7_SR_boosted_an_17 17-bin boosted top/W search-bin template from partial_normalized_yields.search_bins.boosted_an_17",
            "cr": "cat2-cat6 are 6-bin recoil/U_T control channels matching the current CR/SR search-bin plot style",
        },
        "systematics_policy": {
            "cr": "Use partial preview histogram_systematic_variations for recoil_pt in cat2-cat6.",
            "sr_weight": "Reconstruct per-bin shape deltas from nominal shard search_bin_variations using preview normalization_factors.",
            "sr_shape": "Reconstruct JES and MET unclustered per-bin shape deltas from dedicated shifted production output dirs.",
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
        },
        "sr_systematic_aggregation": sr_syst_meta,
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "search_bins": len(bins),
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
        "sr_systs": sr_syst_meta.get("sr_shape_nuisances"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
