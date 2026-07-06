#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from autonomous_allhad.real_subset_worker import compute_weight_bundle

RECOIL_PT_BINS = [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0]
LOWDM_ONEBIN_LABELS = ["lowdm_inclusive"]
LOWDM_REGION_MAP = {
    "LLCR": "cat2_LLCR_lowDeltaM",
    "QCDCR": "cat3_QCDCR_lowDeltaM",
    "GCR": "cat4_GCR_lowDeltaM",
    "DY2E": "cat5_DY2E_lowDeltaM",
    "DY2M": "cat6_DY2M_lowDeltaM",
    "SR": "cat7_SR_lowDeltaM",
}
BASE_REGION_VARIABLES = {
    "LLCR": ("feature_LLCR", "met"),
    "QCDCR": ("feature_QCDCR", "met"),
    "GCR": ("feature_GCR", "recoil_gcr"),
    "DY2E": ("feature_DY2E", "recoil_dy2e"),
    "DY2M": ("feature_DY2M", "recoil_dy2m"),
    "SR": ("feature_SR", "met"),
    "SR_Nt1": ("feature_SR_Nt1", "met"),
}
NTOP_SPLIT_BASE_REGIONS = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR"]
REGION_VARIABLES = dict(BASE_REGION_VARIABLES)
for _region in NTOP_SPLIT_BASE_REGIONS:
    _flag, _var = BASE_REGION_VARIABLES[_region]
    REGION_VARIABLES.setdefault(f"{_region}_Nt1", (_flag, _var))
    REGION_VARIABLES.setdefault(f"{_region}_Nt0", (_flag, _var))
DATA_PROCESSES = {"JetMET", "EGamma", "Muon"}
DATA_PROCESS_BY_REGION = {
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
    "SR": "JetMET",
    "SR_Nt1": "JetMET",
}
for _region in NTOP_SPLIT_BASE_REGIONS:
    _process = DATA_PROCESS_BY_REGION[_region]
    DATA_PROCESS_BY_REGION.setdefault(f"{_region}_Nt1", _process)
    DATA_PROCESS_BY_REGION.setdefault(f"{_region}_Nt0", _process)
for _region, _channel in LOWDM_REGION_MAP.items():
    DATA_PROCESS_BY_REGION.setdefault(_channel, DATA_PROCESS_BY_REGION[_region])
SEARCH_BIN_ORDER = [
    "B0_Nb1", "B0_Nb2plus",
    "Nb1_T0_W1plus", "Nb1_T1plus_W0", "Nb1_T1plus_W1plus",
    "Nb2_T0_W1", "Nb2_T0_W2", "Nb2_T1_W0", "Nb2_T1_W1", "Nb2_T2_W0", "Nb2_TW_ge3",
    "Nb3plus_T0_W1", "Nb3plus_T0_W2", "Nb3plus_T1_W0", "Nb3plus_T1_W1", "Nb3plus_T2_W0", "Nb3plus_TW_ge3",
]
SEARCH_BIN_BRANCHES = ["nb_medium", "nboosted_top", "nboosted_w", "nboosted_total"]
WEIGHT_BRANCHES = [
    "run", "luminosityBlock", "event", "entry", "dataset_id", "year", "mStop", "mLSP",
    "is_data", "is_signal", "is_background", "gen_weight", "pu_ntrueint",
    "n_e_veto", "n_e_medium", "n_m_loose", "n_m_medium",
    "good_jet_pt", "good_jet_eta", "good_jet_hadron_flavour", "good_jet_b_medium",
    "electron_veto_pt", "electron_veto_eta_sc", "electron_veto_phi",
    "electron_medium_pt", "electron_medium_eta_sc", "electron_medium_phi",
    "muon_loose_pt", "muon_loose_eta", "muon_loose_phi",
    "muon_medium_pt", "muon_medium_eta", "muon_medium_phi",
    "photon_medium_pt", "photon_medium_eta", "photon_medium_phi", "gen_top_pt",
]
LOWDM_READ_BRANCHES = [
    "feature_lowdm_sr_base",
    "pass_lowdm_topology_veto", "pass_lowdm_isr", "pass_lowdm_isr_bveto", "pass_lowdm_met_sqrt_ht", "pass_lowdm_mtb",
    "pass_base_common", "pass_signal_trigger", "pass_photon_trigger", "pass_electron_trigger", "pass_muon_trigger",
    "pass_zero_tau", "pass_no_veto_leptons", "pass_one_veto_lepton", "pass_mt_100",
    "pass_met_250", "pass_ht_300", "pass_ht_photon_300", "pass_ht_lepton_300",
    "pass_open_pre", "pass_qcd_open", "pass_dphi123_0p1",
    "met", "ht", "njet", "nb_medium_lowdm", "n_photon_medium",
    "njet_photon_clean", "nb_photon_clean", "ht_photon_clean",
    "njet_lepton_clean", "nb_lepton_clean", "ht_lepton_clean",
    "mee", "pee", "mmm", "pmm", "recoil_gcr", "recoil_dy2e", "recoil_dy2m",
]
READ_BRANCHES = sorted(set(WEIGHT_BRANCHES + SEARCH_BIN_BRANCHES + LOWDM_READ_BRANCHES + [b for pair in REGION_VARIABLES.values() for b in pair]))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_array(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, fill, dtype=float)
    if out.shape == ():
        out = np.full(n, float(out), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill)


def as_bool(values: Any, n: int) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=bool)
    except Exception:
        return np.zeros(n, dtype=bool)
    if out.shape == ():
        out = np.full(n, bool(out), dtype=bool)
    if len(out) != n:
        return np.zeros(n, dtype=bool)
    return out


def ones_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.ones_like(jagged), np.bool_)


def zeros_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.zeros_like(jagged), np.bool_)


def combine_two(a: Any, b: Any) -> Any:
    return ak.concatenate([a, b], axis=1)


def flat_arrays_for_weights(chunk: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(chunk["gen_weight"])
    jet_pt = chunk["good_jet_pt"]
    jet_eta = chunk["good_jet_eta"]
    jet_had = chunk["good_jet_hadron_flavour"]
    b_med = ak.values_astype(chunk["good_jet_b_medium"], np.bool_)

    ev_pt, ev_eta, ev_phi = chunk["electron_veto_pt"], chunk["electron_veto_eta_sc"], chunk["electron_veto_phi"]
    em_pt, em_eta, em_phi = chunk["electron_medium_pt"], chunk["electron_medium_eta_sc"], chunk["electron_medium_phi"]
    e_pt = combine_two(ev_pt, em_pt)
    e_eta = combine_two(ev_eta, em_eta)
    e_phi = combine_two(ev_phi, em_phi)
    e_veto = combine_two(ones_mask(ev_pt), zeros_mask(em_pt))
    e_med = combine_two(zeros_mask(ev_pt), ones_mask(em_pt))
    e_delta_eta_sc = ak.zeros_like(e_eta)

    ml_pt, ml_eta, ml_phi = chunk["muon_loose_pt"], chunk["muon_loose_eta"], chunk["muon_loose_phi"]
    mm_pt, mm_eta, mm_phi = chunk["muon_medium_pt"], chunk["muon_medium_eta"], chunk["muon_medium_phi"]
    m_pt = combine_two(ml_pt, mm_pt)
    m_eta = combine_two(ml_eta, mm_eta)
    m_phi = combine_two(ml_phi, mm_phi)
    m_loose = combine_two(ones_mask(ml_pt), zeros_mask(mm_pt))
    m_med = combine_two(zeros_mask(ml_pt), ones_mask(mm_pt))

    p_pt = chunk["photon_medium_pt"]
    p_eta = chunk["photon_medium_eta"]
    p_phi = chunk["photon_medium_phi"]
    p_med = ones_mask(p_pt)

    top_pt = chunk["gen_top_pt"]
    top_flags = ak.values_astype(ak.ones_like(top_pt) * ((1 << 8) | (1 << 13)), np.int64)
    top_pdg = ak.values_astype(ak.ones_like(top_pt) * 6, np.int64)
    arrays = ak.Array({
        "genWeight": chunk["gen_weight"],
        "Pileup_nTrueInt": chunk["pu_ntrueint"],
        "Jet_hadronFlavour": jet_had,
        "GenPart_pt": top_pt,
        "GenPart_pdgId": top_pdg,
        "GenPart_statusFlags": top_flags,
    })
    inputs = {
        "n": n,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_hadflav": jet_had,
        "b_med": b_med,
        "e_eta": e_eta,
        "e_delta_eta_sc": e_delta_eta_sc,
        "e_pt": e_pt,
        "e_phi": e_phi,
        "e_veto": e_veto,
        "e_med": e_med,
        "n_e_veto": np.asarray(chunk["n_e_veto"], dtype=int),
        "n_e_med": np.asarray(chunk["n_e_medium"], dtype=int),
        "m_eta": m_eta,
        "m_pt": m_pt,
        "m_phi": m_phi,
        "m_loose": m_loose,
        "m_med": m_med,
        "n_m_loose": np.asarray(chunk["n_m_loose"], dtype=int),
        "n_m_med": np.asarray(chunk["n_m_medium"], dtype=int),
        "p_eta": p_eta,
        "p_pt": p_pt,
        "p_phi": p_phi,
        "p_med": p_med,
        "gcr_mask": as_bool(chunk["feature_GCR"], n),
    }
    return arrays, inputs


def process_to_group(process: str, dataset: str = "") -> str:
    if process in DATA_PROCESSES:
        return "Data"
    if process == "VV":
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "Single Top"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "ttbar"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "Gamma + Jets"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "W -> lv"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Z -> vv"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD Multijet"
    return "others"


def canonical_process(process: str, dataset: str = "") -> str:
    group = process_to_group(process, dataset)
    return {
        "VV": "VV",
        "Single Top": "ST",
        "ttbar": "TT",
        "DY": "DY",
        "Gamma + Jets": "GJ",
        "W -> lv": "WtoLNu",
        "Z -> vv": "Zto2Nu",
        "QCD Multijet": "QCD",
    }.get(group, process or "other")


def data_process_allowed(process: str, region: str) -> bool:
    expected = DATA_PROCESS_BY_REGION.get(region)
    return expected is None or process == expected


def note_data_exclusion(summary: dict[str, Any], region: str, process: str, count: int) -> None:
    if count <= 0:
        return
    rec = summary.setdefault("data_stream_exclusions", {}).setdefault(region, {}).setdefault(process, {"entries": 0})
    rec["entries"] = int(rec.get("entries", 0)) + int(count)


def dataset_label(meta: dict[str, Any], dataset_id: int) -> tuple[str, str, bool, bool]:
    rec = (meta.get("datasets") or {}).get(str(int(dataset_id))) or {}
    process = str(rec.get("process") or "unknown")
    dataset = str(rec.get("dataset") or "unknown")
    is_data = bool(rec.get("is_data"))
    is_signal = bool(rec.get("is_signal"))
    if not is_data and not is_signal:
        process = canonical_process(process, dataset)
    return dataset, process, is_data, is_signal


def norm_vector(norm: dict[str, Any], chunk: dict[str, Any], dataset_id: int, is_data: bool, is_signal: bool) -> np.ndarray:
    n = len(chunk["dataset_id"])
    if is_data:
        return np.ones(n, dtype=float)
    if is_signal:
        out = np.zeros(n, dtype=float)
        mstops = np.asarray(chunk["mStop"], dtype=int)
        mlsps = np.asarray(chunk["mLSP"], dtype=int)
        for i, (ms, ml) in enumerate(zip(mstops, mlsps)):
            key = f"mStop{int(ms)}_mLSP{int(ml)}"
            fac = ((norm.get("signal_mass_points") or {}).get(key) or {}).get("normalization_factor")
            out[i] = float(fac) if fac is not None and math.isfinite(float(fac)) else 0.0
        return out
    fac = ((norm.get("dataset_factors") or {}).get(str(int(dataset_id))) or {}).get("normalization_factor")
    if fac is None:
        return np.zeros(n, dtype=float)
    return np.full(n, float(fac), dtype=float)


def sample_label(process: str, is_data: bool, is_signal: bool, chunk: dict[str, Any]) -> str:
    if is_data:
        return "data_obs"
    if is_signal:
        mstops = np.asarray(chunk["mStop"], dtype=int)
        mlsps = np.asarray(chunk["mLSP"], dtype=int)
        if len(mstops):
            pairs, counts = np.unique(np.stack([mstops, mlsps], axis=1), axis=0, return_counts=True)
            pair = pairs[int(np.argmax(counts))]
            return f"T2tt_mStop{int(pair[0])}_mLSP{int(pair[1])}"
        return "T2tt_unknown"
    return process


def empty_hist() -> dict[str, Any]:
    nb = len(RECOIL_PT_BINS) - 1
    return {"sumw": [0.0] * nb, "sumw2": [0.0] * nb, "entries": [0] * nb}


def add_hist(target: dict[str, Any], values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> None:
    vals = values[mask]
    w = weights[mask]
    if vals.size == 0:
        return
    h, _ = np.histogram(vals, bins=np.asarray(RECOIL_PT_BINS, dtype=float), weights=w)
    h2, _ = np.histogram(vals, bins=np.asarray(RECOIL_PT_BINS, dtype=float), weights=w * w)
    e, _ = np.histogram(vals, bins=np.asarray(RECOIL_PT_BINS, dtype=float))
    target["sumw"] = (np.asarray(target["sumw"], dtype=float) + h).tolist()
    target["sumw2"] = (np.asarray(target["sumw2"], dtype=float) + h2).tolist()
    target["entries"] = (np.asarray(target["entries"], dtype=int) + e).astype(int).tolist()



def region_mask(chunk: dict[str, Any], region: str, flag: str, n: int) -> np.ndarray:
    base = as_bool(chunk[flag], n)
    if region.endswith("_Nt0"):
        return base & (np.asarray(chunk["nboosted_top"], dtype=int) == 0)
    if region.endswith("_Nt1") and region != "SR_Nt1":
        return base & (np.asarray(chunk["nboosted_top"], dtype=int) >= 1)
    return base


def bool_field(chunk: dict[str, Any], name: str, n: int, default: bool = False) -> np.ndarray:
    if name not in chunk:
        return np.full(n, default, dtype=bool)
    return as_bool(chunk[name], n)


def float_field(chunk: dict[str, Any], name: str, n: int, fill: float = 0.0) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=float)
    return finite_array(chunk[name], n, fill)


def int_field(chunk: dict[str, Any], name: str, n: int, fill: int = 0) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=int)
    try:
        out = np.asarray(chunk[name], dtype=int)
    except Exception:
        return np.full(n, fill, dtype=int)
    if out.shape == ():
        out = np.full(n, int(out), dtype=int)
    if len(out) != n:
        return np.full(n, fill, dtype=int)
    return out


def lowdm_common_mask(chunk: dict[str, Any], n: int) -> np.ndarray:
    return (
        bool_field(chunk, "pass_base_common", n)
        & bool_field(chunk, "pass_zero_tau", n)
        & bool_field(chunk, "pass_lowdm_topology_veto", n)
        & bool_field(chunk, "pass_lowdm_isr", n)
        & bool_field(chunk, "pass_lowdm_isr_bveto", n)
        & bool_field(chunk, "pass_lowdm_met_sqrt_ht", n)
        & bool_field(chunk, "pass_lowdm_mtb", n)
    )


def lowdm_region_mask(chunk: dict[str, Any], region: str, n: int) -> np.ndarray:
    common = lowdm_common_mask(chunk, n)
    njet = int_field(chunk, "njet", n)
    met = float_field(chunk, "met", n)
    recoil_gcr = float_field(chunk, "recoil_gcr", n)
    recoil_dy2e = float_field(chunk, "recoil_dy2e", n)
    recoil_dy2m = float_field(chunk, "recoil_dy2m", n)
    mee = float_field(chunk, "mee", n, -99.0)
    mmm = float_field(chunk, "mmm", n, -99.0)

    if region == "SR":
        return common & bool_field(chunk, "pass_signal_trigger", n) & bool_field(chunk, "pass_no_veto_leptons", n) & (njet >= 2) & bool_field(chunk, "pass_met_250", n) & bool_field(chunk, "pass_open_pre", n) & bool_field(chunk, "pass_ht_300", n)
    if region == "LLCR":
        return common & bool_field(chunk, "pass_signal_trigger", n) & bool_field(chunk, "pass_one_veto_lepton", n) & bool_field(chunk, "pass_mt_100", n) & (njet >= 2) & bool_field(chunk, "pass_met_250", n) & bool_field(chunk, "pass_open_pre", n) & bool_field(chunk, "pass_ht_300", n)
    if region == "QCDCR":
        return common & bool_field(chunk, "pass_signal_trigger", n) & bool_field(chunk, "pass_no_veto_leptons", n) & (njet >= 2) & bool_field(chunk, "pass_met_250", n) & bool_field(chunk, "pass_qcd_open", n) & bool_field(chunk, "pass_dphi123_0p1", n) & bool_field(chunk, "pass_ht_300", n)
    if region == "GCR":
        return common & bool_field(chunk, "pass_photon_trigger", n) & bool_field(chunk, "pass_no_veto_leptons", n) & (int_field(chunk, "n_photon_medium", n) == 1) & (int_field(chunk, "njet_photon_clean", n) >= 2) & (met < 250.0) & (recoil_gcr > 250.0) & bool_field(chunk, "pass_ht_photon_300", n)
    if region == "DY2E":
        return common & bool_field(chunk, "pass_electron_trigger", n) & (int_field(chunk, "n_m_loose", n) == 0) & (int_field(chunk, "n_e_medium", n) == 2) & (int_field(chunk, "njet_lepton_clean", n) >= 2) & (recoil_dy2e > 200.0) & (mee > 81.0) & (mee < 101.0) & bool_field(chunk, "pass_ht_lepton_300", n)
    if region == "DY2M":
        return common & bool_field(chunk, "pass_muon_trigger", n) & (int_field(chunk, "n_e_veto", n) == 0) & (int_field(chunk, "n_m_medium", n) == 2) & (int_field(chunk, "njet_lepton_clean", n) >= 2) & (recoil_dy2m > 200.0) & (mmm > 81.0) & (mmm < 101.0) & bool_field(chunk, "pass_ht_lepton_300", n)
    raise ValueError(f"unknown low-dM region: {region}")


def empty_index_hist(nbin: int) -> dict[str, Any]:
    return {"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin}


def add_index_hist(target: dict[str, Any], indices: np.ndarray, weights: np.ndarray) -> None:
    nbin = len(target["sumw"])
    idx = np.asarray(indices, dtype=int)
    w = np.asarray(weights, dtype=float)
    mask = (idx >= 0) & (idx < nbin) & np.isfinite(w)
    if not np.any(mask):
        return
    h = np.bincount(idx[mask], weights=w[mask], minlength=nbin)[:nbin]
    h2 = np.bincount(idx[mask], weights=w[mask] * w[mask], minlength=nbin)[:nbin]
    e = np.bincount(idx[mask], minlength=nbin)[:nbin]
    target["sumw"] = (np.asarray(target["sumw"], dtype=float) + h).tolist()
    target["sumw2"] = (np.asarray(target["sumw2"], dtype=float) + h2).tolist()
    target["entries"] = (np.asarray(target["entries"], dtype=int) + e).astype(int).tolist()


def boosted_an17_indices(chunk: dict[str, Any], n: int, sr_mask: np.ndarray) -> np.ndarray:
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    nt = np.asarray(chunk["nboosted_top"], dtype=int)
    nw = np.asarray(chunk["nboosted_w"], dtype=int)
    total = np.asarray(chunk["nboosted_total"], dtype=int)
    out = np.full(n, -1, dtype=int)
    labels = {name: idx for idx, name in enumerate(SEARCH_BIN_ORDER)}
    rules = [
        ("B0_Nb1", (nb == 1) & (total == 0)),
        ("B0_Nb2plus", (nb >= 2) & (total == 0)),
        ("Nb1_T0_W1plus", (nb == 1) & (nt == 0) & (nw >= 1)),
        ("Nb1_T1plus_W0", (nb == 1) & (nt >= 1) & (nw == 0)),
        ("Nb1_T1plus_W1plus", (nb == 1) & (nt >= 1) & (nw >= 1)),
        ("Nb2_T0_W1", (nb == 2) & (nt == 0) & (nw == 1)),
        ("Nb2_T0_W2", (nb == 2) & (nt == 0) & (nw == 2)),
        ("Nb2_T1_W0", (nb == 2) & (nt == 1) & (nw == 0)),
        ("Nb2_T1_W1", (nb == 2) & (nt == 1) & (nw == 1)),
        ("Nb2_T2_W0", (nb == 2) & (nt == 2) & (nw == 0)),
        ("Nb2_TW_ge3", (nb == 2) & (total >= 3)),
        ("Nb3plus_T0_W1", (nb >= 3) & (nt == 0) & (nw == 1)),
        ("Nb3plus_T0_W2", (nb >= 3) & (nt == 0) & (nw == 2)),
        ("Nb3plus_T1_W0", (nb >= 3) & (nt == 1) & (nw == 0)),
        ("Nb3plus_T1_W1", (nb >= 3) & (nt == 1) & (nw == 1)),
        ("Nb3plus_T2_W0", (nb >= 3) & (nt == 2) & (nw == 0)),
        ("Nb3plus_TW_ge3", (nb >= 3) & (total >= 3)),
    ]
    for label, mask in rules:
        fill = sr_mask & mask & (out < 0)
        out[fill] = labels[label]
    return out


def process_root(repo: Path, root_path: Path, norm: dict[str, Any], histograms: dict[str, Any], search_histograms: dict[str, Any], summary: dict[str, Any], step_size: int) -> None:
    meta_path = root_path.with_suffix(".json")
    if not meta_path.exists():
        summary.setdefault("missing_sidecars", []).append(str(root_path))
        return
    meta = read_json(meta_path)
    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        present = set(tree.keys())
        branches = [b for b in READ_BRANCHES if b in present]
        for chunk in tree.iterate(branches, step_size=step_size, library="ak"):
            n = len(chunk["dataset_id"])
            if n == 0:
                continue
            dsids = np.asarray(chunk["dataset_id"], dtype=np.int64)
            for dsid in sorted(set(int(x) for x in dsids)):
                mask_ds = dsids == dsid
                sub = {name: chunk[name][mask_ds] for name in ak.fields(chunk)}
                dataset, process, is_data, is_signal = dataset_label(meta, dsid)
                subgroups: list[np.ndarray]
                if is_signal:
                    mstops = np.asarray(sub["mStop"], dtype=int)
                    mlsps = np.asarray(sub["mLSP"], dtype=int)
                    pairs = sorted(set(zip(mstops.tolist(), mlsps.tolist())))
                    subgroups = [(mstops == ms) & (mlsps == ml) for ms, ml in pairs]
                else:
                    subgroups = [np.ones(int(np.count_nonzero(mask_ds)), dtype=bool)]

                for mask_group in subgroups:
                    if not np.any(mask_group):
                        continue
                    sub_group = {name: arr[mask_group] for name, arr in sub.items()}
                    arrays, inputs = flat_arrays_for_weights(sub_group)
                    year_vals = np.asarray(sub_group["year"], dtype=int)
                    year = str(int(year_vals[0])) if len(year_vals) else "2024"
                    try:
                        _gen, variations, status = compute_weight_bundle(
                            arrays, repo, dataset, process, year, inputs["n"],
                            inputs["jet_pt"], inputs["jet_eta"], inputs["jet_hadflav"], inputs["b_med"],
                            inputs["e_eta"], inputs["e_delta_eta_sc"], inputs["e_pt"], inputs["e_phi"], inputs["e_veto"], inputs["e_med"], inputs["n_e_veto"], inputs["n_e_med"],
                            inputs["m_eta"], inputs["m_pt"], inputs["m_phi"], inputs["m_loose"], inputs["m_med"], inputs["n_m_loose"], inputs["n_m_med"],
                            inputs["p_eta"], inputs["p_pt"], inputs["p_phi"], inputs["p_med"], inputs["gcr_mask"],
                        )
                    except Exception as exc:
                        summary.setdefault("weight_failures", []).append({"root": str(root_path), "dataset_id": dsid, "dataset": dataset, "process": process, "label": sample_label(process, is_data, is_signal, sub_group), "error": f"{type(exc).__name__}: {exc}"[:500]})
                        variations = {"nominal": np.asarray(sub_group["gen_weight"], dtype=float)} if not is_data else {"nominal": np.ones(inputs["n"], dtype=float)}
                        status = {"applied": False, "error": "fallback_raw_gen_weight"}
                    normv = norm_vector(norm, sub_group, dsid, is_data, is_signal)
                    label = sample_label(process, is_data, is_signal, sub_group)
                    summary.setdefault("scale_factor_status", {}).setdefault(label, status)
                    for region, (flag, var) in REGION_VARIABLES.items():
                        rmask = region_mask(sub_group, region, flag, inputs["n"])
                        if is_data and not data_process_allowed(process, region):
                            note_data_exclusion(summary, region, process, int(np.count_nonzero(rmask)))
                            continue
                        values = finite_array(sub_group[var], inputs["n"], 0.0)
                        for vname, wraw in variations.items():
                            weights = finite_array(wraw, inputs["n"], 0.0) * normv
                            target = histograms.setdefault(region, {}).setdefault(label, {}).setdefault(vname, empty_hist())
                            add_hist(target, values, weights, rmask)
                    for scheme, flag_name, data_region in [
                        ("boosted_an_17_SR", "feature_SR", "SR"),
                        ("boosted_an_17_SR_Nt1", "feature_SR_Nt1", "SR_Nt1"),
                    ]:
                        sr_search_mask = as_bool(sub_group[flag_name], inputs["n"])
                        search_indices = boosted_an17_indices(sub_group, inputs["n"], sr_search_mask)
                        if is_data and not data_process_allowed(process, data_region):
                            note_data_exclusion(summary, scheme, process, int(np.count_nonzero(search_indices >= 0)))
                        else:
                            for vname, wraw in variations.items():
                                weights = finite_array(wraw, inputs["n"], 0.0) * normv
                                target = search_histograms.setdefault(scheme, {}).setdefault(label, {}).setdefault(vname, empty_index_hist(len(SEARCH_BIN_ORDER)))
                                add_index_hist(target, search_indices, weights)

                    for lowdm_region, lowdm_channel in LOWDM_REGION_MAP.items():
                        lowdm_mask = lowdm_region_mask(sub_group, lowdm_region, inputs["n"])
                        lowdm_indices = np.where(lowdm_mask, 0, -1)
                        if is_data and not data_process_allowed(process, lowdm_channel):
                            note_data_exclusion(summary, lowdm_channel, process, int(np.count_nonzero(lowdm_mask)))
                            continue
                        for vname, wraw in variations.items():
                            weights = finite_array(wraw, inputs["n"], 0.0) * normv
                            target = search_histograms.setdefault(lowdm_channel, {}).setdefault(label, {}).setdefault(vname, empty_index_hist(len(LOWDM_ONEBIN_LABELS)))
                            add_index_hist(target, lowdm_indices, weights)
                    summary["events_processed"] = int(summary.get("events_processed", 0)) + inputs["n"]


def expand_roots(inputs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            roots.extend(sorted(p.glob("*.root")))
        else:
            roots.extend(sorted(Path(x) for x in p.parent.glob(p.name)) if any(ch in item for ch in "*?[") else [p])
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build boosted Nt>=1 recoil-pT histograms from flat ntuples with post-skim SFs.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-size", type=int, default=200000)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    norm = read_json(Path(args.normalization))
    histograms: dict[str, Any] = {}
    search_histograms: dict[str, Any] = {}
    summary: dict[str, Any] = {"events_processed": 0, "input_roots": []}
    for root_path in expand_roots(args.inputs):
        if root_path.name.startswith("validation") or not root_path.exists():
            continue
        summary["input_roots"].append(str(root_path))
        process_root(repo, root_path, norm, histograms, search_histograms, summary, args.step_size)
    payload = {
        "schema_version": "flat_boosted_recoil_hists_v1",
        "status": "complete" if not summary.get("weight_failures") else "complete_with_weight_fallbacks",
        "recoil_pt_bins": RECOIL_PT_BINS,
        "regions": REGION_VARIABLES,
        "ntop_split_policy": {
            "status": "included",
            "axis": "nboosted_top",
            "split_regions": {region: {"Nt1": f"{region}_Nt1", "Nt0": f"{region}_Nt0"} for region in NTOP_SPLIT_BASE_REGIONS},
            "note": "SR_Nt1 is the feature-side SR nTop>=1 branch. Other *_Nt1 regions are built as base region AND nboosted_top>=1; *_Nt0 regions are base region AND nboosted_top==0.",
        },
        "search_bin_schemes": {
            "boosted_an_17_SR": {"bin_labels": SEARCH_BIN_ORDER, "selection": "feature_SR"},
            "boosted_an_17_SR_Nt1": {"bin_labels": SEARCH_BIN_ORDER, "selection": "feature_SR_Nt1"},
            **{channel: {"bin_labels": LOWDM_ONEBIN_LABELS, "selection": f"lowdm_common_and_{region}", "delta_m": "low", "region": region} for region, channel in LOWDM_REGION_MAP.items()},
        },
        "lowdm_region_policy": {
            "status": "adopted_from_user_2026-07-05",
            "search_bins": "one inclusive bin per low-dM region",
            "regions": LOWDM_REGION_MAP,
            "note": "Low-dM uses the same CR/SR region names as high-dM: LLCR, QCDCR, GCR, DY2E, DY2M, SR. Current flat skim lacks dilepton charge and recoil-phi branches, so DY low-dM CRs are provisional Z-window/dilepton-count/recoil selections in this post-skim builder; exact OS and recoil-phi based CR flags should be added to the next skim schema.",
        },
        "normalization": str(args.normalization),
        "data_region_process_policy": DATA_PROCESS_BY_REGION,
        "summary": summary,
        "histograms": histograms,
        "search_bin_histograms": search_histograms,
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"status": payload["status"], "input_roots": len(summary["input_roots"]), "events_processed": summary["events_processed"], "regions": len(histograms), "search_bin_schemes": len(search_histograms), "output": args.output}, sort_keys=True))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
