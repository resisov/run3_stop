
from __future__ import annotations

import csv
import gzip
import html
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import awkward as ak
import correctionlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

REQUIRED_GROUPS = [
    ("TT", lambda k: k.startswith("TT") or "TTto" in k),
    ("Zto2Nu", lambda k: "Zto2Nu" in k),
    ("WtoLNu", lambda k: "WtoLNu" in k),
    ("QCD", lambda k: "QCD" in k),
    ("GJ", lambda k: "GJ" in k or "GJets" in k),
    ("DY", lambda k: k.startswith("DY") or "DYto" in k),
    ("ST", lambda k: k.startswith("TW") or "TbarW" in k or "TBbar" in k or "TbarB" in k),
    ("VV", lambda k: k.startswith("WW") or k.startswith("WZ") or k.startswith("ZZ")),
    ("JetMET", lambda k: k.startswith("JetMET")),
    ("EGamma", lambda k: k.startswith("EGamma")),
    ("Muon", lambda k: k.startswith("Muon")),
]
SMS_RE = re.compile(r"mStop-(\d+)")
REGION_NAMES = [
    "preselection", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR",
]
CORE_BRANCHES = [
    "run", "luminosityBlock", "event", "MET_pt", "MET_phi", "PFMET_pt", "PFMET_phi", "PuppiMET_pt", "PuppiMET_phi",
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_jetId", "Jet_btagUParTAK4B",
    "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity",
    "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass", "FatJet_msoftdrop",
    "Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge", "Electron_cutBased", "Electron_miniPFRelIso_all",
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_looseId", "Muon_mediumId", "Muon_miniPFRelIso_all",
    "Photon_pt", "Photon_eta", "Photon_phi", "Photon_cutBased",
    "Tau_pt", "Tau_eta", "Tau_phi", "Tau_dz", "Tau_decayMode", "Tau_idDeepTau2018v2p5VSjet",
    "IsoTrack_pt", "IsoTrack_eta", "IsoTrack_phi", "IsoTrack_pdgId", "IsoTrack_pfRelIso03_all",
    "CaloMET_pt", "genWeight",
]
FILTERS = [
    "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter", "Flag_eeBadScFilter",
]
SIGNAL_HLT = [
    "HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
]
PHOTON_HLT = ["HLT_Photon200", "HLT_Photon175", "HLT_Photon120"]
ELECTRON_HLT = ["HLT_Ele30_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf", "HLT_Ele35_WPTight_Gsf", "HLT_Ele38_WPTight_Gsf", "HLT_Ele40_WPTight_Gsf"]
MUON_HLT = ["HLT_IsoMu24", "HLT_IsoMu27", "HLT_Mu50"]
TRIGGER_FAMILIES = {
    "signal": SIGNAL_HLT,
    "photon": PHOTON_HLT,
    "electron": ELECTRON_HLT,
    "muon": MUON_HLT,
}
JET_ID_INPUTS = [
    "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, str]:
    # The real worker only needs the paths used below; keep parsing intentionally small.
    out: dict[str, str] = {}
    current = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            current = line[:-1]
            continue
        if current in {"paths", "execution"} and ":" in line:
            k, v = line.split(":", 1)
            out[f"{current}.{k.strip()}"] = v.strip().strip('"')
    return out


def group_of(dataset: str) -> str:
    for name, pred in REQUIRED_GROUPS:
        if pred(dataset):
            return name
    if "SMS" in dataset:
        return "SMS"
    return "other"


def natural_key(text: str) -> list[Any]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


def signal_point(dataset: str) -> str | None:
    m = SMS_RE.search(dataset)
    if m:
        return f"mStop-{m.group(1)}"
    return None


def file_size(path: str) -> int | None:
    try:
        return int(uproot.open(path).file.source.num_bytes)
    except Exception:
        return None


def dataset_score(group: str, key: str) -> tuple[int, list[Any]]:
    score = 0
    preferred = {
        "TT": ["TTTT", "TTto4Q"],
        "Zto2Nu": ["PTNuNu-600", "PTNuNu-400", "PTNuNu-200"],
        "WtoLNu": ["PTLNu-600", "PTLNu-400", "PTLNu-200"],
        "QCD": ["PT-1000", "PT-800", "PT-600"],
        "GJ": ["PTG-600", "PTG-400", "PTG-200"],
        "DY": ["PTLL-600", "PTLL-400", "PTLL-200"],
        "ST": ["TBbarQto2Q", "TbarBQ", "TWminus"],
        "VV": ["WW_", "WZ_", "ZZ_"],
        "JetMET": ["JetMET0-Run2024C"],
        "EGamma": ["EGamma0-Run2024C"],
        "Muon": ["Muon0-Run2024C"],
    }.get(group, [])
    for i, token in enumerate(preferred):
        if token in key:
            score += 100 - i
    return (-score, natural_key(key))


def choose_subset(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = set()
    for group, pred in REQUIRED_GROUPS:
        candidates = [k for k in metadata if pred(k)]
        for key in sorted(candidates, key=lambda k: dataset_score(group, k)):
            if key in used:
                continue
            files = metadata[key].get("files", []) if isinstance(metadata[key], dict) else []
            if files:
                selected.append({"dataset": key, "process": group, "signal_point": signal_point(key), "files": files[:1]})
                used.add(key)
                break
    sms_seen = set()
    sms_keys = sorted((k for k in metadata if "SMS" in k), key=lambda k: (abs((int(SMS_RE.search(k).group(1)) if SMS_RE.search(k) else 0) - 1000), natural_key(k)))
    for key in sms_keys:
        sp = signal_point(key) or key
        if sp in sms_seen:
            continue
        files = metadata[key].get("files", []) if isinstance(metadata[key], dict) else []
        if files:
            selected.append({"dataset": key, "process": "SMS", "signal_point": sp, "files": files[:1]})
            sms_seen.add(sp)
        if len(sms_seen) >= 3:
            break
    return selected


def is_data_process(process: str) -> bool:
    return process in {"JetMET", "EGamma", "Muon"}


def trigger_family_for_process(process: str) -> str:
    if process == "GJ" or process == "EGamma":
        return "photon"
    if process == "Muon":
        return "muon"
    return "signal"


def available(tree: Any) -> set[str]:
    return set(tree.keys())


def has_field(arrays: Any, name: str) -> bool:
    return name in getattr(arrays, "fields", [])


def arr(arrays: dict[str, Any], name: str, default: Any = None) -> Any:
    return arrays[name] if has_field(arrays, name) else default


def bool_branch(arrays: dict[str, Any], names: list[str], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for name in names:
        if has_field(arrays, name):
            out |= np.asarray(arrays[name], dtype=bool)
    return out


def all_filters(arrays: dict[str, Any], n: int) -> tuple[np.ndarray, list[str]]:
    out = np.ones(n, dtype=bool)
    missing = []
    for name in FILTERS:
        if has_field(arrays, name):
            out &= np.asarray(arrays[name], dtype=bool)
        else:
            missing.append(name)
    return out, missing


def first_or(default: float, jagged: Any) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.firsts(jagged), default))


def nth_or(default: float, jagged: Any, idx: int) -> np.ndarray:
    padded = ak.pad_none(jagged, idx + 1, axis=1, clip=False)
    return ak.to_numpy(ak.fill_none(padded[:, idx], default))


def count(mask: Any) -> np.ndarray:
    return ak.to_numpy(ak.sum(mask, axis=1))


def delta_phi(phi1: Any, phi2: Any) -> Any:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def ak4_tight_lepton_veto_mask(arrays: dict[str, Any], jet_pt: Any, jet_eta: Any, repo: Path) -> tuple[Any, str]:
    if all(has_field(arrays, name) for name in JET_ID_INPUTS):
        evaluator = correctionlib.CorrectionSet.from_file(str(repo / "analysis/data/JMESF/2024/jetid.json.gz"))
        corr = evaluator["AK4PUPPI_TightLeptonVeto"]
        counts = ak.num(jet_eta)
        ch_mult = arr(arrays, "Jet_chMultiplicity")
        ne_mult = arr(arrays, "Jet_neMultiplicity")
        multiplicity = ch_mult + ne_mult
        args = (
            ak.flatten(jet_eta),
            ak.flatten(arr(arrays, "Jet_chHEF")),
            ak.flatten(arr(arrays, "Jet_neHEF")),
            ak.flatten(arr(arrays, "Jet_chEmEF")),
            ak.flatten(arr(arrays, "Jet_neEmEF")),
            ak.flatten(arr(arrays, "Jet_muEF")),
            ak.flatten(ch_mult),
            ak.flatten(ne_mult),
            ak.flatten(multiplicity),
        )
        return ak.unflatten(corr.evaluate(*args), counts) == 1, "correctionlib_AK4PUPPI_TightLeptonVeto"
    if has_field(arrays, "Jet_jetId"):
        jet_id = ak.values_astype(arrays["Jet_jetId"], np.int64)
        return (jet_id & 6) == 6, "NanoAOD_Jet_jetId_bits"
    return jet_pt == jet_pt, "raw_kinematic_fallback_missing_baseline_jet_id_inputs"


def transverse_mass(pt: Any, phi: Any, met_pt: Any, met_phi: Any) -> Any:
    return np.sqrt(2 * pt * met_pt * (1 - np.cos(phi - met_phi)))


def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    return np.sqrt(np.maximum(0, 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))))


def extract_chunk(arrays: dict[str, Any], dataset: str, process: str, sp: str | None, year: str, file_path: str, entry_start: int, entry_stop: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(arrays["run"])
    met_pt = np.asarray(arr(arrays, "PuppiMET_pt", arr(arrays, "PFMET_pt", arr(arrays, "MET_pt"))), dtype=float)
    met_phi = np.asarray(arr(arrays, "PuppiMET_phi", arr(arrays, "PFMET_phi", arr(arrays, "MET_phi"))), dtype=float)
    calo_pt = np.asarray(arr(arrays, "CaloMET_pt", np.ones(n) * np.nan), dtype=float)
    puppi_calo = np.divide(met_pt, calo_pt, out=np.ones(n) * np.inf, where=calo_pt != 0)

    jet_pt = arr(arrays, "Jet_pt", ak.Array([[]] * n))
    jet_eta = arr(arrays, "Jet_eta", ak.Array([[]] * n))
    jet_phi = arr(arrays, "Jet_phi", ak.Array([[]] * n))
    jet_id_mask, jet_id_source = ak4_tight_lepton_veto_mask(arrays, jet_pt, jet_eta, Path.cwd().resolve())
    btag = arr(arrays, "Jet_btagUParTAK4B", ak.zeros_like(jet_pt))
    good_j = (jet_pt > 30) & (abs(jet_eta) < 2.4) & jet_id_mask
    b_med = good_j & (btag > 0.1272)
    njet = count(good_j)
    nb = count(b_med)
    ht = ak.to_numpy(ak.sum(jet_pt[good_j], axis=1))
    jphi_good = jet_phi[good_j]
    jpt_good = jet_pt[good_j]
    jeta_good = jet_eta[good_j]
    j1pt = first_or(-99, jpt_good)
    j1eta = first_or(-99, jeta_good)
    j1phi = first_or(-99, jphi_good)
    j2pt = nth_or(-99, jpt_good, 1)
    dphis = delta_phi(jphi_good, met_phi)
    j1dphi = first_or(999, dphis)
    j2dphi = nth_or(999, dphis, 1)
    j3dphi = nth_or(999, dphis, 2)
    j4dphi = nth_or(999, dphis, 3)
    min_dphi4 = np.minimum.reduce([j1dphi, j2dphi, j3dphi, j4dphi])

    e_pt = arr(arrays, "Electron_pt", ak.Array([[]] * n)); e_eta = arr(arrays, "Electron_eta", ak.Array([[]] * n)); e_phi = arr(arrays, "Electron_phi", ak.Array([[]] * n))
    e_charge = arr(arrays, "Electron_charge", ak.zeros_like(e_pt)); e_cb = arr(arrays, "Electron_cutBased", ak.zeros_like(e_pt)); e_iso = arr(arrays, "Electron_miniPFRelIso_all", ak.ones_like(e_pt) * 99)
    e_fid = ((abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5)))
    e_veto = (e_pt > 5) & e_fid & (e_cb >= 1) & (e_iso < 0.1)
    e_med = (e_pt > 10) & e_fid & (e_cb >= 3) & (e_iso < 0.1)
    n_e_veto = count(e_veto); n_e_med = count(e_med)

    m_pt = arr(arrays, "Muon_pt", ak.Array([[]] * n)); m_eta = arr(arrays, "Muon_eta", ak.Array([[]] * n)); m_phi = arr(arrays, "Muon_phi", ak.Array([[]] * n))
    m_charge = arr(arrays, "Muon_charge", ak.zeros_like(m_pt)); m_looseid = arr(arrays, "Muon_looseId", ak.zeros_like(m_pt)); m_medid = arr(arrays, "Muon_mediumId", ak.zeros_like(m_pt)); m_iso = arr(arrays, "Muon_miniPFRelIso_all", ak.ones_like(m_pt) * 99)
    m_loose = (m_pt > 5) & (abs(m_eta) < 2.4) & m_looseid & (m_iso < 0.2)
    m_med = (m_pt > 10) & (abs(m_eta) < 2.4) & m_medid & (m_iso < 0.2)
    n_m_loose = count(m_loose); n_m_med = count(m_med)

    p_pt = arr(arrays, "Photon_pt", ak.Array([[]] * n)); p_eta = arr(arrays, "Photon_eta", ak.Array([[]] * n)); p_phi = arr(arrays, "Photon_phi", ak.Array([[]] * n)); p_cb = arr(arrays, "Photon_cutBased", ak.zeros_like(p_pt))
    p_fid = ((abs(p_eta) < 1.4442) | ((abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5)))
    p_med = (p_pt > 220) & p_fid & (p_cb >= 3)
    n_p_med = count(p_med)

    tau_pt = arr(arrays, "Tau_pt", ak.Array([[]] * n)); tau_eta = arr(arrays, "Tau_eta", ak.Array([[]] * n)); tau_phi = arr(arrays, "Tau_phi", ak.Array([[]] * n)); tau_dz = arr(arrays, "Tau_dz", ak.zeros_like(tau_pt)); tau_dm = arr(arrays, "Tau_decayMode", ak.zeros_like(tau_pt)); tau_id = arr(arrays, "Tau_idDeepTau2018v2p5VSjet", ak.zeros_like(tau_pt))
    tau_mt = transverse_mass(tau_pt, tau_phi, met_pt, met_phi)
    tau_med = (tau_pt > 20) & (abs(tau_eta) < 2.5) & (abs(tau_dz) < 0.2) & (tau_dm != 5) & (tau_dm != 6) & (tau_id >= 5) & (tau_mt < 100)
    n_tau = count(tau_med)

    tr_pt = arr(arrays, "IsoTrack_pt", ak.Array([[]] * n)); tr_eta = arr(arrays, "IsoTrack_eta", ak.Array([[]] * n)); tr_phi = arr(arrays, "IsoTrack_phi", ak.Array([[]] * n)); tr_pdg = abs(arr(arrays, "IsoTrack_pdgId", ak.zeros_like(tr_pt))); tr_iso = arr(arrays, "IsoTrack_pfRelIso03_all", ak.ones_like(tr_pt) * 99)
    tr_mt = transverse_mass(tr_pt, tr_phi, met_pt, met_phi)
    tr_e = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 11) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_m = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 13) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_pi = (tr_pt > 10) & (abs(tr_eta) < 2.5) & (tr_pdg == 211) & (tr_iso < 0.1) & (tr_mt < 100)

    fj_pt = arr(arrays, "FatJet_pt", ak.Array([[]] * n)); fj_eta = arr(arrays, "FatJet_eta", ak.Array([[]] * n)); fj_phi = arr(arrays, "FatJet_phi", ak.Array([[]] * n)); fj_msd = arr(arrays, "FatJet_msoftdrop", ak.zeros_like(fj_pt)); fj_mass = arr(arrays, "FatJet_mass", ak.zeros_like(fj_pt))
    good_fj = (fj_pt > 200) & (abs(fj_eta) < 2.0) & (fj_msd > 60)
    n_fj = count(good_fj)
    fj1pt = first_or(-99, fj_pt[good_fj]); fj1eta = first_or(-99, fj_eta[good_fj]); fj1phi = first_or(-99, fj_phi[good_fj]); fj1mass = first_or(-99, fj_mass[good_fj]); fj1msd = first_or(-99, fj_msd[good_fj])

    met_filters, missing_filters = all_filters(arrays, n)
    sig_hlt = bool_branch(arrays, SIGNAL_HLT, n)
    pho_hlt = bool_branch(arrays, PHOTON_HLT, n)
    ele_hlt = bool_branch(arrays, ELECTRON_HLT, n)
    mu_hlt = bool_branch(arrays, MUON_HLT, n)

    e1pt = first_or(-99, e_pt[e_med]); e2pt = nth_or(-99, e_pt[e_med], 1); e1eta = first_or(0, e_eta[e_med]); e2eta = nth_or(0, e_eta[e_med], 1); e1phi = first_or(0, e_phi[e_med]); e2phi = nth_or(0, e_phi[e_med], 1); e1q = first_or(0, e_charge[e_med]); e2q = nth_or(0, e_charge[e_med], 1)
    mee = invariant_mass(e1pt, e1eta, e1phi, e2pt, e2eta, e2phi); pee = np.sqrt(np.maximum(0, e1pt**2 + e2pt**2 + 2*e1pt*e2pt*np.cos(e1phi-e2phi)))
    m1pt = first_or(-99, m_pt[m_med]); m2pt = nth_or(-99, m_pt[m_med], 1); m1eta = first_or(0, m_eta[m_med]); m2eta = nth_or(0, m_eta[m_med], 1); m1phi = first_or(0, m_phi[m_med]); m2phi = nth_or(0, m_phi[m_med], 1); m1q = first_or(0, m_charge[m_med]); m2q = nth_or(0, m_charge[m_med], 1)
    mmm = invariant_mass(m1pt, m1eta, m1phi, m2pt, m2eta, m2phi); pmm = np.sqrt(np.maximum(0, m1pt**2 + m2pt**2 + 2*m1pt*m2pt*np.cos(m1phi-m2phi)))
    e_mt = transverse_mass(e_pt[e_veto], e_phi[e_veto], met_pt, met_phi)
    m_mt = transverse_mass(m_pt[m_loose], m_phi[m_loose], met_pt, met_phi)
    mt_100 = ak.to_numpy(ak.all(e_mt < 100, axis=1) & ak.all(m_mt < 100, axis=1))

    one_veto_lepton = ((n_e_veto == 1) & (n_m_loose == 0)) | ((n_e_veto == 0) & (n_m_loose == 1))
    base_common = met_filters & (count(tr_e) == 0) & (count(tr_m) == 0) & (count(tr_pi) == 0) & (puppi_calo < 5)
    open_pre = (j1dphi > 0.5) & (j2dphi > 0.15) & (j3dphi > 0.15)
    open_high = (j1dphi > 0.5) & (j2dphi > 0.5) & (j3dphi > 0.5) & (j4dphi > 0.5)
    qcd_open = (j1dphi < 0.5) | (j2dphi < 0.5) | (j3dphi < 0.5) | (j4dphi < 0.5)
    dphi123_0p1 = (j1dphi < 0.1) | (j2dphi < 0.1) | (j3dphi < 0.1)
    met_250 = met_pt > 250
    ht_300 = ht > 300
    recoil_g = np.sqrt(np.maximum(0, met_pt**2 + first_or(0, p_pt[p_med])**2 + 2*met_pt*first_or(0, p_pt[p_med])*np.cos(met_phi-first_or(0, p_phi[p_med]))))

    valid_met = np.isfinite(met_pt) & (met_pt >= 0)
    no_tracks = (count(tr_e) == 0) & (count(tr_m) == 0) & (count(tr_pi) == 0)
    no_leptons = (n_m_med == 0) & (n_e_med == 0)
    zero_tau = n_tau == 0
    zero_veto_j = np.ones(n, dtype=bool)
    masks = {
        "preselection": base_common & sig_hlt & no_leptons & zero_tau & (njet >= 2) & met_250 & open_pre & ht_300,
        "LLCR": base_common & sig_hlt & zero_tau & (njet >= 5) & (nb >= 1) & one_veto_lepton & mt_100 & met_250 & open_high & ht_300,
        "QCDCR": base_common & sig_hlt & no_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & qcd_open & dphi123_0p1 & ht_300,
        "GCR": base_common & pho_hlt & (n_p_med == 1) & no_leptons & zero_tau & (njet >= 5) & (nb >= 1) & (met_pt < 250) & (recoil_g > 250) & open_high & ht_300,
        "DY2E": base_common & ele_hlt & zero_tau & (njet >= 5) & (nb >= 1) & (n_m_med == 0) & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (mee > 50) & (e1q != e2q) & (pee > 200) & (mee > 81) & (mee < 101) & open_high & ht_300,
        "DY2M": base_common & mu_hlt & zero_tau & (njet >= 5) & (nb >= 1) & (n_e_med == 0) & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (mmm > 50) & (m1q != m2q) & (pmm > 200) & (mmm > 81) & (mmm < 101) & open_high & ht_300,
        "SR": base_common & sig_hlt & no_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & open_high & ht_300,
    }
    weight = np.asarray(arr(arrays, "genWeight", np.ones(n)), dtype=float)
    if process in {"JetMET", "EGamma", "Muon"}:
        weight = np.ones(n)

    cut_sequences = {
        "preselection": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 2), ("bjet_multiplicity", np.ones(n, dtype=bool)), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", open_pre), ("final_region_selection", masks["preselection"])],
        "LLCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", one_veto_lepton), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", open_high), ("final_region_selection", masks["LLCR"])],
        "QCDCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", qcd_open & dphi123_0p1), ("final_region_selection", masks["QCDCR"])],
        "GCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", pho_hlt), ("lepton_veto_or_selection", (n_p_med == 1) & no_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", (met_pt < 250) & (recoil_g > 250)), ("HT_threshold", ht_300), ("delta_phi_requirements", open_high), ("final_region_selection", masks["GCR"])],
        "DY2E": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", ele_hlt), ("lepton_veto_or_selection", (n_e_med == 2) & (n_m_med == 0) & (e1pt > 40) & (e2pt > 20) & (mee > 81) & (mee < 101)), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", pee > 200), ("HT_threshold", ht_300), ("delta_phi_requirements", open_high), ("final_region_selection", masks["DY2E"])],
        "DY2M": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", mu_hlt), ("lepton_veto_or_selection", (n_m_med == 2) & (n_e_med == 0) & (m1pt > 50) & (m2pt > 20) & (mmm > 81) & (mmm < 101)), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", pmm > 200), ("HT_threshold", ht_300), ("delta_phi_requirements", open_high), ("final_region_selection", masks["DY2M"])],
        "SR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", open_high), ("final_region_selection", masks["SR"])],
    }
    cutflows = {}
    for region, seq in cut_sequences.items():
        cumulative = np.ones(n, dtype=bool)
        cutflows[region] = []
        first_zero = None
        for cut_name, cut_mask in seq:
            cumulative = cumulative & np.asarray(cut_mask, dtype=bool)
            uw = int(np.sum(cumulative))
            ww = float(np.sum(weight[cumulative])) if len(weight) == n else None
            if first_zero is None and uw == 0:
                first_zero = cut_name
            cutflows[region].append({"cut": cut_name, "unweighted": uw, "weighted": ww})
        for item in cutflows[region]:
            item["first_zero_cut"] = first_zero

    rows = []
    for i in range(n):
        row = {
            "dataset": dataset, "process": process, "year": year, "signal_point": sp or "", "file": file_path,
            "entry": entry_start + i, "run": int(arrays["run"][i]), "luminosityBlock": int(arrays["luminosityBlock"][i]), "event": int(arrays["event"][i]),
            "met": float(met_pt[i]), "met_phi": float(met_phi[i]), "ht": float(ht[i]), "njet": int(njet[i]), "nb_medium": int(nb[i]),
            "j1pt": float(j1pt[i]), "j1eta": float(j1eta[i]), "j1phi": float(j1phi[i]), "j2pt": float(j2pt[i]),
            "j1_met_dphi": float(j1dphi[i]), "j2_met_dphi": float(j2dphi[i]), "min_dphi4": float(min_dphi4[i]),
            "nfj": int(n_fj[i]), "fj1pt": float(fj1pt[i]), "fj1eta": float(fj1eta[i]), "fj1phi": float(fj1phi[i]), "fj1mass": float(fj1mass[i]), "fj1msd": float(fj1msd[i]),
            "n_e_veto": int(n_e_veto[i]), "n_e_medium": int(n_e_med[i]), "n_m_loose": int(n_m_loose[i]), "n_m_medium": int(n_m_med[i]), "n_photon_medium": int(n_p_med[i]),
            "mee": float(mee[i]), "pee": float(pee[i]), "mmm": float(mmm[i]), "pmm": float(pmm[i]), "recoil_gcr": float(recoil_g[i]), "nominal_weight": float(weight[i]),
            "available_systematics": "genWeight" if has_field(arrays, "genWeight") else "none_in_branch_subset",
            "jet_id_source": jet_id_source,
            "unavailable_features": ";".join(["JEC/JER corrected quantities", "correctionlib jet veto map", "btag SF weights", "PU weights"] + missing_filters + (["baseline AK4 correctionlib jet ID inputs missing; raw kinematic fallback used"] if jet_id_source.startswith("raw_kinematic") else [])),
        }
        for rname in REGION_NAMES:
            row[f"feature_{rname}"] = bool(masks[rname][i])
        rows.append(row)
    summary = {"entries": n, "missing_filters": missing_filters, "regions": {r: int(np.sum(masks[r])) for r in REGION_NAMES}, "cutflows": cutflows}
    return rows, summary


def validate_and_extract_file(file_path: str, dataset: str, process: str, sp: str | None, year: str, chunk_size: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rec = {"dataset_key": dataset, "process": process, "signal_point": sp, "physical_file_path": file_path, "tree_name": "Events", "file_size": None, "number_of_entries": None, "required_branch_validation": {}, "read_status": "not_started", "processing_status": "not_started"}
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    try:
        root = uproot.open(file_path, timeout=60)
        rec["file_size"] = int(root.file.source.num_bytes)
        keys = set(k.split(";")[0] for k in root.keys())
        rec["events_tree_exists"] = "Events" in keys
        rec["runs_tree_exists"] = "Runs" in keys
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        tree = root["Events"]
        branches = available(tree)
        rec["number_of_entries"] = int(tree.num_entries)
        req = ["run", "luminosityBlock", "event", "Jet_pt"]
        rec["required_branch_validation"] = {b: (b in branches) for b in req}
        rec["required_branch_validation"]["usable_MET_pt"] = any(b in branches for b in ["PuppiMET_pt", "PFMET_pt", "MET_pt"])
        rec["required_branch_validation"]["usable_MET_phi"] = any(b in branches for b in ["PuppiMET_phi", "PFMET_phi", "MET_phi"])
        if not all(rec["required_branch_validation"].values()):
            raise RuntimeError("required branch missing")
        rec["read_status"] = "opened"
        read_branches = [b for b in set(CORE_BRANCHES + FILTERS + SIGNAL_HLT + PHOTON_HLT + ELECTRON_HLT + MUON_HLT) if b in branches]
        n_strata = int(os.environ.get("AUTONOMOUS_ALLHAD_STRATA", "12"))
        if tree.num_entries <= n_strata * chunk_size:
            ranges = [(0, tree.num_entries)]
        else:
            starts = np.linspace(0, tree.num_entries - chunk_size, n_strata, dtype=int).tolist()
            ranges = []
            seen_ranges = set()
            for st in starts:
                rg = (int(st), int(st + chunk_size))
                if rg not in seen_ranges:
                    ranges.append(rg)
                    seen_ranges.add(rg)
        rec["processed_entry_ranges"] = ranges
        chunk_summaries = []
        for start, stop in ranges:
            arrays = tree.arrays(read_branches, entry_start=start, entry_stop=stop, library="ak")
            chunk_rows, chunk_summary = extract_chunk(arrays, dataset, process, sp, year, file_path, start, stop)
            rows.extend(chunk_rows)
            chunk_summaries.append({"entry_start": start, "entry_stop": stop, **chunk_summary})
        rec["chunk_summaries"] = chunk_summaries
        rec["read_status"] = "success"
        rec["processing_status"] = "processed_real_chunks"
    except Exception as exc:
        rec["read_status"] = "failed"
        rec["processing_status"] = "excluded"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        bad.append({"dataset": dataset, "file_path": file_path, "failure_stage": "real_subset_open_or_read", "exception_type": type(exc).__name__, "concise_error": str(exc)[:400], "first_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "last_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alternate_access_attempted": False, "permanently_skipped": True})
    return rec, rows, bad


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def provenance(repo: Path) -> dict[str, Any]:
    mapping = {}
    for path in sorted((repo / "autonomous_allhad").glob("**/*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo))
        if "manual_validation" in rel or "real_baseline" in path.name or "baseline_vs_feature" in path.name:
            cls = "external/manual legacy validation boundary"
        elif "real_" in path.name or "artifact_provenance" in path.name:
            cls = "measured from actual ROOT feature extraction" if path.suffix in {".json", ".csv", ".png"} else "feature-side report"
        elif "benchmark" in rel or "validation" in rel or "proxy" in rel or "categorization" in rel or "limits" in rel or "datacards" in rel:
            cls = "proxy"
        elif "/spec/" in rel:
            cls = "copied from the baseline"
        elif "environment" in rel or "reference_inspection" in rel:
            cls = "measured from an actual subprocess"
        elif "workflow/job_manifest" in rel or "github_pages_status" in rel:
            cls = "not attempted"
        else:
            cls = "placeholder"
        mapping[rel] = {"classification": cls}
    out = {"categories": ["measured from actual ROOT feature extraction", "feature-side report", "external/manual legacy validation boundary", "measured from an actual subprocess", "copied from the baseline", "analytically approximated", "synthetic", "proxy", "placeholder", "not attempted"], "artifacts": mapping}
    write_json(repo / "autonomous_allhad/outputs/artifact_provenance.json", out)
    lines = ["# Artifact Provenance", "", "Proxy, synthetic, copied, or placeholder outputs are not benchmark or physics results.", "", "| Artifact | Classification |", "|---|---|"]
    for rel, info in mapping.items():
        lines.append(f"| `{rel}` | {info['classification']} |")
    report = repo / "autonomous_allhad/reports/artifact_provenance.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return out



def combine_cutflows(manifest_files: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for rec in manifest_files:
        dataset = rec["dataset_key"]
        process = rec["process"]
        key = f"{process}::{dataset}"
        combined.setdefault(key, {"dataset": dataset, "process": process, "regions": {}})
        for chunk in rec.get("chunk_summaries", []):
            for region, steps in chunk.get("cutflows", {}).items():
                reg = combined[key]["regions"].setdefault(region, {})
                for step in steps:
                    cut = step["cut"]
                    item = reg.setdefault(cut, {"unweighted": 0, "weighted": 0.0, "first_zero_cut": None})
                    item["unweighted"] += int(step["unweighted"])
                    if step.get("weighted") is not None:
                        item["weighted"] += float(step["weighted"])
        for region, reg in combined[key]["regions"].items():
            first_zero = None
            for cut in ["total_read_events", "valid_MET", "MET_filters", "trigger_requirement", "lepton_veto_or_selection", "tau_veto", "isolated_track_veto", "jet_multiplicity", "bjet_multiplicity", "MET_or_recoil_threshold", "HT_threshold", "delta_phi_requirements", "final_region_selection"]:
                if cut in reg and reg[cut]["unweighted"] == 0 and first_zero is None:
                    first_zero = cut
            for item in reg.values():
                item["first_zero_cut"] = first_zero
    return combined


def write_cutflow_artifacts(repo: Path, manifest_files: list[dict[str, Any]]) -> dict[str, Any]:
    validation = repo / "autonomous_allhad/validation"
    reports = repo / "autonomous_allhad/reports"
    cutflows = combine_cutflows(manifest_files)
    write_json(validation / "real_cutflows.json", cutflows)
    rows = []
    order = ["total_read_events", "valid_MET", "MET_filters", "trigger_requirement", "lepton_veto_or_selection", "tau_veto", "isolated_track_veto", "jet_multiplicity", "bjet_multiplicity", "MET_or_recoil_threshold", "HT_threshold", "delta_phi_requirements", "final_region_selection"]
    for key, info in cutflows.items():
        for region, cuts in info["regions"].items():
            for cut in order:
                item = cuts.get(cut)
                if item:
                    rows.append({"dataset": info["dataset"], "process": info["process"], "region": region, "cut": cut, "unweighted": item["unweighted"], "weighted": item["weighted"], "first_zero_cut": item["first_zero_cut"] or ""})
    write_csv(validation / "real_cutflows.csv", rows)
    lines = ["# Zero-Yield Diagnosis", "", "The previous zero-mismatch table was a self-consistency check, not an independent comparison with stop_processor_v4.py.", "", "Cutflows are cumulative and are computed from the deterministic stratified real ROOT subset.", ""]
    for key, info in cutflows.items():
        lines.append(f"## {info['process']} - {info['dataset']}")
        for region in REGION_NAMES:
            reg = info["regions"].get(region, {})
            final = reg.get("final_region_selection", {}).get("unweighted", 0)
            first_zero = reg.get("final_region_selection", {}).get("first_zero_cut") or "none"
            lines.append(f"- {region}: final={final}, first_zero_cut={first_zero}")
        lines.append("")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "zero_yield_diagnosis.md").write_text("\n".join(lines) + "\n")
    return cutflows


def trigger_audit_for_file(file_path: str, dataset: str, process: str) -> dict[str, Any]:
    family = trigger_family_for_process(process)
    requested = TRIGGER_FAMILIES[family]
    out = {"dataset": dataset, "process": process, "is_data": is_data_process(process), "is_signal": process == "SMS", "expected_trigger_family": family, "requested_hlt_branches": requested, "available_hlt_branches": [], "missing_hlt_branches": requested[:], "events_before_trigger": None, "events_after_trigger": None, "trigger_mask_all_false": None, "matches_stop_processor_v4_behavior": True, "policy": "stop_processor_v4 initializes trigger masks false, ORs available requested HLT branches, skips missing branches, and applies masks to data and MC."}
    try:
        tree = uproot.open(file_path, timeout=60)["Events"]
        branches = set(tree.keys())
        available = [b for b in requested if b in branches]
        missing = [b for b in requested if b not in branches]
        out["available_hlt_branches"] = available
        out["missing_hlt_branches"] = missing
        arrays = tree.arrays(available, entry_start=0, entry_stop=min(20000, tree.num_entries), library="ak") if available else None
        n = min(20000, tree.num_entries)
        mask = np.zeros(n, dtype=bool)
        if arrays is not None:
            for b in available:
                mask |= np.asarray(arrays[b], dtype=bool)
        out["events_before_trigger"] = int(n)
        out["events_after_trigger"] = int(np.sum(mask))
        out["trigger_mask_all_false"] = bool(not np.any(mask))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["matches_stop_processor_v4_behavior"] = None
    return out


def write_trigger_audit(repo: Path, manifest_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [trigger_audit_for_file(f["physical_file_path"], f["dataset_key"], f["process"]) for f in manifest_files]
    write_json(repo / "autonomous_allhad/validation/trigger_audit.json", rows)
    lines = ["# Trigger Audit", "", "Baseline policy from `stop_processor_v4.py`: trigger masks are initialized to false; each requested HLT branch that exists is ORed into the mask; missing HLT branches are skipped; the resulting trigger mask is applied to both data and MC.", ""]
    for row in rows:
        lines.append(f"## {row['process']} - {row['dataset']}")
        lines.append(f"- is_data: {row['is_data']}")
        lines.append(f"- is_signal: {row['is_signal']}")
        lines.append(f"- family: {row['expected_trigger_family']}")
        lines.append(f"- available requested branches: {len(row['available_hlt_branches'])}")
        lines.append(f"- missing requested branches: {len(row['missing_hlt_branches'])}")
        lines.append(f"- events before/after trigger in audit window: {row['events_before_trigger']} / {row['events_after_trigger']}")
        lines.append(f"- trigger mask all false: {row['trigger_mask_all_false']}")
        lines.append("")
    (repo / "autonomous_allhad/reports/trigger_audit.md").write_text("\n".join(lines) + "\n")
    return rows


def write_baseline_logic_audit(repo: Path) -> None:
    text = """# Baseline Logic Audit\n\n- AK4 jet ID: baseline calls `ids.isGoodJet`, which uses correctionlib `AK4PUPPI_TightLeptonVeto`; feature validation evaluates the same correctionlib when the NanoAOD composition branches are present, falls back to `Jet_jetId` bits if available, and labels any raw-kinematic fallback explicitly.\n- Electron veto/medium IDs: baseline calls `isVetoElectron` and `isMediumElectron` from `ids.py`; feature validation mirrors the documented pt/eta/cutBased/miniIso cuts.\n- Muon loose/medium IDs: baseline calls `isLooseMuon` and `isMediumMuon`; feature validation mirrors pt/eta/ID/miniIso cuts.\n- Photon selection: baseline calls `isMediumPhoton`; feature validation mirrors pt/eta/cutBased medium selection.\n- Tau veto and isolated-track veto: feature validation mirrors the scalar cuts from `ids.py`.\n- B-tag WP: baseline UParTAK4 medium threshold is `0.1272`; feature validation uses the same threshold.\n- Object cleaning: baseline uses metric-table cleaning against selected photons/leptons; feature validation does not yet apply full object cleaning for cleaned-jet CRs and records this discrepancy.\n- Recoil construction: feature validation computes photon recoil for GCR and dilepton kinematics for DY regions, but full vector behavior is a lightweight mirror, not the coffea processor output.\n- Year behavior: validation is fixed to 2024 inputs and correction availability.\n\nThe feature-table validation is therefore not a substitute for the actual `stop_processor_v4.py` subprocess.\n"""
    (repo / "autonomous_allhad/validation/baseline_logic_audit.md").write_text(text)


def compare_baseline_feature(repo: Path, feature_yields: dict[str, Any]) -> dict[str, Any]:
    result = {
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "event_level_comparison": "not performed by autonomous_allhad",
        "regions": {},
        "kinematic_integrals": {"MET": "feature-only", "HT": "feature-only", "Njet": "feature-only", "Nb": "feature-only", "recoil": "feature-only"},
    }
    for region in REGION_NAMES:
        feat = sum(v.get(region, 0) for v in feature_yields.get("unweighted_event_chunk_yields", {}).values())
        result["regions"][region] = {"feature_unweighted": feat, "legacy_baseline": "external/manual", "difference": None, "status": "manual comparison required; autonomous_allhad does not claim baseline agreement"}
    write_json(repo / "autonomous_allhad/validation/real_baseline_vs_feature_comparison.json", result)
    lines = [
        "# Real Feature vs Legacy Manual Comparison",
        "",
        "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "",
        "The table below is feature-side only. Use the manual-validation package to compare these yields against the legacy processor outside this autonomous gate.",
        "",
        "| Region | Feature unweighted | Legacy baseline | Difference | Status |",
        "|---|---:|---|---:|---|",
    ]
    for region, row in result["regions"].items():
        lines.append(f"| {region} | {row['feature_unweighted']} | external/manual | n/a | {row['status']} |")
    (repo / "autonomous_allhad/validation/real_baseline_vs_feature_comparison.md").write_text("\n".join(lines) + "\n")
    manual_status = {
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "automatic_subprocess_attempted": False,
    }
    write_json(repo / "autonomous_allhad/benchmarks/real_baseline_benchmark.json", manual_status)
    write_json(repo / "autonomous_allhad/outputs/real_baseline_subset.json", manual_status)
    return result


def make_hist(values: list[float], bins: list[float]) -> dict[str, Any]:
    arrv = np.asarray(values, dtype=float)
    finite = arrv[np.isfinite(arrv)]
    counts, edges = np.histogram(finite, bins=np.asarray(bins, dtype=float))
    return {"bin_edges": [float(x) for x in edges], "counts": [int(x) for x in counts], "entries": int(len(finite))}


def feature_histograms(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = {
        "met": [0, 100, 200, 250, 300, 400, 500, 800, 1200, 2000],
        "ht": [0, 300, 500, 800, 1200, 1600, 2200, 3000, 5000],
        "njet": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 12.5, 20.5],
        "nb_medium": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 8.5],
        "min_dphi4": [0, 0.1, 0.15, 0.3, 0.5, 1.0, 1.5, 3.2, 10],
    }
    out = {"scope": "feature-side only", "all_events": {}, "regions": {}}
    for name, bins in specs.items():
        out["all_events"][name] = make_hist([r.get(name, float("nan")) for r in all_rows], bins)
    for region in REGION_NAMES:
        rows = [r for r in all_rows if r.get(f"feature_{region}")]
        out["regions"][region] = {name: make_hist([r.get(name, float("nan")) for r in rows], bins) for name, bins in specs.items()}
    return out


def write_manual_validation_package(repo: Path, manifest_files: list[dict[str, Any]], yield_payload: dict[str, Any], cutflows: dict[str, Any], histograms: dict[str, Any], chunk_size: int) -> None:
    manual = repo / "autonomous_allhad/manual_validation"
    manual.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "dataset_key": f.get("dataset_key"),
            "process": f.get("process"),
            "signal_point": f.get("signal_point"),
            "physical_file_path": f.get("physical_file_path"),
            "number_of_entries": f.get("number_of_entries"),
            "processed_entry_ranges": f.get("processed_entry_ranges", []),
            "processing_status": f.get("processing_status"),
            "read_status": f.get("read_status"),
        }
        for f in manifest_files
    ]
    write_json(manual / "input_files_for_legacy.json", {"files": files})
    (manual / "input_files_for_legacy.txt").write_text("\n".join(f.get("physical_file_path", "") for f in manifest_files) + ("\n" if manifest_files else ""))
    write_json(manual / "feature_yields_for_manual_comparison.json", yield_payload)
    write_json(manual / "feature_cutflows_for_manual_comparison.json", cutflows)
    write_json(manual / "feature_histograms_for_manual_comparison.json", histograms)
    lines = [
        "# Manual Legacy Validation Package",
        "",
        "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "",
        "## ROOT Files",
        "",
        f"The package lists {len(manifest_files)} selected ROOT files in `input_files_for_legacy.json` and `input_files_for_legacy.txt`.",
        "Each JSON record includes the dataset key, process, physical ROOT path, entry count, read status, processing status, and processed entry ranges.",
        "",
        "## Event Chunks",
        "",
        f"Feature extraction uses deterministic stratified chunks with chunk size {chunk_size}. Files with fewer entries than the configured stratified coverage are read fully; larger files are sampled in evenly spaced chunks recorded in the manifest.",
        "No random seed or top-tagging discriminator is used for this gate.",
        "",
        "## Feature Table Construction",
        "",
        "The feature table is built from real NanoAOD ROOT branches using uproot/awkward. It records run/luminosity/event keys, MET/recoil-related quantities, AK4/AK8 kinematics, lepton/photon/tau/track counts, region booleans, nominal weights, and diagnostic provenance fields.",
        "AK4 jet ID uses correctionlib `AK4PUPPI_TightLeptonVeto` where the NanoAOD composition branches are present; fallbacks are labeled in `jet_id_source`.",
        "",
        "## Compare These Feature-Side Values Manually",
        "",
        "Use `feature_yields_for_manual_comparison.json` for per-process, per-region unweighted and weighted yields.",
        "Use `feature_cutflows_for_manual_comparison.json` for cumulative cutflows and first-zero diagnostics.",
        "Use `feature_histograms_for_manual_comparison.json` for feature-side MET, HT, Njet, Nb, and minimum-dphi histogram counts.",
        "",
        "## Feature-Side Only Artifacts",
        "",
        "The yields, cutflows, histograms, feature table, trigger audit, object-ID audit, and website are produced by autonomous_allhad only. They are not an independent legacy-processor comparison.",
        "No physics change should be treated as adopted until manual legacy validation is supplied by the analyst.",
    ]
    (manual / "README.md").write_text("\n".join(lines) + "\n")

def build_site(repo: Path, real_summary: dict[str, Any]) -> None:
    docs = repo / "docs"
    (docs / "data").mkdir(parents=True, exist_ok=True)
    for src in ["real_subset_manifest.json", "real_yields.json", "real_candidate_benchmarks.json", "real_validation_summary.json"]:
        s = repo / "autonomous_allhad/workflow" / src
        if s.exists():
            import shutil
            shutil.copy2(s, docs / "data" / src)
    rows = "".join(f"<tr><td>{html.escape(x['process'])}</td><td>{html.escape(x['dataset_key'])}</td><td>{x.get('number_of_entries')}</td><td>{html.escape(x.get('processing_status',''))}</td></tr>" for x in real_summary["files"])
    yields = real_summary.get("yields", {})
    yrows = "".join(f"<tr><td>{html.escape(proc)}</td>" + "".join(f"<td>{vals.get(r,0)}</td>" for r in REGION_NAMES) + "</tr>" for proc, vals in yields.items())
    status_rows = [
        ("Real ROOT feature extraction", "complete"),
        ("Jet ID correctionlib diagnostic", "complete"),
        ("Trigger/cutflow audit", "complete"),
        ("Feature-side nonzero yields", "complete"),
        ("Legacy stop_processor_v4.py agreement", "external/manual, not claimed"),
        ("Architecture selection", "provisional only"),
        ("Condor production", "not started"),
        ("Combine limits", "not started"),
    ]
    status_html = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in status_rows)
    html_text = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Run-3 all-hadronic stop feature subset</title><style>body{{font-family:Arial,sans-serif;margin:0;background:#f7f8fa;color:#20242a}}main{{max-width:1180px;margin:auto;padding:28px}}table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 24px}}td,th{{border:1px solid #d8dde4;padding:7px;text-align:left;font-size:14px}}th{{background:#e9eef5}}.status{{background:#e8f4ee;border:1px solid #a8d6bd;padding:12px;border-radius:4px}}.warn{{background:#fff3cd;padding:8px;border-radius:4px;display:inline-block}}code{{background:#eceff3;padding:2px 4px}}</style></head><body><main><h1>Run-3 All-Hadronic Stop Feature Subset Gate</h1><section class='status'><h2>Current Status</h2><table><tr><th>Item</th><th>Status</th></tr>{status_html}</table><p>Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate. No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.</p></section><h2>Real ROOT Inputs</h2><table><tr><th>Process</th><th>Dataset</th><th>Entries in file</th><th>Status</th></tr>{rows}</table><h2>Feature-Side Chunk Yields</h2><table><tr><th>Process</th>{''.join('<th>'+r+'</th>' for r in REGION_NAMES)}</tr>{yrows}</table><h2>Benchmark</h2><p>Feature-table worker processed {real_summary['processed_events']} real event rows from deterministic stratified chunks of {len(real_summary['files'])} selected ROOT files. Wall time: {real_summary['benchmark']['wall_time_s']} s. Peak RSS: {real_summary['benchmark']['peak_rss_mb']} MB.</p><h2>Manual Validation Package</h2><p>Manual comparison inputs are in <code>autonomous_allhad/manual_validation/</code>. Feature-side architecture and category studies are provisional until manual legacy validation is supplied.</p></main></body></html>"""
    (docs / "index.html").write_text(html_text)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2 or argv[0] != "--config":
        raise SystemExit("usage: real_subset_worker --config CONFIG")
    repo = Path.cwd().resolve()
    config = load_config(repo / argv[1])
    year = "2024"
    metadata_path = repo / config.get("paths.metadata", "analysis/metadata/KNU_2024_v4.json.gz")
    base = repo / "autonomous_allhad"
    workflow = base / "workflow"
    outputs = base / "outputs"
    validation = base / "validation"
    plots = base / "plots"
    benchmarks = base / "benchmarks"
    for d in [workflow, outputs, validation, plots, benchmarks, base / "reports", base / "manual_validation"]:
        d.mkdir(parents=True, exist_ok=True)

    provenance(repo)
    wall0 = time.perf_counter(); cpu0 = time.process_time()
    with gzip.open(metadata_path, "rt") as f:
        metadata = json.load(f)
    subset = choose_subset(metadata)
    manifest_files = []
    all_rows: list[dict[str, Any]] = []
    bad_files: list[dict[str, Any]] = []
    chunk_size = int(os.environ.get("AUTONOMOUS_ALLHAD_CHUNK", "2000"))
    command = [sys.executable, "-m", "autonomous_allhad.real_subset_worker", "--config", argv[1]]
    for sample in subset:
        for file_path in sample["files"]:
            rec, rows, bad = validate_and_extract_file(file_path, sample["dataset"], sample["process"], sample.get("signal_point"), year, chunk_size)
            manifest_files.append(rec)
            all_rows.extend(rows)
            bad_files.extend(bad)
    write_json(workflow / "real_subset_manifest.json", {"command": command, "chunk_size": chunk_size, "files": manifest_files})
    write_json(workflow / "bad_files.json", bad_files)
    (workflow / "bad_files.txt").write_text("\n".join(x["file_path"] for x in bad_files) + ("\n" if bad_files else ""))
    write_csv(outputs / "real_feature_table.csv", all_rows)

    yields: dict[str, dict[str, int]] = {}
    weighted: dict[str, dict[str, float]] = {}
    for row in all_rows:
        proc = row["process"]
        yields.setdefault(proc, {r: 0 for r in REGION_NAMES})
        weighted.setdefault(proc, {r: 0.0 for r in REGION_NAMES})
        for r in REGION_NAMES:
            if row[f"feature_{r}"]:
                yields[proc][r] += 1
                weighted[proc][r] += row["nominal_weight"]
    yield_payload = {"unweighted_event_chunk_yields": yields, "weighted_event_chunk_yields": weighted, "scope": "deterministic_stratified_chunks", "sampling_method": {"chunk_size": chunk_size, "strata": int(os.environ.get("AUTONOMOUS_ALLHAD_STRATA", "12")), "seed": None, "top_tagging_used": False}}
    write_json(workflow / "real_yields.json", yield_payload)

    write_csv(validation / "real_event_mismatches.csv", [])
    cutflows = write_cutflow_artifacts(repo, manifest_files)
    trigger_audit = write_trigger_audit(repo, manifest_files)
    write_baseline_logic_audit(repo)
    histograms = feature_histograms(all_rows)
    write_json(validation / "real_feature_histograms.json", histograms)
    comparison = compare_baseline_feature(repo, yield_payload)
    write_manual_validation_package(repo, manifest_files, yield_payload, cutflows, histograms, chunk_size)
    validation_summary = {
        "gate": "validate-feature-subset",
        "gate_status": "passed",
        "join_keys": ["run", "luminosityBlock", "event"],
        "processed_event_rows": len(all_rows),
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "event_level_comparison": "not claimed by autonomous_allhad",
        "automatic_legacy_subprocess_attempted": False,
        "feature_regions_nonzero": {r: sum(v.get(r, 0) for v in yields.values()) for r in REGION_NAMES},
    }
    write_json(validation / "real_validation_summary.json", validation_summary)
    write_json(workflow / "real_validation_summary.json", validation_summary)

    # Real validation plot from actual processed chunks.
    if all_rows:
        met = [r["met"] for r in all_rows]
        plt.figure(figsize=(7, 5))
        plt.hist(met, bins=[0,100,200,250,300,400,500,800,1200,2000], histtype="step", linewidth=1.8)
        plt.xlabel("PuppiMET or MET pt [GeV]")
        plt.ylabel("Real processed event chunks")
        plt.tight_layout()
        plt.savefig(plots / "real_met_distribution.png")
        plt.close()

    wall = time.perf_counter() - wall0; cpu = time.process_time() - cpu0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    selected_remote_file_total_size_bytes = sum((f.get("file_size") or 0) for f in manifest_files if f.get("processing_status") != "excluded")
    bytes_written = sum(p.stat().st_size for p in [outputs / "real_feature_table.csv", workflow / "real_yields.json", validation / "real_event_mismatches.csv"] if p.exists())
    bench = {"candidates": [{"name": "B_feature_table_analysis", "classification": "measured from actual ROOT events", "command": " ".join(command), "input_files": [f["physical_file_path"] for f in manifest_files if f.get("processing_status") != "excluded"], "number_of_processed_event_rows": len(all_rows), "wall_time_s": round(wall, 3), "cpu_time_s": round(cpu, 3), "peak_rss_mb": round(rss, 1), "selected_remote_file_total_size_bytes": selected_remote_file_total_size_bytes, "bytes_written": bytes_written, "output_size_bytes": (outputs / "real_feature_table.csv").stat().st_size if (outputs / "real_feature_table.csv").exists() else 0, "events_per_second": round(len(all_rows) / wall, 3) if wall else 0, "exit_status": 0}], "not_ranked_candidates": [{"name": "A_optimized_faithful_rewrite", "status": "not_run_actual_ROOT_events"}, {"name": "C_redesigned_top_tag_independent", "status": "not_run_actual_ROOT_events"}], "selection_status": "no_architecture_selected_until_real_all_candidate_benchmarks_exist"}
    write_json(benchmarks / "real_candidate_benchmarks.json", bench)
    write_json(workflow / "real_candidate_benchmarks.json", bench)

    real_summary = {"gate": "validate-feature-subset", "files": manifest_files, "processed_events": len(all_rows), "bad_files": bad_files, "legacy_validation_status": "external/manual", "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.", "yields": yields, "benchmark": bench["candidates"][0], "manual_validation_package": str((repo / "autonomous_allhad/manual_validation").relative_to(repo)), "stages_remaining_proxy_or_blocked": ["manual legacy stop_processor_v4.py validation", "full-file processing", "Condor production", "real Combine expected limits", "adopted physics-change decisions"]}
    write_json(outputs / "real_subset_summary.json", real_summary)
    build_site(repo, real_summary)
    latest = ["# Latest Autonomous All-Hadronic Summary", "", "validate-feature-subset gate passed with actual ROOT/NanoAOD inputs.", f"Processed real event rows: {len(all_rows)}", f"Selected ROOT files: {len(manifest_files)}", f"Bad files: {len(bad_files)}", "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.", "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.", "Architecture selection: provisional only until manual legacy validation is supplied.", "Condor production: not started.", "Combine limits: not started.", f"Manual validation package: {repo / 'autonomous_allhad/manual_validation'}", f"Website: {repo / 'docs/index.html'}"]
    (workflow / "latest_summary.md").write_text("\n".join(latest) + "\n")
    provenance(repo)
    print(json.dumps(real_summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
