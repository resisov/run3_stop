from __future__ import annotations
import math
import json
import os
import contextlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import awkward as ak
from .event_selections import legacy_vr_mask as region_mask
from .signal_models import signal_topology


WEIGHT_BRANCHES = [
    "run", "luminosityBlock", "event", "entry", "dataset_id", "year", "mStop", "mLSP",
    "is_data", "is_signal", "is_background", "gen_weight", "pu_ntrueint",
    "n_e_veto", "n_e_medium", "n_m_loose", "n_m_medium",
    "good_jet_pt", "good_jet_eta", "good_jet_phi", "good_jet_hadron_flavour", "good_jet_b_medium",
    "electron_veto_pt", "electron_veto_eta_sc", "electron_veto_phi",
    "electron_veto_eta",
    "electron_medium_pt", "electron_medium_eta_sc", "electron_medium_phi",
    "electron_medium_eta",
    "muon_loose_pt", "muon_loose_eta", "muon_loose_phi",
    "muon_medium_pt", "muon_medium_eta", "muon_medium_phi",
    "photon_medium_pt", "photon_medium_eta", "photon_medium_phi",
    "photon_pt_all", "photon_eta_all", "photon_phi_all", "photon_r9_all",
    "photon_cutbased_all", "photon_electron_veto_all", "gen_top_pt",
]


SIGNAL_BTAG_FASTSIM_MASSES = (
    100, 200, 300, 400, 500, 600, 700, 800, 900, 950,
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
)


SIGNAL_BTAG_EFFICIENCY_DATASETS = {
    mstop: (
        f"SMS-2Stop_Par-mStop-{mstop}_TuneCP5_13p6TeV_madgraphMLM-pythia8-"
        + (
            "RunIII2024Summer24NanoAODv15-FSMiniv6_FSNanov15_150X_mcRun3_2024_realistic_v2-v1"
            if mstop <= 500
            else "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v1"
        )
    )
    for mstop in SIGNAL_BTAG_FASTSIM_MASSES
}


def signal_btag_efficiency_dataset(
    mstop: int,
    source_dataset: str = "",
) -> tuple[int, str]:
    topology = signal_topology(source_dataset)
    if topology in {"T2tb", "T2bW"}:
        parts = str(source_dataset).strip("/").split("/")
        if len(parts) >= 2:
            return int(mstop), f"{parts[0]}-{parts[1]}"
        raise RuntimeError(
            f"cannot derive b-tag efficiency key from signal dataset {source_dataset!r}"
        )
    anchor = min(SIGNAL_BTAG_EFFICIENCY_DATASETS, key=lambda value: (abs(value - int(mstop)), value))
    return anchor, SIGNAL_BTAG_EFFICIENCY_DATASETS[anchor]


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

    ev_pt = chunk["electron_veto_pt"]
    ev_eta_sc = chunk["electron_veto_eta_sc"]
    ev_phi = chunk["electron_veto_phi"]
    em_pt = chunk["electron_medium_pt"]
    em_eta_sc = chunk["electron_medium_eta_sc"]
    em_phi = chunk["electron_medium_phi"]
    raw_eta_available = (
        "electron_veto_eta" in chunk and "electron_medium_eta" in chunk
    )
    ev_eta = chunk["electron_veto_eta"] if raw_eta_available else ev_eta_sc
    em_eta = chunk["electron_medium_eta"] if raw_eta_available else em_eta_sc
    e_pt = combine_two(ev_pt, em_pt)
    e_eta = combine_two(ev_eta, em_eta)
    e_eta_sc = combine_two(ev_eta_sc, em_eta_sc)
    e_phi = combine_two(ev_phi, em_phi)
    e_veto = combine_two(ones_mask(ev_pt), zeros_mask(em_pt))
    e_med = combine_two(zeros_mask(ev_pt), ones_mask(em_pt))
    e_delta_eta_sc = e_eta_sc - e_eta

    ml_pt, ml_eta, ml_phi = chunk["muon_loose_pt"], chunk["muon_loose_eta"], chunk["muon_loose_phi"]
    mm_pt, mm_eta, mm_phi = chunk["muon_medium_pt"], chunk["muon_medium_eta"], chunk["muon_medium_phi"]
    m_pt = combine_two(ml_pt, mm_pt)
    m_eta = combine_two(ml_eta, mm_eta)
    m_phi = combine_two(ml_phi, mm_phi)
    m_loose = combine_two(ones_mask(ml_pt), zeros_mask(mm_pt))
    m_med = combine_two(zeros_mask(ml_pt), ones_mask(mm_pt))

    photon_all_fields = {
        "photon_pt_all", "photon_eta_all", "photon_phi_all", "photon_r9_all",
        "photon_cutbased_all", "photon_electron_veto_all",
    }
    if photon_all_fields <= set(chunk):
        p_pt = chunk["photon_pt_all"]
        p_eta = chunk["photon_eta_all"]
        p_phi = chunk["photon_phi_all"]
        p_r9 = chunk["photon_r9_all"]
        p_med = (
            (p_pt > 220.0)
            & ((abs(p_eta) < 1.4442) | ((abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5)))
            & (chunk["photon_cutbased_all"] >= 2)
            & ak.values_astype(chunk["photon_electron_veto_all"], np.bool_)
        )
    else:
        p_pt = chunk["photon_medium_pt"]
        p_eta = chunk["photon_medium_eta"]
        p_phi = chunk["photon_medium_phi"]
        p_r9 = None
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
        "p_r9": p_r9,
        "p_med": p_med,
        "gcr_mask": (
            as_bool(chunk["feature_GCR"], n)
            | as_bool(chunk["feature_lowdm_GCR"], n)
        ),
        "met_pt": np.asarray(chunk["met"], dtype=float),
        "met_trigger_mask": (
            as_bool(chunk["feature_LLCR"], n)
            | as_bool(chunk["feature_QCDCR"], n)
            | as_bool(chunk["feature_SR"], n)
            | as_bool(chunk["feature_lowdm_LLCR"], n)
            | as_bool(chunk["feature_lowdm_QCDCR"], n)
            | as_bool(chunk["feature_lowdm_SR"], n)
            | region_mask(chunk, "HighDMVR_Nb1", "pass_base_common", n)
            | region_mask(chunk, "HighDMVR_Nb2", "pass_base_common", n)
            | region_mask(chunk, "HighDMVR_Nb3plus", "pass_base_common", n)
        ),
        "electron_eta_source": (
            "raw_eta_with_delta_eta_sc"
            if raw_eta_available
            else "eta_sc_fallback_current_schema"
        ),
    }
    return arrays, inputs

