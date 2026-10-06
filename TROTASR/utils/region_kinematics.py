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
from .ids import object_masks, clean_by_delta_r, as_ak_bool


REGIONS = ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M")


DATA_STREAM = {
    "SR": "JetMET",
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
}


UPART_AK4_MEDIUM_WP_2024 = 0.1272


HISTOGRAMS = {
    "recoil": {
        "edges": [250, 300, 350, 400, 500, 650, 800, 1000, 1500],
        "overflow": "fold",
        "xlabel": "recoil pT (GeV)",
    },
    "met_sqrt_ht": {
        "edges": [10, 12, 15, 20, 25, 30, 40, 60, 100],
        "overflow": "fold",
        "xlabel": "recoil/sqrt(HT)",
    },
    "njet": {
        "edges": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 12.5, 20.5],
        "overflow": "fold",
        "xlabel": "Njet",
    },
    "nb": {
        "edges": [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 12.5],
        "overflow": "fold",
        "xlabel": "Nb",
    },
    "nisr": {
        "edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 12.5],
        "overflow": "fold",
        "xlabel": "NISR",
    },
    "mtb": {
        "edges": [0, 50, 100, 150, 200, 300, 500, 800, 1200, 2000],
        "overflow": "fold",
        "xlabel": "min mT(b,recoil) (GeV)",
    },
    "ptb": {
        "edges": [0, 30, 60, 100, 150, 200, 300, 500, 800, 1500],
        "overflow": "fold",
        "xlabel": "leading b pT (GeV)",
    },
    "ht": {
        "edges": [300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000],
        "overflow": "fold",
        "xlabel": "HT (GeV)",
    },
}


SELECTION_BRANCHES = (
    "physical_dataset_id",
    "dataset_id",
    "is_data",
    "is_signal",
    "is_background",
    "signal_topology_id",
    "mStop",
    "mLSP",
    "year",
    "run",
    "luminosityBlock",
    "event",
    "file_id",
    "entry",
    "gen_weight",
    "feature_LLCR",
    "feature_QCDCR",
    "feature_GCR",
    "feature_SR",
    "feature_lowdm_LLCR",
    "feature_lowdm_QCDCR",
    "feature_lowdm_GCR",
    "feature_lowdm_SR",
    "pass_base_common",
    "pass_signal_trigger",
    "pass_photon_trigger",
    "pass_electron_trigger",
    "pass_muon_trigger",
    "pass_zero_tau",
    "pass_no_veto_leptons",
    "pass_one_veto_lepton",
    "pass_mt_100",
    "pass_met_250",
    "pass_ht_300",
    "pass_qcd_open",
    "pass_dphi123_0p1",
    "met",
    "met_phi",
    "ht",
    "njet",
    "n_e_veto",
    "n_e_medium",
    "n_m_loose",
    "n_m_medium",
    "n_photon_medium",
    "mee",
    "pee",
    "mmm",
    "pmm",
    "recoil_gcr",
    "recoil_gcr_phi",
    "recoil_dy2e",
    "recoil_dy2e_phi",
    "recoil_dy2m",
    "recoil_dy2m_phi",
    "jet_corrected_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_btag_upart_all",
    "jet_id_all",
    "jet_source_index_all",
    "fatjet_corrected_pt",
    "fatjet_eta_all",
    "fatjet_phi_all",
    "fatjet_id_all",
    "fatjet_subjet_index1_all",
    "fatjet_subjet_index2_all",
    "fatjet_boosted_top_pass_all",
    "fatjet_boosted_w_pass_all",
    "subjet_eta_all",
    "subjet_phi_all",
    "electron_pt_all",
    "electron_eta_all",
    "electron_phi_all",
    "electron_charge_all",
    "electron_cutbased_all",
    "electron_mini_iso_all",
    "muon_pt_all",
    "muon_eta_all",
    "muon_phi_all",
    "muon_charge_all",
    "muon_loose_id_all",
    "muon_medium_id_all",
    "muon_mini_iso_all",
    "photon_pt_all",
    "photon_eta_all",
    "photon_phi_all",
    "photon_cutbased_all",
    "photon_electron_veto_all",
)


@dataclass(frozen=True)
class RegionBlock:
    core: np.ndarray
    recoil: np.ndarray
    recoil_phi: np.ndarray
    ht: np.ndarray
    njet: np.ndarray
    nb: np.ndarray
    nt: np.ndarray
    nw: np.ndarray
    nisr: np.ndarray
    met_sqrt_ht: np.ndarray
    mtb: np.ndarray
    ptb: np.ndarray


def as_bool(arrays: ak.Array, name: str) -> np.ndarray:
    return np.asarray(arrays[name], dtype=bool)


def as_float(arrays: ak.Array, name: str) -> np.ndarray:
    return np.asarray(arrays[name], dtype=np.float64)


def as_int(arrays: ak.Array, name: str) -> np.ndarray:
    return np.asarray(arrays[name], dtype=np.int32)


def delta_phi(phi1: Any, phi2: Any) -> Any:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def first(values: ak.Array, fill: float) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), fill)))


def nth(values: ak.Array, index: int, fill: float) -> np.ndarray:
    padded = ak.pad_none(values, index + 1, axis=1, clip=False)
    return np.asarray(ak.to_numpy(ak.fill_none(padded[:, index], fill)))


def jet_kinematics(
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
    jet_btag: ak.Array,
    good: ak.Array,
    recoil_pt: np.ndarray,
    recoil_phi: np.ndarray,
) -> dict[str, np.ndarray]:
    selected_pt = jet_pt[good]
    selected_phi = jet_phi[good]
    selected_btag = jet_btag[good]
    medium = selected_btag > UPART_AK4_MEDIUM_WP_2024
    dphi = delta_phi(selected_phi, recoil_phi[:, None])
    j1 = first(dphi, 999.0)
    j2 = nth(dphi, 1, 999.0)
    j3 = nth(dphi, 2, 999.0)
    njet = np.asarray(ak.num(selected_pt, axis=1), dtype=np.int32)
    nb = np.asarray(ak.sum(medium, axis=1), dtype=np.int32)
    ht = np.asarray(ak.sum(selected_pt, axis=1), dtype=np.float64)
    b_order = ak.argsort(selected_btag[medium], axis=1, ascending=False)
    b_pt = selected_pt[medium][b_order]
    b_phi = selected_phi[medium][b_order]
    ptb = first(b_pt, -99.0).astype(np.float64)
    b_mtb = np.sqrt(
        np.maximum(
            0.0,
            2.0
            * b_pt
            * recoil_pt[:, None]
            * (1.0 - np.cos(b_phi - recoil_phi[:, None])),
        )
    )
    mtb1 = first(b_mtb, 999.0)
    mtb2 = nth(b_mtb, 1, 999.0)
    mtb = np.where(nb >= 2, np.minimum(mtb1, mtb2), np.where(nb == 1, mtb1, -99.0))
    return {
        "njet": njet,
        "nb": nb,
        "ht": ht,
        "open_pre": (j1 > 0.5) & (j2 > 0.15) & (j3 > 0.15),
        "met_sqrt_ht": np.divide(
            recoil_pt,
            np.sqrt(ht),
            out=np.full(len(ht), -99.0, dtype=np.float64),
            where=ht > 0.0,
        ),
        "mtb": np.asarray(mtb, dtype=np.float64),
        "ptb": ptb,
    }


LEPTON_VETO_PT_MIN = 10.0


def update_veto_leptons(
    arrays: Any,
    electron_pt_min: float = LEPTON_VETO_PT_MIN,
    muon_pt_min: float = LEPTON_VETO_PT_MIN,
) -> dict[str, int]:
    """Rebuild veto collections, counts and lepton flags from stored objects."""
    if min(electron_pt_min, muon_pt_min) < 5.0:
        raise ValueError("stored veto collections only support thresholds >=5 GeV")
    fields = set(arrays) if isinstance(arrays, dict) else set(ak.fields(arrays))
    old_e = np.asarray(arrays["n_e_veto"], dtype=int)
    old_m = np.asarray(arrays["n_m_loose"], dtype=int)
    for prefix, threshold in (("electron_veto", electron_pt_min), ("muon_loose", muon_pt_min)):
        keep = arrays[f"{prefix}_pt"] > threshold
        for suffix in ("pt", "eta", "eta_sc", "phi"):
            key = f"{prefix}_{suffix}"
            if key in fields:
                arrays[key] = arrays[key][keep]
    ne = np.asarray(ak.num(arrays["electron_veto_pt"], axis=1), dtype=int)
    nm = np.asarray(ak.num(arrays["muon_loose_pt"], axis=1), dtype=int)
    mt_pass = np.ones(len(ne), dtype=bool)
    met = np.asarray(arrays["met"], dtype=float)
    met_phi = np.asarray(arrays["met_phi"], dtype=float)
    for prefix in ("electron_veto", "muon_loose"):
        mt2 = 2 * arrays[f"{prefix}_pt"] * met[:, None] * (
            1 - np.cos(arrays[f"{prefix}_phi"] - met_phi[:, None])
        )
        mt_pass &= np.asarray(ak.all(mt2 < 100.0**2, axis=1), dtype=bool)
    arrays["n_e_veto"] = ak.Array(ne)
    arrays["n_m_loose"] = ak.Array(nm)
    arrays["pass_no_veto_leptons"] = ak.Array((ne == 0) & (nm == 0))
    arrays["pass_one_veto_lepton"] = ak.Array(((ne == 1) & (nm == 0)) | ((ne == 0) & (nm == 1)))
    arrays["pass_mt_100"] = ak.Array(mt_pass)
    return {
        "events_with_removed_electrons": int(np.count_nonzero(old_e != ne)),
        "events_with_removed_muons": int(np.count_nonzero(old_m != nm)),
    }


def fatjet_kinematics(
    arrays: ak.Array,
    object_eta: ak.Array,
    object_phi: ak.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fat_pt = arrays["fatjet_corrected_pt"]
    fat_eta = arrays["fatjet_eta_all"]
    fat_phi = arrays["fatjet_phi_all"]
    cleaned = clean_by_delta_r(fat_eta, fat_phi, object_eta, object_phi, 0.4)
    selected = (
        (fat_pt > 200.0)
        & (abs(fat_eta) < 2.4)
        & as_ak_bool(arrays["fatjet_id_all"])
        & cleaned
    )
    nisr = np.asarray(ak.sum(selected, axis=1), dtype=np.int32)
    nt = np.asarray(
        ak.sum(as_ak_bool(arrays["fatjet_boosted_top_pass_all"]) & cleaned, axis=1),
        dtype=np.int32,
    )
    nw = np.asarray(
        ak.sum(as_ak_bool(arrays["fatjet_boosted_w_pass_all"]) & cleaned, axis=1),
        dtype=np.int32,
    )
    return nisr, nt, nw


def dycr_lepton_mask(
    arrays: Any,
    channel: str,
    masks: dict[str, Any] | None = None,
    mass_window: tuple[float, float] | None = (71.0, 111.0),
) -> np.ndarray:
    """Rebuild the DY lepton selection from stored objects.

    ``mass_window=None`` deliberately leaves the dilepton mass unconstrained.
    Histogram production uses that mode once to persist the disjoint on-Z and
    off-Z counting matrices needed by the RZ measurement.  The default keeps
    the nominal 91 +/- 20 GeV DY control-region selection unchanged.
    """
    if channel not in {"DY2E", "DY2M"}:
        raise ValueError(f"unknown dilepton channel: {channel}")
    masks = object_masks(arrays) if masks is None else masks
    electron = channel == "DY2E"
    flavor = "electron" if electron else "muon"
    selected = masks[f"{flavor}_medium"]
    pt = arrays[f"{flavor}_pt_all"][selected]
    charge = arrays[f"{flavor}_charge_all"][selected]
    count_name = "n_e_medium" if electron else "n_m_medium"
    rebuilt_count = np.asarray(ak.sum(selected, axis=1), dtype=int)
    if np.any(rebuilt_count != as_int(arrays, count_name)):
        raise RuntimeError(f"{channel}: rebuilt/stored medium-lepton counts differ")
    mass = as_float(arrays, "mee" if electron else "mmm")
    selected = (
        as_bool(arrays, "pass_base_common")
        & as_bool(arrays, "pass_zero_tau")
        & as_bool(arrays, f"pass_{flavor}_trigger")
        & (as_int(arrays, "n_m_loose" if electron else "n_e_veto") == 0)
        & (rebuilt_count == 2)
        & (first(pt, -99.0) > (40.0 if electron else 50.0))
        & (nth(pt, 1, -99.0) > 20.0)
        & (as_float(arrays, "pee" if electron else "pmm") > 200.0)
        & (first(charge, 0.0) != nth(charge, 1, 0.0))
    )
    if mass_window is not None:
        low, high = mass_window
        selected &= (mass > float(low)) & (mass < float(high))
    return selected


def build_region_blocks(
    arrays: ak.Array,
    *,
    dy_mass_window: tuple[float, float] | None = (71.0, 111.0),
) -> tuple[dict[str, RegionBlock], dict[str, int]]:
    n = len(arrays)
    veto_audit = update_veto_leptons(arrays)
    masks = object_masks(arrays)
    audit = {
        **veto_audit,
        "electron_veto_count_mismatches": int(
            np.count_nonzero(
                np.asarray(ak.sum(masks["electron_veto"], axis=1), dtype=int)
                != as_int(arrays, "n_e_veto")
            )
        ),
        "electron_medium_count_mismatches": int(
            np.count_nonzero(
                np.asarray(ak.sum(masks["electron_medium"], axis=1), dtype=int)
                != as_int(arrays, "n_e_medium")
            )
        ),
        "muon_loose_count_mismatches": int(
            np.count_nonzero(
                np.asarray(ak.sum(masks["muon_loose"], axis=1), dtype=int)
                != as_int(arrays, "n_m_loose")
            )
        ),
        "muon_medium_count_mismatches": int(
            np.count_nonzero(
                np.asarray(ak.sum(masks["muon_medium"], axis=1), dtype=int)
                != as_int(arrays, "n_m_medium")
            )
        ),
        "photon_medium_count_mismatches": int(
            np.count_nonzero(
                np.asarray(ak.sum(masks["photon_medium"], axis=1), dtype=int)
                != as_int(arrays, "n_photon_medium")
            )
        ),
    }

    jet_pt = arrays["jet_corrected_pt"]
    jet_eta = arrays["jet_eta_all"]
    jet_phi = arrays["jet_phi_all"]
    jet_btag = arrays["jet_btag_upart_all"]
    good_nominal = (
        (jet_pt > 30.0)
        & (abs(jet_eta) < 2.4)
        & as_ak_bool(arrays["jet_id_all"])
    )
    no_reference = arrays["photon_pt_all"][:, :0]
    nominal_nisr, nominal_nt, nominal_nw = fatjet_kinematics(
        arrays, no_reference, no_reference
    )

    met = as_float(arrays, "met")
    met_phi = as_float(arrays, "met_phi")
    nominal_jets = jet_kinematics(
        jet_pt, jet_eta, jet_phi, jet_btag, good_nominal, met, met_phi
    )
    base = as_bool(arrays, "pass_base_common") & as_bool(arrays, "pass_zero_tau")
    signal_trigger = as_bool(arrays, "pass_signal_trigger")
    no_leptons = as_bool(arrays, "pass_no_veto_leptons")
    common_met = as_bool(arrays, "pass_met_250")
    common_ht = as_bool(arrays, "pass_ht_300")
    common_njet = nominal_jets["njet"] >= 2

    shared_nominal = dict(
        recoil=met,
        recoil_phi=met_phi,
        ht=nominal_jets["ht"],
        njet=nominal_jets["njet"],
        nb=nominal_jets["nb"],
        nt=nominal_nt,
        nw=nominal_nw,
        nisr=nominal_nisr,
        met_sqrt_ht=nominal_jets["met_sqrt_ht"],
        mtb=nominal_jets["mtb"],
        ptb=nominal_jets["ptb"],
    )
    blocks: dict[str, RegionBlock] = {
        "SR": RegionBlock(
            core=(
                base
                & signal_trigger
                & no_leptons
                & common_njet
                & common_met
                & common_ht
                & nominal_jets["open_pre"]
            ),
            **shared_nominal,
        ),
        "LLCR": RegionBlock(
            core=(
                base
                & signal_trigger
                & as_bool(arrays, "pass_one_veto_lepton")
                & as_bool(arrays, "pass_mt_100")
                & common_njet
                & common_met
                & common_ht
                & nominal_jets["open_pre"]
            ),
            **shared_nominal,
        ),
        "QCDCR": RegionBlock(
            core=(
                base
                & signal_trigger
                & no_leptons
                & common_njet
                & common_met
                & common_ht
                & as_bool(arrays, "pass_qcd_open")
                & as_bool(arrays, "pass_dphi123_0p1")
            ),
            **shared_nominal,
        ),
    }

    region_objects = {
        "GCR": (
            arrays["photon_eta_all"][masks["photon_medium"]],
            arrays["photon_phi_all"][masks["photon_medium"]],
            as_float(arrays, "recoil_gcr"),
            as_float(arrays, "recoil_gcr_phi"),
        ),
        "DY2E": (
            arrays["electron_eta_all"][masks["electron_medium"]],
            arrays["electron_phi_all"][masks["electron_medium"]],
            as_float(arrays, "recoil_dy2e"),
            as_float(arrays, "recoil_dy2e_phi"),
        ),
        "DY2M": (
            arrays["muon_eta_all"][masks["muon_medium"]],
            arrays["muon_phi_all"][masks["muon_medium"]],
            as_float(arrays, "recoil_dy2m"),
            as_float(arrays, "recoil_dy2m_phi"),
        ),
    }
    cleaned: dict[str, dict[str, Any]] = {}
    for region, (obj_eta, obj_phi, recoil, recoil_phi) in region_objects.items():
        jet_clean = clean_by_delta_r(jet_eta, jet_phi, obj_eta, obj_phi, 0.2)
        jets = jet_kinematics(
            jet_pt,
            jet_eta,
            jet_phi,
            jet_btag,
            good_nominal & jet_clean,
            recoil,
            recoil_phi,
        )
        nisr, nt, nw = fatjet_kinematics(arrays, obj_eta, obj_phi)
        cleaned[region] = {
            **jets,
            "recoil": recoil,
            "recoil_phi": recoil_phi,
            "nisr": nisr,
            "nt": nt,
            "nw": nw,
        }

    photon = cleaned["GCR"]
    gcr_core = (
        base
        & as_bool(arrays, "pass_photon_trigger")
        & (as_int(arrays, "n_photon_medium") == 1)
        & no_leptons
        & (met < 250.0)
        & (photon["recoil"] > 250.0)
        & (photon["njet"] >= 2)
        & (photon["ht"] > 300.0)
        & photon["open_pre"]
    )
    blocks["GCR"] = RegionBlock(core=gcr_core, **block_fields(photon))

    electron = cleaned["DY2E"]
    dy2e_core = (
        dycr_lepton_mask(
            arrays, "DY2E", masks, mass_window=dy_mass_window
        )
        & (electron["recoil"] > 250.0)
        & (electron["njet"] >= 2)
        & (electron["ht"] > 300.0)
        & electron["open_pre"]
    )
    blocks["DY2E"] = RegionBlock(core=dy2e_core, **block_fields(electron))

    muon = cleaned["DY2M"]
    dy2m_core = (
        dycr_lepton_mask(
            arrays, "DY2M", masks, mass_window=dy_mass_window
        )
        & (muon["recoil"] > 250.0)
        & (muon["njet"] >= 2)
        & (muon["ht"] > 300.0)
        & muon["open_pre"]
    )
    blocks["DY2M"] = RegionBlock(core=dy2m_core, **block_fields(muon))
    audit["events"] = n
    return blocks, audit


def block_fields(values: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "recoil": np.asarray(values["recoil"], dtype=np.float64),
        "recoil_phi": np.asarray(values["recoil_phi"], dtype=np.float64),
        "ht": np.asarray(values["ht"], dtype=np.float64),
        "njet": np.asarray(values["njet"], dtype=np.int32),
        "nb": np.asarray(values["nb"], dtype=np.int32),
        "nt": np.asarray(values["nt"], dtype=np.int32),
        "nw": np.asarray(values["nw"], dtype=np.int32),
        "nisr": np.asarray(values["nisr"], dtype=np.int32),
        "met_sqrt_ht": np.asarray(values["met_sqrt_ht"], dtype=np.float64),
        "mtb": np.asarray(values["mtb"], dtype=np.float64),
        "ptb": np.asarray(values["ptb"], dtype=np.float64),
    }

