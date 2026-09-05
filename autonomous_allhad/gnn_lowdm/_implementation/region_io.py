"""Shared Low-dM region reconstruction, normalization, and histogram I/O.

The four-category masks in this module are inclusive control-region parent
groups. They are not an alternative signal-region binning; the only supported
statistical SR layout is the six-category, 30-bin definition in ``config.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
import uproot


SCRIPT = Path(__file__).resolve()
OUTER_PROJECT = SCRIPT.parents[2]
REPOSITORY = OUTER_PROJECT.parent
for _candidate in (OUTER_PROJECT, OUTER_PROJECT / "workflow"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:  # Repository-root package layout.
    from autonomous_allhad.autonomous_allhad.highdm_resolved_categories import (  # noqa: E402
        boosted_overlap_vetoed_ak4_indices,
        map_candidates_to_events,
        map_candidates_to_events_rle,
        select_exclusive_resolved_candidates,
    )
    from autonomous_allhad.autonomous_allhad.signal_models import (  # noqa: E402
        SIGNAL_TOPOLOGY_IDS,
        signal_topology,
    )
except ImportError:  # Standalone EOS payload layout.
    from autonomous_allhad.highdm_resolved_categories import (  # type: ignore[no-redef]  # noqa: E402
        boosted_overlap_vetoed_ak4_indices,
        map_candidates_to_events,
        map_candidates_to_events_rle,
        select_exclusive_resolved_candidates,
    )
    from autonomous_allhad.signal_models import (  # type: ignore[no-redef]  # noqa: E402
        SIGNAL_TOPOLOGY_IDS,
        signal_topology,
    )


SCHEMA = "gnn_lowdm_region_io_v2"
REQUEST_SCHEMA = "gnn_lowdm_test_histogram_request_v1"
PARTIAL_SCHEMA = "gnn_lowdm_test_histogram_partial_v1"
REGIONS = ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
CATEGORIES = (
    "Nb1_NISR0",
    "Nb1_NISR1plus",
    "Nb2plus_NISR0",
    "Nb2plus_NISR1plus",
)
DATA_STREAM = {
    "SR": "JetMET",
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
}
TOPOLOGY_NAMES = {
    identifier: topology
    for topology, identifier in SIGNAL_TOPOLOGY_IDS.items()
    if topology
}
UPART_AK4_MEDIUM_WP_2024 = 0.1272
MAX_ABS_HIST_WEIGHT = 1.0e12

EXPECTED_TROTA_SCHEMAS = {
    2024: "trota_topresolved_2024_inplace_sparse_v1",
    2025: "trota_topresolved_2025_inplace_sparse_v1",
}
EXPECTED_TROTA_MODEL_SHA256 = (
    "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
)
TROTA_PRIMARY_BRANCHES = (
    "file_id",
    "entry",
    "TopResolved1pct_candidateIndex",
    "TopResolved1pct_sourceJetIdx0",
    "TopResolved1pct_sourceJetIdx1",
    "TopResolved1pct_sourceJetIdx2",
    "TopResolved1pct_eta",
    "TopResolved1pct_mass",
    "TopResolved1pct_QCDDiscriminant",
)
TROTA_FALLBACK_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    *TROTA_PRIMARY_BRANCHES[2:],
)

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

# Rz factors from the adopted 2024 DY normalization payload.  They are applied
# only to the auxiliary ``nominal_rz`` DY histograms; the raw nominal templates
# remain available for auditing.
RZ_FACTORS = {
    "DY2E": {"Nb1": 0.6242885433272645, "Nb2plus": 0.7996132657524656},
    "DY2M": {"Nb1": 0.5985045784169216, "Nb2plus": 0.5574163275293518},
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(payload)
    os.replace(temporary, path)


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


def clean_by_delta_r(
    obj_eta: ak.Array,
    obj_phi: ak.Array,
    ref_eta: ak.Array,
    ref_phi: ak.Array,
    dr_min: float,
) -> ak.Array:
    deta = obj_eta[:, :, None] - ref_eta[:, None, :]
    dphi = delta_phi(obj_phi[:, :, None], ref_phi[:, None, :])
    return ak.all((deta * deta + dphi * dphi) > dr_min * dr_min, axis=2)


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


def object_masks(arrays: ak.Array) -> dict[str, ak.Array]:
    e_pt = arrays["electron_pt_all"]
    e_eta = arrays["electron_eta_all"]
    e_cb = arrays["electron_cutbased_all"]
    e_iso = arrays["electron_mini_iso_all"]
    e_fid = (abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5))
    e_veto = (e_pt > 5.0) & e_fid & (e_cb >= 1) & (e_iso < 0.1)
    e_medium = (e_pt > 10.0) & e_fid & (e_cb >= 3) & (e_iso < 0.1)

    m_pt = arrays["muon_pt_all"]
    m_eta = arrays["muon_eta_all"]
    m_iso = arrays["muon_mini_iso_all"]
    m_loose = (
        (m_pt > 5.0)
        & (abs(m_eta) < 2.4)
        & as_ak_bool(arrays["muon_loose_id_all"])
        & (m_iso < 0.2)
    )
    m_medium = (
        (m_pt > 10.0)
        & (abs(m_eta) < 2.4)
        & as_ak_bool(arrays["muon_medium_id_all"])
        & (m_iso < 0.2)
    )

    p_pt = arrays["photon_pt_all"]
    p_eta = arrays["photon_eta_all"]
    p_fid = (abs(p_eta) < 1.4442) | ((abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5))
    p_medium = (
        (p_pt > 220.0)
        & p_fid
        & (arrays["photon_cutbased_all"] >= 2)
        & as_ak_bool(arrays["photon_electron_veto_all"])
    )
    return {
        "electron_veto": e_veto,
        "electron_medium": e_medium,
        "muon_loose": m_loose,
        "muon_medium": m_medium,
        "photon_medium": p_medium,
    }


def as_ak_bool(values: ak.Array) -> ak.Array:
    return ak.values_astype(values, np.bool_)


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


def build_region_blocks(arrays: ak.Array) -> tuple[dict[str, RegionBlock], dict[str, int]]:
    n = len(arrays)
    masks = object_masks(arrays)
    audit = {
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

    e_medium = masks["electron_medium"]
    e_pt = arrays["electron_pt_all"][e_medium]
    e_charge = arrays["electron_charge_all"][e_medium]
    electron = cleaned["DY2E"]
    dy2e_core = (
        base
        & as_bool(arrays, "pass_electron_trigger")
        & (as_int(arrays, "n_m_loose") == 0)
        & (as_int(arrays, "n_e_medium") == 2)
        & (first(e_pt, -99.0) > 40.0)
        & (nth(e_pt, 1, -99.0) > 20.0)
        & (as_float(arrays, "pee") > 200.0)
        & (first(e_charge, 0.0) != nth(e_charge, 1, 0.0))
        & (as_float(arrays, "mee") > 81.0)
        & (as_float(arrays, "mee") < 101.0)
        & (electron["recoil"] > 250.0)
        & (electron["njet"] >= 2)
        & (electron["ht"] > 300.0)
        & electron["open_pre"]
    )
    blocks["DY2E"] = RegionBlock(core=dy2e_core, **block_fields(electron))

    m_medium = masks["muon_medium"]
    m_pt = arrays["muon_pt_all"][m_medium]
    m_charge = arrays["muon_charge_all"][m_medium]
    muon = cleaned["DY2M"]
    dy2m_core = (
        base
        & as_bool(arrays, "pass_muon_trigger")
        & (as_int(arrays, "n_e_veto") == 0)
        & (as_int(arrays, "n_m_medium") == 2)
        & (first(m_pt, -99.0) > 50.0)
        & (nth(m_pt, 1, -99.0) > 20.0)
        & (as_float(arrays, "pmm") > 200.0)
        & (first(m_charge, 0.0) != nth(m_charge, 1, 0.0))
        & (as_float(arrays, "mmm") > 81.0)
        & (as_float(arrays, "mmm") < 101.0)
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


def validate_trota_provenance(sidecar: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        (year, sidecar.get(f"trota_topresolved_{year}"))
        for year in sorted(EXPECTED_TROTA_SCHEMAS)
        if sidecar.get(f"trota_topresolved_{year}") is not None
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "invalid TROTA provenance: expected exactly one supported year payload"
        )
    application_year, payload = candidates[0]
    assert isinstance(payload, dict)
    marker = payload.get("marker") or {}
    if (
        payload.get("status") != "complete"
        or payload.get("schema_version") != EXPECTED_TROTA_SCHEMAS[application_year]
        or marker.get("status") != "complete"
        or int(marker.get("application_year", -1)) != application_year
        or int(marker.get("model_release_year", -1)) != 2024
        or marker.get("model_sha256") != EXPECTED_TROTA_MODEL_SHA256
    ):
        raise RuntimeError(f"invalid TROTA provenance: {payload}")
    return {
        "application_year": application_year,
        "schema_version": payload["schema_version"],
        "model_sha256": marker["model_sha256"],
    }


def compute_nres(
    arrays: ak.Array,
    trota_tree: Any,
    eligible: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    counts = np.zeros(len(arrays), dtype=np.int16)
    tree_fields = set(trota_tree.keys())
    identity_fallback = 0
    if set(TROTA_PRIMARY_BRANCHES) <= tree_fields:
        fields = TROTA_PRIMARY_BRANCHES
        values_ak = trota_tree.arrays(fields, library="ak")
        values = {name: np.asarray(ak.to_numpy(values_ak[name])) for name in fields}
        candidate_event = map_candidates_to_events(
            np.asarray(arrays["file_id"]),
            np.asarray(arrays["entry"]),
            values["file_id"],
            values["entry"],
        )
    elif set(TROTA_FALLBACK_BRANCHES) <= tree_fields:
        fields = TROTA_FALLBACK_BRANCHES
        values_ak = trota_tree.arrays(fields, library="ak")
        values = {name: np.asarray(ak.to_numpy(values_ak[name])) for name in fields}
        candidate_event = map_candidates_to_events_rle(
            np.asarray(arrays["run"]),
            np.asarray(arrays["luminosityBlock"]),
            np.asarray(arrays["event"]),
            values["run"],
            values["luminosityBlock"],
            values["event"],
        )
        identity_fallback = 1
    else:
        raise RuntimeError("TROTA tree has neither validated identity schema")

    fiducial = (
        eligible[candidate_event]
        & np.isfinite(values["TopResolved1pct_eta"])
        & np.isfinite(values["TopResolved1pct_mass"])
        & np.isfinite(values["TopResolved1pct_QCDDiscriminant"])
        & (np.abs(values["TopResolved1pct_eta"]) < 2.0)
        & (values["TopResolved1pct_mass"] >= 100.0)
        & (values["TopResolved1pct_mass"] <= 250.0)
    )
    selected_rows = np.flatnonzero(fiducial)
    rejected_boosted = 0
    rejected_resolved = 0
    if selected_rows.size:
        order = np.argsort(candidate_event[selected_rows], kind="stable")
        selected_rows = selected_rows[order]
        selected_events = candidate_event[selected_rows]
        boundaries = np.flatnonzero(np.diff(selected_events)) + 1
        for rows in np.split(selected_rows, boundaries):
            event_index = int(candidate_event[rows[0]])
            vetoed = boosted_overlap_vetoed_ak4_indices(
                jet_source_indices=ak.to_list(arrays["jet_source_index_all"][event_index]),
                jet_eta=ak.to_list(arrays["jet_eta_all"][event_index]),
                jet_phi=ak.to_list(arrays["jet_phi_all"][event_index]),
                fatjet_eta=ak.to_list(arrays["fatjet_eta_all"][event_index]),
                fatjet_phi=ak.to_list(arrays["fatjet_phi_all"][event_index]),
                fatjet_subjet_index1=ak.to_list(
                    arrays["fatjet_subjet_index1_all"][event_index]
                ),
                fatjet_subjet_index2=ak.to_list(
                    arrays["fatjet_subjet_index2_all"][event_index]
                ),
                fatjet_top_pass=ak.to_list(
                    arrays["fatjet_boosted_top_pass_all"][event_index]
                ),
                fatjet_w_pass=ak.to_list(
                    arrays["fatjet_boosted_w_pass_all"][event_index]
                ),
                subjet_eta=ak.to_list(arrays["subjet_eta_all"][event_index]),
                subjet_phi=ak.to_list(arrays["subjet_phi_all"][event_index]),
            )
            result = select_exclusive_resolved_candidates(
                candidate_indices=values["TopResolved1pct_candidateIndex"][rows],
                candidate_scores=values["TopResolved1pct_QCDDiscriminant"][rows],
                candidate_source_jets=np.stack(
                    [
                        values["TopResolved1pct_sourceJetIdx0"][rows],
                        values["TopResolved1pct_sourceJetIdx1"][rows],
                        values["TopResolved1pct_sourceJetIdx2"][rows],
                    ],
                    axis=1,
                ),
                boosted_vetoed_ak4_indices=vetoed,
            )
            counts[event_index] = result.nres
            rejected_boosted += len(result.rejected_by_boosted_overlap)
            rejected_resolved += len(result.rejected_by_resolved_overlap)
    return counts, {
        "events": len(arrays),
        "eligible_events": int(np.count_nonzero(eligible)),
        "nres_positive_events": int(np.count_nonzero(eligible & (counts > 0))),
        "trota_rows": int(trota_tree.num_entries),
        "fiducial_rows": int(selected_rows.size),
        "rejected_by_boosted_overlap": int(rejected_boosted),
        "rejected_by_resolved_overlap": int(rejected_resolved),
        "identity_fallback_files": int(identity_fallback),
    }


def final_region_masks(
    blocks: dict[str, RegionBlock],
    nres: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        region: (
            block.core
            & (block.nb >= 1)
            & (block.nt == 0)
            & (block.nw == 0)
            & (nres == 0)
        )
        for region, block in blocks.items()
    }


def category_masks(block: RegionBlock) -> dict[str, np.ndarray]:
    return {
        "Nb1_NISR0": (block.nb == 1) & (block.nisr == 0),
        "Nb1_NISR1plus": (block.nb == 1) & (block.nisr >= 1),
        "Nb2plus_NISR0": (block.nb >= 2) & (block.nisr == 0),
        "Nb2plus_NISR1plus": (block.nb >= 2) & (block.nisr >= 1),
    }


def histogram_values(block: RegionBlock) -> dict[str, np.ndarray]:
    return {
        "recoil": block.recoil,
        "met_sqrt_ht": block.met_sqrt_ht,
        "njet": block.njet,
        "nb": block.nb,
        "nisr": block.nisr,
        "mtb": block.mtb,
        "ptb": block.ptb,
        "ht": block.ht,
    }


def empty_counter(edges: Iterable[float]) -> dict[str, Any]:
    size = len(tuple(edges)) - 1
    return {"sumw": [0.0] * size, "sumw2": [0.0] * size, "entries": [0] * size}


def fill_counter(
    target: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
    spec: dict[str, Any],
) -> None:
    edges = np.asarray(spec["edges"], dtype=np.float64)
    local_values = np.asarray(values, dtype=np.float64)[selected]
    local_weights = np.asarray(weights, dtype=np.float64)[selected]
    finite = (
        np.isfinite(local_values)
        & np.isfinite(local_weights)
        & (np.abs(local_weights) <= MAX_ABS_HIST_WEIGHT)
    )
    local_values = local_values[finite]
    local_weights = local_weights[finite]
    if spec.get("overflow") == "fold" and local_values.size:
        epsilon = max(1.0, abs(edges[-1])) * np.finfo(np.float64).eps * 8.0
        local_values = np.clip(local_values, edges[0] + epsilon, edges[-1] - epsilon)
    sumw = np.histogram(local_values, bins=edges, weights=local_weights)[0]
    sumw2 = np.histogram(local_values, bins=edges, weights=local_weights**2)[0]
    entries = np.histogram(local_values, bins=edges)[0]
    for name, values_to_add in (("sumw", sumw), ("sumw2", sumw2), ("entries", entries)):
        target[name] = (
            np.asarray(target[name], dtype=np.float64) + values_to_add
        ).tolist()


def add_audit(target: dict[str, int], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, np.integer)):
            target[key] = int(target.get(key, 0)) + int(value)


def canonical_process(process: str, dataset: str) -> str:
    if process in {"JetMET", "EGamma", "Muon"}:
        return process
    if process == "VV":
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "ST"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "TT"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "GJ"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "WtoLNu"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Zto2Nu"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD"
    return process or "unclassified"


def dataset_record(sidecar: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    record = (sidecar.get("datasets") or {}).get(str(int(dataset_id)))
    if record is None:
        raise RuntimeError(f"sidecar has no dataset record for dataset_id={dataset_id}")
    return record


def signal_normalization_map(manifest: dict[str, Any]) -> dict[tuple[str, int, int], float]:
    return {
        (str(item["topology"]), int(item["mStop"]), int(item["mLSP"])): float(item["sumw"])
        for item in manifest["normalization"]["signal_mass_points"]
    }


def stop_xsec_map(payload: dict[str, Any]) -> dict[int, float]:
    return {
        int(item["mStop"]): float(item["xsec_pb"])
        for item in payload["records"]
        if item.get("parsing_status") == "parsed"
    }


def normalized_weight_variations(
    sub_group: ak.Array,
    sidecar_record: dict[str, Any],
    process: str,
    dataset: str,
    manifest: dict[str, Any],
    signal_norm: dict[tuple[str, int, int], float],
    stop_xsec: dict[int, float],
    repo: Path,
    new_region_masks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(sub_group)
    is_data = bool(sidecar_record.get("is_data"))
    is_signal = bool(sidecar_record.get("is_signal"))
    if is_data:
        return {"nominal": np.ones(n, dtype=np.float64)}, {
            "mode": "data_unity",
            "components": {},
            "available_variations": ["nominal"],
        }

    if is_signal:
        expected_topology = signal_topology(dataset)
        if not expected_topology:
            raise RuntimeError(
                f"cannot infer signal topology from physical dataset {dataset}"
            )
        topology_ids_for_audit = np.unique(
            np.asarray(sub_group["signal_topology_id"], dtype=int)
        )
        decoded_topologies = {
            TOPOLOGY_NAMES.get(int(identifier))
            for identifier in topology_ids_for_audit
        }
        if None in decoded_topologies or decoded_topologies != {expected_topology}:
            raise RuntimeError(
                "signal topology mismatch: "
                f"dataset={expected_topology}, "
                f"branch_ids={topology_ids_for_audit.tolist()}, "
                f"decoded={sorted(str(value) for value in decoded_topologies)}"
            )

    from build_flat_boosted_recoil_hists import flat_arrays_for_weights  # noqa: PLC0415
    try:
        from autonomous_allhad.autonomous_allhad.real_subset_worker import (  # noqa: PLC0415
            compute_weight_bundle,
        )
    except ImportError:
        from autonomous_allhad.real_subset_worker import compute_weight_bundle  # type: ignore[no-redef]  # noqa: PLC0415

    chunk = {name: sub_group[name] for name in ak.fields(sub_group)}
    arrays_for_weights, inputs = flat_arrays_for_weights(chunk)
    inputs["gcr_mask"] = np.asarray(new_region_masks["GCR"], dtype=bool)
    inputs["met_trigger_mask"] = np.asarray(
        new_region_masks["SR"]
        | new_region_masks["LLCR"]
        | new_region_masks["QCDCR"],
        dtype=bool,
    )
    year_values = np.asarray(sub_group["year"], dtype=int)
    year = str(int(year_values[0])) if len(year_values) else "2024"
    correction_dataset = dataset
    btag_efficiency_anchor = None
    if is_signal:
        from build_flat_boosted_recoil_hists import (  # noqa: PLC0415
            signal_btag_efficiency_dataset,
        )

        mstop_values = np.asarray(sub_group["mStop"], dtype=int)
        btag_efficiency_anchor, correction_dataset = signal_btag_efficiency_dataset(
            int(mstop_values[0]), dataset
        )
    _gen, variations, status = compute_weight_bundle(
        arrays_for_weights,
        repo,
        correction_dataset,
        process,
        year,
        inputs["n"],
        inputs["jet_pt"],
        inputs["jet_eta"],
        inputs["jet_hadflav"],
        inputs["b_med"],
        inputs["e_eta"],
        inputs["e_delta_eta_sc"],
        inputs["e_pt"],
        inputs["e_phi"],
        inputs["e_veto"],
        inputs["e_med"],
        inputs["n_e_veto"],
        inputs["n_e_med"],
        inputs["m_eta"],
        inputs["m_pt"],
        inputs["m_phi"],
        inputs["m_loose"],
        inputs["m_med"],
        inputs["n_m_loose"],
        inputs["n_m_med"],
        inputs["p_eta"],
        inputs["p_pt"],
        inputs["p_phi"],
        inputs["p_med"],
        inputs["gcr_mask"],
        p_r9=inputs["p_r9"],
        met_pt=inputs["met_pt"],
        met_trigger_mask=inputs["met_trigger_mask"],
    )
    required_components = {
        "pileup",
        "btagSF",
        "electron_reco",
        "electron_id",
        "electron_hlt",
        "muon_id",
        "muon_iso",
        "muon_hlt",
        "photon_id",
        "photon_csev",
        "met_trigger",
        "photon_trigger",
        "veto_electron_5to10",
        "loose_muon_5to10",
    }
    known_unavailable_unity: dict[str, str] = {}
    if year == "2025":
        # These two Prompt-2025 working points are not published in the EGM
        # payload yet.  Keep their central factors at unity, but accept that
        # fallback only when the correction code reports the expected explicit
        # unavailability.  This prevents unrelated failures from being hidden.
        expected_unavailable = {
            "electron_hlt": "2025 electron HLT SF is not available",
            "photon_csev": "has no published working-point content",
        }
        for component, expected_error in expected_unavailable.items():
            component_status = (status.get("components") or {}).get(component) or {}
            if component_status.get("applied"):
                continue
            error = str(component_status.get("error") or "")
            if expected_error not in error:
                continue
            required_components.discard(component)
            known_unavailable_unity[component] = error
    failed_components = sorted(
        name
        for name in required_components
        if not ((status.get("components") or {}).get(name) or {}).get("applied")
    )
    if failed_components:
        raise RuntimeError(
            "required nominal correction components were not applied: "
            + ", ".join(failed_components)
        )
    status["known_unavailable_unity_components"] = known_unavailable_unity
    luminosity_pb = float(manifest["normalization"]["luminosity_pb"])
    if is_signal:
        topology_ids = np.asarray(sub_group["signal_topology_id"], dtype=int)
        mstops = np.asarray(sub_group["mStop"], dtype=int)
        mlsps = np.asarray(sub_group["mLSP"], dtype=int)
        factor = np.zeros(n, dtype=np.float64)
        points = np.unique(np.stack([topology_ids, mstops, mlsps], axis=1), axis=0)
        for topology_id, mstop, mlsp in points:
            topology = TOPOLOGY_NAMES.get(int(topology_id))
            if topology is None:
                raise RuntimeError(f"unknown signal topology id {topology_id}")
            denominator = signal_norm.get((topology, int(mstop), int(mlsp)))
            xsec = stop_xsec.get(int(mstop))
            if denominator is None or denominator == 0.0 or xsec is None:
                raise RuntimeError(
                    f"missing signal normalization for {topology}({mstop},{mlsp})"
                )
            selected = (
                (topology_ids == topology_id) & (mstops == mstop) & (mlsps == mlsp)
            )
            factor[selected] = xsec * luminosity_pb / denominator
    else:
        physical_ids = np.asarray(sub_group["physical_dataset_id"], dtype=np.int64)
        factor = np.zeros(n, dtype=np.float64)
        background_norm = manifest["normalization"]["by_physical_dataset_id"]
        for physical_id in np.unique(physical_ids):
            record = background_norm.get(str(int(physical_id)))
            if record is None or not record.get("normalization_complete"):
                raise RuntimeError(
                    f"missing background normalization for physical_dataset_id={physical_id}"
                )
            factor[physical_ids == physical_id] = (
                float(record["xsec_pb"]) * luminosity_pb / float(record["sumw"])
            )
    normalized_variations = {
        str(name): np.asarray(raw, dtype=np.float64) * factor
        for name, raw in variations.items()
    }
    if "nominal" not in normalized_variations:
        raise RuntimeError("weight bundle has no nominal variation")
    nonfinite = sorted(
        name
        for name, weights in normalized_variations.items()
        if not np.all(np.isfinite(weights))
    )
    if nonfinite:
        raise RuntimeError(
            "non-finite normalized weights in variations: " + ", ".join(nonfinite)
        )
    status["available_variations"] = sorted(normalized_variations)
    if is_signal:
        status["signal_btag_efficiency_anchor_mstop"] = btag_efficiency_anchor
        status["signal_correction_dataset"] = correction_dataset
        status["signal_source_dataset"] = dataset
    return normalized_variations, status


def normalized_weights(
    sub_group: ak.Array,
    sidecar_record: dict[str, Any],
    process: str,
    dataset: str,
    manifest: dict[str, Any],
    signal_norm: dict[tuple[str, int, int], float],
    stop_xsec: dict[int, float],
    repo: Path,
    new_region_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the nominal member of the audited normalized weight bundle.

    This wrapper preserves the established nominal-only call contract.  New
    template producers must call :func:`normalized_weight_variations` so that
    the already-evaluated Up/Down weights are not silently discarded.
    """
    variations, status = normalized_weight_variations(
        sub_group,
        sidecar_record,
        process,
        dataset,
        manifest,
        signal_norm,
        stop_xsec,
        repo,
        new_region_masks,
    )
    return variations["nominal"], status


def sample_name(
    sub_group: ak.Array,
    sidecar_record: dict[str, Any],
    process: str,
) -> str:
    if sidecar_record.get("is_data"):
        return "data_obs"
    if not sidecar_record.get("is_signal"):
        return process
    topology_ids = np.asarray(sub_group["signal_topology_id"], dtype=int)
    mstops = np.asarray(sub_group["mStop"], dtype=int)
    mlsps = np.asarray(sub_group["mLSP"], dtype=int)
    if len(np.unique(np.stack([topology_ids, mstops, mlsps], axis=1), axis=0)) != 1:
        raise RuntimeError("signal histogram subgroup contains multiple mass points")
    return (
        f"{TOPOLOGY_NAMES[int(topology_ids[0])]}_"
        f"mStop{int(mstops[0])}_mLSP{int(mlsps[0])}"
    )


def subgroup_masks(arrays: ak.Array, dataset_id: int, is_signal: bool) -> list[np.ndarray]:
    dataset_selected = np.asarray(arrays["dataset_id"], dtype=np.int64) == int(dataset_id)
    if not is_signal:
        return [dataset_selected]
    topology_ids = np.asarray(arrays["signal_topology_id"], dtype=int)
    mstops = np.asarray(arrays["mStop"], dtype=int)
    mlsps = np.asarray(arrays["mLSP"], dtype=int)
    points = np.unique(
        np.stack(
            [topology_ids[dataset_selected], mstops[dataset_selected], mlsps[dataset_selected]],
            axis=1,
        ),
        axis=0,
    )
    return [
        dataset_selected
        & (topology_ids == topology_id)
        & (mstops == mstop)
        & (mlsps == mlsp)
        for topology_id, mstop, mlsp in points
    ]


def new_hist_payload() -> dict[str, Any]:
    return {}


def counter_at(
    histograms: dict[str, Any],
    variation: str,
    region: str,
    category: str,
    sample: str,
    variable: str,
) -> dict[str, Any]:
    return (
        histograms.setdefault(variation, {})
        .setdefault(region, {})
        .setdefault(category, {})
        .setdefault(sample, {})
        .setdefault(variable, empty_counter(HISTOGRAMS[variable]["edges"]))
    )


def process_source(
    record: dict[str, Any],
    manifest: dict[str, Any],
    signal_norm: dict[tuple[str, int, int], float],
    stop_xsec: dict[int, float],
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = str(record["root"])
    sidecar_path = str(record["sidecar"])
    sidecar = json.loads(Path(sidecar_path).read_text())
    trota_provenance = validate_trota_provenance(sidecar)
    with uproot.open(root_path) as root_file:
        if "Events" not in root_file or "TROTA" not in root_file:
            raise RuntimeError("required Events/TROTA trees are missing")
        tree = root_file["Events"]
        from build_flat_boosted_recoil_hists import WEIGHT_BRANCHES  # noqa: PLC0415

        read_branches = tuple(dict.fromkeys((*SELECTION_BRANCHES, *WEIGHT_BRANCHES)))
        missing = sorted(set(read_branches) - set(tree.keys()))
        optional = {"electron_veto_eta", "electron_medium_eta"}
        hard_missing = [name for name in missing if name not in optional]
        if hard_missing:
            raise RuntimeError("missing flat branches: " + ", ".join(hard_missing))
        arrays = tree.arrays([name for name in read_branches if name in tree.keys()], library="ak")
        blocks, reconstruction_audit = build_region_blocks(arrays)
        eligible = np.zeros(len(arrays), dtype=bool)
        for block in blocks.values():
            eligible |= (
                block.core
                & (block.nb >= 1)
                & (block.nt == 0)
                & (block.nw == 0)
                & (block.met_sqrt_ht >= 10.0)
            )
        nres, nres_audit = compute_nres(arrays, root_file["TROTA"], eligible)

    region_masks = final_region_masks(blocks, nres)
    union = np.zeros(len(arrays), dtype=bool)
    for values in region_masks.values():
        union |= values
    histograms = new_hist_payload()
    source_audit: dict[str, Any] = {
        "root": root_path,
        "sidecar": sidecar_path,
        "events": len(arrays),
        "eligible_events": int(np.count_nonzero(eligible)),
        "selected_union_events": int(np.count_nonzero(union)),
        "selected_by_region": {
            region: int(np.count_nonzero(values)) for region, values in region_masks.items()
        },
        "reconstruction": reconstruction_audit,
        "trota": nres_audit,
        "trota_provenance": trota_provenance,
        "weight_status": {},
        "data_stream_exclusions": {},
    }
    if not np.any(union):
        return histograms, source_audit

    dataset_ids = np.asarray(arrays["dataset_id"], dtype=np.int64)
    for dataset_id in sorted(set(int(value) for value in dataset_ids[union])):
        sidecar_dataset = dataset_record(sidecar, dataset_id)
        dataset = str(sidecar_dataset.get("dataset") or "unknown")
        raw_process = str(sidecar_dataset.get("process") or "unknown")
        process = canonical_process(raw_process, dataset)
        is_signal = bool(sidecar_dataset.get("is_signal"))
        for group_mask in subgroup_masks(arrays, dataset_id, is_signal):
            active = group_mask & union
            if not np.any(active):
                continue
            sub_group = arrays[active]
            local_region_masks = {
                region: values[active] for region, values in region_masks.items()
            }
            weights, status = normalized_weights(
                sub_group,
                sidecar_dataset,
                raw_process,
                dataset,
                manifest,
                signal_norm,
                stop_xsec,
                repo,
                local_region_masks,
            )
            sample = sample_name(sub_group, sidecar_dataset, process)
            source_audit["weight_status"].setdefault(sample, status)
            for region, local_region in local_region_masks.items():
                if sidecar_dataset.get("is_data") and process != DATA_STREAM[region]:
                    excluded = int(np.count_nonzero(local_region))
                    if excluded:
                        source_audit["data_stream_exclusions"][region] = (
                            int(source_audit["data_stream_exclusions"].get(region, 0))
                            + excluded
                        )
                    continue
                if not np.any(local_region):
                    continue
                local_block = RegionBlock(
                    **{
                        field: getattr(blocks[region], field)[active]
                        for field in RegionBlock.__dataclass_fields__
                    }
                )
                categories = category_masks(local_block)
                values = histogram_values(local_block)
                for category, category_mask in categories.items():
                    selected = local_region & category_mask
                    if not np.any(selected):
                        continue
                    for variable, variable_values in values.items():
                        fill_counter(
                            counter_at(
                                histograms,
                                "nominal",
                                region,
                                category,
                                sample,
                                variable,
                            ),
                            variable_values,
                            weights,
                            selected,
                            HISTOGRAMS[variable],
                        )
                        if region in RZ_FACTORS and sample == "DY":
                            nb_key = "Nb1" if category.startswith("Nb1_") else "Nb2plus"
                            fill_counter(
                                counter_at(
                                    histograms,
                                    "nominal_rz",
                                    region,
                                    category,
                                    sample,
                                    variable,
                                ),
                                variable_values,
                                weights * RZ_FACTORS[region][nb_key],
                                selected,
                                HISTOGRAMS[variable],
                            )
    return histograms, source_audit


def merge_histograms(target: dict[str, Any], source: dict[str, Any]) -> None:
    for variation, by_region in source.items():
        for region, by_category in by_region.items():
            for category, by_sample in by_category.items():
                for sample, by_variable in by_sample.items():
                    for variable, counter in by_variable.items():
                        output = counter_at(
                            target, variation, region, category, sample, variable
                        )
                        for name in ("sumw", "sumw2", "entries"):
                            output[name] = (
                                np.asarray(output[name], dtype=np.float64)
                                + np.asarray(counter[name], dtype=np.float64)
                            ).tolist()


def worker(request_path: Path) -> int:
    request = json.loads(request_path.read_text())
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise RuntimeError(f"unsupported request schema: {request.get('schema_version')}")
    manifest = json.loads(Path(request["manifest"]).read_text())
    xsec_payload = json.loads(Path(request["stop_xsec"]).read_text())
    signal_norm = signal_normalization_map(manifest)
    stop_xsec = stop_xsec_map(xsec_payload)
    output = Path(request["output"])
    histograms: dict[str, Any] = {}
    valid_files: list[str] = []
    bad_files: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    started = time.time()
    for record in request["inputs"]:
        try:
            partial, audit = process_source(
                record,
                manifest,
                signal_norm,
                stop_xsec,
                Path(
                    os.environ.get("LOWDM_TEST_REPOSITORY_OVERRIDE")
                    or request["repository"]
                ),
            )
            merge_histograms(histograms, partial)
            source_audits.append(audit)
            valid_files.append(str(record["root"]))
        except Exception as error:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            bad_files.append(
                {
                    "dataset": request["kind"],
                    "file": str(record["root"]),
                    "failure_stage": "lowdm_test_histogram_source",
                    "exception_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "traceback": traceback.format_exc(limit=20)[:8000],
                    "first_failure_time": now,
                    "last_failure_time": now,
                    "alternate_access_attempted": False,
                    "permanently_skipped": True,
                }
            )
    payload = {
        "schema_version": PARTIAL_SCHEMA,
        "status": "complete" if not bad_files else "complete_with_bad_files",
        "kind": request["kind"],
        "batch": int(request["batch"]),
        "selection_contract": selection_contract(),
        "histogram_specs": HISTOGRAMS,
        "input_files_requested": len(request["inputs"]),
        "input_files_valid": len(valid_files),
        "input_files": valid_files,
        "histograms": histograms,
        "source_audits": source_audits,
        "bad_files": bad_files,
        "runtime_seconds": time.time() - started,
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "kind": payload["kind"],
                "batch": payload["batch"],
                "valid": len(valid_files),
                "bad": len(bad_files),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # A reproducibly bad source is recorded and skipped by policy.  The batch
    # itself remains a valid resumable product and must not poison the entire
    # campaign queue.
    return 0


def selection_contract() -> dict[str, Any]:
    return {
        "common": (
            "region-specific broad Low-dM preselection && Nb>=1 && Nt=0 "
            "&& Nw=0 && Nres(TROTA)=0"
        ),
        "categories": list(CATEGORIES),
        "nisr_policy": "NISR=0 versus NISR>=1; all overflow retained",
        "explicitly_not_applied": [
            "MET/sqrt(HT)>=10",
            "deltaPhi(ISR,recoil)>2",
            "NISR==1",
            "!feature_SR",
        ],
        "highdm_overlap_policy": (
            "retained in test histograms; overlapping High-dM first category is "
            "removed only in a later combined-category proposal"
        ),
        "dy": "opposite-sign same-flavor, 81<mll<101 GeV, pTll>200 GeV",
        "data_streams": DATA_STREAM,
        "weights": (
            "compute_weight_bundle nominal correction product times physical-dataset "
            "xsec*lumi/sumw; signal uses stop xsec and mass-point sumw"
        ),
        "rz": RZ_FACTORS,
    }


def make_requests(opts: argparse.Namespace) -> int:
    manifest = json.loads(opts.manifest.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("full campaign manifest is not complete")
    opts.output.mkdir(parents=True, exist_ok=True)
    request_dir = opts.output / "requests"
    partial_dir = opts.output / "partials"
    log_dir = opts.output / "logs"
    for directory in (request_dir, partial_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    queue: list[str] = []
    counts: dict[str, int] = {}
    for kind in ("data", "mc", "signal"):
        records = [item for item in manifest["shards"] if item["kind"] == kind]
        counts[kind] = len(records)
        for batch, start in enumerate(range(0, len(records), opts.files_per_batch)):
            request_path = request_dir / f"{kind}_{batch:04d}.json"
            output_path = partial_dir / f"{kind}_{batch:04d}.json"
            write_json(
                request_path,
                {
                    "schema_version": REQUEST_SCHEMA,
                    "kind": kind,
                    "batch": batch,
                    "inputs": [
                        {"root": item["root"], "sidecar": item["sidecar"]}
                        for item in records[start : start + opts.files_per_batch]
                    ],
                    "manifest": str(opts.manifest),
                    "stop_xsec": str(opts.stop_xsec),
                    "repository": str(opts.repository),
                    "output": str(output_path),
                },
            )
            queue.append(str(request_path))
    queue_path = opts.output / "pending_requests.txt"
    queue_path.write_text("\n".join(queue) + "\n")
    payload = {
        "schema_version": SCHEMA,
        "status": "requests_ready",
        "selection_contract": selection_contract(),
        "manifest": str(opts.manifest),
        "stop_xsec": str(opts.stop_xsec),
        "repository": str(opts.repository),
        "output": str(opts.output),
        "files_per_batch": opts.files_per_batch,
        "source_files": counts,
        "requests": len(queue),
        "queue": str(queue_path),
    }
    write_json(opts.output / "campaign_state.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def collect_partial_paths(path: Path) -> list[Path]:
    return sorted((path / "partials").glob("*.json"))


def merge_campaign(opts: argparse.Namespace) -> int:
    partial_paths = collect_partial_paths(opts.input)
    if not partial_paths:
        raise RuntimeError(f"no partial JSON files found under {opts.input}")
    output_json = opts.output.with_suffix(".json")
    output_root = opts.output.with_suffix(".root")
    source_audit_output = opts.output.with_name(
        opts.output.name + "_source_audits.jsonl.gz"
    )
    source_audit_temporary = source_audit_output.with_name(
        source_audit_output.name + f".tmp.{os.getpid()}"
    )
    histograms: dict[str, Any] = {}
    source_files = 0
    valid_files = 0
    bad_files: list[dict[str, Any]] = []
    source_audit_count = 0
    partial_status: dict[str, int] = {}
    source_audit_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(source_audit_temporary, "wt") as audit_handle:
            for path in partial_paths:
                payload = json.loads(path.read_text())
                if payload.get("schema_version") != PARTIAL_SCHEMA:
                    raise RuntimeError(f"unexpected partial schema in {path}")
                status = str(payload.get("status"))
                partial_status[status] = int(partial_status.get(status, 0)) + 1
                source_files += int(payload["input_files_requested"])
                valid_files += int(payload["input_files_valid"])
                bad_files.extend(payload.get("bad_files") or [])
                for audit in payload.get("source_audits") or []:
                    json.dump(
                        audit,
                        audit_handle,
                        sort_keys=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    audit_handle.write("\n")
                    source_audit_count += 1
                merge_histograms(histograms, payload["histograms"])
        os.replace(source_audit_temporary, source_audit_output)
    finally:
        if source_audit_temporary.exists():
            source_audit_temporary.unlink()

    final = {
        "schema_version": SCHEMA,
        "status": "complete" if not bad_files else "complete_with_bad_files",
        "selection_contract": selection_contract(),
        "histogram_specs": HISTOGRAMS,
        "partials": len(partial_paths),
        "partial_status": partial_status,
        "input_files_requested": source_files,
        "input_files_valid": valid_files,
        "histograms": histograms,
        "source_audits": {
            "format": "jsonl+gzip",
            "path": str(source_audit_output),
            "records": source_audit_count,
        },
        "bad_files": bad_files,
        "root_output": str(output_root),
    }
    write_json(output_json, final)
    write_root_histograms(output_root, histograms)
    summary = campaign_summary(final)
    write_json(opts.output.with_name(opts.output.name + "_summary.json"), summary)
    write_json(
        opts.input / "bad_files.json",
        {
            "schema_version": SCHEMA,
            "bad_file_count": len(bad_files),
            "records": bad_files,
        },
    )
    write_text(
        opts.input / "bad_files.txt",
        "".join(f"{record.get('root', '')}\n" for record in bad_files),
    )
    write_json(
        opts.input / "file_validation_summary.json",
        {
            "schema_version": SCHEMA,
            "status": final["status"],
            "partials": len(partial_paths),
            "partial_status": partial_status,
            "input_files_requested": source_files,
            "input_files_valid": valid_files,
            "bad_file_count": len(bad_files),
            "source_audit_records": source_audit_count,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not bad_files else 2


def write_root_histograms(path: Path, histograms: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path, compression=uproot.LZ4(4)) as root_file:
        for variation, by_region in histograms.items():
            for region, by_category in by_region.items():
                for category, by_sample in by_category.items():
                    for sample, by_variable in by_sample.items():
                        safe_sample = sample.replace("/", "_")
                        for variable, counter in by_variable.items():
                            edges = np.asarray(HISTOGRAMS[variable]["edges"], dtype=np.float64)
                            base = f"{variation}/{region}/{category}/{safe_sample}/{variable}"
                            root_file[base] = (
                                np.asarray(counter["sumw"], dtype=np.float64),
                                edges,
                            )
                            root_file[base + "__sumw2"] = (
                                np.asarray(counter["sumw2"], dtype=np.float64),
                                edges,
                            )
                            root_file[base + "__entries"] = (
                                np.asarray(counter["entries"], dtype=np.float64),
                                edges,
                            )


def campaign_summary(payload: dict[str, Any]) -> dict[str, Any]:
    yields: dict[str, Any] = {}
    for variation, by_region in payload["histograms"].items():
        for region, by_category in by_region.items():
            for category, by_sample in by_category.items():
                for sample, by_variable in by_sample.items():
                    counter = by_variable.get("recoil")
                    if counter is None:
                        continue
                    yields.setdefault(variation, {}).setdefault(region, {}).setdefault(
                        category, {}
                    )[sample] = {
                        "yield": float(np.sum(counter["sumw"])),
                        "sumw2": float(np.sum(counter["sumw2"])),
                        "entries": int(round(float(np.sum(counter["entries"])))),
                    }
    return {
        "schema_version": SCHEMA,
        "status": payload["status"],
        "selection_contract": payload["selection_contract"],
        "partials": payload["partials"],
        "input_files_requested": payload["input_files_requested"],
        "input_files_valid": payload["input_files_valid"],
        "bad_file_count": len(payload["bad_files"]),
        "yields": yields,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument("--worker", type=Path)
    action.add_argument("--make-requests", action="store_true")
    action.add_argument("--merge", action="store_true")
    result.add_argument(
        "--manifest",
        type=Path,
        default=OUTER_PROJECT / "gnn_lowdm" / "full_campaign_manifest_2024.json",
    )
    result.add_argument(
        "--stop-xsec",
        type=Path,
        default=OUTER_PROJECT / "signals" / "stop_xsec_13p6TeV.json",
    )
    result.add_argument("--repository", type=Path, default=REPOSITORY)
    result.add_argument("--output", type=Path)
    result.add_argument("--input", type=Path)
    result.add_argument("--files-per-batch", type=int, default=5)
    return result


def main() -> int:
    raise SystemExit(
        "region_io is an internal library; use the public train/eval commands"
    )


if __name__ == "__main__":
    raise SystemExit(main())
