#!/usr/bin/env python3
"""Build the High-dM DY matrix and Low-dM sparse-recovery inventory.

The input is the current decorated nominal feature ROOT format. High-dM is
exact at this stage. Low-dM candidates whose lepton-cleaned AK8 topology
cannot be resolved from the feature table are recorded for sparse NanoAOD
recovery.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from ..real_subset_worker import (
    assign_lowdm_search_bin,
    compute_weight_bundle,
)
from .model import (
    CHANNELS,
    MASS_WINDOWS,
    MLL_EDGES,
    add_yield,
    finalize_rz,
    fill_histogram,
    merge_tree,
    nested_histogram,
    nested_yield,
)

REQUIRED_BRANCHES = {
    "dataset_id",
    "run",
    "luminosityBlock",
    "event",
    "entry",
    "year",
    "gen_weight",
    "pu_ntrueint",
    "feature_GCR",
    "pass_base_common",
    "pass_zero_tau",
    "pass_electron_trigger",
    "pass_muon_trigger",
    "pass_lowdm_topology_veto",
    "pass_dy2e_open_high",
    "pass_dy2m_open_high",
    "n_e_veto",
    "n_e_medium",
    "n_m_loose",
    "n_m_medium",
    "njet_lepton_clean",
    "nb_lepton_clean",
    "ht_lepton_clean",
    "mee",
    "pee",
    "mmm",
    "pmm",
    "recoil_dy2e",
    "recoil_dy2m",
    "recoil_dy2e_phi",
    "recoil_dy2m_phi",
    "good_jet_pt",
    "good_jet_eta",
    "good_jet_phi",
    "good_jet_hadron_flavour",
    "good_jet_b_medium",
    "good_jet_btag_upart",
    "lowdm_fatjet_pt",
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
    "electron_veto_pt",
    "electron_veto_eta_sc",
    "electron_veto_phi",
    "electron_medium_pt",
    "electron_medium_eta_sc",
    "electron_medium_phi",
    "muon_loose_pt",
    "muon_loose_eta",
    "muon_loose_phi",
    "muon_medium_pt",
    "muon_medium_eta",
    "muon_medium_phi",
    "photon_medium_pt",
    "photon_medium_eta",
    "photon_medium_phi",
    "gen_top_pt",
    "file_id",
    "electron_pt_all",
    "electron_eta_all",
    "electron_phi_all",
    "electron_mini_iso_all",
    "electron_charge_all",
    "electron_cutbased_all",
    "muon_pt_all",
    "muon_eta_all",
    "muon_phi_all",
    "muon_mini_iso_all",
    "muon_charge_all",
    "muon_medium_id_all",
}

OPTIONAL_BRANCHES = {"electron_veto_eta", "electron_medium_eta"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    """Convert a flat branch to finite floats with a deterministic fallback."""

    try:
        output = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, fill, dtype=float)
    if output.shape == ():
        output = np.full(n, float(output), dtype=float)
    if len(output) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(output), output, fill)


def bool_field(
    chunk: dict[str, Any],
    name: str,
    n: int,
    default: bool = False,
) -> np.ndarray:
    if name not in chunk:
        return np.full(n, default, dtype=bool)
    try:
        output = np.asarray(chunk[name], dtype=bool)
    except Exception:
        return np.full(n, default, dtype=bool)
    if output.shape == ():
        output = np.full(n, bool(output), dtype=bool)
    if len(output) != n:
        return np.full(n, default, dtype=bool)
    return output


def float_field(
    chunk: dict[str, Any],
    name: str,
    n: int,
    fill: float = 0.0,
) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=float)
    return finite_array(chunk[name], n, fill)


def int_field(
    chunk: dict[str, Any],
    name: str,
    n: int,
    fill: int = 0,
) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=int)
    try:
        output = np.asarray(chunk[name], dtype=int)
    except Exception:
        return np.full(n, fill, dtype=int)
    if output.shape == ():
        output = np.full(n, int(output), dtype=int)
    if len(output) != n:
        return np.full(n, fill, dtype=int)
    return output


def ones_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.ones_like(jagged), np.bool_)


def zeros_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.zeros_like(jagged), np.bool_)


def combine_two(first: Any, second: Any) -> Any:
    return ak.concatenate([first, second], axis=1)


def flat_arrays_for_weights(
    chunk: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt decorated-feature branches to the nominal weight interface."""

    n = len(chunk["gen_weight"])
    jet_pt = chunk["good_jet_pt"]
    jet_eta = chunk["good_jet_eta"]
    jet_had = chunk["good_jet_hadron_flavour"]
    b_medium = ak.values_astype(
        chunk["good_jet_b_medium"], np.bool_
    )

    electron_veto_pt = chunk["electron_veto_pt"]
    electron_veto_eta_sc = chunk["electron_veto_eta_sc"]
    electron_veto_phi = chunk["electron_veto_phi"]
    electron_medium_pt = chunk["electron_medium_pt"]
    electron_medium_eta_sc = chunk["electron_medium_eta_sc"]
    electron_medium_phi = chunk["electron_medium_phi"]
    raw_eta_available = (
        "electron_veto_eta" in chunk
        and "electron_medium_eta" in chunk
    )
    electron_veto_eta = (
        chunk["electron_veto_eta"]
        if raw_eta_available
        else electron_veto_eta_sc
    )
    electron_medium_eta = (
        chunk["electron_medium_eta"]
        if raw_eta_available
        else electron_medium_eta_sc
    )
    electron_pt = combine_two(electron_veto_pt, electron_medium_pt)
    electron_eta = combine_two(
        electron_veto_eta, electron_medium_eta
    )
    electron_eta_sc = combine_two(
        electron_veto_eta_sc, electron_medium_eta_sc
    )
    electron_phi = combine_two(
        electron_veto_phi, electron_medium_phi
    )
    electron_veto = combine_two(
        ones_mask(electron_veto_pt), zeros_mask(electron_medium_pt)
    )
    electron_medium = combine_two(
        zeros_mask(electron_veto_pt), ones_mask(electron_medium_pt)
    )

    muon_loose_pt = chunk["muon_loose_pt"]
    muon_medium_pt = chunk["muon_medium_pt"]
    muon_pt = combine_two(muon_loose_pt, muon_medium_pt)
    muon_eta = combine_two(
        chunk["muon_loose_eta"], chunk["muon_medium_eta"]
    )
    muon_phi = combine_two(
        chunk["muon_loose_phi"], chunk["muon_medium_phi"]
    )
    muon_loose = combine_two(
        ones_mask(muon_loose_pt), zeros_mask(muon_medium_pt)
    )
    muon_medium = combine_two(
        zeros_mask(muon_loose_pt), ones_mask(muon_medium_pt)
    )

    photon_pt = chunk["photon_medium_pt"]
    top_pt = chunk["gen_top_pt"]
    top_flags = ak.values_astype(
        ak.ones_like(top_pt) * ((1 << 8) | (1 << 13)), np.int64
    )
    top_pdg = ak.values_astype(ak.ones_like(top_pt) * 6, np.int64)
    arrays = ak.Array(
        {
            "genWeight": chunk["gen_weight"],
            "Pileup_nTrueInt": chunk["pu_ntrueint"],
            "Jet_hadronFlavour": jet_had,
            "GenPart_pt": top_pt,
            "GenPart_pdgId": top_pdg,
            "GenPart_statusFlags": top_flags,
        }
    )
    inputs = {
        "n": n,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_hadflav": jet_had,
        "b_med": b_medium,
        "e_eta": electron_eta,
        "e_delta_eta_sc": electron_eta_sc - electron_eta,
        "e_pt": electron_pt,
        "e_phi": electron_phi,
        "e_veto": electron_veto,
        "e_med": electron_medium,
        "n_e_veto": np.asarray(chunk["n_e_veto"], dtype=int),
        "n_e_med": np.asarray(chunk["n_e_medium"], dtype=int),
        "m_eta": muon_eta,
        "m_pt": muon_pt,
        "m_phi": muon_phi,
        "m_loose": muon_loose,
        "m_med": muon_medium,
        "n_m_loose": np.asarray(chunk["n_m_loose"], dtype=int),
        "n_m_med": np.asarray(chunk["n_m_medium"], dtype=int),
        "p_eta": chunk["photon_medium_eta"],
        "p_pt": photon_pt,
        "p_phi": chunk["photon_medium_phi"],
        "p_med": ones_mask(photon_pt),
        "gcr_mask": bool_field(chunk, "feature_GCR", n),
    }
    return arrays, inputs


def canonical_process(process: str, dataset: str) -> str:
    """Normalize only process aliases needed by the Run-3 weight code."""

    if process in {"JetMET", "EGamma", "Muon"}:
        return process
    if process == "ST" or dataset.startswith(
        ("TW", "TbarW", "TBbar", "TbarB")
    ):
        return "ST"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "TT"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "WtoLNu"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Zto2Nu"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "GJ"
    return process or "other"


def dataset_label(
    metadata: dict[str, Any], dataset_id: int
) -> tuple[str, str, bool, bool]:
    record = (metadata.get("datasets") or {}).get(str(int(dataset_id))) or {}
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process") or "unknown")
    is_data = bool(record.get("is_data"))
    is_signal = bool(record.get("is_signal"))
    if not is_data and not is_signal:
        process = canonical_process(process, dataset)
    return dataset, process, is_data, is_signal


def dy_dataset_allowed(dataset: str, process: str) -> bool:
    """Reject legacy pT-binned DY when using inclusive 2024 DY samples."""

    if process != "DY":
        return True
    return dataset.startswith(
        (
            "DYto2E-4Jets_Bin-MLL-50",
            "DYto2Mu-4Jets_Bin-MLL-50",
            "DYto2Tau-4Jets_Bin-MLL-50",
        )
    )


def norm_vector(
    normalization: dict[str, Any],
    dataset_id: int,
    n: int,
) -> np.ndarray:
    factor = (
        (normalization.get("dataset_factors") or {})
        .get(str(int(dataset_id)), {})
        .get("normalization_factor")
    )
    try:
        valid = factor is not None and math.isfinite(float(factor))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid or float(factor) <= 0.0:
        raise RuntimeError(
            "missing, non-finite, or non-positive normalization factor "
            f"for dataset_id={int(dataset_id)}"
        )
    return np.full(n, float(factor), dtype=float)



def first_two(values: Any, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    padded = ak.pad_none(values, 2, axis=1, clip=True)
    first = np.asarray(ak.fill_none(padded[:, 0], fill))
    second = np.asarray(ak.fill_none(padded[:, 1], fill))
    return first, second


def zll_sample(dataset: str, process: str) -> bool:
    """Mirror the AN's sample-level Z->ll/other partition.

    The new inclusive VVV/TTVV samples are assigned to the Z-containing side
    when their sample name contains a Z.  This is auditable and conservative;
    the sparse-Nano finalizer records their tiny contribution separately.
    """

    text = str(dataset)
    if process == "DY":
        return True
    z_tokens = (
        "TTZ",
        "WZ",
        "ZZ",
        "WWZ",
        "WZZ",
        "ZZZ",
        "WZG",
    )
    return any(token in text for token in z_tokens)


def medium_charge_pair(
    chunk: dict[str, Any],
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if channel == "DY2E":
        pt = chunk["electron_pt_all"]
        eta = chunk["electron_eta_all"]
        iso = chunk["electron_mini_iso_all"]
        cut = chunk["electron_cutbased_all"]
        charge = chunk["electron_charge_all"]
        fiducial = (abs(eta) < 1.4442) | (
            (abs(eta) > 1.5660) & (abs(eta) < 2.5)
        )
        medium = (pt > 10.0) & fiducial & (cut >= 3) & (iso < 0.1)
    else:
        pt = chunk["muon_pt_all"]
        eta = chunk["muon_eta_all"]
        iso = chunk["muon_mini_iso_all"]
        # The decorated flat tree stores NanoAOD boolean ID branches as
        # integer 0/1 vectors.  Cast explicitly before jagged masking;
        # otherwise Awkward interprets the mask as integer indices and the
        # reconstructed charge pair is wrong for most dimuon events.
        medium_id = chunk["muon_medium_id_all"] != 0
        charge = chunk["muon_charge_all"]
        medium = (
            (pt > 10.0)
            & (abs(eta) < 2.4)
            & medium_id
            & (iso < 0.2)
        )
    selected_charge = charge[medium]
    first, second = first_two(selected_charge, 0)
    return (
        np.asarray(ak.sum(medium, axis=1), dtype=int),
        np.asarray(first, dtype=int),
        np.asarray(second, dtype=int),
    )


def delta_phi(first: Any, second: Any) -> Any:
    return abs(
        np.arctan2(
            np.sin(first - second),
            np.cos(first - second),
        )
    )


def clean_by_delta_r(
    object_eta: Any,
    object_phi: Any,
    lepton_eta: Any,
    lepton_phi: Any,
    threshold: float,
) -> Any:
    deta = object_eta[:, :, None] - lepton_eta[:, None, :]
    dphi = delta_phi(
        object_phi[:, :, None],
        lepton_phi[:, None, :],
    )
    return ak.all(
        deta * deta + dphi * dphi > threshold * threshold,
        axis=2,
    )


def lowdm_feature_decision(
    chunk: dict[str, Any],
    channel: str,
    common: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact feature decisions and the topology-ambiguous subset.

    All adopted Low-dM observables except the boosted-top/W veto are exactly
    reconstructible from the stored corrected object vectors.  If the
    nominal topology veto failed and an AK8 jet overlaps a selected lepton,
    the event is deferred to sparse NanoAOD.  All other decisions are exact.
    """

    n = len(chunk["dataset_id"])
    if channel == "DY2E":
        electron_fiducial = (
            (abs(chunk["electron_eta_all"]) < 1.4442)
            | (
                (abs(chunk["electron_eta_all"]) > 1.5660)
                & (abs(chunk["electron_eta_all"]) < 2.5)
            )
        )
        medium = (
            (chunk["electron_pt_all"] > 10.0)
            & electron_fiducial
            & (chunk["electron_cutbased_all"] >= 3)
            & (chunk["electron_mini_iso_all"] < 0.1)
        )
        lepton_eta = chunk["electron_eta_all"][medium]
        lepton_phi = chunk["electron_phi_all"][medium]
        recoil = float_field(chunk, "recoil_dy2e", n, -99.0)
        recoil_phi = float_field(
            chunk, "recoil_dy2e_phi", n, -99.0
        )
    else:
        medium = (
            (chunk["muon_pt_all"] > 10.0)
            & (abs(chunk["muon_eta_all"]) < 2.4)
            & (chunk["muon_medium_id_all"] != 0)
            & (chunk["muon_mini_iso_all"] < 0.2)
        )
        lepton_eta = chunk["muon_eta_all"][medium]
        lepton_phi = chunk["muon_phi_all"][medium]
        recoil = float_field(chunk, "recoil_dy2m", n, -99.0)
        recoil_phi = float_field(
            chunk, "recoil_dy2m_phi", n, -99.0
        )

    jet_clean = clean_by_delta_r(
        chunk["good_jet_eta"],
        chunk["good_jet_phi"],
        lepton_eta,
        lepton_phi,
        0.2,
    )
    jet_pt = chunk["good_jet_pt"][jet_clean]
    jet_phi = chunk["good_jet_phi"][jet_clean]
    jet_btag = chunk["good_jet_btag_upart"][jet_clean]
    jet_medium = ak.values_astype(
        chunk["good_jet_b_medium"][jet_clean], np.bool_
    )
    njet = np.asarray(ak.num(jet_pt, axis=1), dtype=int)
    nb = np.asarray(ak.sum(jet_medium, axis=1), dtype=int)
    ht = np.asarray(ak.sum(jet_pt, axis=1), dtype=float)
    jet_dphi = delta_phi(jet_phi, recoil_phi)
    j1, j2 = first_two(jet_dphi, 999.0)
    j3 = np.asarray(
        ak.fill_none(
            ak.pad_none(jet_dphi, 3, axis=1, clip=True)[:, 2],
            999.0,
        ),
        dtype=float,
    )
    open_pre = (j1 > 0.5) & (j2 > 0.15) & (j3 > 0.15)

    btag_order = ak.argsort(jet_btag[jet_medium], axis=1, ascending=False)
    b_pt = jet_pt[jet_medium][btag_order]
    ptb = np.asarray(
        ak.fill_none(
            ak.pad_none(b_pt, 1, axis=1, clip=True)[:, 0],
            -99.0,
        ),
        dtype=float,
    )

    fatjet_clean = clean_by_delta_r(
        chunk["lowdm_fatjet_eta"],
        chunk["lowdm_fatjet_phi"],
        lepton_eta,
        lepton_phi,
        0.4,
    )
    fatjet_pt = chunk["lowdm_fatjet_pt"][fatjet_clean]
    fatjet_phi = chunk["lowdm_fatjet_phi"][fatjet_clean]
    n_isr = np.asarray(ak.num(fatjet_pt, axis=1), dtype=int)
    isr_pt = np.asarray(
        ak.fill_none(
            ak.pad_none(fatjet_pt, 1, axis=1, clip=True)[:, 0],
            -99.0,
        ),
        dtype=float,
    )
    isr_phi = np.asarray(
        ak.fill_none(
            ak.pad_none(fatjet_phi, 1, axis=1, clip=True)[:, 0],
            -99.0,
        ),
        dtype=float,
    )
    pass_isr = (n_isr == 1) & (
        delta_phi(isr_phi, recoil_phi) > 2.0
    )
    met_sqrt_ht = np.divide(
        recoil,
        np.sqrt(ht),
        out=np.full(n, -99.0, dtype=float),
        where=ht > 0.0,
    )
    known = (
        common
        & (njet >= 2)
        & (nb >= 1)
        & (ht > 300.0)
        & open_pre
        & pass_isr
        & (met_sqrt_ht >= 10.0)
    )
    nominal_topology = bool_field(
        chunk, "pass_lowdm_topology_veto", n
    )
    has_removed_fatjet = np.asarray(
        ak.any(~fatjet_clean, axis=1), dtype=bool
    )
    exact_selected = known & nominal_topology
    ambiguous = known & (~nominal_topology) & has_removed_fatjet
    indices = np.full(n, -1, dtype=int)
    for event_index in np.flatnonzero(exact_selected):
        raw_index = assign_lowdm_search_bin(
            int(njet[event_index]),
            int(nb[event_index]),
            -1,
            float(isr_pt[event_index]),
            float(ptb[event_index]),
            float(recoil[event_index]),
            float("nan"),
        )
        if raw_index >= 8:
            indices[event_index] = raw_index - 8
    exact_selected &= indices >= 0
    return exact_selected, ambiguous, indices


def channel_masks(
    chunk: dict[str, Any],
    channel: str,
) -> dict[str, np.ndarray]:
    n = len(chunk["dataset_id"])
    rebuilt_count, first_charge, second_charge = medium_charge_pair(
        chunk, channel
    )
    if channel == "DY2E":
        stored_count = int_field(chunk, "n_e_medium", n, -1)
        opposite_count = int_field(chunk, "n_m_loose", n, -1)
        leading, subleading = first_two(chunk["electron_medium_pt"], -99.0)
        trigger = bool_field(chunk, "pass_electron_trigger", n)
        mass = float_field(chunk, "mee", n, -99.0)
        ptll = float_field(chunk, "pee", n, -99.0)
        recoil = float_field(chunk, "recoil_dy2e", n, -99.0)
        open_high = bool_field(chunk, "pass_dy2e_open_high", n)
        leading_min = 40.0
    else:
        stored_count = int_field(chunk, "n_m_medium", n, -1)
        opposite_count = int_field(chunk, "n_e_veto", n, -1)
        leading, subleading = first_two(chunk["muon_medium_pt"], -99.0)
        trigger = bool_field(chunk, "pass_muon_trigger", n)
        mass = float_field(chunk, "mmm", n, -99.0)
        ptll = float_field(chunk, "pmm", n, -99.0)
        recoil = float_field(chunk, "recoil_dy2m", n, -99.0)
        open_high = bool_field(chunk, "pass_dy2m_open_high", n)
        leading_min = 50.0
    mismatch = rebuilt_count != stored_count
    if np.any(mismatch):
        raise RuntimeError(
            f"{channel}: rebuilt/stored medium-lepton counts differ for "
            f"{int(np.sum(mismatch))} events"
        )
    common = (
        bool_field(chunk, "pass_base_common", n)
        & trigger
        & bool_field(chunk, "pass_zero_tau", n)
        & (opposite_count == 0)
        & (stored_count == 2)
        & (leading > leading_min)
        & (subleading > 20.0)
        & (first_charge != second_charge)
        & (ptll > 200.0)
        & (recoil > 250.0)
        & (int_field(chunk, "njet_lepton_clean", n, -1) >= 2)
        & (float_field(chunk, "ht_lepton_clean", n, -99.0) > 300.0)
    )
    on_z = common & (mass > 81.0) & (mass < 101.0)
    off_z = common & (
        ((mass > 50.0) & (mass < 81.0)) | (mass > 101.0)
    )
    high = (
        common
        & (int_field(chunk, "njet_lepton_clean", n, -1) >= 5)
        & (int_field(chunk, "nb_lepton_clean", n, -1) >= 1)
        & open_high
    )
    sparse_low = (
        common
        & (int_field(chunk, "nb_lepton_clean", n, -1) >= 1)
        & (on_z | off_z)
    )
    return {
        "common": common,
        "on": on_z,
        "off": off_z,
        "high": high,
        "sparse_low": sparse_low,
        "mass": mass,
        "recoil": recoil,
    }


def process_root(
    root_name: str,
    repo_name: str,
    normalization_name: str,
    step_size: int,
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, Any]:
    root_path = Path(root_name)
    repo = Path(repo_name)
    normalization = read_json(Path(normalization_name))
    result: dict[str, Any] = {
        "rz_high": {},
        "rz_low_feature": {},
        "mll_high": {},
        "mll_low_feature": {},
        "sparse_low_candidates": {},
        "summary": {
            "input_root": str(root_path),
            "events_scanned": 0,
            "datasets": {},
            "candidate_events": 0,
        },
    }
    sidecar = root_path.with_suffix(".json")
    if not root_path.exists() or not sidecar.exists():
        result["summary"]["missing"] = True
        return result
    metadata = read_json(sidecar)
    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        present = set(tree.keys())
        missing = sorted(REQUIRED_BRANCHES - present)
        if missing:
            raise RuntimeError(
                f"{root_path}: required AN measurement branches absent: "
                + ",".join(missing)
            )
        branches = sorted(
            REQUIRED_BRANCHES | (OPTIONAL_BRANCHES & present)
        )
        iterator = tree.iterate(
            branches, step_size=step_size, library="ak"
        )
        for chunk in iterator:
            n_chunk = len(chunk["dataset_id"])
            result["summary"]["events_scanned"] += n_chunk
            dataset_ids = np.asarray(chunk["dataset_id"], dtype=np.int64)
            for dataset_id in sorted(set(int(value) for value in dataset_ids)):
                selected_dataset = dataset_ids == dataset_id
                sub = {
                    name: chunk[name][selected_dataset]
                    for name in ak.fields(chunk)
                }
                dataset, process, is_data, is_signal = dataset_label(
                    metadata, dataset_id
                )
                if is_signal or not dy_dataset_allowed(dataset, process):
                    continue
                n = len(sub["dataset_id"])
                if not n:
                    continue
                result["summary"]["datasets"][dataset] = (
                    int(result["summary"]["datasets"].get(dataset, 0)) + n
                )

                # Weight evaluation is the dominant cost and is unnecessary
                # for events that cannot enter the dilepton matrix. All predicates
                # below use already stored, corrected feature quantities.
                # The exact same masks are rebuilt after this lossless
                # prefilter, before any yield is filled.
                interest = np.zeros(n, dtype=bool)
                pre_stream_ok = {
                    "DY2E": (not is_data) or process == "EGamma",
                    "DY2M": (not is_data) or process == "Muon",
                }
                for pre_channel in channels:
                    if not pre_stream_ok[pre_channel]:
                        continue
                    pre_masks = channel_masks(sub, pre_channel)
                    pre_low, pre_ambiguous, _pre_indices = (
                        lowdm_feature_decision(
                            sub,
                            pre_channel,
                            pre_masks["common"],
                        )
                    )
                    in_mass_window = pre_masks["on"] | pre_masks["off"]
                    interest |= (
                        pre_masks["high"] | pre_low | pre_ambiguous
                    ) & in_mass_window
                if not np.any(interest):
                    continue
                sub = {
                    name: sub[name][interest]
                    for name in ak.fields(sub)
                }
                n = len(sub["dataset_id"])
                if is_data:
                    weights = np.ones(n, dtype=float)
                    component = "data"
                else:
                    arrays, inputs = flat_arrays_for_weights(sub)
                    years = np.asarray(sub["year"], dtype=int)
                    year = str(int(years[0])) if len(years) else "2024"
                    norm = norm_vector(normalization, dataset_id, n)
                    _generator, variations, status = compute_weight_bundle(
                        arrays,
                        repo,
                        dataset,
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
                    )
                    btag = (status.get("components") or {}).get("btagSF") or {}
                    if not btag.get("applied"):
                        raise RuntimeError(
                            f"{root_path}: btagSF missing for {dataset}"
                        )
                    weights = finite_array(
                        variations["nominal"], n, 0.0
                    ) * norm
                    component = "zll" if zll_sample(dataset, process) else "other"
                stream_ok = {
                    "DY2E": (not is_data) or process == "EGamma",
                    "DY2M": (not is_data) or process == "Muon",
                }
                for channel in channels:
                    if not stream_ok[channel]:
                        continue
                    masks = channel_masks(sub, channel)
                    low_selected, low_ambiguous, low_indices = (
                        lowdm_feature_decision(
                            sub,
                            channel,
                            masks["common"],
                        )
                    )
                    nb = int_field(sub, "nb_lepton_clean", n, -1)
                    groups = {"Nb1": nb == 1, "Nb2plus": nb >= 2}
                    for group, group_mask in groups.items():
                        fill_histogram(
                            nested_histogram(
                                result["mll_high"],
                                (channel, group, component),
                                MLL_EDGES,
                            ),
                            masks["mass"],
                            weights,
                            masks["high"]
                            & group_mask
                            & (masks["mass"] > 50.0),
                            MLL_EDGES,
                        )
                        for window in MASS_WINDOWS:
                            selected = (
                                masks["high"]
                                & masks[window]
                                & group_mask
                            )
                            if np.any(selected):
                                add_yield(
                                    nested_yield(
                                        result["rz_high"],
                                        (
                                            channel,
                                            group,
                                            window,
                                            component,
                                        ),
                                    ),
                                    weights[selected],
                                )
                    for group, group_mask in {
                        "Nb1": low_indices < 16,
                        "Nb2plus": low_indices >= 16,
                    }.items():
                        fill_histogram(
                            nested_histogram(
                                result["mll_low_feature"],
                                (channel, group, component),
                                MLL_EDGES,
                            ),
                            masks["mass"],
                            weights,
                            low_selected
                            & group_mask
                            & (masks["mass"] > 50.0),
                            MLL_EDGES,
                        )
                        for window in MASS_WINDOWS:
                            selected = (
                                low_selected
                                & masks[window]
                                & group_mask
                            )
                            if np.any(selected):
                                add_yield(
                                    nested_yield(
                                        result["rz_low_feature"],
                                        (
                                            channel,
                                            group,
                                            window,
                                            component,
                                        ),
                                    ),
                                    weights[selected],
                                )
                    sparse = (
                        low_ambiguous
                        & masks["sparse_low"]
                    )
                    if np.any(sparse):
                        for index in np.flatnonzero(sparse):
                            file_id = str(int(sub["file_id"][index]))
                            entry = int(sub["entry"][index])
                            record = {
                                "entry": entry,
                                "channel": channel,
                                "mass_window": (
                                    "on"
                                    if bool(masks["on"][index])
                                    else "off"
                                ),
                                "mass": float(masks["mass"][index]),
                                "flat_recoil": float(
                                    masks["recoil"][index]
                                ),
                                "run": int(sub["run"][index]),
                                "luminosityBlock": int(
                                    sub["luminosityBlock"][index]
                                ),
                                "event": int(sub["event"][index]),
                                "dataset": dataset,
                                "process": process,
                                "component": component,
                                "flat_weight": float(weights[index]),
                            }
                            result["sparse_low_candidates"].setdefault(
                                file_id, []
                            ).append(record)
                            result["summary"]["candidate_events"] += 1

    return result



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=100000)
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=CHANNELS,
        default=list(CHANNELS),
        help="Dilepton channel(s) to process; use DY2E for the first staged run.",
    )
    args = parser.parse_args(argv)
    channels = tuple(dict.fromkeys(args.channels))

    roots = [
        line.strip()
        for line in args.input_list.read_text().splitlines()
        if line.strip()
    ]
    if len(roots) != len(set(roots)):
        raise SystemExit("duplicate ROOT paths in input list")
    merged: dict[str, Any] = {
        "schema_version": "dy_estimation_feature_2024_v1",
        "status": "running",
        "rz_high_raw": {},
        "rz_low_feature_raw": {},
        "mll_high": {},
        "mll_low_feature": {},
        "sparse_low_candidates": {},
        "summary": {
            "input_roots": len(roots),
            "completed_roots": 0,
            "missing_roots": [],
            "events_scanned": 0,
            "candidate_events": 0,
            "datasets": {},
        },
        "provenance": {
            "normalization_sha256": file_sha256(args.normalization),
            "dy_dataset_policy": "inclusive_dyto2e_mu_tau_4jets_mll50",
            "jobs": args.jobs,
            "step_size": args.step_size,
            "channels": list(channels),
            "zll_partition": (
                "DY plus sample names containing TTZ/WZ/ZZ/WWZ/WZZ/ZZZ/WZG"
            ),
        },
    }
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.jobs)
    ) as executor:
        futures = {
            executor.submit(
                process_root,
                root,
                str(args.repo),
                str(args.normalization),
                args.step_size,
                channels,
            ): root
            for root in roots
        }
        for future in concurrent.futures.as_completed(futures):
            root = futures[future]
            result = future.result()
            summary = result["summary"]
            if summary.get("missing"):
                merged["summary"]["missing_roots"].append(root)
            else:
                merged["summary"]["completed_roots"] += 1
            merged["summary"]["events_scanned"] += int(
                summary.get("events_scanned", 0)
            )
            merged["summary"]["candidate_events"] += int(
                summary.get("candidate_events", 0)
            )
            for dataset, count in (summary.get("datasets") or {}).items():
                merged["summary"]["datasets"][dataset] = (
                    int(merged["summary"]["datasets"].get(dataset, 0))
                    + int(count)
                )
            merge_tree(merged["rz_high_raw"], result["rz_high"])
            merge_tree(
                merged["rz_low_feature_raw"],
                result["rz_low_feature"],
            )
            merge_tree(merged["mll_high"], result["mll_high"])
            merge_tree(
                merged["mll_low_feature"],
                result["mll_low_feature"],
            )
            for file_id, records in result[
                "sparse_low_candidates"
            ].items():
                merged["sparse_low_candidates"].setdefault(
                    file_id, []
                ).extend(records)
            completed = int(merged["summary"]["completed_roots"])
            if completed and completed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed_roots": completed,
                            "total_roots": len(roots),
                            "events_scanned": merged["summary"][
                                "events_scanned"
                            ],
                            "candidate_events": merged["summary"][
                                "candidate_events"
                            ],
                        }
                    ),
                    flush=True,
                )
    for records in merged["sparse_low_candidates"].values():
        records.sort(key=lambda item: (item["entry"], item["channel"]))
    merged["rz_high"] = finalize_rz(merged["rz_high_raw"])
    merged["rz_low_feature"] = finalize_rz(
        merged["rz_low_feature_raw"]
    )
    merged["status"] = (
        "feature_stage_complete"
        if not merged["summary"]["missing_roots"]
        else "feature_stage_complete_with_missing_roots"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, sort_keys=True, separators=(",", ":"))
    )
    print(
        json.dumps(
            {
                "status": merged["status"],
                "output": str(args.output),
                "completed_roots": merged["summary"]["completed_roots"],
                "candidate_events": merged["summary"]["candidate_events"],
                "candidate_files": len(merged["sparse_low_candidates"]),
                "rz_high": merged["rz_high"]["combined"],
                "rz_low_feature": merged["rz_low_feature"]["combined"],
            }
        )
    )
    return 0 if not merged["summary"]["missing_roots"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
