#!/usr/bin/env python3
"""Build Run-3 inputs for the AN-style Z->nunu background estimate.

The dilepton control sample is deliberately kept separate from the final
likelihood.  It measures the normalization factors R_Z and R_T from the
on-Z/off-Z matrix.  The photon control sample supplies the Q normalization
and the bin-wise S_gamma shape information used by the simultaneous fit.

This first-stage builder consumes the broad nominal feature ROOT files.  It
also writes the exact sparse NanoAOD entry inventory needed to recompute the
Low-dM dilepton topology with lepton-cleaned AK8 jets.  High-dM R_Z is exact at
this stage; Low-dM R_Z is finalized by the sparse-Nano follow-up.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import awkward as ak
import numpy as np
import uproot
from scipy.optimize import minimize

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_flat_boosted_recoil_hists as bh  # noqa: E402


HIGH_EDGES = np.asarray(bh.RECOIL_PT_BINS, dtype=float)
RZ_HIGH_UT_EDGES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1500.0],
    dtype=float,
)
RZ_LOW_UT_EDGES = np.asarray(
    [300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1500.0],
    dtype=float,
)
MLL_EDGES = np.asarray(
    [50.0, 70.0, 81.0, 91.0, 101.0, 120.0, 160.0, 250.0, 500.0],
    dtype=float,
)
HIGH_GROUPS = ("Nb1", "Nb2plus")
CHANNELS = ("DY2E", "DY2M")
MASS_WINDOWS = ("on", "off")
COMPONENTS = ("data", "zll", "other")
EXTRA_BRANCHES = {
    "file_id",
    "feature_flat_preselection",
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
    "pass_dy2e_open_high",
    "pass_dy2m_open_high",
    "recoil_dy2e_phi",
    "recoil_dy2m_phi",
    "good_jet_phi",
    "good_jet_btag_upart",
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_yield() -> dict[str, Any]:
    return {"sumw": 0.0, "sumw2": 0.0, "entries": 0}


def add_yield(target: dict[str, Any], weights: np.ndarray) -> None:
    selected = np.asarray(weights, dtype=float)
    selected = selected[np.isfinite(selected)]
    target["sumw"] = float(target["sumw"]) + float(np.sum(selected))
    target["sumw2"] = float(target["sumw2"]) + float(
        np.sum(selected * selected)
    )
    target["entries"] = int(target["entries"]) + int(len(selected))


def nested_yield(
    payload: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    target = payload
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    return target.setdefault(keys[-1], empty_yield())


def fill_rz_ut_yields(
    target: dict[str, Any],
    channel: str,
    component: str,
    base_mask: np.ndarray,
    masks: dict[str, np.ndarray],
    weights: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Fill exclusive UT-bin inputs for the on/off-Z matrix.

    The final bin contains the overflow, matching the nominal recoil
    histogram convention used by the analysis plots.
    """

    recoil = np.asarray(masks["recoil"], dtype=float)
    valid = np.asarray(base_mask, dtype=bool) & np.isfinite(recoil)
    if not np.any(valid):
        return
    indices = np.searchsorted(edges, recoil, side="right") - 1
    indices = np.minimum(indices, len(edges) - 2)
    valid &= indices >= 0
    for bin_index in range(len(edges) - 1):
        ut_bin = valid & (indices == bin_index)
        for window in MASS_WINDOWS:
            selected = ut_bin & masks[window]
            if np.any(selected):
                add_yield(
                    nested_yield(
                        target,
                        (channel, str(bin_index), window, component),
                    ),
                    weights[selected],
                )


def empty_histogram(edges: np.ndarray) -> dict[str, Any]:
    size = len(edges) - 1
    return {
        "edges": edges.tolist(),
        "sumw": [0.0] * size,
        "sumw2": [0.0] * size,
        "entries": [0] * size,
    }


def nested_histogram(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    edges: np.ndarray,
) -> dict[str, Any]:
    target = payload
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    return target.setdefault(keys[-1], empty_histogram(edges))


def fill_histogram(
    target: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> None:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (values >= edges[0])
    )
    if not np.any(valid):
        return
    indices = np.searchsorted(edges, values[valid], side="right") - 1
    indices = np.minimum(indices, len(edges) - 2)
    selected_weights = weights[valid]
    target["sumw"] = (
        np.asarray(target["sumw"], dtype=float)
        + np.bincount(
            indices,
            weights=selected_weights,
            minlength=len(edges) - 1,
        )
    ).tolist()
    target["sumw2"] = (
        np.asarray(target["sumw2"], dtype=float)
        + np.bincount(
            indices,
            weights=selected_weights * selected_weights,
            minlength=len(edges) - 1,
        )
    ).tolist()
    target["entries"] = (
        np.asarray(target["entries"], dtype=int)
        + np.bincount(indices, minlength=len(edges) - 1)
    ).tolist()


def merge_yield(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sumw", "sumw2", "entries"):
        if isinstance(source[key], list):
            target[key] = (
                np.asarray(target[key]) + np.asarray(source[key])
            ).tolist()
        elif key == "entries":
            target[key] = int(target[key]) + int(source[key])
        else:
            target[key] = float(target[key]) + float(source[key])


def merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    if set(source) >= {"sumw", "sumw2", "entries"}:
        merge_yield(target, source)
        return
    for key, value in source.items():
        if isinstance(value, dict):
            if set(value) >= {"sumw", "sumw2", "entries"}:
                initial = (
                    empty_histogram(np.asarray(value["edges"], dtype=float))
                    if isinstance(value.get("sumw"), list)
                    and "edges" in value
                    else empty_yield()
                )
                merge_yield(target.setdefault(key, initial), value)
            else:
                merge_tree(target.setdefault(key, {}), value)


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
        recoil = bh.float_field(chunk, "recoil_dy2e", n, -99.0)
        recoil_phi = bh.float_field(
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
        recoil = bh.float_field(chunk, "recoil_dy2m", n, -99.0)
        recoil_phi = bh.float_field(
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
    nominal_topology = bh.bool_field(
        chunk, "pass_lowdm_topology_veto", n
    )
    has_removed_fatjet = np.asarray(
        ak.any(~fatjet_clean, axis=1), dtype=bool
    )
    exact_selected = known & nominal_topology
    ambiguous = known & (~nominal_topology) & has_removed_fatjet
    indices = np.full(n, -1, dtype=int)
    for event_index in np.flatnonzero(exact_selected):
        raw_index = bh.assign_lowdm_search_bin(
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
        stored_count = bh.int_field(chunk, "n_e_medium", n, -1)
        opposite_count = bh.int_field(chunk, "n_m_loose", n, -1)
        leading, subleading = first_two(chunk["electron_medium_pt"], -99.0)
        trigger = bh.bool_field(chunk, "pass_electron_trigger", n)
        mass = bh.float_field(chunk, "mee", n, -99.0)
        ptll = bh.float_field(chunk, "pee", n, -99.0)
        recoil = bh.float_field(chunk, "recoil_dy2e", n, -99.0)
        open_high = bh.bool_field(chunk, "pass_dy2e_open_high", n)
        leading_min = 40.0
    else:
        stored_count = bh.int_field(chunk, "n_m_medium", n, -1)
        opposite_count = bh.int_field(chunk, "n_e_veto", n, -1)
        leading, subleading = first_two(chunk["muon_medium_pt"], -99.0)
        trigger = bh.bool_field(chunk, "pass_muon_trigger", n)
        mass = bh.float_field(chunk, "mmm", n, -99.0)
        ptll = bh.float_field(chunk, "pmm", n, -99.0)
        recoil = bh.float_field(chunk, "recoil_dy2m", n, -99.0)
        open_high = bh.bool_field(chunk, "pass_dy2m_open_high", n)
        leading_min = 50.0
    mismatch = rebuilt_count != stored_count
    if np.any(mismatch):
        raise RuntimeError(
            f"{channel}: rebuilt/stored medium-lepton counts differ for "
            f"{int(np.sum(mismatch))} events"
        )
    common = (
        bh.bool_field(chunk, "pass_base_common", n)
        & trigger
        & bh.bool_field(chunk, "pass_zero_tau", n)
        & (opposite_count == 0)
        & (stored_count == 2)
        & (leading > leading_min)
        & (subleading > 20.0)
        & (first_charge != second_charge)
        & (ptll > 200.0)
        & (recoil > 250.0)
        & (bh.int_field(chunk, "njet_lepton_clean", n, -1) >= 2)
        & (bh.float_field(chunk, "ht_lepton_clean", n, -99.0) > 300.0)
    )
    on_z = common & (mass > 81.0) & (mass < 101.0)
    off_z = common & (
        ((mass > 50.0) & (mass < 81.0)) | (mass > 101.0)
    )
    high = (
        common
        & (bh.int_field(chunk, "njet_lepton_clean", n, -1) >= 5)
        & (bh.int_field(chunk, "nb_lepton_clean", n, -1) >= 1)
        & open_high
    )
    sparse_low = (
        common
        & (bh.int_field(chunk, "nb_lepton_clean", n, -1) >= 1)
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
    dy_policy: str,
    step_size: int,
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, Any]:
    root_path = Path(root_name)
    repo = Path(repo_name)
    normalization = read_json(Path(normalization_name))
    result: dict[str, Any] = {
        "rz_high": {},
        "rz_low_feature": {},
        "rz_high_ut": {},
        "rz_low_feature_ut": {},
        "mll_high": {},
        "mll_low_feature": {},
        "gcr_data": {"highdm": {}, "lowdm": {}},
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
        requested = sorted(set(bh.READ_BRANCHES) | EXTRA_BRANCHES)
        missing_extra = sorted(EXTRA_BRANCHES - present)
        if missing_extra:
            raise RuntimeError(
                f"{root_path}: required AN measurement branches absent: "
                + ",".join(missing_extra)
            )
        branches = [branch for branch in requested if branch in present]
        iterator = bh.iterate_tree_with_dy_policy(
            tree,
            branches,
            metadata,
            dy_policy,
            result["summary"],
            step_size,
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
                dataset, process, is_data, is_signal = bh.dataset_label(
                    metadata, dataset_id
                )
                if is_signal or not bh.dy_ptll_dataset_allowed(
                    dataset, process, dy_policy
                ):
                    continue
                n = len(sub["dataset_id"])
                if not n:
                    continue
                result["summary"]["datasets"][dataset] = (
                    int(result["summary"]["datasets"].get(dataset, 0)) + n
                )

                # Weight evaluation is by far the dominant cost and is
                # unnecessary for events that cannot enter the dilepton
                # matrix or the photon-control data counts.  All predicates
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
                if is_data and process == "EGamma":
                    interest |= bh.bool_field(sub, "feature_GCR", n)
                    interest |= bh.lowdm_region_mask(sub, "GCR", n)
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
                    arrays, inputs = bh.flat_arrays_for_weights(sub)
                    years = np.asarray(sub["year"], dtype=int)
                    year = str(int(years[0])) if len(years) else "2024"
                    norm = bh.norm_vector(
                        normalization,
                        sub,
                        dataset_id,
                        dataset,
                        is_data=False,
                        is_signal=False,
                        require_normalization=True,
                    )
                    _generator, variations, status = bh.compute_weight_bundle(
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
                    weights = bh.finite_array(
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
                    fill_rz_ut_yields(
                        result["rz_high_ut"],
                        channel,
                        component,
                        masks["high"],
                        masks,
                        weights,
                        RZ_HIGH_UT_EDGES,
                    )
                    fill_rz_ut_yields(
                        result["rz_low_feature_ut"],
                        channel,
                        component,
                        low_selected,
                        masks,
                        weights,
                        RZ_LOW_UT_EDGES,
                    )
                    nb = bh.int_field(sub, "nb_lepton_clean", n, -1)
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

                if is_data and process == "EGamma":
                    high_mask = bh.bool_field(sub, "feature_GCR", n)
                    nb = bh.int_field(sub, "nb_photon_clean", n, -1)
                    recoil = bh.float_field(sub, "recoil_gcr", n, -1.0)
                    for group, group_mask in {
                        "Nb1": nb == 1,
                        "Nb2": nb == 2,
                        "Nb3plus": nb >= 3,
                    }.items():
                        valid = high_mask & group_mask & (recoil >= HIGH_EDGES[0])
                        if not np.any(valid):
                            continue
                        indices = (
                            np.searchsorted(
                                HIGH_EDGES, recoil[valid], side="right"
                            )
                            - 1
                        )
                        indices = np.minimum(indices, len(HIGH_EDGES) - 2)
                        for bin_index in range(len(HIGH_EDGES) - 1):
                            selected = indices == bin_index
                            if np.any(selected):
                                add_yield(
                                    nested_yield(
                                        result["gcr_data"]["highdm"],
                                        (group, str(bin_index)),
                                    ),
                                    weights[valid][selected],
                                )
                    low_index = bh.lowdm_nbge1_indices(
                        np.where(
                            bh.lowdm_region_mask(sub, "GCR", n),
                            bh.int_field(
                                sub,
                                "lowdm_search_bin_GCR",
                                n,
                                -1,
                            ),
                            -1,
                        )
                    )
                    for bin_index in range(len(bh.LOWDM_34BIN_LABELS)):
                        selected = low_index == bin_index
                        if np.any(selected):
                            add_yield(
                                nested_yield(
                                    result["gcr_data"]["lowdm"],
                                    (str(bin_index),),
                                ),
                                weights[selected],
                            )
    return result


def solve_matrix(
    data_on: float,
    data_off: float,
    z_on: float,
    z_off: float,
    other_on: float,
    other_off: float,
    variances: list[float],
) -> dict[str, Any]:
    """Fit non-negative DY and non-DY normalizations in on/off-Z data.

    The two observed counts are described with Poisson terms.  The four
    weighted-MC template yields are profiled with Gaussian constraints whose
    variances are their sumw2 values.  This is the likelihood counterpart of
    the 2x2 matrix solution, but it remains physical when a low-count bin would
    otherwise return a negative component normalization.
    """

    matrix = np.asarray(
        [[z_on, other_on], [z_off, other_off]], dtype=float
    )
    data = np.asarray([data_on, data_off], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1.0e-12:
        return {
            "status": "singular",
            "determinant": determinant,
            "RZ": None,
            "RT": None,
        }
    nominal = np.asarray([z_on, z_off, other_on, other_off], dtype=float)
    template_variance = np.maximum(
        np.asarray(variances[2:], dtype=float), 0.0
    )
    floating = np.flatnonzero(template_variance > 0.0)

    try:
        algebraic = np.linalg.solve(matrix, data)
    except np.linalg.LinAlgError:
        algebraic = np.asarray([1.0, 1.0], dtype=float)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.asarray(parameters[:2], dtype=float)
        templates = nominal.copy()
        templates[floating] = parameters[2:]
        return scale, templates

    def nll(parameters: np.ndarray) -> float:
        scale, templates = unpack(parameters)
        expectation = np.asarray(
            [
                scale[0] * templates[0] + scale[1] * templates[2],
                scale[0] * templates[1] + scale[1] * templates[3],
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(expectation)) or np.any(expectation <= 0.0):
            return 1.0e100
        poisson = float(np.sum(expectation - data * np.log(expectation)))
        if len(floating):
            delta = templates[floating] - nominal[floating]
            constraint = 0.5 * float(
                np.sum(delta * delta / template_variance[floating])
            )
        else:
            constraint = 0.0
        return poisson + constraint

    clipped = np.maximum(algebraic, 0.0)
    seeds = (
        clipped,
        np.asarray([1.0, 1.0], dtype=float),
        np.asarray(
            [
                max((data_on + data_off) / max(z_on + z_off, 1.0e-9), 0.0),
                0.0,
            ],
            dtype=float,
        ),
    )
    best = None
    bounds = [(0.0, None), (0.0, None)] + [
        (0.0, None) for _ in floating
    ]
    if np.all(algebraic >= 0.0):
        # With two observations and two positive scale factors, the algebraic
        # solution reproduces both Poisson means exactly while leaving every
        # constrained MC template at its nominal value.  It is therefore the
        # global interior likelihood maximum; no numerical minimizer is needed.
        parameters = np.concatenate(
            [algebraic, np.maximum(nominal[floating], 0.0)]
        )
        best = SimpleNamespace(
            x=parameters,
            fun=nll(parameters),
            success=True,
            message="analytic interior maximum",
        )
    else:
        for seed in seeds:
            initial = np.concatenate(
                [seed, np.maximum(nominal[floating], 0.0)]
            )
            candidate = minimize(
                nll,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 5000},
            )
            if np.isfinite(candidate.fun) and (
                best is None or candidate.fun < best.fun
            ):
                best = candidate
    if best is None:
        return {
            "status": "fit_failed",
            "determinant": determinant,
            "message": "no finite likelihood candidate",
            "RZ": None,
            "RT": None,
        }

    scale, templates = unpack(best.x)
    expectation = np.asarray(
        [
            scale[0] * templates[0] + scale[1] * templates[2],
            scale[0] * templates[1] + scale[1] * templates[3],
        ],
        dtype=float,
    )

    # Expected Fisher information for the fitted scale factors and profiled MC
    # template yields.  Its inverse gives the local covariance after profiling.
    dimension = 2 + len(floating)
    information = np.zeros((dimension, dimension), dtype=float)
    template_to_parameter = {
        int(template_index): 2 + position
        for position, template_index in enumerate(floating)
    }
    for window_index in range(2):
        if window_index == 0:
            gradient = np.asarray([templates[0], templates[2]], dtype=float)
            template_indices = (0, 2)
        else:
            gradient = np.asarray([templates[1], templates[3]], dtype=float)
            template_indices = (1, 3)
        full_gradient = np.zeros(dimension, dtype=float)
        full_gradient[:2] = gradient
        for component_index, template_index in enumerate(template_indices):
            parameter_index = template_to_parameter.get(template_index)
            if parameter_index is not None:
                full_gradient[parameter_index] = scale[component_index]
        information += np.outer(full_gradient, full_gradient) / max(
            expectation[window_index], 1.0e-12
        )
    for position, template_index in enumerate(floating):
        information[2 + position, 2 + position] += (
            1.0 / template_variance[template_index]
        )
    covariance_full = np.linalg.pinv(information, hermitian=True)
    covariance = covariance_full[:2, :2]
    return {
        "status": "complete",
        "determinant": determinant,
        "solver": "nonnegative_profile_likelihood_poisson_data_gaussian_mcstat",
        "fit_converged": bool(best.success),
        "fit_message": str(best.message),
        "fit_nll": float(best.fun),
        "fit_expectation": expectation.tolist(),
        "profiled_templates": templates.tolist(),
        "boundary": {
            "RZ": bool(scale[0] <= 1.0e-8),
            "RT": bool(scale[1] <= 1.0e-8),
        },
        "RZ": float(scale[0]),
        "RT": float(scale[1]),
        "RZ_stat": float(math.sqrt(max(covariance[0, 0], 0.0))),
        "RT_stat": float(math.sqrt(max(covariance[1, 1], 0.0))),
        "correlation": float(
            covariance[0, 1]
            / math.sqrt(
                max(covariance[0, 0], 1.0e-300)
                * max(covariance[1, 1], 1.0e-300)
            )
        ),
        "covariance": covariance.tolist(),
    }


def finalize_rz(payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"channels": {}, "combined": {}}
    for channel in CHANNELS:
        output["channels"][channel] = {}
        for group in HIGH_GROUPS:
            source = ((payload.get(channel) or {}).get(group) or {})

            def leaf(window: str, component: str) -> dict[str, Any]:
                return (
                    (source.get(window) or {}).get(component)
                    or empty_yield()
                )

            data_on = leaf("on", "data")
            data_off = leaf("off", "data")
            z_on = leaf("on", "zll")
            z_off = leaf("off", "zll")
            other_on = leaf("on", "other")
            other_off = leaf("off", "other")
            solution = solve_matrix(
                float(data_on["sumw"]),
                float(data_off["sumw"]),
                float(z_on["sumw"]),
                float(z_off["sumw"]),
                float(other_on["sumw"]),
                float(other_off["sumw"]),
                [
                    max(float(data_on["sumw"]), 0.0),
                    max(float(data_off["sumw"]), 0.0),
                    float(z_on["sumw2"]),
                    float(z_off["sumw2"]),
                    float(other_on["sumw2"]),
                    float(other_off["sumw2"]),
                ],
            )
            solution["inputs"] = {
                "data_on": data_on,
                "data_off": data_off,
                "zll_on": z_on,
                "zll_off": z_off,
                "other_on": other_on,
                "other_off": other_off,
            }
            output["channels"][channel][group] = solution
    for group in HIGH_GROUPS:
        measurements = [
            output["channels"][channel][group]
            for channel in CHANNELS
            if output["channels"][channel][group].get("status")
            == "complete"
            and output["channels"][channel][group].get("RZ_stat", 0.0) > 0.0
        ]
        if not measurements:
            output["combined"][group] = {"status": "unavailable"}
            continue
        weights = np.asarray(
            [1.0 / item["RZ_stat"] ** 2 for item in measurements],
            dtype=float,
        )
        values = np.asarray(
            [item["RZ"] for item in measurements], dtype=float
        )
        output["combined"][group] = {
            "status": "complete",
            "RZ": float(np.sum(weights * values) / np.sum(weights)),
            "RZ_stat": float(math.sqrt(1.0 / np.sum(weights))),
            "channels": [
                channel
                for channel in CHANNELS
                if output["channels"][channel][group].get("status")
                == "complete"
            ],
        }
    return output


def finalize_rz_ut(
    payload: dict[str, Any],
    edges: np.ndarray,
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, Any]:
    """Solve the on/off-Z matrix independently in each UT bin."""

    output: dict[str, Any] = {
        "edges": edges.tolist(),
        "overflow_in_last_bin": True,
        "channels": {},
    }
    for channel in channels:
        records: list[dict[str, Any]] = []
        channel_source = payload.get(channel) or {}
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            source = channel_source.get(str(bin_index)) or {}

            def leaf(window: str, component: str) -> dict[str, Any]:
                return (
                    (source.get(window) or {}).get(component)
                    or empty_yield()
                )

            data_on = leaf("on", "data")
            data_off = leaf("off", "data")
            z_on = leaf("on", "zll")
            z_off = leaf("off", "zll")
            other_on = leaf("on", "other")
            other_off = leaf("off", "other")
            solution = solve_matrix(
                float(data_on["sumw"]),
                float(data_off["sumw"]),
                float(z_on["sumw"]),
                float(z_off["sumw"]),
                float(other_on["sumw"]),
                float(other_off["sumw"]),
                [
                    max(float(data_on["sumw"]), 0.0),
                    max(float(data_off["sumw"]), 0.0),
                    float(z_on["sumw2"]),
                    float(z_off["sumw2"]),
                    float(other_on["sumw2"]),
                    float(other_off["sumw2"]),
                ],
            )
            solution.update(
                {
                    "bin": bin_index,
                    "low": float(low),
                    "high": float(high),
                    "overflow": bin_index == len(edges) - 2,
                    "inputs": {
                        "data_on": data_on,
                        "data_off": data_off,
                        "zll_on": z_on,
                        "zll_off": z_off,
                        "other_on": other_on,
                        "other_off": other_off,
                    },
                }
            )
            records.append(solution)
        output["channels"][channel] = {"bins": records}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument(
        "--data-input-list",
        type=Path,
        help=(
            "Optional final nominal ROOT list; only data_balanced20 entries "
            "are appended to the background-MC input list."
        ),
    )
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=100000)
    parser.add_argument("--dy-ptll-policy", default="ptll100_200")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=CHANNELS,
        default=list(CHANNELS),
        help="Dilepton channel(s) to process; use DY2E for the first staged run.",
    )
    args = parser.parse_args()
    channels = tuple(dict.fromkeys(args.channels))

    roots = [
        line.strip()
        for line in args.input_list.read_text().splitlines()
        if line.strip()
    ]
    if args.data_input_list:
        roots.extend(
            line.strip()
            for line in args.data_input_list.read_text().splitlines()
            if line.strip() and "/data_balanced20/" in line
        )
    if len(roots) != len(set(roots)):
        raise SystemExit("duplicate ROOT paths in input list")
    merged: dict[str, Any] = {
        "schema_version": "an_zinv_measurement_inputs_2024_v1",
        "status": "running",
        "rz_high_raw": {},
        "rz_low_feature_raw": {},
        "rz_high_ut_raw": {},
        "rz_low_feature_ut_raw": {},
        "mll_high": {},
        "mll_low_feature": {},
        "gcr_data": {
            "highdm": {
                "recoil_edges": HIGH_EDGES.tolist(),
                "nb_groups": ["Nb1", "Nb2", "Nb3plus"],
                "yields": {},
            },
            "lowdm": {
                "search_bin_labels": list(bh.LOWDM_34BIN_LABELS),
                "yields": {},
            },
        },
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
            "dy_ptll_policy": args.dy_ptll_policy,
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
                args.dy_ptll_policy,
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
            merge_tree(merged["rz_high_ut_raw"], result["rz_high_ut"])
            merge_tree(
                merged["rz_low_feature_raw"],
                result["rz_low_feature"],
            )
            merge_tree(
                merged["rz_low_feature_ut_raw"],
                result["rz_low_feature_ut"],
            )
            merge_tree(merged["mll_high"], result["mll_high"])
            merge_tree(
                merged["mll_low_feature"],
                result["mll_low_feature"],
            )
            merge_tree(
                merged["gcr_data"]["highdm"]["yields"],
                result["gcr_data"]["highdm"],
            )
            merge_tree(
                merged["gcr_data"]["lowdm"]["yields"],
                result["gcr_data"]["lowdm"],
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
    merged["rz_high_ut"] = finalize_rz_ut(
        merged["rz_high_ut_raw"], RZ_HIGH_UT_EDGES, channels
    )
    merged["rz_low_feature_ut"] = finalize_rz_ut(
        merged["rz_low_feature_ut_raw"], RZ_LOW_UT_EDGES, channels
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
