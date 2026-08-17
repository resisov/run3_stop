#!/usr/bin/env python3
"""Unified MET/photon trigger measurement workflow.

This is the only trigger-measurement executable.  Its subcommands build input
records, prepare EOS-only Condor shards, count numerator/denominator histograms,
recover failed files locally, reduce the small JSON outputs to scale factors, and
draw the final trigger efficiencies and scale factors.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from autonomous_allhad.flat_ntuple_worker import physical_dataset_key, read_runs_sumw
from autonomous_allhad.real_subset_worker import (
    ELECTRON_REFERENCE_HLT,
    FILTERS,
    HT_REFERENCE_HLT,
    JET_ID_INPUTS,
    PHOTON_HLT,
    SIGNAL_HLT,
    ak4_jet_veto_mask,
    ak4_tight_lepton_veto_mask,
    all_filters,
    apply_jec,
    arr,
    bool_branch,
    clean_by_delta_r,
    cleanup_xrd_cache,
    count,
    first_or,
    golden_lumi_mask,
    jet_feature_block,
    medium_photon_mask,
    open_root_with_xrd_fallback,
    shifted_met,
    transverse_mass,
    transverse_vector_sum,
)
from workflow.reference_trigger_counts import (
    empty_counts,
    fill_counts,
    json_safe,
    pileup_weight_triplet,
    serialise_counts,
)


MEASUREMENTS = {
    "met_genuine": {
        "probe_hlt": SIGNAL_HLT,
    },
    "photon": {
        "probe_hlt": PHOTON_HLT,
    },
}


COMMON_BRANCHES = [
    "run", "luminosityBlock", "event",
    "MET_pt", "MET_phi", "PFMET_pt", "PFMET_phi", "PuppiMET_pt", "PuppiMET_phi",
    "CaloMET_pt", "Rho_fixedGridRhoFastjetAll",
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_area", "Jet_jetId",
    *JET_ID_INPUTS,
    "Tau_pt", "Tau_eta", "Tau_phi", "Tau_dz", "Tau_decayMode", "Tau_idDeepTau2018v2p5VSjet",
    "IsoTrack_pt", "IsoTrack_eta", "IsoTrack_phi", "IsoTrack_pdgId", "IsoTrack_pfRelIso03_all",
    "Pileup_nTrueInt", "genWeight",
    *FILTERS,
]

MET_MEASUREMENT_BRANCHES = [
    "Electron_pt", "Electron_eta", "Electron_phi", "Electron_cutBased", "Electron_miniPFRelIso_all",
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_looseId", "Muon_mediumId", "Muon_miniPFRelIso_all",
    *ELECTRON_REFERENCE_HLT,
    *SIGNAL_HLT,
]

PHOTON_MEASUREMENT_BRANCHES = [
    "Electron_pt", "Electron_eta", "Electron_cutBased", "Electron_miniPFRelIso_all",
    "Muon_pt", "Muon_eta", "Muon_looseId", "Muon_miniPFRelIso_all",
    "Photon_pt", "Photon_eta", "Photon_phi", "Photon_cutBased", "Photon_electronVeto",
    *HT_REFERENCE_HLT,
    *PHOTON_HLT,
]


def _is_data(record: dict[str, Any]) -> bool:
    process = str(record.get("process_group") or record.get("process") or "")
    return bool(record.get("is_data") or process in {"EGamma", "JetMET", "data"})


def validate_record(record: dict[str, Any], measurement: str) -> None:
    dataset = str(record.get("dataset") or "")
    process = str(record.get("process_group") or record.get("process") or "")
    is_data = _is_data(record)
    if measurement == "met_genuine":
        valid = (
            (is_data and (process == "EGamma" or "EGamma" in dataset))
            or (
                not is_data
                and ("TTtoLNu2Q" in dataset or "TTToSemiLeptonic" in dataset)
            )
        )
        expected = "EGamma data or semileptonic TT (TTtoLNu2Q) MC"
    elif measurement == "photon":
        valid = (
            (is_data and (process == "JetMET" or "JetMET" in dataset))
            or (
                not is_data
                and (process in {"GJ", "GammaJet"} or "GJets" in dataset or "Gamma" in dataset)
            )
        )
        expected = "JetMET data or Gamma+Jet MC"
    else:
        raise ValueError(f"unsupported trigger measurement: {measurement}")
    if not valid:
        raise ValueError(f"{measurement} rejects dataset {dataset!r} ({process!r}); expected {expected}")


def _add_counts(target: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> None:
    for key in target:
        target[key] += source[key]


def _measurement_axes(measurement: str, config: dict[str, Any]) -> tuple[tuple[int, ...], list[np.ndarray]]:
    if measurement == "met_genuine":
        edges = [np.asarray(config["bin_edges_gev"], dtype=float)]
    elif measurement == "photon":
        edges = [
            np.asarray(config["abseta_edges"], dtype=float),
            np.asarray(config["pt_edges_gev"], dtype=float),
        ]
    else:
        raise ValueError(measurement)
    return tuple(len(values) - 1 for values in edges), edges


def measurement_branches(measurement: str) -> list[str]:
    extras = MET_MEASUREMENT_BRANCHES if measurement == "met_genuine" else PHOTON_MEASUREMENT_BRANCHES
    return list(dict.fromkeys(COMMON_BRANCHES + extras))


def _empty_jagged(n: int) -> ak.Array:
    return ak.Array([[]] * n)


def _trigger_values(
    arrays: Any,
    *,
    measurement: str,
    repo: Path,
    process: str,
    year: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]], dict[str, Any]]:
    """Evaluate only the denominator and probe definitions needed by trigger SFs."""
    n = len(arrays["run"])
    empty = _empty_jagged(n)
    met_pt, met_phi, _ = shifted_met(arrays, n, None, process)
    calo_pt = np.asarray(arr(arrays, "CaloMET_pt", np.full(n, np.nan)), dtype=float)
    puppi_calo = np.divide(met_pt, calo_pt, out=np.full(n, np.inf), where=calo_pt != 0)

    jet_pt_raw = arr(arrays, "Jet_pt", empty)
    jet_eta = arr(arrays, "Jet_eta", empty)
    jet_phi = arr(arrays, "Jet_phi", empty)
    jet_mass_raw = arr(arrays, "Jet_mass", ak.zeros_like(jet_pt_raw))
    jet_pt, _, jec_status = apply_jec(
        arrays, repo, year, process, "Jet", jet_pt_raw, jet_eta, jet_phi, jet_mass_raw, None
    )
    if not isinstance(jec_status, dict) or not jec_status.get("applied"):
        raise RuntimeError(f"required AK4 JEC was not applied: {jec_status}")
    jet_id_mask, _ = ak4_tight_lepton_veto_mask(arrays, jet_pt, jet_eta, repo)
    veto_j, _ = ak4_jet_veto_mask(jet_pt, jet_eta, jet_phi, repo)
    zero_veto_j = count(veto_j) == 0
    good_j = (jet_pt > 30) & (abs(jet_eta) < 2.4) & jet_id_mask
    no_b_mask = good_j & False
    jet_nominal = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j, no_b_mask, met_phi)

    tau_pt = arr(arrays, "Tau_pt", empty)
    tau_eta = arr(arrays, "Tau_eta", empty)
    tau_phi = arr(arrays, "Tau_phi", empty)
    tau_dz = arr(arrays, "Tau_dz", ak.zeros_like(tau_pt))
    tau_dm = arr(arrays, "Tau_decayMode", ak.zeros_like(tau_pt))
    tau_id = arr(arrays, "Tau_idDeepTau2018v2p5VSjet", ak.zeros_like(tau_pt))
    tau_mt = transverse_mass(tau_pt, tau_phi, met_pt, met_phi)
    tau_mask = (
        (tau_pt > 20) & (abs(tau_eta) < 2.5) & (abs(tau_dz) < 0.2)
        & (tau_dm != 5) & (tau_dm != 6) & (tau_id >= 5) & (tau_mt < 100)
    )
    zero_tau = count(tau_mask) == 0

    track_pt = arr(arrays, "IsoTrack_pt", empty)
    track_eta = arr(arrays, "IsoTrack_eta", empty)
    track_phi = arr(arrays, "IsoTrack_phi", empty)
    track_pdg = abs(arr(arrays, "IsoTrack_pdgId", ak.zeros_like(track_pt)))
    track_iso = arr(arrays, "IsoTrack_pfRelIso03_all", ak.full_like(track_pt, 99.0))
    track_mt = transverse_mass(track_pt, track_phi, met_pt, met_phi)
    track_e = (track_pt > 5) & (abs(track_eta) < 2.5) & (track_pdg == 11) & (track_iso < 0.2) & (track_mt < 100)
    track_m = (track_pt > 5) & (abs(track_eta) < 2.5) & (track_pdg == 13) & (track_iso < 0.2) & (track_mt < 100)
    track_pi = (track_pt > 10) & (abs(track_eta) < 2.5) & (track_pdg == 211) & (track_iso < 0.1) & (track_mt < 100)
    no_tracks = (count(track_e) == 0) & (count(track_m) == 0) & (count(track_pi) == 0)

    met_filters, missing_filters = all_filters(arrays, n)
    lumi_mask, _ = golden_lumi_mask(arrays, process, repo, n, year)
    base_common = (
        np.isfinite(met_pt) & (met_pt >= 0) & lumi_mask & met_filters
        & no_tracks & zero_veto_j & (puppi_calo < 5)
    )

    if measurement == "met_genuine":
        e_pt = arr(arrays, "Electron_pt", empty)
        e_eta = arr(arrays, "Electron_eta", empty)
        e_phi = arr(arrays, "Electron_phi", empty)
        e_cb = arr(arrays, "Electron_cutBased", ak.zeros_like(e_pt))
        e_iso = arr(arrays, "Electron_miniPFRelIso_all", ak.full_like(e_pt, 99.0))
        e_fid = (abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5))
        e_med = (e_pt > 10) & e_fid & (e_cb >= 3) & (e_iso < 0.1)

        m_pt = arr(arrays, "Muon_pt", empty)
        m_eta = arr(arrays, "Muon_eta", empty)
        m_phi = arr(arrays, "Muon_phi", empty)
        m_loose_id = arr(arrays, "Muon_looseId", ak.zeros_like(m_pt))
        m_medium_id = arr(arrays, "Muon_mediumId", ak.zeros_like(m_pt))
        m_iso = arr(arrays, "Muon_miniPFRelIso_all", ak.full_like(m_pt, 99.0))
        m_loose = (m_pt > 5) & (abs(m_eta) < 2.4) & m_loose_id & (m_iso < 0.2)
        m_med = (m_pt > 10) & (abs(m_eta) < 2.4) & m_medium_id & (m_iso < 0.2)

        lepton_clean_j = (
            clean_by_delta_r(jet_eta, jet_phi, e_eta[e_med], e_phi[e_med], 0.2)
            & clean_by_delta_r(jet_eta, jet_phi, m_eta[m_med], m_phi[m_med], 0.2)
        )
        jet_lepton = jet_feature_block(
            jet_pt, jet_eta, jet_phi, good_j & lepton_clean_j, no_b_mask & lepton_clean_j, met_phi
        )
        denominator = (
            base_common & bool_branch(arrays, ELECTRON_REFERENCE_HLT, n) & zero_tau
            & (count(e_med) >= 1) & (count(m_loose) == 0)
            & (jet_lepton["njet"] >= 2) & (jet_lepton["ht"] > 300)
            & jet_nominal["open_pre"]
        )
        coordinates = [np.asarray(met_pt[denominator], dtype=float)]
    elif measurement == "photon":
        e_pt = arr(arrays, "Electron_pt", empty)
        e_eta = arr(arrays, "Electron_eta", empty)
        e_cb = arr(arrays, "Electron_cutBased", ak.zeros_like(e_pt))
        e_iso = arr(arrays, "Electron_miniPFRelIso_all", ak.full_like(e_pt, 99.0))
        e_fid = (abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5))
        e_veto = (e_pt > 5) & e_fid & (e_cb >= 1) & (e_iso < 0.1)

        m_pt = arr(arrays, "Muon_pt", empty)
        m_eta = arr(arrays, "Muon_eta", empty)
        m_loose_id = arr(arrays, "Muon_looseId", ak.zeros_like(m_pt))
        m_iso = arr(arrays, "Muon_miniPFRelIso_all", ak.full_like(m_pt, 99.0))
        m_loose = (m_pt > 5) & (abs(m_eta) < 2.4) & m_loose_id & (m_iso < 0.2)

        p_pt = arr(arrays, "Photon_pt", empty)
        p_eta = arr(arrays, "Photon_eta", empty)
        p_phi = arr(arrays, "Photon_phi", empty)
        p_cb = arr(arrays, "Photon_cutBased", ak.zeros_like(p_pt))
        p_electron_veto = arr(arrays, "Photon_electronVeto", ak.zeros_like(p_pt))
        p_med = medium_photon_mask(p_pt, p_eta, p_cb, p_electron_veto)
        recoil_pt, recoil_phi = transverse_vector_sum(
            (met_pt, met_phi), (first_or(0, p_pt[p_med]), first_or(0, p_phi[p_med]))
        )
        del recoil_pt
        photon_clean_j = clean_by_delta_r(jet_eta, jet_phi, p_eta[p_med], p_phi[p_med], 0.2)
        jet_photon = jet_feature_block(
            jet_pt, jet_eta, jet_phi, good_j & photon_clean_j, no_b_mask & photon_clean_j, recoil_phi
        )
        denominator = (
            base_common & bool_branch(arrays, HT_REFERENCE_HLT, n) & (count(p_med) >= 1)
            & (count(e_veto) == 0) & (count(m_loose) == 0) & zero_tau
            & (jet_photon["njet"] >= 2) & (jet_photon["ht"] > 300)
            & jet_photon["open_pre"]
        )
        coordinates = [
            np.asarray(abs(first_or(99.0, p_eta[p_med]))[denominator], dtype=float),
            np.asarray(first_or(-1.0, p_pt[p_med])[denominator], dtype=float),
        ]
    else:
        raise ValueError(measurement)

    passed_all = bool_branch(arrays, MEASUREMENTS[measurement]["probe_hlt"], n)
    gen_weight_all = np.asarray(arr(arrays, "genWeight", np.ones(n)), dtype=float)
    ntrue_all = np.asarray(arr(arrays, "Pileup_nTrueInt", np.full(n, -1.0)), dtype=float)
    selected_indices = np.flatnonzero(denominator)
    event_keys = [
        (int(arrays["run"][index]), int(arrays["luminosityBlock"][index]), int(arrays["event"][index]))
        for index in selected_indices
    ]
    diagnostics = {
        "entries": n,
        "selected": int(np.sum(denominator)),
        "ak4_jec_status": jec_status,
        "missing_filters": missing_filters,
        "selection_engine": "dedicated_minimal_trigger_counter_v1",
    }
    return (
        coordinates,
        np.asarray(passed_all[denominator], dtype=bool),
        np.asarray(gen_weight_all[denominator], dtype=float),
        np.asarray(ntrue_all[denominator], dtype=float),
        event_keys,
        diagnostics,
    )


def count_shard(
    *,
    repo: Path,
    shard: dict[str, Any],
    measurement: str,
    config: dict[str, Any],
    step_size: int,
    max_chunks_per_file: int | None,
) -> dict[str, Any]:
    if measurement not in MEASUREMENTS:
        raise ValueError(f"unsupported trigger measurement: {measurement}")
    records = list(shard.get("records") or [])
    if len(records) > 20:
        raise ValueError(f"trigger measurement shard has {len(records)} files; maximum is 20")
    shape, edges = _measurement_axes(measurement, config)
    data_counts = empty_counts(shape)
    mc_groups: dict[str, dict[str, Any]] = {}
    file_records: list[dict[str, Any]] = []
    failed_files: list[dict[str, Any]] = []
    successful_data_events: set[tuple[int, int, int]] = set()
    duplicates_removed = 0
    original_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(repo)
        # The Condor runtime already contains analysis/data locally.  Bypass
        # the full-analysis staging helper, which also requires unrelated
        # b-tag efficiency histograms that trigger measurements never use.
        os.environ.setdefault("AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA", "0")
        os.environ.setdefault("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", "0")
        for record in records:
            file_path = str(record.get("file_path") or "")
            dataset = str(record.get("dataset") or "")
            process = str(record.get("process_group") or record.get("process") or "")
            is_data = _is_data(record)
            root = None
            access_info: dict[str, Any] = {}
            try:
                validate_record(record, measurement)
                if not file_path:
                    raise ValueError("empty file_path")
                root, access_info = open_root_with_xrd_fallback(file_path, timeout=120)
                if "Events" not in {str(key).split(";", 1)[0] for key in root.keys()}:
                    raise RuntimeError("Events tree missing")
                runs_sumw = None
                runs_branch = None
                xsec = None
                if not is_data:
                    runs_info = read_runs_sumw(root)
                    runs_sumw = runs_info.get("generic_sumw")
                    runs_branch = runs_info.get("generic_sumw_branch")
                    if runs_sumw is None or not math.isfinite(float(runs_sumw)) or float(runs_sumw) == 0.0:
                        raise RuntimeError(f"valid Runs.genEventSumw not found: {runs_info}")
                    xsec = float(record.get("xsec_pb"))
                    if not math.isfinite(xsec) or xsec <= 0.0:
                        raise RuntimeError(f"invalid xsec_pb: {record.get('xsec_pb')}")
                tree = root["Events"]
                present = set(tree.keys())
                requested = measurement_branches(measurement)
                branches = [branch for branch in requested if branch in present]
                per_file_data = empty_counts(shape)
                per_file_mc = {
                    variation: empty_counts(shape)
                    for variation in ("nominal", "up", "down")
                }
                pending_data_events: set[tuple[int, int, int]] = set()
                events_read = 0
                selected_rows = 0
                chunks = 0
                for start in range(0, int(tree.num_entries), step_size):
                    if max_chunks_per_file is not None and chunks >= max_chunks_per_file:
                        break
                    stop = min(start + step_size, int(tree.num_entries))
                    arrays = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")
                    coordinates, passed, gen_weight, ntrue, event_keys, summary = _trigger_values(
                        arrays,
                        measurement=measurement,
                        repo=repo,
                        process=process,
                        year=str(record.get("year") or config.get("year") or "2024"),
                    )
                    events_read += int(summary.get("entries", stop - start))
                    selected_rows += len(passed)
                    if is_data:
                        keep = np.ones(len(passed), dtype=bool)
                        for index, key in enumerate(event_keys):
                            if key in successful_data_events or key in pending_data_events:
                                keep[index] = False
                                duplicates_removed += 1
                            else:
                                pending_data_events.add(key)
                        fill_counts(
                            per_file_data,
                            [values[keep] for values in coordinates],
                            edges,
                            passed[keep],
                            np.ones(int(np.sum(keep)), dtype=float),
                        )
                    else:
                        pu_weights = pileup_weight_triplet(repo, ntrue)
                        for variation, pu_weight in zip(("nominal", "up", "down"), pu_weights):
                            fill_counts(per_file_mc[variation], coordinates, edges, passed, gen_weight * pu_weight)
                    chunks += 1
                if is_data:
                    _add_counts(data_counts, per_file_data)
                    successful_data_events.update(pending_data_events)
                    physical = dataset
                else:
                    physical = physical_dataset_key(dataset, is_data=False)
                    group = mc_groups.setdefault(physical, {
                        "physical_dataset": physical,
                        "process": process,
                        "xsec_pb": xsec,
                        "runs_gen_event_sumw": 0.0,
                        "files": 0,
                        "counts": {
                            variation: empty_counts(shape)
                            for variation in ("nominal", "up", "down")
                        },
                    })
                    if not math.isclose(float(group["xsec_pb"]), float(xsec), rel_tol=0.0, abs_tol=1.0e-12):
                        raise RuntimeError(f"conflicting xsec for {physical}: {group['xsec_pb']} versus {xsec}")
                    group["runs_gen_event_sumw"] += float(runs_sumw)
                    group["files"] += 1
                    for variation in ("nominal", "up", "down"):
                        _add_counts(group["counts"][variation], per_file_mc[variation])
                file_records.append({
                    "file_path": file_path,
                    "effective_file_path": access_info.get("effective_file_path", file_path),
                    "dataset": dataset,
                    "physical_dataset": physical,
                    "process": process,
                    "is_data": is_data,
                    "read_status": "success",
                    "events_in_file": int(tree.num_entries),
                    "events_read": events_read,
                    "selected_rows": selected_rows,
                    "chunks_processed": chunks,
                    "full_file_processed": max_chunks_per_file is None or events_read == int(tree.num_entries),
                    "runs_gen_event_sumw": runs_sumw,
                    "runs_sumw_branch": runs_branch,
                    "xsec_pb": xsec,
                })
            except Exception as exc:
                failure = {
                    "file_path": file_path,
                    "dataset": dataset,
                    "process": process,
                    "read_status": "failed",
                    "failure_stage": "trigger_measurement_count",
                    "exception_type": type(exc).__name__,
                    "concise_error": str(exc)[:800],
                    "alternate_access_attempted": bool(access_info.get("alternate_access_attempted", False)),
                }
                file_records.append(failure)
                failed_files.append(failure)
            finally:
                try:
                    if root is not None:
                        root.close()
                finally:
                    cleanup_xrd_cache(access_info)
    finally:
        os.chdir(original_cwd)
    mc_serialised = {}
    for physical, group in mc_groups.items():
        mc_serialised[physical] = {
            **{key: value for key, value in group.items() if key != "counts"},
            "counts": {
                variation: serialise_counts(counts)
                for variation, counts in group["counts"].items()
            },
        }
    successful = sum(record.get("read_status") == "success" for record in file_records)
    status = "success" if successful and not failed_files else ("incomplete" if successful else "failed")
    return json_safe({
        "schema_version": 1,
        "measurement": config["measurement"],
        "measurement_type": measurement,
        "status": status,
        "shard_id": shard.get("shard_id"),
        "files_expected": len(records),
        "files_processed": successful,
        "files_failed": failed_files,
        "duplicates_removed_within_shard": duplicates_removed,
        "data_counts": serialise_counts(data_counts),
        "mc_physical_datasets": mc_serialised,
        "file_records": file_records,
        "step_size": step_size,
        "max_chunks_per_file": max_chunks_per_file,
        "created_unix": started,
        "completed_unix": time.time(),
    })


def build_records(metadata_path: Path, config: dict[str, Any], measurement: str) -> dict[str, Any]:
    """Select the complete allowed data/MC file set from gzipped metadata."""
    with gzip.open(metadata_path, "rt") as source:
        metadata = json.load(source)
    campaign = config["campaign_inputs"]
    data_prefixes = tuple(str(value) for value in campaign["data_dataset_prefixes"])
    data_process = str(campaign["data_process"])
    mc_specs = list(campaign["mc_datasets"])
    records: list[dict[str, Any]] = []
    matched_mc = {str(spec["dataset_contains"]): 0 for spec in mc_specs}
    matched_data = {prefix: 0 for prefix in data_prefixes}
    seen_files: set[str] = set()
    for dataset in sorted(metadata):
        item = metadata[dataset]
        files = list(item.get("files") or [])
        data_prefix = next((prefix for prefix in data_prefixes if dataset.startswith(prefix)), None)
        mc_spec = next((spec for spec in mc_specs if str(spec["dataset_contains"]) in dataset), None)
        if data_prefix is None and mc_spec is None:
            continue
        if data_prefix is not None and mc_spec is not None:
            raise RuntimeError(f"dataset matches data and MC rules: {dataset}")
        for file_index, file_path in enumerate(files):
            if file_path in seen_files:
                raise RuntimeError(f"duplicate ROOT file in metadata selection: {file_path}")
            seen_files.add(file_path)
            if data_prefix is not None:
                record = {
                    "dataset": dataset,
                    "process_group": data_process,
                    "is_data": True,
                    "is_background": False,
                    "year": str(config["year"]),
                    "file_index": file_index,
                    "xsec_pb": -1.0,
                    "file_path": file_path,
                }
                matched_data[data_prefix] += 1
            else:
                record = {
                    "dataset": dataset,
                    "process_group": str(mc_spec["process"]),
                    "is_data": False,
                    "is_background": True,
                    "year": str(config["year"]),
                    "file_index": file_index,
                    "xsec_pb": float(mc_spec["xsec_pb"]),
                    "xsec_source": str(mc_spec["xsec_source"]),
                    "file_path": file_path,
                }
                matched_mc[str(mc_spec["dataset_contains"])] += 1
            validate_record(record, measurement)
            records.append(record)
    missing = [key for key, value in {**matched_data, **matched_mc}.items() if value == 0]
    if missing:
        raise RuntimeError(f"campaign input rules matched no files: {missing}")
    records.sort(key=lambda item: (bool(not item["is_data"]), item["dataset"], item["file_path"]))
    return {
        "schema_version": 1,
        "measurement": config["measurement"],
        "measurement_type": measurement,
        "metadata_source": str(metadata_path),
        "files_per_condor_shard": int(campaign["files_per_condor_shard"]),
        "data_file_counts": matched_data,
        "mc_file_counts": matched_mc,
        "records": records,
    }


def _eos_path(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    text = str(resolved)
    if not text.startswith("/eos/"):
        raise ValueError(f"{label} must be on EOS, got {text}")
    if "/afs/" in text.lower():
        raise ValueError(f"{label} must never reference AFS: {text}")
    return resolved


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"no records found in {path}")
    return [dict(record) for record in records]


def prepare_condor(
    *,
    records_path: Path,
    repo: Path,
    workdir: Path,
    python_archive: Path,
    runtime_archive: Path,
    proxy: Path,
    config: Path,
    measurement: str,
    files_per_shard: int = 20,
) -> dict[str, Any]:
    """Create one EOS-only submit description with 20 ROOT files per shard."""
    repo = _eos_path(repo, "repo")
    workdir = _eos_path(workdir, "workdir")
    python_archive = _eos_path(python_archive, "python_archive")
    runtime_archive = _eos_path(runtime_archive, "runtime_archive")
    proxy = _eos_path(proxy, "proxy")
    config = _eos_path(config, "config")
    records_path = _eos_path(records_path, "records")
    if files_per_shard != 20:
        raise ValueError("trigger measurement production requires exactly 20 ROOT files per full shard")
    required = (
        (repo, "repo"), (python_archive, "python_archive"),
        (runtime_archive, "runtime_archive"), (proxy, "proxy"),
        (config, "config"), (records_path, "records"),
    )
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    records = _load_records(records_path)
    for record in records:
        validate_record(record, measurement)
    manifests = workdir / "manifests"
    outputs = workdir / "shard_outputs"
    logs = workdir / "logs"
    for directory in (workdir, manifests, outputs, logs):
        directory.mkdir(parents=True, exist_ok=True)
    queue_rows = []
    shard_paths = []
    for index, start in enumerate(range(0, len(records), files_per_shard)):
        subset = records[start:start + files_per_shard]
        digest = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()
        name = f"{measurement}_{index:05d}"
        shard_path = manifests / f"{name}.json"
        shard_path.write_text(json.dumps({
            "schema_version": 1,
            "shard_id": name,
            "measurement_type": measurement,
            "files_per_shard": files_per_shard,
            "record_digest": digest,
            "records": subset,
        }, indent=2, sort_keys=True) + "\n")
        output_path = outputs / f"shard_{index:05d}.json"
        queue_rows.append((name, shard_path, shard_path.name, output_path))
        shard_paths.append(str(shard_path))
    wrapper = workdir / "run_trigger_measurement.sh"
    wrapper_text = f'''#!/bin/bash
set -euo pipefail
SHARD_NAME="$1"
RESULT_DEST="$2"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
cd "$WORKDIR"
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$WORKDIR/runtime_xrd"
export AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$AUTONOMOUS_ALLHAD_XRD_CACHE"
export X509_USER_PROXY="$WORKDIR/{proxy.name}"
chmod 600 "$X509_USER_PROXY"
tar -xzf {python_archive.name}
tar -xzf {runtime_archive.name}
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORKDIR/autonomous_allhad"
"$PY" -c 'import numpy, awkward, uproot, correctionlib'
set +e
"$PY" "$WORKDIR/autonomous_allhad/workflow/measure_trigger.py" count --repo "$WORKDIR" --measurement {measurement} --shard "$WORKDIR/$SHARD_NAME" --config "$WORKDIR/{config.name}" --output "$WORKDIR/result.json"
COUNTER_STATUS=$?
set -e
test -s "$WORKDIR/result.json"
"$PY" -c 'import json; p=json.load(open("result.json")); assert p["status"] in ("success", "incomplete", "failed"); assert "files_processed" in p; assert "files_failed" in p'
case "$RESULT_DEST" in /eos/user/t/taiwoo/*) ;; *) echo "refusing non-EOS result destination: $RESULT_DEST" >&2; exit 64;; esac
xrdcp -f --nopbar "$WORKDIR/result.json" "root://eosuser.cern.ch/$RESULT_DEST"
xrdfs eosuser.cern.ch stat "$RESULT_DEST" >/dev/null
exit "$COUNTER_STATUS"
'''
    wrapper.write_text(wrapper_text)
    wrapper.chmod(0o700)
    queue = workdir / "queue.tsv"
    queue.write_text("".join(f"{name}\t{shard}\t{shard_name}\t{output}\n" for name, shard, shard_name, output in queue_rows))
    submit = workdir / "submit.sub"
    submit_text = f'''universe = vanilla
executable = {wrapper}
arguments = $(shard_name) $(result)
initialdir = {workdir}
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/$(name).log
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {python_archive}, {runtime_archive}, {proxy}, {config}, $(shard)
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 1
request_memory = 4000MB
request_disk = 6000MB
+JobFlavour = "tomorrow"
queue name,shard,shard_name,result from {queue}
'''
    rendered = "\n".join((wrapper_text, submit_text, queue.read_text(), *shard_paths))
    if "/afs/" in rendered.lower():
        raise RuntimeError("AFS reference detected; refusing to create EOS trigger submission")
    submit.write_text(submit_text)
    summary = {
        "schema_version": 1,
        "measurement_type": measurement,
        "records": len(records),
        "files_per_shard": files_per_shard,
        "shards": len(queue_rows),
        "repo": str(repo),
        "workdir": str(workdir),
        "python_archive": str(python_archive),
        "runtime_archive": str(runtime_archive),
        "proxy": str(proxy),
        "submit_file": str(submit),
        "queue_file": str(queue),
        "submission_backend": "EOS schedd selected by module load lxbatch/eossubmit",
        "submit_command": f"module load lxbatch/eossubmit && condor_submit {submit}",
        "afs_reference_check": "passed",
    }
    (workdir / "campaign.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _count_arrays(payload: dict[str, Any], shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    keys = ("total", "passed", "sumw_total", "sumw_passed", "sumw2_total", "sumw2_passed")
    expected = int(np.prod(shape))
    result = {}
    for key in keys:
        values = np.asarray(payload[key], dtype=float)
        if values.size != expected:
            raise ValueError(f"{key} has {values.size} bins; expected {expected}")
        result[key] = values.reshape(shape)
    return result


def _add_normalized_counts(
    target: dict[str, np.ndarray], source: dict[str, np.ndarray], factor: float = 1.0,
) -> None:
    for key in ("total", "passed"):
        target[key] += source[key]
    for key in ("sumw_total", "sumw_passed"):
        target[key] += factor * source[key]
    for key in ("sumw2_total", "sumw2_passed"):
        target[key] += factor * factor * source[key]


def reduce_shards(
    *,
    shard_paths: list[Path],
    measurement: str,
    config: dict[str, Any],
    permanently_skipped_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Merge shard/recovery JSONs and calculate data/MC scale factors."""
    from workflow.reference_trigger_counts import add_pileup_uncertainty, reduce_counts

    shape, edges = _measurement_axes(measurement, config)
    permanently_skipped_paths = permanently_skipped_paths or set()
    data_counts = empty_counts(shape)
    groups: dict[str, dict[str, Any]] = {}
    seen_files: set[str] = set()
    failed_by_path: dict[str, dict[str, Any]] = {}
    incomplete_files: list[str] = []
    shard_audit: list[dict[str, Any]] = []
    for path in sorted(shard_paths):
        payload = json.loads(path.read_text())
        if payload.get("measurement") != config["measurement"] or payload.get("measurement_type") != measurement:
            raise ValueError(f"measurement mismatch in {path}")
        successful_records = [item for item in payload.get("file_records", []) if item.get("read_status") == "success"]
        for record in successful_records:
            source = str(record.get("file_path") or "")
            if source in seen_files:
                raise RuntimeError(f"duplicate successful ROOT file across trigger shards: {source}")
            seen_files.add(source)
            failed_by_path.pop(source, None)
            if not record.get("full_file_processed"):
                incomplete_files.append(source)
        for failure in payload.get("files_failed") or []:
            source = str(failure.get("file_path") or "")
            if source not in seen_files:
                failed_by_path[source] = failure
        _add_normalized_counts(data_counts, _count_arrays(payload["data_counts"], shape))
        for physical, item in (payload.get("mc_physical_datasets") or {}).items():
            target = groups.setdefault(physical, {
                "physical_dataset": physical,
                "process": item.get("process"),
                "xsec_pb": float(item["xsec_pb"]),
                "runs_gen_event_sumw": 0.0,
                "files": 0,
                "counts": {variation: empty_counts(shape) for variation in ("nominal", "up", "down")},
            })
            if not math.isclose(float(target["xsec_pb"]), float(item["xsec_pb"]), rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(f"conflicting xsec for {physical}")
            target["runs_gen_event_sumw"] += float(item["runs_gen_event_sumw"])
            target["files"] += int(item["files"])
            for variation in ("nominal", "up", "down"):
                _add_normalized_counts(target["counts"][variation], _count_arrays(item["counts"][variation], shape))
        shard_audit.append({
            "path": str(path),
            "status": payload.get("status"),
            "files_expected": payload.get("files_expected"),
            "files_processed": payload.get("files_processed"),
            "files_failed": len(payload.get("files_failed") or []),
        })
    mc_counts = {variation: empty_counts(shape) for variation in ("nominal", "up", "down")}
    normalization = {}
    for physical, group in sorted(groups.items()):
        denominator = float(group["runs_gen_event_sumw"])
        if not math.isfinite(denominator) or denominator == 0.0:
            raise RuntimeError(f"invalid aggregate Runs.genEventSumw for {physical}: {denominator}")
        factor = float(group["xsec_pb"]) / denominator
        normalization[physical] = {
            "xsec_pb": float(group["xsec_pb"]),
            "runs_gen_event_sumw": denominator,
            "files": int(group["files"]),
            "factor_without_luminosity": factor,
        }
        for variation in ("nominal", "up", "down"):
            _add_normalized_counts(mc_counts[variation], group["counts"][variation], factor=factor)
    data_json = serialise_counts(data_counts)
    mc_json = {variation: serialise_counts(counts) for variation, counts in mc_counts.items()}
    bins = reduce_counts(data_json, mc_json["nominal"])
    bins_up = reduce_counts(data_json, mc_json["up"])
    bins_down = reduce_counts(data_json, mc_json["down"])
    add_pileup_uncertainty(bins, bins_up, bins_down)
    if measurement == "met_genuine":
        for index, item in enumerate(bins):
            item.update({"low_gev": float(edges[0][index]), "high_gev": float(edges[0][index + 1])})
    else:
        for eta_index in range(shape[0]):
            for pt_index in range(shape[1]):
                index = eta_index * shape[1] + pt_index
                bins[index].update({
                    "abseta_low": float(edges[0][eta_index]),
                    "abseta_high": float(edges[0][eta_index + 1]),
                    "pt_low_gev": float(edges[1][pt_index]),
                    "pt_high_gev": float(edges[1][pt_index + 1]),
                })
    permanently_skipped_files = [
        item for source, item in failed_by_path.items()
        if source in permanently_skipped_paths
    ]
    failed_files = [
        item for source, item in failed_by_path.items()
        if source not in permanently_skipped_paths
    ]
    blockers = []
    if failed_files:
        blockers.append(f"{len(failed_files)} ROOT files failed")
    if incomplete_files:
        blockers.append(f"{len(incomplete_files)} ROOT files were not processed completely")
    if not seen_files:
        blockers.append("no successful ROOT files")
    invalid_bins = [item["flat_index"] for item in bins if not item.get("valid")]
    if invalid_bins:
        blockers.append(f"invalid or empty SF bins: {invalid_bins}")
    axis_payload = (
        {"bin_edges_gev": edges[0].tolist()}
        if measurement == "met_genuine"
        else {"abseta_edges": edges[0].tolist(), "pt_edges_gev": edges[1].tolist()}
    )
    return json_safe({
        "schema_version": 1,
        "measurement": config["measurement"],
        "measurement_type": measurement,
        **axis_payload,
        "status": "validation_pending" if not blockers else "blocked",
        "input_model": "NanoAOD ROOT files grouped 20 per Condor shard; shard outputs contain histograms only",
        "data": data_json,
        "mc": mc_json["nominal"],
        "mc_pileup_variations": {"up": mc_json["up"], "down": mc_json["down"]},
        "mc_normalization": normalization,
        "pileup_correction": "analysis/data/PUweight/2024/puWeights.json.gz::Collisions24_BCDEFGHI_goldenJSON",
        "bins": bins,
        "files_processed": len(seen_files),
        "files_failed": failed_files,
        "files_permanently_skipped": permanently_skipped_files,
        "incomplete_files": incomplete_files,
        "shards": shard_audit,
        "adoption_blockers": blockers,
        "adoption_gates": config["adoption_gates"],
        "created_unix": time.time(),
    })


def recover_failed_files(
    *,
    repo: Path,
    records_path: Path,
    failures_path: Path,
    measurement: str,
    config: dict[str, Any],
    output_dir: Path,
    files_per_shard: int,
    step_size: int,
) -> dict[str, Any]:
    """Reprocess failed ROOT files sequentially after staging each to local scratch."""
    if not 1 <= files_per_shard <= 20:
        raise ValueError("recovery files-per-shard must be between 1 and 20")
    all_records = _load_records(records_path)
    failures_payload = json.loads(failures_path.read_text())
    failures = failures_payload.get("files_failed", failures_payload) if isinstance(failures_payload, dict) else failures_payload
    if not isinstance(failures, list):
        raise ValueError(f"failed-file list not found in {failures_path}")
    failed_paths = {
        str(item.get("file_path") if isinstance(item, dict) else item)
        for item in failures
    }
    selected = [record for record in all_records if str(record.get("file_path")) in failed_paths]
    selected_paths = {str(record.get("file_path")) for record in selected}
    missing = sorted(failed_paths - selected_paths)
    if missing:
        raise RuntimeError(f"{len(missing)} failed files are absent from campaign records")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_prefer_cache = os.environ.get("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE")
    old_keep_cache = os.environ.get("AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE")
    os.environ["AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE"] = "1"
    os.environ["AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE"] = "0"
    outputs = []
    try:
        for index, start in enumerate(range(0, len(selected), files_per_shard)):
            subset = selected[start:start + files_per_shard]
            payload = count_shard(
                repo=repo.resolve(),
                shard={"shard_id": f"recovery_{index:05d}", "records": subset},
                measurement=measurement,
                config=config,
                step_size=step_size,
                max_chunks_per_file=None,
            )
            output = output_dir / f"recovery_{index:05d}.json"
            output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            outputs.append({
                "path": str(output),
                "status": payload["status"],
                "files_expected": payload["files_expected"],
                "files_processed": payload["files_processed"],
                "files_failed": len(payload["files_failed"]),
            })
    finally:
        if old_prefer_cache is None:
            os.environ.pop("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", None)
        else:
            os.environ["AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE"] = old_prefer_cache
        if old_keep_cache is None:
            os.environ.pop("AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE", None)
        else:
            os.environ["AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE"] = old_keep_cache
    return {
        "schema_version": 1,
        "measurement_type": measurement,
        "files_requested": len(failed_paths),
        "files_matched": len(selected),
        "files_processed": sum(item["files_processed"] for item in outputs),
        "files_failed": sum(item["files_failed"] for item in outputs),
        "mode": "sequential local xrdcp cache; no Condor resubmission",
        "outputs": outputs,
    }


def export_correctionlib(*, result_path: Path, output: Path, measurement: str) -> dict[str, Any]:
    """Install an explicitly adopted result as a correctionlib JSON.GZ payload."""
    from workflow.sf_payload import correction, correction_set, ensure_adopted, install_adopted_result

    result = json.loads(result_path.read_text())
    if result.get("measurement_type") != measurement:
        raise ValueError(f"measurement mismatch in {result_path}")
    ensure_adopted(result, result_path)
    bins = result["bins"]
    if measurement == "met_genuine":
        items = [correction(
            name="met_trigger_sf_genuine",
            description="2024 analysis MET-trigger data/MC SF for genuine missing momentum",
            axes=[("met", result["bin_edges_gev"])],
            nominal=[item["scale_factor"] for item in bins],
            uncertainty=[item["scale_factor_uncertainty"] for item in bins],
        )]
        description = "Run-3 all-hadronic stop analysis MET-trigger scale factor"
    else:
        items = [correction(
            name="photon_trigger_sf",
            description="2024 Photon175/Photon200 OR data/MC SF from an independent PFHT reference",
            axes=[("abseta", result["abseta_edges"]), ("pt", result["pt_edges_gev"])],
            nominal=[item["scale_factor"] for item in bins],
            uncertainty=[item["scale_factor_uncertainty"] for item in bins],
        )]
        description = "Run-3 all-hadronic stop analysis photon-trigger scale factor"
    return install_adopted_result(result_path, output, correction_set(description, items))


def _add_measurement_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--measurement", choices=tuple(MEASUREMENTS), required=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _permanently_skipped_paths(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text())
    records = payload.get("bad_files", []) if isinstance(payload, dict) else payload
    return {
        str(item.get("path") or item.get("file_path"))
        for item in records
        if isinstance(item, dict) and item.get("permanently_skipped")
    }


def adopt_trigger_result(
    result_path: Path,
    output: Path,
    measurement: str,
    reviewer: str,
    reason: str,
    allow_permanent_skips: bool,
) -> dict[str, Any]:
    """Record an explicit analyst adoption decision without rewriting raw reduction output."""
    result = json.loads(result_path.read_text())
    if result.get("measurement_type") != measurement:
        raise ValueError(f"measurement mismatch in {result_path}")
    blockers = list(result.get("adoption_blockers") or [])
    if blockers:
        raise RuntimeError(f"refusing adoption with blockers: {blockers}")
    invalid = [item.get("flat_index") for item in result.get("bins") or [] if not item.get("valid")]
    if invalid or not result.get("bins"):
        raise RuntimeError(f"refusing adoption with invalid or missing bins: {invalid}")
    for item in result["bins"]:
        nominal = float(item["scale_factor"])
        uncertainty = float(item["scale_factor_uncertainty"])
        if not math.isfinite(nominal) or nominal <= 0.0:
            raise RuntimeError(f"invalid scale factor in bin {item.get('flat_index')}: {nominal}")
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            raise RuntimeError(
                f"invalid scale-factor uncertainty in bin {item.get('flat_index')}: {uncertainty}"
            )
    permanent_skips = list(result.get("files_permanently_skipped") or [])
    if permanent_skips and not allow_permanent_skips:
        raise RuntimeError(
            f"refusing adoption with {len(permanent_skips)} permanent skips; "
            "pass --allow-permanent-skips after reviewing the bad-file manifest"
        )
    source_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    adopted = dict(result)
    adopted["status"] = "adopted"
    adopted["adoption"] = {
        "adopted_unix": time.time(),
        "reviewer": reviewer,
        "reason": reason,
        "source_result": str(result_path),
        "source_sha256": source_sha256,
        "permanent_skips_accepted": len(permanent_skips),
        "declared_validation_gates": list(result.get("adoption_gates") or []),
        "decision_note": (
            "Explicit analyst adoption after final reduction and plot review; "
            "the raw validation_pending result remains unchanged."
        ),
    }
    _write_json(output, adopted)
    return {
        "status": "adopted",
        "measurement_type": measurement,
        "source_result": str(result_path),
        "source_sha256": source_sha256,
        "output": str(output),
        "bins": len(adopted["bins"]),
        "permanent_skips_accepted": len(permanent_skips),
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build-records", help="select allowed ROOT files from metadata")
    _add_measurement_argument(build_parser)
    build_parser.add_argument("--metadata", type=Path, required=True)
    build_parser.add_argument("--config", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = commands.add_parser("prepare", help="prepare EOS-only Condor shards")
    _add_measurement_argument(prepare_parser)
    prepare_parser.add_argument("--records", type=Path, required=True)
    prepare_parser.add_argument("--repo", type=Path, required=True)
    prepare_parser.add_argument("--workdir", type=Path, required=True)
    prepare_parser.add_argument("--python-archive", type=Path, required=True)
    prepare_parser.add_argument("--runtime-archive", type=Path, required=True)
    prepare_parser.add_argument("--proxy", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--files-per-shard", type=int, default=20)

    count_parser = commands.add_parser("count", help="count one numerator/denominator shard")
    _add_measurement_argument(count_parser)
    count_parser.add_argument("--repo", type=Path, required=True)
    count_parser.add_argument("--shard", type=Path, required=True)
    count_parser.add_argument("--config", type=Path, required=True)
    count_parser.add_argument("--output", type=Path, required=True)
    count_parser.add_argument("--step-size", type=int, default=50_000)
    count_parser.add_argument("--max-chunks-per-file", type=int, default=None)

    recover_parser = commands.add_parser("recover", help="recover failures sequentially via local xrdcp cache")
    _add_measurement_argument(recover_parser)
    recover_parser.add_argument("--repo", type=Path, required=True)
    recover_parser.add_argument("--records", type=Path, required=True)
    recover_parser.add_argument("--failed-files", type=Path, required=True)
    recover_parser.add_argument("--config", type=Path, required=True)
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--files-per-shard", type=int, default=20)
    recover_parser.add_argument("--step-size", type=int, default=50_000)

    reduce_parser = commands.add_parser("reduce", help="merge shard JSONs and calculate SFs")
    _add_measurement_argument(reduce_parser)
    reduce_parser.add_argument("--input-dir", type=Path, action="append", required=True)
    reduce_parser.add_argument("--glob", default="shard_*.json")
    reduce_parser.add_argument("--config", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--bad-files", type=Path)

    export_parser = commands.add_parser("export", help="write an adopted result as correctionlib JSON.GZ")
    _add_measurement_argument(export_parser)
    export_parser.add_argument("--result", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    plot_parser = commands.add_parser("plot", help="draw square mplhep efficiency and SF plots")
    _add_measurement_argument(plot_parser)
    plot_parser.add_argument("--result", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)

    adopt_parser = commands.add_parser("adopt", help="record an explicit analyst adoption decision")
    _add_measurement_argument(adopt_parser)
    adopt_parser.add_argument("--result", type=Path, required=True)
    adopt_parser.add_argument("--output", type=Path, required=True)
    adopt_parser.add_argument("--reviewer", required=True)
    adopt_parser.add_argument("--reason", required=True)
    adopt_parser.add_argument("--allow-permanent-skips", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build-records":
        payload = build_records(args.metadata, json.loads(args.config.read_text()), args.measurement)
        _write_json(args.output, payload)
        print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        payload = prepare_condor(
            records_path=args.records,
            repo=args.repo,
            workdir=args.workdir,
            python_archive=args.python_archive,
            runtime_archive=args.runtime_archive,
            proxy=args.proxy,
            config=args.config,
            measurement=args.measurement,
            files_per_shard=args.files_per_shard,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "count":
        payload = count_shard(
            repo=args.repo.resolve(),
            shard=json.loads(args.shard.read_text()),
            measurement=args.measurement,
            config=json.loads(args.config.read_text()),
            step_size=args.step_size,
            max_chunks_per_file=args.max_chunks_per_file,
        )
        _write_json(args.output, payload)
        return 0 if payload["status"] != "failed" else 2
    if args.command == "recover":
        payload = recover_failed_files(
            repo=args.repo,
            records_path=args.records,
            failures_path=args.failed_files,
            measurement=args.measurement,
            config=json.loads(args.config.read_text()),
            output_dir=args.output_dir,
            files_per_shard=args.files_per_shard,
            step_size=args.step_size,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["files_processed"] else 2
    if args.command == "reduce":
        paths = sorted({path for directory in args.input_dir for path in directory.glob(args.glob)})
        if not paths:
            raise FileNotFoundError("no trigger shard JSONs matched")
        payload = reduce_shards(
            shard_paths=paths,
            measurement=args.measurement,
            config=json.loads(args.config.read_text()),
            permanently_skipped_paths=_permanently_skipped_paths(args.bad_files),
        )
        _write_json(args.output, payload)
        return 0 if payload["files_processed"] else 2
    if args.command == "export":
        print(json.dumps(export_correctionlib(
            result_path=args.result,
            output=args.output,
            measurement=args.measurement,
        ), indent=2, sort_keys=True))
        return 0
    if args.command == "plot":
        from plot_measurement import plot_trigger_result

        print(json.dumps(
            plot_trigger_result(args.result, args.output_dir, args.measurement),
            indent=2,
            sort_keys=True,
        ))
        return 0
    if args.command == "adopt":
        print(json.dumps(adopt_trigger_result(
            result_path=args.result,
            output=args.output,
            measurement=args.measurement,
            reviewer=args.reviewer,
            reason=args.reason,
            allow_permanent_skips=args.allow_permanent_skips,
        ), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(cli())
