#!/usr/bin/env python3
"""Finalize Low-dM R_Z/R_T with sparse NanoAOD object reconstruction."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from ..real_subset_worker import (
    BOOSTED_ETA_MAX,
    BOOSTED_TOP_MSD_MIN,
    BOOSTED_TOP_PT_MIN,
    BOOSTED_TOP_SCORE_BRANCH,
    BOOSTED_TOP_SCORE_WP,
    BOOSTED_W_MSD_MAX,
    BOOSTED_W_MSD_MIN,
    BOOSTED_W_PT_MIN,
    BOOSTED_W_SCORE_BRANCH,
    BOOSTED_W_SCORE_WP,
    FATJET_ID_INPUTS,
    JET_ID_INPUTS,
    ak4_tight_lepton_veto_mask,
    ak8_tight_lepton_veto_mask,
    apply_jec,
    arr,
    cleanup_xrd_cache,
    clean_by_delta_r,
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
from .sparse import group_sparse_windows, stable_file_id


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return payload


def sparse_access_path(file_path: str) -> str:
    """Use a concrete data server so uproot can issue sparse range reads."""

    marker = "/store/"
    if not file_path.startswith("root://") or marker not in file_path:
        return file_path
    lfn = file_path[file_path.index(marker) :]
    host = os.environ.get(
        "AUTONOMOUS_ALLHAD_SPARSE_XRD_HOST",
        "cmsxrootd-site1.fnal.gov:1093",
    )
    return f"root://{host}/{lfn}"


def add_source(
    mapping: dict[int, dict[str, Any]],
    record: dict[str, Any],
    wanted_ids: set[int],
) -> None:
    path = str(record.get("file_path") or "")
    if not path:
        return
    file_id = stable_file_id(path)
    if file_id not in wanted_ids:
        return
    normalized = {
        "file_path": path,
        "dataset": str(record.get("dataset") or ""),
        "process": str(
            record.get("process_group") or record.get("process") or ""
        ),
        "year": str(record.get("year") or "2024"),
    }
    previous = mapping.get(file_id)
    if previous and previous["file_path"] != path:
        raise RuntimeError(
            f"stable file-id collision {file_id}: "
            f"{previous['file_path']} versus {path}"
        )
    mapping[file_id] = normalized


def source_map(
    shard_bundle: Path,
    feature_roots: list[str],
    wanted_ids: set[int],
) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    with tarfile.open(shard_bundle, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = json.load(handle)
            for record in payload.get("records") or []:
                add_source(mapping, record, wanted_ids)
    for root_name in feature_roots:
        sidecar = Path(root_name).with_suffix(".json")
        if not sidecar.is_file():
            continue
        metadata = read_json(sidecar)
        for record in metadata.get("files") or []:
            add_source(mapping, record, wanted_ids)
    return mapping



def process_source(task: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the quantities absent from the feature table.

    ``lowdm_feature_decision`` proves that every adopted Low-dM predicate is
    exact for these candidates except the lepton-cleaned boosted-top/W veto.
    The b-jet group is recomputed here as an independent consistency check.
    This reads the canonical AK4/AK8 and lepton branches and calls the same
    JEC and jet-ID helpers as ``real_subset_worker``; it deliberately avoids
    unrelated MET, trigger, generator, photon, tau, and SV branches.
    """

    source = task["source"]
    candidates = task["candidates"]
    repo = Path(task["repo"])
    max_span = int(task["max_span"])
    max_gap = int(task["max_gap"])
    output: dict[str, Any] = {
        "raw": {},
        "mll": {},
        "summary": {
            "file_id": int(task["file_id"]),
            "file_path": source["file_path"],
            "targets": len(candidates),
            "matched": 0,
            "selected": 0,
            "windows": 0,
            "read_mode": "canonical_topology_only",
        },
    }
    by_entry: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_entry.setdefault(int(candidate["entry"]), []).append(candidate)
    required = {
        "run",
        "luminosityBlock",
        "event",
        "Rho_fixedGridRhoFastjetAll",
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
        "Jet_area",
        "Jet_btagUParTAK4B",
        *JET_ID_INPUTS,
        "FatJet_pt",
        "FatJet_eta",
        "FatJet_phi",
        "FatJet_mass",
        "FatJet_area",
        "FatJet_msoftdrop",
        BOOSTED_TOP_SCORE_BRANCH,
        BOOSTED_W_SCORE_BRANCH,
        *FATJET_ID_INPUTS,
        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_cutBased",
        "Electron_miniPFRelIso_all",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mediumId",
        "Muon_miniPFRelIso_all",
    }
    root_file = None
    access_info: dict[str, Any] = {}
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        primary_path = sparse_access_path(source["file_path"])
        marker = "/store/"
        lfn = (
            source["file_path"][source["file_path"].index(marker) :]
            if marker in source["file_path"]
            else ""
        )
        replica_paths = (
            [
                f"root://xrootd-cms.infn.it/{lfn}",
                f"root://cmsxrootd.fnal.gov/{lfn}",
                f"root://cmsxrootd-site1.fnal.gov:1093/{lfn}",
            ]
            if lfn
            else []
        )
        access_paths = list(
            dict.fromkeys((primary_path, source["file_path"], *replica_paths))
        )
        open_errors: list[str] = []
        effective_path = ""
        for direct_path in access_paths:
            try:
                root_file = uproot.open(direct_path, timeout=90)
                effective_path = direct_path
                break
            except Exception as exc:
                open_errors.append(f"{direct_path}: {type(exc).__name__}: {exc}")
        if root_file is None:
            raise OSError("; ".join(open_errors))
        access_info = {
            "access_method": "direct_sparse_xrootd",
            "source_file_path": source["file_path"],
            "effective_file_path": effective_path,
            "open_attempts": access_paths,
            "open_errors": open_errors,
        }
        tree = root_file["Events"]
        missing = sorted(required - set(tree.keys()))
        if missing:
            raise RuntimeError(f"missing exact-topology branches: {missing}")
        branches = sorted(required)
        for start, stop, members in group_sparse_windows(
            by_entry, max_span=max_span, max_gap=max_gap
        ):
            output["summary"]["windows"] += 1
            arrays = tree.arrays(
                branches,
                entry_start=start,
                entry_stop=stop,
                library="ak",
            )
            n = len(arrays["run"])
            jet_pt_raw = arr(arrays, "Jet_pt")
            jet_eta = arr(arrays, "Jet_eta")
            jet_phi = arr(arrays, "Jet_phi")
            jet_mass_raw = arr(arrays, "Jet_mass")
            jet_pt, _jet_mass, _jet_status = apply_jec(
                arrays,
                repo,
                source["year"],
                source["process"],
                "Jet",
                jet_pt_raw,
                jet_eta,
                jet_phi,
                jet_mass_raw,
                "nominal",
            )
            jet_id, _jet_id_source = ak4_tight_lepton_veto_mask(
                arrays, jet_pt, jet_eta, repo
            )
            good_jet = (jet_pt > 30.0) & (abs(jet_eta) < 2.4) & jet_id
            jet_btag = arr(arrays, "Jet_btagUParTAK4B")

            fj_pt_raw = arr(arrays, "FatJet_pt")
            fj_eta = arr(arrays, "FatJet_eta")
            fj_phi = arr(arrays, "FatJet_phi")
            fj_mass_raw = arr(arrays, "FatJet_mass")
            fj_msd = arr(arrays, "FatJet_msoftdrop")
            fj_pt, _fj_mass, _fj_status = apply_jec(
                arrays,
                repo,
                source["year"],
                source["process"],
                "FatJet",
                fj_pt_raw,
                fj_eta,
                fj_phi,
                fj_mass_raw,
                "nominal",
            )
            fj_id, _fj_id_source = ak8_tight_lepton_veto_mask(
                arrays, fj_pt, fj_eta, repo
            )
            top = (
                fj_id
                & (fj_pt > BOOSTED_TOP_PT_MIN)
                & (abs(fj_eta) < BOOSTED_ETA_MAX)
                & (fj_msd > BOOSTED_TOP_MSD_MIN)
                & (arr(arrays, BOOSTED_TOP_SCORE_BRANCH) > BOOSTED_TOP_SCORE_WP)
            )
            wjet = (
                fj_id
                & (fj_pt > BOOSTED_W_PT_MIN)
                & (abs(fj_eta) < BOOSTED_ETA_MAX)
                & (fj_msd > BOOSTED_W_MSD_MIN)
                & (fj_msd < BOOSTED_W_MSD_MAX)
                & (arr(arrays, BOOSTED_W_SCORE_BRANCH) > BOOSTED_W_SCORE_WP)
            )

            e_pt = arr(arrays, "Electron_pt")
            e_eta = arr(arrays, "Electron_eta")
            e_phi = arr(arrays, "Electron_phi")
            e_fid = (abs(e_eta) < 1.4442) | (
                (abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5)
            )
            e_medium = (
                (e_pt > 10.0)
                & e_fid
                & (arr(arrays, "Electron_cutBased") >= 3)
                & (arr(arrays, "Electron_miniPFRelIso_all") < 0.1)
            )
            m_pt = arr(arrays, "Muon_pt")
            m_eta = arr(arrays, "Muon_eta")
            m_phi = arr(arrays, "Muon_phi")
            m_medium = (
                (m_pt > 10.0)
                & (abs(m_eta) < 2.4)
                & (arr(arrays, "Muon_mediumId") != 0)
                & (arr(arrays, "Muon_miniPFRelIso_all") < 0.2)
            )
            channel_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for channel, lep_eta, lep_phi in (
                ("DY2E", e_eta[e_medium], e_phi[e_medium]),
                ("DY2M", m_eta[m_medium], m_phi[m_medium]),
            ):
                clean_fj = clean_by_delta_r(fj_eta, fj_phi, lep_eta, lep_phi, 0.4)
                pass_topology = np.asarray(
                    (ak.sum(top & clean_fj, axis=1) == 0)
                    & (ak.sum(wjet & clean_fj, axis=1) == 0),
                    dtype=bool,
                )
                clean_jet = clean_by_delta_r(
                    jet_eta, jet_phi, lep_eta, lep_phi, 0.2
                )
                nb = np.asarray(
                    ak.sum(
                        good_jet
                        & clean_jet
                        & (jet_btag > 0.1272),
                        axis=1,
                    ),
                    dtype=int,
                )
                channel_values[channel] = (pass_topology, nb)

            for entry in members:
                local = int(entry) - start
                for candidate in by_entry[int(entry)]:
                    if (
                        int(arrays["run"][local]) != int(candidate["run"])
                        or int(arrays["luminosityBlock"][local])
                        != int(candidate["luminosityBlock"])
                        or int(arrays["event"][local]) != int(candidate["event"])
                    ):
                        raise RuntimeError(
                            f"{source['file_path']}:{entry}: event key mismatch"
                        )
                    output["summary"]["matched"] += 1
                    channel = str(candidate["channel"])
                    pass_topology, nb = channel_values[channel]
                    if not pass_topology[local] or nb[local] < 1:
                        continue
                    group = "Nb1" if nb[local] == 1 else "Nb2plus"
                    window = str(candidate["mass_window"])
                    component = str(candidate["component"])
                    weight = float(candidate["flat_weight"])
                    mass = float(candidate["mass"])
                    add_yield(
                        nested_yield(output["raw"], (channel, group, window, component)),
                        [weight],
                    )
                    fill_histogram(
                        nested_histogram(output["mll"], (channel, group, component), MLL_EDGES),
                        np.asarray([mass]),
                        np.asarray([weight]),
                        np.asarray([True]),
                        MLL_EDGES,
                    )
                    output["summary"]["selected"] += 1
    finally:
        os.chdir(previous_cwd)
        try:
            if root_file is not None:
                root_file.close()
        finally:
            cleanup_xrd_cache(access_info)
    return output
