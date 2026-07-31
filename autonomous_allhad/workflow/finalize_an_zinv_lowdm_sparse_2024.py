#!/usr/bin/env python3
"""Finalize Low-dM R_Z/R_T with sparse NanoAOD object reconstruction."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from autonomous_allhad.dycr_sparse_nano import (  # noqa: E402
    group_sparse_windows,
    stable_file_id,
)
from autonomous_allhad.real_subset_worker import (  # noqa: E402
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
    CORE_BRANCHES,
    ELECTRON_HLT,
    FATJET_ID_INPUTS,
    FILTERS,
    JET_ID_INPUTS,
    MUON_HLT,
    PHOTON_HLT,
    SIGNAL_HLT,
    ak4_tight_lepton_veto_mask,
    ak8_tight_lepton_veto_mask,
    apply_jec,
    arr,
    cleanup_xrd_cache,
    clean_by_delta_r,
    extract_chunk,
    open_root_with_xrd_fallback,
)
LIGHT_WORKER = os.environ.get("AUTONOMOUS_ALLHAD_LOWDMSPARSE_LIGHT", "0") == "1"
if not LIGHT_WORKER:
    from build_an_zinv_measurement_inputs_2024 import (  # noqa: E402
        CHANNELS,
        MLL_EDGES,
        MASS_WINDOWS,
        RZ_LOW_UT_EDGES,
        fill_histogram,
        empty_yield,
        finalize_rz,
        finalize_rz_ut,
        merge_tree,
        nested_histogram,
        nested_yield,
        add_yield,
    )
else:
    CHANNELS = ("DY2E", "DY2M")
    MASS_WINDOWS = ("on", "off")
    RZ_LOW_UT_EDGES = np.asarray(
        [300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1500.0], dtype=float
    )
    MLL_EDGES = np.asarray(
        [50.0, 70.0, 81.0, 91.0, 101.0, 120.0, 160.0, 250.0, 500.0],
        dtype=float,
    )

    def empty_yield() -> dict[str, Any]:
        return {"sumw": 0.0, "sumw2": 0.0, "entries": 0}

    def add_yield(target: dict[str, Any], weights: Any) -> None:
        selected = np.asarray(weights, dtype=float)
        selected = selected[np.isfinite(selected)]
        target["sumw"] = float(target["sumw"]) + float(np.sum(selected))
        target["sumw2"] = float(target["sumw2"]) + float(
            np.sum(selected * selected)
        )
        target["entries"] = int(target["entries"]) + int(len(selected))

    def nested_yield(
        payload: dict[str, Any], keys: tuple[str, ...]
    ) -> dict[str, Any]:
        target = payload
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        return target.setdefault(keys[-1], empty_yield())

    def empty_histogram(edges: np.ndarray) -> dict[str, Any]:
        size = len(edges) - 1
        return {
            "edges": edges.tolist(),
            "sumw": [0.0] * size,
            "sumw2": [0.0] * size,
            "entries": [0] * size,
        }

    def nested_histogram(
        payload: dict[str, Any], keys: tuple[str, ...], edges: np.ndarray
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
        selected = weights[valid]
        target["sumw"] = (
            np.asarray(target["sumw"], dtype=float)
            + np.bincount(indices, weights=selected, minlength=len(edges) - 1)
        ).tolist()
        target["sumw2"] = (
            np.asarray(target["sumw2"], dtype=float)
            + np.bincount(
                indices, weights=selected * selected, minlength=len(edges) - 1
            )
        ).tolist()
        target["entries"] = (
            np.asarray(target["entries"], dtype=int)
            + np.bincount(indices, minlength=len(edges) - 1)
        ).tolist()

    def merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
        if set(source) >= {"sumw", "sumw2", "entries"}:
            for key in ("sumw", "sumw2", "entries"):
                if isinstance(source[key], list):
                    target[key] = (
                        np.asarray(target[key]) + np.asarray(source[key])
                    ).tolist()
                elif key == "entries":
                    target[key] = int(target[key]) + int(source[key])
                else:
                    target[key] = float(target[key]) + float(source[key])
            return
        for key, value in source.items():
            if not isinstance(value, dict):
                continue
            if set(value) >= {"sumw", "sumw2", "entries"}:
                initial = (
                    empty_histogram(np.asarray(value["edges"], dtype=float))
                    if isinstance(value.get("sumw"), list) and "edges" in value
                    else empty_yield()
                )
                merge_tree(target.setdefault(key, initial), value)
            else:
                merge_tree(target.setdefault(key, {}), value)


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
    source = task["source"]
    candidates = task["candidates"]
    repo = Path(task["repo"])
    max_span = int(task["max_span"])
    max_gap = int(task["max_gap"])
    output: dict[str, Any] = {
        "raw": {},
        "raw_ut": {},
        "mll": {},
        "summary": {
            "file_id": int(task["file_id"]),
            "file_path": source["file_path"],
            "targets": len(candidates),
            "matched": 0,
            "selected": 0,
            "windows": 0,
        },
    }
    by_entry: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_entry.setdefault(int(candidate["entry"]), []).append(candidate)
    root_file = None
    access_info: dict[str, Any] = {}
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        direct_path = sparse_access_path(source["file_path"])
        try:
            root_file = uproot.open(direct_path, timeout=90)
            access_info = {
                "access_method": "direct_sparse_xrootd",
                "source_file_path": source["file_path"],
                "effective_file_path": direct_path,
            }
        except Exception as direct_error:
            root_file, access_info = open_root_with_xrd_fallback(
                source["file_path"], timeout=90
            )
            access_info["direct_sparse_path"] = direct_path
            access_info["direct_sparse_error"] = (
                f"{type(direct_error).__name__}: {direct_error}"
            )
        tree = root_file["Events"]
        present = set(tree.keys())
        genmodel = [
            name
            for name in present
            if str(name).startswith("GenModel_T2tt_")
        ]
        requested = set(
            CORE_BRANCHES
            + FILTERS
            + SIGNAL_HLT
            + PHOTON_HLT
            + ELECTRON_HLT
            + MUON_HLT
            + genmodel
        )
        branches = sorted(requested & present)
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
            rows, _summary = extract_chunk(
                arrays,
                source["dataset"],
                source["process"],
                None,
                source["year"],
                source["file_path"],
                start,
                stop,
                fastsim_trigger_bypass=False,
                shift_name="nominal",
                compute_weights=False,
                materialize_skim_flag=None,
                dy_mass_window="measurement",
            )
            rows_by_entry = {int(row["entry"]): row for row in rows}
            for entry in members:
                row = rows_by_entry.get(int(entry))
                if row is None:
                    raise RuntimeError(
                        f"{source['file_path']}:{entry}: row not materialized"
                    )
                for candidate in by_entry[int(entry)]:
                    if (
                        int(row["run"]) != int(candidate["run"])
                        or int(row["luminosityBlock"])
                        != int(candidate["luminosityBlock"])
                        or int(row["event"]) != int(candidate["event"])
                    ):
                        raise RuntimeError(
                            f"{source['file_path']}:{entry}: event key mismatch"
                        )
                    output["summary"]["matched"] += 1
                    channel = str(candidate["channel"])
                    if channel not in CHANNELS:
                        raise RuntimeError(f"unsupported channel {channel}")
                    if not bool(row[f"feature_lowdm_{channel}"]):
                        continue
                    raw_bin = int(row[f"lowdm_search_bin_{channel}"])
                    remapped = raw_bin - 8 if raw_bin >= 8 else -1
                    if remapped < 0 or remapped >= 34:
                        continue
                    group = "Nb1" if remapped < 16 else "Nb2plus"
                    mass = float(row["mee" if channel == "DY2E" else "mmm"])
                    if not np.isfinite(mass) or mass <= 50.0:
                        continue
                    window = "on" if 81.0 < mass < 101.0 else "off"
                    if window not in MASS_WINDOWS:
                        raise RuntimeError(
                            f"unsupported mass window {window}"
                        )
                    if window != str(candidate["mass_window"]):
                        raise RuntimeError(
                            f"{source['file_path']}:{entry}: sparse/feature "
                            f"mass-window mismatch ({window} versus "
                            f"{candidate['mass_window']})"
                        )
                    component = str(candidate["component"])
                    recoil = float(row[f"recoil_{channel.lower()}"])
                    add_yield(
                        nested_yield(
                            output["raw"],
                            (channel, group, window, component),
                        ),
                        [float(candidate["flat_weight"])],
                    )
                    if np.isfinite(recoil) and recoil >= RZ_LOW_UT_EDGES[0]:
                        ut_bin = int(
                            np.searchsorted(
                                RZ_LOW_UT_EDGES,
                                recoil,
                                side="right",
                            )
                            - 1
                        )
                        ut_bin = min(ut_bin, len(RZ_LOW_UT_EDGES) - 2)
                        add_yield(
                            nested_yield(
                                output["raw_ut"],
                                (
                                    channel,
                                    str(ut_bin),
                                    window,
                                    component,
                                ),
                            ),
                            [float(candidate["flat_weight"])],
                        )
                    fill_histogram(
                        nested_histogram(
                            output["mll"],
                            (channel, group, component),
                            MLL_EDGES,
                        ),
                        np.asarray([mass]),
                        np.asarray([float(candidate["flat_weight"])]),
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


def process_source_topology_only(task: dict[str, Any]) -> dict[str, Any]:
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
        "raw_ut": {},
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
                    recoil = float(candidate["flat_recoil"])
                    mass = float(candidate["mass"])
                    add_yield(
                        nested_yield(output["raw"], (channel, group, window, component)),
                        [weight],
                    )
                    if np.isfinite(recoil) and recoil >= RZ_LOW_UT_EDGES[0]:
                        ut_bin = int(
                            np.searchsorted(RZ_LOW_UT_EDGES, recoil, side="right") - 1
                        )
                        ut_bin = min(ut_bin, len(RZ_LOW_UT_EDGES) - 2)
                        add_yield(
                            nested_yield(
                                output["raw_ut"],
                                (channel, str(ut_bin), window, component),
                            ),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--feature-inputs", type=Path, required=True)
    parser.add_argument("--shard-bundle", type=Path, required=True)
    parser.add_argument(
        "--source-feature-list", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--max-span", type=int, default=50000)
    parser.add_argument("--max-gap", type=int, default=5000)
    args = parser.parse_args()

    feature = read_json(args.feature_inputs)
    if feature.get("status") not in {
        "feature_stage_complete",
        "feature_stage_complete_with_missing_roots",
    }:
        raise SystemExit(
            f"feature-stage input is not complete: {feature.get('status')}"
        )
    feature_roots = [
        line.strip()
        for source_list in args.source_feature_list
        for line in source_list.read_text().splitlines()
        if line.strip()
    ]
    candidates_by_id = feature.get("sparse_low_candidates") or {}
    candidate_ids = {int(file_id) for file_id in candidates_by_id}
    mapping = source_map(args.shard_bundle, feature_roots, candidate_ids)
    unresolved = sorted(
        int(file_id)
        for file_id in candidates_by_id
        if int(file_id) not in mapping
    )
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} candidate source file IDs are unresolved: "
            + ",".join(str(value) for value in unresolved[:20])
        )
    merged: dict[str, Any] = {
        "schema_version": "an_zinv_lowdm_sparse_2024_v1",
        "status": "running",
        "rz_low_raw": json.loads(
            json.dumps(feature.get("rz_low_feature_raw") or {})
        ),
        "rz_low_ut_raw": json.loads(
            json.dumps(feature.get("rz_low_feature_ut_raw") or {})
        ),
        "mll_low": json.loads(
            json.dumps(feature.get("mll_low_feature") or {})
        ),
        "summary": {
            "candidate_files": len(candidates_by_id),
            "candidate_events": sum(
                len(records) for records in candidates_by_id.values()
            ),
            "completed_files": 0,
            "matched_events": 0,
            "selected_events": 0,
            "read_windows": 0,
            "failures": [],
        },
        "provenance": {
            "feature_inputs": str(args.feature_inputs),
            "shard_bundle": str(args.shard_bundle),
            "source_feature_list": [
                str(path) for path in args.source_feature_list
            ],
            "max_span": args.max_span,
            "max_gap": args.max_gap,
            "jobs": args.jobs,
            "dy_mass_window": (
                "mll>50; on/off classification recomputed from canonical sparse Nano row"
            ),
        },
    }
    tasks = [
        {
            "file_id": int(file_id),
            "source": mapping[int(file_id)],
            "candidates": records,
            "repo": str(args.repo),
            "max_span": args.max_span,
            "max_gap": args.max_gap,
        }
        for file_id, records in candidates_by_id.items()
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.jobs)
    ) as executor:
        futures = {
            executor.submit(process_source, task): task for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                merged["summary"]["failures"].append(
                    {
                        "file_id": task["file_id"],
                        "file_path": task["source"]["file_path"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            merged["summary"]["completed_files"] += 1
            summary = result["summary"]
            merged["summary"]["matched_events"] += int(
                summary["matched"]
            )
            merged["summary"]["selected_events"] += int(
                summary["selected"]
            )
            merged["summary"]["read_windows"] += int(summary["windows"])
            merge_tree(merged["rz_low_raw"], result["raw"])
            merge_tree(merged["rz_low_ut_raw"], result["raw_ut"])
            merge_tree(merged["mll_low"], result["mll"])
            completed = int(merged["summary"]["completed_files"])
            if completed and completed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed_files": completed,
                            "total_files": len(tasks),
                            "matched_events": merged["summary"][
                                "matched_events"
                            ],
                            "selected_events": merged["summary"][
                                "selected_events"
                            ],
                            "failures": len(
                                merged["summary"]["failures"]
                            ),
                        }
                    ),
                    flush=True,
                )
    merged["rz_low"] = finalize_rz(merged["rz_low_raw"])
    channels = tuple(
        str(channel)
        for channel in (
            (feature.get("provenance") or {}).get("channels")
            or CHANNELS
        )
    )
    merged["rz_low_ut"] = finalize_rz_ut(
        merged["rz_low_ut_raw"], RZ_LOW_UT_EDGES, channels
    )
    complete = (
        not merged["summary"]["failures"]
        and merged["summary"]["matched_events"]
        == merged["summary"]["candidate_events"]
    )
    merged["status"] = "complete" if complete else "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, sort_keys=True, separators=(",", ":"))
    )
    print(
        json.dumps(
            {
                "status": merged["status"],
                "output": str(args.output),
                "summary": merged["summary"],
                "rz_low": merged["rz_low"]["combined"],
            }
        )
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
