#!/usr/bin/env python3
"""Build the full-weight 12-bin TROTA component for the 2024 High-dM 61-bin study.

Only the adopted first High-dM category (Nb>=1, Nt=0, NW=0) is refilled.
Its six recoil bins are split into Nres=0 and Nres>=1 using the score-ordered,
AK4-jet-disjoint TROTA candidate multiplicity.  All standard event weights and
their variations are evaluated with the same implementation used by the
canonical flat-histogram production.  TROTA has no dedicated efficiency SF;
that limitation is recorded explicitly in the output provenance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

import build_flat_boosted_recoil_hists as base
from study_trota_highdm_categories_2024 import (
    TROTA_BRANCHES,
    TROTA_FALLBACK_ID_BRANCHES,
    TROTA_VALUE_BRANCHES,
    greedy_disjoint_counts,
    map_candidates_to_events,
    map_candidates_to_events_rle,
)


SCHEME = "trota_nres_split12_nb1plus_nt0_nw0_SR"
EXPECTED_FILES = 5489
EXPECTED_EVENTS = 230_830_776
EXPECTED_BTAG_SHA256 = "5a96f6b7dcd806a10c64dab7ecefd18a13767fa4645bc003bef5716798246563"
EXPECTED_SOURCE_SHA256 = {
    "autonomous_allhad/autonomous_allhad/analysis_scale_factors.py":
        "78b0e034c083fade6d1b225e37b12ecae94966297d2e33d5c5823531281aa0b8",
    "autonomous_allhad/autonomous_allhad/real_subset_worker.py":
        "417e418c6056877733c84d350d67b6a35a6c289977c4f847dfe1aabf520d91b3",
    "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py":
        "4ad1b9f542dc0892dd7ad4555544747f84ec98f0cfc3a216cd296011822e8572",
}
REQUIRED_COMPONENTS = (
    "pileup",
    "btagSF",
    "electron_id",
    "electron_reco",
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
)
ANALYSIS_SF_COMPONENTS = (
    "met_trigger",
    "photon_trigger",
    "veto_electron_5to10",
    "loose_muon_5to10",
)
SIGNAL_LABELS = {
    "T2tt_mStop1000_mLSP1",
    "T2tt_mStop1200_mLSP1",
}
DIRECT_WEIGHT_MASK_BRANCHES = {
    "feature_GCR",
    "feature_lowdm_GCR",
    "feature_LLCR",
    "feature_QCDCR",
    "feature_SR",
    "feature_lowdm_LLCR",
    "feature_lowdm_QCDCR",
    "feature_lowdm_SR",
}
MC_CATEGORY_CUT = (
    "feature_SR & (nb_medium >= 1) & (nboosted_top == 0) & "
    "(nboosted_w == 0) & (met >= 250) & (met < 1500) & (is_data == 0)"
)

_REPO: Path | None = None
_NORMALIZATION: dict[str, Any] | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_worker(repo: str, normalization: str) -> None:
    global _REPO, _NORMALIZATION
    _REPO = Path(repo)
    _NORMALIZATION = json.loads(Path(normalization).read_text())


def split_indices(recoil: np.ndarray, nres: np.ndarray) -> np.ndarray:
    recoil_values = np.asarray(recoil, dtype=float)
    nres_values = np.asarray(nres, dtype=int)
    recoil_index = (
        np.searchsorted(np.asarray(base.RECOIL_PT_BINS), recoil_values, side="right")
        - 1
    )
    valid = (
        np.isfinite(recoil_values)
        & (recoil_index >= 0)
        & (recoil_index < len(base.RECOIL_PT_BINS) - 1)
    )
    result = np.full(recoil_values.size, -1, dtype=np.int16)
    result[valid] = recoil_index[valid] + 6 * (nres_values[valid] >= 1)
    return result


def merge_histograms(target: dict[str, Any], source: dict[str, Any]) -> None:
    for sample, variations in source.items():
        for variation, record in variations.items():
            destination = (
                target.setdefault(sample, {})
                .setdefault(variation, base.empty_index_hist(12))
            )
            for field, dtype in (("sumw", float), ("sumw2", float), ("entries", int)):
                combined = (
                    np.asarray(destination[field], dtype=dtype)
                    + np.asarray(record[field], dtype=dtype)
                )
                destination[field] = combined.astype(dtype).tolist()


def _compact_component_status(status: dict[str, Any], events: int) -> dict[str, Any]:
    components = status.get("components") or {}
    return {
        component: {
            "applied_events": events if (components.get(component) or {}).get("applied") else 0,
            "failed_events": 0 if (components.get(component) or {}).get("applied") else events,
            "source": str((components.get(component) or {}).get("source") or ""),
        }
        for component in REQUIRED_COMPONENTS
    }


def empty_file_result(root_path: Path, event_count: int, started: float) -> dict[str, Any]:
    return {
        "input_root": str(root_path),
        "events": int(event_count),
        "events_selected": 0,
        "trota_rows": 0,
        "identity_fallback_files": 0,
        "weighted_groups": 0,
        "histograms": {},
        "component_audit": {},
        "wall_time_seconds": time.monotonic() - started,
    }


def drop_file_cache(path: Path) -> None:
    """Release completed EOS input pages from this user's memory cgroup.

    LXPLUS charges the FUSE-backed file cache to the user's cgroup.  A long
    sequential campaign can therefore hit the memory ceiling even though the
    workers' anonymous RSS remains small.  POSIX_FADV_DONTNEED is an I/O-only
    hint; it does not alter any event data or calculation.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _process_file(record: dict[str, Any]) -> dict[str, Any]:
    if _REPO is None or _NORMALIZATION is None:
        raise RuntimeError("worker was not initialized")
    started = time.monotonic()
    root_path = Path(str(record["input_root"]))
    metadata = base.read_root_metadata(root_path, fallback=_NORMALIZATION)
    histograms: dict[str, Any] = {}
    component_audit: dict[str, dict[str, int | str]] = {}
    events_selected = 0
    weighted_groups = 0
    trota_rows = 0
    identity_fallback_files = 0

    with uproot.open(root_path, object_cache=None, array_cache=None) as root_file:
        if "Events" not in root_file or "TROTA" not in root_file:
            raise RuntimeError(f"missing Events/TROTA tree in {root_path}")
        event_tree = root_file["Events"]
        present = set(event_tree.keys())
        # This component only evaluates event weights and the SR recoil index.
        # Reading the full distribution/Low-dM schema wastes EOS I/O and file
        # cache without contributing to any 61-bin quantity.
        requested = sorted(
            set(base.WEIGHT_BRANCHES)
            | DIRECT_WEIGHT_MASK_BRANCHES
            | {"file_id", "met"}
        )
        missing = sorted(
            branch
            for branch in requested
            if branch not in present and branch not in base.OPTIONAL_FORWARD_SCHEMA_BRANCHES
        )
        if missing:
            raise RuntimeError(f"required Events branches missing in {root_path}: {missing}")
        branches = [branch for branch in requested if branch in present]
        light_branches = [
            "run", "luminosityBlock", "event", "file_id", "entry",
            "dataset_id", "year", "mStop", "mLSP", "feature_SR",
            "nb_medium", "nboosted_top", "nboosted_w", "met",
        ]
        light = event_tree.arrays(light_branches, library="ak")
        event_count = len(light["entry"])
        if event_count == 0:
            return empty_file_result(root_path, 0, started)

        sr = np.asarray(light["feature_SR"], dtype=bool)
        category = (
            sr
            & (np.asarray(light["nb_medium"], dtype=int) >= 1)
            & (np.asarray(light["nboosted_top"], dtype=int) == 0)
            & (np.asarray(light["nboosted_w"], dtype=int) == 0)
        )
        recoil = np.asarray(light["met"], dtype=float)
        recoil_index = (
            np.searchsorted(np.asarray(base.RECOIL_PT_BINS), recoil, side="right") - 1
        )
        category &= (
            np.isfinite(recoil)
            & (recoil_index >= 0)
            & (recoil_index < len(base.RECOIL_PT_BINS) - 1)
        )

        dataset_ids = np.asarray(light["dataset_id"], dtype=np.int64)
        eligible = np.zeros(event_count, dtype=bool)
        for dataset_id in np.unique(dataset_ids[category]):
            dataset_mask = dataset_ids == dataset_id
            dataset, process, is_data, is_signal = base.dataset_label(
                metadata, int(dataset_id)
            )
            if is_data and not base.data_process_allowed(process, "SR"):
                continue
            if is_signal:
                topology = base.signal_topology(dataset) or "T2tt"
                if topology != "T2tt":
                    continue
                mstops = np.asarray(light["mStop"], dtype=int)
                mlsps = np.asarray(light["mLSP"], dtype=int)
                desired_point = (
                    ((mstops == 1000) & (mlsps == 1))
                    | ((mstops == 1200) & (mlsps == 1))
                )
                eligible |= category & dataset_mask & desired_point
            else:
                eligible |= category & dataset_mask

        if not np.any(eligible):
            return empty_file_result(root_path, event_count, started)

        counts = np.zeros(event_count, dtype=np.int16)
        trota_tree = root_file["TROTA"]
        missing_trota = sorted(
            (set(TROTA_BRANCHES) | set(TROTA_FALLBACK_ID_BRANCHES))
            - set(trota_tree.keys())
        )
        if missing_trota:
            raise RuntimeError(f"missing TROTA branches in {root_path}: {missing_trota}")
        try:
            trota_ak = trota_tree.arrays(TROTA_BRANCHES, library="ak")
            trota = {
                name: np.asarray(ak.to_numpy(trota_ak[name]))
                for name in TROTA_BRANCHES
            }
            candidate_event = map_candidates_to_events(
                np.asarray(light["file_id"]),
                np.asarray(light["entry"]),
                trota["file_id"],
                trota["entry"],
            )
            trota_rows = len(trota["entry"])
        except Exception as primary_error:
            fallback = TROTA_FALLBACK_ID_BRANCHES + TROTA_VALUE_BRANCHES
            try:
                trota_ak = trota_tree.arrays(fallback, library="ak")
                trota = {
                    name: np.asarray(ak.to_numpy(trota_ak[name]))
                    for name in fallback
                }
                candidate_event = map_candidates_to_events_rle(
                    np.asarray(light["run"]),
                    np.asarray(light["luminosityBlock"]),
                    np.asarray(light["event"]),
                    trota["run"],
                    trota["luminosityBlock"],
                    trota["event"],
                )
                trota_rows = len(trota["event"])
                identity_fallback_files = 1
            except Exception as fallback_error:
                raise RuntimeError(
                    "both TROTA identity joins failed; "
                    f"primary={type(primary_error).__name__}: {primary_error}; "
                    f"fallback={type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error
        counts = greedy_disjoint_counts(
            candidate_event,
            trota["TopResolved1pct_sourceJetIdx0"],
            trota["TopResolved1pct_sourceJetIdx1"],
            trota["TopResolved1pct_sourceJetIdx2"],
            trota["TopResolved1pct_QCDDiscriminant"],
            event_count,
            eligible[candidate_event],
        )

        # Data carries a unit nominal weight, so its 12-bin component needs
        # only the already-read scalar selection and TROTA arrays.  Avoid
        # materializing every jagged object branch for large JetMET files.
        # MC and signal still take the strict full AnalysisSF path below.
        data_events = np.zeros(event_count, dtype=bool)
        for dataset_id in np.unique(dataset_ids[category]):
            dataset_mask = eligible & (dataset_ids == dataset_id)
            dataset, process, is_data, _is_signal = base.dataset_label(
                metadata, int(dataset_id)
            )
            if not is_data:
                continue
            data_events |= dataset_ids == dataset_id
            if not np.any(dataset_mask):
                continue
            indices = split_indices(
                np.asarray(light["met"], dtype=float)[dataset_mask],
                counts[dataset_mask],
            )
            target = (
                histograms.setdefault("data_obs", {})
                .setdefault("nominal", base.empty_index_hist(12))
            )
            base.add_index_hist(target, indices, np.ones(int(np.sum(dataset_mask))))
            events_selected += int(np.sum(dataset_mask))
            weighted_groups += 1
            eligible[dataset_mask] = False

        if not np.any(eligible):
            return {
                "input_root": str(root_path),
                "events": int(event_count),
                "events_selected": int(events_selected),
                "trota_rows": int(trota_rows),
                "identity_fallback_files": int(identity_fallback_files),
                "weighted_groups": int(weighted_groups),
                "histograms": histograms,
                "component_audit": component_audit,
                "wall_time_seconds": time.monotonic() - started,
            }

        read_mask = category & ~data_events
        events = event_tree.arrays(branches, cut=MC_CATEGORY_CUT, library="ak")
        expected_entries = np.asarray(light["entry"])[read_mask]
        observed_entries = np.asarray(events["entry"])
        if not np.array_equal(observed_entries, expected_entries):
            raise RuntimeError(
                f"MC category cut identity mismatch in {root_path}: "
                f"observed={len(observed_entries)} expected={len(expected_entries)}"
            )
        eligible = eligible[read_mask]
        dataset_ids = dataset_ids[read_mask]
        counts = counts[read_mask]

        for dataset_id in np.unique(dataset_ids[eligible]):
            dataset_mask = eligible & (dataset_ids == dataset_id)
            dataset, process, is_data, is_signal = base.dataset_label(
                metadata, int(dataset_id)
            )
            if is_signal:
                mstops = np.asarray(events["mStop"], dtype=int)
                mlsps = np.asarray(events["mLSP"], dtype=int)
                point_masks = [
                    dataset_mask & (mstops == mstop) & (mlsps == mlsp)
                    for mstop, mlsp in sorted(
                        set(zip(mstops[dataset_mask].tolist(), mlsps[dataset_mask].tolist()))
                    )
                ]
            else:
                point_masks = [dataset_mask]
            for selected in point_masks:
                if not np.any(selected):
                    continue
                chunk = {name: events[name][selected] for name in ak.fields(events)}
                label = base.sample_label(process, is_data, is_signal, chunk, dataset)
                if is_signal and label not in SIGNAL_LABELS:
                    continue
                arrays, inputs = base.flat_arrays_for_weights(chunk)
                years = np.asarray(chunk["year"], dtype=int)
                if not np.all(years == 2024):
                    raise RuntimeError(f"non-2024 event in {root_path}: {np.unique(years)}")
                correction_dataset = dataset
                if is_signal:
                    correction_dataset = base.signal_btag_efficiency_dataset(
                        int(np.asarray(chunk["mStop"], dtype=int)[0]), dataset
                    )[1]
                normalization = base.norm_vector(
                    _NORMALIZATION,
                    chunk,
                    int(dataset_id),
                    dataset,
                    is_data,
                    is_signal,
                    require_normalization=True,
                )
                _gen, variations, status = base.compute_weight_bundle(
                    arrays,
                    _REPO,
                    correction_dataset,
                    process,
                    "2024",
                    inputs["n"],
                    inputs["jet_pt"], inputs["jet_eta"], inputs["jet_hadflav"], inputs["b_med"],
                    inputs["e_eta"], inputs["e_delta_eta_sc"], inputs["e_pt"], inputs["e_phi"],
                    inputs["e_veto"], inputs["e_med"], inputs["n_e_veto"], inputs["n_e_med"],
                    inputs["m_eta"], inputs["m_pt"], inputs["m_phi"], inputs["m_loose"],
                    inputs["m_med"], inputs["n_m_loose"], inputs["n_m_med"],
                    inputs["p_eta"], inputs["p_pt"], inputs["p_phi"], inputs["p_med"],
                    inputs["gcr_mask"],
                    p_r9=inputs["p_r9"],
                    met_pt=inputs["met_pt"],
                    met_trigger_mask=inputs["met_trigger_mask"],
                    analysis_sf_components=ANALYSIS_SF_COMPONENTS,
                )
                if not is_data:
                    for component in REQUIRED_COMPONENTS:
                        component_status = (status.get("components") or {}).get(component) or {}
                        if not component_status.get("applied"):
                            raise RuntimeError(
                                f"required component {component} unavailable for {dataset}: "
                                f"{component_status}"
                            )
                    if "nominal" not in variations:
                        raise RuntimeError(f"nominal weight missing for {dataset}")
                if not is_data:
                    for component, audit in _compact_component_status(
                        status, inputs["n"]
                    ).items():
                        merged = component_audit.setdefault(
                            component,
                            {"applied_events": 0, "failed_events": 0, "source": audit["source"]},
                        )
                        merged["applied_events"] = int(merged["applied_events"]) + int(audit["applied_events"])
                        merged["failed_events"] = int(merged["failed_events"]) + int(audit["failed_events"])
                indices = split_indices(
                    np.asarray(chunk["met"], dtype=float), counts[selected]
                )
                for variation, raw_weight in variations.items():
                    weights = np.asarray(raw_weight, dtype=float) * normalization
                    if len(weights) != inputs["n"]:
                        raise RuntimeError(
                            f"wrong {variation} weight length in {root_path}: "
                            f"{len(weights)} != {inputs['n']}"
                        )
                    if not np.all(np.isfinite(weights)):
                        raise RuntimeError(f"non-finite {variation} weight in {root_path}")
                    if np.any(np.abs(weights) > base.MAX_ABS_HIST_WEIGHT):
                        raise RuntimeError(f"excessive {variation} weight in {root_path}")
                    target = (
                        histograms.setdefault(label, {})
                        .setdefault(variation, base.empty_index_hist(12))
                    )
                    base.add_index_hist(target, indices, weights)
                events_selected += inputs["n"]
                weighted_groups += 1

    return {
        "input_root": str(root_path),
        "events": int(event_count),
        "events_selected": int(events_selected),
        "trota_rows": int(trota_rows),
        "identity_fallback_files": int(identity_fallback_files),
        "weighted_groups": int(weighted_groups),
        "histograms": histograms,
        "component_audit": component_audit,
        "wall_time_seconds": time.monotonic() - started,
    }


def process_file(record: dict[str, Any]) -> dict[str, Any]:
    root_path = Path(str(record["input_root"]))
    try:
        return _process_file(record)
    finally:
        drop_file_cache(root_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--only-input-root", action="append", default=[])
    parser.add_argument("--expected-files", type=int, default=EXPECTED_FILES)
    parser.add_argument("--expected-events", type=int, default=EXPECTED_EVENTS)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text())
    normalization = json.loads(args.normalization.read_text())
    inputs = list(manifest.get("inputs") or [])
    if args.only_input_root:
        requested = set(args.only_input_root)
        inputs = [item for item in inputs if str(item.get("input_root")) in requested]
        found = {str(item.get("input_root")) for item in inputs}
        if found != requested:
            raise RuntimeError(f"manifest inputs missing: {sorted(requested - found)}")
    if args.max_files:
        inputs = inputs[: args.max_files]
    full_campaign = not args.max_files and not args.only_input_root
    if full_campaign and len(inputs) != args.expected_files:
        raise RuntimeError(f"expected {args.expected_files} inputs, found {len(inputs)}")
    files_expected = len(inputs)
    if normalization.get("status") != "complete":
        raise RuntimeError("normalization manifest is not complete")

    source_sha256 = {
        relative: sha256(args.repo / relative) for relative in EXPECTED_SOURCE_SHA256
    }
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"weight source hash drift: expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    btag_path = args.repo / "analysis/hists/btageff2024.merged"
    btag_sha256 = sha256(btag_path)
    if btag_sha256 != EXPECTED_BTAG_SHA256:
        raise RuntimeError(
            f"btag efficiency checksum drift: {btag_sha256} != {EXPECTED_BTAG_SHA256}"
        )

    input_manifest_sha256 = sha256(args.input_manifest)
    normalization_sha256 = sha256(args.normalization)
    started = time.monotonic()
    previous_wall_time = 0.0
    completed_input_roots: list[str] = []
    totals: defaultdict[str, int] = defaultdict(int)
    failures: list[dict[str, str]] = []
    histograms: dict[str, Any] = {}
    component_audit: dict[str, dict[str, int | str]] = {}
    if args.resume:
        if not args.output.is_file():
            raise RuntimeError(f"resume output does not exist: {args.output}")
        previous = json.loads(args.output.read_text())
        if previous.get("schema_version") != "trota_highdm61_fullweights_2024_v1":
            raise RuntimeError("resume output schema mismatch")
        if previous.get("input_manifest_sha256") != input_manifest_sha256:
            raise RuntimeError("resume input manifest checksum mismatch")
        if previous.get("normalization_sha256") != normalization_sha256:
            raise RuntimeError("resume normalization checksum mismatch")
        if previous.get("source_sha256") != source_sha256:
            raise RuntimeError("resume source checksum mismatch")
        if previous.get("btag_efficiency_sha256") != btag_sha256:
            raise RuntimeError("resume btag checksum mismatch")
        if previous.get("failures"):
            raise RuntimeError("refusing to resume an output containing failures")
        completed_input_roots = list(previous.get("completed_input_roots") or [])
        if len(completed_input_roots) != int(previous.get("files_completed") or 0):
            raise RuntimeError("resume output lacks an exact completed-input manifest")
        if len(set(completed_input_roots)) != len(completed_input_roots):
            raise RuntimeError("resume output contains duplicate completed inputs")
        known_inputs = {str(item.get("input_root")) for item in inputs}
        unknown_completed = set(completed_input_roots) - known_inputs
        if unknown_completed:
            raise RuntimeError(f"resume output contains unknown inputs: {sorted(unknown_completed)}")
        totals.update({
            str(key): int(value)
            for key, value in (previous.get("totals") or {}).items()
        })
        histograms = (
            (previous.get("search_bin_histograms") or {}).get(SCHEME) or {}
        )
        component_audit = previous.get("component_audit") or {}
        previous_wall_time = float(previous.get("wall_time_seconds") or 0.0)
        completed_set = set(completed_input_roots)
        inputs = [
            item for item in inputs
            if str(item.get("input_root")) not in completed_set
        ]
    completed = len(completed_input_roots)

    def snapshot(status: str) -> dict[str, Any]:
        return {
            "schema_version": "trota_highdm61_fullweights_2024_v1",
            "status": status,
            "updated_at": now(),
            "repo": str(args.repo),
            "input_manifest": str(args.input_manifest),
            "input_manifest_sha256": input_manifest_sha256,
            "normalization": str(args.normalization),
            "normalization_sha256": normalization_sha256,
            "source_sha256": source_sha256,
            "builder_sha256": sha256(Path(__file__).resolve()),
            "btag_efficiency_sha256": btag_sha256,
            "files_expected": files_expected,
            "files_completed": completed,
            "files_pending": files_expected - completed,
            "completed_input_roots": completed_input_roots,
            "workers": args.workers,
            "totals": dict(totals),
            "failures": failures,
            "required_components": list(REQUIRED_COMPONENTS),
            "analysis_sf_components": list(ANALYSIS_SF_COMPONENTS),
            "component_audit": component_audit,
            "search_bin_schemes": {
                SCHEME: {
                    "bin_labels": [
                        f"Nb1plus_Nt0_NW0_Nres{nres}_recoil_{label}"
                        for nres in ("0", "1plus")
                        for label in base.RECOIL_BIN_LABELS
                    ],
                    "category_sizes": [6, 6],
                    "recoil_pt_bins": base.RECOIL_PT_BINS,
                    "selection": "feature_SR && Nb>=1 && Nt=0 && NW=0, split by jet-disjoint TROTA Nres=0/≥1",
                    "trota_scale_factor": "unavailable; no dedicated TROTA SF applied",
                }
            },
            "search_bin_histograms": {SCHEME: histograms},
            "wall_time_seconds": previous_wall_time + time.monotonic() - started,
        }

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.repo), str(args.normalization)),
    ) as executor:
        futures = {executor.submit(process_file, item): item for item in inputs}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({
                    "input_root": str(item.get("input_root")),
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                })
            else:
                completed += 1
                completed_input_roots.append(str(item.get("input_root")))
                for field in (
                    "events", "events_selected", "trota_rows",
                    "identity_fallback_files", "weighted_groups",
                ):
                    totals[field] += int(result[field])
                merge_histograms(histograms, result["histograms"])
                for component, audit in result["component_audit"].items():
                    merged = component_audit.setdefault(
                        component,
                        {"applied_events": 0, "failed_events": 0, "source": audit["source"]},
                    )
                    merged["applied_events"] = int(merged["applied_events"]) + int(audit["applied_events"])
                    merged["failed_events"] = int(merged["failed_events"]) + int(audit["failed_events"])
            if completed % 10 == 0 or failures or completed == files_expected:
                atomic_json(args.output, snapshot("running"))

    status = "complete"
    if failures or completed != files_expected:
        status = "failed"
    if full_campaign and int(totals["events"]) != args.expected_events:
        failures.append({
            "input_root": "campaign",
            "error": f"event count {totals['events']} != expected {args.expected_events}",
        })
        status = "failed"
    if full_campaign and int(totals["identity_fallback_files"]) != 1:
        failures.append({
            "input_root": "campaign",
            "error": "expected exactly one validated TROTA identity fallback file",
        })
        status = "failed"
    for component in REQUIRED_COMPONENTS:
        audit = component_audit.get(component) or {}
        if (
            int(audit.get("failed_events") or 0) != 0
            or (full_campaign and int(audit.get("applied_events") or 0) <= 0)
        ):
            failures.append({
                "input_root": "campaign",
                "error": f"required component {component} has failed events: {audit}",
            })
            status = "failed"
    for sample, variations in histograms.items():
        for variation, record in variations.items():
            for field in ("sumw", "sumw2"):
                if not np.all(np.isfinite(np.asarray(record[field], dtype=float))):
                    failures.append({
                        "input_root": "campaign",
                        "error": f"non-finite {sample}/{variation}/{field}",
                    })
                    status = "failed"
    payload = snapshot(status)
    atomic_json(args.output, payload)
    print(json.dumps({
        "status": status,
        "files_completed": completed,
        "events": totals["events"],
        "events_selected": totals["events_selected"],
        "failures": len(failures),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
