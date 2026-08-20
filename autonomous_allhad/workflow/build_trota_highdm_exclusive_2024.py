#!/usr/bin/env python3
"""Build full-weight 2024 High-dM histograms with exclusive TROTA Nres bins.

The source population is the adopted 55-bin High-dM scheme, restricted to the
Run-2 high-mTb regime (mTb >= 175 GeV).  Existing bins explicitly contain only
Nres=0.  Each Nres>0 event is removed from its former bin and assigned once to
one of five coarse Run-2-inspired (Nt, NW, Nres) blocks.  Both a fine 85-bin
diagnostic and an 80-bin tail-merged proposal are emitted.
"""

from __future__ import annotations

import argparse
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
    map_candidates_to_events,
    map_candidates_to_events_rle,
)
from autonomous_allhad.highdm_nres_binning import (
    RECOIL_LABELS,
    adopted55_labels,
    exclusive85_indices,
    exclusive85_labels,
    map60_indices_to_adopted55,
    map85_indices_to_tailmerged80,
    tailmerged80_labels,
)
from autonomous_allhad.highdm_resolved_categories import (
    COARSE_NRES_TOPOLOGIES,
    boosted_overlap_vetoed_ak4_indices,
    select_exclusive_resolved_candidates,
)


SCHEMA_VERSION = "trota_highdm_exclusive_2024_chunk_v1"
BASELINE_SCHEME = "highdm55_mtb175_inclusive_nres_SR"
EXCLUSIVE85_SCHEME = "highdm85_mtb175_exclusive_nres_SR"
TAILMERGED80_SCHEME = "highdm80_mtb175_exclusive_nres_tailmerged_SR"
EXPECTED_INPUT_SCHEMA = "flat_ntuple_shard_v8_float32_fullselection_2024_trota"
EXPECTED_TROTA_SCHEMA = "trota_topresolved_2024_inplace_sparse_v1"
EXPECTED_TROTA_MODEL_SHA256 = (
    "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
)
EXPECTED_BTAG_SHA256 = (
    "5a96f6b7dcd806a10c64dab7ecefd18a13767fa4645bc003bef5716798246563"
)
EXPECTED_PHYSICS_SOURCE_SHA256 = {
    "autonomous_allhad/autonomous_allhad/analysis_scale_factors.py":
        "78b0e034c083fade6d1b225e37b12ecae94966297d2e33d5c5823531281aa0b8",
    "autonomous_allhad/autonomous_allhad/real_subset_worker.py":
        "417e418c6056877733c84d350d67b6a35a6c289977c4f847dfe1aabf520d91b3",
}
REQUIRED_COMPONENTS = (
    "pileup", "btagSF", "electron_id", "electron_reco", "electron_hlt",
    "muon_id", "muon_iso", "muon_hlt", "photon_id", "photon_csev",
    "met_trigger", "photon_trigger", "veto_electron_5to10",
    "loose_muon_5to10",
)
ANALYSIS_SF_COMPONENTS = (
    "met_trigger", "photon_trigger", "veto_electron_5to10",
    "loose_muon_5to10",
)
SIGNAL_LABELS = {
    "T2tt_mStop1000_mLSP1",
    "T2tt_mStop1200_mLSP1",
}
DIRECT_WEIGHT_MASK_BRANCHES = {
    "feature_GCR", "feature_lowdm_GCR", "feature_LLCR", "feature_QCDCR",
    "feature_SR", "feature_lowdm_LLCR", "feature_lowdm_QCDCR",
    "feature_lowdm_SR",
}
LIGHT_SCALAR_BRANCHES = (
    "run", "luminosityBlock", "event", "file_id", "entry", "dataset_id",
    "year", "mStop", "mLSP", "is_data", "is_signal", "feature_SR",
    "nb_medium", "nboosted_top", "nboosted_w", "nboosted_total", "met",
    "lowdm_mtb",
)
OVERLAP_BRANCHES = (
    "jet_source_index_all", "jet_eta_all", "jet_phi_all",
    "fatjet_eta_all", "fatjet_phi_all", "fatjet_subjet_index1_all",
    "fatjet_subjet_index2_all", "fatjet_boosted_top_pass_all",
    "fatjet_boosted_w_pass_all", "subjet_eta_all", "subjet_phi_all",
)
TROTA_PRIMARY_BRANCHES = (
    "file_id", "entry", "TopResolved1pct_candidateIndex",
    "TopResolved1pct_sourceJetIdx0", "TopResolved1pct_sourceJetIdx1",
    "TopResolved1pct_sourceJetIdx2", "TopResolved1pct_eta",
    "TopResolved1pct_mass", "TopResolved1pct_QCDDiscriminant",
)
TROTA_FALLBACK_BRANCHES = (
    "run", "luminosityBlock", "event", *TROTA_PRIMARY_BRANCHES[2:],
)
MC_PRESELECTION_CUT = (
    "feature_SR & (lowdm_mtb >= 175) & (met >= 250) & (met < 1500) "
    "& (is_data == 0)"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def drop_file_cache(path: Path) -> None:
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


def read_input_list(path: Path) -> list[dict[str, str]]:
    inputs = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) == 1:
            local_root = source_root = fields[0]
        elif len(fields) == 2:
            local_root, source_root = fields
        else:
            raise ValueError(f"invalid input-list line {line_number}: {raw!r}")
        inputs.append({"input_root": local_root, "source_root": source_root})
    if not inputs:
        raise ValueError(f"no inputs in {path}")
    if len({item["source_root"] for item in inputs}) != len(inputs):
        raise ValueError("duplicate source ROOT in input list")
    return inputs


def validate_sidecar(root_path: Path) -> dict[str, Any]:
    sidecar = root_path.with_suffix(".json")
    if not sidecar.is_file() or sidecar.stat().st_size == 0:
        raise RuntimeError(f"missing nonempty input sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"input sidecar is not complete: {sidecar}")
    if payload.get("schema_version") != EXPECTED_INPUT_SCHEMA:
        raise RuntimeError(
            f"input schema drift in {sidecar}: {payload.get('schema_version')}"
        )
    trota = payload.get("trota_topresolved_2024") or {}
    marker = trota.get("marker") or {}
    if trota.get("status") != "complete" or marker.get("status") != "complete":
        raise RuntimeError(f"TROTA completion marker missing in {sidecar}")
    if trota.get("schema_version") != EXPECTED_TROTA_SCHEMA:
        raise RuntimeError(f"TROTA schema drift in {sidecar}")
    if marker.get("model_sha256") != EXPECTED_TROTA_MODEL_SHA256:
        raise RuntimeError(f"TROTA model checksum drift in {sidecar}")
    return payload


def merge_histograms(target: dict[str, Any], source: dict[str, Any]) -> None:
    for scheme, samples in source.items():
        for sample, variations in samples.items():
            for variation, record in variations.items():
                bins = len(record["sumw"])
                destination = (
                    target.setdefault(scheme, {})
                    .setdefault(sample, {})
                    .setdefault(variation, base.empty_index_hist(bins))
                )
                if len(destination["sumw"]) != bins:
                    raise RuntimeError(f"histogram length drift for {scheme}/{sample}/{variation}")
                for field, dtype in (("sumw", float), ("sumw2", float), ("entries", int)):
                    values = np.asarray(destination[field], dtype=dtype) + np.asarray(record[field], dtype=dtype)
                    destination[field] = values.astype(dtype).tolist()


def component_status(status: dict[str, Any], events: int) -> dict[str, Any]:
    components = status.get("components") or {}
    return {
        component: {
            "applied_events": events if (components.get(component) or {}).get("applied") else 0,
            "failed_events": 0 if (components.get(component) or {}).get("applied") else events,
            "source": str((components.get(component) or {}).get("source") or ""),
        }
        for component in REQUIRED_COMPONENTS
    }


def recoil_indices(met: np.ndarray) -> np.ndarray:
    values = np.asarray(met, dtype=float)
    result = np.searchsorted(np.asarray(base.RECOIL_PT_BINS), values, side="right") - 1
    return result.astype(np.int16, copy=False)


def accepted_dataset_mask(
    metadata: dict[str, Any],
    light: ak.Array,
    population: np.ndarray,
) -> np.ndarray:
    dataset_ids = np.asarray(light["dataset_id"], dtype=np.int64)
    accepted = np.zeros(len(dataset_ids), dtype=bool)
    for dataset_id in np.unique(dataset_ids[population]):
        dataset, process, is_data, is_signal = base.dataset_label(metadata, int(dataset_id))
        mask = population & (dataset_ids == dataset_id)
        if is_data:
            if base.data_process_allowed(process, "SR"):
                accepted |= mask
            continue
        if not is_signal:
            accepted |= mask
            continue
        if (base.signal_topology(dataset) or "T2tt") != "T2tt":
            continue
        mstop = np.asarray(light["mStop"], dtype=int)
        mlsp = np.asarray(light["mLSP"], dtype=int)
        accepted |= mask & (
            ((mstop == 1000) & (mlsp == 1))
            | ((mstop == 1200) & (mlsp == 1))
        )
    return accepted


def event_nres(
    light: ak.Array,
    trota_tree: Any,
    eligible: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    number_events = len(eligible)
    counts = np.zeros(number_events, dtype=np.int16)
    tree_fields = set(trota_tree.keys())
    identity_fallback = 0
    if set(TROTA_PRIMARY_BRANCHES) <= tree_fields:
        arrays_ak = trota_tree.arrays(TROTA_PRIMARY_BRANCHES, library="ak")
        arrays = {name: np.asarray(ak.to_numpy(arrays_ak[name])) for name in TROTA_PRIMARY_BRANCHES}
        candidate_event = map_candidates_to_events(
            np.asarray(light["file_id"]), np.asarray(light["entry"]),
            arrays["file_id"], arrays["entry"],
        )
    elif set(TROTA_FALLBACK_BRANCHES) <= tree_fields:
        arrays_ak = trota_tree.arrays(TROTA_FALLBACK_BRANCHES, library="ak")
        arrays = {name: np.asarray(ak.to_numpy(arrays_ak[name])) for name in TROTA_FALLBACK_BRANCHES}
        candidate_event = map_candidates_to_events_rle(
            np.asarray(light["run"]), np.asarray(light["luminosityBlock"]),
            np.asarray(light["event"]), arrays["run"],
            arrays["luminosityBlock"], arrays["event"],
        )
        identity_fallback = 1
    else:
        missing = sorted(set(TROTA_PRIMARY_BRANCHES) - tree_fields)
        raise RuntimeError(f"missing TROTA branches: {missing}")

    run2_fiducial = (
        eligible[candidate_event]
        & np.isfinite(arrays["TopResolved1pct_eta"])
        & np.isfinite(arrays["TopResolved1pct_mass"])
        & np.isfinite(arrays["TopResolved1pct_QCDDiscriminant"])
        & (np.abs(arrays["TopResolved1pct_eta"]) < 2.0)
        & (arrays["TopResolved1pct_mass"] >= 100.0)
        & (arrays["TopResolved1pct_mass"] <= 250.0)
    )
    selected_rows = np.flatnonzero(run2_fiducial)
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
                jet_source_indices=ak.to_list(light["jet_source_index_all"][event_index]),
                jet_eta=ak.to_list(light["jet_eta_all"][event_index]),
                jet_phi=ak.to_list(light["jet_phi_all"][event_index]),
                fatjet_eta=ak.to_list(light["fatjet_eta_all"][event_index]),
                fatjet_phi=ak.to_list(light["fatjet_phi_all"][event_index]),
                fatjet_subjet_index1=ak.to_list(light["fatjet_subjet_index1_all"][event_index]),
                fatjet_subjet_index2=ak.to_list(light["fatjet_subjet_index2_all"][event_index]),
                fatjet_top_pass=ak.to_list(light["fatjet_boosted_top_pass_all"][event_index]),
                fatjet_w_pass=ak.to_list(light["fatjet_boosted_w_pass_all"][event_index]),
                subjet_eta=ak.to_list(light["subjet_eta_all"][event_index]),
                subjet_phi=ak.to_list(light["subjet_phi_all"][event_index]),
            )
            result = select_exclusive_resolved_candidates(
                candidate_indices=arrays["TopResolved1pct_candidateIndex"][rows],
                candidate_scores=arrays["TopResolved1pct_QCDDiscriminant"][rows],
                candidate_source_jets=np.stack(
                    [
                        arrays["TopResolved1pct_sourceJetIdx0"][rows],
                        arrays["TopResolved1pct_sourceJetIdx1"][rows],
                        arrays["TopResolved1pct_sourceJetIdx2"][rows],
                    ],
                    axis=1,
                ),
                boosted_vetoed_ak4_indices=vetoed,
            )
            counts[event_index] = result.nres
            rejected_boosted += len(result.rejected_by_boosted_overlap)
            rejected_resolved += len(result.rejected_by_resolved_overlap)
    return counts, {
        "trota_rows": int(trota_tree.num_entries),
        "run2_fiducial_rows": int(selected_rows.size),
        "rejected_by_boosted_overlap": int(rejected_boosted),
        "rejected_by_resolved_overlap": int(rejected_resolved),
        "identity_fallback_files": int(identity_fallback),
    }


def fill_three_schemes(
    histograms: dict[str, Any],
    sample: str,
    variation: str,
    baseline55: np.ndarray,
    extended85: np.ndarray,
    extended80: np.ndarray,
    weights: np.ndarray,
) -> None:
    for scheme, indices, bins in (
        (BASELINE_SCHEME, baseline55, 55),
        (EXCLUSIVE85_SCHEME, extended85, 85),
        (TAILMERGED80_SCHEME, extended80, 80),
    ):
        target = (
            histograms.setdefault(scheme, {})
            .setdefault(sample, {})
            .setdefault(variation, base.empty_index_hist(bins))
        )
        base.add_index_hist(target, indices, weights)


def validate_conservation(histograms: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    baseline = histograms.get(BASELINE_SCHEME) or {}
    for sample, variations in baseline.items():
        for variation, record55 in variations.items():
            key = f"{sample}/{variation}"
            values = {}
            for scheme in (BASELINE_SCHEME, EXCLUSIVE85_SCHEME, TAILMERGED80_SCHEME):
                record = (((histograms.get(scheme) or {}).get(sample) or {}).get(variation))
                if record is None:
                    raise RuntimeError(f"missing conservation record {scheme}/{key}")
                values[scheme] = {
                    "sumw": float(math.fsum(float(value) for value in record["sumw"])),
                    "sumw2": float(math.fsum(float(value) for value in record["sumw2"])),
                    "entries": int(sum(int(value) for value in record["entries"])),
                }
            reference = values[BASELINE_SCHEME]
            for scheme in (EXCLUSIVE85_SCHEME, TAILMERGED80_SCHEME):
                candidate = values[scheme]
                if candidate["entries"] != reference["entries"]:
                    raise RuntimeError(f"entry conservation failure for {key}/{scheme}")
                for field in ("sumw", "sumw2"):
                    if not math.isclose(candidate[field], reference[field], rel_tol=2e-12, abs_tol=2e-9):
                        raise RuntimeError(f"{field} conservation failure for {key}/{scheme}")
            checks[key] = values
    return checks


def process_file(
    record: dict[str, str], repo: Path, normalization: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    root_path = Path(record["input_root"])
    source_root = record["source_root"]
    sidecar = validate_sidecar(root_path)
    metadata = base.read_root_metadata(root_path, fallback=normalization)
    histograms: dict[str, Any] = {}
    audit: dict[str, dict[str, int | str]] = {}
    counters: defaultdict[str, int] = defaultdict(int)
    try:
        with uproot.open(root_path, object_cache=None, array_cache=None) as root_file:
            if "Events" not in root_file or "TROTA" not in root_file:
                raise RuntimeError(f"missing Events/TROTA tree in {root_path}")
            event_tree = root_file["Events"]
            present = set(event_tree.keys())
            missing_light = sorted((set(LIGHT_SCALAR_BRANCHES) | set(OVERLAP_BRANCHES)) - present)
            if missing_light:
                raise RuntimeError(f"missing v8 Events branches in {root_path}: {missing_light}")
            light = event_tree.arrays(
                list(LIGHT_SCALAR_BRANCHES) + list(OVERLAP_BRANCHES), library="ak"
            )
            event_count = len(light["entry"])
            counters["events"] = event_count
            if event_count == 0:
                return {
                    "source_root": source_root, "root_sha256": sidecar.get("root_sha256"),
                    "counters": dict(counters), "histograms": {}, "component_audit": {},
                    "wall_time_seconds": time.monotonic() - started,
                }
            years = np.asarray(light["year"], dtype=int)
            if not np.all(years == 2024):
                raise RuntimeError(f"non-2024 event in {root_path}: {np.unique(years)}")

            sr = np.asarray(light["feature_SR"], dtype=bool)
            source60 = base.selected_an17_recoil60_indices(light, event_count, sr)
            baseline55 = map60_indices_to_adopted55(source60)
            mtb = np.asarray(light["lowdm_mtb"], dtype=float)
            population = (
                sr & np.isfinite(mtb) & (mtb >= 175.0) & (baseline55 >= 0)
            )
            eligible = accepted_dataset_mask(metadata, light, population)
            counters["baseline_population"] = int(np.count_nonzero(population))
            counters["eligible_population"] = int(np.count_nonzero(eligible))
            if not np.any(eligible):
                return {
                    "source_root": source_root, "root_sha256": sidecar.get("root_sha256"),
                    "counters": dict(counters), "histograms": {}, "component_audit": {},
                    "wall_time_seconds": time.monotonic() - started,
                }

            nres, nres_stats = event_nres(light, root_file["TROTA"], eligible)
            counters.update(nres_stats)
            counters["nres_positive_events"] = int(np.count_nonzero(eligible & (nres > 0)))
            recoil = recoil_indices(np.asarray(light["met"], dtype=float))
            extended85 = exclusive85_indices(
                baseline55, recoil, np.asarray(light["nboosted_top"], dtype=int),
                np.asarray(light["nboosted_w"], dtype=int), nres,
            )
            extended80 = map85_indices_to_tailmerged80(extended85)

            dataset_ids = np.asarray(light["dataset_id"], dtype=np.int64)
            is_data_values = np.asarray(light["is_data"], dtype=bool)
            for dataset_id in np.unique(dataset_ids[eligible & is_data_values]):
                selected = eligible & is_data_values & (dataset_ids == dataset_id)
                _dataset, process, is_data, _is_signal = base.dataset_label(metadata, int(dataset_id))
                if not is_data or not base.data_process_allowed(process, "SR"):
                    continue
                unit = np.ones(int(np.count_nonzero(selected)), dtype=float)
                fill_three_schemes(
                    histograms, "data_obs", "nominal", baseline55[selected],
                    extended85[selected], extended80[selected], unit,
                )
                counters["weighted_events"] += int(unit.size)
                counters["weighted_groups"] += 1

            mc_preselection = (
                sr & np.isfinite(mtb) & (mtb >= 175.0)
                & (np.asarray(light["met"], dtype=float) >= 250.0)
                & (np.asarray(light["met"], dtype=float) < 1500.0)
                & ~is_data_values
            )
            mc_eligible = eligible[mc_preselection]
            if np.any(mc_eligible):
                requested = sorted(
                    set(base.WEIGHT_BRANCHES) | DIRECT_WEIGHT_MASK_BRANCHES
                    | {"met", "feature_SR", "lowdm_mtb", "nb_medium", "nboosted_top",
                       "nboosted_w", "nboosted_total"}
                )
                missing = sorted(
                    branch for branch in requested
                    if branch not in present and branch not in base.OPTIONAL_FORWARD_SCHEMA_BRANCHES
                )
                if missing:
                    raise RuntimeError(f"missing weight branches in {root_path}: {missing}")
                branches = [branch for branch in requested if branch in present]
                events = event_tree.arrays(branches, cut=MC_PRESELECTION_CUT, library="ak")
                expected_entries = np.asarray(light["entry"])[mc_preselection]
                observed_entries = np.asarray(events["entry"])
                if not np.array_equal(observed_entries, expected_entries):
                    raise RuntimeError(
                        f"MC preselection identity mismatch: {len(observed_entries)} != {len(expected_entries)}"
                    )
                mc_dataset_ids = dataset_ids[mc_preselection]
                mc_baseline55 = baseline55[mc_preselection]
                mc_extended85 = extended85[mc_preselection]
                mc_extended80 = extended80[mc_preselection]
                for dataset_id in np.unique(mc_dataset_ids[mc_eligible]):
                    dataset_mask = mc_eligible & (mc_dataset_ids == dataset_id)
                    dataset, process, is_data, is_signal = base.dataset_label(metadata, int(dataset_id))
                    if is_data:
                        raise RuntimeError("data passed the MC preselection")
                    if is_signal:
                        mstops = np.asarray(events["mStop"], dtype=int)
                        mlsps = np.asarray(events["mLSP"], dtype=int)
                        point_masks = [
                            dataset_mask & (mstops == mstop) & (mlsps == mlsp)
                            for mstop, mlsp in sorted(set(zip(mstops[dataset_mask], mlsps[dataset_mask])))
                        ]
                    else:
                        point_masks = [dataset_mask]
                    for selected in point_masks:
                        if not np.any(selected):
                            continue
                        chunk = {name: events[name][selected] for name in ak.fields(events)}
                        label = base.sample_label(process, False, is_signal, chunk, dataset)
                        if is_signal and label not in SIGNAL_LABELS:
                            continue
                        arrays, inputs = base.flat_arrays_for_weights(chunk)
                        correction_dataset = dataset
                        if is_signal:
                            correction_dataset = base.signal_btag_efficiency_dataset(
                                int(np.asarray(chunk["mStop"], dtype=int)[0]), dataset,
                            )[1]
                        norm = base.norm_vector(
                            normalization, chunk, int(dataset_id), dataset, False,
                            is_signal, require_normalization=True,
                        )
                        _gen, variations, status = base.compute_weight_bundle(
                            arrays, repo, correction_dataset, process, "2024", inputs["n"],
                            inputs["jet_pt"], inputs["jet_eta"], inputs["jet_hadflav"], inputs["b_med"],
                            inputs["e_eta"], inputs["e_delta_eta_sc"], inputs["e_pt"], inputs["e_phi"],
                            inputs["e_veto"], inputs["e_med"], inputs["n_e_veto"], inputs["n_e_med"],
                            inputs["m_eta"], inputs["m_pt"], inputs["m_phi"], inputs["m_loose"],
                            inputs["m_med"], inputs["n_m_loose"], inputs["n_m_med"],
                            inputs["p_eta"], inputs["p_pt"], inputs["p_phi"], inputs["p_med"],
                            inputs["gcr_mask"], p_r9=inputs["p_r9"], met_pt=inputs["met_pt"],
                            met_trigger_mask=inputs["met_trigger_mask"],
                            analysis_sf_components=ANALYSIS_SF_COMPONENTS,
                        )
                        for component in REQUIRED_COMPONENTS:
                            state = (status.get("components") or {}).get(component) or {}
                            if not state.get("applied"):
                                raise RuntimeError(
                                    f"required component {component} unavailable for {dataset}: {state}"
                                )
                        for component, state in component_status(status, inputs["n"]).items():
                            merged = audit.setdefault(
                                component,
                                {"applied_events": 0, "failed_events": 0, "source": state["source"]},
                            )
                            merged["applied_events"] = int(merged["applied_events"]) + int(state["applied_events"])
                            merged["failed_events"] = int(merged["failed_events"]) + int(state["failed_events"])
                        selected55 = mc_baseline55[selected]
                        selected85 = mc_extended85[selected]
                        selected80 = mc_extended80[selected]
                        for variation, raw_weight in variations.items():
                            weights = np.asarray(raw_weight, dtype=float) * norm
                            if len(weights) != inputs["n"] or not np.all(np.isfinite(weights)):
                                raise RuntimeError(f"invalid {variation} weights in {root_path}")
                            if np.any(np.abs(weights) > base.MAX_ABS_HIST_WEIGHT):
                                raise RuntimeError(f"excessive {variation} weight in {root_path}")
                            fill_three_schemes(
                                histograms, label, variation, selected55, selected85,
                                selected80, weights,
                            )
                        counters["weighted_events"] += inputs["n"]
                        counters["weighted_groups"] += 1

        conservation = validate_conservation(histograms)
        return {
            "source_root": source_root,
            "root_sha256": sidecar.get("root_sha256"),
            "counters": dict(counters),
            "histograms": histograms,
            "component_audit": audit,
            "conservation": conservation,
            "wall_time_seconds": time.monotonic() - started,
        }
    finally:
        drop_file_cache(root_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs = read_input_list(args.input_list)
    normalization = json.loads(args.normalization.read_text())
    if normalization.get("status") != "complete":
        raise RuntimeError("normalization manifest is not complete")
    physics_source_sha256 = {
        relative: sha256(args.repo / relative)
        for relative in EXPECTED_PHYSICS_SOURCE_SHA256
    }
    if physics_source_sha256 != EXPECTED_PHYSICS_SOURCE_SHA256:
        raise RuntimeError(
            f"physics source hash drift: expected {EXPECTED_PHYSICS_SOURCE_SHA256}, "
            f"got {physics_source_sha256}"
        )
    btag_path = args.repo / "analysis/hists/btageff2024.merged"
    btag_sha256 = sha256(btag_path)
    if btag_sha256 != EXPECTED_BTAG_SHA256:
        raise RuntimeError(f"btag checksum drift: {btag_sha256}")

    started = time.monotonic()
    histograms: dict[str, Any] = {}
    totals: defaultdict[str, int] = defaultdict(int)
    audit: dict[str, dict[str, int | str]] = {}
    completed_inputs = []
    root_sha256 = {}
    for record in inputs:
        result = process_file(record, args.repo, normalization)
        completed_inputs.append(result["source_root"])
        root_sha256[result["source_root"]] = result["root_sha256"]
        for key, value in result["counters"].items():
            totals[key] += int(value)
        merge_histograms(histograms, result["histograms"])
        for component, state in result["component_audit"].items():
            merged = audit.setdefault(
                component,
                {"applied_events": 0, "failed_events": 0, "source": state["source"]},
            )
            merged["applied_events"] = int(merged["applied_events"]) + int(state["applied_events"])
            merged["failed_events"] = int(merged["failed_events"]) + int(state["failed_events"])

    conservation = validate_conservation(histograms)
    for scheme, samples in histograms.items():
        for sample, variations in samples.items():
            for variation, record in variations.items():
                for field in ("sumw", "sumw2"):
                    if not np.all(np.isfinite(np.asarray(record[field], dtype=float))):
                        raise RuntimeError(f"non-finite {scheme}/{sample}/{variation}/{field}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": now(),
        "input_list_sha256": sha256(args.input_list),
        "normalization_sha256": sha256(args.normalization),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "physics_source_sha256": physics_source_sha256,
        "category_source_sha256": {
            "highdm_resolved_categories.py": sha256(
                args.repo / "autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py"
            ),
            "highdm_nres_binning.py": sha256(
                args.repo / "autonomous_allhad/autonomous_allhad/highdm_nres_binning.py"
            ),
        },
        "btag_efficiency_sha256": btag_sha256,
        "trota_model_sha256": EXPECTED_TROTA_MODEL_SHA256,
        "input_schema": EXPECTED_INPUT_SCHEMA,
        "files_expected": len(inputs),
        "files_completed": len(completed_inputs),
        "completed_input_roots": completed_inputs,
        "input_root_sha256": root_sha256,
        "totals": dict(totals),
        "required_components": list(REQUIRED_COMPONENTS),
        "analysis_sf_components": list(ANALYSIS_SF_COMPONENTS),
        "component_audit": audit,
        "resolved_top_definition": {
            "working_point": "TROTA TopResolved1pct",
            "fiducial": "abs(eta)<2 and 100<=mass<=250 GeV",
            "boosted_overlap": "dR<0.4 to two valid selected AK8 subjets; otherwise dR<0.8 to AK8 axis",
            "resolved_overlap": "descending QCDDiscriminant, candidateIndex tie-break, jet-disjoint greedy selection",
        },
        "schemes": {
            BASELINE_SCHEME: {
                "bins": 55,
                "bin_labels": adopted55_labels(base.selected_an17_recoil60_labels()),
                "selection": "feature_SR and mTb>=175 GeV; Nres inclusive comparison baseline",
            },
            EXCLUSIVE85_SCHEME: {
                "bins": 85,
                "bin_labels": exclusive85_labels(base.selected_an17_recoil60_labels()),
                "selection": "baseline 55 bins require Nres=0; five Nres>0 topologies x six recoil bins",
                "topologies": list(COARSE_NRES_TOPOLOGIES),
                "recoil_labels": list(RECOIL_LABELS),
            },
            TAILMERGED80_SCHEME: {
                "bins": 80,
                "bin_labels": tailmerged80_labels(base.selected_an17_recoil60_labels()),
                "selection": "exclusive85 with 500-800 and 800-1500 merged in each added topology",
                "topologies": list(COARSE_NRES_TOPOLOGIES),
            },
        },
        "histograms": histograms,
        "conservation": conservation,
        "wall_time_seconds": time.monotonic() - started,
    }
    write_json(args.output, payload)
    print(json.dumps({
        "status": "complete",
        "files": len(completed_inputs),
        "events": totals["events"],
        "eligible": totals["eligible_population"],
        "nres_positive": totals["nres_positive_events"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
