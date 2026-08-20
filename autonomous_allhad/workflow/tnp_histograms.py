#!/usr/bin/env python3
"""Build low-pT lepton tag-and-probe pass/fail mass histograms from NanoAOD."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
import uproot

from workflow.reference_trigger_counts import pileup_source, pileup_weight_triplet


FILTERS = [
    "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
]


GOLDEN_JSONS = {
    "2024": Path("analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json"),
    "2025": Path("analysis/data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json"),
}


def read_file_list(path: Path) -> list[str]:
    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("files"), list):
        return [str(item) for item in payload["files"]]
    raise ValueError(f"unsupported file-list schema: {path}")


def _golden_json(path: Path) -> dict[int, list[tuple[int, int]]]:
    payload = json.loads(path.read_text())
    return {int(run): [(int(lo), int(hi)) for lo, hi in ranges] for run, ranges in payload.items()}


def _lumi_mask(runs: np.ndarray, lumis: np.ndarray, golden: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    result = np.zeros(len(runs), dtype=bool)
    for index, (run, lumi) in enumerate(zip(runs, lumis)):
        result[index] = any(lo <= int(lumi) <= hi for lo, hi in golden.get(int(run), []))
    return result


def _delta_phi(left: Any, right: Any) -> Any:
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def _invariant_mass(tag: Any, probe: Any) -> Any:
    px1 = tag.pt * np.cos(tag.phi); py1 = tag.pt * np.sin(tag.phi); pz1 = tag.pt * np.sinh(tag.eta)
    px2 = probe.pt * np.cos(probe.phi); py2 = probe.pt * np.sin(probe.phi); pz2 = probe.pt * np.sinh(probe.eta)
    e1 = np.sqrt(np.maximum(0.0, tag.mass**2 + px1**2 + py1**2 + pz1**2))
    e2 = np.sqrt(np.maximum(0.0, probe.mass**2 + px2**2 + py2**2 + pz2**2))
    mass2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    return np.sqrt(np.maximum(0.0, mass2))


def _empty(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        "pass_sumw": np.zeros(shape, dtype=float),
        "pass_sumw2": np.zeros(shape, dtype=float),
        "fail_sumw": np.zeros(shape, dtype=float),
        "fail_sumw2": np.zeros(shape, dtype=float),
    }


def _fill(
    target: dict[str, np.ndarray],
    abseta: np.ndarray,
    pt: np.ndarray,
    mass: np.ndarray,
    passed: np.ndarray,
    weights: np.ndarray,
    edges: list[np.ndarray],
) -> None:
    samples = np.column_stack([abseta, pt, mass])
    mask = np.asarray(passed, dtype=bool)
    for label, selected in (("pass", mask), ("fail", ~mask)):
        target[f"{label}_sumw"] += np.histogramdd(samples[selected], bins=edges, weights=weights[selected])[0]
        target[f"{label}_sumw2"] += np.histogramdd(samples[selected], bins=edges, weights=weights[selected] ** 2)[0]


def _serialise(target: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    # Flatten probe dimensions while retaining the mass dimension.
    return {
        key: [[float(value) for value in row] for row in values.reshape(-1, values.shape[-1])]
        for key, values in target.items()
    }


def _access_candidates(file_path: str) -> list[str]:
    """Return independent CMS XRootD routes for one logical file name."""

    if "/store/" not in file_path:
        return [file_path]
    lfn = "/store/" + file_path.split("/store/", 1)[1]
    candidates = [
        f"root://eoscms.cern.ch//eos/cms{lfn}",
        file_path,
        f"root://xrootd-cms.infn.it/{lfn}",
        f"root://cmsxrootd.fnal.gov/{lfn}",
    ]
    return list(dict.fromkeys(candidates))


def _merge_counts(target: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> None:
    for key in target:
        target[key] += source[key]


def _branches(
    kind: str,
    trigger_paths: Iterable[str],
    is_data: bool,
    probe_definition: str | None = None,
    tag_trigger_match_required: bool = True,
) -> list[str]:
    prefix = "Electron" if kind == "electron" else "Muon"
    object_fields = ["pt", "eta", "phi", "mass", "charge", "miniPFRelIso_all"]
    if kind == "electron":
        object_fields += ["cutBased", "convVeto", "lostHits"]
    else:
        object_fields += ["looseId", "tightId"]
        if probe_definition == "loose_id_only":
            object_fields += ["isTracker"]
    branches = ["run", "luminosityBlock", "event", *FILTERS, *trigger_paths]
    branches += [f"{prefix}_{field}" for field in object_fields]
    if tag_trigger_match_required:
        branches += ["TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits"]
    if not is_data:
        branches += ["genWeight", "Pileup_nTrueInt"]
    return branches


def _objects(
    arrays: Any,
    kind: str,
    trigger_filter_bits: int | None,
    probe_definition: str | None = None,
    tag_pt_min_gev: float = 5.0,
    probe_pt_min_gev: float = 5.0,
    probe_pt_max_gev: float = 10.0,
    tag_trigger_match_required: bool = True,
    tag_miniiso_max: float | None = 0.1,
) -> tuple[Any, Any, Any]:
    prefix = "Electron" if kind == "electron" else "Muon"
    pt = arrays[f"{prefix}_pt"]
    eta = arrays[f"{prefix}_eta"]
    phi = arrays[f"{prefix}_phi"]
    mass = arrays[f"{prefix}_mass"]
    charge = arrays[f"{prefix}_charge"]
    iso = arrays[f"{prefix}_miniPFRelIso_all"]
    index = ak.local_index(pt, axis=1)
    if tag_trigger_match_required:
        trig_id = 11 if kind == "electron" else 13
        valid_trigobj = abs(arrays["TrigObj_id"]) == trig_id
        if trigger_filter_bits is not None:
            valid_trigobj = valid_trigobj & (
                (arrays["TrigObj_filterBits"] & int(trigger_filter_bits)) != 0
            )
        deta = eta[:, :, None] - arrays["TrigObj_eta"][:, None, :]
        dphi = _delta_phi(phi[:, :, None], arrays["TrigObj_phi"][:, None, :])
        matched = ak.any(
            valid_trigobj[:, None, :] & ((deta * deta + dphi * dphi) < 0.1**2),
            axis=2,
        )
    else:
        matched = ak.ones_like(pt, dtype=bool)
    if kind == "electron":
        fiducial = (abs(eta) < 1.4442) | ((abs(eta) > 1.5660) & (abs(eta) < 2.5))
        denominator = (
            (pt > probe_pt_min_gev)
            & (pt < probe_pt_max_gev)
            & fiducial
            & arrays["Electron_convVeto"]
            & (arrays["Electron_lostHits"] <= 1)
        )
        if probe_definition in (None, "veto_id_plus_miniiso"):
            passing = denominator & (arrays["Electron_cutBased"] >= 1) & (iso < 0.1)
        elif probe_definition == "veto_id_only":
            passing = denominator & (arrays["Electron_cutBased"] >= 1)
        else:
            raise ValueError(f"unsupported electron probe definition: {probe_definition}")
        tag = (pt > tag_pt_min_gev) & fiducial & (arrays["Electron_cutBased"] >= 4) & matched
        if tag_miniiso_max is not None:
            tag = tag & (iso < tag_miniiso_max)
    else:
        if probe_definition in (None, "miniiso_given_loose_id"):
            denominator = (
                (pt > probe_pt_min_gev)
                & (pt < probe_pt_max_gev)
                & (abs(eta) < 2.4)
                & arrays["Muon_looseId"]
            )
            passing = denominator & (iso < 0.2)
        elif probe_definition == "loose_id_only":
            denominator = (
                (pt > probe_pt_min_gev)
                & (pt < probe_pt_max_gev)
                & (abs(eta) < 2.4)
                & arrays["Muon_isTracker"]
            )
            passing = denominator & arrays["Muon_looseId"]
        else:
            raise ValueError(f"unsupported muon probe definition: {probe_definition}")
        tag = (pt > tag_pt_min_gev) & (abs(eta) < 2.4) & arrays["Muon_tightId"] & matched
        if tag_miniiso_max is not None:
            tag = tag & (iso < tag_miniiso_max)
    records = ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass, "charge": charge, "index": index, "passing": passing})
    return records[tag], records[denominator], passing


def _external_reference_pair_mask(arrays: Any, pairs: Any, config: dict[str, Any]) -> Any:
    """Require a third, offline muon distinct from both TnP legs when configured.

    ParkingSingleMuon trigger objects do not retain a path-specific NanoAOD
    quality bit.  For the low-pT muon ID measurement we therefore make the
    parking trigger external to the measured J/psi pair: an additional barrel
    muon above the parking plateau must be present, and neither TnP leg may be
    that muon.  The same offline topology is imposed on the double-J/psi MC,
    while the parking HLT decision itself is data-only.
    """

    reference = config.get("external_reference_muon")
    if not reference or not bool(reference.get("enabled", False)):
        return ak.ones_like(pairs.tag.pt, dtype=bool)
    pt = arrays["Muon_pt"]
    eta = arrays["Muon_eta"]
    candidate = (
        (pt > float(reference["pt_min_gev"]))
        & (abs(eta) < float(reference["abseta_max"]))
    )
    if bool(reference.get("require_tight_id", True)):
        candidate = candidate & arrays["Muon_tightId"]
    miniiso_max = reference.get("miniiso_max")
    if miniiso_max is not None:
        candidate = candidate & (
            arrays["Muon_miniPFRelIso_all"] < float(miniiso_max)
        )
    indices = ak.local_index(pt, axis=1)
    eligible = (
        candidate[:, None, :]
        & (indices[:, None, :] != pairs.tag.index[:, :, None])
        & (indices[:, None, :] != pairs.probe.index[:, :, None])
    )
    return ak.any(eligible, axis=2)


def build_histograms(
    *,
    kind: str,
    data_files: list[str],
    mc_files: list[str],
    config: dict[str, Any],
    repo: Path,
    step_size: int = 100_000,
) -> dict[str, Any]:
    if kind not in {"electron", "muon"}:
        raise ValueError("kind must be electron or muon")
    eta_edges = np.asarray(config["probe_abseta_edges"], dtype=float)
    pt_edges = np.asarray(config["probe_pt_edges_gev"], dtype=float)
    mass_window = config.get("resonance") or config["resonances"]["primary"]
    fit_window = [float(value) for value in mass_window["mass_window_gev"]]
    mass_edges = np.linspace(fit_window[0], fit_window[1], int(config.get("mass_bins", 50)) + 1)
    edges = [eta_edges, pt_edges, mass_edges]
    shape = tuple(len(values) - 1 for values in edges)
    trigger_paths = [str(path) for path in config.get("reference_paths", [])]
    if not trigger_paths:
        raise ValueError("config must define reference_paths")
    filter_bits = config.get("tag_trigger_object_filter_bits")
    tag_trigger_match_required = bool(config.get("tag_trigger_match_required", True))
    apply_reference_trigger_to_mc = bool(config.get("apply_reference_trigger_to_mc", True))
    probe_definition = str(config.get("probe_definition") or "") or None
    tag_pt_min_gev = float(config.get("tag_pt_min_gev", 5.0))
    tag_miniiso_max_value = config.get("tag_miniiso_max", 0.1)
    tag_miniiso_max = (
        None if tag_miniiso_max_value is None else float(tag_miniiso_max_value)
    )
    probe_pt_min_gev = float(pt_edges[0])
    probe_pt_max_gev = float(pt_edges[-1])
    year = str(config.get("year") or "2024")
    if year not in GOLDEN_JSONS:
        raise ValueError(f"unsupported golden-JSON year for TnP measurement: {year}")
    if not math.isfinite(tag_pt_min_gev) or tag_pt_min_gev < 5.0:
        raise ValueError(f"invalid tag_pt_min_gev: {tag_pt_min_gev}")
    if tag_trigger_match_required and filter_bits is None:
        raise ValueError("tag trigger-object matching requires tag_trigger_object_filter_bits")
    golden_json_path = GOLDEN_JSONS[year]
    golden = _golden_json(repo / golden_json_path)
    samples: dict[str, Any] = {}
    processing: dict[str, Any] = {}
    for sample_name, files in (("data", data_files), ("mc", mc_files)):
        is_data = sample_name == "data"
        require_reference_trigger = is_data or apply_reference_trigger_to_mc
        counts_by_variation = {
            variation: _empty(shape)
            for variation in (("nominal", "up", "down") if not is_data else ("nominal",))
        }
        stats = {
            "files_total": len(files),
            "files_processed": 0,
            "files_successful": [],
            "files_failed": [],
            "file_access": [],
            "events_read": 0,
            "pairs_selected": 0,
            "duplicates_removed": 0,
        }
        seen: set[tuple[int, int, int]] = set()
        for file_path in files:
            access_errors = []
            for access_path in _access_candidates(file_path):
                file_counts = {
                    variation: _empty(shape)
                    for variation in (("nominal", "up", "down") if not is_data else ("nominal",))
                }
                file_seen: set[tuple[int, int, int]] = set()
                file_events_read = 0
                file_pairs_selected = 0
                file_duplicates_removed = 0
                try:
                    tree = uproot.open(access_path)["Events"]
                    present = set(tree.keys())
                    available_paths = [path for path in trigger_paths if path in present]
                    if require_reference_trigger and not available_paths:
                        raise RuntimeError("none of the configured reference trigger paths is present")
                    requested = _branches(
                        kind,
                        available_paths if require_reference_trigger else [],
                        is_data,
                        probe_definition,
                        tag_trigger_match_required,
                    )
                    missing = sorted(set(requested) - present)
                    if missing:
                        raise RuntimeError(f"required branches missing: {', '.join(missing)}")
                    for arrays in tree.iterate(requested, step_size=step_size, library="ak"):
                        n = len(arrays["run"])
                        file_events_read += n
                        event_mask = np.ones(n, dtype=bool)
                        if require_reference_trigger:
                            reference = np.zeros(n, dtype=bool)
                            for path in available_paths:
                                reference |= np.asarray(arrays[path], dtype=bool)
                            event_mask &= reference
                        for flag in FILTERS:
                            event_mask &= np.asarray(arrays[flag], dtype=bool)
                        if is_data:
                            runs = np.asarray(arrays["run"])
                            lumis = np.asarray(arrays["luminosityBlock"])
                            events = np.asarray(arrays["event"])
                            event_mask &= _lumi_mask(runs, lumis, golden)
                            for index in np.flatnonzero(event_mask):
                                key = (int(runs[index]), int(lumis[index]), int(events[index]))
                                if key in seen or key in file_seen:
                                    event_mask[index] = False
                                    file_duplicates_removed += 1
                                else:
                                    file_seen.add(key)
                        if not np.any(event_mask):
                            continue
                        selected_arrays = arrays[event_mask]
                        tags, probes, _ = _objects(
                            selected_arrays,
                            kind,
                            filter_bits,
                            probe_definition,
                            tag_pt_min_gev,
                            probe_pt_min_gev,
                            probe_pt_max_gev,
                            tag_trigger_match_required,
                            tag_miniiso_max,
                        )
                        pairs = ak.cartesian({"tag": tags, "probe": probes}, axis=1)
                        pair_mask = (pairs.tag.index != pairs.probe.index) & (pairs.tag.charge * pairs.probe.charge < 0)
                        if kind == "muon":
                            pair_mask = pair_mask & _external_reference_pair_mask(
                                selected_arrays,
                                pairs,
                                config,
                            )
                        elif config.get("external_reference_muon"):
                            raise ValueError(
                                "external_reference_muon is supported only for muon TnP"
                            )
                        pairs = pairs[pair_mask]
                        masses = _invariant_mass(pairs.tag, pairs.probe)
                        in_window = (masses >= fit_window[0]) & (masses <= fit_window[1])
                        pairs = pairs[in_window]
                        masses = masses[in_window]
                        flat_mass = np.asarray(ak.to_numpy(ak.flatten(masses, axis=1)), dtype=float)
                        flat_pt = np.asarray(ak.to_numpy(ak.flatten(pairs.probe.pt, axis=1)), dtype=float)
                        flat_eta = np.abs(np.asarray(ak.to_numpy(ak.flatten(pairs.probe.eta, axis=1)), dtype=float))
                        flat_pass = np.asarray(ak.to_numpy(ak.flatten(pairs.probe.passing, axis=1)), dtype=bool)
                        if is_data:
                            event_weights = {"nominal": np.ones(int(np.sum(event_mask)), dtype=float)}
                        else:
                            gen_weight = np.asarray(selected_arrays["genWeight"], dtype=float)
                            pileup = pileup_weight_triplet(
                                repo,
                                np.asarray(selected_arrays["Pileup_nTrueInt"], dtype=float),
                                year=year,
                            )
                            event_weights = {
                                variation: gen_weight * pu_weight
                                for variation, pu_weight in zip(("nominal", "up", "down"), pileup)
                            }
                        for variation, values in event_weights.items():
                            _, pair_weights = ak.broadcast_arrays(masses, ak.Array(values))
                            flat_weight = np.asarray(ak.to_numpy(ak.flatten(pair_weights, axis=1)), dtype=float)
                            _fill(file_counts[variation], flat_eta, flat_pt, flat_mass, flat_pass, flat_weight, edges)
                        file_pairs_selected += len(flat_mass)
                    for variation in file_counts:
                        _merge_counts(counts_by_variation[variation], file_counts[variation])
                    seen.update(file_seen)
                    stats["events_read"] += file_events_read
                    stats["pairs_selected"] += file_pairs_selected
                    stats["duplicates_removed"] += file_duplicates_removed
                    stats["files_processed"] += 1
                    stats["files_successful"].append(file_path)
                    stats["file_access"].append({"path": file_path, "access_path": access_path})
                    break
                except Exception as exc:
                    access_errors.append({
                        "access_path": access_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            else:
                stats["files_failed"].append({
                    "path": file_path,
                    "error": access_errors[-1]["error"] if access_errors else "no access candidates",
                    "access_attempts": access_errors,
                })
        samples[sample_name] = _serialise(counts_by_variation["nominal"])
        if not is_data:
            samples["mc_pileup_up"] = _serialise(counts_by_variation["up"])
            samples["mc_pileup_down"] = _serialise(counts_by_variation["down"])
        processing[sample_name] = stats
    return {
        "schema_version": 1,
        "measurement": config["measurement"],
        "year": year,
        "status": "candidate_histograms",
        "kind": kind,
        "probe_abseta_edges": eta_edges.tolist(),
        "probe_pt_edges_gev": pt_edges.tolist(),
        "mass_edges_gev": mass_edges.tolist(),
        "fit_window_gev": fit_window,
        "samples": samples,
        "processing": processing,
        "trigger_object_filter_bits": filter_bits,
        "tag_trigger_match_required": tag_trigger_match_required,
        "reference_trigger_object_kind": config.get("reference_trigger_object_kind", kind),
        "reference_trigger_application": {
            "data": True,
            "mc": apply_reference_trigger_to_mc,
        },
        "tag_pt_min_gev": tag_pt_min_gev,
        "tag_miniiso_max": tag_miniiso_max,
        "tag_selection": config.get("tag"),
        "external_reference_muon": config.get("external_reference_muon"),
        "probe_definition": probe_definition,
        "denominator_selection": config.get("denominator"),
        "target_selection": config.get("target_selection"),
        "golden_json": str(golden_json_path),
        "pileup_correction": pileup_source(year),
        "adoption_blockers": (
            ["tag trigger-object filterBits unresolved"]
            if tag_trigger_match_required and filter_bits is None
            else []
        ),
        "created_unix": time.time(),
    }
