#!/usr/bin/env python3
"""Run independent-fold lost-lepton closure on the trusted 2024 flat tables.

The input ROOT files are the additive event tables produced by
``real_subset_worker.py``.  This program never edits nominal inputs.  It uses
the same normalization and weight implementation as
``build_flat_boosted_recoil_hists.py`` and writes resumable per-ROOT partials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flat_boosted_recoil_hists as flat  # noqa: E402


TARGET_PROCESSES = ("TT", "WtoLNu", "ST")
BACKGROUND_PROCESSES = (
    "TT",
    "WtoLNu",
    "ST",
    "Zto2Nu",
    "DY",
    "GJ",
    "VV",
    "QCD",
)
DATA_PROCESS = "JetMET"
SCHEMA_VERSION = "lost_lepton_closure_2024_v1"
FOLD_HASH = "splitmix64(physical_dataset_id,run,lumi,event); data omits dataset id"
MAX_ABS_WEIGHT = 1.0e12

HIGHDM_MET_BOUNDARIES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0],
    dtype=float,
)
HIGHDM_MET_LABELS = (
    "250-300",
    "300-350",
    "350-400",
    "400-500",
    "500-800",
    "800-1500",
    "1500plus",
)
LOWMET_BOUNDARIES = np.asarray([250.0, 275.0], dtype=float)
LOWMET_LABELS = ("250-275", "275-300")
LOWISR_BOUNDARIES = np.asarray([200.0, 225.0, 250.0, 275.0], dtype=float)
LOWISR_LABELS = ("200-225", "225-250", "250-275", "275-300")
SIGNIFICANCE_BOUNDARIES = np.asarray([7.0, 8.0, 9.0], dtype=float)
SIGNIFICANCE_LABELS = ("7-8", "8-9", "9-10")

SCHEMES: dict[str, dict[str, Any]] = {
    "highdm_met": {
        "bins": len(HIGHDM_MET_LABELS),
        "labels": list(HIGHDM_MET_LABELS),
        "variable": "met",
    },
    "highdm_search60": {
        "bins": len(flat.selected_an17_recoil60_labels()),
        "labels": flat.selected_an17_recoil60_labels(),
        "variable": "search_bin",
    },
    "lowdm_search42": {
        "bins": len(flat.LOWDM_42BIN_LABELS),
        "labels": list(flat.LOWDM_42BIN_LABELS),
        "variable": "search_bin",
    },
}

VR_SPECS: dict[str, dict[str, Any]] = {
    "highdm_nb0": {
        "bins": len(HIGHDM_MET_LABELS),
        "labels": list(HIGHDM_MET_LABELS),
        "variable": "met",
    },
    "highdm_njet3to4_nb1plus": {
        "bins": len(HIGHDM_MET_LABELS),
        "labels": list(HIGHDM_MET_LABELS),
        "variable": "met",
    },
    "lowdm_met250to300": {
        "bins": len(LOWMET_LABELS),
        "labels": list(LOWMET_LABELS),
        "variable": "met",
    },
    "lowdm_isr200to300": {
        "bins": len(LOWISR_LABELS),
        "labels": list(LOWISR_LABELS),
        "variable": "lowdm_isr_pt",
    },
    "lowdm_significance7to10": {
        "bins": len(SIGNIFICANCE_LABELS),
        "labels": list(SIGNIFICANCE_LABELS),
        "variable": "lowdm_met_sqrt_ht",
    },
}

LIGHT_BRANCHES = [
    "run",
    "luminosityBlock",
    "event",
    "dataset_id",
    "physical_dataset_id",
    "is_data",
    "is_background",
    "is_signal",
    "feature_GCR",
    "feature_LLCR",
    "feature_SR",
    "feature_lowdm_LLCR",
    "feature_lowdm_SR",
    "lowdm_search_bin_LLCR",
    "lowdm_search_bin_SR",
    "met",
    "ht",
    "njet",
    "nb_medium",
    "nb_medium_lowdm",
    "nboosted_top",
    "nboosted_w",
    "nboosted_total",
    "lowdm_isr_pt",
    "lowdm_met_sqrt_ht",
    "pass_base_common",
    "pass_signal_trigger",
    "pass_zero_tau",
    "pass_no_veto_leptons",
    "pass_one_veto_lepton",
    "pass_mt_100",
    "pass_met_250",
    "pass_ht_300",
    "pass_open_high",
    "pass_open_pre",
    "pass_lowdm_topology_veto",
    "pass_lowdm_isr",
    "pass_lowdm_met_sqrt_ht",
]
READ_BRANCHES = sorted(set(LIGHT_BRANCHES + flat.WEIGHT_BRANCHES))

_WORKER_CONFIG: dict[str, Any] = {}


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_task_id(root: str) -> str:
    return hashlib.sha256(root.encode()).hexdigest()[:20]


def process_from_name(name: str) -> str:
    if name.startswith("JetMET"):
        return DATA_PROCESS
    match = re.match(r"mc_([^_]+)_", name)
    if not match:
        return "unknown"
    return match.group(1)


def empty_hist(nbin: int) -> dict[str, Any]:
    return {
        "sumw": [0.0] * nbin,
        "sumw2": [0.0] * nbin,
        "entries": [0] * nbin,
    }


def add_hist(
    hist: dict[str, Any],
    indices: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> None:
    idx = np.asarray(indices, dtype=np.int64)
    w = np.asarray(weights, dtype=float)
    selected = (
        np.asarray(mask, dtype=bool)
        & (idx >= 0)
        & (idx < len(hist["sumw"]))
        & np.isfinite(w)
        & (np.abs(w) <= MAX_ABS_WEIGHT)
    )
    if not np.any(selected):
        return
    nbin = len(hist["sumw"])
    sumw = np.bincount(idx[selected], weights=w[selected], minlength=nbin)[:nbin]
    sumw2 = np.bincount(
        idx[selected], weights=np.square(w[selected]), minlength=nbin
    )[:nbin]
    entries = np.bincount(idx[selected], minlength=nbin)[:nbin]
    hist["sumw"] = (np.asarray(hist["sumw"], dtype=float) + sumw).tolist()
    hist["sumw2"] = (np.asarray(hist["sumw2"], dtype=float) + sumw2).tolist()
    hist["entries"] = (
        np.asarray(hist["entries"], dtype=np.int64) + entries
    ).astype(int).tolist()


def merge_hist(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field, dtype in (("sumw", float), ("sumw2", float), ("entries", np.int64)):
        target[field] = (
            np.asarray(target[field], dtype=dtype)
            + np.asarray(source[field], dtype=dtype)
        ).tolist()


def splitmix64(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def event_hash(chunk: dict[str, Any], is_data: bool) -> np.ndarray:
    run = np.asarray(chunk["run"], dtype=np.uint64)
    lumi = np.asarray(chunk["luminosityBlock"], dtype=np.uint64)
    event = np.asarray(chunk["event"], dtype=np.uint64)
    if is_data:
        seed = np.zeros(len(run), dtype=np.uint64)
    else:
        seed = np.asarray(chunk["physical_dataset_id"], dtype=np.uint64)
    mixed = splitmix64(seed ^ np.uint64(0x243F6A8885A308D3))
    mixed ^= splitmix64(run ^ np.uint64(0x13198A2E03707344))
    mixed ^= splitmix64(lumi ^ np.uint64(0xA4093822299F31D0))
    mixed ^= splitmix64(event ^ np.uint64(0x082EFA98EC4E6C89))
    return splitmix64(mixed)


def categorical_indices(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(boundaries, np.asarray(values, dtype=float), side="right") - 1


def closure_masks(chunk: dict[str, Any]) -> dict[str, np.ndarray]:
    n = len(chunk["event"])
    high_cr = np.asarray(chunk["feature_LLCR"], dtype=bool)
    high_sr = np.asarray(chunk["feature_SR"], dtype=bool)
    low_cr = np.asarray(chunk["feature_lowdm_LLCR"], dtype=bool)
    low_sr = np.asarray(chunk["feature_lowdm_SR"], dtype=bool)
    return {
        "high_cr": high_cr,
        "high_sr": high_sr,
        "low_cr": low_cr,
        "low_sr": low_sr,
        "selected": high_cr | high_sr | low_cr | low_sr,
        "all": np.ones(n, dtype=bool),
    }


def vr_masks(chunk: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    n = len(chunk["event"])
    base = (
        np.asarray(chunk["pass_base_common"], dtype=bool)
        & np.asarray(chunk["pass_signal_trigger"], dtype=bool)
        & np.asarray(chunk["pass_zero_tau"], dtype=bool)
        & np.asarray(chunk["pass_ht_300"], dtype=bool)
    )
    nolep = np.asarray(chunk["pass_no_veto_leptons"], dtype=bool)
    onelep = (
        np.asarray(chunk["pass_one_veto_lepton"], dtype=bool)
        & np.asarray(chunk["pass_mt_100"], dtype=bool)
    )
    met = np.asarray(chunk["met"], dtype=float)
    njet = np.asarray(chunk["njet"], dtype=int)
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    open_high = np.asarray(chunk["pass_open_high"], dtype=bool)
    open_pre = np.asarray(chunk["pass_open_pre"], dtype=bool)
    low_topology = np.asarray(chunk["pass_lowdm_topology_veto"], dtype=bool)
    low_isr = np.asarray(chunk["pass_lowdm_isr"], dtype=bool)
    low_significance = np.asarray(chunk["lowdm_met_sqrt_ht"], dtype=float)
    low_isr_pt = np.asarray(chunk["lowdm_isr_pt"], dtype=float)
    high_common = base & (met > 250.0) & open_high
    low_common = base & (njet >= 2) & open_pre & low_topology & low_isr
    definitions = {
        "highdm_nb0": high_common & (njet >= 5) & (nb == 0),
        "highdm_njet3to4_nb1plus": (
            high_common & (njet >= 3) & (njet <= 4) & (nb >= 1)
        ),
        "lowdm_met250to300": (
            low_common
            & (met > 250.0)
            & (met < 300.0)
            & (low_significance >= 10.0)
        ),
        "lowdm_isr200to300": (
            low_common
            & (met > 250.0)
            & (low_significance >= 10.0)
            & (low_isr_pt >= 200.0)
            & (low_isr_pt < 300.0)
        ),
        "lowdm_significance7to10": (
            low_common
            & (met > 250.0)
            & (low_significance >= 7.0)
            & (low_significance < 10.0)
        ),
    }
    return {
        name: {"control": mask & onelep, "target": mask & nolep}
        for name, mask in definitions.items()
    }


def scheme_indices(
    chunk: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    n = len(chunk["event"])
    met_indices = categorical_indices(
        np.asarray(chunk["met"], dtype=float), HIGHDM_MET_BOUNDARIES
    )
    high_cr_60 = flat.selected_an17_recoil60_indices(
        chunk, n, masks["high_cr"]
    )
    high_sr_60 = flat.selected_an17_recoil60_indices(
        chunk, n, masks["high_sr"]
    )
    return {
        "highdm_met": {"control": met_indices, "target": met_indices},
        "highdm_search60": {"control": high_cr_60, "target": high_sr_60},
        "lowdm_search42": {
            "control": np.asarray(chunk["lowdm_search_bin_LLCR"], dtype=int),
            "target": np.asarray(chunk["lowdm_search_bin_SR"], dtype=int),
        },
    }


def vr_indices(chunk: dict[str, Any]) -> dict[str, np.ndarray]:
    met = np.asarray(chunk["met"], dtype=float)
    return {
        "highdm_nb0": categorical_indices(met, HIGHDM_MET_BOUNDARIES),
        "highdm_njet3to4_nb1plus": categorical_indices(
            met, HIGHDM_MET_BOUNDARIES
        ),
        "lowdm_met250to300": categorical_indices(met, LOWMET_BOUNDARIES),
        "lowdm_isr200to300": categorical_indices(
            np.asarray(chunk["lowdm_isr_pt"], dtype=float),
            LOWISR_BOUNDARIES,
        ),
        "lowdm_significance7to10": categorical_indices(
            np.asarray(chunk["lowdm_met_sqrt_ht"], dtype=float),
            SIGNIFICANCE_BOUNDARIES,
        ),
    }


def init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "1"


def weight_for_dataset(
    chunk: dict[str, Any],
    meta: dict[str, Any],
    norm: dict[str, Any],
    repo: Path,
    dataset_id: int,
) -> tuple[np.ndarray, dict[str, Any], str]:
    dataset, process, is_data, is_signal = flat.dataset_label(meta, dataset_id)
    n = len(chunk["event"])
    if is_signal:
        raise RuntimeError("signal record passed to lost-lepton background worker")
    if is_data:
        return np.ones(n, dtype=float), {"data": True}, dataset
    arrays, inputs = flat.flat_arrays_for_weights(chunk)
    normv = flat.norm_vector(
        norm,
        chunk,
        dataset_id,
        dataset,
        is_data=False,
        is_signal=False,
        require_normalization=True,
    )
    _gen, variations, status = flat.compute_weight_bundle(
        arrays,
        repo,
        dataset,
        process,
        "2024",
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
        met_pt=inputs["met_pt"],
        met_trigger_mask=inputs["met_trigger_mask"],
    )
    if "nominal" not in variations:
        raise RuntimeError(f"nominal weight missing for {dataset}")
    for component in (
        "pileup",
        "btagSF",
        "electron_id",
        "muon_id",
        *flat.REQUIRED_ANALYSIS_SF_COMPONENTS,
    ):
        component_status = (status.get("components") or {}).get(component) or {}
        if not component_status.get("applied"):
            raise RuntimeError(
                f"required weight component {component} unavailable for "
                f"{dataset}: {component_status}"
            )
    weight = np.asarray(variations["nominal"], dtype=float) * normv
    if np.any(~np.isfinite(weight)):
        raise RuntimeError(f"non-finite nominal weight for {dataset}")
    if np.any(np.abs(weight) > MAX_ABS_WEIGHT):
        raise RuntimeError(f"excessive nominal weight for {dataset}")
    return weight, status, dataset


def selected_for_any_output(
    closure: dict[str, np.ndarray],
    validation: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    selected = np.asarray(closure["selected"], dtype=bool).copy()
    for sides in validation.values():
        selected |= sides["control"] | sides["target"]
    return selected


def process_root_task(task: dict[str, Any]) -> dict[str, Any]:
    repo = Path(_WORKER_CONFIG["repo"])
    norm = json_load(Path(_WORKER_CONFIG["normalization"]))
    partial_dir = Path(_WORKER_CONFIG["partial_dir"])
    root_path = Path(task["root"])
    process = str(task["process"])
    partial_path = partial_dir / f"{stable_task_id(str(root_path))}.json"
    uid_path = partial_dir / f"{stable_task_id(str(root_path))}.uids.npy"
    if partial_path.exists() and uid_path.exists():
        prior = json_load(partial_path)
        if (
            prior.get("status") == "complete"
            and prior.get("root") == str(root_path)
            and prior.get("schema_version") == SCHEMA_VERSION
            and prior.get("normalization_sha256")
            == _WORKER_CONFIG["normalization_sha256"]
            and prior.get("code_sha256") == _WORKER_CONFIG["code_sha256"]
        ):
            return {
                "status": "reused",
                "partial": str(partial_path),
                "root": str(root_path),
            }
    sidecar = root_path.with_suffix(".json")
    if not sidecar.exists():
        raise RuntimeError(f"missing sidecar for {root_path}")
    meta = json_load(sidecar)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "root": str(root_path),
        "sidecar": str(sidecar),
        "process": process,
        "normalization_sha256": _WORKER_CONFIG["normalization_sha256"],
        "code_sha256": _WORKER_CONFIG["code_sha256"],
        "events_read": 0,
        "events_selected": 0,
        "datasets": {},
        "histograms": {
            scheme: {
                str(fold): {
                    side: empty_hist(int(spec["bins"]))
                    for side in ("control", "target")
                }
                for fold in (0, 1)
            }
            for scheme, spec in SCHEMES.items()
        },
        "validation_regions": {
            name: {
                side: empty_hist(int(spec["bins"]))
                for side in ("control", "target")
            }
            for name, spec in VR_SPECS.items()
        },
    }
    uid_chunks: list[np.ndarray] = []
    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        present = set(tree.keys())
        missing = [branch for branch in READ_BRANCHES if branch not in present]
        missing = [
            branch
            for branch in missing
            if branch not in flat.OPTIONAL_FORWARD_SCHEMA_BRANCHES
        ]
        if missing:
            raise RuntimeError(
                f"{root_path}: required branches missing: {', '.join(missing)}"
            )
        branches = [branch for branch in READ_BRANCHES if branch in present]
        for chunk in tree.iterate(
            branches,
            step_size=int(_WORKER_CONFIG["step_size"]),
            library="ak",
        ):
            n = len(chunk["event"])
            result["events_read"] += n
            closure = closure_masks(chunk)
            validation = vr_masks(chunk)
            selected = selected_for_any_output(closure, validation)
            if not np.any(selected):
                continue
            selected_chunk = {
                name: chunk[name][selected] for name in ak.fields(chunk)
            }
            closure = closure_masks(selected_chunk)
            validation = vr_masks(selected_chunk)
            selected_n = len(selected_chunk["event"])
            result["events_selected"] += selected_n
            is_data_values = np.asarray(selected_chunk["is_data"], dtype=bool)
            if len(np.unique(is_data_values)) != 1:
                raise RuntimeError(f"mixed data/MC chunk in {root_path}")
            is_data = bool(is_data_values[0])
            uids = event_hash(selected_chunk, is_data=is_data)
            uid_chunks.append(uids)
            folds = np.asarray(uids & np.uint64(1), dtype=np.int8)
            scheme_idx = scheme_indices(selected_chunk, closure)
            validation_idx = vr_indices(selected_chunk)
            dataset_ids = np.asarray(selected_chunk["dataset_id"], dtype=np.int64)
            for dataset_id in sorted(set(int(value) for value in dataset_ids)):
                dmask = dataset_ids == dataset_id
                subgroup = {
                    name: values[dmask]
                    for name, values in selected_chunk.items()
                }
                weights, status, dataset = weight_for_dataset(
                    subgroup, meta, norm, repo, dataset_id
                )
                drec = result["datasets"].setdefault(
                    str(dataset_id),
                    {
                        "dataset": dataset,
                        "process": process,
                        "events": 0,
                        "sumw": 0.0,
                        "sumw2": 0.0,
                        "weight_status": status,
                    },
                )
                drec["events"] += len(weights)
                drec["sumw"] += float(np.sum(weights))
                drec["sumw2"] += float(np.sum(np.square(weights)))
                local_folds = folds[dmask]
                local_closure = {
                    key: value[dmask] for key, value in closure.items()
                }
                local_validation = {
                    name: {
                        side: mask[dmask] for side, mask in sides.items()
                    }
                    for name, sides in validation.items()
                }
                local_scheme_idx = {
                    scheme: {
                        side: idx[dmask] for side, idx in sides.items()
                    }
                    for scheme, sides in scheme_idx.items()
                }
                local_validation_idx = {
                    name: idx[dmask] for name, idx in validation_idx.items()
                }
                for fold in (0, 1):
                    fmask = local_folds == fold
                    for scheme in SCHEMES:
                        region_masks = (
                            ("control", local_closure["high_cr"])
                            if scheme.startswith("highdm")
                            else ("control", local_closure["low_cr"])
                        )
                        target_masks = (
                            ("target", local_closure["high_sr"])
                            if scheme.startswith("highdm")
                            else ("target", local_closure["low_sr"])
                        )
                        for side, region_mask in (region_masks, target_masks):
                            add_hist(
                                result["histograms"][scheme][str(fold)][side],
                                local_scheme_idx[scheme][side],
                                weights,
                                fmask & region_mask,
                            )
                for name in VR_SPECS:
                    for side in ("control", "target"):
                        add_hist(
                            result["validation_regions"][name][side],
                            local_validation_idx[name],
                            weights,
                            local_validation[name][side],
                        )
    result["status"] = "complete"
    result["uid_count"] = int(sum(len(values) for values in uid_chunks))
    uids_out = (
        np.concatenate(uid_chunks).astype(np.uint64, copy=False)
        if uid_chunks
        else np.asarray([], dtype=np.uint64)
    )
    np.save(uid_path, uids_out, allow_pickle=False)
    result["uid_file"] = str(uid_path)
    result["uid_sha256"] = sha256_file(uid_path)
    json_dump(partial_path, result)
    return {
        "status": "complete",
        "partial": str(partial_path),
        "root": str(root_path),
    }


def merge_partials(
    tasks: list[dict[str, Any]],
    partial_dir: Path,
    normalization_sha256: str,
    code_sha256: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "merging",
        "fold_hash": FOLD_HASH,
        "normalization_sha256": normalization_sha256,
        "code_sha256": code_sha256,
        "inputs": [],
        "processes": {},
        "input_totals": {
            "roots": 0,
            "events_read": 0,
            "events_selected": 0,
        },
        "duplicate_audit": {},
    }
    uid_by_process: dict[str, list[np.ndarray]] = {}
    for task in tasks:
        partial_path = partial_dir / f"{stable_task_id(task['root'])}.json"
        partial = json_load(partial_path)
        if partial.get("status") != "complete":
            raise RuntimeError(f"incomplete partial {partial_path}")
        if partial.get("normalization_sha256") != normalization_sha256:
            raise RuntimeError(f"normalization mismatch in {partial_path}")
        if partial.get("code_sha256") != code_sha256:
            raise RuntimeError(f"code mismatch in {partial_path}")
        process = str(partial["process"])
        target = merged["processes"].setdefault(
            process,
            {
                "histograms": {
                    scheme: {
                        str(fold): {
                            side: empty_hist(int(spec["bins"]))
                            for side in ("control", "target")
                        }
                        for fold in (0, 1)
                    }
                    for scheme, spec in SCHEMES.items()
                },
                "validation_regions": {
                    name: {
                        side: empty_hist(int(spec["bins"]))
                        for side in ("control", "target")
                    }
                    for name, spec in VR_SPECS.items()
                },
                "datasets": {},
            },
        )
        for scheme in SCHEMES:
            for fold in (0, 1):
                for side in ("control", "target"):
                    merge_hist(
                        target["histograms"][scheme][str(fold)][side],
                        partial["histograms"][scheme][str(fold)][side],
                    )
        for name in VR_SPECS:
            for side in ("control", "target"):
                merge_hist(
                    target["validation_regions"][name][side],
                    partial["validation_regions"][name][side],
                )
        for dataset_id, record in partial["datasets"].items():
            drec = target["datasets"].setdefault(
                dataset_id,
                {
                    "dataset": record["dataset"],
                    "process": process,
                    "events": 0,
                    "sumw": 0.0,
                    "sumw2": 0.0,
                    "weight_status": record["weight_status"],
                },
            )
            drec["events"] += int(record["events"])
            drec["sumw"] += float(record["sumw"])
            drec["sumw2"] += float(record["sumw2"])
        merged["inputs"].append(
            {
                "root": partial["root"],
                "sidecar": partial["sidecar"],
                "process": process,
                "events_read": int(partial["events_read"]),
                "events_selected": int(partial["events_selected"]),
                "partial": str(partial_path),
                "uid_sha256": partial["uid_sha256"],
            }
        )
        merged["input_totals"]["roots"] += 1
        merged["input_totals"]["events_read"] += int(partial["events_read"])
        merged["input_totals"]["events_selected"] += int(
            partial["events_selected"]
        )
        uid_values = np.load(partial["uid_file"], allow_pickle=False)
        uid_by_process.setdefault(process, []).append(uid_values)
    any_duplicates = False
    for process, parts in sorted(uid_by_process.items()):
        values = (
            np.concatenate(parts)
            if parts
            else np.asarray([], dtype=np.uint64)
        )
        unique = np.unique(values)
        duplicate_count = int(len(values) - len(unique))
        any_duplicates |= duplicate_count > 0
        merged["duplicate_audit"][process] = {
            "selected_occurrences": int(len(values)),
            "unique_event_hashes": int(len(unique)),
            "duplicate_occurrences": duplicate_count,
            "hash_collision_probability_assumed_negligible": True,
        }
    merged["duplicate_audit"]["status"] = (
        "failed_duplicates_found" if any_duplicates else "complete_no_duplicates"
    )
    merged["status"] = (
        "blocked_duplicates_found" if any_duplicates else "complete"
    )
    return merged


def sum_process_hist(
    source: dict[str, Any],
    processes: tuple[str, ...],
    scheme: str,
    fold: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbin = int(SCHEMES[scheme]["bins"])
    sumw = np.zeros(nbin, dtype=float)
    sumw2 = np.zeros(nbin, dtype=float)
    entries = np.zeros(nbin, dtype=np.int64)
    for process in processes:
        record = (
            (((source.get("processes") or {}).get(process) or {})
             .get("histograms") or {})
            .get(scheme, {})
            .get(str(fold), {})
            .get(side)
        )
        if not record:
            continue
        sumw += np.asarray(record["sumw"], dtype=float)
        sumw2 += np.asarray(record["sumw2"], dtype=float)
        entries += np.asarray(record["entries"], dtype=np.int64)
    return sumw, sumw2, entries


def finite_or_none(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in np.asarray(values, dtype=float)
    ]


def calculate_direction(
    train_sr: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_cr: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_sr: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_cr: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    sr_a, var_sr_a, ent_sr_a = train_sr
    cr_a, var_cr_a, ent_cr_a = train_cr
    sr_b, var_sr_b, ent_sr_b = test_sr
    cr_b, var_cr_b, ent_cr_b = test_cr
    valid_tf = (cr_a != 0.0) & np.isfinite(cr_a) & np.isfinite(sr_a)
    tf = np.full(len(sr_a), np.nan, dtype=float)
    tf_var = np.full(len(sr_a), np.nan, dtype=float)
    tf[valid_tf] = sr_a[valid_tf] / cr_a[valid_tf]
    tf_var[valid_tf] = (
        var_sr_a[valid_tf] / np.square(cr_a[valid_tf])
        + np.square(sr_a[valid_tf])
        * var_cr_a[valid_tf]
        / np.power(cr_a[valid_tf], 4)
    )
    prediction = tf * cr_b
    prediction_var = np.square(cr_b) * tf_var + np.square(tf) * var_cr_b
    valid_closure = (
        valid_tf
        & (sr_b != 0.0)
        & np.isfinite(prediction)
        & np.isfinite(sr_b)
    )
    ratio = np.full(len(sr_a), np.nan, dtype=float)
    ratio_var = np.full(len(sr_a), np.nan, dtype=float)
    ratio[valid_closure] = prediction[valid_closure] / sr_b[valid_closure]
    ratio_var[valid_closure] = (
        prediction_var[valid_closure] / np.square(sr_b[valid_closure])
        + np.square(prediction[valid_closure])
        * var_sr_b[valid_closure]
        / np.power(sr_b[valid_closure], 4)
    )
    pull_den = np.sqrt(np.maximum(prediction_var + var_sr_b, 0.0))
    pull = np.divide(
        prediction - sr_b,
        pull_den,
        out=np.full(len(sr_a), np.nan, dtype=float),
        where=pull_den > 0.0,
    )
    neff_train_cr = np.divide(
        np.square(cr_a),
        var_cr_a,
        out=np.zeros(len(sr_a), dtype=float),
        where=var_cr_a > 0.0,
    )
    neff_train_sr = np.divide(
        np.square(sr_a),
        var_sr_a,
        out=np.zeros(len(sr_a), dtype=float),
        where=var_sr_a > 0.0,
    )
    return {
        "transfer_factor": finite_or_none(tf),
        "transfer_factor_variance": finite_or_none(tf_var),
        "prediction": finite_or_none(prediction),
        "prediction_variance": finite_or_none(prediction_var),
        "direct": finite_or_none(sr_b),
        "direct_variance": finite_or_none(var_sr_b),
        "closure_ratio": finite_or_none(ratio),
        "closure_ratio_variance": finite_or_none(ratio_var),
        "pull": finite_or_none(pull),
        "valid_transfer_factor": valid_tf.tolist(),
        "valid_closure": valid_closure.tolist(),
        "train_control_entries": ent_cr_a.astype(int).tolist(),
        "train_target_entries": ent_sr_a.astype(int).tolist(),
        "test_control_entries": ent_cr_b.astype(int).tolist(),
        "test_target_entries": ent_sr_b.astype(int).tolist(),
        "train_control_neff": finite_or_none(neff_train_cr),
        "train_target_neff": finite_or_none(neff_train_sr),
    }


def combine_crossfit(
    ab: dict[str, Any],
    ba: dict[str, Any],
) -> dict[str, Any]:
    pred_ab = np.asarray(
        [np.nan if value is None else value for value in ab["prediction"]],
        dtype=float,
    )
    pred_ba = np.asarray(
        [np.nan if value is None else value for value in ba["prediction"]],
        dtype=float,
    )
    var_ab = np.asarray(
        [
            np.nan if value is None else value
            for value in ab["prediction_variance"]
        ],
        dtype=float,
    )
    var_ba = np.asarray(
        [
            np.nan if value is None else value
            for value in ba["prediction_variance"]
        ],
        dtype=float,
    )
    direct_ab = np.asarray(
        [np.nan if value is None else value for value in ab["direct"]],
        dtype=float,
    )
    direct_ba = np.asarray(
        [np.nan if value is None else value for value in ba["direct"]],
        dtype=float,
    )
    direct_var_ab = np.asarray(
        [
            np.nan if value is None else value
            for value in ab["direct_variance"]
        ],
        dtype=float,
    )
    direct_var_ba = np.asarray(
        [
            np.nan if value is None else value
            for value in ba["direct_variance"]
        ],
        dtype=float,
    )
    valid = (
        np.asarray(ab["valid_closure"], dtype=bool)
        & np.asarray(ba["valid_closure"], dtype=bool)
    )
    prediction = pred_ab + pred_ba
    prediction_var = var_ab + var_ba
    direct = direct_ab + direct_ba
    direct_var = direct_var_ab + direct_var_ba
    ratio = np.divide(
        prediction,
        direct,
        out=np.full(len(prediction), np.nan, dtype=float),
        where=valid & (direct != 0.0),
    )
    ratio_var = np.full(len(prediction), np.nan, dtype=float)
    good = valid & (direct != 0.0)
    ratio_var[good] = (
        prediction_var[good] / np.square(direct[good])
        + np.square(prediction[good])
        * direct_var[good]
        / np.power(direct[good], 4)
    )
    pull_den = np.sqrt(np.maximum(prediction_var + direct_var, 0.0))
    pull = np.divide(
        prediction - direct,
        pull_den,
        out=np.full(len(prediction), np.nan, dtype=float),
        where=good & (pull_den > 0.0),
    )
    chi2 = float(np.sum(np.square(pull[np.isfinite(pull)])))
    ndf = int(np.count_nonzero(np.isfinite(pull)))
    return {
        "prediction": finite_or_none(prediction),
        "prediction_variance": finite_or_none(prediction_var),
        "direct": finite_or_none(direct),
        "direct_variance": finite_or_none(direct_var),
        "closure_ratio": finite_or_none(ratio),
        "closure_ratio_variance": finite_or_none(ratio_var),
        "pull": finite_or_none(pull),
        "valid_closure": good.tolist(),
        "diagonal_chi2": chi2,
        "diagonal_ndf": ndf,
        "maximum_absolute_pull": (
            float(np.nanmax(np.abs(pull))) if np.any(np.isfinite(pull)) else None
        ),
    }


def build_mc_closure(merged: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "target_processes": list(TARGET_PROCESSES),
        "schemes": {},
        "process_diagnostics": {},
    }
    for scheme, spec in SCHEMES.items():
        target = {}
        for train, test, label in ((0, 1, "A_to_B"), (1, 0, "B_to_A")):
            target[label] = calculate_direction(
                sum_process_hist(
                    merged, TARGET_PROCESSES, scheme, train, "target"
                ),
                sum_process_hist(
                    merged, TARGET_PROCESSES, scheme, train, "control"
                ),
                sum_process_hist(
                    merged, TARGET_PROCESSES, scheme, test, "target"
                ),
                sum_process_hist(
                    merged, TARGET_PROCESSES, scheme, test, "control"
                ),
            )
        target["crossfit"] = combine_crossfit(
            target["A_to_B"], target["B_to_A"]
        )
        target["labels"] = list(spec["labels"])
        output["schemes"][scheme] = target
        for process in TARGET_PROCESSES:
            prec = output["process_diagnostics"].setdefault(process, {})
            directions = {}
            for train, test, label in ((0, 1, "A_to_B"), (1, 0, "B_to_A")):
                directions[label] = calculate_direction(
                    sum_process_hist(
                        merged, (process,), scheme, train, "target"
                    ),
                    sum_process_hist(
                        merged, (process,), scheme, train, "control"
                    ),
                    sum_process_hist(
                        merged, (process,), scheme, test, "target"
                    ),
                    sum_process_hist(
                        merged, (process,), scheme, test, "control"
                    ),
                )
            directions["crossfit"] = combine_crossfit(
                directions["A_to_B"], directions["B_to_A"]
            )
            directions["labels"] = list(spec["labels"])
            prec[scheme] = directions
    return output


def parse_manifest(
    manifest: dict[str, Any],
    requested_processes: tuple[str, ...],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in manifest.get("mc_groups") or []:
        process = process_from_name(str(record.get("name") or ""))
        if process not in requested_processes:
            continue
        tasks.append(
            {
                "root": str(record["root"]),
                "sidecar": str(record["sidecar"]),
                "process": process,
                "events_written": int(record.get("events_written") or 0),
            }
        )
    if DATA_PROCESS in requested_processes:
        for record in manifest.get("data_groups") or []:
            process = process_from_name(str(record.get("name") or ""))
            if process != DATA_PROCESS:
                continue
            tasks.append(
                {
                    "root": str(record["root"]),
                    "sidecar": str(record["sidecar"]),
                    "process": process,
                    "events_written": int(record.get("events_written") or 0),
                }
            )
    return tasks


def self_test() -> None:
    chunk = {
        "run": np.asarray([1, 1, 1, 2], dtype=np.int64),
        "luminosityBlock": np.asarray([1, 1, 2, 1], dtype=np.int64),
        "event": np.asarray([1, 2, 1, 1], dtype=np.int64),
        "physical_dataset_id": np.asarray([10, 10, 10, 10], dtype=np.int64),
    }
    first = event_hash(chunk, is_data=False)
    second = event_hash(chunk, is_data=False)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    hist = empty_hist(3)
    add_hist(
        hist,
        np.asarray([0, 1, 2]),
        np.asarray([1.0, -2.0, 3.0]),
        np.asarray([True, True, False]),
    )
    assert hist["sumw"] == [1.0, -2.0, 0.0]
    assert hist["sumw2"] == [1.0, 4.0, 0.0]
    assert hist["entries"] == [1, 1, 0]
    train_sr = (
        np.asarray([10.0]),
        np.asarray([10.0]),
        np.asarray([10]),
    )
    train_cr = (
        np.asarray([20.0]),
        np.asarray([20.0]),
        np.asarray([20]),
    )
    test_sr = (
        np.asarray([12.0]),
        np.asarray([12.0]),
        np.asarray([12]),
    )
    test_cr = (
        np.asarray([24.0]),
        np.asarray([24.0]),
        np.asarray([24]),
    )
    direction = calculate_direction(train_sr, train_cr, test_sr, test_cr)
    assert abs(float(direction["transfer_factor"][0]) - 0.5) < 1e-12
    assert abs(float(direction["closure_ratio"][0]) - 1.0) < 1e-12
    print("self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processes", default="TT,WtoLNu,ST")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=100_000)
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for required in (
        "repo",
        "manifest",
        "normalization",
        "output_dir",
    ):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    repo = args.repo.resolve()
    manifest_path = args.manifest.resolve()
    normalization_path = args.normalization.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    requested_processes = tuple(
        part.strip() for part in args.processes.split(",") if part.strip()
    )
    unknown = sorted(
        set(requested_processes)
        - set(BACKGROUND_PROCESSES)
        - {DATA_PROCESS}
    )
    if unknown:
        raise SystemExit(f"unknown processes: {', '.join(unknown)}")
    manifest = json_load(manifest_path)
    if not str(manifest.get("status") or "").startswith("complete"):
        raise SystemExit(f"manifest is not complete: {manifest.get('status')}")
    tasks = parse_manifest(manifest, requested_processes)
    if args.max_roots is not None:
        tasks = tasks[: max(0, int(args.max_roots))]
    if not tasks:
        raise SystemExit("no input tasks selected")
    code_sha256 = sha256_file(Path(__file__).resolve())
    normalization_sha256 = sha256_file(normalization_path)
    config = {
        "repo": str(repo),
        "normalization": str(normalization_path),
        "normalization_sha256": normalization_sha256,
        "code_sha256": code_sha256,
        "partial_dir": str(partial_dir),
        "step_size": int(args.step_size),
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_unix": time.time(),
        "repo": str(repo),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "normalization": str(normalization_path),
        "normalization_sha256": normalization_sha256,
        "code_sha256": code_sha256,
        "processes": list(requested_processes),
        "jobs": int(args.jobs),
        "step_size": int(args.step_size),
        "tasks": tasks,
        "nominal_inputs_modified": False,
        "selection_authority": (
            "autonomous_allhad/autonomous_allhad/real_subset_worker.py"
        ),
    }
    json_dump(output_dir / "run_manifest.json", run_manifest)
    init_worker(config)
    completed = 0
    failed: list[dict[str, Any]] = []
    if int(args.jobs) <= 1:
        iterator = map(process_root_task, tasks)
        pool = None
    else:
        pool = mp.Pool(
            processes=int(args.jobs),
            initializer=init_worker,
            initargs=(config,),
        )
        iterator = pool.imap_unordered(process_root_task, tasks)
    try:
        for record in iterator:
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {record['status']} "
                f"{record['root']}",
                flush=True,
            )
    except Exception as exc:
        failed.append({"error": f"{type(exc).__name__}: {exc}"})
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if failed:
        run_manifest["status"] = "failed"
        run_manifest["failures"] = failed
        json_dump(output_dir / "run_manifest.json", run_manifest)
        return 1
    merged = merge_partials(
        tasks, partial_dir, normalization_sha256, code_sha256
    )
    merged["manifest"] = str(manifest_path)
    merged["manifest_sha256"] = sha256_file(manifest_path)
    merged["normalization"] = str(normalization_path)
    merged["nominal_inputs_modified"] = False
    json_dump(output_dir / "merged_histograms.json", merged)
    if all(process in merged["processes"] for process in TARGET_PROCESSES):
        closure = build_mc_closure(merged)
        closure["input_histograms"] = str(
            output_dir / "merged_histograms.json"
        )
        closure["normalization_sha256"] = normalization_sha256
        closure["code_sha256"] = code_sha256
        closure["duplicate_audit_status"] = merged["duplicate_audit"][
            "status"
        ]
        if merged["status"] != "complete":
            closure["status"] = "blocked"
        json_dump(output_dir / "mc_closure.json", closure)
    run_manifest["status"] = merged["status"]
    run_manifest["completed_at_unix"] = time.time()
    run_manifest["roots_completed"] = completed
    json_dump(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps({"status": merged["status"], "output": str(output_dir)}))
    return 0 if merged["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
