#!/usr/bin/env python3
"""Event-level lepton-removal closure on the trusted 2024 flat tables.

The selection authority remains ``real_subset_worker.py``.  This program
reads the additive flat ROOT tables without modifying them, invisibilizes the
selected veto electron or loose muon,

    pTmiss(removal) = pTmiss + pT(lepton),

and reapplies the SR/validation-region kinematics with the new vector.

TT and ST are always merged into one Top component.  WtoLNu is the only
independently moving target component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lost_lepton_closure_2024 as legacy  # noqa: E402

from autonomous_allhad.real_subset_worker import assign_lowdm_search_bin  # noqa: E402


flat = legacy.flat
SCHEMA_VERSION = "lost_lepton_removal_closure_2024_v1"
TARGET_COMPONENTS = ("Top", "W")
ALL_PROCESSES = (
    "TT",
    "ST",
    "WtoLNu",
    "Zto2Nu",
    "DY",
    "GJ",
    "VV",
    "QCD",
    "JetMET",
)
PREDICTION_KINDS = (
    "prediction_post_e",
    "prediction_post_mu",
    "prediction_strict_e",
    "prediction_strict_mu",
)
TARGET_KINDS = ("target_inclusive", "target_leptonic")
HISTOGRAM_KINDS = PREDICTION_KINDS + TARGET_KINDS
MAX_ABS_WEIGHT = 1.0e12

HIGHDM_UT_BOUNDARIES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0],
    dtype=float,
)
HIGHDM_UT_LABELS = (
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
    "highdm_ut": {
        "bins": len(HIGHDM_UT_LABELS),
        "labels": list(HIGHDM_UT_LABELS),
        "regime": "highdm",
        "blinded_nominal": True,
    },
    "highdm_search60": {
        "bins": len(flat.selected_an17_recoil60_labels()),
        "labels": flat.selected_an17_recoil60_labels(),
        "regime": "highdm",
        "blinded_nominal": True,
    },
    "lowdm_search42": {
        "bins": len(flat.LOWDM_42BIN_LABELS),
        "labels": list(flat.LOWDM_42BIN_LABELS),
        "regime": "lowdm",
        "blinded_nominal": True,
    },
    "highdm_vr_nb0": {
        "bins": len(HIGHDM_UT_LABELS),
        "labels": list(HIGHDM_UT_LABELS),
        "regime": "highdm",
        "blinded_nominal": False,
    },
    "highdm_vr_njet3to4_nb1plus": {
        "bins": len(HIGHDM_UT_LABELS),
        "labels": list(HIGHDM_UT_LABELS),
        "regime": "highdm",
        "blinded_nominal": False,
    },
    "lowdm_vr_met250to300": {
        "bins": len(LOWMET_LABELS),
        "labels": list(LOWMET_LABELS),
        "regime": "lowdm",
        "blinded_nominal": False,
    },
    "lowdm_vr_isr200to300": {
        "bins": len(LOWISR_LABELS),
        "labels": list(LOWISR_LABELS),
        "regime": "lowdm",
        "blinded_nominal": False,
    },
    "lowdm_vr_significance7to10": {
        "bins": len(SIGNIFICANCE_LABELS),
        "labels": list(SIGNIFICANCE_LABELS),
        "regime": "lowdm",
        "blinded_nominal": False,
    },
}

REMOVAL_BRANCHES = [
    "run",
    "luminosityBlock",
    "event",
    "dataset_id",
    "physical_dataset_id",
    "is_data",
    "feature_GCR",
    "feature_LLCR",
    "feature_SR",
    "feature_lowdm_LLCR",
    "feature_lowdm_SR",
    "lowdm_search_bin_SR",
    "met",
    "met_phi",
    "ht",
    "njet",
    "nb_medium",
    "nb_medium_lowdm",
    "nboosted_top",
    "nboosted_w",
    "nboosted_total",
    "n_lowdm_isr",
    "lowdm_isr_pt",
    "lowdm_isr_phi",
    "lowdm_ptb",
    "lowdm_mtb",
    "n_sv_softb",
    "n_e_veto",
    "n_m_loose",
    "pass_base_common",
    "pass_signal_trigger",
    "pass_zero_tau",
    "pass_no_veto_leptons",
    "pass_one_veto_lepton",
    "pass_mt_100",
    "pass_ht_300",
    "pass_lowdm_topology_veto",
    "good_jet_phi",
    "electron_veto_pt",
    "electron_veto_phi",
    "muon_loose_pt",
    "muon_loose_phi",
]
READ_BRANCHES = sorted(set(flat.WEIGHT_BRANCHES + REMOVAL_BRANCHES))
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
    hist["sumw"] = (
        np.asarray(hist["sumw"], dtype=float)
        + np.bincount(idx[selected], weights=w[selected], minlength=nbin)[:nbin]
    ).tolist()
    hist["sumw2"] = (
        np.asarray(hist["sumw2"], dtype=float)
        + np.bincount(
            idx[selected], weights=np.square(w[selected]), minlength=nbin
        )[:nbin]
    ).tolist()
    hist["entries"] = (
        np.asarray(hist["entries"], dtype=np.int64)
        + np.bincount(idx[selected], minlength=nbin)[:nbin]
    ).astype(int).tolist()


def merge_hist(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field, dtype in (
        ("sumw", float),
        ("sumw2", float),
        ("entries", np.int64),
    ):
        target[field] = (
            np.asarray(target[field], dtype=dtype)
            + np.asarray(source[field], dtype=dtype)
        ).tolist()


def component_for_process(process: str) -> str:
    if process in {"TT", "ST"}:
        return "Top"
    if process == "WtoLNu":
        return "W"
    if process == "JetMET":
        return "Data"
    return "Other"


def pure_leptonic_dataset(dataset: str, process: str) -> bool:
    """Identify decay-filtered samples with a guaranteed leptonic W decay."""
    if process == "WtoLNu":
        return True
    if process == "TT":
        return dataset.startswith(("TTtoLNu2Q_", "TTto2L2Nu_"))
    if process == "ST":
        return (
            ("LNu" in dataset or "2L2Nu" in dataset)
            and "to2Q" not in dataset
            and "to4Q" not in dataset
        )
    return False


def categorical_indices(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(boundaries, np.asarray(values, dtype=float), side="right") - 1


def first_or_zero(values: Any) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.firsts(values), 0.0)).astype(float)


def first_jet_phis(values: Any, count: int) -> list[np.ndarray]:
    padded = ak.pad_none(values, count, axis=1, clip=True)
    return [
        ak.to_numpy(ak.fill_none(padded[:, index], 0.0)).astype(float)
        for index in range(count)
    ]


def absolute_delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def emulated_met(chunk: Any) -> dict[str, np.ndarray]:
    met = np.asarray(chunk["met"], dtype=float)
    met_phi = np.asarray(chunk["met_phi"], dtype=float)
    n_e = np.asarray(chunk["n_e_veto"], dtype=int)
    n_m = np.asarray(chunk["n_m_loose"], dtype=int)
    electron = (n_e == 1) & (n_m == 0)
    muon = (n_e == 0) & (n_m == 1)
    e_pt = first_or_zero(chunk["electron_veto_pt"])
    e_phi = first_or_zero(chunk["electron_veto_phi"])
    m_pt = first_or_zero(chunk["muon_loose_pt"])
    m_phi = first_or_zero(chunk["muon_loose_phi"])
    lepton_pt = np.where(electron, e_pt, np.where(muon, m_pt, 0.0))
    lepton_phi = np.where(electron, e_phi, np.where(muon, m_phi, 0.0))
    px = met * np.cos(met_phi) + lepton_pt * np.cos(lepton_phi)
    py = met * np.sin(met_phi) + lepton_pt * np.sin(lepton_phi)
    return {
        "pt": np.hypot(px, py),
        "phi": np.arctan2(py, px),
        "electron": electron,
        "muon": muon,
        "valid": electron | muon,
    }


def lowdm_indices(
    chunk: Any,
    recoil: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    out = np.full(len(recoil), -1, dtype=np.int32)
    njet = np.asarray(chunk["njet"], dtype=int)
    nb = np.asarray(chunk["nb_medium_lowdm"], dtype=int)
    nsv = np.asarray(chunk["n_sv_softb"], dtype=int)
    isr = np.asarray(chunk["lowdm_isr_pt"], dtype=float)
    ptb = np.asarray(chunk["lowdm_ptb"], dtype=float)
    mtb = np.asarray(chunk["lowdm_mtb"], dtype=float)
    for index in np.flatnonzero(selected):
        out[index] = assign_lowdm_search_bin(
            int(njet[index]),
            int(nb[index]),
            int(nsv[index]),
            float(isr[index]),
            float(ptb[index]),
            float(recoil[index]),
            float(mtb[index]),
        )
    return out


def build_masks_and_indices(chunk: Any) -> dict[str, dict[str, np.ndarray]]:
    n = len(chunk["event"])
    removal = emulated_met(chunk)
    recoil = removal["pt"]
    recoil_phi = removal["phi"]
    jet_phis = first_jet_phis(chunk["good_jet_phi"], 4)
    dphi = [absolute_delta_phi(phi, recoil_phi) for phi in jet_phis]
    open_high = (dphi[0] > 0.5) & (dphi[1] > 0.5) & (dphi[2] > 0.5) & (dphi[3] > 0.5)
    open_pre = (dphi[0] > 0.5) & (dphi[1] > 0.15) & (dphi[2] > 0.15)

    base = (
        np.asarray(chunk["pass_base_common"], dtype=bool)
        & np.asarray(chunk["pass_signal_trigger"], dtype=bool)
        & np.asarray(chunk["pass_zero_tau"], dtype=bool)
        & np.asarray(chunk["pass_ht_300"], dtype=bool)
    )
    onelep = (
        np.asarray(chunk["pass_one_veto_lepton"], dtype=bool)
        & np.asarray(chunk["pass_mt_100"], dtype=bool)
        & removal["valid"]
    )
    nolep = np.asarray(chunk["pass_no_veto_leptons"], dtype=bool)
    njet = np.asarray(chunk["njet"], dtype=int)
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    ht = np.asarray(chunk["ht"], dtype=float)
    original_met = np.asarray(chunk["met"], dtype=float)

    high_post = (
        base
        & onelep
        & (njet >= 5)
        & (nb >= 1)
        & (recoil > 250.0)
        & (ht > 300.0)
        & open_high
    )
    high_strict = np.asarray(chunk["feature_LLCR"], dtype=bool) & high_post
    high_target = np.asarray(chunk["feature_SR"], dtype=bool)

    low_topology = np.asarray(chunk["pass_lowdm_topology_veto"], dtype=bool)
    n_isr = np.asarray(chunk["n_lowdm_isr"], dtype=int)
    isr_pt = np.asarray(chunk["lowdm_isr_pt"], dtype=float)
    isr_phi = np.asarray(chunk["lowdm_isr_phi"], dtype=float)
    isr_dphi = absolute_delta_phi(isr_phi, recoil_phi)
    significance = np.divide(
        recoil,
        np.sqrt(ht),
        out=np.full(n, -99.0, dtype=float),
        where=ht > 0.0,
    )
    low_post_prebin = (
        base
        & onelep
        & (njet >= 2)
        & (recoil > 250.0)
        & (ht > 300.0)
        & open_pre
        & low_topology
        & (n_isr == 1)
        & (isr_dphi > 2.0)
        & (significance >= 10.0)
    )
    low_indices_post = lowdm_indices(chunk, recoil, low_post_prebin)
    low_post = low_post_prebin & (low_indices_post >= 0)
    low_strict = np.asarray(chunk["feature_lowdm_LLCR"], dtype=bool) & low_post
    low_target = np.asarray(chunk["feature_lowdm_SR"], dtype=bool)

    transformed_chunk = {
        name: chunk[name]
        for name in ("nb_medium", "nboosted_top", "nboosted_w", "nboosted_total")
    }
    transformed_chunk["met"] = recoil
    high_post_60 = flat.selected_an17_recoil60_indices(
        transformed_chunk, n, high_post
    )
    high_strict_60 = flat.selected_an17_recoil60_indices(
        transformed_chunk, n, high_strict
    )
    high_target_60 = flat.selected_an17_recoil60_indices(
        chunk, n, high_target
    )

    high_ut_idx_removal = categorical_indices(recoil, HIGHDM_UT_BOUNDARIES)
    high_ut_idx_target = categorical_indices(original_met, HIGHDM_UT_BOUNDARIES)
    low_target_idx = np.asarray(chunk["lowdm_search_bin_SR"], dtype=int)

    high_vr_nb0_post = (
        base
        & onelep
        & (njet >= 5)
        & (nb == 0)
        & (recoil > 250.0)
        & open_high
    )
    high_vr_nj34_post = (
        base
        & onelep
        & (njet >= 3)
        & (njet <= 4)
        & (nb >= 1)
        & (recoil > 250.0)
        & open_high
    )
    original_phis = first_jet_phis(chunk["good_jet_phi"], 4)
    original_dphi = [
        absolute_delta_phi(phi, np.asarray(chunk["met_phi"], dtype=float))
        for phi in original_phis
    ]
    original_open_high = (
        (original_dphi[0] > 0.5)
        & (original_dphi[1] > 0.5)
        & (original_dphi[2] > 0.5)
        & (original_dphi[3] > 0.5)
    )
    original_open_pre = (
        (original_dphi[0] > 0.5)
        & (original_dphi[1] > 0.15)
        & (original_dphi[2] > 0.15)
    )
    high_vr_nb0_target = (
        base
        & nolep
        & (njet >= 5)
        & (nb == 0)
        & (original_met > 250.0)
        & original_open_high
    )
    high_vr_nj34_target = (
        base
        & nolep
        & (njet >= 3)
        & (njet <= 4)
        & (nb >= 1)
        & (original_met > 250.0)
        & original_open_high
    )

    original_significance = np.asarray(chunk["met"], dtype=float) / np.sqrt(ht)
    original_isr_dphi = absolute_delta_phi(
        isr_phi, np.asarray(chunk["met_phi"], dtype=float)
    )
    original_low_common = (
        base
        & nolep
        & (njet >= 2)
        & original_open_pre
        & low_topology
        & (n_isr == 1)
        & (original_isr_dphi > 2.0)
    )
    # These validation regions intentionally sit outside the adopted 42-bin
    # search-bin phase space.  Do not inherit the valid-search-bin condition:
    # it would remove every U_T < 300 GeV or pT_ISR < 300 GeV event by
    # construction and produce an artificial zero prediction.
    low_vr_met_post = low_post_prebin & (recoil < 300.0)
    low_vr_isr_post = (
        low_post_prebin & (isr_pt >= 200.0) & (isr_pt < 300.0)
    )
    low_vr_sig_post = (
        base
        & onelep
        & (njet >= 2)
        & (recoil > 250.0)
        & open_pre
        & low_topology
        & (n_isr == 1)
        & (isr_dphi > 2.0)
        & (significance >= 7.0)
        & (significance < 10.0)
    )
    low_vr_met_target = (
        original_low_common
        & (original_met > 250.0)
        & (original_met < 300.0)
        & (original_significance >= 10.0)
    )
    low_vr_isr_target = (
        original_low_common
        & (original_met > 250.0)
        & (original_significance >= 10.0)
        & (isr_pt >= 200.0)
        & (isr_pt < 300.0)
    )
    low_vr_sig_target = (
        original_low_common
        & (original_met > 250.0)
        & (original_significance >= 7.0)
        & (original_significance < 10.0)
    )

    def scheme_payload(
        post: np.ndarray,
        strict: np.ndarray,
        target: np.ndarray,
        pred_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "prediction_post_e": post & removal["electron"],
            "prediction_post_mu": post & removal["muon"],
            "prediction_strict_e": strict & removal["electron"],
            "prediction_strict_mu": strict & removal["muon"],
            "target_inclusive": target,
            "target_leptonic": target,
            "prediction_indices": pred_idx,
            "target_indices": target_idx,
        }

    highdm_search60_payload = scheme_payload(
        high_post,
        high_strict,
        high_target,
        high_post_60,
        high_target_60,
    )
    highdm_search60_payload["strict_prediction_indices"] = high_strict_60

    return {
        "highdm_ut": scheme_payload(
            high_post,
            high_strict,
            high_target,
            high_ut_idx_removal,
            high_ut_idx_target,
        ),
        "highdm_search60": highdm_search60_payload,
        "lowdm_search42": scheme_payload(
            low_post,
            low_strict,
            low_target,
            low_indices_post,
            low_target_idx,
        ),
        "highdm_vr_nb0": scheme_payload(
            high_vr_nb0_post,
            high_vr_nb0_post,
            high_vr_nb0_target,
            high_ut_idx_removal,
            high_ut_idx_target,
        ),
        "highdm_vr_njet3to4_nb1plus": scheme_payload(
            high_vr_nj34_post,
            high_vr_nj34_post,
            high_vr_nj34_target,
            high_ut_idx_removal,
            high_ut_idx_target,
        ),
        "lowdm_vr_met250to300": scheme_payload(
            low_vr_met_post,
            low_vr_met_post,
            low_vr_met_target,
            categorical_indices(recoil, LOWMET_BOUNDARIES),
            categorical_indices(original_met, LOWMET_BOUNDARIES),
        ),
        "lowdm_vr_isr200to300": scheme_payload(
            low_vr_isr_post,
            low_vr_isr_post,
            low_vr_isr_target,
            categorical_indices(isr_pt, LOWISR_BOUNDARIES),
            categorical_indices(isr_pt, LOWISR_BOUNDARIES),
        ),
        "lowdm_vr_significance7to10": scheme_payload(
            low_vr_sig_post,
            low_vr_sig_post,
            low_vr_sig_target,
            categorical_indices(significance, SIGNIFICANCE_BOUNDARIES),
            categorical_indices(
                original_significance, SIGNIFICANCE_BOUNDARIES
            ),
        ),
    }


def init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "1"


def process_root_task(task: dict[str, Any]) -> dict[str, Any]:
    root_path = Path(task["root"])
    process = str(task["process"])
    component = component_for_process(process)
    partial_dir = Path(_WORKER_CONFIG["partial_dir"])
    partial_path = partial_dir / f"{stable_task_id(str(root_path))}.json"
    if partial_path.exists():
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
                "root": str(root_path),
                "partial": str(partial_path),
            }

    sidecar = root_path.with_suffix(".json")
    if not sidecar.exists():
        raise RuntimeError(f"missing sidecar for {root_path}")
    meta = json_load(sidecar)
    norm = json_load(Path(_WORKER_CONFIG["normalization"]))
    repo = Path(_WORKER_CONFIG["repo"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "root": str(root_path),
        "sidecar": str(sidecar),
        "process": process,
        "component": component,
        "normalization_sha256": _WORKER_CONFIG["normalization_sha256"],
        "code_sha256": _WORKER_CONFIG["code_sha256"],
        "events_read": 0,
        "events_selected": 0,
        "datasets": {},
        "histograms": {
            scheme: {
                str(fold): {
                    kind: empty_hist(int(spec["bins"]))
                    for kind in HISTOGRAM_KINDS
                }
                for fold in (0, 1)
            }
            for scheme, spec in SCHEMES.items()
        },
    }

    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        present = set(tree.keys())
        missing = [
            branch
            for branch in READ_BRANCHES
            if branch not in present
            and branch not in flat.OPTIONAL_FORWARD_SCHEMA_BRANCHES
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
            payload = build_masks_and_indices(chunk)
            selected = np.zeros(n, dtype=bool)
            for scheme_payload in payload.values():
                for kind in HISTOGRAM_KINDS:
                    selected |= np.asarray(scheme_payload[kind], dtype=bool)
            if not np.any(selected):
                continue
            result["events_selected"] += int(np.count_nonzero(selected))
            is_data_values = np.asarray(chunk["is_data"], dtype=bool)
            if len(np.unique(is_data_values)) != 1:
                raise RuntimeError(f"mixed data/MC chunk in {root_path}")
            folds = np.asarray(
                legacy.event_hash(chunk, is_data=bool(is_data_values[0]))
                & np.uint64(1),
                dtype=np.int8,
            )
            dataset_ids = np.asarray(chunk["dataset_id"], dtype=np.int64)
            for dataset_id in sorted(set(int(value) for value in dataset_ids)):
                dmask = dataset_ids == dataset_id
                workmask = dmask & selected
                if not np.any(workmask):
                    continue
                subgroup = {
                    name: chunk[name][workmask] for name in ak.fields(chunk)
                }
                weights, status, dataset = legacy.weight_for_dataset(
                    subgroup, meta, norm, repo, dataset_id
                )
                leptonic = pure_leptonic_dataset(dataset, process)
                drec = result["datasets"].setdefault(
                    str(dataset_id),
                    {
                        "dataset": dataset,
                        "process": process,
                        "component": component,
                        "pure_leptonic_decay_filter": leptonic,
                        "events": 0,
                        "sumw": 0.0,
                        "sumw2": 0.0,
                        "weight_status": status,
                    },
                )
                drec["events"] += len(weights)
                drec["sumw"] += float(np.sum(weights))
                drec["sumw2"] += float(np.sum(np.square(weights)))
                local_folds = folds[workmask]
                for scheme, scheme_payload in payload.items():
                    pred_idx = np.asarray(
                        scheme_payload["prediction_indices"], dtype=int
                    )[workmask]
                    strict_idx = np.asarray(
                        scheme_payload.get(
                            "strict_prediction_indices",
                            scheme_payload["prediction_indices"],
                        ),
                        dtype=int,
                    )[workmask]
                    target_idx = np.asarray(
                        scheme_payload["target_indices"], dtype=int
                    )[workmask]
                    for fold in (0, 1):
                        fold_mask = local_folds == fold
                        for kind in HISTOGRAM_KINDS:
                            mask = np.asarray(
                                scheme_payload[kind], dtype=bool
                            )[workmask]
                            if (
                                component == "Data"
                                and SCHEMES[scheme]["blinded_nominal"]
                                and kind.startswith("target")
                            ):
                                mask = np.zeros_like(mask)
                            if kind == "target_leptonic" and not leptonic:
                                mask = np.zeros_like(mask)
                            indices = (
                                target_idx
                                if kind.startswith("target")
                                else strict_idx
                                if kind.startswith("prediction_strict")
                                else pred_idx
                            )
                            add_hist(
                                result["histograms"][scheme][str(fold)][kind],
                                indices,
                                weights,
                                fold_mask & mask,
                            )

    result["status"] = "complete"
    json_dump(partial_path, result)
    return {
        "status": "complete",
        "root": str(root_path),
        "partial": str(partial_path),
    }


def merge_partials(tasks: list[dict[str, Any]], partial_dir: Path) -> dict[str, Any]:
    components = ("Top", "W", "Other", "Data")
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "selection_authority": (
            "autonomous_allhad/autonomous_allhad/real_subset_worker.py"
        ),
        "component_policy": {
            "Top": ["TT", "ST"],
            "W": ["WtoLNu"],
            "independent_target_components": ["Top", "W"],
        },
        "data_nominal_sr_blinded": True,
        "nominal_inputs_modified": False,
        "schemes": SCHEMES,
        "inputs": [],
        "datasets": {},
        "components": {
            component: {
                scheme: {
                    str(fold): {
                        kind: empty_hist(int(spec["bins"]))
                        for kind in HISTOGRAM_KINDS
                    }
                    for fold in (0, 1)
                }
                for scheme, spec in SCHEMES.items()
            }
            for component in components
        },
    }
    for task in tasks:
        root = str(task["root"])
        partial_path = partial_dir / f"{stable_task_id(root)}.json"
        if not partial_path.exists():
            raise RuntimeError(f"missing partial: {partial_path}")
        partial = json_load(partial_path)
        if partial.get("status") != "complete":
            raise RuntimeError(f"incomplete partial: {partial_path}")
        component = str(partial["component"])
        merged["inputs"].append(
            {
                "root": root,
                "partial": str(partial_path),
                "process": partial["process"],
                "component": component,
                "events_read": partial["events_read"],
                "events_selected": partial["events_selected"],
            }
        )
        for dataset_id, record in partial["datasets"].items():
            key = f"{partial['process']}:{dataset_id}"
            target = merged["datasets"].setdefault(
                key,
                {
                    **record,
                    "events": 0,
                    "sumw": 0.0,
                    "sumw2": 0.0,
                },
            )
            target["events"] += int(record["events"])
            target["sumw"] += float(record["sumw"])
            target["sumw2"] += float(record["sumw2"])
        for scheme in SCHEMES:
            for fold in (0, 1):
                for kind in HISTOGRAM_KINDS:
                    merge_hist(
                        merged["components"][component][scheme][str(fold)][kind],
                        partial["histograms"][scheme][str(fold)][kind],
                    )
    merged["input_totals"] = {
        "roots": len(merged["inputs"]),
        "events_read": sum(int(item["events_read"]) for item in merged["inputs"]),
        "events_selected": sum(
            int(item["events_selected"]) for item in merged["inputs"]
        ),
    }
    return merged


def self_test() -> None:
    chunk = {
        "met": ak.Array([300.0, 300.0]),
        "met_phi": ak.Array([0.0, 0.0]),
        "n_e_veto": ak.Array([1, 0]),
        "n_m_loose": ak.Array([0, 1]),
        "electron_veto_pt": ak.Array([[100.0], []]),
        "electron_veto_phi": ak.Array([[0.0], []]),
        "muon_loose_pt": ak.Array([[], [100.0]]),
        "muon_loose_phi": ak.Array([[], [math.pi]]),
    }
    result = emulated_met(chunk)
    assert np.allclose(result["pt"], [400.0, 200.0])
    assert result["electron"].tolist() == [True, False]
    assert result["muon"].tolist() == [False, True]
    assert component_for_process("TT") == "Top"
    assert component_for_process("ST") == "Top"
    assert component_for_process("WtoLNu") == "W"
    print("self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processes", default=",".join(ALL_PROCESSES))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=50_000)
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for required in ("repo", "manifest", "normalization", "output_dir"):
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
    unknown = sorted(set(requested_processes) - set(ALL_PROCESSES))
    if unknown:
        raise SystemExit(f"unknown processes: {', '.join(unknown)}")
    manifest = json_load(manifest_path)
    tasks = legacy.parse_manifest(manifest, requested_processes)
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
        "component_policy": {
            "Top": ["TT", "ST"],
            "W": ["WtoLNu"],
        },
        "jobs": int(args.jobs),
        "step_size": int(args.step_size),
        "tasks": tasks,
        "nominal_inputs_modified": False,
        "data_nominal_sr_blinded": True,
        "selection_authority": (
            "autonomous_allhad/autonomous_allhad/real_subset_worker.py"
        ),
    }
    json_dump(output_dir / "run_manifest.json", run_manifest)
    init_worker(config)
    completed = 0
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
    except Exception:
        if pool is not None:
            pool.terminate()
        run_manifest["status"] = "failed"
        run_manifest["completed_roots_before_failure"] = completed
        json_dump(output_dir / "run_manifest.json", run_manifest)
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    merged = merge_partials(tasks, partial_dir)
    merged["manifest"] = str(manifest_path)
    merged["manifest_sha256"] = run_manifest["manifest_sha256"]
    merged["normalization"] = str(normalization_path)
    merged["normalization_sha256"] = normalization_sha256
    merged["code_sha256"] = code_sha256
    json_dump(output_dir / "merged_histograms.json", merged)
    run_manifest["status"] = "complete"
    run_manifest["completed_at_unix"] = time.time()
    run_manifest["completed_roots"] = completed
    json_dump(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(merged["input_totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
