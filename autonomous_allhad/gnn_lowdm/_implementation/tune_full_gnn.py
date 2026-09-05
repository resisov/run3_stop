from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import uproot
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .compare_oof import weighted_binary_auc
from ..data import (
    BASE_GLOBAL_FEATURE_NAMES,
    TOP_TARGETED_GLOBAL_FEATURE_NAMES,
    GraphEvents,
    load_graph_events,
    split_buckets_2_1_7,
)
from ..model import JetGraphClassifier
from .train_oof import predict, seed_everything, sha256, tensors


CATEGORY_SIZES = np.asarray([4, 4, 4, 4, 3, 3, 3, 3, 3, 3])
TOPOLOGY_NAMES = {1: "T2tt", 2: "T2bW", 3: "T2tb"}
TOPOLOGY_IDS = {name: identifier for identifier, name in TOPOLOGY_NAMES.items()}
DEFAULT_TRIALS = (
    {"name": "h32_l2_lr2e3_d10_tm1_fg0", "hidden": 32, "message_layers": 2, "learning_rate": 2.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 2048, "top_background_multiplier": 1.0, "focal_gamma": 0.0},
    {"name": "h32_l2_lr2e3_d10_tm2_fg0", "hidden": 32, "message_layers": 2, "learning_rate": 2.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 2048, "top_background_multiplier": 2.0, "focal_gamma": 0.0},
    {"name": "h32_l2_lr2e3_d10_tm4_fg0", "hidden": 32, "message_layers": 2, "learning_rate": 2.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 2048, "top_background_multiplier": 4.0, "focal_gamma": 0.0},
    {"name": "h32_l2_lr2e3_d10_tm2_fg1", "hidden": 32, "message_layers": 2, "learning_rate": 2.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 2048, "top_background_multiplier": 2.0, "focal_gamma": 1.0},
    {"name": "h32_l2_lr2e3_d10_tm4_fg1", "hidden": 32, "message_layers": 2, "learning_rate": 2.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 2048, "top_background_multiplier": 4.0, "focal_gamma": 1.0},
    {"name": "h64_l2_lr1e3_d10_tm2_fg1", "hidden": 64, "message_layers": 2, "learning_rate": 1.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 1536, "top_background_multiplier": 2.0, "focal_gamma": 1.0},
    {"name": "h64_l2_lr1e3_d10_tm4_fg1", "hidden": 64, "message_layers": 2, "learning_rate": 1.0e-3, "dropout": 0.10, "weight_decay": 1.0e-4, "batch_size": 1536, "top_background_multiplier": 4.0, "focal_gamma": 1.0},
    {"name": "h64_l3_lr7e4_d15_tm4_fg2", "hidden": 64, "message_layers": 3, "learning_rate": 7.0e-4, "dropout": 0.15, "weight_decay": 3.0e-4, "batch_size": 1536, "top_background_multiplier": 4.0, "focal_gamma": 2.0},
)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def mass_key(topology_id: int, mstop: int, mlsp: int) -> str:
    return f"{TOPOLOGY_NAMES[int(topology_id)]}_{mstop}_{mlsp}"


def load_inputs(opts: argparse.Namespace) -> tuple[GraphEvents, dict[str, object], np.ndarray]:
    state = json.loads((opts.cache / "campaign_state.json").read_text())
    if state.get("status") != "complete":
        raise RuntimeError("feature-cache campaign is not complete")

    def nonempty_cache_paths(kind: str) -> list[Path]:
        output: list[Path] = []
        for path in sorted(opts.cache.glob(f"{kind}_cache_*.root")):
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.exists():
                raise RuntimeError(f"cache shard has no sidecar: {path}")
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar.get("status") != "complete":
                raise RuntimeError(f"cache shard is incomplete: {path}")
            if int(sidecar.get("events_selected", -1)) > 0:
                output.append(path)
        return output

    signal_paths = nonempty_cache_paths("signal")
    background_paths = nonempty_cache_paths("mc")
    shard_limit = getattr(opts, "max_cache_shards_per_kind", None)
    if shard_limit is not None:
        if int(shard_limit) < 1:
            raise ValueError("max_cache_shards_per_kind must be positive")
        signal_paths = signal_paths[: int(shard_limit)]
        background_paths = background_paths[: int(shard_limit)]
    if not signal_paths or not background_paths:
        raise RuntimeError("feature cache is missing signal or background shards")
    events = load_graph_events(
        signal_paths,
        background_paths,
        target_mstop=None,
        target_mlsp=None,
        max_jets=opts.max_jets,
        folds=5,
        require_highdm_exclusive=getattr(opts, "require_highdm_exclusive", True),
        selection_branch=getattr(opts, "selection_branch", "feature_lowdm_SR"),
        include_mass_features=False,
        top_targeted_features=opts.top_targeted_features,
        engineered_features_v2=getattr(opts, "engineered_features_v2", False),
        engineered_features_expanded=getattr(
            opts, "engineered_features_expanded", False
        ),
        engineered_features_diagonal_v3=getattr(
            opts, "engineered_features_diagonal_v3", False
        ),
        signal_topology_ids=tuple(
            TOPOLOGY_IDS[name] for name in opts.signal_topologies
        ),
        signal_delta_m_min=opts.delta_m_min,
        signal_delta_m_max=opts.delta_m_max,
        signal_mstop_min=opts.mstop_min,
        signal_mstop_max=opts.mstop_max,
    )
    signal = events.labels.astype(bool)
    manifest = json.loads(opts.campaign_manifest.read_text())
    normalizable_points = {
        (
            TOPOLOGY_IDS[record["topology"]],
            int(record["mStop"]),
            int(record["mLSP"]),
        )
        for record in manifest["normalization"]["signal_mass_points"]
        if float(record.get("sumw", 0.0)) != 0.0
    }
    signal_normalizable = np.ones(len(events), dtype=bool)
    signal_indices = np.flatnonzero(signal)
    signal_normalizable[signal_indices] = np.fromiter(
        (
            (
                int(events.signal_topology_id[index]),
                int(events.mstop[index]),
                int(events.mlsp[index]),
            )
            in normalizable_points
            for index in signal_indices
        ),
        dtype=bool,
        count=len(signal_indices),
    )
    selected_topologies = np.asarray(
        [TOPOLOGY_IDS[name] for name in opts.signal_topologies], dtype=np.int32
    )
    diagonal = (
        signal
        & np.isin(events.signal_topology_id, selected_topologies)
        & ((events.mstop - events.mlsp) >= opts.delta_m_min)
        & ((events.mstop - events.mlsp) <= opts.delta_m_max)
        & (events.mstop >= opts.mstop_min)
        & (events.mstop <= opts.mstop_max)
    )
    unnormalizable = diagonal & ~signal_normalizable
    diagonal &= signal_normalizable
    state["signal_normalization_filter"] = {
        "policy": (
            "exclude signal events whose topology/mass point has no nonzero generator "
            "sum-of-weights entry; never infer normalization from the selected SR"
        ),
        "events_excluded": int(np.count_nonzero(unnormalizable)),
        "mass_points_excluded": sorted(
            {
                mass_key(
                    int(events.signal_topology_id[index]),
                    int(events.mstop[index]),
                    int(events.mlsp[index]),
                )
                for index in np.flatnonzero(unnormalizable)
            }
        ),
    }
    keep = ~signal | diagonal
    events = events.take(np.flatnonzero(keep))
    if getattr(opts, "require_adopted_lowdm_bins", True) and (
        np.any(events.lowdm_search_bin < 8)
        or np.any(events.lowdm_search_bin > 41)
    ):
        raise RuntimeError("cache contains events outside the adopted 34-bin Low-dM SR")
    keys = np.stack(
        (events.physical_dataset_id, events.run, events.luminosity_block, events.event),
        axis=1,
    )
    if len(np.unique(keys, axis=0)) != len(keys):
        raise RuntimeError("feature cache contains duplicate event keys")
    split = split_buckets_2_1_7(
        events.physical_dataset_id, events.run, events.luminosity_block, events.event
    )
    return events, state, split


def physics_weights(
    events: GraphEvents,
    campaign: dict[str, object],
    xsec_payload: dict[str, object],
) -> np.ndarray:
    manifest = json.loads(Path(campaign["manifest"]).read_text())
    luminosity_pb = float(manifest["normalization"]["luminosity_pb"])
    background_norm = manifest["normalization"]["by_physical_dataset_id"]
    signal_norm = {
        (record["topology"], int(record["mStop"]), int(record["mLSP"])): float(record["sumw"])
        for record in manifest["normalization"]["signal_mass_points"]
    }
    stop_xsec = {
        int(record["mStop"]): float(record["xsec_pb"])
        for record in xsec_payload["records"]
        if record.get("parsing_status") == "parsed"
    }
    output = np.asarray(events.gen_weight, dtype=np.float64).copy()
    background = events.labels == 0
    for physical_id in np.unique(events.physical_dataset_id[background]):
        record = background_norm.get(str(int(physical_id)))
        if record is None or not record.get("normalization_complete"):
            raise RuntimeError(f"incomplete background normalization for {physical_id}")
        selected = background & (events.physical_dataset_id == physical_id)
        output[selected] *= float(record["xsec_pb"]) * luminosity_pb / float(record["sumw"])
    signal = ~background
    signal_points = np.unique(
        np.stack(
            (
                events.signal_topology_id[signal],
                events.mstop[signal],
                events.mlsp[signal],
            ),
            axis=1,
        ),
        axis=0,
    )
    for topology_id, mstop, mlsp in signal_points:
        topology = TOPOLOGY_NAMES[int(topology_id)]
        denominator = signal_norm.get((topology, int(mstop), int(mlsp)))
        if denominator is None or denominator == 0.0:
            raise RuntimeError(f"missing signal sumw for {topology}({mstop},{mlsp})")
        xsec = stop_xsec.get(int(mstop))
        if xsec is None:
            raise RuntimeError(f"missing stop cross section for mStop={mstop}")
        selected = (
            signal
            & (events.signal_topology_id == topology_id)
            & (events.mstop == mstop)
            & (events.mlsp == mlsp)
        )
        output[selected] *= xsec * luminosity_pb / denominator
    if not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite normalized physics weights")
    return output


def background_process_names(
    events: GraphEvents, campaign_manifest: dict[str, object]
) -> np.ndarray:
    background = events.labels == 0
    normalization = campaign_manifest["normalization"]["by_physical_dataset_id"]
    output = np.full(len(events), "signal", dtype=object)
    for physical_id in np.unique(events.physical_dataset_id[background]):
        record = normalization.get(str(int(physical_id)))
        if record is None:
            raise RuntimeError(f"missing process mapping for physical dataset {physical_id}")
        output[background & (events.physical_dataset_id == physical_id)] = record[
            "process"
        ]
    return output


def balanced_loss_weights(
    events: GraphEvents,
    indices: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray | None = None,
    top_background_multiplier: float = 1.0,
) -> np.ndarray:
    labels = events.labels[indices].astype(bool)
    weights = np.abs(analysis_weights[indices]).astype(np.float64)
    background = ~labels
    if not np.any(background) or not np.any(labels):
        raise RuntimeError("training split must contain both classes")
    selected_processes = (
        np.asarray(process_names, dtype=object)[indices]
        if process_names is not None
        else np.full(len(indices), "background", dtype=object)
    )
    top_background = background & np.isin(selected_processes, ("TT", "ST"))
    weights[top_background] *= float(top_background_multiplier)
    weights[background] *= 0.5 / weights[background].sum()
    points = np.unique(
        np.stack(
            (
                events.signal_topology_id[indices][labels],
                events.mstop[indices][labels],
                events.mlsp[indices][labels],
            ),
            axis=1,
        ),
        axis=0,
    )
    topologies = np.unique(points[:, 0])
    for topology_id in topologies:
        topology_points = points[points[:, 0] == topology_id]
        for _, mstop, mlsp in topology_points:
            local = (
                labels
                & (events.signal_topology_id[indices] == topology_id)
                & (events.mstop[indices] == mstop)
                & (events.mlsp[indices] == mlsp)
            )
            total = weights[local].sum()
            if total <= 0.0:
                raise RuntimeError(
                    f"zero training weight for {mass_key(topology_id, mstop, mlsp)}"
                )
            weights[local] *= (
                0.5 / len(topologies) / len(topology_points) / total
            )
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def weighted_dataset(
    events: GraphEvents,
    indices: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray,
    top_background_multiplier: float,
) -> TensorDataset:
    base = tensors(events, indices)
    return TensorDataset(
        *base.tensors,
        torch.from_numpy(
            balanced_loss_weights(
                events,
                indices,
                analysis_weights,
                process_names,
                top_background_multiplier,
            )
        ),
    )


def positive_auc_contributions(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    negative_weights: np.ndarray,
) -> np.ndarray:
    """Return each positive event's weighted probability to outrank a negative.

    Sorting a shared negative class once is exactly equivalent to repeatedly
    evaluating a two-class weighted AUC, including half credit for score ties.
    """
    order = np.argsort(negative_scores, kind="stable")
    sorted_scores = np.asarray(negative_scores, dtype=float)[order]
    sorted_weights = np.asarray(negative_weights, dtype=float)[order]
    prefix = np.r_[0.0, np.cumsum(sorted_weights)]
    total = float(prefix[-1])
    if total <= 0.0:
        raise RuntimeError("AUC negative class has zero weight")
    positive_scores = np.asarray(positive_scores, dtype=float)
    left = np.searchsorted(sorted_scores, positive_scores, side="left")
    right = np.searchsorted(sorted_scores, positive_scores, side="right")
    below = prefix[left]
    tied = prefix[right] - prefix[left]
    return (below + 0.5 * tied) / total


def weighted_efficiency_above(
    threshold: float,
    sorted_scores: np.ndarray,
    weight_prefix: np.ndarray,
) -> float:
    position = int(np.searchsorted(sorted_scores, threshold, side="left"))
    total = float(weight_prefix[-1])
    if total <= 0.0:
        raise RuntimeError("efficiency denominator has zero weight")
    return float((total - weight_prefix[position]) / total)


def validation_metrics(
    events: GraphEvents,
    indices: np.ndarray,
    scores: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray,
) -> dict[str, object]:
    labels = events.labels[indices].astype(np.int8)
    absolute = np.abs(analysis_weights[indices])
    signal = labels == 1
    background = ~signal
    metric_weights = absolute.copy()
    metric_weights[background] /= metric_weights[background].sum()
    points = np.unique(
        np.stack(
            (
                events.signal_topology_id[indices][signal],
                events.mstop[indices][signal],
                events.mlsp[indices][signal],
            ),
            axis=1,
        ),
        axis=0,
    )
    per_mass: dict[str, float] = {}
    per_mass_top: dict[str, float] = {}
    per_mass_top_rejection: dict[str, float] = {}
    top_background = background & np.isin(process_names[indices], ("TT", "ST"))
    if not np.any(top_background):
        raise RuntimeError("validation split has no TT/ST background")
    all_auc_contribution = np.zeros(len(labels), dtype=float)
    all_auc_contribution[signal] = positive_auc_contributions(
        scores[signal], scores[background], absolute[background]
    )
    top_auc_contribution = np.zeros(len(labels), dtype=float)
    top_auc_contribution[signal] = positive_auc_contributions(
        scores[signal], scores[top_background], absolute[top_background]
    )
    top_order = np.argsort(scores[top_background], kind="stable")
    sorted_top_scores = scores[top_background][top_order]
    sorted_top_weight_prefix = np.r_[
        0.0, np.cumsum(absolute[top_background][top_order])
    ]
    for topology_id, mstop, mlsp in points:
        local_signal = (
            signal
            & (events.signal_topology_id[indices] == topology_id)
            & (events.mstop[indices] == mstop)
            & (events.mlsp[indices] == mlsp)
        )
        metric_weights[local_signal] /= metric_weights[local_signal].sum() * len(points)
        key = mass_key(int(topology_id), int(mstop), int(mlsp))
        per_mass[key] = float(
            np.average(
                all_auc_contribution[local_signal],
                weights=absolute[local_signal],
            )
        )
        per_mass_top[key] = float(
            np.average(
                top_auc_contribution[local_signal],
                weights=absolute[local_signal],
            )
        )
        signal_median = float(
            weighted_quantiles(
                scores[local_signal], absolute[local_signal], np.asarray([0.5])
            )[0]
        )
        top_efficiency = weighted_efficiency_above(
            signal_median, sorted_top_scores, sorted_top_weight_prefix
        )
        per_mass_top_rejection[key] = 1.0 - top_efficiency
    values = np.asarray(list(per_mass.values()), dtype=float)
    top_values = np.asarray(list(per_mass_top.values()), dtype=float)
    top_rejection_values = np.asarray(
        list(per_mass_top_rejection.values()), dtype=float
    )
    per_topology = {}
    for topology_id, topology_name in TOPOLOGY_NAMES.items():
        prefix = f"{topology_name}_"
        selected_all = [value for key, value in per_mass.items() if key.startswith(prefix)]
        selected_top = [
            value for key, value in per_mass_top.items() if key.startswith(prefix)
        ]
        selected_top_rejection = [
            value
            for key, value in per_mass_top_rejection.items()
            if key.startswith(prefix)
        ]
        if selected_all:
            per_topology[topology_name] = {
                "mass_points": len(selected_all),
                "macro_all_background_auc": float(np.mean(selected_all)),
                "minimum_all_background_auc": float(np.min(selected_all)),
                "macro_top_background_auc": float(np.mean(selected_top)),
                "minimum_top_background_auc": float(np.min(selected_top)),
                "macro_top_rejection_at_50pct_signal_efficiency": float(
                    np.mean(selected_top_rejection)
                ),
            }
    return {
        "macro_mass_auc": float(np.mean(values)),
        "minimum_mass_auc": float(np.min(values)),
        "mass_auc_std": float(np.std(values)),
        "macro_top_background_auc": float(np.mean(top_values)),
        "minimum_top_background_auc": float(np.min(top_values)),
        "top_background_auc_std": float(np.std(top_values)),
        "macro_top_rejection_at_50pct_signal_efficiency": float(
            np.mean(top_rejection_values)
        ),
        "minimum_top_rejection_at_50pct_signal_efficiency": float(
            np.min(top_rejection_values)
        ),
        "equal_mass_weighted_auc": float(
            np.average(
                all_auc_contribution[signal], weights=metric_weights[signal]
            )
        ),
        "per_mass_auc": per_mass,
        "per_mass_top_background_auc": per_mass_top,
        "per_mass_top_rejection_at_50pct_signal_efficiency": per_mass_top_rejection,
        "per_topology": per_topology,
    }


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    if np.any(sorted_weights < 0.0) or sorted_weights.sum() <= 0.0:
        raise RuntimeError("weighted quantiles require nonnegative, nonzero weights")
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(quantiles, positions, sorted_values)


def optimize_validation_score_edges(
    events: GraphEvents,
    indices: np.ndarray,
    scores: np.ndarray,
    analysis_weights: np.ndarray,
) -> dict[str, object]:
    """Choose two common score edges without opening the locked test split.

    Candidate edges are physics-weighted background quantiles.  Every signal
    mass point contributes equally to the objective, so the choice cannot be
    driven by one abundant benchmark or by the rapidly falling stop xsec.
    """
    labels = events.labels[indices].astype(bool)
    background = ~labels
    category = raw_category(events.lowdm_search_bin[indices])
    if np.any(category < 0):
        raise RuntimeError("validation event outside adopted Low-dM categories")
    absolute = np.abs(analysis_weights[indices])
    low_quantiles = np.asarray([0.50, 0.60, 0.70, 0.75, 0.80])
    high_quantiles = np.asarray([0.85, 0.90, 0.92, 0.95, 0.97])
    quantile_values = weighted_quantiles(
        scores[background], absolute[background], np.r_[low_quantiles, high_quantiles]
    )
    low_values = quantile_values[: len(low_quantiles)]
    high_values = quantile_values[len(low_quantiles) :]
    signal_points = np.unique(
        np.stack(
            (
                events.signal_topology_id[indices][labels],
                events.mstop[indices][labels],
                events.mlsp[indices][labels],
            ),
            axis=1,
        ),
        axis=0,
    )
    candidates: list[dict[str, object]] = []
    nscore = 3
    nbin = len(CATEGORY_SIZES) * nscore
    for low_q, low_edge in zip(low_quantiles, low_values):
        for high_q, high_edge in zip(high_quantiles, high_values):
            if not 0.0 < low_edge < high_edge < 1.0:
                continue
            edges = np.asarray([0.0, low_edge, high_edge, 1.0])
            score_bin = np.clip(
                np.searchsorted(edges, scores, side="right") - 1, 0, nscore - 1
            )
            flat_bin = category * nscore + score_bin
            background_abs = np.bincount(
                flat_bin[background], weights=absolute[background], minlength=nbin
            )
            background_sumw2 = np.bincount(
                flat_bin[background], weights=absolute[background] ** 2, minlength=nbin
            )
            background_fraction = background_abs / background_abs.sum()
            total_effective = (
                background_abs.sum() ** 2 / max(background_sumw2.sum(), 1.0e-30)
            )
            smoothing = 1.0 / max(total_effective, 1.0)
            effective_by_bin = np.divide(
                background_abs**2,
                background_sumw2,
                out=np.zeros_like(background_abs),
                where=background_sumw2 > 0.0,
            )
            separation: list[float] = []
            for topology_id, mstop, mlsp in signal_points:
                local = (
                    labels
                    & (events.signal_topology_id[indices] == topology_id)
                    & (events.mstop[indices] == mstop)
                    & (events.mlsp[indices] == mlsp)
                )
                signal_abs = np.bincount(
                    flat_bin[local], weights=absolute[local], minlength=nbin
                )
                if signal_abs.sum() <= 0.0:
                    raise RuntimeError(
                        f"zero validation weight for {mass_key(topology_id, mstop, mlsp)}"
                    )
                signal_fraction = signal_abs / signal_abs.sum()
                separation.append(
                    float(
                        np.sqrt(
                            np.sum(signal_fraction**2 / (background_fraction + smoothing))
                        )
                    )
                )
            candidates.append(
                {
                    "background_quantiles": [float(low_q), float(high_q)],
                    "edges": edges.tolist(),
                    "macro_equal_mass_separation": float(np.mean(separation)),
                    "minimum_mass_separation": float(np.min(separation)),
                    "background_effective_events_10pct_validation": {
                        "minimum": float(np.min(effective_by_bin)),
                        "p10": float(np.quantile(effective_by_bin, 0.10)),
                        "median": float(np.median(effective_by_bin)),
                    },
                }
            )
    if not candidates:
        raise RuntimeError("no valid validation-only score-binning candidate")
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["macro_equal_mass_separation"]),
            float(row["minimum_mass_separation"]),
            float(row["background_effective_events_10pct_validation"]["p10"]),
        ),
        reverse=True,
    )
    return {
        "policy": (
            "two common edges selected on validation only from physics-weighted background "
            "quantiles; maximize equal-mass macro separation, tie-break minimum-mass separation "
            "and 10th-percentile background effective events"
        ),
        "test_set_touched": False,
        "categories": len(CATEGORY_SIZES),
        "score_bins_per_category": 3,
        "candidate_bins": len(CATEGORY_SIZES) * 3,
        "best": ranked[0],
        "candidates": ranked,
    }


def train_trial(
    opts: argparse.Namespace,
    config: dict[str, object],
    events: GraphEvents,
    split: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    trial_dir = opts.output / "trials" / str(config["name"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "complete" and int(summary.get("epochs", 0)) == opts.epochs:
            return summary
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    train_data = weighted_dataset(
        events,
        train_indices,
        analysis_weights,
        process_names,
        float(config.get("top_background_multiplier", 1.0)),
    )
    validation_data = tensors(events, validation_indices)
    seed = opts.seed + sum(ord(character) for character in str(config["name"]))
    seed_everything(seed)
    model = JetGraphClassifier(
        global_features=events.global_features.shape[1],
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    latest_path = trial_dir / "latest.pt"
    start_epoch = 0
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") == config and int(checkpoint.get("epochs_target")) == opts.epochs:
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"])
            history = checkpoint["history"]
            best = checkpoint["best"]
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if "python_rng_state" in checkpoint:
                random.setstate(checkpoint["python_rng_state"])
            if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
                torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    started = time.time()
    for epoch in range(start_epoch, opts.epochs):
        loader = DataLoader(
            train_data,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        weighted_loss = 0.0
        weight_seen = 0.0
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-2])
            per_event = criterion(logits, batch[-2])
            focal_gamma = float(config.get("focal_gamma", 0.0))
            if focal_gamma > 0.0:
                probability = torch.sigmoid(logits)
                correct_class_probability = torch.where(
                    batch[-2] > 0.5, probability, 1.0 - probability
                )
                per_event = per_event * (1.0 - correct_class_probability).pow(
                    focal_gamma
                )
            loss = (per_event * batch[-1]).sum() / batch[-1].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            weighted_loss += float((per_event.detach() * batch[-1]).sum().cpu())
            weight_seen += float(batch[-1].sum().cpu())
        validation_scores = predict(
            model, validation_data, int(config["batch_size"]), device
        )
        metrics = validation_metrics(
            events,
            validation_indices,
            validation_scores,
            analysis_weights,
            process_names,
        )
        row = {
            "epoch": epoch + 1,
            "training_weighted_bce": weighted_loss / max(weight_seen, 1.0),
            **metrics,
        }
        history.append(row)
        if best is None or (
            row["macro_top_rejection_at_50pct_signal_efficiency"],
            row["macro_top_background_auc"],
            row["minimum_top_background_auc"],
            row["macro_mass_auc"],
            row["minimum_mass_auc"],
        ) > (
            best["macro_top_rejection_at_50pct_signal_efficiency"],
            best["macro_top_background_auc"],
            best["minimum_top_background_auc"],
            best["macro_mass_auc"],
            best["minimum_mass_auc"],
        ):
            best = {**row}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "global_features": events.global_features.shape[1],
                    "mass_features": False,
                    "top_targeted_features": opts.top_targeted_features,
                    "signal_topologies": opts.signal_topologies,
                    "selection_metric": (
                        "validation macro TT/ST rejection at 50% signal efficiency; "
                        "tie-break top-background macro/minimum AUC, then all-background "
                        "macro/minimum AUC"
                    ),
                },
                trial_dir / "best.pt",
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "epochs_target": opts.epochs,
                "history": history,
                "best": best,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
            },
            latest_path,
        )
        write_json(
            trial_dir / "progress.json",
            {
                "status": "running",
                "trial": config,
                "epoch": epoch + 1,
                "epochs": opts.epochs,
                "latest": row,
                "best": best,
                "runtime_seconds": time.time() - started,
            },
        )
        print(
            json.dumps(
                {
                    "trial": config["name"],
                    "epoch": epoch + 1,
                    "train_bce": row["training_weighted_bce"],
                    "validation_macro_auc": row["macro_mass_auc"],
                    "validation_min_auc": row["minimum_mass_auc"],
                    "validation_macro_top_auc": row["macro_top_background_auc"],
                    "validation_min_top_auc": row["minimum_top_background_auc"],
                    "validation_macro_top_rejection_at_50pct_signal": row[
                        "macro_top_rejection_at_50pct_signal_efficiency"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema_version": "gnn_lowdm_hyperparameter_trial_v1",
        "status": "complete",
        "trial": config,
        "epochs": opts.epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_events": len(train_indices),
        "validation_events": len(validation_indices),
        "best": best,
        "history": history,
        "runtime_seconds": time.time() - started,
        "checkpoint": str(trial_dir / "best.pt"),
    }
    write_json(summary_path, summary)
    return summary


def split_audit(events: GraphEvents, split: np.ndarray) -> dict[str, object]:
    names = {0: "train", 1: "validation", 2: "test"}
    output: dict[str, object] = {}
    for code, name in names.items():
        selected = split == code
        signal = selected & (events.labels == 1)
        output[name] = {
            "events": int(np.count_nonzero(selected)),
            "fraction": float(np.mean(selected)),
            "background": int(np.count_nonzero(selected & (events.labels == 0))),
            "signal": int(np.count_nonzero(signal)),
            "signal_by_mass": {
                mass_key(int(topology_id), int(mstop), int(mlsp)): int(
                    np.count_nonzero(
                        signal
                        & (events.signal_topology_id == topology_id)
                        & (events.mstop == mstop)
                        & (events.mlsp == mlsp)
                    )
                )
                for topology_id, mstop, mlsp in np.unique(
                    np.stack(
                        (
                            events.signal_topology_id[signal],
                            events.mstop[signal],
                            events.mlsp[signal],
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
            } if np.any(signal) else {},
        }
    return output


def background_process_audit(
    events: GraphEvents,
    split: np.ndarray,
    analysis_weights: np.ndarray,
    campaign_manifest: dict[str, object],
) -> dict[str, object]:
    background = events.labels == 0
    normalization = campaign_manifest["normalization"]["by_physical_dataset_id"]
    process = np.full(len(events), "signal", dtype=object)
    for physical_id in np.unique(events.physical_dataset_id[background]):
        process[background & (events.physical_dataset_id == physical_id)] = normalization[
            str(int(physical_id))
        ]["process"]
    return {
        name: {
            "events": int(np.count_nonzero(background & (process == name))),
            "events_by_split": {
                "train": int(np.count_nonzero(background & (process == name) & (split == 0))),
                "validation": int(
                    np.count_nonzero(background & (process == name) & (split == 1))
                ),
                "test": int(np.count_nonzero(background & (process == name) & (split == 2))),
            },
            "signed_expected_yield_full_simulated_sample": float(
                np.sum(analysis_weights[background & (process == name)])
            ),
            "absolute_weight_sum": float(
                np.sum(np.abs(analysis_weights[background & (process == name)]))
            ),
        }
        for name in sorted(set(process[background]))
    }


def tuning_stage(
    opts: argparse.Namespace,
    events: GraphEvents,
    split: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    if opts.trials is None:
        configs = [dict(config) for config in DEFAULT_TRIALS]
        write_json(opts.output / "hyperparameter_grid.json", configs)
    else:
        configs = json.loads(opts.trials.read_text())
    started = time.time()
    summaries = []
    for config in configs:
        if time.time() - started > opts.max_tuning_hours * 3600.0:
            break
        summaries.append(
            train_trial(
                opts,
                config,
                events,
                split,
                analysis_weights,
                process_names,
                device,
            )
        )
    complete = [summary for summary in summaries if summary.get("status") == "complete"]
    if not complete:
        raise RuntimeError("no hyperparameter trial completed")
    ranked = sorted(
        complete,
        key=lambda summary: (
            float(
                summary["best"][
                    "macro_top_rejection_at_50pct_signal_efficiency"
                ]
            ),
            float(summary["best"]["macro_top_background_auc"]),
            float(summary["best"]["minimum_top_background_auc"]),
            float(summary["best"]["macro_mass_auc"]),
            float(summary["best"]["minimum_mass_auc"]),
        ),
        reverse=True,
    )
    best_checkpoint = torch.load(
        Path(ranked[0]["checkpoint"]), map_location="cpu", weights_only=False
    )
    best_config = best_checkpoint["config"]
    best_model = JetGraphClassifier(
        global_features=events.global_features.shape[1],
        hidden=int(best_config["hidden"]),
        message_layers=int(best_config["message_layers"]),
        dropout=float(best_config["dropout"]),
    ).to(device)
    best_model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    validation_indices = np.flatnonzero(split == 1)
    validation_scores = predict(
        best_model,
        tensors(events, validation_indices),
        int(best_config["batch_size"]),
        device,
    )
    score_binning = optimize_validation_score_edges(
        events,
        validation_indices,
        validation_scores,
        analysis_weights,
    )
    selection = {
        "schema_version": "gnn_lowdm_hyperparameter_selection_v1",
        "status": "complete" if len(complete) == len(configs) else "time_limited",
        "test_set_touched": False,
        "selection_policy": (
            "maximum validation macro TT/ST rejection at 50% signal efficiency; "
            "tie-break top-background macro/minimum AUC, then all-background macro "
            "and minimum AUC"
        ),
        "trials_requested": len(configs),
        "trials_completed": len(complete),
        "best": ranked[0],
        "score_binning": score_binning,
        "ranking": [
            {
                "rank": rank + 1,
                "name": summary["trial"]["name"],
                "macro_mass_auc": summary["best"]["macro_mass_auc"],
                "minimum_mass_auc": summary["best"]["minimum_mass_auc"],
                "macro_top_background_auc": summary["best"][
                    "macro_top_background_auc"
                ],
                "minimum_top_background_auc": summary["best"][
                    "minimum_top_background_auc"
                ],
                "macro_top_rejection_at_50pct_signal_efficiency": summary["best"][
                    "macro_top_rejection_at_50pct_signal_efficiency"
                ],
                "best_epoch": summary["best"]["epoch"],
                "checkpoint": summary["checkpoint"],
            }
            for rank, summary in enumerate(ranked)
        ],
        "runtime_seconds": time.time() - started,
    }
    write_json(opts.output / "selection.json", selection)
    return selection


def raw_category(raw: np.ndarray) -> np.ndarray:
    boundaries = 8 + np.cumsum(CATEGORY_SIZES)
    output = np.searchsorted(boundaries, raw, side="right")
    output[(raw < 8) | (raw > 41)] = -1
    return output


def asimov_z(signal: np.ndarray, background: np.ndarray) -> float:
    valid = background > 0.0
    terms = np.zeros_like(background, dtype=float)
    terms[valid] = 2.0 * (
        (signal[valid] + background[valid]) * np.log1p(signal[valid] / background[valid])
        - signal[valid]
    )
    return float(np.sqrt(np.sum(np.maximum(terms, 0.0))))


def test_stage(
    opts: argparse.Namespace,
    events: GraphEvents,
    split: np.ndarray,
    analysis_weights: np.ndarray,
    full_process_names: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    selection_path = opts.output / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("test set remains locked until selection.json exists")
    selection = json.loads(selection_path.read_text())
    if selection.get("test_set_touched"):
        existing_summary = Path(selection.get("test_summary", ""))
        if not existing_summary.exists():
            raise RuntimeError("selection says test was opened but its summary is missing")
        return json.loads(existing_summary.read_text())
    best = selection["best"]
    checkpoint_path = Path(best["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = JetGraphClassifier(
        global_features=int(checkpoint["global_features"]),
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_indices = np.flatnonzero(split == 2)
    scores = predict(model, tensors(events, test_indices), int(config["batch_size"]), device)
    metrics = validation_metrics(
        events, test_indices, scores, analysis_weights, full_process_names
    )
    test_events = events.take(test_indices)
    # Only the untouched 70% test partition enters the final templates.  Scale
    # its Monte Carlo weights to the full expected luminosity without reusing
    # the 30% of events consumed by training and validation.
    test_fraction = 0.7
    test_weights = analysis_weights[test_indices] / test_fraction
    labels = test_events.labels.astype(bool)
    category = raw_category(test_events.lowdm_search_bin)
    if np.any(category < 0):
        raise RuntimeError("test event outside adopted Low-dM categories")
    score_edges = np.asarray(selection["score_binning"]["best"]["edges"], dtype=float)
    if len(score_edges) != 4 or not np.all(np.diff(score_edges) > 0.0):
        raise RuntimeError("selected validation-only score edges are invalid")
    score_bin = np.clip(
        np.searchsorted(score_edges, scores, side="right") - 1, 0, len(score_edges) - 2
    )
    flat_bin = category * (len(score_edges) - 1) + score_bin
    nbin = len(CATEGORY_SIZES) * (len(score_edges) - 1)
    manifest = json.loads(opts.campaign_manifest.read_text())
    background_norm = manifest["normalization"]["by_physical_dataset_id"]
    process = np.full(len(test_events), "signal", dtype=object)
    for physical_id in np.unique(test_events.physical_dataset_id[~labels]):
        process[(test_events.physical_dataset_id == physical_id) & ~labels] = background_norm[str(int(physical_id))]["process"]
    process_names = sorted(set(process[~labels]))
    process_hists = {
        name: np.bincount(
            flat_bin[(~labels) & (process == name)],
            weights=test_weights[(~labels) & (process == name)],
            minlength=nbin,
        )
        for name in process_names
    }
    process_sumw2 = {
        name: np.bincount(
            flat_bin[(~labels) & (process == name)],
            weights=test_weights[(~labels) & (process == name)] ** 2,
            minlength=nbin,
        )
        for name in process_names
    }
    high_score = score_bin == (len(score_edges) - 2)
    process_score_diagnostics = {}
    for name in process_names:
        local = (~labels) & (process == name)
        signed_total = float(np.sum(test_weights[local]))
        signed_high = float(np.sum(test_weights[local & high_score]))
        absolute_total = float(np.sum(np.abs(test_weights[local])))
        absolute_high = float(np.sum(np.abs(test_weights[local & high_score])))
        process_score_diagnostics[name] = {
            "events": int(np.count_nonzero(local)),
            "events_high_score": int(np.count_nonzero(local & high_score)),
            "signed_yield": signed_total,
            "signed_yield_high_score": signed_high,
            "absolute_weight_fraction_high_score": (
                absolute_high / absolute_total if absolute_total > 0.0 else None
            ),
        }
    background = sum(process_hists.values(), np.zeros(nbin, dtype=float))
    background_sumw2 = sum(process_sumw2.values(), np.zeros(nbin, dtype=float))
    raw_bin = test_events.lowdm_search_bin - 8
    raw_nbin = int(np.sum(CATEGORY_SIZES))
    raw_background = np.bincount(
        raw_bin[~labels], weights=test_weights[~labels], minlength=raw_nbin
    )
    raw_background_sumw2 = np.bincount(
        raw_bin[~labels], weights=test_weights[~labels] ** 2, minlength=raw_nbin
    )
    signal_hists: dict[str, np.ndarray] = {}
    raw_signal_hists: dict[str, np.ndarray] = {}
    for topology_id, mstop, mlsp in np.unique(
        np.stack(
            (
                test_events.signal_topology_id[labels],
                test_events.mstop[labels],
                test_events.mlsp[labels],
            ),
            axis=1,
        ),
        axis=0,
    ):
        local = (
            labels
            & (test_events.signal_topology_id == topology_id)
            & (test_events.mstop == mstop)
            & (test_events.mlsp == mlsp)
        )
        key = mass_key(int(topology_id), int(mstop), int(mlsp))
        signal_hists[key] = np.bincount(
            flat_bin[local], weights=test_weights[local], minlength=nbin
        )
        raw_signal_hists[key] = np.bincount(
            raw_bin[local], weights=test_weights[local], minlength=raw_nbin
        )
    edges = np.arange(nbin + 1, dtype=float)
    with uproot.recreate(opts.output / "test_scores.root") as root_file:
        root_file["Events"] = {
            "physical_dataset_id": test_events.physical_dataset_id,
            "run": test_events.run,
            "luminosityBlock": test_events.luminosity_block,
            "event": test_events.event,
            "is_signal": labels.astype(np.int32),
            "process_id": test_events.process_id,
            "signal_topology_id": test_events.signal_topology_id,
            "mStop": test_events.mstop,
            "mLSP": test_events.mlsp,
            "lowdm_search_bin_SR": test_events.lowdm_search_bin,
            "lowdm_category": category,
            "gnn_score": scores,
            "candidate_bin": flat_bin,
            "signed_normalized_weight": test_weights,
        }
    with uproot.recreate(opts.output / "lowdm_30bin_test_templates.root") as root_file:
        root_file["background_total"] = (background, edges)
        root_file["background_sumw2"] = (background_sumw2, edges)
        for name in process_names:
            root_file[f"background/{name}"] = (process_hists[name], edges)
            root_file[f"background_process_sumw2/{name}"] = (process_sumw2[name], edges)
        for name, values in signal_hists.items():
            root_file[f"signal/{name}"] = (values, edges)
        raw_edges = np.arange(raw_nbin + 1, dtype=float)
        root_file["baseline_raw34/background_total"] = (raw_background, raw_edges)
        root_file["baseline_raw34/background_sumw2"] = (raw_background_sumw2, raw_edges)
        for name, values in raw_signal_hists.items():
            root_file[f"baseline_raw34/signal/{name}"] = (values, raw_edges)
    mass_diagnostics = {
        name: {
            "yield": float(np.sum(values)),
            "yield_high_score": float(
                np.sum(
                    values[
                        np.arange(len(values)) % (len(score_edges) - 1)
                        == (len(score_edges) - 2)
                    ]
                )
            ),
            "asimov_z_no_nuisances_score30": asimov_z(values, background),
            "asimov_z_no_nuisances_raw34": asimov_z(
                raw_signal_hists[name], raw_background
            ),
            "asimov_z_no_nuisances_inclusive": asimov_z(
                np.asarray([np.sum(values)]), np.asarray([np.sum(background)])
            ),
        }
        for name, values in signal_hists.items()
    }
    for values in mass_diagnostics.values():
        baseline_z = float(values["asimov_z_no_nuisances_raw34"])
        inclusive_z = float(values["asimov_z_no_nuisances_inclusive"])
        score_z = float(values["asimov_z_no_nuisances_score30"])
        values["score30_over_raw34"] = score_z / baseline_z if baseline_z > 0.0 else None
        values["score30_over_inclusive"] = score_z / inclusive_z if inclusive_z > 0.0 else None
        values["high_score_efficiency"] = (
            float(values["yield_high_score"]) / float(values["yield"])
            if float(values["yield"]) != 0.0
            else None
        )
    result = {
        "schema_version": "gnn_lowdm_locked_test_v1",
        "status": "complete",
        "test_opened_after_hyperparameter_selection": True,
        "test_template_weight_scale": 1.0 / test_fraction,
        "template_normalization_policy": (
            "held-out 70% test events only, scaled by 1/0.7 to the full expected luminosity"
        ),
        "selection_sha256_before_test": sha256(selection_path),
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "configuration": config,
        "selected_epoch": checkpoint["epoch"],
        "events": {
            "total": len(test_events),
            "background": int(np.count_nonzero(~labels)),
            "signal": int(np.count_nonzero(labels)),
        },
        "metrics": metrics,
        "high_score_diagnostics": {
            "threshold": float(score_edges[-2]),
            "background_processes": process_score_diagnostics,
            "TT_ST_signed_yield_high_score": float(
                sum(
                    process_score_diagnostics[name]["signed_yield_high_score"]
                    for name in ("TT", "ST")
                    if name in process_score_diagnostics
                )
            ),
            "total_background_signed_yield_high_score": float(
                sum(
                    record["signed_yield_high_score"]
                    for record in process_score_diagnostics.values()
                )
            ),
        },
        "thirty_bin_model": {
            "categories": 10,
            "score_edges": score_edges.tolist(),
            "score_edge_selection": "validation only; copied from selection.json",
            "bins": nbin,
            "background_yield": float(np.sum(background)),
            "background_nonpositive_bins": int(np.count_nonzero(background <= 0.0)),
            "background_bins_above_30pct_mc_stat": int(
                np.count_nonzero(
                    np.divide(
                        np.sqrt(background_sumw2), np.abs(background),
                        out=np.full_like(background, np.inf), where=background != 0.0,
                    ) > 0.30
                )
            ),
            "diagnostic_comparison": (
                "score30 versus original raw 34-bin and inclusive no-nuisance Asimov Z; "
                "not an expected-limit claim"
            ),
            "signals": mass_diagnostics,
        },
        "artifacts": {
            "scores": "test_scores.root",
            "templates": "lowdm_30bin_test_templates.root",
        },
    }
    write_json(opts.output / "test_summary.json", result)
    selection["test_set_touched"] = True
    selection["test_opened_at"] = time.time()
    selection["test_summary"] = str(opts.output / "test_summary.json")
    write_json(selection_path, selection)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune a mass-agnostic multi-signal GNN with a locked 2:1:7 split."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--xsec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("audit", "tune", "test", "all"), default="all"
    )
    parser.add_argument("--trials", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-jets", type=int, default=10)
    parser.add_argument("--delta-m-min", type=int, default=150)
    parser.add_argument("--delta-m-max", type=int, default=300)
    parser.add_argument("--mstop-min", type=int, default=600)
    parser.add_argument("--mstop-max", type=int, default=1800)
    parser.add_argument(
        "--signal-topologies",
        nargs="+",
        choices=tuple(TOPOLOGY_IDS),
        default=tuple(TOPOLOGY_IDS),
    )
    parser.add_argument(
        "--top-targeted-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use ISR, b-MT, jet-MET angles, soft-b and AK8 kinematics; no top-tag "
            "score or pass/fail decision is used."
        ),
    )
    parser.add_argument("--max-tuning-hours", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "mps" if opts.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if opts.device == "auto" else opts.device
    )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    events, cache_state, split = load_inputs(opts)
    # Use the explicitly supplied immutable copy even when cache state points to
    # the remote production path.
    cache_state["manifest"] = str(opts.campaign_manifest)
    xsec_payload = json.loads(opts.xsec.read_text())
    weights = physics_weights(events, cache_state, xsec_payload)
    campaign_manifest = json.loads(opts.campaign_manifest.read_text())
    process_names = background_process_names(events, campaign_manifest)
    audit = {
        "schema_version": "gnn_lowdm_split_audit_v1",
        "status": "complete",
        "policy": "event_hash modulo 10: 0-1 train, 2 validation, 3-9 test",
        "requested_ratio": {"train": 0.2, "validation": 0.1, "test": 0.7},
        "mass_features_in_classifier": False,
        "top_tagger_features_in_classifier": False,
        "top_targeted_features_enabled": opts.top_targeted_features,
        "global_feature_count": int(events.global_features.shape[1]),
        "global_features": list(BASE_GLOBAL_FEATURE_NAMES)
        + (
            list(TOP_TARGETED_GLOBAL_FEATURE_NAMES)
            if opts.top_targeted_features
            else []
        ),
        "signal_policy": {
            "topologies": list(opts.signal_topologies),
            "delta_m_range": [opts.delta_m_min, opts.delta_m_max],
            "mstop_range": [opts.mstop_min, opts.mstop_max],
            "per_mass_training_loss": (
                "equal total signal weight per topology, then equal weight per mass "
                "point within each topology"
            ),
            "top_background_loss": (
                "TT/ST multiplier is tuned; total signal/background class weights "
                "remain balanced"
            ),
        },
        "splits": split_audit(events, split),
        "background_by_process": background_process_audit(
            events, split, weights, campaign_manifest
        ),
        "events": len(events),
        "cache_campaign": cache_state,
    }
    write_json(opts.output / "split_audit.json", audit)
    if opts.stage == "audit":
        print(
            json.dumps(
                {"stage": "audit", "status": "complete", "events": len(events)},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if opts.stage in ("tune", "all"):
        selection = tuning_stage(
            opts, events, split, weights, process_names, device
        )
        print(json.dumps({"stage": "tune", "status": selection["status"], "best": selection["best"]["trial"]["name"]}), flush=True)
    if opts.stage in ("test", "all"):
        result = test_stage(
            opts, events, split, weights, process_names, device
        )
        print(json.dumps({"stage": "test", "status": result["status"], "metrics": result["metrics"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
