"""Differentiable and validation-level S/sqrt(B) utilities."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import numpy as np
import torch

from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, GraphEvents


CATEGORY_NAMES = (
    "Nb1_NISR0",
    "Nb1_NISR1plus",
    "Nb2plus_NISR0",
    "Nb2plus_NISR1plus",
)


def soft_s_over_sqrt_b_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    category_ids: torch.Tensor | None = None,
    thresholds: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5, 2.5),
    temperature: float = 0.35,
    background_floor_fraction: float = 2.0e-3,
) -> torch.Tensor:
    """Negative mean log S/sqrt(B) over differentiable logit thresholds.

    ``weights`` must be non-negative training weights.  A small floor tied to
    the total background weight prevents a one-event batch fluctuation from
    producing a divergent gradient.  The BCE term used by the trainer fixes
    score calibration; this term targets ordering in the search tail.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = []
    groups = (
        (torch.ones_like(labels, dtype=torch.bool),)
        if category_ids is None
        else tuple(category_ids == value for value in torch.unique(category_ids))
    )
    for group in groups:
        signal = (labels > 0.5) & group
        background = (labels <= 0.5) & group
        if not torch.any(signal) or not torch.any(background):
            continue
        signal_weight = weights * signal.to(weights.dtype)
        background_weight = weights * background.to(weights.dtype)
        background_total = background_weight.sum()
        floor = background_floor_fraction * background_total + torch.finfo(
            logits.dtype
        ).eps
        for threshold in thresholds:
            accepted = torch.sigmoid((logits - threshold) / temperature)
            signal_yield = torch.sum(signal_weight * accepted)
            background_yield = torch.sum(background_weight * accepted)
            significance = signal_yield / torch.sqrt(background_yield + floor)
            values.append(torch.log(significance + 1.0e-6))
    if not values:
        return logits.sum() * 0.0
    return -torch.stack(values).mean()


def diagonal_v3_category_ids(events: GraphEvents, indices: np.ndarray) -> np.ndarray:
    names = list(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)
    nb = np.rint(
        events.global_features[indices, names.index("nb_medium")] * 5.0
    ).astype(np.int16)
    nisr = np.rint(
        events.global_features[indices, names.index("n_lowdm_isr")] * 4.0
    ).astype(np.int16)
    return ((nb >= 2).astype(np.int8) * 2 + (nisr >= 1).astype(np.int8))


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probabilities = np.asarray(tuple(probabilities), dtype=float)
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        raise ValueError("weighted quantiles require positive total weight")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    centers /= sorted_weights.sum()
    return np.interp(probabilities, centers, sorted_values)


def _histogram(
    scores: np.ndarray,
    categories: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bins = len(edges) - 1
    output = np.zeros((len(CATEGORY_NAMES), bins), dtype=np.float64)
    sumw2 = np.zeros_like(output)
    score_bin = np.searchsorted(edges, scores, side="right") - 1
    score_bin = np.clip(score_bin, 0, bins - 1)
    flat = categories.astype(np.int64) * bins + score_bin
    output.flat[:] = np.bincount(
        flat, weights=weights, minlength=output.size
    )[: output.size]
    sumw2.flat[:] = np.bincount(
        flat, weights=np.square(weights), minlength=output.size
    )[: output.size]
    return output, sumw2


def evaluate_binning(
    events: GraphEvents,
    indices: np.ndarray,
    scores: np.ndarray,
    physics_weights: np.ndarray,
    edges: Iterable[float],
    *,
    subset_scale: float,
    min_background_neff: float,
    max_relative_mc_stat: float,
) -> dict[str, Any] | None:
    indices = np.asarray(indices, dtype=np.int64)
    edges_array = np.asarray(tuple(edges), dtype=float)
    labels = events.labels[indices].astype(bool)
    categories = diagonal_v3_category_ids(events, indices)
    local_weights = np.asarray(physics_weights[indices], dtype=np.float64)
    background = ~labels
    background_yield, background_sumw2 = _histogram(
        scores[background],
        categories[background],
        local_weights[background] * subset_scale,
        edges_array,
    )
    # Scaling a subset to the full luminosity scales sum(w^2) quadratically.
    _, raw_background_sumw2 = _histogram(
        scores[background],
        categories[background],
        local_weights[background],
        edges_array,
    )
    background_sumw2 = raw_background_sumw2 * subset_scale**2
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = np.square(background_yield) / background_sumw2
        relative_mc_stat = np.sqrt(background_sumw2) / background_yield
    if (
        np.any(background_yield <= 0.0)
        or np.any(~np.isfinite(neff))
        or np.any(neff < min_background_neff)
        or np.any(relative_mc_stat > max_relative_mc_stat)
    ):
        return None

    point_rows = np.unique(
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
    point_metrics: dict[str, dict[str, float]] = {}
    z_values = []
    gains = []
    total_background = float(background_yield.sum())
    for topology, mstop, mlsp in point_rows:
        point = (
            labels
            & (events.signal_topology_id[indices] == topology)
            & (events.mstop[indices] == mstop)
            & (events.mlsp[indices] == mlsp)
        )
        signal_yield, _ = _histogram(
            scores[point],
            categories[point],
            local_weights[point] * subset_scale,
            edges_array,
        )
        positive_signal = np.clip(signal_yield, 0.0, None)
        z_binned = float(
            np.sqrt(np.sum(np.square(positive_signal) / background_yield))
        )
        total_signal = float(signal_yield.sum())
        z_inclusive = (
            total_signal / math.sqrt(total_background)
            if total_signal > 0.0 and total_background > 0.0
            else 0.0
        )
        gain = z_binned / z_inclusive if z_inclusive > 0.0 else 0.0
        key = f"{int(topology)}_{int(mstop)}_{int(mlsp)}"
        point_metrics[key] = {
            "s_over_sqrt_b_binned": z_binned,
            "s_over_sqrt_b_inclusive": z_inclusive,
            "gain": gain,
            "signal_yield": total_signal,
        }
        z_values.append(z_binned)
        gains.append(gain)
    z_array = np.asarray(z_values, dtype=float)
    gain_array = np.asarray(gains, dtype=float)
    return {
        "edges": edges_array.tolist(),
        "score_bins_per_category": len(edges_array) - 1,
        "total_bins": len(CATEGORY_NAMES) * (len(edges_array) - 1),
        "background_yield": background_yield.tolist(),
        "background_neff": neff.tolist(),
        "background_relative_mc_stat": relative_mc_stat.tolist(),
        "minimum_background_neff": float(np.min(neff)),
        "maximum_relative_mc_stat": float(np.max(relative_mc_stat)),
        "median_s_over_sqrt_b": float(np.median(z_array)),
        "p10_s_over_sqrt_b": float(np.quantile(z_array, 0.10)),
        "median_gain": float(np.median(gain_array)),
        "p10_gain": float(np.quantile(gain_array, 0.10)),
        "minimum_gain": float(np.min(gain_array)),
        "per_mass_point": point_metrics,
    }


def binning_selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(result["p10_gain"]),
        float(result["median_gain"]),
        float(result["p10_s_over_sqrt_b"]),
        float(result["median_s_over_sqrt_b"]),
        float(result["minimum_background_neff"]),
        -float(result["total_bins"]),
    )


def optimize_validation_binning(
    events: GraphEvents,
    indices: np.ndarray,
    scores: np.ndarray,
    physics_weights: np.ndarray,
    *,
    subset_scale: float = 10.0,
    candidate_percentiles: tuple[float, ...] = (
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        0.97,
        0.98,
        0.99,
        0.995,
    ),
    score_bins: tuple[int, ...] = (3, 4, 5),
    min_background_neff: float = 8.0,
    max_relative_mc_stat: float = 0.50,
) -> dict[str, Any]:
    labels = events.labels[indices].astype(bool)
    background_scores = scores[~labels]
    background_weights = np.abs(physics_weights[indices][~labels])
    thresholds = np.unique(
        weighted_quantiles(
            background_scores, background_weights, candidate_percentiles
        )
    )
    evaluated = 0
    viable = []
    for bins in score_bins:
        for internal in itertools.combinations(thresholds, bins - 1):
            edges = np.asarray((0.0, *internal, 1.0), dtype=float)
            if np.any(np.diff(edges) <= 1.0e-6):
                continue
            evaluated += 1
            result = evaluate_binning(
                events,
                indices,
                scores,
                physics_weights,
                edges,
                subset_scale=subset_scale,
                min_background_neff=min_background_neff,
                max_relative_mc_stat=max_relative_mc_stat,
            )
            if result is not None:
                viable.append(result)
    if not viable:
        raise RuntimeError("no validation binning satisfies the MC-stat constraints")
    ranked = sorted(viable, key=binning_selection_key, reverse=True)
    return {
        "status": "complete",
        "objective": (
            "maximize p10 then median mass-point gain in combined binned "
            "S/sqrt(B), subject to positive background and MC-stat constraints"
        ),
        "candidate_percentiles": list(candidate_percentiles),
        "candidate_absolute_thresholds": thresholds.tolist(),
        "score_bin_counts": list(score_bins),
        "candidates_evaluated": evaluated,
        "candidates_viable": len(viable),
        "best": ranked[0],
        "top_candidates": ranked[:20],
    }
