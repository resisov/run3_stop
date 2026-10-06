"""Measurement functions migrated verbatim; no external analysis imports."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
HIGH_GROUPS = LOW_GROUPS = ("Nb1", "Nb2plus")

def rebin(values: np.ndarray, source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Sum source bins into target bins with exactly aligned boundaries."""
    if np.array_equal(source_edges, target_edges):
        return np.asarray(values, dtype=float).copy()
    if not set(target_edges).issubset(set(source_edges)):
        raise ValueError(f"target edges {target_edges} are not aligned with {source_edges}")
    output = np.zeros(len(target_edges) - 1, dtype=float)
    for index, (low, high) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        mask = (source_edges[:-1] >= low) & (source_edges[1:] <= high)
        output[index] = float(np.sum(values[mask]))
    return output


def ratio(
    data: np.ndarray,
    data_variance: np.ndarray,
    other: np.ndarray,
    other_variance: np.ndarray,
    target: np.ndarray,
    target_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = data - other
    residual_variance = data_variance + other_variance
    value = np.full_like(residual, np.nan)
    variance = np.full_like(residual, np.nan)
    valid = (target > 0.0) & (residual >= 0.0)
    value[valid] = residual[valid] / target[valid]
    variance[valid] = (
        residual_variance[valid] / target[valid] ** 2
        + residual[valid] ** 2 * target_variance[valid] / target[valid] ** 4
    )
    return value, np.sqrt(np.maximum(variance, 0.0)), residual, residual_variance


def normalized_shape(result: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    normalization_value, normalization_stat, _, _ = ratio(
        np.asarray([np.sum(result["data"])]),
        np.asarray([np.sum(result["data_variance"])]),
        np.asarray([np.sum(result["other"])]),
        np.asarray([np.sum(result["other_variance"])]),
        np.asarray([np.sum(result["target"])]),
        np.asarray([np.sum(result["target_variance"])]),
    )
    norm = float(normalization_value[0])
    norm_stat = float(normalization_stat[0])
    residual = np.asarray(result["residual"], dtype=float)
    residual_variance = np.asarray(result["residual_variance"], dtype=float)
    target = np.asarray(result["target"], dtype=float)
    target_variance = np.asarray(result["target_variance"], dtype=float)
    residual_total = float(np.sum(residual))
    residual_variance_total = float(np.sum(residual_variance))
    target_total = float(np.sum(target))
    target_variance_total = float(np.sum(target_variance))

    value = np.full_like(residual, np.nan)
    stat = np.full_like(residual, np.nan)
    valid = (
        (residual > 0.0)
        & (target > 0.0)
        & np.isfinite(result["value"])
        & (residual_total > 0.0)
        & (target_total > 0.0)
        & np.isfinite(norm)
        & (norm > 0.0)
    )
    value[valid] = result["value"][valid] / norm

    # The inclusive normalization contains the bin being normalized.  Keep
    # that covariance instead of adding the inclusive uncertainty as if it
    # came from an independent sample.  For S_i=(A_i/B_i)/(A_tot/B_tot),
    # propagate the derivatives with respect to every independent bin of A
    # and B.  This is the Run-2 normalized-shape comparison with a correct
    # first-order statistical uncertainty.
    residual_term = (
        (1.0 / residual[valid] - 1.0 / residual_total) ** 2
        * residual_variance[valid]
        + np.maximum(residual_variance_total - residual_variance[valid], 0.0)
        / residual_total**2
    )
    target_term = (
        (-1.0 / target[valid] + 1.0 / target_total) ** 2
        * target_variance[valid]
        + np.maximum(target_variance_total - target_variance[valid], 0.0)
        / target_total**2
    )
    stat[valid] = np.abs(value[valid]) * np.sqrt(
        np.maximum(residual_term + target_term, 0.0)
    )
    return {
        "value": value,
        "stat": stat,
        "normalization": norm,
        "normalization_stat": norm_stat,
    }


def records(dy: dict[str, np.ndarray], photon: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if not np.array_equal(dy["edges"], photon["edges"]):
        raise ValueError("DY and photon recoil edges differ")
    dy_shape = normalized_shape(dy)
    photon_shape = normalized_shape(photon)
    value = dy_shape["value"] / photon_shape["value"]
    stat = np.abs(value) * np.sqrt(
        (dy_shape["stat"] / dy_shape["value"]) ** 2
        + (photon_shape["stat"] / photon_shape["value"]) ** 2
    )
    systematic = np.maximum(np.abs(value - 1.0), stat)
    output = []
    for index in range(len(value)):
        output.append(
            {
                "low": float(dy["edges"][index]),
                "high": float(dy["edges"][index + 1]),
                "z_data_over_mc_raw": float(dy["value"][index]),
                "photon_data_over_mc_raw": float(photon["value"][index]),
                "z_data_over_mc": float(dy_shape["value"][index]),
                "z_stat": float(dy_shape["stat"][index]),
                "photon_data_over_mc": float(photon_shape["value"][index]),
                "photon_stat": float(photon_shape["stat"][index]),
                "z_normalization": float(dy_shape["normalization"]),
                "z_normalization_stat": float(dy_shape["normalization_stat"]),
                "photon_normalization": float(photon_shape["normalization"]),
                "photon_normalization_stat": float(photon_shape["normalization_stat"]),
                "double_ratio": float(value[index]),
                "double_ratio_stat": float(stat[index]),
                "systematic": float(systematic[index]),
                "downstream_central_abs_deviation": float(abs(value[index] - 1.0)),
                "status": "complete" if np.isfinite(value[index]) else "unavailable",
            }
        )
    return output

