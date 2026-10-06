"""Measurement functions migrated verbatim; no external analysis imports."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
HIGH_GROUPS = LOW_GROUPS = ("Nb1", "Nb2plus")

def factor(
    numerator: float,
    numerator_variance: float,
    denominator: float,
    denominator_variance: float,
) -> dict[str, Any]:
    if denominator <= 0.0 or numerator < 0.0:
        return {
            "status": "unavailable",
            "value": None,
            "stat": None,
            "numerator": numerator,
            "denominator": denominator,
        }
    value = numerator / denominator
    variance = (
        numerator_variance / denominator**2
        + numerator**2 * denominator_variance / denominator**4
    )
    return {
        "status": "complete",
        "value": float(value),
        "stat": float(math.sqrt(max(variance, 0.0))),
        "numerator": float(numerator),
        "numerator_variance": float(numerator_variance),
        "denominator": float(denominator),
        "denominator_variance": float(denominator_variance),
    }


def leaf_array(
    payload: dict[str, Any], nbin: int
) -> tuple[np.ndarray, np.ndarray]:
    nominal = (payload or {}).get("nominal") or {}
    values = np.asarray(nominal.get("sumw") or [0.0] * nbin, dtype=float)
    variances = np.asarray(
        nominal.get("sumw2") or [0.0] * nbin, dtype=float
    )
    if len(values) != nbin or len(variances) != nbin:
        raise ValueError(
            f"expected {nbin} bins, got {len(values)}/{len(variances)}"
        )
    return values, variances


def sum_samples(
    by_sample: dict[str, Any],
    nbin: int,
    include: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(nbin, dtype=float)
    variances = np.zeros(nbin, dtype=float)
    for sample, payload in by_sample.items():
        if include is not None and sample not in include:
            continue
        current, current_variance = leaf_array(payload, nbin)
        values += current
        variances += current_variance
    return values, variances


def data_leaf(payload: dict[str, Any]) -> tuple[float, float]:
    return (
        float((payload or {}).get("sumw", 0.0)),
        float((payload or {}).get("sumw2", 0.0)),
    )


def build_q_sgamma(
    measurement: dict[str, Any], exact: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "highdm": {},
        "lowdm": {},
        "lowdm_Q_groups": {},
    }
    high_nbin = len(exact["highdm"]["recoil_edges"]) - 1
    high_data = measurement["gcr_data"]["highdm"]["yields"]
    for group in HIGH_GROUPS:
        by_sample = exact["highdm"]["recoil"]["GCR"][group]
        total_mc, total_mc2 = sum_samples(by_sample, high_nbin)
        gamma, gamma2 = sum_samples(by_sample, high_nbin, {"GJ"})
        other = total_mc - gamma
        other2 = np.maximum(total_mc2 - gamma2, 0.0)
        data = np.asarray(
            [
                data_leaf((high_data.get(group) or {}).get(str(index), {}))[0]
                for index in range(high_nbin)
            ],
            dtype=float,
        )
        data2 = np.asarray(
            [
                data_leaf((high_data.get(group) or {}).get(str(index), {}))[1]
                for index in range(high_nbin)
            ],
            dtype=float,
        )
        q = factor(
            float(np.sum(data - other)),
            float(np.sum(data2 + other2)),
            float(np.sum(gamma)),
            float(np.sum(gamma2)),
        )
        bins = []
        for index in range(high_nbin):
            denominator = (
                float(q["value"]) * float(gamma[index])
                if q["status"] == "complete"
                else 0.0
            )
            denominator_variance = 0.0
            if q["status"] == "complete":
                denominator_variance = (
                    gamma[index] ** 2 * q["stat"] ** 2
                    + q["value"] ** 2 * gamma2[index]
                )
            bins.append(
                {
                    "index": index,
                    "data": float(data[index]),
                    "data_variance": float(data2[index]),
                    "gamma_mc": float(gamma[index]),
                    "gamma_mc_variance": float(gamma2[index]),
                    "other_mc": float(other[index]),
                    "other_mc_variance": float(other2[index]),
                    "Sgamma": factor(
                        float(data[index] - other[index]),
                        float(data2[index] + other2[index]),
                        denominator,
                        denominator_variance,
                    ),
                }
            )
        output["highdm"][group] = {"Q": q, "bins": bins}

    low_measurement = measurement["gcr_data"]["lowdm"]
    if "yields_by_group" in low_measurement:
        low_nbin = len(exact["lowdm"]["recoil_edges"]) - 1
        for group in LOW_GROUPS:
            by_sample = exact["lowdm"]["recoil"]["GCR"][group]
            total_mc, total_mc2 = sum_samples(by_sample, low_nbin)
            gamma, gamma2 = sum_samples(by_sample, low_nbin, {"GJ"})
            other = total_mc - gamma
            other2 = np.maximum(total_mc2 - gamma2, 0.0)
            low_data = low_measurement["yields_by_group"][group]
            data = np.asarray(
                [
                    data_leaf((low_data.get(str(index)) or {}))[0]
                    for index in range(low_nbin)
                ],
                dtype=float,
            )
            data2 = np.asarray(
                [
                    data_leaf((low_data.get(str(index)) or {}))[1]
                    for index in range(low_nbin)
                ],
                dtype=float,
            )
            q = factor(
                float(np.sum(data - other)),
                float(np.sum(data2 + other2)),
                float(np.sum(gamma)),
                float(np.sum(gamma2)),
            )
            bins = []
            for index in range(low_nbin):
                denominator = (
                    float(q["value"]) * float(gamma[index])
                    if q["status"] == "complete"
                    else 0.0
                )
                denominator_variance = 0.0
                if q["status"] == "complete":
                    denominator_variance = (
                        gamma[index] ** 2 * q["stat"] ** 2
                        + q["value"] ** 2 * gamma2[index]
                    )
                bins.append(
                    {
                        "index": index,
                        "data": float(data[index]),
                        "data_variance": float(data2[index]),
                        "gamma_mc": float(gamma[index]),
                        "gamma_mc_variance": float(gamma2[index]),
                        "other_mc": float(other[index]),
                        "other_mc_variance": float(other2[index]),
                        "Sgamma": factor(
                            float(data[index] - other[index]),
                            float(data2[index] + other2[index]),
                            denominator,
                            denominator_variance,
                        ),
                    }
                )
            output["lowdm_Q_groups"][group] = q
            output["lowdm"][group] = {
                "group": group,
                "Q": q,
                "bins": bins,
            }
        return output

    raise ValueError(
        "retired Low-dM search-bin input: expected Nb1/Nb2plus recoil histograms"
    )

