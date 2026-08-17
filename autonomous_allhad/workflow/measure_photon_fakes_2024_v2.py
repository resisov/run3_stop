#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from autonomous_allhad.photon_fake_2024_worker import (
    GCR_REGIONS,
    OUTPUT_SCHEMA,
    TRANSFER_PT_EDGES,
    load_histogram_builder,
)
from measure_photon_fakes_2024 import (
    ELECTRON_NORMALIZATION_UNCERTAINTY,
    PROMPT_NORMALIZATION_UNCERTAINTY,
    _collapse_stratified,
    _distribution_record,
    _empty_leaf,
    _leaf_integral,
    _merge_dataset_record,
    _origin_record,
    _sum_stratified,
    aggregate_component,
    data_channels_from_events,
    deduplicate_data_events,
    qcd_target_origin_histograms,
    read_payload,
    write_payload,
)


MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v2"
ABCD_PROBES = {
    "A": "target",
    "B": "application",
    "C": "measurement_pass",
    "D": "measurement_fail",
}
PLJ_PROBES = ("application", "measurement_pass", "plj_other")
KAPPA_SOURCE_REGION = "GCR_DPhiVR_High"
KAPPA_VALIDATION_REGION = "GCR_DPhiVR_Low"
MIN_COMPONENT_YIELD = 5.0
MAX_RELATIVE_KAPPA_UNCERTAINTY = 0.75
MAX_FACTOR = 5.0
MIN_CLOSURE_UNCERTAINTY = 0.20


def _region_transfer_record(
    channels: dict[str, Any],
    probe: str,
    origin: str,
    region: str,
) -> dict[str, Any] | None:
    record = _origin_record(channels, probe, origin)
    if record is None:
        return None
    regional = (record.get("region_transfers") or {}).get(region)
    if regional is not None:
        return regional
    if region == "GCR":
        return record.get("transfer")
    return None


def _stratum_yields(
    channels: dict[str, Any],
    probe: str,
    origin: str,
    region: str,
) -> tuple[np.ndarray, np.ndarray]:
    count = 2 * (len(TRANSFER_PT_EDGES) - 1)
    record = _region_transfer_record(channels, probe, origin, region)
    if record is None:
        return np.zeros(count), np.zeros(count)
    return (
        np.asarray([leaf["sumw"][0] for leaf in record["strata"]], dtype=float),
        np.asarray([leaf["sumw2"][0] for leaf in record["strata"]], dtype=float),
    )


def _residual_yields(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    probe: str,
    region: str,
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data, data_var = _stratum_yields(
        data_channels, probe, "all", region
    )
    prompt, prompt_var = _stratum_yields(
        prompt_channels, probe, "prompt", region
    )
    electron, electron_var = _stratum_yields(
        electron_channels, probe, "electron", region
    )
    residual = data - prompt_scale * prompt - electron_scale * electron
    variance = (
        data_var
        + prompt_scale * prompt_scale * prompt_var
        + electron_scale * electron_scale * electron_var
    )
    return residual, variance, {
        "data": data,
        "data_variance": data_var,
        "prompt": prompt,
        "prompt_variance": prompt_var,
        "electron": electron,
        "electron_variance": electron_var,
    }


def _coarse_groups() -> list[dict[str, Any]]:
    pt_bins = len(TRANSFER_PT_EDGES) - 1
    groups: list[dict[str, Any]] = []
    for eta_index, eta in enumerate(("EB", "EE")):
        offset = eta_index * pt_bins
        groups.extend(
            [
                {
                    "label": f"{eta}_pt220to400",
                    "indices": [offset, offset + 1],
                },
                {
                    "label": f"{eta}_pt400toinf",
                    "indices": [offset + 2, offset + 3],
                },
            ]
        )
    return groups


def _pool(values: dict[str, np.ndarray], indices: list[int]) -> dict[str, float]:
    return {
        key: float(np.sum(np.asarray(array, dtype=float)[indices]))
        for key, array in values.items()
    }


def _kappa_ratio(record: dict[str, float]) -> dict[str, Any]:
    a, b, c, d = (record[name] for name in ("A", "B", "C", "D"))
    valid = (
        a > 0.0
        and b >= MIN_COMPONENT_YIELD
        and c >= MIN_COMPONENT_YIELD
        and d >= MIN_COMPONENT_YIELD
    )
    if not valid:
        return {
            "valid": False,
            "value": None,
            "variance": None,
            "relative_uncertainty": None,
        }
    value = a * d / (b * c)
    relative_variance = sum(
        record[f"{name}_variance"] / (record[name] * record[name])
        for name in ("A", "B", "C", "D")
    )
    variance = value * value * max(0.0, relative_variance)
    relative_uncertainty = (
        math.sqrt(variance) / value if value > 0.0 else math.inf
    )
    return {
        "valid": bool(np.isfinite(value) and np.isfinite(variance)),
        "value": float(value),
        "variance": float(variance),
        "relative_uncertainty": float(relative_uncertainty),
    }


def fit_kappa(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    region: str,
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> dict[str, Any]:
    components: dict[str, np.ndarray] = {}
    component_audit: dict[str, Any] = {}
    for label, probe in ABCD_PROBES.items():
        value, variance, audit = _residual_yields(
            data_channels,
            prompt_channels,
            electron_channels,
            probe,
            region,
            prompt_scale,
            electron_scale,
        )
        components[label] = value
        components[f"{label}_variance"] = variance
        component_audit[label] = {
            key: array.tolist() for key, array in audit.items()
        }

    pt_bins = len(TRANSFER_PT_EDGES) - 1
    all_indices = list(range(2 * pt_bins))
    eta_indices = {
        eta_index: list(
            range(eta_index * pt_bins, (eta_index + 1) * pt_bins)
        )
        for eta_index in range(2)
    }
    candidate_groups: dict[str, dict[str, Any]] = {}
    for group in _coarse_groups():
        pooled = _pool(components, group["indices"])
        candidate_groups[group["label"]] = {
            **pooled,
            **_kappa_ratio(pooled),
            "indices": group["indices"],
        }
    eta_candidates = {}
    for eta_index, indices in eta_indices.items():
        pooled = _pool(components, indices)
        eta_candidates[eta_index] = {
            **pooled,
            **_kappa_ratio(pooled),
            "indices": indices,
        }
    global_pooled = _pool(components, all_indices)
    global_candidate = {
        **global_pooled,
        **_kappa_ratio(global_pooled),
        "indices": all_indices,
    }

    values = np.ones(2 * pt_bins, dtype=float)
    variances = np.ones(2 * pt_bins, dtype=float)
    sources: list[str] = []
    covariance_keys: list[str] = []
    records: list[dict[str, Any]] = []
    labels = []
    for eta in ("EB", "EE"):
        for low, high in zip(TRANSFER_PT_EDGES[:-1], TRANSFER_PT_EDGES[1:]):
            labels.append(
                f"{eta}_pt{low:g}to"
                + ("inf" if high >= 1_000_000.0 else f"{high:g}")
            )
    coarse_lookup = {}
    for group in _coarse_groups():
        for index in group["indices"]:
            coarse_lookup[index] = group["label"]

    def acceptable(candidate: dict[str, Any]) -> bool:
        return bool(
            candidate.get("valid")
            and float(candidate["relative_uncertainty"])
            <= MAX_RELATIVE_KAPPA_UNCERTAINTY
        )

    for index, label in enumerate(labels):
        coarse_label = coarse_lookup[index]
        direct = candidate_groups[coarse_label]
        eta = eta_candidates[index // pt_bins]
        if acceptable(direct):
            chosen = direct
            source = f"direct:{coarse_label}"
            covariance_key = source
        elif acceptable(eta):
            chosen = eta
            source = "eta_inclusive_fallback"
            covariance_key = f"{source}:{index // pt_bins}"
        elif acceptable(global_candidate):
            chosen = global_candidate
            source = "global_fallback"
            covariance_key = source
        else:
            candidates = (
                (direct, f"direct:{coarse_label}"),
                (eta, f"eta:{index // pt_bins}"),
                (global_candidate, "global"),
            )
            selected = next(
                (
                    (candidate, key)
                    for candidate, key in candidates
                    if candidate.get("valid")
                ),
                None,
            )
            available = selected[0] if selected is not None else None
            if available is None:
                chosen = {"value": 1.0, "variance": 1.0}
                source = "unit_unconstrained"
                covariance_key = f"{source}:{coarse_label}"
            else:
                # Do not move the central value with an imprecise validation
                # ratio.  Retain its distance from unity and its statistical
                # uncertainty as a conservative kappa uncertainty.
                uncertainty = max(
                    0.50,
                    abs(float(available["value"]) - 1.0),
                    math.sqrt(float(available["variance"])),
                )
                chosen = {"value": 1.0, "variance": uncertainty * uncertainty}
                source = "unit_regularized_imprecise_validation"
                covariance_key = f"{source}:{selected[1]}"
        values[index] = float(chosen["value"])
        variances[index] = float(chosen["variance"])
        sources.append(source)
        covariance_keys.append(covariance_key)
        records.append(
            {
                "index": index,
                "label": label,
                "coarse_group": coarse_label,
                "factor": float(values[index]),
                "factor_uncertainty": math.sqrt(float(variances[index])),
                "source": source,
                "covariance_key": covariance_key,
            }
        )
    covariance = np.zeros((len(values), len(values)))
    for first in range(len(values)):
        for second in range(len(values)):
            if covariance_keys[first] == covariance_keys[second]:
                covariance[first, second] = math.sqrt(
                    max(0.0, variances[first] * variances[second])
                )
    return {
        "region": region,
        "factors": values,
        "variances": variances,
        "covariance": covariance,
        "records": records,
        "coarse_candidates": candidate_groups,
        "eta_candidates": eta_candidates,
        "global_candidate": global_candidate,
        "component_audit": component_audit,
        "prompt_scale": prompt_scale,
        "electron_scale": electron_scale,
        "policy": (
            "derive kappa in EB/EE x coarse photon-pT groups; accept only "
            f"relative uncertainty <= {MAX_RELATIVE_KAPPA_UNCERTAINTY:g}; "
            "otherwise fall back to eta/global or retain kappa=1 with the "
            "imprecise validation result assigned as uncertainty"
        ),
    }


def fit_plj_factors(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    region: str,
    kappa: np.ndarray,
    kappa_variance: np.ndarray,
    kappa_covariance: np.ndarray | None = None,
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> dict[str, Any]:
    if kappa_covariance is None:
        kappa_covariance = np.diag(kappa_variance)
    else:
        kappa_covariance = np.asarray(kappa_covariance, dtype=float)
    residual: dict[str, np.ndarray] = {}
    variance: dict[str, np.ndarray] = {}
    for label, probe in {
        "B": "application",
        "C": "measurement_pass",
        "D": "measurement_fail",
        "O": "plj_other",
    }.items():
        residual[label], variance[label], _ = _residual_yields(
            data_channels,
            prompt_channels,
            electron_channels,
            probe,
            region,
            prompt_scale,
            electron_scale,
        )
    residual["L"] = residual["B"] + residual["C"] + residual["O"]
    variance["L"] = variance["B"] + variance["C"] + variance["O"]
    count = len(kappa)

    def estimate(
        indices: list[int],
        kappa_coefficients: np.ndarray,
    ) -> dict[str, Any]:
        kappa_value = float(np.dot(kappa_coefficients, kappa))
        kappa_var = float(
            kappa_coefficients @ kappa_covariance @ kappa_coefficients
        )
        pooled = {
            name: float(np.sum(values[indices]))
            for name, values in residual.items()
        }
        pooled_var = {
            name: float(np.sum(values[indices]))
            for name, values in variance.items()
        }
        b, c, d, o, l = (
            pooled[name] for name in ("B", "C", "D", "O", "L")
        )
        valid = (
            b > 0.0
            and c > 0.0
            and d >= MIN_COMPONENT_YIELD
            and l >= MIN_COMPONENT_YIELD
        )
        if not valid:
            return {
                "valid": False,
                "factor": 0.0,
                "variance": 0.0,
                **pooled,
                **{f"{key}_variance": value for key, value in pooled_var.items()},
            }
        base = b * c / (d * l)
        base_relative_variance = (
            pooled_var["B"] * (1.0 / b - 1.0 / l) ** 2
            + pooled_var["C"] * (1.0 / c - 1.0 / l) ** 2
            + pooled_var["D"] / (d * d)
            + pooled_var["O"] / (l * l)
        )
        base_var = base * base * max(0.0, base_relative_variance)
        factor = kappa_value * base
        factor_var = base * base * kappa_var + kappa_value * kappa_value * base_var
        return {
            "valid": bool(np.isfinite(factor) and np.isfinite(factor_var)),
            "factor": min(MAX_FACTOR, max(0.0, float(factor))),
            "variance": max(0.0, float(factor_var)),
            "base": float(base),
            "base_variance": float(base_var),
            "kappa_value": float(kappa_value),
            "kappa_variance": max(0.0, float(kappa_var)),
            "kappa_coefficients": kappa_coefficients,
            **pooled,
            **{f"{key}_variance": value for key, value in pooled_var.items()},
        }

    pt_bins = len(TRANSFER_PT_EDGES) - 1
    factors = np.zeros(count)
    factor_variances = np.zeros(count)
    chosen_records: list[dict[str, Any]] = []
    base_covariance_keys: list[str] = []
    records = []
    all_indices = list(range(count))
    for index in range(count):
        direct_coefficients = np.zeros(count)
        direct_coefficients[index] = 1.0
        direct = estimate([index], direct_coefficients)
        eta_indices = list(
            range((index // pt_bins) * pt_bins, (index // pt_bins + 1) * pt_bins)
        )
        eta_coefficients = np.zeros(count)
        eta_coefficients[eta_indices] = 1.0 / len(eta_indices)
        eta = estimate(eta_indices, eta_coefficients)
        global_coefficients = np.full(count, 1.0 / count)
        global_fit = estimate(all_indices, global_coefficients)
        if direct["valid"]:
            chosen, source = direct, "direct"
            base_covariance_key = f"direct:{index}"
        elif eta["valid"]:
            chosen, source = eta, "eta_inclusive_fallback"
            base_covariance_key = f"eta:{index // pt_bins}"
        elif global_fit["valid"]:
            chosen, source = global_fit, "global_fallback"
            base_covariance_key = "global"
        else:
            chosen, source = {"factor": 0.0, "variance": 0.0}, "unmeasurable"
            base_covariance_key = f"unmeasurable:{index}"
        factors[index] = float(chosen["factor"])
        factor_variances[index] = float(chosen["variance"])
        chosen_records.append(chosen)
        base_covariance_keys.append(base_covariance_key)
        records.append(
            {
                "index": index,
                "factor": float(factors[index]),
                "factor_uncertainty": math.sqrt(float(factor_variances[index])),
                "kappa": float(kappa[index]),
                "kappa_uncertainty": math.sqrt(float(kappa_variance[index])),
                "source": source,
                "base_covariance_key": base_covariance_key,
                **{
                    key: value
                    for key, value in chosen.items()
                    if key
                    in {
                        "B",
                        "C",
                        "D",
                        "O",
                        "L",
                        "B_variance",
                        "C_variance",
                        "D_variance",
                        "O_variance",
                        "L_variance",
                    }
                },
            }
        )
    factor_covariance = np.zeros((count, count))
    for first in range(count):
        first_record = chosen_records[first]
        if "base" not in first_record:
            continue
        for second in range(count):
            second_record = chosen_records[second]
            if "base" not in second_record:
                continue
            first_coefficients = np.asarray(
                first_record["kappa_coefficients"], dtype=float
            )
            second_coefficients = np.asarray(
                second_record["kappa_coefficients"], dtype=float
            )
            shared_kappa = float(
                first_coefficients
                @ kappa_covariance
                @ second_coefficients
            )
            covariance = (
                float(first_record["base"])
                * float(second_record["base"])
                * shared_kappa
            )
            if base_covariance_keys[first] == base_covariance_keys[second]:
                shared_base = math.sqrt(
                    max(
                        0.0,
                        float(first_record["base_variance"])
                        * float(second_record["base_variance"]),
                    )
                )
                covariance += (
                    float(first_record["kappa_value"])
                    * float(second_record["kappa_value"])
                    * shared_base
                )
            factor_covariance[first, second] = covariance
    factor_variances = np.maximum(np.diag(factor_covariance), 0.0)
    for index, record in enumerate(records):
        record["factor_uncertainty"] = math.sqrt(float(factor_variances[index]))
    return {
        "region": region,
        "factors": factors,
        "variances": factor_variances,
        "covariance": factor_covariance,
        "records": records,
        "prompt_scale": prompt_scale,
        "electron_scale": electron_scale,
    }


def _residual_distribution(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    probes: tuple[str, ...],
    region: str,
    variable: str,
    edges: list[float],
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    bins = len(edges) - 1
    count = 2 * (len(TRANSFER_PT_EDGES) - 1)
    values = [np.zeros(bins) for _ in range(count)]
    variances = [np.zeros(bins) for _ in range(count)]
    entries = [np.zeros(bins, dtype=int) for _ in range(count)]
    for probe in probes:
        records = [
            _distribution_record(data_channels, probe, "all", region, variable),
            _distribution_record(
                prompt_channels, probe, "prompt", region, variable
            ),
            _distribution_record(
                electron_channels, probe, "electron", region, variable
            ),
        ]
        for stratum in range(count):
            arrays = []
            for record in records:
                if record is None:
                    arrays.append(
                        (
                            np.zeros(bins),
                            np.zeros(bins),
                            np.zeros(bins, dtype=int),
                        )
                    )
                else:
                    leaf = record["strata"][stratum]
                    arrays.append(
                        (
                            np.asarray(leaf["sumw"], dtype=float),
                            np.asarray(leaf["sumw2"], dtype=float),
                            np.asarray(leaf["entries"], dtype=int),
                        )
                    )
            data, data_var, data_entries = arrays[0]
            prompt, prompt_var, _ = arrays[1]
            electron, electron_var, _ = arrays[2]
            values[stratum] += (
                data - prompt_scale * prompt - electron_scale * electron
            )
            variances[stratum] += (
                data_var
                + prompt_scale * prompt_scale * prompt_var
                + electron_scale * electron_scale * electron_var
            )
            entries[stratum] += data_entries
    return values, variances, entries


def predict_distribution(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    region: str,
    variable: str,
    edges: list[float],
    factors: np.ndarray,
    factor_variances: np.ndarray,
    factor_covariance: np.ndarray | None = None,
    probes: tuple[str, ...] = PLJ_PROBES,
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if factor_covariance is None:
        factor_covariance = np.diag(factor_variances)
    else:
        factor_covariance = np.asarray(factor_covariance, dtype=float)
    values, variances, entries = _residual_distribution(
        data_channels,
        prompt_channels,
        electron_channels,
        probes,
        region,
        variable,
        edges,
        prompt_scale,
        electron_scale,
    )
    bins = len(edges) - 1
    prediction = np.zeros(bins)
    covariance = np.zeros((bins, bins))
    total_entries = np.zeros(bins, dtype=int)
    for stratum, (value, variance, entry) in enumerate(
        zip(values, variances, entries)
    ):
        factor = float(factors[stratum])
        prediction += factor * value
        covariance += np.diag(factor * factor * variance)
        total_entries += entry
    for first, first_value in enumerate(values):
        for second, second_value in enumerate(values):
            covariance += (
                np.outer(first_value, second_value)
                * float(factor_covariance[first, second])
            )
    negative_bins = [
        {"bin": int(index), "value": float(prediction[index])}
        for index in np.flatnonzero(prediction < 0.0)
    ]
    prediction = np.maximum(prediction, 0.0)
    output = {
        "bin_edges": [float(x) for x in edges],
        "sumw": prediction.tolist(),
        "sumw2": np.maximum(np.diag(covariance), 0.0).tolist(),
        "entries": total_entries.astype(int).tolist(),
    }
    return output, {
        "negative_bins_clipped": negative_bins,
        "integral": float(np.sum(prediction)),
        "integral_variance_with_correlations": float(np.sum(covariance)),
        "covariance": covariance.tolist(),
        "application_probes": list(probes),
    }


def _scaled(nominal: dict[str, Any], scale: float) -> dict[str, Any]:
    output = copy.deepcopy(nominal)
    output["sumw"] = [
        max(0.0, scale * float(value)) for value in nominal["sumw"]
    ]
    return output


def closure_summary(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    kappa_fit: dict[str, Any],
    builder: Any,
    region: str = KAPPA_VALIDATION_REGION,
) -> dict[str, Any]:
    factors = fit_plj_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        region,
        kappa_fit["factors"],
        kappa_fit["variances"],
        kappa_fit["covariance"],
    )
    prediction, audit = predict_distribution(
        data_channels,
        prompt_channels,
        electron_channels,
        region,
        "recoil",
        [float(x) for x in builder.RECOIL_PT_BINS],
        factors["factors"],
        factors["variances"],
        factors["covariance"],
    )
    target_record = _distribution_record(
        data_channels, "target", "all", region, "recoil"
    )
    prompt_record = _distribution_record(
        prompt_channels, "target", "prompt", region, "recoil"
    )
    electron_record = _distribution_record(
        electron_channels, "target", "electron", region, "recoil"
    )
    target, target_var = _sum_stratified(target_record)
    prompt, prompt_var = _sum_stratified(prompt_record)
    electron, electron_var = _sum_stratified(electron_record)
    recoil_edges = [float(x) for x in builder.RECOIL_PT_BINS]
    target_histogram = _collapse_stratified(target_record, recoil_edges)
    prompt_histogram = _collapse_stratified(prompt_record, recoil_edges)
    electron_histogram = _collapse_stratified(electron_record, recoil_edges)
    target_fake_histogram = {
        "bin_edges": recoil_edges,
        "sumw": (
            np.asarray(target_histogram["sumw"], dtype=float)
            - np.asarray(prompt_histogram["sumw"], dtype=float)
            - np.asarray(electron_histogram["sumw"], dtype=float)
        ).tolist(),
        "sumw2": (
            np.asarray(target_histogram["sumw2"], dtype=float)
            + np.asarray(prompt_histogram["sumw2"], dtype=float)
            + np.asarray(electron_histogram["sumw2"], dtype=float)
        ).tolist(),
        "entries": [int(value) for value in target_histogram["entries"]],
    }
    residual_target = target - prompt - electron
    residual_target_var = target_var + prompt_var + electron_var
    predicted, _ = _leaf_integral(prediction)
    predicted_var = float(audit["integral_variance_with_correlations"])
    if residual_target > 0.0 and predicted > 0.0:
        ratio = residual_target / predicted
        ratio_variance = (
            residual_target_var / (predicted * predicted)
            + residual_target
            * residual_target
            * predicted_var
            / (predicted**4)
        )
        ratio_uncertainty = math.sqrt(max(0.0, ratio_variance))
        assigned = min(
            1.0,
            max(
                MIN_CLOSURE_UNCERTAINTY,
                abs(ratio - 1.0),
                ratio_uncertainty,
            ),
        )
        status = "measured"
    else:
        ratio = None
        ratio_uncertainty = None
        assigned = 1.0
        status = "insufficient_validation_statistics"
    return {
        "status": status,
        "source_region": KAPPA_SOURCE_REGION,
        "validation_region": region,
        "target_data": target,
        "target_prompt": prompt,
        "target_electron": electron,
        "target_fake_residual": residual_target,
        "target_fake_residual_variance": residual_target_var,
        "prediction": predicted,
        "prediction_variance_with_correlations": predicted_var,
        "target_over_prediction": ratio,
        "ratio_uncertainty": ratio_uncertainty,
        "assigned_relative_nonclosure": assigned,
        "prediction_histogram": prediction,
        "target_fake_residual_histogram": target_fake_histogram,
        "prediction_audit": audit,
        "factor_records": factors["records"],
        "policy": (
            "kappa is measured in the adjacent 0.30<=minDeltaPhi<0.50 "
            "region and validated without using nominal-GCR target data in "
            "the independent 0.10<=minDeltaPhi<0.30 region"
        ),
    }


def build_prediction(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    kappa_fit: dict[str, Any],
    closure: dict[str, Any],
    builder: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    central = fit_plj_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        "GCR",
        kappa_fit["factors"],
        kappa_fit["variances"],
        kappa_fit["covariance"],
    )
    prompt_up_kappa = fit_kappa(
        data_channels,
        prompt_channels,
        electron_channels,
        KAPPA_SOURCE_REGION,
        prompt_scale=1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
    )
    prompt_down_kappa = fit_kappa(
        data_channels,
        prompt_channels,
        electron_channels,
        KAPPA_SOURCE_REGION,
        prompt_scale=1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
    )
    electron_up_kappa = fit_kappa(
        data_channels,
        prompt_channels,
        electron_channels,
        KAPPA_SOURCE_REGION,
        electron_scale=1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
    )
    electron_down_kappa = fit_kappa(
        data_channels,
        prompt_channels,
        electron_channels,
        KAPPA_SOURCE_REGION,
        electron_scale=1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
    )
    varied_fits = {
        "photonFakePromptUp": (
            1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
            1.0,
            prompt_up_kappa,
        ),
        "photonFakePromptDown": (
            1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
            1.0,
            prompt_down_kappa,
        ),
        "photonFakeElectronUp": (
            1.0,
            1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
            electron_up_kappa,
        ),
        "photonFakeElectronDown": (
            1.0,
            1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
            electron_down_kappa,
        ),
    }
    refits = {
        name: fit_plj_factors(
            data_channels,
            prompt_channels,
            electron_channels,
            "GCR",
            fit["factors"],
            fit["variances"],
            fit["covariance"],
            prompt_scale,
            electron_scale,
        )
        for name, (prompt_scale, electron_scale, fit) in varied_fits.items()
    }
    output: dict[str, Any] = {
        "histograms": {},
        "highdm_variable_histograms": {},
    }
    audits: dict[str, Any] = {}
    for region in ("GCR", "GCR_Nt0", "GCR_Nt1"):
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(x) for x in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            nominal, nominal_audit = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                central["factors"],
                central["variances"],
                central["covariance"],
            )
            uncertainty = np.sqrt(np.maximum(central["variances"], 0.0))
            stat_up, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                np.minimum(MAX_FACTOR, central["factors"] + uncertainty),
                np.zeros_like(uncertainty),
            )
            stat_down, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                np.maximum(0.0, central["factors"] - uncertainty),
                np.zeros_like(uncertainty),
            )
            variations = {
                "nominal": nominal,
                "photonFakeStatUp": stat_up,
                "photonFakeStatDown": stat_down,
            }
            for name, (
                prompt_scale,
                electron_scale,
                _kappa,
            ) in varied_fits.items():
                varied, _ = predict_distribution(
                    data_channels,
                    prompt_channels,
                    electron_channels,
                    region,
                    variable,
                    edges,
                    refits[name]["factors"],
                    np.zeros_like(refits[name]["variances"]),
                    prompt_scale=prompt_scale,
                    electron_scale=electron_scale,
                )
                variations[name] = varied
            b_shape, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                central["factors"],
                np.zeros_like(central["variances"]),
                probes=("application",),
            )
            nominal_integral = sum(nominal["sumw"])
            b_integral = sum(b_shape["sumw"])
            if b_integral > 0.0:
                b_shape = _scaled(b_shape, nominal_integral / b_integral)
            variations["photonFakePLJShapeUp"] = b_shape
            variations["photonFakePLJShapeDown"] = {
                **copy.deepcopy(nominal),
                "sumw": [
                    max(0.0, 2.0 * n - b)
                    for n, b in zip(nominal["sumw"], b_shape["sumw"])
                ],
            }
            closure_uncertainty = float(
                closure["assigned_relative_nonclosure"]
            )
            variations["photonFakeClosureUp"] = _scaled(
                nominal, 1.0 + closure_uncertainty
            )
            variations["photonFakeClosureDown"] = _scaled(
                nominal, max(0.0, 1.0 - closure_uncertainty)
            )
            if variable == "recoil":
                output["histograms"][region] = variations
            else:
                output["highdm_variable_histograms"].setdefault(region, {})[
                    variable
                ] = variations
            audits[f"{region}/{variable}"] = nominal_audit
    return output, {
        "central_plj_factors": central["records"],
        "kappa": {
            "source_region": KAPPA_SOURCE_REGION,
            "records": kappa_fit["records"],
        },
        "variation_factor_records": {
            name: refit["records"] for name, refit in refits.items()
        },
        "prediction_audits": audits,
        "prompt_normalization_uncertainty": PROMPT_NORMALIZATION_UNCERTAINTY,
        "electron_normalization_uncertainty": ELECTRON_NORMALIZATION_UNCERTAINTY,
        "closure_uncertainty": float(
            closure["assigned_relative_nonclosure"]
        ),
    }


def target_summary(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    values = {}
    for name, channels, origin in (
        ("data", data_channels, "all"),
        ("prompt", prompt_channels, "prompt"),
        ("electron", electron_channels, "electron"),
    ):
        record = _distribution_record(
            channels, "target", origin, "GCR", "recoil"
        )
        values[name], values[f"{name}_variance"] = _sum_stratified(record)
    nominal = prediction["histograms"]["GCR"]["nominal"]
    fake, diagonal_variance = _leaf_integral(nominal)
    return {
        "data_target": values["data"],
        "data_target_variance": values["data_variance"],
        "prompt_target": values["prompt"],
        "prompt_target_variance": values["prompt_variance"],
        "electron_target": values["electron"],
        "electron_target_variance": values["electron_variance"],
        "observed_fake_residual_not_used_in_fit": (
            values["data"] - values["prompt"] - values["electron"]
        ),
        "predicted_fake": fake,
        "predicted_fake_diagonal_variance": diagonal_variance,
        "prompt_plus_electron_plus_fake": (
            values["prompt"] + values["electron"] + fake
        ),
        "prediction_over_data": (
            (values["prompt"] + values["electron"] + fake) / values["data"]
            if values["data"] > 0.0
            else None
        ),
        "blinding_policy": (
            "nominal-GCR target data are validation only and do not enter "
            "the kappa or PLJ extrapolation-factor measurement"
        ),
    }


def _coverage(
    summaries: list[dict[str, Any]],
    incomplete: list[str],
    processes: list[str],
    manifest_path: Path | None,
) -> dict[str, Any]:
    attempted = sum(int(item.get("files_attempted") or 0) for item in summaries)
    processed = sum(int(item.get("files_processed") or 0) for item in summaries)
    if manifest_path is None:
        return {
            "status": "partial_diagnostic",
            "reason": "campaign manifest not supplied",
            "observed_sidecars": len(summaries),
            "observed_files_attempted": attempted,
            "observed_files_processed": processed,
        }
    manifest = read_payload(manifest_path)
    expected_sidecars = int(manifest.get("jobs") or 0)
    expected_records = sum(
        int(value) for value in (manifest.get("record_counts") or {}).values()
    )
    expected_processes = sorted(manifest.get("requested_processes") or [])
    errors = []
    if len(summaries) != expected_sidecars:
        errors.append(
            f"sidecars {len(summaries)} != expected {expected_sidecars}"
        )
    if attempted != expected_records:
        errors.append(f"files attempted {attempted} != expected {expected_records}")
    if processed != expected_records:
        errors.append(f"files processed {processed} != expected {expected_records}")
    missing = sorted(set(expected_processes) - set(processes))
    if missing:
        errors.append(f"missing processes {missing}")
    if incomplete:
        errors.append(f"incomplete sidecars {len(incomplete)}")
    return {
        "status": "complete" if not errors else "incomplete",
        "campaign_manifest": str(manifest_path),
        "expected_sidecars": expected_sidecars,
        "observed_sidecars": len(summaries),
        "expected_records": expected_records,
        "observed_files_attempted": attempted,
        "observed_files_processed": processed,
        "expected_processes": expected_processes,
        "observed_processes": processes,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure 2024 fake photons with loose-fail ABCD, an independent "
            "delta-phi kappa validation, and a photon-like-jet shape."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("no photon-fake sidecars found")

    datasets: dict[str, Any] = {}
    data_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    incomplete: list[str] = []
    digests: set[str] = set()
    accepted_paths: list[Path] = []
    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != OUTPUT_SCHEMA:
            continue
        accepted_paths.append(path)
        summary = payload.get("summary") or {}
        digest = str(summary.get("source_record_digest") or "")
        if digest and digest in digests:
            raise RuntimeError(f"duplicate source record digest: {digest}")
        if digest:
            digests.add(digest)
        if payload.get("status") != "complete":
            incomplete.append(str(path))
        summaries.append(summary)
        data_events.extend(payload.get("data_events") or [])
        for physical, incoming in (payload.get("datasets") or {}).items():
            if physical not in datasets:
                datasets[physical] = incoming
            else:
                _merge_dataset_record(datasets[physical], incoming)
    if not accepted_paths:
        raise RuntimeError(f"no {OUTPUT_SCHEMA} sidecars found")
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"{len(incomplete)} sidecars are incomplete")

    builder = load_histogram_builder()
    normalization = read_payload(args.normalization)
    deduplicated, dedup_audit = deduplicate_data_events(data_events)
    data_channels = data_channels_from_events(deduplicated, builder)
    backgrounds = {"GJ", "QCD", "DY", "TT", "WtoLNu", "ST", "VV", "Zto2Nu"}
    prompt_channels, prompt_audit = aggregate_component(
        datasets, normalization, "prompt", backgrounds
    )
    electron_channels, electron_audit = aggregate_component(
        datasets, normalization, "electron", backgrounds
    )
    qcd_fake_channels, qcd_fake_audit = aggregate_component(
        datasets, normalization, "fake", {"QCD"}
    )
    qcd_all_channels, qcd_all_audit = aggregate_component(
        datasets, normalization, "all", {"QCD"}
    )
    qcd_prompt_channels, qcd_prompt_audit = aggregate_component(
        datasets, normalization, "prompt", {"QCD"}
    )
    qcd_electron_channels, qcd_electron_audit = aggregate_component(
        datasets, normalization, "electron", {"QCD"}
    )
    process_origin_channels: dict[str, dict[str, Any]] = {}
    process_origin_audits: dict[str, dict[str, Any]] = {}
    for process in sorted(backgrounds):
        process_origin_channels[process] = {}
        process_origin_audits[process] = {}
        for origin in ("all", "prompt", "electron", "fake"):
            channels, audit = aggregate_component(
                datasets,
                normalization,
                origin,
                {process},
            )
            process_origin_channels[process][origin] = channels
            process_origin_audits[process][origin] = audit

    kappa = fit_kappa(
        data_channels,
        prompt_channels,
        electron_channels,
        KAPPA_SOURCE_REGION,
    )
    closure = closure_summary(
        data_channels,
        prompt_channels,
        electron_channels,
        kappa,
        builder,
    )
    prediction, measurement = build_prediction(
        data_channels,
        prompt_channels,
        electron_channels,
        kappa,
        closure,
        builder,
    )
    validation = target_summary(
        data_channels,
        prompt_channels,
        electron_channels,
        prediction,
    )
    processes = sorted(
        {
            str(dataset.get("process") or "unknown")
            for dataset in datasets.values()
        }
        | ({"EGamma"} if data_events else set())
    )
    coverage = _coverage(
        summaries,
        incomplete,
        processes,
        args.campaign_manifest,
    )
    status = (
        "complete"
        if coverage["status"] == "complete" and not incomplete
        else "partial_diagnostic"
    )
    if status != "complete" and not args.allow_incomplete:
        raise RuntimeError(
            "photon-fake v2 coverage is incomplete: "
            + json.dumps(coverage, sort_keys=True)
        )
    output = {
        "schema_version": MEASUREMENT_SCHEMA,
        "status": status,
        "year": 2024,
        "scope": "high-dM photon control region",
        "selection_source": "real_subset_worker.py",
        "nominal_intermediate_policy": (
            "separate NanoAOD sidecars only; nominal intermediates are unchanged"
        ),
        "method": {
            "abcd": (
                "medium target with a loose-fail guard-band ABCD in "
                "sigma_ieta_ieta and charged isolation"
            ),
            "kappa": (
                "measured in an adjacent 0.30<=minDeltaPhi<0.50 data region; "
                "nominal-GCR target data are not used"
            ),
            "shape": (
                "photon-like-jet union failing exactly one loose photon-ID "
                "criterion while all other criteria pass medium"
            ),
            "validation": (
                "independent 0.10<=minDeltaPhi<0.30 data region plus QCD "
                "truth-origin diagnostics"
            ),
            "statistics": (
                "full target and prediction variance in closure; common-factor "
                "bin covariance retained in every predicted distribution"
            ),
        },
        "input_sidecars": [str(path) for path in accepted_paths],
        "normalization_source": str(args.normalization),
        "coverage": coverage,
        "input_summary": {
            "sidecar_count": len(summaries),
            "incomplete_sidecars": incomplete,
            "files_attempted": sum(
                int(item.get("files_attempted") or 0) for item in summaries
            ),
            "files_processed": sum(
                int(item.get("files_processed") or 0) for item in summaries
            ),
            "events_read": sum(
                int(item.get("events_read") or 0) for item in summaries
            ),
            "selected_events": sum(
                int(item.get("selected_events") or 0) for item in summaries
            ),
            "processes": processes,
            "target_cutbased_mismatch_objects": sum(
                int(item.get("target_cutbased_mismatch_objects") or 0)
                for item in summaries
            ),
        },
        "data_deduplication": dedup_audit,
        "component_audits": {
            "prompt": prompt_audit,
            "electron": electron_audit,
            "qcd_fake": qcd_fake_audit,
            "qcd_all": qcd_all_audit,
            "qcd_prompt": qcd_prompt_audit,
            "qcd_electron": qcd_electron_audit,
            "process_origins": process_origin_audits,
        },
        "kappa": {
            key: value
            for key, value in kappa.items()
            if key not in {"factors", "variances", "covariance"}
        },
        "closure": closure,
        "measurement": measurement,
        "target_validation": validation,
        "fake_prediction": prediction,
        "qcd_target_origin_histograms": qcd_target_origin_histograms(
            {
                "all": qcd_all_channels,
                "prompt": qcd_prompt_channels,
                "electron": qcd_electron_channels,
                "fake": qcd_fake_channels,
            },
            builder,
        ),
        "mc_target_origin_histograms": {
            process: qcd_target_origin_histograms(origins, builder)
            for process, origins in process_origin_channels.items()
        },
    }
    write_payload(args.output, output)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "sidecars": len(accepted_paths),
                "predicted_fake": validation["predicted_fake"],
                "data_target": validation["data_target"],
                "closure_status": closure["status"],
                "closure_ratio": closure["target_over_prediction"],
                "closure_uncertainty": closure[
                    "assigned_relative_nonclosure"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
