#!/usr/bin/env python3
"""Evaluate a v2 fake-photon estimate in the nominal prefit GCR."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


MEASUREMENT_SCHEMAS = {
    "photon_fake_2024_measurement_v2",
    "photon_fake_2024_measurement_v3",
}
BACKGROUND_PROCESSES = (
    "DY",
    "GJ",
    "QCD",
    "ST",
    "TT",
    "VV",
    "WtoLNu",
    "Zto2Nu",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def arrays(leaf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(leaf["sumw"], dtype=float),
        np.asarray(leaf["sumw2"], dtype=float),
    )


def nominal_variable(
    nominal: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    if variable == "recoil":
        return nominal["histograms"][region]
    return nominal["highdm_variable_histograms"][region][variable]


def fake_leaf(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    if variable == "recoil":
        return measurement["fake_prediction"]["histograms"][region]["nominal"]
    return measurement["fake_prediction"]["highdm_variable_histograms"][region][
        variable
    ]["nominal"]


def origin_leaf(
    measurement: dict[str, Any],
    process: str,
    region: str,
    variable: str,
) -> dict[str, Any]:
    return measurement["mc_target_origin_histograms"][process][region][variable]


def origin_fraction(
    origins: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prompt, prompt_var = arrays(origins["prompt"])
    electron, electron_var = arrays(origins["electron"])
    fake, fake_var = arrays(origins["fake"])
    all_values, _ = arrays(origins["all"])
    nonfake = prompt + electron
    nonfake_var = prompt_var + electron_var
    denominator = nonfake + fake
    direct = (
        np.isfinite(denominator)
        & np.isfinite(nonfake)
        & np.isfinite(fake)
        & (denominator > 0.0)
        & (nonfake >= 0.0)
        & (fake >= 0.0)
    )
    fraction = np.ones_like(denominator)
    variance = np.ones_like(denominator)
    fraction[direct] = nonfake[direct] / denominator[direct]
    variance[direct] = (
        np.square(fake[direct]) * nonfake_var[direct]
        + np.square(nonfake[direct]) * fake_var[direct]
    ) / np.power(denominator[direct], 4)

    total_nonfake = float(np.sum(nonfake))
    total_fake = float(np.sum(fake))
    total_denominator = total_nonfake + total_fake
    if total_denominator > 0.0 and total_nonfake >= 0.0 and total_fake >= 0.0:
        inclusive_fraction = total_nonfake / total_denominator
        inclusive_variance = (
            total_fake * total_fake * float(np.sum(nonfake_var))
            + total_nonfake * total_nonfake * float(np.sum(fake_var))
        ) / total_denominator**4
        fallback = ~direct
        fraction[fallback] = inclusive_fraction
        variance[fallback] = inclusive_variance
        fallback_policy = "process-variable inclusive origin fraction"
    else:
        fallback = ~direct
        fallback_policy = "retain MC with a 100% fraction uncertainty"

    raw_fraction = fraction.copy()
    fraction = np.clip(fraction, 0.0, 1.0)
    variance = np.maximum(variance, 0.0)
    partition = all_values - prompt - electron - fake
    return fraction, variance, {
        "prompt": prompt.tolist(),
        "electron": electron.tolist(),
        "fake": fake.tolist(),
        "all": all_values.tolist(),
        "partition_difference": partition.tolist(),
        "nonfake_fraction": fraction.tolist(),
        "nonfake_fraction_variance": variance.tolist(),
        "fallback_bins": np.flatnonzero(fallback).astype(int).tolist(),
        "fraction_clipped_bins": np.flatnonzero(raw_fraction != fraction)
        .astype(int)
        .tolist(),
        "fallback_policy": fallback_policy,
    }


def metrics(
    data: np.ndarray,
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
) -> dict[str, Any]:
    finite = (
        np.isfinite(data)
        & np.isfinite(prediction)
        & np.isfinite(prediction_variance)
    )
    active = (data != 0.0) | (prediction != 0.0)
    valid = finite & (
        (~active)
        | ((data >= 0.0) & (prediction > 0.0) & (prediction_variance >= 0.0))
    )
    if not np.all(valid):
        return {
            "status": "invalid_nonpositive_or_nonfinite_prediction",
            "invalid_bins": np.flatnonzero(~valid).astype(int).tolist(),
        }
    if not np.any(active):
        return {"status": "empty_distribution"}
    selected_data = data[active]
    selected_prediction = prediction[active]
    selected_prediction_variance = prediction_variance[active]
    positive_data = selected_data > 0.0
    deviance = selected_prediction - selected_data
    deviance[positive_data] += selected_data[positive_data] * np.log(
        selected_data[positive_data] / selected_prediction[positive_data]
    )
    variance = np.maximum(selected_data, 0.0) + np.maximum(
        selected_prediction_variance, 0.0
    )
    chi2 = float(
        np.sum(
            np.divide(
                np.square(selected_data - selected_prediction),
                variance,
                out=np.zeros_like(selected_data),
                where=variance > 0.0,
            )
        )
    )
    ratio = selected_data / selected_prediction
    log_ratio = np.log(ratio[ratio > 0.0])
    data_integral = float(np.sum(data))
    prediction_integral = float(np.sum(prediction))
    if data_integral <= 0.0 or prediction_integral <= 0.0:
        return {
            "status": "invalid_nonpositive_integral",
            "integral_data": data_integral,
            "integral_prediction": prediction_integral,
        }
    integral_ratio = data_integral / prediction_integral
    return {
        "status": "valid",
        "poisson_deviance": 2.0 * float(np.sum(deviance)),
        "chi2_data_plus_prediction": chi2,
        "log_ratio_rms": float(np.sqrt(np.mean(np.square(log_ratio)))),
        "max_abs_log_ratio": float(np.max(np.abs(log_ratio))),
        "integral_data": data_integral,
        "integral_prediction": prediction_integral,
        "integral_data_over_prediction": integral_ratio,
        "abs_log_integral_ratio": abs(math.log(integral_ratio)),
        "data_over_prediction_active_bins": ratio.tolist(),
    }


def comparison(
    nominal: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "poisson_deviance",
        "chi2_data_plus_prediction",
        "log_ratio_rms",
        "max_abs_log_ratio",
        "abs_log_integral_ratio",
    )
    checks = {
        key: (
            nominal.get("status") == "valid"
            and candidate.get("status") == "valid"
            and float(candidate[key]) < float(nominal[key])
        )
        for key in keys
    }
    return {
        "checks": checks,
        "all_primary_metrics_improve": all(checks.values()),
        "relative_changes": {
            key: float(candidate[key]) / float(nominal[key]) - 1.0
            if float(nominal[key]) != 0.0
            else None
            for key in keys
            if key in nominal and key in candidate
        },
    }


def evaluate_variable(
    nominal: dict[str, Any],
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    samples = nominal_variable(nominal, region, variable)
    data, _ = arrays(samples["data_obs"]["nominal"])
    nominal_total = np.zeros_like(data)
    nominal_variance = np.zeros_like(data)
    retained_total = np.zeros_like(data)
    retained_stat_variance = np.zeros_like(data)
    origin_fraction_variance = np.zeros_like(data)
    process_audits: dict[str, Any] = {}
    for process in BACKGROUND_PROCESSES:
        values, variances = arrays(samples[process]["nominal"])
        fraction, fraction_variance, audit = origin_fraction(
            origin_leaf(measurement, process, region, variable)
        )
        if len(values) != len(fraction):
            raise RuntimeError(f"bin mismatch for {region}/{variable}/{process}")
        retained = values * fraction
        nominal_total += values
        nominal_variance += variances
        retained_total += retained
        retained_stat_variance += variances * np.square(fraction)
        origin_fraction_variance += np.square(values) * fraction_variance
        process_audits[process] = {
            **audit,
            "nominal_mc": values.tolist(),
            "retained_prompt_plus_electron_mc": retained.tolist(),
        }
    measured_fake = fake_leaf(measurement, region, variable)
    fake, fake_variance = arrays(measured_fake)
    candidate_total = retained_total + fake
    candidate_variance = (
        retained_stat_variance + origin_fraction_variance + fake_variance
    )
    nominal_metrics = metrics(data, nominal_total, nominal_variance)
    candidate_metrics = metrics(data, candidate_total, candidate_variance)
    return {
        "region": region,
        "variable": variable,
        "bin_edges": [float(value) for value in measured_fake["bin_edges"]],
        "data": data.tolist(),
        "nominal_prediction": nominal_total.tolist(),
        "retained_prompt_plus_electron_mc": retained_total.tolist(),
        "data_driven_fake": fake.tolist(),
        "candidate_prediction": candidate_total.tolist(),
        "candidate_prediction_variance": candidate_variance.tolist(),
        "process_origin_audits": process_audits,
        "metrics": {
            "nominal": nominal_metrics,
            "replace_all_mc_truth_fake": candidate_metrics,
        },
        "comparison_to_nominal": comparison(
            nominal_metrics,
            candidate_metrics,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    nominal = read_json(args.nominal)
    measurement = read_json(args.measurement)
    if measurement.get("schema_version") not in MEASUREMENT_SCHEMAS:
        raise RuntimeError(
            f"unexpected measurement schema {measurement.get('schema_version')}"
        )
    if measurement.get("status") != "complete" and not args.allow_partial:
        raise RuntimeError("measurement is not complete")
    variables: list[tuple[str, str]] = [
        ("GCR", "recoil"),
        ("GCR_Nt0", "recoil"),
        ("GCR_Nt1", "recoil"),
    ]
    nominal_highdm_variables = set(
        nominal.get("highdm_distribution_variable_specs") or {}
    )
    measured_highdm_variables = set(
        (
            (measurement.get("fake_prediction") or {})
            .get("highdm_variable_histograms", {})
            .get("GCR", {})
        )
    )
    variables.extend(
        ("GCR", variable)
        for variable in sorted(
            nominal_highdm_variables & measured_highdm_variables
        )
    )
    results = {
        f"{region}/{variable}": evaluate_variable(
            nominal,
            measurement,
            region,
            variable,
        )
        for region, variable in variables
    }
    primary_key = "GCR/ut" if "GCR/ut" in results else "GCR/recoil"
    primary = results[primary_key]
    primary_pass = primary["comparison_to_nominal"][
        "all_primary_metrics_improve"
    ]
    blocked = {
        process: {
            origin: len((audit or {}).get("blocked_datasets") or [])
            for origin, audit in audits.items()
        }
        for process, audits in (
            (measurement.get("component_audits") or {})
            .get("process_origins", {})
            .items()
        )
    }
    used = {
        process: {
            origin: len((audit or {}).get("used_datasets") or [])
            for origin, audit in audits.items()
        }
        for process, audits in (
            (measurement.get("component_audits") or {})
            .get("process_origins", {})
            .items()
        )
    }
    normalization_complete = not any(
        count
        for process in blocked.values()
        for count in process.values()
    ) and all(
        count > 0
        for process in used.values()
        for count in process.values()
    )
    complete_measurement = measurement.get("status") == "complete"
    if not complete_measurement:
        adoption = "diagnostic_only_incomplete_measurement"
    elif primary_pass and normalization_complete:
        adoption = "adopt"
    else:
        adoption = "reject"
    output = {
        "schema_version": "photon_fake_2024_datamc_evaluation_v2",
        "status": (
            "complete"
            if measurement.get("status") == "complete"
            else "partial_diagnostic"
        ),
        "nominal": str(args.nominal),
        "measurement": str(args.measurement),
        "decision": {
            "primary_distribution": primary_key,
        "normalization_complete": normalization_complete,
        "missing_measurement_variables": sorted(
            nominal_highdm_variables - measured_highdm_variables
        ),
            "blocked_normalization_datasets": blocked,
            "used_normalization_datasets": used,
            "replace_all_mc_truth_fake": adoption,
            "adoption_gate": (
                "Adopt only if the prefit GCR U_T Poisson deviance, "
                "data+prediction chi2, log-ratio RMS, worst-bin log ratio, "
                "and integral Data/MC distance all improve, with complete MC "
                "normalization. Nominal target data are unchanged."
            ),
        },
        "results": results,
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "decision": output["decision"],
                "primary_metrics": primary["metrics"],
                "primary_comparison": primary["comparison_to_nominal"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
