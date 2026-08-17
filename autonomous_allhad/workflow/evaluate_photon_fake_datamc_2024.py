#!/usr/bin/env python3
"""Evaluate GCR Data/MC before and after photon-fake replacement policies."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


NON_QCD_PROCESSES = ("DY", "GJ", "ST", "TT", "VV", "WtoLNu", "Zto2Nu")


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
    # Several nominal histograms retain bins below the GCR phase-space
    # threshold.  A bin that is exactly 0/0 is structurally inactive and must
    # not make an otherwise physical distribution fail.  Negative predictions,
    # non-finite values, or a positive data yield with no prediction remain
    # hard failures.
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
        return {
            "status": "empty_distribution",
            "bins": len(data),
            "inactive_zero_over_zero_bins": np.flatnonzero(~active)
            .astype(int)
            .tolist(),
        }
    selected_data = data[active]
    selected_prediction = prediction[active]
    selected_prediction_variance = prediction_variance[active]
    positive_data = selected_data > 0.0
    deviance_terms = selected_prediction - selected_data
    deviance_terms[positive_data] += selected_data[positive_data] * np.log(
        selected_data[positive_data] / selected_prediction[positive_data]
    )
    poisson_deviance = 2.0 * float(np.sum(deviance_terms))
    variance = (
        np.maximum(selected_data, 0.0)
        + np.maximum(selected_prediction_variance, 0.0)
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
    ratio = np.divide(
        selected_data,
        selected_prediction,
        out=np.zeros_like(selected_data),
        where=selected_prediction > 0.0,
    )
    positive_ratio = ratio > 0.0
    log_ratio = np.log(ratio[positive_ratio])
    integral_data = float(np.sum(data))
    integral_prediction = float(np.sum(prediction))
    if integral_data <= 0.0 or integral_prediction <= 0.0:
        return {
            "status": "invalid_nonpositive_integral",
            "bins": len(data),
            "active_bins": int(np.sum(active)),
            "integral_data": integral_data,
            "integral_prediction": integral_prediction,
        }
    integral_ratio = integral_data / integral_prediction
    return {
        "status": "valid",
        "bins": len(data),
        "active_bins": int(np.sum(active)),
        "inactive_zero_over_zero_bins": np.flatnonzero(~active)
        .astype(int)
        .tolist(),
        "poisson_deviance": poisson_deviance,
        "chi2_data_plus_mcstat": chi2,
        "log_ratio_rms": float(np.sqrt(np.mean(np.square(log_ratio)))),
        "max_abs_log_ratio": float(np.max(np.abs(log_ratio))),
        "integral_data": integral_data,
        "integral_prediction": integral_prediction,
        "integral_data_over_prediction": integral_ratio,
        "abs_log_integral_ratio": abs(math.log(integral_ratio)),
        "data_over_prediction_active_bins": ratio.tolist(),
    }


def improvement(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "poisson_deviance",
        "chi2_data_plus_mcstat",
        "log_ratio_rms",
        "abs_log_integral_ratio",
    )
    checks = {
        key: (
            candidate.get("status") == "valid"
            and reference.get("status") == "valid"
            and float(candidate[key]) < float(reference[key])
        )
        for key in keys
    }
    return {
        "checks": checks,
        "all_primary_metrics_improve": all(checks.values()),
        "relative_changes": {
            key: (
                float(candidate[key]) / float(reference[key]) - 1.0
                if float(reference[key]) != 0.0
                else None
            )
            for key in keys
            if key in reference and key in candidate
        },
    }


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


def evaluate_variable(
    nominal: dict[str, Any],
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    samples = nominal_variable(nominal, region, variable)
    data, data_var = arrays(samples["data_obs"]["nominal"])
    qcd, qcd_var = arrays(samples["QCD"]["nominal"])
    non_qcd = np.zeros_like(data)
    non_qcd_var = np.zeros_like(data)
    for process in NON_QCD_PROCESSES:
        values, variances = arrays(samples[process]["nominal"])
        non_qcd += values
        non_qcd_var += variances
    fake, fake_var = arrays(fake_leaf(measurement, region, variable))
    origins = measurement["qcd_target_origin_histograms"][region][variable]
    qcd_all, _ = arrays(origins["all"])
    qcd_prompt, _ = arrays(origins["prompt"])
    qcd_electron, _ = arrays(origins["electron"])
    qcd_fake, _ = arrays(origins["fake"])
    if not (
        len(data)
        == len(qcd)
        == len(non_qcd)
        == len(fake)
        == len(qcd_all)
    ):
        raise RuntimeError(f"bin-count mismatch for {region}/{variable}")

    raw_nonfake_fraction = np.divide(
        qcd_prompt + qcd_electron,
        qcd_all,
        out=np.zeros_like(qcd_all),
        where=qcd_all != 0.0,
    )
    nonfake_fraction = np.clip(raw_nonfake_fraction, 0.0, 1.0)
    retained_qcd_nonfake = qcd * nonfake_fraction
    retained_qcd_nonfake_var = qcd_var * np.square(nonfake_fraction)

    nominal_total = non_qcd + qcd
    nominal_total_var = non_qcd_var + qcd_var
    full_replacement_total = non_qcd + fake
    full_replacement_var = non_qcd_var + fake_var
    origin_replacement_total = non_qcd + retained_qcd_nonfake + fake
    origin_replacement_var = (
        non_qcd_var + retained_qcd_nonfake_var + fake_var
    )
    nominal_metrics = metrics(data, nominal_total, nominal_total_var)
    full_metrics = metrics(data, full_replacement_total, full_replacement_var)
    origin_metrics = metrics(
        data,
        origin_replacement_total,
        origin_replacement_var,
    )
    return {
        "region": region,
        "variable": variable,
        "data": data.tolist(),
        "non_qcd_mc": non_qcd.tolist(),
        "nominal_qcd_mc": qcd.tolist(),
        "qcd_sidecar_origins": {
            "all": qcd_all.tolist(),
            "prompt": qcd_prompt.tolist(),
            "electron": qcd_electron.tolist(),
            "fake": qcd_fake.tolist(),
            "partition_difference": (
                qcd_all - qcd_prompt - qcd_electron - qcd_fake
            ).tolist(),
        },
        "raw_qcd_nonfake_fraction": raw_nonfake_fraction.tolist(),
        "clipped_qcd_nonfake_fraction": nonfake_fraction.tolist(),
        "fraction_clipped_bins": np.flatnonzero(
            raw_nonfake_fraction != nonfake_fraction
        )
        .astype(int)
        .tolist(),
        "retained_qcd_nonfake": retained_qcd_nonfake.tolist(),
        "data_driven_fake": fake.tolist(),
        "predictions": {
            "nominal": nominal_total.tolist(),
            "replace_entire_qcd": full_replacement_total.tolist(),
            "replace_truth_fake_only": origin_replacement_total.tolist(),
        },
        "metrics": {
            "nominal": nominal_metrics,
            "replace_entire_qcd": full_metrics,
            "replace_truth_fake_only": origin_metrics,
        },
        "comparisons_to_nominal": {
            "replace_entire_qcd": improvement(nominal_metrics, full_metrics),
            "replace_truth_fake_only": improvement(nominal_metrics, origin_metrics),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    nominal = read_json(args.nominal)
    measurement = read_json(args.measurement)
    if measurement.get("status") != "complete":
        raise RuntimeError("measurement is not complete")
    variables: list[tuple[str, str]] = [
        ("GCR", "recoil"),
        ("GCR_Nt0", "recoil"),
        ("GCR_Nt1", "recoil"),
    ]
    variables.extend(
        ("GCR", variable)
        for variable in sorted(
            nominal.get("highdm_distribution_variable_specs") or {}
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
    primary = results["GCR/ut"]
    full_improves = primary["comparisons_to_nominal"]["replace_entire_qcd"][
        "all_primary_metrics_improve"
    ]
    origin_improves = primary["comparisons_to_nominal"][
        "replace_truth_fake_only"
    ]["all_primary_metrics_improve"]
    blocked = {
        name: len((audit or {}).get("blocked_datasets") or [])
        for name, audit in (measurement.get("component_audits") or {}).items()
    }
    normalization_complete = not any(blocked.values())
    decision = {
        "primary_distribution": "GCR/ut",
        "normalization_complete": normalization_complete,
        "blocked_normalization_datasets": blocked,
        "entire_qcd_replacement": (
            "reject" if not full_improves else "passes_data_mc_metrics"
        ),
        "truth_fake_only_replacement": (
            "conditional_pass_requires_qcd_gjets_overlap_decision"
            if origin_improves and normalization_complete
            else "reject"
        ),
        "adoption_gate": (
            "All four predeclared GCR U_T metrics must strictly improve over "
            "nominal: Poisson deviance, chi2 using data+MC statistical "
            "variance, log-ratio RMS, and absolute log integral ratio. "
            "Normalization must be complete. A truth-fake-only pass remains "
            "conditional until QCD/GJets prompt overlap is resolved."
        ),
    }
    output = {
        "schema_version": "photon_fake_2024_datamc_evaluation_v1",
        "status": "complete",
        "nominal": str(args.nominal),
        "measurement": str(args.measurement),
        "decision": decision,
        "results": results,
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "decision": decision,
                "gcr_ut_metrics": primary["metrics"],
                "gcr_ut_comparisons": primary["comparisons_to_nominal"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
