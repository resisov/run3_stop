#!/usr/bin/env python3
"""Test physically interpretable one-parameter GCR Data/MC improvements.

The nominal payload is never modified.  Candidate predictions are written to a
standalone audit JSON.  The only fitted degree of freedom in each candidate is
one prompt-photon normalization, determined from the trusted nominal GCR data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


NON_GJ_QCD_PROCESSES = ("DY", "ST", "TT", "VV", "WtoLNu", "Zto2Nu")


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


def nominal_samples(
    nominal: dict[str, Any], region: str, variable: str
) -> dict[str, Any]:
    if variable == "recoil":
        return nominal["histograms"][region]
    return nominal["highdm_variable_histograms"][region][variable]


def fake_leaf(
    measurement: dict[str, Any], region: str, variable: str
) -> dict[str, Any]:
    if variable == "recoil":
        return measurement["fake_prediction"]["histograms"][region]["nominal"]
    return measurement["fake_prediction"]["highdm_variable_histograms"][region][
        variable
    ]["nominal"]


def components(
    nominal: dict[str, Any],
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, np.ndarray]:
    samples = nominal_samples(nominal, region, variable)
    data, data_var = arrays(samples["data_obs"]["nominal"])
    gj, gj_var = arrays(samples["GJ"]["nominal"])
    qcd, qcd_var = arrays(samples["QCD"]["nominal"])
    other = np.zeros_like(data)
    other_var = np.zeros_like(data)
    for process in NON_GJ_QCD_PROCESSES:
        values, variances = arrays(samples[process]["nominal"])
        other += values
        other_var += variances
    fake, fake_var = arrays(fake_leaf(measurement, region, variable))
    origins = measurement["qcd_target_origin_histograms"][region][variable]
    qcd_all, _ = arrays(origins["all"])
    qcd_prompt, _ = arrays(origins["prompt"])
    qcd_electron, _ = arrays(origins["electron"])
    raw_fraction = np.divide(
        qcd_prompt + qcd_electron,
        qcd_all,
        out=np.zeros_like(qcd_all),
        where=qcd_all != 0.0,
    )
    nonfake_fraction = np.clip(raw_fraction, 0.0, 1.0)
    qcd_nonfake = qcd * nonfake_fraction
    qcd_nonfake_var = qcd_var * np.square(nonfake_fraction)
    return {
        "data": data,
        "data_var": data_var,
        "gj": gj,
        "gj_var": gj_var,
        "qcd": qcd,
        "qcd_var": qcd_var,
        "other": other,
        "other_var": other_var,
        "fake": fake,
        "fake_var": fake_var,
        "qcd_nonfake": qcd_nonfake,
        "qcd_nonfake_var": qcd_nonfake_var,
        "qcd_nonfake_fraction": nonfake_fraction,
        "qcd_nonfake_fraction_raw": raw_fraction,
    }


def poisson_fit(
    data: np.ndarray,
    fixed: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
    mask = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(data)
        & np.isfinite(fixed)
        & np.isfinite(template)
        & (data >= 0.0)
        & (fixed >= 0.0)
        & (template >= 0.0)
        & ((data > 0.0) | (fixed > 0.0) | (template > 0.0))
    )
    d = data[mask]
    f = fixed[mask]
    t = template[mask]
    if len(d) == 0 or float(np.sum(t)) <= 0.0:
        return {"status": "unmeasurable", "bins": np.flatnonzero(mask).tolist()}

    def score(alpha: float) -> float:
        prediction = f + alpha * t
        if np.any(prediction <= 0.0):
            return float("inf")
        return float(np.sum(t * (d / prediction - 1.0)))

    low = 0.0
    high = 1.0
    while score(high) > 0.0 and high < 1.0e6:
        high *= 2.0
    if score(low) <= 0.0:
        alpha = 0.0
        boundary = True
    elif high >= 1.0e6 and score(high) > 0.0:
        return {"status": "unbounded", "bins": np.flatnonzero(mask).tolist()}
    else:
        for _ in range(160):
            middle = 0.5 * (low + high)
            if score(middle) > 0.0:
                low = middle
            else:
                high = middle
        alpha = 0.5 * (low + high)
        boundary = False
    prediction = f + alpha * t
    fisher = float(
        np.sum(
            np.divide(
                np.square(t),
                prediction,
                out=np.zeros_like(t),
                where=prediction > 0.0,
            )
        )
    )
    sigma = 1.0 / math.sqrt(fisher) if fisher > 0.0 else None
    return {
        "status": "fit",
        "alpha": alpha,
        "data_stat_sigma": sigma,
        "boundary": boundary,
        "bins": np.flatnonzero(mask).astype(int).tolist(),
        "data_integral": float(np.sum(d)),
        "fixed_integral": float(np.sum(f)),
        "template_integral": float(np.sum(t)),
        "prediction_integral": float(np.sum(prediction)),
    }


def metrics(
    data: np.ndarray,
    prediction: np.ndarray,
    prediction_var: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
    active = (
        np.asarray(mask, dtype=bool)
        & ((data != 0.0) | (prediction != 0.0))
    )
    valid = (
        np.isfinite(data)
        & np.isfinite(prediction)
        & np.isfinite(prediction_var)
        & (data >= 0.0)
        & (prediction > 0.0)
        & (prediction_var >= 0.0)
    )
    if np.any(active & ~valid):
        return {
            "status": "invalid",
            "invalid_bins": np.flatnonzero(active & ~valid).astype(int).tolist(),
        }
    d = data[active]
    p = prediction[active]
    pv = prediction_var[active]
    if len(d) == 0 or float(np.sum(d)) <= 0.0 or float(np.sum(p)) <= 0.0:
        return {"status": "empty"}
    positive = d > 0.0
    terms = p - d
    terms[positive] += d[positive] * np.log(d[positive] / p[positive])
    poisson_deviance = 2.0 * float(np.sum(terms))
    variance = d + pv
    chi2 = float(
        np.sum(
            np.divide(
                np.square(d - p),
                variance,
                out=np.zeros_like(d),
                where=variance > 0.0,
            )
        )
    )
    ratio = d / p
    log_ratio = np.log(ratio[ratio > 0.0])
    integral_ratio = float(np.sum(d) / np.sum(p))
    return {
        "status": "valid",
        "active_bins": int(np.sum(active)),
        "poisson_deviance": poisson_deviance,
        "chi2_data_plus_mcstat": chi2,
        "log_ratio_rms": float(np.sqrt(np.mean(np.square(log_ratio)))),
        "max_abs_log_ratio": float(np.max(np.abs(log_ratio))),
        "integral_data": float(np.sum(d)),
        "integral_prediction": float(np.sum(p)),
        "integral_data_over_prediction": integral_ratio,
        "abs_log_integral_ratio": abs(math.log(integral_ratio)),
        "data_over_prediction": ratio.tolist(),
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
            reference.get("status") == "valid"
            and candidate.get("status") == "valid"
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
                if reference.get("status") == "valid"
                and candidate.get("status") == "valid"
                and float(reference[key]) != 0.0
                else None
            )
            for key in keys
        },
    }


def candidate_definitions(c: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "nominal": {
            "fixed": c["other"] + c["gj"] + c["qcd"],
            "fixed_var": c["other_var"] + c["gj_var"] + c["qcd_var"],
            "template": np.zeros_like(c["data"]),
            "template_var": np.zeros_like(c["data"]),
        },
        "replace_entire_qcd_with_data_fake": {
            "fixed": c["other"] + c["gj"] + c["fake"],
            "fixed_var": c["other_var"] + c["gj_var"] + c["fake_var"],
            "template": np.zeros_like(c["data"]),
            "template_var": np.zeros_like(c["data"]),
        },
        "replace_truth_fake_only": {
            "fixed": c["other"] + c["gj"] + c["qcd_nonfake"] + c["fake"],
            "fixed_var": (
                c["other_var"]
                + c["gj_var"]
                + c["qcd_nonfake_var"]
                + c["fake_var"]
            ),
            "template": np.zeros_like(c["data"]),
            "template_var": np.zeros_like(c["data"]),
        },
        "fit_gjets_keep_nominal_qcd": {
            "fixed": c["other"] + c["qcd"],
            "fixed_var": c["other_var"] + c["qcd_var"],
            "template": c["gj"],
            "template_var": c["gj_var"],
        },
        "fit_gjets_with_data_fake_drop_qcd_prompt": {
            "fixed": c["other"] + c["fake"],
            "fixed_var": c["other_var"] + c["fake_var"],
            "template": c["gj"],
            "template_var": c["gj_var"],
        },
        "fit_prompt_pool_with_data_fake": {
            "fixed": c["other"] + c["fake"],
            "fixed_var": c["other_var"] + c["fake_var"],
            "template": c["gj"] + c["qcd_nonfake"],
            "template_var": c["gj_var"] + c["qcd_nonfake_var"],
        },
    }


def per_bin_alpha(
    data: np.ndarray, fixed: np.ndarray, template: np.ndarray
) -> dict[str, Any]:
    valid = template > 0.0
    alpha = np.divide(
        data - fixed,
        template,
        out=np.full_like(data, np.nan),
        where=valid,
    )
    sigma = np.divide(
        np.sqrt(np.maximum(data, 0.0)),
        template,
        out=np.full_like(data, np.nan),
        where=valid,
    )
    return {
        "alpha": [None if not np.isfinite(x) else float(x) for x in alpha],
        "data_stat_sigma": [
            None if not np.isfinite(x) else float(x) for x in sigma
        ],
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
        if variable != "met"
    )
    primary_components = components(nominal, measurement, "GCR", "ut")
    primary_candidates = candidate_definitions(primary_components)
    fitted_alpha: dict[str, float] = {}
    fits: dict[str, Any] = {}
    low_mask = np.zeros(len(primary_components["data"]), dtype=bool)
    low_mask[:4] = True
    high_mask = ~low_mask
    for name, candidate in primary_candidates.items():
        if not np.any(candidate["template"] > 0.0):
            continue
        inclusive = poisson_fit(
            primary_components["data"],
            candidate["fixed"],
            candidate["template"],
        )
        low = poisson_fit(
            primary_components["data"],
            candidate["fixed"],
            candidate["template"],
            low_mask,
        )
        high = poisson_fit(
            primary_components["data"],
            candidate["fixed"],
            candidate["template"],
            high_mask,
        )
        if inclusive.get("status") != "fit":
            raise RuntimeError(f"primary fit failed for {name}: {inclusive}")
        fitted_alpha[name] = float(inclusive["alpha"])
        fits[name] = {
            "inclusive": inclusive,
            "low_ut_bins_0_to_3": low,
            "high_ut_bins_4_to_7": high,
            "per_bin": per_bin_alpha(
                primary_components["data"],
                candidate["fixed"],
                candidate["template"],
            ),
        }

    results: dict[str, Any] = {}
    for region, variable in variables:
        c = components(nominal, measurement, region, variable)
        definitions = candidate_definitions(c)
        predictions: dict[str, Any] = {}
        for name, candidate in definitions.items():
            alpha = fitted_alpha.get(name, 0.0)
            prediction = candidate["fixed"] + alpha * candidate["template"]
            prediction_var = (
                candidate["fixed_var"]
                + alpha * alpha * candidate["template_var"]
            )
            predictions[name] = {
                "alpha_from_primary_gcr_ut": alpha if name in fitted_alpha else None,
                "prediction": prediction.tolist(),
                "prediction_variance": prediction_var.tolist(),
                "metrics": metrics(c["data"], prediction, prediction_var),
            }
        nominal_metrics = predictions["nominal"]["metrics"]
        comparisons = {
            name: improvement(nominal_metrics, record["metrics"])
            for name, record in predictions.items()
            if name != "nominal"
        }
        results[f"{region}/{variable}"] = {
            "region": region,
            "variable": variable,
            "data": c["data"].tolist(),
            "components": {
                key: c[key].tolist()
                for key in (
                    "gj",
                    "qcd",
                    "other",
                    "fake",
                    "qcd_nonfake",
                    "qcd_nonfake_fraction",
                    "qcd_nonfake_fraction_raw",
                )
            },
            "predictions": predictions,
            "comparisons_to_nominal": comparisons,
        }

    category_fits: dict[str, Any] = {}
    for region in ("GCR", "GCR_Nt0", "GCR_Nt1"):
        c = components(nominal, measurement, region, "recoil")
        definitions = candidate_definitions(c)
        category_fits[region] = {}
        for name, candidate in definitions.items():
            if np.any(candidate["template"] > 0.0):
                category_fits[region][name] = poisson_fit(
                    c["data"], candidate["fixed"], candidate["template"]
                )

    valid_variables = [
        key
        for key, result in results.items()
        if result["predictions"]["nominal"]["metrics"].get("status") == "valid"
    ]
    candidate_summary = {}
    for name in primary_candidates:
        if name == "nominal":
            continue
        improved = [
            key
            for key in valid_variables
            if results[key]["comparisons_to_nominal"][name][
                "all_primary_metrics_improve"
            ]
        ]
        candidate_summary[name] = {
            "alpha_from_primary_gcr_ut": fitted_alpha.get(name),
            "valid_distributions": len(valid_variables),
            "all_metrics_improved_distributions": len(improved),
            "improved_distribution_names": improved,
        }

    payload = {
        "schema_version": "gcr_datamc_improvement_study_v1",
        "status": "complete",
        "selection_source": "real_subset_worker.py",
        "nominal": str(args.nominal),
        "measurement": str(args.measurement),
        "primary_distribution": "GCR/ut",
        "fit_policy": (
            "one Poisson-likelihood prompt normalization is fitted in the "
            "trusted nominal GCR U_T data and then frozen for every other "
            "distribution; data statistical uncertainty only is reported for "
            "the fitted alpha, while prediction metrics retain MC sumw2"
        ),
        "fits": fits,
        "category_fits": category_fits,
        "candidate_summary": candidate_summary,
        "results": results,
        "interpretation_guardrails": [
            "A fitted GCR normalization is a control-region constraint, not a prefit MC validation.",
            "The prompt-pool candidate is conditional on a generator-level QCD/GJets overlap policy.",
            "No candidate modifies the nominal histogram payload.",
            "The structurally empty GCR pTmiss histogram is excluded.",
        ],
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fits": fits,
                "candidate_summary": candidate_summary,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
