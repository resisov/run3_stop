"""Simultaneous pass/fail resonance fits."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np


def apply_fit_config(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply fit-only changes while rejecting selections that require recounting."""

    checks = {
        "measurement": (payload["measurement"], config["measurement"]),
        "probe collection": (
            payload["probe_collection"],
            config["probe"]["collection"],
        ),
        "probe selection": (payload["probe_selection"], config["probe"]["selection"]),
        "pass selection": (payload["pass_selection"], config["probe"]["pass"]),
        "pT edges": (payload["probe_pt_edges_gev"], config["axes"]["pt_edges_gev"]),
        "eta edges": (payload["probe_abseta_edges"], config["axes"]["abseta_edges"]),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(
            f"refit configuration changes counted quantities: {mismatches}; recount first"
        )
    low, high = map(float, config["pair"]["mass_window_gev"])
    expected_edges = np.linspace(low, high, int(config["fit"]["mass_bins"]) + 1)
    actual_edges = np.asarray(payload["mass_edges_gev"], dtype=float)
    if actual_edges.shape != expected_edges.shape or not np.allclose(
        actual_edges, expected_edges, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "refit configuration changes the counted mass window or mass bins; recount first"
        )
    output = copy.deepcopy(dict(payload))
    output["fit"] = copy.deepcopy(dict(config["fit"]))
    return output


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1.0e-12, None)
    return values / np.sum(values)


def _signal(
    x: np.ndarray, mean: float, sigma: float, model: str, fit: Mapping[str, Any]
) -> np.ndarray:
    z = (x - mean) / sigma
    if model == "gaussian":
        values = np.exp(-0.5 * z * z)
    elif model == "double_gaussian":
        values = 0.8 * np.exp(-0.5 * z * z) + 0.2 * np.exp(-0.5 * (z / 2.0) ** 2)
    elif model == "crystal_ball":
        alpha = float(fit.get("crystal_ball_alpha", 1.5))
        power = float(fit.get("crystal_ball_n", 3.0))
        threshold = -abs(alpha)
        coefficient = (power / abs(alpha)) ** power * math.exp(-0.5 * alpha * alpha)
        offset = power / abs(alpha) - abs(alpha)
        values = np.where(
            z > threshold,
            np.exp(-0.5 * z * z),
            coefficient * np.maximum(offset - z, 1.0e-9) ** (-power),
        )
    elif model == "voigt":
        from scipy.special import voigt_profile

        values = voigt_profile(
            x - mean, sigma, float(fit.get("natural_width_gev", 1.2476))
        )
    else:
        raise ValueError(f"unsupported signal model: {model}")
    return _normalise(values)


def _background(x: np.ndarray, slope: float, curve: float, model: str) -> np.ndarray:
    centered = x - np.mean(x)
    span = max(float(np.ptp(x)), 1.0e-9)
    scaled = 2.0 * centered / span
    if model == "exponential":
        values = np.exp(np.clip(slope * centered, -30, 30))
    elif model == "linear":
        values = 1.0 + np.clip(slope, -0.95, 0.95) * scaled
    elif model == "chebyshev2":
        values = (
            1.0
            + np.clip(slope, -0.95, 0.95) * scaled
            + np.clip(curve, -0.95, 0.95) * (2.0 * scaled * scaled - 1.0)
        )
    else:
        raise ValueError(f"unsupported background model: {model}")
    return _normalise(values)


def _rebin(
    edges: np.ndarray, values: np.ndarray, variance: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor == 1:
        return edges, values, variance
    starts = np.arange(0, len(values), factor)
    return (
        np.concatenate([edges[starts], edges[-1:]]),
        np.add.reduceat(values, starts),
        np.add.reduceat(variance, starts),
    )


def fit_pass_fail(
    mass_edges: np.ndarray,
    histograms: Mapping[str, Any],
    fit: Mapping[str, Any],
    *,
    signal_model: str,
    background_model: str,
    rebin_factor: int,
    window: tuple[float, float],
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    pass_edges, passed, passed_variance = _rebin(
        np.asarray(mass_edges, dtype=float),
        np.asarray(histograms["pass_sumw"], dtype=float),
        np.asarray(histograms["pass_sumw2"], dtype=float),
        rebin_factor,
    )
    fail_edges, failed, failed_variance = _rebin(
        np.asarray(mass_edges, dtype=float),
        np.asarray(histograms["fail_sumw"], dtype=float),
        np.asarray(histograms["fail_sumw2"], dtype=float),
        rebin_factor,
    )
    if not np.array_equal(pass_edges, fail_edges):
        raise AssertionError("pass/fail mass edges diverged")
    centers = 0.5 * (pass_edges[:-1] + pass_edges[1:])
    selected = (centers >= window[0]) & (centers <= window[1])
    centers = centers[selected]
    passed = passed[selected]
    failed = failed[selected]
    passed_variance = np.maximum(passed_variance[selected], 1.0)
    failed_variance = np.maximum(failed_variance[selected], 1.0)
    if len(centers) < 8 or np.sum(passed) <= 0 or np.sum(failed) <= 0:
        return {"valid": False, "reason": "insufficient pass/fail spectrum"}
    peak_low, peak_high = map(float, fit["peak_bounds_gev"])
    peak_mask = (centers >= peak_low) & (centers <= peak_high)
    if not np.any(peak_mask):
        return {"valid": False, "reason": "peak bounds are outside the fit window"}
    combined = passed + failed
    mean = float(centers[peak_mask][np.argmax(combined[peak_mask])])
    width = float(window[1] - window[0])
    bin_width = float(np.median(np.diff(centers)))
    total = max(float(np.sum(combined)), 1.0)
    peak_pass = max(float(np.sum(passed[peak_mask])), 0.0)
    peak_fail = max(float(np.sum(failed[peak_mask])), 0.0)
    efficiency = np.clip(
        peak_pass / max(peak_pass + peak_fail, 1.0), 1.0e-3, 1.0 - 1.0e-3
    )
    initial = np.asarray(
        [
            mean,
            math.log(max(bin_width, width / 40.0)),
            math.log(max(0.7 * total, 1.0)),
            math.log(efficiency / (1.0 - efficiency)),
            math.log(max(0.3 * np.sum(passed), 1.0)),
            0.0,
            0.0,
            math.log(max(0.3 * np.sum(failed), 1.0)),
            0.0,
            0.0,
        ]
    )
    lower = np.asarray(
        [
            peak_low,
            math.log(max(bin_width / 4.0, 1.0e-4)),
            math.log(1.0e-6),
            -10.0,
            math.log(1.0e-6),
            -10.0,
            -0.95,
            math.log(1.0e-6),
            -10.0,
            -0.95,
        ]
    )
    upper = np.asarray(
        [
            peak_high,
            math.log(width / 3.0),
            math.log(max(10.0 * total, 10.0)),
            10.0,
            math.log(max(10.0 * total, 10.0)),
            10.0,
            0.95,
            math.log(max(10.0 * total, 10.0)),
            10.0,
            0.95,
        ]
    )

    def model(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        mean_value, log_sigma, log_signal, logit_eff = parameters[:4]
        signal = math.exp(log_signal) * _signal(
            centers, mean_value, math.exp(log_sigma), signal_model, fit
        )
        eff = 1.0 / (1.0 + math.exp(-logit_eff))
        pass_background = math.exp(parameters[4]) * _background(
            centers, parameters[5], parameters[6], background_model
        )
        fail_background = math.exp(parameters[7]) * _background(
            centers, parameters[8], parameters[9], background_model
        )
        return (
            eff * signal + pass_background,
            (1.0 - eff) * signal + fail_background,
            eff,
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        pass_model, fail_model, _ = model(parameters)
        return np.concatenate(
            [
                (passed - pass_model) / np.sqrt(passed_variance),
                (failed - fail_model) / np.sqrt(failed_variance),
            ]
        )

    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20_000)
    pass_model, fail_model, fitted_efficiency = model(result.x)
    chi2 = float(np.sum(residual(result.x) ** 2))
    ndf = max(1, 2 * len(centers) - len(result.x))
    covariance = np.linalg.pinv(result.jac.T @ result.jac, rcond=1.0e-12) * chi2 / ndf
    logit_uncertainty = math.sqrt(max(float(covariance[3, 3]), 0.0))
    efficiency_uncertainty = (
        fitted_efficiency * (1.0 - fitted_efficiency) * logit_uncertainty
    )
    fail_significance = (
        (1.0 - fitted_efficiency) / efficiency_uncertainty
        if efficiency_uncertainty
        else 0.0
    )
    valid = bool(
        result.success
        and math.isfinite(efficiency_uncertainty)
        and efficiency_uncertainty > 0.0
        and 0.0 < fitted_efficiency < 1.0
        and fail_significance >= float(fit.get("min_fail_significance", 0.0))
    )
    return {
        "valid": valid,
        "efficiency": float(fitted_efficiency),
        "efficiency_stat_uncertainty": float(efficiency_uncertainty),
        "fail_significance": float(fail_significance),
        "chi2_ndf": chi2 / ndf,
        "mass_centers_gev": centers.tolist(),
        "pass_observed": passed.tolist(),
        "fail_observed": failed.tolist(),
        "pass_model": pass_model.tolist(),
        "fail_model": fail_model.tolist(),
        "mean_gev": float(result.x[0]),
        "sigma_gev": float(math.exp(result.x[1])),
        "signal_model": signal_model,
        "background_model": background_model,
        "rebin_factor": int(rebin_factor),
        "window_gev": [float(window[0]), float(window[1])],
    }


def _bin(histograms: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {key: value[index] for key, value in histograms.items()}


def _variation_specs(
    fit: Mapping[str, Any], window: tuple[float, float]
) -> list[tuple[str, str, str, int, tuple[float, float]]]:
    signal = str(fit.get("signal_model", "gaussian"))
    background = str(fit.get("background_model", "exponential"))
    rebins = [int(value) for value in fit.get("rebin_factors", [1, 2])]
    shrink = float(fit.get("window_shrink_fraction", 0.05)) * (window[1] - window[0])
    specs = [("nominal", signal, background, rebins[0], window)]
    alternate_signal = str(fit.get("alternate_signal_model", signal))
    alternate_background = str(fit.get("alternate_background_model", background))
    if alternate_signal != signal:
        specs.append(
            ("alternate_signal", alternate_signal, background, rebins[0], window)
        )
    if alternate_background != background:
        specs.append(
            ("alternate_background", signal, alternate_background, rebins[0], window)
        )
    if len(rebins) > 1 and rebins[1] != rebins[0]:
        specs.append(("alternate_binning", signal, background, rebins[1], window))
    if shrink > 0:
        specs.append(
            (
                "narrow_window",
                signal,
                background,
                rebins[0],
                (window[0] + shrink, window[1] - shrink),
            )
        )
    return specs


def fit_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    mass_edges = np.asarray(payload["mass_edges_gev"], dtype=float)
    fit = payload["fit"]
    window = (float(mass_edges[0]), float(mass_edges[-1]))
    data = payload["samples"]["data"]
    mc = payload["samples"]["mc"]
    specs = _variation_specs(fit, window)
    mc_weight_variations = {
        name[4:]: histograms
        for name, histograms in payload["samples"].items()
        if name.startswith("mc__")
    }
    bins = []
    for index in range(len(data["pass_sumw"])):
        fits: dict[str, Any] = {}
        scale_factors: dict[str, float] = {}
        for name, signal, background, rebin, fit_window in specs:
            fits[name] = {
                "data": fit_pass_fail(
                    mass_edges,
                    _bin(data, index),
                    fit,
                    signal_model=signal,
                    background_model=background,
                    rebin_factor=rebin,
                    window=fit_window,
                ),
                "mc": fit_pass_fail(
                    mass_edges,
                    _bin(mc, index),
                    fit,
                    signal_model=signal,
                    background_model=background,
                    rebin_factor=rebin,
                    window=fit_window,
                ),
            }
            if (
                fits[name]["data"].get("valid")
                and fits[name]["mc"].get("valid")
                and fits[name]["mc"]["efficiency"] > 0
            ):
                scale_factors[name] = (
                    fits[name]["data"]["efficiency"] / fits[name]["mc"]["efficiency"]
                )
        nominal = scale_factors.get("nominal")
        weight_scale_factors: dict[str, float] = {}
        if nominal is not None:
            for name, varied in mc_weight_variations.items():
                varied_fit = fit_pass_fail(
                    mass_edges,
                    _bin(varied, index),
                    fit,
                    signal_model=str(fit.get("signal_model", "gaussian")),
                    background_model=str(fit.get("background_model", "exponential")),
                    rebin_factor=int(fit.get("rebin_factors", [1])[0]),
                    window=window,
                )
                fits[f"weight__{name}"] = {
                    "data": fits["nominal"]["data"],
                    "mc": varied_fit,
                }
                if varied_fit.get("valid") and varied_fit["efficiency"] > 0:
                    weight_scale_factors[name] = (
                        fits["nominal"]["data"]["efficiency"] / varied_fit["efficiency"]
                    )
        if nominal is None:
            bins.append(
                {
                    "flat_index": index,
                    "valid": False,
                    "fits": fits,
                    "reason": "nominal fit failed",
                }
            )
            continue
        nominal_data = fits["nominal"]["data"]
        nominal_mc = fits["nominal"]["mc"]
        statistical = nominal * math.hypot(
            nominal_data["efficiency_stat_uncertainty"] / nominal_data["efficiency"],
            nominal_mc["efficiency_stat_uncertainty"] / nominal_mc["efficiency"],
        )
        expected_variations = len(specs) - 1 + len(mc_weight_variations)
        shifts = [
            abs(value - nominal)
            for name, value in scale_factors.items()
            if name != "nominal"
        ]
        shifts.extend(abs(value - nominal) for value in weight_scale_factors.values())
        systematic = max(shifts, default=0.0)
        valid = len(shifts) == expected_variations
        bins.append(
            {
                "flat_index": index,
                "valid": valid,
                "scale_factor": float(nominal),
                "scale_factor_stat_uncertainty": float(statistical),
                "scale_factor_systematic_uncertainty": float(systematic),
                "scale_factor_uncertainty": float(math.hypot(statistical, systematic)),
                "variation_scale_factors": {
                    **scale_factors,
                    **{
                        f"weight__{key}": value
                        for key, value in weight_scale_factors.items()
                    },
                },
                "fits": fits,
            }
        )
    return {
        "schema_version": 1,
        "measurement": payload["measurement"],
        "year": payload["year"],
        "probe_collection": payload["probe_collection"],
        "probe_selection": payload["probe_selection"],
        "pass_selection": payload["pass_selection"],
        "probe_abseta_edges": payload["probe_abseta_edges"],
        "probe_pt_edges_gev": payload["probe_pt_edges_gev"],
        "mass_edges_gev": payload["mass_edges_gev"],
        "fit": fit,
        "correction": payload["correction"],
        "bins": bins,
        "status": "validation_pending",
        "adoption_blockers": list(payload.get("adoption_blockers", [])),
    }
