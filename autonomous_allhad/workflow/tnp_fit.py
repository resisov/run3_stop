#!/usr/bin/env python3
"""Simultaneous pass/fail resonance fits for low-pT lepton tag-and-probe.

Input histograms use one flattened probe (abs-eta, pT) bin per row.  Each
sample provides ``pass_sumw``, ``pass_sumw2``, ``fail_sumw``, and
``fail_sumw2`` arrays with shape ``[probe_bin][mass_bin]``.
"""
from __future__ import annotations

import math
import copy
from typing import Any

import numpy as np
from scipy.optimize import least_squares


def _normalised(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1.0e-12, None)
    total = float(np.sum(clipped))
    return clipped / total if total > 0 else np.full(len(clipped), 1.0 / len(clipped))


def _signal(
    centres: np.ndarray,
    mean: float,
    sigma: float,
    model: str,
    shape_parameters: tuple[float, ...] = (),
) -> np.ndarray:
    # Integrate the signal PDF over each mass bin.  The J/psi resolution is
    # comparable to the 20 MeV histogram bin width, so evaluating only at bin
    # centres produces an artificial alternating residual around the peak.
    centres = np.asarray(centres, dtype=float)
    bin_width = float(np.median(np.diff(centres))) if len(centres) > 1 else sigma
    offsets = ((np.arange(24, dtype=float) + 0.5) / 24.0 - 0.5) * bin_width
    evaluation_points = centres[:, None] + offsets[None, :]
    core = np.exp(-0.5 * ((evaluation_points - mean) / sigma) ** 2)
    if model == "crystal_ball":
        if len(shape_parameters) != 2:
            raise ValueError("crystal_ball requires alpha and n")
        alpha, n = shape_parameters
        t = (evaluation_points - mean) / sigma
        alpha = max(float(alpha), 1.0e-3)
        n = max(float(n), 1.001)
        b = n / alpha - alpha
        core = np.exp(-0.5 * t**2)
        tail = t <= -alpha
        log_a = n * math.log(n / alpha) - 0.5 * alpha**2
        log_tail = log_a - n * np.log(np.maximum(b - t[tail], 1.0e-12))
        core[tail] = np.exp(np.clip(log_tail, -700.0, 700.0))
    elif model == "double_crystal_ball":
        if len(shape_parameters) != 4:
            raise ValueError("double_crystal_ball requires alpha_left, n_left, alpha_right, and n_right")
        alpha_left, n_left, alpha_right, n_right = shape_parameters
        alpha_left = max(float(alpha_left), 1.0e-3)
        alpha_right = max(float(alpha_right), 1.0e-3)
        n_left = max(float(n_left), 1.001)
        n_right = max(float(n_right), 1.001)
        t = (evaluation_points - mean) / sigma
        left = t < -alpha_left
        right = t > alpha_right
        b_left = n_left / alpha_left - alpha_left
        b_right = n_right / alpha_right - alpha_right
        log_a_left = n_left * math.log(n_left / alpha_left) - 0.5 * alpha_left**2
        log_a_right = n_right * math.log(n_right / alpha_right) - 0.5 * alpha_right**2
        core[left] = np.exp(np.clip(
            log_a_left - n_left * np.log(np.maximum(b_left - t[left], 1.0e-12)),
            -700.0,
            700.0,
        ))
        core[right] = np.exp(np.clip(
            log_a_right - n_right * np.log(np.maximum(b_right + t[right], 1.0e-12)),
            -700.0,
            700.0,
        ))
    elif model == "gaussian_exponential":
        if len(shape_parameters) != 2:
            raise ValueError("gaussian_exponential requires left and right transition parameters")
        transition_left, transition_right = (
            max(float(value), 1.0e-3) for value in shape_parameters
        )
        t = (evaluation_points - mean) / sigma
        left = t < -transition_left
        right = t > transition_right
        core[left] = np.exp(0.5 * transition_left**2 + transition_left * t[left])
        core[right] = np.exp(0.5 * transition_right**2 - transition_right * t[right])
    elif model == "double_gaussian":
        tail = np.exp(-0.5 * ((evaluation_points - mean) / (2.0 * sigma)) ** 2)
        core = 0.8 * core + 0.2 * tail
    elif model != "gaussian":
        raise ValueError(f"unknown signal model: {model}")
    return _normalised(np.mean(core, axis=1))


def _background(
    centres: np.ndarray,
    shape: float,
    model: str,
    curvature: float = 0.0,
) -> np.ndarray:
    x = centres - float(np.mean(centres))
    if model == "exponential":
        values = np.exp(np.clip(shape * x, -30.0, 30.0))
    elif model == "chebyshev":
        scaled = x / max(float(np.max(np.abs(x))), 1.0e-12)
        first = scaled
        second = 2.0 * scaled**2 - 1.0
        values = (
            1.0
            + np.clip(shape, -0.95, 0.95) * first
            + np.clip(curvature, -0.95, 0.95) * second
        )
    elif model == "linear":
        span = max(float(np.ptp(centres)), 1.0e-6)
        values = 1.0 + np.clip(shape, -0.95, 0.95) * x / (span / 2.0)
    else:
        raise ValueError(f"unknown background model: {model}")
    return _normalised(values)


def fit_pass_fail(
    mass_edges: np.ndarray,
    pass_sumw: np.ndarray,
    pass_sumw2: np.ndarray,
    fail_sumw: np.ndarray,
    fail_sumw2: np.ndarray,
    *,
    fit_window: tuple[float, float],
    signal_model: str = "gaussian",
    background_model: str = "exponential",
    rebin_factor: int = 1,
    share_pass_fail_shape: bool = True,
) -> dict[str, Any]:
    edges = np.asarray(mass_edges, dtype=float)
    observed_pass_all = np.asarray(pass_sumw, dtype=float)
    variance_pass_all = np.asarray(pass_sumw2, dtype=float)
    observed_fail_all = np.asarray(fail_sumw, dtype=float)
    variance_fail_all = np.asarray(fail_sumw2, dtype=float)
    if rebin_factor < 1:
        raise ValueError("rebin_factor must be positive")
    if rebin_factor > 1:
        starts = np.arange(0, len(observed_pass_all), rebin_factor)
        edges = np.concatenate([edges[starts], edges[-1:]])
        observed_pass_all = np.add.reduceat(observed_pass_all, starts)
        variance_pass_all = np.add.reduceat(variance_pass_all, starts)
        observed_fail_all = np.add.reduceat(observed_fail_all, starts)
        variance_fail_all = np.add.reduceat(variance_fail_all, starts)
    centres_all = 0.5 * (edges[:-1] + edges[1:])
    selected = (centres_all >= fit_window[0]) & (centres_all <= fit_window[1])
    centres = centres_all[selected]
    observed_pass = observed_pass_all[selected]
    observed_fail = observed_fail_all[selected]
    variance_pass = np.maximum(variance_pass_all[selected], 1.0)
    variance_fail = np.maximum(variance_fail_all[selected], 1.0)
    if len(centres) < 8 or np.sum(observed_pass) <= 0 or np.sum(observed_fail) <= 0:
        return {"valid": False, "reason": "insufficient pass/fail mass spectrum"}

    resonance_low = max(float(fit_window[0]), 2.85)
    resonance_high = min(float(fit_window[1]), 3.30)
    resonance_mask = (centres >= resonance_low) & (centres <= resonance_high)
    if not np.any(resonance_mask):
        return {"valid": False, "reason": "J/psi resonance window is absent from fit range"}
    resonance_centres = centres[resonance_mask]
    peak = float(resonance_centres[np.argmax((observed_pass + observed_fail)[resonance_mask])])
    total = float(np.sum(observed_pass + observed_fail))
    resonance_pass = float(np.sum(observed_pass[resonance_mask]))
    resonance_fail = float(np.sum(observed_fail[resonance_mask]))
    efficiency = resonance_pass / max(resonance_pass + resonance_fail, 1.0e-12)
    # mean, log(sigma), log(Nsig), logit(eff), log(NbkgPass), bkgShapePass,
    # log(NbkgFail), bkgShapeFail
    initial_values = [
        peak,
        math.log(max((fit_window[1] - fit_window[0]) / 30.0, 0.01)),
        math.log(max(total * 0.7, 1.0)),
        math.log(max(efficiency, 1.0e-3) / max(1.0 - efficiency, 1.0e-3)),
        math.log(max(np.sum(observed_pass) * 0.3, 1.0)),
        0.0,
        math.log(max(np.sum(observed_fail) * 0.3, 1.0)),
        0.0,
    ]
    width = fit_window[1] - fit_window[0]
    background_lower, background_upper = (
        (-0.95, 0.95) if background_model == "chebyshev" else (-10.0, 10.0)
    )
    lower_values = [resonance_low, math.log(0.005), math.log(1.0e-6), -8.0, math.log(1.0e-6), background_lower, math.log(1.0e-6), background_lower]
    upper_values = [resonance_high, math.log(min(0.30, width / 2.0)), math.log(max(total * 10.0, 10.0)), 8.0, math.log(max(total * 10.0, 10.0)), background_upper, math.log(max(total * 10.0, 10.0)), background_upper]
    parameter_offset = 8
    background_curve_offset: int | None = None
    if background_model == "chebyshev":
        # A first-order Chebyshev is only a straight line.  The Run-2-style
        # curved combinatorial background requires the T2 coefficient as well,
        # independently in pass and fail spectra.
        background_curve_offset = parameter_offset
        initial_values.extend([0.0, 0.0])
        lower_values.extend([-0.95, -0.95])
        upper_values.extend([0.95, 0.95])
        parameter_offset += 2

    # The nominal DP-style simultaneous fit shares the J/psi line shape
    # between pass and fail probes.  An independent pass/fail response is kept
    # as an explicit model variation instead of introducing two weakly
    # constrained parameters in the nominal low-statistics fail spectrum.
    fail_response_offset: int | None = None
    if not share_pass_fail_shape:
        fail_response_offset = parameter_offset
        initial_values.extend([0.0, 0.0])
        lower_values.extend([-0.08, math.log(0.5)])
        upper_values.extend([0.08, math.log(2.0)])
        parameter_offset += 2
    shape_offset = parameter_offset
    if signal_model == "crystal_ball":
        # Fit alpha and n through positive transforms.  The broad bounds cover
        # the radiative low-mass tails seen in Run-3 J/psi data and simulation.
        initial_values.extend([math.log(1.5), math.log(2.0)])
        lower_values.extend([math.log(0.1), math.log(0.01)])
        upper_values.extend([math.log(10.0), math.log(99.0)])
    elif signal_model == "double_crystal_ball":
        initial_values.extend([math.log(1.5), math.log(2.0), math.log(1.8), math.log(2.0)])
        lower_values.extend([math.log(0.5), math.log(0.1), math.log(0.5), math.log(0.1)])
        upper_values.extend([math.log(5.0), math.log(29.0), math.log(5.0), math.log(29.0)])
    elif signal_model == "gaussian_exponential":
        initial_values.extend([math.log(1.5), math.log(1.8)])
        lower_values.extend([math.log(0.3), math.log(0.3)])
        upper_values.extend([math.log(8.0), math.log(8.0)])
    initial = np.asarray(initial_values)
    lower = np.asarray(lower_values)
    upper = np.asarray(upper_values)

    def components(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        mean, log_sigma, log_signal, logit_eff, log_bp, slope_p, log_bf, slope_f = parameters[:8]
        sigma = math.exp(log_sigma)
        signal_yield = math.exp(log_signal)
        fitted_eff = 1.0 / (1.0 + math.exp(-logit_eff))
        signal_parameters: tuple[float, ...] = ()
        curve_p = parameters[background_curve_offset] if background_curve_offset is not None else 0.0
        curve_f = parameters[background_curve_offset + 1] if background_curve_offset is not None else 0.0
        if share_pass_fail_shape:
            fail_mean = mean
            fail_sigma = sigma
        else:
            assert fail_response_offset is not None
            fail_mean = mean + parameters[fail_response_offset]
            fail_sigma = sigma * math.exp(parameters[fail_response_offset + 1])
        if signal_model == "crystal_ball":
            signal_parameters = (
                math.exp(parameters[shape_offset]),
                1.0 + math.exp(parameters[shape_offset + 1]),
            )
        elif signal_model == "double_crystal_ball":
            signal_parameters = (
                math.exp(parameters[shape_offset]),
                1.0 + math.exp(parameters[shape_offset + 1]),
                math.exp(parameters[shape_offset + 2]),
                1.0 + math.exp(parameters[shape_offset + 3]),
            )
        elif signal_model == "gaussian_exponential":
            signal_parameters = (
                math.exp(parameters[shape_offset]),
                math.exp(parameters[shape_offset + 1]),
            )
        pass_signal_shape = _signal(centres, mean, sigma, signal_model, signal_parameters)
        fail_signal_shape = _signal(centres, fail_mean, fail_sigma, signal_model, signal_parameters)
        pass_signal = signal_yield * fitted_eff * pass_signal_shape
        fail_signal = signal_yield * (1.0 - fitted_eff) * fail_signal_shape
        pass_background = math.exp(log_bp) * _background(
            centres, slope_p, background_model, curve_p
        )
        fail_background = math.exp(log_bf) * _background(
            centres, slope_f, background_model, curve_f
        )
        return pass_signal, pass_background, fail_signal, fail_background, fitted_eff

    def model(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        pass_signal, pass_background, fail_signal, fail_background, fitted_eff = components(parameters)
        return pass_signal + pass_background, fail_signal + fail_background, fitted_eff

    def residual(parameters: np.ndarray) -> np.ndarray:
        pass_model, fail_model, _ = model(parameters)
        return np.concatenate([
            (pass_model - observed_pass) / np.sqrt(variance_pass),
            (fail_model - observed_fail) / np.sqrt(variance_fail),
        ])

    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20000)
    fitted_pass, fitted_fail, fitted_eff = model(fit.x)
    pass_signal, pass_background, fail_signal, fail_background, _ = components(fit.x)
    ndf = max(1, 2 * len(centres) - len(fit.x))
    chi2 = float(np.sum(residual(fit.x) ** 2))
    normal_matrix = fit.jac.T @ fit.jac
    covariance_rank = int(np.linalg.matrix_rank(normal_matrix))
    covariance = np.linalg.pinv(normal_matrix, rcond=1.0e-12) * chi2 / ndf
    logit_sigma = math.sqrt(max(float(covariance[3, 3]), 0.0))
    efficiency_uncertainty = fitted_eff * (1.0 - fitted_eff) * logit_sigma
    fail_fraction_significance = (
        (1.0 - fitted_eff) / efficiency_uncertainty
        if efficiency_uncertainty > 0.0
        else 0.0
    )
    valid = bool(
        fit.success
        and math.isfinite(efficiency_uncertainty)
        and efficiency_uncertainty > 0.0
        and 0.01 < fitted_eff < 0.999
        # A numerical minimum with no identifiable peak in the failing probes
        # is not an efficiency measurement.  This implements the failure mode
        # explicitly documented in the Run-2 low-pT electron DP value table.
        and fail_fraction_significance >= 3.0
        and resonance_low + 1.0e-4 < fit.x[0] < resonance_high - 1.0e-4
        and 0.0051 < math.exp(fit.x[1]) < min(0.30, width / 2.0) - 1.0e-4
    )
    return {
        "valid": valid,
        "success": bool(fit.success),
        "message": str(fit.message),
        "efficiency": float(fitted_eff),
        "efficiency_stat_uncertainty": float(efficiency_uncertainty),
        "fail_fraction_significance": float(fail_fraction_significance),
        "fail_peak_significance_threshold": 3.0,
        "chi2": chi2,
        "ndf": int(ndf),
        "chi2_ndf": chi2 / ndf,
        "covariance_method": "Moore-Penrose pseudoinverse",
        "covariance_rank": covariance_rank,
        "fit_parameters": len(fit.x),
        "mean": float(fit.x[0]),
        "sigma": float(math.exp(fit.x[1])),
        "fail_mean": float(
            fit.x[0]
            if share_pass_fail_shape
            else fit.x[0] + fit.x[fail_response_offset]
        ),
        "fail_sigma": float(
            math.exp(fit.x[1])
            if share_pass_fail_shape
            else math.exp(fit.x[1]) * math.exp(fit.x[fail_response_offset + 1])
        ),
        "signal_yield": float(math.exp(fit.x[2])),
        "pass_model": [float(value) for value in fitted_pass],
        "fail_model": [float(value) for value in fitted_fail],
        "pass_signal_model": [float(value) for value in pass_signal],
        "fail_signal_model": [float(value) for value in fail_signal],
        "pass_background_model": [float(value) for value in pass_background],
        "fail_background_model": [float(value) for value in fail_background],
        "fit_window_gev": [float(value) for value in fit_window],
        "signal_model": signal_model,
        "background_model": background_model,
        "background_shape_parameters": {
            "pass_first_order": float(fit.x[5]),
            "fail_first_order": float(fit.x[7]),
            **(
                {
                    "pass_second_order": float(fit.x[background_curve_offset]),
                    "fail_second_order": float(fit.x[background_curve_offset + 1]),
                }
                if background_curve_offset is not None
                else {}
            ),
        },
        "fit_method": "simultaneous pass/fail analytic resonance fit",
        "pass_fail_signal_shape": "shared" if share_pass_fail_shape else "independent",
        "rebin_factor": int(rebin_factor),
        "signal_shape_parameters": (
            {
                "alpha": float(math.exp(fit.x[shape_offset])),
                "n": float(1.0 + math.exp(fit.x[shape_offset + 1])),
            }
            if signal_model == "crystal_ball" else (
                {
                    "alpha_left": float(math.exp(fit.x[shape_offset])),
                    "n_left": float(1.0 + math.exp(fit.x[shape_offset + 1])),
                    "alpha_right": float(math.exp(fit.x[shape_offset + 2])),
                    "n_right": float(1.0 + math.exp(fit.x[shape_offset + 3])),
                }
                if signal_model == "double_crystal_ball" else (
                    {
                        "transition_left": float(math.exp(fit.x[shape_offset])),
                        "transition_right": float(math.exp(fit.x[shape_offset + 1])),
                    }
                    if signal_model == "gaussian_exponential" else {}
                )
            )
        ),
    }


def _rebin_values(values: np.ndarray, factor: int) -> np.ndarray:
    parsed = np.asarray(values, dtype=float)
    if factor == 1:
        return parsed
    starts = np.arange(0, len(parsed), factor)
    return np.add.reduceat(parsed, starts)


def _rebinned_edges(edges: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(edges, dtype=float)
    starts = np.arange(0, len(edges) - 1, factor)
    return np.concatenate([np.asarray(edges, dtype=float)[starts], np.asarray(edges, dtype=float)[-1:]])


def weighted_counting_efficiency(
    mass_edges: np.ndarray,
    pass_sumw: np.ndarray,
    pass_sumw2: np.ndarray,
    fail_sumw: np.ndarray,
    fail_sumw2: np.ndarray,
    *,
    fit_window: tuple[float, float],
    rebin_factor: int = 1,
) -> dict[str, Any]:
    """Return the efficiency for a pure resonant MC sample by weighted counting."""

    edges = _rebinned_edges(np.asarray(mass_edges, dtype=float), rebin_factor)
    centres = 0.5 * (edges[:-1] + edges[1:])
    selected = (centres >= fit_window[0]) & (centres <= fit_window[1])
    passed = _rebin_values(np.asarray(pass_sumw, dtype=float), rebin_factor)[selected]
    failed = _rebin_values(np.asarray(fail_sumw, dtype=float), rebin_factor)[selected]
    passed_var = _rebin_values(np.asarray(pass_sumw2, dtype=float), rebin_factor)[selected]
    failed_var = _rebin_values(np.asarray(fail_sumw2, dtype=float), rebin_factor)[selected]
    pass_yield = float(np.sum(passed))
    fail_yield = float(np.sum(failed))
    total = pass_yield + fail_yield
    if total <= 0.0 or pass_yield <= 0.0 or fail_yield <= 0.0:
        return {"valid": False, "reason": "insufficient weighted MC pass/fail yield"}
    efficiency = pass_yield / total
    variance = (
        fail_yield**2 * float(np.sum(passed_var))
        + pass_yield**2 * float(np.sum(failed_var))
    ) / total**4
    uncertainty = math.sqrt(max(variance, 0.0))
    return {
        "valid": math.isfinite(uncertainty) and uncertainty > 0.0,
        "success": True,
        "message": "pure resonant MC weighted counting",
        "efficiency": float(efficiency),
        "efficiency_stat_uncertainty": float(uncertainty),
        "chi2": 0.0,
        "ndf": max(1, 2 * len(passed)),
        "chi2_ndf": 0.0,
        "fit_method": "weighted counting in pure J/psi simulation",
        "signal_yield": float(total),
        "pass_model": [float(value) for value in passed],
        "fail_model": [float(value) for value in failed],
        "fit_window_gev": [float(value) for value in fit_window],
        "signal_model": "pure_mc_counting",
        "background_model": "none",
        "rebin_factor": int(rebin_factor),
    }


def fit_template_pass_fail(
    mass_edges: np.ndarray,
    pass_sumw: np.ndarray,
    pass_sumw2: np.ndarray,
    fail_sumw: np.ndarray,
    fail_sumw2: np.ndarray,
    mc_pass_template: np.ndarray,
    mc_fail_template: np.ndarray,
    *,
    fit_window: tuple[float, float],
    template_mode: str = "separate",
    background_model: str = "exponential",
    rebin_factor: int = 1,
) -> dict[str, Any]:
    """Simultaneously fit data pass/fail spectra with morphed MC signal templates."""

    if template_mode not in {"separate", "combined"}:
        raise ValueError(f"unknown template mode: {template_mode}")
    base_edges = np.asarray(mass_edges, dtype=float)
    base_centres = 0.5 * (base_edges[:-1] + base_edges[1:])
    edges = _rebinned_edges(base_edges, rebin_factor)
    centres_all = 0.5 * (edges[:-1] + edges[1:])
    selected = (centres_all >= fit_window[0]) & (centres_all <= fit_window[1])
    centres = centres_all[selected]
    observed_pass = _rebin_values(np.asarray(pass_sumw, dtype=float), rebin_factor)[selected]
    observed_fail = _rebin_values(np.asarray(fail_sumw, dtype=float), rebin_factor)[selected]
    variance_pass = np.maximum(
        _rebin_values(np.asarray(pass_sumw2, dtype=float), rebin_factor)[selected], 1.0
    )
    variance_fail = np.maximum(
        _rebin_values(np.asarray(fail_sumw2, dtype=float), rebin_factor)[selected], 1.0
    )
    template_pass = np.clip(np.asarray(mc_pass_template, dtype=float), 0.0, None)
    template_fail = np.clip(np.asarray(mc_fail_template, dtype=float), 0.0, None)
    if template_mode == "combined":
        combined = template_pass + template_fail
        template_pass = combined
        template_fail = combined
    if (
        len(centres) < 8
        or np.sum(observed_pass) <= 0.0
        or np.sum(observed_fail) <= 0.0
        or np.sum(_rebin_values(template_pass, rebin_factor)[selected]) <= 0.0
        or np.sum(_rebin_values(template_fail, rebin_factor)[selected]) <= 0.0
    ):
        return {"valid": False, "reason": "insufficient data or MC template spectrum"}

    def morph(template: np.ndarray, shift: float, width_scale: float) -> np.ndarray:
        pivot = float(np.sum(base_centres * template) / np.sum(template))
        source = pivot + (base_centres - shift - pivot) / width_scale
        values = np.interp(source, base_centres, template, left=0.0, right=0.0)
        return _normalised(_rebin_values(values, rebin_factor)[selected])

    total = float(np.sum(observed_pass + observed_fail))
    raw_efficiency = float(np.sum(observed_pass) / total)
    # log(Nsig), logit(eff), log(NbkgPass), slopePass,
    # log(NbkgFail), slopeFail, template shift, log(width scale)
    initial = np.asarray([
        math.log(max(total * 0.7, 1.0)),
        math.log(max(raw_efficiency, 1.0e-3) / max(1.0 - raw_efficiency, 1.0e-3)),
        math.log(max(np.sum(observed_pass) * 0.3, 1.0)),
        0.0,
        math.log(max(np.sum(observed_fail) * 0.3, 1.0)),
        0.0,
        0.0,
        0.0,
    ])
    lower = np.asarray([
        math.log(1.0e-6), -12.0, math.log(1.0e-6), -10.0,
        math.log(1.0e-6), -10.0, -0.08, math.log(0.55),
    ])
    upper = np.asarray([
        math.log(max(total * 10.0, 10.0)), 12.0,
        math.log(max(total * 10.0, 10.0)), 10.0,
        math.log(max(total * 10.0, 10.0)), 10.0, 0.08, math.log(1.8),
    ])

    def model(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        log_signal, logit_eff, log_bp, slope_p, log_bf, slope_f, shift, log_scale = parameters
        signal_yield = math.exp(log_signal)
        efficiency = 1.0 / (1.0 + math.exp(-logit_eff))
        scale = math.exp(log_scale)
        pass_shape = morph(template_pass, shift, scale)
        fail_shape = morph(template_fail, shift, scale)
        passed = signal_yield * efficiency * pass_shape + math.exp(log_bp) * _background(centres, slope_p, background_model)
        failed = signal_yield * (1.0 - efficiency) * fail_shape + math.exp(log_bf) * _background(centres, slope_f, background_model)
        return passed, failed, efficiency

    def residual(parameters: np.ndarray) -> np.ndarray:
        passed, failed, _ = model(parameters)
        return np.concatenate([
            (passed - observed_pass) / np.sqrt(variance_pass),
            (failed - observed_fail) / np.sqrt(variance_fail),
        ])

    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20000)
    fitted_pass, fitted_fail, fitted_efficiency = model(fit.x)
    ndf = max(1, 2 * len(centres) - len(fit.x))
    chi2 = float(np.sum(residual(fit.x) ** 2))
    normal_matrix = fit.jac.T @ fit.jac
    covariance_rank = int(np.linalg.matrix_rank(normal_matrix))
    covariance = np.linalg.pinv(normal_matrix, rcond=1.0e-12) * chi2 / ndf
    logit_sigma = math.sqrt(max(float(covariance[1, 1]), 0.0))
    efficiency_uncertainty = fitted_efficiency * (1.0 - fitted_efficiency) * logit_sigma
    valid = bool(
        fit.success
        and math.isfinite(efficiency_uncertainty)
        and efficiency_uncertainty > 0.0
        and 0.0 < fitted_efficiency < 1.0
    )
    return {
        "valid": valid,
        "success": bool(fit.success),
        "message": str(fit.message),
        "efficiency": float(fitted_efficiency),
        "efficiency_stat_uncertainty": float(efficiency_uncertainty),
        "chi2": chi2,
        "ndf": int(ndf),
        "chi2_ndf": chi2 / ndf,
        "covariance_method": "Moore-Penrose pseudoinverse",
        "covariance_rank": covariance_rank,
        "fit_parameters": len(fit.x),
        "signal_yield": float(math.exp(fit.x[0])),
        "pass_model": [float(value) for value in fitted_pass],
        "fail_model": [float(value) for value in fitted_fail],
        "fit_window_gev": [float(value) for value in fit_window],
        "fit_method": "simultaneous pass/fail MC-template signal fit",
        "signal_model": f"mc_template_{template_mode}",
        "background_model": background_model,
        "template_shift_gev": float(fit.x[6]),
        "template_width_scale": float(math.exp(fit.x[7])),
        "rebin_factor": int(rebin_factor),
    }


def fit_scale_factor_bin(
    mass_edges: np.ndarray,
    data: dict[str, Any],
    mc: dict[str, Any],
    *,
    fit_window: tuple[float, float],
    nominal_rebin_factor: int = 1,
    nominal_background_model: str = "chebyshev",
) -> dict[str, Any]:
    if nominal_rebin_factor not in (1, 2):
        raise ValueError("nominal_rebin_factor must be 1 (20 MeV) or 2 (40 MeV)")
    alternate_rebin_factor = 2 if nominal_rebin_factor == 1 else 1
    if nominal_background_model not in {"chebyshev", "exponential"}:
        raise ValueError("nominal_background_model must be chebyshev or exponential")
    alternate_background_model = (
        "exponential" if nominal_background_model == "chebyshev" else "chebyshev"
    )
    variations = [
        ("nominal", "double_crystal_ball", nominal_background_model, fit_window, nominal_rebin_factor, True),
        ("alternate_signal", "gaussian_exponential", nominal_background_model, fit_window, nominal_rebin_factor, True),
        ("alternate_background", "double_crystal_ball", alternate_background_model, fit_window, nominal_rebin_factor, True),
        ("pass_fail_shape_independent", "double_crystal_ball", nominal_background_model, fit_window, nominal_rebin_factor, False),
        (
            "mass_window_narrow",
            "double_crystal_ball",
            nominal_background_model,
            (fit_window[0] + 0.08 * (fit_window[1] - fit_window[0]), fit_window[1] - 0.08 * (fit_window[1] - fit_window[0])),
            nominal_rebin_factor,
            True,
        ),
        ("alternate_binning", "double_crystal_ball", nominal_background_model, fit_window, alternate_rebin_factor, True),
    ]
    fits: dict[str, Any] = {}
    scale_factors: dict[str, float] = {}
    for name, signal_model, background_model, window, rebin_factor, share_pass_fail_shape in variations:
        fits[name] = {
            "data": fit_pass_fail(
                mass_edges,
                data["pass_sumw"],
                data["pass_sumw2"],
                data["fail_sumw"],
                data["fail_sumw2"],
                fit_window=window,
                signal_model=signal_model,
                background_model=background_model,
                rebin_factor=rebin_factor,
                share_pass_fail_shape=share_pass_fail_shape,
            ),
            "mc": fit_pass_fail(
                mass_edges,
                mc["pass_sumw"],
                mc["pass_sumw2"],
                mc["fail_sumw"],
                mc["fail_sumw2"],
                fit_window=window,
                signal_model=signal_model,
                background_model=background_model,
                rebin_factor=rebin_factor,
                share_pass_fail_shape=share_pass_fail_shape,
            ),
        }
        data_fit, mc_fit = fits[name]["data"], fits[name]["mc"]
        if data_fit.get("valid") and mc_fit.get("valid") and mc_fit["efficiency"] > 0:
            scale_factors[name] = data_fit["efficiency"] / mc_fit["efficiency"]
    nominal = scale_factors.get("nominal")
    if nominal is None:
        return {"valid": False, "fits": fits, "reason": "nominal data or MC fit failed"}
    nominal_data, nominal_mc = fits["nominal"]["data"], fits["nominal"]["mc"]
    stat = nominal * math.sqrt(
        (nominal_data["efficiency_stat_uncertainty"] / nominal_data["efficiency"]) ** 2
        + (nominal_mc["efficiency_stat_uncertainty"] / nominal_mc["efficiency"]) ** 2
    )
    shifts = [abs(value - nominal) for name, value in scale_factors.items() if name != "nominal"]
    systematic = max(shifts) if shifts else math.nan
    valid = math.isfinite(systematic) and len(scale_factors) == len(variations)
    return {
        "valid": valid,
        "scale_factor": float(nominal),
        "scale_factor_stat_uncertainty": float(stat),
        "scale_factor_systematic_uncertainty": float(systematic),
        "scale_factor_uncertainty": float(math.hypot(stat, systematic)) if math.isfinite(systematic) else math.nan,
        "variation_scale_factors": scale_factors,
        "fits": fits,
    }


def _fit_histogram_payload_nominal(payload: dict[str, Any]) -> dict[str, Any]:
    mass_edges = np.asarray(payload["mass_edges_gev"], dtype=float)
    fit_window = tuple(float(value) for value in payload["fit_window_gev"])
    data = payload["samples"]["data"]
    mc = payload["samples"]["mc"]
    nominal_rebin_factor = int(payload.get("nominal_mass_rebin_factor", 1))
    nominal_background_model = str(payload.get("nominal_background_model", "chebyshev"))
    bins = []
    for index in range(len(data["pass_sumw"])):
        data_bin = {key: data[key][index] for key in ("pass_sumw", "pass_sumw2", "fail_sumw", "fail_sumw2")}
        mc_bin = {key: mc[key][index] for key in ("pass_sumw", "pass_sumw2", "fail_sumw", "fail_sumw2")}
        item = fit_scale_factor_bin(
            mass_edges,
            data_bin,
            mc_bin,
            fit_window=fit_window,
            nominal_rebin_factor=nominal_rebin_factor,
            nominal_background_model=nominal_background_model,
        )
        item["flat_index"] = index
        bins.append(item)
    return {
        "schema_version": 1,
        "measurement": payload["measurement"],
        "year": str(payload.get("year") or "2024"),
        "probe_definition": payload.get("probe_definition"),
        "denominator_selection": payload.get("denominator_selection"),
        "target_selection": payload.get("target_selection"),
        "tag_pt_min_gev": payload.get("tag_pt_min_gev"),
        "tag_miniiso_max": payload.get("tag_miniiso_max", 0.1),
        "tag_selection": payload.get("tag_selection"),
        "external_reference_muon": payload.get("external_reference_muon"),
        "tag_trigger_match_required": payload.get("tag_trigger_match_required", True),
        "reference_trigger_object_kind": payload.get("reference_trigger_object_kind", payload.get("kind")),
        "reference_trigger_application": payload.get("reference_trigger_application"),
        "mc_reference": payload.get("mc_reference"),
        "status": "validation_pending",
        "probe_abseta_edges": payload["probe_abseta_edges"],
        "probe_pt_edges_gev": payload["probe_pt_edges_gev"],
        "mass_edges_gev": payload["mass_edges_gev"],
        "nominal_mass_rebin_factor": nominal_rebin_factor,
        "nominal_background_model": nominal_background_model,
        "nominal_mass_bin_width_mev": float(
            1000.0 * np.median(np.diff(mass_edges)) * nominal_rebin_factor
        ),
        "fit_window_gev": payload["fit_window_gev"],
        "bins": bins,
        "adoption_note": "All nominal and systematic fits, closure tests, and trigger-object audits must pass before status may become adopted.",
    }


def fit_histogram_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _fit_histogram_payload_nominal(payload)
    variations = {
        direction: payload["samples"].get(f"mc_pileup_{direction}")
        for direction in ("up", "down")
    }
    if not all(variations.values()):
        return result
    varied_results = {}
    for direction, mc_histograms in variations.items():
        varied_payload = copy.deepcopy(payload)
        varied_payload["samples"]["mc"] = mc_histograms
        varied_payload["samples"].pop("mc_pileup_up", None)
        varied_payload["samples"].pop("mc_pileup_down", None)
        varied_results[direction] = _fit_histogram_payload_nominal(varied_payload)
    for index, item in enumerate(result["bins"]):
        varied = {
            direction: varied_results[direction]["bins"][index]
            for direction in ("up", "down")
        }
        values = [entry.get("scale_factor") for entry in varied.values()]
        if item.get("valid") and all(entry.get("valid") and value is not None and math.isfinite(float(value)) for entry, value in zip(varied.values(), values)):
            nominal = float(item["scale_factor"])
            pileup_uncertainty = max(abs(float(value) - nominal) for value in values)
            fit_uncertainty = float(item["scale_factor_systematic_uncertainty"])
            systematic = math.hypot(fit_uncertainty, pileup_uncertainty)
            item["scale_factor_fit_systematic_uncertainty"] = fit_uncertainty
            item["scale_factor_pileup_uncertainty"] = pileup_uncertainty
            item["scale_factor_systematic_uncertainty"] = systematic
            item["scale_factor_uncertainty"] = math.hypot(float(item["scale_factor_stat_uncertainty"]), systematic)
            item["pileup_variation_scale_factors"] = {
                direction: float(value) for direction, value in zip(("up", "down"), values)
            }
        else:
            item["valid"] = False
            item["pileup_variation_scale_factors"] = {
                direction: entry.get("scale_factor") for direction, entry in varied.items()
            }
            item["pileup_failure"] = "nominal or pileup-varied fit failed"
    result["pileup_uncertainty_source"] = payload.get("pileup_correction")
    return result
