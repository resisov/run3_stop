#!/usr/bin/env python3
"""Reduce and plot the 2024 event-level lepton-removal closure."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import chi2 as chi2_distribution


SCHEMA_VERSION = "lost_lepton_removal_analysis_2024_v1"
COLORS = {
    "prediction": "#3b73b9",
    "target": "#111111",
    "electron": "#e67e22",
    "muon": "#2a9d8f",
    "Top": "#5e3c99",
    "W": "#1b9e77",
}
VR_TO_OLD = {
    "highdm_vr_nb0": "highdm_nb0",
    "highdm_vr_njet3to4_nb1plus": "highdm_njet3to4_nb1plus",
    "lowdm_vr_met250to300": "lowdm_met250to300",
    "lowdm_vr_isr200to300": "lowdm_isr200to300",
    "lowdm_vr_significance7to10": "lowdm_significance7to10",
}
NOMINAL_BY_REGIME = {
    "highdm": "highdm_ut",
    "lowdm": "lowdm_search42",
}
OLD_MC_SCHEME = {
    "highdm_ut": "highdm_met",
    "highdm_search60": "highdm_search60",
    "lowdm_search42": "lowdm_search42",
}
XLABELS = {
    "highdm_ut": r"$U_{T}$ (GeV)",
    "highdm_search60": "High-$\\Delta m$ search-bin index",
    "lowdm_search42": "Low-$\\Delta m$ search-bin index",
    "highdm_vr_nb0": r"$U_{T}$ (GeV)",
    "highdm_vr_njet3to4_nb1plus": r"$U_{T}$ (GeV)",
    "lowdm_vr_met250to300": r"$p_{T}^{miss}$ (GeV)",
    "lowdm_vr_isr200to300": r"$p_{T}^{ISR}$ (GeV)",
    "lowdm_vr_significance7to10": r"$p_{T}^{miss}/\sqrt{H_{T}}$ ($\sqrt{\mathrm{GeV}}$)",
}


def display_bin_labels(labels: list[str]) -> list[str]:
    """Render compact physics bin labels rather than internal key strings."""
    rendered = []
    for label in labels:
        if label.endswith("plus") and label[:-4].isdigit():
            rendered.append(rf"$\geq {label[:-4]}$")
        else:
            rendered.append(label.replace("-", "\N{EN DASH}"))
    return rendered


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def array(hist: dict[str, Any], field: str = "sumw") -> np.ndarray:
    return np.asarray(hist[field], dtype=float)


def hist_for(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    fold: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    hist = merged["components"][component][scheme][str(fold)][kind]
    return array(hist), array(hist, "sumw2")


def prediction_kind(method: str, flavor: str) -> list[str]:
    if flavor == "combined":
        return [f"prediction_{method}_e", f"prediction_{method}_mu"]
    if flavor not in {"e", "mu"}:
        raise ValueError(flavor)
    return [f"prediction_{method}_{flavor}"]


def summed_prediction(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    fold: int,
    method: str,
    flavor: str = "combined",
) -> tuple[np.ndarray, np.ndarray]:
    values = None
    variance = None
    for kind in prediction_kind(method, flavor):
        current, current_variance = hist_for(
            merged, component, scheme, fold, kind
        )
        values = current.copy() if values is None else values + current
        variance = (
            current_variance.copy()
            if variance is None
            else variance + current_variance
        )
    assert values is not None and variance is not None
    return values, variance


def summed_target(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    fold: int,
    target_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    return hist_for(merged, component, scheme, fold, target_kind)


def ratio_with_variance(
    numerator: float,
    numerator_variance: float,
    denominator: float,
    denominator_variance: float,
) -> tuple[float | None, float | None]:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None, None
    value = numerator / denominator
    variance = numerator_variance / denominator**2
    if denominator_variance >= 0.0:
        variance += numerator**2 * denominator_variance / denominator**4
    return float(value), float(max(variance, 0.0))


def full_normalization(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    method: str = "post",
    target_kind: str = "target_inclusive",
) -> dict[str, Any]:
    prediction = []
    prediction_variance = []
    target = []
    target_variance = []
    for fold in (0, 1):
        p, pv = summed_prediction(
            merged, component, scheme, fold, method, "combined"
        )
        t, tv = summed_target(
            merged, component, scheme, fold, target_kind
        )
        prediction.append(p)
        prediction_variance.append(pv)
        target.append(t)
        target_variance.append(tv)
    p_total = float(np.sum(prediction))
    pv_total = float(np.sum(prediction_variance))
    t_total = float(np.sum(target))
    tv_total = float(np.sum(target_variance))
    alpha, alpha_variance = ratio_with_variance(
        t_total, tv_total, p_total, pv_total
    )
    return {
        "alpha": alpha,
        "alpha_variance": alpha_variance,
        "prediction_before_alpha": p_total,
        "prediction_before_alpha_variance": pv_total,
        "target": t_total,
        "target_variance": tv_total,
    }


def closure_metrics(
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
    target: np.ndarray,
    target_variance: np.ndarray,
) -> dict[str, Any]:
    covariance = np.asarray(prediction_variance) + np.asarray(target_variance)
    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(covariance)
        & (covariance > 0.0)
    )
    pulls = np.full(len(prediction), np.nan, dtype=float)
    pulls[valid] = (
        np.asarray(prediction)[valid] - np.asarray(target)[valid]
    ) / np.sqrt(covariance[valid])
    chi2 = float(np.sum(np.square(pulls[valid])))
    ndf = int(np.count_nonzero(valid))
    pvalue = float(chi2_distribution.sf(chi2, ndf)) if ndf else None
    pred_integral = float(np.sum(prediction))
    target_integral = float(np.sum(target))
    ratio, ratio_variance = ratio_with_variance(
        pred_integral,
        float(np.sum(prediction_variance)),
        target_integral,
        float(np.sum(target_variance)),
    )
    positive = (prediction > 0.0) & (target > 0.0)
    half_l1 = None
    if np.any(positive):
        pred_shape = prediction[positive] / np.sum(prediction[positive])
        target_shape = target[positive] / np.sum(target[positive])
        half_l1 = float(0.5 * np.sum(np.abs(pred_shape - target_shape)))
    return {
        "prediction": prediction.tolist(),
        "prediction_variance": prediction_variance.tolist(),
        "target": target.tolist(),
        "target_variance": target_variance.tolist(),
        "ratio": ratio,
        "ratio_uncertainty": (
            math.sqrt(ratio_variance)
            if ratio_variance is not None
            else None
        ),
        "chi2": chi2,
        "ndf": ndf,
        "chi2_over_ndf": chi2 / ndf if ndf else None,
        "p_value": pvalue,
        "maximum_absolute_pull": (
            float(np.nanmax(np.abs(pulls))) if np.any(valid) else None
        ),
        "pull": [float(value) if np.isfinite(value) else None for value in pulls],
        "positive_bin_half_l1_shape_distance": half_l1,
    }


def crossfit_component(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    method: str,
    flavor: str = "combined",
    target_kind: str = "target_inclusive",
) -> dict[str, Any]:
    predictions = []
    prediction_variances = []
    targets = []
    target_variances = []
    directions = []
    for test_fold in (0, 1):
        train_fold = 1 - test_fold
        train_pred, train_pred_var = summed_prediction(
            merged, component, scheme, train_fold, method, flavor
        )
        train_target, train_target_var = summed_target(
            merged, component, scheme, train_fold, target_kind
        )
        alpha, alpha_variance = ratio_with_variance(
            float(np.sum(train_target)),
            float(np.sum(train_target_var)),
            float(np.sum(train_pred)),
            float(np.sum(train_pred_var)),
        )
        if alpha is None or alpha_variance is None:
            raise RuntimeError(
                f"invalid crossfit normalization: {component} {scheme} "
                f"{method} {flavor} fold {train_fold}"
            )
        test_pred, test_pred_var = summed_prediction(
            merged, component, scheme, test_fold, method, flavor
        )
        test_target, test_target_var = summed_target(
            merged, component, scheme, test_fold, target_kind
        )
        scaled = alpha * test_pred
        scaled_variance = (
            alpha**2 * test_pred_var
            + np.square(test_pred) * alpha_variance
        )
        predictions.append(scaled)
        prediction_variances.append(scaled_variance)
        targets.append(test_target)
        target_variances.append(test_target_var)
        directions.append(
            {
                "train_fold": train_fold,
                "test_fold": test_fold,
                "alpha": alpha,
                "alpha_uncertainty": math.sqrt(alpha_variance),
            }
        )
    metrics = closure_metrics(
        np.sum(predictions, axis=0),
        np.sum(prediction_variances, axis=0),
        np.sum(targets, axis=0),
        np.sum(target_variances, axis=0),
    )
    metrics.update(
        {
            "component": component,
            "scheme": scheme,
            "method": method,
            "flavor": flavor,
            "target_kind": target_kind,
            "directions": directions,
        }
    )
    return metrics


def crossfit_combined(
    component_results: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction = np.sum(
        [np.asarray(item["prediction"], dtype=float) for item in component_results],
        axis=0,
    )
    prediction_variance = np.sum(
        [
            np.asarray(item["prediction_variance"], dtype=float)
            for item in component_results
        ],
        axis=0,
    )
    target = np.sum(
        [np.asarray(item["target"], dtype=float) for item in component_results],
        axis=0,
    )
    target_variance = np.sum(
        [
            np.asarray(item["target_variance"], dtype=float)
            for item in component_results
        ],
        axis=0,
    )
    return closure_metrics(
        prediction,
        prediction_variance,
        target,
        target_variance,
    )


def full_hist(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    variances = []
    for fold in (0, 1):
        current, current_variance = hist_for(
            merged, component, scheme, fold, kind
        )
        values.append(current)
        variances.append(current_variance)
    return np.sum(values, axis=0), np.sum(variances, axis=0)


def data_validation(
    merged: dict[str, Any],
    top_w: dict[str, Any],
    scheme: str,
    normalizations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    regime = merged["schemes"][scheme]["regime"]
    sf_fit = top_w["fits"][regime]
    sf = np.asarray(sf_fit["scale_factors"], dtype=float)
    sf_cov = np.asarray(sf_fit["scale_factor_covariance"], dtype=float)
    component_removal = []
    component_removal_variance = []
    component_alpha = []
    component_alpha_variance = []
    for component in ("Top", "W"):
        electron, electron_var = full_hist(
            merged, component, scheme, "prediction_post_e"
        )
        muon, muon_var = full_hist(
            merged, component, scheme, "prediction_post_mu"
        )
        removal = electron + muon
        removal_variance = electron_var + muon_var
        alpha_record = normalizations[regime][component]
        component_removal.append(removal)
        component_removal_variance.append(removal_variance)
        component_alpha.append(float(alpha_record["alpha"]))
        component_alpha_variance.append(
            float(alpha_record["alpha_variance"])
        )

    removal_matrix = np.vstack(component_removal)
    removal_variance_matrix = np.vstack(component_removal_variance)
    alpha = np.asarray(component_alpha, dtype=float)
    alpha_variance = np.asarray(component_alpha_variance, dtype=float)
    weighted_source = sf[:, None] * removal_matrix
    denominator = np.sum(weighted_source, axis=0)
    numerator = np.sum(alpha[:, None] * weighted_source, axis=0)
    effective_factor = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0.0,
    )

    # Propagate the Top/W MC source statistics, the independent residual-alpha
    # uncertainties, and the fitted Top/W scale-factor covariance to the
    # effective lost/pass factor.  Correlations between each residual alpha
    # and the same MC source histogram are not available in the flat payload
    # and are intentionally left as a documented prototype limitation.
    factor_variance = np.zeros_like(effective_factor)
    valid_denominator = denominator != 0.0
    for component_index in range(2):
        gradient_removal = np.divide(
            sf[component_index]
            * (
                alpha[component_index] * denominator
                - numerator
            ),
            np.square(denominator),
            out=np.zeros_like(denominator),
            where=valid_denominator,
        )
        factor_variance += (
            np.square(gradient_removal)
            * removal_variance_matrix[component_index]
        )
        gradient_alpha = np.divide(
            sf[component_index] * removal_matrix[component_index],
            denominator,
            out=np.zeros_like(denominator),
            where=valid_denominator,
        )
        factor_variance += (
            np.square(gradient_alpha) * alpha_variance[component_index]
        )
    gradient_sf = np.zeros_like(removal_matrix)
    for component_index in range(2):
        gradient_sf[component_index] = np.divide(
            removal_matrix[component_index]
            * (
                alpha[component_index] * denominator
                - numerator
            ),
            np.square(denominator),
            out=np.zeros_like(denominator),
            where=valid_denominator,
        )
    for bin_index in range(len(effective_factor)):
        gradient = gradient_sf[:, bin_index]
        factor_variance[bin_index] += float(gradient @ sf_cov @ gradient)

    data_source_e, data_source_e_variance = full_hist(
        merged, "Data", scheme, "prediction_post_e"
    )
    data_source_mu, data_source_mu_variance = full_hist(
        merged, "Data", scheme, "prediction_post_mu"
    )
    other_source_e, other_source_e_variance = full_hist(
        merged, "Other", scheme, "prediction_post_e"
    )
    other_source_mu, other_source_mu_variance = full_hist(
        merged, "Other", scheme, "prediction_post_mu"
    )
    data_source = data_source_e + data_source_mu
    other_source = other_source_e + other_source_mu
    source_residual = data_source - other_source
    source_residual_variance = (
        data_source_e_variance
        + data_source_mu_variance
        + other_source_e_variance
        + other_source_mu_variance
    )
    prediction = effective_factor * source_residual
    prediction_variance = (
        np.square(effective_factor) * source_residual_variance
        + np.square(source_residual) * factor_variance
    )

    data_target, data_target_variance = full_hist(
        merged, "Data", scheme, "target_inclusive"
    )
    other_target, other_target_variance = full_hist(
        merged, "Other", scheme, "target_inclusive"
    )
    observation = data_target - other_target
    observation_variance = data_target_variance + other_target_variance
    metrics = closure_metrics(
        prediction,
        prediction_variance,
        observation,
        observation_variance,
    )
    old_name = VR_TO_OLD[scheme]
    old = top_w["validation_regions"][old_name]["split_top_w"]
    metrics.update(
        {
            "scheme": scheme,
            "regime": regime,
            "top_scale_factor": float(sf[0]),
            "w_scale_factor": float(sf[1]),
            "prediction_strategy": (
                "data one-lepton removal source minus Other-MC source, "
                "weighted by the fitted Top(TT+ST)/W mixture and their "
                "component residual lost/pass factors"
            ),
            "data_source": data_source.tolist(),
            "other_source": other_source.tolist(),
            "source_residual": source_residual.tolist(),
            "source_residual_variance": source_residual_variance.tolist(),
            "effective_lost_pass_factor": effective_factor.tolist(),
            "effective_lost_pass_factor_variance": (
                factor_variance.tolist()
            ),
            "mc_source_denominator": denominator.tolist(),
            "mc_target_numerator": numerator.tolist(),
            "old_tf_integrated_ratio": old["integrated"]["ratio"],
            "old_tf_maximum_absolute_pull": old["maximum_absolute_pull"],
            "distance_to_unity": (
                abs(float(metrics["ratio"]) - 1.0)
                if metrics["ratio"] is not None
                else None
            ),
            "old_tf_distance_to_unity": abs(
                float(old["integrated"]["ratio"]) - 1.0
            ),
        }
    )
    return metrics


def setup_style() -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 9.0),
            "axes.labelsize": 21,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "savefig.bbox": None,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_dir / "plots" / f"{stem}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def apply_labels(axis: Any) -> None:
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        ax=axis,
    )


def step_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(values) + 1, dtype=float)
    y = np.r_[values, values[-1] if len(values) else 0.0]
    return x, y


def plot_closure(
    result: dict[str, Any],
    labels: list[str],
    xlabel: str,
    output_dir: Path,
    stem: str,
    annotation: str,
    target_label: str = "Direct zero-lepton target",
) -> list[str]:
    prediction = np.asarray(result["prediction"], dtype=float)
    prediction_error = np.sqrt(
        np.clip(np.asarray(result["prediction_variance"], dtype=float), 0.0, None)
    )
    target = np.asarray(result["target"], dtype=float)
    target_error = np.sqrt(
        np.clip(np.asarray(result["target_variance"], dtype=float), 0.0, None)
    )
    x = np.arange(len(prediction), dtype=float)
    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 9.0),
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.16,
        top=0.88,
        hspace=0.05,
    )
    edges, step_prediction = step_values(prediction)
    _, step_low = step_values(prediction - prediction_error)
    _, step_high = step_values(prediction + prediction_error)
    axis.step(
        edges,
        step_prediction,
        where="post",
        color=COLORS["prediction"],
        linewidth=2.0,
        label="Lepton-removal prediction",
    )
    axis.fill_between(
        edges,
        step_low,
        step_high,
        step="post",
        color=COLORS["prediction"],
        alpha=0.22,
        linewidth=0,
        label="Stat. + normalization unc.",
    )
    axis.errorbar(
        x + 0.5,
        target,
        yerr=target_error,
        fmt="o",
        color=COLORS["target"],
        markersize=4.5,
        label=target_label,
    )
    axis.set_ylabel("Events")
    axis.set_xlim(0.0, float(len(prediction)))
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="upper right")
    axis.text(
        0.03,
        0.06,
        annotation,
        transform=axis.transAxes,
        fontsize=14,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.0,
        },
    )
    apply_labels(axis)

    ratio = np.divide(
        prediction,
        target,
        out=np.full_like(prediction, np.nan),
        where=target != 0.0,
    )
    ratio_variance = np.divide(
        np.asarray(result["prediction_variance"], dtype=float),
        np.square(target),
        out=np.full_like(prediction, np.nan),
        where=target != 0.0,
    )
    ratio_variance += np.divide(
        np.square(prediction) * np.asarray(result["target_variance"], dtype=float),
        np.power(target, 4),
        out=np.zeros_like(prediction),
        where=target != 0.0,
    )
    ratio_axis.errorbar(
        x + 0.5,
        ratio,
        yerr=np.sqrt(np.clip(ratio_variance, 0.0, None)),
        fmt="o",
        color=COLORS["prediction"],
        markersize=4.0,
    )
    ratio_axis.axhline(1.0, color="black", linewidth=1.0)
    ratio_axis.set_ylabel("Pred./target")
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.set_xlabel(xlabel)
    if len(labels) <= 10:
        ratio_axis.set_xticks(x + 0.5)
        ratio_axis.set_xticklabels(
            display_bin_labels(labels),
            rotation=35,
            ha="right",
        )
    else:
        spacing = 5 if len(labels) > 45 else 4
        ticks = np.arange(0, len(labels), spacing)
        ratio_axis.set_xticks(ticks + 0.5)
        ratio_axis.set_xticklabels([str(index + 1) for index in ticks])
    return save_figure(fig, output_dir, stem)


def plot_flavor_shapes(
    merged: dict[str, Any],
    component: str,
    scheme: str,
    output_dir: Path,
) -> list[str]:
    electron, _ = full_hist(
        merged, component, scheme, "prediction_post_e"
    )
    muon, _ = full_hist(
        merged, component, scheme, "prediction_post_mu"
    )
    electron_shape = (
        electron / np.sum(electron)
        if np.sum(electron) > 0.0
        else np.zeros_like(electron)
    )
    muon_shape = (
        muon / np.sum(muon)
        if np.sum(muon) > 0.0
        else np.zeros_like(muon)
    )
    fig, axis = plt.subplots(figsize=(9.0, 9.0))
    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.16,
        top=0.88,
    )
    for values, color, label in (
        (electron_shape, COLORS["electron"], "electron removal"),
        (muon_shape, COLORS["muon"], "muon removal"),
    ):
        edges, stepped = step_values(values)
        axis.step(
            edges,
            stepped,
            where="post",
            linewidth=2.2,
            color=color,
            label=label,
        )
    axis.set_xlim(0.0, float(len(electron)))
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Normalized events")
    axis.set_xlabel(XLABELS[scheme])
    labels = merged["schemes"][scheme]["labels"]
    if len(labels) <= 10:
        ticks = np.arange(len(labels), dtype=float) + 0.5
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            display_bin_labels(labels),
            rotation=35,
            ha="right",
        )
    axis.text(
        0.04,
        0.76,
        f"{component}: electron/muon shape test",
        transform=axis.transAxes,
        fontsize=14,
        va="top",
    )
    axis.legend(loc="upper right")
    apply_labels(axis)
    return save_figure(
        fig,
        output_dir,
        f"flavor_shapes_{component.lower()}_{scheme}",
    )


def flavor_shape_distance(
    merged: dict[str, Any],
    component: str,
    scheme: str,
) -> dict[str, Any]:
    electron, _ = full_hist(
        merged, component, scheme, "prediction_post_e"
    )
    muon, _ = full_hist(
        merged, component, scheme, "prediction_post_mu"
    )
    valid = (electron > 0.0) & (muon > 0.0)
    if not np.any(valid):
        return {
            "electron_integral": float(np.sum(electron)),
            "muon_integral": float(np.sum(muon)),
            "positive_bin_half_l1_shape_distance": None,
        }
    electron_shape = electron[valid] / np.sum(electron[valid])
    muon_shape = muon[valid] / np.sum(muon[valid])
    return {
        "electron_integral": float(np.sum(electron)),
        "muon_integral": float(np.sum(muon)),
        "positive_bin_half_l1_shape_distance": float(
            0.5 * np.sum(np.abs(electron_shape - muon_shape))
        ),
    }


def make_html(
    output_dir: Path,
    results: dict[str, Any],
    plot_paths: list[str],
) -> None:
    def formatted(value: Any, digits: int) -> str:
        return f"{value:.{digits}f}" if value is not None else "n/a"

    rows = []
    for scheme, result in results["data_validation"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(scheme)}</td>"
            f"<td>{formatted(result['old_tf_integrated_ratio'], 3)}</td>"
            f"<td>{formatted(result['ratio'], 3)}</td>"
            f"<td>{formatted(result['maximum_absolute_pull'], 2)}</td>"
            "</tr>"
        )
    normalization_rows = []
    for regime, components in results["normalizations"].items():
        for component, record in components.items():
            normalization_rows.append(
                "<tr>"
                f"<td>{html.escape(regime)}</td>"
                f"<td>{html.escape(component)}</td>"
                f"<td>{record['alpha']:.4f} ± "
                f"{math.sqrt(record['alpha_variance']):.4f}</td>"
                "</tr>"
            )
    mc_rows = []
    for scheme, record in results["mc_crossfit_closure"].items():
        removal = record["post"]
        old = record["old_tf_reference"]
        mc_rows.append(
            "<tr>"
            f"<td>{html.escape(scheme)}</td>"
            f"<td>{removal['chi2']:.1f}/{removal['ndf']}</td>"
            f"<td>{removal['maximum_absolute_pull']:.2f}</td>"
            f"<td>{old['diagonal_chi2']:.3f}/{old['diagonal_ndf']}</td>"
            f"<td>{old['maximum_absolute_pull']:.2f}</td>"
            "</tr>"
        )
    images = []
    for path in plot_paths:
        if not path.endswith(".png"):
            continue
        relative = Path(path).relative_to(output_dir)
        images.append(
            f'<a href="{html.escape(str(relative))}">'
            f'<img src="{html.escape(str(relative))}" loading="lazy"></a>'
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>2024 lost-lepton event-removal closure</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1250px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
h1, h2 {{ color: #17365d; }}
.warning {{ background: #fff3cd; border-left: 5px solid #d39e00; padding: 1rem; }}
.ok {{ background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ccd2d8; padding: .55rem; text-align: left; }}
th {{ background: #eef3f8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
.grid img {{ width: 100%; border: 1px solid #ccd2d8; }}
code {{ background: #f3f4f6; padding: .15rem .3rem; }}
</style>
</head>
<body>
<h1>2024 event-level lepton-removal closure</h1>
<p class="ok"><b>Component policy:</b> Top = TT + ST. W→ℓν is the only separately moving target component.</p>
<p class="warning"><b>Prototype scope:</b> the event migration and shape are tested. The residual normalization α is derived from MC because an absolute lepton-loss efficiency payload is not yet available. The current one-lepton flat materialization also requires the original pTmiss to exceed 250 GeV, so migration from below that threshold is absent. Nominal SR data remain blinded.</p>
<p class="warning"><b>Decision:</b> do not adopt this monolithic four-vector-removal estimator. Only {results['comparison']['validation_regions_improved_count']} of {results['comparison']['validation_regions_total']} data validation regions improves over the previous Top/W transfer-factor result, and the MC cross-fit fails the shape test despite its imposed integral normalization.</p>
<h2>Residual MC normalization after event migration</h2>
<table><thead><tr><th>Regime</th><th>Component</th><th>α</th></tr></thead>
<tbody>{''.join(normalization_rows)}</tbody></table>
<h2>MC shape closure after cross-fit normalization</h2>
<table><thead><tr><th>Distribution</th><th>Removal χ²/ndf</th><th>Removal max |pull|</th><th>Old TF χ²/ndf</th><th>Old TF max |pull|</th></tr></thead>
<tbody>{''.join(mc_rows)}</tbody></table>
<h2>Data validation-region comparison</h2>
<table><thead><tr><th>Validation region</th><th>Old TF pred./obs.</th><th>Removal pred./obs.</th><th>Removal max |pull|</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Plots</h2>
<div class="grid">{''.join(images)}</div>
<h2>Reproducibility</h2>
<ul>
<li>Machine-readable result: <a href="removal_closure_results.json">removal_closure_results.json</a></li>
<li>Input ROOT files were read-only; nominal inputs modified: <code>false</code>.</li>
<li>Nominal high-/low-Δm SR data target histograms were explicitly disabled.</li>
</ul>
</body></html>
"""
    (output_dir / "index.html").write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--old-mc-closure", required=True, type=Path)
    parser.add_argument("--top-w-closure", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    merged = load_json(args.merged)
    old_mc = load_json(args.old_mc_closure)
    top_w = load_json(args.top_w_closure)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    normalizations: dict[str, dict[str, Any]] = {}
    for regime, scheme in NOMINAL_BY_REGIME.items():
        normalizations[regime] = {
            component: full_normalization(
                merged,
                component,
                scheme,
                method="post",
                target_kind="target_leptonic",
            )
            for component in ("Top", "W")
        }

    mc_closure: dict[str, Any] = {}
    plot_paths: list[str] = []
    for scheme in ("highdm_ut", "highdm_search60", "lowdm_search42"):
        mc_closure[scheme] = {}
        for method in ("post", "strict"):
            components = [
                crossfit_component(
                    merged,
                    component,
                    scheme,
                    method,
                    flavor="combined",
                    target_kind="target_leptonic",
                )
                for component in ("Top", "W")
            ]
            combined = crossfit_combined(components)
            combined["components"] = {
                item["component"]: item for item in components
            }
            mc_closure[scheme][method] = combined
        old_scheme = OLD_MC_SCHEME[scheme]
        old_crossfit = old_mc["schemes"][old_scheme]["crossfit"]
        mc_closure[scheme]["old_tf_reference"] = {
            "maximum_absolute_pull": old_crossfit[
                "maximum_absolute_pull"
            ],
            "diagonal_chi2": old_crossfit["diagonal_chi2"],
            "diagonal_ndf": old_crossfit["diagonal_ndf"],
        }
        result = mc_closure[scheme]["post"]
        plot_paths.extend(
            plot_closure(
                result,
                merged["schemes"][scheme]["labels"],
                XLABELS[scheme],
                output_dir,
                f"mc_removal_closure_{scheme}",
                (
                    "Top = TT + ST; W separate\n"
                    "Integral normalized to target\n"
                    f"$\\chi^2$/ndf = {result['chi2']:.1f}/{result['ndf']}"
                ),
                target_label="Direct Top/W MC in SR (not data)",
            )
        )
        for component in ("Top", "W"):
            plot_paths.extend(
                plot_flavor_shapes(
                    merged, component, scheme, output_dir
                )
            )

    data_validation_results = {
        scheme: data_validation(
            merged, top_w, scheme, normalizations
        )
        for scheme in VR_TO_OLD
    }
    for scheme, result in data_validation_results.items():
        old_tf_ratio = result["old_tf_integrated_ratio"]
        removal_ratio = result["ratio"]
        old_tf_text = (
            f"{old_tf_ratio:.3f}" if old_tf_ratio is not None else "n/a"
        )
        removal_text = (
            f"{removal_ratio:.3f}" if removal_ratio is not None else "n/a"
        )
        plot_paths.extend(
            plot_closure(
                result,
                merged["schemes"][scheme]["labels"],
                XLABELS[scheme],
                output_dir,
                f"data_vr_removal_closure_{scheme}",
                (
                    "Data 1-lepton source − Other MC\n"
                    "Top = TT + ST; W separate\n"
                    f"Old TF: {old_tf_text}, removal: {removal_text}"
                ),
                target_label="Data - Other MC",
            )
        )

    flavor_results: dict[str, Any] = {}
    for component in ("Top", "W"):
        flavor_results[component] = {}
        for scheme in ("highdm_ut", "highdm_search60", "lowdm_search42"):
            flavor_results[component][scheme] = {
                flavor: crossfit_component(
                    merged,
                    component,
                    scheme,
                    "post",
                    flavor=flavor,
                )
                for flavor in ("e", "mu")
            }
    flavor_shape_distances = {
        component: {
            scheme: flavor_shape_distance(merged, component, scheme)
            for scheme in ("highdm_ut", "highdm_search60", "lowdm_search42")
        }
        for component in ("Top", "W", "Data")
    }
    for scheme in ("highdm_ut", "lowdm_search42"):
        plot_paths.extend(
            plot_flavor_shapes(merged, "Data", scheme, output_dir)
        )

    improved_vrs = [
        scheme
        for scheme, result in data_validation_results.items()
        if result["distance_to_unity"] is not None
        and result["old_tf_distance_to_unity"] is not None
        and result["distance_to_unity"]
        < result["old_tf_distance_to_unity"]
    ]
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "input_merged": str(args.merged.resolve()),
        "input_old_mc_closure": str(args.old_mc_closure.resolve()),
        "input_top_w_closure": str(args.top_w_closure.resolve()),
        "selection_authority": merged["selection_authority"],
        "nominal_inputs_modified": False,
        "data_nominal_sr_blinded": merged["data_nominal_sr_blinded"],
        "component_policy": merged["component_policy"],
        "input_totals": merged["input_totals"],
        "normalizations": normalizations,
        "mc_crossfit_closure": mc_closure,
        "flavor_crossfit": flavor_results,
        "flavor_shape_distances": flavor_shape_distances,
        "data_validation": data_validation_results,
        "comparison": {
            "validation_regions_improved_vs_old_tf": improved_vrs,
            "validation_regions_improved_count": len(improved_vrs),
            "validation_regions_total": len(data_validation_results),
            "adoption_gate": (
                "do_not_adopt_without_absolute_lepton_efficiency, a "
                "one-lepton materialization based on removal recoil rather "
                "than original MET, and additional closure review"
            ),
            "known_materialization_limit": (
                "current feature_flat_preselection requires original MET > "
                "250 GeV for ordinary one-lepton events, so events migrating "
                "from original MET below 250 GeV are unavailable"
            ),
        },
        "plots": plot_paths,
    }
    dump_json(output_dir / "removal_closure_results.json", results)
    make_html(output_dir, results, plot_paths)
    print(
        json.dumps(
            {
                "status": results["status"],
                "improved_vrs": improved_vrs,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
