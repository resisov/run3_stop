#!/usr/bin/env python3
"""Test a same-region shower-shape template estimate for jet-to-photon fakes.

The fake ``sigmaIetaIeta`` template is built from the charged-isolation-fail
data sideband in the same delta-phi topology in which the fake yield is fitted.
This removes the low-delta-phi to high-delta-phi fake-factor transfer.  A
deterministic two-fold MC cross-fit provides a non-tautological truth closure.

This is a diagnostic consumer of compact photon-template events.  It never
modifies nominal intermediate histograms.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import measure_photon_fake_template_2024 as core


OUTPUT_SCHEMA = "photon_fake_same_region_template_2024_v1"
REGIONS = (core.VALIDATION_REGION, core.APPLICATION_REGION)
REGION_LABELS = {
    core.VALIDATION_REGION: r"$0.30\leq\min\Delta\phi<0.50$ validation region",
    core.APPLICATION_REGION: r"Nominal GCR ($\min\Delta\phi\geq0.50$)",
}
COARSE_UT_EDGES = np.asarray([250.0, 400.0, 650.0, 1500.0])


def coarse_groups() -> list[core.Group]:
    return [
        core.Group(f"{eta}_pt220to400", eta, 220.0, 400.0, "coarse")
        for eta in ("EB", "EE")
    ] + [
        core.Group(f"{eta}_pt400toinf", eta, 400.0, 1_000_000.0, "coarse")
        for eta in ("EB", "EE")
    ]


def fixed_coarse_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for eta in ("EB", "EE"):
        for low, high in zip(core.FINE_PT_EDGES[:-1], core.FINE_PT_EDGES[1:]):
            fine = f"{eta}_pt{low:g}to{'inf' if high >= 1_000_000.0 else f'{high:g}'}"
            mapping[fine] = f"{eta}_pt220to400" if high <= 400.0 else f"{eta}_pt400toinf"
    return mapping


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_or_none(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def deterministic_fold(table: core.EventTable) -> np.ndarray:
    # Event identity only: stable across input ordering and repeated executions.
    key = (
        table.event.astype(np.uint64)
        ^ (table.lumi.astype(np.uint64) << np.uint64(21))
        ^ (table.run.astype(np.uint64) << np.uint64(42))
    )
    return (key & np.uint64(1)).astype(np.int8)


def mc_fake_template(
    table: core.EventTable,
    region: str,
    group: core.Group,
    train_mask: np.ndarray,
) -> dict[str, Any]:
    edges = core.sieie_edges(group.eta)
    selected = (
        core.base_mask(table, region, group)
        & train_mask
        & ~table.is_data
        & (table.origin == "fake")
        & (table.charged_level == 0)
    )
    values, variances = core.histogram(table.sieie[selected], table.weight[selected], edges)
    total = float(np.sum(table.weight[selected]))
    passed = float(np.sum(table.weight[selected & (table.shape_level >= 2)]))
    return {
        "edges": edges.tolist(),
        "shape": core.normalized_shape(values, alpha=0.5).tolist(),
        "pass_fraction": min(1.0, max(0.0, passed / total)) if total > 0.0 else 0.0,
        "weighted_yield": total,
        "raw_entries": int(np.count_nonzero(selected)),
        "effective_events": core.effective_events(values, variances),
    }


def crossfit_group(
    table: core.EventTable,
    region: str,
    group: core.Group,
    folds: np.ndarray,
    test_fold: int,
) -> dict[str, Any]:
    train = folds != test_fold
    test = folds == test_fold
    template = mc_fake_template(table, region, group, train)
    edges = np.asarray(template["edges"], dtype=float)
    group_region = core.base_mask(table, region, group)
    selected = (
        group_region
        & test
        & ~table.is_data
        & (table.charged_level >= 2)
    )
    observation, _ = core.histogram(table.sieie[selected], table.weight[selected], edges)
    prompt_source = (
        group_region
        & train
        & ~table.is_data
        & (table.origin == "prompt")
    )
    prompt_shape, prompt_variance = core.histogram(
        table.sieie[prompt_source], table.weight[prompt_source], edges
    )
    electron = selected & (table.origin == "electron")
    electron_hist, electron_variance = core.histogram(
        table.sieie[electron], table.weight[electron], edges
    )
    fit = core.extended_template_fit(
        observation,
        prompt_shape,
        np.asarray(template["shape"], dtype=float),
        electron_hist,
    )
    covariance = np.asarray(fit["covariance"], dtype=float)
    fraction = float(template["pass_fraction"])
    prediction = float(fit["fake_yield"]) * fraction
    prediction_variance = (
        max(0.0, float(covariance[1, 1])) * fraction * fraction
        if covariance.shape == (2, 2) and np.isfinite(covariance[1, 1])
        else 0.0
    )
    sideband = (
        group_region
        & test
        & ~table.is_data
        & (table.origin == "fake")
        & (table.charged_level == 0)
        & (table.shape_level >= 2)
    )
    sideband_yield, sideband_variance = core.histogram(
        table.ut[sideband], table.weight[sideband], core.UT_EDGES
    )
    sideband_total = float(np.sum(sideband_yield))
    sideband_shape = (
        sideband_yield / sideband_total
        if sideband_total > 0.0
        else np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    )
    prediction_by_ut = prediction * sideband_shape
    prediction_variance_by_ut = (
        prediction_variance * sideband_shape * sideband_shape
        + (
            prediction * prediction * sideband_variance / (sideband_total * sideband_total)
            if sideband_total > 0.0
            else 0.0
        )
    )
    truth_mask = selected & (table.origin == "fake") & (table.shape_level >= 2)
    truth, truth_variance = core.histogram(
        table.ut[truth_mask], table.weight[truth_mask], core.UT_EDGES
    )
    usable = bool(
        template["effective_events"] >= core.MIN_MC_TEMPLATE_EFFECTIVE_EVENTS
        and template["pass_fraction"] > 0.0
        and fit["success"]
        and float(np.sum(observation)) > 0.0
        and sideband_total > 0.0
        and core.effective_events(prompt_shape, prompt_variance)
        >= core.MIN_MC_TEMPLATE_EFFECTIVE_EVENTS
    )
    return {
        "test_fold": int(test_fold),
        "group": group.name,
        "template": template,
        "fit": fit,
        "fitted_fake_normalization": prediction,
        "fitted_fake_normalization_variance": prediction_variance,
        "sideband_ut_yield": sideband_yield.tolist(),
        "sideband_ut_variance": sideband_variance.tolist(),
        "sideband_ut_shape": sideband_shape.tolist(),
        "prediction": prediction_by_ut.tolist(),
        "prediction_variance": prediction_variance_by_ut.tolist(),
        "truth": truth.tolist(),
        "truth_variance": truth_variance.tolist(),
        "truth_raw_entries": int(np.count_nonzero(truth_mask)),
        "prompt_template_effective_events": core.effective_events(prompt_shape, prompt_variance),
        "electron_effective_events": core.effective_events(electron_hist, electron_variance),
        "usable": usable,
    }


def structural_group(
    table: core.EventTable,
    region: str,
    group: core.Group,
) -> dict[str, Any]:
    """Full-MC pass/fail structural check; not an independent closure test."""
    all_events = np.ones(table.n, dtype=bool)
    template = mc_fake_template(table, region, group, all_events)
    edges = np.asarray(template["edges"], dtype=float)
    group_region = core.base_mask(table, region, group)
    selected = group_region & ~table.is_data & (table.charged_level >= 2)
    observation, _ = core.histogram(table.sieie[selected], table.weight[selected], edges)
    prompt_source = selected & (table.origin == "prompt")
    prompt_shape, prompt_variance = core.histogram(
        table.sieie[prompt_source], table.weight[prompt_source], edges
    )
    electron = selected & (table.origin == "electron")
    electron_hist, electron_variance = core.histogram(
        table.sieie[electron], table.weight[electron], edges
    )
    fit = core.extended_template_fit(
        observation,
        prompt_shape,
        np.asarray(template["shape"], dtype=float),
        electron_hist,
    )
    covariance = np.asarray(fit["covariance"], dtype=float)
    fraction = float(template["pass_fraction"])
    normalization = float(fit["fake_yield"]) * fraction
    normalization_variance = (
        max(0.0, float(covariance[1, 1])) * fraction * fraction
        if covariance.shape == (2, 2) and np.isfinite(covariance[1, 1])
        else 0.0
    )
    sideband = (
        group_region
        & ~table.is_data
        & (table.origin == "fake")
        & (table.charged_level == 0)
        & (table.shape_level >= 2)
    )
    sideband_yield, sideband_variance = core.histogram(
        table.ut[sideband], table.weight[sideband], core.UT_EDGES
    )
    sideband_total = float(np.sum(sideband_yield))
    sideband_shape = (
        sideband_yield / sideband_total
        if sideband_total > 0.0
        else np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    )
    prediction = normalization * sideband_shape
    prediction_variance = (
        normalization_variance * sideband_shape * sideband_shape
        + (
            normalization * normalization * sideband_variance
            / (sideband_total * sideband_total)
            if sideband_total > 0.0
            else 0.0
        )
    )
    truth_mask = selected & (table.origin == "fake") & (table.shape_level >= 2)
    truth, truth_variance = core.histogram(
        table.ut[truth_mask], table.weight[truth_mask], core.UT_EDGES
    )
    usable = bool(
        template["effective_events"] >= core.MIN_MC_TEMPLATE_EFFECTIVE_EVENTS
        and template["pass_fraction"] > 0.0
        and fit["success"]
        and float(np.sum(observation)) > 0.0
        and sideband_total > 0.0
        and core.effective_events(prompt_shape, prompt_variance)
        >= core.MIN_MC_TEMPLATE_EFFECTIVE_EVENTS
    )
    return {
        "group": group.name,
        "template": template,
        "fit": fit,
        "fitted_fake_normalization": normalization,
        "fitted_fake_normalization_variance": normalization_variance,
        "prediction": prediction.tolist(),
        "prediction_variance": prediction_variance.tolist(),
        "truth": truth.tolist(),
        "truth_variance": truth_variance.tolist(),
        "truth_raw_entries": int(np.count_nonzero(truth_mask)),
        "electron_effective_events": core.effective_events(
            electron_hist, electron_variance
        ),
        "usable": usable,
    }


def rebin_closure(record: dict[str, Any], target_edges: np.ndarray) -> dict[str, Any]:
    source_edges = np.asarray(record["edges"], dtype=float)
    centers = 0.5 * (source_edges[:-1] + source_edges[1:])
    rebinned: dict[str, np.ndarray] = {
        name: np.zeros(len(target_edges) - 1, dtype=float)
        for name in ("prediction", "prediction_variance", "truth", "truth_variance")
    }
    for source_index, center in enumerate(centers):
        target_index = int(np.searchsorted(target_edges, center, side="right") - 1)
        if 0 <= target_index < len(target_edges) - 1:
            for name in rebinned:
                rebinned[name][target_index] += float(record[name][source_index])
    total_prediction = float(np.sum(rebinned["prediction"]))
    total_truth = float(np.sum(rebinned["truth"]))
    return {
        "edges": target_edges.tolist(),
        **{name: values.tolist() for name, values in rebinned.items()},
        "total_prediction": total_prediction,
        "total_truth": total_truth,
        "prediction_over_truth": total_prediction / total_truth if total_truth > 0.0 else None,
    }


def mc_crossfit_closure(table: core.EventTable, region: str) -> dict[str, Any]:
    folds = deterministic_fold(table)
    predictions = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    prediction_variances = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    truths = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    truth_variances = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    details: list[dict[str, Any]] = []
    usable_components = 0
    total_components = 0
    for group in coarse_groups():
        for test_fold in (0, 1):
            item = crossfit_group(table, region, group, folds, test_fold)
            predictions += np.asarray(item["prediction"], dtype=float)
            prediction_variances += np.asarray(item["prediction_variance"], dtype=float)
            truths += np.asarray(item["truth"], dtype=float)
            truth_variances += np.asarray(item["truth_variance"], dtype=float)
            usable_components += int(item["usable"])
            total_components += 1
            details.append(item)
    total_prediction = float(np.sum(predictions))
    total_truth = float(np.sum(truths))
    total_truth_variance = float(np.sum(truth_variances))
    result = {
        "edges": core.UT_EDGES.tolist(),
        "prediction": predictions.tolist(),
        "prediction_variance": prediction_variances.tolist(),
        "truth": truths.tolist(),
        "truth_variance": truth_variances.tolist(),
        "total_prediction": total_prediction,
        "total_truth": total_truth,
        "prediction_over_truth": total_prediction / total_truth if total_truth > 0.0 else None,
        "total_truth_effective_events": (
            total_truth * total_truth / total_truth_variance if total_truth_variance > 0.0 else 0.0
        ),
        "usable_components": usable_components,
        "total_components": total_components,
        "details": details,
    }
    result["coarse_ut_closure"] = rebin_closure(result, COARSE_UT_EDGES)
    return result


def mc_structural_closure(table: core.EventTable, region: str) -> dict[str, Any]:
    predictions = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    prediction_variances = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    truths = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    truth_variances = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    details = []
    usable_components = 0
    for group in coarse_groups():
        item = structural_group(table, region, group)
        predictions += np.asarray(item["prediction"], dtype=float)
        prediction_variances += np.asarray(item["prediction_variance"], dtype=float)
        truths += np.asarray(item["truth"], dtype=float)
        truth_variances += np.asarray(item["truth_variance"], dtype=float)
        usable_components += int(item["usable"])
        details.append(item)
    total_prediction = float(np.sum(predictions))
    total_truth = float(np.sum(truths))
    result = {
        "edges": core.UT_EDGES.tolist(),
        "prediction": predictions.tolist(),
        "prediction_variance": prediction_variances.tolist(),
        "truth": truths.tolist(),
        "truth_variance": truth_variances.tolist(),
        "total_prediction": total_prediction,
        "total_truth": total_truth,
        "prediction_over_truth": (
            total_prediction / total_truth if total_truth > 0.0 else None
        ),
        "usable_components": usable_components,
        "total_components": len(details),
        "details": details,
        "interpretation": (
            "diagnostic only: fake pass/fail samples are disjoint, but the prompt "
            "template and pseudodata share the same finite MC sample"
        ),
    }
    result["coarse_ut_closure"] = rebin_closure(result, COARSE_UT_EDGES)
    return result


def data_same_region_fit(table: core.EventTable, region: str) -> dict[str, Any]:
    prediction = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    prediction_variance = np.zeros(len(core.UT_EDGES) - 1, dtype=float)
    group_records: dict[str, Any] = {}
    for group in coarse_groups():
        template = core.fake_template(table, region, group)
        pass_fit = core.fit_stage(table, region, group, "pass_charged_iso", template)
        covariance = np.asarray(pass_fit["covariance"], dtype=float)
        fraction = float(template["pass_fraction"])
        fitted_normalization = float(pass_fit["fake_yield"]) * fraction
        fitted_normalization_variance = (
            max(0.0, float(covariance[1, 1])) * fraction * fraction
            if covariance.shape == (2, 2) and np.isfinite(covariance[1, 1])
            else 0.0
        )
        group_region = core.base_mask(table, region, group)
        sideband_base = (
            group_region & (table.charged_level == 0) & (table.shape_level >= 2)
        )
        data_sideband = sideband_base & table.is_data
        prompt_sideband = sideband_base & ~table.is_data & (table.origin == "prompt")
        electron_sideband = sideband_base & ~table.is_data & (table.origin == "electron")
        data_yield, data_variance = core.histogram(
            table.ut[data_sideband], np.ones(np.count_nonzero(data_sideband)), core.UT_EDGES
        )
        prompt_yield, prompt_variance = core.histogram(
            table.ut[prompt_sideband], table.weight[prompt_sideband], core.UT_EDGES
        )
        electron_yield, electron_variance = core.histogram(
            table.ut[electron_sideband], table.weight[electron_sideband], core.UT_EDGES
        )
        residual_before_clipping = data_yield - prompt_yield - electron_yield
        residual = np.maximum(0.0, residual_before_clipping)
        residual_variance = data_variance + prompt_variance + electron_variance
        residual_total = float(np.sum(residual))
        sideband_shape = (
            residual / residual_total
            if residual_total > 0.0
            else np.zeros(len(core.UT_EDGES) - 1, dtype=float)
        )
        group_prediction = fitted_normalization * sideband_shape
        group_variance = (
            fitted_normalization_variance * sideband_shape * sideband_shape
            + (
                fitted_normalization * fitted_normalization
                * residual_variance / (residual_total * residual_total)
                if residual_total > 0.0
                else 0.0
            )
        )
        prediction += group_prediction
        prediction_variance += group_variance
        pass_mask = (
            group_region & table.is_data & (table.charged_level >= 2)
        )
        group_records[group.name] = {
            "group": group.as_dict(),
            "fake_template": template,
            "pass_fit": pass_fit,
            "pass_data_events": int(np.count_nonzero(pass_mask)),
            "fitted_fake_normalization": fitted_normalization,
            "fitted_fake_normalization_variance": fitted_normalization_variance,
            "tight_shape_sideband": {
                "data": data_yield.tolist(),
                "prompt_contamination": prompt_yield.tolist(),
                "electron_contamination": electron_yield.tolist(),
                "residual_before_clipping": residual_before_clipping.tolist(),
                "residual": residual.tolist(),
                "variance": residual_variance.tolist(),
                "shape": sideband_shape.tolist(),
                "clipped_bins": int(np.count_nonzero(residual_before_clipping < 0.0)),
            },
            "statistically_usable": bool(
                template["data_events"] >= core.MIN_FAKE_TEMPLATE_EVENTS
                and int(np.count_nonzero(pass_mask)) >= core.MIN_DATA_FIT_EVENTS
                and template["pass_fraction"] > 0.0
                and pass_fit["success"]
                and residual_total > 0.0
            ),
        }
    return {
        "mapping": fixed_coarse_mapping(),
        "groups": group_records,
        "direct_template_fit": {
            "edges": core.UT_EDGES.tolist(),
            "yield": prediction.tolist(),
            "variance": prediction_variance.tolist(),
            "shape_source": "same-region tight-shape charged-isolation-fail data residual",
            "normalization_policy": "one integrated pass-region template fit per EB/EE and photon-pt group",
        },
    }


def tight_region_histograms(table: core.EventTable, region: str) -> dict[str, Any]:
    base = (
        table.region_masks[region]
        & (table.shape_level >= 2)
        & (table.charged_level >= 2)
    )
    result: dict[str, Any] = {"edges": core.UT_EDGES.tolist()}
    data, data_var = core.histogram(
        table.ut[base & table.is_data],
        np.ones(np.count_nonzero(base & table.is_data)),
        core.UT_EDGES,
    )
    result["data"] = data.tolist()
    result["data_variance"] = data_var.tolist()
    for origin in ("prompt", "electron", "fake"):
        selected = base & ~table.is_data & (table.origin == origin)
        values, variances = core.histogram(table.ut[selected], table.weight[selected], core.UT_EDGES)
        result[origin] = values.tolist()
        result[f"{origin}_variance"] = variances.tolist()
    return result


def integral_summary(histograms: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    data = float(np.sum(histograms["data"]))
    prompt = float(np.sum(histograms["prompt"]))
    electron = float(np.sum(histograms["electron"]))
    truth_fake = float(np.sum(histograms["fake"]))
    fitted_fake = float(np.sum(direct["yield"]))
    nominal = prompt + electron + truth_fake
    fitted = prompt + electron + fitted_fake
    return {
        "data": data,
        "prompt_mc": prompt,
        "electron_mc": electron,
        "truth_fake_mc": truth_fake,
        "direct_fitted_fake": fitted_fake,
        "nominal_mc_total": nominal,
        "same_region_template_total": fitted,
        "data_over_nominal_mc": data / nominal if nominal > 0.0 else None,
        "data_over_same_region_template": data / fitted if fitted > 0.0 else None,
    }


def plot_crossfit(
    record: dict[str, Any],
    region: str,
    path: Path,
    prediction_label: str = "2-fold template prediction",
    test_label: str = "Same-region MC cross-fit",
) -> None:
    edges = np.asarray(record["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    truth = np.asarray(record["truth"], dtype=float)
    truth_err = np.sqrt(np.maximum(0.0, np.asarray(record["truth_variance"], dtype=float)))
    prediction = np.asarray(record["prediction"], dtype=float)
    prediction_err = np.sqrt(
        np.maximum(0.0, np.asarray(record["prediction_variance"], dtype=float))
    )
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    ax.stairs(prediction, edges, color="#1f77b4", linewidth=2.0, label=prediction_label)
    ax.fill_between(
        edges,
        np.r_[np.maximum(1.0e-6, prediction - prediction_err), np.maximum(1.0e-6, prediction[-1] - prediction_err[-1])],
        np.r_[prediction + prediction_err, prediction[-1] + prediction_err[-1]],
        step="post", color="#1f77b4", alpha=0.22, linewidth=0,
    )
    ax.errorbar(centers, truth, yerr=truth_err, xerr=xerr, fmt="o", color="black", label="Truth fake MC")
    positive = np.r_[truth[truth > 0.0], prediction[prediction > 0.0]]
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(max(1.0e-3, float(np.min(positive)) * 0.2), float(np.max(positive)) * 20.0)
    ax.set_ylabel("Fake-photon yield")
    ax.legend(frameon=False, fontsize=12)
    ax.text(
        0.04,
        0.06,
        REGION_LABELS[region] + "\n" + test_label,
        transform=ax.transAxes,
        fontsize=13,
    )
    ratio = np.divide(prediction, truth, out=np.full_like(prediction, np.nan), where=truth > 0.0)
    ratio_err = np.divide(prediction_err, truth, out=np.full_like(prediction, np.nan), where=truth > 0.0)
    rax.axhline(1.0, color="0.35", linewidth=1.0)
    rax.errorbar(centers, ratio, yerr=ratio_err, xerr=xerr, fmt="o", color="#1f77b4")
    finite_ratio = ratio[np.isfinite(ratio) & (ratio >= 0.0)]
    ratio_upper = max(3.0, min(10.0, 1.25 * float(np.max(finite_ratio)))) if finite_ratio.size else 3.0
    rax.set_ylim(0.0, ratio_upper)
    rax.set_ylabel("Pred./truth")
    rax.set_xlabel(r"$U_T$ (GeV)", ha="right", x=1.0)
    ax.set_xlim(edges[0], edges[-1])
    rax.margins(x=0)
    hep.cms.label(llabel="Work in progress", rlabel="2024 (13.6 TeV)", ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_data_comparison(
    histograms: dict[str, Any],
    direct: dict[str, Any],
    region: str,
    path: Path,
    production_complete: bool,
) -> None:
    edges = np.asarray(histograms["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    data = np.asarray(histograms["data"], dtype=float)
    data_err = np.sqrt(np.maximum(data, 0.0))
    prompt = np.asarray(histograms["prompt"], dtype=float)
    electron = np.asarray(histograms["electron"], dtype=float)
    truth_fake = np.asarray(histograms["fake"], dtype=float)
    fitted_fake = np.asarray(direct["yield"], dtype=float)
    fitted_fake_err = np.sqrt(np.maximum(0.0, np.asarray(direct["variance"], dtype=float)))
    nominal = prompt + electron + truth_fake
    fitted = prompt + electron + fitted_fake
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    ax.stairs(nominal, edges, color="#7f3c8d", linewidth=2.0, label="Nominal MC")
    ax.stairs(fitted, edges, color="#11a579", linewidth=2.0, label="Same-region template replacement")
    ax.fill_between(
        edges,
        np.r_[np.maximum(1.0e-6, fitted - fitted_fake_err), np.maximum(1.0e-6, fitted[-1] - fitted_fake_err[-1])],
        np.r_[fitted + fitted_fake_err, fitted[-1] + fitted_fake_err[-1]],
        step="post", color="#11a579", alpha=0.20, linewidth=0,
    )
    data_label = "Data" if production_complete else "Partial EGamma data"
    ax.errorbar(centers, data, yerr=data_err, xerr=xerr, fmt="o", color="black", label=data_label)
    positive = np.r_[data[data > 0.0], nominal[nominal > 0.0], fitted[fitted > 0.0]]
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(max(1.0e-3, float(np.min(positive)) * 0.2), float(np.max(positive)) * 20.0)
    ax.set_ylabel("Events")
    ax.legend(frameon=False, fontsize=11)
    test_label = (
        "Full-production same-region test"
        if production_complete
        else "Incomplete-production workflow test"
    )
    ax.text(
        0.04,
        0.06,
        REGION_LABELS[region] + "\n" + test_label,
        transform=ax.transAxes,
        fontsize=12,
    )
    nominal_ratio = np.divide(data, nominal, out=np.full_like(data, np.nan), where=nominal > 0.0)
    fitted_ratio = np.divide(data, fitted, out=np.full_like(data, np.nan), where=fitted > 0.0)
    rax.axhline(1.0, color="0.35", linewidth=1.0)
    rax.plot(centers, nominal_ratio, "o-", color="#7f3c8d", label="Data/nominal")
    rax.plot(centers, fitted_ratio, "o-", color="#11a579", label="Data/template")
    rax.set_ylim(0.0, 2.5)
    rax.set_ylabel("Data/pred.")
    rax.set_xlabel(r"$U_T$ (GeV)", ha="right", x=1.0)
    ax.set_xlim(edges[0], edges[-1])
    rax.margins(x=0)
    hep.cms.label(llabel="Work in progress", rlabel="2024 (13.6 TeV)", ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_report(payload: dict[str, Any], output: Path) -> None:
    production_complete = bool(payload["production_complete"])
    status_text = (
        "complete full-production evaluation"
        if production_complete
        else "incomplete-production diagnostic; no adoption decision"
    )
    rows = []
    for region in REGIONS:
        record = payload["regions"][region]
        summary = record["integral_summary"]
        closure = record["mc_crossfit_closure"]
        structural = record["mc_structural_closure"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(region)}</td>"
            f"<td>{summary['data']:.6g}</td>"
            f"<td>{summary['nominal_mc_total']:.6g}</td>"
            f"<td>{summary['same_region_template_total']:.6g}</td>"
            f"<td>{summary['direct_fitted_fake']:.6g}</td>"
            f"<td>{closure['prediction_over_truth']}</td>"
            f"<td>{structural['prediction_over_truth']}</td>"
            f"<td>{closure['total_truth_effective_events']:.3g}</td>"
            f"<td>{closure['usable_components']}/{closure['total_components']}</td>"
            "</tr>"
        )
    group_rows = []
    for region in REGIONS:
        for name, record in payload["regions"][region]["data_same_region_fit"]["groups"].items():
            template = record["fake_template"]
            group_rows.append(
                "<tr>"
                f"<td>{html.escape(region)}</td><td>{html.escape(name)}</td>"
                f"<td>{template['data_events']}</td><td>{record['pass_data_events']}</td>"
                f"<td>{template['pass_fraction']:.4g}</td><td>{template['clipped_bins']}</td>"
                f"<td>{record['statistically_usable']}</td></tr>"
            )
    images = []
    for region in REGIONS:
        slug = "highvr" if region == core.VALIDATION_REGION else "gcr"
        images.extend(
            [
                f'<h2>{html.escape(REGION_LABELS[region])}: MC cross-fit</h2><img src="plots/mc_crossfit_{slug}_ut_coarse.png"><details><summary>Fine U_T bins</summary><img src="plots/mc_crossfit_{slug}_ut.png"></details>',
                f'<h2>{html.escape(REGION_LABELS[region])}: structural diagnostic</h2><img src="plots/mc_structural_{slug}_ut_coarse.png">',
                f'<h2>{html.escape(REGION_LABELS[region])}: data/MC diagnostic</h2><img src="plots/data_comparison_{slug}_ut.png">',
            ]
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Same-region photon-fake template test</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}img{{width:min(100%,820px)}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eee;padding:.1rem .25rem}}</style></head><body>
<h1>Same-region photon-fake template workflow test</h1>
<p><strong>Status: {html.escape(status_text)}.</strong></p>
<p><strong>Decision: {html.escape(payload['adoption']['reason'])}.</strong></p>
<p>The fake shower-shape template and the pass-region fit use the same delta-phi topology but disjoint charged-isolation states. One integrated fake normalization is fitted per EB/EE and photon-pT group; the U_T shape comes from the tight-shape charged-isolation-fail data residual and is not fitted independently in every U_T bin. The MC closure is a deterministic two-fold cross-fit, so no event is used both to build its template and to test it.</p>
<p>Inputs: {payload['input_audit']['accepted_files']} compact files and {payload['event_count_after_deduplication']} events. Nominal intermediates were not modified.</p>
<table><thead><tr><th>Region</th><th>Data</th><th>Nominal MC</th><th>Template total</th><th>Fitted fake</th><th>2-fold MC pred./truth</th><th>Structural MC pred./truth</th><th>Truth Neff</th><th>Usable fit pieces</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Sideband statistics</h2><table><thead><tr><th>Region</th><th>Group</th><th>Fail-iso data</th><th>Pass-iso data</th><th>Tight-shape fraction</th><th>Clipped bins</th><th>Usable</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table>
{''.join(images)}
<h2>Interpretation guardrails</h2>
<ul><li>All production audits must be complete before an adoption decision is allowed.</li><li>A method is not accepted merely because its direct fit moves Data/MC toward unity.</li><li>Both integral and coarse-U_T shape closure are required; an accidentally closing integral is insufficient.</li><li>Adoption requires stable sideband templates, independent closure, and improved prefit GCR shape.</li></ul>
<p>Detailed report: <a href="report.md"><code>report.md</code></a></p>
<p>Machine-readable output: <a href="evaluation.json"><code>evaluation.json</code></a></p>
</body></html>"""
    (output / "index.html").write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--production-audits",
        nargs="*",
        type=Path,
        default=[],
        help="complete checksum audits for main non-DY, replacement DY2x, and rare",
    )
    args = parser.parse_args()
    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    events, input_audit = core.load_inputs(sorted(set(paths)))
    events, deduplication = core.deduplicate_data(events)
    normalization = core.read_payload(args.normalization)
    if normalization.get("status") != "complete":
        raise RuntimeError(f"normalization is not complete: {normalization.get('status')}")
    table = core.EventTable(events, normalization)
    production_audits = [core.read_payload(path) for path in args.production_audits]
    production_complete = bool(
        len(production_audits) == 3
        and all(
            audit.get("status") == "complete"
            and int(audit.get("valid_outputs", -1))
            == int(audit.get("expected_jobs", -2))
            and int(audit.get("missing_outputs", -1)) == 0
            and int(audit.get("invalid_outputs", -1)) == 0
            for audit in production_audits
        )
    )
    process_counts = {
        str(name): int(np.count_nonzero(table.process == name))
        for name in sorted(set(table.process.tolist()))
    }
    region_results: dict[str, Any] = {}
    for region in REGIONS:
        data_fit = data_same_region_fit(table, region)
        histograms = tight_region_histograms(table, region)
        closure = mc_crossfit_closure(table, region)
        structural = mc_structural_closure(table, region)
        region_results[region] = {
            "data_same_region_fit": data_fit,
            "tight_region_histograms": histograms,
            "mc_crossfit_closure": closure,
            "mc_structural_closure": structural,
            "integral_summary": integral_summary(histograms, data_fit["direct_template_fit"]),
        }
    integral_gate = True
    shape_gate = True
    all_components_usable = True
    data_mc_gate = True
    for region in REGIONS:
        closure = region_results[region]["mc_crossfit_closure"]
        ratio = closure.get("prediction_over_truth")
        integral_gate = bool(
            integral_gate
            and ratio is not None
            and ratio > 0.0
            and abs(math.log(float(ratio))) < 0.35
        )
        coarse = closure["coarse_ut_closure"]
        for prediction, truth in zip(coarse["prediction"], coarse["truth"]):
            if float(truth) > 0.0:
                bin_ratio = float(prediction) / float(truth)
                shape_gate = bool(
                    shape_gate
                    and bin_ratio > 0.0
                    and abs(math.log(bin_ratio)) < math.log(2.0)
                )
        all_components_usable = bool(
            all_components_usable
            and int(closure["usable_components"]) == int(closure["total_components"])
        )
        summary = region_results[region]["integral_summary"]
        data_mc_gate = bool(
            data_mc_gate
            and abs(1.0 - float(summary["data_over_same_region_template"]))
            < abs(1.0 - float(summary["data_over_nominal_mc"]))
        )
    adoption_allowed = bool(
        production_complete
        and integral_gate
        and shape_gate
        and all_components_usable
        and data_mc_gate
    )
    if adoption_allowed:
        adoption_reason = "adopt: full-production integral, shape, stability, and prefit Data/MC gates passed"
    elif not production_complete:
        adoption_reason = "do not adopt: production audits are incomplete"
    else:
        adoption_reason = "do not adopt: one or more closure, stability, or prefit Data/MC gates failed"
    payload = finite_or_none(
        {
            "schema_version": OUTPUT_SCHEMA,
            "status": "complete" if production_complete else "incomplete_production_diagnostic",
            "production_complete": production_complete,
            "production_audits": production_audits,
            "method": {
                "fit_observable": "Photon_sieie",
                "fake_template": "same-region charged-isolation-fail data after prompt/electron subtraction",
                "prompt_template": "same-region prompt-photon MC",
                "electron_component": "same-region fixed electron-origin MC",
                "binning": "EB/EE x photon pT 220-400 and >=400 GeV",
                "ut_shape": "same-region tight-shape charged-isolation-fail residual normalized to the integrated pass-region fit",
                "mc_validation": "deterministic two-fold event-identity cross-fit",
                "selection_source": "real_subset_worker.py via photon_fake_template_2024_worker.py",
                "nominal_intermediate_mutation": False,
            },
            "normalization": {
                "source": str(args.normalization),
                "luminosity_pb": normalization.get("luminosity_pb"),
                "policy": "nominal_weight_without_photon_id_sf times physical-dataset normalization exactly once",
            },
            "input_audit": input_audit,
            "deduplication": deduplication,
            "event_count_after_deduplication": table.n,
            "compact_event_process_counts": process_counts,
            "regions": region_results,
            "adoption": {
                "allowed": adoption_allowed,
                "reason": adoption_reason,
                "gates": {
                    "production_complete": production_complete,
                    "integral_closure": integral_gate,
                    "coarse_ut_shape_closure_within_factor_two": shape_gate,
                    "all_fit_components_usable": all_components_usable,
                    "prefit_data_mc_integral_improves": data_mc_gate,
                },
            },
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    core.write_payload(args.output / "evaluation.json", payload)
    for region in REGIONS:
        slug = "highvr" if region == core.VALIDATION_REGION else "gcr"
        plot_crossfit(
            payload["regions"][region]["mc_crossfit_closure"],
            region,
            args.output / "plots" / f"mc_crossfit_{slug}_ut",
        )
        plot_crossfit(
            payload["regions"][region]["mc_crossfit_closure"]["coarse_ut_closure"],
            region,
            args.output / "plots" / f"mc_crossfit_{slug}_ut_coarse",
        )
        plot_crossfit(
            payload["regions"][region]["mc_structural_closure"]["coarse_ut_closure"],
            region,
            args.output / "plots" / f"mc_structural_{slug}_ut_coarse",
            prediction_label="Full-MC structural prediction",
            test_label="Pass/fail structural diagnostic",
        )
        plot_data_comparison(
            payload["regions"][region]["tight_region_histograms"],
            payload["regions"][region]["data_same_region_fit"]["direct_template_fit"],
            region,
            args.output / "plots" / f"data_comparison_{slug}_ut",
            production_complete,
        )
    render_report(payload, args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "events": table.n,
                "regions": {
                    region: payload["regions"][region]["mc_crossfit_closure"]["prediction_over_truth"]
                    for region in REGIONS
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
