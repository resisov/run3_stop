#!/usr/bin/env python3
"""Rebuild the 2024 lost-lepton closure with separate Top and W components.

This script consumes the additive merged histogram JSON produced by
``run_lost_lepton_closure_2024.py``.  It never reads or modifies the nominal
ROOT intermediates.  Top means TT + single-top.  Independent Top and W
normalizations are solved from W- and Top-enriched one-lepton control
categories and propagated, with their covariance, to zero-lepton validation
regions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2 as chi2_distribution  # noqa: E402


TOP_PROCESSES = ("TT", "ST")
W_PROCESSES = ("WtoLNu",)
DATA_PROCESS = "JetMET"
OTHER_PROCESSES = ("Zto2Nu", "DY", "GJ", "VV", "QCD")

HIGHDM_FIT_CATEGORIES = (
    {
        "name": "highdm_nb0",
        "label": r"$N_{\mathrm{b}}=0$ (W enriched)",
        "region": "validation_regions",
        "indices": None,
    },
    {
        "name": "highdm_njet3to4_nb1plus",
        "label": r"$3\leq N_{\mathrm{j}}\leq4,\ N_{\mathrm{b}}\geq1$ "
        r"(Top enriched)",
        "region": "validation_regions",
        "indices": None,
    },
)

LOWDM_FIT_CATEGORIES = (
    {
        "name": "lowdm_search42",
        "label": r"$N_{\mathrm{b}}=0$ (W enriched)",
        "region": "histograms",
        "indices": tuple(range(0, 8)),
    },
    {
        "name": "lowdm_search42",
        "label": r"$N_{\mathrm{b}}\geq2$ (Top enriched)",
        "region": "histograms",
        "indices": tuple(range(24, 42)),
    },
)

LOWDM_NB1_VALIDATION = {
    "name": "lowdm_search42",
    "label": r"$N_{\mathrm{b}}=1$ (not fitted)",
    "region": "histograms",
    "indices": tuple(range(8, 24)),
}

VR_ANNOTATIONS = {
    "highdm_nb0": (
        r"High-$\Delta m$ validation region",
        r"$N_{\mathrm{b}}=0$",
    ),
    "highdm_njet3to4_nb1plus": (
        r"High-$\Delta m$ validation region",
        r"$3\leq N_{\mathrm{j}}\leq4,\ N_{\mathrm{b}}\geq1$",
    ),
    "lowdm_met250to300": (
        r"Low-$\Delta m$ validation region",
        r"$250 < \mathrm{p}_{\mathrm{T}}^{\mathrm{miss}} < 300$ GeV",
    ),
    "lowdm_isr200to300": (
        r"Low-$\Delta m$ validation region",
        r"$200 < \mathrm{p}_{\mathrm{T}}^{\mathrm{ISR}} < 300$ GeV",
    ),
    "lowdm_significance7to10": (
        r"Low-$\Delta m$ validation region",
        r"$7 < S_{\mathrm{MET}} < 10$",
    ),
}

VR_XLABELS = {
    "highdm_nb0": r"$\mathrm{p}_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)",
    "highdm_njet3to4_nb1plus": (
        r"$\mathrm{p}_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)"
    ),
    "lowdm_met250to300": (
        r"$\mathrm{p}_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)"
    ),
    "lowdm_isr200to300": r"$\mathrm{p}_{\mathrm{T}}^{\mathrm{ISR}}$ (GeV)",
    "lowdm_significance7to10": r"$S_{\mathrm{MET}}$",
}


def json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def finite_list(values: np.ndarray) -> list[float | None]:
    return [finite_or_none(value) for value in np.asarray(values, dtype=float)]


def matrix_list(values: np.ndarray) -> list[list[float | None]]:
    return [finite_list(row) for row in np.asarray(values, dtype=float)]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def hist_for_process(
    merged: dict[str, Any],
    process: str,
    region: str,
    name: str,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    process_record = merged["processes"][process]
    if region == "histograms":
        records = [
            process_record[region][name][fold][side] for fold in ("0", "1")
        ]
        sumw = sum(
            (np.asarray(record["sumw"], dtype=float) for record in records),
            np.zeros(len(records[0]["sumw"]), dtype=float),
        )
        sumw2 = sum(
            (np.asarray(record["sumw2"], dtype=float) for record in records),
            np.zeros(len(records[0]["sumw2"]), dtype=float),
        )
        entries = sum(
            (
                np.asarray(record["entries"], dtype=np.int64)
                for record in records
            ),
            np.zeros(len(records[0]["entries"]), dtype=np.int64),
        )
        return sumw, sumw2, entries
    record = process_record[region][name][side]
    return (
        np.asarray(record["sumw"], dtype=float),
        np.asarray(record["sumw2"], dtype=float),
        np.asarray(record["entries"], dtype=np.int64),
    )


def hist_for_processes(
    merged: dict[str, Any],
    processes: tuple[str, ...],
    region: str,
    name: str,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = [
        hist_for_process(merged, process, region, name, side)
        for process in processes
    ]
    return (
        sum(
            (record[0] for record in records),
            np.zeros_like(records[0][0]),
        ),
        sum(
            (record[1] for record in records),
            np.zeros_like(records[0][1]),
        ),
        sum(
            (record[2] for record in records),
            np.zeros_like(records[0][2]),
        ),
    )


def select_and_sum(
    values: np.ndarray, indices: tuple[int, ...] | None
) -> float:
    if indices is None:
        return float(np.sum(values))
    return float(np.sum(values[np.asarray(indices, dtype=int)]))


def fit_equation(
    merged: dict[str, Any], specification: dict[str, Any]
) -> dict[str, Any]:
    region = str(specification["region"])
    name = str(specification["name"])
    indices = specification["indices"]
    data, data_var, _ = hist_for_process(
        merged, DATA_PROCESS, region, name, "control"
    )
    other, other_var, _ = hist_for_processes(
        merged, OTHER_PROCESSES, region, name, "control"
    )
    top, top_var, _ = hist_for_processes(
        merged, TOP_PROCESSES, region, name, "control"
    )
    wjets, wjets_var, _ = hist_for_processes(
        merged, W_PROCESSES, region, name, "control"
    )
    return {
        "name": name,
        "label": specification["label"],
        "data": select_and_sum(data, indices),
        "data_variance": select_and_sum(data_var, indices),
        "other": select_and_sum(other, indices),
        "other_variance": select_and_sum(other_var, indices),
        "top": select_and_sum(top, indices),
        "top_variance": select_and_sum(top_var, indices),
        "wjets": select_and_sum(wjets, indices),
        "wjets_variance": select_and_sum(wjets_var, indices),
    }


def solve_two_component_fit(
    merged: dict[str, Any],
    specifications: tuple[dict[str, Any], ...],
    regime: str,
) -> dict[str, Any]:
    equations = [fit_equation(merged, spec) for spec in specifications]
    design = np.asarray(
        [[record["top"], record["wjets"]] for record in equations],
        dtype=float,
    )
    residual = np.asarray(
        [record["data"] - record["other"] for record in equations],
        dtype=float,
    )
    if design.shape != (2, 2):
        raise RuntimeError(f"{regime}: expected a 2x2 design matrix")
    condition_number = float(np.linalg.cond(design))
    if not math.isfinite(condition_number) or condition_number > 1.0e6:
        raise RuntimeError(
            f"{regime}: ill-conditioned Top/W fit ({condition_number:g})"
        )
    scale_factors = np.linalg.solve(design, residual)
    if np.any(scale_factors < 0.0):
        raise RuntimeError(
            f"{regime}: negative scale factor solution {scale_factors}"
        )
    equation_variance = np.asarray(
        [
            record["data_variance"]
            + record["other_variance"]
            + np.square(scale_factors[0]) * record["top_variance"]
            + np.square(scale_factors[1]) * record["wjets_variance"]
            for record in equations
        ],
        dtype=float,
    )
    inverse_design = np.linalg.inv(design)
    covariance = (
        inverse_design
        @ np.diag(equation_variance)
        @ inverse_design.T
    )
    correlation = covariance[0, 1] / math.sqrt(
        covariance[0, 0] * covariance[1, 1]
    )
    for index, record in enumerate(equations):
        record["fitted_target_residual"] = float(
            design[index] @ scale_factors
        )
    return {
        "regime": regime,
        "components": ["Top (TT + ST)", "W+jets"],
        "scale_factors": finite_list(scale_factors),
        "scale_factor_covariance": matrix_list(covariance),
        "scale_factor_uncertainties": finite_list(
            np.sqrt(np.diag(covariance))
        ),
        "scale_factor_correlation": finite_or_none(correlation),
        "design_condition_number": finite_or_none(condition_number),
        "fit_strategy": (
            "Two enriched one-lepton control yields solve exactly for "
            "Top and W normalizations; covariance includes data, other-MC, "
            "Top-MC, and W-MC statistical variances."
        ),
        "equations": equations,
    }


def ratio_statistics(
    prediction: np.ndarray,
    prediction_covariance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    observation_variance = np.asarray(observation_variance, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    ratio = np.divide(
        prediction,
        observation,
        out=np.full(len(prediction), np.nan),
        where=valid & (observation != 0.0),
    )
    diagonal_prediction_variance = np.diag(prediction_covariance)
    ratio_variance = np.full(len(prediction), np.nan)
    ratio_variance[valid] = (
        diagonal_prediction_variance[valid]
        / np.square(observation[valid])
        + np.square(prediction[valid])
        * observation_variance[valid]
        / np.power(observation[valid], 4)
    )
    total_covariance = prediction_covariance + np.diag(
        observation_variance
    )
    pull = np.full(len(prediction), np.nan)
    diagonal_total = np.diag(total_covariance)
    pull[valid] = (
        prediction[valid] - observation[valid]
    ) / np.sqrt(diagonal_total[valid])
    indices = np.flatnonzero(valid)
    if len(indices):
        selected_covariance = total_covariance[np.ix_(indices, indices)]
        difference = prediction[indices] - observation[indices]
        chi2 = float(
            difference
            @ np.linalg.pinv(selected_covariance, hermitian=True)
            @ difference
        )
        diagonal_chi2 = float(
            np.sum(np.square(difference) / np.diag(selected_covariance))
        )
        ndf = int(len(indices))
        p_value = float(chi2_distribution.sf(chi2, ndf))
        maximum_pull = float(np.max(np.abs(pull[valid])))
    else:
        chi2 = math.nan
        diagonal_chi2 = math.nan
        ndf = 0
        p_value = math.nan
        maximum_pull = math.nan
    return {
        "prediction": finite_list(prediction),
        "prediction_covariance": matrix_list(prediction_covariance),
        "prediction_variance": finite_list(diagonal_prediction_variance),
        "observation": finite_list(observation),
        "observation_variance": finite_list(observation_variance),
        "valid": valid.tolist(),
        "ratio": finite_list(ratio),
        "ratio_variance": finite_list(ratio_variance),
        "pull": finite_list(pull),
        "chi2": finite_or_none(chi2),
        "diagonal_chi2": finite_or_none(diagonal_chi2),
        "ndf": ndf,
        "p_value": finite_or_none(p_value),
        "maximum_absolute_pull": finite_or_none(maximum_pull),
        "full_statistical_covariance_used": True,
    }


def integrated_ratio(
    prediction: np.ndarray,
    prediction_covariance: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {
            "prediction": None,
            "observation": None,
            "ratio": None,
            "ratio_uncertainty": None,
        }
    prediction_sum = float(np.sum(prediction[indices]))
    observation_sum = float(np.sum(observation[indices]))
    prediction_variance = float(
        np.sum(prediction_covariance[np.ix_(indices, indices)])
    )
    observation_variance_sum = float(
        np.sum(observation_variance[indices])
    )
    ratio = prediction_sum / observation_sum
    ratio_variance = (
        prediction_variance / np.square(observation_sum)
        + np.square(prediction_sum)
        * observation_variance_sum
        / np.power(observation_sum, 4)
    )
    return {
        "prediction": prediction_sum,
        "prediction_variance": prediction_variance,
        "observation": observation_sum,
        "observation_variance": observation_variance_sum,
        "ratio": ratio,
        "ratio_uncertainty": math.sqrt(max(ratio_variance, 0.0)),
    }


def combined_statistics_on_mask(
    combined_record: dict[str, Any],
    observation: np.ndarray,
    observation_variance: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(
        combined_record["predicted_lost_lepton"], dtype=float
    )
    prediction_variance = np.asarray(
        combined_record["predicted_lost_lepton_variance"], dtype=float
    )
    covariance = np.diag(prediction_variance)
    output = ratio_statistics(
        prediction,
        covariance,
        observation,
        observation_variance,
        valid,
    )
    output["integrated"] = integrated_ratio(
        prediction,
        covariance,
        observation,
        observation_variance,
        valid,
    )
    return output


def build_target_closure(
    merged: dict[str, Any],
    combined: dict[str, Any],
    name: str,
    fit: dict[str, Any],
) -> dict[str, Any]:
    top, top_var, top_entries = hist_for_processes(
        merged, TOP_PROCESSES, "validation_regions", name, "target"
    )
    wjets, wjets_var, wjets_entries = hist_for_processes(
        merged, W_PROCESSES, "validation_regions", name, "target"
    )
    other, other_var, _ = hist_for_processes(
        merged, OTHER_PROCESSES, "validation_regions", name, "target"
    )
    data, data_var, data_entries = hist_for_process(
        merged, DATA_PROCESS, "validation_regions", name, "target"
    )
    scale = np.asarray(fit["scale_factors"], dtype=float)
    scale_covariance = np.asarray(
        fit["scale_factor_covariance"], dtype=float
    )
    component_design = np.column_stack((top, wjets))
    prediction = component_design @ scale
    prediction_covariance = (
        np.diag(
            np.square(scale[0]) * top_var
            + np.square(scale[1]) * wjets_var
        )
        + component_design @ scale_covariance @ component_design.T
    )
    observation = data - other
    observation_variance = data_var + other_var
    target_sumw = top + wjets
    target_sumw2 = top_var + wjets_var
    target_neff = np.divide(
        np.square(target_sumw),
        target_sumw2,
        out=np.zeros(len(target_sumw)),
        where=target_sumw2 > 0.0,
    )
    original_valid = np.asarray(
        combined["validation_regions"][name][
            "statistically_sufficient"
        ],
        dtype=bool,
    )
    valid = (
        original_valid
        & np.isfinite(prediction)
        & (prediction > 0.0)
        & (observation > 0.0)
        & (target_neff >= 10.0)
    )
    split_statistics = ratio_statistics(
        prediction,
        prediction_covariance,
        observation,
        observation_variance,
        valid,
    )
    split_statistics["integrated"] = integrated_ratio(
        prediction,
        prediction_covariance,
        observation,
        observation_variance,
        valid,
    )
    combined_statistics = combined_statistics_on_mask(
        combined["validation_regions"][name],
        observation,
        observation_variance,
        valid,
    )
    return {
        "name": name,
        "labels": list(
            combined["validation_regions"][name]["labels"]
        ),
        "fit_regime": fit["regime"],
        "valid_bins": int(np.count_nonzero(valid)),
        "valid_mask": valid.tolist(),
        "top_target": finite_list(top),
        "top_target_variance": finite_list(top_var),
        "top_target_entries": top_entries.tolist(),
        "w_target": finite_list(wjets),
        "w_target_variance": finite_list(wjets_var),
        "w_target_entries": wjets_entries.tolist(),
        "other_mc_target": finite_list(other),
        "data_target": finite_list(data),
        "data_target_entries": data_entries.tolist(),
        "target_residual": finite_list(observation),
        "target_residual_variance": finite_list(observation_variance),
        "split_top_w": split_statistics,
        "combined_baseline": combined_statistics,
        "integrated_ratio_change": finite_or_none(
            split_statistics["integrated"]["ratio"]
            - combined_statistics["integrated"]["ratio"]
        ),
    }


def build_control_shape(
    merged: dict[str, Any],
    name: str,
    fit: dict[str, Any],
) -> dict[str, Any]:
    top, top_var, _ = hist_for_processes(
        merged, TOP_PROCESSES, "validation_regions", name, "control"
    )
    wjets, wjets_var, _ = hist_for_processes(
        merged, W_PROCESSES, "validation_regions", name, "control"
    )
    other, other_var, _ = hist_for_processes(
        merged, OTHER_PROCESSES, "validation_regions", name, "control"
    )
    data, data_var, _ = hist_for_process(
        merged, DATA_PROCESS, "validation_regions", name, "control"
    )
    scale = np.asarray(fit["scale_factors"], dtype=float)
    covariance = np.asarray(fit["scale_factor_covariance"], dtype=float)
    design = np.column_stack((top, wjets))
    prediction = design @ scale
    prediction_covariance = (
        np.diag(
            np.square(scale[0]) * top_var
            + np.square(scale[1]) * wjets_var
        )
        + design @ covariance @ design.T
    )
    observation = data - other
    observation_variance = data_var + other_var
    valid = (
        np.isfinite(prediction)
        & (prediction > 0.0)
        & (observation > 0.0)
    )
    output = ratio_statistics(
        prediction,
        prediction_covariance,
        observation,
        observation_variance,
        valid,
    )
    output["integrated"] = integrated_ratio(
        prediction,
        prediction_covariance,
        observation,
        observation_variance,
        valid,
    )
    return output


def build_lowdm_nb_validation(
    merged: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    groups = (
        LOWDM_FIT_CATEGORIES[0],
        LOWDM_NB1_VALIDATION,
        LOWDM_FIT_CATEGORIES[1],
    )
    scale = np.asarray(fit["scale_factors"], dtype=float)
    covariance = np.asarray(fit["scale_factor_covariance"], dtype=float)
    labels: list[str] = []
    top_values: list[float] = []
    top_variances: list[float] = []
    w_values: list[float] = []
    w_variances: list[float] = []
    observations: list[float] = []
    observation_variances: list[float] = []
    for specification in groups:
        equation = fit_equation(merged, specification)
        labels.append(str(equation["label"]))
        top_values.append(float(equation["top"]))
        top_variances.append(float(equation["top_variance"]))
        w_values.append(float(equation["wjets"]))
        w_variances.append(float(equation["wjets_variance"]))
        observations.append(float(equation["data"] - equation["other"]))
        observation_variances.append(
            float(
                equation["data_variance"]
                + equation["other_variance"]
            )
        )
    design = np.column_stack((top_values, w_values))
    prediction = design @ scale
    prediction_covariance = (
        np.diag(
            np.square(scale[0]) * np.asarray(top_variances)
            + np.square(scale[1]) * np.asarray(w_variances)
        )
        + design @ covariance @ design.T
    )
    observation = np.asarray(observations)
    observation_variance = np.asarray(observation_variances)
    valid = (prediction > 0.0) & (observation > 0.0)
    output = ratio_statistics(
        prediction,
        prediction_covariance,
        observation,
        observation_variance,
        valid,
    )
    output["labels"] = labels
    output["fit_categories"] = [True, False, True]
    output["nb1_prediction_over_observation"] = finite_or_none(
        prediction[1] / observation[1]
    )
    return output


def configure_style() -> None:
    plt.style.use(hep.style.CMS)
    mpl.rcParams.update(
        {
            "figure.figsize": (8.6, 8.6),
            "font.size": 16,
            "axes.labelsize": 22,
            "xtick.labelsize": 15,
            "ytick.labelsize": 17,
            "legend.fontsize": 13,
            "savefig.facecolor": "white",
        }
    )


def configure_xaxis(
    axis: Any,
    ratio_axis: Any,
    labels: list[str],
    xlabel: str,
) -> None:
    nbin = len(labels)
    centers = np.arange(nbin, dtype=float) + 0.5
    for current in (axis, ratio_axis):
        current.set_xlim(0.0, float(nbin))
        current.margins(x=0)
    ratio_axis.set_xticks(centers)
    ratio_axis.set_xticklabels(labels, rotation=40, ha="right")
    ratio_axis.set_xlabel(xlabel, loc="right")
    axis.tick_params(labelbottom=False)


def display_bin_labels(labels: list[str]) -> list[str]:
    output: list[str] = []
    for label in labels:
        if label.endswith("plus") and label[:-4].isdigit():
            output.append(rf"$\geq {label[:-4]}$")
        else:
            output.append(label.replace("-", "–"))
    return output


def step_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(len(values) + 1, dtype=float)
    y_values = (
        np.r_[values, values[-1]] if len(values) else np.asarray([])
    )
    return x_values, y_values


def plot_closure_comparison(
    record: dict[str, Any], output_stem: Path
) -> None:
    labels = display_bin_labels(list(record["labels"]))
    nbin = len(labels)
    centers = np.arange(nbin, dtype=float) + 0.5
    split = record["split_top_w"]
    baseline = record["combined_baseline"]
    observation = np.asarray(split["observation"], dtype=float)
    observation_uncertainty = np.sqrt(
        np.asarray(split["observation_variance"], dtype=float)
    )
    split_prediction = np.asarray(split["prediction"], dtype=float)
    split_uncertainty = np.sqrt(
        np.asarray(split["prediction_variance"], dtype=float)
    )
    combined_prediction = np.asarray(baseline["prediction"], dtype=float)
    combined_uncertainty = np.sqrt(
        np.asarray(baseline["prediction_variance"], dtype=float)
    )
    valid = np.asarray(record["valid_mask"], dtype=bool)
    fig = plt.figure(figsize=(8.6, 8.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.2, 1.0),
        hspace=0.04,
        left=0.16,
        right=0.97,
        bottom=0.16,
        top=0.88,
    )
    axis = fig.add_subplot(grid[0])
    ratio_axis = fig.add_subplot(grid[1], sharex=axis)
    x_step, split_step = step_values(split_prediction)
    _, combined_step = step_values(combined_prediction)
    axis.step(
        x_step,
        split_step,
        where="post",
        color="#d62728",
        linewidth=2.2,
        label="Top/W split prediction",
    )
    axis.fill_between(
        x_step,
        np.r_[split_prediction - split_uncertainty,
              split_prediction[-1] - split_uncertainty[-1]],
        np.r_[split_prediction + split_uncertainty,
              split_prediction[-1] + split_uncertainty[-1]],
        step="post",
        color="#d62728",
        alpha=0.18,
        linewidth=0,
        label="Split stat. unc.",
    )
    axis.step(
        x_step,
        combined_step,
        where="post",
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label="Combined-TF baseline",
    )
    axis.errorbar(
        centers[valid],
        observation[valid],
        yerr=observation_uncertainty[valid],
        color="black",
        marker="o",
        linestyle="none",
        capsize=2,
        label="Data − other MC",
    )
    nonpositive = observation <= 0.0
    if np.any(nonpositive):
        positive_values = np.r_[
            observation[observation > 0.0],
            split_prediction[split_prediction > 0.0],
        ]
        marker_y = (
            float(np.min(positive_values)) * 0.12
            if len(positive_values)
            else 0.1
        )
        axis.scatter(
            centers[nonpositive],
            np.full(np.count_nonzero(nonpositive), marker_y),
            marker="v",
            color="#aa3377",
            label="Nonpositive residual",
            zorder=5,
        )
    positive_values = np.r_[
        observation[observation > 0.0],
        split_prediction[split_prediction > 0.0],
        combined_prediction[combined_prediction > 0.0],
    ]
    if len(positive_values):
        axis.set_yscale("log")
        axis.set_ylim(
            max(float(np.min(positive_values)) * 0.06, 1.0e-3),
            float(np.max(positive_values)) * 8.0,
        )
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right", fontsize=12)
    annotation = "\n".join(VR_ANNOTATIONS[record["name"]])
    axis.text(
        0.025,
        0.04,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    split_ratio = np.asarray(split["ratio"], dtype=float)
    split_ratio_uncertainty = np.sqrt(
        np.asarray(split["ratio_variance"], dtype=float)
    )
    combined_ratio = np.asarray(baseline["ratio"], dtype=float)
    combined_ratio_uncertainty = np.sqrt(
        np.asarray(baseline["ratio_variance"], dtype=float)
    )
    ratio_axis.axhline(1.0, color="black", linewidth=1.2)
    ratio_axis.errorbar(
        centers[valid],
        split_ratio[valid],
        yerr=split_ratio_uncertainty[valid],
        color="#d62728",
        marker="o",
        linestyle="none",
        capsize=2,
    )
    ratio_axis.errorbar(
        centers[valid],
        combined_ratio[valid],
        yerr=combined_ratio_uncertainty[valid],
        color="#1f77b4",
        marker="s",
        linestyle="none",
        capsize=2,
        markersize=4,
    )
    ratio_axis.set_ylabel("Pred./residual")
    finite_ratios = np.r_[
        split_ratio[np.isfinite(split_ratio)],
        combined_ratio[np.isfinite(combined_ratio)],
    ]
    if len(finite_ratios):
        lower = max(0.0, float(np.min(finite_ratios)) - 0.2)
        upper = max(1.25, float(np.max(finite_ratios)) + 0.2)
        ratio_axis.set_ylim(lower, min(upper, 2.2))
    ratio_axis.grid(axis="y", alpha=0.25)
    configure_xaxis(
        axis, ratio_axis, labels, VR_XLABELS[record["name"]]
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_control_shape(
    name: str,
    labels: list[str],
    record: dict[str, Any],
    output_stem: Path,
) -> None:
    labels = display_bin_labels(labels)
    centers = np.arange(len(labels), dtype=float) + 0.5
    prediction = np.asarray(record["prediction"], dtype=float)
    prediction_uncertainty = np.sqrt(
        np.asarray(record["prediction_variance"], dtype=float)
    )
    observation = np.asarray(record["observation"], dtype=float)
    observation_uncertainty = np.sqrt(
        np.asarray(record["observation_variance"], dtype=float)
    )
    ratio = np.asarray(record["ratio"], dtype=float)
    ratio_uncertainty = np.sqrt(
        np.asarray(record["ratio_variance"], dtype=float)
    )
    valid = np.asarray(record["valid"], dtype=bool)
    fig = plt.figure(figsize=(8.6, 8.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.2, 1.0),
        hspace=0.04,
        left=0.16,
        right=0.97,
        bottom=0.16,
        top=0.88,
    )
    axis = fig.add_subplot(grid[0])
    ratio_axis = fig.add_subplot(grid[1], sharex=axis)
    x_step, y_step = step_values(prediction)
    axis.step(
        x_step,
        y_step,
        where="post",
        color="#d62728",
        linewidth=2.2,
        label="Top/W fit prediction",
    )
    axis.fill_between(
        x_step,
        np.r_[
            prediction - prediction_uncertainty,
            prediction[-1] - prediction_uncertainty[-1],
        ],
        np.r_[
            prediction + prediction_uncertainty,
            prediction[-1] + prediction_uncertainty[-1],
        ],
        step="post",
        color="#d62728",
        alpha=0.18,
        linewidth=0,
        label="Stat. unc.",
    )
    axis.errorbar(
        centers[valid],
        observation[valid],
        yerr=observation_uncertainty[valid],
        color="black",
        marker="o",
        linestyle="none",
        capsize=2,
        label="Data − other MC",
    )
    positive_values = np.r_[
        prediction[prediction > 0.0],
        observation[observation > 0.0],
    ]
    if len(positive_values):
        axis.set_yscale("log")
        axis.set_ylim(
            max(float(np.min(positive_values)) * 0.12, 1.0e-3),
            float(np.max(positive_values)) * 5.0,
        )
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right")
    annotation = {
        "highdm_nb0": (
            r"High-$\Delta m$ one-lepton control region"
            "\n"
            r"$N_{\mathrm{b}}=0$ (W enriched)"
        ),
        "highdm_njet3to4_nb1plus": (
            r"High-$\Delta m$ one-lepton control region"
            "\n"
            r"$3\leq N_{\mathrm{j}}\leq4,\ N_{\mathrm{b}}\geq1$ "
            r"(Top enriched)"
        ),
    }[name]
    axis.text(
        0.025,
        0.04,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    ratio_axis.axhline(1.0, color="black", linewidth=1.2)
    ratio_axis.errorbar(
        centers[valid],
        ratio[valid],
        yerr=ratio_uncertainty[valid],
        color="#d62728",
        marker="o",
        linestyle="none",
        capsize=2,
    )
    ratio_axis.set_ylabel("Pred./residual")
    finite_ratio = ratio[np.isfinite(ratio)]
    if len(finite_ratio):
        ratio_axis.set_ylim(
            max(0.0, float(np.min(finite_ratio)) - 0.2),
            max(1.25, float(np.max(finite_ratio)) + 0.2),
        )
    ratio_axis.grid(axis="y", alpha=0.25)
    configure_xaxis(
        axis,
        ratio_axis,
        labels,
        r"$\mathrm{p}_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_scale_factors(
    highdm_fit: dict[str, Any],
    lowdm_fit: dict[str, Any],
    output_stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.3, 8.3))
    x_values = np.arange(2, dtype=float)
    offsets = (-0.08, 0.08)
    for offset, fit, color, label in (
        (offsets[0], highdm_fit, "#1f77b4", r"High-$\Delta m$"),
        (offsets[1], lowdm_fit, "#d62728", r"Low-$\Delta m$"),
    ):
        values = np.asarray(fit["scale_factors"], dtype=float)
        errors = np.asarray(
            fit["scale_factor_uncertainties"], dtype=float
        )
        axis.errorbar(
            x_values + offset,
            values,
            yerr=errors,
            color=color,
            marker="o",
            linestyle="none",
            capsize=3,
            label=label,
        )
    axis.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
    axis.set_xlim(-0.5, 1.5)
    axis.margins(x=0)
    axis.set_xticks(x_values)
    axis.set_xticklabels([r"Top ($\mathrm{t\bar t}+\mathrm{ST}$)",
                          r"W+jets"])
    axis.set_ylabel("Control-region normalization")
    axis.set_ylim(0.6, 1.15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper right")
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_lowdm_nb_validation(
    record: dict[str, Any], output_stem: Path
) -> None:
    labels = list(record["labels"])
    prediction = np.asarray(record["prediction"], dtype=float)
    prediction_uncertainty = np.sqrt(
        np.asarray(record["prediction_variance"], dtype=float)
    )
    observation = np.asarray(record["observation"], dtype=float)
    observation_uncertainty = np.sqrt(
        np.asarray(record["observation_variance"], dtype=float)
    )
    ratio = np.asarray(record["ratio"], dtype=float)
    ratio_uncertainty = np.sqrt(
        np.asarray(record["ratio_variance"], dtype=float)
    )
    centers = np.arange(len(labels), dtype=float) + 0.5
    fig = plt.figure(figsize=(8.6, 8.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.2, 1.0),
        hspace=0.04,
        left=0.16,
        right=0.97,
        bottom=0.17,
        top=0.88,
    )
    axis = fig.add_subplot(grid[0])
    ratio_axis = fig.add_subplot(grid[1], sharex=axis)
    axis.errorbar(
        centers,
        prediction,
        yerr=prediction_uncertainty,
        color="#d62728",
        marker="s",
        linestyle="none",
        capsize=3,
        label="Top/W fit prediction",
    )
    axis.errorbar(
        centers,
        observation,
        yerr=observation_uncertainty,
        color="black",
        marker="o",
        linestyle="none",
        capsize=3,
        label="Data − other MC",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.025,
        0.04,
        "Low-$\\Delta m$ one-lepton control region\n"
        "$N_{\\mathrm{b}}=1$ is not used in the fit",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    ratio_axis.axhline(1.0, color="black", linewidth=1.2)
    ratio_axis.errorbar(
        centers,
        ratio,
        yerr=ratio_uncertainty,
        color="#d62728",
        marker="o",
        linestyle="none",
        capsize=3,
    )
    ratio_axis.set_ylabel("Pred./residual")
    ratio_axis.set_ylim(0.6, 1.2)
    ratio_axis.grid(axis="y", alpha=0.25)
    configure_xaxis(
        axis,
        ratio_axis,
        labels,
        "One-lepton control category",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    lines = [
        "# 2024 Top/W-split lost-lepton closure",
        "",
        "- Top definition: `TT + ST`",
        "- W definition: `WtoLNu`",
        "- Selection authority: `real_subset_worker.py`",
        "- Nominal intermediates modified: `false`",
        "",
        "## Control-region scale factors",
        "",
        "| Regime | Top | W+jets | Correlation |",
        "|---|---:|---:|---:|",
    ]
    for key in ("highdm", "lowdm"):
        fit = result["fits"][key]
        values = fit["scale_factors"]
        errors = fit["scale_factor_uncertainties"]
        lines.append(
            f"| {key} | {values[0]:.4f} ± {errors[0]:.4f} "
            f"| {values[1]:.4f} ± {errors[1]:.4f} "
            f"| {fit['scale_factor_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Data residual closure",
            "",
            "| Validation region | Combined ratio | Top/W ratio | "
            "Top/W p-value | max abs. pull |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, record in result["validation_regions"].items():
        combined_ratio = record["combined_baseline"]["integrated"]["ratio"]
        split_ratio = record["split_top_w"]["integrated"]["ratio"]
        p_value = record["split_top_w"]["p_value"]
        maximum_pull = record["split_top_w"]["maximum_absolute_pull"]
        lines.append(
            f"| {name} | {combined_ratio:.4f} | {split_ratio:.4f} "
            f"| {p_value:.4g} | {maximum_pull:.3f} |"
        )
    lowdm_nb1 = result["control_validations"]["lowdm_nb_groups"][
        "nb1_prediction_over_observation"
    ]
    lines.extend(
        [
            "",
            "## Independent CR diagnostic",
            "",
            "| High-dM control category | Integrated ratio | p-value "
            "| max abs. pull |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, record in result["control_validations"][
        "highdm_shapes"
    ].items():
        lines.append(
            f"| {name} | {record['integrated']['ratio']:.4f} "
            f"| {record['p_value']:.4g} "
            f"| {record['maximum_absolute_pull']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The low-dM `Nb=1` one-lepton category was excluded from the "
            "two-component normalization fit.",
            "",
            f"- Predicted / residual data in the excluded `Nb=1` category: "
            f"`{lowdm_nb1:.4f}`",
            "",
            "Only statistical covariance is propagated here. Detector/model "
            "systematics and a possible adopted nonclosure nuisance are not "
            "included.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def write_index(output_dir: Path, result: dict[str, Any]) -> None:
    rows = []
    for name, record in result["validation_regions"].items():
        combined_ratio = record["combined_baseline"]["integrated"]["ratio"]
        split_ratio = record["split_top_w"]["integrated"]["ratio"]
        p_value = record["split_top_w"]["p_value"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{combined_ratio:.4f}</td>"
            f"<td>{split_ratio:.4f}</td>"
            f"<td>{p_value:.4g}</td>"
            f"<td><a href=\"plots/top_w_closure_{html.escape(name)}.png\">"
            "plot</a></td>"
            "</tr>"
        )
    scale_rows = []
    for key in ("highdm", "lowdm"):
        fit = result["fits"][key]
        scale_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{fit['scale_factors'][0]:.4f} ± "
            f"{fit['scale_factor_uncertainties'][0]:.4f}</td>"
            f"<td>{fit['scale_factors'][1]:.4f} ± "
            f"{fit['scale_factor_uncertainties'][1]:.4f}</td>"
            f"<td>{fit['scale_factor_correlation']:.3f}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2024 Top/W-split lost-lepton closure</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #bbb; padding: .55rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(330px,1fr));
         gap: 1rem; }}
.grid img {{ width: 100%; height: auto; }}
code {{ background: #eee; padding: .15rem .3rem; }}
</style>
</head>
<body>
<h1>2024 Top/W-split lost-lepton closure</h1>
<p>Top = TT + single-top. W = W+jets. Two enriched one-lepton control
categories determine their normalizations independently.</p>
<h2>Control-region normalization</h2>
<table><thead><tr><th>Regime</th><th>Top</th><th>W+jets</th>
<th>Correlation</th></tr></thead><tbody>
{''.join(scale_rows)}
</tbody></table>
<p><a href="plots/top_w_scale_factors.png">Scale-factor plot</a> ·
<a href="plots/lowdm_nb_control_validation.png">Low-dM Nb validation</a> ·
<a href="plots/highdm_control_shape_highdm_nb0.png">High-dM Nb=0 CR</a> ·
<a href="plots/highdm_control_shape_highdm_njet3to4_nb1plus.png">
High-dM Top-enriched CR</a></p>
<h2>Data residual closure</h2>
<table><thead><tr><th>Region</th><th>Combined TF</th><th>Top/W split</th>
<th>p-value</th><th>Plot</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<div class="grid">
{''.join(f'<a href="plots/top_w_closure_{html.escape(name)}.png">'
         f'<img src="plots/top_w_closure_{html.escape(name)}.png" '
         f'alt="{html.escape(name)}"></a>'
         for name in result["validation_regions"])}
</div>
<h2>Files</h2>
<ul>
<li><a href="top_w_closure.json">Machine-readable result</a></li>
<li><a href="summary.md">Summary</a></li>
<li><a href="interpretation_ko.md">Korean interpretation</a></li>
<li><a href="failed_llcr_closure_report_en.md">
English failed-closure report and analysis decision</a></li>
<li><a href="output/pdf/failed_llcr_closure_report_2024.pdf">
Illustrated English PDF report</a></li>
</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    merged = json_load(args.merged)
    combined = json_load(args.combined)
    missing = [
        process
        for process in (
            *TOP_PROCESSES,
            *W_PROCESSES,
            *OTHER_PROCESSES,
            DATA_PROCESS,
        )
        if process not in merged.get("processes", {})
    ]
    if missing:
        raise RuntimeError(f"missing processes: {', '.join(missing)}")
    highdm_fit = solve_two_component_fit(
        merged, HIGHDM_FIT_CATEGORIES, "highdm"
    )
    lowdm_fit = solve_two_component_fit(
        merged, LOWDM_FIT_CATEGORIES, "lowdm"
    )
    validation_regions: dict[str, Any] = {}
    control_shapes: dict[str, Any] = {}
    for name in combined["validation_regions"]:
        fit = highdm_fit if name.startswith("highdm") else lowdm_fit
        validation_regions[name] = build_target_closure(
            merged, combined, name, fit
        )
        if name.startswith("highdm"):
            control_shapes[name] = build_control_shape(
                merged, name, highdm_fit
            )
    lowdm_nb_validation = build_lowdm_nb_validation(merged, lowdm_fit)
    result = {
        "status": "complete",
        "schema_version": "lost_lepton_top_w_split_closure_2024_v1",
        "input_merged_histograms": str(args.merged.resolve()),
        "input_merged_histograms_sha256": sha256_file(args.merged),
        "input_combined_closure": str(args.combined.resolve()),
        "input_combined_closure_sha256": sha256_file(args.combined),
        "selection_authority": "real_subset_worker.py",
        "nominal_intermediates_modified": False,
        "top_processes": list(TOP_PROCESSES),
        "w_processes": list(W_PROCESSES),
        "other_processes": list(OTHER_PROCESSES),
        "fits": {"highdm": highdm_fit, "lowdm": lowdm_fit},
        "control_validations": {
            "highdm_shapes": control_shapes,
            "lowdm_nb_groups": lowdm_nb_validation,
        },
        "validation_regions": validation_regions,
        "uncertainty_scope": (
            "Statistical covariance from data, other MC, Top/W templates, "
            "and common fitted Top/W normalizations. Detector/model "
            "systematics are not included."
        ),
    }
    json_dump(output_dir / "top_w_closure.json", result)
    configure_style()
    plot_dir = output_dir / "plots"
    plot_scale_factors(
        highdm_fit,
        lowdm_fit,
        plot_dir / "top_w_scale_factors",
    )
    plot_lowdm_nb_validation(
        lowdm_nb_validation,
        plot_dir / "lowdm_nb_control_validation",
    )
    for name, record in control_shapes.items():
        plot_control_shape(
            name,
            list(combined["validation_regions"][name]["labels"]),
            record,
            plot_dir / f"highdm_control_shape_{name}",
        )
    for name, record in validation_regions.items():
        plot_closure_comparison(
            record, plot_dir / f"top_w_closure_{name}"
        )
    write_summary(output_dir, result)
    write_index(output_dir, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "plots": len(list(plot_dir.glob("*.png"))),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
