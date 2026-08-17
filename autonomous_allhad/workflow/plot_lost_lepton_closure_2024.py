#!/usr/bin/env python3
"""Postprocess and plot the 2024 lost-lepton closure payload."""

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

try:
    from scipy.stats import chi2 as chi2_distribution
except Exception:  # pragma: no cover - optional runtime fallback
    chi2_distribution = None


hep.style.use("CMS")

TARGET_PROCESSES = ("TT", "WtoLNu", "ST")
DATA_PROCESS = "JetMET"
PROCESS_LABELS = {
    "TT": r"$t\bar{t}$",
    "WtoLNu": r"$W\to\ell\nu$",
    "ST": "Single top",
}
SCHEME_XLABELS = {
    "highdm_met": r"$p^{\mathrm{miss}}_{\mathrm{T}}$ (GeV)",
    "highdm_search60": r"High-$\Delta m$ search-bin index",
    "lowdm_search42": r"Low-$\Delta m$ search-bin index",
}
VR_LABELS = {
    "highdm_nb0": (
        r"High-$\Delta m$ validation region",
        r"$N_b=0$",
        r"$p^{\mathrm{miss}}_{\mathrm{T}}$ (GeV)",
    ),
    "highdm_njet3to4_nb1plus": (
        r"High-$\Delta m$ validation region",
        r"$3\leq N_j\leq4,\ N_b\geq1$",
        r"$p^{\mathrm{miss}}_{\mathrm{T}}$ (GeV)",
    ),
    "lowdm_met250to300": (
        r"Low-$\Delta m$ validation region",
        r"$250<p^{\mathrm{miss}}_{\mathrm{T}}<300$ GeV",
        r"$p^{\mathrm{miss}}_{\mathrm{T}}$ (GeV)",
    ),
    "lowdm_isr200to300": (
        r"Low-$\Delta m$ validation region",
        r"$200<p_{\mathrm{T}}^{\mathrm{ISR}}<300$ GeV",
        r"$p_{\mathrm{T}}^{\mathrm{ISR}}$ (GeV)",
    ),
    "lowdm_significance7to10": (
        r"Low-$\Delta m$ validation region",
        r"$7<p^{\mathrm{miss}}_{\mathrm{T}}/\sqrt{H_{\mathrm{T}}}<10$",
        r"$p^{\mathrm{miss}}_{\mathrm{T}}/\sqrt{H_{\mathrm{T}}}$",
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def array(record: dict[str, Any], field: str) -> np.ndarray:
    return np.asarray(record[field], dtype=float)


def maybe_array(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def finite_list(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in np.asarray(values, dtype=float)
    ]


def hist_for_processes(
    merged: dict[str, Any],
    processes: tuple[str, ...],
    section: str,
    name: str,
    side: str,
    fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if section == "histograms":
        first = next(
            (
                (((merged.get("processes") or {}).get(process) or {})
                 .get(section, {})
                 .get(name, {})
                 .get(str(fold), {})
                 .get(side))
                for process in processes
                if (((merged.get("processes") or {}).get(process) or {})
                    .get(section, {})
                    .get(name, {})
                    .get(str(fold), {})
                    .get(side))
            ),
            None,
        )
    else:
        first = next(
            (
                (((merged.get("processes") or {}).get(process) or {})
                 .get(section, {})
                 .get(name, {})
                 .get(side))
                for process in processes
                if (((merged.get("processes") or {}).get(process) or {})
                    .get(section, {})
                    .get(name, {})
                    .get(side))
            ),
            None,
        )
    if first is None:
        raise KeyError(f"missing {section}/{name}/{side} for {processes}")
    nbin = len(first["sumw"])
    sumw = np.zeros(nbin, dtype=float)
    sumw2 = np.zeros(nbin, dtype=float)
    entries = np.zeros(nbin, dtype=np.int64)
    for process in processes:
        process_record = (merged.get("processes") or {}).get(process) or {}
        if section == "histograms":
            record = (
                process_record.get(section, {})
                .get(name, {})
                .get(str(fold), {})
                .get(side)
            )
        else:
            record = (
                process_record.get(section, {})
                .get(name, {})
                .get(side)
            )
        if not record:
            continue
        sumw += array(record, "sumw")
        sumw2 += array(record, "sumw2")
        entries += np.asarray(record["entries"], dtype=np.int64)
    return sumw, sumw2, entries


def ratio_payload(
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
    direct: np.ndarray,
    direct_variance: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    ratio = np.divide(
        prediction,
        direct,
        out=np.full(len(prediction), np.nan, dtype=float),
        where=valid & (direct != 0.0),
    )
    ratio_variance = np.full(len(prediction), np.nan, dtype=float)
    good = valid & (direct != 0.0)
    ratio_variance[good] = (
        prediction_variance[good] / np.square(direct[good])
        + np.square(prediction[good])
        * direct_variance[good]
        / np.power(direct[good], 4)
    )
    total_variance = prediction_variance + direct_variance
    pull = np.divide(
        prediction - direct,
        np.sqrt(np.maximum(total_variance, 0.0)),
        out=np.full(len(prediction), np.nan, dtype=float),
        where=good & (total_variance > 0.0),
    )
    finite = np.isfinite(pull)
    diagonal_chi2 = float(np.sum(np.square(pull[finite])))
    ndf = int(np.count_nonzero(finite))
    p_value = (
        float(chi2_distribution.sf(diagonal_chi2, ndf))
        if chi2_distribution is not None and ndf > 0
        else None
    )
    return {
        "prediction": finite_list(prediction),
        "prediction_variance": finite_list(prediction_variance),
        "direct": finite_list(direct),
        "direct_variance": finite_list(direct_variance),
        "closure_ratio": finite_list(ratio),
        "closure_ratio_variance": finite_list(ratio_variance),
        "pull": finite_list(pull),
        "valid": good.tolist(),
        "diagonal_statistical_covariance": np.diag(
            np.where(np.isfinite(ratio_variance), ratio_variance, 0.0)
        ).tolist(),
        "diagonal_chi2": diagonal_chi2,
        "ndf": ndf,
        "p_value": p_value,
        "maximum_absolute_pull": (
            float(np.max(np.abs(pull[finite]))) if np.any(finite) else None
        ),
    }


def apply_statistical_gate(group: dict[str, Any]) -> None:
    """Decorate a two-direction closure with the predeclared Neff gate."""
    first = group["A_to_B"]
    second = group["B_to_A"]
    crossfit = group["crossfit"]
    sufficient = (
        np.asarray(first["valid_closure"], dtype=bool)
        & np.asarray(second["valid_closure"], dtype=bool)
        & (maybe_array(first["train_control_neff"]) >= 25.0)
        & (maybe_array(second["train_control_neff"]) >= 25.0)
        & (maybe_array(first["train_target_neff"]) >= 10.0)
        & (maybe_array(second["train_target_neff"]) >= 10.0)
    )
    prediction = maybe_array(crossfit["prediction"])
    prediction_variance = maybe_array(crossfit["prediction_variance"])
    direct = maybe_array(crossfit["direct"])
    direct_variance = maybe_array(crossfit["direct_variance"])
    sufficient &= (
        np.isfinite(prediction)
        & np.isfinite(direct)
        & (prediction > 0.0)
        & (direct > 0.0)
    )
    gated = ratio_payload(
        prediction,
        prediction_variance,
        direct,
        direct_variance,
        sufficient,
    )
    crossfit["statistically_sufficient"] = sufficient.tolist()
    crossfit["insufficient_statistics_bins_zero_based"] = np.flatnonzero(
        ~sufficient
    ).astype(int).tolist()
    crossfit["statistical_gate"] = {
        "minimum_train_control_neff_in_each_direction": 25.0,
        "minimum_train_target_neff_in_each_direction": 10.0,
        "positive_crossfit_prediction_and_direct_yield": True,
    }
    crossfit["gated_metrics"] = {
        "valid_bins": int(np.count_nonzero(sufficient)),
        "diagonal_chi2": gated["diagonal_chi2"],
        "ndf": gated["ndf"],
        "p_value": gated["p_value"],
        "maximum_absolute_pull": gated["maximum_absolute_pull"],
        "diagonal_statistical_covariance": gated[
            "diagonal_statistical_covariance"
        ],
    }


def decorate_mc_statistical_gates(mc: dict[str, Any]) -> None:
    for group in mc.get("schemes", {}).values():
        apply_statistical_gate(group)
    for process in TARGET_PROCESSES:
        for group in (
            (mc.get("process_diagnostics") or {}).get(process) or {}
        ).values():
            apply_statistical_gate(group)


def decorate_full_mixture_statistical_gates(
    full: dict[str, Any], mc: dict[str, Any]
) -> None:
    """Use the pure-target TF-statistics gate for the mixture pseudodata test."""
    for scheme, group in full.get("schemes", {}).items():
        record = group["crossfit"]
        sufficient = np.asarray(
            mc["schemes"][scheme]["crossfit"]["statistically_sufficient"],
            dtype=bool,
        )
        prediction = maybe_array(record["prediction"])
        prediction_variance = maybe_array(record["prediction_variance"])
        direct = maybe_array(record["direct"])
        direct_variance = maybe_array(record["direct_variance"])
        sufficient &= (
            np.asarray(record["valid"], dtype=bool)
            & np.isfinite(prediction)
            & np.isfinite(direct)
            & (prediction > 0.0)
            & (direct > 0.0)
        )
        gated = ratio_payload(
            prediction,
            prediction_variance,
            direct,
            direct_variance,
            sufficient,
        )
        record["statistically_sufficient"] = sufficient.tolist()
        record["insufficient_statistics_bins_zero_based"] = np.flatnonzero(
            ~sufficient
        ).astype(int).tolist()
        record["statistical_gate"] = {
            "source": "pure_target_transfer_factor_gate",
            "minimum_train_control_neff_in_each_direction": 25.0,
            "minimum_train_target_neff_in_each_direction": 10.0,
            "positive_crossfit_prediction_and_direct_yield": True,
        }
        record["gated_metrics"] = {
            "valid_bins": int(np.count_nonzero(sufficient)),
            "diagonal_chi2": gated["diagonal_chi2"],
            "ndf": gated["ndf"],
            "p_value": gated["p_value"],
            "maximum_absolute_pull": gated["maximum_absolute_pull"],
            "diagonal_statistical_covariance": gated[
                "diagonal_statistical_covariance"
            ],
        }


def fold_hist(
    merged: dict[str, Any],
    processes: tuple[str, ...],
    scheme: str,
    fold: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return hist_for_processes(
        merged, processes, "histograms", scheme, side, fold=fold
    )


def calculate_direction(
    merged: dict[str, Any],
    target_processes: tuple[str, ...],
    direct_processes: tuple[str, ...],
    other_processes: tuple[str, ...],
    scheme: str,
    train: int,
    test: int,
    full_mixture: bool,
) -> dict[str, Any]:
    train_target_sr, train_target_sr_var, _ = fold_hist(
        merged, target_processes, scheme, train, "target"
    )
    train_target_cr, train_target_cr_var, _ = fold_hist(
        merged, target_processes, scheme, train, "control"
    )
    valid_tf = train_target_cr != 0.0
    tf = np.divide(
        train_target_sr,
        train_target_cr,
        out=np.full(len(train_target_sr), np.nan),
        where=valid_tf,
    )
    tf_var = np.full(len(tf), np.nan)
    tf_var[valid_tf] = (
        train_target_sr_var[valid_tf]
        / np.square(train_target_cr[valid_tf])
        + np.square(train_target_sr[valid_tf])
        * train_target_cr_var[valid_tf]
        / np.power(train_target_cr[valid_tf], 4)
    )
    test_target_cr, test_target_cr_var, _ = fold_hist(
        merged, target_processes, scheme, test, "control"
    )
    test_target_sr, test_target_sr_var, _ = fold_hist(
        merged, target_processes, scheme, test, "target"
    )
    if not full_mixture:
        residual_cr = test_target_cr
        residual_cr_var = test_target_cr_var
        direct = test_target_sr
        direct_var = test_target_sr_var
        other_target = np.zeros_like(direct)
        other_target_var = np.zeros_like(direct)
    else:
        all_cr, all_cr_var, _ = fold_hist(
            merged, direct_processes, scheme, test, "control"
        )
        other_cr, other_cr_var, _ = fold_hist(
            merged, other_processes, scheme, test, "control"
        )
        residual_cr = all_cr - other_cr
        residual_cr_var = all_cr_var + other_cr_var
        direct, direct_var, _ = fold_hist(
            merged, direct_processes, scheme, test, "target"
        )
        other_target, other_target_var, _ = fold_hist(
            merged, other_processes, scheme, test, "target"
        )
    prediction_target = tf * residual_cr
    prediction_target_var = (
        np.square(residual_cr) * tf_var + np.square(tf) * residual_cr_var
    )
    prediction = prediction_target + other_target
    prediction_var = prediction_target_var + other_target_var
    valid = valid_tf & np.isfinite(prediction) & (direct != 0.0)
    output = ratio_payload(
        prediction, prediction_var, direct, direct_var, valid
    )
    output.update(
        {
            "transfer_factor": finite_list(tf),
            "transfer_factor_variance": finite_list(tf_var),
            "residual_control": finite_list(residual_cr),
            "residual_control_variance": finite_list(residual_cr_var),
            "predicted_target_component": finite_list(prediction_target),
            "predicted_target_component_variance": finite_list(
                prediction_target_var
            ),
        }
    )
    return output


def combine_directions(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    prediction = maybe_array(first["prediction"]) + maybe_array(
        second["prediction"]
    )
    prediction_var = maybe_array(first["prediction_variance"]) + maybe_array(
        second["prediction_variance"]
    )
    direct = maybe_array(first["direct"]) + maybe_array(second["direct"])
    direct_var = maybe_array(first["direct_variance"]) + maybe_array(
        second["direct_variance"]
    )
    valid = np.asarray(first["valid"], dtype=bool) & np.asarray(
        second["valid"], dtype=bool
    )
    return ratio_payload(
        prediction, prediction_var, direct, direct_var, valid
    )


def build_full_mixture_closure(
    merged: dict[str, Any], labels_by_scheme: dict[str, list[str]]
) -> dict[str, Any]:
    available = tuple(
        process
        for process in merged.get("processes", {})
        if process != DATA_PROCESS
    )
    other = tuple(
        process for process in available if process not in TARGET_PROCESSES
    )
    output: dict[str, Any] = {
        "status": "complete",
        "target_processes": list(TARGET_PROCESSES),
        "pseudodata_processes": list(available),
        "subtracted_processes": list(other),
        "schemes": {},
    }
    for scheme, labels in labels_by_scheme.items():
        a_to_b = calculate_direction(
            merged,
            TARGET_PROCESSES,
            available,
            other,
            scheme,
            0,
            1,
            full_mixture=True,
        )
        b_to_a = calculate_direction(
            merged,
            TARGET_PROCESSES,
            available,
            other,
            scheme,
            1,
            0,
            full_mixture=True,
        )
        output["schemes"][scheme] = {
            "labels": labels,
            "A_to_B": a_to_b,
            "B_to_A": b_to_a,
            "crossfit": combine_directions(a_to_b, b_to_a),
        }
    return output


def build_data_vr_closure(
    merged: dict[str, Any], vr_specs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    available_backgrounds = tuple(
        process
        for process in merged.get("processes", {})
        if process != DATA_PROCESS
    )
    missing = [
        process
        for process in (*TARGET_PROCESSES, DATA_PROCESS)
        if process not in merged.get("processes", {})
    ]
    if missing:
        return {
            "status": "blocked_missing_processes",
            "missing_processes": missing,
        }
    other = tuple(
        process
        for process in available_backgrounds
        if process not in TARGET_PROCESSES
    )
    output: dict[str, Any] = {
        "status": "complete",
        "target_processes": list(TARGET_PROCESSES),
        "subtracted_processes": list(other),
        "validation_regions": {},
    }
    for name, spec in vr_specs.items():
        target_cr, target_cr_var, target_cr_entries = hist_for_processes(
            merged,
            TARGET_PROCESSES,
            "validation_regions",
            name,
            "control",
        )
        target_sr, target_sr_var, target_sr_entries = hist_for_processes(
            merged,
            TARGET_PROCESSES,
            "validation_regions",
            name,
            "target",
        )
        other_cr, other_cr_var, _ = hist_for_processes(
            merged, other, "validation_regions", name, "control"
        )
        other_sr, other_sr_var, _ = hist_for_processes(
            merged, other, "validation_regions", name, "target"
        )
        data_cr, data_cr_var, data_cr_entries = hist_for_processes(
            merged,
            (DATA_PROCESS,),
            "validation_regions",
            name,
            "control",
        )
        data_sr, data_sr_var, data_sr_entries = hist_for_processes(
            merged,
            (DATA_PROCESS,),
            "validation_regions",
            name,
            "target",
        )
        valid_tf = target_cr != 0.0
        tf = np.divide(
            target_sr,
            target_cr,
            out=np.full(len(target_sr), np.nan),
            where=valid_tf,
        )
        tf_var = np.full(len(tf), np.nan)
        tf_var[valid_tf] = (
            target_sr_var[valid_tf] / np.square(target_cr[valid_tf])
            + np.square(target_sr[valid_tf])
            * target_cr_var[valid_tf]
            / np.power(target_cr[valid_tf], 4)
        )
        residual_cr = data_cr - other_cr
        residual_cr_var = data_cr_var + other_cr_var
        predicted_target = tf * residual_cr
        predicted_target_var = (
            np.square(residual_cr) * tf_var
            + np.square(tf) * residual_cr_var
        )
        residual_target = data_sr - other_sr
        residual_target_var = data_sr_var + other_sr_var
        total_prediction = predicted_target + other_sr
        total_prediction_var = predicted_target_var + other_sr_var
        valid = (
            valid_tf
            & np.isfinite(predicted_target)
            & (residual_target > 0.0)
            & (residual_cr > 0.0)
        )
        record = ratio_payload(
            predicted_target,
            predicted_target_var,
            residual_target,
            residual_target_var,
            valid,
        )
        total_valid = (
            valid_tf
            & np.isfinite(total_prediction)
            & (total_prediction > 0.0)
            & (data_sr > 0.0)
            & (residual_cr > 0.0)
        )
        total_comparison = ratio_payload(
            total_prediction,
            total_prediction_var,
            data_sr,
            data_sr_var,
            total_valid,
        )
        neff_cr = np.divide(
            np.square(target_cr),
            target_cr_var,
            out=np.zeros(len(target_cr)),
            where=target_cr_var > 0.0,
        )
        neff_sr = np.divide(
            np.square(target_sr),
            target_sr_var,
            out=np.zeros(len(target_sr)),
            where=target_sr_var > 0.0,
        )
        statistically_sufficient = (
            valid
            & (neff_cr >= 25.0)
            & (neff_sr >= 10.0)
            & (predicted_target > 0.0)
        )
        gated = ratio_payload(
            predicted_target,
            predicted_target_var,
            residual_target,
            residual_target_var,
            statistically_sufficient,
        )
        record.update(
            {
                "labels": list(spec["labels"]),
                "statistically_sufficient": statistically_sufficient.tolist(),
                "insufficient_statistics_bins_zero_based": np.flatnonzero(
                    ~statistically_sufficient
                ).astype(int).tolist(),
                "statistical_gate": {
                    "minimum_target_mc_control_neff": 25.0,
                    "minimum_target_mc_target_neff": 10.0,
                    "positive_control_residual": True,
                    "positive_prediction_and_observed_target_yield": True,
                },
                "gated_metrics": {
                    "valid_bins": int(
                        np.count_nonzero(statistically_sufficient)
                    ),
                    "diagonal_chi2": gated["diagonal_chi2"],
                    "ndf": gated["ndf"],
                    "p_value": gated["p_value"],
                    "maximum_absolute_pull": gated[
                        "maximum_absolute_pull"
                    ],
                    "diagonal_statistical_covariance": gated[
                        "diagonal_statistical_covariance"
                    ],
                },
                "nonpositive_control_residual_bins_zero_based": np.flatnonzero(
                    residual_cr <= 0.0
                ).astype(int).tolist(),
                "nonpositive_target_residual_bins_zero_based": np.flatnonzero(
                    residual_target <= 0.0
                ).astype(int).tolist(),
                "transfer_factor": finite_list(tf),
                "transfer_factor_variance": finite_list(tf_var),
                "target_mc_control": finite_list(target_cr),
                "target_mc_target": finite_list(target_sr),
                "target_mc_control_entries": target_cr_entries.tolist(),
                "target_mc_target_entries": target_sr_entries.tolist(),
                "target_mc_control_neff": finite_list(neff_cr),
                "target_mc_target_neff": finite_list(neff_sr),
                "data_control": finite_list(data_cr),
                "data_target": finite_list(data_sr),
                "data_control_entries": data_cr_entries.tolist(),
                "data_target_entries": data_sr_entries.tolist(),
                "other_mc_control": finite_list(other_cr),
                "other_mc_target": finite_list(other_sr),
                "control_residual": finite_list(residual_cr),
                "control_residual_variance": finite_list(residual_cr_var),
                "target_residual": finite_list(residual_target),
                "target_residual_variance": finite_list(
                    residual_target_var
                ),
                "predicted_lost_lepton": finite_list(predicted_target),
                "predicted_lost_lepton_variance": finite_list(
                    predicted_target_var
                ),
                "total_prediction": finite_list(total_prediction),
                "total_prediction_variance": finite_list(
                    total_prediction_var
                ),
                "total_prediction_vs_data": total_comparison,
                "contamination_fraction_control": finite_list(
                    np.divide(
                        other_cr,
                        data_cr,
                        out=np.full(len(data_cr), np.nan),
                        where=data_cr != 0.0,
                    )
                ),
            }
        )
        output["validation_regions"][name] = record
    return output


def step_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(values) + 1, dtype=float)
    y = np.r_[values, values[-1]] if len(values) else np.asarray([])
    return x, y


def configure_xaxis(
    axis: Any, ratio_axis: Any, labels: list[str], xlabel: str
) -> None:
    nbin = len(labels)
    for current in (axis, ratio_axis):
        current.set_xlim(0.0, float(nbin))
        current.margins(x=0)
    centers = np.arange(nbin, dtype=float) + 0.5
    if nbin <= 12:
        ratio_axis.set_xticks(centers)
        display = [
            label.replace("1500plus", r"$\geq1500$")
            for label in labels
        ]
        ratio_axis.set_xticklabels(
            display, rotation=35 if nbin > 4 else 0, ha="right"
        )
    else:
        stride = 5 if nbin <= 45 else 10
        selected = np.arange(0, nbin, stride, dtype=int)
        ratio_axis.set_xticks(selected + 0.5)
        ratio_axis.set_xticklabels([str(index + 1) for index in selected])
    ratio_axis.set_xlabel(xlabel)


def plot_closure(
    record: dict[str, Any],
    labels: list[str],
    xlabel: str,
    annotation_lines: tuple[str, ...],
    output_stem: Path,
    is_observed_data: bool,
    direct_label: str | None = None,
    ratio_label: str | None = None,
) -> None:
    prediction = maybe_array(record["prediction"])
    prediction_unc = np.sqrt(
        np.maximum(maybe_array(record["prediction_variance"]), 0.0)
    )
    direct = maybe_array(record["direct"])
    direct_unc = np.sqrt(
        np.maximum(maybe_array(record["direct_variance"]), 0.0)
    )
    ratio = maybe_array(record["closure_ratio"])
    ratio_unc = np.sqrt(
        np.maximum(maybe_array(record["closure_ratio_variance"]), 0.0)
    )
    base_valid = np.asarray(
        record.get("valid", record.get("valid_closure", [])), dtype=bool
    )
    valid = np.asarray(
        record.get("statistically_sufficient", base_valid), dtype=bool
    )
    insufficient = base_valid & ~valid
    nbin = len(labels)
    centers = np.arange(nbin, dtype=float) + 0.5
    fig = plt.figure(figsize=(8.6, 8.6))
    grid = fig.add_gridspec(
        2, 1, height_ratios=(3.1, 1.0), hspace=0.06
    )
    axis = fig.add_subplot(grid[0])
    ratio_axis = fig.add_subplot(grid[1], sharex=axis)
    xstep, pred_step = step_xy(prediction)
    axis.step(
        xstep,
        pred_step,
        where="post",
        color="#0072B2",
        linewidth=2.2,
        label="Prediction",
    )
    axis.fill_between(
        xstep,
        step_xy(prediction - prediction_unc)[1],
        step_xy(prediction + prediction_unc)[1],
        step="post",
        color="#0072B2",
        alpha=0.25,
        linewidth=0,
        label="Stat. unc.",
    )
    axis.errorbar(
        centers[valid],
        direct[valid],
        yerr=direct_unc[valid],
        fmt="o",
        color="black",
        markersize=5.2,
        capsize=2.0,
        label=direct_label
        or ("Observed data" if is_observed_data else "Direct simulation"),
        zorder=4,
    )
    visible_insufficient = (
        insufficient & np.isfinite(direct) & (direct > 0.0)
    )
    if np.any(visible_insufficient):
        axis.scatter(
            centers[visible_insufficient],
            direct[visible_insufficient],
            marker="x",
            color="#777777",
            s=34,
            linewidth=1.4,
            label="Insufficient MC stat.",
            zorder=5,
        )
    axis.set_ylabel("Events")
    positive = np.r_[
        prediction[np.isfinite(prediction) & (prediction > 0)],
        direct[np.isfinite(direct) & (direct > 0)],
    ]
    if len(positive) and np.max(positive) / max(np.min(positive), 1e-12) > 80:
        axis.set_yscale("log")
        axis.set_ylim(
            max(np.min(positive) * 0.06, 0.01),
            np.max(positive) * 30,
        )
    else:
        finite_yields = np.r_[prediction, direct]
        finite_yields = finite_yields[np.isfinite(finite_yields)]
        upper = (
            max(float(np.max(finite_yields)) * 1.45, 1.0)
            if len(finite_yields)
            else 1.0
        )
        axis.set_ylim(0.0, upper)
    invalid_residual = (
        is_observed_data
        & ~base_valid
        & np.isfinite(direct)
    )
    if np.any(invalid_residual):
        lower, upper = axis.get_ylim()
        marker_y = (
            lower * 1.5
            if axis.get_yscale() == "log"
            else lower + 0.025 * (upper - lower)
        )
        axis.scatter(
            centers[invalid_residual],
            np.full(np.count_nonzero(invalid_residual), marker_y),
            marker="v",
            color="#AA3377",
            s=38,
            label="Nonpositive residual",
            zorder=6,
        )
    axis.legend(frameon=False, loc="upper right", fontsize=14)
    annotation = "\n".join(annotation_lines)
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
        yerr=ratio_unc[valid],
        fmt="o",
        color="#0072B2",
        markersize=4.8,
        capsize=2.0,
    )
    finite_ratio = ratio[valid & np.isfinite(ratio)]
    if len(finite_ratio):
        extent = max(
            0.35,
            float(np.nanmax(np.abs(finite_ratio - 1.0))) * 1.25,
        )
        extent = min(extent, 1.5)
        ratio_axis.set_ylim(max(0.0, 1.0 - extent), 1.0 + extent)
    else:
        ratio_axis.set_ylim(0.0, 2.0)
    if np.any(insufficient & np.isfinite(ratio)):
        lower, upper = ratio_axis.get_ylim()
        displayed = np.clip(
            ratio[insufficient & np.isfinite(ratio)],
            lower + 0.03 * (upper - lower),
            upper - 0.03 * (upper - lower),
        )
        ratio_axis.scatter(
            centers[insufficient & np.isfinite(ratio)],
            displayed,
            marker="x",
            color="#777777",
            s=30,
            linewidth=1.3,
            zorder=4,
        )
    ratio_axis.set_ylabel(
        ratio_label
        or ("Pred./data" if is_observed_data else "Pred./direct")
    )
    ratio_axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelbottom=False)
    configure_xaxis(axis, ratio_axis, labels, xlabel)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_process_ratios(
    mc: dict[str, Any],
    scheme: str,
    labels: list[str],
    output_stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.3, 8.3))
    centers = np.arange(len(labels), dtype=float) + 0.5
    colors = {"TT": "#D55E00", "WtoLNu": "#0072B2", "ST": "#009E73"}
    offsets = {"TT": -0.12, "WtoLNu": 0.0, "ST": 0.12}
    for process in TARGET_PROCESSES:
        record = mc["process_diagnostics"][process][scheme]["crossfit"]
        ratio = maybe_array(record["closure_ratio"])
        uncertainty = np.sqrt(
            np.maximum(maybe_array(record["closure_ratio_variance"]), 0.0)
        )
        valid = np.asarray(
            record.get(
                "statistically_sufficient", record["valid_closure"]
            ),
            dtype=bool,
        )
        axis.errorbar(
            centers[valid] + offsets[process],
            ratio[valid],
            yerr=uncertainty[valid],
            fmt="o",
            color=colors[process],
            markersize=5,
            capsize=2,
            label=PROCESS_LABELS[process],
        )
    axis.axhline(1.0, color="black", linewidth=1.2)
    axis.set_ylabel("Prediction / direct simulation")
    axis.set_ylim(0.0, 2.0)
    axis.legend(frameon=False, loc="upper right", fontsize=14)
    axis.grid(axis="y", alpha=0.25)
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    dummy = axis
    configure_xaxis(axis, dummy, labels, SCHEME_XLABELS[scheme])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_process_transfer_factors(
    mc: dict[str, Any],
    scheme: str,
    labels: list[str],
    output_stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.3, 8.3))
    centers = np.arange(len(labels), dtype=float) + 0.5
    colors = {"TT": "#D55E00", "WtoLNu": "#0072B2", "ST": "#009E73"}
    offsets = {"TT": -0.12, "WtoLNu": 0.0, "ST": 0.12}
    finite_values: list[np.ndarray] = []
    for process in TARGET_PROCESSES:
        group = mc["process_diagnostics"][process][scheme]
        first = group["A_to_B"]
        second = group["B_to_A"]
        first_tf = maybe_array(first["transfer_factor"])
        second_tf = maybe_array(second["transfer_factor"])
        first_var = maybe_array(first["transfer_factor_variance"])
        second_var = maybe_array(second["transfer_factor_variance"])
        transfer_factor = 0.5 * (first_tf + second_tf)
        uncertainty = 0.5 * np.sqrt(
            np.maximum(first_var + second_var, 0.0)
        )
        crossfit = group["crossfit"]
        gate = crossfit.get(
            "statistically_sufficient",
            crossfit.get("valid_closure", crossfit.get("valid", [])),
        )
        valid = (
            np.asarray(gate, dtype=bool)
            & np.isfinite(transfer_factor)
            & np.isfinite(uncertainty)
            & (transfer_factor >= 0.0)
        )
        finite_values.append(transfer_factor[valid])
        axis.errorbar(
            centers[valid] + offsets[process],
            transfer_factor[valid],
            yerr=uncertainty[valid],
            fmt="o",
            color=colors[process],
            markersize=5,
            capsize=2,
            label=PROCESS_LABELS[process],
        )
    axis.set_ylabel(r"Transfer factor $N_{0\ell}/N_{1\ell}$")
    displayed = [
        values for values in finite_values if len(values)
    ]
    maximum = (
        float(np.max(np.concatenate(displayed))) if displayed else 1.0
    )
    axis.set_ylim(0.0, max(1.0, 1.3 * maximum))
    axis.legend(frameon=False, loc="upper right", fontsize=14)
    axis.grid(axis="y", alpha=0.25)
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 (13.6 TeV)",
        loc=0,
        ax=axis,
    )
    configure_xaxis(axis, axis, labels, SCHEME_XLABELS[scheme])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    merged: dict[str, Any],
    mc: dict[str, Any],
    full: dict[str, Any] | None,
    data_vr: dict[str, Any] | None,
) -> None:
    lines = [
        "# 2024 Lost-lepton closure",
        "",
        f"- Input status: `{merged.get('status')}`",
        f"- Duplicate audit: `{(merged.get('duplicate_audit') or {}).get('status')}`",
        f"- ROOT files: {merged.get('input_totals', {}).get('roots', 0)}",
        f"- Events scanned: {merged.get('input_totals', {}).get('events_read', 0):,}",
        f"- Events retained for closure/VR: {merged.get('input_totals', {}).get('events_selected', 0):,}",
        "- Target: `TT + WtoLNu + ST`",
        "- Fold split: event-level stable SplitMix64 hash; file/shard splitting is not used.",
        "- Selection authority: `real_subset_worker.py`.",
        "",
        "## Independent MC closure",
        "",
        "| Scheme | Valid bins | diagonal chi2/ndf | p-value | max |pull| |",
        "|---|---:|---:|---:|---:|",
    ]
    for scheme, record in mc["schemes"].items():
        crossfit = record["crossfit"]
        gated = crossfit.get("gated_metrics") or {}
        p_value = gated.get("p_value")
        if p_value is None and chi2_distribution is not None:
            ndf = int(gated.get("ndf") or 0)
            p_value = (
                float(
                    chi2_distribution.sf(
                        float(gated.get("diagonal_chi2") or 0.0), ndf
                    )
                )
                if ndf > 0
                else None
            )
        lines.append(
            f"| {scheme} | {gated.get('valid_bins', 0)} "
            f"| {gated.get('diagonal_chi2', 0):.2f}/{gated.get('ndf', 0)} "
            f"| {p_value:.3g} "
            f"| {gated.get('maximum_absolute_pull'):.3g} |"
            if p_value is not None
            and gated.get("maximum_absolute_pull") is not None
            else f"| {scheme} | 0 | n/a | n/a | n/a |"
        )
    if full is not None:
        lines.extend(["", "## Full-mixture MC pseudodata closure", ""])
        for scheme, record in full.get("schemes", {}).items():
            crossfit = record["crossfit"]
            gated = crossfit.get("gated_metrics") or {}
            lines.append(
                f"- `{scheme}`: {gated.get('valid_bins', 0)} valid bins, "
                f"p={gated.get('p_value')}, "
                f"max |pull|={gated.get('maximum_absolute_pull')}"
            )
    if data_vr is not None:
        lines.extend(["", "## Data validation regions", ""])
        if data_vr.get("status") != "complete":
            lines.append(f"- Status: `{data_vr.get('status')}`")
        else:
            lines.extend(
                [
                    "| Validation region | Valid bins | MC LL purity (1l / 0l) | p-value | max |pull| |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for name, record in data_vr["validation_regions"].items():
                gated = record.get("gated_metrics") or {}
                target_control = np.nansum(
                    maybe_array(record["target_mc_control"])
                )
                target_target = np.nansum(
                    maybe_array(record["target_mc_target"])
                )
                other_control = np.nansum(
                    maybe_array(record["other_mc_control"])
                )
                other_target = np.nansum(
                    maybe_array(record["other_mc_target"])
                )
                control_purity = np.divide(
                    target_control,
                    target_control + other_control,
                )
                target_purity = np.divide(
                    target_target,
                    target_target + other_target,
                )
                lines.append(
                    f"| {name} | {gated.get('valid_bins', 0)} "
                    f"| {control_purity:.3f} / {target_purity:.3f} "
                    f"| {gated.get('p_value')} "
                    f"| {gated.get('maximum_absolute_pull')} |"
                )
    lines.extend(
        [
            "",
            "Statistical covariance is diagonal because one original event contributes to one bin in each displayed categorization. Detector/model systematics and any adopted nonclosure nuisance are separate from this statistical covariance.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def write_html(output_dir: Path) -> None:
    pngs = sorted((output_dir / "plots").glob("*.png"))
    cards = "\n".join(
        f"<section><h2>{html.escape(path.stem.replace('_', ' '))}</h2>"
        f"<a href='plots/{path.with_suffix('.pdf').name}'>PDF</a>"
        f"<img src='plots/{path.name}' alt='{html.escape(path.stem)}'></section>"
        for path in pngs
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>2024 lost-lepton closure</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f4f5f7;color:#1d2530}}
main{{max-width:1100px;margin:auto;padding:24px}}
section{{background:white;padding:18px;margin:18px 0;border-radius:8px}}
img{{display:block;max-width:900px;width:100%;margin:10px auto}}
a{{color:#1261a0}}
</style></head><body><main>
<h1>2024 Lost-lepton closure</h1>
<p>Independent event-fold MC closure and orthogonal data validation regions.</p>
{cards}
</main></body></html>
"""
    (output_dir / "index.html").write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--mc-closure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merged = load_json(args.merged)
    mc = load_json(args.mc_closure)
    decorate_mc_statistical_gates(mc)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_by_scheme = {
        scheme: list(record["labels"])
        for scheme, record in mc["schemes"].items()
    }
    full = None
    data_vr = None
    available = set(merged.get("processes", {}))
    if (
        set(TARGET_PROCESSES).issubset(available)
        and bool(available - set(TARGET_PROCESSES) - {DATA_PROCESS})
    ):
        full = build_full_mixture_closure(merged, labels_by_scheme)
        decorate_full_mixture_statistical_gates(full, mc)
        dump_json(output_dir / "full_mixture_mc_closure.json", full)
    if DATA_PROCESS in available:
        fixed_labels = {
            "highdm_nb0": [
                "250-300",
                "300-350",
                "350-400",
                "400-500",
                "500-800",
                "800-1500",
                "1500plus",
            ],
            "highdm_njet3to4_nb1plus": [
                "250-300",
                "300-350",
                "350-400",
                "400-500",
                "500-800",
                "800-1500",
                "1500plus",
            ],
            "lowdm_met250to300": ["250-275", "275-300"],
            "lowdm_isr200to300": [
                "200-225",
                "225-250",
                "250-275",
                "275-300",
            ],
            "lowdm_significance7to10": ["7-8", "8-9", "9-10"],
        }
        vr_specs = {
            name: {"labels": labels} for name, labels in fixed_labels.items()
        }
        data_vr = build_data_vr_closure(merged, vr_specs)
        dump_json(output_dir / "data_vr_closure.json", data_vr)
    dump_json(output_dir / "mc_closure.json", mc)
    plot_dir = output_dir / "plots"
    for scheme, record in mc["schemes"].items():
        plot_closure(
            record["crossfit"],
            list(record["labels"]),
            SCHEME_XLABELS[scheme],
            (
                (
                    r"High-$\Delta m$: independent A/B MC closure"
                    if scheme.startswith("highdm")
                    else r"Low-$\Delta m$: independent A/B MC closure"
                ),
                r"Target: $t\bar{t}+W\to\ell\nu+$ single top",
            ),
            plot_dir / f"mc_closure_{scheme}",
            is_observed_data=False,
        )
        plot_process_ratios(
            mc,
            scheme,
            list(record["labels"]),
            plot_dir / f"mc_closure_processes_{scheme}",
        )
        plot_process_transfer_factors(
            mc,
            scheme,
            list(record["labels"]),
            plot_dir / f"mc_transfer_factors_processes_{scheme}",
        )
    if full is not None:
        for scheme, record in full["schemes"].items():
            plot_closure(
                record["crossfit"],
                list(record["labels"]),
                SCHEME_XLABELS[scheme],
                (
                    (
                        r"High-$\Delta m$: full-mixture MC pseudodata"
                        if scheme.startswith("highdm")
                        else r"Low-$\Delta m$: full-mixture MC pseudodata"
                    ),
                    r"Non-lost-lepton backgrounds subtracted in control",
                ),
                plot_dir / f"full_mixture_mc_closure_{scheme}",
                is_observed_data=False,
            )
    if data_vr is not None and data_vr.get("status") == "complete":
        for name, record in data_vr["validation_regions"].items():
            branch, selection, xlabel = VR_LABELS[name]
            plot_closure(
                record,
                list(record["labels"]),
                xlabel,
                (branch, selection),
                plot_dir / f"data_vr_closure_{name}",
                is_observed_data=True,
                direct_label="Data \N{MINUS SIGN} other MC",
                ratio_label="Pred./residual",
            )
    write_summary(output_dir, merged, mc, full, data_vr)
    write_html(output_dir)
    dump_json(
        output_dir / "page_summary.json",
        {
            "status": "complete",
            "input_status": merged.get("status"),
            "duplicate_audit_status": (
                merged.get("duplicate_audit") or {}
            ).get("status"),
            "plots": [str(path) for path in sorted(plot_dir.glob("*.png"))],
            "nominal_inputs_modified": False,
        },
    )
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
