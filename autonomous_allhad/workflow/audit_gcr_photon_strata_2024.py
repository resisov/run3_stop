#!/usr/bin/env python3
"""Audit nominal-target yields in photon eta/pT strata from complete sidecars."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomous_allhad.photon_fake_2024_worker import OUTPUT_SCHEMA
from measure_photon_fakes_2024 import (
    _distribution_record,
    _merge_dataset_record,
    _origin_record,
    _stratum_transfer,
    aggregate_component,
    data_channels_from_events,
    deduplicate_data_events,
    load_histogram_builder,
    read_payload,
)
from study_gcr_datamc_improvement_2024 import metrics


PROCESSES = ("GJ", "QCD", "DY", "TT", "WtoLNu", "ST", "VV", "Zto2Nu")
ORIGINS = ("all", "prompt", "electron", "fake")


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


def weighted_target_from_dataset(
    dataset: dict[str, Any],
    normalization_factor: float,
    origin: str,
) -> tuple[np.ndarray, np.ndarray]:
    record = _origin_record(dataset.get("channels") or {}, "target", origin)
    if record is None:
        return np.zeros(8), np.zeros(8)
    values = np.asarray(
        [float(leaf["sumw"][0]) for leaf in record["transfer"]["strata"]]
    )
    variances = np.asarray(
        [float(leaf["sumw2"][0]) for leaf in record["transfer"]["strata"]]
    )
    return (
        normalization_factor * values,
        normalization_factor * normalization_factor * variances,
    )


def stratified_distribution(
    channels: dict[str, Any],
    probe: str,
    origin: str,
    region: str,
    variable: str,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    record = _distribution_record(
        channels, probe, origin, region, variable
    )
    if record is None:
        return np.zeros((8, bins)), np.zeros((8, bins))
    return (
        np.asarray([leaf["sumw"] for leaf in record["strata"]], dtype=float),
        np.asarray([leaf["sumw2"] for leaf in record["strata"]], dtype=float),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    paths = sorted(set(paths))
    datasets: dict[str, Any] = {}
    data_events: list[dict[str, Any]] = []
    sidecars = 0
    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != OUTPUT_SCHEMA:
            continue
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete sidecar: {path}")
        sidecars += 1
        data_events.extend(payload.get("data_events") or [])
        for physical, incoming in (payload.get("datasets") or {}).items():
            if physical not in datasets:
                datasets[physical] = incoming
            else:
                _merge_dataset_record(datasets[physical], incoming)
    if sidecars == 0:
        raise RuntimeError("no complete photon-fake sidecars found")

    normalization = read_payload(args.normalization)
    measurement = read_payload(args.measurement)
    physical_norm = normalization.get("physical_datasets") or {}
    builder = load_histogram_builder()
    deduplicated, dedup_audit = deduplicate_data_events(data_events)
    data_channels = data_channels_from_events(deduplicated, builder)
    data, data_var = _stratum_transfer(data_channels, "target", "all")

    process_records: dict[str, Any] = {}
    process_channels: dict[str, dict[str, dict[str, Any]]] = {}
    total_mc = np.zeros_like(data)
    total_mc_var = np.zeros_like(data)
    for process in PROCESSES:
        process_records[process] = {}
        for origin in ORIGINS:
            channels, audit = aggregate_component(
                datasets, normalization, origin, {process}
            )
            process_channels.setdefault(process, {})[origin] = channels
            values, variances = _stratum_transfer(channels, "target", origin)
            process_records[process][origin] = {
                "sumw": values.tolist(),
                "sumw2": variances.tolist(),
                "integral": float(np.sum(values)),
                "blocked_datasets": audit["blocked_datasets"],
            }
            if origin == "all":
                total_mc += values
                total_mc_var += variances

    physical_records: dict[str, Any] = {}
    for physical, dataset in sorted(datasets.items()):
        norm = physical_norm.get(physical) or {}
        factor = norm.get("normalization_factor")
        if factor is None or not np.isfinite(float(factor)):
            continue
        per_origin = {}
        for origin in ORIGINS:
            values, variances = weighted_target_from_dataset(
                dataset, float(factor), origin
            )
            per_origin[origin] = {
                "sumw": values.tolist(),
                "sumw2": variances.tolist(),
                "integral": float(np.sum(values)),
            }
        physical_records[physical] = {
            "process": dataset.get("process"),
            "normalization_factor": float(factor),
            "xsec_pb": norm.get("xsec_pb"),
            "sumw": norm.get("sumw"),
            "origins": per_origin,
        }

    labels_record = _origin_record(data_channels, "target", "all")
    labels = labels_record["transfer"]["transfer_labels"]
    ratio = np.divide(
        data,
        total_mc,
        out=np.full_like(data, np.nan),
        where=total_mc > 0.0,
    )
    prompt_channels, _ = aggregate_component(
        datasets, normalization, "prompt", set(PROCESSES)
    )
    electron_channels, _ = aggregate_component(
        datasets, normalization, "electron", set(PROCESSES)
    )
    data_application, data_application_var = _stratum_transfer(
        data_channels, "application", "all"
    )
    prompt_application, prompt_application_var = _stratum_transfer(
        prompt_channels, "application", "prompt"
    )
    electron_application, electron_application_var = _stratum_transfer(
        electron_channels, "application", "electron"
    )
    factors = np.asarray(
        [
            float(record["factor"])
            for record in measurement["measurement"]["central_transfer_factors"]
        ],
        dtype=float,
    )
    application_residual = (
        data_application - prompt_application - electron_application
    )
    data_driven_fake = np.maximum(0.0, factors * application_residual)
    data_driven_fake_var = np.square(factors) * (
        data_application_var
        + prompt_application_var
        + electron_application_var
    )
    gj_all = np.asarray(process_records["GJ"]["all"]["sumw"], dtype=float)
    qcd_prompt = np.asarray(
        process_records["QCD"]["prompt"]["sumw"], dtype=float
    )
    other_all = np.zeros_like(data)
    for process in PROCESSES:
        if process not in {"GJ", "QCD"}:
            other_all += np.asarray(
                process_records[process]["all"]["sumw"], dtype=float
            )
    prompt_pool = gj_all + qcd_prompt
    fixed = other_all + data_driven_fake
    prompt_alpha = np.divide(
        data - fixed,
        prompt_pool,
        out=np.full_like(data, np.nan),
        where=prompt_pool > 0.0,
    )
    eta_fits = {}
    for eta, indices in {"EB": np.arange(0, 4), "EE": np.arange(4, 8)}.items():
        numerator = float(np.sum(data[indices] - fixed[indices]))
        denominator = float(np.sum(prompt_pool[indices]))
        eta_fits[eta] = {
            "alpha_integral": numerator / denominator,
            "data_stat_sigma": (
                math.sqrt(float(np.sum(data[indices]))) / denominator
            ),
            "data_integral": float(np.sum(data[indices])),
            "fixed_integral": float(np.sum(fixed[indices])),
            "prompt_pool_integral": denominator,
        }
    global_alpha = float(np.sum(data - fixed) / np.sum(prompt_pool))
    eta_alpha = np.asarray(
        [eta_fits["EB"]["alpha_integral"]] * 4
        + [eta_fits["EE"]["alpha_integral"]] * 4,
        dtype=float,
    )

    stratified_study: dict[str, Any] = {}
    variables: list[tuple[str, str, list[float]]] = [
        ("GCR", "recoil", [float(x) for x in builder.RECOIL_PT_BINS]),
        ("GCR_Nt0", "recoil", [float(x) for x in builder.RECOIL_PT_BINS]),
        ("GCR_Nt1", "recoil", [float(x) for x in builder.RECOIL_PT_BINS]),
    ]
    variables.extend(
        (
            "GCR",
            variable,
            [float(x) for x in spec["bins"]],
        )
        for variable, spec in builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
        if variable != "met"
    )
    for region, variable, edges in variables:
        bins = len(edges) - 1
        data_s, data_var_s = stratified_distribution(
            data_channels, "target", "all", region, variable, bins
        )
        gj_s, gj_var_s = stratified_distribution(
            process_channels["GJ"]["all"],
            "target",
            "all",
            region,
            variable,
            bins,
        )
        qcd_all_s, qcd_all_var_s = stratified_distribution(
            process_channels["QCD"]["all"],
            "target",
            "all",
            region,
            variable,
            bins,
        )
        qcd_prompt_s, qcd_prompt_var_s = stratified_distribution(
            process_channels["QCD"]["prompt"],
            "target",
            "prompt",
            region,
            variable,
            bins,
        )
        other_s = np.zeros_like(data_s)
        other_var_s = np.zeros_like(data_s)
        for process in PROCESSES:
            if process in {"GJ", "QCD"}:
                continue
            values, variances = stratified_distribution(
                process_channels[process]["all"],
                "target",
                "all",
                region,
                variable,
                bins,
            )
            other_s += values
            other_var_s += variances
        app_data_s, app_data_var_s = stratified_distribution(
            data_channels, "application", "all", region, variable, bins
        )
        app_prompt_s, app_prompt_var_s = stratified_distribution(
            prompt_channels,
            "application",
            "prompt",
            region,
            variable,
            bins,
        )
        app_electron_s, app_electron_var_s = stratified_distribution(
            electron_channels,
            "application",
            "electron",
            region,
            variable,
            bins,
        )
        fake_s = factors[:, None] * (
            app_data_s - app_prompt_s - app_electron_s
        )
        fake_var_s = np.square(factors[:, None]) * (
            app_data_var_s + app_prompt_var_s + app_electron_var_s
        )
        data_hist = np.sum(data_s, axis=0)
        data_hist_var = np.sum(data_var_s, axis=0)
        prompt_s = gj_s + qcd_prompt_s
        prompt_var_s = gj_var_s + qcd_prompt_var_s
        fixed_s = other_s + fake_s
        fixed_var_s = other_var_s + fake_var_s
        predictions = {
            "sidecar_nominal": (
                np.sum(other_s + gj_s + qcd_all_s, axis=0),
                np.sum(other_var_s + gj_var_s + qcd_all_var_s, axis=0),
            ),
            "origin_dd_unfitted": (
                np.sum(fixed_s + prompt_s, axis=0),
                np.sum(fixed_var_s + prompt_var_s, axis=0),
            ),
            "prompt_global_fit": (
                np.sum(fixed_s + global_alpha * prompt_s, axis=0),
                np.sum(
                    fixed_var_s
                    + global_alpha * global_alpha * prompt_var_s,
                    axis=0,
                ),
            ),
            "prompt_eb_ee_fit": (
                np.sum(fixed_s + eta_alpha[:, None] * prompt_s, axis=0),
                np.sum(
                    fixed_var_s
                    + np.square(eta_alpha[:, None]) * prompt_var_s,
                    axis=0,
                ),
            ),
            "prompt_eight_strata_fit_diagnostic": (
                np.sum(fixed_s + prompt_alpha[:, None] * prompt_s, axis=0),
                np.sum(
                    fixed_var_s
                    + np.square(prompt_alpha[:, None]) * prompt_var_s,
                    axis=0,
                ),
            ),
        }
        stratified_study[f"{region}/{variable}"] = {
            "bin_edges": edges,
            "data": data_hist.tolist(),
            "data_variance": data_hist_var.tolist(),
            "predictions": {
                name: {
                    "sumw": values.tolist(),
                    "sumw2": variances.tolist(),
                    "metrics": metrics(data_hist, values, variances),
                }
                for name, (values, variances) in predictions.items()
            },
        }
    payload = {
        "schema_version": "gcr_photon_strata_audit_v1",
        "status": "complete",
        "selection_source": "real_subset_worker.py",
        "sidecars": sidecars,
        "normalization": str(args.normalization),
        "measurement": str(args.measurement),
        "luminosity_pb": normalization.get("luminosity_pb"),
        "transfer_labels": labels,
        "data": {
            "sumw": data.tolist(),
            "sumw2": data_var.tolist(),
            "integral": float(np.sum(data)),
        },
        "total_mc": {
            "sumw": total_mc.tolist(),
            "sumw2": total_mc_var.tolist(),
            "integral": float(np.sum(total_mc)),
        },
        "data_over_mc": [
            None if not np.isfinite(value) else float(value) for value in ratio
        ],
        "processes": process_records,
        "data_driven_fake": {
            "sumw": data_driven_fake.tolist(),
            "sumw2": data_driven_fake_var.tolist(),
            "integral": float(np.sum(data_driven_fake)),
            "application_data": data_application.tolist(),
            "application_prompt": prompt_application.tolist(),
            "application_electron": electron_application.tolist(),
        },
        "prompt_pool_diagnostic": {
            "definition": (
                "GJ all-origin plus QCD truth-prompt; fixed component is all "
                "other process yields plus the data-driven fake estimate"
            ),
            "prompt_pool": prompt_pool.tolist(),
            "fixed": fixed.tolist(),
            "per_stratum_alpha": [
                None if not np.isfinite(value) else float(value)
                for value in prompt_alpha
            ],
            "eta_integral_fits": eta_fits,
            "global_alpha_integral": global_alpha,
        },
        "stratified_distribution_study": stratified_study,
        "physical_datasets": physical_records,
        "data_deduplication": dedup_audit,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "sidecars": sidecars,
                "data": float(np.sum(data)),
                "mc": float(np.sum(total_mc)),
                "data_over_mc": payload["data_over_mc"],
                "deduplication": dedup_audit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
