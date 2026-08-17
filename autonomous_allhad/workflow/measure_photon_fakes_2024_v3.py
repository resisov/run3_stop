#!/usr/bin/env python3
"""Cross-validated, correlation-corrected fake-photon ABCD measurement.

This measurement consumes the complete v1 sidecars, whose photon target is the
trusted ``real_subset_worker.py`` GCR selection and whose sidebands are defined
by passing/failing the *medium* photon VID components.  The data transfer factor
is C/D.  A truth-fake QCD correction

    kappa = A D / (B C)

accounts for the residual correlation between shower shape and charged
isolation.  QCD shards are split deterministically into two folds; each fold is
predicted with kappa measured in the other fold.  Nominal-GCR target data never
enter either the transfer-factor or kappa fit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from autonomous_allhad.photon_fake_2024_worker import (
    GCR_REGIONS,
    TRANSFER_PT_EDGES,
    load_histogram_builder,
)
from measure_photon_fakes_2024 import (
    ELECTRON_NORMALIZATION_UNCERTAINTY,
    MAX_FAKE_FACTOR,
    PROMPT_NORMALIZATION_UNCERTAINTY,
    _collapse_stratified,
    _distribution_record,
    _leaf_integral,
    _merge_dataset_record,
    _stratum_transfer,
    _sum_stratified,
    aggregate_component,
    data_channels_from_events,
    deduplicate_data_events,
    diagnostic_histograms,
    fit_transfer_factors,
    qcd_target_origin_histograms,
    read_payload,
    write_payload,
)
from measure_photon_fakes_2024_v2 import (
    predict_distribution as predict_distribution_covariance,
)


INPUT_SCHEMA = "photon_fake_2024_sidecar_shard_v1"
MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v3"
BACKGROUND_PROCESSES = {
    "DY",
    "GJ",
    "QCD",
    "ST",
    "TT",
    "VV",
    "WtoLNu",
    "Zto2Nu",
}
ABCD_PROBES = {
    "A": "target",
    "B": "application",
    "C": "measurement_pass",
    "D": "measurement_fail",
}
MAX_RELATIVE_KAPPA_UNCERTAINTY = 0.75
MIN_COMPONENT_EFFECTIVE_EVENTS = 4.0
MIN_CLOSURE_UNCERTAINTY = 0.20
SHARD_PATTERN = re.compile(r"_(\d+)\.json\.gz$")


def _merge_into(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for physical, record in incoming.items():
        if physical not in target:
            target[physical] = copy.deepcopy(record)
        else:
            _merge_dataset_record(target[physical], record)


def _fold_for_path(path: Path) -> int:
    match = SHARD_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"cannot determine deterministic fold for {path}")
    return int(match.group(1)) % 2


def _candidate(
    components: dict[str, np.ndarray],
    variances: dict[str, np.ndarray],
    indices: list[int],
) -> dict[str, Any]:
    values = {
        label: float(np.sum(components[label][indices]))
        for label in ABCD_PROBES
    }
    value_variances = {
        label: float(np.sum(variances[label][indices]))
        for label in ABCD_PROBES
    }
    effective = {
        label: (
            values[label] * values[label] / value_variances[label]
            if value_variances[label] > 0.0
            else 0.0
        )
        for label in ABCD_PROBES
    }
    positive = all(values[label] > 0.0 for label in ABCD_PROBES)
    enough = all(
        effective[label] >= MIN_COMPONENT_EFFECTIVE_EVENTS
        for label in ABCD_PROBES
    )
    if not positive:
        return {
            **values,
            **{f"{key}_variance": value for key, value in value_variances.items()},
            "effective_events": effective,
            "indices": indices,
            "valid": False,
            "acceptable": False,
            "value": None,
            "variance": None,
            "relative_uncertainty": None,
        }
    kappa = values["A"] * values["D"] / (values["B"] * values["C"])
    relative_variance = sum(
        value_variances[label] / (values[label] * values[label])
        for label in ABCD_PROBES
    )
    variance = kappa * kappa * relative_variance
    relative_uncertainty = math.sqrt(max(0.0, relative_variance))
    valid = bool(np.isfinite(kappa) and np.isfinite(variance) and kappa > 0.0)
    acceptable = bool(
        valid
        and enough
        and relative_uncertainty <= MAX_RELATIVE_KAPPA_UNCERTAINTY
        and kappa <= MAX_FAKE_FACTOR
    )
    return {
        **values,
        **{f"{key}_variance": value for key, value in value_variances.items()},
        "effective_events": effective,
        "indices": indices,
        "valid": valid,
        "acceptable": acceptable,
        "value": float(kappa) if valid else None,
        "variance": float(variance) if valid else None,
        "relative_uncertainty": (
            float(relative_uncertainty) if valid else None
        ),
    }


def fit_qcd_kappa(qcd_fake_channels: dict[str, Any]) -> dict[str, Any]:
    count = 2 * (len(TRANSFER_PT_EDGES) - 1)
    components: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for label, probe in ABCD_PROBES.items():
        values, value_variances = _stratum_transfer(
            qcd_fake_channels,
            probe,
            "fake",
        )
        components[label] = values
        variances[label] = value_variances

    pt_bins = len(TRANSFER_PT_EDGES) - 1
    direct = {
        index: _candidate(components, variances, [index])
        for index in range(count)
    }
    coarse_indices: dict[str, list[int]] = {}
    for eta_index, eta in enumerate(("EB", "EE")):
        offset = eta_index * pt_bins
        coarse_indices[f"{eta}_pt220to400"] = [offset, offset + 1]
        coarse_indices[f"{eta}_pt400toinf"] = [offset + 2, offset + 3]
    coarse = {
        key: _candidate(components, variances, indices)
        for key, indices in coarse_indices.items()
    }
    eta = {
        eta_index: _candidate(
            components,
            variances,
            list(range(eta_index * pt_bins, (eta_index + 1) * pt_bins)),
        )
        for eta_index in range(2)
    }
    global_candidate = _candidate(
        components,
        variances,
        list(range(count)),
    )
    labels = []
    for eta_label in ("EB", "EE"):
        for low, high in zip(TRANSFER_PT_EDGES[:-1], TRANSFER_PT_EDGES[1:]):
            labels.append(
                f"{eta_label}_pt{low:g}to"
                + ("inf" if high >= 1_000_000.0 else f"{high:g}")
            )

    factors = np.ones(count)
    covariance = np.zeros((count, count))
    choices: list[dict[str, Any]] = []
    chosen_candidates: list[dict[str, Any]] = []
    covariance_keys: list[str] = []
    for index, label in enumerate(labels):
        eta_index = index // pt_bins
        coarse_key = (
            f"{('EB', 'EE')[eta_index]}_"
            + ("pt220to400" if index % pt_bins < 2 else "pt400toinf")
        )
        options = (
            (direct[index], f"direct:{index}", "direct"),
            (coarse[coarse_key], f"coarse:{coarse_key}", "coarse"),
            (eta[eta_index], f"eta:{eta_index}", "eta_inclusive"),
            (global_candidate, "global", "global"),
        )
        selected = next((item for item in options if item[0]["acceptable"]), None)
        if selected is None:
            valid = next((item for item in options[::-1] if item[0]["valid"]), None)
            if valid is None:
                candidate = {"value": 1.0, "variance": 1.0}
                covariance_key = f"unit:{coarse_key}"
                source = "unit_unconstrained"
            else:
                raw, raw_key, _ = valid
                uncertainty = max(
                    0.50,
                    abs(float(raw["value"]) - 1.0),
                    math.sqrt(float(raw["variance"])),
                )
                candidate = {"value": 1.0, "variance": uncertainty * uncertainty}
                covariance_key = f"regularized:{raw_key}"
                source = "unit_regularized_imprecise_qcd"
        else:
            candidate, covariance_key, source = selected
        factors[index] = float(candidate["value"])
        chosen_candidates.append(candidate)
        covariance_keys.append(covariance_key)
        choices.append(
            {
                "index": index,
                "label": label,
                "factor": float(candidate["value"]),
                "factor_uncertainty": math.sqrt(float(candidate["variance"])),
                "source": source,
                "covariance_key": covariance_key,
            }
        )
    for first in range(count):
        for second in range(count):
            if covariance_keys[first] == covariance_keys[second]:
                covariance[first, second] = math.sqrt(
                    max(
                        0.0,
                        float(chosen_candidates[first]["variance"])
                        * float(chosen_candidates[second]["variance"]),
                    )
                )
    return {
        "factors": factors,
        "variances": np.maximum(np.diag(covariance), 0.0),
        "covariance": covariance,
        "records": choices,
        "candidates": {
            "direct": direct,
            "coarse": coarse,
            "eta": eta,
            "global": global_candidate,
        },
        "policy": (
            "truth-fake QCD kappa=A*D/(B*C); direct pT-eta strata are used "
            "only with >=4 effective events in every ABCD component and <=75% "
            "relative uncertainty, followed by coarse-pT, eta-inclusive, and "
            "global fallbacks; imprecise values are regularized to unity"
        ),
    }


def combine_data_tf_and_kappa(
    transfer_fit: dict[str, Any],
    kappa_fit: dict[str, Any],
) -> dict[str, Any]:
    transfer = np.asarray(transfer_fit["factors"], dtype=float)
    transfer_variance = np.asarray(transfer_fit["variances"], dtype=float)
    kappa = np.asarray(kappa_fit["factors"], dtype=float)
    kappa_covariance = np.asarray(kappa_fit["covariance"], dtype=float)
    factors = transfer * kappa
    covariance = np.outer(transfer, transfer) * kappa_covariance
    covariance += np.diag(np.square(kappa) * transfer_variance)
    records = []
    for index, base in enumerate(transfer_fit["records"]):
        records.append(
            {
                **base,
                "data_transfer_factor": float(transfer[index]),
                "data_transfer_factor_uncertainty": math.sqrt(
                    max(0.0, float(transfer_variance[index]))
                ),
                "kappa": float(kappa[index]),
                "kappa_uncertainty": math.sqrt(
                    max(0.0, float(kappa_covariance[index, index]))
                ),
                "corrected_factor": float(factors[index]),
                "corrected_factor_uncertainty": math.sqrt(
                    max(0.0, float(covariance[index, index]))
                ),
                "kappa_source": kappa_fit["records"][index]["source"],
            }
        )
    return {
        "factors": factors,
        "variances": np.maximum(np.diag(covariance), 0.0),
        "covariance": covariance,
        "records": records,
    }


def _qcd_as_data(qcd_fake_channels: dict[str, Any]) -> dict[str, Any]:
    return {
        probe: {"all": origins["fake"]}
        for probe, origins in qcd_fake_channels.items()
        if "fake" in origins
    }


def _closure_fold(
    training: dict[str, Any],
    validation: dict[str, Any],
    builder: Any,
) -> dict[str, Any]:
    blank: dict[str, Any] = {}
    kappa = fit_qcd_kappa(training)
    validation_as_data = _qcd_as_data(validation)
    transfer = fit_transfer_factors(validation_as_data, blank, blank)
    corrected = combine_data_tf_and_kappa(transfer, kappa)
    edges = [float(value) for value in builder.RECOIL_PT_BINS]
    prediction, audit = predict_distribution_covariance(
        validation_as_data,
        blank,
        blank,
        "GCR",
        "recoil",
        edges,
        corrected["factors"],
        corrected["variances"],
        corrected["covariance"],
        probes=("application",),
    )
    target_record = _distribution_record(
        validation_as_data,
        "target",
        "all",
        "GCR",
        "recoil",
    )
    target_histogram = _collapse_stratified(target_record, edges)
    target, target_variance = _sum_stratified(target_record)
    predicted, _ = _leaf_integral(prediction)
    predicted_variance = float(audit["integral_variance_with_correlations"])
    ratio = target / predicted if target > 0.0 and predicted > 0.0 else None
    ratio_uncertainty = None
    if ratio is not None:
        ratio_variance = (
            target_variance / (predicted * predicted)
            + target
            * target
            * predicted_variance
            / (predicted**4)
        )
        ratio_uncertainty = math.sqrt(max(0.0, ratio_variance))
    return {
        "target": target,
        "target_variance": target_variance,
        "prediction": predicted,
        "prediction_variance": predicted_variance,
        "target_over_prediction": ratio,
        "ratio_uncertainty": ratio_uncertainty,
        "target_histogram": target_histogram,
        "prediction_histogram": prediction,
        "prediction_audit": audit,
        "kappa_records": kappa["records"],
        "corrected_factor_records": corrected["records"],
    }


def cross_validated_closure(
    folds: list[dict[str, Any]],
    builder: Any,
) -> dict[str, Any]:
    results = [
        _closure_fold(folds[1], folds[0], builder),
        _closure_fold(folds[0], folds[1], builder),
    ]
    target = sum(record["target"] for record in results)
    target_variance = sum(record["target_variance"] for record in results)
    prediction = sum(record["prediction"] for record in results)
    prediction_variance = sum(
        record["prediction_variance"] for record in results
    )
    ratio = target / prediction if target > 0.0 and prediction > 0.0 else None
    ratio_uncertainty = None
    if ratio is not None:
        ratio_variance = (
            target_variance / (prediction * prediction)
            + target * target * prediction_variance / (prediction**4)
        )
        ratio_uncertainty = math.sqrt(max(0.0, ratio_variance))
        fold_ratios = [
            value
            for value in (
                record["target_over_prediction"] for record in results
            )
            if value is not None
        ]
        fold_spread = (
            0.5 * abs(fold_ratios[0] - fold_ratios[1])
            if len(fold_ratios) == 2
            else 1.0
        )
        assigned = min(
            1.0,
            max(
                MIN_CLOSURE_UNCERTAINTY,
                abs(ratio - 1.0),
                ratio_uncertainty,
                fold_spread,
            ),
        )
        status = "measured"
    else:
        fold_spread = None
        assigned = 1.0
        status = "insufficient_qcd_fake_statistics"

    target_bins = sum(
        (
            np.asarray(record["target_histogram"]["sumw"], dtype=float)
            for record in results
        ),
        start=np.zeros(len(builder.RECOIL_PT_BINS) - 1),
    )
    prediction_bins = sum(
        (
            np.asarray(record["prediction_histogram"]["sumw"], dtype=float)
            for record in results
        ),
        start=np.zeros(len(builder.RECOIL_PT_BINS) - 1),
    )
    raw_ratio = np.divide(
        target_bins,
        prediction_bins,
        out=np.ones_like(target_bins),
        where=prediction_bins > 0.0,
    )
    raw_deviation = np.clip(raw_ratio - 1.0, -1.0, 1.0)
    if float(np.sum(prediction_bins)) > 0.0:
        normalization_component = float(
            np.average(raw_deviation, weights=prediction_bins)
        )
    else:
        normalization_component = 0.0
    shape_deviation = np.clip(
        raw_deviation - normalization_component,
        -1.0,
        1.0,
    )
    return {
        "status": status,
        "strategy": (
            "two-fold cross-validation: even QCD shards are predicted with "
            "kappa from odd shards and vice versa"
        ),
        "folds": results,
        "global_target": target,
        "global_target_variance": target_variance,
        "global_prediction": prediction,
        "global_prediction_variance": prediction_variance,
        "global_target_over_prediction": ratio,
        "global_ratio_uncertainty": ratio_uncertainty,
        "fold_ratio_half_spread": fold_spread,
        "assigned_relative_nonclosure": assigned,
        "recoil_target": target_bins.tolist(),
        "recoil_prediction": prediction_bins.tolist(),
        "recoil_target_over_prediction": raw_ratio.tolist(),
        "recoil_shape_deviation": shape_deviation.tolist(),
        "central_value_policy": (
            "cross-validation does not fit or rescale the data central value; "
            "it defines the closure normalization and recoil-shape uncertainties"
        ),
    }


def _scaled(leaf: dict[str, Any], scale: float) -> dict[str, Any]:
    output = copy.deepcopy(leaf)
    output["sumw"] = [
        max(0.0, scale * float(value)) for value in leaf["sumw"]
    ]
    return output


def _shape_variations(
    nominal: dict[str, Any],
    deviation: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(nominal["sumw"], dtype=float)
    delta = np.asarray(deviation, dtype=float)
    if len(values) != len(delta):
        return copy.deepcopy(nominal), copy.deepcopy(nominal)
    magnitude = np.minimum(1.0, np.abs(delta))
    up = copy.deepcopy(nominal)
    down = copy.deepcopy(nominal)
    up["sumw"] = (values * (1.0 + magnitude)).tolist()
    down["sumw"] = (values * (1.0 - magnitude)).tolist()
    return up, down


def build_prediction(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    kappa: dict[str, Any],
    closure: dict[str, Any],
    builder: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    central_tf = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
    )
    central = combine_data_tf_and_kappa(central_tf, kappa)
    prompt_up = combine_data_tf_and_kappa(
        fit_transfer_factors(
            data_channels,
            prompt_channels,
            electron_channels,
            prompt_scale=1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
        ),
        kappa,
    )
    prompt_down = combine_data_tf_and_kappa(
        fit_transfer_factors(
            data_channels,
            prompt_channels,
            electron_channels,
            prompt_scale=1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
        ),
        kappa,
    )
    electron_up = combine_data_tf_and_kappa(
        fit_transfer_factors(
            data_channels,
            prompt_channels,
            electron_channels,
            electron_scale=1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
        ),
        kappa,
    )
    electron_down = combine_data_tf_and_kappa(
        fit_transfer_factors(
            data_channels,
            prompt_channels,
            electron_channels,
            electron_scale=1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
        ),
        kappa,
    )
    output: dict[str, Any] = {
        "histograms": {},
        "highdm_variable_histograms": {},
    }
    audits: dict[str, Any] = {}
    closure_uncertainty = float(closure["assigned_relative_nonclosure"])
    for region in ("GCR", "GCR_Nt0", "GCR_Nt1"):
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(value) for value in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            nominal, audit = predict_distribution_covariance(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                central["factors"],
                central["variances"],
                central["covariance"],
                probes=("application",),
            )
            uncertainty = np.sqrt(np.maximum(central["variances"], 0.0))
            variations = {"nominal": nominal}
            for name, factors in (
                (
                    "photonFakeStatUp",
                    np.minimum(MAX_FAKE_FACTOR, central["factors"] + uncertainty),
                ),
                (
                    "photonFakeStatDown",
                    np.maximum(0.0, central["factors"] - uncertainty),
                ),
            ):
                variations[name], _ = predict_distribution_covariance(
                    data_channels,
                    prompt_channels,
                    electron_channels,
                    region,
                    variable,
                    edges,
                    factors,
                    np.zeros_like(factors),
                    probes=("application",),
                )
            for name, fit, prompt_scale, electron_scale in (
                (
                    "photonFakePromptUp",
                    prompt_up,
                    1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
                    1.0,
                ),
                (
                    "photonFakePromptDown",
                    prompt_down,
                    1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
                    1.0,
                ),
                (
                    "photonFakeElectronUp",
                    electron_up,
                    1.0,
                    1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
                ),
                (
                    "photonFakeElectronDown",
                    electron_down,
                    1.0,
                    1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
                ),
            ):
                variations[name], _ = predict_distribution_covariance(
                    data_channels,
                    prompt_channels,
                    electron_channels,
                    region,
                    variable,
                    edges,
                    fit["factors"],
                    np.zeros_like(fit["factors"]),
                    probes=("application",),
                    prompt_scale=prompt_scale,
                    electron_scale=electron_scale,
                )
            variations["photonFakeClosureUp"] = _scaled(
                nominal,
                1.0 + closure_uncertainty,
            )
            variations["photonFakeClosureDown"] = _scaled(
                nominal,
                max(0.0, 1.0 - closure_uncertainty),
            )
            if variable == "recoil":
                shape_up, shape_down = _shape_variations(
                    nominal,
                    closure["recoil_shape_deviation"],
                )
            else:
                shape_up, shape_down = copy.deepcopy(nominal), copy.deepcopy(nominal)
            variations["photonFakeClosureShapeUp"] = shape_up
            variations["photonFakeClosureShapeDown"] = shape_down
            if variable == "recoil":
                output["histograms"][region] = variations
            else:
                output["highdm_variable_histograms"].setdefault(region, {})[
                    variable
                ] = variations
            audits[f"{region}/{variable}"] = audit
    return output, {
        "central_corrected_factors": central["records"],
        "kappa_records": kappa["records"],
        "prediction_audits": audits,
        "prompt_normalization_uncertainty": PROMPT_NORMALIZATION_UNCERTAINTY,
        "electron_normalization_uncertainty": ELECTRON_NORMALIZATION_UNCERTAINTY,
        "closure_uncertainty": closure_uncertainty,
    }


def target_summary(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, float] = {}
    for name, channels, origin in (
        ("data", data_channels, "all"),
        ("prompt", prompt_channels, "prompt"),
        ("electron", electron_channels, "electron"),
    ):
        record = _distribution_record(
            channels,
            "target",
            origin,
            "GCR",
            "recoil",
        )
        values[name], values[f"{name}_variance"] = _sum_stratified(record)
    nominal = prediction["histograms"]["GCR"]["nominal"]
    fake, fake_diagonal_variance = _leaf_integral(nominal)
    total = values["prompt"] + values["electron"] + fake
    return {
        "data_target": values["data"],
        "data_target_variance": values["data_variance"],
        "prompt_target": values["prompt"],
        "prompt_target_variance": values["prompt_variance"],
        "electron_target": values["electron"],
        "electron_target_variance": values["electron_variance"],
        "observed_fake_residual_not_used_in_fit": (
            values["data"] - values["prompt"] - values["electron"]
        ),
        "predicted_fake": fake,
        "predicted_fake_diagonal_variance": fake_diagonal_variance,
        "prompt_plus_electron_plus_fake": total,
        "prediction_over_data": (
            total / values["data"] if values["data"] > 0.0 else None
        ),
        "blinding_policy": (
            "nominal-GCR target data are validation only and do not enter the "
            "data transfer-factor or simulation-kappa measurement"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("no photon-fake sidecars found")

    datasets: dict[str, Any] = {}
    qcd_fold_datasets = [{}, {}]
    data_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    accepted_paths: list[Path] = []
    digests: set[str] = set()
    incomplete: list[str] = []
    sidecar_variable_specs: dict[str, Any] | None = None
    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != INPUT_SCHEMA:
            continue
        incoming_specs = payload.get("highdm_distribution_variable_specs") or {}
        if sidecar_variable_specs is None:
            sidecar_variable_specs = copy.deepcopy(incoming_specs)
        elif incoming_specs != sidecar_variable_specs:
            raise RuntimeError(
                f"high-dM variable schema differs across sidecars: {path}"
            )
        accepted_paths.append(path)
        summary = payload.get("summary") or {}
        digest = str(summary.get("source_record_digest") or "")
        if digest and digest in digests:
            raise RuntimeError(f"duplicate source record digest: {digest}")
        if digest:
            digests.add(digest)
        if payload.get("status") != "complete":
            incomplete.append(str(path))
        summaries.append(summary)
        data_events.extend(payload.get("data_events") or [])
        incoming = payload.get("datasets") or {}
        _merge_into(datasets, incoming)
        qcd_incoming = {
            physical: record
            for physical, record in incoming.items()
            if str(record.get("process") or "") == "QCD"
        }
        if qcd_incoming:
            _merge_into(qcd_fold_datasets[_fold_for_path(path)], qcd_incoming)
    if not accepted_paths:
        raise RuntimeError(f"no {INPUT_SCHEMA} sidecars found")
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} sidecars are incomplete")

    builder = load_histogram_builder()
    # The common histogram builder may gain variables after a frozen sidecar
    # campaign (for example the DY-only ptll diagnostic).  Reconstruct exactly
    # the variable schema embedded in the complete photon sidecars.
    builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS = (
        sidecar_variable_specs or {}
    )
    normalization = read_payload(args.normalization)
    deduplicated, dedup_audit = deduplicate_data_events(data_events)
    data_channels = data_channels_from_events(deduplicated, builder)
    prompt_channels, prompt_audit = aggregate_component(
        datasets,
        normalization,
        "prompt",
        BACKGROUND_PROCESSES,
    )
    electron_channels, electron_audit = aggregate_component(
        datasets,
        normalization,
        "electron",
        BACKGROUND_PROCESSES,
    )
    qcd_fake_channels, qcd_fake_audit = aggregate_component(
        datasets,
        normalization,
        "fake",
        {"QCD"},
    )
    fold_channels = [
        aggregate_component(
            fold,
            normalization,
            "fake",
            {"QCD"},
        )[0]
        for fold in qcd_fold_datasets
    ]
    process_origin_channels: dict[str, dict[str, Any]] = {}
    process_origin_audits: dict[str, dict[str, Any]] = {}
    for process in sorted(BACKGROUND_PROCESSES):
        process_origin_channels[process] = {}
        process_origin_audits[process] = {}
        for origin in ("all", "prompt", "electron", "fake"):
            channels, audit = aggregate_component(
                datasets,
                normalization,
                origin,
                {process},
            )
            process_origin_channels[process][origin] = channels
            process_origin_audits[process][origin] = audit

    kappa = fit_qcd_kappa(qcd_fake_channels)
    closure = cross_validated_closure(fold_channels, builder)
    prediction, measurement = build_prediction(
        data_channels,
        prompt_channels,
        electron_channels,
        kappa,
        closure,
        builder,
    )
    validation = target_summary(
        data_channels,
        prompt_channels,
        electron_channels,
        prediction,
    )
    processes = sorted(
        {
            str(record.get("process") or "unknown")
            for record in datasets.values()
        }
        | ({"EGamma"} if data_events else set())
    )
    files_attempted = sum(
        int(summary.get("files_attempted") or 0) for summary in summaries
    )
    files_processed = sum(
        int(summary.get("files_processed") or 0) for summary in summaries
    )
    coverage_errors: list[str] = []
    coverage: dict[str, Any] = {
        "observed_sidecars": len(summaries),
        "observed_files_attempted": files_attempted,
        "observed_files_processed": files_processed,
        "observed_processes": processes,
    }
    if args.campaign_manifest is not None:
        manifest = read_payload(args.campaign_manifest)
        expected_sidecars = int(manifest.get("jobs") or 0)
        expected_records = sum(
            int(value) for value in (manifest.get("record_counts") or {}).values()
        )
        expected_processes = sorted(manifest.get("requested_processes") or [])
        if len(summaries) != expected_sidecars:
            coverage_errors.append(
                f"sidecars {len(summaries)} != expected {expected_sidecars}"
            )
        if files_attempted != expected_records:
            coverage_errors.append(
                f"files attempted {files_attempted} != expected {expected_records}"
            )
        if files_processed != expected_records:
            coverage_errors.append(
                f"files processed {files_processed} != expected {expected_records}"
            )
        missing = sorted(set(expected_processes) - set(processes))
        if missing:
            coverage_errors.append(f"missing processes {missing}")
        coverage.update(
            {
                "campaign_manifest": str(args.campaign_manifest),
                "expected_sidecars": expected_sidecars,
                "expected_records": expected_records,
                "expected_processes": expected_processes,
            }
        )
    else:
        coverage_errors.append("campaign manifest not supplied")
    coverage["errors"] = coverage_errors
    coverage["status"] = "complete" if not coverage_errors else "incomplete"
    if coverage_errors:
        raise RuntimeError(
            "photon-fake v3 coverage is incomplete: "
            + json.dumps(coverage, sort_keys=True)
        )

    qcd_origins = process_origin_channels["QCD"]
    output = {
        "schema_version": MEASUREMENT_SCHEMA,
        "status": "complete",
        "year": 2024,
        "scope": "high-dM photon control region",
        "selection_source": "real_subset_worker.py",
        "nominal_intermediate_policy": (
            "read-only complete sidecars; nominal intermediates are unchanged"
        ),
        "method": {
            "abcd": (
                "medium pass/fail ABCD in sigma_ieta_ieta and charged "
                "isolation: A=target, B=charged-isolation fail, "
                "C=shower-shape fail, D=both fail"
            ),
            "data_transfer_factor": (
                "(C_data-C_prompt-C_electron)/(D_data-D_prompt-D_electron)"
            ),
            "correlation_correction": (
                "truth-fake QCD kappa=A*D/(B*C), measured without target data"
            ),
            "validation": (
                "deterministic even/odd QCD-shard two-fold cross-validation"
            ),
            "shape": (
                "prompt/electron-subtracted charged-isolation-fail data in "
                "every nominal GCR distribution"
            ),
            "statistics": (
                "target variance, prediction variance, shared-kappa covariance, "
                "fold spread, and recoil nonclosure shape retained"
            ),
        },
        "input_sidecars": [str(path) for path in accepted_paths],
        "normalization_source": str(args.normalization),
        "coverage": coverage,
        "input_summary": {
            "sidecar_count": len(summaries),
            "files_attempted": files_attempted,
            "files_processed": files_processed,
            "events_read": sum(
                int(summary.get("events_read") or 0) for summary in summaries
            ),
            "selected_events": sum(
                int(summary.get("selected_events") or 0) for summary in summaries
            ),
            "processes": processes,
            "target_cutbased_mismatch_objects": sum(
                int(summary.get("target_cutbased_mismatch_objects") or 0)
                for summary in summaries
            ),
        },
        "data_deduplication": dedup_audit,
        "component_audits": {
            "prompt": prompt_audit,
            "electron": electron_audit,
            "qcd_fake": qcd_fake_audit,
            "process_origins": process_origin_audits,
        },
        "kappa": {
            key: value
            for key, value in kappa.items()
            if key not in {"factors", "variances", "covariance"}
        },
        "closure": closure,
        "measurement": measurement,
        "target_validation": validation,
        "diagnostic_histograms": diagnostic_histograms(
            data_channels,
            prompt_channels,
            electron_channels,
            builder,
        ),
        "fake_prediction": prediction,
        "qcd_target_origin_histograms": qcd_target_origin_histograms(
            qcd_origins,
            builder,
        ),
        "mc_target_origin_histograms": {
            process: qcd_target_origin_histograms(origins, builder)
            for process, origins in process_origin_channels.items()
        },
    }
    write_payload(args.output, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "sidecars": len(accepted_paths),
                "predicted_fake": validation["predicted_fake"],
                "data_target": validation["data_target"],
                "closure_ratio": closure["global_target_over_prediction"],
                "closure_uncertainty": closure[
                    "assigned_relative_nonclosure"
                ],
                "kappa": [
                    {
                        "label": record["label"],
                        "factor": record["factor"],
                        "source": record["source"],
                    }
                    for record in kappa["records"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
