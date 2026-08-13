#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from autonomous_allhad.photon_fake_2024_worker import (
    GCR_REGIONS,
    ORIGINS,
    OUTPUT_SCHEMA,
    PROBE_KINDS,
    TRANSFER_PT_EDGES,
    _add_value,
    _channel_origin_record,
    load_histogram_builder,
)


MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v1"
PROMPT_NORMALIZATION_UNCERTAINTY = 0.30
ELECTRON_NORMALIZATION_UNCERTAINTY = 0.50
MAX_FAKE_FACTOR = 5.0
MIN_DIRECT_DENOMINATOR = 5.0


def read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        if path.suffix == ".gz":
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(
                    payload,
                    handle,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
        else:
            partial.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _merge_stratified(
    target: dict[str, Any],
    source: dict[str, Any],
    scale: float = 1.0,
) -> None:
    if target.get("transfer_labels") != source.get("transfer_labels"):
        raise RuntimeError("transfer stratum labels differ while merging")
    if len(target["strata"]) != len(source["strata"]):
        raise RuntimeError("transfer stratum counts differ while merging")
    for target_leaf, source_leaf in zip(target["strata"], source["strata"]):
        if target_leaf.get("bin_edges") != source_leaf.get("bin_edges"):
            raise RuntimeError("histogram edges differ while merging")
        for index in range(len(target_leaf["sumw"])):
            target_leaf["sumw"][index] += scale * float(source_leaf["sumw"][index])
            target_leaf["sumw2"][index] += (
                scale * scale * float(source_leaf["sumw2"][index])
            )
            target_leaf["entries"][index] += int(source_leaf["entries"][index])


def _merge_origin_record(
    target: dict[str, Any] | None,
    source: dict[str, Any],
    scale: float = 1.0,
) -> dict[str, Any]:
    if target is None:
        target = copy.deepcopy(source)
        if scale != 1.0:
            for record in _walk_stratified(target):
                for leaf in record["strata"]:
                    leaf["sumw"] = [scale * float(x) for x in leaf["sumw"]]
                    leaf["sumw2"] = [
                        scale * scale * float(x) for x in leaf["sumw2"]
                    ]
        return target
    _merge_stratified(target["transfer"], source["transfer"], scale)
    for region, record in (source.get("region_transfers") or {}).items():
        target_transfers = target.setdefault("region_transfers", {})
        if region not in target_transfers:
            target_transfers[region] = copy.deepcopy(record)
            if scale != 1.0:
                for leaf in target_transfers[region]["strata"]:
                    leaf["sumw"] = [scale * float(x) for x in leaf["sumw"]]
                    leaf["sumw2"] = [
                        scale * scale * float(x) for x in leaf["sumw2"]
                    ]
        else:
            _merge_stratified(target_transfers[region], record, scale)
    for region, variables in source.get("distributions", {}).items():
        for variable, record in variables.items():
            _merge_stratified(
                target["distributions"][region][variable],
                record,
                scale,
            )
    return target


def _walk_stratified(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "strata" in value and "transfer_labels" in value:
            found.append(value)
        else:
            for child in value.values():
                found.extend(_walk_stratified(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_stratified(child))
    return found


def _merge_dataset_record(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for field in ("physical_dataset", "process", "xsec_pb"):
        if target.get(field) != source.get(field):
            raise RuntimeError(
                f"dataset field mismatch for {source.get('physical_dataset')}: {field}"
            )
    target["dataset_splits"] = sorted(
        set(target.get("dataset_splits") or [])
        | set(source.get("dataset_splits") or [])
    )
    for field in ("files_processed", "events_read", "selected_events"):
        target[field] = int(target.get(field) or 0) + int(source.get(field) or 0)
    for probe, origins in (source.get("channels") or {}).items():
        for origin, record in origins.items():
            current = (target.setdefault("channels", {}).setdefault(probe, {})).get(
                origin
            )
            target["channels"][probe][origin] = _merge_origin_record(
                current,
                record,
            )


def _physical_data_source(dataset: str) -> str:
    return str(dataset).split("____", 1)[0]


def _data_source_rank(dataset: str) -> tuple[int, int, int, str]:
    text = _physical_data_source(dataset)
    special_v2 = int("NANOv15_v2" in text)
    version = 0
    if "-v" in text:
        suffix = text.rsplit("-v", 1)[-1]
        if suffix.isdigit():
            version = int(suffix)
    stream_rank = 1 if text.startswith("EGamma0-") else 0
    return special_v2, version, stream_rank, text


def deduplicate_data_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_event_and_source: dict[
        tuple[int, int, int], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for event in events:
        key = (
            int(event["run"]),
            int(event["luminosityBlock"]),
            int(event["event"]),
        )
        source = _physical_data_source(str(event["source_dataset"]))
        by_event_and_source[key][source].append(event)
    selected: list[dict[str, Any]] = []
    duplicate_event_keys = 0
    discarded_records = 0
    differing_probe_sets = 0
    source_choices: dict[str, int] = defaultdict(int)
    for sources in by_event_and_source.values():
        chosen_source = max(sources, key=_data_source_rank)
        chosen_by_probe: dict[str, dict[str, Any]] = {}
        for record in sources[chosen_source]:
            probe = str(record["probe"])
            if probe in chosen_by_probe:
                discarded_records += 1
                continue
            chosen_by_probe[probe] = record
        chosen = list(chosen_by_probe.values())
        selected.extend(chosen)
        source_choices[chosen_source] += 1
        if len(sources) > 1:
            duplicate_event_keys += 1
            discarded_records += sum(
                len(records)
                for source, records in sources.items()
                if source != chosen_source
            )
            probe_sets = {
                tuple(sorted(str(item["probe"]) for item in records))
                for records in sources.values()
            }
            if len(probe_sets) > 1:
                differing_probe_sets += 1
    selected.sort(
        key=lambda item: (
            int(item["run"]),
            int(item["luminosityBlock"]),
            int(item["event"]),
            str(item["probe"]),
        )
    )
    return selected, {
        "input_records": len(events),
        "unique_event_keys": len(by_event_and_source),
        "selected_records": len(selected),
        "duplicate_event_keys": duplicate_event_keys,
        "discarded_records": discarded_records,
        "duplicate_keys_with_differing_probe_sets": differing_probe_sets,
        "source_choice_event_counts": dict(sorted(source_choices.items())),
        "policy": (
            "deduplicate by run-lumi-event; prefer NANOv15_v2, then the highest "
            "processing version, then EGamma0; retain every probe category from "
            "the chosen source"
        ),
    }


def data_channels_from_events(
    events: list[dict[str, Any]],
    builder: Any,
) -> dict[str, Any]:
    dataset_record: dict[str, Any] = {"channels": {}}
    specs = builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS
    for event in events:
        probe = str(event["probe"])
        stratum = int(event["transfer_stratum"])
        n_top = int(event["nboosted_top"])
        target = _channel_origin_record(
            dataset_record,
            probe,
            "all",
            builder,
        )
        regions = list(event.get("regions") or [])
        if not regions:
            regions = ["GCR", "GCR_Nt0" if n_top == 0 else "GCR_Nt1"]
        if "GCR" in regions:
            _add_value(target["transfer"], stratum, 0.5, 1.0)
        for region in regions:
            if region in target.get("region_transfers", {}):
                _add_value(
                    target["region_transfers"][region],
                    stratum,
                    0.5,
                    1.0,
                )
            _add_value(
                target["distributions"][region]["recoil"],
                stratum,
                float(event["values"]["recoil"]),
                1.0,
            )
            if region == "GCR":
                for variable in specs:
                    _add_value(
                        target["distributions"][region][variable],
                        stratum,
                        float(event["values"][variable]),
                        1.0,
                    )
    return dataset_record["channels"]


def _empty_channels() -> dict[str, Any]:
    return {}


def aggregate_component(
    datasets: dict[str, Any],
    normalization: dict[str, Any],
    origin: str,
    include_processes: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _empty_channels()
    used: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    physical_norm = normalization.get("physical_datasets") or {}
    for physical, dataset in sorted(datasets.items()):
        process = str(dataset.get("process") or "unknown")
        if process not in include_processes:
            continue
        norm = physical_norm.get(physical) or {}
        factor = norm.get("normalization_factor")
        status = str(norm.get("normalization_status") or "missing")
        if factor is None or not np.isfinite(float(factor)):
            blocked.append(
                {
                    "physical_dataset": physical,
                    "process": process,
                    "normalization_status": status,
                }
            )
            continue
        factor = float(factor)
        selected = 0
        for probe, origins in (dataset.get("channels") or {}).items():
            source = origins.get(origin)
            if source is None:
                continue
            current = target.setdefault(probe, {}).get(origin)
            target[probe][origin] = _merge_origin_record(
                current,
                source,
                factor,
            )
            selected += sum(
                int(sum(leaf["entries"]))
                for record in _walk_stratified(source)
                for leaf in record["strata"][:1]
            )
        used.append(
            {
                "physical_dataset": physical,
                "process": process,
                "normalization_factor": factor,
                "normalization_status": status,
                "files_processed": int(dataset.get("files_processed") or 0),
                "selected_event_proxy": selected,
            }
        )
    return target, {
        "origin": origin,
        "included_processes": sorted(include_processes),
        "used_datasets": used,
        "blocked_datasets": blocked,
    }


def _origin_record(
    channels: dict[str, Any],
    probe: str,
    origin: str,
) -> dict[str, Any] | None:
    return ((channels.get(probe) or {}).get(origin))


def _stratum_transfer(
    channels: dict[str, Any],
    probe: str,
    origin: str,
) -> tuple[np.ndarray, np.ndarray]:
    record = _origin_record(channels, probe, origin)
    count = 2 * (len(TRANSFER_PT_EDGES) - 1)
    if record is None:
        return np.zeros(count), np.zeros(count)
    values = np.asarray(
        [leaf["sumw"][0] for leaf in record["transfer"]["strata"]],
        dtype=float,
    )
    variances = np.asarray(
        [leaf["sumw2"][0] for leaf in record["transfer"]["strata"]],
        dtype=float,
    )
    return values, variances


def fit_transfer_factors(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> dict[str, Any]:
    data_pass, data_pass_var = _stratum_transfer(
        data_channels, "measurement_pass", "all"
    )
    data_fail, data_fail_var = _stratum_transfer(
        data_channels, "measurement_fail", "all"
    )
    prompt_pass, prompt_pass_var = _stratum_transfer(
        prompt_channels, "measurement_pass", "prompt"
    )
    prompt_fail, prompt_fail_var = _stratum_transfer(
        prompt_channels, "measurement_fail", "prompt"
    )
    electron_pass, electron_pass_var = _stratum_transfer(
        electron_channels, "measurement_pass", "electron"
    )
    electron_fail, electron_fail_var = _stratum_transfer(
        electron_channels, "measurement_fail", "electron"
    )
    numerator = (
        data_pass
        - prompt_scale * prompt_pass
        - electron_scale * electron_pass
    )
    denominator = (
        data_fail
        - prompt_scale * prompt_fail
        - electron_scale * electron_fail
    )
    numerator_var = (
        data_pass_var
        + prompt_scale * prompt_scale * prompt_pass_var
        + electron_scale * electron_scale * electron_pass_var
    )
    denominator_var = (
        data_fail_var
        + prompt_scale * prompt_scale * prompt_fail_var
        + electron_scale * electron_scale * electron_fail_var
    )
    n_strata = len(numerator)
    factors = np.zeros(n_strata, dtype=float)
    variances = np.zeros(n_strata, dtype=float)
    sources = ["invalid"] * n_strata

    def ratio(indices: list[int]) -> tuple[float, float, bool]:
        num = float(np.sum(numerator[indices]))
        den = float(np.sum(denominator[indices]))
        num_var = float(np.sum(numerator_var[indices]))
        den_var = float(np.sum(denominator_var[indices]))
        valid = num > 0.0 and den >= MIN_DIRECT_DENOMINATOR
        if not valid:
            return 0.0, 0.0, False
        value = num / den
        variance = num_var / (den * den) + (
            num * num * den_var / (den**4)
        )
        if not np.isfinite(value) or not np.isfinite(variance):
            return 0.0, 0.0, False
        return min(MAX_FAKE_FACTOR, max(0.0, value)), max(0.0, variance), True

    all_indices = list(range(n_strata))
    global_value, global_variance, global_valid = ratio(all_indices)
    per_eta: dict[int, tuple[float, float, bool]] = {}
    pt_bins = len(TRANSFER_PT_EDGES) - 1
    for eta_index in range(2):
        indices = list(range(eta_index * pt_bins, (eta_index + 1) * pt_bins))
        per_eta[eta_index] = ratio(indices)

    for index in range(n_strata):
        direct_value, direct_variance, direct_valid = ratio([index])
        if direct_valid:
            factors[index] = direct_value
            variances[index] = direct_variance
            sources[index] = "direct"
            continue
        eta_value, eta_variance, eta_valid = per_eta[index // pt_bins]
        if eta_valid:
            factors[index] = eta_value
            variances[index] = eta_variance
            sources[index] = "eta_inclusive_fallback"
            continue
        if global_valid:
            factors[index] = global_value
            variances[index] = global_variance
            sources[index] = "global_fallback"
            continue
        factors[index] = 0.0
        variances[index] = 0.0
        sources[index] = "unmeasurable"

    labels = (
        (
            _origin_record(data_channels, "measurement_pass", "all") or {}
        ).get("transfer", {}).get("transfer_labels")
        or [f"stratum_{index}" for index in range(n_strata)]
    )
    records = []
    for index, label in enumerate(labels):
        records.append(
            {
                "index": index,
                "label": label,
                "data_pass": float(data_pass[index]),
                "data_fail": float(data_fail[index]),
                "prompt_pass": float(prompt_pass[index]),
                "prompt_fail": float(prompt_fail[index]),
                "electron_pass": float(electron_pass[index]),
                "electron_fail": float(electron_fail[index]),
                "fake_pass": float(numerator[index]),
                "fake_fail": float(denominator[index]),
                "factor": float(factors[index]),
                "factor_uncertainty": float(math.sqrt(variances[index])),
                "source": sources[index],
            }
        )
    return {
        "factors": factors,
        "variances": variances,
        "records": records,
        "prompt_scale": float(prompt_scale),
        "electron_scale": float(electron_scale),
    }


def _distribution_record(
    channels: dict[str, Any],
    probe: str,
    origin: str,
    region: str,
    variable: str,
) -> dict[str, Any] | None:
    origin_record = _origin_record(channels, probe, origin)
    if origin_record is None:
        return None
    return (
        ((origin_record.get("distributions") or {}).get(region) or {}).get(variable)
    )


def _empty_leaf(edges: list[float]) -> dict[str, Any]:
    bins = len(edges) - 1
    return {
        "bin_edges": [float(x) for x in edges],
        "sumw": [0.0] * bins,
        "sumw2": [0.0] * bins,
        "entries": [0] * bins,
    }


def predict_distribution(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    region: str,
    variable: str,
    edges: list[float],
    factors: np.ndarray,
    factor_variances: np.ndarray,
    prompt_scale: float = 1.0,
    electron_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = [
        _distribution_record(
            data_channels,
            "application",
            "all",
            region,
            variable,
        ),
        _distribution_record(
            prompt_channels,
            "application",
            "prompt",
            region,
            variable,
        ),
        _distribution_record(
            electron_channels,
            "application",
            "electron",
            region,
            variable,
        ),
    ]
    bins = len(edges) - 1
    output = _empty_leaf(edges)
    negative_before_clip: list[dict[str, Any]] = []
    for stratum in range(len(factors)):
        arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for source in sources:
            if source is None:
                arrays.append(
                    (
                        np.zeros(bins),
                        np.zeros(bins),
                        np.zeros(bins, dtype=int),
                    )
                )
                continue
            leaf = source["strata"][stratum]
            arrays.append(
                (
                    np.asarray(leaf["sumw"], dtype=float),
                    np.asarray(leaf["sumw2"], dtype=float),
                    np.asarray(leaf["entries"], dtype=int),
                )
            )
        data, data_var, data_entries = arrays[0]
        prompt, prompt_var, _ = arrays[1]
        electron, electron_var, _ = arrays[2]
        residual = data - prompt_scale * prompt - electron_scale * electron
        residual_var = (
            data_var
            + prompt_scale * prompt_scale * prompt_var
            + electron_scale * electron_scale * electron_var
        )
        factor = float(factors[stratum])
        factor_var = float(factor_variances[stratum])
        output["sumw"] = (
            np.asarray(output["sumw"], dtype=float) + factor * residual
        ).tolist()
        output["sumw2"] = (
            np.asarray(output["sumw2"], dtype=float)
            + factor * factor * residual_var
            + residual * residual * factor_var
        ).tolist()
        output["entries"] = (
            np.asarray(output["entries"], dtype=int) + data_entries
        ).astype(int).tolist()
    values = np.asarray(output["sumw"], dtype=float)
    for index in np.flatnonzero(values < 0.0):
        negative_before_clip.append(
            {
                "bin": int(index),
                "value": float(values[index]),
            }
        )
    values = np.maximum(values, 0.0)
    output["sumw"] = values.tolist()
    return output, {
        "negative_bins_clipped": negative_before_clip,
        "integral": float(np.sum(values)),
    }


def _leaf_integral(leaf: dict[str, Any] | None) -> tuple[float, float]:
    if leaf is None:
        return 0.0, 0.0
    return (
        float(np.sum(np.asarray(leaf["sumw"], dtype=float))),
        float(np.sum(np.asarray(leaf["sumw2"], dtype=float))),
    )


def _sum_stratified(record: dict[str, Any] | None) -> tuple[float, float]:
    if record is None:
        return 0.0, 0.0
    value = 0.0
    variance = 0.0
    for leaf in record["strata"]:
        value += float(np.sum(np.asarray(leaf["sumw"], dtype=float)))
        variance += float(np.sum(np.asarray(leaf["sumw2"], dtype=float)))
    return value, variance


def _collapse_stratified(
    record: dict[str, Any] | None,
    edges: list[float],
) -> dict[str, Any]:
    output = _empty_leaf(edges)
    if record is None:
        return output
    for leaf in record["strata"]:
        if [float(value) for value in leaf["bin_edges"]] != [
            float(value) for value in edges
        ]:
            raise RuntimeError("diagnostic histogram edges differ across strata")
        output["sumw"] = (
            np.asarray(output["sumw"], dtype=float)
            + np.asarray(leaf["sumw"], dtype=float)
        ).tolist()
        output["sumw2"] = (
            np.asarray(output["sumw2"], dtype=float)
            + np.asarray(leaf["sumw2"], dtype=float)
        ).tolist()
        output["entries"] = (
            np.asarray(output["entries"], dtype=int)
            + np.asarray(leaf["entries"], dtype=int)
        ).astype(int).tolist()
    return output


def diagnostic_histograms(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    builder: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    probes = ("measurement_pass", "measurement_fail", "application", "target")
    for region in GCR_REGIONS:
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(x) for x in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            variable_output = output.setdefault(region, {}).setdefault(variable, {})
            for probe in probes:
                data = _collapse_stratified(
                    _distribution_record(
                        data_channels,
                        probe,
                        "all",
                        region,
                        variable,
                    ),
                    edges,
                )
                prompt = _collapse_stratified(
                    _distribution_record(
                        prompt_channels,
                        probe,
                        "prompt",
                        region,
                        variable,
                    ),
                    edges,
                )
                electron = _collapse_stratified(
                    _distribution_record(
                        electron_channels,
                        probe,
                        "electron",
                        region,
                        variable,
                    ),
                    edges,
                )
                residual = _empty_leaf(edges)
                residual["sumw"] = (
                    np.asarray(data["sumw"], dtype=float)
                    - np.asarray(prompt["sumw"], dtype=float)
                    - np.asarray(electron["sumw"], dtype=float)
                ).tolist()
                residual["sumw2"] = (
                    np.asarray(data["sumw2"], dtype=float)
                    + np.asarray(prompt["sumw2"], dtype=float)
                    + np.asarray(electron["sumw2"], dtype=float)
                ).tolist()
                residual["entries"] = list(data["entries"])
                variable_output[probe] = {
                    "data": data,
                    "prompt": prompt,
                    "electron": electron,
                    "data_minus_prompt_electron": residual,
                }
    return output


def qcd_target_origin_histograms(
    qcd_origin_channels: dict[str, dict[str, Any]],
    builder: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for region in GCR_REGIONS:
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(x) for x in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            target = output.setdefault(region, {}).setdefault(variable, {})
            for origin, channels in qcd_origin_channels.items():
                target[origin] = _collapse_stratified(
                    _distribution_record(
                        channels,
                        "target",
                        origin,
                        region,
                        variable,
                    ),
                    edges,
                )
    return output


def closure_study(
    qcd_channels: dict[str, Any],
    builder: Any,
) -> dict[str, Any]:
    blank: dict[str, Any] = {}
    qcd_as_data = {
        probe: {"all": origins["fake"]}
        for probe, origins in qcd_channels.items()
        if "fake" in origins
    }
    fit = fit_transfer_factors(qcd_as_data, blank, blank)
    factors = fit["factors"]
    variances = fit["variances"]
    results: dict[str, Any] = {}
    target_total = 0.0
    predicted_total = 0.0
    predicted_variance = 0.0
    for region in GCR_REGIONS:
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(x) for x in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            prediction, audit = predict_distribution(
                qcd_as_data,
                blank,
                blank,
                region,
                variable,
                edges,
                factors,
                variances,
            )
            target_record = _distribution_record(
                qcd_as_data,
                "target",
                "all",
                region,
                variable,
            )
            target, target_var = _sum_stratified(target_record)
            predicted, pred_var = _leaf_integral(prediction)
            ratio = target / predicted if predicted > 0.0 else None
            results[f"{region}/{variable}"] = {
                "target": target,
                "target_variance": target_var,
                "predicted": predicted,
                "predicted_variance": pred_var,
                "target_over_prediction": ratio,
                "target_histogram": _collapse_stratified(target_record, edges),
                "predicted_histogram": prediction,
                "negative_bins_clipped": audit["negative_bins_clipped"],
            }
            if region == "GCR" and variable == "recoil":
                target_total = target
                predicted_total = predicted
                predicted_variance = pred_var
    if predicted_total > 0.0 and target_total > 0.0:
        ratio = target_total / predicted_total
        ratio_variance = (
            predicted_variance * target_total * target_total / predicted_total**4
        )
        ratio_uncertainty = math.sqrt(max(0.0, ratio_variance))
        nonclosure = min(
            1.0,
            max(0.30, abs(ratio - 1.0), ratio_uncertainty),
        )
        status = "measured"
    else:
        ratio = None
        ratio_uncertainty = None
        nonclosure = 1.0
        status = "insufficient_qcd_fake_statistics"
    return {
        "status": status,
        "transfer_factors": fit["records"],
        "global_target": target_total,
        "global_prediction": predicted_total,
        "global_target_over_prediction": ratio,
        "global_ratio_uncertainty": ratio_uncertainty,
        "assigned_relative_nonclosure": nonclosure,
        "distributions": results,
        "central_value_policy": (
            "QCD closure does not rescale the data-driven central value; the "
            "larger of 30%, observed nonclosure, and closure statistical "
            "uncertainty is assigned, capped at 100%"
        ),
    }


def _scaled_variation(
    nominal: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    output = copy.deepcopy(nominal)
    output["sumw"] = [
        max(0.0, scale * float(value)) for value in nominal["sumw"]
    ]
    return output


def build_prediction(
    data_channels: dict[str, Any],
    prompt_channels: dict[str, Any],
    electron_channels: dict[str, Any],
    closure: dict[str, Any],
    builder: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    central_fit = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
    )
    prompt_up_fit = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        prompt_scale=1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
    )
    prompt_down_fit = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        prompt_scale=1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
    )
    electron_up_fit = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        electron_scale=1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
    )
    electron_down_fit = fit_transfer_factors(
        data_channels,
        prompt_channels,
        electron_channels,
        electron_scale=1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
    )
    central = central_fit["factors"]
    variance = central_fit["variances"]
    uncertainty = np.sqrt(np.maximum(variance, 0.0))
    factor_up = np.minimum(MAX_FAKE_FACTOR, central + uncertainty)
    factor_down = np.maximum(0.0, central - uncertainty)
    closure_uncertainty = float(closure["assigned_relative_nonclosure"])

    output: dict[str, Any] = {
        "histograms": {},
        "highdm_variable_histograms": {},
    }
    audits: dict[str, Any] = {}
    for region in GCR_REGIONS:
        variables = {"recoil": [float(x) for x in builder.RECOIL_PT_BINS]}
        if region == "GCR":
            variables.update(
                {
                    variable: [float(x) for x in spec["bins"]]
                    for variable, spec in (
                        builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
                    )
                }
            )
        for variable, edges in variables.items():
            nominal, nominal_audit = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                central,
                variance,
            )
            tf_up, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                factor_up,
                np.zeros_like(variance),
            )
            tf_down, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                factor_down,
                np.zeros_like(variance),
            )
            prompt_up, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                prompt_up_fit["factors"],
                np.zeros_like(variance),
                prompt_scale=1.0 + PROMPT_NORMALIZATION_UNCERTAINTY,
            )
            prompt_down, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                prompt_down_fit["factors"],
                np.zeros_like(variance),
                prompt_scale=1.0 - PROMPT_NORMALIZATION_UNCERTAINTY,
            )
            electron_up, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                electron_up_fit["factors"],
                np.zeros_like(variance),
                electron_scale=1.0 + ELECTRON_NORMALIZATION_UNCERTAINTY,
            )
            electron_down, _ = predict_distribution(
                data_channels,
                prompt_channels,
                electron_channels,
                region,
                variable,
                edges,
                electron_down_fit["factors"],
                np.zeros_like(variance),
                electron_scale=1.0 - ELECTRON_NORMALIZATION_UNCERTAINTY,
            )
            variations = {
                "nominal": nominal,
                "photonFakeTFUp": tf_up,
                "photonFakeTFDown": tf_down,
                "photonFakePromptUp": prompt_up,
                "photonFakePromptDown": prompt_down,
                "photonFakeElectronUp": electron_up,
                "photonFakeElectronDown": electron_down,
                "photonFakeClosureUp": _scaled_variation(
                    nominal, 1.0 + closure_uncertainty
                ),
                "photonFakeClosureDown": _scaled_variation(
                    nominal, max(0.0, 1.0 - closure_uncertainty)
                ),
            }
            if variable == "recoil":
                output["histograms"][region] = variations
            else:
                output["highdm_variable_histograms"].setdefault(region, {})[
                    variable
                ] = variations
            audits[f"{region}/{variable}"] = nominal_audit
    return output, {
        "central_transfer_factors": central_fit["records"],
        "prompt_up_transfer_factors": prompt_up_fit["records"],
        "prompt_down_transfer_factors": prompt_down_fit["records"],
        "electron_up_transfer_factors": electron_up_fit["records"],
        "electron_down_transfer_factors": electron_down_fit["records"],
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
    data_target = _distribution_record(
        data_channels, "target", "all", "GCR", "recoil"
    )
    prompt_target = _distribution_record(
        prompt_channels, "target", "prompt", "GCR", "recoil"
    )
    electron_target = _distribution_record(
        electron_channels, "target", "electron", "GCR", "recoil"
    )
    data_value, data_var = _sum_stratified(data_target)
    prompt_value, prompt_var = _sum_stratified(prompt_target)
    electron_value, electron_var = _sum_stratified(electron_target)
    fake_value, fake_var = _leaf_integral(
        prediction["histograms"]["GCR"]["nominal"]
    )
    residual = data_value - prompt_value - electron_value
    total = prompt_value + electron_value + fake_value
    return {
        "data_target": data_value,
        "data_target_variance": data_var,
        "prompt_target": prompt_value,
        "prompt_target_variance": prompt_var,
        "electron_target": electron_value,
        "electron_target_variance": electron_var,
        "observed_fake_residual_not_used_in_fit": residual,
        "predicted_fake": fake_value,
        "predicted_fake_variance": fake_var,
        "prompt_plus_electron_plus_fake": total,
        "prediction_over_data": total / data_value if data_value > 0.0 else None,
        "blinding_policy": (
            "the target-region residual is reported only as validation and is "
            "not used to normalize or fit the fake prediction"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge 2024 photon-fake sidecars and perform the fake-factor measurement."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--contamination-process-policy",
        choices=("all-backgrounds", "exclude-qcd"),
        default="all-backgrounds",
        help=(
            "Subtract prompt/electron origins from every background process by "
            "default. exclude-qcd reproduces the legacy diagnostic policy."
        ),
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json.gz")))
        else:
            paths.append(path)
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("no photon fake sidecars found")

    datasets: dict[str, Any] = {}
    data_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    incomplete: list[str] = []
    seen_digests: set[str] = set()
    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != OUTPUT_SCHEMA:
            continue
        digest = str((payload.get("summary") or {}).get("source_record_digest") or "")
        if digest and digest in seen_digests:
            raise RuntimeError(f"duplicate source record digest: {digest}")
        if digest:
            seen_digests.add(digest)
        status = str(payload.get("status") or "unknown")
        if status != "complete":
            incomplete.append(str(path))
        summaries.append(payload.get("summary") or {})
        data_events.extend(payload.get("data_events") or [])
        for physical, incoming in (payload.get("datasets") or {}).items():
            if physical not in datasets:
                datasets[physical] = incoming
            else:
                _merge_dataset_record(datasets[physical], incoming)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete)} sidecars are incomplete; use --allow-incomplete only "
            "for a diagnostic partial measurement"
        )

    builder = load_histogram_builder()
    normalization = read_payload(args.normalization)
    deduplicated, dedup_audit = deduplicate_data_events(data_events)
    data_channels = data_channels_from_events(deduplicated, builder)
    non_qcd = {"GJ", "DY", "TT", "WtoLNu", "ST", "VV", "Zto2Nu"}
    contamination_processes = set(non_qcd)
    if args.contamination_process_policy == "all-backgrounds":
        contamination_processes.add("QCD")
    prompt_channels, prompt_audit = aggregate_component(
        datasets,
        normalization,
        "prompt",
        contamination_processes,
    )
    electron_channels, electron_audit = aggregate_component(
        datasets,
        normalization,
        "electron",
        contamination_processes,
    )
    qcd_fake_channels, qcd_audit = aggregate_component(
        datasets,
        normalization,
        "fake",
        {"QCD"},
    )
    qcd_all_channels, qcd_all_audit = aggregate_component(
        datasets,
        normalization,
        "all",
        {"QCD"},
    )
    qcd_prompt_channels, qcd_prompt_audit = aggregate_component(
        datasets,
        normalization,
        "prompt",
        {"QCD"},
    )
    qcd_electron_channels, qcd_electron_audit = aggregate_component(
        datasets,
        normalization,
        "electron",
        {"QCD"},
    )
    qcd_origins = {
        "all": qcd_all_channels,
        "prompt": qcd_prompt_channels,
        "electron": qcd_electron_channels,
        "fake": qcd_fake_channels,
    }
    closure = closure_study(qcd_fake_channels, builder)
    prediction, measurement = build_prediction(
        data_channels,
        prompt_channels,
        electron_channels,
        closure,
        builder,
    )
    diagnostics = diagnostic_histograms(
        data_channels,
        prompt_channels,
        electron_channels,
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
            str(dataset.get("process") or "unknown")
            for dataset in datasets.values()
        }
    )
    if data_events:
        processes = sorted(set(processes) | {"EGamma"})
    files_attempted = sum(int(s.get("files_attempted") or 0) for s in summaries)
    files_processed = sum(int(s.get("files_processed") or 0) for s in summaries)
    coverage: dict[str, Any]
    if args.campaign_manifest is None:
        coverage = {
            "status": "partial_diagnostic",
            "reason": "campaign manifest not supplied",
            "observed_sidecars": len(summaries),
            "observed_files_attempted": files_attempted,
            "observed_files_processed": files_processed,
        }
    else:
        manifest = read_payload(args.campaign_manifest)
        expected_sidecars = int(manifest.get("jobs") or 0)
        expected_records = sum(
            int(value) for value in (manifest.get("record_counts") or {}).values()
        )
        expected_processes = sorted(manifest.get("requested_processes") or [])
        coverage_errors: list[str] = []
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
        missing_processes = sorted(set(expected_processes) - set(processes))
        if missing_processes:
            coverage_errors.append(f"missing processes {missing_processes}")
        if incomplete:
            coverage_errors.append(f"incomplete sidecars {len(incomplete)}")
        coverage = {
            "status": "complete" if not coverage_errors else "incomplete",
            "campaign_manifest": str(args.campaign_manifest),
            "source_campaign": manifest.get("source_campaign"),
            "manifest_selection_source": manifest.get("selection_source"),
            "expected_sidecars": expected_sidecars,
            "observed_sidecars": len(summaries),
            "expected_records": expected_records,
            "observed_files_attempted": files_attempted,
            "observed_files_processed": files_processed,
            "expected_processes": expected_processes,
            "observed_processes": processes,
            "errors": coverage_errors,
        }
    final_status = (
        "complete"
        if coverage["status"] == "complete" and not incomplete
        else "partial_diagnostic"
    )
    if final_status != "complete" and not args.allow_incomplete:
        raise RuntimeError(
            "photon fake measurement coverage is incomplete: "
            + json.dumps(coverage, sort_keys=True)
        )
    output = {
        "schema_version": MEASUREMENT_SCHEMA,
        "status": final_status,
        "year": 2024,
        "scope": "high-dM photon control region",
        "selection_source": "real_subset_worker.py",
        "nominal_intermediate_policy": (
            "read-only; fake sidebands are extracted directly from NanoAOD into "
            "a separate campaign and no nominal intermediate is modified"
        ),
        "method": (
            "sieie anti-ID measurement region fake factor applied to the "
            "charged-isolation-fail application region"
        ),
        "central_value": (
            "EGamma data application region after prompt-photon and electron "
            "contamination subtraction, weighted by data-measured fake factors"
        ),
        "mc_roles": {
            "prompt_and_electron_contamination": sorted(contamination_processes),
            "qcd": (
                "prompt/electron origins participate in contamination "
                "subtraction and are retained in the target; the truth-fake "
                "origin is used for closure and nonclosure uncertainty"
            ),
        },
        "contamination_process_policy": args.contamination_process_policy,
        "input_sidecars": [str(path) for path in paths],
        "coverage": coverage,
        "input_summary": {
            "sidecar_count": len(summaries),
            "incomplete_sidecars": incomplete,
            "files_attempted": files_attempted,
            "files_processed": files_processed,
            "events_read": sum(int(s.get("events_read") or 0) for s in summaries),
            "selected_events": sum(int(s.get("selected_events") or 0) for s in summaries),
            "processes": processes,
            "target_cutbased_mismatch_objects": sum(
                int(s.get("target_cutbased_mismatch_objects") or 0)
                for s in summaries
            ),
        },
        "data_deduplication": dedup_audit,
        "normalization_source": str(args.normalization),
        "component_audits": {
            "prompt": prompt_audit,
            "electron": electron_audit,
            "qcd_fake": qcd_audit,
            "qcd_all": qcd_all_audit,
            "qcd_prompt": qcd_prompt_audit,
            "qcd_electron": qcd_electron_audit,
        },
        "measurement": measurement,
        "diagnostic_histograms": diagnostics,
        "qcd_target_origin_histograms": qcd_target_origin_histograms(
            qcd_origins,
            builder,
        ),
        "closure": closure,
        "target_validation": validation,
        "fake_prediction": prediction,
    }
    write_payload(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(args.output),
                "sidecars": len(paths),
                "processes": processes,
                "data_events_before_dedup": len(data_events),
                "data_events_after_dedup": len(deduplicated),
                "predicted_fake": validation["predicted_fake"],
                "data_target": validation["data_target"],
                "prompt_target": validation["prompt_target"],
                "closure_status": closure["status"],
                "closure_uncertainty": closure["assigned_relative_nonclosure"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
