#!/usr/bin/env python3
"""Build the adopted 2024 LLCR/QCDCR/GCR transfer-factor likelihood.

The dilepton regions are external measurements of R_Z, not Poisson channels.
The photon control region remains in the simultaneous fit.  Its residual
normalization is shared with the matched invisible-Z signal-region template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import re
from pathlib import Path
from typing import Any

import numpy as np

from background_process_groups import (
    BACKGROUND_PROCESS_ORDER,
    background_grouping_contract,
)
from build_boosted_an17_combine_inputs import (
    LUMI_LNN,
    signal_process_name,
    stable_path,
    write_hist,
    write_json,
)
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner
from build_nb_recoil_tf_combine_inputs_2024 import (
    MIN_BIN,
    build_root,
    mass_points,
    one_bin_background,
    sum_one_bin_backgrounds,
)


HIGH_CONTROL_REGIONS = ("LLCR", "QCDCR", "GCR")
LOW_CONTROL_REGIONS = HIGH_CONTROL_REGIONS
HIGH_PHYSICAL_GROUPS = {
    "Nb1": ("Nb1",),
    "Nb2plus": ("Nb2", "Nb3plus"),
}
RARE_PROCESSES = ("VV_VVV", "DY", "PhotonJet")
CONTROLLED_PROCESSES = ("Top", "WtoLNu", "QCD", "Zto2Nu")
HIGH_MERGE_PAIRS = ((22, 23), (34, 35), (40, 41), (52, 53), (58, 59))
HIGH_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"
LOW_SCHEME = "cat7_SR_lowDeltaM"
RZ_CATEGORIES = (
    "highdm_Nb1",
    "highdm_Nb2plus",
    "lowdm_Nb1",
    "lowdm_Nb2plus",
)
NPS_LUMI_NAME = "lumi_13p6TeV_2024"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value_end(buffer: mmap.mmap, start: int) -> int:
    opening = buffer[start]
    if opening not in (ord("{"), ord("[")):
        raise ValueError(f"expected JSON object/array at byte {start}")
    closing = ord("}") if opening == ord("{") else ord("]")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(buffer)):
        value = buffer[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value == opening:
            depth += 1
        elif value == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unterminated JSON value at byte {start}")


def extract_search_sample(path: Path, scheme: str, sample: str) -> dict[str, Any]:
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            section = payload.find(b'"search_bin_histograms":')
            if section < 0:
                raise ValueError(f"search_bin_histograms absent from {path}")
            scheme_marker = json.dumps(scheme).encode() + b":"
            scheme_start = payload.find(scheme_marker, section)
            if scheme_start < 0:
                raise ValueError(f"scheme {scheme} absent from {path}")
            sample_marker = json.dumps(sample).encode() + b":"
            sample_start = payload.find(sample_marker, scheme_start + len(scheme_marker))
            if sample_start < 0:
                raise ValueError(f"sample {sample} absent from {scheme}")
            value_start = sample_start + len(sample_marker)
            while payload[value_start] in b" \t\r\n":
                value_start += 1
            value_end = json_value_end(payload, value_start)
            return json.loads(payload[value_start:value_end])


def extract_small_histogram_input(
    path: Path, masses: list[str]
) -> dict[str, Any]:
    samples = ["Zto2Nu"] + [f"T2tt_{mass}" for mass in masses]
    return {
        "search_bin_histograms": {
            scheme: {
                sample: extract_search_sample(path, scheme, sample)
                for sample in samples
            }
            for scheme in (HIGH_SCHEME, LOW_SCHEME)
        }
    }


def record_arrays(record: dict[str, Any], nbin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(record.get("sumw") or [], dtype=float)
    sumw2 = np.asarray(record.get("sumw2") or [], dtype=float)
    entries = np.asarray(record.get("entries") or np.zeros(nbin), dtype=float)
    if len(values) != nbin or len(sumw2) != nbin or len(entries) != nbin:
        raise ValueError(f"wrong Z histogram length {len(values)}/{len(sumw2)}/{len(entries)} != {nbin}")
    return values, sumw2, entries


def normalized_fractions(values: list[float], fallback: list[float]) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    if float(np.sum(positive)) <= 0.0:
        positive = np.maximum(np.asarray(fallback, dtype=float), 0.0)
    if float(np.sum(positive)) <= 0.0:
        positive = np.ones(len(values), dtype=float)
    return positive / np.sum(positive)


def materialize_zinv_components(
    small_hists: dict[str, Any], split_reference: dict[str, Any]
) -> dict[str, Any]:
    output = {
        "schema_version": "zinv_sr_inputs_2024_v1",
        "status": "complete",
        "highdm": {group: {"Zto2Nu": {}} for group in HIGH_PHYSICAL_GROUPS["Nb1"] + HIGH_PHYSICAL_GROUPS["Nb2plus"]},
        "lowdm": {group: {"Zto2Nu": {}} for group in ("Nb1", "Nb2plus")},
        "validation": {},
    }
    high_current = small_hists["search_bin_histograms"][HIGH_SCHEME]["Zto2Nu"]
    high_reference = split_reference["highdm"]["sr_components"]
    physical_groups = ("Nb1", "Nb2", "Nb3plus")
    mixed_bins = set(range(12))
    fixed_group_by_category = {
        2: "Nb1",
        3: "Nb1",
        4: "Nb2",
        5: "Nb2",
        6: "Nb3plus",
        7: "Nb3plus",
        8: "Nb3plus",
        9: "Nb2",
    }
    variations = sorted(high_current)
    for variation in variations:
        current_values, current_sumw2, current_entries = record_arrays(
            high_current[variation], 60
        )
        nominal_reference = {
            group: record_arrays(
                high_reference[group]["Zto2Nu"]["nominal"], 60
            )
            for group in physical_groups
        }
        varied_reference = {
            group: record_arrays(
                high_reference[group]["Zto2Nu"].get(variation)
                or high_reference[group]["Zto2Nu"]["nominal"],
                60,
            )
            for group in physical_groups
        }
        leaves = {
            group: {
                "sumw": np.zeros(60),
                "sumw2": np.zeros(60),
                "entries": np.zeros(60),
            }
            for group in physical_groups
        }
        for source_bin in range(60):
            if source_bin not in mixed_bins:
                group = fixed_group_by_category[source_bin // 6]
                leaves[group]["sumw"][source_bin] = current_values[source_bin]
                leaves[group]["sumw2"][source_bin] = current_sumw2[source_bin]
                leaves[group]["entries"][source_bin] = current_entries[source_bin]
                continue
            yield_fractions = normalized_fractions(
                [varied_reference[group][0][source_bin] for group in physical_groups],
                [nominal_reference[group][0][source_bin] for group in physical_groups],
            )
            variance_fractions = normalized_fractions(
                [varied_reference[group][1][source_bin] for group in physical_groups],
                [nominal_reference[group][1][source_bin] for group in physical_groups],
            )
            entry_fractions = normalized_fractions(
                [varied_reference[group][2][source_bin] for group in physical_groups],
                [nominal_reference[group][2][source_bin] for group in physical_groups],
            )
            for index, group in enumerate(physical_groups):
                leaves[group]["sumw"][source_bin] = current_values[source_bin] * yield_fractions[index]
                leaves[group]["sumw2"][source_bin] = current_sumw2[source_bin] * variance_fractions[index]
                leaves[group]["entries"][source_bin] = current_entries[source_bin] * entry_fractions[index]
        for group in physical_groups:
            output["highdm"][group]["Zto2Nu"][variation] = {
                key: values.tolist() for key, values in leaves[group].items()
            }
        reconstructed = sum(
            (leaves[group]["sumw"] for group in physical_groups), np.zeros(60)
        )
        reconstructed_sumw2 = sum(
            (leaves[group]["sumw2"] for group in physical_groups), np.zeros(60)
        )
        if not np.allclose(reconstructed, current_values, rtol=1.0e-12, atol=1.0e-10):
            raise ValueError(f"High-dM Z split does not reproduce {variation} yield")
        if not np.allclose(reconstructed_sumw2, current_sumw2, rtol=1.0e-12, atol=1.0e-10):
            raise ValueError(f"High-dM Z split does not reproduce {variation} sumw2")

    low_current = small_hists["search_bin_histograms"][LOW_SCHEME]["Zto2Nu"]
    for variation, record in sorted(low_current.items()):
        values, sumw2, entries = record_arrays(record, 34)
        for group, selected in (
            ("Nb1", np.arange(34) < 16),
            ("Nb2plus", np.arange(34) >= 16),
        ):
            output["lowdm"][group]["Zto2Nu"][variation] = {
                "sumw": np.where(selected, values, 0.0).tolist(),
                "sumw2": np.where(selected, sumw2, 0.0).tolist(),
                "entries": np.where(selected, entries, 0.0).tolist(),
            }
    output["validation"] = {
        "highdm_variations": variations,
        "highdm_variation_count": len(variations),
        "highdm_total_reproduced": True,
        "highdm_mixed_nb_source_bins_zero_based": sorted(mixed_bins),
        "highdm_mixed_nb_split_policy": (
            "Current absolute Z yields/sumw2 are split with the validated "
            "pre-existing exact Nb fractions only for the two Nb>=1 categories; "
            "all 60-bin totals and every variation are reproduced exactly."
        ),
        "lowdm_variation_count": len(low_current),
        "lowdm_total_reproduced": True,
    }
    return output


def scaled_record(
    record: dict[str, Any] | None, scale: float
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "nominal": record["nominal"] * scale,
        "sumw2": record["sumw2"] * scale * scale,
        "variations": {
            nuisance: {
                "up": pair["up"] * scale,
                "down": pair["down"] * scale,
            }
            for nuisance, pair in record["variations"].items()
        },
    }


def require_positive(record: dict[str, Any], label: str, key: str = "value") -> float:
    if record.get("status") != "complete":
        raise ValueError(f"{label} is not complete: {record}")
    value = float(record[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} is nonpositive/nonfinite: {value}")
    return value


def central_or_unity(record: dict[str, Any]) -> float:
    if record.get("status") == "complete":
        value = float(record.get("value", float("nan")))
        if np.isfinite(value) and value > 0.0:
            return value
    return 1.0


def logical_records(
    sources: dict[str, dict[str, Any]],
    logical_group: str,
    process: str,
    source_bin: int,
    nbin: int,
    scales: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    records = []
    for physical_group in HIGH_PHYSICAL_GROUPS[logical_group]:
        record = one_bin_background(
            sources[physical_group], process, source_bin, nbin
        )
        if scales is not None:
            record = scaled_record(record, scales[physical_group])
        records.append(record)
    return sum_one_bin_backgrounds(records)


def low_family(label: str) -> str:
    return re.sub(r"_recoil_[0-9]+$", "", label)


def low_sgamma(sgamma: dict[str, Any], label: str) -> tuple[float, float, str, int]:
    family = low_family(label)
    payload = sgamma["lowdm_families"][family]
    source_bin = int(re.search(r"_recoil_([0-9]+)$", label).group(1)) - 1
    if source_bin < 0 or source_bin >= len(payload["bins"]):
        raise ValueError(f"invalid Low-dM recoil bin for {label}")
    q = require_positive(payload["Q"], f"Qgamma/{family}")
    shape = central_or_unity(payload["bins"][source_bin]["Sgamma"])
    return q, shape, family, source_bin


def high_sgamma(
    sgamma: dict[str, Any], physical_group: str, recoil_bin: int
) -> tuple[float, float]:
    payload = sgamma["highdm"][physical_group]
    q = require_positive(payload["Q"], f"Qgamma/highdm/{physical_group}")
    shape = central_or_unity(payload["bins"][recoil_bin]["Sgamma"])
    return q, shape


def build_rz_covariance(
    high_summary: dict[str, Any], low_summary: dict[str, Any]
) -> dict[str, Any]:
    records = {
        "highdm_Nb1": high_summary["rz_high"]["combined"]["Nb1"],
        "highdm_Nb2plus": high_summary["rz_high"]["combined"]["Nb2plus"],
        "lowdm_Nb1": low_summary["rz_low"]["combined"]["Nb1"],
        "lowdm_Nb2plus": low_summary["rz_low"]["combined"]["Nb2plus"],
    }
    central = np.asarray(
        [require_positive(records[key], f"RZ/{key}", "RZ") for key in RZ_CATEGORIES]
    )
    errors = np.asarray([float(records[key]["RZ_stat"]) for key in RZ_CATEGORIES])
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError(f"invalid RZ statistical errors: {errors}")
    covariance_r = np.diag(errors * errors)
    covariance_eta = covariance_r / np.outer(central, central)
    cholesky = np.linalg.cholesky(covariance_eta)
    nuisances = []
    for index, category in enumerate(RZ_CATEGORIES):
        nuisances.append(
            {
                "name": f"CMS_SUS26090_RZstat_{category}_2024",
                "category": category,
                "log_coefficient": float(cholesky[index, index]),
            }
        )
    return {
        "schema_version": "rz_covariance_2024_v1",
        "status": "temporary_statistical_only",
        "warning": (
            "Diagonal statistical-only covariance. Replace with the final "
            "documented cross-category covariance before the final result."
        ),
        "categories": list(RZ_CATEGORIES),
        "central": central.tolist(),
        "statistical_errors": errors.tolist(),
        "covariance_r": covariance_r.tolist(),
        "covariance_log_r": covariance_eta.tolist(),
        "cholesky_log_r": cholesky.tolist(),
        "nuisances": nuisances,
    }


def rz_value(covariance: dict[str, Any], category: str) -> float:
    return float(covariance["central"][covariance["categories"].index(category)])


def rz_nuisances(covariance: dict[str, Any], category: str) -> list[dict[str, Any]]:
    index = covariance["categories"].index(category)
    output = []
    for column, nuisance in enumerate(covariance["nuisances"]):
        coefficient = float(covariance["cholesky_log_r"][index][column])
        if coefficient == 0.0:
            continue
        output.append(
            {
                "name": nuisance["name"],
                "down": math.exp(-coefficient),
                "up": math.exp(coefficient),
            }
        )
    return output


def closure_record(
    double_ratio: dict[str, Any], regime: str, low: float, high: float
) -> tuple[str, float, list[int]]:
    selected = []
    for index, record in enumerate(double_ratio[regime]["bins"]):
        if record.get("status") != "complete":
            continue
        if float(record["high"]) <= low or float(record["low"]) >= high:
            continue
        value = float(record["double_ratio"])
        if np.isfinite(value):
            selected.append((index, abs(value - 1.0)))
    if not selected:
        return "", 0.0, []
    delta = max(value for _, value in selected)
    high_label = "Inf" if not np.isfinite(high) else str(int(high))
    name = f"CMS_SUS26090_zgammaNonclosure_{regime}_u{int(low)}to{high_label}_2024"
    return name, delta, [index for index, _ in selected]


def low_geometry(label: str) -> tuple[float, float]:
    family = low_family(label)
    local_bin = int(re.search(r"_recoil_([0-9]+)$", label).group(1)) - 1
    if "PISR300to500" in family:
        low = 300.0 + 100.0 * local_bin
    elif "PISR500plus" in family:
        low = 450.0 + 100.0 * local_bin
    else:
        raise ValueError(f"cannot infer Low-dM U_T geometry for {label}")
    family_bins = 4 if family.startswith("Nb1_") else 3
    high = low + 100.0 if local_bin + 1 < family_bins else float("inf")
    return low, high


def rate_parameter(kind: str, regime: str, group: str, bin_index: int) -> str:
    return f"{kind}_{regime}_{group}_bin{bin_index}"


def add_extra(
    channel: dict[str, Any], process: str, records: list[dict[str, Any]]
) -> None:
    if records:
        channel.setdefault("extra_lnN", {}).setdefault(process, []).extend(records)


def top_w_composition(
    top: dict[str, Any] | None,
    wjets: dict[str, Any] | None,
    name: str,
) -> dict[str, list[dict[str, float | str]]]:
    """MC-statistical log-ratio prior with no first-order total-yield change."""

    if top is None or wjets is None:
        return {}
    top_yield = float(top["nominal"][0])
    w_yield = float(wjets["nominal"][0])
    if top_yield <= 0.0 or w_yield <= 0.0:
        return {}
    sigma = math.sqrt(
        max(float(top["sumw2"][0]), 0.0) / (top_yield * top_yield)
        + max(float(wjets["sumw2"][0]), 0.0) / (w_yield * w_yield)
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        return {}
    top_fraction = top_yield / (top_yield + w_yield)
    top_shift = (1.0 - top_fraction) * sigma
    w_shift = top_fraction * sigma
    return {
        "Top": [{"name": name, "down": math.exp(-top_shift), "up": math.exp(top_shift)}],
        "WtoLNu": [{"name": name, "down": math.exp(w_shift), "up": math.exp(-w_shift)}],
    }


def composition_name(regime: str, group: str, bin_index: int) -> str:
    return f"CMS_SUS26090_topWComposition_{regime}_{group}_bin{bin_index}_2024"


def high_output_bins() -> list[list[int]]:
    merged_by_first = {first: second for first, second in HIGH_MERGE_PAIRS}
    merged_seconds = {second for _, second in HIGH_MERGE_PAIRS}
    output = []
    for source_bin in range(60):
        if source_bin in merged_seconds:
            continue
        output.append(
            [source_bin, merged_by_first[source_bin]]
            if source_bin in merged_by_first
            else [source_bin]
        )
    if len(output) != 55 or sorted(sum(output, [])) != list(range(60)):
        raise AssertionError("invalid adopted High-dM 55-bin mapping")
    return output


def build_channels(
    exact: dict[str, Any],
    gcr_exact: dict[str, Any],
    zinv: dict[str, Any],
    sgamma: dict[str, Any],
    rz_covariance: dict[str, Any],
    double_ratio: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    bin_map: dict[str, Any] = {"highdm": [], "lowdm": []}
    high = exact["highdm"]
    high_gcr = gcr_exact["highdm"]
    high_nbin = len(high["recoil_edges"]) - 1

    for region in HIGH_CONTROL_REGIONS:
        sources = (high_gcr if region == "GCR" else high)["recoil"][region]
        for group in HIGH_PHYSICAL_GROUPS:
            for recoil_bin in range(high_nbin):
                backgrounds: dict[str, Any] = {}
                for process in BACKGROUND_PROCESS_ORDER:
                    scales = None
                    if region == "GCR" and process == "PhotonJet":
                        scales = {
                            physical: math.prod(high_sgamma(sgamma, physical, recoil_bin))
                            for physical in HIGH_PHYSICAL_GROUPS[group]
                        }
                    record = logical_records(
                        sources, group, process, recoil_bin, high_nbin, scales
                    )
                    if record is not None:
                        backgrounds[process] = record
                rate_params: dict[str, str] = {}
                if region == "LLCR":
                    parameter = rate_parameter("ll_norm", "highdm", group, recoil_bin)
                    for process in ("Top", "WtoLNu"):
                        if process in backgrounds:
                            rate_params[process] = parameter
                elif region == "QCDCR" and "QCD" in backgrounds:
                    rate_params["QCD"] = rate_parameter(
                        "qcd_norm", "highdm", group, recoil_bin
                    )
                elif region == "GCR" and "PhotonJet" in backgrounds:
                    rate_params["PhotonJet"] = rate_parameter(
                        "zg_norm", "highdm", group, recoil_bin
                    )
                observation = None
                if region == "GCR":
                    observation = sum(
                        float(
                            sources[physical]["data_obs"]["nominal"]["sumw"][recoil_bin]
                        )
                        for physical in HIGH_PHYSICAL_GROUPS[group]
                    )
                channel = {
                        "name": f"{region}_highdm_{group}_bin{recoil_bin}",
                        "kind": "highdm_control",
                        "regime": "highdm",
                        "region": region,
                        "nb_group": group,
                        "source_bin": recoil_bin,
                        "backgrounds": backgrounds,
                        "rate_params": rate_params,
                        "rate_initial": {process: 1.0 for process in rate_params},
                        "signal_source": None,
                        "observation": observation,
                        "extra_lnN": {},
                    }
                if region == "LLCR":
                    for process, records in top_w_composition(
                        backgrounds.get("Top"),
                        backgrounds.get("WtoLNu"),
                        composition_name("highdm", group, recoil_bin),
                    ).items():
                        add_extra(channel, process, records)
                channels.append(channel)

    high_components = high["sr_components"]
    rz_scale = {
        group: rz_value(rz_covariance, f"highdm_{group}")
        for group in HIGH_PHYSICAL_GROUPS
    }
    high_closure = []
    for recoil_bin in range(high_nbin):
        low, high_edge = high["recoil_edges"][recoil_bin : recoil_bin + 2]
        high_closure.append(closure_record(double_ratio, "highdm", low, high_edge))
    for output_bin, source_bins in enumerate(high_output_bins()):
        channel = {
            "name": f"SR_highdm_bin{output_bin}",
            "kind": "highdm_signal_searchbin",
            "regime": "highdm",
            "region": "SR",
            "source_bin": output_bin,
            "source_bins": source_bins,
            "backgrounds": {},
            "rate_params": {},
            "rate_initial": {},
            "signal_source": None,
            "signal_sources": [("highdm", source_bin) for source_bin in source_bins],
            "observation": None,
            "extra_lnN": {},
        }
        for source_bin in source_bins:
            recoil_bin = source_bin % high_nbin
            for group in HIGH_PHYSICAL_GROUPS:
                for process in ("Top", "WtoLNu", "QCD"):
                    record = logical_records(
                        high_components,
                        group,
                        process,
                        source_bin,
                        60,
                    )
                    if record is None:
                        continue
                    component = f"{process}_{group}_u{recoil_bin}"
                    channel["backgrounds"][component] = record
                    kind = (
                        "ll_norm"
                        if process in ("Top", "WtoLNu")
                        else "qcd_norm"
                        if process == "QCD"
                        else "zg_norm"
                    )
                    parameter = rate_parameter(kind, "highdm", group, recoil_bin)
                    channel["rate_params"][component] = parameter
                    channel["rate_initial"][component] = 1.0
                    if process in ("Top", "WtoLNu"):
                        composition = top_w_composition(
                            logical_records(
                                high["recoil"]["LLCR"],
                                group,
                                "Top",
                                recoil_bin,
                                high_nbin,
                            ),
                            logical_records(
                                high["recoil"]["LLCR"],
                                group,
                                "WtoLNu",
                                recoil_bin,
                                high_nbin,
                            ),
                            composition_name("highdm", group, recoil_bin),
                        )
                        add_extra(channel, component, composition.get(process, []))
                z_record = logical_records(
                    zinv["highdm"],
                    group,
                    "Zto2Nu",
                    source_bin,
                    60,
                    {
                        physical: rz_scale[group]
                        * high_sgamma(sgamma, physical, recoil_bin)[1]
                        for physical in HIGH_PHYSICAL_GROUPS[group]
                    },
                )
                if z_record is not None:
                    component = f"Zto2Nu_{group}_u{recoil_bin}"
                    channel["backgrounds"][component] = z_record
                    parameter = rate_parameter(
                        "zg_norm", "highdm", group, recoil_bin
                    )
                    channel["rate_params"][component] = parameter
                    channel["rate_initial"][component] = 1.0
                    add_extra(
                        channel,
                        component,
                        rz_nuisances(rz_covariance, f"highdm_{group}"),
                    )
                    name, delta, source = high_closure[recoil_bin]
                    if delta > 0.0:
                        add_extra(
                            channel,
                            component,
                            [{"name": name, "down": 1.0 / (1.0 + delta), "up": 1.0 + delta}],
                        )
        for process in RARE_PROCESSES:
            record = sum_one_bin_backgrounds(
                [
                    logical_records(
                        high_components, group, process, source_bin, 60
                    )
                    for source_bin in source_bins
                    for group in HIGH_PHYSICAL_GROUPS
                ]
            )
            if record is not None:
                channel["backgrounds"][process] = record
        channels.append(channel)
        bin_map["highdm"].append(
            {
                "channel": channel["name"],
                "source_bins_zero_based": source_bins,
                "native_recoil_bins": [source_bin % high_nbin for source_bin in source_bins],
            }
        )

    low = exact["lowdm"]
    low_gcr = gcr_exact["lowdm"]
    labels = low["search_bin_labels"]
    low_nbin = len(labels)
    for region in LOW_CONTROL_REGIONS:
        source_payload = (
            low_gcr["search_components"] if region == "GCR" else low["search_components"]
        )
        for source_bin, label in enumerate(labels):
            group = "Nb1" if source_bin < 16 else "Nb2plus"
            by_sample = source_payload[region][group]
            q, shape, family, local_bin = low_sgamma(sgamma, label)
            backgrounds = {}
            for process in BACKGROUND_PROCESS_ORDER:
                record = one_bin_background(by_sample, process, source_bin, low_nbin)
                if region == "GCR" and process == "PhotonJet":
                    record = scaled_record(record, q * shape)
                if record is not None:
                    backgrounds[process] = record
            rate_params: dict[str, str] = {}
            if region == "LLCR":
                parameter = rate_parameter("ll_norm", "lowdm", group, source_bin)
                for process in ("Top", "WtoLNu"):
                    if process in backgrounds:
                        rate_params[process] = parameter
            elif region == "QCDCR" and "QCD" in backgrounds:
                rate_params["QCD"] = rate_parameter(
                    "qcd_norm", "lowdm", group, source_bin
                )
            elif region == "GCR" and "PhotonJet" in backgrounds:
                rate_params["PhotonJet"] = rate_parameter(
                    "zg_norm", "lowdm", group, source_bin
                )
            observation = None
            if region == "GCR":
                observation = float(
                    by_sample["data_obs"]["nominal"]["sumw"][source_bin]
                )
            channel = {
                    "name": f"{region}_lowdm_bin{source_bin}",
                    "kind": "lowdm_control",
                    "regime": "lowdm",
                    "region": region,
                    "nb_group": group,
                    "source_bin": source_bin,
                    "bin_label": label,
                    "backgrounds": backgrounds,
                    "rate_params": rate_params,
                    "rate_initial": {process: 1.0 for process in rate_params},
                    "signal_source": None,
                    "observation": observation,
                    "extra_lnN": {},
                }
            if region == "LLCR":
                for process, records in top_w_composition(
                    backgrounds.get("Top"),
                    backgrounds.get("WtoLNu"),
                    composition_name("lowdm", group, source_bin),
                ).items():
                    add_extra(channel, process, records)
            channels.append(channel)

    for source_bin, label in enumerate(labels):
        group = "Nb1" if source_bin < 16 else "Nb2plus"
        by_sample = low["search_components"]["SR"][group]
        q, shape, family, local_bin = low_sgamma(sgamma, label)
        backgrounds = {}
        rate_params = {}
        rate_initial = {}
        channel = {
            "name": f"SR_lowdm_bin{source_bin}",
            "kind": "lowdm_signal_searchbin",
            "regime": "lowdm",
            "region": "SR",
            "nb_group": group,
            "source_bin": source_bin,
            "bin_label": label,
            "backgrounds": backgrounds,
            "rate_params": rate_params,
            "rate_initial": rate_initial,
            "signal_source": ("lowdm", source_bin),
            "observation": None,
            "extra_lnN": {},
        }
        for process in BACKGROUND_PROCESS_ORDER:
            if process == "Zto2Nu":
                continue
            record = one_bin_background(by_sample, process, source_bin, low_nbin)
            if record is None:
                continue
            if process in ("Top", "WtoLNu"):
                parameter = rate_parameter("ll_norm", "lowdm", group, source_bin)
            elif process == "QCD":
                parameter = rate_parameter("qcd_norm", "lowdm", group, source_bin)
            else:
                parameter = None
            backgrounds[process] = record
            if parameter:
                rate_params[process] = parameter
                rate_initial[process] = 1.0
                if process in ("Top", "WtoLNu"):
                    composition = top_w_composition(
                        one_bin_background(
                            low["search_components"]["LLCR"][group],
                            "Top",
                            source_bin,
                            low_nbin,
                        ),
                        one_bin_background(
                            low["search_components"]["LLCR"][group],
                            "WtoLNu",
                            source_bin,
                            low_nbin,
                        ),
                        composition_name("lowdm", group, source_bin),
                    )
                    add_extra(channel, process, composition.get(process, []))
        z_record = one_bin_background(
            zinv["lowdm"][group], "Zto2Nu", source_bin, low_nbin
        )
        if z_record is not None:
            z_record = scaled_record(
                z_record,
                rz_value(rz_covariance, f"lowdm_{group}") * shape,
            )
            backgrounds["Zto2Nu"] = z_record
            parameter = rate_parameter("zg_norm", "lowdm", group, source_bin)
            rate_params["Zto2Nu"] = parameter
            rate_initial["Zto2Nu"] = 1.0
            add_extra(
                channel,
                "Zto2Nu",
                rz_nuisances(rz_covariance, f"lowdm_{group}"),
            )
            low_edge, high_edge = low_geometry(label)
            name, delta, source = closure_record(
                double_ratio, "lowdm", low_edge, high_edge
            )
            if delta > 0.0:
                add_extra(
                    channel,
                    "Zto2Nu",
                    [{"name": name, "down": 1.0 / (1.0 + delta), "up": 1.0 + delta}],
                )
        channels.append(channel)
        bin_map["lowdm"].append(
            {
                "channel": channel["name"],
                "source_bin_zero_based": source_bin,
                "label": label,
                "family": family,
                "family_recoil_bin_zero_based": local_bin,
                "sgamma": shape,
                "qgamma": q,
            }
        )

    scopes: dict[str, set[str]] = {}
    for channel in channels:
        scope = "sr" if channel["region"] == "SR" else "cr"
        for parameter in channel["rate_params"].values():
            scopes.setdefault(parameter, set()).add(scope)
    invalid = sorted(
        parameter for parameter, scope in scopes.items() if scope != {"cr", "sr"}
    )
    invalid_set = set(invalid)
    for channel in channels:
        channel["rate_params"] = {
            process: parameter
            for process, parameter in channel["rate_params"].items()
            if parameter not in invalid_set
        }
        channel["rate_initial"] = {
            process: value
            for process, value in channel["rate_initial"].items()
            if process in channel["rate_params"]
        }
    return channels, invalid, bin_map


def overwrite_observations(
    output_root: Path,
    channels: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    import ROOT

    root_file = ROOT.TFile(str(output_root), "UPDATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot update {output_root}")
    try:
        for channel in channels:
            observation = channel.get("observation")
            if observation is None:
                continue
            directory = root_file.GetDirectory(channel["name"])
            if not directory:
                raise RuntimeError(f"missing ROOT directory {channel['name']}")
            value = np.asarray([max(float(observation), 0.0)])
            write_hist(directory, "data_obs", value, value, np.asarray([0.0, 1.0]))
            summary["channels"][channel["name"]]["observation"] = float(observation)
            summary["channels"][channel["name"]]["observation_source"] = "2024 data"
    finally:
        root_file.Close()


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    summary: dict[str, Any],
    auto_mc_stats: int,
) -> str:
    signal = signal_process_name(mass_key)
    channel_names = [channel["name"] for channel in channels]
    backgrounds = sorted(
        {
            process
            for channel in channels
            for process in channel["backgrounds"]
        }
    )
    background_ids = {
        process: index + 1 for index, process in enumerate(backgrounds)
    }
    channel_map = {channel["name"]: channel for channel in channels}
    columns: list[tuple[str, str, int]] = []
    for channel in channels:
        name = channel["name"]
        if summary["signals"][mass_key]["channels"].get(name, 0.0) > 0.0:
            columns.append((name, signal, 0))
        for process in sorted(channel["backgrounds"]):
            columns.append((name, process, background_ids[process]))
    nuisances = sorted(
        {
            nuisance
            for channel in channels
            for record in channel["backgrounds"].values()
            for nuisance in record["variations"]
        }
        | set(summary["signals"][mass_key]["weight_nuisances"])
    )
    lines = [
        "# 2024 adopted likelihood: simultaneous LLCR/QCDCR/GCR; external RZ; High-dM55 + Low-dM34.",
        "# RZ covariance is explicitly temporary diagonal statistical-only pending the final cross-category matrix.",
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {stable_path(template_root)} $CHANNEL/$PROCESS",
        "------------",
        "bin " + " ".join(channel_names),
        "observation " + " ".join(["-1"] * len(channel_names)),
        "------------",
        "bin " + " ".join(item[0] for item in columns),
        "process " + " ".join(item[1] for item in columns),
        "process " + " ".join(str(item[2]) for item in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
    ]
    signal_factors = summary["signals"][mass_key]["nuisance_factors"]
    for nuisance in nuisances:
        mask = []
        for channel_name, process, _ in columns:
            if process == signal:
                pair = (signal_factors.get(channel_name) or {}).get(nuisance)
            else:
                pair = (
                    (
                        summary["channels"][channel_name]["backgrounds"].get(process)
                        or {}
                    ).get("nuisance_factors")
                    or {}
                ).get(nuisance)
            mask.append("-" if not pair else f"{pair['down']:.8g}/{pair['up']:.8g}")
        lines.append(nuisance + " lnN " + " ".join(mask))
    lines.append(
        NPS_LUMI_NAME
        + " lnN "
        + " ".join(
            (
                f"{LUMI_LNN:.3f}"
                if process == signal
                or (
                    process in {"VV_VVV", "DY", "PhotonJet"}
                    and process not in channel_map[channel_name]["rate_params"]
                )
                else "-"
            )
            for channel_name, process, _ in columns
        )
    )
    extra_names = sorted(
        {
            item["name"]
            for channel in channels
            for records in channel.get("extra_lnN", {}).values()
            for item in records
        }
    )
    for nuisance in extra_names:
        mask = []
        for channel_name, process, _ in columns:
            item = next(
                (
                    record
                    for record in channel_map[channel_name]
                    .get("extra_lnN", {})
                    .get(process, [])
                    if record["name"] == nuisance
                ),
                None,
            )
            mask.append("-" if item is None else f"{item['down']:.8g}/{item['up']:.8g}")
        lines.append(nuisance + " lnN " + " ".join(mask))
    rate_lines = []
    for channel in channels:
        for process, parameter in sorted(channel["rate_params"].items()):
            initial = min(max(float(channel["rate_initial"].get(process, 1.0)), 1.0e-4), 9.999)
            rate_lines.append(
                f"{parameter} rateParam {channel['name']} {process} {initial:.8g} [0,10]"
            )
    lines.extend(rate_lines)
    if auto_mc_stats >= 0:
        lines.append(f"* autoMCStats {auto_mc_stats}")
    lines.extend(
        [
            "# Top=TT+ST. One ll_norm is shared by Top and W in each matched LLCR/SR bin.",
            "# A shared MC-derived anti-correlated Top/W composition nuisance accompanies each ll_norm in its matched LLCR/SR bin.",
            "# GCR statistics enter only through its Poisson channel; no Qgamma/Sgamma statistical Gaussian is duplicated.",
            "# DY2E/DY2M are excluded as Poisson channels and enter only through external RZ and Z/gamma closure.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_cards(
    channels: list[dict[str, Any]],
    masses: list[str],
    template_root: Path,
    summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for mass_key in masses:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(
            datacard_text(template_root, channels, mass_key, summary, auto_mc_stats)
        )
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--gcr-inputs", type=Path, required=True)
    parser.add_argument("--zinv-split-reference", type=Path, required=True)
    parser.add_argument("--sgamma", type=Path, required=True)
    parser.add_argument("--rz-high", type=Path, required=True)
    parser.add_argument("--rz-low", type=Path, required=True)
    parser.add_argument("--zgamma-double-ratio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=4)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    input_paths = {
        "hists": args.hists,
        "exact": args.exact_inputs,
        "gcr_exact": args.gcr_inputs,
        "zinv_split_reference": args.zinv_split_reference,
        "sgamma": args.sgamma,
        "rz_high": args.rz_high,
        "rz_low": args.rz_low,
        "zgamma_double_ratio": args.zgamma_double_ratio,
    }
    exact = read_json(args.exact_inputs)
    gcr_exact = read_json(args.gcr_inputs)
    zinv_split_reference = read_json(args.zinv_split_reference)
    sgamma = read_json(args.sgamma)
    rz_high = read_json(args.rz_high)
    rz_low = read_json(args.rz_low)
    double_ratio = read_json(args.zgamma_double_ratio)
    for label, payload in (
        ("exact", exact),
        ("gcr exact", gcr_exact),
        ("Z split reference", zinv_split_reference),
        ("Sgamma", sgamma),
        ("RZ high", rz_high),
        ("RZ low", rz_low),
        ("Z/gamma double ratio", double_ratio),
    ):
        if payload.get("status") != "complete":
            raise SystemExit(f"{label} input incomplete: {payload.get('status')}")

    requested_masses = list(args.only or [])
    if not requested_masses:
        raise SystemExit("--only is required for memory-safe extraction from hists.json")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hists = extract_small_histogram_input(args.hists, requested_masses)
    write_json(output_dir / "hist_inputs.json", hists)
    zinv = materialize_zinv_components(hists, zinv_split_reference)
    zinv["provenance"] = {
        "current_histogram": str(args.hists),
        "current_histogram_sha256": sha256(args.hists),
        "nb_split_reference": str(args.zinv_split_reference),
        "nb_split_reference_sha256": sha256(args.zinv_split_reference),
    }
    write_json(output_dir / "zinv_sr_inputs.json", zinv)
    rz_covariance = build_rz_covariance(rz_high, rz_low)
    write_json(output_dir / "rz_covariance.json", rz_covariance)
    channels, unmatched_rate_parameters, bin_map = build_channels(
        exact, gcr_exact, zinv, sgamma, rz_covariance, double_ratio
    )
    write_json(output_dir / "bin_map.json", bin_map)
    masses = mass_points(hists, args.only, args.max_mstop)
    if not masses:
        raise SystemExit("no signal mass points selected")
    template_root = output_dir / "templates.root"
    card_dir = output_dir / "cards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_limits.sh"
    summary = build_root(channels, hists, masses, template_root)
    overwrite_observations(template_root, channels, summary)
    cards = write_cards(
        channels, masses, template_root, summary, card_dir, args.auto_mc_stats
    )
    write_parallel_runner(
        cards, limit_dir, runner, args.runner_jobs, args.point_timeout
    )
    rate_parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel["rate_params"].values()
        }
    )
    scopes = {
        parameter: sorted(
            {
                "sr" if channel["region"] == "SR" else "cr"
                for channel in channels
                if parameter in channel["rate_params"].values()
            }
        )
        for parameter in rate_parameters
    }
    channel_counts = {
        "total": len(channels),
        "highdm_control": sum(item["kind"] == "highdm_control" for item in channels),
        "highdm_signal": sum(item["kind"] == "highdm_signal_searchbin" for item in channels),
        "lowdm_control": sum(item["kind"] == "lowdm_control" for item in channels),
        "lowdm_signal": sum(item["kind"] == "lowdm_signal_searchbin" for item in channels),
    }
    manifest = {
        "schema_version": "highdm55_lowdm34_an_zinv_cr_transfer_model_2024_v1",
        "status": "combine_inputs_ready",
        "model": {
            "simultaneous_control_regions": list(HIGH_CONTROL_REGIONS),
            "dilepton_poisson_channels": False,
            "lost_lepton": "one free ll_norm shared by Top and W per matched bin",
            "top_w_composition": "MC-derived log-ratio nuisance, anti-correlated at fixed total to first order",
            "qcd": "one free qcd_norm per matched QCDCR/SR bin",
            "zinv": "GCR Poisson rho shared with RZ*Sgamma-scaled Z SR",
            "rz_covariance": rz_covariance["status"],
            "zgamma_nonclosure": "central abs(D-1), Z SR only",
            "highdm_bins": 55,
            "lowdm_bins": 34,
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in input_paths.items()
        },
        "template_root": str(template_root),
        "cards": cards,
        "runner": str(runner),
        "mass_points": masses,
        "channels": channel_counts,
        "rate_parameter_count": len(rate_parameters),
        "rate_parameters": rate_parameters,
        "rate_parameter_scopes": scopes,
        "unmatched_rate_parameters_dropped": unmatched_rate_parameters,
        "auto_mc_stats": args.auto_mc_stats,
        "background_grouping": background_grouping_contract(),
        "root_summary": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "channels": channel_counts,
                "rate_parameters": len(rate_parameters),
                "unmatched": unmatched_rate_parameters,
                "mass_points": masses,
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
