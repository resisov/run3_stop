#!/usr/bin/env python3
"""Propagate the 2024+2025 CR-only fit into the High-dM VR MET templates.

VR observations are read only for the final comparison and never enter the
likelihood.  The recoil-binned CR parameters map exactly onto the canonical VR
MET bins; non-template diagnostic variables are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
from pathlib import Path
from typing import Any

import numpy as np

import build_combine_inputs as model


VR_REGIONS = ("HighDMVR_Nb1", "HighDMVR_Nb2", "HighDMVR_Nb3plus")
REGION_GROUP = {
    "HighDMVR_Nb1": "Nb1",
    "HighDMVR_Nb2": "Nb2plus",
    "HighDMVR_Nb3plus": "Nb2plus",
}
PROCESSES = tuple(model.BACKGROUND_PROCESS_ORDER)
CONTROL_REGION = {
    "Top": "LLCR",
    "WtoLNu": "LLCR",
    "QCD": "QCDCR",
    "Zto2Nu": "GCR",
}
RECOIL_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def bounded_value(payload: mmap.mmap, marker: bytes) -> tuple[int, int]:
    member = payload.find(marker)
    if member < 0:
        raise ValueError(f"canonical member {marker!r} is absent")
    start = member + len(marker)
    while payload[start] in b" \t\r\n":
        start += 1
    return start, model.json_value_end(payload, start)


def extract_vr_met(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    if path.name != "hists.json":
        raise ValueError("only promoted canonical hists.json is allowed")
    forbidden = ("nanoaod", "flat2024", "flat2025", "outputs/nominal")
    if any(token in str(path).lower() for token in forbidden):
        raise ValueError(f"forbidden downstream input path: {path}")
    output = {}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            section = bounded_value(payload, b'"highdm_variable_histograms":')
            region_members = dict(model.object_member_bounds(payload, section))
            for region in VR_REGIONS:
                variables = dict(
                    model.object_member_bounds(payload, region_members[region])
                )
                output[region] = model.project_sample_tree(
                    payload,
                    variables["met"],
                    0,
                    lambda sample: sample in model.CANONICAL_BACKGROUND_SAMPLES,
                )
            specs_bounds = bounded_value(
                payload, b'"highdm_distribution_variable_specs":'
            )
            met_bounds = model.named_member_bounds(payload, specs_bounds, "met")
            met_spec = json.loads(payload[met_bounds[0] : met_bounds[1]])
    edges = np.asarray(met_spec["bins"], dtype=float)
    if not np.array_equal(edges, np.asarray([250, 300, 350, 400, 500, 650, 800, 1000, 1500], dtype=float)):
        raise ValueError(f"unexpected canonical VR MET edges: {edges.tolist()}")
    return output, edges


def recoil_source_bins(edges: np.ndarray) -> list[int]:
    output = []
    for low, high in zip(edges[:-1], edges[1:]):
        matches = np.flatnonzero(
            (low >= RECOIL_EDGES[:-1]) & (high <= RECOIL_EDGES[1:])
        )
        if len(matches) != 1:
            raise ValueError(f"VR MET bin {low:g}-{high:g} crosses CR recoil bins")
        output.append(int(matches[0]))
    return output


def data_arrays(by_sample: dict[str, Any], nbin: int) -> tuple[np.ndarray, np.ndarray]:
    record = (by_sample.get("data_obs") or {}).get("nominal") or {}
    values = np.asarray(record.get("sumw") or [], dtype=float)
    sumw2 = np.asarray(record.get("sumw2") or [], dtype=float)
    if len(values) != nbin or len(sumw2) != nbin:
        raise ValueError("VR data histogram has the wrong bin count")
    return values, sumw2


def factor_value_and_slope(theta: float, down: float, up: float) -> tuple[float, float]:
    if not (down > 0.0 and up > 0.0):
        raise ValueError(f"nonpositive lnN factor: {down}/{up}")
    if theta > 0.0:
        slope = math.log(up)
        value = math.exp(theta * slope)
    elif theta < 0.0:
        slope = -math.log(down)
        value = math.exp(theta * slope)
    else:
        slope = 0.5 * (math.log(up) - math.log(down))
        value = 1.0
    return value, slope


def add_factor(target: dict[str, tuple[float, float]], item: dict[str, Any]) -> None:
    name = str(item["name"])
    pair = (float(item["down"]), float(item["up"]))
    if name in target and not np.allclose(target[name], pair, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"conflicting nuisance factor for {name}")
    target[name] = pair


def variation_factors(
    by_sample: dict[str, Any], process: str, source_bin: int, nbin: int
) -> dict[str, tuple[float, float]]:
    nominal, _ = model.leaf_arrays(by_sample, process, "nominal", nbin)
    value = max(float(nominal[source_bin]), model.MIN_BIN)
    floor = max(model.MIN_BIN, value * model.MIN_VARIATION_RATIO)
    output = {}
    for nuisance, pair in model.variation_pairs(by_sample, process, nbin).items():
        down = max(float(pair["down"][source_bin]), floor) / value
        up = max(float(pair["up"][source_bin]), floor) / value
        if np.isclose(down, 1.0, rtol=1.0e-12, atol=1.0e-15) and np.isclose(up, 1.0, rtol=1.0e-12, atol=1.0e-15):
            continue
        output[nuisance] = (down, up)
    return output


def build_year_components(
    year: str,
    hists: Path,
    sgamma: dict[str, Any],
    rz_covariance: dict[str, Any],
    double_ratio: dict[str, Any],
) -> dict[str, Any]:
    model.CAMPAIGN_YEAR = year
    model.NPS_LUMI_NAME = f"lumi_13p6TeV_{year}"
    vr, edges = extract_vr_met(hists)
    nbin = len(edges) - 1
    source_bins = recoil_source_bins(edges)
    control = model.extract_component_tree(
        hists, "highdm_control_components", levels_before_samples=2
    )
    output = {"edges": edges.tolist(), "regions": {}}

    for region in VR_REGIONS:
        by_sample = vr[region]
        group = REGION_GROUP[region]
        data, data_sumw2 = data_arrays(by_sample, nbin)
        region_output = {
            "data": data.tolist(),
            "data_sumw2": data_sumw2.tolist(),
            "components": [],
        }
        for plot_bin, recoil_bin in enumerate(source_bins):
            low, high = RECOIL_EDGES[recoil_bin : recoil_bin + 2]
            for process in PROCESSES:
                nominal, sumw2 = model.leaf_arrays(
                    by_sample, process, "nominal", nbin
                )
                base = float(nominal[plot_bin])
                variance = float(sumw2[plot_bin])
                if base <= 0.0 and variance <= 0.0:
                    continue
                base = max(base, model.MIN_BIN)
                static_scale = 1.0
                rate_parameter = None
                factors = {
                    name: pair
                    for name, pair in variation_factors(
                        by_sample, process, plot_bin, nbin
                    ).items()
                }
                if process in ("Top", "WtoLNu"):
                    rate_parameter = model.rate_parameter(
                        "ll_norm", "highdm", group, recoil_bin
                    )
                    composition = model.top_w_composition(
                        model.logical_records(
                            control["LLCR"], group, "Top", recoil_bin, len(RECOIL_EDGES) - 1
                        ),
                        model.logical_records(
                            control["LLCR"], group, "WtoLNu", recoil_bin, len(RECOIL_EDGES) - 1
                        ),
                        model.composition_name("highdm", group, recoil_bin),
                    )
                    for item in composition.get(process, []):
                        add_factor(factors, item)
                elif process == "QCD":
                    rate_parameter = model.rate_parameter(
                        "qcd_norm", "highdm", group, recoil_bin
                    )
                elif process == "Zto2Nu":
                    static_scale = model.rz_value(
                        rz_covariance, f"highdm_{group}"
                    )
                    rate_parameter = model.rate_parameter(
                        "sgamma_shape", "highdm", group, recoil_bin
                    )
                    for item in model.rz_nuisances(
                        rz_covariance, f"highdm_{group}"
                    ):
                        add_factor(factors, item)
                    closure_name, delta, _ = model.closure_record(
                        double_ratio, "highdm", float(low), float(high)
                    )
                    if delta > 0.0:
                        add_factor(
                            factors,
                            {
                                "name": closure_name,
                                "down": 1.0 / (1.0 + delta),
                                "up": 1.0 + delta,
                            },
                        )
                if process in model.RARE_PROCESSES:
                    add_factor(
                        factors,
                        {
                            "name": model.NPS_LUMI_NAME,
                            "down": 1.0 / model.LUMI_LNN,
                            "up": model.LUMI_LNN,
                        },
                    )

                transfer_factor = None
                if process in CONTROL_REGION:
                    if process == "Zto2Nu":
                        scales = {
                            physical: model.high_sgamma(
                                sgamma, physical, recoil_bin
                            )[0]
                            for physical in model.HIGH_PHYSICAL_GROUPS[group]
                        }
                        denominator_record = model.logical_records(
                            control["GCR"],
                            group,
                            "PhotonJet",
                            recoil_bin,
                            len(RECOIL_EDGES) - 1,
                            scales,
                        )
                        numerator = base * static_scale
                    else:
                        denominator_record = model.logical_records(
                            control[CONTROL_REGION[process]],
                            group,
                            process,
                            recoil_bin,
                            len(RECOIL_EDGES) - 1,
                        )
                        numerator = base
                    denominator = (
                        float(denominator_record["nominal"][0])
                        if denominator_record is not None
                        else 0.0
                    )
                    if denominator > 0.0:
                        transfer_factor = numerator / denominator

                region_output["components"].append(
                    {
                        "plot_bin": plot_bin,
                        "recoil_bin": recoil_bin,
                        "process": process,
                        "base": base,
                        "sumw2": max(variance, 0.0),
                        "static_scale": static_scale,
                        "rate_parameter": rate_parameter,
                        "nuisance_factors": {
                            name: {"down": pair[0], "up": pair[1]}
                            for name, pair in sorted(factors.items())
                        },
                        "cr_to_vr_transfer_factor": transfer_factor,
                    }
                )
        output["regions"][region] = region_output
    return output


def prediction(
    years: dict[str, Any], fit: dict[str, Any]
) -> dict[str, Any]:
    fit_order = list(fit["parameter_order"])
    fit_parameters = fit["parameters"]
    covariance = np.asarray(fit["covariance"], dtype=float)
    needed_nuisances = sorted(
        {
            nuisance
            for year in years.values()
            for region in year["regions"].values()
            for component in region["components"]
            for nuisance in component["nuisance_factors"]
        }
    )
    missing = [name for name in needed_nuisances if name not in fit_parameters]
    order = fit_order + missing
    index = {name: position for position, name in enumerate(order)}
    augmented = np.zeros((len(order), len(order)), dtype=float)
    augmented[: len(fit_order), : len(fit_order)] = covariance
    for name in missing:
        augmented[index[name], index[name]] = 1.0

    parameter_value = {
        name: float(fit_parameters[name]["value"]) for name in fit_order
    }
    parameter_initial = {
        name: float(fit_parameters[name]["initial"])
        for name in fit_order
        if fit_parameters[name].get("initial") is not None
    }
    for name in missing:
        parameter_value[name] = 0.0
        parameter_initial[name] = 0.0

    def evaluate_component(component: dict[str, Any], values: dict[str, float]):
        nominal = float(component["base"]) * float(component["static_scale"])
        log_slopes: dict[str, float] = {}
        rate_name = component.get("rate_parameter")
        if rate_name:
            if rate_name not in values:
                raise ValueError(f"fit result is missing rate parameter {rate_name}")
            rate = float(values[rate_name])
            if rate <= 0.0:
                raise ValueError(f"nonpositive fitted rate parameter {rate_name}={rate}")
            nominal *= rate
            log_slopes[rate_name] = 1.0 / rate
        for name, pair in component["nuisance_factors"].items():
            theta = float(values.get(name, 0.0))
            factor, slope = factor_value_and_slope(
                theta, float(pair["down"]), float(pair["up"])
            )
            nominal *= factor
            log_slopes[name] = log_slopes.get(name, 0.0) + slope
        gradient = {
            name: nominal * slope for name, slope in log_slopes.items()
        }
        scale = nominal / max(float(component["base"]), model.MIN_BIN)
        mc_variance = float(component["sumw2"]) * scale * scale
        return nominal, gradient, mc_variance

    evaluated = {}
    for year, year_data in years.items():
        evaluated[year] = {}
        nbin = len(year_data["edges"]) - 1
        for region_name, region in year_data["regions"].items():
            process_values = {process: np.zeros(nbin) for process in PROCESSES}
            process_prefit = {process: np.zeros(nbin) for process in PROCESSES}
            gradients = np.zeros((nbin, len(order)))
            prefit_gradients = np.zeros((nbin, len(order)))
            mc_variance = np.zeros(nbin)
            prefit_mc_variance = np.zeros(nbin)
            for component in region["components"]:
                plot_bin = int(component["plot_bin"])
                process = component["process"]
                value, gradient, variance = evaluate_component(
                    component, parameter_value
                )
                prefit_value, prefit_gradient, prefit_variance = evaluate_component(
                    component, parameter_initial
                )
                process_values[process][plot_bin] += value
                process_prefit[process][plot_bin] += prefit_value
                mc_variance[plot_bin] += variance
                prefit_mc_variance[plot_bin] += prefit_variance
                for name, derivative in gradient.items():
                    gradients[plot_bin, index[name]] += derivative
                for name, derivative in prefit_gradient.items():
                    prefit_gradients[plot_bin, index[name]] += derivative
            total = sum(process_values.values(), np.zeros(nbin))
            prefit_total = sum(process_prefit.values(), np.zeros(nbin))
            postfit_covariance = gradients @ augmented @ gradients.T
            postfit_covariance += np.diag(mc_variance)
            prefit_parameter_covariance = np.zeros_like(augmented)
            for nuisance in needed_nuisances:
                prefit_parameter_covariance[index[nuisance], index[nuisance]] = 1.0
            prefit_covariance = (
                prefit_gradients @ prefit_parameter_covariance @ prefit_gradients.T
                + np.diag(prefit_mc_variance)
            )
            evaluated[year][region_name] = {
                "processes": {key: value.tolist() for key, value in process_values.items()},
                "prefit_processes": {key: value.tolist() for key, value in process_prefit.items()},
                "total": total.tolist(),
                "prefit_total": prefit_total.tolist(),
                "uncertainty": np.sqrt(np.maximum(np.diag(postfit_covariance), 0.0)).tolist(),
                "prefit_uncertainty": np.sqrt(np.maximum(np.diag(prefit_covariance), 0.0)).tolist(),
                "covariance": postfit_covariance.tolist(),
                "prefit_covariance": prefit_covariance.tolist(),
                "gradient": gradients.tolist(),
                "prefit_gradient": prefit_gradients.tolist(),
                "mc_stat_variance": mc_variance.tolist(),
                "prefit_mc_stat_variance": prefit_mc_variance.tolist(),
                "data": region["data"],
                "data_sumw2": region["data_sumw2"],
            }

    combined = {}
    for region_name in VR_REGIONS:
        nbin = len(next(iter(years.values()))["edges"]) - 1
        processes = {process: np.zeros(nbin) for process in PROCESSES}
        prefit_processes = {process: np.zeros(nbin) for process in PROCESSES}
        total = np.zeros(nbin)
        prefit_total = np.zeros(nbin)
        data = np.zeros(nbin)
        data_sumw2 = np.zeros(nbin)
        gradient = np.zeros((nbin, len(order)))
        prefit_gradient = np.zeros((nbin, len(order)))
        mc_stat_variance = np.zeros(nbin)
        prefit_mc_stat_variance = np.zeros(nbin)
        covariance_mc = np.zeros((nbin, nbin))
        for year in years:
            record = evaluated[year][region_name]
            for process in PROCESSES:
                processes[process] += np.asarray(record["processes"][process])
                prefit_processes[process] += np.asarray(
                    record["prefit_processes"][process]
                )
            total += np.asarray(record["total"])
            prefit_total += np.asarray(record["prefit_total"])
            data += np.asarray(record["data"])
            data_sumw2 += np.asarray(record["data_sumw2"])
            gradient += np.asarray(record["gradient"])
            prefit_gradient += np.asarray(record["prefit_gradient"])
            year_mc_stat = np.asarray(record["mc_stat_variance"])
            mc_stat_variance += year_mc_stat
            covariance_mc += np.diag(year_mc_stat)
            prefit_mc_stat_variance += np.asarray(
                record["prefit_mc_stat_variance"]
            )
        postfit_covariance = gradient @ augmented @ gradient.T + covariance_mc
        prefit_parameter_covariance = np.zeros_like(augmented)
        for nuisance in needed_nuisances:
            prefit_parameter_covariance[index[nuisance], index[nuisance]] = 1.0
        prefit_covariance = (
            prefit_gradient
            @ prefit_parameter_covariance
            @ prefit_gradient.T
            + np.diag(prefit_mc_stat_variance)
        )
        combined[region_name] = {
            "processes": {key: value.tolist() for key, value in processes.items()},
            "prefit_processes": {
                key: value.tolist() for key, value in prefit_processes.items()
            },
            "total": total.tolist(),
            "prefit_total": prefit_total.tolist(),
            "uncertainty": np.sqrt(np.maximum(np.diag(postfit_covariance), 0.0)).tolist(),
            "prefit_uncertainty": np.sqrt(
                np.maximum(np.diag(prefit_covariance), 0.0)
            ).tolist(),
            "covariance": postfit_covariance.tolist(),
            "prefit_covariance": prefit_covariance.tolist(),
            "mc_stat_variance": mc_stat_variance.tolist(),
            "data": data.tolist(),
            "data_sumw2": data_sumw2.tolist(),
        }
    return {
        "status": "complete",
        "fit": "2024+2025 observed CR-only background-only fit",
        "template_observable": "met",
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "parameter_order": order,
        "fit_parameter_count": len(fit_order),
        "prediction_only_prior_parameters": missing,
        "years": evaluated,
        "combined": combined,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for year in ("2024", "2025"):
        parser.add_argument(f"--hists-{year}", required=True, type=Path)
        parser.add_argument(f"--sgamma-{year}", required=True, type=Path)
        parser.add_argument(f"--rz-covariance-{year}", required=True, type=Path)
        parser.add_argument(f"--zgamma-double-ratio-{year}", required=True, type=Path)
    parser.add_argument("--fit-covariance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    years = {}
    for year in ("2024", "2025"):
        years[year] = build_year_components(
            year,
            getattr(args, f"hists_{year}"),
            read_json(getattr(args, f"sgamma_{year}")),
            read_json(getattr(args, f"rz_covariance_{year}")),
            read_json(getattr(args, f"zgamma_double_ratio_{year}")),
        )
    result = prediction(years, read_json(args.fit_covariance))
    result["edges"] = years["2024"]["edges"]
    result["inputs"] = {
        year: {
            "hists": str(getattr(args, f"hists_{year}")),
            "sgamma": str(getattr(args, f"sgamma_{year}")),
            "rz_covariance": str(getattr(args, f"rz_covariance_{year}")),
            "zgamma_double_ratio": str(getattr(args, f"zgamma_double_ratio_{year}")),
        }
        for year in ("2024", "2025")
    }
    result["components"] = years
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "template_observable": result["template_observable"],
                "regions": list(VR_REGIONS),
                "prediction_only_priors": len(result["prediction_only_prior_parameters"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
