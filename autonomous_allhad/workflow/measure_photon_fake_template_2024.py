#!/usr/bin/env python3
"""Measure a Run-2-style shower-shape template fake-photon factor.

Inputs are compact events produced by
``autonomous_allhad.photon_fake_template_2024_worker``.  The measurement is
performed in ``GCR_DPhiVR_Low`` and validated in the disjoint
``GCR_DPhiVR_High`` region before being evaluated in the nominal GCR.  Nominal
histograms are never modified by this program.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


INPUT_SCHEMA = "photon_fake_template_events_2024_v1"
OUTPUT_SCHEMA = "photon_fake_template_measurement_2024_v1"
MEASUREMENT_REGION = "GCR_DPhiVR_Low"
VALIDATION_REGION = "GCR_DPhiVR_High"
APPLICATION_REGION = "GCR"
UT_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0])
FINE_PT_EDGES = (220.0, 300.0, 400.0, 600.0, 1_000_000.0)
MIN_DATA_FIT_EVENTS = 12
MIN_FAKE_TEMPLATE_EVENTS = 20
MIN_MC_TEMPLATE_EFFECTIVE_EVENTS = 4.0


def read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Group:
    name: str
    eta: str
    pt_low: float
    pt_high: float
    tier: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eta": self.eta,
            "pt_low": self.pt_low,
            "pt_high": self.pt_high,
            "tier": self.tier,
        }


def groups() -> list[Group]:
    result: list[Group] = []
    for eta in ("EB", "EE"):
        for low, high in zip(FINE_PT_EDGES[:-1], FINE_PT_EDGES[1:]):
            upper = "inf" if high >= 1_000_000.0 else f"{high:g}"
            result.append(Group(f"{eta}_pt{low:g}to{upper}", eta, low, high, "fine"))
        result.extend(
            [
                Group(f"{eta}_pt220to400", eta, 220.0, 400.0, "coarse"),
                Group(f"{eta}_pt400toinf", eta, 400.0, 1_000_000.0, "coarse"),
                Group(f"{eta}_inclusive", eta, 220.0, 1_000_000.0, "inclusive"),
            ]
        )
    return result


class EventTable:
    def __init__(self, events: list[dict[str, Any]], normalization: dict[str, Any]):
        physical = normalization.get("physical_datasets") or {}
        n = len(events)
        self.n = n
        self.run = np.empty(n, dtype=np.int64)
        self.lumi = np.empty(n, dtype=np.int64)
        self.event = np.empty(n, dtype=np.int64)
        self.is_data = np.empty(n, dtype=bool)
        self.process = np.empty(n, dtype=object)
        self.origin = np.empty(n, dtype=object)
        self.pt = np.empty(n, dtype=float)
        self.eta = np.empty(n, dtype=float)
        self.sieie = np.empty(n, dtype=float)
        self.charged_iso = np.empty(n, dtype=float)
        self.shape_level = np.empty(n, dtype=np.int8)
        self.charged_level = np.empty(n, dtype=np.int8)
        self.ut = np.empty(n, dtype=float)
        self.weight = np.empty(n, dtype=float)
        self.regions: list[set[str]] = []
        missing_normalization: set[str] = set()
        for index, item in enumerate(events):
            probe = item["probe"]
            self.run[index] = int(item["run"])
            self.lumi[index] = int(item["luminosityBlock"])
            self.event[index] = int(item["event"])
            self.is_data[index] = bool(item["is_data"])
            self.process[index] = str(item["process"])
            self.origin[index] = str(probe["origin"])
            self.pt[index] = float(probe["pt"])
            self.eta[index] = float(probe["eta"])
            self.sieie[index] = float(probe["sieie"])
            self.charged_iso[index] = float(probe["charged_iso"])
            self.shape_level[index] = int(probe["shape_level"])
            self.charged_level[index] = int(probe["charged_iso_level"])
            self.ut[index] = float(item["values"]["ut"])
            self.regions.append(set(item.get("regions") or []))
            if self.is_data[index]:
                self.weight[index] = 1.0
            else:
                key = str(item["physical_dataset"])
                record = physical.get(key)
                if record is None:
                    missing_normalization.add(key)
                    self.weight[index] = np.nan
                else:
                    factor = record.get("normalization_factor")
                    if factor is None:
                        xsec = record.get("xsec_pb")
                        sumw = record.get("sumw")
                        luminosity = normalization.get("luminosity_pb")
                        if xsec is None or sumw in (None, 0) or luminosity is None:
                            missing_normalization.add(key)
                            self.weight[index] = np.nan
                        else:
                            factor = float(luminosity) * float(xsec) / float(sumw)
                    self.weight[index] = float(item["nominal_weight_without_photon_id_sf"]) * float(factor)
        if missing_normalization:
            raise RuntimeError(
                f"missing normalization for {len(missing_normalization)} physical datasets: "
                f"{sorted(missing_normalization)[:10]}"
            )
        self.region_masks = {
            region: np.asarray([region in memberships for memberships in self.regions], dtype=bool)
            for region in (MEASUREMENT_REGION, VALIDATION_REGION, APPLICATION_REGION)
        }

    def group_mask(self, group: Group) -> np.ndarray:
        eta_mask = abs(self.eta) < 1.4442 if group.eta == "EB" else abs(self.eta) > 1.5660
        return eta_mask & (self.pt >= group.pt_low) & (self.pt < group.pt_high)


def deduplicate_data(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique: dict[tuple[int, int, int], dict[str, Any]] = {}
    exact_duplicates = 0
    conflicts: list[tuple[int, int, int]] = []
    mc: list[dict[str, Any]] = []
    for item in events:
        if not bool(item.get("is_data")):
            mc.append(item)
            continue
        key = (int(item["run"]), int(item["luminosityBlock"]), int(item["event"]))
        if key not in unique:
            unique[key] = item
        else:
            def physics_signature(record: dict[str, Any]) -> str:
                return digest_payload(
                    {
                        "process": record.get("process"),
                        "is_data": record.get("is_data"),
                        "probe": record.get("probe"),
                        "regions": sorted(record.get("regions") or []),
                        "values": record.get("values"),
                    }
                )
            if physics_signature(unique[key]) == physics_signature(item):
                exact_duplicates += 1
            else:
                conflicts.append(key)
    if conflicts:
        raise RuntimeError(f"conflicting duplicate data events: {conflicts[:10]}")
    return mc + list(unique.values()), {
        "input_data_records": len(events) - len(mc),
        "unique_data_events": len(unique),
        "exact_duplicates_removed": exact_duplicates,
        "conflicting_duplicates": len(conflicts),
    }


def load_inputs(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    digests: set[str] = set()
    accepted: list[str] = []
    incomplete: list[str] = []
    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != INPUT_SCHEMA:
            continue
        summary = payload.get("summary") or {}
        digest = str(summary.get("source_record_digest") or "")
        if digest and digest in digests:
            raise RuntimeError(f"duplicate source record digest: {digest}")
        if digest:
            digests.add(digest)
        if payload.get("status") != "complete":
            incomplete.append(str(path))
        events.extend(payload.get("events") or [])
        accepted.append(str(path))
    if not accepted:
        raise RuntimeError(f"no {INPUT_SCHEMA} inputs found")
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} template-event inputs are incomplete")
    return events, {
        "accepted_files": len(accepted),
        "source_record_digests": len(digests),
        "event_records": len(events),
    }


def sieie_edges(eta: str) -> np.ndarray:
    return np.linspace(0.005, 0.030, 26) if eta == "EB" else np.linspace(0.012, 0.060, 25)


def histogram(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(values, edges[0] + 1.0e-9, edges[-1] - 1.0e-9)
    sumw = np.histogram(clipped, bins=edges, weights=weights)[0].astype(float)
    sumw2 = np.histogram(clipped, bins=edges, weights=weights * weights)[0].astype(float)
    return sumw, sumw2


def effective_events(sumw: np.ndarray, sumw2: np.ndarray) -> float:
    total = float(np.sum(sumw))
    variance = float(np.sum(sumw2))
    return total * total / variance if variance > 0.0 else 0.0


def normalized_shape(values: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 0.0) + float(alpha)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full(len(clipped), 1.0 / len(clipped))
    return clipped / total


def extended_template_fit(
    observation: np.ndarray,
    prompt_shape: np.ndarray,
    fake_shape: np.ndarray,
    fixed_electron: np.ndarray,
) -> dict[str, Any]:
    observation = np.asarray(observation, dtype=float)
    fixed_electron = np.maximum(np.asarray(fixed_electron, dtype=float), 0.0)
    prompt_shape = normalized_shape(prompt_shape, alpha=1.0e-9)
    fake_shape = normalized_shape(fake_shape, alpha=1.0e-9)
    free_total = max(float(np.sum(observation) - np.sum(fixed_electron)), 1.0)

    def nll(parameters: np.ndarray) -> float:
        expectation = fixed_electron + parameters[0] * prompt_shape + parameters[1] * fake_shape
        expectation = np.maximum(expectation, 1.0e-12)
        return float(np.sum(expectation - observation * np.log(expectation)))

    result = minimize(
        nll,
        np.asarray([0.5 * free_total, 0.5 * free_total]),
        method="L-BFGS-B",
        bounds=((0.0, None), (0.0, None)),
    )
    parameters = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    expectation = fixed_electron + parameters[0] * prompt_shape + parameters[1] * fake_shape
    expectation = np.maximum(expectation, 1.0e-12)
    information = np.asarray(
        [
            [np.sum(prompt_shape * prompt_shape / expectation), np.sum(prompt_shape * fake_shape / expectation)],
            [np.sum(prompt_shape * fake_shape / expectation), np.sum(fake_shape * fake_shape / expectation)],
        ],
        dtype=float,
    )
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    positive_observation = observation[observation > 0.0]
    saturated = float(np.sum(positive_observation - positive_observation * np.log(positive_observation)))
    deviance = max(0.0, 2.0 * (float(result.fun) - saturated))
    return {
        "success": bool(result.success and np.all(np.isfinite(parameters))),
        "message": str(result.message),
        "prompt_yield": float(parameters[0]),
        "fake_yield": float(parameters[1]),
        "covariance": covariance.tolist(),
        "deviance": deviance,
        "degrees_of_freedom": max(0, len(observation) - 2),
        "observation": observation.tolist(),
        "expectation": expectation.tolist(),
        "prompt_component": (parameters[0] * prompt_shape).tolist(),
        "fake_component": (parameters[1] * fake_shape).tolist(),
        "electron_component": fixed_electron.tolist(),
    }


def base_mask(table: EventTable, region: str, group: Group) -> np.ndarray:
    return table.region_masks[region] & table.group_mask(group)


def stage_mask(table: EventTable, stage: str) -> np.ndarray:
    if stage == "pass_charged_iso":
        return table.charged_level >= 2
    if stage == "loose_charged_iso":
        return table.charged_level == 1
    if stage == "fake_template":
        return table.charged_level == 0
    raise KeyError(stage)


def fake_template(
    table: EventTable,
    region: str,
    group: Group,
    partition: str = "all",
    electron_scale: float = 1.0,
    prompt_scale: float = 1.0,
) -> dict[str, Any]:
    edges = sieie_edges(group.eta)
    base = base_mask(table, region, group) & stage_mask(table, "fake_template")
    data = base & table.is_data
    partition_boundary = None
    if partition in ("low_charged_iso", "high_charged_iso"):
        sideband_values = table.charged_iso[data]
        if len(sideband_values) >= 2 * MIN_FAKE_TEMPLATE_EVENTS:
            partition_boundary = float(np.median(sideband_values))
            if partition == "low_charged_iso":
                base &= table.charged_iso <= partition_boundary
            else:
                base &= table.charged_iso > partition_boundary
            data = base & table.is_data
    elif partition != "all":
        raise KeyError(partition)
    prompt = base & ~table.is_data & (table.origin == "prompt")
    electron = base & ~table.is_data & (table.origin == "electron")
    data_hist, data_var = histogram(table.sieie[data], np.ones(np.count_nonzero(data)), edges)
    prompt_hist, prompt_var = histogram(table.sieie[prompt], table.weight[prompt], edges)
    electron_hist, electron_var = histogram(table.sieie[electron], table.weight[electron], edges)
    prompt_hist *= float(prompt_scale)
    prompt_var *= float(prompt_scale) * float(prompt_scale)
    electron_hist *= float(electron_scale)
    electron_var *= float(electron_scale) * float(electron_scale)
    residual = data_hist - prompt_hist - electron_hist
    clipped_bins = int(np.count_nonzero(residual < 0.0))
    shape = normalized_shape(residual, alpha=0.5)
    shape_pass_data = float(np.count_nonzero(data & (table.shape_level >= 2)))
    shape_pass_prompt = float(prompt_scale) * float(
        np.sum(table.weight[prompt & (table.shape_level >= 2)])
    )
    shape_pass_electron = float(electron_scale) * float(
        np.sum(table.weight[electron & (table.shape_level >= 2)])
    )
    pass_residual = max(0.0, shape_pass_data - shape_pass_prompt - shape_pass_electron)
    total_residual = max(float(np.sum(np.maximum(residual, 0.0))), 1.0e-12)
    pass_fraction = min(1.0, max(0.0, pass_residual / total_residual))
    return {
        "edges": edges.tolist(),
        "partition": partition,
        "partition_boundary": partition_boundary,
        "electron_scale": float(electron_scale),
        "prompt_scale": float(prompt_scale),
        "shape": shape.tolist(),
        "pass_fraction": pass_fraction,
        "data_events": int(np.count_nonzero(data)),
        "data_histogram": data_hist.tolist(),
        "prompt_contamination": prompt_hist.tolist(),
        "electron_contamination": electron_hist.tolist(),
        "residual_before_clipping": residual.tolist(),
        "clipped_bins": clipped_bins,
        "effective_prompt_events": effective_events(prompt_hist, prompt_var),
        "effective_electron_events": effective_events(electron_hist, electron_var),
        "data_variance": data_var.tolist(),
    }


def fit_stage(
    table: EventTable,
    region: str,
    group: Group,
    stage: str,
    template: dict[str, Any],
    extra_mask: np.ndarray | None = None,
    electron_scale: float = 1.0,
    stage_specific_prompt_shape: bool = False,
) -> dict[str, Any]:
    edges = np.asarray(template["edges"], dtype=float)
    selected = base_mask(table, region, group) & stage_mask(table, stage)
    if extra_mask is not None:
        selected &= extra_mask
    data = selected & table.is_data
    prompt = selected & ~table.is_data & (table.origin == "prompt")
    electron = selected & ~table.is_data & (table.origin == "electron")
    observation, _ = histogram(table.sieie[data], np.ones(np.count_nonzero(data)), edges)
    # Use all charged-isolation states in the same topology for a statistically
    # stable prompt shower-shape template; only its fitted normalization moves.
    prompt_shape_source = base_mask(table, region, group) & ~table.is_data & (table.origin == "prompt")
    if stage_specific_prompt_shape:
        prompt_shape_source &= stage_mask(table, stage)
    if extra_mask is not None:
        prompt_shape_source &= extra_mask
    prompt_shape, prompt_shape_var = histogram(
        table.sieie[prompt_shape_source], table.weight[prompt_shape_source], edges
    )
    electron_hist, electron_var = histogram(table.sieie[electron], table.weight[electron], edges)
    electron_hist *= float(electron_scale)
    electron_var *= float(electron_scale) * float(electron_scale)
    fit = extended_template_fit(
        observation,
        prompt_shape,
        np.asarray(template["shape"], dtype=float),
        electron_hist,
    )
    fit.update(
        {
            "stage": stage,
            "data_events": int(np.count_nonzero(data)),
            "prompt_expected_before_fit": float(np.sum(table.weight[prompt])),
            "electron_expected": float(np.sum(electron_hist)),
            "electron_scale": float(electron_scale),
            "stage_specific_prompt_shape": bool(stage_specific_prompt_shape),
            "effective_prompt_template_events": effective_events(prompt_shape, prompt_shape_var),
            "effective_electron_events": effective_events(electron_hist, electron_var),
            "truth_fake_yield": float(np.sum(table.weight[selected & ~table.is_data & (table.origin == "fake")])),
            "truth_fake_tight_shape_yield": float(
                np.sum(table.weight[selected & ~table.is_data & (table.origin == "fake") & (table.shape_level >= 2)])
            ),
        }
    )
    return fit


def mc_fake_template(table: EventTable, region: str, group: Group) -> dict[str, Any]:
    edges = sieie_edges(group.eta)
    selected = (
        base_mask(table, region, group)
        & stage_mask(table, "fake_template")
        & ~table.is_data
        & (table.origin == "fake")
    )
    values, variances = histogram(table.sieie[selected], table.weight[selected], edges)
    total = float(np.sum(table.weight[selected]))
    passed = float(np.sum(table.weight[selected & (table.shape_level >= 2)]))
    return {
        "edges": edges.tolist(),
        "shape": normalized_shape(values, alpha=0.5).tolist(),
        "pass_fraction": min(1.0, max(0.0, passed / total)) if total > 0.0 else 0.0,
        "weighted_yield": total,
        "effective_events": effective_events(values, variances),
    }


def fit_stage_mc(
    table: EventTable,
    region: str,
    group: Group,
    stage: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    edges = np.asarray(template["edges"], dtype=float)
    selected = base_mask(table, region, group) & stage_mask(table, stage) & ~table.is_data
    observation, _ = histogram(table.sieie[selected], table.weight[selected], edges)
    prompt_source = base_mask(table, region, group) & ~table.is_data & (table.origin == "prompt")
    prompt_shape, prompt_variance = histogram(
        table.sieie[prompt_source], table.weight[prompt_source], edges
    )
    electron = selected & (table.origin == "electron")
    electron_hist, _ = histogram(table.sieie[electron], table.weight[electron], edges)
    fit = extended_template_fit(
        observation,
        prompt_shape,
        np.asarray(template["shape"], dtype=float),
        electron_hist,
    )
    truth_fake = selected & (table.origin == "fake")
    fit.update(
        {
            "stage": stage,
            "effective_prompt_template_events": effective_events(prompt_shape, prompt_variance),
            "truth_fake_yield": float(np.sum(table.weight[truth_fake])),
            "truth_fake_tight_shape_yield": float(
                np.sum(table.weight[truth_fake & (table.shape_level >= 2)])
            ),
        }
    )
    return fit


def ratio_record(numerator: float, numerator_variance: float, denominator: float, denominator_variance: float) -> dict[str, Any]:
    if (
        denominator <= 0.0
        or numerator < 0.0
        or not np.isfinite(numerator_variance)
        or not np.isfinite(denominator_variance)
    ):
        return {"value": None, "variance": None, "uncertainty": None, "valid": False}
    value = numerator / denominator
    variance = value * value * (
        (numerator_variance / (numerator * numerator) if numerator > 0.0 else 0.0)
        + denominator_variance / (denominator * denominator)
    )
    return {
        "value": float(value),
        "variance": float(max(0.0, variance)),
        "uncertainty": float(math.sqrt(max(0.0, variance))),
        "valid": bool(np.isfinite(value) and np.isfinite(variance)),
    }


def fitted_factor_record(
    template: dict[str, Any],
    pass_fit: dict[str, Any],
    loose_fit: dict[str, Any],
) -> dict[str, Any]:
    fraction = float(template["pass_fraction"])
    pass_cov = np.asarray(pass_fit["covariance"], dtype=float)
    loose_cov = np.asarray(loose_fit["covariance"], dtype=float)
    pass_fake = float(pass_fit["fake_yield"]) * fraction
    loose_fake = float(loose_fit["fake_yield"]) * fraction
    pass_variance = float(pass_cov[1, 1]) * fraction * fraction if pass_cov.shape == (2, 2) else float("nan")
    loose_variance = float(loose_cov[1, 1]) * fraction * fraction if loose_cov.shape == (2, 2) else float("nan")
    factor = ratio_record(pass_fake, pass_variance, loose_fake, loose_variance)
    return {
        "tight_fake_yield": pass_fake,
        "tight_fake_variance": pass_variance,
        "loose_tight_shape_fake_yield": loose_fake,
        "loose_tight_shape_fake_variance": loose_variance,
        "fake_factor": factor,
    }


def _factor_variation(
    table: EventTable,
    group: Group,
    region: str,
    *,
    partition: str = "all",
    electron_scale: float = 1.0,
    prompt_scale: float = 1.0,
    stage_specific_prompt_shape: bool = False,
) -> dict[str, Any]:
    template = fake_template(
        table,
        region,
        group,
        partition=partition,
        electron_scale=electron_scale,
        prompt_scale=prompt_scale,
    )
    pass_fit = fit_stage(
        table,
        region,
        group,
        "pass_charged_iso",
        template,
        electron_scale=electron_scale,
        stage_specific_prompt_shape=stage_specific_prompt_shape,
    )
    loose_fit = fit_stage(
        table,
        region,
        group,
        "loose_charged_iso",
        template,
        electron_scale=electron_scale,
        stage_specific_prompt_shape=stage_specific_prompt_shape,
    )
    return {
        "fake_template": template,
        "pass_fit": pass_fit,
        "loose_fit": loose_fit,
        **fitted_factor_record(template, pass_fit, loose_fit),
    }


def measure_group(table: EventTable, group: Group, region: str) -> dict[str, Any]:
    nominal = _factor_variation(table, group, region)
    template = nominal["fake_template"]
    pass_fit = nominal["pass_fit"]
    loose_fit = nominal["loose_fit"]
    factor = nominal["fake_factor"]
    variations = {
        "fake_template_low_charged_iso": _factor_variation(
            table, group, region, partition="low_charged_iso"
        ),
        "fake_template_high_charged_iso": _factor_variation(
            table, group, region, partition="high_charged_iso"
        ),
        "electron_normalization_up": _factor_variation(
            table, group, region, electron_scale=1.5
        ),
        "electron_normalization_down": _factor_variation(
            table, group, region, electron_scale=0.5
        ),
        "prompt_contamination_up": _factor_variation(
            table, group, region, prompt_scale=1.3
        ),
        "prompt_contamination_down": _factor_variation(
            table, group, region, prompt_scale=0.7
        ),
        "stage_specific_prompt_shape": _factor_variation(
            table, group, region, stage_specific_prompt_shape=True
        ),
    }
    nominal_value = factor.get("value") if factor["valid"] else None
    components: dict[str, float] = {}
    if nominal_value is not None:
        for component, names in {
            "fake_template_sideband": (
                "fake_template_low_charged_iso",
                "fake_template_high_charged_iso",
            ),
            "electron_normalization": (
                "electron_normalization_up",
                "electron_normalization_down",
            ),
            "prompt_contamination": (
                "prompt_contamination_up",
                "prompt_contamination_down",
            ),
            "prompt_template_shape": ("stage_specific_prompt_shape",),
        }.items():
            shifts = []
            for name in names:
                varied = variations[name]["fake_factor"]
                if varied["valid"]:
                    shifts.append(abs(float(varied["value"]) - float(nominal_value)))
            components[component] = max(shifts) if shifts else 0.0
        systematic_variance = float(sum(value * value for value in components.values()))
        statistical_variance = float(factor["variance"])
        factor.update(
            {
                "statistical_uncertainty": math.sqrt(max(0.0, statistical_variance)),
                "systematic_components": components,
                "systematic_uncertainty": math.sqrt(max(0.0, systematic_variance)),
                "total_variance": statistical_variance + systematic_variance,
                "total_uncertainty": math.sqrt(max(0.0, statistical_variance + systematic_variance)),
            }
        )
    statistically_usable = bool(
        template["data_events"] >= MIN_FAKE_TEMPLATE_EVENTS
        and pass_fit["data_events"] >= MIN_DATA_FIT_EVENTS
        and loose_fit["data_events"] >= MIN_DATA_FIT_EVENTS
        and pass_fit["effective_prompt_template_events"] >= MIN_MC_TEMPLATE_EFFECTIVE_EVENTS
        and pass_fit["success"]
        and loose_fit["success"]
        and factor["valid"]
    )
    return {
        "group": group.as_dict(),
        "region": region,
        "fake_template": template,
        "pass_fit": pass_fit,
        "loose_fit": loose_fit,
        "tight_fake_yield": nominal["tight_fake_yield"],
        "tight_fake_variance": nominal["tight_fake_variance"],
        "loose_tight_shape_fake_yield": nominal["loose_tight_shape_fake_yield"],
        "loose_tight_shape_fake_variance": nominal["loose_tight_shape_fake_variance"],
        "fake_factor": factor,
        "fit_variations": variations,
        "statistically_usable": statistically_usable,
    }


def measure_group_mc(table: EventTable, group: Group, region: str) -> dict[str, Any]:
    template = mc_fake_template(table, region, group)
    pass_fit = fit_stage_mc(table, region, group, "pass_charged_iso", template)
    loose_fit = fit_stage_mc(table, region, group, "loose_charged_iso", template)
    fraction = float(template["pass_fraction"])
    pass_covariance = np.asarray(pass_fit["covariance"], dtype=float)
    loose_covariance = np.asarray(loose_fit["covariance"], dtype=float)
    fitted_pass = float(pass_fit["fake_yield"]) * fraction
    fitted_loose = float(loose_fit["fake_yield"]) * fraction
    fitted_factor = ratio_record(
        fitted_pass,
        float(pass_covariance[1, 1]) * fraction * fraction,
        fitted_loose,
        float(loose_covariance[1, 1]) * fraction * fraction,
    )
    truth_factor = ratio_record(
        float(pass_fit["truth_fake_tight_shape_yield"]),
        0.0,
        float(loose_fit["truth_fake_tight_shape_yield"]),
        0.0,
    )
    fitted_over_truth = None
    if fitted_factor["valid"] and truth_factor["valid"] and float(truth_factor["value"]) > 0.0:
        fitted_over_truth = float(fitted_factor["value"]) / float(truth_factor["value"])
    return {
        "group": group.as_dict(),
        "region": region,
        "fake_template": template,
        "pass_fit": pass_fit,
        "loose_fit": loose_fit,
        "fake_factor": fitted_factor,
        "truth_fake_factor": truth_factor,
        "fitted_over_truth_factor": fitted_over_truth,
    }


def finest_group_map(measurements: dict[str, dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    by_name = {group.name: group for group in groups()}
    for eta in ("EB", "EE"):
        for low, high in zip(FINE_PT_EDGES[:-1], FINE_PT_EDGES[1:]):
            fine_name = f"{eta}_pt{low:g}to{'inf' if high >= 1_000_000 else f'{high:g}'}"
            coarse_name = f"{eta}_pt220to400" if high <= 400.0 else f"{eta}_pt400toinf"
            candidates = [fine_name, coarse_name, f"{eta}_inclusive"]
            chosen = next((name for name in candidates if measurements[name]["statistically_usable"]), candidates[-1])
            if chosen not in by_name:
                raise RuntimeError(f"unknown fallback group {chosen}")
            mapping[fine_name] = chosen
    return mapping


def used_group_for_event(table: EventTable, index: int, mapping: dict[str, str]) -> str:
    eta = "EB" if abs(float(table.eta[index])) < 1.4442 else "EE"
    pt = float(table.pt[index])
    position = int(np.searchsorted(np.asarray(FINE_PT_EDGES), pt, side="right") - 1)
    position = max(0, min(position, len(FINE_PT_EDGES) - 2))
    low = FINE_PT_EDGES[position]
    high = FINE_PT_EDGES[position + 1]
    name = f"{eta}_pt{low:g}to{'inf' if high >= 1_000_000 else f'{high:g}'}"
    return mapping[name]


def weighted_application_histogram(
    table: EventTable,
    region: str,
    measurements: dict[str, dict[str, Any]],
    mapping: dict[str, str],
    edges: np.ndarray = UT_EDGES,
) -> dict[str, Any]:
    prediction = np.zeros(len(edges) - 1, dtype=float)
    statistical_variance = np.zeros(len(edges) - 1, dtype=float)
    data_loose = np.zeros(len(edges) - 1, dtype=float)
    contamination = np.zeros(len(edges) - 1, dtype=float)
    signed_by_group: dict[str, np.ndarray] = {
        name: np.zeros(len(edges) - 1, dtype=float)
        for name in sorted(set(mapping.values()))
    }
    base = table.region_masks[region] & (table.shape_level >= 2) & (table.charged_level == 1)
    for index in np.flatnonzero(base):
        used = used_group_for_event(table, int(index), mapping)
        factor = measurements[used]["fake_factor"]
        if not factor["valid"]:
            continue
        value = float(factor["value"])
        bin_index = int(np.searchsorted(edges, table.ut[index], side="right") - 1)
        if bin_index < 0 or bin_index >= len(prediction):
            continue
        if table.is_data[index]:
            signed_weight = 1.0
            data_loose[bin_index] += 1.0
        elif table.origin[index] in ("prompt", "electron"):
            signed_weight = -float(table.weight[index])
            contamination[bin_index] += float(table.weight[index])
        else:
            continue
        prediction[bin_index] += signed_weight * value
        statistical_variance[bin_index] += signed_weight * signed_weight * value * value
        signed_by_group[used][bin_index] += signed_weight
    factor_variance = np.zeros(len(edges) - 1, dtype=float)
    for name, signed in signed_by_group.items():
        factor = measurements[name]["fake_factor"]
        if factor["valid"]:
            factor_variance += signed * signed * float(
                factor.get("total_variance", factor["variance"])
            )
    total_variance = statistical_variance + factor_variance
    return {
        "edges": edges.tolist(),
        "prediction": prediction.tolist(),
        "statistical_variance": statistical_variance.tolist(),
        "factor_variance": factor_variance.tolist(),
        "total_variance": total_variance.tolist(),
        "data_loose": data_loose.tolist(),
        "prompt_electron_contamination": contamination.tolist(),
    }


def direct_fit_histogram(
    table: EventTable,
    region: str,
    measurements: dict[str, dict[str, Any]],
    mapping: dict[str, str],
    edges: np.ndarray = UT_EDGES,
) -> dict[str, Any]:
    yields = np.zeros(len(edges) - 1, dtype=float)
    variances = np.zeros(len(edges) - 1, dtype=float)
    details: list[dict[str, Any]] = []
    lookup = {group.name: group for group in groups()}
    fine_groups = {group.name: group for group in groups() if group.tier == "fine"}
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        ut_mask = (table.ut >= low) & (table.ut < high)
        bin_details = []
        for fine_name, used_name in sorted(mapping.items()):
            group = fine_groups[fine_name]
            template = measurements[used_name]["fake_template"]
            fit = fit_stage(table, region, group, "pass_charged_iso", template, extra_mask=ut_mask)
            fraction = float(template["pass_fraction"])
            covariance = np.asarray(fit["covariance"], dtype=float)
            value = float(fit["fake_yield"]) * fraction
            variance = float(covariance[1, 1]) * fraction * fraction if covariance.shape == (2, 2) else float("nan")
            if np.isfinite(value):
                yields[bin_index] += value
            if np.isfinite(variance):
                variances[bin_index] += max(0.0, variance)
            bin_details.append(
                {
                    "fine_group": fine_name,
                    "template_group": used_name,
                    "fit": fit,
                    "tight_fake_yield": value,
                    "variance": variance,
                }
            )
        details.append({"ut_low": float(low), "ut_high": float(high), "groups": bin_details})
    return {"edges": edges.tolist(), "yield": yields.tolist(), "variance": variances.tolist(), "details": details}


def truth_closure(
    table: EventTable,
    region: str,
    factor_measurements: dict[str, dict[str, Any]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    truth = np.zeros(len(UT_EDGES) - 1, dtype=float)
    prediction = np.zeros(len(UT_EDGES) - 1, dtype=float)
    truth_var = np.zeros(len(UT_EDGES) - 1, dtype=float)
    prediction_stat_var = np.zeros(len(UT_EDGES) - 1, dtype=float)
    loose_by_group: dict[str, np.ndarray] = {
        name: np.zeros(len(UT_EDGES) - 1, dtype=float)
        for name in sorted(set(mapping.values()))
    }
    mc = table.region_masks[region] & ~table.is_data & (table.origin == "fake")
    for index in np.flatnonzero(mc):
        bin_index = int(np.searchsorted(UT_EDGES, table.ut[index], side="right") - 1)
        if bin_index < 0 or bin_index >= len(truth):
            continue
        weight = float(table.weight[index])
        if table.shape_level[index] >= 2 and table.charged_level[index] >= 2:
            truth[bin_index] += weight
            truth_var[bin_index] += weight * weight
        if table.shape_level[index] >= 2 and table.charged_level[index] == 1:
            used = used_group_for_event(table, int(index), mapping)
            factor = factor_measurements[used]["fake_factor"]
            if factor["valid"]:
                predicted_weight = weight * float(factor["value"])
                prediction[bin_index] += predicted_weight
                prediction_stat_var[bin_index] += predicted_weight * predicted_weight
                loose_by_group[used][bin_index] += weight
    prediction_factor_var = np.zeros(len(UT_EDGES) - 1, dtype=float)
    for name, loose in loose_by_group.items():
        factor = factor_measurements[name]["fake_factor"]
        if factor["valid"]:
            prediction_factor_var += loose * loose * float(factor["variance"])
    prediction_var = prediction_stat_var + prediction_factor_var
    total_truth = float(np.sum(truth))
    total_prediction = float(np.sum(prediction))
    return {
        "edges": UT_EDGES.tolist(),
        "truth": truth.tolist(),
        "truth_variance": truth_var.tolist(),
        "prediction": prediction.tolist(),
        "prediction_variance": prediction_var.tolist(),
        "prediction_statistical_variance": prediction_stat_var.tolist(),
        "prediction_factor_variance": prediction_factor_var.tolist(),
        "prediction_over_truth": total_prediction / total_truth if total_truth > 0.0 else None,
        "total_truth": total_truth,
        "total_prediction": total_prediction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    events, input_audit = load_inputs(sorted(set(paths)))
    events, dedup_audit = deduplicate_data(events)
    normalization = read_payload(args.normalization)
    if normalization.get("status") != "complete":
        raise RuntimeError(f"normalization is not complete: {normalization.get('status')}")
    table = EventTable(events, normalization)

    measurement_groups = {group.name: measure_group(table, group, MEASUREMENT_REGION) for group in groups()}
    simulation_measurement_groups = {
        group.name: measure_group_mc(table, group, MEASUREMENT_REGION)
        for group in groups()
    }
    mapping = finest_group_map(measurement_groups)
    validation_direct = direct_fit_histogram(table, VALIDATION_REGION, measurement_groups, mapping)
    validation_prediction = weighted_application_histogram(table, VALIDATION_REGION, measurement_groups, mapping)
    application_direct = direct_fit_histogram(table, APPLICATION_REGION, measurement_groups, mapping)
    application_prediction = weighted_application_histogram(table, APPLICATION_REGION, measurement_groups, mapping)
    simulation_closure = truth_closure(
        table,
        VALIDATION_REGION,
        simulation_measurement_groups,
        mapping,
    )
    simulation_ratio = (
        float(np.sum(simulation_closure["prediction"]))
        / float(np.sum(simulation_closure["truth"]))
        if float(np.sum(simulation_closure["truth"])) > 0.0
        else None
    )
    validation_ratio = (
        float(np.sum(validation_prediction["prediction"]))
        / float(np.sum(validation_direct["yield"]))
        if float(np.sum(validation_direct["yield"])) > 0.0
        else None
    )
    closure_shifts = [
        abs(value - 1.0)
        for value in (simulation_ratio, validation_ratio)
        if value is not None and np.isfinite(value)
    ]
    global_closure_uncertainty = max(closure_shifts) if closure_shifts else 1.0
    application_central = np.asarray(application_prediction["prediction"], dtype=float)
    application_method_variance = (
        application_central * global_closure_uncertainty
    ) ** 2
    application_prediction["closure_method_variance"] = application_method_variance.tolist()
    application_prediction["total_variance_with_closure"] = (
        np.asarray(application_prediction["total_variance"], dtype=float)
        + application_method_variance
    ).tolist()

    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "complete",
        "method": {
            "measurement_region": MEASUREMENT_REGION,
            "validation_region": VALIDATION_REGION,
            "application_region": APPLICATION_REGION,
            "fit_observable": "Photon_sieie",
            "prompt_template": "prompt-photon MC",
            "fake_template": "charged-isolation fail-loose data after prompt/electron subtraction",
            "electron_component": "fixed normalized electron-origin MC",
            "factor": "fitted tight-shape fake yield at charged-iso pass divided by fitted tight-shape fake yield at charged-iso loose-not-medium",
            "selection_source": "real_subset_worker.py",
            "nominal_intermediate_mutation": False,
        },
        "normalization": {
            "source": str(args.normalization),
            "luminosity_pb": normalization.get("luminosity_pb"),
            "policy": "nominal_weight_without_photon_id_sf * physical-dataset normalization_factor exactly once",
        },
        "input_audit": input_audit,
        "data_deduplication": dedup_audit,
        "event_count_after_deduplication": table.n,
        "groups": measurement_groups,
        "simulation_template_fit_groups": simulation_measurement_groups,
        "fine_to_used_group": mapping,
        "simulation_validation_region_closure": simulation_closure,
        "data_validation_region": {
            "direct_template_fit": validation_direct,
            "loose_prediction": validation_prediction,
        },
        "nominal_gcr_validation": {
            "direct_template_fit": application_direct,
            "loose_prediction": application_prediction,
            "note": "nominal GCR target data are validation only and do not enter the factor measurement",
        },
        "uncertainty_summary": {
            "per_factor_components": [
                "fit statistical",
                "charged-isolation fake-template sideband",
                "electron normalization (plus/minus 50 percent)",
                "prompt contamination normalization (plus/minus 30 percent)",
                "prompt-template charged-isolation dependence",
            ],
            "simulation_validation_prediction_over_truth": simulation_ratio,
            "data_validation_prediction_over_direct_fit": validation_ratio,
            "global_closure_relative_uncertainty": global_closure_uncertainty,
            "global_closure_policy": (
                "maximum absolute integral nonclosure in simulation and the "
                "independent data validation region"
            ),
        },
        "adoption_policy": {
            "required": [
                "fit convergence and statistically usable measurement groups",
                "simulation closure compatible with unity within assigned uncertainty",
                "independent data validation-region closure compatible with unity",
                "no material degradation of nominal-GCR UT shape",
            ],
            "automatic_adoption": False,
        },
    }
    write_payload(args.output, payload)
    print(json.dumps({"status": "complete", "output": str(args.output), "events": table.n}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
