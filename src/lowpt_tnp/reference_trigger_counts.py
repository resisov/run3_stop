"""Shared counting and reduction utilities for reference-trigger measurements."""
from __future__ import annotations

import math
import json
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import correctionlib
import numpy as np
from scipy.special import betaincinv


CONFIDENCE = 0.682689492
COUNT_KEYS = ("total", "passed", "sumw_total", "sumw_passed", "sumw2_total", "sumw2_passed")


def mc_normalization_factors(input_dir: Path) -> tuple[dict[int, float], dict[str, Any]]:
    """Aggregate file-level Runs.genEventSumw and return xsec/sumw by dataset id.

    The flat worker stores a complete per-file Runs denominator even for a
    limited event pilot.  Files are deduplicated by their source path so retry
    outputs cannot silently double-count the normalization denominator.
    """
    dataset_to_physical: dict[int, str] = {}
    dataset_name_to_physical: dict[str, str] = {}
    xsecs: dict[str, set[float]] = {}
    file_sumw: dict[str, dict[str, float]] = {}
    duplicate_files: list[dict[str, str]] = []
    sidecars = sorted(input_dir.glob("*.json"))
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text())
        for item in payload.get("datasets", {}).values():
            if item.get("is_data") or item.get("is_signal"):
                continue
            physical = str(item.get("physical_dataset") or item.get("dataset") or "")
            dataset = str(item.get("dataset") or "")
            dataset_id = int(item["dataset_id"])
            dataset_to_physical[dataset_id] = physical
            dataset_name_to_physical[dataset] = physical
            xsec = item.get("xsec_pb")
            if xsec is not None:
                xsecs.setdefault(physical, set()).add(float(xsec))
        for item in payload.get("files", []):
            if item.get("read_status") != "success":
                continue
            dataset = str(item.get("dataset") or "")
            physical = dataset_name_to_physical.get(dataset)
            if not physical:
                continue
            path = str(item.get("file_path") or item.get("effective_file_path") or "")
            value = float(item.get("sumw") or 0.0)
            known = file_sumw.setdefault(physical, {})
            if path in known:
                duplicate_files.append({"physical_dataset": physical, "file_path": path, "sidecar": str(sidecar)})
                continue
            known[path] = value
    if duplicate_files:
        raise RuntimeError(f"duplicate successful MC file coverage in trigger campaign: {duplicate_files[:5]}")
    physical_factors: dict[str, float] = {}
    groups: dict[str, Any] = {}
    for physical in sorted(set(dataset_to_physical.values())):
        values = xsecs.get(physical, set())
        if len(values) != 1:
            raise RuntimeError(f"missing or conflicting xsec for {physical}: {sorted(values)}")
        denominator = float(sum(file_sumw.get(physical, {}).values()))
        if not math.isfinite(denominator) or denominator == 0.0:
            raise RuntimeError(f"invalid aggregate generator sumw for {physical}: {denominator}")
        xsec = next(iter(values))
        physical_factors[physical] = xsec / denominator
        groups[physical] = {
            "xsec_pb": xsec,
            "sumw": denominator,
            "files": len(file_sumw.get(physical, {})),
            "factor_without_luminosity": physical_factors[physical],
        }
    factors = {
        dataset_id: physical_factors[physical]
        for dataset_id, physical in dataset_to_physical.items()
    }
    return factors, {"sidecars_scanned": len(sidecars), "physical_datasets": groups}


PILEUP_SOURCES = {
    "2024": (
        Path("data/pileup/puWeights_2024.json.gz"),
        "Collisions24_BCDEFGHI_goldenJSON",
    ),
    "2025": (
        Path("data/pileup/puWeights_2025.json.gz"),
        "Collisions25_goldenJSON",
    ),
}


def pileup_source(year: str) -> str:
    normalized = str(year)
    if normalized not in PILEUP_SOURCES:
        raise ValueError(f"unsupported pileup year for SF measurement: {year}")
    path, correction = PILEUP_SOURCES[normalized]
    return f"{path}::{correction}"


@lru_cache(maxsize=None)
def _pileup_correction(repo: str, year: str) -> Any:
    normalized = str(year)
    if normalized not in PILEUP_SOURCES:
        raise ValueError(f"unsupported pileup year for SF measurement: {year}")
    relative_path, correction_name = PILEUP_SOURCES[normalized]
    path = Path(repo) / relative_path
    return correctionlib.CorrectionSet.from_file(str(path))[correction_name]


def pileup_weight_triplet(
    repo: Path,
    ntrue: np.ndarray,
    *,
    year: str = "2024",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correction = _pileup_correction(str(repo.resolve()), str(year))
    values = np.asarray(ntrue, dtype=float)
    return tuple(
        np.asarray(correction.evaluate(values, variation), dtype=float)
        for variation in ("nominal", "up", "down")
    )


def add_pileup_uncertainty(
    nominal: list[dict[str, Any]],
    up: list[dict[str, Any]],
    down: list[dict[str, Any]],
) -> None:
    if not (len(nominal) == len(up) == len(down)):
        raise ValueError("nominal/up/down trigger bins have inconsistent lengths")
    for item, item_up, item_down in zip(nominal, up, down):
        stat = float(item["scale_factor_uncertainty"]) if item.get("scale_factor_uncertainty") is not None else math.nan
        values = [item.get("scale_factor"), item_up.get("scale_factor"), item_down.get("scale_factor")]
        if all(value is not None and math.isfinite(float(value)) for value in values):
            systematic = max(abs(float(values[1]) - float(values[0])), abs(float(values[2]) - float(values[0])))
        else:
            systematic = math.nan
        item["scale_factor_stat_uncertainty"] = stat
        item["scale_factor_pileup_uncertainty"] = systematic
        item["scale_factor_uncertainty"] = math.hypot(stat, systematic) if math.isfinite(stat) and math.isfinite(systematic) else math.nan
        item["pileup_variation_scale_factors"] = {"up": item_up.get("scale_factor"), "down": item_down.get("scale_factor")}
        item["valid"] = bool(item.get("valid") and item_up.get("valid") and item_down.get("valid") and math.isfinite(item["scale_factor_uncertainty"]))


def empty_counts(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {key: np.zeros(shape, dtype=float) for key in COUNT_KEYS}


def fill_counts(
    counts: dict[str, np.ndarray],
    coordinates: list[np.ndarray],
    edges: list[np.ndarray],
    passed: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not coordinates:
        raise ValueError("at least one coordinate is required")
    n = len(coordinates[0])
    if any(len(values) != n for values in coordinates) or len(passed) != n or len(weights) != n:
        raise ValueError("coordinate, passed, and weight arrays must have equal length")
    samples = np.column_stack(coordinates)
    selected = np.asarray(passed, dtype=bool)
    values = np.asarray(weights, dtype=float)
    counts["total"] += np.histogramdd(samples, bins=edges)[0]
    counts["passed"] += np.histogramdd(samples[selected], bins=edges)[0]
    counts["sumw_total"] += np.histogramdd(samples, bins=edges, weights=values)[0]
    counts["sumw_passed"] += np.histogramdd(samples[selected], bins=edges, weights=values[selected])[0]
    counts["sumw2_total"] += np.histogramdd(samples, bins=edges, weights=values * values)[0]
    counts["sumw2_passed"] += np.histogramdd(samples[selected], bins=edges, weights=values[selected] ** 2)[0]


def serialise_counts(counts: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: [float(value) for value in np.asarray(counts[key]).reshape(-1)] for key in COUNT_KEYS}


def _clopper_pearson(passed: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0 or passed < 0 or passed > total:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    low = 0.0 if passed == 0 else float(betaincinv(passed, total - passed + 1, alpha / 2))
    high = 1.0 if passed == total else float(betaincinv(passed + 1, total - passed, 1 - alpha / 2))
    return low, high


def _wilson(efficiency: float, entries: float, confidence: float) -> tuple[float, float]:
    if entries <= 0 or not 0.0 <= efficiency <= 1.0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / entries
    centre = (efficiency + z * z / (2 * entries)) / denominator
    half = z * math.sqrt(efficiency * (1 - efficiency) / entries + z * z / (4 * entries**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def reduce_counts(
    data: dict[str, Iterable[float]],
    mc: dict[str, Iterable[float]],
    *,
    confidence: float = CONFIDENCE,
) -> list[dict[str, Any]]:
    arrays = {
        f"data_{key}": np.asarray(data[key], dtype=float) for key in COUNT_KEYS
    }
    arrays.update({
        f"mc_{key}": np.asarray(mc[key], dtype=float) for key in COUNT_KEYS
    })
    sizes = {len(values) for values in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("all flattened count arrays must have equal length")
    result = []
    for index in range(next(iter(sizes))):
        dt = int(round(arrays["data_total"][index]))
        dp = int(round(arrays["data_passed"][index]))
        de = dp / dt if dt > 0 else math.nan
        dl, dh = _clopper_pearson(dp, dt, confidence)
        mt = float(arrays["mc_sumw_total"][index])
        mp = float(arrays["mc_sumw_passed"][index])
        mt2 = float(arrays["mc_sumw2_total"][index])
        mc_valid = mt > 0 and mt2 > 0 and 0 <= mp <= mt
        me = mp / mt if mc_valid else math.nan
        neff = mt * mt / mt2 if mc_valid else 0.0
        ml, mh = _wilson(me, neff, confidence)
        valid = dt > 0 and mc_valid and me > 0 and de > 0
        sf = de / me if valid else math.nan
        if valid:
            data_unc = max(de - dl, dh - de)
            mc_unc = max(me - ml, mh - me)
            sf_unc = sf * math.sqrt((data_unc / de) ** 2 + (mc_unc / me) ** 2)
        else:
            sf_unc = math.nan
        result.append({
            "flat_index": index,
            "data_total": dt,
            "data_passed": dp,
            "data_efficiency": de,
            "data_interval": [dl, dh],
            "mc_sumw_total": mt,
            "mc_sumw_passed": mp,
            "mc_sumw2_total": mt2,
            "mc_effective_entries": neff,
            "mc_efficiency": me,
            "mc_interval": [ml, mh],
            "scale_factor": sf,
            "scale_factor_uncertainty": sf_unc,
            "valid": valid,
        })
    return result


def json_safe(value: Any) -> Any:
    """Replace non-finite diagnostics with null for strict machine JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value
