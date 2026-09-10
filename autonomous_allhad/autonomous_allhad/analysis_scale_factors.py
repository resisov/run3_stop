"""Analysis-owned scale factors that are not supplied by the standard POG set.

Payloads are installed only after their measurement result is explicitly
adopted.  Missing payloads raise ``AnalysisScaleFactorUnavailable`` so callers
can label the missing correction instead of extrapolating a 10 GeV edge bin.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import awkward as ak
import correctionlib
import numpy as np


class AnalysisScaleFactorUnavailable(RuntimeError):
    """Raised when an analysis-owned correction has not been installed."""


PAYLOAD_FILENAMES = {
    "met_trigger": "met_trigger_sf.json.gz",
    "photon_trigger": "photon_trigger_sf.json.gz",
    "veto_electron_5to10": "veto_electron_5to10_sf.json.gz",
    "loose_muon_5to10": "loose_muon_5to10_sf.json.gz",
}

# Backward-compatible registry for callers that only need the component list.
# Evaluation itself resolves the data-taking year explicitly below.
PAYLOADS = {
    component: Path("analysis/data/AnalysisSF/2024") / filename
    for component, filename in PAYLOAD_FILENAMES.items()
}

# Production histogramming must fail closed if any adopted analysis-owned
# payload is absent.  Keep this contract next to the payload registry so every
# entry point uses the same list.
REQUIRED_ANALYSIS_SF_COMPONENTS = tuple(PAYLOADS)
DEFAULT_ANALYSIS_SF_COMPONENTS = ("met_trigger", "photon_trigger")
REQUIRED_ANALYSIS_SF_VARIATIONS = tuple(
    f"{component}{direction}"
    for component in REQUIRED_ANALYSIS_SF_COMPONENTS
    for direction in ("Up", "Down")
)


TOPW_CORRECTION_BRANCHES = (
    "run", "luminosityBlock", "event", "entry", "file_id",
    "fatjet_source_index_all", "fatjet_decay_flavor_all",
)


def topw_fit_categories(flavor: Any, *, top_like: bool) -> np.ndarray:
    """Map decay containment to JME fit processes for an explicit sample group."""
    flavor = np.asarray(flavor)
    if (flavor.ndim != 1 or flavor.dtype.kind not in "iu"
        or np.any((flavor < 0) | (flavor > 4))):
        raise ValueError("Top/W decay flavors must be a flat integer array in [0, 4]")
    if not isinstance(top_like, bool):
        raise ValueError("Top/W sample-group membership must be explicit")
    if not top_like:
        return np.full(flavor.shape, "other", dtype="U5")
    return np.asarray(["tp1", "tp1", "tp1", "tp2", "tp3"])[flavor]


def topw_file_input_policy(root_file: Any) -> dict[str, Any]:
    """Missing correction inputs do not remove an otherwise valid event file."""
    events = root_file["Events"]
    if "TopWTruth" not in root_file:
        missing = ["TopWTruth"]
    else:
        truth = root_file["TopWTruth"]
        missing = [
            "TopWTruth/" + name for name in TOPW_CORRECTION_BRANCHES
            if name not in truth.keys()
        ]
    if missing:
        return {
            "mode": "unity_missing_branches", "missing": missing,
            "sf": 1.0, "uncertainty": 0.0,
        }
    if truth.num_entries != events.num_entries:
        raise RuntimeError("TopWTruth/Events entry mismatch")
    marker = json.loads(str(root_file["TopWTruth_metadata"]))
    if (marker.get("status") != "complete"
        or marker.get("schema_version") != "topw_truth_v1"
        or marker.get("events_entries") != events.num_entries):
        raise RuntimeError("invalid TopWTruth completion metadata")
    return {"mode": "available"}


def topw_file_sf_triplet(
    n: int, policy: dict[str, Any], evaluate: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if policy.get("mode") == "unity_missing_branches" and policy.get("missing"):
        return tuple(np.ones(n, dtype=float) for _ in range(3))
    if policy.get("mode") != "available":
        raise ValueError("unknown Top/W correction input policy")
    values = tuple(np.asarray(value, dtype=float) for value in evaluate())
    if len(values) != 3 or any(value.shape != (n,) for value in values):
        raise ValueError("Top/W event SF triplet has an invalid shape")
    if any(not np.all(np.isfinite(value)) or np.any(value < 0) for value in values):
        raise AnalysisScaleFactorUnavailable("invalid Top/W event SF")
    return values


def apply_topw_missing_input_fallback(
    variations: dict[str, Any], status: dict[str, Any],
    policy: dict[str, Any] | None, n: int,
) -> dict[str, Any]:
    if not policy or policy.get("mode") != "unity_missing_branches":
        return variations
    nominal, _up, _down = topw_file_sf_triplet(n, policy, None)
    status["topw_correction"] = dict(policy)
    return {
        name: np.asarray(weight, dtype=float) * nominal
        for name, weight in variations.items()
    }


def apply_topw_highpt_extrapolation(
    pt: Any,
    nominal: Any,
    up: Any,
    down: Any,
    *,
    tagger: str,
    year: str,
    fit_failed: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use unity for high-pT extrapolation and explicitly failed fit bins."""
    if str(year) not in {"2024", "2025"}:
        raise AnalysisScaleFactorUnavailable(f"unsupported Top/W SF year: {year}")
    limits = {"top": 1200.0, "w": 800.0}
    if tagger not in limits:
        raise ValueError(f"unsupported Top/W tagger: {tagger}")
    pt = np.asarray(pt, dtype=float)
    if pt.ndim != 1 or not np.all(np.isfinite(pt)) or np.any(pt < 0):
        raise ValueError("Top/W SF pT must be a finite nonnegative flat array")
    highpt = pt >= limits[tagger]
    failed = np.zeros(pt.shape, dtype=bool)
    if fit_failed is not None:
        failed = np.asarray(fit_failed)
        if failed.shape != pt.shape or failed.dtype != np.bool_:
            raise ValueError("Top/W fit-failure mask must be a matching boolean array")
    unity = highpt | failed
    result = []
    for values in (nominal, up, down):
        values = np.asarray(values, dtype=float)
        if values.shape != pt.shape:
            raise ValueError("Top/W SF and pT array shapes differ")
        values = np.where(unity, 1.0, values)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise AnalysisScaleFactorUnavailable("invalid in-range Top/W SF")
        result.append(values)
    nominal, up, down = result
    if np.any(down > nominal) or np.any(nominal > up):
        raise AnalysisScaleFactorUnavailable("Top/W SF variations do not bracket nominal")
    return nominal, up, down


def topw_pass_fail_triplet(
    tagged: Any,
    efficiency: Any,
    nominal: Any,
    up: Any,
    down: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use min(SF * efficiency, 1) for each nominal/Up/Down probability."""
    tagged = np.asarray(tagged)
    efficiency = np.asarray(efficiency, dtype=float)
    if tagged.ndim != 1 or tagged.dtype != np.bool_ or efficiency.shape != tagged.shape:
        raise ValueError("Top/W tag decisions and efficiencies must be aligned flat arrays")
    if (not np.all(np.isfinite(efficiency))
        or np.any((efficiency < 0) | (efficiency > 1))):
        raise AnalysisScaleFactorUnavailable("invalid Top/W MC efficiency")
    if np.any(tagged & (efficiency == 0)) or np.any(~tagged & (efficiency == 1)):
        raise AnalysisScaleFactorUnavailable("tag decision has no MC efficiency support")
    variations = [np.asarray(value, dtype=float) for value in (nominal, up, down)]
    for scale in variations:
        if scale.shape != tagged.shape:
            raise ValueError("Top/W SF and efficiency array shapes differ")
        if not np.all(np.isfinite(scale)) or np.any(scale < 0):
            raise AnalysisScaleFactorUnavailable("invalid Top/W SF variation")
        if np.any((efficiency == 1) & (scale < 1)):
            raise AnalysisScaleFactorUnavailable("no failing MC support for a reduced Top/W efficiency")
    if np.any(variations[2] > variations[0]) or np.any(variations[0] > variations[1]):
        raise AnalysisScaleFactorUnavailable("Top/W SF variations do not bracket nominal")
    result = []
    for scale in variations:
        corrected = np.minimum(scale * efficiency, 1.0)
        passed = np.ones_like(efficiency)
        fail = np.ones_like(efficiency)
        np.divide(corrected, efficiency, out=passed, where=efficiency > 0)
        np.divide(1 - corrected, 1 - efficiency, out=fail, where=efficiency < 1)
        result.append(np.where(tagged, passed, fail))
    return tuple(result)


@lru_cache(maxsize=None)
def _load(path: str) -> correctionlib.CorrectionSet:
    return correctionlib.CorrectionSet.from_file(path)


def payload_path(repo: Path, key: str, year: str) -> Path:
    if key not in PAYLOAD_FILENAMES:
        raise KeyError(f"unknown analysis SF component: {key}")
    if str(year) not in {"2024", "2025"}:
        raise AnalysisScaleFactorUnavailable(
            f"analysis SF component {key!r} has no payload campaign for year {year!r}"
        )
    return (
        repo
        / "analysis/data/AnalysisSF"
        / str(year)
        / PAYLOAD_FILENAMES[key]
    ).resolve()


def _payload(repo: Path, key: str, year: str) -> correctionlib.CorrectionSet:
    path = payload_path(repo, key, year)
    if not path.is_file():
        raise AnalysisScaleFactorUnavailable(f"analysis SF payload is not installed: {path}")
    try:
        return _load(str(path))
    except Exception as exc:
        raise AnalysisScaleFactorUnavailable(
            f"analysis SF payload is invalid: {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _flat_triplet(
    correction: Any,
    *coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(correction.evaluate(variation, *coordinates), dtype=float)
        for variation in ("nominal", "up", "down")
    )


def met_trigger_triplet(
    repo: Path,
    met: Any,
    *,
    qcd: bool,
    year: str = "2024",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the analysis-owned MET-trigger SF for one data-taking year.

    The adopted Run-3 campaign uses one single-electron-reference measurement
    for every MET-triggered MC process.  ``qcd`` is retained for call-site
    compatibility, but it must not request a nonexistent QCD-only payload and
    silently fall back to unity.
    """
    del qcd
    values = np.asarray(met, dtype=float)
    correction = _payload(repo, "met_trigger", year)["met_trigger_sf_genuine"]
    return _flat_triplet(correction, values)


def photon_trigger_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
    *,
    year: str = "2024",
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(
        _payload(repo, "photon_trigger", year)["photon_trigger_sf"],
        flat_abseta,
        flat_pt,
    )
    return tuple(ak.unflatten(values, counts) for values in result)


def veto_electron_lowpt_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
    *,
    year: str = "2024",
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(
        _payload(repo, "veto_electron_5to10", year)["veto_electron_id_5to10_sf"],
        flat_abseta,
        flat_pt,
    )
    return tuple(ak.unflatten(values, counts) for values in result)


def loose_muon_lowpt_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
    *,
    year: str = "2024",
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(
        _payload(repo, "loose_muon_5to10", year)["loose_muon_id_5to10_sf"],
        flat_abseta,
        flat_pt,
    )
    return tuple(ak.unflatten(values, counts) for values in result)
