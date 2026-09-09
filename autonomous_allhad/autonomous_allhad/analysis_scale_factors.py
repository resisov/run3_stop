"""Analysis-owned scale factors that are not supplied by the standard POG set.

Payloads are installed only after their measurement result is explicitly
adopted.  Missing payloads raise ``AnalysisScaleFactorUnavailable`` so callers
can label the missing correction instead of extrapolating a 10 GeV edge bin.
"""
from __future__ import annotations

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


def apply_topw_highpt_extrapolation(
    pt: Any,
    nominal: Any,
    up: Any,
    down: Any,
    *,
    tagger: str,
    year: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the adopted high-pT SF=1, uncertainty=0 prescription."""
    if str(year) not in {"2024", "2025"}:
        raise AnalysisScaleFactorUnavailable(f"unsupported Top/W SF year: {year}")
    limits = {"top": 1200.0, "w": 800.0}
    if tagger not in limits:
        raise ValueError(f"unsupported Top/W tagger: {tagger}")
    pt = np.asarray(pt, dtype=float)
    if pt.ndim != 1 or not np.all(np.isfinite(pt)) or np.any(pt < 0):
        raise ValueError("Top/W SF pT must be a finite nonnegative flat array")
    highpt = pt >= limits[tagger]
    result = []
    for values in (nominal, up, down):
        values = np.asarray(values, dtype=float)
        if values.shape != pt.shape:
            raise ValueError("Top/W SF and pT array shapes differ")
        values = np.where(highpt, 1.0, values)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise AnalysisScaleFactorUnavailable("invalid in-range Top/W SF")
        result.append(values)
    nominal, up, down = result
    if np.any(down > nominal) or np.any(nominal > up):
        raise AnalysisScaleFactorUnavailable("Top/W SF variations do not bracket nominal")
    return nominal, up, down


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
