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


PAYLOADS = {
    "met_trigger": Path("analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz"),
    "photon_trigger": Path("analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz"),
    "veto_electron_5to10": Path("analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz"),
    "loose_muon_5to10": Path("analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz"),
}

# Production histogramming must fail closed if any adopted analysis-owned
# payload is absent.  Keep this contract next to the payload registry so every
# entry point uses the same list.
REQUIRED_ANALYSIS_SF_COMPONENTS = tuple(PAYLOADS)
REQUIRED_ANALYSIS_SF_VARIATIONS = tuple(
    f"{component}{direction}"
    for component in REQUIRED_ANALYSIS_SF_COMPONENTS
    for direction in ("Up", "Down")
)


@lru_cache(maxsize=None)
def _load(path: str) -> correctionlib.CorrectionSet:
    return correctionlib.CorrectionSet.from_file(path)


def _payload(repo: Path, key: str) -> correctionlib.CorrectionSet:
    path = (repo / PAYLOADS[key]).resolve()
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the adopted 2024 MET-trigger SF.

    The adopted Run-3 campaign uses one single-electron-reference measurement
    for every MET-triggered MC process.  ``qcd`` is retained for call-site
    compatibility, but it must not request a nonexistent QCD-only payload and
    silently fall back to unity.
    """
    del qcd
    values = np.asarray(met, dtype=float)
    correction = _payload(repo, "met_trigger")["met_trigger_sf_genuine"]
    return _flat_triplet(correction, values)


def photon_trigger_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(_payload(repo, "photon_trigger")["photon_trigger_sf"], flat_abseta, flat_pt)
    return tuple(ak.unflatten(values, counts) for values in result)


def veto_electron_lowpt_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(
        _payload(repo, "veto_electron_5to10")["veto_electron_5to10_sf"],
        flat_abseta,
        flat_pt,
    )
    return tuple(ak.unflatten(values, counts) for values in result)


def loose_muon_lowpt_triplet(
    repo: Path,
    eta: Any,
    pt: Any,
) -> tuple[Any, Any, Any]:
    counts = ak.num(pt, axis=1)
    flat_pt = np.asarray(ak.to_numpy(ak.flatten(pt, axis=1)), dtype=float)
    flat_abseta = np.abs(np.asarray(ak.to_numpy(ak.flatten(eta, axis=1)), dtype=float))
    result = _flat_triplet(
        _payload(repo, "loose_muon_5to10")["loose_muon_5to10_sf"],
        flat_abseta,
        flat_pt,
    )
    return tuple(ak.unflatten(values, counts) for values in result)
