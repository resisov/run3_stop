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
    "topw_tagging": "topw_tagging_sf.json.gz",
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
REQUIRED_ANALYSIS_SF_COMPONENTS = tuple(key for key in PAYLOADS if key != "topw_tagging")
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
TOPW_EVENT_BRANCHES = (
    "run", "luminosityBlock", "event", "entry", "file_id",
    "fatjet_source_index_all", "fatjet_corrected_pt", "fatjet_eta_all",
    "fatjet_phi_all", "fatjet_msoftdrop_all", "fatjet_id_all",
    "fatjet_boosted_top_pass_all", "fatjet_boosted_w_pass_all",
)
TOPW_PT_EDGES = {"top": (300, 400, 480, 600, 1200), "w": (200, 300, 400, 800)}


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
    missing.extend("Events/" + name for name in TOPW_EVENT_BRANCHES if name not in events.keys())
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


class TopWEvents:
    """Read aligned correction columns from the same integrated ROOT file."""

    def __init__(self, root_file: Any, policy: dict[str, Any]):
        self.root_file = root_file
        self.events = root_file["Events"]
        self.policy = policy
        self.num_entries = self.events.num_entries

    def arrays(self, branches: Any, *, entry_start: int = 0,
               entry_stop: int | None = None, library: str = "ak") -> Any:
        if library != "ak":
            raise ValueError("Top/W input reader requires awkward arrays")
        available = self.policy.get("mode") == "available"
        requested = list(dict.fromkeys([*branches, *(TOPW_EVENT_BRANCHES if available else ())]))
        result = self.events.arrays(requested, entry_start=entry_start,
                                    entry_stop=entry_stop, library="ak")
        if not available:
            return result
        truth = self.root_file["TopWTruth"].arrays(
            TOPW_CORRECTION_BRANCHES, entry_start=entry_start, entry_stop=entry_stop, library="ak",
        )
        counts = np.asarray(ak.num(result["fatjet_source_index_all"], axis=1))
        if (not np.array_equal(counts, np.asarray(ak.num(truth["fatjet_source_index_all"], axis=1)))
            or not np.array_equal(counts, np.asarray(ak.num(truth["fatjet_decay_flavor_all"], axis=1)))):
            raise RuntimeError("TopWTruth fatjet count mismatch")
        for name in TOPW_CORRECTION_BRANCHES[:-1]:
            if not np.array_equal(np.asarray(ak.flatten(result[name], axis=None)),
                                  np.asarray(ak.flatten(truth[name], axis=None))):
                raise RuntimeError("TopWTruth/Events identity mismatch: " + name)
        return ak.with_field(result, truth["fatjet_decay_flavor_all"], "fatjet_decay_flavor_all")

    def iterate(self, branches: Any, *, step_size: int, library: str = "ak") -> Any:
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("Top/W chunk size must be a positive entry count")
        for start in range(0, self.num_entries, step_size):
            yield self.arrays(branches, entry_start=start,
                              entry_stop=min(start + step_size, self.num_entries), library=library)


def _topw_dataset_primary(dataset: str) -> str:
    return dataset.lstrip("/").split("/", 1)[0].split("_Tune", 1)[0]


@lru_cache(maxsize=2)
def _topw_efficiencies(path: str) -> Any:
    from coffea.util import load
    return load(path)


def topw_fit_variations(payload: Any) -> dict[str, tuple[str, str, int, int, tuple[float, ...]]]:
    result = {}
    for tag, edges in TOPW_PT_EDGES.items():
        for cat in ("tp1", "tp2", "tp3", "other"):
            for lo, hi in zip(edges[:-1], edges[1:]):
                if tag == "top" and hi <= 400:
                    continue
                scales = tuple(float(payload[f"topw_{tag}_{cat}_sf"].evaluate(v, (lo + hi) / 2.))
                               for v in ("nominal", "up", "down"))
                if scales[1] != scales[0] or scales[2] != scales[0]:
                    result[f"topw_{tag}_{cat}_pt{lo}to{hi}"] = (tag, cat, lo, hi, scales)
    return result


def topw_event_variations(
    repo: Path, year: str, dataset: str, process: str, arrays: Any,
    policy: dict[str, Any], *, cleaned: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate analysis-eligible AK8 jets; keep fit-bin variations separate."""
    n = len(arrays["gen_weight"])
    payload = _payload(repo, "topw_tagging", str(year))
    sources = topw_fit_variations(payload)
    if policy.get("mode") == "unity_missing_branches":
        unity = topw_file_sf_triplet(n, policy, None)[0]
        return {"nominal": unity, **{name + direction: unity for name in sources
                                     for direction in ("Up", "Down")}}, dict(policy)
    if policy.get("mode") != "available":
        raise ValueError("Top/W correction input policy is required")
    efficiency_path = repo / "analysis/hists" / f"topwtageff{year}.merged"
    histograms = _topw_efficiencies(str(efficiency_path))
    primary = _topw_dataset_primary(dataset)
    keys = list(histograms["GlobalParT3_Top"])
    matches = [key for key in keys if key == dataset]
    if not matches:
        matches = [key for key in keys if _topw_dataset_primary(key) == primary]
    if len(matches) != 1:
        raise AnalysisScaleFactorUnavailable(f"Top/W efficiency dataset match is not unique: {dataset} ({len(matches)})")
    key = matches[0]
    top_like = process in {"TT", "ST", "TTV", "TTW", "TTZ"} or primary.startswith(("TT", "TW", "Tbar", "TB", "SMS-"))
    counts = ak.num(arrays["fatjet_corrected_pt"], axis=1)
    flat = lambda value: np.asarray(ak.to_numpy(ak.flatten(value, axis=1)))
    pt = flat(arrays["fatjet_corrected_pt"])
    eta = np.abs(flat(arrays["fatjet_eta_all"]))
    mass = flat(arrays["fatjet_msoftdrop_all"])
    category = topw_fit_categories(flat(arrays["fatjet_decay_flavor_all"]), top_like=top_like)
    base = flat(arrays["fatjet_id_all"]).astype(bool) & (eta < 2.)
    if cleaned is not None:
        if not np.array_equal(np.asarray(ak.num(cleaned, axis=1)), np.asarray(counts)):
            raise ValueError("Top/W cleaned-jet mask is not aligned")
        base &= flat(cleaned).astype(bool)
    if not np.all(np.isfinite(pt)) or not np.all(np.isfinite(eta)) or not np.all(np.isfinite(mass)):
        raise AnalysisScaleFactorUnavailable("non-finite Top/W jet kinematics")
    nominal = np.ones(len(pt), dtype=float)
    alternatives: list[tuple[str, np.ndarray, np.ndarray]] = []
    audit = {"mode": "measured", "efficiency_dataset": key, "eligible_jets": 0,
             "zero_denominator_jets": 0, "saturated_nominal_jets": 0,
             "highpt_unity_jets": 0}
    groups = {"tp1": [0, 1, 2], "tp2": [3], "tp3": [4]} if top_like else {"other": [0, 1, 2, 3, 4]}
    for tag, variable in (("top", "GlobalParT3_Top"), ("w", "GlobalParT3_W")):
        eligible = base & ((pt > 400.) & (mass > 105.) if tag == "top" else (pt > 200.) & (mass > 60.) & (mass < 105.))
        audit["eligible_jets"] += int(eligible.sum())
        audit["highpt_unity_jets"] += int((eligible & (pt >= TOPW_PT_EDGES[tag][-1])).sum())
        h = histograms[variable][key]
        values = h.values()[0]
        pt_index = np.clip(np.searchsorted(h.axes["pt"].edges, pt, side="right") - 1, 0, h.axes["pt"].size - 1)
        eta_index = np.clip(np.searchsorted(h.axes["abseta"].edges, eta, side="right") - 1, 0, h.axes["abseta"].size - 1)
        for cat, flavors in groups.items():
            selected = eligible & (category == cat)
            correction = payload[f"topw_{tag}_{cat}_sf"]
            for lo, hi in zip(TOPW_PT_EDGES[tag][:-1], TOPW_PT_EDGES[tag][1:]):
                if tag == "top" and hi <= 400:
                    continue
                active = selected & (pt >= lo) & (pt < hi)
                scales = tuple(float(correction.evaluate(v, (lo + hi) / 2.)) for v in ("nominal", "up", "down"))
                if not np.any(active):
                    continue
                passed = values[0, flavors, :, :].sum(axis=0)[pt_index[active], eta_index[active]]
                failed = values[1, flavors, :, :].sum(axis=0)[pt_index[active], eta_index[active]]
                denominator = passed + failed
                efficiency = np.divide(passed, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
                zero = (denominator == 0) | (passed == 0) | (failed == 0)
                audit["zero_denominator_jets"] += int(zero.sum())
                audit["saturated_nominal_jets"] += int(((efficiency * scales[0] > 1) & ~zero).sum())
                weights = topw_pass_fail_triplet(
                    flat(arrays[f"fatjet_boosted_{tag}_pass_all"])[active].astype(bool), efficiency,
                    *(np.full(efficiency.shape, value) for value in scales), denominator=denominator,
                )
                nominal[active] = weights[0]
                if scales[1] != scales[0] or scales[2] != scales[0]:
                    name = f"topw_{tag}_{cat}_pt{lo}to{hi}"
                    for direction, varied in zip(("Up", "Down"), weights[1:]):
                        alternatives.append((name + direction, active, varied))
    product = lambda values: np.asarray(ak.prod(ak.unflatten(values, counts), axis=1), dtype=float)
    event_nominal = product(nominal)
    result = {"nominal": event_nominal, **{name + direction: event_nominal for name in sources
                                          for direction in ("Up", "Down")}}
    for name, active, changed in alternatives:
        varied = nominal.copy()
        varied[active] = changed
        result[name] = product(varied)
    return result, audit


def apply_topw_event_weights(
    variations: dict[str, Any], status: dict[str, Any], repo: Path, year: str,
    dataset: str, process: str, arrays: Any, policy: dict[str, Any], *, cleaned: Any = None,
) -> dict[str, np.ndarray]:
    topw, audit = topw_event_variations(repo, year, dataset, process, arrays, policy, cleaned=cleaned)
    output = {name: np.asarray(value) * topw["nominal"] for name, value in variations.items()}
    for name, weight in topw.items():
        if name != "nominal":
            if name in output:
                raise ValueError("duplicate Top/W variation: " + name)
            output[name] = np.asarray(variations["nominal"]) * weight
    status["topw_correction"] = audit
    status.setdefault("components", {})["topw_tagging"] = {
        "applied": True, "source": "integrated_TopWTruth", "reason": audit["mode"],
    }
    status["available_variations"] = sorted(output)
    return output


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
    *,
    denominator: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound corrected probabilities; use unity for zero-denominator cells."""
    tagged = np.asarray(tagged)
    efficiency = np.asarray(efficiency, dtype=float)
    if tagged.ndim != 1 or tagged.dtype != np.bool_ or efficiency.shape != tagged.shape:
        raise ValueError("Top/W tag decisions and efficiencies must be aligned flat arrays")
    zero_denominator = np.zeros(tagged.shape, dtype=bool)
    if denominator is not None:
        denominator = np.asarray(denominator, dtype=float)
        if (denominator.shape != tagged.shape or not np.all(np.isfinite(denominator))
            or np.any(denominator < 0)):
            raise ValueError("Top/W efficiency denominators must be aligned nonnegative finite counts")
        zero_denominator = denominator == 0
        efficiency = np.where(zero_denominator, 0.0, efficiency)
    if (not np.all(np.isfinite(efficiency))
        or np.any((efficiency < 0) | (efficiency > 1))):
        raise AnalysisScaleFactorUnavailable("invalid Top/W MC efficiency")
    zero_denominator |= (efficiency == 0) | (efficiency == 1)
    variations = []
    for value in (nominal, up, down):
        scale = np.asarray(value, dtype=float)
        if scale.shape != tagged.shape:
            raise ValueError("Top/W SF and efficiency array shapes differ")
        scale = np.where(zero_denominator, 1.0, scale)
        if not np.all(np.isfinite(scale)) or np.any(scale < 0):
            raise AnalysisScaleFactorUnavailable("invalid Top/W SF variation")
        variations.append(scale)
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
