#!/usr/bin/env python3
"""Quantify a TROTA resolved-top extension of the 2024 High-dM categories.

The sparse TROTA tree contains every triplet above the 1% QCD-mistag working
point.  It is therefore not an event-level Nres collection.  This study builds
an event-level multiplicity by sorting candidates by QCD discriminant and
greedily rejecting candidates that share an AK4 jet with an already accepted
candidate, matching the candidate-candidate arbitration described in the
Run-2 analysis note.

The physics-safe proposal studied here is deliberately limited to events with
no selected boosted top or W.  In that block, no AK8-tag cross-cleaning is
needed, so the intermediate format's lack of subjet identity is irrelevant.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
import uproot


REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")
REGION_BRANCHES = {region: f"feature_{region}" for region in REGIONS}
DATA_PROCESS_BY_REGION = {
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
    "SR": "JetMET",
}
RECOIL_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0])
RECOIL_BRANCH = {
    "LLCR": "met",
    "QCDCR": "met",
    "GCR": "recoil_gcr",
    "DY2E": "recoil_dy2e",
    "DY2M": "recoil_dy2m",
    "SR": "met",
}
TOPOLOGY_BY_ID = {1: "T2tt", 2: "T2tb", 3: "T2bW"}
DEFAULT_SIGNAL_POINTS = {
    ("T2tt", 1200, 500),
    ("T2tb", 1200, 500),
    ("T2bW", 1200, 500),
    ("T2tt", 1600, 1),
    ("T2tb", 1600, 1),
    ("T2bW", 1600, 1),
}

EVENT_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    "file_id",
    "entry",
    "dataset_id",
    "is_data",
    "is_signal",
    "signal_topology_id",
    "mStop",
    "mLSP",
    "gen_weight",
    "njet",
    "nb_medium",
    "nboosted_top",
    "nboosted_w",
    "nboosted_total",
    "met",
    "recoil_gcr",
    "recoil_dy2e",
    "recoil_dy2m",
    *REGION_BRANCHES.values(),
)
TROTA_PRIMARY_ID_BRANCHES = (
    "file_id",
    "entry",
)
TROTA_FALLBACK_ID_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
)
TROTA_VALUE_BRANCHES = (
    "TopResolved1pct_sourceJetIdx0",
    "TopResolved1pct_sourceJetIdx1",
    "TopResolved1pct_sourceJetIdx2",
    "TopResolved1pct_eta",
    "TopResolved1pct_mass",
    "TopResolved1pct_QCDDiscriminant",
)
TROTA_BRANCHES = TROTA_PRIMARY_ID_BRANCHES + TROTA_VALUE_BRANCHES

_NORMALIZATION: dict[str, Any] | None = None
_SIGNAL_POINTS: set[tuple[str, int, int]] | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_key(file_id: np.ndarray, entry: np.ndarray) -> np.ndarray:
    file_values = np.asarray(file_id, dtype=np.uint64)
    entry_values = np.asarray(entry, dtype=np.uint64)
    if np.any(file_values >= (1 << 31)) or np.any(entry_values >= (1 << 32)):
        raise RuntimeError("file_id/entry exceeds the validated 31/32-bit join key")
    return (file_values << np.uint64(32)) | entry_values


def map_candidates_to_events(
    event_file_id: np.ndarray,
    event_entry: np.ndarray,
    candidate_file_id: np.ndarray,
    candidate_entry: np.ndarray,
) -> np.ndarray:
    event_keys = combined_key(event_file_id, event_entry)
    candidate_keys = combined_key(candidate_file_id, candidate_entry)
    if np.unique(event_keys).size != event_keys.size:
        raise RuntimeError("(file_id, entry) is not unique in Events")
    order = np.argsort(event_keys, kind="stable")
    sorted_keys = event_keys[order]
    positions = np.searchsorted(sorted_keys, candidate_keys)
    if np.any(positions >= sorted_keys.size):
        raise RuntimeError("TROTA candidate does not map to an Events entry")
    mapped = order[positions]
    if not np.array_equal(event_keys[mapped], candidate_keys):
        raise RuntimeError("TROTA candidate identity differs from Events identity")
    return np.asarray(mapped, dtype=np.int64)


def map_candidates_to_events_rle(
    event_run: np.ndarray,
    event_lumi: np.ndarray,
    event_number: np.ndarray,
    candidate_run: np.ndarray,
    candidate_lumi: np.ndarray,
    candidate_number: np.ndarray,
) -> np.ndarray:
    """Fallback join for a malformed TROTA ``entry`` basket.

    The fallback is deliberately tuple-based rather than a lossy packed hash.
    It is only used after the primary ``(file_id, entry)`` read fails.
    """

    event_keys = list(
        zip(
            np.asarray(event_run).tolist(),
            np.asarray(event_lumi).tolist(),
            np.asarray(event_number).tolist(),
        )
    )
    lookup = {key: index for index, key in enumerate(event_keys)}
    if len(lookup) != len(event_keys):
        raise RuntimeError("(run, luminosityBlock, event) is not unique in Events")
    candidate_keys = zip(
        np.asarray(candidate_run).tolist(),
        np.asarray(candidate_lumi).tolist(),
        np.asarray(candidate_number).tolist(),
    )
    mapped: list[int] = []
    for key in candidate_keys:
        if key not in lookup:
            raise RuntimeError("TROTA run/lumi/event identity does not map to Events")
        mapped.append(lookup[key])
    return np.asarray(mapped, dtype=np.int64)


def greedy_disjoint_counts(
    event_index: np.ndarray,
    source0: np.ndarray,
    source1: np.ndarray,
    source2: np.ndarray,
    score: np.ndarray,
    number_of_events: int,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return score-prioritized, jet-disjoint candidate multiplicity per event."""

    event_index = np.asarray(event_index, dtype=np.int64)
    source0 = np.asarray(source0, dtype=np.int64)
    source1 = np.asarray(source1, dtype=np.int64)
    source2 = np.asarray(source2, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if candidate_mask is None:
        selected = np.ones(event_index.size, dtype=bool)
    else:
        selected = np.asarray(candidate_mask, dtype=bool)
    selected &= (
        (event_index >= 0)
        & (event_index < number_of_events)
        & (source0 >= 0)
        & (source1 >= 0)
        & (source2 >= 0)
        & (source0 < 63)
        & (source1 < 63)
        & (source2 < 63)
        & np.isfinite(score)
    )
    indices = np.flatnonzero(selected)
    counts = np.zeros(number_of_events, dtype=np.int16)
    if indices.size == 0:
        return counts
    ordered = indices[np.lexsort((-score[indices], event_index[indices]))]
    current_event = -1
    used_jets = 0
    for candidate in ordered.tolist():
        event = int(event_index[candidate])
        if event != current_event:
            current_event = event
            used_jets = 0
        jet_mask = (
            (1 << int(source0[candidate]))
            | (1 << int(source1[candidate]))
            | (1 << int(source2[candidate]))
        )
        if used_jets & jet_mask:
            continue
        used_jets |= jet_mask
        counts[event] += 1
    return counts


def signal_mass_key(topology: str, mstop: int, mlsp: int) -> str:
    prefix = "" if topology == "T2tt" else f"{topology}_"
    return f"{prefix}mStop{mstop}_mLSP{mlsp}"


def canonical_process(process: str, dataset: str) -> str:
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "ST"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "TT"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "GJ"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "WtoLNu"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Zto2Nu"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD"
    if process == "VV":
        return "VV"
    return process or "other"


def parse_signal_points(values: Iterable[str]) -> set[tuple[str, int, int]]:
    if not values:
        return set(DEFAULT_SIGNAL_POINTS)
    out: set[tuple[str, int, int]] = set()
    for value in values:
        fields = value.split(":")
        if len(fields) != 3 or fields[0] not in TOPOLOGY_BY_ID.values():
            raise ValueError(f"invalid signal point {value!r}; expected T2tt:1200:500")
        out.add((fields[0], int(fields[1]), int(fields[2])))
    return out


def sample_and_weight(
    events: dict[str, np.ndarray],
    normalization: dict[str, Any],
    selected_signal_points: set[tuple[str, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(events["dataset_id"])
    labels = np.full(n, "skip", dtype=object)
    weights = np.zeros(n, dtype=np.float64)
    process = np.full(n, "unknown", dtype=object)
    dataset_factors = normalization.get("dataset_factors") or {}
    signal_factors = normalization.get("signal_mass_points") or {}
    dataset_ids = np.asarray(events["dataset_id"], dtype=np.int64)
    is_data = np.asarray(events["is_data"], dtype=bool)
    is_signal = np.asarray(events["is_signal"], dtype=bool)
    gen_weight = np.asarray(events["gen_weight"], dtype=np.float64)
    for dataset_id in np.unique(dataset_ids):
        mask = dataset_ids == dataset_id
        record = dataset_factors.get(str(int(dataset_id))) or {}
        dataset = str(record.get("dataset") or "unknown")
        raw_process = str(record.get("process") or "unknown")
        current_process = canonical_process(raw_process, dataset)
        process[mask] = current_process
        if np.any(is_data[mask]):
            labels[mask] = "data_obs"
            weights[mask] = 1.0
        elif not np.any(is_signal[mask]):
            factor = record.get("normalization_factor")
            if factor is None or not math.isfinite(float(factor)) or float(factor) <= 0:
                raise RuntimeError(f"invalid normalization factor for dataset_id={dataset_id}")
            labels[mask] = current_process
            weights[mask] = gen_weight[mask] * float(factor)
    signal_indices = np.flatnonzero(is_signal)
    for index in signal_indices.tolist():
        topology = TOPOLOGY_BY_ID.get(int(events["signal_topology_id"][index]), "")
        point = (topology, int(events["mStop"][index]), int(events["mLSP"][index]))
        if point not in selected_signal_points:
            continue
        key = signal_mass_key(*point)
        factor = (signal_factors.get(key) or {}).get("normalization_factor")
        if factor is None or not math.isfinite(float(factor)) or float(factor) <= 0:
            raise RuntimeError(f"invalid signal normalization factor for {key}")
        labels[index] = f"{topology}_mStop{point[1]}_mLSP{point[2]}"
        process[index] = topology
        weights[index] = gen_weight[index] * float(factor)
    return labels, weights, process


def initialize_worker(
    normalization_path: str,
    signal_points: tuple[tuple[str, int, int], ...],
) -> None:
    global _NORMALIZATION, _SIGNAL_POINTS
    _NORMALIZATION = json.loads(Path(normalization_path).read_text())
    _SIGNAL_POINTS = set(signal_points)


def category(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    return "2plus"


def nb_category(value: int) -> str:
    if value <= 1:
        return "Nb1"
    if value == 2:
        return "Nb2"
    return "Nb3plus"


def njet_category(value: int) -> str:
    return "Nj7plus" if value >= 7 else "Nj5to6"


def add_stat(
    target: dict[str, dict[str, float | int]], key: str, weight: float
) -> None:
    record = target.setdefault(key, {"entries": 0, "sumw": 0.0, "sumw2": 0.0})
    record["entries"] = int(record["entries"]) + 1
    record["sumw"] = float(record["sumw"]) + float(weight)
    record["sumw2"] = float(record["sumw2"]) + float(weight) * float(weight)


def process_file(
    input_root: str,
    kind: str,
) -> dict[str, Any]:
    start = time.monotonic()
    if _NORMALIZATION is None or _SIGNAL_POINTS is None:
        raise RuntimeError("worker was not initialized with normalization metadata")
    with uproot.open(input_root, object_cache=None, array_cache=None) as root_file:
        if "Events" not in root_file or "TROTA" not in root_file:
            raise RuntimeError(f"missing Events/TROTA tree in {input_root}")
        event_tree = root_file["Events"]
        available = set(event_tree.keys())
        missing = sorted(set(EVENT_BRANCHES) - available)
        if missing:
            raise RuntimeError(f"missing Events branches in {input_root}: {missing}")
        events_ak = event_tree.arrays(EVENT_BRANCHES, library="ak")
        events = {name: np.asarray(ak.to_numpy(events_ak[name])) for name in EVENT_BRANCHES}
        number_of_events = len(events["entry"])
        labels, weights, processes = sample_and_weight(
            events, _NORMALIZATION, _SIGNAL_POINTS
        )
        nt0w0 = (
            (np.asarray(events["nboosted_top"], dtype=int) == 0)
            & (np.asarray(events["nboosted_w"], dtype=int) == 0)
            & (np.asarray(events["nboosted_total"], dtype=int) == 0)
            & (np.asarray(events["nb_medium"], dtype=int) >= 1)
        )
        any_region = np.zeros(number_of_events, dtype=bool)
        for branch in REGION_BRANCHES.values():
            any_region |= np.asarray(events[branch], dtype=bool)
        study_events = nt0w0 & any_region & (labels != "skip")

        raw_counts = np.zeros(number_of_events, dtype=np.int16)
        disjoint_counts = np.zeros(number_of_events, dtype=np.int16)
        fiducial_counts = np.zeros(number_of_events, dtype=np.int16)
        trota_rows = 0
        selected_rows = 0
        identity_fallback = 0
        if np.any(study_events):
            trota_tree = root_file["TROTA"]
            missing_trota = sorted(
                (set(TROTA_BRANCHES) | set(TROTA_FALLBACK_ID_BRANCHES))
                - set(trota_tree.keys())
            )
            if missing_trota:
                raise RuntimeError(f"missing TROTA branches in {input_root}: {missing_trota}")
            try:
                trota_ak = trota_tree.arrays(TROTA_BRANCHES, library="ak")
                trota = {
                    name: np.asarray(ak.to_numpy(trota_ak[name]))
                    for name in TROTA_BRANCHES
                }
                candidate_event = map_candidates_to_events(
                    events["file_id"],
                    events["entry"],
                    trota["file_id"],
                    trota["entry"],
                )
            except Exception as primary_identity_error:
                fallback_branches = TROTA_FALLBACK_ID_BRANCHES + TROTA_VALUE_BRANCHES
                try:
                    trota_ak = trota_tree.arrays(fallback_branches, library="ak")
                    trota = {
                        name: np.asarray(ak.to_numpy(trota_ak[name]))
                        for name in fallback_branches
                    }
                    candidate_event = map_candidates_to_events_rle(
                        events["run"],
                        events["luminosityBlock"],
                        events["event"],
                        trota["run"],
                        trota["luminosityBlock"],
                        trota["event"],
                    )
                    identity_fallback = 1
                except Exception as fallback_identity_error:
                    raise RuntimeError(
                        "both TROTA identity joins failed; "
                        f"primary={type(primary_identity_error).__name__}: "
                        f"{primary_identity_error}; fallback="
                        f"{type(fallback_identity_error).__name__}: "
                        f"{fallback_identity_error}"
                    ) from fallback_identity_error
            trota_rows = len(trota["event"] if identity_fallback else trota["entry"])
            selected_candidate = study_events[candidate_event]
            selected_rows = int(np.count_nonzero(selected_candidate))
            raw_counts = np.bincount(
                candidate_event[selected_candidate], minlength=number_of_events
            ).astype(np.int16, copy=False)
            score = trota["TopResolved1pct_QCDDiscriminant"]
            source0 = trota["TopResolved1pct_sourceJetIdx0"]
            source1 = trota["TopResolved1pct_sourceJetIdx1"]
            source2 = trota["TopResolved1pct_sourceJetIdx2"]
            disjoint_counts = greedy_disjoint_counts(
                candidate_event,
                source0,
                source1,
                source2,
                score,
                number_of_events,
                selected_candidate,
            )
            run2_kinematic = (
                selected_candidate
                & (np.abs(trota["TopResolved1pct_eta"]) < 2.0)
                & (trota["TopResolved1pct_mass"] >= 100.0)
                & (trota["TopResolved1pct_mass"] <= 250.0)
            )
            fiducial_counts = greedy_disjoint_counts(
                candidate_event,
                source0,
                source1,
                source2,
                score,
                number_of_events,
                run2_kinematic,
            )

    stats: dict[str, dict[str, float | int]] = {}
    schemes = {
        "raw_pass_triplets": raw_counts,
        "trota_disjoint": disjoint_counts,
        "trota_disjoint_run2_kinematic": fiducial_counts,
    }
    nb_values = np.asarray(events["nb_medium"], dtype=int)
    njet_values = np.asarray(events["njet"], dtype=int)
    is_data = np.asarray(events["is_data"], dtype=bool)
    for region, branch in REGION_BRANCHES.items():
        recoil = np.asarray(events[RECOIL_BRANCH[region]], dtype=float)
        recoil_index = np.searchsorted(RECOIL_EDGES, recoil, side="right") - 1
        region_mask = study_events & np.asarray(events[branch], dtype=bool)
        region_mask &= (recoil_index >= 0) & (recoil_index < RECOIL_EDGES.size - 1)
        if np.any(is_data):
            region_mask &= (~is_data) | (processes == DATA_PROCESS_BY_REGION[region])
        for event_index in np.flatnonzero(region_mask).tolist():
            recoil_label = (
                f"{int(RECOIL_EDGES[recoil_index[event_index]])}-"
                f"{int(RECOIL_EDGES[recoil_index[event_index] + 1])}"
            )
            common = (
                region,
                str(labels[event_index]),
                nb_category(int(nb_values[event_index])),
                njet_category(int(njet_values[event_index])),
                recoil_label,
            )
            for scheme, counts in schemes.items():
                key = "|".join((scheme, *common, category(int(counts[event_index]))))
                add_stat(stats, key, float(weights[event_index]))

    selected_event_count = int(np.count_nonzero(study_events))
    return {
        "input_root": input_root,
        "kind": kind,
        "events": number_of_events,
        "study_events": selected_event_count,
        "trota_rows": trota_rows,
        "study_candidate_rows": selected_rows,
        "events_with_raw_candidate": int(np.count_nonzero(study_events & (raw_counts > 0))),
        "events_with_disjoint_candidate": int(
            np.count_nonzero(study_events & (disjoint_counts > 0))
        ),
        "events_with_run2_kinematic_candidate": int(
            np.count_nonzero(study_events & (fiducial_counts > 0))
        ),
        "identity_fallback_files": identity_fallback,
        "stats": stats,
        "wall_time_seconds": time.monotonic() - start,
    }


def merge_stats(
    target: dict[str, dict[str, float | int]],
    source: dict[str, dict[str, float | int]],
) -> None:
    for key, record in source.items():
        destination = target.setdefault(key, {"entries": 0, "sumw": 0.0, "sumw2": 0.0})
        destination["entries"] = int(destination["entries"]) + int(record["entries"])
        destination["sumw"] = float(destination["sumw"]) + float(record["sumw"])
        destination["sumw2"] = float(destination["sumw2"]) + float(record["sumw2"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--only-input-root", action="append", default=[])
    parser.add_argument("--signal-point", action="append", default=[])
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text())
    normalization = json.loads(args.normalization.read_text())
    if normalization.get("status") != "complete":
        raise RuntimeError("normalization manifest is not complete")
    inputs = list(manifest.get("inputs") or [])
    if args.only_input_root:
        requested = set(args.only_input_root)
        inputs = [record for record in inputs if str(record.get("input_root")) in requested]
        found = {str(record.get("input_root")) for record in inputs}
        if found != requested:
            raise RuntimeError(
                f"requested inputs missing from manifest: {sorted(requested - found)}"
            )
    if args.max_files:
        inputs = inputs[: args.max_files]
    signal_points = tuple(sorted(parse_signal_points(args.signal_point)))
    input_manifest_sha256 = file_sha256(args.input_manifest)
    normalization_sha256 = file_sha256(args.normalization)
    merged_stats: dict[str, dict[str, float | int]] = {}
    totals = defaultdict(int)
    failures: list[dict[str, str]] = []
    completed = 0
    started = time.monotonic()

    def snapshot(status: str) -> dict[str, Any]:
        return {
            "schema_version": "trota_highdm_category_study_2024_v1",
            "status": status,
            "updated_at": now(),
            "input_manifest": str(args.input_manifest),
            "input_manifest_sha256": input_manifest_sha256,
            "normalization": str(args.normalization),
            "normalization_sha256": normalization_sha256,
            "files_expected": len(inputs),
            "files_completed": completed,
            "workers": args.workers,
            "signal_points": [list(point) for point in signal_points],
            "totals": dict(totals),
            "failures": failures,
            "stats": merged_stats,
            "wall_time_seconds": time.monotonic() - started,
            "method": {
                "candidate_wp": "TROTA 1% QCD-mistag sparse pass tree",
                "disjoint_arbitration": "descending QCDDiscriminant, greedily reject shared AK4 jets",
                "physics_safe_scope": "current High-dM region events with Nb>=1, Nt=0, Nw=0",
                "weights": "raw gen_weight times validated xsec*lumi/sumw normalization; no post-skim AnalysisSF and no TROTA SF",
                "run2_kinematic_variant": "disjoint plus 100<=m(trijet)<=250 GeV and abs(eta(trijet))<2.0",
            },
        }

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.normalization), signal_points),
    ) as executor:
        future_map = {
            executor.submit(
                process_file,
                str(record["input_root"]),
                str(record["kind"]),
            ): record
            for record in inputs
        }
        for future in concurrent.futures.as_completed(future_map):
            record = future_map[future]
            try:
                result = future.result()
                merge_stats(merged_stats, result["stats"])
                for key in (
                    "events",
                    "study_events",
                    "trota_rows",
                    "study_candidate_rows",
                    "events_with_raw_candidate",
                    "events_with_disjoint_candidate",
                    "events_with_run2_kinematic_candidate",
                    "identity_fallback_files",
                ):
                    totals[key] += int(result[key])
                totals[f"files_{result['kind']}"] += 1
            except Exception as exc:
                failures.append(
                    {
                        "input_root": str(record.get("input_root")),
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                )
            completed += 1
            if completed % 25 == 0 or completed == len(inputs):
                atomic_json(args.output, snapshot("running"))
                print(
                    f"completed={completed}/{len(inputs)} failures={len(failures)} "
                    f"study_events={totals['study_events']}",
                    flush=True,
                )

    status = "complete" if completed == len(inputs) and not failures else "failed"
    atomic_json(args.output, snapshot(status))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
