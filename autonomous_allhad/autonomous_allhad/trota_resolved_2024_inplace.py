from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import awkward as ak
import numpy as np
import uproot

try:
    from .trota_resolved_2024 import (
        MODEL_INPUT_BRANCHES,
        SELECTED_TOPRESOLVED_2024_WORKING_POINT,
        TOPRESOLVED_2024_MODEL_SHA256,
        TOPRESOLVED_2024_QCD_WORKING_POINTS,
        TROTA_COMMIT,
        sha256,
    )
except ImportError:
    from trota_resolved_2024 import (  # type: ignore[no-redef]
        MODEL_INPUT_BRANCHES,
        SELECTED_TOPRESOLVED_2024_WORKING_POINT,
        TOPRESOLVED_2024_MODEL_SHA256,
        TOPRESOLVED_2024_QCD_WORKING_POINTS,
        TROTA_COMMIT,
        sha256,
    )


DEFAULT_TARGET_YEAR = 2024
SUPPORTED_TARGET_YEARS = (2024, 2025)
MODEL_RELEASE_YEAR = 2024
SCHEMA_VERSION = "trota_topresolved_2024_inplace_sparse_v1"
TREE_NAME = "TROTA"
MARKER_NAME = "TROTA_metadata"
WP_THRESHOLD = np.float32(
    TOPRESOLVED_2024_QCD_WORKING_POINTS[SELECTED_TOPRESOLVED_2024_WORKING_POINT]
)

EVENT_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    "entry",
    "dataset_id",
    "file_id",
)
JET_BRANCHES = (
    "jet_nanoaod_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_nanoaod_mass",
    "jet_area",
    "jet_btag_upart_all",
    "jet_id_all",
)

OUTPUT_DTYPES: dict[str, np.dtype[Any]] = {
    "run": np.dtype(np.uint32),
    "luminosityBlock": np.dtype(np.uint32),
    "event": np.dtype(np.uint64),
    "entry": np.dtype(np.int64),
    "dataset_id": np.dtype(np.int64),
    "file_id": np.dtype(np.int64),
    "TopResolved1pct_candidateIndex": np.dtype(np.int32),
    "TopResolved1pct_idxJet0": np.dtype(np.int32),
    "TopResolved1pct_idxJet1": np.dtype(np.int32),
    "TopResolved1pct_idxJet2": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx0": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx1": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx2": np.dtype(np.int32),
    "TopResolved1pct_pt": np.dtype(np.float32),
    "TopResolved1pct_eta": np.dtype(np.float32),
    "TopResolved1pct_phi": np.dtype(np.float32),
    "TopResolved1pct_mass": np.dtype(np.float32),
    "TopResolved1pct_FTScore": np.dtype(np.float32),
    "TopResolved1pct_TTScore": np.dtype(np.float32),
    "TopResolved1pct_QCDScore": np.dtype(np.float32),
    "TopResolved1pct_QCDDiscriminant": np.dtype(np.float32),
}

FLOAT_OUTPUT_BRANCHES = tuple(
    name for name, dtype in OUTPUT_DTYPES.items() if dtype == np.dtype(np.float32)
)


def schema_version_for_year(target_year: int) -> str:
    if target_year not in SUPPORTED_TARGET_YEARS:
        raise ValueError(f"unsupported TROTA target year: {target_year}")
    return f"trota_topresolved_{target_year}_inplace_sparse_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_candidates_impl(
    offsets: np.ndarray,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    area: np.ndarray,
    btag: np.ndarray,
    jet_id: np.ndarray,
) -> tuple[np.ndarray, ...]:
    number_of_events = offsets.size - 1
    counts = np.zeros(number_of_events, dtype=np.int32)
    total_candidates = 0
    for event_index in range(number_of_events):
        number_good = 0
        for flat_index in range(offsets[event_index], offsets[event_index + 1]):
            if jet_id[flat_index] and pt[flat_index] > 25.0 and abs(eta[flat_index]) < 2.5:
                number_good += 1
        if number_good >= 3:
            count = number_good * (number_good - 1) * (number_good - 2) // 6
            counts[event_index] = count
            total_candidates += count

    features = np.empty((total_candidates, 3, 8), dtype=np.float32)
    candidate_event_index = np.empty(total_candidates, dtype=np.int32)
    candidate_index = np.empty(total_candidates, dtype=np.int32)
    good_idx0 = np.empty(total_candidates, dtype=np.int32)
    good_idx1 = np.empty(total_candidates, dtype=np.int32)
    good_idx2 = np.empty(total_candidates, dtype=np.int32)
    source_idx0 = np.empty(total_candidates, dtype=np.int32)
    source_idx1 = np.empty(total_candidates, dtype=np.int32)
    source_idx2 = np.empty(total_candidates, dtype=np.int32)
    candidate_pt = np.empty(total_candidates, dtype=np.float32)
    candidate_eta = np.empty(total_candidates, dtype=np.float32)
    candidate_phi = np.empty(total_candidates, dtype=np.float32)
    candidate_mass = np.empty(total_candidates, dtype=np.float32)

    output_index = 0
    for event_index in range(number_of_events):
        event_start = offsets[event_index]
        event_stop = offsets[event_index + 1]
        good_sources = np.empty(event_stop - event_start, dtype=np.int32)
        number_good = 0
        for flat_index in range(event_start, event_stop):
            if jet_id[flat_index] and pt[flat_index] > 25.0 and abs(eta[flat_index]) < 2.5:
                good_sources[number_good] = flat_index
                number_good += 1

        local_candidate_index = 0
        for idx0 in range(number_good):
            for idx1 in range(idx0):
                for idx2 in range(idx1):
                    source0 = good_sources[idx0]
                    source1 = good_sources[idx1]
                    source2 = good_sources[idx2]

                    sum_px = 0.0
                    sum_py = 0.0
                    sum_pz = 0.0
                    sum_energy = 0.0
                    sources = (source0, source1, source2)
                    for source in sources:
                        jet_pt = float(pt[source])
                        jet_eta = float(eta[source])
                        jet_phi = float(phi[source])
                        jet_mass = float(mass[source])
                        px = jet_pt * math.cos(jet_phi)
                        py = jet_pt * math.sin(jet_phi)
                        pz = jet_pt * math.sinh(jet_eta)
                        energy2 = jet_mass * jet_mass + px * px + py * py + pz * pz
                        sum_px += px
                        sum_py += py
                        sum_pz += pz
                        sum_energy += math.sqrt(max(energy2, 0.0))

                    top_pt = math.hypot(sum_px, sum_py)
                    top_phi = math.atan2(sum_py, sum_px)
                    top_eta = math.asinh(sum_pz / top_pt) if top_pt > 0.0 else 0.0
                    top_mass2 = (
                        sum_energy * sum_energy
                        - sum_px * sum_px
                        - sum_py * sum_py
                        - sum_pz * sum_pz
                    )
                    top_mass = math.sqrt(max(top_mass2, 0.0))

                    for leg_index in range(3):
                        source = sources[leg_index]
                        jet_minus_top_phi = float(phi[source]) - top_phi
                        while jet_minus_top_phi > math.pi:
                            jet_minus_top_phi -= 2.0 * math.pi
                        while jet_minus_top_phi < -math.pi:
                            jet_minus_top_phi += 2.0 * math.pi
                        jet_phi = float(phi[source])
                        while jet_phi > math.pi:
                            jet_phi -= 2.0 * math.pi
                        while jet_phi < -math.pi:
                            jet_phi += 2.0 * math.pi
                        features[output_index, leg_index, 0] = np.float32(area[source])
                        features[output_index, leg_index, 1] = np.float32(btag[source])
                        features[output_index, leg_index, 2] = np.float32(
                            float(eta[source]) - top_eta
                        )
                        features[output_index, leg_index, 3] = np.float32(mass[source])
                        features[output_index, leg_index, 4] = np.float32(
                            jet_minus_top_phi
                        )
                        features[output_index, leg_index, 5] = np.float32(pt[source])
                        features[output_index, leg_index, 6] = np.float32(jet_phi)
                        features[output_index, leg_index, 7] = np.float32(eta[source])

                    candidate_event_index[output_index] = event_index
                    candidate_index[output_index] = local_candidate_index
                    good_idx0[output_index] = idx0
                    good_idx1[output_index] = idx1
                    good_idx2[output_index] = idx2
                    source_idx0[output_index] = source0 - event_start
                    source_idx1[output_index] = source1 - event_start
                    source_idx2[output_index] = source2 - event_start
                    candidate_pt[output_index] = np.float32(top_pt)
                    candidate_eta[output_index] = np.float32(top_eta)
                    candidate_phi[output_index] = np.float32(top_phi)
                    candidate_mass[output_index] = np.float32(top_mass)
                    output_index += 1
                    local_candidate_index += 1

    return (
        counts,
        features,
        candidate_event_index,
        candidate_index,
        good_idx0,
        good_idx1,
        good_idx2,
        source_idx0,
        source_idx1,
        source_idx2,
        candidate_pt,
        candidate_eta,
        candidate_phi,
        candidate_mass,
    )


_COMPILED_BUILDER: Callable[..., tuple[np.ndarray, ...]] | None = None


def candidate_builder(use_numba: bool = True) -> Callable[..., tuple[np.ndarray, ...]]:
    global _COMPILED_BUILDER
    if not use_numba:
        return _build_candidates_impl
    if _COMPILED_BUILDER is None:
        import numba

        _COMPILED_BUILDER = numba.njit(_build_candidates_impl, cache=False)
    return _COMPILED_BUILDER


def _flatten_jets(arrays: ak.Array) -> tuple[np.ndarray, ...]:
    reference_counts = np.asarray(ak.to_numpy(ak.num(arrays[JET_BRANCHES[0]])), dtype=np.int64)
    for branch in JET_BRANCHES[1:]:
        counts = np.asarray(ak.to_numpy(ak.num(arrays[branch])), dtype=np.int64)
        if not np.array_equal(counts, reference_counts):
            raise ValueError(f"unaligned intermediate jet vector: {branch}")
    offsets = np.empty(reference_counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(reference_counts, out=offsets[1:])

    flattened = []
    dtypes = (
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.bool_,
    )
    for branch, dtype in zip(JET_BRANCHES, dtypes):
        values = ak.to_numpy(ak.flatten(arrays[branch], axis=1))
        flattened.append(np.asarray(values, dtype=dtype))
    return (offsets, *flattened)


def _events_schema_digest(tree: Any) -> str:
    payload = json.dumps(tree.typenames(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_marker(root_file: Any) -> dict[str, Any] | None:
    key_names = {str(key).split(";", 1)[0] for key in root_file.keys()}
    if MARKER_NAME not in key_names:
        return None
    try:
        return json.loads(str(root_file[MARKER_NAME]))
    except Exception as exc:
        raise RuntimeError(f"invalid {MARKER_NAME}: {type(exc).__name__}: {exc}") from exc


def inspect_existing_state(
    input_path: Path,
    *,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> dict[str, Any]:
    expected_schema_version = schema_version_for_year(target_year)
    with uproot.open(input_path) as root_file:
        key_names = {str(key).split(";", 1)[0] for key in root_file.keys()}
        has_tree = TREE_NAME in key_names
        marker = _read_marker(root_file)
        if has_tree and marker is None:
            raise RuntimeError(
                "TROTA tree exists without a completion marker; refusing a non-idempotent retry"
            )
        if marker is not None and not has_tree:
            raise RuntimeError("TROTA completion marker exists but the TROTA tree is missing")
        if marker is None:
            return {"status": "absent"}
        expected = {
            "schema_version": expected_schema_version,
            "status": "complete",
            "model_sha256": TOPRESOLVED_2024_MODEL_SHA256,
            "selected_working_point": SELECTED_TOPRESOLVED_2024_WORKING_POINT,
            "selected_threshold": float(WP_THRESHOLD),
        }
        mismatches = {
            name: {"expected": value, "actual": marker.get(name)}
            for name, value in expected.items()
            if marker.get(name) != value
        }
        if target_year != DEFAULT_TARGET_YEAR and marker.get("application_year") != target_year:
            mismatches["application_year"] = {
                "expected": target_year,
                "actual": marker.get("application_year"),
            }
        if mismatches:
            raise RuntimeError(f"existing TROTA payload has incompatible provenance: {mismatches}")
        return {"status": "complete", "marker": marker}


def _empty_selected_chunks() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in OUTPUT_DTYPES}


def _append_selected_chunk(
    output: dict[str, list[np.ndarray]],
    arrays: ak.Array,
    candidate_payload: tuple[np.ndarray, ...],
    scores: np.ndarray,
    qcd_discriminant: np.ndarray,
    selected: np.ndarray,
) -> int:
    (
        _,
        _,
        candidate_event_index,
        candidate_index,
        good_idx0,
        good_idx1,
        good_idx2,
        source_idx0,
        source_idx1,
        source_idx2,
        candidate_pt,
        candidate_eta,
        candidate_phi,
        candidate_mass,
    ) = candidate_payload
    selected_event_index = candidate_event_index[selected]

    event_dtypes = {
        "run": np.uint32,
        "luminosityBlock": np.uint32,
        "event": np.uint64,
        "entry": np.int64,
        "dataset_id": np.int64,
        "file_id": np.int64,
    }
    for name, dtype in event_dtypes.items():
        event_values = np.asarray(ak.to_numpy(arrays[name]), dtype=dtype)
        output[name].append(event_values[selected_event_index])

    integer_payload = {
        "TopResolved1pct_candidateIndex": candidate_index,
        "TopResolved1pct_idxJet0": good_idx0,
        "TopResolved1pct_idxJet1": good_idx1,
        "TopResolved1pct_idxJet2": good_idx2,
        "TopResolved1pct_sourceJetIdx0": source_idx0,
        "TopResolved1pct_sourceJetIdx1": source_idx1,
        "TopResolved1pct_sourceJetIdx2": source_idx2,
    }
    for name, values in integer_payload.items():
        output[name].append(np.asarray(values[selected], dtype=np.int32))

    float_payload = {
        "TopResolved1pct_pt": candidate_pt,
        "TopResolved1pct_eta": candidate_eta,
        "TopResolved1pct_phi": candidate_phi,
        "TopResolved1pct_mass": candidate_mass,
        "TopResolved1pct_FTScore": scores[:, 0],
        "TopResolved1pct_TTScore": scores[:, 1],
        "TopResolved1pct_QCDScore": scores[:, 2],
        "TopResolved1pct_QCDDiscriminant": qcd_discriminant,
    }
    for name, values in float_payload.items():
        output[name].append(np.asarray(values[selected], dtype=np.float32))
    return int(selected_event_index.size)


def _concatenate_selected(output: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, chunks in output.items():
        dtype = OUTPUT_DTYPES[name]
        if chunks:
            result[name] = np.concatenate(chunks).astype(dtype, copy=False)
        else:
            result[name] = np.empty(0, dtype=dtype)
    return result


def _validate_appended_tree(
    input_path: Path,
    *,
    expected_event_entries: int,
    expected_schema_digest: str,
    expected_selected_candidates: int,
) -> dict[str, Any]:
    with uproot.open(input_path) as root_file:
        if "Events" not in root_file or TREE_NAME not in root_file:
            raise RuntimeError("Events or TROTA tree is missing after update")
        events = root_file["Events"]
        trota = root_file[TREE_NAME]
        if int(events.num_entries) != expected_event_entries:
            raise RuntimeError("Events entry count changed during TROTA update")
        if _events_schema_digest(events) != expected_schema_digest:
            raise RuntimeError("Events branch schema changed during TROTA update")
        if int(trota.num_entries) != expected_selected_candidates:
            raise RuntimeError(
                f"TROTA row count mismatch: {trota.num_entries} != {expected_selected_candidates}"
            )
        missing = sorted(set(OUTPUT_DTYPES) - set(trota.keys()))
        if missing:
            raise RuntimeError(f"TROTA tree is missing branches: {missing}")
        typenames = trota.typenames()
        non_float32 = {
            name: typenames.get(name)
            for name in FLOAT_OUTPUT_BRANCHES
            if typenames.get(name) != "float"
        }
        if non_float32:
            raise RuntimeError(f"non-float32 TROTA score/kinematic branches: {non_float32}")

        rows_checked = 0
        maximum_probability_deviation = 0.0
        minimum_discriminant = math.inf
        for arrays in trota.iterate(
            [
                "TopResolved1pct_FTScore",
                "TopResolved1pct_TTScore",
                "TopResolved1pct_QCDScore",
                "TopResolved1pct_QCDDiscriminant",
            ],
            step_size=500_000,
            library="np",
        ):
            ft = np.asarray(arrays["TopResolved1pct_FTScore"], dtype=np.float32)
            tt = np.asarray(arrays["TopResolved1pct_TTScore"], dtype=np.float32)
            qcd = np.asarray(arrays["TopResolved1pct_QCDScore"], dtype=np.float32)
            disc = np.asarray(
                arrays["TopResolved1pct_QCDDiscriminant"], dtype=np.float32
            )
            if not all(np.all(np.isfinite(values)) for values in (ft, tt, qcd, disc)):
                raise RuntimeError("non-finite value in appended TROTA tree")
            if disc.size and np.any(disc < WP_THRESHOLD):
                raise RuntimeError("appended TROTA tree contains a candidate below the 1% WP")
            if disc.size:
                minimum_discriminant = min(minimum_discriminant, float(np.min(disc)))
                maximum_probability_deviation = max(
                    maximum_probability_deviation,
                    float(np.max(np.abs(ft + tt + qcd - np.float32(1.0)))),
                )
            rows_checked += int(disc.size)
        if rows_checked != expected_selected_candidates:
            raise RuntimeError("TROTA validation did not read the expected number of rows")
        return {
            "events_entries": expected_event_entries,
            "trota_rows": expected_selected_candidates,
            "float_branches_are_float32": True,
            "minimum_qcd_discriminant": (
                minimum_discriminant if expected_selected_candidates else None
            ),
            "maximum_probability_sum_deviation": maximum_probability_deviation,
        }


def _validate_partial_matches_expected(
    input_path: Path,
    expected_payload: dict[str, np.ndarray],
    *,
    step_size: int = 500_000,
) -> dict[str, Any]:
    """Require an unmarked TROTA tree to match a fresh inference result.

    Event and candidate identity fields must agree exactly. Float32 kinematics and
    scores are compared with a small tolerance because TensorFlow CPU kernels can
    differ at the last few bits between worker hosts.
    """
    expected_rows = len(expected_payload["entry"])
    with uproot.open(input_path) as root_file:
        trota = root_file[TREE_NAME]
        if int(trota.num_entries) != expected_rows:
            raise RuntimeError(
                f"partial TROTA row count differs from fresh inference: "
                f"{trota.num_entries} != {expected_rows}"
            )
        branches = list(OUTPUT_DTYPES)
        rows_checked = 0
        maximum_float_absolute_difference = 0.0
        for start in range(0, expected_rows, step_size):
            stop = min(start + step_size, expected_rows)
            actual = trota.arrays(
                branches,
                entry_start=start,
                entry_stop=stop,
                library="np",
            )
            for name, expected_dtype in OUTPUT_DTYPES.items():
                values = np.asarray(actual[name])
                expected = expected_payload[name][start:stop]
                if values.dtype != expected_dtype:
                    raise RuntimeError(
                        f"partial TROTA branch {name} has dtype {values.dtype}, "
                        f"expected {expected_dtype}"
                    )
                if expected_dtype == np.dtype(np.float32):
                    if values.size:
                        maximum_float_absolute_difference = max(
                            maximum_float_absolute_difference,
                            float(np.max(np.abs(values - expected))),
                        )
                    if not np.allclose(values, expected, rtol=2e-5, atol=2e-6):
                        raise RuntimeError(
                            f"partial TROTA float branch {name} differs from fresh inference"
                        )
                elif not np.array_equal(values, expected):
                    raise RuntimeError(
                        f"partial TROTA identity branch {name} differs from fresh inference"
                    )
            rows_checked += stop - start
    return {
        "partial_tree_rows_compared": rows_checked,
        "partial_tree_matches_fresh_inference": True,
        "maximum_float_absolute_difference": maximum_float_absolute_difference,
    }


def _append_tree_with_hadd_repair(
    input_path: Path,
    selected_payload: dict[str, np.ndarray],
    *,
    selected_candidates: int,
    marker: dict[str, Any],
    target_year: int = DEFAULT_TARGET_YEAR,
) -> dict[str, Any]:
    """Repair an unreadable ROOT free-segment table with native ROOT fast cloning."""
    with tempfile.TemporaryDirectory(prefix="trota_hadd_repair_", dir=input_path.parent) as tmp:
        temporary = Path(tmp)
        sidecar = temporary / "trota.root"
        repaired = temporary / "repaired.root"
        with uproot.recreate(sidecar) as output_file:
            output_tree = output_file.mktree(
                TREE_NAME,
                {name: dtype for name, dtype in OUTPUT_DTYPES.items()},
                title=(
                    f"{target_year} TROTA TopResolved candidates passing the "
                    "1% QCD-mistag WP"
                ),
            )
            if selected_candidates:
                output_tree.extend(selected_payload)
            output_file[MARKER_NAME] = json.dumps(
                marker,
                sort_keys=True,
                allow_nan=False,
            )
        result = subprocess.run(
            ["hadd", "-f", str(repaired), str(input_path), str(sidecar)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0 or not repaired.is_file():
            raise RuntimeError(
                f"hadd repair failed with exit {result.returncode}: "
                f"{result.stdout[-4000:]}"
            )
        os.replace(repaired, input_path)
    return {
        "root_free_segments_repaired_with_hadd": True,
        "hadd_policy": "native ROOT fast clone of Events plus a separately written TROTA tree",
    }


def verify_complete_root(
    input_path: Path,
    *,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> dict[str, Any]:
    """Deep-check a completed ROOT without modifying it."""
    input_path = input_path.absolute()
    expected_schema_version = schema_version_for_year(target_year)
    state = inspect_existing_state(input_path, target_year=target_year)
    if state["status"] != "complete":
        raise RuntimeError("TROTA payload is not complete")
    marker = state["marker"]
    with uproot.open(input_path) as root_file:
        events = root_file["Events"]
        event_entries = int(events.num_entries)
        events_schema_digest = _events_schema_digest(events)
    validation = _validate_appended_tree(
        input_path,
        expected_event_entries=int(marker["events_entries"]),
        expected_schema_digest=str(marker["events_schema_digest"]),
        expected_selected_candidates=int(marker["selected_candidates"]),
    )
    if event_entries != int(marker["events_entries"]):
        raise RuntimeError("completion marker Events count does not match the ROOT file")
    if events_schema_digest != marker["events_schema_digest"]:
        raise RuntimeError("completion marker Events schema does not match the ROOT file")
    return {
        "schema_version": expected_schema_version,
        "status": "verified_complete",
        "input": str(input_path),
        "counts": {
            "events": event_entries,
            "candidates_evaluated": int(marker["candidates_evaluated"]),
            "selected_candidates": int(marker["selected_candidates"]),
            "events_with_selected_candidates": int(
                marker["events_with_selected_candidates"]
            ),
        },
        "storage": {"file_size": input_path.stat().st_size},
        "score_validation": validation,
        "marker": marker,
    }


def update_root_in_place(
    input_path: Path,
    model_path: Path,
    *,
    chunk_events: int,
    batch_size: int,
    use_numba: bool,
    recover_partial: bool = False,
    allow_hadd_repair: bool = False,
    target_year: int = DEFAULT_TARGET_YEAR,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = input_path.absolute()
    model_path = model_path.absolute()
    expected_schema_version = schema_version_for_year(target_year)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not os.access(input_path, os.W_OK):
        raise PermissionError(f"input ROOT is not writable: {input_path}")
    actual_model_hash = sha256(model_path)
    if actual_model_hash != TOPRESOLVED_2024_MODEL_SHA256:
        raise RuntimeError(
            f"unexpected TopResolved model hash {actual_model_hash}; "
            f"expected {TOPRESOLVED_2024_MODEL_SHA256}"
        )

    try:
        existing = inspect_existing_state(input_path, target_year=target_year)
    except RuntimeError as exc:
        if recover_partial and "exists without a completion marker" in str(exc):
            existing = {"status": "partial"}
        else:
            raise
    if existing["status"] == "complete":
        return {
            "schema_version": expected_schema_version,
            "status": "already_complete",
            "input": str(input_path),
            "marker": existing["marker"],
            "wall_time_seconds": time.perf_counter() - started,
        }

    file_size_before = input_path.stat().st_size
    with uproot.open(input_path) as root_file:
        if "Events" not in root_file:
            raise RuntimeError("Events tree is missing")
        tree = root_file["Events"]
        available = set(tree.keys())
        missing = sorted(set(MODEL_INPUT_BRANCHES) - available)
        if missing:
            raise RuntimeError(f"intermediate ROOT is missing TROTA inputs: {missing}")
        event_entries = int(tree.num_entries)
        events_schema_digest = _events_schema_digest(tree)

    import tensorflow as tf

    model = tf.keras.models.load_model(str(model_path), compile=False)
    builder = candidate_builder(use_numba=use_numba)
    selected_chunks = _empty_selected_chunks()
    events_processed = 0
    candidates_evaluated = 0
    selected_candidates = 0
    events_with_selected = 0
    maximum_probability_deviation = 0.0

    with uproot.open(input_path) as root_file:
        tree = root_file["Events"]
        for arrays in tree.iterate(
            list(MODEL_INPUT_BRANCHES),
            step_size=chunk_events,
            library="ak",
        ):
            number_of_chunk_events = len(arrays)
            flattened = _flatten_jets(arrays)
            candidate_payload = builder(*flattened)
            features = candidate_payload[1]
            if features.dtype != np.float32:
                raise RuntimeError(f"model features are not float32: {features.dtype}")
            number_of_candidates = int(features.shape[0])
            scores = np.empty((number_of_candidates, 3), dtype=np.float32)
            for start in range(0, number_of_candidates, batch_size):
                stop = min(start + batch_size, number_of_candidates)
                scores[start:stop] = np.asarray(
                    model({"jet": features[start:stop]}, training=False), dtype=np.float32
                )
            if not np.all(np.isfinite(scores)):
                raise RuntimeError("non-finite TopResolved model scores")
            if number_of_candidates:
                maximum_probability_deviation = max(
                    maximum_probability_deviation,
                    float(
                        np.max(
                            np.abs(
                                np.sum(scores, axis=1, dtype=np.float32)
                                - np.float32(1.0)
                            )
                        )
                    ),
                )
            denominator = scores[:, 1] + scores[:, 2]
            qcd_discriminant = np.divide(
                scores[:, 1],
                denominator,
                out=np.zeros(number_of_candidates, dtype=np.float32),
                where=denominator > np.float32(0.0),
            )
            selected = qcd_discriminant >= WP_THRESHOLD
            selected_in_chunk = _append_selected_chunk(
                selected_chunks,
                arrays,
                candidate_payload,
                scores,
                qcd_discriminant,
                selected,
            )
            if selected_in_chunk:
                selected_event_indices = np.unique(candidate_payload[2][selected])
                events_with_selected += int(selected_event_indices.size)
            events_processed += number_of_chunk_events
            candidates_evaluated += number_of_candidates
            selected_candidates += selected_in_chunk

    if events_processed != event_entries:
        raise RuntimeError(f"processed {events_processed} events, expected {event_entries}")
    selected_payload = _concatenate_selected(selected_chunks)
    if any(values.size != selected_candidates for values in selected_payload.values()):
        raise RuntimeError("unaligned sparse TROTA output arrays")

    recovery_validation: dict[str, Any] = {}
    repair_validation: dict[str, Any] = {}
    marker = {
        "schema_version": expected_schema_version,
        "status": "complete",
        "completed_at": now(),
        "application_year": target_year,
        "model_release_year": MODEL_RELEASE_YEAR,
        "model_sha256": actual_model_hash,
        "trota_commit": TROTA_COMMIT,
        "selected_working_point": SELECTED_TOPRESOLVED_2024_WORKING_POINT,
        "selected_threshold": float(WP_THRESHOLD),
        "selection_discriminant": "TTScore / (TTScore + QCDScore)",
        "storage_policy": "flat rows for passing candidates only",
        "float_policy": (
            "model features, model outputs, derived discriminant, stored scores, "
            "and stored candidate kinematics are float32; p4 accumulation is float64"
        ),
        "events_entries": event_entries,
        "candidates_evaluated": candidates_evaluated,
        "selected_candidates": selected_candidates,
        "events_with_selected_candidates": events_with_selected,
        "events_schema_digest": events_schema_digest,
    }
    if existing["status"] == "partial":
        recovery_validation = _validate_partial_matches_expected(
            input_path,
            selected_payload,
        )
        with uproot.update(input_path) as output_file:
            output_file[MARKER_NAME] = json.dumps(
                marker,
                sort_keys=True,
                allow_nan=False,
            )
    else:
        # The original Events tree is never rewritten. Uproot appends one sparse flat
        # tree to the existing file; all score and kinematic branches are float32.
        try:
            with uproot.update(input_path) as output_file:
                output_tree = output_file.mktree(
                    TREE_NAME,
                    {name: dtype for name, dtype in OUTPUT_DTYPES.items()},
                    title=(
                        f"{target_year} TROTA TopResolved candidates passing the "
                        "1% QCD-mistag WP"
                    ),
                )
                if selected_candidates:
                    output_tree.extend(selected_payload)
                output_file[MARKER_NAME] = json.dumps(
                    marker,
                    sort_keys=True,
                    allow_nan=False,
                )
        except AssertionError:
            if not allow_hadd_repair:
                raise
            repair_validation = _append_tree_with_hadd_repair(
                input_path,
                selected_payload,
                selected_candidates=selected_candidates,
                marker=marker,
                target_year=target_year,
            )

    validation = _validate_appended_tree(
        input_path,
        expected_event_entries=event_entries,
        expected_schema_digest=events_schema_digest,
        expected_selected_candidates=selected_candidates,
    )
    final_state = inspect_existing_state(input_path, target_year=target_year)
    if final_state["status"] != "complete":
        raise RuntimeError("TROTA completion marker validation failed")
    file_size_after = input_path.stat().st_size
    return {
        "schema_version": expected_schema_version,
        "status": (
            "recovered_complete" if existing["status"] == "partial" else "complete"
        ),
        "input": str(input_path),
        "model": {
            "path": str(model_path),
            "sha256": actual_model_hash,
            "trota_commit": TROTA_COMMIT,
            "release_year": MODEL_RELEASE_YEAR,
            "application_year": target_year,
        },
        "working_point": {
            "name": SELECTED_TOPRESOLVED_2024_WORKING_POINT,
            "threshold": float(WP_THRESHOLD),
            "discriminant": "TTScore / (TTScore + QCDScore)",
            "scale_factor_applied": False,
        },
        "runtime": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "awkward": ak.__version__,
            "uproot": uproot.__version__,
            "numba": use_numba,
            "chunk_events": chunk_events,
            "batch_size": batch_size,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "counts": {
            "events": event_entries,
            "candidates_evaluated": candidates_evaluated,
            "selected_candidates": selected_candidates,
            "events_with_selected_candidates": events_with_selected,
        },
        "storage": {
            "file_size_before": file_size_before,
            "file_size_after": file_size_after,
            "bytes_added": file_size_after - file_size_before,
            "tree": TREE_NAME,
            "rows_are_only_1pct_wp_candidates": True,
            "float_output_branches": list(FLOAT_OUTPUT_BRANCHES),
        },
        "score_validation": {
            "maximum_probability_sum_deviation_during_inference": (
                maximum_probability_deviation
            ),
            **validation,
            **recovery_validation,
            **repair_validation,
        },
        "marker": marker,
    }


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description=(
            "Append a sparse float32 TROTA TopResolved tree to an existing 2024/2025 "
            "intermediate ROOT. Only candidates passing the 1% QCD-mistag WP are stored."
        )
    )
    out.add_argument("--input", required=True, type=Path)
    out.add_argument("--model", required=True, type=Path)
    out.add_argument("--metadata-output", required=True, type=Path)
    out.add_argument("--chunk-events", type=int, default=20_000)
    out.add_argument("--batch-size", type=int, default=8192)
    out.add_argument(
        "--target-year",
        type=int,
        choices=SUPPORTED_TARGET_YEARS,
        default=DEFAULT_TARGET_YEAR,
        help="application year recorded in the schema and provenance marker",
    )
    out.add_argument("--no-numba", action="store_true")
    out.add_argument(
        "--recover-partial",
        action="store_true",
        help=(
            "re-run inference, require an existing unmarked TROTA tree to match, "
            "then append only the completion marker"
        ),
    )
    out.add_argument(
        "--verify-only",
        action="store_true",
        help="deep-check Events, TROTA, float32 branches, WP rows, and marker without writing",
    )
    out.add_argument(
        "--input-label",
        help="provenance label recorded in metadata when --input is a staged local copy",
    )
    out.add_argument(
        "--allow-hadd-repair",
        action="store_true",
        help=(
            "if uproot cannot deserialize the original free-segment table, rebuild "
            "the local staged file with ROOT hadd before validation"
        ),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.verify_only:
            report = verify_complete_root(args.input, target_year=args.target_year)
        else:
            report = update_root_in_place(
                args.input,
                args.model,
                chunk_events=args.chunk_events,
                batch_size=args.batch_size,
                use_numba=not args.no_numba,
                recover_partial=args.recover_partial,
                allow_hadd_repair=args.allow_hadd_repair,
                target_year=args.target_year,
            )
        if args.input_label:
            report["input"] = args.input_label
    except Exception as exc:
        report = {
            "schema_version": schema_version_for_year(args.target_year),
            "status": "failed",
            "input": str(args.input.absolute()),
            "failed_at": now(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        atomic_write_json(args.metadata_output, report)
        raise
    atomic_write_json(args.metadata_output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "input": report["input"],
                "selected_candidates": report.get("counts", {}).get(
                    "selected_candidates"
                ),
                "bytes_added": report.get("storage", {}).get("bytes_added"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
