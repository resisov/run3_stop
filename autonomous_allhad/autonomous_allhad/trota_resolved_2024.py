from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import awkward as ak
import numpy as np
import uproot


TROTA_COMMIT = "38fb282d5c3479d2eec96cc57d60fac7fd412d7f"
TOPRESOLVED_2024_MODEL_SHA256 = (
    "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
)
MODEL_INPUT_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    "entry",
    "dataset_id",
    "file_id",
    "jet_nanoaod_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_nanoaod_mass",
    "jet_area",
    "jet_btag_upart_all",
    "jet_id_all",
)
MODEL_CLASS_ORDER = ("FTScore", "TTScore", "QCDScore")
TOPRESOLVED_2024_QCD_WORKING_POINTS = {
    "10pct_qcd_mistag": 0.4694105386734009,
    "5pct_qcd_mistag": 0.7234321236610413,
    "1pct_qcd_mistag": 0.9433798789978027,
    "0p1pct_qcd_mistag": 0.9895168542861938,
}
SELECTED_TOPRESOLVED_2024_WORKING_POINT = "1pct_qcd_mistag"
TOPRESOLVED_2024_FT_WORKING_POINTS = {
    "10pct_ft_mistag": 0.7186799645423889,
    "5pct_ft_mistag": 0.8069701194763184,
    "1pct_ft_mistag": 0.9033783078193665,
    "0p1pct_ft_mistag": 0.9501141905784607,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delta_phi(phi1: float, phi2: float) -> float:
    value = float(phi1) - float(phi2)
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def sum_p4(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
) -> tuple[float, float, float, float]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(mass * mass + px * px + py * py + pz * pz, 0.0))
    sum_px = float(np.sum(px))
    sum_py = float(np.sum(py))
    sum_pz = float(np.sum(pz))
    sum_energy = float(np.sum(energy))
    sum_pt = math.hypot(sum_px, sum_py)
    sum_phi = math.atan2(sum_py, sum_px)
    sum_eta = math.asinh(sum_pz / sum_pt) if sum_pt > 0.0 else 0.0
    mass2 = sum_energy * sum_energy - sum_px * sum_px - sum_py * sum_py - sum_pz * sum_pz
    return sum_pt, sum_eta, sum_phi, math.sqrt(max(mass2, 0.0))


def selected_jet_indices(
    pt: Sequence[float], eta: Sequence[float], jet_id: Sequence[int]
) -> list[int]:
    if not (len(pt) == len(eta) == len(jet_id)):
        raise ValueError("unaligned jet pt/eta/id vectors")
    return [
        index
        for index, (jet_pt, jet_eta, passed_id) in enumerate(zip(pt, eta, jet_id))
        if bool(passed_id) and float(jet_pt) > 25.0 and abs(float(jet_eta)) < 2.5
    ]


def upstream_ordered_triplets(number_of_good_jets: int) -> list[tuple[int, int, int]]:
    return [
        (idx_j0, idx_j1, idx_j2)
        for idx_j0 in range(number_of_good_jets)
        for idx_j1 in range(idx_j0)
        for idx_j2 in range(idx_j1)
    ]


def build_event_candidates(
    pt: Sequence[float],
    eta: Sequence[float],
    phi: Sequence[float],
    mass: Sequence[float],
    area: Sequence[float],
    btag: Sequence[float],
    jet_id: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
    vectors = [pt, eta, phi, mass, area, btag, jet_id]
    if len({len(values) for values in vectors}) != 1:
        raise ValueError("unaligned intermediate jet vectors")

    pt_array = np.asarray(pt, dtype=np.float64)
    eta_array = np.asarray(eta, dtype=np.float64)
    phi_array = np.asarray(phi, dtype=np.float64)
    mass_array = np.asarray(mass, dtype=np.float64)
    area_array = np.asarray(area, dtype=np.float64)
    btag_array = np.asarray(btag, dtype=np.float64)
    good_indices = selected_jet_indices(pt_array, eta_array, jet_id)
    triplets = upstream_ordered_triplets(len(good_indices))
    features = np.zeros((len(triplets), 3, 8), dtype=np.float32)
    candidates: list[dict[str, Any]] = []

    for candidate_index, good_triplet in enumerate(triplets):
        source_indices = [good_indices[index] for index in good_triplet]
        source = np.asarray(source_indices, dtype=np.int64)
        candidate_pt, candidate_eta, candidate_phi, candidate_mass = sum_p4(
            pt_array[source], eta_array[source], phi_array[source], mass_array[source]
        )
        for leg_index, source_index in enumerate(source_indices):
            features[candidate_index, leg_index] = (
                area_array[source_index],
                btag_array[source_index],
                eta_array[source_index] - candidate_eta,
                mass_array[source_index],
                delta_phi(phi_array[source_index], candidate_phi),
                pt_array[source_index],
                delta_phi(phi_array[source_index], 0.0),
                eta_array[source_index],
            )
        candidates.append(
            {
                "good_jet_indices": list(good_triplet),
                "source_jet_indices": source_indices,
                "pt": candidate_pt,
                "eta": candidate_eta,
                "phi": candidate_phi,
                "mass": candidate_mass,
            }
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite TROTA TopResolved model input")
    return features, candidates, good_indices


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {name: None for name in ("min", "q25", "median", "mean", "q75", "max")}
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def evaluate_intermediate(
    input_path: str,
    model_path: Path,
    *,
    max_events: int,
    batch_size: int,
    sample_events: int,
    allow_unpinned_model: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    actual_model_hash = sha256(model_path)
    if not allow_unpinned_model and actual_model_hash != TOPRESOLVED_2024_MODEL_SHA256:
        raise RuntimeError(
            f"unexpected TopResolved model hash {actual_model_hash}; "
            f"expected {TOPRESOLVED_2024_MODEL_SHA256}"
        )

    with uproot.open(input_path) as root_file:
        if "Events" not in {str(key).split(";", 1)[0] for key in root_file.keys()}:
            raise RuntimeError("Events tree is missing")
        tree = root_file["Events"]
        available = set(tree.keys())
        missing = sorted(set(MODEL_INPUT_BRANCHES) - available)
        if missing:
            raise RuntimeError(f"intermediate ROOT is missing TROTA inputs: {missing}")
        entries_to_read = min(max(0, int(max_events)), int(tree.num_entries))
        arrays = tree.arrays(
            list(MODEL_INPUT_BRANCHES), entry_start=0, entry_stop=entries_to_read, library="ak"
        )
        total_tree_entries = int(tree.num_entries)

    event_candidate_counts: list[int] = []
    event_good_jet_counts: list[int] = []
    event_records: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    candidate_blocks: list[list[dict[str, Any]]] = []
    event_offsets = [0]

    vector_branches = (
        "jet_nanoaod_pt",
        "jet_eta_all",
        "jet_phi_all",
        "jet_nanoaod_mass",
        "jet_area",
        "jet_btag_upart_all",
        "jet_id_all",
    )
    for event_index in range(entries_to_read):
        vectors = [ak.to_list(arrays[name][event_index]) for name in vector_branches]
        features, candidates, good_indices = build_event_candidates(*vectors)
        feature_blocks.append(features)
        candidate_blocks.append(candidates)
        event_good_jet_counts.append(len(good_indices))
        event_candidate_counts.append(len(candidates))
        event_offsets.append(event_offsets[-1] + len(candidates))
        event_records.append(
            {
                "run": int(arrays["run"][event_index]),
                "luminosityBlock": int(arrays["luminosityBlock"][event_index]),
                "event": int(arrays["event"][event_index]),
                "entry": int(arrays["entry"][event_index]),
                "dataset_id": int(arrays["dataset_id"][event_index]),
                "file_id": int(arrays["file_id"][event_index]),
                "number_of_input_jets": len(vectors[0]),
                "number_of_good_jets": len(good_indices),
                "number_of_candidates": len(candidates),
            }
        )

    total_candidates = event_offsets[-1]
    if total_candidates == 0:
        raise RuntimeError("the selected intermediate event range contains no resolved candidates")
    all_features = np.concatenate(feature_blocks, axis=0)

    import tensorflow as tf

    model = tf.keras.models.load_model(str(model_path), compile=False)
    score_blocks = []
    for start in range(0, total_candidates, batch_size):
        stop = min(start + batch_size, total_candidates)
        score_blocks.append(
            np.asarray(model({"jet": all_features[start:stop]}, training=False), dtype=np.float64)
        )
    scores = np.concatenate(score_blocks, axis=0)
    if scores.shape != (total_candidates, 3):
        raise RuntimeError(f"unexpected model output shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("non-finite TopResolved model scores")

    probability_sums = np.sum(scores, axis=1)
    probability_deviation = np.abs(probability_sums - 1.0)
    tt_scores = scores[:, 1]
    ft_denominator = tt_scores + scores[:, 0]
    qcd_denominator = tt_scores + scores[:, 2]
    ft_discriminant = np.divide(
        tt_scores,
        ft_denominator,
        out=np.zeros_like(tt_scores),
        where=ft_denominator > 0.0,
    )
    qcd_discriminant = np.divide(
        tt_scores,
        qcd_denominator,
        out=np.zeros_like(tt_scores),
        where=qcd_denominator > 0.0,
    )
    events_with_candidates = int(np.count_nonzero(np.asarray(event_candidate_counts) > 0))
    qcd_working_point_results: dict[str, dict[str, float | int]] = {}
    for name, threshold in TOPRESOLVED_2024_QCD_WORKING_POINTS.items():
        candidate_pass = qcd_discriminant >= threshold
        event_pass_count = sum(
            bool(np.any(candidate_pass[event_offsets[index] : event_offsets[index + 1]]))
            for index in range(entries_to_read)
        )
        candidate_pass_count = int(np.count_nonzero(candidate_pass))
        qcd_working_point_results[name] = {
            "threshold": threshold,
            "candidate_pass_count": candidate_pass_count,
            "candidate_pass_fraction": candidate_pass_count / total_candidates,
            "event_at_least_one_pass_count": event_pass_count,
            "event_at_least_one_pass_fraction_all_events": event_pass_count / entries_to_read,
            "event_at_least_one_pass_fraction_events_with_candidates": (
                event_pass_count / events_with_candidates
            ),
        }
    sample: list[dict[str, Any]] = []
    for event_index, record in enumerate(event_records):
        start, stop = event_offsets[event_index], event_offsets[event_index + 1]
        if stop == start:
            continue
        event_scores = scores[start:stop]
        best_local = int(np.argmax(event_scores[:, 1]))
        best_candidate = dict(candidate_blocks[event_index][best_local])
        best_candidate["scores"] = {
            name: float(event_scores[best_local, score_index])
            for score_index, name in enumerate(MODEL_CLASS_ORDER)
        }
        sample.append({**record, "best_TTScore_candidate": best_candidate})
        if len(sample) >= sample_events:
            break

    winner_indices = np.argmax(scores, axis=1)
    report = {
        "schema_version": "trota_topresolved_intermediate_2024_test_v2",
        "status": "passed",
        "input": {
            "path": input_path,
            "tree": "Events",
            "tree_entries": total_tree_entries,
            "entries_tested": entries_to_read,
            "branches": list(MODEL_INPUT_BRANCHES),
            "jet_policy": "stored jet_id_all and NanoAOD pt>25 GeV, abs(eta)<2.5",
            "kinematic_policy": "jet_nanoaod_pt and jet_nanoaod_mass are model inputs",
        },
        "model": {
            "path": str(model_path),
            "sha256": actual_model_hash,
            "expected_sha256": TOPRESOLVED_2024_MODEL_SHA256,
            "hash_valid": actual_model_hash == TOPRESOLVED_2024_MODEL_SHA256,
            "trota_commit": TROTA_COMMIT,
            "class_order": list(MODEL_CLASS_ORDER),
            "input_shape": [None, 3, 8],
        },
        "runtime": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "awkward": ak.__version__,
            "uproot": uproot.__version__,
            "batch_size": batch_size,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "counts": {
            "events_tested": entries_to_read,
            "events_with_at_least_three_good_jets": int(
                np.count_nonzero(np.asarray(event_good_jet_counts) >= 3)
            ),
            "resolved_candidates": total_candidates,
            "good_jets_per_event": _quantiles(event_good_jet_counts),
            "candidates_per_event": _quantiles(event_candidate_counts),
        },
        "scores": {
            name: _quantiles(scores[:, index])
            for index, name in enumerate(MODEL_CLASS_ORDER)
        },
        "derived_discriminants": {
            "QCD_Score": {
                "definition": "TTScore / (TTScore + QCDScore)",
                "distribution": _quantiles(qcd_discriminant),
            },
            "FT_Score": {
                "definition": "TTScore / (TTScore + FTScore)",
                "distribution": _quantiles(ft_discriminant),
            },
        },
        "working_points": {
            "status": "training-study thresholds; 2024 scale factors are not applied",
            "source": (
                "TROTA NanoAODTools/python/postprocessing/training_studies/"
                "variables_studies_post_training.py"
            ),
            "selection_discriminant": "QCD_Score",
            "selected": SELECTED_TOPRESOLVED_2024_WORKING_POINT,
            "selected_threshold": TOPRESOLVED_2024_QCD_WORKING_POINTS[
                SELECTED_TOPRESOLVED_2024_WORKING_POINT
            ],
            "QCD_Score": qcd_working_point_results,
            "FT_Score_thresholds_not_applied": TOPRESOLVED_2024_FT_WORKING_POINTS,
        },
        "score_validation": {
            "finite": True,
            "minimum_score": float(np.min(scores)),
            "maximum_score": float(np.max(scores)),
            "maximum_probability_sum_deviation": float(np.max(probability_deviation)),
            "probability_sum_within_1e-5": bool(np.all(probability_deviation < 1.0e-5)),
            "winning_class_counts": {
                name: int(np.count_nonzero(winner_indices == index))
                for index, name in enumerate(MODEL_CLASS_ORDER)
            },
        },
        "sample_events": sample,
    }
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Test the official 2024 TROTA TopResolved model on a 2024 intermediate ROOT."
    )
    out.add_argument("--input", required=True, help="2024 intermediate ROOT path")
    out.add_argument("--model", required=True, type=Path, help="official TopResolved 2024 HDF5 model")
    out.add_argument("--output", required=True, type=Path, help="machine-readable JSON report")
    out.add_argument("--max-events", type=int, default=1000)
    out.add_argument("--batch-size", type=int, default=8192)
    out.add_argument("--sample-events", type=int, default=10)
    out.add_argument("--allow-unpinned-model", action="store_true")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = evaluate_intermediate(
        args.input,
        args.model,
        max_events=args.max_events,
        batch_size=args.batch_size,
        sample_events=args.sample_events,
        allow_unpinned_model=args.allow_unpinned_model,
    )
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "events_tested": report["counts"]["events_tested"],
        "resolved_candidates": report["counts"]["resolved_candidates"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
