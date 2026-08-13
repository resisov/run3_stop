#!/usr/bin/env python3
"""Full exact-selected-entry GCR prompt-photon overlap scan for 2024.

This is a read-only diagnostic.  It obtains the event set exclusively from
the already materialized ``feature_GCR`` rows produced by
``real_subset_worker.py``.  It then joins only those selected rows back to the
source NanoAOD by stable ``file_id`` and original ``entry``.  It never
reconstructs the GCR selection and never writes to nominal inputs.

The representative scanner remains the source of the selection/join and
generator-object definitions.  This program removes its sampling limits,
adds resumable per-source-file generator caches, and evaluates weighted
stability quantities needed before any GJets/QCD stitching decision.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
import uproot

import scan_gcr_prompt_overlap_2024 as base


RADII = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
UT_EDGES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0]
)
PHOTON_PT_EDGES = np.asarray(
    [220.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 3000.0]
)
HT_EDGES = np.asarray(
    [300.0, 500.0, 700.0, 900.0, 1200.0, 1500.0, 2000.0, 3000.0]
)
GENERATOR_BINVAR_EDGES = np.asarray(
    [
        0.0,
        170.0,
        300.0,
        470.0,
        600.0,
        800.0,
        1000.0,
        1500.0,
        2000.0,
        3000.0,
        10000.0,
    ]
)
BOUNDARY_HALF_WIDTH = 0.025


def expected_generator_groups() -> set[str]:
    return {
        *(f"GJ_PTG_{item}" for item in base.GJ_BINS),
        *(f"QCD_PT_{item}" for item in base.QCD_BINS),
    }


def collect_manifest_recovery_candidates(
    shard_bundle: Path,
    missing_groups: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Recover missing sidecar groups from the frozen nominal shard manifest."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with tarfile.open(shard_bundle, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "/mc_shard_" not in member.name:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            shard = json.load(extracted)
            for record in shard.get("records") or []:
                dataset = str(record.get("dataset") or "")
                file_path = str(record.get("file_path") or "")
                if not file_path:
                    continue
                for process in ("GJ", "QCD"):
                    label = base.group_label(dataset, process)
                    if label not in missing_groups:
                        continue
                    leaf = grouped[label].setdefault(
                        file_path,
                        {
                            "file_path": file_path,
                            "fake_sidecar_dataset": dataset,
                            "process": process,
                            "sidecar_selected_events": 0,
                            "sidecar_events_read": 0,
                            "sidecar_segments": 0,
                            "candidate_source": (
                                "nominal_shard_manifest_recovery"
                            ),
                        },
                    )
                    leaf["sidecar_segments"] += 1
    return {
        label: sorted(files.values(), key=lambda item: item["file_path"])
        for label, files in grouped.items()
    }


def merge_weight_status(
    target: dict[str, Any], addition: dict[str, Any]
) -> None:
    for dataset_id, record in addition.items():
        if dataset_id not in target:
            target[dataset_id] = record
            continue
        target[dataset_id]["events"] = int(
            target[dataset_id].get("events") or 0
        ) + int(record.get("events") or 0)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(partial, path)


def atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    os.replace(partial, path)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def event_cache_key(file_path: str) -> str:
    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()


def preferred_urls(url: str, local_eos_only: bool = False) -> list[str]:
    marker = "//store/"
    if marker not in url:
        return [url]
    suffix = "/store/" + url.split(marker, 1)[1]
    if local_eos_only:
        return ["/eos/cms" + suffix]
    ordered = [
        "/eos/cms" + suffix,
        url,
        "root://eoscms.cern.ch/" + suffix,
        "root://xrootd-cms.infn.it/" + suffix,
        "root://cmsxrootd.fnal.gov/" + suffix,
    ]
    return list(dict.fromkeys(ordered))


class SelectedEventTree:
    """Expose one event from an in-memory batch through the TTree interface."""

    def __init__(self, arrays: Any, index: int):
        self.arrays_in_memory = arrays
        self.index = int(index)

    def keys(self) -> list[str]:
        return list(ak.fields(self.arrays_in_memory))

    def arrays(
        self,
        branches: list[str],
        entry_start: int,
        entry_stop: int,
        library: str,
    ) -> Any:
        del entry_start, entry_stop, library
        return self.arrays_in_memory[branches][self.index : self.index + 1]


def selected_entry_batches(
    rows: list[dict[str, Any]],
    max_gap: int = 1000,
    max_span: int = 20000,
) -> list[list[dict[str, Any]]]:
    """Cluster selected entries to avoid one NanoAOD read call per event."""
    ordered = sorted(rows, key=lambda item: int(item["entry"]))
    if not ordered:
        return []
    output: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    start = int(ordered[0]["entry"])
    previous = start
    for row in ordered[1:]:
        entry = int(row["entry"])
        if entry - previous > max_gap or entry - start >= max_span:
            output.append(current)
            current = [row]
            start = entry
        else:
            current.append(row)
        previous = entry
    output.append(current)
    return output


def compact_gen_update(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "group",
        "process",
        "file_path",
        "file_id",
        "dataset",
        "dataset_id",
        "nominal_root",
        "flat_index",
        "entry",
        "event",
        "recoil_gcr",
        "ht_photon_clean",
        "ut_bin_index",
        "ut_bin",
        "gen_weight_flat",
        "analysis_nominal_weight",
        "normalized_gen_weight",
        "photon_medium_pt",
        "photon_medium_eta",
        "photon_medium_phi",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def restore_file_cache(
    rows: list[dict[str, Any]], cache_path: Path
) -> tuple[bool, dict[str, Any] | None]:
    if not cache_path.is_file():
        return False, None
    try:
        payload = base.read_json(cache_path)
        if payload.get("file_path") != rows[0]["file_path"]:
            return False, None
        if (payload.get("file_status") or {}).get("status") != "complete":
            return False, None
        cached = {
            (int(item["entry"]), int(item["event"])): item["diagnostic"]
            for item in payload.get("events") or []
        }
        if len(cached) != len(rows):
            return False, None
        for row in rows:
            key = (int(row["entry"]), int(row["event"]))
            if key not in cached:
                return False, None
            row.update(cached[key])
        return True, payload.get("file_status") or {}
    except Exception:
        return False, None


def scan_one_source_file(
    file_path: str,
    rows: list[dict[str, Any]],
    cache_path: Path,
    local_eos_only: bool = False,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    root_file = None
    errors: list[str] = []
    used_url = None
    for url in preferred_urls(file_path, local_eos_only=local_eos_only):
        try:
            root_file = uproot.open(url, timeout=60)
            used_url = url
            break
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    updates: list[dict[str, Any]] = []
    if root_file is None:
        status = {
            "status": "open_failed",
            "events_requested": len(rows),
            "errors": errors,
        }
        for row in rows:
            updates.append(
                {
                    "entry": int(row["entry"]),
                    "event": int(row["event"]),
                    "diagnostic": {
                        "gen_diagnostic_status": "file_open_failed"
                    },
                }
            )
    else:
        successes = 0
        failures: list[dict[str, Any]] = []
        try:
            tree = root_file["Events"]
            needed = [
                "Photon_pt",
                "Photon_eta",
                "Photon_phi",
                "Photon_genPartIdx",
                "Photon_genPartFlav",
                "GenPart_pt",
                "GenPart_eta",
                "GenPart_phi",
                "GenPart_pdgId",
                "GenPart_status",
                "GenPart_statusFlags",
                "GenPart_genPartIdxMother",
                "Generator_binvar",
                "LHE_HT",
            ]
            if rows[0]["process"] == "GJ":
                needed.extend(
                    [
                        "LHEPart_pt",
                        "LHEPart_eta",
                        "LHEPart_phi",
                        "LHEPart_pdgId",
                        "LHEPart_status",
                    ]
                )
            present = set(tree.keys())
            branches = [name for name in needed if name in present]
            for batch in selected_entry_batches(rows):
                batch_start = int(batch[0]["entry"])
                batch_stop = int(batch[-1]["entry"]) + 1
                try:
                    arrays = tree.arrays(
                        branches,
                        entry_start=batch_start,
                        entry_stop=batch_stop,
                        library="ak",
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    for row in batch:
                        failures.append(
                            {
                                "entry": int(row["entry"]),
                                "event": int(row["event"]),
                                "error": error,
                            }
                        )
                        updates.append(
                            {
                                "entry": int(row["entry"]),
                                "event": int(row["event"]),
                                "diagnostic": {
                                    "gen_diagnostic_status": "event_failed",
                                    "gen_diagnostic_error": error,
                                },
                            }
                        )
                    continue
                for row in batch:
                    diagnostic: dict[str, Any] = {}
                    try:
                        event_index = int(row["entry"]) - batch_start
                        view = SelectedEventTree(arrays, event_index)
                        diagnostic.update(
                            base.gen_diagnostic_for_event(view, row)
                        )
                        diagnostic["gen_diagnostic_status"] = "complete"
                        successes += 1
                    except Exception as exc:
                        diagnostic["gen_diagnostic_status"] = "event_failed"
                        diagnostic["gen_diagnostic_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        failures.append(
                            {
                                "entry": int(row["entry"]),
                                "event": int(row["event"]),
                                "error": diagnostic[
                                    "gen_diagnostic_error"
                                ],
                            }
                        )
                    updates.append(
                        {
                            "entry": int(row["entry"]),
                            "event": int(row["event"]),
                            "diagnostic": diagnostic,
                        }
                    )
        finally:
            root_file.close()
        status = {
            "status": "complete" if successes == len(rows) else "partial",
            "used_url": used_url,
            "events_requested": len(rows),
            "events_complete": successes,
            "failures": failures,
            "open_errors_before_success": errors,
        }

    atomic_json(
        cache_path,
        {
            "schema_version": "gcr_selected_gen_file_cache_v1",
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "file_path": file_path,
            "file_status": status,
            "events": updates,
        },
    )
    return file_path, status, updates


def attach_gen_diagnostics_resumable(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    workers: int,
    local_eos_only: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["file_path"])].append(row)

    file_status: dict[str, Any] = {}
    pending: list[tuple[str, list[dict[str, Any]], Path]] = []
    restored = 0
    for file_path, file_rows in sorted(by_file.items()):
        cache_path = cache_dir / f"{event_cache_key(file_path)}.json"
        ok, status = restore_file_cache(file_rows, cache_path)
        if ok:
            restored += 1
            file_status[file_path] = status
        else:
            pending.append((file_path, file_rows, cache_path))

    completed = 0
    started = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    scan_one_source_file,
                    file_path,
                    file_rows,
                    cache_path,
                    local_eos_only,
                ): (file_path, file_rows)
                for file_path, file_rows, cache_path in pending
            }
            for future in as_completed(futures):
                file_path, file_rows = futures[future]
                try:
                    _, status, updates = future.result()
                except Exception as exc:
                    status = {
                        "status": "worker_failed",
                        "events_requested": len(file_rows),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    updates = [
                        {
                            "entry": int(row["entry"]),
                            "event": int(row["event"]),
                            "diagnostic": {
                                "gen_diagnostic_status": "worker_failed",
                                "gen_diagnostic_error": status["error"],
                            },
                        }
                        for row in file_rows
                    ]
                update_by_key = {
                    (int(item["entry"]), int(item["event"])): item[
                        "diagnostic"
                    ]
                    for item in updates
                }
                for row in file_rows:
                    row.update(
                        update_by_key[(int(row["entry"]), int(row["event"]))]
                    )
                file_status[file_path] = status
                completed += 1
                if completed == 1 or completed % 20 == 0:
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (
                        (len(pending) - completed) / rate if rate > 0 else None
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "gen_join",
                                "new_files_complete": completed,
                                "new_files_total": len(pending),
                                "cache_files_restored": restored,
                                "rate_files_per_s": round(rate, 4),
                                "eta_s": (
                                    round(remaining, 1)
                                    if remaining is not None
                                    else None
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    return file_status, {
        "source_files_with_selected_rows": len(by_file),
        "cache_files_restored": restored,
        "source_files_scanned_now": completed,
    }


def eligible_prompt(row: dict[str, Any]) -> bool:
    dr = finite_number(row.get("min_dr_status23_parton"))
    return bool(
        row.get("gen_diagnostic_status") == "complete"
        and row.get("prompt_flavour")
        and row.get("valid_gen_match")
        and row.get("gen_photon_pdgId") == 22
        and row.get("gen_photon_status") == 1
        and dr is not None
    )


def event_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "process": row["process"],
        "generator_bin": row["group"],
        "dataset": row["dataset"],
        "file_path": row["file_path"],
        "entry": int(row["entry"]),
        "event": int(row["event"]),
        "weight": finite_number(row.get("analysis_nominal_weight")),
        "min_dr_status23_parton": finite_number(
            row.get("min_dr_status23_parton")
        ),
        "photon_pt": (
            finite_number((row.get("photon_medium_pt") or [None])[0])
        ),
        "ut": finite_number(row.get("recoil_gcr")),
        "ht_photon_clean": finite_number(row.get("ht_photon_clean")),
        "generator_binvar": finite_number(row.get("generator_binvar")),
    }


def jackknife_relative(weights: np.ndarray) -> float | None:
    count = len(weights)
    total = float(np.sum(weights))
    if count < 2 or total == 0:
        return None
    leave_one_out = total - weights
    mean = float(np.mean(leave_one_out))
    variance = float(
        ((count - 1.0) / count)
        * np.sum(np.square(leave_one_out - mean))
    )
    return math.sqrt(max(variance, 0.0)) / abs(total)


def weighted_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = []
    weights = []
    for row in rows:
        weight = finite_number(row.get("analysis_nominal_weight"))
        if weight is None:
            continue
        valid_rows.append(row)
        weights.append(weight)
    array = np.asarray(weights, dtype=float)
    if len(array) == 0:
        return {
            "events": 0,
            "source_files": 0,
            "sumw": 0.0,
            "sumw2": 0.0,
            "sum_abs_w": 0.0,
            "neff_signed": None,
            "neff_abs": None,
        }

    sumw = float(np.sum(array))
    sumw2 = float(np.sum(np.square(array)))
    abs_weights = np.abs(array)
    sum_abs = float(np.sum(abs_weights))
    order = np.argsort(abs_weights)[::-1]
    top_indices = order[: min(10, len(order))]

    file_signed: dict[str, float] = defaultdict(float)
    file_abs: dict[str, float] = defaultdict(float)
    for row, weight in zip(valid_rows, array):
        file_path = str(row["file_path"])
        file_signed[file_path] += float(weight)
        file_abs[file_path] += abs(float(weight))
    file_signed_values = np.asarray(list(file_signed.values()), dtype=float)
    max_file = (
        max(file_abs, key=file_abs.get) if file_abs else None
    )
    max_abs_weight = float(abs_weights[order[0]])
    loo_signed_ratios = (
        (sumw - array) / sumw if sumw != 0 else np.asarray([])
    )
    file_loo_signed_ratios = (
        (sumw - file_signed_values) / sumw
        if sumw != 0
        else np.asarray([])
    )
    return {
        "events": len(array),
        "source_files": len(file_signed),
        "sumw": sumw,
        "sumw2": sumw2,
        "sum_abs_w": sum_abs,
        "negative_weight_events": int(np.sum(array < 0)),
        "negative_weight_fraction": float(np.mean(array < 0)),
        "neff_signed": sumw * sumw / sumw2 if sumw2 > 0 else None,
        "neff_abs": sum_abs * sum_abs / sumw2 if sumw2 > 0 else None,
        "max_event_abs_weight": max_abs_weight,
        "max_event_fraction_abs_sumw": (
            max_abs_weight / sum_abs if sum_abs > 0 else None
        ),
        "max_event_to_abs_signed_sumw": (
            max_abs_weight / abs(sumw) if sumw != 0 else None
        ),
        "top3_event_fraction_abs_sumw": (
            float(np.sum(abs_weights[order[:3]])) / sum_abs
            if sum_abs > 0
            else None
        ),
        "top10_event_fraction_abs_sumw": (
            float(np.sum(abs_weights[top_indices])) / sum_abs
            if sum_abs > 0
            else None
        ),
        "event_leave_one_out_signed_ratio_min": (
            float(np.min(loo_signed_ratios))
            if len(loo_signed_ratios)
            else None
        ),
        "event_leave_one_out_signed_ratio_max": (
            float(np.max(loo_signed_ratios))
            if len(loo_signed_ratios)
            else None
        ),
        "event_jackknife_relative_sigma": jackknife_relative(array),
        "max_source_file_fraction_abs_sumw": (
            file_abs[max_file] / sum_abs
            if max_file is not None and sum_abs > 0
            else None
        ),
        "max_source_file_to_abs_signed_sumw": (
            abs(file_signed[max_file]) / abs(sumw)
            if max_file is not None and sumw != 0
            else None
        ),
        "file_leave_one_out_signed_ratio_min": (
            float(np.min(file_loo_signed_ratios))
            if len(file_loo_signed_ratios)
            else None
        ),
        "file_leave_one_out_signed_ratio_max": (
            float(np.max(file_loo_signed_ratios))
            if len(file_loo_signed_ratios)
            else None
        ),
        "file_jackknife_relative_sigma": jackknife_relative(
            file_signed_values
        ),
        "largest_source_file": max_file,
        "top_abs_weight_events": [
            event_identity(valid_rows[int(index)]) for index in top_indices
        ],
    }


def summarize_policy(
    rows: list[dict[str, Any]], process: str, radius: float
) -> dict[str, Any]:
    eligible = [row for row in rows if eligible_prompt(row)]
    if process == "GJ":
        kept = [
            row
            for row in eligible
            if float(row["min_dr_status23_parton"]) >= radius
        ]
        policy = "direct: min_dR(gen photon, status-23 q/g) >= R"
    else:
        kept = [
            row
            for row in eligible
            if float(row["min_dr_status23_parton"]) < radius
        ]
        policy = "fragmentation: min_dR(gen photon, status-23 q/g) < R"
    boundary = [
        row
        for row in eligible
        if abs(float(row["min_dr_status23_parton"]) - radius)
        < BOUNDARY_HALF_WIDTH
    ]
    total = weighted_stability(eligible)
    surviving = weighted_stability(kept)
    boundary_stats = weighted_stability(boundary)
    return {
        "policy": policy,
        "eligible": total,
        "surviving": surviving,
        "survival_unweighted": (
            len(kept) / len(eligible) if eligible else None
        ),
        "survival_signed_weighted": (
            surviving["sumw"] / total["sumw"]
            if total["sumw"] != 0
            else None
        ),
        "survival_abs_weighted": (
            surviving["sum_abs_w"] / total["sum_abs_w"]
            if total["sum_abs_w"] != 0
            else None
        ),
        "boundary_window": {
            "half_width": BOUNDARY_HALF_WIDTH,
            "events": boundary_stats["events"],
            "sumw": boundary_stats["sumw"],
            "sum_abs_w": boundary_stats["sum_abs_w"],
            "fraction_abs_sumw": (
                boundary_stats["sum_abs_w"] / total["sum_abs_w"]
                if total["sum_abs_w"] != 0
                else None
            ),
        },
    }


def value_from_row(row: dict[str, Any], field: str) -> float | None:
    if field == "photon_pt":
        values = row.get("photon_medium_pt") or []
        return finite_number(values[0]) if len(values) == 1 else None
    return finite_number(row.get(field))


def bin_label(low: float, high: float) -> str:
    return f"{low:g}to{high:g}"


def partition_by_edges(
    rows: list[dict[str, Any]], field: str, edges: np.ndarray
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        output[bin_label(float(low), float(high))] = [
            row
            for row in rows
            if (value := value_from_row(row, field)) is not None
            and low <= value < high
        ]
    output["overflow"] = [
        row
        for row in rows
        if (value := value_from_row(row, field)) is not None
        and value >= float(edges[-1])
    ]
    return output


def summarize_axis(
    partitions: dict[str, list[dict[str, Any]]],
    process: str,
    radius: float,
) -> dict[str, Any]:
    return {
        label: summarize_policy(items, process, radius)
        for label, items in partitions.items()
    }


def dr_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if eligible_prompt(row)]
    dr = np.asarray(
        [float(row["min_dr_status23_parton"]) for row in eligible],
        dtype=float,
    )
    weights = np.asarray(
        [float(row["analysis_nominal_weight"]) for row in eligible],
        dtype=float,
    )
    edges = np.asarray(
        [
            0.0,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.60,
            0.80,
            1.00,
            1.50,
            2.00,
            3.20,
            6.40,
        ]
    )
    return {
        "edges": edges.tolist(),
        "unweighted": np.histogram(dr, bins=edges)[0].astype(int).tolist(),
        "sumw": np.histogram(dr, bins=edges, weights=weights)[0].tolist(),
        "sumw2": np.histogram(
            dr, bins=edges, weights=np.square(weights)
        )[0].tolist(),
        "sum_abs_w": np.histogram(
            dr, bins=edges, weights=np.abs(weights)
        )[0].tolist(),
        "underflow_events": int(np.sum(dr < edges[0])),
        "overflow_events": int(np.sum(dr >= edges[-1])),
        "minimum": float(np.min(dr)) if len(dr) else None,
        "median": float(np.median(dr)) if len(dr) else None,
        "maximum": float(np.max(dr)) if len(dr) else None,
    }


def build_full_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "process": {},
        "generator_dataset_bin": {},
    }
    for process in ("GJ", "QCD"):
        process_rows = [row for row in rows if row["process"] == process]
        process_payload: dict[str, Any] = {
            "dr_distribution": dr_distribution(process_rows),
            "pre_policy_stability": weighted_stability(
                [row for row in process_rows if eligible_prompt(row)]
            ),
            "radii": {},
        }
        for radius in RADII:
            process_payload["radii"][f"{radius:.2f}"] = {
                "inclusive": summarize_policy(
                    process_rows, process, radius
                ),
                "by_ut": summarize_axis(
                    partition_by_edges(
                        process_rows, "recoil_gcr", UT_EDGES
                    ),
                    process,
                    radius,
                ),
                "by_photon_pt": summarize_axis(
                    partition_by_edges(
                        process_rows, "photon_pt", PHOTON_PT_EDGES
                    ),
                    process,
                    radius,
                ),
                "by_ht_photon_clean": summarize_axis(
                    partition_by_edges(
                        process_rows, "ht_photon_clean", HT_EDGES
                    ),
                    process,
                    radius,
                ),
                "by_generator_binvar": summarize_axis(
                    partition_by_edges(
                        process_rows,
                        "generator_binvar",
                        GENERATOR_BINVAR_EDGES,
                    ),
                    process,
                    radius,
                ),
            }
        result["process"][process] = process_payload

    for label in sorted(set(row["group"] for row in rows)):
        group_rows = [row for row in rows if row["group"] == label]
        process = str(group_rows[0]["process"])
        result["generator_dataset_bin"][label] = {
            "process": process,
            "dr_distribution": dr_distribution(group_rows),
            "pre_policy_stability": weighted_stability(
                [row for row in group_rows if eligible_prompt(row)]
            ),
            "radii": {
                f"{radius:.2f}": summarize_policy(
                    group_rows, process, radius
                )
                for radius in RADII
            },
        }
    return result


def assign_flat_weights_full(
    repo: Path,
    rows: list[dict[str, Any]],
    builder: Any,
    normalization: dict[str, Any],
    step_size: int = 50000,
    targeted: bool = False,
) -> tuple[int, dict[str, Any], list[str]]:
    """Read flat rows in chunks and attach the trusted nominal weight bundle."""
    from autonomous_allhad.real_subset_worker import compute_weight_bundle

    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_root[str(row["nominal_root"])].append(row)
    attached = 0
    weight_status: dict[str, Any] = {}
    failures: list[str] = []
    branches = sorted(
        set(builder.WEIGHT_BRANCHES)
        | {
            "file_id",
            "dataset_id",
            "run",
            "luminosityBlock",
            "entry",
            "event",
            "feature_GCR",
            "recoil_gcr",
            "ht_photon_clean",
            "photon_medium_pt",
            "photon_medium_eta",
            "photon_medium_phi",
        }
    )
    for root_index, (root_path, root_rows) in enumerate(
        sorted(rows_by_root.items()), start=1
    ):
        row_by_key = {
            (
                int(row["file_id"]),
                int(row["entry"]),
                int(row["event"]),
            ): row
            for row in root_rows
        }
        seen: set[tuple[int, int, int]] = set()
        try:
            with uproot.open(root_path) as root_file:
                tree = root_file["Events"]
                present = set(tree.keys())
                identity_branches = {
                    "file_id",
                    "dataset_id",
                    "run",
                    "luminosityBlock",
                    "entry",
                    "event",
                    "feature_GCR",
                    "recoil_gcr",
                    "ht_photon_clean",
                    "photon_medium_pt",
                    "photon_medium_eta",
                    "photon_medium_phi",
                }
                missing_identity = sorted(identity_branches - present)
                if missing_identity:
                    raise RuntimeError(
                        "missing flat identity branches: "
                        + ", ".join(missing_identity)
                    )
                selected_branches = [
                    branch for branch in branches if branch in present
                ]
                if targeted:
                    chunk_iterator = (
                        tree.arrays(
                            selected_branches,
                            entry_start=int(batch[0]["flat_index"]),
                            entry_stop=int(batch[-1]["flat_index"]) + 1,
                            library="ak",
                        )
                        for batch in flat_identity_batches(
                            root_rows,
                            max_gap=1000,
                            max_span=20000,
                        )
                    )
                else:
                    chunk_iterator = tree.iterate(
                        selected_branches,
                        step_size=step_size,
                        library="ak",
                    )
                for chunk in chunk_iterator:
                    n = len(chunk["dataset_id"])
                    if n == 0:
                        continue
                    file_ids = np.asarray(chunk["file_id"], dtype=np.int64)
                    entries = np.asarray(chunk["entry"], dtype=np.int64)
                    events = np.asarray(chunk["event"], dtype=np.int64)
                    gcr = np.asarray(chunk["feature_GCR"], dtype=bool)
                    keys = [
                        (int(file_id), int(entry), int(event))
                        for file_id, entry, event in zip(
                            file_ids, entries, events
                        )
                    ]
                    selected = np.asarray(
                        [
                            bool(is_gcr and key in row_by_key)
                            for key, is_gcr in zip(keys, gcr)
                        ],
                        dtype=bool,
                    )
                    if not np.any(selected):
                        continue
                    selected_chunk = chunk[selected]
                    selected_keys = [
                        key for key, keep in zip(keys, selected) if keep
                    ]
                    dsids = np.asarray(
                        selected_chunk["dataset_id"], dtype=np.int64
                    )
                    for dataset_id in sorted(
                        set(int(value) for value in dsids)
                    ):
                        dataset_mask = dsids == dataset_id
                        sub = {
                            name: selected_chunk[name][dataset_mask]
                            for name in ak.fields(selected_chunk)
                        }
                        sub_keys = [
                            key
                            for key, keep in zip(selected_keys, dataset_mask)
                            if keep
                        ]
                        reference = row_by_key[sub_keys[0]]
                        dataset = str(reference["dataset"])
                        process = str(reference["process"])
                        arrays, inputs = builder.flat_arrays_for_weights(sub)
                        year_values = np.asarray(sub["year"], dtype=int)
                        year = (
                            str(int(year_values[0]))
                            if len(year_values)
                            else "2024"
                        )
                        normv = builder.norm_vector(
                            normalization,
                            sub,
                            dataset_id,
                            False,
                            False,
                            require_normalization=True,
                        )
                        try:
                            _gen, variations, sf_status = (
                                compute_weight_bundle(
                                    arrays,
                                    repo,
                                    dataset,
                                    process,
                                    year,
                                    inputs["n"],
                                    inputs["jet_pt"],
                                    inputs["jet_eta"],
                                    inputs["jet_hadflav"],
                                    inputs["b_med"],
                                    inputs["e_eta"],
                                    inputs["e_delta_eta_sc"],
                                    inputs["e_pt"],
                                    inputs["e_phi"],
                                    inputs["e_veto"],
                                    inputs["e_med"],
                                    inputs["n_e_veto"],
                                    inputs["n_e_med"],
                                    inputs["m_eta"],
                                    inputs["m_pt"],
                                    inputs["m_phi"],
                                    inputs["m_loose"],
                                    inputs["m_med"],
                                    inputs["n_m_loose"],
                                    inputs["n_m_med"],
                                    inputs["p_eta"],
                                    inputs["p_pt"],
                                    inputs["p_phi"],
                                    inputs["p_med"],
                                    inputs["gcr_mask"],
                                )
                            )
                            nominal = (
                                np.asarray(
                                    variations["nominal"], dtype=float
                                )
                                * normv
                            )
                            status_label = (
                                "nominal_sf_and_normalization_applied"
                            )
                            error = None
                        except Exception as exc:
                            nominal = (
                                np.asarray(sub["gen_weight"], dtype=float)
                                * normv
                            )
                            sf_status = {
                                "applied": False,
                                "error": "fallback_raw_gen_weight",
                            }
                            status_label = "fallback_normalized_gen_weight"
                            error = f"{type(exc).__name__}: {exc}"
                        raw_normalized = (
                            np.asarray(sub["gen_weight"], dtype=float) * normv
                        )
                        record = weight_status.setdefault(
                            str(dataset_id),
                            {
                                "dataset": dataset,
                                "process": process,
                                "events": 0,
                                "status": status_label,
                                "normalization_factor": (
                                    float(normv[0]) if len(normv) else None
                                ),
                                "scale_factor_status": sf_status,
                            },
                        )
                        record["events"] += len(sub_keys)
                        if error is not None:
                            record["error"] = error
                        photon_pts = ak.to_list(sub["photon_medium_pt"])
                        photon_etas = ak.to_list(sub["photon_medium_eta"])
                        photon_phis = ak.to_list(sub["photon_medium_phi"])
                        runs = np.asarray(sub["run"], dtype=np.int64)
                        luminosity_blocks = np.asarray(
                            sub["luminosityBlock"], dtype=np.int64
                        )
                        for (
                            key,
                            weight,
                            raw_weight,
                            run,
                            luminosity_block,
                            photon_pt,
                            photon_eta,
                            photon_phi,
                        ) in zip(
                            sub_keys,
                            nominal,
                            raw_normalized,
                            runs,
                            luminosity_blocks,
                            photon_pts,
                            photon_etas,
                            photon_phis,
                        ):
                            row = row_by_key[key]
                            row["analysis_nominal_weight"] = float(weight)
                            row["normalized_gen_weight"] = float(raw_weight)
                            row["run"] = int(run)
                            row["luminosityBlock"] = int(
                                luminosity_block
                            )
                            row["photon_medium_pt"] = photon_pt
                            row["photon_medium_eta"] = photon_eta
                            row["photon_medium_phi"] = photon_phi
                            seen.add(key)
                            attached += 1
        except Exception as exc:
            failures.append(
                f"{root_path}: {type(exc).__name__}: {exc}"
            )
        missing_keys = sorted(set(row_by_key) - seen)
        if missing_keys:
            failures.append(
                f"{root_path}: {len(missing_keys)} exact GCR rows were not "
                "matched while attaching flat weights"
            )
        if root_index == 1 or root_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "flat_weight_attachment",
                        "roots_complete": root_index,
                        "roots_total": len(rows_by_root),
                        "rows_attached": attached,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return attached, weight_status, failures


def flat_identity_batches(
    rows: list[dict[str, Any]],
    max_gap: int = 10000,
    max_span: int = 250000,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: int(item["flat_index"]))
    if not ordered:
        return []
    output: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    start = int(ordered[0]["flat_index"])
    previous = start
    for row in ordered[1:]:
        index = int(row["flat_index"])
        if index - previous > max_gap or index - start >= max_span:
            output.append(current)
            current = [row]
            start = index
        else:
            current.append(row)
        previous = index
    output.append(current)
    return output


def attach_flat_run_lumi(
    rows: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """Restore the full event triplet needed for production-version remaps."""
    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_root[str(row["nominal_root"])].append(row)
    attached = 0
    failures: list[str] = []
    for root_index, (root_path, root_rows) in enumerate(
        sorted(rows_by_root.items()), start=1
    ):
        try:
            with uproot.open(root_path) as root_file:
                tree = root_file["Events"]
                for batch in flat_identity_batches(root_rows):
                    start = int(batch[0]["flat_index"])
                    stop = int(batch[-1]["flat_index"]) + 1
                    arrays = tree.arrays(
                        ["run", "luminosityBlock", "event"],
                        entry_start=start,
                        entry_stop=stop,
                        library="np",
                    )
                    for row in batch:
                        local = int(row["flat_index"]) - start
                        event = int(arrays["event"][local])
                        if event != int(row["event"]):
                            failures.append(
                                f"{root_path}: flat_index "
                                f"{row['flat_index']} event mismatch "
                                f"{event} != {row['event']}"
                            )
                            continue
                        row["run"] = int(arrays["run"][local])
                        row["luminosityBlock"] = int(
                            arrays["luminosityBlock"][local]
                        )
                        attached += 1
        except Exception as exc:
            failures.append(
                f"{root_path}: {type(exc).__name__}: {exc}"
            )
        if root_index == 1 or root_index % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "flat_run_lumi_attachment",
                        "roots_complete": root_index,
                        "roots_total": len(rows_by_root),
                        "rows_attached": attached,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return attached, failures


def compact_event(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "group",
        "process",
        "file_path",
        "dataset",
        "entry",
        "event",
        "run",
        "luminosityBlock",
        "recoil_gcr",
        "ht_photon_clean",
        "gen_weight_flat",
        "analysis_nominal_weight",
        "normalized_gen_weight",
        "photon_medium_pt",
        "photon_medium_eta",
        "photon_medium_phi",
        "gen_diagnostic_status",
        "gen_diagnostic_error",
        "reco_match_dr",
        "photon_gen_part_flavour",
        "valid_gen_match",
        "prompt_flavour",
        "generator_binvar",
        "lhe_ht",
        "gen_photon_pt",
        "gen_photon_eta",
        "gen_photon_pdgId",
        "gen_photon_status",
        "gen_photon_statusFlags",
        "gen_photon_isPrompt",
        "gen_photon_isHardProcess",
        "gen_photon_fromHardProcess",
        "gen_photon_fromHardProcessBeforeFSR",
        "hard_parton_count_status23",
        "min_dr_status23_parton",
        "nearest_status23_parton_pdgId",
        "nearest_status23_parton_statusFlags",
        "min_dr_status23_hardflag_parton",
        "lhe_photon_count",
        "min_dr_lhe_photon",
        "nearest_lhe_photon_pt",
    ]
    return {key: row[key] for key in keys if key in row}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--join-only", action="store_true")
    parser.add_argument("--local-eos-only", action="store_true")
    parser.add_argument("--enrich-run-lumi", action="store_true")
    args = parser.parse_args()

    started = time.time()
    repo = Path(args.repo).resolve()
    snapshot = Path(args.snapshot)
    campaign = Path(args.campaign)
    output = Path(args.output)
    cache_dir = Path(args.cache_dir)
    prepare_path = output.with_name(f"{output.stem}.prepare.json")
    prepared_rows_path = output.with_name(
        f"{output.stem}.prepared_rows.json.gz"
    )
    selection_path = output.with_name(f"{output.stem}.selection.json")
    selected_rows_path = output.with_name(
        f"{output.stem}.selected_rows.json.gz"
    )
    sys.path.insert(0, str(repo / "autonomous_allhad"))
    os.chdir(repo)
    os.environ.setdefault("AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA", "0")

    expected_inputs = {
        "repo": str(repo),
        "snapshot": str(snapshot),
        "campaign": str(campaign),
        "normalization": str(args.normalization),
    }
    use_prepared = False
    if prepare_path.is_file() and prepared_rows_path.is_file():
        candidate_prepared = base.read_json(prepare_path)
        if (
            candidate_prepared.get("status") == "prepared"
            and candidate_prepared.get("inputs") == expected_inputs
            and not candidate_prepared.get("flat_row_read_failures")
            and int(
                candidate_prepared.get("rows_with_flat_weight_inputs") or -1
            )
            == int(candidate_prepared.get("exact_gcr_rows") or -2)
        ):
            with gzip.open(
                prepared_rows_path, "rt", encoding="utf-8"
            ) as handle:
                rows = json.load(handle)
            prepared = candidate_prepared
            use_prepared = len(rows) == int(
                prepared.get("exact_gcr_rows") or -1
            )
    if use_prepared:
        print(
            json.dumps(
                {
                    "stage": "prepared_rows_restored",
                    "rows": len(rows),
                    "path": str(prepared_rows_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        unmapped = list(prepared["unmapped_candidate_paths"])
        merged_resolution = prepared["merged_root_resolution"]
        nominal_stats = prepared["nominal_selection_stats"]
        flat_failures = list(prepared["flat_row_read_failures"])
        weight_status = prepared["weight_status"]
        attached_rows = int(prepared["rows_with_flat_weight_inputs"])
        missing_prepared_groups = (
            expected_generator_groups()
            - set(prepared["candidate_file_counts"])
        )
        if missing_prepared_groups:
            shard_bundle = (
                campaign / "bundles" / "fullselection_shards.tgz"
            )
            recovery_candidates = collect_manifest_recovery_candidates(
                shard_bundle, missing_prepared_groups
            )
            recovery_mapped, recovery_unmapped = (
                base.map_candidates_to_nominal_shards(
                    recovery_candidates,
                    shard_bundle,
                    campaign / "outputs" / "nominal",
                )
            )
            recovery_resolution = (
                base.resolve_removed_source_roots_to_merged(
                    recovery_mapped,
                    campaign
                    / "final_nominal_inputs_20260725"
                    / "nominal_input_roots.txt",
                )
            )
            recovery_rows, recovery_stats = base.inspect_nominal_rows(
                recovery_candidates,
                recovery_mapped,
                sys.maxsize,
                sys.maxsize,
                include_out_of_range=True,
            )
            builder = base.load_builder(repo)
            normalization = base.read_json(Path(args.normalization))
            (
                recovery_attached,
                recovery_weight_status,
                recovery_flat_failures,
            ) = assign_flat_weights_full(
                repo,
                recovery_rows,
                builder,
                normalization,
                targeted=True,
            )
            rows.extend(recovery_rows)
            unmapped.extend(recovery_unmapped)
            flat_failures.extend(recovery_flat_failures)
            merge_weight_status(weight_status, recovery_weight_status)
            attached_rows += recovery_attached
            prepared["candidate_file_counts"].update(
                {
                    label: len(items)
                    for label, items in recovery_candidates.items()
                }
            )
            prepared["candidate_files_total"] += sum(
                len(items) for items in recovery_candidates.values()
            )
            prepared["mapped_files"] += len(recovery_mapped)
            prepared["unmapped_candidate_paths"] = unmapped
            prepared["flat_row_read_failures"] = flat_failures
            prepared["weight_status"] = weight_status
            prepared["exact_gcr_rows"] = len(rows)
            prepared["rows_with_flat_weight_inputs"] = attached_rows
            prepared["nominal_selection_stats"]["files"].update(
                recovery_stats["files"]
            )
            prepared["nominal_selection_stats"]["groups"].update(
                recovery_stats["groups"]
            )
            prepared["manifest_recovery"] = {
                "requested_groups": sorted(missing_prepared_groups),
                "recovered_groups": sorted(recovery_candidates),
                "candidate_files": sum(
                    len(items)
                    for items in recovery_candidates.values()
                ),
                "mapped_files": len(recovery_mapped),
                "unmapped_files": len(recovery_unmapped),
                "exact_gcr_rows": len(recovery_rows),
                "rows_with_flat_weight_inputs": recovery_attached,
                "flat_row_read_failures": recovery_flat_failures,
                "merged_root_resolution": recovery_resolution,
            }
            atomic_gzip_json(prepared_rows_path, rows)
            atomic_json(prepare_path, prepared)
            print(
                json.dumps(
                    {
                        "stage": "missing_generator_group_recovery",
                        "requested_groups": sorted(
                            missing_prepared_groups
                        ),
                        "recovered_groups": sorted(
                            recovery_candidates
                        ),
                        "candidate_files": sum(
                            len(items)
                            for items in recovery_candidates.values()
                        ),
                        "exact_gcr_rows": len(recovery_rows),
                        "rows_with_flat_weight_inputs": (
                            recovery_attached
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.enrich_run_lumi and rows and (
            "run" not in rows[0] or "luminosityBlock" not in rows[0]
        ):
            identity_attached, identity_failures = attach_flat_run_lumi(
                rows
            )
            prepared["rows_with_run_lumi"] = identity_attached
            prepared["flat_identity_failures"] = identity_failures
            if not identity_failures and identity_attached == len(rows):
                atomic_gzip_json(prepared_rows_path, rows)
                atomic_json(prepare_path, prepared)
        elif args.enrich_run_lumi:
            identity_attached = len(rows)
            identity_failures = list(
                prepared.get("flat_identity_failures") or []
            )
    else:
        use_selection = False
        if selection_path.is_file() and selected_rows_path.is_file():
            selection = base.read_json(selection_path)
            if (
                selection.get("status") == "selected"
                and selection.get("inputs") == expected_inputs
            ):
                with gzip.open(
                    selected_rows_path, "rt", encoding="utf-8"
                ) as handle:
                    rows = json.load(handle)
                use_selection = len(rows) == int(
                    selection.get("exact_gcr_rows") or -1
                )
        if use_selection:
            print(
                json.dumps(
                    {
                        "stage": "selected_rows_restored",
                        "rows": len(rows),
                        "path": str(selected_rows_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "stage": "candidate_discovery",
                        "snapshot": str(snapshot),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            candidates = base.collect_candidate_files(
                snapshot, sys.maxsize
            )
            mapped, unmapped = base.map_candidates_to_nominal_shards(
                candidates,
                campaign / "bundles" / "fullselection_shards.tgz",
                campaign / "outputs" / "nominal",
            )
            merged_resolution = (
                base.resolve_removed_source_roots_to_merged(
                    mapped,
                    campaign
                    / "final_nominal_inputs_20260725"
                    / "nominal_input_roots.txt",
                )
            )
            rows, nominal_stats = base.inspect_nominal_rows(
                candidates,
                mapped,
                sys.maxsize,
                sys.maxsize,
                include_out_of_range=True,
            )
            selection = {
                "schema_version": "gcr_prompt_overlap_selection_v1",
                "status": "selected",
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "inputs": expected_inputs,
                "candidate_file_counts": {
                    label: len(items)
                    for label, items in candidates.items()
                },
                "candidate_files_total": sum(
                    len(items) for items in candidates.values()
                ),
                "mapped_files": len(mapped),
                "unmapped_candidate_paths": unmapped,
                "merged_root_resolution": merged_resolution,
                "nominal_selection_stats": nominal_stats,
                "exact_gcr_rows": len(rows),
                "selected_rows_path": str(selected_rows_path),
            }
            atomic_gzip_json(selected_rows_path, rows)
            atomic_json(selection_path, selection)
            print(
                json.dumps(
                    {
                        "stage": "exact_gcr_rows_collected",
                        "rows": len(rows),
                        "source_candidate_files": selection[
                            "candidate_files_total"
                        ],
                        "mapped_files": len(mapped),
                        "unmapped_files": len(unmapped),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        unmapped = list(selection["unmapped_candidate_paths"])
        merged_resolution = selection["merged_root_resolution"]
        nominal_stats = selection["nominal_selection_stats"]
        builder = base.load_builder(repo)
        normalization = base.read_json(Path(args.normalization))
        attached_rows, weight_status, flat_failures = (
            assign_flat_weights_full(
                repo,
                rows,
                builder,
                normalization,
            )
        )
        prepared = {
            "schema_version": "gcr_prompt_overlap_full_prepare_v1",
            "status": "prepared",
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "scope": {
                "selection": (
                    "All exact high-dM feature_GCR rows associated with "
                    "complete GJets/QCD source-file records in the frozen "
                    "photon-fake snapshot; no GCR selection is rerun."
                ),
                "nominal_and_sidecar_untouched": True,
            },
            "inputs": expected_inputs,
            "candidate_file_counts": selection["candidate_file_counts"],
            "candidate_files_total": selection[
                "candidate_files_total"
            ],
            "mapped_files": selection["mapped_files"],
            "unmapped_candidate_paths": unmapped,
            "merged_root_resolution": merged_resolution,
            "nominal_selection_stats": nominal_stats,
            "flat_row_read_failures": flat_failures,
            "weight_status": weight_status,
            "exact_gcr_rows": len(rows),
            "rows_with_flat_weight_inputs": attached_rows,
            "rows_with_run_lumi": attached_rows,
            "flat_identity_failures": [],
            "prepared_rows_path": str(prepared_rows_path),
        }
        atomic_gzip_json(prepared_rows_path, rows)
        atomic_json(prepare_path, prepared)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "output": str(prepare_path),
                    "prepared_rows": str(prepared_rows_path),
                },
                sort_keys=True,
            )
        )
        return 0

    gen_file_status, cache_summary = attach_gen_diagnostics_resumable(
        rows,
        cache_dir,
        args.workers,
        local_eos_only=args.local_eos_only,
    )
    if args.join_only:
        print(
            json.dumps(
                {
                    "stage": "gen_join_only_complete",
                    "cache_summary": cache_summary,
                    "complete_files": sum(
                        status.get("status") == "complete"
                        for status in gen_file_status.values()
                    ),
                    "noncomplete_files": sum(
                        status.get("status") != "complete"
                        for status in gen_file_status.values()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    scan = build_full_scan(rows)
    diagnostic_complete = [
        row
        for row in rows
        if row.get("gen_diagnostic_status") == "complete"
    ]
    prompt_complete = [
        row for row in diagnostic_complete if row.get("prompt_flavour")
    ]
    eligible = [row for row in rows if eligible_prompt(row)]
    failed_files = {
        path: status
        for path, status in gen_file_status.items()
        if status.get("status") != "complete"
    }
    completeness_pass = bool(
        not unmapped
        and not flat_failures
        and attached_rows == len(rows)
        and not failed_files
        and len(diagnostic_complete) == len(rows)
    )
    payload = {
        "schema_version": "gcr_hardparton_dr_full_selected_v1",
        "status": "complete" if completeness_pass else "partial",
        "artifact_status": (
            "full_selected_diagnostic_complete_not_adopted"
            if completeness_pass
            else "full_selected_diagnostic_partial_not_adopted"
        ),
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "wall_time_s": round(time.time() - started, 3),
        "scope": {
            "type": "full_exact_gcr_selected_entry_diagnostic",
            "selection": (
                "All exact nominal high-dM feature_GCR rows for GJets and QCD "
                "found from the complete frozen sidecar source-file records, "
                "joined by stable file_id and original entry to NanoAOD. No "
                "selection is rerun or approximated."
            ),
            "includes_ut_overflow_rows": True,
            "nominal_and_sidecar_untouched": True,
            "workers": int(args.workers),
        },
        "definitions": {
            "prompt_photon": (
                "Selected reco photon has Photon_genPartFlav==1, a valid "
                "Photon_genPartIdx matched to abs(GenPart_pdgId)==22 with "
                "GenPart_status==1."
            ),
            "hard_parton": (
                "GenPart status==23 and abs(pdgId) in {1,2,3,4,5,6,21}."
            ),
            "gj_direct_keep": (
                "min dR(prompt gen photon, status-23 q/g) >= R"
            ),
            "qcd_fragmentation_keep": (
                "min dR(prompt gen photon, status-23 q/g) < R"
            ),
            "radii": list(RADII),
            "ut_edges_gev": UT_EDGES.tolist(),
            "photon_pt_edges_gev": PHOTON_PT_EDGES.tolist(),
            "ht_photon_clean_edges_gev": HT_EDGES.tolist(),
            "generator_binvar_edges": GENERATOR_BINVAR_EDGES.tolist(),
            "neff_signed": "(sum w)^2 / sum(w^2)",
            "neff_abs": "(sum |w|)^2 / sum(w^2)",
            "leave_one_out": (
                "Event- and source-file-level signed yield ratios and "
                "jackknife relative sigma after removing one contributor."
            ),
        },
        "inputs": prepared["inputs"],
        "prepare_artifact": str(prepare_path),
        "candidate_file_counts": prepared["candidate_file_counts"],
        "candidate_files_total": prepared["candidate_files_total"],
        "mapped_files": prepared["mapped_files"],
        "unmapped_candidate_paths": unmapped,
        "missing_expected_generator_bins": sorted(
            expected_generator_groups()
            - set(prepared["candidate_file_counts"])
        ),
        "merged_root_resolution": merged_resolution,
        "nominal_selection_stats": nominal_stats,
        "flat_row_read_failures": flat_failures,
        "weight_status": weight_status,
        "gen_cache_summary": cache_summary,
        "gen_file_status": gen_file_status,
        "failed_gen_files": failed_files,
        "event_counts": {
            "exact_gcr_rows": len(rows),
            "rows_with_flat_weight_inputs": attached_rows,
            "gen_diagnostic_complete": len(diagnostic_complete),
            "prompt_flavour_complete": len(prompt_complete),
            "eligible_primary_dr": len(eligible),
            "by_process": {
                process: {
                    "exact_gcr_rows": sum(
                        row["process"] == process for row in rows
                    ),
                    "gen_diagnostic_complete": sum(
                        row["process"] == process
                        and row.get("gen_diagnostic_status") == "complete"
                        for row in rows
                    ),
                    "eligible_primary_dr": sum(
                        row["process"] == process and eligible_prompt(row)
                        for row in rows
                    ),
                }
                for process in ("GJ", "QCD")
            },
        },
        "completeness": {
            "pass": completeness_pass,
            "unmapped_candidate_paths": len(unmapped),
            "flat_row_read_failures": len(flat_failures),
            "failed_gen_files": len(failed_files),
            "all_exact_rows_have_gen_diagnostic": (
                len(diagnostic_complete) == len(rows)
            ),
        },
        "scan": scan,
        "events": [compact_event(row) for row in rows],
        "decision_guardrails": [
            (
                "No radius is adopted merely because it improves Data/MC; "
                "the radius must be generator-motivated and stable against "
                "event/file leave-one-out tests."
            ),
            (
                "Removing QCD prompt photons lowers the current prefit MC "
                "prediction and cannot by itself repair the GCR rate deficit."
            ),
            (
                "Any failed source-file join makes this audit partial and "
                "prevents an overlap-policy adoption."
            ),
        ],
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "exact_gcr_rows": len(rows),
                "eligible_primary_dr": len(eligible),
                "source_files": cache_summary[
                    "source_files_with_selected_rows"
                ],
                "wall_time_s": payload["wall_time_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if completeness_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
