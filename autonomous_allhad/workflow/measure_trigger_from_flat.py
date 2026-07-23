#!/usr/bin/env python3
"""Fast 2024 MET-trigger measurement from existing flat ROOT shards.

This is the quick measurement pass.  It uses the already evaluated object IDs,
JEC, lumimask, MET filters and trigger OR flags in the flat feature tables.
TrigObj matching and run-dependent prescale validation remain adoption gates.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import uproot


EDGES = np.asarray([100, 120, 140, 160, 180, 200, 220, 240, 250, 260, 270,
                    280, 290, 300, 350, 400, 500, 650, 800], dtype=float)
BRANCHES = [
    "run", "luminosityBlock", "event", "dataset_id", "met", "gen_weight",
    "n_e_medium", "n_m_loose", "njet_lepton_clean", "nb_lepton_clean",
    "electron_medium_pt", "pass_base_common", "pass_electron_trigger",
    "pass_signal_trigger", "pass_zero_tau", "pass_ht_lepton_300", "pass_open_pre",
]


def dataset_ids(sidecar: Path, kind: str) -> set[int]:
    payload = json.loads(sidecar.read_text())
    result = set()
    for key, item in payload.get("datasets", {}).items():
        name = str(item.get("dataset", ""))
        process = str(item.get("process", ""))
        if kind == "data" and process == "EGamma":
            result.add(int(item.get("dataset_id", key)))
        elif kind == "mc" and "TTtoLNu2Q" in name:
            result.add(int(item.get("dataset_id", key)))
    return result


def first_pt(values, size: int) -> np.ndarray:
    out = np.full(size, -1.0)
    for i, value in enumerate(values):
        if len(value):
            out[i] = float(value[0])
    return out


def selected(arrays, ids: set[int]) -> np.ndarray:
    n = len(arrays["met"])
    return (
        np.isin(np.asarray(arrays["dataset_id"]), list(ids))
        & np.asarray(arrays["pass_base_common"], dtype=bool)
        & np.asarray(arrays["pass_electron_trigger"], dtype=bool)
        & np.asarray(arrays["pass_zero_tau"], dtype=bool)
        & np.asarray(arrays["pass_ht_lepton_300"], dtype=bool)
        & np.asarray(arrays["pass_open_pre"], dtype=bool)
        & (np.asarray(arrays["n_e_medium"]) == 1)
        & (np.asarray(arrays["n_m_loose"]) == 0)
        & (np.asarray(arrays["njet_lepton_clean"]) >= 2)
        & (first_pt(arrays["electron_medium_pt"], n) > 40.0)
    )


def empty_counts() -> dict[str, np.ndarray]:
    n = len(EDGES) - 1
    return {key: np.zeros(n, dtype=float) for key in
            ["total", "passed", "sumw_total", "sumw_passed", "sumw2_total", "sumw2_passed"]}


def fill(counts, met, passed, weights):
    counts["total"] += np.histogram(met, EDGES)[0]
    counts["passed"] += np.histogram(met[passed], EDGES)[0]
    counts["sumw_total"] += np.histogram(met, EDGES, weights=weights)[0]
    counts["sumw_passed"] += np.histogram(met[passed], EDGES, weights=weights[passed])[0]
    counts["sumw2_total"] += np.histogram(met, EDGES, weights=weights * weights)[0]
    counts["sumw2_passed"] += np.histogram(met[passed], EDGES, weights=weights[passed] ** 2)[0]


def serialise(counts):
    return {key: [float(x) for x in values] for key, values in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = sorted(args.input_dir.glob("data_shard_*.root")) + sorted(args.input_dir.glob("mc_shard_*.root"))
    if args.recovery_dir:
        roots += sorted(args.recovery_dir.glob("*.root"))
    data_counts, mc_counts = empty_counts(), empty_counts()
    seen_data: set[tuple[int, int, int]] = set()
    stats = {"roots_total": len(roots), "roots_processed": 0, "roots_skipped_process": 0,
             "roots_failed": [], "data_duplicates_removed": 0, "data_selected": 0, "mc_selected": 0,
             "started_unix": time.time(), "status": "running"}
    state_path = args.output_dir / "state.json"
    for index, root_path in enumerate(roots, start=1):
        sidecar = root_path.with_suffix(".json")
        if not sidecar.exists():
            stats["roots_failed"].append({"path": str(root_path), "error": "sidecar missing"})
            continue
        kind = "data" if root_path.name.startswith("data_") else "mc"
        ids = dataset_ids(sidecar, kind)
        if not ids:
            stats["roots_skipped_process"] += 1
            continue
        try:
            arrays = uproot.open(f"{root_path}:Events").arrays(BRANCHES, library="ak")
            mask = selected(arrays, ids)
            positions = np.flatnonzero(mask)
            met = np.asarray(arrays["met"])[positions]
            passed = np.asarray(arrays["pass_signal_trigger"], dtype=bool)[positions]
            if kind == "data":
                keep = np.ones(len(positions), dtype=bool)
                runs = np.asarray(arrays["run"])[positions]
                lumis = np.asarray(arrays["luminosityBlock"])[positions]
                events = np.asarray(arrays["event"])[positions]
                for j, key in enumerate(zip(runs, lumis, events)):
                    event_key = tuple(int(x) for x in key)
                    if event_key in seen_data:
                        keep[j] = False
                        stats["data_duplicates_removed"] += 1
                    else:
                        seen_data.add(event_key)
                met, passed = met[keep], passed[keep]
                weights = np.ones(len(met))
                fill(data_counts, met, passed, weights)
                stats["data_selected"] += len(met)
            else:
                weights = np.asarray(arrays["gen_weight"])[positions].astype(float)
                fill(mc_counts, met, passed, weights)
                stats["mc_selected"] += len(met)
            stats["roots_processed"] += 1
        except Exception as exc:
            stats["roots_failed"].append({"path": str(root_path), "error": f"{type(exc).__name__}: {exc}"})
        if index % args.progress_every == 0:
            state_path.write_text(json.dumps({**stats, "updated_unix": time.time()}, indent=2) + "\n")
    stats["status"] = "complete" if not stats["roots_failed"] else "complete_with_failures"
    stats["completed_unix"] = time.time()
    payload = {"schema_version": 1, "measurement": "met_or_2024_flat_quick",
               "status": "preliminary_not_for_adoption", "bin_edges_gev": EDGES.tolist(),
               "selection": "base_common & electron_HLT & zero_tau & exactly_one_medium_e & zero_loose_mu & electron_pt>40 & njet_lepton_clean>=2 & ht_lepton_clean>300 & open_pre",
               "data": serialise(data_counts), "mc": serialise(mc_counts), "processing": stats,
               "adoption_gates": ["TrigObj matching", "run-dependent reference-trigger prescale audit", "per-era closure"]}
    (args.output_dir / "counts.json").write_text(json.dumps(payload, indent=2) + "\n")
    state_path.write_text(json.dumps(stats, indent=2) + "\n")
    return 0 if stats["roots_processed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
