#!/usr/bin/env python3
"""Representative GCR prompt-photon overlap diagnostic for 2024.

Physics guardrail: this program MUST NOT reconstruct or approximate the GCR
selection.  It reads exact ``feature_GCR`` rows from the already-produced
nominal flat ntuples, joins them to the original NanoAOD by stable ``file_id``
and ``entry``, and retrieves only the generator information omitted from the
flat output.  It never writes to nominal or photon-fake sidecar inputs and its
result is explicitly a representative diagnostic, not a full-campaign yield.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot


UT_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0])
RADII = (0.15, 0.20, 0.30, 0.40)
PARTON_ABS_PDGS = {1, 2, 3, 4, 5, 6, 21}
GJ_BINS = ("100to200", "200to400", "400to600", "600")
QCD_BINS = (
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000to1500",
    "1500to2000",
    "2000to3000",
    "3000",
)


def stable_id(text: str) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def group_label(dataset: str, process: str) -> str | None:
    physical = str(dataset).split("____", 1)[0]
    if process == "GJ":
        match = re.search(r"GJ_Bin-PTG-([^_]+)", physical)
        if match and match.group(1) in GJ_BINS:
            return f"GJ_PTG_{match.group(1)}"
    if process == "QCD":
        match = re.search(r"QCD_Bin-PT-([^_]+)", physical)
        if match and match.group(1) in QCD_BINS:
            return f"QCD_PT_{match.group(1)}"
    return None


def ut_bin_index(value: float) -> int:
    index = int(np.searchsorted(UT_EDGES, value, side="right") - 1)
    return index if 0 <= index < len(UT_EDGES) - 1 else -1


def ut_bin_label(index: int) -> str:
    return f"{int(UT_EDGES[index])}to{int(UT_EDGES[index + 1])}"


def delta_phi(a: float, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def min_dr(eta: float, phi: float, etas: np.ndarray, phis: np.ndarray) -> tuple[float | None, int | None]:
    if len(etas) == 0:
        return None, None
    dr = np.hypot(eta - etas, delta_phi(phi, phis))
    index = int(np.argmin(dr))
    return float(dr[index]), index


def collect_candidate_files(snapshot: Path, candidates_per_bin: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for process in ("GJ", "QCD"):
        for path in sorted((snapshot / "inputs" / process).glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            for record in (payload.get("summary") or {}).get("file_records") or []:
                if record.get("status") != "complete":
                    continue
                label = group_label(str(record.get("dataset") or ""), process)
                if label is None:
                    continue
                file_path = str(record.get("file_path") or "")
                if not file_path:
                    continue
                leaf = grouped[label].setdefault(
                    file_path,
                    {
                        "file_path": file_path,
                        "fake_sidecar_dataset": str(record.get("dataset") or ""),
                        "process": process,
                        "sidecar_selected_events": 0,
                        "sidecar_events_read": 0,
                        "sidecar_segments": 0,
                    },
                )
                leaf["sidecar_selected_events"] += int(record.get("selected_events") or 0)
                leaf["sidecar_events_read"] += int(record.get("events_read") or 0)
                leaf["sidecar_segments"] += 1
    out: dict[str, list[dict[str, Any]]] = {}
    for label, files in grouped.items():
        ordered = sorted(
            files.values(),
            key=lambda item: (
                -int(item["sidecar_selected_events"]),
                -int(item["sidecar_events_read"]),
                item["file_path"],
            ),
        )
        out[label] = ordered[:candidates_per_bin]
    return out


def map_candidates_to_nominal_shards(
    candidate_files: dict[str, list[dict[str, Any]]],
    shard_bundle: Path,
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    wanted = {
        item["file_path"]
        for items in candidate_files.values()
        for item in items
    }
    mapped: dict[str, dict[str, Any]] = {}
    with tarfile.open(shard_bundle, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "/mc_shard_" not in member.name:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            shard = json.load(extracted)
            for record in shard.get("records") or []:
                file_path = str(record.get("file_path") or "")
                if file_path not in wanted:
                    continue
                shard_id = str(shard["shard_id"])
                root_path = output_dir / f"{shard_id}.root"
                mapped[file_path] = {
                    "shard_id": shard_id,
                    "nominal_root": str(root_path),
                    "nominal_sidecar": str(output_dir / f"{shard_id}.json"),
                    "nominal_record": record,
                    "file_id": stable_id(file_path),
                    "dataset_id": stable_id(str(record.get("dataset") or "unknown")),
                    "root_exists": root_path.is_file(),
                }
            if len(mapped) == len(wanted):
                break
    return mapped, sorted(wanted - set(mapped))


def resolve_removed_source_roots_to_merged(
    mapped: dict[str, dict[str, Any]],
    nominal_root_list: Path,
) -> dict[str, Any]:
    unresolved_shards = {
        str(item["shard_id"])
        for item in mapped.values()
        if not item.get("root_exists")
    }
    shard_to_merged: dict[str, str] = {}
    checked = 0
    if unresolved_shards and nominal_root_list.is_file():
        for line in nominal_root_list.read_text().splitlines():
            root_path = Path(line.strip())
            if not root_path.name.startswith("mc_"):
                continue
            meta_path = root_path.with_suffix(".json")
            if not root_path.is_file() or not meta_path.is_file():
                continue
            checked += 1
            meta = read_json(meta_path)
            for shard_id in meta.get("source_shards") or []:
                shard_id = str(shard_id)
                if shard_id in unresolved_shards:
                    shard_to_merged[shard_id] = str(root_path)
            if len(shard_to_merged) == len(unresolved_shards):
                break
    for item in mapped.values():
        if item.get("root_exists"):
            item["nominal_root_source"] = "source_shard"
            continue
        merged = shard_to_merged.get(str(item["shard_id"]))
        if merged:
            item["nominal_root"] = merged
            item["nominal_sidecar"] = str(Path(merged).with_suffix(".json"))
            item["root_exists"] = True
            item["nominal_root_source"] = "final_merged_root"
        else:
            item["nominal_root_source"] = "unresolved_removed_source_shard"
    return {
        "nominal_root_list": str(nominal_root_list),
        "merged_sidecars_checked": checked,
        "removed_source_shards_requested": len(unresolved_shards),
        "removed_source_shards_resolved": len(shard_to_merged),
        "unresolved_shards": sorted(unresolved_shards - set(shard_to_merged)),
    }


def inspect_nominal_rows(
    candidate_files: dict[str, list[dict[str, Any]]],
    mapped: dict[str, dict[str, Any]],
    max_files_per_bin: int,
    max_per_ut: int,
    include_out_of_range: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_to_files: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for label, candidates in candidate_files.items():
        located = 0
        for candidate in candidates:
            file_path = candidate["file_path"]
            mapping = mapped.get(file_path)
            if not mapping or not mapping["root_exists"]:
                continue
            root_to_files[mapping["nominal_root"]].append((label, candidate, mapping))
            located += 1
            if located >= max_files_per_bin:
                break

    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_stats: dict[str, Any] = {}
    scalar_branches = [
        "file_id",
        "dataset_id",
        "entry",
        "event",
        "feature_GCR",
        "recoil_gcr",
        "ht_photon_clean",
        "gen_weight",
    ]
    for root_path, items in sorted(root_to_files.items()):
        with uproot.open(root_path) as root_file:
            tree = root_file["Events"]
            arrays = tree.arrays(scalar_branches, library="ak")
            file_ids = np.asarray(arrays["file_id"], dtype=np.int64)
            gcr = np.asarray(arrays["feature_GCR"], dtype=bool)
            recoil = np.asarray(arrays["recoil_gcr"], dtype=float)
            ht_photon_clean = np.asarray(
                arrays["ht_photon_clean"], dtype=float
            )
            entries = np.asarray(arrays["entry"], dtype=np.int64)
            events = np.asarray(arrays["event"], dtype=np.int64)
            dataset_ids = np.asarray(arrays["dataset_id"], dtype=np.int64)
            gen_weights = np.asarray(arrays["gen_weight"], dtype=float)
            for label, candidate, mapping in items:
                file_id = int(mapping["file_id"])
                indices = np.flatnonzero(gcr & (file_ids == file_id))
                file_stats[candidate["file_path"]] = {
                    **candidate,
                    **mapping,
                    "nominal_exact_gcr_rows": int(len(indices)),
                    "nominal_root_entries": int(tree.num_entries),
                }
                for flat_index in indices:
                    ut_index = ut_bin_index(float(recoil[flat_index]))
                    if ut_index < 0 and not include_out_of_range:
                        continue
                    if ut_index >= 0:
                        ut_label = ut_bin_label(ut_index)
                    elif float(recoil[flat_index]) < float(UT_EDGES[0]):
                        ut_label = "underflow"
                    else:
                        ut_label = "overflow"
                    rows_by_group[label].append(
                        {
                            "group": label,
                            "process": candidate["process"],
                            "file_path": candidate["file_path"],
                            "file_id": file_id,
                            "dataset": str(mapping["nominal_record"].get("dataset") or ""),
                            "dataset_id": int(dataset_ids[flat_index]),
                            "nominal_root": root_path,
                            "flat_index": int(flat_index),
                            "entry": int(entries[flat_index]),
                            "event": int(events[flat_index]),
                            "recoil_gcr": float(recoil[flat_index]),
                            "ht_photon_clean": float(
                                ht_photon_clean[flat_index]
                            ),
                            "ut_bin_index": ut_index,
                            "ut_bin": ut_label,
                            "gen_weight_flat": float(gen_weights[flat_index]),
                        }
                    )

    sampled: list[dict[str, Any]] = []
    group_stats: dict[str, Any] = {}
    for label, rows in sorted(rows_by_group.items()):
        chosen: list[dict[str, Any]] = []
        before_by_ut: dict[str, int] = {}
        after_by_ut: dict[str, int] = {}
        for index in range(len(UT_EDGES) - 1):
            in_bin = [row for row in rows if row["ut_bin_index"] == index]
            in_bin.sort(
                key=lambda row: hashlib.sha256(
                    f"{row['file_id']}:{row['entry']}".encode()
                ).hexdigest()
            )
            label_ut = ut_bin_label(index)
            before_by_ut[label_ut] = len(in_bin)
            selected = in_bin[:max_per_ut]
            after_by_ut[label_ut] = len(selected)
            chosen.extend(selected)
        if include_out_of_range:
            for label_ut in ("underflow", "overflow"):
                in_bin = [row for row in rows if row["ut_bin"] == label_ut]
                in_bin.sort(
                    key=lambda row: hashlib.sha256(
                        f"{row['file_id']}:{row['entry']}".encode()
                    ).hexdigest()
                )
                before_by_ut[label_ut] = len(in_bin)
                selected = in_bin[:max_per_ut]
                after_by_ut[label_ut] = len(selected)
                chosen.extend(selected)
        sampled.extend(chosen)
        group_stats[label] = {
            "nominal_exact_gcr_rows": len(rows),
            "sampled_rows": len(chosen),
            "exact_gcr_rows_by_ut": before_by_ut,
            "sampled_rows_by_ut": after_by_ut,
        }
    return sampled, {"files": file_stats, "groups": group_stats}


def load_builder(repo: Path) -> Any:
    path = repo / "autonomous_allhad" / "workflow" / "build_flat_boosted_recoil_hists.py"
    spec = importlib.util.spec_from_file_location("_gcr_scan_hist_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load histogram builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_flat_sample_rows(
    sampled: list[dict[str, Any]],
    builder: Any,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        by_root[row["nominal_root"]].append(row)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[str] = []
    branches = sorted(
        set(builder.WEIGHT_BRANCHES)
        | {
            "file_id",
            "feature_GCR",
            "recoil_gcr",
            "photon_medium_pt",
            "photon_medium_eta",
            "photon_medium_phi",
        }
    )
    for root_path, rows in sorted(by_root.items()):
        try:
            with uproot.open(root_path) as root_file:
                tree = root_file["Events"]
                present = set(tree.keys())
                selected_branches = [name for name in branches if name in present]
                for row in sorted(rows, key=lambda item: item["flat_index"]):
                    index = int(row["flat_index"])
                    array = tree.arrays(
                        selected_branches,
                        entry_start=index,
                        entry_stop=index + 1,
                        library="ak",
                    )
                    out[(root_path, index)] = {
                        name: array[name]
                        for name in ak.fields(array)
                    }
        except Exception as exc:
            failures.append(f"{root_path}: {type(exc).__name__}: {exc}")
    return out, failures


def assign_analysis_weights(
    repo: Path,
    sampled: list[dict[str, Any]],
    flat_rows: dict[tuple[str, int], dict[str, Any]],
    builder: Any,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    from autonomous_allhad.real_subset_worker import compute_weight_bundle

    by_dataset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        if (row["nominal_root"], row["flat_index"]) in flat_rows:
            by_dataset[int(row["dataset_id"])].append(row)
    status: dict[str, Any] = {}
    for dataset_id, rows in sorted(by_dataset.items()):
        dataset = rows[0]["dataset"]
        process = rows[0]["process"]
        try:
            records = [
                ak.zip(
                    flat_rows[(row["nominal_root"], row["flat_index"])],
                    depth_limit=1,
                )
                for row in rows
            ]
            batch = ak.concatenate(records, axis=0)
            chunk = {name: batch[name] for name in ak.fields(batch)}
            arrays, inputs = builder.flat_arrays_for_weights(chunk)
            _gen, variations, sf_status = compute_weight_bundle(
                arrays,
                repo,
                dataset,
                process,
                "2024",
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
            normv = builder.norm_vector(
                normalization,
                chunk,
                dataset_id,
                False,
                False,
            )
            nominal = np.asarray(variations["nominal"], dtype=float) * normv
            raw_normalized = np.asarray(chunk["gen_weight"], dtype=float) * normv
            status[str(dataset_id)] = {
                "dataset": dataset,
                "process": process,
                "events": len(rows),
                "status": "nominal_sf_and_normalization_applied",
                "normalization_factor": float(normv[0]) if len(normv) else None,
                "scale_factor_status": sf_status,
            }
        except Exception as exc:
            factor_record = (
                (normalization.get("dataset_factors") or {}).get(str(dataset_id))
                or {}
            )
            factor = float(factor_record.get("normalization_factor") or 0.0)
            raw_normalized = np.asarray(
                [row["gen_weight_flat"] for row in rows],
                dtype=float,
            ) * factor
            nominal = raw_normalized.copy()
            status[str(dataset_id)] = {
                "dataset": dataset,
                "process": process,
                "events": len(rows),
                "status": "fallback_normalized_gen_weight",
                "normalization_factor": factor,
                "error": f"{type(exc).__name__}: {exc}",
            }
        for row, weight, raw_weight in zip(rows, nominal, raw_normalized):
            row["analysis_nominal_weight"] = float(weight)
            row["normalized_gen_weight"] = float(raw_weight)
            flat = flat_rows[(row["nominal_root"], row["flat_index"])]
            for name in ("photon_medium_pt", "photon_medium_eta", "photon_medium_phi"):
                values = ak.to_list(flat[name][0])
                row[name] = values
    return status


def alternate_urls(url: str) -> list[str]:
    out = [url]
    marker = "//store/"
    if marker in url:
        suffix = "/store/" + url.split(marker, 1)[1]
        out.extend(
            [
                "root://cmsxrootd.fnal.gov/" + suffix,
                "root://xrootd-cms.infn.it/" + suffix,
            ]
        )
    unique = []
    for item in out:
        if item not in unique:
            unique.append(item)
    return unique


def gen_diagnostic_for_event(tree: Any, row: dict[str, Any]) -> dict[str, Any]:
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
    present = set(tree.keys())
    if row["process"] == "GJ":
        needed.extend(
            [
                "LHEPart_pt",
                "LHEPart_eta",
                "LHEPart_phi",
                "LHEPart_pdgId",
                "LHEPart_status",
            ]
        )
    branches = [name for name in needed if name in present]
    arrays = tree.arrays(
        branches,
        entry_start=int(row["entry"]),
        entry_stop=int(row["entry"]) + 1,
        library="ak",
    )
    reco_eta = np.asarray(ak.to_numpy(arrays["Photon_eta"][0]), dtype=float)
    reco_phi = np.asarray(ak.to_numpy(arrays["Photon_phi"][0]), dtype=float)
    reco_pt = np.asarray(ak.to_numpy(arrays["Photon_pt"][0]), dtype=float)
    flat_eta_values = row.get("photon_medium_eta") or []
    flat_phi_values = row.get("photon_medium_phi") or []
    flat_pt_values = row.get("photon_medium_pt") or []
    if len(flat_eta_values) != 1 or len(flat_phi_values) != 1:
        raise RuntimeError(
            f"nominal GCR row has {len(flat_eta_values)} medium photons"
        )
    flat_eta = float(flat_eta_values[0])
    flat_phi = float(flat_phi_values[0])
    dr_reco, reco_index = min_dr(flat_eta, flat_phi, reco_eta, reco_phi)
    if reco_index is None:
        raise RuntimeError("no reco photon in original event")
    gen_indices = np.asarray(
        ak.to_numpy(arrays["Photon_genPartIdx"][0]), dtype=int
    )
    gen_flavours = np.asarray(
        ak.to_numpy(arrays["Photon_genPartFlav"][0]), dtype=int
    )
    gen_index = int(gen_indices[reco_index])
    flavour = int(gen_flavours[reco_index])

    gen_pt = np.asarray(ak.to_numpy(arrays["GenPart_pt"][0]), dtype=float)
    gen_eta = np.asarray(ak.to_numpy(arrays["GenPart_eta"][0]), dtype=float)
    gen_phi = np.asarray(ak.to_numpy(arrays["GenPart_phi"][0]), dtype=float)
    gen_pdg = np.asarray(ak.to_numpy(arrays["GenPart_pdgId"][0]), dtype=int)
    gen_status = np.asarray(
        ak.to_numpy(arrays["GenPart_status"][0]), dtype=int
    )
    gen_flags = np.asarray(
        ak.to_numpy(arrays["GenPart_statusFlags"][0]), dtype=np.int64
    )
    gen_mother = np.asarray(
        ak.to_numpy(arrays["GenPart_genPartIdxMother"][0]), dtype=int
    )
    valid_gen = 0 <= gen_index < len(gen_pt)
    diag: dict[str, Any] = {
        "reco_match_dr": dr_reco,
        "reco_match_index": reco_index,
        "reco_original_pt": float(reco_pt[reco_index]),
        "reco_flat_pt": float(flat_pt_values[0]) if flat_pt_values else None,
        "photon_gen_part_flavour": flavour,
        "photon_gen_part_index": gen_index,
        "valid_gen_match": valid_gen,
        "prompt_flavour": flavour == 1,
    }
    if "Generator_binvar" in ak.fields(arrays):
        diag["generator_binvar"] = float(arrays["Generator_binvar"][0])
    if "LHE_HT" in ak.fields(arrays):
        diag["lhe_ht"] = float(arrays["LHE_HT"][0])
    if not valid_gen:
        return diag

    photon_eta = float(gen_eta[gen_index])
    photon_phi = float(gen_phi[gen_index])
    status23_parton = np.asarray(
        [
            int(abs(pdg)) in PARTON_ABS_PDGS and int(status) == 23
            for pdg, status in zip(gen_pdg, gen_status)
        ],
        dtype=bool,
    )
    hard_flag = (
        ((gen_flags & (1 << 7)) != 0)
        | ((gen_flags & (1 << 8)) != 0)
        | ((gen_flags & (1 << 11)) != 0)
    )
    primary_indices = np.flatnonzero(status23_parton)
    hardflag_indices = np.flatnonzero(status23_parton & hard_flag)
    primary_dr, primary_local = min_dr(
        photon_eta,
        photon_phi,
        gen_eta[primary_indices],
        gen_phi[primary_indices],
    )
    hardflag_dr, hardflag_local = min_dr(
        photon_eta,
        photon_phi,
        gen_eta[hardflag_indices],
        gen_phi[hardflag_indices],
    )
    nearest_index = (
        int(primary_indices[primary_local])
        if primary_local is not None
        else None
    )
    nearest_hardflag_index = (
        int(hardflag_indices[hardflag_local])
        if hardflag_local is not None
        else None
    )
    mother_index = int(gen_mother[gen_index])
    mother_chain = []
    seen = set()
    cursor = mother_index
    while 0 <= cursor < len(gen_pdg) and cursor not in seen and len(mother_chain) < 12:
        seen.add(cursor)
        mother_chain.append(
            {
                "index": int(cursor),
                "pdgId": int(gen_pdg[cursor]),
                "status": int(gen_status[cursor]),
                "statusFlags": int(gen_flags[cursor]),
            }
        )
        cursor = int(gen_mother[cursor])
    diag.update(
        {
            "gen_photon_pt": float(gen_pt[gen_index]),
            "gen_photon_eta": photon_eta,
            "gen_photon_phi": photon_phi,
            "gen_photon_pdgId": int(gen_pdg[gen_index]),
            "gen_photon_status": int(gen_status[gen_index]),
            "gen_photon_statusFlags": int(gen_flags[gen_index]),
            "gen_photon_isPrompt": bool(gen_flags[gen_index] & (1 << 0)),
            "gen_photon_isHardProcess": bool(gen_flags[gen_index] & (1 << 7)),
            "gen_photon_fromHardProcess": bool(gen_flags[gen_index] & (1 << 8)),
            "gen_photon_fromHardProcessBeforeFSR": bool(
                gen_flags[gen_index] & (1 << 11)
            ),
            "hard_parton_count_status23": int(len(primary_indices)),
            "hard_parton_count_status23_hardflag": int(len(hardflag_indices)),
            "min_dr_status23_parton": primary_dr,
            "nearest_status23_parton_index": nearest_index,
            "nearest_status23_parton_pdgId": (
                int(gen_pdg[nearest_index]) if nearest_index is not None else None
            ),
            "nearest_status23_parton_statusFlags": (
                int(gen_flags[nearest_index]) if nearest_index is not None else None
            ),
            "min_dr_status23_hardflag_parton": hardflag_dr,
            "nearest_status23_hardflag_parton_index": nearest_hardflag_index,
            "mother_chain": mother_chain,
        }
    )

    if row["process"] == "GJ" and "LHEPart_pdgId" in ak.fields(arrays):
        lhe_pdg = np.asarray(
            ak.to_numpy(arrays["LHEPart_pdgId"][0]), dtype=int
        )
        lhe_eta = np.asarray(
            ak.to_numpy(arrays["LHEPart_eta"][0]), dtype=float
        )
        lhe_phi = np.asarray(
            ak.to_numpy(arrays["LHEPart_phi"][0]), dtype=float
        )
        lhe_pt = np.asarray(
            ak.to_numpy(arrays["LHEPart_pt"][0]), dtype=float
        )
        lhe_photons = np.flatnonzero(abs(lhe_pdg) == 22)
        lhe_dr, lhe_local = min_dr(
            photon_eta,
            photon_phi,
            lhe_eta[lhe_photons],
            lhe_phi[lhe_photons],
        )
        lhe_index = (
            int(lhe_photons[lhe_local])
            if lhe_local is not None
            else None
        )
        diag.update(
            {
                "lhe_photon_count": int(len(lhe_photons)),
                "min_dr_lhe_photon": lhe_dr,
                "nearest_lhe_photon_pt": (
                    float(lhe_pt[lhe_index]) if lhe_index is not None else None
                ),
            }
        )
    return diag


def attach_gen_diagnostics(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        by_file[row["file_path"]].append(row)
    file_status: dict[str, Any] = {}
    for file_path, rows in sorted(by_file.items()):
        root_file = None
        errors = []
        used_url = None
        for url in alternate_urls(file_path):
            try:
                root_file = uproot.open(url, timeout=60)
                used_url = url
                break
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        if root_file is None:
            file_status[file_path] = {
                "status": "open_failed",
                "events_requested": len(rows),
                "errors": errors,
            }
            for row in rows:
                row["gen_diagnostic_status"] = "file_open_failed"
            continue
        successes = 0
        failures = []
        try:
            tree = root_file["Events"]
            for row in sorted(rows, key=lambda item: item["entry"]):
                try:
                    row.update(gen_diagnostic_for_event(tree, row))
                    row["gen_diagnostic_status"] = "complete"
                    successes += 1
                except Exception as exc:
                    row["gen_diagnostic_status"] = "event_failed"
                    row["gen_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
                    failures.append(
                        {
                            "entry": row["entry"],
                            "error": row["gen_diagnostic_error"],
                        }
                    )
        finally:
            root_file.close()
        file_status[file_path] = {
            "status": "complete" if successes == len(rows) else "partial",
            "used_url": used_url,
            "events_requested": len(rows),
            "events_complete": successes,
            "failures": failures,
            "open_errors_before_success": errors,
        }
    return file_status


def summarize_leaf(rows: list[dict[str, Any]], process: str, radius: float) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("gen_diagnostic_status") == "complete"
        and row.get("prompt_flavour")
        and row.get("valid_gen_match")
        and row.get("gen_photon_pdgId") == 22
        and row.get("gen_photon_status") == 1
        and row.get("min_dr_status23_parton") is not None
        and math.isfinite(float(row["min_dr_status23_parton"]))
    ]
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

    def sum_field(items: list[dict[str, Any]], field: str, absolute: bool = False) -> float:
        values = []
        for item in items:
            value = float(item.get(field, 0.0))
            if math.isfinite(value):
                values.append(abs(value) if absolute else value)
        return float(math.fsum(values))

    total_weight = sum_field(eligible, "analysis_nominal_weight")
    kept_weight = sum_field(kept, "analysis_nominal_weight")
    total_abs = sum_field(eligible, "analysis_nominal_weight", True)
    kept_abs = sum_field(kept, "analysis_nominal_weight", True)
    return {
        "policy": policy,
        "eligible_prompt_events": len(eligible),
        "surviving_events": len(kept),
        "unweighted_survival": (
            len(kept) / len(eligible) if eligible else None
        ),
        "total_nominal_sumw": total_weight,
        "surviving_nominal_sumw": kept_weight,
        "weighted_survival": (
            kept_weight / total_weight if total_weight != 0 else None
        ),
        "total_abs_nominal_sumw": total_abs,
        "surviving_abs_nominal_sumw": kept_abs,
        "abs_weighted_survival": (
            kept_abs / total_abs if total_abs != 0 else None
        ),
    }


def build_scan_summary(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"process": {}, "dataset_bin": {}}
    for process in ("GJ", "QCD"):
        process_rows = [row for row in sampled if row["process"] == process]
        process_payload: dict[str, Any] = {}
        for radius in RADII:
            by_ut = {
                ut_bin_label(index): summarize_leaf(
                    [
                        row
                        for row in process_rows
                        if row["ut_bin_index"] == index
                    ],
                    process,
                    radius,
                )
                for index in range(len(UT_EDGES) - 1)
            }
            process_payload[f"{radius:.2f}"] = {
                "inclusive": summarize_leaf(process_rows, process, radius),
                "by_ut": by_ut,
            }
        output["process"][process] = process_payload

    for label in sorted(set(row["group"] for row in sampled)):
        group_rows = [row for row in sampled if row["group"] == label]
        process = str(group_rows[0]["process"])
        group_payload: dict[str, Any] = {}
        for radius in RADII:
            group_payload[f"{radius:.2f}"] = {
                "inclusive": summarize_leaf(group_rows, process, radius),
                "by_ut": {
                    ut_bin_label(index): summarize_leaf(
                        [
                            row
                            for row in group_rows
                            if row["ut_bin_index"] == index
                        ],
                        process,
                        radius,
                    )
                    for index in range(len(UT_EDGES) - 1)
                },
            }
        output["dataset_bin"][label] = group_payload
    return output


def compact_event(row: dict[str, Any]) -> dict[str, Any]:
    omit = {
        "nominal_root",
        "flat_index",
        "file_id",
        "dataset_id",
        "ut_bin_index",
    }
    return {key: value for key, value in row.items() if key not in omit}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-files-per-bin", type=int, default=5)
    parser.add_argument("--files-per-bin", type=int, default=2)
    parser.add_argument("--max-per-ut", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    repo = Path(args.repo).resolve()
    snapshot = Path(args.snapshot)
    campaign = Path(args.campaign)
    output = Path(args.output)
    os.chdir(repo)
    os.environ.setdefault("AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA", "0")

    candidates = collect_candidate_files(
        snapshot,
        args.candidate_files_per_bin,
    )
    mapped, unmapped = map_candidates_to_nominal_shards(
        candidates,
        campaign / "bundles" / "fullselection_shards.tgz",
        campaign / "outputs" / "nominal",
    )
    merged_resolution = resolve_removed_source_roots_to_merged(
        mapped,
        campaign
        / "final_nominal_inputs_20260725"
        / "nominal_input_roots.txt",
    )
    sampled, nominal_stats = inspect_nominal_rows(
        candidates,
        mapped,
        args.files_per_bin,
        args.max_per_ut,
    )
    builder = load_builder(repo)
    flat_rows, flat_failures = read_flat_sample_rows(sampled, builder)
    normalization = read_json(Path(args.normalization))
    weight_status = assign_analysis_weights(
        repo,
        sampled,
        flat_rows,
        builder,
        normalization,
    )
    checkpoint = output.with_name(
        f"{output.stem}.pre_gen_checkpoint.json"
    )
    checkpoint.write_text(
        json.dumps(
            {
                "status": "pre_gen_checkpoint",
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "sampled_exact_gcr": len(sampled),
                "nominal_selection_stats": nominal_stats,
                "weight_status": weight_status,
                "flat_row_read_failures": flat_failures,
                "rows": [compact_event(row) for row in sampled],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    gen_file_status = attach_gen_diagnostics(sampled)
    scan = build_scan_summary(sampled)

    prompt_complete = [
        row
        for row in sampled
        if row.get("gen_diagnostic_status") == "complete"
        and row.get("prompt_flavour")
    ]
    payload = {
        "schema_version": "gcr_hardparton_dr_representative_v1",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": round(time.time() - started, 3),
        "scope": {
            "type": "representative_diagnostic_not_full_campaign",
            "selection": (
                "Exact nominal high-dM feature_GCR rows from the existing "
                "flat ntuple, joined by stable file_id and original entry to "
                "the source NanoAOD. No selection is rerun."
            ),
            "sampling": (
                f"Up to {args.files_per_bin} source files per generator pT bin "
                f"and {args.max_per_ut} exact-GCR rows per U_T bin, chosen "
                "deterministically by SHA256(file_id:entry)."
            ),
            "nominal_and_sidecar_untouched": True,
        },
        "definitions": {
            "prompt_photon": (
                "Selected reco photon has Photon_genPartFlav==1, a valid "
                "Photon_genPartIdx matched to abs(GenPart_pdgId)==22 with "
                "GenPart_status==1."
            ),
            "hard_parton_primary": (
                "GenPart status==23 and abs(pdgId) in {1,2,3,4,5,6,21}; "
                "no statusFlags requirement, matching the Run-2 AN geometric "
                "direct/fragmentation definition."
            ),
            "hard_parton_crosscheck": (
                "Also stored: primary candidates restricted to statusFlags "
                "isHardProcess OR fromHardProcess OR "
                "fromHardProcessBeforeFSR."
            ),
            "gj_direct_keep": "min dR(prompt gen photon, status-23 q/g) >= R",
            "qcd_fragmentation_keep": "min dR(prompt gen photon, status-23 q/g) < R",
            "radii": list(RADII),
            "ut_edges_gev": UT_EDGES.tolist(),
            "weighted_survival": (
                "Uses the current post-skim nominal scale-factor bundle times "
                "the campaign normalization factor. Raw normalized-gen-weight "
                "values and absolute-weight survival are also stored."
            ),
        },
        "inputs": {
            "repo": str(repo),
            "snapshot": str(snapshot),
            "campaign": str(campaign),
            "normalization": str(args.normalization),
            "pre_gen_checkpoint": str(checkpoint),
            "selection_source": (
                "autonomous_allhad.real_subset_worker via existing nominal "
                "flat-ntuple feature_GCR"
            ),
        },
        "candidate_files": candidates,
        "unmapped_candidate_paths": unmapped,
        "merged_root_resolution": merged_resolution,
        "nominal_selection_stats": nominal_stats,
        "flat_row_read_failures": flat_failures,
        "weight_status": weight_status,
        "gen_file_status": gen_file_status,
        "event_counts": {
            "sampled_exact_gcr": len(sampled),
            "gen_diagnostic_complete": sum(
                row.get("gen_diagnostic_status") == "complete"
                for row in sampled
            ),
            "prompt_flavour_complete": len(prompt_complete),
            "eligible_primary_dr": sum(
                row.get("gen_diagnostic_status") == "complete"
                and row.get("prompt_flavour")
                and row.get("valid_gen_match")
                and row.get("gen_photon_pdgId") == 22
                and row.get("gen_photon_status") == 1
                and row.get("min_dr_status23_parton") is not None
                for row in sampled
            ),
        },
        "scan": scan,
        "events": [compact_event(row) for row in sampled],
        "interpretation_guardrails": [
            (
                "This representative scan estimates the geometric partition "
                "efficiency; it is not a replacement for a full-campaign "
                "overlap-removed yield."
            ),
            (
                "A QCD prompt veto will lower the current MC prediction and "
                "therefore cannot by itself repair the observed GCR MC deficit."
            ),
            (
                "Photon_genPartFlav==1 alone is not a direct-photon tag; the "
                "hard-parton dR partition is required."
            ),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sampled": len(sampled),
                "prompt": len(prompt_complete),
                "eligible": payload["event_counts"]["eligible_primary_dr"],
                "wall_time_s": payload["wall_time_s"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
