from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from .real_subset_worker import (
    CORE_BRANCHES,
    ELECTRON_HLT,
    FILTERS,
    MUON_HLT,
    PHOTON_HLT,
    REGION_NAMES,
    SIGNAL_HLT,
    RootOpenFailure,
    cleanup_xrd_cache,
    extract_chunk,
    open_root_with_xrd_fallback,
)


FOCUS_REGIONS = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR"]
DATA_PROCESSES = {"JetMET", "EGamma", "Muon"}

VARIABLES: dict[str, tuple[str, list[float]]] = {
    "recoil_pt": ("recoil_gcr", [0, 100, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2500]),
    "metpt": ("met", [0, 100, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2500]),
    "ht": ("ht", [0, 300, 500, 800, 1000, 1200, 1500, 2000, 3000, 5000]),
    "njet": ("njet", [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 14.5]),
    "nb": ("nb_medium", [-0.5, 0.5, 1.5, 2.5, 3.5, 6.5]),
    "nfj": ("nfj", [-0.5, 0.5, 1.5, 2.5, 3.5, 6.5]),
    "min_dphi": ("min_dphi4", [0, 0.1, 0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 3.2]),
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def empty_counter() -> dict[str, Any]:
    return {"unweighted": 0, "raw_weighted": 0.0, "raw_sumw2": 0.0}


def add_counter(target: dict[str, Any], weight: float) -> None:
    target["unweighted"] += 1
    target["raw_weighted"] += float(weight)
    target["raw_sumw2"] += float(weight) * float(weight)


def empty_hist(edges: list[float]) -> dict[str, Any]:
    n = len(edges) - 1
    return {
        "bin_edges": [float(x) for x in edges],
        "raw_values": [0.0] * n,
        "raw_sumw2": [0.0] * n,
        "entries": [0] * n,
    }


def fill_hist(hist: dict[str, Any], value: float, weight: float) -> None:
    if not math.isfinite(value):
        return
    edges = hist["bin_edges"]
    idx = int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right") - 1)
    if 0 <= idx < len(edges) - 1:
        hist["raw_values"][idx] += float(weight)
        hist["raw_sumw2"][idx] += float(weight) * float(weight)
        hist["entries"][idx] += 1


def candidate_definitions() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    return {
        "minimal_njet_nb_met": [
            ("Njet2to4_Nb0_MET250to400", {"njet": [2, 5], "nb_medium": {"eq": 0}, "met": [250, 400]}),
            ("Njet2to4_Nb1plus_MET250to400", {"njet": [2, 5], "nb_medium": {"min": 1}, "met": [250, 400]}),
            ("Njet5to6_Nb1_MET250to400", {"njet": [5, 7], "nb_medium": {"eq": 1}, "met": [250, 400]}),
            ("Njet5to6_Nb2plus_MET250to600", {"njet": [5, 7], "nb_medium": {"min": 2}, "met": [250, 600]}),
            ("Njet7plus_Nb1plus_MET250to600", {"njet": {"min": 7}, "nb_medium": {"min": 1}, "met": [250, 600]}),
            ("Njet5plus_Nb1plus_MET600plus", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "met": {"min": 600}}),
        ],
        "resolved_kinematics": [
            ("highHT_Njet5plus_Nb1_MET250plus", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "ht": {"min": 1000}, "met": {"min": 250}}),
            ("veryHighHT_Njet6plus_Nb1_MET250plus", {"njet": {"min": 6}, "nb_medium": {"min": 1}, "ht": {"min": 1500}, "met": {"min": 250}}),
            ("twoB_highHT_MET400plus", {"nb_medium": {"min": 2}, "ht": {"min": 800}, "met": {"min": 400}}),
            ("cleanDphi_Njet5plus_Nb1_MET400plus", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "min_dphi4": {"min": 0.5}, "met": {"min": 400}}),
        ],
        "isr_sensitive": [
            ("hardJ1_Njet2plus_MET250to500", {"njet": {"min": 2}, "j1pt": {"min": 500}, "met": [250, 500]}),
            ("hardJ1_Njet2plus_MET500plus", {"njet": {"min": 2}, "j1pt": {"min": 500}, "met": {"min": 500}}),
            ("hardJ1_lowB_MET400plus", {"j1pt": {"min": 500}, "nb_medium": [0, 2], "met": {"min": 400}}),
        ],
        "ak8_kinematics_no_tag_scores": [
            ("AK8one_Njet5plus_Nb1_MET250plus", {"nfj": {"min": 1}, "njet": {"min": 5}, "nb_medium": {"min": 1}, "met": {"min": 250}}),
            ("AK8one_highMET", {"nfj": {"min": 1}, "met": {"min": 500}}),
            ("AK8two_Njet5plus", {"nfj": {"min": 2}, "njet": {"min": 5}}),
        ],
        "optimized_hybrid_no_tags": [
            ("hybrid_lowMET_multijet", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "met": [250, 400], "ht": {"min": 800}, "min_dphi4": {"min": 0.5}}),
            ("hybrid_midMET_multijet", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "met": [400, 600], "ht": {"min": 800}, "min_dphi4": {"min": 0.5}}),
            ("hybrid_highMET_anyHT", {"njet": {"min": 5}, "nb_medium": {"min": 1}, "met": {"min": 600}, "min_dphi4": {"min": 0.5}}),
            ("hybrid_AK8_highHT", {"nfj": {"min": 1}, "njet": {"min": 5}, "nb_medium": {"min": 1}, "ht": {"min": 1200}, "met": {"min": 250}}),
        ],
    }


def bin_mask(row: dict[str, Any], definition: dict[str, Any], region: str = "SR") -> bool:
    if str(row.get(f"feature_{region}", "")).lower() not in {"true", "1", "yes"} and row.get(f"feature_{region}") is not True:
        return False
    for key, rule in definition.items():
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            return False
        if isinstance(rule, list) and len(rule) == 2:
            if not (value >= float(rule[0]) and value < float(rule[1])):
                return False
        elif isinstance(rule, dict):
            if "min" in rule and value < float(rule["min"]):
                return False
            if "max" in rule and value >= float(rule["max"]):
                return False
            if "eq" in rule and value != float(rule["eq"]):
                return False
        else:
            return False
    return True


def split_keys(root: Any) -> set[str]:
    return {str(k).split(";")[0] for k in root.keys()}


def read_runs_sumw(root: Any) -> tuple[float | None, list[str]]:
    keys = split_keys(root)
    if "Runs" not in keys:
        return None, []
    runs = root["Runs"]
    branches = [str(b) for b in runs.keys()]
    candidates = [b for b in branches if b == "genEventSumw"]
    candidates += [b for b in branches if "geneventsumw" in b.lower() and not b.startswith("genEventSumw_T2tt_") and b not in candidates]
    if not candidates:
        return None, []
    for branch in candidates:
        try:
            vals = np.asarray(runs[branch].array(library="np"), dtype=float)
            return float(np.sum(vals)), candidates
        except Exception:
            continue
    return None, candidates


def ensure_dataset(payload: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    datasets = payload.setdefault("datasets", {})
    dataset = record["dataset"]
    rec = datasets.setdefault(
        dataset,
        {
            "dataset": dataset,
            "process": record["process_group"],
            "xsec_pb": record.get("xsec_pb"),
            "is_data": bool(record.get("is_data")),
            "is_background": bool(record.get("is_background")),
            "files_attempted": 0,
            "files_processed": 0,
            "events_read": 0,
            "sumw": 0.0,
            "sumw2": 0.0,
            "sumw_source_counts": {},
            "regions": {region: empty_counter() for region in FOCUS_REGIONS},
            "histograms": {},
            "search_bins": {},
        },
    )
    return rec


def merge_counter(into: dict[str, Any], source: dict[str, Any]) -> None:
    into["unweighted"] += int(source.get("unweighted", 0))
    into["raw_weighted"] += float(source.get("raw_weighted", 0.0))
    into["raw_sumw2"] += float(source.get("raw_sumw2", 0.0))


def process_file(record: dict[str, Any], repo: Path, chunk_size: int) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    file_path = record["file_path"]
    dataset = record["dataset"]
    process = record["process_group"]
    is_data = bool(record.get("is_data"))
    file_summary: dict[str, Any] = {
        "dataset": dataset,
        "process": process,
        "file_path": file_path,
        "file_index": record.get("file_index"),
        "read_status": "not_started",
        "processing_status": "not_started",
        "events_read": 0,
        "region_counts": {region: 0 for region in FOCUS_REGIONS},
    }
    bad: list[dict[str, Any]] = []
    file_payload = {
        "sumw": 0.0,
        "sumw2": 0.0,
        "sumw_source": "data_unweighted" if is_data else "events_genWeight_sum",
        "regions": {region: empty_counter() for region in FOCUS_REGIONS},
        "histograms": {},
        "search_bins": {},
    }
    root = None
    access_info: dict[str, Any] = {}
    cwd = Path.cwd()
    try:
        os.chdir(repo)
        root, access_info = open_root_with_xrd_fallback(file_path, timeout=60)
        file_summary["file_access"] = access_info
        file_summary["effective_file_path"] = access_info.get("effective_file_path", file_path)
        keys = split_keys(root)
        file_summary["events_tree_exists"] = "Events" in keys
        file_summary["runs_tree_exists"] = "Runs" in keys
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        tree = root["Events"]
        branches = set(tree.keys())
        required = {
            "run": "run" in branches,
            "luminosityBlock": "luminosityBlock" in branches,
            "event": "event" in branches,
            "Jet_pt": "Jet_pt" in branches,
            "usable_MET_pt": any(b in branches for b in ["PuppiMET_pt", "PFMET_pt", "MET_pt"]),
            "usable_MET_phi": any(b in branches for b in ["PuppiMET_phi", "PFMET_phi", "MET_phi"]),
        }
        file_summary["required_branch_validation"] = required
        if not all(required.values()):
            raise RuntimeError("required branch missing")
        runs_sumw, runs_branches = read_runs_sumw(root)
        file_summary["runs_sumw_branches"] = runs_branches
        if runs_sumw is not None and not is_data:
            file_payload["sumw"] = float(runs_sumw)
            file_payload["sumw_source"] = "Runs.genEventSumw"
        genmodel_branches = sorted([b for b in branches if str(b).startswith("GenModel_T2tt_")])
        read_branches = [b for b in set(CORE_BRANCHES + FILTERS + SIGNAL_HLT + PHOTON_HLT + ELECTRON_HLT + MUON_HLT + genmodel_branches) if b in branches]
        file_summary["number_of_entries"] = int(tree.num_entries)
        file_summary["read_status"] = "opened"
        file_summary["processed_entry_ranges"] = []
        for start in range(0, int(tree.num_entries), chunk_size):
            stop = min(start + chunk_size, int(tree.num_entries))
            arrays = tree.arrays(read_branches, entry_start=start, entry_stop=stop, library="ak")
            rows, chunk_summary = extract_chunk(arrays, dataset, process, record.get("signal_point") or None, str(record.get("year", "")), file_path, start, stop, fastsim_trigger_bypass=False)
            file_summary["processed_entry_ranges"].append({"entry_start": start, "entry_stop": stop})
            file_summary["events_read"] += len(rows)
            if not is_data and file_payload["sumw_source"] != "Runs.genEventSumw":
                weights = np.asarray([float(r.get("nominal_weight", 1.0)) for r in rows], dtype=float)
                file_payload["sumw"] += float(np.sum(weights))
                file_payload["sumw2"] += float(np.sum(weights * weights))
            for row in rows:
                event_weight = 1.0 if is_data else float(row.get("nominal_weight", 1.0))
                for region in FOCUS_REGIONS:
                    if row.get(f"feature_{region}") is not True:
                        continue
                    file_summary["region_counts"][region] += 1
                    add_counter(file_payload["regions"][region], event_weight)
                    for variable, (column, edges) in VARIABLES.items():
                        try:
                            value = float(row.get(column, float("nan")))
                        except Exception:
                            continue
                        hist = file_payload["histograms"].setdefault(region, {}).setdefault(variable, empty_hist(edges))
                        fill_hist(hist, value, event_weight)
                for scheme, defs in candidate_definitions().items():
                    for ibin, (bin_name, definition) in enumerate(defs):
                        if not bin_mask(row, definition, "SR"):
                            continue
                        rec = file_payload["search_bins"].setdefault(scheme, {}).setdefault(bin_name, empty_counter())
                        add_counter(rec, event_weight)
                        hist = file_payload["histograms"].setdefault("SR", {}).setdefault(
                            f"search_bin_index::{scheme}",
                            empty_hist([float(x) - 0.5 for x in range(len(defs) + 1)]),
                        )
                        fill_hist(hist, float(ibin), event_weight)
            file_summary.setdefault("chunk_summaries", []).append({"entry_start": start, "entry_stop": stop, **chunk_summary})
        if is_data:
            file_payload["sumw"] = float(file_summary["events_read"])
            file_payload["sumw2"] = float(file_summary["events_read"])
        file_summary["read_status"] = "success"
        file_summary["processing_status"] = "processed_full_file"
    except Exception as exc:
        if isinstance(exc, RootOpenFailure):
            access_info = exc.access_info
            file_summary["file_access"] = access_info
        file_summary["read_status"] = "failed"
        file_summary["processing_status"] = "excluded"
        file_summary["error"] = f"{type(exc).__name__}: {exc}"
        error_blob = " ".join([
            str(access_info.get("direct_open_error", "")),
            str(access_info.get("xrdcp_stderr_tail", "")),
            " ".join(str(a.get("stderr_tail", "")) for a in access_info.get("xrdcp_attempts", []) if isinstance(a, dict)),
            str(exc),
        ]).lower()
        external_access_blocker = any(token in error_blob for token in ["redirect limit", "permission denied", "timed out", "operation expired", "certificate", "proxy", "auth"])
        bad.append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "failure_stage": "full_production_open_or_read",
                "exception_type": type(exc).__name__,
                "concise_error": str(exc)[:400],
                "first_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "alternate_access_attempted": bool(access_info.get("alternate_access_attempted", False)),
                "external_access_blocker": external_access_blocker,
                "direct_open_error": access_info.get("direct_open_error"),
                "fallback_status": access_info.get("fallback_status", "not_attempted"),
                "xrdcp_exit_status": access_info.get("xrdcp_exit_status"),
                "xrdcp_stderr_tail": access_info.get("xrdcp_stderr_tail", ""),
                "xrdcp_attempts": access_info.get("xrdcp_attempts", []),
                "permanently_skipped": not external_access_blocker,
            }
        )
        return file_summary, None, bad
    finally:
        try:
            if root is not None:
                root.close()
        except Exception:
            pass
        cleanup_xrd_cache(access_info)
        os.chdir(cwd)
    return file_summary, file_payload, bad


def merge_file_payload(dataset_rec: dict[str, Any], file_payload: dict[str, Any]) -> None:
    dataset_rec["events_read"] += int(file_payload.get("sumw", 0.0) if dataset_rec.get("is_data") else 0)
    if not dataset_rec.get("is_data"):
        dataset_rec["sumw"] += float(file_payload.get("sumw", 0.0))
        dataset_rec["sumw2"] += float(file_payload.get("sumw2", 0.0))
    else:
        dataset_rec["sumw"] += float(file_payload.get("sumw", 0.0))
        dataset_rec["sumw2"] += float(file_payload.get("sumw2", 0.0))
    src = str(file_payload.get("sumw_source", "unknown"))
    dataset_rec["sumw_source_counts"][src] = dataset_rec["sumw_source_counts"].get(src, 0) + 1
    for region, counter in file_payload.get("regions", {}).items():
        merge_counter(dataset_rec["regions"].setdefault(region, empty_counter()), counter)
    for region, by_var in file_payload.get("histograms", {}).items():
        for variable, hist in by_var.items():
            target = dataset_rec["histograms"].setdefault(region, {}).setdefault(variable, empty_hist(hist["bin_edges"]))
            for key in ["raw_values", "raw_sumw2", "entries"]:
                target[key] = [float(a) + float(b) for a, b in zip(target[key], hist[key])]
    for scheme, by_bin in file_payload.get("search_bins", {}).items():
        for bin_name, counter in by_bin.items():
            target = dataset_rec["search_bins"].setdefault(scheme, {}).setdefault(bin_name, empty_counter())
            merge_counter(target, counter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("AUTONOMOUS_ALLHAD_FULL_CHUNK", "50000")))
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    shard_path = Path(args.shard)
    output_path = Path(args.output)
    shard = json.loads(shard_path.read_text())
    start = time.time()
    payload: dict[str, Any] = {
        "schema_version": "full_production_shard_v1",
        "status": "running",
        "shard_id": shard.get("shard_id"),
        "record_digest": shard.get("record_digest"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": None,
        "records_in_shard": len(shard.get("records", [])),
        "files_attempted": 0,
        "files_processed": 0,
        "bad_files": [],
        "file_summaries": [],
        "datasets": {},
        "chunk_size": args.chunk_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    for record in shard.get("records", []):
        payload["files_attempted"] += 1
        dataset_rec = ensure_dataset(payload, record)
        dataset_rec["files_attempted"] += 1
        file_summary, file_payload, bad = process_file(record, repo, args.chunk_size)
        payload["file_summaries"].append(file_summary)
        payload["bad_files"].extend(bad)
        if file_payload is not None:
            payload["files_processed"] += 1
            dataset_rec["files_processed"] += 1
            merge_file_payload(dataset_rec, file_payload)
        write_json(output_path, {**payload, "status": "running"})
    payload["status"] = "complete" if payload["files_processed"] == payload["files_attempted"] else ("complete_with_bad_files" if payload["files_processed"] > 0 else "failed")
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["wall_time_s"] = round(time.time() - start, 3)
    write_json(output_path, payload)
    return 0 if payload["files_processed"] > 0 or payload["files_attempted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
