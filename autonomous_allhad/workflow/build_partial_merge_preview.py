#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DATA_PROCESSES = {"JetMET", "EGamma", "Muon"}
FINAL_STATUSES = {"complete", "complete_with_bad_files"}
LUMI_FB = 109.82
LUMI_PB = LUMI_FB * 1000.0
REGION_MAP = {
    "LLCR": "cat2_LLCR_highDeltaM",
    "QCDCR": "cat3_QCDCR_highDeltaM",
    "GCR": "cat4_GCR_highDeltaM",
    "DY2E": "cat5_DY2E_highDeltaM",
    "DY2M": "cat6_DY2M_highDeltaM",
    "SR": "cat7_SR_highDeltaM",
}
REGION_ORDER = list(REGION_MAP.values())
REGION_SHORT = {v: k for k, v in REGION_MAP.items()}
DATA_PROCESS_BY_REGION = {
    REGION_MAP["LLCR"]: "JetMET",
    REGION_MAP["QCDCR"]: "JetMET",
    REGION_MAP["GCR"]: "EGamma",
    REGION_MAP["DY2E"]: "EGamma",
    REGION_MAP["DY2M"]: "Muon",
    REGION_MAP["SR"]: "JetMET",
}
VARIABLE_LABELS = {
    "recoil_pt": "recoil pT [GeV]",
    "metpt": "MET [GeV]",
    "ht": "HT [GeV]",
    "njet": "Njet",
    "nb": "Nb",
    "nfj": "N AK8 jets",
    "min_dphi": "min Delta phi",
}
PLOT_VARIABLES = ["recoil_pt", "metpt", "ht", "njet", "nb", "nfj", "min_dphi"]
SIGNAL_OVERLAYS = [
    {"key": "mStop1000_mLSP1", "label": "T2tt mStop1000 mLSP1", "color": "#d62728"},
    {"key": "mStop1200_mLSP1", "label": "T2tt mStop1200 mLSP1", "color": "#1f77b4"},
]
SIGNAL_VAR_MAP = {
    "metpt": "met",
    "ht": "ht",
    "njet": "njet",
    "nb": "nb_medium",
    "min_dphi": "min_dphi4",
}
GROUP_ORDER = ["VV", "Single Top", "ttbar", "DY", "Gamma + Jets", "W -> lv", "Z -> vv", "QCD Multijet", "others"]
GROUP_COLORS = {
    "VV": "#6f7661",
    "Single Top": "#8f7cc2",
    "ttbar": "#9ec5b8",
    "DY": "#23c9c8",
    "Gamma + Jets": "#800080",
    "W -> lv": "#eadac8",
    "Z -> vv": "#f2c58f",
    "QCD Multijet": "#d798a5",
    "others": "#6a625f",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def full_region(short: str) -> str:
    return REGION_MAP.get(short, short)


def data_process_for_region(region: str):
    return DATA_PROCESS_BY_REGION.get(full_region(region))


def data_process_allowed(process: str, region: str) -> bool:
    expected = data_process_for_region(region)
    return expected is None or process == expected


def process_to_group(process: str, dataset: str = "") -> str:
    if process in DATA_PROCESSES:
        return "Data"
    if process == "VV":
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "Single Top"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "ttbar"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "Gamma + Jets"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "W -> lv"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Z -> vv"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD Multijet"
    return "others"


def canonical_process(process: str, dataset: str = "") -> str:
    group = process_to_group(process, dataset)
    return {
        "VV": "VV",
        "Single Top": "ST",
        "ttbar": "TT",
        "DY": "DY",
        "Gamma + Jets": "GJ",
        "W -> lv": "WtoLNu",
        "Z -> vv": "Zto2Nu",
        "QCD Multijet": "QCD",
    }.get(group, process or "other")


def physical_dataset_key(dataset: str) -> str:
    return str(dataset or "unknown").split("____", 1)[0]

def empty_counter() -> dict[str, Any]:
    return {"unweighted": 0, "raw_weighted": 0.0, "raw_sumw2": 0.0}


def add_region_counter(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["unweighted"] += int(source.get("unweighted", 0))
    target["raw_weighted"] += float(source.get("raw_weighted", 0.0))
    target["raw_sumw2"] += float(source.get("raw_sumw2", 0.0))


def merge_hist_payload(target: dict[str, Any], source: dict[str, Any], factor: float) -> None:
    edges = [float(x) for x in source.get("bin_edges", [])]
    if not target:
        n = max(0, len(edges) - 1)
        target.update({"bin_edges": edges, "values": [0.0] * n, "sumw2": [0.0] * n, "raw_values": [0.0] * n, "entries": [0.0] * n})
    for idx, raw in enumerate(source.get("raw_values", [])):
        if idx < len(target["values"]):
            target["raw_values"][idx] += float(raw)
            target["values"][idx] += float(raw) * factor
    for idx, raw2 in enumerate(source.get("raw_sumw2", [])):
        if idx < len(target["sumw2"]):
            target["sumw2"][idx] += float(raw2) * factor * factor
    for idx, ent in enumerate(source.get("entries", [])):
        if idx < len(target["entries"]):
            target["entries"][idx] += float(ent)


def source_manifest_name(name: str) -> str:
    return name.replace(".running", "")


def valid_final_payload(payload: dict[str, Any], records_expected: int | None) -> bool:
    if payload.get("status") not in FINAL_STATUSES:
        return False
    attempted = payload.get("files_attempted")
    processed = payload.get("files_processed")
    if not isinstance(attempted, int) or not isinstance(processed, int):
        return False
    if records_expected is not None and attempted != records_expected:
        return False
    if processed < 0 or processed > attempted:
        return False
    return bool(payload.get("datasets"))


def usable_running_payload(payload: dict[str, Any]) -> bool:
    attempted = payload.get("files_attempted")
    processed = payload.get("files_processed")
    if not isinstance(attempted, int) or not isinstance(processed, int):
        return False
    if processed < 0 or processed > attempted:
        return False
    return processed > 0 and bool(payload.get("datasets"))


def select_sources(repo: Path, shard_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    status = Counter()
    examples: dict[str, list[str]] = {"running_used": [], "final_used": [], "missing_or_empty": []}
    for shard_path in sorted(shard_dir.glob("shard_*.json")):
        shard_name = shard_path.name
        try:
            shard = read_json(shard_path)
        except Exception:
            shard = {}
        records_expected = len(shard.get("records") or []) if isinstance(shard, dict) else None
        final_path = output_dir / shard_name
        running_path = output_dir / f"{shard_name}.running"
        candidates: list[tuple[str, Path, dict[str, Any]]] = []
        if final_path.exists():
            try:
                payload = read_json(final_path)
                if isinstance(payload, dict) and valid_final_payload(payload, records_expected):
                    candidates.append(("final", final_path, payload))
                else:
                    status[f"final_{payload.get('status', 'invalid') if isinstance(payload, dict) else 'invalid'}"] += 1
            except Exception:
                status["final_unreadable"] += 1
        if running_path.exists():
            try:
                payload = read_json(running_path)
                if isinstance(payload, dict) and usable_running_payload(payload):
                    candidates.append(("terminal_running_checkpoint", running_path, payload))
                else:
                    status[f"running_{payload.get('status', 'invalid') if isinstance(payload, dict) else 'invalid'}_unused"] += 1
            except Exception:
                status["running_unreadable"] += 1
        if not candidates:
            status["missing_or_empty"] += 1
            if len(examples["missing_or_empty"]) < 5:
                examples["missing_or_empty"].append(shard_name)
            continue
        # Prefer the payload with the most successful files; prefer final on exact ties.
        candidates.sort(key=lambda row: (int(row[2].get("files_processed") or 0), int(row[2].get("files_attempted") or 0), 1 if row[0] == "final" else 0), reverse=True)
        kind, path, payload = candidates[0]
        status[kind] += 1
        if kind == "final" and len(examples["final_used"]) < 5:
            examples["final_used"].append(path.name)
        if kind != "final" and len(examples["running_used"]) < 5:
            examples["running_used"].append(path.name)
        sources.append({"kind": kind, "path": path, "payload": payload, "records_expected": records_expected, "shard_name": shard_name})
    return sources, {"expected_shards": len(list(shard_dir.glob('shard_*.json'))), "source_status_counts": dict(status), "examples": examples}


def merge_background_payloads(repo: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    bad_files: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    files_attempted = 0
    files_processed = 0
    source_shards: list[str] = []
    source_details: list[dict[str, Any]] = []
    for item in sources:
        payload = item["payload"]
        source_shards.append(source_manifest_name(item["path"].name))
        source_details.append({
            "source": item["kind"],
            "path": str(item["path"].relative_to(repo)),
            "shard": item["shard_name"],
            "files_attempted": payload.get("files_attempted", 0),
            "files_processed": payload.get("files_processed", 0),
            "bad_files": len(payload.get("bad_files") or []),
        })
        files_attempted += int(payload.get("files_attempted") or 0)
        files_processed += int(payload.get("files_processed") or 0)
        bad_files.extend(payload.get("bad_files") or [])
        file_summaries.extend(payload.get("file_summaries") or [])
        for ds, rec in (payload.get("datasets") or {}).items():
            if rec.get("process") == "SMS":
                continue
            process = canonical_process(rec.get("process") or "other", ds)
            target = datasets.setdefault(ds, {
                "dataset": ds,
                "process": process,
                "xsec_pb": rec.get("xsec_pb"),
                "is_data": rec.get("is_data"),
                "is_background": rec.get("is_background"),
                "files_attempted": 0,
                "files_processed": 0,
                "events_read": 0,
                "sumw": 0.0,
                "sumw2": 0.0,
                "sumw_source_counts": {},
                "regions": {},
                "histograms": {},
                "search_bins": {},
            })
            target["files_attempted"] += int(rec.get("files_attempted") or 0)
            target["files_processed"] += int(rec.get("files_processed") or 0)
            target["events_read"] += int(rec.get("events_read") or 0)
            target["sumw"] += float(rec.get("sumw") or 0.0)
            target["sumw2"] += float(rec.get("sumw2") or 0.0)
            for key, val in (rec.get("sumw_source_counts") or {}).items():
                target["sumw_source_counts"][key] = target["sumw_source_counts"].get(key, 0) + int(val)
            for region, counter in (rec.get("regions") or {}).items():
                add_region_counter(target["regions"].setdefault(region, empty_counter()), counter)
            for region, by_var in (rec.get("histograms") or {}).items():
                for variable, hist in (by_var or {}).items():
                    merge_hist_payload(target["histograms"].setdefault(region, {}).setdefault(variable, {}), hist, 1.0)
            for scheme, by_bin in (rec.get("search_bins") or {}).items():
                for bin_name, counter in (by_bin or {}).items():
                    dest = target["search_bins"].setdefault(scheme, {}).setdefault(bin_name, empty_counter())
                    add_region_counter(dest, counter)

    physical_norm: dict[str, Any] = {}
    physical_dataset_split_counts = Counter()
    for ds, rec in sorted(datasets.items()):
        proc = rec.get("process", "unknown")
        is_data = bool(rec.get("is_data")) or proc in DATA_PROCESSES
        phys = physical_dataset_key(ds) if not is_data else ds
        physical_dataset_split_counts[phys] += 1
        prec = physical_norm.setdefault(phys, {
            "physical_dataset": phys,
            "process": proc,
            "is_data": is_data,
            "xsec_pb": rec.get("xsec_pb"),
            "sumw": 0.0,
            "sumw2": 0.0,
            "files_processed": 0,
            "files_attempted": 0,
            "split_datasets": [],
            "sumw_source_counts": {},
            "xsec_conflicts": [],
        })
        xs = rec.get("xsec_pb")
        if not is_data and isinstance(xs, (int, float)) and isinstance(prec.get("xsec_pb"), (int, float)) and abs(float(xs) - float(prec["xsec_pb"])) > 1e-12:
            prec["xsec_conflicts"].append({"dataset": ds, "xsec_pb": xs})
        elif prec.get("xsec_pb") is None:
            prec["xsec_pb"] = xs
        prec["sumw"] += float(rec.get("sumw") or 0.0)
        prec["sumw2"] += float(rec.get("sumw2") or 0.0)
        prec["files_processed"] += int(rec.get("files_processed") or 0)
        prec["files_attempted"] += int(rec.get("files_attempted") or 0)
        prec["split_datasets"].append(ds)
        for key, val in (rec.get("sumw_source_counts") or {}).items():
            prec["sumw_source_counts"][key] = prec["sumw_source_counts"].get(key, 0) + int(val)
    for phys, prec in physical_norm.items():
        xs = prec.get("xsec_pb")
        sumw = float(prec.get("sumw") or 0.0)
        if prec.get("is_data"):
            prec["normalization_factor"] = 1.0
            prec["normalization_status"] = "data_unscaled"
        elif prec.get("xsec_conflicts"):
            prec["normalization_factor"] = None
            prec["normalization_status"] = "blocked_inconsistent_xsec_across_split_datasets"
        elif isinstance(xs, (int, float)) and float(xs) > 0 and sumw != 0.0:
            prec["normalization_factor"] = float(xs) * LUMI_PB / sumw
            prec["normalization_status"] = "normalized_with_metadata_xsec_and_physical_dataset_sumw_partial"
        elif not isinstance(xs, (int, float)) or float(xs or 0) <= 0:
            prec["normalization_factor"] = None
            prec["normalization_status"] = "blocked_missing_positive_xsec"
        else:
            prec["normalization_factor"] = None
            prec["normalization_status"] = "blocked_zero_sumw"

    norm_factors: dict[str, Any] = {}
    normalized_by_process: dict[str, Any] = {}
    normalized_by_dataset: dict[str, Any] = {}
    histograms: dict[str, Any] = {"data": {}, "background": {}}
    search_bins: dict[str, Any] = {}
    data_stream_exclusions: dict[str, dict[str, Any]] = {}
    region_totals = {region: {"data": 0.0, "background": 0.0, "signal": 0.0} for region in REGION_ORDER}
    blocked_datasets = []
    for ds, rec in sorted(datasets.items()):
        proc = rec.get("process", "unknown")
        is_data = bool(rec.get("is_data")) or proc in DATA_PROCESSES
        xs = rec.get("xsec_pb")
        sumw = float(rec.get("sumw") or 0.0)
        phys = physical_dataset_key(ds) if not is_data else ds
        prec = physical_norm.get(phys, {})
        factor = prec.get("normalization_factor")
        nstatus = prec.get("normalization_status", "blocked_missing_physical_dataset_norm")
        norm_factors[ds] = {
            "dataset": ds,
            "physical_dataset": phys,
            "process": proc,
            "is_data": is_data,
            "xsec_pb": xs,
            "sumw": sumw,
            "physical_dataset_sumw": prec.get("sumw"),
            "sumw2": rec.get("sumw2", 0.0),
            "sumw_source_counts": rec.get("sumw_source_counts", {}),
            "physical_sumw_source_counts": prec.get("sumw_source_counts", {}),
            "files_processed": rec.get("files_processed", 0),
            "files_attempted": rec.get("files_attempted", 0),
            "physical_files_processed": prec.get("files_processed"),
            "physical_files_attempted": prec.get("files_attempted"),
            "physical_split_datasets": len(prec.get("split_datasets", [])),
            "normalization_factor": factor,
            "normalization_status": nstatus,
        }
        if factor is None:
            blocked_datasets.append(ds)
            continue
        kind = "data" if is_data else "background"
        for region, counter in (rec.get("regions") or {}).items():
            fregion = full_region(region)
            raw = float(counter.get("raw_weighted", 0.0))
            raw2 = float(counter.get("raw_sumw2", 0.0))
            for target in [normalized_by_dataset.setdefault(ds, {}).setdefault(region, {"unweighted": 0, "raw_weighted": 0.0, "normalized_weighted": 0.0, "normalized_sumw2": 0.0}), normalized_by_process.setdefault(proc, {}).setdefault(region, {"unweighted": 0, "raw_weighted": 0.0, "normalized_weighted": 0.0, "normalized_sumw2": 0.0})]:
                target["unweighted"] += int(counter.get("unweighted", 0))
                target["raw_weighted"] += raw
                target["normalized_weighted"] += raw * factor
                target["normalized_sumw2"] += raw2 * factor * factor
            if fregion in region_totals:
                if not is_data or data_process_allowed(proc, fregion):
                    region_totals[fregion][kind] += raw * factor
                else:
                    excluded = data_stream_exclusions.setdefault(fregion, {}).setdefault(proc, {"normalized_weighted": 0.0, "raw_weighted": 0.0, "unweighted": 0})
                    excluded["normalized_weighted"] += raw * factor
                    excluded["raw_weighted"] += raw
                    excluded["unweighted"] += int(counter.get("unweighted", 0))
        for region, by_var in (rec.get("histograms") or {}).items():
            fregion = full_region(region)
            if is_data and not data_process_allowed(proc, fregion):
                continue
            for variable, hist in (by_var or {}).items():
                if str(variable).startswith("search_bin_index::"):
                    continue
                dest = histograms.setdefault(kind, {}).setdefault(variable, {}).setdefault(fregion, {}).setdefault(proc, {})
                merge_hist_payload(dest, hist, factor)
        for scheme, by_bin in (rec.get("search_bins") or {}).items():
            if is_data and not data_process_allowed(proc, REGION_MAP["SR"]):
                continue
            for bin_name, counter in (by_bin or {}).items():
                dest = search_bins.setdefault(scheme, {}).setdefault(bin_name, {}).setdefault(proc, {"unweighted": 0, "raw_weighted": 0.0, "normalized_weighted": 0.0, "normalized_sumw2": 0.0, "kind": kind})
                raw = float(counter.get("raw_weighted", 0.0))
                raw2 = float(counter.get("raw_sumw2", 0.0))
                dest["unweighted"] += int(counter.get("unweighted", 0))
                dest["raw_weighted"] += raw
                dest["normalized_weighted"] += raw * factor
                dest["normalized_sumw2"] += raw2 * factor * factor
    return {
        "datasets": datasets,
        "normalization_factors": norm_factors,
        "physical_normalization_factors": physical_norm,
        "physical_dataset_split_counts": dict(physical_dataset_split_counts),
        "normalization_blocked_datasets": blocked_datasets,
        "region_yields_by_process": normalized_by_process,
        "region_yields_by_dataset": normalized_by_dataset,
        "regions": region_totals,
        "histograms": histograms,
        "search_bins": search_bins,
        "data_region_process_policy": DATA_PROCESS_BY_REGION,
        "data_stream_exclusions": data_stream_exclusions,
        "files_attempted": files_attempted,
        "files_processed": files_processed,
        "bad_files": bad_files,
        "file_summaries": file_summaries,
        "events_processed": sum(int(s.get("events_read") or 0) for s in file_summaries if isinstance(s, dict)),
        "source_shards": sorted(set(source_shards)),
        "source_details": source_details,
    }


def merge_signal_shards(repo: Path) -> dict[str, Any]:
    outdir = repo / "autonomous_allhad/workflow/signal_outputs_fastsim_full"
    datasets: dict[str, Any] = {}
    shard_rows = []
    files_attempted = files_processed = bad = events = 0
    statuses = Counter()
    for path in sorted(outdir.glob("signal_shard_*.json")):
        try:
            payload = read_json(path)
        except Exception:
            statuses["unreadable"] += 1
            continue
        status = str(payload.get("status", "unknown"))
        statuses[status] += 1
        shard_rows.append({"path": str(path.relative_to(repo)), "status": status, "files_attempted": payload.get("files_attempted", 0), "files_processed": payload.get("files_processed", 0), "bad_files": len(payload.get("bad_files") or [])})
        files_attempted += int(payload.get("files_attempted") or 0)
        files_processed += int(payload.get("files_processed") or 0)
        bad += len(payload.get("bad_files") or [])
        for ds, rec in (payload.get("datasets") or {}).items():
            target = datasets.setdefault(ds, {"dataset": ds, "process": rec.get("process"), "files_attempted": 0, "files_processed": 0, "events_read": 0, "sumw": 0.0, "sumw2": 0.0, "regions": {}, "search_bins": {}})
            target["files_attempted"] += int(rec.get("files_attempted") or 0)
            target["files_processed"] += int(rec.get("files_processed") or 0)
            target["events_read"] += int(rec.get("events_read") or 0)
            target["sumw"] += float(rec.get("sumw") or 0.0)
            target["sumw2"] += float(rec.get("sumw2") or 0.0)
            events += int(rec.get("events_read") or 0)
            for region, counter in (rec.get("regions") or {}).items():
                add_region_counter(target["regions"].setdefault(region, empty_counter()), counter)
            for scheme, by_bin in (rec.get("search_bins") or {}).items():
                for bin_name, counter in (by_bin or {}).items():
                    add_region_counter(target["search_bins"].setdefault(scheme, {}).setdefault(bin_name, empty_counter()), counter)
    return {"status": "complete" if statuses.get("complete", 0) == 3 and files_processed == 61 and bad == 0 else "partial", "shards": shard_rows, "shard_status_counts": dict(statuses), "files_attempted": files_attempted, "files_processed": files_processed, "bad_files": bad, "events_processed": events, "datasets": datasets, "normalization_note": "FastSim signal shard JSONs are full-production-style raw aggregates. Mass-point overlays in the preview are loaded from signal_cutflows.json and signal_yields_by_mass.json."}


def load_signal_overlay_inputs(repo: Path) -> dict[str, Any]:
    cutflow_path = repo / "autonomous_allhad/outputs/signal_cutflows.json"
    yields_path = repo / "autonomous_allhad/outputs/signal_yields_by_mass.json"
    overlay_hists: dict[str, Any] = {}
    region_yields: dict[str, Any] = {}
    points: list[dict[str, Any]] = []
    if not cutflow_path.exists() or not yields_path.exists():
        return {
            "status": "missing_sources",
            "points": points,
            "histograms": overlay_hists,
            "region_yields": region_yields,
            "variable_map": SIGNAL_VAR_MAP,
            "source_files": [str(cutflow_path.relative_to(repo)), str(yields_path.relative_to(repo))],
        }
    try:
        cutflows = read_json(cutflow_path)
        mass_yields = read_json(yields_path)
    except Exception as exc:
        return {
            "status": "unreadable_sources",
            "error": str(exc),
            "points": points,
            "histograms": overlay_hists,
            "region_yields": region_yields,
            "variable_map": SIGNAL_VAR_MAP,
            "source_files": [str(cutflow_path.relative_to(repo)), str(yields_path.relative_to(repo))],
        }

    cutflow_hists = (((cutflows.get("signal_histograms") or {}).get("histograms") or {}))
    mass_points = mass_yields.get("mass_points") or {}
    for spec in SIGNAL_OVERLAYS:
        key = spec["key"]
        mass_info = mass_points.get(key) or {}
        factor = mass_info.get("normalization_factor")
        if not isinstance(factor, (int, float)):
            continue
        factor = float(factor)
        point = {
            "key": key,
            "label": spec["label"],
            "color": spec["color"],
            "mStop": mass_info.get("mStop"),
            "mLSP": mass_info.get("mLSP"),
            "xsec_pb": mass_info.get("xsec_pb"),
            "normalization_factor": factor,
            "normalization_status": mass_info.get("normalization_status"),
        }
        points.append(point)
        for short, full in [("SR", REGION_MAP["SR"])]:
            rec = (mass_info.get("regions") or {}).get(short)
            if not isinstance(rec, dict):
                continue
            region_yields.setdefault(full, {})[key] = {
                "label": spec["label"],
                "color": spec["color"],
                "normalized_weighted": float(rec.get("normalized_weighted") or 0.0),
                "normalized_sumw2": float(rec.get("normalized_sumw2") or 0.0),
                "unweighted": int(rec.get("unweighted") or 0),
            }
        by_region = cutflow_hists.get(key) or {}
        for variable, signal_variable in SIGNAL_VAR_MAP.items():
            for short, full in [("SR", REGION_MAP["SR"])]:
                hist = (by_region.get(short) or {}).get(signal_variable)
                if not isinstance(hist, dict):
                    continue
                edges = [float(x) for x in hist.get("bins", [])]
                raw = [float(x) for x in hist.get("raw_weighted", [])]
                raw2 = [float(x) for x in hist.get("raw_sumw2", [])]
                if len(edges) != len(raw) + 1:
                    continue
                if len(raw2) != len(raw):
                    raw2 = [0.0] * len(raw)
                overlay_hists.setdefault(variable, {}).setdefault(full, {})[key] = {
                    "key": key,
                    "label": spec["label"],
                    "color": spec["color"],
                    "bin_edges": edges,
                    "values": [x * factor for x in raw],
                    "sumw2": [x * factor * factor for x in raw2],
                    "normalization_factor": factor,
                    "source_variable": signal_variable,
                }
    return {
        "status": "complete" if points else "no_requested_points",
        "points": points,
        "histograms": overlay_hists,
        "region_yields": region_yields,
        "variable_map": SIGNAL_VAR_MAP,
        "source_files": [str(cutflow_path.relative_to(repo)), str(yields_path.relative_to(repo))],
        "missing_variable_note": "Mass-point overlays are drawn only in cat7/SR for metpt, ht, njet, nb, and min_dphi; recoil_pt and nfj have no matching signal_cutflows histogram.",
    }


def sum_hist_list(hists: list[dict[str, Any]]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for hist in hists:
        merge_hist_payload(out, {"bin_edges": hist.get("bin_edges", []), "raw_values": hist.get("values", []), "raw_sumw2": hist.get("sumw2", []), "entries": hist.get("entries", [])}, 1.0)
    return out or None


def group_background_hists(payload: dict[str, Any], variable: str, region: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_proc = (((payload.get("histograms") or {}).get("background") or {}).get(variable) or {}).get(region, {})
    for proc, hist in by_proc.items():
        group = process_to_group(proc)
        merge_hist_payload(out.setdefault(group, {}), {"bin_edges": hist.get("bin_edges", []), "raw_values": hist.get("values", []), "raw_sumw2": hist.get("sumw2", []), "entries": hist.get("entries", [])}, 1.0)
    return out


def plot_variable(payload: dict[str, Any], variable: str, region: str, outbase: Path) -> dict[str, Any] | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")

    bkg = group_background_hists(payload, variable, region)
    data_by_proc = (((payload.get("histograms") or {}).get("data") or {}).get(variable) or {}).get(region, {})
    signal_by_key = (((payload.get("signal_overlay_hists") or {}).get(variable) or {}).get(region) or {}) if region == REGION_MAP["SR"] else {}
    if not bkg and not data_by_proc and not signal_by_key:
        return None
    ref = next(iter(bkg.values()), None) or next(iter(data_by_proc.values()), None) or next(iter(signal_by_key.values()), None)
    edges = np.asarray(ref.get("bin_edges", []), dtype=float)
    if len(edges) < 2:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}, sharex=True)
    ax, rax = axes

    total = np.zeros(len(centers))
    total_s2 = np.zeros(len(centers))
    stack_inputs = []
    stack_weights = []
    stack_colors = []
    stack_labels = []
    for group in GROUP_ORDER:
        hist = bkg.get(group)
        if not hist:
            continue
        vals = np.asarray(hist.get("values", []), dtype=float)
        s2 = np.asarray(hist.get("sumw2", []), dtype=float)
        if len(vals) != len(centers):
            continue
        stack_inputs.append(centers.copy())
        stack_weights.append(vals)
        stack_colors.append(GROUP_COLORS.get(group, "0.7"))
        stack_labels.append(group)
        total += vals
        if len(s2) == len(centers):
            total_s2 += s2
    if stack_inputs:
        ax.hist(
            stack_inputs,
            bins=edges,
            weights=stack_weights,
            stacked=True,
            histtype="stepfilled",
            color=stack_colors,
            label=stack_labels,
            edgecolor="black",
            linewidth=0.7,
        )

    unc = np.sqrt(total_s2)
    if np.any(total > 0):
        lower = np.maximum(total - unc, 1e-12)
        upper = np.maximum(total + unc, 1e-12)
        ax.fill_between(
            edges,
            np.r_[lower, lower[-1]],
            np.r_[upper, upper[-1]],
            step="post",
            facecolor="0.85",
            edgecolor="0.35",
            hatch="////",
            linewidth=0.0,
            alpha=0.35,
            label="Stat. unc.",
        )

    signal_records = []
    signal_arrays = []
    for spec in SIGNAL_OVERLAYS:
        hist = signal_by_key.get(spec["key"])
        if not hist:
            continue
        sig_edges = np.asarray(hist.get("bin_edges", []), dtype=float)
        vals = np.asarray(hist.get("values", []), dtype=float)
        if len(vals) != len(centers) or len(sig_edges) != len(edges) or not np.allclose(sig_edges, edges):
            continue
        ax.hist(
            centers,
            bins=edges,
            weights=vals,
            histtype="step",
            linewidth=2.0,
            color=hist.get("color", spec["color"]),
            label=hist.get("label", spec["label"]),
        )
        signal_arrays.append(vals)
        signal_records.append({
            "key": spec["key"],
            "label": hist.get("label", spec["label"]),
            "normalization_factor": hist.get("normalization_factor"),
            "source_variable": hist.get("source_variable"),
            "yield": float(np.sum(vals)),
        })

    blinded = region.endswith("SR_highDeltaM")
    data_vals = np.zeros(len(centers))
    data_s2 = np.zeros(len(centers))
    for hist in data_by_proc.values():
        vals = np.asarray(hist.get("values", []), dtype=float)
        s2 = np.asarray(hist.get("sumw2", []), dtype=float)
        if len(vals) == len(centers):
            data_vals += vals
            if len(s2) == len(centers):
                data_s2 += s2
    if not blinded and np.any(data_vals > 0):
        ax.errorbar(centers, data_vals, yerr=np.sqrt(data_s2), fmt="o", color="black", markersize=4, label="Data 2024")
        ratio = np.divide(data_vals, total, out=np.full_like(data_vals, np.nan), where=total > 0)
        ratio_err = np.divide(np.sqrt(data_s2), total, out=np.full_like(data_vals, np.nan), where=total > 0)
        rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o", color="black", markersize=3)
    rel = np.divide(unc, total, out=np.zeros_like(unc), where=total > 0)
    rax.fill_between(centers, np.maximum(0, 1 - rel), 1 + rel, step="mid", color="0.85")
    rax.axhline(1.0, color="0.45", linewidth=1)
    rax.set_ylim(0, 2)
    rax.set_ylabel("Data/MC")
    rax.set_xlabel(VARIABLE_LABELS.get(variable, variable))
    ax.set_ylabel("Events")

    positive = []
    for arr in [total + unc] + ([] if blinded else [data_vals + np.sqrt(data_s2)]) + signal_arrays:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    if positive:
        ax.set_yscale("log")
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "name": outbase.name,
        "variable": variable,
        "region": region,
        "png": str(Path("plots") / outbase.with_suffix(".png").name),
        "pdf": str(Path("plots") / outbase.with_suffix(".pdf").name),
        "blinded": blinded,
        "signal_overlays": signal_records,
    }


def plot_region_summary(payload: dict[str, Any], outbase: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    regions = REGION_ORDER
    x = np.arange(len(regions), dtype=float)
    bins = np.arange(-0.5, len(regions) + 0.5, 1.0)
    bkg = np.asarray([payload["regions"][r]["background"] for r in regions], dtype=float)
    data = np.asarray([payload["regions"][r]["data"] for r in regions], dtype=float)
    labels = [r.split("_")[1] for r in regions]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(x, bins=bins, weights=bkg, histtype="stepfilled", color="#9ec5b8", edgecolor="black", linewidth=0.7, label="Background")
    ax.errorbar(x[:-1], data[:-1], yerr=np.sqrt(data[:-1]), fmt="o", color="black", label="Data 2024")
    signal_records = []
    positive = [v for v in bkg.tolist() + data[:-1].tolist() if v > 0]
    ax.set_xticks(x, labels)
    if positive:
        ax.set_yscale("log")
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    ax.set_ylabel("Events")
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=9, frameon=False)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"name": outbase.name, "png": str(Path("plots") / outbase.with_suffix(".png").name), "pdf": str(Path("plots") / outbase.with_suffix(".pdf").name), "blinded_regions": [REGION_MAP["SR"]], "signal_overlays": signal_records}


def copy_preview_to_docs(preview_dir: Path, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ["partial_normalized_yields.json", "partial_normalization_factors.json", "partial_preview_manifest.json", "partial_plot_summary.json", "partial_plot_summary.md", "partial_yield_summary.md", "fastsim_signal_partial_summary.json", "signal_overlay_summary.json"]:
        src = preview_dir / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
    plot_src = preview_dir / "plots"
    plot_dst = docs_dir / "plots"
    plot_dst.mkdir(parents=True, exist_ok=True)
    if plot_src.exists():
        for src in plot_src.glob("partial_*.png"):
            shutil.copy2(src, plot_dst / src.name)
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, plot_dst / pdf.name)


def write_preview_index(docs_dir: Path, plot_records: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    cards = []
    cache = str(payload.get("created_at", utc_now())).replace(":", "").replace("-", "")
    title = str(payload.get("preview_label") or "Partial Merge Preview")
    for rec in plot_records:
        rel = "plots/" + Path(rec["png"]).name
        rel_v = f"{rel}?v={cache}"
        cards.append(f"<a class='plot' href='{rel_v}'><img src='{rel_v}' loading='lazy'><span>{Path(rec['png']).stem}</span></a>")
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f6f7f9;color:#111}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}}.plot{{display:block;background:white;border:1px solid #ddd;padding:10px;color:#123;text-decoration:none}}img{{width:100%;max-width:100%;display:block}}code{{background:#eee;padding:2px 4px}}</style></head>
<body><h1>{title}</h1>
<p>Snapshot: <code>{payload['created_at']}</code>. Sources: <code>{len(payload['source_shards'])}</code> shard payloads; files processed <code>{payload['files_processed']}</code>; bad entries <code>{payload['bad_files']}</code>.</p>
<p>This preview includes valid final shard JSONs plus terminal <code>.json.running</code> checkpoints at snapshot time. It is not a final production result.</p>
<p><a href='partial_normalized_yields.json'>partial_normalized_yields.json</a> · <a href='partial_normalization_factors.json'>partial_normalization_factors.json</a> · <a href='fastsim_signal_partial_summary.json'>FastSim signal summary</a> · <a href='signal_overlay_summary.json'>signal overlay summary</a> · <a href='partial_yield_summary.md'>summary</a></p>
<div class='grid'>{''.join(cards)}</div>
</body></html>"""
    (docs_dir / "index.html").write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/eos/user/t/taiwoo/run3_stop/decaf")
    parser.add_argument("--preview-dir", default="autonomous_allhad/workflow/partial_merge_preview_878535")
    parser.add_argument("--docs-dir", default="docs/partial_merge_preview_878535")
    parser.add_argument("--output-dir", default="autonomous_allhad/workflow/production_outputs_eos_full")
    parser.add_argument("--shard-dir", default="autonomous_allhad/workflow/production_shards_eos_full")
    parser.add_argument("--label", default="Partial Merge Preview 878535")
    parser.add_argument("--source-cluster-id", default="878535")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    preview_dir = (repo / args.preview_dir).resolve()
    docs_dir = (repo / args.docs_dir).resolve()
    output_dir = (repo / args.output_dir).resolve()
    shard_dir = (repo / args.shard_dir).resolve()
    sources, shard_status = select_sources(repo, shard_dir, output_dir)
    merged = merge_background_payloads(repo, sources)
    signal = merge_signal_shards(repo)
    signal_overlays = load_signal_overlay_inputs(repo)
    recovery = {}
    rec_path = repo / "autonomous_allhad/workflow/local_checkpoint_recovery_878535_current.json"
    if rec_path.exists():
        try:
            recovery = read_json(rec_path)
        except Exception:
            recovery = {}
    payload = {
        "schema_version": "partial_merge_preview_v2",
        "status": "partial_preview",
        "scope": "PRELIMINARY partial merge over valid final shard JSONs plus terminal .json.running checkpoints at snapshot time; not a final production output.",
        "created_at": utc_now(),
        "preview_label": args.label,
        "source_cluster_id": args.source_cluster_id,
        "source_output_directory": str(output_dir.relative_to(repo)),
        "preview_directory": str(preview_dir.relative_to(repo)),
        "shard_directory": str(shard_dir.relative_to(repo)),
        "luminosity_fb": LUMI_FB,
        "luminosity_pb": LUMI_PB,
        "formula": "DATA factor=1.0; MC normalization_factor = xsec_pb * lumi_pb / physical_dataset_sumw, where physical_dataset strips metadata split suffixes like ____12_; selected final/running payloads only.",
        "normalization_grouping_policy": "MC metadata records split as <dataset>____N_ share one physical-dataset denominator; this prevents reapplying the full cross section once per split shard.",
        "data_region_process_policy": merged["data_region_process_policy"],
        "data_stream_exclusions": merged["data_stream_exclusions"],
        "sms_policy": "SMS FastSim mass-point overlays are drawn only in cat7/SR plots for mStop1000/mLSP1 and mStop1200/mLSP1 where signal_cutflows histograms exist; completed FastSim shard JSONs are summarized separately in fastsim_signal_partial_summary.json.",
        "final_normalization_complete": False,
        "normalization_status": "partial_preview_complete" if not merged["normalization_blocked_datasets"] else "partial_preview_incomplete_blocked_datasets",
        "normalization_blocked_datasets": merged["normalization_blocked_datasets"],
        "normalization_factors": merged["normalization_factors"],
        "physical_normalization_factors": merged["physical_normalization_factors"],
        "physical_dataset_split_counts": merged["physical_dataset_split_counts"],
        "files_attempted": merged["files_attempted"],
        "files_processed": merged["files_processed"],
        "bad_files": len(merged["bad_files"]),
        "bad_file_categories": dict(Counter(str(b.get("exception_type", "unknown")) for b in merged["bad_files"] if isinstance(b, dict))),
        "events_processed": merged["events_processed"],
        "region_yields_by_process": merged["region_yields_by_process"],
        "region_yields_by_dataset": merged["region_yields_by_dataset"],
        "regions": merged["regions"],
        "histograms": merged["histograms"],
        "search_bins": merged["search_bins"],
        "source_shards": merged["source_shards"],
        "source_details_preview": merged["source_details"][:200],
        "shard_status": {**shard_status, "selected_source_payloads": len(sources), "final_payloads_used": sum(1 for s in sources if s["kind"] == "final"), "running_payloads_used": sum(1 for s in sources if s["kind"] != "final")},
        "local_recovery_snapshot": {"candidate_kind_counts": recovery.get("candidate_kind_counts"), "status_counts": recovery.get("status_counts"), "apply_status_counts": recovery.get("apply_status_counts"), "results_done": len(recovery.get("results") or []), "updated_at": recovery.get("updated_at")},
        "fastsim_signal_summary": {k: signal.get(k) for k in ["status", "shard_status_counts", "files_attempted", "files_processed", "bad_files", "events_processed", "normalization_note"]},
        "signal_overlay_summary": {k: signal_overlays.get(k) for k in ["status", "points", "variable_map", "source_files", "missing_variable_note"]},
        "signal_overlay_hists": signal_overlays.get("histograms", {}),
        "signal_region_overlay_yields": signal_overlays.get("region_yields", {}),
        "warnings": [
            "This is based on a live partial recovery snapshot and will change as more .running records are recovered.",
            "Process composition may be biased until recovery and all shard promotion are complete.",
            "Do not use this preview for Combine, final datacards, or physics conclusions.",
        ],
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_json(preview_dir / "partial_normalized_yields.json", payload)
    write_json(preview_dir / "partial_normalization_factors.json", {"schema_version": "partial_normalization_factors_v3", "status": "partial_preview", "normalization_status": payload["normalization_status"], "expected_shards": shard_status["expected_shards"], "source_completed_or_checkpoint_shards": len(sources), "luminosity_fb": LUMI_FB, "luminosity_pb": LUMI_PB, "formula": payload["formula"], "normalization_grouping_policy": payload["normalization_grouping_policy"], "final_normalization_complete": False, "factors": merged["normalization_factors"], "physical_factors": merged["physical_normalization_factors"], "physical_dataset_split_counts": merged["physical_dataset_split_counts"], "warnings": payload["warnings"]})
    write_json(preview_dir / "fastsim_signal_partial_summary.json", signal)
    write_json(preview_dir / "signal_overlay_summary.json", {**payload["signal_overlay_summary"], "region_yields": payload["signal_region_overlay_yields"]})

    plot_records: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        for variable in PLOT_VARIABLES:
            rec = plot_variable(payload, variable, region, preview_dir / "plots" / f"partial_{variable}_{region}")
            if rec:
                plot_records.append(rec)
    plot_records.append(plot_region_summary(payload, preview_dir / "plots" / "partial_cr_sr_region_bins"))
    plot_summary = {"status": "partial_preview_cms_style", "timestamp_utc": payload["created_at"], "source": f"{payload['preview_directory']}/partial_normalized_yields.json", "completed_or_checkpoint_shards_used": len(sources), "variables": PLOT_VARIABLES, "regions": REGION_ORDER, "region_variable_plots": plot_records, "search_bin_note": "cat7 SR data are blinded. Signal overlays are included only for cat7/SR plots for mStop1000/mLSP1 and mStop1200/mLSP1 where matching signal_cutflows histograms exist."}
    write_json(preview_dir / "partial_plot_summary.json", plot_summary)
    summary_lines = [
        f"# {args.label}",
        "",
        f"Timestamp UTC: `{payload['created_at']}`",
        "",
        "This is a live partial preview over valid final shard JSONs plus terminal `.json.running` checkpoints.",
        "It is not a final production output and must not be used for Combine or final datacards.",
        "",
        f"Source payloads used: `{len(sources)}` (`{payload['shard_status']['final_payloads_used']}` final, `{payload['shard_status']['running_payloads_used']}` running checkpoints)",
        f"Files attempted in selected payloads: `{payload['files_attempted']}`",
        f"Files processed in selected payloads: `{payload['files_processed']}`",
        f"Bad file entries retained: `{payload['bad_files']}`",
        f"Events processed: `{payload['events_processed']}`",
        f"FastSim signal shards: `{signal['files_processed']}/{signal['files_attempted']}` processed, bad `{signal['bad_files']}`",
        "",
        "| region | data | background |",
        "|---|---:|---:|",
    ]
    for region in REGION_ORDER:
        vals = payload["regions"][region]
        summary_lines.append(f"| {region} | {vals['data']:.6g} | {vals['background']:.6g} |")
    (preview_dir / "partial_yield_summary.md").write_text("\n".join(summary_lines) + "\n")
    (preview_dir / "partial_plot_summary.md").write_text("\n".join(["# Partial Plot Summary", "", f"Plots: `{len(plot_records)}`", f"Source payloads used: `{len(sources)}`", "", "cat7 SR data are blinded. Signal overlays are included only in cat7/SR plots for mStop1000/mLSP1 and mStop1200/mLSP1 where available."]) + "\n")
    write_json(preview_dir / "partial_preview_manifest.json", {"status": "partial_preview_complete", "created_at": payload["created_at"], "preview_directory": payload["preview_directory"], "plot_status": plot_summary["status"], "shard_status": payload["shard_status"], "artifacts": ["partial_normalized_yields.json", "partial_normalization_factors.json", "fastsim_signal_partial_summary.json", "partial_plot_summary.json", "partial_yield_summary.md"]})
    copy_preview_to_docs(preview_dir, docs_dir)
    write_preview_index(docs_dir, plot_records, payload)
    print(json.dumps({"status": "partial_preview_complete", "source_payloads": len(sources), "final_payloads": payload["shard_status"]["final_payloads_used"], "running_payloads": payload["shard_status"]["running_payloads_used"], "files_processed": payload["files_processed"], "bad_files": payload["bad_files"], "fastsim_processed": signal["files_processed"], "plots": len(plot_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
