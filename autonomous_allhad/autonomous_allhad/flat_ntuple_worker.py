from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from .signal_models import (
    signal_genmodel_branch,
    signal_runs_sumw_branch,
)

from .real_subset_worker import (
    CORE_BRANCHES,
    ELECTRON_HLT,
    ELECTRON_REFERENCE_HLT,
    FILTERS,
    HT_REFERENCE_HLT,
    MUON_HLT,
    PHOTON_HLT,
    SIGNAL_HLT,
    RootOpenFailure,
    cleanup_xrd_cache,
    extract_chunk,
    finite_float,
    open_root_with_xrd_fallback,
    validate_shift_name,
)

SCHEMA_VERSION = "flat_ntuple_shard_v5_fullselection_2024"
TREE_NAME = "Events"
LUMI_FB = 109.82
LUMI_PB = LUMI_FB * 1000.0
DATA_PROCESSES = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}

DEFERRED_WEIGHT_VARIATIONS = [
    "pileupUp",
    "pileupDown",
    "btagSF_bc_correlatedUp",
    "btagSF_bc_correlatedDown",
    "btagSF_bc_uncorrelatedUp",
    "btagSF_bc_uncorrelatedDown",
    "btagSF_light_correlatedUp",
    "btagSF_light_correlatedDown",
    "btagSF_light_uncorrelatedUp",
    "btagSF_light_uncorrelatedDown",
    "electron_idUp",
    "electron_idDown",
    "electron_hltUp",
    "electron_hltDown",
    "muon_idUp",
    "muon_idDown",
    "muon_hltUp",
    "muon_hltDown",
    "photon_idUp",
    "photon_idDown",
    "veto_electron_5to10Up",
    "veto_electron_5to10Down",
    "loose_muon_5to10Up",
    "loose_muon_5to10Down",
    "photon_triggerUp",
    "photon_triggerDown",
    "met_triggerUp",
    "met_triggerDown",
]


INT64_FIELDS = ["run", "luminosityBlock", "event", "entry"]
INT32_FIELDS = [
    "dataset_id",
    "physical_dataset_id",
    "process_id",
    "file_id",
    "year",
    "mStop",
    "mLSP",
    "signal_topology_id",
    "njet",
    "nb_medium",
    "nb_loose",
    "nb_medium_lowdm",
    "nb_loose_lowdm",
    "njet_photon_clean",
    "nb_photon_clean",
    "njet_lepton_clean",
    "nb_lepton_clean",
    "nfj",
    "nboosted_top",
    "nboosted_w",
    "nboosted_total",
    "n_lowdm_isr",
    "n_sv_softb",
    "lowdm_search_bin",
    "lowdm_search_bin_LLCR",
    "lowdm_search_bin_QCDCR",
    "lowdm_search_bin_GCR",
    "lowdm_search_bin_DY2E",
    "lowdm_search_bin_DY2M",
    "lowdm_search_bin_SR",
    "n_e_veto",
    "n_e_medium",
    "n_m_loose",
    "n_m_medium",
    "n_photon_medium",
]
FLOAT_FIELDS = [
    "met",
    "met_phi",
    "ht",
    "ht_photon_clean",
    "ht_lepton_clean",
    "j1pt",
    "j1eta",
    "j1phi",
    "j2pt",
    "j1_met_dphi",
    "j2_met_dphi",
    "j3_met_dphi",
    "j4_met_dphi",
    "min_dphi4",
    "fj1pt",
    "fj1eta",
    "fj1phi",
    "fj1mass",
    "fj1msd",
    "fj1_top_score",
    "fj1_w_score",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
    "lowdm_met_sqrt_ht",
    "lowdm_ptb",
    "lowdm_mtb",
    "lowdm_isr_subjet_btag_max",
    "mee",
    "pee",
    "mmm",
    "pmm",
    "recoil_gcr",
    "recoil_gcr_phi",
    "recoil_dy2e",
    "recoil_dy2m",
    "recoil_dy2e_phi",
    "recoil_dy2m_phi",
    "gen_weight",
    "pu_ntrueint",
]
BOOL_FIELDS = [
    "is_data",
    "is_background",
    "is_signal",
    "feature_flat_preselection",
    "feature_lowdm_preselection",
    "feature_lowdm_sr_base",
    "feature_met_trigger_genuine_measurement",
    "feature_met_trigger_qcd_measurement",
    "feature_photon_trigger_measurement",
    "feature_lowdm_LLCR",
    "feature_lowdm_QCDCR",
    "feature_lowdm_GCR",
    "feature_lowdm_DY2E",
    "feature_lowdm_DY2M",
    "feature_lowdm_SR",
    "feature_preselection",
    "feature_LLCR",
    "feature_QCDCR",
    "feature_GCR",
    "feature_DY2E",
    "feature_DY2M",
    "feature_SR",
    "feature_SR_Nt1",
    "pass_lowdm_topology_veto",
    "pass_lowdm_isr",
    "pass_lowdm_isr_bveto",
    "lowdm_isr_subjet_bveto_available",
    "pass_lowdm_met_sqrt_ht",
    "pass_lowdm_mtb",
    "pass_base_common",
    "pass_any_analysis_trigger",
    "pass_signal_trigger",
    "pass_photon_trigger",
    "pass_ht_reference_trigger",
    "pass_electron_trigger",
    "pass_electron_reference_trigger",
    "pass_muon_trigger",
    "pass_zero_tau",
    "pass_no_tracks",
    "pass_jet_veto_map",
    "pass_no_veto_leptons",
    "pass_one_veto_lepton",
    "pass_mt_100",
    "pass_met_250",
    "pass_common_recoil_or_met",
    "pass_ht_300",
    "pass_ht_photon_300",
    "pass_ht_lepton_300",
    "pass_common_ht",
    "pass_common_njet2",
    "pass_open_pre",
    "pass_open_high",
    "pass_qcd_open",
    "pass_dphi123_0p1",
    "pass_dy2e_ut_250",
    "pass_dy2m_ut_250",
    "pass_dy2e_open_high",
    "pass_dy2m_open_high",
]

VECTOR_FLOAT_FIELDS = [
    "good_jet_pt",
    "good_jet_eta",
    "good_jet_phi",
    "good_jet_btag_upart",
    "lowdm_fatjet_pt",
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
    "lowdm_fatjet_msd",
    "electron_veto_pt",
    "electron_veto_eta",
    "electron_veto_eta_sc",
    "electron_veto_phi",
    "electron_medium_pt",
    "electron_medium_eta",
    "electron_medium_eta_sc",
    "electron_medium_phi",
    "muon_loose_pt",
    "muon_loose_eta",
    "muon_loose_phi",
    "muon_medium_pt",
    "muon_medium_eta",
    "muon_medium_phi",
    "photon_medium_pt",
    "photon_medium_eta",
    "photon_medium_phi",
    "gen_top_pt",
]
VECTOR_INT_FIELDS = [
    "good_jet_hadron_flavour",
    "good_jet_b_loose",
    "good_jet_b_medium",
    "lowdm_fatjet_subjet_idx1",
    "lowdm_fatjet_subjet_idx2",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def sanitize_branch(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return out or "unnamed"


def stable_id(text: str) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def physical_dataset_key(dataset: str, is_data: bool = False) -> str:
    return str(dataset or "unknown") if is_data else str(dataset or "unknown").split("____", 1)[0]


def finite(value: Any, fill: float = 0.0) -> float:
    return finite_float(value, fill)


def int_value(value: Any, fill: int = -1) -> int:
    try:
        if value == "" or value is None:
            return fill
        out = int(value)
    except Exception:
        return fill
    return out


def numeric_xsec(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def split_keys(root: Any) -> set[str]:
    return {str(k).split(";", 1)[0] for k in root.keys()}


def sum_tree_branch(tree: Any, branch: str) -> float | None:
    try:
        vals = np.asarray(tree[branch].array(library="np"), dtype=float)
    except Exception:
        return None
    vals = vals[np.isfinite(vals)]
    return float(np.sum(vals)) if vals.size else 0.0


def genmodel_branch_from_runs_sumw(branch: str) -> str:
    if branch.startswith("genEventSumw_"):
        return "GenModel_" + branch[len("genEventSumw_"):]
    return branch


def mass_from_genmodel(branch: str) -> tuple[int | None, int | None]:
    nums = re.findall(r"(\d+)", str(branch))
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return None, None


def read_runs_sumw(root: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "generic_sumw": None,
        "generic_sumw_branch": None,
        "generic_candidates": [],
        "signal_sumw_by_genmodel": {},
        "signal_runs_branches": [],
    }
    if "Runs" not in split_keys(root):
        return info
    runs = root["Runs"]
    branches = [str(b).split(";", 1)[0] for b in runs.keys()]
    generic = [b for b in branches if b == "genEventSumw"]
    generic += [
        b for b in branches
        if "geneventsumw" in b.lower() and not signal_runs_sumw_branch(b) and b not in generic
    ]
    info["generic_candidates"] = generic
    for branch in generic:
        val = sum_tree_branch(runs, branch)
        if val is not None:
            info["generic_sumw"] = val
            info["generic_sumw_branch"] = branch
            break
    for branch in sorted(b for b in branches if signal_runs_sumw_branch(b)):
        val = sum_tree_branch(runs, branch)
        if val is None:
            continue
        genmodel = genmodel_branch_from_runs_sumw(branch)
        info["signal_sumw_by_genmodel"][genmodel] = val
        info["signal_runs_branches"].append(branch)
    return info


def empty_dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process_group") or record.get("process") or "unknown")
    is_data = bool(record.get("is_data") or process in DATA_PROCESSES)
    is_signal = bool(record.get("is_signal") or process == "SMS")
    is_background = bool(record.get("is_background") if "is_background" in record else (not is_data and not is_signal))
    physical = physical_dataset_key(dataset, is_data=is_data)
    return {
        "dataset": dataset,
        "dataset_id": stable_id(dataset),
        "physical_dataset": physical,
        "physical_dataset_id": stable_id(physical),
        "process": process,
        "process_id": stable_id(process),
        "xsec_pb": numeric_xsec(record.get("xsec_pb")),
        "raw_xsec_pb": record.get("xsec_pb"),
        "is_data": is_data,
        "is_signal": is_signal,
        "is_background": is_background,
        "files_attempted": 0,
        "files_processed": 0,
        "events_read": 0,
        "events_written": 0,
        "sumw": 0.0,
        "sumw2": 0.0,
        "sumw_source_counts": {},
        "signal_sumw_by_genmodel": {},
        "signal_event_genweight_sum_by_genmodel": {},
        "signal_runs_sumw_source_counts": {},
        "normalization_note": "ROOT event weights are raw; aggregate all shard sidecars before applying luminosity/xsec factors.",
    }


def bump(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = int(counter.get(key, 0)) + int(amount)


def add_float_map(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, val in source.items():
        target[key] = float(target.get(key, 0.0)) + finite(val, 0.0)


def row_passes(row: dict[str, Any], skim_flag: str, keep_all: bool) -> bool:
    if keep_all:
        return True
    return bool(row.get(skim_flag, False))


def branch_types() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in INT64_FIELDS:
        out[field] = np.int64
    for field in INT32_FIELDS:
        out[field] = np.int32
    for field in FLOAT_FIELDS:
        out[field] = np.float64
    for field in BOOL_FIELDS:
        out[field] = np.bool_
    for field in VECTOR_FLOAT_FIELDS:
        out[field] = "var * float64"
    for field in VECTOR_INT_FIELDS:
        out[field] = "var * int32"
    return out


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for field in INT64_FIELDS:
        arrays[field] = np.asarray([int_value(row.get(field), -1) for row in rows], dtype=np.int64)
    for field in INT32_FIELDS:
        arrays[field] = np.asarray([int_value(row.get(field), -1) for row in rows], dtype=np.int32)
    for field in FLOAT_FIELDS:
        arrays[field] = np.asarray([finite(row.get(field), 0.0) for row in rows], dtype=np.float64)
    for field in BOOL_FIELDS:
        arrays[field] = np.asarray([bool(row.get(field, False)) for row in rows], dtype=np.bool_)
    for field in VECTOR_FLOAT_FIELDS:
        arrays[field] = ak.Array([[finite(x, 0.0) for x in (row.get(field) or [])] for row in rows])
    for field in VECTOR_INT_FIELDS:
        arrays[field] = ak.Array([[int_value(x, 0) for x in (row.get(field) or [])] for row in rows])
    return arrays


class RootRowWriter:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        self.root_file: Any = None

    def _ensure_open(self) -> None:
        if self.root_file is not None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.root_file = uproot.recreate(self.tmp)
        self.root_file.mktree(TREE_NAME, branch_types())

    def extend(self, rows: list[dict[str, Any]]) -> None:
        self._ensure_open()
        if rows:
            self.root_file[TREE_NAME].extend(rows_to_arrays(rows))

    def close(self, commit: bool) -> None:
        if self.root_file is None:
            if commit:
                write_root(self.output, [])
            return
        try:
            self.root_file.close()
            self.root_file = None
            if commit:
                os.replace(self.tmp, self.output)
        finally:
            if not commit:
                try:
                    self.tmp.unlink(missing_ok=True)
                except Exception:
                    pass


def decorate_row(row: dict[str, Any], record: dict[str, Any], file_id: int) -> dict[str, Any]:
    dataset = str(record.get("dataset") or row.get("dataset") or "unknown")
    process = str(record.get("process_group") or row.get("process") or "unknown")
    is_data = bool(record.get("is_data") or process in DATA_PROCESSES)
    is_signal = bool(record.get("is_signal") or process == "SMS")
    is_background = bool(record.get("is_background") if "is_background" in record else (not is_data and not is_signal))
    physical = physical_dataset_key(dataset, is_data=is_data)
    out = dict(row)
    out.update({
        "dataset_id": stable_id(dataset),
        "physical_dataset_id": stable_id(physical),
        "process_id": stable_id(process),
        "file_id": file_id,
        "year": int_value(row.get("year", record.get("year")), -1),
        "is_data": is_data,
        "is_signal": is_signal,
        "is_background": is_background,
        "feature_SR_Nt1": bool(row.get("feature_SR")) and int_value(row.get("nboosted_top"), 0) >= 1,
    })
    return out


def signal_fallback_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    accum: dict[str, float] = {}
    for row in rows:
        branch = str(row.get("genmodel_branch") or "")
        if not branch:
            continue
        accum[branch] = accum.get(branch, 0.0) + finite(row.get("gen_weight"), 0.0)
    return accum


def update_signal_fallback(meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    add_float_map(meta["signal_event_genweight_sum_by_genmodel"], signal_fallback_from_rows(rows))


def update_physical_norm(payload: dict[str, Any]) -> None:
    physical: dict[str, Any] = {}
    split_counts: dict[str, int] = {}
    for dsid, rec in payload["datasets"].items():
        phys = rec["physical_dataset"]
        split_counts[phys] = split_counts.get(phys, 0) + 1
        prec = physical.setdefault(phys, {
            "physical_dataset": phys,
            "physical_dataset_id": stable_id(phys),
            "process": rec.get("process"),
            "is_data": rec.get("is_data"),
            "is_signal": rec.get("is_signal"),
            "xsec_pb": rec.get("xsec_pb"),
            "sumw": 0.0,
            "sumw2": 0.0,
            "files_processed": 0,
            "files_attempted": 0,
            "events_written": 0,
            "split_dataset_ids": [],
            "sumw_source_counts": {},
            "xsec_conflicts": [],
        })
        xs = rec.get("xsec_pb")
        if not rec.get("is_data") and xs is not None and prec.get("xsec_pb") is not None and abs(float(xs) - float(prec["xsec_pb"])) > 1.0e-12:
            prec["xsec_conflicts"].append({"dataset_id": dsid, "xsec_pb": xs})
        elif prec.get("xsec_pb") is None:
            prec["xsec_pb"] = xs
        prec["sumw"] += finite(rec.get("sumw"), 0.0)
        prec["sumw2"] += finite(rec.get("sumw2"), 0.0)
        prec["files_processed"] += int(rec.get("files_processed") or 0)
        prec["files_attempted"] += int(rec.get("files_attempted") or 0)
        prec["events_written"] += int(rec.get("events_written") or 0)
        prec["split_dataset_ids"].append(dsid)
        for key, val in (rec.get("sumw_source_counts") or {}).items():
            bump(prec["sumw_source_counts"], key, int(val))
    for rec in physical.values():
        if rec.get("is_data"):
            rec["normalization_factor"] = 1.0
            rec["normalization_status"] = "data_unscaled"
        elif rec.get("is_signal"):
            rec["normalization_factor"] = None
            rec["normalization_status"] = "signal_uses_mass_point_sumw_not_physical_dataset_sumw"
        else:
            rec["normalization_factor"] = None
            rec["normalization_status"] = "partial_denominator_only_aggregate_all_flat_ntuple_sidecars_before_applying_xsec_lumi"
    payload["physical_datasets"] = physical
    payload["physical_dataset_split_counts"] = split_counts


def process_record(record: dict[str, Any], repo: Path, chunk_size: int, shift_name: str, skim_flag: str, keep_all: bool, max_chunks: int | None, require_object_corrections: bool, row_output: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process_group") or "unknown")
    is_data = bool(record.get("is_data") or process in DATA_PROCESSES)
    is_signal = bool(record.get("is_signal") or process == "SMS")
    file_path = str(record.get("file_path") or "")
    file_id = stable_id(file_path)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "process": process,
        "file_path": file_path,
        "file_id": file_id,
        "read_status": "not_started",
        "processing_status": "not_started",
        "events_read": 0,
        "events_written": 0,
        "sumw": 0.0,
        "sumw2": 0.0,
        "sumw_source": "data_unweighted" if is_data else "events_genWeight_sum",
        "shape_shift": shift_name,
    }
    bad: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    row_writer = RootRowWriter(Path(row_output)) if row_output is not None else None
    events_written = 0
    signal_fallback_accum: dict[str, float] = {}
    root = None
    access_info: dict[str, Any] = {}
    cwd = Path.cwd()
    try:
        os.chdir(repo)
        root, access_info = open_root_with_xrd_fallback(file_path, timeout=60)
        summary["file_access"] = access_info
        summary["effective_file_path"] = access_info.get("effective_file_path", file_path)
        keys = split_keys(root)
        summary["events_tree_exists"] = "Events" in keys
        summary["runs_tree_exists"] = "Runs" in keys
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        runs_info = read_runs_sumw(root)
        summary["runs_sumw"] = runs_info
        if not is_data and runs_info.get("generic_sumw") is not None and not is_signal:
            summary["sumw"] = float(runs_info["generic_sumw"])
            summary["sumw_source"] = f"Runs.{runs_info.get('generic_sumw_branch')}"
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
        summary["required_branch_validation"] = required
        if not all(required.values()):
            raise RuntimeError("required branch missing")
        genmodel_branches = sorted(
            b for b in branches if signal_genmodel_branch(str(b))
        )
        read_branches = [b for b in set(CORE_BRANCHES + FILTERS + SIGNAL_HLT + PHOTON_HLT + HT_REFERENCE_HLT + ELECTRON_HLT + ELECTRON_REFERENCE_HLT + MUON_HLT + genmodel_branches) if b in branches]
        summary["number_of_entries"] = int(tree.num_entries)
        summary["read_status"] = "opened"
        summary["processed_entry_ranges"] = []
        summary["chunk_summaries"] = []
        fastsim_trigger_bypass = bool(is_signal and process == "SMS" and str(record.get("simulation_type") or "") == "FastSim signal dataset")
        summary["fastsim_trigger_bypass"] = fastsim_trigger_bypass
        fallback_sumw = 0.0
        fallback_sumw2 = 0.0
        chunks_seen = 0
        for start in range(0, int(tree.num_entries), chunk_size):
            if max_chunks is not None and chunks_seen >= max_chunks:
                break
            stop = min(start + chunk_size, int(tree.num_entries))
            arrays = tree.arrays(read_branches, entry_start=start, entry_stop=stop, library="ak")
            rows, chunk_summary = extract_chunk(
                arrays,
                dataset,
                process,
                record.get("signal_point") or None,
                str(record.get("year", "")),
                file_path,
                start,
                stop,
                fastsim_trigger_bypass=fastsim_trigger_bypass,
                shift_name=shift_name,
                compute_weights=False,
                materialize_skim_flag=None if keep_all else skim_flag,
            )
            if require_object_corrections:
                missing = []
                for label, key in [("AK4 JEC", "ak4_jec_status"), ("AK8 FJEC", "ak8_jec_status")]:
                    status = chunk_summary.get(key) if isinstance(chunk_summary, dict) else None
                    if not isinstance(status, dict) or not status.get("applied"):
                        missing.append({"correction": label, "status": status})
                if missing:
                    raise RuntimeError(f"required object corrections were not applied before flat skim: {missing}")
            summary["processed_entry_ranges"].append({"entry_start": start, "entry_stop": stop})
            summary["chunk_summaries"].append({"entry_start": start, "entry_stop": stop, **chunk_summary})
            summary["events_read"] += int(chunk_summary.get("entries", len(rows)))
            if not is_data and summary["sumw_source"] == "events_genWeight_sum":
                if "gen_weight_sum" in chunk_summary:
                    fallback_sumw += float(chunk_summary["gen_weight_sum"])
                    fallback_sumw2 += float(chunk_summary["gen_weight_sum2"])
                else:
                    gen_weights = np.asarray([finite(row.get("gen_weight"), 0.0) for row in rows], dtype=float)
                    fallback_sumw += float(np.sum(gen_weights))
                    fallback_sumw2 += float(np.sum(gen_weights * gen_weights))
            selected = [decorate_row(row, record, file_id) for row in rows if row_passes(row, skim_flag, keep_all)]
            if row_writer is not None:
                row_writer.extend(selected)
                add_float_map(signal_fallback_accum, signal_fallback_from_rows(selected))
                events_written += len(selected)
            else:
                kept.extend(selected)
            chunks_seen += 1
        summary["events_written"] = events_written if row_writer is not None else len(kept)
        if is_data:
            summary["sumw"] = float(summary["events_read"])
            summary["sumw2"] = float(summary["events_read"])
            summary["sumw_source"] = "data_unweighted"
        elif summary["sumw_source"] == "events_genWeight_sum":
            summary["sumw"] = float(fallback_sumw)
            summary["sumw2"] = float(fallback_sumw2)
        summary["processing_status"] = "processed_full_file" if max_chunks is None else "processed_limited_chunks"
        summary["read_status"] = "success"
        if row_writer is not None:
            summary["signal_event_genweight_sum_by_genmodel"] = signal_fallback_accum
            row_writer.close(commit=True)
            row_writer = None
            return [], summary, bad
        return kept, summary, bad
    except Exception as exc:
        if isinstance(exc, RootOpenFailure):
            summary["file_access"] = exc.access_info
            access_info = exc.access_info
        summary["read_status"] = "failed"
        summary["processing_status"] = "excluded"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        bad.append({
            "dataset": dataset,
            "process": process,
            "file_path": file_path,
            "failure_stage": "flat_ntuple_open_or_read",
            "exception_type": type(exc).__name__,
            "concise_error": str(exc)[:400],
            "first_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if row_writer is not None:
            row_writer.close(commit=False)
            row_writer = None
        return kept, summary, bad
    finally:
        try:
            if root is not None:
                root.close()
        except Exception:
            pass
        if row_writer is not None:
            row_writer.close(commit=False)
        cleanup_xrd_cache(access_info)
        os.chdir(cwd)


def write_root(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    types = branch_types()
    arrays = rows_to_arrays(rows) if rows else {}
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    try:
        with uproot.recreate(tmp) as root_file:
            root_file.mktree(TREE_NAME, types)
            if rows:
                root_file[TREE_NAME].extend(arrays)
        os.replace(tmp, output)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def fragment_work_dir(output: Path) -> Path:
    base = Path(os.environ.get("AUTONOMOUS_ALLHAD_FRAGMENT_DIR", "/tmp/taiwoo/autonomous_allhad_flat_fragments"))
    return base / f"{output.stem}.{os.getpid()}.{int(time.time())}"


def merge_root_fragments(output: Path, fragments: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not fragments:
        write_root(output, [])
        return
    hadd = shutil.which("hadd")
    if not hadd:
        raise RuntimeError("hadd executable not found; cannot merge flat ntuple fragments without parent rows")
    tmp = output.with_name(f"{output.name}.tmp.hadd.{os.getpid()}")
    try:
        cmd = [hadd, "-f", str(tmp)] + [str(path) for path in fragments]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"hadd failed with rc={proc.returncode}: {proc.stdout[-4000:]}")
        os.replace(tmp, output)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def process_record_fragment(record_index: int, record: dict[str, Any], fragment_dir: str, repo: Path, chunk_size: int, shift_name: str, skim_flag: str, keep_all: bool, max_chunks: int | None, require_object_corrections: bool) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    fragment_path = Path(fragment_dir) / f"record_{record_index + 1:05d}.root"
    rows, summary, bad = process_record(record, repo, chunk_size, shift_name, skim_flag, keep_all, max_chunks, require_object_corrections, row_output=fragment_path)
    if summary.get("read_status") == "success":
        summary["fragment_file"] = str(fragment_path)
        return str(fragment_path), summary, bad
    return None, summary, bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a preselection-level flat ROOT ntuple from production shard records.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("AUTONOMOUS_ALLHAD_FLAT_CHUNK", "50000")))
    parser.add_argument("--shift", default=os.environ.get("AUTONOMOUS_ALLHAD_PRODUCTION_SHIFT", "nominal"))
    parser.add_argument("--skim-flag", default="feature_flat_preselection")
    parser.add_argument("--keep-all-rows", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-chunks-per-file", type=int, default=None)
    parser.add_argument("--record-workers", type=int, default=int(os.environ.get("AUTONOMOUS_ALLHAD_RECORD_WORKERS", "1")), help="Process records/files within one shard concurrently. Parallel mode writes per-record ROOT fragments and merges them with hadd.")
    parser.add_argument(
        "--allow-missing-object-corrections",
        action="store_true",
        help="Allow required nominal object-correction fallbacks. Intended only for synthetic/debug inputs.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    shard_path = Path(args.shard)
    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(output.suffix + ".json")
    shift_name = validate_shift_name(args.shift)
    shard = json.loads(shard_path.read_text())
    records = list(shard.get("records") or [])
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": None,
        "repo": str(repo),
        "shard": str(shard_path),
        "shard_id": shard.get("shard_id"),
        "record_digest": shard.get("record_digest"),
        "records_in_shard": len(records),
        "root_file": str(output),
        "tree": TREE_NAME,
        "shape_shift": shift_name,
        "skim_flag": args.skim_flag,
        "keep_all_rows": bool(args.keep_all_rows),
        "require_object_corrections": not bool(args.allow_missing_object_corrections),
        "chunk_size": args.chunk_size,
        "record_workers": max(1, int(args.record_workers)),
        "files_attempted": 0,
        "files_processed": 0,
        "events_read": 0,
        "events_written": 0,
        "datasets": {},
        "files": [],
        "bad_files": [],
        "raw_weight_branch": "gen_weight",
        "deferred_weight_variations": list(DEFERRED_WEIGHT_VARIATIONS),
        "normalization_policy": {
            "root_event_weight_status": "raw_gen_weight_only_no_luminosity_xsec_or_scale_factor_normalization",
            "x_axis_corrections_in_skim": "Nominal 2024 electron, muon, photon, tau, AK4, AK8, and propagated PuppiMET kinematic corrections are required before selection by default. Non-nominal object/JES/JER/MET variations are produced only when --shift is not nominal.",
            "y_axis_scale_factor_status": "pileup, btag, lepton/photon ID/HLT, and top-pT scale factors are intentionally deferred to post-skim processing",
            "background_formula_after_campaign_sidecar_merge": "normalized_weight = gen_weight * post_skim_sf_weight * xsec_pb * lumi_pb / physical_dataset_sumw",
            "signal_formula_after_campaign_sidecar_merge": "select mStop/mLSP events, then normalized_weight = gen_weight * post_skim_sf_weight * xsec_pb(mStop) * lumi_pb / sumw_mass_point, with sumw_mass_point from Runs.genEventSumw_T2tt_<mStop>_<mLSP>",
            "data_formula": "data event weight = 1",
            "luminosity_fb": LUMI_FB,
            "luminosity_pb": LUMI_PB,
            "normalization_grouping_policy": "background MC metadata records split as <dataset>____N_ share one physical-dataset denominator; signal SMS records use mass-point denominators instead.",
            "factor_scope_warning": "This sidecar contains shard-local denominator pieces only. Aggregate all sidecars in a campaign before computing final MC factors.",
        },
    }

    all_rows: list[dict[str, Any]] = []
    fragment_paths: list[Path] = []
    fragment_dir: Any = None
    start_time = time.time()

    def register_attempt(record_index: int, record: dict[str, Any]) -> None:
        payload["files_attempted"] += 1
        ds_meta = payload["datasets"].setdefault(str(stable_id(str(record.get("dataset") or "unknown"))), empty_dataset_record(record))
        ds_meta["files_attempted"] += 1
        print(json.dumps({
            "stage": "record_start",
            "shard_id": payload.get("shard_id"),
            "record": record_index + 1,
            "records_in_shard": len(records),
            "dataset": record.get("dataset"),
            "process": record.get("process_group") or record.get("process"),
            "file_index": record.get("file_index"),
        }, sort_keys=True), flush=True)

    def failed_worker_result(record: dict[str, Any], exc: BaseException) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        dataset = str(record.get("dataset") or "unknown")
        process = str(record.get("process_group") or record.get("process") or "unknown")
        file_path = str(record.get("file_path") or "")
        summary = {
            "dataset": dataset,
            "process": process,
            "file_path": file_path,
            "file_id": stable_id(file_path),
            "read_status": "failed",
            "processing_status": "record_worker_exception",
            "events_read": 0,
            "events_written": 0,
            "sumw": 0.0,
            "sumw2": 0.0,
            "sumw_source": "record_worker_exception",
            "shape_shift": shift_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
        bad = [{
            "dataset": dataset,
            "process": process,
            "file_path": file_path,
            "failure_stage": "flat_ntuple_record_worker",
            "exception_type": type(exc).__name__,
            "concise_error": str(exc)[:400],
            "first_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }]
        return [], summary, bad

    def merge_result(record_index: int, record: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any], bad: list[dict[str, Any]], fragment_path: Any = None) -> None:
        print(json.dumps({
            "stage": "record_done",
            "shard_id": payload.get("shard_id"),
            "record": record_index + 1,
            "records_in_shard": len(records),
            "dataset": summary.get("dataset"),
            "process": summary.get("process"),
            "read_status": summary.get("read_status"),
            "processing_status": summary.get("processing_status"),
            "events_read": summary.get("events_read"),
            "events_written": summary.get("events_written"),
            "bad_files": len(bad),
        }, sort_keys=True), flush=True)
        ds_meta = payload["datasets"].setdefault(str(stable_id(str(record.get("dataset") or "unknown"))), empty_dataset_record(record))
        payload["files"].append(summary)
        payload["bad_files"].extend(bad)
        payload["events_read"] += int(summary.get("events_read") or 0)
        payload["events_written"] += int(summary.get("events_written") or 0)
        ds_meta["events_read"] += int(summary.get("events_read") or 0)
        ds_meta["events_written"] += int(summary.get("events_written") or 0)
        if summary.get("read_status") == "success":
            payload["files_processed"] += 1
            ds_meta["files_processed"] += 1
            ds_meta["sumw"] += finite(summary.get("sumw"), 0.0)
            ds_meta["sumw2"] += finite(summary.get("sumw2"), 0.0)
            bump(ds_meta["sumw_source_counts"], str(summary.get("sumw_source") or "unknown"))
            runs_info = summary.get("runs_sumw") if isinstance(summary.get("runs_sumw"), dict) else {}
            add_float_map(ds_meta["signal_sumw_by_genmodel"], runs_info.get("signal_sumw_by_genmodel") or {})
            if runs_info.get("signal_sumw_by_genmodel"):
                bump(
                    ds_meta["signal_runs_sumw_source_counts"],
                    "Runs.genEventSumw_<topology>_<mStop>_<mLSP>",
                )
            fallback_map = summary.get("signal_event_genweight_sum_by_genmodel")
            if isinstance(fallback_map, dict):
                add_float_map(ds_meta["signal_event_genweight_sum_by_genmodel"], fallback_map)
            else:
                update_signal_fallback(ds_meta, rows)
        if fragment_path:
            fragment_paths.append(Path(fragment_path))
        else:
            all_rows.extend(rows)

    worker_args = (repo, args.chunk_size, shift_name, args.skim_flag, args.keep_all_rows, args.max_chunks_per_file, not bool(args.allow_missing_object_corrections))
    record_workers = max(1, int(args.record_workers))
    use_fragment_mode = record_workers > 1 and len(records) > 1
    payload["record_output_mode"] = "fragments_hadd" if use_fragment_mode else "parent_rows"
    if use_fragment_mode:
        fragment_dir = fragment_work_dir(output)
        fragment_dir.mkdir(parents=True, exist_ok=True)
    if record_workers <= 1 or len(records) <= 1:
        for record_index, record in enumerate(records):
            register_attempt(record_index, record)
            rows, summary, bad = process_record(record, *worker_args)
            merge_result(record_index, record, rows, summary, bad)
    else:
        max_workers = min(record_workers, len(records))
        futures: list[tuple[int, dict[str, Any], Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            for record_index, record in enumerate(records):
                register_attempt(record_index, record)
                if use_fragment_mode:
                    futures.append((record_index, record, pool.submit(process_record_fragment, record_index, record, str(fragment_dir), *worker_args)))
                else:
                    futures.append((record_index, record, pool.submit(process_record, record, *worker_args)))
            for record_index, record, future in futures:
                fragment_path = None
                try:
                    if use_fragment_mode:
                        fragment_path, summary, bad = future.result()
                        rows = []
                    else:
                        rows, summary, bad = future.result()
                except BaseException as exc:
                    rows, summary, bad = failed_worker_result(record, exc)
                merge_result(record_index, record, rows, summary, bad, fragment_path)

    payload["branch_schema"] = {
        "int64": INT64_FIELDS,
        "int32": INT32_FIELDS,
        "float64": FLOAT_FIELDS,
        "bool": BOOL_FIELDS,
        "vector_float64": VECTOR_FLOAT_FIELDS,
        "vector_int32": VECTOR_INT_FIELDS,
    }
    update_physical_norm(payload)
    payload["status"] = "complete" if payload["files_processed"] == payload["files_attempted"] else ("complete_with_bad_files" if payload["files_processed"] else "failed")
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["wall_time_s"] = round(time.time() - start_time, 3)

    try:
        if use_fragment_mode:
            merge_root_fragments(output, fragment_paths)
        else:
            write_root(output, all_rows)
    finally:
        if fragment_dir is not None:
            shutil.rmtree(fragment_dir, ignore_errors=True)
    write_json(metadata_output, payload)
    print(json.dumps({
        "status": payload["status"],
        "root_file": str(output),
        "metadata": str(metadata_output),
        "files_processed": payload["files_processed"],
        "files_attempted": payload["files_attempted"],
        "events_read": payload["events_read"],
        "events_written": payload["events_written"],
        "raw_weight_branch": "gen_weight",
        "deferred_weight_variations": len(DEFERRED_WEIGHT_VARIATIONS),
    }, sort_keys=True))
    return 0 if payload["files_processed"] > 0 or payload["files_attempted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
