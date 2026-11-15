"""Native Events schema, source identities and NanoAOD column decoration."""
from __future__ import annotations
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any
import awkward as ak
import numpy as np
import uproot
from . import nanoaod_features as baseline
from .nanoaod_features import finite_float
from .paths import ROOT, analysis_era, EARLY_ERAS
from .signal_models import signal_genmodel_branch, signal_runs_sumw_branch
from .io import write_json


def configure_era(era):
    global VECTOR_FLOAT_FIELDS, VECTOR_INT_FIELDS, FLOAT_FIELDS, INT32_FIELDS
    global SCHEMA_VERSION
    era = analysis_era(era)
    baseline.configure_era(era)
    FLOAT_FIELDS = list(dict.fromkeys(FLOAT_FIELDS + EXTRA_FLOAT_FIELDS + list(JME_SCALAR_INPUTS)))
    VECTOR_FLOAT_FIELDS = list(dict.fromkeys(VECTOR_FLOAT_FIELDS + EXTRA_VECTOR_FLOAT_FIELDS
                                            + list(JME_VECTOR_INPUTS) + list(MIXED_INPUTS[era in EARLY_ERAS])
                                            + ['genjet_eta_all', 'genjet_phi_all', 'genjet_mass_all']))
    VECTOR_INT_FIELDS = list(dict.fromkeys(VECTOR_INT_FIELDS + EXTRA_VECTOR_INT_FIELDS
                                          + ['jet_nanoaod_id', 'fatjet_nanoaod_id']))
    INT32_FIELDS = list(dict.fromkeys(INT32_FIELDS + ['pv_npvs_good']))
    if era in EARLY_ERAS:
        VECTOR_FLOAT_FIELDS = [n.replace('btag_upart', 'btag_pnet') for n in VECTOR_FLOAT_FIELDS]
        VECTOR_FLOAT_FIELDS = [n for n in VECTOR_FLOAT_FIELDS if n != 'fatjet_btagdeepb_all']
    configure_float_storage(float32=True, keep_float64={'gen_weight'})
    SCHEMA_VERSION = f'flat_ntuple_shard_v8_float32_fullselection_{era}_trota'


MIXED_INPUTS = {
    False: {'fatjet_globalpart3_xbb_all': 'FatJet_globalParT3_Xbb',
            'fatjet_globalpart3_qcd_all': 'FatJet_globalParT3_QCD'},
    True: {'jet_btag_deepflav_all': 'Jet_btagDeepFlavB',
           'fatjet_pnetwmass_qcd_all': 'FatJet_particleNetWithMass_QCD',
           'fatjet_pnet_qcd_all': 'FatJet_particleNet_QCD',
           'fatjet_pnet_xbbvsqcd_all': 'FatJet_particleNet_XbbVsQCD',
           'fatjet_pnet_xqqvsqcd_all': 'FatJet_particleNet_XqqVsQCD'}}


def decorate(rows, raw, corrected, start, era, record, file_id):
    early = analysis_era(era) in EARLY_ERAS
    _decorate_rows(rows, raw, corrected, start,
                   btag_branch='Jet_btagPNetB' if early else 'Jet_btagUParTAK4B',
                   btag_field='jet_btag_pnet_all' if early else 'jet_btag_upart_all')
    decorate_jme_inputs(rows, raw, start)
    ids = {prefix: baseline.jet_id(raw, prefix) for prefix in ('Jet', 'FatJet')}
    for row in rows:
        index = int(row['entry']) - start
        for target, source in MIXED_INPUTS[early].items():
            row[target] = ak.to_list(raw[source][index])
        row['jet_nanoaod_id'] = ak.to_list(ids['Jet'][index])
        row['fatjet_nanoaod_id'] = ak.to_list(ids['FatJet'][index])
        row['pv_npvs_good'] = int(raw['PV_npvsGood'][index])
        for axis in ('eta', 'phi', 'mass'):
            row['genjet_' + axis + '_all'] = ak.to_list(raw['GenJet_' + axis][index])
        row['year'] = int(str(era)[:4])
    return [decorate_row(row, record, file_id) for row in rows]

SCHEMA_VERSION = "flat_ntuple_shard_v5_fullselection_2024"

TREE_NAME = "Events"

DATA_PROCESSES = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}

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

FLOAT64_STORAGE_FIELDS = set(FLOAT_FIELDS)

VECTOR_FLOAT64_STORAGE_FIELDS = set(VECTOR_FLOAT_FIELDS)

def configure_float_storage(
    *,
    float32: bool,
    keep_float64: set[str] | None = None,
    keep_vector_float64: set[str] | None = None,
) -> None:
    global FLOAT64_STORAGE_FIELDS, VECTOR_FLOAT64_STORAGE_FIELDS
    if float32:
        scalar_fields = set(keep_float64 or set())
        vector_fields = set(keep_vector_float64 or set())
    else:
        scalar_fields = set(FLOAT_FIELDS)
        vector_fields = set(VECTOR_FLOAT_FIELDS)

    unknown_scalars = scalar_fields - set(FLOAT_FIELDS)
    unknown_vectors = vector_fields - set(VECTOR_FLOAT_FIELDS)
    if unknown_scalars or unknown_vectors:
        raise RuntimeError(
            "float storage policy references unknown branches: "
            f"scalars={sorted(unknown_scalars)}, vectors={sorted(unknown_vectors)}"
        )
    FLOAT64_STORAGE_FIELDS = scalar_fields
    VECTOR_FLOAT64_STORAGE_FIELDS = vector_fields

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

def branch_types() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in INT64_FIELDS:
        out[field] = np.int64
    for field in INT32_FIELDS:
        out[field] = np.int32
    for field in FLOAT_FIELDS:
        out[field] = np.float64 if field in FLOAT64_STORAGE_FIELDS else np.float32
    for field in BOOL_FIELDS:
        out[field] = np.bool_
    for field in VECTOR_FLOAT_FIELDS:
        out[field] = (
            "var * float64"
            if field in VECTOR_FLOAT64_STORAGE_FIELDS
            else "var * float32"
        )
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
        dtype = np.float64 if field in FLOAT64_STORAGE_FIELDS else np.float32
        arrays[field] = np.asarray(
            [finite(row.get(field), 0.0) for row in rows], dtype=dtype
        )
    for field in BOOL_FIELDS:
        arrays[field] = np.asarray([bool(row.get(field, False)) for row in rows], dtype=np.bool_)
    for field in VECTOR_FLOAT_FIELDS:
        dtype = np.float64 if field in VECTOR_FLOAT64_STORAGE_FIELDS else np.float32
        arrays[field] = ak.values_astype(
            ak.Array(
                [
                    [finite(x, 0.0) for x in (row.get(field) or [])]
                    for row in rows
                ]
            ),
            dtype,
        )
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

JME_SCALAR_INPUTS = {
    "puppi_met_unclustered_up": "PuppiMET_ptUnclusteredUp",
    "puppi_met_phi_unclustered_up": "PuppiMET_phiUnclusteredUp",
    "puppi_met_unclustered_down": "PuppiMET_ptUnclusteredDown",
    "puppi_met_phi_unclustered_down": "PuppiMET_phiUnclusteredDown",
}

JME_VECTOR_INPUTS = {
    "jet_ch_em_ef": "Jet_chEmEF",
    "jet_ne_em_ef": "Jet_neEmEF",
}

def decorate_jme_inputs(rows, raw, entry_start):
    missing = sorted((set(JME_SCALAR_INPUTS.values()) | set(JME_VECTOR_INPUTS.values())) - set(raw.fields))
    if missing:
        raise ValueError("Missing Jet/MET uncertainty inputs: " + str(missing))
    for row in rows:
        index = int(row["entry"]) - entry_start
        for target, source in JME_SCALAR_INPUTS.items():
            row[target] = float(raw[source][index])
        for target, source in JME_VECTOR_INPUTS.items():
            row[target] = ak.to_list(raw[source][index])

EXTRA_FLOAT_FIELDS = [
    "rho",
    "lhe_vpt",
    "puppi_met_nanoaod",
    "puppi_met_nanoaod_phi",
    "puppi_met_corrected",
    "puppi_met_corrected_phi",
]

EXTRA_VECTOR_FLOAT_FIELDS = [
    "jet_nanoaod_pt",
    "jet_raw_pt",
    "jet_corrected_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_nanoaod_mass",
    "jet_corrected_mass",
    "jet_raw_factor",
    "jet_area",
    "jet_muon_subtr_factor",
    "jet_btag_upart_all",
    "fatjet_nanoaod_pt",
    "fatjet_raw_pt",
    "fatjet_corrected_pt",
    "fatjet_eta_all",
    "fatjet_phi_all",
    "fatjet_nanoaod_mass",
    "fatjet_corrected_mass",
    "fatjet_raw_factor",
    "fatjet_area",
    "fatjet_msoftdrop_all",
    "fatjet_top_score_all",
    "fatjet_w_score_all",
    "subjet_eta_all",
    "subjet_phi_all",
    "electron_nanoaod_pt",
    "electron_corrected_pt",
    "electron_nanoaod_mass",
    "electron_corrected_mass",
    "electron_pt_all",
    "electron_eta_all",
    "electron_eta_sc_all",
    "electron_phi_all",
    "electron_mass_all",
    "electron_mini_iso_all",
    "electron_r9_all",
    "muon_nanoaod_pt",
    "muon_corrected_pt",
    "muon_nanoaod_mass",
    "muon_corrected_mass",
    "muon_pt_all",
    "muon_eta_all",
    "muon_phi_all",
    "muon_mass_all",
    "muon_mini_iso_all",
    "photon_nanoaod_pt",
    "photon_corrected_pt",
    "photon_pt_all",
    "photon_eta_all",
    "photon_phi_all",
    "photon_r9_all",
    "tau_nanoaod_pt",
    "tau_corrected_pt",
    "tau_nanoaod_mass",
    "tau_corrected_mass",
    "tau_pt_all",
    "tau_eta_all",
    "tau_phi_all",
    "tau_mass_all",
    "tau_dz_all",
    "genjet_pt_all",
    "genjet_eta_all",
    "genjet_phi_all",
    "genjet_mass_all",
    "genjetak8_pt_all",
    "genjetak8_eta_all",
    "genjetak8_phi_all",
    "genjetak8_mass_all",
]

EXTRA_VECTOR_INT_FIELDS = [
    "jet_id_all",
    "jet_source_index_all",
    "fatjet_id_all",
    "fatjet_source_index_all",
    "fatjet_subjet_index1_all",
    "fatjet_subjet_index2_all",
    "fatjet_boosted_top_pass_all",
    "fatjet_boosted_w_pass_all",
    "jet_hadron_flavour_all",
    "jet_genjet_index_all",
    "fatjet_genjetak8_index_all",
    "electron_charge_all",
    "electron_cutbased_all",
    "electron_seed_gain_all",
    "muon_charge_all",
    "muon_loose_id_all",
    "muon_medium_id_all",
    "muon_tracker_layers_all",
    "photon_cutbased_all",
    "photon_seed_gain_all",
    "photon_electron_veto_all",
    "photon_pixel_seed_all",
    "tau_decay_mode_all",
    "tau_deeptau_vse_all",
    "tau_deeptau_vsmu_all",
    "tau_deeptau_vsjet_all",
    "tau_genpart_flavour_all",
]

def _as_list(values: Any, index: int) -> list[Any]:
    return ak.to_list(values[index])

def _boosted_overlap_sources(
    raw: Any,
    corrected: Any,
    fatjet_id: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact per-object inputs used for boosted/resolved cleaning."""
    fatjet_pt = corrected["FatJet_pt"]
    fatjet_eta = raw["FatJet_eta"]
    fatjet_msoftdrop = raw["FatJet_msoftdrop"]
    top_score = raw[baseline.BOOSTED_TOP_SCORE_BRANCH]
    w_score = raw[baseline.BOOSTED_W_SCORE_BRANCH]

    boosted_top = (
        fatjet_id
        & (fatjet_pt > baseline.BOOSTED_TOP_PT_MIN)
        & (abs(fatjet_eta) < baseline.BOOSTED_ETA_MAX)
        & (fatjet_msoftdrop > baseline.BOOSTED_TOP_MSD_MIN)
        & (top_score > baseline.BOOSTED_TOP_SCORE_WP)
        & (top_score <= baseline.BOOSTED_SCORE_MAX)
    )
    boosted_w = (
        fatjet_id
        & (fatjet_pt > baseline.BOOSTED_W_PT_MIN)
        & (abs(fatjet_eta) < baseline.BOOSTED_ETA_MAX)
        & (fatjet_msoftdrop > baseline.BOOSTED_W_MSD_MIN)
        & (fatjet_msoftdrop < baseline.BOOSTED_W_MSD_MAX)
        & (w_score > baseline.BOOSTED_W_SCORE_WP)
        & (w_score <= baseline.BOOSTED_SCORE_MAX)
    )

    vector_sources = {
        "fatjet_top_score_all": top_score,
        "fatjet_w_score_all": w_score,
        "subjet_eta_all": raw["SubJet_eta"],
        "subjet_phi_all": raw["SubJet_phi"],
    }
    integer_sources = {
        "jet_source_index_all": ak.values_astype(
            ak.local_index(raw["Jet_pt"], axis=1), np.int32,
        ),
        "fatjet_source_index_all": ak.values_astype(
            ak.local_index(raw["FatJet_pt"], axis=1), np.int32,
        ),
        "fatjet_subjet_index1_all": ak.values_astype(
            raw["FatJet_subJetIdx1"], np.int32,
        ),
        "fatjet_subjet_index2_all": ak.values_astype(
            raw["FatJet_subJetIdx2"], np.int32,
        ),
        "fatjet_boosted_top_pass_all": ak.values_astype(boosted_top, np.int32),
        "fatjet_boosted_w_pass_all": ak.values_astype(boosted_w, np.int32),
    }
    return vector_sources, integer_sources

def _decorate_rows(
    rows: list[dict[str, Any]], raw: Any, corrected: Any, entry_start: int,
    *, btag_branch: str = "Jet_btagUParTAK4B",
    btag_field: str = "jet_btag_upart_all",
) -> None:
    jet_id, _ = baseline.ak4_tight_lepton_veto_mask(
        corrected, corrected["Jet_pt"], corrected["Jet_eta"], Path.cwd(),
    )
    fatjet_id, _ = baseline.ak8_tight_lepton_veto_mask(
        corrected, corrected["FatJet_pt"], corrected["FatJet_eta"], Path.cwd(),
    )
    boosted_vector_sources, boosted_integer_sources = _boosted_overlap_sources(
        raw, corrected, fatjet_id,
    )
    vector_sources = {
        "jet_nanoaod_pt": raw["Jet_pt"],
        "jet_raw_pt": raw["Jet_pt"] * (1.0 - raw["Jet_rawFactor"]),
        "jet_corrected_pt": corrected["Jet_pt"],
        "jet_eta_all": raw["Jet_eta"],
        "jet_phi_all": raw["Jet_phi"],
        "jet_nanoaod_mass": raw["Jet_mass"],
        "jet_corrected_mass": corrected["Jet_mass"],
        "jet_raw_factor": raw["Jet_rawFactor"],
        "jet_area": raw["Jet_area"],
        "jet_muon_subtr_factor": raw["Jet_muonSubtrFactor"],
        btag_field: raw[btag_branch],
        "fatjet_nanoaod_pt": raw["FatJet_pt"],
        "fatjet_raw_pt": raw["FatJet_pt"] * (1.0 - raw["FatJet_rawFactor"]),
        "fatjet_corrected_pt": corrected["FatJet_pt"],
        "fatjet_eta_all": raw["FatJet_eta"],
        "fatjet_phi_all": raw["FatJet_phi"],
        "fatjet_nanoaod_mass": raw["FatJet_mass"],
        "fatjet_corrected_mass": corrected["FatJet_mass"],
        "fatjet_raw_factor": raw["FatJet_rawFactor"],
        "fatjet_area": raw["FatJet_area"],
        "fatjet_msoftdrop_all": raw["FatJet_msoftdrop"],
        "electron_nanoaod_pt": raw["Electron_pt"],
        "electron_corrected_pt": corrected["Electron_pt"],
        "electron_nanoaod_mass": raw["Electron_mass"],
        "electron_corrected_mass": corrected["Electron_mass"],
        "electron_pt_all": corrected["Electron_pt"],
        "electron_eta_all": raw["Electron_eta"],
        "electron_eta_sc_all": raw["Electron_eta"] + raw["Electron_deltaEtaSC"],
        "electron_phi_all": raw["Electron_phi"],
        "electron_mass_all": corrected["Electron_mass"],
        "electron_mini_iso_all": raw["Electron_miniPFRelIso_all"],
        "electron_r9_all": raw["Electron_r9"],
        "muon_nanoaod_pt": raw["Muon_pt"],
        "muon_corrected_pt": corrected["Muon_pt"],
        "muon_nanoaod_mass": raw["Muon_mass"],
        "muon_corrected_mass": corrected["Muon_mass"],
        "muon_pt_all": corrected["Muon_pt"],
        "muon_eta_all": raw["Muon_eta"],
        "muon_phi_all": raw["Muon_phi"],
        "muon_mass_all": corrected["Muon_mass"],
        "muon_mini_iso_all": raw["Muon_miniPFRelIso_all"],
        "photon_nanoaod_pt": raw["Photon_pt"],
        "photon_corrected_pt": corrected["Photon_pt"],
        "photon_pt_all": corrected["Photon_pt"],
        "photon_eta_all": raw["Photon_eta"],
        "photon_phi_all": raw["Photon_phi"],
        "photon_r9_all": raw["Photon_r9"],
        "tau_nanoaod_pt": raw["Tau_pt"],
        "tau_corrected_pt": corrected["Tau_pt"],
        "tau_nanoaod_mass": raw["Tau_mass"],
        "tau_corrected_mass": corrected["Tau_mass"],
        "tau_pt_all": corrected["Tau_pt"],
        "tau_eta_all": raw["Tau_eta"],
        "tau_phi_all": raw["Tau_phi"],
        "tau_mass_all": corrected["Tau_mass"],
        "tau_dz_all": raw["Tau_dz"],
    }
    vector_sources.update(boosted_vector_sources)
    integer_sources = {
        "jet_id_all": ak.values_astype(jet_id, np.int32),
        "fatjet_id_all": ak.values_astype(fatjet_id, np.int32),
        "jet_hadron_flavour_all": raw["Jet_hadronFlavour"] if "Jet_hadronFlavour" in raw.fields else ak.zeros_like(raw["Jet_pt"], dtype=np.int32),
        "jet_genjet_index_all": raw["Jet_genJetIdx"] if "Jet_genJetIdx" in raw.fields else ak.ones_like(raw["Jet_pt"], dtype=np.int32) * -1,
        "fatjet_genjetak8_index_all": raw["FatJet_genJetAK8Idx"] if "FatJet_genJetAK8Idx" in raw.fields else ak.ones_like(raw["FatJet_pt"], dtype=np.int32) * -1,
        "electron_charge_all": raw["Electron_charge"],
        "electron_cutbased_all": raw["Electron_cutBased"],
        "electron_seed_gain_all": raw["Electron_seedGain"],
        "muon_charge_all": raw["Muon_charge"],
        "muon_loose_id_all": raw["Muon_looseId"],
        "muon_medium_id_all": raw["Muon_mediumId"],
        "muon_tracker_layers_all": raw["Muon_nTrackerLayers"],
        "photon_cutbased_all": raw["Photon_cutBased"],
        "photon_seed_gain_all": raw["Photon_seedGain"],
        "photon_electron_veto_all": raw["Photon_electronVeto"],
        "photon_pixel_seed_all": raw["Photon_pixelSeed"],
        "tau_decay_mode_all": raw["Tau_decayMode"],
        "tau_deeptau_vse_all": raw["Tau_idDeepTau2018v2p5VSe"],
        "tau_deeptau_vsmu_all": raw["Tau_idDeepTau2018v2p5VSmu"],
        "tau_deeptau_vsjet_all": raw["Tau_idDeepTau2018v2p5VSjet"],
        "tau_genpart_flavour_all": raw["Tau_genPartFlav"] if "Tau_genPartFlav" in raw.fields else ak.zeros_like(raw["Tau_pt"], dtype=np.int32),
    }
    integer_sources.update(boosted_integer_sources)
    if "GenJet_pt" in raw.fields:
        for target, source in (
            ("genjet_pt_all", "GenJet_pt"),
            ("genjet_eta_all", "GenJet_eta"),
            ("genjet_phi_all", "GenJet_phi"),
            ("genjet_mass_all", "GenJet_mass"),
            ("genjetak8_pt_all", "GenJetAK8_pt"),
            ("genjetak8_eta_all", "GenJetAK8_eta"),
            ("genjetak8_phi_all", "GenJetAK8_phi"),
            ("genjetak8_mass_all", "GenJetAK8_mass"),
        ):
            vector_sources[target] = raw[source]
    for row in rows:
        index = int(row["entry"]) - entry_start
        row["rho"] = float(raw["Rho_fixedGridRhoFastjetAll"][index])
        row["lhe_vpt"] = (
            float(raw["LHE_Vpt"][index])
            if "LHE_Vpt" in raw.fields
            else -99.0
        )
        row["puppi_met_nanoaod"] = float(raw["PuppiMET_pt"][index])
        row["puppi_met_nanoaod_phi"] = float(raw["PuppiMET_phi"][index])
        row["puppi_met_corrected"] = float(corrected["PuppiMET_pt"][index])
        row["puppi_met_corrected_phi"] = float(corrected["PuppiMET_phi"][index])
        for name, values in vector_sources.items():
            row[name] = _as_list(values, index)
        for name, values in integer_sources.items():
            row[name] = _as_list(values, index)
        if "GenJet_pt" not in raw.fields:
            for name in (
                "genjet_pt_all", "genjet_eta_all", "genjet_phi_all", "genjet_mass_all",
                "genjetak8_pt_all", "genjetak8_eta_all", "genjetak8_phi_all", "genjetak8_mass_all",
            ):
                row[name] = []


@__import__("numba").njit(cache=False)
def decay_flavor(jet_counts, jet_eta, jet_phi, gen_counts, pdg, mother, eta, phi):
    """Classify AK8 containment of direct hadronic W/top decay partons."""
    result = np.zeros(len(jet_eta), dtype=np.int64)
    jo, go = 0, 0
    for event in range(len(jet_counts)):
        ng = gen_counts[event]
        for iw in range(ng):
            if abs(pdg[go + iw]) != 24:
                continue
            quarks = []
            for iq in range(ng):
                if mother[go + iq] == iw and 1 <= abs(pdg[go + iq]) <= 5:
                    quarks.append(iq)
            if len(quarks) != 2:
                continue
            parent = mother[go + iw]
            for _ in range(ng):
                if parent < 0 or abs(pdg[go + parent]) != 24:
                    break
                parent = mother[go + parent]
            top = parent if parent >= 0 and abs(pdg[go + parent]) == 6 else -1
            # Identify the same top across generator copy chains.
            if top >= 0:
                for _ in range(ng):
                    p = mother[go + top]
                    if p < 0 or pdg[go + p] != pdg[go + top]:
                        break
                    top = p
            bs = []
            if top >= 0:
                for ib in range(ng):
                    if abs(pdg[go + ib]) != 5:
                        continue
                    p = mother[go + ib]
                    if p < 0 or abs(pdg[go + p]) != 6:
                        continue
                    for _ in range(ng):
                        pp = mother[go + p]
                        if pp < 0 or pdg[go + pp] != pdg[go + p]:
                            break
                        p = pp
                    if p == top:
                        bs.append(ib)
            for ij in range(jet_counts[event]):
                j = jo + ij
                nq, nb = 0, 0
                for iq in quarks:
                    dp = (jet_phi[j] - phi[go + iq] + np.pi) % (2 * np.pi) - np.pi
                    nq += (jet_eta[j] - eta[go + iq]) ** 2 + dp ** 2 < 0.64
                for ib in bs:
                    dp = (jet_phi[j] - phi[go + ib] + np.pi) % (2 * np.pi) - np.pi
                    nb += (jet_eta[j] - eta[go + ib]) ** 2 + dp ** 2 < 0.64
                # Priority: merged top > merged W > partial top > partial W > other.
                category = 4 if nq == 2 and nb else 3 if nq == 2 else 2 if top >= 0 and (nq or nb) else 1 if nq else 0
                result[j] = max(result[j], category)
        jo += jet_counts[event]
        go += ng
    return result


TOPW_GEN_FIELDS = ("pdgId", "genPartIdxMother", "eta", "phi")

TOPW_ID_FIELDS = ("run", "luminosityBlock", "event", "entry", "file_id")

def topw_truth_payload(stored: Any, nano: Any, classifier: Any) -> dict[str, Any]:
    """Match the retained rows/jets, then reuse the efficiency truth classifier."""
    if len(stored) != len(nano):
        raise ValueError("NanoAOD and retained event counts differ")
    for name in ("run", "luminosityBlock", "event"):
        if not np.array_equal(np.asarray(stored[name]), np.asarray(nano[name])):
            raise ValueError(f"NanoAOD event identity mismatch: {name}")
    indices = stored["fatjet_source_index_all"]
    counts = ak.num(indices)
    if bool(ak.any((indices < 0) | (indices >= ak.num(nano["FatJet_eta"])))):
        raise ValueError("invalid stored FatJet source index")
    for axis in ("eta", "phi"):
        original = nano["FatJet_" + axis][indices]
        retained = stored["fatjet_" + axis + "_all"]
        if not np.array_equal(np.asarray(ak.num(retained)), np.asarray(counts)):
            raise ValueError("misaligned stored FatJet vectors")
        if not bool(ak.all(np.isfinite(original))) or not bool(ak.all(np.isfinite(retained))):
            raise ValueError("non-finite FatJet direction")
        if not np.allclose(np.asarray(ak.flatten(original)), np.asarray(ak.flatten(retained)),
                           rtol=0, atol=1e-6):
            raise ValueError(f"FatJet source identity mismatch: {axis}")
    gen_counts = ak.num(nano["GenPart_pdgId"])
    for name in TOPW_GEN_FIELDS:
        if not np.array_equal(np.asarray(ak.num(nano["GenPart_" + name])), np.asarray(gen_counts)):
            raise ValueError("misaligned GenPart vectors")
    mother = nano["GenPart_genPartIdxMother"]
    if bool(ak.any((mother < -1) | (mother >= gen_counts))):
        raise ValueError("invalid GenPart mother index")
    for name in ("eta", "phi"):
        if not bool(ak.all(np.isfinite(nano["GenPart_" + name]))):
            raise ValueError("non-finite GenPart direction")
    flat = lambda value: np.asarray(ak.to_numpy(ak.flatten(value)))
    flavor = classifier(
        np.asarray(counts), flat(stored["fatjet_eta_all"]), flat(stored["fatjet_phi_all"]),
        np.asarray(gen_counts), *(flat(nano["GenPart_" + name]) for name in TOPW_GEN_FIELDS),
    )
    if len(flavor) != int(ak.sum(counts)) or np.any((flavor < 0) | (flavor > 4)):
        raise ValueError("invalid Top/W truth classifier output")
    payload = {name: np.asarray(stored[name]) for name in TOPW_ID_FIELDS}
    payload["fatjet_source_index_all"] = ak.values_astype(indices, np.int32)
    payload["fatjet_decay_flavor_all"] = ak.unflatten(flavor.astype(np.int8), counts)
    return payload
