from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from . import flat_ntuple_worker as flat
from . import real_subset_worker as baseline
from .object_corrections_2025 import (
    REQUIRED_BRANCHES,
    SHAPE_VARIATIONS,
    branch_audit,
    calibrate_jets_and_met,
    validate_payloads,
    validate_shift,
)


EXTRA_FLOAT_FIELDS = [
    "rho",
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
    "fatjet_id_all",
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


_ORIGINAL_EXTRACT = baseline.extract_chunk


def _as_list(values: Any, index: int) -> list[Any]:
    return ak.to_list(values[index])


def _decorate_rows(rows: list[dict[str, Any]], raw: Any, corrected: Any) -> None:
    jet_id, _ = baseline.ak4_tight_lepton_veto_mask(
        corrected, corrected["Jet_pt"], corrected["Jet_eta"], Path.cwd(),
    )
    fatjet_id, _ = baseline.ak8_tight_lepton_veto_mask(
        corrected, corrected["FatJet_pt"], corrected["FatJet_eta"], Path.cwd(),
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
        "jet_btag_upart_all": raw["Jet_btagUParTAK4B"],
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
    for index, row in enumerate(rows):
        row["rho"] = float(raw["Rho_fixedGridRhoFastjetAll"][index])
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


def _passthrough_jec(
    arrays: Any,
    repo: Path,
    year: str,
    process: str,
    prefix: str,
    pt: Any,
    eta: Any,
    phi: Any,
    mass: Any,
    shift_name: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    return pt, mass, {
        "object": "AK8" if prefix == "FatJet" else "AK4",
        "applied": True,
        "source": "object_corrections_2025_precalibrated",
        "shift": validate_shift(shift_name),
        "shift_applied": validate_shift(shift_name) != "nominal",
    }


def _passthrough_met(
    arrays: Any,
    n: int,
    shift_name: str | None,
    process: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return (
        np.asarray(arrays["PuppiMET_pt"], dtype=float),
        np.asarray(arrays["PuppiMET_phi"], dtype=float),
        {
            "collection": "PuppiMET",
            "shift": validate_shift(shift_name),
            "applied": True,
            "source": "object_corrections_2025_precalibrated_and_propagated",
        },
    )


def extract_chunk_2025(
    arrays: Any,
    dataset: str,
    process: str,
    sp: str | None,
    year: str,
    file_path: str,
    entry_start: int,
    entry_stop: int,
    fastsim_trigger_bypass: bool = False,
    shift_name: str | None = None,
    compute_weights: bool = True,
    materialize_skim_flag: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(year) != "2025":
        raise RuntimeError(f"intermediate_2025_data_worker only accepts year 2025, got {year!r}")
    is_data = process in flat.DATA_PROCESSES
    audit = branch_audit(set(arrays.fields), is_data=is_data)
    if audit["status"] != "valid":
        raise RuntimeError(f"required 2025 object branches missing: {audit['missing_required']}")
    root = Path.cwd()
    payload_status = validate_payloads(root)
    if payload_status["status"] != "valid":
        raise RuntimeError(f"2025 correction payload validation failed: {payload_status['errors']}")
    corrected, calibration = calibrate_jets_and_met(arrays, is_data=is_data, shift=validate_shift(shift_name), root=root)
    rows, summary = _ORIGINAL_EXTRACT(
        corrected,
        dataset,
        process,
        sp,
        year,
        file_path,
        entry_start,
        entry_stop,
        fastsim_trigger_bypass=fastsim_trigger_bypass,
        shift_name=validate_shift(shift_name),
        compute_weights=compute_weights,
        materialize_skim_flag=materialize_skim_flag,
    )
    _decorate_rows(rows, arrays, corrected, entry_start)
    summary["object_corrections_2025"] = calibration
    summary["object_branch_audit"] = audit
    summary["payload_status"] = payload_status
    summary["intermediate_schema"] = "flat_ntuple_shard_v4_objectcorr_2025_data"
    summary["dy_recoil_selection"] = {
        "DY2E": "electron-cleaned jets; deltaPhi(j1..j4,uT)>0.5; uT>250 GeV",
        "DY2M": "muon-cleaned jets; deltaPhi(j1..j4,uT)>0.5; uT>250 GeV",
    }
    return rows, summary


def install_backend() -> None:
    baseline.analysis_year = lambda year: "2025" if baseline.campaign_year(str(year)) == "2025" else baseline.campaign_year(str(year))
    baseline.JET_VETO_MAP_RELATIVE_PATH = Path("analysis/data/JMESF/2024/jetvetomaps.json.gz")
    baseline.JET_VETO_MAP_CORRECTION = "Summer24Prompt25_RunCDEFG_V1"
    baseline._CORRECTION_CACHE.clear()
    for names in REQUIRED_BRANCHES.values():
        for name in names:
            if name not in flat.CORE_BRANCHES:
                flat.CORE_BRANCHES.append(name)
    for name in EXTRA_FLOAT_FIELDS:
        if name not in flat.FLOAT_FIELDS:
            flat.FLOAT_FIELDS.append(name)
    for name in EXTRA_VECTOR_FLOAT_FIELDS:
        if name not in flat.VECTOR_FLOAT_FIELDS:
            flat.VECTOR_FLOAT_FIELDS.append(name)
    for name in EXTRA_VECTOR_INT_FIELDS:
        if name not in flat.VECTOR_INT_FIELDS:
            flat.VECTOR_INT_FIELDS.append(name)
    flat.SCHEMA_VERSION = "flat_ntuple_shard_v4_objectcorr_2025_data"
    baseline.SHAPE_SHIFT_NAMES = set(SHAPE_VARIATIONS) - {"nominal"}
    baseline.validate_shift_name = validate_shift
    baseline.apply_jec = _passthrough_jec
    baseline.shifted_met = _passthrough_met
    flat.validate_shift_name = validate_shift
    flat.extract_chunk = extract_chunk_2025


def main(argv: list[str] | None = None) -> int:
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    install_backend()
    return flat.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
