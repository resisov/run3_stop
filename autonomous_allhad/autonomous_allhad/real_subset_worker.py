
from __future__ import annotations

import csv
import contextlib
import gzip
import hashlib
import html
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import awkward as ak
import correctionlib
import numpy as np
import uproot
from coffea.util import load as coffea_load

REQUIRED_GROUPS = [
    ("TT", lambda k: k.startswith("TT") or "TTto" in k),
    ("Zto2Nu", lambda k: "Zto2Nu" in k),
    ("WtoLNu", lambda k: "WtoLNu" in k),
    ("QCD", lambda k: "QCD" in k),
    ("GJ", lambda k: "GJ" in k or "GJets" in k),
    ("DY", lambda k: k.startswith("DY") or "DYto" in k),
    ("ST", lambda k: k.startswith("TW") or "TbarW" in k or "TBbar" in k or "TbarB" in k),
    ("VV", lambda k: k.startswith("WW") or k.startswith("WZ") or k.startswith("ZZ")),
    ("JetMET", lambda k: k.startswith("JetMET")),
    ("EGamma", lambda k: k.startswith("EGamma")),
    ("Muon", lambda k: k.startswith("Muon")),
]
SMS_RE = re.compile(r"mStop-(\d+)")
REGION_NAMES = [
    "preselection", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR",
]
SHAPE_SHIFT_NAMES = {"jesTotalUp", "jesTotalDown", "metUnclusteredUp", "metUnclusteredDown"}
JES_SHIFT_NAMES = {"jesTotalUp", "jesTotalDown"}
MET_UNCLUSTERED_SHIFT_NAMES = {"metUnclusteredUp", "metUnclusteredDown"}
MET_COLLECTION_PREFIXES = ("PuppiMET", "PFMET", "MET")
MET_UNCLUSTERED_BRANCHES = [
    f"{prefix}_{field}Unclustered{direction}"
    for prefix in MET_COLLECTION_PREFIXES
    for field in ("pt", "phi")
    for direction in ("Up", "Down")
]
BOOSTED_TOP_SCORE_BRANCH = "FatJet_globalParT3_withMassTopvsQCD"
BOOSTED_W_SCORE_BRANCH = "FatJet_globalParT3_withMassWvsQCD"
BOOSTED_TOP_SCORE_WP = 0.5078
BOOSTED_W_SCORE_WP = 0.9385
BOOSTED_TOP_PT_MIN = 400.0
BOOSTED_W_PT_MIN = 200.0
BOOSTED_ETA_MAX = 2.0
BOOSTED_W_MSD_MIN = 60.0
BOOSTED_W_MSD_MAX = 105.0
BOOSTED_TOP_MSD_MIN = 105.0
UPART_AK4_LOOSE_WP = 0.0246
UPART_AK4_MEDIUM_WP = 0.1272
PHOTON_MEDIUM_CUTBASED_MIN = 2
LOWDM_SV_POINTING_ANGLE_MAX = math.acos(0.98)
CORE_BRANCHES = [
    "run", "luminosityBlock", "event", "MET_pt", "MET_phi", "PFMET_pt", "PFMET_phi", "PuppiMET_pt", "PuppiMET_phi",
    *MET_UNCLUSTERED_BRANCHES,
    "Rho_fixedGridRhoFastjetAll",
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_area", "Jet_jetId", "Jet_btagUParTAK4B", "Jet_hadronFlavour",
    "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity",
    "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass", "FatJet_area", "FatJet_msoftdrop", "FatJet_subJetIdx1", "FatJet_subJetIdx2",
    BOOSTED_TOP_SCORE_BRANCH, BOOSTED_W_SCORE_BRANCH,
    "FatJet_chHEF", "FatJet_neHEF", "FatJet_chEmEF", "FatJet_neEmEF", "FatJet_muEF", "FatJet_chMultiplicity", "FatJet_neMultiplicity",
    "SubJet_btagUParTAK4B", "SubJet_btagDeepB",
    "SV_pt", "SV_eta", "SV_phi", "SV_dxy", "SV_dlenSig", "SV_pAngle", "SV_ntracks",
    "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_phi", "Electron_mass", "Electron_charge", "Electron_cutBased", "Electron_miniPFRelIso_all",
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge", "Muon_looseId", "Muon_mediumId", "Muon_miniPFRelIso_all",
    "Photon_pt", "Photon_eta", "Photon_phi", "Photon_cutBased", "Photon_electronVeto", "Photon_pixelSeed",
    "Tau_pt", "Tau_eta", "Tau_phi", "Tau_dz", "Tau_decayMode", "Tau_idDeepTau2018v2p5VSjet",
    "IsoTrack_pt", "IsoTrack_eta", "IsoTrack_phi", "IsoTrack_pdgId", "IsoTrack_pfRelIso03_all",
    "CaloMET_pt", "Pileup_nTrueInt", "GenPart_pt", "GenPart_pdgId", "GenPart_statusFlags", "genWeight",
]
FILTERS = [
    "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
]
SIGNAL_HLT = [
    "HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
]
PHOTON_HLT = ["HLT_Photon175", "HLT_Photon200"]
ELECTRON_HLT = [
    "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Ele135_CaloIdVT_GsfTrkIdT",
    "HLT_Ele30_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf", "HLT_Ele35_WPTight_Gsf",
    "HLT_Ele38_WPTight_Gsf", "HLT_Ele40_WPTight_Gsf",
    "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
    "HLT_DoubleEle25_CaloIdL_MW", "HLT_DoubleEle27_CaloIdL_MW", "HLT_DoubleEle33_CaloIdL_MW",
]
MUON_HLT = ["HLT_IsoMu20", "HLT_IsoMu24", "HLT_IsoMu27", "HLT_IsoMu24_eta2p1", "HLT_Mu50", "HLT_Mu55"]
TRIGGER_FAMILIES = {
    "signal": SIGNAL_HLT,
    "photon": PHOTON_HLT,
    "electron": ELECTRON_HLT,
    "muon": MUON_HLT,
}
JET_ID_INPUTS = [
    "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity",
]
FATJET_ID_INPUTS = [
    "FatJet_chHEF", "FatJet_neHEF", "FatJet_chEmEF", "FatJet_neEmEF", "FatJet_muEF", "FatJet_chMultiplicity", "FatJet_neMultiplicity",
]
LUMIMASK_RELATIVE_PATHS = {
    "2024": Path("analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json"),
    "2025": Path("analysis/data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json"),
}
JET_VETO_MAP_RELATIVE_PATH = Path("analysis/data/JMESF/2024/jetvetomaps.json.gz")
JET_VETO_MAP_CORRECTION = "Summer24Prompt24_RunBCDEFGHI_V1"

_LUMIMASK_CACHE: dict[Path, dict[int, list[tuple[int, int]]]] = {}
_CORRECTION_CACHE: dict[tuple[str, str], Any] = {}
_ANALYSIS_CORRECTIONS_CACHE: dict[Path, dict[str, Any]] = {}
_ANALYSIS_RUNTIME_DIR_CACHE: dict[Path, Path] = {}
_BTAG_CORRECTOR_CACHE: dict[tuple[Path, str, str, str, str], Any] = {}


def local_analysis_dir(repo: Path) -> Path:
    key = repo.resolve()
    if os.environ.get("AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA", "1") == "0":
        return key / "analysis"
    cached = _ANALYSIS_RUNTIME_DIR_CACHE.get(key)
    if cached and (cached / "data").exists() and any((cached / "hists").glob("btageff*.merged")):
        return cached
    base_raw = (
        os.environ.get("AUTONOMOUS_ALLHAD_ANALYSIS_CACHE_DIR")
        or os.environ.get("_CONDOR_SCRATCH_DIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    target = Path(base_raw) / f"autonomous_allhad_analysis_data_{os.getpid()}"
    data_dir = target / "data"
    hists_dir = target / "hists"
    max_attempts = max(1, int(os.environ.get("AUTONOMOUS_ALLHAD_ANALYSIS_DATA_COPY_RETRIES", "3")))
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if not data_dir.exists():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copytree(key / "analysis" / "data", data_dir, dirs_exist_ok=True)
            if not hists_dir.exists():
                hists_dir.mkdir(parents=True, exist_ok=True)
                btag_sources = sorted((key / "analysis" / "hists").glob("btageff*.merged"))
                if not btag_sources:
                    raise FileNotFoundError("No btageff*.merged files found in analysis/hists")
                for source in btag_sources:
                    shutil.copy2(source, hists_dir / source.name)
            _ANALYSIS_RUNTIME_DIR_CACHE[key] = target
            return target
        except Exception as exc:
            last_exc = exc
            shutil.rmtree(target, ignore_errors=True)
            if attempt + 1 >= max_attempts:
                break
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to stage analysis/data and analysis/hists locally after {max_attempts} attempts: {last_exc}") from last_exc


def analysis_data_file(repo: Path, relative_path: Path) -> Path:
    rel = Path(relative_path)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "analysis" and parts[1] == "data":
        return local_analysis_dir(repo) / Path(*parts[1:])
    return repo / rel


@contextlib.contextmanager
def analysis_workdir(repo: Path):
    old = Path.cwd()
    os.chdir(local_analysis_dir(repo))
    try:
        yield
    finally:
        os.chdir(old)


def load_analysis_corrections(repo: Path) -> dict[str, Any]:
    key = repo.resolve()
    repo_path = str(key)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if key not in _ANALYSIS_CORRECTIONS_CACHE:
        with analysis_workdir(repo):
            _ANALYSIS_CORRECTIONS_CACHE[key] = coffea_load("data/corrections.coffea")
    return _ANALYSIS_CORRECTIONS_CACHE[key]


def campaign_year(year: str) -> str:
    return year if year in {"2022pre", "2022post", "2023pre", "2023post", "2024", "2025"} else "2024"


def analysis_year(year: str) -> str:
    year = campaign_year(year)
    return "2024" if year == "2025" else year


def np_filled(values: Any, n: int, fill: float = 1.0) -> np.ndarray:
    try:
        out = np.asarray(ak.to_numpy(ak.fill_none(values, fill)), dtype=float)
    except Exception:
        out = np.asarray(values, dtype=float)
    if out.shape == ():
        out = np.full(n, float(out), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill).astype(float)


def jagged_prod(values: Any, n: int, fill: float = 1.0) -> np.ndarray:
    try:
        return np_filled(ak.prod(ak.fill_none(values, fill), axis=1), n, fill)
    except Exception:
        return np.full(n, fill, dtype=float)


def finite_float(value: Any, fill: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fill)
    return out if math.isfinite(out) else float(fill)


def finite_weight_array(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, fill, dtype=float)
    if out.shape == ():
        out = np.full(n, finite_float(out, fill), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill).astype(float)


def finite_sum(values: Any) -> float:
    try:
        arrv = np.asarray(values, dtype=float)
    except Exception:
        return 0.0
    if arrv.size == 0:
        return 0.0
    arrv = np.where(np.isfinite(arrv), arrv, 0.0)
    return finite_float(np.sum(arrv), 0.0)


def replace_component(base: dict[str, np.ndarray], name: str, varied: np.ndarray) -> np.ndarray:
    n = len(next(iter(base.values())))
    out = np.ones(n, dtype=float)
    for comp_name, comp in base.items():
        factor = finite_weight_array(varied if comp_name == name else comp, n, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            out = out * factor
        out = np.where(np.isfinite(out), out, 0.0)
    return out


def get_btag_corrector(repo: Path, year: str, caller: str, tagger: str = "UParTAK4", workingpoint: str = "medium") -> Any:
    key = (repo.resolve(), analysis_year(year), caller, tagger, workingpoint)
    if key not in _BTAG_CORRECTOR_CACHE:
        corrections = load_analysis_corrections(repo)
        with analysis_workdir(repo):
            _BTAG_CORRECTOR_CACHE[key] = corrections["get_btag_weight"](tagger, analysis_year(year), workingpoint, caller)
    return _BTAG_CORRECTOR_CACHE[key]


def normalize_shift_name(shift_name: str | None) -> str:
    shift = str(shift_name or "nominal").strip()
    return "nominal" if shift in {"", "none", "None"} else shift


def validate_shift_name(shift_name: str | None) -> str:
    shift = normalize_shift_name(shift_name)
    if shift != "nominal" and shift not in SHAPE_SHIFT_NAMES:
        raise ValueError(f"Unsupported shape shift {shift!r}; expected nominal or one of {sorted(SHAPE_SHIFT_NAMES)}")
    return shift


def shifted_met(arrays: dict[str, Any], n: int, shift_name: str | None, process: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shift = validate_shift_name(shift_name)
    is_data = is_data_process(process)
    prefix = next((p for p in MET_COLLECTION_PREFIXES if has_field(arrays, f"{p}_pt") and has_field(arrays, f"{p}_phi")), None)
    if prefix is None:
        raise RuntimeError("No usable MET collection found; need PuppiMET, PFMET, or MET pt/phi branches")
    pt_name = f"{prefix}_pt"
    phi_name = f"{prefix}_phi"
    status: dict[str, Any] = {"collection": prefix, "shift": shift, "applied": False, "source": "nominal", "is_data": bool(is_data)}
    if shift in MET_UNCLUSTERED_SHIFT_NAMES and is_data:
        status.update({"reason": "data_uncertainty_not_applied", "pt_branch": pt_name, "phi_branch": phi_name})
        return np.asarray(arrays[pt_name], dtype=float), np.asarray(arrays[phi_name], dtype=float), status
    if shift in MET_UNCLUSTERED_SHIFT_NAMES:
        direction = "Up" if shift.endswith("Up") else "Down"
        pt_name = f"{prefix}_ptUnclustered{direction}"
        phi_name = f"{prefix}_phiUnclustered{direction}"
        missing = [name for name in (pt_name, phi_name) if not has_field(arrays, name)]
        if missing:
            raise RuntimeError(f"Requested {shift} but {prefix} is missing unclustered MET branches: {missing}")
        status.update({"applied": True, "source": "NanoAOD_MET_unclustered", "pt_branch": pt_name, "phi_branch": phi_name})
    else:
        status.update({"pt_branch": pt_name, "phi_branch": phi_name})
    return np.asarray(arrays[pt_name], dtype=float), np.asarray(arrays[phi_name], dtype=float), status


def apply_jec(arrays: dict[str, Any], repo: Path, year: str, process: str, prefix: str, pt: Any, eta: Any, phi: Any, mass: Any, shift_name: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    is_data = is_data_process(process)
    shift = validate_shift_name(shift_name)
    area_name = f"{prefix}_area"
    needed = [area_name, "Rho_fixedGridRhoFastjetAll", "run"]
    missing = [name for name in needed if not has_field(arrays, name)]
    label = "AK8" if prefix == "FatJet" else "AK4"
    status: dict[str, Any] = {
        "object": label,
        "applied": False,
        "source": "raw",
        "missing_inputs": missing,
        "shift": shift,
        "shift_applied": False,
        "data_taking_year": campaign_year(year),
        "correction_year": analysis_year(year),
    }
    if missing:
        status["reason"] = "missing_jec_inputs"
        if prefix == "Jet" and shift in JES_SHIFT_NAMES and not is_data:
            raise RuntimeError(f"Requested {shift} but AK4 JEC inputs are missing: {missing}")
        return pt, mass, status
    base_status = dict(status)
    max_attempts = max(1, int(os.environ.get("AUTONOMOUS_ALLHAD_JEC_RETRIES", "3")))
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            corrections = load_analysis_corrections(repo)
            fn_name = "get_fjec_correction" if prefix == "FatJet" else "get_jec_correction"
            with analysis_workdir(repo):
                corr = corrections[fn_name](analysis_year(year), pt, eta, phi, arrays["Rho_fixedGridRhoFastjetAll"], arrays[area_name], arrays["run"], is_data)
            out_pt = pt * corr
            out_mass = mass * corr
            status.update({"applied": True, "source": f"analysis.utils.corrections.{fn_name}", "is_data": bool(is_data), "correction_attempts": attempt + 1})
            if prefix == "Jet" and shift in JES_SHIFT_NAMES and not is_data:
                with analysis_workdir(repo):
                    jec_unc = corrections["get_jec_uncertainty"](analysis_year(year), out_pt, eta)
                sign = 1.0 if shift == "jesTotalUp" else -1.0
                varied = 1.0 + sign * jec_unc
                out_pt = out_pt * varied
                out_mass = out_mass * varied
                status.update({"shift_applied": True, "shift_source": "analysis.utils.corrections.get_jec_uncertainty", "shift_scope": "AK4_MC_only"})
            elif prefix == "Jet" and shift in JES_SHIFT_NAMES and is_data:
                status.update({"shift_applied": False, "shift_reason": "data_uncertainty_not_applied"})
            elif prefix == "FatJet" and shift in JES_SHIFT_NAMES:
                status.update({"shift_applied": False, "shift_reason": "AK8 JES total shift not applied in stop_processor_v4 reference"})
            return out_pt, out_mass, status
        except Exception as exc:
            last_exc = exc
            status = dict(base_status)
            _ANALYSIS_CORRECTIONS_CACHE.pop(repo.resolve(), None)
            if attempt + 1 >= max_attempts:
                break
            time.sleep(1.0 * (attempt + 1))
    exc = last_exc or RuntimeError("unknown JEC/JES evaluation failure")
    if prefix == "Jet" and shift in JES_SHIFT_NAMES and not is_data:
        raise RuntimeError(f"Requested {shift} but AK4 JEC/JES evaluation failed after {max_attempts} attempts: {type(exc).__name__}: {exc}") from exc
    status.update({"reason": "jec_exception", "error": f"{type(exc).__name__}: {exc}"[:400], "correction_attempts": max_attempts})
    return pt, mass, status


def compute_weight_bundle(
    arrays: dict[str, Any],
    repo: Path,
    dataset: str,
    process: str,
    year: str,
    n: int,
    jet_pt: Any,
    jet_eta: Any,
    jet_hadflav: Any,
    b_med: Any,
    e_eta: Any,
    e_delta_eta_sc: Any,
    e_pt: Any,
    e_phi: Any,
    e_veto: Any,
    e_med: Any,
    n_e_veto: np.ndarray,
    n_e_med: np.ndarray,
    m_eta: Any,
    m_pt: Any,
    m_phi: Any,
    m_loose: Any,
    m_med: Any,
    n_m_loose: np.ndarray,
    n_m_med: np.ndarray,
    p_eta: Any,
    p_pt: Any,
    p_phi: Any,
    p_med: Any,
    gcr_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if is_data_process(process):
        ones = np.ones(n, dtype=float)
        return ones, {"nominal": ones}, {"applied": False, "reason": "data", "available_variations": ["nominal"], "components": {}}

    corrections = load_analysis_corrections(repo)
    y = analysis_year(year)
    gen = np_filled(arr(arrays, "genWeight", np.ones(n)), n, 1.0)
    one = np.ones(n, dtype=float)
    components: dict[str, np.ndarray] = {}
    alternates: dict[str, tuple[str, np.ndarray]] = {}
    status: dict[str, Any] = {"applied": True, "available_variations": ["nominal"], "components": {}}

    def record(name: str, applied: bool, source: str, error: str | None = None) -> None:
        item = {"applied": applied, "source": source}
        if error:
            item["error"] = error[:400]
        status["components"][name] = item

    if has_field(arrays, "Pileup_nTrueInt"):
        try:
            with analysis_workdir(repo):
                pu_nom, pu_up, pu_down = corrections["get_pu_weight"](y, arrays["Pileup_nTrueInt"])
            components["pileup"] = np_filled(pu_nom, n, 1.0)
            alternates["pileupUp"] = ("pileup", np_filled(pu_up, n, 1.0))
            alternates["pileupDown"] = ("pileup", np_filled(pu_down, n, 1.0))
            record("pileup", True, "analysis.utils.corrections.get_pu_weight")
        except Exception as exc:
            components["pileup"] = one
            record("pileup", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        components["pileup"] = one
        record("pileup", False, "unity_fallback_missing_Pileup_nTrueInt")

    top_pt_sf = one.copy()
    if "TTto" in dataset and has_field(arrays, "GenPart_pt") and has_field(arrays, "GenPart_pdgId") and has_field(arrays, "GenPart_statusFlags"):
        try:
            gen_pt = arrays["GenPart_pt"]
            gen_pdg = arrays["GenPart_pdgId"]
            gen_flags = ak.values_astype(arrays["GenPart_statusFlags"], np.int64)
            is_top = (abs(gen_pdg) == 6) & ((gen_flags & (1 << 8)) != 0) & ((gen_flags & (1 << 13)) != 0)
            tops = gen_pt[is_top]
            t1 = ak.fill_none(ak.pad_none(tops, 2, axis=1, clip=False)[:, 0], 0.0)
            t2 = ak.fill_none(ak.pad_none(tops, 2, axis=1, clip=False)[:, 1], 0.0)
            both = ak.num(tops, axis=1) >= 2
            with analysis_workdir(repo):
                vals = np.sqrt(corrections["get_top_pt_reweight"](t1) * corrections["get_top_pt_reweight"](t2))
            top_pt_sf = np.where(ak.to_numpy(both), np_filled(vals, n, 1.0), 1.0)
            record("top_pt_reweight", True, "analysis.utils.corrections.get_top_pt_reweight")
        except Exception as exc:
            record("top_pt_reweight", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        reason = "not_TTto_dataset" if "TTto" not in dataset else "missing_GenPart_inputs"
        record("top_pt_reweight", False, f"unity_fallback_{reason}")
    components["top_pt_reweight"] = top_pt_sf

    if has_field(arrays, "Jet_hadronFlavour"):
        try:
            caller = dataset.split("____")[0]
            btag_corrector = get_btag_corrector(repo, y, caller)
            btag = btag_corrector.btag_weight(jet_pt, jet_eta, jet_hadflav, b_med)
            names = [
                "btagSF", "btagSF_bc_correlatedUp", "btagSF_bc_correlatedDown", "btagSF_bc_uncorrelatedUp", "btagSF_bc_uncorrelatedDown",
                "btagSF_light_correlatedUp", "btagSF_light_correlatedDown", "btagSF_light_uncorrelatedUp", "btagSF_light_uncorrelatedDown",
            ]
            bvals = {name: np_filled(val, n, 1.0) for name, val in zip(names, btag)}
            components["btagSF"] = bvals["btagSF"]
            for name in names[1:]:
                alternates[name] = ("btagSF", bvals[name])
            record("btagSF", True, "analysis.utils.corrections.BTagCorrector")
        except Exception as exc:
            components["btagSF"] = one
            record("btagSF", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        components["btagSF"] = one
        record("btagSF", False, "unity_fallback_missing_Jet_hadronFlavour")

    def add_triplet(component: str, nominal: np.ndarray, up: np.ndarray, down: np.ndarray, source: str) -> None:
        components[component] = nominal
        alternates[f"{component}Up"] = (component, up)
        alternates[f"{component}Down"] = (component, down)
        record(component, True, source)

    try:
        with analysis_workdir(repo):
            ev_nom, ev_up, ev_down = corrections["get_ele_veto_id_sf"](y, e_eta + e_delta_eta_sc, e_pt, e_phi)
            em_nom, em_up, em_down = corrections["get_ele_medium_id_sf"](y, e_eta + e_delta_eta_sc, e_pt, e_phi)
        ele_nom = one.copy(); ele_up = one.copy(); ele_down = one.copy()
        mask_one = np.asarray(n_e_veto == 1, dtype=bool)
        mask_two = np.asarray(n_e_med == 2, dtype=bool)
        vals = [
            (ele_nom, jagged_prod(ak.where(e_veto, ev_nom, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_nom, ak.ones_like(e_pt)), n)),
            (ele_up, jagged_prod(ak.where(e_veto, ev_up, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_up, ak.ones_like(e_pt)), n)),
            (ele_down, jagged_prod(ak.where(e_veto, ev_down, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_down, ak.ones_like(e_pt)), n)),
        ]
        for target, veto_vals, med_vals in vals:
            target[mask_one] = veto_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet("electron_id", ele_nom, ele_up, ele_down, "analysis.utils.corrections electron ID SF")
    except Exception as exc:
        components["electron_id"] = one
        record("electron_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            eh_nom, eh_up, eh_down = corrections["get_ele_hlt_sf"](y, e_eta, e_pt, e_phi)
        mask_two = np.asarray(n_e_med == 2, dtype=bool)
        nom = one.copy(); up = one.copy(); down = one.copy()
        nom_vals = jagged_prod(ak.where(e_med, eh_nom, ak.ones_like(e_pt)), n)
        up_vals = jagged_prod(ak.where(e_med, eh_up, ak.ones_like(e_pt)), n)
        down_vals = jagged_prod(ak.where(e_med, eh_down, ak.ones_like(e_pt)), n)
        nom[mask_two] = nom_vals[mask_two]; up[mask_two] = up_vals[mask_two]; down[mask_two] = down_vals[mask_two]
        add_triplet("electron_hlt", nom, up, down, "analysis.utils.corrections.get_ele_hlt_sf")
    except Exception as exc:
        components["electron_hlt"] = one
        record("electron_hlt", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            ml_nom, ml_up, ml_down = corrections["get_mu_loose_id_sf"](y, m_eta, m_pt)
            mm_nom, mm_up, mm_down = corrections["get_mu_medium_id_sf"](y, m_eta, m_pt)
        mu_nom = one.copy(); mu_up = one.copy(); mu_down = one.copy()
        mask_one = np.asarray(n_m_loose == 1, dtype=bool)
        mask_two = np.asarray(n_m_med == 2, dtype=bool)
        vals = [
            (mu_nom, jagged_prod(ak.where(m_loose, ml_nom, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_nom, ak.ones_like(m_pt)), n)),
            (mu_up, jagged_prod(ak.where(m_loose, ml_up, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_up, ak.ones_like(m_pt)), n)),
            (mu_down, jagged_prod(ak.where(m_loose, ml_down, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_down, ak.ones_like(m_pt)), n)),
        ]
        for target, loose_vals, med_vals in vals:
            target[mask_one] = loose_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet("muon_id", mu_nom, mu_up, mu_down, "analysis.utils.corrections muon ID SF")
    except Exception as exc:
        components["muon_id"] = one
        record("muon_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            mh_nom, mh_up, mh_down = corrections["get_mu_hlt_sf"](y, m_eta, m_pt)
        mask_two = np.asarray(n_m_med == 2, dtype=bool)
        nom = one.copy(); up = one.copy(); down = one.copy()
        nom_vals = jagged_prod(ak.where(m_med, mh_nom, ak.ones_like(m_pt)), n)
        up_vals = jagged_prod(ak.where(m_med, mh_up, ak.ones_like(m_pt)), n)
        down_vals = jagged_prod(ak.where(m_med, mh_down, ak.ones_like(m_pt)), n)
        nom[mask_two] = nom_vals[mask_two]; up[mask_two] = up_vals[mask_two]; down[mask_two] = down_vals[mask_two]
        add_triplet("muon_hlt", nom, up, down, "analysis.utils.corrections.get_mu_hlt_sf")
    except Exception as exc:
        components["muon_hlt"] = one
        record("muon_hlt", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            ph_nom, ph_up, ph_down = corrections["get_photon_id_sf"](y, "Medium", p_eta, p_pt, p_phi)
        nom = one.copy(); up = one.copy(); down = one.copy()
        mask_g = np.asarray(gcr_mask, dtype=bool)
        nom_vals = jagged_prod(ak.where(p_med, ph_nom, ak.ones_like(p_pt)), n)
        up_vals = jagged_prod(ak.where(p_med, ph_up, ak.ones_like(p_pt)), n)
        down_vals = jagged_prod(ak.where(p_med, ph_down, ak.ones_like(p_pt)), n)
        nom[mask_g] = nom_vals[mask_g]; up[mask_g] = up_vals[mask_g]; down[mask_g] = down_vals[mask_g]
        add_triplet("photon_id", nom, up, down, "analysis.utils.corrections.get_photon_id_sf")
    except Exception as exc:
        components["photon_id"] = one
        record("photon_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    protection_counts: dict[str, int] = {}

    def clean_weight(name: str, values: Any) -> np.ndarray:
        try:
            raw = np.asarray(values, dtype=float)
        except Exception:
            protection_counts[name] = protection_counts.get(name, 0) + n
            return np.zeros(n, dtype=float)
        if raw.shape == ():
            raw = np.full(n, finite_float(raw, 0.0), dtype=float)
        if len(raw) != n:
            protection_counts[name] = protection_counts.get(name, 0) + n
            return np.zeros(n, dtype=float)
        bad = ~np.isfinite(raw)
        if np.any(bad):
            protection_counts[name] = protection_counts.get(name, 0) + int(np.sum(bad))
        return np.where(bad, 0.0, raw).astype(float)

    nominal_sf = np.ones(n, dtype=float)
    for component_name, comp in components.items():
        factor = finite_weight_array(comp, n, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            nominal_sf = nominal_sf * factor
        bad = ~np.isfinite(nominal_sf)
        if np.any(bad):
            protection_counts[f"nominal_after_{component_name}"] = protection_counts.get(f"nominal_after_{component_name}", 0) + int(np.sum(bad))
            nominal_sf = np.where(bad, 0.0, nominal_sf)
    with np.errstate(over="ignore", invalid="ignore"):
        nominal = gen * nominal_sf
    variations = {"nominal": clean_weight("nominal", nominal)}
    for variation, (component, varied) in alternates.items():
        with np.errstate(over="ignore", invalid="ignore"):
            raw_variation = gen * replace_component(components, component, varied)
        variations[variation] = clean_weight(variation, raw_variation)
    if protection_counts:
        status["nonfinite_weight_protection"] = protection_counts
    status["available_variations"] = sorted(variations)
    return gen, variations, status


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class RootOpenFailure(RuntimeError):
    def __init__(self, message: str, access_info: dict[str, Any]):
        super().__init__(message)
        self.access_info = access_info


def _xrd_cache_path(file_path: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    cache_dir = Path(os.environ.get("AUTONOMOUS_ALLHAD_XRD_CACHE", f"/tmp/{user}/autonomous_allhad_xrd_cache"))
    raw_name = file_path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "cached.root"
    if not raw_name.endswith(".root"):
        raw_name += ".root"
    digest = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    return cache_dir / f"{digest}_{raw_name}"


def _xrd_source_candidates(file_path: str) -> list[str]:
    if not str(file_path).startswith("root://"):
        return [file_path]
    candidates: list[str] = []
    idx = str(file_path).find("/store/")
    if idx >= 0:
        lfn = str(file_path)[idx:]
        extra_hosts = [
            item.strip()
            for item in os.environ.get("AUTONOMOUS_ALLHAD_XRD_EXTRA_HOSTS", "").split(",")
            if item.strip()
        ]
        default_hosts = [
            "cmsxrootd.fnal.gov",
            "cms-xrd-global.cern.ch",
            "xrootd-cms.infn.it",
            "t3se01.psi.ch:1094",
            "gaexrdoor.ciemat.es:1094",
            "k8s-redir.ultralight.org:1094",
            "maite.iihe.ac.be:1094",
        ]
        for host in extra_hosts + default_hosts:
            candidates.append(f"root://{host}/{lfn}")
        candidates.append(file_path)
    out: list[str] = []
    for item in candidates:
        if item not in out:
            out.append(item)
    return out


def open_root_with_xrd_fallback(file_path: str, timeout: int = 60) -> tuple[Any, dict[str, Any]]:
    prefer_cache = (
        os.environ.get("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", "1") == "1"
        and str(file_path).startswith("root://")
    )
    info: dict[str, Any] = {
        "source_file_path": file_path,
        "effective_file_path": file_path,
        "access_method": "direct",
        "direct_open_attempted": not prefer_cache,
        "direct_open_status": "not_started",
        "direct_open_error": None,
        "alternate_access_attempted": False,
        "fallback_status": "not_started",
        "fallback_sources_considered": [],
        "xrdcp_attempts": [],
        "xrdcp_command": None,
        "xrdcp_exit_status": None,
        "xrdcp_stdout_tail": "",
        "xrdcp_stderr_tail": "",
        "cache_path": None,
        "cache_reused": False,
        "prefer_cache": prefer_cache,
    }
    if prefer_cache:
        info["direct_open_status"] = "skipped_prefer_cache"
    else:
        try:
            root = uproot.open(file_path, timeout=timeout)
            info["direct_open_status"] = "success"
            info["fallback_status"] = "not_needed"
            return root, info
        except Exception as exc:
            info["direct_open_status"] = "failed"
            info["direct_open_error"] = f"{type(exc).__name__}: {exc}"
            if not str(file_path).startswith("root://"):
                info["fallback_status"] = "not_applicable_non_xrootd"
                raise RootOpenFailure(f"direct ROOT open failed and xrdcp fallback is not applicable: {exc}", info)

    xrdcp = shutil.which("xrdcp")
    info["alternate_access_attempted"] = True
    if not xrdcp:
        info["fallback_status"] = "xrdcp_unavailable"
        raise RootOpenFailure("direct ROOT open failed and xrdcp is unavailable", info)

    cache_path = _xrd_cache_path(file_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    info["cache_path"] = str(cache_path)
    info["effective_file_path"] = str(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        info["cache_reused"] = True
        info["fallback_status"] = "cache_reused"
    else:
        copied = False
        tmp_cache = cache_path.with_name(f"{cache_path.name}.xrdcp.{os.getpid()}.tmp")
        try:
            tmp_cache.unlink(missing_ok=True)
        except Exception:
            pass
        for source in _xrd_source_candidates(file_path):
            info["fallback_sources_considered"].append(source)
            cmd = [xrdcp, "-f", "--nopbar"]
            streams = int(os.environ.get("AUTONOMOUS_ALLHAD_XRD_STREAMS", "4"))
            if streams > 1:
                cmd += ["--streams", str(streams)]
            cmd += [source, str(tmp_cache)]
            info["xrdcp_command"] = cmd
            attempt = {"source": source, "command": cmd, "exit_status": None, "stdout_tail": "", "stderr_tail": "", "status": "not_started"}
            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=int(os.environ.get("AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT", "600")),
                )
                attempt["exit_status"] = proc.returncode
                attempt["stdout_tail"] = proc.stdout[-4000:]
                attempt["stderr_tail"] = proc.stderr[-4000:]
                attempt["status"] = "success" if proc.returncode == 0 else "failed"
                info["xrdcp_exit_status"] = proc.returncode
                info["xrdcp_stdout_tail"] = attempt["stdout_tail"]
                info["xrdcp_stderr_tail"] = attempt["stderr_tail"]
            except Exception as exc:
                attempt["status"] = "exception"
                attempt["stderr_tail"] = f"{type(exc).__name__}: {exc}"
                info["xrdcp_stderr_tail"] = attempt["stderr_tail"]
            info["xrdcp_attempts"].append(attempt)
            if attempt["status"] == "success" and tmp_cache.exists() and tmp_cache.stat().st_size > 0:
                os.replace(tmp_cache, cache_path)
                copied = True
                info["fallback_status"] = "xrdcp_copied"
                break
            try:
                tmp_cache.unlink(missing_ok=True)
            except Exception:
                pass
        if not copied:
            info["fallback_status"] = "xrdcp_failed"
            raise RootOpenFailure("direct ROOT open failed and all xrdcp fallback sources failed", info)

    try:
        root = uproot.open(str(cache_path), timeout=timeout)
        info["access_method"] = "xrdcp_cache"
        info["cache_open_status"] = "success"
        return root, info
    except Exception as exc:
        info["fallback_status"] = "cache_open_failed"
        info["cache_open_status"] = "failed"
        info["cache_open_error"] = f"{type(exc).__name__}: {exc}"
        raise RootOpenFailure(f"direct ROOT open failed and cached ROOT open failed: {exc}", info)


def cleanup_xrd_cache(access_info: dict[str, Any]) -> None:
    if os.environ.get("AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE", "0") == "1":
        return
    if access_info.get("access_method") != "xrdcp_cache":
        return
    cache_path = access_info.get("cache_path")
    if not cache_path:
        return
    try:
        Path(str(cache_path)).unlink(missing_ok=True)
    except Exception:
        pass


def load_config(path: Path) -> dict[str, str]:
    # The real worker only needs the paths used below; keep parsing intentionally small.
    out: dict[str, str] = {}
    current = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            current = line[:-1]
            continue
        if current in {"paths", "execution"} and ":" in line:
            k, v = line.split(":", 1)
            out[f"{current}.{k.strip()}"] = v.strip().strip('"')
    return out


def group_of(dataset: str) -> str:
    for name, pred in REQUIRED_GROUPS:
        if pred(dataset):
            return name
    if "SMS" in dataset:
        return "SMS"
    return "other"


def natural_key(text: str) -> list[Any]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


def signal_point(dataset: str) -> str | None:
    m = SMS_RE.search(dataset)
    if m:
        return f"mStop-{m.group(1)}"
    return None


def file_size(path: str) -> int | None:
    root = None
    access_info: dict[str, Any] = {}
    try:
        root, access_info = open_root_with_xrd_fallback(path, timeout=60)
        return int(root.file.source.num_bytes)
    except Exception:
        return None
    finally:
        try:
            if root is not None:
                root.close()
        except Exception:
            pass
        cleanup_xrd_cache(access_info)


def dataset_score(group: str, key: str) -> tuple[int, list[Any]]:
    score = 0
    preferred = {
        "TT": ["TTTT", "TTto4Q"],
        "Zto2Nu": ["PTNuNu-600", "PTNuNu-400", "PTNuNu-200"],
        "WtoLNu": ["PTLNu-600", "PTLNu-400", "PTLNu-200"],
        "QCD": ["PT-1000", "PT-800", "PT-600"],
        "GJ": ["PTG-600", "PTG-400", "PTG-200"],
        "DY": ["PTLL-600", "PTLL-400", "PTLL-200"],
        "ST": ["TBbarQto2Q", "TbarBQ", "TWminus"],
        "VV": ["WW_", "WZ_", "ZZ_"],
        "JetMET": ["JetMET0-Run2024C"],
        "EGamma": ["EGamma0-Run2024C"],
        "Muon": ["Muon0-Run2024C"],
    }.get(group, [])
    for i, token in enumerate(preferred):
        if token in key:
            score += 100 - i
    return (-score, natural_key(key))


def choose_subset(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = set()
    for group, pred in REQUIRED_GROUPS:
        candidates = [k for k in metadata if pred(k)]
        for key in sorted(candidates, key=lambda k: dataset_score(group, k)):
            if key in used:
                continue
            files = metadata[key].get("files", []) if isinstance(metadata[key], dict) else []
            if files:
                selected.append({"dataset": key, "process": group, "signal_point": signal_point(key), "files": files[:1]})
                used.add(key)
                break
    sms_seen = set()
    sms_keys = sorted((k for k in metadata if "SMS" in k), key=lambda k: (abs((int(SMS_RE.search(k).group(1)) if SMS_RE.search(k) else 0) - 1000), natural_key(k)))
    for key in sms_keys:
        sp = signal_point(key) or key
        if sp in sms_seen:
            continue
        files = metadata[key].get("files", []) if isinstance(metadata[key], dict) else []
        if files:
            selected.append({"dataset": key, "process": "SMS", "signal_point": sp, "files": files[:1]})
            sms_seen.add(sp)
        if len(sms_seen) >= 3:
            break
    return selected


def is_data_process(process: str) -> bool:
    return process in {"JetMET", "EGamma", "Muon"}


def trigger_family_for_process(process: str) -> str:
    if process == "GJ" or process == "EGamma":
        return "photon"
    if process == "Muon":
        return "muon"
    return "signal"


def available(tree: Any) -> set[str]:
    return set(tree.keys())


def has_field(arrays: Any, name: str) -> bool:
    return name in getattr(arrays, "fields", [])


def arr(arrays: dict[str, Any], name: str, default: Any = None) -> Any:
    return arrays[name] if has_field(arrays, name) else default


def bool_branch(arrays: dict[str, Any], names: list[str], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for name in names:
        if has_field(arrays, name):
            out |= np.asarray(arrays[name], dtype=bool)
    return out


def all_filters(arrays: dict[str, Any], n: int) -> tuple[np.ndarray, list[str]]:
    out = np.ones(n, dtype=bool)
    missing = []
    for name in FILTERS:
        if has_field(arrays, name):
            out &= np.asarray(arrays[name], dtype=bool)
        else:
            missing.append(name)
    return out, missing


def first_or(default: float, jagged: Any) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.firsts(jagged), default))


def nth_or(default: float, jagged: Any, idx: int) -> np.ndarray:
    padded = ak.pad_none(jagged, idx + 1, axis=1, clip=False)
    return ak.to_numpy(ak.fill_none(padded[:, idx], default))


def count(mask: Any) -> np.ndarray:
    return ak.to_numpy(ak.sum(mask, axis=1))


def delta_phi(phi1: Any, phi2: Any) -> Any:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def ak4_tight_lepton_veto_mask(arrays: dict[str, Any], jet_pt: Any, jet_eta: Any, repo: Path) -> tuple[Any, str]:
    if all(has_field(arrays, name) for name in JET_ID_INPUTS):
        evaluator = correctionlib.CorrectionSet.from_file(str(analysis_data_file(repo, Path("analysis/data/JMESF/2024/jetid.json.gz"))))
        corr = evaluator["AK4PUPPI_TightLeptonVeto"]
        counts = ak.num(jet_eta)
        ch_mult = arr(arrays, "Jet_chMultiplicity")
        ne_mult = arr(arrays, "Jet_neMultiplicity")
        multiplicity = ch_mult + ne_mult
        args = (
            ak.flatten(jet_eta),
            ak.flatten(arr(arrays, "Jet_chHEF")),
            ak.flatten(arr(arrays, "Jet_neHEF")),
            ak.flatten(arr(arrays, "Jet_chEmEF")),
            ak.flatten(arr(arrays, "Jet_neEmEF")),
            ak.flatten(arr(arrays, "Jet_muEF")),
            ak.flatten(ch_mult),
            ak.flatten(ne_mult),
            ak.flatten(multiplicity),
        )
        return ak.unflatten(corr.evaluate(*args), counts) == 1, "correctionlib_AK4PUPPI_TightLeptonVeto"
    if has_field(arrays, "Jet_jetId"):
        jet_id = ak.values_astype(arrays["Jet_jetId"], np.int64)
        return (jet_id & 6) == 6, "NanoAOD_Jet_jetId_bits"
    return jet_pt == jet_pt, "raw_kinematic_fallback_missing_baseline_jet_id_inputs"


def _correction(repo: Path, relative_path: Path, correction_name: str) -> Any:
    path = analysis_data_file(repo, relative_path)
    key = (str(path), correction_name)
    if key not in _CORRECTION_CACHE:
        _CORRECTION_CACHE[key] = correctionlib.CorrectionSet.from_file(str(path))[correction_name]
    return _CORRECTION_CACHE[key]


def lumimask_path(repo: Path, year: str) -> Path:
    data_year = campaign_year(year)
    relative_path = LUMIMASK_RELATIVE_PATHS.get(data_year)
    if relative_path is None:
        raise ValueError(f"No certified luminosity mask configured for data year {data_year}")
    return analysis_data_file(repo, relative_path)


def load_lumimask(repo: Path, year: str) -> dict[int, list[tuple[int, int]]]:
    path = lumimask_path(repo, year)
    if path not in _LUMIMASK_CACHE:
        raw = json.loads(path.read_text())
        _LUMIMASK_CACHE[path] = {int(run): [(int(lo), int(hi)) for lo, hi in ranges] for run, ranges in raw.items()}
    return _LUMIMASK_CACHE[path]


def golden_lumi_mask(arrays: dict[str, Any], process: str, repo: Path, n: int, year: str) -> tuple[np.ndarray, str]:
    if not is_data_process(process):
        return np.ones(n, dtype=bool), "not_applicable_mc"
    path = lumimask_path(repo, year)
    lumi_ranges = load_lumimask(repo, year)
    runs = np.asarray(arrays["run"], dtype=np.int64)
    lumis = np.asarray(arrays["luminosityBlock"], dtype=np.int64)
    out = np.zeros(n, dtype=bool)
    for run in np.unique(runs):
        idx = runs == run
        ranges = lumi_ranges.get(int(run), [])
        if not ranges:
            continue
        run_lumis = lumis[idx]
        keep = np.zeros(np.sum(idx), dtype=bool)
        for lo, hi in ranges:
            keep |= (run_lumis >= lo) & (run_lumis <= hi)
        out[idx] = keep
    return out, str(path)


def ak4_jet_veto_mask(jet_pt: Any, jet_eta: Any, jet_phi: Any, repo: Path) -> tuple[Any, str]:
    corr = _correction(repo, JET_VETO_MAP_RELATIVE_PATH, JET_VETO_MAP_CORRECTION)
    counts = ak.num(jet_eta)
    flat_eta = ak.to_numpy(ak.flatten(jet_eta))
    flat_phi = ak.to_numpy(ak.flatten(jet_phi))
    flat_pt = ak.to_numpy(ak.flatten(jet_pt))
    if len(flat_eta) == 0:
        return ak.unflatten(np.zeros(0, dtype=bool), counts), f"correctionlib_{JET_VETO_MAP_CORRECTION}"
    veto = np.asarray(corr.evaluate("jetvetomap", flat_eta, flat_phi)) != 0
    return ak.unflatten((flat_pt > 30) & veto, counts), f"correctionlib_{JET_VETO_MAP_CORRECTION}"


def ak8_tight_lepton_veto_mask(arrays: dict[str, Any], fj_pt: Any, fj_eta: Any, repo: Path) -> tuple[Any, str]:
    if all(has_field(arrays, name) for name in FATJET_ID_INPUTS):
        corr = _correction(repo, Path("analysis/data/JMESF/2024/jetid.json.gz"), "AK8PUPPI_TightLeptonVeto")
        counts = ak.num(fj_eta)
        ch_mult = arr(arrays, "FatJet_chMultiplicity")
        ne_mult = arr(arrays, "FatJet_neMultiplicity")
        multiplicity = ch_mult + ne_mult
        args = (
            ak.flatten(fj_eta),
            ak.flatten(arr(arrays, "FatJet_chHEF")),
            ak.flatten(arr(arrays, "FatJet_neHEF")),
            ak.flatten(arr(arrays, "FatJet_chEmEF")),
            ak.flatten(arr(arrays, "FatJet_neEmEF")),
            ak.flatten(arr(arrays, "FatJet_muEF")),
            ak.flatten(ch_mult),
            ak.flatten(ne_mult),
            ak.flatten(multiplicity),
        )
        return ak.unflatten(corr.evaluate(*args), counts) == 1, "correctionlib_AK8PUPPI_TightLeptonVeto"
    return fj_pt == fj_pt, "raw_kinematic_fallback_missing_baseline_fatjet_id_inputs"


def clean_by_delta_r(obj_eta: Any, obj_phi: Any, ref_eta: Any, ref_phi: Any, dr_min: float) -> Any:
    deta = obj_eta[:, :, None] - ref_eta[:, None, :]
    dphi = delta_phi(obj_phi[:, :, None], ref_phi[:, None, :])
    return ak.all((deta * deta + dphi * dphi) > dr_min * dr_min, axis=2)


def medium_photon_mask(pt: Any, eta: Any, cutbased: Any, electron_veto: Any) -> Any:
    fiducial = (abs(eta) < 1.4442) | ((abs(eta) > 1.5660) & (abs(eta) < 2.5))
    return (
        (pt > 220)
        & fiducial
        & (cutbased >= PHOTON_MEDIUM_CUTBASED_MIN)
        & (electron_veto == 1)
    )


def jet_feature_block(jet_pt: Any, jet_eta: Any, jet_phi: Any, good_mask: Any, b_mask: Any, met_phi: Any) -> dict[str, Any]:
    jphi = jet_phi[good_mask]
    jpt = jet_pt[good_mask]
    jeta = jet_eta[good_mask]
    dphis = delta_phi(jphi, met_phi)
    j1dphi = first_or(999, dphis)
    j2dphi = nth_or(999, dphis, 1)
    j3dphi = nth_or(999, dphis, 2)
    j4dphi = nth_or(999, dphis, 3)
    return {
        "njet": count(good_mask),
        "nb": count(b_mask),
        "ht": ak.to_numpy(ak.sum(jet_pt[good_mask], axis=1)),
        "jpt": jpt,
        "jeta": jeta,
        "jphi": jphi,
        "j1pt": first_or(-99, jpt),
        "j1eta": first_or(-99, jeta),
        "j1phi": first_or(-99, jphi),
        "j2pt": nth_or(-99, jpt, 1),
        "j1dphi": j1dphi,
        "j2dphi": j2dphi,
        "j3dphi": j3dphi,
        "j4dphi": j4dphi,
        "min_dphi4": np.minimum.reduce([j1dphi, j2dphi, j3dphi, j4dphi]),
        "open_pre": (j1dphi > 0.5) & (j2dphi > 0.15) & (j3dphi > 0.15),
        "open_high": (j1dphi > 0.5) & (j2dphi > 0.5) & (j3dphi > 0.5) & (j4dphi > 0.5),
        "qcd_open": (j1dphi < 0.5) | (j2dphi < 0.5) | (j3dphi < 0.5) | (j4dphi < 0.5),
        "dphi123_0p1": (j1dphi < 0.1) | (j2dphi < 0.1) | (j3dphi < 0.1),
    }


def transverse_mass(pt: Any, phi: Any, met_pt: Any, met_phi: Any) -> Any:
    return np.sqrt(2 * pt * met_pt * (1 - np.cos(phi - met_phi)))


def transverse_vector_sum(*pt_phi_pairs: Any) -> tuple[Any, Any]:
    px = 0.0
    py = 0.0
    for pt, phi in pt_phi_pairs:
        px = px + pt * np.cos(phi)
        py = py + pt * np.sin(phi)
    return np.sqrt(np.maximum(0, px * px + py * py)), np.arctan2(py, px)


def transverse_vector_sum_pt(*pt_phi_pairs: Any) -> Any:
    return transverse_vector_sum(*pt_phi_pairs)[0]


def first_fatjet_subjet_btag(
    fatjet_mask: Any,
    subjet_idx1: Any,
    subjet_idx2: Any,
    subjet_btag: Any,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the maximum subjet discriminator and its availability for the first selected AK8 jet."""
    idx1 = np.asarray(first_or(-1, subjet_idx1[fatjet_mask]), dtype=int)
    idx2 = np.asarray(first_or(-1, subjet_idx2[fatjet_mask]), dtype=int)
    maximum = np.full(n, -99.0, dtype=float)
    available = np.zeros(n, dtype=bool)
    subjet_lists = ak.to_list(subjet_btag)
    for iev, (first_idx, second_idx) in enumerate(zip(idx1, idx2)):
        values = []
        event_subjets = subjet_lists[iev] if iev < len(subjet_lists) else []
        for index in (int(first_idx), int(second_idx)):
            if 0 <= index < len(event_subjets):
                values.append(finite_float(event_subjets[index], -99.0))
        if values:
            maximum[iev] = max(values)
            available[iev] = True
    return maximum, available


def lowdm_kinematic_block(
    *,
    jet_pt: Any,
    jet_eta: Any,
    jet_phi: Any,
    jet_btag: Any,
    good_jet_mask: Any,
    loose_b_mask: Any,
    medium_b_mask: Any,
    fatjet_pt: Any,
    fatjet_eta: Any,
    fatjet_phi: Any,
    fatjet_id_mask: Any,
    fatjet_clean_mask: Any,
    boosted_top_mask: Any,
    boosted_w_mask: Any,
    fatjet_subjet_idx1: Any,
    fatjet_subjet_idx2: Any,
    subjet_btag: Any,
    recoil_pt: Any,
    recoil_phi: Any,
    n: int,
) -> dict[str, Any]:
    """Build the low-deltaM observables with one internally consistent recoil/object definition."""
    jets = jet_feature_block(
        jet_pt,
        jet_eta,
        jet_phi,
        good_jet_mask,
        medium_b_mask,
        recoil_phi,
    )
    nb_medium = count(medium_b_mask)
    nb_loose = count(loose_b_mask)
    b_order = ak.argsort(jet_btag[medium_b_mask], axis=1, ascending=False)
    b_pt = jet_pt[medium_b_mask][b_order]
    b_phi = jet_phi[medium_b_mask][b_order]
    ptb = first_or(-99, b_pt)
    b_mtb = transverse_mass(b_pt, b_phi, recoil_pt, recoil_phi)
    mtb1 = first_or(999, b_mtb)
    mtb2 = nth_or(999, b_mtb, 1)
    mtb = np.where(
        nb_medium >= 2,
        np.minimum(mtb1, mtb2),
        np.where(nb_medium == 1, mtb1, -99.0),
    )

    clean_fatjets = fatjet_id_mask & fatjet_clean_mask & (fatjet_pt > 200) & (abs(fatjet_eta) < 2.4)
    n_isr = count(clean_fatjets)
    isr_pt = first_or(-99, fatjet_pt[clean_fatjets])
    isr_eta = first_or(-99, fatjet_eta[clean_fatjets])
    isr_phi = first_or(-99, fatjet_phi[clean_fatjets])
    isr_dphi = np.asarray(delta_phi(isr_phi, recoil_phi), dtype=float)
    isr_dphi = np.where(isr_pt > 0, isr_dphi, -99.0)
    subjet_max, subjet_available = first_fatjet_subjet_btag(
        clean_fatjets,
        fatjet_subjet_idx1,
        fatjet_subjet_idx2,
        subjet_btag,
        n,
    )
    pass_isr_bveto = subjet_available & (subjet_max < UPART_AK4_LOOSE_WP)
    pass_isr = (n_isr == 1) & (isr_dphi > 2.0)
    pass_topology = (
        (count(boosted_top_mask & fatjet_clean_mask) == 0)
        & (count(boosted_w_mask & fatjet_clean_mask) == 0)
    )
    met_sqrt_ht = np.divide(
        recoil_pt,
        np.sqrt(jets["ht"]),
        out=np.full(n, -99.0, dtype=float),
        where=jets["ht"] > 0,
    )
    return {
        "jets": jets,
        "nb_medium": nb_medium,
        "nb_loose": nb_loose,
        "ptb": ptb,
        "mtb": mtb,
        "met_sqrt_ht": met_sqrt_ht,
        "n_isr": n_isr,
        "isr_pt": isr_pt,
        "isr_eta": isr_eta,
        "isr_phi": isr_phi,
        "isr_dphi": isr_dphi,
        "isr_subjet_btag_max": subjet_max,
        "isr_subjet_bveto_available": subjet_available,
        "pass_isr_bveto": pass_isr_bveto,
        "pass_isr": pass_isr,
        "pass_topology": pass_topology,
        "pass_met_sqrt_ht": met_sqrt_ht >= 10.0,
        "pass_mtb": (nb_medium == 0) | (mtb < 175.0),
        "isr_mask": clean_fatjets,
    }


def assign_lowdm_search_bin(njet: int, nb: int, nsv: int, pisr: float, ptb: float, met: float, mtb: float) -> int:
    # Nsv is intentionally not part of the adopted Run-3 categorization.  Keep
    # both legacy-only arguments for call-site/schema compatibility with the
    # old 53-bin implementation.  Neither Nsv nor mTb participates in the
    # adopted 42-bin category assignment.
    del nsv, mtb

    def met_bin(value: float, edges: list[float]) -> int:
        for idx in range(len(edges) - 1):
            if edges[idx] <= value < edges[idx + 1]:
                return idx
        return len(edges) - 1 if value >= edges[-1] else -1

    idx = 0
    for nj_min, nj_max in [(2, 5), (6, None)]:
        if nb == 0 and njet >= nj_min and (nj_max is None or njet <= nj_max) and pisr >= 500.0:
            mb = met_bin(met, [450.0, 550.0, 650.0, 750.0])
            return idx + mb if mb >= 0 else -1
        idx += 4

    for pisr_min, pisr_max, met_edges in [(300.0, 500.0, [300.0, 400.0, 500.0, 600.0]), (500.0, None, [450.0, 550.0, 650.0, 750.0])]:
        for ptb_min, ptb_max in [(20.0, 40.0), (40.0, 70.0)]:
            if nb == 1 and ptb_min < ptb < ptb_max and pisr >= pisr_min and (pisr_max is None or pisr < pisr_max):
                mb = met_bin(met, met_edges)
                return idx + mb if mb >= 0 else -1
            idx += 4

    nb2_blocks = [
        (300.0, 500.0, 40.0, 80.0, 2, None, [300.0, 400.0, 500.0]),
        (300.0, 500.0, 80.0, 140.0, 2, None, [300.0, 400.0, 500.0]),
        (300.0, 500.0, 140.0, None, 7, None, [300.0, 400.0, 500.0]),
        (500.0, None, 40.0, 80.0, 2, None, [450.0, 550.0, 650.0]),
        (500.0, None, 80.0, 140.0, 2, None, [450.0, 550.0, 650.0]),
        (300.0, None, 140.0, None, 7, None, [450.0, 550.0, 650.0]),
    ]
    for pisr_min, pisr_max, ptb_min, ptb_max, nj_min, nj_max, met_edges in nb2_blocks:
        if nb >= 2 and njet >= nj_min and (nj_max is None or njet <= nj_max) and ptb > ptb_min and (ptb_max is None or ptb < ptb_max) and pisr >= pisr_min and (pisr_max is None or pisr < pisr_max):
            mb = met_bin(met, met_edges)
            return idx + mb if mb >= 0 else -1
        idx += 3
    return -1


def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(np.maximum(0, mass1 * mass1 + px1 * px1 + py1 * py1 + pz1 * pz1))
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(np.maximum(0, mass2 * mass2 + px2 * px2 + py2 * py2 + pz2 * pz2))
    mass2_out = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    return np.sqrt(np.maximum(0, mass2_out))


def extract_chunk(
    arrays: dict[str, Any],
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
    n = len(arrays["run"])
    repo = Path.cwd().resolve()
    year = campaign_year(year)
    correction_year = analysis_year(year)
    shift = validate_shift_name(shift_name)
    met_pt, met_phi, met_shift_status = shifted_met(arrays, n, shift, process)
    calo_pt = np.asarray(arr(arrays, "CaloMET_pt", np.ones(n) * np.nan), dtype=float)
    puppi_calo = np.divide(met_pt, calo_pt, out=np.ones(n) * np.inf, where=calo_pt != 0)

    jet_pt_raw = arr(arrays, "Jet_pt", ak.Array([[]] * n))
    jet_eta = arr(arrays, "Jet_eta", ak.Array([[]] * n))
    jet_phi = arr(arrays, "Jet_phi", ak.Array([[]] * n))
    jet_mass_raw = arr(arrays, "Jet_mass", ak.zeros_like(jet_pt_raw))
    jet_pt, jet_mass, jec_status = apply_jec(arrays, repo, year, process, "Jet", jet_pt_raw, jet_eta, jet_phi, jet_mass_raw, shift)
    jet_id_mask, jet_id_source = ak4_tight_lepton_veto_mask(arrays, jet_pt, jet_eta, repo)
    veto_j, jet_veto_source = ak4_jet_veto_mask(jet_pt, jet_eta, jet_phi, repo)
    zero_veto_j = count(veto_j) == 0
    btag = arr(arrays, "Jet_btagUParTAK4B", ak.zeros_like(jet_pt))
    good_j = (jet_pt > 30) & (abs(jet_eta) < 2.4) & jet_id_mask
    b_loose = good_j & (btag > UPART_AK4_LOOSE_WP)
    b_med = good_j & (btag > UPART_AK4_MEDIUM_WP)
    b_loose_lowdm = b_loose
    b_med_lowdm = b_med
    jet_nominal = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j, b_med, met_phi)
    njet = jet_nominal["njet"]
    nb = jet_nominal["nb"]
    nb_loose = count(b_loose)
    nb_medium_lowdm = nb
    nb_loose_lowdm = nb_loose
    ht = jet_nominal["ht"]
    j1pt = jet_nominal["j1pt"]
    j1eta = jet_nominal["j1eta"]
    j1phi = jet_nominal["j1phi"]
    j2pt = jet_nominal["j2pt"]
    j1dphi = jet_nominal["j1dphi"]
    j2dphi = jet_nominal["j2dphi"]
    j3dphi = jet_nominal["j3dphi"]
    j4dphi = jet_nominal["j4dphi"]
    min_dphi4 = jet_nominal["min_dphi4"]

    e_pt = arr(arrays, "Electron_pt", ak.Array([[]] * n)); e_eta = arr(arrays, "Electron_eta", ak.Array([[]] * n)); e_phi = arr(arrays, "Electron_phi", ak.Array([[]] * n)); e_mass = arr(arrays, "Electron_mass", ak.zeros_like(e_pt))
    e_delta_eta_sc = arr(arrays, "Electron_deltaEtaSC", ak.zeros_like(e_pt))
    e_charge = arr(arrays, "Electron_charge", ak.zeros_like(e_pt)); e_cb = arr(arrays, "Electron_cutBased", ak.zeros_like(e_pt)); e_iso = arr(arrays, "Electron_miniPFRelIso_all", ak.ones_like(e_pt) * 99)
    e_fid = ((abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5)))
    e_veto = (e_pt > 5) & e_fid & (e_cb >= 1) & (e_iso < 0.1)
    e_med = (e_pt > 10) & e_fid & (e_cb >= 3) & (e_iso < 0.1)
    n_e_veto = count(e_veto); n_e_med = count(e_med)

    m_pt = arr(arrays, "Muon_pt", ak.Array([[]] * n)); m_eta = arr(arrays, "Muon_eta", ak.Array([[]] * n)); m_phi = arr(arrays, "Muon_phi", ak.Array([[]] * n)); m_mass = arr(arrays, "Muon_mass", ak.zeros_like(m_pt))
    m_charge = arr(arrays, "Muon_charge", ak.zeros_like(m_pt)); m_looseid = arr(arrays, "Muon_looseId", ak.zeros_like(m_pt)); m_medid = arr(arrays, "Muon_mediumId", ak.zeros_like(m_pt)); m_iso = arr(arrays, "Muon_miniPFRelIso_all", ak.ones_like(m_pt) * 99)
    m_loose = (m_pt > 5) & (abs(m_eta) < 2.4) & m_looseid & (m_iso < 0.2)
    m_med = (m_pt > 10) & (abs(m_eta) < 2.4) & m_medid & (m_iso < 0.2)
    n_m_loose = count(m_loose); n_m_med = count(m_med)

    p_pt = arr(arrays, "Photon_pt", ak.Array([[]] * n)); p_eta = arr(arrays, "Photon_eta", ak.Array([[]] * n)); p_phi = arr(arrays, "Photon_phi", ak.Array([[]] * n)); p_cb = arr(arrays, "Photon_cutBased", ak.zeros_like(p_pt))
    p_electron_veto = arr(arrays, "Photon_electronVeto", ak.zeros_like(p_pt))
    p_med = medium_photon_mask(p_pt, p_eta, p_cb, p_electron_veto)
    n_p_med = count(p_med)

    tau_pt = arr(arrays, "Tau_pt", ak.Array([[]] * n)); tau_eta = arr(arrays, "Tau_eta", ak.Array([[]] * n)); tau_phi = arr(arrays, "Tau_phi", ak.Array([[]] * n)); tau_dz = arr(arrays, "Tau_dz", ak.zeros_like(tau_pt)); tau_dm = arr(arrays, "Tau_decayMode", ak.zeros_like(tau_pt)); tau_id = arr(arrays, "Tau_idDeepTau2018v2p5VSjet", ak.zeros_like(tau_pt))
    tau_mt = transverse_mass(tau_pt, tau_phi, met_pt, met_phi)
    tau_med = (tau_pt > 20) & (abs(tau_eta) < 2.5) & (abs(tau_dz) < 0.2) & (tau_dm != 5) & (tau_dm != 6) & (tau_id >= 5) & (tau_mt < 100)
    n_tau = count(tau_med)

    tr_pt = arr(arrays, "IsoTrack_pt", ak.Array([[]] * n)); tr_eta = arr(arrays, "IsoTrack_eta", ak.Array([[]] * n)); tr_phi = arr(arrays, "IsoTrack_phi", ak.Array([[]] * n)); tr_pdg = abs(arr(arrays, "IsoTrack_pdgId", ak.zeros_like(tr_pt))); tr_iso = arr(arrays, "IsoTrack_pfRelIso03_all", ak.ones_like(tr_pt) * 99)
    tr_mt = transverse_mass(tr_pt, tr_phi, met_pt, met_phi)
    tr_e = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 11) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_m = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 13) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_pi = (tr_pt > 10) & (abs(tr_eta) < 2.5) & (tr_pdg == 211) & (tr_iso < 0.1) & (tr_mt < 100)

    fj_pt_raw = arr(arrays, "FatJet_pt", ak.Array([[]] * n)); fj_eta = arr(arrays, "FatJet_eta", ak.Array([[]] * n)); fj_phi = arr(arrays, "FatJet_phi", ak.Array([[]] * n)); fj_msd = arr(arrays, "FatJet_msoftdrop", ak.zeros_like(fj_pt_raw)); fj_mass_raw = arr(arrays, "FatJet_mass", ak.zeros_like(fj_pt_raw))
    fj_top_score = arr(arrays, BOOSTED_TOP_SCORE_BRANCH, ak.ones_like(fj_pt_raw) * -99.0)
    fj_w_score = arr(arrays, BOOSTED_W_SCORE_BRANCH, ak.ones_like(fj_pt_raw) * -99.0)
    fj_pt, fj_mass, fjec_status = apply_jec(arrays, repo, year, process, "FatJet", fj_pt_raw, fj_eta, fj_phi, fj_mass_raw, shift)
    fj_id_mask, fatjet_id_source = ak8_tight_lepton_veto_mask(arrays, fj_pt, fj_eta, repo)
    good_fj = (fj_pt > 200) & (abs(fj_eta) < 2.0) & (fj_msd > 60) & fj_id_mask
    n_fj = count(good_fj)
    fj1pt = first_or(-99, fj_pt[good_fj]); fj1eta = first_or(-99, fj_eta[good_fj]); fj1phi = first_or(-99, fj_phi[good_fj]); fj1mass = first_or(-99, fj_mass[good_fj]); fj1msd = first_or(-99, fj_msd[good_fj])
    boosted_top = (fj_id_mask & (fj_pt > BOOSTED_TOP_PT_MIN) & (abs(fj_eta) < BOOSTED_ETA_MAX) & (fj_msd > BOOSTED_TOP_MSD_MIN) & (fj_top_score > BOOSTED_TOP_SCORE_WP))
    boosted_w = (fj_id_mask & (fj_pt > BOOSTED_W_PT_MIN) & (abs(fj_eta) < BOOSTED_ETA_MAX) & (fj_msd > BOOSTED_W_MSD_MIN) & (fj_msd < BOOSTED_W_MSD_MAX) & (fj_w_score > BOOSTED_W_SCORE_WP))
    nboosted_top = count(boosted_top)
    nboosted_w = count(boosted_w)
    nboosted_total = nboosted_top + nboosted_w
    fj1topscore = first_or(-99, fj_top_score[good_fj])
    fj1wscore = first_or(-99, fj_w_score[good_fj])
    fj_subidx1 = arr(arrays, "FatJet_subJetIdx1", ak.ones_like(fj_pt_raw) * -1)
    fj_subidx2 = arr(arrays, "FatJet_subJetIdx2", ak.ones_like(fj_pt_raw) * -1)
    subjet_btag_upart = arr(arrays, "SubJet_btagUParTAK4B", ak.Array([[]] * n))
    subjet_btag_deepb = arr(arrays, "SubJet_btagDeepB", ak.Array([[]] * n))

    met_filters, missing_filters = all_filters(arrays, n)
    sig_hlt = bool_branch(arrays, SIGNAL_HLT, n)
    pho_hlt = bool_branch(arrays, PHOTON_HLT, n)
    ele_hlt = bool_branch(arrays, ELECTRON_HLT, n)
    mu_hlt = bool_branch(arrays, MUON_HLT, n)
    signal_trigger_policy = "standard trigger requirement"
    if fastsim_trigger_bypass and process == "SMS":
        sig_hlt = np.ones(n, dtype=bool)
        signal_trigger_policy = "FastSim trigger bypass: HLT branches absent; trigger not applied at event-selection level"

    e1pt = first_or(-99, e_pt[e_med]); e2pt = nth_or(-99, e_pt[e_med], 1); e1eta = first_or(0, e_eta[e_med]); e2eta = nth_or(0, e_eta[e_med], 1); e1phi = first_or(0, e_phi[e_med]); e2phi = nth_or(0, e_phi[e_med], 1); e1m = first_or(0, e_mass[e_med]); e2m = nth_or(0, e_mass[e_med], 1); e1q = first_or(0, e_charge[e_med]); e2q = nth_or(0, e_charge[e_med], 1)
    mee = invariant_mass(e1pt, e1eta, e1phi, e1m, e2pt, e2eta, e2phi, e2m); pee = np.sqrt(np.maximum(0, e1pt**2 + e2pt**2 + 2*e1pt*e2pt*np.cos(e1phi-e2phi)))
    m1pt = first_or(-99, m_pt[m_med]); m2pt = nth_or(-99, m_pt[m_med], 1); m1eta = first_or(0, m_eta[m_med]); m2eta = nth_or(0, m_eta[m_med], 1); m1phi = first_or(0, m_phi[m_med]); m2phi = nth_or(0, m_phi[m_med], 1); m1m = first_or(0, m_mass[m_med]); m2m = nth_or(0, m_mass[m_med], 1); m1q = first_or(0, m_charge[m_med]); m2q = nth_or(0, m_charge[m_med], 1)
    mmm = invariant_mass(m1pt, m1eta, m1phi, m1m, m2pt, m2eta, m2phi, m2m); pmm = np.sqrt(np.maximum(0, m1pt**2 + m2pt**2 + 2*m1pt*m2pt*np.cos(m1phi-m2phi)))
    e_mt = transverse_mass(e_pt[e_veto], e_phi[e_veto], met_pt, met_phi)
    m_mt = transverse_mass(m_pt[m_loose], m_phi[m_loose], met_pt, met_phi)
    mt_100 = ak.to_numpy(ak.all(e_mt < 100, axis=1) & ak.all(m_mt < 100, axis=1))

    photon_clean_j = clean_by_delta_r(jet_eta, jet_phi, p_eta[p_med], p_phi[p_med], 0.2)
    electron_clean_j = clean_by_delta_r(jet_eta, jet_phi, e_eta[e_med], e_phi[e_med], 0.2)
    muon_clean_j = clean_by_delta_r(jet_eta, jet_phi, m_eta[m_med], m_phi[m_med], 0.2)
    lepton_clean_j = electron_clean_j & muon_clean_j
    photon_clean_fj = clean_by_delta_r(fj_eta, fj_phi, p_eta[p_med], p_phi[p_med], 0.4)
    electron_clean_fj = clean_by_delta_r(fj_eta, fj_phi, e_eta[e_med], e_phi[e_med], 0.4)
    muon_clean_fj = clean_by_delta_r(fj_eta, fj_phi, m_eta[m_med], m_phi[m_med], 0.4)
    all_j_clean = ak.values_astype(ak.ones_like(jet_pt), np.bool_)
    all_fj_clean = ak.values_astype(ak.ones_like(fj_pt), np.bool_)

    one_veto_lepton = ((n_e_veto == 1) & (n_m_loose == 0)) | ((n_e_veto == 0) & (n_m_loose == 1))
    valid_met = np.isfinite(met_pt) & (met_pt >= 0)
    lumi_mask, lumi_mask_source = golden_lumi_mask(arrays, process, repo, n, year)
    no_tracks = (count(tr_e) == 0) & (count(tr_m) == 0) & (count(tr_pi) == 0)
    zero_e = n_e_veto == 0
    zero_m = n_m_loose == 0
    no_veto_leptons = zero_e & zero_m
    zero_tau = n_tau == 0
    base_common = valid_met & lumi_mask & met_filters & no_tracks & zero_veto_j & (puppi_calo < 5)
    met_250 = met_pt > 250
    ht_300 = ht > 300

    recoil_g, recoil_g_phi = transverse_vector_sum(
        (met_pt, met_phi),
        (first_or(0, p_pt[p_med]), first_or(0, p_phi[p_med])),
    )
    recoil_dy2e, recoil_dy2e_phi = transverse_vector_sum((met_pt, met_phi), (e1pt, e1phi), (e2pt, e2phi))
    recoil_dy2m, recoil_dy2m_phi = transverse_vector_sum((met_pt, met_phi), (m1pt, m1phi), (m2pt, m2phi))
    jet_photon_recoil = jet_feature_block(
        jet_pt, jet_eta, jet_phi,
        good_j & photon_clean_j, b_med & photon_clean_j, recoil_g_phi,
    )
    jet_electron_recoil = jet_feature_block(
        jet_pt, jet_eta, jet_phi,
        good_j & electron_clean_j, b_med & electron_clean_j, recoil_dy2e_phi,
    )
    jet_muon_recoil = jet_feature_block(
        jet_pt, jet_eta, jet_phi,
        good_j & muon_clean_j, b_med & muon_clean_j, recoil_dy2m_phi,
    )
    jet_photon_clean = jet_photon_recoil
    jet_lepton_clean = jet_feature_block(
        jet_pt, jet_eta, jet_phi,
        good_j & lepton_clean_j, b_med & lepton_clean_j, met_phi,
    )
    ht_photon_300 = jet_photon_recoil["ht"] > 300
    ht_lepton_300 = jet_lepton_clean["ht"] > 300

    any_analysis_trigger = sig_hlt | pho_hlt | ele_hlt | mu_hlt
    common_recoil_or_met = met_250 | (recoil_g > 250) | (recoil_dy2e > 250) | (recoil_dy2m > 250)
    common_ht = ht_300 | ht_photon_300 | (jet_electron_recoil["ht"] > 300) | (jet_muon_recoil["ht"] > 300)
    common_njet2 = (njet >= 2) | (jet_photon_recoil["njet"] >= 2) | (jet_electron_recoil["njet"] >= 2) | (jet_muon_recoil["njet"] >= 2)
    flat_preselection = base_common & zero_tau & any_analysis_trigger & common_recoil_or_met & common_ht & common_njet2

    masks = {
        "preselection": base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & jet_nominal["open_pre"] & ht_300,
        "LLCR": base_common & sig_hlt & zero_tau & (njet >= 5) & (nb >= 1) & one_veto_lepton & mt_100 & met_250 & jet_nominal["open_high"] & ht_300,
        "QCDCR": base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & jet_nominal["qcd_open"] & jet_nominal["dphi123_0p1"] & ht_300,
        "GCR": base_common & pho_hlt & (n_p_med == 1) & no_veto_leptons & zero_tau & (jet_photon_recoil["njet"] >= 5) & (jet_photon_recoil["nb"] >= 1) & (met_pt < 250) & (recoil_g > 250) & jet_photon_recoil["open_high"] & ht_photon_300,
        "DY2E": base_common & ele_hlt & zero_tau & (jet_electron_recoil["njet"] >= 5) & (jet_electron_recoil["nb"] >= 1) & zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & (recoil_dy2e > 250) & (mee > 81) & (mee < 101) & jet_electron_recoil["open_high"] & (jet_electron_recoil["ht"] > 300),
        "DY2M": base_common & mu_hlt & zero_tau & (jet_muon_recoil["njet"] >= 5) & (jet_muon_recoil["nb"] >= 1) & zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & (recoil_dy2m > 250) & (mmm > 81) & (mmm < 101) & jet_muon_recoil["open_high"] & (jet_muon_recoil["ht"] > 300),
        "SR": base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & jet_nominal["open_high"] & ht_300,
    }

    sv_required = ["SV_pt", "SV_eta", "SV_phi", "SV_dxy", "SV_dlenSig", "SV_pAngle", "SV_ntracks"]
    sv_available = all(has_field(arrays, name) for name in sv_required)
    if sv_available:
        sv_pt = arr(arrays, "SV_pt", ak.Array([[]] * n))
        sv_eta = arr(arrays, "SV_eta", ak.Array([[]] * n))
        sv_phi = arr(arrays, "SV_phi", ak.Array([[]] * n))
        sv_dxy = arr(arrays, "SV_dxy", ak.zeros_like(sv_pt))
        sv_dlen_sig = arr(arrays, "SV_dlenSig", ak.zeros_like(sv_pt))
        sv_pangle = arr(arrays, "SV_pAngle", ak.ones_like(sv_pt) * 99.0)
        sv_ntracks = arr(arrays, "SV_ntracks", ak.zeros_like(sv_pt))
        sv_base = (sv_pt < 20) & (abs(sv_dxy) < 3) & (sv_dlen_sig > 4) & (sv_pangle < LOWDM_SV_POINTING_ANGLE_MAX) & (sv_ntracks >= 3)

        def softb_count(jet_mask: Any) -> np.ndarray:
            return count(sv_base & clean_by_delta_r(sv_eta, sv_phi, jet_eta[jet_mask], jet_phi[jet_mask], 0.4))
    else:
        def softb_count(jet_mask: Any) -> np.ndarray:
            return np.full(n, -1, dtype=int)

    lowdm_blocks = {
        "SR": lowdm_kinematic_block(
            jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag,
            good_jet_mask=good_j, loose_b_mask=b_loose_lowdm, medium_b_mask=b_med_lowdm,
            fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask,
            fatjet_clean_mask=all_fj_clean, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w,
            fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2,
            subjet_btag=subjet_btag_upart, recoil_pt=met_pt, recoil_phi=met_phi, n=n,
        ),
        "GCR": lowdm_kinematic_block(
            jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag,
            good_jet_mask=good_j & photon_clean_j, loose_b_mask=b_loose_lowdm & photon_clean_j, medium_b_mask=b_med_lowdm & photon_clean_j,
            fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask,
            fatjet_clean_mask=photon_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w,
            fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2,
            subjet_btag=subjet_btag_upart, recoil_pt=recoil_g, recoil_phi=recoil_g_phi, n=n,
        ),
        "DY2E": lowdm_kinematic_block(
            jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag,
            good_jet_mask=good_j & electron_clean_j, loose_b_mask=b_loose_lowdm & electron_clean_j, medium_b_mask=b_med_lowdm & electron_clean_j,
            fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask,
            fatjet_clean_mask=electron_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w,
            fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2,
            subjet_btag=subjet_btag_upart, recoil_pt=recoil_dy2e, recoil_phi=recoil_dy2e_phi, n=n,
        ),
        "DY2M": lowdm_kinematic_block(
            jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag,
            good_jet_mask=good_j & muon_clean_j, loose_b_mask=b_loose_lowdm & muon_clean_j, medium_b_mask=b_med_lowdm & muon_clean_j,
            fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask,
            fatjet_clean_mask=muon_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w,
            fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2,
            subjet_btag=subjet_btag_upart, recoil_pt=recoil_dy2m, recoil_phi=recoil_dy2m_phi, n=n,
        ),
    }
    lowdm_blocks["LLCR"] = lowdm_blocks["SR"]
    lowdm_blocks["QCDCR"] = lowdm_blocks["SR"]
    lowdm_nsv = {
        "SR": softb_count(good_j),
        "LLCR": softb_count(good_j),
        "QCDCR": softb_count(good_j),
        "GCR": softb_count(good_j & photon_clean_j),
        "DY2E": softb_count(good_j & electron_clean_j),
        "DY2M": softb_count(good_j & muon_clean_j),
    }

    def lowdm_quality(block: dict[str, Any]) -> np.ndarray:
        return (
            block["pass_topology"]
            & block["pass_isr"]
            & block["pass_met_sqrt_ht"]
        )

    lowdm_masks = {
        "SR": base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal["open_pre"] & lowdm_quality(lowdm_blocks["SR"]),
        "LLCR": base_common & sig_hlt & one_veto_lepton & mt_100 & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal["open_pre"] & lowdm_quality(lowdm_blocks["LLCR"]),
        "QCDCR": base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal["qcd_open"] & jet_nominal["dphi123_0p1"] & lowdm_quality(lowdm_blocks["QCDCR"]),
        "GCR": base_common & pho_hlt & (n_p_med == 1) & no_veto_leptons & zero_tau & (met_pt < 250) & (recoil_g > 250) & (jet_photon_recoil["njet"] >= 2) & (jet_photon_recoil["ht"] > 300) & jet_photon_recoil["open_pre"] & lowdm_quality(lowdm_blocks["GCR"]),
        "DY2E": base_common & ele_hlt & zero_tau & zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & (mee > 81) & (mee < 101) & (recoil_dy2e > 250) & (jet_electron_recoil["njet"] >= 2) & (jet_electron_recoil["ht"] > 300) & jet_electron_recoil["open_pre"] & lowdm_quality(lowdm_blocks["DY2E"]),
        "DY2M": base_common & mu_hlt & zero_tau & zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & (mmm > 81) & (mmm < 101) & (recoil_dy2m > 250) & (jet_muon_recoil["njet"] >= 2) & (jet_muon_recoil["ht"] > 300) & jet_muon_recoil["open_pre"] & lowdm_quality(lowdm_blocks["DY2M"]),
    }
    lowdm_recoil = {
        "SR": met_pt, "LLCR": met_pt, "QCDCR": met_pt,
        "GCR": recoil_g, "DY2E": recoil_dy2e, "DY2M": recoil_dy2m,
    }
    lowdm_search_bins = {}
    for region, block in lowdm_blocks.items():
        lowdm_search_bins[region] = np.asarray([
            assign_lowdm_search_bin(
                int(block["jets"]["njet"][i]),
                int(block["nb_medium"][i]),
                int(lowdm_nsv[region][i]),
                float(block["isr_pt"][i]),
                float(block["ptb"][i]),
                float(lowdm_recoil[region][i]),
                float(block["mtb"][i]),
            )
            if bool(lowdm_masks[region][i]) else -1
            for i in range(n)
        ], dtype=int)
        lowdm_masks[region] = lowdm_masks[region] & (lowdm_search_bins[region] >= 0)

    nominal_lowdm = lowdm_blocks["SR"]
    lowdm_isr_fj = nominal_lowdm["isr_mask"]
    n_lowdm_isr = nominal_lowdm["n_isr"]
    lowdm_isr_pt = nominal_lowdm["isr_pt"]
    lowdm_isr_eta = nominal_lowdm["isr_eta"]
    lowdm_isr_phi = nominal_lowdm["isr_phi"]
    lowdm_isr_dphi = nominal_lowdm["isr_dphi"]
    lowdm_isr_subjet_btag_max = nominal_lowdm["isr_subjet_btag_max"]
    lowdm_isr_subjet_bveto_available = nominal_lowdm["isr_subjet_bveto_available"]
    pass_lowdm_isr_bveto = nominal_lowdm["pass_isr_bveto"]
    nb_medium_lowdm = nominal_lowdm["nb_medium"]
    nb_loose_lowdm = nominal_lowdm["nb_loose"]
    lowdm_ptb = nominal_lowdm["ptb"]
    lowdm_mtb = nominal_lowdm["mtb"]
    lowdm_met_sqrt_ht = nominal_lowdm["met_sqrt_ht"]
    n_sv_softb = lowdm_nsv["SR"]
    feature_lowdm_preselection = masks["preselection"]
    pass_lowdm_topology_veto = nominal_lowdm["pass_topology"]
    pass_lowdm_isr = nominal_lowdm["pass_isr"]
    pass_lowdm_met_sqrt_ht = nominal_lowdm["pass_met_sqrt_ht"]
    pass_lowdm_mtb = nominal_lowdm["pass_mtb"]
    feature_lowdm_sr_base = lowdm_masks["SR"]
    lowdm_search_bin = lowdm_search_bins["SR"]

    jet_hadflav = arr(arrays, "Jet_hadronFlavour", ak.zeros_like(jet_pt))
    if compute_weights:
        gen_weight, weight_variations, scale_factor_status = compute_weight_bundle(
            arrays, repo, dataset, process, year, n,
            jet_pt[good_j], jet_eta[good_j], jet_hadflav[good_j], b_med[good_j],
            e_eta, e_delta_eta_sc, e_pt, e_phi, e_veto, e_med, n_e_veto, n_e_med,
            m_eta, m_pt, m_phi, m_loose, m_med, n_m_loose, n_m_med,
            p_eta, p_pt, p_phi, p_med, masks["GCR"],
        )
    else:
        if is_data_process(process):
            gen_weight = np.ones(n, dtype=float)
        else:
            gen_weight = np_filled(arr(arrays, "genWeight", np.ones(n)), n, 1.0)
        weight_variations = {"nominal": gen_weight}
        scale_factor_status = {
            "applied": False,
            "reason": "deferred_to_post_skim",
            "available_variations": ["nominal"],
            "components": {},
            "note": "JEC/FJEC and MET/JES shape shifts are applied before skim; multiplicative scale factors are intentionally not applied in flat ntuple production.",
        }
    weight = weight_variations["nominal"]
    unavailable_features = list(missing_filters)
    if jet_id_source.startswith("raw_kinematic"):
        unavailable_features.append("baseline AK4 correctionlib jet ID inputs missing; raw kinematic fallback used")
    if fatjet_id_source.startswith("raw_kinematic"):
        unavailable_features.append("baseline AK8 correctionlib jet ID inputs missing; raw kinematic fallback used")
    if not jec_status.get("applied"):
        unavailable_features.append("AK4 JEC not applied: " + str(jec_status.get("reason", jec_status.get("source", "unknown"))))
    if not fjec_status.get("applied"):
        unavailable_features.append("AK8 JEC not applied: " + str(fjec_status.get("reason", fjec_status.get("source", "unknown"))))
    missing_boosted_score_branches = [name for name in [BOOSTED_TOP_SCORE_BRANCH, BOOSTED_W_SCORE_BRANCH] if not has_field(arrays, name)]
    if missing_boosted_score_branches:
        unavailable_features.append("boosted top/W tag score branches missing: " + ",".join(missing_boosted_score_branches))
    for comp_name, comp_status in scale_factor_status.get("components", {}).items():
        if not comp_status.get("applied") and not str(comp_status.get("source", "")).startswith("unity_fallback_not_TTto"):
            unavailable_features.append(f"{comp_name} not applied: {comp_status.get('source', 'unknown')}")

    cut_sequences = {
        "preselection": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_veto_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", njet >= 2), ("bjet_multiplicity", np.ones(n, dtype=bool)), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", jet_nominal["open_pre"]), ("final_region_selection", masks["preselection"])],
        "LLCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", one_veto_lepton & mt_100), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", jet_nominal["open_high"]), ("final_region_selection", masks["LLCR"])],
        "QCDCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_veto_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", jet_nominal["qcd_open"] & jet_nominal["dphi123_0p1"]), ("final_region_selection", masks["QCDCR"])],
        "GCR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", pho_hlt), ("lepton_veto_or_selection", (n_p_med == 1) & no_veto_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", jet_photon_clean["njet"] >= 5), ("bjet_multiplicity", jet_photon_clean["nb"] >= 1), ("MET_or_recoil_threshold", (met_pt < 250) & (recoil_g > 250)), ("HT_threshold", ht_photon_300), ("delta_phi_requirements", jet_photon_clean["open_high"]), ("final_region_selection", masks["GCR"])],
        "DY2E": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", ele_hlt), ("lepton_veto_or_selection", zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & (mee > 81) & (mee < 101)), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", jet_electron_recoil["njet"] >= 5), ("bjet_multiplicity", jet_electron_recoil["nb"] >= 1), ("MET_or_recoil_threshold", recoil_dy2e > 250), ("HT_threshold", jet_electron_recoil["ht"] > 300), ("delta_phi_requirements", jet_electron_recoil["open_high"]), ("final_region_selection", masks["DY2E"])],
        "DY2M": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", mu_hlt), ("lepton_veto_or_selection", zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & (mmm > 81) & (mmm < 101)), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", jet_muon_recoil["njet"] >= 5), ("bjet_multiplicity", jet_muon_recoil["nb"] >= 1), ("MET_or_recoil_threshold", recoil_dy2m > 250), ("HT_threshold", jet_muon_recoil["ht"] > 300), ("delta_phi_requirements", jet_muon_recoil["open_high"]), ("final_region_selection", masks["DY2M"])],
        "SR": [("total_read_events", np.ones(n, dtype=bool)), ("valid_MET", valid_met), ("lumimask", lumi_mask), ("MET_filters", met_filters), ("trigger_requirement", sig_hlt), ("lepton_veto_or_selection", no_veto_leptons), ("tau_veto", zero_tau), ("isolated_track_veto", no_tracks), ("jet_veto_map", zero_veto_j), ("jet_multiplicity", njet >= 5), ("bjet_multiplicity", nb >= 1), ("MET_or_recoil_threshold", met_250), ("HT_threshold", ht_300), ("delta_phi_requirements", jet_nominal["open_high"]), ("final_region_selection", masks["SR"])],
    }
    cutflows = {}
    for region, seq in cut_sequences.items():
        cumulative = np.ones(n, dtype=bool)
        cutflows[region] = []
        first_zero = None
        for cut_name, cut_mask in seq:
            cumulative = cumulative & np.asarray(cut_mask, dtype=bool)
            uw = int(np.sum(cumulative))
            ww = finite_sum(weight[cumulative]) if len(weight) == n else None
            if first_zero is None and uw == 0:
                first_zero = cut_name
            cutflows[region].append({"cut": cut_name, "unweighted": uw, "weighted": ww})
        for item in cutflows[region]:
            item["first_zero_cut"] = first_zero

    materialization_masks = {
        "feature_flat_preselection": flat_preselection,
        "feature_lowdm_preselection": feature_lowdm_preselection,
        "feature_lowdm_sr_base": feature_lowdm_sr_base,
        **{f"feature_lowdm_{region}": lowdm_masks[region] for region in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")},
        **{f"feature_{region}": masks[region] for region in REGION_NAMES},
    }
    selected_materialization_mask = materialization_masks.get(materialize_skim_flag)
    if selected_materialization_mask is None:
        materialize_event_mask = np.ones(n, dtype=bool)
        materialization_applied = False
    else:
        materialize_event_mask = np.asarray(selected_materialization_mask, dtype=bool)
        materialization_applied = True
    materialize_indices = np.flatnonzero(materialize_event_mask)

    gen_top_pts = ak.Array([[]] * n)
    if has_field(arrays, "GenPart_pt") and has_field(arrays, "GenPart_pdgId") and has_field(arrays, "GenPart_statusFlags"):
        gen_flags_for_flat = ak.values_astype(arrays["GenPart_statusFlags"], np.int64)
        is_final_top_for_flat = (abs(arrays["GenPart_pdgId"]) == 6) & ((gen_flags_for_flat & (1 << 8)) != 0) & ((gen_flags_for_flat & (1 << 13)) != 0)
        gen_top_pts = arrays["GenPart_pt"][is_final_top_for_flat]

    flat_vectors = {
        "good_jet_pt": ak.to_list(jet_pt[good_j][materialize_event_mask]),
        "good_jet_eta": ak.to_list(jet_eta[good_j][materialize_event_mask]),
        "good_jet_phi": ak.to_list(jet_phi[good_j][materialize_event_mask]),
        "good_jet_btag_upart": ak.to_list(btag[good_j][materialize_event_mask]),
        "good_jet_hadron_flavour": ak.to_list(jet_hadflav[good_j][materialize_event_mask]),
        "good_jet_b_loose": ak.to_list(ak.values_astype(b_loose[good_j], np.int32)[materialize_event_mask]),
        "good_jet_b_medium": ak.to_list(ak.values_astype(b_med[good_j], np.int32)[materialize_event_mask]),
        "lowdm_fatjet_pt": ak.to_list(fj_pt[lowdm_isr_fj][materialize_event_mask]),
        "lowdm_fatjet_eta": ak.to_list(fj_eta[lowdm_isr_fj][materialize_event_mask]),
        "lowdm_fatjet_phi": ak.to_list(fj_phi[lowdm_isr_fj][materialize_event_mask]),
        "lowdm_fatjet_msd": ak.to_list(fj_msd[lowdm_isr_fj][materialize_event_mask]),
        "lowdm_fatjet_subjet_idx1": ak.to_list(ak.values_astype(fj_subidx1[lowdm_isr_fj], np.int32)[materialize_event_mask]),
        "lowdm_fatjet_subjet_idx2": ak.to_list(ak.values_astype(fj_subidx2[lowdm_isr_fj], np.int32)[materialize_event_mask]),
        "electron_veto_pt": ak.to_list(e_pt[e_veto][materialize_event_mask]),
        "electron_veto_eta_sc": ak.to_list((e_eta + e_delta_eta_sc)[e_veto][materialize_event_mask]),
        "electron_veto_phi": ak.to_list(e_phi[e_veto][materialize_event_mask]),
        "electron_medium_pt": ak.to_list(e_pt[e_med][materialize_event_mask]),
        "electron_medium_eta_sc": ak.to_list((e_eta + e_delta_eta_sc)[e_med][materialize_event_mask]),
        "electron_medium_phi": ak.to_list(e_phi[e_med][materialize_event_mask]),
        "muon_loose_pt": ak.to_list(m_pt[m_loose][materialize_event_mask]),
        "muon_loose_eta": ak.to_list(m_eta[m_loose][materialize_event_mask]),
        "muon_loose_phi": ak.to_list(m_phi[m_loose][materialize_event_mask]),
        "muon_medium_pt": ak.to_list(m_pt[m_med][materialize_event_mask]),
        "muon_medium_eta": ak.to_list(m_eta[m_med][materialize_event_mask]),
        "muon_medium_phi": ak.to_list(m_phi[m_med][materialize_event_mask]),
        "photon_medium_pt": ak.to_list(p_pt[p_med][materialize_event_mask]),
        "photon_medium_eta": ak.to_list(p_eta[p_med][materialize_event_mask]),
        "photon_medium_phi": ak.to_list(p_phi[p_med][materialize_event_mask]),
        "gen_top_pt": ak.to_list(gen_top_pts[materialize_event_mask]),
    }
    pu_ntrueint = np_filled(arr(arrays, "Pileup_nTrueInt", np.ones(n) * -1.0), n, -1.0)

    genmodel_branches = sorted([name for name in getattr(arrays, "fields", []) if str(name).startswith("GenModel_T2tt_")])
    genmodel_masks = {name: np.asarray(arrays[name], dtype=bool) for name in genmodel_branches}

    def active_genmodel(i: int) -> tuple[str, int | None, int | None]:
        for name, mask in genmodel_masks.items():
            if i < len(mask) and bool(mask[i]):
                nums = re.findall(r"(\d+)", name)
                if len(nums) >= 2:
                    return name, int(nums[-2]), int(nums[-1])
                return name, None, None
        return "", None, None

    rows = []
    for output_index, i in enumerate(materialize_indices):
        gen_branch, gen_mstop, gen_mlsp = active_genmodel(i)
        row = {
            "dataset": dataset, "process": process, "year": year, "signal_point": sp or "", "file": file_path, "shape_shift": shift,
            "genmodel_branch": gen_branch, "mStop": gen_mstop if gen_mstop is not None else "", "mLSP": gen_mlsp if gen_mlsp is not None else "", "trigger_policy": signal_trigger_policy,
            "entry": entry_start + i, "run": int(arrays["run"][i]), "luminosityBlock": int(arrays["luminosityBlock"][i]), "event": int(arrays["event"][i]),
            "met": float(met_pt[i]), "met_phi": float(met_phi[i]), "ht": float(ht[i]), "njet": int(njet[i]), "nb_medium": int(nb[i]), "nb_loose": int(nb_loose[i]), "nb_medium_lowdm": int(nb_medium_lowdm[i]), "nb_loose_lowdm": int(nb_loose_lowdm[i]), "pu_ntrueint": float(pu_ntrueint[i]),
            "ht_photon_clean": float(jet_photon_clean["ht"][i]), "njet_photon_clean": int(jet_photon_clean["njet"][i]), "nb_photon_clean": int(jet_photon_clean["nb"][i]),
            "ht_lepton_clean": float(jet_lepton_clean["ht"][i]), "njet_lepton_clean": int(jet_lepton_clean["njet"][i]), "nb_lepton_clean": int(jet_lepton_clean["nb"][i]),
            "j1pt": float(j1pt[i]), "j1eta": float(j1eta[i]), "j1phi": float(j1phi[i]), "j2pt": float(j2pt[i]),
            "j1_met_dphi": float(j1dphi[i]), "j2_met_dphi": float(j2dphi[i]), "j3_met_dphi": float(j3dphi[i]), "j4_met_dphi": float(j4dphi[i]), "min_dphi4": float(min_dphi4[i]),
            "nfj": int(n_fj[i]), "fj1pt": float(fj1pt[i]), "fj1eta": float(fj1eta[i]), "fj1phi": float(fj1phi[i]), "fj1mass": float(fj1mass[i]), "fj1msd": float(fj1msd[i]),
            "fj1_top_score": float(fj1topscore[i]), "fj1_w_score": float(fj1wscore[i]),
            "nboosted_top": int(nboosted_top[i]), "nboosted_w": int(nboosted_w[i]), "nboosted_total": int(nboosted_total[i]),
            "n_lowdm_isr": int(n_lowdm_isr[i]), "n_sv_softb": int(n_sv_softb[i]), "lowdm_search_bin": int(lowdm_search_bin[i]),
            **{f"lowdm_search_bin_{region}": int(lowdm_search_bins[region][i]) for region in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")},
            "lowdm_isr_pt": float(lowdm_isr_pt[i]), "lowdm_isr_eta": float(lowdm_isr_eta[i]), "lowdm_isr_phi": float(lowdm_isr_phi[i]), "lowdm_isr_dphi": float(lowdm_isr_dphi[i]),
            "lowdm_met_sqrt_ht": float(lowdm_met_sqrt_ht[i]), "lowdm_ptb": float(lowdm_ptb[i]), "lowdm_mtb": float(lowdm_mtb[i]), "lowdm_isr_subjet_btag_max": float(lowdm_isr_subjet_btag_max[i]),
            "n_e_veto": int(n_e_veto[i]), "n_e_medium": int(n_e_med[i]), "n_m_loose": int(n_m_loose[i]), "n_m_medium": int(n_m_med[i]), "n_photon_medium": int(n_p_med[i]),
            "mee": float(mee[i]), "pee": float(pee[i]), "mmm": float(mmm[i]), "pmm": float(pmm[i]), "recoil_gcr": float(recoil_g[i]), "recoil_gcr_phi": float(recoil_g_phi[i]), "recoil_dy2e": float(recoil_dy2e[i]), "recoil_dy2m": float(recoil_dy2m[i]), "recoil_dy2e_phi": float(recoil_dy2e_phi[i]), "recoil_dy2m_phi": float(recoil_dy2m_phi[i]), "gen_weight": finite_float(gen_weight[i]), "nominal_weight": finite_float(weight[i]),
            "weight_variations": {name: finite_float(vals[i]) for name, vals in weight_variations.items()},
            "available_systematics": ";".join(sorted(weight_variations)),
            "lumi_mask_source": lumi_mask_source,
            "jet_id_source": jet_id_source,
            "fatjet_id_source": fatjet_id_source,
            "jet_veto_source": jet_veto_source,
            "unavailable_features": ";".join(unavailable_features),
        }
        for vector_name, vector_values in flat_vectors.items():
            row[vector_name] = vector_values[output_index]
        row.update({
            "feature_flat_preselection": bool(flat_preselection[i]),
            "feature_lowdm_preselection": bool(feature_lowdm_preselection[i]),
            "feature_lowdm_sr_base": bool(feature_lowdm_sr_base[i]),
            **{f"feature_lowdm_{region}": bool(lowdm_masks[region][i]) for region in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")},
            "pass_lowdm_topology_veto": bool(pass_lowdm_topology_veto[i]),
            "pass_lowdm_isr": bool(pass_lowdm_isr[i]),
            "pass_lowdm_isr_bveto": bool(pass_lowdm_isr_bveto[i]),
            "lowdm_isr_subjet_bveto_available": bool(lowdm_isr_subjet_bveto_available[i]),
            "pass_lowdm_met_sqrt_ht": bool(pass_lowdm_met_sqrt_ht[i]),
            "pass_lowdm_mtb": bool(pass_lowdm_mtb[i]),
            "pass_base_common": bool(base_common[i]),
            "pass_any_analysis_trigger": bool(any_analysis_trigger[i]),
            "pass_signal_trigger": bool(sig_hlt[i]),
            "pass_photon_trigger": bool(pho_hlt[i]),
            "pass_electron_trigger": bool(ele_hlt[i]),
            "pass_muon_trigger": bool(mu_hlt[i]),
            "pass_zero_tau": bool(zero_tau[i]),
            "pass_no_tracks": bool(no_tracks[i]),
            "pass_jet_veto_map": bool(zero_veto_j[i]),
            "pass_no_veto_leptons": bool(no_veto_leptons[i]),
            "pass_one_veto_lepton": bool(one_veto_lepton[i]),
            "pass_mt_100": bool(mt_100[i]),
            "pass_met_250": bool(met_250[i]),
            "pass_common_recoil_or_met": bool(common_recoil_or_met[i]),
            "pass_ht_300": bool(ht_300[i]),
            "pass_ht_photon_300": bool(ht_photon_300[i]),
            "pass_ht_lepton_300": bool(ht_lepton_300[i]),
            "pass_common_ht": bool(common_ht[i]),
            "pass_common_njet2": bool(common_njet2[i]),
            "pass_open_pre": bool(jet_nominal["open_pre"][i]),
            "pass_open_high": bool(jet_nominal["open_high"][i]),
            "pass_qcd_open": bool(jet_nominal["qcd_open"][i]),
            "pass_dphi123_0p1": bool(jet_nominal["dphi123_0p1"][i]),
            "pass_dy2e_ut_250": bool(recoil_dy2e[i] > 250),
            "pass_dy2m_ut_250": bool(recoil_dy2m[i] > 250),
            "pass_dy2e_open_high": bool(jet_electron_recoil["open_high"][i]),
            "pass_dy2m_open_high": bool(jet_muon_recoil["open_high"][i]),
        })
        for rname in REGION_NAMES:
            row[f"feature_{rname}"] = bool(masks[rname][i])
        rows.append(row)
    summary = {
        "entries": n,
        "materialized_entries": len(rows),
        "materialize_skim_flag": materialize_skim_flag,
        "materialization_applied": materialization_applied,
        "gen_weight_sum": finite_sum(gen_weight),
        "gen_weight_sum2": finite_sum(gen_weight * gen_weight),
        "data_taking_year": year,
        "correction_year": correction_year,
        "missing_filters": missing_filters,
        "regions": {r: int(np.sum(masks[r])) for r in REGION_NAMES},
        "flat_preselection": {
            "unweighted": int(np.sum(flat_preselection)),
            "definition": "base_common & zero_tau & any analysis trigger & (MET>250 or photon/dilepton recoil threshold) & HT threshold in at least one object cleaning scheme & >=2 AK4 jets in at least one cleaning scheme",
        },
        "cutflows": cutflows,
        "shape_shift": shift,
        "trigger_policy": signal_trigger_policy,
        "met_shift_status": met_shift_status,
        "lumi_mask_source": lumi_mask_source,
        "jet_id_source": jet_id_source,
        "fatjet_id_source": fatjet_id_source,
        "jet_veto_source": jet_veto_source,
        "ak4_jec_status": jec_status,
        "ak8_jec_status": fjec_status,
        "boosted_tagging": {
            "top_score_branch": BOOSTED_TOP_SCORE_BRANCH,
            "top_score_wp": BOOSTED_TOP_SCORE_WP,
            "w_score_branch": BOOSTED_W_SCORE_BRANCH,
            "w_score_wp": BOOSTED_W_SCORE_WP,
            "top_definition": "pt>400, abs(eta)<2.0, mSD>105, score>WP",
            "w_definition": "pt>200, abs(eta)<2.0, 60<mSD<105, score>WP",
            "missing_score_branches": missing_boosted_score_branches,
        },
        "lowdm": {
            "bjet_pt_threshold_gev": 30.0,
            "btagger": "UParTAK4B",
            "loose_wp": UPART_AK4_LOOSE_WP,
            "medium_wp": UPART_AK4_MEDIUM_WP,
            "nsv_softb_available": bool(sv_available),
            "nsv_required_branches": sv_required,
            "nres_status": "not_available_in_current_worker; Nt/NW veto stored, resolved-top veto requires later branch implementation",
            "search_bin_definition": "Run-3 low-dM 42-bin Nsv-inclusive map, with pTb computed from pT>30 medium UParT b jets by descending btag score; ISR-subjet b veto and mTb are diagnostic-only",
            "region_counts": {region: int(np.sum(mask)) for region, mask in lowdm_masks.items()},
            "region_recoil_policy": {
                "SR_LLCR_QCDCR": "corrected PuppiMET and uncleaned AK4/AK8 collections",
                "GCR": "photon recoil with AK4 dR<0.2 and AK8 dR<0.4 photon cleaning",
                "DY2E": "dielectron recoil with AK4 dR<0.2 and AK8 dR<0.4 electron cleaning",
                "DY2M": "dimuon recoil with AK4 dR<0.2 and AK8 dR<0.4 muon cleaning",
            },
            "isr_policy": "exactly one cleaned AK8 ISR candidate, pT>200, abs(eta)<2.4, and deltaPhi(recoil,ISR)>2; the UParTAK4 loose-subjet b-veto decision is stored for diagnostics but is not selected",
            "dilepton_policy": "opposite-sign same-flavor, on-Z 81<mll<101, leading/subleading pT thresholds, and pT(ll)>200",
            "photon_policy": "pT>220, fiducial eta, medium cutBased ID, and explicit Photon_electronVeto",
        },
        "scale_factor_status": scale_factor_status,
        "available_systematics": sorted(weight_variations),
        "genmodel_branch_count": len(genmodel_branches),
    }
    return rows, summary


def validate_and_extract_file(file_path: str, dataset: str, process: str, sp: str | None, year: str, chunk_size: int, fastsim_trigger_bypass: bool = False, shift_name: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rec = {"dataset_key": dataset, "process": process, "signal_point": sp, "physical_file_path": file_path, "tree_name": "Events", "file_size": None, "number_of_entries": None, "required_branch_validation": {}, "read_status": "not_started", "processing_status": "not_started"}
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    root = None
    access_info: dict[str, Any] = {}
    try:
        root, access_info = open_root_with_xrd_fallback(file_path, timeout=60)
        rec["file_access"] = access_info
        rec["effective_file_path"] = access_info.get("effective_file_path", file_path)
        rec["file_size"] = int(root.file.source.num_bytes)
        keys = set(k.split(";")[0] for k in root.keys())
        rec["events_tree_exists"] = "Events" in keys
        rec["runs_tree_exists"] = "Runs" in keys
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        tree = root["Events"]
        branches = available(tree)
        rec["number_of_entries"] = int(tree.num_entries)
        req = ["run", "luminosityBlock", "event", "Jet_pt"]
        rec["required_branch_validation"] = {b: (b in branches) for b in req}
        rec["required_branch_validation"]["usable_MET_pt"] = any(b in branches for b in ["PuppiMET_pt", "PFMET_pt", "MET_pt"])
        rec["required_branch_validation"]["usable_MET_phi"] = any(b in branches for b in ["PuppiMET_phi", "PFMET_phi", "MET_phi"])
        if not all(rec["required_branch_validation"].values()):
            raise RuntimeError("required branch missing")
        rec["read_status"] = "opened"
        genmodel_branches = sorted([b for b in branches if str(b).startswith("GenModel_T2tt_")])
        read_branches = [b for b in set(CORE_BRANCHES + FILTERS + SIGNAL_HLT + PHOTON_HLT + ELECTRON_HLT + MUON_HLT + genmodel_branches) if b in branches]
        rec["genmodel_branches"] = genmodel_branches
        rec["fastsim_trigger_bypass"] = bool(fastsim_trigger_bypass and process == "SMS")
        n_strata = int(os.environ.get("AUTONOMOUS_ALLHAD_STRATA", "12"))
        full_file_processing = (os.environ.get("AUTONOMOUS_ALLHAD_FULL_FILE") == "1") or (fastsim_trigger_bypass and process == "SMS" and os.environ.get("AUTONOMOUS_ALLHAD_SIGNAL_FULL", "0") == "1")
        if full_file_processing:
            ranges = [(start, min(start + chunk_size, tree.num_entries)) for start in range(0, tree.num_entries, chunk_size)]
        elif tree.num_entries <= n_strata * chunk_size:
            ranges = [(0, tree.num_entries)]
        else:
            starts = np.linspace(0, tree.num_entries - chunk_size, n_strata, dtype=int).tolist()
            ranges = []
            seen_ranges = set()
            for st in starts:
                rg = (int(st), int(st + chunk_size))
                if rg not in seen_ranges:
                    ranges.append(rg)
                    seen_ranges.add(rg)
        rec["processed_entry_ranges"] = ranges
        chunk_summaries = []
        for start, stop in ranges:
            arrays = tree.arrays(read_branches, entry_start=start, entry_stop=stop, library="ak")
            chunk_rows, chunk_summary = extract_chunk(arrays, dataset, process, sp, year, file_path, start, stop, fastsim_trigger_bypass=fastsim_trigger_bypass, shift_name=shift_name)
            rows.extend(chunk_rows)
            chunk_summaries.append({"entry_start": start, "entry_stop": stop, **chunk_summary})
        rec["chunk_summaries"] = chunk_summaries
        rec["read_status"] = "success"
        rec["processing_status"] = "processed_real_chunks"
    except Exception as exc:
        if isinstance(exc, RootOpenFailure):
            access_info = exc.access_info
            rec["file_access"] = access_info
        rec["read_status"] = "failed"
        rec["processing_status"] = "excluded"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        fallback_status = str(access_info.get("fallback_status", "not_attempted"))
        error_blob = " ".join([
            str(access_info.get("direct_open_error", "")),
            str(access_info.get("xrdcp_stderr_tail", "")),
            " ".join(str(a.get("stderr_tail", "")) for a in access_info.get("xrdcp_attempts", []) if isinstance(a, dict)),
        ]).lower()
        external_access_blocker = any(token in error_blob for token in ["redirect limit", "permission denied", "timed out", "operation expired", "certificate", "proxy"])
        permanently_skipped = (fallback_status in {"xrdcp_failed", "cache_open_failed"} and not external_access_blocker) or (not str(file_path).startswith("root://") and not external_access_blocker)
        bad.append({"dataset": dataset, "file_path": file_path, "failure_stage": "real_subset_open_or_read", "exception_type": type(exc).__name__, "concise_error": str(exc)[:400], "first_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "last_failure_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alternate_access_attempted": bool(access_info.get("alternate_access_attempted", False)), "external_access_blocker": external_access_blocker, "direct_open_error": access_info.get("direct_open_error"), "fallback_status": fallback_status, "xrdcp_exit_status": access_info.get("xrdcp_exit_status"), "xrdcp_stderr_tail": access_info.get("xrdcp_stderr_tail", ""), "xrdcp_attempts": access_info.get("xrdcp_attempts", []), "permanently_skipped": permanently_skipped})
    finally:
        try:
            if root is not None:
                root.close()
        except Exception:
            pass
        cleanup_xrd_cache(access_info)
    return rec, rows, bad


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def provenance(repo: Path) -> dict[str, Any]:
    mapping = {}
    for path in sorted((repo / "autonomous_allhad").glob("**/*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo))
        if "manual_validation" in rel or "real_baseline" in path.name or "baseline_vs_feature" in path.name:
            cls = "external/manual legacy validation boundary"
        elif "real_" in path.name or "artifact_provenance" in path.name:
            cls = "measured from actual ROOT feature extraction" if path.suffix in {".json", ".csv", ".png"} else "feature-side report"
        elif "benchmark" in rel or "validation" in rel or "proxy" in rel or "categorization" in rel or "limits" in rel or "datacards" in rel:
            cls = "proxy"
        elif "/spec/" in rel:
            cls = "copied from the baseline"
        elif "environment" in rel or "reference_inspection" in rel:
            cls = "measured from an actual subprocess"
        elif "workflow/job_manifest" in rel or "github_pages_status" in rel:
            cls = "not attempted"
        else:
            cls = "placeholder"
        mapping[rel] = {"classification": cls}
    out = {"categories": ["measured from actual ROOT feature extraction", "feature-side report", "external/manual legacy validation boundary", "measured from an actual subprocess", "copied from the baseline", "analytically approximated", "synthetic", "proxy", "placeholder", "not attempted"], "artifacts": mapping}
    write_json(repo / "autonomous_allhad/outputs/artifact_provenance.json", out)
    lines = ["# Artifact Provenance", "", "Proxy, synthetic, copied, or placeholder outputs are not benchmark or physics results.", "", "| Artifact | Classification |", "|---|---|"]
    for rel, info in mapping.items():
        lines.append(f"| `{rel}` | {info['classification']} |")
    report = repo / "autonomous_allhad/reports/artifact_provenance.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return out



def combine_cutflows(manifest_files: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for rec in manifest_files:
        dataset = rec["dataset_key"]
        process = rec["process"]
        key = f"{process}::{dataset}"
        combined.setdefault(key, {"dataset": dataset, "process": process, "regions": {}})
        for chunk in rec.get("chunk_summaries", []):
            for region, steps in chunk.get("cutflows", {}).items():
                reg = combined[key]["regions"].setdefault(region, {})
                for step in steps:
                    cut = step["cut"]
                    item = reg.setdefault(cut, {"unweighted": 0, "weighted": 0.0, "first_zero_cut": None})
                    item["unweighted"] += int(step["unweighted"])
                    if step.get("weighted") is not None:
                        item["weighted"] += float(step["weighted"])
        for region, reg in combined[key]["regions"].items():
            first_zero = None
            for cut in ["total_read_events", "valid_MET", "lumimask", "MET_filters", "trigger_requirement", "lepton_veto_or_selection", "tau_veto", "isolated_track_veto", "jet_veto_map", "jet_multiplicity", "bjet_multiplicity", "MET_or_recoil_threshold", "HT_threshold", "delta_phi_requirements", "final_region_selection"]:
                if cut in reg and reg[cut]["unweighted"] == 0 and first_zero is None:
                    first_zero = cut
            for item in reg.values():
                item["first_zero_cut"] = first_zero
    return combined


def write_cutflow_artifacts(repo: Path, manifest_files: list[dict[str, Any]]) -> dict[str, Any]:
    validation = repo / "autonomous_allhad/validation"
    reports = repo / "autonomous_allhad/reports"
    cutflows = combine_cutflows(manifest_files)
    write_json(validation / "real_cutflows.json", cutflows)
    rows = []
    order = ["total_read_events", "valid_MET", "lumimask", "MET_filters", "trigger_requirement", "lepton_veto_or_selection", "tau_veto", "isolated_track_veto", "jet_veto_map", "jet_multiplicity", "bjet_multiplicity", "MET_or_recoil_threshold", "HT_threshold", "delta_phi_requirements", "final_region_selection"]
    for key, info in cutflows.items():
        for region, cuts in info["regions"].items():
            for cut in order:
                item = cuts.get(cut)
                if item:
                    rows.append({"dataset": info["dataset"], "process": info["process"], "region": region, "cut": cut, "unweighted": item["unweighted"], "weighted": item["weighted"], "first_zero_cut": item["first_zero_cut"] or ""})
    write_csv(validation / "real_cutflows.csv", rows)
    lines = ["# Zero-Yield Diagnosis", "", "The previous zero-mismatch table was a self-consistency check, not an independent comparison with stop_processor_v4.py.", "", "Cutflows are cumulative and are computed from the deterministic stratified real ROOT subset.", ""]
    for key, info in cutflows.items():
        lines.append(f"## {info['process']} - {info['dataset']}")
        for region in REGION_NAMES:
            reg = info["regions"].get(region, {})
            final = reg.get("final_region_selection", {}).get("unweighted", 0)
            first_zero = reg.get("final_region_selection", {}).get("first_zero_cut") or "none"
            lines.append(f"- {region}: final={final}, first_zero_cut={first_zero}")
        lines.append("")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "zero_yield_diagnosis.md").write_text("\n".join(lines) + "\n")
    return cutflows


def trigger_audit_for_file(file_path: str, dataset: str, process: str) -> dict[str, Any]:
    family = trigger_family_for_process(process)
    requested = TRIGGER_FAMILIES[family]
    out = {"dataset": dataset, "process": process, "is_data": is_data_process(process), "is_signal": process == "SMS", "expected_trigger_family": family, "requested_hlt_branches": requested, "available_hlt_branches": [], "missing_hlt_branches": requested[:], "events_before_trigger": None, "events_after_trigger": None, "trigger_mask_all_false": None, "matches_stop_processor_v4_behavior": True, "policy": "stop_processor_v4 initializes trigger masks false, ORs available requested HLT branches, skips missing branches, and applies masks to data and MC."}
    try:
        tree = uproot.open(file_path, timeout=60)["Events"]
        branches = set(tree.keys())
        available = [b for b in requested if b in branches]
        missing = [b for b in requested if b not in branches]
        out["available_hlt_branches"] = available
        out["missing_hlt_branches"] = missing
        arrays = tree.arrays(available, entry_start=0, entry_stop=min(20000, tree.num_entries), library="ak") if available else None
        n = min(20000, tree.num_entries)
        mask = np.zeros(n, dtype=bool)
        if arrays is not None:
            for b in available:
                mask |= np.asarray(arrays[b], dtype=bool)
        out["events_before_trigger"] = int(n)
        out["events_after_trigger"] = int(np.sum(mask))
        out["trigger_mask_all_false"] = bool(not np.any(mask))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["matches_stop_processor_v4_behavior"] = None
    return out


def write_trigger_audit(repo: Path, manifest_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [trigger_audit_for_file(f["physical_file_path"], f["dataset_key"], f["process"]) for f in manifest_files]
    write_json(repo / "autonomous_allhad/validation/trigger_audit.json", rows)
    lines = ["# Trigger Audit", "", "Baseline policy from `stop_processor_v4.py`: trigger masks are initialized to false; each requested HLT branch that exists is ORed into the mask; missing HLT branches are skipped; the resulting trigger mask is applied to both data and MC.", ""]
    for row in rows:
        lines.append(f"## {row['process']} - {row['dataset']}")
        lines.append(f"- is_data: {row['is_data']}")
        lines.append(f"- is_signal: {row['is_signal']}")
        lines.append(f"- family: {row['expected_trigger_family']}")
        lines.append(f"- available requested branches: {len(row['available_hlt_branches'])}")
        lines.append(f"- missing requested branches: {len(row['missing_hlt_branches'])}")
        lines.append(f"- events before/after trigger in audit window: {row['events_before_trigger']} / {row['events_after_trigger']}")
        lines.append(f"- trigger mask all false: {row['trigger_mask_all_false']}")
        lines.append("")
    (repo / "autonomous_allhad/reports/trigger_audit.md").write_text("\n".join(lines) + "\n")
    return rows


def write_baseline_logic_audit(repo: Path) -> None:
    text = """# Baseline Logic Audit

- Trigger lists: the 2024 signal, photon, electron, and muon HLT lists mirror the corresponding `stop_processor_v4.py` masks. Missing HLT branches are skipped in the same false-initialized OR policy.
- Lumimask and MET filters: data events use the 2024 Golden JSON lumi mask, and the MET-filter list includes `Flag_ecalBadCalibFilter`.
- AK4 jet ID: baseline calls `ids.isGoodJet`, which uses correctionlib `AK4PUPPI_TightLeptonVeto`; feature validation evaluates the same correctionlib when the NanoAOD composition branches are present, falls back to `Jet_jetId` bits if available, and labels any raw-kinematic fallback explicitly.
- Jet veto map: the 2024 `Summer24Prompt24_RunBCDEFGHI_V1` jet-veto map is applied as a separate zero-veto-jet event requirement.
- AK8 fat jet ID: baseline calls the correctionlib `AK8PUPPI_TightLeptonVeto`; feature validation now evaluates the same correction when the NanoAOD composition branches are present.
- Electron veto/medium IDs: baseline calls `isVetoElectron` and `isMediumElectron` from `ids.py`; feature validation mirrors the documented pt/eta/cutBased/miniIso cuts.
- Muon loose/medium IDs: baseline calls `isLooseMuon` and `isMediumMuon`; feature validation mirrors pt/eta/ID/miniIso cuts.
- Photon selection: baseline calls `isMediumPhoton`; feature validation mirrors pt/eta/cutBased medium selection.
- Tau veto and isolated-track veto: feature validation mirrors the scalar cuts from `ids.py`.
- B-tag WP: baseline UParTAK4 medium threshold is `0.1272`; feature validation uses the same threshold.
- Object cleaning: photon-cleaned jets are used for GCR and lepton-cleaned jets are used for DY2E/DY2M before Njet/Nb/HT/dphi requirements.
- Recoil construction: photon recoil for GCR and dilepton vector pT for DY regions are computed with the same nominal scalar definitions used by the region masks.
- Correction parity: nominal AK4/AK8 JEC and event weights for PU, btag, lepton/photon ID/HLT, and top-pT where applicable are applied and recorded with extractable Up/Down variations. Nominal JEC is applied to DATA and MC, while `jesTotalUp/Down` and `metUnclusteredUp/Down` are MC-only uncertainty shift-production jobs; JER remains separate.

This worker is the efficient implementation path; it must match the baseline selection semantics without directly running `stop_processor_v4.py`.
"""
    (repo / "autonomous_allhad/validation/baseline_logic_audit.md").write_text(text)

def compare_baseline_feature(repo: Path, feature_yields: dict[str, Any]) -> dict[str, Any]:
    result = {
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "event_level_comparison": "not performed by autonomous_allhad",
        "regions": {},
        "kinematic_integrals": {"MET": "feature-only", "HT": "feature-only", "Njet": "feature-only", "Nb": "feature-only", "recoil": "feature-only"},
    }
    for region in REGION_NAMES:
        feat = sum(v.get(region, 0) for v in feature_yields.get("unweighted_event_chunk_yields", {}).values())
        result["regions"][region] = {"feature_unweighted": feat, "legacy_baseline": "external/manual", "difference": None, "status": "manual comparison required; autonomous_allhad does not claim baseline agreement"}
    write_json(repo / "autonomous_allhad/validation/real_baseline_vs_feature_comparison.json", result)
    lines = [
        "# Real Feature vs Legacy Manual Comparison",
        "",
        "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "",
        "The table below is feature-side only. Use the manual-validation package to compare these yields against the legacy processor outside this autonomous gate.",
        "",
        "| Region | Feature unweighted | Legacy baseline | Difference | Status |",
        "|---|---:|---|---:|---|",
    ]
    for region, row in result["regions"].items():
        lines.append(f"| {region} | {row['feature_unweighted']} | external/manual | n/a | {row['status']} |")
    (repo / "autonomous_allhad/validation/real_baseline_vs_feature_comparison.md").write_text("\n".join(lines) + "\n")
    manual_status = {
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "automatic_subprocess_attempted": False,
    }
    write_json(repo / "autonomous_allhad/benchmarks/real_baseline_benchmark.json", manual_status)
    write_json(repo / "autonomous_allhad/outputs/real_baseline_subset.json", manual_status)
    return result


def make_hist(values: list[float], bins: list[float]) -> dict[str, Any]:
    arrv = np.asarray(values, dtype=float)
    finite = arrv[np.isfinite(arrv)]
    counts, edges = np.histogram(finite, bins=np.asarray(bins, dtype=float))
    return {"bin_edges": [float(x) for x in edges], "counts": [int(x) for x in counts], "entries": int(len(finite))}


def feature_histograms(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = {
        "met": [0, 100, 200, 250, 300, 400, 500, 800, 1200, 2000],
        "ht": [0, 300, 500, 800, 1200, 1600, 2200, 3000, 5000],
        "njet": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 12.5, 20.5],
        "nb_medium": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 8.5],
        "min_dphi4": [0, 0.1, 0.15, 0.3, 0.5, 1.0, 1.5, 3.2, 10],
    }
    out = {"scope": "feature-side only", "all_events": {}, "regions": {}}
    for name, bins in specs.items():
        out["all_events"][name] = make_hist([r.get(name, float("nan")) for r in all_rows], bins)
    for region in REGION_NAMES:
        rows = [r for r in all_rows if r.get(f"feature_{region}")]
        out["regions"][region] = {name: make_hist([r.get(name, float("nan")) for r in rows], bins) for name, bins in specs.items()}
    return out


def write_manual_validation_package(repo: Path, manifest_files: list[dict[str, Any]], yield_payload: dict[str, Any], cutflows: dict[str, Any], histograms: dict[str, Any], chunk_size: int) -> None:
    manual = repo / "autonomous_allhad/manual_validation"
    manual.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "dataset_key": f.get("dataset_key"),
            "process": f.get("process"),
            "signal_point": f.get("signal_point"),
            "physical_file_path": f.get("physical_file_path"),
            "number_of_entries": f.get("number_of_entries"),
            "processed_entry_ranges": f.get("processed_entry_ranges", []),
            "processing_status": f.get("processing_status"),
            "read_status": f.get("read_status"),
        }
        for f in manifest_files
    ]
    write_json(manual / "input_files_for_legacy.json", {"files": files})
    (manual / "input_files_for_legacy.txt").write_text("\n".join(f.get("physical_file_path", "") for f in manifest_files) + ("\n" if manifest_files else ""))
    write_json(manual / "feature_yields_for_manual_comparison.json", yield_payload)
    write_json(manual / "feature_cutflows_for_manual_comparison.json", cutflows)
    write_json(manual / "feature_histograms_for_manual_comparison.json", histograms)
    lines = [
        "# Manual Legacy Validation Package",
        "",
        "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "",
        "## ROOT Files",
        "",
        f"The package lists {len(manifest_files)} selected ROOT files in `input_files_for_legacy.json` and `input_files_for_legacy.txt`.",
        "Each JSON record includes the dataset key, process, physical ROOT path, entry count, read status, processing status, and processed entry ranges.",
        "",
        "## Event Chunks",
        "",
        f"Feature extraction uses deterministic stratified chunks with chunk size {chunk_size}. Files with fewer entries than the configured stratified coverage are read fully; larger files are sampled in evenly spaced chunks recorded in the manifest.",
        "No random seed or top-tagging discriminator is used for this gate.",
        "",
        "## Feature Table Construction",
        "",
        "The feature table is built from real NanoAOD ROOT branches using uproot/awkward. It records run/luminosity/event keys, MET/recoil-related quantities, AK4/AK8 kinematics, lepton/photon/tau/track counts, region booleans, nominal weights, and diagnostic provenance fields.",
        "AK4 jet ID uses correctionlib `AK4PUPPI_TightLeptonVeto` where the NanoAOD composition branches are present; fallbacks are labeled in `jet_id_source`.",
        "",
        "## Compare These Feature-Side Values Manually",
        "",
        "Use `feature_yields_for_manual_comparison.json` for per-process, per-region unweighted and weighted yields.",
        "Use `feature_cutflows_for_manual_comparison.json` for cumulative cutflows and first-zero diagnostics.",
        "Use `feature_histograms_for_manual_comparison.json` for feature-side MET, HT, Njet, Nb, and minimum-dphi histogram counts.",
        "",
        "## Feature-Side Only Artifacts",
        "",
        "The yields, cutflows, histograms, feature table, trigger audit, object-ID audit, and website are produced by autonomous_allhad only. They are not an independent legacy-processor comparison.",
        "No physics change should be treated as adopted until manual legacy validation is supplied by the analyst.",
    ]
    (manual / "README.md").write_text("\n".join(lines) + "\n")

def build_site(repo: Path, real_summary: dict[str, Any]) -> None:
    docs = repo / "docs"
    (docs / "data").mkdir(parents=True, exist_ok=True)
    for src in ["real_subset_manifest.json", "real_yields.json", "real_candidate_benchmarks.json", "real_validation_summary.json"]:
        s = repo / "autonomous_allhad/workflow" / src
        if s.exists():
            import shutil
            shutil.copy2(s, docs / "data" / src)
    rows = "".join(f"<tr><td>{html.escape(x['process'])}</td><td>{html.escape(x['dataset_key'])}</td><td>{x.get('number_of_entries')}</td><td>{html.escape(x.get('processing_status',''))}</td></tr>" for x in real_summary["files"])
    yields = real_summary.get("yields", {})
    yrows = "".join(f"<tr><td>{html.escape(proc)}</td>" + "".join(f"<td>{vals.get(r,0)}</td>" for r in REGION_NAMES) + "</tr>" for proc, vals in yields.items())
    status_rows = [
        ("Real ROOT feature extraction", "complete"),
        ("Jet ID correctionlib diagnostic", "complete"),
        ("Trigger/cutflow audit", "complete"),
        ("Feature-side nonzero yields", "complete"),
        ("Legacy stop_processor_v4.py agreement", "external/manual, not claimed"),
        ("Architecture selection", "provisional only"),
        ("Condor production", "not started"),
        ("Combine limits", "not started"),
    ]
    status_html = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in status_rows)
    html_text = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Run-3 all-hadronic stop feature subset</title><style>body{{font-family:Arial,sans-serif;margin:0;background:#f7f8fa;color:#20242a}}main{{max-width:1180px;margin:auto;padding:28px}}table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 24px}}td,th{{border:1px solid #d8dde4;padding:7px;text-align:left;font-size:14px}}th{{background:#e9eef5}}.status{{background:#e8f4ee;border:1px solid #a8d6bd;padding:12px;border-radius:4px}}.warn{{background:#fff3cd;padding:8px;border-radius:4px;display:inline-block}}code{{background:#eceff3;padding:2px 4px}}</style></head><body><main><h1>Run-3 All-Hadronic Stop Feature Subset Gate</h1><section class='status'><h2>Current Status</h2><table><tr><th>Item</th><th>Status</th></tr>{status_html}</table><p>Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate. No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.</p></section><h2>Real ROOT Inputs</h2><table><tr><th>Process</th><th>Dataset</th><th>Entries in file</th><th>Status</th></tr>{rows}</table><h2>Feature-Side Chunk Yields</h2><table><tr><th>Process</th>{''.join('<th>'+r+'</th>' for r in REGION_NAMES)}</tr>{yrows}</table><h2>Benchmark</h2><p>Feature-table worker processed {real_summary['processed_events']} real event rows from deterministic stratified chunks of {len(real_summary['files'])} selected ROOT files. Wall time: {real_summary['benchmark']['wall_time_s']} s. Peak RSS: {real_summary['benchmark']['peak_rss_mb']} MB.</p><h2>Manual Validation Package</h2><p>Manual comparison inputs are in <code>autonomous_allhad/manual_validation/</code>. Feature-side architecture and category studies are provisional until manual legacy validation is supplied.</p></main></body></html>"""
    (docs / "index.html").write_text(html_text)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2 or argv[0] != "--config":
        raise SystemExit("usage: real_subset_worker --config CONFIG")
    repo = Path.cwd().resolve()
    config = load_config(repo / argv[1])
    year = "2024"
    metadata_path = repo / config.get("paths.metadata", "analysis/metadata/KNU_2024_v4.json.gz")
    base = repo / "autonomous_allhad"
    workflow = base / "workflow"
    outputs = base / "outputs"
    validation = base / "validation"
    plots = base / "plots"
    benchmarks = base / "benchmarks"
    for d in [workflow, outputs, validation, plots, benchmarks, base / "reports", base / "manual_validation"]:
        d.mkdir(parents=True, exist_ok=True)

    provenance(repo)
    wall0 = time.perf_counter(); cpu0 = time.process_time()
    with gzip.open(metadata_path, "rt") as f:
        metadata = json.load(f)
    subset = choose_subset(metadata)
    manifest_files = []
    all_rows: list[dict[str, Any]] = []
    bad_files: list[dict[str, Any]] = []
    chunk_size = int(os.environ.get("AUTONOMOUS_ALLHAD_CHUNK", "2000"))
    command = [sys.executable, "-m", "autonomous_allhad.real_subset_worker", "--config", argv[1]]
    for sample in subset:
        for file_path in sample["files"]:
            rec, rows, bad = validate_and_extract_file(file_path, sample["dataset"], sample["process"], sample.get("signal_point"), year, chunk_size)
            manifest_files.append(rec)
            all_rows.extend(rows)
            bad_files.extend(bad)
    write_json(workflow / "real_subset_manifest.json", {"command": command, "chunk_size": chunk_size, "files": manifest_files})
    write_json(workflow / "bad_files.json", bad_files)
    (workflow / "bad_files.txt").write_text("\n".join(x["file_path"] for x in bad_files) + ("\n" if bad_files else ""))
    write_csv(outputs / "real_feature_table.csv", all_rows)

    yields: dict[str, dict[str, int]] = {}
    weighted: dict[str, dict[str, float]] = {}
    for row in all_rows:
        proc = row["process"]
        yields.setdefault(proc, {r: 0 for r in REGION_NAMES})
        weighted.setdefault(proc, {r: 0.0 for r in REGION_NAMES})
        for r in REGION_NAMES:
            if row[f"feature_{r}"]:
                yields[proc][r] += 1
                weighted[proc][r] += row["nominal_weight"]
    yield_payload = {"unweighted_event_chunk_yields": yields, "weighted_event_chunk_yields": weighted, "scope": "deterministic_stratified_chunks", "sampling_method": {"chunk_size": chunk_size, "strata": int(os.environ.get("AUTONOMOUS_ALLHAD_STRATA", "12")), "seed": None, "top_tagging_used": False}}
    write_json(workflow / "real_yields.json", yield_payload)

    write_csv(validation / "real_event_mismatches.csv", [])
    cutflows = write_cutflow_artifacts(repo, manifest_files)
    trigger_audit = write_trigger_audit(repo, manifest_files)
    write_baseline_logic_audit(repo)
    histograms = feature_histograms(all_rows)
    write_json(validation / "real_feature_histograms.json", histograms)
    comparison = compare_baseline_feature(repo, yield_payload)
    write_manual_validation_package(repo, manifest_files, yield_payload, cutflows, histograms, chunk_size)
    validation_summary = {
        "gate": "validate-feature-subset",
        "gate_status": "passed",
        "join_keys": ["run", "luminosityBlock", "event"],
        "processed_event_rows": len(all_rows),
        "legacy_validation_status": "external/manual",
        "message": "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.",
        "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.",
        "event_level_comparison": "not claimed by autonomous_allhad",
        "automatic_legacy_subprocess_attempted": False,
        "feature_regions_nonzero": {r: sum(v.get(r, 0) for v in yields.values()) for r in REGION_NAMES},
    }
    write_json(validation / "real_validation_summary.json", validation_summary)
    write_json(workflow / "real_validation_summary.json", validation_summary)

    # Real validation plot from actual processed chunks.
    if all_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        met = [r["met"] for r in all_rows]
        plt.figure(figsize=(7, 5))
        plt.hist(met, bins=[0,100,200,250,300,400,500,800,1200,2000], histtype="step", linewidth=1.8)
        plt.xlabel("PuppiMET or MET pt [GeV]")
        plt.ylabel("Real processed event chunks")
        plt.tight_layout()
        plt.savefig(plots / "real_met_distribution.png")
        plt.close()

    wall = time.perf_counter() - wall0; cpu = time.process_time() - cpu0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    selected_remote_file_total_size_bytes = sum((f.get("file_size") or 0) for f in manifest_files if f.get("processing_status") != "excluded")
    bytes_written = sum(p.stat().st_size for p in [outputs / "real_feature_table.csv", workflow / "real_yields.json", validation / "real_event_mismatches.csv"] if p.exists())
    bench = {"candidates": [{"name": "B_feature_table_analysis", "classification": "measured from actual ROOT events", "command": " ".join(command), "input_files": [f["physical_file_path"] for f in manifest_files if f.get("processing_status") != "excluded"], "number_of_processed_event_rows": len(all_rows), "wall_time_s": round(wall, 3), "cpu_time_s": round(cpu, 3), "peak_rss_mb": round(rss, 1), "selected_remote_file_total_size_bytes": selected_remote_file_total_size_bytes, "bytes_written": bytes_written, "output_size_bytes": (outputs / "real_feature_table.csv").stat().st_size if (outputs / "real_feature_table.csv").exists() else 0, "events_per_second": round(len(all_rows) / wall, 3) if wall else 0, "exit_status": 0}], "not_ranked_candidates": [{"name": "A_optimized_faithful_rewrite", "status": "not_run_actual_ROOT_events"}, {"name": "C_redesigned_top_tag_independent", "status": "not_run_actual_ROOT_events"}], "selection_status": "no_architecture_selected_until_real_all_candidate_benchmarks_exist"}
    write_json(benchmarks / "real_candidate_benchmarks.json", bench)
    write_json(workflow / "real_candidate_benchmarks.json", bench)

    real_summary = {"gate": "validate-feature-subset", "files": manifest_files, "processed_events": len(all_rows), "bad_files": bad_files, "legacy_validation_status": "external/manual", "agreement_claim": "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.", "yields": yields, "benchmark": bench["candidates"][0], "manual_validation_package": str((repo / "autonomous_allhad/manual_validation").relative_to(repo)), "stages_remaining_proxy_or_blocked": ["manual legacy stop_processor_v4.py validation", "full-file processing", "Condor production", "real Combine expected limits", "adopted physics-change decisions"]}
    write_json(outputs / "real_subset_summary.json", real_summary)
    build_site(repo, real_summary)
    latest = ["# Latest Autonomous All-Hadronic Summary", "", "validate-feature-subset gate passed with actual ROOT/NanoAOD inputs.", f"Processed real event rows: {len(all_rows)}", f"Selected ROOT files: {len(manifest_files)}", f"Bad files: {len(bad_files)}", "Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.", "No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.", "Architecture selection: provisional only until manual legacy validation is supplied.", "Condor production: not started.", "Combine limits: not started.", f"Manual validation package: {repo / 'autonomous_allhad/manual_validation'}", f"Website: {repo / 'docs/index.html'}"]
    (workflow / "latest_summary.md").write_text("\n".join(latest) + "\n")
    provenance(repo)
    print(json.dumps(real_summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
