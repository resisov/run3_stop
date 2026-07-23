from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import correctionlib
import numpy as np
from scipy.special import erf, erfinv


ERA = "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"
CVMFS_BASE = Path("/cvmfs/cms-griddata.cern.ch/cat/metadata")
JEC_TAG = "Summer24Prompt24_V5"
JER_TAG = "Summer24Prompt24_JRV2"


@dataclass(frozen=True)
class Payload:
    group: str
    filename: str
    sha256: str
    required_corrections: tuple[str, ...] = ()
    required_compounds: tuple[str, ...] = ()

    @property
    def source(self) -> Path:
        return CVMFS_BASE / self.group / ERA / "latest" / self.filename

    @property
    def relative(self) -> Path:
        return Path("payloads") / self.group / self.filename


PAYLOADS = (
    Payload(
        "JME",
        "jet_jerc.json.gz",
        "b0e60dd2244f9d4cfa3215eafb94afa5b6a491b18af4584522b749d56f5c9540",
        (
            f"{JEC_TAG}_MC_Total_AK4PFPuppi",
            f"{JEC_TAG}_MC_Regrouped_Total_AK4PFPuppi",
            f"{JER_TAG}_MC_PtResolution_AK4PFPuppi",
            f"{JER_TAG}_MC_ScaleFactor_AK4PFPuppi",
            f"{JER_TAG}_MC_SFUncertainty_AK4PFPuppi",
        ),
        (
            f"{JEC_TAG}_MC_L1L2L3Res_AK4PFPuppi",
            f"{JEC_TAG}_DATA_L1L2L3Res_AK4PFPuppi",
        ),
    ),
    Payload(
        "JME",
        "fatJet_jerc.json.gz",
        "09dd3f22f0fa8d283140e03edb4f5d60b840f1cea5183225ccb095a68dcb24b1",
        (
            f"{JEC_TAG}_MC_Total_AK8PFPuppi",
            f"{JER_TAG}_MC_PtResolution_AK8PFPuppi",
            f"{JER_TAG}_MC_ScaleFactor_AK8PFPuppi",
            f"{JER_TAG}_MC_SFUncertainty_AK8PFPuppi",
        ),
        (
            f"{JEC_TAG}_MC_L1L2L3Res_AK8PFPuppi",
            f"{JEC_TAG}_DATA_L1L2L3Res_AK8PFPuppi",
        ),
    ),
    Payload("JME", "jetid.json.gz", "2ae07e7b7c67332fa4ccc374c7b3cfe667c653ce65c3f421f956460e46b1eb35", ("AK4PUPPI_TightLeptonVeto", "AK8PUPPI_TightLeptonVeto")),
    Payload("JME", "jetvetomaps.json.gz", "849cd1098814175ab58faac4834df0a353de63482f4017e22c0134ee7e840aec", ("Summer24Prompt24_RunBCDEFGHI_V1",)),
    Payload("EGM", "electron.json.gz", "fbdf9e241414f98cdc3185d65f83f2b329a8876f4fe69b1175b47ec673db6973", ("Electron-ID-SF",)),
    Payload("EGM", "electronHlt.json.gz", "566ee3b86d174c0e961890bf730100b6422d39e55cbc9b5164eda50eee60e856", ("Electron-HLT-SF", "Electron-HLT-DataEff", "Electron-HLT-McEff")),
    Payload("EGM", "electronSS_EtDependent.json.gz", "b464129d2c0c3974618d0d0dbda7d1bdff73641a0b8d04c2ac4164180ae5ed67", ("SmearAndSyst",), ("Scale",)),
    Payload("EGM", "photon.json.gz", "2b7a9667078bbead1a811aac3d5f950e788301df6e18f3e4b817c48963d87988", ("Photon-ID-SF", "Photon-CSEV-SF", "Photon-PixVeto-SF")),
    Payload("EGM", "photonSS_EtDependent.json.gz", "49dd136db56ba99874cb0d0bb2d46317689de7f28e7b7fdf0b4c8a1021ab9fac", ("SmearAndSyst",), ("Scale",)),
    Payload("MUO", "muon_Z.json.gz", "a103af495ff5d7c96d4d2b3dd50927fc942484bffbb392553ab059fddeded37f", (
        "NUM_LooseID_DEN_TrackerMuons",
        "NUM_MediumID_DEN_TrackerMuons",
        "NUM_LooseMiniIso_DEN_LooseID",
        "NUM_LooseMiniIso_DEN_MediumID",
        "NUM_IsoMu24_or_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdMedium_and_PFIsoMedium",
    )),
    Payload("MUO", "muon_HighPt.json.gz", "a1b199422bf0e789641ee1d2569e8a369cc4ff5380c302a84cf745e83bd3752e", ("NUM_HighPtID_DEN_GlobalMuonProbes",)),
    Payload("MUO", "muon_scalesmearing.json.gz", "d9ca1368ae0bd09535d8fa8ea90df07242cee4d8e9ca9279c0032657624fa041", ("m_data", "a_data", "m_mc", "a_mc", "k_data", "k_mc", "RandomSmearing", "cb_params", "poly_params")),
    Payload("BTV", "btagging.json.gz", "cac3a896a0f220fc748345aaa247b890bdf2b48ca998d244716cafcc5c9c2a33", ("UParTAK4_wp_values", "UParTAK4_comb", "UParTAK4_light")),
    Payload("TAU", "tau.json.gz", "2f36d1ff0df7bce33e3bddd5f01177961393bbb1f6a43b5f22e9c7d5a5caadf3", ("DeepTau2018v2p5VSe", "DeepTau2018v2p5VSmu", "DeepTau2018v2p5VSjet", "tau_energy_scale")),
    Payload("LUM", "puWeights_BCDEFGHI.json.gz", "8e196122cf6ea4f873678ef0f5b814d4d1fd8ed168f824f0857d8ece0337cead", ("Collisions24_BCDEFGHI_goldenJSON",)),
)


JES_SOURCES = {
    "jesTotal": "Total",
    "jesRegroupedTotal": "Regrouped_Total",
    "jesFlavorQCD": "Regrouped_FlavorQCD",
    "jesRelativeBal": "Regrouped_RelativeBal",
    "jesHF": "Regrouped_HF",
    "jesBBEC1": "Regrouped_BBEC1",
    "jesEC2": "Regrouped_EC2",
    "jesAbsolute": "Regrouped_Absolute",
    "jesAbsolute2024": "Regrouped_Absolute_2024",
    "jesHF2024": "Regrouped_HF_2024",
    "jesEC22024": "Regrouped_EC2_2024",
    "jesRelativeSample2024": "Regrouped_RelativeSample_2024",
    "jesBBEC12024": "Regrouped_BBEC1_2024",
}
OBJECT_SHAPE_VARIATIONS = (
    "electronScaleUp", "electronScaleDown",
    "electronSmearUp", "electronSmearDown",
    "photonScaleUp", "photonScaleDown",
    "photonSmearUp", "photonSmearDown",
    "muonScaleUp", "muonScaleDown",
    "muonResolutionUp", "muonResolutionDown",
    "tauEnergyScaleUp", "tauEnergyScaleDown",
)
SHAPE_VARIATIONS = (
    "nominal",
    "jerUp",
    "jerDown",
    "metUnclusteredUp",
    "metUnclusteredDown",
    *(f"{name}{direction}" for name in JES_SOURCES for direction in ("Up", "Down")),
    *OBJECT_SHAPE_VARIATIONS,
)


REQUIRED_BRANCHES = {
    "event": ("run", "luminosityBlock", "event", "Rho_fixedGridRhoFastjetAll"),
    "ak4": (
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_rawFactor", "Jet_area",
        "Jet_muonSubtrFactor", "Jet_chEmEF", "Jet_neEmEF", "Jet_btagUParTAK4B",
    ),
    "ak8": (
        "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass",
        "FatJet_rawFactor", "FatJet_area", "FatJet_msoftdrop",
    ),
    "mc_matching": (
        "Jet_hadronFlavour", "Jet_genJetIdx", "FatJet_genJetAK8Idx",
        "GenJet_pt", "GenJet_eta", "GenJet_phi", "GenJet_mass",
        "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi", "GenJetAK8_mass",
        "Tau_genPartFlav",
    ),
    "met": (
        "PuppiMET_pt", "PuppiMET_phi",
        "PuppiMET_ptUnclusteredUp", "PuppiMET_phiUnclusteredUp",
        "PuppiMET_ptUnclusteredDown", "PuppiMET_phiUnclusteredDown",
    ),
    "electron": (
        "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_phi",
        "Electron_mass", "Electron_charge", "Electron_cutBased",
        "Electron_miniPFRelIso_all", "Electron_r9", "Electron_seedGain",
    ),
    "muon": (
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Muon_looseId", "Muon_mediumId", "Muon_miniPFRelIso_all",
        "Muon_nTrackerLayers",
    ),
    "photon": (
        "Photon_pt", "Photon_eta", "Photon_phi", "Photon_cutBased",
        "Photon_r9", "Photon_seedGain", "Photon_electronVeto", "Photon_pixelSeed",
    ),
    "tau": (
        "Tau_pt", "Tau_eta", "Tau_phi", "Tau_mass", "Tau_dz", "Tau_decayMode",
        "Tau_idDeepTau2018v2p5VSe", "Tau_idDeepTau2018v2p5VSmu",
        "Tau_idDeepTau2018v2p5VSjet",
    ),
    "lowdm": (
        "FatJet_subJetIdx1", "FatJet_subJetIdx2",
        "SubJet_btagUParTAK4B",
        "FatJet_globalParT3_withMassTopvsQCD",
        "FatJet_globalParT3_withMassWvsQCD",
        "SV_pt", "SV_eta", "SV_phi", "SV_dxy",
        "SV_dlenSig", "SV_pAngle", "SV_ntracks",
    ),
}


EXTERNAL_FINAL_WEIGHT_DEPENDENCIES = {
    "met_trigger": "No analysis-specific 2024 MET trigger efficiency/SF payload has been supplied.",
    "photon_trigger": "No analysis-specific 2024 Photon175/Photon200 OR trigger efficiency/SF payload has been supplied.",
    "top_tag": "Run-3 data/MC top-tag SF payload and decorrelation prescription have not been supplied.",
    "w_tag": "Run-3 data/MC W-tag SF payload and decorrelation prescription have not been supplied.",
    "soft_sv": "Run-3 soft-SV tagging/veto efficiency and SF payload have not been supplied.",
    "jms_jmr": "No 2024 JMS/JMR payload is published in this correction-era bundle.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_path(root: Path, payload: Payload) -> Path:
    bundled = root / payload.relative
    return bundled if bundled.is_file() else payload.source


@lru_cache(maxsize=None)
def _load_correction_set(path: str) -> correctionlib.CorrectionSet:
    return correctionlib.CorrectionSet.from_file(path)


@lru_cache(maxsize=4)
def validate_payloads(root: Path) -> dict[str, Any]:
    files = []
    errors = []
    for spec in PAYLOADS:
        path = payload_path(root, spec)
        item = {"group": spec.group, "filename": spec.filename, "path": str(path)}
        if not path.is_file():
            item["status"] = "missing"
            errors.append(f"missing {path}")
            files.append(item)
            continue
        actual = sha256(path)
        item["sha256"] = actual
        if actual != spec.sha256:
            item["status"] = "checksum_mismatch"
            errors.append(f"checksum mismatch {path}: {actual}")
            files.append(item)
            continue
        try:
            cset = _load_correction_set(str(path))
            missing = [name for name in spec.required_corrections if name not in cset]
            missing_compound = [name for name in spec.required_compounds if name not in cset.compound]
        except Exception as exc:
            item["status"] = "invalid_correctionlib"
            item["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"invalid correctionlib payload {path}: {exc}")
        else:
            if missing or missing_compound:
                item["status"] = "missing_keys"
                item["missing_corrections"] = missing
                item["missing_compounds"] = missing_compound
                errors.append(f"missing correction keys in {path}")
            else:
                item["status"] = "valid"
        files.append(item)
    return {
        "status": "valid" if not errors else "invalid",
        "era": ERA,
        "jec_tag": JEC_TAG,
        "jer_tag": JER_TAG,
        "files": files,
        "errors": errors,
    }


def branch_audit(branches: set[str], is_data: bool) -> dict[str, Any]:
    groups = {}
    missing_required = []
    for group, names in REQUIRED_BRANCHES.items():
        if group == "mc_matching" and is_data:
            continue
        missing = sorted(set(names) - branches)
        groups[group] = {"required": len(names), "missing": missing}
        missing_required.extend(missing)
    return {
        "status": "valid" if not missing_required else "invalid",
        "groups": groups,
        "missing_required": sorted(set(missing_required)),
    }


def validate_shift(shift: str | None) -> str:
    normalized = str(shift or "nominal")
    if normalized not in SHAPE_VARIATIONS:
        raise ValueError(f"Unsupported 2024 object shift {normalized!r}")
    return normalized


def _flatten(values: Any) -> tuple[np.ndarray, np.ndarray]:
    counts = ak.to_numpy(ak.num(values, axis=1))
    return ak.to_numpy(ak.flatten(values, axis=1)), counts


def _broadcast_flat(event_values: Any, like: Any) -> np.ndarray:
    return ak.to_numpy(ak.flatten(ak.broadcast_arrays(like, event_values)[1], axis=1))


def _evaluate_jagged(correction: Any, like: Any, *inputs: Any) -> Any:
    _, counts = _flatten(like)
    flat_inputs = []
    for value in inputs:
        if isinstance(value, (str, bytes)) or np.isscalar(value):
            flat_inputs.append(value)
            continue
        try:
            is_jagged = ak.ndim(value) > 1
        except Exception:
            is_jagged = False
        flat_inputs.append(_flatten(value)[0] if is_jagged else _broadcast_flat(value, like))
    if int(np.sum(counts)) == 0:
        return ak.unflatten(np.asarray([], dtype=float), counts)
    return ak.unflatten(np.asarray(correction.evaluate(*flat_inputs), dtype=float), counts)


def _deterministic_normal(run: int, lumi: int, event: int, prefix: str, index: int) -> float:
    token = f"{run}:{lumi}:{event}:{prefix}:{index}".encode()
    seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return float(np.random.default_rng(seed).standard_normal())


def _deterministic_normal_jagged(like: Any, run: Any, lumi: Any, event: Any, prefix: str) -> Any:
    counts = ak.to_list(ak.num(like, axis=1))
    out = []
    for iev, count in enumerate(counts):
        out.append([
            _deterministic_normal(int(run[iev]), int(lumi[iev]), int(event[iev]), prefix, index)
            for index in range(int(count))
        ])
    return ak.Array(out)


def _valid_scaled_pt(original: Any, candidate: Any, valid: Any) -> Any:
    ratio = candidate / ak.where(original > 0, original, 1.0)
    usable = valid & np.isfinite(candidate) & (candidate > 0) & (ratio >= 0.1) & (ratio <= 2.0)
    return ak.where(usable, candidate, original)


def _calibrate_egm_collection(
    arrays: Any,
    prefix: str,
    is_data: bool,
    shift: str,
    root: Path,
) -> tuple[Any, Any | None, dict[str, Any]]:
    payload_name = "electronSS_EtDependent.json.gz" if prefix == "Electron" else "photonSS_EtDependent.json.gz"
    spec = next(item for item in PAYLOADS if item.filename == payload_name)
    cset = _load_correction_set(str(payload_path(root, spec)))
    pt = arrays[f"{prefix}_pt"]
    eta_sc = arrays["Electron_eta"] + arrays["Electron_deltaEtaSC"] if prefix == "Electron" else arrays["Photon_eta"]
    r9 = arrays[f"{prefix}_r9"]
    seed_gain = ak.values_astype(arrays[f"{prefix}_seedGain"], np.float64)
    valid = (pt >= 20.0) & (abs(eta_sc) < 2.5) & np.isfinite(pt) & np.isfinite(eta_sc) & np.isfinite(r9)
    safe_pt = ak.where(valid, pt, 25.0)
    safe_eta_sc = ak.where(valid, eta_sc, 0.0)
    safe_r9 = ak.where(valid, r9, 0.95)
    safe_gain = ak.where(valid, seed_gain, 12.0)
    status = {
        "payload": payload_name,
        "prefix": prefix,
        "minimum_pt_gev": 20.0,
        "shift": shift,
    }
    if is_data:
        scale = _evaluate_jagged(
            cset.compound["Scale"], pt, "scale", arrays["run"], safe_eta_sc, safe_r9, safe_pt, safe_gain,
        )
        corrected_pt = _valid_scaled_pt(pt, pt * scale, valid)
        status.update({"mode": "data_scale", "correction": "Scale(scale,run,ScEta,r9,pt,seedGain)"})
    else:
        smear_syst = "smear"
        if shift == f"{prefix.lower()}SmearUp":
            smear_syst = "smear_up"
        elif shift == f"{prefix.lower()}SmearDown":
            smear_syst = "smear_down"
        width = _evaluate_jagged(cset["SmearAndSyst"], pt, smear_syst, safe_pt, safe_r9, safe_eta_sc)
        random_value = _deterministic_normal_jagged(
            pt, arrays["run"], arrays["luminosityBlock"], arrays["event"], prefix,
        )
        corrected_pt = pt * (1.0 + width * random_value)
        scale_syst = None
        if shift == f"{prefix.lower()}ScaleUp":
            scale_syst = "scale_up"
        elif shift == f"{prefix.lower()}ScaleDown":
            scale_syst = "scale_down"
        if scale_syst:
            scale = _evaluate_jagged(cset["SmearAndSyst"], pt, scale_syst, safe_pt, safe_r9, safe_eta_sc)
            corrected_pt = corrected_pt * scale
        corrected_pt = _valid_scaled_pt(pt, corrected_pt, valid)
        status.update({
            "mode": "mc_gaussian_smearing",
            "smearing_variation": smear_syst,
            "scale_variation": scale_syst or "nominal",
            "random_seed": "blake2b(run,lumi,event,object,index)",
        })
    mass = arrays["Electron_mass"] if prefix == "Electron" else None
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0) if mass is not None else None
    return corrected_pt, corrected_mass, status


def _crystal_ball_invcdf(uniform: Any, mean: Any, sigma: Any, alpha: Any, power: Any) -> Any:
    flat_u, counts = _flatten(uniform)
    if flat_u.size == 0:
        return ak.unflatten(np.asarray([], dtype=float), counts)
    m = _flatten(mean)[0]
    s = np.maximum(abs(_flatten(sigma)[0]), 1e-9)
    a = np.maximum(abs(_flatten(alpha)[0]), 1e-9)
    n = np.maximum(_flatten(power)[0], 1.000001)
    u = np.clip(flat_u, 1e-12, 1.0 - 1e-12)
    sqrt_pi_over_2 = np.sqrt(np.pi / 2.0)
    sqrt2 = np.sqrt(2.0)
    exponent = np.exp(-a * a / 2.0)
    c1 = n / a / (n - 1.0) * exponent
    d1 = 2.0 * sqrt_pi_over_2 * erf(a / sqrt2)
    c = (d1 + 2.0 * c1) / c1
    d = (d1 + 2.0 * c1) / 2.0
    norm = 1.0 / s / (d1 + 2.0 * c1)
    ns = norm * s
    nc = ns * c1
    f = 1.0 - a * a / n
    g = s * n / a

    def cdf(x: np.ndarray) -> np.ndarray:
        delta = (x - m) / s
        result = np.ones_like(delta)
        left_base = f - s * delta / g
        right_base = f + s * delta / g
        left = (delta < -a) & (left_base > 0)
        left_edge = (delta < -a) & ~left
        right = (delta > a) & (right_base > 0)
        right_edge = (delta > a) & ~right
        core = ~(left | left_edge | right | right_edge)
        result[left] = nc[left] / np.power(left_base[left], n[left] - 1.0)
        result[left_edge] = nc[left_edge]
        result[right] = nc[right] * (c[right] - np.power(right_base[right], 1.0 - n[right]))
        result[right_edge] = nc[right_edge] * c[right_edge]
        result[core] = ns[core] * (d[core] - sqrt_pi_over_2 * erf(-delta[core] / sqrt2))
        return result

    cdf_minus = cdf(m - a * s)
    cdf_plus = cdf(m + a * s)
    result = np.zeros_like(u)
    left = (u < cdf_minus) & (nc / u > 0)
    left_edge = (u < cdf_minus) & ~left
    right = (u > cdf_plus) & (c - u / nc > 0)
    right_edge = (u > cdf_plus) & ~right
    core = ~(left | left_edge | right | right_edge)
    result[left] = m[left] + g[left] * (f[left] - np.power(nc[left] / u[left], 1.0 / (n[left] - 1.0)))
    result[left_edge] = m[left_edge] + g[left_edge] * f[left_edge]
    result[right] = m[right] - g[right] * (
        f[right] - np.power(c[right] - u[right] / nc[right], -1.0 / (n[right] - 1.0))
    )
    result[right_edge] = m[right_edge] - g[right_edge] * f[right_edge]
    argument = (d[core] - u[core] / ns[core]) / sqrt_pi_over_2
    result[core] = m[core] - sqrt2 * s[core] * erfinv(np.clip(argument, -1.0, 1.0))
    return ak.unflatten(result, counts)


def _calibrate_muons(arrays: Any, is_data: bool, shift: str, root: Path) -> tuple[Any, Any, dict[str, Any]]:
    spec = next(item for item in PAYLOADS if item.filename == "muon_scalesmearing.json.gz")
    cset = _load_correction_set(str(payload_path(root, spec)))
    pt = arrays["Muon_pt"]
    eta = arrays["Muon_eta"]
    phi = arrays["Muon_phi"]
    charge = arrays["Muon_charge"]
    layers = ak.values_astype(arrays["Muon_nTrackerLayers"], np.float64)
    valid = (pt >= 26.0) & (pt <= 200.0) & (abs(eta) < 2.4) & np.isfinite(pt)
    safe_pt = ak.where(valid, pt, 50.0)
    safe_eta = ak.where(valid, eta, 0.0)
    safe_phi = ak.where(valid, phi, 0.0)
    safe_charge = ak.where(valid, charge, 1)
    safe_layers = ak.where(valid, layers, 10.0)
    sample = "data" if is_data else "mc"
    additive = _evaluate_jagged(cset[f"a_{sample}"], pt, safe_eta, safe_phi, "nom")
    multiplicative = _evaluate_jagged(cset[f"m_{sample}"], pt, safe_eta, safe_phi, "nom")
    scaled = 1.0 / (multiplicative / safe_pt + safe_charge * additive)
    scaled = _valid_scaled_pt(pt, scaled, valid)
    status = {
        "payload": "muon_scalesmearing.json.gz",
        "scale_formula": "1/(m/pt + charge*a)",
        "valid_pt_gev": [26.0, 200.0],
        "shift": shift,
    }
    if is_data:
        corrected_pt = scaled
        status["mode"] = "data_scale"
    else:
        p0 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 0)
        p1 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 1)
        p2 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 2)
        resolution = np.maximum(0.0, p0 + p1 * scaled + p2 * scaled * scaled)
        mean = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 0)
        sigma = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 1)
        power = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 2)
        alpha = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 3)
        uniform = _evaluate_jagged(
            cset["RandomSmearing"], pt, arrays["event"], arrays["luminosityBlock"], safe_phi,
        )
        random_value = _crystal_ball_invcdf(uniform, mean, sigma, alpha, power)
        k_data = _evaluate_jagged(cset["k_data"], pt, abs(safe_eta), "nom")
        k_mc = _evaluate_jagged(cset["k_mc"], pt, abs(safe_eta), "nom")
        residual_k = np.sqrt(np.maximum(k_data * k_data - k_mc * k_mc, 0.0))
        nominal = _valid_scaled_pt(pt, scaled * (1.0 + residual_k * resolution * random_value), valid)
        corrected_pt = nominal
        if shift in {"muonResolutionUp", "muonResolutionDown"}:
            k_unc = _evaluate_jagged(cset["k_mc"], pt, abs(safe_eta), "stat")
            response = (nominal / ak.where(scaled > 0, scaled, 1.0) - 1.0) / ak.where(k_mc > 0, k_mc, 1.0)
            varied_k = k_mc + k_unc if shift.endswith("Up") else k_mc - k_unc
            candidate = scaled * (1.0 + varied_k * response)
            corrected_pt = _valid_scaled_pt(pt, ak.where(k_mc > 0, candidate, nominal), valid)
        elif shift in {"muonScaleUp", "muonScaleDown"}:
            stat_a = _evaluate_jagged(cset["a_mc"], pt, safe_eta, safe_phi, "stat")
            stat_m = _evaluate_jagged(cset["m_mc"], pt, safe_eta, safe_phi, "stat")
            stat_rho = _evaluate_jagged(cset["m_mc"], pt, safe_eta, safe_phi, "rho_stat")
            variance = stat_m * stat_m / (nominal * nominal) + stat_a * stat_a
            variance = variance + 2.0 * safe_charge * stat_rho * stat_m / nominal * stat_a
            uncertainty = nominal * nominal * np.sqrt(np.maximum(variance, 0.0))
            candidate = nominal + uncertainty if shift.endswith("Up") else nominal - uncertainty
            corrected_pt = _valid_scaled_pt(pt, candidate, valid)
        status.update({
            "mode": "mc_scale_and_crystal_ball_resolution",
            "random_source": "correctionlib RandomSmearing(event,lumi,phi)",
        })
    mass = arrays["Muon_mass"]
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0)
    return corrected_pt, corrected_mass, status


def _calibrate_taus(arrays: Any, is_data: bool, shift: str, root: Path) -> tuple[Any, Any, dict[str, Any]]:
    pt = arrays["Tau_pt"]
    mass = arrays["Tau_mass"]
    status = {
        "payload": "tau.json.gz",
        "selection_wp": "DeepTau2018v2p5 VSjet Medium",
        "vse_wp_for_tes": "VVLoose (least restrictive payload category; baseline veto has no VSe cut)",
        "supported_decay_modes": [0, 1, 10, 11],
        "shift": shift,
    }
    if is_data:
        status["mode"] = "not_applicable_data"
        return pt, mass, status
    spec = next(item for item in PAYLOADS if item.filename == "tau.json.gz")
    cset = _load_correction_set(str(payload_path(root, spec)))
    eta = arrays["Tau_eta"]
    decay_mode = arrays["Tau_decayMode"]
    genmatch = arrays["Tau_genPartFlav"]
    valid_dm = (decay_mode == 0) | (decay_mode == 1) | (decay_mode == 10) | (decay_mode == 11)
    valid = (pt >= 20.0) & (abs(eta) < 2.5) & valid_dm & (genmatch >= 0) & (genmatch <= 6)
    safe_pt = ak.where(valid, pt, 25.0)
    safe_eta = ak.where(valid, eta, 0.0)
    safe_dm = ak.where(valid, decay_mode, 0)
    safe_genmatch = ak.where(valid, genmatch, 0)
    variation = "up" if shift == "tauEnergyScaleUp" else "down" if shift == "tauEnergyScaleDown" else "nom"
    tes = _evaluate_jagged(
        cset["tau_energy_scale"], pt, safe_pt, safe_eta, safe_dm, safe_genmatch,
        "DeepTau2018v2p5", "Medium", "VVLoose", variation,
    )
    corrected_pt = _valid_scaled_pt(pt, pt * tes, valid)
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0)
    status.update({"mode": "mc_tau_energy_scale", "variation": variation})
    return corrected_pt, corrected_mass, status


def _jer_smear(
    corrected_pt: Any,
    eta: Any,
    resolution: Any,
    scale_factor: Any,
    gen_index: Any,
    gen_pt: Any,
    run: np.ndarray,
    lumi: np.ndarray,
    event: np.ndarray,
    prefix: str,
) -> Any:
    out = []
    for iev, (pts, etas, resolutions, sfs, indices) in enumerate(
        zip(
            ak.to_list(corrected_pt),
            ak.to_list(eta),
            ak.to_list(resolution),
            ak.to_list(scale_factor),
            ak.to_list(gen_index),
        )
    ):
        gen_pts = ak.to_list(gen_pt[iev])
        event_out = []
        for ijet, (pt, _jet_eta, res, sf, index) in enumerate(zip(pts, etas, resolutions, sfs, indices)):
            pt = float(pt)
            res = max(0.0, float(res))
            sf = max(0.0, float(sf))
            matched = 0 <= int(index) < len(gen_pts)
            if matched:
                matched_pt = float(gen_pts[int(index)])
                matched = abs(pt - matched_pt) < 3.0 * res * pt
            if matched:
                factor = 1.0 + (sf - 1.0) * (pt - matched_pt) / max(pt, 1e-6)
            else:
                random_value = _deterministic_normal(int(run[iev]), int(lumi[iev]), int(event[iev]), prefix, ijet)
                factor = 1.0 + random_value * res * math.sqrt(max(sf * sf - 1.0, 0.0))
            event_out.append(max(0.0, factor))
        out.append(event_out)
    return ak.Array(out)


def _jes_source(shift: str) -> tuple[str | None, float]:
    for public_name, source in JES_SOURCES.items():
        if shift == f"{public_name}Up":
            return source, 1.0
        if shift == f"{public_name}Down":
            return source, -1.0
    return None, 0.0


def _calibrate_jet_collection(
    arrays: Any,
    prefix: str,
    gen_prefix: str,
    radius: str,
    is_data: bool,
    shift: str,
    root: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    payload_name = "fatJet_jerc.json.gz" if prefix == "FatJet" else "jet_jerc.json.gz"
    spec = next(item for item in PAYLOADS if item.filename == payload_name)
    cset = _load_correction_set(str(payload_path(root, spec)))
    pt_nano = arrays[f"{prefix}_pt"]
    mass_nano = arrays[f"{prefix}_mass"]
    raw_factor = arrays[f"{prefix}_rawFactor"]
    eta = arrays[f"{prefix}_eta"]
    phi = arrays[f"{prefix}_phi"]
    area = arrays[f"{prefix}_area"]
    rho = arrays["Rho_fixedGridRhoFastjetAll"]
    run = np.asarray(arrays["run"])
    raw_pt = pt_nano * (1.0 - raw_factor)
    raw_mass = mass_nano * (1.0 - raw_factor)
    tag_kind = "DATA" if is_data else "MC"
    compound = cset.compound[f"{JEC_TAG}_{tag_kind}_L1L2L3Res_{radius}"]
    args = (area, eta, raw_pt, rho, phi, arrays["run"]) if is_data else (area, eta, raw_pt, rho, phi)
    correction = _evaluate_jagged(compound, raw_pt, *args)
    corrected_pt = raw_pt * correction
    corrected_mass = raw_mass * correction
    status = {
        "prefix": prefix,
        "jec": f"{JEC_TAG}_{tag_kind}_L1L2L3Res_{radius}",
        "jec_input": "NanoAOD pt/mass multiplied by (1-rawFactor)",
        "jer": "not_applicable_data" if is_data else JER_TAG,
        "shift": shift,
    }
    if is_data:
        if shift != "nominal":
            raise RuntimeError(f"Object shape shift {shift} requested for data")
        return corrected_pt, corrected_mass, status

    resolution = _evaluate_jagged(cset[f"{JER_TAG}_MC_PtResolution_{radius}"], corrected_pt, eta, corrected_pt, rho)
    sf_nom = _evaluate_jagged(cset[f"{JER_TAG}_MC_ScaleFactor_{radius}"], corrected_pt, eta, corrected_pt)
    sf_unc = _evaluate_jagged(cset[f"{JER_TAG}_MC_SFUncertainty_{radius}"], corrected_pt, eta, corrected_pt)
    sf = sf_nom + sf_unc if shift == "jerUp" else sf_nom - sf_unc if shift == "jerDown" else sf_nom
    smear = _jer_smear(
        corrected_pt,
        eta,
        resolution,
        sf,
        arrays[f"{prefix}_genJetIdx" if prefix == "Jet" else "FatJet_genJetAK8Idx"],
        arrays[f"{gen_prefix}_pt"],
        run,
        np.asarray(arrays["luminosityBlock"]),
        np.asarray(arrays["event"]),
        prefix,
    )
    corrected_pt = corrected_pt * smear
    corrected_mass = corrected_mass * smear
    source, direction = _jes_source(shift)
    if source is not None:
        uncertainty = _evaluate_jagged(cset[f"{JEC_TAG}_MC_{source}_{radius}"], corrected_pt, eta, corrected_pt)
        factor = 1.0 + direction * uncertainty
        corrected_pt = corrected_pt * factor
        corrected_mass = corrected_mass * factor
        status["jes_source"] = source
    return corrected_pt, corrected_mass, status


def calibrate_jets_and_met(arrays: Any, is_data: bool, shift: str, root: Path) -> tuple[Any, dict[str, Any]]:
    shift = validate_shift(shift)
    ak4_pt, ak4_mass, ak4_status = _calibrate_jet_collection(arrays, "Jet", "GenJet", "AK4PFPuppi", is_data, shift, root)
    ak8_pt, ak8_mass, ak8_status = _calibrate_jet_collection(arrays, "FatJet", "GenJetAK8", "AK8PFPuppi", is_data, shift, root)
    electron_pt, electron_mass, electron_status = _calibrate_egm_collection(arrays, "Electron", is_data, shift, root)
    photon_pt, _, photon_status = _calibrate_egm_collection(arrays, "Photon", is_data, shift, root)
    muon_pt, muon_mass, muon_status = _calibrate_muons(arrays, is_data, shift, root)
    tau_pt, tau_mass, tau_status = _calibrate_taus(arrays, is_data, shift, root)
    met_suffix = ""
    if shift == "metUnclusteredUp":
        met_suffix = "UnclusteredUp"
    elif shift == "metUnclusteredDown":
        met_suffix = "UnclusteredDown"
    met_pt = np.asarray(arrays[f"PuppiMET_pt{met_suffix}"], dtype=float)
    met_phi = np.asarray(arrays[f"PuppiMET_phi{met_suffix}"], dtype=float)
    met_px = met_pt * np.cos(met_phi)
    met_py = met_pt * np.sin(met_phi)
    type1_mask = (
        (ak4_pt > 15.0)
        & (abs(arrays["Jet_eta"]) < 5.2)
        & ((arrays["Jet_chEmEF"] + arrays["Jet_neEmEF"]) < 0.9)
    )
    muon_fraction = arrays["Jet_muonSubtrFactor"]
    delta_pt = (arrays["Jet_pt"] - ak4_pt) * (1.0 - muon_fraction)
    met_px = met_px + ak.to_numpy(ak.sum(ak.where(type1_mask, delta_pt * np.cos(arrays["Jet_phi"]), 0.0), axis=1))
    met_py = met_py + ak.to_numpy(ak.sum(ak.where(type1_mask, delta_pt * np.sin(arrays["Jet_phi"]), 0.0), axis=1))

    for prefix, corrected_pt in (
        ("Electron", electron_pt),
        ("Muon", muon_pt),
        ("Photon", photon_pt),
        ("Tau", tau_pt),
    ):
        object_delta_pt = arrays[f"{prefix}_pt"] - corrected_pt
        met_px = met_px + ak.to_numpy(ak.sum(object_delta_pt * np.cos(arrays[f"{prefix}_phi"]), axis=1))
        met_py = met_py + ak.to_numpy(ak.sum(object_delta_pt * np.sin(arrays[f"{prefix}_phi"]), axis=1))
    calibrated = arrays
    for name, value in (
        ("Jet_pt", ak4_pt),
        ("Jet_mass", ak4_mass),
        ("FatJet_pt", ak8_pt),
        ("FatJet_mass", ak8_mass),
        ("Electron_pt", electron_pt),
        ("Electron_mass", electron_mass),
        ("Muon_pt", muon_pt),
        ("Muon_mass", muon_mass),
        ("Photon_pt", photon_pt),
        ("Tau_pt", tau_pt),
        ("Tau_mass", tau_mass),
        ("PuppiMET_pt", np.hypot(met_px, met_py)),
        ("PuppiMET_phi", np.arctan2(met_py, met_px)),
    ):
        calibrated = ak.with_field(calibrated, value, name)
    return calibrated, {
        "status": "applied",
        "shift": shift,
        "ak4": ak4_status,
        "ak8": ak8_status,
        "electron": electron_status,
        "photon": photon_status,
        "muon": muon_status,
        "tau": tau_status,
        "met": {
            "base": f"NanoAOD PuppiMET{met_suffix}",
            "propagation": "AK4 V5/JRV2 type-1 delta plus electron, muon, photon, and tau momentum deltas",
        },
    }


def _reused_status(status: dict[str, Any], shift: str) -> dict[str, Any]:
    reused = dict(status)
    reused["shift"] = shift
    reused["reused_from_shift"] = "nominal"
    return reused


def _assemble_calibrated_objects_and_met(
    arrays: Any,
    *,
    shift: str,
    ak4_pt: Any,
    ak4_mass: Any,
    ak8_pt: Any,
    ak8_mass: Any,
    electron_pt: Any,
    electron_mass: Any,
    photon_pt: Any,
    muon_pt: Any,
    muon_mass: Any,
    tau_pt: Any,
    tau_mass: Any,
) -> tuple[Any, dict[str, Any]]:
    met_suffix = ""
    if shift == "metUnclusteredUp":
        met_suffix = "UnclusteredUp"
    elif shift == "metUnclusteredDown":
        met_suffix = "UnclusteredDown"
    met_pt = np.asarray(arrays[f"PuppiMET_pt{met_suffix}"], dtype=float)
    met_phi = np.asarray(arrays[f"PuppiMET_phi{met_suffix}"], dtype=float)
    met_px = met_pt * np.cos(met_phi)
    met_py = met_pt * np.sin(met_phi)
    type1_mask = (
        (ak4_pt > 15.0)
        & (abs(arrays["Jet_eta"]) < 5.2)
        & ((arrays["Jet_chEmEF"] + arrays["Jet_neEmEF"]) < 0.9)
    )
    muon_fraction = arrays["Jet_muonSubtrFactor"]
    delta_pt = (arrays["Jet_pt"] - ak4_pt) * (1.0 - muon_fraction)
    met_px = met_px + ak.to_numpy(
        ak.sum(ak.where(type1_mask, delta_pt * np.cos(arrays["Jet_phi"]), 0.0), axis=1)
    )
    met_py = met_py + ak.to_numpy(
        ak.sum(ak.where(type1_mask, delta_pt * np.sin(arrays["Jet_phi"]), 0.0), axis=1)
    )
    for prefix, corrected_pt in (
        ("Electron", electron_pt),
        ("Muon", muon_pt),
        ("Photon", photon_pt),
        ("Tau", tau_pt),
    ):
        object_delta_pt = arrays[f"{prefix}_pt"] - corrected_pt
        met_px = met_px + ak.to_numpy(
            ak.sum(object_delta_pt * np.cos(arrays[f"{prefix}_phi"]), axis=1)
        )
        met_py = met_py + ak.to_numpy(
            ak.sum(object_delta_pt * np.sin(arrays[f"{prefix}_phi"]), axis=1)
        )
    calibrated = arrays
    for name, value in (
        ("Jet_pt", ak4_pt),
        ("Jet_mass", ak4_mass),
        ("FatJet_pt", ak8_pt),
        ("FatJet_mass", ak8_mass),
        ("Electron_pt", electron_pt),
        ("Electron_mass", electron_mass),
        ("Muon_pt", muon_pt),
        ("Muon_mass", muon_mass),
        ("Photon_pt", photon_pt),
        ("Tau_pt", tau_pt),
        ("Tau_mass", tau_mass),
        ("PuppiMET_pt", np.hypot(met_px, met_py)),
        ("PuppiMET_phi", np.arctan2(met_py, met_px)),
    ):
        calibrated = ak.with_field(calibrated, value, name)
    return calibrated, {
        "base": f"NanoAOD PuppiMET{met_suffix}",
        "propagation": "AK4 V5/JRV2 type-1 delta plus electron, muon, photon, and tau momentum deltas",
    }


def iter_calibrated_variations(
    arrays: Any,
    *,
    is_data: bool,
    shifts: tuple[str, ...] | list[str],
    root: Path,
):
    """Yield one shifted chunk at a time while retaining only the nominal correction cache."""
    selected = tuple(validate_shift(item) for item in shifts)
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate shifts in calibration bundle")
    if is_data and any(item != "nominal" for item in selected):
        raise RuntimeError("Object shape shifts cannot be evaluated for data")

    nominal_arrays, nominal_status = calibrate_jets_and_met(arrays, is_data, "nominal", root)
    for shift in selected:
        if shift == "nominal":
            yield shift, (nominal_arrays, nominal_status)
            continue

        ak4_pt = nominal_arrays["Jet_pt"]
        ak4_mass = nominal_arrays["Jet_mass"]
        ak8_pt = nominal_arrays["FatJet_pt"]
        ak8_mass = nominal_arrays["FatJet_mass"]
        electron_pt = nominal_arrays["Electron_pt"]
        electron_mass = nominal_arrays["Electron_mass"]
        photon_pt = nominal_arrays["Photon_pt"]
        muon_pt = nominal_arrays["Muon_pt"]
        muon_mass = nominal_arrays["Muon_mass"]
        tau_pt = nominal_arrays["Tau_pt"]
        tau_mass = nominal_arrays["Tau_mass"]

        is_jme_shift = shift in {"jerUp", "jerDown"} or _jes_source(shift)[0] is not None
        if is_jme_shift:
            ak4_pt, ak4_mass, ak4_status = _calibrate_jet_collection(
                arrays, "Jet", "GenJet", "AK4PFPuppi", is_data, shift, root
            )
            ak8_pt, ak8_mass, ak8_status = _calibrate_jet_collection(
                arrays, "FatJet", "GenJetAK8", "AK8PFPuppi", is_data, shift, root
            )
        else:
            ak4_status = _reused_status(nominal_status["ak4"], shift)
            ak8_status = _reused_status(nominal_status["ak8"], shift)

        if shift.startswith("electron"):
            electron_pt, electron_mass, electron_status = _calibrate_egm_collection(
                arrays, "Electron", is_data, shift, root
            )
        else:
            electron_status = _reused_status(nominal_status["electron"], shift)
        if shift.startswith("photon"):
            photon_pt, _, photon_status = _calibrate_egm_collection(
                arrays, "Photon", is_data, shift, root
            )
        else:
            photon_status = _reused_status(nominal_status["photon"], shift)
        if shift.startswith("muon"):
            muon_pt, muon_mass, muon_status = _calibrate_muons(
                arrays, is_data, shift, root
            )
        else:
            muon_status = _reused_status(nominal_status["muon"], shift)
        if shift.startswith("tau"):
            tau_pt, tau_mass, tau_status = _calibrate_taus(
                arrays, is_data, shift, root
            )
        else:
            tau_status = _reused_status(nominal_status["tau"], shift)

        calibrated, met_status = _assemble_calibrated_objects_and_met(
            arrays,
            shift=shift,
            ak4_pt=ak4_pt,
            ak4_mass=ak4_mass,
            ak8_pt=ak8_pt,
            ak8_mass=ak8_mass,
            electron_pt=electron_pt,
            electron_mass=electron_mass,
            photon_pt=photon_pt,
            muon_pt=muon_pt,
            muon_mass=muon_mass,
            tau_pt=tau_pt,
            tau_mass=tau_mass,
        )
        yield shift, (
            calibrated,
            {
                "status": "applied",
                "shift": shift,
                "cache_policy": "nominal calibration reused for unaffected object families",
                "ak4": ak4_status,
                "ak8": ak8_status,
                "electron": electron_status,
                "photon": photon_status,
                "muon": muon_status,
                "tau": tau_status,
                "met": met_status,
            },
        )


def calibrate_variation_bundle(
    arrays: Any,
    *,
    is_data: bool,
    shifts: tuple[str, ...] | list[str],
    root: Path,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Materialize the streaming calibration iterator; intended for compact tests only."""
    return dict(
        iter_calibrated_variations(
            arrays,
            is_data=is_data,
            shifts=shifts,
            root=root,
        )
    )


def manifest() -> dict[str, Any]:
    return {
        "schema_version": "object_corrections_2024_v1",
        "era": ERA,
        "jec_tag": JEC_TAG,
        "jer_tag": JER_TAG,
        "payloads": [
            {
                "group": item.group,
                "filename": item.filename,
                "sha256": item.sha256,
                "source": str(item.source),
                "required_corrections": list(item.required_corrections),
                "required_compounds": list(item.required_compounds),
            }
            for item in PAYLOADS
        ],
        "shape_variations": list(SHAPE_VARIATIONS),
        "nominal_kinematic_corrections": {
            "ak4_ak8": "Summer24Prompt24 V5 JEC and JRV2 JER",
            "electron_photon": "EGM data scale or deterministic MC Gaussian smearing",
            "muon": "MuonScaRe scale and MC Crystal Ball resolution for 26 <= pt <= 200 GeV",
            "tau": "DeepTau2018v2p5 VSjet Medium tau energy scale in MC",
            "met": "PuppiMET propagation from all corrected object momenta",
        },
        "required_branches": {key: list(value) for key, value in REQUIRED_BRANCHES.items()},
        "external_final_weight_dependencies": EXTERNAL_FINAL_WEIGHT_DEPENDENCIES,
        "dy_recoil_policy": {
            "definition": "uT vector = PuppiMET vector + both selected dilepton vectors",
            "threshold_gev": 250.0,
            "opening_angle_reference": "channel-specific uT phi after electron or muon jet cleaning",
        },
    }


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=2, sort_keys=True) + "\n")
