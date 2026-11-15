"""Retained-event features from calibrated NanoAOD signal arrays.

This is the existing event extraction. Scale factors, normalization, final
region assignment and GNN binning remain in the canonical downstream stages.
"""
from __future__ import annotations
import re
import math
from pathlib import Path
from typing import Any
import awkward as ak
import numpy as np
from .paths import ROOT, analysis_era, tagging_definition, muon_pt_min
from .signal_models import signal_genmodel_branch, signal_mass_from_genmodel, signal_topology_id
from .ids import clean_by_delta_r
from .nanoaod_calibration import jet_id, jet_veto


def configure_era(era):
    global FILTERS
    global CURRENT_ERA, AK4_BTAG_BRANCH, AK4_BTAG_OUTPUT_FIELD
    global BOOSTED_TOP_SCORE_BRANCH, BOOSTED_W_SCORE_BRANCH
    global BOOSTED_TOP_SCORE_WP, BOOSTED_W_SCORE_WP
    global BOOSTED_SCORE_MAX
    global UPART_AK4_LOOSE_WP, UPART_AK4_MEDIUM_WP
    CURRENT_ERA = analysis_era(era)
    definition = tagging_definition(era)
    early = CURRENT_ERA in ('2022', '2022EE', '2023', '2023BPix')
    FILTERS = ['Flag_goodVertices', 'Flag_globalSuperTightHalo2016Filter',
               'Flag_EcalDeadCellTriggerPrimitiveFilter', 'Flag_BadPFMuonFilter',
               'Flag_BadPFMuonDzFilter', 'Flag_eeBadScFilter', 'Flag_ecalBadCalibFilter']
    FILTERS += (['Flag_hfNoisyHitsFilter'] if early else
                ['Flag_HBHENoiseFilter', 'Flag_HBHENoiseIsoFilter'])
    BOOSTED_SCORE_MAX = 1.0 if early else float('inf')
    AK4_BTAG_BRANCH = 'Jet_btagPNetB' if early else 'Jet_btagUParTAK4B'
    AK4_BTAG_OUTPUT_FIELD = 'good_jet_btag_pnet' if early else 'good_jet_btag_upart'
    BOOSTED_TOP_SCORE_BRANCH = 'FatJet_particleNetWithMass_TvsQCD' if early else 'FatJet_globalParT3_withMassTopvsQCD'
    BOOSTED_W_SCORE_BRANCH = 'FatJet_particleNetWithMass_WvsQCD' if early else 'FatJet_globalParT3_withMassWvsQCD'
    BOOSTED_TOP_SCORE_WP, BOOSTED_W_SCORE_WP = definition['top'], definition['w']
    UPART_AK4_LOOSE_WP, UPART_AK4_MEDIUM_WP = definition['loose'], definition['medium']


def campaign_year(year):
    return analysis_era(year)


analysis_year = campaign_year


def validate_shift_name(shift):
    if shift not in (None, '', 'nominal'):
        raise ValueError('Intermediate signal production is nominal only')
    return 'nominal'


def apply_jec(arrays, repo, year, process, prefix, pt, eta, phi, mass, shift):
    return pt, mass, {'applied': True, 'source': 'canonical_calibrated_objects'}


def shifted_met(arrays, n, shift, process):
    return np.asarray(arrays['PuppiMET_pt']), np.asarray(arrays['PuppiMET_phi']), {'applied': True}


def golden_lumi_mask(arrays, process, repo, n, year):
    if is_data_process(process):
        raise ValueError('This production stage accepts signal MC only')
    return np.ones(n, dtype=bool), 'not_applicable_mc'


def ak4_tight_lepton_veto_mask(arrays, pt, eta, repo, year=None):
    return jet_id(arrays, 'Jet'), 'correctionlib_AK4PUPPI_TightLeptonVeto'


def ak8_tight_lepton_veto_mask(arrays, pt, eta, repo):
    return jet_id(arrays, 'FatJet'), 'correctionlib_AK8PUPPI_TightLeptonVeto'


def ak4_jet_veto_mask(pt, eta, phi, repo, year=None):
    return jet_veto(pt, eta, phi, CURRENT_ERA), 'target_era_jet_veto_map'

REGION_NAMES = ['preselection', 'LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M', 'SR']

BOOSTED_TOP_SCORE_BRANCH = 'FatJet_globalParT3_withMassTopvsQCD'

BOOSTED_W_SCORE_BRANCH = 'FatJet_globalParT3_withMassWvsQCD'

BOOSTED_TOP_SCORE_WP = 0.5078

BOOSTED_W_SCORE_WP = 0.9385

BOOSTED_SCORE_MAX = float('inf')

AK4_BTAG_BRANCH = 'Jet_btagUParTAK4B'

AK4_BTAG_OUTPUT_FIELD = 'good_jet_btag_upart'

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

FILTERS = ['Flag_goodVertices', 'Flag_globalSuperTightHalo2016Filter', 'Flag_HBHENoiseFilter', 'Flag_HBHENoiseIsoFilter', 'Flag_EcalDeadCellTriggerPrimitiveFilter', 'Flag_BadPFMuonFilter', 'Flag_BadPFMuonDzFilter', 'Flag_eeBadScFilter', 'Flag_ecalBadCalibFilter']

SIGNAL_HLT = ['HLT_PFMET120_PFMHT120_IDTight', 'HLT_PFMET130_PFMHT130_IDTight', 'HLT_PFMET140_PFMHT140_IDTight', 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight', 'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight', 'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight']

PHOTON_HLT = ['HLT_Photon175', 'HLT_Photon200']

HT_REFERENCE_HLT = ['HLT_PFHT180', 'HLT_PFHT250', 'HLT_PFHT350', 'HLT_PFHT370', 'HLT_PFHT430', 'HLT_PFHT510', 'HLT_PFHT590', 'HLT_PFHT680', 'HLT_PFHT780', 'HLT_PFHT890', 'HLT_PFHT1050']

ELECTRON_HLT = ['HLT_Ele115_CaloIdVT_GsfTrkIdT', 'HLT_Ele135_CaloIdVT_GsfTrkIdT', 'HLT_Ele30_WPTight_Gsf', 'HLT_Ele32_WPTight_Gsf', 'HLT_Ele35_WPTight_Gsf', 'HLT_Ele38_WPTight_Gsf', 'HLT_Ele40_WPTight_Gsf', 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 'HLT_DoubleEle25_CaloIdL_MW', 'HLT_DoubleEle27_CaloIdL_MW', 'HLT_DoubleEle33_CaloIdL_MW']

ELECTRON_REFERENCE_HLT = ['HLT_Ele30_WPTight_Gsf', 'HLT_Ele32_WPTight_Gsf', 'HLT_Ele35_WPTight_Gsf', 'HLT_Ele38_WPTight_Gsf', 'HLT_Ele40_WPTight_Gsf']

MUON_HLT = ['HLT_IsoMu20', 'HLT_IsoMu24', 'HLT_IsoMu27', 'HLT_IsoMu24_eta2p1', 'HLT_Mu50', 'HLT_Mu55']

def np_filled(values: Any, n: int, fill: float=1.0) -> np.ndarray:
    try:
        out = np.asarray(ak.to_numpy(ak.fill_none(values, fill)), dtype=float)
    except Exception:
        out = np.asarray(values, dtype=float)
    if out.shape == ():
        out = np.full(n, float(out), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill).astype(float)

def finite_float(value: Any, fill: float=0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fill)
    return out if math.isfinite(out) else float(fill)

def finite_weight_array(values: Any, n: int, fill: float=0.0) -> np.ndarray:
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

def cumulative_cutflow_histogram(sequence: list[tuple[str, Any]], weights: Any) -> dict[str, Any]:
    finite_weights = finite_weight_array(weights, len(weights), 0.0)
    cumulative = np.ones(len(finite_weights), dtype=bool)
    labels: list[str] = []
    entries: list[int] = []
    raw_values: list[float] = []
    raw_sumw2: list[float] = []
    for cut_name, cut_mask in sequence:
        cumulative &= np.asarray(cut_mask, dtype=bool)
        selected_weights = finite_weights[cumulative]
        labels.append(str(cut_name))
        entries.append(int(np.sum(cumulative)))
        raw_values.append(finite_sum(selected_weights))
        raw_sumw2.append(finite_sum(selected_weights * selected_weights))
    return {'schema_version': 'cumulative_cutflow_histogram_v1', 'cumulative': True, 'labels': labels, 'bin_edges': [float(i) - 0.5 for i in range(len(labels) + 1)], 'entries': entries, 'raw_values': raw_values, 'raw_sumw2': raw_sumw2}

def is_data_process(process: str) -> bool:
    return process in {'JetMET', 'EGamma', 'Muon'}

def available(tree: Any) -> set[str]:
    return set(tree.keys())

def has_field(arrays: Any, name: str) -> bool:
    return name in getattr(arrays, 'fields', [])

def arr(arrays: dict[str, Any], name: str, default: Any=None) -> Any:
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
    return (out, missing)

def first_or(default: float, jagged: Any) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.firsts(jagged), default))

def nth_or(default: float, jagged: Any, idx: int) -> np.ndarray:
    padded = ak.pad_none(jagged, idx + 1, axis=1, clip=False)
    return ak.to_numpy(ak.fill_none(padded[:, idx], default))

def count(mask: Any) -> np.ndarray:
    return ak.to_numpy(ak.sum(mask, axis=1))

def delta_phi(phi1: Any, phi2: Any) -> Any:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))

def medium_photon_mask(pt: Any, eta: Any, cutbased: Any, electron_veto: Any) -> Any:
    fiducial = (abs(eta) < 1.4442) | (abs(eta) > 1.566) & (abs(eta) < 2.5)
    return (pt > 220) & fiducial & (cutbased >= PHOTON_MEDIUM_CUTBASED_MIN) & (electron_veto == 1)

def jet_feature_block(jet_pt: Any, jet_eta: Any, jet_phi: Any, good_mask: Any, b_mask: Any, met_phi: Any) -> dict[str, Any]:
    jphi = jet_phi[good_mask]
    jpt = jet_pt[good_mask]
    jeta = jet_eta[good_mask]
    dphis = delta_phi(jphi, met_phi)
    j1dphi = first_or(999, dphis)
    j2dphi = nth_or(999, dphis, 1)
    j3dphi = nth_or(999, dphis, 2)
    j4dphi = nth_or(999, dphis, 3)
    return {'njet': count(good_mask), 'nb': count(b_mask), 'ht': ak.to_numpy(ak.sum(jet_pt[good_mask], axis=1)), 'jpt': jpt, 'jeta': jeta, 'jphi': jphi, 'j1pt': first_or(-99, jpt), 'j1eta': first_or(-99, jeta), 'j1phi': first_or(-99, jphi), 'j2pt': nth_or(-99, jpt, 1), 'j1dphi': j1dphi, 'j2dphi': j2dphi, 'j3dphi': j3dphi, 'j4dphi': j4dphi, 'min_dphi4': np.minimum.reduce([j1dphi, j2dphi, j3dphi, j4dphi]), 'open_pre': (j1dphi > 0.5) & (j2dphi > 0.15) & (j3dphi > 0.15), 'open_high': (j1dphi > 0.5) & (j2dphi > 0.5) & (j3dphi > 0.5) & (j4dphi > 0.5), 'qcd_open': (j1dphi < 0.5) | (j2dphi < 0.5) | (j3dphi < 0.5) | (j4dphi < 0.5), 'dphi123_0p1': (j1dphi < 0.1) | (j2dphi < 0.1) | (j3dphi < 0.1)}

def transverse_mass(pt: Any, phi: Any, met_pt: Any, met_phi: Any) -> Any:
    return np.sqrt(2 * pt * met_pt * (1 - np.cos(phi - met_phi)))

def transverse_vector_sum(*pt_phi_pairs: Any) -> tuple[Any, Any]:
    px = 0.0
    py = 0.0
    for pt, phi in pt_phi_pairs:
        px = px + pt * np.cos(phi)
        py = py + pt * np.sin(phi)
    return (np.sqrt(np.maximum(0, px * px + py * py)), np.arctan2(py, px))

def first_fatjet_subjet_btag(fatjet_mask: Any, subjet_idx1: Any, subjet_idx2: Any, subjet_btag: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
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
    return (maximum, available)

def lowdm_kinematic_block(*, jet_pt: Any, jet_eta: Any, jet_phi: Any, jet_btag: Any, good_jet_mask: Any, loose_b_mask: Any, medium_b_mask: Any, fatjet_pt: Any, fatjet_eta: Any, fatjet_phi: Any, fatjet_id_mask: Any, fatjet_clean_mask: Any, boosted_top_mask: Any, boosted_w_mask: Any, fatjet_subjet_idx1: Any, fatjet_subjet_idx2: Any, subjet_btag: Any, recoil_pt: Any, recoil_phi: Any, n: int) -> dict[str, Any]:
    """Build the low-deltaM observables with one internally consistent recoil/object definition."""
    jets = jet_feature_block(jet_pt, jet_eta, jet_phi, good_jet_mask, medium_b_mask, recoil_phi)
    nb_medium = count(medium_b_mask)
    nb_loose = count(loose_b_mask)
    b_order = ak.argsort(jet_btag[medium_b_mask], axis=1, ascending=False)
    b_pt = jet_pt[medium_b_mask][b_order]
    b_phi = jet_phi[medium_b_mask][b_order]
    ptb = first_or(-99, b_pt)
    b_mtb = transverse_mass(b_pt, b_phi, recoil_pt, recoil_phi)
    mtb1 = first_or(999, b_mtb)
    mtb2 = nth_or(999, b_mtb, 1)
    mtb = np.where(nb_medium >= 2, np.minimum(mtb1, mtb2), np.where(nb_medium == 1, mtb1, -99.0))
    clean_fatjets = fatjet_id_mask & fatjet_clean_mask & (fatjet_pt > 200) & (abs(fatjet_eta) < 2.4)
    n_isr = count(clean_fatjets)
    isr_pt = first_or(-99, fatjet_pt[clean_fatjets])
    isr_eta = first_or(-99, fatjet_eta[clean_fatjets])
    isr_phi = first_or(-99, fatjet_phi[clean_fatjets])
    isr_dphi = np.asarray(delta_phi(isr_phi, recoil_phi), dtype=float)
    isr_dphi = np.where(isr_pt > 0, isr_dphi, -99.0)
    subjet_max, subjet_available = first_fatjet_subjet_btag(clean_fatjets, fatjet_subjet_idx1, fatjet_subjet_idx2, subjet_btag, n)
    pass_isr_bveto = subjet_available & (subjet_max < UPART_AK4_LOOSE_WP)
    pass_isr = (n_isr == 1) & (isr_dphi > 2.0)
    pass_topology = (count(boosted_top_mask & fatjet_clean_mask) == 0) & (count(boosted_w_mask & fatjet_clean_mask) == 0)
    met_sqrt_ht = np.divide(recoil_pt, np.sqrt(jets['ht']), out=np.full(n, -99.0, dtype=float), where=jets['ht'] > 0)
    return {'jets': jets, 'nb_medium': nb_medium, 'nb_loose': nb_loose, 'ptb': ptb, 'mtb': mtb, 'met_sqrt_ht': met_sqrt_ht, 'n_isr': n_isr, 'isr_pt': isr_pt, 'isr_eta': isr_eta, 'isr_phi': isr_phi, 'isr_dphi': isr_dphi, 'isr_subjet_btag_max': subjet_max, 'isr_subjet_bveto_available': subjet_available, 'pass_isr_bveto': pass_isr_bveto, 'pass_isr': pass_isr, 'pass_topology': pass_topology, 'pass_met_sqrt_ht': met_sqrt_ht >= 10.0, 'pass_mtb': (nb_medium == 0) | (mtb < 175.0), 'isr_mask': clean_fatjets}

def assign_lowdm_search_bin(njet: int, nb: int, nsv: int, pisr: float, ptb: float, met: float, mtb: float) -> int:
    del nsv, mtb

    def met_bin(value: float, edges: list[float]) -> int:
        for idx in range(len(edges) - 1):
            if edges[idx] <= value < edges[idx + 1]:
                return idx
        return len(edges) - 1 if value >= edges[-1] else -1
    idx = 0
    for nj_min, nj_max in [(2, 5), (6, None)]:
        if nb == 0 and njet >= nj_min and (nj_max is None or njet <= nj_max) and (pisr >= 500.0):
            mb = met_bin(met, [450.0, 550.0, 650.0, 750.0])
            return idx + mb if mb >= 0 else -1
        idx += 4
    for pisr_min, pisr_max, met_edges in [(300.0, 500.0, [300.0, 400.0, 500.0, 600.0]), (500.0, None, [450.0, 550.0, 650.0, 750.0])]:
        for ptb_min, ptb_max in [(20.0, 40.0), (40.0, 70.0)]:
            if nb == 1 and ptb_min < ptb < ptb_max and (pisr >= pisr_min) and (pisr_max is None or pisr < pisr_max):
                mb = met_bin(met, met_edges)
                return idx + mb if mb >= 0 else -1
            idx += 4
    nb2_blocks = [(300.0, 500.0, 40.0, 80.0, 2, None, [300.0, 400.0, 500.0]), (300.0, 500.0, 80.0, 140.0, 2, None, [300.0, 400.0, 500.0]), (300.0, 500.0, 140.0, None, 7, None, [300.0, 400.0, 500.0]), (500.0, None, 40.0, 80.0, 2, None, [450.0, 550.0, 650.0]), (500.0, None, 80.0, 140.0, 2, None, [450.0, 550.0, 650.0]), (300.0, None, 140.0, None, 7, None, [450.0, 550.0, 650.0])]
    for pisr_min, pisr_max, ptb_min, ptb_max, nj_min, nj_max, met_edges in nb2_blocks:
        if nb >= 2 and njet >= nj_min and (nj_max is None or njet <= nj_max) and (ptb > ptb_min) and (ptb_max is None or ptb < ptb_max) and (pisr >= pisr_min) and (pisr_max is None or pisr < pisr_max):
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

def extract_chunk(arrays: dict[str, Any], dataset: str, process: str, sp: str | None, year: str, file_path: str, entry_start: int, entry_stop: int, fastsim_trigger_bypass: bool=False, shift_name: str | None=None, compute_weights: bool=True, materialize_skim_flag: str | None=None, dy_mass_window: str='on_z') -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(arrays['run'])
    repo = ROOT
    year = campaign_year(year)
    correction_year = analysis_year(year)
    shift = validate_shift_name(shift_name)
    met_pt, met_phi, met_shift_status = shifted_met(arrays, n, shift, process)
    calo_pt = np.asarray(arr(arrays, 'CaloMET_pt', np.ones(n) * np.nan), dtype=float)
    puppi_calo = np.divide(met_pt, calo_pt, out=np.ones(n) * np.inf, where=calo_pt != 0)
    jet_pt_raw = arr(arrays, 'Jet_pt', ak.Array([[]] * n))
    jet_eta = arr(arrays, 'Jet_eta', ak.Array([[]] * n))
    jet_phi = arr(arrays, 'Jet_phi', ak.Array([[]] * n))
    jet_mass_raw = arr(arrays, 'Jet_mass', ak.zeros_like(jet_pt_raw))
    jet_pt, jet_mass, jec_status = apply_jec(arrays, repo, year, process, 'Jet', jet_pt_raw, jet_eta, jet_phi, jet_mass_raw, shift)
    jet_id_mask, jet_id_source = ak4_tight_lepton_veto_mask(arrays, jet_pt, jet_eta, repo, year)
    veto_j, jet_veto_source = ak4_jet_veto_mask(jet_pt, jet_eta, jet_phi, repo, year)
    zero_veto_j = count(veto_j) == 0
    btag = arr(arrays, AK4_BTAG_BRANCH, ak.zeros_like(jet_pt))
    good_j = (jet_pt > 30) & (abs(jet_eta) < 2.4) & jet_id_mask
    b_loose = good_j & (btag > UPART_AK4_LOOSE_WP)
    b_med = good_j & (btag > UPART_AK4_MEDIUM_WP)
    b_loose_lowdm = b_loose
    b_med_lowdm = b_med
    jet_nominal = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j, b_med, met_phi)
    njet = jet_nominal['njet']
    nb = jet_nominal['nb']
    nb_loose = count(b_loose)
    nb_medium_lowdm = nb
    nb_loose_lowdm = nb_loose
    ht = jet_nominal['ht']
    j1pt = jet_nominal['j1pt']
    j1eta = jet_nominal['j1eta']
    j1phi = jet_nominal['j1phi']
    j2pt = jet_nominal['j2pt']
    j1dphi = jet_nominal['j1dphi']
    j2dphi = jet_nominal['j2dphi']
    j3dphi = jet_nominal['j3dphi']
    j4dphi = jet_nominal['j4dphi']
    min_dphi4 = jet_nominal['min_dphi4']
    e_pt = arr(arrays, 'Electron_pt', ak.Array([[]] * n))
    e_eta = arr(arrays, 'Electron_eta', ak.Array([[]] * n))
    e_phi = arr(arrays, 'Electron_phi', ak.Array([[]] * n))
    e_mass = arr(arrays, 'Electron_mass', ak.zeros_like(e_pt))
    e_delta_eta_sc = arr(arrays, 'Electron_deltaEtaSC', ak.zeros_like(e_pt))
    e_charge = arr(arrays, 'Electron_charge', ak.zeros_like(e_pt))
    e_cb = arr(arrays, 'Electron_cutBased', ak.zeros_like(e_pt))
    e_iso = arr(arrays, 'Electron_miniPFRelIso_all', ak.ones_like(e_pt) * 99)
    e_fid = (abs(e_eta) < 1.4442) | (abs(e_eta) > 1.566) & (abs(e_eta) < 2.5)
    e_veto = (e_pt > 10) & e_fid & (e_cb >= 1) & (e_iso < 0.1)
    e_med = (e_pt > 10) & e_fid & (e_cb >= 3) & (e_iso < 0.1)
    n_e_veto = count(e_veto)
    n_e_med = count(e_med)
    m_pt = arr(arrays, 'Muon_pt', ak.Array([[]] * n))
    m_eta = arr(arrays, 'Muon_eta', ak.Array([[]] * n))
    m_phi = arr(arrays, 'Muon_phi', ak.Array([[]] * n))
    m_mass = arr(arrays, 'Muon_mass', ak.zeros_like(m_pt))
    m_charge = arr(arrays, 'Muon_charge', ak.zeros_like(m_pt))
    m_looseid = arr(arrays, 'Muon_looseId', ak.zeros_like(m_pt))
    m_medid = arr(arrays, 'Muon_mediumId', ak.zeros_like(m_pt))
    m_iso = arr(arrays, 'Muon_miniPFRelIso_all', ak.ones_like(m_pt) * 99)
    m_loose = (m_pt > muon_pt_min(year)) & (abs(m_eta) < 2.4) & m_looseid & (m_iso < 0.2)
    m_med = (m_pt > muon_pt_min(year)) & (abs(m_eta) < 2.4) & m_medid & (m_iso < 0.2)
    n_m_loose = count(m_loose)
    n_m_med = count(m_med)
    p_pt = arr(arrays, 'Photon_pt', ak.Array([[]] * n))
    p_eta = arr(arrays, 'Photon_eta', ak.Array([[]] * n))
    p_phi = arr(arrays, 'Photon_phi', ak.Array([[]] * n))
    p_cb = arr(arrays, 'Photon_cutBased', ak.zeros_like(p_pt))
    p_r9 = arr(arrays, 'Photon_r9', ak.zeros_like(p_pt))
    p_electron_veto = arr(arrays, 'Photon_electronVeto', ak.zeros_like(p_pt))
    p_med = medium_photon_mask(p_pt, p_eta, p_cb, p_electron_veto)
    n_p_med = count(p_med)
    tau_pt = arr(arrays, 'Tau_pt', ak.Array([[]] * n))
    tau_eta = arr(arrays, 'Tau_eta', ak.Array([[]] * n))
    tau_phi = arr(arrays, 'Tau_phi', ak.Array([[]] * n))
    tau_dz = arr(arrays, 'Tau_dz', ak.zeros_like(tau_pt))
    tau_dm = arr(arrays, 'Tau_decayMode', ak.zeros_like(tau_pt))
    tau_id = arr(arrays, 'Tau_idDeepTau2018v2p5VSjet', ak.zeros_like(tau_pt))
    tau_mt = transverse_mass(tau_pt, tau_phi, met_pt, met_phi)
    tau_med = (tau_pt > 20) & (abs(tau_eta) < 2.5) & (abs(tau_dz) < 0.2) & (tau_dm != 5) & (tau_dm != 6) & (tau_id >= 5) & (tau_mt < 100)
    n_tau = count(tau_med)
    tr_pt = arr(arrays, 'IsoTrack_pt', ak.Array([[]] * n))
    tr_eta = arr(arrays, 'IsoTrack_eta', ak.Array([[]] * n))
    tr_phi = arr(arrays, 'IsoTrack_phi', ak.Array([[]] * n))
    tr_pdg = abs(arr(arrays, 'IsoTrack_pdgId', ak.zeros_like(tr_pt)))
    tr_iso = arr(arrays, 'IsoTrack_pfRelIso03_all', ak.ones_like(tr_pt) * 99)
    tr_mt = transverse_mass(tr_pt, tr_phi, met_pt, met_phi)
    tr_e = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 11) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_m = (tr_pt > 5) & (abs(tr_eta) < 2.5) & (tr_pdg == 13) & (tr_iso < 0.2) & (tr_mt < 100)
    tr_pi = (tr_pt > 10) & (abs(tr_eta) < 2.5) & (tr_pdg == 211) & (tr_iso < 0.1) & (tr_mt < 100)
    fj_pt_raw = arr(arrays, 'FatJet_pt', ak.Array([[]] * n))
    fj_eta = arr(arrays, 'FatJet_eta', ak.Array([[]] * n))
    fj_phi = arr(arrays, 'FatJet_phi', ak.Array([[]] * n))
    fj_msd = arr(arrays, 'FatJet_msoftdrop', ak.zeros_like(fj_pt_raw))
    fj_mass_raw = arr(arrays, 'FatJet_mass', ak.zeros_like(fj_pt_raw))
    fj_top_score = arr(arrays, BOOSTED_TOP_SCORE_BRANCH, ak.ones_like(fj_pt_raw) * -99.0)
    fj_w_score = arr(arrays, BOOSTED_W_SCORE_BRANCH, ak.ones_like(fj_pt_raw) * -99.0)
    fj_pt, fj_mass, fjec_status = apply_jec(arrays, repo, year, process, 'FatJet', fj_pt_raw, fj_eta, fj_phi, fj_mass_raw, shift)
    fj_id_mask, fatjet_id_source = ak8_tight_lepton_veto_mask(arrays, fj_pt, fj_eta, repo)
    good_fj = (fj_pt > 200) & (abs(fj_eta) < 2.0) & (fj_msd > 60) & fj_id_mask
    n_fj = count(good_fj)
    fj1pt = first_or(-99, fj_pt[good_fj])
    fj1eta = first_or(-99, fj_eta[good_fj])
    fj1phi = first_or(-99, fj_phi[good_fj])
    fj1mass = first_or(-99, fj_mass[good_fj])
    fj1msd = first_or(-99, fj_msd[good_fj])
    boosted_top = fj_id_mask & (fj_pt > BOOSTED_TOP_PT_MIN) & (abs(fj_eta) < BOOSTED_ETA_MAX) & (fj_msd > BOOSTED_TOP_MSD_MIN) & (fj_top_score > BOOSTED_TOP_SCORE_WP) & (fj_top_score <= BOOSTED_SCORE_MAX)
    boosted_w = fj_id_mask & (fj_pt > BOOSTED_W_PT_MIN) & (abs(fj_eta) < BOOSTED_ETA_MAX) & (fj_msd > BOOSTED_W_MSD_MIN) & (fj_msd < BOOSTED_W_MSD_MAX) & (fj_w_score > BOOSTED_W_SCORE_WP) & (fj_w_score <= BOOSTED_SCORE_MAX)
    nboosted_top = count(boosted_top)
    nboosted_w = count(boosted_w)
    nboosted_total = nboosted_top + nboosted_w
    fj1topscore = first_or(-99, fj_top_score[good_fj])
    fj1wscore = first_or(-99, fj_w_score[good_fj])
    fj_subidx1 = arr(arrays, 'FatJet_subJetIdx1', ak.ones_like(fj_pt_raw) * -1)
    fj_subidx2 = arr(arrays, 'FatJet_subJetIdx2', ak.ones_like(fj_pt_raw) * -1)
    subjet_btag_upart = arr(arrays, 'SubJet_btagUParTAK4B', ak.Array([[]] * n))
    subjet_btag_deepb = arr(arrays, 'SubJet_btagDeepB', ak.Array([[]] * n))
    met_filters, missing_filters = all_filters(arrays, n)
    sig_hlt = bool_branch(arrays, SIGNAL_HLT, n)
    pho_hlt = bool_branch(arrays, PHOTON_HLT, n)
    ht_reference_hlt = bool_branch(arrays, HT_REFERENCE_HLT, n)
    ele_hlt = bool_branch(arrays, ELECTRON_HLT, n)
    electron_reference_hlt = bool_branch(arrays, ELECTRON_REFERENCE_HLT, n)
    mu_hlt = bool_branch(arrays, MUON_HLT, n)
    signal_trigger_policy = 'standard trigger requirement'
    if fastsim_trigger_bypass and process == 'SMS':
        sig_hlt = np.ones(n, dtype=bool)
        signal_trigger_policy = 'FastSim trigger bypass: HLT branches absent; trigger not applied at event-selection level'
    e1pt = first_or(-99, e_pt[e_med])
    e2pt = nth_or(-99, e_pt[e_med], 1)
    e1eta = first_or(0, e_eta[e_med])
    e2eta = nth_or(0, e_eta[e_med], 1)
    e1phi = first_or(0, e_phi[e_med])
    e2phi = nth_or(0, e_phi[e_med], 1)
    e1m = first_or(0, e_mass[e_med])
    e2m = nth_or(0, e_mass[e_med], 1)
    e1q = first_or(0, e_charge[e_med])
    e2q = nth_or(0, e_charge[e_med], 1)
    mee = invariant_mass(e1pt, e1eta, e1phi, e1m, e2pt, e2eta, e2phi, e2m)
    pee = np.sqrt(np.maximum(0, e1pt ** 2 + e2pt ** 2 + 2 * e1pt * e2pt * np.cos(e1phi - e2phi)))
    m1pt = first_or(-99, m_pt[m_med])
    m2pt = nth_or(-99, m_pt[m_med], 1)
    m1eta = first_or(0, m_eta[m_med])
    m2eta = nth_or(0, m_eta[m_med], 1)
    m1phi = first_or(0, m_phi[m_med])
    m2phi = nth_or(0, m_phi[m_med], 1)
    m1m = first_or(0, m_mass[m_med])
    m2m = nth_or(0, m_mass[m_med], 1)
    m1q = first_or(0, m_charge[m_med])
    m2q = nth_or(0, m_charge[m_med], 1)
    mmm = invariant_mass(m1pt, m1eta, m1phi, m1m, m2pt, m2eta, m2phi, m2m)
    pmm = np.sqrt(np.maximum(0, m1pt ** 2 + m2pt ** 2 + 2 * m1pt * m2pt * np.cos(m1phi - m2phi)))
    if dy_mass_window == 'on_z':
        dy2e_mass_mask = (mee > 71) & (mee < 111)
        dy2m_mass_mask = (mmm > 71) & (mmm < 111)
    elif dy_mass_window == 'measurement':
        dy2e_mass_mask = mee > 50
        dy2m_mass_mask = mmm > 50
    else:
        raise ValueError("dy_mass_window must be 'on_z' or 'measurement'")
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
    one_veto_lepton = (n_e_veto == 1) & (n_m_loose == 0) | (n_e_veto == 0) & (n_m_loose == 1)
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
    recoil_g, recoil_g_phi = transverse_vector_sum((met_pt, met_phi), (first_or(0, p_pt[p_med]), first_or(0, p_phi[p_med])))
    recoil_dy2e, recoil_dy2e_phi = transverse_vector_sum((met_pt, met_phi), (e1pt, e1phi), (e2pt, e2phi))
    recoil_dy2m, recoil_dy2m_phi = transverse_vector_sum((met_pt, met_phi), (m1pt, m1phi), (m2pt, m2phi))
    jet_photon_recoil = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j & photon_clean_j, b_med & photon_clean_j, recoil_g_phi)
    jet_electron_recoil = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j & electron_clean_j, b_med & electron_clean_j, recoil_dy2e_phi)
    jet_muon_recoil = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j & muon_clean_j, b_med & muon_clean_j, recoil_dy2m_phi)
    jet_photon_clean = jet_photon_recoil
    jet_lepton_clean = jet_feature_block(jet_pt, jet_eta, jet_phi, good_j & lepton_clean_j, b_med & lepton_clean_j, met_phi)
    ht_photon_300 = jet_photon_recoil['ht'] > 300
    ht_lepton_300 = jet_lepton_clean['ht'] > 300
    any_analysis_trigger = sig_hlt | pho_hlt | ele_hlt | mu_hlt
    common_recoil_or_met = met_250 | (recoil_g > 250) | (recoil_dy2e > 250) | (recoil_dy2m > 250)
    common_ht = ht_300 | ht_photon_300 | (jet_electron_recoil['ht'] > 300) | (jet_muon_recoil['ht'] > 300)
    common_njet2 = (njet >= 2) | (jet_photon_recoil['njet'] >= 2) | (jet_electron_recoil['njet'] >= 2) | (jet_muon_recoil['njet'] >= 2)
    flat_preselection = base_common & zero_tau & any_analysis_trigger & common_recoil_or_met & common_ht & common_njet2
    met_trigger_genuine_measurement = base_common & electron_reference_hlt & zero_tau & (n_e_med >= 1) & zero_m & (jet_lepton_clean['njet'] >= 2) & ht_lepton_300 & jet_nominal['open_pre']
    met_trigger_qcd_measurement = base_common & ht_reference_hlt & no_veto_leptons & zero_tau & (njet >= 2) & ht_300 & jet_nominal['qcd_open'] & jet_nominal['dphi123_0p1']
    photon_trigger_measurement = base_common & ht_reference_hlt & (n_p_med >= 1) & no_veto_leptons & zero_tau & (jet_photon_recoil['njet'] >= 2) & ht_photon_300 & jet_photon_recoil['open_pre']
    masks = {'preselection': base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & jet_nominal['open_pre'] & ht_300, 'LLCR': base_common & sig_hlt & zero_tau & (njet >= 5) & (nb >= 1) & one_veto_lepton & mt_100 & met_250 & jet_nominal['open_high'] & ht_300, 'QCDCR': base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & jet_nominal['qcd_open'] & jet_nominal['dphi123_0p1'] & ht_300, 'GCR': base_common & pho_hlt & (n_p_med == 1) & no_veto_leptons & zero_tau & (jet_photon_recoil['njet'] >= 5) & (jet_photon_recoil['nb'] >= 1) & (met_pt < 250) & (recoil_g > 250) & jet_photon_recoil['open_high'] & ht_photon_300, 'DY2E': base_common & ele_hlt & zero_tau & (jet_electron_recoil['njet'] >= 5) & (jet_electron_recoil['nb'] >= 1) & zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & (recoil_dy2e > 250) & dy2e_mass_mask & jet_electron_recoil['open_high'] & (jet_electron_recoil['ht'] > 300), 'DY2M': base_common & mu_hlt & zero_tau & (jet_muon_recoil['njet'] >= 5) & (jet_muon_recoil['nb'] >= 1) & zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & (recoil_dy2m > 250) & dy2m_mass_mask & jet_muon_recoil['open_high'] & (jet_muon_recoil['ht'] > 300), 'SR': base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 5) & (nb >= 1) & met_250 & jet_nominal['open_high'] & ht_300}
    sv_required = ['SV_pt', 'SV_eta', 'SV_phi', 'SV_dxy', 'SV_dlenSig', 'SV_pAngle', 'SV_ntracks']
    sv_available = all((has_field(arrays, name) for name in sv_required))
    if sv_available:
        sv_pt = arr(arrays, 'SV_pt', ak.Array([[]] * n))
        sv_eta = arr(arrays, 'SV_eta', ak.Array([[]] * n))
        sv_phi = arr(arrays, 'SV_phi', ak.Array([[]] * n))
        sv_dxy = arr(arrays, 'SV_dxy', ak.zeros_like(sv_pt))
        sv_dlen_sig = arr(arrays, 'SV_dlenSig', ak.zeros_like(sv_pt))
        sv_pangle = arr(arrays, 'SV_pAngle', ak.ones_like(sv_pt) * 99.0)
        sv_ntracks = arr(arrays, 'SV_ntracks', ak.zeros_like(sv_pt))
        sv_base = (sv_pt < 20) & (abs(sv_dxy) < 3) & (sv_dlen_sig > 4) & (sv_pangle < LOWDM_SV_POINTING_ANGLE_MAX) & (sv_ntracks >= 3)

        def softb_count(jet_mask: Any) -> np.ndarray:
            return count(sv_base & clean_by_delta_r(sv_eta, sv_phi, jet_eta[jet_mask], jet_phi[jet_mask], 0.4))
    else:

        def softb_count(jet_mask: Any) -> np.ndarray:
            return np.full(n, -1, dtype=int)
    lowdm_blocks = {'SR': lowdm_kinematic_block(jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag, good_jet_mask=good_j, loose_b_mask=b_loose_lowdm, medium_b_mask=b_med_lowdm, fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask, fatjet_clean_mask=all_fj_clean, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w, fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2, subjet_btag=subjet_btag_upart, recoil_pt=met_pt, recoil_phi=met_phi, n=n), 'GCR': lowdm_kinematic_block(jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag, good_jet_mask=good_j & photon_clean_j, loose_b_mask=b_loose_lowdm & photon_clean_j, medium_b_mask=b_med_lowdm & photon_clean_j, fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask, fatjet_clean_mask=photon_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w, fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2, subjet_btag=subjet_btag_upart, recoil_pt=recoil_g, recoil_phi=recoil_g_phi, n=n), 'DY2E': lowdm_kinematic_block(jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag, good_jet_mask=good_j & electron_clean_j, loose_b_mask=b_loose_lowdm & electron_clean_j, medium_b_mask=b_med_lowdm & electron_clean_j, fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask, fatjet_clean_mask=electron_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w, fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2, subjet_btag=subjet_btag_upart, recoil_pt=recoil_dy2e, recoil_phi=recoil_dy2e_phi, n=n), 'DY2M': lowdm_kinematic_block(jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_btag=btag, good_jet_mask=good_j & muon_clean_j, loose_b_mask=b_loose_lowdm & muon_clean_j, medium_b_mask=b_med_lowdm & muon_clean_j, fatjet_pt=fj_pt, fatjet_eta=fj_eta, fatjet_phi=fj_phi, fatjet_id_mask=fj_id_mask, fatjet_clean_mask=muon_clean_fj, boosted_top_mask=boosted_top, boosted_w_mask=boosted_w, fatjet_subjet_idx1=fj_subidx1, fatjet_subjet_idx2=fj_subidx2, subjet_btag=subjet_btag_upart, recoil_pt=recoil_dy2m, recoil_phi=recoil_dy2m_phi, n=n)}
    lowdm_blocks['LLCR'] = lowdm_blocks['SR']
    lowdm_blocks['QCDCR'] = lowdm_blocks['SR']
    lowdm_nsv = {'SR': softb_count(good_j), 'LLCR': softb_count(good_j), 'QCDCR': softb_count(good_j), 'GCR': softb_count(good_j & photon_clean_j), 'DY2E': softb_count(good_j & electron_clean_j), 'DY2M': softb_count(good_j & muon_clean_j)}

    def lowdm_quality(block: dict[str, Any]) -> np.ndarray:
        return block['pass_topology'] & block['pass_isr'] & block['pass_met_sqrt_ht']
    lowdm_masks = {'SR': base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal['open_pre'] & lowdm_quality(lowdm_blocks['SR']), 'LLCR': base_common & sig_hlt & one_veto_lepton & mt_100 & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal['open_pre'] & lowdm_quality(lowdm_blocks['LLCR']), 'QCDCR': base_common & sig_hlt & no_veto_leptons & zero_tau & (njet >= 2) & met_250 & ht_300 & jet_nominal['qcd_open'] & jet_nominal['dphi123_0p1'] & lowdm_quality(lowdm_blocks['QCDCR']), 'GCR': base_common & pho_hlt & (n_p_med == 1) & no_veto_leptons & zero_tau & (met_pt < 250) & (recoil_g > 250) & (jet_photon_recoil['njet'] >= 2) & (jet_photon_recoil['ht'] > 300) & jet_photon_recoil['open_pre'] & lowdm_quality(lowdm_blocks['GCR']), 'DY2E': base_common & ele_hlt & zero_tau & zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & dy2e_mass_mask & (recoil_dy2e > 250) & (jet_electron_recoil['njet'] >= 2) & (jet_electron_recoil['ht'] > 300) & jet_electron_recoil['open_pre'] & lowdm_quality(lowdm_blocks['DY2E']), 'DY2M': base_common & mu_hlt & zero_tau & zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & dy2m_mass_mask & (recoil_dy2m > 250) & (jet_muon_recoil['njet'] >= 2) & (jet_muon_recoil['ht'] > 300) & jet_muon_recoil['open_pre'] & lowdm_quality(lowdm_blocks['DY2M'])}
    lowdm_recoil = {'SR': met_pt, 'LLCR': met_pt, 'QCDCR': met_pt, 'GCR': recoil_g, 'DY2E': recoil_dy2e, 'DY2M': recoil_dy2m}
    lowdm_search_bins = {}
    for region, block in lowdm_blocks.items():
        lowdm_search_bins[region] = np.asarray([assign_lowdm_search_bin(int(block['jets']['njet'][i]), int(block['nb_medium'][i]), int(lowdm_nsv[region][i]), float(block['isr_pt'][i]), float(block['ptb'][i]), float(lowdm_recoil[region][i]), float(block['mtb'][i])) if bool(lowdm_masks[region][i]) else -1 for i in range(n)], dtype=int)
        lowdm_masks[region] = lowdm_masks[region] & (lowdm_search_bins[region] >= 0)
    nominal_lowdm = lowdm_blocks['SR']
    lowdm_isr_fj = nominal_lowdm['isr_mask']
    n_lowdm_isr = nominal_lowdm['n_isr']
    lowdm_isr_pt = nominal_lowdm['isr_pt']
    lowdm_isr_eta = nominal_lowdm['isr_eta']
    lowdm_isr_phi = nominal_lowdm['isr_phi']
    lowdm_isr_dphi = nominal_lowdm['isr_dphi']
    lowdm_isr_subjet_btag_max = nominal_lowdm['isr_subjet_btag_max']
    lowdm_isr_subjet_bveto_available = nominal_lowdm['isr_subjet_bveto_available']
    pass_lowdm_isr_bveto = nominal_lowdm['pass_isr_bveto']
    nb_medium_lowdm = nominal_lowdm['nb_medium']
    nb_loose_lowdm = nominal_lowdm['nb_loose']
    lowdm_ptb = nominal_lowdm['ptb']
    lowdm_mtb = nominal_lowdm['mtb']
    lowdm_met_sqrt_ht = nominal_lowdm['met_sqrt_ht']
    n_sv_softb = lowdm_nsv['SR']
    feature_lowdm_preselection = masks['preselection']
    pass_lowdm_topology_veto = nominal_lowdm['pass_topology']
    pass_lowdm_isr = nominal_lowdm['pass_isr']
    pass_lowdm_met_sqrt_ht = nominal_lowdm['pass_met_sqrt_ht']
    pass_lowdm_mtb = nominal_lowdm['pass_mtb']
    feature_lowdm_sr_base = lowdm_masks['SR']
    lowdm_search_bin = lowdm_search_bins['SR']
    jet_hadflav = arr(arrays, 'Jet_hadronFlavour', ak.zeros_like(jet_pt))
    if is_data_process(process):
        gen_weight = np.ones(n, dtype=float)
    else:
        gen_weight = np_filled(arr(arrays, 'genWeight', np.ones(n)), n, 1.0)
    weight_variations = {'nominal': gen_weight}
    scale_factor_status = {'applied': False, 'reason': 'deferred_to_post_skim', 'available_variations': ['nominal'], 'components': {}, 'note': 'JEC/FJEC and MET/JES shape shifts are applied before skim; multiplicative scale factors are intentionally not applied in flat ntuple production.'}
    weight = weight_variations['nominal']
    unavailable_features = list(missing_filters)
    if jet_id_source.startswith('raw_kinematic'):
        unavailable_features.append('baseline AK4 correctionlib jet ID inputs missing; raw kinematic fallback used')
    if fatjet_id_source.startswith('raw_kinematic'):
        unavailable_features.append('baseline AK8 correctionlib jet ID inputs missing; raw kinematic fallback used')
    if not jec_status.get('applied'):
        unavailable_features.append('AK4 JEC not applied: ' + str(jec_status.get('reason', jec_status.get('source', 'unknown'))))
    if not fjec_status.get('applied'):
        unavailable_features.append('AK8 JEC not applied: ' + str(fjec_status.get('reason', fjec_status.get('source', 'unknown'))))
    missing_boosted_score_branches = [name for name in [BOOSTED_TOP_SCORE_BRANCH, BOOSTED_W_SCORE_BRANCH] if not has_field(arrays, name)]
    if missing_boosted_score_branches:
        unavailable_features.append('boosted top/W tag score branches missing: ' + ','.join(missing_boosted_score_branches))
    for comp_name, comp_status in scale_factor_status.get('components', {}).items():
        if not comp_status.get('applied') and (not str(comp_status.get('source', '')).startswith('unity_fallback_not_TTto')):
            unavailable_features.append(f"{comp_name} not applied: {comp_status.get('source', 'unknown')}")
    cut_sequences = {'preselection': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', njet >= 2), ('bjet_multiplicity', np.ones(n, dtype=bool)), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_pre']), ('final_region_selection', masks['preselection'])], 'LLCR': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', one_veto_lepton & mt_100), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', njet >= 5), ('bjet_multiplicity', nb >= 1), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_high']), ('final_region_selection', masks['LLCR'])], 'QCDCR': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', njet >= 5), ('bjet_multiplicity', nb >= 1), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['qcd_open'] & jet_nominal['dphi123_0p1']), ('final_region_selection', masks['QCDCR'])], 'GCR': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', pho_hlt), ('lepton_veto_or_selection', (n_p_med == 1) & no_veto_leptons), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', jet_photon_clean['njet'] >= 5), ('bjet_multiplicity', jet_photon_clean['nb'] >= 1), ('MET_or_recoil_threshold', (met_pt < 250) & (recoil_g > 250)), ('HT_threshold', ht_photon_300), ('delta_phi_requirements', jet_photon_clean['open_high']), ('final_region_selection', masks['GCR'])], 'DY2E': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', ele_hlt), ('lepton_veto_or_selection', zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & dy2e_mass_mask), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', jet_electron_recoil['njet'] >= 5), ('bjet_multiplicity', jet_electron_recoil['nb'] >= 1), ('MET_or_recoil_threshold', recoil_dy2e > 250), ('HT_threshold', jet_electron_recoil['ht'] > 300), ('delta_phi_requirements', jet_electron_recoil['open_high']), ('final_region_selection', masks['DY2E'])], 'DY2M': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', mu_hlt), ('lepton_veto_or_selection', zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & dy2m_mass_mask), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', jet_muon_recoil['njet'] >= 5), ('bjet_multiplicity', jet_muon_recoil['nb'] >= 1), ('MET_or_recoil_threshold', recoil_dy2m > 250), ('HT_threshold', jet_muon_recoil['ht'] > 300), ('delta_phi_requirements', jet_muon_recoil['open_high']), ('final_region_selection', masks['DY2M'])], 'SR': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', njet >= 5), ('bjet_multiplicity', nb >= 1), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_high']), ('final_region_selection', masks['SR'])]}
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
            cutflows[region].append({'cut': cut_name, 'unweighted': uw, 'weighted': ww})
        for item in cutflows[region]:
            item['first_zero_cut'] = first_zero
    signal_cutflow_histograms: dict[str, Any] = {}
    if process == 'SMS':
        highdm_cut_sequences = {region: seq[:4] + [('PuppiMET_over_CaloMET', puppi_calo < 5)] + seq[4:] for region, seq in cut_sequences.items()}
        lowdm_cut_sequences = {'preselection': [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('PuppiMET_over_CaloMET', puppi_calo < 5), ('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j), ('jet_multiplicity', njet >= 2), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_pre']), ('final_region_selection', feature_lowdm_preselection)], 'LLCR': [('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', one_veto_lepton & mt_100), ('jet_multiplicity', njet >= 2), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_pre'])], 'QCDCR': [('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('jet_multiplicity', njet >= 2), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['qcd_open'] & jet_nominal['dphi123_0p1'])], 'GCR': [('trigger_requirement', pho_hlt), ('lepton_veto_or_selection', (n_p_med == 1) & no_veto_leptons), ('jet_multiplicity', jet_photon_recoil['njet'] >= 2), ('MET_or_recoil_threshold', (met_pt < 250) & (recoil_g > 250)), ('HT_threshold', jet_photon_recoil['ht'] > 300), ('delta_phi_requirements', jet_photon_recoil['open_pre'])], 'DY2E': [('trigger_requirement', ele_hlt), ('lepton_veto_or_selection', zero_m & (n_e_med == 2) & (e1pt > 40) & (e2pt > 20) & (pee > 200) & (e1q != e2q) & dy2e_mass_mask), ('jet_multiplicity', jet_electron_recoil['njet'] >= 2), ('MET_or_recoil_threshold', recoil_dy2e > 250), ('HT_threshold', jet_electron_recoil['ht'] > 300), ('delta_phi_requirements', jet_electron_recoil['open_pre'])], 'DY2M': [('trigger_requirement', mu_hlt), ('lepton_veto_or_selection', zero_e & (n_m_med == 2) & (m1pt > 50) & (m2pt > 20) & (pmm > 200) & (m1q != m2q) & dy2m_mass_mask), ('jet_multiplicity', jet_muon_recoil['njet'] >= 2), ('MET_or_recoil_threshold', recoil_dy2m > 250), ('HT_threshold', jet_muon_recoil['ht'] > 300), ('delta_phi_requirements', jet_muon_recoil['open_pre'])], 'SR': [('trigger_requirement', sig_hlt), ('lepton_veto_or_selection', no_veto_leptons), ('jet_multiplicity', njet >= 2), ('MET_or_recoil_threshold', met_250), ('HT_threshold', ht_300), ('delta_phi_requirements', jet_nominal['open_pre'])]}
        lowdm_common_prefix = [('total_read_events', np.ones(n, dtype=bool)), ('valid_MET', valid_met), ('lumimask', lumi_mask), ('MET_filters', met_filters), ('PuppiMET_over_CaloMET', puppi_calo < 5)]
        lowdm_common_vetoes = [('tau_veto', zero_tau), ('isolated_track_veto', no_tracks), ('jet_veto_map', zero_veto_j)]
        for region in ('LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M', 'SR'):
            block = lowdm_blocks[region]
            region_steps = lowdm_cut_sequences[region]
            lowdm_cut_sequences[region] = lowdm_common_prefix + region_steps[:2] + lowdm_common_vetoes + region_steps[2:] + [('boosted_topology_veto', block['pass_topology']), ('exactly_one_ISR', block['pass_isr']), ('ISR_subjet_b_veto', block['pass_isr_bveto']), ('MET_over_sqrt_HT', block['pass_met_sqrt_ht']), ('mTb_requirement', block['pass_mtb']), ('valid_search_bin', lowdm_search_bins[region] >= 0), ('final_region_selection', lowdm_masks[region])]
        signal_cutflow_histograms = {'schema_version': 'signal_cutflow_histograms_v1', 'sample_scope': 'signal_only', 'weight_scope': 'raw_nominal_event_weight_before_lumi_xsec_normalization', 'topologies': {'highdm': {region: cumulative_cutflow_histogram(seq, weight) for region, seq in highdm_cut_sequences.items()}, 'lowdm': {region: cumulative_cutflow_histogram(seq, weight) for region, seq in lowdm_cut_sequences.items()}}}
    materialization_masks = {'feature_flat_preselection': flat_preselection, 'feature_lowdm_preselection': feature_lowdm_preselection, 'feature_lowdm_sr_base': feature_lowdm_sr_base, 'feature_met_trigger_genuine_measurement': met_trigger_genuine_measurement, 'feature_met_trigger_qcd_measurement': met_trigger_qcd_measurement, 'feature_photon_trigger_measurement': photon_trigger_measurement, **{f'feature_lowdm_{region}': lowdm_masks[region] for region in ('LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M', 'SR')}, **{f'feature_{region}': masks[region] for region in REGION_NAMES}}
    selected_materialization_mask = materialization_masks.get(materialize_skim_flag)
    if selected_materialization_mask is None:
        materialize_event_mask = np.ones(n, dtype=bool)
        materialization_applied = False
    else:
        materialize_event_mask = np.asarray(selected_materialization_mask, dtype=bool)
        materialization_applied = True
    materialize_indices = np.flatnonzero(materialize_event_mask)
    gen_top_pts = ak.Array([[]] * n)
    if has_field(arrays, 'GenPart_pt') and has_field(arrays, 'GenPart_pdgId') and has_field(arrays, 'GenPart_statusFlags'):
        gen_flags_for_flat = ak.values_astype(arrays['GenPart_statusFlags'], np.int64)
        is_final_top_for_flat = (abs(arrays['GenPart_pdgId']) == 6) & (gen_flags_for_flat & 1 << 8 != 0) & (gen_flags_for_flat & 1 << 13 != 0)
        gen_top_pts = arrays['GenPart_pt'][is_final_top_for_flat]
    flat_vectors = {'good_jet_pt': ak.to_list(jet_pt[good_j][materialize_event_mask]), 'good_jet_eta': ak.to_list(jet_eta[good_j][materialize_event_mask]), 'good_jet_phi': ak.to_list(jet_phi[good_j][materialize_event_mask]), AK4_BTAG_OUTPUT_FIELD: ak.to_list(btag[good_j][materialize_event_mask]), 'good_jet_hadron_flavour': ak.to_list(jet_hadflav[good_j][materialize_event_mask]), 'good_jet_b_loose': ak.to_list(ak.values_astype(b_loose[good_j], np.int32)[materialize_event_mask]), 'good_jet_b_medium': ak.to_list(ak.values_astype(b_med[good_j], np.int32)[materialize_event_mask]), 'lowdm_fatjet_pt': ak.to_list(fj_pt[lowdm_isr_fj][materialize_event_mask]), 'lowdm_fatjet_eta': ak.to_list(fj_eta[lowdm_isr_fj][materialize_event_mask]), 'lowdm_fatjet_phi': ak.to_list(fj_phi[lowdm_isr_fj][materialize_event_mask]), 'lowdm_fatjet_msd': ak.to_list(fj_msd[lowdm_isr_fj][materialize_event_mask]), 'lowdm_fatjet_subjet_idx1': ak.to_list(ak.values_astype(fj_subidx1[lowdm_isr_fj], np.int32)[materialize_event_mask]), 'lowdm_fatjet_subjet_idx2': ak.to_list(ak.values_astype(fj_subidx2[lowdm_isr_fj], np.int32)[materialize_event_mask]), 'electron_veto_pt': ak.to_list(e_pt[e_veto][materialize_event_mask]), 'electron_veto_eta': ak.to_list(e_eta[e_veto][materialize_event_mask]), 'electron_veto_eta_sc': ak.to_list((e_eta + e_delta_eta_sc)[e_veto][materialize_event_mask]), 'electron_veto_phi': ak.to_list(e_phi[e_veto][materialize_event_mask]), 'electron_medium_pt': ak.to_list(e_pt[e_med][materialize_event_mask]), 'electron_medium_eta': ak.to_list(e_eta[e_med][materialize_event_mask]), 'electron_medium_eta_sc': ak.to_list((e_eta + e_delta_eta_sc)[e_med][materialize_event_mask]), 'electron_medium_phi': ak.to_list(e_phi[e_med][materialize_event_mask]), 'muon_loose_pt': ak.to_list(m_pt[m_loose][materialize_event_mask]), 'muon_loose_eta': ak.to_list(m_eta[m_loose][materialize_event_mask]), 'muon_loose_phi': ak.to_list(m_phi[m_loose][materialize_event_mask]), 'muon_medium_pt': ak.to_list(m_pt[m_med][materialize_event_mask]), 'muon_medium_eta': ak.to_list(m_eta[m_med][materialize_event_mask]), 'muon_medium_phi': ak.to_list(m_phi[m_med][materialize_event_mask]), 'photon_medium_pt': ak.to_list(p_pt[p_med][materialize_event_mask]), 'photon_medium_eta': ak.to_list(p_eta[p_med][materialize_event_mask]), 'photon_medium_phi': ak.to_list(p_phi[p_med][materialize_event_mask]), 'gen_top_pt': ak.to_list(gen_top_pts[materialize_event_mask])}
    pu_ntrueint = np_filled(arr(arrays, 'Pileup_nTrueInt', np.ones(n) * -1.0), n, -1.0)
    genmodel_branches = sorted((name for name in getattr(arrays, 'fields', []) if signal_genmodel_branch(str(name))))
    genmodel_masks = {name: np.asarray(arrays[name], dtype=bool) for name in genmodel_branches}

    def active_genmodel(i: int) -> tuple[str, int | None, int | None]:
        for name, mask in genmodel_masks.items():
            if i < len(mask) and bool(mask[i]):
                nums = re.findall('(\\d+)', name)
                if len(nums) >= 2:
                    return (name, int(nums[-2]), int(nums[-1]))
                return (name, None, None)
        return ('', None, None)
    rows = []
    for output_index, i in enumerate(materialize_indices):
        gen_branch, gen_mstop, gen_mlsp = active_genmodel(i)
        row = {'dataset': dataset, 'process': process, 'year': year, 'signal_point': sp or '', 'file': file_path, 'shape_shift': shift, 'genmodel_branch': gen_branch, 'signal_topology_id': signal_topology_id(gen_branch or dataset), 'mStop': gen_mstop if gen_mstop is not None else '', 'mLSP': gen_mlsp if gen_mlsp is not None else '', 'trigger_policy': signal_trigger_policy, 'entry': entry_start + i, 'run': int(arrays['run'][i]), 'luminosityBlock': int(arrays['luminosityBlock'][i]), 'event': int(arrays['event'][i]), 'met': float(met_pt[i]), 'met_phi': float(met_phi[i]), 'ht': float(ht[i]), 'njet': int(njet[i]), 'nb_medium': int(nb[i]), 'nb_loose': int(nb_loose[i]), 'nb_medium_lowdm': int(nb_medium_lowdm[i]), 'nb_loose_lowdm': int(nb_loose_lowdm[i]), 'pu_ntrueint': float(pu_ntrueint[i]), 'ht_photon_clean': float(jet_photon_clean['ht'][i]), 'njet_photon_clean': int(jet_photon_clean['njet'][i]), 'nb_photon_clean': int(jet_photon_clean['nb'][i]), 'gcr_j1_recoil_dphi': float(jet_photon_recoil['j1dphi'][i]), 'gcr_j2_recoil_dphi': float(jet_photon_recoil['j2dphi'][i]), 'gcr_j3_recoil_dphi': float(jet_photon_recoil['j3dphi'][i]), 'gcr_j4_recoil_dphi': float(jet_photon_recoil['j4dphi'][i]), 'gcr_min_recoil_dphi4': float(jet_photon_recoil['min_dphi4'][i]), 'pass_gcr_open_high': bool(jet_photon_recoil['open_high'][i]), 'ht_lepton_clean': float(jet_lepton_clean['ht'][i]), 'njet_lepton_clean': int(jet_lepton_clean['njet'][i]), 'nb_lepton_clean': int(jet_lepton_clean['nb'][i]), 'j1pt': float(j1pt[i]), 'j1eta': float(j1eta[i]), 'j1phi': float(j1phi[i]), 'j2pt': float(j2pt[i]), 'j1_met_dphi': float(j1dphi[i]), 'j2_met_dphi': float(j2dphi[i]), 'j3_met_dphi': float(j3dphi[i]), 'j4_met_dphi': float(j4dphi[i]), 'min_dphi4': float(min_dphi4[i]), 'nfj': int(n_fj[i]), 'fj1pt': float(fj1pt[i]), 'fj1eta': float(fj1eta[i]), 'fj1phi': float(fj1phi[i]), 'fj1mass': float(fj1mass[i]), 'fj1msd': float(fj1msd[i]), 'fj1_top_score': float(fj1topscore[i]), 'fj1_w_score': float(fj1wscore[i]), 'nboosted_top': int(nboosted_top[i]), 'nboosted_w': int(nboosted_w[i]), 'nboosted_total': int(nboosted_total[i]), 'n_lowdm_isr': int(n_lowdm_isr[i]), 'n_sv_softb': int(n_sv_softb[i]), 'lowdm_search_bin': int(lowdm_search_bin[i]), **{f'lowdm_search_bin_{region}': int(lowdm_search_bins[region][i]) for region in ('LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M', 'SR')}, 'lowdm_isr_pt': float(lowdm_isr_pt[i]), 'lowdm_isr_eta': float(lowdm_isr_eta[i]), 'lowdm_isr_phi': float(lowdm_isr_phi[i]), 'lowdm_isr_dphi': float(lowdm_isr_dphi[i]), 'lowdm_met_sqrt_ht': float(lowdm_met_sqrt_ht[i]), 'lowdm_ptb': float(lowdm_ptb[i]), 'lowdm_mtb': float(lowdm_mtb[i]), 'lowdm_isr_subjet_btag_max': float(lowdm_isr_subjet_btag_max[i]), 'n_e_veto': int(n_e_veto[i]), 'n_e_medium': int(n_e_med[i]), 'n_m_loose': int(n_m_loose[i]), 'n_m_medium': int(n_m_med[i]), 'n_photon_medium': int(n_p_med[i]), 'mee': float(mee[i]), 'pee': float(pee[i]), 'mmm': float(mmm[i]), 'pmm': float(pmm[i]), 'recoil_gcr': float(recoil_g[i]), 'recoil_gcr_phi': float(recoil_g_phi[i]), 'recoil_dy2e': float(recoil_dy2e[i]), 'recoil_dy2m': float(recoil_dy2m[i]), 'recoil_dy2e_phi': float(recoil_dy2e_phi[i]), 'recoil_dy2m_phi': float(recoil_dy2m_phi[i]), 'gen_weight': finite_float(gen_weight[i]), 'nominal_weight': finite_float(weight[i]), 'weight_variations': {name: finite_float(vals[i]) for name, vals in weight_variations.items()}, 'available_systematics': ';'.join(sorted(weight_variations)), 'lumi_mask_source': lumi_mask_source, 'jet_id_source': jet_id_source, 'fatjet_id_source': fatjet_id_source, 'jet_veto_source': jet_veto_source, 'unavailable_features': ';'.join(unavailable_features)}
        for vector_name, vector_values in flat_vectors.items():
            row[vector_name] = vector_values[output_index]
        row.update({'feature_flat_preselection': bool(flat_preselection[i]), 'feature_lowdm_preselection': bool(feature_lowdm_preselection[i]), 'feature_lowdm_sr_base': bool(feature_lowdm_sr_base[i]), 'feature_met_trigger_genuine_measurement': bool(met_trigger_genuine_measurement[i]), 'feature_met_trigger_qcd_measurement': bool(met_trigger_qcd_measurement[i]), 'feature_photon_trigger_measurement': bool(photon_trigger_measurement[i]), **{f'feature_lowdm_{region}': bool(lowdm_masks[region][i]) for region in ('LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M', 'SR')}, 'pass_lowdm_topology_veto': bool(pass_lowdm_topology_veto[i]), 'pass_lowdm_isr': bool(pass_lowdm_isr[i]), 'pass_lowdm_isr_bveto': bool(pass_lowdm_isr_bveto[i]), 'lowdm_isr_subjet_bveto_available': bool(lowdm_isr_subjet_bveto_available[i]), 'pass_lowdm_met_sqrt_ht': bool(pass_lowdm_met_sqrt_ht[i]), 'pass_lowdm_mtb': bool(pass_lowdm_mtb[i]), 'pass_base_common': bool(base_common[i]), 'pass_any_analysis_trigger': bool(any_analysis_trigger[i]), 'pass_signal_trigger': bool(sig_hlt[i]), 'pass_photon_trigger': bool(pho_hlt[i]), 'pass_ht_reference_trigger': bool(ht_reference_hlt[i]), 'pass_electron_trigger': bool(ele_hlt[i]), 'pass_electron_reference_trigger': bool(electron_reference_hlt[i]), 'pass_muon_trigger': bool(mu_hlt[i]), 'pass_zero_tau': bool(zero_tau[i]), 'pass_no_tracks': bool(no_tracks[i]), 'pass_jet_veto_map': bool(zero_veto_j[i]), 'pass_no_veto_leptons': bool(no_veto_leptons[i]), 'pass_one_veto_lepton': bool(one_veto_lepton[i]), 'pass_mt_100': bool(mt_100[i]), 'pass_met_250': bool(met_250[i]), 'pass_common_recoil_or_met': bool(common_recoil_or_met[i]), 'pass_ht_300': bool(ht_300[i]), 'pass_ht_photon_300': bool(ht_photon_300[i]), 'pass_ht_lepton_300': bool(ht_lepton_300[i]), 'pass_common_ht': bool(common_ht[i]), 'pass_common_njet2': bool(common_njet2[i]), 'pass_open_pre': bool(jet_nominal['open_pre'][i]), 'pass_open_high': bool(jet_nominal['open_high'][i]), 'pass_qcd_open': bool(jet_nominal['qcd_open'][i]), 'pass_dphi123_0p1': bool(jet_nominal['dphi123_0p1'][i]), 'pass_dy2e_ut_250': bool(recoil_dy2e[i] > 250), 'pass_dy2m_ut_250': bool(recoil_dy2m[i] > 250), 'pass_dy2e_open_high': bool(jet_electron_recoil['open_high'][i]), 'pass_dy2m_open_high': bool(jet_muon_recoil['open_high'][i])})
        for rname in REGION_NAMES:
            row[f'feature_{rname}'] = bool(masks[rname][i])
        rows.append(row)
    summary = {'entries': n, 'materialized_entries': len(rows), 'materialize_skim_flag': materialize_skim_flag, 'materialization_applied': materialization_applied, 'gen_weight_sum': finite_sum(gen_weight), 'gen_weight_sum2': finite_sum(gen_weight * gen_weight), 'data_taking_year': year, 'correction_year': correction_year, 'missing_filters': missing_filters, 'regions': {r: int(np.sum(masks[r])) for r in REGION_NAMES}, 'flat_preselection': {'unweighted': int(np.sum(flat_preselection)), 'definition': 'base_common & zero_tau & any analysis trigger & (MET>250 or photon/dilepton recoil threshold) & HT threshold in at least one object cleaning scheme & >=2 AK4 jets in at least one cleaning scheme'}, 'cutflows': cutflows, 'signal_cutflow_histograms': signal_cutflow_histograms, 'shape_shift': shift, 'dy_mass_window': dy_mass_window, 'trigger_policy': signal_trigger_policy, 'met_shift_status': met_shift_status, 'lumi_mask_source': lumi_mask_source, 'jet_id_source': jet_id_source, 'fatjet_id_source': fatjet_id_source, 'jet_veto_source': jet_veto_source, 'ak4_jec_status': jec_status, 'ak8_jec_status': fjec_status, 'boosted_tagging': {'top_score_branch': BOOSTED_TOP_SCORE_BRANCH, 'top_score_wp': BOOSTED_TOP_SCORE_WP, 'w_score_branch': BOOSTED_W_SCORE_BRANCH, 'w_score_wp': BOOSTED_W_SCORE_WP, 'top_definition': 'pt>400, abs(eta)<2.0, mSD>105, score>WP', 'w_definition': 'pt>200, abs(eta)<2.0, 60<mSD<105, score>WP', 'missing_score_branches': missing_boosted_score_branches}, 'lowdm': {'bjet_pt_threshold_gev': 30.0, 'btagger': 'UParTAK4B', 'loose_wp': UPART_AK4_LOOSE_WP, 'medium_wp': UPART_AK4_MEDIUM_WP, 'nsv_softb_available': bool(sv_available), 'nsv_required_branches': sv_required, 'nres_status': 'not_available_in_current_worker; Nt/NW veto stored, resolved-top veto requires later branch implementation', 'search_bin_definition': 'Run-3 low-dM 42-bin Nsv-inclusive map, with pTb computed from pT>30 medium UParT b jets by descending btag score; ISR-subjet b veto and mTb are diagnostic-only', 'region_counts': {region: int(np.sum(mask)) for region, mask in lowdm_masks.items()}, 'region_recoil_policy': {'SR_LLCR_QCDCR': 'corrected PuppiMET and uncleaned AK4/AK8 collections', 'GCR': 'photon recoil with AK4 dR<0.2 and AK8 dR<0.4 photon cleaning', 'DY2E': 'dielectron recoil with AK4 dR<0.2 and AK8 dR<0.4 electron cleaning', 'DY2M': 'dimuon recoil with AK4 dR<0.2 and AK8 dR<0.4 muon cleaning'}, 'isr_policy': 'exactly one cleaned AK8 ISR candidate, pT>200, abs(eta)<2.4, and deltaPhi(recoil,ISR)>2; the UParTAK4 loose-subjet b-veto decision is stored for diagnostics but is not selected', 'dilepton_policy': 'opposite-sign same-flavor, on-Z 81<mll<101, leading/subleading pT thresholds, and pT(ll)>200', 'photon_policy': 'pT>220, fiducial eta, medium cutBased ID, and explicit Photon_electronVeto'}, 'scale_factor_status': scale_factor_status, 'available_systematics': sorted(weight_variations), 'genmodel_branch_count': len(genmodel_branches)}
    return (rows, summary)
