#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from autonomous_allhad.signal_models import (
    signal_mass_key,
    signal_topology,
)
from autonomous_allhad.sidecar_store import read_root_metadata
from autonomous_allhad.analysis_scale_factors import (
    REQUIRED_ANALYSIS_SF_COMPONENTS,
    REQUIRED_ANALYSIS_SF_VARIATIONS,
)

from autonomous_allhad.real_subset_worker import assign_lowdm_search_bin, compute_weight_bundle
from autonomous_allhad.dy_ptll_policy import dataset_id_prefilter_plan, dy_ptll_dataset_allowed
from autonomous_allhad.highdm_resolved_categories import (
    boosted_overlap_vetoed_ak4_indices,
    select_exclusive_resolved_candidates,
)
from study_trota_highdm_categories_2024 import (
    map_candidates_to_events,
    map_candidates_to_events_rle,
)

RECOIL_PT_BINS = [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0]
LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES = [
    ("Nb0_Nj2to5_PISR500plus", 4),
    ("Nb0_Nj6plus_PISR500plus", 4),
    ("Nb1_PISR300to500_PTb20to40", 4),
    ("Nb1_PISR300to500_PTb40to70", 4),
    ("Nb1_PISR500plus_PTb20to40", 4),
    ("Nb1_PISR500plus_PTb40to70", 4),
    ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", 3),
    ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", 3),
]
LOWDM_42BIN_LABELS = [
    f"{category}_recoil_{position + 1}"
    for category, size in LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES
    for position in range(size)
]
LOWDM_REMOVED_NB0_CATEGORY_SIZES = LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES[:2]
LOWDM_REMOVED_NB0_BIN_COUNT = sum(
    size for _, size in LOWDM_REMOVED_NB0_CATEGORY_SIZES
)
LOWDM_NBGE1_CATEGORY_SIZES = LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES[2:]
LOWDM_34BIN_LABELS = [
    f"{category}_recoil_{position + 1}"
    for category, size in LOWDM_NBGE1_CATEGORY_SIZES
    for position in range(size)
]
SELECTED_AN17_RECOIL_BINS_1BASED = [4, 5, 8, 9, 14, 15, 16]
SELECTED_RECOIL54_SCHEME = "boosted_an17_selected_recoil6_with_nt0_wsplit_SR"
EXTENDED_RECOIL60_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"
EXTENDED_RECOIL60_CATEGORY_KEY = "Nb2_Nt2plus_W0"
EXTENDED_RECOIL60_INSERTION_BIN = 36
NT0_RECOIL_CATEGORY_KEYS = ["NT0_Nb1plus_T0_W0", "NT0_Nb1plus_T0_W1plus"]
RECOIL_BIN_LABELS = [
    f"{int(RECOIL_PT_BINS[i])}-{int(RECOIL_PT_BINS[i + 1])}"
    for i in range(len(RECOIL_PT_BINS) - 1)
]
LOWDM_REGION_MAP = {
    "LLCR": "cat2_LLCR_lowDeltaM",
    "QCDCR": "cat3_QCDCR_lowDeltaM",
    "GCR": "cat4_GCR_lowDeltaM",
    "DY2E": "cat5_DY2E_lowDeltaM",
    "DY2M": "cat6_DY2M_lowDeltaM",
    "SR": "cat7_SR_lowDeltaM",
}
BASE_REGION_VARIABLES = {
    "LLCR": ("feature_LLCR", "met"),
    "QCDCR": ("feature_QCDCR", "met"),
    "GCR": ("feature_GCR", "recoil_gcr"),
    "DY2E": ("feature_DY2E", "recoil_dy2e"),
    "DY2M": ("feature_DY2M", "recoil_dy2m"),
    "HighDMVR_Nb1": ("pass_base_common", "met"),
    "HighDMVR_Nb2": ("pass_base_common", "met"),
    "HighDMVR_Nb3plus": ("pass_base_common", "met"),
    "SR": ("feature_SR", "met"),
    "SR_Nt1": ("feature_SR_Nt1", "met"),
}
NTOP_SPLIT_BASE_REGIONS = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR"]
REGION_VARIABLES = dict(BASE_REGION_VARIABLES)
for _region in NTOP_SPLIT_BASE_REGIONS:
    _flag, _var = BASE_REGION_VARIABLES[_region]
    REGION_VARIABLES.setdefault(f"{_region}_Nt1", (_flag, _var))
    REGION_VARIABLES.setdefault(f"{_region}_Nt0", (_flag, _var))
DATA_PROCESSES = {"JetMET", "EGamma", "Muon"}
DATA_PROCESS_BY_REGION = {
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
    "HighDMVR_Nb1": "JetMET",
    "HighDMVR_Nb2": "JetMET",
    "HighDMVR_Nb3plus": "JetMET",
    "SR": "JetMET",
    "SR_Nt1": "JetMET",
}
for _region in NTOP_SPLIT_BASE_REGIONS:
    _process = DATA_PROCESS_BY_REGION[_region]
    DATA_PROCESS_BY_REGION.setdefault(f"{_region}_Nt1", _process)
    DATA_PROCESS_BY_REGION.setdefault(f"{_region}_Nt0", _process)
for _region, _channel in LOWDM_REGION_MAP.items():
    DATA_PROCESS_BY_REGION.setdefault(_channel, DATA_PROCESS_BY_REGION[_region])
SEARCH_BIN_ORDER = [
    "B0_Nb1", "B0_Nb2plus",
    "Nb1_T0_W1plus", "Nb1_T1plus_W0", "Nb1_T1plus_W1plus",
    "Nb2_T0_W1", "Nb2_T0_W2", "Nb2_T1_W0", "Nb2_T1_W1", "Nb2_T2_W0", "Nb2_TW_ge3",
    "Nb3plus_T0_W1", "Nb3plus_T0_W2", "Nb3plus_T1_W0", "Nb3plus_T1_W1", "Nb3plus_T2_W0", "Nb3plus_TW_ge3",
]
SEARCH_BIN_BRANCHES = ["nb_medium", "nboosted_top", "nboosted_w", "nboosted_total"]
WEIGHT_BRANCHES = [
    "run", "luminosityBlock", "event", "entry", "dataset_id", "year", "mStop", "mLSP",
    "is_data", "is_signal", "is_background", "gen_weight", "pu_ntrueint",
    "n_e_veto", "n_e_medium", "n_m_loose", "n_m_medium",
    "good_jet_pt", "good_jet_eta", "good_jet_hadron_flavour", "good_jet_b_medium",
    "electron_veto_pt", "electron_veto_eta_sc", "electron_veto_phi",
    "electron_veto_eta",
    "electron_medium_pt", "electron_medium_eta_sc", "electron_medium_phi",
    "electron_medium_eta",
    "muon_loose_pt", "muon_loose_eta", "muon_loose_phi",
    "muon_medium_pt", "muon_medium_eta", "muon_medium_phi",
    "photon_medium_pt", "photon_medium_eta", "photon_medium_phi",
    "photon_pt_all", "photon_eta_all", "photon_phi_all", "photon_r9_all",
    "photon_cutbased_all", "photon_electron_veto_all", "gen_top_pt",
]
OPTIONAL_FORWARD_SCHEMA_BRANCHES = {
    "electron_veto_eta",
    "electron_medium_eta",
}
GCR_PHOTON_POLICY_BRANCHES = [
    "photon_pt_all",
    "photon_eta_all",
    "photon_cutbased_all",
    "photon_electron_veto_all",
]
GCR_PREFILTER_BRANCHES = [
    "dataset_id",
    "feature_GCR",
    "feature_lowdm_GCR",
    "nb_medium_lowdm",
]
LOWDM_READ_BRANCHES = [
    "feature_lowdm_preselection",
    "feature_lowdm_LLCR", "feature_lowdm_QCDCR", "feature_lowdm_GCR",
    "feature_lowdm_DY2E", "feature_lowdm_DY2M", "feature_lowdm_SR",
    "lowdm_search_bin_LLCR", "lowdm_search_bin_QCDCR", "lowdm_search_bin_GCR",
    "lowdm_search_bin_DY2E", "lowdm_search_bin_DY2M", "lowdm_search_bin_SR",
    "pass_base_common", "pass_signal_trigger", "pass_photon_trigger", "pass_electron_trigger", "pass_muon_trigger",
    "pass_zero_tau", "pass_no_veto_leptons", "pass_one_veto_lepton", "pass_mt_100",
    "pass_met_250", "pass_ht_300", "pass_ht_photon_300", "pass_ht_lepton_300",
    "pass_open_pre", "pass_qcd_open", "pass_dphi123_0p1",
    "pass_lowdm_topology_veto", "pass_lowdm_isr", "pass_lowdm_isr_bveto",
    "pass_lowdm_met_sqrt_ht", "pass_lowdm_mtb",
    "j1_met_dphi", "j2_met_dphi", "j3_met_dphi", "j4_met_dphi",
    "met", "ht", "njet", "nb_medium_lowdm", "nb_loose_lowdm", "n_sv_softb", "n_photon_medium",
    "njet_photon_clean", "nb_photon_clean", "ht_photon_clean",
    "njet_lepton_clean", "nb_lepton_clean", "ht_lepton_clean",
    "mee", "pee", "mmm", "pmm", "recoil_gcr", "recoil_dy2e", "recoil_dy2m",
    "lowdm_mtb", "lowdm_met_sqrt_ht", "lowdm_isr_pt", "lowdm_isr_dphi", "lowdm_ptb", "n_lowdm_isr",
    "lowdm_fatjet_pt", "lowdm_fatjet_msd",
]
TROTA_LOWDM_LIGHT_BRANCHES = (
    "run", "luminosityBlock", "event", "file_id", "entry",
    "feature_lowdm_preselection", "feature_lowdm_LLCR",
    "feature_lowdm_QCDCR", "feature_lowdm_GCR", "feature_lowdm_DY2E",
    "feature_lowdm_DY2M", "feature_lowdm_SR", "nb_medium_lowdm",
    "pass_lowdm_topology_veto", "pass_lowdm_isr",
    "pass_lowdm_met_sqrt_ht",
)
TROTA_LOWDM_OVERLAP_BRANCHES = (
    "jet_source_index_all", "jet_eta_all", "jet_phi_all",
    "fatjet_eta_all", "fatjet_phi_all", "fatjet_subjet_index1_all",
    "fatjet_subjet_index2_all", "fatjet_boosted_top_pass_all",
    "fatjet_boosted_w_pass_all", "subjet_eta_all", "subjet_phi_all",
)
TROTA_PRIMARY_BRANCHES = (
    "file_id", "entry", "TopResolved1pct_candidateIndex",
    "TopResolved1pct_sourceJetIdx0", "TopResolved1pct_sourceJetIdx1",
    "TopResolved1pct_sourceJetIdx2", "TopResolved1pct_eta",
    "TopResolved1pct_mass", "TopResolved1pct_QCDDiscriminant",
)
TROTA_FALLBACK_BRANCHES = (
    "run", "luminosityBlock", "event", *TROTA_PRIMARY_BRANCHES[2:],
)
EXPECTED_TROTA_SCHEMA_BY_YEAR = {
    "2024": "trota_topresolved_2024_inplace_sparse_v1",
    "2025": "trota_topresolved_2025_inplace_sparse_v1",
}
EXPECTED_TROTA_MODEL_SHA256 = (
    "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
)
DERIVED_NRES_BRANCH = "nresolved_top_trota"
LOWDM_VARIABLE_SPECS = {
    "met": {"branch": "met", "bins": [0, 100, 150, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"$p_{T}^{miss}$ (GeV)"},
    "ht": {"branch": "ht", "bins": [0, 300, 500, 700, 1000, 1500, 2000, 3000], "xlabel": r"$H_{T}$ (GeV)"},
    "njet": {"branch": "njet", "bins": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 12.5], "xlabel": r"$N_{j}$"},
    "nb_medium_lowdm": {"branch": "nb_medium_lowdm", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5], "xlabel": r"$N_{b}$ medium"},
    "nb_loose_lowdm": {"branch": "nb_loose_lowdm", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5], "xlabel": r"$N_{b}$ loose"},
    "n_e_veto": {"branch": "n_e_veto", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5], "xlabel": r"$N_{e}$ veto"},
    "n_m_loose": {"branch": "n_m_loose", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5], "xlabel": r"$N_{\mu}$ loose"},
    "lowdm_mtb": {"branch": "lowdm_mtb", "bins": [0, 50, 100, 150, 200, 300, 500, 800, 1200], "xlabel": r"low-$\Delta m$ $m_{T}^{b}$ (GeV)"},
    "lowdm_met_sqrt_ht": {"branch": "lowdm_met_sqrt_ht", "bins": [0, 5, 10, 15, 20, 25, 30, 40, 60], "xlabel": r"$p_{T}^{miss}/\sqrt{H_{T}}$"},
    "lowdm_isr_pt": {"branch": "lowdm_isr_pt", "bins": [0, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"ISR $p_{T}$ (GeV)"},
    "lowdm_isr_dphi": {"branch": "lowdm_isr_dphi", "bins": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.1416], "xlabel": r"$\Delta\phi$(ISR, $p_{T}^{miss}$)"},
    "lowdm_ptb": {"branch": "lowdm_ptb", "bins": [0, 30, 60, 100, 150, 200, 300, 500, 800], "xlabel": r"low-$\Delta m$ $p_{T}^{b}$ (GeV)"},
    "n_lowdm_isr": {"branch": "n_lowdm_isr", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5], "xlabel": r"$N_{ISR}$"},
    "recoil_gcr": {"branch": "recoil_gcr", "bins": [0, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"Photon recoil $p_{T}$ (GeV)"},
    "recoil_dy2e": {"branch": "recoil_dy2e", "bins": [0, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"Dielectron recoil $p_{T}$ (GeV)"},
    "recoil_dy2m": {"branch": "recoil_dy2m", "bins": [0, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"Dimuon recoil $p_{T}$ (GeV)"},
    "mee": {"branch": "mee", "bins": [50, 70, 81, 86, 91, 96, 101, 120, 150], "xlabel": r"$m_{ee}$ (GeV)"},
    "mmm": {"branch": "mmm", "bins": [50, 70, 81, 86, 91, 96, 101, 120, 150], "xlabel": r"$m_{\mu\mu}$ (GeV)"},
    "n_photon_medium": {"branch": "n_photon_medium", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5], "xlabel": r"$N_{\gamma}$ medium"},
    "njet_photon_clean": {"branch": "njet_photon_clean", "bins": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 12.5], "xlabel": r"$N_{j}$ photon-cleaned"},
    "nb_photon_clean": {"branch": "nb_photon_clean", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5], "xlabel": r"$N_{b}$ photon-cleaned"},
    "ht_photon_clean": {"branch": "ht_photon_clean", "bins": [0, 300, 500, 700, 1000, 1500, 2000, 3000], "xlabel": r"Photon-cleaned $H_{T}$ (GeV)"},
    "njet_lepton_clean": {"branch": "njet_lepton_clean", "bins": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 12.5], "xlabel": r"$N_{j}$ lepton-cleaned"},
    "nb_lepton_clean": {"branch": "nb_lepton_clean", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5], "xlabel": r"$N_{b}$ lepton-cleaned"},
    "ht_lepton_clean": {"branch": "ht_lepton_clean", "bins": [0, 300, 500, 700, 1000, 1500, 2000, 3000], "xlabel": r"Lepton-cleaned $H_{T}$ (GeV)"},
    "leading_lowdm_fatjet_pt": {"branch": "lowdm_fatjet_pt", "source": "first", "bins": [0, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"Leading low-$\Delta m$ AK8 $p_{T}$ (GeV)"},
    "leading_lowdm_fatjet_msd": {"branch": "lowdm_fatjet_msd", "source": "first", "bins": [0, 30, 50, 70, 90, 110, 140, 200, 300], "xlabel": r"Leading low-$\Delta m$ AK8 $m_{SD}$ (GeV)"},
}
COMMON_LOWDM_VARIABLES = [
    "met", "ht", "njet", "nb_medium_lowdm", "nb_loose_lowdm", "n_e_veto", "n_m_loose",
    "lowdm_mtb", "lowdm_met_sqrt_ht", "lowdm_isr_pt", "lowdm_isr_dphi", "lowdm_ptb", "n_lowdm_isr",
    "leading_lowdm_fatjet_pt", "leading_lowdm_fatjet_msd",
]
LOWDM_REGION_VARIABLES = {
    "SR": COMMON_LOWDM_VARIABLES,
    "LLCR": COMMON_LOWDM_VARIABLES,
    "QCDCR": COMMON_LOWDM_VARIABLES,
    "GCR": COMMON_LOWDM_VARIABLES + ["n_photon_medium", "recoil_gcr", "njet_photon_clean", "nb_photon_clean", "ht_photon_clean"],
    "DY2E": COMMON_LOWDM_VARIABLES + ["recoil_dy2e", "mee", "njet_lepton_clean", "nb_lepton_clean", "ht_lepton_clean"],
    "DY2M": COMMON_LOWDM_VARIABLES + ["recoil_dy2m", "mmm", "njet_lepton_clean", "nb_lepton_clean", "ht_lepton_clean"],
}
HIGHDM_DISTRIBUTION_VARIABLE_SPECS = {
    "nb": {
        "branch": "nb_medium",
        "branch_by_region": {"GCR": "nb_photon_clean", "DY2E": "nb_lepton_clean", "DY2M": "nb_lepton_clean"},
        "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "overflow_policy": "exclude",
        "xlabel": r"$N_{b}$",
    },
    "njet": {
        "branch": "njet",
        "branch_by_region": {"GCR": "njet_photon_clean", "DY2E": "njet_lepton_clean", "DY2M": "njet_lepton_clean"},
        "bins": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 10.5, 12.5, 16.5],
        "overflow_policy": "exclude",
        "xlabel": r"$N_{j}$",
    },
    "nfatjet": {
        "branch": "nfj",
        "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "xlabel": r"$N_{fj}$",
    },
    "ntop": {"branch": "nboosted_top", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5], "xlabel": r"$N_{top}$"},
    "nw": {"branch": "nboosted_w", "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5], "xlabel": r"$N_{W}$"},
    "ht": {
        "branch": "ht",
        "branch_by_region": {"GCR": "ht_photon_clean", "DY2E": "ht_lepton_clean", "DY2M": "ht_lepton_clean"},
        "bins": [0, 300, 500, 700, 900, 1200, 1500, 2000, 2500, 3000],
        "overflow_policy": "exclude",
        "xlabel": r"$H_{T}$ (GeV)",
    },
    "ut": {
        "branch": "met",
        "branch_by_region": {"GCR": "recoil_gcr", "DY2E": "recoil_dy2e", "DY2M": "recoil_dy2m"},
        "regions": ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"],
        "bins": [250, 300, 350, 400, 500, 650, 800, 1000, 1500],
        "overflow_policy": "exclude",
        "xlabel": r"$U_{T}$ (GeV)",
    },
    "ptll": {
        "branch": "pee",
        "branch_by_region": {"DY2E": "pee", "DY2M": "pmm"},
        "regions": ["DY2E", "DY2M"],
        "bins": [200, 210, 220, 240, 260, 300, 350, 400, 500, 650, 800, 1000, 1500],
        "overflow_policy": "exclude",
        "xlabel": r"$p_{T}(\ell\ell)$ (GeV)",
    },
    "met": {"branch": "met", "bins": [250, 300, 350, 400, 500, 650, 800, 1000, 1500], "xlabel": r"$p_{T}^{miss}$ (GeV)"},
    "jet_pt": {
        "branch": "j1pt",
        "bins": [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 800, 1200, 1600],
        "xlabel": r"Leading Jet $p_{T}$ (GeV)",
    },
    "fatjet_pt": {
        "branch": "fj1pt",
        "bins": [200, 250, 300, 350, 400, 500, 650, 800, 1000, 1500],
        "xlabel": r"Leading FatJet $p_{T}$ (GeV)",
    },
    "bjet_pt": {
        "branch": "good_jet_pt", "mask_branch": "good_jet_b_medium", "source": "masked_first",
        "bins": [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 800, 1200],
        "xlabel": r"Leading b-jet $p_{T}$ (GeV)",
    },
}

HIGHDM_CR_REGIONS = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"]
HIGHDM_VR_REGIONS = ["HighDMVR_Nb1", "HighDMVR_Nb2", "HighDMVR_Nb3plus"]
HIGHDM_SR_CATEGORY_KEYS = [
    "SR_Nb1plus_T0_W0", "SR_Nb1plus_T0_W1plus",
    "SR_Nb1_T1plus_W0", "SR_Nb1_T1plus_W1plus",
    "SR_Nb2_T1_W0", "SR_Nb2_T1_W1",
    "SR_Nb3plus_T1_W0", "SR_Nb3plus_T1_W1", "SR_Nb3plus_T2_W0",
    "SR_Nb2_Nt2plus_W0",
]

READ_BRANCHES = sorted(set(
    WEIGHT_BRANCHES + SEARCH_BIN_BRANCHES + LOWDM_READ_BRANCHES
    + [spec["branch"] for spec in LOWDM_VARIABLE_SPECS.values()]
    + [spec["branch"] for spec in HIGHDM_DISTRIBUTION_VARIABLE_SPECS.values()]
    + [spec["mask_branch"] for spec in HIGHDM_DISTRIBUTION_VARIABLE_SPECS.values() if spec.get("mask_branch")]
    + [branch for spec in HIGHDM_DISTRIBUTION_VARIABLE_SPECS.values() for branch in (spec.get("branch_by_region") or {}).values()]
    + [b for pair in REGION_VARIABLES.values() for b in pair]
))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EXECUTION_CONTRACT_COMMON_PATHS = (
    "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
    "autonomous_allhad/workflow/run_flat_hists_chunked.py",
    "autonomous_allhad/autonomous_allhad/analysis_scale_factors.py",
    "autonomous_allhad/autonomous_allhad/real_subset_worker.py",
    "autonomous_allhad/autonomous_allhad/dy_ptll_policy.py",
    "autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py",
    "autonomous_allhad/workflow/study_trota_highdm_categories_2024.py",
    "analysis/utils/corrections.py",
    "analysis/data/corrections.coffea",
)
EXECUTION_CONTRACT_YEAR_PATHS = {
    "2024": (
        "analysis/data/PUweight/2024/puWeights.json.gz",
        "analysis/data/BTVSF/2024/btagging.json.gz",
        "analysis/data/EGammaSF/2024/electron.json.gz",
        "analysis/data/EGammaSF/2024/electronHlt.json.gz",
        "analysis/data/EGammaSF/2024/photon.json.gz",
        "analysis/data/MuonSF/2024/muon_Z.json.gz",
        "analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz",
        "analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz",
        "analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz",
        "analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz",
    ),
    "2025": (
        "analysis/data/PUweight/2025/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
        "analysis/data/BTVSF/2025/btagging.json.gz",
        "analysis/data/EGammaSF/2025/electron.json.gz",
        "analysis/data/EGammaSF/2025/photon.json.gz",
        "analysis/data/MuonSF/2025/muon_Z.json.gz",
        "analysis/data/AnalysisSF/2025/met_trigger_sf.json.gz",
        "analysis/data/AnalysisSF/2025/photon_trigger_sf.json.gz",
        "analysis/data/AnalysisSF/2025/veto_electron_5to10_sf.json.gz",
        "analysis/data/AnalysisSF/2025/loose_muon_5to10_sf.json.gz",
    ),
}
EXPECTED_BTAG_EFFICIENCY_SHA256_2024 = (
    "03524e9ae28110814f336eafc887e60d54b495a7b8dec7cda59bd792f56feaf4"
)
BTAG_EFFICIENCY_RELATIVE_PATHS = {
    "2024": "analysis/hists/btageff2024.merged",
    "2025": "analysis/hists/btageff2025.merged",
}


def execution_code_sha256(repo: Path, campaign_year: str) -> dict[str, str]:
    return {
        relative_path: file_sha256(repo / relative_path)
        for relative_path in (
            EXECUTION_CONTRACT_COMMON_PATHS
            + EXECUTION_CONTRACT_YEAR_PATHS[campaign_year]
        )
    }


def btag_efficiency_contract(
    repo: Path,
    expected_sha256: str,
    required: bool,
    campaign_year: str,
) -> dict[str, Any]:
    relative_path = BTAG_EFFICIENCY_RELATIVE_PATHS[campaign_year]
    path = repo / relative_path
    if not path.exists():
        if required:
            raise RuntimeError(f"required b-tag efficiency payload is missing: {path}")
        return {
            "path": relative_path,
            "exists": False,
            "expected_sha256": expected_sha256,
        }
    actual_sha256 = file_sha256(path)
    matches = not expected_sha256 or actual_sha256 == expected_sha256
    if required and not matches:
        raise RuntimeError(
            f"b-tag efficiency SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return {
        "path": relative_path,
        "exists": True,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "matches_expected": matches,
    }


MAX_ABS_HIST_WEIGHT = 1.0e12
SIGNAL_BTAG_FASTSIM_MASSES = (
    600, 700, 800, 900, 950,
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
)
SIGNAL_BTAG_EFFICIENCY_DATASETS = {
    mstop: (
        f"SMS-2Stop_Par-mStop-{mstop}_TuneCP5_13p6TeV_madgraphMLM-pythia8-"
        "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v1"
    )
    for mstop in SIGNAL_BTAG_FASTSIM_MASSES
}


def signal_btag_efficiency_dataset(
    mstop: int,
    source_dataset: str = "",
) -> tuple[int, str]:
    topology = signal_topology(source_dataset)
    if topology in {"T2tb", "T2bW"}:
        parts = str(source_dataset).strip("/").split("/")
        if len(parts) >= 2:
            return int(mstop), f"{parts[0]}-{parts[1]}"
        raise RuntimeError(
            f"cannot derive b-tag efficiency key from signal dataset {source_dataset!r}"
        )
    anchor = min(SIGNAL_BTAG_EFFICIENCY_DATASETS, key=lambda value: (abs(value - int(mstop)), value))
    return anchor, SIGNAL_BTAG_EFFICIENCY_DATASETS[anchor]


def finite_array(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, fill, dtype=float)
    if out.shape == ():
        out = np.full(n, float(out), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill)


def as_bool(values: Any, n: int) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=bool)
    except Exception:
        return np.zeros(n, dtype=bool)
    if out.shape == ():
        out = np.full(n, bool(out), dtype=bool)
    if len(out) != n:
        return np.zeros(n, dtype=bool)
    return out


def gcr_photon_policy_mask(
    chunk: Any,
    policy: str,
    n: int,
) -> np.ndarray:
    if policy == "nominal":
        return np.ones(n, dtype=bool)
    if policy != "tight_eb":
        raise ValueError(f"unknown GCR photon policy: {policy}")
    required = set(GCR_PHOTON_POLICY_BRANCHES)
    present = set(ak.fields(chunk))
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(
            "tight-EB GCR photon policy requires missing branches: "
            + ", ".join(missing)
        )
    tight_eb = (
        (chunk["photon_pt_all"] > 220.0)
        & (abs(chunk["photon_eta_all"]) < 1.4442)
        & (chunk["photon_cutbased_all"] >= 3)
        & ak.values_astype(chunk["photon_electron_veto_all"], np.bool_)
    )
    return np.asarray(ak.sum(tight_eb, axis=1) == 1, dtype=bool)


def gcr_region_masks(
    chunk: Any,
    photon_policy: str,
) -> dict[str, np.ndarray]:
    n = len(chunk["dataset_id"])
    nominal_highdm = as_bool(chunk["feature_GCR"], n)
    nominal_lowdm = (
        as_bool(chunk["feature_lowdm_GCR"], n)
        & (np.asarray(chunk["nb_medium_lowdm"], dtype=int) >= 1)
    )
    photon_mask = gcr_photon_policy_mask(chunk, photon_policy, n)
    return {
        "nominal_highdm": nominal_highdm,
        "nominal_lowdm": nominal_lowdm,
        "selected_highdm": nominal_highdm & photon_mask,
        "selected_lowdm": nominal_lowdm & photon_mask,
    }


def record_gcr_selection_audit(
    summary: dict[str, Any],
    meta: dict[str, Any],
    dataset_ids: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    for dataset_id in sorted(set(int(value) for value in dataset_ids)):
        dataset_mask = dataset_ids == dataset_id
        dataset, process, is_data, is_signal = dataset_label(meta, dataset_id)
        label = (
            "data_obs"
            if is_data
            else process_to_group(process, dataset)
            if not is_signal
            else "signal"
        )
        record = (
            summary
            .setdefault("gcr_photon_selection_audit", {})
            .setdefault(label, {})
            .setdefault(dataset, {})
        )
        for region in ("highdm", "lowdm"):
            nominal = int(
                np.count_nonzero(
                    dataset_mask & masks[f"nominal_{region}"]
                )
            )
            selected = int(
                np.count_nonzero(
                    dataset_mask & masks[f"selected_{region}"]
                )
            )
            region_record = record.setdefault(
                region,
                {"nominal_entries": 0, "selected_entries": 0},
            )
            region_record["nominal_entries"] = (
                int(region_record.get("nominal_entries", 0)) + nominal
            )
            region_record["selected_entries"] = (
                int(region_record.get("selected_entries", 0)) + selected
            )


def ones_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.ones_like(jagged), np.bool_)


def zeros_mask(jagged: Any) -> Any:
    return ak.values_astype(ak.zeros_like(jagged), np.bool_)


def combine_two(a: Any, b: Any) -> Any:
    return ak.concatenate([a, b], axis=1)


def flat_arrays_for_weights(chunk: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(chunk["gen_weight"])
    jet_pt = chunk["good_jet_pt"]
    jet_eta = chunk["good_jet_eta"]
    jet_had = chunk["good_jet_hadron_flavour"]
    b_med = ak.values_astype(chunk["good_jet_b_medium"], np.bool_)

    ev_pt = chunk["electron_veto_pt"]
    ev_eta_sc = chunk["electron_veto_eta_sc"]
    ev_phi = chunk["electron_veto_phi"]
    em_pt = chunk["electron_medium_pt"]
    em_eta_sc = chunk["electron_medium_eta_sc"]
    em_phi = chunk["electron_medium_phi"]
    raw_eta_available = (
        "electron_veto_eta" in chunk and "electron_medium_eta" in chunk
    )
    ev_eta = chunk["electron_veto_eta"] if raw_eta_available else ev_eta_sc
    em_eta = chunk["electron_medium_eta"] if raw_eta_available else em_eta_sc
    e_pt = combine_two(ev_pt, em_pt)
    e_eta = combine_two(ev_eta, em_eta)
    e_eta_sc = combine_two(ev_eta_sc, em_eta_sc)
    e_phi = combine_two(ev_phi, em_phi)
    e_veto = combine_two(ones_mask(ev_pt), zeros_mask(em_pt))
    e_med = combine_two(zeros_mask(ev_pt), ones_mask(em_pt))
    e_delta_eta_sc = e_eta_sc - e_eta

    ml_pt, ml_eta, ml_phi = chunk["muon_loose_pt"], chunk["muon_loose_eta"], chunk["muon_loose_phi"]
    mm_pt, mm_eta, mm_phi = chunk["muon_medium_pt"], chunk["muon_medium_eta"], chunk["muon_medium_phi"]
    m_pt = combine_two(ml_pt, mm_pt)
    m_eta = combine_two(ml_eta, mm_eta)
    m_phi = combine_two(ml_phi, mm_phi)
    m_loose = combine_two(ones_mask(ml_pt), zeros_mask(mm_pt))
    m_med = combine_two(zeros_mask(ml_pt), ones_mask(mm_pt))

    photon_all_fields = {
        "photon_pt_all", "photon_eta_all", "photon_phi_all", "photon_r9_all",
        "photon_cutbased_all", "photon_electron_veto_all",
    }
    if photon_all_fields <= set(chunk):
        p_pt = chunk["photon_pt_all"]
        p_eta = chunk["photon_eta_all"]
        p_phi = chunk["photon_phi_all"]
        p_r9 = chunk["photon_r9_all"]
        p_med = (
            (p_pt > 220.0)
            & ((abs(p_eta) < 1.4442) | ((abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5)))
            & (chunk["photon_cutbased_all"] >= 2)
            & ak.values_astype(chunk["photon_electron_veto_all"], np.bool_)
        )
    else:
        p_pt = chunk["photon_medium_pt"]
        p_eta = chunk["photon_medium_eta"]
        p_phi = chunk["photon_medium_phi"]
        p_r9 = None
        p_med = ones_mask(p_pt)

    top_pt = chunk["gen_top_pt"]
    top_flags = ak.values_astype(ak.ones_like(top_pt) * ((1 << 8) | (1 << 13)), np.int64)
    top_pdg = ak.values_astype(ak.ones_like(top_pt) * 6, np.int64)
    arrays = ak.Array({
        "genWeight": chunk["gen_weight"],
        "Pileup_nTrueInt": chunk["pu_ntrueint"],
        "Jet_hadronFlavour": jet_had,
        "GenPart_pt": top_pt,
        "GenPart_pdgId": top_pdg,
        "GenPart_statusFlags": top_flags,
    })
    inputs = {
        "n": n,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_hadflav": jet_had,
        "b_med": b_med,
        "e_eta": e_eta,
        "e_delta_eta_sc": e_delta_eta_sc,
        "e_pt": e_pt,
        "e_phi": e_phi,
        "e_veto": e_veto,
        "e_med": e_med,
        "n_e_veto": np.asarray(chunk["n_e_veto"], dtype=int),
        "n_e_med": np.asarray(chunk["n_e_medium"], dtype=int),
        "m_eta": m_eta,
        "m_pt": m_pt,
        "m_phi": m_phi,
        "m_loose": m_loose,
        "m_med": m_med,
        "n_m_loose": np.asarray(chunk["n_m_loose"], dtype=int),
        "n_m_med": np.asarray(chunk["n_m_medium"], dtype=int),
        "p_eta": p_eta,
        "p_pt": p_pt,
        "p_phi": p_phi,
        "p_r9": p_r9,
        "p_med": p_med,
        "gcr_mask": (
            as_bool(chunk["feature_GCR"], n)
            | as_bool(chunk["feature_lowdm_GCR"], n)
        ),
        "met_pt": np.asarray(chunk["met"], dtype=float),
        "met_trigger_mask": (
            as_bool(chunk["feature_LLCR"], n)
            | as_bool(chunk["feature_QCDCR"], n)
            | as_bool(chunk["feature_SR"], n)
            | as_bool(chunk["feature_lowdm_LLCR"], n)
            | as_bool(chunk["feature_lowdm_QCDCR"], n)
            | as_bool(chunk["feature_lowdm_SR"], n)
            | region_mask(chunk, "HighDMVR_Nb1", "pass_base_common", n)
            | region_mask(chunk, "HighDMVR_Nb2", "pass_base_common", n)
            | region_mask(chunk, "HighDMVR_Nb3plus", "pass_base_common", n)
        ),
        "electron_eta_source": (
            "raw_eta_with_delta_eta_sc"
            if raw_eta_available
            else "eta_sc_fallback_current_schema"
        ),
    }
    return arrays, inputs


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


def data_process_allowed(process: str, region: str) -> bool:
    expected = DATA_PROCESS_BY_REGION.get(region)
    return expected is None or process == expected


def note_data_exclusion(summary: dict[str, Any], region: str, process: str, count: int) -> None:
    if count <= 0:
        return
    rec = summary.setdefault("data_stream_exclusions", {}).setdefault(region, {}).setdefault(process, {"entries": 0})
    rec["entries"] = int(rec.get("entries", 0)) + int(count)


def dataset_label(meta: dict[str, Any], dataset_id: int) -> tuple[str, str, bool, bool]:
    rec = (meta.get("datasets") or {}).get(str(int(dataset_id))) or {}
    process = str(rec.get("process") or "unknown")
    dataset = str(rec.get("dataset") or "unknown")
    is_data = bool(rec.get("is_data"))
    is_signal = bool(rec.get("is_signal"))
    if not is_data and not is_signal:
        process = canonical_process(process, dataset)
    return dataset, process, is_data, is_signal


def norm_vector(
    norm: dict[str, Any],
    chunk: dict[str, Any],
    dataset_id: int,
    dataset: str,
    is_data: bool,
    is_signal: bool,
    require_normalization: bool = False,
) -> np.ndarray:
    n = len(chunk["dataset_id"])
    if is_data:
        return np.ones(n, dtype=float)
    if is_signal:
        out = np.zeros(n, dtype=float)
        mstops = np.asarray(chunk["mStop"], dtype=int)
        mlsps = np.asarray(chunk["mLSP"], dtype=int)
        topology = signal_topology(dataset)
        for i, (ms, ml) in enumerate(zip(mstops, mlsps)):
            key = signal_mass_key(topology or "T2tt", int(ms), int(ml))
            fac = ((norm.get("signal_mass_points") or {}).get(key) or {}).get("normalization_factor")
            try:
                finite_factor = fac is not None and math.isfinite(float(fac))
            except (TypeError, ValueError, OverflowError):
                finite_factor = False
            positive_factor = finite_factor and float(fac) > 0.0
            if not finite_factor or (require_normalization and not positive_factor):
                if require_normalization:
                    raise RuntimeError(
                        "missing, non-finite, or non-positive signal "
                        f"normalization factor for {key}"
                    )
                continue
            out[i] = float(fac)
        return out
    fac = ((norm.get("dataset_factors") or {}).get(str(int(dataset_id))) or {}).get("normalization_factor")
    try:
        finite_factor = fac is not None and math.isfinite(float(fac))
    except (TypeError, ValueError, OverflowError):
        finite_factor = False
    positive_factor = finite_factor and float(fac) > 0.0
    if not finite_factor or (require_normalization and not positive_factor):
        if require_normalization:
            raise RuntimeError(
                "missing, non-finite, or non-positive background "
                "normalization factor for "
                f"dataset_id={int(dataset_id)}"
            )
        return np.zeros(n, dtype=float)
    return np.full(n, float(fac), dtype=float)


def histogram_variations(
    variations: dict[str, Any],
    nominal_only: bool,
) -> dict[str, Any]:
    if not nominal_only:
        return variations
    if "nominal" not in variations:
        raise RuntimeError("nominal-only histogramming requested but nominal weight is absent")
    return {"nominal": variations["nominal"]}


def sample_label(
    process: str,
    is_data: bool,
    is_signal: bool,
    chunk: dict[str, Any],
    dataset: str = "",
) -> str:
    if is_data:
        return "data_obs"
    if is_signal:
        topology = signal_topology(dataset) or "T2tt"
        mstops = np.asarray(chunk["mStop"], dtype=int)
        mlsps = np.asarray(chunk["mLSP"], dtype=int)
        if len(mstops):
            pairs, counts = np.unique(np.stack([mstops, mlsps], axis=1), axis=0, return_counts=True)
            pair = pairs[int(np.argmax(counts))]
            return f"{topology}_mStop{int(pair[0])}_mLSP{int(pair[1])}"
        return f"{topology}_unknown"
    return process


def record_scale_factor_audit(
    summary: dict[str, Any],
    label: str,
    dataset: str,
    status: dict[str, Any],
    events: int,
) -> None:
    if events <= 0:
        return
    dataset_record = (
        summary.setdefault("scale_factor_status_audit", {})
        .setdefault(label, {})
        .setdefault("datasets", {})
        .setdefault(
            dataset,
            {
                "events": 0,
                "groups": 0,
                "components": {},
            },
        )
    )
    dataset_record["events"] = int(dataset_record.get("events", 0)) + int(events)
    dataset_record["groups"] = int(dataset_record.get("groups", 0)) + 1
    for component, component_status in sorted(
        (status.get("components") or {}).items()
    ):
        component_record = dataset_record["components"].setdefault(
            component,
            {
                "applied_events": 0,
                "failed_events": 0,
                "reasons": {},
            },
        )
        applied = bool((component_status or {}).get("applied"))
        counter = "applied_events" if applied else "failed_events"
        component_record[counter] = int(component_record.get(counter, 0)) + int(events)
        reason = str(
            (component_status or {}).get("reason")
            or (component_status or {}).get("error")
            or ("applied" if applied else "not_applied")
        )
        reasons = component_record.setdefault("reasons", {})
        reasons[reason] = int(reasons.get(reason, 0)) + int(events)


def empty_hist() -> dict[str, Any]:
    nb = len(RECOIL_PT_BINS) - 1
    return {"sumw": [0.0] * nb, "sumw2": [0.0] * nb, "entries": [0] * nb}


def add_hist(target: dict[str, Any], values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> None:
    vals = np.asarray(values, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    good = np.isfinite(vals) & finite_hist_weight_mask(w)
    if not np.any(good):
        return
    bins = np.asarray(RECOIL_PT_BINS, dtype=float)
    h, _ = np.histogram(vals[good], bins=bins, weights=w[good])
    h2, _ = np.histogram(vals[good], bins=bins, weights=finite_weight_square(w[good]))
    e, _ = np.histogram(vals[good], bins=bins)
    target["sumw"] = (np.asarray(target["sumw"], dtype=float) + h).tolist()
    target["sumw2"] = (np.asarray(target["sumw2"], dtype=float) + h2).tolist()
    target["entries"] = (np.asarray(target["entries"], dtype=int) + e).astype(int).tolist()



def region_mask(chunk: dict[str, Any], region: str, flag: str, n: int) -> np.ndarray:
    if region.startswith("HighDMVR_"):
        j1 = float_field(chunk, "j1_met_dphi", n, 999.0)
        j2 = float_field(chunk, "j2_met_dphi", n, 999.0)
        j3 = float_field(chunk, "j3_met_dphi", n, 999.0)
        j4 = float_field(chunk, "j4_met_dphi", n, 999.0)
        medium_dphi = (
            (j1 > 0.5)
            & (j2 > 0.15)
            & (j3 > 0.15)
            & ((j2 < 0.5) | (j3 < 0.5) | (j4 < 0.5))
        )
        nb = int_field(chunk, "nb_medium", n)
        nb_mask = {
            "HighDMVR_Nb1": nb == 1,
            "HighDMVR_Nb2": nb == 2,
            "HighDMVR_Nb3plus": nb >= 3,
        }[region]
        return (
            bool_field(chunk, "pass_base_common", n)
            & bool_field(chunk, "pass_signal_trigger", n)
            & bool_field(chunk, "pass_zero_tau", n)
            & bool_field(chunk, "pass_no_veto_leptons", n)
            & (int_field(chunk, "njet", n) >= 5)
            & nb_mask
            & bool_field(chunk, "pass_met_250", n)
            & bool_field(chunk, "pass_ht_300", n)
            & medium_dphi
        )
    base = as_bool(chunk[flag], n)
    if region.endswith("_Nt0"):
        return base & (np.asarray(chunk["nboosted_top"], dtype=int) == 0)
    if region.endswith("_Nt1") and region != "SR_Nt1":
        return base & (np.asarray(chunk["nboosted_top"], dtype=int) >= 1)
    return base


def bool_field(chunk: dict[str, Any], name: str, n: int, default: bool = False) -> np.ndarray:
    if name not in chunk:
        return np.full(n, default, dtype=bool)
    return as_bool(chunk[name], n)


def float_field(chunk: dict[str, Any], name: str, n: int, fill: float = 0.0) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=float)
    return finite_array(chunk[name], n, fill)


def int_field(chunk: dict[str, Any], name: str, n: int, fill: int = 0) -> np.ndarray:
    if name not in chunk:
        return np.full(n, fill, dtype=int)
    try:
        out = np.asarray(chunk[name], dtype=int)
    except Exception:
        return np.full(n, fill, dtype=int)
    if out.shape == ():
        out = np.full(n, int(out), dtype=int)
    if len(out) != n:
        return np.full(n, fill, dtype=int)
    return out


def lowdm_nres_zero_mask(chunk: dict[str, Any], n: int) -> np.ndarray:
    """Apply the TROTA resolved-top veto when the derived branch is present."""
    if DERIVED_NRES_BRANCH not in chunk:
        return np.ones(n, dtype=bool)
    return int_field(chunk, DERIVED_NRES_BRANCH, n, -1) == 0


def lowdm_common_mask(chunk: dict[str, Any], n: int) -> np.ndarray:
    return (
        bool_field(chunk, "pass_base_common", n)
        & bool_field(chunk, "pass_zero_tau", n)
        & bool_field(chunk, "pass_lowdm_topology_veto", n)
        & lowdm_nres_zero_mask(chunk, n)
        & bool_field(chunk, "pass_lowdm_isr", n)
        & bool_field(chunk, "pass_lowdm_isr_bveto", n)
        & bool_field(chunk, "pass_lowdm_met_sqrt_ht", n)
        & bool_field(chunk, "pass_lowdm_mtb", n)
    )


def lowdm_region_mask(chunk: dict[str, Any], region: str, n: int) -> np.ndarray:
    if region not in LOWDM_REGION_MAP:
        raise ValueError(f"unknown low-dM region: {region}")
    return (
        bool_field(chunk, f"feature_lowdm_{region}", n)
        & (int_field(chunk, "nb_medium_lowdm", n, -1) >= 1)
        & lowdm_nres_zero_mask(chunk, n)
    )


def lowdm_nbge1_indices(indices: np.ndarray) -> np.ndarray:
    """Drop the two leading Nb=0 categories and remap 42 bins to 34."""
    raw = np.asarray(indices, dtype=int)
    return np.where(raw >= LOWDM_REMOVED_NB0_BIN_COUNT, raw - LOWDM_REMOVED_NB0_BIN_COUNT, -1)


def lowdm_nsv_inclusive_sr_indices(chunk: dict[str, Any], n: int) -> np.ndarray:
    """Rebuild the adopted 34-bin, Nb>=1 SR from the broad intermediate.

    The ISR-subjet b veto and mTb requirement are intentionally not part of the
    adopted SR selection.  Their diagnostic branches remain in the broad
    intermediate for cutflow and comparison studies.
    """
    base = (
        bool_field(chunk, "feature_lowdm_preselection", n)
        & bool_field(chunk, "pass_lowdm_topology_veto", n)
        & lowdm_nres_zero_mask(chunk, n)
        & bool_field(chunk, "pass_lowdm_isr", n)
        & bool_field(chunk, "pass_lowdm_met_sqrt_ht", n)
        & (int_field(chunk, "nb_medium_lowdm", n, -1) >= 1)
    )
    njet = int_field(chunk, "njet", n, -1)
    nb = int_field(chunk, "nb_medium_lowdm", n, -1)
    pisr = float_field(chunk, "lowdm_isr_pt", n, -1.0)
    ptb = float_field(chunk, "lowdm_ptb", n, -1.0)
    met = float_field(chunk, "met", n, -1.0)
    mtb = float_field(chunk, "lowdm_mtb", n, float("nan"))
    out = np.full(n, -1, dtype=int)
    for index in np.flatnonzero(base):
        out[index] = assign_lowdm_search_bin(
            int(njet[index]),
            int(nb[index]),
            0,
            float(pisr[index]),
            float(ptb[index]),
            float(met[index]),
            float(mtb[index]),
        )
    return lowdm_nbge1_indices(out)


def empty_index_hist(nbin: int) -> dict[str, Any]:
    return {"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin}


def empty_binned_hist(edges: list[float]) -> dict[str, Any]:
    nb = max(0, len(edges) - 1)
    return {"sumw": [0.0] * nb, "sumw2": [0.0] * nb, "entries": [0] * nb}


def finite_hist_weight_mask(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return np.isfinite(w) & (np.abs(w) <= MAX_ABS_HIST_WEIGHT)


def finite_weight_square(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.where(finite_hist_weight_mask(w), w, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        squared = w * w
    return np.nan_to_num(squared, nan=0.0, posinf=0.0, neginf=0.0)


def add_binned_hist(
    target: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    edges: list[float],
    *,
    overflow_policy: str = "exclude",
) -> None:
    vals = np.asarray(values, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    good = np.isfinite(vals) & finite_hist_weight_mask(w)
    if not np.any(good):
        return
    bins = np.asarray(edges, dtype=float)
    if overflow_policy not in {"exclude", "fold"}:
        raise ValueError(f"unsupported histogram overflow policy: {overflow_policy!r}")
    selected_values = vals[good]
    if overflow_policy == "fold":
        selected_values = np.where(
            selected_values > bins[-1],
            np.nextafter(bins[-1], bins[0]),
            selected_values,
        )
    h, _ = np.histogram(selected_values, bins=bins, weights=w[good])
    h2, _ = np.histogram(
        selected_values,
        bins=bins,
        weights=finite_weight_square(w[good]),
    )
    e, _ = np.histogram(selected_values, bins=bins)
    target["sumw"] = (np.asarray(target["sumw"], dtype=float) + h).tolist()
    target["sumw2"] = (np.asarray(target["sumw2"], dtype=float) + h2).tolist()
    target["entries"] = (np.asarray(target["entries"], dtype=int) + e).astype(int).tolist()


def lowdm_variable_values(chunk: dict[str, Any], spec: dict[str, Any], n: int) -> np.ndarray:
    branch = spec["branch"]
    fill = float(spec.get("fill", -99.0))
    if branch not in chunk:
        return np.full(n, fill, dtype=float)
    if spec.get("source") == "first":
        return finite_array(ak.fill_none(ak.firsts(chunk[branch]), fill), n, fill)
    return finite_array(chunk[branch], n, fill)


def highdm_base_region(region: str) -> str:
    return "SR" if region.startswith("SR_") else region


def highdm_distribution_masks(chunk: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for region in HIGHDM_CR_REGIONS + HIGHDM_VR_REGIONS:
        flag, _variable = REGION_VARIABLES[region]
        masks[region] = region_mask(chunk, region, flag, n)

    sr = bool_field(chunk, "feature_SR", n)
    nb = int_field(chunk, "nb_medium", n)
    nt = int_field(chunk, "nboosted_top", n)
    nw = int_field(chunk, "nboosted_w", n)
    rules = [
        (nb >= 1) & (nt == 0) & (nw == 0),
        (nb >= 1) & (nt == 0) & (nw >= 1),
        (nb == 1) & (nt >= 1) & (nw == 0),
        (nb == 1) & (nt >= 1) & (nw >= 1),
        (nb == 2) & (nt == 1) & (nw == 0),
        (nb == 2) & (nt == 1) & (nw == 1),
        (nb >= 3) & (nt == 1) & (nw == 0),
        (nb >= 3) & (nt == 1) & (nw == 1),
        (nb >= 3) & (nt == 2) & (nw == 0),
        (nb == 2) & (nt >= 2) & (nw == 0),
    ]
    for key, category_mask in zip(HIGHDM_SR_CATEGORY_KEYS, rules):
        masks[key] = sr & category_mask
    return masks


def highdm_variable_values(
    chunk: dict[str, Any], region: str, spec: dict[str, Any], n: int
) -> tuple[str, Any] | None:
    base_region = highdm_base_region(region)
    allowed_regions = spec.get("regions")
    if allowed_regions and base_region not in allowed_regions:
        return None
    branch = (spec.get("branch_by_region") or {}).get(base_region, spec["branch"])
    if branch not in chunk:
        return None
    source = spec.get("source", "scalar")
    if source == "count":
        return "scalar", np.asarray(ak.num(chunk[branch], axis=1), dtype=float)
    if source == "all":
        return "jagged", chunk[branch]
    if source == "masked":
        mask_branch = spec.get("mask_branch")
        if not mask_branch or mask_branch not in chunk:
            return None
        return "jagged", chunk[branch][ak.values_astype(chunk[mask_branch], np.bool_)]
    if source == "masked_first":
        mask_branch = spec.get("mask_branch")
        if not mask_branch or mask_branch not in chunk:
            return None
        selected = chunk[branch][ak.values_astype(chunk[mask_branch], np.bool_)]
        fill = float(spec.get("fill", -99.0))
        return "scalar", finite_array(ak.fill_none(ak.firsts(selected), fill), n, fill)
    return "scalar", finite_array(chunk[branch], n, float(spec.get("fill", -99.0)))


def add_jagged_binned_hist(
    target: dict[str, Any],
    values: Any,
    weights: np.ndarray,
    mask: np.ndarray,
    edges: list[float],
    *,
    overflow_policy: str = "exclude",
) -> None:
    event_mask = np.asarray(mask, dtype=bool)
    selected = values[event_mask]
    if len(selected) == 0:
        return
    selected_weights = ak.Array(np.asarray(weights, dtype=float)[event_mask])
    _, object_weights = ak.broadcast_arrays(selected, selected_weights)
    flat_values = np.asarray(ak.to_numpy(ak.flatten(selected, axis=1)), dtype=float)
    flat_weights = np.asarray(ak.to_numpy(ak.flatten(object_weights, axis=1)), dtype=float)
    fill_mask = np.ones(len(flat_values), dtype=bool)
    add_binned_hist(
        target,
        flat_values,
        flat_weights,
        fill_mask,
        edges,
        overflow_policy=overflow_policy,
    )


def fill_highdm_distribution_histograms(
    chunk: dict[str, Any],
    variations: dict[str, Any],
    normv: np.ndarray,
    label: str,
    process: str,
    is_data: bool,
    output: dict[str, Any],
    summary: dict[str, Any],
    only_regions: list[str] | None = None,
    only_variables: list[str] | None = None,
) -> None:
    n = len(normv)
    masks = highdm_distribution_masks(chunk, n)
    if only_regions:
        requested = set(only_regions)
        masks = {region: mask for region, mask in masks.items() if region in requested}
    for region, region_event_mask in masks.items():
        data_region = "SR" if region.startswith("SR_") else region
        if is_data and not data_process_allowed(process, data_region):
            note_data_exclusion(summary, region, process, int(np.count_nonzero(region_event_mask)))
            continue
        if not np.any(region_event_mask):
            continue
        for variable, spec in HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items():
            if only_variables and variable not in only_variables:
                continue
            prepared = highdm_variable_values(chunk, region, spec, n)
            if prepared is None:
                continue
            value_kind, values = prepared
            overflow_policy = str(spec.get("overflow_policy", "exclude"))
            if value_kind == "scalar":
                selected_values = np.asarray(values, dtype=float)[region_event_mask]
                finite_values = selected_values[np.isfinite(selected_values)]
                below = int(np.count_nonzero(finite_values < float(spec["bins"][0])))
                above = int(np.count_nonzero(finite_values > float(spec["bins"][-1])))
                excluded_above = above if overflow_policy == "exclude" else 0
                if below or excluded_above:
                    range_record = (
                        summary.setdefault("histogram_range_exclusions", {})
                        .setdefault(region, {})
                        .setdefault(variable, {})
                        .setdefault(label, {"below": 0, "above_or_equal": 0})
                    )
                    range_record["below"] = int(range_record.get("below", 0)) + below
                    range_record["above_or_equal"] = (
                        int(range_record.get("above_or_equal", 0)) + excluded_above
                    )
                if above and overflow_policy == "fold":
                    flow_record = (
                        summary.setdefault("histogram_folded_flow", {})
                        .setdefault(region, {})
                        .setdefault(variable, {})
                        .setdefault(label, {"above": 0})
                    )
                    flow_record["above"] = int(flow_record.get("above", 0)) + above
            for variation_name, raw_weights in variations.items():
                weights = finite_array(raw_weights, n, 0.0) * normv
                target = (
                    output.setdefault(region, {})
                    .setdefault(variable, {})
                    .setdefault(label, {})
                    .setdefault(variation_name, empty_binned_hist(spec["bins"]))
                )
                if value_kind == "jagged":
                    add_jagged_binned_hist(
                        target,
                        values,
                        weights,
                        region_event_mask,
                        spec["bins"],
                        overflow_policy=overflow_policy,
                    )
                else:
                    add_binned_hist(
                        target,
                        values,
                        weights,
                        region_event_mask,
                        spec["bins"],
                        overflow_policy=overflow_policy,
                    )


def add_index_hist(target: dict[str, Any], indices: np.ndarray, weights: np.ndarray) -> None:
    nbin = len(target["sumw"])
    idx = np.asarray(indices, dtype=int)
    w = np.asarray(weights, dtype=float)
    mask = (idx >= 0) & (idx < nbin) & finite_hist_weight_mask(w)
    if not np.any(mask):
        return
    h = np.bincount(idx[mask], weights=w[mask], minlength=nbin)[:nbin]
    h2 = np.bincount(idx[mask], weights=finite_weight_square(w[mask]), minlength=nbin)[:nbin]
    e = np.bincount(idx[mask], minlength=nbin)[:nbin]
    target["sumw"] = (np.asarray(target["sumw"], dtype=float) + h).tolist()
    target["sumw2"] = (np.asarray(target["sumw2"], dtype=float) + h2).tolist()
    target["entries"] = (np.asarray(target["entries"], dtype=int) + e).astype(int).tolist()


def boosted_an17_indices(chunk: dict[str, Any], n: int, sr_mask: np.ndarray) -> np.ndarray:
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    nt = np.asarray(chunk["nboosted_top"], dtype=int)
    nw = np.asarray(chunk["nboosted_w"], dtype=int)
    total = np.asarray(chunk["nboosted_total"], dtype=int)
    out = np.full(n, -1, dtype=int)
    labels = {name: idx for idx, name in enumerate(SEARCH_BIN_ORDER)}
    rules = [
        ("B0_Nb1", (nb == 1) & (total == 0)),
        ("B0_Nb2plus", (nb >= 2) & (total == 0)),
        ("Nb1_T0_W1plus", (nb == 1) & (nt == 0) & (nw >= 1)),
        ("Nb1_T1plus_W0", (nb == 1) & (nt >= 1) & (nw == 0)),
        ("Nb1_T1plus_W1plus", (nb == 1) & (nt >= 1) & (nw >= 1)),
        ("Nb2_T0_W1", (nb == 2) & (nt == 0) & (nw == 1)),
        ("Nb2_T0_W2", (nb == 2) & (nt == 0) & (nw == 2)),
        ("Nb2_T1_W0", (nb == 2) & (nt == 1) & (nw == 0)),
        ("Nb2_T1_W1", (nb == 2) & (nt == 1) & (nw == 1)),
        ("Nb2_T2_W0", (nb == 2) & (nt == 2) & (nw == 0)),
        ("Nb2_TW_ge3", (nb == 2) & (total >= 3)),
        ("Nb3plus_T0_W1", (nb >= 3) & (nt == 0) & (nw == 1)),
        ("Nb3plus_T0_W2", (nb >= 3) & (nt == 0) & (nw == 2)),
        ("Nb3plus_T1_W0", (nb >= 3) & (nt == 1) & (nw == 0)),
        ("Nb3plus_T1_W1", (nb >= 3) & (nt == 1) & (nw == 1)),
        ("Nb3plus_T2_W0", (nb >= 3) & (nt == 2) & (nw == 0)),
        ("Nb3plus_TW_ge3", (nb >= 3) & (total >= 3)),
    ]
    for label, mask in rules:
        fill = sr_mask & mask & (out < 0)
        out[fill] = labels[label]
    return out


def selected_an17_recoil_labels() -> list[str]:
    labels = []
    for bin_number in SELECTED_AN17_RECOIL_BINS_1BASED:
        category = SEARCH_BIN_ORDER[bin_number - 1]
        for recoil_label in RECOIL_BIN_LABELS:
            labels.append(f"AN17_{bin_number}_{category}_recoil_{recoil_label}")
    return labels


def selected_an17_recoil54_labels() -> list[str]:
    labels = [
        f"{category}_recoil_{recoil_label}"
        for category in NT0_RECOIL_CATEGORY_KEYS
        for recoil_label in RECOIL_BIN_LABELS
    ]
    labels.extend(selected_an17_recoil_labels())
    return labels


def selected_an17_recoil60_labels() -> list[str]:
    base = selected_an17_recoil54_labels()
    extra = [
        f"{EXTENDED_RECOIL60_CATEGORY_KEY}_recoil_{recoil_label}"
        for recoil_label in RECOIL_BIN_LABELS
    ]
    return (
        base[:EXTENDED_RECOIL60_INSERTION_BIN]
        + extra
        + base[EXTENDED_RECOIL60_INSERTION_BIN:]
    )


def selected_an17_recoil6_indices(chunk: dict[str, Any], n: int, sr_mask: np.ndarray) -> np.ndarray:
    search_indices = boosted_an17_indices(chunk, n, sr_mask)
    selected_zero_based = [idx - 1 for idx in SELECTED_AN17_RECOIL_BINS_1BASED]
    selected_map = {search_idx: pos for pos, search_idx in enumerate(selected_zero_based)}
    recoil = finite_array(chunk["met"], n, 0.0)
    recoil_idx = np.searchsorted(np.asarray(RECOIL_PT_BINS, dtype=float), recoil, side="right") - 1
    out = np.full(n, -1, dtype=int)
    for search_idx, category_pos in selected_map.items():
        mask = (search_indices == search_idx) & (recoil_idx >= 0) & (recoil_idx < len(RECOIL_PT_BINS) - 1)
        out[mask] = category_pos * (len(RECOIL_PT_BINS) - 1) + recoil_idx[mask]
    return out


def selected_an17_recoil54_indices(chunk: dict[str, Any], n: int, sr_mask: np.ndarray) -> np.ndarray:
    recoil = finite_array(chunk["met"], n, 0.0)
    recoil_idx = np.searchsorted(np.asarray(RECOIL_PT_BINS, dtype=float), recoil, side="right") - 1
    valid_recoil = (recoil_idx >= 0) & (recoil_idx < len(RECOIL_PT_BINS) - 1)
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    nt = np.asarray(chunk["nboosted_top"], dtype=int)
    nw = np.asarray(chunk["nboosted_w"], dtype=int)
    out = np.full(n, -1, dtype=int)

    nt0_categories = [
        (nb >= 1) & (nt == 0) & (nw == 0),
        (nb >= 1) & (nt == 0) & (nw >= 1),
    ]
    bins_per_category = len(RECOIL_PT_BINS) - 1
    for category_pos, category_mask in enumerate(nt0_categories):
        mask = sr_mask & valid_recoil & category_mask
        out[mask] = category_pos * bins_per_category + recoil_idx[mask]

    selected = selected_an17_recoil6_indices(chunk, n, sr_mask)
    selected_mask = selected >= 0
    out[selected_mask] = len(NT0_RECOIL_CATEGORY_KEYS) * bins_per_category + selected[selected_mask]
    return out


def selected_an17_recoil60_indices(chunk: dict[str, Any], n: int, sr_mask: np.ndarray) -> np.ndarray:
    out = selected_an17_recoil54_indices(chunk, n, sr_mask)
    shift = out >= EXTENDED_RECOIL60_INSERTION_BIN
    out[shift] += len(RECOIL_BIN_LABELS)
    recoil = finite_array(chunk["met"], n, 0.0)
    recoil_idx = np.searchsorted(np.asarray(RECOIL_PT_BINS, dtype=float), recoil, side="right") - 1
    valid_recoil = (recoil_idx >= 0) & (recoil_idx < len(RECOIL_PT_BINS) - 1)
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    nt = np.asarray(chunk["nboosted_top"], dtype=int)
    nw = np.asarray(chunk["nboosted_w"], dtype=int)
    extra = sr_mask & valid_recoil & (nb == 2) & (nt >= 2) & (nw == 0)
    out[extra] = EXTENDED_RECOIL60_INSERTION_BIN + recoil_idx[extra]
    return out


def compute_trota_lowdm_nres(
    event_tree: Any,
    trota_tree: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the exclusive TROTA Nres count aligned with the Events tree.

    Candidate fiducial cuts and both boosted/resolved overlap vetoes are the
    same as in the validated 2024 High-dM TROTA study.  Candidate processing is
    restricted to events that can enter one of the adopted low-dM regions.
    """
    required = set(TROTA_LOWDM_LIGHT_BRANCHES) | set(TROTA_LOWDM_OVERLAP_BRANCHES)
    missing = sorted(required - set(event_tree.keys()))
    if missing:
        raise RuntimeError(
            "TROTA low-dM veto requires missing Events branches: "
            + ", ".join(missing)
        )
    light = event_tree.arrays(sorted(required), library="ak")
    number_events = int(event_tree.num_entries)
    nb_ge1 = np.asarray(light["nb_medium_lowdm"], dtype=int) >= 1
    regular_regions = np.zeros(number_events, dtype=bool)
    for region in LOWDM_REGION_MAP:
        regular_regions |= np.asarray(light[f"feature_lowdm_{region}"], dtype=bool)
    focused_sr = (
        np.asarray(light["feature_lowdm_preselection"], dtype=bool)
        & np.asarray(light["pass_lowdm_topology_veto"], dtype=bool)
        & np.asarray(light["pass_lowdm_isr"], dtype=bool)
        & np.asarray(light["pass_lowdm_met_sqrt_ht"], dtype=bool)
        & nb_ge1
    )
    eligible = nb_ge1 & (regular_regions | focused_sr)
    counts = np.zeros(number_events, dtype=np.int16)

    tree_fields = set(trota_tree.keys())
    identity_fallback = 0
    if set(TROTA_PRIMARY_BRANCHES) <= tree_fields:
        arrays_ak = trota_tree.arrays(TROTA_PRIMARY_BRANCHES, library="ak")
        arrays = {
            name: np.asarray(ak.to_numpy(arrays_ak[name]))
            for name in TROTA_PRIMARY_BRANCHES
        }
        candidate_event = map_candidates_to_events(
            np.asarray(light["file_id"]), np.asarray(light["entry"]),
            arrays["file_id"], arrays["entry"],
        )
    elif set(TROTA_FALLBACK_BRANCHES) <= tree_fields:
        arrays_ak = trota_tree.arrays(TROTA_FALLBACK_BRANCHES, library="ak")
        arrays = {
            name: np.asarray(ak.to_numpy(arrays_ak[name]))
            for name in TROTA_FALLBACK_BRANCHES
        }
        candidate_event = map_candidates_to_events_rle(
            np.asarray(light["run"]), np.asarray(light["luminosityBlock"]),
            np.asarray(light["event"]), arrays["run"],
            arrays["luminosityBlock"], arrays["event"],
        )
        identity_fallback = 1
    else:
        missing_trota = sorted(set(TROTA_PRIMARY_BRANCHES) - tree_fields)
        raise RuntimeError("missing TROTA branches: " + ", ".join(missing_trota))

    run2_fiducial = (
        eligible[candidate_event]
        & np.isfinite(arrays["TopResolved1pct_eta"])
        & np.isfinite(arrays["TopResolved1pct_mass"])
        & np.isfinite(arrays["TopResolved1pct_QCDDiscriminant"])
        & (np.abs(arrays["TopResolved1pct_eta"]) < 2.0)
        & (arrays["TopResolved1pct_mass"] >= 100.0)
        & (arrays["TopResolved1pct_mass"] <= 250.0)
    )
    selected_rows = np.flatnonzero(run2_fiducial)
    rejected_boosted = 0
    rejected_resolved = 0
    if selected_rows.size:
        order = np.argsort(candidate_event[selected_rows], kind="stable")
        selected_rows = selected_rows[order]
        selected_events = candidate_event[selected_rows]
        boundaries = np.flatnonzero(np.diff(selected_events)) + 1
        for rows in np.split(selected_rows, boundaries):
            event_index = int(candidate_event[rows[0]])
            vetoed = boosted_overlap_vetoed_ak4_indices(
                jet_source_indices=ak.to_list(light["jet_source_index_all"][event_index]),
                jet_eta=ak.to_list(light["jet_eta_all"][event_index]),
                jet_phi=ak.to_list(light["jet_phi_all"][event_index]),
                fatjet_eta=ak.to_list(light["fatjet_eta_all"][event_index]),
                fatjet_phi=ak.to_list(light["fatjet_phi_all"][event_index]),
                fatjet_subjet_index1=ak.to_list(light["fatjet_subjet_index1_all"][event_index]),
                fatjet_subjet_index2=ak.to_list(light["fatjet_subjet_index2_all"][event_index]),
                fatjet_top_pass=ak.to_list(light["fatjet_boosted_top_pass_all"][event_index]),
                fatjet_w_pass=ak.to_list(light["fatjet_boosted_w_pass_all"][event_index]),
                subjet_eta=ak.to_list(light["subjet_eta_all"][event_index]),
                subjet_phi=ak.to_list(light["subjet_phi_all"][event_index]),
            )
            result = select_exclusive_resolved_candidates(
                candidate_indices=arrays["TopResolved1pct_candidateIndex"][rows],
                candidate_scores=arrays["TopResolved1pct_QCDDiscriminant"][rows],
                candidate_source_jets=np.stack(
                    [
                        arrays["TopResolved1pct_sourceJetIdx0"][rows],
                        arrays["TopResolved1pct_sourceJetIdx1"][rows],
                        arrays["TopResolved1pct_sourceJetIdx2"][rows],
                    ],
                    axis=1,
                ),
                boosted_vetoed_ak4_indices=vetoed,
            )
            counts[event_index] = result.nres
            rejected_boosted += len(result.rejected_by_boosted_overlap)
            rejected_resolved += len(result.rejected_by_resolved_overlap)

    return counts, {
        "events": number_events,
        "eligible_events": int(np.count_nonzero(eligible)),
        "nres_positive_events": int(np.count_nonzero(eligible & (counts > 0))),
        "trota_rows": int(trota_tree.num_entries),
        "run2_fiducial_rows": int(selected_rows.size),
        "rejected_by_boosted_overlap": int(rejected_boosted),
        "rejected_by_resolved_overlap": int(rejected_resolved),
        "identity_fallback_files": int(identity_fallback),
    }


def iterate_tree_with_dy_policy(
    tree: Any,
    branches: list[str],
    meta: dict[str, Any],
    policy: str,
    summary: dict[str, Any],
    step_size: int,
) -> Any:
    """Read only allowed dataset spans for a non-default DY sample policy.

    Balanced process ROOTs contain long contiguous source-dataset segments.
    A cheap first pass over ``dataset_id`` avoids loading all object branches
    for the PTLL-100/400/600 segments when constructing the PTLL-200-only
    candidate.  The default ``all`` path is byte-for-byte the previous
    iterator behavior.
    """
    if policy == "all":
        yield from tree.iterate(branches, step_size=step_size, library="ak")
        return

    allowed_ids = {
        int(dataset_id)
        for dataset_id in (meta.get("datasets") or {})
        if dy_ptll_dataset_allowed(
            *dataset_label(meta, int(dataset_id))[:2],
            policy,
        )
    }
    prefilter = summary.setdefault(
        "dy_ptll_prefilter",
        {"entries_scanned": 0, "entries_loaded": 0, "read_ranges": 0},
    )
    num_entries = int(tree.num_entries)
    for chunk_start in range(0, num_entries, max(1, int(step_size))):
        chunk_stop = min(num_entries, chunk_start + max(1, int(step_size)))
        dataset_ids = np.asarray(
            tree["dataset_id"].array(
                entry_start=chunk_start,
                entry_stop=chunk_stop,
                library="np",
            ),
            dtype=np.int64,
        )
        plan = dataset_id_prefilter_plan(dataset_ids, allowed_ids)
        prefilter["entries_scanned"] = int(prefilter.get("entries_scanned", 0)) + int(
            plan["entries_scanned"]
        )
        for dataset_id, count in plan["excluded_counts"].items():
            dataset, process, _is_data, _is_signal = dataset_label(meta, int(dataset_id))
            rejected = summary.setdefault("dy_ptll_dataset_exclusions", {}).setdefault(
                dataset,
                {"dataset_id": int(dataset_id), "entries": 0, "policy": policy},
            )
            rejected["entries"] = int(rejected.get("entries", 0)) + int(count)
            summary["events_processed"] = int(summary.get("events_processed", 0)) + int(count)
            if process != "DY":
                raise RuntimeError(
                    f"DY policy unexpectedly excluded non-DY dataset {dataset} ({process})"
                )
        for local_start, local_stop in plan["ranges"]:
            entry_start = chunk_start + local_start
            entry_stop = chunk_start + local_stop
            prefilter["entries_loaded"] = int(prefilter.get("entries_loaded", 0)) + (
                entry_stop - entry_start
            )
            prefilter["read_ranges"] = int(prefilter.get("read_ranges", 0)) + 1
            yield tree.arrays(
                branches,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="ak",
            )


def iterate_tree_for_gcr_study(
    tree: Any,
    branches: list[str],
    meta: dict[str, Any],
    photon_policy: str,
    summary: dict[str, Any],
    step_size: int,
) -> Any:
    light_branches = list(GCR_PREFILTER_BRANCHES)
    if photon_policy == "tight_eb":
        light_branches.extend(GCR_PHOTON_POLICY_BRANCHES)
    prefilter = summary.setdefault(
        "gcr_prefilter",
        {
            "entries_scanned": 0,
            "entries_selected_highdm": 0,
            "entries_selected_lowdm": 0,
            "entries_loaded": 0,
            "read_ranges": 0,
        },
    )
    num_entries = int(tree.num_entries)
    stride = max(1, int(step_size))
    for entry_start in range(0, num_entries, stride):
        entry_stop = min(num_entries, entry_start + stride)
        light = tree.arrays(
            light_branches,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        )
        masks = gcr_region_masks(light, photon_policy)
        selected = masks["selected_highdm"] | masks["selected_lowdm"]
        dataset_ids = np.asarray(light["dataset_id"], dtype=np.int64)
        record_gcr_selection_audit(summary, meta, dataset_ids, masks)
        prefilter["entries_scanned"] = (
            int(prefilter.get("entries_scanned", 0)) + len(dataset_ids)
        )
        prefilter["entries_selected_highdm"] = (
            int(prefilter.get("entries_selected_highdm", 0))
            + int(np.count_nonzero(masks["selected_highdm"]))
        )
        prefilter["entries_selected_lowdm"] = (
            int(prefilter.get("entries_selected_lowdm", 0))
            + int(np.count_nonzero(masks["selected_lowdm"]))
        )
        if not np.any(selected):
            continue
        full = tree.arrays(
            branches,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        )
        prefilter["entries_loaded"] = (
            int(prefilter.get("entries_loaded", 0)) + (entry_stop - entry_start)
        )
        prefilter["read_ranges"] = int(prefilter.get("read_ranges", 0)) + 1
        yield full[selected]


def process_root(repo: Path, root_path: Path, norm: dict[str, Any], histograms: dict[str, Any], search_histograms: dict[str, Any], lowdm_variable_histograms: dict[str, Any], highdm_variable_histograms: dict[str, Any], summary: dict[str, Any], step_size: int, campaign_year: str = "2024", only_regions: list[str] | None = None, require_btag: bool = False, require_weight_components: list[str] | None = None, analysis_sf_components: list[str] | None = None, require_branches: bool = False, require_normalization: bool = False, nominal_only: bool = False, distribution_only: bool = False, only_variables: list[str] | None = None, only_signal_mass: tuple[int, int] | None = None, only_lowdm_sr_nsv_inclusive: bool = False, only_lowdm_nsv_repair: bool = False, lowdm_only: bool = False, require_lowdm_nres_zero: bool = False, dy_ptll_policy: str = "all", gcr_only: bool = False, gcr_photon_policy: str = "nominal") -> None:
    try:
        meta = read_root_metadata(root_path, fallback=norm)
    except FileNotFoundError:
        summary.setdefault("missing_sidecars", []).append(str(root_path))
        return
    schema_version = str(meta.get("schema_version") or "unknown")
    schema_counts = summary.setdefault("input_sidecar_schema_versions", {})
    schema_counts[schema_version] = int(schema_counts.get(schema_version, 0)) + 1
    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        if tree.num_entries == 0:
            summary.setdefault("zero_entry_roots", []).append(str(root_path))
            return
        present = set(tree.keys())
        effective_read_branches = list(READ_BRANCHES)
        if gcr_only and gcr_photon_policy == "tight_eb":
            effective_read_branches = sorted(
                set(effective_read_branches + GCR_PHOTON_POLICY_BRANCHES)
            )
        missing_required_branches = [
            branch
            for branch in effective_read_branches
            if branch not in present and branch not in OPTIONAL_FORWARD_SCHEMA_BRANCHES
        ]
        if require_branches and missing_required_branches:
            raise RuntimeError(
                f"{root_path}: required flat branches are missing: "
                + ", ".join(missing_required_branches)
            )
        branches = [b for b in effective_read_branches if b in present]
        trota_nres = None
        trota_nres_cursor = 0
        if require_lowdm_nres_zero:
            expected_trota_schema = EXPECTED_TROTA_SCHEMA_BY_YEAR[campaign_year]
            if "TROTA" not in root_file:
                raise RuntimeError(f"{root_path}: required TROTA tree is missing")
            trota_meta = meta.get(f"trota_topresolved_{campaign_year}") or {}
            marker = trota_meta.get("marker") or {}
            if (
                trota_meta.get("status") != "complete"
                or trota_meta.get("schema_version") != expected_trota_schema
                or marker.get("status") != "complete"
                or marker.get("model_sha256") != EXPECTED_TROTA_MODEL_SHA256
            ):
                raise RuntimeError(f"{root_path}: invalid TROTA provenance: {trota_meta}")
            trota_nres, trota_stats = compute_trota_lowdm_nres(
                tree,
                root_file["TROTA"],
            )
            audit = summary.setdefault("trota_lowdm_nres_audit", {})
            for key, value in trota_stats.items():
                audit[key] = int(audit.get(key, 0)) + int(value)
        if gcr_only:
            chunk_iterator = iterate_tree_for_gcr_study(
                tree,
                branches,
                meta,
                gcr_photon_policy,
                summary,
                step_size,
            )
        else:
            chunk_iterator = iterate_tree_with_dy_policy(
                tree,
                branches,
                meta,
                dy_ptll_policy,
                summary,
                step_size,
            )
        for chunk in chunk_iterator:
            n = len(chunk["dataset_id"])
            if n == 0:
                continue
            if trota_nres is not None:
                stop = trota_nres_cursor + n
                if stop > len(trota_nres):
                    raise RuntimeError(f"{root_path}: TROTA Nres/event chunk alignment overflow")
                chunk = ak.with_field(
                    chunk,
                    ak.Array(trota_nres[trota_nres_cursor:stop]),
                    DERIVED_NRES_BRANCH,
                )
                trota_nres_cursor = stop
            dsids = np.asarray(chunk["dataset_id"], dtype=np.int64)
            for dsid in sorted(set(int(x) for x in dsids)):
                mask_ds = dsids == dsid
                sub = {name: chunk[name][mask_ds] for name in ak.fields(chunk)}
                dataset, process, is_data, is_signal = dataset_label(meta, dsid)
                dataset_entries = int(np.count_nonzero(mask_ds))
                if not dy_ptll_dataset_allowed(dataset, process, dy_ptll_policy):
                    rejected = summary.setdefault("dy_ptll_dataset_exclusions", {}).setdefault(
                        dataset,
                        {"dataset_id": int(dsid), "entries": 0, "policy": dy_ptll_policy},
                    )
                    rejected["entries"] = int(rejected.get("entries", 0)) + dataset_entries
                    summary["events_processed"] = int(summary.get("events_processed", 0)) + dataset_entries
                    continue
                subgroups: list[np.ndarray]
                if is_signal:
                    mstops = np.asarray(sub["mStop"], dtype=int)
                    mlsps = np.asarray(sub["mLSP"], dtype=int)
                    pairs = sorted(set(zip(mstops.tolist(), mlsps.tolist())))
                    if only_signal_mass is not None:
                        pairs = [pair for pair in pairs if pair == only_signal_mass]
                    subgroups = [(mstops == ms) & (mlsps == ml) for ms, ml in pairs]
                else:
                    subgroups = [np.ones(int(np.count_nonzero(mask_ds)), dtype=bool)]

                for mask_group in subgroups:
                    if not np.any(mask_group):
                        continue
                    sub_group = {name: arr[mask_group] for name, arr in sub.items()}
                    original_n = len(sub_group["dataset_id"])
                    if gcr_only:
                        policy_mask = gcr_photon_policy_mask(
                            sub_group,
                            gcr_photon_policy,
                            original_n,
                        )
                        highdm_gcr = (
                            as_bool(sub_group["feature_GCR"], original_n)
                            & policy_mask
                        )
                        lowdm_gcr = (
                            lowdm_region_mask(sub_group, "GCR", original_n)
                            & policy_mask
                        )
                        selected_gcr = highdm_gcr | lowdm_gcr
                        if not np.any(selected_gcr):
                            continue
                        sub_group["feature_GCR"] = ak.Array(highdm_gcr)
                        sub_group["feature_lowdm_GCR"] = ak.Array(lowdm_gcr)
                        sub_group = {
                            name: arr[selected_gcr]
                            for name, arr in sub_group.items()
                        }
                    focused_lowdm_indices = None
                    if only_lowdm_sr_nsv_inclusive:
                        focused_lowdm_indices = lowdm_nsv_inclusive_sr_indices(sub_group, original_n)
                        selected = focused_lowdm_indices >= 0
                        if only_lowdm_nsv_repair:
                            nb_values = int_field(sub_group, "nb_medium_lowdm", original_n, -1)
                            nsv_values = int_field(sub_group, "n_sv_softb", original_n, -1)
                            selected &= (
                                ((nb_values == 0) & (nsv_values < 0))
                                | ((nb_values == 1) & (nsv_values != 0))
                            )
                        if is_data and not data_process_allowed(process, "cat7_SR_lowDeltaM"):
                            note_data_exclusion(
                                summary,
                                "cat7_SR_lowDeltaM",
                                process,
                                int(np.count_nonzero(selected)),
                            )
                            summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                            continue
                        if not np.any(selected):
                            summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                            continue
                        sub_group = {name: arr[selected] for name, arr in sub_group.items()}
                        focused_lowdm_indices = focused_lowdm_indices[selected]
                    if only_regions:
                        selected = np.zeros(original_n, dtype=bool)
                        for region in only_regions:
                            flag, _var = REGION_VARIABLES[region]
                            selected |= region_mask(sub_group, region, flag, original_n)
                        if not np.any(selected):
                            summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                            continue
                        sub_group = {name: arr[selected] for name, arr in sub_group.items()}
                    arrays, inputs = flat_arrays_for_weights(sub_group)
                    if gcr_only:
                        inputs["gcr_mask"] = (
                            as_bool(sub_group["feature_GCR"], inputs["n"])
                            | lowdm_region_mask(sub_group, "GCR", inputs["n"])
                        )
                    eta_source = str(inputs.get("electron_eta_source") or "unknown")
                    eta_sources = summary.setdefault("electron_eta_sources", {})
                    eta_sources[eta_source] = int(eta_sources.get(eta_source, 0)) + int(
                        inputs["n"]
                    )
                    year_vals = np.asarray(sub_group["year"], dtype=int)
                    year = str(int(year_vals[0])) if len(year_vals) else "2024"
                    if year != campaign_year:
                        raise RuntimeError(
                            f"campaign year {campaign_year} does not match ROOT event year {year} "
                            f"in {root_path}"
                        )
                    correction_dataset = dataset
                    btag_efficiency_anchor = None
                    if is_signal:
                        mstop_values = np.asarray(sub_group["mStop"], dtype=int)
                        btag_efficiency_anchor, correction_dataset = signal_btag_efficiency_dataset(
                            int(mstop_values[0]) if len(mstop_values) else 1000,
                            dataset,
                        )
                    normv = norm_vector(
                        norm,
                        sub_group,
                        dsid,
                        dataset,
                        is_data,
                        is_signal,
                        require_normalization=require_normalization,
                    )
                    try:
                        _gen, variations, status = compute_weight_bundle(
                            arrays, repo, correction_dataset, process, year, inputs["n"],
                            inputs["jet_pt"], inputs["jet_eta"], inputs["jet_hadflav"], inputs["b_med"],
                            inputs["e_eta"], inputs["e_delta_eta_sc"], inputs["e_pt"], inputs["e_phi"], inputs["e_veto"], inputs["e_med"], inputs["n_e_veto"], inputs["n_e_med"],
                            inputs["m_eta"], inputs["m_pt"], inputs["m_phi"], inputs["m_loose"], inputs["m_med"], inputs["n_m_loose"], inputs["n_m_med"],
                            inputs["p_eta"], inputs["p_pt"], inputs["p_phi"], inputs["p_med"], inputs["gcr_mask"],
                            p_r9=inputs["p_r9"],
                            photon_id_wp=(
                                "Tight"
                                if gcr_only and gcr_photon_policy == "tight_eb"
                                else "Medium"
                            ),
                            met_pt=inputs["met_pt"],
                            met_trigger_mask=inputs["met_trigger_mask"],
                            analysis_sf_components=analysis_sf_components,
                        )
                        if is_signal:
                            btag_status = (status.get("components") or {}).get("btagSF") or {}
                            btag_status.update({
                                "efficiency_anchor_mstop": btag_efficiency_anchor,
                                "efficiency_dataset": correction_dataset,
                                "source_dataset": dataset,
                            })
                    except Exception as exc:
                        summary.setdefault("weight_failures", []).append({"root": str(root_path), "dataset_id": dsid, "dataset": dataset, "process": process, "label": sample_label(process, is_data, is_signal, sub_group, dataset), "error": f"{type(exc).__name__}: {exc}"[:500]})
                        variations = {"nominal": np.asarray(sub_group["gen_weight"], dtype=float)} if not is_data else {"nominal": np.ones(inputs["n"], dtype=float)}
                        status = {"applied": False, "error": "fallback_raw_gen_weight"}
                    btag_status = (status.get("components") or {}).get("btagSF") or {}
                    if require_btag and not is_data and not btag_status.get("applied"):
                        raise RuntimeError(f"Required btagSF is unavailable for {dataset}: {btag_status}")
                    if not is_data:
                        for component_name in require_weight_components or []:
                            component_status = (
                                (status.get("components") or {}).get(component_name) or {}
                            )
                            if not component_status.get("applied"):
                                raise RuntimeError(
                                    f"Required weight component {component_name} is unavailable "
                                    f"for {dataset}: {component_status}"
                                )
                        required_analysis_variations = {
                            variation
                            for variation in REQUIRED_ANALYSIS_SF_VARIATIONS
                            if any(
                                variation == f"{component}{direction}"
                                for component in (require_weight_components or [])
                                for direction in ("Up", "Down")
                            )
                        }
                        missing_variations = sorted(
                            required_analysis_variations
                            - set(status.get("available_variations") or [])
                        )
                        if missing_variations:
                            raise RuntimeError(
                                "Required analysis SF weight variations are unavailable "
                                f"for {dataset}: {missing_variations}"
                            )
                    variations = histogram_variations(variations, nominal_only)
                    label = sample_label(process, is_data, is_signal, sub_group, dataset)
                    record_scale_factor_audit(
                        summary,
                        label,
                        dataset,
                        status,
                        inputs["n"],
                    )
                    for variation_name, raw_weight in variations.items():
                        raw_array = np.asarray(raw_weight, dtype=float) * normv
                        nonfinite = int(np.count_nonzero(~np.isfinite(raw_array)))
                        excessive = int(
                            np.count_nonzero(
                                np.isfinite(raw_array)
                                & (np.abs(raw_array) > MAX_ABS_HIST_WEIGHT)
                            )
                        )
                        if nonfinite or excessive:
                            rejection = (
                                summary.setdefault("weight_rejections", {})
                                .setdefault(label, {})
                                .setdefault(
                                    variation_name,
                                    {"nonfinite": 0, "excessive_magnitude": 0},
                                )
                            )
                            rejection["nonfinite"] = (
                                int(rejection.get("nonfinite", 0)) + nonfinite
                            )
                            rejection["excessive_magnitude"] = (
                                int(rejection.get("excessive_magnitude", 0)) + excessive
                            )
                    summary.setdefault("scale_factor_status", {}).setdefault(label, status)
                    if only_lowdm_sr_nsv_inclusive:
                        for vname, wraw in variations.items():
                            weights = finite_array(wraw, inputs["n"], 0.0) * normv
                            target = (
                                search_histograms
                                .setdefault("cat7_SR_lowDeltaM", {})
                                .setdefault(label, {})
                                .setdefault(vname, empty_index_hist(len(LOWDM_34BIN_LABELS)))
                            )
                            add_index_hist(target, focused_lowdm_indices, weights)
                        summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                        continue
                    if gcr_only:
                        active_regions = {"GCR": REGION_VARIABLES["GCR"]}
                    elif lowdm_only:
                        active_regions = {}
                    else:
                        active_regions = {region: REGION_VARIABLES[region] for region in only_regions} if only_regions else REGION_VARIABLES
                    for region, (flag, var) in active_regions.items():
                        rmask = region_mask(sub_group, region, flag, inputs["n"])
                        if is_data and not data_process_allowed(process, region):
                            note_data_exclusion(summary, region, process, int(np.count_nonzero(rmask)))
                            continue
                        values = finite_array(sub_group[var], inputs["n"], 0.0)
                        for vname, wraw in variations.items():
                            weights = finite_array(wraw, inputs["n"], 0.0) * normv
                            target = histograms.setdefault(region, {}).setdefault(label, {}).setdefault(vname, empty_hist())
                            add_hist(target, values, weights, rmask)
                    if not lowdm_only:
                        fill_highdm_distribution_histograms(
                            sub_group, variations, normv, label, process, is_data,
                            highdm_variable_histograms, summary,
                            ["GCR"] if gcr_only else only_regions,
                            only_variables,
                        )
                    if distribution_only:
                        summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                        continue
                    if only_regions:
                        summary["events_processed"] = int(summary.get("events_processed", 0)) + original_n
                        continue
                    if not gcr_only and not lowdm_only:
                        sr_mask = as_bool(sub_group["feature_SR"], inputs["n"])
                        selected_recoil54_indices = selected_an17_recoil54_indices(sub_group, inputs["n"], sr_mask)
                        selected_recoil60_indices = selected_an17_recoil60_indices(sub_group, inputs["n"], sr_mask)
                        if is_data and not data_process_allowed(process, "SR"):
                            note_data_exclusion(summary, SELECTED_RECOIL54_SCHEME, process, int(np.count_nonzero(selected_recoil54_indices >= 0)))
                            note_data_exclusion(summary, EXTENDED_RECOIL60_SCHEME, process, int(np.count_nonzero(selected_recoil60_indices >= 0)))
                        else:
                            for vname, wraw in variations.items():
                                weights = finite_array(wraw, inputs["n"], 0.0) * normv
                                target = search_histograms.setdefault(SELECTED_RECOIL54_SCHEME, {}).setdefault(label, {}).setdefault(vname, empty_index_hist(len(selected_an17_recoil54_labels())))
                                add_index_hist(target, selected_recoil54_indices, weights)
                                target60 = search_histograms.setdefault(EXTENDED_RECOIL60_SCHEME, {}).setdefault(label, {}).setdefault(vname, empty_index_hist(len(selected_an17_recoil60_labels())))
                                add_index_hist(target60, selected_recoil60_indices, weights)

                    lowdm_regions = (
                        {"GCR": LOWDM_REGION_MAP["GCR"]}
                        if gcr_only
                        else LOWDM_REGION_MAP
                    )
                    for lowdm_region, lowdm_channel in lowdm_regions.items():
                        lowdm_mask = lowdm_region_mask(sub_group, lowdm_region, inputs["n"])
                        if not np.any(lowdm_mask):
                            continue
                        lowdm_indices = int_field(
                            sub_group,
                            f"lowdm_search_bin_{lowdm_region}",
                            inputs["n"],
                            -1,
                        )
                        lowdm_indices = lowdm_nbge1_indices(
                            np.where(lowdm_mask, lowdm_indices, -1)
                        )
                        if is_data and not data_process_allowed(process, lowdm_channel):
                            note_data_exclusion(summary, lowdm_channel, process, int(np.count_nonzero(lowdm_mask)))
                            continue
                        lowdm_assigned_mask = (
                            (lowdm_indices >= 0)
                            & (lowdm_indices < len(LOWDM_34BIN_LABELS))
                        )
                        selected_lowdm = int(np.count_nonzero(lowdm_mask))
                        assigned_lowdm = int(np.count_nonzero(lowdm_assigned_mask))
                        lowdm_entry_record = (
                            summary
                            .setdefault("lowdm_search_bin_entry_accounting", {})
                            .setdefault(lowdm_channel, {})
                            .setdefault(
                                label,
                                {
                                    "selected_entries": 0,
                                    "assigned_entries": 0,
                                    "unassigned_entries": 0,
                                },
                            )
                        )
                        lowdm_entry_record["selected_entries"] = (
                            int(lowdm_entry_record.get("selected_entries", 0))
                            + selected_lowdm
                        )
                        lowdm_entry_record["assigned_entries"] = (
                            int(lowdm_entry_record.get("assigned_entries", 0))
                            + assigned_lowdm
                        )
                        lowdm_entry_record["unassigned_entries"] = (
                            int(lowdm_entry_record.get("unassigned_entries", 0))
                            + selected_lowdm
                            - assigned_lowdm
                        )
                        lowdm_values = {
                            var_name: lowdm_variable_values(sub_group, LOWDM_VARIABLE_SPECS[var_name], inputs["n"])
                            for var_name in LOWDM_REGION_VARIABLES.get(lowdm_region, [])
                        }
                        for vname, wraw in variations.items():
                            weights = finite_array(wraw, inputs["n"], 0.0) * normv
                            target = search_histograms.setdefault(lowdm_channel, {}).setdefault(label, {}).setdefault(vname, empty_index_hist(len(LOWDM_34BIN_LABELS)))
                            add_index_hist(target, lowdm_indices, weights)
                            for var_name, values in lowdm_values.items():
                                spec = LOWDM_VARIABLE_SPECS[var_name]
                                vtarget = lowdm_variable_histograms.setdefault(lowdm_channel, {}).setdefault(var_name, {}).setdefault(label, {}).setdefault(vname, empty_binned_hist(spec["bins"]))
                                add_binned_hist(vtarget, values, weights, lowdm_mask, spec["bins"])
                    summary["events_processed"] = int(summary.get("events_processed", 0)) + inputs["n"]
        if trota_nres is not None and trota_nres_cursor != len(trota_nres):
            raise RuntimeError(
                f"{root_path}: TROTA Nres/event chunk alignment incomplete: "
                f"{trota_nres_cursor} != {len(trota_nres)}"
            )


def expand_roots(inputs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            roots.extend(sorted(p.glob("*.root")))
        else:
            roots.extend(sorted(Path(x) for x in p.parent.glob(p.name)) if any(ch in item for ch in "*?[") else [p])
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build boosted Nt>=1 recoil-pT histograms from flat ntuples with post-skim SFs.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--campaign-year", choices=sorted(BTAG_EFFICIENCY_RELATIVE_PATHS), default="2024")
    parser.add_argument("--step-size", type=int, default=200000)
    parser.add_argument("--only-regions", nargs="+", choices=sorted(REGION_VARIABLES))
    parser.add_argument("--require-btag", action="store_true")
    parser.add_argument(
        "--expected-btag-efficiency-sha256",
        default=EXPECTED_BTAG_EFFICIENCY_SHA256_2024,
        help="Expected SHA256 for analysis/hists/btageff2024.merged.",
    )
    parser.add_argument(
        "--require-weight-components",
        nargs="+",
        default=list(REQUIRED_ANALYSIS_SF_COMPONENTS),
        help=(
            "Fail a non-data dataset immediately unless every named component "
            "is recorded as applied by real_subset_worker.compute_weight_bundle. "
            "Production defaults require all adopted analysis-owned SF payloads."
        ),
    )
    parser.add_argument(
        "--analysis-sf-components",
        nargs="*",
        choices=list(REQUIRED_ANALYSIS_SF_COMPONENTS),
        default=None,
        help=(
            "Analysis-owned SF components included in nominal and Up/Down weights. "
            "Omit the option for the campaign defaults; pass the option with no "
            "values only when the campaign has no analysis-owned payloads."
        ),
    )
    parser.add_argument(
        "--require-branches",
        action="store_true",
        help="Fail if any branch in the current flat histogram schema is absent.",
    )
    parser.add_argument(
        "--require-normalization",
        action="store_true",
        help="Fail on a missing, non-finite, or non-positive MC normalization factor.",
    )
    parser.add_argument(
        "--allow-zero-entry-roots",
        action="store_true",
        help=(
            "Treat readable flat ROOT files with an empty Events tree as valid "
            "zero-contribution inputs."
        ),
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help=(
            "Compute and validate the full weight bundle, but fill histogram "
            "payloads with the nominal variation only."
        ),
    )
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--only-signal-mass", nargs=2, type=int, metavar=("MSTOP", "MLSP"))
    parser.add_argument(
        "--only-lowdm-sr-nsv-inclusive",
        action="store_true",
        help="Rebuild only the adopted Nsv-independent, Nb>=1, 34-bin Low-dM SR from broad nominal intermediates.",
    )
    parser.add_argument(
        "--only-lowdm-nsv-repair",
        action="store_true",
        help="With --only-lowdm-sr-nsv-inclusive, retain only events absent or non-projectable from the old Nsv-split bins.",
    )
    parser.add_argument("--only-variables", nargs="+", choices=sorted(HIGHDM_DISTRIBUTION_VARIABLE_SPECS))
    parser.add_argument(
        "--gcr-only",
        action="store_true",
        help="Build only high- and low-dM GCR histograms using a GCR-specific sparse prefilter.",
    )
    parser.add_argument(
        "--lowdm-only",
        action="store_true",
        help="Build only the six adopted low-dM search-bin and distribution containers.",
    )
    parser.add_argument(
        "--require-lowdm-nres-zero",
        action="store_true",
        help=(
            "Require Nres=0 in every low-dM SR/CR using the validated 2024 TROTA "
            "candidate, fiducial, and overlap-removal definition."
        ),
    )
    parser.add_argument(
        "--gcr-photon-policy",
        choices=("nominal", "tight_eb"),
        default="nominal",
        help="Photon selection applied as a strict subset of the trusted nominal GCR.",
    )
    parser.add_argument(
        "--dy-ptll-policy",
        choices=("all", "ptll100_only", "ptll200_only", "ptll100_200"),
        default="all",
        help=(
            "DY sample-family policy. 'ptll100_200' keeps only the PTLL-100 "
            "and PTLL-200 production bins and excludes PTLL-400/600."
        ),
    )
    args = parser.parse_args()
    if args.only_lowdm_nsv_repair and not args.only_lowdm_sr_nsv_inclusive:
        parser.error("--only-lowdm-nsv-repair requires --only-lowdm-sr-nsv-inclusive")
    if args.gcr_only and args.only_regions:
        parser.error("--gcr-only cannot be combined with --only-regions")
    if args.lowdm_only and (args.only_regions or args.gcr_only):
        parser.error("--lowdm-only cannot be combined with --only-regions or --gcr-only")
    if args.require_lowdm_nres_zero and args.dy_ptll_policy != "all":
        parser.error("--require-lowdm-nres-zero requires --dy-ptll-policy all")
    if args.gcr_photon_policy != "nominal" and not args.gcr_only:
        parser.error("--gcr-photon-policy requires --gcr-only")
    analysis_sf_components = (
        list(REQUIRED_ANALYSIS_SF_COMPONENTS)
        if args.analysis_sf_components is None
        else list(args.analysis_sf_components)
    )
    unavailable_required = (
        set(args.require_weight_components)
        & set(REQUIRED_ANALYSIS_SF_COMPONENTS)
    ) - set(analysis_sf_components)
    if unavailable_required:
        parser.error(
            "--require-weight-components contains disabled analysis SFs: "
            + ", ".join(sorted(unavailable_required))
        )
    repo = Path(args.repo).resolve()
    norm = read_json(Path(args.normalization))
    histograms: dict[str, Any] = {}
    search_histograms: dict[str, Any] = {}
    lowdm_variable_histograms: dict[str, Any] = {}
    highdm_variable_histograms: dict[str, Any] = {}
    btag_payload_required = bool(
        args.require_btag or "btagSF" in args.require_weight_components
    )
    build_options = {
        "step_size": int(args.step_size),
        "only_regions": list(args.only_regions) if args.only_regions else None,
        "only_variables": list(args.only_variables) if args.only_variables else None,
        "campaign_year": str(args.campaign_year),
        "require_btag": bool(args.require_btag),
        "require_weight_components": list(args.require_weight_components),
        "analysis_sf_components": analysis_sf_components,
        "require_branches": bool(args.require_branches),
        "require_normalization": bool(args.require_normalization),
        "allow_zero_entry_roots": bool(args.allow_zero_entry_roots),
        "nominal_only": bool(args.nominal_only),
        "distribution_only": bool(args.distribution_only),
        "only_signal_mass": list(args.only_signal_mass) if args.only_signal_mass else None,
        "only_lowdm_sr_nsv_inclusive": bool(args.only_lowdm_sr_nsv_inclusive),
        "only_lowdm_nsv_repair": bool(args.only_lowdm_nsv_repair),
        "lowdm_only": bool(args.lowdm_only),
        "require_lowdm_nres_zero": bool(args.require_lowdm_nres_zero),
        "dy_ptll_policy": str(args.dy_ptll_policy),
        "gcr_only": bool(args.gcr_only),
        "gcr_photon_policy": str(args.gcr_photon_policy),
        "local_analysis_data": os.environ.get(
            "AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA",
            "0",
        ),
        "normalization_sha256": file_sha256(Path(args.normalization)),
        "code_sha256": execution_code_sha256(repo, args.campaign_year),
        "btag_efficiency": btag_efficiency_contract(
            repo,
            str(args.expected_btag_efficiency_sha256),
            btag_payload_required,
            args.campaign_year,
        ),
    }
    summary: dict[str, Any] = {
        "events_processed": 0,
        "input_roots": [],
        "region_filter": args.only_regions,
        "variable_filter": args.only_variables,
        "dy_ptll_policy": args.dy_ptll_policy,
        "gcr_photon_policy": args.gcr_photon_policy,
        "build_options": build_options,
    }
    for root_path in expand_roots(args.inputs):
        if root_path.name.startswith("validation"):
            continue
        if not root_path.exists():
            summary.setdefault("missing_input_roots", []).append(str(root_path))
            continue
        summary["input_roots"].append(str(root_path))
        process_root(
            repo,
            root_path,
            norm,
            histograms,
            search_histograms,
            lowdm_variable_histograms,
            highdm_variable_histograms,
            summary,
            step_size=args.step_size,
            campaign_year=args.campaign_year,
            only_regions=args.only_regions,
            require_btag=args.require_btag,
            require_weight_components=args.require_weight_components,
            analysis_sf_components=analysis_sf_components,
            require_branches=args.require_branches,
            require_normalization=args.require_normalization,
            nominal_only=args.nominal_only,
            distribution_only=args.distribution_only,
            only_variables=args.only_variables,
            only_signal_mass=(
                tuple(args.only_signal_mass) if args.only_signal_mass else None
            ),
            only_lowdm_sr_nsv_inclusive=args.only_lowdm_sr_nsv_inclusive,
            only_lowdm_nsv_repair=args.only_lowdm_nsv_repair,
            lowdm_only=args.lowdm_only,
            require_lowdm_nres_zero=args.require_lowdm_nres_zero,
            dy_ptll_policy=args.dy_ptll_policy,
            gcr_only=args.gcr_only,
            gcr_photon_policy=args.gcr_photon_policy,
        )
    if summary.get("weight_failures"):
        payload_status = "complete_with_weight_fallbacks"
    elif any(
        summary.get(key)
        for key in (
            "missing_input_roots",
            "missing_sidecars",
            "weight_rejections",
        )
    ) or (
        bool(summary.get("zero_entry_roots"))
        and not args.allow_zero_entry_roots
    ):
        payload_status = "complete_with_warnings"
    else:
        payload_status = "complete"
    payload = {
        "schema_version": "flat_boosted_recoil_hists_v1",
        "status": payload_status,
        "recoil_pt_bins": RECOIL_PT_BINS,
        "regions": (
            {"GCR": REGION_VARIABLES["GCR"]}
            if args.gcr_only
            else {region: REGION_VARIABLES[region] for region in args.only_regions}
            if args.only_regions
            else REGION_VARIABLES
        ),
        "gcr_photon_policy": {
            "name": args.gcr_photon_policy,
            "baseline": "trusted feature_GCR/feature_lowdm_GCR from real_subset_worker.py",
            "candidate": (
                "strict nominal-GCR subset with exactly one photon satisfying "
                "corrected pT>220 GeV, |eta|<1.4442, cutBased>=3, and electronVeto"
                if args.gcr_photon_policy == "tight_eb"
                else "nominal medium-photon selection and fiducial EB-or-EE acceptance"
            ),
            "photon_id_scale_factor_working_point": (
                "Tight" if args.gcr_photon_policy == "tight_eb" else "Medium"
            ),
        },
        "ntop_split_policy": {
            "status": "included",
            "axis": "nboosted_top",
            "split_regions": {region: {"Nt1": f"{region}_Nt1", "Nt0": f"{region}_Nt0"} for region in NTOP_SPLIT_BASE_REGIONS},
            "note": "SR_Nt1 is the feature-side SR nTop>=1 branch. Other *_Nt1 regions are built as base region AND nboosted_top>=1; *_Nt0 regions are base region AND nboosted_top==0.",
        },
        "search_bin_schemes": {
            SELECTED_RECOIL54_SCHEME: {
                "bin_labels": selected_an17_recoil54_labels(),
                "selection": "feature_SR, categories Nb>=1,Nt=0,NW=0 and Nb>=1,Nt=0,NW>=1, followed by selected AN17 bins 4,5,8,9,14,15,16, all split into six recoil/MET bins",
                "selected_an17_bins_1based": SELECTED_AN17_RECOIL_BINS_1BASED,
                "recoil_pt_bins": RECOIL_PT_BINS,
            },
            EXTENDED_RECOIL60_SCHEME: {
                "bin_labels": selected_an17_recoil60_labels(),
                "selection": "the adopted feature_SR categorization with Nb=2,Nt>=2,NW=0 inserted as bins 37--42, all classes split into six recoil/MET bins",
                "base_scheme": SELECTED_RECOIL54_SCHEME,
                "extra_category": EXTENDED_RECOIL60_CATEGORY_KEY,
                "selected_an17_bins_1based": SELECTED_AN17_RECOIL_BINS_1BASED,
                "recoil_pt_bins": RECOIL_PT_BINS,
            },
            **{
                channel: {
                    "bin_labels": LOWDM_34BIN_LABELS,
                    "selection": (
                        "feature_lowdm_preselection && pass_lowdm_topology_veto && "
                        + (
                            "Nres(TROTA)=0 && "
                            if args.require_lowdm_nres_zero
                            else ""
                        )
                        + "pass_lowdm_isr && pass_lowdm_met_sqrt_ht && Nb>=1; "
                        "ISR-subjet b veto and mTb requirement removed"
                        if args.only_lowdm_sr_nsv_inclusive and region == "SR"
                        else (
                            f"feature_lowdm_{region} && Nb>=1"
                            + (
                                " && Nres(TROTA)=0"
                                if args.require_lowdm_nres_zero
                                else ""
                            )
                        )
                    ),
                    "delta_m": "low",
                    "region": region,
                    "nsv_policy": "inclusive; Nsv is not used in category assignment",
                    "isr_subjet_bveto_policy": "not applied",
                    "mtb_policy": "not applied",
                    "category_sizes": LOWDM_NBGE1_CATEGORY_SIZES,
                    "removed_categories": LOWDM_REMOVED_NB0_CATEGORY_SIZES,
                }
                for region, channel in LOWDM_REGION_MAP.items()
            },
        },
        "lowdm_region_policy": {
            "status": (
                f"physics_proposal_trota_nres0_{args.campaign_year}_2026-08-20"
                if args.require_lowdm_nres_zero
                else "adopted_from_user_2026-07-24"
            ),
            "search_bins": (
                "34 low-dM bins per region with explicit Nb>=1 and TROTA Nres=0 "
                "after removing the two leading Nb=0 categories; Nsv, the ISR-subjet "
                "b veto, and the mTb requirement are not applied"
                if args.require_lowdm_nres_zero
                else "34 low-dM bins per region with explicit Nb>=1 after removing the two leading Nb=0 categories; Nsv, the ISR-subjet b veto, and the mTb requirement are not applied"
            ),
            "regions": LOWDM_REGION_MAP,
            "note": (
                "Every Low-dM CR and SR explicitly requires Nb>=1 and TROTA Nres=0. "
                f"This is a {args.campaign_year} physics proposal pending comparison with the previous "
                "boosted-only topology veto. Low-dM is Nsv-inclusive and does not require "
                "the ISR-subjet b veto or mTb<175."
                if args.require_lowdm_nres_zero
                else "Every Low-dM CR and SR explicitly requires Nb>=1. Low-dM is Nsv-inclusive and does not require the ISR-subjet b veto or mTb<175. GCR and DY use photon/dilepton recoil directions and object-cleaned AK4/AK8 collections; DY also requires OS, on-Z, and pT(ll)>200."
            ),
            "resolved_top_veto": {
                "applied": bool(args.require_lowdm_nres_zero),
                "branch": DERIVED_NRES_BRANCH if args.require_lowdm_nres_zero else None,
                "requirement": "Nres == 0" if args.require_lowdm_nres_zero else None,
                "trota_schema": EXPECTED_TROTA_SCHEMA_BY_YEAR[args.campaign_year] if args.require_lowdm_nres_zero else None,
                "trota_model_sha256": EXPECTED_TROTA_MODEL_SHA256 if args.require_lowdm_nres_zero else None,
                "candidate_fiducial": "abs(eta)<2, 100<=mass<=250 GeV",
                "overlap_policy": "exclusive resolved candidates after boosted-object AK4 veto",
            },
        },
        "highdm_distribution_variable_specs": HIGHDM_DISTRIBUTION_VARIABLE_SPECS,
        "highdm_distribution_regions": {
            "control": HIGHDM_CR_REGIONS,
            "validation": HIGHDM_VR_REGIONS,
            "signal_categories": HIGHDM_SR_CATEGORY_KEYS,
        },
        "lowdm_variable_specs": LOWDM_VARIABLE_SPECS,
        "lowdm_region_variables": LOWDM_REGION_VARIABLES,
        "normalization": str(args.normalization),
        "data_region_process_policy": DATA_PROCESS_BY_REGION,
        "summary": summary,
        "histograms": histograms,
        "search_bin_histograms": search_histograms,
        "lowdm_variable_histograms": lowdm_variable_histograms,
        "highdm_variable_histograms": highdm_variable_histograms,
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"status": payload["status"], "input_roots": len(summary["input_roots"]), "events_processed": summary["events_processed"], "regions": len(histograms), "search_bin_schemes": len(search_histograms), "lowdm_variable_regions": len(lowdm_variable_histograms), "highdm_variable_regions": len(highdm_variable_histograms), "output": args.output}, sort_keys=True))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
