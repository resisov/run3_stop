from __future__ import annotations
import math
import json
import os
import contextlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import awkward as ak
from .nanoaod_ids import isTrackElectron, isTrackMuon, isTrackPion, isVetoElectron, isMediumElectron, isLooseMuon, isMediumMuon, isMediumTau, isGoodJet, isGoodFatJet, isJetVeto


def delta_phi(phi1: Any, phi2: Any) -> Any:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def clean_by_delta_r(
    obj_eta: ak.Array,
    obj_phi: ak.Array,
    ref_eta: ak.Array,
    ref_phi: ak.Array,
    dr_min: float,
) -> ak.Array:
    deta = obj_eta[:, :, None] - ref_eta[:, None, :]
    dphi = delta_phi(obj_phi[:, :, None], ref_phi[:, None, :])
    return ak.all((deta * deta + dphi * dphi) > dr_min * dr_min, axis=2)


LEPTON_VETO_PT_MIN = 10.0


def object_masks(arrays: ak.Array) -> dict[str, ak.Array]:
    e_pt = arrays["electron_pt_all"]
    e_eta = arrays["electron_eta_all"]
    e_cb = arrays["electron_cutbased_all"]
    e_iso = arrays["electron_mini_iso_all"]
    e_fid = (abs(e_eta) < 1.4442) | ((abs(e_eta) > 1.5660) & (abs(e_eta) < 2.5))
    e_veto = (e_pt > LEPTON_VETO_PT_MIN) & e_fid & (e_cb >= 1) & (e_iso < 0.1)
    e_medium = (e_pt > 10.0) & e_fid & (e_cb >= 3) & (e_iso < 0.1)

    m_pt = arrays["muon_pt_all"]
    m_eta = arrays["muon_eta_all"]
    m_iso = arrays["muon_mini_iso_all"]
    m_loose = (
        (m_pt > LEPTON_VETO_PT_MIN)
        & (abs(m_eta) < 2.4)
        & as_ak_bool(arrays["muon_loose_id_all"])
        & (m_iso < 0.2)
    )
    m_medium = (
        (m_pt > 10.0)
        & (abs(m_eta) < 2.4)
        & as_ak_bool(arrays["muon_medium_id_all"])
        & (m_iso < 0.2)
    )

    p_pt = arrays["photon_pt_all"]
    p_eta = arrays["photon_eta_all"]
    p_fid = (abs(p_eta) < 1.4442) | ((abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5))
    p_medium = (
        (p_pt > 220.0)
        & p_fid
        & (arrays["photon_cutbased_all"] >= 2)
        & as_ak_bool(arrays["photon_electron_veto_all"])
    )
    return {
        "electron_veto": e_veto,
        "electron_medium": e_medium,
        "muon_loose": m_loose,
        "muon_medium": m_medium,
        "photon_medium": p_medium,
    }


def as_ak_bool(values: ak.Array) -> ak.Array:
    return ak.values_astype(values, np.bool_)

