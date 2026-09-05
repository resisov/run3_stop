from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import uproot


EVENT_BRANCHES = (
    "physical_dataset_id",
    "run",
    "luminosityBlock",
    "event",
    "file_id",
    "is_signal",
    "is_background",
    "mStop",
    "mLSP",
    "signal_topology_id",
    "process_id",
    "dataset_id",
    "gen_weight",
    "feature_lowdm_preselection",
    "feature_SR",
    "met",
    "met_phi",
    "ht",
    "njet",
    "nb_medium",
    "jet_corrected_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_corrected_mass",
    "jet_btag_upart_all",
    "jet_id_all",
)

TOP_TARGETED_BRANCHES = (
    "j1_met_dphi",
    "j2_met_dphi",
    "j3_met_dphi",
    "j4_met_dphi",
    "min_dphi4",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
    "lowdm_met_sqrt_ht",
    "lowdm_ptb",
    "lowdm_mtb",
    "lowdm_isr_subjet_btag_max",
    "lowdm_fatjet_pt",
    "lowdm_fatjet_msd",
)

ENGINEERED_V2_EXTRA_BRANCHES = (
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
)

ENGINEERED_EXPANDED_EXTRA_BRANCHES = (
    *ENGINEERED_V2_EXTRA_BRANCHES,
    "n_lowdm_isr",
)

DIAGONAL_V3_EXTRA_BRANCHES = ENGINEERED_EXPANDED_EXTRA_BRANCHES

BASE_GLOBAL_FEATURE_NAMES = (
    "log1p_met",
    "sin_met_phi",
    "cos_met_phi",
    "log1p_ht",
    "njet",
    "nb_medium",
)

TOP_TARGETED_GLOBAL_FEATURE_NAMES = (
    "log1p_lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_dphi",
    "log1p_lowdm_met_sqrt_ht",
    "log1p_lowdm_ptb",
    "log1p_lowdm_mtb",
    "j1_met_dphi",
    "j2_met_dphi",
    "j3_met_dphi",
    "j4_met_dphi",
    "min_dphi4",
    "lowdm_isr_subjet_btag_max",
    "log1p_leading_lowdm_fatjet_msd",
)

ENGINEERED_V2_GLOBAL_FEATURE_NAMES = (
    "log1p_lowdm_met_sqrt_ht",
    "min_dphi4",
    "lowdm_isr_subjet_btag_max",
    "met_over_isr_pt",
    "recoil_scalar_balance",
    "recoil_vector_balance",
    "met_parallel_over_isr",
    "ht_over_isr",
    "log1p_leading_lowdm_fatjet_pt",
    "leading_lowdm_fatjet_eta",
    "leading_lowdm_fatjet_met_dphi",
    "leading_lowdm_fatjet_msd_over_pt",
    "log1p_leading_b_pt",
    "log1p_subleading_b_pt",
    "has_two_medium_b",
    "log1p_min_mt_b_met",
    "log1p_max_mt_b_met",
    "log1p_m_bb",
    "delta_r_bb",
    "delta_phi_bb",
    "log1p_mct_bb",
    "has_resolved_top",
    "resolved_w_mass_residual",
    "resolved_top_mass_residual",
    "log1p_resolved_top_chi2",
    "log1p_resolved_w_pt",
    "log1p_resolved_top_pt",
    "transverse_sphericity",
    "centrality",
    "met_over_meff",
)

# The expanded domain admits NISR=0 and NISR>=2.  Keep the proven v2 feature
# block, but restore the basic event kinematics and add explicit ISR presence
# information so that missing-ISR sentinels never masquerade as physics.
ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES = (
    *BASE_GLOBAL_FEATURE_NAMES,
    *ENGINEERED_V2_GLOBAL_FEATURE_NAMES,
    "n_lowdm_isr",
    "has_lowdm_isr",
    "log1p_lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_dphi",
)

# Full replacement schema for the MET/sqrt(HT)-inclusive diagonal study.  It
# keeps the official Nt=NW=Nres=0 veto, but exposes continuous sub-threshold
# resolved masses and ISR/recoil information.  Every conditionally defined
# quantity has an explicit presence bit; missing ISR or a missing second b jet
# is never encoded as a physical zero without its mask.
DIAGONAL_V3_GLOBAL_FEATURE_NAMES = (
    "log1p_met",
    "log1p_ht",
    "njet",
    "nb_medium",
    "log1p_lowdm_met_sqrt_ht",
    "min_dphi4",
    "log1p_leading_b_pt",
    "log1p_subleading_b_pt",
    "has_two_medium_b",
    "log1p_leading_mt_b_met",
    "log1p_min_mt_b_met",
    "log1p_max_mt_b_met",
    "log1p_m_bb",
    "delta_r_bb",
    "delta_phi_bb",
    "log1p_mct_bb",
    "has_mjj_mjjj_candidate",
    "log1p_mjj",
    "log1p_mjjj",
    "resolved_w_mass_residual",
    "resolved_top_mass_residual",
    "log1p_resolved_top_chi2",
    "log1p_resolved_w_pt",
    "log1p_resolved_top_pt",
    "has_mt2_bb",
    "log1p_mt2_bb",
    "transverse_sphericity",
    "centrality",
    "met_over_meff",
    "n_lowdm_isr",
    "has_lowdm_isr",
    "log1p_lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_dphi",
    "lowdm_isr_subjet_btag_max",
    "met_over_isr_pt",
    "recoil_scalar_balance",
    "recoil_vector_balance",
    "met_parallel_over_isr",
    "ht_over_isr",
)

UPART_AK4_MEDIUM_WP_2024 = 0.1272


CANONICAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def canonical_config() -> dict:
    """Load the single adopted Low-dM analysis configuration."""
    return json.loads(CANONICAL_CONFIG_PATH.read_text())


def lowdm30_category_ids(nb: np.ndarray, nisr: np.ndarray) -> np.ndarray:
    """Map events to the adopted six SR categories."""
    nb = np.asarray(nb)
    nisr = np.asarray(nisr)
    nisr_group = np.where(nisr == 0, 0, np.where(nisr == 1, 1, 2))
    return ((nb >= 2).astype(np.int8) * 3 + nisr_group).astype(np.int8)


def lowdm30_coordinates(
    nb: np.ndarray, nisr: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """Return the flattened 0--29 coordinate of the adopted SR binning."""
    definition = canonical_config()["sr_binning"]
    labels = definition["category_labels"]
    categories = lowdm30_category_ids(nb, nisr)
    coordinates = np.empty(len(scores), dtype=np.int16)
    offset = 0
    for category, label in enumerate(labels):
        selected = categories == category
        edges = np.asarray(definition["edges_by_category"][label], dtype=float)
        local = np.searchsorted(edges, np.asarray(scores)[selected], side="right") - 1
        coordinates[selected] = offset + np.clip(local, 0, len(edges) - 2)
        offset += len(edges) - 1
    if offset != 30:
        raise RuntimeError(f"canonical Low-dM SR binning has {offset} bins, not 30")
    return coordinates


def _delta_phi_numpy(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def _four_vectors(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum((pt * np.cosh(eta)) ** 2 + mass**2, 0.0))
    return px, py, pz, energy


def _invariant_mass(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0)
    )


def _engineered_v2_features(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    btag: np.ndarray,
    mask: np.ndarray,
    met: np.ndarray,
    met_phi: np.ndarray,
    ht: np.ndarray,
    arrays: ak.Array,
) -> list[np.ndarray]:
    """Top-tagger-independent Low-dM physics features for the v2 campaign."""
    event_count = len(met)
    isr_pt = np.asarray(arrays["lowdm_isr_pt"], dtype=np.float32)
    isr_phi = np.asarray(arrays["lowdm_isr_phi"], dtype=np.float32)
    met_sqrt_ht = np.asarray(arrays["lowdm_met_sqrt_ht"], dtype=np.float32)
    min_dphi4 = np.asarray(arrays["min_dphi4"], dtype=np.float32)
    isr_subjet_btag = np.asarray(
        arrays["lowdm_isr_subjet_btag_max"], dtype=np.float32
    )
    safe_isr = np.maximum(isr_pt, 1.0)
    recoil_scalar_balance = np.abs(met - isr_pt) / np.maximum(met + isr_pt, 1.0)
    recoil_vector_balance = np.sqrt(
        np.maximum(
            met**2
            + isr_pt**2
            + 2.0 * met * isr_pt * np.cos(met_phi - isr_phi),
            0.0,
        )
    ) / np.maximum(met + isr_pt, 1.0)
    met_parallel_over_isr = -met * np.cos(met_phi - isr_phi) / safe_isr

    fat_pt = arrays["lowdm_fatjet_pt"]
    fat_order = ak.argsort(fat_pt, axis=1, ascending=False)

    def leading(branch: str, fill: float = 0.0) -> np.ndarray:
        values = arrays[branch][fat_order]
        return np.asarray(
            ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), fill)),
            dtype=np.float32,
        )

    leading_fat_pt = leading("lowdm_fatjet_pt")
    leading_fat_eta = leading("lowdm_fatjet_eta")
    leading_fat_phi = leading("lowdm_fatjet_phi")
    leading_fat_msd = leading("lowdm_fatjet_msd")

    bmask = mask & (btag >= UPART_AK4_MEDIUM_WP_2024)
    b_order = np.argsort(
        np.where(bmask, btag, -np.inf), axis=1, kind="stable"
    )[:, ::-1]
    row = np.arange(event_count)
    b1_index = b_order[:, 0]
    b2_index = b_order[:, 1]
    b1_valid = bmask[row, b1_index]
    b2_valid = bmask[row, b2_index]

    def selected(values: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.where(valid, values[row, indices], 0.0).astype(np.float32)

    b1_pt = selected(pt, b1_index, b1_valid)
    b1_eta = selected(eta, b1_index, b1_valid)
    b1_phi = selected(phi, b1_index, b1_valid)
    b1_mass = selected(mass, b1_index, b1_valid)
    b2_pt = selected(pt, b2_index, b2_valid)
    b2_eta = selected(eta, b2_index, b2_valid)
    b2_phi = selected(phi, b2_index, b2_valid)
    b2_mass = selected(mass, b2_index, b2_valid)
    all_b_mt = np.sqrt(
        np.maximum(
            2.0
            * pt
            * met[:, None]
            * (1.0 - np.cos(phi - met_phi[:, None])),
            0.0,
        )
    )
    min_b_mt = np.min(np.where(bmask, all_b_mt, np.inf), axis=1)
    max_b_mt = np.max(np.where(bmask, all_b_mt, -np.inf), axis=1)
    min_b_mt = np.where(np.isfinite(min_b_mt), min_b_mt, 0.0)
    max_b_mt = np.where(np.isfinite(max_b_mt), max_b_mt, 0.0)

    b1_px, b1_py, b1_pz, b1_energy = _four_vectors(
        b1_pt, b1_eta, b1_phi, b1_mass
    )
    b2_px, b2_py, b2_pz, b2_energy = _four_vectors(
        b2_pt, b2_eta, b2_phi, b2_mass
    )
    m_bb = _invariant_mass(
        b1_px + b2_px,
        b1_py + b2_py,
        b1_pz + b2_pz,
        b1_energy + b2_energy,
    )
    delta_eta_bb = b1_eta - b2_eta
    delta_phi_bb = _delta_phi_numpy(b1_phi, b2_phi)
    delta_r_bb = np.sqrt(delta_eta_bb**2 + delta_phi_bb**2)
    mct_bb = np.sqrt(
        np.maximum(2.0 * b1_pt * b2_pt * (1.0 + np.cos(delta_phi_bb)), 0.0)
    )
    for values in (m_bb, delta_r_bb, delta_phi_bb, mct_bb):
        values[~b2_valid] = 0.0

    px, py, pz, energy = _four_vectors(pt, eta, phi, mass)
    max_resolved_jets = min(8, pt.shape[1])
    best_chi2 = np.full(event_count, np.inf, dtype=np.float32)
    best_w_mass = np.zeros(event_count, dtype=np.float32)
    best_top_mass = np.zeros(event_count, dtype=np.float32)
    best_w_pt = np.zeros(event_count, dtype=np.float32)
    best_top_pt = np.zeros(event_count, dtype=np.float32)
    for first in range(max_resolved_jets):
        for second in range(first + 1, max_resolved_jets):
            w_px = px[:, first] + px[:, second]
            w_py = py[:, first] + py[:, second]
            w_pz = pz[:, first] + pz[:, second]
            w_energy = energy[:, first] + energy[:, second]
            w_mass = _invariant_mass(w_px, w_py, w_pz, w_energy)
            w_valid = (
                mask[:, first]
                & mask[:, second]
                & (btag[:, first] < UPART_AK4_MEDIUM_WP_2024)
                & (btag[:, second] < UPART_AK4_MEDIUM_WP_2024)
            )
            for third in range(max_resolved_jets):
                if third in (first, second):
                    continue
                candidate_valid = (
                    w_valid
                    & mask[:, third]
                    & (btag[:, third] >= UPART_AK4_MEDIUM_WP_2024)
                )
                top_px = w_px + px[:, third]
                top_py = w_py + py[:, third]
                top_pz = w_pz + pz[:, third]
                top_energy = w_energy + energy[:, third]
                top_mass = _invariant_mass(top_px, top_py, top_pz, top_energy)
                chi2 = ((w_mass - 80.4) / 15.0) ** 2 + (
                    (top_mass - 172.5) / 25.0
                ) ** 2
                better = candidate_valid & (chi2 < best_chi2)
                best_chi2[better] = chi2[better]
                best_w_mass[better] = w_mass[better]
                best_top_mass[better] = top_mass[better]
                best_w_pt[better] = np.hypot(w_px[better], w_py[better])
                best_top_pt[better] = np.hypot(top_px[better], top_py[better])
    has_resolved_top = np.isfinite(best_chi2)
    best_chi2 = np.where(has_resolved_top, best_chi2, 0.0)
    w_mass_residual = np.where(
        has_resolved_top, np.abs(best_w_mass - 80.4) / 80.4, 0.0
    )
    top_mass_residual = np.where(
        has_resolved_top, np.abs(best_top_mass - 172.5) / 172.5, 0.0
    )

    sum_pt2 = np.sum(np.where(mask, pt**2, 0.0), axis=1)
    sxx = np.sum(np.where(mask, px**2, 0.0), axis=1)
    syy = np.sum(np.where(mask, py**2, 0.0), axis=1)
    sxy = np.sum(np.where(mask, px * py, 0.0), axis=1)
    transverse_sphericity = 1.0 - np.sqrt(
        np.maximum((sxx - syy) ** 2 + 4.0 * sxy**2, 0.0)
    ) / np.maximum(sum_pt2, 1.0)
    scalar_energy = np.sum(np.where(mask, energy, 0.0), axis=1)
    centrality = ht / np.maximum(scalar_energy, 1.0)

    return [
        np.log1p(np.clip(met_sqrt_ht, 0.0, None)) / 4.0,
        np.clip(min_dphi4 / np.pi, 0.0, 1.0),
        np.clip(isr_subjet_btag, -1.0, 1.0),
        np.clip(met / safe_isr, 0.0, 3.0) / 3.0,
        np.clip(recoil_scalar_balance, 0.0, 1.0),
        np.clip(recoil_vector_balance, 0.0, 1.0),
        np.clip(met_parallel_over_isr, -2.0, 2.0) / 2.0,
        np.clip(ht / safe_isr, 0.0, 6.0) / 6.0,
        np.log1p(np.clip(leading_fat_pt, 0.0, None)) / 8.0,
        np.clip(leading_fat_eta / 3.0, -1.5, 1.5),
        _delta_phi_numpy(leading_fat_phi, met_phi) / np.pi,
        np.clip(leading_fat_msd / np.maximum(leading_fat_pt, 1.0), 0.0, 1.0),
        np.log1p(np.clip(b1_pt, 0.0, None)) / 7.0,
        np.log1p(np.clip(b2_pt, 0.0, None)) / 7.0,
        b2_valid.astype(np.float32),
        np.log1p(np.clip(min_b_mt, 0.0, None)) / 8.0,
        np.log1p(np.clip(max_b_mt, 0.0, None)) / 8.0,
        np.log1p(np.clip(m_bb, 0.0, None)) / 8.0,
        np.clip(delta_r_bb / 5.0, 0.0, 1.5),
        np.clip(delta_phi_bb / np.pi, 0.0, 1.0),
        np.log1p(np.clip(mct_bb, 0.0, None)) / 8.0,
        has_resolved_top.astype(np.float32),
        np.clip(w_mass_residual, 0.0, 3.0) / 3.0,
        np.clip(top_mass_residual, 0.0, 3.0) / 3.0,
        np.log1p(np.clip(best_chi2, 0.0, None)) / 6.0,
        np.log1p(np.clip(best_w_pt, 0.0, None)) / 8.0,
        np.log1p(np.clip(best_top_pt, 0.0, None)) / 8.0,
        np.clip(transverse_sphericity, 0.0, 1.0),
        np.clip(centrality, 0.0, 1.5),
        np.clip(met / np.maximum(met + ht, 1.0), 0.0, 1.0),
    ]


def _engineered_expanded_features(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    btag: np.ndarray,
    mask: np.ndarray,
    met: np.ndarray,
    met_phi: np.ndarray,
    ht: np.ndarray,
    njet: np.ndarray,
    nb: np.ndarray,
    arrays: ak.Array,
) -> list[np.ndarray]:
    """Sentinel-safe features for the NISR-unrestricted Low-dM domain."""
    engineered = _engineered_v2_features(
        pt=pt,
        eta=eta,
        phi=phi,
        mass=mass,
        btag=btag,
        mask=mask,
        met=met,
        met_phi=met_phi,
        ht=ht,
        arrays=arrays,
    )
    n_isr = np.asarray(arrays["n_lowdm_isr"], dtype=np.float32)
    raw_isr_pt = np.asarray(arrays["lowdm_isr_pt"], dtype=np.float32)
    raw_isr_eta = np.asarray(arrays["lowdm_isr_eta"], dtype=np.float32)
    raw_isr_dphi = np.asarray(arrays["lowdm_isr_dphi"], dtype=np.float32)
    has_isr = (n_isr > 0.0) & (raw_isr_pt > 0.0)

    # Positions 2--7 are the ISR-subjet and recoil/ISR ratios in v2.  For an
    # event without an ISR candidate they have no physical definition.
    engineered[2] = np.where(has_isr, engineered[2], -1.0).astype(np.float32)
    for index in range(3, 8):
        engineered[index] = np.where(has_isr, engineered[index], 0.0).astype(
            np.float32
        )

    basic = [
        np.log1p(np.clip(met, 0.0, None)) / 8.0,
        np.sin(met_phi),
        np.cos(met_phi),
        np.log1p(np.clip(ht, 0.0, None)) / 9.0,
        np.clip(njet / 12.0, 0.0, 1.5),
        np.clip(nb / 5.0, 0.0, 1.5),
    ]
    isr_features = [
        np.clip(n_isr / 4.0, 0.0, 1.5),
        has_isr.astype(np.float32),
        np.where(
            has_isr,
            np.log1p(np.clip(raw_isr_pt, 0.0, None)) / 8.0,
            0.0,
        ),
        np.where(has_isr, np.clip(raw_isr_eta / 3.0, -1.5, 1.5), 0.0),
        np.where(has_isr, np.clip(raw_isr_dphi / np.pi, 0.0, 1.0), 0.0),
    ]
    return [*basic, *engineered, *isr_features]


def _diagonal_v3_features(
    *,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    btag: np.ndarray,
    mask: np.ndarray,
    met: np.ndarray,
    met_phi: np.ndarray,
    ht: np.ndarray,
    njet: np.ndarray,
    nb: np.ndarray,
    arrays: ak.Array,
) -> list[np.ndarray]:
    """Rotation-invariant diagonal features with explicit missingness masks."""
    try:
        from mt2 import mt2 as compute_mt2
    except ImportError as error:  # pragma: no cover - exercised by environment audit
        raise RuntimeError(
            "the diagonal-v3 schema requires the pinned 'mt2' package"
        ) from error

    event_count = len(met)
    row = np.arange(event_count)
    px, py, pz, energy = _four_vectors(pt, eta, phi, np.abs(mass))

    bmask = mask & (btag >= UPART_AK4_MEDIUM_WP_2024)
    b_order = np.argsort(
        np.where(bmask, btag, -np.inf), axis=1, kind="stable"
    )[:, ::-1]
    b1_index = b_order[:, 0]
    b2_index = b_order[:, 1]
    b1_valid = bmask[row, b1_index]
    b2_valid = bmask[row, b2_index]

    def selected(
        values: np.ndarray, indices: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        return np.where(valid, values[row, indices], 0.0).astype(np.float32)

    b1_pt = selected(pt, b1_index, b1_valid)
    b1_eta = selected(eta, b1_index, b1_valid)
    b1_phi = selected(phi, b1_index, b1_valid)
    b1_mass = np.abs(selected(mass, b1_index, b1_valid))
    b2_pt = selected(pt, b2_index, b2_valid)
    b2_eta = selected(eta, b2_index, b2_valid)
    b2_phi = selected(phi, b2_index, b2_valid)
    b2_mass = np.abs(selected(mass, b2_index, b2_valid))

    all_b_mt = np.sqrt(
        np.maximum(
            2.0
            * pt
            * met[:, None]
            * (1.0 - np.cos(phi - met_phi[:, None])),
            0.0,
        )
    )
    min_b_mt = np.min(np.where(bmask, all_b_mt, np.inf), axis=1)
    max_b_mt = np.max(np.where(bmask, all_b_mt, -np.inf), axis=1)
    min_b_mt = np.where(np.isfinite(min_b_mt), min_b_mt, 0.0)
    max_b_mt = np.where(np.isfinite(max_b_mt), max_b_mt, 0.0)
    leading_b_mt = np.sqrt(
        np.maximum(
            2.0
            * b1_pt
            * met
            * (1.0 - np.cos(b1_phi - met_phi)),
            0.0,
        )
    )

    b1_px, b1_py, b1_pz, b1_energy = _four_vectors(
        b1_pt, b1_eta, b1_phi, b1_mass
    )
    b2_px, b2_py, b2_pz, b2_energy = _four_vectors(
        b2_pt, b2_eta, b2_phi, b2_mass
    )
    m_bb = _invariant_mass(
        b1_px + b2_px,
        b1_py + b2_py,
        b1_pz + b2_pz,
        b1_energy + b2_energy,
    )
    delta_phi_bb = _delta_phi_numpy(b1_phi, b2_phi)
    delta_r_bb = np.sqrt((b1_eta - b2_eta) ** 2 + delta_phi_bb**2)
    mct_bb = np.sqrt(
        np.maximum(2.0 * b1_pt * b2_pt * (1.0 + np.cos(delta_phi_bb)), 0.0)
    )
    for values in (m_bb, delta_phi_bb, delta_r_bb, mct_bb):
        values[~b2_valid] = 0.0

    met_px = met * np.cos(met_phi)
    met_py = met * np.sin(met_phi)
    mt2_bb = np.zeros(event_count, dtype=np.float32)
    if np.any(b2_valid):
        valid = b2_valid
        mt2_bb[valid] = np.asarray(
            compute_mt2(
                b1_mass[valid],
                b1_px[valid],
                b1_py[valid],
                b2_mass[valid],
                b2_px[valid],
                b2_py[valid],
                met_px[valid],
                met_py[valid],
                0.0,
                0.0,
                0.1,
            ),
            dtype=np.float32,
        )

    max_resolved_jets = min(8, pt.shape[1])
    best_chi2 = np.full(event_count, np.inf, dtype=np.float32)
    best_mjj = np.zeros(event_count, dtype=np.float32)
    best_mjjj = np.zeros(event_count, dtype=np.float32)
    best_w_pt = np.zeros(event_count, dtype=np.float32)
    best_top_pt = np.zeros(event_count, dtype=np.float32)
    for first in range(max_resolved_jets):
        for second in range(first + 1, max_resolved_jets):
            w_px = px[:, first] + px[:, second]
            w_py = py[:, first] + py[:, second]
            w_pz = pz[:, first] + pz[:, second]
            w_energy = energy[:, first] + energy[:, second]
            mjj = _invariant_mass(w_px, w_py, w_pz, w_energy)
            w_valid = (
                mask[:, first]
                & mask[:, second]
                & (btag[:, first] < UPART_AK4_MEDIUM_WP_2024)
                & (btag[:, second] < UPART_AK4_MEDIUM_WP_2024)
            )
            for third in range(max_resolved_jets):
                if third in (first, second):
                    continue
                candidate_valid = (
                    w_valid
                    & mask[:, third]
                    & (btag[:, third] >= UPART_AK4_MEDIUM_WP_2024)
                )
                top_px = w_px + px[:, third]
                top_py = w_py + py[:, third]
                top_pz = w_pz + pz[:, third]
                top_energy = w_energy + energy[:, third]
                mjjj = _invariant_mass(top_px, top_py, top_pz, top_energy)
                chi2 = ((mjj - 80.4) / 15.0) ** 2 + (
                    (mjjj - 172.5) / 25.0
                ) ** 2
                better = candidate_valid & (chi2 < best_chi2)
                best_chi2[better] = chi2[better]
                best_mjj[better] = mjj[better]
                best_mjjj[better] = mjjj[better]
                best_w_pt[better] = np.hypot(w_px[better], w_py[better])
                best_top_pt[better] = np.hypot(top_px[better], top_py[better])
    has_candidate = np.isfinite(best_chi2)
    best_chi2 = np.where(has_candidate, best_chi2, 0.0)
    w_mass_residual = np.where(
        has_candidate, np.abs(best_mjj - 80.4) / 80.4, 0.0
    )
    top_mass_residual = np.where(
        has_candidate, np.abs(best_mjjj - 172.5) / 172.5, 0.0
    )

    sum_pt2 = np.sum(np.where(mask, pt**2, 0.0), axis=1)
    sxx = np.sum(np.where(mask, px**2, 0.0), axis=1)
    syy = np.sum(np.where(mask, py**2, 0.0), axis=1)
    sxy = np.sum(np.where(mask, px * py, 0.0), axis=1)
    transverse_sphericity = 1.0 - np.sqrt(
        np.maximum((sxx - syy) ** 2 + 4.0 * sxy**2, 0.0)
    ) / np.maximum(sum_pt2, 1.0)
    scalar_energy = np.sum(np.where(mask, energy, 0.0), axis=1)
    centrality = ht / np.maximum(scalar_energy, 1.0)

    n_isr = np.asarray(arrays["n_lowdm_isr"], dtype=np.float32)
    isr_pt = np.asarray(arrays["lowdm_isr_pt"], dtype=np.float32)
    isr_eta = np.asarray(arrays["lowdm_isr_eta"], dtype=np.float32)
    isr_phi = np.asarray(arrays["lowdm_isr_phi"], dtype=np.float32)
    isr_dphi = np.asarray(arrays["lowdm_isr_dphi"], dtype=np.float32)
    isr_subjet_btag = np.asarray(
        arrays["lowdm_isr_subjet_btag_max"], dtype=np.float32
    )
    has_isr = (n_isr > 0.0) & np.isfinite(isr_pt) & (isr_pt > 0.0)
    safe_isr = np.maximum(isr_pt, 1.0)
    recoil_scalar_balance = np.abs(met - isr_pt) / np.maximum(
        met + isr_pt, 1.0
    )
    recoil_vector_balance = np.sqrt(
        np.maximum(
            met**2
            + isr_pt**2
            + 2.0 * met * isr_pt * np.cos(met_phi - isr_phi),
            0.0,
        )
    ) / np.maximum(met + isr_pt, 1.0)
    met_parallel_over_isr = -met * np.cos(met_phi - isr_phi) / safe_isr

    def only_with_isr(values: np.ndarray) -> np.ndarray:
        return np.where(has_isr, values, 0.0).astype(np.float32)

    met_sqrt_ht = np.asarray(arrays["lowdm_met_sqrt_ht"], dtype=np.float32)
    min_dphi4 = np.asarray(arrays["min_dphi4"], dtype=np.float32)
    return [
        np.log1p(np.clip(met, 0.0, None)) / 8.0,
        np.log1p(np.clip(ht, 0.0, None)) / 9.0,
        np.clip(njet / 12.0, 0.0, 1.5),
        np.clip(nb / 5.0, 0.0, 1.5),
        np.log1p(np.clip(met_sqrt_ht, 0.0, None)) / 4.0,
        np.clip(min_dphi4 / np.pi, 0.0, 1.0),
        np.log1p(np.clip(b1_pt, 0.0, None)) / 7.0,
        np.log1p(np.clip(b2_pt, 0.0, None)) / 7.0,
        b2_valid.astype(np.float32),
        np.log1p(np.clip(leading_b_mt, 0.0, None)) / 8.0,
        np.log1p(np.clip(min_b_mt, 0.0, None)) / 8.0,
        np.log1p(np.clip(max_b_mt, 0.0, None)) / 8.0,
        np.log1p(np.clip(m_bb, 0.0, None)) / 8.0,
        np.clip(delta_r_bb / 5.0, 0.0, 1.5),
        np.clip(delta_phi_bb / np.pi, 0.0, 1.0),
        np.log1p(np.clip(mct_bb, 0.0, None)) / 8.0,
        has_candidate.astype(np.float32),
        np.log1p(np.clip(best_mjj, 0.0, None)) / 7.0,
        np.log1p(np.clip(best_mjjj, 0.0, None)) / 8.0,
        np.clip(w_mass_residual, 0.0, 3.0) / 3.0,
        np.clip(top_mass_residual, 0.0, 3.0) / 3.0,
        np.log1p(np.clip(best_chi2, 0.0, None)) / 6.0,
        np.log1p(np.clip(best_w_pt, 0.0, None)) / 8.0,
        np.log1p(np.clip(best_top_pt, 0.0, None)) / 8.0,
        b2_valid.astype(np.float32),
        np.log1p(np.clip(mt2_bb, 0.0, None)) / 8.0,
        np.clip(transverse_sphericity, 0.0, 1.0),
        np.clip(centrality, 0.0, 1.5),
        np.clip(met / np.maximum(met + ht, 1.0), 0.0, 1.0),
        np.clip(n_isr / 4.0, 0.0, 1.5),
        has_isr.astype(np.float32),
        only_with_isr(np.log1p(np.clip(isr_pt, 0.0, None)) / 8.0),
        only_with_isr(np.clip(isr_eta / 3.0, -1.5, 1.5)),
        only_with_isr(np.clip(isr_dphi / np.pi, 0.0, 1.0)),
        only_with_isr(np.clip(isr_subjet_btag, -1.0, 1.0)),
        only_with_isr(np.clip(met / safe_isr, 0.0, 3.0) / 3.0),
        only_with_isr(np.clip(recoil_scalar_balance, 0.0, 1.0)),
        only_with_isr(np.clip(recoil_vector_balance, 0.0, 1.0)),
        only_with_isr(
            np.clip(met_parallel_over_isr, -2.0, 2.0) / 2.0
        ),
        only_with_isr(np.clip(ht / safe_isr, 0.0, 6.0) / 6.0),
    ]


@dataclass(frozen=True)
class GraphEvents:
    node_features: np.ndarray
    node_mask: np.ndarray
    node_eta: np.ndarray
    node_phi: np.ndarray
    global_features: np.ndarray
    labels: np.ndarray
    process_id: np.ndarray
    signal_topology_id: np.ndarray
    gen_weight: np.ndarray
    sampling_weight: np.ndarray
    fold: np.ndarray
    physical_dataset_id: np.ndarray
    run: np.ndarray
    luminosity_block: np.ndarray
    event: np.ndarray
    mstop: np.ndarray
    mlsp: np.ndarray
    lowdm_search_bin: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))

    def take(self, indices: np.ndarray) -> "GraphEvents":
        return GraphEvents(**{
            field: getattr(self, field)[indices]
            for field in self.__dataclass_fields__
        })


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 used only for deterministic event partitioning."""
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def event_hash(
    physical_dataset_id: np.ndarray,
    run: np.ndarray,
    luminosity_block: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Stable hash; the dataset id keeps independent MC samples independent."""
    fields = (physical_dataset_id, run, luminosity_block, event)
    out = np.zeros(len(event), dtype=np.uint64)
    for index, values in enumerate(fields):
        salt = np.uint64(
            (0xD1B54A32D192ED03 + index * 0x165667B19E3779F9)
            & 0xFFFFFFFFFFFFFFFF
        )
        out = splitmix64(out ^ splitmix64(np.asarray(values, dtype=np.uint64) ^ salt))
    return out


def fold_ids(
    physical_dataset_id: np.ndarray,
    run: np.ndarray,
    luminosity_block: np.ndarray,
    event: np.ndarray,
    folds: int,
) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least two")
    return (event_hash(physical_dataset_id, run, luminosity_block, event) % folds).astype(np.int64)


def split_buckets_2_1_7(
    physical_dataset_id: np.ndarray,
    run: np.ndarray,
    luminosity_block: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Return 0=train (20%), 1=validation (10%), 2=test (70%)."""
    remainder = event_hash(
        physical_dataset_id, run, luminosity_block, event
    ) % np.uint64(10)
    return np.where(remainder < 2, 0, np.where(remainder == 2, 1, 2)).astype(np.int8)


def _pad(values: ak.Array, length: int, fill: float = 0.0) -> np.ndarray:
    padded = ak.fill_none(ak.pad_none(values, length, axis=1, clip=True), fill)
    return np.asarray(ak.to_numpy(padded))


def supervised_signal_domain_mask(
    is_signal: np.ndarray,
    topology: np.ndarray,
    mstop: np.ndarray,
    mlsp: np.ndarray,
    *,
    topology_ids: tuple[int, ...] | None,
    delta_m_min: int | None,
    delta_m_max: int | None,
    mstop_min: int | None,
    mstop_max: int | None,
) -> np.ndarray:
    """Keep every background and only requested signal mass hypotheses."""
    signal_keep = np.ones(len(is_signal), dtype=bool)
    delta_m = np.asarray(mstop) - np.asarray(mlsp)
    if topology_ids is not None:
        signal_keep &= np.isin(topology, topology_ids)
    if delta_m_min is not None:
        signal_keep &= delta_m >= delta_m_min
    if delta_m_max is not None:
        signal_keep &= delta_m <= delta_m_max
    if mstop_min is not None:
        signal_keep &= mstop >= mstop_min
    if mstop_max is not None:
        signal_keep &= mstop <= mstop_max
    return ~np.asarray(is_signal, dtype=bool) | signal_keep


def _read_one(
    path: Path,
    *,
    target_mstop: int | None,
    target_mlsp: int | None,
    max_jets: int,
    folds: int,
    require_highdm_exclusive: bool,
    selection_branch: str,
    include_mass_features: bool,
    top_targeted_features: bool,
    engineered_features_v2: bool,
    engineered_features_expanded: bool,
    engineered_features_diagonal_v3: bool,
    signal_topology_ids: tuple[int, ...] | None = None,
    signal_delta_m_min: int | None = None,
    signal_delta_m_max: int | None = None,
    signal_mstop_min: int | None = None,
    signal_mstop_max: int | None = None,
    allow_empty: bool = False,
) -> GraphEvents | None:
    with uproot.open(path) as root_file:
        if "Events" not in root_file:
            raise RuntimeError(f"{path}: Events tree is missing")
        tree = root_file["Events"]
        # uproot exposes TTree fields through ``keys`` and RNTuple fields
        # through ``fields``.  Campaign caches can legitimately contain both
        # representations, so discover either without changing their schema.
        if hasattr(tree, "keys"):
            available_names = tree.keys()
        elif hasattr(tree, "fields"):
            available_names = tree.fields
        else:
            raise RuntimeError(f"{path}: unsupported Events object {type(tree)!r}")
        available = {str(name).split(";", 1)[0] for name in available_names}
        optional_branches = tuple(
            branch
            for branch in ("training_sampling_weight", "lowdm_search_bin_SR")
            if branch in available
        )
        requested_top = (
            TOP_TARGETED_BRANCHES
            if top_targeted_features
            or engineered_features_v2
            or engineered_features_expanded
            or engineered_features_diagonal_v3
            else ()
        )
        if engineered_features_diagonal_v3:
            requested_engineered = DIAGONAL_V3_EXTRA_BRANCHES
        elif engineered_features_expanded:
            requested_engineered = ENGINEERED_EXPANDED_EXTRA_BRANCHES
        elif engineered_features_v2:
            requested_engineered = ENGINEERED_V2_EXTRA_BRANCHES
        else:
            requested_engineered = ()
        read_branches = tuple(
            dict.fromkeys(
                (
                    *EVENT_BRANCHES,
                    selection_branch,
                    *optional_branches,
                    *requested_top,
                    *requested_engineered,
                )
            )
        )
        missing = sorted(set(read_branches) - available)
        if missing:
            raise RuntimeError(f"{path}: missing branches: {', '.join(missing)}")
        arrays = tree.arrays(read_branches, library="ak")

    is_signal = np.asarray(arrays["is_signal"], dtype=bool)
    is_background = np.asarray(arrays["is_background"], dtype=bool)
    if np.any(is_signal & is_background):
        raise RuntimeError(f"{path}: events cannot be both signal and background")
    if np.any(~(is_signal | is_background)):
        raise RuntimeError(
            f"{path}: collision data or unlabeled events cannot enter supervised training"
        )
    keep = np.asarray(arrays[selection_branch], dtype=bool)
    if require_highdm_exclusive:
        keep &= ~np.asarray(arrays["feature_SR"], dtype=bool)
    if target_mstop is not None or target_mlsp is not None:
        if target_mstop is None or target_mlsp is None:
            raise ValueError("target_mstop and target_mlsp must both be set or both be None")
        keep &= (
            ~is_signal
            | (
                (np.asarray(arrays["mStop"], dtype=np.int32) == target_mstop)
                & (np.asarray(arrays["mLSP"], dtype=np.int32) == target_mlsp)
            )
        )
    signal_filter_requested = any(
        value is not None
        for value in (
            signal_topology_ids,
            signal_delta_m_min,
            signal_delta_m_max,
            signal_mstop_min,
            signal_mstop_max,
        )
    )
    if signal_filter_requested:
        topology = np.asarray(arrays["signal_topology_id"], dtype=np.int32)
        mstop_values = np.asarray(arrays["mStop"], dtype=np.int32)
        mlsp_values = np.asarray(arrays["mLSP"], dtype=np.int32)
        keep &= supervised_signal_domain_mask(
            is_signal,
            topology,
            mstop_values,
            mlsp_values,
            topology_ids=signal_topology_ids,
            delta_m_min=signal_delta_m_min,
            delta_m_max=signal_delta_m_max,
            mstop_min=signal_mstop_min,
            mstop_max=signal_mstop_max,
        )
    if not np.any(keep):
        if allow_empty:
            return None
        raise RuntimeError(f"{path}: no events satisfy the requested domain")
    arrays = arrays[keep]
    is_signal = np.asarray(arrays["is_signal"], dtype=bool)

    pt = arrays["jet_corrected_pt"]
    eta = arrays["jet_eta_all"]
    phi = arrays["jet_phi_all"]
    mass = arrays["jet_corrected_mass"]
    btag = arrays["jet_btag_upart_all"]
    jet_id = arrays["jet_id_all"]
    good = (pt > 30.0) & (abs(eta) < 2.5) & (jet_id != 0)
    pt, eta, phi, mass, btag = (
        values[good] for values in (pt, eta, phi, mass, btag)
    )
    order = ak.argsort(pt, axis=1, ascending=False)
    pt, eta, phi, mass, btag = (
        values[order] for values in (pt, eta, phi, mass, btag)
    )

    pt_np = _pad(pt, max_jets)
    eta_np = _pad(eta, max_jets)
    phi_np = _pad(phi, max_jets)
    mass_np = _pad(mass, max_jets)
    btag_np = _pad(btag, max_jets, fill=-1.0)
    mask = pt_np > 0.0

    node_features = np.stack(
        (
            np.log1p(np.clip(pt_np, 0.0, None)) / 8.0,
            eta_np / 3.0,
            np.sin(phi_np),
            np.cos(phi_np),
            np.log1p(np.clip(abs(mass_np), 0.0, None)) / 6.0,
            np.clip(btag_np, -1.0, 1.0),
        ),
        axis=-1,
    ).astype(np.float32)
    node_features[~mask] = 0.0

    n = len(arrays)
    mstop = np.asarray(arrays["mStop"], dtype=np.int32)
    mlsp = np.asarray(arrays["mLSP"], dtype=np.int32)
    met = np.asarray(arrays["met"], dtype=np.float32)
    met_phi = np.asarray(arrays["met_phi"], dtype=np.float32)
    ht = np.asarray(arrays["ht"], dtype=np.float32)
    njet = np.asarray(arrays["njet"], dtype=np.float32)
    nb = np.asarray(arrays["nb_medium"], dtype=np.float32)
    if engineered_features_diagonal_v3:
        global_columns = _diagonal_v3_features(
            pt=pt_np,
            eta=eta_np,
            phi=phi_np,
            mass=mass_np,
            btag=btag_np,
            mask=mask,
            met=met,
            met_phi=met_phi,
            ht=ht,
            njet=njet,
            nb=nb,
            arrays=arrays,
        )
        relative_phi = np.arctan2(
            np.sin(phi_np - met_phi[:, None]),
            np.cos(phi_np - met_phi[:, None]),
        ).astype(np.float32)
        relative_phi[~mask] = 0.0
        phi_np = relative_phi
        node_features[..., 2] = np.sin(relative_phi)
        node_features[..., 3] = np.cos(relative_phi)
        node_features[~mask] = 0.0
    elif engineered_features_expanded:
        global_columns = _engineered_expanded_features(
            pt=pt_np,
            eta=eta_np,
            phi=phi_np,
            mass=mass_np,
            btag=btag_np,
            mask=mask,
            met=met,
            met_phi=met_phi,
            ht=ht,
            njet=njet,
            nb=nb,
            arrays=arrays,
        )
    elif engineered_features_v2:
        global_columns = _engineered_v2_features(
            pt=pt_np,
            eta=eta_np,
            phi=phi_np,
            mass=mass_np,
            btag=btag_np,
            mask=mask,
            met=met,
            met_phi=met_phi,
            ht=ht,
            arrays=arrays,
        )
    else:
        global_columns = [
            np.log1p(np.clip(met, 0.0, None)) / 8.0,
            np.sin(met_phi),
            np.cos(met_phi),
            np.log1p(np.clip(ht, 0.0, None)) / 9.0,
            np.clip(njet / 12.0, 0.0, 1.5),
            np.clip(nb / 5.0, 0.0, 1.5),
        ]
    if (
        top_targeted_features
        and not engineered_features_v2
        and not engineered_features_expanded
        and not engineered_features_diagonal_v3
    ):
        isr_pt = np.asarray(arrays["lowdm_isr_pt"], dtype=np.float32)
        isr_eta = np.asarray(arrays["lowdm_isr_eta"], dtype=np.float32)
        isr_dphi = np.asarray(arrays["lowdm_isr_dphi"], dtype=np.float32)
        met_sqrt_ht = np.asarray(arrays["lowdm_met_sqrt_ht"], dtype=np.float32)
        ptb = np.asarray(arrays["lowdm_ptb"], dtype=np.float32)
        mtb = np.asarray(arrays["lowdm_mtb"], dtype=np.float32)
        jet_met_dphi = [
            np.asarray(arrays[f"j{index}_met_dphi"], dtype=np.float32)
            for index in range(1, 5)
        ]
        min_dphi4 = np.asarray(arrays["min_dphi4"], dtype=np.float32)
        isr_subjet_btag = np.asarray(
            arrays["lowdm_isr_subjet_btag_max"], dtype=np.float32
        )
        fat_pt = arrays["lowdm_fatjet_pt"]
        fat_order = ak.argsort(fat_pt, axis=1, ascending=False)
        fat_msd = arrays["lowdm_fatjet_msd"][fat_order]

        def leading(values: ak.Array, fill: float = 0.0) -> np.ndarray:
            return np.asarray(
                ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), fill)),
                dtype=np.float32,
            )

        leading_fat_msd = leading(fat_msd)
        global_columns.extend(
            [
                np.log1p(np.clip(isr_pt, 0.0, None)) / 8.0,
                np.clip(isr_eta / 3.0, -2.0, 2.0),
                np.clip(isr_dphi / np.pi, 0.0, 1.0),
                np.log1p(np.clip(met_sqrt_ht, 0.0, None)) / 4.0,
                np.log1p(np.clip(ptb, 0.0, None)) / 6.0,
                np.log1p(np.clip(mtb, 0.0, None)) / 7.0,
                *[
                    np.clip(current / np.pi, 0.0, 1.0)
                    for current in jet_met_dphi
                ],
                np.clip(min_dphi4 / np.pi, 0.0, 1.0),
                np.clip(isr_subjet_btag, -1.0, 1.0),
                np.log1p(np.clip(abs(leading_fat_msd), 0.0, None)) / 6.0,
            ]
        )
    if include_mass_features:
        if target_mstop is None or target_mlsp is None:
            raise ValueError(
                "mass-hypothesis inputs require one target mass point; disable them for multi-mass training"
            )
        hypothesis_mstop = np.where(is_signal, mstop, target_mstop)
        hypothesis_mlsp = np.where(is_signal, mlsp, target_mlsp)
        global_columns.extend(
            [
                np.asarray(hypothesis_mstop, dtype=np.float32) / 2000.0,
                np.asarray(hypothesis_mlsp, dtype=np.float32) / 2000.0,
                np.asarray(hypothesis_mstop - hypothesis_mlsp, dtype=np.float32) / 500.0,
            ]
        )
    global_features = np.stack(global_columns, axis=-1).astype(np.float32)

    physical = np.asarray(arrays["physical_dataset_id"], dtype=np.int64)
    run = np.asarray(arrays["run"], dtype=np.int64)
    lumi = np.asarray(arrays["luminosityBlock"], dtype=np.int64)
    event_number = np.asarray(arrays["event"], dtype=np.int64)
    assigned_fold = fold_ids(physical, run, lumi, event_number, folds)
    return GraphEvents(
        node_features=node_features,
        node_mask=mask.astype(bool),
        node_eta=eta_np.astype(np.float32),
        node_phi=phi_np.astype(np.float32),
        global_features=global_features,
        labels=is_signal.astype(np.float32),
        process_id=np.asarray(arrays["process_id"], dtype=np.int64),
        signal_topology_id=np.asarray(arrays["signal_topology_id"], dtype=np.int32),
        gen_weight=np.asarray(arrays["gen_weight"], dtype=np.float64),
        sampling_weight=(
            np.asarray(arrays["training_sampling_weight"], dtype=np.float64)
            if "training_sampling_weight" in arrays.fields
            else np.ones(len(arrays), dtype=np.float64)
        ),
        fold=assigned_fold,
        physical_dataset_id=physical,
        run=run,
        luminosity_block=lumi,
        event=event_number,
        mstop=np.asarray(mstop, dtype=np.int32),
        mlsp=np.asarray(mlsp, dtype=np.int32),
        lowdm_search_bin=(
            np.asarray(arrays["lowdm_search_bin_SR"], dtype=np.int32)
            if "lowdm_search_bin_SR" in arrays.fields
            else np.full(len(arrays), -1, dtype=np.int32)
        ),
    )


def concatenate(parts: Iterable[GraphEvents]) -> GraphEvents:
    materialized = list(parts)
    if not materialized:
        raise ValueError("at least one GraphEvents object is required")
    return GraphEvents(**{
        field: np.concatenate([getattr(part, field) for part in materialized], axis=0)
        for field in GraphEvents.__dataclass_fields__
    })


def load_graph_events(
    signal_paths: Iterable[Path],
    background_paths: Iterable[Path],
    *,
    target_mstop: int | None,
    target_mlsp: int | None,
    max_jets: int,
    folds: int,
    require_highdm_exclusive: bool = True,
    selection_branch: str = "feature_lowdm_preselection",
    include_mass_features: bool = True,
    top_targeted_features: bool = False,
    engineered_features_v2: bool = False,
    engineered_features_expanded: bool = False,
    engineered_features_diagonal_v3: bool = False,
    signal_topology_ids: tuple[int, ...] | None = None,
    signal_delta_m_min: int | None = None,
    signal_delta_m_max: int | None = None,
    signal_mstop_min: int | None = None,
    signal_mstop_max: int | None = None,
) -> GraphEvents:
    feature_schemas = sum(
        int(value)
        for value in (
            top_targeted_features,
            engineered_features_v2,
            engineered_features_expanded,
            engineered_features_diagonal_v3,
        )
    )
    if feature_schemas > 1:
        raise ValueError(
            "top_targeted_features, engineered_features_v2, and "
            "engineered_features_expanded, and engineered_features_diagonal_v3 "
            "are mutually exclusive schemas"
        )
    paths = [*(Path(path) for path in signal_paths), *(Path(path) for path in background_paths)]
    signal_filter_requested = any(
        value is not None
        for value in (
            signal_topology_ids,
            signal_delta_m_min,
            signal_delta_m_max,
            signal_mstop_min,
            signal_mstop_max,
        )
    )
    parts = []
    for path in paths:
        part = _read_one(
            path,
            target_mstop=target_mstop,
            target_mlsp=target_mlsp,
            max_jets=max_jets,
            folds=folds,
            require_highdm_exclusive=require_highdm_exclusive,
            selection_branch=selection_branch,
            include_mass_features=include_mass_features,
            top_targeted_features=top_targeted_features,
            engineered_features_v2=engineered_features_v2,
            engineered_features_expanded=engineered_features_expanded,
            engineered_features_diagonal_v3=engineered_features_diagonal_v3,
            signal_topology_ids=signal_topology_ids,
            signal_delta_m_min=signal_delta_m_min,
            signal_delta_m_max=signal_delta_m_max,
            signal_mstop_min=signal_mstop_min,
            signal_mstop_max=signal_mstop_max,
            allow_empty=signal_filter_requested,
        )
        if part is not None:
            parts.append(part)
    events = concatenate(parts)
    for name in ("node_features", "node_eta", "node_phi", "global_features"):
        values = getattr(events, name)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"non-finite classifier input in {name}: "
                f"{np.count_nonzero(~np.isfinite(values))} values"
            )
    order = np.argsort(
        event_hash(
            events.physical_dataset_id,
            events.run,
            events.luminosity_block,
            events.event,
        ),
        kind="stable",
    )
    return events.take(order)
