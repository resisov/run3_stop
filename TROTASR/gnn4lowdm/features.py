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


def _pad(values: ak.Array, length: int, fill: float = 0.0) -> np.ndarray:
    padded = ak.fill_none(ak.pad_none(values, length, axis=1, clip=True), fill)
    return np.asarray(ak.to_numpy(padded))

