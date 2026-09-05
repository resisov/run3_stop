"""Reconstruct the frozen diagonal-v3 GNN inputs in SR and control regions.

The SR, LLCR, and QCDCR use the nominal MET and nominal object collection.
GCR and DYCR use their recoil and object-cleaned jet/fat-jet collections.  The
feature order and scaling are delegated to :mod:`gnn_lowdm.data`, so a change
to the frozen schema cannot silently diverge between training and inference.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np

try:
    from . import region_io as base
    from ..data import (
        DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
        _diagonal_v3_features,
        _pad,
    )
except ImportError:  # EOS worker executes this file from the payload tree.
    import region_io as base  # type: ignore[no-redef]
    from data import (  # type: ignore[no-redef]
        DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
        _diagonal_v3_features,
        _pad,
    )


NOMINAL_REGIONS = {"SR", "LLCR", "QCDCR"}


def _reference_objects(
    arrays: ak.Array, region: str
) -> tuple[ak.Array, ak.Array]:
    if region in NOMINAL_REGIONS:
        empty = arrays["jet_eta_all"][:, :0]
        return empty, empty
    masks = base.object_masks(arrays)
    if region == "GCR":
        return (
            arrays["photon_eta_all"][masks["photon_medium"]],
            arrays["photon_phi_all"][masks["photon_medium"]],
        )
    if region == "DY2E":
        return (
            arrays["electron_eta_all"][masks["electron_medium"]],
            arrays["electron_phi_all"][masks["electron_medium"]],
        )
    if region == "DY2M":
        return (
            arrays["muon_eta_all"][masks["muon_medium"]],
            arrays["muon_phi_all"][masks["muon_medium"]],
        )
    raise ValueError("unknown Low-dM region: " + region)


def _leading(values: ak.Array, fill: float) -> np.ndarray:
    return np.asarray(
        ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), fill)),
        dtype=np.float32,
    )


def _region_isr(
    arrays: ak.Array,
    region: str,
    selected: np.ndarray,
    reference_eta: ak.Array,
    reference_phi: ak.Array,
    recoil_phi: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return ISR inputs and audit nominal-subjet information reuse.

    The intermediate ROOT schema stores the subjet discriminator only for the
    nominal leading low-dM ISR fat jet.  In a photon/dilepton CR it is reused
    only when the cleaned leading fat jet is demonstrably the same object.
    Otherwise the trained missing-value sentinel (-1 after clipping) is used.
    """
    if region in NOMINAL_REGIONS:
        n_isr = np.asarray(arrays["n_lowdm_isr"][selected], dtype=np.float32)
        isr_pt = np.asarray(arrays["lowdm_isr_pt"][selected], dtype=np.float32)
        isr_eta = np.asarray(arrays["lowdm_isr_eta"][selected], dtype=np.float32)
        isr_phi = np.asarray(arrays["lowdm_isr_phi"][selected], dtype=np.float32)
        isr_dphi = np.asarray(arrays["lowdm_isr_dphi"][selected], dtype=np.float32)
        isr_btag = np.asarray(
            arrays["lowdm_isr_subjet_btag_max"][selected], dtype=np.float32
        )
        has_isr = (n_isr > 0.0) & np.isfinite(isr_pt) & (isr_pt > 0.0)
        return {
            "n_lowdm_isr": n_isr,
            "lowdm_isr_pt": isr_pt,
            "lowdm_isr_eta": isr_eta,
            "lowdm_isr_phi": isr_phi,
            "lowdm_isr_dphi": isr_dphi,
            "lowdm_isr_subjet_btag_max": isr_btag,
        }, {
            "events": int(len(n_isr)),
            "has_isr": int(np.count_nonzero(has_isr)),
            "subjet_btag_nominal_match": int(np.count_nonzero(has_isr)),
            "subjet_btag_missing_after_cleaning": 0,
        }

    fat_pt = arrays["fatjet_corrected_pt"]
    fat_eta = arrays["fatjet_eta_all"]
    fat_phi = arrays["fatjet_phi_all"]
    clean = base.clean_by_delta_r(
        fat_eta, fat_phi, reference_eta, reference_phi, 0.4
    )
    good = (
        (fat_pt > 200.0)
        & (abs(fat_eta) < 2.4)
        & base.as_ak_bool(arrays["fatjet_id_all"])
        & clean
    )
    order = ak.argsort(fat_pt[good], axis=1, ascending=False)
    selected_pt = fat_pt[good][order][selected]
    selected_eta = fat_eta[good][order][selected]
    selected_phi = fat_phi[good][order][selected]
    n_isr = np.asarray(ak.num(selected_pt, axis=1), dtype=np.float32)
    isr_pt = _leading(selected_pt, -99.0)
    isr_eta = _leading(selected_eta, -99.0)
    isr_phi = _leading(selected_phi, -99.0)
    has_isr = (n_isr > 0.0) & np.isfinite(isr_pt) & (isr_pt > 0.0)
    isr_dphi = np.abs(
        np.arctan2(
            np.sin(isr_phi - recoil_phi), np.cos(isr_phi - recoil_phi)
        )
    ).astype(np.float32)
    isr_dphi[~has_isr] = -99.0

    nominal_n = np.asarray(arrays["n_lowdm_isr"][selected], dtype=np.float32)
    nominal_pt = np.asarray(arrays["lowdm_isr_pt"][selected], dtype=np.float32)
    nominal_eta = np.asarray(arrays["lowdm_isr_eta"][selected], dtype=np.float32)
    nominal_phi = np.asarray(arrays["lowdm_isr_phi"][selected], dtype=np.float32)
    nominal_btag = np.asarray(
        arrays["lowdm_isr_subjet_btag_max"][selected], dtype=np.float32
    )
    nominal_has = (nominal_n > 0.0) & np.isfinite(nominal_pt) & (nominal_pt > 0.0)
    same_object = (
        has_isr
        & nominal_has
        & np.isclose(isr_pt, nominal_pt, rtol=2.0e-5, atol=2.0e-3)
        & np.isclose(isr_eta, nominal_eta, rtol=0.0, atol=2.0e-5)
        & (
            np.abs(
                np.arctan2(
                    np.sin(isr_phi - nominal_phi),
                    np.cos(isr_phi - nominal_phi),
                )
            )
            < 2.0e-5
        )
    )
    isr_btag = np.full(len(n_isr), -1.0, dtype=np.float32)
    isr_btag[same_object] = nominal_btag[same_object]
    return {
        "n_lowdm_isr": n_isr,
        "lowdm_isr_pt": isr_pt,
        "lowdm_isr_eta": isr_eta,
        "lowdm_isr_phi": isr_phi,
        "lowdm_isr_dphi": isr_dphi,
        "lowdm_isr_subjet_btag_max": isr_btag,
    }, {
        "events": int(len(n_isr)),
        "has_isr": int(np.count_nonzero(has_isr)),
        "subjet_btag_nominal_match": int(np.count_nonzero(same_object)),
        "subjet_btag_missing_after_cleaning": int(
            np.count_nonzero(has_isr & ~same_object)
        ),
    }


def feature_arrays(
    arrays: ak.Array,
    block: base.RegionBlock,
    region: str,
    selected: np.ndarray,
    max_jets: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    """Return the exact five model inputs plus a reconstruction audit."""
    if region not in base.REGIONS:
        raise ValueError("unknown Low-dM region: " + region)
    selected = np.asarray(selected, dtype=bool)
    reference_eta, reference_phi = _reference_objects(arrays, region)

    jet_pt = arrays["jet_corrected_pt"]
    jet_eta = arrays["jet_eta_all"]
    jet_phi = arrays["jet_phi_all"]
    jet_mass = arrays["jet_corrected_mass"]
    jet_btag = arrays["jet_btag_upart_all"]
    jet_clean = base.clean_by_delta_r(
        jet_eta, jet_phi, reference_eta, reference_phi, 0.2
    )
    graph_good = (
        (jet_pt > 30.0)
        & (abs(jet_eta) < 2.5)
        & base.as_ak_bool(arrays["jet_id_all"])
        & jet_clean
    )
    order = ak.argsort(jet_pt[graph_good], axis=1, ascending=False)
    graph_values = [
        values[graph_good][order][selected]
        for values in (jet_pt, jet_eta, jet_phi, jet_mass, jet_btag)
    ]
    pt, eta, phi, mass, btag = [
        _pad(values, max_jets, fill=-1.0 if index == 4 else 0.0)
        for index, values in enumerate(graph_values)
    ]
    # Keep the cache branch precision through engineered-feature arithmetic.
    # The training loader casts only the final model tensors to float32.
    pt = np.asarray(pt)
    eta = np.asarray(eta)
    phi = np.asarray(phi)
    mass = np.asarray(mass)
    btag = np.asarray(btag)
    mask = pt > 0.0

    recoil = np.asarray(block.recoil[selected], dtype=np.float32)
    recoil_phi = np.asarray(block.recoil_phi[selected], dtype=np.float32)
    ht = np.asarray(block.ht[selected], dtype=np.float32)
    njet = np.asarray(block.njet[selected], dtype=np.float32)
    nb = np.asarray(block.nb[selected], dtype=np.float32)
    if region in NOMINAL_REGIONS:
        met_sqrt_ht = np.asarray(
            arrays["lowdm_met_sqrt_ht"][selected], dtype=np.float32
        )
        min_dphi4 = np.asarray(arrays["min_dphi4"][selected], dtype=np.float32)
    else:
        met_sqrt_ht = np.divide(
            recoil,
            np.sqrt(ht),
            out=np.zeros(len(ht), dtype=np.float32),
            where=ht > 0.0,
        )
        analysis_good = (
            (jet_pt > 30.0)
            & (abs(jet_eta) < 2.4)
            & base.as_ak_bool(arrays["jet_id_all"])
            & jet_clean
        )
        analysis_order = ak.argsort(
            jet_pt[analysis_good], axis=1, ascending=False
        )
        analysis_phi = jet_phi[analysis_good][analysis_order][selected]
        dphi4 = base.delta_phi(analysis_phi[:, :4], recoil_phi[:, None])
        min_dphi4 = np.asarray(
            ak.to_numpy(ak.fill_none(ak.min(dphi4, axis=1), 0.0)),
            dtype=np.float32,
        )

    isr, isr_audit = _region_isr(
        arrays,
        region,
        selected,
        reference_eta,
        reference_phi,
        recoil_phi,
    )
    feature_source: dict[str, Any] = {
        **isr,
        "lowdm_met_sqrt_ht": met_sqrt_ht,
        "min_dphi4": min_dphi4,
    }
    global_columns = _diagonal_v3_features(
        pt=pt,
        eta=eta,
        phi=phi,
        mass=mass,
        btag=btag,
        mask=mask,
        met=recoil,
        met_phi=recoil_phi,
        ht=ht,
        njet=njet,
        nb=nb,
        arrays=feature_source,
    )
    global_features = np.stack(global_columns, axis=-1).astype(np.float32)
    if global_features.shape[1] != len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES):
        raise RuntimeError("diagonal-v3 global feature width mismatch")

    relative_phi = np.arctan2(
        np.sin(phi - recoil_phi[:, None]),
        np.cos(phi - recoil_phi[:, None]),
    ).astype(np.float32)
    relative_phi[~mask] = 0.0
    node_features = np.stack(
        (
            np.log1p(np.clip(pt, 0.0, None)) / 8.0,
            eta / 3.0,
            np.sin(relative_phi),
            np.cos(relative_phi),
            np.log1p(np.clip(np.abs(mass), 0.0, None)) / 6.0,
            np.clip(btag, -1.0, 1.0),
        ),
        axis=-1,
    ).astype(np.float32)
    node_features[~mask] = 0.0
    for name, values in (
        ("node_features", node_features),
        ("node_eta", eta),
        ("node_phi", relative_phi),
        ("global_features", global_features),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "%s: non-finite diagonal-v3 input in %s" % (region, name)
            )
    isr_index = list(DIAGONAL_V3_GLOBAL_FEATURE_NAMES).index("n_lowdm_isr")
    feature_nisr = np.rint(global_features[:, isr_index] * 4.0).astype(int)
    block_nisr = np.asarray(block.nisr[selected], dtype=int)
    audit = {
        **isr_audit,
        "selected_events": int(np.count_nonzero(selected)),
        "nisr_block_feature_mismatches": int(
            np.count_nonzero(feature_nisr != block_nisr)
        ),
    }
    return (
        node_features,
        mask.astype(bool),
        eta.astype(np.float32),
        relative_phi,
        global_features,
        audit,
    )
