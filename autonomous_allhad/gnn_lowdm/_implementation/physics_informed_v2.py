"""Physics-informed inputs and training weights for the common Low-dM GNN.

The helpers in this module deliberately do not modify the legacy feature or
weight definitions.  Existing checkpoints therefore keep their original
semantics while the v2 studies can be audited independently.
"""

from __future__ import annotations

import numpy as np

from ..data import ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES, GraphEvents


CORE_GLOBAL_FEATURE_NAMES = (
    "log1p_met",
    "log1p_ht",
    "nb_medium",
    "log1p_lowdm_met_sqrt_ht",
    "min_dphi4",
    "met_over_isr_pt",
    "recoil_scalar_balance",
    "recoil_vector_balance",
    "met_parallel_over_isr",
    "log1p_leading_b_pt",
    "log1p_subleading_b_pt",
    "has_two_medium_b",
    "log1p_min_mt_b_met",
    "log1p_max_mt_b_met",
    "log1p_m_bb",
    "delta_r_bb",
    "delta_phi_bb",
    "log1p_mct_bb",
    "n_lowdm_isr",
    "has_lowdm_isr",
    "log1p_lowdm_isr_pt",
    "lowdm_isr_dphi",
)

RESOLVED_GLOBAL_FEATURE_NAMES = (
    *CORE_GLOBAL_FEATURE_NAMES,
    "has_resolved_top",
    "resolved_w_mass_residual",
    "resolved_top_mass_residual",
    "log1p_resolved_top_chi2",
    "log1p_resolved_w_pt",
    "log1p_resolved_top_pt",
)

# Absolute azimuth is removed in every v2 feature set.  Only relative angles
# and rotational invariants are retained.
FULL_ROTATION_INVARIANT_GLOBAL_FEATURE_NAMES = tuple(
    name
    for name in ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES
    if name not in {"sin_met_phi", "cos_met_phi"}
)

RESOLVED_RECONSTRUCTION_FEATURE_NAMES = (
    "has_resolved_top",
    "resolved_w_mass_residual",
    "resolved_top_mass_residual",
    "log1p_resolved_top_chi2",
    "log1p_resolved_w_pt",
    "log1p_resolved_top_pt",
)

EXTENDED_NO_RESOLVED_GLOBAL_FEATURE_NAMES = tuple(
    name
    for name in FULL_ROTATION_INVARIANT_GLOBAL_FEATURE_NAMES
    if name not in set(RESOLVED_RECONSTRUCTION_FEATURE_NAMES)
)

FEATURE_SETS = {
    "core": CORE_GLOBAL_FEATURE_NAMES,
    "core_resolved": RESOLVED_GLOBAL_FEATURE_NAMES,
    "full_rotation_invariant": FULL_ROTATION_INVARIANT_GLOBAL_FEATURE_NAMES,
    "extended_no_resolved": EXTENDED_NO_RESOLVED_GLOBAL_FEATURE_NAMES,
}


def wrap_delta_phi(values: np.ndarray) -> np.ndarray:
    """Wrap an angle to [-pi, pi] without a branch-dependent convention."""
    values = np.asarray(values)
    return np.arctan2(np.sin(values), np.cos(values))


def rotate_graph_to_met_frame_inplace(events: GraphEvents) -> None:
    """Rotate the process-local graph arrays so MET defines phi=0.

    The full expanded cache stores sin(MET phi) and cos(MET phi) at fixed
    positions.  This function mutates only the arrays owned by the current
    training process; no ROOT cache or previous checkpoint is rewritten.
    """
    names = list(ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES)
    sin_index = names.index("sin_met_phi")
    cos_index = names.index("cos_met_phi")
    met_phi = np.arctan2(
        events.global_features[:, sin_index],
        events.global_features[:, cos_index],
    ).astype(np.float32)
    relative_phi = wrap_delta_phi(events.node_phi - met_phi[:, None]).astype(
        np.float32
    )
    relative_phi[~events.node_mask] = 0.0
    events.node_phi[...] = relative_phi
    events.node_features[..., 2] = np.sin(relative_phi)
    events.node_features[..., 3] = np.cos(relative_phi)
    events.node_features[~events.node_mask] = 0.0


def select_global_features(events: GraphEvents, feature_set: str) -> GraphEvents:
    """Return a shallow event view with one explicitly named global schema."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown physics-informed feature set: {feature_set}")
    source_names = list(ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES)
    names = FEATURE_SETS[feature_set]
    indices = np.asarray([source_names.index(name) for name in names], dtype=np.int64)
    payload = {
        field: getattr(events, field)
        for field in GraphEvents.__dataclass_fields__
    }
    payload["global_features"] = np.asarray(
        events.global_features[:, indices], dtype=np.float32
    )
    return GraphEvents(**payload)


def _effective_events(weights: np.ndarray) -> float:
    total = float(weights.sum())
    sumw2 = float(np.square(weights).sum())
    return total * total / sumw2 if sumw2 > 0.0 else 0.0


def physics_informed_loss_weights(
    events: GraphEvents,
    indices: np.ndarray,
    analysis_weights: np.ndarray,
    *,
    delta_m_edges: tuple[int, ...] = (150, 201, 251, 301),
    cap_quantile: float = 0.995,
) -> np.ndarray:
    """Build xsec-aware weights without giving every sparse point equal budget.

    Background keeps its physical cross-section mixture.  Signal first shares
    equal total budget among topologies and populated delta-M bands.  Within a
    band, a mass point receives a budget proportional to sqrt(N_eff), so sparse
    points remain represented but cannot dominate through a handful of very
    large event weights.
    """
    if not 0.9 <= cap_quantile <= 1.0:
        raise ValueError("cap_quantile must be in [0.9, 1.0]")
    if len(delta_m_edges) < 2 or any(
        right <= left for left, right in zip(delta_m_edges, delta_m_edges[1:])
    ):
        raise ValueError("delta_m_edges must be strictly increasing")
    indices = np.asarray(indices, dtype=np.int64)
    labels = events.labels[indices].astype(bool)
    absolute = np.abs(np.asarray(analysis_weights)[indices]).astype(np.float64)
    if not np.any(labels) or not np.any(~labels):
        raise RuntimeError("training split must contain signal and background")
    output = np.zeros(len(indices), dtype=np.float64)
    output[~labels] = absolute[~labels]
    output[~labels] *= 0.5 / output[~labels].sum()

    topology = events.signal_topology_id[indices]
    mstop = events.mstop[indices]
    mlsp = events.mlsp[indices]
    delta_m = mstop - mlsp
    edges = np.asarray(delta_m_edges, dtype=np.int32)
    band = np.searchsorted(edges, delta_m, side="right") - 1
    valid_band = (band >= 0) & (band < len(edges) - 1)
    if np.any(labels & ~valid_band):
        bad = np.unique(delta_m[labels & ~valid_band]).tolist()
        raise RuntimeError(f"signal delta-M values outside configured bands: {bad}")

    topologies = np.unique(topology[labels])
    topology_budget = 0.5 / len(topologies)
    for topology_id in topologies:
        topology_mask = labels & (topology == topology_id)
        populated_bands = np.unique(band[topology_mask])
        cell_budget = topology_budget / len(populated_bands)
        for band_id in populated_bands:
            cell = topology_mask & (band == band_id)
            point_values = np.unique(
                np.stack((mstop[cell], mlsp[cell]), axis=1), axis=0
            )
            point_masks: list[np.ndarray] = []
            point_scales: list[float] = []
            for point_mstop, point_mlsp in point_values:
                local = cell & (mstop == point_mstop) & (mlsp == point_mlsp)
                point_masks.append(local)
                point_scales.append(np.sqrt(_effective_events(absolute[local])))
            scale_total = float(np.sum(point_scales))
            if scale_total <= 0.0:
                raise RuntimeError("zero effective signal weight in topology/delta-M cell")
            for local, scale in zip(point_masks, point_scales, strict=True):
                local_total = float(absolute[local].sum())
                if local_total <= 0.0:
                    raise RuntimeError("zero signal weight in mass point")
                output[local] = (
                    cell_budget * scale / scale_total * absolute[local] / local_total
                )
            positive = output[cell & (output > 0.0)]
            if len(positive) and cap_quantile < 1.0:
                cap = float(np.quantile(positive, cap_quantile))
                output[cell] = np.minimum(output[cell], cap)
                cell_total = float(output[cell].sum())
                output[cell] *= cell_budget / cell_total

    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise RuntimeError("invalid physics-informed loss weights")
    output *= len(output) / output.sum()
    return output.astype(np.float32)
