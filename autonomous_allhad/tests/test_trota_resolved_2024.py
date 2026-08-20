from __future__ import annotations

import math

import numpy as np

from autonomous_allhad.trota_resolved_2024 import (
    SELECTED_TOPRESOLVED_2024_WORKING_POINT,
    TOPRESOLVED_2024_QCD_WORKING_POINTS,
    build_event_candidates,
    delta_phi,
    selected_jet_indices,
    upstream_ordered_triplets,
)


def test_selected_working_point_is_one_percent_qcd_mistag() -> None:
    assert SELECTED_TOPRESOLVED_2024_WORKING_POINT == "1pct_qcd_mistag"
    assert math.isclose(
        TOPRESOLVED_2024_QCD_WORKING_POINTS[SELECTED_TOPRESOLVED_2024_WORKING_POINT],
        0.9433798789978027,
    )


def test_upstream_ordered_triplets_matches_combination_count() -> None:
    assert upstream_ordered_triplets(2) == []
    assert upstream_ordered_triplets(3) == [(2, 1, 0)]
    assert len(upstream_ordered_triplets(8)) == math.comb(8, 3)


def test_selected_jet_indices_uses_stored_id_and_trota_kinematics() -> None:
    assert selected_jet_indices(
        [25.0, 25.1, 40.0, 60.0],
        [0.0, 0.1, 2.5, -2.49],
        [1, 1, 1, 0],
    ) == [1]


def test_build_event_candidates_has_official_resolved_shape_and_order() -> None:
    features, candidates, good = build_event_candidates(
        pt=[40.0, 50.0, 60.0, 70.0],
        eta=[0.1, -0.2, 0.3, -0.4],
        phi=[0.2, -0.4, 0.8, -1.0],
        mass=[5.0, 6.0, 7.0, 8.0],
        area=[0.4, 0.5, 0.6, 0.7],
        btag=[0.01, 0.02, 0.03, 0.04],
        jet_id=[1, 1, 1, 1],
    )

    assert good == [0, 1, 2, 3]
    assert features.shape == (4, 3, 8)
    assert candidates[0]["source_jet_indices"] == [2, 1, 0]
    np.testing.assert_allclose(features[0, :, 0], [0.6, 0.5, 0.4])
    np.testing.assert_allclose(features[0, :, 1], [0.03, 0.02, 0.01])
    np.testing.assert_allclose(features[0, :, 5], [60.0, 50.0, 40.0])
    np.testing.assert_allclose(features[0, :, 6], [0.8, -0.4, 0.2])
    np.testing.assert_allclose(features[0, :, 7], [0.3, -0.2, 0.1])
    assert np.all(np.isfinite(features))


def test_delta_phi_matches_signed_wrapping() -> None:
    assert math.isclose(delta_phi(3.0, -3.0), 6.0 - 2.0 * math.pi)
    assert math.isclose(delta_phi(-3.0, 3.0), -6.0 + 2.0 * math.pi)
