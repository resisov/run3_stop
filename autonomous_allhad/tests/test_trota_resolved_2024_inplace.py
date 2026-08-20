from __future__ import annotations

import numpy as np
import pytest
import uproot

from autonomous_allhad.trota_resolved_2024 import build_event_candidates
from autonomous_allhad.trota_resolved_2024_inplace import (
    FLOAT_OUTPUT_BRANCHES,
    OUTPUT_DTYPES,
    WP_THRESHOLD,
    _build_candidates_impl,
    _validate_partial_matches_expected,
    schema_version_for_year,
    verify_complete_root,
)


def test_sparse_builder_matches_official_candidate_order_and_features() -> None:
    pt = np.asarray([40.0, 50.0, 60.0, 70.0], dtype=np.float32)
    eta = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    phi = np.asarray([0.2, -0.4, 0.8, -1.0], dtype=np.float32)
    mass = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    area = np.asarray([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    btag = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    jet_id = np.asarray([True, True, True, True], dtype=np.bool_)

    output = _build_candidates_impl(
        np.asarray([0, 4], dtype=np.int64),
        pt,
        eta,
        phi,
        mass,
        area,
        btag,
        jet_id,
    )
    reference_features, reference_candidates, _ = build_event_candidates(
        pt, eta, phi, mass, area, btag, jet_id
    )

    counts, features = output[0], output[1]
    assert counts.tolist() == [4]
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, reference_features, rtol=2e-6, atol=2e-6)
    assert output[3].tolist() == [0, 1, 2, 3]
    assert list(zip(output[4], output[5], output[6])) == [
        (2, 1, 0),
        (3, 1, 0),
        (3, 2, 0),
        (3, 2, 1),
    ]
    assert list(zip(output[7], output[8], output[9])) == [
        tuple(candidate["source_jet_indices"]) for candidate in reference_candidates
    ]


def test_persisted_floats_and_working_point_are_float32() -> None:
    assert WP_THRESHOLD.dtype == np.dtype(np.float32)
    assert float(WP_THRESHOLD) == 0.9433798789978027
    assert FLOAT_OUTPUT_BRANCHES
    assert all(OUTPUT_DTYPES[name] == np.dtype(np.float32) for name in FLOAT_OUTPUT_BRANCHES)


def test_application_year_has_distinct_provenance_schema() -> None:
    assert schema_version_for_year(2024) == "trota_topresolved_2024_inplace_sparse_v1"
    assert schema_version_for_year(2025) == "trota_topresolved_2025_inplace_sparse_v1"
    with pytest.raises(ValueError, match="unsupported TROTA target year"):
        schema_version_for_year(2026)


def test_partial_recovery_requires_fresh_inference_match(tmp_path) -> None:
    root_path = tmp_path / "partial.root"
    payload = {
        name: np.arange(3, dtype=dtype)
        for name, dtype in OUTPUT_DTYPES.items()
    }
    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree("TROTA", OUTPUT_DTYPES)
        tree.extend(payload)

    result = _validate_partial_matches_expected(root_path, payload, step_size=2)
    assert result["partial_tree_rows_compared"] == 3
    assert result["partial_tree_matches_fresh_inference"] is True

    mismatched = {name: values.copy() for name, values in payload.items()}
    mismatched["TopResolved1pct_candidateIndex"][1] += 1
    with pytest.raises(RuntimeError, match="identity branch"):
        _validate_partial_matches_expected(root_path, mismatched, step_size=2)


def test_verify_complete_root_rejects_an_unmarked_tree(tmp_path) -> None:
    root_path = tmp_path / "unmarked.root"
    payload = {name: np.empty(0, dtype=dtype) for name, dtype in OUTPUT_DTYPES.items()}
    with uproot.recreate(root_path) as root_file:
        root_file.mktree("Events", {"entry": np.int64})
        root_file.mktree("TROTA", OUTPUT_DTYPES).extend(payload)
    with pytest.raises(RuntimeError, match="without a completion marker"):
        verify_complete_root(root_path)
