from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "workflow" / "highdm_selection_fom_2024.py"
SPEC = importlib.util.spec_from_file_location("highdm_selection_fom_2024", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_highdm_mask_matrix_matches_current_sr_order() -> None:
    arrays = {
        "pass_base_common": np.asarray([True, True, True]),
        "pass_signal_trigger": np.asarray([True, True, True]),
        "pass_no_veto_leptons": np.asarray([True, True, True]),
        "pass_zero_tau": np.asarray([True, True, True]),
        "njet": np.asarray([5, 4, 7]),
        "nb_medium": np.asarray([1, 1, 2]),
        "pass_met_250": np.asarray([True, True, True]),
        "pass_ht_300": np.asarray([True, True, True]),
        "pass_open_high": np.asarray([True, True, True]),
        "feature_SR": np.asarray([True, False, True]),
        "met": np.asarray([300.0, 300.0, 1600.0]),
        "nboosted_top": np.asarray([0, 0, 1]),
        "nboosted_w": np.asarray([0, 0, 0]),
    }
    matrix = MODULE.mask_matrix(arrays)
    assert matrix.shape == (3, len(MODULE.SELECTIONS))
    assert matrix[0].all()
    assert not matrix[1, 4]
    assert not matrix[1, -1]
    assert not matrix[2, -1]


def test_valid_54bin_coverage_matches_adopted_selected_categories() -> None:
    arrays = {
        "feature_SR": np.ones(10, dtype=bool),
        "met": np.full(10, 400.0),
        "nb_medium": np.asarray([1, 2, 1, 1, 2, 2, 3, 3, 3, 3]),
        "nboosted_top": np.asarray([0, 0, 1, 2, 1, 1, 1, 1, 2, 2]),
        "nboosted_w": np.asarray([0, 3, 0, 4, 0, 2, 0, 1, 0, 1]),
    }
    expected = np.asarray([True, True, True, True, True, False, True, True, True, False])
    np.testing.assert_array_equal(MODULE.valid_54bin_mask(arrays), expected)


def test_cumulative_stats_and_absolute_weighted_correlation() -> None:
    n = 4
    arrays = {
        "pass_base_common": np.ones(n, dtype=bool),
        "pass_signal_trigger": np.asarray([True, True, False, True]),
        "pass_no_veto_leptons": np.ones(n, dtype=bool),
        "pass_zero_tau": np.ones(n, dtype=bool),
        "njet": np.asarray([5, 5, 5, 4]),
        "nb_medium": np.ones(n, dtype=int),
        "pass_met_250": np.ones(n, dtype=bool),
        "pass_ht_300": np.ones(n, dtype=bool),
        "pass_open_high": np.ones(n, dtype=bool),
        "feature_SR": np.asarray([True, True, False, False]),
        "met": np.full(n, 400.0),
        "nboosted_top": np.zeros(n, dtype=int),
        "nboosted_w": np.zeros(n, dtype=int),
    }
    weights = np.asarray([1.0, -0.5, 2.0, 3.0])
    stats = MODULE.stats_for_mask(arrays, np.ones(n, dtype=bool), weights)
    assert stats["entries"][0] == 4
    assert stats["entries"][2] == 3
    assert stats["entries"][5] == 2
    assert stats["entries"][-1] == 2
    assert MODULE.correlation(stats).shape == (len(MODULE.SELECTIONS), len(MODULE.SELECTIONS))
