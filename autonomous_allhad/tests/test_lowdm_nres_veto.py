from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

import build_flat_boosted_recoil_hists as hist


def test_lowdm_region_mask_preserves_legacy_selection_without_trota() -> None:
    chunk = {
        "feature_lowdm_SR": np.asarray([True, True, False]),
        "nb_medium_lowdm": np.asarray([1, 0, 2]),
    }
    assert hist.lowdm_region_mask(chunk, "SR", 3).tolist() == [True, False, False]


def test_lowdm_region_mask_requires_zero_trota_resolved_tops() -> None:
    chunk = {
        "feature_lowdm_SR": np.asarray([True, True, True]),
        "nb_medium_lowdm": np.asarray([1, 2, 1]),
        hist.DERIVED_NRES_BRANCH: np.asarray([0, 1, 2]),
    }
    assert hist.lowdm_region_mask(chunk, "SR", 3).tolist() == [True, False, False]


def test_focused_lowdm_sr_indices_apply_trota_veto() -> None:
    n = 2
    chunk = {
        "feature_lowdm_preselection": np.ones(n, dtype=bool),
        "pass_lowdm_topology_veto": np.ones(n, dtype=bool),
        "pass_lowdm_isr": np.ones(n, dtype=bool),
        "pass_lowdm_met_sqrt_ht": np.ones(n, dtype=bool),
        "nb_medium_lowdm": np.ones(n, dtype=int),
        hist.DERIVED_NRES_BRANCH: np.asarray([0, 1]),
        "njet": np.asarray([4, 4]),
        "lowdm_isr_pt": np.asarray([550.0, 550.0]),
        "lowdm_ptb": np.asarray([50.0, 50.0]),
        "met": np.asarray([550.0, 550.0]),
        "lowdm_mtb": np.asarray([100.0, 100.0]),
    }
    indices = hist.lowdm_nsv_inclusive_sr_indices(chunk, n)
    assert indices[0] >= 0
    assert indices[1] == -1
