from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_broad_lowdm_mask_uses_exact_topology_and_trota_contract() -> None:
    block = SimpleNamespace(
        core=np.asarray([True, True, True, True, True, False]),
        nb=np.asarray([1, 0, 1, 1, 1, 2]),
        nt=np.asarray([0, 0, 1, 0, 0, 0]),
        nw=np.asarray([0, 0, 0, 1, 0, 0]),
    )
    chunk = {
        hist.DERIVED_NRES_BRANCH: np.asarray([0, 0, 0, 0, 1, 0]),
        # These retired requirements must not affect the broad selection.
        "pass_lowdm_isr": np.zeros(6, dtype=bool),
        "pass_lowdm_met_sqrt_ht": np.zeros(6, dtype=bool),
        "pass_lowdm_topology_veto": np.zeros(6, dtype=bool),
        "feature_lowdm_SR": np.zeros(6, dtype=bool),
        "feature_SR": np.ones(6, dtype=bool),
    }
    assert hist.broad_lowdm_region_mask(block, chunk, 6).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_2025_known_unavailable_weight_components_are_exactly_scoped() -> None:
    status = {
        "components": {
            "electron_hlt": {
                "applied": False,
                "error": "2025 electron HLT SF is not available in the payload",
            },
            "photon_csev": {
                "applied": False,
                "error": "requested correction has no published working-point content",
            },
            "muon_hlt": {"applied": False, "error": "unrelated failure"},
        }
    }
    required = ["electron_hlt", "photon_csev", "muon_hlt"]
    accepted = hist.accepted_known_unavailable_weight_components(
        "2025", required, status
    )
    assert set(accepted) == {"electron_hlt", "photon_csev"}
    assert hist.accepted_known_unavailable_weight_components(
        "2024", required, status
    ) == {}
