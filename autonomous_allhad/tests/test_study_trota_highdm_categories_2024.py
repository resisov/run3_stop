from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "workflow"
    / "study_trota_highdm_categories_2024.py"
)
SPEC = importlib.util.spec_from_file_location("trota_highdm_study", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_event_join_uses_file_and_entry() -> None:
    mapped = MODULE.map_candidates_to_events(
        np.asarray([8, 3, 8]),
        np.asarray([2, 9, 7]),
        np.asarray([8, 8, 3, 8]),
        np.asarray([7, 2, 9, 7]),
    )
    assert mapped.tolist() == [2, 0, 1, 2]


def test_candidate_event_fallback_join_uses_run_lumi_event() -> None:
    mapped = MODULE.map_candidates_to_events_rle(
        np.asarray([1, 1, 2]),
        np.asarray([10, 10, 20]),
        np.asarray([101, 102, 201]),
        np.asarray([2, 1, 1, 2]),
        np.asarray([20, 10, 10, 20]),
        np.asarray([201, 102, 101, 201]),
    )
    assert mapped.tolist() == [2, 1, 0, 2]


def test_disjoint_arbitration_prefers_highest_score() -> None:
    counts = MODULE.greedy_disjoint_counts(
        event_index=np.asarray([0, 0, 0, 1]),
        source0=np.asarray([0, 0, 3, 0]),
        source1=np.asarray([1, 3, 4, 1]),
        source2=np.asarray([2, 4, 5, 2]),
        score=np.asarray([0.95, 0.99, 0.96, 0.97]),
        number_of_events=2,
    )
    # Event 0 chooses (0,3,4) first.  Both remaining candidates overlap it.
    assert counts.tolist() == [1, 1]


def test_disjoint_arbitration_keeps_two_nonoverlapping_triplets() -> None:
    counts = MODULE.greedy_disjoint_counts(
        event_index=np.asarray([0, 0, 0]),
        source0=np.asarray([0, 3, 0]),
        source1=np.asarray([1, 4, 4]),
        source2=np.asarray([2, 5, 6]),
        score=np.asarray([0.99, 0.98, 0.97]),
        number_of_events=1,
    )
    assert counts.tolist() == [2]
