from __future__ import annotations

import itertools

import pytest

from autonomous_allhad.highdm_resolved_categories import (
    COARSE_NRES_TOPOLOGIES,
    boosted_overlap_vetoed_ak4_indices,
    coarse_nres_topology_label,
    high_mtb_topology_label,
    select_exclusive_resolved_candidates,
)


def _vetoed(**overrides):
    values = {
        "jet_source_indices": [0, 1, 2],
        "jet_eta": [0.05, 1.05, 2.5],
        "jet_phi": [0.05, 1.05, 2.5],
        "fatjet_eta": [0.0],
        "fatjet_phi": [0.0],
        "fatjet_subjet_index1": [0],
        "fatjet_subjet_index2": [1],
        "fatjet_top_pass": [1],
        "fatjet_w_pass": [0],
        "subjet_eta": [0.0, 1.0],
        "subjet_phi": [0.0, 1.0],
    }
    values.update(overrides)
    return boosted_overlap_vetoed_ak4_indices(**values)


def test_two_valid_subjets_use_dr_0p4_around_subjets() -> None:
    assert _vetoed() == frozenset({0, 1})


def test_missing_second_subjet_uses_dr_0p8_around_fatjet() -> None:
    assert _vetoed(
        jet_eta=[0.70, 1.05, 2.5],
        jet_phi=[0.0, 1.05, 2.5],
        fatjet_subjet_index2=[-1],
    ) == frozenset({0})


def test_nonselected_fatjet_does_not_veto_ak4_jets() -> None:
    assert _vetoed(fatjet_top_pass=[0], fatjet_w_pass=[0]) == frozenset()


def test_resolved_selection_applies_boosted_veto_before_greedy_arbitration() -> None:
    result = select_exclusive_resolved_candidates(
        candidate_indices=[4, 7, 9, 11],
        candidate_scores=[0.99, 0.90, 0.80, 0.70],
        candidate_source_jets=[
            [1, 2, 3],
            [10, 11, 12],
            [12, 13, 14],
            [100, 101, 102],
        ],
        boosted_vetoed_ak4_indices={2},
    )
    assert result.selected_candidate_indices == (7, 11)
    assert result.rejected_by_boosted_overlap == (4,)
    assert result.rejected_by_resolved_overlap == (9,)
    assert result.nres == 2


def test_resolved_selection_tie_breaks_on_candidate_index() -> None:
    result = select_exclusive_resolved_candidates(
        candidate_indices=[8, 3],
        candidate_scores=[0.5, 0.5],
        candidate_source_jets=[[1, 2, 3], [3, 4, 5]],
    )
    assert result.selected_candidate_indices == (3,)
    assert result.rejected_by_resolved_overlap == (8,)


def test_resolved_selection_supports_source_indices_above_63() -> None:
    result = select_exclusive_resolved_candidates(
        candidate_indices=[0, 1],
        candidate_scores=[0.9, 0.8],
        candidate_source_jets=[[64, 65, 66], [67, 68, 69]],
    )
    assert result.nres == 2


@pytest.mark.parametrize(
    ("nb", "nt", "nw", "nres", "expected"),
    [
        (1, 0, 0, 0, "nb1_none"),
        (1, 2, 0, 0, "nb1_top"),
        (1, 1, 0, 2, "nb1_top_resolved"),
        (1, 1, 1, 1, "nb1_sum3plus"),
        (2, 0, 0, 1, "nb2_resolved1"),
        (2, 0, 0, 2, "nb2_resolved2"),
        (2, 1, 0, 1, "nb2_top1_resolved1"),
        (3, 0, 1, 1, "nb3plus_w1_resolved1"),
        (4, 1, 1, 1, "nb3plus_sum3plus"),
    ],
)
def test_high_mtb_topology_labels(
    nb: int, nt: int, nw: int, nres: int, expected: str,
) -> None:
    assert high_mtb_topology_label(
        nb=nb, nt=nt, nw=nw, nres=nres, high_mtb=True,
    ) == expected


def test_low_mtb_and_zero_b_are_outside_scope() -> None:
    assert high_mtb_topology_label(
        nb=2, nt=0, nw=0, nres=1, high_mtb=False,
    ) is None
    assert high_mtb_topology_label(
        nb=0, nt=0, nw=0, nres=1, high_mtb=True,
    ) is None


def test_nb2plus_category_space_is_exhaustive_through_large_counts() -> None:
    for nb, nt, nw, nres in itertools.product((2, 3), range(5), range(5), range(5)):
        label = high_mtb_topology_label(
            nb=nb, nt=nt, nw=nw, nres=nres, high_mtb=True,
        )
        assert label is not None


@pytest.mark.parametrize(
    ("nt", "nw", "nres", "expected"),
    [
        (0, 0, 1, "resolved1_only"),
        (0, 0, 2, "resolved2plus_only"),
        (0, 0, 7, "resolved2plus_only"),
        (1, 0, 1, "top_resolved"),
        (3, 0, 2, "top_resolved"),
        (0, 1, 1, "w_resolved"),
        (0, 4, 2, "w_resolved"),
        (1, 1, 1, "top_w_resolved"),
        (2, 3, 2, "top_w_resolved"),
    ],
)
def test_coarse_nres_topology_labels(
    nt: int, nw: int, nres: int, expected: str,
) -> None:
    assert coarse_nres_topology_label(
        nt=nt, nw=nw, nres=nres, high_mtb=True,
    ) == expected


def test_coarse_nres_topology_keeps_nres0_in_baseline() -> None:
    assert coarse_nres_topology_label(
        nt=2, nw=1, nres=0, high_mtb=True,
    ) is None
    assert coarse_nres_topology_label(
        nt=0, nw=0, nres=2, high_mtb=False,
    ) is None


def test_coarse_nres_topology_space_is_exhaustive() -> None:
    observed = {
        coarse_nres_topology_label(
            nt=nt, nw=nw, nres=nres, high_mtb=True,
        )
        for nt, nw, nres in itertools.product(range(4), range(4), range(1, 5))
    }
    assert observed == set(COARSE_NRES_TOPOLOGIES)
