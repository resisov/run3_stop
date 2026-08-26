from __future__ import annotations

import numpy as np

from autonomous_allhad.search_bin_categorization import (
    ADDITIONAL_CATEGORY_SOURCE_OFFSET,
    EXCLUSIVE_SOURCE_BIN_COUNT,
    adopted55_mapping,
    configured_bin_position_groups,
    configured_exclusive_bin_count,
    configured_exclusive_groups,
    configured_exclusive_mapping,
    configured_projection_groups,
    exclusive_category_source_indices,
    exclusive_category_source_labels,
    map60_indices_to_adopted55,
    map_category_sources_to_configured,
)


CURRENT_CONFIGURATION = {
    "baseline_category_sizes": [6, 6, 6, 5, 6, 5, 5, 6, 5, 5],
    "layout": [
        {"kind": "baseline", "index": 0},
        {"kind": "topology", "name": "resolved1_only"},
        {"kind": "topology", "name": "resolved2plus_only"},
        {"kind": "baseline", "index": 1},
        {"kind": "topology", "name": "w_resolved"},
        {"kind": "topology", "name": "top_resolved"},
        {"kind": "topology", "name": "top_w_resolved"},
        {"kind": "baseline", "index": 2},
        {"kind": "baseline", "index": 3},
        {"kind": "category", "name": "nb2_nt0_nw2_nres0"},
        *({"kind": "baseline", "index": index} for index in range(4, 10)),
    ],
    "omitted_topologies": [],
    "omitted_categories": [],
}

MERGED_CONFIGURATION = {
    **CURRENT_CONFIGURATION,
    "bin_merges_1based": [
        [17, 18],
        [19, 54],
        [20, 55],
        [21, 56],
        [22, 57],
        [23, 58],
        [24, 59],
        [82, 87],
        [83, 88],
        [84, 89],
        [85, 90],
        [86, 91],
    ],
}


def test_adopted55_mapping_covers_precursor_once() -> None:
    mapping = adopted55_mapping()
    assert len(mapping) == 55
    assert sorted(sum((list(group) for group in mapping), [])) == list(range(60))
    assert [group for group in mapping if len(group) == 2] == [
        (22, 23), (34, 35), (40, 41), (52, 53), (58, 59),
    ]


def test_exclusive_category_sources_reassign_every_baseline_event_once() -> None:
    baseline60 = np.arange(60, dtype=int)
    baseline55 = map60_indices_to_adopted55(baseline60)
    baseline55 = np.tile(baseline55, 6)
    recoil = np.tile(np.arange(6), 60)
    nb = np.ones(360, dtype=int)
    nt = np.asarray(([0] * 60) + ([0] * 60) + ([1] * 60) + ([0] * 60) + ([1] * 60) + ([2] * 60))
    nw = np.asarray(([0] * 60) + ([0] * 60) + ([0] * 60) + ([1] * 60) + ([1] * 60) + ([3] * 60))
    nres = np.asarray(([0] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([2] * 60))
    output = exclusive_category_source_indices(
        baseline55, recoil, nb, nt, nw, nres
    )
    assert len(output) == len(baseline55)
    assert np.all((output >= 0) & (output < EXCLUSIVE_SOURCE_BIN_COUNT))
    assert np.array_equal(output[:60], baseline55[:60])


def test_nb2_w2_nres0_moves_out_of_the_baseline_exclusively() -> None:
    output = exclusive_category_source_indices(
        np.asarray([6, 6, 6]),
        np.asarray([0, 3, 5]),
        np.asarray([2, 2, 3]),
        np.asarray([0, 0, 0]),
        np.asarray([2, 2, 2]),
        np.asarray([0, 0, 0]),
    )
    assert output.tolist() == [
        ADDITIONAL_CATEGORY_SOURCE_OFFSET,
        ADDITIONAL_CATEGORY_SOURCE_OFFSET + 3,
        6,
    ]


def test_labels_have_expected_sizes() -> None:
    source = [f"source_{index}" for index in range(60)]
    labels = exclusive_category_source_labels(source)
    assert len(labels) == EXCLUSIVE_SOURCE_BIN_COUNT
    assert labels[0].endswith("__Nres0")
    assert labels[-1] == "nb2_nt0_nw2_nres0__recoil_800plus"


def test_current_search_bins_follow_requested_category_order() -> None:
    mapping = configured_exclusive_mapping(CURRENT_CONFIGURATION)
    assert len(mapping) == 91
    assert mapping[:6] == tuple(range(6))
    assert mapping[6:12] == tuple(range(55, 61))
    assert mapping[12:18] == tuple(range(61, 67))
    assert mapping[18:24] == tuple(range(6, 12))
    assert mapping[24:30] == tuple(range(73, 79))
    assert mapping[30:36] == tuple(range(67, 73))
    assert mapping[36:42] == tuple(range(79, 85))
    assert mapping[53:59] == tuple(range(85, 91))
    assert set(mapping) == set(range(EXCLUSIVE_SOURCE_BIN_COUNT))


def test_current_configuration_maps_every_category_source() -> None:
    source = np.arange(EXCLUSIVE_SOURCE_BIN_COUNT, dtype=int)
    mapped = map_category_sources_to_configured(source, CURRENT_CONFIGURATION)
    assert np.all(mapped >= 0)


def test_configured_merges_absorb_the_sparse_nb2_w2_category() -> None:
    positions = configured_bin_position_groups(MERGED_CONFIGURATION)
    groups = configured_exclusive_groups(MERGED_CONFIGURATION)
    assert configured_exclusive_bin_count(MERGED_CONFIGURATION) == 79
    assert positions[16] == (16, 17)
    assert groups[16] == (65, 66)
    assert groups[17:23] == (
        (6, 85),
        (7, 86),
        (8, 87),
        (9, 88),
        (10, 89),
        (11, 90),
    )
    assert groups[-5:] == ((45, 50), (46, 51), (47, 52), (48, 53), (49, 54))


def test_configured_merges_assign_all_merged_sources_to_one_target() -> None:
    mapped = map_category_sources_to_configured(
        np.arange(EXCLUSIVE_SOURCE_BIN_COUNT, dtype=int), MERGED_CONFIGURATION
    )
    assert mapped[65] == mapped[66] == 16
    for parent, sparse in zip(range(6, 12), range(85, 91)):
        assert mapped[parent] == mapped[sparse]
    for left, right in ((45, 50), (46, 51), (47, 52), (48, 53), (49, 54)):
        assert mapped[left] == mapped[right]


def test_existing_85_bins_project_exactly_into_the_requested_79_bins() -> None:
    previous_merges = [
        [17, 18],
        [82, 87],
        [83, 88],
        [84, 89],
        [85, 90],
        [86, 91],
    ]
    projection = configured_projection_groups(
        MERGED_CONFIGURATION,
        input_bin_count=85,
        input_bin_merges_1based=previous_merges,
    )
    assert len(projection) == 79
    assert projection[17:23] == (
        (17, 52),
        (18, 53),
        (19, 54),
        (20, 55),
        (21, 56),
        (22, 57),
    )
    assert sorted(index for group in projection for index in group) == list(range(85))
