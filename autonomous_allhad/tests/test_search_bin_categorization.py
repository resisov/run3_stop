from __future__ import annotations

import numpy as np

from autonomous_allhad.search_bin_categorization import (
    adopted55_mapping,
    configured_exclusive_mapping,
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
        *({"kind": "baseline", "index": index} for index in range(2, 10)),
    ],
    "omitted_topologies": ["top_w_resolved"],
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
    nt = np.asarray(([0] * 60) + ([0] * 60) + ([1] * 60) + ([0] * 60) + ([1] * 60) + ([2] * 60))
    nw = np.asarray(([0] * 60) + ([0] * 60) + ([0] * 60) + ([1] * 60) + ([1] * 60) + ([3] * 60))
    nres = np.asarray(([0] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([2] * 60))
    output = exclusive_category_source_indices(baseline55, recoil, nt, nw, nres)
    assert len(output) == len(baseline55)
    assert np.all((output >= 0) & (output < 85))
    assert np.array_equal(output[:60], baseline55[:60])


def test_labels_have_expected_sizes() -> None:
    source = [f"source_{index}" for index in range(60)]
    labels85 = exclusive_category_source_labels(source)
    assert len(labels85) == 85
    assert labels85[0].endswith("__Nres0")


def test_current_search_bins_follow_requested_category_order() -> None:
    mapping = configured_exclusive_mapping(CURRENT_CONFIGURATION)
    assert len(mapping) == 79
    assert mapping[:6] == tuple(range(6))
    assert mapping[6:12] == tuple(range(55, 61))
    assert mapping[12:18] == tuple(range(61, 67))
    assert mapping[18:24] == tuple(range(6, 12))
    assert mapping[24:30] == tuple(range(73, 79))
    assert mapping[30:36] == tuple(range(67, 73))
    assert set(mapping).isdisjoint(range(79, 85))


def test_omitted_topology_is_explicitly_unmapped() -> None:
    source = np.arange(85, dtype=int)
    mapped = map_category_sources_to_configured(source, CURRENT_CONFIGURATION)
    assert np.all(mapped[np.asarray(configured_exclusive_mapping(CURRENT_CONFIGURATION))] >= 0)
    assert np.all(mapped[79:85] == -1)
