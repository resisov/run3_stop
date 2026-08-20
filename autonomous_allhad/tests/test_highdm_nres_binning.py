from __future__ import annotations

import numpy as np

from autonomous_allhad.highdm_nres_binning import (
    adopted55_mapping,
    exclusive85_indices,
    exclusive85_labels,
    extended85_to_tailmerged80,
    map60_indices_to_adopted55,
    map85_indices_to_tailmerged80,
    tailmerged80_labels,
)


def test_adopted55_mapping_covers_precursor_once() -> None:
    mapping = adopted55_mapping()
    assert len(mapping) == 55
    assert sorted(sum((list(group) for group in mapping), [])) == list(range(60))
    assert [group for group in mapping if len(group) == 2] == [
        (22, 23), (34, 35), (40, 41), (52, 53), (58, 59),
    ]


def test_exclusive85_reassigns_every_baseline_event_once() -> None:
    baseline60 = np.arange(60, dtype=int)
    baseline55 = map60_indices_to_adopted55(baseline60)
    baseline55 = np.tile(baseline55, 6)
    recoil = np.tile(np.arange(6), 60)
    nt = np.asarray(([0] * 60) + ([0] * 60) + ([1] * 60) + ([0] * 60) + ([1] * 60) + ([2] * 60))
    nw = np.asarray(([0] * 60) + ([0] * 60) + ([0] * 60) + ([1] * 60) + ([1] * 60) + ([3] * 60))
    nres = np.asarray(([0] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([1] * 60) + ([2] * 60))
    output = exclusive85_indices(baseline55, recoil, nt, nw, nres)
    assert len(output) == len(baseline55)
    assert np.all((output >= 0) & (output < 85))
    assert np.array_equal(output[:60], baseline55[:60])


def test_tailmerged80_covers_exclusive85_once() -> None:
    mapping = extended85_to_tailmerged80()
    assert len(mapping) == 80
    assert sorted(sum((list(group) for group in mapping), [])) == list(range(85))
    mapped = map85_indices_to_tailmerged80(np.arange(85))
    assert np.all((mapped >= 0) & (mapped < 80))


def test_labels_have_expected_sizes() -> None:
    source = [f"source_{index}" for index in range(60)]
    labels85 = exclusive85_labels(source)
    labels80 = tailmerged80_labels(source)
    assert len(labels85) == 85
    assert len(labels80) == 80
    assert labels85[0].endswith("__Nres0")
    assert labels80[-1].endswith("__recoil_500to1500")
