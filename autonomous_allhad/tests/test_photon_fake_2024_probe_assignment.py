from __future__ import annotations

import awkward as ak
import numpy as np
from unittest.mock import patch

from autonomous_allhad import photon_fake_2024_worker as worker


def _arrays() -> ak.Array:
    return ak.Array(
        {
            "Photon_pt": [
                [500.0, 300.0],
                [510.0, 320.0],
                [520.0, 410.0, 290.0],
                [530.0],
                [540.0],
                [550.0],
            ],
            "Photon_eta": [
                [0.2, 0.3],
                [0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.2],
                [0.2],
                [0.2],
            ],
        }
    )


def test_probe_assignment_matches_nominal_target_multiplicity() -> None:
    arrays = _arrays()
    masks = {
        # Event 0: one target plus one anti-ID photon -> keep the target.
        # Event 1: two target photons -> reject, matching nominal GCR.
        # Event 2: no target and multiple anti-ID photons -> reject.
        # Event 3: one ordinary target -> keep the target.
        # Event 4: no target and exactly one anti-ID photon -> keep it.
        "target": ak.Array(
            [
                [True, False],
                [True, True],
                [False, False, False],
                [True],
                [False],
                [False],
            ]
        ),
        "measurement_pass": ak.Array(
            [
                [False, True],
                [False, False],
                [False, True, False],
                [False],
                [False],
                [False],
            ]
        ),
        "measurement_fail": ak.Array(
            [
                [False, False],
                [False, False],
                [True, False, False],
                [False],
                [False],
                [False],
            ]
        ),
        "application": ak.Array(
            [
                [False, False],
                [False, False],
                [False, False, True],
                [False],
                [True],
                [False],
            ]
        ),
        "plj_other": ak.Array(
            [
                [False, False],
                [False, False],
                [False, False, False],
                [False],
                [False],
                [True],
            ]
        ),
    }
    with patch.object(worker, "probe_masks", return_value=masks):
        _all_masks, selected, codes, candidates = worker._probe_assignment(arrays)

    assert ak.to_list(selected) == [
        [True, False],
        [False, False],
        [False, False, False],
        [True],
        [True],
        [True],
    ]
    assert codes.tolist() == [0, -1, -1, 0, 3, 4]
    assert candidates.tolist() == [True, False, False, True, True, True]


def test_selected_probe_values_uses_only_assigned_probe() -> None:
    arrays = _arrays()
    masks = {
        "target": ak.Array(
            [
                [True, False],
                [True, True],
                [False, False, False],
                [True],
                [False],
                [False],
            ]
        ),
        "measurement_pass": ak.Array(
            [
                [False, True],
                [False, False],
                [False, True, False],
                [False],
                [False],
                [False],
            ]
        ),
        "measurement_fail": ak.Array(
            [
                [False, False],
                [False, False],
                [True, False, False],
                [False],
                [False],
                [False],
            ]
        ),
        "application": ak.Array(
            [
                [False, False],
                [False, False],
                [False, False, True],
                [False],
                [True],
                [False],
            ]
        ),
        "plj_other": ak.Array(
            [
                [False, False],
                [False, False],
                [False, False, False],
                [False],
                [False],
                [True],
            ]
        ),
    }
    with patch.object(worker, "probe_masks", return_value=masks):
        _all_masks, selected, _codes, _candidates = worker._probe_assignment(arrays)
    pt, eta, flavour = worker._selected_probe_values(arrays, selected)

    assert np.allclose(pt[[0, 3, 4, 5]], [500.0, 530.0, 540.0, 550.0])
    assert np.allclose(eta[[0, 3, 4, 5]], [0.2, 0.2, 0.2, 0.2])
    assert np.isnan(pt[1])
    assert np.isnan(pt[2])
    assert flavour.tolist() == [0, 0, 0, 0, 0, 0]


def _bitmap(**levels: int) -> int:
    value = 0
    defaults = {
        "pt": 2,
        "eta": 2,
        "sieie": 2,
        "hoe": 2,
        "charged_iso": 2,
        "ecal_iso": 2,
        "hcal_iso": 2,
    }
    defaults.update(levels)
    for name, index in worker.VID_CUT_INDEX.items():
        value |= int(defaults[name]) << (2 * index)
    return value


def test_probe_masks_use_loose_fail_guard_band_and_plj_other() -> None:
    arrays = ak.Array(
        {
            "Photon_pt": [[300.0] for _ in range(7)],
            "Photon_eta": [[0.2] for _ in range(7)],
            "Photon_electronVeto": [[True] for _ in range(7)],
            "Photon_vidNestedWPBitmap": [
                [_bitmap()],
                [_bitmap(sieie=0)],
                [_bitmap(sieie=1)],
                [_bitmap(sieie=0, charged_iso=0)],
                [_bitmap(charged_iso=0)],
                [_bitmap(charged_iso=1)],
                [_bitmap(ecal_iso=0)],
            ],
        }
    )
    masks = worker.probe_masks(arrays)

    assert ak.to_list(masks["target"]) == [
        [True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
    ]
    assert ak.to_list(masks["measurement_pass"]) == [
        [False],
        [True],
        [False],
        [False],
        [False],
        [False],
        [False],
    ]
    assert ak.to_list(masks["measurement_fail"]) == [
        [False],
        [False],
        [False],
        [True],
        [False],
        [False],
        [False],
    ]
    assert ak.to_list(masks["application"]) == [
        [False],
        [False],
        [False],
        [False],
        [True],
        [False],
        [False],
    ]
    assert ak.to_list(masks["plj_other"]) == [
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [True],
    ]
