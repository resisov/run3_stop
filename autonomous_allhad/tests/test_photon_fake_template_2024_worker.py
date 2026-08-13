from __future__ import annotations

import awkward as ak

from autonomous_allhad import photon_fake_2024_worker as common
from autonomous_allhad import photon_fake_template_2024_worker as worker


def _bitmap(**levels: int) -> int:
    configured = {
        "pt": 2,
        "eta": 2,
        "sieie": 2,
        "hoe": 2,
        "charged_iso": 2,
        "ecal_iso": 2,
        "hcal_iso": 2,
    }
    configured.update(levels)
    value = 0
    for name, index in common.VID_CUT_INDEX.items():
        value |= int(configured[name]) << (2 * index)
    return value


def _arrays() -> ak.Array:
    return ak.Array(
        {
            "Photon_pt": [
                [500.0, 300.0],
                [510.0, 320.0],
                [520.0, 410.0],
                [530.0],
            ],
            "Photon_eta": [[0.2, 0.3], [0.2, 0.3], [0.2, 0.3], [0.2]],
            "Photon_electronVeto": [[True, True], [True, True], [True, True], [True]],
            "Photon_vidNestedWPBitmap": [
                [_bitmap(), _bitmap(charged_iso=1)],
                [_bitmap(), _bitmap()],
                [_bitmap(sieie=1), _bitmap(charged_iso=1)],
                [_bitmap(sieie=1, charged_iso=1)],
            ],
            "Photon_sieie": [[0.009, 0.012], [0.009, 0.010], [0.014, 0.012], [0.013]],
            "Photon_pfRelIso03_chg_quadratic": [[0.01, 0.08], [0.01, 0.02], [0.03, 0.08], [0.07]],
        }
    )


def test_assignment_preserves_nominal_target_and_rejects_ambiguity() -> None:
    selected, has_selected = worker.template_probe_assignment(_arrays())
    assert ak.to_list(selected) == [
        [True, False],
        [False, False],
        [False, False],
        [True],
    ]
    assert has_selected.tolist() == [True, False, False, True]


def test_continuous_template_observables_follow_selected_photon() -> None:
    arrays = _arrays()
    selected, _ = worker.template_probe_assignment(arrays)
    values = worker.selected_probe_observables(arrays, selected, is_data=True)
    assert values["pt"][[0, 3]].tolist() == [500.0, 530.0]
    assert values["sieie"][[0, 3]].tolist() == [0.009, 0.013]
    assert values["shape_level"][[0, 3]].tolist() == [2, 1]
    assert values["charged_iso_level"][[0, 3]].tolist() == [2, 1]
    assert worker.photon_category(2, 2) == "tight"
    assert worker.photon_category(2, 1) == "loose_charged_iso"
    assert worker.photon_category(0, 0) == "fail_loose_charged_iso"
