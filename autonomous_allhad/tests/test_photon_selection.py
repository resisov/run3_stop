from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from autonomous_allhad.real_subset_worker import (  # noqa: E402
    PHOTON_MEDIUM_CUTBASED_MIN,
    medium_photon_mask,
)


def test_medium_photon_accepts_cutbased_two_with_explicit_electron_veto() -> None:
    pt = np.asarray([221.0, 221.0, 221.0, 221.0, 220.0, 221.0, 221.0])
    eta = np.asarray([0.5, 0.5, 1.5, 0.5, 0.5, 2.0, 2.5])
    cutbased = np.asarray([2, 1, 2, 2, 2, 2, 2])
    electron_veto = np.asarray([1, 1, 1, 0, 1, 1, 1])

    assert PHOTON_MEDIUM_CUTBASED_MIN == 2
    assert medium_photon_mask(pt, eta, cutbased, electron_veto).tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
