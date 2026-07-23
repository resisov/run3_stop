import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "workflow" / "build_flat_boosted_recoil_hists.py"
SPEC = importlib.util.spec_from_file_location("build_flat_boosted_recoil_hists", MODULE_PATH)
HISTS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HISTS)


def test_selected_recoil54_labels_match_adopted_order():
    labels = HISTS.selected_an17_recoil54_labels()
    assert len(labels) == 54
    assert labels[:2] == [
        "NT0_Nb1plus_T0_W0_recoil_250-300",
        "NT0_Nb1plus_T0_W0_recoil_300-350",
    ]
    assert labels[6] == "NT0_Nb1plus_T0_W1plus_recoil_250-300"
    assert labels[12] == "AN17_4_Nb1_T1plus_W0_recoil_250-300"
    assert labels[-1] == "AN17_16_Nb3plus_T2_W0_recoil_800-1500"


def test_selected_recoil54_indices_cover_nt0_and_selected_an17_blocks():
    chunk = {
        "met": np.asarray([275.0, 325.0, 375.0, 450.0, 900.0, 275.0]),
        "nb_medium": np.asarray([1, 3, 1, 2, 3, 1]),
        "nboosted_top": np.asarray([0, 0, 1, 1, 2, 0]),
        "nboosted_w": np.asarray([0, 2, 0, 1, 0, 0]),
        "nboosted_total": np.asarray([0, 2, 1, 2, 2, 0]),
    }
    sr_mask = np.asarray([True, True, True, True, True, False])
    indices = HISTS.selected_an17_recoil54_indices(chunk, len(sr_mask), sr_mask)
    assert indices.tolist() == [0, 7, 14, 33, 53, -1]
