import importlib.util
import json
import sys
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


def test_selected_recoil60_adds_exact_nb2_nt2plus_w0_block():
    labels = HISTS.selected_an17_recoil60_labels()
    assert len(labels) == 60
    assert labels[:36] == HISTS.selected_an17_recoil54_labels()[:36]
    assert labels[36] == "Nb2_Nt2plus_W0_recoil_250-300"
    assert labels[41] == "Nb2_Nt2plus_W0_recoil_800-1500"
    assert labels[42:] == HISTS.selected_an17_recoil54_labels()[36:]

    chunk = {
        "met": np.asarray([275.0, 325.0, 375.0, 450.0, 550.0, 900.0, 275.0, 275.0]),
        "nb_medium": np.asarray([2, 2, 2, 2, 2, 2, 2, 3]),
        "nboosted_top": np.asarray([2, 3, 2, 4, 2, 3, 1, 1]),
        "nboosted_w": np.asarray([0, 0, 0, 0, 0, 0, 0, 0]),
        "nboosted_total": np.asarray([2, 3, 2, 4, 2, 3, 1, 1]),
    }
    sr_mask = np.asarray([True, True, True, True, True, True, True, True])
    assert HISTS.selected_an17_recoil60_indices(
        chunk, len(sr_mask), sr_mask
    ).tolist() == [36, 37, 38, 39, 40, 41, 24, 42]


def test_nominal_only_keeps_nominal_after_full_bundle_is_available():
    variations = {
        "nominal": np.asarray([1.0, 2.0]),
        "pileupUp": np.asarray([1.1, 2.2]),
        "btagDown": np.asarray([0.9, 1.8]),
    }
    selected = HISTS.histogram_variations(variations, nominal_only=True)
    assert list(selected) == ["nominal"]
    assert selected["nominal"] is variations["nominal"]
    assert HISTS.histogram_variations(variations, nominal_only=False) is variations


def test_required_normalization_rejects_missing_and_nonfinite_factors():
    chunk = {
        "dataset_id": np.asarray([17, 17]),
        "mStop": np.asarray([0, 0]),
        "mLSP": np.asarray([0, 0]),
    }
    for normalization in (
        {"dataset_factors": {}},
        {
            "dataset_factors": {
                "17": {"normalization_factor": float("nan")}
            }
        },
        {"dataset_factors": {"17": {"normalization_factor": 0.0}}},
        {"dataset_factors": {"17": {"normalization_factor": -0.25}}},
    ):
        try:
            HISTS.norm_vector(
                normalization,
                chunk,
                17,
                "DYto2L-2Jets",
                is_data=False,
                is_signal=False,
                require_normalization=True,
            )
        except RuntimeError as exc:
            assert "normalization factor" in str(exc)
        else:
            raise AssertionError("invalid normalization factor was silently accepted")

    weights = HISTS.norm_vector(
        {"dataset_factors": {"17": {"normalization_factor": 0.25}}},
        chunk,
        17,
        "DYto2L-2Jets",
        is_data=False,
        is_signal=False,
        require_normalization=True,
    )
    assert weights.tolist() == [0.25, 0.25]


def test_missing_explicit_input_root_marks_builder_incomplete(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({}))
    output = tmp_path / "output.json"
    old_argv = sys.argv
    sys.argv = [
        str(MODULE_PATH),
        "--repo",
        str(repo),
        "--inputs",
        str(tmp_path / "missing.root"),
        "--normalization",
        str(normalization),
        "--output",
        str(output),
    ]
    try:
        assert HISTS.main() == 2
    finally:
        sys.argv = old_argv
    payload = json.loads(output.read_text())
    assert payload["status"] == "complete_with_warnings"
    assert payload["summary"]["missing_input_roots"] == [
        str(tmp_path / "missing.root")
    ]
