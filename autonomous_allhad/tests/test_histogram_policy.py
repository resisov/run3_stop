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


def test_configured_highdm_search_bins_assign_and_omit_exclusively():
    configuration = json.loads(
        (Path(__file__).parents[1] / "configs" / "search_bins_2024.json")
        .read_text()
    )
    chunk = {
        "feature_SR": np.ones(6, dtype=bool),
        "lowdm_mtb": np.asarray([200.0, 200.0, 200.0, 200.0, 200.0, 100.0]),
        "met": np.full(6, 275.0),
        "nb_medium": np.ones(6, dtype=int),
        "nboosted_top": np.asarray([0, 0, 0, 1, 1, 0]),
        "nboosted_w": np.asarray([0, 0, 0, 0, 1, 0]),
        "nboosted_total": np.asarray([0, 0, 0, 1, 2, 0]),
        HISTS.DERIVED_NRES_BRANCH: np.asarray([0, 1, 2, 1, 1, 1]),
    }
    indices, population, omitted = HISTS.configured_highdm_search_indices(
        chunk, 6, configuration
    )
    assert indices.tolist() == [0, 6, 12, 29, 35, -1]
    assert population.tolist() == [True, True, True, True, True, False]
    assert omitted.tolist() == [False, False, False, False, False, False]


def test_nb2_w2_nres0_is_a_dedicated_six_bin_category():
    configuration = json.loads(
        (Path(__file__).parents[1] / "configs" / "search_bins_2024.json")
        .read_text()
    )
    chunk = {
        "feature_SR": np.ones(6, dtype=bool),
        "lowdm_mtb": np.full(6, 200.0),
        "met": np.asarray([275.0, 325.0, 375.0, 450.0, 650.0, 1200.0]),
        "nb_medium": np.full(6, 2, dtype=int),
        "nboosted_top": np.zeros(6, dtype=int),
        "nboosted_w": np.full(6, 2, dtype=int),
        "nboosted_total": np.full(6, 2, dtype=int),
        HISTS.DERIVED_NRES_BRANCH: np.zeros(6, dtype=int),
    }
    indices, population, omitted = HISTS.configured_highdm_search_indices(
        chunk, 6, configuration
    )
    assert indices.tolist() == [52, 53, 54, 55, 56, 57]
    assert np.all(population)
    assert not np.any(omitted)


def test_top_w_resolved_category_includes_all_nb_and_object_multiplicities():
    configuration = json.loads(
        (Path(__file__).parents[1] / "configs" / "search_bins_2024.json")
        .read_text()
    )
    chunk = {
        "feature_SR": np.ones(3, dtype=bool),
        "lowdm_mtb": np.full(3, 200.0),
        "met": np.asarray([275.0, 325.0, 1200.0]),
        "nb_medium": np.asarray([1, 2, 3]),
        "nboosted_top": np.asarray([1, 2, 2]),
        "nboosted_w": np.asarray([1, 1, 2]),
        "nboosted_total": np.asarray([2, 3, 4]),
        HISTS.DERIVED_NRES_BRANCH: np.asarray([1, 1, 2]),
    }
    indices, population, omitted = HISTS.configured_highdm_search_indices(
        chunk, 3, configuration
    )
    assert indices.tolist() == [35, 36, 40]
    assert np.all(population)
    assert not np.any(omitted)


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
