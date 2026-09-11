import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _lowdm_cr_weight_fixture(monkeypatch):
    n = 4
    chunk = {
        "dataset_id": np.ones(n, dtype=int),
        "nb_medium": np.asarray([1, 1, 2, 2]),
        "nb_photon_clean": np.asarray([1, 1, 2, 2]),
        "nb_lepton_clean": np.asarray([1, 1, 2, 2]),
        "mee": np.asarray([91., 120., 91., 120.]),
        "mmm": np.asarray([91., 120., 91., 120.]),
        "pass_dy2e_open_high": np.ones(n, dtype=bool),
        "pass_dy2m_open_high": np.ones(n, dtype=bool),
    }
    for _, branch in HISTS.BASE_REGION_VARIABLES.values():
        chunk[branch] = np.full(n, 275.)
    block = SimpleNamespace(
        core=np.ones(n, dtype=bool), nb=chunk["nb_medium"],
        nt=np.zeros(n, dtype=int), nw=np.zeros(n, dtype=int),
        njet=np.full(n, 5), recoil=np.full(n, 275.),
    )
    blocks = dict.fromkeys(HISTS.BACKGROUND_ESTIMATION_REGIONS, block)
    monkeypatch.setattr(HISTS, "region_mask", lambda *args: np.ones(n, dtype=bool))
    monkeypatch.setattr(HISTS, "lowdm_nres_zero_mask", lambda *args: np.ones(n, dtype=bool))
    variations = {"nominal": np.full(n, 2.), "topwUp": np.full(n, 2.5)}
    regional = {
        region: {"nominal": np.full(n, i), "topwUp": np.full(n, i + .5)}
        for region, i in (("GCR", 3.), ("DY2E", 4.), ("DY2M", 5.))
    }
    return chunk, blocks, variations, regional


def test_lowdm_cr_weights_preserve_highdm_and_other_regions(monkeypatch):
    chunk, blocks, variations, regional = _lowdm_cr_weight_fixture(monkeypatch)
    outputs = []
    for weights in (None, regional):
        output = HISTS.empty_background_estimation_inputs()
        HISTS.fill_background_estimation_histograms(
            chunk, variations, np.full(4, .5), "TT", "TT", "TT",
            False, False, output, blocks, blocks, weights,
        )
        outputs.append(output)
    before, after = outputs
    assert before["highdm"] == after["highdm"]
    assert before["dy_rz"]["highdm"] == after["dy_rz"]["highdm"]
    for region in ("SR", "LLCR", "QCDCR"):
        assert before["lowdm"]["recoil"][region] == after["lowdm"]["recoil"][region]
    for region, weights in regional.items():
        for group in ("Nb1", "Nb2plus"):
            for variation, weight in weights.items():
                leaf = after["lowdm"]["recoil"][region][group]["TT"][variation]
                assert sum(leaf["entries"]) == 2
                assert sum(leaf["sumw"]) == weight[0]
                assert sum(leaf["sumw2"]) == 2 * (weight[0] * .5) ** 2
    for channel in ("DY2E", "DY2M"):
        for group in ("Nb1", "Nb2plus"):
            expected = regional[channel]["nominal"][0] * .5
            for window in ("on", "off"):
                leaf = after["dy_rz"]["lowdm"]["yields"][channel][group][window]["other"]["nominal"]
                assert sum(leaf["sumw"]) == expected
                assert sum(leaf["sumw2"]) == expected ** 2
                assert sum(leaf["entries"]) == 1
            leaf = after["dy_rz"]["lowdm"]["mll"][channel][group]["other"]["nominal"]
            assert sum(leaf["sumw"]) == 2 * expected
            assert sum(leaf["sumw2"]) == 2 * expected ** 2


def test_lowdm_cr_distribution_uses_same_weights_as_estimator(monkeypatch):
    chunk, blocks, variations, regional = _lowdm_cr_weight_fixture(monkeypatch)
    monkeypatch.setattr(HISTS, "LOWDM_REGION_VARIABLES", {
        region: ["met"] for region in HISTS.BACKGROUND_ESTIMATION_REGIONS
    })
    monkeypatch.setattr(HISTS, "broad_lowdm_variable_values", lambda *args: np.full(4, 275.))
    histograms = {}
    HISTS.fill_broad_lowdm_distribution_histograms(
        chunk, variations, np.full(4, .5), "TT", "TT", False,
        histograms, {}, blocks=blocks, object_audit={}, region_variations=regional,
    )
    for region, channel in HISTS.LOWDM_REGION_MAP.items():
        for variation, weight in regional.get(region, variations).items():
            leaf = histograms[channel]["met"]["TT"][variation]
            assert sum(leaf["entries"]) == 4
            assert sum(leaf["sumw"]) == 2 * weight[0]
            assert sum(leaf["sumw2"]) == weight[0] ** 2


def test_lowdm_cr_reuses_gnn_cleaning_without_double_weighting(monkeypatch):
    import awkward as ak
    from autonomous_allhad import analysis_scale_factors as sf
    from gnn_lowdm._implementation import region_io

    assert HISTS.broad_lowdm_topw_weights is region_io.topw_region_weight_variations
    arrays = {"year": np.asarray([2024]),
              "fatjet_eta_all": ak.Array([[0., 2.]]),
              "fatjet_phi_all": ak.Array([[0., 0.]])}
    for obj in ("photon", "electron", "muon"):
        arrays[obj + "_eta_all"] = ak.Array([[0.]])
        arrays[obj + "_phi_all"] = ak.Array([[0.]])
    monkeypatch.setattr(region_io, "object_masks", lambda arrays: {
        obj + "_medium": ak.Array([[True]]) for obj in ("photon", "electron", "muon")
    })
    masks = []
    def apply(weights, status, *args, cleaned=None):
        masks.append(ak.to_list(cleaned))
        status["topw_correction"] = {"mode": "available"}
        return {name: values * 3. for name, values in weights.items()}
    monkeypatch.setattr(sf, "apply_topw_event_weights", apply)
    weights = {"nominal": np.asarray([2.])}
    record = {"is_data": False, "dataset": "TT", "process": "TT"}
    manifest = {"analysis_sf_components": ["topw_tagging"]}
    for region in ("GCR", "DY2E", "DY2M"):
        result = HISTS.broad_lowdm_topw_weights(
            weights, {}, arrays, region, record, manifest, Path("."), {"mode": "available"},
        )
        assert result["nominal"].tolist() == [6.]
        assert weights["nominal"].tolist() == [2.]
    assert masks == [[[False, True]]] * 3
    assert HISTS.broad_lowdm_topw_weights(
        weights, {}, arrays, "GCR", dict(record, is_data=True), manifest, Path("."), None,
    ) is weights


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
