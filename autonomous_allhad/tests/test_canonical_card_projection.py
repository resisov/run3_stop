from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT / "workflow"
sys.path.insert(0, str(WORKFLOW))
MODULE_PATH = WORKFLOW / "build_combine_inputs.py"
SPEC = importlib.util.spec_from_file_location("canonical_card_projection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def leaf(value: float) -> dict:
    return {
        "nominal": {
            "sumw": [value],
            "sumw2": [value],
            "entries": [1],
        }
    }


def test_card_projection_keeps_only_requested_topology_and_backgrounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hists.json"
    samples = {
        "data_obs": leaf(10.0),
        "TT": leaf(4.0),
        "T2tt_mStop1200_mLSP500": leaf(1.0),
        "T2bW_mStop1200_mLSP500": leaf(2.0),
    }
    payload = {
        "search_bin_histograms": {
            "highdm_search_bins": samples,
        },
        "highdm_control_components": {
            "LLCR": {"Nb1": samples},
        },
        "highdm_search_bin_components": {
            "highdm_search_bins": {
                "Nb1": {"recoil0": samples},
            }
        },
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    scheme = MODULE.extract_search_scheme(
        path,
        "highdm_search_bins",
        "T2tt",
    )
    assert set(scheme) == {"data_obs", "TT", "T2tt_mStop1200_mLSP500"}

    controls = MODULE.extract_component_tree(
        path,
        "highdm_control_components",
        levels_before_samples=2,
    )
    assert set(controls["LLCR"]["Nb1"]) == {"data_obs", "TT"}

    search = MODULE.extract_component_tree(
        path,
        "highdm_search_bin_components",
        levels_before_samples=2,
        member="highdm_search_bins",
    )
    assert set(search["Nb1"]["recoil0"]) == {"data_obs", "TT"}


def test_configured_highdm_projection_merges_every_bounded_component() -> None:
    configuration = json.loads(
        (PROJECT / "configs" / "search_bins_2024.json").read_text()
    )
    values = list(range(1, 80))
    source_leaf = {
        "sumw": values,
        "sumw2": [value * value for value in values],
        "entries": [1] * 79,
    }
    hists = {
        "search_bin_histograms": {
            MODULE.HIGH_SCHEME: {
                "T2tt_mStop1200_mLSP500": {
                    "nominal": source_leaf,
                    "pileupUp": source_leaf,
                }
            }
        }
    }
    exact = {
        "highdm": {
            "search_bin_labels": [f"bin{index}" for index in range(1, 80)],
            "sr_components": {
                "Nb3plus": {
                    "recoil0": {"TT": {"nominal": source_leaf}}
                }
            },
        }
    }

    summary = MODULE.apply_configured_highdm_bin_merges(
        hists, exact, configuration
    )

    assert summary["source_bin_count"] == 79
    assert summary["final_bin_count"] == 73
    assert summary["position_groups_zero_based"][16] == [16, 17]
    assert summary["position_groups_zero_based"][-5:] == [
        [69, 74],
        [70, 75],
        [71, 76],
        [72, 77],
        [73, 78],
    ]
    signal = hists["search_bin_histograms"][MODULE.HIGH_SCHEME][
        "T2tt_mStop1200_mLSP500"
    ]
    assert len(signal["nominal"]["sumw"]) == 73
    assert signal["nominal"]["sumw"][16] == 17 + 18
    assert signal["pileupUp"]["sumw2"][-1] == 74 * 74 + 79 * 79
    component = exact["highdm"]["sr_components"]["Nb3plus"]["recoil0"][
        "TT"
    ]["nominal"]
    assert component["entries"][16] == 2
    assert sum(component["sumw"]) == sum(values)
    assert "bin17__plus__bin18" in exact["highdm"]["search_bin_labels"]


def complete(value: float) -> dict:
    return {"status": "complete", "value": value}


def background(value: float) -> dict:
    return {
        "nominal": np.asarray([value]),
        "sumw2": np.asarray([value]),
        "variations": {},
    }


def test_high_effective_sgamma_is_qgamma_photon_weighted(monkeypatch) -> None:
    sgamma = {
        "highdm": {
            "Nb2": {"Q": complete(2.0), "bins": [{"Sgamma": complete(0.5)}]},
            "Nb3plus": {
                "Q": complete(3.0),
                "bins": [{"Sgamma": complete(1.5)}],
            },
        }
    }
    sources = {"Nb2": {"tag": 10.0}, "Nb3plus": {"tag": 20.0}}

    def fake_one_bin(source, process, source_bin, nbin):
        assert process == "PhotonJet"
        assert source_bin == 0
        assert nbin == 1
        return background(source["tag"])

    monkeypatch.setattr(MODULE, "one_bin_background", fake_one_bin)
    result = MODULE.high_effective_sgamma(sgamma, sources, "Nb2plus", 0, 1)
    expected = (2.0 * 10.0 * 0.5 + 3.0 * 20.0 * 1.5) / (
        2.0 * 10.0 + 3.0 * 20.0
    )
    assert result == pytest.approx(expected)


def test_incomplete_sgamma_is_a_hard_failure() -> None:
    sgamma = {
        "highdm": {
            "Nb1": {
                "Q": complete(1.0),
                "bins": [{"Sgamma": {"status": "incomplete", "value": 1.0}}],
            }
        }
    }
    with pytest.raises(ValueError, match="Sgamma/highdm/Nb1/bin0 is not complete"):
        MODULE.high_sgamma(sgamma, "Nb1", 0)


def test_analysis_specific_nuisances_use_nps26012_prefix() -> None:
    MODULE.CAMPAIGN_YEAR = "2025"
    assert MODULE.nps_nuisance_name("met_trigger").startswith("CMS_NPS26012_")
    covariance = {
        "categories": ["highdm_Nb1"],
        "central": [1.0],
        "cholesky_log_r": [[0.1]],
        "nuisances": [{"name": "CMS_NPS26012_RZstat_highdm_Nb1_2025"}],
    }
    assert MODULE.rz_nuisances(covariance, "highdm_Nb1")[0]["name"].startswith(
        "CMS_NPS26012_"
    )


def test_control_rate_parameters_are_year_specific() -> None:
    MODULE.CAMPAIGN_YEAR = "2024"
    left = MODULE.rate_parameter("sgamma_shape", "highdm", "Nb1", 0)
    MODULE.CAMPAIGN_YEAR = "2025"
    right = MODULE.rate_parameter("sgamma_shape", "highdm", "Nb1", 0)
    assert left == "sgamma_shape_highdm_Nb1_bin0_2024"
    assert right == "sgamma_shape_highdm_Nb1_bin0_2025"
    assert left != right


def test_unavailable_lowdm_sgamma_is_pooled_with_adjacent_bin() -> None:
    MODULE.CAMPAIGN_YEAR = "2024"
    family = "Nb2plus_PISR500plus_PTb140plus_Nj7plus"
    labels = [f"{family}_recoil_{index}" for index in (1, 2, 3)]
    sgamma = {
        "lowdm_families": {
            family: {
                "Q": complete(2.0),
                "bins": [
                    {
                        "Sgamma": {
                            "status": "unavailable",
                            "value": None,
                            "numerator": -0.8,
                            "denominator": 0.4,
                        }
                    },
                    {
                        "Sgamma": {
                            "status": "complete",
                            "value": 2.5,
                            "numerator": 1.0,
                            "denominator": 0.4,
                        }
                    },
                    {
                        "Sgamma": {
                            "status": "complete",
                            "value": 1.0,
                            "numerator": 0.7,
                            "denominator": 0.7,
                        }
                    },
                ],
            }
        }
    }
    models = MODULE.low_sgamma_models(sgamma, labels)
    assert models[0]["sgamma"] == pytest.approx(0.25)
    assert models[1]["sgamma"] == pytest.approx(0.25)
    assert models[0]["parameter"] == models[1]["parameter"]
    assert models[0]["pool_source_bins_zero_based"] == [0, 1]
    assert models[2]["sgamma"] == pytest.approx(1.0)
    assert models[2]["parameter"] != models[0]["parameter"]
