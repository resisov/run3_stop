from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "workflow"))
import build_combine_inputs as card
import gnn_background_histograms as gnn


@pytest.fixture
def inputs(monkeypatch):
    config = json.loads((PROJECT / "gnn_lowdm/config.json").read_text())
    histograms = {"nominal": {}}
    for region in ("SR", "LLCR", "QCDCR", "GCR"):
        labels = config["sr_binning" if region == "SR" else "cr_binning"]["category_labels"]
        histograms["nominal"][region] = {}
        for category in labels:
            samples = {}
            for sample in gnn.SAMPLES - ({"data_obs"} if region == "SR" else set()):
                joint = np.ones((5, 8))
                samples[sample] = {
                    "gnn_score": {field: joint.sum(axis=1).tolist() for field in gnn.FIELDS},
                    "gnn_score_ut": {field: joint.tolist() for field in gnn.FIELDS},
                }
            histograms["nominal"][region][category] = samples
    shapes = [0.8, 0.9, 1.0, 1.1, 1.2]
    sgamma = {"lowdm_families": {
        group: {"Q": {"status": "complete", "value": 3.0},
                "bins": [{"Sgamma": {"status": "complete", "value": value}} for value in shapes]}
        for group in ("Nb1", "Nb2plus")
    }}
    covariance = {
        "categories": ["lowdm_Nb1", "lowdm_Nb2plus"], "central": [2.0, 2.0],
        "cholesky_log_r": [[0.1, 0.0], [0.0, 0.2]],
        "nuisances": [{"name": "rz_Nb1"}, {"name": "rz_Nb2plus"}],
    }
    edges = [250, 300, 350, 400, 500, 1500]
    ratio = {"lowdm": {"bins": [
        {"status": "complete", "low": low, "high": high,
         "double_ratio": 1.1, "systematic": 0.9}
        for low, high in zip(edges[:-1], edges[1:])
    ]}}
    mapping = {
        "histograms": histograms, "ut_edges": [250, 300, 350, 400, 500, 650, 800, 1000, 1500],
        "sr_binning": config["sr_binning"], "cr_binning": config["cr_binning"],
        "zinv_projection": {label: {"rz_sgamma": [2 * (sum(shapes[:4]) + 4 * shapes[4])] * 5}
                            for label in config["sr_binning"]["category_labels"]},
    }
    monkeypatch.setattr(gnn, "require_gnn_mapping", lambda source: source["lowdm_gnn"])
    monkeypatch.setattr(card, "CAMPAIGN_YEAR", "2024")
    return {"lowdm_gnn": mapping}, sgamma, covariance, ratio


def test_frozen_channels_and_parameter_sharing(inputs):
    channels, bins = card.build_gnn_channels(*inputs)
    assert len(channels) == 90 and len(bins) == 30
    assert {c["region"] for c in channels} == {"SR", "LLCR", "QCDCR", "GCR"}
    assert all(c["observation"] is None for c in channels if c["region"] == "SR")
    parameters = {p for c in channels for p in c["rate_params"].values()}
    assert len(parameters) == 18
    assert sum("sgamma_shape" in p for p in parameters) == 10
    assert not any("zinv_norm" in p for p in parameters)
    for c in channels:
        if c["region"] in ("SR", "LLCR"):
            assert c["rate_params"]["Top"] == c["rate_params"]["WtoLNu"]
    _, dropped, _, empty = card.finalize_channels(channels, {"lowdm": bins})
    assert dropped == [] and empty == []


def test_qgamma_is_not_transferred_to_z(inputs):
    first, _ = card.build_gnn_channels(*inputs)
    changed = copy.deepcopy(inputs)
    for group in changed[1]["lowdm_families"].values():
        group["Q"]["value"] *= 2
    second, _ = card.build_gnn_channels(*changed)
    for before, after in zip(first, second):
        for process, record in before["backgrounds"].items():
            scale = 2 if before["region"] == "GCR" and process.startswith("PhotonJet_") else 1
            np.testing.assert_allclose(after["backgrounds"][process]["nominal"], record["nominal"] * scale)


def test_double_ratio_uses_central_deviation_and_shared_ut(inputs):
    channels, _ = card.build_gnn_channels(*inputs)
    seen = set()
    for c in channels:
        for process, extras in c["extra_lnN"].items():
            assert c["region"] == "SR" and process.startswith("Zto2Nu_")
            for extra in extras:
                if "zgammaNonclosure" in extra["name"]:
                    assert extra["up"] == pytest.approx(1.1)
                    assert extra["down"] == pytest.approx(1 / 1.1)
                    seen.add(extra["name"])
    assert len(seen) == 5
    assert any("u500toInf" in name for name in seen)


def test_shape_variations_use_same_projection(inputs):
    source, *rest = inputs
    hist = source["lowdm_gnn"]["histograms"]
    for name, scale in (("pileupUp", 1.1), ("pileupDown", 0.9)):
        hist[name] = copy.deepcopy(hist["nominal"])
        for categories in hist[name].values():
            for samples in categories.values():
                for sample, record in samples.items():
                    if sample == "data_obs":
                        continue
                    for axis in record.values():
                        axis["sumw"] = (np.asarray(axis["sumw"]) * scale).tolist()
    channels, _ = card.build_gnn_channels(source, *rest)
    for c in channels:
        for record in c["backgrounds"].values():
            np.testing.assert_allclose(record["variations"]["CMS_pileup_2024"]["up"], 1.1 * record["nominal"])


def test_empty_qcd_sr_bin_stays_absent(inputs):
    h = inputs[0]["lowdm_gnn"]["histograms"]["nominal"]
    for field in gnn.FIELDS:
        h["SR"]["Nb1_NISR0"]["QCD"]["gnn_score"][field][0] = 0
        h["SR"]["Nb1_NISR0"]["QCD"]["gnn_score_ut"][field][0] = [0] * 8
    channels, _ = card.build_gnn_channels(*inputs)
    selected = next(c for c in channels if c["name"] == "SR_lowdm_gnn_Nb1_NISR0_bin0")
    assert "QCD" not in selected["backgrounds"]


def test_negative_controlled_component_is_not_floored(inputs):
    inputs[0]["lowdm_gnn"]["histograms"]["nominal"]["SR"]["Nb1_NISR0"]["QCD"]["gnn_score"]["sumw"][0] = -1
    with pytest.raises(ValueError, match="signed GNN controlled"):
        card.build_gnn_channels(*inputs)


def test_negative_minor_cr_component_is_audited(inputs):
    inputs[0]["lowdm_gnn"]["histograms"]["nominal"]["LLCR"]["Nb1_NISR0"]["Zto2Nu"]["gnn_score"]["sumw"][0] = -0.1
    channels, _ = card.build_gnn_channels(*inputs)
    selected = next(c for c in channels if c["name"] == "LLCR_lowdm_gnn_Nb1_NISR0_bin0")
    assert selected["negative_mc_treatment"][0]["input_sumw"] == -0.1
    assert selected["backgrounds"]["Zto2Nu"]["nominal"][0] == card.MIN_BIN


def test_signal_reader_and_mass_union(tmp_path):
    config = json.loads((PROJECT / "gnn_lowdm/config.json").read_text())
    leaf = {"gnn_score": {field: [1, 2, 3, 4, 5] for field in gnn.FIELDS}}
    sample = "T2tt_mStop1200_mLSP500"
    payload = {"histograms": {"nominal": {"SR": {
        "Nb1_NISR0": {sample: leaf, "data_obs": {"never_decode": True}},
    }}}}
    path = tmp_path / "hists.json"
    path.write_text(json.dumps(payload, indent=2))
    result = card.gnn_signal_histograms(path, "T2tt", config, ["mStop1200_mLSP500"])
    assert set(result) == {sample}
    assert result[sample]["nominal"]["sumw"] == [1, 2, 3, 4, 5] + [0] * 25
    hists = {"search_bin_histograms": {card.GNN_SCHEME: result}}
    assert card.mass_points(hists, None, 1800, "T2tt") == ["mStop1200_mLSP500"]
    assert card.signal_leaf(hists, "lowdm_gnn", "mStop1200_mLSP500", "nominal", "T2tt")[0].sum() == 15
