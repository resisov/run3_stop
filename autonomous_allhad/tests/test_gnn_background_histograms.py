from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "workflow" / "gnn_background_histograms.py"
spec = importlib.util.spec_from_file_location("gnn_background_histograms", SOURCE)
gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn)


def record(value=1.0):
    return {"gnn_score": {f: [8.0 * value] * 5 for f in gnn.FIELDS},
            "gnn_score_ut": {f: [[value] * 8 for _ in range(5)] for f in gnn.FIELDS}}


def toy():
    cfg = json.loads((SOURCE.parents[1] / "gnn_lowdm/config.json").read_text())
    regions = {}
    for region in ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M"):
        labels = cfg["sr_binning" if region == "SR" else "cr_binning"]["category_labels"]
        regions[region] = {c: {s: record() for s in gnn.SAMPLES if s != "data_obs" or region != "SR"} for c in labels}
    return cfg, {"nominal": regions}


def test_selects_backgrounds_and_never_exports_sr_observation(tmp_path):
    _, hist = toy()
    hist["nominal"]["SR"]["Nb1_NISR0"]["data_obs"] = record(999)
    hist["nominal"]["SR"]["Nb1_NISR0"]["signal_T2tt"] = record(456)
    path = tmp_path / "hists.json"
    path.write_text(json.dumps({"histograms": hist}, indent=2))
    selected = gnn.read_backgrounds(path)
    assert "data_obs" not in selected["nominal"]["SR"]["Nb1_NISR0"]
    assert "signal_T2tt" not in selected["nominal"]["SR"]["Nb1_NISR0"]
    assert selected["nominal"]["GCR"]["Nb1_NISR0"]["data_obs"] == record()


def test_gnn_coefficients_use_parent_integral_not_same_score_index():
    config, hist = toy()
    result = gnn.transfer_records(hist, config)["nominal"]["top_llcr"]
    record = result["Nb1_NISR2plus"]
    assert record["parent"] == "Nb1_NISR1plus"
    np.testing.assert_allclose(record["transfer_factor"], [0.2] * 5)
    assert record["score_edges"] != record["cr_score_edges"]
    assert np.asarray(record["covariance"])[0, 1] > 0
    np.testing.assert_allclose(np.asarray(record["transfer_factor"]) * record["denominator_total"], record["numerator"])


def test_joint_z_projection_preserves_actual_ut_mixture_and_excludes_Q():
    _, hist = toy()
    factors = {"lowdm_families": {g: {"Q": {"value": 100}, "bins": [{"Sgamma": {"value": s}} for s in (1,2,3,4,5)]} for g in ("Nb1", "Nb2plus")}}
    rz = {"rz_low": {"combined": {g: {"RZ": 0.5} for g in ("Nb1", "Nb2plus")}}}
    result = gnn.project_zinv(hist, factors, rz)["Nb1_NISR1"]
    np.testing.assert_allclose(result["rz_sgamma"], [15] * 5)
    np.testing.assert_allclose(result["shape_components"], [[1,1,1,1,4]] * 5)
    assert result["parent"] == "Nb1_NISR1plus"


def test_reader_rejects_unrecognized_serialization(tmp_path):
    _, hist = toy()
    path = tmp_path / "hists.json"
    path.write_text(json.dumps({"histograms": hist}))
    with pytest.raises(ValueError, match="indent=2"):
        gnn.read_backgrounds(path)


def uncertainty_inputs():
    config, hist = toy()
    shape = {"lowdm_families": {g: {"bins": [{"Sgamma": {"value": s}} for s in (1, 2, 3, 4, 5)]} for g in ("Nb1", "Nb2plus")}}
    rz = {"rz_low": {"combined": {g: {"RZ": 0.5} for g in ("Nb1", "Nb2plus")}}}
    mapping = {"sr_binning": config["sr_binning"], "zinv_projection": gnn.project_zinv(hist, shape, rz)}
    edges = [250, 300, 350, 400, 500, 1500]
    factors = {"adoption_status": "adopted", "lowdm": {"edges": edges, "bins": [
        {"low": low, "high": high, "double_ratio": 1.1, "downstream_central_abs_deviation": 0.1,
         "systematic": 0.9, "status": "complete"} for low, high in zip(edges, edges[1:])]}}
    return mapping, factors


def test_double_ratio_response_preserves_nominal_and_uses_central_deviation():
    mapping, factors = uncertainty_inputs()
    result = gnn.project_double_ratio_uncertainty(mapping, factors)
    assert result["axis"] == "GNN output"
    assert result["nominal_changed"] is False
    assert result["statistical_error_used_as_nuisance"] is False
    assert sum(len(node["responses"]) for node in result["categories"].values()) == 30
    for category, node in result["categories"].items():
        original = mapping["zinv_projection"][category]
        assert node["nominal"] == original["rz_sgamma"]
        assert node["score_edges"] == mapping["sr_binning"]["edges_by_category"][category]
        first, tail = node["responses"][0], node["responses"][-1]
        assert first["delta"] == 0.1 and first["reporting_band_not_used"] == 0.9
        np.testing.assert_allclose(first["up"], np.asarray(node["nominal"]) + 0.05)
        np.testing.assert_allclose(first["down"], np.asarray(node["nominal"]) + 0.5 * (1 / 1.1 - 1))
        np.testing.assert_allclose(tail["up"], np.asarray(node["nominal"]) + 1.0)
        assert tail["open_ended"] is True


@pytest.mark.parametrize("corruption", ["domain", "unapproved", "deviation"])
def test_double_ratio_response_rejects_unapproved_or_inconsistent_inputs(corruption):
    mapping, factors = uncertainty_inputs()
    if corruption == "domain":
        factors["lowdm"]["edges"][0] = 300
    elif corruption == "unapproved":
        factors["adoption_status"] = "proposal_only"
    else:
        factors["lowdm"]["bins"][0]["downstream_central_abs_deviation"] = 0.9
    with pytest.raises(ValueError):
        gnn.project_double_ratio_uncertainty(mapping, factors)


@pytest.mark.parametrize("corruption", ["absent", "ut_axis", "missing_route", "wrong_edges", "wrong_parent"])
def test_final_template_consumer_rejects_ut_or_incomplete_gnn(corruption):
    mapping, _ = uncertainty_inputs()
    config, hist = toy()
    mapping.update(status="complete", axis="GNN output", cr_binning=config["cr_binning"],
                   transfer_factors=gnn.transfer_records(hist, config))
    source = {"lowdm": {"recoil": {}}, "lowdm_gnn": mapping}
    assert gnn.require_gnn_mapping(source) is mapping
    if corruption == "absent":
        del source["lowdm_gnn"]
    elif corruption == "ut_axis":
        mapping["axis"] = "U_T"
    elif corruption == "missing_route":
        del mapping["transfer_factors"]["nominal"]["zinv_gcr"]
    elif corruption == "wrong_edges":
        mapping["transfer_factors"]["nominal"]["top_llcr"]["Nb1_NISR0"]["score_edges"] = [0, 1]
    else:
        mapping["transfer_factors"]["nominal"]["top_llcr"]["Nb1_NISR0"]["parent"] = "Nb1_NISR1plus"
    with pytest.raises(ValueError):
        gnn.require_gnn_mapping(source)


def test_missing_minor_gnn_background_is_not_silently_zero():
    config, hist = toy()
    del hist["nominal"]["SR"]["Nb1_NISR0"]["VV"]
    with pytest.raises(ValueError, match="missing GNN process templates"):
        gnn.validate(hist, config, {})
