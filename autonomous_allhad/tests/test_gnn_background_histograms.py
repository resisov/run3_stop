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
