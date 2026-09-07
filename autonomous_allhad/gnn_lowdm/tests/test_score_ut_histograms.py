from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gnn_lowdm._implementation.build_diagonal_v3_cr_nnout_partial import (
    SR_CATEGORIES, campaign_year, category_masks, fill, fill_score_ut, merge,
    sr_score_edges, validate_score_ut,
)
from gnn_lowdm._implementation.region_io import make_requests
from gnn_lowdm._implementation import cli
from gnn_lowdm._implementation import merge_diagonal_v3_cr_nnout_partials as merger


@pytest.mark.parametrize("manifest_year", [None, 2024, 2025])
def test_merger_checks_present_year_and_accepts_legacy_manifest(tmp_path, monkeypatch, manifest_year):
    partial_dir = tmp_path / "partials"
    partial_dir.mkdir()
    partial = {
        "schema_version": "gnn_lowdm_diagonal_v3_srcr_nnout_partial_v3",
        "year": 2024, "status": "complete", "kind": "mc",
        "input_files": ["test.root"], "bad_files": [],
        "score_edges": [0, .2, .4, .6, .8, 1],
        "selection_contract": {"sr_data": "blinded"},
        "checkpoint": {}, "histogram_specs": {}, "histograms": histogram(),
    }
    (partial_dir / "mc_0000.json").write_text(json.dumps(partial))
    manifest = {"status": "complete", "shards": [{"root": "test.root"}]}
    if manifest_year is not None:
        manifest["year"] = manifest_year
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "merged.json"
    monkeypatch.setattr(cli.sys, "argv", ["cr-merge", "--partials", str(partial_dir),
        "--manifest", str(manifest_path), "--output", str(output)])
    if manifest_year == 2025:
        with pytest.raises(RuntimeError, match="year does not match"):
            merger.main()
    else:
        assert merger.main() == 0
        result = json.loads(output.read_text())
        assert result["manifest_audit"]["exact_input_set_match"]
        assert result["year"] == 2024


@pytest.mark.parametrize("package", ["gnn_lowdm", "autonomous_allhad.gnn_lowdm"])
def test_dispatch_uses_the_invoking_package(monkeypatch, package):
    imported = []
    monkeypatch.setattr(cli, "__package__", package + "._implementation")
    monkeypatch.setattr(cli.sys, "argv", ["eval", "prepare-hists"])
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: (
        imported.append(name) or SimpleNamespace(prepare_histograms_main=lambda: 0)))
    assert cli.dispatch({"prepare-hists": "autonomous_allhad.gnn_lowdm._implementation.region_io:prepare_histograms_main"}, "test") == 0
    assert imported == [package + "._implementation.region_io"]


@pytest.mark.parametrize("year", [2024, 2025])
def test_campaign_year_uses_validated_trota_metadata(year):
    audits = [{"trota_provenance": {"application_year": year}}] * 2
    assert campaign_year({}, audits) == year
    assert campaign_year({"year": year}, audits) == year


@pytest.mark.parametrize("manifest,audits", [
    ({}, []),
    ({"year": 2024}, [{"trota_provenance": {"application_year": 2025}}]),
    ({"year": 2023}, []),
])
def test_missing_or_mixed_campaign_year_is_rejected(manifest, audits):
    with pytest.raises(RuntimeError, match="campaign year"):
        campaign_year(manifest, audits)


def histogram():
    score = np.array([0.0, 0.4, 0.7, 1.0, 0.95])
    recoil = np.array([250., 300., 1000., 1500., 10000.])
    weight = np.array([1., -2., 3., 4., 5.])
    selected = np.ones(5, dtype=bool)
    edges = np.linspace(0, 1, 6)
    ut = np.array([250., 300., 350., 400., 500., 650., 800., 1000., 1500.])
    result = {}
    fill(result, "nominal", "SR", "Nb1_NISR0", "Zinv", score,
         weight, selected, edges)
    fill_score_ut(result, "nominal", "SR", "Nb1_NISR0", "Zinv", score,
                  recoil, weight, selected, edges, ut)
    return result


def test_joint_projection_and_open_ended_overflow():
    result = histogram()
    validate_score_ut(result)
    record = result["nominal"]["SR"]["Nb1_NISR0"]["Zinv"]
    joint = np.asarray(record["gnn_score_ut"]["sumw"])
    assert joint.shape == (5, 8)
    assert joint[-1, -1] == 9.
    assert np.sum(record["gnn_score_ut"]["entries"]) == 5
    assert np.sum(record["gnn_score_ut"]["sumw2"]) == 55


def test_joint_merge_preserves_both_axes_and_signed_weights():
    one = histogram()
    merged = copy.deepcopy(one)
    merge(merged, one, 5)
    validate_score_ut(merged)
    for field in ("sumw", "sumw2", "entries"):
        original = one["nominal"]["SR"]["Nb1_NISR0"]["Zinv"]["gnn_score_ut"][field]
        actual = merged["nominal"]["SR"]["Nb1_NISR0"]["Zinv"]["gnn_score_ut"][field]
        np.testing.assert_allclose(actual, 2 * np.asarray(original))


def test_projection_mismatch_is_rejected():
    result = histogram()
    result["nominal"]["SR"]["Nb1_NISR0"]["Zinv"]["gnn_score_ut"]["sumw"][0][0] += 1
    with pytest.raises(RuntimeError, match="projection differs"):
        validate_score_ut(result)


def test_sr_uses_frozen_six_categories_and_thirty_bins():
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    edges = sr_score_edges(config)
    assert list(edges) == list(SR_CATEGORIES)
    assert sum(len(value) - 1 for value in edges.values()) == 30
    block = SimpleNamespace(nb=np.array([1, 1, 1, 2, 3, 2]),
                            nisr=np.array([0, 1, 2, 0, 1, 3]))
    masks = category_masks(block, "SR")
    np.testing.assert_array_equal(np.stack(list(masks.values())), np.eye(6, dtype=bool))
    assert len(category_masks(block, "GCR")) == 4


@pytest.mark.parametrize("recoil", [np.nan, 249.])
def test_invalid_joint_input_is_rejected(recoil):
    with pytest.raises(RuntimeError):
        fill_score_ut({}, "nominal", "GCR", "Nb1_NISR0", "Photon",
                      np.array([.5]), np.array([recoil]), np.array([1.]),
                      np.array([True]), np.linspace(0, 1, 6), np.array([250., 1500.]))


@pytest.mark.parametrize("status,allow_skips", [
    ("complete", False), ("complete_with_permanent_skips", True),
])
def test_histogram_requests_preserve_manifest_status(tmp_path, status, allow_skips):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": status,
        "audit": {"source_bad_files": [{"process": "Muon", "permanently_skipped": True}],
                  "source_bad_file_count": 1} if allow_skips else {},
        "shards": [{"kind": kind, "root": f"{kind}.root", "sidecar": f"{kind}.json"}
                   for kind in ("data", "mc", "signal")],
    }))
    output = tmp_path / "requests_only"
    opts = SimpleNamespace(manifest=manifest, output=output, files_per_batch=5,
                           repository=tmp_path, stop_xsec=tmp_path / "xsec.json",
                           allow_permanent_skips=allow_skips)
    assert make_requests(opts) == 0
    state = json.loads((output / "campaign_state.json").read_text())
    assert state["manifest_status"] == status
    assert state["source_bad_file_count"] == int(allow_skips)
    assert state["requests"] == 3
    assert state["source_files"] == {"data": 1, "mc": 1, "signal": 1}
    for kind in ("data", "mc", "signal"):
        request = json.loads((output / "requests" / f"{kind}_0000.json").read_text())
        assert request["inputs"] == [{"root": f"{kind}.root", "sidecar": f"{kind}.json"}]


@pytest.mark.parametrize("allow_skips,audit", [
    (False, {"source_bad_files": [{"permanently_skipped": True}]}),
    (True, {}),
    (True, {"source_bad_files": [{"permanently_skipped": True}], "missing_roots": ["missing.root"]}),
])
def test_unaccepted_incomplete_manifest_cannot_prepare_requests(tmp_path, allow_skips, audit):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "complete_with_permanent_skips", "audit": audit}))
    opts = SimpleNamespace(manifest=manifest, allow_permanent_skips=allow_skips)
    with pytest.raises(RuntimeError, match="manifest is not complete"):
        make_requests(opts)
