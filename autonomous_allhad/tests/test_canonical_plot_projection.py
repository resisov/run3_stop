from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "workflow" / "plot_control_search_bins_style.py"
SPEC = importlib.util.spec_from_file_location("canonical_plot_projection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(value: float) -> dict:
    leaf = {"sumw": [value], "sumw2": [value], "entries": [1]}
    return {
        "nominal": leaf,
        "pileupUp": leaf,
        "pileupDown": leaf,
        "unplottedUp": leaf,
        "unplottedDown": leaf,
    }


def samples() -> dict:
    return {
        "data_obs": record(10.0),
        "TT": record(4.0),
        "T2tt_mStop1000_mLSP1": record(1.0),
        "T2tt_mStop1500_mLSP1": record(2.0),
    }


def test_canonical_projection_is_bounded_to_plotted_records(tmp_path: Path) -> None:
    path = tmp_path / "hists.json"
    payload = {
        "recoil_pt_bins": [250.0, 300.0],
        "lowdm_region_policy": {
            "resolved_top_veto": {"applied": True}
        },
        "lowdm_region_variables": {"SR": ["met"]},
        "lowdm_variable_specs": {"met": {"bins": [0.0, 100.0]}},
        "highdm_distribution_regions": {"signal_categories": ["SR_test"]},
        "highdm_distribution_variable_specs": {
            "met": {"bins": [0.0, 100.0]}
        },
        "search_bin_schemes": {
            "highdm_search_bins": {"bin_labels": ["test__recoil_250to300"]},
            "cat7_SR_lowDeltaM": {"bin_labels": ["lowdm"]},
        },
        "histograms": {"SR": samples()},
        "highdm_control_components": {},
        "search_bin_histograms": {
            "highdm_search_bins": samples(),
            "cat7_SR_lowDeltaM": samples(),
        },
        "highdm_search_bin_components": {},
        "lowdm_variable_histograms": {
            "cat7_SR_lowDeltaM": {"met": samples()}
        },
        "highdm_variable_histograms": {"SR_test": {"met": samples()}},
        "normalization": "norm.json",
        "summary": {},
        "status": "complete",
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    projected = MODULE.load_canonical_plot_payload(path)

    high = projected["search_bin_histograms"]["highdm_search_bins"]
    assert set(high) == {"data_obs", "TT", "T2tt_mStop1000_mLSP1"}
    assert set(high["TT"]) == {"nominal", "pileupUp", "pileupDown"}
    assert set(high["T2tt_mStop1000_mLSP1"]) == {"nominal"}
    high_variables = projected["highdm_variable_histograms"]["SR_test"]["met"]
    assert set(high_variables) == {"data_obs", "TT"}
    assert projected["lowdm_region_policy"]["resolved_top_veto"]["applied"]


def test_webpage_publishes_machine_derived_limit_and_impact_pairs(
    tmp_path: Path,
) -> None:
    summary_2024 = tmp_path / "summary_2024.json"
    summary_2025 = tmp_path / "summary_2025.json"
    summary_2024.write_text(json.dumps({"year": "2024", "plots": []}))
    summary_2025.write_text(json.dumps({"year": "2025", "plots": []}))
    assets = {}
    for name in ("limit.png", "limit.pdf", "limit.json", "impact.png", "impact.pdf", "impact.json"):
        path = tmp_path / name
        path.write_bytes(b"test")
        assets[name] = str(path)
    manifest = tmp_path / "results.json"
    manifest.write_text(json.dumps({
        "status": "complete",
        "results": [
            {
                "year": "2024",
                "kind": "Limit",
                "title": "2024 T2tt expected limit",
                "slug": "t2tt_limit",
                "png": assets["limit.png"],
                "pdf": assets["limit.pdf"],
                "json": assets["limit.json"],
            },
            {
                "year": "2025",
                "kind": "Impact",
                "title": "2025 T2tt impacts",
                "slug": "t2tt_impact",
                "png": assets["impact.png"],
                "pdf": assets["impact.pdf"],
                "json": assets["impact.json"],
            },
        ],
    }))

    output = tmp_path / "site"
    result = MODULE.write_highdm_distribution_webpage(
        summary_2024,
        summary_2025,
        output,
        result_manifest=manifest,
    )
    assert result["status"] == "complete"
    assert len(result["results"]) == 2
    page = (output / "index.html").read_text()
    assert "2024 T2tt expected limit" in page
    assert "2025 T2tt impacts" in page
    assert "data-kind='Limit'" in page
    assert "data-kind='Impact'" in page
