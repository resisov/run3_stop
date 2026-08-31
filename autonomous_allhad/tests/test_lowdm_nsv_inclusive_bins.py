import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).parents[1]
HIST_MODULE_PATH = PROJECT / "workflow" / "build_flat_boosted_recoil_hists.py"
PLOT_MODULE_PATH = PROJECT / "workflow" / "plot_control_search_bins_style.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HISTS = load_module("build_flat_boosted_recoil_hists_lowdm_test", HIST_MODULE_PATH)
PLOTS = load_module("plot_control_search_bins_style_lowdm_test", PLOT_MODULE_PATH)


def test_lowdm_search_bins_are_nsv_independent_and_span_42_bins():
    cases = [
        ((2, 0, 500.0, 35.0, 450.0, 0.0), 0),
        ((6, 0, 500.0, 35.0, 760.0, 0.0), 7),
        ((2, 1, 350.0, 35.0, 300.0, 100.0), 8),
        ((2, 1, 350.0, 50.0, 650.0, 100.0), 15),
        ((2, 1, 550.0, 35.0, 450.0, 100.0), 16),
        ((2, 1, 550.0, 50.0, 800.0, 100.0), 23),
        ((2, 2, 350.0, 50.0, 300.0, 100.0), 24),
        ((2, 2, 350.0, 100.0, 450.0, 100.0), 28),
        ((7, 2, 350.0, 150.0, 600.0, 100.0), 32),
        ((2, 2, 550.0, 50.0, 450.0, 100.0), 33),
        ((2, 2, 550.0, 100.0, 600.0, 100.0), 37),
        ((7, 2, 550.0, 150.0, 700.0, 100.0), 41),
    ]
    for (njet, nb, pisr, ptb, met, mtb), expected in cases:
        for nsv in (-1, 0, 1, 4):
            assert HISTS.assign_lowdm_search_bin(njet, nb, nsv, pisr, ptb, met, mtb) == expected
    assert len(HISTS.LOWDM_42BIN_LABELS) == 42
    assert HISTS.assign_lowdm_search_bin(2, 1, 0, 550.0, 35.0, 450.0, 250.0) == 16


def test_lowdm_sr_rebuilder_uses_broad_preselection_and_quality_flags():
    chunk = {
        "feature_lowdm_preselection": np.asarray([True, True, False]),
        "pass_lowdm_topology_veto": np.asarray([True, True, True]),
        "pass_lowdm_isr": np.asarray([True, True, True]),
        "pass_lowdm_isr_bveto": np.asarray([False, True, True]),
        "pass_lowdm_met_sqrt_ht": np.asarray([True, False, True]),
        "pass_lowdm_mtb": np.asarray([False, True, True]),
        "njet": np.asarray([2, 2, 2]),
        "nb_medium_lowdm": np.asarray([1, 1, 1]),
        "lowdm_isr_pt": np.asarray([550.0, 550.0, 550.0]),
        "lowdm_ptb": np.asarray([35.0, 35.0, 35.0]),
        "met": np.asarray([500.0, 500.0, 500.0]),
        "lowdm_mtb": np.asarray([0.0, 0.0, 0.0]),
    }
    assert HISTS.lowdm_nsv_inclusive_sr_indices(chunk, 3).tolist() == [8, -1, -1]


def test_lowdm_plot_uses_requested_mass_points_and_vivid_colors():
    zeros = [0.0] * 42
    ones = [1.0] * 42
    leaf = lambda values: {"nominal": {"sumw": values, "sumw2": values, "entries": [1] * 42}}
    payload = {
        "lowdm_region_policy": {"resolved_top_veto": {"applied": True}},
        "search_bin_schemes": {
            "cat7_SR_lowDeltaM": {
                "bin_labels": HISTS.LOWDM_42BIN_LABELS,
                "category_sizes": HISTS.LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES,
            }
        },
        "search_bin_histograms": {
            "cat7_SR_lowDeltaM": {
                "data_obs": leaf(zeros),
                "TT": leaf(ones),
                "T2tt_mStop600_mLSP400": leaf(ones),
                "T2tt_mStop900_mLSP700": leaf(ones),
            }
        },
    }
    blocks = PLOTS.lowdm_nsv_inclusive_blocks(payload, "cat7_SR_lowDeltaM")
    assert len(blocks) == 12
    assert sum(block["nbin"] for block in blocks) == 42
    assert [spec["key"] for spec in blocks[0]["signal_specs"]] == [
        "mStop600_mLSP400",
        "mStop900_mLSP700",
    ]
    assert [spec["color"] for spec in blocks[0]["signal_specs"]] == ["#54FAFD", "#FFD500"]
    assert all("N_{res}" not in block["label"] for block in blocks)


def test_highdm_plot_applies_configured_merges_and_three_line_labels():
    configuration = json.loads(
        (PROJECT / "configs" / "search_bins_2024.json").read_text()
    )
    labels = HISTS.configured_exclusive_labels(
        HISTS.selected_an17_recoil60_labels(),
        {**configuration, "bin_merges_1based": []},
    )
    values = [1.0] * len(labels)
    payload = {
        "search_bin_schemes": {
            "highdm_search_bins": {"bin_labels": labels},
        },
        "search_bin_histograms": {
            "highdm_search_bins": {
                "TT": {
                    "nominal": {
                        "sumw": values,
                        "sumw2": values,
                        "entries": values,
                    }
                }
            }
        },
    }
    summary = PLOTS.apply_configured_search_bin_merges(
        payload, "highdm_search_bins", configuration
    )
    assert summary["source_bin_count"] == 91
    assert summary["final_bin_count"] == 85
    rebinned = payload["search_bin_histograms"]["highdm_search_bins"]["TT"]["nominal"]
    assert rebinned["sumw"][16] == 2.0
    assert rebinned["sumw"][-5:] == [2.0] * 5
    blocks = PLOTS.selected_an17_recoil_blocks(payload, "highdm_search_bins")
    assert len(blocks) == 15
    assert sum(block["nbin"] for block in blocks) == 85
    assert all(block["label"].count("\n") == 2 for block in blocks)
    assert all(block["label_fontsize"] == 12.0 for block in blocks)
    assert all(block["figure_width"] == 22.0 for block in blocks)
    assert "N_{b}\\geq3" in blocks[-1]["label"]
    assert "$N_{t}=1,2$" in blocks[-1]["label"]
    assert "$N_{W}=1,0$" in blocks[-1]["label"]
    assert blocks[-1]["category_key"] == "merged_high_nt"
