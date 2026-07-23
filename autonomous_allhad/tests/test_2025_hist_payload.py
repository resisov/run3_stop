import importlib.util
import math
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "workflow" / "build_2025_hist_payload.py"
SPEC = importlib.util.spec_from_file_location("build_2025_hist_payload", MODULE_PATH)
PAYLOAD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PAYLOAD)


def leaf(sumw, sumw2, entries):
    return {"sumw": sumw, "sumw2": sumw2, "entries": entries}


def test_build_payload_scales_mc_and_replaces_data():
    base = {
        "status": "complete",
        "summary": {"events_processed": 100},
        "histograms": {
            "SR": {
                "TT": {"nominal": leaf([2.0], [4.0], [3])},
                "data_obs": {"nominal": leaf([10.0], [10.0], [10])},
            }
        },
        "search_bin_histograms": {},
        "lowdm_variable_histograms": {},
    }
    data = {
        "status": "complete",
        "summary": {"events_processed": 20},
        "histograms": {"SR": {"data_obs": {"nominal": leaf([7.0], [7.0], [7])}}},
        "search_bin_histograms": {},
        "lowdm_variable_histograms": {},
    }
    out, stats = PAYLOAD.build_payload(base, data, 100.0, 110.0, "norm.json", "base.json", "data.json")
    mc_leaf = out["histograms"]["SR"]["TT"]["nominal"]
    assert math.isclose(mc_leaf["sumw"][0], 2.2)
    assert math.isclose(mc_leaf["sumw2"][0], 4.84)
    assert mc_leaf["entries"] == [3]
    assert out["histograms"]["SR"]["data_obs"]["nominal"] == leaf([7.0], [7.0], [7])
    assert out["summary"]["data_events_processed"] == 20
    assert stats["scaled_nondata_leaves"] == 1
    assert stats["replaced_data_leaves"] == 1


def test_build_payload_classifies_retired_data_scheme_separately():
    base = {
        "status": "complete",
        "summary": {},
        "histograms": {},
        "search_bin_histograms": {
            "boosted_an17_selected_recoil6_with_nt0_SR": {
                "data_obs": {"nominal": leaf([4.0], [4.0], [4])},
            },
        },
        "lowdm_variable_histograms": {},
    }
    data = {
        "status": "complete",
        "summary": {},
        "histograms": {},
        "search_bin_histograms": {},
        "lowdm_variable_histograms": {},
    }
    out, stats = PAYLOAD.build_payload(base, data, 100.0, 100.0, "norm.json", "base.json", "data.json")
    assert out["status"] == "complete"
    assert stats["missing_data_paths"] == []
    assert stats["zeroed_retired_data_leaves"] == 1
    assert stats["retired_data_paths"] == [
        "search_bin_histograms/boosted_an17_selected_recoil6_with_nt0_SR/data_obs"
    ]
