from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "workflow" / "build_combine_inputs.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "highdm_orthogonality_projection", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def leaf(nbin: int) -> dict[str, list[float]]:
    return {
        "entries": list(range(nbin)),
        "sumw": [float(value) for value in range(nbin)],
        "sumw2": [float(value * value) for value in range(nbin)],
    }


def payloads() -> tuple[dict, dict]:
    leading = [
        f"NT0_Nb1plus_T0_W0_recoil_{index}__Nres0"
        for index in range(6)
    ]
    labels = leading + [f"kept_{index}" for index in range(73)]
    projection = {
        "source_bin_count": 91,
        "input_bin_count": 79,
        "final_bin_count": 79,
    }
    hists = {
        "search_bin_histograms": {
            MODULE.HIGH_SCHEME: {"sample": {"nominal": leaf(79)}}
        }
    }
    exact = {
        "highdm": {
            "search_bin_labels": labels,
            "sr_components": {"SR": {"sample": {"nominal": leaf(79)}}},
            "bin_projection": projection,
        }
    }
    return hists, exact


def test_drop_highdm_leading_overlap_category() -> None:
    hists, exact = payloads()
    projection = MODULE.drop_highdm_leading_bins(hists, exact, 6)

    assert len(exact["highdm"]["search_bin_labels"]) == 73
    assert exact["highdm"]["search_bin_labels"][0] == "kept_0"
    record = hists["search_bin_histograms"][MODULE.HIGH_SCHEME]["sample"][
        "nominal"
    ]
    assert len(record["sumw"]) == 73
    assert record["sumw"][0] == 6.0
    assert projection["configured_final_bin_count"] == 79
    assert projection["final_bin_count"] == 73
    assert projection["dropped_final_bins_1based"] == [1, 2, 3, 4, 5, 6]


def test_drop_rejects_unrecognized_leading_category() -> None:
    hists, exact = payloads()
    exact["highdm"]["search_bin_labels"][0] = "unexpected"
    with pytest.raises(ValueError, match="allowed only"):
        MODULE.drop_highdm_leading_bins(hists, exact, 6)
