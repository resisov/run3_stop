from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW))
SCRIPT = WORKFLOW / "plot_trota_highdm61_2024.py"
SPEC = importlib.util.spec_from_file_location("plot_trota_highdm61_2024", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_adopted55_mapping_is_complete_and_has_five_tail_merges() -> None:
    mapping = MODULE.adopted55_mapping()
    assert len(mapping) == 55
    assert sorted(sum(mapping, [])) == list(range(60))
    assert [group for group in mapping if len(group) == 2] == [
        [22, 23],
        [34, 35],
        [40, 41],
        [52, 53],
        [58, 59],
    ]


def test_61_layout_replaces_six_bins_with_twelve() -> None:
    assert sum(size for _category, size in MODULE.OUTPUT_LAYOUT) == 61
    assert MODULE.OUTPUT_LAYOUT[:2] == (
        ("Nb1plus_T0_W0_Nres0", 6),
        ("Nb1plus_T0_W0_Nres1plus", 6),
    )


def test_split_recombination_and_49_bin_tail_are_exact() -> None:
    canonical = {
        "nominal": {
            "sumw": [float(index + 1) for index in range(60)],
            "sumw2": [float((index + 1) ** 2) for index in range(60)],
            "entries": [index + 1 for index in range(60)],
        }
    }
    component = {
        "nominal": {
            "sumw": [0.25 * (index + 1) for index in range(6)]
            + [0.75 * (index + 1) for index in range(6)],
            "sumw2": [0.4 * (index + 1) ** 2 for index in range(6)]
            + [0.6 * (index + 1) ** 2 for index in range(6)],
            "entries": [0 for _ in range(6)] + [index + 1 for index in range(6)],
        }
    }
    merged, audit = MODULE.merge_sample(
        "sample", canonical, component, MODULE.adopted55_mapping()
    )
    assert all(item["matched"] for item in audit["nominal"].values())
    assert len(merged["nominal"]["sumw"]) == 61
    adopted = MODULE.combine_record(canonical["nominal"], MODULE.adopted55_mapping())
    assert merged["nominal"]["sumw"][12:] == adopted["sumw"][6:]


def test_five_bin_recoil_label_records_the_merged_tail() -> None:
    assert MODULE.recoil_labels(5) == [
        "250-300",
        "300-350",
        "350-400",
        "400-500",
        "500-1500",
    ]
