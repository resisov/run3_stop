from __future__ import annotations

import unittest

from autonomous_allhad.merge_shape_histogram_2024 import (
    attach_to_nominal,
    merge_scaled_leaf,
)
from autonomous_allhad.shape_histogram_2024_worker import (
    FINAL_JES_PUBLIC_SOURCES,
    FINAL_SHAPE_NUISANCES,
    FINAL_SHAPE_VARIATIONS,
    load_histogram_builder,
)


class ShapeHistogram2024Test(unittest.TestCase):
    def test_final_jes_source_policy(self) -> None:
        self.assertEqual(len(FINAL_JES_PUBLIC_SOURCES), 11)
        self.assertNotIn("jesTotal", FINAL_JES_PUBLIC_SOURCES)
        self.assertNotIn("jesRegroupedTotal", FINAL_JES_PUBLIC_SOURCES)
        self.assertEqual(len(FINAL_SHAPE_NUISANCES), 20)
        self.assertEqual(len(FINAL_SHAPE_VARIATIONS), 40)
        self.assertFalse(any(name.startswith("jesTotal") for name in FINAL_SHAPE_VARIATIONS))
        self.assertFalse(any(name.startswith("jesRegroupedTotal") for name in FINAL_SHAPE_VARIATIONS))

    def test_current_search_bin_definition_is_54_bins(self) -> None:
        builder = load_histogram_builder()
        self.assertEqual(len(builder.selected_an17_recoil54_labels()), 54)
        self.assertEqual(
            builder.SELECTED_RECOIL54_SCHEME,
            "boosted_an17_selected_recoil6_with_nt0_wsplit_SR",
        )

    def test_normalization_scales_sumw2_quadratically(self) -> None:
        target: dict[str, list[float] | list[int]] = {}
        source = {"sumw": [1.0, -2.0], "sumw2": [1.0, 4.0], "entries": [1, 2]}
        merge_scaled_leaf(target, source, 3.0)
        self.assertEqual(target["sumw"], [3.0, -6.0])
        self.assertEqual(target["sumw2"], [9.0, 36.0])
        self.assertEqual(target["entries"], [1, 2])

    def test_attach_preserves_nominal_and_adds_all_shapes(self) -> None:
        nominal_leaf = {"sumw": [10.0], "sumw2": [4.0], "entries": [3]}
        nominal = {
            "histograms": {"SR": {"TT": {"nominal": nominal_leaf.copy()}, "data_obs": {"nominal": nominal_leaf.copy()}}},
            "search_bin_histograms": {},
            "highdm_variable_histograms": {},
            "summary": {},
        }
        first = FINAL_SHAPE_VARIATIONS[0]
        shapes = {
            "schema_version": "shape_histogram_2024_merged_v1",
            "status": "complete",
            "jes_source_policy": {},
            "shape_nuisances": list(FINAL_SHAPE_NUISANCES),
            "summary": {"process_datasets": {"TT": ["TT_sample"]}},
            "histograms": {"SR": {"TT": {first: {"sumw": [11.0], "sumw2": [5.0], "entries": [3]}}}},
            "search_bin_histograms": {},
            "highdm_variable_histograms": {},
        }
        combined = attach_to_nominal(nominal, shapes)
        self.assertEqual(combined["histograms"]["SR"]["TT"]["nominal"]["sumw"], [10.0])
        self.assertEqual(combined["histograms"]["SR"]["TT"][first]["sumw"], [11.0])
        self.assertEqual(len(combined["histograms"]["SR"]["TT"]), 41)
        self.assertEqual(len(combined["histograms"]["SR"]["data_obs"]), 1)


if __name__ == "__main__":
    unittest.main()
