from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_allhad.merge_shape_histogram_2024 import (
    attach_to_nominal,
    merge_scaled_leaf,
)
from autonomous_allhad.shape_histogram_2024_worker import (
    FINAL_JES_PUBLIC_SOURCES,
    FINAL_SHAPE_NUISANCES,
    FINAL_SHAPE_VARIATIONS,
    VARIATION_GROUPS,
    load_histogram_builder,
    record_entry_bounds,
)
from autonomous_allhad.shape_histogram_pair_2024 import (
    combine_pair_payloads,
    finalize_pair_accumulator,
    merge_single_source_pair,
    write_pair_with_sidecar,
)


class ShapeHistogram2024Test(unittest.TestCase):
    def test_segment_entry_bounds_are_exact_and_validated(self) -> None:
        self.assertEqual(record_entry_bounds({}, 100), (0, 100))
        self.assertEqual(
            record_entry_bounds(
                {"entry_start": 20, "entry_stop": 70, "segment_events": 50},
                100,
            ),
            (20, 70),
        )
        with self.assertRaises(RuntimeError):
            record_entry_bounds({"entry_start": -1, "entry_stop": 10}, 100)
        with self.assertRaises(RuntimeError):
            record_entry_bounds({"entry_start": 50, "entry_stop": 101}, 100)
        with self.assertRaises(RuntimeError):
            record_entry_bounds(
                {"entry_start": 20, "entry_stop": 70, "segment_events": 49},
                100,
            )

    def test_final_jes_source_policy(self) -> None:
        self.assertEqual(len(FINAL_JES_PUBLIC_SOURCES), 11)
        self.assertNotIn("jesTotal", FINAL_JES_PUBLIC_SOURCES)
        self.assertNotIn("jesRegroupedTotal", FINAL_JES_PUBLIC_SOURCES)
        self.assertEqual(len(FINAL_SHAPE_NUISANCES), 20)
        self.assertEqual(len(FINAL_SHAPE_VARIATIONS), 40)
        self.assertFalse(any(name.startswith("jesTotal") for name in FINAL_SHAPE_VARIATIONS))
        self.assertFalse(any(name.startswith("jesRegroupedTotal") for name in FINAL_SHAPE_VARIATIONS))
        for nuisance in FINAL_SHAPE_NUISANCES:
            self.assertEqual(
                VARIATION_GROUPS[nuisance],
                (f"{nuisance}Up", f"{nuisance}Down"),
            )

    def test_pair_accumulator_tracks_source_coverage(self) -> None:
        nuisance = "jer"
        source = {
            "schema_version": "ignored_by_accumulator",
            "status": "complete",
            "variations": ["jerUp", "jerDown"],
            "recoil_pt_bins": [0.0, 1.0],
            "regions": {},
            "ntop_split_policy": {},
            "search_bin_schemes": {},
            "lowdm_region_policy": {},
            "highdm_distribution_variable_specs": {},
            "highdm_distribution_regions": {},
            "lowdm_variable_specs": {},
            "lowdm_region_variables": {},
            "output_policy": {"sections": []},
            "datasets": {},
            "summary": {
                "source_record_digest": "record-a",
                "files_attempted": 1,
                "files_processed": 1,
                "events_read": 12,
                "variation_event_evaluations": 24,
                "bad_files": [],
                "btag_sf_status": {
                    "jerUp": {"applied": True},
                    "jerDown": {"applied": True},
                },
            },
        }
        accumulator = merge_single_source_pair(None, source, nuisance)
        final = finalize_pair_accumulator(accumulator, expected_sources=1)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["summary"]["source_record_count"], 1)
        self.assertEqual(final["summary"]["events_read"], 12)

    def test_twenty_pair_payloads_recombine_to_all_variations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for nuisance in FINAL_SHAPE_NUISANCES:
                variations = [f"{nuisance}Up", f"{nuisance}Down"]
                source = {
                    "schema_version": "ignored_by_accumulator",
                    "status": "complete",
                    "variations": variations,
                    "recoil_pt_bins": [0.0, 1.0],
                    "regions": {},
                    "ntop_split_policy": {},
                    "search_bin_schemes": {},
                    "lowdm_region_policy": {},
                    "highdm_distribution_variable_specs": {},
                    "highdm_distribution_regions": {},
                    "lowdm_variable_specs": {},
                    "lowdm_region_variables": {},
                    "output_policy": {"sections": []},
                    "jes_source_policy": {},
                    "datasets": {},
                    "summary": {
                        "source_record_digest": "record-a",
                        "files_attempted": 1,
                        "files_processed": 1,
                        "events_read": 12,
                        "variation_event_evaluations": 24,
                        "bad_files": [],
                        "btag_sf_status": {
                            variation: {"applied": True}
                            for variation in variations
                        },
                    },
                }
                accumulator = merge_single_source_pair(
                    None, source, nuisance
                )
                payload = finalize_pair_accumulator(
                    accumulator, expected_sources=1
                )
                histogram = Path(directory) / f"{nuisance}.json.gz"
                metadata = Path(directory) / f"{nuisance}.meta.json"
                write_pair_with_sidecar(histogram, metadata, payload)
                paths.append(histogram)
            combined = combine_pair_payloads(paths)
            self.assertEqual(combined["status"], "complete")
            self.assertEqual(
                combined["variations"], list(FINAL_SHAPE_VARIATIONS)
            )
            self.assertEqual(
                set(combined["summary"]["btag_sf_status"]),
                set(FINAL_SHAPE_VARIATIONS),
            )
            self.assertEqual(
                combined["summary"]["variation_event_evaluations"], 480
            )

    def test_current_search_bin_definition_includes_60_bins(self) -> None:
        builder = load_histogram_builder()
        self.assertEqual(len(builder.selected_an17_recoil54_labels()), 54)
        self.assertEqual(len(builder.selected_an17_recoil60_labels()), 60)
        self.assertEqual(len(builder.LOWDM_42BIN_LABELS), 42)
        self.assertFalse(hasattr(builder, "LOWDM_53BIN_LABELS"))
        self.assertEqual(
            builder.SELECTED_RECOIL54_SCHEME,
            "boosted_an17_selected_recoil6_with_nt0_wsplit_SR",
        )
        self.assertEqual(
            builder.EXTENDED_RECOIL60_SCHEME,
            "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR",
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
