import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import awkward as ak
import correctionlib
import numpy as np

from autonomous_allhad.analysis_scale_factors import (
    AnalysisScaleFactorUnavailable,
    TOPW_CORRECTION_BRANCHES,
    TOPW_EVENT_BRANCHES,
    TOPW_PT_EDGES,
    TopWEvents,
    topw_event_variations,
    apply_topw_event_weights,
    topw_fit_categories,
    topw_file_input_policy,
    topw_file_sf_triplet,
    apply_topw_missing_input_fallback,
    apply_topw_highpt_extrapolation,
    topw_pass_fail_triplet,
    loose_muon_lowpt_triplet,
    met_trigger_triplet,
    photon_trigger_triplet,
    veto_electron_lowpt_triplet,
)
from workflow.sf_payload import correction, correction_set, install_adopted_result, write_json_gz, topw_measurement_payload


class TopWIntegrationTest(unittest.TestCase):
    def payload(self):
        records = []
        for tag, edges in TOPW_PT_EDGES.items():
            for cat in ("tp1", "tp2", "tp3", "other"):
                values = (1.25, 1.5, 1.) if (tag, cat) == ("top", "tp3") else (1., 1., 1.)
                records.append(correction(
                    name=f"topw_{tag}_{cat}_sf", description="test", axes=[("pt", edges)],
                    **{key: [value] * (len(edges) - 1)
                       for key, value in zip(("nominal", "up", "down"), values)},
                ))
        return correctionlib.CorrectionSet.from_string(json.dumps(correction_set("test", records)))

    def arrays(self):
        return ak.Array({
            "gen_weight": [2., 3., 4.], "year": [2024] * 3,
            "fatjet_corrected_pt": [[450., 450.], [450.], []],
            "fatjet_eta_all": [[0., 0.], [0.], []],
            "fatjet_phi_all": [[0., 1.], [0.], []],
            "fatjet_msoftdrop_all": [[150., 150.], [150.], []],
            "fatjet_id_all": [[True, True], [True], []],
            "fatjet_decay_flavor_all": [[4, 4], [4], []],
            "fatjet_boosted_top_pass_all": [[True, False], [True], []],
            "fatjet_boosted_w_pass_all": [[False, False], [False], []],
        })

    def histograms(self, empty=False):
        import hist
        h = hist.Hist(hist.axis.StrCategory(["analysis", "score_only"]),
                      hist.axis.StrCategory(["pass", "fail"]),
                      hist.axis.IntCategory([0, 1, 2, 3, 4]),
                      hist.axis.Variable([200, 3000], name="pt"),
                      hist.axis.Variable([0, 2], name="abseta"))
        h.view()[0, 0] = 0 if empty else 80
        h.view()[0, 1] = 0 if empty else 20
        return {key: {"TT_Tune_test": h} for key in ("GlobalParT3_Top", "GlobalParT3_W")}

    def evaluate(self, *, empty=False, policy=None, cleaned=None):
        with mock.patch("autonomous_allhad.analysis_scale_factors._payload", return_value=self.payload()), \
             mock.patch("autonomous_allhad.analysis_scale_factors._topw_efficiencies", return_value=self.histograms(empty)):
            return topw_event_variations(Path("/eos/test"), "2024", "TT_Tune_test", "TT",
                                         self.arrays(), policy or {"mode": "available"}, cleaned=cleaned)

    def test_product_preserves_nonzero_variation_at_zero_nominal(self):
        values, audit = self.evaluate()
        np.testing.assert_array_equal(values["nominal"], [0, 1.25, 1])
        np.testing.assert_array_equal(values["topw_top_tp3_pt400to480Down"], [1, 1, 1])
        np.testing.assert_array_equal(values["topw_top_tp3_pt480to600Down"], values["nominal"])
        self.assertEqual(audit["eligible_jets"], 3)
        self.assertTrue(all(np.all(np.isfinite(v) & (v >= 0)) for v in values.values()))

    def test_empty_efficiency_and_missing_truth_keep_all_variations_at_unity(self):
        for kwargs in ({"empty": True}, {"policy": {"mode": "unity_missing_branches", "missing": ["TopWTruth"]}}):
            values, _ = self.evaluate(**kwargs)
            self.assertGreater(len(values), 1)
            for value in values.values():
                np.testing.assert_array_equal(value, np.ones(3))

    def test_region_cleaning_removes_only_overlapping_jets(self):
        values, audit = self.evaluate(cleaned=ak.Array([[True, False], [True], []]))
        np.testing.assert_array_equal(values["nominal"], [1.25, 1.25, 1])
        self.assertEqual(audit["eligible_jets"], 2)

    def test_all_existing_variations_receive_nominal_topw(self):
        with mock.patch("autonomous_allhad.analysis_scale_factors.topw_event_variations", return_value=self.evaluate()):
            result = apply_topw_event_weights(
                {"nominal": [2, 3, 4], "pileupUp": [4, 6, 8]}, {}, Path("/eos/test"),
                "2024", "TT", "TT", self.arrays(), {"mode": "available"},
            )
        np.testing.assert_array_equal(result["pileupUp"], [0, 7.5, 8])
        np.testing.assert_array_equal(result["topw_top_tp3_pt400to480Down"], [2, 3, 4])

    def test_measurement_parser_requires_explicit_failure(self):
        pages = {}
        for tag, edges in TOPW_PT_EDGES.items():
            pages[tag] = "".join(
                f'<h2 id="fit-mistag1pct-pt{lo}to{hi}">bin</h2>\n'
                + "Warning - No valid low-error found for : SF_other\n"
                + "\n".join(f"SF_{cat} : +1.1 -0.1/+0.2 (68%)" for cat in ("tp1", "tp2", "tp3", "other"))
                for lo, hi in zip(edges[:-1], edges[1:])
            )
        payload, audit = topw_measurement_payload("2024", pages)
        evaluator = correctionlib.CorrectionSet.from_string(json.dumps(payload))
        self.assertEqual(evaluator["topw_top_other_sf"].evaluate("nominal", 450.), 1.)
        self.assertEqual(evaluator["topw_top_tp3_sf"].evaluate("nominal", 450.), 1.1)
        self.assertEqual(len(audit["top"]["failed_fit_unity_bins"]), 4)
        with self.assertRaises(ValueError):
            topw_measurement_payload("2024", {**pages, "w": pages["w"].replace("SF_tp3 :", "lost :")})

    def test_reader_checks_every_event_and_jet_identity(self):
        class Tree:
            def __init__(self, array):
                self.array, self.num_entries = array, len(array)
            def arrays(self, branches, *, entry_start=0, entry_stop=None, library="ak"):
                return self.array[list(branches)][entry_start:entry_stop]
        events = self.arrays()
        for name in TOPW_CORRECTION_BRANCHES[:-2]:
            events = ak.with_field(events, [1, 2, 3], name)
        events = ak.with_field(events, [[0, 1], [0], []], "fatjet_source_index_all")
        root = {"Events": Tree(events), "TopWTruth": Tree(events[list(TOPW_CORRECTION_BRANCHES)])}
        reader = TopWEvents(root, {"mode": "available"})
        self.assertEqual(sum(len(c) for c in reader.iterate(["gen_weight"], step_size=2)), 3)
        root["TopWTruth"] = Tree(ak.with_field(root["TopWTruth"].array, [1, 9, 3], "event"))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            reader.arrays(["gen_weight"])


class TopWFitCategoryTest(unittest.TestCase):
    def test_top_like_includes_partial_and_mistag_in_tp1(self):
        np.testing.assert_array_equal(
            topw_fit_categories([0, 1, 2, 3, 4], top_like=True),
            ["tp1", "tp1", "tp1", "tp2", "tp3"],
        )

    def test_non_top_is_other_even_with_contained_w(self):
        np.testing.assert_array_equal(
            topw_fit_categories([0, 1, 2, 3, 4], top_like=False),
            ["other"] * 5,
        )

    def test_rejects_ambiguous_group_and_invalid_flavor(self):
        for flavor in ([5], [-1], [1.0], [[1]], [True]):
            with self.subTest(flavor=flavor), self.assertRaises(ValueError):
                topw_fit_categories(flavor, top_like=True)
        for group in (None, 1, "TT"):
            with self.subTest(group=group), self.assertRaises(ValueError):
                topw_fit_categories([0], top_like=group)


class AsymmetricScaleFactorPayloadTest(unittest.TestCase):
    def test_roundtrip_preserves_asymmetry_and_zero_endpoint(self):
        payload = correction_set("test", [correction(
            name="topw", description="test", axes=[("pt", [400, 480, 600])],
            nominal=[1.103, 1.0], up=[1.224, 1.0], down=[0.0, 1.0],
        )])
        evaluator = correctionlib.CorrectionSet.from_string(json.dumps(payload))["topw"]
        for name, expected in (("nominal", 1.103), ("up", 1.224), ("down", 0.0)):
            self.assertEqual(evaluator.evaluate(name, 450.0), expected)
        for name in ("nominal", "up", "down"):
            self.assertEqual(evaluator.evaluate(name, 500.0), 1.0)

    def test_rejects_mixed_missing_or_invalid_endpoints(self):
        for endpoints in (
            {}, {"up": [1.1]}, {"down": [0.9]},
            {"up": [1.1], "down": [0.9], "uncertainty": [0.1]},
            {"up": [0.9], "down": [0.8]},
            {"up": [1.1], "down": [-0.1]},
            {"up": [float("nan")], "down": [0.9]},
            {"up": [1.1, 1.2], "down": [0.9]},
        ):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                correction(name="topw", description="test", axes=[("pt", [400, 480])],
                           nominal=[1.0], **endpoints)


class TopWMissingInputTest(unittest.TestCase):
    class Tree:
        num_entries = 3

        def __init__(self, branches=()):
            self.branches = list(branches)

        def keys(self):
            return self.branches

    def root(self):
        return {
            "Events": self.Tree(TOPW_EVENT_BRANCHES),
            "TopWTruth": self.Tree(TOPW_CORRECTION_BRANCHES),
            "TopWTruth_metadata": json.dumps({
                "status": "complete", "schema_version": "topw_truth_v1",
                "events_entries": 3,
            }),
        }

    def test_missing_tree_or_each_correction_branch_is_unity(self):
        for missing in ("TopWTruth", *TOPW_CORRECTION_BRANCHES):
            with self.subTest(missing=missing):
                root = self.root()
                if missing == "TopWTruth":
                    del root[missing]
                else:
                    root["TopWTruth"].branches.remove(missing)
                policy = topw_file_input_policy(root)
                self.assertEqual(policy["sf"], 1.0)
                self.assertEqual(policy["uncertainty"], 0.0)
                evaluator = mock.Mock(side_effect=AssertionError("must not evaluate"))
                for weights in topw_file_sf_triplet(3, policy, evaluator):
                    np.testing.assert_array_equal(weights, np.ones(3))
                evaluator.assert_not_called()
                self.assertEqual(root["Events"].num_entries, 3)

    def test_complete_inputs_keep_measured_triplet(self):
        policy = topw_file_input_policy(self.root())
        expected = ([0.9, 1.1, 1], [1, 1.2, 1], [0.8, 1, 1])
        evaluator = mock.Mock(return_value=expected)
        actual = topw_file_sf_triplet(3, policy, evaluator)
        evaluator.assert_called_once_with()
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)

    def test_fallback_keeps_all_events_and_other_weight_variations(self):
        policy = topw_file_input_policy({"Events": self.Tree()})
        variations = {"nominal": np.asarray([2., -3., 4.]),
                      "pileupUp": np.asarray([2.2, -3.3, 4.4])}
        status = {"components": {"pileup": {"applied": True}}}
        result = apply_topw_missing_input_fallback(variations, status, policy, 3)
        self.assertEqual(set(result), set(variations))
        for key in variations:
            np.testing.assert_array_equal(result[key], variations[key])
        self.assertTrue(status["components"]["pileup"]["applied"])
        self.assertEqual(status["topw_correction"]["uncertainty"], 0.0)

    def test_complete_file_does_not_change_weight_bundle(self):
        variations = {"nominal": np.asarray([2., 3., 4.])}
        status = {}
        self.assertIs(apply_topw_missing_input_fallback(
            variations, status, topw_file_input_policy(self.root()), 3,
        ), variations)
        self.assertEqual(status, {})

    def test_corruption_and_invalid_measurements_are_not_missing_branches(self):
        root = self.root()
        root["TopWTruth"].num_entries = 2
        with self.assertRaisesRegex(RuntimeError, "entry mismatch"):
            topw_file_input_policy(root)
        with self.assertRaises(KeyError):
            topw_file_input_policy({})
        with self.assertRaisesRegex(RuntimeError, "broken payload"):
            topw_file_sf_triplet(3, {"mode": "available"},
                                mock.Mock(side_effect=RuntimeError("broken payload")))
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            topw_file_sf_triplet(3, {"mode": "available"},
                                lambda: ([np.nan] * 3, [1] * 3, [1] * 3))

    def test_histogram_validator_retains_file_without_truth(self):
        from workflow import validate_histogram_root_inputs as validator

        tree = self.Tree(["dataset_id"])
        tree.iterate = lambda *args, **kwargs: iter([{"dataset_id": np.zeros(3)}])
        root = {"Events": tree}
        metadata = {"events_written": 3, "datasets": {"test": {"is_data": False}}}
        with mock.patch.object(validator.uproot, "open", return_value=nullcontext(root)), \
             mock.patch.object(validator, "read_json", return_value=metadata), \
             mock.patch.object(Path, "exists", return_value=True):
            result = validator.validate_root(Path("/eos/test.root"), step_size=10)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["entries_read"], 3)
        self.assertEqual(result["topw_correction_inputs"]["sf"], 1.0)


class TopWExtrapolationTest(unittest.TestCase):
    def test_explicit_fit_failure_is_unity_for_both_years_and_taggers(self):
        for year in ("2024", "2025"):
            for tagger in ("top", "w"):
                with self.subTest(year=year, tagger=tagger):
                    result = apply_topw_highpt_extrapolation(
                        [450, 500, 650], [1.1, 1.293, np.nan],
                        [1.2, 1.444, np.inf], [0.9, 0.0, -1.0],
                        tagger=tagger, year=year, fit_failed=[False, True, True],
                    )
                    for actual, expected in zip(result, ([1.1, 1, 1], [1.2, 1, 1], [0.9, 1, 1])):
                        np.testing.assert_array_equal(actual, expected)

    def test_fit_failure_preserves_pass_and_fail_yields(self):
        scales = apply_topw_highpt_extrapolation(
            [500, 500], [1.293, 1.293], [1.444, 1.444], [0, 0],
            tagger="top", year="2024", fit_failed=[True, True],
        )
        for weights in topw_pass_fail_triplet([True, False], [0.8, 0.8], *scales):
            np.testing.assert_array_equal(weights, [1, 1])

    def test_fit_failure_mask_is_explicit_and_aligned(self):
        for mask in ([1], [True, False], True):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                apply_topw_highpt_extrapolation(
                    [500], [1], [1], [1], tagger="top", year="2024", fit_failed=mask,
                )
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            apply_topw_highpt_extrapolation(
                [500], [np.nan], [1], [1], tagger="top", year="2024", fit_failed=[False],
            )

    def test_both_years_and_exact_upper_edges(self):
        for year in ("2024", "2025"):
            for tagger, edge in (("top", 1200.0), ("w", 800.0)):
                with self.subTest(year=year, tagger=tagger):
                    pt = [np.nextafter(edge, 0.0), edge, edge + 1000.0]
                    result = apply_topw_highpt_extrapolation(
                        pt, [1.1, np.nan, np.nan], [1.3, np.nan, np.nan],
                        [0.8, np.nan, np.nan], tagger=tagger, year=year,
                    )
                    for actual, expected in zip(result, ([1.1, 1, 1], [1.3, 1, 1], [0.8, 1, 1])):
                        np.testing.assert_array_equal(actual, expected)

    def test_zero_uncertainty_overrides_extrapolated_variations(self):
        result = apply_topw_highpt_extrapolation(
            [1500], [1.3], [1.5], [0.9], tagger="top", year="2024",
        )
        for values in result:
            np.testing.assert_array_equal(values, [1.0])

    def test_in_range_missing_values_do_not_fall_back_to_unity(self):
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            apply_topw_highpt_extrapolation(
                [500], [np.nan], [1.2], [0.8], tagger="top", year="2024",
            )

    def test_rejects_invalid_inputs(self):
        for pt, nominal, up, down in (
            ([np.inf], [1], [1], [1]),
            ([np.nan], [1], [1], [1]),
            ([-1], [1], [1], [1]),
            ([[500]], [[1]], [[1]], [[1]]),
            ([500, 600], [1], [1], [1]),
        ):
            with self.subTest(pt=pt), self.assertRaises(ValueError):
                apply_topw_highpt_extrapolation(
                    pt, nominal, up, down, tagger="top", year="2024",
                )

    def test_rejects_invalid_in_range_variations(self):
        for nominal, up, down in ((1, 0.9, 0.8), (1, 1.2, 1.1), (1, 1.2, -0.1)):
            with self.subTest(nominal=nominal, up=up, down=down), self.assertRaises(AnalysisScaleFactorUnavailable):
                apply_topw_highpt_extrapolation(
                    [500], [nominal], [up], [down], tagger="top", year="2024",
                )

    def test_does_not_mutate_input(self):
        values = np.asarray([1.2])
        apply_topw_highpt_extrapolation(
            [1300], values, values, values, tagger="top", year="2024",
        )
        np.testing.assert_array_equal(values, [1.2])

    def test_unknown_year_or_tagger_is_not_accepted(self):
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            apply_topw_highpt_extrapolation([1300], [1], [1], [1], tagger="top", year="2023")
        with self.assertRaises(ValueError):
            apply_topw_highpt_extrapolation([1300], [1], [1], [1], tagger="resolved", year="2024")


class TopWPassFailTest(unittest.TestCase):
    def test_pass_fail_sum_and_anticorrelated_variations(self):
        result = topw_pass_fail_triplet([True, False], [0.4, 0.4],
                                       [1.1, 1.1], [1.3, 1.3], [0.8, 0.8])
        for weights in result:
            self.assertAlmostEqual(0.4 * weights[0] + 0.6 * weights[1], 1.0)
        self.assertGreater(result[1][0], result[0][0])
        self.assertLess(result[1][1], result[0][1])
        self.assertLess(result[2][0], result[0][0])
        self.assertGreater(result[2][1], result[0][1])

    def test_highpt_extrapolation_is_unity_for_pass_and_fail(self):
        scales = apply_topw_highpt_extrapolation([1200, 1500], [np.nan] * 2,
                   [np.nan] * 2, [np.nan] * 2, tagger="top", year="2025")
        for weights in topw_pass_fail_triplet([True, False], [0.8, 0.4], *scales):
            np.testing.assert_array_equal(weights, [1, 1])

    def test_unsupported_efficiencies_are_rejected(self):
        for tagged, eff, sf in ((True, np.nan, 1), (False, -0.1, 1), (True, 1.1, 1)):
            with self.subTest(tagged=tagged, eff=eff, sf=sf), self.assertRaises(AnalysisScaleFactorUnavailable):
                topw_pass_fail_triplet([tagged], [eff], [sf], [sf], [sf])

    def test_zero_pass_or_fail_denominator_has_zero_uncertainty(self):
        for tagged in (True, False):
            for weights in topw_pass_fail_triplet(
                [tagged, tagged], [0., 1.], [0.9, 0.9], [1.3, 1.3], [0.6, 0.6],
            ):
                np.testing.assert_array_equal(weights, [1., 1.])

    def test_empty_efficiency_denominator_is_explicit_cell_local_fallback(self):
        nominal, up, down = topw_pass_fail_triplet(
            [True, False], [np.nan, .4], [np.nan, 1.1], [np.nan, 1.3], [np.nan, .8],
            denominator=[0, 10],
        )
        for weights in (nominal, up, down):
            self.assertEqual(weights[0], 1.0)
        np.testing.assert_allclose([nominal[1], up[1], down[1]], [.56/.6, .48/.6, .68/.6])
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            topw_pass_fail_triplet([True], [np.nan], [1.], [1.], [1.], denominator=[10])

    def test_invalid_denominator_is_not_an_empty_cell(self):
        for denominator in ([np.nan], [np.inf], [-1], [0, 10]):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                topw_pass_fail_triplet([True], [.5], [1.], [1.], [1.], denominator=denominator)

    def test_observed_t2tt_case_saturates_pass_and_fail_together(self):
        eff = 9972.0 / (9972.0 + 496.0)
        nominal, up, down = topw_pass_fail_triplet(
            [True, False], [eff, eff], [1.103] * 2, [1.224] * 2, [1.017] * 2,
        )
        np.testing.assert_allclose(nominal, [1.0 / eff, 0.0])
        np.testing.assert_array_equal(up, nominal)
        self.assertAlmostEqual(down[0], 1.017)
        self.assertGreater(down[1], 0.0)
        for weights in (nominal, up, down):
            self.assertTrue(np.all(weights >= 0))
            self.assertAlmostEqual(eff * weights[0] + (1 - eff) * weights[1], 1.0)

    def test_only_up_endpoint_can_saturate(self):
        nominal, up, down = topw_pass_fail_triplet(
            [True, False], [0.8, 0.8], [1.1] * 2, [1.4] * 2, [0.9] * 2,
        )
        np.testing.assert_allclose(nominal, [1.1, 0.6])
        np.testing.assert_allclose(up, [1.25, 0.0])
        np.testing.assert_allclose(down, [0.9, 1.4])

    def test_projection_preserves_total_across_efficiencies(self):
        eff = np.concatenate(([1.e-12, np.nextafter(1., 0.)], np.linspace(.01, .99, 99)))
        scales = [np.full_like(eff, sf) for sf in (1.3, 2.0, 0.0)]
        passed = topw_pass_fail_triplet(np.ones(eff.shape, dtype=bool), eff, *scales)
        failed = topw_pass_fail_triplet(np.zeros(eff.shape, dtype=bool), eff, *scales)
        for p, f in zip(passed, failed):
            self.assertTrue(np.all(np.isfinite(p)) and np.all(np.isfinite(f)))
            self.assertTrue(np.all(p >= 0) and np.all(f >= 0))
            np.testing.assert_allclose(eff * p + (1 - eff) * f, 1.0, rtol=0, atol=2.e-15)
        self.assertTrue(np.all(passed[1] >= passed[0]))
        self.assertTrue(np.all(passed[0] >= passed[2]))
        self.assertTrue(np.all(failed[1] <= failed[0]))
        self.assertTrue(np.all(failed[0] <= failed[2]))

    def test_unit_efficiency_accepts_saturated_scales(self):
        for weights in topw_pass_fail_triplet([True], [1.], [1.2], [1.4], [1.]):
            np.testing.assert_array_equal(weights, [1.])

    def test_supported_boundary_efficiencies(self):
        for weights in topw_pass_fail_triplet([True, False], [1, 0], [1, 1], [1, 1], [1, 1]):
            np.testing.assert_array_equal(weights, [1, 1])

    def test_variations_must_be_valid_and_bracketed(self):
        for nominal, up, down in ((1, np.nan, 0.8), (1, 0.9, 0.8), (1, 1.2, -0.1)):
            with self.assertRaises(AnalysisScaleFactorUnavailable):
                topw_pass_fail_triplet([True], [0.4], [nominal], [up], [down])


class AnalysisScaleFactorTest(unittest.TestCase):
    @staticmethod
    def _install_constant_payloads(repo: Path, year: str = "2024") -> Path:
        target = repo / "analysis/data/AnalysisSF" / year
        target.mkdir(parents=True)
        write_json_gz(
            target / "met_trigger_sf.json.gz",
            correction_set(
                "met",
                [
                    correction(name="met_trigger_sf_genuine", description="genuine", axes=[("met", [100, 300, 800])], nominal=[0.9, 1.0], uncertainty=[0.1, 0.02]),
                    correction(name="met_trigger_sf_qcd", description="legacy sentinel", axes=[("met", [100, 300, 800])], nominal=[0.7, 0.8], uncertainty=[0.1, 0.02]),
                ],
            ),
        )
        write_json_gz(
            target / "photon_trigger_sf.json.gz",
            correction_set("photon", [correction(name="photon_trigger_sf", description="ph", axes=[("abseta", [0, 2.5]), ("pt", [220, 1000])], nominal=[0.98], uncertainty=[0.01])]),
        )
        write_json_gz(
            target / "veto_electron_5to10_sf.json.gz",
            correction_set("electron", [correction(name="veto_electron_id_5to10_sf", description="e ID only", axes=[("abseta", [0, 2.5]), ("pt", [5, 10])], nominal=[1.05], uncertainty=[0.03])]),
        )
        write_json_gz(
            target / "loose_muon_5to10_sf.json.gz",
            correction_set("muon", [correction(name="loose_muon_id_5to10_sf", description="m ID only", axes=[("abseta", [0, 2.4]), ("pt", [5, 10])], nominal=[0.97], uncertainty=[0.02])]),
        )
        return target

    def test_correctionlib_roundtrip_and_variations(self):
        payload = correction_set(
            "test",
            [
                correction(
                    name="test_sf",
                    description="test",
                    axes=[("abseta", [0.0, 1.0, 2.5]), ("pt", [5.0, 7.0, 10.0])],
                    nominal=[0.9, 1.0, 1.1, 1.2],
                    uncertainty=[0.1, 0.2, 0.05, 0.1],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.gz"
            digest = write_json_gz(path, payload)
            self.assertEqual(len(digest), 64)
            evaluator = correctionlib.CorrectionSet.from_file(str(path))["test_sf"]
            self.assertAlmostEqual(evaluator.evaluate("nominal", 0.5, 6.0), 0.9)
            self.assertAlmostEqual(evaluator.evaluate("up", 0.5, 6.0), 1.0)
            self.assertAlmostEqual(evaluator.evaluate("down", 0.5, 6.0), 0.8)

    def test_install_refuses_preliminary_result(self):
        payload = correction_set(
            "test",
            [correction(name="x", description="x", axes=[("pt", [1, 2])], nominal=[1], uncertainty=[0.1])],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result.json"
            result.write_text(json.dumps({"status": "validation_pending"}))
            with self.assertRaises(RuntimeError):
                install_adopted_result(result, root / "x.json.gz", payload)

    def test_analysis_helpers_evaluate_installed_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            self.assertTrue(np.allclose(met_trigger_triplet(repo, [250], qcd=False)[0], [0.9]))
            self.assertTrue(np.allclose(met_trigger_triplet(repo, [250], qcd=True)[0], [0.9]))
            self.assertAlmostEqual(float(photon_trigger_triplet(repo, ak.Array([[0.2]]), ak.Array([[300.0]]))[0][0][0]), 0.98)
            self.assertAlmostEqual(float(veto_electron_lowpt_triplet(repo, ak.Array([[0.2]]), ak.Array([[7.0]]))[0][0][0]), 1.05)
            self.assertAlmostEqual(float(loose_muon_lowpt_triplet(repo, ak.Array([[0.2]]), ak.Array([[7.0]]))[0][0][0]), 0.97)

    def test_missing_payload_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnalysisScaleFactorUnavailable):
                met_trigger_triplet(Path(tmp), [300.0], qcd=False)

    def test_2025_helpers_select_2025_payload_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo, year="2025")
            self.assertTrue(
                np.allclose(
                    met_trigger_triplet(repo, [250], qcd=False, year="2025")[0],
                    [0.9],
                )
            )
            self.assertAlmostEqual(
                float(
                    veto_electron_lowpt_triplet(
                        repo,
                        ak.Array([[0.2]]),
                        ak.Array([[7.0]]),
                        year="2025",
                    )[0][0][0]
                ),
                1.05,
            )

    def test_installed_lowpt_payloads_are_id_only(self):
        repo = Path(__file__).resolve().parents[2]
        electron = correctionlib.CorrectionSet.from_file(
            str(repo / "analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz")
        )
        muon = correctionlib.CorrectionSet.from_file(
            str(repo / "analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz")
        )
        self.assertIn("veto_electron_id_5to10_sf", electron)
        self.assertIn("loose_muon_id_5to10_sf", muon)
        self.assertIn("excluding isolation", electron["veto_electron_id_5to10_sf"].description)
        self.assertIn("excluding isolation", muon["loose_muon_id_5to10_sf"].description)

    def test_adopted_lowpt_payloads_reach_histogram_weight_variations(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        repo = Path(__file__).resolve().parents[2]
        ones = lambda values: ak.ones_like(values, dtype=float)
        corrections = {
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_mu_loose_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_medium_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_hlt_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
        }
        empty_float = ak.Array([[], []])
        empty_bool = ak.Array([[], []])
        with mock.patch(
            "autonomous_allhad.real_subset_worker.load_analysis_corrections",
            return_value=corrections,
        ):
            _generator, variations, status = compute_weight_bundle(
                {"genWeight": np.ones(2)},
                repo,
                "WJetsToLNu",
                "WJets",
                "2024",
                2,
                empty_float,
                empty_float,
                empty_float,
                empty_bool,
                ak.Array([[0.4], []]),
                ak.Array([[0.0], []]),
                ak.Array([[7.5], []]),
                ak.Array([[0.0], []]),
                ak.Array([[True], []]),
                ak.Array([[False], []]),
                np.asarray([1, 0]),
                np.asarray([0, 0]),
                ak.Array([[], [1.0]]),
                ak.Array([[], [7.5]]),
                ak.Array([[], [0.0]]),
                ak.Array([[], [True]]),
                ak.Array([[], [False]]),
                np.asarray([0, 1]),
                np.asarray([0, 0]),
                empty_float,
                empty_float,
                empty_float,
                empty_bool,
                np.asarray([False, False]),
                met_pt=np.asarray([300.0, 300.0]),
                met_trigger_mask=np.asarray([False, False]),
                analysis_sf_components=("met_trigger", "photon_trigger", "veto_electron_5to10", "loose_muon_5to10"),
            )
        electron = status["components"]["veto_electron_5to10"]
        muon = status["components"]["loose_muon_5to10"]
        self.assertTrue(electron["applied"])
        self.assertTrue(muon["applied"])
        for name in (
            "veto_electron_5to10Up",
            "veto_electron_5to10Down",
            "loose_muon_5to10Up",
            "loose_muon_5to10Down",
        ):
            self.assertIn(name, variations)
        self.assertNotEqual(variations["veto_electron_5to10Up"][0], variations["nominal"][0])
        self.assertNotEqual(variations["veto_electron_5to10Down"][0], variations["nominal"][0])
        self.assertNotEqual(variations["loose_muon_5to10Up"][1], variations["nominal"][1])
        self.assertNotEqual(variations["loose_muon_5to10Down"][1], variations["nominal"][1])

    def test_lowpt_edge_clipping_is_replaced_and_all_analysis_sf_shapes_fill(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
            from workflow.build_flat_boosted_recoil_hists import (
                add_hist,
                empty_hist,
                histogram_variations,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        # Probe both open edges of the measured interval and the exact handoff
        # point.  7.5 GeV is not a threshold; the production domain is 5<pT<10.
        n = 8
        empty_float = ak.Array([[] for _ in range(n)])
        empty_bool = ak.Array([[] for _ in range(n)])

        def triplet(pt, nominal, up, down):
            return tuple(ak.ones_like(pt, dtype=float) * value for value in (nominal, up, down))

        corrections = {
            # Distinct non-unity edge values prove that the old 10 GeV clamp is
            # removed below 10 GeV instead of being multiplied by the new SF.
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: triplet(pt, 1.4, 1.5, 1.3),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
            "get_mu_loose_id_sf": lambda year, eta, pt: triplet(pt, 1.3, 1.4, 1.2),
            "get_mu_medium_id_sf": lambda year, eta, pt: triplet(pt, 1.0, 1.0, 1.0),
            "get_mu_hlt_sf": lambda year, eta, pt: triplet(pt, 1.0, 1.0, 1.0),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            with mock.patch(
                "autonomous_allhad.real_subset_worker.load_analysis_corrections",
                return_value=corrections,
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.analysis_workdir",
                side_effect=lambda repo: nullcontext(),
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.veto_electron_lowpt_triplet",
                wraps=veto_electron_lowpt_triplet,
            ) as electron_lowpt_evaluator:
                _generator, variations, status = compute_weight_bundle(
                    {"genWeight": np.ones(n)},
                    repo,
                    "WJetsToLNu",
                    "WJets",
                    "2024",
                    n,
                    empty_float,
                    empty_float,
                    empty_float,
                    empty_bool,
                    ak.Array([[0.4], [0.4], [0.4], [], [], [], [], []]),
                    ak.Array([[0.2], [0.2], [0.2], [], [], [], [], []]),
                    ak.Array([[5.001], [9.999], [10.0], [], [], [], [], []]),
                    ak.Array([[0.0], [0.0], [0.0], [], [], [], [], []]),
                    ak.Array([[True], [True], [True], [], [], [], [], []]),
                    ak.Array([[False], [False], [False], [], [], [], [], []]),
                    np.asarray([1, 1, 1, 0, 0, 0, 0, 0]),
                    np.zeros(n, dtype=int),
                    ak.Array([[], [], [], [0.5], [0.5], [0.5], [], []]),
                    ak.Array([[], [], [], [5.001], [9.999], [10.0], [], []]),
                    ak.Array([[], [], [], [0.0], [0.0], [0.0], [], []]),
                    ak.Array([[], [], [], [True], [True], [True], [], []]),
                    ak.Array([[], [], [], [False], [False], [False], [], []]),
                    np.asarray([0, 0, 0, 1, 1, 1, 0, 0]),
                    np.zeros(n, dtype=int),
                    ak.Array([[], [], [], [], [], [], [0.2], []]),
                    ak.Array([[], [], [], [], [], [], [300.0], []]),
                    ak.Array([[], [], [], [], [], [], [0.0], []]),
                    ak.Array([[], [], [], [], [], [], [True], []]),
                    np.asarray([False, False, False, False, False, False, True, False]),
                    met_pt=np.asarray([250.0] * n),
                    met_trigger_mask=np.asarray([False, False, False, False, False, False, False, True]),
                    analysis_sf_components=("met_trigger", "photon_trigger", "veto_electron_5to10", "loose_muon_5to10"),
                )

        # The adopted low-pT measurement is parameterized in reconstructed
        # Electron_eta, while the official EGM ID payload uses etaSC.
        lowpt_eta = electron_lowpt_evaluator.call_args.args[1]
        np.testing.assert_allclose(
            np.asarray(ak.to_list(lowpt_eta[:3]), dtype=float).reshape(-1),
            np.asarray([0.4, 0.4, 0.4]),
        )

        self.assertTrue(
            status["components"]["electron_id"]["applied"],
            status["components"]["electron_id"],
        )
        self.assertTrue(
            status["components"]["muon_id"]["applied"],
            status["components"]["muon_id"],
        )
        np.testing.assert_allclose(
            variations["nominal"],
            np.asarray([1.05, 1.05, 1.4, 0.97, 0.97, 1.3, 0.98, 0.9]),
            rtol=0.0,
            atol=1.0e-12,
        )
        expected_shapes = (
            "veto_electron_5to10Up",
            "veto_electron_5to10Down",
            "loose_muon_5to10Up",
            "loose_muon_5to10Down",
            "photon_triggerUp",
            "photon_triggerDown",
            "met_triggerUp",
            "met_triggerDown",
        )
        histogram_weights = histogram_variations(variations, nominal_only=False)
        for name in expected_shapes:
            self.assertIn(name, histogram_weights)
        for component, event_index in (
            ("veto_electron_5to10", 0),
            ("veto_electron_5to10", 1),
            ("loose_muon_5to10", 3),
            ("loose_muon_5to10", 4),
            ("photon_trigger", 6),
            ("met_trigger", 7),
        ):
            for direction in ("Up", "Down"):
                name = f"{component}{direction}"
                self.assertNotEqual(
                    histogram_weights[name][event_index],
                    variations["nominal"][event_index],
                )

        # Exercise the same leaf filler used by the production histogrammer.
        for name in ("nominal", *expected_shapes):
            leaf = empty_hist()
            add_hist(
                leaf,
                np.full(n, 350.0),
                histogram_weights[name],
                np.ones(n, dtype=bool),
            )
            self.assertAlmostEqual(sum(leaf["sumw"]), float(np.sum(histogram_weights[name])))

        for component in (
            "met_trigger",
            "photon_trigger",
            "veto_electron_5to10",
            "loose_muon_5to10",
        ):
            self.assertTrue(status["components"][component]["applied"])

    def test_trigger_only_configuration_excludes_lowpt_lepton_nominal_and_shapes(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        n = 4
        empty_float = ak.Array([[] for _ in range(n)])
        empty_bool = ak.Array([[] for _ in range(n)])
        ones = lambda values: ak.ones_like(values, dtype=float)
        corrections = {
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_mu_loose_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_medium_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_hlt_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            with mock.patch(
                "autonomous_allhad.real_subset_worker.load_analysis_corrections",
                return_value=corrections,
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.analysis_workdir",
                side_effect=lambda repo: nullcontext(),
            ):
                _generator, variations, status = compute_weight_bundle(
                    {"genWeight": np.ones(n)},
                    repo, "WJetsToLNu", "WJets", "2024", n,
                    empty_float, empty_float, empty_float, empty_bool,
                    ak.Array([[0.4], [], [], []]),
                    ak.Array([[0.0], [], [], []]),
                    ak.Array([[7.5], [], [], []]),
                    ak.Array([[0.0], [], [], []]),
                    ak.Array([[True], [], [], []]),
                    ak.Array([[False], [], [], []]),
                    np.asarray([1, 0, 0, 0]), np.zeros(n, dtype=int),
                    ak.Array([[], [0.5], [], []]),
                    ak.Array([[], [7.5], [], []]),
                    ak.Array([[], [0.0], [], []]),
                    ak.Array([[], [True], [], []]),
                    ak.Array([[], [False], [], []]),
                    np.asarray([0, 1, 0, 0]), np.zeros(n, dtype=int),
                    ak.Array([[], [], [0.2], []]),
                    ak.Array([[], [], [300.0], []]),
                    ak.Array([[], [], [0.0], []]),
                    ak.Array([[], [], [True], []]),
                    np.asarray([False, False, True, False]),
                    met_pt=np.asarray([250.0] * n),
                    met_trigger_mask=np.asarray([False, False, False, True]),
                    # Canonical 10-GeV production defaults must exclude low-pT SFs.
                )

        np.testing.assert_allclose(
            variations["nominal"],
            np.asarray([1.0, 1.0, 0.98, 0.9]),
            rtol=0.0,
            atol=1.0e-12,
        )
        for component in ("veto_electron_5to10", "loose_muon_5to10"):
            self.assertFalse(status["components"][component]["applied"])
            self.assertEqual(
                status["components"][component]["source"],
                "disabled_by_analysis_sf_configuration",
            )
            self.assertNotIn(f"{component}Up", variations)
            self.assertNotIn(f"{component}Down", variations)
        for component in ("met_trigger", "photon_trigger"):
            self.assertTrue(status["components"][component]["applied"])
            self.assertIn(f"{component}Up", variations)
            self.assertIn(f"{component}Down", variations)


if __name__ == "__main__":
    unittest.main()
