import math
import unittest

from workflow.reference_trigger_counts import reduce_counts


class TriggerEfficiencyTest(unittest.TestCase):
    def test_unweighted_data_and_weighted_mc(self):
        data = {"total": [100], "passed": [98], "sumw_total": [100], "sumw_passed": [98],
                "sumw2_total": [100], "sumw2_passed": [98]}
        mc = {"total": [100], "passed": [95], "sumw_total": [100], "sumw_passed": [95],
              "sumw2_total": [100], "sumw2_passed": [95]}
        item = reduce_counts(data, mc)[0]
        self.assertAlmostEqual(item["data_efficiency"], 0.98)
        self.assertAlmostEqual(item["mc_effective_entries"], 100)
        self.assertAlmostEqual(item["scale_factor"], 0.98 / 0.95)
        self.assertTrue(item["valid"])

    def test_invalid_weighted_bin_is_not_used(self):
        data = {"total": [1], "passed": [1], "sumw_total": [1], "sumw_passed": [1],
                "sumw2_total": [1], "sumw2_passed": [1]}
        mc = {"total": [1], "passed": [0], "sumw_total": [-1], "sumw_passed": [0],
              "sumw2_total": [2], "sumw2_passed": [0]}
        item = reduce_counts(data, mc)[0]
        self.assertFalse(item["valid"])
        self.assertTrue(math.isnan(item["scale_factor"]))

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            reduce_counts(
                {"total": [1], "passed": [1], "sumw_total": [1], "sumw_passed": [1],
                 "sumw2_total": [1], "sumw2_passed": [1]},
                {"total": [], "passed": [], "sumw_total": [], "sumw_passed": [],
                 "sumw2_total": [], "sumw2_passed": []},
            )


if __name__ == "__main__":
    unittest.main()
