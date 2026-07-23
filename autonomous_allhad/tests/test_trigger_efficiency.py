import math
import unittest

from workflow.measure_trigger_efficiency import reduce_counts


class TriggerEfficiencyTest(unittest.TestCase):
    def test_unweighted_data_and_weighted_mc(self):
        payload = {"measurement": "test", "bin_edges_gev": [250, 300],
                   "data": [{"total": 100, "passed": 98}],
                   "mc": [{"sumw_total": 100, "sumw2_total": 100, "sumw_passed": 95}]}
        result = reduce_counts(payload)
        item = result["bins"][0]
        self.assertAlmostEqual(item["data"]["efficiency"], 0.98)
        self.assertAlmostEqual(item["mc"]["effective_entries"], 100)
        self.assertAlmostEqual(item["scale_factor"], 0.98 / 0.95)
        self.assertTrue(item["scale_factor_valid"])

    def test_invalid_weighted_bin_is_not_used(self):
        payload = {"bin_edges_gev": [250, 300], "data": [{"total": 1, "passed": 1}],
                   "mc": [{"sumw_total": -1, "sumw2_total": 2, "sumw_passed": 0}]}
        item = reduce_counts(payload)["bins"][0]
        self.assertFalse(item["mc"]["valid"])
        self.assertTrue(math.isnan(item["scale_factor"]))

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            reduce_counts({"bin_edges_gev": [1, 2], "data": [], "mc": []})


if __name__ == "__main__":
    unittest.main()
