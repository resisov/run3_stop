import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from workflow.reference_trigger_counts import (
    add_pileup_uncertainty,
    empty_counts,
    fill_counts,
    mc_normalization_factors,
    pileup_weight_triplet,
    reduce_counts,
    serialise_counts,
)


class ReferenceTriggerCountsTest(unittest.TestCase):
    def test_2d_flattening_matches_correctionlib_axis_order(self):
        edges = [np.asarray([0.0, 1.0, 2.5]), np.asarray([200.0, 300.0, 500.0])]
        counts = empty_counts((2, 2))
        fill_counts(
            counts,
            [np.asarray([0.2, 0.2, 1.2, 1.2]), np.asarray([250, 350, 250, 350])],
            edges,
            np.asarray([True, False, True, False]),
            np.ones(4),
        )
        serialised = serialise_counts(counts)
        self.assertEqual(serialised["total"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(serialised["passed"], [1.0, 0.0, 1.0, 0.0])

    def test_reduction_returns_valid_sf(self):
        data = {key: [0.0] for key in ("total", "passed", "sumw_total", "sumw_passed", "sumw2_total", "sumw2_passed")}
        mc = {key: [0.0] for key in data}
        data.update({"total": [100.0], "passed": [98.0]})
        mc.update({"sumw_total": [100.0], "sumw_passed": [95.0], "sumw2_total": [100.0]})
        item = reduce_counts(data, mc)[0]
        self.assertTrue(item["valid"])
        self.assertAlmostEqual(item["scale_factor"], 0.98 / 0.95)

    def test_mc_normalization_aggregates_split_dataset_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            physical = "/QCD/Test/NANOAODSIM"
            for index, (dataset_id, sumw) in enumerate(((11, 200.0), (12, 300.0))):
                payload = {
                    "datasets": {
                        str(dataset_id): {
                            "dataset": f"QCD____{index}",
                            "dataset_id": dataset_id,
                            "physical_dataset": physical,
                            "process": "QCD",
                            "is_data": False,
                            "is_signal": False,
                            "xsec_pb": 100.0,
                        }
                    },
                    "files": [{
                        "dataset": f"QCD____{index}",
                        "file_path": f"file_{index}.root",
                        "read_status": "success",
                        "sumw": sumw,
                    }],
                }
                (directory / f"part_{index}.json").write_text(json.dumps(payload))
            factors, audit = mc_normalization_factors(directory)
            self.assertEqual(factors, {11: 0.2, 12: 0.2})
            self.assertEqual(audit["physical_datasets"][physical]["files"], 2)
            self.assertEqual(audit["physical_datasets"][physical]["sumw"], 500.0)

    def test_pileup_triplet_and_uncertainty(self):
        repo = Path(__file__).resolve().parents[2]
        nominal, up, down = pileup_weight_triplet(repo, np.asarray([10.0, 30.0, 50.0]))
        self.assertEqual(nominal.shape, (3,))
        self.assertTrue(np.all(np.isfinite(nominal)))
        bins = [{"scale_factor": 1.0, "scale_factor_uncertainty": 0.03, "valid": True}]
        bins_up = [{"scale_factor": 1.04, "valid": True}]
        bins_down = [{"scale_factor": 0.98, "valid": True}]
        add_pileup_uncertainty(bins, bins_up, bins_down)
        self.assertAlmostEqual(bins[0]["scale_factor_pileup_uncertainty"], 0.04)
        self.assertAlmostEqual(bins[0]["scale_factor_uncertainty"], 0.05)


if __name__ == "__main__":
    unittest.main()
