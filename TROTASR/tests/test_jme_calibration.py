import math
import unittest
import awkward as ak
import numpy as np
from TROTASR.utils.jme_calibration import _jer_smear, _deterministic_normal, TAGS


class JmeCalibrationTests(unittest.TestCase):
    def smear(self, pt, mass, sf, gen=None):
        return np.asarray(ak.flatten(_jer_smear(ak.Array([[pt]]), ak.Array([[0.]]),
            ak.Array([[.1]]), ak.Array([[sf]]), ak.Array([[0 if gen is not None else -1]]),
            ak.Array([[] if gen is None else [gen]]), np.array([1]), np.array([2]), np.array([3]),
            'Jet', corrected_mass=ak.Array([[mass]]))))[0]

    def test_hybrid_matched(self):
        self.assertAlmostEqual(self.smear(100., 10., 1.2, 95.), 1.01)

    def test_zero_pt_unmodified(self):
        self.assertEqual(self.smear(0., 0., 1.2), 1.)

    def test_minimum_energy_not_minimum_pt(self):
        factor = self.smear(1., 3., 20., 1.2)
        self.assertAlmostEqual(factor, .01/math.sqrt(10.))

    def test_rng_reused_for_all_endpoints(self):
        self.assertEqual(_deterministic_normal(1, 2, 3, 'Jet', 0), _deterministic_normal(1, 2, 3, 'Jet', 0))
        self.assertNotEqual(_deterministic_normal(1, 2, 3, 'Jet', 0), _deterministic_normal(1, 2, 3, 'FatJet', 0))

    def test_negative_pt_rejected(self):
        with self.assertRaises(ValueError):
            self.smear(-1., 0., 1.)

    def test_only_current_years(self):
        self.assertEqual(set(TAGS), {'2022', '2022EE', '2023', '2023BPix', 2024, 2025})


if __name__ == '__main__':
    unittest.main()
