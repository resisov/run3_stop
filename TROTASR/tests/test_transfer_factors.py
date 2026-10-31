import copy
import json
import unittest
import numpy as np
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT
from TROTASR.utils.transfer_factors import build_transfer_factors, recoil_ratio, score_ratio
from test_statistics_inputs import fixture


class TransferFactorTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT/'gnn4lowdm/config.json')
        self.payload = fixture()[0]

    def test_high_process_denominators_are_not_combined_by_rate_parameter(self):
        out = build_transfer_factors(self.payload, self.config)
        top = out['highdm']['records']['top_llcr']['Nb1']
        # ST+TT=2 in LLCR; W=1 is NOT added although alpha_LL is shared.
        self.assertEqual(top['denominator'], [2.]*8)
        self.assertEqual(top['transfer_factor'], [.5]*8)
        self.assertEqual(out['highdm']['records']['w_llcr']['Nb1']['denominator'], [1.]*8)
        json.dumps(out, allow_nan=False)

    def test_high_topology_is_pooled_only_for_nb_recoil_tf_display(self):
        out = build_transfer_factors(self.payload, self.config)
        self.assertEqual(out['highdm']['records']['top_llcr']['Nb2plus']['numerator'], [5.]*8)
        self.assertEqual(out['highdm']['records']['top_llcr']['Nb2plus']['numerator_sumw2'], [13.]*8)

    def test_low_parent_has_all_score_bins_and_all_nisr(self):
        out = build_transfer_factors(self.payload, self.config)
        records = out['lowdm']['records']['top_llcr']
        for category in self.config['sr_binning']['category_labels']:
            self.assertEqual(records[category]['denominator_total'], 70.)
            self.assertEqual(records[category]['transfer_factor'], [.1]*5)
            self.assertNotIn('NISR', records[category]['parent'])

    def test_shared_denominator_correlates_different_nisr_categories(self):
        out = build_transfer_factors(self.payload, self.config)
        joint = np.asarray(out['lowdm']['shared_parent_covariance']['top_llcr']['Nb1']['covariance'])
        self.assertEqual(joint.shape, (15,15))
        self.assertGreater(joint[0,5], 0.)
        expected = 7.*7.*70./70.**4
        self.assertAlmostEqual(joint[0,5], expected)
        np.testing.assert_allclose(joint[:5,:5], out['lowdm']['records']['top_llcr']['Nb1_NISR0']['covariance'])

    def test_raw_zgamma_factor_has_no_measured_correction(self):
        out = build_transfer_factors(self.payload, self.config)
        self.assertEqual(out['lowdm']['records']['zinv_gcr']['Nb1_NISR0']['transfer_factor'], [.2]*5)
        self.assertFalse(out['extra_weight_applied'])

    def test_legacy_high_ratio_and_stat_formula(self):
        r = recoil_ratio(np.array([2.,0.]), np.array([4.,0.]), np.array([4.,0.]), np.array([16.,0.]))
        self.assertAlmostEqual(r['transfer_factor'][0], .5)
        self.assertAlmostEqual(r['mcstat'][0]**2, .5)
        self.assertEqual(r['valid'].tolist(), [True,False])

    def test_legacy_low_signed_and_invalid_denominator_are_visible(self):
        r = score_ratio(np.array([-1.,0.]), np.array([1.,0.]), np.array([2.,2.]), np.array([4.,4.]))
        self.assertEqual(r['status'], 'signed_mc_bins')
        self.assertEqual(r['valid'].tolist(), [False,True])
        self.assertEqual(score_ratio(np.ones(2),np.ones(2),np.zeros(2),np.zeros(2))['status'],'invalid_denominator')

    def test_blinding_edges_and_missing_coverage_fail_explicitly(self):
        p = copy.deepcopy(self.payload)
        p['histograms']['lowdm']['nominal']['SR']['Nb1_NISR0']['data'] = {}
        with self.assertRaisesRegex(ValueError, 'SR data'):
            build_transfer_factors(p, self.config)
        p = copy.deepcopy(self.payload)
        p['histograms']['highdm']['nominal']['SR']['1,0,0,1,0']['ST']['recoil']['edges'][1] += 1
        with self.assertRaisesRegex(ValueError, 'edges differ'):
            build_transfer_factors(p, self.config)
        out = build_transfer_factors(self.payload, self.config)
        self.assertEqual(out['lowdm']['records']['w_llcr']['Nb1_NISR0']['status'], 'missing_process_coverage')


if __name__ == '__main__':
    unittest.main()
