import copy
import unittest
import numpy as np
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT
from TROTASR.utils.sr_binning import expand, project, NATIVE_EDGES
from TROTASR.utils.plot_adapter import blocks


class AdoptedSR162Tests(unittest.TestCase):
    def setUp(self):
        self.definition = read_json(ROOT / 'highdm_sr_binning.json')
        self.payload = dict(highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                            histograms={'highdm': {'nominal': {'SR': {}}}})

    def put(self, key, values, sample='ST'):
        self.payload['histograms']['highdm']['nominal']['SR'][key] = {
            sample: {'recoil': dict(edges=NATIVE_EDGES, sumw=values,
                                     sumw2=[v*v for v in values], entries=[1]*8)}}

    def test_fixed_28_categories_162_bins(self):
        layout = expand(self.definition)
        self.assertEqual(len(layout), 28)
        self.assertEqual(sum(len(c['edges'])-1 for c in layout), 162)

    def test_empty_input_still_has_162_bins(self):
        plotted = blocks(self.payload, 'highdm', 'SR', 'recoil', self.definition)
        self.assertEqual(sum(c['nbin'] for c in plotted), 162)
        self.assertTrue(all(c['blind_data'] for c in plotted))

    def test_nb2_and_nb3plus_not_confused(self):
        self.put('2,0,0,0,1', [1.]*8)
        self.put('3,0,0,0,1', [2.]*8)
        p, _ = project(self.payload, self.definition)
        actual = p['histograms']['highdm']['nominal']['SR']
        self.assertEqual(len(actual), 2)
        self.assertEqual(sorted(sum(r['ST']['recoil']['sumw']) for r in actual.values()), [8.,16.])

    def test_pooled_resolved_retains_yield_and_variance(self):
        self.put('1,0,0,1,0', [1.]*8)
        self.put('3,0,0,1,0', [2.]*8)
        p, _ = project(self.payload, self.definition)
        actual = p['histograms']['highdm']['nominal']['SR']
        self.assertEqual(len(actual), 1)
        leaf = next(iter(actual.values()))['ST']['recoil']
        self.assertEqual(sum(leaf['sumw']), 24.)
        self.assertEqual(sum(leaf['sumw2']), 40.)
        # Projection is a view product; exact Nb source components survive.
        self.assertEqual(len(self.payload['histograms']['highdm']['nominal']['SR']), 2)

    def test_five_bin_tail_is_exact_sum(self):
        self.put('2,0,2,0,0', list(range(1,9)))
        p, _ = project(self.payload, self.definition)
        leaf = next(iter(p['histograms']['highdm']['nominal']['SR'].values()))['ST']['recoil']
        self.assertEqual(leaf['sumw'], [1.,2.,3.,4.,26.])
        self.assertEqual(leaf['sumw2'][-1], 5**2+6**2+7**2+8**2)

    def test_mixed_and_independent_w_stay_separate(self):
        self.put('2,0,1,0,0', [1.]*8)
        self.put('2,0,1,0,1', [1.]*8)
        p, _ = project(self.payload, self.definition)
        self.assertEqual(len(p['histograms']['highdm']['nominal']['SR']), 2)

    def test_old_nb_pooling_cannot_be_relabelled(self):
        del self.payload['highdm_sr_source_axes']
        with self.assertRaisesRegex(ValueError, 'old Nb>=2'):
            project(self.payload, self.definition)

    def test_zero_topology_rejected(self):
        self.put('1,0,0,0,0', [1.]*8)
        with self.assertRaisesRegex(ValueError, 'Uncovered'):
            project(self.payload, self.definition)

    def test_high_multiplicity_is_covered(self):
        self.put('8,7,6,5,4', [1.]*8)
        p, _ = project(self.payload, self.definition)
        self.assertEqual(len(p['histograms']['highdm']['nominal']['SR']), 1)

    def test_sr_data_cannot_be_plotted(self):
        self.put('1,0,1,0,0', [1.]*8, sample='data')
        with self.assertRaisesRegex(ValueError, 'SR data'):
            blocks(self.payload, 'highdm', 'SR', 'recoil', self.definition)

    def test_t2tt_mass_key_is_not_misclassified_as_background(self):
        self.put('1,0,1,0,0', [1.]*8, sample='mStop1200_mLSP500')
        plotted = blocks(self.payload, 'highdm', 'SR', 'recoil', self.definition)
        self.assertEqual(sum(float(c['background'].sum()) for c in plotted), 0.)

    def test_wrong_native_edges_rejected(self):
        self.put('1,0,1,0,0', [1.]*8)
        leaf = self.payload['histograms']['highdm']['nominal']['SR']['1,0,1,0,0']['ST']['recoil']
        leaf['edges'] = [250,300,350,400,500,800,1500]
        with self.assertRaisesRegex(ValueError, 'native recoil'):
            project(self.payload, self.definition)

    def test_overlapping_physics_map_rejected(self):
        self.definition['topology_rules'][1]['Nres'] = {'ge': 1}
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            expand(self.definition)


if __name__ == '__main__':
    unittest.main()
