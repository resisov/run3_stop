"""Synthetic layout fixtures only: these do not define physical SR binning."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.utils.plot_adapter import blocks, category_label, full_sr_layout
from TROTASR.workflows.plotting_histogram import main


class FullSRLayoutTests(unittest.TestCase):
    def setUp(self):
        edges = [250, 300, 350, 400, 500, 800, 1500]
        self.keys = ['1,0,0,1,0', '1,0,1,0,0', '2,1,0,0,0']
        self.bin_map = dict(schema='trotasr_full_sr_bin_map_v1', scope='full',
                            mode='highdm', region='SR', observable='recoil',
                            category_axis_order=['Nb', 'Nbst', 'Nmix', 'Nres', 'Nw'],
                            categories=[dict(key=k, edges=edges[:]) for k in self.keys],
                            total_bins=18)
        self.leaf = dict(edges=edges[:], sumw=[0, 2, 0, 0, 0, 0],
                         sumw2=[0, 4, 0, 0, 0, 0])
        self.categories = {self.keys[1]: {'ST': {'recoil': self.leaf}}}
        self.payload = dict(histograms={'highdm': {'nominal': {'SR': self.categories}}})

    def build(self):
        return blocks(self.payload, 'highdm', 'SR', 'recoil', bin_map=self.bin_map)

    def test_requires_full_map_even_for_one_file(self):
        with self.assertRaisesRegex(ValueError, 'explicit full bin map'):
            blocks(self.payload, 'highdm', 'SR', 'recoil')

    def test_empty_categories_and_recoil_bins_preserved(self):
        result = self.build()
        self.assertEqual([b['nbin'] for b in result], [6, 6, 6])
        self.assertEqual(sum(b['nbin'] for b in result), self.bin_map['total_bins'])
        self.assertEqual([b['label'] for b in result],
                         [category_label('highdm', 'SR', k) for k in self.keys])
        np.testing.assert_array_equal(result[0]['background'], np.zeros(6))
        np.testing.assert_array_equal(result[1]['background'], self.leaf['sumw'])
        np.testing.assert_array_equal(result[2]['background'], np.zeros(6))
        self.assertTrue(all(b['blind_data'] for b in result))

    def test_completely_empty_input_keeps_full_map(self):
        self.categories.clear()
        result = self.build()
        self.assertEqual(sum(b['nbin'] for b in result), 18)
        self.assertTrue(all(np.all(b['background'] == 0) for b in result))

    def test_unknown_category_is_not_silently_dropped(self):
        self.categories['2,0,0,0,1'] = {'ST': {'recoil': copy.deepcopy(self.leaf)}}
        with self.assertRaisesRegex(ValueError, 'outside the full bin map'):
            self.build()

    def test_no_automatic_rebinning(self):
        self.leaf['edges'][1] = 310
        with self.assertRaisesRegex(ValueError, 'automatic rebinning is forbidden'):
            self.build()

    def test_no_array_truncation_or_broadcast(self):
        self.leaf['sumw'] = [2]
        with self.assertRaisesRegex(ValueError, 'histogram array'):
            self.build()

    def test_missing_observable_is_not_silently_dropped(self):
        self.categories[self.keys[1]]['ST'].clear()
        with self.assertRaisesRegex(ValueError, 'Missing SR observable'):
            self.build()

    def test_rejects_duplicate_or_malformed_map(self):
        for change in ('duplicate', 'order', 'total', 'subset', 'edges', 'axis'):
            altered = copy.deepcopy(self.bin_map)
            if change == 'duplicate': altered['categories'].append(altered['categories'][0])
            if change == 'order': altered['categories'].reverse()
            if change == 'total': altered['total_bins'] = 6
            if change == 'subset': altered['scope'] = 'observed_only'
            if change == 'edges': altered['categories'][0]['edges'] = [300, 250]
            if change == 'axis': altered['category_axis_order'][2:4] = ['Nres', 'Nmix']
            with self.subTest(change=change), self.assertRaises(ValueError):
                full_sr_layout(altered, 'recoil')

    def test_cli_rejects_missing_map_before_reading_input(self):
        args = ['plotting_histogram', '--input', '/not-read.json', '--output', '/not-created',
                '--mode', 'highdm', '--region', 'SR']
        with patch('sys.argv', args), patch('sys.stderr'), \
                patch('TROTASR.workflows.plotting_histogram.read_json') as read:
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
