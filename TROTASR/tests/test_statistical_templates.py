"""Synthetic contract tests only; never full-workflow or physical-limit evidence."""
import copy
import unittest
from TROTASR.utils.statistical_templates import assemble, card_text, add_signal_xsec_lnN


def component(value, variance=1., variations=None):
    return dict(nominal=[value], sumw2=[variance], variations=variations or {})


def bin_record(name, region, backgrounds, parameters):
    return dict(name=name, region=region, regime='highdm', observation=None if region == 'SR' else 20.,
                backgrounds=backgrounds, rate_params=parameters,
                rate_initial={p: 1. for p in backgrounds}, extra_lnN={}, physical={})


class NativeTemplateTests(unittest.TestCase):
    def setUp(self):
        self.nb1 = 'CMS_NPS26012_ll_highdm_Nb1_bin1_2024'
        self.nb2 = 'CMS_NPS26012_ll_highdm_Nb2_bin1_2024'
        self.sr = bin_record('sr001', 'SR', {'Top_a': component(2), 'Top_b': component(3)},
                             {'Top_a': self.nb1, 'Top_b': self.nb2})
        self.sr['physical']['signal'] = dict(nominal=1., sumw2=0.1, variations={})
        self.cr = bin_record('cr001', 'LLCR', {'Top_a': component(8), 'Top_b': component(9)},
                             {'Top_a': self.nb1, 'Top_b': self.nb2})
        self.payload = {'year': '2024', 'groups': {'SR': [self.sr], 'LLCR': [self.cr]}}

    def test_pooled_sr_preserves_both_nb_normalizations(self):
        channels, data, clipped, issues = assemble([self.payload])
        self.assertFalse(issues)
        sr = [r for (c, _), r in data.items() if c.startswith('SR_') and r['family'] == 'Top']
        self.assertEqual({r['parameter'] for r in sr}, {self.nb1, self.nb2})
        self.assertEqual(sum(float(r['nominal'].sum()) for r in sr), 5.)
        self.assertEqual(sum(map(len, channels.values())), 2)

    def test_negative_policy_retains_variance_and_audit(self):
        self.sr['backgrounds']['Top_a'] = component(-2, 7)
        _, data, clipped, issues = assemble([self.payload])
        record = next(r for (c, _), r in data.items() if c.startswith('SR') and r['parameter'] == self.nb1)
        self.assertEqual(record['nominal'].tolist(), [0.])
        self.assertEqual(record['sumw2'].tolist(), [7.])
        self.assertEqual(clipped[0]['before'], -2.)
        self.assertTrue(issues)  # A zero-integral component must not be called ready.

    def test_shape_endpoint_kept(self):
        self.sr['backgrounds']['Top_a']['variations'] = {'JESTotal_2024': {'up': [4.], 'down': [1.]}}
        channels, data, _, _ = assemble([self.payload])
        text = card_text(channels, data)
        self.assertIn('JESTotal_2024 shape ', text)
        self.assertIn('* autoMCStats 10 1 1', text)

    def test_empty_bin_shape_migration_is_not_zero_integral_failure(self):
        self.sr['backgrounds']['Top_a'] = component(1., 1., {'JES': {'up': [.9], 'down': [1.]}})
        empty_bin = copy.deepcopy(self.sr)
        empty_bin['name'] = 'sr002'
        empty_bin['backgrounds']['Top_a'] = component(0., 0., {'JES': {'up': [.1], 'down': [0.]}})
        self.payload['groups']['SR'].append(empty_bin)
        channels, data, _, issues = assemble([self.payload], preserve_flat=True)
        self.assertFalse(issues)
        record = next(r for (c, _), r in data.items() if c.startswith('SR') and r['parameter'] == self.nb1)
        self.assertEqual(record['nominal'].tolist(), [1., 0.])
        self.assertEqual(record['variations']['JES']['up'].tolist(), [.9, .1])
        self.assertIn('* autoMCStats 10 1 1', card_text(channels, data))

    def test_cronly_excludes_sr(self):
        channels, data, _, _ = assemble([self.payload])
        text = card_text(channels, data, cr_only=True)
        self.assertNotIn('SR_highdm', text)
        self.assertIn('LLCR_highdm', text)

    def test_signal_theory_is_signal_only_once(self):
        channels, data, _, _ = assemble([self.payload])
        theory = {'records': {'1200': {'uncertainty_down_relative': 0.1, 'uncertainty_up_relative': 0.2}}}
        text = add_signal_xsec_lnN(card_text(channels, data), 1200, theory)
        line = next(l for l in text.splitlines() if l.startswith('CMS_NPS26012_cross_section_stop_13p6TeV '))
        self.assertEqual(line.split()[2:].count('0.9/1.2'), 1)
        with self.assertRaises(ValueError):
            add_signal_xsec_lnN(text, 1200, theory)

    def test_duplicate_bins_rejected(self):
        self.payload['groups']['SR'].append(copy.deepcopy(self.sr))
        with self.assertRaises(ValueError):
            assemble([self.payload])

    def test_invalid_observation_rejected(self):
        self.cr['observation'] = float('nan')
        with self.assertRaises(ValueError):
            assemble([self.payload])


if __name__ == '__main__':
    unittest.main()
