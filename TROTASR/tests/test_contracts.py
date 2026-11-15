import unittest
from TROTASR.utils.paths import ROOT, payload_path
from TROTASR.utils.plot_adapter import category_label


class Contracts(unittest.TestCase):


    def test_payloads_do_not_escape(self):
        for p in ('../../outside', '/tmp/outside'):
            with self.assertRaises(ValueError): payload_path(p)
        self.assertEqual(payload_path('scales/JME/2024/jet_jerc.json.gz'), ROOT / 'scales/JME/2024/jet_jerc.json.gz')

    def test_all_five_sr_axes_visible(self):
        label = category_label('highdm', 'SR', '2,1,1,0,0')
        positions = [label.index(token) for token in ('N_b', 'N_{bst}', 'N_{mix}', 'N_{res}', 'N_W')]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('N_W=0', label)

    def test_cr_not_topology_replicated(self):
        self.assertEqual(category_label('highdm', 'LLCR', 'Nb1'), r'$N_b=1$')
        self.assertNotIn('N_{mix}', category_label('lowdm', 'GCR', 'Nb2'))

    def test_adl_companions_present(self):
        for name in ('ids', 'corrections', 'event_selections'):
            self.assertTrue((ROOT / 'utils' / (name + '.adl')).stat().st_size > 100)


if __name__ == '__main__': unittest.main()
