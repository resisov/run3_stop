import copy
import importlib.util
import unittest
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json

AVAILABLE=importlib.util.find_spec('pdfplumber') is not None
if AVAILABLE:
    from TROTASR.workflows.verify_trigger_reference_figures import affine,verify_pdf


@unittest.skipUnless(AVAILABLE,'Local read-only PDF binding dependency unavailable')
class TriggerBindingTest(unittest.TestCase):
    def setUp(self):
        self.figure=ROOT/'plots/reference_assets_20260923/figures/trigger_efficiencies/met_trigger_scale_factor_2024.pdf'
        if not self.figure.exists():self.skipTest('Preserved PDF available on local QA host only')
        self.result=read_json(ROOT/'validation/reference_assets_20260923/provenance/triggers/2024_met_trigger_result.json')

    def test_actual_reference(self):
        r=verify_pdf(self.figure,self.result,'met','scale_factor')
        self.assertEqual(r['points_checked'],18)
        self.assertLess(r['max_residual_pdf_points'],1e-6)

    def test_changed_measurement_rejected(self):
        changed=copy.deepcopy(self.result);changed['bins'][0]['scale_factor']+=.01
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            verify_pdf(self.figure,changed,'met','scale_factor')

    def test_changed_uncertainty_rejected(self):
        changed=copy.deepcopy(self.result);changed['bins'][0]['scale_factor_uncertainty']*=2
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            verify_pdf(self.figure,changed,'met','scale_factor')

    def test_changed_bin_extent_rejected(self):
        changed=copy.deepcopy(self.result);changed['bin_edges_gev'][0]-=1
        with self.assertRaisesRegex(ValueError,'Missing PDF marker'):
            verify_pdf(self.figure,changed,'met','scale_factor')

    def test_nonlinear_or_sparse_ticks_rejected(self):
        with self.assertRaises(ValueError):affine([(0,0),(1,10),(2,21)])
        with self.assertRaises(ValueError):affine([(0,0),(1,10)])


if __name__=='__main__':unittest.main()
