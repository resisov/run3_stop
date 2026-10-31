import copy
import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from TROTASR.utils.systematic_views import anchored_endpoints
from TROTASR.utils.shape_kinematics import jet_block, met_contribution
from TROTASR.utils.jme_calibration import ENDPOINTS


class SystematicViewsTests(unittest.TestCase):
    def example(self):
        a = {}
        v = {s:{} for s in ENDPOINTS}
        for c in ('jet','fatjet'):
            a[c+'_corrected_pt'] = ak.Array([[100.,0.]])
            a[c+'_corrected_mass'] = ak.Array([[10.,0.]])
            a[c+'_eta_all'] = ak.Array([[0.,2.]])
            for s in ENDPOINTS:
                sign = 1 if s.endswith('Up') else -1 if s.endswith('Down') else 0
                for field, nom in [('pt',[100.000001,.01/np.cosh(2.)]),('mass',[10.000001,0.])]:
                    v[s][c+'_corrected_'+field] = ak.Array([[x*(1+.1*sign) for x in nom]])
        return ak.Array(a),v

    def test_storage_precision_and_floor(self):
        a,v = self.example()
        with patch('TROTASR.utils.systematic_views.calibrated_endpoints',return_value=v):
            out=anchored_endpoints(a,2024)
        self.assertEqual(float(out['nominal']['jet_corrected_pt'][0,0]),100.)
        self.assertAlmostEqual(float(out['nominal']['jet_corrected_pt'][0,1]),.01/np.cosh(2.))
        self.assertAlmostEqual(float(out['jesTotalUp']['jet_corrected_pt'][0,0]),110.0000001)

    def test_unexplained_calibration_rejected(self):
        a,v=self.example()
        a=ak.with_field(a,ak.Array([[101.,0.]]),'jet_corrected_pt')
        with patch('TROTASR.utils.systematic_views.calibrated_endpoints',return_value=v):
            with self.assertRaisesRegex(ValueError,'Stored calibration'):
                anchored_endpoints(a,2024)

    def test_positive_subfloor_uses_physical_endpoints_without_double_offset(self):
        a,v=self.example()
        pt=.01/np.cosh(2.)
        a=ak.with_field(a,ak.Array([[100.,pt*.2]]),'jet_corrected_pt')
        audit={}
        with patch('TROTASR.utils.systematic_views.calibrated_endpoints',return_value=copy.deepcopy(v)):
            out=anchored_endpoints(a,2024,audit)
        self.assertEqual(audit['jme_positive_subfloor_jet'],1)
        for shift in ENDPOINTS:
            self.assertEqual(float(out[shift]['jet_corrected_pt'][0,1]),float(v[shift]['jet_corrected_pt'][0,1]))

    def test_already_accepted_positive_floor_stays_byte_exact(self):
        a,v=self.example()
        pt=.01/np.cosh(2.)
        stored=pt-1e-6
        a=ak.with_field(a,ak.Array([[100.,stored]]),'jet_corrected_pt')
        with patch('TROTASR.utils.systematic_views.calibrated_endpoints',return_value=v):
            out=anchored_endpoints(a,2024)
        self.assertEqual(float(out['nominal']['jet_corrected_pt'][0,1]),stored)

    def test_nonproportional_subfloor_is_not_a_blanket_tolerance(self):
        a,v=self.example()
        a=ak.with_field(a,ak.Array([[100.,.0001]]),'jet_corrected_pt')
        a=ak.with_field(a,ak.Array([[10.,.0003]]),'jet_corrected_mass')
        with patch('TROTASR.utils.systematic_views.calibrated_endpoints',return_value=v):
            with self.assertRaisesRegex(ValueError,'Stored calibration'):
                anchored_endpoints(a,2024)

    def test_type1_threshold_migration(self):
        a=ak.Array(dict(jet_eta_all=[[0.]],jet_phi_all=[[0.]],jet_ch_em_ef=[[0.]],
            jet_ne_em_ef=[[0.]],jet_nanoaod_pt=[[20.]],jet_muon_subtr_factor=[[.1]]))
        self.assertEqual(float(met_contribution(a,ak.Array([[14.]]))[0][0]),0.)
        self.assertAlmostEqual(float(met_contribution(a,ak.Array([[16.]]))[0][0]),3.6)

    def test_angular_predicates_preserve_array_order(self):
        pt=ak.Array([[50.,200.,80.,40.]])
        eta=ak.zeros_like(pt);phi=ak.Array([[1.,2.,2.5,.4]])
        good=ak.ones_like(pt,dtype=bool)
        b=jet_block(pt,eta,phi,good,good,np.array([0.]))
        self.assertEqual(float(b['j1pt'][0]),50.)
        self.assertTrue(b['open_pre'][0]);self.assertFalse(b['open_high'][0])
        self.assertTrue(b['qcd_open'][0]);self.assertFalse(b['dphi123_0p1'][0])


if __name__=='__main__':
    unittest.main()
