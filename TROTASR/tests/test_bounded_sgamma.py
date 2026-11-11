import ast
import importlib.util
import math
import struct
from pathlib import Path
import unittest
from unittest.mock import patch

source = Path(__file__).with_name('recover_impacts.py')
if not source.exists():
    source = Path(__file__).resolve().parents[1]/'workflows/recover_impacts.py'
spec = importlib.util.spec_from_file_location('TROTASR.workflows.bounded_test',source)
mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from TROTASR.workflows.profile_sgamma import fixed_command

NAME='CMS_NPS26012_sgamma_shape_lowdm_Nb1_bin1_2024'
def p(x,nll,r=1.):return dict(status='complete',theta=x,deltaNLL=nll,r=r)

class BoundedTests(unittest.TestCase):
    def test_earlier_upper_endpoint_retained_as_seed(self):
        rows=lambda upper:[dict(theta=.928,r=1.,deltaNLL=0.),dict(theta=.928,r=1.,deltaNLL=0.),dict(theta=upper,r=1.,deltaNLL=0.)]
        s=mod.saved_sgamma_seeds([rows(1.89),rows(.928)],[.01,5.])
        self.assertEqual(s['upper'],1.89)
        self.assertTrue(s['saved_endpoints_are_seeds_only'])
        self.assertIsNone(mod.saved_sgamma_seeds([rows(.928)],[.01,5.])['upper'])
        with self.assertRaises(ValueError):mod.saved_sgamma_seeds([], [.01,5.])

    def test_exact_binary32_endpoint_reconciliation(self):
        rounded=struct.unpack('f',struct.pack('f',.01))[0]
        self.assertLess(rounded,.01)
        self.assertEqual(mod.reconcile_seed_bounds({'x':rounded},{'x':[.01,5.]}),{'x':.01})
        self.assertEqual(mod.reconcile_seed_bounds({'x':1.2},{'x':[.01,5.]}),{'x':1.2})
        for value in (0.,.009999,5.000001,float('nan')):
            with self.assertRaises(ValueError):mod.reconcile_seed_bounds({'x':value},{'x':[.01,5.]})

    def test_bounded_receipt_namespace(self):
        out=Path('/campaign/bounded_sgamma_20260930')
        self.assertEqual((out/'points'/NAME/'key').parents[2],out)

    def test_exact_scope(self):
        parent=dict(status='needs_attention',failed_nuisances=[NAME],nuisance_count=2)
        records={NAME:dict(status='failed'),'other':dict(status='complete')}
        self.assertEqual(mod.bounded_sgamma_scope(parent,records,{NAME:[.01,5.]}),[NAME])
        for bounds in ([0,10],[0,5],[.01,10],[.02,5]):
            with self.assertRaises(ValueError):mod.bounded_sgamma_scope(parent,records,{NAME:bounds})

    def test_no_other_failure_or_success_recovery(self):
        for name,status in [('CMS_other','failed'),(NAME,'complete')]:
            with self.assertRaises(ValueError):
                mod.bounded_sgamma_scope(dict(status='needs_attention',failed_nuisances=[name],nuisance_count=1),
                    {name:dict(status=status)},{name:[.01,5.]})

    def test_same_bound_qcd_failure_uses_existing_profile(self):
        name='CMS_NPS26012_qcd_norm_highdm_Nb1_bin2_2024'
        parent=dict(status='needs_attention',failed_nuisances=[NAME,name],nuisance_count=3)
        records={NAME:dict(status='failed'),name:dict(status='failed'),'other':dict(status='complete')}
        self.assertEqual(mod.bounded_sgamma_scope(parent,records,{NAME:[.01,5.],name:[.01,5.]}),[NAME,name])
        with self.assertRaises(ValueError):
            mod.bounded_sgamma_scope(parent,records,{NAME:[.01,5.],name:[0.,10.]})

    def test_chain_within_current_range(self):
        for center in (.011,.9,1.,4.99):
            values=mod.bounded_lower_values(center,[.01,5.])
            self.assertEqual(len(values),4);self.assertEqual(values[-1],.01)
            self.assertTrue(all(.01<=v<center for v in values))
            self.assertTrue(all(a>b for a,b in zip(values,values[1:])))

    def test_invalid_center_rejected(self):
        for center in (0,.01,5,10,float('nan')):
            with self.assertRaises(ValueError):mod.bounded_lower_values(center,[.01,5.])

    def test_independent_boundary_agreement(self):
        mod.bounded_boundary_pair(p(.01,.1),p(.009999999776,.1001),.01)
        for other in (p(0.,.1),p(.01,.51),p(.01,.11),p(.01,.1,1.001),dict(p(.01,.1),status='failed')):
            with self.assertRaises(ValueError):mod.bounded_boundary_pair(p(.01,.1),other,.01)

    def test_likelihood_agreement_retains_r_as_diagnostic(self):
        mod.bounded_boundary_pair(p(.01,.1),p(.01,.1004,1.002),.01,r_tolerance=None)
        for other in (p(.01,.11,1.002),p(.01,.51),dict(p(.01,.1),status='failed')):
            with self.assertRaises(ValueError):
                mod.bounded_boundary_pair(p(.01,.1),other,.01,r_tolerance=None)

    def test_fixed_command_preserves_data_and_common_minimum(self):
        c=fixed_command(dict(snapshot='same.root',asimov='same_asimov.root'),NAME,.01,'_test',
            extra=['--cminDefaultMinimizerStrategy','1'],seeds={NAME:1.,'r':1.,'other':.7})
        expected={'--snapshotName':'impact_initial','--toysFile':'same_asimov.root',
            '--setParameterRanges':'r=0,20','--expectSignal':'1','--floatOtherPOIs':'1',
            '--cminDefaultMinimizerTolerance':'0.0001','--cminDefaultMinimizerStrategy':'1'}
        for k,v in expected.items():self.assertEqual(c[c.index(k)+1],v)
        self.assertEqual(c.count('--cminDefaultMinimizerStrategy'),1)
        self.assertIn('MINIMIZER_MaxCalls=2000000',c)
        self.assertIn('MINIMIZER_no_analytic',c)
        self.assertNotIn('--freezeParameters',c)
        self.assertIn(NAME+'=0.01',c[c.index('--fixedPointPOIs')+1])

    def test_exact_crossing_math_unchanged(self):
        self.assertEqual(mod.BoundedSgamma.crossing,mod.legacy.Recovery.crossing)
        self.assertEqual(mod.legacy.TARGET,.5);self.assertEqual(mod.legacy.NLL_TOL,.002)

    def test_fixed_range_guard_before_execution(self):
        obj=mod.BoundedSgamma.__new__(mod.BoundedSgamma)
        obj.inputs=dict(parameter_bounds={NAME:[.01,5.]})
        for value in (0,5.1,float('nan')):
            with self.assertRaises(ValueError):obj.fixed(NAME,value)


if __name__=='__main__':unittest.main()
