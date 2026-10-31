import unittest
from TROTASR.workflows.profile_sgamma import fixed_command
from TROTASR.workflows.finalize_sgamma_profiles import validate_boundary_pair


class SgammaNumericsTests(unittest.TestCase):
    def command(self,**kwargs):
        return fixed_command(dict(snapshot='same.root',asimov='same_asimov.root'),
                             'theta',0.,'_test',**kwargs)

    def test_strategy_override_is_unique(self):
        c=self.command(extra=['--cminDefaultMinimizerStrategy','1'])
        self.assertEqual(c.count('--cminDefaultMinimizerStrategy'),1)
        self.assertEqual(c[c.index('--cminDefaultMinimizerStrategy')+1],'1')

    def test_only_initial_values_change(self):
        c=self.command(seeds={'theta':1.,'other':2.,'r':1.})
        self.assertEqual(c[c.index('--fixedPointPOIs')+1],'theta=0,other=2,r=1')
        self.assertEqual(c[c.index('-P')+1],'theta')
        self.assertEqual(c[c.index('--floatOtherPOIs')+1],'1')
        self.assertEqual(c[c.index('--snapshotName')+1],'impact_initial')
        self.assertNotIn('--freezeParameters',c)

    def test_bounds_data_and_tolerance_preserved(self):
        c=self.command()
        for key,value in [('--setParameterRanges','r=0,20'),('--toysFile','same_asimov.root'),
                          ('--cminDefaultMinimizerTolerance','0.0001')]:
            self.assertEqual(c[c.index(key)+1],value)

    def test_boundary_requires_two_agreeing_converged_points(self):
        good=dict(status='complete',theta=0.,deltaNLL=.068,r=1.)
        validate_boundary_pair(good,dict(good,deltaNLL=.06801))
        for change in [dict(status='failed'),dict(theta=.01),dict(deltaNLL=.51),
                       dict(deltaNLL=.1),dict(r=1.001)]:
            with self.assertRaises(ValueError):validate_boundary_pair(good,dict(good,**change))
