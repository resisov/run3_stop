"""Numerical control-flow tests only; these do not produce physics fit data."""
import math
import unittest
from TROTASR.workflows.continue_impact_profiles import distinct, endpoint, MAX_BRACKET_STEPS


def point(theta, nll):
    return dict(theta=theta, deltaNLL=nll)


class BracketTests(unittest.TestCase):
    def test_center_serialization_is_not_an_upper_seed(self):
        self.assertFalse(distinct(1.0000001192092896, 1.000000085542056))
        self.assertTrue(distinct(1.001, 1.))

    def test_supported_interior_crossing_never_calls_failed_boundary(self):
        center = point(1., 0.)
        points = [center, point(.75, 362.)]
        calls = []
        def crossing(inner, outer):
            calls.append((inner, outer))
            return point(.99, .5)
        result, boundary = endpoint(center, points, -1, [0,10],
            lambda value: self.fail('Unneeded theta=0 fit'), crossing)
        self.assertEqual(result['theta'], .99)
        self.assertFalse(boundary)
        self.assertEqual(len(calls), 1)

    def test_nearest_bracket_is_used(self):
        center = point(1.,0.)
        points = [point(4.,100.),point(2.2,.51),point(2.1,.49)]
        used = []
        result, boundary = endpoint(center,points,1,[0,10],
            lambda value:self.fail('Existing bracket available'),
            lambda a,b:used.append((a['theta'],b['theta'])) or point(2.15,.5))
        self.assertEqual(used,[(2.1,2.2)])
        self.assertFalse(boundary)

    def test_validated_physical_boundary_is_kept(self):
        result, boundary = endpoint(point(1.,0.),[point(0.,.1)],-1,[0,10],
            lambda value:self.fail(),lambda a,b:self.fail())
        self.assertEqual(result['theta'],0.)
        self.assertTrue(boundary)

    def test_failed_point_never_becomes_boundary(self):
        def fail(value):
            raise RuntimeError('conditional did not commit')
        with self.assertRaisesRegex(RuntimeError,'did not commit'):
            endpoint(point(1.,0.),[], -1,[0,10],fail,lambda a,b:self.fail())

    def test_interior_probes_stop_at_finite_budget(self):
        seen = []
        def evaluate(value):
            seen.append(value)
            self.assertGreater(value,0.)
            return point(value,.1)
        with self.assertRaisesRegex(ValueError,'twelve'):
            endpoint(point(1.,0.),[point(.25,.05)],-1,[0,10],evaluate,lambda a,b:self.fail())
        self.assertEqual(len(seen),MAX_BRACKET_STEPS)

    def test_upper_seed_cannot_collapse_to_float_epsilon(self):
        center=point(1.000000085542056,0.)
        seen=[]
        def evaluate(value):
            seen.append(value)
            return point(value,.5)
        result,boundary=endpoint(center,[point(1.0000001192092896,0.),point(.99,.5)],1,[0,10],
            evaluate,lambda a,b:self.fail())
        self.assertGreater(seen[0]-center['theta'],.009)
        self.assertFalse(boundary)

    def test_evaluated_crossing_uses_original_tolerance(self):
        result,boundary=endpoint(point(1.,0.),[],1,[0,10],lambda v:point(v,.5019),lambda a,b:self.fail())
        self.assertFalse(boundary)
        self.assertEqual(result['deltaNLL'],.5019)

    def test_non_crossing_outside_tolerance_is_not_accepted(self):
        calls=[]
        result,boundary=endpoint(point(1.,0.),[],1,[0,10],lambda v:point(v,.5021),
            lambda a,b:calls.append(b['deltaNLL']) or point(1.2,.5))
        self.assertEqual(calls,[.5021])
        self.assertFalse(boundary)

    def test_probe_stays_inside_upper_bound(self):
        seen=[]
        with self.assertRaisesRegex(ValueError,'twelve|resolution'):
            endpoint(point(9.,0.),[point(9.9,.1)],1,[0,10],
                lambda v:seen.append(v) or point(v,.1),lambda a,b:self.fail())
        self.assertTrue(all(9.9<v<10 for v in seen))


if __name__ == '__main__':
    unittest.main()
