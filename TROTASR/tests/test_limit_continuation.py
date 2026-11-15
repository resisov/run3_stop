"""Numerical policies used by the canonical finite recovery stage."""
import unittest
from TROTASR.workflows.limit_recovery import choose_range, command_for, target_nll


def block(rows):
    return 'Will search for NLL crossing by bisection\n'+''.join(
        'At r = %s: delta(nll) = %s\n'%(r,n) for r,n in rows)


class NumericalContinuationTests(unittest.TestCase):
    def test_exhausted_current_scan_expands_seed_not_limit(self):
        seed,evidence=choose_range([('refined',block([(20,.06786),(30,.14213),(39.84375,.23368)]),40)],40)
        self.assertGreater(seed,300)
        self.assertLess(seed,1000)
        self.assertEqual(len(evidence),1)
        self.assertEqual(evidence[0]['sampled_r'],39.84375)

    def test_adequate_range_does_not_expand_for_asimov_warning(self):
        text='Best fit of asimov dataset is at r = 0.027\n'+block([(10,6.66),(5,2.08),(4.76,1.918)])
        self.assertEqual(choose_range([('initial',text,20)],40),(40,[]))

    def test_partial_sequence_uses_only_measured_median(self):
        median=block([(20,.06786),(30,.14213),(39.84375,.23368)])
        for count in (2,3,4):
            text=median+block([(20,1e-15),(30,2e-15),(39,3e-15)])*(count-1)
            seed,evidence=choose_range([('refined',text,40)],40)
            self.assertEqual(seed,430)
            self.assertEqual(len(evidence),1)
            self.assertEqual(evidence[0]['quantile'],.5)
            self.assertEqual(evidence[0]['search_scope'],'median_only_from_incomplete_sequence')

    def test_partial_without_valid_median_does_not_use_later_blocks(self):
        text=block([])+block([(20,.06786),(30,.14213),(39.84375,.23368)])
        self.assertEqual(choose_range([('refined',text,40)],40),(40,[]))

    def test_extra_searches_are_not_assigned_quantiles(self):
        text=block([(20,.06786),(30,.14213),(39.84375,.23368)])*6
        self.assertEqual(choose_range([('refined',text,40)],40),(40,[]))

    def test_no_scan_is_not_invented_range_evidence(self):
        self.assertEqual(choose_range([('refined','failed NaN',40)],40),(40,[]))

    def test_no_boundary_samples_no_extrapolation(self):
        self.assertEqual(choose_range([('a',block([(1,.01),(2,.02),(3,.03)]),40)],40),(40,[]))

    def test_zero_nll_no_division_by_zero(self):
        self.assertEqual(choose_range([('a',block([(20,0),(30,0),(39,0)]),40)],40),(40,[]))

    def test_measured_crossing_prevents_range_failure_claim(self):
        self.assertEqual(choose_range([('a',block([(20,.1),(30,3),(39,.5)]),40)],40),(40,[]))

    def test_finite_cap_and_invalid_range(self):
        with self.assertRaises(ValueError):choose_range([('a',block([(20,1e-12),(30,2e-12),(39,3e-12)]),40)],40)
        with self.assertRaises(ValueError):choose_range([],float('nan'))
        with self.assertRaises(ValueError):choose_range([],0)

    def test_target_matches_median_convention(self):
        self.assertAlmostEqual(target_nll(.5),1.920729410347062,places=12)
