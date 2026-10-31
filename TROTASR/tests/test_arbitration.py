"""Exact, deterministic user-authorized test policy; no production adoption."""
import unittest
from TROTASR.utils.event_selections import arbitrate, CATEGORY_ORDER


class ArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.fp = {0: {0, 1}, 1: {2, 3}, 2: {4, 5}, 3: {6, 7}}
        self.m = dict(index=0, fat=1, jets=[8], score=.95)
        self.r = dict(index=0, jets=[8, 9, 10], score=.99)

    def test_order(self):
        self.assertEqual(CATEGORY_ORDER, ('Nb', 'Nbst', 'Nmix', 'Nres', 'Nw'))

    def test_boosted_beats_mixed(self):
        s = arbitrate([1], [self.m], [], [], self.fp)
        self.assertEqual(s['B'], [1]); self.assertEqual(s['M'], [])

    def test_mixed_beats_resolved(self):
        s = arbitrate([], [self.m], [self.r], [], self.fp)
        self.assertEqual(len(s['M']), 1); self.assertEqual(s['R'], [])

    def test_shared_mixed_w_rejected_independent_retained(self):
        s = arbitrate([], [self.m], [], [1, 2], self.fp)
        self.assertEqual(s['W'], [2]); self.assertEqual(len(s['M']), 1)

    def test_w_resolved_overlap(self):
        r = dict(self.r, jets=[4, 9, 10])
        s = arbitrate([], [], [r], [1, 2], self.fp)
        self.assertEqual(s['W'], [1])

    def test_distinct_fat_shared_ak4(self):
        m = dict(self.m, jets=[0])
        self.assertEqual(arbitrate([0], [m], [], [], self.fp)['M'], [])

    def test_stable_competing_mixed(self):
        second = dict(self.m, index=2)
        for candidates in ([second, self.m], [self.m, second]):
            self.assertEqual(arbitrate([], candidates, [], [], self.fp)['M'][0]['index'], 0)

    def test_disjoint_coexistence(self):
        r = dict(self.r, jets=[20, 21, 22])
        s = arbitrate([0], [self.m], [r], [3], self.fp)
        self.assertEqual([len(s[x]) for x in ('B', 'M', 'R', 'W')], [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
