import copy
import struct
import unittest
from unittest.mock import patch
from TROTASR.workflows.limit_serialization import rounding_interval, reconcile_crossings, validate_saved_limit


class SerializationTests(unittest.TestCase):
    def fixture(self):
        value=struct.unpack('>f',struct.pack('>f',103.069119))[0]
        crossings=[dict(quantile=q,result=value,target_delta_nll=6.541562677718649,passed=False)
                   for q in (.5,.025,.16,.84,.975)]
        text=('Will search for NLL crossing by bisection\n'
              'At r = 103.069119: delta(nll) = 6.54469\n')*5
        return text,dict(crossings=crossings,quantiles={str(q):value for q in (.5,.025,.16,.84,.975)})

    def test_actual_float_return_and_print_interval(self):
        value=103.06912231445312
        low,high=rounding_interval(value)
        self.assertLess(low,103.069119)
        self.assertGreater(high,103.069119)
        self.assertLess(high-low,.00001)
        self.assertGreater(abs(value-103.069119),.000002)

    def test_only_representation_match_is_added_original_kept(self):
        text,ev=self.fixture();before=copy.deepcopy(ev)
        passed,result=reconcile_crossings(text,ev,.1)
        self.assertTrue(passed);self.assertEqual(ev,before)
        self.assertTrue(all(c['matching_scan_rows'] for c in result['crossings']))

    def test_stale_value_not_accepted(self):
        text,ev=self.fixture()
        self.assertFalse(reconcile_crossings(text.replace('103.069119','103.0691'),ev,.1)[0])

    def test_nll_stopping_tolerance_unchanged(self):
        text,ev=self.fixture()
        self.assertFalse(reconcile_crossings(text.replace('6.54469','6.55'),ev,.1)[0])
        self.assertFalse(reconcile_crossings(text,ev,.01)[0])

    def test_missing_search_rejected_warning_is_diagnostic_only(self):
        text,ev=self.fixture()
        self.assertFalse(reconcile_crossings(text.split('Will search',1)[1],ev,.1)[0])
        self.assertTrue(reconcile_crossings('Best fit of asimov dataset is at r = .1\n'+text,ev,.1)[0])

    def test_asimov_warning_and_large_r_are_retained_without_veto(self):
        from TROTASR.workflows.limit_validation import asimov_diagnostics
        result = asimov_diagnostics('Best fit of asimov dataset is at r = 3\n'
            'Fit to asimov dataset:\n r 0 3 +/- 0.2\nWill search for NLL crossing by bisection')
        self.assertEqual(result['fitted_r'], 3.)
        self.assertEqual(len(result['warning_lines']), 1)
        self.assertIsNone(result['acceptance_threshold'])
        self.assertTrue(result['diagnostic_only'])

    def test_nonrepresentable_and_invalid_values_rejected(self):
        for value in (0,-1,float('nan'),float('inf'),103.069119):
            with self.assertRaises(ValueError):rounding_interval(value)

    def test_original_quantile_and_warning_gates_cannot_be_bypassed(self):
        for reason in ('not five expected quantiles','nonzero background-only Asimov best fit warning'):
            with patch('TROTASR.workflows.limit_serialization.validate',return_value=(False,dict(reason=reason))):
                self.assertEqual(validate_saved_limit(None,None,.1),(False,dict(reason=reason)))

    def test_already_valid_results_unchanged(self):
        with patch('TROTASR.workflows.limit_serialization.validate',return_value=(True,dict(valid=True))):
            self.assertEqual(validate_saved_limit(None,None,.1),(True,dict(valid=True)))


if __name__=='__main__':unittest.main()
