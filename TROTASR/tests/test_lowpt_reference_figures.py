"""Read-only reference QA regression tests; no physics or plot modifications."""
import copy
import unittest
from unittest.mock import patch
from TROTASR.workflows.verify_lowpt_reference_figures import (
    compare_summary, verify_summary_measurement, verify_canonical_calibration,
    reference_summary_contract,
)


def one_point(y=1.0):
    return dict(points=[dict(bin=0, series='sf', present=True, x=6., xlo=5.,
                            xhi=7., y=y, lo=y-.02, hi=y+.02)],
                xaxis=dict(slope=90.), yaxis=dict(slope=500.))


def calibration_fixture():
    values = dict(nominal=[1.1]*4+[1.]*2, down=[1.08]*4+[1.]*2, up=[1.12]*4+[1.]*2)
    payload = dict(corrections=[dict(description='forward unity', data=dict(content=[
        dict(key=key, value=dict(content=rows)) for key, rows in values.items()]))])
    summary = [dict(bin=i, series='sf', present=True, x=6.+i, xlo=5.5+i, xhi=6.5+i,
                    y=y, lo=values['down'][i], hi=values['up'][i])
               for i, y in enumerate(values['nominal'])]
    efficiency = [dict(bin=i, series=series, present=True, y=y, lo=y-.02, hi=y+.02)
                  for i in range(4) for series, y in (('data', .88), ('mc', .8))]
    measurement = dict(bins=[dict(valid=i < 4, nominal={series: dict(
        efficiency=y, efficiency_stat_uncertainty=.02) for series, y in (('data', .88), ('mc', .8))})
        for i in range(6)])
    calibration = dict(year=2025, kind='electron', measurement=measurement)
    scale_figure = dict(year=2025, kind='electron', view='scale_factor',
        an_path='synthetic/electron_scale_factor.pdf',
        canonical_summary=dict(points=summary, xaxis=dict(slope=90.), yaxis=dict(slope=500.)),
        reference_scope=dict(kind='valid_measurement_only', definition_audit='synthetic_audit.json',
                             omitted_invalid_bins=[4, 5]))
    efficiency_figure = dict(year=2025, kind='electron', view='efficiency',
                             canonical_summary=dict(points=efficiency))
    return calibration, [scale_figure, efficiency_figure], payload


class LowPtReferenceTests(unittest.TestCase):
    def test_exact_summary(self):
        self.assertEqual(compare_summary(one_point(), one_point())['status'], 'verified')

    def test_changed_uncertainty_fails(self):
        actual = one_point(); actual['points'][0]['hi'] += .001
        self.assertEqual(compare_summary(actual, one_point())['status'], 'reference_numeric_review_required')

    def test_missing_policy_point_fails(self):
        actual = one_point(); actual['points'][0] = dict(bin=0, series='sf', present=False)
        self.assertEqual(compare_summary(actual, one_point())['problems'][0]['issue'], 'missing_policy_point')

    def test_rounded_historical_display_is_explicit(self):
        expected = one_point(1.00029844271056)
        expected['points'][0].update(lo=1.00029844271056-.00188627567,
                                     hi=1.00029844271056+.00188627567)
        actual = one_point(1.0003)
        actual['points'][0].update(lo=.9984, hi=1.0022)
        self.assertNotEqual(compare_summary(actual, expected)['status'], 'verified')
        self.assertEqual(compare_summary(actual, expected, 4)['status'], 'verified')

    def test_duplicate_identity_rejected(self):
        expected = one_point(); expected['points'].append(copy.deepcopy(expected['points'][0]))
        with self.assertRaises(ValueError): compare_summary(one_point(), expected)

    def test_calibration_binding_and_changed_payload(self):
        cal, figures, payload = calibration_fixture()
        check = verify_canonical_calibration(cal, figures, payload)
        self.assertEqual(check['canonical_sf_points_bound_to_payload'], list(range(6)))
        self.assertEqual(check['adopted_unity_policy_cells'], [4, 5])
        changed = copy.deepcopy(payload)
        changed['corrections'][0]['data']['content'][0]['value']['content'][0] += .01
        with self.assertRaisesRegex(ValueError, 'adopted payload differ'):
            verify_canonical_calibration(cal, figures, changed)

    def test_fit_result_binding_and_changed_measurement(self):
        cal, figures, _ = calibration_fixture()
        summary = figures[1]['canonical_summary']
        self.assertEqual(verify_summary_measurement(summary, cal['measurement'], 'efficiency'), 8)
        changed = copy.deepcopy(cal['measurement'])
        changed['bins'][0]['nominal']['data']['efficiency'] += .01
        with self.assertRaisesRegex(ValueError, 'differs from measurement'):
            verify_summary_measurement(summary, changed, 'efficiency')

    def scoped_summary(self, figure, calibration):
        definition = dict(measurement_only_figures=['synthetic/electron_scale_factor.pdf'],
                          scope='valid_measurement_only')
        with patch('TROTASR.workflows.verify_lowpt_reference_figures.read_json', return_value=definition):
            return reference_summary_contract(figure, calibration)

    def test_measurement_only_scope_requires_invalid_fit_evidence(self):
        cal, (fig, _), _ = calibration_fixture()
        expected, audit = self.scoped_summary(fig, cal)
        self.assertEqual(audit['omitted_invalid_fit_bins'], [4, 5])
        self.assertEqual(sum(p['present'] for p in expected['points']), 4)
        self.assertEqual(sum(p['present'] for p in fig['canonical_summary']['points']), 6)
        changed = copy.deepcopy(cal); changed['measurement']['bins'][4]['valid'] = True
        with self.assertRaisesRegex(ValueError, 'actual invalid'):
            self.scoped_summary(fig, changed)
        changed = copy.deepcopy(fig); changed['reference_scope']['omitted_invalid_bins'] = [0, 4, 5]
        with self.assertRaisesRegex(ValueError, 'actual invalid'):
            self.scoped_summary(changed, cal)

    def test_measurement_scope_does_not_hide_missing_valid_points(self):
        cal, (fig, _), _ = calibration_fixture()
        expected, _ = self.scoped_summary(fig, cal)
        actual = copy.deepcopy(expected); actual['points'][0]['present'] = False
        self.assertEqual(compare_summary(actual, expected)['status'], 'reference_numeric_review_required')


if __name__ == '__main__': unittest.main()
