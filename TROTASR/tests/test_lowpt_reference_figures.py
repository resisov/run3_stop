"""Read-only reference QA regression tests; no physics or plot modifications."""
import copy
import json
import unittest
from pathlib import Path
from TROTASR.workflows.verify_lowpt_reference_figures import (
    compare_summary, digest, verify_summary_measurement, verify_canonical_calibration,
    reference_summary_contract,
)
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT


def one_point(y=1.0):
    return dict(points=[dict(bin=0, series='sf', present=True, x=6., xlo=5.,
                            xhi=7., y=y, lo=y-.02, hi=y+.02)],
                xaxis=dict(slope=90.), yaxis=dict(slope=500.))


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

    def test_real_calibration_binding_and_changed_payload(self):
        source = read_json(ROOT/'validation/reference_assets_20260923/provenance/lowpt_leptons/source_binding.json')
        for cal in source['calibrations']:
            payload = read_json(ROOT/cal['payload'])
            self.assertEqual(digest(payload), cal['decoded_payload_sha256'])
            check = verify_canonical_calibration(cal, source['figures'], payload)
            self.assertTrue(check['canonical_sf_points_bound_to_payload'])
            changed = copy.deepcopy(payload)
            changed['corrections'][0]['data']['content'][0]['value']['content'][0] += .01
            with self.assertRaises(ValueError):
                verify_canonical_calibration(cal, source['figures'], changed)

    def test_real_fit_result_binding_and_changed_measurement(self):
        source = read_json(ROOT/'validation/reference_assets_20260923/provenance/lowpt_leptons/source_binding.json')
        for cal in source['calibrations']:
            if 'measurement' not in cal: continue
            figure = next(f for f in source['figures'] if f['year'] == cal['year']
                          and f['kind'] == cal['kind'] and f['view'] == 'efficiency')
            self.assertGreater(verify_summary_measurement(figure['canonical_summary'], cal['measurement'], 'efficiency'), 0)
            changed = copy.deepcopy(cal['measurement'])
            changed['bins'][0]['nominal']['data']['efficiency'] += .01
            with self.assertRaises(ValueError):
                verify_summary_measurement(figure['canonical_summary'], changed, 'efficiency')

    def test_inventory_and_visual_audit_bound_by_exact_path(self):
        inventory = read_json(ROOT/'an_figure_inventory.json')
        qa = read_json(ROOT/'validation/reference_assets_20260923/visual_qa.json')
        by_path = {f['an_path']:f for f in inventory['figures']}
        self.assertEqual(len(qa['records']), 67)
        for record in qa['records']:
            figure = by_path[record['an_path']]
            for key in ('status', 'provenance_qa', 'numeric_binding_audit'):
                self.assertEqual(record.get(key), figure.get(key), record['an_path'])

    def test_measurement_only_scope_requires_invalid_fit_evidence(self):
        source = read_json(ROOT/'validation/reference_assets_20260923/provenance/lowpt_leptons/source_binding.json')
        fig = next(f for f in source['figures'] if f['an_path'] == 'figures/lowpt_lepton_id_2025/electron_scale_factor.pdf')
        cal = next(c for c in source['calibrations'] if c['year'] == 2025 and c['kind'] == 'electron')
        expected, audit = reference_summary_contract(fig, cal)
        self.assertEqual(audit['omitted_invalid_fit_bins'], [4, 5])
        self.assertEqual(sum(p['present'] for p in expected['points']), 4)
        self.assertEqual(sum(p['present'] for p in fig['canonical_summary']['points']), 6)
        changed = copy.deepcopy(cal); changed['measurement']['bins'][4]['valid'] = True
        with self.assertRaisesRegex(ValueError, 'actual invalid'):
            reference_summary_contract(fig, changed)
        changed = copy.deepcopy(fig); changed['reference_scope']['omitted_invalid_bins'] = [0, 4, 5]
        with self.assertRaisesRegex(ValueError, 'actual invalid'):
            reference_summary_contract(changed, cal)

    def test_measurement_scope_does_not_hide_missing_valid_points(self):
        source = read_json(ROOT/'validation/reference_assets_20260923/provenance/lowpt_leptons/source_binding.json')
        fig = next(f for f in source['figures'] if f.get('reference_scope'))
        cal = next(c for c in source['calibrations'] if c['year'] == fig['year'] and c['kind'] == fig['kind'])
        expected, _ = reference_summary_contract(fig, cal)
        actual = copy.deepcopy(expected); actual['points'][0]['present'] = False
        self.assertEqual(compare_summary(actual, expected)['status'], 'reference_numeric_review_required')
        changed = copy.deepcopy(fig); changed['reference_scope']['definition_audit_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Changed reference-scope'):
            reference_summary_contract(changed, cal)


if __name__ == '__main__': unittest.main()
