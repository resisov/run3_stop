"""Contour input/accounting tests; no PDF authoring and no fabricated limits."""
import copy
import unittest
import tempfile
from pathlib import Path
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT
from TROTASR.workflows import plotting_limit_contour as contour
from TROTASR.workflows.plotting_limit_contour import validate_collected, sr_bin_counts


def fixture():
    return dict(status='complete', scope='full_combined_systematic_production', observed_sr_used=False,
        requested_point_count=1, collected_point_count=1, missing_points=[],
        points={'mStop1200_mLSP500': dict(mStop=1200, mLSP=500, expected_m2=.5,
            expected_m1=.7, expected=1., expected_p1=1.3, expected_p2=1.6)})


class NominalContourTests(unittest.TestCase):
    def test_canonical_bin_identity_comes_from_accepted_grid(self):
        self.assertEqual(sr_bin_counts(dict(canonical_limits_pipeline=True,
            sr_bins_per_year=dict(highdm=118,lowdm=30))), (118,30))
        self.assertEqual(sr_bin_counts(dict(canonical_limits_pipeline=True,
            statistical_revision=dict(preserve_sr_bins=[118,30]))), (118,30))
        with self.assertRaises(ValueError):
            sr_bin_counts(dict(canonical_limits_pipeline=True))

    def test_mass_support_mask_never_fills_a_missing_vertex(self):
        import numpy as np
        from TROTASR.workflows.plotting_limit_contour import support_mask
        masses=['mStop100_mLSP1','mStop100_mLSP99','mStop200_mLSP1',
                'mStop200_mLSP99','mStop150_mLSP50']
        for missing in (masses[0],masses[-1]):
            mask,stats=support_mask(masses,set(masses)-{missing})
            from TROTASR.workflows.signal_grid_inputs import mass_pair
            x,y=mass_pair(missing)
            self.assertFalse(mask(np.array([x]),np.array([y]))[0])
            self.assertEqual(stats['verified'],4)
        mask,_=support_mask(masses,set(masses))
        self.assertFalse(mask(np.array([110.]),np.array([120.]))[0])

    def test_complete_accounting(self):
        validate_collected(fixture(), ['mStop1200_mLSP500'])

    def test_a_dropped_failed_point_is_not_complete(self):
        with self.assertRaisesRegex(ValueError, 'accounting'):
            validate_collected(fixture(), ['mStop1200_mLSP500', 'mStop1300_mLSP500'])

    def test_partial_is_explicit_and_retains_missing_points(self):
        payload = fixture()
        payload.update(status='partial', requested_point_count=2, missing_points=['mStop1300_mLSP500'])
        before = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, 'Incomplete limits'):
            validate_collected(payload, ['mStop1200_mLSP500', 'mStop1300_mLSP500'])
        validate_collected(payload, ['mStop1200_mLSP500', 'mStop1300_mLSP500'], allow_partial=True)
        self.assertEqual(payload, before)

    def test_observed_or_validation_input_rejected(self):
        for key, value in (('observed_sr_used', True), ('scope', 'representative'),
                           ('scope', 'full_run3_hybrid_production')):
            payload = fixture(); payload[key] = value
            with self.assertRaises(ValueError):
                validate_collected(payload, list(payload['points']))

    def test_nonordered_quantiles_rejected(self):
        payload = fixture(); payload['points']['mStop1200_mLSP500']['expected_p2'] = .2
        with self.assertRaisesRegex(ValueError, 'quantiles'):
            validate_collected(payload, list(payload['points']))


    def test_contour_does_not_accept_median_only(self):
        payload = fixture()
        row = payload['points']['mStop1200_mLSP500']
        for key in ('expected_m2','expected_m1','expected_p1','expected_p2'):
            row.pop(key)
        payload['quantiles_provided']=[.5]
        with self.assertRaises(ValueError):
            validate_collected(payload, list(payload['points']))

    def test_read_saved_points_requires_current_five_quantiles_and_artifacts(self):
        point = dict(model='T2tt',mass='mStop1200_mLSP500',card_sha256='approved')
        failed = dict(point,mass='mStop1300_mLSP500')
        key = point['model']+'/'+point['mass']
        missing_key = failed['model']+'/'+failed['mass']
        with tempfile.TemporaryDirectory(dir=ROOT/'tests',prefix='test-contour-') as folder:
            base=Path(folder)
            paths={name: str(base/(name+'.fixture')) for name in ('output','log','workspace')}
            for path in paths.values():
                Path(path).touch()
            quantiles = {str(q): index+1. for index,q in enumerate((.025,.16,.5,.84,.975))}
            attempt = dict(output=paths['output'], log=paths['log'], exit_code=0, passed=True,
                           command=['combine','-M','AsymptoticLimits',paths['workspace']],
                           validation=dict(quantiles=quantiles))
            record = dict(point, **paths, status='complete', exit_code=0,
                          validation=dict(quantiles=quantiles), accepted_attempt=attempt)
            grid = dict(points=[point,failed], sr_data_blinded=True,
                        canonical_limits_pipeline=True, era_merged=True,
                        scope='full_combined_systematic_production')
            state = dict(canonical_limits_pipeline=True, analysis_root=str(ROOT),
                         canonical_model='era_merged', observed_sr_used=False,
                         results={key:record, missing_key:dict(failed,status='running')})
            receipt=base/'workflow_state.json'
            contour.write_json(receipt,state)
            with self.assertRaises(ValueError):
                contour.verified_current_results(grid,base)
            results,evidence=contour.verified_current_results(grid,base,allow_partial=True)
            self.assertEqual(set(results),{key,missing_key})
            self.assertEqual(results[key]['validation']['quantiles'],quantiles)
            self.assertNotIn('validation',results[missing_key])
            self.assertIsNone(evidence[missing_key]['accepted'])
            quantiles.pop('0.025')
            contour.write_json(receipt,state)
            with self.assertRaises(ValueError):
                contour.verified_current_results(grid,base,allow_partial=True)

    def test_theory_and_reference_inputs_are_installed(self):
        config = read_json(ROOT/'stats/contour_config.json')
        for spec in list(config['references'].values())+[config['theory']]:
            self.assertTrue((ROOT/spec['path']).is_file())
            self.assertTrue(spec['physics_values_unchanged'])


class RendererInputTests(unittest.TestCase):
    def test_renderer_uses_explicit_configured_theory_and_thermal_inputs(self):
        from TROTASR.utils.renderers import postprocess_limits as renderer
        config = read_json(ROOT/'stats/contour_config.json')
        xsecs = renderer.load_stop_pair_xsecs(ROOT/config['theory']['path'])
        self.assertGreater(xsecs[1200], 0.)
        source = config['references']['thermal_relic']
        thermal = renderer.load_thermal_relic_contour(ROOT/source['path'])
        self.assertEqual(thermal['omega_h2'], .12)
        self.assertTrue(renderer.thermal_overabundance_strips(thermal['points']))

    def test_current_signed_coordinates_and_half_weighted_mixed_branch(self):
        import numpy as np
        from TROTASR.utils.renderers import postprocess_limits as renderer
        daughter = np.array([500., 500., 500.])
        stop = daughter + np.array([170., renderer.TOP_MASS_GEV, 180.])
        top = renderer.topology_phase_space_coordinate('T2tt', stop, daughter)
        w = renderer.topology_phase_space_coordinate('T2bW', stop, daughter)
        mixed = renderer.topology_phase_space_coordinate('T2tb', stop, daughter)
        self.assertLess(top[0], 0.)
        self.assertEqual(top[1], 0.)
        self.assertGreater(top[2], 0.)
        np.testing.assert_array_equal(mixed, .5 * (top + w))

    def test_no_autonomous_runtime_or_unverified_root_collector(self):
        from TROTASR.utils.renderers import postprocess_limits as renderer
        self.assertTrue(callable(renderer.plot_contour))
        for name in ('main', 'validate_runtime', 'load_excluded_points',
                     'collect_limits', 'parse_limit_file'):
            self.assertFalse(hasattr(renderer, name), name)
