"""Explicit JME input acceptance, with real card/provenance checks and no ROOT I/O."""
import copy
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows import continue_limit_fits as w


class JMEInputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'results', prefix='.jme_contract_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.manifest = self.base/'fit_grid/manifest.json'
        self.output = self.base/'limits'
        self.output.mkdir()
        self.template_path = self.base/'fit_grid/templates/model_manifest.json'
        exclusions = [dict(channel=f'SR_highdm_c1_{year}', process='QCD_Nb1_u6',
                           root_objects_preserved=True, preliminary=True,
                           decision='user_approved_fit_only_exclusion') for year in (2024, 2025)]
        support = dict(status='ready', auto_mc_stats=[10, 1, 1], interpolation_changed=False,
                       endpoints_modified=False, all_template_objects_preserved=True,
                       background_issues=[], blocked_signal_points=[], preliminary_fit_exclusions=exclusions)
        channels = {f'{region}_{mode}_c1_{year}': list(range(count))
                    for year in (2024, 2025) for region in ('SR', 'LLCR', 'QCDCR', 'GCR')
                    for mode, count in ((('highdm', 118), ('lowdm', 30)) if region == 'SR'
                                        else (('highdm', 16), ('lowdm', 10)))}
        merge_path = self.base/'sr_merge.json'
        write_json(merge_path, {'fixture': 'projection'})
        merge = dict(path=str(merge_path.relative_to(ROOT)), sha256=sha256(merge_path))
        self.template = dict(status='templates_ready', scope=w.JME_SCOPE, systematic_scope=w.JME_SYSTEMATICS,
                             root_integrity_checked=True, sr_data_blinded=True, auto_mc_stats=[10, 1, 1],
                             bins=452, channels=channels, sr_merge=merge, fit_compatibility=support,
                             files={f'template_{mode}_{year}.root': 'synthetic_digest_only'
                                    for mode in ('highdm', 'lowdm') for year in (2024, 2025)})
        write_json(self.template_path, self.template)
        origin = [dict(path=str(self.template_path.parent.relative_to(ROOT)), sha256=sha256(self.template_path))]
        points = []
        for model, count in (('T2tt', 205), ('T2bW', 354), ('T2tb', 365)):
            for i in range(count):
                mass = f'mStop{1000+i}_mLSP1'
                card = self.base/f'fit_grid/datacards/{model}_{mass}.txt'
                card.parent.mkdir(parents=True, exist_ok=True)
                card.write_text(''.join(f'shapes * {n} ../templates/{n} $CHANNEL/$PROCESS\n'
                                        for n in self.template['files'])+'* autoMCStats 10 1 1\n')
                meta = dict(card_sha256=sha256(card), cr_only=False, sr_data_blinded=True,
                            years=['2024', '2025'], auto_mc_stats=[10, 1, 1], topology=model,
                            mStop=1000+i, mLSP=1, inputs=origin, fit_ready=True, fit_issues=[],
                            preliminary_fit_exclusions=exclusions)
                write_json(card.with_suffix('.manifest.json'), meta)
                points.append(dict(model=model, mass=mass, card=str(card.relative_to(ROOT)),
                                   card_sha256=sha256(card), fit_ready=True))
        self.grid = dict(status='cards_ready', scope=w.JME_SCOPE, systematic_scope=w.JME_SYSTEMATICS,
                         points=points, expected_points_by_model={'T2tt': 205, 'T2bW': 354, 'T2tb': 365},
                         blocked=[], ready_inputs=924, template_points=924, root_integrity_checked=True,
                         sr_data_blinded=True, bins=452, th1_channels=16, sr_merge=merge,
                         fit_compatibility=support, templates_manifest=str(self.template_path.relative_to(ROOT)))
        write_json(self.manifest, self.grid)
        write_json(self.output/'contract.json', dict(manifest_sha256=sha256(self.manifest), code={}))
        self.hashes = dict(manifest=sha256(self.manifest), primary_contract=sha256(self.output/'contract.json'),
                           templates=sha256(self.template_path))
        self.hash_patch = mock.patch.object(w, 'JME_INPUT_HASHES', self.hashes)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def digest(self, path):
        # No local ROOT file is created or opened, even in a synthetic fixture.
        return 'synthetic_digest_only' if Path(path).suffix == '.root' else sha256(path)

    def verify(self, grid=None, workers=10, digest=None):
        return w.verify_grid_inputs(self.manifest, self.output, self.grid if grid is None else grid,
                                    workers, self.digest if digest is None else digest)

    def mutate_meta(self, **changes):
        path = (ROOT/self.grid['points'][0]['card']).with_suffix('.manifest.json')
        meta = read_json(path); meta.update(changes); write_json(path, meta)

    def test_approved_jme_uses_all_existing_card_checks(self):
        with mock.patch.object(w, 'verify_card', wraps=w.verify_card) as verify:
            result = self.verify()
        self.assertEqual(verify.call_count, 924)
        self.assertEqual(len(result['card_manifests_sha256']), 924)
        self.assertFalse(result['nonjme16_included'])
        self.assertEqual(result['input_hashes'], self.hashes)
        self.assertFalse(list(self.base.rglob('*.root')))
        self.assertFalse(list(self.output.glob('numerical*')))

    def test_historical_nominal_branch_unchanged_and_other_scope_rejected(self):
        self.assertIsNone(self.verify(dict(status='cards_ready', scope='full_nominal_production'), workers=47))
        for scope in ('validation_only', 'full_stored_trota_nonjme_production', None):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, 'scope'):
                self.verify(dict(self.grid, scope=scope))

    def test_hash_pins_are_required(self):
        for name in self.hashes:
            with self.subTest(name=name), mock.patch.dict(self.hashes, {name: 'changed'}), \
                    self.assertRaises(ValueError):
                self.verify()

    def test_no_foreign_output_or_excess_pool(self):
        with self.assertRaisesRegex(ValueError, 'ten workers'): self.verify(workers=11)
        with self.assertRaisesRegex(ValueError, 'primary limits directory'):
            w.verify_grid_inputs(self.manifest, self.base/'elsewhere', self.grid, 10, self.digest)

    def test_incomplete_duplicate_or_unready_grid_rejected(self):
        cases = [dict(self.grid, status='running'), dict(self.grid, points=self.grid['points'][:-1]),
                 dict(self.grid, points=[self.grid['points'][0], *self.grid['points'][0:-1]]),
                 dict(self.grid, root_integrity_checked=False), dict(self.grid, sr_data_blinded=False),
                 dict(self.grid, systematic_scope='stored-TROTA/non-JME16'), dict(self.grid, blocked=['bad'])]
        for grid in cases:
            with self.subTest(changes=[k for k in grid if grid[k] != self.grid[k]]), self.assertRaises(ValueError):
                self.verify(grid)

    def test_card_metadata_blinding_statistics_inputs_exclusions_rejected(self):
        path = (ROOT/self.grid['points'][0]['card']).with_suffix('.manifest.json')
        original = read_json(path)
        for changes in (dict(sr_data_blinded=False), dict(auto_mc_stats=[10, 1, 2]),
                        dict(inputs=[]), dict(preliminary_fit_exclusions=[]), dict(fit_ready=False),
                        dict(years=['2023', '2025']), dict(fit_issues=['bad'])):
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.verify()

    def test_changed_actual_card_or_root_digest_rejected(self):
        first = ROOT/self.grid['points'][0]['card']
        first.write_text(first.read_text()+'# unexpected change\n')
        with self.assertRaisesRegex(ValueError, 'Card changed'): self.verify()
        first.write_text(first.read_text().replace('# unexpected change\n', ''))
        with self.assertRaisesRegex(ValueError, 'Template file changed'):
            self.verify(digest=lambda p: 'changed' if Path(p).suffix == '.root' else sha256(p))

    def test_template_semantics_fail_even_with_test_hash_pin_updated(self):
        original = copy.deepcopy(self.template)
        for changes in (dict(scope='full_nominal_production'), dict(systematic_scope='stored'),
                        dict(files={}), dict(auto_mc_stats=[10, 1, 2]), dict(bins=338),
                        dict(root_integrity_checked=False), dict(sr_data_blinded=False)):
            write_json(self.template_path, dict(original, **changes))
            self.hashes['templates'] = sha256(self.template_path)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'template scope'):
                self.verify()

    def test_changed_projection_rejected(self):
        write_json(ROOT/self.grid['sr_merge']['path'], {'changed': True})
        with self.assertRaisesRegex(ValueError, 'SR projection'): self.verify()

    def test_primary_lock_prevents_any_plan_or_fit(self):
        write_json(self.output/'state.json', dict(status='needs_attention'))
        with (self.output/'controller.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(w, 'verify_grid_inputs', return_value={}), \
                    mock.patch.object(w, 'runtime') as runtime, mock.patch.object(w, 'make_plan') as make_plan:
                with self.assertRaisesRegex(TimeoutError, 'Primary controller'):
                    w.run(self.manifest, self.output, 10, 0)
            runtime.assert_not_called(); make_plan.assert_not_called()
        self.assertFalse((self.output/'numerical_plan.json').exists())

    def test_orphan_plan_cannot_receive_a_new_jme_contract(self):
        write_json(self.output/'state.json', dict(status='needs_attention'))
        write_json(self.output/'numerical_plan.json', dict(points=[], unsupported=[]))
        with mock.patch.object(w, 'verify_grid_inputs', return_value={}), \
                mock.patch.object(w, 'runtime') as runtime:
            with self.assertRaisesRegex(ValueError, 'Unbound JME numerical plan'):
                w.run(self.manifest, self.output, 10, 0)
        runtime.assert_not_called()
        self.assertFalse((self.output/'numerical_contract.json').exists())

    def test_inputs_rechecked_under_primary_lock_before_plan(self):
        write_json(self.output/'state.json', dict(status='needs_attention'))
        with mock.patch.object(w, 'verify_grid_inputs', side_effect=[{}, {'changed': True}]) as verify, \
                mock.patch.object(w, 'make_plan') as plan:
            with self.assertRaisesRegex(ValueError, 'changed while waiting'):
                w.run(self.manifest, self.output, 10, 0)
        self.assertEqual(verify.call_count, 2)
        plan.assert_not_called()
        self.assertFalse((self.output/'numerical_plan.json').exists())

    def test_complete_point_never_refitted_and_jme_contract_survives_resume(self):
        # Exercise controller persistence separately from the full-grid guard.
        # The above tests use all 924 real card checks; here one complete fixture
        # is sufficient to prove the controller never sends success to a worker.
        grid = dict(self.grid, points=self.grid['points'][:1])
        write_json(self.manifest, grid)
        write_json(self.output/'contract.json', dict(manifest_sha256=sha256(self.manifest), code={}))
        write_json(self.output/'state.json', dict(status='needs_attention'))
        point = grid['points'][0]
        record = dict(point, status='complete', validation={'quantiles':
                      {str(q): float(i+1) for i, q in enumerate((.025, .16, .5, .84, .975))}})
        state_path = self.output/point['model']/point['mass']/'state.json'
        write_json(state_path, record)
        initial = sha256(state_path)
        proof = {'input_hashes': self.hashes, 'scope': w.JME_SCOPE}
        with mock.patch.object(w, 'verify_grid_inputs', return_value=proof), \
                mock.patch.object(w, 'SOURCES', ()), mock.patch.object(w, 'runtime', return_value={}), \
                mock.patch.object(w, 'CommandRunner') as runner, mock.patch.object(w, 'run_point') as run_point, \
                mock.patch.object(w, 'validate_complete'), mock.patch.object(w, 'verify_card'):
            self.assertEqual(w.run(self.manifest, self.output, 10, 0), 0)
            self.assertEqual(w.run(self.manifest, self.output, 10, 0), 0)
        run_point.assert_not_called(); runner.return_value.run.assert_not_called()
        self.assertEqual(sha256(state_path), initial)
        self.assertEqual(read_json(self.output/'numerical_contract.json')['jme_inputs'], proof)

    def test_changed_primary_code_is_rejected_before_command(self):
        base = read_json(self.output/'contract.json')
        base['code'] = {'workflows/limit_validation.py': 'changed'}
        write_json(self.output/'contract.json', base)
        with mock.patch.object(w, 'verify_grid_inputs', return_value={}), mock.patch.object(w, 'runtime') as runtime:
            with self.assertRaisesRegex(ValueError, 'Active fit code changed'):
                w.run(self.manifest, self.output, 10, 0)
        runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
