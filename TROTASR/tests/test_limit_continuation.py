import unittest
from unittest import mock
from pathlib import Path
import tempfile
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows import continue_limit_fits as workflow
from TROTASR.workflows.continue_limit_fits import choose_range, configs, command_for, target_nll


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

    def test_exactly_two_additional_configurations(self):
        values=configs(500)
        self.assertEqual(len(values),2)
        self.assertEqual([x['strategy'] for x in values],[0,1])
        self.assertTrue(all(x['tolerance']==.1 and x['rmax']==500 for x in values))

    def test_blind_command_and_original_accuracy_unchanged(self):
        cmd=command_for('/hep2-scratch/workspace.root','mStop1200_mLSP500',configs(500)[1])
        self.assertEqual(cmd[cmd.index('--run')+1],'blind')
        self.assertEqual(cmd[cmd.index('--cminDefaultMinimizerStrategy')+1],'1')
        self.assertEqual(cmd[cmd.index('--rMax')+1],'500')
        self.assertIn('--picky',cmd)
        self.assertEqual(cmd[cmd.index('--rRelAcc')+1],'0.0025')
        self.assertNotIn('--freezeParameters',cmd)

    def test_target_matches_median_convention(self):
        self.assertAlmostEqual(target_nll(.5),1.920729410347062,places=12)


class PersistenceTests(unittest.TestCase):
    def fixture(self, base):
        point=dict(model='T2tt',mass='mStop1200_mLSP500',card_sha256='card')
        key=point['model']+'/'+point['mass'];work=base/key;work.mkdir(parents=True)
        record=dict(point,status='failed',pid=None,workspace_valid=True,workspace_sha256='workspace',
                    attempts=[dict(name='bisection_s0'),dict(name='bisection_s0_refined')])
        write_json(work/'state.json',record)
        plan=dict(point=point,key=key,source_state_sha256=sha256(work/'state.json'),
                  workspace_sha256='workspace',logs_sha256={},rmax=40,attempts=configs(40))
        return work,record,plan

    def digest(self, path):
        path=Path(path)
        if path.name=='workspace.root':return 'workspace'
        if path.suffix=='.root':return 'synthetic_product'
        return sha256(path)

    def test_two_failures_persist_and_resume_does_not_rerun(self):
        with tempfile.TemporaryDirectory(prefix='.limit_test_',dir=ROOT/'tests') as name:
            base=Path(name);work,original,plan=self.fixture(base)
            runner=mock.Mock();runner.run.return_value=dict(status='complete',exit_code=0)
            with mock.patch.object(workflow,'verify_card'), mock.patch.object(workflow,'sha256',side_effect=self.digest), \
                 mock.patch.object(workflow,'validate',return_value=(False,{'reason':'synthetic failed crossing'})):
                first=workflow.run_point(plan,base,runner,{},None)
                second=workflow.run_point(plan,base,runner,{},None)
            self.assertEqual(runner.run.call_count,2)
            self.assertEqual(first['status'],'failed')
            self.assertEqual(len(second['attempts']),4)
            self.assertEqual(read_json(base/'history/numerical_20260924'/plan['key']/'state.json'),original)

    def test_success_reused_and_original_failure_preserved(self):
        with tempfile.TemporaryDirectory(prefix='.limit_test_',dir=ROOT/'tests') as name:
            base=Path(name);work,original,plan=self.fixture(base)
            runner=mock.Mock();runner.run.return_value=dict(status='complete',exit_code=0)
            with mock.patch.object(workflow,'verify_card'), mock.patch.object(workflow,'sha256',side_effect=self.digest), \
                 mock.patch.object(workflow,'validate',return_value=(True,{'quantiles':{'0.5':1.0}})):
                first=workflow.run_point(plan,base,runner,{},None)
                second=workflow.run_point(plan,base,runner,{},None)
            self.assertEqual(runner.run.call_count,1)
            self.assertEqual(first['status'],'complete')
            self.assertEqual(second,first)
            self.assertEqual(read_json(base/'history/numerical_20260924'/plan['key']/'state.json'),original)

    def test_divergent_failed_point_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='.limit_test_',dir=ROOT/'tests') as name:
            base=Path(name);work,record,plan=self.fixture(base)
            record['extra']='changed';write_json(work/'state.json',record)
            with mock.patch.object(workflow,'verify_card'):
                with self.assertRaisesRegex(ValueError,'Changed failed point'):
                    workflow.run_point(plan,base,mock.Mock(),{},None)
            self.assertEqual(read_json(work/'state.json'),record)

    def test_saved_first_attempt_revalidated_without_refit(self):
        with tempfile.TemporaryDirectory(prefix='.limit_test_',dir=ROOT/'tests') as name:
            base=Path(name);work,record,plan=self.fixture(base)
            for attempt in record['attempts']:
                attempt.update(exit_code=0,tolerance=.1,
                    log=str((work/attempt['name']/'combine.log').relative_to(ROOT)))
            write_json(work/'state.json',record)
            plan['source_state_sha256']=sha256(work/'state.json')
            runner=mock.Mock()
            with mock.patch.object(workflow,'verify_card'), mock.patch.object(workflow,'sha256',side_effect=self.digest), \
                 mock.patch.object(workflow,'validate',return_value=(True,{'quantiles':{'0.5':1.0},'validation_adapter':'binary32_return_and_six_decimal_log'})):
                first=workflow.run_point(plan,base,runner,{},None)
                second=workflow.run_point(plan,base,runner,{},None)
            runner.run.assert_not_called()
            self.assertEqual(second,first)
            self.assertEqual(first['current_attempt'],'bisection_s0')
            self.assertTrue(first['saved_output_revalidated_without_refit'])
            self.assertEqual(len(first['attempts']),2)
            self.assertEqual(read_json(base/'history/numerical_20260924'/plan['key']/'state.json'),record)

    def test_no_extra_third_configuration(self):
        with tempfile.TemporaryDirectory(prefix='.limit_test_',dir=ROOT/'tests') as name:
            base=Path(name);work,record,plan=self.fixture(base)
            plan['attempts'].append(dict(name='third',strategy=1,tolerance=.001,rmax=400))
            with self.assertRaisesRegex(ValueError,'finite numerical configurations'):
                workflow.run_point(plan,base,mock.Mock(),{},None)


class Run3ExtendedCapacityTests(unittest.TestCase):
    """Admission includes all reserved futures, not only visible subprocesses."""

    def capacity(self, **updates):
        from TROTASR.workflows.validate_limit_precision import run3_extended_capacity
        arguments = dict(workers=115, external=0, memory=1024*1024**3,
                         active=0, absent_reserved=0, disk_free=100*1024**3)
        return run3_extended_capacity(**dict(arguments, **updates))

    def test_controller_services_and_unspawned_futures_are_reserved(self):
        self.assertEqual(self.capacity(), 115)
        self.assertEqual(self.capacity(external=3, active=90, absent_reserved=5), 107)
        self.assertEqual(self.capacity(external=115), 0)
        self.assertEqual(self.capacity(external=90, absent_reserved=40), 0)
        self.assertEqual(self.capacity(workers=20, external=5), 20)

    def test_future_memory_growth_and_disk_floor(self):
        gib = 1024**3
        self.assertEqual(self.capacity(memory=52*gib, active=10, absent_reserved=3), 12)
        self.assertEqual(self.capacity(memory=32*gib, active=10, absent_reserved=2), 10)
        self.assertEqual(self.capacity(memory=23*gib, active=10), 0)
        self.assertEqual(self.capacity(disk_free=79*gib, active=10), 0)
        with self.assertRaises(ValueError):
            self.capacity(workers=116)

    def test_signal_intent_is_durable_before_once_only_main_signal(self):
        from TROTASR.workflows import validate_limit_precision as precision
        import signal
        identity = dict(pid=123456789, startticks=4321, argv=['dummy_controller'])
        evidence = dict(trigger='T2tt/mStop1000_mLSP1')
        with tempfile.TemporaryDirectory(prefix='.extended_signal_test_', dir=ROOT/'tests') as name:
            path = Path(name)/'signal.json'
            def observed_signal(pid, number):
                self.assertEqual((pid, number), (identity['pid'], signal.SIGINT))
                self.assertEqual(read_json(path)['status'], 'intent')
                self.assertFalse(read_json(path)['native_children_signaled'])
            with mock.patch.object(precision, '_run3_identity', return_value=identity), \
                 mock.patch.object(precision.os, 'kill', side_effect=observed_signal) as kill:
                first = precision.run3_signal_once(path, identity, evidence)
                second = precision.run3_signal_once(path, identity, evidence)
            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'delivered')
            kill.assert_called_once()

    def test_uncertain_or_changed_signal_is_never_repeated(self):
        from TROTASR.workflows import validate_limit_precision as precision
        identity = dict(pid=123456789, startticks=4321, argv=['dummy_controller'])
        evidence = dict(trigger='T2tt/mStop1000_mLSP1')
        with tempfile.TemporaryDirectory(prefix='.extended_intent_test_', dir=ROOT/'tests') as name:
            path = Path(name)/'signal.json'
            write_json(path, dict(status='intent', identity=identity, evidence=evidence))
            with mock.patch.object(precision.os, 'kill') as kill:
                with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                    precision.run3_signal_once(path, identity, evidence)
                with self.assertRaisesRegex(ValueError, 'Divergent'):
                    precision.run3_signal_once(path, dict(identity, startticks=4322), evidence)
            kill.assert_not_called()

    def test_finite_recovery_precedes_unstarted_primary_without_fixed_reservation(self):
        from TROTASR.workflows import validate_limit_precision as precision
        points = [dict(model='T2tt', mass='mStop%d_mLSP1'%mass) for mass in (1000, 1050, 1100)]
        with tempfile.TemporaryDirectory(prefix='.extended_candidates_test_', dir=ROOT/'tests') as name:
            limits = Path(name)
            for point in points[1:]:
                key = point['model']+'/'+point['mass']
                write_json(limits/key/'state.json', dict(status='failed', pid=None, workspace_valid=True))
            key = points[2]['model']+'/'+points[2]['mass']
            write_json(limits/'numerical_all_unfinished_20260930'/key/'state.json',
                       dict(status='failed', pid=None, error='two_unfinished_recovery_configurations_exhausted'))
            candidates, counts = precision.run3_extended_candidates(points, limits, set(), set(), True)
            self.assertEqual([kind for point, kind in candidates], ['advanced', 'base', 'primary'])
            candidates, _ = precision.run3_extended_candidates(points, limits, set(), set(), False)
            self.assertEqual([kind for point, kind in candidates], ['advanced'])
            candidates, _ = precision.run3_extended_candidates(points, limits, {key}, set(), False)
            self.assertEqual(candidates, [])


if __name__=='__main__':unittest.main()
