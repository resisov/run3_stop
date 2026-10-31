import unittest
import contextlib
import fcntl
import io
import tempfile
import threading
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from TROTASR.utils.paths import ROOT
from TROTASR.workflows import validate_limit_precision as workflow
from TROTASR.workflows import diagnostic_common as common, continue_limit_fits as continuation
from TROTASR.workflows import reconcile_limit_fits as recovery, combine_limits as combine
from TROTASR.workflows.limit_validation import QUANTILES
from TROTASR.workflows.validate_limit_precision import command,control_comparison,CONTROL,FAILURES


class PrecisionStudyTests(unittest.TestCase):
    def test_measured_start_preserves_model_and_uses_actual_profile(self):
        previous=dict(status='failed',pid=None,attempts=[dict(passed=False,
            command=workflow.fit_command('workspace.root','mass',143.643,.001))])
        primary=dict(status='failed',pid=None)
        text='Will search for NLL crossing by bisection\nAt r = 39.8: delta(nll) = 1.89\n'
        before=deepcopy(previous)
        specs=workflow.measured_start_commands('workspace.root','mass',primary,previous,[('actual.log',text)])
        self.assertEqual(previous,before);self.assertEqual(len(specs),2)
        for spec in specs:
            self.assertEqual(spec['rmax'],50)
            self.assertEqual(spec['range_evidence']['sampled_r'],39.8)
            self.assertTrue(spec['seed_is_not_a_limit'])
            cmd=spec['command']
            self.assertIn('--picky',cmd)
            self.assertEqual(cmd[cmd.index('--run')+1],'blind')
            self.assertNotIn('--freezeParameters',cmd)
            self.assertNotIn('MINIMIZER_no_analytic',cmd)
            self.assertEqual(cmd[cmd.index('--rRelAcc')+1],'0.0025')

    def test_measured_start_rejects_live_successful_or_unmeasured_inputs(self):
        previous=dict(status='failed',pid=None,attempts=[dict(passed=False,
            command=workflow.fit_command('workspace.root','mass',143.643,.001))])
        primary=dict(status='failed',pid=None)
        for bad in (dict(previous,pid=20),dict(previous,status='complete')):
            with self.assertRaises(ValueError):
                workflow.measured_start_commands('workspace.root','mass',primary,bad,[])
        with self.assertRaisesRegex(ValueError,'No measured'):
            workflow.measured_start_commands('workspace.root','mass',primary,previous,[])
        previous['attempts'][0]['command']+=['--freezeParameters','x']
        with self.assertRaises(ValueError):
            workflow.measured_start_commands('workspace.root','mass',primary,previous,[])

    def stepping_fixture(self):
        base=workflow.minimum_commands(workflow.fit_command('workspace.root','mass',542.559,.001))[1:]
        small=dict(status='failed',pid=None,attempts=[dict(a,passed=False,exit_code=0,
            validation=dict(crossings=[dict(quantile=.5,passed=False)])) for a in base])
        large=dict(status='failed',pid=None,attempts=[])
        for t in (.001,.0001):
            cmd=workflow.refinement_command(base[0]['command'],t);cmd[cmd.index('--rMax')+1]='7260'
            large['attempts'].append(dict(name='winner_measured_range_s1_tol'+str(t),command=cmd,
                exit_code=0,passed=False,validation={'reason':'not five expected quantiles'}))
        return large,small

    def test_stepping_is_bounded_same_strategy0_model_and_two_attempts(self):
        large,small=self.stepping_fixture();before=deepcopy((large,small))
        specs=workflow.winner_stepping_commands(large,small)
        self.assertEqual((large,small),before);self.assertEqual(len(specs),2)
        for spec,old in zip(specs,small['attempts']):
            cmd=spec['command'];original=list(old['command'])
            for flag in ('--rMax','--minosAlgo','--rRelAcc','--rAbsAcc'):
                original[original.index(flag)+1]=cmd[cmd.index(flag)+1]
            self.assertEqual(cmd,original+['--strictBounds'])
            self.assertEqual(cmd[cmd.index('--minosAlgo')+1],'stepping')
            self.assertEqual(spec['rmax'],7260);self.assertEqual(spec['first_step'],363)
            self.assertEqual(spec['tolerance'],.01);self.assertEqual(spec['strategy'],0)
        small['attempts'][0]['validation']['crossings'][0]['passed']=True
        with self.assertRaises(ValueError):workflow.winner_stepping_commands(large,small)

    def test_stepping_refuses_changed_model_or_nonterminal_predecessor(self):
        for change in (lambda a:a.update(pid=10),lambda a:a['attempts'][0].update(passed=True),
                       lambda a:a['attempts'][0]['command'].extend(['--freezeParameters','r'])):
            large,small=self.stepping_fixture();change(small)
            with self.assertRaises(ValueError):workflow.winner_stepping_commands(large,small)

    def test_stepping_median_needs_measured_crossing_not_a_returned_bound(self):
        marker='Will search for NLL crossing by stepping\n'
        text=marker+'At r = 99: delta(nll) = 1.90\nAt r = 101: delta(nll) = 1.94\n'
        self.assertFalse(workflow.stepping_median_evidence(text,100.,.01,7260)['passed'])
        text=marker+'At r = 99.9: delta(nll) = 1.9205\nAt r = 100.1: delta(nll) = 1.921\n'
        self.assertTrue(workflow.stepping_median_evidence(text,100.,.01,7260)['passed'])
        text=marker+'At r = 100: delta(nll) = 1.0\n'
        self.assertFalse(workflow.stepping_median_evidence(text,100.,.01,7260)['passed'])
        for invalid in ('',text+marker,text+'At r = 100001: delta(nll) = 2.0\n'):
            with self.assertRaises(ValueError):workflow.stepping_median_evidence(invalid,100.,.01,7260)

    def test_stepping_median_retains_binary32_log_reconciliation(self):
        import struct
        value=struct.unpack('f',struct.pack('f',4030.123456))[0]
        text='Will search for NLL crossing by stepping\nAt r = 4030.123456: delta(nll) = 1.92073\n'
        result=workflow.stepping_median_evidence(text,value,.01,7260)
        self.assertTrue(result['passed']);self.assertEqual(len(result['matching_scan_rows']),1)

    def test_winner_range_uses_existing_measured_policy_not_returned_limit(self):
        base=workflow.minimum_commands(workflow.fit_command('workspace.root','mass',40,.001))[1:]
        previous=dict(status='failed',pid=None,attempts=[dict(s,exit_code=0,passed=False,
            validation=dict(crossings=[dict(quantile=.5,passed=False)])) for s in base])
        text='Will search for NLL crossing by bisection\n'+''.join(
            'At r = %s: delta(nll) = %s\n'%(r,n) for r,n in [(20,.1),(30,.2),(39,.3)])
        original=deepcopy(previous);specs=workflow.winner_range_commands(previous,[('saved.log',text,40)])
        self.assertEqual(previous,original);self.assertEqual(len(specs),2)
        for s in specs:
            c=s['command'];self.assertGreater(s['rmax'],40);self.assertLessEqual(s['rmax'],100000)
            self.assertEqual(float(c[c.index('--rMax')+1]),s['rmax'])
            self.assertIn('MINIMIZER_no_analytic',c);self.assertIn('--picky',c)
            self.assertTrue(s['seed_is_not_a_limit']);self.assertTrue(s['range_evidence'])
            self.assertNotIn('--freezeParameters',c)
        previous['attempts'][0]['validation']['crossings'][0]['passed']=True
        with self.assertRaises(ValueError):workflow.winner_range_commands(previous,[('saved.log',text,40)])
        with self.assertRaises(ValueError):workflow.winner_range_commands(original,[('saved.log','',40)])

    def test_winner_range_scope_is_one_known_point_not_other_successes(self):
        base=ROOT/'stats/topology_transition_nominal/combined_optimization/winner/medium'
        with patch.object(workflow,'winner_recovery_contract',return_value={}):
            self.assertEqual(workflow.refinement_scope(base/'grid/manifest.json',base/'limits',
                'T2tb/mStop1600_mLSP1450',True)[1:],(workflow.COMPARISON_RECOVERY,'numerical_crossing_20261011'))
            with self.assertRaises(ValueError):workflow.refinement_scope(base/'grid/manifest.json',base/'limits',
                'T2tb/mStop1600_mLSP1350',True)
            self.assertEqual(workflow.refinement_scope(base/'grid/manifest.json',base/'limits',
                'T2tb/mStop1600_mLSP1450',True,True)[1:],
                ('numerical_crossing_20261011','numerical_native_bracket_20261011'))

    def test_winner_native_is_one_attempt_after_exact_failed_predecessors(self):
        command=workflow.minimum_commands(workflow.fit_command('workspace.root','mass',7260,.001))[1]['command']
        previous=dict(status='failed',pid=None,attempts=[dict(name='winner_measured_range_s1_tol'+str(t),
            command=workflow.refinement_command(command,t),exit_code=0,passed=False,
            validation={'reason':'not five expected quantiles'}) for t in (.001,.0001)])
        specs=workflow.winner_native_commands(previous);self.assertEqual(len(specs),1)
        c=specs[0]['command'];self.assertEqual(c[c.index('--minosAlgo')+1],'new')
        self.assertEqual(float(c[c.index('--rMax')+1]),7260);self.assertIn('MINIMIZER_no_analytic',c)
        previous['attempts'][0]['passed']=True
        with self.assertRaises(ValueError):workflow.winner_native_commands(previous)

    def test_winner_medium_recovery_requires_complete_ranking_and_exact_contract(self):
        from TROTASR.workflows import build_nominal_grid as builder, combine_limits as combine
        base=ROOT/'stats/topology_transition_nominal/combined_optimization'
        limits=base/'winner/medium/limits';manifest=limits.parent/'grid/manifest.json'
        candidate=dict(candidate_id='winner',contract_path='candidate-contract',contract_sha256='candidate-sha')
        grid=dict(variant='medium',medium_candidate=candidate)
        rows=[dict(candidate_id='winner' if i==0 else 'other%d'%i,ranking_eligible=True,pending={},
            validation_candidate=candidate,manifest_sha256='digest') for i in range(25)]
        ranking=dict(status='complete',winner='winner',baseline_pending={},candidates=rows)
        pool=dict(points_per_candidate=955,systematics=False,control_region_fit=False,
            candidates=[dict(candidate_id='winner',contract_sha256='candidate-sha')])
        def read(p):
            return {'ranking_primary_pool25.json':ranking,'candidate_pool.json':pool}.get(Path(p).name,grid)
        with patch.object(workflow,'read_json',side_effect=read),patch.object(workflow,'sha256',return_value='digest'), \
                patch.object(combine,'verify_comparison_grid'),patch.object(builder,'load_medium_candidate',return_value=candidate):
            self.assertEqual(workflow.comparison_recovery_scope(manifest,limits,95),'medium')
            self.assertEqual(len(workflow.winner_recovery_contract(manifest,limits,95)),3)
            for workers in (0,96):
                with self.assertRaises(ValueError):workflow.winner_recovery_contract(manifest,limits,workers)
            for change in [dict(status='provisional'),dict(winner='other'),dict(baseline_pending={'p':'missing'})]:
                old=dict(ranking);ranking.update(change)
                with self.assertRaises(ValueError):workflow.winner_recovery_contract(manifest,limits,1)
                ranking.clear();ranking.update(old)
            rows[0]['manifest_sha256']='changed'
            with self.assertRaises(ValueError):workflow.winner_recovery_contract(manifest,limits,1)
            rows[0]['manifest_sha256']='digest';rows[1]['pending']={'p':'missing'}
            with self.assertRaises(ValueError):workflow.winner_recovery_contract(manifest,limits,1)
            rows[1]['pending']={};pool['systematics']=True
            with self.assertRaises(ValueError):workflow.winner_recovery_contract(manifest,limits,1)

    def test_winner_bb_recovery_only_failed_points_and_preserves_primary(self):
        from TROTASR.workflows import continue_limit_fits as continuation, combine_limits as combine
        from TROTASR.workflows import nominal_campaign as campaign
        with tempfile.TemporaryDirectory(dir=ROOT/'tests',prefix='winner_bb_') as folder:
            root=Path(folder);limits=root/'medium/limits';limits.mkdir(parents=True)
            (limits/'controller.lock').touch();key='T2tt/mStop900_mLSP100';work=limits/key;work.mkdir(parents=True)
            point=dict(model='T2tt',mass='mStop900_mLSP100',fit_ready=True)
            other=dict(model='T2tt',mass='mStop1200_mLSP500',fit_ready=True)
            second=limits/other['model']/other['mass'];second.mkdir(parents=True)
            product=root/'accepted.bin';product.write_bytes(b'no-ROOT-test-fixture')
            hashof=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            success=dict(status='complete',pid=None,output='accepted.bin',output_sha256=hashof(product))
            (second/'state.json').write_text(json.dumps(success))
            command=workflow.fit_command('workspace.root',point['mass'],40,.001)
            record=dict(status='failed',pid=None,workspace_sha256='workspace',attempts=[])
            for n in range(4):
                here=work/str(n);here.mkdir();log=here/'combine.log';log.write_text('primary log')
                (here/'combine_logger.out').write_text('The NLL has zero floating parameters\nMinimization finished with status=-1')
                a=dict(command=command,passed=False,exit_code=0,validation={'reason':'not five expected quantiles'},
                    log=str(log.relative_to(root)),log_sha256=hashof(log),output='not-created.bin')
                (here/'validation.json').write_text(json.dumps(a));record['attempts'].append(a)
            (work/'state.json').write_text(json.dumps(record))
            summary=limits/'state.json';summary.write_text(json.dumps(dict(status='needs_attention',pid=None,failed=[key])))
            grid=dict(variant='medium',points=[point,other],medium_candidate=dict(candidate_id='winner'))
            manifest=limits.parent/'manifest.json';manifest.write_text(json.dumps(grid))
            (limits/'contract.json').write_text(json.dumps(dict(manifest_sha256=hashof(manifest),code={})))
            source=root/'workflows/continue_limit_fits.py';source.parent.mkdir();source.write_text('# test fixture\n')
            originals={p:p.read_bytes() for p in (summary,work/'state.json',second/'state.json',product)}
            with patch.object(workflow,'ROOT',root),patch.object(workflow,'internal_path',side_effect=lambda p:p), \
                    patch.object(workflow,'SOURCES',[]),patch.object(workflow,'comparison_recovery_scope',return_value='medium'), \
                    patch.object(workflow,'winner_recovery_contract',return_value={}),patch.object(workflow,'verify_card'), \
                    patch.object(combine,'verify_comparison_grid'),patch.object(continuation,'verify_attempt',return_value=(False,{})), \
                    patch.object(workflow,'runtime',return_value={}),patch.object(campaign,'resources',return_value=(0,128*1024**3)), \
                    patch.object(workflow,'minimum_point',return_value=dict(status='verified_not_promoted')) as run:
                self.assertEqual(workflow.recover_comparison(manifest,limits,1),0)
                self.assertEqual(run.call_count,1)
                self.assertEqual(run.call_args.args[0]['point'],point)
                self.assertEqual(len(run.call_args.args[0]['attempts']),2)
                with self.assertRaisesRegex(ValueError,'Existing finite'):
                    workflow.recover_comparison(manifest,limits,1)
            self.assertEqual({p:p.read_bytes() for p in originals},originals)

    def candidate_failure(self):
        command=workflow.fit_command('workspace.root','mStop1700_mLSP1500',192.634,.001)
        validation=dict(quantiles={str(q):v for q,v in zip((.025,.16,.5,.84,.975),(33.78,45.63,64.90,93.23,128.41))},
            crossings=[dict(quantile=q,passed=q!=.5,bracket_width=.376239)
                for q in (.5,.025,.16,.84,.975)])
        return dict(status='failed',pid=None,workspace_valid=True,workspace_sha256='workspace',
            attempts=[dict(name=n,passed=False,exit_code=0,command=command,validation=deepcopy(validation))
                for n in ('bisection_s0','bisection_s0_refined','nominal_mcstat_s1','nominal_mcstat_s2')])

    def test_candidate_crossing_tightens_only_numerical_search(self):
        before=self.candidate_failure();record=deepcopy(before)
        specs=workflow.candidate_crossing_commands(record)
        self.assertEqual(record,before);self.assertEqual(len(specs),2)
        self.assertEqual([s['tolerance'] for s in specs],[.001,.0001])
        for s in specs:
            c=s['command'];self.assertNotIn('MINIMIZER_no_analytic',c)
            self.assertEqual(c[c.index('--rMax')+1],'192.634')
            self.assertEqual(c[c.index('--run')+1],'blind')
            self.assertEqual(c[c.index('--minosAlgo')+1],'bisection')
            self.assertEqual(c[c.index('--rRelAcc')+1],'0.00025')
            self.assertEqual(c[c.index('--rAbsAcc')+1],'1e-06')
            self.assertEqual(c[c.index('--cminDefaultMinimizerStrategy')+1],'1')
            self.assertNotIn('--freezeParameters',c)
        # The historical explicit-BB entry remains strict by default.
        with self.assertRaises(ValueError):workflow.refinement_command(before['attempts'][-1]['command'],.001)

    def test_candidate_crossing_refuses_success_running_other_failure_or_model_change(self):
        changes=[lambda r:r.update(status='complete'),lambda r:r.update(pid=100),
            lambda r:r.update(workspace_valid=False),lambda r:r['attempts'].pop(),
            lambda r:r['attempts'][0].update(passed=True),
            lambda r:r['attempts'][-1].update(exit_code=1),
            lambda r:r['attempts'][-1]['validation'].update(crossings=[]),
            lambda r:r['attempts'][-1]['validation']['quantiles'].update({'0.5':float('nan')}),
            lambda r:r['attempts'][-1]['validation']['quantiles'].update({'0.5':-1}),
            lambda r:r['attempts'][-1]['command'].extend(['--freezeParameters','r']),
            lambda r:r['attempts'][-1]['command'].__setitem__(
                r['attempts'][-1]['command'].index('--rMax')+1,'100001')]
        for i,change in enumerate(changes):
            r=self.candidate_failure();change(r)
            with self.subTest(case=i),self.assertRaises(ValueError):workflow.candidate_crossing_commands(r)

    def test_candidate_scope_requires_frozen_pool_and_exact_candidate(self):
        from TROTASR.workflows import build_nominal_grid as builder
        from TROTASR.workflows import combine_limits as combine
        base=ROOT/'stats/topology_transition_nominal/combined_optimization/unit/combined'
        c=dict(candidate_id='unit',contract_path='unit-contract',contract_sha256='digest')
        point=dict(model='T2tt',mass='mStop1700_mLSP1500',fit_ready=True)
        g=dict(variant='combined',medium_candidate=c,points=[point])
        pool=dict(schema='combined_expected_cls_candidate_pool_v1',points_per_candidate=955,
            systematics=False,control_region_fit=False,candidates=[dict(candidate_id='unit',contract_sha256='digest')])
        def read(p):return pool if Path(p).name=='candidate_pool.json' else g
        with patch.object(workflow,'read_json',side_effect=read),patch.object(combine,'verify_comparison_grid'), \
                patch.object(builder,'load_medium_candidate',return_value=c):
            args=(base/'grid/manifest.json',base/'limits','T2tt/mStop1700_mLSP1500')
            self.assertEqual(workflow.candidate_recovery_scope(*args,1)[:2],(g,point))
            with self.assertRaises(ValueError):workflow.candidate_recovery_scope(*args,2)
            with self.assertRaises(ValueError):workflow.candidate_recovery_scope(args[0],ROOT/'other',args[2],1)
            with self.assertRaises(ValueError):workflow.candidate_recovery_scope(args[0],args[1],'T2tt/mStop1_mLSP1',1)
            pool['candidates'][0]['contract_sha256']='other'
            with self.assertRaises(ValueError):workflow.candidate_recovery_scope(*args,1)

    def score_pool_fixture(self):
        base=ROOT/'stats/topology_transition_nominal/combined_optimization'
        definition=dict(nbins=20,groups=[dict(conditions=[['category',i,True]],
            score_edges=[0.,.2,.4,.6,.8,1.]) for i in range(4)])
        candidate=dict(candidate_id='unit',contract_path='/approved/unit/contract.json',
            contract_sha256='digest',category_definition=definition)
        point=dict(model='T2tb',mass='mStop1600_mLSP1500',fit_ready=True)
        grid=dict(variant='combined',medium_candidate=candidate,points=[point])
        reference=dict(id='tr_sig005_latest_9c05c80dfe80',category_definition=deepcopy(definition))
        row=dict(id='unit',contract=candidate['contract_path'],contract_sha256='digest',
            category_definition=deepcopy(definition),bins=20,status='complete')
        pool=dict(schema='score_edge_comparison_pool_v1',points_per_candidate=955,
            systematics=False,control_region_fit=False,partition='validation',
            physical_partitions_frozen=True,reference=reference['id'],candidates=[reference,row])
        paths=(base/'unit/combined/grid/manifest.json',base/'unit/combined/limits',
               'T2tb/mStop1600_mLSP1500',1,base/'score_edges_20261013_pool.json')
        return candidate,point,grid,pool,paths

    def test_explicit_score_pool_binds_id_contract_and_physical_categories(self):
        from TROTASR.workflows import build_nominal_grid as builder,combine_limits as combine
        candidate,point,grid,pool,args=self.score_pool_fixture()
        reads=[]
        def read(path):
            reads.append(path)
            return pool if path==args[-1] else grid
        with patch.object(workflow,'read_json',side_effect=read),patch.object(combine,'verify_comparison_grid'), \
                patch.object(builder,'load_medium_candidate',return_value=candidate):
            self.assertEqual(workflow.candidate_recovery_scope(*args),(grid,point,args[-1]))
            self.assertNotIn(args[-1].parent/'candidate_pool.json',reads)
            for wrong in (ROOT/'jsons/pool.json',args[-1].parent/'subdir/pool.json',args[-1].with_suffix('.txt')):
                with self.assertRaisesRegex(ValueError,'explicit JSON'):
                    workflow.candidate_recovery_scope(*args[:-1],wrong)

    def test_explicit_score_pool_rejects_changed_or_ambiguous_contracts(self):
        from TROTASR.workflows import build_nominal_grid as builder,combine_limits as combine
        candidate,point,grid,original,args=self.score_pool_fixture()
        changes=[lambda p:p.update(schema='unknown'),lambda p:p.update(partition='test'),
            lambda p:p.update(test_events_used=True),lambda p:p.update(physical_partitions_frozen=False),
            lambda p:p.update(systematics=True),lambda p:p.update(control_region_fit=True),
            lambda p:p.update(points_per_candidate=954),lambda p:p.update(reference='other'),
            lambda p:p['candidates'].append(deepcopy(p['candidates'][-1])),
            lambda p:p['candidates'][-1].update(id='another'),
            lambda p:p['candidates'][-1].update(contract_sha256='wrong'),
            lambda p:p['candidates'][-1].update(contract='/another/contract.json'),
            lambda p:p['candidates'][-1].update(status='running'),
            lambda p:p['candidates'][-1].update(bins=25),
            lambda p:p['candidates'][0]['category_definition']['groups'][0].update(conditions=[]),
            lambda p:p['candidates'][-1]['category_definition']['groups'][0].update(score_edges=[0,.1,.4,.6,.8,1])]
        for i,change in enumerate(changes):
            pool=deepcopy(original);change(pool)
            with self.subTest(case=i),patch.object(workflow,'read_json',side_effect=lambda p:pool if p==args[-1] else grid), \
                    patch.object(combine,'verify_comparison_grid'),patch.object(builder,'load_medium_candidate',return_value=candidate):
                with self.assertRaises(ValueError):workflow.candidate_recovery_scope(*args)

    def test_candidate_recovery_preserves_primary_and_rejects_resubmission(self):
        from TROTASR.workflows import continue_limit_fits as continuation
        from TROTASR.workflows import nominal_campaign as campaign
        with tempfile.TemporaryDirectory(dir=ROOT/'tests',prefix='candidate_recovery_') as folder:
            root=Path(folder);limits=root/'limits';limits.mkdir();(limits/'controller.lock').touch()
            point=dict(model='T2tt',mass='mStop1700_mLSP1500',fit_ready=True)
            key=point['model']+'/'+point['mass'];work=limits/key;work.mkdir(parents=True)
            other=dict(model='T2tt',mass='mStop1200_mLSP500',fit_ready=True)
            second=limits/other['model']/other['mass'];second.mkdir(parents=True)
            product=root/'accepted.bin';product.write_bytes(b'no-ROOT-test-fixture')
            hashof=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            success=dict(status='complete',pid=None,output='accepted.bin',output_sha256=hashof(product))
            (second/'state.json').write_text(json.dumps(success))
            record=self.candidate_failure()
            for a in record['attempts']:
                here=work/a['name'];here.mkdir();a['log']=str((here/'combine.log').relative_to(root))
                (here/'validation.json').write_text(json.dumps(a))
            (work/'state.json').write_text(json.dumps(record))
            summary=limits/'state.json';summary.write_text(json.dumps(dict(status='needs_attention',pid=None,failed=[key])))
            grid=dict(points=[point,other],medium_candidate=dict(contract_sha256='candidate'))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(grid))
            (limits/'contract.json').write_text(json.dumps(dict(manifest_sha256=hashof(manifest),code={})))
            pool=root/'candidate_pool.json';pool.write_text('{}')
            dependency=root/'workflows/continue_limit_fits.py';dependency.parent.mkdir();dependency.write_text('# fixture\n')
            originals={p:p.read_bytes() for p in (product,summary,second/'state.json',work/'state.json')}
            with patch.object(workflow,'ROOT',root),patch.object(workflow,'SOURCES',[]), \
                    patch.object(workflow,'candidate_recovery_scope',return_value=(grid,point,pool)) as scope, \
                    patch.object(workflow,'verify_card'),patch.object(workflow,'runtime',return_value={}), \
                    patch.object(continuation,'verify_attempt',return_value=(False,{})), \
                    patch.object(campaign,'resources',return_value=(0,128*1024**3)), \
                    patch.object(workflow,'minimum_point',return_value=dict(status='verified_not_promoted')) as run:
                self.assertEqual(workflow.refine_candidate_point(manifest,limits,key,1,candidate_pool=pool),0)
                scope.assert_called_once_with(manifest,limits,key,1,pool)
                self.assertEqual(run.call_count,1)
                self.assertEqual(len(run.call_args.args[0]['attempts']),2)
                with self.assertRaisesRegex(ValueError,'Existing finite'):
                    workflow.refine_candidate_point(manifest,limits,key,1)
                self.assertEqual(run.call_count,1)
            self.assertEqual({p:p.read_bytes() for p in originals},originals)
            out=limits/'numerical_crossing_20261011'
            contract=json.loads((out/'contract.json').read_text())
            self.assertEqual(contract['pool'],str(pool.relative_to(root)))
            self.assertEqual(contract['maximum_new_attempts_per_point'],2)
            self.assertFalse(contract['primary_promotion']);self.assertFalse(contract['physics_model_changed'])

    def test_contour_only_never_exports_failed_two_sigma_or_promotes_full_fit(self):
        from TROTASR.workflows import plotting_limit_contour as contour
        q={.025:69.,.16:97.,.5:148.,.84:256.,.975:449.}
        validation=dict(quantiles=q,crossings=[dict(quantile=x,passed=x!=.975) for x in q])
        selected=contour.contour_quantiles(validation)
        self.assertEqual(selected,{.16:97.,.5:148.,.84:256.})
        validation['crossings'][1]['passed']=False
        self.assertIsNone(contour.contour_quantiles(validation))
        p=dict(model='T2bW',mass='mStop1650_mLSP1350');key=p['model']+'/'+p['mass']
        grid=dict(points=[p],scope='sr_only_nominal_mcstat_comparison')
        results={key:dict(status='failed')}
        evidence={key:dict(accepted=None,contour_only=dict(quantiles=selected,full_five_quantile_passed=False))}
        payload=contour.collect_contour_quantiles(grid,results,evidence,'T2bW')
        self.assertEqual(results[key]['status'],'failed');self.assertIsNone(evidence[key]['accepted'])
        self.assertEqual(payload['full_five_quantile_status'],'partial')
        self.assertEqual(payload['full_five_quantile_missing'],[p['mass']])
        self.assertEqual(payload['collected_point_count'],1);self.assertEqual(payload['missing_points'],[])
        self.assertNotIn('expected_p2',payload['points'][p['mass']])
        self.assertNotIn('expected_m2',payload['points'][p['mass']])
        contour.validate_collected(payload,[p['mass']])
        payload['points'][p['mass']]['expected_p2']=449.
        with self.assertRaises(ValueError):contour.validate_collected(payload,[p['mass']])

    def test_native_log_parser_excludes_unprofiled_approximate_and_bad_bare_rows(self):
        header='Will search for NLL crossing with new algorithm\n'
        block=('At r = 10.0: delta(nll unprof) = 8.0 delta(nll prof) = 1.9 kappa=0\n'
               'At r = 11.0: delta(nll unprof) = 9.0 delta(nll appr) = 1.9 kappa=0\n'
               'At r = 12.0: delta(nll prof) = 9.0 kappa=0\n')
        normalized=workflow.native_profiled_text((header+block)*5)
        self.assertEqual(normalized.count('At r = '),5)
        self.assertNotIn('11.0',normalized);self.assertNotIn('12.0',normalized)
        self.assertEqual(normalized.count('delta(nll) = 1.9'),5)
        for text in ((header+block)*4,(header+'At r = 10.0: delta(nll unprof) = 9.0\n')*5,
                     (header+block.replace('10.0','100001.0'))*5):
            with self.assertRaises(ValueError):workflow.native_profiled_text(text)

    def test_native_comparison_has_separate_finite_history(self):
        base=ROOT/'stats/topology_transition_nominal/low'
        self.assertEqual(workflow.refinement_scope(base/'grid/manifest.json',base/'limits',
            'T2bW/mStop1650_mLSP1350',True,True)[1:],
            ('numerical_crossing_20261011','numerical_native_bracket_20261011'))
        with self.assertRaises(ValueError):workflow.refinement_scope(base/'grid/manifest.json',
            base/'limits','T2bW/mStop650_mLSP1',False,True)

    def test_comparison_refinement_only_diagnosed_point(self):
        base=ROOT/'stats/topology_transition_nominal/low'
        args=(base/'grid/manifest.json',base/'limits','T2bW/mStop1650_mLSP1350')
        self.assertEqual(workflow.refinement_scope(*args,comparison=True),
            ('sr_only_nominal_mcstat_comparison','numerical_comparison_20261011',
             'numerical_crossing_20261011'))
        with self.assertRaises(ValueError):workflow.refinement_scope(*args)
        with self.assertRaises(ValueError):workflow.refinement_scope(args[0],args[1],'T2bW/mStop650_mLSP1',True)
        with self.assertRaises(ValueError):workflow.refinement_scope(args[0],base.parent/'high/limits',args[2],True)

    def test_historical_source_uses_exact_hash_without_replacing_old_archive(self):
        from TROTASR.workflows import reconcile_limit_fits as reconcile
        def digest_of(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'stats/campaign/limits'
            current=root/'workflows/example.py';current.parent.mkdir(parents=True)
            current.write_text('current\n')
            legacy=output.parent/'history/recovery_20261009/source/example.py'
            legacy.parent.mkdir(parents=True);legacy.write_text('older\n')
            digest=hashlib.sha256(b'previous\n').hexdigest()
            archive=output.parent/('history/comparison/source/example_'+digest[:8]+'.py')
            archive.parent.mkdir(parents=True);archive.write_text('previous\n')
            with patch.object(reconcile,'ROOT',root),patch.object(reconcile,'sha256',side_effect=digest_of):
                self.assertEqual(reconcile.historical_source(output,'workflows/example.py',digest),archive)
                self.assertEqual(reconcile.historical_source(output,'workflows/example.py',digest_of(legacy)),legacy)
                with self.assertRaises(ValueError):reconcile.historical_source(output,'workflows/example.py','0'*64)
            self.assertEqual(legacy.read_text(),'older\n')

    def test_comparison_only_two_explicit_strategies_and_same_range(self):
        base=workflow.fit_command('workspace.root','mStop400_mLSP1',.22896,.001)
        record=dict(status='failed',pid=None,attempts=[dict(command=base,passed=False,
            validation={'reason':'not five expected quantiles'}) for _ in range(4)])
        logger='The NLL has zero floating parameters\nMinimization finished with status=-1'
        specs=workflow.comparison_commands(record,logger)
        self.assertEqual([s['name'] for s in specs],['explicit_s0_prefit','explicit_combined_s0'])
        for s in specs:
            c=s['command']
            self.assertIn('MINIMIZER_no_analytic',c)
            self.assertEqual(c[c.index('--rMax')+1],'.22896' if '.22896' in base else '0.22896')
            self.assertEqual(c[c.index('--run')+1],'blind')
            self.assertNotIn('--freezeParameters',c)
        for change in (dict(status='complete'),dict(pid=42),dict(attempts=record['attempts'][:3])):
            with self.assertRaises(ValueError):workflow.comparison_commands(dict(record,**change),logger)
        with self.assertRaises(ValueError):workflow.comparison_commands(record,'range too small')
        record['attempts'][0]['passed']=True
        with self.assertRaises(ValueError):workflow.comparison_commands(record,logger)

    def test_comparison_recovery_scope_is_seven_diagnosed_points(self):
        self.assertEqual(len(workflow.COMPARISON_FAILURES),7)
        self.assertEqual(workflow.COMPARISON_RECOVERY,'numerical_comparison_20261011')
        self.assertTrue(all(int(k.split('mStop')[1].split('_')[0])<600 for k in workflow.COMPARISON_FAILURES))

    def test_single_sr_recovery_keeps_scope_and_worker_bound_explicit(self):
        for variant in ('high','low'):
            base=ROOT/'stats/topology_transition_nominal'/variant
            self.assertEqual(workflow.comparison_recovery_scope(base/'grid/manifest.json',base/'limits',40,True),variant)
            with self.assertRaises(ValueError):workflow.comparison_recovery_scope(base/'grid/manifest.json',base/'limits',41,True)
            with self.assertRaises(ValueError):workflow.comparison_recovery_scope(base/'grid/manifest.json',base/'limits',7)
        base=ROOT/'stats/topology_transition_nominal/medium'
        self.assertEqual(workflow.comparison_recovery_scope(base/'grid/manifest.json',base/'limits',7),'medium')
        with self.assertRaises(ValueError):workflow.comparison_recovery_scope(base/'grid/manifest.json',base/'limits',7,True)

    def test_minimum_recovery_preserves_likelihood_and_acceptance(self):
        base=command('workspace.root','mStop650_mLSP1',.146211)
        specs=workflow.minimum_commands(base)
        self.assertEqual(len(specs),3)
        self.assertEqual(len({tuple(x['command']) for x in specs}),3)
        self.assertEqual([s['explicit_stats'] for s in specs],[False,True,True])
        for spec in specs:
            c=spec['command']
            for flag,val in [('--rMax','0.146211'),('--run','blind'),('--minosAlgo','bisection'),
                    ('--rRelAcc','0.0025'),('--rAbsAcc','1e-05'),
                    ('--cminDefaultMinimizerStrategy','0'),('--cminDefaultMinimizerTolerance','0.01')]:
                self.assertEqual(c[c.index(flag)+1],val)
            self.assertIn('--picky',c)
            self.assertIn('MINIMIZER_MaxCalls=5000000',c)
            self.assertNotIn('MINIMIZER_MaxCalls=2000000',c)
            for flag in ('--freezeParameters','--setParameters','--setParameterRanges','--noFitAsimov'):
                self.assertNotIn(flag,c)
        for extra in [[],['--freezeParameters','x'],['--snapshotName','other']]:
            invalid=list(base)+extra
            if not extra:invalid.remove('--picky')
            with self.assertRaises(ValueError):workflow.minimum_commands(invalid)

    def test_refinement_preserves_model_and_range(self):
        base=command('workspace.root','mStop650_mLSP1',.146211)
        for tolerance in (.001,.0001):
            refined=workflow.refinement_command(base,tolerance)
            self.assertEqual(refined[refined.index('--rMax')+1],'0.146211')
            self.assertEqual(refined[refined.index('--cminFallbackAlgo')+1],'Minuit2,Migrad,2:'+str(tolerance))
            self.assertEqual(refined[refined.index('--rRelAcc')+1],'0.00025')
            for flag in ('--freezeParameters','--setParameterRanges','--noFitAsimov'):
                self.assertNotIn(flag,refined)
        with self.assertRaises(ValueError):workflow.refinement_command(base,.1)

    def test_exact_scope(self):
        self.assertEqual(len(FAILURES),3)
        self.assertNotIn(CONTROL,FAILURES)
        self.assertEqual(CONTROL,'T2tt/mStop1200_mLSP500')

    def test_original_control_point_level_command_schema(self):
        keys=[CONTROL,*FAILURES]
        grid={'points':[dict(zip(('model','mass'),k.split('/'))) for k in keys]}
        records={}
        for k in keys:
            records[k]=dict(status='complete' if k==CONTROL else 'failed',pid=None,
                workspace_sha256='sha',command=['combine','--rMax','40'],
                attempts=[dict(validation={'reason':FAILURES.get(k)})])
        def read(path):return records['/'.join(Path(path).parts[-3:-1])]
        with patch.object(workflow,'read_json',side_effect=read),patch.object(workflow,'sha256',return_value='sha'):
            plan=workflow.plan_points(grid,ROOT/'stats/study_test')
        self.assertEqual(len(plan),4)
        self.assertTrue(all(p['command'][p['command'].index('--rMax')+1]=='40.0' for p in plan))

    def test_same_likelihood_and_blind_scan(self):
        c=command('workspace.root','mass',310)
        for k,v in [('--run','blind'),('--minosAlgo','bisection'),('--rMax','310'),
                    ('--rRelAcc','0.0025'),('--rAbsAcc','1e-05'),
                    ('--cminDefaultMinimizerTolerance','0.0001')]:
            self.assertEqual(c[c.index(k)+1],v)
        for k in ['--freezeParameters','--setParameterRanges','--noFitAsimov','--toysFrequentist']:
            self.assertNotIn(k,c)
        self.assertIn('--picky',c)
        self.assertIn('MINIMIZER_no_analytic',c)
        self.assertIn('MINIMIZER_MaxCalls=2000000',c)

    def test_bad_ranges(self):
        for r in [0,-1,10001,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):command('w','m',r)

    def test_control_agreement(self):
        a={str(q):i+1 for i,q in enumerate([.025,.16,.5,.84,.975])}
        self.assertTrue(control_comparison(a,a)['passed'])
        self.assertFalse(control_comparison(a,{k:1.02*v for k,v in a.items()})['passed'])

    def test_control_missing_quantiles(self):
        with self.assertRaises(ValueError):control_comparison({'0.5':1},{'0.5':1})


class Run3FiniteContinuationTests(unittest.TestCase):
    """No native fits: saved-state, bounded-command and admission contracts."""
    def base(self):
        return workflow.minimum_commands(workflow.fit_command('workspace.root','mass',100,.01))[1]['command']

    def crossing(self):
        return dict(name='new_minimum',command=self.base(),exit_code=0,passed=False,log_sha256='log',
            validation=dict(quantiles={str(q):i+1. for i,q in enumerate(QUANTILES)},
                crossings=[dict(quantile=q,passed=i!=4) for i,q in enumerate(QUANTILES)]))

    def test_precision_four_distinct_and_exact_dedup(self):
        with patch.object(continuation,'adaptive_seed',return_value=(100,{'saved':'profile'})):
            values,evidence=workflow.run3_precision_commands('workspace.root','mass',[],[])
            self.assertEqual(len(values),4)
            self.assertEqual(len({tuple(v['command']) for v in values}),4)
            self.assertEqual([v['explicit_stats'] for v in values],[False,False,True,True])
            for value in values:
                cmd=value['command'];self.assertEqual(cmd[cmd.index('--run')+1],'blind')
                self.assertEqual(cmd[cmd.index('--cminDefaultMinimizerPrecision')+1],'1e-08')
                self.assertNotIn('--snapshotName',cmd)
            before=deepcopy(values)
            left,_=workflow.run3_precision_commands('workspace.root','mass',[],[values[0]['command']])
            self.assertEqual(len(left),3);self.assertEqual(values,before)

    def test_crossing_diagnosis_and_dedup(self):
        attempt=self.crossing();before=deepcopy(attempt)
        specs,diagnosis=workflow.run3_crossing_commands([attempt]);self.assertEqual(len(specs),2)
        self.assertEqual(attempt,before);self.assertEqual(diagnosis['basis'],'new_minimum')
        again,diagnosis=workflow.run3_crossing_commands([attempt],[s['command'] for s in specs])
        self.assertFalse(again);self.assertEqual(diagnosis['status'],'not_applicable')
        for mutate in (lambda a:a.update(passed=True),lambda a:a.update(exit_code=1),
                lambda a:a['validation']['quantiles'].update({'0.5':float('nan')}),
                lambda a:a['validation']['crossings'].clear(),
                lambda a:[c.update(passed=True) for c in a['validation']['crossings']]):
            candidate=deepcopy(attempt);mutate(candidate)
            self.assertFalse(workflow.run3_crossing_commands([candidate])[0])

    def test_admission_116_restore_and_no_fork_inheritance(self):
        manifest=common.ROOT/'stats/run3_systematics_20261025/grid/manifest.json'
        self.assertEqual(common.fit_admission_slots(),100)
        with workflow.run3_continuation_admission(manifest):
            self.assertEqual(common.fit_admission_slots(),116)
            values=[]
            worker=threading.Thread(target=lambda:values.append(common.fit_admission_slots()))
            worker.start();worker.join();self.assertEqual(values,[116])
            pid=common.os.getpid()
            with patch.object(common.os,'getpid',return_value=pid+1):
                self.assertEqual(common.fit_admission_slots(),100)
        self.assertEqual(common.fit_admission_slots(),100)
        with self.assertRaises(ValueError):
            with workflow.run3_continuation_admission(manifest.with_name('wrong.json')):pass

    def test_quiescent_guard_holds_both_locks_and_rejects_live_orphan(self):
        with tempfile.TemporaryDirectory(dir=workflow.ROOT/'tests') as folder:
            root=Path(folder);limits=root/'limits';point=dict(model='T2tt',mass='mStop950_mLSP1')
            directories=(limits,limits/'numerical_all_unfinished_20260930')
            for directory in directories:
                directory.mkdir(parents=True,exist_ok=True);(directory/'controller.lock').touch()
                workflow.write_json(directory/'state.json',dict(status='needs_attention',pid=None,active=[]))
            with workflow.run3_quiescent_pool(limits,[point]):
                for directory in directories:
                    with (directory/'controller.lock').open('r') as lock:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            workflow.write_json(directories[1]/'state.json',dict(status='running',pid=123))
            with self.assertRaises(ValueError):
                with workflow.run3_quiescent_pool(limits,[point]):pass
            workflow.write_json(directories[1]/'state.json',dict(status='needs_attention',pid=None))
            path=limits/point['model']/point['mass']/'state.json';path.parent.mkdir(parents=True)
            workflow.write_json(path,dict(status='running',pid=456))
            with self.assertRaises(ValueError):
                with workflow.run3_quiescent_pool(limits,[point]):pass

    def test_stage_merges_distinct_keys_and_rejects_contract_drift(self):
        with tempfile.TemporaryDirectory(dir=workflow.ROOT/'tests') as folder:
            out=Path(folder);contract=dict(code={'a':'hash'});errors=[]
            def update(index):
                try:workflow._run3_stage_update(out,contract,str(index),dict(status='failed'))
                except Exception as exc:errors.append(exc)
            workers=[threading.Thread(target=update,args=(i,)) for i in range(8)]
            for worker in workers:worker.start()
            for worker in workers:worker.join()
            self.assertFalse(errors)
            self.assertEqual(len(workflow.read_json(out/'state.json')['results']),8)
            with self.assertRaises(ValueError):
                workflow._run3_stage_update(out,{'changed':True},'9',dict(status='failed'))

    def exercise_minimum(self,conditional,pass_at=None):
        with tempfile.TemporaryDirectory(dir=workflow.ROOT/'tests') as folder:
            root=Path(folder);limits=root/'limits';point=dict(model='T2tt',mass='mStop950_mLSP1')
            work=limits/point['model']/point['mass'];work.mkdir(parents=True);(work/'point.lock').touch()
            workspace=work/'workspace.root';workspace.write_bytes(b'test-not-ROOT')
            out=root/'stage';out.mkdir();specs=workflow.minimum_commands(self.base());calls=[]
            spec=dict(point=point,workspace_sha256=workflow.sha256(workspace),protected_states={},attempts=specs,
                run3_conditional_crossings=conditional,run3_finite_continuation=conditional)
            class Runner:
                def __init__(self,output):pass
                def run(self,command,cwd,log,products,env):
                    calls.append(command);log.write_text('mock native log');products[0].write_bytes(b'mock product')
                    return dict(exit_code=0)
            def verify(point,attempt):
                if pass_at==len(calls):return True,{'quantiles':{}}
                return False,self.crossing()['validation']
            disk=type('Disk',(),{'free':100*1024**3})()
            with patch.object(workflow,'ROOT',root),patch.object(workflow,'verify_card'), \
                    patch.object(workflow,'CommandRunner',Runner), \
                    patch.object(continuation,'verify_attempt',side_effect=verify), \
                    patch.object(workflow.shutil,'disk_usage',return_value=disk):
                result=workflow.minimum_point(spec,limits,out,{}, {}, None)
            saved=workflow.read_json(out/point['model']/point['mass']/'state.json')
            return calls,result,saved

    def test_new_minimum_result_enables_only_two_crossings(self):
        calls,result,state=self.exercise_minimum(True)
        self.assertEqual(len(calls),5);self.assertEqual(result['status'],'failed')
        self.assertEqual(state['conditional_crossings']['basis'],'explicit_combined_s0')
        self.assertEqual([a['name'] for a in state['attempts'][:3]],
            ['analytic_s0_prefit','explicit_s0_prefit','explicit_combined_s0'])

    def test_legacy_stays_three_and_first_success_stops(self):
        self.assertEqual(len(self.exercise_minimum(False)[0]),3)
        calls,result,state=self.exercise_minimum(True,pass_at=2)
        self.assertEqual(len(calls),2);self.assertEqual(result['status'],'verified_not_promoted')
        self.assertNotIn('conditional_crossings',state)

    def test_preflight_during_active_pool_but_actual_run_forbidden(self):
        with tempfile.TemporaryDirectory(dir=workflow.ROOT/'tests') as folder,contextlib.ExitStack() as stack:
            root=Path(folder);limits=root/'stats/run3_systematics_20261025/limits';limits.mkdir(parents=True)
            point=dict(model='T2tt',mass='mStop950_mLSP1');key=point['model']+'/'+point['mass']
            work=limits/key;work.mkdir(parents=True);(work/'point.lock').touch()
            workspace=work/'workspace.root';workspace.write_bytes(b'fixture-only');ws=workflow.sha256(workspace)
            primary_names={'workflows/combine_limits.py','workflows/limit_validation.py','stats/runtime/sitecustomize.py',
                'stats/runtime.json','workflows/reconcile_limit_fits.py','workflows/build_nominal_grid.py'}
            for name in primary_names|set(workflow.SOURCES)|set(recovery.SOURCES):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# exact fixture\n')
            manifest=limits.parent/'grid/manifest.json';manifest.parent.mkdir()
            workflow.write_json(manifest,{'points':[point]});selection={'selected':895}
            workflow.write_json(limits/'contract.json',dict(manifest_sha256=workflow.sha256(manifest),
                fit_selection=selection,code={name:workflow.sha256(root/name) for name in primary_names}))
            original=dict(status='failed',pid=None,workspace_valid=True,workspace_sha256=ws,attempts=[])
            workflow.write_json(work/'state.json',original)
            previous=limits/'numerical_all_unfinished_20260930';target=previous/key;target.mkdir(parents=True)
            workflow.write_json(previous/'contract.json',dict(primary_contract_sha256=workflow.sha256(limits/'contract.json'),
                manifest_sha256=workflow.sha256(manifest),fit_selection=selection,code={}))
            workflow.write_json(target/'contract.json',dict(point=point,code={},maximum_new_attempts=2,
                protected_states={str((work/'state.json').relative_to(root)):workflow.sha256(work/'state.json')},workspace_sha256=ws))
            workflow.write_json(target/'state.json',dict(status='failed',pid=None,attempts=[{'command':['old1']},{'command':['old2']}],
                error='two_unfinished_recovery_configurations_exhausted',contract_sha256=workflow.sha256(target/'contract.json')))
            for path in (limits,previous):
                (path/'controller.lock').touch();workflow.write_json(path/'state.json',dict(status='running',pid=123))
            stack.enter_context(patch.object(workflow,'ROOT',root))
            stack.enter_context(patch.object(workflow,'internal_path',side_effect=Path))
            stack.enter_context(patch.object(combine,'selected_fit_points',return_value=([point],selection)))
            stack.enter_context(patch.object(recovery,'recovery_code_versions',return_value=[{}]))
            stack.enter_context(patch.object(workflow,'_run3_verify_attempts',return_value=[]))
            stack.enter_context(patch.object(workflow,'verify_card'))
            run=stack.enter_context(patch.object(workflow,'minimum_point'))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(workflow.continue_run3_point(manifest,limits,key,True),0)
            self.assertIn('preflight_passed',output.getvalue())
            self.assertFalse((limits/workflow.MINIMUM_RECOVERY).exists())
            with self.assertRaisesRegex(ValueError,'terminal drain'):
                workflow.continue_run3_point(manifest,limits,key)
            run.assert_not_called();self.assertFalse((limits/workflow.MINIMUM_RECOVERY).exists())
            with self.assertRaisesRegex(ValueError,'outside'):
                workflow.continue_run3_point(manifest,limits,'T2tt/mStop400_mLSP1',True)
            workflow.write_json(work/'state.json',dict(original,pid=321,status='running'))
            with self.assertRaisesRegex(ValueError,'terminal failed'):
                workflow.continue_run3_point(manifest,limits,key,True)

    def test_retained_success_requires_validated_attempt_and_output_presence(self):
        with tempfile.TemporaryDirectory(dir=workflow.ROOT/'tests') as folder:
            root=Path(folder);log=root/'combine.log';log.write_text('fixture')
            product=root/'output.bin';product.write_bytes(b'fixture')
            attempt=dict(log='combine.log',output='output.bin',log_sha256=workflow.sha256(log),
                output_sha256=workflow.sha256(product),command=[],exit_code=0)
            workflow.write_json(root/'validation.json',attempt)
            saved=dict(status='verified_not_promoted',attempts=[attempt],accepted_attempt=attempt,
                output='output.bin',output_sha256=attempt['output_sha256'])
            with patch.object(workflow,'ROOT',root),patch.object(continuation,'verify_attempt',return_value=(False,{})):
                with self.assertRaisesRegex(ValueError,'lacks exact'):
                    workflow._run3_verify_attempts({},None,saved)
            with patch.object(workflow,'ROOT',root),patch.object(continuation,'verify_attempt',return_value=(True,{})):
                self.assertEqual(len(workflow._run3_verify_attempts({},None,saved)),1)
                # Delete only this temporary, non-ROOT test fixture.
                product.unlink()
                with self.assertRaisesRegex(ValueError,'evidence changed'):
                    workflow._run3_verify_attempts({},None,saved)

    def native_minima(self):
        specs=workflow.minimum_commands(workflow.fit_command('workspace.root','mass',100,.01))[1:]
        return [dict(s,exit_code=0,passed=False,log=s['name']+'.log',log_sha256='loghash',output_sha256='roothash',
            validation=dict(quantiles={str(q):i+1. for i,q in enumerate(QUANTILES)},
                crossings=[dict(quantile=q,passed=q!=.5) for q in QUANTILES])) for s in specs]

    def native_texts(self,values):
        text=('Will search for NLL crossing by bisection\nAt r = 50: delta(nll) = 0.1\n'
              'At r = 75: delta(nll) = 0.2\nAt r = 99: delta(nll) = 0.3\n')
        return {a['log']:text for a in values}

    def native_attempted(self,values):
        return [dict(s,exit_code=0,passed=False,log=s['name']+'.log',log_sha256='loghash',output_sha256='root',
            validation=dict(reason='not five expected quantiles')) for s in values]

    def test_upper_only_native_exact_one_and_dedup(self):
        base=self.native_minima()[0]['command'];values=[]
        for tolerance in (.001,.0001):
            attempt=self.native_minima()[0]
            attempt.update(name='crossing_refined_s1_tol'+str(tolerance),
                command=workflow.refinement_command(base,tolerance))
            for crossing in attempt['validation']['crossings']:
                crossing['passed']=crossing['quantile']!=.975
            values.append(attempt)
        before=deepcopy(values);specs,diagnosis=workflow.run3_native_stepping_proposals(values,{})
        self.assertEqual(values,before);self.assertEqual(len(specs),1)
        self.assertEqual(diagnosis['branch'],'upper_only_native');self.assertIn('new',specs[0]['command'])
        self.assertFalse(workflow.run3_native_stepping_proposals(values,{},[specs[0]['command']])[0])
        self.assertFalse(workflow.run3_native_stepping_proposals(values+self.native_attempted(specs),{})[0])

    def test_measured_range_native_stepping_are_finite_and_real(self):
        minima=self.native_minima();text=self.native_texts(minima)
        ranges,diagnosis=workflow.run3_native_stepping_proposals(minima,text)
        self.assertEqual(len(ranges),2);self.assertEqual(diagnosis['branch'],'measured_median_range')
        attempted_range=self.native_attempted(ranges)
        native,diagnosis=workflow.run3_native_stepping_proposals(minima+attempted_range,text)
        self.assertEqual(len(native),1);self.assertEqual(diagnosis['branch'],'median_native')
        attempts=minima+attempted_range+self.native_attempted(native)
        stepping,diagnosis=workflow.run3_native_stepping_proposals(attempts,text)
        self.assertEqual(len(stepping),2);self.assertEqual(diagnosis['branch'],'median_stepping')
        for spec in stepping:
            self.assertIn('--strictBounds',spec['command'])
            self.assertEqual(spec['run3_numerical_adapter'],'stepping_full5')
            self.assertNotIn('--snapshotName',spec['command'])
        self.assertFalse(workflow.run3_native_stepping_proposals(
            attempts+self.native_attempted(stepping),text)[0])

    def test_absent_numerical_evidence_is_na_not_folder_scope(self):
        minima=self.native_minima()
        specs,diagnosis=workflow.run3_native_stepping_proposals(minima,{a['log']:'' for a in minima})
        self.assertFalse(specs);self.assertIn('No measured exhausted-range',diagnosis['reason'])
        for attempt in minima:
            for crossing in attempt['validation']['crossings']:crossing['passed']=True
        specs,diagnosis=workflow.run3_native_stepping_proposals(minima,self.native_texts(minima))
        self.assertFalse(specs);self.assertIn('median range failure',diagnosis['reason'])

    def test_native_unfinished_or_successful_attempts_rejected(self):
        minima=self.native_minima();minima[0]['exit_code']=None
        with self.assertRaises(ValueError):workflow.run3_native_stepping_proposals(minima,{})
        minima=self.native_minima();minima[0]['passed']=True
        with self.assertRaises(ValueError):workflow.run3_native_stepping_proposals(minima,{})

    def test_native_full5_and_stepping_median_only_not_accepted(self):
        attempt=self.native_minima()[0];attempt['command']=list(attempt['command'])
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='new'
        attempt['run3_numerical_adapter']='native_profiled';evidence=deepcopy(attempt['validation'])
        for crossing in evidence['crossings']:crossing['passed']=True
        with patch.object(workflow,'verify_native_attempt',return_value=(True,evidence)):
            passed,checked=workflow.verify_run3_numerical_attempt({},attempt)
            self.assertTrue(passed);self.assertTrue(checked['full_five_quantile_passed'])
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='stepping'
        attempt['run3_numerical_adapter']='stepping_full5'
        with patch.object(workflow,'verify_stepping_attempt',return_value=(True,
                dict(quantiles={'.5':1},crossings=[dict(quantile=.5,passed=True)],median_only_verified=True))):
            self.assertFalse(workflow.verify_run3_numerical_attempt({},attempt)[0])

    def test_incomplete_native_log_does_not_stop_next_finite_method(self):
        minima=self.native_minima();text=self.native_texts(minima)
        ranges,_=workflow.run3_native_stepping_proposals(minima,text)
        native,_=workflow.run3_native_stepping_proposals(minima+self.native_attempted(ranges),text)
        attempt=self.native_attempted(native)[0]
        with patch.object(workflow,'verify_native_attempt',side_effect=ValueError('Expected five native bracket searches')), \
                patch.object(continuation,'verify_attempt',return_value=(False,dict(reason='expected_five_crossing_searches'))):
            passed,evidence=workflow.verify_run3_numerical_attempt({},attempt)
            self.assertFalse(passed);self.assertTrue(evidence['numerical_search_incomplete'])
        attempt['validation']=evidence
        self.assertEqual(len(workflow.run3_native_stepping_proposals(
            minima+self.native_attempted(ranges)+[attempt],text)[0]),2)

    def test_native_bad_hash_or_model_is_not_numerical_na(self):
        attempt=self.native_minima()[0]
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='new'
        attempt['run3_numerical_adapter']='native_profiled'
        with patch.object(workflow,'verify_native_attempt',side_effect=ValueError('Attempt ROOT hash mismatch')):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                workflow.verify_run3_numerical_attempt({},attempt)
        attempt['command']+=['--snapshotName','fit']
        with self.assertRaisesRegex(ValueError,'Changed likelihood'):
            workflow.verify_run3_numerical_attempt({},attempt)


if __name__=='__main__':unittest.main()
