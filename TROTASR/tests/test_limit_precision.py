"""Measured finite recovery and native five-quantile adapters, without fits."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from TROTASR.utils.paths import ROOT
from TROTASR.workflows import limit_recovery as workflow
from TROTASR.workflows.limit_validation import QUANTILES


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

    def test_minimum_recovery_preserves_likelihood_and_acceptance(self):
        base=workflow.fit_command('workspace.root','mStop650_mLSP1',.146211,.0001)
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
        primary=workflow.fit_command('workspace.root','mStop650_mLSP1',.146211,.0001)
        base=workflow.minimum_commands(primary)[1]['command']
        for tolerance in (.001,.0001):
            refined=workflow.refinement_command(base,tolerance)
            self.assertEqual(refined[refined.index('--rMax')+1],'0.146211')
            self.assertEqual(refined[refined.index('--cminFallbackAlgo')+1],'Minuit2,Migrad,2:'+str(tolerance))
            self.assertEqual(refined[refined.index('--rRelAcc')+1],'0.00025')
            for flag in ('--freezeParameters','--setParameterRanges','--noFitAsimov'):
                self.assertNotIn(flag,refined)
        with self.assertRaises(ValueError):workflow.refinement_command(base,.1)


class Run3FiniteContinuationTests(unittest.TestCase):
    """No native fits: saved-state, bounded-command and admission contracts."""
    def base(self):
        return workflow.minimum_commands(workflow.fit_command('workspace.root','mass',100,.01))[1]['command']

    def crossing(self):
        return dict(name='new_minimum',command=self.base(),exit_code=0,passed=False,log_sha256='log',
            validation=dict(quantiles={str(q):i+1. for i,q in enumerate(QUANTILES)},
                crossings=[dict(quantile=q,passed=i!=4) for i,q in enumerate(QUANTILES)]))

    def test_precision_four_distinct_and_exact_dedup(self):
        with patch.object(workflow,'adaptive_seed',return_value=(100,{'saved':'profile'})):
            values,evidence=workflow.precision_commands('workspace.root','mass',[],[])
            self.assertEqual(len(values),4)
            self.assertEqual(len({tuple(v['command']) for v in values}),4)
            self.assertEqual([v['explicit_stats'] for v in values],[False,False,True,True])
            for value in values:
                cmd=value['command'];self.assertEqual(cmd[cmd.index('--run')+1],'blind')
                self.assertEqual(cmd[cmd.index('--cminDefaultMinimizerPrecision')+1],'1e-08')
                self.assertNotIn('--snapshotName',cmd)
            before=deepcopy(values)
            left,_=workflow.precision_commands('workspace.root','mass',[],[values[0]['command']])
            self.assertEqual(len(left),3);self.assertEqual(values,before)

    def test_crossing_diagnosis_and_dedup(self):
        attempt=self.crossing();before=deepcopy(attempt)
        specs,diagnosis=workflow.crossing_commands([attempt]);self.assertEqual(len(specs),2)
        self.assertEqual(attempt,before);self.assertEqual(diagnosis['basis'],'new_minimum')
        again,diagnosis=workflow.crossing_commands([attempt],[s['command'] for s in specs])
        self.assertFalse(again);self.assertEqual(diagnosis['status'],'not_applicable')
        for mutate in (lambda a:a.update(passed=True),lambda a:a.update(exit_code=1),
                lambda a:a['validation']['quantiles'].update({'0.5':float('nan')}),
                lambda a:a['validation']['crossings'].clear(),
                lambda a:[c.update(passed=True) for c in a['validation']['crossings']]):
            candidate=deepcopy(attempt);mutate(candidate)
            self.assertFalse(workflow.crossing_commands([candidate])[0])

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
        before=deepcopy(values);specs,diagnosis=workflow.native_stepping_proposals(values,{})
        self.assertEqual(values,before);self.assertEqual(len(specs),1)
        self.assertEqual(diagnosis['branch'],'upper_only_native');self.assertIn('new',specs[0]['command'])
        self.assertFalse(workflow.native_stepping_proposals(values,{},[specs[0]['command']])[0])
        self.assertFalse(workflow.native_stepping_proposals(values+self.native_attempted(specs),{})[0])

    def test_measured_range_native_stepping_are_finite_and_real(self):
        minima=self.native_minima();text=self.native_texts(minima)
        ranges,diagnosis=workflow.native_stepping_proposals(minima,text)
        self.assertEqual(len(ranges),2);self.assertEqual(diagnosis['branch'],'measured_median_range')
        attempted_range=self.native_attempted(ranges)
        native,diagnosis=workflow.native_stepping_proposals(minima+attempted_range,text)
        self.assertEqual(len(native),1);self.assertEqual(diagnosis['branch'],'median_native')
        attempts=minima+attempted_range+self.native_attempted(native)
        stepping,diagnosis=workflow.native_stepping_proposals(attempts,text)
        self.assertEqual(len(stepping),2);self.assertEqual(diagnosis['branch'],'median_stepping')
        for spec in stepping:
            self.assertIn('--strictBounds',spec['command'])
            self.assertEqual(spec['run3_numerical_adapter'],'stepping_full5')
            self.assertNotIn('--snapshotName',spec['command'])
        self.assertFalse(workflow.native_stepping_proposals(
            attempts+self.native_attempted(stepping),text)[0])

    def test_absent_numerical_evidence_is_na_not_folder_scope(self):
        minima=self.native_minima()
        specs,diagnosis=workflow.native_stepping_proposals(minima,{a['log']:'' for a in minima})
        self.assertFalse(specs);self.assertIn('No measured exhausted-range',diagnosis['reason'])
        for attempt in minima:
            for crossing in attempt['validation']['crossings']:crossing['passed']=True
        specs,diagnosis=workflow.native_stepping_proposals(minima,self.native_texts(minima))
        self.assertFalse(specs);self.assertIn('median range failure',diagnosis['reason'])

    def test_native_unfinished_or_successful_attempts_rejected(self):
        minima=self.native_minima();minima[0]['exit_code']=None
        with self.assertRaises(ValueError):workflow.native_stepping_proposals(minima,{})
        minima=self.native_minima();minima[0]['passed']=True
        with self.assertRaises(ValueError):workflow.native_stepping_proposals(minima,{})

    def test_native_full5_and_stepping_median_only_not_accepted(self):
        attempt=self.native_minima()[0];attempt['command']=list(attempt['command'])
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='new'
        attempt['run3_numerical_adapter']='native_profiled';evidence=deepcopy(attempt['validation'])
        for crossing in evidence['crossings']:crossing['passed']=True
        with patch.object(workflow,'verify_native_attempt',return_value=(True,evidence)):
            passed,checked=workflow.validate_attempt({},attempt)
            self.assertTrue(passed);self.assertTrue(checked['full_five_quantile_passed'])
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='stepping'
        attempt['run3_numerical_adapter']='stepping_full5'
        with patch.object(workflow,'verify_stepping_attempt',return_value=(True,
                dict(quantiles={'.5':1},crossings=[dict(quantile=.5,passed=True)],median_only_verified=True))):
            self.assertFalse(workflow.validate_attempt({},attempt)[0])

    def test_incomplete_native_log_does_not_stop_next_finite_method(self):
        minima=self.native_minima();text=self.native_texts(minima)
        ranges,_=workflow.native_stepping_proposals(minima,text)
        native,_=workflow.native_stepping_proposals(minima+self.native_attempted(ranges),text)
        attempt=self.native_attempted(native)[0]
        with patch.object(workflow,'verify_native_attempt',side_effect=ValueError('Expected five native bracket searches')), \
                patch.object(workflow,'_verify_bisection_attempt',return_value=(False,dict(reason='expected_five_crossing_searches'))):
            passed,evidence=workflow.validate_attempt({},attempt)
            self.assertFalse(passed);self.assertTrue(evidence['numerical_search_incomplete'])
        attempt['validation']=evidence
        self.assertEqual(len(workflow.native_stepping_proposals(
            minima+self.native_attempted(ranges)+[attempt],text)[0]),2)

    def test_native_bad_hash_or_model_is_not_numerical_na(self):
        attempt=self.native_minima()[0]
        attempt['command'][attempt['command'].index('--minosAlgo')+1]='new'
        attempt['run3_numerical_adapter']='native_profiled'
        with patch.object(workflow,'verify_native_attempt',side_effect=ValueError('Attempt ROOT hash mismatch')):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                workflow.validate_attempt({},attempt)
        attempt['command']+=['--snapshotName','fit']
        with self.assertRaisesRegex(ValueError,'Changed likelihood'):
            workflow.validate_attempt({},attempt)


class MeasuredSpeedPolicyTests(unittest.TestCase):
    mass = 'mStop1000_mLSP100'
    workspace = ROOT/'stats/run3/limits/T2tt'/mass/'workspace.root'
    log = 'stats/run3/limits/T2tt/'+mass+'/base/combine.log'

    def receipts(self, text, **fields):
        command = workflow.fit_command(self.workspace, self.mass, 40., .1)
        attempt = dict(name='synthetic', log=self.log, command=command, exit_code=0, passed=False,
                       log_sha256='synthetic')
        attempt.update(fields)
        return [(self.log,text)], [command], [attempt]

    def profile(self, status=0, maximum=20.):
        return ('Fit to asimov dataset:\n r 0 0.01 +/- 0.08 <none>\n'
                'Minimization finished with status = 0\n'
                'Will search for NLL crossing by bisection\n'
                'Minimization finished with status = 0\nAt r = 1: delta(nll) = 0.1\n'
                f'Minimization finished with status = {status}\n'
                f'At r = 2: delta(nll) = {maximum}\n')

    def optimize(self, text, **fields):
        logs, commands, attempts = self.receipts(text, **fields)
        requested = workflow.fit_command(self.workspace, self.mass, 100., .01)
        optimized, evidence = workflow.optimize_saved_command(
            self.workspace, self.mass, requested, logs, commands, attempts=attempts)
        return requested, optimized, evidence

    def test_approved_range_only_changes_one_option_with_all_five_targets(self):
        requested, optimized, evidence = self.optimize(self.profile())
        index = requested.index('--rMax') + 1
        self.assertEqual(optimized[index], '2.5')
        self.assertEqual(optimized[:index] + optimized[index+1:], requested[:index] + requested[index+1:])
        self.assertEqual(len(evidence['measured_envelope']['brackets']), 5)
        self.assertEqual(evidence['warm_start']['status'], 'unavailable')

    def test_failed_unknown_partial_and_unbound_profiles_do_not_shrink_rmax(self):
        scans = (self.profile(-1), self.profile(1), self.profile(0, maximum=1.),
                 self.profile().replace('Minimization finished with status = 0\n',''))
        for text in scans:
            requested, optimized, _ = self.optimize(text)
            self.assertEqual(optimized, requested)
        requested, optimized, _ = self.optimize(self.profile(), exit_code=None)
        self.assertEqual(optimized, requested)
