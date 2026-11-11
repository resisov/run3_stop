"""Diagnostic contracts and numerical-policy tests; no synthetic physics fits."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows import diagnostic_common as common
from TROTASR.workflows import combine_impacts as impact
from TROTASR.workflows.combine_cronly import fit_command as cr_command
from TROTASR.workflows.impact_legacy import check_endpoints, fit_command
from TROTASR.workflows.impact_recovery_legacy import Recovery, missing_endpoint_side, missing_endpoint_rows, TARGET, NLL_TOL
from TROTASR.workflows.recover_impacts import recovery_scope


def cr_text():
    channels = [f'{r}_{m}_c1_{y}' for r in ('LLCR', 'QCDCR', 'GCR')
                for m in ('highdm', 'lowdm') for y in ('2024', '2025')]
    return '\n'.join(['bin ' + ' '.join(channels), 'process background background',
                      'process 1 1', '* autoMCStats 10 1 1'])


def revised_record():
    from TROTASR.workflows.build_nominal_grid import CR12_POLICY, background_shape_names
    from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
    shape = dict(id='jme_background_shapes_20260929',mode='shape',auto_mc_stats=[10,1,1],user_approved=True)
    record = dict(scope='full_combined_systematic_production', bins=428,
        statistical_revision=copy.deepcopy(CR12_POLICY), rate_parameters=96, signal_contamination=False,
        auto_mc_stats=[10,1,1], sr_data_blinded=True, background_estimation_uncertainties=shape,
        systematic_combination=copy.deepcopy(COMBINATION_POLICY), preliminary=True,
        central_conventions_consistent=False, systematic_scope='Explicit as-is preliminary test contract',
        channels={f'{r}_{m}_c1_{y}':['bin']*((118 if m=='highdm' else 30) if r=='SR' else (12 if m=='highdm' else 10))
            for r in ('SR','LLCR','QCDCR','GCR') for m in ('highdm','lowdm') for y in ('2024','2025')},
        files={'template_'+m+'_'+y+'.root':'digest' for m in ('highdm','lowdm') for y in ('2024','2025')},
        fit_compatibility=dict(background_error_shapes=dict(nuisances=34,nuisance_names=sorted(background_shape_names()),
            measured_factors_unchanged=True,existing_shape_endpoints_unchanged=True)))
    record['contract'] = {k:record[k] for k in ('statistical_revision','background_estimation_uncertainties','systematic_combination')}
    return record


class ProfileReuseTests(unittest.TestCase):
    def test_success_is_reused_across_phases_without_copied_record(self):
        from TROTASR.workflows.combine_cronly import prior_completed_profile
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            output=Path(directory)
            for phase,record in [('profile_pulls',dict(status='needs_attention')),
                                 ('profile_pulls_warm_after_nll',dict(status='complete',tag='verified'))]:
                (output/phase/'nuisance').mkdir(parents=True)
                write_json(output/phase/'nuisance/state.json',record)
            self.assertEqual(prior_completed_profile(output,output/'profile_pulls_gaussian_scan','nuisance'),
                             dict(status='complete',tag='verified'))
            self.assertIsNone(prior_completed_profile(output,output/'profile_pulls_warm','nuisance'))
            self.assertIsNone(prior_completed_profile(output,output/'profile_pulls_direct','other'))
            (output/'profile_pulls_direct/nuisance').mkdir(parents=True)
            write_json(output/'profile_pulls_direct/nuisance/state.json',
                       dict(status='complete',method='direct_CR_profile_likelihood'))
            self.assertEqual(prior_completed_profile(output,output/'profile_pulls_direct','nuisance'),
                             dict(status='complete',tag='verified'))


class RevisedDiagnosticPolicyTests(unittest.TestCase):
    def test_exact_opt_in_policy_and_preliminary_scope(self):
        self.assertFalse(common.require_revised_record(dict(scope='full_nominal_production')))
        record = revised_record()
        self.assertTrue(common.require_revised_record(record))
        for key,value in [('bins',452),('rate_parameters',112),('signal_contamination',True),
                          ('auto_mc_stats',[10,0,1]),('sr_data_blinded',False),
                          ('statistical_revision',None),('systematic_scope',''),
                          ('central_conventions_consistent',True),('systematic_combination',{})]:
            bad = dict(record,**{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError): common.require_revised_record(bad)

    def test_actual_card_and_template_contract(self):
        record = revised_record()
        card = ROOT/'results/not_created/datacard.txt'
        meta = dict(record,fit_ready=True,inputs=[dict(path='results/not_created/templates',sha256='digest')])
        text = '\n'.join(['bin SR_highdm_c1_2024','bin SR_highdm_c1_2024 SR_highdm_c1_2024',
            'process Top signal','process 1 0','* autoMCStats 10 1 1']+
            [f'par{i} rateParam SR_highdm_c1_2024 Top 1 [0.01,5]' for i in range(96)])
        def run(tm=record,metadata=meta,actual=text):
            with patch.object(common,'read_json',side_effect=lambda p:metadata if p.name=='datacard.manifest.json' else tm), \
                 patch.object(Path,'read_text',return_value=actual):
                common.verify_revised_card(card,record,lambda p:'digest')
        run()
        for bad in (text.replace('[0.01,5]','[0,10]'),text.replace('par95 rateParam','par94 rateParam'),
                    text.replace('bin SR_highdm_c1_2024 SR_highdm_c1_2024','bin SR_highdm_c1_2024 GCR_highdm_c1_2024'),
                    text+'\nCMS_NPS26012_RZstat_highdm_Nb1_2024 lnN 1.1 -'):
            with self.assertRaises(ValueError): run(actual=bad)
        for change in ('binning','file','contract','audit','metadata'):
            tm = copy.deepcopy(record); metadata = copy.deepcopy(meta)
            if change=='binning': tm['channels']['GCR_highdm_c1_2024'].append('bad')
            if change=='file': tm['files']['extra.root']='digest'
            if change=='contract': tm['contract']['systematic_combination']={}
            if change=='audit': tm['fit_compatibility']['background_error_shapes']['nuisances']=33
            if change=='metadata': metadata['rate_parameters']=112
            with self.subTest(change=change),self.assertRaises(ValueError): run(tm,metadata)

    def test_cr_only_revised_support_gate_still_rejects_cr_failures(self):
        record = revised_record()
        record['status']='templates_verified_fit_blocked'
        record['fit_compatibility'].update(status='blocked',cr_only_ready=True,auto_mc_stats=[10,1,1],
            interpolation_changed=True,endpoints_modified=False,
            background_issues=[dict(channel='SR_highdm_c1_2024',problem='zero_integral_variation')])
        self.assertTrue(common.cr_only_support_ready(record))
        record['fit_compatibility']['background_issues'][0]['channel']='QCDCR_highdm_c1_2024'
        self.assertFalse(common.cr_only_support_ready(record))


class ImpactBenchmarkTests(unittest.TestCase):
    def test_default_labels_unchanged(self):
        self.assertEqual(impact.benchmark_labels('T2tt', 'mStop1200_mLSP500'),
            ('T2tt_mStop1200_mLSP500', 'mStop1200_mLSP500', 'impacts_mStop1200_mLSP500'))

    def test_requested_benchmarks_have_unique_labels(self):
        stems = set()
        for model in ('T2bW', 'T2tb'):
            for mass in ('mStop1200_mLSP500', 'mStop1000_mLSP800'):
                identity, label, stem = impact.benchmark_labels(model, mass)
                self.assertEqual(identity, model+'_'+mass)
                self.assertEqual(label, identity)
                self.assertEqual(stem, 'impacts_'+identity)
                stems.add(stem)
        self.assertEqual(len(stems), 4)

    def test_invalid_labels_rejected(self):
        for model, mass in (('other', 'mStop1200_mLSP500'),
                            ('T2tb', '../mStop1200_mLSP500'), ('T2bW', '1200,500')):
            with self.assertRaises(ValueError):
                impact.benchmark_labels(model, mass)

    def test_reuse_requires_exact_point_card_and_workspace_hash(self):
        card = ROOT/'stats/not_created/card.txt'
        state_path = ROOT/'stats/not_created/state.json'
        state = dict(model='T2bW', mass='mStop1000_mLSP800', workspace_valid=True,
                     card=str(card.relative_to(ROOT)), card_sha256='digest', workspace_sha256='a'*64)
        with patch.object(impact, 'read_json', return_value=state), \
             patch.object(impact, 'sha256', return_value='digest'), \
             patch.object(Path,'is_file',return_value=True), \
             patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=1)):
            target, provenance = impact.saved_workspace(state_path, card, 'T2bW', 'mStop1000_mLSP800')
            self.assertEqual(target, state_path.parent/'workspace.root')
            self.assertEqual(provenance['workspace_sha256'], 'a'*64)
            for key, value in (('model','T2tb'), ('mass','mStop1200_mLSP500'),
                               ('workspace_valid',False), ('workspace_sha256','changed'),
                               ('card_sha256','changed'), ('card','stats/not_created/other.txt')):
                with self.subTest(key=key), patch.object(impact, 'read_json', return_value=dict(state,**{key:value})):
                    with self.assertRaises(ValueError):
                        impact.saved_workspace(state_path, card, 'T2bW', 'mStop1000_mLSP800')

    def test_exact_card_selected_without_default_fallback(self):
        requested = ('mStop1200_mLSP500', 'mStop1000_mLSP800')
        points = []
        for model, count in (('T2tt', 205), ('T2tb', 365), ('T2bW', 354)):
            masses = [*requested, *[f'mStop{i}_mLSP0' for i in range(count-2)]]
            points.extend(dict(model=model, mass=mass) for mass in masses)
        manifest = dict(status='cards_ready', scope='full_nominal_production',
            bins=540, th1_channels=16, root_integrity_checked=True, points=points,
            expected_points_by_model={'T2tt':205, 'T2tb':365, 'T2bW':354})
        card = ROOT/'stats/not_created/card.txt'
        with patch.object(common, 'read_json', return_value=manifest), \
             patch.object(common, 'IntegrityCache', return_value=lambda p:'digest'), \
             patch.object(common, 'verify_card', return_value=card) as verify, \
             patch.object(Path, 'read_text', return_value=''):
            for model in ('T2bW', 'T2tb'):
                for mass in requested:
                    actual, origins = common.grid_card(ROOT/'stats/not_created/manifest.json', 'impact', model, mass)
                    self.assertEqual(actual, card)
                    self.assertEqual(verify.call_args.args[0], dict(model=model, mass=mass))
                    self.assertEqual(origins['card_sha256'], 'digest')
            with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                common.grid_card(ROOT/'stats/not_created/manifest.json', 'impact', 'T2tb', 'mStop9999_mLSP1')


class DiagnosticPolicyTests(unittest.TestCase):
    def test_cr_independent_seeds_include_central_and_distinct_saved_points(self):
        from TROTASR.workflows.combine_cronly import profile_start_points
        samples=[dict(theta=-4.28,deltaNLL=.55,root='near'),
                 dict(theta=-4.29,deltaNLL=.501,root='crossing')]
        seeds=profile_start_points(samples,-4.28)
        self.assertEqual(seeds,[None,samples[0],samples[1]])
        self.assertEqual(profile_start_points([],0),[None])

    def test_cr_profiles_use_observed_cr_only_at_fixed_zero_signal(self):
        from TROTASR.workflows.combine_cronly import cr_profile_command
        command,tag=cr_profile_command('/same/cr_workspace.root','CMS_x',1)
        self.assertNotIn('-t',command)
        self.assertNotIn('--expectSignal',command)
        self.assertEqual(command[command.index('--setParameters')+1],'r=0')
        self.assertEqual(command[command.index('--freezeParameters')+1],'r')
        self.assertEqual(command[command.index('-P')+1],'CMS_x')
        self.assertEqual(command[command.index('--floatOtherPOIs')+1],'1')

    def test_cr_covariance_recovery_keeps_model_and_warm_start(self):
        from TROTASR.workflows.combine_cronly import covariance_commands
        commands=list(covariance_commands('/same/workspace.root',{'r':0.,'x':.928}))
        self.assertEqual(len(commands),2)
        for name,command in commands:
            self.assertEqual(command[command.index('--freezeParameters')+1],'r')
            self.assertIn('r=0,x=0.928',command[command.index('--setParameters')+1])
            self.assertIn('/same/workspace.root',command)
            self.assertNotIn('--setParameterRanges',command)
        self.assertNotIn('--robustHesse',commands[0][1])
        self.assertIn('--robustHesse',commands[1][1])


    def test_exact_cr_channels(self):
        self.assertEqual(len(common.cr_channels(cr_text())), 12)
        with self.assertRaises(ValueError):
            common.cr_channels(cr_text().replace('LLCR_highdm_c1_2024', 'SR_highdm_c1_2024'))


    def test_signal_and_mcstat_changes_rejected(self):
        for text in (cr_text().replace('process 1 1', 'process 0 1'),
                     cr_text().replace('autoMCStats 10 1 1', 'autoMCStats 0 1 1')):
            with self.assertRaises(ValueError):
                common.cr_channels(text)

    def test_cr_only_prescription(self):
        cmd = cr_command(Path('workspace.root'))
        for token in ('--skipSBFit', '--robustFit', '--freezeParameters'):
            self.assertIn(token, cmd)
        self.assertEqual(cmd[cmd.index('--setParameters')+1], 'r=0')
        self.assertEqual(cmd[cmd.index('--cminDefaultMinimizerStrategy')+1], '0')
        self.assertNotIn('-t', cmd)  # measured CR data, not SR or fake data

    def test_impact_is_signal_plus_background_asimov(self):
        for name in ('r', 'CMS_test'):
            for strategy in (0, 1):
                cmd, tag = fit_command(Path('workspace.root'), name, 1, strategy)
                self.assertEqual(cmd[cmd.index('-t')+1], '-1')
                self.assertEqual(cmd[cmd.index('--expectSignal')+1], '1')
                self.assertEqual(cmd[cmd.index('--setParameterRanges')+1], 'r=0,20')
                self.assertEqual(cmd[cmd.index('--cminDefaultMinimizerStrategy')+1], str(strategy))
                self.assertEqual(cmd[cmd.index('--algo')+1], 'singles' if name == 'r' else 'impact')

    def test_endpoint_validation(self):
        check_endpoints([[1., 1., 0.], [.8, .5, .5], [1.2, 1.5, .5]], [0, 10])
        bad = [
            [[1., 1., 0.], [1., 1., .5], [1., 1., .5]],
            [[1., 1., 0.], [1., 2., .5], [1., 3., .5]],
            [[1., 1., 0.], [1., float('nan'), .5], [1., 2., .5]],
            [[1., 1., 0.], [1., 1., .5], [1., 2., .5]]]
        for values in bad:
            with self.assertRaises(ValueError):
                check_endpoints(values, [0, 10])

    def test_boundary_endpoint_retained_by_original_policy(self):
        check_endpoints([[1., 0., 0.], [1., 0., 0.], [1.2, 2., .5]], [0, 10])

    def test_partial_grid_is_not_diagnostic_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            path = Path(folder)/'manifest.json'
            write_json(path, dict(status='cards_ready', scope='representative_validation'))
            with self.assertRaisesRegex(ValueError, 'full-input'):
                common.grid_card(path, 'impact')

    def test_preserved_sr_issue_cannot_waive_cr_support_or_model_changes(self):
        record = dict(status='templates_verified_fit_blocked', fit_compatibility=dict(
            status='blocked', cr_only_ready=True, auto_mc_stats=[10,1,1],
            interpolation_changed=False, endpoints_modified=False,
            background_issues=[dict(channel='SR_highdm_c1_2024', problem='zero_integral_nonempty_component')]))
        self.assertTrue(common.cr_only_support_ready(record))
        for key, value in [('cr_only_ready', False), ('auto_mc_stats', [10,1,2]),
                           ('interpolation_changed', True), ('endpoints_modified', True),
                           ('background_issues', [dict(channel='QCDCR_highdm_c1_2024', problem='zero_integral_nonempty_component')]),
                           ('background_issues', [dict(channel='SR_highdm_c1_2024', problem='sr_only_parameter')])]:
            bad = copy.deepcopy(record); bad['fit_compatibility'][key] = value
            self.assertFalse(common.cr_only_support_ready(bad))

    def test_sr_fit_remains_blocked_for_verified_unfit_templates(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            path = Path(folder)/'manifest.json'
            write_json(path, dict(status='templates_verified_fit_blocked', scope='full_cms_trota_jme_production',
                bins=540, th1_channels=16, root_integrity_checked=True,
                fit_compatibility=dict(status='blocked', cr_only_ready=True, auto_mc_stats=[10,1,1],
                    interpolation_changed=False, endpoints_modified=False, background_issues=[])))
            with self.assertRaisesRegex(ValueError, 'full-input'):
                common.grid_card(path, 'impact')

    def test_collection_matches_exact_saved_endpoints(self):
        initial = dict(entries=[[1.,1.,0.],[.8,.8,.5],[1.2,1.2,.5]])
        result = dict(entries=[[1.,0.,0.],[.9,-1.,.5],[1.1,1.,.5]])
        data = dict(POIs=[dict(name='r', fit=[.8,1.,1.2])],
            params=[dict(name='a',fit=[-1.,0.,1.],r=[.9,1.,1.1],impact_r=.1)])
        impact.validate_collection(data, ['a'], {'a':result}, initial)
        wrong = copy.deepcopy(data); wrong['params'][0]['r'][0] = .3
        with self.assertRaises(ValueError):
            impact.validate_collection(wrong, ['a'], {'a':result}, initial)
        wrong = copy.deepcopy(data); wrong['params'].append(wrong['params'][0])
        with self.assertRaises(ValueError):
            impact.validate_collection(wrong, ['a'], {'a':result}, initial)

    def test_resource_count_ignores_zombies(self):
        with patch.object(common.subprocess,'check_output',return_value='S python\nZ python\nR combine\nS ssh\n'), \
             patch.object(Path,'read_text',return_value='MemAvailable: 100000000 kB\n'):
            self.assertEqual(common.live_resources(), (2,102400000000))

    def test_native_reexec_resolves_paths_before_cwd_change(self):
        args=['--manifest','TROTASR/stats/grid.json','--output=TROTASR/stats/test','--workers','49']
        with patch('os.getcwd',return_value=str(ROOT.parent)):
            result=common.native_arguments(args)
        self.assertEqual(result,['--manifest',str(ROOT/'stats/grid.json'),
                                 '--output='+str(ROOT/'stats/test'),'--workers','49'])
        self.assertEqual(args[1],'TROTASR/stats/grid.json')
        with self.assertRaises(ValueError):
            common.native_arguments(['--work','/home/forbidden'])

    def test_recovery_scope_is_exact_and_precision_pass_finite(self):
        status = dict(status='needs_attention', failed_nuisances=['qcd_norm_a'], nuisance_count=2)
        records = {'qcd_norm_a':dict(status='failed'), 'a':dict(status='complete')}
        self.assertEqual(recovery_scope(status,records,[],[],{}), ['qcd_norm_a'])
        with self.assertRaises(ValueError):
            recovery_scope(status, records, ['a'], [], {})
        with self.assertRaises(ValueError):
            recovery_scope(status, records, [], ['qcd_norm_a'], {})
        previous = dict(status='needs_attention', failed={'qcd_norm_a':'error'})
        recovery_scope(status,records,[],['qcd_norm_a'],previous)
        previous['refined_parameters']=['qcd_norm_a']
        with self.assertRaises(ValueError):
            recovery_scope(status,records,[],['qcd_norm_a'],previous)

    def test_missing_endpoint_scope(self):
        center=dict(theta=0.,r=1.,deltaNLL=0.,quantile=-1.)
        upper=dict(theta=1.,r=1.1,deltaNLL=.5,quantile=.16)
        self.assertEqual(missing_endpoint_side([center,upper],0.,[-5,5]), -1)
        with self.assertRaises(ValueError):
            missing_endpoint_side([center,upper],.1,[-5,5])

    def test_original_profile_crossing(self):
        recovery = Recovery.__new__(Recovery)
        def submit(name, value):
            return types.SimpleNamespace(result=lambda: dict(theta=value,deltaNLL=value))
        recovery.submit = submit
        point, trace = recovery.crossing('x',dict(theta=0.,deltaNLL=0.),dict(theta=1.,deltaNLL=1.))
        self.assertAlmostEqual(point['deltaNLL'], TARGET)
        self.assertEqual(NLL_TOL, .002)
        self.assertLessEqual(len(trace),20)
        with self.assertRaises(ValueError):
            recovery.crossing('x',dict(theta=0.,deltaNLL=.7),dict(theta=1.,deltaNLL=1.))

    def test_zero_width_failed_endpoint_adapter_preserves_input(self):
        center=dict(theta=1.,r=1.,deltaNLL=0.,quantile=-1.)
        lower=dict(theta=.99,r=.999,deltaNLL=0.,quantile=.16)
        upper=dict(theta=1.01,r=1.001,deltaNLL=0.,quantile=.84)
        for rows, endpoint, side in (([center,lower,center],lower,1),
                                     ([center,center,upper],upper,-1)):
            before=copy.deepcopy(rows)
            selected=missing_endpoint_rows(rows,1.,[0,10])
            self.assertEqual(selected,[center,endpoint])
            self.assertEqual(missing_endpoint_side(selected,1.,[0,10]),side)
            self.assertEqual(rows,before)
        self.assertEqual(missing_endpoint_rows([center,upper],1.,[0,10]),[center,upper])

    def test_zero_width_adapter_rejects_ambiguous_or_valid_fits(self):
        def row(theta):return dict(theta=theta,r=1.,deltaNLL=0.,quantile=0.)
        for rows, center, bounds in (([row(1),row(1),row(1)],1,[0,10]),
                ([row(1),row(.9),row(1.1)],1,[0,10]),
                ([row(1),row(1),row(.9)],1,[0,10]),
                ([row(1),row(1.1),row(1)],1,[0,10]),
                ([row(1),row(1),row(11)],1,[0,10]),
                ([row(0),row(0),row(1)],0,[0,10]),
                ([row(1),row(1),row(float('nan'))],1,[0,10]),
                ([row(1),row(1),row(2)],1.1,[0,10])):
            with self.assertRaises(ValueError):
                missing_endpoint_rows(rows,center,bounds)

    def test_explicit_failed_ll_scope_not_automatically_broadened(self):
        name='CMS_NPS26012_ll_norm_highdm_Nb1_bin1_2024'
        status=dict(status='needs_attention',failed_nuisances=[name],nuisance_count=1)
        records={name:dict(status='failed')}
        with self.assertRaises(ValueError):recovery_scope(status,records,[],[],{})
        self.assertEqual(recovery_scope(status,records,[name],[],{}),[name])

    def test_zero_width_entry_uses_existing_finite_profile_algorithm(self):
        from TROTASR.workflows import impact_recovery_legacy as profile
        name='CMS_NPS26012_ll_norm_highdm_Nb1_bin1_2024'
        center=dict(theta=1.,r=1.,deltaNLL=0.,quantile=-1.)
        lower=dict(theta=.99,r=.999,deltaNLL=0.,quantile=.16)
        rows=[center,lower,center]
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            worker=Recovery.__new__(Recovery)
            worker.work=worker.out=Path(folder)
            worker.refined=set()
            worker.inputs=dict(bestfit={name:1.,'r':1.},parameter_bounds={name:[0,10]})
            values=[]
            def submit(n,value):
                values.append(value)
                return types.SimpleNamespace(result=lambda:dict(theta=value,r=1.+(value-1.)*.1,
                    deltaNLL=.5*((value-1.)/.01)**2))
            worker.submit=submit
            with patch.object(profile,'tree_rows',return_value=rows):
                result=worker.recover(name)
            self.assertEqual(result['original_entries'],rows)
            self.assertEqual(result['lower'],lower)
            self.assertAlmostEqual(result['upper']['theta'],1.01)
            self.assertLessEqual(len(values),2)
            self.assertFalse(result['boundary_limited_upper'])


class CommandIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.work = Path(self.tmp.name)
        self.runner = common.CommandRunner(self.work)
        self.product = self.work/'product.json'

    def tearDown(self):
        self.tmp.cleanup()

    def successful(self, command, **kwargs):
        write_json(self.product, {'unit_test':True})
        return types.SimpleNamespace(pid=123,wait=lambda:0)

    def test_completed_command_is_reused(self):
        with patch.object(common, 'admitted_popen', side_effect=self.successful) as popen:
            a=self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])
            b=self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])
        self.assertEqual(a,b); self.assertEqual(popen.call_count,1)

    def test_failed_command_is_not_retried_on_resume(self):
        proc=types.SimpleNamespace(pid=123,wait=lambda:1)
        with patch.object(common,'admitted_popen',return_value=proc) as popen:
            for _ in range(2):
                self.assertEqual(self.runner.run(['test-fail'],self.work,self.work/'log.txt',[self.product])['status'],'failed')
        self.assertEqual(popen.call_count,1)

    def test_zero_exit_missing_product_is_failure(self):
        with patch.object(common,'admitted_popen',return_value=types.SimpleNamespace(pid=123,wait=lambda:0)):
            self.assertEqual(self.runner.run(['test-missing'],self.work,self.work/'log.txt',[self.product])['status'],'failed')

    def test_changed_success_product_is_blocked(self):
        with patch.object(common,'admitted_popen',side_effect=self.successful):
            self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])
        write_json(self.product, {'changed':True})
        with self.assertRaisesRegex(ValueError,'changed'):
            self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])

    def test_interrupted_command_is_not_duplicated(self):
        key=common.command_key(['test-command'],self.work)
        write_json(self.work/'commands'/(key+'.json'),dict(status='running',pid=123))
        with patch.object(common,'process_identity',return_value=None), patch.object(common,'admitted_popen') as popen:
            result=self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])
            self.assertEqual(result['status'],'failed')
            self.assertIsNone(result['exit_code'])
            popen.assert_not_called()

    def test_live_child_is_not_restarted(self):
        key=common.command_key(['test-command'],self.work)
        write_json(self.work/'commands'/(key+'.json'),dict(status='running',pid=123,pid_start_ticks=5))
        with patch.object(common,'process_identity',return_value=5), self.assertRaises(BlockingIOError):
            self.runner.run(['test-command'],self.work,self.work/'log.txt',[self.product])

    def test_code_history_and_worker_count_do_not_block_resume(self):
        old=dict(origins={'input':'x'}, code={'deleted_old_source.py':'old'}, options={'workers':1,'strategy':0})
        write_json(self.work/'contract.json',old)
        result=common.freeze_contract(self.work,{'input':'x'},['workflows/current.py'],{'workers':40,'strategy':0})
        self.assertEqual(result,old)
        self.assertEqual(read_json(self.work/'contract.json'),old)

    def test_cr_main_advances_finite_recovery(self):
        from TROTASR.workflows import combine_cronly as cr
        write_json(self.work/'state.json',dict(stage='fit',status='needs_attention'))
        (self.work/'workspace.root').touch()
        with patch.object(cr,'_run_once',side_effect=common.CommandFailure('failed')), \
             patch.object(cr,'recover',side_effect=[2,2,0]) as recover:
            self.assertEqual(cr.run('manifest',self.work),0)
        self.assertEqual(recover.call_count,3)
        self.assertEqual(recover.call_args_list[-1].kwargs,{'robust':True})

    def test_accepted_cr_result_does_not_launch_recovery(self):
        from TROTASR.workflows import combine_cronly as cr
        with patch.object(cr,'_run_once',return_value=0), patch.object(cr,'recover') as recover:
            self.assertEqual(cr.run('manifest',self.work),0)
            recover.assert_not_called()

    def test_impact_main_exhausts_finite_profiles(self):
        from TROTASR.workflows import recover_impacts as recovery
        write_json(self.work/'impact_status.json',dict(status='needs_attention'))
        with patch.object(impact,'_run',return_value=2), \
             patch.object(recovery,'run_bounded',return_value=2) as fits:
            self.assertEqual(impact.run_with_recovery('manifest',self.work,4),2)
        self.assertEqual(fits.call_count,6)
        self.assertEqual(len(read_json(self.work/'impact_status.json')['recovery_attempts']),7)

    def test_policy_labels_do_not_replace_physics_checks(self):
        from TROTASR.workflows import build_nominal_grid as cards
        policy=dict(cards.CR12_POLICY, id='renamed', user_approved=False)
        self.assertTrue(cards.require_statistical_revision(policy))
        with self.assertRaises(ValueError):
            cards.require_statistical_revision(dict(policy,auto_mc_stats=[0,1,1]))

    def test_cleanup_preserves_input_identity(self):
        from TROTASR.workflows.build_nominal_grid import same_extraction_contract
        old=dict(inputs={'2025':'input'}, code={'removed.py':'old'})
        self.assertTrue(same_extraction_contract(old,dict(old,code={'current.py':'new'})))
        self.assertFalse(same_extraction_contract(old,dict(old,inputs={'2025':'different'})))

    def test_covqual_two_is_accepted_without_changing_reported_quality(self):
        from TROTASR.workflows.combine_cronly import validate_fit
        fit=types.SimpleNamespace(status=lambda:0,covQual=lambda:2,edm=lambda:0.001)
        source=types.SimpleNamespace(IsZombie=lambda:False,Get=lambda name:fit,Close=lambda:None)
        root=types.SimpleNamespace(TFile=types.SimpleNamespace(Open=lambda path:source))
        with patch.dict('sys.modules',{'ROOT':root}):
            self.assertEqual(validate_fit(self.work/'fit.root')['covariance_quality'],2)

    def test_external_path_blocked(self):
        with self.assertRaisesRegex(ValueError,'escapes TROTASR'):
            self.runner.run(['test'],ROOT.parent,self.work/'log.txt')

    def test_divergent_contract_preserved(self):
        common.freeze_contract(self.work, {'input':'x'}, [], {'strategy':0})
        with self.assertRaisesRegex(ValueError,'Divergent'):
            common.freeze_contract(self.work, {'input':'x'}, [], {'strategy':1})

    def test_impact_attempts_are_finite_across_resumes(self):
        runner=types.SimpleNamespace(run=lambda *a,**k:dict(status='failed',exit_code=1,log='unit_test',products={}))
        with patch.object(impact,'validate_fit',return_value=dict(name='a',valid=False,error='unit_test')):
            impact.fit_one(self.work,self.work/'workspace.root','a',[-5,5],runner)
            with patch.object(runner,'run',side_effect=AssertionError('retry reset')):
                impact.fit_one(self.work,self.work/'workspace.root','a',[-5,5],runner)
        self.assertEqual(len(read_json(self.work/'fit_state/a.json')['attempts']),2)


class ConditionalBracketTests(unittest.TestCase):
    def test_midpoint_failure_uses_interior_quarter(self):
        from TROTASR.workflows.combine_cronly import conditional_bracket_trial
        seen=[]
        def fixed(value):
            seen.append(value)
            if len(seen)==1:raise ValueError('CR conditional minimization failed at '+str(value))
            return dict(theta=value,deltaNLL=.501)
        result=conditional_bracket_trial(fixed,dict(theta=-4.),dict(theta=-5.),True)
        self.assertEqual(seen,[-4.5,-4.25])
        self.assertEqual(result,dict(theta=-4.25,deltaNLL=.501))

    def test_failures_bounded_and_integrity_errors_not_swallowed(self):
        from TROTASR.workflows.combine_cronly import conditional_bracket_trial
        from unittest.mock import Mock
        fixed=Mock(side_effect=ValueError('CR conditional minimization failed at x'))
        with self.assertRaisesRegex(ValueError,'Three conditional'):
            conditional_bracket_trial(fixed,dict(theta=1.),dict(theta=2.),True)
        self.assertEqual(fixed.call_count,3)
        fixed=Mock(side_effect=ValueError('Saved conditional ROOT changed'))
        with self.assertRaisesRegex(ValueError,'ROOT changed'):
            conditional_bracket_trial(fixed,dict(theta=1.),dict(theta=2.),True)
        self.assertEqual(fixed.call_count,1)


if __name__=='__main__':
    unittest.main()
