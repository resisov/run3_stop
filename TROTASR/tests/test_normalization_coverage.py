import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.utils.paths import ROOT
from TROTASR.workflows.audit_normalization_coverage import missing_tasks, extend
from TROTASR.workflows.normalization_metadata_legacy import build_signal_mass_points, load_signal_xsec
from TROTASR.workflows.normalization_recovery import verify_extension,retry_keys,payload_contract
from TROTASR.workflows import normalization_recovery as compatibility
from TROTASR.workflows import promote_normalization_recovery as promotion


class CommonCrossSectionTest(unittest.TestCase):
    primary = 'TTto4Q_TuneCP5_13p6TeV_powheg-pythia8'

    def reference(self):
        return dict(datasets={'1': dict(dataset=self.primary+
            '-RunIII2024Summer24NanoAODv15-v2____1_', xsec_pb=419.82,
            is_data=False, is_signal=False, sumw=123., luminosity_pb=999.)})

    def test_production_labels_do_not_select_cross_section(self):
        from TROTASR.workflows.install_assets import primary_dataset, background_xsec_index, apply_background_xsecs
        from TROTASR.workflows.normalization_metadata_legacy import build_physical
        ref = self.reference()
        index = background_xsec_index([('existing', ref)])
        self.assertEqual(primary_dataset(ref['datasets']['1']['dataset']), self.primary)
        for era, lumi, denominator in (('2022',7980.4,100.), ('2022EE',26671.7,200.),
                                       ('2023',17794.,300.), ('2023BPix',9451.,400.)):
            records = {'42': dict(dataset=f'/{self.primary}/Run3Summer{era}NanoAODv12-v9/NANOAODSIM',
                                   xsec_pb=None, is_data=False, is_signal=False,
                                   sumw=denominator, files_processed=8)}
            apply_background_xsecs(records, index)
            physical, _, factors = build_physical(records, lumi)
            self.assertEqual(records['42']['xsec_pb'], 419.82)
            self.assertEqual(records['42']['sumw'], denominator)
            self.assertEqual(factors['42']['normalization_factor'], 419.82*lumi/denominator)
            self.assertEqual(factors['42']['physical_dataset_sumw'], denominator)
        self.assertEqual(ref, self.reference())

    def test_different_process_energy_and_filter_are_not_aliased(self):
        from TROTASR.workflows.install_assets import background_xsec_index, apply_background_xsecs
        index = background_xsec_index([('existing', self.reference())])
        for name in (self.primary.replace('4Q','2L2Nu'), self.primary.replace('13p6','13'),
                     self.primary.replace('_Tune','_HT-600_Tune')):
            records = {'1':dict(dataset=f'/{name}/any/NANOAODSIM', xsec_pb=None)}
            self.assertEqual(len(apply_background_xsecs(records, index)), 1)
            self.assertIsNone(records['1']['xsec_pb'])

    def test_conflicts_are_not_hidden_by_era_or_source_order(self):
        from TROTASR.workflows.install_assets import background_xsec_index
        a, b = self.reference(), self.reference()
        b['datasets']['1']['xsec_pb'] = 400.
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            background_xsec_index([('old', a), ('new', b)])

    def test_existing_values_and_data_signal_are_preserved(self):
        from TROTASR.workflows.install_assets import background_xsec_index, apply_background_xsecs
        index = background_xsec_index([('existing', self.reference())])
        records = {'1':dict(dataset=self.primary, xsec_pb=419.82),
                   '2':dict(dataset='data', is_data=True, xsec_pb=None),
                   '3':dict(dataset='SMS', is_signal=True, xsec_pb=None)}
        before=copy.deepcopy(records)
        self.assertEqual(apply_background_xsecs(records, index), [])
        self.assertEqual(records['1']['xsec_pb'],before['1']['xsec_pb'])
        self.assertEqual(records['2'],before['2'])
        self.assertEqual(records['3'],before['3'])

    def test_current_analysis_references_agree(self):
        from TROTASR.workflows.install_assets import adopted_background_xsecs
        index=adopted_background_xsecs()
        self.assertEqual(index[self.primary]['xsec_pb'],419.82)
        self.assertEqual(len(index[self.primary]['sources']),2)

    def test_wz_name_permutation_keeps_every_generated_bin(self):
        from TROTASR.workflows.install_assets import adopted_background_xsecs, apply_background_xsecs
        index = adopted_background_xsecs()
        for process, variable in (('WtoLNu', 'PTLNu'), ('Zto2Nu', 'PTNuNu')):
            for interval in ('40to100', '100to200', '200to400', '400to600', '600'):
                for jets in ('1J', '2J'):
                    tail = 'TuneCP5_13p6TeV_amcatnloFXFX-pythia8'
                    early = f'{process}-2Jets_{variable}-{interval}_{jets}_{tail}'
                    canonical = f'{process}-2Jets_Bin-{jets}-{variable}-{interval}_{tail}'
                    records = {'1':dict(dataset=f'/{early}/campaign/NANOAODSIM',xsec_pb=None)}
                    self.assertEqual(apply_background_xsecs(records,index),[])
                    self.assertEqual(records['1']['xsec_pb'],index[canonical]['xsec_pb'])
                    self.assertEqual(records['1']['xsec_source']['matched_sample'],canonical)

    def test_name_mapping_does_not_alias_generator_or_parton_multiplicity(self):
        from TROTASR.workflows.install_assets import cross_section_sample_key
        base = 'WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8'
        for other in (base.replace('1J','0J'),base.replace('FXFX','FXFXold'),
                      base.replace('13p6','13'),base.replace('PTLNu','PTNuNu')):
            self.assertEqual(cross_section_sample_key(other),other)

    def test_extension_statistics_share_one_cross_section(self):
        from TROTASR.workflows.install_assets import build_era_physical
        records={str(i):dict(dataset=f'/{self.primary}/Run3Summer22{ext}/NANOAODSIM',
                            physical_dataset=f'/{self.primary}/Run3Summer22{ext}/NANOAODSIM',
                            xsec_pb=419.82,sumw=weight)
                 for i,(ext,weight) in enumerate((('',100.),('_ext1',300.)))}
        before=copy.deepcopy(records)
        physical,_,factors=build_era_physical(records,7980.4)
        self.assertEqual(len(physical),1)
        self.assertEqual(records,before)
        for row in factors.values():
            self.assertEqual(row['physical_dataset_sumw'],400.)
            self.assertEqual(row['normalization_factor'],419.82*7980.4/400.)
        total=sum(records[k]['sumw']*v['normalization_factor'] for k,v in factors.items())
        self.assertAlmostEqual(total,419.82*7980.4)

    def test_registered_official_values_preserve_sample_definitions(self):
        from TROTASR.workflows.install_assets import adopted_background_xsecs
        index=adopted_background_xsecs()
        self.assertEqual(index['QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8']['xsec_pb'],1961000.)
        self.assertEqual(index['TTWW_TuneCP5_13p6TeV_madgraph-madspin-pythia8']['xsec_pb'],.008203)
        self.assertEqual(index['TTWW_TuneCP5_13p6TeV_madgraph-pythia8']['xsec_pb'],.008191)


class CoverageTest(unittest.TestCase):
    def test_current_payloads_reproduce_without_additions(self):
        xsec=load_signal_xsec(ROOT/'stats/stop_xsec_13p6TeV.json')
        for year in (2024,2025):
            old=read_json(ROOT/'estimations'/('normalization_%s.json.gz'%year))
            new,audit=extend(old,[],xsec)
            self.assertEqual(new['signal_mass_points'],old['signal_mass_points'])
            self.assertEqual(new['dataset_factors'],old['dataset_factors'])
            self.assertEqual(audit['changed_existing_points'],{})

    def test_unique_missing_signal_coverage(self):
        old={'source_sidecars':[{'path':'old/signal_shard_00000.json'}]}
        tasks=[{'key':'2024_signal_shard_00000','metadata':'inputs/2024/signal_shard_00000.json'},
               {'key':'2024_signal_shard_00001','metadata':'inputs/2024/signal_shard_00001.json'}]
        self.assertEqual(missing_tasks(tasks,old),tasks[1:])
        with self.assertRaisesRegex(ValueError,'uniquely'):
            missing_tasks(tasks[:1],{'source_sidecars':[{'path':'other.json'}]})
        with self.assertRaisesRegex(ValueError,'restricted'):
            missing_tasks(tasks+[{'key':'2024_mc_shard_00001','metadata':'inputs/2024/mc_shard_00001.json'}],old)

    def fixture(self):
        rec=dict(dataset='SMS-2Stop-T2bW_example',physical_dataset='example',is_signal=True,
                 is_data=False,process='T2bW',events_written=10,events_read=20,
                 files_attempted=1,files_processed=1,sumw=100.,sumw2=100.,
                 signal_sumw_by_genmodel={'GenModel_T2bW_900_300':100.},
                 signal_event_genweight_sum_by_genmodel={'GenModel_T2bW_900_300':10.})
        old=dict(datasets={},signal_mass_points={},dataset_factors={},source_sidecars=[],
                 source_sidecar_count=0,signal_cutflow_histograms={},luminosity_pb=1000.)
        task=dict(key='2024_signal_shard_00517',metadata='inputs/2024/signal_shard_00517.json',entries=10)
        meta=dict(status='complete',bad_files=[],events_written=10,files_attempted=1,
                  files_processed=1,datasets={'42':rec})
        return old,task,meta,{'mStop900':{'xsec_pb':2.,'xsec_uncertainty_relative':.1}}

    def test_runs_formula_and_no_input_mutation(self):
        old,task,meta,xsec=self.fixture();snapshot=copy.deepcopy(old)
        candidate,audit=extend(old,[(task,meta)],xsec)
        p=candidate['signal_mass_points']['T2bW_mStop900_mLSP300']
        self.assertEqual(p['normalization_factor'],20.)
        self.assertEqual(p['sumw_mass_point'],100.)
        self.assertEqual(p['event_genweight_sum_fallback'],10.)
        self.assertEqual(old,snapshot)
        self.assertEqual(audit['added_shards'],1)
        self.assertEqual(audit['legacy_cutflow_unavailable_for_added_shards'],[task['key']])

    def test_event_sumw_does_not_replace_missing_runs(self):
        old,task,meta,xsec=self.fixture()
        meta['datasets']['42']['signal_sumw_by_genmodel']={}
        with self.assertRaisesRegex(ValueError,'Runs'):
            extend(old,[(task,meta)],xsec)

    def test_changed_existing_points_are_explicit(self):
        old,task,meta,xsec=self.fixture()
        old,_=extend(old,[(task,meta)],xsec)
        new,audit=extend(old,[(dict(task,key='new'),meta)],xsec)
        self.assertEqual(audit['changed_existing_points']['T2bW_mStop900_mLSP300'],
                         {'old':20.,'new':10.,'old_sumw':100.,'new_sumw':200.})

    def test_incomplete_or_invalid_source_rejected(self):
        old,task,meta,xsec=self.fixture()
        meta['files_processed']=0
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            extend(old,[(task,meta)],xsec)
        meta['files_processed']=1
        meta['datasets']['42']['signal_sumw_by_genmodel']['GenModel_T2bW_900_300']=float('nan')
        with self.assertRaisesRegex(ValueError,'Runs'):
            extend(old,[(task,meta)],xsec)

    def test_extension_gate_rejects_changed_old_factor(self):
        old,task,meta,xsec=self.fixture()
        old['status']='complete';old['blocked_signal_mass_points']={}
        new,_=extend(old,[(task,meta)],xsec)
        self.assertEqual(verify_extension(old,new),['T2bW_mStop900_mLSP300'])
        before=copy.deepcopy(new);after=copy.deepcopy(new)
        after['signal_mass_points']['T2bW_mStop900_mLSP300']['normalization_factor']=999.
        with self.assertRaisesRegex(ValueError,'record changed'):
            verify_extension(before,after)
        after=copy.deepcopy(new);after['luminosity_pb']=99.
        with self.assertRaisesRegex(ValueError,'prescription changed'):
            verify_extension(old,after)

    def test_one_corrected_attempt_only(self):
        plan={'normalization_extension_recovery':{'retry_keys':['signal']}}
        failed=dict(status='failed',attempt=1,exception_type='RuntimeError',
                    error='missing, non-finite, or non-positive signal normalization factor for x')
        self.assertEqual(retry_keys(plan,{'signal':failed}),['signal'])
        self.assertEqual(retry_keys(plan,{'signal':dict(failed,attempt=2)}),[])
        self.assertEqual(retry_keys(plan,{'signal':dict(status='complete')}),[])
        with self.assertRaisesRegex(ValueError,'Different failure'):
            retry_keys(plan,{'signal':dict(failed,error='other error')})
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            retry_keys(plan,{'signal':dict(status='running',attempt=1)})

    def test_exact_prior_contract_only(self):
        old={'code':{'a':'old'},'normalization_sha256':'old'}
        new={'code':{'a':'old'},'normalization_sha256':'new'}
        self.assertEqual(payload_contract({'contract':dict(old,input='file')},new,old),old)
        self.assertEqual(payload_contract({'contract':dict(old,code={'a':'BAD'})},new,old),new)

    def test_staged_runner_changes_are_only_explicit_recovery_hooks(self):
        staged=ROOT/'validation/normalization_recovery_20260923/staged_workflows'
        original=(ROOT/'workflows/nominal_campaign.py').read_text()
        candidate=(staged/'nominal_campaign.py').read_text()
        hook=('        from TROTASR.workflows.normalization_recovery import prior_contracts, retry_keys, adopt_worker\n'
              '        prior_contracts(plan)\n        pending.extend(retry_keys(plan, states))\n')
        restored=candidate.replace(hook,'').replace(
            '        active = {}\n        # Explicitly retained original workers finish in place, without restart.\n',
            '        # An interrupted child is adopted only after its saved PID has exited.\n').replace(
            '                retained = adopt_worker(plan, campaign, key, s)\n'
            '                if retained is not None:\n'
            '                    active[key] = retained\n'
            '                    continue\n','').replace(
            '        environment = dict(os.environ, **THREAD_ENV)\n',
            '        active = {}\n        environment = dict(os.environ, **THREAD_ENV)\n')
        baseline=read_json(ROOT/'normalization_recovery_sources.json')['original_active_files_sha256']['workflows/nominal_campaign.py']
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),baseline)
        self.assertIn(original,(candidate,restored))

    def test_retained_worker_pid_identity_and_exit_receipt(self):
        record=dict(pid=100,attempt=1,identity=dict(pid=100,start_ticks=123,arguments=['worker']),
                    history='history/signal.attempt1.json')
        proc=compatibility.RetainedProcess(record)
        with patch.object(compatibility,'process_identity',return_value=record['identity']):
            self.assertIsNone(proc.poll())
        with patch.object(compatibility,'process_identity',return_value=dict(record['identity'],start_ticks=456)):
            with self.assertRaisesRegex(RuntimeError,'reused'):proc.poll()
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as directory:
            root=Path(directory)
            with patch.object(compatibility,'ROOT',root),patch.object(compatibility,'process_identity',return_value=None):
                self.assertEqual(proc.poll(),1)
                write_json(root/record['history'],dict(status='complete'))
                self.assertEqual(proc.poll(),0)

    def test_only_explicit_original_worker_is_adopted(self):
        record=dict(pid=100,attempt=1,identity=dict(pid=100,start_ticks=123,arguments=['worker']),
                    history='history/signal.attempt1.json')
        plan=dict(normalization_extension_recovery=dict(retained_workers={'signal':record}))
        self.assertIsNone(compatibility.adopt_worker({},ROOT,'signal',{}))
        with self.assertRaisesRegex(ValueError,'state changed'):
            compatibility.adopt_worker(plan,ROOT,'signal',dict(pid=101,attempt=1))
        with patch.object(compatibility,'process_identity',return_value=record['identity']):
            proc,stream,attempt=compatibility.adopt_worker(plan,ROOT,'signal',dict(pid=100,attempt=1))
            stream.close()
            self.assertEqual(attempt,1)
            self.assertEqual(proc.pid,100)

    def test_transaction_preserves_completed_histogram_and_failed_attempt(self):
        # Synthetic bookkeeping only: no ROOT, workspace, cards or fits.
        with tempfile.TemporaryDirectory(prefix='normalization-transaction-test-',dir=ROOT/'validation') as directory:
            root=Path(directory);campaign=root/'campaigns/test';products=root/'stats/test'
            recovery=campaign/'recovery/missing_normalization_20260923'
            staged=root/'validation/normalization_recovery_20260923/staged_workflows'
            for p in (products,recovery,staged,root/'workflows'):p.mkdir(parents=True,exist_ok=True)
            for name in ('nominal_campaign.py','merge_nominal.py'):
                (root/'workflows'/name).write_text('old '+name)
                (staged/name).write_text('recovered '+name)
            assets={'files':[]};audit={'status':'candidate_prepared_not_promoted','years':{}}
            original_paths={};candidate_paths={}
            for year in (2024,2025):
                old=dict(status='complete',luminosity_pb=1000.,luminosity_fb=1.,normalization_policy={},
                    background_xsec_overrides={},signal_xsec_source={},signal_mass_points={},
                    datasets={},dataset_factors={},blocked_signal_mass_points={})
                new=copy.deepcopy(old);new['signal_mass_points']['new_point']={'normalization_factor':2.}
                a=root/'estimations'/('normalization_%s.json.gz'%year);b=recovery/a.name
                write_json(a,old);write_json(b,new);original_paths[str(year)]=a;candidate_paths[str(year)]=b
                assets['files'].append(dict(target=str(a.relative_to(root)),sha256=sha256(a)))
                audit['years'][str(year)]=dict(original_sha256=sha256(a),candidate_sha256=sha256(b),
                    candidate=str(b.relative_to(root)),added_mass_points=['new_point'],changed_existing_points={},
                    blocked_points=[],missing_keys=['signal_'+str(year)])
            write_json(root/'assets.json',assets)
            tasks=[];failed={}
            for key in ('signal_2024','signal_2025','success'):
                src=root/'inputs'/(key+'.dat');src.parent.mkdir(exist_ok=True);src.write_text('not ROOT')
                tasks.append(dict(key=key,input=str(src.relative_to(root)),fingerprint=promotion.fingerprint(src)))
                if key=='success':
                    out=campaign/'outputs/success.json.gz';write_json(out,{'synthetic':'histogram-content-unchanged'})
                    state=dict(status='complete',output=str(out.relative_to(root)),output_sha256=sha256(out))
                    success_bytes=out.read_bytes()
                else:
                    state=dict(status='failed',attempt=1,exception_type='RuntimeError',
                        error='missing, non-finite, or non-positive signal normalization factor for new_point')
                    failed[key]=copy.deepcopy(state)
                write_json(campaign/'state'/(key+'.json'),state)
            contracts=lambda y:dict(year=y,code={'unchanged':'yes'},normalization_sha256=sha256(original_paths[str(y)]),
                                     assets_sha256=sha256(root/'assets.json'))
            plan=dict(tasks=tasks,contracts={str(y):contracts(y) for y in (2024,2025)},
                      runner_sha256=sha256(root/'workflows/nominal_campaign.py'))
            write_json(campaign/'plan.json',plan);audit['plan_sha256']=sha256(campaign/'plan.json')
            write_json(recovery/'audit.json',audit)
            write_json(campaign/'state.json',dict(status='complete_with_failures',active=0,pending=0,failed=list(failed)))
            write_json(campaign/'chain_state.json',dict(status='needs_attention',stages={'histograms':{'exit_code':1}}))
            write_json(products/'products_state.json',dict(status='controller_error_children_retained',stages={},child_pids={}))
            files=[str(p.relative_to(root)) for p in staged.iterdir()]
            sources=dict(promotion_files_sha256={n:sha256(root/n) for n in files},
                original_active_files_sha256={'workflows/'+n:sha256(root/'workflows'/n) for n in ('nominal_campaign.py','merge_nominal.py')})
            write_json(root/'normalization_recovery_sources.json',sources)
            with patch.object(promotion,'ROOT',root),patch.object(compatibility,'ROOT',root),patch.object(promotion,'current_contract',contracts):
                self.assertEqual(promotion.run(campaign,False),0)
                self.assertEqual(promotion.run(campaign,True),0)
                new_plan=read_json(campaign/'plan.json')
                compatibility.prior_contracts(new_plan)
                self.assertEqual(retry_keys(new_plan,failed),sorted(failed))
                self.assertEqual(out.read_bytes(),success_bytes)
                for k,state in failed.items():self.assertEqual(read_json(campaign/'state'/(k+'.json')),state)
                self.assertFalse((products/'products_state.json').exists())
                with self.assertRaisesRegex(ValueError,'Already promoted'):
                    promotion.run(campaign,True)


if __name__=='__main__':unittest.main()
