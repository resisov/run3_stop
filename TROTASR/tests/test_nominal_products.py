"""Controller contract tests; no fit or synthetic production success."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows import run_nominal_products as runner
from TROTASR.workflows import run_nominal_chain as chain


class Run3PlotPriorityTests(unittest.TestCase):
    def setUp(self):
        self.campaign=ROOT/'campaigns/early_systematics_20261025_preparation'
        self.output=ROOT/'stats/run3_systematics_20261025'
        self.config=ROOT/'jsons/run3_systematics_20261025_config.json'
        self.old=runner.stage_plan(self.campaign,self.output,self.config,98)
        self.new=runner.stage_plan(self.campaign,self.output,self.config,98,True)
        self.done={kind+'_'+year for kind in ('measure','distributions','measurements','transfer_factors')
                   for year in ('2022','2022EE','2023','2023BPix')}

    def recovery(self):
        old,new=copy.deepcopy(self.old),copy.deepcopy(self.new)
        old['grid']['command'].append('--resume-extracted')
        new['grid']['command'].extend(['--template-verify-workers','12','--resume-written-templates'])
        before=dict(campaign='same',plan_sha256='same',grid_config_sha256='same',stages=old)
        after=dict(before,stages=new,approved_plot_priority=dict(id=runner.RUN3_PLOT_PRIORITY,user_approved=True))
        records={n:dict(status='succeeded',exit_code=0,pid=None) for n in self.done}
        records['grid']=dict(status='failed',exit_code=-15,pid=None)
        records.update({n:dict(status='blocked') for n in old if n not in records})
        return dict(status='needs_attention',pid=None,child_pids={},contract=before,stages=records),after

    def test_opt_in_adds_only_late_gnn_and_moves_thirteen_unstarted_plots(self):
        self.assertEqual(set(self.new)-set(self.old),{'gnn_controls_2024','gnn_controls_2025'})
        self.assertEqual(sum(s['lane']=='plot' for s in self.new.values()),15)
        for name,spec in self.old.items():
            after=self.new[name]
            if name in runner.RUN3_PLOT_NAMES:
                self.assertEqual(after,{**spec,'lane':'plot','priority':0})
            else:self.assertEqual(spec,after)
        self.assertEqual({n for n,_ in runner.stage_order(self.new)[:15]},set(runner.RUN3_PLOT_NAMES))
        self.assertEqual(list(runner.stage_order(self.old)),list(self.old.items()))
        self.assertTrue(all(self.new[n]==self.old[n] for n in self.done))
        with self.assertRaises(ValueError):
            runner.stage_plan(self.campaign,self.output,ROOT/'stats/nominal_grid_config.json',98,True)

    def test_exact_migration_preserves_all_sixteen_successes(self):
        previous,contract=self.recovery()
        with patch.object(runner,'verify_receipt',return_value=(True,{})) as verify:
            retained=runner.resume_records(previous,contract,self.output)
        self.assertEqual(set(retained),self.done)
        self.assertEqual(verify.call_count,16)
        self.assertTrue(all(r['reused_unchanged'] for r in retained.values()))

    def test_migration_rejects_scope_or_started_work_changes(self):
        for kind in ('no_approval','extra_stage','done_changed','plot_command','plot_started',
                     'fit_changed','late_source','missing_success','live_parent','live_child','readers'):
            previous,contract=self.recovery()
            if kind=='no_approval':contract.pop('approved_plot_priority')
            elif kind=='extra_stage':contract['stages']['extra']=dict(contract['stages']['gnn_controls_2024'])
            elif kind=='done_changed':contract['stages']['measure_2022']['priority']=0
            elif kind=='plot_command':contract['stages']['predictions_2022']['command'].append('--different')
            elif kind=='plot_started':previous['stages']['predictions_2022']['started']=123
            elif kind=='fit_changed':contract['stages']['limits']['command'].append('--different')
            elif kind=='late_source':contract['stages']['gnn_controls_2024']['command'][4]='wrong'
            elif kind=='missing_success':previous['stages']['measure_2022']['status']='failed'
            elif kind=='live_parent':previous['pid']=123
            elif kind=='live_child':previous['child_pids']={'grid':123}
            elif kind=='readers':
                command=contract['stages']['grid']['command'];command[command.index('--template-verify-workers')+1]='11'
            with self.subTest(kind=kind),patch.object(runner,'verify_receipt',return_value=(True,{})):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.resume_records(previous,contract,self.output)

    def test_four_plot_lane_and_global_resources_are_bounded(self):
        plots=list(runner.RUN3_PLOT_NAMES)
        self.assertTrue(runner.lane_available(self.new,plots[:3],self.new[plots[4]]))
        self.assertFalse(runner.lane_available(self.new,plots[:4],self.new[plots[4]]))
        self.assertFalse(runner.lane_available(self.new,['cronly'],self.new['grid']))
        self.assertTrue(runner.lane_available(self.new,plots[:4],self.new['limits']))
        gib=1024**3
        self.assertTrue(runner.resource_admission(118,48*gib,3,True,80*gib))
        for slots,memory,disk in ((119,100*gib,80*gib),(118,47*gib,80*gib),(1,24*gib,80*gib),(1,100*gib,79*gib)):
            self.assertFalse(runner.resource_admission(slots,memory,3,True,disk))
        self.assertTrue(runner.resource_admission(98,36*gib))
        self.assertFalse(runner.resource_admission(99,36*gib))

    def test_wrapper_dispatch_does_not_rewrite_recorded_plan(self):
        command=copy.deepcopy(self.new['predictions_2022']['command'])
        before=copy.deepcopy(command)
        actual=runner.worker_command(command)
        self.assertEqual(actual[:2],['bash','/hep2-scratch/twkim/postprocessing/TROTASR/workflows/run_inference.sh'])
        self.assertEqual(actual[2:],command[2:])
        self.assertEqual(command,before)
        self.assertEqual(runner.command_entrypoint(actual),runner.command_entrypoint(command))
        with self.assertRaises(ValueError):runner.worker_command(['python','-u','outside.py'])

    def test_only_hash_bound_early_qcd_reference_can_change_config(self):
        current=read_json(self.config)
        prior=copy.deepcopy(current);prior.pop('early_qcd_fit_exclusions')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);(folder/'grid').mkdir()
            old_path=folder/'prior_config.json';new_path=folder/'current_config.json'
            write_json(old_path,prior);write_json(new_path,current)
            write_json(folder/'grid/written_template_resume_approval.json',
                       dict(prior_config=str(old_path.relative_to(ROOT))))
            previous,contract=self.recovery()
            previous['contract']['grid_config_sha256']=sha256(old_path)
            contract['grid_config_sha256']=sha256(new_path)
            command=contract['stages']['grid']['command']
            command[command.index('--config')+1]=str(new_path)
            runner.approved_products_early_qcd_config_change(previous,contract,folder)
            changed=copy.deepcopy(current);changed['unrelated_new_option']=True
            write_json(new_path,changed);contract['grid_config_sha256']=sha256(new_path)
            with self.assertRaisesRegex(ValueError,'unrelated configuration'):
                runner.approved_products_early_qcd_config_change(previous,contract,folder)
            write_json(new_path,current);contract['grid_config_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                runner.approved_products_early_qcd_config_change(previous,contract,folder)
            contract['grid_config_sha256']=sha256(new_path)
            previous['stages']['grid']['status']='running'
            with self.assertRaisesRegex(ValueError,'stopped written-template'):
                runner.approved_products_early_qcd_config_change(previous,contract,folder)

    def minimal_recovery(self):
        previous,current=self.recovery()
        previous['contract']=copy.deepcopy(current)
        for name in current['stages']:
            previous['stages'].setdefault(name,dict(status='blocked'))
        current['stages']['grid']['command'].extend(['--template-check-mode','minimal'])
        current['approved_template_check_mode']=dict(
            id=runner.MINIMAL_TEMPLATE_CHECK,user_approved=True,check_mode='minimal')
        return previous,current

    def test_minimal_mode_only_addition_preserves_existing_52_stage_plan(self):
        previous,current=self.minimal_recovery()
        before=copy.deepcopy(previous)
        with patch.object(runner,'verify_receipt',return_value=(True,{})) as verify:
            retained=runner.resume_records(previous,current,self.output)
        self.assertEqual(set(retained),self.done)
        self.assertEqual(verify.call_count,16)
        self.assertEqual(previous,before)
        self.assertEqual(current['stages']['grid']['command'],
            previous['contract']['stages']['grid']['command']+['--template-check-mode','minimal'])

    def test_minimal_mode_migration_rejects_any_other_change(self):
        for kind in ('no_approval','config','extra_stage','done','unstarted_plot','fit','workers',
                     'remove_resume','wrong_mode','grid_succeeded','missing_success','live'):
            previous,current=self.minimal_recovery()
            command=current['stages']['grid']['command']
            if kind=='no_approval':current.pop('approved_template_check_mode')
            elif kind=='config':current['grid_config_sha256']='changed'
            elif kind=='extra_stage':current['stages']['extra']=copy.deepcopy(current['stages']['grid'])
            elif kind=='done':current['stages']['measure_2022']['priority']=0
            elif kind=='unstarted_plot':current['stages']['predictions_2022']['priority']=1
            elif kind=='fit':current['stages']['limits']['command'].append('--different')
            elif kind=='workers':command[command.index('--template-verify-workers')+1]='11'
            elif kind=='remove_resume':command.remove('--resume-written-templates')
            elif kind=='wrong_mode':command[-1]='exhaustive'
            elif kind=='grid_succeeded':previous['stages']['grid']['status']='succeeded'
            elif kind=='missing_success':previous['stages']['measure_2022']['status']='failed'
            elif kind=='live':previous['child_pids']={'grid':123}
            with self.subTest(kind=kind),patch.object(runner,'verify_receipt',return_value=(True,{})):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.resume_records(previous,current,self.output)

    def test_minimal_grid_receipt_cannot_claim_exhaustive_or_omit_scope(self):
        value=dict(status='cards_ready',template_check_mode='minimal',numeric_readback_performed=False,
            numeric_readback_checked=False,root_integrity_checked=True,
            root_integrity_scope='file_open_header_object_inventory_types_and_checksums',
            template_check_user_approved=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/'manifest.json'
            spec=dict(receipt=str(path.relative_to(ROOT)),statuses=['cards_ready'],
                command=['python','-u',str(ROOT/'workflows/build_nominal_grid.py'),'--template-check-mode','minimal'])
            write_json(path,value)
            self.assertTrue(runner.verify_receipt(spec,0)[0])
            self.assertFalse(runner.verify_receipt(spec,1)[0])
            for key in set(value)-{'status'}:
                for change in ('missing','wrong'):
                    bad=dict(value)
                    if change=='missing':bad.pop(key)
                    else:bad[key]=not value[key] if isinstance(value[key],bool) else 'exhaustive'
                    write_json(path,bad)
                    with self.subTest(key=key,change=change):
                        with self.assertRaisesRegex(ValueError,'non-exhaustive check scope'):
                            runner.verify_receipt(spec,0)

    def display_range_recovery(self, minimal=True):
        previous,current=self.minimal_recovery()
        if not minimal:
            del current['stages']['grid']['command'][-2:]
            current.pop('approved_template_check_mode')
        for name,args in runner.LIMIT_RANGE_ARGUMENTS.items():
            current['stages'][name]['command'].extend(args)
        current['approved_limit_plot_range']=dict(id=runner.DISPLAY_LIMIT_RANGE,user_approved=True,
            mstop_gev=[600,1800],mlsp_gev=[1,1800],expected_display_points=895)
        return previous,current

    def test_display_range_forwarding_changes_only_two_limits_and_contours(self):
        plan=runner.stage_plan(self.campaign,self.output,self.config,98,True,True)
        for name,before in self.new.items():
            if name in runner.LIMIT_RANGE_ARGUMENTS:
                self.assertEqual(plan[name],{**before,'command':before['command']+runner.LIMIT_RANGE_ARGUMENTS[name]})
            else:self.assertEqual(plan[name],before)
        self.assertNotIn('--plot-range-only',self.old['limits']['command'])
        self.assertNotIn('--plot-range-only',self.new['limits']['command'])

    def test_display_range_migration_with_or_without_minimal_preserves_successes(self):
        for minimal in (False,True):
            previous,current=self.display_range_recovery(minimal)
            with self.subTest(minimal=minimal),patch.object(runner,'verify_receipt',return_value=(True,{})) as verify:
                retained=runner.resume_records(previous,current,self.output)
                self.assertEqual(set(retained),self.done)
                self.assertEqual(verify.call_count,16)

    def test_display_range_rejects_started_fits_or_any_unrelated_change(self):
        for kind in ('no_approval','bounds','count','config','only_primary','started_primary',
                     'started_recovery','started_contours','extra_flag','changed_grid','changed_plot','changed_done'):
            previous,current=self.display_range_recovery()
            if kind=='no_approval':current.pop('approved_limit_plot_range')
            elif kind=='bounds':current['approved_limit_plot_range']['mstop_gev']=[500,1800]
            elif kind=='count':current['approved_limit_plot_range']['expected_display_points']=955
            elif kind=='config':current['grid_config_sha256']='different'
            elif kind=='only_primary':current['stages']['limit_recovery']['command'].pop()
            elif kind=='started_primary':previous['stages']['limits']['started']=1
            elif kind=='started_recovery':previous['stages']['limit_recovery']['log']='earlier.log'
            elif kind=='started_contours':previous['stages']['contours']['started']=1
            elif kind=='extra_flag':current['stages']['limits']['command'].append('--different')
            elif kind=='changed_grid':current['stages']['grid']['command'].append('--different')
            elif kind=='changed_plot':current['stages']['predictions_2022']['priority']=1
            elif kind=='changed_done':current['stages']['measurements_2022']['priority']=0
            with self.subTest(kind=kind),patch.object(runner,'verify_receipt',return_value=(True,{})):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.resume_records(previous,current,self.output)

    def test_display_range_receipts_require_exact_selected_not_source_totals(self):
        value=dict(status='fits_complete_pending_plots',plot_range_only=True,source_total=955,total=895,
            excluded_out_of_plot_range=60,expected_points_by_model=dict(T2tt=205,T2bW=345,T2tb=345))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/'state.json'
            spec=dict(receipt=str(path.relative_to(ROOT)),statuses=['fits_complete_pending_plots'],
                command=['python','-u',str(ROOT/'workflows/combine_limits.py'),'--plot-range-only'])
            write_json(path,value)
            self.assertTrue(runner.verify_receipt(spec,0)[0])

            for key in set(value)-{'status'}:
                bad=dict(value);bad.pop(key);write_json(path,bad)
                with self.subTest(missing=key):
                    with self.assertRaisesRegex(ValueError,'only 895 selected points'):
                        runner.verify_receipt(spec,0)
            bad=dict(value,total=955);write_json(path,bad)
            with self.assertRaises(ValueError):runner.verify_receipt(spec,0)
            recovery=dict(value,status='all_verified_pending_reconciliation',targets=7,remaining=[])
            write_json(path,recovery)
            spec['statuses']=['all_verified_pending_reconciliation']
            spec['command'][2]=str(ROOT/'workflows/reconcile_limit_fits.py')
            self.assertTrue(runner.verify_receipt(spec,0)[0])

    def card_recovery(self):
        previous,current=self.display_range_recovery()
        previous['contract']=copy.deepcopy(current)
        current['stages']['grid']['command']=runner.card_resume_command(current['stages']['grid']['command'])
        current['approved_card_resume']=dict(id=runner.PARALLEL_CARD_RESUME,user_approved=True,card_workers=16)
        return previous,current

    def test_card_resume_changes_only_grid_and_preserves_sixteen_successes(self):
        previous,current=self.card_recovery();before=copy.deepcopy(previous)
        with patch.object(runner,'verify_receipt',return_value=(True,{})) as verify:
            retained=runner.resume_records(previous,current,self.output)
        self.assertEqual(previous,before)
        self.assertEqual(set(retained),self.done)
        self.assertEqual(verify.call_count,16)
        self.assertEqual(len(current['stages']),52)
        for name,spec in previous['contract']['stages'].items():
            if name!='grid':self.assertEqual(current['stages'][name],spec)
        command=current['stages']['grid']['command']
        self.assertIn('--resume-cards',command)
        self.assertNotIn('--resume-written-templates',command)
        self.assertNotIn('--resume-extracted',command)
        self.assertEqual(command[-2:],['--card-workers','16'])
        for name in ('limits','limit_recovery'):
            command=current['stages'][name]['command']
            self.assertEqual(command[command.index('--workers')+1],'45')
            self.assertIn('--plot-range-only',command)
        self.assertIn('--assist-active-primary',current['stages']['limit_recovery']['command'])

    def test_card_resume_rejects_any_other_input_stage_or_scope_change(self):
        for kind in ('no_approval','workers','config','campaign','plan','extra_stage','grid_receipt',
                     'done','missing_success','plot','fit','recovery','contour','grid_succeeded',
                     'live_parent','live_child','missing_minimal','missing_readers','both_resumes'):
            previous,current=self.card_recovery();command=current['stages']['grid']['command']
            if kind=='no_approval':current.pop('approved_card_resume')
            elif kind=='workers':command[-1]='15'
            elif kind in ('config','campaign','plan'):
                current[dict(config='grid_config_sha256',campaign='campaign',plan='plan_sha256')[kind]]='changed'
            elif kind=='extra_stage':current['stages']['extra']=copy.deepcopy(current['stages']['grid'])
            elif kind=='grid_receipt':current['stages']['grid']['receipt']='other.json'
            elif kind=='done':current['stages']['measure_2022']['priority']=0
            elif kind=='missing_success':previous['stages']['measure_2022']['status']='failed'
            elif kind=='plot':current['stages']['gnn_controls_2024']['priority']=1
            elif kind=='fit':current['stages']['limits']['command'].append('--different')
            elif kind=='recovery':current['stages']['limit_recovery']['command'].remove('--assist-active-primary')
            elif kind=='contour':current['stages']['contours']['command'].remove('--live-snapshot')
            elif kind=='grid_succeeded':previous['stages']['grid']['status']='succeeded'
            elif kind=='live_parent':previous['pid']=123
            elif kind=='live_child':previous['child_pids']={'grid':123}
            elif kind in ('missing_minimal','missing_readers'):
                flag='--template-check-mode' if kind=='missing_minimal' else '--template-verify-workers'
                old_command=previous['contract']['stages']['grid']['command']
                del old_command[old_command.index(flag):old_command.index(flag)+2]
            elif kind=='both_resumes':command.append('--resume-written-templates')
            with self.subTest(kind=kind),patch.object(runner,'verify_receipt',return_value=(True,{})):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.resume_records(previous,current,self.output)

    def test_card_resume_same_plan_can_continue_after_a_stopped_failure(self):
        previous,current=self.card_recovery();previous['contract']=copy.deepcopy(current)
        with patch.object(runner,'verify_receipt',return_value=(True,{})):
            self.assertEqual(set(runner.resume_records(previous,current,self.output)),self.done)

    def test_card_command_rejects_duplicate_or_conflicting_resume_options(self):
        previous,_=self.card_recovery();command=previous['contract']['stages']['grid']['command']
        with self.assertRaises(ValueError):runner.card_resume_command(command,15)
        for flag in ('--resume-extracted','--resume-cards','--card-workers','--resume-written-templates'):
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                runner.card_resume_command(command+[flag])

    def test_card_grid_reservation_counts_sixteen_workers_and_both_controllers(self):
        _,current=self.card_recovery();plan=current['stages']
        self.assertTrue(runner.limit_priority_admission('grid',plan,[],{},18))
        self.assertFalse(runner.limit_priority_admission('grid',plan,[],{},17))

    def test_card_receipt_requires_hash_bound_complete_workers_and_exact_reuse(self):
        cards=dict(status='cards_complete',completed=955,total=955,workers=16,active=0,pending=0,
            pid=None,worker_pids=[],worker_exit_codes={'123':0},root_and_cache_rewrite=False,
            reused_pairs_preserved=True,reused=197,execution=dict(resume_cards=True,
                representative_check=dict(text_exact=True,metadata_exact=True,existing_bytes_modified=False)))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);receipt=folder/'card_execution.json';manifest=folder/'manifest.json'
            spec=dict(receipt=str(manifest.relative_to(ROOT)),statuses=['cards_ready'],command=['--resume-cards'])
            def save(value):
                write_json(receipt,value)
                write_json(manifest,dict(status='cards_ready',card_execution=dict(
                    path=str(receipt.relative_to(ROOT)),sha256=sha256(receipt))))
            for reused in (197,300,955):
                save(dict(cards,reused=reused,worker_exit_codes={} if reused==955 else {'123':0}))
                self.assertTrue(runner.verify_receipt(spec,0)[0])
            for key,value in (('status','writing_cards'),('completed',954),('total',895),('workers',15),
                              ('active',1),('pending',1),('pid',123),('worker_pids',[123]),
                              ('worker_exit_codes',{'123':-15}),('root_and_cache_rewrite',True),
                              ('reused_pairs_preserved',False),('execution',{})):
                save(dict(cards,**{key:value}))
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'worker-exit evidence'):
                    runner.verify_receipt(spec,0)
            save(cards);write_json(receipt,dict(cards,reused=198))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):runner.verify_receipt(spec,0)
            save(cards);value=read_json(manifest);value['card_execution']['path']='other/card_execution.json'
            write_json(manifest,value)
            with self.assertRaisesRegex(ValueError,'path/checksum mismatch'):runner.verify_receipt(spec,0)
            write_json(manifest,dict(status='cards_ready'));spec['command']=[]
            self.assertTrue(runner.verify_receipt(spec,0)[0])

    def test_run_forwards_card_options_without_extracted_or_written_resume(self):
        class ReachedResume(Exception):pass
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);campaign=folder/'campaign';output=folder/'products'
            write_json(campaign/'plan.json',{})
            write_json(output/'products_state.json',dict(stages={'grid':dict(status='failed')}))
            with patch.object(runner.sys,'platform','linux'), \
                 patch.object(runner,'resume_records',side_effect=ReachedResume) as resume:
                with self.assertRaises(ReachedResume):
                    runner.run(campaign,output,self.config,worker_budget=98,resume_failed=True,
                        prioritize_run3_plots=True,template_verify_workers=12,template_check_mode='minimal',
                        limit_plot_range_only=True,card_workers=16,resume_cards=True)
            contract=resume.call_args.args[1];command=contract['stages']['grid']['command']
            self.assertEqual(command[-6:],['12','--resume-cards','--template-check-mode','minimal','--card-workers','16'])
            self.assertNotIn('--resume-extracted',command)
            self.assertNotIn('--resume-written-templates',command)
            self.assertEqual(contract['approved_card_resume']['card_workers'],16)
            expected=runner.stage_plan(campaign,output,self.config,98,True,True)
            self.assertTrue(all(contract['stages'][n]==expected[n] for n in expected if n!='grid'))

    def test_card_cli_mode_rejects_missing_or_conflicting_authority(self):
        options=dict(worker_budget=98,resume_failed=True,prioritize_run3_plots=True,
            template_verify_workers=12,template_check_mode='minimal',limit_plot_range_only=True,
            card_workers=16,resume_cards=True)
        for key,value in (('resume_failed',False),('prioritize_run3_plots',False),
                          ('template_verify_workers',None),('template_check_mode',None),
                          ('limit_plot_range_only',False),('card_workers',None),
                          ('resume_cards',False),('plots_only',True),('resume_written_templates',True)):
            with self.subTest(key=key),patch.object(runner.sys,'platform','linux'):
                with self.assertRaisesRegex(ValueError,'sixteen-worker minimal-grid resume'):
                    runner.run(self.campaign,self.output,self.config,**{**options,key:value})

    def test_live_recovery_readiness_requires_actual_primary_evidence(self):
        plan=runner.stage_plan(self.campaign,self.output,self.config,98,True,True)
        spec=plan['limit_recovery'];records={'limits':dict(status='running',pid=123)}
        with patch.object(runner,'live_primary_ready',return_value=False):
            self.assertEqual(runner.disposition('limit_recovery',spec,records),'wait')
        with patch.object(runner,'live_primary_ready',return_value=True):
            self.assertEqual(runner.disposition('limit_recovery',spec,records),'ready')
        self.assertEqual(runner.disposition('limit_recovery',spec,{'limits':dict(status='succeeded')}),'not_needed')
        self.assertEqual(runner.disposition('limit_recovery',spec,{'limits':dict(status='failed',
            receipt_snapshot=dict(status='needs_attention',results={'point':{}}))}),'ready')

    def test_live_limit_pair_reserves_pools_and_defers_impacts(self):
        plan=runner.stage_plan(self.campaign,self.output,self.config,98,True,True)
        plots=list(runner.RUN3_PLOT_NAMES)[:4]
        records={'grid':dict(status='succeeded'),'limits':dict(status='running')}
        active=['limits',*plots]
        self.assertTrue(runner.limit_priority_admission('limit_recovery',plan,active,records,98))
        self.assertFalse(runner.limit_priority_admission('impacts',plan,active,records,98))
        self.assertFalse(runner.limit_priority_admission('cronly',plan,active,records,98))
        self.assertTrue(runner.limit_priority_admission('cronly',plan,['limits',*plots[:3]],records,98))
        terminal={**records,'limits':dict(status='failed'),'limit_recovery':dict(status='succeeded')}
        self.assertTrue(runner.limit_priority_admission('impacts',plan,[],terminal,98))

    def test_live_primary_gate_binds_actual_pid_ticks_argv_and_selection(self):
        selection=dict(plot_range_only=True,source_total=955,total=895,excluded_out_of_plot_range=60,
            expected_points_by_model=dict(T2tt=205,T2bW=345,T2tb=345),selected_keys_sha256='a'*64)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);manifest=folder/'grid.json';write_json(manifest,{})
            args=['--manifest',str(manifest),'--output',str(folder),'--workers','45','--plot-range-only']
            spec=dict(command=['python','-u',str(ROOT/'workflows/reconcile_limit_fits.py'),*args,'--assist-active-primary'])
            parent=dict(pid=123,start_ticks=456,status='running')
            self.assertFalse(runner.live_primary_ready(spec,parent))
            state=dict(status='running',pid=123,observed_sr_used=False,**{k:v for k,v in selection.items() if k!='selected_keys_sha256'})
            contract=dict(manifest_sha256=sha256(manifest),fit_selection=selection)
            write_json(folder/'state.json',state);write_json(folder/'contract.json',contract)
            identity=dict(pid=123,start_ticks=456,argv=['python','-u',str(ROOT/'workflows/combine_limits.py'),*args])
            with patch('TROTASR.workflows.combine_limits.selected_fit_points',return_value=([],selection)), \
                 patch('TROTASR.workflows.continue_limit_fits.assist_identity',return_value=identity):
                self.assertTrue(runner.live_primary_ready(spec,parent))
                for field,value in (('pid',124),('start_ticks',457)):
                    with self.subTest(field=field),self.assertRaises(ValueError):
                        runner.live_primary_ready(spec,{**parent,field:value})
                write_json(folder/'contract.json',{**contract,'manifest_sha256':'0'*64})
                with self.assertRaisesRegex(ValueError,'selection contract changed'):
                    runner.live_primary_ready(spec,parent)


class ProductControllerTests(unittest.TestCase):
    def test_finite_continuation_adopts_running_and_advances_only_failed_phase(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory)
            receipt=folder/'numerical_all_unfinished_20260930/state.json'
            write_json(receipt,dict(status='running',pid=123,results={}))
            owner=dict(pid=123,start_ticks=456,arguments=['python',str(folder)])
            with patch('TROTASR.workflows.normalization_recovery.process_identity',return_value=owner):
                step=chain.recovery_step(folder,'limits',folder/'manifest.json',8)
                self.assertEqual(step['action'],'wait')
            write_json(receipt,dict(status='needs_attention',pid=None,results={'point':dict(status='failed')}))
            step=chain.recovery_step(folder,'limits',folder/'manifest.json',8)
            self.assertEqual(step['phase'],'numerical_remaining_20261009')
            self.assertIn('--recover-remaining',step['command'])
            write_json(folder/'numerical_remaining_20261009/state.json',dict(
                status='all_verified_pending_reconciliation',pid=None,results={}))
            step=chain.recovery_step(folder,'limits',folder/'manifest.json',8)
            self.assertEqual(step['phase'],'plot')

    def test_impact_continuation_uses_next_policy_not_same_failed_command(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory)
            write_json(folder/'impact_status.json',dict(status='needs_attention'))
            write_json(folder/'bounded_sgamma_complete/status.json',dict(status='needs_attention',controller_pid=None))
            step=chain.recovery_step(folder,'impacts',folder/'manifest.json',8,'T2bW','mStop1000_mLSP800')
            self.assertIn('--polish-bounded-sgamma',step['command'])
            write_json(folder/'impact_status.json',dict(status='fits_complete'))
            step=chain.recovery_step(folder,'impacts',folder/'manifest.json',8,'T2bW','mStop1000_mLSP800')
            self.assertEqual(step['phase'],'plot')
            self.assertIn('--plot-only',step['command'])

    def test_failed_fit_recovery_keeps_completed_inputs_out_of_plan(self):
        plan=runner.failed_fit_plan(ROOT/'stats/combined_systematics_sr118')
        commands=[s['command'] for s in plan.values()]
        self.assertEqual(sum('--complete-bounded-sgamma' in c for c in commands),5)
        self.assertEqual(sum('--recover-remaining' in c for c in commands),1)
        self.assertEqual(sum('--recover-failed' in c for c in commands),1)
        self.assertFalse(any('build_' in Path(c[2]).name for c in commands))
        self.assertTrue(plan['contours']['partial'])

    def test_parallel_refresh_owner_is_exact_canonical_controller(self):
        campaign=ROOT/'campaigns/nonjme_20260927'
        owner=dict(pid=123,start_ticks=456,arguments=['python3','-u',
            str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign),
            '--refresh-year','2025','--workers','88'])
        with patch.object(chain,'read_json',return_value=dict(pid=123,status='payload_refresh_running')), \
             patch('TROTASR.workflows.normalization_recovery.process_identity',return_value=owner):
            self.assertEqual(chain.refresh_component_owner(campaign),owner)
            with self.assertRaises(ValueError):
                chain.refresh_component_owner(ROOT/'campaigns/systematics_20260924')

    def test_parallel_refresh_does_not_accept_disappeared_incomplete_owner(self):
        with patch.object(chain,'read_json',return_value=dict(pid=123,status='payload_refresh_running')), \
             patch('TROTASR.workflows.normalization_recovery.process_identity',return_value=None):
            with self.assertRaises(RuntimeError):chain.refresh_component_owner(ROOT/'campaigns/nonjme_20260927')

    def test_parallel_refresh_requires_worker_exits_and_exact_merge(self):
        from TROTASR.workflows.run_systematic_campaign import REFRESH_ID
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            c=Path(tmp); product=c/'merged/systematics_2025.json.gz'
            write_json(product,dict(test=True))
            summary=dict(status='complete',payload_revision=REFRESH_ID,sha256=sha256(product),
                coverage=dict(full_input_inventory=True,failed_files=0,completed_files=1,expected_files=1))
            write_json(c/'merged/systematics_2025.summary.json',summary)
            write_json(c/'state.json',dict(status='histograms_complete',pid=None,active=0,pending=0,
                failed=[],payload_revision=REFRESH_ID,refreshed_files=1))
            receipt=dict(status='complete',exit_code=0,payload_revision=REFRESH_ID)
            write_json(c/'state/one.json',receipt)
            plan=dict(payload_refresh=dict(tasks=[dict(key='one',refresh_required=True)],validation=dict(status='passed')))
            with patch('TROTASR.workflows.run_systematic_campaign.refresh_plan',return_value=plan):
                result=chain.verify_refreshed_component(c)
                self.assertIsNone(result['exit_code'])
                self.assertEqual(result['worker_exit_zero_receipts'],1)
                write_json(c/'state/one.json',dict(receipt,exit_code=1))
                with self.assertRaises(ValueError):chain.verify_refreshed_component(c)
                write_json(c/'state/one.json',receipt)
                write_json(product,dict(changed=True))
                with self.assertRaises(ValueError):chain.verify_refreshed_component(c)

    def test_combined_model_uses_canonical_measures_and_gated_fits(self):
        plan = runner.stage_plan(ROOT/'campaigns/nonjme_20260927', ROOT/'stats/combined_systematics_sr118',
            ROOT/'jsons/combined_systematics_config.json')
        self.assertEqual(plan['grid']['dependencies'], ['measure_2024','measure_2025'])
        self.assertIn('--preserve-unsupported-templates', plan['grid']['command'])
        for year in ('2024','2025'):
            command = plan['measure_'+year]['command']
            self.assertTrue(command[2].endswith('/workflows/build_background_estimation.py'))
            self.assertNotIn('/validation/', ' '.join(command))
        for name in ('limits','impacts'):
            self.assertEqual(plan[name]['dependencies'], ['grid'])
        self.assertEqual(runner.disposition('impact_recovery', plan['impact_recovery'],
            {'impacts': {'status':'failed'}}), 'blocked')
        self.assertEqual(runner.disposition('contours', plan['contours'], {}), 'wait')
        self.assertIn('--complete-bounded-sgamma',plan['impact_recovery']['command'])
        self.assertIn('--recover-unfinished',plan['limit_recovery']['command'])
        self.assertEqual(runner.disposition('limit_recovery',plan['limit_recovery'],
            {'limits':{'status':'succeeded'}}),'not_needed')
        self.assertEqual(runner.disposition('contours',plan['contours'],
            {'limits':{'status':'succeeded'},'limit_recovery':{'status':'not_needed'}}),'ready')
        self.assertEqual(runner.disposition('contours',plan['contours'],
            {'limits':{'status':'failed'},'limit_recovery':{'status':'failed'}}),'ready')
        for year in ('2024','2025'):
            self.assertNotIn('execution_blocked_reason',plan['distributions_'+year])
            self.assertIn('partial_generated_pending_visual_QA',plan['transfer_factors_'+year]['statuses'])
            self.assertIn('--allow-partial',plan['transfer_factors_'+year]['command'])
            self.assertIn('--manifest',plan['gnn_controls_'+year]['command'])
            self.assertEqual(plan['gnn_controls_'+year]['dependencies'], ['grid'])

    def test_jme_reuses_existing_five_fit_entrypoints_with_bounded_workers(self):
        plan=runner.stage_plan(ROOT/'campaigns/systematics_20260924',ROOT/'stats/jme_weights_20260928_sr118',ROOT/'stats/jme_weights_20260928_config.json')
        self.assertEqual(set(plan),{'grid','limits','impacts','cronly','impact_recovery'})
        for name in ('limits','impacts','impact_recovery'):
            cmd=plan[name]['command'];self.assertEqual(cmd[cmd.index('--workers')+1],'10')
        self.assertLessEqual(72+1+1+2+20+2,100)

    def setUp(self):
        self.plan = runner.stage_plan(ROOT/'campaigns/nominal_20260923',
            ROOT/'stats/nominal_20260923', ROOT/'stats/nominal_grid_config.json')

    def test_exact_internal_stage_plan_and_no_histogram_resubmission(self):
        self.assertEqual(len(self.plan), 17)
        for name, spec in self.plan.items():
            self.assertTrue(Path(spec['command'][2]).is_file())
            self.assertNotIn('nominal_campaign.py', spec['command'][2])
            self.assertNotIn('--allow-validation', spec['command'])
            self.assertNotIn('--allow-partial', spec['command'])
            self.assertNotIn('--missing-endpoint', spec['command'])
        for name in ('limits', 'impacts', 'impact_recovery'):
            cmd = self.plan[name]['command']
            self.assertEqual(cmd[cmd.index('--workers')+1], '47')
        self.assertEqual(1+2+2*runner.FIT_WORKERS+2, 99)
        self.assertTrue(self.plan['cronly']['receipt'].endswith('cronly/state.json'))
        self.assertTrue(self.plan['impact_recovery']['receipt'].endswith('impacts/direct_profile/status.json'))

    def test_grid_failure_blocks_only_dependent_stages(self):
        state = {'grid': {'status': 'failed'}}
        for name in ('limits', 'impacts', 'cronly', 'predictions'):
            self.assertEqual(runner.disposition(name, self.plan[name], state), 'blocked')
        for name in ('measurements_2024', 'transfer_factors_2025'):
            self.assertEqual(runner.disposition(name, self.plan[name], state), 'ready')
        self.assertEqual(runner.disposition('gnn_controls_2024', self.plan['gnn_controls_2024'], state), 'blocked')
        self.assertEqual(runner.disposition('contours', self.plan['contours'], state), 'wait')

    def test_explicit_resume_keeps_successes_without_rerunning_them(self):
        contract = dict(campaign='same', plan_sha256='same', grid_config_sha256='same', code_sha256='new',
                        stages={'done':dict(lane='aux'), 'grid':dict(lane='aux'), 'limits':dict(lane='fit')})
        old = dict(contract, code_sha256='old')
        previous = dict(status='needs_attention', pid=None, contract=old, stages={
            'done':dict(status='succeeded',exit_code=0), 'grid':dict(status='failed'),
            'limits':dict(status='blocked')})
        with patch.object(runner, 'verify_receipt', return_value=(True, {})) as check:
            kept = runner.resume_records(previous, contract, ROOT/'stats/test')
            self.assertEqual(set(kept), {'done'})
            self.assertTrue(kept['done']['reused_unchanged'])
            check.assert_called_once()
        for mutation in ('running', 'inputs', 'fit_failure', 'changed_success'):
            before, after = copy.deepcopy(previous), copy.deepcopy(contract)
            if mutation == 'running': before['pid'] = 123
            if mutation == 'inputs': after['grid_config_sha256'] = 'different'
            if mutation == 'fit_failure': before['stages']['limits']['status'] = 'failed'
            if mutation == 'changed_success': after['stages']['done']['command'] = ['different']
            with self.subTest(mutation=mutation), patch.object(runner, 'verify_receipt', return_value=(True, {})):
                with self.assertRaises((ValueError, RuntimeError)):
                    runner.resume_records(before, after, ROOT/'stats/test')

    def test_plot_resume_never_relaunches_failed_fits(self):
        stages={n:dict(lane=lane,command=['python','-u',script]) for n,lane,script in (
            ('limits','fit','combine_limits.py'),('grid','aux','build_nominal_grid.py'),
            ('distributions_2024','aux','plotting_nominal_distributions.py'))}
        contract=dict(campaign='same',plan_sha256='same',grid_config_sha256='same',stages=stages)
        previous=dict(status='needs_attention',pid=None,contract=contract,stages={
            'limits':dict(status='failed',pid=None),'grid':dict(status='succeeded',exit_code=0),
            'distributions_2024':dict(status='failed',pid=None)})
        with patch.object(runner,'verify_receipt',return_value=(True,{})):
            kept=runner.resume_records(previous,contract,ROOT/'stats/test',plots_only=True)
        self.assertEqual(set(kept),{'limits','grid'})
        self.assertEqual(kept['limits']['status'],'failed')
        self.assertTrue(kept['limits']['retained_for_finite_recovery'])

    def test_finite_recovery_only_for_actual_supported_failures(self):
        p = self.plan['impact_recovery']
        self.assertEqual(runner.disposition('impact_recovery', p, {}), 'wait')
        self.assertEqual(runner.disposition('impact_recovery', p, {'impacts':{'status':'succeeded'}}), 'not_needed')
        record = dict(status='failed', receipt_snapshot=dict(phase='finite_fits_exhausted',
            failed_nuisances=['qcd_norm_2024_Nb1']))
        self.assertEqual(runner.disposition('impact_recovery', p, {'impacts':record}), 'ready')
        for key, value in (('phase','initial_fit'), ('failed_nuisances',['CMS_unknown']), ('failed_nuisances',[])):
            bad = copy.deepcopy(record); bad['receipt_snapshot'][key] = value
            self.assertEqual(runner.disposition('impact_recovery', p, {'impacts':bad}), 'blocked')

    def test_plot_after_success_or_successful_recovery_only(self):
        p = self.plan['impacts_plot']
        self.assertEqual(runner.disposition('impacts_plot',p,{'impacts':{'status':'succeeded'}}),'ready')
        self.assertEqual(runner.disposition('impacts_plot',p,{'impacts':{'status':'failed'},
            'impact_recovery':{'status':'succeeded'}}),'ready')
        self.assertEqual(runner.disposition('impacts_plot',p,{'impacts':{'status':'failed'},
            'impact_recovery':{'status':'failed'}}),'blocked')

    def test_upstream_exit_and_full_scope_are_required(self):
        state = dict(status=runner.READY, stages={k:{'exit_code':0} for k in runner.STAGES})
        with patch.object(runner,'read_json',side_effect=[{'scope':'full_nominal_production'},state]):
            self.assertTrue(runner.upstream_gate(ROOT/'campaigns/test'))
        for bad in (dict(state,status='needs_attention'),dict(state,stages={}),
                    dict(state,stages={**state['stages'],'histograms':{'exit_code':1}})):
            with patch.object(runner,'read_json',side_effect=[{'scope':'full_nominal_production'},bad]):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.upstream_gate(ROOT/'campaigns/test')
        with patch.object(runner,'read_json',return_value={'scope':'representative_validation'}):
            with self.assertRaises(ValueError):runner.upstream_gate(ROOT/'campaigns/test')

    def test_receipt_status_exit_and_output_hash_all_required(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder = Path(tmp); product = folder/'plot.png'; product.write_bytes(b'unit-test-not-an-image')
            receipt = folder/'manifest.json'
            write_json(receipt,dict(status='generated_pending_visual_QA',files={'plot.png':sha256(product)}))
            spec = dict(receipt=str(receipt.relative_to(ROOT)), statuses=['generated_pending_visual_QA'])
            self.assertTrue(runner.verify_receipt(spec,0)[0])
            self.assertFalse(runner.verify_receipt(spec,2)[0])
            self.assertFalse(runner.verify_receipt(dict(spec,statuses=['complete']),0)[0])
            product.write_bytes(b'changed')
            with self.assertRaises(ValueError):runner.verify_receipt(spec,0)

    def diagnostic_fixture(self):
        row=dict(process_coverage=dict(numerator=True,denominator=True),
            valid=[True]*6+[False]*2, numerator=[1.]*6+[0.,0.],
            numerator_sumw2=[1.]*6+[0.,0.], denominator=[2.]*8,
            denominator_sumw2=[1.]*8, transfer_factor=[.5]*6+[None,None],
            mcstat=[.5]*6+[None,None], mechanical_relative_residual=[0.]*6+[None,None])
        return dict(schema='trotasr_transfer_factors_v2',status='unsupported_cells',
            issues=[dict(mode='highdm',route='qcd_qcdcr',category='Nb1',status='unsupported_cells')],
            highdm=dict(records=dict(qcd_qcdcr=dict(Nb1=row))))

    def test_only_reviewed_empty_sr_diagnostic_cells_can_be_deferred(self):
        source=self.diagnostic_fixture(); before=copy.deepcopy(source)
        self.assertEqual([x['native_bin_1based'] for x in runner.diagnostic_tf_gaps(source)],[7,8])
        self.assertEqual(source,before)
        for field,value in [('numerator',-1.),('numerator_sumw2',1.),
                ('denominator',0.),('denominator',float('nan')),('transfer_factor',1.)]:
            bad=copy.deepcopy(source);bad['highdm']['records']['qcd_qcdcr']['Nb1'][field][6]=value
            with self.assertRaises(ValueError):runner.diagnostic_tf_gaps(bad)
        bad=copy.deepcopy(source);bad['issues'][0]['mode']='lowdm'
        with self.assertRaises(ValueError):runner.diagnostic_tf_gaps(bad)

    def test_measurement_exit2_requires_independent_diagnostic_proof(self):
        state=dict(status='needs_attention',stages={k:dict(exit_code=2 if k.startswith('measure') else 0)
                                                  for k in runner.STAGES})
        with patch.object(runner,'read_json',side_effect=[dict(scope='full_nominal_production'),state]):
            self.assertTrue(runner.upstream_gate(ROOT/'campaigns/test'))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp); campaign=root/'campaigns/test';config={'years':{}}
            manifests={}
            for year in ('2024','2025'):
                hist=campaign/'merged'/('nominal_'+year+'.json.gz');write_json(hist,dict(test=True))
                folder=root/'estimations/test'/year
                stages={n:'complete' for n in ('sgamma','rz_high','rz_low','double_ratio')}
                stages['transfer_factors']='unsupported_cells'
                fields=dict(hists=str(hist.relative_to(root)))
                files={}
                for name in stages:
                    path=folder/(name+'.json.gz')
                    write_json(path,self.diagnostic_fixture() if name=='transfer_factors' else dict(status='complete'))
                    files[path.name]=sha256(path)
                    fields[name]=str(path.relative_to(root))
                config['years'][year]=fields
                manifest=dict(status='blocked',stages=stages,files=files,
                    provenance=dict(hist_input_sha256=sha256(hist),campaign_year=year,code={}))
                manifests[year]=(folder/'measurement_manifest.json',manifest)
                write_json(*manifests[year])
            cfg=root/'config.json';write_json(cfg,config)
            with patch.object(runner,'ROOT',root):
                result=runner.verify_measurements(campaign,cfg)
                self.assertEqual(len(result['2024']['diagnostic_tf_gaps']),2)
                path,manifest=manifests['2024'];manifest['stages']['rz_high']='failed';write_json(path,manifest)
                with self.assertRaises(ValueError):runner.verify_measurements(campaign,cfg)

    def test_early_diagnostic_proof_preserves_values_and_checks_covariance(self):
        import numpy as np
        from TROTASR.utils.transfer_factors import score_ratio
        from TROTASR.utils.background_estimation import finite_json
        source = self.diagnostic_fixture()
        row = source['highdm']['records']['qcd_qcdcr']['Nb1']
        for k in ('numerator', 'numerator_sumw2'): row[k][5] = 0.
        row['valid'][5] = False
        for k in ('transfer_factor', 'mcstat', 'mechanical_relative_residual'): row[k][5] = None
        self.assertEqual(len(runner.diagnostic_tf_gaps(source, '2022')), 3)
        source['year'] = '2023BPix'
        row2 = copy.deepcopy(self.diagnostic_fixture()['highdm']['records']['qcd_qcdcr']['Nb1'])
        row2['valid'][6] = True
        for k, v in [('numerator',1.), ('numerator_sumw2',1.), ('transfer_factor',.5),
                     ('mcstat',.5), ('mechanical_relative_residual',0.)]: row2[k][6] = v
        source['highdm']['records']['qcd_qcdcr']['Nb2plus'] = row2
        source['issues'].extend([
            dict(mode='highdm', route='qcd_qcdcr', category='Nb2plus', status='unsupported_cells'),
            dict(mode='lowdm', route='w_llcr', category='Nb2plus_NISR0', status='signed_mc_bins')])
        signed = finite_json(score_ratio(np.array([1.,2.,-.2,3.,4.]), np.ones(5), np.ones(5), np.ones(5)))
        signed['process_coverage'] = dict(numerator=True, denominator=True)
        source['lowdm'] = dict(records=dict(w_llcr=dict(Nb2plus_NISR0=signed)))
        before = copy.deepcopy(source)
        self.assertEqual(len(runner.diagnostic_tf_gaps(source, '2023BPix')), 5)
        self.assertEqual(source, before)
        signed['covariance'][0][1] += .01
        with self.assertRaisesRegex(ValueError, 'covariance mismatch'):
            runner.diagnostic_tf_gaps(source, '2023BPix')

    def test_only_empty_preflight_error_can_change_unstarted_stage_inventory(self):
        contract = dict(campaign='same', plan_sha256='same', grid_config_sha256='same', stages={'grid':{}})
        old = dict(contract, stages={})
        previous = dict(status='controller_error_children_retained', pid=None,
                        child_pids={}, stages={}, contract=old)
        self.assertEqual(runner.resume_records(previous, contract, ROOT/'stats/test'), {})
        for field, value in [('pid',123), ('child_pids',{'grid':123}), ('stages',{'grid':{'pid':None}})]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.resume_records(dict(previous, **{field:value}), contract, ROOT/'stats/test')

    def test_path_escapes_rejected(self):
        with self.assertRaises(ValueError):
            runner.stage_plan(Path('/tmp/external'),ROOT/'stats/test',ROOT/'stats/nominal_grid_config.json')


if __name__ == '__main__':unittest.main()
