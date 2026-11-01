"""Continue nominal histograms -> strict yearly merge -> new background measures.

Only implemented, validated stages are invoked. Plots/cards/fits are explicit
remaining stages, never fabricated completion markers.
"""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path, EARLY_ERAS
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.nominal_campaign import THREAD_ENV


PARALLEL_REFRESH = 'parallel_refresh'


def canonical_limit_step(folder, manifest, workers):
    """New six-era caller only; never reinterpret legacy recovery ownership."""
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.run_nominal_products import require_canonical_limit_completion
    if manifest != folder.parent/'grid/manifest.json':
        raise ValueError('Canonical limit continuation requires its unchanged grid path')
    receipt = folder/'workflow_state.json'
    command = [ROOT/'workflows/combine_limits.py', '--manifest', manifest,
               '--output', folder, '--plot-range-only', '--resume', '--workers', 115]
    if not receipt.exists():
        if (folder/'contract.json').exists() or (folder/'state.json').exists():
            return dict(action='done', status='failed',
                error='Legacy limit receipts retained unchanged; explicit canonical workflow handoff required')
        return dict(action='launch', phase='canonical_limits', receipt=str(receipt), command=command)
    saved = read_json(receipt)
    if saved.get('canonical_limits_pipeline') is not True:
        return dict(action='done', status='failed', error='Unrecognized canonical limit workflow receipt')
    if saved.get('pid'):
        identity = process_identity(saved['pid'])
        arguments = identity.get('arguments', []) if identity else []
        if not identity or any(str(item) not in arguments for item in
                (ROOT/'workflows/combine_limits.py', folder, manifest)):
            return dict(action='done', status='failed', error='Canonical limit controller identity requires inspection')
        return dict(action='wait', phase='canonical_limits', receipt=str(receipt), identity=identity)
    try:
        require_canonical_limit_completion(saved)
    except ValueError as error:
        return dict(action='done', status='failed', error=str(error), receipt=str(receipt))
    destination = folder.parent/'plots/contours'
    plot_receipt = destination/'plot_manifest.json'
    if plot_receipt.exists():
        plotted = read_json(plot_receipt)
        rows = plotted.get('plots', [])
        expected = {'T2tt':205, 'T2bW':345, 'T2tb':345}
        if (plotted.get('status') != 'generated_pending_visual_QA' or len(rows) != 3
                or {row.get('model'):row.get('plotted_points') for row in rows} != expected
                or any(row.get('plotted_grid_status') != 'complete' or row.get('plotted_missing') != []
                       for row in rows)):
            return dict(action='done', status='failed', receipt=str(plot_receipt),
                error='Existing contours do not cover all 895 displayed points; preserve for explicit handoff')
        return dict(action='done', status='succeeded', receipt=str(plot_receipt))
    return dict(action='launch', phase='plot', receipt=str(plot_receipt), command=[
        ROOT/'workflows/plotting_limit_contour.py', '--limits', folder, '--grid', manifest,
        '--output', destination, '--allow-partial'])


def recovery_step(folder, kind, manifest, workers, model=None, mass=None):
    """Select a finite saved-stage continuation, never rerun a terminal stage."""
    if kind == 'limits' and folder == ROOT/'stats/run3_systematics_20261025/limits':
        return canonical_limit_step(folder, manifest, workers)
    from TROTASR.workflows.normalization_recovery import process_identity
    if kind == 'limits':
        phases = [('numerical_all_unfinished_20260930', '--recover-unfinished'),
                  ('numerical_remaining_20261009', '--recover-remaining')]
        unresolved = False
    else:
        parent = read_json(folder/'impact_status.json')
        if parent['status'] == 'generated_pending_visual_QA':
            return dict(action='done', status='succeeded')
        if parent['status'] == 'fits_complete':
            return dict(action='launch', phase='plot', receipt=str(folder/'impact_status.json'),
                command=[ROOT/'workflows/combine_impacts.py', '--manifest', manifest,
                    '--output', folder, '--model', model, '--mass', mass, '--plot-only'])
        phases = [('bounded_sgamma_complete', '--complete-bounded-sgamma'),
                  ('bounded_sgamma_precision', '--polish-bounded-sgamma'),
                  ('bounded_sgamma_minimizer', '--hybrid-bounded-sgamma'),
                  ('bounded_sgamma_likelihood', '--likelihood-bounded-sgamma'),
                  ('bounded_sgamma_refined', '--refine-bounded-sgamma'),
                  ('bounded_sgamma_final', '--refine-bounded-sgamma')]
    for phase, option in phases:
        receipt = folder/phase/('state.json' if kind == 'limits' else 'status.json')
        if receipt.exists():
            saved = read_json(receipt)
            pid = saved.get('pid') or saved.get('controller_pid')
            if pid:
                identity = process_identity(pid)
                if not identity or str(folder) not in identity['arguments']:
                    return dict(action='done', status='failed', error='Recovery PID no longer owns '+str(receipt))
                return dict(action='wait', phase=phase, receipt=str(receipt), identity=identity)
            if saved['status'] in ('all_verified_pending_reconciliation', 'fits_complete'):
                if kind == 'limits':
                    unresolved = False
                    break
                return dict(action='done', status='failed', error='Completed recovery was not collected into impact_status')
            if saved['status'] != 'needs_attention':
                return dict(action='done', status='failed', error='Nonterminal receipt without live owner: '+str(receipt))
            if kind == 'limits':
                unresolved = True
                if any(r['status'] == 'needs_attention' for r in saved['results'].values()):
                    return dict(action='done', status='failed', error='Limit recovery has an execution error; inspect saved results')
            continue
        if (receipt.parent/'contract.json').exists():
            return dict(action='done', status='failed', error='Interrupted contract has no terminal receipt: '+str(receipt))
        script = 'reconcile_limit_fits.py' if kind == 'limits' else 'recover_impacts.py'
        command = [ROOT/'workflows'/script, '--manifest', manifest,
                   '--output' if kind == 'limits' else '--work', folder, option, '--workers', workers]
        if kind != 'limits':
            command += ['--model', model, '--mass', mass]
        return dict(action='launch', phase=phase, receipt=str(receipt), command=command)
    if kind != 'limits':
        return dict(action='done', status='failed', error='Finite impact recovery policies exhausted')
    destination = folder.parent/'plots/contours_recovery_20261021'
    receipt = destination/'plot_manifest.json'
    if receipt.exists():
        return dict(action='done', status='failed' if unresolved else 'succeeded',
            receipt=str(receipt), remaining_limit_failures=unresolved)
    return dict(action='launch', phase='plot', receipt=str(receipt), command=[
        ROOT/'workflows/plotting_limit_contour.py', '--limits', folder, '--grid', manifest,
        '--output', destination, '--allow-partial'])


def continue_failed_products(campaign, workers):
    """Adopt active recoveries and finish only failed fits and their plots."""
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.diagnostic_common import live_resources
    campaign = internal_path(campaign)
    if campaign != ROOT/'campaigns/nonjme_20260927' or not 10 <= workers <= 96:
        raise ValueError('Current combined campaign and aggregate budget required')
    output = ROOT/'stats/combined_systematics_sr118'
    manifest = output/'grid/manifest.json'
    with (campaign/'chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        chain_path = campaign/'chain_state.json'
        chain = read_json(chain_path)
        if chain.get('pid') and process_identity(chain['pid']):
            raise RuntimeError('A chain controller is already alive')
        path = output/'fit_recovery_state.json'
        old = read_json(path) if path.exists() else {}
        if old.get('pid') and process_identity(old['pid']):
            raise RuntimeError('A fit recovery controller is already alive')
        history = output/'history'/('continuation_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime()))
        history.mkdir(parents=True, exist_ok=False)
        write_json(history/'chain_state.json', chain)
        if old: write_json(history/'fit_recovery_state.json', old)
        impact_workers = min(8, max(1, (workers-24)//5))
        lanes = {'limits': dict(folder=output/'limits',kind='limits',workers=min(16,workers//3))}
        for model,mass,folder in [('T2tt','mStop1200_mLSP500',output/'impacts')]+[
                (m,p,output/'impacts_benchmarks_20261001'/(m+'_'+p))
                for m in ('T2bW','T2tb') for p in ('mStop1200_mLSP500','mStop1000_mLSP800')]:
            lanes[model+'_'+mass] = dict(folder=folder,kind='impacts',workers=impact_workers,model=model,mass=mass)
        state = dict(status='running',pid=os.getpid(),started=time.time(),stages={},commands={},
            source_sha256=sha256(Path(__file__)),manifest_sha256=sha256(manifest),worker_budget=workers,
            physics_changed=False,intermediate_root_rebuilt=False,histograms_rebuilt=False,
            original_history=str(history.relative_to(ROOT)),full_workflow_complete=False)
        chain.update(status='running',pid=os.getpid(),identity=process_identity(os.getpid()),
            stage='downstream_recovery',recovery_state=str(path.relative_to(ROOT)))
        write_json(chain_path,chain)
        active = {}
        def save():
            state['updated']=time.time();write_json(path,state)
        try:
            save()
            while True:
                for lane,(proc,stream,key) in list(active.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    stream.close();del active[lane]
                    state['commands'][key].update(pid=None,exit_code=rc,finished=time.time())
                for lane,spec in lanes.items():
                    if lane in active:continue
                    step=recovery_step(manifest=manifest,**spec)
                    state['stages'][lane]={k:v for k,v in step.items() if k!='command'}
                    if step['action']!='launch':continue
                    key=lane+'_'+step['phase']
                    if key in state['commands']:
                        state['stages'][lane]=dict(action='done',status='failed',
                            error='Command ended without the required receipt',command_record=key)
                        continue
                    with (ROOT/'stats/fit_admission.lock').open('a') as admission:
                        fcntl.flock(admission,fcntl.LOCK_EX)
                        slots,memory=live_resources()
                        if slots+2>100 or memory<36*1024**3:continue
                        log=output/'logs'/('continuation_20261021_'+key+'.log')
                        stream=log.open('x')
                        command=[sys.executable,'-u',*map(str,step['command'])]
                        proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**THREAD_ENV),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        active[lane]=(proc,stream,key)
                        state['commands'][key]=dict(command=command,pid=proc.pid,
                            identity=process_identity(proc.pid),log=str(log.relative_to(ROOT)),started=time.time())
                        state['stages'][lane].update(action='wait',pid=proc.pid)
                plots=read_json(output/'products_state.json')
                state['plot_controller_pid']=plots.get('pid')
                save()
                if not active and not plots.get('pid') and all(s['action']=='done' for s in state['stages'].values()):
                    break
                time.sleep(5)
            failed=[k for k,v in state['stages'].items() if v['status']!='succeeded']
            # Failed historical fit entries stay in products_state; the current
            # recovery receipt, not a rewrite of those entries, resolves them.
            plot_failures=[k for k,v in plots['stages'].items() if v['status']!='succeeded'
                and (k.startswith(('distributions_','gnn_controls_')) or k=='predictions')]
            failed += plot_failures
            state.update(status='needs_attention' if failed else 'generated_pending_visual_QA',failed_stages=failed)
            chain.update(status=state['status'],downstream_recovery_result=state['status'],
                remaining_failures=failed)
            return 2 if failed else 0
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error),child_pids={k:v[0].pid for k,v in active.items()})
            chain.update(status='needs_attention',error=repr(error));raise
        finally:
            state.update(pid=None,finished=time.time());save()
            chain.update(pid=None,finished=time.time());write_json(chain_path,chain)


def update_central_years(campaign, years, workers, resume=False):
    """Continue the two-year correction update through native products."""
    from TROTASR.workflows.run_systematic_campaign import UPDATE_ID, preserve_before_refresh
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
    campaign = internal_path(campaign)
    if campaign != ROOT/'campaigns/nonjme_20260927' or sorted(years) != [2024,2025]:
        raise ValueError('Use the current combined 2024/2025 central campaign')
    if not 10 <= workers <= 96:
        raise ValueError('Reserve the chain and both histogram controllers within 100 slots')
    output = ROOT/'stats/combined_systematics_sr118'
    config = ROOT/'jsons/combined_systematics_config.json'
    if read_json(config)['systematic_combination'] != COMBINATION_POLICY:
        raise ValueError('Adopted systematic combination changed')
    with (campaign/'chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = campaign/'chain_state.json'
        old = read_json(path) if path.exists() else {}
        if old.get('pid') and process_identity(old['pid']):
            raise RuntimeError('Existing chain is still running')
        if resume:
            if (old.get('payload_revision')!=UPDATE_ID or set(old.get('stages',{}))!={'jme','nonjme'}
                    or old.get('config_sha256')!=sha256(config) or old.get('years')!=years):
                raise ValueError('Resume requires the inspected waiting two-component chain')
            preserve_before_refresh(campaign,path,update_id=UPDATE_ID,
                revision='downstream_scope_handoff_%s_%s'%(old.get('pid'), workers))
            state=dict(old, status='running',pid=os.getpid(),identity=process_identity(os.getpid()),
                source_sha256=sha256(Path(__file__)),resumed=time.time(),
                previous_chain_identity=old.get('identity'),worker_budget=workers)
        else:
            if old.get('payload_revision') == UPDATE_ID:
                raise RuntimeError('Existing update retained; use an inspected explicit continuation')
            preserve_before_refresh(campaign, path, update_id=UPDATE_ID)
            state = dict(status='running', payload_revision=UPDATE_ID, years=years,
                pid=os.getpid(), identity=process_identity(os.getpid()), started=time.time(),
                config_sha256=sha256(config), source_sha256=sha256(Path(__file__)),
                stages={}, full_workflow_complete=False, intermediate_root_rebuilt=False)
        active = {}
        def save():
            state['updated']=time.time(); write_json(path,state)
        def launch(name, command):
            logfile=campaign/'logs'/('chain_'+UPDATE_ID+'_'+name+'.log')
            stream=logfile.open('x')
            proc=subprocess.Popen([sys.executable,'-u',*map(str,command)],cwd=ROOT,
                env=dict(os.environ,**THREAD_ENV),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            active[name]=(proc,stream)
            state['stages'][name]=dict(status='running',pid=proc.pid,identity=process_identity(proc.pid),
                command=list(map(str,command)),log=str(logfile.relative_to(ROOT)),started=time.time())
            save()
        def wait_children():
            while active:
                for name,(proc,stream) in list(active.items()):
                    if proc is None:
                        record=state['stages'][name]
                        if process_identity(record['identity']['pid'])==record['identity']:continue
                        component=ROOT/'campaigns'/('systematics_20260924' if name=='jme' else 'nonjme_20260927')
                        receipt=read_json(component/'state.json')
                        rc=0 if (receipt.get('status')=='histograms_complete'
                            and receipt.get('payload_revision')==UPDATE_ID
                            and not receipt.get('active') and not receipt.get('failed')
                            and not receipt.get('pending')) else 2
                        if rc==0:
                            for year in years:
                                output=component/'merged'/('systematics_'+str(year)+'.json.gz')
                                summary=read_json(output.with_name('systematics_'+str(year)+'.summary.json'))
                                if (summary.get('status')!='complete' or summary.get('payload_revision')!=UPDATE_ID
                                        or summary.get('sha256')!=sha256(output)):
                                    raise ValueError('Adopted component merge is incomplete')
                        record.update(controller_exit_observation='not_observed_after_parent_handoff')
                    else:
                        rc=proc.poll()
                        if rc is None:continue
                        stream.close()
                    del active[name]
                    state['stages'][name].update(status='succeeded' if rc==0 else 'failed',
                        pid=None,exit_code=rc if proc is not None else None,result_code=rc,finished=time.time());save()
                if active:time.sleep(5)
            return any(s.get('result_code',s.get('exit_code',0)) for s in state['stages'].values())
        try:
            save()
            nonjme_workers=max(4,workers//4)
            for name,component,count in (
                ('jme',ROOT/'campaigns/systematics_20260924',workers-nonjme_workers),
                ('nonjme',campaign,nonjme_workers)):
                if resume:
                    record=state['stages'][name]
                    if record['status']=='running':active[name]=(None,None)
                    elif record['status']!='succeeded':
                        raise RuntimeError('A component needs its own scoped recovery: '+name)
                else:
                    launch(name,[ROOT/'workflows/run_systematic_campaign.py','--campaign',component,
                        '--update-years',*years,'--workers',count])
            if wait_children():
                state['status']='needs_attention';return 2
            for year in years:
                for suffix in ('.json.gz','.summary.json'):
                    preserve_before_refresh(campaign,campaign/'combined/merged'/('systematics_'+str(year)+suffix),
                        move=True,update_id=UPDATE_ID)
                launch('combine_'+str(year),[ROOT/'workflows/merge_nominal.py','--campaign',
                    ROOT/'campaigns/systematics_20260924','--combine-with',campaign,'--year',year,
                    '--config',config,'--output',campaign/'combined/merged'/('systematics_'+str(year)+'.json.gz')])
                if wait_children():
                    state['status']='needs_attention';return 2
            # Old results remain recoverable in the existing central directory's
            # history. New results use exactly the same canonical output paths.
            with contextlib.ExitStack() as locks:
                for p in (output/'products.lock',output/'limits/controller.lock',
                          output/'limits/numerical_controller.lock',output/'impacts/impact.lock',
                          output/'cronly/cronly.lock'):
                    if p.exists():
                        stream=locks.enter_context(p.open('r'))
                        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                    try:
                        if proc.stat().st_uid != os.getuid():continue
                        if any(str(output).encode() in a for a in (proc/'cmdline').read_bytes().split(b'\0')):
                            raise RuntimeError('Existing downstream process retained: '+proc.name)
                    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
                for name in ('grid','limits','impacts','cronly','logs','products_state.json',
                             'fit_recovery_state.json','impacts_benchmarks_20261001','plots'):
                    p=output/name
                    preserve_before_refresh(output,p,move=True,update_id=UPDATE_ID)
                for year in years:
                    preserve_before_refresh(campaign,campaign/'combined/measurements'/str(year),
                        move=True,update_id=UPDATE_ID)
            launch('products',[ROOT/'workflows/run_nominal_products.py','--campaign',campaign,
                '--output',output,'--config',config,'--worker-budget',workers])
            wait_children()
            if state['stages']['products']['exit_code'] == 2:
                launch('remaining_fits',[ROOT/'workflows/run_nominal_products.py','--campaign',campaign,
                    '--output',output,'--config',config,'--recover-failed-fits','--worker-budget',workers])
                wait_children()
            unresolved=(state['stages'].get('remaining_fits',state['stages']['products'])['exit_code']!=0)
            state['status']='needs_attention' if unresolved else 'generated_pending_visual_QA_and_inventory'
            return 2 if unresolved else 0
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error),
                child_pids={n:state['stages'][n]['pid'] for n in active})
            raise
        finally:
            state.update(pid=None,finished=time.time());save()


def refresh_component_owner(campaign):
    """Identify an already-running canonical refresh without submitting work."""
    from TROTASR.workflows.normalization_recovery import process_identity
    state = read_json(campaign/'state.json')
    pid = state.get('pid')
    identity = process_identity(pid) if isinstance(pid, int) and pid > 1 else None
    if identity:
        argv = identity['arguments']
        required = [str(ROOT/'workflows/run_systematic_campaign.py'), '--campaign',
                    str(campaign), '--refresh-year', '2025']
        if not any(argv[i:i+len(required)] == required for i in range(len(argv))):
            raise ValueError('Different live refresh owner: '+str(campaign))
    elif state.get('status') != 'histograms_complete':
        raise RuntimeError('Refresh is incomplete without its live controller: '+str(campaign))
    return identity


def verify_refreshed_component(campaign):
    """Adopt saved worker/merge evidence, never invent an orphan's exit code."""
    from TROTASR.workflows.run_systematic_campaign import refresh_plan, refresh_task_required, REFRESH_ID
    plan = refresh_plan(campaign); spec = plan['payload_refresh']
    state = read_json(campaign/'state.json')
    tasks = [t for t in spec['tasks'] if refresh_task_required(t)]
    if (state.get('status') != 'histograms_complete' or state.get('pid')
            or state.get('active') or state.get('pending') or state.get('failed')
            or state.get('payload_revision') != REFRESH_ID
            or spec['validation'].get('status') != 'passed'
            or state.get('refreshed_files') != len(tasks)):
        raise ValueError('Existing refresh has not completed: '+str(campaign))
    output = campaign/'merged/systematics_2025.json.gz'
    summary = read_json(output.with_name('systematics_2025.summary.json'))
    coverage = summary.get('coverage', {})
    if (summary.get('status') != 'complete' or summary.get('payload_revision') != REFRESH_ID
            or summary.get('sha256') != sha256(output)
            or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
            or coverage.get('completed_files') != len(spec['tasks'])
            or coverage.get('expected_files') != len(spec['tasks'])):
        raise ValueError('Incomplete refreshed merge: '+str(output))
    for task in tasks:
        receipt = read_json(campaign/'state'/(task['key']+'.json'))
        if (receipt.get('status') != 'complete' or receipt.get('exit_code') != 0
                or receipt.get('payload_revision') != REFRESH_ID):
            raise ValueError('Missing successful refresh worker receipt: '+task['key'])
    return dict(status='reused_verified', exit_code=None,
        controller_exit_observation='not_observed_after_parent_handoff',
        worker_exit_zero_receipts=len(tasks), merge_sha256=summary['sha256'],
        summary_sha256=sha256(output.with_name('systematics_2025.summary.json')),
        finished=time.time())


def refresh_2025(campaign, workers, resume_admission=False, resume_refresh=False, resume_paused=False,
                 resume_parallel=False):
    """Update existing central products; no alternate campaign or entrypoint."""
    from TROTASR.workflows.run_systematic_campaign import (REFRESH_ID, preserve_before_refresh,
        REFERENCE_REPAIR, repair_refresh_references, SUPPORT_REFRESH, resume_paused_refresh)
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
    campaign=internal_path(campaign)
    if campaign != ROOT/'campaigns/nonjme_20260927' or not 1<=workers<=97:
        raise ValueError('Current combined campaign only; reserve both controllers')
    combined=campaign/'combined';output=ROOT/'stats/combined_systematics_sr118'
    config=ROOT/'jsons/combined_systematics_config.json'
    if read_json(config)['systematic_combination'] != COMBINATION_POLICY:
        raise ValueError('Adopted systematic combination changed')
    with (campaign/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=campaign/'chain_state.json'
        if path.exists():
            old=read_json(path)
            if old.get('payload_revision')==REFRESH_ID:
                if resume_parallel:
                    if (old.get('status') != 'running' or old.get('stage') != 'refresh_jme'
                            or old.get('stages') or process_identity(old['identity']['pid']) == old['identity']):
                        raise RuntimeError('Parallel handoff requires the stopped waiting chain, not its workers')
                    if sha256(config) != old['config_sha256']:
                        raise ValueError('Parallel handoff configuration changed')
                    for name,digest in old['preserved_2024'].items():
                        if sha256(ROOT/name) != digest:raise ValueError('Preserved 2024 changed')
                    for name,digest in old['sources'].items():
                        if sha256(ROOT/name) == digest:continue
                        backup=campaign/'history'/REFRESH_ID/PARALLEL_REFRESH/'sources'/name
                        if (name != 'workflows/run_nominal_chain.py' or not backup.is_file()
                                or sha256(backup) != digest):
                            raise ValueError('Unregistered parallel handoff source: '+name)
                    for component in (ROOT/'campaigns/systematics_20260924', campaign):
                        refresh_component_owner(component)
                elif resume_paused:
                    if (old.get('status')!='paused_by_user' or old.get('child_pid') or old.get('pid')
                            or old.get('stage')!='refresh_jme' or old.get('stages')
                            or process_identity(old['identity']['pid'])==old['identity']):
                        raise RuntimeError('Require the inspected paused first-stage chain')
                    if sha256(config)!=old['config_sha256']:
                        raise ValueError('Paused chain physics configuration changed')
                    for name,digest in old['preserved_2024'].items():
                        if sha256(ROOT/name)!=digest: raise ValueError('Preserved 2024 changed')
                    for name,digest in old['sources'].items():
                        if sha256(ROOT/name)==digest:continue
                        backup=(ROOT/'campaigns/systematics_20260924/history'/REFRESH_ID/
                                SUPPORT_REFRESH/'sources'/name)
                        if not backup.is_file() or sha256(backup)!=digest:
                            raise ValueError('Missing paused-source handoff: '+name)
                    resume_paused_refresh(ROOT/'campaigns/systematics_20260924')
                elif (not (resume_admission or resume_refresh) or old.get('status')!='needs_attention' or old.get('child_pid')
                        or set(old.get('stages',{}))!={'refresh_jme'}
                        or old['stages']['refresh_jme']['exit_code']==0
                        or process_identity(old['identity']['pid'])==old['identity']):
                    raise RuntimeError('Existing refresh chain retained; inspect its exact stage before resuming')
                if resume_refresh and not resume_paused:
                    if sha256(config) != old['config_sha256']:
                        raise ValueError('Interrupted chain configuration changed')
                    for name,digest in old['preserved_2024'].items():
                        if sha256(ROOT/name) != digest:
                            raise ValueError('Interrupted chain 2024 product changed: '+name)
                    replacements={'workflows/run_nominal_chain.py','workflows/run_systematic_campaign.py',
                                  'workflows/merge_nominal.py'}
                    for name,digest in old['sources'].items():
                        if sha256(ROOT/name)==digest:continue
                        backup=(ROOT/'campaigns/systematics_20260924/history'/REFRESH_ID/
                                REFERENCE_REPAIR/'sources'/name)
                        if name not in replacements or not backup.is_file() or sha256(backup)!=digest:
                            raise ValueError('Unregistered chain source replacement: '+name)
                    repair_refresh_references(ROOT/'campaigns/systematics_20260924')
                elif not (resume_paused or resume_parallel) and any(read_json(folder/'plan.json').get('payload_refresh') for folder in
                         (campaign,ROOT/'campaigns/systematics_20260924')):
                    raise RuntimeError('Admission resume cannot alter a frozen running/failed refresh')
            elif resume_refresh or resume_parallel:
                raise RuntimeError('No interrupted payload refresh to resume')
            preserve_before_refresh(campaign,path,revision=PARALLEL_REFRESH if resume_parallel else SUPPORT_REFRESH if resume_paused else REFERENCE_REPAIR if resume_refresh else None)
        elif resume_refresh or resume_parallel:
            raise RuntimeError('No interrupted chain state')
        stages=[('refresh_jme',[str(ROOT/'workflows/run_systematic_campaign.py'),
            '--campaign',str(ROOT/'campaigns/systematics_20260924'),'--refresh-year','2025','--workers',str(workers)]),
            ('refresh_nonjme',[str(ROOT/'workflows/run_systematic_campaign.py'),
            '--campaign',str(campaign),'--refresh-year','2025','--workers',str(workers)]),
            ('combine_2025',[str(ROOT/'workflows/merge_nominal.py'),'--campaign',str(ROOT/'campaigns/systematics_20260924'),
            '--combine-with',str(campaign),'--year','2025','--config',str(config),
            '--output',str(combined/'merged/systematics_2025.json.gz')]),
            ('products',[str(ROOT/'workflows/run_nominal_products.py'),'--campaign',str(campaign),
            '--output',str(output),'--config',str(config),'--refresh-year','2025','--worker-budget',str(workers)])]
        sources={str(Path(c[0]).relative_to(ROOT)):sha256(Path(c[0])) for _,c in stages}
        sources[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
        state=dict(status='running',pid=os.getpid(),identity=process_identity(os.getpid()),
            payload_revision=REFRESH_ID,started=time.time(),stages={},sources=sources,
            config_sha256=sha256(config),preserved_2024={str(p.relative_to(ROOT)):sha256(p) for p in
            (combined/'merged/systematics_2024.json.gz',combined/'merged/systematics_2024.summary.json')},
            full_workflow_complete=False)
        if resume_parallel:
            state['parallel_refresh_owners']={str(c.relative_to(ROOT)):refresh_component_owner(c)
                for c in (ROOT/'campaigns/systematics_20260924',campaign)}
            state['handoff']='waiting_chain_only; original refresh controllers and workers retained'
        try:
            for stage,command in stages:
                for name,digest in sources.items():
                    if sha256(ROOT/name)!=digest:raise ValueError('Active chain source changed: '+name)
                if sha256(config)!=state['config_sha256']:raise ValueError('Active configuration changed')
                for name,digest in state['preserved_2024'].items():
                    if sha256(ROOT/name)!=digest:raise ValueError('Preserved 2024 input changed')
                if resume_parallel and stage in ('refresh_jme','refresh_nonjme'):
                    component=ROOT/'campaigns/systematics_20260924' if stage=='refresh_jme' else campaign
                    identity=state['parallel_refresh_owners'][str(component.relative_to(ROOT))]
                    state.update(stage=stage,child_pid=None,child_identity=identity,command=command)
                    while identity and process_identity(identity['pid']) == identity:
                        state['parallel_progress']={str(c.relative_to(ROOT)):read_json(c/'state.json')
                            for c in (ROOT/'campaigns/systematics_20260924',campaign)}
                        state['updated']=time.time();write_json(path,state);time.sleep(3)
                    state['stages'][stage]=verify_refreshed_component(component)
                    state.update(child_identity=None,updated=time.time());write_json(path,state)
                    continue
                if stage=='combine_2025':
                    for suffix in ('.json.gz','.summary.json'):
                        preserve_before_refresh(campaign,combined/'merged'/('systematics_2025'+suffix),move=True)
                if stage=='products':
                    # Hold all existing controller locks before moving an old
                    # model into recoverable history. Never drain/kill a worker.
                    with contextlib.ExitStack() as stack:
                        paths=[output/'products.lock',output/'limits/controller.lock',
                               output/'limits/numerical_controller.lock',output/'impacts/impact.lock',
                               output/'cronly/cronly.lock']
                        for p in paths:
                            if p.exists():
                                stream=stack.enter_context(p.open('r'))
                                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        for proc in Path('/proc').iterdir():
                            if not proc.name.isdigit():continue
                            try:
                                if proc.stat().st_uid!=os.getuid():continue
                                argv=(proc/'cmdline').read_bytes().split(b'\0')
                                if any(str(output).encode() in a for a in argv) and int(proc.name)!=os.getpid():
                                    raise RuntimeError('Current downstream worker retained: '+proc.name)
                            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
                        for name in ('grid','limits','impacts','cronly','logs','products_state.json',
                                     'impacts_benchmarks_20261001','plots/2025',
                                     'plots/predictions','plots/contours'):
                            p=output/name
                            if p.exists():preserve_before_refresh(output,p,move=True)
                        factors=combined/'measurements/2025'
                        if factors.exists():preserve_before_refresh(campaign,factors,move=True)
                logfile=campaign/'logs'/('chain_'+stage+'.log')
                if logfile.exists():preserve_before_refresh(campaign,logfile,move=True,
                    revision=SUPPORT_REFRESH if resume_paused else REFERENCE_REPAIR if resume_refresh else None)
                state.update(stage=stage,updated=time.time());write_json(path,state)
                with logfile.open('x') as stream:
                    child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,
                        env=dict(os.environ,**THREAD_ENV),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid,child_identity=process_identity(child.pid),command=command)
                    write_json(path,state);rc=child.wait()
                state['stages'][stage]=dict(exit_code=rc,finished=time.time(),log=str(logfile.relative_to(ROOT)))
                state.update(child_pid=None,child_identity=None);write_json(path,state)
                if rc:
                    state['status']='needs_attention';return rc
            state['status']='generated_pending_visual_QA_and_inventory'
            return 0
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error));raise
        finally:
            state.update(pid=None,finished=time.time());write_json(path,state)


def prepare_early(campaign, source_root, workers):
    """Reuse delivered ntuples, validate representatives, then register production."""
    from TROTASR.workflows.nominal_campaign import early_inventory, prepare, run as histograms, current_contract, validate_payload
    campaign = internal_path(campaign)
    if (campaign/'plan.json').exists():
        return 0
    inventory_path = ROOT/'inputs/manifests'/(campaign.name+'.json')
    inventory = early_inventory(source_root, EARLY_ERAS, inventory_path)
    chosen = {}
    for task in sorted(inventory['tasks'], key=lambda t:(t['entries'],t['key'])):
        if task['entries'] <= 0:
            continue
        for group in task['groups']:
            chosen.setdefault((task['year'],group),task)
    for era in EARLY_ERAS:
        if not all((era,'signal_'+top) in chosen for top in ('T2tt','T2bW','T2tb')):
            raise ValueError('Missing representative signal topology for '+era)
    selected = {t['key']:t for t in chosen.values()}
    validation = ROOT/'validation'/campaign.name
    manifest = dict(inventory, tasks=sorted(selected.values(),key=lambda t:t['key']))
    representative_path = inventory_path.with_name(campaign.name+'_representatives.json')
    write_json(representative_path,manifest)
    if not (validation/'plan.json').exists():
        prepare(representative_path,validation,min(workers,12),2000)
    rc = histograms(validation)
    if rc:
        return rc
    proofs = {}
    for task in manifest['tasks']:
        state = read_json(validation/'state'/(task['key']+'.json'))
        if state['status'] != 'complete':
            raise ValueError('Representative did not complete: '+task['key'])
        output = ROOT/state['output']
        payload = read_json(output)
        validate_payload(payload,task,current_contract(task['year']))
        if sha256(output) != state['output_sha256']:
            raise ValueError('Representative output changed')
        proofs[task['key']] = dict(output=state['output'],sha256=state['output_sha256'],groups=task['groups'])
        selected[task['key']]['seed_output'] = state['output']
    gate = validation/'gate.json'
    write_json(gate,dict(status='passed',three_signal_topologies_tested=True,
        contracts={e:current_contract(e) for e in EARLY_ERAS}, proofs=proofs))
    full = dict(inventory,scope='full_nominal_production',validation_gate=str(gate.relative_to(ROOT)),
        tasks=[selected.get(t['key'],t) for t in inventory['tasks']])
    full_path = inventory_path.with_name(campaign.name+'_full.json')
    write_json(full_path,full)
    prepare(full_path,campaign,workers,2000)
    return 0


def run(campaign):
    campaign=internal_path(campaign)
    with (campaign/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read_json(campaign/'plan.json')
        if plan['scope']!='full_nominal_production':
            raise ValueError('The production chain requires a full-input campaign')
        calls=[('histograms',[str(ROOT/'workflows/nominal_campaign.py'),'run','--campaign',str(campaign)])]
        for year in plan['contracts']:
            calls.append(('merge_'+str(year),[str(ROOT/'workflows/merge_nominal.py'),
                '--campaign',str(campaign),'--year',str(year)]))
        for year in plan['contracts']:
            calls.append(('measure_'+str(year),[str(ROOT/'workflows/build_background_estimation.py'),
                '--hists',str(campaign/'merged'/('nominal_%s.json.gz'%year)),
                '--output',str(ROOT/'estimations'/campaign.name/str(year))]))
        state=dict(status='running',pid=os.getpid(),started=time.time(),stages={},
            source_sha256=sha256(Path(__file__)),full_workflow_complete=False)
        for stage,command in calls:
            state['stage']=stage
            write_json(campaign/'chain_state.json',state)
            log=campaign/'logs'/('chain_'+stage+'.log')
            with log.open('a') as stream:
                p=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,
                    env=dict(os.environ,**THREAD_ENV),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
                state['child_pid']=p.pid
                write_json(campaign/'chain_state.json',state)
                rc=p.wait()
            state['stages'][stage]=dict(exit_code=rc,finished=time.time(),log=str(log.relative_to(ROOT)))
            if rc and not stage.startswith('measure_'):
                state.update(status='needs_attention',child_pid=None)
                write_json(campaign/'chain_state.json',state)
                return rc
        failed=any(v['exit_code'] for v in state['stages'].values())
        state.update(status='needs_attention' if failed else 'measurements_complete_pending_plots_cards_fits',
                     child_pid=None,finished=time.time())
        write_json(campaign/'chain_state.json',state)
        return 1 if failed else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--prepare-early-eras',type=Path,help='Delivered workflow root; validate and reuse early-era inputs before nominal production')
    p.add_argument('--products-config',type=Path,help='Continue verified nominal measurements to plots, cards and expected limits')
    p.add_argument('--products-output',type=Path,help='New nominal downstream output directory')
    p.add_argument('--refresh-year',type=int,choices=[2025])
    p.add_argument('--update-years',type=int,nargs='+',choices=[2024,2025])
    p.add_argument('--resume-update',action='store_true',help='Adopt an inspected waiting update without resubmitting histogram workers')
    p.add_argument('--workers',type=int,default=97)
    p.add_argument('--continue-failed-products',action='store_true',help='Adopt active downstream recoveries and finish finite fits/plots without rebuilding inputs')
    p.add_argument('--resume-admission',action='store_true',help='Explicitly resume a rejected refresh before any worker or frozen refresh plan existed')
    p.add_argument('--resume-refresh',action='store_true',help='Resume the inspected missing-seed failure without repeating completed outputs')
    p.add_argument('--resume-paused',action='store_true',help='Resume the user-paused SF update with reduced inference support and retained successes')
    p.add_argument('--resume-parallel',action='store_true',help='Adopt the two live refresh components after a waiting-parent-only handoff, then merge and finish products')
    a=p.parse_args()
    if bool(a.products_config) != bool(a.products_output):
        p.error('--products-config and --products-output must be provided together')
    if a.products_config:
        config=read_json(internal_path(a.products_config))
        if (not a.prepare_early_eras or tuple(config['years']) != EARLY_ERAS
                or config.get('systematic_evaluation_deferred') is not True):
            p.error('Early nominal continuation requires four target eras and systematic evaluation deferred')
    if a.prepare_early_eras:
        rc=prepare_early(a.campaign,a.prepare_early_eras,min(a.workers,96))
        if rc:
            raise SystemExit(rc)
    if a.continue_failed_products:
        raise SystemExit(continue_failed_products(a.campaign,a.workers))
    if a.update_years:
        if a.refresh_year or a.resume_admission or a.resume_refresh or a.resume_paused or a.resume_parallel:
            p.error('--update-years starts a new revision, not an old refresh resume')
        raise SystemExit(update_central_years(a.campaign,a.update_years,a.workers,a.resume_update))
    if a.resume_update:p.error('--resume-update requires --update-years')
    if a.resume_admission and not a.refresh_year:p.error('--resume-admission requires --refresh-year')
    if a.resume_refresh and (not a.refresh_year or a.resume_admission):
        p.error('--resume-refresh requires --refresh-year and cannot use --resume-admission')
    if a.resume_paused and (not a.refresh_year or a.resume_admission or a.resume_refresh):
        p.error('--resume-paused requires --refresh-year and no other resume flag')
    if a.resume_parallel and (not a.refresh_year or a.resume_admission or a.resume_refresh or a.resume_paused):
        p.error('--resume-parallel requires --refresh-year and no other resume flag')
    rc=refresh_2025(a.campaign,a.workers,a.resume_admission,a.resume_refresh,a.resume_paused,a.resume_parallel) if a.refresh_year else run(a.campaign)
    if a.products_config:
        from TROTASR.workflows.run_nominal_products import run as products
        state=read_json(internal_path(a.campaign)/'chain_state.json')
        if state.get('stages', {}).get('histograms', {}).get('exit_code') == 0:
            # The downstream entrypoint verifies each merge and measurement;
            # a diagnostic display failure cannot fabricate a fit-ready model.
            rc=products(a.campaign,a.products_output,a.products_config,worker_budget=min(a.workers,96))
    raise SystemExit(rc)
