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
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.nominal_campaign import THREAD_ENV


PARALLEL_REFRESH = 'parallel_refresh'


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


def run(campaign):
    campaign=internal_path(campaign)
    with (campaign/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read_json(campaign/'plan.json')
        if plan['scope']!='full_nominal_production':
            raise ValueError('The production chain requires a full-input campaign')
        calls=[('histograms',[str(ROOT/'workflows/nominal_campaign.py'),'run','--campaign',str(campaign)])]
        for year in (2024,2025):
            calls.append(('merge_'+str(year),[str(ROOT/'workflows/merge_nominal.py'),
                '--campaign',str(campaign),'--year',str(year)]))
        for year in (2024,2025):
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
    p.add_argument('--refresh-year',type=int,choices=[2025])
    p.add_argument('--workers',type=int,default=97)
    p.add_argument('--resume-admission',action='store_true',help='Explicitly resume a rejected refresh before any worker or frozen refresh plan existed')
    p.add_argument('--resume-refresh',action='store_true',help='Resume the inspected missing-seed failure without repeating completed outputs')
    p.add_argument('--resume-paused',action='store_true',help='Resume the user-paused SF update with reduced inference support and retained successes')
    p.add_argument('--resume-parallel',action='store_true',help='Adopt the two live refresh components after a waiting-parent-only handoff, then merge and finish products')
    a=p.parse_args()
    if a.resume_admission and not a.refresh_year:p.error('--resume-admission requires --refresh-year')
    if a.resume_refresh and (not a.refresh_year or a.resume_admission):
        p.error('--resume-refresh requires --refresh-year and cannot use --resume-admission')
    if a.resume_paused and (not a.refresh_year or a.resume_admission or a.resume_refresh):
        p.error('--resume-paused requires --refresh-year and no other resume flag')
    if a.resume_parallel and (not a.refresh_year or a.resume_admission or a.resume_refresh or a.resume_paused):
        p.error('--resume-parallel requires --refresh-year and no other resume flag')
    raise SystemExit(refresh_2025(a.campaign,a.workers,a.resume_admission,a.resume_refresh,a.resume_paused,a.resume_parallel) if a.refresh_year else run(a.campaign))
