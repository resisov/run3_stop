"""Execute the validated new central/weights/JME stage, then strict year merges.

This does not run an old fit graph or claim the remaining object systematics,
background propagation, cards, fits or plots are finished.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.nominal_campaign import run, file_worker
from TROTASR.workflows.merge_nominal import merge


REFRESH_ID = 'payload_update_20261005'
REFERENCE_REPAIR = 'reference_repair'
SUPPORT_REFRESH = 'support_refresh'
MISSING_REFRESH_SEEDS = frozenset({
    '2025_data_shard_01052', '2025_mc_shard_00000', '2025_mc_shard_03059',
    '2025_signal_shard_00077', '2025_signal_shard_00294', '2025_signal_shard_00568',
})


def refresh_required(task):
    match=re.fullmatch(r'2025_(data|mc|signal)_shard_[0-9]+',task['key'])
    if not match or task['year']!=2025:
        raise ValueError('Unrecognized 2025 data/MC/signal input identity')
    if Path(task['input']).name!=task['key'].removeprefix('2025_')+'.root':
        raise ValueError('Shard key/input identity mismatch')
    return match.group(1)!='data'


def preserve_before_refresh(campaign, path, move=False, revision=None):
    """Recoverable history only; the active product keeps its canonical path."""
    campaign, path = map(internal_path, (campaign, path))
    base = campaign/'history'/REFRESH_ID
    target = (base/revision if revision else base)/path.relative_to(campaign)
    if not path.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if path.is_file() and sha256(path) == sha256(target):
            if move:
                # The verified recoverable copy already exists.
                path.unlink()
            return target
        raise FileExistsError('Different existing refresh backup: '+str(target))
    if move:
        path.rename(target)
    elif path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    return target


def refresh_output_record(campaign, record):
    """Resolve surviving outputs by exact hash; never follow retired directories."""
    campaign = internal_path(campaign)
    original = internal_path(ROOT/record['output'])
    canonical = campaign/'outputs'/(record['key']+'.json.gz')
    candidates = [canonical, campaign/'history'/REFRESH_ID/'outputs'/canonical.name]
    if campaign in original.parents:
        candidates.insert(0, original)
    for path in dict.fromkeys(candidates):
        if path.is_file() and sha256(path) == record['sha256']:
            return dict(record, output=str(path.relative_to(ROOT)))
    raise FileNotFoundError('No hash-matched retained output: '+record['key'])


def refresh_task_required(task):
    return task['refresh_required'] or task.get('rebuild_missing_seed', False)


def refreshed_population_verified(task, state):
    if state.get('previous_populations_identical') is True:
        return True
    return (task.get('rebuild_missing_seed') is True
            and task['key'] in MISSING_REFRESH_SEEDS
            and state.get('previous_populations_identical') is None
            and state.get('population_validation') == 'retained_input_contract_old_seed_unavailable')


def repair_refresh_references(campaign):
    """One explicit stopped-controller handoff; retain all valid completed work."""
    from TROTASR.workflows.nominal_campaign import current_contract, verify_compiled, fingerprint, relocated_seed
    from TROTASR.workflows.normalization_recovery import process_identity
    campaign = internal_path(campaign)
    if campaign != ROOT/'campaigns/systematics_20260924':
        raise ValueError('Reference repair is limited to the interrupted current JME refresh')
    with (campaign/'controller.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = read_json(campaign/'plan.json'); spec = plan['payload_refresh']
        if spec.get(REFERENCE_REPAIR):
            return refresh_plan(campaign)
        state = read_json(campaign/'state.json')
        if state.get('active') or state.get('status') != 'payload_refresh_needs_attention':
            raise RuntimeError('Reference repair requires a stopped, failed refresh')
        verify_compiled()
        if current_contract(2025, plan['shape_mode'], spec['variations']) != spec['contract']:
            raise ValueError('Physics/compiled/input contract must remain unchanged')
        changed = {'workflows/run_systematic_campaign.py', 'workflows/nominal_campaign.py',
                   'workflows/merge_nominal.py'}
        for name, digest in spec['sources'].items():
            if sha256(ROOT/name) == digest:
                continue
            backup = campaign/'history'/REFRESH_ID/REFERENCE_REPAIR/'sources'/name
            if name not in changed or not backup.is_file() or sha256(backup) != digest:
                raise ValueError('Unregistered source replacement: '+name)
        tasks = {t['key']:t for t in spec['tasks']}
        seeds = {t['key']:t.get('seed_output') for t in plan['tasks']}
        missing = []; completed = {}; requeue = []
        for key, task in tasks.items():
            canonical = relocated_seed(task)
            task.update(canonical); task.pop('seed_output', None)
            records = []
            for record in spec['previous_outputs'][key]:
                try:
                    records.append(refresh_output_record(campaign, record))
                except FileNotFoundError:
                    if (key not in MISSING_REFRESH_SEEDS or record['output'] != seeds[key]
                            or len(spec['previous_outputs'][key]) != 1):
                        raise
                    if (campaign/'outputs'/(key+'.json.gz')).exists():
                        raise ValueError('Do not replace a divergent existing seed: '+key)
                    task['rebuild_missing_seed'] = True
                    task['unavailable_seed_sha256'] = record['sha256']
                    missing.append(key)
            spec['previous_outputs'][key] = records
            path = campaign/'state'/(key+'.json')
            receipt = read_json(path)
            if receipt.get('pid') and process_identity(receipt['pid']) == receipt.get('identity'):
                raise RuntimeError('Live refresh worker: '+key)
            if receipt.get('payload_revision') != REFRESH_ID:
                continue
            if receipt.get('status') == 'complete':
                output = internal_path(ROOT/receipt['output'])
                if (output != campaign/'outputs'/(key+'.json.gz') or receipt.get('exit_code') != 0
                        or not refreshed_population_verified(task, receipt)
                        or sha256(output) != receipt['output_sha256']):
                    raise ValueError('Invalid completed refresh: '+key)
                completed[key] = receipt['output_sha256']
            elif (key in missing and receipt.get('status') == 'failed'
                  and receipt.get('exception_type') == 'FileNotFoundError'):
                requeue.append(key)
            else:
                raise RuntimeError('Unrelated failed/interrupted refresh: '+key)
        if set(missing) != MISSING_REFRESH_SEEDS or set(state['failed']) != set(requeue):
            raise ValueError('Missing-seed/failed scope differs from the inspected six files')
        for path in (campaign/'plan.json', campaign/'state.json'):
            preserve_before_refresh(campaign, path, revision=REFERENCE_REPAIR)
        for key in requeue:
            path = campaign/'state'/(key+'.json')
            preserve_before_refresh(campaign, path, revision=REFERENCE_REPAIR)
            log = campaign/'logs'/(key+'.log')
            if log.exists():
                preserve_before_refresh(campaign, log, move=True, revision=REFERENCE_REPAIR)
        old_sources = dict(spec['sources'])
        spec['sources'] = {name:sha256(ROOT/name) for name in spec['sources']}
        spec[REFERENCE_REPAIR] = dict(status='verified', time=time.time(), missing_seeds=sorted(missing),
            requeue=requeue, retained_completed=completed, previous_sources=old_sources,
            current_sources=dict(spec['sources']), physics_contract_unchanged=True,
            old_population_comparison_unavailable=sorted(missing))
        write_json(campaign/'plan.json', plan)
        for key in requeue:
            write_json(campaign/'state'/(key+'.json'), dict(key=key,status='pending',pid=None,
                payload_revision=REFRESH_ID,authorized_reference_repair=True))
        return refresh_plan(campaign)


def refresh_plan(campaign):
    from TROTASR.workflows.nominal_campaign import current_contract, verify_compiled
    plan = read_json(campaign/'plan.json')
    spec = plan['payload_refresh']
    if spec['id'] != REFRESH_ID or spec['year'] != 2025:
        raise ValueError('Unapproved payload refresh')
    verify_compiled()
    if current_contract(2025, plan['shape_mode'], spec['variations']) != spec['contract']:
        raise ValueError('Active refresh correction/selection contract changed')
    for name, digest in spec['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Active refresh source changed: '+name)
    return plan


def resume_paused_refresh(campaign):
    """Explicit stopped-state handoff, retaining completed bytes and old contract."""
    from TROTASR.workflows.nominal_campaign import current_contract, verify_compiled
    from TROTASR.workflows.normalization_recovery import process_identity
    campaign = internal_path(campaign)
    if campaign != ROOT/'campaigns/systematics_20260924':
        raise ValueError('Only the user-paused 2025 refresh can use this handoff')
    with (campaign/'controller.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = read_json(campaign/'plan.json'); spec = plan['payload_refresh']
        if SUPPORT_REFRESH in spec:
            return refresh_plan(campaign)
        state = read_json(campaign/'state.json')
        if state.get('status') != 'paused_by_user' or state.get('active') or state.get('pid'):
            raise RuntimeError('Require the actual user-paused controller state')
        verify_compiled()
        old_contract = spec['contract']
        current = current_contract(2025, plan['shape_mode'], spec['variations'])
        if {k:v for k,v in old_contract.items() if k!='code'} != {k:v for k,v in current.items() if k!='code'}:
            raise ValueError('Execution handoff must not alter the physics/payload contract')
        changed = {k for k in old_contract['code'] if old_contract['code'][k] != current['code'].get(k)}
        if changed != {'utils/systematic_views.py', 'utils/histogramming.py'}:
            raise ValueError('Unexpected inference support source changes: '+str(changed))
        for name,digest in {**old_contract['code'], **spec['sources']}.items():
            if sha256(ROOT/name) == digest: continue
            backup = campaign/'history'/REFRESH_ID/SUPPORT_REFRESH/'sources'/name
            if not backup.is_file() or sha256(backup) != digest:
                raise ValueError('Missing exact stopped-source backup: '+name)
        completed = {}; requeue = []
        for task in spec['tasks']:
            key=task['key']; path=campaign/'state'/(key+'.json')
            receipt=read_json(path)
            if receipt.get('payload_revision') != REFRESH_ID: continue
            if receipt.get('pid') and process_identity(receipt['pid']) == receipt.get('identity'):
                raise RuntimeError('Live retained worker: '+key)
            if receipt.get('status') == 'complete':
                if (receipt.get('exit_code') != 0 or not refreshed_population_verified(task,receipt)
                        or sha256(ROOT/receipt['output']) != receipt['output_sha256']):
                    raise ValueError('Invalid retained completion: '+key)
                completed[key]=receipt['output_sha256']
            elif receipt.get('status') == 'paused_by_user':
                # Interrupted workers had already moved the previous canonical
                # histogram into this campaign's history before computing.
                spec['previous_outputs'][key] = [refresh_output_record(campaign,r)
                    for r in spec['previous_outputs'][key]]
                preserve_before_refresh(campaign,path,revision=SUPPORT_REFRESH)
                log=campaign/'logs'/(key+'.log')
                if log.exists(): preserve_before_refresh(campaign,log,move=True,revision=SUPPORT_REFRESH)
                requeue.append(key)
            else:
                raise ValueError('Not a user-interrupted worker: '+key)
        for path in (campaign/'plan.json',campaign/'state.json'):
            preserve_before_refresh(campaign,path,revision=SUPPORT_REFRESH)
        spec[SUPPORT_REFRESH]=dict(time=time.time(),previous_contract=old_contract,
            retained_completed=completed,requeue=requeue,physics_definitions_changed=False,
            inference_support='all_region_pretopology_union',previous_sources=spec['sources'])
        spec['contract']=current
        spec['sources']={name:sha256(ROOT/name) for name in spec['sources']}
        write_json(campaign/'plan.json',plan)
        for key in requeue:
            write_json(campaign/'state'/(key+'.json'),dict(key=key,status='pending',pid=None,
                payload_revision=REFRESH_ID,authorized_support_resume=True))
        return refresh_plan(campaign)


def refresh_result_contract(spec, task, digest):
    """Only specifically hash-registered prior successes use the old code contract."""
    previous=spec.get(SUPPORT_REFRESH,{})
    if previous.get('retained_completed',{}).get(task['key']) == digest:
        return previous['previous_contract']
    return spec['contract']


def prepare_refresh(campaign, workers):
    """Revise the existing campaign, not a sidecar histogram/campaign."""
    from TROTASR.workflows import nominal_campaign as execution
    if not 1 <= workers <= 98:
        raise ValueError('Leave controller/diagnostic capacity within 100 slots')
    plan = read_json(campaign/'plan.json')
    if plan.get('payload_refresh'):
        return refresh_plan(campaign)
    if plan.get('shape_mode') not in ('cms_trota_jme', 'stored_trota_nonjme'):
        raise ValueError('Refresh only the two current systematic components')
    state = read_json(campaign/'state.json')
    if state.get('status') != 'histograms_complete' or state.get('active') or state.get('failed'):
        raise ValueError('Prior histogram campaign must be complete and idle')
    old_plan = preserve_before_refresh(campaign, campaign/'plan.json')
    preserve_before_refresh(campaign, campaign/'state.json')
    if (campaign/'stage_result.json').exists():
        preserve_before_refresh(campaign, campaign/'stage_result.json')
    expanded = execution.execution_tasks(plan)
    previous = {}
    for t in expanded:
        if t['year'] != 2025:
            continue
        s = read_json(campaign/'state'/(t['key']+'.json'))
        if s['status'] != 'complete':
            raise ValueError('Nonterminal old output: '+t['key'])
        record = dict(
            key=t['key'], output=s['output'], sha256=s['output_sha256'],
            endpoint=t.get('object_endpoint'))
        previous.setdefault(t.get('parent_key', t['key']), []).append(
            refresh_output_record(campaign, record))
    tasks = []
    for original in plan['tasks']:
        if original['year'] != 2025:
            continue
        task = execution.relocated_seed(original)
        task.pop('seed_output', None)
        if execution.fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Changed retained input: '+task['key'])
        if task.get('metadata') and execution.fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
            raise ValueError('Changed input metadata: '+task['key'])
        task['refresh_required'] = refresh_required(task)
        tasks.append(task)
    if sum(t['refresh_required'] for t in tasks) != 4661:
        raise ValueError('Expected all 4661 current 2025 MC/signal inputs')
    variations = ['all_weights'] if plan['shape_mode'] == 'cms_trota_jme' else ['nominal']
    sources = {name: sha256(ROOT/name) for name in (
        'workflows/run_systematic_campaign.py', 'workflows/build_systematics.py',
        'workflows/nominal_campaign.py', 'workflows/merge_nominal.py',
        'utils/compiled/manifest.json')}
    chunk_size = plan['chunk_size']
    if plan['shape_mode'] == 'stored_trota_nonjme':
        execution_contract = read_json(campaign/'execution.json')
        if execution_contract.get('schema') != 'trotasr_canonical_execution_v1':
            raise ValueError('Missing adopted branch-pruned execution contract')
        chunk_size = execution_contract['chunk_size']
        if chunk_size != 25000:
            raise ValueError('Unexpected adopted non-JME batch size')
        sources[str((campaign/'execution.json').relative_to(ROOT))] = sha256(campaign/'execution.json')
    spec = dict(id=REFRESH_ID, year=2025, previous_plan=str(old_plan.relative_to(ROOT)),
        previous_plan_sha256=sha256(old_plan), tasks=tasks, previous_outputs=previous,
        contract=execution.current_contract(2025, plan['shape_mode'], variations),
        sources=sources, variations=variations, workers=workers, chunk_size=chunk_size,
        aggregate_slots=100, reserve_bytes=32*1024**3, emergency_bytes=24*1024**3,
        created=time.time(), central_paths_preserved=True, event_sidecar=False,
        physics_change='2025 official heavy-flavour btag v4 and measured photon CSEV endpoints',
        validation=dict(status='pending_real_files', records={}))
    # Bind the untouched 2024 merge; neither its shards nor its weights are run.
    preserved = {}
    for name in ('systematics_2024.json.gz', 'systematics_2024.summary.json'):
        path = campaign/'merged'/name
        preserved[str(path.relative_to(ROOT))] = sha256(path)
    spec['preserved_2024'] = preserved
    plan['payload_refresh'] = spec
    write_json(campaign/'plan.json', plan)
    return refresh_plan(campaign)


def compare_refresh_populations(before, after):
    """SF changes may change sumw/sumw2, never bin edges or event populations."""
    def compare(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                raise ValueError('Changed histogram structure: '+path)
            if 'sumw' in a:
                for key in a:
                    if key not in ('sumw', 'sumw2') and a[key] != b.get(key):
                        raise ValueError('Changed unweighted histogram: '+path+'/'+key)
                return
            for key, value in a.items():
                if key not in b:
                    raise ValueError('Lost histogram population: '+path+'/'+key)
                compare(value, b[key], path+'/'+key)
        elif a != b:
            raise ValueError('Changed histogram metadata: '+path)
    for section in ('histograms', 'physical_histograms', 'dy_rz', 'dy_rz_variations'):
        compare(before.get(section, {}), after.get(section, {}), section)
    if before['input_metadata'] != after['input_metadata']:
        raise ValueError('Changed input/normalization bookkeeping')
    if before['audit']['events_read'] != after['audit']['events_read']:
        raise ValueError('Changed retained-event traversal')


def refresh_worker(campaign, key):
    from TROTASR.workflows.nominal_campaign import validate_payload, fingerprint
    from TROTASR.workflows.campaign_resources import chunk_collection
    from TROTASR.utils.histogramming import build
    path = campaign/'state'/(key+'.json')
    ownership = {}
    try:
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            ownership = read_json(path) if path.exists() else {}
            if ownership.get('pid') == os.getpid() and ownership.get('payload_revision') == REFRESH_ID:
                break
            time.sleep(.1)
        plan = refresh_plan(campaign); spec = plan['payload_refresh']
        task = next(t for t in spec['tasks'] if t['key'] == key and refresh_task_required(t))
        if ownership.get('pid') != os.getpid() or ownership.get('payload_revision') != REFRESH_ID:
            raise ValueError('Refresh worker ownership mismatch')
        if fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Refresh ROOT fingerprint changed')
        previous = spec['previous_outputs'][key]
        if not previous and not (task.get('rebuild_missing_seed') and key in MISSING_REFRESH_SEEDS
                                 and key in spec[REFERENCE_REPAIR]['missing_seeds']):
            raise ValueError('Missing unregistered previous output: '+key)
        # Validate old output bytes before any canonical-path replacement.
        for record in previous:
            p = ROOT/record['output']
            if sha256(p) != record['sha256']:
                raise ValueError('Prior histogram changed: '+record['output'])
        output = campaign/'outputs'/(key+'.json.gz')
        old_output = preserve_before_refresh(campaign, output, move=True) if output.exists() else None
        with chunk_collection(True):
            payload = build(ROOT/task['input'], 2025, output,
                variations=tuple(spec['variations']), shape_mode=plan['shape_mode'],
                chunk_size=spec['chunk_size'], candidate_budget=plan.get('candidate_budget',150000),
                support_only=bool(spec.get(SUPPORT_REFRESH)))
        validate_payload(payload, task, spec['contract'])
        from TROTASR.workflows.continue_systematic_campaign import finite_histogram_arrays
        for section in ('histograms','physical_histograms','dy_rz','dy_rz_variations'):
            finite_histogram_arrays(payload.get(section,{}))
        for record in previous:
            p = ROOT/record['output']
            if p == output and old_output is not None:
                p = old_output
            compare_refresh_populations(read_json(p), payload)
        # Read back actual output, not just the in-memory builder return.
        validate_payload(read_json(output), task, spec['contract'])
        refresh_plan(campaign)
        result = dict(ownership, status='complete', pid=None, finished=time.time(),
            output=str(output.relative_to(ROOT)), output_sha256=sha256(output),
            entries=task['entries'], year=2025, seconds=payload['seconds'],
            previous_populations_identical=True if previous else None,
            population_validation=('compared_to_retained_output' if previous else
                                   'retained_input_contract_old_seed_unavailable'), exit_code=0)
    except BaseException as exc:
        result = dict(ownership, status='failed', pid=None, finished=time.time(), exit_code=1,
            error=str(exc), exception_type=type(exc).__name__, traceback=traceback.format_exc())
    write_json(path, result)
    return result['exit_code']


def execute_refresh(campaign, workers):
    from TROTASR.workflows.nominal_campaign import resources, THREAD_ENV
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.campaign_resources import memory_usage
    from TROTASR.workflows.merge_nominal import merge_refreshed_year
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = prepare_refresh(campaign, workers); spec = plan['payload_refresh']
        tasks = [t for t in spec['tasks'] if refresh_task_required(t)]
        # Start with small nonempty real shards; these production outputs are
        # reused, not a second pilot campaign. Every later file is also compared.
        tasks.sort(key=lambda t:(t['entries']==0, t['entries'], t['key']))
        pending=[]; completed={}; failed={}; active={}
        for t in tasks:
            path=campaign/'state'/(t['key']+'.json')
            s=read_json(path) if path.exists() else {}
            if s.get('payload_revision') == REFRESH_ID:
                if s.get('status') == 'complete':
                    if sha256(ROOT/s['output']) != s['output_sha256']:
                        raise ValueError('Changed completed refresh')
                    completed[t['key']]=s;continue
                if s.get('pid') and process_identity(s['pid']) == s.get('identity'):
                    raise RuntimeError('Live refresh child retained; do not duplicate')
                if (s.get('status') == 'pending' and s.get('authorized_reference_repair')
                        and t['key'] in spec.get(REFERENCE_REPAIR, {}).get('requeue', [])):
                    pending.append(t)
                    continue
                if (s.get('status') == 'pending' and s.get('authorized_support_resume')
                        and t['key'] in spec.get(SUPPORT_REFRESH,{}).get('requeue',[])):
                    pending.append(t)
                    continue
                raise RuntimeError('Prior failed/interrupted refresh retained: '+t['key'])
            pending.append(t)
        state=dict(status='payload_refresh_running',payload_revision=REFRESH_ID,
            pid=os.getpid(),updated=time.time(),year=2025,total=len(tasks),
            completed=len(completed),active=0,pending=len(pending),failed=[],full_workflow_complete=False)
        while pending or active:
            for key,(proc,stream) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                stream.close();del active[key]
                s=read_json(campaign/'state'/(key+'.json'))
                if rc == 0 and s.get('status') == 'complete' and s.get('exit_code') == 0 and sha256(ROOT/s['output']) == s['output_sha256']:
                    completed[key]=s
                else:
                    failed[key]=dict(s, observed_exit_code=rc)
            external, memory = resources()
            usage=[memory_usage(p.pid) for p,_ in active.values()]
            budget=max(2*1024**3, int(max([0]+[u['VmHWM'] for u in usage])*1.15))
            growth=sum(max(0,budget-u['VmRSS']) for u in usage)
            limit=max(0,min(workers,99-external,
                len(active)+max(0,(memory-spec['reserve_bytes']-growth)//budget)))
            if memory < spec['emergency_bytes'] or failed:limit=0
            # Validate the first four actual production outputs before scaling.
            if len(completed)<4:limit=min(limit,1)
            for _ in range(min(8,len(pending),max(0,limit-len(active)))):
                task=pending.pop(0);key=task['key'];path=campaign/'state'/(key+'.json')
                if (path.exists() and key not in spec.get(REFERENCE_REPAIR, {}).get('requeue', [])
                        and key not in spec.get(SUPPORT_REFRESH,{}).get('requeue',[])):
                    preserve_before_refresh(campaign,path)
                log=campaign/'logs'/(key+'.log')
                if log.exists():preserve_before_refresh(campaign,log,move=True)
                stream=log.open('w')
                command=[sys.executable,'-u',str(ROOT/'workflows/run_systematic_campaign.py'),
                    '--campaign',str(campaign),'--refresh-worker',key]
                # A worker waits for its registered process identity below.
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**THREAD_ENV),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                write_json(path,dict(status='running',key=key,pid=proc.pid,identity=process_identity(proc.pid),
                    command=command,started=time.time(),payload_revision=REFRESH_ID))
                active[key]=(proc,stream)
            state=dict(status='payload_refresh_running',payload_revision=REFRESH_ID,
                pid=os.getpid(),updated=time.time(),year=2025,total=len(tasks),
                completed=len(completed),active=len(active),pending=len(pending),failed=list(failed),
                memory_available_bytes=memory,worker_budget_bytes=budget,external_slots=external,
                admission_limit=limit,aggregate_slots=100,full_workflow_complete=False)
            write_json(campaign/'state.json',state)
            if failed and not active:
                write_json(campaign/'state.json',dict(state,status='payload_refresh_needs_attention',errors=failed))
                return 1
            if pending or active:time.sleep(3)
        for name,digest in spec['preserved_2024'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('2024 product changed')
        plan['payload_refresh']['validation']=dict(status='passed',
            production_files=len(completed),
            every_file_population_comparison=not any(t.get('rebuild_missing_seed') for t in tasks),
            retained_input_contract_for_deleted_seeds=[t['key'] for t in tasks if t.get('rebuild_missing_seed')],
            corrected_compiled_payload=True,completed=time.time())
        write_json(campaign/'plan.json',plan)
        for suffix in ('.json.gz','.summary.json'):
            p=campaign/'merged'/('systematics_2025'+suffix)
            if p.exists():preserve_before_refresh(campaign,p,move=True)
        merge_refreshed_year(campaign)
        write_json(campaign/'state.json',dict(state,status='histograms_complete',pid=None,
            total=len(plan['tasks']),completed=len(plan['tasks']),active=0,pending=0,failed=[],
            refreshed_files=len(completed),completed_time=time.time()))
        return 0


def compare_values(before, after, path=''):
    """Counts/structure are exact; the user-approved 1e-7 roundoff is nonfatal."""
    if isinstance(before, dict):
        if not isinstance(after, dict) or set(before) != set(after):
            raise ValueError('Changed fields: '+path)
        for key in before:
            compare_values(before[key], after[key], path+'/'+str(key))
    elif isinstance(before, list):
        if not isinstance(after, list) or len(before) != len(after):
            raise ValueError('Changed dimensions: '+path)
        for i, (a,b) in enumerate(zip(before,after)):
            compare_values(a,b,path+'/'+str(i))
    elif isinstance(before, float):
        if not isinstance(after,(float,int)) or not math.isfinite(before) or not math.isfinite(after) or not math.isclose(before,after,rel_tol=1e-7,abs_tol=1e-7):
            raise ValueError('Changed numerical result: '+path)
    elif before != after:
        raise ValueError('Changed count/value: '+path)


def check_integration(campaign):
    """Compare direct canonical code with saved successes, never rerun production."""
    from TROTASR.workflows import nominal_campaign as execution
    from TROTASR.utils.histogramming import build
    from TROTASR.workflows.campaign_resources import chunk_collection
    execution.verify_compiled()
    plan = read_json(campaign/'plan.json')
    if plan.get('shape_mode') != 'stored_trota_nonjme' or plan.get('variations') != ['nominal']:
        raise ValueError('Integration comparison requires stored-TROTA non-JME')
    folder = campaign/'evidence'/'canonical_integration'
    sources = execution.integration_sources()
    previous_gate = folder/'report.json'
    if previous_gate.exists():
        saved = read_json(previous_gate)
        if saved.get('status') == 'passed' and saved.get('sources') == sources:
            for rec in saved['records']:
                if sha256(ROOT/rec['reference']) != rec['reference_sha256'] or sha256(ROOT/rec['output']) != rec['output_sha256']:
                    raise ValueError('Saved comparison output changed')
            return previous_gate
        raise ValueError('Preserve divergent prior integration evidence')
    # Lost pilot artifacts are rebuilt only when the original frozen plan
    # explicitly names them as seeds. Intact production outputs are never rebuilt.
    missing_seeds=[]
    for task in plan['tasks']:
        state_path=campaign/'state'/(task['key']+'.json')
        if not state_path.exists():continue
        state=read_json(state_path)
        if state.get('status')=='complete' and not (ROOT/state['output']).is_file():
            if not task.get('seed_output') or not state['output'].startswith('validation/'):
                raise ValueError('Unexpected missing production artifact: '+task['key'])
            missing_seeds.append((task,state,None))
    chosen = list(missing_seeds)
    for year in (() if missing_seeds else (2024,2025)):
        quota = {'background':6,'T2tt':1,'T2bW':1,'T2tb':1}
        for task in sorted((t for t in plan['tasks'] if t['year']==year and t['entries']>0),key=lambda t:(t['entries'],t['key'])):
            state_path = campaign/'state'/(task['key']+'.json')
            if not state_path.exists():
                continue
            state = read_json(state_path)
            if state.get('status') != 'complete':
                continue
            path = ROOT/state['output']
            if sha256(path) != state['output_sha256']:
                raise ValueError('Existing successful output changed: '+task['key'])
            before = read_json(path)
            names = str(before.get('input_metadata',{}))
            models = [m for m in ('T2tt','T2bW','T2tb') if m in names]
            if len(models)>1:
                raise ValueError('Ambiguous representative signal input')
            group = models[0] if models else 'background'
            if quota[group] <= 0:
                continue
            quota[group] -= 1
            chosen.append((task,state,group))
            if not any(quota.values()):
                break
        if any(quota.values()):
            raise ValueError('Insufficient existing successful representative inputs: '+str((year,quota)))
    records = []
    for task,state,group in chosen:
        original_task=task
        task=execution.relocated_seed(task)
        external, memory = execution.resources()
        if external+1>100 or memory<34*1024**3:
            raise RuntimeError('No single validation slot/memory reservation')
        if execution.fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Representative input changed')
        reference=ROOT/state['output']
        missing=not reference.is_file()
        size=plan['chunk_size']
        if missing:
            reference=folder/'reference'/(task['key']+'.json.gz')
            with chunk_collection(True):
                before=build(ROOT/task['input'],task['year'],reference,chunk_size=size,
                    shape_mode='stored_trota_nonjme',variations=('nominal',),reference_execution=True)
        else:
            before=read_json(reference)
            size=before['contract'].get('execution_adapter',{}).get('chunk_size',size)
        models=[m for m in ('T2tt','T2bW','T2tb') if m in str(before.get('input_metadata',{}))]
        if len(models)>1:raise ValueError('Ambiguous signal reference')
        group=models[0] if models else 'background'
        output = folder/(task['key']+'.json.gz')
        with chunk_collection(True):
            after = build(ROOT/task['input'], task['year'], output, chunk_size=size,
                shape_mode='stored_trota_nonjme', variations=('nominal',))
        for section in ('histograms','physical_histograms','dy_rz','dy_rz_variations','audit','input_metadata'):
            compare_values(before.get(section,{}),after.get(section,{}),section)
        records.append(dict(key=task['key'],year=task['year'],group=group,events=task['entries'],
            reference=execution.relative(reference),reference_sha256=sha256(reference),
            missing_seed_regenerated=missing,original_state=state,original_task=original_task,task=task,
            output=execution.relative(output),output_sha256=sha256(output),status='passed',
            branch_read=after['branch_read']))
        print(task['key'], 'equivalent', flush=True)
    for year in (2024,2025):
        counts={g:sum(r['year']==year and r['group']==g for r in records)
                for g in ('background','T2tt','T2bW','T2tb')}
        if counts!={'background':6,'T2tt':1,'T2bW':1,'T2tb':1}:
            raise ValueError('Incomplete representative comparison: '+str((year,counts)))
    report = dict(status='passed',sources=sources,records=records,histograms_and_audits_equal=True,
        contracts={str(y):execution.current_contract(y,'stored_trota_nonjme',['nominal']) for y in (2024,2025)},
        plan_sha256=sha256(campaign/'plan.json'),integer_counts_exact=True,float_roundoff_tolerance=1e-7,
        intact_production_outputs_recomputed=False,
        missing_seed_outputs_regenerated=sum(r['missing_seed_regenerated'] for r in records),checked=time.time())
    write_json(previous_gate,report)
    return previous_gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--check-integration',action='store_true',help='Compare canonical implementation with saved representative results')
    action.add_argument('--prepare-integration',action='store_true',help='Prepare a locked canonical handoff; do not start workers')
    action.add_argument('--file-worker',help=argparse.SUPPRESS)
    action.add_argument('--refresh-year',type=int,choices=[2025],
                        help='Apply the approved 2025 SF update in this existing central campaign')
    action.add_argument('--refresh-worker',help=argparse.SUPPRESS)
    p.add_argument('--workers',type=int,default=98)
    p.add_argument('--adopt-execution',type=Path,help='Existing execution contract whose live workers and successful receipts must be retained')
    a=p.parse_args();campaign=internal_path(a.campaign)
    if a.refresh_worker:
        return refresh_worker(campaign,a.refresh_worker)
    if a.refresh_year:
        return execute_refresh(campaign,a.workers)
    if a.file_worker:
        return file_worker(campaign,a.file_worker)
    if a.check_integration:
        print(check_integration(campaign))
        return 0
    if a.prepare_integration:
        from TROTASR.workflows.nominal_campaign import prepare_integration
        prepare_integration(campaign,a.adopt_execution)
        return 0
    plan=read_json(campaign/'plan.json')
    weights_only=plan.get('shape_mode') is None and plan.get('variations')==['all_weights']
    nonjme=plan.get('shape_mode')=='stored_trota_nonjme'
    if not weights_only and not nonjme and plan.get('shape_mode')!='cms_trota_jme':
        raise ValueError('This entry point requires a validated JME or stored-TROTA weight campaign')
    rc=run(campaign)
    if rc:return rc
    merged={str(y):str(merge(campaign,y)) for y in (2024,2025)}
    write_json(campaign/'stage_result.json',dict(status='stored_trota_nonjme_histograms_merged' if nonjme else 'stored_trota_weight_histograms_merged' if weights_only else 'central_weights_jme_histograms_merged',
        completed=time.time(),merged=merged,full_workflow_complete=False,
        remaining=([] if nonjme else ['other object-shape propagation with unchanged strategy'])+[
                   'background measurements and TF/nuisance propagation',
                   'templates, cards, limits, impacts and final figure QA']))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
