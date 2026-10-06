"""Internal, bounded nominal execution; no staging or external analysis imports.

Inputs must already be provisioned inside TROTASR. One fresh subprocess per
ROOT releases all analysis allocations at file boundaries. Histograms remain
per-file until a strict, all-input merge. A small validation campaign is never
reported as a full production campaign.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

COMPLETE = 'complete_one_file_test'
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                  'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def relative(path):
    return str(internal_path(path).relative_to(ROOT))


def fingerprint(path):
    s = internal_path(path).stat()
    return dict(inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def relocated_seed(task):
    """Use canonical retained inputs only, with the original inode/size/mtime."""
    result = dict(task)
    for field, stamp in (('input', 'fingerprint'), ('metadata', 'metadata_fingerprint')):
        if not task.get(field):
            continue
        target = ROOT/'inputs'/str(task['year'])/Path(task[field]).name
        if fingerprint(target) != task[stamp]:
            raise ValueError('Seed relocation is not the same original inode: '+task['key'])
        result[field] = relative(target)
    return result


def current_contract(year, shape_mode=None, variations=None):
    contract = dict(year=int(year),
        code={str(p.relative_to(ROOT)): sha256(p) for d in ('utils', 'gnn4lowdm')
              for p in sorted((ROOT/d).rglob('*.py'))},
        normalization_sha256=sha256(ROOT/'estimations'/('normalization_%s.json.gz' % year)),
        assets_sha256=sha256(ROOT/'jsons/assets.json'), modes=['highdm', 'lowdm'], variations=['nominal'],
        configuration_sha256=sha256(ROOT/'gnn4lowdm/config.json'),
        sr_binning_sha256=sha256(ROOT/'jsons/highdm_sr_binning.json'))
    if shape_mode is not None:
        if shape_mode not in ('cms_trota_jme', 'stored_trota_nonjme'):
            raise ValueError('Unsupported shape mode')
        contract['shape_mode'] = shape_mode
        if shape_mode == 'cms_trota_jme':
            contract.update(variations=['all_weights'],
                trota_models_sha256=sha256(ROOT/'models/TROTA/manifest.json'),
                jme_payloads_sha256=sha256(ROOT/'scales/JME/manifest.json'))
        else:
            from TROTASR.utils.corrections import nonjme_payload_contract, NONJME_ENDPOINTS
            contract.update(object_payloads=nonjme_payload_contract(year), object_endpoints=list(NONJME_ENDPOINTS))
    if variations is not None:
        if list(variations) not in (['nominal'], ['all_weights']):
            raise ValueError('Unsupported campaign weight selection')
        if shape_mode == 'cms_trota_jme' and list(variations) != ['all_weights']:
            raise ValueError('JME campaign retains all existing weight variations')
        contract['variations'] = list(variations)
    return contract


def validate_tasks(manifest):
    tasks = manifest['tasks']
    if not tasks:
        raise ValueError('Empty input inventory')
    keys, inodes = set(), set()
    for t in tasks:
        key = t['key']
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):
            raise ValueError('Invalid task key')
        p = internal_path(ROOT/t['input'])
        if p.suffix != '.root' or t['year'] not in (2024, 2025) or t['entries'] < 0:
            raise ValueError('Invalid nominal input')
        actual = fingerprint(p)
        if actual != t['fingerprint']:
            raise ValueError('Staged input fingerprint changed: ' + key)
        identity = (p.stat().st_dev, actual['inode'])
        if key in keys or identity in inodes:
            raise ValueError('Duplicate input key or inode: ' + key)
        keys.add(key); inodes.add(identity)
        if t.get('metadata'):
            if fingerprint(ROOT/t['metadata']) != t['metadata_fingerprint']:
                raise ValueError('Input metadata changed: ' + key)
        if t.get('seed_output'):
            internal_path(ROOT/t['seed_output'])
    if manifest.get('scope') not in ('representative_validation', 'full_nominal_production', 'full_cms_trota_jme_production', 'full_weight_systematic_production', 'full_nonjme_systematic_production'):
        raise ValueError('Explicit campaign scope required')
    if manifest['scope'] in ('full_nominal_production', 'full_cms_trota_jme_production', 'full_weight_systematic_production', 'full_nonjme_systematic_production'):
        expected = manifest.get('expected_files_by_year', {})
        counts = {str(y): sum(t['year'] == y for t in tasks) for y in (2024, 2025)}
        if counts != expected or not manifest.get('integrated_input_inventory_complete'):
            raise ValueError('Full campaign requires the complete validated integrated input inventory')
        gate = read_json(ROOT/manifest['validation_gate'])
        if gate.get('status') != 'passed' or not gate.get('three_signal_topologies_tested'):
            raise ValueError('Full production requires a completed representative validation gate')
        for year in (2024, 2025):
            if gate.get('contracts', {}).get(str(year)) != current_contract(year, manifest.get('shape_mode'), manifest.get('variations')):
                raise ValueError('Validation gate does not cover the current physics implementation')
        if manifest['scope'] == 'full_cms_trota_jme_production' and (
                manifest.get('shape_mode') != 'cms_trota_jme' or not gate.get('histogram_shape_validation')):
            raise ValueError('Re-inferred JME production requires a real histogram validation gate')
        if manifest['scope'] == 'full_weight_systematic_production' and (
                manifest.get('shape_mode') is not None or manifest.get('variations') != ['all_weights'] or
                not gate.get('stored_trota_weights_validation')):
            raise ValueError('Weight production requires validated stored-TROTA inputs')
        if manifest['scope'] == 'full_nonjme_systematic_production' and (
                manifest.get('shape_mode') != 'stored_trota_nonjme' or
                not gate.get('stored_trota_nonjme_validation')):
            raise ValueError('Non-JME production requires actual shifted histogram validation')
    return sorted(tasks, key=lambda t: t['key'])


def verify_compiled():
    m = read_json(ROOT/'utils/compiled/manifest.json')
    if m['status'] != 'complete':
        raise ValueError('Incomplete compiled definitions')
    for p, h in m['sources'].items():
        if sha256(ROOT/p) != h:
            raise ValueError('Stale compiled definition source: ' + p)
    for p, h in m['artifacts'].items():
        if sha256(ROOT/'utils/compiled'/p) != h:
            raise ValueError('Changed compiled artifact: ' + p)


def prepare(manifest_path, campaign, workers=99, chunk_size=2000):
    campaign = internal_path(campaign)
    if not 1 <= workers <= 99 or chunk_size < 1:
        raise ValueError('Invalid resource settings')
    manifest = read_json(manifest_path)
    tasks = validate_tasks(manifest)
    verify_compiled()
    if campaign.exists():
        raise FileExistsError('Campaign already exists; resume run instead of replacing the plan')
    campaign.mkdir(parents=True)
    for d in ('state', 'outputs', 'logs', 'history', 'merged'):
        (campaign/d).mkdir()
    plan = dict(schema='trotasr_nominal_plan_v1', scope=manifest['scope'], tasks=tasks,
                source_manifest=relative(manifest_path), source_manifest_sha256=sha256(manifest_path),
                contracts={str(y): current_contract(y, manifest.get('shape_mode'), manifest.get('variations')) for y in sorted({t['year'] for t in tasks})},
                runner_sha256=sha256(Path(__file__)), max_workers=workers,
                aggregate_slots=100, minimum_free_bytes=40*1024**3,
                minimum_available_memory_bytes=32*1024**3, admission_bytes_per_worker=4*1024**3,
                chunk_size=chunk_size, max_read_attempts=2)
    if manifest.get('variations') is not None:
        plan['variations'] = list(manifest['variations'])
    if manifest.get('shape_mode'):
        plan['shape_mode'] = manifest['shape_mode']
        plan['candidate_budget'] = 150000
        plan['admission_bytes_per_worker'] = 6*1024**3
        plan['common_failure_circuit_breaker'] = 3
    write_json(campaign/'plan.json', plan)
    write_json(campaign/'state.json', dict(status='prepared', scope=plan['scope'], total=len(tasks)))
    return plan


def validate_payload(payload, task, expected):
    if payload.get('status') != COMPLETE or not payload.get('sr_data_blinded'):
        raise ValueError('Missing complete blinded histogram output')
    c = payload['contract']
    if {k: c.get(k) for k in expected} != expected:
        raise ValueError('Histogram physics contract differs from frozen campaign')
    if internal_path(c['input']) != internal_path(ROOT/task['input']):
        raise ValueError('Histogram input mismatch')
    if c['fingerprint'] != {k: task['fingerprint'][k] for k in ('size', 'mtime_ns')}:
        raise ValueError('Histogram fingerprint mismatch')
    if payload['audit']['events_read'] != task['entries']:
        raise ValueError('Histogram event traversal incomplete')
    if expected.get('shape_mode') == 'stored_trota_nonjme':
        endpoints = payload['audit'].get('object_endpoints', {})
        empty_input = task['entries'] == 0 and not endpoints
        if (not empty_input and set(endpoints) != set(expected['object_endpoints']) or
                any(v.get('events') != task['entries'] for v in endpoints.values()) or
                'trota_inference' in payload['audit']):
            raise ValueError('Incomplete stored-TROTA object endpoint traversal')
    endpoint = task.get('object_endpoint')
    if endpoint is not None:
        if c.get('object_task_endpoint') != endpoint:
            raise ValueError('Independent endpoint identity mismatch')
        for section in ('histograms', 'physical_histograms'):
            if any(set(v) - {endpoint} for v in payload.get(section, {}).values()):
                raise ValueError('Unexpected variation in independent endpoint')
        if endpoint != 'nominal' and (payload.get('dy_rz') or payload['audit'].get('migrations')):
            raise ValueError('Duplicate nominal accounting in independent endpoint')
        if any(c.get('endpoint', 'nominal') != endpoint for c in payload['audit']['chunks']):
            raise ValueError('Unexpected selection audit endpoint')
        if any(w.get('endpoint', 'nominal') != endpoint for w in payload['audit']['weights']):
            raise ValueError('Unexpected weight audit endpoint')
        for section in ('dy_rz_variations',):
            if set(payload.get(section,{})) - ({endpoint} if endpoint != 'nominal' else set()):
                raise ValueError('Unexpected DY endpoint')
        for section in ('shape_migrations','object_response','object_acceptance'):
            if set(payload['audit'].get(section,{})) - ({endpoint} if endpoint != 'nominal' else set()):
                raise ValueError('Unexpected object audit endpoint')
    for mode in payload['histograms'].values():
        for variation in mode.values():
            for samples in variation.get('SR', {}).values():
                if 'data' in samples:
                    raise ValueError('SR data in histogram output')
    for chunk in payload['audit']['chunks']:
        for v in chunk.values():
            if isinstance(v, dict) and v.get('intersection', 0):
                raise ValueError('Nonzero high/low intersection')
    return payload


def execution_tasks(plan):
    """Expand only explicitly unstarted files; original input inventory stays unique."""
    spec = plan.get('endpoint_execution_update')
    if not spec:
        return plan['tasks']
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    split = set(spec['split_input_keys'])
    if (len(split) != len(spec['split_input_keys']) or
            not split <= {t['key'] for t in plan['tasks']} or
            plan.get('shape_mode') != 'stored_trota_nonjme' or
            plan.get('variations') != ['nominal']):
        raise ValueError('Invalid independent endpoint inventory')
    tasks = []
    for t in plan['tasks']:
        if t['key'] not in split:
            tasks.append(t)
            continue
        if t.get('seed_output'):
            raise ValueError('Preserve reusable seed output instead of splitting it')
        for endpoint in ('nominal', *NONJME_ENDPOINTS):
            child = dict(t, key=t['key']+'__'+endpoint,
                         parent_key=t['key'], object_endpoint=endpoint)
            tasks.append(child)
    if len({t['key'] for t in tasks}) != len(tasks):
        raise ValueError('Endpoint task key collision')
    return tasks


def task_contract(plan, task):
    expected = dict(plan['contracts'][str(task['year'])])
    if 'object_endpoint' in task:
        endpoint = task['object_endpoint']
        if endpoint != 'nominal' and endpoint not in expected['object_endpoints']:
            raise ValueError('Unprovided independent endpoint')
        expected.update(object_task_endpoint=endpoint,
                        object_endpoints=[] if endpoint == 'nominal' else [endpoint])
    return expected


def file_groups(plan, states):
    """Batch only unstarted endpoints of the same input, preserving all states."""
    from TROTASR.utils.histogramming import endpoint_tuple
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    tasks = {t['key']: t for t in execution_tasks(plan)}
    parents = {t['key']: t for t in plan['tasks']}
    if set(states) - set(tasks):
        raise ValueError('Foreign task state')
    groups = {}
    for key, task in tasks.items():
        if key in states:
            continue
        if task.get('seed_output'):
            raise ValueError('Existing seed must be reconciled, not recalculated: ' + key)
        parent = task.get('parent_key', key)
        group = groups.setdefault(parent, dict(parent_key=parent, task=parents[parent], task_keys=[], endpoints=[]))
        group['task_keys'].append(key)
        group['endpoints'].extend([task['object_endpoint']] if 'object_endpoint' in task else ['nominal', *NONJME_ENDPOINTS])
    for group in groups.values():
        group['endpoints'] = list(endpoint_tuple(group['endpoints']))
    return sorted(groups.values(), key=lambda g: (-g['task']['entries'], g['parent_key']))


def file_worker(campaign, parent):
    """Canonical file-local worker; executable only through run_systematic_campaign."""
    started = time.time()
    folder = campaign/'execution'
    try:
        plan = frozen_plan(campaign)
        spec = plan['_execution']
        group = next(g for g in spec['groups'] if g['parent_key'] == parent)
        task = group['task']
        from TROTASR.workflows.normalization_recovery import process_identity
        barrier = folder/'groups'/(parent+'.start.json')
        deadline = time.monotonic() + 30
        while not barrier.exists() and time.monotonic() < deadline:
            time.sleep(.1)
        launch = read_json(barrier)
        if launch['identity'] != process_identity(os.getpid()) or launch['contract_sha256'] != sha256(campaign/'execution.json'):
            raise ValueError('Worker launch identity/contract mismatch')
        if fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Input changed')
        if task.get('metadata') and fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
            raise ValueError('Metadata changed')
        for key in group['task_keys']:
            s = read_json(campaign/'state'/(key+'.json'))
            if s.get('batch_parent') != parent or s.get('pid') != os.getpid() or s['status'] != 'running':
                raise ValueError('Worker task ownership changed: '+key)
        combined = folder/'groups'/(parent+'.json.gz')
        from TROTASR.utils.histogramming import build, split_endpoint
        from TROTASR.workflows.campaign_resources import chunk_collection
        with chunk_collection(True):
            payload = build(ROOT/task['input'], task['year'], combined,
                chunk_size=spec['chunk_size'], variations=('nominal',),
                shape_mode='stored_trota_nonjme', object_endpoints=group['endpoints'])
        tasks = {t['key']: t for t in execution_tasks(plan)}
        results = {}
        for key in group['task_keys']:
            child = tasks[key]
            part = split_endpoint(payload, child['object_endpoint']) if 'object_endpoint' in child else payload
            validate_payload(part, child, task_contract(plan, child))
            out = folder/'outputs'/(key+'.json.gz')
            if out.exists():
                raise FileExistsError('Preserve existing endpoint output: '+key)
            write_json(out, part)
            results[key] = dict(status='complete', key=key, attempt=1, started=started, finished=time.time(),
                output=relative(out), output_sha256=sha256(out), entries=child['entries'], year=child['year'],
                batch_parent=parent, execution_contract_sha256=sha256(campaign/'execution.json'),
                seconds=payload['seconds'], reused=False)
        receipt = dict(status='complete', parent=parent, results=results, finished=time.time(),
            combined_output=relative(combined), combined_sha256=sha256(combined), returncode=0)
    except Exception as e:
        receipt = dict(status='failed', parent=parent, error=str(e), exception_type=type(e).__name__,
            traceback=traceback.format_exc(), finished=time.time(), returncode=1)
    write_json(folder/'receipts'/(parent+'.json'), receipt)
    return receipt['returncode']


def load_execution(campaign, plan):
    """Bind a canonical execution revision without rewriting the frozen plan."""
    spec = read_json(campaign/'execution.json')
    if spec.get('schema') != 'trotasr_canonical_execution_v1' or spec['plan_sha256'] != sha256(campaign/'plan.json'):
        raise ValueError('Canonical execution does not match the original plan')
    for path, digest in spec['sources'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Canonical execution source changed: '+path)
    if spec.get('legacy') and sha256(ROOT/spec['legacy']['path']) != spec['legacy']['sha256']:
        raise ValueError('Retained execution contract changed')
    if plan.get('shape_mode') != 'stored_trota_nonjme' or plan.get('variations') != ['nominal']:
        raise ValueError('File-local execution is only supported for stored-TROTA non-JME')
    gate = read_json(ROOT/spec['evidence'])
    if (sha256(ROOT/spec['evidence']) != spec['evidence_sha256'] or gate.get('status') != 'passed'
            or gate.get('contracts') != spec['contracts'] or not gate.get('histograms_and_audits_equal')):
        raise ValueError('Missing exact canonical execution equivalence evidence')
    for year, current in spec['contracts'].items():
        if current_contract(int(year), plan['shape_mode'], plan['variations']) != current:
            raise ValueError('Current implementation differs from tested execution')
        before = plan['contracts'][year]
        if {k:v for k,v in current.items() if k != 'code'} != {k:v for k,v in before.items() if k != 'code'}:
            raise ValueError('Execution integration changed physics inputs')
    result = copy.deepcopy(plan)
    result['contracts'] = spec['contracts']
    for i, task in enumerate(result['tasks']):
        record = spec.get('recovered', {}).get(task['key'])
        if record:
            if record['original_task'] != task or relocated_seed(task) != record['task']:
                raise ValueError('Recovered seed input relocation changed')
            result['tasks'][i] = record['task']
    result['_execution'] = spec
    return result


def integration_sources():
    names = ('nominal_campaign.py','run_systematic_campaign.py','build_systematics.py',
             'merge_nominal.py','campaign_resources.py','normalization_recovery.py')
    return {**{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'utils').rglob('*.py')},
            **{'workflows/'+name:sha256(ROOT/'workflows'/name) for name in names},
            **{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'gnn4lowdm').rglob('*.py')}}


def prepare_integration(campaign, adopt_path=None):
    """Register one tested canonical revision and retained work under the lock."""
    from TROTASR.workflows.normalization_recovery import process_identity
    campaign = internal_path(campaign)
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (campaign/'execution.json').exists():
            raise FileExistsError('Existing canonical execution must be resumed, not replaced')
        plan = read_json(campaign/'plan.json')
        evidence = campaign/'evidence/canonical_integration/report.json'
        gate = read_json(evidence)
        if (gate.get('status') != 'passed' or gate.get('sources') != integration_sources()
                or gate.get('plan_sha256') != sha256(campaign/'plan.json')
                or not gate.get('histograms_and_audits_equal') or len(gate.get('records',[])) != 18):
            raise ValueError('Exact canonical representative comparison has not passed')
        states = {p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        tasks = {t['key']:t for t in execution_tasks(plan)}
        preserved = {}
        recovered = {r['key']:r for r in gate['records'] if r.get('missing_seed_regenerated')}
        for key,s in states.items():
            if key not in tasks:
                raise ValueError('Foreign state: '+key)
            if s['status'] == 'complete':
                if key in recovered:
                    r=recovered[key]
                    if (s!=r['original_state'] or (ROOT/s['output']).exists()
                            or sha256(ROOT/r['output'])!=r['output_sha256']):
                        raise ValueError('Missing-seed recovery ownership changed: '+key)
                    if r['original_task']!=tasks[key] or relocated_seed(tasks[key])!=r['task']:
                        raise ValueError('Recovered seed input identity differs: '+key)
                    validate_payload(read_json(ROOT/r['output']),r['task'],
                        task_contract(dict(plan,contracts=gate['contracts']),r['task']))
                    continue
                if sha256(ROOT/s['output']) != s['output_sha256'] or fingerprint(ROOT/tasks[key]['input']) != tasks[key]['fingerprint']:
                    raise ValueError('Preserved input/output changed: '+key)
                preserved[key] = dict(state_sha256=sha256(campaign/'state'/(key+'.json')),
                    output=s['output'],output_sha256=s['output_sha256'])
        retained = []
        covered = set()
        legacy = None
        if adopt_path is not None:
            adopt_path = internal_path(adopt_path)
            if campaign not in adopt_path.parents:
                raise ValueError('Retained execution must belong to this campaign')
            old = read_json(adopt_path)
            if old.get('plan_sha256') != sha256(campaign/'plan.json'):
                raise ValueError('Retained execution plan differs')
            legacy = dict(path=relative(adopt_path),sha256=sha256(adopt_path),sources=old['sources'])
            for group in old['groups']:
                keys = group['task_keys']
                if not all(k in tasks for k in keys):
                    raise ValueError('Foreign retained group')
                subset = [states.get(k) for k in keys]
                if not any(s is not None for s in subset):
                    continue
                if all(s and s['status']=='complete' for s in subset):
                    continue
                if not all(s and s['status']=='running' for s in subset):
                    raise ValueError('Partial/failed group requires separate reconciliation')
                start = adopt_path.parent/'groups'/(group['parent_key']+'.start.json')
                barrier = read_json(start)
                ident = barrier['identity']
                actual = process_identity(ident['pid'])
                if actual is not None and actual != ident:
                    raise ValueError('Retained PID reused')
                if barrier['contract_sha256'] != legacy['sha256'] or any(s['pid']!=ident['pid'] for s in subset):
                    raise ValueError('Retained ownership/contract differs')
                retained.append(dict(kind='group',group=group,identity=ident,
                    receipt=relative(adopt_path.parent/'receipts'/(group['parent_key']+'.json')),
                    barrier=relative(start),barrier_sha256=sha256(start)))
                covered.update(keys)
            for record in old.get('retained',[]):
                ident = record.get('identity') or dict(pid=record['pid'],start_ticks=record['start_ticks'],arguments=record['args'])
                args = ident['arguments']
                key = args[args.index('--key')+1]
                s = states.get(key,{})
                if s.get('status') != 'running':
                    continue
                actual = process_identity(ident['pid'])
                if actual is not None and actual != ident:
                    raise ValueError('Retained original PID reused')
                if s.get('pid') != ident['pid']:
                    raise ValueError('Retained original task ownership differs')
                retained.append(dict(kind='single',key=key,identity=ident,
                    receipt=relative(campaign/'history'/(key+'.attempt'+str(s['attempt'])+'.json'))))
                covered.add(key)
        running = {k for k,s in states.items() if s['status']=='running'}
        if running != covered:
            raise ValueError('Unaccounted running tasks: '+str(sorted(running-covered)))
        spec = dict(schema='trotasr_canonical_execution_v1',created=time.time(),
            plan_sha256=sha256(campaign/'plan.json'),sources=integration_sources(),
            evidence=relative(evidence),evidence_sha256=sha256(evidence),contracts=gate['contracts'],
            original_contracts=plan['contracts'],
            legacy=legacy,preserved=preserved,recovered=recovered,retained=retained,groups=file_groups(plan,states),
            aggregate_slots=100,max_workers=99,reserve_bytes=32*1024**3,emergency_bytes=24*1024**3,
            chunk_size=25000,physics_changes=False,original_plan_unchanged=True)
        for name in ('groups','outputs','receipts','logs'):
            (campaign/'execution'/name).mkdir(parents=True,exist_ok=True)
        write_json(campaign/'execution.json',spec)
        load_execution(campaign,plan)
        print({'status':'prepared','retained_processes':len(retained),'pending_files':len(spec['groups'])},flush=True)


def integration_payload(payload, task, plan, state):
    """A legacy result must be a pinned preserved result or an exact retained receipt."""
    spec = plan['_execution']
    expected = task_contract(plan,task)
    if all(payload['contract'].get(k)==v for k,v in expected.items()):
        return validate_payload(payload,task,expected)
    old = read_json(ROOT/spec['legacy']['path']) if spec.get('legacy') else None
    # Match every physical/configuration field; only the recorded source version
    # may differ. Acceptance is tied to a file hash, not an arbitrary old code map.
    before = dict(expected,code=payload['contract'].get('code'))
    if not isinstance(before['code'],dict) or not before['code']:
        raise ValueError('Legacy output has no recorded source contract')
    saved = spec['preserved'].get(task['key'])
    authorized = bool(saved and state['output']==saved['output'] and state['output_sha256']==saved['output_sha256'])
    if not authorized and old:
        adapter = payload['contract'].get('execution_adapter',{})
        allowed = [v for k,v in old['sources'].items() if k.endswith('/adapter.py')]
        authorized = (state.get('execution_contract_sha256')==spec['legacy']['sha256']
                      and adapter.get('source_sha256') in allowed)
    if not authorized and any(r['kind']=='single' and r['key']==task['key'] for r in spec['retained']):
        original = spec['original_contracts'][str(task['year'])]
        authorized = all(payload['contract'].get(k)==v for k,v in original.items())
    if not authorized:
        raise ValueError('Unregistered legacy result cannot enter canonical execution: '+task['key'])
    return validate_payload(payload,task,before)


def run_file_groups(campaign, plan):
    """One campaign scheduler, retaining old processes without loading their code."""
    from TROTASR.workflows import campaign_resources as control
    from TROTASR.workflows.normalization_recovery import process_identity
    spec = plan['_execution']
    folder = campaign/'execution'
    tasks = {t['key']:t for t in execution_tasks(plan)}
    states = {p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
    for key,r in spec.get('recovered',{}).items():
        s=states[key]
        if s.get('output')==r['output'] and s.get('output_sha256')==r['output_sha256']:
            continue
        if s!=r['original_state'] or (ROOT/s['output']).exists() or sha256(ROOT/r['output'])!=r['output_sha256']:
            raise ValueError('Recovered seed cannot replace this state: '+key)
        validate_payload(read_json(ROOT/r['output']),tasks[key],task_contract(plan,tasks[key]))
        restored=dict(status='complete',key=key,attempt=s.get('attempt',1),output=r['output'],
            output_sha256=r['output_sha256'],entries=tasks[key]['entries'],year=tasks[key]['year'],
            acceptance_source='canonical_missing_seed_reconstruction',original_state=s,
            finished=time.time(),reused=False)
        write_json(campaign/'state'/(key+'.json'),restored);states[key]=restored
    for key,r in spec['preserved'].items():
        if sha256(campaign/'state'/(key+'.json'))!=r['state_sha256'] or sha256(ROOT/r['output'])!=r['output_sha256']:
            raise ValueError('Preserved successful state/output changed: '+key)
    retained = list(spec['retained'])
    active = {}
    pending = []
    for group in spec['groups']:
        subset=[states.get(k) for k in group['task_keys']]
        if all(s is None for s in subset):
            pending.append(group)
        elif all(s and s['status']=='complete' for s in subset):
            for s in subset:
                if sha256(ROOT/s['output'])!=s['output_sha256']:
                    raise ValueError('Completed canonical output changed')
        elif all(s and s['status']=='running' for s in subset):
            barrier=folder/'groups'/(group['parent_key']+'.start.json')
            start=read_json(barrier)
            retained.append(dict(kind='canonical',group=group,identity=start['identity'],
                receipt=relative(folder/'receipts'/(group['parent_key']+'.json')),
                barrier=relative(barrier),barrier_sha256=sha256(barrier)))
        elif not all(s and s['status']=='failed' for s in subset):
            raise ValueError('Partial canonical launch requires reconciliation')
    def accept(record, receipt, returncode=None):
        if record['kind']=='single':
            results={record['key']:receipt}
            keys=[record['key']]
        else:
            keys=record['group']['task_keys']
            results=receipt.get('results',{})
        if receipt.get('status')!='complete' or returncode not in (None,0) or set(results)!=set(keys):
            for key in keys:
                if states.get(key,{}).get('status')=='complete':
                    continue
                result=dict(status='failed',key=key,attempt=states.get(key,{}).get('attempt',1),
                    error=receipt.get('error','Worker exited without valid terminal receipt'),
                    returncode=returncode,finished=time.time(),retryable_read_error=False)
                write_json(campaign/'state'/(key+'.json'),result);states[key]=result
            return
        for key,s in results.items():
            if s.get('key')!=key or s.get('status')!='complete' or sha256(ROOT/s['output'])!=s['output_sha256']:
                raise ValueError('Terminal receipt/output differs: '+key)
            if fingerprint(ROOT/tasks[key]['input'])!=tasks[key]['fingerprint']:
                raise ValueError('Terminal input fingerprint differs')
            raw=read_json(ROOT/s['output'])
            if record['kind']=='canonical':
                if s.get('execution_contract_sha256')!=sha256(campaign/'execution.json'):
                    raise ValueError('Canonical receipt contract changed')
                validate_payload(raw,tasks[key],task_contract(plan,tasks[key]))
            else:
                integration_payload(raw,tasks[key],plan,s)
            if states.get(key,{}).get('status')=='complete':
                if states[key]!=s:raise ValueError('Do not overwrite an existing success')
            else:
                write_json(campaign/'state'/(key+'.json'),s);states[key]=s
    last_rss={}
    new_errors={}
    write_json(folder/'controller.json',dict(identity=process_identity(os.getpid()),started=time.time(),
        contract_sha256=sha256(campaign/'execution.json')))
    while pending or active or retained:
        for r in retained[:]:
            keys=[r['key']] if r['kind']=='single' else r['group']['task_keys']
            if all(states.get(k,{}).get('status')=='complete' for k in keys):
                retained.remove(r);continue
            actual=process_identity(r['identity']['pid'])
            if actual is not None:
                if actual!=r['identity']:raise ValueError('Retained PID identity changed')
                continue
            path=ROOT/r['receipt']
            receipt=read_json(path) if path.exists() else dict(status='failed',error='Missing retained terminal receipt')
            accept(r,receipt)
            retained.remove(r)
        for parent,(proc,stream,record) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            stream.close();path=ROOT/record['receipt']
            receipt=read_json(path) if path.exists() else dict(status='failed',error='Missing canonical receipt')
            accept(record,receipt,rc)
            if rc or receipt.get('status')!='complete':
                error=(receipt.get('exception_type',''),receipt.get('error','')[:200])
                new_errors[error]=new_errors.get(error,0)+1
            del active[parent]
        external,available=resources()
        live_retained=[r['identity']['pid'] for r in retained]
        external=max(0,external-len(live_retained))
        pids=[p.pid for p,_,_ in active.values()]+live_retained
        usage=[control.memory_usage(pid) for pid in pids]
        peak=max([0]+[u['VmHWM'] for u in usage])
        growth=max([0]+[u['VmRSS']-last_rss.get(pid,u['VmRSS']) for pid,u in zip(pids,usage)])
        budget=max(1024**3,int(peak*1.1)+2*max(0,growth))
        last_rss={pid:u['VmRSS'] for pid,u in zip(pids,usage)}
        reserve_growth=sum(max(0,budget-u['VmRSS']) for u in usage)
        allowed=max(0,min(spec['max_workers'],spec['aggregate_slots']-1-external,
            len(pids)+max(0,(available-spec['reserve_bytes']-reserve_growth)//budget)))
        if available<spec['emergency_bytes']:
            allowed=0
        common_failure=any(n>=plan.get('common_failure_circuit_breaker',3) for n in new_errors.values())
        if common_failure:
            allowed=0
        admissions=0
        while pending and len(active)+len(retained)<allowed and admissions<8 and shutil.disk_usage(campaign).free>plan['minimum_free_bytes']:
            group=pending.pop(0);parent=group['parent_key']
            if any((campaign/'state'/(k+'.json')).exists() for k in group['task_keys']):
                raise ValueError('Task ownership raced before admission')
            stream=(folder/'logs'/(parent+'.log')).open('a')
            command=[sys.executable,'-u',str(ROOT/'workflows/run_systematic_campaign.py'),
                     '--campaign',str(campaign),'--file-worker',parent]
            proc=subprocess.Popen(command,cwd=str(ROOT),env=dict(os.environ,**THREAD_ENV),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            for key in group['task_keys']:
                s=dict(status='running',key=key,attempt=1,batch_parent=parent,pid=proc.pid,started=time.time())
                write_json(campaign/'state'/(key+'.json'),s);states[key]=s
            write_json(folder/'groups'/(parent+'.start.json'),dict(identity=process_identity(proc.pid),command=command,
                contract_sha256=sha256(campaign/'execution.json')))
            record=dict(kind='canonical',group=group,receipt=relative(folder/'receipts'/(parent+'.json')))
            active[parent]=(proc,stream,record);admissions+=1
        summary=dict(status='running',updated=time.time(),pid=os.getpid(),**input_progress(plan,tasks,states),
            active=len(active)+len(retained),retained=len(retained),pending_groups=len(pending),
            failed=[k for k,s in states.items() if s['status']=='failed'],external_slots=external,
            aggregate_ceiling=spec['aggregate_slots'],memory_available_bytes=available,worker_budget_bytes=budget,
            full_workflow_complete=False)
        write_json(campaign/'state.json',summary)
        if common_failure and not active and not retained:
            write_json(campaign/'state.json',dict(summary,status='stopped_common_failure'))
            return 1
        if pending or active or retained:time.sleep(3)
    failed=[k for k,s in states.items() if s['status']!='complete']
    write_json(campaign/'state.json',dict(status='histograms_complete' if not failed else 'complete_with_failures',
        updated=time.time(),**input_progress(plan,tasks,states),active=0,pending=0,failed=failed,full_workflow_complete=False))
    return int(bool(failed) or len(states)!=len(tasks))


def input_progress(plan, tasks, states):
    """Do not call seventeen completed endpoint tasks seventeen input files."""
    grouped = {}
    for key, task in tasks.items():
        grouped.setdefault(task.get('parent_key', key), []).append(key)
    complete = [t for t in plan['tasks'] if all(
        states.get(k, {}).get('status') == 'complete' for k in grouped[t['key']])]
    return dict(total=len(plan['tasks']), completed=len(complete),
                completed_events=sum(t['entries'] for t in complete),
                execution_total=len(tasks), execution_completed=sum(
                    s['status'] == 'complete' for s in states.values()))


def frozen_plan(campaign):
    plan = read_json(campaign/'plan.json')
    if (campaign/'execution.json').exists():
        verify_compiled()
        return load_execution(campaign, plan)
    if sha256(Path(__file__)) != plan['runner_sha256']:
        raise ValueError('Runner changed; preserve campaign and make an explicit recovery plan')
    for year, expected in plan['contracts'].items():
        if current_contract(int(year), plan.get('shape_mode'), plan.get('variations')) != expected:
            raise ValueError('Physics code/configuration changed during campaign')
    for name,digest in plan.get('memory_control_policy',{}).get('workflow_sha256',{}).items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Memory controller source differs from validated plan: '+name)
    verify_compiled()
    return plan


def worker(campaign, key, attempt):
    started = time.time()
    task = None
    try:
        plan = frozen_plan(campaign)
        task = next(t for t in execution_tasks(plan) if t['key'] == key)
        if fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Input fingerprint changed before worker')
        if task.get('metadata') and fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
            raise ValueError('Metadata fingerprint changed before worker')
        output = ROOT/task['seed_output'] if task.get('seed_output') else campaign/'outputs'/(key+'.json.gz')
        if not output.exists():
            from TROTASR.utils.histogramming import build
            from TROTASR.workflows.campaign_resources import chunk_collection
            with chunk_collection(plan.get('memory_control_policy',{}).get('collect_after_chunk',False)):
                size = (plan['endpoint_execution_update']['chunk_size']
                        if 'object_endpoint' in task else plan['chunk_size'])
                build(ROOT/task['input'], task['year'], output, chunk_size=size,
                      variations=tuple(plan.get('variations', ('all_weights',) if plan.get('shape_mode') else ('nominal',))),
                      shape_mode=plan.get('shape_mode'), candidate_budget=plan.get('candidate_budget',150000),
                      object_endpoint=task.get('object_endpoint'))
        raw = read_json(output)
        from TROTASR.workflows.normalization_recovery import prior_contracts, payload_contract
        from TROTASR.workflows.systematic_continuation import prior_contracts as policy_contracts
        previous = {}
        if 'object_endpoint' not in task:
            previous = prior_contracts(plan)
            previous.update(policy_contracts(plan))
        expected = payload_contract(raw, task_contract(plan, task),
            None if 'object_endpoint' in task else previous.get(str(task['year'])))
        payload = validate_payload(raw, task, expected)
        result = dict(status='complete', key=key, attempt=attempt, started=started, finished=time.time(),
                      output=relative(output), output_sha256=sha256(output), entries=task['entries'],
                      year=task['year'], seconds=payload['seconds'], reused=bool(task.get('seed_output')))
    except Exception as e:
        # Only I/O/EOF failures receive a second process/read. Physics, missing
        # normalization, index and schema exceptions do not get blind retries.
        retryable = isinstance(e, (OSError, EOFError)) and not isinstance(e, FileExistsError)
        result = dict(status='failed', key=key, attempt=attempt, started=started, finished=time.time(),
                      exception_type=type(e).__name__, error=str(e), retryable_read_error=retryable,
                      traceback=traceback.format_exc(), input=task.get('input') if task else None)
    write_json(campaign/'history'/(key+'.attempt%s.json' % attempt), result)
    return 0 if result['status'] == 'complete' else 1


def external_process_slots(rows, own_pid):
    ours = {own_pid}
    while True:
        more = {pid for pid, parent, _ in rows if parent in ours}
        if more <= ours:
            break
        ours |= more
    return sum(pid not in ours and (comm.startswith(('python', 'root', 'combine', 'xrdcp', 'rsync'))
               or comm in ('hadd', 'cp')) for pid, _, comm in rows)


def resources():
    raw = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,ppid=,comm='], text=True)
    rows = [(int(p), int(pp), c) for p, pp, c in (s.split(None, 2) for s in raw.splitlines())]
    external = external_process_slots(rows, os.getpid())
    memory = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:'))
    return external, memory


def capacity(plan, external, memory, active):
    # MemAvailable already excludes allocations of active children. Reserve
    # additional memory only for newly admitted workers, not a second time.
    by_memory = active + max(0, (memory-plan['minimum_available_memory_bytes'])//plan['admission_bytes_per_worker'])
    return max(0, min(plan['max_workers'], plan['aggregate_slots']-1-external, by_memory))


def run(campaign):
    campaign = internal_path(campaign)
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = frozen_plan(campaign)
        if '_execution' in plan:
            return run_file_groups(campaign,plan)
        tasks = {t['key']: t for t in execution_tasks(plan)}
        states = {p.stem: read_json(p) for p in (campaign/'state').glob('*.json')}
        if set(states)-set(tasks):
            raise ValueError('Unexpected state outside plan')
        for key, s in states.items():
            if s['status'] == 'complete':
                if sha256(ROOT/s['output']) != s['output_sha256'] or fingerprint(ROOT/tasks[key]['input']) != tasks[key]['fingerprint']:
                    raise ValueError('Previously completed input/output changed: ' + key)
        pending = [k for k in tasks if k not in states]
        from TROTASR.workflows.normalization_recovery import prior_contracts, retry_keys, adopt_worker
        prior_contracts(plan)
        pending.extend(retry_keys(plan, states))
        from TROTASR.workflows import systematic_continuation as policy
        from TROTASR.workflows import campaign_resources as control
        policy.prior_contracts(plan)
        pending = list(dict.fromkeys(control.retry_keys(plan,states) + policy.retry_keys(plan,states) + pending))
        if plan.get('endpoint_execution_update'):
            # Legacy strategy: start expensive files first; nominal is a
            # distinct task, never a duplicated byproduct of each shift.
            pending.sort(key=lambda k: (-tasks[k]['entries'], k))
        active = {}
        # Explicitly retained original workers finish in place, without restart.
        for key, s in states.items():
            if s['status'] == 'running':
                retained = control.adopt_worker(plan, campaign, key, s)
                if retained is None:
                    retained = adopt_worker(plan, campaign, key, s)
                if retained is None:
                    retained = policy.adopt_worker(plan,campaign,key,s)
                if retained is not None:
                    active[key] = retained
                    continue
                try:
                    os.kill(s['pid'], 0)
                except ProcessLookupError:
                    pending.append(key)
                else:
                    raise RuntimeError('Prior worker still alive; do not duplicate: ' + key)
        environment = dict(os.environ, **THREAD_ENV)
        started = time.time()
        common_failure = None
        observed_peak = 0
        memory_audit = {}
        while pending or active:
            for key, (proc, stream, attempt) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                stream.close()
                result_path = campaign/'history'/(key+'.attempt%s.json' % attempt)
                s = read_json(result_path) if result_path.exists() else None
                if s is None or (rc != 0 and s['status'] == 'complete'):
                    s = dict(status='failed', key=key, attempt=attempt, finished=time.time(),
                             error='Worker exited without validated result', returncode=rc,
                             retryable_read_error=False)
                    if getattr(proc, 'identity_mismatch', None):
                        s['identity_issue'] = proc.identity_mismatch
                    write_json(campaign/'history'/(key+'.attempt%s.json' % attempt), s)
                write_json(campaign/'state'/(key+'.json'), s)
                states[key] = s
                del active[key]
                if s['status'] == 'failed' and s.get('retryable_read_error') and attempt < plan['max_read_attempts']:
                    pending.append(key)
                elif policy.approved_retry(plan,key,s):
                    pending.insert(0,key)
                elif key in control.retry_keys(plan, {key:s}):
                    # A precisely recorded retained-worker scheduling failure
                    # gets one validation attempt; an existing output is reused.
                    pending.insert(0,key)
            threshold = plan.get('common_failure_circuit_breaker')
            if threshold:
                errors = {}
                for key,item in states.items():
                    if item['status'] == 'failed' and key not in pending:
                        signature = (item.get('exception_type',''), item.get('error','')[:200])
                        errors[signature] = errors.get(signature,0)+1
                repeated = [s for s,n in errors.items() if n >= threshold]
                if repeated:
                    common_failure = repeated[0]
            external, memory = resources()
            # Adopted workers have PPID1 and would otherwise count twice.
            retained_pids={r['pid'] for field in (policy.FIELD,control.FIELD,'normalization_extension_recovery')
                           for r in plan.get(field,{}).get('retained_workers',{}).values()}
            external=max(0,external-sum(proc.pid in retained_pids for proc,_,_ in active.values()))
            if control.FIELD in plan:
                usage = [control.memory_usage(proc.pid) for proc,_,_ in active.values()]
                observed_peak = max([observed_peak]+[u['VmHWM'] for u in usage])
                limit,memory_audit = control.reserved_capacity(plan,external,memory,
                    [u['VmRSS'] for u in usage],observed_peak,time.time()-started)
                # Do not allow known pressure to become another global OOM.
                # Fail this attempt explicitly; it is not automatically retried.
                if memory < plan[control.FIELD]['emergency_available_bytes'] and active:
                    key = max(active,key=lambda k:states[k].get('started',0))
                    proc,stream,attempt = active[key]
                    from TROTASR.workflows.normalization_recovery import process_identity
                    ident = process_identity(proc.pid)
                    expected = [str(Path(__file__).resolve()),'worker','--campaign',str(campaign),
                                '--key',key,'--attempt',str(attempt)]
                    valid = ident and ident['arguments'][-len(expected):] == expected
                    if isinstance(proc,control.OrphanProcess):
                        valid = valid and ident == proc.record['identity']
                    receipt = campaign/'history'/(key+'.attempt%s.json'%attempt)
                    if valid and not receipt.exists():
                        import signal
                        os.kill(proc.pid,signal.SIGTERM)
                        write_json(receipt,dict(status='failed',key=key,attempt=attempt,finished=time.time(),
                            exception_type='MemoryPressure',error='Stopped before global OOM at memory safety threshold',
                            memory_available_bytes=memory,retryable_read_error=False))
                    limit = 0
            else:
                limit = capacity(plan, external, memory, len(active))
            free = shutil.disk_usage(campaign).free
            while pending and not common_failure and len(active) < limit and free > plan['minimum_free_bytes']:
                key = pending.pop(0)
                attempt = states.get(key, {}).get('attempt', 0) + 1
                log = campaign/'logs'/(key+'.attempt%s.log' % attempt)
                stream = log.open('a')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                    '--campaign', str(campaign), '--key', key, '--attempt', str(attempt)],
                    cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                s = dict(status='running', key=key, attempt=attempt, pid=proc.pid, started=time.time())
                write_json(campaign/'state'/(key+'.json'), s)
                states[key] = s
                active[key] = (proc, stream, attempt)
            completed = [k for k, s in states.items() if s['status'] == 'complete']
            failed = [k for k, s in states.items() if s['status'] == 'failed' and k not in pending]
            write_json(campaign/'state.json', dict(status='running' if active else 'waiting_for_resources',
                scope=plan['scope'], pid=os.getpid(), started=started, updated=time.time(),
                **input_progress(plan, tasks, states), failed=failed, pending=len(pending), active=len(active),
                active_limit=limit, external_slots=external, free_bytes=free, memory_available_bytes=memory,
                memory_control=memory_audit,
                full_workflow_complete=False))
            if common_failure and not active:
                write_json(campaign/'state.json', dict(status='stopped_common_failure',scope=plan['scope'],
                    pid=os.getpid(),updated=time.time(),**input_progress(plan,tasks,states),failed=failed,
                    pending=len(pending),active=0,error=list(common_failure),full_workflow_complete=False))
                return 1
            if pending or active:
                time.sleep(3)
        failed = [s for s in states.values() if s['status'] != 'complete']
        failed_inputs = {tasks[s['key']].get('parent_key',s['key']) for s in failed}
        write_json(campaign/'bad_files.json', dict(status='requires_investigation' if failed else 'none',
            files=failed, normalization_recomputed=False, lost_entries_known=sum(
                t['entries'] for t in plan['tasks'] if t['key'] in failed_inputs),
            fraction_files=len(failed_inputs)/len(plan['tasks']), dependent_fits_blocked=bool(failed)))
        write_json(campaign/'state.json', dict(status='histograms_complete' if not failed else 'complete_with_failures',
            scope=plan['scope'], pid=os.getpid(), updated=time.time(), **input_progress(plan,tasks,states),
            failed=[s['key'] for s in failed], active=0, pending=0,
            full_workflow_complete=False))
        return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'worker'))
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--workers', type=int, default=99)
    p.add_argument('--chunk-size', type=int, default=2000)
    p.add_argument('--key')
    p.add_argument('--attempt', type=int, default=1)
    a = p.parse_args()
    folder = internal_path(a.campaign)
    if a.action == 'prepare':
        if not a.manifest:
            p.error('prepare requires --manifest')
        prepare(a.manifest, folder, a.workers, a.chunk_size)
        return 0
    if a.action == 'worker':
        return worker(folder, a.key, a.attempt)
    return run(folder)


if __name__ == '__main__':
    raise SystemExit(main())
