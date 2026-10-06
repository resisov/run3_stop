"""Bounded correction of two demonstrated execution/representation defects.

No new fit policy for other failures: reconcile exact float serialization, and
apply the existing two strategies once to the two range-parser omissions.
Original controllers, contracts, ROOT outputs and failed attempts are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import fcntl
import math
import os
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.combine_limits import IntegrityCache, verify_card, runtime, collect
from TROTASR.workflows.continue_limit_fits import choose_range, command_for, MAX_RMAX
from TROTASR.workflows.diagnostic_common import CommandRunner
from TROTASR.workflows.limit_serialization import validate_saved_limit

RANGE_KEYS = {'T2tt/mStop1700_mLSP1500', 'T2tt/mStop1800_mLSP1600'}
SOURCES = ('workflows/reconcile_limit_fits.py', 'workflows/limit_serialization.py',
           'workflows/limit_validation.py', 'workflows/continue_limit_fits.py',
           'workflows/combine_limits.py', 'workflows/diagnostic_common.py')


def range_configs(seed):
    if not 40 < seed <= MAX_RMAX:
        raise ValueError('Invalid bounded range seed')
    return [dict(name='bisection_s0_range',strategy=0,tolerance=.1,rmax=seed),
            dict(name='bisection_s1_range',strategy=1,tolerance=.1,rmax=seed)]


def attempt_output(point, attempt):
    return ROOT/attempt['log'], (ROOT/attempt['log']).parent/(
        'higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')


def range_spec(record):
    logs=[]
    for attempt,old_range in zip(record['attempts'][:2],(20.,40.)):
        path=ROOT/attempt['log'];text=path.read_text(errors='replace')
        blocks=text.split('Will search for NLL crossing by bisection')[1:]
        if not 1<len(blocks)<5:
            continue
        # The first search is unambiguously median even if later searches fail.
        logs.append((attempt['name'], 'Will search for NLL crossing by bisection'+blocks[0], old_range))
    seed,evidence=choose_range(logs,40.)
    if not evidence or seed<=40:
        raise ValueError('No demonstrated missed range evidence')
    return seed,evidence


def make_plan(grid, output):
    points=[]
    original=read_json(output/'numerical_plan.json')
    old_plans={p['key']:p for p in original['points']}
    for point in grid['points']:
        key=point['model']+'/'+point['mass'];path=output/key/'state.json';record=read_json(path)
        if record['status']=='complete':continue
        if record['status']!='failed' or record.get('pid'):
            raise ValueError('Point has not finished: '+key)
        selected=None
        for attempt in record['attempts']:
            if attempt.get('exit_code')!=0:continue
            log,product=attempt_output(point,attempt)
            passed,evidence=validate_saved_limit(product,log,attempt['tolerance'])
            if passed and evidence.get('validation_adapter'):
                selected=dict(kind='serialization',attempt=attempt['name'],log=str(log.relative_to(ROOT)),
                    log_sha256=sha256(log),output=str(product.relative_to(ROOT)),output_sha256=sha256(product))
                break
        if selected is None and key in RANGE_KEYS:
            before=old_plans[key]
            if before['rmax']!=40 or before['range_evidence']:
                raise ValueError('Not the demonstrated range-parser omission')
            if [a['name'] for a in record['attempts']]!=[
                    'bisection_s0','bisection_s0_refined','bisection_s0_extended','bisection_s1']:
                raise ValueError('Unexpected prior attempts')
            seed,evidence=range_spec(record)
            selected=dict(kind='range_parser',rmax=seed,evidence=evidence,
                logs_sha256={a['log']:sha256(ROOT/a['log']) for a in record['attempts'][:2]},
                workspace_sha256=record['workspace_sha256'],attempts=range_configs(seed))
        if selected:
            points.append(dict(point=point,key=key,source_state_sha256=sha256(path),**selected))
    return dict(points=points,range_scope=sorted(RANGE_KEYS),maximum_range_attempts=2,
        physics_changed=False,original_validator_unchanged=True,other_failures_not_resubmitted=True)


def verified_complete(point,record):
    product=ROOT/record['output']
    if sha256(product)!=record['output_sha256']:raise ValueError('Accepted output changed')
    attempts=[a for a in record['attempts'] if attempt_output(point,a)[1]==product]
    if len(attempts)!=1:raise ValueError('Accepted output not linked to exactly one attempt')
    attempt=attempts[0]
    if attempt.get('exit_code')!=0:raise ValueError('Accepted command failed')
    passed,_=validate_saved_limit(product,ROOT/attempt['log'],attempt['tolerance'])
    if not passed:raise ValueError('Accepted output no longer validates')


def run_point(spec,output,runner,env,digest,contract_sha):
    point,key=spec['point'],spec['key'];work=output/key;path=work/'state.json'
    with (work/'point.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=read_json(path);verify_card(point,digest)
        if record.get('pid'):raise ValueError('A point process is still registered')
        if record['status']=='complete':
            verified_complete(point,record);return 'complete'
        backup=output/'history/validation_20260924'/key/'state.json'
        if backup.exists():
            if sha256(backup)!=spec['source_state_sha256']:raise ValueError('Original failure changed')
        else:
            if sha256(path)!=spec['source_state_sha256']:raise ValueError('Point changed after planning')
            write_json(backup,record)
            if sha256(backup)!=spec['source_state_sha256']:raise ValueError('Snapshot bytes differ')
        candidates=[]
        if spec['kind']=='serialization':
            for field in ('log','output'):
                if sha256(ROOT/spec[field])!=spec[field+'_sha256']:raise ValueError('Saved fit/log changed')
            candidates=[a for a in record['attempts'] if a['name']==spec['attempt']]
        elif spec['kind']=='range_parser':
            if key not in RANGE_KEYS or spec['attempts']!=range_configs(spec['rmax']):
                raise ValueError('Range scope broadened')
            ws=work/'workspace.root'
            if sha256(ws)!=spec['workspace_sha256']:raise ValueError('Workspace changed')
            for name,want in spec['logs_sha256'].items():
                if sha256(ROOT/name)!=want:raise ValueError('Range evidence changed')
            for configuration in spec['attempts']:
                existing=next((a for a in record['attempts'] if a['name']==configuration['name']),None)
                if existing is not None:
                    candidates=[existing]
                else:
                    here=work/configuration['name'];here.mkdir(exist_ok=True);(here/'tmp').mkdir(exist_ok=True)
                    product=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
                    log=here/'combine.log';command=command_for(ws,point['mass'],configuration)
                    receipt=runner.run(command,here,log,[product],env=dict(env,TMPDIR=str(here/'tmp')))
                    passed,evidence=validate_saved_limit(product,log,.1) if receipt['status']=='complete' else (False,{'reason':'command/output failed'})
                    existing=dict(configuration,passed=passed,validation=evidence,exit_code=receipt['exit_code'],
                        log=str(log.relative_to(ROOT)),command=command)
                    record['attempts'].append(existing)
                    record.update(command=command,current_attempt=configuration['name'])
                    write_json(here/'validation.json',existing);write_json(path,record)
                    candidates=[existing]
                if existing.get('passed'):break
        else:
            raise ValueError('Unknown validation correction')
        for attempt in candidates:
            if attempt.get('exit_code')!=0:continue
            log,product=attempt_output(point,attempt)
            passed,evidence=validate_saved_limit(product,log,attempt['tolerance'])
            if not passed:continue
            verify_card(point,digest)
            record.update(status='complete',pid=None,finished=time.time(),output=str(product.relative_to(ROOT)),
                output_sha256=sha256(product),validation=evidence,current_attempt=attempt['name'],
                validation_reconciliation=dict(contract_sha256=contract_sha,kind=spec['kind'],
                    source_state_sha256=spec['source_state_sha256'],original_attempts_retained=True))
            record.pop('error',None);write_json(path,record);return 'complete'
        record.update(status='failed',pid=None,finished=time.time(),
            error='finite_validation_correction_exhausted')
        write_json(path,record)
        return 'failed'


def run(manifest,output):
    manifest,output=internal_path(manifest),internal_path(output)
    with (output/'controller.lock').open('a') as lock, (output/'numerical_controller.lock').open('a') as numerical:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(numerical,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=read_json(output/'state.json');old=read_json(output/'numerical_state.json')
        if old.get('pid') or old['status'] not in ('needs_attention','fits_complete_pending_plots'):
            raise ValueError('Prior bounded continuation is not finished')
        if base['status'] not in ('needs_attention','fits_complete_pending_plots'):raise ValueError('Prior fits not finished')
        grid=read_json(manifest)
        if grid['status']!='cards_ready' or grid['scope']!='full_nominal_production':
            raise ValueError('Invalid grid')
        if read_json(output/'contract.json')['manifest_sha256']!=sha256(manifest):raise ValueError('Wrong grid')
        for contract_file in ('contract.json','numerical_contract.json'):
            for name,want in read_json(output/contract_file)['code'].items():
                if sha256(ROOT/name)!=want:raise ValueError('Previous source changed')
        plan_path=output/'validation_plan.json'
        if plan_path.exists():plan=read_json(plan_path)
        else:
            plan=make_plan(grid,output);write_json(plan_path,plan)
            write_json(output/'history/validation_20260924/controller_state.json',base)
        contract=dict(manifest_sha256=sha256(manifest),plan_sha256=sha256(plan_path),code={s:sha256(ROOT/s) for s in SOURCES})
        cp=output/'validation_contract.json'
        if cp.exists() and read_json(cp)!=contract:raise ValueError('Validation contract changed')
        write_json(cp,contract);digest=IntegrityCache();env=runtime(output)
        runner=CommandRunner(output/'validation_commands')
        path=output/'validation_state.json'
        state=dict(status='running',pid=os.getpid(),started=time.time(),results={},full_workflow_complete=False)
        write_json(path,state)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks={pool.submit(run_point,s,output,runner,env,digest,sha256(cp)):s['key'] for s in plan['points']}
                for future in as_completed(tasks):
                    key=tasks[future]
                    try:state['results'][key]=dict(status=future.result())
                    except Exception as exc:state['results'][key]=dict(status='failed',error=repr(exc))
                    state['updated']=time.time();write_json(path,state)
            results={}
            for point in grid['points']:
                key=point['model']+'/'+point['mass'];record=read_json(output/key/'state.json')
                verify_card(point,digest)
                if record['status']=='complete':verified_complete(point,record)
                results[key]=record
            for model in ('T2tt','T2bW','T2tb'):
                write_json(output/model/'expected_limits.json',collect(grid['points'],results,model,grid['scope']))
            failed=[k for k,r in results.items() if r['status']!='complete']
            final=dict(status='needs_attention' if failed else 'fits_complete_pending_plots',total=len(results),
                completed=len(results)-len(failed),failed=failed,results=results,observed_sr_used=False,
                validation_contract_sha256=sha256(cp),full_workflow_complete=False)
            write_json(output/'state.json',final);state.update(status=final['status'],completed=final['completed'],failed=failed)
        except BaseException as exc:
            state.update(status='needs_attention',error=repr(exc));raise
        finally:
            state.update(pid=None,finished=time.time());write_json(path,state)
    return 2 if failed else 0


def profile_recovery_configs(seed, max_calls=None):
    """Finite numerical alternatives; never change or freeze model parameters."""
    if not math.isfinite(seed) or not .0001 <= seed <= 100000:
        raise ValueError('Invalid measured-NLL range')
    if max_calls is not None:
        if max_calls != 2000000:
            raise ValueError('Only the finite measured-call-limit correction is admitted')
        return [dict(name='migrad_s0_calls2m', strategy=0, tolerance=.1, rmax=seed,
                     algorithm='Migrad', precision=None, max_calls=max_calls),
                dict(name='combined_s1_calls2m', strategy=1, tolerance=.1, rmax=seed,
                     algorithm='Combined', precision=None, max_calls=max_calls)]
    return [dict(name='combined_s1', strategy=1, tolerance=.1, rmax=seed,
                 algorithm='Combined', precision=None),
            dict(name='migrad_precision8', strategy=1, tolerance=.1, rmax=seed,
                 algorithm='Migrad', precision=1e-8)]


def profile_recovery_command(workspace, mass, spec):
    command = command_for(workspace, mass, spec)
    command[command.index('-v')+1] = '3'
    command += ['--cminDefaultMinimizerAlgo', spec['algorithm']]
    if spec['precision'] is not None:
        command += ['--cminDefaultMinimizerPrecision', str(spec['precision'])]
    if spec.get('max_calls'):
        command += ['--X-rtd', 'MINIMIZER_MaxCalls='+str(spec['max_calls'])]
    return command


def recover_profile(manifest, output, key, max_calls=None, verified_inputs=None):
    """One actual failed-point fit, through this existing canonical entrypoint.

    The active primary and adaptive sources/contracts/states are read-only.
    Point locking and shared admission serialize against their normal workers.
    Outputs remain a separately validated candidate until a locked final handoff.
    """
    from TROTASR.workflows.continue_limit_fits import (
        verify_assist_inputs, verify_attempt, same_primary_failure)
    manifest, output = internal_path(manifest), internal_path(output)
    if verified_inputs is None:
        digest = IntegrityCache()
        grid, primary_contract = verify_assist_inputs(manifest, output, digest)
    else:
        grid, primary_contract, digest = verified_inputs
    matches = [p for p in grid['points'] if p['model']+'/'+p['mass'] == key]
    if len(matches) != 1:
        raise ValueError('Point must belong to the current approved grid')
    point = matches[0]
    work = output/key
    adaptive = output/'numerical_adaptive_nll'
    predecessor = read_json(adaptive/'contract.json')
    for name, expected in predecessor['code'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Active adaptive source changed: '+name)
    target = output/('numerical_profile_calls2m' if max_calls else 'numerical_profile_recovery')/key
    with (work/'point.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read_json(work/'state.json')
        if original['status'] == 'complete':
            verified_complete(point, original)
            return 0
        prior = read_json(adaptive/key/'state.json')
        if (original['status'] != 'failed' or original.get('pid')
                or not original.get('workspace_valid')
                or prior['status'] != 'failed' or not prior.get('attempts')):
            raise ValueError('Requires an exhausted, terminal adaptive point')
        workspace = work/'workspace.root'
        if digest(workspace) != original['workspace_sha256']:
            raise ValueError('Workspace changed')
        if not same_primary_failure(original, read_json(adaptive/key/'primary_state.json')):
            raise ValueError('Primary receipt changed since adaptive retry')
        if target.exists():
            raise ValueError('Existing profile recovery retained; no automatic reset')
        # Retain all history. Saved outputs must be checked before another fit.
        previous = [*prior.get('previous_attempts', []), *prior['attempts']]
        earlier = output/'numerical_profile_recovery'/key/'state.json'
        if max_calls and earlier.exists():
            saved = read_json(earlier)
            if saved.get('pid') or saved['status'] not in ('failed','verified_not_promoted'):
                raise ValueError('Earlier profile recovery is not terminal')
            previous.extend(saved.get('attempts', []))
        for attempt in previous:
            passed, evidence = verify_attempt(point, attempt)
            if passed:
                raise ValueError('A stored passing result exists; reconcile without refitting')
        seed = prior['attempts'][-1]['rmax']
        specs = profile_recovery_configs(seed, max_calls)
        contract = dict(point=point, manifest_sha256=digest(manifest),
            primary_contract_sha256=digest(output/'contract.json'),
            adaptive_contract_sha256=digest(adaptive/'contract.json'),
            adaptive_point_sha256=digest(adaptive/key/'state.json'),
            original_state_sha256=digest(work/'state.json'),
            workspace_sha256=digest(workspace),
            code={n:digest(ROOT/n) for n in SOURCES}, attempts=specs,
            maximum_new_attempts=2, physics_model_changed=False,
            original_validator_unchanged=True, warnings_diagnostic_only=True,
            primary_promotion=False, observed_sr_used=False)
        if max_calls:
            contract['call_limit_evidence'] = dict(
                point='T2bW/mStop800_mLSP200', previous_max_calls=248000,
                observed_calls=[296645,259596], observed_status=4,
                log='stats/combined_systematics_sr118/limits/numerical_profile_recovery/'
                    'T2bW/mStop800_mLSP200/combined_s1/combine.log')
            contract['call_limit_evidence']['log_sha256'] = sha256(ROOT/contract['call_limit_evidence']['log'])
        target.mkdir(parents=True)
        write_json(target/'contract.json', contract)
        write_json(target/'primary_state.json', original)
        state = dict(status='running', pid=os.getpid(), started=time.time(),
                     point=point, attempts=[], contract_sha256=sha256(target/'contract.json'))
        write_json(target/'state.json', state)
        env = runtime(target)
        runner = CommandRunner(target/'command_receipts')
        try:
            for spec in specs:
                here = target/spec['name']; (here/'tmp').mkdir(parents=True)
                product = here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
                log = here/'combine.log'
                command = profile_recovery_command(workspace, point['mass'], spec)
                receipt = runner.run(command, here, log, [product], env=dict(env,TMPDIR=str(here/'tmp')))
                attempt = dict(spec, command=command, exit_code=receipt['exit_code'],
                    log=str(log.relative_to(ROOT)), log_sha256=sha256(log),
                    output=str(product.relative_to(ROOT)),
                    output_sha256=sha256(product) if product.is_file() else None)
                passed, evidence = verify_attempt(point, attempt)
                attempt.update(passed=passed, validation=evidence)
                for name, expected in contract['code'].items():
                    if sha256(ROOT/name) != expected:
                        raise ValueError('Profile recovery code changed while active')
                verify_card(point, digest)
                if (sha256(workspace) != contract['workspace_sha256']
                        or sha256(work/'state.json') != contract['original_state_sha256']
                        or sha256(adaptive/key/'state.json') != contract['adaptive_point_sha256']):
                    raise ValueError('Protected point inputs/receipts changed')
                write_json(here/'validation.json', attempt)
                state['attempts'].append(attempt)
                write_json(target/'state.json', state)
                if passed:
                    state.update(status='verified_not_promoted', accepted_attempt=attempt,
                        output=attempt['output'], output_sha256=attempt['output_sha256'], validation=evidence)
                    return 0
            state.update(status='failed', error='two_profile_configurations_exhausted')
            return 2
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error))
            raise
        finally:
            state.update(pid=None, finished=time.time())
            write_json(target/'state.json', state)


def recover_profile_batch(manifest, output, keys, workers, max_calls):
    """A frozen explicit point list; shared hash cache and aggregate admission."""
    from TROTASR.workflows.continue_limit_fits import verify_assist_inputs, assist_identity, assist_capacity
    from TROTASR.workflows.nominal_campaign import resources
    if max_calls != 2000000 or not 1 <= workers <= 99 or len(keys) != len(set(keys)):
        raise ValueError('Invalid finite recovery batch')
    manifest, output = internal_path(manifest), internal_path(output)
    digest = IntegrityCache()
    grid, primary = verify_assist_inputs(manifest, output, digest)
    valid = {p['model']+'/'+p['mass'] for p in grid['points']}
    if not keys or not set(keys) <= valid:
        raise ValueError('Recovery points must be explicit current-grid keys')
    import hashlib, json
    batch_id = hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()[:16]
    directory = output/'numerical_profile_calls2m/batches'/batch_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'contract.json',dict(points=keys,max_calls=max_calls,
        source_sha256=sha256(Path(__file__)),workers=workers,aggregate_slots=100,
        manifest_sha256=sha256(manifest),reserved_diagnostic_slots=1))
    state = dict(status='running',pid=os.getpid(),started=time.time(),results={})
    pending = list(keys); active = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    key = active.pop(future)
                    try: state['results'][key] = dict(exit_code=future.result())
                    except Exception as exc: state['results'][key] = dict(error=repr(exc))
            external, memory = resources()
            ps = read_json(output/'state.json')
            identity = assist_identity(ps.get('pid')) if ps.get('pid') else None
            primary_slots = children = 0
            if identity:
                args = identity['argv']; primary_slots = int(args[args.index('--workers')+1])
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit(): continue
                    try:
                        fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
                        children += fields[0] != 'Z' and int(fields[1]) == identity['pid']
                    except (OSError,IndexError,ValueError): pass
            capacity = assist_capacity(workers,external,memory,len(active),primary_slots,children)
            while pending and len(active) < capacity:
                key = pending.pop(0)
                active[pool.submit(recover_profile,manifest,output,key,max_calls,(grid,primary,digest))] = key
            state.update(active=list(active.values()),pending=len(pending),updated=time.time(),
                         admission_limit=capacity,available_memory_bytes=memory)
            write_json(directory/'state.json',state)
            if active: wait(active,timeout=5,return_when=FIRST_COMPLETED)
            elif pending: time.sleep(5)
    state.update(status='finished_pending_reconciliation',pid=None,finished=time.time())
    write_json(directory/'state.json',state)
    return 0 if all(x.get('exit_code') == 0 for x in state['results'].values()) else 2



# 2026-09-30 user request: cover every unfinished current-model point.
UNFINISHED_HISTORY = ('numerical_rmax100000', 'numerical_adaptive_nll',
                      'numerical_profile_recovery', 'numerical_profile_calls2m')


def unfinished_configs(seed, already_tried_calls2m):
    profile_recovery_configs(seed, 2000000)  # Validate the common range ceiling.
    if not already_tried_calls2m:
        return profile_recovery_configs(seed, 2000000)
    # The four exhausted compressed fits stopped with large EDM/status3, not
    # an Asimov warning. Change finite-difference precision and strategy;
    # never reset or repeat the previous two command identities.
    return [dict(name='precision8_s%d_calls2m' % strategy, strategy=strategy,
                 tolerance=.1, rmax=seed, algorithm='Migrad',
                 precision=1e-8, max_calls=2000000) for strategy in (1, 2)]


def unfinished_history(point, output):
    key = point['model']+'/'+point['mass']
    records = [('primary', output/key/'state.json')]
    records += [(name, output/name/key/'state.json') for name in UNFINISHED_HISTORY
                if (output/name/key/'state.json').exists()]
    history, hashes = [], {}
    for origin, path in records:
        record = read_json(path)
        if record.get('status') not in ('complete','failed','verified_not_promoted'):
            raise ValueError('Nonterminal predecessor retained: '+str(path))
        if record.get('pid'):
            raise ValueError('Registered predecessor process retained: '+str(path))
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        history.extend((origin, attempt) for attempt in record.get('attempts', []))
    return history, hashes


def unfinished_saved(point, output, history):
    from TROTASR.workflows.continue_limit_fits import verify_attempt
    from TROTASR.workflows.diagnostic_common import command_key
    key = point['model']+'/'+point['mass']
    logs = []
    for origin, attempt in history:
        log = ROOT/attempt['log']
        if sha256(log) != attempt['log_sha256']:
            raise ValueError('Historical log changed')
        logs.append((attempt['log'], log.read_text(errors='replace')))
        if attempt.get('exit_code') != 0:
            continue
        if read_json(log.parent/'validation.json') != attempt:
            raise ValueError('Historical validation receipt differs')
        if origin != 'primary':
            directory = output/origin
            if origin in ('numerical_profile_recovery','numerical_profile_calls2m'):
                directory = directory/key
            execution = read_json(directory/'command_receipts/commands'/
                (command_key(attempt['command'], log.parent)+'.json'))
            if (execution['status'] != 'complete' or execution['exit_code'] != 0
                    or execution.get('pid') or execution['command'] != attempt['command']
                    or execution['products'].get(attempt['output']) != attempt['output_sha256']):
                raise ValueError('Historical execution receipt differs')
        passed, evidence = verify_attempt(point, attempt)
        if passed:
            return dict(origin=origin, attempt=attempt, validation=evidence), logs
    return None, logs


def recover_unfinished_point(point, output, auxiliary, digest, env, code):
    from TROTASR.workflows.continue_limit_fits import adaptive_seed, verify_attempt
    key=point['model']+'/'+point['mass']; work=output/key
    with (work/'point.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='deferred_existing_worker')
        original=read_json(work/'state.json')
        if original['status']=='complete':
            verified_complete(point,original)
            return dict(status='existing_verified',output_sha256=original['output_sha256'])
        if (original['status']!='failed' or original.get('pid')
                or not original.get('workspace_valid')):
            raise ValueError('Requires terminal workspace-valid primary failure')
        target=auxiliary/key
        if target.exists():
            raise ValueError('Existing recovery retained; no automatic reset')
        history, protected=unfinished_history(point,output)
        accepted, logs=unfinished_saved(point,output,history)
        if accepted:
            return dict(status='existing_verified',**accepted)
        verify_card(point,digest)
        workspace=work/'workspace.root'
        if digest(workspace)!=original['workspace_sha256']:
            raise ValueError('Workspace changed')
        seed, range_evidence=adaptive_seed(logs)
        previous_calls=any(origin=='numerical_profile_calls2m' for origin,a in history)
        specs=unfinished_configs(seed,previous_calls)
        target.mkdir(parents=True)
        contract=dict(point=point,code=code,protected_states=protected,
            workspace_sha256=original['workspace_sha256'],attempts=specs,
            range_evidence=range_evidence,maximum_new_attempts=2,
            original_attempts_retained=True,physics_model_changed=False,
            primary_promotion=False,warnings_diagnostic_only=True,
            tolerance=.1,rRelAcc=.0025,rAbsAcc=.00001)
        write_json(target/'contract.json',contract)
        state=dict(status='running',point=point,pid=os.getpid(),attempts=[],
            started=time.time(),contract_sha256=sha256(target/'contract.json'))
        write_json(target/'state.json',state)
        runner=CommandRunner(target/'command_receipts')
        try:
            for spec in specs:
                here=target/spec['name'];(here/'tmp').mkdir(parents=True)
                product=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
                log=here/'combine.log';command=profile_recovery_command(workspace,point['mass'],spec)
                receipt=runner.run(command,here,log,[product],env=dict(env,TMPDIR=str(here/'tmp')))
                attempt=dict(spec,command=command,exit_code=receipt['exit_code'],
                    log=str(log.relative_to(ROOT)),log_sha256=sha256(log),
                    output=str(product.relative_to(ROOT)),
                    output_sha256=sha256(product) if product.is_file() else None)
                passed,evidence=verify_attempt(point,attempt)
                attempt.update(passed=passed,validation=evidence)
                for name,want in code.items():
                    if sha256(ROOT/name)!=want:raise ValueError('Active source changed')
                for name,want in protected.items():
                    if sha256(ROOT/name)!=want:raise ValueError('Predecessor state changed')
                verify_card(point,digest)
                if sha256(workspace)!=original['workspace_sha256']:raise ValueError('Workspace changed')
                write_json(here/'validation.json',attempt)
                state['attempts'].append(attempt);write_json(target/'state.json',state)
                if passed:
                    state.update(status='verified_not_promoted',accepted_attempt=attempt,
                        output=attempt['output'],output_sha256=attempt['output_sha256'],validation=evidence)
                    break
            else:
                state.update(status='failed',error='two_unfinished_recovery_configurations_exhausted')
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error))
            raise
        finally:
            state.update(pid=None,finished=time.time());write_json(target/'state.json',state)
        return dict(status=state['status'],state=str((target/'state.json').relative_to(ROOT)),
                    state_sha256=sha256(target/'state.json'))


def run_unfinished(manifest, output, workers):
    from TROTASR.workflows.continue_limit_fits import verify_assist_inputs, assist_capacity
    from TROTASR.workflows.nominal_campaign import resources
    if not 1 <= workers <= 99:raise ValueError('Invalid aggregate-constrained worker ceiling')
    manifest,output=internal_path(manifest),internal_path(output)
    auxiliary=output/'numerical_all_unfinished_20260930'
    auxiliary.mkdir(exist_ok=True)
    with (auxiliary/'controller.lock').open('a') as lock, (output/'controller.lock').open('r') as primary_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(primary_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (auxiliary/'contract.json').exists():raise ValueError('Frozen recovery already exists; no reset')
        digest=IntegrityCache();grid,primary=verify_assist_inputs(manifest,output,digest)
        points=[];preserved={}
        for point in grid['points']:
            key=point['model']+'/'+point['mass'];s=read_json(output/key/'state.json')
            if s['status']=='complete':
                verified_complete(point,s);preserved[key]=s['output_sha256']
            elif s['status']=='failed' and not s.get('pid'):
                points.append(point)
            else:raise ValueError('Primary campaign has not finished: '+key)
        code={name:sha256(ROOT/name) for name in SOURCES}
        contract=dict(manifest_sha256=sha256(manifest),primary_contract_sha256=sha256(output/'contract.json'),
            code=code,points=points,preserved_primary_successes=preserved,
            workers=workers,aggregate_slots=100,reserve_GiB=32,maximum_new_attempts_per_point=2,
            scope='all_current_model_unfinished',skip_any_saved_valid_result=True,
            source_histories=list(UNFINISHED_HISTORY),primary_promotion=False,physics_model_changed=False)
        write_json(auxiliary/'contract.json',contract)
        state=dict(status='running',pid=os.getpid(),started=time.time(),results={},
                   contract_sha256=sha256(auxiliary/'contract.json'),targets=len(points))
        env=runtime(auxiliary);pending=list(points);active={}
        path=auxiliary/'state.json'
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while pending or active:
                    for future in list(active):
                        if not future.done():continue
                        point=active.pop(future);key=point['model']+'/'+point['mass']
                        try:
                            result=future.result()
                            if result['status']=='deferred_existing_worker':
                                pending.append(point)
                            else:state['results'][key]=result
                        except Exception as exc:
                            state['results'][key]=dict(status='needs_attention',error=repr(exc))
                    external,memory=resources()
                    capacity=assist_capacity(workers,external,memory,len(active),0,0)
                    # Point locks prevent overlap with the draining old pool;
                    # CommandRunner also shares the global admission lock.
                    for _ in range(min(len(pending),max(0,capacity-len(active)))):
                        point=pending.pop(0)
                        active[pool.submit(recover_unfinished_point,point,output,auxiliary,digest,env,code)]=point
                    state.update(active=[p['model']+'/'+p['mass'] for p in active.values()],
                        pending=[p['model']+'/'+p['mass'] for p in pending],updated=time.time(),
                        admission_limit=capacity,available_memory_bytes=memory)
                    write_json(path,state)
                    if active:wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    # A point held by the old worker may be deferred immediately.
                    if pending:time.sleep(3)
            failed=[k for k,r in state['results'].items()
                    if r['status'] not in ('existing_verified','verified_not_promoted')]
            state.update(status='needs_attention' if failed else 'all_verified_pending_reconciliation',
                         failed=failed)
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error));raise
        finally:
            state.update(pid=None,finished=time.time());write_json(path,state)
    return 2 if failed else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profile-recovery',action='store_true')
    parser.add_argument('--recover-unfinished',action='store_true')
    parser.add_argument('--point',action='append',help='Exact model/mass key for a terminal adaptive failure')
    parser.add_argument('--max-calls',type=int,choices=[2000000])
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.recover_unfinished:
        if args.profile_recovery or args.point or args.max_calls is not None:
            parser.error('--recover-unfinished has its own fixed finite policy')
        raise SystemExit(run_unfinished(args.manifest,args.output,args.workers))
    if args.profile_recovery:
        if not args.point: parser.error('--profile-recovery requires --point')
        if len(args.point)>1:
            raise SystemExit(recover_profile_batch(args.manifest,args.output,args.point,args.workers,args.max_calls))
        raise SystemExit(recover_profile(args.manifest,args.output,args.point[0],args.max_calls))
    if args.point: parser.error('--point requires --profile-recovery')
    raise SystemExit(run(args.manifest,args.output))
