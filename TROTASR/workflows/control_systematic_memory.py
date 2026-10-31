"""Identity-checked resource handoffs; never change a physics contract."""
import argparse
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.normalization_recovery import process_identity


def kernel_state(identity):
    """Read the original process, including an unreaped child's exit status."""
    folder = Path('/proc')/str(identity['pid'])
    def checked_stat():
        if folder.stat().st_uid != os.getuid():
            raise RuntimeError('Process owner changed')
        fields = (folder/'stat').read_text().rsplit(') ', 1)[1].split()
        if int(fields[19]) != identity['start_ticks']:
            raise RuntimeError('Process PID was reused')
        return fields
    try:
        for _ in range(3):
            fields = checked_stat()
            if fields[0] == 'Z':
                return 'Z', int(fields[49])
            arguments = [a.decode() for a in (folder/'cmdline').read_bytes().split(b'\0') if a]
            fields = checked_stat()
            if fields[0] == 'Z':
                return 'Z', int(fields[49])
            if arguments == identity['arguments']:
                return fields[0], None
            if arguments:
                raise RuntimeError('Process command changed')
            # Linux can clear cmdline before publishing the zombie state.
            time.sleep(.01)
        return 'X', None  # Exit in flight; retain the slot, do not signal it.
    except (FileNotFoundError, ProcessLookupError):
        return None, None


def signal_exact(identity, sig):
    status, _ = kernel_state(identity)
    if status not in (None, 'Z', 'X'):
        try:
            os.kill(identity['pid'], sig)
        except ProcessLookupError:
            pass


def stop_exact(identity):
    signal_exact(identity, signal.SIGSTOP)
    for _ in range(100):
        if kernel_state(identity)[0] in ('T', 't', None, 'Z'):
            return
        time.sleep(.05)
    raise RuntimeError('Process did not suspend')


def terminate_controller(identity):
    signal_exact(identity, signal.SIGTERM)
    signal_exact(identity, signal.SIGCONT)
    for _ in range(100):
        if kernel_state(identity)[0] in (None, 'Z'):
            return
        time.sleep(.1)
    raise RuntimeError('Controller did not exit; no replacement submitted')


def completed_retained_output(record, status, exit_status, receipt, revision):
    """Accept an exact worker's output, retaining unknown orphan wait status."""
    observed_exit = status == 'Z' and exit_status == 0
    handed_off_exit = (status is None and exit_status is None
                      and record.get('controller_handoff') is not None)
    return bool((observed_exit or handed_off_exit)
        and receipt.get('identity') == record['identity']
        and receipt.get('status') == 'complete' and receipt.get('exit_code') == 0
        and receipt.get('payload_revision') == revision
        and sha256(ROOT/receipt['output']) == receipt['output_sha256'])


CPU_EXPANSION_20261026 = dict(id='hep2_110_slots_20261026', user_approved=True,
    aggregate_user_slots=110, downstream_slots=106,
    component_targets={'cms_trota_jme': 85, 'stored_trota_nonjme': 25})

CPU_EXPANSION_20261027 = dict(id='hep2_120_slots_20261027', user_approved=True,
    aggregate_user_slots=120, downstream_slots=116,
    component_targets={'cms_trota_jme': 85, 'stored_trota_nonjme': 35},
    borrow_after_peer_drained=True,
    peer_campaigns={'cms_trota_jme': 'campaigns/early_nonjme_optimized_20261025',
                    'stored_trota_nonjme': 'campaigns/early_jme_optimized_20261025'})


def retuned_plan(plan, chunk_size, retained, source_hashes, evidence, resource_policy=None):
    """Execution-only 2024 settings; the analysis and input contracts stay exact."""
    expected = {'cms_trota_jme': 1000, 'stored_trota_nonjme': 25000}
    if chunk_size != expected.get(plan.get('shape_mode')):
        raise ValueError('Require the adopted 2024 component chunk size')
    result = copy.deepcopy(plan)
    if resource_policy is not None:
        if resource_policy == CPU_EXPANSION_20261026:
            valid = (plan['aggregate_slots'] == 96 and plan['max_workers'] ==
                     {'cms_trota_jme': 75, 'stored_trota_nonjme': 25}[plan['shape_mode']])
        elif resource_policy == CPU_EXPANSION_20261027:
            valid = (plan['aggregate_slots'] == 106 and plan['max_workers'] ==
                     CPU_EXPANSION_20261026['component_targets'][plan['shape_mode']]
                     and plan.get('resource_allocation') == CPU_EXPANSION_20261026)
        else:
            valid = False
        if not valid:
            raise ValueError('Require an exact approved, not-yet-applied slot expansion')
        result.update(aggregate_slots=resource_policy['downstream_slots'],
                      max_workers=resource_policy['component_targets'][plan['shape_mode']],
                      resource_allocation=copy.deepcopy(resource_policy))
    result.update(chunk_size=chunk_size, admission_bytes_per_worker=1024**3,
                  runner_sha256=source_hashes['workflows/nominal_campaign.py'])
    policy = result['memory_control_policy']
    policy.update(budget_mode='adaptive_peak_growth', initial_workers=result['max_workers'],
                  admission_batch=8, retained_workers=retained)
    policy.pop('post_collection_stages', None)
    policy['workflow_sha256']['workflows/campaign_resources.py'] = source_hashes['workflows/campaign_resources.py']
    result['resource_retune'] = evidence
    return result


def retune_campaigns(request_path):
    """Replace only admission controllers; all in-flight workers finish in place."""
    from TROTASR.workflows import nominal_campaign as nc
    from TROTASR.utils.paths import internal_path
    request_path = internal_path(request_path)
    request = read_json(request_path)
    folder = request_path.parent
    report = folder/'handoff.json'
    sources = {'workflows/nominal_campaign.py', 'workflows/campaign_resources.py'}
    if set(request['sources']) != sources or report.exists():
        raise ValueError('Unexpected source set or an existing handoff receipt')
    for item in request['validation']:
        if sha256(ROOT/item['path']) != item['sha256'] or read_json(ROOT/item['path'])['status'] != 'passed':
            raise ValueError('Missing exact successful validation')
    for name, item in request['sources'].items():
        if sha256(ROOT/name) != item['before'] or sha256(ROOT/item['staged']) != item['after']:
            raise ValueError('Execution source changed before handoff: '+name)
    plans, owners, launches = {}, {}, {}
    for item in request['campaigns']:
        campaign = internal_path(ROOT/item['path'])
        if sha256(campaign/'plan.json') != item['plan_sha256']:
            raise ValueError('Campaign plan changed before handoff')
        plans[campaign] = nc.frozen_plan(campaign)
        launches[campaign] = read_json(campaign/'launch.json')
        owners[campaign] = launches[campaign]['identity']
        if (process_identity(owners[campaign]['pid']) != owners[campaign]
                or read_json(campaign/'state.json')['pid'] != owners[campaign]['pid']):
            raise RuntimeError('Controller identity differs from exact launch')
        retuned_plan(plans[campaign], item['chunk_size'], {},
                     {k:v['after'] for k,v in request['sources'].items()}, {}, request.get('resource_policy'))
    audit = dict(status='suspending_admission', started=time.time(),
                 request_sha256=sha256(request_path), controllers={}, preserved={}, retained={})
    write_json(report, audit)
    terminated = False
    try:
        for owner in owners.values():
            stop_exact(owner)
        for campaign, plan in plans.items():
            keypath = str(campaign.relative_to(ROOT))
            retained, preserved = {}, {}
            tasks = {t['key']:t for t in plan['tasks']}
            for path in (campaign/'state').glob('*.json'):
                state = read_json(path); key = path.stem
                if key not in tasks:
                    raise ValueError('Unexpected task state')
                if state['status'] == 'complete':
                    if sha256(ROOT/state['output']) != state['output_sha256']:
                        raise ValueError('Completed output changed: '+key)
                    preserved[key] = dict(output=state['output'],sha256=state['output_sha256'],
                                          state_sha256=sha256(path))
                elif state['status'] == 'running':
                    identity = process_identity(state['pid'])
                    history = campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
                    if identity is None:
                        result = read_json(history)
                        if result['status'] == 'complete' and sha256(ROOT/result['output']) != result['output_sha256']:
                            raise ValueError('Exited worker output changed')
                        write_json(path,result)
                        continue
                    expected = [str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',
                                str(campaign),'--key',key,'--attempt',str(state['attempt'])]
                    if identity['arguments'][-len(expected):] != expected:
                        raise ValueError('Worker command identity mismatch: '+key)
                    retained[key] = dict(pid=state['pid'],attempt=state['attempt'],identity=identity,
                                         history=str(history.relative_to(ROOT)))
            audit['retained'][keypath] = retained
            audit['preserved'][keypath] = preserved
            audit['controllers'][keypath] = dict(previous=owners[campaign])
            registered = {record['pid'] for record in retained.values()}
            # Refuse a Popen/state-registration race; resume the old scheduler.
            for process in Path('/proc').glob('[0-9]*'):
                try:
                    fields=(process/'stat').read_text().rsplit(') ',1)[1].split()
                    if (int(fields[1])==owners[campaign]['pid'] and fields[0]!='Z'
                            and int(process.name) not in registered):
                        raise RuntimeError('Controller has an unregistered live child')
                except (FileNotFoundError,ProcessLookupError):
                    continue
        audit['status']='retaining_workers';write_json(report,audit)
        for owner in owners.values():
            terminate_controller(owner)
        terminated = True
        with contextlib.ExitStack() as stack:
            for campaign in plans:
                lock=stack.enter_context((campaign/'controller.lock').open('a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for name,item in request['sources'].items():
                target=ROOT/name;backup=folder/'before'/name
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
                temporary=target.with_name(target.name+'.retune.'+str(os.getpid()))
                shutil.copy2(ROOT/item['staged'],temporary);os.replace(temporary,target)
                if sha256(target)!=item['after']:raise ValueError('Execution deployment hash mismatch')
            for item in request['campaigns']:
                campaign=ROOT/item['path'];keypath=item['path']
                backup=folder/'before'/campaign.name;backup.mkdir(parents=True,exist_ok=True)
                for name in ('plan.json','launch.json','state.json'):
                    shutil.copy2(campaign/name,backup/name)
                evidence=dict(request=str(request_path.relative_to(ROOT)),request_sha256=sha256(request_path),
                              previous_plan=str((backup/'plan.json').relative_to(ROOT)),
                              previous_plan_sha256=item['plan_sha256'],physics_changes=False,
                              retained_chunk_size=plans[campaign]['chunk_size'],new_chunk_size=item['chunk_size'])
                updated=retuned_plan(plans[campaign],item['chunk_size'],audit['retained'][keypath],
                                    {k:v['after'] for k,v in request['sources'].items()},evidence,
                                    request.get('resource_policy'))
                write_json(campaign/'plan.json',updated)
                nc.frozen_plan(campaign)
        audit['status']='launching_controllers';write_json(report,audit)
        for campaign in plans:
            keypath=str(campaign.relative_to(ROOT));command=launches[campaign]['command']
            log=folder/(campaign.name+'.controller.log')
            with log.open('x') as stream:
                proc=subprocess.Popen(command,cwd=ROOT.parent,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            identity=None
            for _ in range(100):
                identity=process_identity(proc.pid)
                if (identity and Path(identity['arguments'][0]).name.startswith('python')
                        and identity['arguments'][-len(command[2:]):]==command[2:]):break
                if proc.poll() is not None:raise RuntimeError('Replacement controller exited')
                time.sleep(.1)
            else:raise RuntimeError('Replacement did not enter the pinned Python runtime')
            launch=dict(command=command,pid=proc.pid,identity=identity,started=time.time(),
                        previous_launch=str((folder/'before'/campaign.name/'launch.json').relative_to(ROOT)),
                        plan_sha256=sha256(campaign/'plan.json'),resource_handoff=str(report.relative_to(ROOT)))
            write_json(campaign/'launch.json',launch)
            audit['controllers'][keypath].update(replacement=identity,plan_sha256=launch['plan_sha256'])
            write_json(report,audit)
        audit.update(status='handed_off',finished=time.time(),workers_restarted=0,physics_changes=False)
        write_json(report,audit)
        print(json.dumps(dict(status=audit['status'],retained={k:len(v) for k,v in audit['retained'].items()},
                              completed_preserved={k:len(v) for k,v in audit['preserved'].items()})))
    except BaseException as error:
        if not terminated:
            for owner in owners.values():signal_exact(owner,signal.SIGCONT)
        audit.update(status='needs_attention',error=repr(error),finished=time.time())
        write_json(report,audit)
        raise


def live_worker_cap(workers, resume=False, refill=False, resize=False, retry_runtime=False):
    """Drain current work at the requested cap, then restart only controllers.

    Frozen compute sources stay byte-identical. Suspended workers keep their
    in-memory progress and are resumed, never killed or submitted a second time.
    """
    if not 10 <= workers <= 100:
        raise ValueError('Expected 10--100 compute workers')
    if resize and not resume:
        raise ValueError('Resizing requires the existing retained-worker receipt')
    if retry_runtime and not resume:
        raise ValueError('Runtime recovery requires the existing worker receipt')
    campaign = ROOT/'campaigns/nonjme_20260927'
    path = campaign/'worker_control.json'
    revision = 'object_sf_update_20261018'
    with (campaign/'worker_control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and not resume:
            raise RuntimeError('Existing worker-control receipt retained; inspect before reuse')
        if resume and not path.exists():
            raise RuntimeError('No resource handoff exists to resume')
        chain = read_json(campaign/'chain_state.json')
        if (chain.get('payload_revision') != revision or set(chain['stages']) != {'jme', 'nonjme'}
                or chain['status'] != 'running'):
            raise RuntimeError('Require the waiting two-component update')
        owners = {'chain': chain['identity']}
        components = {}
        for name, folder in [('jme', 'systematics_20260924'), ('nonjme', 'nonjme_20260927')]:
            component = ROOT/'campaigns'/folder
            spec = read_json(component/'plan.json')['payload_refresh']
            state = read_json(component/'state.json')
            if spec['id'] != revision or state['failed'] or state['status'] != 'payload_refresh_running':
                raise RuntimeError('Component is not a healthy active update')
            if 'workflows/control_systematic_memory.py' in spec['sources']:
                raise RuntimeError('Resource controller unexpectedly frozen in compute contract')
            for source, digest in spec['sources'].items():
                if sha256(ROOT/source) != digest:
                    raise RuntimeError('Compute source mismatch: '+source)
            owners[name] = chain['stages'][name]['identity']
            if state['pid'] != owners[name]['pid']:
                raise RuntimeError('Component controller changed')
            components[name] = str(component)
        for identity in owners.values():
            if process_identity(identity['pid']) != identity:
                raise RuntimeError('Controller identity no longer matches')
        if resume:
            audit = read_json(path)
            if (audit.get('status') not in ('needs_attention', 'draining_at_cap', 'refilling_at_cap')
                    or audit.get('payload_revision') != revision
                    or (audit.get('worker_cap') != workers and not resize)
                    or audit.get('owners') != owners or audit.get('components') != components
                    or audit.get('original_chain') != chain or not audit.get('retained')
                    or audit.get('replacement_controllers')
                    or (audit.get('failed') and not retry_runtime)):
                raise RuntimeError('Receipt is not an interrupted retained-worker drain')
            if process_identity(audit['identity']['pid']) == audit['identity']:
                raise RuntimeError('Previous resource controller is still alive')
            for identity in owners.values():
                if kernel_state(identity)[0] not in ('T', 't'):
                    raise RuntimeError('Original admission controller is no longer suspended')
            runtime_failures = []
            for record in audit['retained']:
                status, wait_status = kernel_state(record['identity'])
                receipt = read_json(Path(components[record['component']])/'state'/(record['key']+'.json'))
                if (receipt.get('identity') != record['identity']
                        or receipt.get('payload_revision') != revision):
                    raise RuntimeError('Retained worker receipt changed: '+record['key'])
                if record.get('finished') and not record.get('verified'):
                    if (not retry_runtime or status not in (None, 'Z')
                            or wait_status not in (None, 256)
                            or record.get('kernel_wait_status') != 256
                            or receipt.get('status') != 'failed' or receipt.get('exit_code') != 1
                            or receipt.get('exception_type') != 'RuntimeError'
                            or receipt.get('error') != "the diagonal-v3 schema requires the pinned 'mt2' package"
                            or record['component']+'/'+record['key'] not in audit['failed']):
                        raise RuntimeError('Not the inspected missing-runtime failure: '+record['key'])
                    runtime_failures.append(record)
                elif record.get('finished') and status not in (None, 'Z'):
                    raise RuntimeError('Completed retained worker changed: '+record['key'])
                if not record.get('finished'):
                    record['controller_handoff'] = dict(time=time.time(),
                        previous_controller=audit['identity'], worker_identity=record['identity'])
            audit.setdefault('resumptions', []).append(dict(time=time.time(),
                previous_identity=audit['identity'], previous_error=audit.get('error'),
                previous_status=audit['status'], previous_worker_cap=audit['worker_cap'],
                worker_cap=workers, explicit_resize=resize))
            audit.update(status='draining_at_cap', pid=os.getpid(),
                         identity=process_identity(os.getpid()), worker_cap=workers)
            audit.pop('error', None)
            if runtime_failures:
                # Fail before changing any receipt if the original runtime is
                # not available. Recovery never installs a substitute package.
                import mt2
                if not str(Path(mt2.__file__).resolve()).startswith(
                        '/hep2-scratch/twkim/envs/trotasr_inference_py39/'):
                    raise RuntimeError('Require the original pinned mt2 runtime')
                recovery = 'worker_cap_100_runtime_20261020'
                from TROTASR.workflows.run_systematic_campaign import preserve_before_refresh
                for record in runtime_failures:
                    component = Path(components[record['component']]); key = record['key']
                    spec = read_json(component/'plan.json')['payload_refresh']
                    for previous in spec['previous_outputs'][key]:
                        output = ROOT/previous['output']
                        saved = component/'history'/revision/output.relative_to(component)
                        if not output.exists():
                            if sha256(saved) != previous['sha256']:
                                raise RuntimeError('Prior histogram backup changed: '+key)
                            shutil.copy2(saved, output)
                        if sha256(output) != previous['sha256']:
                            raise RuntimeError('Do not overwrite an existing output: '+key)
                    for folder, suffix in [('state', '.json'), ('logs', '.log')]:
                        preserve_before_refresh(component, component/folder/(key+suffix),
                            move=True, revision=recovery, update_id=revision)
                audit.setdefault('runtime_retries', []).append(dict(time=time.time(),
                    reason='Original pinned Python runtime restored; retry only missing-mt2 failures',
                    records=runtime_failures))
                audit['retained'] = [r for r in audit['retained'] if r not in runtime_failures]
                audit['failed'] = [k for k in audit['failed'] if k not in
                    {r['component']+'/'+r['key'] for r in runtime_failures}]
        else:
            audit = dict(status='suspending_admission', pid=os.getpid(), identity=process_identity(os.getpid()),
                payload_revision=revision, worker_cap=workers, created=time.time(), owners=owners,
                original_chain=chain, components=components, retained=[], completed=[], failed=[],
                inputs_modified=0, completed_outputs_deleted=0, workers_killed=0)
        def save():
            audit['updated'] = time.time()
            write_json(path, audit)
        refill = refill or audit.get('refill', False)
        audit['refill'] = refill
        pending = {name: [] for name in components}
        children = {}
        if refill:
            from TROTASR.workflows.run_systematic_campaign import refresh_task_required, preserve_before_refresh
            known = {(r['component'], r['key']) for r in audit['retained']}
            for name, component in components.items():
                spec = read_json(Path(component)/'plan.json')['payload_refresh']
                for task in spec['tasks']:
                    if not refresh_task_required(task) or (name, task['key']) in known:
                        continue
                    receipt_path = Path(component)/'state'/(task['key']+'.json')
                    receipt = read_json(receipt_path) if receipt_path.exists() else {}
                    if receipt.get('payload_revision') == revision:
                        if receipt.get('status') == 'complete' and receipt.get('exit_code') == 0:
                            continue
                        raise RuntimeError('Unregistered current worker or failure: '+task['key'])
                    pending[name].append(task)
                pending[name].sort(key=lambda t: (t['entries']==0, t['entries'], t['key']))
        save()
        try:
            # Stop admission before enumerating children; do not signal a process group.
            for name in ('chain', 'jme', 'nonjme'):
                stop_exact(owners[name])
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    if proc.stat().st_uid != os.getuid():
                        continue
                    fields = (proc/'stat').read_text().rsplit(') ', 1)[1].split()
                    name = next((n for n in components if owners[n]['pid'] == int(fields[1])), None)
                    if name is None or fields[0] == 'Z':
                        continue
                    identity = process_identity(int(proc.name))
                    if identity is None:
                        continue
                    args = identity['arguments']
                    expected = [str(ROOT/'workflows/run_systematic_campaign.py'), '--campaign',
                                components[name], '--refresh-worker']
                    if args[-5:-1] != expected:
                        raise RuntimeError('Unexpected controller child: '+proc.name)
                    key = args[-1]
                    state = read_json(Path(components[name])/'state'/(key+'.json'))
                    if state.get('identity') != identity or state.get('payload_revision') != revision:
                        raise RuntimeError('Worker receipt mismatch: '+key)
                    if resume:
                        if not any(r['component'] == name and r['key'] == key and r['identity'] == identity
                                   for r in audit['retained']):
                            raise RuntimeError('Unregistered controller child: '+proc.name)
                    else:
                        audit['retained'].append(dict(component=name, key=key, identity=identity,
                                                      started=state['started']))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            audit['retained'].sort(key=lambda r: (r['started'], r['key']))
            if not resume:
                audit['before_workers'] = len(audit['retained'])
            save()
            # Pause newest work first. No partial output or progress is discarded.
            audit['status'] = 'limiting_workers'; save()
            if not resume:
                for record in audit['retained'][workers:]:
                    stop_exact(record['identity'])
            audit['status'] = 'draining_at_cap'
            save()
            while True:
                live = []
                for record in audit['retained']:
                    if record.get('finished'):
                        continue
                    status, exit_status = kernel_state(record['identity'])
                    if status not in (None, 'Z'):
                        live.append((record, status))
                        continue
                    receipt = read_json(Path(components[record['component']])/'state'/(record['key']+'.json'))
                    valid = completed_retained_output(record, status, exit_status, receipt, revision)
                    record.update(finished=time.time(), kernel_wait_status=exit_status, verified=valid,
                        completion_evidence=('kernel_exit_and_worker_receipt' if exit_status is not None
                            else 'handed_off_worker_receipt_and_output'))
                    audit['completed' if valid else 'failed'].append(record['component']+'/'+record['key'])
                    child = children.pop((record['component'], record['key']), None)
                    if child is not None:
                        child[0].wait(); child[1].close()
                running = [(r, s) for r, s in live if s not in ('T', 't')]
                if len(running) > workers:
                    for r, _ in running[workers:]:
                        stop_exact(r['identity'])
                slots = max(0, workers-min(workers, len(running)))
                memory = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                              if s.startswith('MemAvailable:'))
                if memory < 24*1024**3:
                    for r, s in live:
                        if s not in ('T', 't'):
                            stop_exact(r['identity'])
                elif memory >= 32*1024**3:
                    for r, s in live:
                        if slots and s in ('T', 't'):
                            signal_exact(r['identity'], signal.SIGCONT)
                            slots -= 1
                statuses = [kernel_state(r['identity'])[0] for r, _ in live]
                if refill and memory >= 32*1024**3 and not audit['failed']:
                    from TROTASR.workflows.nominal_campaign import THREAD_ENV
                    targets = {'jme': workers-max(4, workers//4), 'nonjme': max(4, workers//4)}
                    active_counts = {n: sum(r['component']==n and s not in (None,'Z')
                                          for (r,_),s in zip(live,statuses)) for n in components}
                    raw = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','stat=,comm='],text=True)
                    account_slots = sum(not state.startswith('Z') and
                        (comm.startswith(('python','root','combine','xrdcp','rsync')) or comm in ('hadd','cp'))
                        for state,comm in (line.split(None,1) for line in raw.splitlines()))
                    free = min(workers-sum(active_counts.values()), max(0,100-account_slots),
                               max(0, (memory-32*1024**3)//(4*1024**3)))
                    for _ in range(min(8, free)):
                        choices = [n for n in components if pending[n]]
                        if not choices: break
                        name = max(choices, key=lambda n: targets[n]-active_counts[n])
                        task = pending[name][0]; key = task['key']; component = Path(components[name])
                        receipt_path = component/'state'/(key+'.json')
                        if receipt_path.exists():
                            preserve_before_refresh(component, receipt_path, update_id=revision)
                        logfile = component/'logs'/(key+'.log')
                        if logfile.exists():
                            preserve_before_refresh(component, logfile, move=True, update_id=revision)
                        command = [owners[name]['arguments'][0], '-u',
                            str(ROOT/'workflows/run_systematic_campaign.py'), '--campaign',
                            str(component), '--refresh-worker', key]
                        stream = logfile.open('x')
                        proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                        identity = process_identity(proc.pid)
                        receipt = dict(status='running',key=key,pid=proc.pid,identity=identity,
                            command=command,started=time.time(),payload_revision=revision)
                        write_json(receipt_path, receipt)
                        audit['retained'].append(dict(component=name,key=key,identity=identity,
                            started=receipt['started'],admitted_by_resource_controller=True))
                        children[(name,key)] = (proc,stream)
                        pending[name].pop(0); active_counts[name] += 1
                        statuses.append('R')
                        save()
                    audit['pending'] = {n:len(v) for n,v in pending.items()}
                    audit['status'] = 'refilling_at_cap'
                audit.update(running=sum(s not in (None, 'Z', 'T', 't') for s in statuses),
                    suspended=sum(s in ('T', 't') for s in statuses), memory_available_bytes=memory)
                save()
                if not live and not children and not any(pending.values()):
                    break
                time.sleep(2)
            if audit['failed']:
                raise RuntimeError('Retained worker failure; outputs retained, no duplicate submission')
            audit['status'] = 'controller_handoff'; save()
            for name in ('chain', 'jme', 'nonjme'):
                terminate_controller(owners[name])
            for name in components:
                with (Path(components[name])/'controller.lock').open('r') as controller_lock:
                    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The original controller validates/reuses every completed file.
            # The downstream chain reserves four of the aggregate 100 slots
            # for its controllers; the refill loop above counts them live.
            downstream_workers = min(workers, 96)
            counts = {'jme': downstream_workers-max(4, downstream_workers//4),
                      'nonjme': max(4, downstream_workers//4)}
            audit['downstream_worker_budget'] = downstream_workers
            new_chain = dict(chain, worker_budget=downstream_workers, resource_handoff=path.name)
            new_chain['stages'] = {n: dict(s) for n, s in chain['stages'].items()}
            for name, count in counts.items():
                command = list(owners[name]['arguments'])
                command[command.index('--workers')+1] = str(count)
                logfile = campaign/'logs'/('worker_cap_%s_%s.log'%(workers, name))
                with logfile.open('x') as stream:
                    proc = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(proc.pid)
                new_chain['stages'][name].update(pid=proc.pid, identity=identity,
                    command=command[2:], started=time.time(), previous_controller_identity=owners[name])
                audit.setdefault('replacement_controllers', {})[name] = identity
                save()
            write_json(campaign/'chain_state.json', new_chain)
            command = ['bash', str(ROOT/'workflows/run_inference.sh'),
                str(ROOT/'workflows/run_nominal_chain.py'), '--campaign', str(campaign),
                '--update-years', '2024', '2025', '--workers', str(downstream_workers), '--resume-update']
            logfile = campaign/'logs'/('worker_cap_%s_chain.log'%workers)
            with logfile.open('x') as stream:
                proc = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            audit.update(status='handed_off', replacement_chain_pid=proc.pid, finished=time.time())
            save()
        except BaseException as error:
            if audit['status'] == 'suspending_admission':
                # Before any worker was paused, restore original admission on
                # a discovery race instead of leaving a partial handoff behind.
                for identity in owners.values():
                    signal_exact(identity, signal.SIGCONT)
            audit.update(status='needs_attention', error=repr(error)); save()
            raise


def legacy_reduction():
    campaign = ROOT/'campaigns/systematics_20260924'
    report = campaign/'memory_control.json'
    if report.exists():
        raise RuntimeError('Reduction already recorded; inspect before another action')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = read_json(campaign/'controller.json')
        ident = process_identity(controller['pid'])
        if ident and ident['start_ticks'] == controller['start_ticks']:
            raise RuntimeError('Controller still alive; no worker changes made')
        states = {p.stem: read_json(p) for p in (campaign/'state').glob('*.json')}
        active = []
        for key, state in states.items():
            if state['status'] != 'running':
                continue
            ident = process_identity(state['pid'])
            if not ident:
                continue
            expected = [str(ROOT/'workflows/nominal_campaign.py'), 'worker', '--campaign',
                        str(campaign), '--key', key, '--attempt', str(state['attempt'])]
            if ident['arguments'][-len(expected):] != expected:
                raise RuntimeError('Unexpected worker identity: '+key)
            active.append(dict(key=key, state=state, identity=ident))
        # Preserve the oldest work first, with corrected second attempts first.
        active.sort(key=lambda r: (-r['state']['attempt'], r['state']['started'], r['key']))
        audit = dict(status='reducing', started=time.time(), target_workers=32,
                     worker_cap=50, previous_controller=controller,
                     before_active=len(active), retained=active[:32], interrupted=[],
                     plan_before=read_json(campaign/'plan.json'), states_before=states,
                     completed_outputs_deleted=0, inputs_modified=0)
        write_json(report, audit)
        for record in active[32:]:
            ident = record['identity']
            if process_identity(ident['pid']) != ident:
                continue
            receipt = campaign/'history'/(record['key']+'.attempt%s.json'%record['state']['attempt'])
            if receipt.exists() and read_json(receipt).get('status') == 'complete':
                continue
            os.kill(ident['pid'], signal.SIGTERM)
            record['signal'] = 'SIGTERM'
            record['time'] = time.time()
            audit['interrupted'].append(record)
            write_json(report, audit)
        for _ in range(100):
            alive = [r for r in audit['interrupted'] if process_identity(r['identity']['pid']) == r['identity']]
            if not alive:
                break
            time.sleep(.1)
        audit.update(status='reduced' if not alive else 'termination_pending', finished=time.time(),
                     interrupted_still_alive=[r['key'] for r in alive])
        write_json(report, audit)
        print(json.dumps({k: audit[k] for k in ('status','before_active','target_workers','worker_cap',
                                               'interrupted_still_alive')}))
        print('Interrupted unfinished workers:', len(audit['interrupted']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retune-campaigns', type=Path,
                        help='Apply an exact validated resource-only controller handoff request')
    parser.add_argument('--live-worker-cap', type=int,
                        help='Preserve current work with SIGSTOP/SIGCONT and cap the canonical continuation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted resource drain from its existing receipt')
    parser.add_argument('--refill', action='store_true',
                        help='Fill free slots immediately using the existing canonical file worker')
    parser.add_argument('--resize', action='store_true',
                        help='Explicitly change the resumed worker cap, preserving retained work')
    parser.add_argument('--retry-runtime-failures', action='store_true',
                        help='Resume inspected missing-mt2 failures using the original pinned runtime')
    args = parser.parse_args()
    if args.retune_campaigns is not None:
        if args.live_worker_cap is not None or args.resume or args.refill or args.resize or args.retry_runtime_failures:
            parser.error('--retune-campaigns is a separate exact handoff action')
        retune_campaigns(args.retune_campaigns)
    elif args.live_worker_cap is not None:
        live_worker_cap(args.live_worker_cap, args.resume, args.refill, args.resize,
                        args.retry_runtime_failures)
    else:
        if args.resume or args.resize or args.retry_runtime_failures:
            parser.error('--resume/--resize/--retry-runtime-failures requires --live-worker-cap')
        legacy_reduction()
