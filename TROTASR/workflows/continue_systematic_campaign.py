"""Promote the validated input policies and retain already-running workers.

Only the exact old controller is stopped; its worker processes and completed
outputs are preserved. Frozen previous plan/source hashes remain in history.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
if __package__ in (None,''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.nominal_campaign import current_contract,fingerprint,verify_compiled
from TROTASR.workflows.normalization_recovery import process_identity,retained_worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--negative-pt-selection',action='store_true')
    p.add_argument('--nonjme-extension',action='store_true')
    p.add_argument('--object-response-audit',action='store_true')
    p.add_argument('--nonjme-throughput',action='store_true')
    p.add_argument('--statistics-adapter',action='store_true')
    p.add_argument('--endpoint-execution',action='store_true')
    p.add_argument('--recover-retained',action='store_true',
                   help='Verify and adopt existing terminal retained outputs; never rerun events')
    p.add_argument('--evidence',type=Path,
                   help='Relocated, byte-identical endpoint equivalence evidence')
    args=p.parse_args();campaign=internal_path(args.campaign)
    if args.recover_retained:
        if args.evidence is None:
            p.error('--recover-retained requires --evidence')
        return recover_retained(campaign,internal_path(args.evidence),args.apply)
    if args.endpoint_execution:
        return apply_nonjme_throughput(campaign,args.apply,endpoint_execution=True)
    if args.statistics_adapter:
        return apply_nonjme_throughput(campaign,args.apply,statistics_adapter=True)
    if args.nonjme_throughput:
        return apply_nonjme_throughput(campaign,args.apply)
    if args.object_response_audit:
        return apply_object_response_update(args.apply)
    if args.nonjme_extension:
        return apply_nonjme_extension(campaign,args.apply)
    if args.negative_pt_selection:
        return apply_negative_pt_selection(campaign,args.apply)
    folder=ROOT/'validation/jme_nominal_20260924';stage=folder/'candidate_sources'
    evidence=read_json(folder/'policy_validation.json')
    workflow=read_json(folder/'workflow_validation.json')
    plan=read_json(campaign/'plan.json');status=read_json(campaign/'state.json')
    if plan.get('systematic_input_policy_update'):raise ValueError('Already applied; do not duplicate controller')
    if (evidence.get('status')!='passed' or evidence.get('unit_tests',0)<22
        or workflow.get('status')!='passed' or workflow.get('unit_tests',0)<10):
        raise ValueError('Missing actual code/workflow validation')
    if plan.get('shape_mode')!='cms_trota_jme' or plan.get('normalization_extension_recovery'):
        raise ValueError('Wrong campaign')
    verify_compiled()
    expected_sources={**evidence['tested_sources'],**workflow['tested_sources']}
    for name,digest in expected_sources.items():
        if sha256(stage/name)!=digest:raise ValueError('Tested candidate changed: '+name)
    for name,digest in evidence['previous_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Original source changed: '+name)
    if sha256(ROOT/'workflows/nominal_campaign.py')!=plan['runner_sha256']:
        raise ValueError('Original runner changed')
    identity=process_identity(status.get('pid'))
    controller=read_json(campaign/'controller.json')
    controller_ticks=controller.get('start_ticks',controller.get('identity',{}).get('start_ticks'))
    if identity and (identity['pid']!=controller['pid'] or
                     identity['start_ticks']!=controller_ticks):
        raise ValueError('Controller identity differs from launch receipt')
    if identity and identity['arguments'][-3:]!=[str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign)]:
        raise ValueError('Unexpected controller arguments')
    if not args.apply:
        print(dict(status='ready',controller=identity,active=status.get('active'),completed=status['completed']),flush=True)
        return
    checkpoint=campaign/'history/input_policy_20260924'
    checkpoint.mkdir(exist_ok=False)
    for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                   *[ROOT/n for n in expected_sources if (ROOT/n).exists()]]:
        target=checkpoint/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    # Stop only this controller, never its process group or active children.
    if identity:
        if process_identity(identity['pid'])!=identity:raise ValueError('Controller changed before handoff')
        os.kill(identity['pid'],signal.SIGTERM)
        for _ in range(50):
            if process_identity(identity['pid']) is None:break
            time.sleep(.1)
        else:raise RuntimeError('Controller did not exit; no source was deployed')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read_json(campaign/'plan.json')!=plan:raise ValueError('Concurrent plan mutation')
        states={p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        retained={}
        original=[]
        for key,s in states.items():
            if s['status']=='running':
                if process_identity(s['pid']):
                    retained[key]=retained_worker(campaign,key,s)
                else:
                    receipt=campaign/'history'/(key+'.attempt%s.json'%s['attempt'])
                    if not receipt.exists():raise ValueError('Missing original worker receipt: '+key)
                    s=read_json(receipt);states[key]=s;write_json(campaign/'state'/(key+'.json'),s)
            if s['status'] in ('failed','running'):original.append(key)
        for t in plan['tasks']:
            if fingerprint(ROOT/t['input'])!=t['fingerprint']:raise ValueError('Input changed: '+t['key'])
            s=states.get(t['key'],{})
            if s.get('status')=='complete' and sha256(ROOT/s['output'])!=s['output_sha256']:
                raise ValueError('Successful output changed: '+t['key'])
        for name in expected_sources:
            target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,temporary);os.replace(temporary,target)
        # No ids/corrections dependency changed; assert their compiled hashes.
        verify_compiled()
        new=copy.deepcopy(plan)
        new['contracts']={str(y):current_contract(y,'cms_trota_jme') for y in (2024,2025)}
        new['runner_sha256']=sha256(ROOT/'workflows/nominal_campaign.py')
        oldplan=checkpoint/(campaign/'plan.json').relative_to(ROOT)
        new['systematic_input_policy_update']=dict(
            validation=str((folder/'policy_validation.json').relative_to(ROOT)),
            validation_sha256=sha256(folder/'policy_validation.json'),
            previous_plan=str(oldplan.relative_to(ROOT)),previous_plan_sha256=sha256(oldplan),
            prior_contracts=plan['contracts'],original_attempt_keys=sorted(original),retained_workers=retained,
            workflow_sha256=workflow['tested_sources'],successful_outputs_rewritten=0)
        # Validate the bridge before installing the amended frozen plan.
        from TROTASR.workflows.systematic_continuation import prior_contracts,retry_keys
        prior_contracts(new)
        write_json(campaign/'plan.json',new)
        write_json(campaign/'input_policy_update.json',dict(status='applied_pending_controller_start',
            time=time.time(),previous_controller=identity,retained_workers=len(retained),
            successful_outputs_preserved=sum(s['status']=='complete' for s in states.values()),
            retry_files=retry_keys(new,states),plan_sha256=sha256(campaign/'plan.json'),
            validation=new['systematic_input_policy_update']['validation'],full_workflow_complete=False))
    command=['bash',str(ROOT/'workflows/run_inference.sh'),str(ROOT/'workflows/run_systematic_campaign.py'),
             '--campaign',str(campaign)]
    log=campaign/'logs/controller.input_policy_20260924.log'
    with log.open('a') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    ident=process_identity(proc.pid)
    if ident is None:raise RuntimeError('New controller exited at launch; inspect log')
    write_json(campaign/'controller.json',dict(pid=proc.pid,start_ticks=ident['start_ticks'],
        command=command,started=time.time(),max_workers=new['max_workers'],aggregate_slots=100,
        monitoring_automation='PAUSED',runner_sha256=new['runner_sha256'],log=str(log.relative_to(ROOT))))
    receipt=read_json(campaign/'input_policy_update.json')
    receipt.update(status='controller_started',controller_pid=proc.pid,controller_start_ticks=ident['start_ticks'])
    write_json(campaign/'input_policy_update.json',receipt)
    print(dict(status='controller_started',pid=proc.pid,retained_workers=len(retained),log=str(log)),flush=True)


def apply_negative_pt_selection(campaign, apply):
    """One bounded handoff; only remove the redundant finite-negative file guard."""
    folder=ROOT/'validation/jme_nominal_20260924'
    stage=folder/'selection_sources'
    validation=folder/'negative_pt_validation.json'
    evidence=read_json(validation)
    if (evidence.get('status')!='passed' or not evidence.get('contract_bridge_passed') or evidence.get('unit_tests',0)<25 or
        len(evidence.get('affected_files',[]))!=4 or
        not all(r['all_five_endpoints_and_selections_pass'] for r in evidence['affected_files']) or
        {r['year'] for r in evidence.get('full_pilot_files',[])}!={2024,2025} or
        not all(r['histograms_audits_equal'] for r in evidence['full_pilot_files'])):
        raise ValueError('Missing actual negative-pT validation')
    from TROTASR.workflows.nominal_campaign import frozen_plan
    old=frozen_plan(campaign)
    if old.get('negative_pt_selection'):raise ValueError('Already applied; do not duplicate')
    for name,digest in evidence['tested_sources'].items():
        if sha256(stage/name)!=digest or sha256(ROOT/name)!=evidence['previous_sources'][name]:
            raise ValueError('Source differs from tested deployment: '+name)
    controller=read_json(campaign/'controller.json')
    ident=process_identity(controller['pid'])
    if ident and (ident['start_ticks']!=controller['start_ticks'] or
        ident['arguments'][-3:]!=[str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign)]):
        raise ValueError('Controller identity changed')
    if not apply:
        print(dict(status='validated_ready',controller=ident),flush=True);return
    checkpoint=campaign/'history/negative_pt_selection_20260925'
    checkpoint.mkdir(exist_ok=False)
    for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                   *[ROOT/n for n in evidence['tested_sources']]]:
        target=checkpoint/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    # Signal only the verified controller. In-flight worker processes are kept.
    if ident:
        if process_identity(ident['pid'])!=ident:raise RuntimeError('Controller changed before handoff')
        os.kill(ident['pid'],signal.SIGTERM)
        for _ in range(100):
            if process_identity(ident['pid']) is None:break
            time.sleep(.1)
        else:raise RuntimeError('Controller did not exit')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read_json(campaign/'plan.json')!=old:raise ValueError('Concurrent plan mutation')
        states={p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        retained={};attempts={};complete={}
        for key,state in states.items():
            if state['status']=='running':
                identity=process_identity(state['pid'])
                receipt=campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
                if identity:
                    expected=[str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',str(campaign),
                              '--key',key,'--attempt',str(state['attempt'])]
                    if identity['arguments'][-len(expected):]!=expected:raise ValueError('Changed worker identity')
                    retained[key]=dict(pid=state['pid'],attempt=state['attempt'],identity=identity,
                                       history=str(receipt.relative_to(ROOT)))
                    attempts[key]=state['attempt']
                elif receipt.exists():
                    state=read_json(receipt);states[key]=state
                    write_json(campaign/'state'/(key+'.json'),state)
                else:raise ValueError('Missing exited worker receipt: '+key)
            if state['status']=='complete':
                if sha256(ROOT/state['output'])!=state['output_sha256']:raise ValueError('Changed completed output')
                complete[key]=state['output_sha256']
            if (state['status']=='failed' and state.get('exception_type')=='ValueError' and
                state.get('error') in ('Negative TROTA pT: jet_nanoaod_pt','Negative TROTA pT: fatjet_nanoaod_pt')):
                attempts[key]=state['attempt']
        for task in old['tasks']:
            if fingerprint(ROOT/task['input'])!=task['fingerprint']:raise ValueError('Changed original ROOT')
        for name in evidence['tested_sources']:
            target=ROOT/name;temp=target.with_name(target.name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,temp);os.replace(temp,target)
        verify_compiled()
        new=copy.deepcopy(old)
        new['contracts']={str(y):current_contract(y,'cms_trota_jme') for y in (2024,2025)}
        new['memory_control_policy']['retained_workers']=retained
        previous=checkpoint/(campaign/'plan.json').relative_to(ROOT)
        new['negative_pt_selection']=dict(previous_plan=str(previous.relative_to(ROOT)),
            previous_plan_sha256=sha256(previous),validation=str(validation.relative_to(ROOT)),
            validation_sha256=sha256(validation),original_attempts=attempts,
            workflow_sha256={n:h for n,h in evidence['tested_sources'].items() if n.startswith('workflows/')},
            existing_pt_thresholds_unchanged=True,endpoint_p4_unchanged=True)
        import importlib
        from TROTASR.workflows import systematic_continuation as policy
        importlib.reload(policy)
        policy.prior_contracts(new)
        write_json(campaign/'plan.json',new)
        frozen_plan(campaign)
        report=dict(status='deployed_pending_controller_start',time=time.time(),
            retained_workers=len(retained),successful_outputs_preserved=len(complete),
            successful_output_hashes=complete,corrected_attempt_keys=policy.retry_keys(new,states),
            workers_restarted=0,source_sha256=evidence['tested_sources'],full_workflow_complete=False)
        write_json(campaign/'negative_pt_selection.json',report)
    log=campaign/'logs/controller.negative_pt_selection_20260925.log'
    with log.open('a') as stream:
        proc=subprocess.Popen(controller['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_identity(proc.pid)
    if not identity:raise RuntimeError('Controller did not start')
    controller.update(pid=proc.pid,start_ticks=identity['start_ticks'],started=time.time(),
                      log=str(log.relative_to(ROOT)))
    write_json(campaign/'controller.json',controller)
    report.update(status='controller_started',pid=proc.pid,start_ticks=identity['start_ticks'])
    write_json(campaign/'negative_pt_selection.json',report)
    print({k:v for k,v in report.items() if k!='successful_output_hashes'},flush=True)


def apply_nonjme_extension(campaign, apply):
    """Deploy tested canonical additions while preserving in-flight weight work."""
    import importlib
    from TROTASR.workflows import nominal_campaign as runner
    folder=ROOT/'validation/nonjme_20260927'
    stage=folder/'source/TROTASR'
    gate_path=stage/'validation/validation_gate.json'
    gate=read_json(gate_path)
    deployment=read_json(folder/'deployment.json')
    old=runner.frozen_plan(campaign)
    if old.get('nonjme_code_extension'):raise ValueError('Already deployed; do not duplicate')
    if old.get('shape_mode') is not None or old.get('variations')!=['all_weights']:
        raise ValueError('Expected the existing weight-only campaign')
    if (gate.get('status')!='passed' or not gate.get('stored_trota_nonjme_validation') or
        gate.get('unit_tests',0)<44 or len(gate.get('records',[]))<10 or
        set(gate.get('regions',[]))!={'SR','LLCR','QCDCR','GCR','DY2E','DY2M'} or
        gate.get('baseline_weight_contracts')!=old['contracts']):
        raise ValueError('Missing tested weight-preserving extension')
    for name,record in deployment['files'].items():
        if sha256(stage/name)!=record['after']:
            raise ValueError('Candidate changed after deployment manifest: '+name)
        before=sha256(ROOT/name) if (ROOT/name).exists() else None
        if before!=record['before']:raise ValueError('Canonical source divergence: '+name)
    controller=read_json(campaign/'controller.json')
    identity=process_identity(controller['pid'])
    if identity and (identity['start_ticks']!=controller['start_ticks'] or
        identity['arguments'][-3:]!=[str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign)]):
        raise ValueError('Controller identity changed')
    if not apply:
        print(dict(status='validated_ready',controller=identity,files=len(deployment['files'])),flush=True);return
    checkpoint=campaign/'history/nonjme_20260927'
    checkpoint.mkdir(exist_ok=False)
    for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                   *[ROOT/n for n in deployment['files'] if (ROOT/n).exists()]]:
        target=checkpoint/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    if identity:
        if process_identity(identity['pid'])!=identity:raise ValueError('Controller changed before handoff')
        os.kill(identity['pid'],signal.SIGTERM)
        for _ in range(100):
            if process_identity(identity['pid']) is None:break
            time.sleep(.1)
        else:raise RuntimeError('Controller did not exit; no code deployed')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read_json(campaign/'plan.json')!=old:raise ValueError('Concurrent plan mutation')
        states={p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        retained={};complete={}
        for key,state in states.items():
            if state['status']=='running':
                ident=process_identity(state['pid'])
                receipt=campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
                if ident:
                    expected=[str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',str(campaign),
                              '--key',key,'--attempt',str(state['attempt'])]
                    if ident['arguments'][-len(expected):]!=expected:raise ValueError('Unexpected worker identity')
                    retained[key]=dict(pid=state['pid'],attempt=state['attempt'],identity=ident,
                                       history=str(receipt.relative_to(ROOT)))
                elif receipt.exists():
                    state=read_json(receipt);states[key]=state
                    write_json(campaign/'state'/(key+'.json'),state)
                else:raise ValueError('Missing exited worker receipt: '+key)
            if state['status']=='complete':
                if sha256(ROOT/state['output'])!=state['output_sha256']:raise ValueError('Completed output changed')
                complete[key]=state['output_sha256']
        for task in old['tasks']:
            if fingerprint(ROOT/task['input'])!=task['fingerprint']:raise ValueError('Input changed')
        for name in deployment['files']:
            target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,temp);os.replace(temp,target)
        importlib.invalidate_caches()
        importlib.reload(runner)
        runner.verify_compiled()
        new=copy.deepcopy(old)
        new['contracts']={str(y):runner.current_contract(y,variations=['all_weights']) for y in (2024,2025)}
        new['runner_sha256']=sha256(ROOT/'workflows/nominal_campaign.py')
        new['memory_control_policy']['retained_workers']=retained
        for name in new['memory_control_policy'].get('workflow_sha256',{}):
            new['memory_control_policy']['workflow_sha256'][name]=sha256(ROOT/name)
        previous=checkpoint/(campaign/'plan.json').relative_to(ROOT)
        new['nonjme_code_extension']=dict(previous_plan=str(previous.relative_to(ROOT)),
            previous_plan_sha256=sha256(previous),validation=str(gate_path.relative_to(ROOT)),
            validation_sha256=sha256(gate_path),successful_outputs_rewritten=0)
        # Reuse old-code real-file weight baselines only for still-pending keys.
        # Their stored input path is canonical, unlike the shape test hardlinks.
        added_seeds=[]
        for task in new['tasks']:
            baseline=stage/'validation/baseline'/(task['key']+'.json.gz')
            if task['key'] not in states and baseline.exists():
                runner.validate_payload(read_json(baseline),task,old['contracts'][str(task['year'])])
                task['seed_output']=str(baseline.relative_to(ROOT));added_seeds.append(task['key'])
        from TROTASR.workflows import systematic_continuation as policy
        importlib.reload(policy)
        policy.prior_contracts(new)
        write_json(campaign/'plan.json',new)
        runner.frozen_plan(campaign)
        report=dict(status='deployed_pending_controller_start',time=time.time(),
            previous_controller=identity,retained_workers=len(retained),workers_restarted=0,
            successful_outputs_preserved=len(complete),successful_output_hashes=complete,
            additional_weight_validation_seeds=added_seeds,
            deployment_sha256=sha256(folder/'deployment.json'),full_production_complete=False)
        write_json(campaign/'nonjme_extension.json',report)
    log=campaign/'logs/controller.nonjme_20260927.log'
    with log.open('a') as stream:
        proc=subprocess.Popen(controller['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    ident=process_identity(proc.pid)
    if not ident:raise RuntimeError('New controller exited; inspect log')
    controller.update(pid=proc.pid,start_ticks=ident['start_ticks'],started=time.time(),
        runner_sha256=new['runner_sha256'],log=str(log.relative_to(ROOT)))
    write_json(campaign/'controller.json',controller)
    report.update(status='controller_started',pid=proc.pid,start_ticks=ident['start_ticks'])
    write_json(campaign/'nonjme_extension.json',report)
    print({k:v for k,v in report.items() if k!='successful_output_hashes'},flush=True)


def apply_object_response_update(apply):
    """Coordinate the two active pools for an audit-only, bounded correction."""
    import importlib
    from contextlib import ExitStack
    from TROTASR.workflows import nominal_campaign as runner
    from TROTASR.workflows import systematic_continuation as policy
    folder=ROOT/'validation/nonjme_20260927'
    stage=folder/'response_sources/TROTASR'
    # Inspect the tested policy without changing the module in active workers.
    spec=importlib.util.spec_from_file_location('TROTASR.workflows.object_response_policy',
                                               stage/'workflows/systematic_continuation.py')
    candidate_policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate_policy)
    evidence=stage/'validation/object_response_validation.json';gate=read_json(evidence)
    deployment=read_json(folder/'object_response_deployment.json')
    campaigns={m:ROOT/'campaigns'/n for m,n in [('weights','weights_20260926'),
                                              ('stored_trota_nonjme','nonjme_20260927')]}
    plans={m:runner.frozen_plan(c) for m,c in campaigns.items()}
    if any(p.get('object_response_audit_update') for p in plans.values()):raise ValueError('Already applied')
    if (gate.get('status')!='passed' or gate.get('unit_tests',0)<48 or not gate.get('multi_dataset_files_passed')
        or not gate.get('old_success_histograms_equal') or not gate.get('weight_histograms_equal')
        or gate['before_contracts']!={m:p['contracts'] for m,p in plans.items()}):
        raise ValueError('Missing actual audit-only validation')
    for name,hashes in deployment['files'].items():
        if sha256(stage/name)!=hashes['after'] or sha256(ROOT/name)!=hashes['before']:
            raise ValueError('Deployment source divergence: '+name)
    controllers={m:read_json(c/'controller.json') for m,c in campaigns.items()}
    identities={m:process_identity(ctrl['pid']) for m,ctrl in controllers.items()}
    for m,ident in identities.items():
        if ident and (ident['start_ticks']!=controllers[m]['start_ticks'] or
            ident['arguments'][-3:]!=[str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaigns[m])]):
            raise ValueError('Controller identity changed')
    if not apply:
        print(dict(status='validated_ready',controllers=identities),flush=True);return
    checkpoints={m:c/'history/object_response_20260927' for m,c in campaigns.items()}
    for m,c in campaigns.items():
        checkpoint=checkpoints[m];checkpoint.mkdir(exist_ok=True)
        for source in [c/'plan.json',c/'state.json',c/'controller.json',*[ROOT/n for n in deployment['files']]]:
            target=checkpoint/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if source==c/'plan.json' and sha256(target)!=sha256(source):raise ValueError('Different checkpoint plan')
            else:shutil.copy2(source,target)
    for ident in identities.values():
        if ident:
            if process_identity(ident['pid'])!=ident:raise ValueError('Controller changed before stop')
            os.kill(ident['pid'],signal.SIGTERM)
    for ident in identities.values():
        if ident:
            for _ in range(100):
                if process_identity(ident['pid']) is None:break
                time.sleep(.1)
            else:raise RuntimeError('Controller failed to exit')
    reports={};newplans={}
    with ExitStack() as stack:
        for c in campaigns.values():
            lock=stack.enter_context((c/'controller.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for m,c in campaigns.items():
            old=plans[m]
            if read_json(c/'plan.json')!=old:raise ValueError('Concurrent plan mutation')
            states={p.stem:read_json(p) for p in (c/'state').glob('*.json')}
            retained={};attempts={};complete={}
            for key,state in states.items():
                if state['status']=='running':
                    ident=process_identity(state['pid']);receipt=c/'history'/(key+'.attempt%s.json'%state['attempt'])
                    expected=[str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',str(c),
                              '--key',key,'--attempt',str(state['attempt'])]
                    if ident and ident['arguments'][-len(expected):]!=expected:
                        # /proc/stat and cmdline can straddle the worker's exit.
                        # A vanished process is reconciled only via its receipt.
                        ident=process_identity(state['pid'])
                        if ident and ident['arguments'][-len(expected):]!=expected:
                            raise ValueError('Worker identity changed: '+key)
                    if ident:
                        retained[key]=dict(pid=state['pid'],attempt=state['attempt'],identity=ident,
                                           history=str(receipt.relative_to(ROOT)))
                        if m=='stored_trota_nonjme':attempts[key]=state['attempt']
                    elif receipt.exists():
                        state=read_json(receipt);states[key]=state;write_json(c/'state'/(key+'.json'),state)
                    else:raise ValueError('Missing exited worker receipt')
                if state['status']=='complete':
                    if sha256(ROOT/state['output'])!=state['output_sha256']:raise ValueError('Successful output changed')
                    complete[key]=state['output_sha256']
                probe={'object_response_audit_update':{'original_attempts':{key:state['attempt']}}}
                if m=='stored_trota_nonjme' and candidate_policy.approved_retry(probe,key,state):attempts[key]=state['attempt']
            for task in old['tasks']:
                if fingerprint(ROOT/task['input'])!=task['fingerprint']:raise ValueError('Original input changed')
            new=copy.deepcopy(old)
            new['contracts']=gate['after_contracts'][m]
            new['memory_control_policy']['retained_workers']=retained
            previous=checkpoints[m]/(c/'plan.json').relative_to(ROOT)
            new['object_response_audit_update']=dict(previous_plan=str(previous.relative_to(ROOT)),
                previous_plan_sha256=sha256(previous),validation=str(evidence.relative_to(ROOT)),
                validation_sha256=sha256(evidence),original_attempts=attempts,
                old_audit_warning='Previous successful mixed-dataset object_response counters may be inaccurate; histogram arrays remain valid.')
            seeds=[]
            if m=='stored_trota_nonjme':
                tested={r['key']:r for r in gate['records']}
                inputs={t['key']:t for t in read_json(stage/'validation/inputs.json')}
                for task in new['tasks']:
                    key=task['key'];state=states.get(key,{})
                    if key not in tested or (state and not candidate_policy.approved_retry(new,key,state)):continue
                    record=tested[key];source=inputs[key];output=stage/record['output']
                    if sha256(output)!=record['sha256'] or fingerprint(stage/source['input'])!=task['fingerprint']:
                        raise ValueError('Validated output/input changed: '+key)
                    task['input']=str((stage/source['input']).relative_to(ROOT))
                    if source.get('metadata'):
                        task['metadata']=str((stage/source['metadata']).relative_to(ROOT))
                        task['metadata_fingerprint']=fingerprint(ROOT/task['metadata'])
                    task['seed_output']=str(output.relative_to(ROOT));seeds.append(key)
            newplans[m]=new
            reports[m]=dict(status='deployed_pending_controller_start',time=time.time(),retained_workers=len(retained),
                successful_outputs_preserved=len(complete),workers_restarted=0,successful_outputs_rewritten=0,
                original_failure_keys=candidate_policy.retry_keys(new,states),validated_outputs_reused=seeds,
                old_audit_warning=new['object_response_audit_update']['old_audit_warning'])
        for name in deployment['files']:
            target=ROOT/name;tmp=target.with_name(target.name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,tmp);os.replace(tmp,target)
        importlib.invalidate_caches();importlib.reload(policy);runner.verify_compiled()
        for m,c in campaigns.items():
            policy.prior_contracts(newplans[m])
            write_json(c/'plan.json',newplans[m]);runner.frozen_plan(c)
            write_json(c/'object_response_update.json',reports[m])
    for m,c in campaigns.items():
        ctrl=controllers[m];log=c/'logs/controller.object_response_20260927.log'
        with log.open('a') as stream:
            proc=subprocess.Popen(ctrl['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        ident=process_identity(proc.pid)
        if not ident:raise RuntimeError('Controller launch failed; inspect log')
        ctrl.update(pid=proc.pid,start_ticks=ident['start_ticks'],started=time.time(),log=str(log.relative_to(ROOT)))
        write_json(c/'controller.json',ctrl)
        reports[m].update(status='controller_started',pid=proc.pid,start_ticks=ident['start_ticks'])
        write_json(c/'object_response_update.json',reports[m])
    print(reports,flush=True)


def apply_nonjme_throughput(campaign, apply, statistics_adapter=False, endpoint_execution=False):
    """Validated controller-only optimization; preserve all running workers."""
    import importlib.util
    from TROTASR.workflows import nominal_campaign as runner
    update_name = ('endpoint_execution' if endpoint_execution else
                   'statistics_adapter' if statistics_adapter else 'nonjme_throughput')
    update_field = update_name+'_update'
    folder = ROOT/('validation/'+update_name+'_20260928')
    stage = folder/'candidate/TROTASR'
    evidence = folder/'validation.json'
    gate = read_json(evidence)
    deployment = read_json(folder/'deployment.json')
    old = runner.frozen_plan(campaign)
    if old.get(update_field):
        raise ValueError('Already applied; do not duplicate controller')
    if old['contracts'] != gate['before_contracts']:
        raise ValueError('Campaign differs from measured baseline')
    spec = importlib.util.spec_from_file_location('TROTASR.workflows._throughput_policy',
                                                stage/'workflows/systematic_continuation.py')
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    bridge = dict(previous_plan=str((folder/'before_plan.json').relative_to(ROOT)),
                  previous_plan_sha256=sha256(folder/'before_plan.json'),
                  validation=str(evidence.relative_to(ROOT)), validation_sha256=sha256(evidence))
    if endpoint_execution:
        bridge['chunk_size'] = gate['selected_chunk_size']
    probe = dict(old, contracts=gate['after_contracts'], **{update_field:bridge})
    policy.prior_contracts(probe)
    for name, hashes in deployment['files'].items():
        if sha256(ROOT/name) != hashes['before'] or sha256(stage/name) != hashes['after']:
            raise ValueError('Deployment source divergence: '+name)
    if endpoint_execution:
        allowed = {'utils/histogramming.py','workflows/build_systematics.py','workflows/nominal_campaign.py',
                   'workflows/merge_nominal.py','workflows/systematic_continuation.py',
                   'workflows/continue_systematic_campaign.py'}
        if set(deployment['files']) != allowed:
            raise ValueError('Unexpected endpoint deployment targets')
        for name, hashes in deployment['files'].items():
            if gate['tested_sources'].get(name) != hashes['after']:
                raise ValueError('Untested deployment source: '+name)
    controller = read_json(campaign/'controller.json')
    ident = process_identity(controller['pid'])
    expected = [str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign)]
    if not ident or ident['start_ticks'] != controller['start_ticks'] or ident['arguments'][-3:] != expected:
        raise ValueError('Active controller identity changed')
    if not apply:
        print(dict(status='validated_ready',controller=ident),flush=True); return
    checkpoint = campaign/('history/'+update_name+'_20260928')
    checkpoint.mkdir(exist_ok=False)
    for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                   *[ROOT/n for n in deployment['files']]]:
        target = checkpoint/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    if process_identity(ident['pid']) != ident:
        raise ValueError('Controller changed before handoff')
    os.kill(ident['pid'],signal.SIGTERM)
    for _ in range(100):
        if process_identity(ident['pid']) is None: break
        time.sleep(.1)
    else: raise RuntimeError('Controller did not exit; no source deployed')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read_json(campaign/'plan.json') != old:
            raise ValueError('Concurrent plan mutation')
        retained, completed = {}, {}
        states = {p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        for key, state in states.items():
            if state['status'] == 'running':
                current = process_identity(state['pid'])
                receipt = campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
                expected = [str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',
                            str(campaign),'--key',key,'--attempt',str(state['attempt'])]
                if current and current['arguments'][-len(expected):] != expected:
                    current = process_identity(state['pid'])
                    if current and current['arguments'][-len(expected):] != expected:
                        raise ValueError('Worker identity changed: '+key)
                if current:
                    retained[key] = dict(pid=state['pid'],attempt=state['attempt'],identity=current,
                                         history=str(receipt.relative_to(ROOT)))
                elif receipt.exists():
                    state = read_json(receipt); states[key] = state
                    write_json(campaign/'state'/(key+'.json'),state)
                else: raise ValueError('Missing worker receipt: '+key)
            if state['status'] == 'complete':
                if sha256(ROOT/state['output']) != state['output_sha256']:
                    raise ValueError('Completed output changed: '+key)
                completed[key] = state['output_sha256']
            elif state['status'] == 'failed':
                raise ValueError('Investigate existing failure before optimization: '+key)
        for task in old['tasks']:
            if fingerprint(ROOT/task['input']) != task['fingerprint']:
                raise ValueError('Original input changed: '+task['key'])
        if endpoint_execution:
            bridge['split_input_keys'] = sorted(t['key'] for t in old['tasks']
                if t['key'] not in states and t['entries'] > old['chunk_size'] and not t.get('seed_output') and
                not (campaign/'outputs'/(t['key']+'.json.gz')).exists())
            bridge['small_input_policy'] = ('Files already fitting in one original batch retain '
                'shared reads/cache; independent tasks cannot gain batching there. Empty and '
                'already validated seed outputs are preserved.')
            bridge['minimum_split_entries'] = old['chunk_size']+1
            if not bridge['split_input_keys']:
                raise ValueError('No unstarted files to migrate')
        # No workers are signaled and the unchanged runner finishes against its
        # captured old contract. The new policy accepts both validated versions.
        for name in deployment['files']:
            target = ROOT/name; tmp = target.with_name(target.name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,tmp); os.replace(tmp,target)
        runner.verify_compiled()
        if endpoint_execution:
            import importlib
            importlib.invalidate_caches()
            importlib.reload(runner)
        new = copy.deepcopy(old)
        new['contracts'] = gate['after_contracts']
        previous = checkpoint/(campaign/'plan.json').relative_to(ROOT)
        bridge.update(previous_plan=str(previous.relative_to(ROOT)),previous_plan_sha256=sha256(previous))
        new[update_field] = bridge
        if endpoint_execution:
            new['runner_sha256'] = sha256(ROOT/'workflows/nominal_campaign.py')
            new['max_workers'] = 99  # aggregate and memory admission still bound every launch
        new['memory_control_policy']['retained_workers'] = retained
        # Keep reserve/emergency thresholds, the empirical peak safety factor,
        # and the existing 72-worker/100-total ceilings. 3 GiB exceeds measured
        # production peaks and validation peaks, and rises automatically if needed.
        observed_peak = read_json(campaign/'state.json').get('memory_control',{}).get('observed_peak_bytes',0)
        tested_peak = (gate['tested_peak_bytes'] if endpoint_execution else
                       max(r['peak_rss_bytes'] for r in gate['records']))
        safety = new['memory_control_policy']['peak_safety_factor']
        if endpoint_execution:
            import math
            quantum = new['memory_control_policy']['budget_quantum_bytes']
            new['admission_bytes_per_worker'] = max(3*1024**3,
                math.ceil(max(observed_peak,tested_peak)*safety/quantum)*quantum)
        elif not statistics_adapter and max(observed_peak,tested_peak)*safety < 3*1024**3:
            new['admission_bytes_per_worker'] = 3*1024**3
        new['memory_control_policy']['post_collection_stages'] = [new['max_workers']]
        new['memory_control_policy']['note'] = (
            'Validated '+update_name+' handoff; retained and new workers both '
            'collect after chunk500. Weights are complete; aggregate100 remains enforced.')
        if endpoint_execution:
            new['memory_control_policy']['note'] = (
                'Preserve existing full-file workers at chunk500. Only unstarted files use '
                'independent nominal+16 endpoints at the validated batch size; all collect '
                'after each chunk. Aggregate100, reserve32GiB/emergency24GiB remain enforced.')
        for name in new['memory_control_policy']['workflow_sha256']:
            new['memory_control_policy']['workflow_sha256'][name] = sha256(ROOT/name)
        policy.prior_contracts(new)
        write_json(campaign/'plan.json',new); runner.frozen_plan(campaign)
        report = dict(status='deployed_pending_controller_start',time=time.time(),
                      retained_workers=len(retained),workers_restarted=0,
                      successful_outputs_preserved=len(completed),completed_output_hashes=completed,
                      successful_outputs_rewritten=0,validation=str(evidence.relative_to(ROOT)),
                      admission_bytes_per_worker=new['admission_bytes_per_worker'],
                      previous_controller=copy.deepcopy(controller),full_workflow_complete=False)
        write_json(campaign/(update_field+'.json'),report)
    log = campaign/('logs/controller.'+update_name+'_20260928.log')
    with log.open('a') as stream:
        proc = subprocess.Popen(controller['command'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    current = process_identity(proc.pid)
    if not current: raise RuntimeError('New controller exited; inspect log')
    controller.update(pid=proc.pid,start_ticks=current['start_ticks'],started=time.time(),
                      plan_sha256=sha256(campaign/'plan.json'),log=str(log.relative_to(ROOT)),weight_workers=0)
    write_json(campaign/'controller.json',controller)
    report.update(status='controller_started',pid=proc.pid,start_ticks=current['start_ticks'])
    write_json(campaign/(update_field+'.json'),report)
    print({k:v for k,v in report.items() if k!='completed_output_hashes'},flush=True)


def retained_contract_bridge(plan, evidence, previous):
    """Only the proved full-file -> endpoint scheduling transition is accepted."""
    if (evidence.get('status') != 'passed' or evidence.get('unit_tests', 0) < 99
            or len(evidence.get('records', [])) < 18
            or {r['year'] for r in evidence['records']} != {2024, 2025}
            or not all(r.get('status') == 'passed' for r in evidence['records'])
            or not evidence.get('three_signal_topologies_tested')
            or not evidence.get('large_batch_validation_passed')
            or not evidence.get('strict_endpoint_merge_tested')
            or set(evidence.get('regions', [])) != {'SR','LLCR','QCDCR','GCR','DY2E','DY2M'}
            or evidence.get('before_contracts') != previous['contracts']
            or evidence.get('after_contracts') != plan['contracts']
            or evidence.get('selected_chunk_size') != plan['endpoint_execution_update']['chunk_size']):
        raise ValueError('Missing exact endpoint equivalence evidence')
    for year, after in plan['contracts'].items():
        before = previous['contracts'][year]
        if {k:v for k,v in before.items() if k != 'code'} != {k:v for k,v in after.items() if k != 'code'}:
            raise ValueError('Retained recovery cannot change physics inputs')
        changes = {k for k in set(before['code']) | set(after['code'])
                   if before['code'].get(k) != after['code'].get(k)}
        if changes != {'utils/histogramming.py'}:
            raise ValueError('Retained recovery is not the validated scheduling-only transition')


def finite_histogram_arrays(value):
    """Check signed contents, nonnegative variances/counts, and finite entries."""
    import math
    if not isinstance(value, dict):
        raise ValueError('Malformed histogram node')
    if 'sumw' in value:
        if not {'sumw2','entries','edges'} <= set(value):
            raise ValueError('Incomplete histogram leaf')
        def check(array, nonnegative):
            if isinstance(array, list):
                for item in array:
                    check(item, nonnegative)
            elif (isinstance(array, bool) or not isinstance(array, (int,float))
                  or not math.isfinite(array) or nonnegative and array < 0):
                raise ValueError('Invalid histogram contents or variance')
        for name in ('sumw','sumw2','entries'):
            check(value[name], name != 'sumw')
    else:
        for child in value.values():
            finite_histogram_arrays(child)


def recover_retained(campaign, evidence_path, apply=False):
    """Reconcile terminal metadata failure without rewriting any histogram.

    The original plan, execution revision, failure states and worker receipts
    remain recoverable. The new execution revision only pins validated outputs
    as preserved results. No failed command is relabeled as exit zero.
    """
    from TROTASR.workflows import nominal_campaign as runner
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = read_json(campaign/'plan.json')
        current = runner.frozen_plan(campaign)
        spec = current['_execution']
        if (plan.get('shape_mode') != 'stored_trota_nonjme'
                or plan.get('variations') != ['nominal']):
            raise ValueError('Recovery requires the canonical non-JME campaign')
        endpoint = plan['endpoint_execution_update']
        previous_path = internal_path(ROOT/endpoint['previous_plan'])
        if (sha256(previous_path) != endpoint['previous_plan_sha256']
                or sha256(evidence_path) != endpoint['validation_sha256']):
            raise ValueError('Relocated historical evidence hash mismatch')
        previous, evidence = read_json(previous_path), read_json(evidence_path)
        retained_contract_bridge(plan, evidence, previous)
        for r in spec['retained']:
            if process_identity(r['identity']['pid']) is not None:
                raise ValueError('Retained process is still alive or its PID was reused')
        states = {p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        if any(s.get('status') == 'running' for s in states.values()):
            raise ValueError('Recovery requires terminal task states')
        tasks = {t['key']:t for t in runner.execution_tasks(current)}
        failed = {k for k,s in states.items() if s.get('status') == 'failed'}
        if not failed:
            print(dict(status='no_failed_tasks',fits_or_events_launched=0),flush=True)
            return
        retained = {r['key']:r for r in spec['retained'] if r['kind'] == 'single'}
        if not failed <= set(retained):
            raise ValueError('Failure outside retained full-file workers')
        results = {}
        for key in sorted(failed):
            task, old = tasks[key], states[key]
            record = read_json(ROOT/retained[key]['receipt'])
            missing = str(ROOT/'validation/statistics_adapter_20260928/validation.json')
            if (record.get('status') != 'failed' or record.get('exception_type') != 'FileNotFoundError'
                    or record.get('error') != "[Errno 2] No such file or directory: '"+missing+"'"
                    or old.get('error') != record['error']
                    or 'in prior_contracts' not in record.get('traceback','')):
                raise ValueError('Not a proved post-build metadata failure: '+key)
            if fingerprint(ROOT/task['input']) != task['fingerprint']:
                raise ValueError('Original input changed: '+key)
            if fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
                raise ValueError('Original metadata changed: '+key)
            output = campaign/'outputs'/(key+'.json.gz')
            digest = sha256(output)
            raw = read_json(output)
            expected = previous['contracts'][str(task['year'])]
            runner.validate_payload(raw,task,expected)
            for section in ('histograms','physical_histograms','dy_rz','dy_rz_variations'):
                finite_histogram_arrays(raw.get(section,{}))
            if sha256(output) != digest:
                raise ValueError('Output changed during recovery')
            results[key] = dict(status='complete',key=key,attempt=old['attempt'],
                output=runner.relative(output),output_sha256=digest,entries=task['entries'],
                year=task['year'],seconds=raw['seconds'],reused=True,finished=time.time(),
                acceptance_source='retained_output_integrity_recovery',
                original_worker_status='failed',original_worker_receipt=retained[key]['receipt'],
                original_worker_receipt_sha256=sha256(ROOT/retained[key]['receipt']),
                validation_evidence=runner.relative(evidence_path),validation_evidence_sha256=sha256(evidence_path))
        if not apply:
            print(dict(status='verified_ready',results=results,events_recomputed=0),flush=True)
            return
        folder = campaign/'history/retained_output_recovery'
        folder.mkdir(exist_ok=False)
        pins = {}
        for path in (campaign/'plan.json',campaign/'execution.json',campaign/'state.json',
                     *[campaign/'state'/(k+'.json') for k in results]):
            target = folder/path.relative_to(campaign)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            pins[runner.relative(target)] = sha256(target)
        amended = copy.deepcopy(spec)
        for key, state in results.items():
            write_json(campaign/'state'/(key+'.json'),state)
            amended['preserved'][key] = dict(output=state['output'],output_sha256=state['output_sha256'],
                state_sha256=sha256(campaign/'state'/(key+'.json')))
        amended['retained_output_recovery'] = dict(previous_execution=runner.relative(folder/'execution.json'),
            previous_execution_sha256=sha256(folder/'execution.json'),keys=sorted(results),
            evidence=runner.relative(evidence_path),evidence_sha256=sha256(evidence_path),
            source_sha256=sha256(Path(__file__)),physics_changes=False,histograms_rewritten=0)
        write_json(campaign/'execution.json',amended)
        runner.frozen_plan(campaign)
        write_json(folder/'receipt.json',dict(status='recovered',results=results,originals=pins,
            execution_sha256=sha256(campaign/'execution.json'),events_recomputed=0,histograms_rewritten=0))
        print(dict(status='recovered',keys=sorted(results),events_recomputed=0,histograms_rewritten=0),flush=True)


if __name__=='__main__':main()
