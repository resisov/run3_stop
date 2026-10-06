"""Deploy tested scheduling-only controls; preserve all physics and outputs."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import unittest
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.normalization_recovery import process_identity, payload_contract


def load_stage(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    campaign=ROOT/'campaigns/systematics_20260924'
    stage=ROOT/'validation/jme_nominal_20260924/memory_sources'
    sources=['workflows/nominal_campaign.py','workflows/campaign_resources.py']
    reduction=read_json(campaign/'memory_control.json')
    if reduction['status']!='reduced':raise ValueError('Initial 32-worker reduction not finished')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read_json(campaign/'plan.json')
        if old.get('memory_control_policy'):raise ValueError('Already deployed; do not duplicate')
        if sha256(ROOT/'workflows/nominal_campaign.py')!=old['runner_sha256']:
            raise ValueError('Unrecorded runner change')
        controller=read_json(campaign/'controller.json')
        live=process_identity(controller['pid'])
        if live and live['start_ticks']==controller['start_ticks']:
            raise ValueError('A controller is already alive')
        control=load_stage('TROTASR.workflows.campaign_resources',stage/sources[1])
        runner=load_stage('TROTASR.workflows.nominal_campaign',stage/sources[0])
        suite=unittest.TestSuite()
        for module in ('test_campaign_resources','test_nominal_campaign','test_systematic_continuation'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName('TROTASR.tests.'+module))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.testsRun!=19:raise ValueError('Resource regression tests failed')
        runner.verify_compiled()
        if any(runner.current_contract(int(y),old['shape_mode'])!=c for y,c in old['contracts'].items()):
            raise ValueError('Physics contract changed')
        from TROTASR.workflows import systematic_continuation as policy
        previous=policy.prior_contracts(old)
        checkpoint=campaign/'history/memory_control_20260924'
        checkpoint.mkdir(exist_ok=False)
        for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                       *[ROOT/name for name in sources if (ROOT/name).exists()]]:
            destination=checkpoint/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
        states={p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        tasks={t['key']:t for t in old['tasks']}
        retained={}; resubmissions={}; reconciled=[]; complete_hashes={}
        terminated={r['key']:r for r in reduction['interrupted']}
        for key,state in states.items():
            original=copy.deepcopy(state)
            if state['status']=='running':
                ident=process_identity(state['pid'])
                if ident:
                    expected=[str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',str(campaign),
                              '--key',key,'--attempt',str(state['attempt'])]
                    if ident['arguments'][-len(expected):]!=expected:
                        raise ValueError('Unexpected live worker: '+key)
                    retained[key]=dict(pid=state['pid'],attempt=state['attempt'],identity=ident,
                        history=str((campaign/'history'/(key+'.attempt%s.json'%state['attempt'])).relative_to(ROOT)))
                    continue
                receipt=campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
                if receipt.exists():
                    state=read_json(receipt)
                else:
                    output=campaign/'outputs'/(key+'.json.gz')
                    if output.exists():
                        payload=read_json(output)
                        expected=payload_contract(payload,old['contracts'][str(tasks[key]['year'])],
                                                  previous.get(str(tasks[key]['year'])))
                        runner.validate_payload(payload,tasks[key],expected)
                        state=dict(status='complete',key=key,attempt=state['attempt'],started=state['started'],
                            finished=time.time(),output=str(output.relative_to(ROOT)),output_sha256=sha256(output),
                            entries=tasks[key]['entries'],year=tasks[key]['year'],seconds=payload['seconds'],
                            receipt_from_validated_committed_output=True)
                    else:
                        state=dict(status='failed',key=key,attempt=state['attempt'],finished=time.time(),
                            exception_type='ResourceInterruption',retryable_read_error=False,
                            error=('Stopped unfinished worker to apply approved 32-worker memory limit'
                                   if key in terminated else 'Worker exited without validated result'),
                            original_pid=state['pid'],os_exit_status_unknown=True)
                    write_json(receipt,state)
                write_json(campaign/'state'/(key+'.json'),state)
                states[key]=state
                reconciled.append(dict(key=key,before=original,after=state))
            if state['status']=='complete':
                digest=sha256(ROOT/state['output'])
                if digest!=state['output_sha256']:raise ValueError('Changed successful output: '+key)
                complete_hashes[key]=digest
            elif state['status']=='failed' and state.get('error') in (
                'Worker exited without validated result','Stopped unfinished worker to apply approved 32-worker memory limit'):
                resubmissions[key]=dict(failed_attempt=state['attempt'],error=state['error'],
                    reason='One bounded resource resubmission after memory admission correction; original failure retained')
        if len(retained)>32:raise ValueError('Initial active worker cap exceeded')
        for task in tasks.values():
            if runner.fingerprint(ROOT/task['input'])!=task['fingerprint']:
                raise ValueError('Input fingerprint changed: '+task['key'])
        for name in sources:
            temporary=(ROOT/name).with_name(Path(name).name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,temporary);os.replace(temporary,ROOT/name)
        new=copy.deepcopy(old)
        new['max_workers']=50
        new['runner_sha256']=sha256(ROOT/sources[0])
        new['minimum_available_memory_bytes']=32*1024**3
        new['admission_bytes_per_worker']=6*1024**3
        hashes={name:sha256(ROOT/name) for name in sources}
        new[control.FIELD]=dict(initial_workers=32,ramp_seconds=600,peak_safety_factor=1.25,
            emergency_available_bytes=24*1024**3,retained_workers=retained,resubmissions=resubmissions,
            workflow_sha256=hashes,previous_plan=str((checkpoint/(campaign/'plan.json').relative_to(ROOT)).relative_to(ROOT)),
            unit_tests=result.testsRun,physics_contract_unchanged=True)
        new[policy.FIELD]['workflow_sha256'][sources[0]]=new['runner_sha256']
        policy.prior_contracts(new)
        write_json(campaign/'plan.json',new)
        runner.frozen_plan(campaign)
        audit=dict(status='deployed_pending_launch',time=time.time(),unit_tests=result.testsRun,
            workflow_sha256=hashes,retained_workers=len(retained),resubmissions=resubmissions,
            reconciled=reconciled,successful_outputs_preserved=len(complete_hashes),
            successful_output_hashes=complete_hashes,plan_sha256=sha256(campaign/'plan.json'),
            physics_contract_unchanged=new['contracts']==old['contracts'],full_workflow_complete=False)
        write_json(campaign/'memory_deployment.json',audit)
    command=['bash',str(ROOT/'workflows/run_inference.sh'),str(ROOT/'workflows/run_systematic_campaign.py'),
             '--campaign',str(campaign)]
    log=campaign/'logs/controller.memory_control_20260924.log'
    with log.open('a') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    ident=process_identity(proc.pid)
    if not ident:raise RuntimeError('Controller exited at launch')
    write_json(campaign/'controller.json',dict(pid=proc.pid,start_ticks=ident['start_ticks'],command=command,
        started=time.time(),max_workers=50,initial_workers=32,aggregate_slots=100,
        monitoring_automation='PAUSED',runner_sha256=new['runner_sha256'],log=str(log.relative_to(ROOT))))
    audit.update(status='controller_started',controller_pid=proc.pid,controller_start_ticks=ident['start_ticks'])
    write_json(campaign/'memory_deployment.json',audit)
    print(json.dumps(dict(status=audit['status'],pid=proc.pid,retained=len(retained),
        resource_resubmissions=len(resubmissions),successful_outputs_preserved=len(complete_hashes))))


def enable_collection(scaling=False):
    campaign=ROOT/'campaigns/systematics_20260924'
    folder=ROOT/'validation/jme_nominal_20260924/memory_validation'
    evidence_path=folder/('scaling.json' if scaling else 'closure.json')
    evidence=read_json(evidence_path)
    if (evidence.get('status')!='passed' or evidence['unit_tests']!=(25 if scaling else 21) or
        {r['year'] for r in evidence['full_pilot_files']}!={2024,2025} or
        not all(r['all_histograms_audits_contracts_equal'] for r in evidence['full_pilot_files'])):
        raise ValueError('Collection closure has not passed')
    stage=ROOT/'validation/jme_nominal_20260924/memory_sources'
    for name,digest in evidence['source_sha256'].items():
        if sha256(stage/name)!=digest:raise ValueError('Tested source changed: '+name)
    old=read_json(campaign/'plan.json')
    legacy_outcomes={}
    if scaling:
        if (not old['memory_control_policy'].get('collect_after_chunk') or
            old['memory_control_policy'].get('post_collection_stages')):
            raise ValueError('Collection required; scaling must not already be deployed')
        if evidence.get('changed_only')!='reserved_capacity':raise ValueError('Not scheduling-only validation')
        for name,digest in evidence['previous_sources'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Deployed source differs from scaling validation')
        for key,record in old['memory_control_policy']['retained_workers'].items():
            live=process_identity(record['pid'])
            if live and live['start_ticks']==record['identity']['start_ticks']:
                raise ValueError('Legacy worker has not exited; no changes made: '+key)
            history=ROOT/record['history'];result=read_json(history)
            if result.get('status')!='complete' or result['attempt']!=record['attempt']:
                raise ValueError('Legacy attempt not successfully complete: '+key)
            if sha256(ROOT/result['output'])!=result['output_sha256']:
                raise ValueError('Legacy completed output integrity mismatch: '+key)
            legacy_outcomes[key]=dict(pid=record['pid'],start_ticks=record['identity']['start_ticks'],
                attempt=result['attempt'],finished=result['finished'],output_sha256=result['output_sha256'],
                receipt=record['history'],receipt_sha256=sha256(history))
    elif old['memory_control_policy'].get('collect_after_chunk'):
        raise ValueError('Already enabled')
    if sha256(ROOT/'workflows/nominal_campaign.py')!=old['runner_sha256']:
        raise ValueError('Unexpected deployed runner')
    controller=read_json(campaign/'controller.json')
    ident=process_identity(controller['pid'])
    if ident and (ident['start_ticks']!=controller['start_ticks'] or
        ident['arguments'][-3:]!=[str(ROOT/'workflows/run_systematic_campaign.py'),'--campaign',str(campaign)]):
        raise ValueError('Unexpected controller identity')
    checkpoint=campaign/('history/worker_scaling_20260924' if scaling else 'history/chunk_collection_20260924')
    receipt_path=campaign/('worker_scaling.json' if scaling else 'chunk_collection.json')
    checkpoint.mkdir(exist_ok=False)
    for source in [campaign/'plan.json',campaign/'state.json',campaign/'controller.json',
                   *[ROOT/name for name in evidence['source_sha256']]]:
        destination=checkpoint/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    if ident:
        if process_identity(ident['pid'])!=ident:raise RuntimeError('Controller changed before signal')
        os.kill(ident['pid'],signal.SIGTERM)
        for _ in range(100):
            if process_identity(ident['pid']) is None:break
            time.sleep(.1)
        else:raise RuntimeError('Controller did not exit')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read_json(campaign/'plan.json')!=old:raise ValueError('Concurrent plan mutation')
        states={p.stem:read_json(p) for p in (campaign/'state').glob('*.json')}
        retained={}
        for key,state in states.items():
            if state['status']!='running':continue
            identity=process_identity(state['pid'])
            history=campaign/'history'/(key+'.attempt%s.json'%state['attempt'])
            if identity:
                expected=[str(ROOT/'workflows/nominal_campaign.py'),'worker','--campaign',str(campaign),
                          '--key',key,'--attempt',str(state['attempt'])]
                if identity['arguments'][-len(expected):]!=expected:raise ValueError('Unexpected active worker')
                retained[key]=dict(pid=state['pid'],attempt=state['attempt'],identity=identity,
                                   history=str(history.relative_to(ROOT)))
            elif history.exists():
                write_json(campaign/'state'/(key+'.json'),read_json(history))
            else:
                raise ValueError('Dead worker without receipt during handoff: '+key)
        for name in evidence['source_sha256']:
            temp=(ROOT/name).with_name(Path(name).name+'.pending.'+str(os.getpid()))
            shutil.copy2(stage/name,temp);os.replace(temp,ROOT/name)
        new=copy.deepcopy(old)
        new['runner_sha256']=evidence['source_sha256']['workflows/nominal_campaign.py']
        new['systematic_input_policy_update']['workflow_sha256']['workflows/nominal_campaign.py']=new['runner_sha256']
        new['memory_control_policy'].update(collect_after_chunk=True,retained_workers=retained,
            workflow_sha256=evidence['source_sha256'],unit_tests=evidence['unit_tests'])
        if scaling:
            new.update(max_workers=98,admission_bytes_per_worker=2*1024**3)
            new['memory_control_policy'].update(initial_workers=64,ramp_seconds=600,
                post_collection_stages=[64,80,98],budget_quantum_bytes=1024**3//4,
                legacy_workers_exited_verified=True,legacy_completion=legacy_outcomes,
                collection_legacy_workers=old['memory_control_policy']['retained_workers'],
                scaling_validation=str(evidence_path.relative_to(ROOT)),
                scaling_validation_sha256=sha256(evidence_path))
        else:
            new['memory_control_policy'].update(collection_validation=str(evidence_path.relative_to(ROOT)),
                collection_validation_sha256=sha256(evidence_path))
        from TROTASR.workflows.systematic_continuation import prior_contracts
        prior_contracts(new)
        write_json(campaign/'plan.json',new)
        write_json(receipt_path,dict(status='deployed_pending_launch',time=time.time(),
            retained_workers=len(retained),old_controller=controller,
            source_sha256=evidence['source_sha256'],validation=str(evidence_path.relative_to(ROOT)),
            physics_contract_unchanged=new['contracts']==old['contracts'],workers_restarted=0,
            legacy_complete_verified=len(legacy_outcomes),max_workers=new['max_workers'],
            stages=new['memory_control_policy'].get('post_collection_stages')))
    command=controller['command'];log=ROOT/controller['log']
    with log.open('a') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_identity(proc.pid)
    if not identity:raise RuntimeError('Controller exited during launch')
    controller.update(pid=proc.pid,start_ticks=identity['start_ticks'],started=time.time(),
                      runner_sha256=new['runner_sha256'],max_workers=new['max_workers'],
                      initial_workers=new['memory_control_policy']['initial_workers'])
    write_json(campaign/'controller.json',controller)
    receipt=read_json(receipt_path)
    receipt.update(status='controller_started',pid=proc.pid,start_ticks=identity['start_ticks'])
    write_json(receipt_path,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    if '--scale-new-workers' in sys.argv:
        enable_collection(scaling=True)
    elif '--enable-collection' in sys.argv:
        enable_collection()
    else:
        main()
