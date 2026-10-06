"""After all original controllers exit, preserve and promote a proved extension.

No process is killed. No ROOT or completed histogram is changed. Run without
--apply for readiness. Start existing chain/products separately after success.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.nominal_campaign import current_contract, fingerprint
from TROTASR.workflows.normalization_recovery import verify_extension, retry_keys, retained_worker


def live(pid):
    if not isinstance(pid,int) or pid<2:return False
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return False
    return p.read_text().split(') ',1)[1].split()[0]!='Z'


def run(campaign, apply=False, retain_running=False):
    campaign=internal_path(campaign)
    products=ROOT/'stats'/campaign.name
    recovery=campaign/'recovery/missing_normalization_20260923'
    audit=read_json(recovery/'audit.json')
    plan_path=campaign/'plan.json'; plan=read_json(plan_path)
    if plan.get('normalization_extension_recovery'):
        raise ValueError('Already promoted; inspect/resume, never promote twice')
    if sha256(plan_path)!=audit['plan_sha256']:
        raise ValueError('Original plan changed since normalization audit')
    states=[read_json(p) for p in (campaign/'state.json',campaign/'chain_state.json',products/'products_state.json')]
    active=[s.get('pid') for s in states if live(s.get('pid'))]
    active += [p for s in states for p in s.get('child_pids',{}).values() if live(p)]
    if active:
        print(json.dumps(dict(status='waiting_for_original_controllers_to_exit',active_pids=active,
                              promoted=False)))
        return 75
    locks=[]
    try:
        for path in (campaign/'chain.lock',campaign/'controller.lock',products/'products.lock'):
            lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        if states[0]['pending']:
            raise ValueError('Original controller still has pending work')
        if not retain_running and (states[0]['status']!='complete_with_failures' or states[0]['active']):
            raise ValueError('Original campaign has not drained with explicit failures')
        if not retain_running and states[1].get('stages',{}).get('histograms',{}).get('exit_code')!=1:
            raise ValueError('Unexpected original chain outcome')
        if retain_running:
            transition=read_json(campaign/'controller_transition.json')
            if (transition.get('status')!='controllers_stopped_workers_retained'
                or transition.get('plan_sha256')!=audit['plan_sha256']
                or transition.get('previous_controller_pids')!=[s.get('pid') for s in states]
                or not transition.get('user_ordered_immediate_execution')):
                raise ValueError('Missing explicit immediate controller handoff')
        if states[2].get('stages') or states[2].get('child_pids'):
            raise ValueError('Downstream stage already started; inspect before recovery')
        expected_failures=sorted(k for y in audit['years'].values() for k in y['missing_keys'])
        if sorted(states[0]['failed'])!=expected_failures:
            raise ValueError('Failures differ from missing normalization coverage')
        perfile={t['key']:read_json(campaign/'state'/(t['key']+'.json')) for t in plan['tasks']}
        retained={key:retained_worker(campaign,key,state) for key,state in perfile.items()
                  if retain_running and state.get('status')=='running'}
        approved=dict(normalization_extension_recovery=dict(retry_keys=expected_failures))
        if retry_keys(approved,perfile)!=expected_failures:
            raise ValueError('Not exactly one first-attempt normalization failure per missing shard')
        for task in plan['tasks']:
            if fingerprint(ROOT/task['input'])!=task['fingerprint']:
                raise ValueError('Input changed: '+task['key'])
            state=perfile[task['key']]
            if task['key'] in retained:
                continue
            if task['key'] not in expected_failures:
                if state.get('status')!='complete' or sha256(ROOT/state['output'])!=state['output_sha256']:
                    raise ValueError('Previously successful histogram changed: '+task['key'])
        assets=read_json(ROOT/'jsons/assets.json')
        originals={}
        for year,record in audit['years'].items():
            old=ROOT/'estimations'/('normalization_%s.json.gz'%year)
            candidate=internal_path(ROOT/record['candidate'])
            if sha256(old)!=record['original_sha256'] or sha256(candidate)!=record['candidate_sha256']:
                raise ValueError('Normalization differs from tested candidate')
            if verify_extension(read_json(old),read_json(candidate))!=record['added_mass_points']:
                raise ValueError('Not the verified disjoint extension')
            originals[year]=old
        staged=ROOT/'validation/normalization_recovery_20260923/staged_workflows'
        sources=read_json(ROOT/'jsons/normalization_recovery_sources.json')
        expected=sources['promotion_files_sha256']
        for name,digest in expected.items():
            if sha256(internal_path(ROOT/name))!=digest:raise ValueError('Changed tested source: '+name)
        if sha256(ROOT/'workflows/nominal_campaign.py')!=plan['runner_sha256']:
            raise ValueError('Original frozen runner changed')
        for name,digest in sources['original_active_files_sha256'].items():
            if sha256(internal_path(ROOT/name))!=digest:raise ValueError('Original active source changed: '+name)
        if not apply:
            print(json.dumps(dict(status='ready_to_promote',retry_files=len(expected_failures),
                                  successful_histograms_preserved=len(perfile)-len(expected_failures)-len(retained),
                                  workers_retained=len(retained))))
            return 0
        checkpoint=recovery/'checkpoint_before_promotion'
        if checkpoint.exists():raise FileExistsError('Interrupted promotion checkpoint; inspect, never retry blindly')
        checkpoint.mkdir()
        targets=[plan_path,ROOT/'jsons/assets.json',ROOT/'workflows/nominal_campaign.py',ROOT/'workflows/merge_nominal.py',
                 campaign/'state.json',campaign/'chain_state.json',products/'products_state.json',*originals.values()]
        if retain_running:targets.append(campaign/'controller_transition.json')
        for source in targets:
            target=checkpoint/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        # Immutable archive hashes make the original statuses and payloads auditable.
        write_json(checkpoint/'manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in targets})
        for year,old in originals.items():
            rec=audit['years'][year];candidate=internal_path(ROOT/rec['candidate'])
            temp=old.with_name(old.name+'.pending.'+str(os.getpid()))
            if temp.exists():raise FileExistsError(temp)
            shutil.copyfile(candidate,temp);os.replace(str(temp),str(old))
            entry=next(r for r in assets['files'] if r['target']==str(old.relative_to(ROOT)))
            entry.update(source=rec['candidate'],source_sha256=rec['candidate_sha256'],
                         sha256=rec['candidate_sha256'],transform='validated missing signal metadata extension; existing factors identical')
        write_json(ROOT/'jsons/assets.json',assets)
        for name in ('nominal_campaign.py','merge_nominal.py'):
            shutil.copyfile(staged/name,ROOT/'workflows'/name)
        new_plan=copy.deepcopy(plan)
        new_plan['contracts']={str(y):current_contract(y) for y in (2024,2025)}
        new_plan['runner_sha256']=sha256(ROOT/'workflows/nominal_campaign.py')
        spec=dict(audit=str((recovery/'audit.json').relative_to(ROOT)),audit_sha256=sha256(recovery/'audit.json'),
                  prior_contracts=plan['contracts'],retry_keys=expected_failures,
                  original_normalizations={y:str((checkpoint/p.relative_to(ROOT)).relative_to(ROOT)) for y,p in originals.items()},
                  original_assets=str((checkpoint/'assets.json').relative_to(ROOT)),
                  code_sha256={name:digest for name,digest in expected.items() if name.startswith('workflows/')},
                  original_plan_sha256=audit['plan_sha256'],histograms_rewritten=0)
        if retained:spec['retained_workers']=retained
        spec['code_sha256']['workflows/merge_nominal.py']=sha256(ROOT/'workflows/merge_nominal.py')
        new_plan['normalization_extension_recovery']=spec
        write_json(plan_path,new_plan)
        # Preserve failed states/history in place. New runner schedules attempt2;
        # it never overwrites attempt1 and never retries a second corrected failure.
        # Products binds a plan hash at startup, so archive its exited controller receipt.
        archived=recovery/'products_state_before_promotion.json'
        if archived.exists():raise FileExistsError(archived)
        os.replace(str(products/'products_state.json'),str(archived))
        from TROTASR.workflows.normalization_recovery import prior_contracts
        prior_contracts(new_plan)
        write_json(recovery/'promotion.json',dict(status='promoted_pending_controller_restart',
            finished=time.time(),plan_sha256=sha256(plan_path),retry_keys=expected_failures,
            successful_histograms_preserved=len(perfile)-len(expected_failures)-len(retained),histograms_rewritten=0,
            workers_retained=len(retained),
            previous_controller_pids=[s.get('pid') for s in states],full_workflow_complete=False))
        print(json.dumps(dict(status='promoted_pending_controller_restart',retry_files=len(expected_failures))))
        return 0
    finally:
        for lock in reversed(locks):lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--retain-running',action='store_true')
    args=p.parse_args();raise SystemExit(run(args.campaign,args.apply,args.retain_running))
