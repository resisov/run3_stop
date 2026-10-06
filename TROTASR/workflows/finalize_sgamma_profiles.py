"""Validate warm-start Sgamma boundaries/crossings and collect all916 impacts."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import math
import os
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.diagnostic_common import native_runtime,grid_card,freeze_contract,CommandRunner
from TROTASR.workflows.profile_sgamma import WORK,MANIFEST,NAMES,fit
from TROTASR.workflows.continue_sgamma_profiles import tracked_start
from TROTASR.workflows.continue_impact_profiles import validate_point
from TROTASR.workflows import impact_recovery_legacy as legacy


def require_point(point):
    return validate_point(point,point['name'])


def validate_boundary_pair(first,second):
    for p in (first,second):
        if p['status']!='complete' or p['theta']!=0. or not -.005<=p['deltaNLL']<.5:
            raise ValueError('Not a validated original-boundary endpoint')
    if abs(first['deltaNLL']-second['deltaNLL'])>.002 or abs(first['r']-second['r'])>.0002:
        raise ValueError('Independent boundary minima disagree')


def parameter(work,out,runner,inputs,name):
    warm=work/'profile_warmstart'/name
    seed=require_point(read_json(warm/'seed/state.json'))
    interior=require_point(read_json(warm/'interior/state.json'))
    boundary=require_point(read_json(warm/'boundary/state.json'))
    allowed=set(inputs['bestfit'])
    # Independent path: go directly from the seed to zero, with strategy1.
    check=require_point(fit(runner,out,inputs,name,0.,'boundary_strategy1',
        extra=['--cminDefaultMinimizerStrategy','1'],seeds=tracked_start(seed,allowed)))
    validate_boundary_pair(boundary,check)
    # Return to the global central coordinate from the boundary nuisance seed;
    # common nll0 is still the unmodified initial snapshot before assignments.
    theta=inputs['bestfit'][name]
    central=require_point(fit(runner,out,inputs,name,theta,'central_check',
        seeds=tracked_start(boundary,allowed)))
    if abs(central['deltaNLL'])>.005 or abs(central['r']-inputs['bestfit']['r'])>.0002:
        raise ValueError('Warm path does not reproduce common minimum')
    points=[seed,interior,boundary,check,central]
    prior={}
    for sub in ('points','points_refined'):
        for path in (work/'direct_profile'/sub/name).glob('*/state.json'):
            p=read_json(path)
            if p['status']=='complete' and p['theta']>theta:
                prior[format(p['theta'],'.9g')]=require_point(p)
    inner=max((p for p in prior.values() if p['deltaNLL']<.5),key=lambda p:p['theta'])
    outer=min((p for p in prior.values() if p['deltaNLL']>=.5),key=lambda p:p['theta'])
    points += [inner,outer]
    # Same18-step crossing solver and0.002 tolerance as the adopted routine.
    for index in range(18):
        if abs(outer['deltaNLL']-.5)<=.002:
            upper=outer
            break
        fraction=max(.2,min(.8,(.5-inner['deltaNLL'])/(outer['deltaNLL']-inner['deltaNLL'])))
        value=inner['theta']+fraction*(outer['theta']-inner['theta'])
        p=require_point(fit(runner,out,inputs,name,value,'upper_%02d'%index))
        points.append(p)
        if abs(p['deltaNLL']-.5)<=.002:
            upper=p
            break
        if p['deltaNLL']<.5:inner=p
        else:outer=p
    else:
        raise ValueError('Upper crossing not validated in18 steps')
    result=dict(status='complete',name=name,method='direct_profile_neighbour_seeded',
        nominal=dict(theta=theta,r=inputs['bestfit']['r'],deltaNLL=0.),lower=boundary,
        upper=upper,points=points,boundary_limited_lower=True,boundary_limited_upper=False,
        target_deltaNLL=.5,crossing_tolerance=.002,independent_boundary_check=check,
        central_check=central,physical_model_unchanged=True)
    write_json(out/'parameters'/(name+'.json'),result)
    return result


def main():
    import ROOT as root
    root.gROOT.SetBatch(True);root.EnableThreadSafety()
    work=internal_path(WORK);out=work/'profile_validation';out.mkdir(exist_ok=True)
    with (work/'impact.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,origins=grid_card(internal_path(MANIFEST),'impact')
        parent=read_json(work/'impact_status.json')
        if parent['status'] not in ('needs_attention','fits_complete'):
            raise ValueError('Unexpected parent state')
        inputs=read_json(work/'direct_profile/inputs.json')
        for key in ('snapshot','asimov'):
            if sha256(inputs[key])!=inputs[key+'_sha256']:raise ValueError('Changed inputs')
        if sha256(parent['workspace'])!=parent['workspace_sha256']:raise ValueError('Changed workspace')
        freeze_contract(out,origins,['workflows/finalize_sgamma_profiles.py',
            'workflows/continue_sgamma_profiles.py','workflows/profile_sgamma.py'],dict(
            names=NAMES,maximum_new_fits=40,workers=2,rate_bounds=[0,10],r_bounds=[0,20],
            reason='Warm-start succeeded at both previously failed interior points and physical zero boundary',
            checks='Independent strategy1 direct-seed boundary; return-to-common-minimum; upper crossing0.5+-0.002',
            physical_model_unchanged=True))
        runner=CommandRunner(out)
        state=dict(status='running',controller_pid=os.getpid(),started=time.time(),completed=[],failed={})
        write_json(out/'status.json',state)
        try:
            recovered={}
            for folder in ('direct_profile','profile_search'):
                for p in (work/folder/'parameters').glob('*.json'):
                    d=read_json(p)
                    if d['status']=='complete':
                        for point in d['points']:validate_point(point,d['name'])
                        recovered[d['name']]=d
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks={n:pool.submit(parameter,work,out,runner,inputs,n) for n in NAMES}
                for n,t in tasks.items():
                    try:recovered[n]=t.result();state['completed'].append(n)
                    except Exception as error:state['failed'][n]=repr(error)
                    write_json(out/'status.json',state)
            if state['failed']:
                state['status']='needs_attention'
                return
            if len(recovered)!=21:raise ValueError('Expected21 validated follow-up parameters')
            target=legacy.collect(work,out,recovered,Path(parent['workspace']))
            payload=read_json(target)
            if len(payload['params'])!=916:raise ValueError('Not916 nuisances')
            # Preserve the pre-collection diagnostic state and every old output.
            old=out/'parent_before_collection.json'
            if not old.exists():write_json(old,parent)
            state.update(status='fits_complete',total_valid=916,json=str(target.relative_to(ROOT)),
                json_sha256=sha256(target),boundary_limited_parameters=payload['recovery_metadata']['boundary_limited_parameters'])
            parent.update(status='fits_complete',phase='collected_with_validated_neighbour_profiles',
                json=state['json'],json_sha256=state['json_sha256'],failed_nuisances=[],valid_nuisances=916,
                profile_validation_status=str((out/'status.json').relative_to(ROOT)),
                boundary_limited_parameters=state['boundary_limited_parameters'],
                fit_products={str(p.relative_to(ROOT)):sha256(p) for p in work.rglob('*.root')},
                plots_status='pending_original_plotImpacts')
            write_json(work/'impact_status.json',parent)
        finally:
            state.update(controller_pid=None,finished=time.time())
            write_json(out/'status.json',state)
        print({k:state.get(k) for k in ('status','total_valid','failed','boundary_limited_parameters')})


if __name__=='__main__':
    native_runtime(internal_path(WORK)/'profile_validation')
    main()
