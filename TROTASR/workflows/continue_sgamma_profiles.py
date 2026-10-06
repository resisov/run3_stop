"""Bounded neighbour-seeded conditional profiles; unchanged likelihood/limits.

Combine doFixedPoint computes nll0 before applying --fixedPointPOIs. Passing
the previous conditional nuisance values in that option changes only the
starting point: only the scanned POI is fixed, while other POIs/nuisances float.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import math
import os
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.workflows.diagnostic_common import native_runtime,grid_card,freeze_contract,CommandRunner
from TROTASR.workflows.profile_sgamma import WORK,MANIFEST,NAMES,VALUES,fit


def tracked_start(point, allowed):
    import ROOT as root
    src=root.TFile.Open(point['root'])
    try:
        tree=src.Get('limit')
        if tree.GetEntries()!=2:
            raise ValueError('No converged conditional seed')
        tree.GetEntry(1)
        result={name:float(getattr(tree,'trackedParam_'+name)) for name in allowed}
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite conditional seed')
        return result
    finally:
        src.Close()


def chain(work,out,runner,inputs,name,value):
    allowed=set(inputs['bestfit'])
    # Exactly one recorded warm-start path: previous valid coordinate, failed
    # interior coordinate, then the original physical lower boundary.
    seedvalue=2*value
    first=fit(runner,out,inputs,name,seedvalue,'seed')
    if first['status']!='complete':
        return dict(status='failed',stage='seed',points=[first])
    seeds=tracked_start(first,allowed)
    second=fit(runner,out,inputs,name,value,'interior',seeds=seeds)
    if second['status']!='complete':
        return dict(status='failed',stage='interior',points=[first,second])
    seeds=tracked_start(second,allowed)
    third=fit(runner,out,inputs,name,0.,'boundary',seeds=seeds)
    return dict(status='boundary_evaluated' if third['status']=='complete' else 'failed',
                stage='boundary',points=[first,second,third])


def main():
    import ROOT as root
    root.gROOT.SetBatch(True)
    root.EnableThreadSafety()
    work=internal_path(WORK);out=work/'profile_warmstart';out.mkdir(exist_ok=True)
    with (work/'impact.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,origins=grid_card(internal_path(MANIFEST),'impact')
        prior=read_json(work/'profile_search/status.json')
        if sorted(prior['failed'])!=NAMES or prior['controller_pid'] is not None:
            raise ValueError('Not the two terminal Sgamma failures')
        inputs=read_json(work/'direct_profile/inputs.json')
        for key in ('snapshot','asimov'):
            if sha256(inputs[key])!=inputs[key+'_sha256']:
                raise ValueError('Changed inputs')
        freeze_contract(out,origins,['workflows/continue_sgamma_profiles.py','workflows/profile_sgamma.py'],dict(
            parameters=NAMES,maximum_new_fits=6,workers=2,
            evidence='Verbose Migrad: no improvement at first line search; EDM3230/3939; below2000000call limit',
            numerical_change='Neighbour conditional nuisance starting values only, applied after common nll0 evaluation',
            physical_model_unchanged=True,rate_bounds=[0,10],r_bounds=[0,20],target=.5,tolerance=.002))
        runner=CommandRunner(out)
        state=dict(status='running',controller_pid=os.getpid(),started=time.time(),results={})
        write_json(out/'status.json',state)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks={n:pool.submit(chain,work,out,runner,inputs,n,v) for n,v in zip(NAMES,VALUES)}
                state['results']={n:t.result() for n,t in tasks.items()}
            state['status']='studied'
        finally:
            state.update(controller_pid=None,finished=time.time())
            write_json(out/'status.json',state)
        print({n:dict(status=x['status'],points=[{k:p.get(k) for k in ('value','status','deltaNLL','error')} for p in x['points']]) for n,x in state['results'].items()})


if __name__=='__main__':
    native_runtime(internal_path(WORK)/'profile_warmstart')
    main()
