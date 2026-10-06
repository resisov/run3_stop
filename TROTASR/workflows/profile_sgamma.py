"""Finite, provenance-checked diagnostics for the two SR118 Sgamma profiles.

Never changes the likelihood, physical bounds, Asimov data or validation target.
Diagnostic outputs are not accepted as endpoints merely because Combine exits0.
"""
import argparse
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
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import native_runtime, grid_card, CommandRunner, freeze_contract
from TROTASR.workflows.impact_recovery_legacy import tree_rows

WORK = 'stats/nominal_20260924_sr118/impacts'
MANIFEST = 'stats/nominal_20260924_sr118/grid/manifest.json'
NAMES = ['CMS_NPS26012_sgamma_shape_lowdm_Nb2plus_bin1_' + str(y) for y in (2024,2025)]
VALUES = [0.06019309535622597, 0.06167682632803917]


def fixed_command(inputs, name, value, tag, extra=(), seeds=None):
    start = {name: float(value)}
    start.update({k: float(v) for k,v in (seeds or {}).items() if k != name})
    assignments = ','.join(k+'='+format(v,'.15g') for k,v in start.items())
    command = ['combine','-M','MultiDimFit','-d',inputs['snapshot'],'-m','120','-n',tag,
        '--algo','fixed','--redefineSignalPOIs','r','-P',name,'--floatOtherPOIs','1',
        '--saveInactivePOI','1','--fixedPointPOIs',assignments,
        '--snapshotName','impact_initial','--skipInitialFit','-t','-1','--expectSignal','1',
        '--toysFile',inputs['asimov'],'--setParameterRanges','r=0,20',
        '--cminDefaultMinimizerStrategy','0','--cminDefaultMinimizerTolerance','0.0001',
        '--X-rtd','MINIMIZER_no_analytic','--X-rtd','MINIMIZER_MaxCalls=2000000',
        '--trackParameters','rgx{.*}','-v','3']
    extra=list(extra)
    for option in ('--cminDefaultMinimizerStrategy','--cminDefaultMinimizerTolerance'):
        if option in extra:
            pos=extra.index(option)
            command[command.index(option)+1]=extra[pos+1]
            del extra[pos:pos+2]
    return command+extra


def fit(runner, out, inputs, name, value, label, extra=(), seeds=None):
    folder = out / name / label
    folder.mkdir(parents=True,exist_ok=True)
    tag = '_sgamma_'+name[-4:]+'_'+label
    product = folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
    command = fixed_command(inputs,name,value,tag,extra,seeds)
    record = runner.run(command,folder,folder/'fit.log',[product])
    result = dict(name=name,value=value,configuration=label,command=command,
                  root=str(product),status='failed',command_status=record['status'])
    try:
        rows = tree_rows(product,name)
        result['rows']=rows
        if record['status']!='complete' or len(rows)!=2:
            raise ValueError('Conditional fit must commit two finite rows')
        point=rows[-1]
        if not all(math.isfinite(v) for r in rows for v in r.values()):
            raise ValueError('Nonfinite profile point')
        if not math.isclose(point['theta'],value,rel_tol=2e-6,abs_tol=2e-7):
            raise ValueError('Wrong fixed coordinate')
        if point['deltaNLL'] < -.005:
            raise ValueError('Below common minimum')
        result.update(point,status='complete',sha256=sha256(product))
    except Exception as error:
        result['error']=repr(error)
    write_json(folder/'state.json',result)
    return result


def main():
    work=internal_path(WORK)
    out=work/'profile_numerics'
    out.mkdir(exist_ok=True)
    with (work/'impact.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,origins=grid_card(internal_path(MANIFEST),'impact')
        prior=read_json(work/'profile_search/status.json')
        if sorted(prior['failed'])!=NAMES or prior['controller_pid'] is not None:
            raise ValueError('Requires exactly the two terminal Sgamma failures')
        inputs=read_json(work/'direct_profile/inputs.json')
        for pathkey,hashkey in [('snapshot','snapshot_sha256'),('asimov','asimov_sha256')]:
            if sha256(inputs[pathkey])!=inputs[hashkey]:
                raise ValueError('Changed common profile inputs')
        freeze_contract(out,origins,['workflows/profile_sgamma.py'],dict(
            parameters=NAMES,phase='diagnose',maximum_new_fits=2,workers=2,
            reason='Identify interior fixed-point minimizer failure at verbosity3; no repeated old journal',
            likelihood_changed=False,common_minimum_unchanged=True,rate_bounds=[0,10],r_bounds=[0,20]))
        runner=CommandRunner(out)
        state=dict(status='running',controller_pid=os.getpid(),started=time.time(),results=[])
        write_json(out/'status.json',state)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks=[pool.submit(fit,runner,out,inputs,n,v,'diagnostic') for n,v in zip(NAMES,VALUES)]
                state['results']=[t.result() for t in tasks]
            state['status']='diagnosed'
        finally:
            state.update(controller_pid=None,finished=time.time())
            write_json(out/'status.json',state)
        print({r['name']:{k:r.get(k) for k in ('status','deltaNLL','error')} for r in state['results']})


if __name__=='__main__':
    native_runtime(internal_path(WORK)/'profile_numerics')
    main()
