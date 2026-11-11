"""Conditional profile commands and numerical acceptance used by the impacts stage."""
import math
from TROTASR.utils.io import write_json, sha256
from TROTASR.workflows.impact_recovery_legacy import tree_rows

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
