"""Existing observed-CR background-only fit, with no observed SR and no new model.

Fits and original pull rendering are separate commands. --plot-only authors a
PDF; perform the PDF authoring/visual-QA procedure before/after that command.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import (native_runtime, grid_card,
    cr_channels, freeze_contract, CommandRunner, require_success, workspace, published_diagnostic,
    CommandFailure, process_active)

SOURCES = ['workflows/combine_cronly.py', 'workflows/diagnostic_common.py',
           'workflows/diagnostic_payload.py', 'workflows/extract_cronly_covariance.py',
           'workflows/renderers/cronly_pulls.py', 'workflows/combine_limits.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py']


def fit_command(workspace_path):
    return ['combine', '-M', 'FitDiagnostics', str(workspace_path), '-m', '120', '-n', '_cronly',
            '--skipSBFit', '--robustFit', '1', '--rMin', '0', '--rMax', '10',
            '--setParameters', 'r=0', '--freezeParameters', 'r',
            '--cminDefaultMinimizerStrategy', '0', '--cminDefaultMinimizerTolerance', '0.1',
            '--cminFallbackAlgo', 'Minuit2,Migrad,1:0.1']


def validate_fit(path):
    import ROOT as root
    source = root.TFile.Open(str(internal_path(path)))
    try:
        fit = source.Get('fit_b') if source and not source.IsZombie() else None
        if not fit:
            raise ValueError('fit_b is missing')
        result = dict(fit_status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()))
        if result['fit_status'] != 0 or result['covariance_quality'] < 2 or not math.isfinite(result['edm']):
            raise ValueError('CR fit did not pass status 0 / covariance quality >= 2 / finite EDM: ' + str(result))
        return result
    finally:
        if source:
            source.Close()


def plot(runner, output, state, ylim):
    pulls = output / 'pulls.json'
    if state.get('status') not in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
        raise ValueError('Pulls require a successful verified CR fit')
    if state.get('stage') != 'profile_pulls_complete':
        for name, digest in state['products'].items():
            path=internal_path(ROOT/name)
            if not path.is_file() or (path.suffix != '.root' and sha256(path) != digest):
                raise ValueError('CR fit/extraction product changed')
    stem = output / ('cronly_nuisance_pulls_ylim' + format(ylim, 'g').replace('.', '')+'_split2')
    # Same Python/scientific software and same plotting body as the original.
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(output / 'cache'),
               TMPDIR=str(output / 'tmp'))
    command = ['/hep2-scratch/twkim/envs/py38/bin/python', ROOT / 'workflows/diagnostic_payload.py',
               '--operation', 'plot_cronly', '--input', pulls, '--output-stem', stem, '--ylim', str(ylim)]
    figures = [stem.with_suffix('.pdf'),stem.parent/(stem.name+'_part1.png'),
               stem.parent/(stem.name+'_part2.png')]
    require_success(runner.run(command, output, output / 'plot.log', figures, env=env))
    state.update(status='generated_pending_visual_QA', figures={str(p.relative_to(ROOT)): sha256(p) for p in figures})
    return state


def recovery_commands(target):
    for strategy, tolerance, explicit in ((1, .01, False), (2, .001, False), (1, .001, True)):
        command = fit_command(target)
        command[command.index('--cminDefaultMinimizerStrategy')+1] = str(strategy)
        command[command.index('--cminDefaultMinimizerTolerance')+1] = str(tolerance)
        command += ['--X-rtd', 'MINIMIZER_MaxCalls=2000000']
        if explicit: command += ['--X-rtd', 'MINIMIZER_no_analytic']
        yield ('strategy%d%s' % (strategy, '_explicit' if explicit else '')), command


def covariance_commands(target, starting):
    for robust in (False,True):
        command=fit_command(target)
        command[command.index('--setParameters')+1]=','.join(
            ['r=0']+[n+'='+format(v,'.17g') for n,v in sorted(starting.items()) if n!='r'])
        command[command.index('--cminDefaultMinimizerStrategy')+1]='1'
        command[command.index('--cminDefaultMinimizerTolerance')+1]='0.001'
        command += ['--cminDefaultMinimizerPrecision','1e-8','--X-rtd','MINIMIZER_MaxCalls=2000000']
        if robust:command += ['--robustHesse','1']
        yield ('warm_robust_hesse' if robust else 'warm_precision8'),command


def robust_hesse_commands(target):
    for strategy,tolerance in ((0,.1),(1,.01)):
        command=fit_command(target)
        command[command.index('--cminDefaultMinimizerStrategy')+1]=str(strategy)
        command[command.index('--cminDefaultMinimizerTolerance')+1]=str(tolerance)
        command += ['--robustHesse','1','--X-rtd','MINIMIZER_MaxCalls=2000000','-v','2']
        yield 'robust_hesse_s'+str(strategy),command


def recover(manifest, output, covariance=False, robust=False):
    output = internal_path(output)
    card, origins = grid_card(manifest, 'cronly')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read_json(output/'state.json')
        if process_active(original) or original['origins'] != origins:
            raise ValueError('CR fit is active or belongs to another input')
        if original['status'] in ('fits_complete_pending_plots','generated_pending_visual_QA'):
            for name,digest in original['products'].items():
                if sha256(ROOT/name) != digest: raise ValueError('Completed CR product changed')
            return 0
        out = output/('fit_robust_hesse' if robust else 'fit_covariance_recovery' if covariance else 'fit_recovery')
        if covariance and (out/'state.json').exists():
            rejected=read_json(out/'state.json')
            attempts=rejected.get('attempts',[])
            if (process_active(rejected) or not attempts or any(a.get('exit_code')!=185 or
                    'outside its range' not in (ROOT/a['fit']).parent.joinpath('combine.log').read_text()
                    for a in attempts)):
                raise ValueError('Existing covariance recovery retained; not an initialization-parser failure')
            out=output/'fit_covariance_recovery_workspace_seed'
        out.mkdir(exist_ok=True)
        protected = {str(p.relative_to(ROOT)):sha256(p) for p in output.glob('*.root')}
        starting={};seed_file=None;omitted_seeds={}
        if covariance:
            import ROOT as root
            previous=read_json(output/'fit_recovery/state.json')
            if process_active(previous) or previous['status']!='needs_attention':
                raise ValueError('Requires stopped prior numerical CR recovery')
            seeds=[]
            for attempt in previous['attempts']:
                p=ROOT/attempt['fit'];f=root.TFile.Open(str(p));fit=f.Get('fit_b')
                if fit and fit.status()==0 and math.isfinite(fit.edm()):
                    pars=fit.floatParsFinal()
                    seeds.append((fit.edm(),str(p),{pars.at(i).GetName():pars.at(i).getVal() for i in range(pars.getSize())}))
                f.Close()
            if not seeds:raise ValueError('No converged warm-start fit')
            _,seed_file,starting=min(seeds,key=lambda s:s[0])
            source=root.TFile.Open(str(output/'workspace.root'));ws=source.Get('w')
            try:
                for name,value in list(starting.items()):
                    variable=ws.var(name)
                    if not variable or not variable.getMin()<=value<=variable.getMax():
                        omitted_seeds[name]=value;del starting[name]
            finally:source.Close()
            protected[str(Path(seed_file).relative_to(ROOT))]=sha256(seed_file)
        freeze_contract(out,origins,SOURCES,dict(original_contract=sha256(output/'contract.json'),
            protected=protected,signal_strength=0,maximum_attempts=2 if covariance or robust else 3,
            warm_start=seed_file,omitted_out_of_initial_workspace_range_seeds=omitted_seeds,physics_changed=False))
        if not (out/'original_state.json').exists(): write_json(out/'original_state.json',original)
        runner = CommandRunner(out)
        target = output/'workspace.root'
        if not target.exists(): raise ValueError('Original CR workspace missing')
        state = read_json(out/'state.json') if (out/'state.json').exists() else dict(attempts=[])
        state.update(status='running',pid=os.getpid(),origins=origins)
        try:
            accepted = None
            commands=(robust_hesse_commands(target) if robust else covariance_commands(target,starting)
                      if covariance else recovery_commands(target))
            for name,command in commands:
                here=out/name;here.mkdir(exist_ok=True)
                fit=here/'fitDiagnostics_cronly.root'
                receipt=runner.run(command,here,here/'combine.log',[fit])
                record=dict(name=name,exit_code=receipt.get('exit_code'),fit=str(fit.relative_to(ROOT)))
                text=(here/'combine.log').read_text(errors='replace') if (here/'combine.log').is_file() else ''
                dropped=[];in_dropped=False
                for line in text.splitlines():
                    if 'parameters were dropped' in line:in_dropped=True
                    elif in_dropped and line.strip().startswith('- '):dropped.append(line.strip()[2:])
                    elif in_dropped:in_dropped=False
                record['robust_hesse_dropped_parameters']=dropped
                try:
                    require_success(receipt)
                    record.update(validate_fit(fit));record['passed']=True
                except Exception as exc: record.update(passed=False,error=repr(exc))
                if not any(a['name']==name for a in state['attempts']): state['attempts'].append(record)
                write_json(out/'state.json',state)
                if record['passed']: accepted=record;break
            if accepted is None:
                state['status']='needs_attention';return 2
            fit=ROOT/accepted['fit'];parameters=out/'fit_parameters.json'
            require_success(runner.run(['python3',ROOT/'workflows/diagnostic_payload.py',
                '--operation','extract_cronly','--fit-diagnostics',fit,'--output',parameters],
                out,out/'extract.log',[parameters]))
            payload=read_json(parameters)
            dropped=accepted.get('robust_hesse_dropped_parameters',[])
            if any(payload['parameters'].get(n,{}).get('constraint_model')=='standard_normal' for n in dropped):
                raise ValueError('Robust Hessian dropped a plotted constrained nuisance; profile recovery required')
            payload['covariance_recovery']=dict(robust_hesse_dropped_parameters=dropped,
                warm_start=seed_file,model_parameters_fixed_or_removed=False)
            for name,item in payload['parameters'].items():
                if (not math.isfinite(item['value']) or not math.isfinite(item['error']) or item['error']<0
                    or any(item.get(k) is not None and not math.isfinite(item[k]) for k in ('pull','constraint'))):
                    raise ValueError('Invalid recovered CR parameter: '+name)
            payload.pop('covariance',None)
            payload.update(sr_observed_data_used=False,source_card=str(card),channels=original['channels'])
            pulls=output/'pulls.json'
            if pulls.exists() and read_json(pulls)!=payload: raise ValueError('Retain existing different pulls')
            write_json(pulls,payload)
            for name,digest in protected.items():
                if sha256(ROOT/name)!=digest: raise ValueError('Original CR ROOT changed')
            original.update(status='fits_complete_pending_plots',stage='fit_complete',pid=None,
                recovery_state=str((out/'state.json').relative_to(ROOT)),fit_status=accepted['fit_status'],
                covariance_quality=accepted['covariance_quality'],edm=accepted['edm'],
                products={str(p.relative_to(ROOT)):sha256(p) for p in (target,fit,parameters,pulls)},
                plotted_nuisances=sum(p['constraint_model']=='standard_normal' for p in payload['parameters'].values()))
            original.pop('error',None);write_json(output/'state.json',original)
            state['status']='fits_complete_pending_plots';return 0
        finally:
            state.update(pid=None,finished=time.time());write_json(out/'state.json',state)


def _run_once(manifest, output, plot_only=False, ylim=6.0):
    if not math.isfinite(ylim) or ylim <= 0:
        raise ValueError('Invalid pull range')
    output = internal_path(output)
    published=published_diagnostic(output,manifest,'cronly')
    if published is not None:
        with (output/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if plot_only:
                state=plot(CommandRunner(output),output,read_json(output/'state.json'),ylim)
                write_json(output/'state.json',state)
        return 0
    card, origins = grid_card(manifest, 'cronly')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (plot_only and (output/'fit_recovery/state.json').exists()):
            freeze_contract(output, origins, SOURCES, dict(signal_strength=0, workers=1, strategy=0, pull_ylim=ylim))
        elif read_json(output/'state.json')['origins'] != origins:
            raise ValueError('Recovered pull input provenance changed')
        runner = CommandRunner(output)
        state_path = output / 'state.json'
        state = read_json(state_path) if state_path.exists() else dict(
            status='running', started=time.time(), origins=origins, sr_observed_data_used=False,
            channels=cr_channels(card.read_text(), tuple(read_json(card.with_suffix('.manifest.json'))['years'])), workers=1, strategy=0, auto_mc_stats=[10, 1, 1],
            full_workflow_complete=False)
        try:
            state['pid'] = os.getpid()
            if plot_only:
                state = plot(runner, output, state, ylim)
            elif state.get('status') in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
                for name, digest in state['products'].items():
                    if sha256(ROOT / name) != digest:
                        raise ValueError('Completed CR fit/extraction product changed')
                validate_fit(output / 'fitDiagnostics_cronly.root')
                return 0
            else:
                target = workspace(runner, card, output, cr_only=True)
                fit = output / 'fitDiagnostics_cronly.root'
                state.update(status='running', stage='fit')
                write_json(state_path, state)
                require_success(runner.run(fit_command(target), output, output / 'combine.log', [fit]))
                state.update(validate_fit(fit))
                parameters = output / 'fit_parameters.json'
                require_success(runner.run(['python3', ROOT / 'workflows/diagnostic_payload.py',
                    '--operation', 'extract_cronly', '--fit-diagnostics', fit, '--output', parameters],
                    output, output / 'extract.log', [parameters]))
                payload = read_json(parameters)
                for name, record in payload['parameters'].items():
                    if (not math.isfinite(record['value']) or not math.isfinite(record['error'])
                            or record['error'] < 0 or any(record.get(k) is not None
                            and not math.isfinite(record[k]) for k in ('pull', 'constraint'))):
                        raise ValueError('Nonfinite/invalid extracted fit parameter: ' + name)
                payload.pop('covariance', None)  # full covariance stays in the remote extraction product
                payload.update(sr_observed_data_used=False, source_card=str(card), channels=state['channels'])
                pulls = output / 'pulls.json'
                if pulls.exists() and read_json(pulls) != payload:
                    raise ValueError('Existing pull payload is divergent')
                if not pulls.exists():
                    write_json(pulls, payload)
                state.update(status='fits_complete_pending_plots', stage='fit_complete', finished=time.time(),
                    plotted_nuisances=sum(p['constraint_model'] == 'standard_normal' for p in payload['parameters'].values()),
                    products={str(p.relative_to(ROOT)): sha256(p) for p in (target, fit, parameters, pulls)})
            grid_card(manifest, 'cronly')  # unchanged card and TH1 files after execution
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error), finished=time.time())
            raise
        finally:
            state['pid'] = None
            write_json(state_path, state)
    return 0


def cr_profile_command(target, name, strategy):
    from TROTASR.workflows.impact_legacy import fit_command
    command,tag=fit_command(target,name,0,strategy)
    for option in ('-t','--expectSignal'):
        index=command.index(option);del command[index:index+2]
    command[command.index('--setParameterRanges')+1]='r=0,10'
    command += ['--setParameters','r=0','--freezeParameters','r',
        '--cminDefaultMinimizerTolerance','0.01','--X-rtd','MINIMIZER_MaxCalls=2000000']
    return command,tag


def conditional_bracket_trial(fixed, inner, outer, recovery=False):
    for fraction in ((.5,.25,.75) if recovery else (.5,)):
        value=inner['theta']+fraction*(outer['theta']-inner['theta'])
        try:return fixed(value)
        except ValueError as error:
            if not recovery or not str(error).startswith('CR conditional minimization failed at '):raise
    raise ValueError('Three conditional bracket coordinates did not converge')


def profile_start_points(samples, value):
    """Independent nearby and crossing seeds; never alter nuisance bounds."""
    chosen=[]
    for ordered in (sorted(samples,key=lambda p:abs(p['theta']-value)),
                    sorted(samples,key=lambda p:abs(p['deltaNLL']-.5))):
        for sample in ordered:
            if sample['root'] not in {p['root'] for p in chosen}:
                chosen.append(sample)
                break
    return [None,*chosen]


def direct_cr_profile(output, out, name, runner, neighbour=False, bracket_recovery=False, multistart=False, full_start=False):
    """Bracket the actual CR-only profile, independently of Minos stepping."""
    from TROTASR.workflows.impact_recovery_legacy import tree_rows
    from TROTASR.workflows.profile_sgamma import tracked_start
    from TROTASR.workflows.recover_impacts import reconcile_seed_bounds
    import ROOT as root
    prior=read_json(output/'profile_pulls_warm_after_nll'/name/'state.json')
    base=list(prior['attempts'][0]['execution']['command'])
    point=out/name;point.mkdir(exist_ok=True)
    products={};snapshot=None
    if neighbour:
        for folder in sorted((output/'profile_pulls_direct'/name).glob('minimum_s*')):
            candidate=next(folder.glob('higgsCombine*.root'));fitpath=next(folder.glob('multidimfit*.root'))
            f=root.TFile.Open(str(fitpath));fit=f.Get('fit_mdf')
            good=fit and fit.status()==0 and math.isfinite(fit.edm())
            if good:minimum=dict(status=int(fit.status()),covQual=int(fit.covQual()),edm=float(fit.edm()))
            f.Close();rows=tree_rows(candidate,name)
            if good and len(rows)==1 and rows[0]['r']==0.:
                snapshot=candidate;center=rows[0]['theta']
                products.update({str(p.relative_to(ROOT)):sha256(p) for p in (candidate,fitpath)})
                break
    for strategy in (0,1,2):
        if snapshot is not None:break
        folder=point/('minimum_s'+str(strategy));folder.mkdir(exist_ok=True)
        tag='_cr_profile_min_s'+str(strategy)
        command=list(base)
        command[command.index('-n')+1]=tag
        command[command.index('--algo')+1]='none'
        command[command.index('--cminDefaultMinimizerStrategy')+1]=str(strategy)
        command[command.index('--setParameterRanges')+1]='r=0,10:'+name+'=-10,10'
        command += ['--saveWorkspace']
        candidate=folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
        fitpath=folder/('multidimfit'+tag+'.root')
        receipt=runner.run(command,folder,folder/'combine.log',[candidate,fitpath])
        if receipt['status']!='complete':continue
        f=root.TFile.Open(str(fitpath));fit=f.Get('fit_mdf')
        good=fit and fit.status()==0 and math.isfinite(fit.edm())
        if good:minimum=dict(status=int(fit.status()),covQual=int(fit.covQual()),edm=float(fit.edm()))
        f.Close()
        rows=tree_rows(candidate,name)
        if good and len(rows)==1 and rows[0]['r']==0. and math.isfinite(rows[0]['theta']):
            snapshot=candidate;center=rows[0]['theta'];products.update(receipt['products']);break
    if snapshot is None:raise ValueError('No converged CR profile minimum')
    f=root.TFile.Open(str(snapshot));ws=f.Get('w');pars=list(ws.obj('ModelConfig').GetNuisanceParameters())
    if full_start:pars += [p for p in ws.allVars() if p.getAttribute('flatParam') and not p.isConstant()]
    seed_bounds={p.GetName():([-math.inf,math.inf] if p.getAttribute('optimizeBounds') else
                            [float(p.getMin()),float(p.getMax())]) for p in pars}
    f.Close()
    reusable={}
    old_profiles=[output/'profile_pulls_neighbour']
    if bracket_recovery:old_profiles.append(output/'profile_pulls_neighbour_resume')
    if multistart:old_profiles.append(output/'profile_pulls_bracket')
    if full_start:old_profiles.append(output/'profile_pulls_multistart')
    if neighbour:
        for previous in old_profiles:
            if previous==out or not previous.exists():continue
            for p in (previous/'commands').glob('*.json'):
                receipt=read_json(p)
                if receipt['status']=='complete' and receipt['exit_code']==0 and not receipt.get('pid'):
                    reusable.update(receipt['products'])
    samples=[];seed_samples=[]
    if multistart:
        for previous in old_profiles:
            for product in sorted((previous/name).glob('point_*/higgsCombine*.root')):
                rel=str(product.relative_to(ROOT))
                if reusable.get(rel)!=sha256(product):continue
                rows=tree_rows(product,name)
                if (len(rows)==2 and all(math.isfinite(v) for row in rows for v in row.values())
                        and rows[-1]['r']==0. and rows[-1]['deltaNLL']>=-.005):
                    seed_samples.append(dict(rows[-1],root=rel,sha256=sha256(product)))
    def fixed(value):
        key='point_'+format(value,'.10f').replace('-','m').replace('.','p')
        if reusable and not multistart:
            for product in sorted(p for previous in old_profiles
                                  for p in (previous/name/key).glob('higgsCombine*.root')):
                rel=str(product.relative_to(ROOT))
                if reusable.get(rel)!=sha256(product):raise ValueError('Saved conditional ROOT changed')
                rows=tree_rows(product,name)
                if (len(rows)==2 and all(math.isfinite(v) for row in rows for v in row.values())
                        and rows[-1]['r']==0. and abs(rows[-1]['theta']-value)<2e-6
                        and rows[-1]['deltaNLL']>=-.005):
                    result=dict(rows[-1],root=rel,sha256=sha256(product))
                    samples.append(result);products[rel]=sha256(product);return result
        folder=point/key;folder.mkdir(exist_ok=True)
        starts_to_try=profile_start_points(seed_samples+samples,value) if multistart else [None]
        candidates=[]
        if full_start:
            for seed in seed_samples:
                if abs(seed['theta']-value)<2e-7:
                    candidates.append(seed);products[seed['root']]=seed['sha256']
        for start_index,start in enumerate(starts_to_try):
          for strategy in ((1,2) if multistart else (0,1)):
            tag='_cr_'+key+'_s'+str(strategy)+('_start'+str(start_index) if multistart else '')
            command=list(base)
            command[command.index('-d')+1]=str(snapshot)
            command[command.index('-n')+1]=tag
            command[command.index('--algo')+1]='fixed'
            command[command.index('--setParameters')+1]='r=0'
            command[command.index('--setParameterRanges')+1]='r=0,10:'+name+'=-10,10'
            command[command.index('--cminDefaultMinimizerStrategy')+1]=str(strategy)
            command[command.index('--cminDefaultMinimizerTolerance')+1]=('0.001' if multistart else '0.1' if strategy==0 else '0.01')
            command.remove('--saveFitResult')
            index=command.index('SETPARAMETERS_AFTER_NLL');del command[index-1:index+1]
            command += ['--snapshotName','MultiDimFit','--skipInitialFit','--fixedPointPOIs',name+'='+format(value,'.17g')]
            if neighbour:
                if start is not None or (samples and not multistart):
                    nearest=start if multistart else min(samples,key=lambda p:abs(p['theta']-value))
                    seed=dict(nearest,root=str(ROOT/nearest['root']))
                    starts=reconcile_seed_bounds(tracked_start(seed,set(seed_bounds)),seed_bounds)
                    command[-1]+=','+','.join(n+'='+format(v,'.17g') for n,v in starts.items() if n!=name)
                command += ['--trackParameters','rgx{.*}']
            if multistart:
                if '--cminFallbackAlgo' in command:
                    command[command.index('--cminFallbackAlgo')+1]='Minuit2,Migrad,2:0.001'
                else:command += ['--cminFallbackAlgo','Minuit2,Migrad,2:0.001']
                command += ['--X-rtd','MINIMIZER_no_analytic']
            product=folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
            receipt=runner.run(command,folder,folder/('s'+str(strategy)+('_start'+str(start_index) if multistart else '')+'.log'),[product])
            if receipt['status']!='complete':continue
            rows=tree_rows(product,name)
            if (len(rows)==2 and all(math.isfinite(v) for row in rows for v in row.values())
                    and rows[-1]['r']==0. and abs(rows[-1]['theta']-value)<2e-6
                    and rows[-1]['deltaNLL']>=-.005):
                result=dict(rows[-1],root=str(product.relative_to(ROOT)),sha256=sha256(product))
                products.update(receipt['products'])
                if not multistart:samples.append(result);return result
                candidates.append(result);break
        if candidates:
            result=min(candidates,key=lambda p:p['deltaNLL'])
            result=dict(result,independent_start_results=candidates)
            samples.append(result)
            write_json(point/'scan_progress.json',dict(samples=samples,center=center,latest=value))
            return result
        raise ValueError('CR conditional minimization failed at '+str(value))
    endpoints=[]
    for sign in (-1,1):
        inner=dict(theta=center,deltaNLL=0.)
        outer=None
        for distance in ((.25,.5,.75,1.,1.25,1.5,2.,3.,4.) if neighbour else (.25,.5,1.,2.,4.)):
            value=center+sign*distance
            if not -10.<value<10.:break
            trial=fixed(value)
            if trial['deltaNLL']>=.5:outer=trial;break
            inner=trial
        if outer is None:raise ValueError('No measured CR profile crossing')
        endpoint=None
        for _ in range(18):
            trial=conditional_bracket_trial(fixed,inner,outer,bracket_recovery)
            if abs(trial['deltaNLL']-.5)<=.002:endpoint=trial;break
            if trial['deltaNLL']<.5:inner=trial
            else:outer=trial
        if endpoint is None:raise ValueError('CR profile crossing did not reach NLL precision')
        endpoints.append(endpoint)
    low,high=endpoints
    if not low['theta']<center<high['theta']:raise ValueError('CR profile does not bracket minimum')
    result=dict(status='complete',method='direct_CR_profile_likelihood',central_fit=minimum,
        validation=dict(valid=True,entries=[[0.,center,0.],[0.,low['theta'],low['deltaNLL']],[0.,high['theta'],high['deltaNLL']]],
                        crossing_target=.5,crossing_tolerance=.002),
        profile_products=products,samples=samples,multistart=multistart,free_normalizations_in_starts=full_start)
    write_json(point/'state.json',result)
    return result


PROFILE_PHASES = ('profile_pulls', 'profile_pulls_minimizer', 'profile_pulls_warm',
    'profile_pulls_warm_after_nll', 'profile_pulls_gaussian_scan', 'profile_pulls_direct',
    'profile_pulls_neighbour', 'profile_pulls_neighbour_resume', 'profile_pulls_bracket',
    'profile_pulls_multistart', 'profile_pulls_full_start')


def prior_completed_profile(output, previous_profiles, name):
    """Keep the first accepted interval, including phases without copied records."""
    stop = PROFILE_PHASES.index(previous_profiles.name)
    for phase in PROFILE_PHASES[:stop + 1]:
        path = output / phase / name / 'state.json'
        if path.exists():
            record = read_json(path)
            if record['status'] == 'complete':
                return record
    return None


def recover_profile_pulls(manifest,output,workers,continuation=False,warm=False):
    """Measure plotted CR nuisance intervals directly, without a Hessian veto."""
    from TROTASR.workflows.impact_legacy import validate_fit as validate_profile
    import ROOT as root
    root.gROOT.SetBatch(True);root.EnableThreadSafety()
    if not 1<=workers<=90:raise ValueError('Invalid shared worker limit')
    output=internal_path(output);card,origins=grid_card(manifest,'cronly')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=read_json(output/'state.json')
        if process_active(parent) or parent['origins']!=origins:raise ValueError('CR input changed or active')
        previous=read_json(output/'fit_recovery/state.json')
        # Failed covariance validation records still retain the actual fit ROOT.
        candidates=[]
        for attempt in previous['attempts']:
            p=ROOT/attempt['fit'];f=root.TFile.Open(str(p));fit=f.Get('fit_b')
            if fit and fit.status()==0 and fit.covQual()>=2 and math.isfinite(fit.edm()):
                candidates.append((fit.edm(),p))
            f.Close()
        if not candidates:raise ValueError('No converged CR central fit')
        _,central=min(candidates,key=lambda x:x[0]);target=output/'workspace.root'
        delayed_warm=warm and (output/'profile_pulls_warm/state.json').exists()
        gaussian_scan=delayed_warm and (output/'profile_pulls_warm_after_nll/state.json').exists()
        direct_scan=gaussian_scan and (output/'profile_pulls_gaussian_scan/state.json').exists()
        neighbour_scan=direct_scan and (output/'profile_pulls_direct/state.json').exists()
        resume_neighbour=neighbour_scan and (output/'profile_pulls_neighbour/state.json').exists()
        bracket_recovery=resume_neighbour and (output/'profile_pulls_neighbour_resume/state.json').exists()
        multistart=bracket_recovery and (output/'profile_pulls_bracket/state.json').exists()
        full_start=multistart and (output/'profile_pulls_multistart/state.json').exists()
        previous_profiles=output/('profile_pulls_neighbour' if resume_neighbour else 'profile_pulls_direct' if neighbour_scan else 'profile_pulls_gaussian_scan' if direct_scan else 'profile_pulls_warm_after_nll' if gaussian_scan else 'profile_pulls_warm' if delayed_warm else 'profile_pulls_minimizer' if warm else 'profile_pulls') if continuation or warm else None
        if bracket_recovery:previous_profiles=output/'profile_pulls_neighbour_resume'
        if multistart:previous_profiles=output/'profile_pulls_bracket'
        if full_start:previous_profiles=output/'profile_pulls_multistart'
        if previous_profiles:
            prior=read_json(previous_profiles/'state.json')
            if process_active(prior) or prior['status']!='needs_attention':raise ValueError('Requires terminal prior CR profiles')
            if gaussian_scan and set(prior['failed'])!={'CMS_NPS26012_eff_g_id_2025'}:
                raise ValueError('Gaussian range recovery is specific to the diagnosed photon-ID endpoint')
        starting={};omitted={}
        if warm:
            f=root.TFile.Open(str(central));pars=f.Get('fit_b').floatParsFinal()
            starting={pars.at(i).GetName():float(pars.at(i).getVal()) for i in range(pars.getSize())}
            f.Close()
            f=root.TFile.Open(str(target));ws=f.Get('w')
            for name,value in list(starting.items()):
                variable=ws.var(name)
                optimized=variable and variable.getAttribute('optimizeBounds')
                if not variable or (not variable.getMin()<=value<=variable.getMax()
                                    and not (delayed_warm and optimized)):
                    omitted[name]=value;del starting[name]
            f.Close()
        out=output/('profile_pulls_full_start' if full_start else 'profile_pulls_multistart' if multistart else 'profile_pulls_bracket' if bracket_recovery else 'profile_pulls_neighbour_resume' if resume_neighbour else 'profile_pulls_neighbour' if neighbour_scan else 'profile_pulls_direct' if direct_scan else 'profile_pulls_gaussian_scan' if gaussian_scan else 'profile_pulls_warm_after_nll' if delayed_warm else 'profile_pulls_warm' if warm else 'profile_pulls_minimizer' if continuation else 'profile_pulls');out.mkdir(exist_ok=True)
        files=[*SOURCES,'workflows/impact_legacy.py']
        if direct_scan:files+=['workflows/profile_sgamma.py','workflows/recover_impacts.py']
        contract=freeze_contract(out,origins,files,dict(
            central_fit=str(central.relative_to(ROOT)),central_sha256=sha256(central),
            workspace_sha256=sha256(target),signal_strength=0,observed_sr_used=False,
            maximum_configurations=2 if continuation else 3,workers=workers,uncertainty_method='profile_likelihood',
            warm_start=starting,omitted_out_of_workspace_range_seeds=omitted,
            set_parameters_after_nll=delayed_warm,
            gaussian_computational_range={'CMS_NPS26012_eff_g_id_2025':[-10,10]} if gaussian_scan else {},
            direct_profile=direct_scan,neighbour_seed=neighbour_scan,maximum_bracket_steps=9 if neighbour_scan else 5,maximum_bisection_steps=18,
            crossing_fractions=[.5,.25,.75] if bracket_recovery else [.5],
            independent_conditional_starts=3 if multistart else 1,
            explicit_statistical_nuisances=multistart,
            free_normalizations_in_starts=full_start,
            previous_profile_state_sha256=sha256(previous_profiles/'state.json') if previous_profiles else None,physics_changed=False))
        runner=CommandRunner(out)
        original=out/'central_parameters.json'
        require_success(runner.run(['python3',ROOT/'workflows/diagnostic_payload.py','--operation','extract_cronly',
            '--fit-diagnostics',central,'--output',original],out,out/'extract.log',[original]))
        payload=read_json(original)
        names=sorted(n for n,p in payload['parameters'].items() if p['constraint_model']=='standard_normal')
        f=root.TFile.Open(str(target));ws=f.Get('w')
        bounds={n:[ws.var(n).getMin(),ws.var(n).getMax()] for n in names};f.Close()
        state=dict(status='running',pid=os.getpid(),started=time.time(),parameters=names,completed=[],failed={},workers=workers)
        write_json(out/'state.json',state)
        def one(name):
            if previous_profiles:
                prior_record=prior_completed_profile(output,previous_profiles,name)
                if prior_record is not None:
                    checked=validate_profile(ROOT/prior_record['folder'],name,prior_record['tag'],bounds[name])
                    if not checked['valid'] or checked['sha256']!=prior_record['validation']['sha256']:
                        raise ValueError('Completed CR profile changed')
                    return prior_record
            if direct_scan:return direct_cr_profile(output,out,name,runner,neighbour_scan,bracket_recovery,multistart,full_start)
            point=out/name;point.mkdir(exist_ok=True)
            if (point/'state.json').exists():
                saved=read_json(point/'state.json')
                if saved['status']=='complete':
                    checked=validate_profile(ROOT/saved['folder'],name,saved['tag'],bounds[name])
                    if checked['valid'] and checked['sha256']==saved['validation']['sha256']:return saved
                raise ValueError('Existing failed profile retained')
            attempts=[]
            for strategy in ((0,1) if continuation else (0,1,2)):
                folder=point/('strategy'+str(strategy));folder.mkdir(exist_ok=True)
                command,tag=cr_profile_command(target,name,strategy)
                if gaussian_scan:
                    if payload['parameters'][name]['constraint_model']!='standard_normal':
                        raise ValueError('Cannot extend a physical rate-parameter range')
                    command[command.index('--setParameterRanges')+1]='r=0,10:'+name+'=-10,10'
                if warm:
                    command[command.index('--setParameters')+1]=','.join(
                        ['r=0']+[n+'='+format(v,'.17g') for n,v in sorted(starting.items()) if n!='r'])
                    command[command.index('--cminDefaultMinimizerTolerance')+1]=str((.1,.01,.001)[strategy])
                    command += ['--cminDefaultMinimizerPrecision','1e-8','-v','2']
                    if delayed_warm:command += ['--X-rtd','SETPARAMETERS_AFTER_NLL']
                if continuation:
                    command += ['--cminDefaultMinimizerAlgo','Combined','--cminDefaultMinimizerPrecision','1e-8','-v','3']
                    command[command.index('--cminDefaultMinimizerTolerance')+1]='0.1' if strategy==0 else '0.001'
                    if strategy==1:command += ['--X-rtd','MINIMIZER_no_analytic']
                product=folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
                receipt=runner.run(command,folder,folder/'combine.log',[product])
                checked=validate_profile(folder,name,tag,bounds[name])
                if checked['valid'] and gaussian_scan and any(abs(row[1])>=9.999 for row in checked['entries']):
                    checked.update(valid=False,error='Gaussian interval still touches computational scan range')
                if checked['valid'] and any(row[0]!=0. for row in checked['entries']):
                    checked.update(valid=False,error='CR-only profile did not keep r=0')
                attempts.append(dict(command=command,execution=receipt,validation=checked))
                if receipt['exit_code']==0 and checked['valid']:
                    record=dict(status='complete',folder=str(folder.relative_to(ROOT)),tag=tag,
                        validation=checked,attempts=attempts);write_json(point/'state.json',record);return record
            write_json(point/'state.json',dict(status='needs_attention',attempts=attempts))
            raise ValueError('Finite CR profile attempts exhausted')
        try:
            recovered={}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(one,n):n for n in names}
                for future in as_completed(futures):
                    name=futures[future]
                    try:recovered[name]=future.result();state['completed'].append(name)
                    except Exception as exc:state['failed'][name]=repr(exc)
                    write_json(out/'state.json',state)
            if state['failed']:state['status']='needs_attention';return 2
            grid_card(manifest,'cronly')
            if sha256(central)!=contract['options']['central_sha256'] or sha256(target)!=contract['options']['workspace_sha256']:
                raise ValueError('Original CR fit/workspace changed')
            for name,record in recovered.items():
                rows=record['validation']['entries'];center,lower,upper=[r[1] for r in rows]
                item=payload['parameters'][name]
                item.update(value=center,pull=center,error=(upper-lower)/2.,constraint=(upper-lower)/2.,
                    error_low=center-lower,error_high=upper-center,uncertainty_method='profile_likelihood')
            payload.pop('covariance',None)
            payload.update(sr_observed_data_used=False,source_card=str(card),channels=parent['channels'],
                plotted_uncertainties='asymmetric_profile_likelihood',global_covariance_accepted=False)
            pulls=output/'pulls.json'
            if pulls.exists():raise ValueError('Existing pull payload retained')
            write_json(pulls,payload)
            products={str(p.relative_to(ROOT)):sha256(p) for p in (target,central,original,pulls)}
            for record in recovered.values():
                if 'profile_products' in record:products.update(record['profile_products'])
                else:products[str((ROOT/record['folder']/('higgsCombine'+record['tag']+'.MultiDimFit.mH120.root')).relative_to(ROOT))]=record['validation']['sha256']
            parent.update(status='fits_complete_pending_plots',stage='profile_pulls_complete',pid=None,
                recovery_status=str((out/'state.json').relative_to(ROOT)),products=products,
                fit_status=payload['fit_status'],covariance_quality=payload['covariance_quality'],
                global_covariance_accepted=False,plotted_nuisances=len(names),
                plotted_uncertainty_method='profile_likelihood')
            parent.pop('error',None);write_json(output/'state.json',parent)
            state['status']='fits_complete_pending_plots'
            return 0
        finally:
            state.update(pid=None,finished=time.time());write_json(out/'state.json',state)


def run(manifest, output, plot_only=False, ylim=6.0):
    """Reuse accepted products; exhaust applicable finite recovery on failure."""
    try:
        return _run_once(manifest, output, plot_only, ylim)
    except (CommandFailure, ValueError) as error:
        output = internal_path(output)
        state_path = output / 'state.json'
        if plot_only or not state_path.is_file():
            raise
        state = read_json(state_path)
        if state.get('stage') != 'fit' or not (output / 'workspace.root').is_file():
            raise
        failures = [dict(method='initial_fit', outcome='failed', reason=str(error))]
    for name, options in (('minimizer', {}), ('covariance', {'covariance': True}),
                          ('robust_hesse', {'robust': True})):
        try:
            result = recover(manifest, output, **options)
        except BlockingIOError:
            raise
        except ValueError as error:
            failures.append(dict(method=name, outcome='inapplicable', reason=str(error)))
        else:
            failures.append(dict(method=name, outcome='complete' if result == 0 else 'exhausted'))
            if result == 0:
                return 0
    state = read_json(state_path)
    state.update(recovery_attempts=failures, status='needs_attention', pid=None)
    write_json(state_path, state)
    return 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'stats/run3/grid/manifest.json')
    p.add_argument('--output', type=Path, default=ROOT/'stats/run3/cronly')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--pull-ylim', type=float, default=6.0)
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    native_runtime(a.output)
    raise SystemExit(run(a.manifest, a.output, a.plot_only, a.pull_ylim))
