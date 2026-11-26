"""Observed-CR background-only fits and saved diagnostics, with no observed SR.

The current provisional result is reused unless a refit is requested. Original pull
rendering authors a PDF and requires visual QA.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.selection_contract import require_current_llcr
from TROTASR.utils.io import read_json, write_json
from TROTASR.workflows.diagnostic_common import (native_runtime, grid_card,
    cr_channels, freeze_contract, CommandRunner, require_success, workspace,
    published_diagnostic, admitted_popen, live_resources, CommandFailure, process_active)

CRONLY_MODEL_REVISION = 'photon_gnn_highdm_llcr_mtb175'
RUN3 = ROOT / 'stats/run3'
LLCR_MTB = RUN3 / 'new_signal/llcr_mtb'
SOURCES = ['workflows/combine_cronly.py', 'workflows/diagnostic_common.py',
           'workflows/diagnostic_payload.py', 'workflows/extract_cronly_covariance.py',
           'workflows/renderers/cronly_pulls.py', 'workflows/combine_limits.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py']
PROFILE_WARM_STAGES = ('profile_pulls_warm', 'profile_pulls_warm_after_nll',
    'profile_pulls_gaussian_scan', 'profile_pulls_direct', 'profile_pulls_neighbour',
    'profile_pulls_neighbour_resume', 'profile_pulls_bracket', 'profile_pulls_multistart',
    'profile_pulls_full_start')


def fit_command(workspace_path):
    return ['combine', '-M', 'FitDiagnostics', str(workspace_path), '-m', '120', '-n', '_cronly',
            '--skipSBFit', '--robustFit', '1', '--rMin', '0', '--rMax', '10',
            '--setParameters', 'r=0', '--freezeParameters', 'r',
            '--cminDefaultMinimizerStrategy', '0', '--cminDefaultMinimizerTolerance', '0.1',
            '--cminFallbackAlgo', 'Minuit2,Migrad,1:0.1']


def validate_fit(path, *, photon_gnn=False, both_nb=False):
    import ROOT as root
    source = root.TFile.Open(str(internal_path(path)))
    try:
        fit = source.Get('fit_b') if source and not source.IsZombie() else None
        if not fit:
            raise ValueError('fit_b is missing')
        result = dict(fit_status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()))
        if result['fit_status'] != 0 or result['covariance_quality'] < 2 or not math.isfinite(result['edm']):
            raise ValueError('CR fit did not pass status 0 / covariance quality >= 2 / finite EDM: ' + str(result))
        if photon_gnn:
            from TROTASR.workflows.cr_linked_workspace import GNN_BY_GROUP, PHOTON_GROUPS, RECOIL_BY_GROUP
            groups = PHOTON_GROUPS if both_nb else ("Nb1",)
            gnn = [name for nb in groups for name in GNN_BY_GROUP[nb]]
            recoil = [name for nb in groups for name in RECOIL_BY_GROUP[nb]]
            floats = fit.floatParsFinal()
            fixed = fit.constPars().find('r')
            tree = source.Get('tree_fit_b')
            r = tree.GetLeaf('r') if tree else None
            if (floats.find('r') or (fixed and fixed.getVal() != 0)
                    or (r and (tree.GetEntry(0) <= 0 or r.GetValue() != 0))
                    or (not fixed and not r)
                    or any(not floats.find(name) for name in gnn)
                    or any(floats.find(name) for name in recoil)):
                raise ValueError('Fit must keep r=0 and float only the adopted low Photon GNN parametrization')
        return result
    finally:
        if source:
            source.Close()


def retain_pulls(path, payload):
    """Preserve numerical results when only the card location has changed."""
    if path.is_file():
        saved = read_json(path)
        if ({key: value for key, value in saved.items() if key != 'source_card'}
                != {key: value for key, value in payload.items() if key != 'source_card'}):
            raise ValueError('Existing pull payload is divergent')
    else:
        write_json(path, payload)


def central_fits(root, attempts):
    """Read only available central results from this diagnostic's own attempts."""
    for attempt in attempts:
        path = internal_path(ROOT/attempt['fit'])
        if not path.is_file():
            continue
        source = root.TFile.Open(str(path))
        try:
            fit = source.Get('fit_b') if source and not source.IsZombie() else None
            if fit and fit.status() == 0 and math.isfinite(fit.edm()):
                yield path, fit
        finally:
            if source:
                source.Close()


def refit_current(manifest, output, *, initial_values=None):
    """Repeat the selected CR-only fit without replacing its published products."""
    output = internal_path(output)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        published = published_diagnostic(output, manifest, 'cronly')
        if not published or published.get('status') != 'provisional':
            raise ValueError('The current provisional CR-only workspace is required')
        target = internal_path(output / published['workspace'])
        suffix = 'refit_fixed'
        if initial_values is not None:
            assignments = [value.split('=',1) for value in initial_values.split(',')]
            if any(len(item) != 2 or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',item[0])
                   for item in assignments):
                raise ValueError('Expected explicit NAME=value initial parameters')
            values = {name:float(value) for name,value in assignments}
            if (len(values) != len(assignments) or values.get('r', 0) != 0
                    or not all(map(math.isfinite,values.values()))):
                raise ValueError('Initial parameters must be unique and finite with r=0')
            values.setdefault('r', 0)
            initial_values = ','.join(name+'='+format(value, '.17g') for name, value in values.items())
            suffix = 'refit_matched'
        fit = output / ('fitDiagnostics_cronly_' + suffix + '.root')
        log = output / (suffix + '.log')
        minimizer_log = output / (suffix + '_minimizer.log')
        receipt = output / 'commands' / ('cronly_' + suffix + '.json')
        if any(path.exists() for path in (receipt, fit, log, minimizer_log)):
            raise FileExistsError('The current CR-only refit has already been attempted')
        slots, memory = live_resources()
        if slots >= 116 or memory < 36 * 1024**3 or shutil.disk_usage(output).free < 80 * 1024**3:
            raise RuntimeError('Insufficient hep2 headroom for the CR-only refit')
        command = fit_command(target)
        command[command.index('-n') + 1] = '_cronly_' + suffix
        command[command.index('--cminDefaultMinimizerStrategy') + 1] = '0' if initial_values is not None else '1'
        if initial_values is not None:
            command[command.index('--setParameters') + 1] = initial_values
        command[command.index('--cminDefaultMinimizerTolerance') + 1] = '0.001'
        command += ['--X-rtd', 'MINIMIZER_MaxCalls=2000000', '--X-rtd', 'MINIMIZER_no_analytic']
        # Combine v10.5.1 appends to VectorizedHistFunc::yvals_ on a data reset.
        # A fresh NLL keeps the same bins and avoids writing past partialSum_.
        command += ['--forceRecreateNLL', '-v', '1']
        state = dict(status='running', command=command, started=time.time(),
                     controller_pid=os.getpid(), pid=None, sr_observed_data_used=False,
                     nll_cache_recreated=True, matched_initialization=initial_values is not None)
        write_json(receipt, state)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            with tempfile.TemporaryDirectory(prefix='.fit-', dir=output) as scratch:
                scratch = Path(scratch)
                with log.open('x') as stream:
                    proc = admitted_popen(command, cwd=scratch, stdin=subprocess.DEVNULL,
                                          stdout=stream, stderr=subprocess.STDOUT)
                    state['pid'] = proc.pid
                    try:
                        proc_stat = Path(f'/proc/{proc.pid}/stat').read_text()
                        state['start_ticks'] = int(proc_stat.rsplit(')', 1)[1].split()[19])
                        write_json(receipt, state)
                    finally:
                        while True:
                            try:
                                state.update(exit_code=proc.wait(), pid=None)
                                break
                            except KeyboardInterrupt:
                                continue
                for source, destination in ((scratch / fit.name, fit),
                                            (scratch / 'combine_logger.out', minimizer_log)):
                    if source.is_file() and source.stat().st_size:
                        os.replace(source, destination)
            if fit.is_file():
                import ROOT
                source = ROOT.TFile.Open(str(fit))
                try:
                    result = (source.Get('fit_b') if source and not source.IsZombie()
                              and not source.TestBit(ROOT.TFile.kRecovered) else None)
                    if result:
                        tree = source.Get('tree_fit_b')
                        strength = (float(tree.GetLeaf('r').GetValue()) if tree
                                    and tree.GetLeaf('r') and tree.GetEntry(0) > 0 else None)
                        state.update(fit_status=int(result.status()), covariance_quality=int(result.covQual()),
                                     signal_strength=strength,
                                     float_parameters=int(result.floatParsFinal().getSize()),
                                     invalid_nll=int(result.numInvalidNLL()),
                                     fit_steps={str(result.statusLabelHistory(i)): int(result.statusCodeHistory(i))
                                                for i in range(result.numStatusHistory())},
                                     edm=float(result.edm()) if math.isfinite(result.edm()) else None)
                finally:
                    if source:
                        source.Close()
            good = (state['exit_code'] == 0 and state.get('fit_status') == 0
                    and state.get('covariance_quality', -1) >= 2 and state.get('edm') is not None
                    and state.get('signal_strength') == 0)
            state['status'] = 'complete' if good else 'needs_attention'
        except Exception as error:
            state.update(status='needs_attention', error=str(error))
            raise
        finally:
            state['finished'] = time.time()
            write_json(receipt, state)
        print(json.dumps({k:v for k,v in state.items() if k != 'command'}, sort_keys=True))
        return 0 if good else 1


def plot(runner, output, state, ylim):
    pulls = output / 'pulls.json'
    if state.get('status') not in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
        raise ValueError('Pulls require a successful verified CR fit')
    if state.get('stage') != 'profile_pulls_complete':
        for name in state['products']:
            path = internal_path(ROOT/name)
            if not path.is_file() or not path.stat().st_size:
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
    state.update(status='generated_pending_visual_QA', figures=dict.fromkeys(str(p.relative_to(ROOT)) for p in figures))
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


def _changed_cr_model(manifest, output):
    """The LLCR cut revision must never overwrite the frozen CR fit."""
    manifest, output = internal_path(manifest), internal_path(output)
    current=RUN3/'top_pt'
    if manifest==current/'grid/manifest.json' or output==current/'cronly':
        if manifest!=current/'grid/manifest.json' or output!=current/'cronly':
            raise ValueError('Bind the TT CR-only fit to its own updated grid')
        m=read_json(manifest)
        if not m.get('tt_card_input') or m.get('bins')!=208 or m.get('floating_normalization_parameters')!=44:
            raise ValueError('Missing TT and high-tail card inputs')
        return True
    selected = (manifest == LLCR_MTB/'grid/manifest.json' or output == LLCR_MTB/'cronly')
    if selected and (manifest != LLCR_MTB/'grid/manifest.json' or output != LLCR_MTB/'cronly'
            or read_json(manifest).get('highdm_llcr_mtb_min_gev') != 175.0):
        raise ValueError('Expected the changed LLCR grid and its separate CR-only result path')
    return selected


def _linked_cr_products(output, linked):
    return (output/'linked_card.txt', output/'linked_data_obs.root') if linked else ()


def _linked_cr_workspace(runner, card, manifest, output):
    """Keep the native card preparation, conversion and graft receipts separate."""
    linked_card, linked_data = _linked_cr_products(output, True)
    native, target = output/'native_cr_workspace.root', output/'workspace.root'
    worker = ['python3', ROOT/'workflows/combine_cronly.py', '--native-runtime',
              '--manifest', manifest, '--output', output, '--workspace-step']
    require_success(runner.run(
        [*worker, 'prepare', '--workspace-input', card], output,
        output/'workspace_prepare.log', [linked_card, linked_data]))
    require_success(runner.run(
        ['text2workspace.py', linked_card, '-m', '120', '-o', native, '--X-allow-no-signal'],
        output, output/'workspace_native.log', [native]))
    require_success(runner.run(
        [*worker, 'link', '--workspace-input', native], output,
        output/'workspace_link.log', [target]))
    return target


def recover(manifest, output, covariance=False, robust=False):
    output = internal_path(output)
    linked = _changed_cr_model(manifest, output)
    card, origins = grid_card(manifest, 'cronly')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read_json(output/'state.json')
        if process_active(original):
            raise ValueError('CR fit is active')
        if original['status'] in ('fits_complete_pending_plots','generated_pending_visual_QA'):
            for name in original['products']:
                if not internal_path(ROOT/name).is_file():
                    raise FileNotFoundError('Completed CR product missing: '+name)
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
        starting={};seed_file=None;omitted_seeds={}
        if covariance:
            import ROOT as root
            previous=read_json(output/'fit_recovery/state.json')
            if process_active(previous) or previous['status']!='needs_attention':
                raise ValueError('Requires stopped prior numerical CR recovery')
            seeds=[]
            for p, fit in central_fits(root, previous['attempts']):
                pars=fit.floatParsFinal()
                seeds.append((fit.edm(),str(p),{pars.at(i).GetName():pars.at(i).getVal() for i in range(pars.getSize())}))
            if not seeds:raise ValueError('No converged warm-start fit')
            _,seed_file,starting=min(seeds,key=lambda s:s[0])
            source=root.TFile.Open(str(output/'workspace.root'));ws=source.Get('w')
            try:
                for name,value in list(starting.items()):
                    variable=ws.var(name)
                    if not variable or not variable.getMin()<=value<=variable.getMax():
                        omitted_seeds[name]=value;del starting[name]
            finally:source.Close()
        freeze_contract(out,origins,SOURCES,dict(signal_strength=0,maximum_attempts=2 if covariance or robust else 3,
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
                if linked:
                    command += ['--forceRecreateNLL']
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
                    from TROTASR.workflows.cr_linked_workspace import TOP_PT_CRONLY_REVISION
                    record.update(validate_fit(fit, photon_gnn=linked,
                                               both_nb=original.get('model_revision') == TOP_PT_CRONLY_REVISION))
                    record['passed']=True
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
            retain_pulls(pulls, payload)
            original.update(status='fits_complete_pending_plots',stage='fit_complete',pid=None,
                recovery_state=str((out/'state.json').relative_to(ROOT)),fit_status=accepted['fit_status'],
                covariance_quality=accepted['covariance_quality'],edm=accepted['edm'],
                products=dict.fromkeys(str(p.relative_to(ROOT)) for p in (
                    target,fit,parameters,pulls,*_linked_cr_products(output, linked))),
                plotted_nuisances=sum(p['constraint_model']=='standard_normal' for p in payload['parameters'].values()))
            original.pop('error',None);write_json(output/'state.json',original)
            state['status']='fits_complete_pending_plots';return 0
        finally:
            state.update(pid=None,finished=time.time());write_json(out/'state.json',state)


def _run_once(manifest, output, plot_only=False, ylim=6.0):
    if not math.isfinite(ylim) or ylim <= 0:
        raise ValueError('Invalid pull range')
    output = internal_path(output)
    linked = _changed_cr_model(manifest, output)
    published=published_diagnostic(output,manifest,'cronly')
    if published is not None:
        if linked:
            raise ValueError('The changed LLCR model requires its own native fit receipt')
        with (output/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if published.get('status') == 'provisional':
                figures = published.get('figures', [])
                missing = [name for name in figures if not internal_path(ROOT/name).is_file()]
                print(json.dumps(dict(status='provisional', reused=True, fit_recalculated=False,
                    payload=str(output/published['payload']), figures=figures, missing_figures=missing,
                    fit_status=published['fit_status'], covariance_quality=published['covariance_quality'],
                    process_exit_code=published['process_exit_code']), sort_keys=True))
            elif plot_only:
                state=plot(CommandRunner(output),output,read_json(output/'state.json'),ylim)
                write_json(output/'state.json',state)
        return 0
    card, origins = grid_card(manifest, 'cronly')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from TROTASR.workflows.cr_linked_workspace import TOP_PT_CRONLY_REVISION
        state_path = output / 'state.json'
        state = read_json(state_path) if state_path.exists() else dict(
            status='running', started=time.time(), origins=origins, sr_observed_data_used=False,
            channels=cr_channels(card.read_text(), tuple(read_json(card.with_suffix('.manifest.json'))['years'])),
            workers=1, strategy=0, auto_mc_stats=[10, 1, 1], full_workflow_complete=False)
        if linked:
            default_revision = TOP_PT_CRONLY_REVISION if output.parent == RUN3/'top_pt' else CRONLY_MODEL_REVISION
            state.setdefault('model_revision', default_revision)
            state['fitted_card'] = str((output/'linked_card.txt').relative_to(ROOT))
        both_nb = state.get('model_revision') == TOP_PT_CRONLY_REVISION
        if not (plot_only and (output/'fit_recovery/state.json').exists()):
            options = dict(signal_strength=0, workers=1, strategy=0, pull_ylim=ylim)
            sources = SOURCES
            if linked:
                sources = [*SOURCES, 'workflows/cr_linked_workspace.py']
                options.update(model_revision=state['model_revision'],
                    photon_cr_structure_donor=str((RUN3/'cronly/current_workspace.root').relative_to(ROOT)),
                    photon_cr_donor_fit_values_used=False,
                    photon_cr_observation_scope=('lowdm_both_Nb_GNN' if both_nb else 'lowdm_Nb1_GNN'),
                    force_recreate_nll=True)
            freeze_contract(output, origins, sources, options)
        runner = CommandRunner(output)
        try:
            state['pid'] = os.getpid()
            if plot_only:
                state = plot(runner, output, state, ylim)
            elif state.get('status') in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
                pulls = output/'pulls.json'
                if not pulls.is_file() or read_json(pulls).get('sr_observed_data_used') is not False:
                    raise ValueError('Completed CR fit lacks its blinded pull payload')
                if linked and any(not p.is_file() or str(p.relative_to(ROOT)) not in state.get('products', {})
                                  for p in _linked_cr_products(output, linked)):
                    raise ValueError('Completed CR fit lacks its registered linked observations')
                return 0
            else:
                if linked:
                    state.update(status='running', stage='workspace')
                    write_json(state_path, state)
                target = (_linked_cr_workspace(runner, card, internal_path(manifest), output)
                          if linked else workspace(runner, card, output, cr_only=True))
                fit = output / 'fitDiagnostics_cronly.root'
                state.update(status='running', stage='fit')
                write_json(state_path, state)
                command = fit_command(target)
                if linked:
                    command += ['--forceRecreateNLL']
                require_success(runner.run(command, output, output / 'combine.log', [fit]))
                state.update(validate_fit(fit, photon_gnn=linked, both_nb=both_nb))
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
                retain_pulls(pulls, payload)
                state.update(status='fits_complete_pending_plots', stage='fit_complete', finished=time.time(),
                    plotted_nuisances=sum(p['constraint_model'] == 'standard_normal' for p in payload['parameters'].values()),
                    products=dict.fromkeys(str(p.relative_to(ROOT)) for p in (
                        target, fit, parameters, pulls, *_linked_cr_products(output, linked))))
            grid_card(manifest, 'cronly')  # unchanged card and TH1 files after execution
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error), finished=time.time())
            raise
        finally:
            state['pid'] = None
            write_json(state_path, state)
    return 0


def cr_profile_command(target, name, strategy):
    from TROTASR.workflows.combine_impacts import fit_command
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
    from TROTASR.workflows.combine_impacts import tree_rows, tracked_start, reconcile_seed_bounds
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
                products.update(dict.fromkeys(str(p.relative_to(ROOT)) for p in (candidate,fitpath)))
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
                if rel not in reusable:continue
                rows=tree_rows(product,name)
                if (len(rows)==2 and all(math.isfinite(v) for row in rows for v in row.values())
                        and rows[-1]['r']==0. and rows[-1]['deltaNLL']>=-.005):
                    seed_samples.append(dict(rows[-1],root=rel))
    def fixed(value):
        key='point_'+format(value,'.10f').replace('-','m').replace('.','p')
        if reusable and not multistart:
            for product in sorted(p for previous in old_profiles
                                  for p in (previous/name/key).glob('higgsCombine*.root')):
                rel=str(product.relative_to(ROOT))
                if rel not in reusable:continue
                rows=tree_rows(product,name)
                if (len(rows)==2 and all(math.isfinite(v) for row in rows for v in row.values())
                        and rows[-1]['r']==0. and abs(rows[-1]['theta']-value)<2e-6
                        and rows[-1]['deltaNLL']>=-.005):
                    result=dict(rows[-1],root=rel)
                    samples.append(result);products[rel]=None;return result
        folder=point/key;folder.mkdir(exist_ok=True)
        starts_to_try=profile_start_points(seed_samples+samples,value) if multistart else [None]
        candidates=[]
        if full_start:
            for seed in seed_samples:
                if abs(seed['theta']-value)<2e-7:
                    candidates.append(seed);products[seed['root']]=None
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
                result=dict(rows[-1],root=str(product.relative_to(ROOT)))
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


PROFILE_PHASES = ('profile_pulls', 'profile_pulls_minimizer', *PROFILE_WARM_STAGES)


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


def recover_profile_pulls(manifest, output, workers, *, continuation=False, warm_step=None):
    """Measure plotted CR nuisance intervals directly, without a Hessian veto."""
    from TROTASR.workflows.combine_impacts import validate_fit as validate_profile
    import ROOT as root
    root.gROOT.SetBatch(True);root.EnableThreadSafety()
    if not 1<=workers<=90:raise ValueError('Invalid shared worker limit')
    output=internal_path(output);card,origins=grid_card(manifest,'cronly')
    linked = _changed_cr_model(manifest, output)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=read_json(output/'state.json')
        if process_active(parent):raise ValueError('CR controller is active')
        previous=read_json(output/'fit_recovery/state.json')
        # Failed covariance validation records still retain the actual fit ROOT.
        candidates = [(fit.edm(), path) for path, fit in central_fits(root, previous['attempts'])
                      if fit.covQual() >= 2]
        if not candidates:raise ValueError('No converged CR central fit')
        _,central=min(candidates,key=lambda x:x[0]);target=output/'workspace.root'
        warm = warm_step is not None
        step = warm_step if warm else -1
        delayed_warm, gaussian_scan, direct_scan, neighbour_scan = (step >= k for k in (1, 2, 3, 4))
        bracket_recovery, multistart, full_start = (step >= k for k in (6, 7, 8))
        if warm:
            folder = PROFILE_WARM_STAGES[step]
            previous_profiles = output/(PROFILE_WARM_STAGES[step-1] if step else 'profile_pulls_minimizer')
        elif continuation:
            folder, previous_profiles = 'profile_pulls_minimizer', output/'profile_pulls'
        else:
            folder, previous_profiles = 'profile_pulls', None
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
        out = output/folder
        out.mkdir(exist_ok=True)
        files=[*SOURCES,'workflows/combine_impacts.py']
        contract=freeze_contract(out,origins,files,dict(
            central_fit=str(central.relative_to(ROOT)),signal_strength=0,observed_sr_used=False,
            maximum_configurations=2 if continuation else 3,workers=workers,uncertainty_method='profile_likelihood',
            warm_start=starting,omitted_out_of_workspace_range_seeds=omitted,
            set_parameters_after_nll=delayed_warm,
            gaussian_computational_range={'CMS_NPS26012_eff_g_id_2025':[-10,10]} if gaussian_scan else {},
            direct_profile=direct_scan,neighbour_seed=neighbour_scan,maximum_bracket_steps=9 if neighbour_scan else 5,maximum_bisection_steps=18,
            crossing_fractions=[.5,.25,.75] if bracket_recovery else [.5],
            independent_conditional_starts=3 if multistart else 1,
            explicit_statistical_nuisances=multistart,
            free_normalizations_in_starts=full_start,
            physics_changed=False))
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
                    if prior_record.get('method') == 'direct_CR_profile_likelihood':
                        checked = prior_record['validation']
                        center, low, high = checked['entries']
                        if (checked.get('valid') is not True or not all(
                                row[0] == 0 and all(map(math.isfinite, row)) for row in (center, low, high))
                                or not low[1] < center[1] < high[1]
                                or any(abs(row[2]-.5) > .002 for row in (low, high))
                                or not prior_record.get('profile_products')):
                            raise ValueError('Invalid completed direct CR profile')
                        for path in prior_record['profile_products']:
                            if not internal_path(ROOT/path).is_file():
                                raise ValueError('Missing completed direct CR profile product')
                        return prior_record
                    checked=validate_profile(ROOT/prior_record['folder'],name,prior_record['tag'],bounds[name])
                    if not checked['valid']:
                        raise ValueError('Completed CR profile changed')
                    return prior_record
            if direct_scan:return direct_cr_profile(output,out,name,runner,neighbour_scan,bracket_recovery,multistart,full_start)
            point=out/name;point.mkdir(exist_ok=True)
            if (point/'state.json').exists():
                saved=read_json(point/'state.json')
                if saved['status']=='complete':
                    checked=validate_profile(ROOT/saved['folder'],name,saved['tag'],bounds[name])
                    if checked['valid']:return saved
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
            for name,record in recovered.items():
                rows=record['validation']['entries'];center,lower,upper=[r[1] for r in rows]
                item=payload['parameters'][name]
                item.update(value=center,pull=center,error=(upper-lower)/2.,constraint=(upper-lower)/2.,
                    error_low=center-lower,error_high=upper-center,uncertainty_method='profile_likelihood')
            payload.pop('covariance',None)
            payload.update(sr_observed_data_used=False,source_card=str(card),channels=parent['channels'],
                plotted_uncertainties='asymmetric_profile_likelihood',global_covariance_accepted=False)
            pulls=output/'pulls.json'
            retain_pulls(pulls, payload)
            products=dict.fromkeys(str(p.relative_to(ROOT)) for p in (
                target,central,original,pulls,*_linked_cr_products(output, linked)))
            for record in recovered.values():
                if 'profile_products' in record:products.update(record['profile_products'])
                else:products[str((ROOT/record['folder']/('higgsCombine'+record['tag']+'.MultiDimFit.mH120.root')).relative_to(ROOT))]=None
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
        except (CommandFailure, FileNotFoundError, ValueError) as error:
            failures.append(dict(method=name, outcome='inapplicable', reason=str(error)))
        else:
            failures.append(dict(method=name, outcome='complete' if result == 0 else 'exhausted'))
            if result == 0:
                return 0
    if (output/'fit_recovery/state.json').is_file():
        stages = [('profile_pulls', {}), ('profile_pulls_minimizer', {'continuation': True})]
        stages += [(stage, {'warm_step': step}) for step, stage in enumerate(PROFILE_WARM_STAGES)]
        for stage, options in stages:
            try:
                result = recover_profile_pulls(manifest, output, workers=1, **options)
            except BlockingIOError:
                raise
            except (CommandFailure, FileNotFoundError, ValueError) as error:
                failures.append(dict(method=stage, outcome='inapplicable', reason=str(error)))
                if not (output/stage/'state.json').is_file():
                    break
            else:
                failures.append(dict(method=stage, outcome='complete' if result == 0 else 'exhausted'))
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
    p.add_argument('--refit-current', action='store_true')
    p.add_argument('--refit-initial-values', help='Explicit initial parameter values for the requested refit')
    p.add_argument('--pull-ylim', type=float, default=6.0)
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--workspace-step', choices=('prepare', 'link'), help=argparse.SUPPRESS)
    p.add_argument('--workspace-input', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    require_current_llcr(a.manifest, a.output, a.workspace_input)
    if a.refit_initial_values and not a.refit_current:
        p.error('--refit-initial-values requires --refit-current')
    if a.refit_current and a.plot_only:
        p.error('Choose a refit or plotting')
    if bool(a.workspace_step) != bool(a.workspace_input) or (
            a.workspace_step and (a.refit_current or a.plot_only)):
        p.error('Invalid native CR-only workspace step')
    native_runtime(a.output)
    if a.workspace_step:
        from TROTASR.workflows.cr_linked_workspace import cronly_workspace_step
        if not _changed_cr_model(a.manifest, a.output):
            raise ValueError('The chosen workspace step requires the changed LLCR model')
        output, source = internal_path(a.output), internal_path(a.workspace_input)
        expected = (grid_card(a.manifest, 'cronly')[0] if a.workspace_step == 'prepare'
                    else output/'native_cr_workspace.root')
        if source != expected:
            raise ValueError('Unexpected source for the native CR-only workspace step')
        cronly_workspace_step(a.workspace_step, source, output, RUN3)
        raise SystemExit(0)
    if a.refit_current:
        raise SystemExit(refit_current(a.manifest, a.output, initial_values=a.refit_initial_values))
    raise SystemExit(run(a.manifest, a.output, a.plot_only, a.pull_ylim))
