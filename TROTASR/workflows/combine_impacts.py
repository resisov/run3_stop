"""Current-grid benchmark, mu=1 S+B Asimov impacts with finite legacy fits.

The old robustFit strategy0 then strategy1 prescription and validation kernels
are preserved. No observed SR is fitted. Failed/boundary endpoints remain
explicit. Plotting is a separate --plot-only invocation of plotImpacts.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import os
from pathlib import Path
import re
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import (native_runtime, grid_card,
    freeze_contract, CommandRunner, require_success, workspace, published_diagnostic, process_active)
from TROTASR.workflows.impact_legacy import check_endpoints, fit_command, validate_fit

SOURCES = ['workflows/combine_impacts.py', 'workflows/diagnostic_common.py',
           'workflows/impact_legacy.py', 'workflows/combine_limits.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py']


def benchmark_labels(model, mass):
    if model not in ('T2tt', 'T2bW', 'T2tb') or not re.fullmatch(r'mStop\d+_mLSP\d+', mass):
        raise ValueError('Invalid impact benchmark')
    identity = model + '_' + mass
    # Preserve historical default labels and filenames, without ambiguous
    # cross-topology names for newly requested benchmarks.
    label = mass if identity == 'T2tt_mStop1200_mLSP500' else identity
    return identity, label, 'impacts_' + label


def saved_workspace(state_path, card, model, mass):
    """Reuse this exact point's validated workspace; do not rebuild templates."""
    state_path = internal_path(state_path)
    state = read_json(state_path)
    target = internal_path(state_path.parent / 'workspace.root')
    if ((state.get('model'), state.get('mass')) != (model, mass)
            or state.get('workspace_valid') is not True
            or internal_path(ROOT / state['card']) != card
            or state.get('card_sha256') != sha256(card)
            or not re.fullmatch('[0-9a-f]{64}',state.get('workspace_sha256',''))
            or not target.is_file() or target.stat().st_size <= 0):
        raise ValueError('Saved workspace does not match the verified benchmark/card')
    return target, dict(state=str(state_path.relative_to(ROOT)), state_sha256=sha256(state_path),
                       workspace=str(target.relative_to(ROOT)), workspace_sha256=state['workspace_sha256'],
                       input_check_scope='accepted_limit_workspace_receipt; no_workspace_rehash')


def publish_output(source, destination):
    """Read-only-use link within this new fit run; never change input TH1s."""
    source, destination = internal_path(source), internal_path(destination)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError('Divergent fit output retained: ' + str(destination))
    else:
        os.link(source, destination)


def fit_one(output, workspace_path, name, bounds, runner):
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError('Unsafe nuisance name')
    path = output / 'fit_state' / (name + '.json')
    state = read_json(path) if path.exists() else dict(status='running', attempts=[])
    if state.get('bounds') is not None and state['bounds'] != list(bounds):
        raise ValueError('Parameter bounds changed')
    state['bounds'] = list(bounds)
    for strategy in (0, 1):
        here = output / 'attempts' / name / ('strategy' + str(strategy))
        here.mkdir(parents=True, exist_ok=True)
        command, tag = fit_command(workspace_path, name, 1, strategy)
        result = here / ('higgsCombine' + tag + '.MultiDimFit.mH120.root')
        saved = here / ('multidimfit' + tag + '.root')
        products = [result, saved] if name == 'r' else [result]
        old = next((a for a in state['attempts'] if a['strategy'] == strategy), None)
        if old is None:
            record = runner.run(command, here, here / 'combine.log', products)
            validation = validate_fit(here, name, tag, bounds)
            if record['status'] != 'complete':
                validation.update(valid=False, error='Combine command/output failure: ' + record['log'])
            old = dict(strategy=strategy, exit_code=record['exit_code'], validation=validation,
                       folder=str(here.relative_to(ROOT)), products=record.get('products', {}))
            state['attempts'].append(old)
            write_json(path, state)
        if old['validation']['valid']:
            for product, digest in old['products'].items():
                if sha256(ROOT / product) != digest:
                    raise ValueError('Previously valid saved fit changed: ' + product)
            validation = validate_fit(here, name, tag, bounds)
            if not validation['valid'] or validation['sha256'] != old['validation']['sha256']:
                raise ValueError('Previously valid impact endpoint changed: ' + name)
            for product in products:
                publish_output(product, output / product.name)
            rows = validation['entries']
            validation['boundary_endpoint_lower'] = math.isclose(rows[1][1], bounds[0], rel_tol=1e-7, abs_tol=1e-8)
            validation['boundary_endpoint_upper'] = math.isclose(rows[2][1], bounds[1], rel_tol=1e-7, abs_tol=1e-8)
            state.update(status='complete', validation=validation)
            write_json(path, state)
            return validation
    # Expose only the final failed attempt for the existing bounded recovery
    # reader. Its FAILED state prevents treating this file as a valid impact.
    if result.exists():
        publish_output(result, output / result.name)
    state.update(status='failed', validation=old['validation'])
    write_json(path, state)
    return state['validation']


def enumerate_parameters(workspace_path):
    import ROOT as root
    from HiggsAnalysis.CombinedLimit.tool_base.Impacts import Impacts
    names = Impacts().all_free_parameters(str(workspace_path), 'w', 'ModelConfig', ['r'])
    if len(set(names)) != len(names) or not names or any(not re.fullmatch(r'[A-Za-z0-9_]+', n) for n in names):
        raise ValueError('Invalid impact parameter enumeration')
    source = root.TFile.Open(str(workspace_path))
    try:
        ws = source.Get('w')
        bounds = {n: [float(ws.var(n).getMin()), float(ws.var(n).getMax())] for n in names}
    finally:
        source.Close()
    if any(not all(math.isfinite(x) for x in b) or b[0] >= b[1] for b in bounds.values()):
        raise ValueError('Invalid parameter bounds')
    bounds['r'] = [0., 20.]
    return names, bounds


def validate_collection(collected, names, results, initial):
    if len(collected['params']) != len(names) or {p['name'] for p in collected['params']} != set(names):
        raise ValueError('Impact collection lost/duplicated nuisances')
    expected_poi = [initial['entries'][i][0] for i in (1, 0, 2)]
    if len(collected['POIs']) != 1 or collected['POIs'][0]['name'] != 'r':
        raise ValueError('Unexpected signal POI')
    def equal(a, b):
        return len(a) == len(b) and all(math.isfinite(float(x)) and math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-7)
                                       for x, y in zip(a, b))
    if not equal(collected['POIs'][0]['fit'], expected_poi):
        raise ValueError('Collected initial POI differs from validated fit')
    for p in collected['params']:
        rows = results[p['name']]['entries']
        if not equal(p['fit'], [rows[i][1] for i in (1, 0, 2)]) or not equal(p['r'], [rows[i][0] for i in (1, 0, 2)]):
            raise ValueError('Collected endpoints differ from validated fits: ' + p['name'])
        impact = max(abs(p['r'][i] - p['r'][1]) for i in (0, 2))
        if not math.isfinite(p['impact_r']) or not math.isclose(p['impact_r'], impact, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError('Collected impact is inconsistent')


def plot(runner, output, state, stem_name='impacts_mStop1200_mLSP500'):
    if state.get('status')=='needs_attention' and state.get('recovery_status'):
        recovery=read_json(ROOT/state['recovery_status'])
        old_log=output/'plot.log'
        if (recovery['status']=='fits_complete' and recovery['json_sha256']==state.get('json_sha256')
                and old_log.exists() and "No module named 'HiggsAnalysis'" in old_log.read_text()):
            state['status']='fits_complete'
            state['previous_plot_error']=state.pop('error',None)
    if state.get('status') not in ('fits_complete', 'generated_pending_visual_QA'):
        raise ValueError('Do not plot incomplete impact fits')
    payload = internal_path(ROOT / state['json'])
    if sha256(payload) != state['json_sha256']:
        raise ValueError('Impact JSON changed')
    stem = output / stem_name
    import shutil
    script=shutil.which('plotImpacts.py')
    if not script:raise ValueError('Installed Combine impact renderer is unavailable')
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(env['CMSSW_BASE'])/'python')+os.pathsep+env.get('PYTHONPATH','')
    require_success(runner.run([sys.executable, script, '-i', payload, '-o', stem], output,
                               output / 'plot_runtime.log', [stem.with_suffix('.pdf')],env=env))
    # plotImpacts produces a multi-page PDF. Poppler rendering and all-page QA
    # are deliberately a later explicit step, not fabricated by this runner.
    state.update(status='generated_pending_visual_QA', plot_source_sha256=sha256(Path(__file__)),figures={str(stem.with_suffix('.pdf').relative_to(ROOT)):
                                                               sha256(stem.with_suffix('.pdf'))})
    return state


def run(manifest, output, workers=40, plot_only=False, model='T2tt', mass='mStop1200_mLSP500', workspace_state=None):
    output=internal_path(output)
    published=published_diagnostic(output,manifest,'impact',model,mass)
    if published is not None:
        with (output/'impact.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if plot_only:
                state=read_json(output/'impact_status.json')
                state=plot(CommandRunner(output),output,state)
                write_json(output/'impact_status.json',state)
        return 0
    if workspace_state is None:
        candidate=internal_path(manifest).parents[1]/'limits'/model/mass/'state.json'
        if candidate.is_file():
            workspace_state=candidate
    if workspace_state is not None:
        with (internal_path(workspace_state).parent/'point.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            return run_with_recovery(manifest,output,workers,plot_only,model,mass,workspace_state)
    return run_with_recovery(manifest,output,workers,plot_only,model,mass)


def _run(manifest, output, workers=40, plot_only=False, model='T2tt', mass='mStop1200_mLSP500', workspace_state=None):
    if not 1 <= workers <= 99:
        raise ValueError('Reserve at least one controller slot')
    identity, label, stem = benchmark_labels(model, mass)
    card, origins = grid_card(manifest, 'impact', model, mass)
    reused = None
    options = dict(benchmark=identity, expect_signal=1, r_range=[0, 20], strategies=[0, 1], robust_fit=1)
    if workspace_state is not None:
        reused, options['reused_workspace'] = saved_workspace(workspace_state, card, model, mass)
    output = internal_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'impact.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plot_only:
            contract=read_json(output/'contract.json')
            if contract['origins']!=origins or contract['options']!=options:
                raise ValueError('Original impact input contract changed')
        else:
            freeze_contract(output, origins, SOURCES, options)
        runner = CommandRunner(output)
        path = output / 'impact_status.json'
        state = read_json(path) if path.exists() else dict(status='running', started=time.time(),
            benchmark=label, asimov_expect_signal=1, sr_data_used=False,
            approximate_impacts=False, full_workflow_complete=False, origins=origins)
        try:
            state.update(controller_pid=os.getpid(), workers=workers)
            if plot_only:
                state = plot(runner, output, state, stem)
            elif state.get('status') in ('fits_complete', 'generated_pending_visual_QA'):
                if sha256(ROOT / state['json']) != state['json_sha256']:
                    raise ValueError('Completed impact JSON changed')
                for product, digest in state.get('fit_products', {}).items():
                    if sha256(ROOT / product) != digest:
                        raise ValueError('Completed impact fit changed: ' + product)
                return 0
            else:
                import ROOT as root
                root.gROOT.SetBatch(True)
                root.EnableThreadSafety()
                target = reused if reused is not None else workspace(runner, card, output)
                names, bounds = enumerate_parameters(target)
                if state.get('names') is not None and state['names'] != names:
                    raise ValueError('Workspace parameter enumeration changed')
                state.update(workspace=str(target), workspace_sha256=(
                    options['reused_workspace']['workspace_sha256'] if reused is not None else sha256(target)), names=names,
                             nuisance_count=len(names), status='running', phase='initial_fit')
                write_json(path, state)
                initial = fit_one(output, target, 'r', bounds['r'], runner)
                if not initial['valid']:
                    raise RuntimeError('Finite initial Asimov fit attempts failed')
                state.update(initial_fit=initial, phase='parameter_fits')
                write_json(path, state)
                results = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    tasks = {pool.submit(fit_one, output, target, n, bounds[n], runner): n for n in names}
                    for task in as_completed(tasks):
                        name = tasks[task]
                        try:
                            results[name] = task.result()
                        except Exception as error:
                            results[name] = dict(name=name, valid=False, error=repr(error))
                        state.update(valid_nuisances=sum(v['valid'] for v in results.values()),
                            failed_nuisances=[n for n, v in results.items() if not v['valid']],
                            evaluated_nuisances=len(results), updated=time.time())
                        write_json(path, state)
                failed = [n for n in names if not results[n]['valid']]
                state.update(failed_nuisances=failed, finished=time.time(),
                    parameters_with_boundary_endpoints=[n for n, v in results.items() if v.get('boundary_endpoint_lower')
                                                        or v.get('boundary_endpoint_upper')])
                write_json(output / 'fit_endpoint_validation.json', dict(valid=not failed, results=results))
                if failed:
                    state.update(status='needs_attention', phase='finite_fits_exhausted',
                                 recovery='pending_existing_bounded_direct_profile_for_supported_failures')
                    return 2
                payload = output / (stem + '.json')
                require_success(runner.run(['combineTool.py', '-M', 'Impacts', '-d', target, '-m', '120',
                    '-o', payload, '--parallel', '1'], output, output / 'collect.log', [payload]))
                validate_collection(read_json(payload), names, results, initial)
                state.update(status='fits_complete', phase='collected', json=str(payload.relative_to(ROOT)),
                    json_sha256=sha256(payload), plots_status='pending_original_plotImpacts',
                    fit_products={str(p.relative_to(ROOT)): sha256(p) for p in output.glob('*.root')},
                    convergence_scope='Initial fit status/covariance and finite bracketed impact endpoints; no fabricated endpoint fit status')
            grid_card(manifest, 'impact', model, mass)
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error), finished=time.time())
            raise
        finally:
            state['controller_pid'] = None
            write_json(path, state)
    return 0


def run_with_recovery(manifest, output, workers, plot_only=False, model='T2tt',
                      mass='mStop1200_mLSP500', workspace_state=None):
    """The impacts stage owns primary fits and their finite bounded profiles."""
    from TROTASR.workflows.recover_impacts import run_bounded, run as direct_profile
    result = _run(manifest, output, workers, plot_only, model, mass, workspace_state)
    if result != 2 or plot_only:
        return result
    path = internal_path(output) / 'impact_status.json'
    attempts = []
    phases = (('bounded_sgamma_complete', {}),
              ('bounded_sgamma_precision', {'polish': True}),
              ('bounded_sgamma_minimizer', {'hybrid': True}),
              ('bounded_sgamma_likelihood', {'likelihood': True}),
              ('bounded_sgamma_refined', {'refine': True}),
              ('bounded_sgamma_final', {'refine': True, 'final': True}))
    for phase, options in phases:
        receipt = internal_path(output) / phase / 'status.json'
        saved = read_json(receipt) if receipt.is_file() else {}
        if process_active(saved, 'controller_pid'):
            raise BlockingIOError('Impact profile controller is still active')
        if saved.get('status') == 'needs_attention' and saved.get('failed'):
            attempts.append(dict(method=phase, outcome='exhausted'))
            continue
        try:
            result = run_bounded(manifest, internal_path(output), workers, model=model,
                                 mass=mass, complete=True, **options)
        except BlockingIOError:
            raise
        except ValueError as error:
            attempts.append(dict(method=phase, outcome='inapplicable', reason=str(error)))
        else:
            attempts.append(dict(method=phase, outcome='complete' if result == 0 else 'exhausted'))
            if result == 0:
                return 0
    # The older direct recipe is applicable only to its physical [0,10]
    # rate bounds or a numerically established missing endpoint. Its own
    # preflight checks these conditions before issuing any Combine command.
    state = read_json(path)
    failed = state.get('failed_nuisances', [])
    records = [read_json(internal_path(output)/'fit_state'/(name+'.json')) for name in failed]
    bounded = records and all(record.get('bounds') == [.01,5.] for record in records)
    if bounded or not failed or (model,mass) != ('T2tt','mStop1200_mLSP500'):
        attempts.append(dict(method='direct_profile', outcome='inapplicable',
                             reason='Requires original benchmark and compatible direct-profile endpoints/bounds'))
    else:
        missing = [name for name in failed if 'sgamma_shape' not in name and 'qcd_norm' not in name]
        try:
            result = direct_profile(manifest, internal_path(output), workers, missing=missing)
        except BlockingIOError:
            raise
        except ValueError as error:
            attempts.append(dict(method='direct_profile', outcome='inapplicable', reason=str(error)))
        else:
            attempts.append(dict(method='direct_profile', outcome='complete' if result == 0 else 'exhausted'))
            if result == 0:
                return 0
    state = read_json(path)
    state['recovery_attempts'] = attempts
    write_json(path, state)
    return 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'stats/run3/grid/manifest.json')
    p.add_argument('--output', type=Path, default=ROOT/'stats/run3/impacts')
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--model', choices=('T2tt', 'T2bW', 'T2tb'), default='T2tt')
    p.add_argument('--mass', default='mStop1200_mLSP500')
    p.add_argument('--workspace-state', type=Path, help='Reuse the exact verified limit-point workspace')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    native_runtime(a.output)
    raise SystemExit(run(a.manifest, a.output, a.workers, a.plot_only, a.model, a.mass, a.workspace_state))
