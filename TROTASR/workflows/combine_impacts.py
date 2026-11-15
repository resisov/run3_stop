"""Current-grid benchmark, mu=1 S+B Asimov impacts with finite recovery.

The old robustFit strategy0 then strategy1 prescription and validation kernels
are preserved. No observed SR is fitted. Failed/boundary endpoints remain
explicit. Plotting is a separate --plot-only invocation of plotImpacts.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import (native_runtime, grid_card,
    freeze_contract, CommandRunner, require_success, workspace, published_diagnostic, process_active)
SOURCES = ['workflows/combine_impacts.py', 'workflows/diagnostic_common.py',
           'workflows/combine_limits.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py']
MASS = '120'
FIT_ROOT_LOCK = threading.Lock()
PROFILE_ROOT_LOCK = threading.Lock()
PROFILE_TARGET = .5
PROFILE_NLL_TOL = .002


def benchmark_labels(model, mass):
    if model not in ('T2tt', 'T2bW', 'T2tb') or not re.fullmatch(r'mStop\d+_mLSP\d+', mass):
        raise ValueError('Invalid impact benchmark')
    identity = model + '_' + mass
    # Preserve historical default labels and filenames, without ambiguous
    # cross-topology names for newly requested benchmarks.
    label = mass if identity == 'T2tt_mStop1200_mLSP500' else identity
    return identity, label, 'impacts_' + label


def saved_workspace(state_path, card, model, mass, revision=None):
    """Reuse this exact point's validated workspace; do not rebuild templates."""
    state_path = internal_path(state_path)
    state = read_json(state_path)
    target = internal_path(state_path.parent / 'workspace.root')
    if ((state.get('model'), state.get('mass')) != (model, mass)
            or state.get('model_revision') != revision
            or state.get('workspace_valid') is not True
            or internal_path(ROOT / state['card']) != card
            or state.get('card_sha256') != sha256(card)
            or not re.fullmatch('[0-9a-f]{64}',state.get('workspace_sha256',''))
            or not target.is_file() or target.stat().st_size <= 0):
        raise ValueError('Saved workspace does not match the verified benchmark/card')
    binding = dict(state=str(state_path.relative_to(ROOT)),
                   workspace=str(target.relative_to(ROOT)), workspace_sha256=state['workspace_sha256'],
                   input_check_scope='accepted_limit_workspace_receipt; no_workspace_rehash')
    if revision:
        # The limit point can update its fit receipts while both stages read
        # the already accepted workspace. Bind to that artifact, not the
        # changing limit attempts.
        binding['model_revision'] = revision
    else:
        binding['state_sha256'] = sha256(state_path)
    return target, binding


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
    run_root=ROOT/'stats/run3'
    if ROOT/'stats/references' in output.parents:
        raise ValueError('Reference fit products are not an execution path')
    marker=run_root/'limits/model.json'
    revision=(read_json(marker).get('model_revision') if marker.is_file()
              and internal_path(manifest)==run_root/'grid/manifest.json' else None)
    if revision:
        from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION
        canonical_state=run_root/'limits'/model/mass/'state.json'
        if (revision != MODEL_REVISION or output != run_root/'impacts' or workers > 40
                or (workspace_state is not None and internal_path(workspace_state) != canonical_state)):
            raise ValueError('Use the updated canonical Run 3 benchmark workspace')
        if (not canonical_state.is_file() or read_json(canonical_state).get('model_revision') != revision
                or read_json(canonical_state).get('workspace_valid') is not True):
            raise ValueError('The updated benchmark workspace is not ready for impacts')
        workspace_state=canonical_state
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
    if revision:
        # A workspace_valid point is immutable. Its point lock protects the
        # limit attempts; the impact lock protects our own outputs and model
        # adoption. There is no need to wait for the limit's five crossings.
        return run_with_recovery(manifest,output,workers,plot_only,model,mass,workspace_state,revision)
    if workspace_state is not None:
        with (internal_path(workspace_state).parent/'point.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            return run_with_recovery(manifest,output,workers,plot_only,model,mass,workspace_state)
    return run_with_recovery(manifest,output,workers,plot_only,model,mass)


def _run(manifest, output, workers=40, plot_only=False, model='T2tt', mass='mStop1200_mLSP500',
         workspace_state=None, revision=None):
    if not 1 <= workers <= 99:
        raise ValueError('Reserve at least one controller slot')
    identity, label, stem = benchmark_labels(model, mass)
    card, origins = grid_card(manifest, 'impact', model, mass)
    reused = None
    options = dict(benchmark=identity, expect_signal=1, r_range=[0, 20], strategies=[0, 1], robust_fit=1)
    if revision:
        options.update(model_revision=revision, signal_uncertainties=False)
    if workspace_state is not None:
        reused, options['reused_workspace'] = saved_workspace(workspace_state, card, model, mass, revision)
    output = internal_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'impact.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plot_only:
            contract=read_json(output/'contract.json')
            if contract['origins']!=origins or contract['options']!=options:
                raise ValueError('Original impact input contract changed')
        else:
            sources = SOURCES + (['workflows/cr_linked_workspace.py'] if revision else [])
            freeze_contract(output, origins, sources, options)
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


def check_endpoints(values, bounds):
    if len(values) != 3 or not all(math.isfinite(x) for row in values for x in row):
        raise ValueError('expected three finite fit entries')
    nominal, lower, upper = [row[1] for row in values]
    tol = 1e-6 * max(1., abs(nominal), abs(lower), abs(upper))
    if nominal == lower == upper:
        raise ValueError('identical parameter endpoints')
    if not lower <= nominal + tol or not nominal <= upper + tol:
        raise ValueError('endpoints do not bracket nominal')
    if lower == nominal and not math.isclose(nominal, bounds[0], rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('zero-width lower endpoint away from boundary')
    if upper == nominal and not math.isclose(nominal, bounds[1], rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('zero-width upper endpoint away from boundary')


def fit_command(workspace, name, expect_signal, strategy):
    initial = name == 'r'
    tag = '_initialFit_Test' if initial else '_paramFit_Test_' + name
    command = ['combine', '-M', 'MultiDimFit', '-n', tag, '--algo',
               'singles' if initial else 'impact', '--redefineSignalPOIs', 'r']
    if not initial:
        command += ['-P', name, '--floatOtherPOIs', '1', '--saveInactivePOI', '1']
    command += ['-m', MASS, '-d', str(workspace), '--robustFit', '1',
                '--cminDefaultMinimizerStrategy', str(strategy), '-t', '-1',
                '--expectSignal', str(expect_signal), '--setParameterRanges',
                'r=%s,20' % ('-20' if expect_signal == 0 else '0'), '--saveFitResult']
    return command, tag


def validate_fit(folder, name, tag, bounds):
    import ROOT
    result = dict(name=name, valid=False)
    source = fit_source = None
    with FIT_ROOT_LOCK:
        try:
            path = folder / ('higgsCombine%s.MultiDimFit.mH%s.root' % (tag, MASS))
            source = ROOT.TFile.Open(str(path))
            if not source or source.IsZombie():
                raise ValueError('unreadable fit ROOT')
            tree = source.Get('limit')
            if not tree:
                raise ValueError('missing limit tree')
            values = [[float(row.r), float(getattr(row, name)), float(row.deltaNLL)] for row in tree]
            check_endpoints(values, bounds)
            if name == 'r':
                fit_source = ROOT.TFile.Open(str(folder / ('multidimfit%s.root' % tag)))
                fit = fit_source.Get('fit_mdf') if fit_source and not fit_source.IsZombie() else None
                if not fit:
                    raise ValueError('missing saved RooFitResult')
                result.update(fit_status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()))
                if result['fit_status'] != 0 or result['covariance_quality'] < 2 or not math.isfinite(result['edm']):
                    raise ValueError('nominal fit convergence/covariance validation failed')
            else:
                # Combine v10.5.1 does not call saveResult for algo=impact.
                # doImpact commits each endpoint only if minim.minimize succeeds.
                result['validation_scope'] = 'three finite bracketed entries; both endpoint minimizations succeeded'
            result.update(valid=True, entries=values, bounds=list(bounds), sha256=sha256(path))
        except Exception as error:
            result['error'] = str(error)
        finally:
            if source:
                source.Close()
            if fit_source:
                fit_source.Close()
    return result


# Finite impact profiles; the common Asimov sample and likelihood are unchanged.
def artifact_stat(path):
    """Pin an accepted file during collection without reading its ROOT payload."""
    item = internal_path(path).stat()
    return dict(device=item.st_dev,inode=item.st_ino,size=item.st_size,
                mtime_ns=item.st_mtime_ns,ctime_ns=item.st_ctime_ns)


def tree_rows(path, name):
    import ROOT
    path = internal_path(path)
    with PROFILE_ROOT_LOCK:
        src = ROOT.TFile.Open(str(path))
        try:
            if not src or src.IsZombie() or not src.Get('limit'):
                raise ValueError('missing limit tree: ' + str(path))
            return [dict(theta=float(getattr(row, name)), r=float(row.r),
                         deltaNLL=float(row.deltaNLL), quantile=float(row.quantileExpected))
                    for row in src.Get('limit')]
        finally:
            if src:
                src.Close()


def missing_endpoint_side(rows, center, bounds):
    """Infer the absent side from a valid central row plus one saved endpoint."""
    if len(rows) != 2 or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError('missing-endpoint recovery needs exactly two finite entries')
    if not math.isclose(rows[0]['theta'], center, rel_tol=0, abs_tol=1e-4):
        raise ValueError('saved central value differs from the common best fit')
    endpoint = rows[1]['theta']
    if abs(endpoint-center) < 1e-5 or not bounds[0] < center < bounds[1]:
        raise ValueError('endpoint or common best fit is degenerate')
    if not bounds[0] <= endpoint <= bounds[1]:
        raise ValueError('saved endpoint is outside original parameter bounds')
    return -1 if endpoint > center else 1


def missing_endpoint_rows(rows, center, bounds):
    """Recognize a failed zero-width endpoint without accepting it as a fit.

    Combine may retain the failed side as a third row equal to the central
    theta. Keep the original ROOT untouched and pass only the central row and
    the other saved endpoint to the existing bounded missing-side algorithm.
    This adapter is used only for explicitly named failed parameters.
    """
    if len(rows) == 2:
        missing_endpoint_side(rows, center, bounds)
        return rows
    if len(rows) != 3 or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError('Expected two finite rows or one zero-width failed side')
    if not bounds[0] < center < bounds[1]:
        raise ValueError('A physical-boundary endpoint is not a missing side')
    saved_center = rows[0]['theta']
    zero = [abs(row['theta']-saved_center) < 1e-5 for row in rows[1:]]
    if sum(zero) != 1:
        raise ValueError('Exactly one saved endpoint must be zero-width')
    if zero[0]:
        endpoint = rows[2]
        if endpoint['theta'] <= saved_center:
            raise ValueError('Remaining endpoint is not on the upper side')
    else:
        endpoint = rows[1]
        if endpoint['theta'] >= saved_center:
            raise ValueError('Remaining endpoint is not on the lower side')
    selected = [rows[0], endpoint]
    missing_endpoint_side(selected, center, bounds)
    return selected


def prepare_profile_inputs(work, out, source, runner):
    import ROOT
    setup = out / 'inputs.json'
    snapshot = out / 'workspace_bestfit_snapshot.root'
    toys = out / 'higgsCombine_profileAsimov.GenerateOnly.mH120.123456.root'
    provenance = dict(workspace=str(source), workspace_sha256=sha256(source),
                      initial_fit_sha256=sha256(work / 'multidimfit_initialFit_Test.root'))
    if setup.exists():
        old = read_json(setup)
        if any(old[k] != v for k, v in provenance.items()):
            raise ValueError('recovery input provenance changed')
        if sha256(snapshot) != old['snapshot_sha256'] or sha256(toys) != old['asimov_sha256']:
            raise ValueError('recovery cache changed')
        return old
    if snapshot.exists():
        raise ValueError('Unvalidated recovery snapshot retained; explicit recovery required')
    command = ['combine', '-M', 'GenerateOnly', '-d', str(source), '-m', '120',
               '-t', '-1', '--expectSignal', '1', '--saveToys', '-n', '_profileAsimov',
               '--setParameterRanges', 'r=0,20']
    require_success(runner.run(command, out, out / 'generate_asimov.log', [toys]))
    source_file = ROOT.TFile.Open(str(source))
    fit_file = ROOT.TFile.Open(str(work / 'multidimfit_initialFit_Test.root'))
    w, fit = source_file.Get('w'), fit_file.Get('fit_mdf')
    if fit.status() != 0 or fit.covQual() < 2:
        raise ValueError('invalid common starting fit')
    bestfit = {}
    for parameter in fit.floatParsFinal():
        var = w.var(parameter.GetName())
        if var:
            var.setVal(parameter.getVal())
            if parameter.getError() > 0:
                var.setError(parameter.getError())
            bestfit[var.GetName()] = var.getVal()
    w.var('r').setRange(0, 20)
    w.saveSnapshot('impact_initial', w.allVars())
    w.writeToFile(str(snapshot))
    fit_file.Close()
    source_file.Close()
    toy_file = ROOT.TFile.Open(str(toys))
    dataset = toy_file.Get('toys/toy_asimov')
    if not dataset:
        raise ValueError('GenerateOnly did not save toy_asimov')
    provenance.update(asimov_entries=dataset.numEntries(), asimov_sum_weights=dataset.sumEntries())
    toy_file.Close()
    provenance.update(snapshot=str(snapshot), snapshot_sha256=sha256(snapshot),
                      asimov=str(toys), asimov_sha256=sha256(toys),
                      generation_command=command, bestfit=bestfit,
                      physical_model_unchanged=True, observed_sr_used=False)
    write_json(setup, provenance)
    return provenance


class ImpactProfileRecovery:
    def __init__(self, work, out, inputs, workers, runner, refined=()):
        self.runner = runner
        self.work, self.out, self.inputs = work, out, inputs
        self.refined = set(refined)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.cache = {}
        self.cache_lock = threading.Lock()

    def submit(self, name, value):
        value = float(value)
        key = (name, format(value, '.12g'))
        with self.cache_lock:
            if key not in self.cache:
                self.cache[key] = self.pool.submit(self.fixed, name, value)
            return self.cache[key]

    def fixed(self, name, value):
        key = hashlib.sha256((name + ':' + format(value, '.12g')).encode()).hexdigest()[:14]
        folder = self.out / ('points_refined' if name in self.refined else 'points') / name / key
        folder.mkdir(parents=True, exist_ok=True)
        state = folder / 'state.json'
        if state.exists() and read_json(state).get('status') == 'complete':
            old = read_json(state)
            if sha256(old['root']) == old['sha256']:
                return old
        attempts = read_json(state).get('attempts', []) if state.exists() else []
        configurations = [
            ('s1', 1, []), ('s2', 2, []),
            ('explicit_stats', 1, ['--X-rtd', 'MINIMIZER_no_analytic',
                                  '--cminDefaultMinimizerAlgo', 'Combined',
                                  '--cminFallbackAlgo', 'Minuit2,Migrad,2:0.01', '-v', '3']),
            # Fixed-point profiles require a converged minimum, not a Hessian.
            # Avoid Hesse call-limit failures from weak, bounded rate parameters.
            ('explicit_stats_s0', 0, ['--X-rtd', 'MINIMIZER_no_analytic', '-v', '3']),
        ]
        tolerance = '0.01'
        if name in self.refined:
            # Keep the likelihood and priors unchanged. Only numerical precision
            # and finite function-call allowance differ for the failed profiles.
            extra = ['--X-rtd','MINIMIZER_no_analytic',
                     '--X-rtd','MINIMIZER_MaxCalls=2000000']
            configurations = [('precise_s0',0,extra),('precise_s1',1,extra)]
            tolerance = '0.0001'
        tried = {a.get('configuration', 's' + str(a['strategy'])) for a in attempts}
        error = attempts[-1]['error'] if attempts else 'not attempted'
        for configuration, strategy, extra in configurations:
            if configuration in tried:
                continue
            tag = '_profile_' + key + '_' + configuration
            root = folder / ('higgsCombine' + tag + '.MultiDimFit.mH120.root')
            cmd = ['combine', '-M', 'MultiDimFit', '-d', self.inputs['snapshot'],
                   '-m', '120', '-n', tag, '--algo', 'fixed', '--redefineSignalPOIs', 'r',
                   '-P', name, '--floatOtherPOIs', '1', '--saveInactivePOI', '1',
                   '--fixedPointPOIs', name + '=' + format(value, '.12g'),
                   '--snapshotName', 'impact_initial', '--skipInitialFit',
                   '-t', '-1', '--expectSignal', '1', '--toysFile', self.inputs['asimov'],
                   '--setParameterRanges', 'r=0,20', '--cminDefaultMinimizerStrategy', str(strategy),
                   '--cminDefaultMinimizerTolerance', tolerance] + extra
            started = time.time()
            write_json(state, dict(status='running', name=name, value=value, command=cmd,
                              started=started, attempts=attempts))
            record = self.runner.run(cmd, folder, folder / (configuration + '.log'), [root])
            exit_code = 0 if record['status'] == 'complete' else (record['exit_code'] or 1)
            error = None
            try:
                rows = tree_rows(root, name)
                if exit_code != 0 or len(rows) != 2:
                    raise ValueError('fixed point did not commit a successful conditional fit')
                row = rows[-1]
                if not all(math.isfinite(x) for x in row.values()):
                    raise ValueError('nonfinite conditional fit')
                if not math.isclose(row['theta'], value, rel_tol=2e-6, abs_tol=2e-7):
                    raise ValueError('fixed parameter differs from requested point')
                if row['deltaNLL'] < -0.005:
                    raise ValueError('conditional fit is below common minimum; investigate before collecting')
                result = dict(status='complete', name=name, value=value, **row,
                              root=str(root), sha256=sha256(root), strategy=strategy,
                              configuration=configuration,
                              seconds=time.time()-started, command=cmd, exit_code=exit_code,
                              attempts=attempts)
                write_json(state, result)
                return result
            except Exception as exc:
                error = str(exc)
            attempts.append(dict(strategy=strategy, configuration=configuration, exit_code=exit_code,
                                 seconds=time.time()-started, error=error))
        write_json(state, dict(status='failed', name=name, value=value, attempts=attempts))
        raise RuntimeError(name + ' at ' + str(value) + ': ' + error)

    def crossing(self, name, inner, outer):
        if not inner['deltaNLL'] < PROFILE_TARGET or not outer['deltaNLL'] >= PROFILE_TARGET:
            raise ValueError('crossing is not bracketed')
        points = [inner, outer]
        for _ in range(18):
            if abs(outer['deltaNLL'] - PROFILE_TARGET) <= PROFILE_NLL_TOL:
                return outer, points
            fraction = (PROFILE_TARGET-inner['deltaNLL'])/(outer['deltaNLL']-inner['deltaNLL'])
            fraction = min(0.8, max(0.2, fraction))
            value = inner['theta'] + fraction*(outer['theta']-inner['theta'])
            point = self.submit(name, value).result()
            points.append(point)
            if abs(point['deltaNLL']-PROFILE_TARGET) <= PROFILE_NLL_TOL:
                return point, points
            if point['deltaNLL'] < PROFILE_TARGET:
                inner = point
            else:
                outer = point
        raise RuntimeError('profile crossing did not converge: ' + name)

    def recover(self, name):
        saved = self.out / 'parameters' / (name + '.json')
        if saved.exists() and read_json(saved).get('status') == 'complete':
            result = read_json(saved)
            for point in result['points']:
                if sha256(point['root']) != point['sha256']:
                    raise ValueError('Completed recovery profile changed')
            return result
        oldroot = self.work / ('higgsCombine_paramFit_Test_' + name + '.MultiDimFit.mH120.root')
        oldrows = tree_rows(oldroot, name)
        center = dict(theta=float(self.inputs['bestfit'][name]),
                      r=float(self.inputs['bestfit']['r']), deltaNLL=0.0)
        if 'sgamma_shape' in name or 'qcd_norm' in name:
            maximum = oldrows[-1]['theta']
            values = [0, 0.25*center['theta'], 0.5*center['theta'], 0.75*center['theta'],
                      center['theta'], maximum, min(10., center['theta']+2*(maximum-center['theta']))]
            if name in self.refined:
                oldpoints = [read_json(p) for p in (self.out/'points'/name).glob('*/state.json')]
                candidates = [p for p in oldpoints if p['status']=='complete'
                              and p['theta']>center['theta'] and p['deltaNLL']>=0]
                if not candidates:
                    raise ValueError('refinement requires an existing upper-side profile seed')
                seed = min(candidates,key=lambda p:abs(p['deltaNLL']-PROFILE_TARGET))['theta']
                values = [0.,center['theta'],seed]
            pending = [self.submit(name, x) for x in values]
            points = [f.result() for f in pending]
            central = min(points, key=lambda p: abs(p['theta']-center['theta']))
            if abs(central['deltaNLL']) > 0.005:
                raise ValueError('saved global minimum not reproduced')
            lower_points = sorted((p for p in points if p['theta'] < center['theta']),
                                  key=lambda p: -p['theta'])
            previous, lower, lower_boundary = central, None, False
            for point in lower_points:
                if point['deltaNLL'] >= PROFILE_TARGET:
                    lower, extra = self.crossing(name, previous, point)
                    points.extend(extra)
                    break
                previous = point
            if lower is None:
                lower = previous
                if lower['theta'] != 0:
                    raise ValueError('lower profile did not reach physical boundary')
                lower_boundary = True
            upper_points = sorted((p for p in points if p['theta'] > center['theta']),
                                  key=lambda p: p['theta'])
            previous, upper, upper_boundary = central, None, False
            for point in upper_points:
                if point['deltaNLL'] >= PROFILE_TARGET:
                    upper, extra = self.crossing(name, previous, point)
                    points.extend(extra)
                    break
                previous = point
            if upper is None:
                for _ in range(12):
                    value = 10.
                    if name in self.refined:
                        distance = max(.25,previous['theta']-center['theta'])
                        value = min(10.,center['theta']+1.25*distance)
                    point = self.submit(name, value).result()
                    points.append(point)
                    if point['deltaNLL'] >= PROFILE_TARGET:
                        upper, extra = self.crossing(name, previous, point)
                        points.extend(extra)
                        break
                    if value == 10.:
                        upper, upper_boundary = point, True
                        break
                    previous = point
                if upper is None:
                    raise ValueError('upper profile not bracketed within finite refinement steps')
            result = dict(name=name, method='direct_profile', status='complete',
                          nominal=center, lower=lower, upper=upper,
                          boundary_limited_lower=lower_boundary, boundary_limited_upper=upper_boundary,
                          points=points, target_deltaNLL=PROFILE_TARGET, crossing_tolerance=PROFILE_NLL_TOL)
        else:
            bounds = self.inputs['parameter_bounds'][name]
            selected_rows = missing_endpoint_rows(oldrows, center['theta'], bounds)
            side = missing_endpoint_side(selected_rows, center['theta'], bounds)
            existing = selected_rows[-1]
            central = self.submit(name, center['theta']).result()
            if abs(central['deltaNLL']) > 0.005:
                raise ValueError('saved global minimum not reproduced')
            points, inner = [central], central
            distance = abs(existing['theta']-center['theta'])
            boundary = bounds[0] if side < 0 else bounds[1]
            for _ in range(8):
                guess = max(bounds[0], min(bounds[1], center['theta']+side*distance))
                point = self.submit(name, guess).result()
                points.append(point)
                if point['deltaNLL'] >= PROFILE_TARGET or abs(point['deltaNLL']-PROFILE_TARGET) <= PROFILE_NLL_TOL:
                    break
                if guess == boundary:
                    raise ValueError('missing endpoint has no 1-sigma crossing within original bounds')
                inner, distance = point, distance*1.5
            else:
                raise ValueError('missing endpoint could not be bracketed in eight steps')
            if abs(point['deltaNLL']-PROFILE_TARGET) > PROFILE_NLL_TOL:
                point, extra = self.crossing(name, inner, point)
                points.extend(extra)
            result = dict(name=name, method='missing_endpoint_profile', status='complete',
                          nominal=oldrows[0], lower=point if side < 0 else existing,
                          upper=point if side > 0 else existing, points=points,
                          boundary_limited_lower=False, boundary_limited_upper=False,
                          original_entries=oldrows,
                          existing_endpoint_preserved=True, target_deltaNLL=PROFILE_TARGET,
                          crossing_tolerance=PROFILE_NLL_TOL)
        if not result['lower']['theta'] < result['nominal']['theta'] < result['upper']['theta']:
            raise ValueError('recovered endpoints do not bracket nominal')
        write_json(self.out / 'parameters' / (name + '.json'), result)
        return result


def collect_profile_endpoints(work, out, recovered, source, accepted_original_stats=None, prefit=None):
    records = {p.stem: read_json(p) for p in (work / 'fit_state').glob('*.json') if p.stem != 'r'}
    if prefit is None:
        from HiggsAnalysis.CombinedLimit.tool_base import utils
        prefit = utils.prefit_from_workspace(str(source), 'w', list(records), None)
    if set(prefit) != set(records):
        raise ValueError('Prefit metadata must cover exactly the accepted nuisance inventory')
    initial = read_json(work / 'fit_state/r.json')['validation']['entries']
    data = dict(POIs=[dict(name='r', fit=[initial[1][0], initial[0][0], initial[2][0]])], params=[])
    for name, original in sorted(records.items()):
        parameter = dict(name=name, **prefit[name])
        if name in recovered:
            item = recovered[name]
            rows = [item[x] for x in ('lower', 'nominal', 'upper')]
            parameter.update(fit=[row['theta'] for row in rows], r=[row['r'] for row in rows],
                             recovery_method=item['method'],
                             boundary_limited_lower=item['boundary_limited_lower'],
                             boundary_limited_upper=item['boundary_limited_upper'])
        else:
            if original['status'] != 'complete':
                raise ValueError('missing recovered result: ' + name)
            root = work / ('higgsCombine_paramFit_Test_'+name+'.MultiDimFit.mH120.root')
            if accepted_original_stats is None:
                if sha256(root) != original['validation']['sha256']:
                    raise ValueError('successful output changed: ' + name)
            else:
                if artifact_stat(root) != accepted_original_stats[str(root)]:
                    raise ValueError('Accepted output changed during collection: '+name)
            rows = original['validation']['entries']
            parameter.update(fit=[rows[i][1] for i in (1,0,2)], r=[rows[i][0] for i in (1,0,2)])
        parameter['impact_r'] = max(abs(parameter['r'][i]-parameter['r'][1]) for i in (0,2))
        data['params'].append(parameter)
    expected_count = read_json(work / 'impact_status.json')['nuisance_count']
    if len(data['params']) != expected_count:
        raise ValueError('unexpected parameter count')
    data['recovery_metadata'] = dict(reused_parameters=len(records)-len(recovered), recovered_parameters=len(recovered),
        boundary_limited_parameters=[n for n,v in recovered.items() if v['boundary_limited_lower'] or v['boundary_limited_upper']],
        note='Boundary-limited rateParam endpoints are physical boundaries, not two-sided 1-sigma crossings.')
    target = out / 'impacts_mStop1200_mLSP500.json'
    write_json(target, data)
    return target


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


def publish_completed_profiles(work, output, shard_statuses, cached_collection=None):
    """Collect terminal accepted profiles, without replaying the fit preflight.

    Producer-source changes cannot invalidate a completed fit's receipt. This
    stage launches no fit and does not rehash/reopen accepted fit ROOT files.
    Endpoint extraction and impact definitions stay in the existing collector.
    """
    work, output = internal_path(work), internal_path(output)
    statuses = [internal_path(p) for p in shard_statuses]
    # status.json is in a finite phase below its independently locked shard.
    with ExitStack() as stack:
        for folder in [work, *dict.fromkeys(p.parent.parent for p in statuses)]:
            lock = stack.enter_context((folder/'impact.lock').open('r'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent = read_json(work/'impact_status.json')
        if (process_active(parent, 'controller_pid') or parent.get('sr_data_used') is not False
                or parent.get('asimov_expect_signal') != 1):
            raise ValueError('Require terminal blinded mu=1 Asimov impact receipts')
        if parent.get('status') in ('fits_complete','generated_pending_visual_QA') and parent.get('json'):
            return internal_path(parent['json'])
        records = {p.stem:read_json(p) for p in (work/'fit_state').glob('*.json') if p.stem!='r'}
        if set(records)!=set(parent['names']) or len(records)!=parent['nuisance_count']:
            raise ValueError('Incomplete terminal impact inventory')
        original, metadata = set(), {}
        for name, item in records.items():
            if item.get('status') != 'complete':
                continue
            validation = item.get('validation',{})
            accepted = [a for a in item.get('attempts',[]) if a.get('exit_code')==0
                and a.get('validation',{}).get('valid') is True
                and a['validation'].get('sha256')==validation.get('sha256')]
            if validation.get('valid') is not True or not accepted:
                raise ValueError('Missing accepted execution receipt: '+name)
            original.add(name)
            product = work/('higgsCombine_paramFit_Test_'+name+'.MultiDimFit.mH120.root')
            metadata[str(product)] = artifact_stat(product)
        initial = read_json(work/'fit_state/r.json')
        if initial.get('status')!='complete' or initial.get('validation',{}).get('valid') is not True:
            raise ValueError('Missing accepted initial fit')
        recovered, origins = {}, []
        # Product paths in these completed receipts are relative to the same
        # accepted analysis root as the original workspace, not the code tree.
        analysis = None
        # Resolve the root from the published primary's absolute location.
        # This avoids interpreting old relative receipts against a new code root.
        for candidate in work.parents:
            if (candidate/'stats/runtime.json').is_file():
                analysis = candidate
                break
        if analysis is None:
            raise ValueError('Accepted analysis root is unavailable')
        for path in statuses:
            status = read_json(path)
            if (status.get('status')!='profiles_complete' or status.get('controller_pid') is not None
                    or bool(status.get('failed')) or status.get('observed_sr_used') is not False
                    or internal_path(status['primary_work'])!=work
                    or set(status.get('parameter_products',{}))!=set(status['parameters'])):
                raise ValueError('Recovery shard is not terminal and complete')
            for name, product in status['parameter_products'].items():
                if name in recovered or name in original:
                    raise ValueError('Duplicate accepted nuisance: '+name)
                receipt = internal_path(analysis/product['path'])
                if path.parent.parent not in receipt.parents:
                    raise ValueError('Parameter receipt outside its completed shard')
                item = read_json(receipt)
                if (item.get('status')!='complete' or item.get('name')!=name
                        or item.get('bounds')!=records[name]['bounds']):
                    raise ValueError('Accepted recovery identity differs')
                recovered[name] = item
            origins.append(str(path))
        if set(recovered)!=set(parent['failed_nuisances']) or original|set(recovered)!=set(records):
            raise ValueError('Complete collection must include each nuisance exactly once')
        prefit = {}
        if cached_collection is not None:
            cache = read_json(internal_path(cached_collection))
            for item in cache['params']:
                prefit[item['name']] = {k:item[k] for k in ('prefit','type','groups') if k in item}
        missing = sorted(set(records)-set(prefit))
        if missing:
            cmssw_python = str(Path(os.environ['CMSSW_BASE'])/'python')
            if cmssw_python not in sys.path:
                sys.path.insert(0,cmssw_python)
            from HiggsAnalysis.CombinedLimit.tool_base import utils
            prefit.update(utils.prefit_from_workspace(str(parent['workspace']),'w',missing,None))
        output.mkdir(parents=True,exist_ok=True)
        target = collect_profile_endpoints(work,output,recovered,Path(parent['workspace']),
            accepted_original_stats=metadata,prefit=prefit)
        payload = read_json(target)
        if (len(payload['params'])!=len(records) or any(not all(math.isfinite(float(v))
                for k in ('fit','r') for v in p[k]) for p in payload['params'])):
            raise ValueError('Nonfinite or incomplete collected impact endpoints')
        status = dict(status='fits_complete',controller_pid=None,finished=time.time(),
            json=str(target),json_sha256=sha256(target),total_valid=len(records),
            recovered_parameters=sorted(recovered),shard_statuses=origins,
            fits_launched=False,upstream_revalidated=False,accepted_fit_roots_reopened=0,
            cached_prefit_count=len(records)-len(missing),new_prefit_metadata_count=len(missing),
            boundary_limited_parameters=payload['recovery_metadata']['boundary_limited_parameters'])
        write_json(output/'status.json',status)
        parent.update(status='fits_complete',phase='collected_with_parallel_bounded_sgamma',
            json=str(target),json_sha256=status['json_sha256'],failed_nuisances=[],
            valid_nuisances=len(records),recovery_status=str(output/'status.json'),
            plots_status='pending_original_plotImpacts')
        write_json(work/'impact_status.json',parent)
        return target


def bounded_sgamma_scope(parent, records, bounds):
    names = recovery_scope(parent, records, (), (), {})
    if any(not re.fullmatch(r'CMS_NPS26012_(?:sgamma_shape_lowdm|qcd_norm_highdm)_Nb(?:1|2plus)_bin\d+(?:_202[2345](?:EE|BPix)?)?', n)
           or bounds.get(n) != [.01, 5.] for n in names):
        raise ValueError('Requires failed Sgamma/QCD rate parameters with adopted [0.01,5] bounds')
    return names


def bounded_lower_values(center, bounds):
    lo, hi = bounds
    if bounds != [.01, 5.] or not lo < center < hi:
        raise ValueError('Invalid adopted Sgamma center/bounds')
    return [lo + f*(center-lo) for f in (.75, .5, .25, 0.)]


def bounded_boundary_pair(first, second, edge, r_tolerance=.0002):
    for point in (first, second):
        if (point['status'] != 'complete' or not math.isclose(point['theta'], edge, abs_tol=2e-7)
                or not all(math.isfinite(point[k]) for k in ('theta','r','deltaNLL'))
                or not -.005 <= point['deltaNLL'] < PROFILE_TARGET):
            raise ValueError('Not a validated approved-boundary endpoint')
    if (abs(first['deltaNLL']-second['deltaNLL']) > PROFILE_NLL_TOL
            or r_tolerance is not None and abs(first['r']-second['r']) > r_tolerance):
        raise ValueError('Independent boundary minima disagree')


def reconcile_seed_bounds(seeds, bounds):
    """Restore only the exact binary32 image of an endpoint.

    Combine's trackedParam branches are floats. The stored value of 0.01 is
    0.009999999776..., which its double-precision CLI correctly rejects.
    This is serialization reconciliation, never clipping an out-of-range fit.
    """
    result = dict(seeds)
    for name, value in result.items():
        lo, hi = bounds[name]
        if not math.isfinite(value):
            raise ValueError('Nonfinite conditional seed')
        edge = lo if value < lo else hi if value > hi else None
        if edge is not None:
            if value != struct.unpack('f', struct.pack('f', edge))[0]:
                raise ValueError('Conditional seed genuinely outside range: '+name)
            result[name] = edge
    return result


def saved_sgamma_seeds(attempt_rows, bounds):
    """Use earlier endpoints as scan seeds, never as accepted profile results."""
    centers, upper = [], []
    for rows in attempt_rows:
        if not rows or not all(math.isfinite(v) for row in rows for v in row.values()):
            continue
        center = rows[0]['theta']
        if not bounds[0] < center < bounds[1]:
            continue
        centers.append(center)
        upper.extend(row['theta'] for row in rows[1:]
                     if center < row['theta'] <= bounds[1])
    if not centers:
        raise ValueError('No finite interior starting point in saved Sgamma attempts')
    return dict(centers=centers, upper=upper[0] if upper else None,
                saved_endpoints_are_seeds_only=True)


class BoundedSgamma(ImpactProfileRecovery):
    """Current-model profile geometry; reuse fixed commands and crossing math."""
    def __init__(self, work, out, inputs, workers, runner, previous=None, earlier=None):
        super().__init__(work, out, inputs, workers, runner)
        self.points = {}
        self.previous = previous
        self.prior_roots = [p for p in (previous, earlier) if p]

    def fixed(self, name, value, label=None, seed=None, strategies=(0, 1)):
        lo, hi = self.inputs['parameter_bounds'][name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Conditional point outside approved range')
        if label == 'lower_independent':
            old = next((read_json(p/'points'/name/label/'state.json') for p in self.prior_roots
                        if (p/'points'/name/label/'state.json').exists()), None)
            if (old and old['status'] == 'failed') or self.inputs.get('complete_recovery'):
                # Independent strategy1 from the measured halfway point, not
                # the successful boundary fit or a repeated failed cold start.
                theta = self.inputs['bestfit'][name]
                seed = self.fixed(name, lo+.5*(theta-lo))
                label = 'lower_independent_neighbour'
        key = label or hashlib.sha256(format(value, '.15g').encode()).hexdigest()[:16]
        polish = name in self.inputs.get('polish_parameters', [])
        hybrid = name in self.inputs.get('hybrid_parameters', [])
        likelihood = name in self.inputs.get('likelihood_parameters', [])
        if polish:
            key = 'precision_' + key
        if hybrid and (value in (lo,hi) or label and label.startswith(('lower_independent','upper_independent','return_'))):
            key = 'hybrid_' + key
        folder = self.out/'points'/name/key
        state_path = folder/'state.json'
        if state_path.exists() and read_json(state_path)['status'] == 'complete':
            result = read_json(state_path)
            self.validate_point(result)
            self.points.setdefault(name, []).append(result)
            return result
        for previous in self.prior_roots:
            old = previous/'points'/name/key/'state.json'
            if old.exists() and read_json(old)['status'] == 'complete':
                result = read_json(old)
                self.validate_point(result)
                self.points.setdefault(name, []).append(result)
                return result
        if seed is None:
            candidates = self.points.get(name, [])
            if candidates:
                seed = min(candidates, key=lambda p:abs(p['theta']-value))
        starts = tracked_start(seed, set(self.inputs['bestfit'])) if seed else None
        if starts is not None and (self.previous or self.inputs.get('complete_recovery')):
            starts = reconcile_seed_bounds(starts, self.inputs['all_parameter_bounds'])
            if label and label.startswith('return_neighbour_'):
                # Release starting values pinned to another parameter's bound;
                # only initialization changes, all other nuisances still float.
                starts = {n:.5*v+.5*self.inputs['bestfit'][n] for n,v in starts.items()}
        folder.mkdir(parents=True, exist_ok=True)
        attempts = []
        for strategy in strategies:
            tag = '_bounded_'+key+'_s'+str(strategy)
            product = folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
            extra = ['--cminDefaultMinimizerStrategy', str(strategy)]
            if self.previous or self.inputs.get('complete_recovery'):
                extra += ['--cminDefaultMinimizerTolerance', '0.001']
            if polish:
                extra[extra.index('--cminDefaultMinimizerTolerance')+1]='0.00001'
                extra += ['--cminDefaultMinimizerPrecision','1e-10']
            command = fixed_command(self.inputs, name, value, tag, extra=extra, seeds=starts)
            if hybrid:
                command[command.index('--cminDefaultMinimizerTolerance')+1]='0.0001'
                command += ['--cminDefaultMinimizerAlgo','Combined','--cminDefaultMinimizerPrecision','1e-8']
                index=command.index('MINIMIZER_no_analytic');del command[index-1:index+1]
            if likelihood:
                command[command.index('--cminDefaultMinimizerTolerance')+1]='0.01' if strategy==0 else '0.1'
            record = self.runner.run(command, folder, folder/('s'+str(strategy)+'.log'), [product])
            try:
                if record['status'] != 'complete' or record['exit_code'] != 0:
                    raise ValueError('Conditional command did not complete')
                rows = tree_rows(product, name)
                if len(rows) != 2 or not all(math.isfinite(v) for row in rows for v in row.values()):
                    raise ValueError('Conditional minimization did not commit two finite rows')
                point = rows[-1]
                if (not math.isclose(point['theta'], value, rel_tol=2e-6, abs_tol=2e-7)
                        or point['deltaNLL'] < -.005):
                    raise ValueError('Wrong coordinate or below common minimum')
                result = dict(status='complete', name=name, value=value, **point,
                    root=str(product), sha256=sha256(product), command=command,
                    strategy=strategy, exit_code=0, seed_root=seed['root'] if seed else None,
                    log=record['log'], log_sha256=sha256(ROOT/record['log']), attempts=attempts)
                write_json(state_path, result)
                self.points.setdefault(name, []).append(result)
                return result
            except Exception as error:
                attempts.append(dict(strategy=strategy, command=command, error=repr(error),
                    execution=record))
                write_json(state_path, dict(status='failed', name=name, value=value, attempts=attempts))
        raise RuntimeError(name+' conditional fit failed at '+str(value))

    def validate_point(self, point):
        from TROTASR.workflows.diagnostic_common import command_key
        if point['status'] != 'complete' or sha256(point['root']) != point['sha256']:
            raise ValueError('Conditional result changed')
        rows = tree_rows(point['root'], point['name'])
        if (len(rows) != 2 or not all(math.isfinite(v) for row in rows for v in row.values())
                or any(rows[-1][k] != point[k] for k in ('theta','r','deltaNLL'))
                or not math.isclose(point['theta'],point['value'],rel_tol=2e-6,abs_tol=2e-7)
                or point['deltaNLL'] < -.005):
            raise ValueError('Conditional ROOT/receipt disagreement')
        if sha256(ROOT/point['log']) != point['log_sha256']:
            raise ValueError('Conditional log changed')
        cwd = Path(point['root']).parent
        receipt_out = cwd.parents[2]
        if receipt_out not in [self.out, *self.prior_roots]:
            raise ValueError('Conditional result outside the frozen recovery namespaces')
        record = read_json(receipt_out/'commands'/(command_key(point['command'],cwd)+'.json'))
        if (record['status'] != 'complete' or record['exit_code'] != 0 or record.get('pid')
                or record['products'].get(str(Path(point['root']).relative_to(ROOT))) != point['sha256']):
            raise ValueError('Conditional execution receipt invalid')

    def recover(self, name):
        target = self.out/'parameters'/(name+'.json')
        if not target.exists():
            for previous in self.prior_roots:
                if (previous/'parameters'/(name+'.json')).exists():
                    result = read_json(previous/'parameters'/(name+'.json'))
                    for p in result['points']: self.validate_point(p)
                    return result
        if target.exists():
            result = read_json(target)
            for p in result['points']: self.validate_point(p)
            return result
        theta = self.inputs['bestfit'][name]
        bounds = self.inputs['parameter_bounds'][name]
        likelihood = name in self.inputs.get('likelihood_parameters', [])
        def common_matches(point):
            return abs(point['deltaNLL']) <= .005 and (likelihood or
                abs(point['r']-self.inputs['bestfit']['r']) <= .0002)
        diagnostics={}
        central = self.fixed(name, theta)
        if not common_matches(central):
            raise ValueError('Common minimum not reproduced')
        lower = None; lower_boundary = False; inner = central
        for value in bounded_lower_values(theta, bounds):
            outer = self.fixed(name, value, seed=inner)
            if abs(outer['deltaNLL']-PROFILE_TARGET) <= PROFILE_NLL_TOL:
                lower = outer; break
            if outer['deltaNLL'] >= PROFILE_TARGET:
                lower, _ = self.crossing(name, inner, outer); break
            inner = outer
        if lower is None:
            lower = inner
            # Independent direct seed and strategy: never infer a bound from
            # failed minimization or a missing crossing alone.
            check = self.fixed(name, bounds[0], 'lower_independent', central, (1,))
            if self.inputs.get('refine_boundaries') and abs(lower['deltaNLL']-check['deltaNLL'])>PROFILE_NLL_TOL:
                best=min((lower,check),key=lambda point:point['deltaNLL'])
                if self.inputs.get('final_boundary'):
                    checked=self.fixed(name,bounds[0],'lower_final_same_minimum',best,(2,))
                else:
                    neighbour=self.fixed(name,bounds[0]+.1*(theta-bounds[0]),'lower_refine_neighbour',best,(1,))
                    checked=self.fixed(name,bounds[0],'lower_refine_return',neighbour,(1,))
                candidates=sorted((lower,check,checked),key=lambda point:point['deltaNLL'])
                lower,check=candidates[:2]
            bounded_boundary_pair(lower, check, bounds[0],None if likelihood else .0002)
            diagnostics['independent_lower_r_difference']=abs(lower['r']-check['r'])
            diagnostics['independent_lower_nll_difference']=abs(lower['deltaNLL']-check['deltaNLL'])
            lower_boundary = True
        returned = self.fixed(name, theta, 'return_to_common', lower)
        if not common_matches(returned):
            returned = lower
            for index, fraction in enumerate((.25,.5,.75,1.)):
                returned = self.fixed(name, bounds[0]+fraction*(theta-bounds[0]),
                    'return_neighbour_'+str(index), returned)
        if not common_matches(returned):
            raise ValueError('Return from lower endpoint does not reproduce common minimum')
        diagnostics['return_r_difference']=abs(returned['r']-self.inputs['bestfit']['r'])
        diagnostics['return_deltaNLL']=returned['deltaNLL']
        if self.inputs.get('complete_recovery'):
            saved = self.inputs['endpoint_seeds'][name]['upper']
            value = min(bounds[1], max(theta+.01, saved if saved is not None else theta+.25*(bounds[1]-theta)))
        else:
            oldrows = tree_rows(self.work/('higgsCombine_paramFit_Test_'+name+'.MultiDimFit.mH120.root'),name)
            value = min(bounds[1], max(theta+.01, oldrows[-1]['theta']))
        inner = central; upper = None; upper_boundary = False
        for _ in range(12):
            outer = self.fixed(name, value, seed=inner)
            if abs(outer['deltaNLL']-PROFILE_TARGET) <= PROFILE_NLL_TOL:
                upper = outer; break
            if outer['deltaNLL'] >= PROFILE_TARGET:
                upper, _ = self.crossing(name, inner, outer); break
            if value == bounds[1]:
                check = self.fixed(name, bounds[1], 'upper_independent', central, (1,))
                bounded_boundary_pair(outer, check, bounds[1],None if likelihood else .0002)
                diagnostics['independent_upper_r_difference']=abs(outer['r']-check['r'])
                diagnostics['independent_upper_nll_difference']=abs(outer['deltaNLL']-check['deltaNLL'])
                upper, upper_boundary = outer, True; break
            inner = outer
            value = min(bounds[1], theta+1.5*(value-theta))
        if upper is None or not lower['theta'] < theta < upper['theta']:
            raise ValueError('Finite profile endpoints did not bracket the common minimum')
        result = dict(status='complete', name=name, method='bounded_sgamma_neighbour_profile',
            nominal=dict(theta=theta,r=self.inputs['bestfit']['r'],deltaNLL=0.),
            lower=lower,upper=upper,boundary_limited_lower=lower_boundary,
            boundary_limited_upper=upper_boundary,points=self.points[name],bounds=bounds,
            target_deltaNLL=PROFILE_TARGET,crossing_tolerance=PROFILE_NLL_TOL,
            numerical_diagnostics=diagnostics,r_agreement_is_diagnostic=likelihood)
        for p in result['points']: self.validate_point(p)
        write_json(target,result)
        return result


def bounded_preflight(manifest, work, model, mass):
    import ROOT as root
    root.gROOT.SetBatch(True); root.EnableThreadSafety()
    _, origins = grid_card(manifest, 'impact', model, mass)
    grid = read_json(manifest)
    if grid['scope'] != 'full_combined_systematic_production':
        raise ValueError('Bounded Sgamma requires the current combined model')
    parent = read_json(work/'impact_status.json')
    contract = read_json(work/'contract.json')
    if contract['origins'] != origins or process_active(parent, 'controller_pid'):
        raise ValueError('Impact provenance changed or primary still active')
    if sha256(parent['workspace']) != parent['workspace_sha256']:
        raise ValueError('Impact workspace changed')
    records = {p.stem:read_json(p) for p in (work/'fit_state').glob('*.json') if p.stem != 'r'}
    protected = {str(p.relative_to(ROOT)):sha256(p) for p in (work/'fit_state').glob('*.json')}
    protected.update({str(p.relative_to(ROOT)):sha256(p) for p in work.glob('*.root')})
    initial = read_json(work/'fit_state/r.json')
    for name,record in {'r':initial,**records}.items():
        for attempt in record['attempts']:
            for path,digest in attempt['products'].items():
                if sha256(ROOT/path) != digest: raise ValueError('Original fit changed')
                protected[path] = digest
            if record['status'] == 'complete' and attempt['validation']['valid']:
                if attempt['exit_code'] != 0: raise ValueError('Invalid original execution')
                tag = '_initialFit_Test' if name == 'r' else '_paramFit_Test_'+name
                checked = validate_fit(ROOT/attempt['folder'],name,tag,record['bounds'])
                if not checked['valid'] or checked['sha256'] != attempt['validation']['sha256']:
                    raise ValueError('Existing valid fit no longer validates: '+name)
                break
    if initial['status'] != 'complete': raise ValueError('Initial fit not validated')
    source = root.TFile.Open(parent['workspace'])
    try:
        ws = source.Get('w')
        bounds = {n:[float(ws.var(n).getMin()),float(ws.var(n).getMax())] for n in parent['failed_nuisances']}
    finally: source.Close()
    names = bounded_sgamma_scope(parent,records,bounds)
    seeds = {}
    for n in names:
        attempts = [a for a in records[n]['attempts'] if a.get('exit_code') == 0]
        rows = [tree_rows(ROOT/a['folder']/('higgsCombine_paramFit_Test_'+n+'.MultiDimFit.mH120.root'),n)
                for a in attempts]
        seeds[n] = saved_sgamma_seeds(rows, bounds[n])
    return dict(origins=origins,names=names,bounds=bounds,protected=protected,
        endpoint_seeds=seeds,
        parent=parent,original_status_sha256=sha256(work/'impact_status.json'),
        original_valid=len(records)-len(names),nuisance_count=len(records))


def run_bounded(manifest, work, workers, model, mass, *,
                polish=False, hybrid=False, likelihood=False, refine=False, final=False):
    if not 1 <= workers <= 98: raise ValueError('Reserve controller and diagnostic slots')
    manifest,work = internal_path(manifest),internal_path(work)
    with (work/'impact.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit = bounded_preflight(manifest,work,model,mass)
        previous = None
        if polish:previous=work/'bounded_sgamma_complete'
        if hybrid:previous=work/'bounded_sgamma_precision'
        if likelihood:previous=work/'bounded_sgamma_minimizer'
        if refine:previous=work/'bounded_sgamma_likelihood'
        if final:previous=work/'bounded_sgamma_refined'
        earlier = None
        if hybrid or likelihood or refine:earlier=work/'bounded_sgamma_complete'
        prior = {}
        if previous:
            old_state = read_json(previous/'status.json')
            old_contract = read_json(previous/'contract.json')
            if (old_state['status'] != 'needs_attention' or process_active(old_state, 'controller_pid')
                    or not old_state.get('failed') or old_contract['origins'] != audit['origins']):
                raise ValueError('Requires terminal, unpromoted original bounded recovery')
            prior = dict(previous_contract_sha256=sha256(previous/'contract.json'),
                previous_status_sha256=sha256(previous/'status.json'),
                failed_parameters=sorted(old_state['failed']),
                reused_complete_parameters=sorted(old_state['completed']),
                seed_reconciliation='exact_binary32_image_of_approved_boundary_only')
            audit['protected'].update({str(p.relative_to(ROOT)):sha256(p) for p in previous.rglob('*') if p.is_file()})
        phase = ('bounded_sgamma_final' if final else
                 'bounded_sgamma_refined' if refine else
                 'bounded_sgamma_likelihood' if likelihood else
                 'bounded_sgamma_minimizer' if hybrid else
                 'bounded_sgamma_precision' if polish else 'bounded_sgamma_complete')
        out = work/phase
        out.mkdir(exist_ok=True)
        contract = freeze_contract(out,audit['origins'],SOURCES,dict(parameters=audit['names'],
            bounds=audit['bounds'],original_status_sha256=audit['original_status_sha256'],
            protected=audit['protected'],expect_signal=1,r_range=[0,20],target=.5,
            crossing_tolerance=.002,maximum_conditional_configurations=2,
            lower_chain_fractions=[.75,.5,.25,0.],upper_steps=12,crossing_steps=18,
            max_calls=2000000,tolerance=.0001 if hybrid else .00001 if polish else .001,
            precision=1e-8 if hybrid else 1e-10 if polish else None,algorithm='Combined' if hybrid else 'Migrad',
            analytic_stats=hybrid,strategies=[0,1],workers=workers,
            configuration_tolerances=[.01,.1] if likelihood else None,
            independent_r_agreement_diagnostic_only=likelihood,
            boundary_refinement=refine,refinement_neighbour_fraction=.1 if refine else None,
            final_boundary_strategy=2 if final else None,
            complete_recovery=True,endpoint_seeds=audit['endpoint_seeds'],
            physical_model_unchanged=True,common_minimum_unchanged=True,**prior))
        if not (out/'original_impact_status.json').exists():
            write_json(out/'original_impact_status.json',audit['parent'])
        runner = CommandRunner(out)
        state = dict(status='running',controller_pid=os.getpid(),started=time.time(),
            completed=[],failed={},original_successes=audit['original_valid'],
            parameters=audit['names'],workers=workers,observed_sr_used=False)
        write_json(out/'status.json',state)
        worker = None
        try:
            inputs = prepare_profile_inputs(work,earlier or previous or out,Path(audit['parent']['workspace']),runner)
            inputs['parameter_bounds'] = audit['bounds']
            inputs.update(complete_recovery=True,endpoint_seeds=audit['endpoint_seeds'])
            if polish:inputs['polish_parameters']=prior['failed_parameters']
            if hybrid:inputs['hybrid_parameters']=prior['failed_parameters']
            if likelihood:inputs['likelihood_parameters']=prior['failed_parameters']
            if refine:inputs['refine_boundaries']=True
            if final:inputs['final_boundary']=True
            import ROOT as root
            source = root.TFile.Open(inputs['snapshot'])
            try:
                ws = source.Get('w')
                inputs['all_parameter_bounds'] = {n:[float(ws.var(n).getMin()),float(ws.var(n).getMax())]
                    for n in inputs['bestfit']}
            finally: source.Close()
            worker = BoundedSgamma(work,out,inputs,workers,runner,previous,earlier)
            if final:worker.prior_roots.append(work/'bounded_sgamma_likelihood')
            recovered = {}
            with ThreadPoolExecutor(max_workers=min(workers,len(audit['names']))) as pool:
                tasks = {pool.submit(worker.recover,n):n for n in audit['names']}
                for task in as_completed(tasks):
                    n = tasks[task]
                    try: recovered[n]=task.result();state['completed'].append(n)
                    except Exception as error: state['failed'][n]=repr(error)
                    write_json(out/'status.json',state)
            for name,digest in audit['protected'].items():
                if sha256(ROOT/name) != digest: raise ValueError('Protected original changed: '+name)
            state['protected_files_unchanged'] = len(audit['protected'])
            if state['failed']:
                state['status']='needs_attention';return 2
            cmssw_python = str(Path(os.environ['CMSSW_BASE'])/'python')
            if cmssw_python not in sys.path: sys.path.insert(0, cmssw_python)
            target = collect_profile_endpoints(work,out,recovered,Path(audit['parent']['workspace']))
            named = out/('impacts_'+model+'_'+mass+'.json')
            if named != target: target.rename(named)
            target = named
            payload = read_json(target)
            if len(payload['params']) != audit['nuisance_count'] or any(
                    not all(math.isfinite(x) for key in ('fit','r') for x in p[key]) for p in payload['params']):
                raise ValueError('Invalid complete impact collection')
            state.update(status='fits_complete',total_valid=audit['nuisance_count'],
                json=str(target.relative_to(ROOT)),json_sha256=sha256(target),
                boundary_limited_parameters=payload['recovery_metadata']['boundary_limited_parameters'])
            parent=audit['parent']
            parent.update(status='fits_complete',phase='collected_with_bounded_sgamma',
                json=state['json'],json_sha256=state['json_sha256'],failed_nuisances=[],
                valid_nuisances=audit['nuisance_count'],recovery_status=str((out/'status.json').relative_to(ROOT)),
                boundary_limited_parameters=state['boundary_limited_parameters'],
                plots_status='pending_original_plotImpacts')
            write_json(work/'impact_status.json',parent)
        except BaseException as error:
            state.update(status='needs_attention',error=repr(error));raise
        finally:
            if worker: worker.pool.shutdown()
            state.update(controller_pid=None,finished=time.time());write_json(out/'status.json',state)
    return 0


def recovery_scope(status, records, missing, refined, previous):
    failed = {name for name, value in records.items() if value['status'] == 'failed'}
    names = status['failed_nuisances']
    if status['status'] != 'needs_attention' or not names or set(names) != failed:
        raise ValueError('Recovery requires exact recorded failed parameters')
    if len(records) != status['nuisance_count'] or not set(missing).issubset(failed):
        raise ValueError('Missing or unrequested impact recovery records')
    if any(not ('sgamma_shape' in n or 'qcd_norm' in n or n in missing) for n in names):
        raise ValueError('Unsupported failure; inspect and explicitly identify a missing endpoint')
    if refined and (previous.get('status') != 'needs_attention'
                    or not set(refined).issubset(previous.get('failed', {})) or previous.get('refined_parameters')):
        raise ValueError('Only one finite precision pass on previously failed direct profiles')
    return names


def run_direct_profile(manifest, work, workers, missing=(), refined=()):
    if not 1 <= workers <= 99:
        raise ValueError('Reserve one controller slot')
    work = internal_path(work)
    _, origins = grid_card(manifest, 'impact')
    contract = read_json(work / 'contract.json')
    if contract['origins'] != origins:
        raise ValueError('Original impact origins changed')
    out = work / 'direct_profile'
    out.mkdir(exist_ok=True)
    with (work / 'impact.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / 'status.json'
        previous = read_json(state_path) if state_path.exists() else {}
        freeze_contract(out, origins, SOURCES, dict(expect_signal=1, r_range=[0, 20], delta_nll=.5,
            crossing_tolerance=.002, missing_endpoint=sorted(missing)))
        if previous.get('status') == 'fits_complete':
            if sha256(ROOT / previous['json']) != previous['json_sha256']:
                raise ValueError('Completed recovery JSON changed')
            for name, digest in previous['products'].items():
                if sha256(ROOT / name) != digest:
                    raise ValueError('Completed recovery output changed')
            return 0
        parent = read_json(work / 'impact_status.json')
        records = {p.stem: read_json(p) for p in (work / 'fit_state').glob('*.json') if p.stem != 'r'}
        names = recovery_scope(parent, records, missing, refined, previous)
        if sha256(parent['workspace']) != parent['workspace_sha256']:
            raise ValueError('Original impact workspace changed')
        initial = read_json(work / 'fit_state/r.json')
        if initial['status'] != 'complete':
            raise ValueError('No successful initial fit for recovery')
        for attempt in initial['attempts']:
            if attempt['validation']['valid']:
                for name, digest in attempt['products'].items():
                    if sha256(ROOT / name) != digest:
                        raise ValueError('Initial fit changed')
        import ROOT as root
        root.gROOT.SetBatch(True)
        root.EnableThreadSafety()
        source = root.TFile.Open(parent['workspace'])
        bounds = {}
        try:
            ws = source.Get('w')
            for name in names:
                var = ws.var(name)
                if not var:
                    raise ValueError('Missing workspace parameter: ' + name)
                bounds[name] = [float(var.getMin()), float(var.getMax())]
                if name in missing:
                    rows = tree_rows(work / ('higgsCombine_paramFit_Test_' + name + '.MultiDimFit.mH120.root'), name)
                    missing_endpoint_rows(rows, rows[0]['theta'], bounds[name])
                elif bounds[name] != [0, 10]:
                    raise ValueError('Unexpected physical rate-parameter bounds: ' + name)
        finally:
            source.Close()
        if refined:
            backup = out / 'status_before_precision_refinement.json'
            if backup.exists():
                raise ValueError('A precision refinement was already attempted')
            write_json(backup, previous)
        state = dict(status='running', controller_pid=os.getpid(), started=time.time(),
            original_successes=len(records)-len(names), recovery_count=len(names), workers=workers,
            refined_parameters=sorted(refined), completed=[], failed={}, observed_sr_used=False,
            physical_model_unchanged=True, full_workflow_complete=False)
        write_json(state_path, state)
        runner = CommandRunner(out)
        recovery = None
        try:
            inputs = prepare_profile_inputs(work, out, Path(parent['workspace']), runner)
            inputs['parameter_bounds'] = bounds
            recovery = ImpactProfileRecovery(work, out, inputs, workers, runner, refined)
            recovered = {}
            with ThreadPoolExecutor(max_workers=min(len(names), workers)) as pool:
                tasks = {pool.submit(recovery.recover, n): n for n in names}
                for task in as_completed(tasks):
                    name = tasks[task]
                    try:
                        recovered[name] = task.result()
                        state['completed'].append(name)
                    except Exception as error:
                        state['failed'][name] = repr(error)
                    write_json(state_path, state)
            if state['failed']:
                state['status'] = 'needs_attention'
                return 2
            target = collect_profile_endpoints(work, out, recovered, Path(parent['workspace']))
            payload = read_json(target)
            for p in payload['params']:
                if not all(math.isfinite(x) for k in ('fit', 'r') for x in p[k]):
                    raise ValueError('Nonfinite recovered impact')
            state.update(status='fits_complete', json=str(target.relative_to(ROOT)), json_sha256=sha256(target),
                total_valid=parent['nuisance_count'], products={str(p.relative_to(ROOT)): sha256(p)
                    for p in out.rglob('*.root')}, boundary_limited_parameters=payload['recovery_metadata']['boundary_limited_parameters'])
            parent.update(status='fits_complete', phase='collected_with_bounded_recovery',
                json=state['json'], json_sha256=state['json_sha256'], failed_nuisances=[],
                valid_nuisances=parent['nuisance_count'], recovery_status=str(state_path.relative_to(ROOT)),
                boundary_limited_parameters=state['boundary_limited_parameters'],
                fit_products={**{str(p.relative_to(ROOT)): sha256(p) for p in work.glob('*.root')}, **state['products']},
                plots_status='pending_original_plotImpacts')
            write_json(work / 'impact_status.json', parent)
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error))
            raise
        finally:
            if recovery:
                recovery.pool.shutdown()
            state.update(controller_pid=None, finished=time.time())
            write_json(state_path, state)
    return 0


def run_with_recovery(manifest, output, workers, plot_only=False, model='T2tt',
                      mass='mStop1200_mLSP500', workspace_state=None, revision=None):
    """The impacts stage owns primary fits and their finite bounded profiles."""
    result = _run(manifest, output, workers, plot_only, model, mass, workspace_state, revision)
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
                                 mass=mass, **options)
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
            result = run_direct_profile(manifest, internal_path(output), workers, missing=missing)
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
