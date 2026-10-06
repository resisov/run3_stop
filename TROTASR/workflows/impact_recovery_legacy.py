"""Existing bounded-profile recovery math, with internal I/O and finite commands.

Numerical formulas, configurations, bracketing and crossing iteration limits
are preserved. Only command dispatch and completed-product integrity differ.
This is a library; recover_impacts.py owns provenance, locks and scope checks.
"""
import concurrent.futures as futures
import hashlib
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from TROTASR.utils.paths import internal_path
from TROTASR.utils.io import read_json as read, write_json as write, sha256 as sha
from TROTASR.workflows.diagnostic_common import require_success
ROOT_LOCK = threading.Lock()
TARGET = 0.5
NLL_TOL = 0.002
RUNNER = None


def tree_rows(path, name):
    import ROOT
    path = internal_path(path)
    with ROOT_LOCK:
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


def prepare(work, out, source):
    import ROOT
    setup = out / 'inputs.json'
    snapshot = out / 'workspace_bestfit_snapshot.root'
    toys = out / 'higgsCombine_profileAsimov.GenerateOnly.mH120.123456.root'
    provenance = dict(workspace=str(source), workspace_sha256=sha(source),
                      initial_fit_sha256=sha(work / 'multidimfit_initialFit_Test.root'))
    if setup.exists():
        old = read(setup)
        if any(old[k] != v for k, v in provenance.items()):
            raise ValueError('recovery input provenance changed')
        if sha(snapshot) != old['snapshot_sha256'] or sha(toys) != old['asimov_sha256']:
            raise ValueError('recovery cache changed')
        return old
    if snapshot.exists():
        raise ValueError('Unvalidated recovery snapshot retained; explicit recovery required')
    command = ['combine', '-M', 'GenerateOnly', '-d', str(source), '-m', '120',
               '-t', '-1', '--expectSignal', '1', '--saveToys', '-n', '_profileAsimov',
               '--setParameterRanges', 'r=0,20']
    require_success(RUNNER.run(command, out, out / 'generate_asimov.log', [toys]))
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
    provenance.update(snapshot=str(snapshot), snapshot_sha256=sha(snapshot),
                      asimov=str(toys), asimov_sha256=sha(toys),
                      generation_command=command, bestfit=bestfit,
                      physical_model_unchanged=True, observed_sr_used=False)
    write(setup, provenance)
    return provenance


class Recovery:
    def __init__(self, work, out, inputs, workers, refined=()):
        self.work, self.out, self.inputs = work, out, inputs
        self.refined = set(refined)
        self.pool = futures.ThreadPoolExecutor(max_workers=workers)
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
        if state.exists() and read(state).get('status') == 'complete':
            old = read(state)
            if sha(old['root']) == old['sha256']:
                return old
        attempts = read(state).get('attempts', []) if state.exists() else []
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
            write(state, dict(status='running', name=name, value=value, command=cmd,
                              started=started, attempts=attempts))
            record = RUNNER.run(cmd, folder, folder / (configuration + '.log'), [root])
            run = SimpleNamespace(returncode=0 if record['status'] == 'complete' else (record['exit_code'] or 1))
            error = None
            try:
                rows = tree_rows(root, name)
                if run.returncode != 0 or len(rows) != 2:
                    raise ValueError('fixed point did not commit a successful conditional fit')
                row = rows[-1]
                if not all(math.isfinite(x) for x in row.values()):
                    raise ValueError('nonfinite conditional fit')
                if not math.isclose(row['theta'], value, rel_tol=2e-6, abs_tol=2e-7):
                    raise ValueError('fixed parameter differs from requested point')
                if row['deltaNLL'] < -0.005:
                    raise ValueError('conditional fit is below common minimum; investigate before collecting')
                result = dict(status='complete', name=name, value=value, **row,
                              root=str(root), sha256=sha(root), strategy=strategy,
                              configuration=configuration,
                              seconds=time.time()-started, command=cmd, exit_code=run.returncode,
                              attempts=attempts)
                write(state, result)
                return result
            except Exception as exc:
                error = str(exc)
            attempts.append(dict(strategy=strategy, configuration=configuration, exit_code=run.returncode,
                                 seconds=time.time()-started, error=error))
        write(state, dict(status='failed', name=name, value=value, attempts=attempts))
        raise RuntimeError(name + ' at ' + str(value) + ': ' + error)

    def crossing(self, name, inner, outer):
        if not inner['deltaNLL'] < TARGET or not outer['deltaNLL'] >= TARGET:
            raise ValueError('crossing is not bracketed')
        points = [inner, outer]
        for _ in range(18):
            if abs(outer['deltaNLL'] - TARGET) <= NLL_TOL:
                return outer, points
            fraction = (TARGET-inner['deltaNLL'])/(outer['deltaNLL']-inner['deltaNLL'])
            fraction = min(0.8, max(0.2, fraction))
            value = inner['theta'] + fraction*(outer['theta']-inner['theta'])
            point = self.submit(name, value).result()
            points.append(point)
            if abs(point['deltaNLL']-TARGET) <= NLL_TOL:
                return point, points
            if point['deltaNLL'] < TARGET:
                inner = point
            else:
                outer = point
        raise RuntimeError('profile crossing did not converge: ' + name)

    def recover(self, name):
        saved = self.out / 'parameters' / (name + '.json')
        if saved.exists() and read(saved).get('status') == 'complete':
            result = read(saved)
            for point in result['points']:
                if sha(point['root']) != point['sha256']:
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
                oldpoints = [read(p) for p in (self.out/'points'/name).glob('*/state.json')]
                candidates = [p for p in oldpoints if p['status']=='complete'
                              and p['theta']>center['theta'] and p['deltaNLL']>=0]
                if not candidates:
                    raise ValueError('refinement requires an existing upper-side profile seed')
                seed = min(candidates,key=lambda p:abs(p['deltaNLL']-TARGET))['theta']
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
                if point['deltaNLL'] >= TARGET:
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
                if point['deltaNLL'] >= TARGET:
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
                    if point['deltaNLL'] >= TARGET:
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
                          points=points, target_deltaNLL=TARGET, crossing_tolerance=NLL_TOL)
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
                if point['deltaNLL'] >= TARGET or abs(point['deltaNLL']-TARGET) <= NLL_TOL:
                    break
                if guess == boundary:
                    raise ValueError('missing endpoint has no 1-sigma crossing within original bounds')
                inner, distance = point, distance*1.5
            else:
                raise ValueError('missing endpoint could not be bracketed in eight steps')
            if abs(point['deltaNLL']-TARGET) > NLL_TOL:
                point, extra = self.crossing(name, inner, point)
                points.extend(extra)
            result = dict(name=name, method='missing_endpoint_profile', status='complete',
                          nominal=oldrows[0], lower=point if side < 0 else existing,
                          upper=point if side > 0 else existing, points=points,
                          boundary_limited_lower=False, boundary_limited_upper=False,
                          original_entries=oldrows,
                          existing_endpoint_preserved=True, target_deltaNLL=TARGET,
                          crossing_tolerance=NLL_TOL)
        if not result['lower']['theta'] < result['nominal']['theta'] < result['upper']['theta']:
            raise ValueError('recovered endpoints do not bracket nominal')
        write(self.out / 'parameters' / (name + '.json'), result)
        return result


def collect(work, out, recovered, source):
    from HiggsAnalysis.CombinedLimit.tool_base import utils
    records = {p.stem: read(p) for p in (work / 'fit_state').glob('*.json') if p.stem != 'r'}
    prefit = utils.prefit_from_workspace(str(source), 'w', list(records), None)
    initial = read(work / 'fit_state/r.json')['validation']['entries']
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
            if sha(root) != original['validation']['sha256']:
                raise ValueError('successful output changed: ' + name)
            rows = original['validation']['entries']
            parameter.update(fit=[rows[i][1] for i in (1,0,2)], r=[rows[i][0] for i in (1,0,2)])
        parameter['impact_r'] = max(abs(parameter['r'][i]-parameter['r'][1]) for i in (0,2))
        data['params'].append(parameter)
    expected_count = read(work / 'impact_status.json')['nuisance_count']
    if len(data['params']) != expected_count:
        raise ValueError('unexpected parameter count')
    data['recovery_metadata'] = dict(reused_parameters=len(records)-len(recovered), recovered_parameters=len(recovered),
        boundary_limited_parameters=[n for n,v in recovered.items() if v['boundary_limited_lower'] or v['boundary_limited_upper']],
        note='Boundary-limited rateParam endpoints are physical boundaries, not two-sided 1-sigma crossings.')
    target = out / 'impacts_mStop1200_mLSP500.json'
    write(target, data)
    return target
