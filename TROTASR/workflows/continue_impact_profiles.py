"""One finite bracket-search continuation after exhausted direct profiles.

The original likelihood, fixed-point configurations, crossing solver and
acceptance thresholds are unchanged. Reuse verified conditional fits. Do not
require an unneeded remote boundary to converge before solving an interior
crossing, or mistake float serialization of the center for an upper seed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from TROTASR.workflows.diagnostic_common import native_runtime, grid_card, freeze_contract, CommandRunner
from TROTASR.workflows import impact_recovery_legacy as legacy

MAX_BRACKET_STEPS = 12


def distinct(a, b):
    return not math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7)


def endpoint(center, points, side, bounds, evaluate, crossing):
    """Use the nearest valid bracket; a failed boundary is NEVER an endpoint."""
    edge = bounds[0 if side < 0 else 1]
    candidates = sorted((p for p in points if distinct(p['theta'], center['theta'])
                         and side * (p['theta'] - center['theta']) > 0),
                        key=lambda p: abs(p['theta'] - center['theta']))
    inner = center
    for outer in candidates:
        if abs(outer['deltaNLL'] - legacy.TARGET) <= legacy.NLL_TOL:
            return outer, False
        if outer['deltaNLL'] >= legacy.TARGET:
            return crossing(inner, outer), False
        if outer['theta'] == edge:
            return outer, True
        inner = outer
    # If no genuine upper seed survives serialization, use a finite distance
    # inferred from the nearest supported opposite-side sample, not epsilon.
    opposite = [abs(p['theta']-center['theta']) for p in points
                if distinct(p['theta'], center['theta'])
                and side*(p['theta']-center['theta']) < 0]
    distance = min(opposite) if opposite else .25
    distance = max(.01, min(.25, distance))
    for _ in range(MAX_BRACKET_STEPS):
        if distinct(inner['theta'], center['theta']):
            distance = 1.5 * abs(inner['theta'] - center['theta'])
        value = center['theta'] + side * distance
        # Probe the interior when the next coarse step would hit the edge.
        # A validated edge from the original run was already considered above.
        if side * (value - edge) >= 0:
            value = .5 * (inner['theta'] + edge)
        if not distinct(value, inner['theta']):
            raise ValueError('Finite bracket search reached serialization resolution')
        point = evaluate(value)  # Failure persists; never convert it to a bound.
        points.append(point)
        if abs(point['deltaNLL'] - legacy.TARGET) <= legacy.NLL_TOL:
            return point, False
        if point['deltaNLL'] >= legacy.TARGET:
            return crossing(inner, point), False
        inner = point
    raise ValueError('No validated crossing/boundary in twelve bracket steps')


def validate_point(point, name):
    if point.get('status') != 'complete' or point.get('name') != name:
        raise ValueError('Not an accepted conditional point')
    if sha256(point['root']) != point['sha256']:
        raise ValueError('Conditional ROOT changed')
    rows = legacy.tree_rows(point['root'], name)
    if len(rows) != 2 or not all(math.isfinite(v) for r in rows for v in r.values()):
        raise ValueError('Conditional fit did not commit two finite rows')
    for key in ('theta', 'r', 'deltaNLL'):
        if rows[-1][key] != point[key]:
            raise ValueError('Conditional receipt differs from ROOT')
    if (distinct(point['theta'], point['value']) or point['deltaNLL'] < -.005
            or not 0 <= point['theta'] <= 10):
        raise ValueError('Invalid conditional point')
    return point


class BracketSearch(legacy.Recovery):
    def __init__(self, work, out, inputs, workers, names, previous):
        super().__init__(work, out, inputs, workers, names)
        self.previous = previous

    def recover(self, name):
        target = self.out / 'parameters' / (name + '.json')
        if target.exists():
            result = read_json(target)
            for p in result['points']:
                validate_point(p, name)
            return result
        points = []
        # Prefer higher precision at duplicate requested coordinates.
        by_value = {}
        for folder in ('points', 'points_refined'):
            for path in sorted((self.previous / folder / name).glob('*/state.json')):
                point = read_json(path)
                if point['status'] == 'complete':
                    validate_point(point, name)
                    by_value[format(point['value'], '.12g')] = point
        points.extend(by_value.values())
        theta = self.inputs['bestfit'][name]
        central = [p for p in points if not distinct(p['theta'], theta)]
        if not central:
            raise ValueError('No validated common central conditional fit')
        central = min(central, key=lambda p: abs(p['deltaNLL']))
        if abs(central['deltaNLL']) > .005:
            raise ValueError('Common minimum not reproduced')
        # Use the common best-fit theta for geometry, not Float_t roundoff.
        center = dict(central, theta=theta)
        def evaluate(value):
            point = self.submit(name, value).result()
            return validate_point(point, name)
        def crossing(inner, outer):
            result, trace = self.crossing(name, inner, outer)
            for point in trace:
                if point is not center:
                    validate_point(point, name)
                    points.append(point)
            return result
        lower, lb = endpoint(center, points, -1, [0., 10.], evaluate, crossing)
        upper, ub = endpoint(center, points, 1, [0., 10.], evaluate, crossing)
        if not lower['theta'] < theta < upper['theta']:
            raise ValueError('Endpoints do not bracket common minimum')
        nominal = dict(theta=theta, r=self.inputs['bestfit']['r'], deltaNLL=0.)
        result = dict(status='complete', name=name, method='direct_profile_bounded_bracket',
            nominal=nominal, lower=lower, upper=upper, points=points,
            boundary_limited_lower=lb, boundary_limited_upper=ub,
            target_deltaNLL=legacy.TARGET, crossing_tolerance=legacy.NLL_TOL,
            max_bracket_steps_per_side=MAX_BRACKET_STEPS)
        write_json(target, result)
        return result


def run(manifest, work, workers):
    if not 1 <= workers <= 47:
        raise ValueError('Bounded worker pool required')
    work = internal_path(work)
    _, origins = grid_card(manifest, 'impact')
    prior = work / 'direct_profile'
    out = work / 'profile_search'
    out.mkdir(exist_ok=True)
    with (work / 'impact.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = read_json(work / 'impact_status.json')
        previous = read_json(prior / 'status.json')
        names = sorted(previous['failed'])
        if (previous['status'] != 'needs_attention' or previous['controller_pid']
                or not previous['refined_parameters'] or not names
                or not set(names).issubset(previous['refined_parameters'])):
            raise ValueError('Requires terminal, exhausted finite precision profiles')
        if (parent['status'] != 'needs_attention'
                or set(previous['completed']) & set(names)
                or set(previous['completed']) | set(names) != set(parent['failed_nuisances'])
                or previous['original_successes'] + len(previous['completed']) + len(names)
                   != parent['nuisance_count']):
            raise ValueError('Prior parameter accounting is inconsistent')
        if any(not ('qcd_norm' in n or 'sgamma_shape' in n) for n in names):
            raise ValueError('Unsupported parameter class')
        for contract in (read_json(work / 'contract.json'), read_json(prior / 'contract.json')):
            if contract['origins'] != origins:
                raise ValueError('Different impact origins')
            for name, digest in contract['code'].items():
                if sha256(ROOT / name) != digest:
                    raise ValueError('Prior fit code changed')
        if sha256(parent['workspace']) != parent['workspace_sha256']:
            raise ValueError('Workspace changed')
        receipt = out / 'status.json'
        if receipt.exists():
            raise ValueError('Bracket continuation already attempted; inspect, do not reset')
        freeze_contract(out, origins, ['workflows/continue_impact_profiles.py',
            'workflows/impact_recovery_legacy.py', 'workflows/diagnostic_common.py'],
            dict(parameters=names, previous_status_sha256=sha256(prior/'status.json'),
                 r_range=[0,20], rate_range=[0,10], target=.5, crossing_tolerance=.002,
                 bracket_steps=12, crossing_steps=18, numerical_configs=['precise_s0','precise_s1']))
        import ROOT as root
        root.gROOT.SetBatch(True)
        root.EnableThreadSafety()
        source = root.TFile.Open(parent['workspace'])
        try:
            for name in names:
                var = source.Get('w').var(name)
                if not var or [var.getMin(),var.getMax()] != [0.,10.]:
                    raise ValueError('Physical parameter bounds changed')
        finally:
            source.Close()
        inputs = legacy.prepare(work, prior, Path(parent['workspace']))
        legacy.RUNNER = CommandRunner(out)
        worker = BracketSearch(work, out, inputs, workers, names, prior)
        state = dict(status='running', controller_pid=os.getpid(), started=time.time(),
            parameters=names, completed=[], failed={}, observed_sr_used=False,
            physical_model_unchanged=True, full_workflow_complete=False)
        write_json(receipt, state)
        try:
            recovered = {}
            for name in previous['completed']:
                item = read_json(prior/'parameters'/(name+'.json'))
                for p in item['points']:
                    validate_point(p, name)
                recovered[name] = item
            with ThreadPoolExecutor(max_workers=min(workers, len(names))) as pool:
                tasks = {pool.submit(worker.recover, n): n for n in names}
                for task in as_completed(tasks):
                    name = tasks[task]
                    try:
                        recovered[name] = task.result()
                        state['completed'].append(name)
                    except Exception as error:
                        state['failed'][name] = repr(error)
                    write_json(receipt, state)
            if state['failed']:
                state['status'] = 'needs_attention'
                return 2
            target = legacy.collect(work, out, recovered, Path(parent['workspace']))
            payload = read_json(target)
            state.update(status='fits_complete', total_valid=parent['nuisance_count'],
                json=str(target.relative_to(ROOT)), json_sha256=sha256(target),
                boundary_limited_parameters=payload['recovery_metadata']['boundary_limited_parameters'])
            parent.update(status='fits_complete', phase='collected_with_bounded_bracket',
                json=state['json'], json_sha256=state['json_sha256'], failed_nuisances=[],
                valid_nuisances=parent['nuisance_count'], profile_search_status=str(receipt.relative_to(ROOT)),
                boundary_limited_parameters=state['boundary_limited_parameters'],
                fit_products={str(p.relative_to(ROOT)):sha256(p) for p in work.rglob('*.root')},
                plots_status='pending_original_plotImpacts')
            write_json(work/'impact_status.json', parent)
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error))
            raise
        finally:
            worker.pool.shutdown()
            state.update(controller_pid=None, finished=time.time())
            write_json(receipt, state)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--workers', type=int, default=47)
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    native_runtime(a.work)
    raise SystemExit(run(a.manifest, a.work, a.workers))
