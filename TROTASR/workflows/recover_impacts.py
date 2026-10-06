"""Finite recovery for explicitly recorded failed current impact endpoints.

No new fit ranges/priors/physical samples. Supported [0,10] Sgamma/QCD rate
parameters use the original direct-profile algorithm; a constrained missing
endpoint must be named explicitly after inspecting its actual failed ROOT.
The explicit --bounded-sgamma mode uses the adopted [0.01,5] workspace
bounds and neighbour-seeded fixed profiles, without changing the legacy mode.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import hashlib
import os
import re
import struct
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import native_runtime, grid_card, freeze_contract, CommandRunner
from TROTASR.workflows import impact_recovery_legacy as legacy


def bounded_sgamma_scope(parent, records, bounds):
    names = recovery_scope(parent, records, (), (), {})
    if any(not re.fullmatch(r'CMS_NPS26012_sgamma_shape_lowdm_Nb(?:1|2plus)_bin\d+_202[45]', n)
           or bounds.get(n) != [.01, 5.] for n in names):
        raise ValueError('Requires failed low-dM Sgamma with approved [0.01,5] bounds')
    return names


def bounded_lower_values(center, bounds):
    lo, hi = bounds
    if bounds != [.01, 5.] or not lo < center < hi:
        raise ValueError('Invalid adopted Sgamma center/bounds')
    return [lo + f*(center-lo) for f in (.75, .5, .25, 0.)]


def bounded_boundary_pair(first, second, edge):
    for point in (first, second):
        if (point['status'] != 'complete' or not math.isclose(point['theta'], edge, abs_tol=2e-7)
                or not all(math.isfinite(point[k]) for k in ('theta','r','deltaNLL'))
                or not -.005 <= point['deltaNLL'] < legacy.TARGET):
            raise ValueError('Not a validated approved-boundary endpoint')
    if (abs(first['deltaNLL']-second['deltaNLL']) > legacy.NLL_TOL
            or abs(first['r']-second['r']) > .0002):
        raise ValueError('Independent boundary minima disagree')


def reconcile_seed_bounds(seeds, bounds):
    """Restore only the exact binary32 image of an approved endpoint.

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


class BoundedSgamma(legacy.Recovery):
    """Current-model profile geometry; reuse fixed commands and crossing math."""
    def __init__(self, work, out, inputs, workers, previous=None, earlier=None, finish=False):
        super().__init__(work, out, inputs, workers)
        self.points = {}
        self.previous = previous
        self.prior_roots = [p for p in (previous, earlier) if p]
        self.finish = finish

    def fixed(self, name, value, label=None, seed=None, strategies=(0, 1)):
        from TROTASR.workflows.profile_sgamma import fixed_command
        from TROTASR.workflows.continue_sgamma_profiles import tracked_start
        lo, hi = self.inputs['parameter_bounds'][name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Conditional point outside approved range')
        if self.finish and label == 'lower_independent':
            old = next((read_json(p/'points'/name/label/'state.json') for p in self.prior_roots
                        if (p/'points'/name/label/'state.json').exists()), None)
            if old and old['status'] == 'failed':
                # Independent strategy1 from the measured halfway point, not
                # the successful boundary fit or a repeated failed cold start.
                theta = self.inputs['bestfit'][name]
                seed = self.fixed(name, lo+.5*(theta-lo))
                label = 'lower_independent_neighbour'
        key = label or hashlib.sha256(format(value, '.15g').encode()).hexdigest()[:16]
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
        if starts is not None and self.previous:
            starts = reconcile_seed_bounds(starts, self.inputs['all_parameter_bounds'])
            if self.finish and label and label.startswith('return_neighbour_'):
                # Release starting values pinned to another parameter's bound;
                # only initialization changes, all other nuisances still float.
                starts = {n:.5*v+.5*self.inputs['bestfit'][n] for n,v in starts.items()}
        folder.mkdir(parents=True, exist_ok=True)
        attempts = []
        for strategy in strategies:
            tag = '_bounded_'+key+'_s'+str(strategy)
            product = folder/('higgsCombine'+tag+'.MultiDimFit.mH120.root')
            extra = ['--cminDefaultMinimizerStrategy', str(strategy)]
            if self.previous:
                extra += ['--cminDefaultMinimizerTolerance', '0.001']
            command = fixed_command(self.inputs, name, value, tag, extra=extra, seeds=starts)
            record = legacy.RUNNER.run(command, folder, folder/('s'+str(strategy)+'.log'), [product])
            try:
                if record['status'] != 'complete' or record['exit_code'] != 0:
                    raise ValueError('Conditional command did not complete')
                rows = legacy.tree_rows(product, name)
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
        rows = legacy.tree_rows(point['root'], point['name'])
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
        central = self.fixed(name, theta)
        if abs(central['deltaNLL']) > .005 or abs(central['r']-self.inputs['bestfit']['r']) > .0002:
            raise ValueError('Common minimum not reproduced')
        lower = None; lower_boundary = False; inner = central
        for value in bounded_lower_values(theta, bounds):
            outer = self.fixed(name, value, seed=inner)
            if abs(outer['deltaNLL']-legacy.TARGET) <= legacy.NLL_TOL:
                lower = outer; break
            if outer['deltaNLL'] >= legacy.TARGET:
                lower, _ = self.crossing(name, inner, outer); break
            inner = outer
        if lower is None:
            lower = inner
            # Independent direct seed and strategy: never infer a bound from
            # failed minimization or a missing crossing alone.
            check = self.fixed(name, bounds[0], 'lower_independent', central, (1,))
            bounded_boundary_pair(lower, check, bounds[0])
            lower_boundary = True
        returned = self.fixed(name, theta, 'return_to_common', lower)
        if self.finish and (abs(returned['deltaNLL']) > .005
                            or abs(returned['r']-self.inputs['bestfit']['r']) > .0002):
            returned = lower
            for index, fraction in enumerate((.25,.5,.75,1.)):
                returned = self.fixed(name, bounds[0]+fraction*(theta-bounds[0]),
                    'return_neighbour_'+str(index), returned)
        if abs(returned['deltaNLL']) > .005 or abs(returned['r']-self.inputs['bestfit']['r']) > .0002:
            raise ValueError('Return from lower endpoint does not reproduce common minimum')
        oldrows = legacy.tree_rows(self.work/('higgsCombine_paramFit_Test_'+name+'.MultiDimFit.mH120.root'),name)
        value = min(bounds[1], max(theta+.01, oldrows[-1]['theta']))
        inner = central; upper = None; upper_boundary = False
        for _ in range(12):
            outer = self.fixed(name, value, seed=inner)
            if abs(outer['deltaNLL']-legacy.TARGET) <= legacy.NLL_TOL:
                upper = outer; break
            if outer['deltaNLL'] >= legacy.TARGET:
                upper, _ = self.crossing(name, inner, outer); break
            if value == bounds[1]:
                check = self.fixed(name, bounds[1], 'upper_independent', central, (1,))
                bounded_boundary_pair(outer, check, bounds[1])
                upper, upper_boundary = outer, True; break
            inner = outer
            value = min(bounds[1], theta+1.5*(value-theta))
        if upper is None or not lower['theta'] < theta < upper['theta']:
            raise ValueError('Finite profile endpoints did not bracket the common minimum')
        result = dict(status='complete', name=name, method='bounded_sgamma_neighbour_profile',
            nominal=dict(theta=theta,r=self.inputs['bestfit']['r'],deltaNLL=0.),
            lower=lower,upper=upper,boundary_limited_lower=lower_boundary,
            boundary_limited_upper=upper_boundary,points=self.points[name],bounds=bounds,
            target_deltaNLL=legacy.TARGET,crossing_tolerance=legacy.NLL_TOL)
        for p in result['points']: self.validate_point(p)
        write_json(target,result)
        return result


def bounded_preflight(manifest, work):
    from TROTASR.workflows.impact_legacy import validate_fit
    import ROOT as root
    root.gROOT.SetBatch(True); root.EnableThreadSafety()
    _, origins = grid_card(manifest, 'impact')
    grid = read_json(manifest)
    if grid['scope'] != 'full_combined_systematic_production':
        raise ValueError('Bounded Sgamma requires the current combined model')
    parent = read_json(work/'impact_status.json')
    contract = read_json(work/'contract.json')
    if contract['origins'] != origins or parent.get('controller_pid'):
        raise ValueError('Impact provenance changed or primary still active')
    for name,digest in contract['code'].items():
        if sha256(ROOT/name) != digest: raise ValueError('Impact source changed: '+name)
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
    for n in names:
        rows = legacy.tree_rows(work/('higgsCombine_paramFit_Test_'+n+'.MultiDimFit.mH120.root'),n)
        legacy.missing_endpoint_rows(rows,rows[0]['theta'],bounds[n])
    return dict(origins=origins,names=names,bounds=bounds,protected=protected,
        parent=parent,original_status_sha256=sha256(work/'impact_status.json'),
        original_valid=len(records)-len(names),nuisance_count=len(records))


def run_bounded(manifest, work, workers, preflight_only=False, resume=False, finish=False):
    if not 1 <= workers <= 98: raise ValueError('Reserve controller and diagnostic slots')
    manifest,work = internal_path(manifest),internal_path(work)
    with (work/'impact.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit = bounded_preflight(manifest,work)
        previous = work/('bounded_sgamma_resume_20260930' if finish else 'bounded_sgamma_20260930') if resume or finish else None
        earlier = work/'bounded_sgamma_20260930' if finish else None
        prior = {}
        if previous:
            old_state = read_json(previous/'status.json')
            old_contract = read_json(previous/'contract.json')
            if (old_state['status'] != 'needs_attention' or old_state.get('controller_pid')
                    or not old_state.get('failed') or old_contract['origins'] != audit['origins']
                    or old_contract['options']['original_status_sha256'] != audit['original_status_sha256']):
                raise ValueError('Requires terminal, unpromoted original bounded recovery')
            backup = work.parent/'history/sgamma_recovery_20260930'/('source_before_finish' if finish else 'source_before_resume')/'recover_impacts.py'
            for name,digest in old_contract['code'].items():
                source = backup if name == 'workflows/recover_impacts.py' else ROOT/name
                if sha256(source) != digest: raise ValueError('Frozen recovery source changed: '+name)
            prior = dict(previous_contract_sha256=sha256(previous/'contract.json'),
                previous_status_sha256=sha256(previous/'status.json'),
                failed_parameters=sorted(old_state['failed']),
                reused_complete_parameters=sorted(old_state['completed']),
                seed_reconciliation='exact_binary32_image_of_approved_boundary_only')
            if finish:
                supported = {'CMS_NPS26012_sgamma_shape_lowdm_Nb1_bin1_'+y for y in ('2024','2025')}
                if not set(old_state['failed']).issubset(supported):
                    raise ValueError('Final neighbour pass only covers diagnosed Nb1 bin1 failures')
                prior.update(return_neighbour_fractions=[.25,.5,.75,1.],
                    independent_lower_seed_fraction=.5,maximum_neighbour_passes=1,
                    return_seed_common_minimum_mixing_fraction=.5)
                for name,digest in old_contract['options']['protected'].items():
                    if sha256(ROOT/name) != digest: raise ValueError('Prior protected output changed')
                    audit['protected'][name] = digest
            audit['protected'].update({str(p.relative_to(ROOT)):sha256(p) for p in previous.rglob('*') if p.is_file()})
        if preflight_only:
            print({k:v for k,v in audit.items() if k not in ('protected','parent')})
            print('Protected files:',len(audit['protected']))
            return 0
        out = work/('bounded_sgamma_finish_20260930' if finish else 'bounded_sgamma_resume_20260930' if resume else 'bounded_sgamma_20260930');out.mkdir(exist_ok=True)
        files = ['workflows/recover_impacts.py','workflows/impact_recovery_legacy.py',
                 'workflows/profile_sgamma.py','workflows/continue_sgamma_profiles.py',
                 'workflows/diagnostic_common.py','workflows/impact_legacy.py']
        contract = freeze_contract(out,audit['origins'],files,dict(parameters=audit['names'],
            bounds=audit['bounds'],original_status_sha256=audit['original_status_sha256'],
            protected=audit['protected'],expect_signal=1,r_range=[0,20],target=.5,
            crossing_tolerance=.002,maximum_conditional_configurations=2,
            lower_chain_fractions=[.75,.5,.25,0.],upper_steps=12,crossing_steps=18,
            max_calls=2000000,tolerance=.001 if resume or finish else .0001,strategies=[0,1],workers=workers,
            physical_model_unchanged=True,common_minimum_unchanged=True,**prior))
        if not (out/'original_impact_status.json').exists():
            write_json(out/'original_impact_status.json',audit['parent'])
        legacy.RUNNER = CommandRunner(out)
        state = dict(status='running',controller_pid=os.getpid(),started=time.time(),
            completed=[],failed={},original_successes=audit['original_valid'],
            parameters=audit['names'],workers=workers,observed_sr_used=False)
        write_json(out/'status.json',state)
        worker = None
        try:
            inputs = legacy.prepare(work,earlier or previous or out,Path(audit['parent']['workspace']))
            inputs['parameter_bounds'] = audit['bounds']
            if previous:
                import ROOT as root
                source = root.TFile.Open(inputs['snapshot'])
                try:
                    ws = source.Get('w')
                    inputs['all_parameter_bounds'] = {n:[float(ws.var(n).getMin()),float(ws.var(n).getMax())]
                        for n in inputs['bestfit']}
                finally: source.Close()
            worker = BoundedSgamma(work,out,inputs,workers,previous,earlier,finish)
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
            for name,digest in contract['code'].items():
                if sha256(ROOT/name) != digest: raise ValueError('Active recovery source changed: '+name)
            state['protected_files_unchanged'] = len(audit['protected'])
            if state['failed']:
                state['status']='needs_attention';return 2
            target = legacy.collect(work,out,recovered,Path(audit['parent']['workspace']))
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


def run(manifest, work, workers, missing=(), refined=()):
    if not 1 <= workers <= 99:
        raise ValueError('Reserve one controller slot')
    work = internal_path(work)
    _, origins = grid_card(manifest, 'impact')
    contract = read_json(work / 'contract.json')
    if contract['origins'] != origins:
        raise ValueError('Original impact origins changed')
    for name, digest in contract['code'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Original impact source changed')
    out = work / 'direct_profile'
    out.mkdir(exist_ok=True)
    with (work / 'impact.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / 'status.json'
        previous = read_json(state_path) if state_path.exists() else {}
        freeze_contract(out, origins, ['workflows/recover_impacts.py', 'workflows/impact_recovery_legacy.py',
            'workflows/diagnostic_common.py'], dict(expect_signal=1, r_range=[0, 20], delta_nll=.5,
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
                    rows = legacy.tree_rows(work / ('higgsCombine_paramFit_Test_' + name + '.MultiDimFit.mH120.root'), name)
                    legacy.missing_endpoint_rows(rows, rows[0]['theta'], bounds[name])
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
        legacy.RUNNER = CommandRunner(out)
        recovery = None
        try:
            inputs = legacy.prepare(work, out, Path(parent['workspace']))
            inputs['parameter_bounds'] = bounds
            recovery = legacy.Recovery(work, out, inputs, workers, refined)
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
            target = legacy.collect(work, out, recovered, Path(parent['workspace']))
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--workers', type=int, default=49)
    p.add_argument('--missing-endpoint', action='append', default=[])
    p.add_argument('--refine-parameter', action='append', default=[])
    p.add_argument('--bounded-sgamma', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume-bounded-sgamma', action='store_true')
    p.add_argument('--finish-bounded-sgamma', action='store_true')
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    native_runtime(a.work)
    if a.bounded_sgamma or a.resume_bounded_sgamma or a.finish_bounded_sgamma:
        if sum((a.bounded_sgamma,a.resume_bounded_sgamma,a.finish_bounded_sgamma)) != 1:
            p.error('Choose initial or finite resume, not both')
        if a.missing_endpoint or a.refine_parameter:
            p.error('Bounded current-model Sgamma is separate from legacy [0,10] recovery')
        raise SystemExit(run_bounded(a.manifest,a.work,a.workers,a.preflight_only,a.resume_bounded_sgamma,a.finish_bounded_sgamma))
    if a.preflight_only: p.error('--preflight-only requires --bounded-sgamma')
    raise SystemExit(run(a.manifest, a.work, a.workers, a.missing_endpoint, a.refine_parameter))
