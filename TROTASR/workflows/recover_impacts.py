"""Finite recovery for explicitly recorded failed current impact endpoints.

No new fit ranges/priors/physical samples. Supported [0,10] Sgamma/QCD rate
parameters use the original direct-profile algorithm; a constrained missing
endpoint must be named explicitly after inspecting its actual failed ROOT.
The explicit --bounded-sgamma mode uses the adopted [0.01,5] workspace
bounds and neighbour-seeded fixed profiles, without changing the legacy mode.
"""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import hashlib
import os
import re
import struct
import sys
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import grid_card, freeze_contract, CommandRunner, process_active
from TROTASR.workflows import impact_recovery_legacy as legacy


def publish_completed_profiles(work, output, shard_statuses, cached_collection=None):
    """Collect terminal accepted profiles, without replaying the fit preflight.

    Producer-source changes cannot invalidate a completed fit's receipt. This
    stage launches no fit and does not rehash/reopen accepted fit ROOT files.
    Endpoint extraction and impact definitions stay in the existing collector.
    """
    artifact_stat = legacy.artifact_stat
    work, output = internal_path(work), internal_path(output)
    statuses = [internal_path(p) for p in shard_statuses]
    # status.json is in a finite phase below its independently locked shard.
    with ExitStack() as stack:
        for folder in [work, *dict.fromkeys(p.parent.parent for p in statuses)]:
            lock = stack.enter_context((folder/'impact.lock').open('r'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent = read_json(work/'impact_status.json')
        if (process_active(parent, 'controller_pid') is not None or parent.get('sr_data_used') is not False
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
        target = legacy.collect(work,output,recovered,Path(parent['workspace']),
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
                or not -.005 <= point['deltaNLL'] < legacy.TARGET):
            raise ValueError('Not a validated approved-boundary endpoint')
    if (abs(first['deltaNLL']-second['deltaNLL']) > legacy.NLL_TOL
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
        from TROTASR.workflows.profile_sgamma import tracked_start
        lo, hi = self.inputs['parameter_bounds'][name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Conditional point outside approved range')
        if self.finish and label == 'lower_independent':
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
            if self.inputs.get('refine_boundaries') and abs(lower['deltaNLL']-check['deltaNLL'])>legacy.NLL_TOL:
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
        if self.finish and not common_matches(returned):
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
            target_deltaNLL=legacy.TARGET,crossing_tolerance=legacy.NLL_TOL,
            numerical_diagnostics=diagnostics,r_agreement_is_diagnostic=likelihood)
        for p in result['points']: self.validate_point(p)
        write_json(target,result)
        return result


def bounded_preflight(manifest, work, model='T2tt', mass='mStop1200_mLSP500', complete=False):
    from TROTASR.workflows.impact_legacy import validate_fit
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
        if complete:
            attempts = [a for a in records[n]['attempts'] if a.get('exit_code') == 0]
            rows = [legacy.tree_rows(ROOT/a['folder']/('higgsCombine_paramFit_Test_'+n+'.MultiDimFit.mH120.root'),n)
                    for a in attempts]
            seeds[n] = saved_sgamma_seeds(rows, bounds[n])
        else:
            rows = legacy.tree_rows(work/('higgsCombine_paramFit_Test_'+n+'.MultiDimFit.mH120.root'),n)
            legacy.missing_endpoint_rows(rows,rows[0]['theta'],bounds[n])
    return dict(origins=origins,names=names,bounds=bounds,protected=protected,
        endpoint_seeds=seeds,
        parent=parent,original_status_sha256=sha256(work/'impact_status.json'),
        original_valid=len(records)-len(names),nuisance_count=len(records))


def run_bounded(manifest, work, workers, preflight_only=False, resume=False, finish=False,
                model='T2tt', mass='mStop1200_mLSP500', complete=False, polish=False, hybrid=False, likelihood=False, refine=False, final=False):
    if not 1 <= workers <= 98: raise ValueError('Reserve controller and diagnostic slots')
    manifest,work = internal_path(manifest),internal_path(work)
    with (work/'impact.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit = bounded_preflight(manifest,work,model,mass,complete)
        previous = work/('bounded_sgamma_resume_20260930' if finish else 'bounded_sgamma_20260930') if resume or finish else None
        if polish:previous=work/'bounded_sgamma_complete'
        if hybrid:previous=work/'bounded_sgamma_precision'
        if likelihood:previous=work/'bounded_sgamma_minimizer'
        if refine:previous=work/'bounded_sgamma_likelihood'
        if final:previous=work/'bounded_sgamma_refined'
        earlier = work/'bounded_sgamma_20260930' if finish else None
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
        out = work/('bounded_sgamma_final' if final else 'bounded_sgamma_refined' if refine else 'bounded_sgamma_likelihood' if likelihood else 'bounded_sgamma_minimizer' if hybrid else 'bounded_sgamma_precision' if polish else 'bounded_sgamma_complete' if complete else 'bounded_sgamma_finish_20260930' if finish else 'bounded_sgamma_resume_20260930' if resume else 'bounded_sgamma_20260930');out.mkdir(exist_ok=True)
        files = ['workflows/recover_impacts.py','workflows/impact_recovery_legacy.py',
                 'workflows/profile_sgamma.py','workflows/profile_sgamma.py',
                 'workflows/diagnostic_common.py','workflows/impact_legacy.py']
        contract = freeze_contract(out,audit['origins'],files,dict(parameters=audit['names'],
            bounds=audit['bounds'],original_status_sha256=audit['original_status_sha256'],
            protected=audit['protected'],expect_signal=1,r_range=[0,20],target=.5,
            crossing_tolerance=.002,maximum_conditional_configurations=2,
            lower_chain_fractions=[.75,.5,.25,0.],upper_steps=12,crossing_steps=18,
            max_calls=2000000,tolerance=.0001 if hybrid else .00001 if polish else .001 if resume or finish or complete else .0001,
            precision=1e-8 if hybrid else 1e-10 if polish else None,algorithm='Combined' if hybrid else 'Migrad',
            analytic_stats=hybrid,strategies=[0,1],workers=workers,
            configuration_tolerances=[.01,.1] if likelihood else None,
            independent_r_agreement_diagnostic_only=likelihood,
            boundary_refinement=refine,refinement_neighbour_fraction=.1 if refine else None,
            final_boundary_strategy=2 if final else None,
            complete_recovery=complete,endpoint_seeds=audit['endpoint_seeds'],
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
            inputs.update(complete_recovery=complete,endpoint_seeds=audit['endpoint_seeds'])
            if polish:inputs['polish_parameters']=prior['failed_parameters']
            if hybrid:inputs['hybrid_parameters']=prior['failed_parameters']
            if likelihood:inputs['likelihood_parameters']=prior['failed_parameters']
            if refine:inputs['refine_boundaries']=True
            if final:inputs['final_boundary']=True
            if previous or complete:
                import ROOT as root
                source = root.TFile.Open(inputs['snapshot'])
                try:
                    ws = source.Get('w')
                    inputs['all_parameter_bounds'] = {n:[float(ws.var(n).getMin()),float(ws.var(n).getMax())]
                        for n in inputs['bestfit']}
                finally: source.Close()
            worker = BoundedSgamma(work,out,inputs,workers,previous,earlier,finish or complete)
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
            target = legacy.collect(work,out,recovered,Path(audit['parent']['workspace']))
            if complete:
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


def run(manifest, work, workers, missing=(), refined=()):
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
