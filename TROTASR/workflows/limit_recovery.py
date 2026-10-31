"""Internal finite numerical recovery for ready datacards/workspaces.

Extracted without numerical changes from the canonical recovery workflows.
No campaign admission, host runtime, scheduler, input-production validation or
CLI lives here. The caller owns point locks, finite-stage bookkeeping and
receipts. Every accepted new output still needs all five measured crossings.
"""
import math
import re
import statistics
import hashlib
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import sha256
from TROTASR.workflows.limit_validation import SCAN, QUANTILES, REL_ACC, ABS_ACC
from TROTASR.workflows.limit_serialization import validate_saved_limit as _validate_saved_limit

QUANTILE_ORDER = (.5, .025, .16, .84, .975)
MAX_RMAX = 10000.  # Existing measured exhausted-boundary ceiling, unchanged.
STAGE_LIMITS = dict(base=2, precision=4, measured=2, minima=3, crossings=2, branches=5)
ADVANCED_MAX_ATTEMPTS = 16

def fit_command(workspace, mass, rmax, tolerance):
    return ['combine','-M','AsymptoticLimits',str(workspace),'-m','120',
        '--run','blind','-n','_'+mass,
        '--cminDefaultMinimizerStrategy','0','--cminDefaultMinimizerTolerance',str(tolerance),
        '--cminFallbackAlgo','Minuit2,Migrad,1:0.1',
        '--minosAlgo','bisection','--picky','--rMax',str(rmax),
        '--rRelAcc',str(REL_ACC),'--rAbsAcc',str(ABS_ACC),'-v','2']


def target_nll(q):
    normal = statistics.NormalDist()
    return .5*(normal.inv_cdf(q)+normal.inv_cdf(1-q*.05))**2


def choose_range(logs, previous_rmax):
    """A search-range seed only, never an accepted/extrapolated limit.

    Require actual positive NLL samples near an exhausted scan boundary.
    A quadratic estimate to the largest expected-quantile target, with a factor
    two safety margin, proposes a finite new bracket. Only measured crossings
    in the subsequent Combine run can establish a limit.
    """
    if not math.isfinite(previous_rmax) or previous_rmax <= 0:
        raise ValueError('Invalid previous numerical range')
    seed = previous_rmax
    evidence = []
    for name, text, old_range in logs:
        if not math.isfinite(old_range) or old_range <= 0:
            raise ValueError('Invalid logged numerical range')
        blocks = text.split('Will search for NLL crossing by bisection')[1:]
        if len(blocks) > 5:
            continue
        # In a truncated search the first block is still unambiguously median.
        # This is the existing range-parser correction, not new quantile values.
        partial = 1 < len(blocks) < 5
        if partial:
            blocks = blocks[:1]
        for q, block in zip(QUANTILE_ORDER, blocks):
            scans = [(float(r),float(n)) for r,n in SCAN.findall(block)]
            scans = [(r,n) for r,n in scans if math.isfinite(r) and math.isfinite(n) and r>0 and n>0]
            if len(scans)<3 or any(n>=target_nll(q) for r,n in scans):
                continue
            r, nll = max(scans)
            if r < .9*old_range:
                continue
            proposal = max(2*old_range, 2*r*math.sqrt(target_nll(.975)/nll))
            seed = max(seed, proposal)
            evidence.append(dict(attempt=name, quantile=q, sampled_r=r, sampled_delta_nll=nll,
                                 previous_rmax=old_range, proposed_rmax=proposal))
            if partial:
                evidence[-1]['search_scope'] = 'median_only_from_incomplete_sequence'
    seed = math.ceil(seed/10.)*10.
    if seed > MAX_RMAX:
        raise ValueError('Required numerical seed exceeds finite10000 cap; inspect explicitly')
    return seed, evidence


def command_for(workspace, mass, spec):
    command = fit_command(workspace, mass, spec['rmax'], spec['tolerance'])
    command[command.index('--cminDefaultMinimizerStrategy')+1] = str(spec['strategy'])
    return command


def adaptive_seed(logs, ceiling=100000.):
    """Use measured profile samples to seed the next full expected-limit fit.

    No fitted limit (including a broken bisection return) enters this estimate.
    Log-log interpolation or a quadratic extrapolation is a proposal only.
    Prefer the attempt sampling nearest the required target, not the one with
    the strongest returned limit. If no profile evaluations exist,
    use the saved Asimov fit error solely as an explicitly unverified seed.
    """
    if ceiling != 100000.:
        raise ValueError('Adaptive ceiling must be the user-approved 100000')
    target = target_nll(.975)
    candidates = []
    for name, text in logs:
        rows = [(float(r), float(n)) for r, n in SCAN.findall(text)]
        rows = sorted(set((r, n) for r, n in rows
                          if math.isfinite(r) and math.isfinite(n) and r > 0 and n > 0))
        if not rows:
            continue
        below = [(r, n) for r, n in rows if n < target]
        above = [(r, n) for r, n in rows if n >= target]
        lo = max(below, key=lambda row: row[1]) if below else None
        hi = min(above, key=lambda row: row[1]) if above else None
        if lo and hi and lo[0] < hi[0] and lo[1] < hi[1]:
            fraction = math.log(target/lo[1])/math.log(hi[1]/lo[1])
            estimate = math.exp(math.log(lo[0])+fraction*math.log(hi[0]/lo[0]))
            evidence = dict(method='log_log_measured_bracket_seed', samples=[list(lo), list(hi)])
        else:
            r, n = min(rows, key=lambda row: abs(math.log(row[1]/target)))
            estimate = r*math.sqrt(target/n)
            evidence = dict(method='quadratic_measured_profile_seed', samples=[[r, n]])
        proposal = max(.0001, min(ceiling, 1.5*estimate))
        score = min(abs(math.log(n/target)) for r,n in rows)
        candidates.append((score, float(format(proposal, '.6g')), dict(evidence, source_log=name,
            target_delta_nll=target, safety_factor=1.5, unbounded_proposal=1.5*estimate,
            ceiling_reached=1.5*estimate >= ceiling, measured_crossing_verified=False)))
    if candidates:
        _, seed, evidence = min(candidates, key=lambda row: row[0])
        return seed, evidence
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    # Original history order avoids taking the huge error/step induced solely
    # by the later uniform rMax100000 run as a new measurement of sensitivity.
    for name, text in logs:
        block = text.split('Fit to asimov dataset:')
        if len(block) != 2:
            continue
        block = block[1].split('Will search for NLL crossing by bisection')[0]
        values = re.findall(r'^\s*r\s+('+number+r')\s+('+number+r')\s+\+/-\s+('+number+r')', block, re.M)
        if len(values) == 1:
            error = float(values[0][2])
            if math.isfinite(error) and error > 0:
                proposal = min(ceiling, max(.0001, 1.5*math.sqrt(2*target)*error))
                return float(format(proposal, '.6g')), dict(method='asimov_error_seed_no_profile_rows',
                    source_log=name, fit_error=error, measured_crossing_verified=False,
                    warning_is_fatal=False)
    raise ValueError('No measured NLL or finite Asimov fit error; explicit diagnosis required')


def _verify_bisection_attempt(point, attempt):
    """Validate a saved exact attempt, never the last/strongest fitted value."""
    if attempt.get('exit_code') != 0:
        return False, {'reason': 'nonzero_command_exit'}
    log = internal_path(ROOT/attempt['log'])
    product = log.parent/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
    if attempt.get('output') and internal_path(ROOT/attempt['output']) != product:
        raise ValueError('Attempt output/log mismatch')
    if attempt.get('log_sha256') != sha256(log):
        raise ValueError('Attempt log hash mismatch')
    if not product.is_file():
        return False, {'reason': 'missing_saved_output'}
    if attempt.get('output_sha256') != sha256(product):
        raise ValueError('Attempt ROOT hash mismatch')
    return _validate_saved_limit(product, log, attempt['tolerance'])


def profile_recovery_configs(seed, max_calls=None):
    """Finite numerical alternatives; never change or freeze model parameters."""
    if not math.isfinite(seed) or not .0001 <= seed <= 100000:
        raise ValueError('Invalid measured-NLL range')
    if max_calls is not None:
        if max_calls != 2000000:
            raise ValueError('Only the finite measured-call-limit correction is admitted')
        return [dict(name='migrad_s0_calls2m', strategy=0, tolerance=.1, rmax=seed,
                     algorithm='Migrad', precision=None, max_calls=max_calls),
                dict(name='combined_s1_calls2m', strategy=1, tolerance=.1, rmax=seed,
                     algorithm='Combined', precision=None, max_calls=max_calls)]
    return [dict(name='combined_s1', strategy=1, tolerance=.1, rmax=seed,
                 algorithm='Combined', precision=None),
            dict(name='migrad_precision8', strategy=1, tolerance=.1, rmax=seed,
                 algorithm='Migrad', precision=1e-8)]


def profile_recovery_command(workspace, mass, spec):
    command = command_for(workspace, mass, spec)
    command[command.index('-v')+1] = '3'
    command += ['--cminDefaultMinimizerAlgo', spec['algorithm']]
    if spec['precision'] is not None:
        command += ['--cminDefaultMinimizerPrecision', str(spec['precision'])]
    if spec.get('max_calls'):
        command += ['--X-rtd', 'MINIMIZER_MaxCalls='+str(spec['max_calls'])]
    if spec.get('explicit_stats'):
        command += ['--X-rtd', 'MINIMIZER_no_analytic']
    return command


def unfinished_configs(seed, already_tried_calls2m):
    profile_recovery_configs(seed, 2000000)  # Validate the common range ceiling.
    if not already_tried_calls2m:
        return profile_recovery_configs(seed, 2000000)
    # The four exhausted compressed fits stopped with large EDM/status3, not
    # an Asimov warning. Change finite-difference precision and strategy;
    # never reset or repeat the previous two command identities.
    return [dict(name='precision8_s%d_calls2m' % strategy, strategy=strategy,
                 tolerance=.1, rmax=seed, algorithm='Migrad',
                 precision=1e-8, max_calls=2000000) for strategy in (1, 2)]


def measured_start_commands(workspace, mass, primary, previous, logs, measured_envelope=None):
    """Choose a smaller initial bracket from actual near-median profile rows."""
    from TROTASR.workflows.limit_validation import SCAN
    if any(r.get('pid') or r.get('status') != 'failed' for r in (primary, previous)):
        raise ValueError('Only terminal failed points can use measured-start recovery')
    attempts = previous.get('attempts', [])
    if not attempts or any(a.get('passed') for a in attempts):
        raise ValueError('A completed or missing predecessor must not be refitted')
    old = attempts[-1]['command']
    if (old[old.index('--run')+1] != 'blind' or '--picky' not in old
            or any(f in old for f in ('--freezeParameters','--setParameters',
                                      '--setParameterRanges','--snapshotName','--noFitAsimov'))):
        raise ValueError('Changed likelihood or Asimov definition')
    if measured_envelope is not None:
        if measured_envelope.get('status') != 'available' or len(measured_envelope.get('brackets', [])) != 5:
            raise ValueError('Require five measured target brackets')
        evidence = dict(measured_envelope)
        seed = evidence['proposed_rmax']
    else:
        target = target_nll(.5)
        rows = []
        for name, text in logs:
            blocks = text.split('Will search for NLL crossing by bisection')[1:]
            if not blocks: continue
            for r, nll in SCAN.findall(blocks[0]):
                r, nll = float(r), float(nll)
                if math.isfinite(r) and math.isfinite(nll) and r > 0 and .5*target <= nll < target:
                    rows.append(dict(log=name, sampled_r=r, sampled_delta_nll=nll))
        if not rows: raise ValueError('No measured near-median starting region')
        evidence = min(rows, key=lambda row: abs(row['sampled_delta_nll']-target))
        seed = math.ceil(1.25*evidence['sampled_r']/10.)*10.
    if not 0 < seed < float(old[old.index('--rMax')+1]) <= 100000:
        raise ValueError('Measured start does not reduce the failed initial jump')
    specs = []
    for strategy, tolerance in ((0,.01),(1,.001)):
        cmd = fit_command(workspace, mass, seed, tolerance)
        cmd[cmd.index('--cminDefaultMinimizerStrategy')+1] = str(strategy)
        cmd[cmd.index('-v')+1] = '3'
        cmd += ['--X-rtd','MINIMIZER_MaxCalls=5000000']
        specs.append(dict(name='measured_start_s%d' % strategy, command=cmd,
            strategy=strategy, tolerance=tolerance, rmax=seed,
            range_evidence=evidence, seed_is_not_a_limit=True))
    return specs


def minimum_commands(previous):
    """Improve starting minima without tightening acceptance or fixing nuisances."""
    base=list(previous)
    if (base[base.index('--run')+1]!='blind' or '--picky' not in base
            or base[base.index('--minosAlgo')+1]!='bisection'):
        raise ValueError('Require the existing blind, checked bisection command')
    forbidden=('--freezeParameters','--setParameterRanges','--setParameters',
               '--noFitAsimov','--toysFrequentist','--snapshotName')
    if any(x in base for x in forbidden):raise ValueError('Changed statistical model or Asimov input')
    rmax=float(base[base.index('--rMax')+1])
    if not 0<rmax<=100000:raise ValueError('Invalid previous measured range')
    specs=[]
    for name,explicit,algo in [('analytic_s0_prefit',False,'Migrad'),
                                ('explicit_s0_prefit',True,'Migrad'),
                                ('explicit_combined_s0',True,'Combined')]:
        cmd=[];i=0
        while i<len(base):
            if base[i]=='--X-rtd' and (base[i+1]=='MINIMIZER_no_analytic'
                    or base[i+1].startswith('MINIMIZER_MaxCalls=')):
                i+=2;continue
            cmd.append(base[i]);i+=1
        for flag,value in [('--cminDefaultMinimizerStrategy','0'),
                ('--cminDefaultMinimizerTolerance','0.01'),
                ('--cminFallbackAlgo','Minuit2,Combined,0:0.01'),
                ('--cminDefaultMinimizerAlgo',algo),
                ('--cminDefaultMinimizerPrecision','1e-08'),
                ('--rRelAcc','0.0025'),('--rAbsAcc','1e-05'),('-v','3')]:
            if flag in cmd:cmd[cmd.index(flag)+1]=value
            else:cmd += [flag,value]
        cmd += ['--cminApproxPreFitTolerance','1.0','--cminApproxPreFitStrategy','0',
                '--X-rtd','MINIMIZER_MaxCalls=5000000']
        if explicit:cmd += ['--X-rtd','MINIMIZER_no_analytic']
        specs.append(dict(name=name,tolerance=.01,strategy=0,command=cmd,
                          explicit_stats=explicit,algorithm=algo,max_calls=5000000))
    return specs


def refinement_command(previous, tolerance, allow_analytic=False):
    """Resolve an unfinished crossing without changing its likelihood or range."""
    if tolerance not in (.001,.0001):raise ValueError('Unknown finite refinement')
    result=list(previous)
    for option,value in [('--run','blind'),('--minosAlgo','bisection')]:
        if result[result.index(option)+1]!=value:raise ValueError('Not a blind bisection fit')
    if '--picky' not in result or (not allow_analytic and 'MINIMIZER_no_analytic' not in result):
        raise ValueError('Expected explicit-statistical-nuisance fit')
    for option,value in [('--cminDefaultMinimizerStrategy','1'),
                         ('--cminDefaultMinimizerTolerance',str(tolerance)),
                         ('--cminFallbackAlgo','Minuit2,Migrad,2:'+str(tolerance)),
                         ('--rRelAcc','0.00025'),('--rAbsAcc','1e-06')]:
        result[result.index(option)+1]=value
    return result


def precision_commands(workspace, mass, logs, previous_commands):
    seed, evidence = adaptive_seed(logs)
    known = {tuple(command) for command in previous_commands}
    specs = []
    for explicit in (False, True):
        for strategy,tolerance in ((1,.01),(2,.001)):
            spec = dict(name=('explicit' if explicit else 'analytic')+'_precision8_s'+str(strategy),
                strategy=strategy,tolerance=tolerance,rmax=seed,algorithm='Migrad',
                precision=1e-8,max_calls=2000000,explicit_stats=explicit)
            command = profile_recovery_command(workspace,mass,spec)
            if tuple(command) in known:
                continue
            known.add(tuple(command));specs.append(dict(spec,command=command))
    return specs, evidence


def crossing_commands(attempts, previous_commands=()):
    """Use only newly finished, explicit-stat minima with five valid quantiles."""
    from TROTASR.workflows.limit_validation import QUANTILES
    for attempt in reversed(attempts):
        validation = attempt.get('validation', {})
        values = {float(q):v for q,v in validation.get('quantiles', {}).items()}
        crossings = validation.get('crossings', [])
        command = attempt.get('command', [])
        if (attempt.get('exit_code') != 0 or attempt.get('passed')
                or 'MINIMIZER_no_analytic' not in command
                or set(values) != set(QUANTILES) or len(crossings) != 5
                or {float(c['quantile']) for c in crossings} != set(QUANTILES)
                or not any(c.get('passed') for c in crossings)
                or all(c.get('passed') for c in crossings)
                or not all(math.isfinite(v) and v > 0 for v in values.values())
                or not all(values[a] < values[b] for a,b in zip(QUANTILES,QUANTILES[1:]))):
            continue
        specs = [dict(name='crossing_refined_s1_tol'+str(t),tolerance=t,
            strategy=1,command=refinement_command(command,t)) for t in (.001,.0001)]
        known={tuple(c) for c in previous_commands}
        specs=[s for s in specs if tuple(s['command']) not in known]
        if not specs:
            return [],dict(status='not_applicable',reason='both_exact_crossing_commands_already_attempted',maximum_new_attempts=0)
        return specs, dict(status='applicable',basis=attempt['name'],
            basis_log_sha256=attempt['log_sha256'],maximum_new_attempts=len(specs))
    return [], dict(status='not_applicable',
        reason='no_new_explicit_minimum_with_five_valid_quantiles_and_partial_crossing_failure',
        maximum_new_attempts=0)


def five_quantile_structure(evidence):
    from TROTASR.workflows.limit_validation import QUANTILES
    try:
        values = {float(q): float(v) for q, v in evidence.get('quantiles', {}).items()}
        crossings = evidence.get('crossings', [])
        return (set(values) == set(QUANTILES) and len(crossings) == 5
            and {float(c['quantile']) for c in crossings} == set(QUANTILES)
            and all(math.isfinite(v) and v > 0 for v in values.values())
            and all(values[a] < values[b] for a, b in zip(QUANTILES, QUANTILES[1:])))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def native_stepping_proposals(attempts, log_texts, previous_commands=()):
    """Finite existing numerical branches after failed minima/refinements.

    Upper-only branch: exactly two completed explicit refinements with five
    finite ordered quantiles and only +2sigma crossing failed -> one native
    profiled search. Median/range branch: the two actual explicit strategy-0
    minima failed their median crossing -> two measured range refinements;
    if both exit0/not-five-quantiles -> one native search; if that also fails
    -> two existing strategy-0 strict stepping searches. Never accept median
    only. These branches reuse numerical diagnoses, not historical folder IDs.

    Maximum added configurations is five (range2 + native1 + stepping2),
    or one for the upper-only branch. Exact commands already attempted anywhere
    in this point's protected history are not proposed again.
    """
    if any(a.get('passed') for a in attempts):
        raise ValueError('A verified fit must not enter another numerical branch')
    if any(a.get('exit_code') is None for a in attempts):
        raise ValueError('Native/stepping planning requires terminal command receipts')
    by_name = {a['name']: a for a in attempts}
    if len(by_name) != len(attempts):
        raise ValueError('Duplicate attempt identity in numerical continuation')
    known = {tuple(c) for c in previous_commands}
    known.update(tuple(a['command']) for a in attempts)

    def not_applicable(reason, **evidence):
        return [], dict(status='not_applicable', reason=reason,
                        maximum_new_attempts=0, **evidence)

    def proposed(specs, branch, basis):
        selected = [dict(s) for s in specs if tuple(s['command']) not in known]
        if not selected:
            return not_applicable('all_exact_branch_commands_already_attempted', branch=branch)
        return selected, dict(status='applicable', branch=branch,
            maximum_new_attempts=len(selected), basis=[dict(name=a['name'],
                log_sha256=a.get('log_sha256'), output_sha256=a.get('output_sha256')) for a in basis])

    refined_names = ('crossing_refined_s1_tol0.001', 'crossing_refined_s1_tol0.0001')
    refined = [by_name[n] for n in refined_names if n in by_name]
    upper_only = (len(refined) == 2 and all(a.get('exit_code') == 0
        and 'MINIMIZER_no_analytic' in a['command']
        and five_quantile_structure(a.get('validation', {}))
        and {float(c['quantile']) for c in a['validation']['crossings'] if not c['passed']} == {.975}
        for a in refined))
    native_name = 'native_bracket_s1_tol0.0001'
    if upper_only:
        if native_name in by_name:
            return not_applicable('upper_only_native_attempt_finished_no_median_range_diagnosis',
                                  branch='upper_only_native')
        command = refinement_command(refined[-1]['command'], .0001)
        command[command.index('--minosAlgo') + 1] = 'new'
        command[command.index('--rRelAcc') + 1] = '0.00001'
        spec = dict(name=native_name, tolerance=.0001, command=command,
                    run3_numerical_adapter='native_profiled')
        return proposed([spec], 'upper_only_native', refined)

    minima_names = ('explicit_s0_prefit', 'explicit_combined_s0')
    minima = [by_name[n] for n in minima_names if n in by_name]
    if len(minima) != 2:
        return not_applicable('two_explicit_strategy0_minimum_results_not_available')
    minimum_state = dict(status='failed', pid=None, attempts=minima)
    range_names = ('winner_measured_range_s1_tol0.001', 'winner_measured_range_s1_tol0.0001')
    ranges = [by_name[n] for n in range_names if n in by_name]
    if not ranges:
        logs = [(a['log'], log_texts[a['log']],
                 float(a['command'][a['command'].index('--rMax') + 1])) for a in minima]
        try:
            specs = winner_range_commands(minimum_state, logs)
        except ValueError as exc:
            if str(exc) not in ('Requires the diagnosed winner median range failure',
                               'No measured exhausted-range evidence within the approved ceiling'):
                raise
            return not_applicable(str(exc), branch='measured_median_range')
        return proposed(specs, 'measured_median_range', minima)
    if len(ranges) != 2:
        return not_applicable('both_measured_range_refinements_have_not_finished')
    range_state = dict(status='failed', pid=None, attempts=ranges)
    try:
        native_specs = winner_native_commands(range_state)
    except ValueError as exc:
        if str(exc) != 'Requires exhausted winner large-step minimization attempts':
            raise
        return not_applicable(str(exc), branch='median_native')
    if native_name not in by_name:
        native_specs = [dict(s, run3_numerical_adapter='native_profiled') for s in native_specs]
        return proposed(native_specs, 'median_native', ranges)
    stepping_names = ('stepping_explicit_s0_prefit', 'stepping_explicit_combined_s0')
    if any(name in by_name for name in stepping_names):
        return not_applicable('finite_strategy0_stepping_attempts_finished', branch='median_stepping')
    # The actual failed native result is retained; there is no synthetic success
    # or manufactured prior stage. Both earlier median/range diagnoses are
    # rechecked by the canonical command builder before stepping is proposed.
    specs = winner_stepping_commands(range_state, minimum_state)
    specs = [dict(s, run3_numerical_adapter='stepping_full5') for s in specs]
    return proposed(specs, 'median_stepping', [*minima, *ranges, by_name[native_name]])


def winner_range_commands(previous, logs):
    attempts=previous.get('attempts',[])
    if (previous.get('pid') or previous.get('status')!='failed' or len(attempts)!=2
            or [a['name'] for a in attempts]!=['explicit_s0_prefit','explicit_combined_s0']
            or any(a.get('passed') or a.get('exit_code')!=0 for a in attempts)
            or any(next((c['passed'] for c in a['validation'].get('crossings',[])
                         if c['quantile']==.5),True) for a in attempts)):
        raise ValueError('Requires the diagnosed winner median range failure')
    basis=attempts[0];old=float(basis['command'][basis['command'].index('--rMax')+1])
    seed,evidence=choose_range(logs,old)
    if not evidence or not old<seed<=100000:
        raise ValueError('No measured exhausted-range evidence within the approved ceiling')
    specs=[]
    for tolerance in (.001,.0001):
        cmd=refinement_command(basis['command'],tolerance)
        cmd[cmd.index('--rMax')+1]=str(seed)
        specs.append(dict(name='winner_measured_range_s1_tol'+str(tolerance),tolerance=tolerance,
            command=cmd,rmax=seed,range_evidence=evidence,range_policy='existing_choose_range',
            seed_is_not_a_limit=True))
    return specs


def winner_native_commands(previous):
    attempts=previous.get('attempts',[])
    if (previous.get('pid') or previous.get('status')!='failed' or len(attempts)!=2
            or [a['name'] for a in attempts]!=['winner_measured_range_s1_tol0.001','winner_measured_range_s1_tol0.0001']
            or any(a.get('passed') or a.get('exit_code')!=0
                   or a['validation'].get('reason')!='not five expected quantiles' for a in attempts)):
        raise ValueError('Requires exhausted winner large-step minimization attempts')
    cmd=refinement_command(attempts[-1]['command'],.0001)
    if not 0<float(cmd[cmd.index('--rMax')+1])<=100000:
        raise ValueError('Native seed beyond the approved range')
    cmd[cmd.index('--minosAlgo')+1]='new';cmd[cmd.index('--rRelAcc')+1]='0.00001'
    return [dict(name='native_bracket_s1_tol0.0001',tolerance=.0001,command=cmd)]


def winner_stepping_commands(previous, small_range):
    """Reuse the measured strategy-0 path with a finite, strict stepping range."""
    winner_native_commands(previous)  # Exact terminal large-step failure scope.
    old=small_range.get('attempts',[])
    if (small_range.get('pid') or small_range.get('status')!='failed' or len(old)!=2
            or [a['name'] for a in old]!=['explicit_s0_prefit','explicit_combined_s0']
            or any(a.get('passed') or a.get('exit_code')!=0 for a in old)
            or any(next((c['passed'] for c in a['validation'].get('crossings',[])
                         if c['quantile']==.5),True) for a in old)):
        raise ValueError('Requires both measured strategy-0 median failures')
    seed=float(previous['attempts'][0]['command'][previous['attempts'][0]['command'].index('--rMax')+1])
    if not 0<seed<=100000:raise ValueError('Stepping range exceeds the approved ceiling')
    specs=[]
    for a in old:
        cmd=list(a['command'])
        if ('MINIMIZER_no_analytic' not in cmd or '--picky' not in cmd
                or cmd[cmd.index('--run')+1]!='blind'
                or cmd[cmd.index('--cminDefaultMinimizerStrategy')+1]!='0'
                or float(cmd[cmd.index('--cminDefaultMinimizerTolerance')+1])!=.01
                or any(f in cmd for f in ('--freezeParameters','--setParameters','--snapshotName'))):
            raise ValueError('Unexpected original strategy-0 model or command')
        for flag,value in [('--rMax',str(seed)),('--minosAlgo','stepping'),
                           ('--rRelAcc','0.00025'),('--rAbsAcc','1e-06')]:
            cmd[cmd.index(flag)+1]=value
        if '--strictBounds' not in cmd:cmd.append('--strictBounds')
        specs.append(dict(name='stepping_'+a['name'],tolerance=.01,strategy=0,
                          command=cmd,rmax=seed,first_step=.05*seed))
    return specs


def native_profiled_text(text):
    """Map only genuine profiled native-'new' rows to the shared crossing checker.

    Unprofiled/approximate rows and the native logger's two bare bracket rows
    are excluded: its second bare row incorrectly repeats the lower r.
    The original execution log is never changed.
    """
    blocks=text.split('Will search for NLL crossing with new algorithm')[1:]
    if len(blocks)!=5:raise ValueError('Expected five native bracket searches')
    number=r'[-+\d.eE]+'
    pattern=re.compile(r'At r = ('+number+r'):\s*delta\(nll unprof\) = '+number+
                       r'\s+delta\(nll prof\) = ('+number+r')')
    output=[]
    for block in blocks:
        rows=pattern.findall(block)
        if not rows or any(not 0<=float(r)<=100000 for r,n in rows):
            raise ValueError('Missing/beyond-authorized native profiled NLL rows')
        output.append('Will search for NLL crossing by bisection\n'+
                      '\n'.join('At r = '+r+': delta(nll) = '+n for r,n in rows))
    return '\n'.join(output)


def verify_native_attempt(point, attempt):
    from TROTASR.workflows.limit_validation import crossing_check
    from TROTASR.workflows.limit_serialization import reconcile_crossings
    command=attempt['command']
    if (command[command.index('--minosAlgo')+1]!='new' or '--picky' not in command
            or command[command.index('--run')+1]!='blind' or 'MINIMIZER_no_analytic' not in command):
        raise ValueError('Requires the explicit native bracket-search command')
    passed,evidence=_verify_bisection_attempt(point,attempt)  # Exact ROOT/log identities and all five quantiles.
    if passed or evidence.get('reason')!='expected_five_crossing_searches' or evidence.get('count')!=0:
        return passed,evidence
    text=native_profiled_text((ROOT/attempt['log']).read_text(errors='replace'))
    values={float(q):v for q,v in evidence['quantiles'].items()}
    passed,check=crossing_check(text,values,attempt['tolerance'])
    evidence.pop('reason',None);evidence.pop('count',None);evidence.update(check)
    if not passed:passed,evidence=reconcile_crossings(text,evidence,attempt['tolerance'])
    evidence['validation_adapter']='native_new_actual_profiled_rows_same_crossing_thresholds'
    return passed,evidence


def stepping_median_evidence(text, value, tolerance, rmax):
    """Same measured-crossing/serialization policy, for the requested median only."""
    import math
    import statistics
    from TROTASR.workflows.limit_validation import SCAN, REL_ACC, ABS_ACC
    from TROTASR.workflows.limit_serialization import rounding_interval
    marker='Will search for NLL crossing by stepping'
    if text.count(marker)!=1 or not math.isfinite(value) or not 0<value<=rmax<=100000:
        raise ValueError('Missing finite authorized median stepping search')
    block=text.split(marker)[1].split('Will search for NLL crossing')[0]
    rows=[(float(r),float(n)) for r,n in SCAN.findall(block)]
    if not rows or any(not math.isfinite(n) or not 0<=r<=rmax for r,n in rows):
        raise ValueError('Invalid actual stepping profile rows')
    target=.5*statistics.NormalDist().inv_cdf(.975)**2
    low,high=rounding_interval(value)
    close=[(r,n) for r,n in rows if low<=r<=high and abs(n-target)<=.05*tolerance+.00001]
    below=[r for r,n in rows if n<=target and r<=value+.000002]
    above=[r for r,n in rows if n>=target and r>=value-.000002]
    width=min(above)-max(below) if below and above else None
    passed=bool(close or (width is not None and 0<=width<=2*(max(REL_ACC*value,ABS_ACC)+.000002)))
    return dict(quantile=.5,result=value,target_delta_nll=target,scan_evaluations=len(rows),
        bracket_width=width,passed=passed,matching_scan_rows=close,
        serialization_interval=[low,high],reconciliation='binary32_return_and_six_decimal_log')


def verify_stepping_attempt(point, attempt):
    from TROTASR.workflows.limit_validation import crossing_check, asimov_diagnostics
    from TROTASR.workflows.limit_serialization import reconcile_crossings
    passed,evidence=_verify_bisection_attempt(point,attempt)  # Exact saved ROOT/log identity.
    if attempt.get('exit_code')!=0 or evidence.get('reason')=='missing_saved_output':return passed,evidence
    cmd=attempt['command']
    if (cmd[cmd.index('--minosAlgo')+1]!='stepping' or '--strictBounds' not in cmd
            or '--picky' not in cmd or cmd[cmd.index('--run')+1]!='blind'
            or 'MINIMIZER_no_analytic' not in cmd):
        raise ValueError('Not the bounded explicit-BB stepping command')
    text=(ROOT/attempt['log']).read_text(errors='replace')
    rmax=float(cmd[cmd.index('--rMax')+1])
    values={float(q):v for q,v in evidence.get('quantiles',{}).items()}
    if .5 not in values:
        import uproot
        with uproot.open(ROOT/attempt['output']) as root:
            rows=root['limit'].arrays(['quantileExpected','limit'],library='np')
        medians=[float(v) for q,v in zip(rows['quantileExpected'],rows['limit']) if abs(float(q)-.5)<1e-5]
        if len(medians)!=1:return False,dict(evidence,reason='missing_unique_saved_median')
        values={.5:medians[0]}
    median=stepping_median_evidence(text,values[.5],attempt['tolerance'],rmax)
    if len(values)==5:
        normalized=text.replace('Will search for NLL crossing by stepping','Will search for NLL crossing by bisection')
        passed,check=crossing_check(normalized,values,attempt['tolerance'])
        evidence=dict(check,quantiles=values,asimov=asimov_diagnostics(normalized))
        if not passed:passed,evidence=reconcile_crossings(normalized,evidence,attempt['tolerance'])
    else:
        passed=False;evidence.update(quantiles=values,crossings=[median],full_five_quantile_passed=False)
    evidence.update(median_only_verified=median['passed'],median_crossing=median,
                    validation_adapter='stepping_actual_profile_same_crossing_thresholds')
    return passed,evidence


def validate_attempt(point, attempt):
    """Canonical native/stepping adapters, with full five-quantile acceptance."""
    command = attempt['command']
    algorithm = command[command.index('--minosAlgo') + 1]
    if algorithm == 'bisection':
        return _verify_bisection_attempt(point, attempt)
    if any(flag in command for flag in ('--freezeParameters', '--setParameters',
            '--setParameterRanges', '--snapshotName', '--noFitAsimov', '--toysFrequentist')):
        raise ValueError('Changed likelihood or Asimov definition in numerical adapter')
    expected = {'new': 'native_profiled', 'stepping': 'stepping_full5'}
    if algorithm not in expected or attempt.get('run3_numerical_adapter') != expected[algorithm]:
        raise ValueError('Missing exact numerical-adapter provenance')
    verifier = verify_native_attempt if algorithm == 'new' else verify_stepping_attempt
    try:
        passed, evidence = verifier(point, attempt)
    except ValueError as exc:
        numerical_failures = {
            'Expected five native bracket searches',
            'Missing/beyond-authorized native profiled NLL rows',
            'Missing finite authorized median stepping search',
            'Invalid actual stepping profile rows',
        }
        if str(exc) not in numerical_failures:
            raise  # In particular: no identity, hash, command or source bypass.
        _, evidence = _verify_bisection_attempt(point, attempt)
        return False, dict(evidence, reason=str(exc),
            full_five_quantile_passed=False, numerical_search_incomplete=True,
            median_only_success_is_not_accepted=True)
    full = (passed and five_quantile_structure(evidence)
            and all(c['passed'] for c in evidence['crossings']))
    return bool(full), dict(evidence, full_five_quantile_passed=bool(full),
                           median_only_success_is_not_accepted=True)



def _same_point_terminal_logs(workspace, mass, primary, previous, logs, commands):
    """Bind an already read failed log to its actual command, never another point."""
    known = {tuple(command) for command in commands}
    attempts = {a.get('log'): a for record in (primary or {}, previous or {})
                for a in record.get('attempts', []) if a.get('log')}
    selected = []
    for name, text in logs:
        attempt = attempts.get(name)
        if not attempt or attempt.get('exit_code') is None or attempt.get('passed'):
            continue
        command = attempt.get('command', [])
        try:
            valid = (tuple(command) in known and command[command.index('-M')+1] == 'AsymptoticLimits'
                and internal_path(ROOT/command[command.index('-M')+2]) == internal_path(workspace)
                and command[command.index('-n')+1] == '_'+mass
                and command[command.index('--run')+1] == 'blind'
                and command[command.index('--minosAlgo')+1] == 'bisection'
                and '--picky' in command and not any(flag in command for flag in
                    ('--setParameters', '--freezeParameters', '--setParameterRanges',
                     '--snapshotName', '--noFitAsimov', '--toysFrequentist')))
        except (ValueError, IndexError, TypeError):
            valid = False
        if valid:
            selected.append((name, text, attempt))
    return selected


_MINIMIZER_STATUS = re.compile(
    r'Minimization (?:finished|success!)\s*(?:with\s+)?status\s*=\s*([+-]?\d+)')
_CROSSING_START = 'Will search for NLL crossing'


def _profile_stream_diagnostics(text):
    """Keep signed failures and associate each NLL row with a successful fit."""
    start = text.find(_CROSSING_START)
    initial = text if start < 0 else text[:start]
    conditional = '' if start < 0 else text[start:]
    statuses = [int(value) for value in _MINIMIZER_STATUS.findall(text)]
    initial_statuses = [int(value) for value in _MINIMIZER_STATUS.findall(initial)]
    conditional_statuses = [int(value) for value in _MINIMIZER_STATUS.findall(conditional)]
    certified = []
    # A fallback may finish successfully after an earlier failure. Rows can be
    # associated here, but envelope reuse below rejects the whole failed attempt.
    for block in conditional.split(_CROSSING_START)[1:]:
        cursor = 0
        status = None
        for row in SCAN.finditer(block):
            preceding = _MINIMIZER_STATUS.findall(block[cursor:row.start()])
            if preceding:
                status = int(preceding[-1])
            if status == 0:
                certified.append((float(row[1]), float(row[2])))
            cursor = row.end()
            status = None  # A previous evaluation's success cannot certify this one.
    return dict(initial_statuses=initial_statuses,
                conditional_statuses=conditional_statuses,
                failed_statuses=[status for status in statuses if status != 0],
                conditional_failures=[status for status in conditional_statuses if status not in (0, 1)],
                certified_rows=certified)


def _terminal_profile_diagnostics(name, text, attempt):
    """Read only this terminal attempt's adjacent diagnostic log, never history.

    At verbosity 2, Combine suppresses conditional statuses in stdout while
    writing them to combine_logger.out. Its failure must not be treated as a
    valid profile sample just because stdout has an ``At r`` line.
    """
    streams = [(name, text, attempt.get('log_sha256'))]
    logger_error = None
    try:
        log = internal_path(ROOT/name)
        logger = log.parent/'combine_logger.out'
        if logger.is_file():
            if logger.resolve().parent != log.resolve().parent:
                raise ValueError('Diagnostic logger escapes the exact attempt directory')
            if logger.stat().st_size > 8*1024**2:
                raise ValueError('Diagnostic logger exceeds bounded recovery read')
            raw = logger.read_bytes()
            streams.append((str(logger), raw.decode(errors='replace'), hashlib.sha256(raw).hexdigest()))
    except (OSError, ValueError) as exc:
        logger_error = str(exc)
    parsed = [(source, _profile_stream_diagnostics(content), checksum)
              for source, content, checksum in streams]
    failed = [status for _, item, _ in parsed for status in item['failed_statuses']]
    conditional = [status for _, item, _ in parsed for status in item['conditional_failures']]
    certified = {row for _, item, _ in parsed for row in item['certified_rows']}
    rows = {(float(r), float(n)) for r, n in SCAN.findall(text)}
    return dict(source_log=name, diagnostic_sources=[dict(path=source, sha256=checksum)
                    for source, _, checksum in parsed], logger_error=logger_error,
                failed_statuses=failed, conditional_failed_statuses=conditional,
                initial_failed_statuses=[status for _, item, _ in parsed
                    for status in item['initial_statuses'] if status not in (0, 1)],
                certified_rows=sorted(certified),
                stdout_profile_rows=len(rows),
                all_stdout_rows_certified=bool(rows) and rows <= certified,
                range_evidence_usable=(not logger_error and not failed and bool(rows) and rows <= certified))


def _conditional_failure_evidence(workspace, mass, primary, previous, logs, commands):
    result = []
    for name, text, attempt in _same_point_terminal_logs(
            workspace, mass, primary, previous, logs, commands):
        diagnosis = _terminal_profile_diagnostics(name, text, attempt)
        if diagnosis['conditional_failed_statuses']:
            result.append(diagnosis)
    return result


def measured_five_target_envelope(workspace, mass, primary, previous, logs, commands):
    """Reuse only actual same-attempt profile pairs spanning ALL five targets.

    A single blind Asimov profile has the same NLL baseline for all targets.
    Its search blocks may be incomplete; rows are never pooled across attempts,
    fitted values are not extrapolated, and no artificial zero point is added.
    Inconsistent/non-monotonic or changed-minimum profiles are not reused.
    """
    from TROTASR.workflows.limit_validation import asimov_diagnostics
    candidates = []
    rejected = []
    for name, text, attempt in _same_point_terminal_logs(
            workspace, mass, primary, previous, logs, commands):
        convergence = _terminal_profile_diagnostics(name, text, attempt)
        if not convergence['range_evidence_usable']:
            rejected.append(dict(log=name, reason='profile_rows_not_certified_converged',
                                 convergence=convergence))
            continue
        if text.count('Fit to asimov dataset:') != 1:
            rejected.append(dict(log=name, reason='no_unique_asimov_baseline'))
            continue
        tail = text.split('Fit to asimov dataset:', 1)[1]
        blocks = tail.split('Will search for NLL crossing by bisection')[1:]
        rows = [(float(r), float(n)) for block in blocks for r, n in SCAN.findall(block)]
        fitted_r = asimov_diagnostics(text).get('fitted_r')
        old_max = float(attempt['command'][attempt['command'].index('--rMax')+1])
        if (not math.isfinite(old_max) or not 0 < old_max <= 100000
                or not 1 <= len(blocks) <= 5 or not rows or fitted_r is None
                or not math.isfinite(fitted_r)
                or any(not math.isfinite(r) or not math.isfinite(n) or r <= 0 or n <= 0
                       or r > old_max for r, n in rows)):
            rejected.append(dict(log=name, reason='invalid_or_unbounded_profile_rows'))
            continue
        rows = sorted(set(rows))
        if (any(r1 >= r2 or n1 > n2 for (r1, n1), (r2, n2) in zip(rows, rows[1:]))
                or convergence['failed_statuses']):
            rejected.append(dict(log=name, reason='inconsistent_or_failed_profile_minimum'))
            continue
        brackets = []
        for q in QUANTILES:
            target = target_nll(q)
            below = [(r, n) for r, n in rows if n < target]
            above = [(r, n) for r, n in rows if n > target]
            if not below or not above or below[-1][0] >= above[0][0]:
                break
            brackets.append(dict(quantile=q, target_delta_nll=target,
                                 below=list(below[-1]), above=list(above[0])))
        if len(brackets) != 5 or fitted_r >= min(b['below'][0] for b in brackets):
            rejected.append(dict(log=name, reason='not_all_five_targets_bracketed_above_minimum'))
            continue
        measured_upper = max(b['above'][0] for b in brackets)
        proposal = 1.25*measured_upper
        if not 0 < proposal <= 100000:
            rejected.append(dict(log=name, reason='measured_envelope_outside_finite_ceiling'))
            continue
        candidates.append(dict(status='available', source_log=name,
            source_log_sha256=attempt.get('log_sha256'), source_attempt=attempt.get('name'),
            brackets=brackets, search_blocks=len(blocks), fitted_r=fitted_r,
            convergence=convergence,
            sampled_upper_r=measured_upper, proposed_rmax=proposal, safety_factor=1.25,
            lower_bound_unchanged=True, extrapolated=False, seed_is_not_a_limit=True))
    if not candidates:
        return dict(status='unavailable', reason='no_same_attempt_consistent_five_target_envelope',
                    rejected=rejected)
    return min(candidates, key=lambda evidence: evidence['proposed_rmax'])


def recovery_stage_order(workspace, mass, primary, previous, logs, commands, *, speed_policy=False):
    """Propose an order only for a NEW ledger; callers must freeze it once."""
    original = list(STAGE_LIMITS)
    if not speed_policy:
        return original, dict(status='unchanged', reason='speed_policy_not_requested')
    envelope = measured_five_target_envelope(workspace, mass, primary, previous, logs, commands)
    if envelope['status'] == 'available':
        return ['base', 'measured', 'precision', 'minima', 'crossings', 'branches'], dict(
            reason='all_five_targets_have_same_attempt_measured_brackets',
            measured_envelope=envelope, finite_stage_budgets=dict(STAGE_LIMITS))
    conditional = _conditional_failure_evidence(workspace, mass, primary, previous, logs, commands)
    if conditional:
        return ['base', 'precision', 'minima', 'measured', 'crossings', 'branches'], dict(
            reason='conditional_minimization_failed_try_distinct_algorithm_then_precision',
            basis=conditional, measured_envelope=envelope,
            finite_stage_budgets=dict(STAGE_LIMITS))
    initial_failure = []
    for name, text, attempt in _same_point_terminal_logs(workspace, mass, primary, previous, logs, commands):
        if 'Will search for NLL crossing' in text:
            continue  # A later profile difficulty is not an initial-minimum failure.
        diagnosis = _terminal_profile_diagnostics(name, text, attempt)
        if diagnosis['initial_failed_statuses']:
            initial_failure.append(diagnosis)
    if initial_failure:
        return ['base', 'minima', 'precision', 'measured', 'crossings', 'branches'], dict(
            reason='initial_minimum_failure_try_existing_analytic_prefit_stage_first',
            basis=initial_failure, finite_stage_budgets=dict(STAGE_LIMITS))
    return original, dict(status='unchanged', reason='no_supported_failure_specific_reordering',
                          measured_envelope=envelope)


def optimize_saved_command(workspace, mass, command, logs, previous_commands, *, attempts=()):
    """Narrow a NEW command's rMax only from five measured same-point targets.

    The caller supplies terminal attempt receipts to bind each log to its exact
    command. Do not call this on an already frozen plan or running command.
    Absence of a complete saved nuisance snapshot is recorded, never replaced
    by guessed parameter values or a skipped Asimov fit.
    """
    result = list(command)
    evidence = dict(status='not_applied', acceptance_unchanged=True,
        warm_start=dict(status='unavailable',
            reason='no_verified_full_nuisance_snapshot_in_saved_attempt_receipts',
            parameters_changed=False))
    logs, previous_commands, attempts = list(logs), list(previous_commands), list(attempts)
    if not logs or not attempts:
        return result, dict(evidence, reason='no_bound_terminal_attempt_logs')
    try:
        eligible = (result[result.index('-M')+1] == 'AsymptoticLimits'
            and internal_path(ROOT/result[result.index('-M')+2]) == internal_path(workspace)
            and result[result.index('-n')+1] == '_'+mass
            and result[result.index('--run')+1] == 'blind'
            and result[result.index('--minosAlgo')+1] == 'bisection'
            and '--picky' in result and not any(flag in result for flag in
                ('--setParameters', '--freezeParameters', '--setParameterRanges',
                 '--snapshotName', '--noFitAsimov', '--toysFrequentist'))
            and ('--rMin' not in result or float(result[result.index('--rMin')+1]) == 0))
        old_max = float(result[result.index('--rMax')+1])
    except (ValueError, IndexError, TypeError):
        eligible = False
    if not eligible or not math.isfinite(old_max) or old_max <= 0:
        return result, dict(evidence, reason='command_not_eligible_for_range_only_reuse')
    envelope = measured_five_target_envelope(workspace, mass, {}, dict(attempts=attempts),
                                             logs, previous_commands)
    evidence['measured_envelope'] = envelope
    if envelope['status'] != 'available':
        return result, dict(evidence, reason=envelope['reason'])
    proposal = envelope['proposed_rmax']
    if proposal >= old_max:
        return result, dict(evidence, reason='five_target_envelope_does_not_narrow_command')
    result[result.index('--rMax')+1] = repr(proposal)
    if tuple(result) in {tuple(c) for c in previous_commands}:
        return list(command), dict(evidence, reason='range_reuse_would_repeat_an_existing_command')
    return result, dict(evidence, status='applied', previous_rmax=old_max, proposed_rmax=proposal,
                        changed_arguments=['--rMax'], lower_bound_unchanged=True)


def recovery_commands(stage, workspace, mass, primary, previous, logs, commands, *, speed_policy=False):
    """Return (new command specs, applicability evidence) without running fits.

    Stages are base, precision, measured, minima, crossings, branches. The
    caller visits each finite stage once and keeps the same stage attempt
    ledger when branches append more configurations. There are at most two
    base and sixteen advanced configurations per point. previous is the
    latest real terminal point/stage record; crossings and branches receive
    the accumulated minima-stage attempts. logs contains (path, text)
    pairs and commands contains every already executed argv for this
    point. An optional previous["already_tried_calls2m"] carries the existing
    base-stage alternative without inspecting historical directory names.
    """
    if stage not in STAGE_LIMITS:
        raise ValueError('Unknown numerical recovery stage: '+str(stage))
    primary, previous = primary or {}, previous or {}
    logs, commands = list(logs), list(commands)
    envelope = (measured_five_target_envelope(workspace, mass, primary, previous, logs, commands)
                if speed_policy else None)
    conditional = []
    seed_logs = logs
    range_floor = None
    discarded_profiles = []
    if speed_policy:
        seed_logs = []
        for name, text, attempt in _same_point_terminal_logs(
                workspace, mass, primary, previous, logs, commands):
            diagnosis = _terminal_profile_diagnostics(name, text, attempt)
            if diagnosis['conditional_failed_statuses']:
                conditional.append(diagnosis)
            if not diagnosis['range_evidence_usable']:
                # Keep the existing Asimov-error fallback, but never feed a
                # failed or uncertified conditional evaluation to adaptive_seed.
                text = SCAN.sub('', text)
                discarded_profiles.append(dict(log=name, reason='uncertified_profile_rows',
                                               convergence=diagnosis))
            seed_logs.append((name, text))
            old = attempt['command']
            try:
                maximum = float(old[old.index('--rMax')+1])
                if math.isfinite(maximum) and 0 < maximum <= 100000:
                    range_floor = maximum  # Existing chronological receipt order.
            except (ValueError, IndexError):
                pass
    for index, record in enumerate((primary, previous)):
        conditional_stage = index == 1 and stage in ('crossings', 'branches')
        if not conditional_stage and (record.get('pid') or record.get('status') == 'running'):
            raise ValueError('Numerical recovery requires terminal predecessors')
        if conditional_stage and any(a.get('exit_code') is None for a in record.get('attempts', [])):
            raise ValueError('Conditional recovery planning requires terminal command receipts')
        if record.get('status') in ('complete', 'verified_not_promoted', 'existing_verified') or any(
                attempt.get('passed') for attempt in record.get('attempts', [])):
            raise ValueError('Reuse the already valid saved result')
    try:
        if stage == 'base':
            seed, evidence = adaptive_seed(seed_logs)
            specs = [dict(spec, command=profile_recovery_command(workspace, mass, spec))
                     for spec in unfinished_configs(seed, bool(previous.get('already_tried_calls2m', False)))]
            if conditional:
                # Only reorder the two existing configurations of this NEW
                # finite plan. The caller never calls us for an already frozen plan.
                specs.sort(key=lambda spec: spec.get('algorithm') != 'Combined')
            diagnosis = dict(range_evidence=evidence)
        elif stage == 'precision':
            specs, evidence = precision_commands(workspace, mass, seed_logs, commands)
            diagnosis = dict(range_evidence=evidence)
        elif stage == 'measured':
            if speed_policy and envelope['status'] != 'available':
                return [], dict(stage=stage, status='not_applicable',
                    reason='no_converged_same_attempt_five_target_bracket_for_reduced_range',
                    measured_envelope=envelope, maximum_new_attempts=0)
            specs = measured_start_commands(workspace, mass, primary, previous, logs,
                measured_envelope=envelope if envelope and envelope['status'] == 'available' else None)
            diagnosis = {}
        elif stage == 'minima':
            attempts = previous.get('attempts', [])
            if not attempts:
                return [], dict(stage=stage, status='not_applicable',
                    reason='no_actual_previous_command_for_minimum_search', maximum_new_attempts=0)
            specs, diagnosis = minimum_commands(attempts[-1]['command']), {}
        elif stage == 'crossings':
            specs, diagnosis = crossing_commands(previous.get('attempts', []), commands)
        else:
            specs, diagnosis = native_stepping_proposals(previous.get('attempts', []), dict(logs), commands)
    except ValueError as error:
        reason = str(error)
        if reason not in (
                'No measured NLL or finite Asimov fit error; explicit diagnosis required',
                'No measured near-median starting region',
                'Measured start does not reduce the failed initial jump'):
            raise
        return [], dict(stage=stage, status='not_applicable', reason=reason, maximum_new_attempts=0)
    known = {tuple(command) for command in commands}
    selected = []
    for spec in specs:
        if speed_policy:
            spec = dict(spec, command=list(spec['command']))
            floor_evidence = None
            if (stage in ('base', 'precision') and range_floor is not None
                    and envelope['status'] != 'available'):
                cmd = spec['command']
                index = cmd.index('--rMax')+1
                proposed = float(cmd[index])
                if proposed < range_floor:
                    cmd[index] = repr(range_floor)
                    spec['rmax'] = range_floor
                    floor_evidence = dict(reason='no_five_target_support_keep_terminal_range',
                                          discarded_proposal=proposed, retained_rmax=range_floor)
            if stage in ('base', 'precision', 'measured', 'minima'):
                command, reuse = optimize_saved_command(workspace, mass, spec['command'], logs, commands,
                    attempts=[*primary.get('attempts', []), *previous.get('attempts', [])])
                spec['command'] = command
                if reuse['status'] == 'applied':
                    spec['rmax'] = reuse['proposed_rmax']
                if stage == 'measured' and envelope['status'] == 'available':
                    reuse = dict(reuse, measured_envelope=envelope,
                                 measured_start_seed_uses_all_five_targets=True)
            else:
                reuse = dict(status='not_applied', reason='conditional_crossing_or_range_branch_preserved',
                    warm_start=dict(status='unavailable',
                        reason='no_verified_full_nuisance_snapshot_in_saved_attempt_receipts'))
            spec['speed_optimization'] = reuse
            if floor_evidence:
                spec['speed_optimization'] = dict(reuse, range_floor=floor_evidence)
        identity = tuple(spec['command'])
        if identity in known:
            continue
        known.add(identity)
        selected.append(spec)
    if len(selected) > STAGE_LIMITS[stage]:
        raise ValueError('Finite numerical stage budget exceeded')
    evidence = dict(diagnosis, stage=stage, status='applicable' if selected else 'not_applicable',
                    maximum_new_attempts=len(selected))
    if speed_policy:
        evidence['speed_optimization'] = dict(measured_envelope=envelope,
            warm_start=dict(status='unavailable', reason='no_verified_full_nuisance_snapshot_in_saved_attempt_receipts'),
            conditional_failure_basis=conditional,
            base_order='existing_combined_first' if stage == 'base' and conditional else 'existing_order',
            discarded_profile_evidence=discarded_profiles,
            stage_budget_unchanged=True, acceptance_unchanged=True)
    if not selected:
        evidence.setdefault('reason', 'all_exact_commands_already_attempted')
    return selected, evidence

