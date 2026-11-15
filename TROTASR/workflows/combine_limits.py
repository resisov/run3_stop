"""Nominal expected Combine grid with the adopted finite crossing-checked policy.

No cards, ROOT files, fit values, payloads or analysis modules outside TROTASR
are read. CMSSW is a separately identified numerical software runtime. Plotting
is a later stage; completed fits do not imply complete AN deliverables.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.limit_validation import validate, REL_ACC, ABS_ACC
from TROTASR.workflows import resources

FIELDS = {.025:'expected_m2', .16:'expected_m1', .5:'expected', .84:'expected_p1', .975:'expected_p2'}
ATTEMPTS = (('bisection_s0', .1, 1.), ('bisection_s0_refined', .01, 2.))
GIB = 1024**3
POINT_RESERVATION = 3*GIB


def verify_combined_grid(record):
    """Check the adopted combined model and exact card-grid inventory."""
    from TROTASR.utils.paths import ERAS
    if record.get('scope') != 'full_combined_systematic_production':
        if tuple(map(str, record.get('years', ()))) == ERAS:
            raise ValueError('Six-era grid must identify the adopted combined scope')
        return
    from TROTASR.workflows.build_nominal_grid import require_statistical_revision
    from TROTASR.workflows.diagnostic_common import require_revised_record, record_years
    require_revised_record(record)
    policy = record.get('statistical_revision', record.get('contract', {}).get('statistical_revision'))
    points = record.get('points', [])
    counts = {m: sum(p['model'] == m for p in points) for m in ('T2tt', 'T2bW', 'T2tb')}
    expected_bins = 214 if record.get('era_merged') else 214 * len(record_years(record))
    if (not require_statistical_revision(policy) or record.get('bins') != expected_bins
            or counts != {'T2tt': 225, 'T2bW': 365, 'T2tb': 365}
            or len({(p['model'], p['mass']) for p in points}) != 955
            or record.get('sr_data_blinded') is not True
            or not record.get('systematic_scope')):
        raise ValueError('Combined grid does not match the approved CR12/955 contract')




def runtime(output):
    if sys.platform != 'linux':
        raise RuntimeError('Combine and all ROOT products must stay on hep2 scratch')
    config = read_json(ROOT/'stats/runtime.json')
    cmssw = Path(config['cmssw'])
    if not (cmssw/'src').is_dir() or not Path(config['cmsset_default']).is_file():
        raise FileNotFoundError('Configured numerical runtime is unavailable')
    guard = internal_path(ROOT/config['guard'])
    for d in ('tmp','cache'):
        (output/d).mkdir(parents=True,exist_ok=True)
    env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(output/'tmp'),
        XDG_CACHE_HOME=str(output/'cache'))
    raw = subprocess.check_output(['bash','--noprofile','--norc','-c',
        'source '+shlex.quote(config['cmsset_default'])+' && eval "$(scramv1 runtime -sh)" && env -0'],
        cwd=cmssw/'src',env=env)
    env = dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    # Discard inherited Python analysis paths. CMSSW's Python3 software path
    # is added by this internal copy of the adopted compatibility guard.
    env['PYTHONPATH'] = str(guard)
    fit_stack_limit()
    return env


def fit_stack_limit():
    soft,hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < 256*1024**2:
        resource.setrlimit(resource.RLIMIT_STACK,(min(256*1024**2,hard) if hard!=resource.RLIM_INFINITY else 256*1024**2,hard))


def fit_command(workspace, mass, rmax, tolerance):
    return ['combine','-M','AsymptoticLimits',str(workspace),'-m','120',
        '--run','blind','-n','_'+mass,
        '--cminDefaultMinimizerStrategy','0','--cminDefaultMinimizerTolerance',str(tolerance),
        '--cminFallbackAlgo','Minuit2,Migrad,1:0.1',
        '--minosAlgo','bisection','--picky','--rMax',str(rmax),
        '--rRelAcc',str(REL_ACC),'--rAbsAcc',str(ABS_ACC),'-v','2']


def convert_workspace(card, workspace, record, state_path, env, stream):
    from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION

    def execute(command, cwd):
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
        record.update(child_identity(proc.pid), phase='workspace', command=command)
        write_json(state_path, record)
        return proc.wait()

    convert = ['text2workspace.py', str(card), '-m', '120', '-o', str(workspace), '--verbose', '0']
    revision = record.get('model_revision')
    if revision is None:
        return execute(convert, card.parent)
    if revision != MODEL_REVISION:
        raise ValueError('Unknown Run 3 likelihood revision')

    with tempfile.TemporaryDirectory(prefix='.workspace-', dir=workspace.parent) as directory:
        directory = Path(directory)
        def worker(stage, source):
            return [sys.executable, '-m', 'TROTASR', 'limits', '--workspace-step', stage,
                    '--workspace-input', str(source), '--output', str(directory)]

        native = directory/'native.root'
        convert[1], convert[5] = str(directory/'card.txt'), str(native)
        for command, cwd in ((worker('prepare', card), ROOT.parent), (convert, directory),
                             (worker('link', native), ROOT.parent)):
            rc = execute(command, cwd)
            if rc:
                return rc
        (directory/'linked.root').rename(workspace)
        return 0


def verify_card(point, digest=sha256):
    from TROTASR.utils.paths import ERAS, EARLY_ERAS
    from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
    card = internal_path(ROOT/point['card'])
    if digest(card) != point['card_sha256']:
        raise ValueError('Card changed: '+str(card))
    meta = read_json(card.with_suffix('.manifest.json'))
    mass = re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass'])
    if (not mass or point['model'] not in ('T2tt','T2tb','T2bW') or meta['cr_only']
        or not meta['sr_data_blinded'] or meta['card_sha256']!=point['card_sha256']
        or tuple(map(str,meta['years'])) not in (('2024','2025'), EARLY_ERAS, ERAS)
        or (tuple(map(str,meta['years'])) == ERAS and not require_run3_systematics_policy(meta))
        or meta['auto_mc_stats'] != [10,1,1]
        or (meta['topology'],meta['mStop'],meta['mLSP']) != (point['model'],int(mass[1]),int(mass[2]))):
        raise ValueError('Card/point/blinding contract mismatch')
    for source in meta['inputs']:
        folder = internal_path(ROOT/source['path'])
        manifest = folder/'model_manifest.json'
        if digest(manifest) != source['sha256']:
            raise ValueError('Template manifest changed')
        record=read_json(manifest)
        if record.get('years', ['2024', '2025']) != list(map(str, meta['years'])):
            raise ValueError('Card and template target eras differ')
        if (record['status']!='templates_ready' or not record['root_integrity_checked']
            or record.get('scope') not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production')):
            raise ValueError('Template ROOT not verified')
        verify_combined_grid(record)
        template_folder = internal_path(ROOT/record['template_directory']) if record.get('template_directory') else folder
        for name,expected_digest in record['files'].items():
            if digest(template_folder/name) != expected_digest:
                raise ValueError('Template file changed: '+name)
        declared = {internal_path(template_folder/name) for name in record['files']}
        referenced = {internal_path(card.parent/f[3]) for f in
            (line.split() for line in card.read_text().splitlines()) if f and f[0]=='shapes'}
        if referenced != declared:
            raise ValueError('Card shapes must reference exactly the verified era template files')
    # Every shapes directive must resolve inside the package, even if a card
    # was authored by a different caller. ROOT may otherwise read any path.
    for line in card.read_text().splitlines():
        f=line.split()
        if f and f[0]=='shapes':
            internal_path(card.parent/f[3])
    return card


def wait_for_inference(process, log):
    """Stop only persistent nonfinite-POI runaway output, not a healthy fit.

    This does not alter any numerical option or acceptance criterion. Bounded
    tail reads prevent a failed native minimizer from filling the shared disk.
    """
    consecutive, previous_size = 0, -1
    while process.poll() is None:
        time.sleep(1)
        size = log.stat().st_size
        if size < 64*1024**2 or size <= previous_size:
            consecutive = 0
            previous_size = size
            continue
        previous_size = size
        with log.open('rb') as stream:
            stream.seek(max(0, size-65536))
            tail = stream.read(65536)
        nonfinite = len(re.findall(rb'At r\s*=\s*[-+]?nan\b', tail, re.I))
        consecutive = consecutive+1 if nonfinite >= 3 else 0
        if consecutive < 3 or process.poll() is not None:
            continue
        evidence = dict(reason='persistent_nonfinite_poi_runaway', pid=process.pid,
            log_bytes=size, tail_nonfinite_poi_count=nonfinite,
            consecutive_growing_samples=consecutive, time=time.time(),
            signals=['SIGINT'], acceptance_relaxed=False)
        process.send_signal(signal.SIGINT)
        try:
            result = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            evidence['signals'].append('SIGTERM')
            result = process.wait()
        evidence['exit_code'] = result
        write_json(log.with_name('nonfinite_runaway.json'), evidence)
        return result
    return process.returncode


def failed_log_text(path):
    """A stopped NaN loop is failure evidence, never a measured NLL bracket."""
    path = internal_path(path)
    if (path.parent/'nonfinite_runaway.json').exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536))
            return 'Nonfinite runaway terminated; no reusable profile evidence.\n'+stream.read(65536).decode(errors='replace')
    return path.read_text(errors='replace')


def run_primary_point(point, output, env):
    """Run the two primary configurations under the caller's point lock."""
    from TROTASR.workflows.limit_recovery import optimize_saved_command

    card = ready_card(point)
    work = output/point['model']/point['mass']
    work.mkdir(parents=True,exist_ok=True)
    state_path=work/'state.json'
    record=read_json(state_path) if state_path.exists() else dict(point,status='running',started=time.time(),attempts=[])
    if (record['card_sha256'] != point['card_sha256'] or
            record.get('model_revision') != point.get('model_revision')):
        raise ValueError('Preserve old fit for divergent card')
    if record['status'] == 'complete':
        return record
    assert_previous_child_exited(record, work)
    workspace=work/'workspace.root'
    if not record.get('workspace_valid'):
        if workspace.exists():
            raise ValueError('Interrupted workspace output retained; explicit recovery required')
        with (work/'workspace.log').open('x') as stream:
            rc=convert_workspace(card,workspace,record,state_path,env,stream)
        if rc or not workspace.is_file() or workspace.stat().st_size<1000:
            record.update(status='failed',error='workspace_conversion_failed',exit_code=rc,finished=time.time(),pid=None)
            write_json(state_path,record);return record
        import uproot
        with uproot.open(workspace) as root:
            if 'w' not in root:raise ValueError('Missing RooWorkspace w')
        record.update(workspace_valid=True,workspace_sha256=sha256(workspace),pid=None)
        write_json(state_path,record)
    elif not workspace.is_file():
        raise ValueError('Previously verified workspace missing')
    completed_attempts={a['name'] for a in record['attempts']}
    for label, tolerance, scale in ATTEMPTS:
        if label in completed_attempts:continue
        here=work/label
        if here.exists():
            raise ValueError('Interrupted fit attempt retained; explicit recovery required')
        here.mkdir();(here/'tmp').mkdir()
        rmax = float(point.get('rmax_seed',20.))*scale
        command=fit_command(workspace,point['mass'],rmax,tolerance)
        saved_attempts = record['attempts']
        saved_logs = [(a['log'], failed_log_text(ROOT/a['log']))
                      for a in saved_attempts if a.get('log')
                      and internal_path(ROOT/a['log']).is_file()]
        command, speed_evidence = optimize_saved_command(
            workspace, point['mass'], command, saved_logs,
            [a['command'] for a in saved_attempts if a.get('command')],
            attempts=saved_attempts)
        log=here/'combine.log'
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=here,env=dict(env,TMPDIR=str(here/'tmp')),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            record.update(child_identity(proc.pid), phase='fit', command=command, current_attempt=label)
            write_json(state_path,record);rc=wait_for_inference(proc,log)
        result=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
        passed,evidence=validate(result,log,tolerance) if rc==0 else (False,dict(exit_code=rc))
        ready_card(point)
        attempt=dict(name=label,tolerance=tolerance,passed=passed,exit_code=rc,
            log=str(log.relative_to(ROOT)), log_sha256=sha256(log), command=command,
            output=str(result.relative_to(ROOT)), output_sha256=sha256(result) if result.is_file() else None,
            validation=evidence)
        if speed_evidence is not None:
            attempt['numerical_speed'] = speed_evidence
        record['attempts'].append(attempt)
        write_json(here/'validation.json',attempt)
        record.update(pid=None)
        if passed:
            record.update(status='complete',finished=time.time(),output=str(result.relative_to(ROOT)),
                output_sha256=sha256(result),validation=evidence)
            write_json(state_path,record);return record
        write_json(state_path,record)
    record.update(status='failed',finished=time.time(),error='finite_attempts_exhausted',pid=None)
    write_json(state_path,record);return record


def collect(points,results,model,scope):
    selected=[p for p in points if p['model']==model]
    complete={}
    for point in selected:
        r=results.get(model+'/'+point['mass'],{})
        if r.get('status')!='complete':continue
        q={float(k):v for k,v in r['validation']['quantiles'].items()}
        ms,ml=map(int,re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass']).groups())
        complete[point['mass']]=dict(mStop=ms,mLSP=ml,**{name:q[value] for value,name in FIELDS.items()})
    missing=[p['mass'] for p in selected if p['mass'] not in complete]
    return dict(status='complete' if not missing else 'partial' if complete else 'no_combine_outputs',
        scope=scope,points=complete,requested_point_count=len(selected),collected_point_count=len(complete),
        missing_points=missing,observed_sr_used=False,plots_status='not_generated')


EXPECTED_DISPLAY_COUNTS = {'T2tt': 205, 'T2bW': 345, 'T2tb': 345}
INPUT_CHECK_SCOPE = 'accepted_cards_templates_or_workspace; no_upstream_revalidation'


def ready_fit_points(source):
    """Consume the completed card-stage receipt, not its production ancestry."""
    if (source.get('status') != 'cards_ready'
            or source.get('scope') != 'full_combined_systematic_production'
            or source.get('era_merged') is not True
            or source.get('sr_data_blinded') is not True):
        raise ValueError('The limit stage requires the accepted era-merged, blinded card grid')
    points = source['points']
    keys = [p['model']+'/'+p['mass'] for p in points]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate point ownership in the delivered grid')
    selected = []
    for point in points:
        match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', point['mass'])
        if not match or point['model'] not in EXPECTED_DISPLAY_COUNTS:
            raise ValueError('Invalid delivered signal point')
        stop, lsp = map(int, match.groups())
        if 600 <= stop <= 1800 and 1 <= lsp <= 1800:
            selected.append(point)
    counts = {m: sum(p['model'] == m for p in selected) for m in EXPECTED_DISPLAY_COUNTS}
    if len(points) != 955 or counts != EXPECTED_DISPLAY_COUNTS:
        raise ValueError('Expected the delivered 955 cards and exactly 895 displayed points')
    return selected, dict(plot_range_only=True, source_total=len(points), total=len(selected),
        excluded_out_of_plot_range=len(points)-len(selected), expected_points_by_model=counts)


def ready_card(point):
    card = internal_path(ROOT/point['card'])
    if not card.is_file():
        raise FileNotFoundError('Delivered datacard is missing: '+str(card))
    return card


def accepted_fit_record(point, record, origin):
    """Reuse a completed receipt. Only NEW fits are opened/numerically checked."""
    if record.get('status') not in ('complete', 'verified_not_promoted', 'existing_verified'):
        return None
    identity = record.get('point', record)
    for name in ('model', 'mass', 'card_sha256', 'model_revision'):
        if identity.get(name) != point.get(name):
            raise ValueError('Saved point/card identity differs: '+origin)
    attempt = record.get('accepted_attempt')
    if attempt is None:
        matches = [a for a in record.get('attempts', []) if a.get('passed')
                   and a.get('exit_code') == 0 and a.get('output') == record.get('output')]
        if len(matches) != 1:
            raise ValueError('Accepted receipt must identify one successful command: '+origin)
        attempt = matches[0]
    if attempt.get('exit_code') != 0 or not attempt.get('passed'):
        raise ValueError('Saved receipt has no successful fit execution: '+origin)
    evidence = record.get('validation', attempt.get('validation', {}))
    values = {float(q): float(v) for q, v in evidence.get('quantiles', {}).items()}
    ordered = [values[q] for q in sorted(FIELDS) if q in values]
    if (set(values) != set(FIELDS) or any(not math.isfinite(v) or v <= 0 for v in values.values())
            or any(a >= b for a, b in zip(ordered, ordered[1:]))):
        raise ValueError('Accepted receipt lacks five finite ordered expected quantiles: '+origin)
    paths = dict(output=attempt['output'], log=attempt['log'],
                 workspace=record.get('workspace', str(Path(origin).parent/'workspace.root')))
    # Recovery receipts refer to the primary workspace, not a duplicate copy.
    if 'workspace' not in record:
        command = attempt.get('command', [])
        if '-M' in command and len(command) > command.index('-M')+2:
            paths['workspace'] = command[command.index('-M')+2]
    for path in paths.values():
        if not internal_path(ROOT/path).is_file():
            raise FileNotFoundError('Accepted fit artifact is missing: '+str(path))
    paths = {name: str(internal_path(ROOT/path)) for name, path in paths.items()}
    return dict(record, **point, status='complete', **paths, validation=evidence,
        accepted_attempt=attempt, command=attempt['command'], exit_code=0,
        receipt_origin=origin, input_check_scope=INPUT_CHECK_SCOPE)


def point_receipts(point, output):
    key = point['model']+'/'+point['mass']
    records = []
    paths = [('workflow', output/key/'workflow.json'), ('primary', output/key/'state.json')]
    for stage, path in paths:
        if path.is_file():
            record = read_json(path)
            identity = record.get('point', record)
            for name in ('model', 'mass', 'card_sha256', 'model_revision'):
                if identity.get(name) != point.get(name):
                    raise ValueError('Different point/card receipt retained: '+str(path))
            records.append((stage, path, record))
    return records


def saved_point_result(point, output):
    if 'model_revision' not in point and (output/'model.json').is_file():
        revision = read_json(output/'model.json').get('model_revision')
        if revision:
            point = dict(point, model_revision=revision)
    for stage, path, record in point_receipts(point, output):
        accepted = accepted_fit_record(point, record, str(path.relative_to(ROOT)))
        if accepted is not None:
            return accepted
    return None


def integrated_attempt(point, workspace, target, spec, env, state, state_path, progress):
    from TROTASR.workflows.limit_recovery import validate_attempt
    target.mkdir(parents=True, exist_ok=False)
    (target/'tmp').mkdir()
    log = target/'combine.log'
    product = target/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
    attempt = dict(spec, log=str(log.relative_to(ROOT)), output=str(product.relative_to(ROOT)),
                   started=time.time(), exit_code=None, passed=False)
    state['attempts'].append(attempt)
    write_json(state_path, state)
    with log.open('x') as stream:
        process = subprocess.Popen(spec['command'], cwd=target, env=dict(env, TMPDIR=str(target/'tmp')),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        state.update(child_identity(process.pid), phase=spec['stage'], command=spec['command'])
        write_json(state_path, state)
        progress(spec['stage'], process.pid)
        attempt['exit_code'] = wait_for_inference(process,log)
    attempt.update(finished=time.time(), log_sha256=sha256(log),
                   output_sha256=sha256(product) if product.is_file() else None)
    passed, evidence = validate_attempt(point, attempt)
    attempt.update(passed=passed, validation=evidence)
    state.update(pid=None)
    write_json(target/'validation.json', attempt)
    write_json(state_path, state)
    return attempt


def historical_success_commands(point, workspace, output, previous_commands=()):
    """Optional same-point successful numerical recipes; never reuse fit values.

    Only the canonical stage's successful numerical settings are read.
    Old campaign/history launchers, source trees and ROOT files are not read.
    The caller keeps its point lock and charges the existing branch budget.
    """
    from TROTASR.workflows.limit_recovery import profile_recovery_command
    source = output/'successful_recipes.json'
    if not source.is_file():
        return [], dict(status='unavailable', reason='no_canonical_successful_recipe')
    key = point['model']+'/'+point['mass']
    accepted = read_json(source).get('points',{}).get(key)
    if accepted is None:
        return [], dict(status='unavailable', reason='no_successful_same_point_recipe')
    command = list(accepted['command'])
    if (command[command.index('-M')+1] != 'AsymptoticLimits'
            or command[command.index('--run')+1] != 'blind' or '--picky' not in command
            or any(flag in command for flag in ('--freezeParameters','--setParameters',
                                               '--snapshotName','--noFitAsimov'))):
        raise ValueError('Historical recipe changed the expected-limit dataset/model')
    command[command.index('-M')+2] = str(workspace)
    rmax = float(command[command.index('--rMax')+1])
    if not math.isfinite(rmax) or not 0 < rmax <= 100000:
        raise ValueError('Invalid historical numerical seed')
    specs = [dict(name='prior_success_'+accepted['name'], command=command,
                  rmax=rmax, tolerance=float(command[command.index('--cminDefaultMinimizerTolerance')+1]))]
    for algorithm, strategy in (('Migrad',0),('Combined',1)):
        spec = dict(name='prior_success_'+algorithm.lower()+'_s'+str(strategy)+'_calls2m',
            algorithm=algorithm, strategy=strategy, tolerance=.1, precision=None,
            max_calls=2000000, rmax=rmax)
        specs.append(dict(spec, command=profile_recovery_command(workspace,point['mass'],spec)))
    known = set()
    for previous in previous_commands:
        normalized=list(previous)
        if '-M' in normalized:
            normalized[normalized.index('-M')+2]=str(workspace)
        known.add(tuple(normalized))
    selected = []
    for spec in specs:
        identity = tuple(spec['command'])
        if identity in known:
            continue
        known.add(identity)
        selected.append(dict(spec, stage='branches', recipe_source=str(source)))
    return selected, dict(status='applicable' if selected else 'already_consumed',
        source=str(source), rmax_seed=rmax)


def run_integrated_point(point, output, env, progress=lambda phase, pid: None):
    """One point lock and one slot for primary -> all finite recovery -> receipt."""
    from TROTASR.workflows.limit_recovery import (
        recovery_commands, recovery_stage_order, validate_attempt, STAGE_LIMITS)
    work = output/point['model']/point['mass']
    work.mkdir(parents=True, exist_ok=True)
    with (work/'point.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = saved_point_result(point, output)
        if saved is not None:
            return saved
        records = point_receipts(point, output)
        for stage, path, record in records:
            assert_previous_child_exited(record, work)
            if any(a.get('exit_code') is None for a in record.get('attempts', [])):
                raise RuntimeError('Interrupted command receipt retained; diagnose before new admission')
        progress('primary', None)
        primary = next((r for stage, _, r in records if stage == 'primary'), None)
        if primary is None or primary.get('error') != 'finite_attempts_exhausted':
            primary = run_primary_point(point, output, env)
        if primary.get('status') == 'complete':
            return accepted_fit_record(point, primary, str((work/'state.json').relative_to(ROOT)))
        if not primary.get('workspace_valid'):
            return dict(primary, status='needs_attention', error='workspace_conversion_requires_diagnosis')
        workspace = work/'workspace.root'
        if not workspace.is_file():
            raise FileNotFoundError('Accepted workspace is missing')
        state_path = work/'workflow.json'
        state = (read_json(state_path) if state_path.is_file() else
                 dict(point, attempts=[], stages={}, started=time.time()))
        state.update(status='running', pid=None, canonical_limits_pipeline=True,
                     workspace=str(workspace.relative_to(ROOT)), input_check_scope=INPUT_CHECK_SCOPE)
        state.setdefault('stages', {})
        history = list(primary.get('attempts', [])) + state['attempts']
        # First reconcile the actual saved, failed
        # output (float/log serialization). Retain that no-refit route, once per
        # failed attempt, without touching already accepted points or inputs.
        checked = state.setdefault('saved_attempt_checks', {})
        for attempt in history:
            if attempt.get('exit_code') != 0 or attempt.get('passed') or not attempt.get('output'):
                continue
            identity = json.dumps([attempt.get('output_sha256'), attempt.get('log_sha256')])
            if identity in checked:
                continue
            passed, evidence = validate_attempt(point, attempt)
            checked[identity] = dict(passed=passed, validation=evidence, output=attempt['output'])
            if passed:
                accepted = dict(attempt, passed=True, validation=evidence)
                state.update(status='complete', finished=time.time(), accepted_attempt=accepted,
                    output=accepted['output'], output_sha256=accepted['output_sha256'],
                    validation=evidence, pid=None)
                write_json(state_path, state)
                return accepted_fit_record(point, state, str(state_path.relative_to(ROOT)))
            write_json(state_path, state)
        # Try successful same-point numerical settings first, using this
        # model's workspace and the finite branch budget. Accepted points
        # return before this lookup; fitted values are not reused.
        prescription = state.get('historical_success_prescription')
        if prescription is None:
            specs, diagnosis = historical_success_commands(point, workspace, output,
                [a['command'] for a in history if a.get('command')])
            prescription = dict(diagnosis, plans=specs)
            state['historical_success_prescription'] = prescription
            write_json(state_path, state)
        branch_attempts = [a for a in state['attempts'] if a.get('stage') == 'branches']
        known = {tuple(a['command']) for a in history if a.get('command')}
        specs = [s for s in prescription['plans'] if tuple(s['command']) not in known]
        for spec in specs[:max(0, STAGE_LIMITS['branches']-len(branch_attempts))]:
            ledger = state['stages'].setdefault('branches',dict(status='pending',applicability=[],plans=[]))
            ledger.setdefault('plans',[]).append([spec])
            ledger['status'] = 'pending'
            write_json(state_path,state)
            attempt = integrated_attempt(point,workspace,work/'recovery/branches'/spec['name'],
                                         spec,env,state,state_path,progress)
            history.append(attempt)
            if attempt['passed']:
                state.update(status='complete',finished=time.time(),accepted_attempt=attempt,
                    output=attempt['output'],output_sha256=attempt['output_sha256'],
                    validation=attempt['validation'],pid=None)
                ledger['status'] = 'complete'
                write_json(state_path,state)
                return accepted_fit_record(point,state,str(state_path.relative_to(ROOT)))
        # Preserve existing plans; derive a conditional-status-based stage
        # order only for a new recovery ledger.
        if 'recovery_stage_order' not in state:
            fixed = bool(state['attempts'] or any(
                ledger.get('plans') for ledger in state['stages'].values()))
            if fixed:
                order, diagnosis = list(STAGE_LIMITS), dict(status='retained_existing_frozen_plan')
            else:
                logs = [(a['log'], failed_log_text(ROOT/a['log']))
                        for a in history if a.get('log') and internal_path(ROOT/a['log']).is_file()]
                order, diagnosis = recovery_stage_order(workspace, point['mass'], primary,
                    dict(status='failed',pid=None,attempts=history),logs,
                    [a['command'] for a in history if a.get('command')],speed_policy=True)
            if len(order) != len(STAGE_LIMITS) or set(order) != set(STAGE_LIMITS):
                raise ValueError('Preserve all finite recovery stage budgets')
            state['recovery_stage_order'] = order
            state['numerical_speed_policy'] = diagnosis
            write_json(state_path,state)
        # Keep exact command identity; only failed-point logs feed numerical planning.
        for stage in state['recovery_stage_order']:
            maximum = STAGE_LIMITS[stage]
            ledger = state['stages'].setdefault(stage, dict(status='pending', applicability=[]))
            if ledger['status'] in ('exhausted', 'not_applicable'):
                continue
            stage_attempts = [a for a in state['attempts'] if a.get('stage') == stage]
            while len(stage_attempts) < maximum:
                commands = [a['command'] for a in history if a.get('command')]
                logs = [(a['log'], failed_log_text(ROOT/a['log']))
                        for a in history if a.get('log') and internal_path(ROOT/a['log']).is_file()]
                previous_attempts = [a for a in history if a.get('command')]
                if stage in ('crossings', 'branches'):
                    previous_attempts = [a for a in state['attempts']
                                         if a.get('stage') in ('minima', 'crossings', 'branches')]
                previous = dict(status='failed', pid=None, attempts=previous_attempts)
                plans = ledger.setdefault('plans', [])
                known = {tuple(command) for command in commands}
                specs = [spec for plan in plans for spec in plan
                         if tuple(spec['command']) not in known]
                if not specs and plans and stage != 'branches':
                    break
                if not specs:
                    specs, diagnosis = recovery_commands(stage, workspace, point['mass'], primary,
                                                          previous, logs, commands, speed_policy=True)
                    ledger['applicability'].append(diagnosis)
                    specs = specs[:maximum-len(stage_attempts)]
                    if specs:
                        plans.append(specs)
                        # Freeze the entire finite batch before its first command.
                        # Resume cannot reseed the unfinished half from new logs.
                        write_json(state_path, state)
                if not specs:
                    ledger['status'] = 'not_applicable' if not stage_attempts else 'exhausted'
                    break
                # Persist the selected command list before launching. All commands
                # are finite and have distinct argv; a restart never resets budget.
                for spec in specs[:maximum-len(stage_attempts)]:
                    spec = dict(spec, stage=stage)
                    target = work/'recovery'/stage/spec['name']
                    attempt = integrated_attempt(point, workspace, target, spec, env,
                                                 state, state_path, progress)
                    stage_attempts.append(attempt)
                    history.append(attempt)
                    if attempt['passed']:
                        state.update(status='complete', finished=time.time(), accepted_attempt=attempt,
                            output=attempt['output'], output_sha256=attempt['output_sha256'],
                            validation=attempt['validation'], pid=None)
                        write_json(state_path, state)
                        return accepted_fit_record(point, state, str(state_path.relative_to(ROOT)))
                if stage != 'branches':
                    break  # Only conditional branches unlock later configurations.
            if ledger['status'] == 'pending':
                ledger['status'] = 'exhausted'
            write_json(state_path, state)
        state.update(status='failed', error='all_applicable_finite_methods_exhausted',
                     pid=None, finished=time.time())
        write_json(state_path, state)
        return state


def child_identity(pid):
    actual = resources.process_identity(pid)
    return dict(pid=pid, start_ticks=actual['start_ticks'] if actual else None)


def owned_point_path(path, output):
    if not Path(path).is_absolute():
        return None
    try:
        parts = Path(path).relative_to(output).parts
    except ValueError:
        return None
    if len(parts) < 2 or parts[0] not in EXPECTED_DISPLAY_COUNTS:
        return None
    if re.fullmatch(r'mStop\d+_mLSP\d+', parts[1]):
        return '/'.join(parts[:2])
    return None


def native_fit_process(arguments):
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    if executable in {'combine', 'text2workspace.py', 'text2workspace'}:
        return True
    return (executable.startswith('python') and len(arguments) > 1
            and Path(arguments[1]).name == 'text2workspace.py')


def assert_previous_child_exited(record, work):
    actual = resources.process_identity(record.get('pid'))
    if actual is None:
        return
    started = record.get('start_ticks')
    if started is not None:
        owned = actual['start_ticks'] == started
    elif (command := record.get('command')) and len(command) > 1:
        # Older current receipts did not carry ticks. A foreign process with a
        # recycled PID is not an owner; the exact recorded argv is evidence.
        actual_argv = actual['arguments']
        owned = (len(actual_argv) >= len(command)
                 and actual_argv[1-len(command):] == command[1:]
                 and any(owned_point_path(arg, work.parents[1]) == '/'.join(work.parts[-2:])
                         for arg in actual_argv))
    else:
        owned = False
    if owned:
        raise RuntimeError('Previous point process is still alive; preserve it')


def native_fit_workers(output):
    """Identify live native processes by their exact canonical point paths."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            actual = resources.process_identity(int(proc.name))
            if not actual or not native_fit_process(actual['arguments']):
                continue
            args = actual['arguments']
            keys = {key for path in args if (key := owned_point_path(path, output))}
            if not keys:
                continue
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z' or int(fields[19]) != actual['start_ticks']:
                continue
            yield actual, keys, int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def fit_pool_resident_memory(output, active_keys):
    """Credit only measured native RSS, capped once per materialized future."""
    resident = {}
    for _, keys, rss in native_fit_workers(output):
        owned = keys & active_keys
        if len(owned) == 1:
            key = owned.pop()
            resident[key] = resident.get(key, 0) + max(0, rss)
    return sum(min(rss, POINT_RESERVATION) for rss in resident.values())


def fit_pool_capacity(workers, external, memory, disk_free, resident_memory=0):
    if memory < 24*GIB or disk_free < 80*GIB:
        return 0
    # Reserve 3 GiB for EVERY future, including those without subprocesses.
    # Only measured native RSS is credited back, never active*reservation.
    total_memory_slots = max(0, (memory+resident_memory-32*GIB)//POINT_RESERVATION)
    return max(0, min(workers, 70, 116-1-external, total_memory_slots))


@contextmanager
def canonical_controller_ownership(output):
    """Require the current flock and no surviving worker of this stage."""
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = output/'workflow_state.json'
        if state.is_file():
            record = read_json(state)
            started = record.get('start_ticks')
            live = resources.process_identity(record.get('pid'))
            if started is not None and live and live['start_ticks'] == started:
                raise RuntimeError('Previous limit controller is still alive; preserve it')
        worker = next(native_fit_workers(output), None)
        if worker is not None:
            actual, _, _ = worker
            raise RuntimeError(f"Previous limit worker {actual['pid']} is still alive; preserve it")
        yield


@contextmanager
def accepted_analysis_root(manifest, output):
    """Bind accepted relative artifacts to their manifest's root, not code root.

    A canonical installation may read an immutable nested campaign. The
    package's internal_path still bounds every path to that installation; no
    manifest, card or existing receipt is rewritten or copied for this handoff.
    """
    global ROOT
    from TROTASR.workflows import limit_recovery
    manifest, output = internal_path(manifest), internal_path(output)
    if (manifest.name != 'manifest.json' or manifest.parent.name != 'grid'
            or manifest.parents[2].name != 'stats'
            or output != manifest.parents[1]/'limits'):
        raise ValueError('Use one accepted analysis-root/stats/run/{grid,limits} handoff')
    analysis_root = internal_path(manifest.parents[3])
    previous, previous_recovery = ROOT, limit_recovery.ROOT
    ROOT = limit_recovery.ROOT = analysis_root
    try:
        yield analysis_root
    finally:
        ROOT, limit_recovery.ROOT = previous, previous_recovery


def adopt_cr_linked_model(manifest, output):
    """Advance the sole Run 3 fit stages; preserve the prior scientific results."""
    from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION, donor_workspace
    run_root = ROOT/'stats/run3'
    if manifest != run_root/'grid/manifest.json' or output != run_root/'limits':
        raise ValueError('The Photon CR model applies only to the canonical Run 3 stages')
    source = read_json(manifest)
    points, _ = ready_fit_points(source)
    if not source.get('era_merged') or source.get('sr_data_blinded') is not True:
        raise ValueError('Accepted era-merged blinded grid required')
    donor = donor_workspace(run_root)
    impacts = run_root/'impacts'
    reference = ROOT/'stats/references/era_merged_recoil'
    with ExitStack() as locks:
        def hold(path):
            lock = locks.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        hold(run_root/'model_handoff.lock')
        marker = read_json(output/'model.json')
        if marker.get('model_revision') == MODEL_REVISION:
            return
        if marker.get('canonical_model') != 'era_merged' or marker.get('model_revision'):
            raise ValueError('Unknown current Run 3 result')
        hold(output/'controller.lock')
        hold(output/'T2tt/mStop1200_mLSP500/point.lock')
        hold(impacts/'impact.lock')
        prior = read_json(output/'workflow_state.json')
        if (prior.get('completed') != len(points) or prior.get('active') or prior.get('pid')
                or not read_json(impacts/'publication.json').get('complete')):
            raise RuntimeError('Preserve the prior official result until its stages are complete')
        rows = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','comm=,args='],text=True)
        if any((line.lstrip().startswith('combine ') or 'text2workspace.py ' in line)
               and (str(output) in line or str(impacts) in line) for line in rows.splitlines()):
            raise RuntimeError('A native worker still owns a prior result path')
        with tempfile.TemporaryDirectory(prefix='.model-', dir=run_root) as directory:
            staged = Path(directory)/'limits'
            staged.mkdir()
            write_json(staged/'model.json', dict(
                canonical_model='era_merged', model_revision=MODEL_REVISION,
                adopted=time.time(), observed_sr_used=False, signal_uncertainties=False,
                photon_cr_workspace=str(donor.relative_to(ROOT)),
                photon_cr_workspace_mtime_ns=donor.stat().st_mtime_ns,
                photon_cr_workspace_size=donor.stat().st_size,
                prior_results=str(reference.relative_to(ROOT)),
                status='fits_pending', input_check_scope=INPUT_CHECK_SCOPE))
            recipes = output/'successful_recipes.json'
            if recipes.is_file():
                shutil.copyfile(recipes, staged/recipes.name)
            reference.mkdir()
            moved = []
            try:
                for path in (output, impacts):
                    target = reference/path.name
                    path.rename(target)
                    moved.append((target,path))
                staged.rename(output)
            except OSError:
                for old, current in reversed(moved):
                    old.rename(current)
                reference.rmdir()
                raise


def run_integrated_grid(manifest, output, workers):
    if not 1 <= workers <= 70:
        raise ValueError('Use 1..70 point futures within aggregate120/downstream116')
    if sys.platform != 'linux':
        raise RuntimeError('Expected fits run only in the existing hep2 runtime')
    manifest, output = internal_path(manifest), internal_path(output)
    run_root = ROOT/'stats/run3'
    if manifest != run_root/'grid/manifest.json' or output != run_root/'limits':
        raise ValueError('Expected fitting uses only the canonical Run 3 grid and limits paths')
    source = read_json(manifest)
    points, selection = ready_fit_points(source)
    with ExitStack() as locks:
        # Share only the current model-transition lock. Acquire it before
        # opening controller.lock so adoption cannot relocate the locked inode.
        model_lock = locks.enter_context((run_root/'model_handoff.lock').open('a'))
        fcntl.flock(model_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=True)
        locks.enter_context(canonical_controller_ownership(output))
        model_marker = output/'model.json'
        if model_marker.is_file():
            model_record = read_json(model_marker)
            if model_record.get('canonical_model') != 'era_merged':
                raise ValueError('Unknown canonical limit model')
        else:
            if any(p.name != 'controller.lock' for p in output.iterdir()):
                raise ValueError('Existing limits are not identified by the current model marker')
            model_record = dict(canonical_model='era_merged', observed_sr_used=False,
                                input_check_scope=INPUT_CHECK_SCOPE)
            write_json(model_marker, model_record)
        revision = model_record.get('model_revision')
        if revision:
            from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION, donor_workspace
            if revision != MODEL_REVISION:
                raise ValueError('Unknown canonical CR-linked model revision')
            donor = donor_workspace(run_root).stat()
            if (donor.st_mtime_ns, donor.st_size) != (model_record['photon_cr_workspace_mtime_ns'],
                                                       model_record['photon_cr_workspace_size']):
                raise ValueError('The adopted CR-only source changed')
            points = [dict(p, model_revision=revision) for p in points]
        results = {}
        pending = []
        state_path = output/'workflow_state.json'
        previous_failures = set(read_json(state_path).get('failed', ())) if state_path.is_file() else set()
        for point in points:
            key = point['model']+'/'+point['mass']
            saved = saved_point_result(point, output)
            if saved is not None:
                results[key] = saved
            else:
                pending.append(point)
        # Existing failures go first; there are no idle recovery-only slots.
        pending.sort(key=lambda p: (p['model']+'/'+p['mass'] not in previous_failures,
                                    bool(revision) and (p['model'],p['mass']) != ('T2tt','mStop1200_mLSP500'),
                                    not (output/p['model']/p['mass']/'state.json').exists(),
                                    p['model'], p['mass']))
        env = runtime(output) if pending else None
        stopped = threading.Event()
        old_handler = signal.signal(signal.SIGINT, lambda *_: stopped.set())
        progress = {}
        started = time.time()
        controller = child_identity(os.getpid())

        def snapshot(active, external=0, memory=0, admission_limit=0, final=False):
            complete = sum(r.get('status') == 'complete' for r in results.values())
            official = source.get('official_result', False) and (not revision or complete == len(points))
            failed = [k for k, r in results.items() if r.get('status') != 'complete']
            phase = 'fits_complete_pending_plots' if complete == len(points) else (
                'needs_attention' if final and failed else 'drained' if final else
                'draining' if stopped.is_set() else 'running')
            record = dict(selection, canonical_limits_pipeline=True,
                canonical_model='era_merged',
                model_revision=revision,
                official_result=official,
                analysis_root=str(ROOT),
                status=phase, pid=None if final else controller['pid'],
                start_ticks=None if final else controller['start_ticks'],
                started=started, updated=time.time(),
                manifest=str(manifest.relative_to(ROOT)), input_check_scope=INPUT_CHECK_SCOPE,
                completed=complete, remaining=len(points)-complete, failed=failed,
                active_primary=[k for k, v in progress.items() if v['phase'] == 'primary'],
                active_recovery=[k for k, v in progress.items() if v['phase'] != 'primary'],
                active=len(active), pending=len(pending),
                exclusive_controller_ownership=True, external_slots=external,
                available_memory_bytes=memory, admission_limit=admission_limit, results=results,
                requested_workers=workers, worker_memory_reservation_bytes=POINT_RESERVATION,
                full_workflow_complete=False, observed_sr_used=False)
            write_json(state_path, record)
            if final:
                recipes={}
                for key,result in results.items():
                    if result.get('status') != 'complete':
                        continue
                    accepted=result['accepted_attempt']
                    command=list(accepted['command'])
                    command[command.index('-M')+2]=str(output/key/'workspace.root')
                    recipes[key]=dict(name=accepted['name'],command=command)
                write_json(output/'successful_recipes.json',dict(
                    schema='canonical_limit_successful_numerical_recipes_v1',points=recipes,
                    numerical_settings_only=True,fit_values_reused=False))
            for model in EXPECTED_DISPLAY_COUNTS:
                summary = collect(points, results, model, source['scope'])
                summary.update(selection, canonical_limits_pipeline=True,
                    canonical_model='era_merged',
                    official_result=official, model_revision=revision)
                write_json(output/model/'expected_limits.json', summary)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}
                snapshot(active)
                while active or (pending and not stopped.is_set()):
                    external, memory = resources.resources()
                    active_keys = {p['model']+'/'+p['mass'] for p in active.values()}
                    resident = fit_pool_resident_memory(output, active_keys) if active_keys else 0
                    limit = fit_pool_capacity(workers, external, memory, shutil.disk_usage(output).free,
                                              resident_memory=resident)
                    while pending and not stopped.is_set() and len(active) < limit:
                        point = pending.pop(0)
                        key = point['model']+'/'+point['mass']
                        progress[key] = dict(phase='primary', pid=None)
                        def report(phase, pid, key=key):
                            progress[key] = dict(phase=phase, pid=pid)
                        active[pool.submit(run_integrated_point, point, output, env, report)] = point
                    done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED) if active else (set(), set())
                    for future in done:
                        point = active.pop(future)
                        key = point['model']+'/'+point['mass']
                        progress.pop(key, None)
                        try:
                            results[key] = future.result()
                        except Exception as error:
                            results[key] = dict(point, status='needs_attention', error=repr(error))
                            # Shared environment/ownership failure stops NEW admission;
                            # healthy active points drain and retain their outputs.
                            stopped.set()
                    snapshot(active, external, memory, limit)
                    if not active and pending and not stopped.is_set():
                        time.sleep(5)
                snapshot(active, final=True)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        return 0 if sum(r.get('status') == 'complete' for r in results.values()) == len(points) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'stats/run3/grid/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'stats/run3/limits')
    p.add_argument('--workers',type=int,default=70)
    p.add_argument('--resume', action='store_true',
        help='Reuse accepted point receipts and continue unstarted finite commands (also the default)')
    p.add_argument('--plot-range-only',action='store_true',
        help='Fit only the approved 895 displayed points (also the default)')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--workspace-step', choices=('prepare', 'link'), help=argparse.SUPPRESS)
    modes.add_argument('--adopt-cr-link', action='store_true',
                       help='Start the approved Run 3 Photon CR model with nominal signal')
    p.add_argument('--workspace-input', type=Path, help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.workspace_step:
        from TROTASR.workflows.cr_linked_workspace import workspace_step
        run_root = ROOT/'stats/run3'
        directory = internal_path(a.output)
        if (sys.platform != 'linux' or run_root/'limits' not in directory.parents
                or not directory.is_dir() or not directory.name.startswith('.workspace-')
                or a.workspace_input is None):
            p.error('Use a temporary directory owned by the canonical Run 3 limit point')
        source = internal_path(a.workspace_input)
        if a.workspace_step == 'link' and source != directory/'native.root':
            p.error('The native workspace must belong to the same point conversion')
        workspace_step(a.workspace_step, source, directory, run_root)
        raise SystemExit(0)
    if a.adopt_cr_link:
        if sys.platform != 'linux' or not 1 <= a.workers <= 70:
            p.error('Start the canonical hep2 pool with 1–70 point futures')
        adopt_cr_linked_model(internal_path(a.manifest), internal_path(a.output))
    raise SystemExit(run_integrated_grid(a.manifest,a.output,a.workers))
