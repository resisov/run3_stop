"""Nominal expected Combine grid with the adopted finite crossing-checked policy.

No cards, ROOT files, fit values, payloads or analysis modules outside TROTASR
are read. CMSSW is a separately identified numerical software runtime. Plotting
is a later stage; completed fits do not imply complete AN deliverables.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager
import errno
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
from TROTASR.utils.selection_contract import require_current_llcr
from TROTASR.utils.io import read_json, write_json
from TROTASR.workflows.limit_validation import REL_ACC, ABS_ACC
from TROTASR.workflows.limit_serialization import validate_saved_limit
from TROTASR.workflows import resources

FIELDS = {.025:'expected_m2', .16:'expected_m1', .5:'expected', .84:'expected_p1', .975:'expected_p2'}
ATTEMPTS = (('bisection_s0', .1, 1.), ('bisection_s0_refined', .01, 2.))
GIB = 1024**3
POINT_RESERVATION = 3*GIB
NEW_SIGNAL_PRODUCTS = 'new_signal'
LLCR_MTB_PRODUCTS = Path(NEW_SIGNAL_PRODUCTS)/'llcr_mtb'
LLCR_MTB_FIELD = 'highdm_llcr_mtb_min_gev'
POINT_IDENTITY = ('model', 'mass', 'model_revision', LLCR_MTB_FIELD)


def run3_product_roots(run_root):
    return run_root, run_root/NEW_SIGNAL_PRODUCTS, run_root/'top_pt'


def run3_grid_products(manifest, run_root):
    """Resolve only the published canonical grid roots, not arbitrary siblings."""
    return next((root for root in run3_product_roots(run_root)
                 if manifest == root/'grid/manifest.json'), None)


def require_llcr_mtb_grid(record):
    from TROTASR.utils.paths import ERAS
    if (record.get(LLCR_MTB_FIELD) != 175.0
            or record.get('signal_templates_nominal_only') is not True
            or record.get('official_result') is not False
            or record.get('sr_data_blinded') is not True
            or not isinstance(record.get('signal_input_revision'), str)
            or not record['signal_input_revision']
            or set(record.get('replacement_signal_sources', {})) != set(ERAS)):
        raise ValueError('The scoped 175 GeV LLCR grid requires all six blinded nominal-signal inputs')


def require_llcr_mtb_model(record, signal_revision=None):
    from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION
    if (record.get('canonical_model') != 'era_merged'
            or record.get('model_revision') != MODEL_REVISION
            or record.get(LLCR_MTB_FIELD) != 175.0
            or record.get('signal_uncertainties') is not False
            or not isinstance(record.get('signal_input_revision'), str)
            or not record['signal_input_revision']
            or (signal_revision is not None and record['signal_input_revision'] != signal_revision)
            or record.get('observed_sr_used') is not False):
        raise ValueError('The scoped 175 GeV LLCR fit requires the adopted blinded Photon CR model')


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
    from TROTASR.utils.cr_binning import N_BINS
    expected_bins = N_BINS if record.get('era_merged') else N_BINS * len(record_years(record))
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


def convert_workspace(card, workspace, record, state_path, env, stream, lock_fd):
    from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION, TOP_PT_MODEL_REVISION

    def execute(command, cwd):
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        record.update(child_identity(proc.pid), phase='workspace', command=command)
        write_json(state_path, record)
        return proc.wait()

    convert = ['text2workspace.py', str(card), '-m', '120', '-o', str(workspace), '--verbose', '0']
    revision = record.get('model_revision')
    if revision is None:
        return execute(convert, card.parent)
    if revision not in (MODEL_REVISION, TOP_PT_MODEL_REVISION):
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


def verify_card(point):
    from TROTASR.utils.paths import ERAS, EARLY_ERAS
    from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
    card = internal_path(ROOT/point['card'])
    if not card.is_file() or not card.stat().st_size:
        raise ValueError('Card missing or empty: '+str(card))
    meta = read_json(card.with_suffix('.manifest.json'))
    mass = re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass'])
    if (not mass or point['model'] not in ('T2tt','T2tb','T2bW') or meta['cr_only']
        or not meta['sr_data_blinded']
        or tuple(map(str,meta['years'])) not in (('2024','2025'), EARLY_ERAS, ERAS)
        or (tuple(map(str,meta['years'])) == ERAS and not require_run3_systematics_policy(meta))
        or meta['auto_mc_stats'] != [10,1,1]
        or (meta['topology'],meta['mStop'],meta['mLSP']) != (point['model'],int(mass[1]),int(mass[2]))):
        raise ValueError('Card/point/blinding contract mismatch')
    for source in meta['inputs']:
        folder = internal_path(ROOT/source['path'])
        manifest = folder/'model_manifest.json'
        record=read_json(manifest)
        if record.get('years', ['2024', '2025']) != list(map(str, meta['years'])):
            raise ValueError('Card and template target eras differ')
        if (record['status']!='templates_ready' or not record['root_integrity_checked']
            or record.get('scope') not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production')):
            raise ValueError('Template ROOT not verified')
        verify_combined_grid(record)
        template_folder = internal_path(ROOT/record['template_directory']) if record.get('template_directory') else folder
        for name in record['files']:
            template = internal_path(template_folder/name)
            if not template.is_file() or not template.stat().st_size:
                raise ValueError('Template file missing or empty: '+name)
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


def reconcile_interrupted_attempts(record, state_path):
    """Recover committed validation or keep an interrupted command consumed."""
    changed = False
    for attempt in record.get('attempts', ()):
        if attempt.get('exit_code') is not None or attempt.get('interruption'):
            continue
        saved = {}
        if attempt.get('log'):
            folder = (ROOT / attempt['log']).resolve().parent
            if folder.is_relative_to(state_path.parent):
                try:
                    candidate = read_json(folder/'validation.json')
                    if isinstance(candidate, dict):
                        saved = candidate
                except (FileNotFoundError, ValueError):
                    pass
        fields = ('name', 'command', 'log', 'output')
        if (type(saved.get('exit_code')) is int
                and all(saved.get(field) == attempt.get(field) for field in fields)):
            attempt.update(saved)
        else:
            attempt.update(passed=False, interruption='execution ended without a recorded exit status')
        changed = True
    if changed:
        record.update(pid=None)
        write_json(state_path, record)


def prior_range_plan(points, run_root):
    """Use the same point's last accepted upper expected quantile as a numerical hint."""
    sources=(run_root/'limits/workflow_state.json',run_root.parent/'references/era_merged_recoil/limits/workflow_state.json')
    target={p['model']+'/'+p['mass'] for p in points}; plan={}
    for path in sources:
        state=read_json(path)
        if state.get('observed_sr_used') is not False:
            raise ValueError('Prior is not a blinded expected-limit workflow')
        for key,row in state['results'].items():
            if key not in target or key in plan or row.get('status')!='complete':continue
            q={float(k):float(v) for k,v in row['validation']['quantiles'].items()}
            if set(q)!=set(FIELDS):continue
            vals=[q[k] for k in sorted(q)]
            if not all(math.isfinite(v) and v>0 for v in vals) or not all(a<b for a,b in zip(vals,vals[1:])):continue
            # A historical limit can widen the starting interval. Only real
            # same-model conditional NLL rows can justify lowering its upper bound.
            proposal=1.5*q[.975]
            seed=max(20.,proposal)
            if seed*2>100000:raise ValueError('Prior exceeds the finite range ceiling: '+key)
            plan[key]=dict(rmin=0.,rmax=seed,prior_proposal=proposal,prior_p975=q[.975],
                           source=str(path.relative_to(ROOT)))
    if set(plan)!=target:raise ValueError('Incomplete same-point prior range coverage')
    return plan


def run_primary_point(point, output, env, lock_fd):
    """Run the two primary configurations under the caller's point lock."""
    from TROTASR.workflows.limit_recovery import optimize_saved_command, validate_attempt

    card = ready_card(point)
    work = output/point['model']/point['mass']
    work.mkdir(parents=True,exist_ok=True)
    state_path=work/'state.json'
    record=read_json(state_path) if state_path.exists() else dict(point,status='running',started=time.time(),attempts=[])
    if any(record.get(key) != point.get(key) for key in POINT_IDENTITY):
        raise ValueError('Preserve old fit for a different point or statistical model')
    attempts = record['attempts']
    reconcile_interrupted_attempts(record, state_path)
    for accepted in reversed(attempts):
        # Older primary receipts used the printed r literally. Check the saved
        # ROOT/log pair before paying for another fit, under the same point lock.
        if (accepted.get('exit_code') == 0 and not accepted.get('passed')
                and not accepted.get('primary_serialization_checked')
                and accepted.get('command') and accepted.get('log') and accepted.get('output')
                and accepted.get('validation', {}).get('crossings')):
            passed, evidence = validate_attempt(point, accepted)
            accepted.update(passed=passed, validation=evidence, primary_serialization_checked=True)
            write_json(state_path, record)
        if accepted.get('exit_code') == 0 and accepted.get('passed'):
            record.update(status='complete', finished=record.get('finished') or time.time(),
                          pid=None, output=accepted['output'],
                          validation=accepted['validation'], accepted_attempt=accepted)
            write_json(state_path, record)
            return record
    workspace=work/'workspace.root'
    if not record.get('workspace_valid'):
        rc = record.get('workspace_conversion_exit')
        if rc is None:
            if workspace.exists() or record.get('phase') == 'workspace':
                raise ValueError('Interrupted workspace conversion has no recorded exit status')
            with (work/'workspace.log').open('a') as stream:
                rc=convert_workspace(card,workspace,record,state_path,env,stream,lock_fd)
            record.update(workspace_conversion_exit=rc, pid=None)
            write_json(state_path, record)
        if rc or not workspace.is_file() or workspace.stat().st_size<1000:
            record.update(status='failed',error='workspace_conversion_failed',exit_code=rc,finished=time.time(),pid=None)
            write_json(state_path,record);return record
        import uproot
        with uproot.open(workspace) as root:
            if 'w' not in root:raise ValueError('Missing RooWorkspace w')
        record.update(workspace_valid=True,pid=None)
        write_json(state_path,record)
    elif not workspace.is_file():
        raise ValueError('Previously verified workspace missing')
    completed_attempts={a['name'] for a in attempts}
    for label, tolerance, scale in ATTEMPTS:
        if point.get('prior_range'):
            label='prior_range_'+label
            record['prior_range']=point['prior_range']
        if label in completed_attempts:continue
        here=work/label
        rmax = float(point.get('rmax_seed',20.))*scale
        command=fit_command(workspace,point['mass'],rmax,tolerance)
        log=here/'combine.log'
        result=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
        attempt=dict(name=label,tolerance=tolerance,passed=False,exit_code=None,
            log=str(log.relative_to(ROOT)), output=str(result.relative_to(ROOT)))
        if (record.get('current_attempt') == label
                or (here.is_dir() and any(path != here/'tmp' for path in here.iterdir()))):
            if record.get('current_attempt') == label and record.get('command'):
                attempt['command'] = record['command']
            elif (here/'validation.json').is_file():
                attempt['command'] = read_json(here/'validation.json').get('command')
            attempts.append(attempt)
            reconcile_interrupted_attempts(record, state_path)
            if attempt.get('exit_code') == 0 and attempt.get('passed'):
                record.update(status='complete', pid=None, output=attempt['output'],
                              validation=attempt['validation'], accepted_attempt=attempt)
                write_json(state_path, record)
                return record
            continue
        known = [a['command'] for a in attempts if a.get('command') and not (
            point.get('prior_range') and a.get('interruption') and a['name'] in {name for name,_,_ in ATTEMPTS}
            and a.get('exit_code') is None)]
        saved_logs = [(a['log'], failed_log_text(ROOT/a['log']))
                      for a in attempts if a.get('log')
                      and internal_path(ROOT/a['log']).is_file()]
        command, speed_evidence = optimize_saved_command(
            workspace, point['mass'], command, saved_logs, known, attempts=attempts)
        if command in known:
            continue
        here.mkdir(exist_ok=True)
        (here/'tmp').mkdir(exist_ok=True)
        attempt['command'] = command
        if speed_evidence is not None:
            attempt['numerical_speed'] = speed_evidence
        attempts.append(attempt)
        record.update(phase='fit', command=command, current_attempt=label, pid=None)
        write_json(state_path, record)
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=here,env=dict(env,TMPDIR=str(here/'tmp')),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
            record.update(child_identity(proc.pid))
            write_json(state_path,record)
            rc=wait_for_inference(proc,log)
        attempt['exit_code'] = rc
        record['pid'] = None
        write_json(state_path, record)
        passed,evidence=validate_saved_limit(result,log,tolerance) if rc==0 else (False,dict(exit_code=rc))
        ready_card(point)
        attempt.update(passed=passed, exit_code=rc, validation=evidence, primary_serialization_checked=rc==0)
        write_json(here/'validation.json',attempt)
        record.update(pid=None)
        if passed:
            record.update(status='complete',finished=time.time(),output=str(result.relative_to(ROOT)),
                validation=evidence)
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
    require_current_llcr(record=source)
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


def replacement_card(point, product_root, signal_revision, *, llcr_mtb=False):
    """Accept a small replacement-card receipt without reopening inherited ROOT templates."""
    card = ready_card(point)
    if card.parent != product_root/'grid/datacards' or point.get('fit_ready') is not True:
        raise ValueError('The replacement grid must describe new signal cards only')
    receipt = read_json(card.with_suffix('.manifest.json'))
    if receipt.get('sr_data_blinded') is not True:
        raise ValueError('The replacement signal card must preserve SR blinding')
    if llcr_mtb and (receipt.get(LLCR_MTB_FIELD) != 175.0
                     or receipt.get('signal_templates_nominal_only') is not True
                     or receipt.get('signal_input_revision') != signal_revision
                     or receipt.get('fit_ready') is not True):
        raise ValueError('The card is not bound to the scoped LLCR fit model: '+str(card))
    return card


def accepted_fit_record(point, record, origin):
    """Reuse a completed receipt. Only NEW fits are opened/numerically checked."""
    if record.get('status') not in ('complete', 'verified_not_promoted', 'existing_verified'):
        return None
    identity = record.get('point', record)
    for name in POINT_IDENTITY:
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
    paths = dict(output=attempt['output'])
    if attempt.get('log'):
        paths['log'] = attempt['log']
    if record.get('workspace'):
        paths['workspace'] = record['workspace']
    # Recovery receipts refer to the primary workspace, not a duplicate copy.
    if 'workspace' not in record:
        command = attempt.get('command', [])
        if '-M' in command and len(command) > command.index('-M')+2:
            paths['workspace'] = command[command.index('-M')+2]
    product = internal_path(ROOT/paths['output'])
    if not product.is_file() or product.stat().st_size <= 0:
        raise FileNotFoundError('Accepted fit ROOT is missing or empty: '+str(product))
    ancillary = {}
    for name in ('log', 'workspace'):
        if name in paths and (path := (ROOT/paths[name]).resolve()).is_relative_to(ROOT):
            ancillary[name] = str(path)
    paths = dict(ancillary, output=str(product))
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
            for name in POINT_IDENTITY:
                if identity.get(name) != point.get(name):
                    raise ValueError('Different point/card receipt retained: '+str(path))
            records.append((stage, path, record))
    return records


def saved_point_result(point, output):
    if 'model_revision' not in point and (output/'model.json').is_file():
        model = read_json(output/'model.json')
        revision = model.get('model_revision')
        if revision:
            point = dict(point, model_revision=revision)
        if LLCR_MTB_FIELD in model:
            require_llcr_mtb_model(model)
            point = dict(point, **{LLCR_MTB_FIELD: model[LLCR_MTB_FIELD]})
    for stage, path, record in point_receipts(point, output):
        accepted = accepted_fit_record(point, record, str(path.relative_to(ROOT)))
        if accepted is not None:
            return accepted
    return None


def integrated_attempt(point, target, spec, env, state, state_path, progress, lock_fd):
    from TROTASR.workflows.limit_recovery import validate_attempt
    log = target/'combine.log'
    product = target/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
    attempt = dict(spec, log=str(log.relative_to(ROOT)), output=str(product.relative_to(ROOT)),
                   started=time.time(), exit_code=None, passed=False)
    state['attempts'].append(attempt)
    write_json(state_path, state)
    if target.is_dir() and any(path != target/'tmp' for path in target.iterdir()):
        reconcile_interrupted_attempts(state, state_path)
        return attempt
    target.mkdir(parents=True, exist_ok=True)
    (target/'tmp').mkdir(exist_ok=True)
    with log.open('x') as stream:
        process = subprocess.Popen(spec['command'], cwd=target, env=dict(env, TMPDIR=str(target/'tmp')),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        state.update(child_identity(process.pid), phase=spec['stage'], command=spec['command'])
        write_json(state_path, state)
        progress(spec['stage'], process.pid)
        attempt['exit_code'] = wait_for_inference(process,log)
    attempt.update(finished=time.time())
    state['pid'] = None
    write_json(state_path, state)
    passed, evidence = validate_attempt(point, attempt)
    attempt.update(passed=passed, validation=evidence)
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
    try:
        accepted = read_json(source).get('points',{}).get(key)
        if accepted is None:
            return [], dict(status='unavailable', reason='no_successful_same_point_recipe')
        name = accepted['name']
        command = list(accepted['command'])
        if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name)
                or not all(isinstance(value, str) for value in command)):
            raise ValueError('Invalid historical command or recipe name')
        if (Path(command[0]).name != 'combine'
                or command[command.index('-M')+1] != 'AsymptoticLimits'
                or command[command.index('-n')+1] != '_'+point['mass']
                or command[command.index('--run')+1] != 'blind' or '--picky' not in command
                or any(flag in command for flag in ('--freezeParameters','--setParameters',
                        '--setParameterRanges', '--snapshotName', '--noFitAsimov', '--toysFrequentist'))):
            raise ValueError('Historical recipe changed the expected-limit dataset/model')
        command[command.index('-M')+2] = str(workspace)
        rmax = float(command[command.index('--rMax')+1])
        tolerance = float(command[command.index('--cminDefaultMinimizerTolerance')+1])
        if not 0 < rmax <= 100000 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Invalid historical numerical seed')
        algorithm = command[command.index('--minosAlgo')+1]
        adapters = {'new': 'native_profiled', 'stepping': 'stepping_full5'}
        if (algorithm not in ('bisection', *adapters)
                or (algorithm in adapters and 'MINIMIZER_no_analytic' not in command)
                or (algorithm == 'stepping' and '--strictBounds' not in command)):
            raise ValueError('Historical recipe lacks a supported numerical algorithm')
    except (AttributeError, FileNotFoundError, IndexError, KeyError, TypeError, ValueError) as error:
        return [], dict(status='inapplicable', reason=str(error))
    specs = [dict(name='prior_success_'+name, command=command, rmax=rmax, tolerance=tolerance)]
    if algorithm in adapters:
        specs[0]['run3_numerical_adapter'] = adapters[algorithm]
    for algorithm, strategy in (('Migrad',0),('Combined',1)):
        spec = dict(name='prior_success_'+algorithm.lower()+'_s'+str(strategy)+'_calls2m',
            algorithm=algorithm, strategy=strategy, tolerance=.1, precision=None,
            max_calls=2000000, rmax=rmax)
        specs.append(dict(spec, command=profile_recovery_command(workspace,point['mass'],spec)))
    known = set()
    for previous in previous_commands:
        normalized=list(previous)
        if '-M' in normalized and normalized.index('-M')+2 < len(normalized):
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
            reconcile_interrupted_attempts(record, path)
        progress('primary', None)
        primary = next((r for stage, _, r in records if stage == 'primary'), None)
        if primary is None or primary.get('error') != 'finite_attempts_exhausted':
            primary = run_primary_point(point, output, env, lock.fileno())
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
            if attempt.get('exit_code') != 0 or not attempt.get('output'):
                continue
            identity = json.dumps([attempt['output'], attempt.get('log')])
            if attempt.get('passed'):
                passed, evidence = True, attempt['validation']
            else:
                if identity in checked:
                    continue
                if not attempt.get('command') or not attempt.get('log') or not (
                        ROOT/attempt['log']).is_file():
                    passed, evidence = False, dict(reason='saved_command_or_log_unavailable')
                else:
                    passed, evidence = validate_attempt(point, attempt)
                checked[identity] = dict(passed=passed, validation=evidence, output=attempt['output'])
            if passed:
                accepted = dict(attempt, passed=True, validation=evidence)
                state.update(status='complete', finished=time.time(), accepted_attempt=accepted,
                    output=accepted['output'],
                    validation=evidence, pid=None)
                write_json(state_path, state)
                return accepted_fit_record(point, state, str(state_path.relative_to(ROOT)))
            write_json(state_path, state)
        # Try successful same-point numerical settings first, using this
        # model's workspace and the finite branch budget. Accepted points
        # return before this lookup; fitted values are not reused.
        prescription = state.get('historical_success_prescription')
        if not isinstance(prescription, dict) or not isinstance(prescription.get('plans'), list):
            specs, diagnosis = historical_success_commands(point, workspace, output,
                [a['command'] for a in history if a.get('command')])
            prescription = dict(diagnosis, plans=specs)
            state['historical_success_prescription'] = prescription
            write_json(state_path, state)
        if (output == ROOT/'stats/run3/top_pt/limits' and
                (state.get('error') == 'all_applicable_finite_methods_exhausted' or
                 any(a.get('interruption') for a in history)) and
                state.get('terminal_reference_accuracy') != 'canonical_grid_default'):
            from TROTASR.workflows.limit_recipe_hints import terminal_reference_plans
            state['terminal_reference_plans'] = terminal_reference_plans(output, point, history)
            state['terminal_reference_accuracy'] = 'canonical_grid_default'
            write_json(state_path, state)
        branch_attempts = [a for a in state['attempts'] if a.get('stage') == 'branches']
        known = {tuple(a['command']) for a in history if a.get('command')}
        specs = [s for s in prescription['plans'] + state.get('terminal_reference_plans', [])
                 if tuple(s['command']) not in known]
        for spec in specs[:max(0, STAGE_LIMITS['branches']-len(branch_attempts))]:
            ledger = state['stages'].setdefault('branches',dict(status='pending',applicability=[],plans=[]))
            ledger.setdefault('plans',[]).append([spec])
            ledger['status'] = 'pending'
            write_json(state_path,state)
            attempt = integrated_attempt(point, work/'recovery/branches'/spec['name'],
                                         spec, env, state, state_path, progress, lock.fileno())
            history.append(attempt)
            if attempt['passed']:
                state.pop('error', None)
                state.update(status='complete',finished=time.time(),accepted_attempt=attempt,
                    output=attempt['output'],
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
                    attempt = integrated_attempt(point, target, spec, env, state, state_path,
                                                 progress, lock.fileno())
                    stage_attempts.append(attempt)
                    history.append(attempt)
                    if attempt['passed']:
                        state.update(status='complete', finished=time.time(), accepted_attempt=attempt,
                            output=attempt['output'],
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


def independent_benchmark_impact(output, args, cwd, parent):
    """A live impact child reads the shared workspace, but never owns its limit."""
    impacts = output.parent/'impacts'
    if (len(args) < 4 or Path(args[0]).name != 'combine' or args[1:3] != ['-M', 'MultiDimFit']
            or impacts not in cwd.parents or '-d' not in args
            or args[args.index('-d')+1:args.index('-d')+2] != [str(output/'T2tt/mStop1200_mLSP500/workspace.root')]):
        return False
    receipt = impacts/'impact_status.json'
    if not receipt.is_file():
        return False
    record = read_json(receipt)
    owner = resources.process_identity(parent)
    return bool(owner and parent == record.get('controller_pid')
                and owner['start_ticks'] == record.get('controller_pid_start_ticks')
                and any(Path(a).name == 'combine_impacts.py' for a in owner['arguments']))


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
            if args[1:3] == ['-M', 'MultiDimFit'] and independent_benchmark_impact(
                    output, args, (proc/'cwd').resolve(strict=True), int(fields[1])):
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


def run_integrated_grid(manifest, output, workers, prior_ranges=False):
    if not 1 <= workers <= 70:
        raise ValueError('Use 1..70 point futures within aggregate120/downstream116')
    if sys.platform != 'linux':
        raise RuntimeError('Expected fits run only in the existing hep2 runtime')
    manifest, output = internal_path(manifest), internal_path(output)
    run_root = ROOT/'stats/run3'
    product_root = run3_grid_products(manifest, run_root)
    if product_root is None or output != product_root/'limits':
        raise ValueError('Expected fitting uses only the canonical Run 3 grid and limits paths')
    generation = product_root != run_root
    llcr_mtb = product_root == run_root/LLCR_MTB_PRODUCTS
    source = read_json(manifest)
    points, selection = ready_fit_points(source)
    signal_input_revision = source.get('signal_input_revision') if generation else None
    current_tt=product_root==run_root/'top_pt'
    if current_tt and (not source.get('tt_card_input') or source.get('signal_templates_nominal_only') is not True
                       or source.get('bins')!=208 or source.get('floating_normalization_parameters')!=44):
        raise ValueError('The new likelihood requires updated TT, nominal signals and the pooled CR tail')
    if generation and not current_tt and source.get('official_result') is not False:
        raise ValueError('The replacement grid must describe new signal cards only')
    if llcr_mtb:
        require_llcr_mtb_grid(source)
    with ExitStack() as locks:
        # Share only the current model-transition lock. Acquire it before
        # opening controller.lock so adoption cannot relocate the locked inode.
        model_lock = locks.enter_context((run_root/'model_handoff.lock').open('a'))
        fcntl.flock(model_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if generation:
            for point in points:
                replacement_card(point, product_root, signal_input_revision, llcr_mtb=llcr_mtb)
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
            if generation:
                from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION, donor_workspace
                baseline = read_json(run_root/'limits/model.json')
                donor = donor_workspace(run_root).stat()
                if (baseline.get('model_revision') != MODEL_REVISION
                        or baseline.get('canonical_model') != 'era_merged'
                        or donor.st_size <= 0):
                    raise ValueError('The accepted Photon CR workspace is required for new signals')
                model_record = {k: baseline[k] for k in ('canonical_model', 'model_revision')}
                model_record['photon_cr_workspace'] = str(donor_workspace(run_root).relative_to(ROOT))
                model_record.update(signal_input_revision=signal_input_revision,
                                    signal_uncertainties=False, observed_sr_used=False,
                                    input_check_scope=INPUT_CHECK_SCOPE)
            else:
                model_record = dict(canonical_model='era_merged', observed_sr_used=False,
                                    input_check_scope=INPUT_CHECK_SCOPE)
            if llcr_mtb:
                model_record[LLCR_MTB_FIELD] = 175.0
                require_llcr_mtb_model(model_record, signal_input_revision)
            if current_tt:
                from TROTASR.workflows.cr_linked_workspace import TOP_PT_MODEL_REVISION
                model_record.update(tt_card_input=source['tt_card_input'],bins=208,floating_normalization_parameters=44,
                                    model_revision=TOP_PT_MODEL_REVISION, photon_gnn_weights='current_native_prefit')
            write_json(model_marker, model_record)
        if llcr_mtb:
            require_llcr_mtb_model(model_record, signal_input_revision)
        revision = model_record.get('model_revision')
        if revision:
            from TROTASR.workflows.cr_linked_workspace import MODEL_REVISION, TOP_PT_MODEL_REVISION, donor_workspace
            if revision not in ((MODEL_REVISION, TOP_PT_MODEL_REVISION) if current_tt else (MODEL_REVISION,)):
                raise ValueError('Unknown canonical CR-linked model revision')
            if donor_workspace(run_root).stat().st_size <= 0:
                raise ValueError('The adopted CR-only workspace is empty')
            points = [dict(p, model_revision=revision) for p in points]
        if llcr_mtb:
            points = [dict(p, **{LLCR_MTB_FIELD: 175.0}) for p in points]
        if prior_ranges:
            if not current_tt:raise ValueError('Prior-range restart is scoped to the new TT grid')
            plan=prior_range_plan(points,run_root)
            write_json(output/'range_plan.json',dict(points=plan,prior_fit_values_reused=False,
                upper_margin=1.5,physical_lower=0.,smaller_bounds_require_current_measured_crossings=True))
            points=[dict(p,prior_range=plan[p['model']+'/'+p['mass']],
                         rmax_seed=plan[p['model']+'/'+p['mass']]['rmax']) for p in points]
        results = {}
        pending = []
        state_path = output/'workflow_state.json'
        previous_failures = set(read_json(state_path).get('failed', ())) if state_path.is_file() else set()
        for point in points:
            key = point['model']+'/'+point['mass']
            try:
                saved = saved_point_result(point, output)
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                results[key] = dict(point, status='needs_attention', error=repr(error))
                continue
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
                **({'signal_input_revision': signal_input_revision} if generation else {}),
                **({LLCR_MTB_FIELD: 175.0} if llcr_mtb else {}),
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
                if llcr_mtb:
                    summary[LLCR_MTB_FIELD] = 175.0
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
                            if isinstance(error, OSError) and (
                                    error.errno in (errno.EDQUOT, errno.ENOSPC, errno.EROFS,
                                        errno.EIO, errno.ENOMEM, errno.EMFILE, errno.ENFILE)
                                    or error.filename in ('combine', 'text2workspace.py')):
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
    p.add_argument('--prior-ranges',action='store_true',help='Start the current grid using blinded same-point prior range hints')
    p.add_argument('--resume', action='store_true',
        help='Reuse accepted point receipts and continue unstarted finite commands (also the default)')
    p.add_argument('--plot-range-only',action='store_true',
        help='Fit only the approved 895 displayed points (also the default)')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--seed-recovery-hints', action='store_true',
                       help='Publish same-point historical numerical settings only; launch no fits')
    modes.add_argument('--workspace-step', choices=('prepare', 'link'), help=argparse.SUPPRESS)
    modes.add_argument('--adopt-cr-link', action='store_true',
                       help='Start the approved Run 3 Photon CR model with nominal signal')
    p.add_argument('--workspace-input', type=Path, help=argparse.SUPPRESS)
    a=p.parse_args()
    require_current_llcr(a.manifest, a.output, a.workspace_input)
    if a.seed_recovery_hints:
        from TROTASR.workflows.limit_recipe_hints import seed_top_pt_recovery_hints
        if internal_path(a.output) != ROOT/'stats/run3/top_pt/limits':
            p.error('Command-only hints belong to the active top-pT grid')
        print(json.dumps(seed_top_pt_recovery_hints(publish=True)))
        raise SystemExit(0)
    if a.workspace_step:
        from TROTASR.workflows.cr_linked_workspace import workspace_step
        run_root = ROOT/'stats/run3'
        directory = internal_path(a.output)
        point_dir = directory.parent
        limits = point_dir.parent.parent
        if (sys.platform != 'linux' or limits not in {root/'limits' for root in run3_product_roots(run_root)}
                or not directory.is_dir() or not directory.name.startswith('.workspace-')
                or a.workspace_input is None):
            p.error('Use a temporary directory owned by the canonical Run 3 limit point')
        source = internal_path(a.workspace_input)
        if limits.parent == run_root/LLCR_MTB_PRODUCTS:
            model_record = read_json(limits/'model.json')
            require_llcr_mtb_model(model_record)
            if a.workspace_step == 'prepare':
                model, mass = point_dir.parent.name, point_dir.name
                if source.name != f'datacard_{model}_{mass}.txt':
                    p.error('Workspace input must match the scoped LLCR fit point')
                replacement_card(dict(card=str(source.relative_to(ROOT)), fit_ready=True),
                    limits.parent, model_record['signal_input_revision'], llcr_mtb=True)
        if a.workspace_step == 'link' and source != directory/'native.root':
            p.error('The native workspace must belong to the same point conversion')
        workspace_step(a.workspace_step, source, directory, run_root, products=limits.parent)
        raise SystemExit(0)
    if a.adopt_cr_link:
        if sys.platform != 'linux' or not 1 <= a.workers <= 70:
            p.error('Start the canonical hep2 pool with 1–70 point futures')
        adopt_cr_linked_model(internal_path(a.manifest), internal_path(a.output))
    raise SystemExit(run_integrated_grid(a.manifest,a.output,a.workers,prior_ranges=a.prior_ranges))
