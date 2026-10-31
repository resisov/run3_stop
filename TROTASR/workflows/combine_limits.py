"""Nominal expected Combine grid with the adopted finite crossing-checked policy.

No cards, ROOT files, fit values, payloads or analysis modules outside TROTASR
are read. CMSSW is a separately identified numerical software runtime. Plotting
is a later stage; completed fits do not imply complete AN deliverables.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager, nullcontext
import fcntl
import hashlib
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
import tarfile
import threading
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.limit_validation import validate, REL_ACC, ABS_ACC

def resources(*args, **kwargs):
    from TROTASR.workflows.nominal_campaign import resources as implementation
    return implementation(*args, **kwargs)


def capacity(*args, **kwargs):
    from TROTASR.workflows.nominal_campaign import capacity as implementation
    return implementation(*args, **kwargs)

FIELDS = {.025:'expected_m2', .16:'expected_m1', .5:'expected', .84:'expected_p1', .975:'expected_p2'}
ATTEMPTS = (('bisection_s0', .1, 1.), ('bisection_s0_refined', .01, 2.))


def verify_combined_grid(record):
    """The new scope is accepted only with the exact adopted model and grid."""
    if record.get('scope') not in ('full_combined_systematic_production', 'full_run3_hybrid_production'):
        return
    from TROTASR.workflows.build_nominal_grid import CR12_POLICY, HYBRID_SCOPE, require_hybrid_policy
    from TROTASR.workflows.diagnostic_common import require_revised_record, record_years
    require_revised_record(record)
    hybrid = record.get('scope') == HYBRID_SCOPE
    if hybrid and not require_hybrid_policy(record):
        raise ValueError('Missing hybrid treatment')
    policy = record.get('statistical_revision', record.get('contract', {}).get('statistical_revision'))
    points = record.get('points', [])
    counts = {m: sum(p['model'] == m for p in points) for m in ('T2tt', 'T2bW', 'T2tb')}
    expected_bins = 214 if record.get('era_merged') else 214 * len(record_years(record))
    if (policy != CR12_POLICY or record.get('bins') != expected_bins
            or counts != {'T2tt': 225, 'T2bW': 365, 'T2tb': 365}
            or len({(p['model'], p['mass']) for p in points}) != 955
            or record.get('sr_data_blinded') is not True
            or not record.get('systematic_scope')):
        raise ValueError('Combined grid does not match the approved CR12/955 contract')


class IntegrityCache:
    """Hash immutable shared templates once per run, not once per mass/fit.

    Every use still checks inode, size, mtime_ns and ctime_ns. Any change forces
    a fresh hash; a concurrent modification during hashing is rejected.
    """
    def __init__(self):
        self.entries = {}
        self.lock = threading.Lock()

    def __call__(self, path):
        path = internal_path(path)
        def fingerprint():
            s = path.stat()
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        with self.lock:
            before = fingerprint()
            previous = self.entries.get(path)
            if previous and previous[0] == before:
                return previous[1]
            digest = sha256(path)
            if fingerprint() != before:
                raise ValueError('File changed during integrity check: ' + str(path))
            self.entries[path] = before, digest
            return digest


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


def verify_card(point, digest=sha256):
    from TROTASR.utils.paths import ERAS, EARLY_ERAS
    from TROTASR.workflows.build_nominal_grid import require_hybrid_policy, require_run3_systematics_policy
    card = internal_path(ROOT/point['card'])
    if digest(card) != point['card_sha256']:
        raise ValueError('Card changed: '+str(card))
    meta = read_json(card.with_suffix('.manifest.json'))
    mass = re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass'])
    if (not mass or point['model'] not in ('T2tt','T2tb','T2bW') or meta['cr_only']
        or not meta['sr_data_blinded'] or meta['card_sha256']!=point['card_sha256']
        or tuple(map(str,meta['years'])) not in (('2024','2025'), EARLY_ERAS, ERAS)
        or (tuple(map(str,meta['years'])) == ERAS and not (
            require_hybrid_policy(meta) or require_run3_systematics_policy(meta)))
        or meta['auto_mc_stats'] != [10,1,1]
        or (meta['topology'],meta['mStop'],meta['mLSP']) != (point['model'],int(mass[1]),int(mass[2]))):
        raise ValueError('Card/point/blinding contract mismatch')
    if meta.get('scope') == 'sr_only_nominal_mcstat_comparison':
        from TROTASR.workflows.build_nominal_grid import comparison_card, COMPARISON_VARIANTS
        manifest = internal_path(ROOT/meta['comparison_manifest'])
        if digest(manifest) != meta['comparison_manifest_sha256']:
            raise ValueError('Comparison template manifest changed')
        record = read_json(manifest)
        if (record['scope'] != meta['scope'] or record['variant'] != meta['variant']
                or record['variant'] not in COMPARISON_VARIANTS or record['status'] != 'templates_ready'
                or not record['root_integrity_checked'] or not record['sr_data_blinded']
                or any(not c.startswith('SR_') for c in record['channels'])
                or meta['signal_channels'] != point['signal_channels']):
            raise ValueError('Invalid explicit SR-only comparison contract')
        test=record.get('medium_test_comparison')
        if test:
            from TROTASR.workflows.build_nominal_grid import test_comparison_bin_count
            test_comparison_bin_count(test)
            if meta.get('medium_test_comparison_sha256')!=test['contract_sha256']:
                raise ValueError('Card belongs to a different frozen test comparison')
        elif meta.get('medium_test_comparison_sha256') is not None:
            raise ValueError('Test card lost its frozen comparison provenance')
        from TROTASR.workflows.build_nominal_grid import CUTBASED_SCHEMA,cutbased_candidate_bin_count
        candidate=record.get('medium_candidate') or {}
        if candidate.get('schema')==CUTBASED_SCHEMA:
            cutbased_candidate_bin_count(candidate)
            if meta.get('medium_cutbased_candidate_sha256')!=candidate['contract_sha256']:
                raise ValueError('Card belongs to a different full-MC cut-based candidate')
        elif meta.get('medium_cutbased_candidate_sha256') is not None:
            raise ValueError('Cut-based card lost its full-MC provenance')
        from TROTASR.workflows.build_nominal_grid import MET_CATEGORY_SCHEMA,met_category_candidate_bin_count
        if candidate.get('schema')==MET_CATEGORY_SCHEMA:
            met_category_candidate_bin_count(candidate)
            if meta.get('medium_met_category_candidate_sha256')!=candidate['contract_sha256']:
                raise ValueError('Card belongs to a different MET-category selection candidate')
        elif meta.get('medium_met_category_candidate_sha256') is not None:
            raise ValueError('MET-category card lost its selection and parent-population provenance')
        for name, expected in record['files'].items():
            if digest(ROOT/name) != expected: raise ValueError('Comparison ROOT changed')
        if card.read_text() != comparison_card(record['channels'], point, card.parent):
            raise ValueError('Comparison card is not the nominal-MC/MCstat-only model')
        return card
    for source in meta['inputs']:
        folder = internal_path(ROOT/source['path'])
        manifest = folder/'model_manifest.json'
        if digest(manifest) != source['sha256']:
            raise ValueError('Template manifest changed')
        record=read_json(manifest)
        if record.get('years', ['2024', '2025']) != list(map(str, meta['years'])):
            raise ValueError('Card and template target eras differ')
        if (record['status']!='templates_ready' or not record['root_integrity_checked']
            or record.get('scope') not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production','full_run3_hybrid_production')):
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


def run_point(point, output, env, digest=sha256, input_check=None, lock_owned=False,
              numerical_speed=False):
    # Ready-artifact handoffs do not revisit upstream production/policy files.
    check_card = verify_card if input_check is None else input_check
    card = check_card(point, digest)
    comparison = (input_check is None and
        read_json(card.with_suffix('.manifest.json')).get('scope') == 'sr_only_nominal_mcstat_comparison')
    work = output/point['model']/point['mass']
    work.mkdir(parents=True,exist_ok=True)
    with (nullcontext() if lock_owned else (work/'point.lock').open('a')) as lock:
        if not lock_owned:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=work/'state.json'
        record=read_json(state_path) if state_path.exists() else dict(point,status='running',started=time.time(),attempts=[])
        if record['card_sha256']!=point['card_sha256']:
            raise ValueError('Preserve old fit for divergent card')
        if record['status']=='complete':
            result=internal_path(ROOT/record['output'])
            if sha256(result)!=record['output_sha256']:
                raise ValueError('Completed fit ROOT changed')
            matching = [a for a in record['attempts'] if a.get('exit_code') == 0
                and internal_path(ROOT/a['log']).parent == result.parent]
            if len(matching) != 1:
                raise ValueError('Accepted ROOT does not identify one exact successful execution attempt')
            accepted = matching[0]
            if accepted.get('log_sha256') and sha256(ROOT/accepted['log']) != accepted['log_sha256']:
                raise ValueError('Accepted fit log changed')
            passed,_=validate(result,ROOT/accepted['log'],accepted['tolerance'])
            if not passed:raise ValueError('Completed limit no longer passes numerical validation')
            return record
        if record.get('pid'):
            try:os.kill(record['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('A previous Combine subprocess is still alive')
        workspace=work/'workspace.root'
        if not record.get('workspace_valid'):
            if workspace.exists():
                raise ValueError('Interrupted workspace output retained; explicit recovery required')
            command=['text2workspace.py',str(card),'-m','120','-o',str(workspace),'--verbose','0']
            with (work/'workspace.log').open('x') as stream:
                proc=subprocess.Popen(command,cwd=card.parent,env=env,stdin=subprocess.DEVNULL,
                                      stdout=stream,stderr=subprocess.STDOUT)
                record.update(phase='workspace',pid=proc.pid,command=command)
                write_json(state_path,record)
                rc=proc.wait()
            if rc or not workspace.is_file() or workspace.stat().st_size<1000:
                record.update(status='failed',error='workspace_conversion_failed',exit_code=rc,finished=time.time(),pid=None)
                write_json(state_path,record);return record
            import uproot
            with uproot.open(workspace) as root:
                if 'w' not in root:raise ValueError('Missing RooWorkspace w')
            record.update(workspace_valid=True,workspace_sha256=sha256(workspace),pid=None)
            write_json(state_path,record)
        elif input_check is not None:
            if not workspace.is_file():
                raise ValueError('Previously verified workspace missing')
        elif sha256(workspace)!=record['workspace_sha256']:
            raise ValueError('Workspace changed')
        completed_attempts={a['name'] for a in record['attempts']}
        configurations = [(label, tolerance, scale, 0) for label, tolerance, scale in ATTEMPTS]
        if comparison:
            configurations += [('nominal_mcstat_s1', .01, 1., 1), ('nominal_mcstat_s2', .001, 1., 2)]
        for label,tolerance,scale,strategy in configurations:
            if label in completed_attempts:continue
            here=work/label
            if here.exists():
                raise ValueError('Interrupted fit attempt retained; explicit recovery required')
            here.mkdir();(here/'tmp').mkdir()
            rmax = float(point.get('rmax_seed',20.))*scale
            if comparison:
                if record['attempts']:
                    from TROTASR.workflows.continue_limit_fits import adaptive_seed
                    try:
                        rmax, seed_evidence = adaptive_seed([(a['log'], (ROOT/a['log']).read_text(errors='replace'))
                                                             for a in record['attempts']])
                    except ValueError as error:
                        # No scan is not a fabricated range measurement. Change only
                        # the finite minimizer strategy at the existing nominal seed.
                        rmax = float(point['rmax_seed'])
                        seed_evidence = dict(method='unchanged_nominal_seed_no_profile', detail=str(error))
                    record['range_diagnostic'] = seed_evidence
                rmax = min(100000., rmax)
            command=fit_command(workspace,point['mass'],rmax,tolerance)
            if comparison:
                command[command.index('--cminDefaultMinimizerStrategy')+1] = str(strategy)
            speed_evidence = None
            if numerical_speed:
                from TROTASR.workflows.limit_recovery import optimize_saved_command
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
                record.update(phase='fit',pid=proc.pid,command=command,current_attempt=label)
                write_json(state_path,record);rc=wait_for_inference(proc,log)
            result=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            passed,evidence=validate(result,log,tolerance) if rc==0 else (False,dict(exit_code=rc))
            check_card(point, digest)
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


def eos_path(path):
    """The existing eossubmit path rule, including resolved symlink targets."""
    path = Path(path)
    for candidate in (path, path.resolve()):
        if (not candidate.is_absolute() or any(x in ('tmp', 'afs', '..') for x in candidate.parts)
                or not str(candidate).startswith(('/eos/user/', '/eos/home-t/taiwoo/'))):
            raise ValueError('CERN submission requires an EOS path: ' + str(path))
    if re.search(r'[\s\"\x27;$`\\]', str(path)):
        raise ValueError('Unsupported characters in EOS submission path')
    return path


def verify_comparison_grid(source):
    from TROTASR.workflows.build_nominal_grid import (
        COMPARISON_VARIANTS, candidate_bin_count, test_comparison_bin_count, TEST_COMPARISON_IDENTITIES,
        CUTBASED_SCHEMA,CUTBASED_CAVEAT,MET_CATEGORY_SCHEMA,MET_CATEGORY_CAVEAT,met_category_caveat)
    variant = source.get('variant')
    if (source.get('scope') != 'sr_only_nominal_mcstat_comparison'
            or source.get('status') != 'cards_ready' or variant not in COMPARISON_VARIANTS
            or source.get('sr_data_blinded') is not True or source.get('auto_mc_stats') != [10,1,1]):
        raise ValueError('Require the explicit completed nominal SR comparison')
    modes = {'baseline':{'highdm':118,'lowdm':30}, 'low':{'lowdm':30}, 'high':{'highdm':118}, 'medium':{'mediumdm':20},
             'combined':{'highdm':118,'lowdm':30,'mediumdm':20}}[variant]
    candidate = source.get('medium_candidate')
    test = source.get('medium_test_comparison')
    if candidate is not None and test is not None:
        raise ValueError('Validation ranking and frozen test provenance cannot be mixed')
    if candidate is not None:
        if variant not in ('medium','combined'):
            raise ValueError('A validation candidate cannot replace the fixed high/low inputs')
        modes['mediumdm'] = candidate_bin_count(candidate)
        if candidate.get('schema')==CUTBASED_SCHEMA:
            if (source.get('evaluation_partition')!='all_mc' or source.get('neural_network_used') is not False
                    or source.get('independent_validation') is not False
                    or source.get('optimization_population')!=candidate['populations']
                    or source.get('historical_partitions_used')!=candidate['historical_partitions_used']
                    or source.get('statistical_caveat')!=CUTBASED_CAVEAT):
                raise ValueError('Full-MC cut-based statistical provenance changed')
        if candidate.get('schema')==MET_CATEGORY_SCHEMA:
            if (source.get('evaluation_partition')!='all_mc' or source.get('neural_network_used') is not False
                    or source.get('independent_validation') is not False
                    or source.get('parent_population')!=candidate['parent_populations']
                    or source.get('selection')!=candidate['selection']
                    or source.get('full_tr_retained') is not candidate['full_tr_retained']
                    or source.get('full_tr_accounted') is not True
                    or source.get('historical_partitions_used')!=candidate['historical_partitions_used']
                    or source.get('statistical_caveat')!=met_category_caveat(candidate)):
                raise ValueError('MET-category selection and parent-population provenance changed')
        if variant == 'combined':
            if not candidate.get('fixed_baseline_files') or any(
                    source.get('files',{}).get(name) != digest
                    for name,digest in candidate['fixed_baseline_files'].items()):
                raise ValueError('Candidate combination changed fixed high+low templates')
    if test is not None:
        if variant not in ('medium','combined'):
            raise ValueError('A frozen test cannot replace fixed high/low inputs')
        modes['mediumdm']=test_comparison_bin_count(test)
        if (source.get('evaluation_partition')!='test' or source.get('reselection_allowed') is not False
                or source.get('rebinning_allowed') is not False or source.get('test_support_is_diagnostic_only') is not True):
            raise ValueError('Frozen test evaluation policy changed')
        provenance=source.get('medium_test_provenance',{})
        if (set(provenance)!={'2024','2025'} or any(
                not re.fullmatch(r'[0-9a-f]{64}',provenance[y].get(k,''))
                for y in provenance for k in TEST_COMPARISON_IDENTITIES)):
            raise ValueError('Frozen test event/physical-weight identities are required')
        if variant=='combined' and (not test.get('fixed_baseline_files') or any(
                source.get('files',{}).get(name)!=digest for name,digest in test['fixed_baseline_files'].items())):
            raise ValueError('Test combination changed fixed high+low templates')
    expected = {f'SR_{mode}_c1_{year}':count for mode,count in modes.items() for year in ('2024','2025')}
    if ({c:r['bins'] for c,r in source['channels'].items()} != expected
            or source['bins'] != sum(expected.values())):
        raise ValueError('SR comparison channel/bin inventory changed')
    points = source['points']; counts = {m:sum(p['model']==m for p in points) for m in ('T2tt','T2bW','T2tb')}
    if (counts != {'T2tt':225,'T2bW':365,'T2tb':365} or counts != source['expected_points_by_model']
            or len({(p['model'],p['mass']) for p in points}) != 955
            or any(type(p.get('fit_ready')) is not bool for p in points)):
        raise ValueError('The unchanged 955-point comparison grid is required')


def eos_submit_text(output, proxy, runtime_archive, bundle, wrapper, keys):
    """Reuse the existing limits_eossubmit.sub layout with canonical workers.

    Persistent files stay on EOS. HTCondor transfers the frozen input bundle
    into its own worker scratch, and transfers the result archive back to EOS.
    No worker opens a hep2 path or an old physics implementation.
    """
    output, proxy, runtime_archive, bundle, wrapper = map(eos_path,
        (output, proxy, runtime_archive, bundle, wrapper))
    if not keys or any(not re.fullmatch(r'T2(?:tt|bW|tb)_mStop\d+_mLSP\d+', k) for k in keys):
        raise ValueError('Empty or invalid Condor point list')
    return f'''universe = vanilla
executable = {wrapper}
initialdir = {output}/condor
arguments = $(point)
getenv = False
output = {output}/condor_logs/$(point).out
error = {output}/condor_logs/$(point).err
log = {output}/condor_logs/cluster_$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {runtime_archive}, {bundle}
transfer_output_files = result.tar
transfer_output_remaps = "result.tar = {output}/incoming/$(point).tar"
use_x509userproxy = true
x509userproxy = {proxy}
request_cpus = 1
request_memory = 6000MB
request_disk = 12000MB
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "TROTASR_nominal_{output.parent.name}"
+AllHadCampaign = "topology_transition_nominal"
+AllHadEOSReserveBytes = 268435456
queue point from (
{chr(10).join(keys)}
)
'''


def eos_wrapper(manifest, output, archive_name, archive_digest, bundle_digest):
    """Existing Combine runtime wrapper; only the invocation is canonical."""
    if archive_name != 'combine_cmssw_14_1_0_pre4.tgz' or any(
            not re.fullmatch('[0-9a-f]{64}', h) for h in (archive_digest, bundle_digest)):
        raise ValueError('Invalid frozen Combine runtime/bundle')
    mr, out = str(internal_path(manifest).relative_to(ROOT)), str(internal_path(output).relative_to(ROOT))
    return '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail',
        ': "${_CONDOR_SCRATCH_DIR:?Condor scratch directory is required}"',
        'SCRATCH_BASE="$(readlink -f "$_CONDOR_SCRATCH_DIR")"',
        'case "$SCRATCH_BASE" in /tmp|/tmp/*|/afs|/afs/*|/home|/home/*|/data|/data/*) exit 64 ;; esac',
        'cd "$SCRATCH_BASE"', 'POINT=$1',
        'export TMPDIR="$SCRATCH_BASE/temporary" XDG_CACHE_HOME="$SCRATCH_BASE/cache"',
        'mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"',
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ROOTENV_NO_HOME=1',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'unset PYTHONPATH PYTHONHOME',
        f'echo "{archive_digest}  {archive_name}" | sha256sum -c -',
        f'echo "{bundle_digest}  inputs.tar.gz" | sha256sum -c -',
        f'tar -xzf {archive_name}', 'tar -xzf inputs.tar.gz',
        'CMSSW="$SCRATCH_BASE/CMSSW_14_1_0_pre4"',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh', 'cd "$CMSSW/src"',
        'scramv1 b ProjectRename >/dev/null', 'eval "$(scramv1 runtime -sh)"',
        'case "$(command -v combine)" in "$CMSSW"/*) ;; *) exit 70 ;; esac',
        'case "$(command -v text2workspace.py)" in "$CMSSW"/*) ;; *) exit 70 ;; esac',
        'export PYTHONPATH="$SCRATCH_BASE:$ROOTSYS/lib:/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/lib/python3.9/site-packages"',
        'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cvmfs/sft.cern.ch/lcg/releases/blas/0.3.20.openblas-c07f1/x86_64-el9-gcc13-opt/lib:/cvmfs/sft.cern.ch/lcg/releases/hdf5/1.12.2-4c7fc/x86_64-el9-gcc13-opt/lib"',
        'cd "$SCRATCH_BASE"',
        f'python3 -u TROTASR/workflows/combine_limits.py --manifest TROTASR/{shlex.quote(mr)} --output TROTASR/{shlex.quote(out)} --condor-point "$POINT"',
        ''])


def prepare_eos(manifest, output, runtime_manifest, proxy):
    """Prepare, never submit. Run after immutable cards are staged on EOS."""
    if sys.platform != 'linux': raise RuntimeError('Do not copy ROOT products to the laptop')
    manifest, output = internal_path(manifest), internal_path(output)
    for path in (ROOT, manifest, output, runtime_manifest, proxy): eos_path(path)
    source = read_json(manifest)
    verify_comparison_grid(source)
    integrity = IntegrityCache()
    for point in source['points']: verify_card(point, integrity)
    # This is the numerical-software manifest, not an external analysis input.
    software = json.loads(Path(runtime_manifest).read_text())
    archive = eos_path(software['combine']['path']); checksum = software['combine']['sha256']
    if software.get('status') != 'ready' or archive.name != 'combine_cmssw_14_1_0_pre4.tgz':
        raise ValueError('Existing Combine runtime is not ready')
    import hashlib
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): digest.update(block)
    if digest.hexdigest() != checksum: raise ValueError('Numerical runtime archive changed')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = output/'contract.json'
        if frozen.exists():
            old = read_json(frozen)
            if 'submission_files' not in old:
                raise ValueError('Interrupted preparation preserved; reconcile before preparing again')
            if (old.get('backend') != 'eosschedd' or old['manifest_sha256'] != sha256(manifest)
                    or old['runtime_sha256'] != checksum):
                raise ValueError('Preserve divergent frozen Condor contract')
            for name, expected in old['code'].items():
                if sha256(ROOT/name) != expected: raise ValueError('Frozen Condor source changed')
            for name, expected in old['submission_files'].items():
                if sha256(output/name) != expected: raise ValueError('Frozen submission changed')
            return 0
        for folder in ('condor', 'condor_logs', 'incoming'): (output/folder).mkdir(exist_ok=True)
        sources = set(p for folder in ('utils', 'workflows') for p in (ROOT/folder).rglob('*.py'))
        sources.update((ROOT/'__init__.py', ROOT/'stats/runtime/sitecustomize.py'))
        sources = {p for p in sources if p.is_file()}
        inputs = {manifest} | {ROOT/p for p in source['files']}
        for point in source['points']:
            card = ROOT/point['card']; inputs.update((card, card.with_suffix('.manifest.json')))
            inputs.add(ROOT/read_json(card.with_suffix('.manifest.json'))['comparison_manifest'])
        code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)}
        receipt = dict(backend='eosschedd', manifest_sha256=sha256(manifest), code=code,
            runtime_sha256=checksum, runtime_archive=str(archive), proxy=str(eos_path(proxy)),
            prepared=time.time(), observed_sr_used=False)
        # Include the contract in the worker bundle, before adding transport hashes.
        write_json(frozen, receipt)
        worker_contract_hash = sha256(frozen)
        bundle = output/'condor/inputs.tar.gz'
        with tarfile.open(bundle, 'x:gz') as packed:
            for path in sorted(sources | inputs | {frozen}):
                packed.add(path, arcname='TROTASR/'+str(path.relative_to(ROOT)), recursive=False)
        wrapper = output/'condor/run_condor_limit_point.sh'
        wrapper.write_text(eos_wrapper(manifest, output, archive.name, checksum, sha256(bundle)))
        wrapper.chmod(0o755)
        keys = [p['model']+'_'+p['mass'] for p in source['points'] if p.get('fit_ready')]
        submit = output/'condor/limits_eossubmit.sub'
        submit.write_text(eos_submit_text(output, proxy, archive, bundle, wrapper, keys))
        receipt.update(points=keys, worker_contract_sha256=worker_contract_hash,
                       submission_files={str(p.relative_to(output)):sha256(p)
                       for p in (bundle, wrapper, submit)},
                       scheduler_command='module load lxbatch/eossubmit; canonical --submit-eos')
        write_json(frozen, receipt)
        write_json(output/'state.json', dict(status='prepared_not_submitted', total=len(source['points']),
            requested=len(keys), blocked=[p['model']+'/'+p['mass'] for p in source['points'] if not p.get('fit_ready')],
            observed_sr_used=False, full_workflow_complete=False))
    return 0


def submit_eos(output):
    """Existing eossubmit dry-run/receipt logic, integrated into this entrypoint.

    Reuses prepare_2024_fullselection_campaign.py's native scheduler environment,
    EOS proxy/lifetime rules, exclusive submission receipt and ambiguous-submit
    handling; no old analysis code or physical inputs are imported.
    """
    output = internal_path(output); eos_path(output); eos_path(Path.cwd())
    if os.environ.get('_myschedd_POOL') != 'eossubmit' or not os.environ.get('_condor_CONDOR_HOST'):
        raise RuntimeError('Load the official lxbatch/eossubmit module in an EOS directory')
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH')}
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = output/'submission.json'
        if receipt.exists():
            old = read_json(receipt)
            if old['status'] != 'submitted':
                raise RuntimeError('Reconcile the uncertain submission with eosschedd before retrying')
            print(json.dumps(old)); return 0
        contract = read_json(output/'contract.json')
        if contract['backend'] != 'eosschedd': raise ValueError('Not an EOS submission')
        for name, expected in contract['code'].items():
            if sha256(ROOT/name) != expected: raise ValueError('Frozen Condor source changed')
        for name, expected in contract['submission_files'].items():
            if sha256(output/name) != expected: raise ValueError('Frozen submission changed')
        proxy = eos_path(contract['proxy'])
        if (proxy.is_symlink() or not proxy.is_file() or proxy.stat().st_uid != os.getuid()
                or proxy.stat().st_mode & 0o077): raise ValueError('Proxy must be owner-only')
        # Inspect EOS ACL metadata too, not only the FUSE permission bits.
        acl = subprocess.check_output(['eos','root://eosuser.cern.ch','ls','-ld',str(proxy)], text=True, env=env)
        if not acl.strip(): raise ValueError('EOS proxy permissions could not be inspected')
        for path in (proxy.parent, proxy):
            attrs = subprocess.check_output(['eos','root://eosuser.cern.ch','attr','ls',str(path)], text=True, env=env)
            for line in attrs.splitlines():
                if re.search(r'(?:sys|user)\.acl=', line) and line.split('=',1)[1].strip(' "'):
                    raise ValueError('Nonempty proxy EOS ACL requires explicit access inspection: '+str(path))
        lifetimes = {name:int(subprocess.check_output(['voms-proxy-info','--file',str(proxy),'--'+name],
                      text=True, env=env).strip()) for name in ('timeleft','actimeleft')}
        if min(lifetimes.values()) <= 0: raise ValueError('Proxy or VOMS lifetime expired')
        queue = subprocess.check_output(['condor_q','-json','-attributes',
            'JobStatus,AllHadEOSReserveBytes,RequestDisk'], text=True, env=env)
        jobs = json.loads(queue if queue.strip() else '[]')
        reserved = sum(int(j.get('AllHadEOSReserveBytes', int(j.get('RequestDisk',0))*1024)) for j in jobs)
        raw = subprocess.check_output(['eos','root://eosuser.cern.ch','quota','-m'], text=True, env=env)
        rows = [dict(f.split('=',1) for f in shlex.split(line) if '=' in f)
                for line in raw.splitlines() if line.startswith('quota=')]
        rows = [r for r in rows if r.get('space') == '/eos/user/t/taiwoo/' and 'maxlogicalbytes' in r]
        if len(rows) != 1: raise ValueError('Account-wide EOS quota unavailable')
        quota = rows[0]; required = len(contract['points'])*268435456
        if int(quota['maxlogicalbytes'])-int(quota['usedlogicalbytes'])-reserved-32*1024**3 < required:
            raise RuntimeError('Insufficient EOS space for pending outputs; no deletion or job cancellation performed')
        submit = output/'condor/limits_eossubmit.sub'; dry = output/'condor/dryrun.ads'
        subprocess.run(['condor_submit','-dry-run',str(dry),str(submit)], check=True, env=env, cwd=output)
        if re.search(r'/(?:tmp|afs)(?:/|["\s])', dry.read_text()):
            raise ValueError('Expanded submit references forbidden AFS or /tmp')
        state = dict(status='submitting', started=time.time(), contract_sha256=sha256(output/'contract.json'),
                     quota=quota, reserved_existing_bytes=reserved, proxy_lifetimes_seconds=lifetimes)
        with receipt.open('x') as stream:
            json.dump(state,stream); stream.flush(); os.fsync(stream.fileno())
        try:
            result = subprocess.run(['condor_submit','-terse',str(submit)], check=True, capture_output=True,
                                    text=True, env=env, cwd=output)
            match = re.fullmatch(r'\s*(\d+)\.(\d+)\s*(?:-\s*\1\.(\d+))?\s*', result.stdout)
            if not match: raise ValueError('Unknown submit receipt; inspect schedd before retrying')
            state.update(status='submitted', cluster_id=int(match[1]), first_proc=int(match[2]),
                         last_proc=int(match[3] or match[2]), receipt=result.stdout.strip(), submitted=time.time())
        except BaseException as exc:
            state.update(status='submission_requires_reconciliation', error=str(exc)); write_json(receipt,state); raise
        write_json(receipt,state); print(json.dumps(state))
    return 0


def condor_point(manifest, output, key):
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR','')).resolve()
    if (not os.environ.get('_CONDOR_JOB_AD') or scratch not in ROOT.parents
            or str(scratch).startswith(('/tmp','/afs','/home','/data'))):
        raise RuntimeError('Canonical EOS worker may run only inside Condor scratch')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError('Condor fit must be single threaded')
    def storage_guard(event, args):
        if event in ('open','os.chdir','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
            value = os.path.abspath(os.fsdecode(args[0]))
            if any(value == p or value.startswith(p+'/') for p in ('/home','/data','/afs','/tmp')):
                raise PermissionError('Forbidden worker storage: '+value)
    sys.addaudithook(storage_guard)
    fit_stack_limit()
    output, manifest = internal_path(output), internal_path(manifest)
    contract = read_json(output/'contract.json'); source = read_json(manifest)
    verify_comparison_grid(source)
    if contract['backend'] != 'eosschedd' or contract['manifest_sha256'] != sha256(manifest):
        raise ValueError('Worker input contract changed')
    for name, expected in contract['code'].items():
        if sha256(ROOT/name) != expected: raise ValueError('Worker source changed')
    points = [p for p in source['points'] if p['model']+'_'+p['mass'] == key and p.get('fit_ready')]
    if len(points) != 1: raise ValueError('Unknown or unsupported mass point')
    point = points[0]; work = output/point['model']/point['mass']
    try:
        result = run_point(point, output, dict(os.environ), IntegrityCache())
    except Exception as exc:
        result = dict(point, status='failed', error=repr(exc), finished=time.time())
        work.mkdir(parents=True,exist_ok=True); write_json(work/'worker_error.json',result)
    write_json(work/'execution_receipt.json',dict(status=result['status'], host=os.uname().nodename,
        condor_job_ad=Path(os.environ['_CONDOR_JOB_AD']).name, manifest_sha256=sha256(manifest),
        worker_contract_sha256=sha256(output/'contract.json'), finished=time.time()))
    with tarfile.open(scratch/'result.tar','x') as packed:
        packed.add(work, arcname=str(work.relative_to(ROOT)))
    # Transport must retain numerical failures as well as successes.
    return 0


def collect_eos(manifest, output):
    """Import scheduler-returned point files, then apply the original validator."""
    manifest, output = internal_path(manifest), internal_path(output)
    source = read_json(manifest); contract = read_json(output/'contract.json')
    verify_comparison_grid(source)
    if contract['backend'] != 'eosschedd' or contract['manifest_sha256'] != sha256(manifest):
        raise ValueError('Collection contract changed')
    results = {}; integrity = IntegrityCache()
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for point in source['points']:
            key = point['model']+'/'+point['mass']; work = output/key
            if not point.get('fit_ready'):
                results[key] = dict(status='blocked',error='no_supported_signal_template'); continue
            archive = output/'incoming'/(key.replace('/','_')+'.tar')
            if not archive.exists(): continue
            prefix = str(work.relative_to(ROOT))
            with tarfile.open(archive,'r:') as packed:
                members = packed.getmembers()
                for member in members:
                    if ((member.name != prefix and not member.name.startswith(prefix+'/'))
                            or '..' in Path(member.name).parts or not (member.isfile() or member.isdir())):
                        raise ValueError('Unexpected returned archive member')
                    target = internal_path(ROOT/member.name)
                    if member.isfile():
                        raw = packed.extractfile(member).read()
                        if target.exists():
                            if target.read_bytes() != raw: raise ValueError('Preserve divergent returned point output')
                        else:
                            target.parent.mkdir(parents=True,exist_ok=True)
                            with target.open('xb') as stream: stream.write(raw)
            execution = read_json(work/'execution_receipt.json')
            if (execution['manifest_sha256'] != sha256(manifest)
                    or execution['worker_contract_sha256'] != contract['worker_contract_sha256']):
                raise ValueError('Wrong execution receipt')
            state = read_json(work/'worker_error.json') if (work/'worker_error.json').exists() else read_json(work/'state.json')
            if state['status'] == 'complete':
                # The complete branch only reads/hash-checks/validates; it cannot refit.
                state = run_point(point, output, {}, integrity)
            results[key] = state
        for model in ('T2tt','T2bW','T2tb'):
            write_json(output/model/'expected_limits.json',collect(source['points'],results,model,source['scope']))
        failed = [k for k,r in results.items() if r['status'] != 'complete']
        write_json(output/'state.json',dict(status='awaiting_condor_results' if len(results)<len(source['points'])
            else 'needs_attention' if failed else 'fits_complete_pending_plots',total=len(source['points']),
            completed=sum(r['status']=='complete' for r in results.values()),failed=failed,results=results,
            observed_sr_used=False,full_workflow_complete=False))
    return 0


def admission_order(source):
    """Prioritize the adopted display window without changing the frozen grid."""
    points = source['points']
    if source.get('run3_systematics'):
        from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
        from TROTASR.workflows.reconcile_limit_fits import plot_range_first
        require_run3_systematics_policy(source)
        return plot_range_first(points)
    return points


def selected_fit_points(source, plot_range_only=False):
    """Keep the full card grid immutable; select only the explicitly approved fits."""
    if not plot_range_only:
        return source['points'], {}
    from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
    from TROTASR.workflows.reconcile_limit_fits import in_plot_range
    if (not require_run3_systematics_policy(source) or source.get('status') != 'cards_ready'
            or source.get('scope') != 'full_combined_systematic_production'):
        raise ValueError('Visible-only fits require the new six-era systematic model')
    verify_combined_grid(source)
    points = [p for p in source['points'] if in_plot_range(p)]
    counts = {m:sum(p['model'] == m for p in points) for m in ('T2tt','T2bW','T2tb')}
    if len(source['points']) != 955 or counts != {'T2tt':205,'T2bW':345,'T2tb':345}:
        raise ValueError('Displayed fit selection must contain exactly 895 of 955 points')
    keys = sorted(p['model']+'/'+p['mass'] for p in points)
    selection = dict(plot_range_only=True, source_total=955, total=895,
        excluded_out_of_plot_range=60, expected_points_by_model=counts,
        mStop=[600,1800], mLSP=[1,1800], outside='not_requested',
        selected_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest())
    return points, selection


def selection_metadata(selection):
    return {name:selection[name] for name in ('plot_range_only','source_total','total',
        'excluded_out_of_plot_range','expected_points_by_model')} if selection else {}


def run_primary_grid(manifest,output,workers,plot_range_only=False):
    if not 1<=workers<=99:raise ValueError('Reserve one CPU slot for controller')
    output=internal_path(output);manifest=internal_path(manifest)
    source=read_json(manifest)
    study = source.get('scope') == 'sr_only_nominal_mcstat_comparison'
    if source['status']!='cards_ready' or source['scope'] not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production','full_run3_hybrid_production','sr_only_nominal_mcstat_comparison'):
        raise ValueError('Production expected contours require complete full-input cards')
    if study:
        verify_comparison_grid(source)
        if source.get('medium_candidate'):
            from TROTASR.workflows.build_nominal_grid import load_medium_candidate
            if load_medium_candidate(source['medium_candidate']['contract_path']) != source['medium_candidate']:
                raise ValueError('Medium candidate contract changed before fitting')
        if source.get('medium_test_comparison'):
            from TROTASR.workflows.build_nominal_grid import load_medium_test_comparison
            test=source['medium_test_comparison']
            expected=ROOT/'stats/topology_transition_nominal/test_comparison'/test['candidate_id']/source['variant']
            if (load_medium_test_comparison(test['contract_path'])!=test
                    or manifest!=expected/'grid/manifest.json' or output!=expected/'limits'):
                raise ValueError('Frozen test contract or fit output directory changed')
    verify_combined_grid(source)
    points=source['points']
    counts={m:sum(p['model']==m for p in points) for m in ('T2tt','T2tb','T2bW')}
    if counts!=source.get('expected_points_by_model') or not all(counts.values()):
        raise ValueError('All three complete signal grids are required')
    keys=[p['model']+'/'+p['mass'] for p in points]
    if len(set(keys))!=len(keys) or not points:raise ValueError('Empty/duplicate grid')
    points, selection = selected_fit_points(source, plot_range_only)
    selected_metadata = selection_metadata(selection)
    integrity = IntegrityCache()
    for p in points:verify_card(p, integrity)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dependencies = ['workflows/combine_limits.py','workflows/limit_validation.py',
                        'stats/runtime/sitecustomize.py','stats/runtime.json']
        if source.get('run3_systematics'):
            dependencies += ['workflows/reconcile_limit_fits.py', 'workflows/build_nominal_grid.py']
        if study:
            dependencies += ['workflows/build_nominal_grid.py','workflows/continue_limit_fits.py',
                             'workflows/nominal_campaign.py']
        contract=dict(manifest_sha256=sha256(manifest),code={n:sha256(ROOT/n) for n in dependencies})
        if selection:contract['fit_selection'] = selection
        contract_path=output/'contract.json'
        if contract_path.exists() and read_json(contract_path)!=contract:raise ValueError('Divergent fit contract')
        write_json(contract_path,contract)
        env=runtime(output)
        results={p['model']+'/'+p['mass']:dict(status='blocked',error='no_supported_signal_template',
                 issues=p.get('signal_issues',[])) for p in points if study and not p.get('fit_ready')}
        pending=iter(p for p in admission_order(dict(source,points=points))
                     if not study or p.get('fit_ready'));exhausted=False
        budget=dict(max_workers=workers,aggregate_slots=100,minimum_available_memory_bytes=32*1024**3,
                    admission_bytes_per_worker=4*1024**3)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active={}
            while active or not exhausted:
                external,memory=resources()
                limit=capacity(budget,external,memory,len(active))
                while len(active)<limit and not exhausted:
                    p=next(pending,None)
                    if p is None:exhausted=True
                    else:active[pool.submit(run_point,p,output,env,integrity)]=p
                done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED) if active else (set(),set())
                for f in done:
                    p=active.pop(f);key=p['model']+'/'+p['mass']
                    try:results[key]=f.result()
                    except Exception as e:results[key]=dict(status='failed',error=repr(e))
                write_json(output/'state.json',dict(status='running',pid=os.getpid(),updated=time.time(),
                    **dict(selected_metadata,total=len(points)),active=len(active),
                    remaining=len(points)-sum(r['status']=='complete' for r in results.values()),
                    completed=sum(r['status']=='complete' for r in results.values()),
                    failed=[k for k,r in results.items() if r['status']!='complete'],external_slots=external,
                    observed_sr_used=False,full_workflow_complete=False))
                if not active and not exhausted:time.sleep(5)
        for model in ('T2tt','T2tb','T2bW'):
            summary = collect(points,results,model,source['scope'])
            if selection:summary.update(fit_selection=selection,**selected_metadata)
            write_json(output/model/'expected_limits.json',summary)
        failed=[k for k,r in results.items() if r['status']!='complete']
        write_json(output/'state.json',dict(status='fits_complete_pending_plots' if not failed else 'needs_attention',
            **dict(selected_metadata,total=len(points)),completed=len(points)-len(failed),
            remaining=len(failed),failed=failed,results=results,
            full_workflow_complete=False,observed_sr_used=False))
        return 1 if failed else 0


EXPECTED_DISPLAY_COUNTS = {'T2tt': 205, 'T2bW': 345, 'T2tb': 345}
# Read compatibility only: new execution never launches these old controllers.
LEGACY_RECOVERY_STAGES = (
    ('numerical_all_unfinished_20260930', 'base'),
    ('numerical_remaining_20261009', 'precision'),
    ('numerical_targeted_20261009', 'measured'),
    ('numerical_minima_20261009', 'minima'),
)
INPUT_CHECK_SCOPE = 'accepted_cards_templates_or_workspace; no_upstream_revalidation'


def ready_fit_points(source):
    """Consume the completed card-stage receipt, not its production ancestry."""
    if source.get('status') != 'cards_ready' or source.get('scope') != 'full_combined_systematic_production':
        raise ValueError('The integrated limit stage requires the accepted combined card grid')
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


def ready_card(point, digest=None):
    card = internal_path(ROOT/point['card'])
    if not card.is_file():
        raise FileNotFoundError('Delivered datacard is missing: '+str(card))
    return card


def accepted_fit_record(point, record, origin):
    """Reuse a completed receipt. Only NEW fits are opened/numerically checked."""
    if record.get('status') not in ('complete', 'verified_not_promoted', 'existing_verified'):
        return None
    identity = record.get('point', record)
    for name in ('model', 'mass', 'card_sha256'):
        if identity.get(name) != point[name]:
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
    paths += [(stage, output/name/key/'state.json') for name, stage in LEGACY_RECOVERY_STAGES]
    for stage, path in paths:
        if path.is_file():
            record = read_json(path)
            identity = record.get('point', record)
            for name in ('model', 'mass', 'card_sha256'):
                if identity.get(name) != point[name]:
                    raise ValueError('Different point/card receipt retained: '+str(path))
            records.append((stage, path, record))
    return records


def saved_point_result(point, output):
    for stage, path, record in point_receipts(point, output):
        accepted = accepted_fit_record(point, record, str(path.relative_to(ROOT)))
        if accepted is not None:
            return accepted
    return None


def recovery_stage_for_attempt(stage, attempt):
    """The old minima directory contained three later numerical stages."""
    if stage != 'minima':
        return stage
    name = attempt.get('name', '')
    command = attempt.get('command', [])
    if attempt.get('run3_numerical_adapter') or any(word in name for word in ('range', 'native', 'stepping')):
        return 'branches'
    if any(word in name for word in ('crossing', 'refine')):
        return 'crossings'
    return 'minima'


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
        state.update(pid=process.pid, phase=spec['stage'], command=spec['command'])
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
        selected.append(dict(spec, stage='branches', recipe_source=str(source),
            historical_recipe_only=True, historical_seed_not_new_model_bracket=True))
    return selected, dict(status='applicable' if selected else 'already_consumed',
        source=str(source), historical_rmax=rmax, historical_quantiles_reused=False,
        historical_source_imported=False, acceptance_unchanged=True,
        basis='user_requested_same_point_successful_numerical_recipe')


def run_integrated_point(point, output, env, progress=lambda phase, pid: None):
    """One point lock and one slot for primary -> all finite recovery -> receipt."""
    from TROTASR.workflows.limit_recovery import (
        recovery_commands, recovery_stage_order, validate_attempt, profile_recovery_command, STAGE_LIMITS)
    work = output/point['model']/point['mass']
    work.mkdir(parents=True, exist_ok=True)
    with (work/'point.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = saved_point_result(point, output)
        if saved is not None:
            return saved
        records = point_receipts(point, output)
        for stage, path, record in records:
            if record.get('pid'):
                try:
                    os.kill(int(record['pid']), 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous point process is still alive; preserve it')
            if any(a.get('exit_code') is None for a in record.get('attempts', [])):
                raise RuntimeError('Interrupted command receipt retained; diagnose before new admission')
        progress('primary', None)
        primary = next((r for stage, _, r in records if stage == 'primary'), None)
        if primary is None or primary.get('error') != 'finite_attempts_exhausted':
            primary = run_point(point, output, env, input_check=ready_card, lock_owned=True,
                                numerical_speed=True)
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
        inherited = []
        for stage, path, record in records:
            if stage in ('workflow', 'primary'):
                continue
            inherited.extend(dict(a, stage=recovery_stage_for_attempt(stage, a),
                                  inherited_receipt=str(path.relative_to(ROOT)))
                             for a in record.get('attempts', []))
            # An interrupted old stage can already have a fixed finite pair.
            # Consume that small plan receipt instead of reseeding it from logs.
            contract_path = path.with_name('contract.json')
            if stage not in state['stages'] and contract_path.is_file():
                planned = read_json(contract_path).get('attempts', [])
                if planned:
                    specs = [dict(spec, command=spec.get('command') or
                                  profile_recovery_command(workspace, point['mass'], spec))
                             for spec in planned]
                    state['stages'][stage] = dict(status='pending', applicability=[], plans=[specs],
                        inherited_plan=str(contract_path.relative_to(ROOT)))
        history = list(primary.get('attempts', [])) + inherited + state['attempts']
        # The old finite recovery first reconciled the actual saved, failed
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
        # The user requested successful same-point prescriptions first. Reuse
        # settings only, on this model's workspace, within the existing finite
        # branch allowance. Already accepted points returned before this lookup.
        prescription = state.get('historical_success_prescription')
        if prescription is None:
            specs, diagnosis = historical_success_commands(point, workspace, output,
                [a['command'] for a in history if a.get('command')])
            prescription = dict(diagnosis, plans=specs)
            state['historical_success_prescription'] = prescription
            write_json(state_path, state)
        branch_attempts = [a for a in inherited+state['attempts'] if a.get('stage') == 'branches']
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
        # Retain existing frozen plans. Only a new ledger gets the already
        # approved conditional-status-based finite scheduling policy.
        if 'recovery_stage_order' not in state:
            fixed = bool(inherited or state['attempts'] or any(
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
            stage_attempts = [a for a in inherited + state['attempts'] if a.get('stage') == stage]
            while len(stage_attempts) < maximum:
                commands = [a['command'] for a in history if a.get('command')]
                logs = [(a['log'], failed_log_text(ROOT/a['log']))
                        for a in history if a.get('log') and internal_path(ROOT/a['log']).is_file()]
                previous_attempts = [a for a in history if a.get('command')]
                if stage in ('crossings', 'branches'):
                    previous_attempts = [a for a in inherited + state['attempts']
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


def fit_pool_resources():
    """Account occupancy without importing any upstream analysis configuration."""
    raw = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,ppid=,comm='], text=True)
    rows = [(int(p), int(pp), Path(c).name)
            for p, pp, c in (s.split(None, 2) for s in raw.splitlines())]
    own = {os.getpid()}
    while True:
        expanded = own | {pid for pid, parent, _ in rows if parent in own}
        if expanded == own:
            break
        own = expanded
    external = sum(pid not in own and (comm.startswith(('python', 'root', 'combine', 'xrdcp', 'rsync'))
                   or comm in ('hadd', 'cp')) for pid, _, comm in rows)
    memory = next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()
                  if row.startswith('MemAvailable:'))
    return external, memory


def fit_pool_resident_memory(output):
    """Credit only materialized native RSS already deducted from MemAvailable."""
    resident = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            name = (proc/'comm').read_text().strip()
            if str(output) not in command or not (name == 'combine' or 'text2workspace.py' in command):
                continue
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            resident += min(int(fields[21])*os.sysconf('SC_PAGE_SIZE'), 3*1024**3)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return resident


def fit_pool_capacity(workers, external, memory, active, disk_free,
                      resident_memory=0, reserved=0):
    gib = 1024**3
    if memory < 24*gib or disk_free < 80*gib:
        return 0
    # Reserve 3 GiB for EVERY future, including those without subprocesses.
    # Only measured native RSS is credited back, never active*reservation.
    total_memory_slots = max(0, (memory+resident_memory-32*gib)//(3*gib))
    return max(0, min(workers-reserved, 116-1-external,
                      total_memory_slots-reserved))


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


def _handoff_process(pid):
    try:
        proc = Path('/proc')/str(pid)
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (proc/'cmdline').read_bytes()
        return dict(pid=pid, startticks=int(fields[19]),
                    cmdline_sha256=hashlib.sha256(command).hexdigest(),
                    argv=command.decode().split('\0'), threads=len(list((proc/'task').iterdir())))
    except FileNotFoundError:
        return None


def _handoff_lock_owners(path):
    stat = path.stat()
    target = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    owners = set()
    for row in Path('/proc/locks').read_text().splitlines():
        fields = row.split()
        if '->' in fields or 'FLOCK' not in fields:
            continue
        index = fields.index('FLOCK')
        device = fields[index+4].split(':')
        if (int(device[0], 16), int(device[1], 16), int(device[2])) == target:
            owners.add(int(fields[index+3]))
    return owners


def _refresh_draining_handoff(token):
    """Release a reservation only after its controller exits AND its locks free."""
    if token is None or token['exclusive']:
        return set()
    if sha256(token['path']) != token['sha256']:
        raise ValueError('Explicit draining handoff changed')
    if token.get('kind') == 'pool_resize':
        return _refresh_pool_resize(token)
    released = set()
    for owner in list(token['owners']):
        live = _handoff_process(owner['pid'])
        locks = [token['output']/name for name in owner['locks']]
        if live is not None:
            if any(live[key] != owner[key] for key in ('pid', 'startticks', 'cmdline_sha256')):
                raise ValueError('Draining owner identity changed')
            if live['threads'] > 2:
                raise ValueError('Unaccounted draining worker threads')
            if any(_handoff_lock_owners(path) != {owner['pid']} for path in locks):
                raise ValueError('Draining controller lock ownership changed')
            continue
        for path in locks:
            with path.open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in owner['points']:
            with (token['output']/key/'point.lock').open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        released.update(owner['points'])
        token['owners'].remove(owner)
    token['reserved'].difference_update(released)
    if not token['owners']:
        lock = token['stack'].enter_context((token['output']/'controller.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token['exclusive'] = True
    return released


def _refresh_pool_resize(token):
    """Release individual terminal points from a main-only draining owner."""
    owner = token['owners'][0]
    live = _handoff_process(owner['pid'])
    if live is not None:
        if any(live[key] != owner[key] for key in ('pid', 'startticks', 'cmdline_sha256')):
            raise ValueError('Draining owner identity changed')
        if _handoff_lock_owners(token['output']/'controller.lock') != {owner['pid']}:
            raise ValueError('Draining owner lost its exclusive lock')
        state = read_json(token['output']/'workflow_state.json')
        if state.get('pid') != owner['pid'] or state.get('status') != 'draining':
            raise ValueError('Require the updated main-only draining state')
        active = set(state['active_primary']) | set(state['active_recovery'])
        updated = state['updated']
    else:
        active, updated = set(), None
    released = set()
    for key in token['reserved']-active:
        observations = token['release_observations'].setdefault(key, set())
        observations.add(updated)
        if live is not None and len(observations) < 2:
            continue
        work = token['output']/key
        terminal = False
        for name in ('workflow.json', 'state.json'):
            if (work/name).is_file():
                record = read_json(work/name)
                if record.get('status') in ('complete', 'failed', 'needs_attention'):
                    terminal = not any(a.get('exit_code') is None for a in record.get('attempts', []))
                    if terminal:
                        break
        if not terminal:
            continue
        try:
            with (work/'point.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        released.add(key)
    token['reserved'].difference_update(released)
    if live is None and not token['reserved']:
        lock = token['stack'].enter_context((token['output']/'controller.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token['exclusive'] = True
    return released


@contextmanager
def pool_resize_ownership(manifest, output, path, record):
    owner = record['owner']
    if (record.get('manifest') != str(manifest) or record.get('limits') != str(output)
            or record.get('signal') != 'SIGINT_MAIN_ONCE'
            or record.get('native_children_signaled') is not False
            or record.get('canonical_model') != 'era_merged'
            or not 1 <= record.get('workers', 0) <= 115):
        raise ValueError('Invalid canonical pool-resize handoff')
    live = _handoff_process(owner['pid'])
    if live is not None and any(live[key] != owner[key]
                               for key in ('pid', 'startticks', 'cmdline_sha256')):
        raise ValueError('Pool-resize owner identity mismatch')
    with ExitStack() as stack:
        private_name = record.get('private_lock', 'canonical_handoff.lock')
        if private_name not in ('canonical_handoff.lock','controller_handoff.lock'):
            raise ValueError('Unknown canonical handoff lock')
        private = stack.enter_context((output/private_name).open('a'))
        fcntl.flock(private, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = dict(kind='pool_resize', path=path, sha256=sha256(path), output=output,
                     stack=stack, owners=[owner], reserved=set(record['points']),
                     exclusive=False, release_observations={},
                     state_file='controller_handoff_state.json' if private_name == 'controller_handoff.lock'
                                else 'canonical_handoff_state.json')
        _refresh_draining_handoff(token)
        yield token


@contextmanager
def canonical_controller_ownership(manifest, output, draining_handoff=None):
    if draining_handoff is None:
        with (output/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield None
        return
    path = internal_path(draining_handoff)
    record = read_json(path)
    if record.get('schema') == 'canonical_pool_resize_v1':
        with pool_resize_ownership(manifest, output, path, record) as token:
            yield token
        return
    if (record.get('schema') != 'canonical_draining_handoff_v1'
            or record.get('manifest') != str(manifest) or record.get('limits') != str(output)
            or output != ROOT/'stats/run3_systematics_20261025/limits'):
        raise ValueError('Explicit draining handoff scope mismatch')
    expected = {
        'shared': ('continue_limit_fits.py', ['numerical_all_unfinished_20260930/controller.lock'],
                   ['T2bW/mStop1000_mLSP200']),
        'advanced': ('validate_limit_precision.py', ['controller.lock', 'numerical_extended_shared/controller.lock'],
                     ['T2bW/mStop1650_mLSP1450'])}
    owners = record.get('owners', [])
    if len(owners) != 2 or {o.get('role') for o in owners} != set(expected):
        raise ValueError('Require the two exact draining controllers')
    for owner in owners:
        script, locks, points = expected[owner['role']]
        if owner.get('locks') != locks or owner.get('points') != points:
            raise ValueError('Draining lock or reserved-point scope changed')
        if (not isinstance(owner.get('pid'), int) or not isinstance(owner.get('startticks'), int)
                or not re.fullmatch('[0-9a-f]{64}', owner.get('cmdline_sha256', ''))):
            raise ValueError('Missing exact draining process identity')
        evidence = owner['drain_signal']
        receipt_path = internal_path(ROOT/evidence['path'])
        if sha256(receipt_path) != evidence['sha256']:
            raise ValueError('Draining signal evidence changed')
        receipt = read_json(receipt_path)
        identity = receipt.get('identity', {})
        if (receipt.get('status') != 'delivered' or receipt.get('signal') != 'SIGINT_MAIN_ONCE'
                or receipt.get('native_children_signaled') is not False
                or identity.get('pid') != owner['pid']
                or int(identity.get('startticks', identity.get('start_ticks', -1))) != owner['startticks']):
            raise ValueError('Require delivered main-only drain evidence')
        live = _handoff_process(owner['pid'])
        if live is not None and str(ROOT/'workflows'/script) not in live['argv']:
            raise ValueError('Draining controller launch does not belong to this analysis')
    with ExitStack() as stack:
        own = stack.enter_context((output/'canonical_handoff.lock').open('a'))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = dict(path=path, sha256=sha256(path), output=output, stack=stack,
                     owners=list(owners), reserved={key for owner in owners for key in owner['points']},
                     exclusive=False)
        _refresh_draining_handoff(token)
        yield token


def run_integrated_grid(manifest, output, workers, draining_handoff=None):
    with accepted_analysis_root(manifest, output):
        return _run_integrated_grid(manifest, output, workers, draining_handoff)


def canonical_model_output(source, output):
    """Keep the official path while preserving the superseded model as history."""
    if not source.get('era_merged'):
        return
    parent = output.parent
    with (parent/'model_handoff.lock').open('a') as handoff:
        fcntl.flock(handoff, fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker = output/'model.json'
        if marker.exists():
            if read_json(marker).get('canonical_model') != 'era_merged':
                raise ValueError('Unknown official output model')
            return
        previous = parent/'history/separate_eras/limits'
        has_previous = output.exists() and any(output.iterdir())
        unresolved = []
        if has_previous:
            if previous.exists():
                raise FileExistsError('Preserve the previous model history')
            with ExitStack() as locks:
                for name in ('controller.lock','canonical_conditional_handoff.lock','canonical_handoff.lock'):
                    lock = locks.enter_context((output/name).open('a'))
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                # A released controller lock is insufficient if native children
                # survived it. Never relocate an output still in use by a fit.
                rows = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','comm=,args='],text=True)
                for row in rows.splitlines():
                    comm, _, command = row.strip().partition(' ')
                    if comm in ('combine','text2workspace') or 'text2workspace.py ' in command:
                        if str(output) in command:
                            raise RuntimeError('A previous native worker is still using the official output')
                aggregate = output/'workflow_state.json'
                if aggregate.exists():
                    old_results = read_json(aggregate).get('results',{})
                    selected, _ = ready_fit_points(source)
                    unresolved = [p['model']+'/'+p['mass'] for p in selected
                        if old_results.get(p['model']+'/'+p['mass'],{}).get('status') != 'complete']
                previous.parent.mkdir(parents=True,exist_ok=True)
                output.rename(previous)
        output.mkdir(parents=True,exist_ok=True)
        write_json(marker,dict(canonical_model='era_merged',official_result=True,
            adopted=time.time(),previous_model='separate_eras',
            previous_output=str(previous) if has_previous else None,
            previous_artifact_paths_relocated_from=str(output) if has_previous else None,
            previously_unresolved=unresolved,
            observed_sr_used=False,input_check_scope=INPUT_CHECK_SCOPE))


def _run_integrated_grid(manifest, output, workers, draining_handoff=None):
    if not 1 <= workers <= 115:
        raise ValueError('Use 1..115 workers within aggregate120/downstream116')
    if sys.platform != 'linux':
        raise RuntimeError('Local fitting is retired; run the shared pool on hep2')
    manifest, output = internal_path(manifest), internal_path(output)
    source = read_json(manifest)
    points, selection = ready_fit_points(source)
    canonical_model_output(source, output)
    model_marker = output/'model.json'
    previous_failures = set(read_json(model_marker).get('previously_unresolved',[])) if model_marker.exists() else set()
    output.mkdir(parents=True, exist_ok=True)
    with canonical_controller_ownership(manifest, output, draining_handoff) as handoff:
        results = {}
        pending = []
        reserved = {}
        for point in points:
            key = point['model']+'/'+point['mass']
            if handoff is not None and key in handoff['reserved']:
                reserved[key] = point
                continue
            saved = saved_point_result(point, output)
            if saved is not None:
                results[key] = saved
            else:
                pending.append(point)
        # Existing failures go first; there are no idle recovery-only slots.
        pending.sort(key=lambda p: (p['model']+'/'+p['mass'] not in previous_failures,
                                    not (output/p['model']/p['mass']/'state.json').exists(),
                                    p['model'], p['mass']))
        env = runtime(output) if pending or reserved else None
        stopped = threading.Event()
        old_handler = signal.signal(signal.SIGINT, lambda *_: stopped.set())
        progress = {}
        state_path = output/'workflow_state.json'
        started = time.time()

        def snapshot(active, external=0, memory=0, admission_limit=0, final=False):
            complete = sum(r.get('status') == 'complete' for r in results.values())
            failed = [k for k, r in results.items() if r.get('status') != 'complete']
            phase = 'fits_complete_pending_plots' if complete == len(points) else (
                'needs_attention' if final and failed else 'drained' if final else
                'draining' if stopped.is_set() else 'running')
            exclusive = handoff is None or handoff['exclusive']
            snapshot_path = state_path if exclusive else output/handoff.get('state_file','canonical_handoff_state.json')
            record = dict(selection, canonical_limits_pipeline=True,
                canonical_model=source.get('canonical_model','separate_eras'),
                official_result=source.get('official_result',False),
                analysis_root=str(ROOT),
                status=phase, pid=None if final else os.getpid(), started=started, updated=time.time(),
                manifest=str(manifest.relative_to(ROOT)), input_check_scope=INPUT_CHECK_SCOPE,
                completed=complete, remaining=len(points)-complete, failed=failed,
                active_primary=[k for k, v in progress.items() if v['phase'] == 'primary'],
                active_recovery=[k for k, v in progress.items() if v['phase'] != 'primary'],
                active=len(active), pending=len(pending), externally_reserved=sorted(reserved),
                draining_handoff_sha256=handoff['sha256'] if handoff else None,
                exclusive_controller_ownership=exclusive, external_slots=external,
                available_memory_bytes=memory, admission_limit=admission_limit, results=results,
                requested_workers=workers, worker_memory_reservation_bytes=3*1024**3,
                full_workflow_complete=False, observed_sr_used=False)
            write_json(snapshot_path, record)
            if handoff is not None and exclusive:
                write_json(output/'canonical_handoff_state.json', dict(record, handoff_completed=True))
            if not exclusive:
                return
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
                    canonical_model=source.get('canonical_model','separate_eras'),
                    official_result=source.get('official_result',False))
                write_json(output/model/'expected_limits.json', summary)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}
                snapshot(active)
                while active or ((pending or reserved) and not stopped.is_set()):
                    for key in _refresh_draining_handoff(handoff):
                        point = reserved.pop(key)
                        saved = saved_point_result(point, output)
                        if saved is not None:
                            results[key] = saved
                        else:
                            pending.append(point)
                    pending.sort(key=lambda p: (p['model']+'/'+p['mass'] not in previous_failures,
                                                not (output/p['model']/p['mass']/'state.json').exists(),
                                                p['model'], p['mass']))
                    external, memory = fit_pool_resources()
                    resident = fit_pool_resident_memory(output)
                    limit = fit_pool_capacity(workers, external, memory, len(active),
                                              shutil.disk_usage(output).free,
                                              resident_memory=resident, reserved=len(reserved))
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
                    if not active and (pending or reserved) and not stopped.is_set():
                        time.sleep(5)
                snapshot(active, final=True)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        return 0 if sum(r.get('status') == 'complete' for r in results.values()) == len(points) else 1


def run(manifest, output, workers, plot_range_only=False, draining_handoff=None):
    source = read_json(internal_path(manifest))
    if source.get('scope') == 'full_combined_systematic_production' and source.get('run3_systematics'):
        return run_integrated_grid(manifest, output, workers, draining_handoff)
    if draining_handoff is not None:
        raise ValueError('Draining handoff is only for the approved six-era canonical pool')
    return run_primary_grid(manifest, output, workers, plot_range_only)


def comparison_candidate_score(points, baseline, candidate):
    """Fixed topology/dM cell weights; incomplete coverage never selects a winner."""
    edges = (0,175,250,350,500,750,3000)
    cells = {}; rows = {}; missing = []
    for point in points:
        key = point['model']+'/'+point['mass']
        match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass'])
        if not match or key in rows or key in missing:
            raise ValueError('Invalid or duplicate ranking mass point')
        stop,lsp = map(int,match.groups()); delta = stop-lsp
        cell = next(((a,b) for a,b in zip(edges,edges[1:]) if a <= delta < b),None)
        if cell is None: raise ValueError('Ranking mass outside the fixed delta-M cells')
        label = point['model']+':'+str(cell[0])+','+str(cell[1])
        group = cells.setdefault(label,dict(model=point['model'],delta_m=list(cell),requested=0,values=[]))
        group['requested'] += 1
        if key not in baseline or key not in candidate:
            missing.append(key); continue
        reference,value = baseline[key],candidate[key]
        if not all(math.isfinite(x) and x > 0 for x in (reference,value)):
            raise ValueError('Nonpositive/nonfinite actual expected median')
        ratio = value/reference; log_ratio = math.log(ratio)
        rows[key] = dict(baseline_expected_median=reference,combined_expected_median=value,
            ratio=ratio,log_ratio=log_ratio,cell=label)
        group['values'].append(log_ratio)
    reports = []
    for label,group in sorted(cells.items()):
        values = group.pop('values')
        reports.append(dict(group,cell=label,verified=len(values),
            mean_log_ratio=sum(values)/len(values) if values else None))
    complete = bool(points) and not missing
    score = sum(c['mean_log_ratio'] for c in reports)/len(reports) if complete else None
    return dict(status='complete' if complete else 'incomplete',ranking_eligible=complete,
        requested=len(points),verified=len(rows),missing_points=missing,
        mean_log_ratio=score,geometric_mean_ratio=math.exp(score) if complete else None,
        direction='lower_is_better',cell_weighting='equal_nonempty_topology_delta_m_cells',
        within_cell_weighting='equal_mass_points',delta_m_edges=list(edges),
        mass_regression_veto=False,cells=reports,points=rows)


def ranking_verified_medians(grid, limits):
    """Read exact saved numerical attempts without launching or promoting a fit."""
    from TROTASR.workflows.plotting_limit_contour import verified_current_results
    digest = IntegrityCache(); values = {}; pending = {}; terminal = []
    for point in grid['points']:
        verify_card(point,digest)
        key = point['model']+'/'+point['mass']; path = limits/key/'state.json'
        if not path.exists():
            pending[key] = 'not_started'; continue
        state = read_json(path)
        if state.get('card_sha256') != point['card_sha256']:
            raise ValueError('Ranking point belongs to another card')
        if state.get('pid') or state.get('status') not in ('complete','failed'):
            pending[key] = state.get('status','unknown'); continue
        terminal.append(point)
    # Reuse the canonical collector, including exact recovery receipts/native
    # brackets, in history order. A contour-only result is not a full fit.
    results,evidence = verified_current_results(dict(grid,points=terminal),limits,
        allow_partial=True,contour_only=False) if terminal else ({},{})
    for point in terminal:
        key = point['model']+'/'+point['mass']; accepted = evidence[key]['accepted']
        if accepted:
            values[key] = {float(k):v for k,v in accepted['validation']['quantiles'].items()}[.5]
        else:
            pending[key] = 'no_verified_saved_limit'
    return values,evidence,pending


def rank_comparison_candidates(manifests, output):
    """Use actual CLs; keep validation-NN and explicitly all-MC cut studies separate."""
    from TROTASR.workflows.build_nominal_grid import (
        FIXED_BASELINE_GRID,FIXED_BASELINE_SHA256,CANDIDATE_RANKING_POLICY,load_medium_candidate,
        CUTBASED_SCHEMA,CUTBASED_CAVEAT,MET_CATEGORY_SCHEMA,MET_CATEGORY_CAVEAT,met_category_caveat)
    if sys.platform != 'linux': raise RuntimeError('Ranking ROOT checks must remain on hep2')
    output = internal_path(output)
    directory = ROOT/'stats/topology_transition_nominal/combined_optimization'
    if output.parent != directory or output.suffix != '.json':
        raise ValueError('Ranking output must stay in the active combined optimization directory')
    if output.exists(): raise ValueError('Preserve an existing ranking snapshot; choose a new filename')
    baseline_path = ROOT/FIXED_BASELINE_GRID
    if sha256(baseline_path) != FIXED_BASELINE_SHA256: raise ValueError('Fixed ranking baseline changed')
    baseline = read_json(baseline_path); verify_comparison_grid(baseline)
    baseline_limits = baseline_path.parent.parent/'limits'
    if read_json(baseline_limits/'contract.json')['manifest_sha256'] != FIXED_BASELINE_SHA256:
        raise ValueError('Baseline limits/grid contract mismatch')
    references,baseline_evidence,baseline_pending = ranking_verified_medians(baseline,baseline_limits)
    rows = []; seen = set(); population_kind = None; population_identity = None
    for path in manifests:
        path = internal_path(path); grid = read_json(path); verify_comparison_grid(grid)
        if grid.get('medium_test_comparison') is not None:
            raise ValueError('Frozen test results cannot be used for candidate ranking or reselection')
        candidate = grid.get('medium_candidate')
        if not candidate or grid['variant'] != 'combined' or load_medium_candidate(candidate['contract_path']) != candidate:
            raise ValueError('Ranking requires the exact combined medium candidate')
        kind=('all_mc_met_categories' if candidate['schema']==MET_CATEGORY_SCHEMA else
              'all_mc_cutbased' if candidate['schema']==CUTBASED_SCHEMA else 'validation_nn')
        if population_kind is not None and kind!=population_kind:
            raise ValueError('Full-MC cut-based and historical validation-NN candidates cannot share a ranking')
        population_kind=kind
        if kind=='all_mc_cutbased':
            identity=(candidate['populations'],candidate['feature_definitions'])
            if population_identity is not None and identity!=population_identity:
                raise ValueError('Rank cut-based candidates on the same full-MC population and physical definitions')
            population_identity=identity
        if kind=='all_mc_met_categories':
            identity=(candidate['parent_populations'],candidate['feature_definitions'],candidate['category_definition'])
            if population_identity is not None and identity!=population_identity:
                raise ValueError('S0/S1 require identical parent MC, physical definitions and MET categories')
            population_identity=identity
        identity = candidate['candidate_id']
        if identity in seen or path != directory/identity/'combined/grid/manifest.json':
            raise ValueError('Duplicate candidate or incorrect ranking directory')
        seen.add(identity)
        if [(p['model'],p['mass']) for p in grid['points']] != [(p['model'],p['mass']) for p in baseline['points']]:
            raise ValueError('Rank every candidate on the same complete baseline mass grid')
        limits = path.parent.parent/'limits'; contract = limits/'contract.json'
        if contract.exists():
            if read_json(contract)['manifest_sha256'] != sha256(path):
                raise ValueError('Candidate limits/grid contract mismatch')
            values,evidence,pending = ranking_verified_medians(grid,limits)
        else:
            values,evidence,pending = {},{},dict.fromkeys(
                (p['model']+'/'+p['mass'] for p in grid['points']),'not_started')
        score = comparison_candidate_score(baseline['points'],references,values)
        candidate_field=('met_category_candidate' if kind=='all_mc_met_categories' else
                         'cutbased_candidate' if kind=='all_mc_cutbased' else 'validation_candidate')
        rows.append(dict(score,candidate_id=identity,manifest=str(path.relative_to(ROOT)),
            manifest_sha256=sha256(path),contract_sha256=sha256(contract) if contract.exists() else None,
            evidence=evidence,pending=pending,**{candidate_field:candidate}))
    if not rows: raise ValueError('No candidates requested')
    exact_global_pool=population_kind=='all_mc_met_categories' and all(
        'global_cuts' in r['met_category_candidate']['selection'] for r in rows)
    if population_kind=='all_mc_met_categories' and not exact_global_pool and (len(rows)!=2 or
            {r['met_category_candidate']['selection']['id'] for r in rows}!={'S0','S1'}):
        raise ValueError('MET-category comparison is exactly the approved S0 and S1 pair')
    ordered = sorted((r for r in rows if r['ranking_eligible']),key=lambda r:(r['mean_log_ratio'],r['candidate_id']))
    complete = len(ordered)==len(rows) and not baseline_pending
    report = dict(schema='combined_expected_cls_ranking_v1',policy=CANDIDATE_RANKING_POLICY,
        status='complete' if complete else 'provisional',observed_sr_used=False,
        baseline_manifest_sha256=FIXED_BASELINE_SHA256,baseline_evidence=baseline_evidence,
        baseline_pending=baseline_pending,candidates=rows,
        ranking_of_complete_candidates=[r['candidate_id'] for r in ordered],
        winner=ordered[0]['candidate_id'] if complete else None,
        claim='Best only within the explicitly listed validation candidate pool; not a global optimum',
        physical_model='Corrected nominal SR-only MC with autoMCStats 10 1 1; no propagated systematics')
    if population_kind=='all_mc_cutbased':
        report.update(schema='combined_cutbased_expected_cls_ranking_v1',evaluation_partition='all_mc',
            neural_network_used=False,independent_validation=False,statistical_caveat=CUTBASED_CAVEAT,
            populations=population_identity[0],historical_partitions_used=['train','validation','test'],
            claim='Best only within the explicitly listed full-MC cut-based pool; not a global optimum or independent validation')
    if population_kind=='all_mc_met_categories':
        report.update(schema='combined_met_category_expected_cls_comparison_v1',evaluation_partition='all_mc',
            neural_network_used=False,independent_validation=False,statistical_caveat=met_category_caveat(rows[0]['met_category_candidate']),
            parent_populations=population_identity[0],category_definition=population_identity[2],
            historical_partitions_used=['train','validation','test'],automatic_selection=False,final_adoption=False,
            best_by_metric=report['winner'],winner=None,
            claim='Explicit S0/S1 comparison on the same parent MC and MET categories; no automatic adoption or independent validation')
        if exact_global_pool:
            report['claim']='Explicit global-cut candidates on the same parent MC and fixed MET categories; no automatic adoption or independent validation'
    write_json(output,report)
    return 0 if complete else 1


def run_met_candidate_batch(catalogue, output, workers, simultaneous=3):
    """Build and fit a declared fixed-MET pool with canonical subprocesses.

    Independent candidates share the existing account-wide admission guard.
    One failed candidate does not cancel valid work in the rest of the pool.
    """
    from TROTASR.workflows.build_nominal_grid import load_medium_candidate, FIXED_BASELINE_GRID, MET_CATEGORY_SCHEMA
    import hashlib
    output=internal_path(output);catalogue=Path(catalogue).resolve()
    approved=ROOT.parent/'sidestudy/topology_transition/cutbased/met_categories'
    if catalogue.parent!=approved or catalogue.name not in ('state.json','exact_selection_state.json','shap_selection_state.json'):
        raise ValueError('Use the explicitly approved fixed-MET candidate catalogue')
    # This is the same explicit sidestudy boundary as load_medium_candidate;
    # do not weaken internal_path globally for ordinary production inputs.
    catalog=json.loads(catalogue.read_text());paths=[Path(c['contract']) for c in catalog['candidates']]
    if not paths or not 1<=simultaneous<=3 or not 1<=workers<=24:
        raise ValueError('MET batch uses 1-3 candidates, at most24 workers each')
    candidates=[load_medium_candidate(p) for p in paths]
    if any(c['contract_sha256']!=r['contract_sha256'] for c,r in zip(candidates,catalog['candidates'])):
        raise ValueError('Candidate catalogue contract hash changed')
    if len({c['candidate_id'] for c in candidates})!=len(candidates) or any(c['schema']!=MET_CATEGORY_SCHEMA for c in candidates):
        raise ValueError('One explicit, unique fixed-MET candidate pool required')
    parent=ROOT/'stats/combined_systematics_sr118/grid/manifest.json'
    baseline=ROOT/FIXED_BASELINE_GRID
    output.parent.mkdir(parents=True,exist_ok=True)
    statepath=output.with_name(output.stem+'_state.json')
    identity=dict(catalogue=str(catalogue),catalogue_sha256=hashlib.sha256(catalogue.read_bytes()).hexdigest(),
        contracts={c['candidate_id']:c['contract_sha256'] for c in candidates})
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if statepath.exists() and read_json(statepath)['identity']!=identity:
            raise ValueError('Preserve a batch with different candidate definitions')
        env=runtime(output.with_suffix(''))
        statuses={};manifests=[]
        def execute(candidate,path):
            directory=ROOT/'stats/topology_transition_nominal/combined_optimization'/candidate['candidate_id']
            directory.mkdir(parents=True,exist_ok=True)
            grid=directory/'combined/grid/manifest.json'
            with (directory/'batch.log').open('a') as log:
                try:
                    for variant in ('medium','combined'):
                        command=['python3',str(ROOT/'workflows/build_nominal_grid.py'),
                            '--sr-only-comparison',variant,'--output',str(directory/variant/'grid'),
                            '--parent-grid',str(parent),'--medium-candidate',str(path)]
                        command += (['--medium-inputs',*[str(path.parent/('medium_'+y+'.json')) for y in ('2024','2025')]]
                            if variant=='medium' else ['--component-grids',str(baseline),str(directory/'medium/grid/manifest.json')])
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run(['python3',str(Path(__file__).resolve()),'--manifest',str(grid),
                        '--output',str(directory/'combined/limits'),'--workers',str(workers)],
                        env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    return dict(status='complete',manifest=str(grid))
                except subprocess.CalledProcessError as error:
                    return dict(status='needs_attention',manifest=str(grid),returncode=error.returncode,log=str(directory/'batch.log'))
        with ThreadPoolExecutor(max_workers=simultaneous) as pool:
            active={pool.submit(execute,c,p):c['candidate_id'] for c,p in zip(candidates,paths)}
            while active:
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    key=active.pop(future)
                    try:statuses[key]=future.result()
                    except Exception as error:statuses[key]=dict(status='needs_attention',error=repr(error))
                    print('MET_CLS_CANDIDATE',key,statuses[key],flush=True)
                write_json(statepath,dict(identity=identity,pid=os.getpid(),status='running',updated=time.time(),
                    total=len(candidates),finished=statuses,pending=list(active.values()),workers_per_candidate=workers,
                    simultaneous=simultaneous,expected_only=True))
        manifests=[Path(statuses[c['candidate_id']]['manifest']) for c in candidates
                   if statuses[c['candidate_id']].get('manifest') and Path(statuses[c['candidate_id']]['manifest']).exists()]
        all_complete=all(s['status']=='complete' for s in statuses.values())
        if all_complete and not output.exists():rank_comparison_candidates(manifests,output)
        plots={}
        if all_complete:
            references=[c for c in candidates if c['selection'].get('global_cuts')==[]]
            if len(references)==1:
                reference=references[0]['candidate_id'];ranking=read_json(output)
                winner=ranking['ranking_of_complete_candidates'][0]
                # If S2 itself wins, show the strongest tested extra cut against
                # it, not an overlay of S2 on itself labelled an improvement.
                challenger=next((c for c in ranking['ranking_of_complete_candidates'] if c!=reference),None)
                if challenger:
                    directory=ROOT/'stats/topology_transition_nominal/combined_optimization'
                    figures=ROOT/'plots/topology_transition_nominal'/output.stem
                    command=['bash',str(ROOT/'workflows/run_inference.sh'),str(ROOT/'workflows/plotting_limit_contour.py'),
                        '--global-cut-comparison','--grid',str(directory/challenger/'combined/grid/manifest.json'),
                        '--limits',str(directory/challenger/'combined/limits'),'--output',str(figures),
                        '--overlay-grid',str(directory/reference/'combined/grid/manifest.json'),
                        '--overlay-limits',str(directory/reference/'combined/limits'),
                        '--baseline-grid',str(baseline),'--baseline-limits',str(baseline.parent.parent/'limits'),
                        '--comparison-style-reference',str(ROOT/'plots/topology_transition_nominal/score_edges_20261013/limits/plot_manifest.json')]
                    with output.with_name(output.stem+'_plots.log').open('a') as log:
                        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                    plots=dict(status='rendered_pending_visual_qa' if code==0 else 'needs_attention',
                        output=str(figures),reference=reference,challenger=challenger,best_by_metric=winner,returncode=code)
        write_json(statepath,dict(identity=identity,status='fits_complete' if all_complete else 'needs_attention',
            finished=statuses,total=len(candidates),expected_only=True,plots=plots))
        return 0 if all_complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'stats/run3/grid/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'stats/run3/limits')
    p.add_argument('--workers',type=int,default=70)
    p.add_argument('--resume', action='store_true',
        help='Reuse accepted point receipts and continue unstarted finite commands (also the default)')
    p.add_argument('--plot-range-only',action='store_true',
        help='Fit only the approved 895 displayed points of the new six-era 955-card grid')
    p.add_argument('--draining-handoff',type=Path,
        help='Explicit two-controller main-only drain evidence for reserved-point coexistence')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-eos',action='store_true')
    modes.add_argument('--submit-eos',action='store_true')
    modes.add_argument('--collect-eos',action='store_true')
    modes.add_argument('--condor-point',help=argparse.SUPPRESS)
    modes.add_argument('--rank-candidates',nargs='+',type=Path,
        help='Rank saved combined expected CLs within one validation-NN or full-MC cut-based pool')
    modes.add_argument('--met-candidate-batch',type=Path,help='Canonical build/fit/rank for a declared fixed-MET candidate catalogue')
    modes.add_argument('--sensitivity-map',action='store_true',help='Read-only-baseline nuisance-group expected-median comparisons')
    modes.add_argument('--plot-sensitivity-map',action='store_true',help='Plot saved conditional sensitivity improvements')
    p.add_argument('--sensitivity-prepare-only',action='store_true')
    p.add_argument('--sensitivity-points',nargs='+',help='Optional exact model/mass keys; never changes the full-grid contract')
    p.add_argument('--eos-runtime-manifest',type=Path)
    p.add_argument('--eos-proxy',type=Path)
    a=p.parse_args()
    if a.manifest == ROOT/'stats/run3/grid/manifest.json':
        a.plot_range_only=True
    if a.plot_range_only and any((a.prepare_eos,a.submit_eos,a.collect_eos,a.condor_point,
            a.rank_candidates,a.met_candidate_batch,a.sensitivity_map,a.plot_sensitivity_map)):
        p.error('--plot-range-only applies only to the canonical local fit grid')
    if a.plot_sensitivity_map:
        from TROTASR.workflows.sensitivity_map import plot
        raise SystemExit(plot(a.output))
    if a.sensitivity_map:
        if not a.manifest:p.error('--manifest is required')
        from TROTASR.workflows.sensitivity_map import run as sensitivity_run
        raise SystemExit(sensitivity_run(a.manifest,a.output,a.workers,a.sensitivity_prepare_only,a.sensitivity_points))
    if a.met_candidate_batch:
        raise SystemExit(run_met_candidate_batch(a.met_candidate_batch,a.output,a.workers))
    if a.rank_candidates:
        if a.manifest: p.error('--rank-candidates takes its own manifest list')
        raise SystemExit(rank_comparison_candidates(a.rank_candidates,a.output))
    if not a.manifest: p.error('--manifest is required for fits or EOS operations')
    if a.prepare_eos:
        if not a.eos_runtime_manifest or not a.eos_proxy: p.error('Explicit EOS runtime manifest and proxy are required')
        raise SystemExit(prepare_eos(a.manifest,a.output,a.eos_runtime_manifest,a.eos_proxy))
    if a.submit_eos: raise SystemExit(submit_eos(a.output))
    if a.collect_eos: raise SystemExit(collect_eos(a.manifest,a.output))
    if a.condor_point: raise SystemExit(condor_point(a.manifest,a.output,a.condor_point))
    raise SystemExit(run(a.manifest,a.output,a.workers,a.plot_range_only,a.draining_handoff))
