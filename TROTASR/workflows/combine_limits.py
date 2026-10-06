"""Nominal expected Combine grid with the adopted finite crossing-checked policy.

No cards, ROOT files, fit values, payloads or analysis modules outside TROTASR
are read. CMSSW is a separately identified numerical software runtime. Plotting
is a later stage; completed fits do not imply complete AN deliverables.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.limit_validation import validate, REL_ACC, ABS_ACC
from TROTASR.workflows.nominal_campaign import resources, capacity

FIELDS = {.025:'expected_m2', .16:'expected_m1', .5:'expected', .84:'expected_p1', .975:'expected_p2'}
ATTEMPTS = (('bisection_s0', .1, 1.), ('bisection_s0_refined', .01, 2.))


def verify_combined_grid(record):
    """The new scope is accepted only with the exact adopted model and grid."""
    if record.get('scope') != 'full_combined_systematic_production':
        return
    from TROTASR.workflows.build_nominal_grid import CR12_POLICY
    from TROTASR.workflows.diagnostic_common import require_revised_record
    require_revised_record(record)
    policy = record.get('statistical_revision', record.get('contract', {}).get('statistical_revision'))
    points = record.get('points', [])
    counts = {m: sum(p['model'] == m for p in points) for m in ('T2tt', 'T2bW', 'T2tb')}
    if (policy != CR12_POLICY or record.get('bins') != 428
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
    if cmssw != Path('/hep2-scratch/twkim/envs/combine_20260914/CMSSW_14_1_0_pre4'):
        raise ValueError('Unverified numerical runtime')
    if config['cmsset_default'] != '/cvmfs/cms.cern.ch/cmsset_default.sh':
        raise ValueError('Unverified CMS software initialization')
    guard = internal_path(ROOT/config['guard'])
    for d in ('tmp','cache'):
        (output/d).mkdir(parents=True,exist_ok=True)
    env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(output/'tmp'),
        XDG_CACHE_HOME=str(output/'cache'))
    raw = subprocess.check_output(['bash','--noprofile','--norc','-c',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(scramv1 runtime -sh)" && env -0'],
        cwd=cmssw/'src',env=env)
    env = dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    # Discard inherited Python analysis paths. CMSSW's Python3 software path
    # is added by this internal copy of the adopted compatibility guard.
    env['PYTHONPATH'] = str(guard)
    soft,hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < 256*1024**2:
        resource.setrlimit(resource.RLIMIT_STACK,(min(256*1024**2,hard) if hard!=resource.RLIM_INFINITY else 256*1024**2,hard))
    return env


def fit_command(workspace, mass, rmax, tolerance):
    return ['combine','-M','AsymptoticLimits',str(workspace),'-m','120',
        '--run','blind','-n','_'+mass,
        '--cminDefaultMinimizerStrategy','0','--cminDefaultMinimizerTolerance',str(tolerance),
        '--cminFallbackAlgo','Minuit2,Migrad,1:0.1',
        '--minosAlgo','bisection','--picky','--rMax',str(rmax),
        '--rRelAcc',str(REL_ACC),'--rAbsAcc',str(ABS_ACC),'-v','2']


def verify_card(point, digest=sha256):
    card = internal_path(ROOT/point['card'])
    if digest(card) != point['card_sha256']:
        raise ValueError('Card changed: '+str(card))
    meta = read_json(card.with_suffix('.manifest.json'))
    mass = re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass'])
    if (not mass or point['model'] not in ('T2tt','T2tb','T2bW') or meta['cr_only']
        or not meta['sr_data_blinded'] or meta['card_sha256']!=point['card_sha256']
        or list(map(int,meta['years'])) != [2024,2025]
        or meta['auto_mc_stats'] != [10,1,1]
        or (meta['topology'],meta['mStop'],meta['mLSP']) != (point['model'],int(mass[1]),int(mass[2]))):
        raise ValueError('Card/point/blinding contract mismatch')
    for source in meta['inputs']:
        folder = internal_path(ROOT/source['path'])
        manifest = folder/'model_manifest.json'
        if digest(manifest) != source['sha256']:
            raise ValueError('Template manifest changed')
        record=read_json(manifest)
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
            raise ValueError('Card shapes must reference exactly the verified four template files')
    # Every shapes directive must resolve inside the package, even if a card
    # was authored by a different caller. ROOT may otherwise read any path.
    for line in card.read_text().splitlines():
        f=line.split()
        if f and f[0]=='shapes':
            internal_path(card.parent/f[3])
    return card


def run_point(point, output, env, digest=sha256):
    card = verify_card(point, digest)
    work = output/point['model']/point['mass']
    work.mkdir(parents=True,exist_ok=True)
    with (work/'point.lock').open('a') as lock:
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
        elif sha256(workspace)!=record['workspace_sha256']:
            raise ValueError('Workspace changed')
        completed_attempts={a['name'] for a in record['attempts']}
        for label,tolerance,scale in ATTEMPTS:
            if label in completed_attempts:continue
            here=work/label
            if here.exists():
                raise ValueError('Interrupted fit attempt retained; explicit recovery required')
            here.mkdir();(here/'tmp').mkdir()
            command=fit_command(workspace,point['mass'],float(point.get('rmax_seed',20.))*scale,tolerance)
            log=here/'combine.log'
            with log.open('x') as stream:
                proc=subprocess.Popen(command,cwd=here,env=dict(env,TMPDIR=str(here/'tmp')),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                record.update(phase='fit',pid=proc.pid,command=command,current_attempt=label)
                write_json(state_path,record);rc=proc.wait()
            result=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            passed,evidence=validate(result,log,tolerance) if rc==0 else (False,dict(exit_code=rc))
            verify_card(point, digest)
            attempt=dict(name=label,tolerance=tolerance,passed=passed,exit_code=rc,
                log=str(log.relative_to(ROOT)), log_sha256=sha256(log), command=command,
                output=str(result.relative_to(ROOT)), output_sha256=sha256(result) if result.is_file() else None,
                validation=evidence)
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


def run(manifest,output,workers):
    if not 1<=workers<=99:raise ValueError('Reserve one CPU slot for controller')
    output=internal_path(output);manifest=internal_path(manifest)
    source=read_json(manifest)
    if source['status']!='cards_ready' or source['scope'] not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production'):
        raise ValueError('Production expected contours require complete full-input cards')
    verify_combined_grid(source)
    points=source['points']
    counts={m:sum(p['model']==m for p in points) for m in ('T2tt','T2tb','T2bW')}
    if counts!=source.get('expected_points_by_model') or not all(counts.values()):
        raise ValueError('All three complete signal grids are required')
    keys=[p['model']+'/'+p['mass'] for p in points]
    if len(set(keys))!=len(keys) or not points:raise ValueError('Empty/duplicate grid')
    integrity = IntegrityCache()
    for p in points:verify_card(p, integrity)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=dict(manifest_sha256=sha256(manifest),code={n:sha256(ROOT/n) for n in (
            'workflows/combine_limits.py','workflows/limit_validation.py','stats/runtime/sitecustomize.py','stats/runtime.json')})
        contract_path=output/'contract.json'
        if contract_path.exists() and read_json(contract_path)!=contract:raise ValueError('Divergent fit contract')
        write_json(contract_path,contract)
        env=runtime(output)
        results={};pending=iter(points);exhausted=False
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
                    total=len(points),active=len(active),completed=sum(r['status']=='complete' for r in results.values()),
                    failed=[k for k,r in results.items() if r['status']!='complete'],external_slots=external,
                    observed_sr_used=False,full_workflow_complete=False))
                if not active and not exhausted:time.sleep(5)
        for model in ('T2tt','T2tb','T2bW'):
            write_json(output/model/'expected_limits.json',collect(points,results,model,source['scope']))
        failed=[k for k,r in results.items() if r['status']!='complete']
        write_json(output/'state.json',dict(status='fits_complete_pending_plots' if not failed else 'needs_attention',
            total=len(points),completed=len(points)-len(failed),failed=failed,results=results,
            full_workflow_complete=False,observed_sr_used=False))
        return 1 if failed else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=99)
    a=p.parse_args();raise SystemExit(run(a.manifest,a.output,a.workers))
