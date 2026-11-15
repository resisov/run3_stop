#!/usr/bin/env python3
"""Produce signal intermediates: one NanoAOD file and one target era per job.

The stage owns Condor submission, bounded retries and collection. Production
outputs stay separate from the accepted input registry until explicitly adopted.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# EosSubmit transfers this module with the package and runtime archive.
if not __package__:
    with tarfile.open('producer.tgz') as bundle:
        bundle.extractall(Path.cwd())
    command = ('source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh\n'
               'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
               'export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1\n'
               'exec python3 -m TROTASR produce ' + shlex.join(sys.argv[1:]))
    raise SystemExit(subprocess.call(['/bin/bash', '--noprofile', '--norc', '-c', command]))

from TROTASR.utils.paths import ROOT, ERAS, EARLY_ERAS, internal_path
from TROTASR.utils.io import read_json, write_json

PRODUCTION = ROOT / 'production'
SIGNAL_NORM_BASIS = "registered_mc_runs_sumw_v1"
REPLACEMENT_SIGNAL_POPULATION = "regenerated_signal_uuid"
REPLACEMENT_SIGNAL_NAME = re.compile(
    r"signal_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.root", re.IGNORECASE)
REPLACEMENT_FILES_PER_ERA = 187
EOS_ROOT = '/eos/user/t/taiwoo/run3_stop/decaf/TROTASR'
PROXY = '/eos/user/t/taiwoo/decaf/x509up_u147757'
LCG = '/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh'
TENSORFLOW = '/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh'
CMSSW = '/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_15_0_10/src'


def read(path):
    return json.loads(Path(path).read_text())


def run(command, **kwargs):
    print(shlex.join(map(str, command)), flush=True)
    attempts = 3 if str(command[0]) in ('xrdcp', 'xrdfs') else 1
    if attempts > 1:
        kwargs.setdefault('timeout', 1800 if str(command[0]) == 'xrdcp' else 120)
    for attempt in range(attempts):
        try:
            return subprocess.run(list(map(str, command)), check=True, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt + 1 == attempts:
                if exc.stdout:
                    print(exc.stdout, flush=True)
                if exc.stderr:
                    print(exc.stderr, file=sys.stderr, flush=True)
                raise
            time.sleep(5 * (attempt + 1))


def in_environment(setup, arguments, *, log=None):
    """Use the installed production runtimes, with no inherited LCG libraries."""
    env = {k: v for k, v in os.environ.items() if k not in (
        'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'ROOTSYS', 'CMAKE_PREFIX_PATH')}
    env.update(PATH='/usr/local/bin:/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
    command = setup + '\ncd ' + shlex.quote(str(ROOT.parent)) + '\n'
    command += 'exec python3 -m TROTASR produce ' + shlex.join(arguments)
    run(['/bin/bash', '--noprofile', '--norc', '-c', command], env=env,
        stdout=log, stderr=subprocess.STDOUT if log is not None else None)


def prepare():
    inventory = read(ROOT / 'results/signal_source_inventory.json')
    sources = []
    for dataset in inventory['datasets']:
        if dataset['replacement_das_status'] != 'VALID':
            raise ValueError('Production source is not VALID')
        for source in dataset['files']:
            if not source.get('parents'):
                raise ValueError('Missing exact MiniAOD parents: ' + source['name'])
            sources.append({**source, 'dataset': dataset['replacement_dataset']})
    # Small files first gives an early measured throughput without partial files.
    sources.sort(key=lambda source: (source['nevents'], source['name']))
    tasks = []
    for era in ERAS:
        for source in sources:
            tasks.append(dict(era=era, source=source,
                              name='signal_' + Path(source['name']).stem))
    PRODUCTION.mkdir(parents=True, exist_ok=True)
    manifest = dict(eras=list(ERAS), files=len(sources), tasks=tasks,
                    events=sum(s['nevents'] for s in sources),
                    physics='current target-era objects; raw signed generator weights; no multiplicative SF',
                    accepted_inputs_replaced=False)
    path = PRODUCTION / 'tasks.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Active production task definitions differ')
    write_json(path, manifest)
    (PRODUCTION / 'logs').mkdir(exist_ok=True)
    (PRODUCTION / 'outputs').mkdir(exist_ok=True)
    (PRODUCTION / 'bundles').mkdir(exist_ok=True)
    bundle = PRODUCTION / 'bundles/producer.tgz'
    if not bundle.exists():
        with tarfile.open(bundle, 'w:gz') as archive:
            for relative in ('__init__.py', '__main__.py', 'utils', 'models/TROTA',
                             'scales/JME', 'scales/EGM', 'scales/MUO', 'scales/TAU',
                             'scales/object_shapes.json', 'production/tasks.json'):
                archive.add(ROOT / relative, arcname='TROTASR/' + relative,
                            filter=lambda item: None if '__pycache__' in item.name or item.name.endswith('.pyc') else item)
            archive.add(ROOT / 'workflows/__init__.py', arcname='TROTASR/workflows/__init__.py')
            archive.add(ROOT / 'workflows/produce.py', arcname='TROTASR/workflows/produce.py')
    print(json.dumps(dict(files=len(sources), eras=list(ERAS), jobs=len(tasks),
                          bundle_bytes=bundle.stat().st_size)), flush=True)


def receipt_path(task):
    return PRODUCTION / 'collected' / task['era'] / (task['name'] + '.json')


def submit(eras, limit=None):
    tasks = read(PRODUCTION / 'tasks.json')['tasks']
    pending = []
    queue = subprocess.run(['condor_q', os.environ.get('USER', 'taiwoo'), '-json',
                            '-attributes', 'ClusterId,ProcId,TrotasrTask,JobStatus'],
                           check=True, text=True, capture_output=True)
    active = {int(j['TrotasrTask']) for j in json.loads(queue.stdout.strip() or '[]') if 'TrotasrTask' in j}
    state = read(PRODUCTION / 'state.json') if (PRODUCTION / 'state.json').exists() else {'submissions': []}
    submitted = {n for item in state['submissions'] for n in item['tasks']}
    for number, task in enumerate(tasks):
        done = PRODUCTION / 'outputs' / task['era'] / (task['name'] + '.json')
        if (task['era'] in eras and number not in active and number not in submitted
                and not done.exists() and not receipt_path(task).exists()):
            pending.append(number)
    pending.sort(key=lambda number: (tasks[number]['source']['nevents'], number))
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print('No pending tasks'); return
    batch_root = EOS_ROOT + '/production'
    description = '\n'.join([
        'universe = vanilla', 'executable = ' + EOS_ROOT + '/workflows/produce.py',
        'arguments = --worker $(task)', 'initialdir = ' + batch_root,
        'output = ' + batch_root + '/logs/$(Cluster).$(Process).out',
        'error = ' + batch_root + '/logs/$(Cluster).$(Process).err',
        'log = ' + batch_root + '/logs/$(Cluster).log', 'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT', 'transfer_output_files = ""',
        'transfer_input_files = ' + batch_root + '/bundles/producer.tgz', 'use_x509userproxy = True',
        'x509userproxy = ' + PROXY, 'request_cpus = 1', 'request_memory = 12000MB',
        'request_disk = 12000MB', '+JobFlavour = "workday"',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        '+TrotasrTask = $(task)', '+TrotasrProduction = "signals"',
        'max_materialize = ' + str(len(pending)), 'max_idle = ' + str(len(pending)),
        'queue task from (', *map(str, pending), ')', ''])
    submit_file = PRODUCTION / 'signals.sub'
    submit_file.write_text(description)
    result = run(['condor_submit', '-terse', submit_file], capture_output=True, text=True)
    state['submissions'].append(dict(time=time.time(), tasks=pending, condor=result.stdout.strip()))
    write_json(PRODUCTION / 'state.json', state)
    print(result.stdout.strip(), 'jobs', len(pending), flush=True)


def worker(number, phase=None, corrected_binding=None):
    task = read(PRODUCTION / 'tasks.json')['tasks'][number]
    if phase == 'histograms':
        return signal_histogram(task, corrected_binding=corrected_binding)
    era, source = task['era'], task['source']
    directory = PRODUCTION / 'work' / era
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (task['name'] + '.root')
    cache = directory / (task['name'] + '.cache')
    cache.mkdir(exist_ok=True)
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'KERAS_HOME',
                 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR'):
        os.environ[name] = str(cache)
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    if phase == 'parent':
        from TROTASR.utils.nanoaod_parent import append_parent_deepcsv
        append_parent_deepcsv([output], output.with_suffix('.edm.root'), source['parents'])
        return
    from TROTASR.utils import nanoaod_production as production
    if phase == 'trota':
        production.complete_trota(output, era)
        return
    metadata = read(output.with_suffix('.json')) if output.with_suffix('.json').exists() else {}
    if metadata.get('status') not in ('events_complete', 'complete'):
        production.build_events(source, era, output)
    if era in EARLY_ERAS and not metadata.get('topmixed_inputs'):
        from TROTASR.utils.nanoaod_parent import deepcsv_cmssw_config
        edm = output.with_suffix('.edm.root')
        config, count = deepcsv_cmssw_config([output], source['parents'], edm)
        config_path = output.with_suffix('.cmssw.py')
        config_path.write_text(config)
        setup = ('source /cvmfs/cms.cern.ch/cmsset_default.sh\nexport SCRAM_ARCH=el9_amd64_gcc12\n'
                 + 'cd ' + CMSSW + '\neval "$(scram runtime -sh)"\n')
        # FWLite and cmsRun use the same installed CMSSW release.
        if count:
            clean = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'ROOTSYS')}
            clean['PATH'] = '/usr/local/bin:/usr/bin:/bin'
            run(['/bin/bash', '--noprofile', '--norc', '-c', setup + 'cd ' + shlex.quote(str(directory))
                 + '\nexec cmsRun ' + shlex.quote(str(config_path))], env=clean)
        in_environment(setup, ['--worker', str(number), '--phase', 'parent'])
        edm.unlink(missing_ok=True)
        config_path.unlink()
    if read(output.with_suffix('.json')).get('status') != 'complete':
        in_environment('source ' + TENSORFLOW,
                       ['--worker', str(number), '--phase', 'trota'])
    metadata = production.validate_output(output, era, source)
    destination = EOS_ROOT + '/production/outputs/' + era
    run(['xrdfs', 'root://eosuser.cern.ch', 'mkdir', '-p', destination])
    for path in (output, output.with_suffix('.json')):
        # JSON is published last and acts as the complete-file receipt.
        run(['xrdcp', '--force', path,
             'root://eosuser.cern.ch/' + destination + '/' + path.name])
    print(json.dumps(dict(status='published', era=era, name=task['name'],
                          events=metadata['events_read'], retained=metadata['events_written'])), flush=True)
    (directory / (Path(source['name']).stem + '.nanoaod.root')).unlink(missing_ok=True)


def collect_file(task, ready):
    """Copy one completed file, then retire its verified EOS staging copy."""
    from TROTASR.utils.nanoaod_production import validate_output
    receipt = receipt_path(task)
    remote = EOS_ROOT + '/production/outputs/' + task['era'] + '/' + task['name']
    target = ROOT / 'inputs' / task['era'] / (task['name'] + '.root')
    target.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        metadata = read(receipt)
        if not target.exists() or target.stat().st_size != metadata['bytes']:
            raise ValueError('Collected ROOT no longer matches its receipt: ' + str(target))
    else:
        if shutil.disk_usage(ROOT).free < 80 * 1024**3:
            raise OSError('hep2 free disk is below the required 80 GiB reserve')
        for suffix in ('.json', '.root'):
            path = target.with_suffix(suffix)
            if not path.exists():
                pending = path.with_name(path.name + '.partial')
                run(['xrdcp', '--force',
                     'root://eosuser.cern.ch/' + remote + suffix, pending])
                os.replace(pending, path)
        metadata = validate_output(target, task['era'], task['source'])
        write_json(receipt, dict(status='collected', path=str(target), time=time.time(),
                                era=task['era'], source=task['source']['name'],
                                events=metadata['events_read'], bytes=target.stat().st_size))
    # A receipt is published before deleting either staging file. If cleanup
    # fails, the next pass reuses the accepted hep2 file and retries cleanup.
    eos_receipt = EOS_ROOT + '/production/collected/' + task['era']
    run(['xrdcp', '--force', receipt, 'root://eosuser.cern.ch/' + eos_receipt + '/' + receipt.name])
    for suffix in ('.root', '.json'):
        if remote + suffix in ready:
            run(['xrdfs', 'root://eosuser.cern.ch', 'rm', remote + suffix])
    print('COLLECTED', task['era'], task['name'], flush=True)


def collect(follow=False, workers=16):
    """Continuously drain completed ROOTs to hep2 with bounded parallel I/O."""
    if not str(ROOT).startswith('/hep2-scratch/'):
        raise ValueError('Collection runs on hep2, where final ROOT files remain')
    tasks = read(PRODUCTION / 'tasks.json')['tasks']
    for key, value in dict(X509_USER_PROXY='/hep2-scratch/twkim/x509up_u147757',
                           X509_CERT_DIR='/cvmfs/grid.cern.ch/etc/grid-security/certificates').items():
        os.environ[key] = value
    lock_path = PRODUCTION / 'collect.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        (PRODUCTION / 'collect.pid').write_text(str(os.getpid()) + '\n')
        for era in ERAS:
            run(['xrdfs', 'root://eosuser.cern.ch', 'mkdir', '-p',
                 EOS_ROOT + '/production/collected/' + era])
        ready, active, retry_after, failures = set(), {}, {}, {}
        retired, refreshed = set(), 0.0
        completed = {i for i, task in enumerate(tasks) if receipt_path(task).exists()}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                now = time.time()
                if not refreshed or follow and now - refreshed >= 20:
                    try:
                        listing = run(['xrdfs', 'root://eosuser.cern.ch', 'ls', '-R',
                                       EOS_ROOT + '/production/outputs'], text=True, capture_output=True)
                        ready = set(listing.stdout.splitlines())
                        refreshed = now
                        failures.pop('listing', None)
                    except (subprocess.SubprocessError, OSError) as exc:
                        failures['listing'] = str(exc)
                        refreshed = now
                        if not follow: raise
                busy = set(active.values())
                for i, task in enumerate(tasks):
                    if len(active) >= workers: break
                    if i in busy or i in retired or now < retry_after.get(i, 0): continue
                    remote = EOS_ROOT + '/production/outputs/' + task['era'] + '/' + task['name']
                    available = remote + '.json' in ready or i in completed and remote + '.root' in ready
                    if available:
                        active[pool.submit(collect_file, task, ready.copy())] = i
                if active:
                    finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        i = active.pop(future)
                        try:
                            future.result()
                            completed.add(i); retired.add(i); failures.pop(str(i), None)
                        except Exception as exc:
                            failures[str(i)] = str(exc)[-600:]
                            retry_after[i] = time.time() + 60
                            print('COLLECTION RETRY', i, str(exc), flush=True)
                state = dict(completed=len(completed), total=len(tasks), time=time.time(),
                             workers=workers, active=len(active), errors=failures)
                write_json(PRODUCTION / 'collection.json', state)
                if len(completed) == len(tasks) and not active and not failures: return
                if not follow and not active: return
                if not active: time.sleep(10)


def recover(numbers, workers=2):
    """Retry failed files locally on lxplus, without submitting Condor jobs."""
    if not str(ROOT).startswith('/eos/'):
        raise ValueError('Signal recovery runs locally on lxplus')
    tasks = read(PRODUCTION / 'tasks.json')['tasks']
    numbers = list(dict.fromkeys(numbers))
    if any(n < 0 or n >= len(tasks) for n in numbers):
        raise ValueError('Unknown production task')
    queue = run(['condor_q', os.environ.get('USER', 'taiwoo'), '-json',
                 '-attributes', 'TrotasrTask'], capture_output=True, text=True)
    active = {int(j['TrotasrTask']) for j in json.loads(queue.stdout.strip() or '[]')
              if 'TrotasrTask' in j}
    if active.intersection(numbers):
        raise ValueError('Requested recovery still has an active Condor job')

    def retry(number):
        task = tasks[number]
        published = PRODUCTION / 'outputs' / task['era'] / (task['name'] + '.json')
        if receipt_path(task).exists() or published.exists():
            return dict(status='complete', attempts=0)
        log_path = PRODUCTION / 'logs' / ('recover.%d.log' % number)
        for attempt in range(1, 4):
            with log_path.open('ab') as log:
                log.write(('\nLOCAL RECOVERY attempt %d\n' % attempt).encode()); log.flush()
                try:
                    in_environment('source ' + LCG, ['--worker', str(number)], log=log)
                    return dict(status='complete', attempts=attempt)
                except subprocess.SubprocessError as exc:
                    error = str(exc)
            if attempt < 3: time.sleep(30)
        return dict(status='failed', attempts=3, error=error)

    with (PRODUCTION / 'recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        state = dict(pid=os.getpid(), started=time.time(), workers=min(workers, 2),
                     tasks={str(n): dict(status='pending') for n in numbers})
        write_json(PRODUCTION / 'recovery.json', state)
        with ThreadPoolExecutor(max_workers=state['workers']) as pool:
            pending = {pool.submit(retry, n): n for n in numbers}
            while pending:
                for future, n in pending.items():
                    if future.running(): state['tasks'][str(n)]['status'] = 'running'
                finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    n = pending.pop(future)
                    try: state['tasks'][str(n)] = future.result()
                    except Exception as exc: state['tasks'][str(n)] = dict(status='failed', error=str(exc))
                state['time'] = time.time()
                write_json(PRODUCTION / 'recovery.json', state)
        if any(item['status'] == 'failed' for item in state['tasks'].values()):
            raise RuntimeError('Local recovery exhausted; see production/recovery.json')


def efficiencies(workers=8, limit=None):
    """Measure each source once for all eras, resuming saved file counts."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from TROTASR.utils.tag_efficiencies import DIRECTORY, DEFINITION, measure_logged, assemble

    DIRECTORY.mkdir(parents=True, exist_ok=True)
    sources = list({task['source']['name']: task['source']
                    for task in read(PRODUCTION / 'tasks.json')['tasks']}.values())
    sources.sort(key=lambda item: (item['nevents'], item['name']))
    selected = sources[:limit] if limit is not None else sources
    state_path = DIRECTORY / 'state.json'
    with (DIRECTORY / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        state = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                     started=time.time(), total=len(sources), selected=len(selected),
                     definition=DEFINITION, eras=list(ERAS), completed={}, errors={}, active=[],
                     status='measuring', workers=workers)
        pending = list(selected)
        attempts = {}
        futures = {}
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            while pending or futures:
                memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                          for line in Path('/proc/meminfo').read_text().splitlines()}
                processes = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid='], text=True)
                room = (shutil.disk_usage(ROOT).free > 80 * 1024**3 +
                        sum(int(s.get('size', 0)) for s in pending[:workers]) and
                        memory['MemAvailable'] > (32 + workers * 4) * 1024**3 and
                        len(processes.splitlines()) + workers * 2 + 2 <= 120)
                while pending and len(futures) < workers and room:
                    source = pending.pop(0)
                    key = Path(source['name']).stem
                    attempts[key] = attempts.get(key, 0) + 1
                    futures[pool.submit(measure_logged, source)] = source
                state.update(time=time.time(), active=[Path(s['name']).stem for s in futures.values()])
                write_json(state_path, state)
                if not futures:
                    if pending:
                        state['status'] = 'waiting_for_resources'
                        write_json(state_path, state)
                        time.sleep(30)
                        continue
                    break
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    source = futures.pop(future)
                    key = Path(source['name']).stem
                    try:
                        result = future.result()
                    except Exception as exc:
                        state['errors'][key] = dict(error=str(exc)[-1500:], attempts=attempts[key])
                        if attempts[key] < 3:
                            pending.append(source)
                    else:
                        state['completed'][key] = result
                        state['errors'].pop(key, None)
                    print('EFFICIENCIES', len(state['completed']), '/', len(selected),
                          'failed', len(state['errors']), flush=True)
        state.update(active=[], time=time.time())
        if state['errors']:
            state['status'] = 'failed'
            write_json(state_path, state)
            raise RuntimeError('Efficiency retries exhausted; see production/efficiencies/state.json')
        if len(selected) == len(sources):
            state['status'] = 'installing'
            write_json(state_path, state)
            state['installed'] = assemble(sources)
            state['datasets'] = sorted({s['dataset'] for s in sources})
            state['status'] = 'complete'
        else:
            state['status'] = 'partial'
        state['time'] = time.time()
        write_json(state_path, state)


def _corrected_signal_directory():
    return ROOT / 'stats/run3/new_signal/signal_histograms'


def _replacement_fingerprint(path):
    if internal_path(path) != path or not path.is_file():
        raise ValueError("Expected a canonical replacement signal file: " + str(path))
    stamp = path.stat()
    return dict(inode=stamp.st_ino, mtime_ns=stamp.st_mtime_ns, size=stamp.st_size)


def _replacement_basis(era, tasks, names, runs, normalization, xsecs, shown):
    import numpy as np

    scope = normalization.get("signal_replacement", {})
    datasets = {str(i): row["dataset"] for i, row in normalization["datasets"].items()
                if row.get("is_signal") and not row.get("is_data")}
    lumi = float(normalization["luminosity_pb"])
    if (datasets != names or scope.get("files") != len(tasks)
            or not np.isfinite(lumi) or lumi <= 0):
        raise ValueError("Replacement signal Runs normalization inventory disagrees for " + era)
    factors = {}
    for point in sorted(shown):
        saved = normalization["signal_mass_points"].get(point, {})
        sumw = runs.get(point, 0.)
        xs = xsecs.get(int(saved.get("mStop", -1)), 0.)
        expected_factor = xs * lumi / sumw if sumw else 0.
        if (not np.isfinite(expected_factor) or expected_factor <= 0
                or not np.isclose(saved.get("sumw_mass_point", np.nan), sumw, rtol=2e-12, atol=0.)
                or not np.isclose(saved.get("normalization_factor", np.nan) or np.nan,
                                  expected_factor, rtol=2e-12, atol=0.)):
            raise ValueError("Replacement signal Runs normalization disagrees for " + era + "/" + point)
        factors[point] = {"normalization_factor": expected_factor}
    return dict(basis=SIGNAL_NORM_BASIS, era=era, datasets=names,
                signal_population=REPLACEMENT_SIGNAL_POPULATION, norm=dict(signal_mass_points=factors))


def _replacement_signal_inputs():
    """Validate the approved UUID collection and exact retained/Run coverage."""
    from collections import defaultdict
    import numpy as np
    from TROTASR.utils.signal_models import signal_mass_from_genmodel, signal_mass_key, signal_topology
    from TROTASR.workflows.validation_regions import layout

    prod = PRODUCTION
    manifest, collection = (read_json(prod / name) for name in ("tasks.json", "collection.json"))
    records = manifest["tasks"]
    n = REPLACEMENT_FILES_PER_ERA * len(ERAS)
    if (tuple(manifest["eras"]) != ERAS or manifest["files"] != REPLACEMENT_FILES_PER_ERA
            or len(records) != n or collection.get("completed") != n or collection.get("total") != n
            or collection.get("active") != 0 or collection.get("errors") != {}):
        raise ValueError("The six-era replacement signal collection is not complete")
    tasks = {era: [] for era in ERAS}
    names, runs, lfns = ({era: {} for era in ERAS}, {era: defaultdict(float) for era in ERAS},
                         {era: set() for era in ERAS})
    for record in records:
        era, source = record["era"], record["source"]
        lfn, source_events = source["name"], source["nevents"]
        stem = record["name"]
        filename = stem + ".root"
        if (era not in tasks or lfn in lfns[era] or stem != "signal_" + Path(lfn).stem
                or not REPLACEMENT_SIGNAL_NAME.fullmatch(filename) or type(source_events) is not int
                or source_events <= 0):
            raise ValueError("Unexpected replacement signal task")
        lfns[era].add(lfn)
        relative = Path("inputs") / era / filename
        metadata_path = relative.with_suffix(".json")
        receipt = read_json(prod / "collected" / era / metadata_path.name)
        fingerprint = _replacement_fingerprint(ROOT / relative)
        metadata = read_json(ROOT / metadata_path)
        files, datasets = metadata["files"], metadata["datasets"]
        retained = metadata["events_written"]
        if (receipt.get("status") != "collected" or receipt.get("era") != era
                or receipt.get("bytes") != fingerprint["size"] or receipt.get("events") != source_events
                or metadata.get("status") != "complete" or metadata.get("analysis_era") != era
                or metadata.get("input_nanoaod_version") != 15 or metadata.get("events_read") != source_events
                or type(retained) is not int or not 0 <= retained <= source_events
                or len(files) != 1 or len(datasets) != 1):
            raise ValueError("Unexpected replacement signal receipt or metadata: " + str(relative))
        dsid, ds = next(iter(datasets.items()))
        dataset = source["dataset"]
        weight_sums = ds.get("signal_sumw_by_genmodel", {})
        if (ds.get("dataset") != dataset or ds.get("is_signal") is not True
                or ds.get("is_data") is not False or not weight_sums
                or names[era].setdefault(str(dsid), dataset) != dataset):
            raise ValueError("Invalid replacement signal dataset metadata: " + str(relative))
        for branch, value in weight_sums.items():
            topology, mass, lsp = signal_mass_from_genmodel(branch)
            if topology != signal_topology(dataset) or mass is None or lsp is None or not np.isfinite(float(value)):
                raise ValueError("Invalid replacement signal Runs sum: " + str(relative))
            runs[era][signal_mass_key(topology, mass, lsp)] += float(value)
        tasks[era].append(dict(year=era, input=str(relative), metadata=str(metadata_path),
                               entries=retained, fingerprint=fingerprint,
                               signal_population=REPLACEMENT_SIGNAL_POPULATION))
    if (any(len(t) != REPLACEMENT_FILES_PER_ERA for t in tasks.values())
            or any(lfns[era] != lfns[ERAS[0]] for era in ERAS)
            or len({t["input"] for group in tasks.values() for t in group}) != n):
        raise ValueError("Incomplete or duplicate replacement signal source coverage")
    xsecs = {int(r["mStop"]): float(r["xsec_pb"])
             for r in read_json(ROOT / "stats/stop_xsec_13p6TeV.json")["records"]}
    bases = {}
    for era in ERAS:
        normalization = read_json(prod / "histograms" / f"normalization_{era}.json.gz")
        bases[era] = _replacement_basis(era, tasks[era], names[era], runs[era], normalization, xsecs, layout(era)[2])
    return tasks, bases


def _replacement_task_source(task):
    era = str(task['year'])
    relative = Path(task['input'])
    if (task.get('signal_population') != REPLACEMENT_SIGNAL_POPULATION
            or era not in ERAS or relative.parent != Path('inputs') / era
            or not REPLACEMENT_SIGNAL_NAME.fullmatch(relative.name)
            or task.get('metadata') != str(relative.with_suffix('.json'))):
        raise ValueError('Replacement signal requires its registered ROOT and metadata identities')
    output = ROOT / relative
    if _replacement_fingerprint(output)['size'] != task['fingerprint']['size']:
        raise ValueError('Unexpected replacement signal ROOT size: ' + str(relative))
    return output


def _corrected_signal_inputs(tasks):
    """Reuse the same verified UUID ROOT/Run registry as corrected nominal VR."""
    from TROTASR.utils.io import read_json
    inventory, _ = _replacement_signal_inputs()
    by_path = {t['input']: t for group in inventory.values() for t in group}
    norm = {}
    for era in ERAS:
        accepted = ROOT / 'estimations' / f'normalization_{era}.json.gz'
        generated = PRODUCTION / 'histograms' / accepted.name
        fields = ('luminosity_pb', 'signal_mass_points')
        installed, measured = read_json(accepted), read_json(generated)
        if any(installed[k] != measured[k] for k in fields):
            raise ValueError('Published and production Runs normalization differ: ' + era)
        norm[era] = dict(path=str(accepted.relative_to(ROOT)))
    bindings = [dict(task=by_path[f"inputs/{task['era']}/{task['name']}.root"],
                     normalization=norm[task['era']]) for task in tasks]
    if len(bindings) != len(by_path) or len({t['task']['input'] for t in bindings}) != len(by_path):
        raise ValueError('Corrected signal histogram inventory differs from the accepted UUID collection')
    return bindings, norm


def _corrected_histogram_result(result, binding, source):
    task = binding['task']
    contract = result['contract']
    if (result.get('status') != 'complete' or result.get('sr_data_blinded') is not True
            or result.get('highdm_sr_source_axes') != 'exact_nb_exact_topology_native_recoil_v2'
            or contract.get('input') != str(source) or str(contract.get('year')) != str(task['year'])
            or contract.get('fingerprint', {}).get('size') != task['fingerprint']['size']
            or contract.get('shape_mode') != 'cms_trota_jme'
            or contract.get('nominal_calibrations_only') is not True
            or contract.get('signal_sr_only') is not True
            or contract.get('nominal_trota_friend') is not None or contract.get('lhe_weights') is not None
            or contract.get('variations') != ['nominal']
            or result['audit']['events_read'] != task['entries']
            or not result['input_metadata']
            or any(row.get('is_signal') is not True or row.get('is_data') is not False
                   for row in result['input_metadata'].values())
            or result['physical_histograms']
            or any(set(endpoint) != {'nominal'} or set(endpoint['nominal']) != {'SR'}
                   for endpoint in result['histograms'].values())):
        raise ValueError('Incomplete or misbound corrected signal histogram: ' + str(source))
    return contract


def _corrected_signal_histogram(task, binding):
    from TROTASR.utils.histogramming import build
    from TROTASR.utils.io import read_json

    era, name = task['era'], task['name']
    relative = Path('inputs') / era / (name + '.root')
    normalization = ROOT / 'estimations' / f'normalization_{era}.json.gz'
    source = _replacement_task_source(binding['task'])
    if source != ROOT / relative:
        raise ValueError('Incorrect registered UUID signal input')
    output = _corrected_signal_directory() / era / (name + '.json.gz')
    if internal_path(output) != output:
        raise ValueError('Corrected signal output does not resolve to its canonical path')
    if output.exists():
        result = read_json(output)
    else:
        # Do not modify accepted ROOT metadata; the original publication
        # finalized it. Do not read stored TROTA, friend ROOTs or NanoAOD.
        result = build(source, era, output, normalization_path=normalization,
            shape_mode='cms_trota_jme', use_trota_friend=False, nominal_calibrations_only=True,
            signal_sr_only=True, support_only=False, variations=('nominal',))
    _corrected_histogram_result(result, binding, source)
    if _replacement_task_source(binding['task']) != source:
        raise ValueError('Corrected signal ROOT changed during event processing')
    return result


def signal_histogram(task, *, corrected_binding=None):
    if corrected_binding is not None:
        result = _corrected_signal_histogram(task, corrected_binding)
        print('CORRECTED_SIGNAL_HISTOGRAM', task['era'], task['name'],
              result['audit']['events_read'], flush=True)
        return
    from TROTASR.utils.histogramming import build
    from TROTASR.utils.nanoaod_production import complete_metadata
    from TROTASR.utils.io import read_json
    era = task['era']
    directory = PRODUCTION / 'histograms'
    source = ROOT / 'inputs' / era / (task['name'] + '.root')
    output = directory / era / (task['name'] + '.json.gz')
    if output.exists():
        result = read_json(output)
        if (result.get('status') not in ('complete', 'complete_one_file_test')
                or result['contract']['input'] != str(source)
                or str(result['contract']['year']) != era):
            raise ValueError('Incomplete or wrong-input signal histogram')
    else:
        complete_metadata(source, era)
        result = build(source, era, output,
            normalization_path=directory / f'normalization_{era}.json.gz')
    print('SIGNAL_HISTOGRAM', era, task['name'], result['audit']['events_read'], flush=True)


def _merge_corrected_signals(era, inputs):
    """Merge only corrected nominal signal payloads, never current physics inputs."""
    from TROTASR.utils.io import file_identity, read_json, scientific_settings
    from TROTASR.workflows.merge_nominal import merge_datasets, merge_histograms
    from TROTASR.workflows.validation_regions import _available_memory

    directory = _corrected_signal_directory()
    destination = directory / f'systematics_{era}.json.gz'
    if internal_path(destination) != destination:
        raise ValueError('Corrected signal merge does not resolve to its canonical path')
    histograms, physical, datasets = {}, {}, {}
    physics = None
    identities = []
    norm = None
    events = 0
    for binding in inputs:
        if _available_memory() < 24*1024**3 or shutil.disk_usage(directory).free < 80*1024**3:
            raise RuntimeError('Corrected signal merge stopped for hep2 memory or disk emergency')
        if str(binding['task']['year']) != era:
            raise ValueError('Wrong target era in corrected signal merge')
        task = binding['task']
        source = ROOT / task['input']
        path = directory / era / (source.stem + '.json.gz')
        identity = file_identity(path)
        result = read_json(path)
        contract = _corrected_histogram_result(result, binding, source)
        if path.stat().st_size != identity[2]:
            raise RuntimeError('Corrected signal histogram size changed while reading ' + str(path))
        identities.append(dict(path=str(path.relative_to(ROOT)), identity=identity))
        common = scientific_settings({k: v for k, v in contract.items() if k not in (
            'input', 'fingerprint', 'registered_signal_source')})
        if physics is None:
            physics, norm = common, binding['normalization']
        elif physics != common:
            raise ValueError('Corrected nominal signal contracts differ for ' + era)
        merge_histograms(histograms, result['histograms'])
        merge_histograms(physical, result['physical_histograms'])
        merge_datasets(datasets, result['input_metadata'])
        events += result['audit']['events_read']
    if (len(inputs) != REPLACEMENT_FILES_PER_ERA or set(histograms) != {'highdm', 'lowdm'}
            or not datasets or len({row['path'] for row in identities}) != len(inputs)):
        raise ValueError('Incomplete corrected signal merge for ' + era)
    if any(record['identity'][2] != (ROOT / record['path']).stat().st_size for record in identities):
        raise ValueError('Corrected signal inputs changed before merge publication: ' + era)
    coverage = dict(full_input_inventory=True, expected_files=len(inputs), completed_files=len(inputs),
                    failed_files=0, expected_events=sum(task['task']['entries'] for task in inputs),
                    processed_events=events)
    replacement = dict(files=len(inputs), datasets=len(datasets), retained_events=events,
                       signal_uncertainties=False, background_and_data_unchanged=True,
                       central='cms_trota_jme_nominal', signal_input='regenerated_signal_uuid')
    scope = 'full_cms_trota_jme_production'
    payload = dict(status='complete', scope=scope, contract=physics, coverage=coverage,
                   sr_data_blinded=True, highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                   histograms=histograms, physical_histograms=physical, input_metadata=datasets,
                   signal_replacement=replacement)
    if destination.exists():
        if scientific_settings(read_json(destination)) != scientific_settings(payload):
            raise ValueError('Preserve divergent completed corrected signal merge: ' + era)
    else:
        write_json(destination, payload)
    return dict(status='complete', output=str(destination.relative_to(ROOT)),
                output_identity=list(file_identity(destination)), coverage=coverage, physics_contract=physics,
                scope=scope, normalization=norm, signal_replacement=replacement,
                files=len(inputs), datasets=len(datasets), retained_events=events)


def refresh_signals(workers=8, limit=None, *, corrected_nominal=False):
    """Follow measurement with nominal histograms and signal-only replacement."""
    from TROTASR.utils.normalization import prepare_signal_replacement
    from TROTASR.workflows.merge_nominal import replace_signals
    production_directory = PRODUCTION / 'histograms'
    directory = _corrected_signal_directory() if corrected_nominal else production_directory
    if corrected_nominal:
        if ROOT != Path('/hep2-scratch/twkim/postprocessing/TROTASR') or limit is not None:
            raise ValueError('Corrected nominal signal production requires the full accepted hep2 collection')
        if internal_path(directory) != directory:
            raise ValueError('Corrected signals require a canonical output directory')
        from TROTASR.utils.io import file_identity
        from TROTASR.workflows.validation_regions import _resource_room
    production_directory.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'logs').mkdir(exist_ok=True)
    with (production_directory / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        state_path = directory / 'state.json'
        state = read(state_path) if state_path.exists() else dict(completed={}, errors={}, merged={})
        tasks_path = PRODUCTION / 'tasks.json'
        tasks = read(tasks_path)['tasks']
        bindings = norm = None
        if corrected_nominal:
            bindings, norm = _corrected_signal_inputs(tasks)
            registration = dict(signal_input='regenerated_signal_uuid', central='cms_trota_jme_nominal',
                                normalization_by_era=norm)
            if state_path.exists() and any(state.get(k) != registration[k]
                    for k in ('signal_input', 'central')):
                raise ValueError('Preserve corrected signal results with a different population or central view')
            state.update(registration)
        state['previous_failed_files'] = len(state.get('errors', {}))
        state['errors'] = {}
        state.update(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                     status='waiting_for_efficiencies', workers=workers, time=time.time(), started=time.time())
        write_json(state_path, state)
        efficiency_lock = PRODUCTION / 'efficiencies/lock'
        with efficiency_lock.open('a+') as measuring:
            fcntl.flock(measuring, fcntl.LOCK_EX)
            complete = read(PRODUCTION / 'efficiencies/state.json')
            if complete['status'] != 'complete':
                raise RuntimeError('Full-source efficiency measurement has not completed')
        if not corrected_nominal:
            for era in ERAS:
                sidecars = [ROOT / 'inputs' / era / (task['name'] + '.json')
                            for task in tasks if task['era'] == era]
                prepare_signal_replacement(era, sidecars, directory / f'normalization_{era}.json.gz')
        selected = list(enumerate(tasks))
        if limit is not None: selected = selected[:limit]
        if corrected_nominal:
            for key, receipt in state['completed'].items():
                task = tasks[int(key)]
                path = directory / task['era'] / (task['name'] + '.json.gz')
                if (receipt.get('era') != task['era']
                        or not path.is_file() or receipt.get('identity', [None, None, path.stat().st_size])[2]
                        != path.stat().st_size):
                    raise ValueError('Preserve a divergent corrected signal completion receipt: ' + str(path))
            for era, receipt in state['merged'].items():
                path = directory / f'systematics_{era}.json.gz'
                if (not path.is_file()
                        or receipt.get('output_identity', [None, None, path.stat().st_size])[2]
                        != path.stat().st_size):
                    raise ValueError('Preserve a divergent corrected signal merge receipt: ' + str(path))
        pending = [(n,t) for n,t in selected if str(n) not in state['completed']]
        attempts, futures = {}, {}
        state.update(status='histograms', total=len(tasks), selected=len(selected))
        inherited = {}
        for row in subprocess.check_output(
                ['ps','-u',str(os.getuid()),'-o','pid=,args='], text=True).splitlines():
            pid, command = row.strip().split(None, 1)
            arguments = command.split()
            if not all(flag in arguments for flag in ('TROTASR', 'produce', '--worker', '--phase', 'histograms')):
                continue
            if ('--corrected-nominal' in arguments) != corrected_nominal:
                if corrected_nominal:
                    # A legacy worker left by an interrupted controller may
                    # still finish its accepted output/metadata.
                    try:
                        process = Path('/proc') / pid
                        if process.joinpath('cwd').resolve() == ROOT.parent:
                            raise RuntimeError('An inherited legacy signal histogram worker is still active')
                    except FileNotFoundError:
                        continue
                continue
            try:
                process = Path('/proc') / pid
                if process.joinpath('cwd').resolve() != ROOT.parent:
                    continue
                number = int(arguments[arguments.index('--worker')+1])
                inherited[number] = (int(pid), process.joinpath('stat').read_text().split()[21])
            except (FileNotFoundError, ProcessLookupError):
                continue
        state['inherited_workers'] = {str(n): pid for n,(pid,ticks) in inherited.items()}
        stopped = threading.Event()
        child_lock = threading.Lock()
        children_started = {}

        def stop_own_workers():
            with child_lock:
                stopped.set()
                for child in children_started.values():
                    if child.poll() is None:
                        child.kill()

        def execute(number, task):
            if number in inherited:
                pid, ticks = inherited[number]
                while not stopped.is_set():
                    try:
                        status = Path('/proc',str(pid),'stat').read_text().split()
                    except FileNotFoundError:
                        break
                    if status[21] != ticks or status[2] == 'Z':
                        break
                    time.sleep(3)
            log = directory / 'logs' / (task['era'] + '_' + task['name'] + '.log')
            output = directory / task['era'] / (task['name'] + '.json.gz')
            command = [sys.executable, '-m', 'TROTASR', 'produce', '--worker', str(number),
                       '--phase', 'histograms']
            with log.open('a') as stream:
                if corrected_nominal:
                    command += ['--corrected-nominal', '--input-binding', json.dumps(bindings[number])]
                    with child_lock:
                        if stopped.is_set():
                            raise RuntimeError('Corrected signal resource stop; preserve completed results')
                        child = subprocess.Popen(command, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
                        children_started[number] = child
                    try:
                        if child.wait():
                            raise subprocess.CalledProcessError(child.returncode, command)
                    finally:
                        with child_lock:
                            children_started.pop(number, None)
                else:
                    run(command, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
            result = dict(era=task['era'], output=str(output.relative_to(ROOT)))
            if corrected_nominal:
                result['identity'] = list(file_identity(output))
            return result

        resource_stop = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or futures:
                memory = {line.split(':')[0]:int(line.split()[1])*1024
                          for line in Path('/proc/meminfo').read_text().splitlines()}
                process_rows = subprocess.check_output(
                    ['ps','-u',str(os.getuid()),'-o','pid=,ppid=,rss=,comm='], text=True).splitlines()
                children = [list(map(int,row.split()[:3])) for row in process_rows
                            if (int(row.split()[1]) == os.getpid()
                                or int(row.split()[0]) in state['inherited_workers'].values())
                            and row.split()[3].startswith('python')]
                worker_memory = 3*1024**3
                if corrected_nominal:
                    worker_memory = 6*1024**3
                    slots, memory_budget = _resource_room(reserve_vr_data=True, reserve_bytes=32*1024**3)
                    if memory['MemAvailable'] < 24*1024**3 or shutil.disk_usage(ROOT).free < 80*1024**3:
                        resource_stop = ('memory_stopped' if memory['MemAvailable'] < 24*1024**3
                                         else 'disk_stopped')
                        stop_own_workers()
                        break
                resident_credit = sum(min(row[2]*1024, worker_memory) for row in children)
                disk_room = shutil.disk_usage(ROOT).free > 80*1024**3
                while pending and len(futures) < workers and disk_room:
                    next_count = len(futures) + 1
                    reservation = max(0, next_count*worker_memory - resident_credit)
                    process_count = len(process_rows) + max(0, next_count-len(children)) + 2
                    if corrected_nominal:
                        if reservation > memory_budget or max(0, next_count-len(children)) > slots:
                            break
                    elif memory['MemAvailable'] <= 32*1024**3 + reservation or process_count > 120:
                        break
                    number, task = pending.pop(0)
                    attempts[number] = attempts.get(number,0)+1
                    futures[pool.submit(execute, number, task)] = (number,task)
                state.update(time=time.time(), active=[n for n,t in futures.values()])
                write_json(state_path, state)
                if not futures:
                    if pending: time.sleep(3 if corrected_nominal else 30); continue
                    break
                done, _ = wait(futures, timeout=3 if corrected_nominal else 30, return_when=FIRST_COMPLETED)
                for future in done:
                    number, task = futures.pop(future)
                    try:
                        state['completed'][str(number)] = future.result()
                        state['errors'].pop(str(number),None)
                    except Exception as exc:
                        state['errors'][str(number)] = dict(error=str(exc)[-1000:], attempts=attempts[number])
                        if attempts[number] < 3: pending.append((number,task))
                    print('SIGNALS',len(state['completed']),'/',len(selected),'failed',len(state['errors']),flush=True)
        if resource_stop:
            for future, (number, _) in futures.items():
                try:
                    result = future.result()
                except Exception:
                    continue
                state['completed'][str(number)] = result
            state.update(status=resource_stop, active=[], time=time.time())
            write_json(state_path, state)
            raise RuntimeError('Corrected signal production paused by ' + resource_stop)
        state.update(active=[],time=time.time())
        if state['errors']:
            state['status']='failed'; write_json(state_path,state)
            raise RuntimeError('Signal histogram retries exhausted; see production/histograms/logs')
        if limit is None:
            state['status']='merging'; write_json(state_path,state)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            remaining = [era for era in ERAS if era not in state['merged']]
            while remaining:
                memory = {line.split(':')[0]:int(line.split()[1])*1024
                          for line in Path('/proc/meminfo').read_text().splitlines()}
                processes = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid='], text=True)
                if corrected_nominal:
                    processes_free, memory_budget = _resource_room(reserve_vr_data=True, reserve_bytes=32*1024**3)
                    slots = min(len(remaining), workers, processes_free-1, memory_budget//(16*1024**3))
                else:
                    slots = min(len(remaining), workers, 118-len(processes.splitlines()),
                                int((memory['MemAvailable']-32*1024**3)//(16*1024**3)))
                if slots > 0 and shutil.disk_usage(ROOT).free > 80*1024**3:
                    break
                if corrected_nominal and (memory['MemAvailable'] < 24*1024**3
                                          or shutil.disk_usage(ROOT).free < 80*1024**3):
                    state.update(status='memory_stopped' if memory['MemAvailable'] < 24*1024**3
                                 else 'disk_stopped', active=[], time=time.time())
                    write_json(state_path, state)
                    raise RuntimeError('Corrected signal merge resource stop; retained results are reusable')
                time.sleep(30)
            if remaining:
                state['merge_workers'] = slots
                state['merge_errors'] = {}
                write_json(state_path,state)
                with ProcessPoolExecutor(max_workers=slots, mp_context=multiprocessing.get_context('spawn')) as pool:
                    merges = {}
                    for era in remaining:
                        if corrected_nominal:
                            group = [bindings[n] for n, task in enumerate(tasks) if task['era'] == era]
                            from TROTASR.workflows.produce import _merge_corrected_signals as merge
                            future = pool.submit(merge, era, group)
                        else:
                            paths=[directory/era/(t['name']+'.json.gz') for t in tasks if t['era']==era]
                            future = pool.submit(replace_signals,era,paths,directory/f'normalization_{era}.json.gz')
                        merges[future] = era
                    for future in as_completed(merges):
                        era = merges[future]
                        try:
                            state['merged'][era] = future.result()
                        except Exception as exc:
                            state['merge_errors'][era] = str(exc)[-1000:]
                        state['time']=time.time(); write_json(state_path,state)
                        print('SIGNALS_MERGED',era,len(state['merged']),'/',len(ERAS),
                              'failed',len(state['merge_errors']),flush=True)
                if state['merge_errors']:
                    state['status']='failed'; write_json(state_path,state)
                    raise RuntimeError('Signal merge failed; completed eras are preserved')
            state['status']='complete'
        else:
            state['status']='partial'
        state['time']=time.time(); write_json(state_path,state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.argv[1:2] == ['signal-histograms']:
        corrected = argparse.ArgumentParser(prog='python -m TROTASR produce signal-histograms',
                                            allow_abbrev=False)
        corrected.add_argument('--corrected-nominal', action='store_true', required=True)
        corrected.add_argument('--workers', type=int, default=8, choices=range(1, 71))
        opts = corrected.parse_args(sys.argv[2:])
        return refresh_signals(opts.workers, corrected_nominal=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--submit', action='store_true')
    actions.add_argument('--worker', type=int)
    actions.add_argument('--collect', action='store_true')
    actions.add_argument('--recover', type=int, nargs='+', metavar='TASK')
    actions.add_argument('--efficiencies', action='store_true', help='Measure full-source signal tag efficiencies on hep2')
    actions.add_argument('--refresh-signals', action='store_true', help='After efficiencies, rebuild nominal signals and update merged inputs')
    parser.add_argument('--era', choices=ERAS, action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--phase', choices=('parent', 'trota', 'histograms'))
    parser.add_argument('--corrected-nominal', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input-binding', type=json.loads, help=argparse.SUPPRESS)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--workers', type=int, default=16, choices=range(1, 71),
                        metavar='1..70', help='Maximum concurrent workers; resource limits may reduce dispatch (default: 16; use 8 for efficiencies)')
    args = parser.parse_args()
    if (args.corrected_nominal != (args.input_binding is not None) or
            args.corrected_nominal and (args.worker is None or args.phase != 'histograms')):
        parser.error('Corrected nominal workers require a controller-owned UUID binding')
    if args.prepare: prepare()
    elif args.submit: submit(args.era or ERAS, args.limit)
    elif args.worker is not None: worker(args.worker, args.phase, args.input_binding)
    elif args.recover is not None: recover(args.recover, args.workers)
    elif args.efficiencies: efficiencies(args.workers, args.limit)
    elif args.refresh_signals: refresh_signals(args.workers, args.limit)
    else: collect(args.follow, args.workers)


if __name__ == '__main__':
    main()
