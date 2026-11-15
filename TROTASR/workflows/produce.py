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
import shlex
import shutil
import subprocess
import sys
import tarfile
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

from TROTASR.utils.paths import ROOT, ERAS, EARLY_ERAS
from TROTASR.utils.io import write_json

PRODUCTION = ROOT / 'production'
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


def worker(number, phase=None):
    task = read(PRODUCTION / 'tasks.json')['tasks'][number]
    if phase == 'histograms':
        return signal_histogram(task)
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
        run(['xrdcp', '--force', '--cksum', 'adler32', path,
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
                run(['xrdcp', '--force', '--cksum', 'adler32',
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


def signal_histogram(task):
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


def register_efficiencies():
    """Update existing asset records for the twelve newly measured payloads."""
    from TROTASR.utils.io import read_json, sha256
    changes = {f'estimations/{kind}eff{era}.merged': sha256(ROOT / 'estimations' / f'{kind}eff{era}.merged')
               for era in ERAS for kind in ('btag', 'topwtag')}
    def update(value):
        if isinstance(value, list):
            for item in value: update(item)
        elif isinstance(value, dict):
            target = value.get('target')
            if isinstance(target, str) and target in changes and 'sha256' in value:
                value['sha256'] = changes[target]
            for item in value.values():
                if isinstance(item, (dict, list)): update(item)
    for name in ('assets.json', 'early_run3_assets.json'):
        path = ROOT / 'jsons' / name
        payload = read_json(path)
        update(payload)
        write_json(path, payload)


def refresh_signals(workers=8, limit=None):
    """Follow measurement with nominal histograms and signal-only replacement."""
    from TROTASR.utils.normalization import prepare_signal_replacement
    from TROTASR.workflows.merge_nominal import replace_signals
    directory = PRODUCTION / 'histograms'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'logs').mkdir(exist_ok=True)
    with (directory / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        state_path = directory / 'state.json'
        state = read(state_path) if state_path.exists() else dict(completed={}, errors={}, merged={})
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
        register_efficiencies()
        tasks = read(PRODUCTION / 'tasks.json')['tasks']
        for era in ERAS:
            sidecars = [ROOT / 'inputs' / era / (task['name'] + '.json')
                        for task in tasks if task['era'] == era]
            prepare_signal_replacement(era, sidecars, directory / f'normalization_{era}.json.gz')
        selected = list(enumerate(tasks))
        if limit is not None: selected = selected[:limit]
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
            try:
                process = Path('/proc') / pid
                if process.joinpath('cwd').resolve() != ROOT.parent:
                    continue
                number = int(arguments[arguments.index('--worker')+1])
                inherited[number] = (int(pid), process.joinpath('stat').read_text().split()[21])
            except (FileNotFoundError, ProcessLookupError):
                continue
        state['inherited_workers'] = {str(n): pid for n,(pid,ticks) in inherited.items()}
        def execute(number, task):
            if number in inherited:
                pid, ticks = inherited[number]
                while True:
                    try:
                        status = Path('/proc',str(pid),'stat').read_text().split()
                    except FileNotFoundError:
                        break
                    if status[21] != ticks or status[2] == 'Z':
                        break
                    time.sleep(3)
            log = directory / 'logs' / (task['era'] + '_' + task['name'] + '.log')
            with log.open('a') as stream:
                run([sys.executable, '-m', 'TROTASR', 'produce', '--worker', str(number),
                     '--phase', 'histograms'], cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
            return dict(era=task['era'], output=str((directory/task['era']/(task['name']+'.json.gz')).relative_to(ROOT)))
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
                resident_credit = sum(min(row[2]*1024, worker_memory) for row in children)
                disk_room = shutil.disk_usage(ROOT).free > 80*1024**3
                while pending and len(futures) < workers and disk_room:
                    next_count = len(futures) + 1
                    reservation = max(0, next_count*worker_memory - resident_credit)
                    process_count = len(process_rows) + max(0, next_count-len(children)) + 2
                    if memory['MemAvailable'] <= 32*1024**3 + reservation or process_count > 120:
                        break
                    number, task = pending.pop(0)
                    attempts[number] = attempts.get(number,0)+1
                    futures[pool.submit(execute, number, task)] = (number,task)
                state.update(time=time.time(), active=[n for n,t in futures.values()])
                write_json(state_path, state)
                if not futures:
                    if pending: time.sleep(30); continue
                    break
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    number, task = futures.pop(future)
                    try:
                        state['completed'][str(number)] = future.result()
                        state['errors'].pop(str(number),None)
                    except Exception as exc:
                        state['errors'][str(number)] = dict(error=str(exc)[-1000:], attempts=attempts[number])
                        if attempts[number] < 3: pending.append((number,task))
                    print('SIGNALS',len(state['completed']),'/',len(selected),'failed',len(state['errors']),flush=True)
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
                slots = min(len(remaining), workers, 118-len(processes.splitlines()),
                            int((memory['MemAvailable']-32*1024**3)//(16*1024**3)))
                if slots > 0 and shutil.disk_usage(ROOT).free > 80*1024**3:
                    break
                time.sleep(30)
            if remaining:
                state['merge_workers'] = slots
                state['merge_errors'] = {}
                write_json(state_path,state)
                with ProcessPoolExecutor(max_workers=slots, mp_context=multiprocessing.get_context('spawn')) as pool:
                    merges = {}
                    for era in remaining:
                        paths=[directory/era/(t['name']+'.json.gz') for t in tasks if t['era']==era]
                        merges[pool.submit(replace_signals,era,paths,directory/f'normalization_{era}.json.gz')] = era
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
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--workers', type=int, default=16, choices=range(1, 71),
                        metavar='1..70', help='Maximum concurrent workers; resource limits may reduce dispatch (default: 16; use 8 for efficiencies)')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.submit: submit(args.era or ERAS, args.limit)
    elif args.worker is not None: worker(args.worker, args.phase)
    elif args.recover is not None: recover(args.recover, args.workers)
    elif args.efficiencies: efficiencies(args.workers, args.limit)
    elif args.refresh_signals: refresh_signals(args.workers, args.limit)
    else: collect(args.follow, args.workers)


if __name__ == '__main__':
    main()
