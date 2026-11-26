"""Histogram accepted events and original NanoAOD weight variations."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.histogramming import build


_LLCR_MEMORY_PREFIX = 'TROTASR_LLCR_WORKER_MEMORY '
_LLCR_MEMORY_FIELDS = {'VmHWM': 'peak_rss_bytes', 'VmPeak': 'peak_vms_bytes',
                       'RssAnon': 'rss_anon_bytes_at_exit'}
_SCOPED_CHUNK = 1_000_000


def _measurable(options):
    return type(options.get('chunk_size')) is int and options['chunk_size'] > 0


def _worker_memory_usage():
    """Read kernel high-water marks in the one-file worker itself."""
    import resource
    import sys

    sample = {}
    try:
        lines = Path('/proc/self/status').read_text().splitlines()
    except OSError:
        lines = ()
    for line in lines:
        key, _, value = line.partition(':')
        fields = value.split()
        if key in _LLCR_MEMORY_FIELDS and len(fields) == 2 and fields[1] == 'kB' and fields[0].isdigit():
            sample[_LLCR_MEMORY_FIELDS[key]] = int(fields[0])*1024
    if not sample.get('peak_rss_bytes') and (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        # Fall back only when procfs is unavailable. Linux reports KiB,
        # whereas macOS reports bytes.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1024 if sys.platform.startswith('linux') else 1
        sample['peak_rss_bytes'] = rss
    return sample


def _emit_worker_memory(output_built):
    import json

    try:
        sample = _worker_memory_usage()
        if sample.get('peak_rss_bytes', 0) > 0:
            print(_LLCR_MEMORY_PREFIX + json.dumps(dict(sample, output_built=output_built)), flush=True)
    except (OSError, MemoryError):
        # Telemetry must not change the worker's scientific result or exit.
        pass


def _worker_memory_record(stdout):
    import json

    lines = [line[len(_LLCR_MEMORY_PREFIX):] for line in stdout.splitlines()
             if line.startswith(_LLCR_MEMORY_PREFIX)]
    if len(lines) != 1:
        return None
    try:
        sample = json.loads(lines[0])
    except ValueError:
        return None
    if (not isinstance(sample, dict) or type(sample.get('output_built')) is not bool
            or type(sample.get('peak_rss_bytes')) is not int or sample['peak_rss_bytes'] <= 0):
        return None
    result = {k: sample[k] for k in ('peak_rss_bytes', 'output_built')}
    for key in ('peak_vms_bytes', 'rss_anon_bytes_at_exit'):
        number = sample.get(key)
        if type(number) is int and number >= (1 if key == 'peak_vms_bytes' else 0):
            result[key] = number
    return result


def run_task(task, options, *, force_recompute=False, measure_memory=False):
    """Process one inventory entry, normally reusing its completed output."""
    from TROTASR.utils.paths import ROOT, internal_path
    source = internal_path(ROOT / task['input'])
    output = internal_path(ROOT / task['output'])
    if 'force_recompute' in options or 'expected_entries' in options:
        raise ValueError('Histogram inventory options cannot set replacement controls')
    if 'measure_memory' in options:
        raise ValueError('Histogram inventory options cannot set memory controls')
    if options.get('dataset_primaries'):
        options = dict(options)
        if options.setdefault('chunk_size', _SCOPED_CHUNK) != _SCOPED_CHUNK:
            raise ValueError('Dataset-scoped inventories use fixed 1,000,000-event chunks')
    if measure_memory and not _measurable(options):
        raise ValueError('Memory telemetry requires a fixed-chunk scoped inventory task')
    if force_recompute:
        options = dict(options, force_recompute=True, expected_entries=task['entries'])
    # A reused receipt does not measure histogram production. Conservatively
    # exclude every already existing output unless explicitly rebuilding it.
    fresh = measure_memory and (force_recompute or not output.exists())
    output_built = False
    try:
        payload = build(source, task['year'], output, **options)
        if payload['audit']['events_read'] != task['entries']:
            raise ValueError('Histogram event count differs from the inventory')
        if options.get('top_pt_prescription'):
            import json
            from TROTASR.workflows.top_pt_inventory import RECEIPT_PREFIX, product_receipt
            print(RECEIPT_PREFIX + json.dumps(product_receipt(task, options, payload)), flush=True)
        output_built = fresh and output.is_file()
    finally:
        if measure_memory:
            _emit_worker_memory(output_built)
    return dict(input=task['input'], status='complete', output=task['output'])


def run_file_process(inventory, index, task, options, *, retry_smaller_chunk=True,
                     address_space_bytes=None, force_recompute=False, measure_memory=False):
    """Run a file, optionally retaining its exact execution chunk on failure."""
    import subprocess
    import sys
    import time
    from TROTASR.utils.paths import ROOT
    if force_recompute and (options.get('highdm_llcr_only') is not True or retry_smaller_chunk):
        raise ValueError('Only fixed-chunk high-dM LLCR tasks can replace an earlier result')
    if measure_memory and (not _measurable(options) or (retry_smaller_chunk and
                           (options.get('highdm_llcr_only') or options.get('dataset_primaries')))):
        raise ValueError('Only fixed-chunk attempts can measure worker memory in a fixed-chunk inventory')
    chunk = options.get('chunk_size', 2000)
    attempts = []
    sizes = (chunk, max(1, chunk//2)) if retry_smaller_chunk else (chunk,)
    for size in dict.fromkeys(sizes):
        command = [sys.executable, '-m', 'TROTASR', 'histograms', '--inventory', str(inventory),
                   '--task-index', str(index), '--chunk-size', str(size)]
        if force_recompute:
            command.append('--force-recompute')
        if measure_memory:
            command.append('--measure-memory')
        if address_space_bytes is not None:
            command[:0] = ('/usr/bin/prlimit', f'--as={address_space_bytes}:{address_space_bytes}', '--')
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
        attempt = dict(chunk_size=size, exit_code=result.returncode, error=result.stderr[-2000:])
        if measure_memory and not options.get('highdm_llcr_only'):
            attempt['seconds'] = time.monotonic() - started
        if measure_memory:
            sample = _worker_memory_record(result.stdout)
            if sample is not None:
                attempt['memory'] = sample
        attempts.append(attempt)
        if result.returncode == 0:
            row = dict(input=task['input'], status='complete', output=task['output'], attempts=attempts)
            if options.get('top_pt_prescription'):
                import json
                from TROTASR.workflows.top_pt_inventory import RECEIPT_PREFIX
                receipts = [x[len(RECEIPT_PREFIX):] for x in result.stdout.splitlines() if x.startswith(RECEIPT_PREFIX)]
                if len(receipts) != 1:
                    raise ValueError('Top-pT worker did not return a unique product receipt')
                row['top_pt_receipt'] = json.loads(receipts[0])
            return row
    return dict(input=task['input'], status='failed', attempts=attempts)


def run_inventory(path, workers, chunk_size=None):
    """Run the current task list, with bounded admission and per-file resume."""
    import fcntl
    import os
    import shutil
    import time
    from collections import deque
    from contextlib import ExitStack
    from itertools import islice
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from TROTASR.utils.paths import ROOT, internal_path
    from TROTASR.utils.io import read_json, write_json
    from TROTASR.workflows.resources import resources
    from TROTASR.workflows.histogram_resources import DirectTopPtAdmission, HistogramAdmission, histogram_residents
    from TROTASR.workflows.llcr_memory import AdaptiveAdmission, measured_file_bucket, GIB
    from TROTASR.utils.selection_contract import require_current_llcr
    if not 1 <= workers <= 116:
        raise ValueError('Histogram workers must be between 1 and 116')
    path = internal_path(path)
    job = read_json(path)
    require_current_llcr(path, record=job)
    tasks = job['tasks']
    if not tasks or len({t['input'] for t in tasks}) != len(tasks) or len({t['output'] for t in tasks}) != len(tasks):
        raise ValueError('Missing or duplicate histogram tasks')
    options = dict(job.get('histogram_options', {}))
    if chunk_size is not None:
        options['chunk_size'] = chunk_size
    scoped = options.get('dataset_primaries') is not None
    selected_jme = (path == ROOT/'stats/run3/dataset_replay/top_pt/existing_jme_weights_nominal_pass.json')
    if (options.get('nominal_pass_only') is True) != selected_jme:
        raise ValueError('Nominal-pass TT replay needs its own typed inventory')
    if selected_jme and any(internal_path(ROOT/t['output']).parent.parent != path.with_suffix('') for t in tasks):
        raise ValueError('Nominal-pass TT outputs cannot overlap full-migration results')
    tt_jme = (scoped and (selected_jme or path == ROOT/'stats/run3/dataset_replay/top_pt/existing_jme_weights.json')
              and job.get('schema') == 'trotasr_tt_existing_systematics_v1'
              and job.get('population') == 'TT_slot_only' and options.get('shape_mode') == 'cms_trota_jme'
              and options.get('variations') == ['all_weights'] and options.get('use_trota_friend') is True)
    if selected_jme and not tt_jme:
        raise ValueError('Invalid TT-slot nominal-pass JME inventory')
    if tt_jme:
        workers = min(workers, 24)
    top_pt = bool(scoped and options.get('top_pt_prescription')
                  and path.parent == ROOT/'stats/run3/dataset_replay/top_pt'
                  and path.name in ('nominal.json', 'uncertainty.json'))
    if top_pt and path.name == 'uncertainty.json':
        workers = min(workers, 24)
    # The approved stored-endpoint TT replay fills its requested slots without
    # waiting for samples in each era/file class. JME has a different footprint.
    direct_nonjme = (scoped and path == ROOT/'stats/run3/dataset_replay/top_pt/existing_nonjme.json'
                     and job.get('schema') == 'trotasr_tt_existing_systematics_v1'
                     and job.get('population') == 'TT_slot_only'
                     and options.get('shape_mode') == 'stored_trota_nonjme')
    if direct_nonjme:
        workers = min(workers, 24)
    # Pool the approved disjoint nominal rare-Top input across eras. The actual
    # concurrency is limited by live account slots and measured host headroom.
    rare_pool = (path == ROOT/'stats/run3/dataset_replay/top_pt/remainder.json'
                 and job.get('schema') == 'trotasr_top_pt_nominal_remainder_v1'
                 and options.get('variations') == ['nominal'] and options.get('use_trota_friend') is True
                 and set(options.get('dataset_primaries') or ()) ==
                     {'TTTT','TTW-WtoQQ-1Jets','TTWW','TTWZ','TTZ-ZtoQQ-1Jets','TTZZ'})
    direct_slots = direct_nonjme or rare_pool or (top_pt and (workers == 48 or path.name == 'uncertainty.json'))
    if scoped:
        replay_root = ROOT/'stats/run3/dataset_replay'
        if replay_root not in path.parents or any(
                replay_root not in internal_path(ROOT / task['output']).parents for task in tasks):
            raise ValueError('Dataset-scoped inventories and outputs belong in stats/run3/dataset_replay')
        if not isinstance(options['dataset_primaries'], (list, tuple)) or not options['dataset_primaries']:
            raise ValueError('Supply nonempty background dataset primaries')
        if options.setdefault('chunk_size', _SCOPED_CHUNK) != _SCOPED_CHUNK:
            raise ValueError('Dataset-scoped inventories use fixed 1,000,000-event chunks')
        scoped_limit = 116 if rare_pool else 48
        if workers > scoped_limit:
            raise ValueError(f'Dataset-scoped inventories allow at most {scoped_limit} workers')
        if not tt_jme and not os.access('/usr/bin/prlimit', os.X_OK):
            raise ValueError('This capped dataset-scoped inventory requires Linux prlimit')
        if tt_jme:
            # Physical RSS estimate, never a process limit. Do not cap by VmPeak
            # or require address-space calibration. Keep host and account reserves.
            from TROTASR.workflows.histogram_resources import JMEPhysicalAdmission
            adaptive = JMEPhysicalAdmission(workers, default_bytes=48*GIB)
        elif direct_nonjme:
            # Start with 8.5 GiB per slot (first large-file peaks up to 8 GiB).
            # Observed peaks still raise bookings; the host reserve, 32 GiB
            # worker limit, account and disk checks apply on every dispatch.
            adaptive = HistogramAdmission(workers, default_bytes=8*GIB + GIB//2)
            adaptive.worker_limit_bytes = 32*GIB
        else:
            adaptive = DirectTopPtAdmission(workers) if direct_slots else AdaptiveAdmission(workers, 32*GIB, min_samples=6)
    else:
        options.setdefault('chunk_size', 2000)
        if not _measurable(options):
            raise ValueError('Histogram chunk size must be a positive integer')
        adaptive = HistogramAdmission(workers)
    sizes = [internal_path(ROOT / task['input']).stat().st_size for task in tasks]
    if scoped:
        mode = 'jme' if options.get('shape_mode') == 'cms_trota_jme' else 'nonjme'
        buckets = [measured_file_bucket(mode, str(task['year']), 'background',
                   task['entries'], _SCOPED_CHUNK, size) for task, size in zip(tasks, sizes)]
    else:
        # Shape/view settings, population scopes and chunk size are fixed within
        # this inventory. Never transfer a class envelope between inventories.
        def file_class(task, size):
            name = Path(task['input']).name
            kind = task.get('vr_kind') or next((k for k in ('mc','data','signal')
                    if name.startswith(k+'_')), None)
            return ((str(task['year']), kind, size.bit_length(), task['entries'].bit_length())
                    if kind else None)
        buckets = [file_class(t, size) for t, size in zip(tasks, sizes)]
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    state_path = path.with_name(path.stem + '.state.json')
    results = {}
    with ExitStack() as ownership:
        if scoped:
            # Separate inventories cannot independently book the same host headroom.
            shared = ownership.enter_context((ROOT/'stats/run3/histogram_subset.lock').open('a'))
            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = ownership.enter_context(path.with_suffix('.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if top_pt:
            from TROTASR.workflows.top_pt_inventory import resume_top_pt_results
            results = resume_top_pt_results(path, job)
        old_rows = {}
        history = {}
        retained = None
        if (direct_nonjme or tt_jme) and state_path.is_file():
            previous_state = read_json(state_path)
            if previous_state.get('total') == len(tasks):
                old_rows = previous_state.get('results', {})
                if tt_jme:
                    history = dict(previous_state.get('prior_failures', {}))
                    history.update({k:v for k,v in old_rows.items() if v.get('status') == 'failed'})
                    for t in tasks:
                        row = old_rows.get(t['input'], {})
                        if (row.get('status') != 'complete' or row.get('input') != t['input']
                                or row.get('output') != t['output'] or not row.get('attempts')
                                or row['attempts'][-1].get('exit_code') != 0
                                or not internal_path(ROOT/t['output']).is_file()):
                            continue
                        product = read_json(ROOT/t['output'])
                        c, a = product['contract'],product['audit']
                        if (product.get('status') == 'complete_dataset_contribution'
                                and product.get('scope') == 'background_dataset_contribution'
                                and product.get('sr_data_blinded') is True
                                and c.get('input') == str(internal_path(ROOT/t['input']))
                                and str(c.get('year')) == str(t['year'])
                                and all(c.get(k) == options[k] for k in ('modes','shape_mode','variations'))
                                and c.get('nominal_trota_friend')
                                and (c.get('event_population') == 'corrected_nominal_pass_sr_cr_and_dy_control_v1'
                                     if selected_jme else 'event_population' not in c)
                                and c['dataset_replay']['primaries'] == options['dataset_primaries']
                                and a['events_read'] == t['entries']
                                and sum(a['dataset_rows'].values()) == t['target_entries']):
                            results[t['input']] = row
        if tt_jme:
            import json
            from TROTASR.workflows.histogram_resources import RetainedJMEWorkers
            retained = RetainedJMEWorkers(path,tasks,json.loads(os.environ.pop('TROTASR_JME_HANDOFF','[]')))
            ownership.callback(retained.close)
        reused_at_start = len(results)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = [i for i, task in enumerate(tasks) if task['input'] not in results
                         and (not retained or i not in retained.rows)]
            pending = deque(sorted(remaining, key=sizes.__getitem__, reverse=True))
            if direct_nonjme:
                # Revalidate already published outputs through the normal worker
                # path first, including the old controller's last in-flight files.
                pending = deque(sorted(pending, key=lambda i: not (ROOT/tasks[i]['output']).is_file()))
            if tt_jme:
                pending = deque(sorted(pending, key=lambda i: tasks[i]['input'] not in history))
            active = {}
            booked = {}
            done_bytes = done_events = 0
            if reused_at_start:
                for i, task in enumerate(tasks):
                    prev = results.get(task['input'])
                    if prev is None:
                        continue
                    done_bytes += sizes[i]
                    done_events += task['entries']
                    attempts = prev.get('attempts') or []
                    if len(attempts) == 1:
                        sample = attempts[0].get('memory') or {}
                        if sample.get('output_built'):
                            adaptive.observe(buckets[i], sample.get('peak_rss_bytes'), sample.get('peak_vms_bytes'))
            if tt_jme:
                for i,t in enumerate(tasks):
                    old = history.get(t['input'])
                    if old:
                        mem = (old.get('attempts') or [{}])[-1].get('memory',{})
                        adaptive.observe(buckets[i], mem.get('peak_rss_bytes'), success=False)
            total_events = sum(t['entries'] for t in tasks)
            total_bytes = sum(sizes)
            finalize_reuse = not pending and bool(reused_at_start)
            while pending or active or finalize_reuse or (retained and retained.rows):
                finalize_reuse = False
                inherited = []
                if retained:
                    finished,inherited = retained.poll()
                    for i,tail in finished.items():
                        t = tasks[i]
                        if not retained.other_population and not (ROOT/t['output']).is_file() and tail:
                            history.setdefault(t['input'], dict(input=t['input'],status='failed',
                                               error=tail,source='predecessor',output=t['output']))
                        # Only the canonical worker can validate an adopted
                        # output; a PID disappearing cannot establish success.
                        if t['input'] not in results:
                            pending.appendleft(i)
                external, memory = resources(all_account=True, top_pt_pool=True) if (top_pt or rare_pool) else resources(all_account=True)
                disk_ok = shutil.disk_usage(ROOT).free >= 80*GIB
                resident = histogram_residents(path, booked)
                grants = adaptive.admit(memory,
                    inherited + [(max(v, adaptive.reservation(buckets[i])), resident.get(i, 0)) for i, v in booked.items()],
                    [buckets[i] for i in islice(pending, workers)], account_slots=119-external,
                    now=time.monotonic(), dispatch_limit=(workers if direct_slots or tt_jme else 6) if disk_ok else 0)
                for grant in grants:
                    index = pending.popleft()
                    booked[index] = grant
                    controls = dict(measure_memory=True)
                    if scoped:
                        controls.update(retry_smaller_chunk=False)
                        if not tt_jme:
                            controls['address_space_bytes'] = adaptive.worker_limit_bytes
                    active[pool.submit(run_file_process, path, index, tasks[index], options, **controls)] = index
                completed, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED) if active else ((), ())
                for future in completed:
                    index = active.pop(future)
                    task = tasks[index]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = dict(input=task['input'], status='failed', error=str(error))
                    if direct_nonjme and result.get('status') == 'complete':
                        # Only after the canonical worker has accepted the product,
                        # keep its original production attempt and memory telemetry.
                        old = old_rows.get(task['input'], {})
                        if (old.get('input') == task['input'] and old.get('output') == task['output']
                                and old.get('status') == 'complete' and old.get('attempts')
                                and old['attempts'][-1].get('exit_code') == 0
                                and result['attempts'][-1].get('memory', {}).get('output_built') is False):
                            result = dict(old, revalidated=True)
                    results[task['input']] = result
                    booked.pop(index)
                    ok = result['status'] == 'complete'
                    attempts = result.get('attempts') or [{}]
                    sample = attempts[0].get('memory', {})
                    if not ok or len(attempts) > 1 or sample.get('output_built'):
                        adaptive.observe(buckets[index], sample.get('peak_rss_bytes'),
                                         sample.get('peak_vms_bytes'), success=ok and len(attempts)==1)
                    if ok:
                        done_bytes += sizes[index]
                        done_events += task['entries']
                failures = sum(r['status']=='failed' for r in results.values())
                status = 'running' if pending or active or inherited else 'failed' if failures else 'complete'
                progress = dict(execution_chunk_size=options['chunk_size'],
                    worker_ceiling=workers, admission_ceiling=adaptive.stage_ceiling,
                    complete_input_bytes=done_bytes,
                    total_input_bytes=total_bytes, complete_input_entries=done_events,
                    total_input_entries=total_events)
                if scoped:
                    progress.update(dataset_primaries=options['dataset_primaries'])
                    if not tt_jme:
                        progress['worker_address_space_bytes'] = adaptive.worker_limit_bytes
                if tt_jme:
                    progress.update(worker_address_space_policy='inherit', reused_at_start=reused_at_start,
                                    retained_workers=len(inherited), prior_failures=history,
                                    recovered_prior_failures=sum(results.get(k,{}).get('status')=='complete' for k in history))
                    if selected_jme:
                        progress['event_population'] = 'corrected_nominal_pass_sr_cr_and_dy_control_v1'
                if top_pt or rare_pool:
                    progress.update(reused_at_start=reused_at_start, direct_worker_target=workers if direct_slots else None,
                                    account_slot_policy='qcd_pool_one_per_pair',
                                    account_external_slots=external, account_worker_slots=max(0, 119-external))
                if direct_nonjme:
                    progress.update(direct_worker_target=workers, sample_gate=False,
                                    account_slot_policy='all_account', account_external_slots=external,
                                    account_worker_slots=max(0, 119-external))
                write_json(state_path, dict(status=status, pid=os.getpid(), total=len(tasks),
                    complete=len(results)-failures, failed=failures, active=len(active)+len(inherited),
                    pending=len(pending), results=results, updated=time.time(), **progress))
                if (pending or inherited) and not active and not completed:
                    time.sleep(5)
    return int(any(r['status']=='failed' for r in results.values()))


def prepare_lhe_inventory(population):
    """Bind raw generator weighting to the current MC-only input registry."""
    import fcntl
    from TROTASR.utils.io import read_json, write_json
    from TROTASR.utils.lhe_weights import MC_SHARD
    from TROTASR.utils.paths import ERAS, ROOT, internal_path
    from TROTASR.workflows.validation_regions import inventory

    if population != 'background':
        raise ValueError('Generator nuisances are applied to backgrounds only')
    tasks = {era: [t for t in rows if MC_SHARD.fullmatch(Path(t['input']).name)]
             for era, rows in inventory().items()}
    if set(tasks) != set(ERAS) or not all(tasks.values()):
        raise ValueError('No registered generator coverage in one or more target eras')

    directory = ROOT/'stats/run3/new_signal/lhe'/population
    if internal_path(directory) != directory:
        raise ValueError('Generator inputs and receipts must stay in the canonical product directory')
    path = directory/'inventory.json'
    selected = []
    for era in ERAS:
        for row in tasks[era]:
            relative = Path(row['input'])
            if relative.parent != Path('inputs')/era or str(row['year']) != era:
                raise ValueError('Unexpected generator input path or target era')
            output = directory/'shards'/era/f'{relative.stem}.json.gz'
            selected.append(dict(row, output=str(output.relative_to(ROOT))))
    if len({t['input'] for t in selected}) != len(selected):
        raise ValueError('Duplicate generator input ownership')
    document = dict(schema='trotasr_lhe_histogram_inventory_v1', population=population,
                    tasks=selected, files_by_era={era: len(tasks[era]) for era in ERAS},
                    histogram_options={'lhe_weights': True, 'use_trota_friend': True}, sr_observations_used=False)
    directory.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.is_file():
            def input_scope(record):
                settings = record.get('histogram_options', {})
                return (record.get('population'), {field: settings.get(field) for field in
                        ('lhe_weights', 'use_trota_friend', 'shape_mode', 'variations', 'modes')},
                        {(task['input'], task['output'], task['year'], task['entries'])
                         for task in record['tasks']})
            if input_scope(read_json(path)) != input_scope(document):
                raise ValueError('Preserve the prior generator inventory; inputs changed')
        write_json(path, document)
    return path


def main(default='highdm'):
    p = argparse.ArgumentParser(prog='python -m TROTASR histograms', description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--inventory', type=Path, help='Current per-file task list for batch production and resume')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--task-index', type=int, help=argparse.SUPPRESS)
    p.add_argument('--force-recompute', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--measure-memory', action='store_true', help=argparse.SUPPRESS)
    from TROTASR.utils.paths import ERAS, analysis_era
    p.add_argument('--year', type=analysis_era, choices=ERAS)
    p.add_argument('--output', type=Path)
    p.add_argument('--mode', choices=('highdm', 'lowdm', 'both'), default=None)
    p.add_argument('--chunk-size', type=int)
    p.add_argument('--shape-mode', choices=('cms_trota_jme', 'stored_trota_nonjme'))
    p.add_argument('--variations', nargs='+')
    p.add_argument('--highdm-llcr-only', action='store_true',
                   help='Produce only the canonical high-dM LLCR (no mTb cut)')
    p.add_argument('--dataset-primaries', nargs='+', metavar='PRIMARY',
                   help='Replay only these exact background MC primaries across all requested SR/CRs; '
                        'write contributions below stats/run3/dataset_replay, never full histograms')
    p.add_argument('--trota-friend', action='store_true', default=None,
                   help='Use validated nominal corrected TROTA tags for the SR/CR MC input')
    p.add_argument('--lhe-weights', action='store_true',
                   help='Accumulate original NanoAOD PDF/scale members with the nominal selection')
    p.add_argument('--prepare-lhe-inventory', choices=('background',),
                   help='Register a generator-weight pass for the current background MC')
    p.add_argument('--prepare-top-pt-inventory', action='store_true',
                   help='Register the six-era nominal inclusive-ttbar replay; do not start workers')
    p.add_argument('--run-top-pt-replay', action='store_true',
                   help='Complete nominal ttbar first, then its weight uncertainty on nominal objects')
    p.add_argument('--run-top-pt-existing-systematics', action='store_true',
                   help='Collect the saved ttbar weight pair and propagate existing experimental endpoints only')
    p.add_argument('--cutflow-config', type=Path, help='Read-only SR cutflows; no histograms or inference rerun')
    p.add_argument('--cutflow-worker', type=int, help=argparse.SUPPRESS)
    p.add_argument('--vr-mc', action='store_true', help='Build MC-only VRs with the current SR template bins')
    p.add_argument('--vr-tight-mixed', action='store_true',
                   help='Evaluate only the MC tight-angle single-Mixed mTb<175 GeV candidate')
    p.add_argument('--vr-pilot', action='store_true', help='Use one background and one signal MC shard per era')
    vr_population = p.add_mutually_exclusive_group()
    vr_population.add_argument('--vr-background-only', action='store_true',
                               help='Build the complete approved VR background without old signal inputs')
    vr_population.add_argument('--vr-replacement-signals', action='store_true',
                               help='Complete the approved VR MC with registered UUID signal inputs')
    p.add_argument('--vr-response', action='store_true',
                   help='After nominal VR completion, compute the missing VR background responses')
    p.add_argument('--vr-plot-after-response', action='store_true',
                   help='Render the VR comparison with the saved CR fit when responses are complete')
    p.add_argument('--vr-workers', type=int, default=4)
    args = p.parse_args()
    if args.run_top_pt_existing_systematics:
        if any(getattr(args,key) != p.get_default(key) for key in vars(args)
               if key not in ('run_top_pt_existing_systematics','workers')):
            p.error('The existing-systematics replay accepts only --workers')
        from TROTASR.workflows.top_pt_inventory import run_existing
        raise SystemExit(run_existing(args.workers))
    if args.run_top_pt_replay:
        if any(getattr(args, key) != p.get_default(key) for key in vars(args)
               if key not in ('run_top_pt_replay', 'workers')):
            p.error('--run-top-pt-replay takes only --workers')
        from TROTASR.workflows.top_pt_inventory import run
        raise SystemExit(run(args.workers))
    if args.prepare_top_pt_inventory:
        if any(getattr(args, key) != p.get_default(key) for key in vars(args) if key != 'prepare_top_pt_inventory'):
            p.error('--prepare-top-pt-inventory is a standalone nominal inventory preparation step')
        from TROTASR.workflows.top_pt_inventory import prepare
        print(prepare(), flush=True)
        return
    if args.inventory or args.output:
        from TROTASR.utils.selection_contract import require_current_llcr
        require_current_llcr(args.inventory, args.output)
    if args.dataset_primaries is not None and (
            args.inventory or args.vr_mc or args.cutflow_config or args.prepare_lhe_inventory
            or args.highdm_llcr_only or args.lhe_weights):
        p.error('--dataset-primaries requires a standalone single-file MC histogram; '
                'for an inventory specify histogram_options.dataset_primaries instead')
    if args.force_recompute and (args.inventory is None or args.task_index is None
                                or args.chunk_size is None):
        p.error('--force-recompute requires a single inventory task and fixed --chunk-size')
    if args.measure_memory and (args.inventory is None or args.task_index is None
                                or args.chunk_size is None or args.chunk_size < 1):
        p.error('--measure-memory requires a single inventory task and positive fixed --chunk-size')
    if args.prepare_lhe_inventory:
        if (args.inventory or args.input or args.output or args.year or args.vr_mc or args.cutflow_config
                or args.vr_tight_mixed or args.vr_pilot or args.vr_response or args.vr_plot_after_response
                or args.vr_background_only or args.vr_replacement_signals or args.task_index is not None
                or args.lhe_weights or args.trota_friend or args.shape_mode or args.variations is not None
                or args.mode is not None or args.highdm_llcr_only):
            p.error('--prepare-lhe-inventory is a standalone preparation step')
        print(prepare_lhe_inventory(args.prepare_lhe_inventory))
        return
    if args.inventory:
        if (args.input or args.output or args.year or args.vr_mc or args.cutflow_config
                or args.vr_tight_mixed or args.vr_pilot or args.vr_response or args.vr_plot_after_response
                or args.vr_background_only or args.vr_replacement_signals
                or args.lhe_weights or args.trota_friend or args.shape_mode or args.variations is not None
                or args.mode is not None or args.highdm_llcr_only):
            p.error('--inventory supplies the input/output task list and cannot be combined with single-file or VR options')
        if args.task_index is not None:
            from TROTASR.utils.io import read_json
            job = read_json(args.inventory)
            require_current_llcr(record=job)
            if not 0 <= args.task_index < len(job['tasks']):
                p.error('--task-index is outside the inventory')
            options = dict(job.get('histogram_options', {}))
            if args.chunk_size is not None:
                options['chunk_size'] = args.chunk_size
            controls = {key: True for key in ('force_recompute', 'measure_memory') if getattr(args, key)}
            if options.get('highdm_llcr_only') is not True:
                if args.force_recompute:
                    p.error('--force-recompute is only for high-dM LLCR inventory tasks')
                if args.measure_memory and not _measurable(options):
                    p.error('--measure-memory requires a scoped inventory task')
            run_task(job['tasks'][args.task_index], options, **controls)
            return
        raise SystemExit(run_inventory(args.inventory, args.workers, args.chunk_size))
    if args.task_index is not None:
        p.error('--task-index requires --inventory')
    if args.vr_tight_mixed and not args.vr_mc:
        p.error('--vr-tight-mixed requires --vr-mc')
    if args.vr_tight_mixed and args.vr_response:
        p.error('--vr-tight-mixed is nominal MC-only; cannot combine it with --vr-response')
    if (args.vr_background_only or args.vr_replacement_signals) and (
            not args.vr_mc or args.vr_pilot or args.vr_tight_mixed or args.vr_response):
        p.error('A VR population option requires --vr-mc and cannot be combined with pilot, tight-Mixed or response')
    if args.vr_plot_after_response and not (args.vr_mc and args.vr_response):
        p.error('--vr-plot-after-response requires --vr-mc --vr-response')
    if args.vr_mc:
        if args.trota_friend:
            p.error('--vr-mc uses the accepted nominal friend registry automatically')
        if args.lhe_weights:
            p.error('--lhe-weights is background SR/CR production, not VR production')
        if args.input or args.year or args.output or args.cutflow_config or args.cutflow_worker or args.highdm_llcr_only:
            p.error('--vr-mc uses only the canonical MC inventory and stats/run3/vr output')
        if args.vr_response and args.vr_pilot:
            p.error('--vr-response requires the complete nominal VR and background inventory')
        from TROTASR.workflows.validation_regions import run
        status = run(pilot=args.vr_pilot, workers=args.vr_workers, response=args.vr_response,
                     tight_mixed=args.vr_tight_mixed, background_only=args.vr_background_only,
                     replacement_signals=args.vr_replacement_signals)
        if status == 0 and args.vr_plot_after_response:
            from TROTASR.workflows.plotting_vr_predictions import compare_saved_fit
            compare_saved_fit()
        raise SystemExit(status)
    if args.vr_pilot or args.vr_response:
        p.error('--vr-pilot and --vr-response require --vr-mc')
    if args.output is None:
        p.error('--output is required')
    if args.cutflow_config:
        if args.trota_friend or args.highdm_llcr_only:
            p.error('--trota-friend and --highdm-llcr-only are for histogram production, not cutflows')
        from TROTASR.utils.cutflow import run, worker
        (run if args.cutflow_worker is None else worker)(args.cutflow_config, args.output,
            *(() if args.cutflow_worker is None else (args.cutflow_worker,)))
        return
    if args.input is None or args.year is None:
        p.error('--input and --year are required for histogram production')
    mode = args.mode or default
    modes = ('highdm', 'lowdm') if mode == 'both' else (mode,)
    result = build(args.input, args.year, args.output, modes=modes, chunk_size=2000 if args.chunk_size is None else args.chunk_size,
                   lhe_weights=args.lhe_weights, use_trota_friend=args.trota_friend,
                   shape_mode=args.shape_mode, variations=args.variations or ('nominal',),
                   highdm_llcr_only=args.highdm_llcr_only, dataset_primaries=args.dataset_primaries)
    print(result['status'], 'events=', result['audit']['events_read'], 'seconds=', round(result['seconds'], 2))


if __name__ == '__main__':
    main()
