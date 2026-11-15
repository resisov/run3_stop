"""Histogram accepted events and original NanoAOD weight variations."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.histogramming import build


def run_task(task, options):
    """Process one inventory entry, reusing its completed output."""
    from TROTASR.utils.paths import ROOT, internal_path
    source = internal_path(ROOT / task['input'])
    output = internal_path(ROOT / task['output'])
    stat = source.stat()
    if task['fingerprint']['size'] != stat.st_size:
        raise ValueError('Histogram inventory input size differs: ' + task['input'])
    payload = build(source, task['year'], output, **options)
    if payload['audit']['events_read'] != task['entries']:
        raise ValueError('Histogram event count differs from the inventory')
    return dict(input=task['input'], status='complete', output=task['output'])


def run_file_process(inventory, index, task, options):
    """Retry a failed process once with a smaller event chunk."""
    import subprocess
    import sys
    from TROTASR.utils.io import read_json
    from TROTASR.utils.paths import ROOT
    chunk = options.get('chunk_size', 2000)
    attempts = []
    for size in dict.fromkeys((chunk, max(1, chunk//2))):
        command = [sys.executable, '-m', 'TROTASR', 'histograms', '--inventory', str(inventory),
                   '--task-index', str(index), '--chunk-size', str(size)]
        result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
        attempts.append(dict(chunk_size=size, exit_code=result.returncode, error=result.stderr[-2000:]))
        if result.returncode == 0:
            return dict(input=task['input'], status='complete', output=task['output'], attempts=attempts)
        output = ROOT / task['output']
        if output.exists() and read_json(output).get('status') in ('complete', 'complete_one_file_test', 'complete_member_sums'):
            break
    return dict(input=task['input'], status='failed', attempts=attempts)


def run_inventory(path, workers, chunk_size=None):
    """Run the current task list, with bounded admission and per-file resume."""
    import fcntl
    import os
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from TROTASR.utils.paths import ROOT, internal_path
    from TROTASR.utils.io import read_json, write_json
    from TROTASR.workflows.resources import resources, capacity
    if not 1 <= workers <= 116:
        raise ValueError('Histogram workers must be between 1 and 116')
    path = internal_path(path)
    job = read_json(path)
    tasks = job['tasks']
    if not tasks or len({t['input'] for t in tasks}) != len(tasks) or len({t['output'] for t in tasks}) != len(tasks):
        raise ValueError('Missing or duplicate histogram tasks')
    options = dict(job.get('histogram_options', {}))
    if chunk_size is not None:
        options['chunk_size'] = chunk_size
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    state_path = path.with_name(path.stem + '.state.json')
    plan = dict(max_workers=workers, aggregate_slots=120,
                minimum_available_memory_bytes=32*1024**3, admission_bytes_per_worker=6*1024**3)
    results = {}
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = list(enumerate(tasks))
            active = {}
            while pending or active:
                external, memory = resources()
                limit = capacity(plan, external, memory, len(active))
                if shutil.disk_usage(ROOT).free < 80*1024**3:
                    limit = 0
                while pending and len(active) < limit:
                    index, task = pending.pop(0)
                    active[pool.submit(run_file_process, path, index, task, options)] = task
                completed, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED) if active else ((), ())
                for future in completed:
                    task = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = dict(input=task['input'], status='failed', error=str(error))
                    results[task['input']] = result
                failures = sum(r['status']=='failed' for r in results.values())
                status = 'running' if pending or active else 'failed' if failures else 'complete'
                write_json(state_path, dict(status=status, pid=os.getpid(), total=len(tasks),
                    complete=len(results)-failures, failed=failures, active=len(active),
                    pending=len(pending), results=results, updated=time.time()))
                if pending and not active:
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
            if read_json(path) != document:
                raise ValueError('Preserve the prior generator inventory; inputs changed')
        else:
            write_json(path, document)
    return path


def main(default='highdm'):
    p = argparse.ArgumentParser(prog='python -m TROTASR histograms', description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--inventory', type=Path, help='Current per-file task list for batch production and resume')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--task-index', type=int, help=argparse.SUPPRESS)
    from TROTASR.utils.paths import ERAS, analysis_era
    p.add_argument('--year', type=analysis_era, choices=ERAS)
    p.add_argument('--output', type=Path)
    p.add_argument('--mode', choices=('highdm', 'lowdm', 'both'), default=None)
    p.add_argument('--chunk-size', type=int)
    p.add_argument('--shape-mode', choices=('cms_trota_jme', 'stored_trota_nonjme'))
    p.add_argument('--variations', nargs='+')
    p.add_argument('--trota-friend', action='store_true', default=None,
                   help='Use validated nominal corrected TROTA tags for the SR/CR MC input')
    p.add_argument('--lhe-weights', action='store_true',
                   help='Accumulate original NanoAOD PDF/scale members with the nominal selection')
    p.add_argument('--prepare-lhe-inventory', choices=('background',),
                   help='Register a generator-weight pass for the current background MC')
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
    if args.prepare_lhe_inventory:
        if (args.inventory or args.input or args.output or args.year or args.vr_mc or args.cutflow_config
                or args.vr_tight_mixed or args.vr_pilot or args.vr_response or args.vr_plot_after_response
                or args.vr_background_only or args.vr_replacement_signals or args.task_index is not None
                or args.lhe_weights or args.trota_friend or args.shape_mode or args.variations is not None
                or args.mode is not None):
            p.error('--prepare-lhe-inventory is a standalone preparation step')
        print(prepare_lhe_inventory(args.prepare_lhe_inventory))
        return
    if args.inventory:
        if (args.input or args.output or args.year or args.vr_mc or args.cutflow_config
                or args.vr_tight_mixed or args.vr_pilot or args.vr_response or args.vr_plot_after_response
                or args.vr_background_only or args.vr_replacement_signals
                or args.lhe_weights or args.trota_friend or args.shape_mode or args.variations is not None or args.mode is not None):
            p.error('--inventory supplies the input/output task list and cannot be combined with single-file or VR options')
        if args.task_index is not None:
            from TROTASR.utils.io import read_json
            job = read_json(args.inventory)
            if not 0 <= args.task_index < len(job['tasks']):
                p.error('--task-index is outside the inventory')
            options = dict(job.get('histogram_options', {}))
            if args.chunk_size is not None:
                options['chunk_size'] = args.chunk_size
            run_task(job['tasks'][args.task_index], options)
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
        if args.input or args.year or args.output or args.cutflow_config or args.cutflow_worker:
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
        if args.trota_friend:
            p.error('--trota-friend is for histogram production, not cutflows')
        from TROTASR.utils.cutflow import run, worker
        (run if args.cutflow_worker is None else worker)(args.cutflow_config, args.output,
            *(() if args.cutflow_worker is None else (args.cutflow_worker,)))
        return
    if args.input is None or args.year is None:
        p.error('--input and --year are required for histogram production')
    mode = args.mode or default
    modes = ('highdm', 'lowdm') if mode == 'both' else (mode,)
    use_friend = args.trota_friend
    if use_friend is None:
        from TROTASR.utils.paths import internal_path
        from TROTASR.utils.trota_friend import find_friend_task, friend_production_complete
        if friend_production_complete() and find_friend_task(internal_path(args.input), args.year):
            use_friend = True
    result = build(args.input, args.year, args.output, modes=modes, chunk_size=2000 if args.chunk_size is None else args.chunk_size,
                   lhe_weights=args.lhe_weights, use_trota_friend=use_friend,
                   shape_mode=args.shape_mode, variations=args.variations or ('nominal',))
    print(result['status'], 'events=', result['audit']['events_read'], 'seconds=', round(result['seconds'], 2))


if __name__ == '__main__':
    main()
