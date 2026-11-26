"""Register nominal inclusive-ttbar inputs from the canonical six-era metadata."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json

from TROTASR.utils.io import read_json, write_json
from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT, internal_path
from TROTASR.utils.top_pt import CONTRIBUTION_DIRECTORY, ENDPOINTS, PRESCRIPTION, PRIMARIES
from TROTASR.workflows.rare_top_policy import primary


RELATIVE = Path(CONTRIBUTION_DIRECTORY)
RECEIPT_PREFIX = 'TROTASR_TOP_PT_RECEIPT '


def _options(variations):
    return dict(dataset_primaries=list(PRIMARIES), modes=['highdm', 'lowdm'], top_pt_prescription=PRESCRIPTION,
                chunk_size=1_000_000, use_trota_friend=True, variations=list(variations))


def _metadata_header(path, early):
    """Parse top-level JSON values up to the small dataset summary, not per-source file records."""
    wanted = {'datasets', 'events_written'} | ({'campaign_year'} if early else set())
    decoder = json.JSONDecoder()

    def read_prefix(text):
        pos, stop = 0, len(text)
        def token(at):
            while at < stop and text[at] in ' \r\t\n':
                at += 1
            if at >= stop:
                raise ValueError('incomplete JSON prefix')
            return at
        pos = token(pos)
        if text[pos] != '{':
            raise ValueError('expected a JSON object')
        pos = token(pos + 1)
        found = {}
        while text[pos] != '}':
            name, pos = decoder.raw_decode(text, pos)
            pos = token(pos)
            if not isinstance(name, str) or text[pos] != ':':
                raise ValueError('invalid object member')
            value, pos = decoder.raw_decode(text, token(pos + 1))
            pos = token(pos)
            if text[pos] not in ',}':
                raise ValueError('incomplete object value')
            if name in wanted or name == 'status':
                found[name] = value
                if wanted <= found.keys():
                    return found
            if text[pos] == '}':
                break
            pos = token(pos + 1)
        return found

    with path.open(encoding='utf-8') as stream:
        prefix = ''
        for size in (16384, 65536, 262144, 524288):
            prefix += stream.read(size - len(prefix))
            try:
                return read_prefix(prefix)
            except ValueError:
                if len(prefix) < size:
                    raise ValueError('Invalid metadata JSON: ' + str(path))
        # Nonstandard member order, or exceptionally large dataset summary.
        stream.seek(0)
        meta = json.load(stream)
        return {key: meta[key] for key in wanted | {'status'} if key in meta}


def _target_task(task):
    """Never open ROOT objects or inspect any observed events to make a plan."""
    era, source = str(task['year']), Path(task['input'])
    sidecar = Path(task['metadata'])
    if (era not in ERAS or source.parent != Path('inputs') / era or source.suffix != '.root'
            or sidecar != source.with_suffix('.json') or task.get('vr_kind') != 'background'):
        raise ValueError('Unexpected accepted background input: ' + str(source))
    meta = _metadata_header(internal_path(ROOT / sidecar), era in EARLY_ERAS)
    # The accepted inventory already certifies these files. The worker validates the
    # full input status before processing; do not scan GB of later records for it here.
    if ('status' in meta and meta['status'] not in ('complete', 'complete_with_bad_files')
            or type(task['entries']) is not int or task['entries'] < 0
            or (era in EARLY_ERAS and meta.get('campaign_year') != era)
            or meta.get('events_written') != task['entries']):
        raise ValueError('Background metadata event coverage differs: ' + str(sidecar))
    datasets = meta['datasets']
    if (not isinstance(datasets, dict) or (not datasets and task['entries'])
            or any(not isinstance(r.get('dataset'), str) or not r['dataset']
                   or r.get('is_data') or r.get('is_signal') for r in datasets.values())):
        raise ValueError('Incomplete or non-background dataset identity: ' + str(sidecar))
    matches = [(primary(r), r.get('events_written')) for r in datasets.values() if primary(r) in PRIMARIES]
    if any(n is not None and (type(n) is not int or n < 0) for _, n in matches):
        raise ValueError('Invalid per-dataset event coverage: ' + str(sidecar))
    primaries = sorted({p for p, n in matches if n != 0})
    if not primaries or task['entries'] == 0:
        return None
    target_entries = None if any(n is None for _, n in matches) else sum(n for _, n in matches)
    if target_entries is not None and target_entries > task['entries']:
        raise ValueError('Target dataset entries exceed file coverage: ' + str(sidecar))
    return dict(year=era, input=str(source), entries=task['entries'], vr_kind='background',
                primaries_present=primaries, target_entries=target_entries,
                output=str(RELATIVE / 'nominal' / era / (source.stem + '.json.gz')))


def prepare():
    from TROTASR.workflows.validation_regions import _input_inventory

    accepted = _input_inventory()
    if set(accepted) != set(ERAS):
        raise ValueError('Missing canonical target eras')
    background = {era: [t for t in accepted[era] if t['vr_kind'] == 'background'] for era in ERAS}
    directory = internal_path(ROOT / RELATIVE)
    path = directory / 'nominal.json'
    directory.mkdir(exist_ok=True, parents=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, coverage = [], {}
        with ThreadPoolExecutor(max_workers=4) as io:
            for era in ERAS:
                tasks = background[era]
                selected = [row for row in io.map(_target_task, tasks) if row is not None]
                if not selected:
                    raise ValueError('No accepted inclusive ttbar inputs in ' + era)
                coverage[era] = dict(background_files_scanned=len(tasks), files=len(selected),
                                     whole_file_entries=sum(t['entries'] for t in selected),
                                     target_entries=(sum(t['target_entries'] for t in selected)
                                                     if all(t['target_entries'] is not None for t in selected)
                                                     else None),
                                     primaries_present=sorted({p for row in selected for p in row['primaries_present']}))
                rows.extend(selected)
                print('top-pT nominal', era, coverage[era], flush=True)
        if len(set(row['input'] for row in rows)) != len(rows):
            raise ValueError('Duplicate input ownership in the nominal ttbar inventory')
        document = dict(schema='trotasr_ttbar_top_pt_nominal_inventory_v1', top_pt_prescription=PRESCRIPTION,
                        coverage=coverage, tasks=rows, sr_observations_used=False,
                        histogram_options=_options(('nominal',)))
        if path.exists():
            saved = read_json(path)
            saved_options = {key: value for key, value in saved['histogram_options'].items()
                             if key != 'chunk_size'}
            expected_options = {key: value for key, value in document['histogram_options'].items()
                                if key != 'chunk_size'}
            if (saved['tasks'] != rows or saved.get('top_pt_prescription') != PRESCRIPTION
                    or saved_options != expected_options or saved.get('sr_observations_used') is not False):
                raise ValueError('Existing nominal ttbar inventory has different inputs or physics settings')
        else:
            write_json(path, document)
    return path


def product_receipt(task, options, payload):
    """Check the committed (or scientifically reused) per-file result in the worker."""
    contract, audit = payload['contract'], payload['audit']
    rows = audit['dataset_rows']
    if (options.get('top_pt_prescription') != PRESCRIPTION or contract.get('top_pt_prescription') != PRESCRIPTION
            or payload.get('status') != 'complete_dataset_contribution'
            or payload.get('scope') != 'background_dataset_contribution'
            or set(contract.get('modes', ())) != {'highdm', 'lowdm'}
            or contract.get('variations') != options['variations']
            or contract.get('shape_mode') != 'cms_trota_jme' or not contract.get('nominal_calibrations_only')
            or (contract.get('nominal_trota_friend') or {}).get('nominal_only') is not True
            or str(contract['year']) != task['year'] or Path(contract['input']) != ROOT / task['input']
            or set(contract['dataset_replay']['primaries']) != set(PRIMARIES)
            or set(rows) != set(contract['dataset_replay']['dataset_ids'])
            or any(type(n) is not int or n < 0 for n in rows.values())
            or audit['events_read'] != task['entries']
            or task.get('target_entries') is not None and sum(rows.values()) != task['target_entries']
            or not internal_path(ROOT / task['output']).is_file()):
        raise ValueError('Invalid corrected nominal-geometry top-pT contribution: ' + task['input'])
    return dict(prescription=PRESCRIPTION, variations=options['variations'], target_entries=sum(rows.values()),
                entries=task['entries'], year=task['year'])


def _matches_receipt(task, variations, result):
    """The result is indexed by its exact input; validate its scientific receipt."""
    if not isinstance(result, dict) or result.get('status') != 'complete':
        return False
    receipt = result.get('top_pt_receipt')
    if not isinstance(receipt, dict):
        return False
    n = receipt.get('target_entries')
    return (receipt.get('prescription') == PRESCRIPTION and receipt.get('year') == task['year']
            and receipt.get('variations') == variations
            and type(receipt.get('entries')) is int and receipt['entries'] == task['entries']
            and type(n) is int and 0 <= n <= task['entries']
            and (task.get('target_entries') is None or n == task['target_entries'])
            and internal_path(ROOT / task['output']).is_file())


def resume_top_pt_results(inventory, plan):
    """Restore committed rows under the inventory lock; other rows go to the worker."""
    stage = inventory.stem
    if stage not in ('nominal', 'uncertainty') or inventory != ROOT / RELATIVE / (stage + '.json'):
        return {}
    variations = ['nominal'] if stage == 'nominal' else list(ENDPOINTS)
    if (plan.get('schema') != f'trotasr_ttbar_top_pt_{stage}_inventory_v1'
            or plan.get('top_pt_prescription') != PRESCRIPTION
            or any(plan.get('histogram_options', {}).get(key) != value
                   for key, value in _options(variations).items() if key != 'chunk_size')):
        return {}
    try:
        state = read_json(inventory.with_suffix('.state.json'))
    except FileNotFoundError:
        return {}
    tasks = plan['tasks']
    if (not isinstance(state, dict) or state.get('status') not in ('running', 'failed', 'complete')
            or state.get('total') != len(tasks) or not isinstance(state.get('results'), dict)
            or state.get('dataset_primaries') != list(PRIMARIES)):
        return {}
    return {t['input']: state['results'][t['input']] for t in tasks
            if _matches_receipt(t, variations, state['results'].get(t['input']))}


def _require_finished(inventory, plan):
    state = read_json(inventory.with_suffix('.state.json'))
    tasks = plan['tasks']
    if (state.get('status') != 'complete' or state.get('failed') != 0
            or state.get('total') != len(tasks) or state.get('complete') != len(tasks)
            or len(state.get('results', {})) != len(tasks)):
        raise ValueError('Not all top-pT contributions completed: ' + str(inventory))
    for task in tasks:
        if not _matches_receipt(task, plan['histogram_options']['variations'], state['results'].get(task['input'])):
            raise ValueError('Missing matching top-pT worker receipt: ' + task['input'])


def experimental_plans(nominal, remainder):
    """Rebuild the TT histogram *slot*, not all grouped Top. ST is untouched."""
    from TROTASR.workflows.rare_top_policy import POLICY
    all_primaries = sorted(set(PRIMARIES) | set(POLICY['top_primaries']))
    inputs = {}
    for plan, allowed in ((nominal, set(PRIMARIES)), (remainder, set(POLICY['top_primaries']))):
        if set(plan['histogram_options']['dataset_primaries']) != allowed:
            raise ValueError('Changed TT histogram-slot membership')
        for t in plan['tasks']:
            era, source = str(t['year']), Path(t['input'])
            prim = set(t['primaries_present'])
            if (era not in ERAS or source.parent != Path('inputs')/era or not prim or not prim <= allowed
                    or not 0 <= t['target_entries'] <= t['entries']):
                raise ValueError('Invalid experimental replay source: '+str(source))
            row = inputs.setdefault(str(source), dict(input=str(source), year=era, entries=t['entries'],
                primaries_present=[], target_entries=0, vr_kind='background'))
            if row['entries'] != t['entries'] or set(row['primaries_present']) & prim:
                raise ValueError('Duplicate or incompatible MC dataset rows in '+str(source))
            row['primaries_present'] = sorted(set(row['primaries_present']) | prim)
            row['target_entries'] += t['target_entries']
            if row['target_entries'] > row['entries']:
                raise ValueError('Overlapping TT contributions in '+str(source))
    if set(r['year'] for r in inputs.values()) != set(ERAS):
        raise ValueError('Incomplete six-era TT plan')
    for stage, mode, variations, friend in (
            ('existing_nonjme', 'stored_trota_nonjme', ['nominal'], False),
            ('existing_jme_weights_nominal_pass', 'cms_trota_jme', ['all_weights'], True)):
        yield stage, dict(schema='trotasr_tt_existing_systematics_v1', top_pt_prescription=PRESCRIPTION,
            population='TT_slot_only', sr_observations_used=False, lhe_weights=False,
            histogram_options=dict(dataset_primaries=all_primaries, modes=['highdm','lowdm'],
                chunk_size=1_000_000, shape_mode=mode, variations=variations, use_trota_friend=friend,
                **({'nominal_pass_only': True} if friend else {})),
            tasks=[dict(t, output=str(RELATIVE/stage/t['year']/(Path(t['input']).stem+'.json.gz')))
                   for _, t in sorted(inputs.items())])


def collect_top_pt_pair(era, nominal, shifted):
    """Collect both *inclusive ttbar* absolute templates; no event ROOT reads."""
    from TROTASR.workflows.merge_nominal import merge_histograms
    import numpy as np
    def leaves(tree, prefix=()):
        if 'sumw' in tree:
            yield prefix, tree
        else:
            for k, v in tree.items():
                yield from leaves(v, prefix+(k,))
    result = {'histograms': {}, 'physical_histograms': {}}
    nt = {t['input']:t for t in nominal['tasks'] if t['year'] == era}
    st = {t['input']:t for t in shifted['tasks'] if t['year'] == era}
    if not nt or nt.keys() != st.keys():
        raise ValueError('Nominal/Up/Down inventory coverage mismatch: '+era)
    destination = ROOT/RELATIVE/'uncertainty'/(era+'.json.gz')
    if destination.is_file():
        saved = read_json(destination)
        expected = dict(schema='trotasr_ttbar_top_pt_paired_contribution_v1', year=era,
                        status='complete', scope='inclusive_ttbar_only', prescription=PRESCRIPTION,
                        files=len(nt), target_entries=sum(t['target_entries'] for t in nt.values()),
                        endpoints=list(ENDPOINTS), sr_data_blinded=True)
        if (not all(saved.get(k) == v for k, v in expected.items())
                or any(t['target_entries'] != st[k]['target_entries'] for k,t in nt.items())
                or not isinstance(saved.get('endpoint_pairs_checked'), int) or saved['endpoint_pairs_checked'] < 1
                or not all(saved.get(section) for section in result)):
            raise ValueError('Existing paired output does not match this TT replay: '+era)
        return {k:saved[k] for k in ('year','files','target_entries','endpoint_pairs_checked')}
    for key in nt:
        n, s = nt[key], st[key]
        if n['target_entries'] != s['target_entries']:
            raise ValueError('Different target populations: '+key)
        for t,plan in ((n,nominal),(s,shifted)):
            product = read_json(ROOT/t['output'])
            product_receipt(t,plan['histogram_options'],product)
            for section, out in result.items():
                merge_histograms(out,product[section])
    count = 0
    for section in result.values():
        for mode, tree in section.items():
            all_leaves = [dict(leaves(tree[v])) for v in ('nominal',*ENDPOINTS)]
            if not (all_leaves[0].keys() == all_leaves[1].keys() == all_leaves[2].keys()):
                raise ValueError('Different bin support for a fixed-object weight: '+mode)
            for key in all_leaves[0]:
                n,u,d = [rows[key] for rows in all_leaves]
                if (key[2] != 'TT' or not n['entries'] == u['entries'] == d['entries']
                        or not n['edges'] == u['edges'] == d['edges']
                        or not np.allclose(np.asarray(u['sumw'])+d['sumw'], 2*np.asarray(n['sumw']),
                                           rtol=1e-8,atol=1e-8)):
                    raise ValueError('Top-pT fixed-bin or signed mirror mismatch: '+str(key))
                count += 1
    if not count:
        raise ValueError('No top-pT histogram cells: '+era)
    result.update(schema='trotasr_ttbar_top_pt_paired_contribution_v1',year=era,status='complete',
        scope='inclusive_ttbar_only',prescription=PRESCRIPTION,files=len(nt),
        target_entries=sum(t['target_entries'] for t in nt.values()),endpoint_pairs_checked=count,
        endpoints=list(ENDPOINTS),sr_data_blinded=True)
    write_json(destination,result)
    return {k:result[k] for k in ('year','files','target_entries','endpoint_pairs_checked')}


def run_existing(workers):
    """Collect the completed pair and replay precisely the two adopted detector conventions."""
    from TROTASR.workflows.build_highdm_histogram import run_inventory
    if not 1 <= workers <= 24:
        raise ValueError('Use 1–24 workers for TT systematic replay')
    directory = ROOT/RELATIVE
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = {}
        for stage in ('nominal','uncertainty'):
            path = directory/(stage+'.json')
            plans[stage] = read_json(path)
            _require_finished(path, plans[stage])
        remainder = read_json(directory/'remainder.json')
        jobs = list(experimental_plans(plans['nominal'], remainder))
        for stage, job in jobs:
            path = directory/(stage+'.json')
            if path.exists():
                if read_json(path) != job:
                    raise ValueError('Existing experimental inventory differs: '+stage)
            else:
                write_json(path,job)
        with ThreadPoolExecutor(max_workers=1) as collect:
            def collect_all():
                for era in ERAS:
                    print('top_pt_pair',collect_top_pt_pair(era,plans['nominal'],plans['uncertainty']),flush=True)
            pair = collect.submit(collect_all)
            for stage, job in jobs:
                if pair.done(): pair.result()
                path = directory/(stage+'.json')
                if run_inventory(path,workers): return 1
            pair.result()
    return 0


def run(workers):
    """One controller: nominal commit barrier, then only the weight endpoints on nominal objects."""
    from TROTASR.workflows.build_highdm_histogram import run_inventory

    if not 1 <= workers <= 48:
        raise ValueError('Top-pT replay requires 1–48 workers')
    directory = internal_path(ROOT / RELATIVE)
    directory.mkdir(exist_ok=True, parents=True)
    with (directory / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        nominal = prepare()
        plan = read_json(nominal)
        if run_inventory(nominal, workers):
            return 1
        _require_finished(nominal, plan)
        shifted = directory / 'uncertainty.json'
        variation_plan = dict(plan, schema='trotasr_ttbar_top_pt_uncertainty_inventory_v1',
                              histogram_options=_options(ENDPOINTS), tasks=[
                                  dict(t, output=str(RELATIVE / 'uncertainty' / t['year'] /
                                                     (Path(t['input']).stem + '.json.gz')))
                                  for t in plan['tasks']])
        with shifted.with_suffix('.lock').open('a') as plan_lock:
            fcntl.flock(plan_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if shifted.exists():
                if read_json(shifted) != variation_plan:
                    raise ValueError('Existing top-pT uncertainty inventory does not match nominal coverage')
            else:
                write_json(shifted, variation_plan)
        if run_inventory(shifted, min(workers, 24)):
            return 1
        _require_finished(shifted, variation_plan)
    return 0
