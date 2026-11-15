"""Merge per-file histograms and combine the adopted systematic endpoints."""
import argparse
import copy
import math
import gzip
import json
import os
import re
import time
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, scientific_settings, without_provenance_fields


def add_numbers(a, b):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('Histogram/covariance dimensions differ')
        return [add_numbers(x, y) for x, y in zip(a, b)]
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            raise ValueError('Inconsistent boolean invariant')
        return a
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError('Nonnumeric additive field')
    if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(a+b):
        raise ValueError('Nonfinite additive field')
    return a+b


def merge_histograms(target, source):
    if 'sumw' in source:
        if not {'sumw', 'sumw2', 'entries', 'edges'} <= set(source):
            raise ValueError('Incomplete histogram leaf')
        if set(target) != set(source):
            raise ValueError('Histogram leaf schema changed')
        for key, value in source.items():
            if key in ('sumw', 'sumw2', 'entries'):
                target[key] = add_numbers(target[key], value)
            elif target[key] != value:
                raise ValueError('Histogram binning/label changed: ' + key)
        return
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_histograms(target[key], value)
        else:
            raise ValueError('Unexpected histogram node: ' + key)


def merge_counters(target, source):
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_counters(target[key], value)
        else:
            target[key] = add_numbers(target[key], value)


DATASET_SUM_FIELDS = {'events_read', 'events_written', 'files_attempted', 'files_processed',
    'sumw', 'sumw2', 'signal_event_genweight_sum_by_genmodel', 'signal_runs_sumw_source_counts',
    'signal_sumw_by_genmodel', 'sumw_source_counts'}


def merge_datasets(target, source):
    for key, record in source.items():
        if key not in target:
            target[key] = copy.deepcopy(record)
            continue
        previous = target[key]
        if set(previous) != set(record):
            raise ValueError('Dataset metadata schema differs: ' + key)
        for field, value in record.items():
            if field in DATASET_SUM_FIELDS:
                if isinstance(value, dict):
                    merge_counters(previous[field], value)
                else:
                    previous[field] = add_numbers(previous[field], value)
            elif previous[field] != value:
                raise ValueError('Dataset identity/normalization metadata differs: ' + key+'/'+field)


class JsonStream:
    """Copy unchanged JSON values in bounded chunks, without loading the grid."""
    structural = re.compile(r'[{}\[\]"\\]')
    delimiter = re.compile(r'[,\]}\s]')

    def __init__(self, stream):
        self.stream, self.buffer, self.position = stream, '', 0

    def refill(self):
        self.buffer = self.buffer[self.position:] + self.stream.read(65536)
        self.position = 0

    def peek(self):
        while True:
            if self.position == len(self.buffer):
                self.refill()
                if not self.buffer:
                    raise ValueError('Truncated merged JSON')
            if not self.buffer[self.position].isspace():
                return self.buffer[self.position]
            self.position += 1

    def consume(self, expected):
        if self.peek() != expected:
            raise ValueError('Unexpected merged JSON structure')
        self.position += 1

    def copy_value(self, write):
        first = self.peek()
        composite = first in '[{'
        quoted = first == '"'
        depth = 1 if composite else 0
        if composite or quoted:
            write(first); self.position += 1
        while True:
            pattern = self.structural if composite or quoted else self.delimiter
            match = pattern.search(self.buffer, self.position)
            if match is None:
                write(self.buffer[self.position:]); self.position = len(self.buffer)
                self.refill()
                if not self.buffer:
                    if not composite and not quoted:
                        return
                    raise ValueError('Truncated JSON value')
                continue
            index, token = match.start(), match.group()
            if not composite and not quoted:
                write(self.buffer[self.position:index]); self.position = index
                return
            if quoted and token == '\\':
                if index + 1 == len(self.buffer):
                    write(self.buffer[self.position:index]); self.position = index
                    self.refill(); index = 0
                    if len(self.buffer) < 2:
                        raise ValueError('Truncated JSON escape')
                write(self.buffer[self.position:index+2]); self.position = index+2
                continue
            if token == '"':
                quoted = not quoted
            elif not quoted:
                if token in '[{': depth += 1
                elif token in ']}': depth -= 1
            write(self.buffer[self.position:index+1]); self.position = index+1
            if depth == 0 and not quoted:
                return

    def value(self):
        parts = []
        self.copy_value(parts.append)
        return json.loads(''.join(parts))

    def items(self):
        self.consume('{')
        if self.peek() == '}':
            self.consume('}'); return
        while True:
            key = self.value()
            self.consume(':')
            yield key
            if self.peek() == '}':
                self.consume('}'); return
            self.consume(',')


def replace_signal_tree(reader, write, replacement, depth=0):
    """Replace sample nodes, retaining all background/data bytes and axes."""
    write('{'); comma = ''; seen = set()
    for key in reader.items():
        if depth == 4 and re.fullmatch(r'(?:T2bW_|T2tb_)?mStop\d+_mLSP\d+', key):
            reader.copy_value(lambda _: None)
            continue
        write(comma + json.dumps(key) + ':'); comma = ','; seen.add(key)
        if depth < 4:
            replace_signal_tree(reader, write, replacement.get(key, {}), depth+1)
        else:
            reader.copy_value(write)
    for key in sorted(replacement.keys() - seen):
        write(comma + json.dumps(key) + ':' + json.dumps(replacement[key], allow_nan=False, separators=(',', ':')))
        comma = ','
    write('}')


def replace_signals(year, payload_paths, normalization_path):
    """Install complete new signal yields into the current six-era inputs."""
    histograms = dict(histograms={}, physical_histograms={})
    datasets = {}
    events = 0
    for path in payload_paths:
        payload = read_json(path)
        if payload.get('status') not in ('complete', 'complete_one_file_test') or not payload.get('sr_data_blinded'):
            raise ValueError('Incomplete or unblinded signal histogram input')
        if any(not d.get('is_signal') for d in payload['input_metadata'].values()):
            raise ValueError('Signal replacement includes background or data')
        for section in histograms:
            merge_histograms(histograms[section], payload[section])
        merge_datasets(datasets, payload['input_metadata'])
        events += payload['audit']['events_read']
    receipt = dict(files=len(payload_paths), datasets=len(datasets), retained_events=events,
        histograms='production/histograms/'+str(year), efficiencies='production/efficiencies/state.json',
        signal_uncertainties=False, background_and_data_unchanged=True, time=time.time())
    target = ROOT / 'results/merged' / f'systematics_{year}.json.gz'
    temporary = target.with_name(target.name + '.pending.' + str(os.getpid()))
    with gzip.open(target, 'rt') as source, gzip.open(temporary, 'wt', compresslevel=6) as output:
        reader = JsonStream(source)
        output.write('{'); comma = ''
        for key in reader.items():
            if key == 'signal_replacement':
                reader.copy_value(lambda _: None); continue
            output.write(comma + json.dumps(key) + ':'); comma = ','
            if key in histograms:
                replace_signal_tree(reader, output.write, histograms[key])
            elif key == 'input_metadata':
                output.write('{'); inside = ''
                for dataset_id in reader.items():
                    record = reader.value()
                    if not record.get('is_signal'):
                        output.write(inside + json.dumps(dataset_id) + ':' + json.dumps(record, allow_nan=False))
                        inside = ','
                for dataset_id, record in datasets.items():
                    output.write(inside + json.dumps(dataset_id) + ':' + json.dumps(record, allow_nan=False))
                    inside = ','
                output.write('}')
            else:
                reader.copy_value(output.write)
        output.write(comma + '"signal_replacement":' + json.dumps(receipt) + '}\n')
    os.replace(temporary, target)
    write_json(ROOT / 'estimations' / f'normalization_{year}.json.gz', read_json(normalization_path))
    summary_path = target.with_name(f'systematics_{year}.summary.json')
    summary = read_json(summary_path)
    summary.pop('sha256', None)
    summary.update(output=str(target.relative_to(ROOT)), signal_replacement=receipt, completed=time.time())
    write_json(summary_path, summary)
    return receipt


def assemble_endpoints(payloads, endpoints):
    """Join disjoint variations, counting metadata and central traversal once.

    The caller supplies nominal first and validates each shard before yielding
    it. Only one shard is read at a time.
    """
    result, seen = None, set()
    required = {'nominal', *endpoints}
    for payload in payloads:
        endpoint = payload['contract'].get('object_task_endpoint')
        if endpoint not in required or endpoint in seen:
            raise ValueError('Duplicate or unknown endpoint shard')
        seen.add(endpoint)
        if result is None:
            if endpoint != 'nominal':
                raise ValueError('Nominal endpoint must be assembled first')
            result = payload
            continue
        if (without_provenance_fields(payload['input_metadata']) != without_provenance_fields(result['input_metadata']) or
                payload['audit']['events_read'] != result['audit']['events_read']):
            raise ValueError('Endpoint input/normalization accounting mismatch')
        for field in ('schema','sr_data_blinded','bins_merged','highdm_sr_source_axes',
                      'mixed_sf','category_order'):
            if payload.get(field) != result.get(field):
                raise ValueError('Endpoint invariant differs: '+field)
        if Path(payload['contract']['input']).name != Path(result['contract']['input']).name:
            raise ValueError('Endpoint source shard differs')
        for field in ('year', 'modes', 'variations'):
            if payload['contract'].get(field) != result['contract'].get(field):
                raise ValueError('Endpoint contract differs: '+field)
        if payload.get('dy_rz') or payload['audit'].get('migrations'):
            raise ValueError('Shift shard repeats nominal accounting')
        for section in ('histograms','physical_histograms'):
            for mode, variations in payload.get(section, {}).items():
                if set(variations) - {endpoint}:
                    raise ValueError('Endpoint histogram contamination')
                result.setdefault(section, {}).setdefault(mode, {}).update(variations)
        if set(payload.get('dy_rz_variations',{})) - {endpoint}:
            raise ValueError('Endpoint DY contamination')
        result.setdefault('dy_rz_variations',{}).update(payload.get('dy_rz_variations',{}))
        for section in ('shape_migrations','object_endpoints','object_response','object_acceptance'):
            values = payload['audit'].get(section,{})
            if set(values) - {endpoint}:
                raise ValueError('Endpoint audit contamination')
            result['audit'].setdefault(section,{}).update(values)
        for section in ('chunks','weights'):
            values = payload['audit'].get(section,[])
            if any(v.get('endpoint','nominal') != endpoint for v in values):
                raise ValueError('Endpoint chunk/weight contamination')
            result['audit'][section].extend(values)
    if seen != required:
        raise ValueError('Incomplete independent endpoint coverage')
    return result






COMBINATION_POLICY = dict(central='corrected_central_jme', nonjme_endpoints='stored_absolute_histograms',
    nominal_mismatch_acknowledged=True, reanchor=False, recompute_events=False)


def combine_saved_payloads(jme, nonjme, policy):
    """Combine JME central histograms with stored absolute non-JME endpoints.

    Nominal, weights, JME, data and normalization come from JME unchanged.
    The sixteen non-JME absolute endpoint populations retain their own stored
    central convention. This is preliminary and is not central equivalence.
    Inputs may be consumed in memory; persisted source files stay untouched.
    """
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    if scientific_settings(policy) != scientific_settings(COMBINATION_POLICY):
        raise ValueError('Incompatible JME/non-JME central and endpoint definitions')
    endpoints = set(NONJME_ENDPOINTS)
    for payload, scope, mode in ((jme, 'full_cms_trota_jme_production', 'cms_trota_jme'),
            (nonjme, 'full_nonjme_systematic_production', 'stored_trota_nonjme')):
        from TROTASR.workflows.rare_top_policy import require_clean
        require_clean(payload)
        coverage = payload.get('coverage', {})
        if (payload.get('status') != 'complete' or payload.get('scope') != scope
                or payload.get('contract', {}).get('shape_mode') != mode
                or payload.get('sr_data_blinded') is not True
                or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
                or coverage.get('expected_files') != coverage.get('completed_files')):
            raise ValueError('Incomplete or incompatible source campaign')
    for field in ('year', 'modes'):
        if jme['contract'].get(field) != nonjme['contract'].get(field):
            raise ValueError('Systematic input identity differs: '+field)
    for field in ('schema', 'bins_merged', 'highdm_sr_source_axes', 'category_order', 'mixed_sf'):
        if jme.get(field) != nonjme.get(field):
            raise ValueError('Histogram representation differs: '+field)
    # Dataset inventory and bookkeeping are not added a second time.
    mc = {k:v for k,v in jme['input_metadata'].items() if not v.get('is_data')}
    if set(mc) != set(nonjme['input_metadata']):
        raise ValueError('JME/non-JME MC dataset inventories differ')
    compare_values(without_provenance_fields(mc), without_provenance_fields(nonjme['input_metadata']),
                   'MC bookkeeping')
    if set(nonjme['contract'].get('object_endpoints', [])) != endpoints:
        raise ValueError('All sixteen non-JME endpoints are required')
    for section in ('histograms', 'physical_histograms'):
        if set(jme[section]) != {'highdm', 'lowdm'} or set(nonjme[section]) != {'highdm', 'lowdm'}:
            raise ValueError('Both histogram regimes required')
        for mode in ('highdm', 'lowdm'):
            source = nonjme[section][mode]
            target = jme[section][mode]
            if set(source) != {'nominal', *endpoints} or endpoints & set(target):
                raise ValueError('Missing or overlapping systematic endpoints')
            if not {'nominal', 'jesTotalUp', 'jesTotalDown', 'jerUp', 'jerDown'} <= set(target):
                raise ValueError('Missing original JME/nominal endpoints')
            for endpoint in sorted(endpoints):
                for categories in source[endpoint].values():
                    for samples in categories.values():
                        if {'data', 'data_obs'} & set(samples):
                            raise ValueError('Non-JME variation contains collision data')
                target[endpoint] = source[endpoint]
    # On/off-Z endpoint records use endpoint as their outer key.
    varied = nonjme.get('dy_rz_variations', {})
    if set(varied) - endpoints or set(varied) & set(jme.get('dy_rz_variations', {})):
        raise ValueError('Overlapping or unknown DY endpoint records')
    jme.setdefault('dy_rz_variations', {}).update(varied)
    original_contract = jme['contract']
    jme['contract'] = dict(original_contract, shape_mode='jme_with_stored_nonjme',
        object_endpoints=list(NONJME_ENDPOINTS), combination_policy=dict(policy),
        component_contracts=dict(jme=original_contract, nonjme=nonjme['contract']))
    jme.update(scope='full_combined_systematic_production', preliminary=True,
        systematic_scope='JME central/weights/JES/JER plus stored-TROTA non-JME16 absolute endpoints; unequal nominal conventions acknowledged by user',
        central_conventions_consistent=False)
    jme['audit']['nonjme_source_events_read'] = nonjme['audit']['events_read']
    jme['audit']['nonjme_source_coverage'] = nonjme['coverage']
    return jme






def compare_values(before, after, path=''):
    """Require exact counts/structure and 1e-7 relative/absolute float agreement."""
    if isinstance(before, dict):
        if not isinstance(after, dict) or set(before) != set(after):
            raise ValueError('Changed fields: '+path)
        for key in before:
            compare_values(before[key], after[key], path+'/'+str(key))
    elif isinstance(before, list):
        if not isinstance(after, list) or len(before) != len(after):
            raise ValueError('Changed dimensions: '+path)
        for i, (a,b) in enumerate(zip(before,after)):
            compare_values(a,b,path+'/'+str(i))
    elif isinstance(before, float):
        if not isinstance(after,(float,int)) or not math.isfinite(before) or not math.isfinite(after) or not math.isclose(before,after,rel_tol=1e-7,abs_tol=1e-7):
            raise ValueError('Changed numerical result: '+path)
    elif before != after:
        raise ValueError('Changed count/value: '+path)


def validate_input(payload, task, settings):
    from TROTASR.utils.histogramming import histogram_settings
    from TROTASR.utils.paths import analysis_era
    if payload.get('status') not in ('complete', 'complete_one_file_test') or payload.get('sr_data_blinded') is not True:
        raise ValueError('Incomplete or unblinded histogram: ' + task['input'])
    contract = payload['contract']
    if analysis_era(contract['year']) != analysis_era(task['year']):
        raise ValueError('Histogram belongs to a different target era')
    if Path(contract['input']).name != Path(task['input']).name:
        raise ValueError('Histogram belongs to a different input')
    if payload['audit']['events_read'] != task['entries']:
        raise ValueError('Histogram event coverage differs from the input inventory')
    common = {k: v for k, v in contract.items() if k != 'signal_normalization'}
    if histogram_settings(common) != histogram_settings(settings):
        raise ValueError('Histogram corrections, region definitions or binning differ')
    for section in ('histograms', 'physical_histograms'):
        validate_histograms(payload.get(section, {}))
        for modes in payload.get(section, {}).values():
            for regions in modes.values():
                for samples in regions.get('SR', {}).values():
                    if {'data', 'data_obs'} & set(samples):
                        raise ValueError('SR observations in histogram input')
    for chunk in payload['audit'].get('chunks', ()):
        if any(isinstance(v, dict) and v.get('intersection', 0) for v in chunk.values()):
            raise ValueError('High-/low-dM selections overlap')
    from TROTASR.workflows.rare_top_policy import require_clean
    require_clean(payload)
    return payload


def validate_histograms(node):
    """Keep signed yields, nonnegative variances and the declared axis shape."""
    import numpy as np
    if not isinstance(node, dict):
        raise ValueError('Histogram node must be a mapping')
    if 'sumw' not in node:
        for child in node.values():
            validate_histograms(child)
        return
    if not {'sumw', 'sumw2', 'entries', 'edges'} <= node.keys():
        raise ValueError('Missing histogram fields')
    arrays = {key: np.asarray(node[key]) for key in ('sumw','sumw2','entries')}
    if any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite or nonnumeric histogram content')
    if any(a.shape != arrays['sumw'].shape for a in arrays.values()):
        raise ValueError('Histogram yield, variance and entry shapes differ')
    if (arrays['sumw2'] < 0).any() or (arrays['entries'] < 0).any() or (arrays['entries'] % 1 != 0).any():
        raise ValueError('Invalid histogram variance or entry count')
    edges = node['edges']
    axes = edges if arrays['sumw'].ndim > 1 else [edges]
    if len(axes) != arrays['sumw'].ndim or tuple(len(e)-1 for e in axes) != arrays['sumw'].shape:
        raise ValueError('Histogram axes do not match its contents')
    if any(not np.isfinite(e).all() or not (np.diff(e) > 0).all() for e in axes):
        raise ValueError('Histogram edges must be finite and increasing')


def merge_fields(path, fields):
    """Read the requested fields of a closed merge without retaining its arrays."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        reader = JsonStream(stream)
        result = {}
        for key in reader.items():
            if key in fields:
                result[key] = reader.value()
            else:
                reader.copy_value(lambda _: None)
        if reader.buffer[reader.position:].strip() or stream.read().strip():
            raise ValueError('Unexpected data after merged JSON')
    return result


def merge(inventory, year, output):
    """Merge the complete per-era task list produced by the histograms stage."""
    import fcntl
    from TROTASR.utils.histogramming import histogram_settings
    from TROTASR.utils.paths import analysis_era
    year = analysis_era(year)
    inventory, output = internal_path(inventory), internal_path(output)
    document = read_json(inventory)
    tasks = [t for t in document['tasks'] if analysis_era(t['year']) == year]
    if not tasks or len({t['input'] for t in tasks}) != len(tasks):
        raise ValueError('Missing or duplicate source files in the merge inventory')
    sources = {t['input']: dict(output=t['output']) for t in tasks}
    expected_events = sum(t['entries'] for t in tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(output.name.removesuffix('.gz').removesuffix('.json') + '.summary.json')
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            saved = merge_fields(output, {'status', 'scope', 'contract', 'coverage',
                                          'sr_data_blinded', 'merge_provenance'})
            coverage = saved.get('coverage', {})
            contract = saved.get('contract', {})
            saved_sources = saved.get('merge_provenance', {}).get('sources', {})
            options = document.get('histogram_options', {})
            shape = options.get('shape_mode')
            if shape is None and options.get('use_trota_friend'):
                shape = 'cms_trota_jme'
            if (saved.get('status') != 'complete' or saved.get('sr_data_blinded') is not True
                    or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
                    or coverage.get('completed_files') != len(tasks)
                    or coverage.get('expected_files') != len(tasks)
                    or coverage.get('expected_events') != expected_events
                    or str(contract.get('year')) != year
                    or (('shape_mode' in options or 'use_trota_friend' in options)
                        and contract.get('shape_mode') != shape)
                    or any(contract.get(field) != list(options[field])
                           for field in ('modes', 'variations') if field in options)
                    or (options.get('object_endpoints') is not None
                        and set(contract.get('object_endpoints', ())) != set(options['object_endpoints']) - {'nominal'})
                    or sorted(Path(p).name for p in saved_sources) != sorted(Path(p).name for p in sources)):
                raise ValueError('Existing merge uses different histogram inputs: ' + str(output))
            if not summary_path.is_file():
                write_json(summary_path, dict(status='complete', output=str(output.relative_to(ROOT)),
                    sources=sources, coverage=coverage, physics_contract=contract,
                    scope=saved['scope'], completed=time.time()))
            return output
        merged = None
        for task in tasks:
            payload = read_json(ROOT / task['output'])
            if merged is None:
                settings = histogram_settings({k: v for k, v in payload['contract'].items()
                                               if k != 'signal_normalization'})
                mode = settings.get('shape_mode')
                scope = {'cms_trota_jme': 'full_cms_trota_jme_production',
                         'stored_trota_nonjme': 'full_nonjme_systematic_production'}.get(mode, 'full_nominal_production')
                merged = {k: copy.deepcopy(payload[k]) for k in
                          ('schema', 'sr_data_blinded', 'bins_merged', 'highdm_sr_source_axes', 'mixed_sf', 'category_order')}
                merged.update(status='complete', scope=scope, contract=settings,
                    histograms={}, physical_histograms={}, dy_rz={}, dy_rz_variations={}, input_metadata={},
                    audit=dict(events_read=0, migrations={}, selection_totals={}),
                    background_policy=dict(excluded_primaries=['TTWH','TTZH'], excluded_files=[]))
            validate_input(payload, task, settings)
            for section in ('histograms', 'physical_histograms', 'dy_rz', 'dy_rz_variations'):
                merge_histograms(merged[section], payload.get(section, {}))
            merge_datasets(merged['input_metadata'], payload['input_metadata'])
            merged['audit']['events_read'] += payload['audit']['events_read']
            for section in ('migrations', 'trota_inference', 'shape_migrations', 'object_endpoints', 'object_response', 'object_acceptance'):
                if section in payload['audit']:
                    merge_counters(merged['audit'].setdefault(section, {}), payload['audit'][section])
            for chunk in payload['audit'].get('chunks', ()):
                endpoint = chunk.get('endpoint', 'nominal')
                values = {k: v for k, v in chunk.items() if k not in ('SR','endpoint')}
                merge_counters(merged['audit']['selection_totals'].setdefault(endpoint, {}), values)
            merged['background_policy']['excluded_files'].extend(payload.get('background_policy', {}).get('excluded_files', []))
        if merged['audit']['events_read'] != expected_events:
            raise ValueError('Merged event coverage is incomplete')
        merged['coverage'] = dict(expected_files=len(tasks), completed_files=len(tasks), failed_files=0,
                                  expected_events=expected_events, full_input_inventory=True)
        merged['merge_provenance'] = dict(inventory=str(inventory.relative_to(ROOT)), sources=sources)
        write_json(output, merged)
        write_json(summary_path, dict(status='complete', output=str(output.relative_to(ROOT)),
            sources=sources, coverage=merged['coverage'],
            physics_contract=settings, scope=merged['scope'], completed=time.time()))
        return output


def combine(jme_path, nonjme_path, year, output):
    """Combine completed endpoint sets, reusing a matching completed output."""
    import fcntl
    from TROTASR.utils.histogramming import histogram_settings
    paths = [internal_path(p) for p in (jme_path, nonjme_path)]
    output = internal_path(output)
    sources = [str(p.relative_to(ROOT)) for p in paths]
    policy = read_json(ROOT / 'jsons/analysis.json')['systematic_combination']
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(output.name.removesuffix('.gz').removesuffix('.json') + '.summary.json')
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            saved = merge_fields(output, {'status', 'scope', 'contract', 'coverage', 'sr_data_blinded'})
            contract = saved.get('contract', {})
            coverage = saved.get('coverage', {})
            components = contract.get('component_contracts', {})
            if (saved.get('status') != 'complete' or saved.get('sr_data_blinded') is not True
                    or saved.get('scope') != 'full_combined_systematic_production'
                    or str(contract.get('year')) != str(year)
                    or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
                    or not coverage.get('expected_files')
                    or coverage.get('expected_files') != coverage.get('completed_files')
                    or set(components) != {'jme', 'nonjme'}
                    or scientific_settings(contract.get('combination_policy')) != scientific_settings(policy)):
                raise ValueError('Existing combined histogram uses different inputs or endpoints: ' + str(output))
            for path, component in zip(paths, ('jme', 'nonjme')):
                if path.is_file():
                    source = merge_fields(path, {'contract', 'status', 'sr_data_blinded'})
                    if (source.get('status') != 'complete' or source.get('sr_data_blinded') is not True
                            or histogram_settings(source['contract']) != histogram_settings(components[component])):
                        raise ValueError('Combined histogram component physics changed: ' + component)
            if not summary_path.is_file():
                write_json(summary_path, dict(status='complete', output=str(output.relative_to(ROOT)),
                    sources=sources, combination_policy=policy, coverage=coverage,
                    scope=saved['scope'], completed=time.time()))
            return output
        inputs = [read_json(p) for p in paths]
        if any(str(p['contract']['year']) != str(year) for p in inputs):
            raise ValueError('Systematic inputs belong to a different era')
        payload = combine_saved_payloads(*inputs, policy)
        write_json(output, payload)
        write_json(summary_path, dict(status='complete', output=str(output.relative_to(ROOT)),
            sources=sources, combination_policy=policy,
            coverage=payload['coverage'], scope=payload['scope'], completed=time.time()))
    return output


def merge_generator_weights(inventory, year, output):
    """Publish member sums, not nuisance endpoints, from a registered theory inventory."""
    import fcntl
    from TROTASR.utils.lhe_weights import merge_lhe_member_payloads
    from TROTASR.utils.paths import analysis_era

    year = analysis_era(year)
    inventory, output = internal_path(inventory), internal_path(output)
    lhe_root = ROOT/'stats/run3/new_signal/lhe'
    population = inventory.parent.name
    if (population != 'background' or inventory != lhe_root/population/'inventory.json'
            or output != lhe_root/population/f'members_{year}.json.gz'):
        raise ValueError('Use the canonical generator member paths for this population and era')
    with inventory.with_suffix('.lock').open('a') as active:
        fcntl.flock(active, fcntl.LOCK_SH | fcntl.LOCK_NB)
        document = read_json(inventory)
        tasks = [t for t in document['tasks'] if analysis_era(t['year']) == year]
        if (document.get('schema') != 'trotasr_lhe_histogram_inventory_v1'
                or document.get('population') != population
                or document.get('histogram_options', {}).get('lhe_weights') is not True
                or not tasks or len(tasks) != document.get('files_by_era', {}).get(year)
                or len({t['output'] for t in tasks}) != len(tasks)):
            raise ValueError('Invalid generator production inventory')
        def load(task):
            path = internal_path(ROOT/task['output'])
            if path.parent != lhe_root/population/'shards'/year:
                raise ValueError('Cross-population or wrong-era LHE histogram path')
            return read_json(path) if path.is_file() else None

        with output.with_suffix('.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if output.exists():
                previous = read_json(output)
                if previous.get('status') == 'complete_member_sums':
                    return 0
            result = merge_lhe_member_payloads(tasks, load)
            result.update(inventory=str(inventory.relative_to(ROOT)), population=population, era=year,
                          sr_observations_used=False, endpoints_installed=False)
            write_json(output, result)
            return 0 if result.get('status') == 'complete_member_sums' else 2


def main():
    from TROTASR.utils.paths import ERAS
    parser = argparse.ArgumentParser(prog='python -m TROTASR merge', description=__doc__)
    parser.add_argument('--year', required=True, choices=ERAS)
    parser.add_argument('--inventory', type=Path, help='Histogram task list with input, output, entries and fingerprint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jme', type=Path, help='Complete JME year histogram')
    parser.add_argument('--nonjme', type=Path, help='Complete stored non-JME year histogram')
    parser.add_argument('--lhe', action='store_true', help='Merge raw generator members without nuisance reduction')
    args = parser.parse_args()
    if args.lhe:
        if args.jme or args.nonjme or not args.inventory:
            parser.error('--lhe requires its own generator histogram inventory')
        raise SystemExit(merge_generator_weights(args.inventory, args.year, args.output))
    if args.jme or args.nonjme:
        if not args.jme or not args.nonjme or args.inventory:
            parser.error('--jme and --nonjme are used together, without --inventory')
        print(combine(args.jme, args.nonjme, args.year, args.output))
    else:
        if not args.inventory:
            parser.error('--inventory is required for per-file merging')
        print(merge(args.inventory, args.year, args.output))


if __name__ == '__main__':
    main()
