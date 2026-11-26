"""Project nominal signal histograms once; reuse the verified SR162/SR30 map.

This is a histogram boundary adapter, not an event cache or a new selection.
It does not read ROOT, infer scores, change normalization or merge observed bins.
The native signal packing below follows build_run2_linked_grid.signal_data.
"""
from collections import defaultdict
import gzip
import math
import re
import numpy as np
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ERAS, ROOT, internal_path
from TROTASR.utils.selection_contract import require_current_llcr
from TROTASR.utils.signal_models import signal_mass_from_genmodel, signal_mass_key
from TROTASR.utils.sr_binning import AXES, NATIVE_EDGES, expand, matches
from TROTASR.utils.statistics_inputs import build_inputs, HistogramView
from TROTASR.utils.statistical_templates import assemble

MODELS = ('T2tt', 'T2bW', 'T2tb')
FIELDS = ('hists', 'sgamma', 'rz_high', 'rz_low', 'double_ratio')
FIT_SUPPORT_PROBLEMS = frozenset(('zero_integral_nonempty_component',
    'zero_integral_variation', 'zero_nominal_integral',
    'zero_variation_integral', 'zero_signal_acceptance'))


def require_fit_support_issues(issues):
    """Only zero support may be serialized as explicitly unfit templates.

    Invalid inputs, observations, parameter links or numerical arrays are not
    waived by the separate template-preservation path.
    """
    if any(i.get('problem') not in FIT_SUPPORT_PROBLEMS for i in issues):
        raise ValueError('Not a zero-support fit-compatibility issue: ' + str(issues))


def mass_pair(mass):
    match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', mass)
    if not match:
        raise ValueError('Invalid signal mass key: ' + mass)
    return tuple(map(int, match.groups()))


def requested_grid(path):
    grid = read_json(internal_path(path))
    if set(grid['models']) != set(MODELS):
        raise ValueError('All three signal topologies are required')
    for model, masses in grid['models'].items():
        if not masses or len(set(masses)) != len(masses):
            raise ValueError('Empty or duplicate signal grid')
        for mass in masses:
            stop, lsp = mass_pair(mass)
            if not 0 <= lsp < stop:
                raise ValueError('Invalid signal masses')
        if len(masses) != grid['expected_points_by_model'][model]:
            raise ValueError('Incomplete requested mass inventory')
    return grid


def represented_signals(payload):
    """Use Runs bookkeeping, not SR acceptance, to establish input coverage."""
    totals = defaultdict(float)
    for record in payload['input_metadata'].values():
        if not record.get('is_signal'):
            continue
        for name, value in record.get('signal_sumw_by_genmodel', {}).items():
            model, stop, lsp = signal_mass_from_genmodel(name)
            if model not in MODELS or stop is None or lsp is None or not math.isfinite(float(value)):
                raise ValueError('Malformed signal Runs mass-point bookkeeping')
            totals[model + '/mStop%d_mLSP%d' % (stop, lsp)] += float(value)
    return {name: value for name, value in totals.items() if value != 0.}


def project_signals(payload, model_input, grid, gnn, *, nominal_only=False):
    """Linear signed-yield/sumw2 projection, performed once per populated cell."""
    requested = {signal_mass_key(model, *mass_pair(mass)): (model, mass)
                 for model in MODELS for mass in grid['models'][model]}
    signals = {model: {mass: {mode: dict(nominal=np.zeros(n), sumw2=np.zeros(n), variations={})
                             for mode, n in (('highdm', 162), ('lowdm', 30))}
                       for mass in grid['models'][model]} for model in MODELS}
    layout = expand()
    destination = {c['key']: {} for c in layout}
    for target, entry in enumerate(model_input['highdm_mapping']):
        for native in entry['native_recoil_bins']:
            if native in destination[entry['category']]:
                raise ValueError('Duplicate native recoil cell')
            destination[entry['category']][native] = target
    if any(set(cells) != set(range(8)) for cells in destination.values()):
        raise ValueError('Incomplete native recoil projection')
    low_categories = gnn['sr_binning']['category_labels']
    excluded = set()
    conservation = {}
    view = None if nominal_only else HistogramView(payload, payload['contract']['year'], gnn)
    for mode in ('highdm', 'lowdm'):
        source = payload['histograms'][mode]
        hist = {'nominal': source['nominal']} if nominal_only else source
        nominal = hist['nominal'].get('SR', {})
        categories = {c for v in hist.values() for c in v.get('SR', {})}
        destinations = {}
        for category in categories:
            if mode == 'highdm':
                counts = tuple(map(int, category.split(',')))
                hits = [c for c in layout if len(counts) == 5 and
                        all(matches(v, c['predicates'][a]) for a, v in zip(AXES, counts))]
                if len(hits) != 1:
                    raise ValueError('Uncovered/overlapping exact high-dM SR category')
                indices = [destination[hits[0]['key']][i] for i in range(8)]
                observable, edges = 'recoil', NATIVE_EDGES
            else:
                offset = 5 * low_categories.index(category)
                indices = list(range(offset, offset + 5))
                observable, edges = 'gnn_score', gnn['sr_binning']['edges_by_category'][category]
            destinations[category] = indices, observable, edges
        conservation[mode] = {}
        for endpoint in sorted(hist, key=lambda n: (n != 'nominal', n)):
            varied = hist[endpoint].get('SR', {})
            is_shape = view is not None and endpoint in view.shape_endpoints
            if endpoint != 'nominal':
                base = endpoint[:-2] if endpoint.endswith('Up') else endpoint[:-4]
                nuisance = view.variations[mode][base]
                direction = 'up' if endpoint.endswith('Up') else 'down'
                for masses in signals.values():
                    for point in masses.values():
                        point[mode]['variations'].setdefault(nuisance, {})[direction] = np.zeros_like(point[mode]['nominal'])
            totals = conservation[mode][endpoint] = dict(sumw=0., sumw2=0., entries=0)
            for category, (indices, observable, edges) in destinations.items():
                old = nominal.get(category, {})
                now = varied.get(category, {})
                samples = set(now) if is_shape else set(old) | set(now)
                for sample in samples:
                    if sample == 'data':
                        raise ValueError('SR data are forbidden')
                    if sample not in requested:
                        if sample.startswith(('mStop', 'T2tt_', 'T2tb_', 'T2bW_')):
                            excluded.add(sample)
                        continue
                    record = now.get(sample) if is_shape else now.get(sample, old.get(sample))
                    if record is None:
                        continue
                    leaf = record[observable]
                    arrays = [np.asarray(leaf[field], dtype=float) for field in ('sumw', 'sumw2', 'entries')]
                    if leaf['edges'] != edges or any(a.shape != (len(indices),) or not np.isfinite(a).all() for a in arrays):
                        raise ValueError('Malformed signal histogram axis/content')
                    if any((a < 0).any() for a in arrays[1:]) or not np.array_equal(arrays[2], np.floor(arrays[2])):
                        raise ValueError('Invalid signal variance/event counts')
                    model, mass = requested[sample]
                    target = signals[model][mass][mode]
                    if endpoint == 'nominal':
                        for field, values in zip(('nominal', 'sumw2'), arrays):
                            np.add.at(target[field], indices, values)
                    else:
                        np.add.at(target['variations'][nuisance][direction], indices, arrays[0])
                    for field, values in zip(('sumw', 'sumw2', 'entries'), arrays):
                        totals[field] += float(values.sum())
            def content(s):
                return s[mode]['nominal'] if endpoint == 'nominal' else s[mode]['variations'][nuisance][direction]
            total = sum(content(s).sum() for masses in signals.values() for s in masses.values())
            if not math.isclose(total, totals['sumw'], rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError('Signal endpoint projection does not conserve ' + mode + '/' + endpoint)
            if endpoint == 'nominal':
                total2 = sum(s[mode]['sumw2'].sum() for masses in signals.values() for s in masses.values())
                if not math.isclose(total2, totals['sumw2'], rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError('Signal projection does not conserve variance')
    source_bins = {}
    for mode in ('highdm', 'lowdm'):
        for i, channel in enumerate(model_input['groups']['SR_' + mode]):
            source_bins[channel['name']] = [(mode, i)]
    return signals, source_bins, dict(conservation=conservation,
        unrequested_histogram_signal_keys=sorted(excluded), bins=dict(highdm=162, lowdm=30))


def replacement_points(accepted_grid):
    """Select the published 895 points from the accepted 955-point inventory."""
    from TROTASR.utils.era_merge import displayed_points
    if (accepted_grid.get('status') != 'cards_ready'
            or accepted_grid.get('scope') != 'full_combined_systematic_production'
            or accepted_grid.get('era_merged') is not True
            or accepted_grid.get('sr_data_blinded') is not True):
        raise ValueError('Signal replacement requires the accepted blinded era-merged grid')
    points = displayed_points(accepted_grid['points'])
    if not all(p.get('fit_ready') is True and isinstance(p.get('card'), str) for p in points):
        raise ValueError('Displayed point has no accepted card binding')
    return points


def replacement_production_receipts(state):
    """Return the six committed generations, not a partial replacement."""
    merged, completed = state.get('merged', {}), state.get('completed', {})
    count = state.get('total')
    if (state.get('central') != 'cms_trota_jme_nominal'
            or state.get('signal_input') != 'regenerated_signal_uuid'
            or set(merged) != set(ERAS) or type(count) is not int or count <= 0
            or len(completed) != count or set(state.get('normalization_by_era', {})) != set(ERAS)):
        raise ValueError('Complete six-era signal refresh receipt required')
    by_era = dict.fromkeys(ERAS, 0)
    for item in completed.values():
        era = item['era']
        if era not in by_era:
            raise ValueError('Unexpected target era in signal refresh receipt')
        by_era[era] += 1
    for era, item in merged.items():
        c = item.get('coverage', {})
        replacement = item.get('signal_replacement', {})
        if (item.get('status') != 'complete' or item.get('scope') != 'full_cms_trota_jme_production'
                or c.get('full_input_inventory') is not True or c.get('failed_files') != 0
                or c.get('expected_files') != by_era[era] or c.get('completed_files') != by_era[era]
                or by_era[era] <= 0 or not c.get('expected_events')
                or c.get('expected_events') != c.get('processed_events')
                or replacement.get('central') != 'cms_trota_jme_nominal'
                or replacement.get('signal_input') != 'regenerated_signal_uuid'
                or replacement.get('signal_uncertainties') is not False
                or replacement.get('background_and_data_unchanged') is not True
                or replacement.get('files') != by_era[era]
                or replacement.get('retained_events') != c['expected_events']):
            raise ValueError('Incomplete nominal-only signal merge: '+era)
    return {era:merged[era] for era in ERAS}


def _corrected_signal_contract(contract, era):
    """Check the physical configuration. The histogram reader checks SR coverage."""
    return (str(contract.get('year')) == era and contract.get('shape_mode') == 'cms_trota_jme'
            and contract.get('variations') == ['nominal']
            and contract.get('nominal_calibrations_only') is True
            and contract.get('signal_sr_only') is True
            and contract.get('nominal_trota_friend') is None and contract.get('lhe_weights') is None)


def _read_nominal_replacement(path, names):
    """Stream only signal MC leaves; never decode background/observation arrays."""
    from TROTASR.workflows.merge_nominal import JsonStream

    def select(reader, keys):
        for key in reader.items():
            if key in keys:
                yield key
            else:
                reader.copy_value(lambda _: None)

    metadata = ('status', 'scope', 'sr_data_blinded', 'contract', 'coverage', 'highdm_sr_source_axes',
                'input_metadata', 'histograms', 'physical_histograms', 'signal_replacement')
    result = {}
    with gzip.open(path, 'rt') as stream:
        reader = JsonStream(stream)
        for field in select(reader, metadata):
            if field == 'histograms':
                modes = {}
                for mode in reader.items():
                    if mode not in ('highdm', 'lowdm'):
                        raise ValueError('Corrected signal contains an unapproved histogram mode')
                    for endpoint in reader.items():
                        if endpoint != 'nominal':
                            raise ValueError('Corrected SR signal contains a non-nominal endpoint')
                        for region in reader.items():
                            if region != 'SR':
                                raise ValueError('Corrected signal histogram contains a non-SR region')
                            categories = {}
                            for category in reader.items():
                                samples = {}
                                for sample in reader.items():
                                    if sample in ('data', 'data_obs'):
                                        raise ValueError('Observed SR histograms are forbidden')
                                    if not re.fullmatch(r'(?:T2bW_|T2tb_)?mStop\d+_mLSP\d+', sample):
                                        raise ValueError('Corrected signal-only SR contains a non-signal sample')
                                    if sample in names:
                                        samples[sample] = reader.value()
                                    else:
                                        reader.copy_value(lambda _: None)
                                if samples:
                                    categories[category] = samples
                            modes[mode] = {'nominal': {'SR': categories}}
                result[field] = modes
            elif field == 'physical_histograms':
                for _ in reader.items():
                    raise ValueError('Corrected SR signal has unexpected auxiliary histograms')
                result[field] = {}
            elif field == 'input_metadata':
                datasets = {}
                for key in reader.items():
                    record = reader.value()
                    if record.get('is_signal') is not True or record.get('is_data'):
                        raise ValueError('Corrected signal-only input registers background or data')
                    datasets[key] = record
                result[field] = datasets
            else:
                result[field] = reader.value()
    return result


def replacement_source_binding(path, year, production_receipt, normalization_path=None):
    """Select the completed corrected signal input and its Runs table."""
    path = internal_path(path)
    if (production_receipt.get('status') != 'complete'
            or production_receipt.get('output') != str(path.relative_to(ROOT))
            or production_receipt.get('scope') != 'full_cms_trota_jme_production'
            or not path.is_file()):
        raise ValueError('Corrected signal merge receipt does not match production: '+year)
    normalization = internal_path(normalization_path or ROOT/'estimations'/f'normalization_{year}.json.gz')
    if not normalization.is_file():
        raise FileNotFoundError('Missing approved signal Runs normalization: '+year)
    return dict(path=str(path.relative_to(ROOT)), normalization=str(normalization.relative_to(ROOT)),
        replacement=production_receipt['signal_replacement'])


def read_replacement_model(path):
    """Read only the native mapping and SR ordinal labels from an accepted cache."""
    from TROTASR.workflows.merge_nominal import JsonStream
    result, background = {}, {}
    with gzip.open(internal_path(path), 'rt') as stream:
        reader = JsonStream(stream)
        for key in reader.items():
            if key in ('year', 'status', 'sr_data_blinded'):
                result[key] = reader.value()
            elif key == 'background':
                for field in reader.items():
                    if field in ('year', 'highdm_mapping'):
                        background[field] = reader.value()
                    elif field == 'groups':
                        groups = {}
                        for group in reader.items():
                            if group not in ('SR_highdm','SR_lowdm'):
                                reader.copy_value(lambda _:None)
                                continue
                            rows = []
                            reader.consume('[')
                            while reader.peek() != ']':
                                row = {}
                                for column in reader.items():
                                    if column == 'name':
                                        row[column] = reader.value()
                                    else:
                                        reader.copy_value(lambda _:None)
                                rows.append(row)
                                if reader.peek() == ']':
                                    break
                                reader.consume(',')
                            reader.consume(']')
                            groups[group] = rows
                        background[field] = groups
                    else:
                        reader.copy_value(lambda _:None)
            else:
                reader.copy_value(lambda _:None)
    if (result.get('status') != 'complete' or result.get('sr_data_blinded') is not True
            or result.get('year') != background.get('year') or len(background.get('highdm_mapping', [])) != 162
            or {k:len(v) for k,v in background.get('groups', {}).items()} != {'SR_highdm':162,'SR_lowdm':30}):
        raise ValueError('Accepted cache does not contain the expected native SR mapping')
    return background


def extract_replacement_year(path, year, points, model_input, gnn, *, production_receipt,
                             accepted_contract, normalization_path=None):
    """Project only the current replacement's nominal signals on the accepted native bins."""
    from TROTASR.gnn4lowdm.configuration import configuration_path
    if year not in ERAS or str(model_input['year']) != year:
        raise ValueError('Wrong accepted native model era')
    path = internal_path(path)
    source = replacement_source_binding(path, year, production_receipt, normalization_path)
    models = {m:[] for m in MODELS}
    for p in points:
        models[p['model']].append(p['mass'])
    names = {signal_mass_key(m, *mass_pair(mass)) for m in models for mass in models[m]}
    payload = _read_nominal_replacement(path, names)
    contract = payload['contract']
    coverage = payload.get('coverage', {})
    if (payload.get('status') != 'complete' or payload.get('sr_data_blinded') is not True
            or payload.get('scope') != 'full_cms_trota_jme_production'
            or not _corrected_signal_contract(contract, year)
            or any(coverage.get(k) != production_receipt['coverage'].get(k) for k in
                   ('expected_files', 'completed_files', 'expected_events', 'processed_events', 'failed_files'))
            or payload.get('highdm_sr_source_axes') != 'exact_nb_exact_topology_native_recoil_v2'
            or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
            or coverage.get('expected_files') != coverage.get('completed_files')
            or not coverage.get('expected_files')
            or set(payload['histograms']) != {'highdm', 'lowdm'} or payload.get('physical_histograms') != {}
            or not payload['input_metadata']
            or len(payload['input_metadata']) != production_receipt['signal_replacement'].get('datasets')):
        raise ValueError('Incomplete, misbound or unblinded merged signal input: '+year)
    gnn_path = configuration_path(year)
    fields = ('category_labels', 'edges_by_category')
    adopted = read_json(gnn_path)['sr_binning']
    if any(gnn['sr_binning'][key] != adopted[key] for key in fields):
        raise ValueError('Replacement signal does not match the accepted GNN/SR binning: '+year)
    normalized = internal_path(ROOT/source['normalization'])
    table = read_json(normalized)
    registration = table.get('signal_replacement', {})
    if (registration.get('files') != coverage['expected_files']
            or registration.get('datasets') != len(payload['input_metadata'])
            or registration.get('luminosity_changed') is not False
            or registration.get('background_factors_changed') is not False):
        raise ValueError('Replacement signal does not match the published Runs table: '+year)
    represented = represented_signals(payload)
    for point in points:
        model, mass = point['model'], point['mass']
        runs = represented.get(model+'/'+mass)
        spec = table['signal_mass_points'].get(signal_mass_key(model, *mass_pair(mass)), {})
        if (runs is None or not math.isclose(runs, float(spec.get('sumw_mass_point', 0.)), rel_tol=2e-12)
                or spec.get('normalization_factor') is None
                or not math.isfinite(float(spec['normalization_factor']))):
            raise ValueError('Replacement/normalization Runs mismatch: '+year+'/'+model+'/'+mass)
    signals, sources, audit = project_signals(payload, model_input, dict(models=models), gnn, nominal_only=True)
    return dict(year=year, signals=signals, source_bins=sources, projection_audit=audit,
                source=source, signal_input_coverage=represented)


def replacement_signal_object(channel, point):
    """Use the same ordinal ROOT key as the accepted card shape binding."""
    if (not re.fullmatch(r'SR_(?:highdm|lowdm)_c1_(?:'+'|'.join(ERAS)+r')', channel)
            or point['model'] not in MODELS):
        raise ValueError('Unexpected nominal signal template channel or model')
    mass_pair(point['mass'])
    return channel+'/signal_'+point['model']+'_'+point['mass']


def replacement_channel_records(extracted, year, point, channels, highdm_groups):
    """Clip SR162 source bins before the approved SR118 union; preserve sumw2."""
    if year != extracted['year'] or sorted(i for group in highdm_groups for i in group) != list(range(162)):
        raise ValueError('Wrong target era or non-partitioned high-dM projection')
    expected = {f'SR_{mode}_c1_{year}' for mode in ('highdm', 'lowdm')}
    if set(channels) != expected:
        raise ValueError('Expected exactly the two adopted SR channels for '+year)
    native, sources = {}, {}
    for channel, labels in channels.items():
        if channel.startswith('SR_highdm_'):
            if len(highdm_groups) != 118 or labels != [f'SR_highdm_bin{i}' for i in range(118)]:
                raise ValueError('Changed high-dM SR118 template labels')
            for label, group in zip(labels, highdm_groups):
                sources[year, label] = [('highdm', i) for i in group]
        else:
            lowdm = [name for name, entries in extracted['source_bins'].items() if entries[0][0] == 'lowdm']
            if len(lowdm) != 30 or labels != lowdm:
                raise ValueError('Changed low-dM SR30 template labels')
            for label in labels:
                sources[year, label] = extracted['source_bins'][label]
        native[channel] = [dict(year=year, name=label) for label in labels]
    # Retain empty *era/channel* columns where needed by the published shape
    # bindings; callers must reject a mass point with no total SR support.
    return signal_data(native, sources, {year:dict(signals=extracted['signals'][point['model']])},
                       point['mass'], keep_empty=True)


def signal_data(channels, source_bins, grids, mass, keep_empty=False):
    """Legacy native signal packing, with per-bin negative clipping audit."""
    result, issues, clipped = {}, [], []
    for c, bins in channels.items():
        if not c.startswith('SR_'):
            continue
        year = bins[0]['year']
        records = []
        for b in bins:
            sources = source_bins[year, b['name']]
            regimes = {r for r, i in sources}
            if len(regimes) != 1:
                raise ValueError('mixed signal axes')
            regime = next(iter(regimes)); indices = [i for r, i in sources]
            r = grids[year]['signals'][mass][regime]
            # Preserve the historical order: signed cells -> SR162 bins ->
            # per-bin negative clipping -> the adopted SR118 partition.
            for endpoint, array in [('nominal', r['nominal'])] + [
                    (n+d, v) for n, pair in r['variations'].items() for d, v in pair.items()]:
                for i in indices:
                    if array[i] < 0:
                        clipped.append(dict(year=year, channel=c, bin=b['name'], mass=mass,
                            source_bin=i, endpoint=endpoint, before=float(array[i]), after=0.))
            records.append(dict(nominal=sum(max(r['nominal'][i], 0.) for i in indices),
                sumw2=sum(r['sumw2'][i] for i in indices),
                variations={n: {d: sum(max(v[i], 0.) for i in indices) for d, v in pair.items()}
                            for n, pair in r['variations'].items()}))
        names = {n for r in records for n in r['variations']}
        values = np.asarray([r['nominal'] for r in records]); variance = np.asarray([r['sumw2'] for r in records])
        if not np.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('invalid signal variance')
        pairs = {n: {d: np.asarray([r['variations'].get(n, {}).get(d, r['nominal']) for r in records])
                     for d in ('up', 'down')} for n in names}
        endpoints = [('nominal', values)] + [(n + d, v) for n, pair in pairs.items() for d, v in pair.items()]
        for endpoint, a in endpoints:
            if not np.isfinite(a).all():
                raise ValueError('nonfinite signal')
            for index in np.flatnonzero(a < 0):
                clipped.append(dict(year=year, channel=c, bin=bins[index]['name'], mass=mass,
                                    endpoint=endpoint, before=float(a[index]), after=0.))
            a[a < 0] = 0.
        if (not keep_empty and values.sum() == 0 and not variance.any()
                and not any(v.any() for p in pairs.values() for v in p.values())):
            continue
        if values.sum() <= 0:
            issues.append(dict(channel=c, problem='zero_nominal_integral'))
        for n, pair in pairs.items():
            for d, v in pair.items():
                if v.sum() <= 0:
                    issues.append(dict(channel=c, problem='zero_variation_integral', nuisance=n, direction=d))
        result[c, 'signal'] = dict(family='signal', parameter='', extra_lnN=(), initial=1.,
                                  nominal=values, sumw2=variance, variations=pairs)
    if not result:
        issues.append(dict(problem='zero_signal_acceptance'))
    return result, issues, clipped


def check_reference(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('benchmark signal channel mismatch')
    for key in expected:
        for field in ('nominal', 'sumw2'):
            np.testing.assert_allclose(actual[key][field], expected[key][field], rtol=1e-12, atol=1e-14)
        for n in set(expected[key]['variations']) | set(actual[key]['variations']):
            for d in ('up', 'down'):
                np.testing.assert_allclose(actual[key]['variations'].get(n, {}).get(d, actual[key]['nominal']),
                    expected[key]['variations'].get(n, {}).get(d, expected[key]['nominal']), rtol=1e-12, atol=1e-14)


def extract_year(paths, grid, allow_validation=False, preserve_unsupported=False,
                 replacement=None):
    """Load each large histogram input once; no repeated ROOT or JSON passes."""
    paths = {key: internal_path(ROOT / paths[key]) for key in FIELDS}
    inputs = {key: read_json(path) for key, path in paths.items()}
    payload = inputs['hists']
    if replacement is None:
        require_current_llcr(paths['hists'], record=payload)
    from TROTASR.workflows.rare_top_policy import require_clean
    require_clean(payload)
    coverage = payload.get('coverage', {})
    production_coverage = (
        payload.get('scope') in ('full_nominal_production', 'full_cms_trota_jme_production',
                                 'full_combined_systematic_production')
        and bool(coverage.get('full_input_inventory'))
        and coverage.get('failed_files') == 0
        and 'expected_files' in coverage
        and coverage.get('expected_files') == coverage.get('completed_files'))
    if not allow_validation and not production_coverage:
        raise ValueError('Full-input production coverage is required')
    if payload.get('scope') == 'full_combined_systematic_production':
        from TROTASR.utils.statistics_inputs import combined_shape_endpoints
        combined_shape_endpoints(payload)
    if replacement is None:
        represented = represented_signals(payload)
        missing = [model + '/' + mass for model in MODELS for mass in grid['models'][model]
                   if model + '/' + mass not in represented]
        if missing:
            raise ValueError('Requested signal inputs absent from nonzero Runs bookkeeping: ' + ', '.join(missing))
    else:
        if ((not payload.get('tt_card_input') if replacement.get('tt_card') else
             payload['contract'].get('highdm_llcr_mtb_min_gev') != 175.0)
                or payload.get('scope') != 'full_combined_systematic_production'
                or not production_coverage):
            raise ValueError('Missing complete high-dM LLCR cut systematic inputs')
    model_input = build_inputs(*(inputs[k] for k in FIELDS), 'T2tt', 1200, 500)
    from TROTASR.gnn4lowdm.configuration import configuration_path
    gnn = read_json(configuration_path(payload['contract']['year']))
    year = model_input['year']
    if replacement is None:
        signals, source_bins, audit = project_signals(payload, model_input, grid, gnn)
    else:
        points = [dict(model=model, mass=mass) for model in MODELS for mass in grid['models'][model]]
        projected = extract_replacement_year(ROOT/replacement['receipt']['output'], year, points,
            model_input, gnn, production_receipt=replacement['receipt'],
            accepted_contract=replacement['accepted_contract'])
        signals, source_bins, audit = (projected[k] for k in
            ('signals', 'source_bins', 'projection_audit'))
        represented = projected['signal_input_coverage']
        for rows in model_input['groups'].values():
            for row in rows:
                row['physical'].pop('signal', None)
    channels, data, clipping, issues = assemble([model_input])
    if model_input['issues']:
        raise ValueError('Invalid statistical input model: ' + str(model_input['issues']))
    require_fit_support_issues(issues)
    if issues and not preserve_unsupported:
        raise ValueError('Unsupported statistical background/benchmark model: ' + str(issues))
    if replacement is None and 'mStop1200_mLSP500' in grid['models']['T2tt']:
        packed, problems, _ = signal_data(channels, {(year, k): v for k, v in source_bins.items()},
            {year: dict(signals=signals['T2tt'])}, 'mStop1200_mLSP500')
        check_reference({k: v for k, v in data.items() if k[1] == 'signal'}, packed)
        audit['benchmark_signal_matches_single_point'] = True
    for bins in model_input['groups'].values():
        for channel in bins:
            channel['physical'].pop('signal', None)
    return dict(schema='trotasr_nominal_grid_inputs_v1', status='complete', year=year,
        scope=payload['scope'], production_coverage=production_coverage,
        background=model_input, signals=signals, source_bins=source_bins,
        inputs={k: dict(path=str(p.relative_to(ROOT))) for k,p in paths.items()},
        signal_input_coverage={k: represented[k] for k in sorted(represented)},
        projection_audit=audit, negative_background_audit=[a for a in clipping if a['process'] != 'signal'],
        full_workflow_complete=False, sr_data_blinded=True,
        fit_support_issues=issues, preserved_unsupported_templates=preserve_unsupported,
        **({'replacement_signal_source': projected['source'],
            **({'tt_card_input':payload['tt_card_input']} if replacement.get('tt_card') else
               {'highdm_llcr_mtb_min_gev':175.0})} if replacement is not None else {}))
