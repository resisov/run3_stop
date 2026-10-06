"""Project nominal signal histograms once; reuse the verified SR162/SR30 map.

This is a histogram boundary adapter, not an event cache or a new selection.
It does not read ROOT, infer scores, change normalization or merge observed bins.
The native signal packing below follows build_run2_linked_grid.signal_data.
"""
from collections import defaultdict
import math
import re
import numpy as np
from TROTASR.utils.io import read_json, sha256
from TROTASR.utils.paths import ROOT, internal_path
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


def project_signals(payload, model_input, grid, gnn):
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
    view = HistogramView(payload, payload['contract']['year'], gnn)
    for mode in ('highdm', 'lowdm'):
        hist = payload['histograms'][mode]
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
            is_shape = endpoint in view.shape_endpoints
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


def extract_year(paths, grid, allow_validation=False, preserve_unsupported=False):
    """Load each large histogram input once; no repeated ROOT or JSON passes."""
    paths = {key: internal_path(ROOT / paths[key]) for key in FIELDS}
    digests = {key: sha256(path) for key, path in paths.items()}
    inputs = {key: read_json(path) for key, path in paths.items()}
    payload = inputs['hists']
    if not allow_validation and (payload.get('scope') not in ('full_nominal_production', 'full_cms_trota_jme_production', 'full_combined_systematic_production')
            or not payload.get('coverage', {}).get('full_input_inventory')
            or payload['coverage']['failed_files'] != 0
            or payload['coverage']['expected_files'] != payload['coverage']['completed_files']):
        raise ValueError('Full-input production coverage is required')
    if payload.get('scope') == 'full_combined_systematic_production':
        from TROTASR.utils.statistics_inputs import combined_shape_endpoints
        combined_shape_endpoints(payload)
    for key in FIELDS[1:]:
        if inputs[key].get('provenance', {}).get('hist_input_sha256') != digests['hists']:
            raise ValueError('Background measurement does not match supplied histograms: ' + key)
    for field, path in (('sr_binning_sha256', ROOT / 'jsons/highdm_sr_binning.json'),
                        ('configuration_sha256', ROOT / 'gnn4lowdm/config.json')):
        if payload['contract'].get(field) != sha256(path):
            raise ValueError('Histogram physics contract mismatch: ' + field)
    represented = represented_signals(payload)
    missing = [model + '/' + mass for model in MODELS for mass in grid['models'][model]
               if model + '/' + mass not in represented]
    if missing:
        raise ValueError('Requested signal inputs absent from nonzero Runs bookkeeping: ' + ', '.join(missing))
    model_input = build_inputs(*(inputs[k] for k in FIELDS), 'T2tt', 1200, 500)
    gnn = read_json(ROOT / 'gnn4lowdm/config.json')
    signals, source_bins, audit = project_signals(payload, model_input, grid, gnn)
    year = model_input['year']
    channels, data, clipping, issues = assemble([model_input])
    if model_input['issues']:
        raise ValueError('Invalid statistical input model: ' + str(model_input['issues']))
    require_fit_support_issues(issues)
    if issues and not preserve_unsupported:
        raise ValueError('Unsupported statistical background/benchmark model: ' + str(issues))
    if 'mStop1200_mLSP500' in grid['models']['T2tt']:
        packed, problems, _ = signal_data(channels, {(year, k): v for k, v in source_bins.items()},
            {year: dict(signals=signals['T2tt'])}, 'mStop1200_mLSP500')
        check_reference({k: v for k, v in data.items() if k[1] == 'signal'}, packed)
        audit['benchmark_signal_matches_single_point'] = True
    for bins in model_input['groups'].values():
        for channel in bins:
            channel['physical'].pop('signal', None)
    return dict(schema='trotasr_nominal_grid_inputs_v1', status='complete', year=year,
        scope=payload['scope'], background=model_input, signals=signals, source_bins=source_bins,
        inputs={k: dict(path=str(p.relative_to(ROOT)), sha256=digests[k]) for k, p in paths.items()},
        signal_input_coverage={k: represented[k] for k in sorted(represented)},
        projection_audit=audit, negative_background_audit=[a for a in clipping if a['process'] != 'signal'],
        full_workflow_complete=False, sr_data_blinded=True,
        fit_support_issues=issues, preserved_unsupported_templates=preserve_unsupported)
