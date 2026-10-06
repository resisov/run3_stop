"""TROTASR histograms -> native-card inputs, without reading event files.

SR display merging never merges Nb/recoil normalization components. The low-dM
CR is already NISR-pooled; Sgamma uses the actual GNN x recoil cells. Measured
background factors are mandatory arguments, never looked up in another campaign.
"""
from collections import OrderedDict, defaultdict
import math
import numpy as np
from . import statistics_model as model
from .io import read_json
from .paths import ROOT
from .signal_models import signal_mass_key, SUPPORTED_SIGNAL_TOPOLOGIES
from .sr_binning import AXES, NATIVE_EDGES, expand, matches, label
from .renderers.background_process_groups import BACKGROUND_GROUP_SPECS

FAMILIES = {p: raw for p, _, raw in BACKGROUND_GROUP_SPECS if raw}
REGIONS = ('SR', 'LLCR', 'QCDCR', 'GCR')
NB_GROUPS = ('Nb1', 'Nb2plus')
UT_GROUPS = ([0], [1], [2], [3], [4, 5, 6, 7])
UT_BOUNDS = (250., 300., 350., 400., 500., float('inf'))
SHAPE_PREFIXES = ('jes', 'jer', 'metUnclustered', 'metuncl', 'electronScale',
                  'electronSmear', 'muonScale', 'muonResolution', 'photonScale',
                  'photonSmear', 'tauEnergyScale')


def combined_shape_endpoints(payload):
    """Validate the explicitly approved absolute-endpoint union, not a reanchor."""
    contract = payload.get('contract', {})
    policy = dict(id='jme_with_stored_nonjme_as_is', user_approved=True,
        central='corrected_central_jme', nonjme_endpoints='stored_absolute_histograms',
        nominal_mismatch_acknowledged=True, reanchor=False, recompute_events=False)
    nonjme = {n+d for n in ('electronScale', 'electronSmear', 'photonScale',
        'photonSmear', 'muonScale', 'muonResolution', 'tauEnergyScale', 'metUnclustered')
        for d in ('Up', 'Down')}
    provenance = payload.get('combination_provenance', {})
    components = contract.get('component_contracts', {})
    if (payload.get('scope') != 'full_combined_systematic_production'
            or payload.get('status') != 'complete'
            or payload.get('preliminary') is not True
            or payload.get('central_conventions_consistent') is not False
            or contract.get('shape_mode') != 'jme_with_stored_nonjme'
            or contract.get('combination_policy') != policy
            or provenance.get('policy') != policy
            or set(provenance.get('sources', {})) != {'jme', 'nonjme'}
            or set(components) != {'jme', 'nonjme'}
            or components['jme'].get('shape_mode') != 'cms_trota_jme'
            or components['nonjme'].get('shape_mode') != 'stored_trota_nonjme'
            or set(contract.get('object_endpoints', [])) != nonjme):
        raise ValueError('Unverified or unapproved combined histogram contract')
    for coverage in (payload.get('coverage', {}),
                     payload.get('audit', {}).get('nonjme_source_coverage', {})):
        if (coverage.get('full_input_inventory') is not True
                or coverage.get('failed_files') != 0
                or not coverage.get('expected_files')
                or coverage.get('completed_files') != coverage['expected_files']):
            raise ValueError('Incomplete combined histogram source coverage')
    for source in provenance['sources'].values():
        if not all(isinstance(source.get(k), str) and len(source[k]) == 64
                   and all(c in '0123456789abcdef' for c in source[k])
                   for k in ('sha256', 'summary_sha256')):
            raise ValueError('Missing combined histogram source identity')
    return nonjme | {n+d for n in ('jesTotal', 'jer') for d in ('Up', 'Down')}


class HistogramView:
    def __init__(self, payload, year, gnn):
        self.histograms = payload['histograms']
        self.year, self.gnn = str(year), gnn
        self.variations = {}
        self.shape_endpoints = set()
        shape_mode = payload.get('contract', {}).get('shape_mode')
        if shape_mode == 'cms_trota_jme':
            self.shape_endpoints = {n+d for n in ('jesTotal', 'jer') for d in ('Up', 'Down')}
        elif shape_mode == 'stored_trota_nonjme':
            self.shape_endpoints = set(payload['contract']['object_endpoints'])
        elif shape_mode == 'jme_with_stored_nonjme':
            self.shape_endpoints = combined_shape_endpoints(payload)
        allowed = {s for sources in FAMILIES.values() for s in sources} | {'data'}
        for mode in ('highdm', 'lowdm'):
            variations = self.histograms.get(mode, {})
            if 'nominal' not in variations:
                raise ValueError('Missing nominal histogram mode: ' + mode)
            names = set(variations) - {'nominal'}
            bases = {n[:-2] if n.endswith('Up') else n[:-4] if n.endswith('Down') else n for n in names}
            if names != {n + d for n in bases for d in ('Up', 'Down')}:
                raise ValueError('Unpaired systematic endpoints in ' + mode)
            shapes = {n for n in names if n.startswith(SHAPE_PREFIXES)}
            if shapes != self.shape_endpoints:
                raise ValueError('Object-shape endpoints differ from the explicit histogram contract')
            self.variations[mode] = {n: model.nps_nuisance_name(n, self.year) for n in sorted(bases)}
            for regions in variations.values():
                for region, categories in regions.items():
                    if region in REGIONS and (mode == 'lowdm' or region != 'SR'):
                        expected_categories = (set(gnn['sr_binning']['category_labels'])
                                               if mode == 'lowdm' and region == 'SR' else {'Nb1', 'Nb2'})
                        if set(categories) - expected_categories:
                            raise ValueError('Unknown category outside the fixed layout: ' + mode + '/' + region)
                    for samples in categories.values():
                        if region == 'SR' and 'data' in samples:
                            raise ValueError('SR observations are forbidden')
                        for sample in samples:
                            if sample not in allowed and not sample.startswith(('mStop', 'T2tt_', 'T2bW_', 'T2tb_')):
                                raise ValueError('Unclassified sample: ' + sample)
            for variation, regions in variations.items():
                if variation == 'nominal' or variation in self.shape_endpoints:
                    continue
                for region, categories in regions.items():
                    for category, samples in categories.items():
                        nominal = variations['nominal'].get(region, {}).get(category, {})
                        if set(samples) - set(nominal):
                            raise ValueError('Weight-only endpoint has non-nominal event cells')

    def _cell(self, mode, region, category, sample, variation, index, ut=None):
        nominal = self.histograms[mode]['nominal'].get(region, {}).get(category, {}).get(sample)
        varied = self.histograms[mode].get(variation, {}).get(region, {}).get(category, {}).get(sample)
        if variation in self.shape_endpoints and sample != 'data':
            # Shape histograms are complete sparse endpoint populations, not
            # patches to nominal. Missing cells are genuine zero acceptance.
            if varied is None:
                return 0., 0.
            record = varied
        elif nominal is None:
            if varied is not None:
                raise ValueError('Weight-only endpoint has a non-nominal event cell')
            return 0., 0.
        else:
            record = nominal if varied is None else varied
        observable = 'recoil' if mode == 'highdm' else 'gnn_score' if ut is None else 'gnn_recoil'
        leaf = record[observable]
        score_edges = (self.gnn['sr_binning']['edges_by_category'][category] if region == 'SR'
                       else self.gnn['cr_binning']['score_edges']) if mode == 'lowdm' else None
        edges = NATIVE_EDGES if mode == 'highdm' else score_edges if ut is None else [score_edges, NATIVE_EDGES]
        if leaf['edges'] != edges:
            raise ValueError('Changed histogram edges: ' + mode + '/' + region + '/' + category)
        shape = (8,) if mode == 'highdm' else (5,) if ut is None else (5, 8)
        arrays = [np.asarray(leaf[f], dtype=float) for f in ('sumw', 'sumw2', 'entries')]
        if any(a.shape != shape or not np.isfinite(a).all() for a in arrays) or any((a < 0).any() for a in arrays[1:]):
            raise ValueError('Malformed signed histogram or variance')
        if ut is not None:
            marginal = record['gnn_score']
            if marginal['edges'] != score_edges:
                raise ValueError('GNN marginal edges differ from GNN x recoil')
            for field, a in zip(('sumw', 'sumw2', 'entries'), arrays):
                expected = np.asarray(marginal[field], dtype=float)
                if expected.shape != (5,) or not np.allclose(a.sum(axis=1), expected, rtol=1e-10, atol=1e-10):
                    raise ValueError('GNN x recoil does not conserve its GNN marginal: ' + field)
        return tuple(float(a[index]) if ut is None else float(a[index, ut].sum()) for a in arrays[:2])

    def record(self, mode, region, cells, family, ut=None):
        samples = FAMILIES.get(family, (family,))
        def total(variation):
            pairs = [self._cell(mode, region, cat, sample, variation, i, ut)
                     for cat, i in cells for sample in samples]
            return (sum(p[0] for p in pairs), sum(p[1] for p in pairs))
        value, variance = total('nominal')
        variations = {}
        for base, name in self.variations[mode].items():
            up, down = total(base + 'Up')[0], total(base + 'Down')[0]
            supplied = any(sample in self.histograms[mode].get(base+d, {}).get(region, {}).get(cat, {})
                           for d in ('Up', 'Down') for cat, _ in cells for sample in samples)
            if supplied or not (math.isclose(up, value, rel_tol=1e-12, abs_tol=1e-15)
                                and math.isclose(down, value, rel_tol=1e-12, abs_tol=1e-15)):
                variations[name] = {'up': np.asarray([up]), 'down': np.asarray([down])}
        if value == 0 and variance == 0 and not variations:
            return None
        return dict(nominal=np.asarray([value]), sumw2=np.asarray([variance]), variations=variations)

    def observation(self, mode, region, cells):
        if region == 'SR':
            return None
        value = sum(self._cell(mode, region, c, 'data', 'nominal', i)[0] for c, i in cells)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid CR observation')
        return value


def _channel(name, mode, region, group=None):
    return dict(name=name, regime=mode, region=region, nb_group=group,
                control_group=group, observation=None, backgrounds={}, rate_params={},
                rate_initial={}, extra_lnN={}, physical={})


def _attach(channel, component, record, year, kind=None, group=None, unit=None, scale=1., initial=1., extras=()):
    if record is None:
        return
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(initial) or initial <= 0:
        raise ValueError('Invalid measured scale or parameter initial value')
    channel['backgrounds'][component] = model.scaled_record(record, scale)
    if kind is not None:
        channel['rate_params'][component] = model.rate_parameter(kind, channel['regime'], group, unit, str(year))
        channel['rate_initial'][component] = initial
    model.add_extra(channel, component, list(extras))


def _gamma(sgamma, mode, group, shape_bin):
    payload = sgamma['highdm' if mode == 'highdm' else 'lowdm_families'][group]
    if len(payload['bins']) != 5:
        raise ValueError('Expected the adopted five recoil Sgamma measurement bins')
    return (model.require_positive(payload['Q'], 'Qgamma/' + mode + '/' + group),
            model.require_positive(payload['bins'][shape_bin]['Sgamma'], 'Sgamma/' + mode + '/' + group))


def _extras(rz, double_ratio, mode, group, shape_bin, year, interval=None):
    result = model.rz_nuisances(rz, mode + '_' + group)
    bounds = UT_BOUNDS[shape_bin:shape_bin+2] if interval is None else interval
    name, delta, sources = model.closure_record(double_ratio, mode, *bounds, str(year))
    if not sources:
        raise ValueError('Missing Z/gamma closure measurement for a supported Z contribution')
    if delta > 0:
        result.append(dict(name=name, down=1. / (1. + delta), up=1. + delta))
    return result


def _signal(channel, record):
    if record is not None:
        channel['physical']['signal'] = dict(nominal=float(record['nominal'][0]), sumw2=float(record['sumw2'][0]),
            variations={n: {d: float(v[0]) for d, v in pair.items()} for n, pair in record['variations'].items()})


def build_inputs(payload, sgamma, rz_high, rz_low, double_ratio, topology, mstop, mlsp,
                 definition=None, gnn=None):
    """Use measured factors supplied by the caller; never read old fit results."""
    if payload.get('status') not in ('complete', 'complete_one_file_test') or not payload.get('sr_data_blinded'):
        raise ValueError('Require completed, SR-blinded TROTASR histograms')
    if payload.get('highdm_sr_source_axes') != 'exact_nb_exact_topology_native_recoil_v2':
        raise ValueError('Exact Nb/native recoil SR components are required')
    year = str(payload['contract']['year'])
    if year not in ('2024', '2025') or topology not in SUPPORTED_SIGNAL_TOPOLOGIES:
        raise ValueError('Unsupported year/topology')
    for name, factor in [('sgamma', sgamma), ('rz_high', rz_high), ('rz_low', rz_low), ('double_ratio', double_ratio)]:
        allowed = ('complete', 'feature_stage_complete') if name == 'rz_low' else ('complete',)
        if factor.get('status') not in allowed:
            raise ValueError('Incomplete measured factor: ' + name)
        if str(factor.get('provenance', {}).get('campaign_year')) != year:
            raise ValueError('Measured factor year mismatch: ' + name)
    streams = {r.get('process') for r in payload.get('input_metadata', {}).values() if r.get('is_data')}
    if not {'JetMET', 'EGamma'} <= streams:
        raise ValueError('CR data streams are missing; no pseudo-observations substituted')
    gnn = read_json(ROOT / 'gnn4lowdm/config.json') if gnn is None else gnn
    layout = expand(definition)
    view = HistogramView(payload, year, gnn)
    rz = model.build_rz_covariance(rz_high, rz_low, year)
    mass_key = signal_mass_key(topology, mstop, mlsp)
    groups, mapping = OrderedDict(), []

    def put(channel):
        groups.setdefault(channel['region'] + '_' + channel['regime'], []).append(channel)

    # The fixed CR axes never depend on which SR topology was populated.
    for region in REGIONS[1:]:
        for group, category in zip(NB_GROUPS, ('Nb1', 'Nb2')):
            for i in range(8):
                channel = _channel(f'{region}_highdm_{group}_bin{i}', 'highdm', region, group)
                cells = [(category, i)]
                channel['observation'] = view.observation('highdm', region, cells)
                for family in FAMILIES:
                    record = view.record('highdm', region, cells, family)
                    if record is None:
                        continue
                    if region == 'LLCR' and family in ('Top', 'WtoLNu'):
                        _attach(channel, family, record, year, 'll_norm', group, i)
                    elif region == 'QCDCR' and family == 'QCD':
                        _attach(channel, family, record, year, 'qcd_norm', group, i)
                    elif region == 'GCR' and family == 'PhotonJet':
                        q, s = _gamma(sgamma, 'highdm', group, min(i, 4))
                        _attach(channel, family, record, year, 'sgamma_shape', group, min(i, 4), q, s)
                    else:
                        _attach(channel, family, record, year)
                put(channel)

    exact_keys = {key for endpoint in payload['histograms']['highdm'].values()
                  for key in endpoint.get('SR', {})}
    by_layout = {c['key']: [] for c in layout}
    for key in sorted(exact_keys):
        counts = tuple(map(int, key.split(',')))
        hits = [c for c in layout if len(counts) == 5 and all(matches(v, c['predicates'][a]) for a, v in zip(AXES, counts))]
        if len(hits) != 1:
            raise ValueError('Uncovered/overlapping SR category: ' + key)
        by_layout[hits[0]['key']].append((key, 'Nb1' if counts[0] == 1 else 'Nb2plus'))
    output_index = 0
    for category in layout:
        for low, high in zip(category['edges'][:-1], category['edges'][1:]):
            native = list(range(NATIVE_EDGES.index(low), NATIVE_EDGES.index(high)))
            channel = _channel(f'SR_highdm_bin{output_index}', 'highdm', 'SR')
            all_cells = [(key, i) for key, _ in by_layout[category['key']] for i in native]
            for group in NB_GROUPS:
                for i in native:
                    cells = [(key, i) for key, g in by_layout[category['key']] if g == group]
                    for family in ('Top', 'WtoLNu', 'QCD', 'Zto2Nu'):
                        record = view.record('highdm', 'SR', cells, family)
                        if record is None:
                            continue
                        component = f'{family}_{group}_u{i}'
                        if family == 'Zto2Nu':
                            _, s = _gamma(sgamma, 'highdm', group, min(i, 4))
                            _attach(channel, component, record, year, 'sgamma_shape', group, min(i, 4),
                                    model.rz_value(rz, 'highdm_' + group), s,
                                    _extras(rz, double_ratio, 'highdm', group, min(i, 4), year, NATIVE_EDGES[i:i+2]))
                        else:
                            kind = 'qcd_norm' if family == 'QCD' else 'll_norm'
                            _attach(channel, component, record, year, kind, group, i)
            for family in ('VV_VVV', 'DY', 'PhotonJet'):
                _attach(channel, family, view.record('highdm', 'SR', all_cells, family), year)
            _signal(channel, view.record('highdm', 'SR', all_cells, mass_key))
            mapping.append(dict(channel=channel['name'], category=category['key'], label=label(category),
                                native_recoil_bins=native, low=low, high=high, last_bin_open_ended=high == NATIVE_EDGES[-1]))
            put(channel)
            output_index += 1
    if output_index != 162:
        raise ValueError('The full SR map must contain 162 bins')

    for region in REGIONS:
        categories = gnn['sr_binning']['category_labels'] if region == 'SR' else ('Nb1', 'Nb2')
        for category in categories:
            group = 'Nb1' if category.startswith('Nb1') else 'Nb2plus'
            for i in range(5):
                channel = _channel(f'{region}_lowdm_gnn_{category}_bin{i}', 'lowdm', region, group)
                cells = [(category, i)]
                channel['observation'] = view.observation('lowdm', region, cells)
                for family in FAMILIES:
                    split = (region == 'SR' and family == 'Zto2Nu') or (region == 'GCR' and family == 'PhotonJet')
                    for shape_bin in range(5) if split else (None,):
                        record = view.record('lowdm', region, cells, family, UT_GROUPS[shape_bin] if split else None)
                        if record is None:
                            continue
                        component = f'{family}_{group}_u{shape_bin}' if split else family
                        if split:
                            q, s = _gamma(sgamma, 'lowdm', group, shape_bin)
                            scale = model.rz_value(rz, 'lowdm_' + group) if region == 'SR' else q
                            extras = _extras(rz, double_ratio, 'lowdm', group, shape_bin, year) if region == 'SR' else ()
                            _attach(channel, component, record, year, 'sgamma_shape', group, shape_bin, scale, s, extras)
                        elif region in ('SR', 'LLCR') and family in ('Top', 'WtoLNu'):
                            _attach(channel, component, record, year, 'll_norm', group, 'inclusive')
                        elif region in ('SR', 'QCDCR') and family == 'QCD':
                            _attach(channel, component, record, year, 'qcd_norm', group, 'inclusive')
                        else:
                            _attach(channel, component, record, year)
                if region == 'SR':
                    _signal(channel, view.record('lowdm', region, cells, mass_key))
                put(channel)

    scopes, initials = defaultdict(set), {}
    for bins in groups.values():
        for channel in bins:
            for process, parameter in channel['rate_params'].items():
                scopes[parameter].add('sr' if channel['region'] == 'SR' else 'cr')
                initial = channel['rate_initial'][process]
                if not math.isclose(initials.setdefault(parameter, initial), initial, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError('Inconsistent shared parameter initial value')
    # Retain the adopted CR-only unity policy with an explicit audit. Never fix
    # an SR-only parameter silently, and never merge/drop analysis bins for it.
    cr_only = {p for p, s in scopes.items() if s == {'cr'}}
    fixed = []
    for bins in groups.values():
        for channel in bins:
            for process, parameter in list(channel['rate_params'].items()):
                if parameter in cr_only:
                    fixed.append(dict(channel=channel['name'], process=process, parameter=parameter,
                                      previous_initial=channel['rate_initial'][process], fixed=1.))
                    del channel['rate_params'][process]
                    del channel['rate_initial'][process]
    issues = [dict(problem='sr_only_parameter', parameter=p) for p, s in sorted(scopes.items()) if s == {'sr'}]
    issues.extend(dict(problem='data_without_background', channel=c['name']) for bins in groups.values()
                  for c in bins if c['region'] != 'SR' and c['observation'] > 0 and not c['backgrounds'])
    counts = {name: len(bins) for name, bins in groups.items()}
    expected = {'SR_highdm': 162, 'SR_lowdm': 30,
                **{r + '_highdm': 16 for r in REGIONS[1:]}, **{r + '_lowdm': 10 for r in REGIONS[1:]}}
    if counts != expected:
        raise ValueError('Changed analysis-bin layout')
    return dict(schema='trotasr_statistics_inputs_v1', status='complete' if not issues else 'blocked',
                year=year, scope=payload.get('scope'), topology=topology, mStop=int(mstop), mLSP=int(mlsp),
                groups=groups, bin_counts=counts, highdm_mapping=mapping, issues=issues,
                cr_only_unity_audit=fixed, rz_covariance=rz, sr_data_blinded=True,
                signal_in_control_regions=False, mixed_sf='not_provided', full_workflow_complete=False)
