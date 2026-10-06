"""One bounded input pass shared by high/low-dM and weight variations.

Sparse SR axes are not merged here. CR axes never inherit SR topology.
No event sidecars, shifted ROOT files, or campaign imports are produced.
"""
from pathlib import Path
import ast
import copy
from functools import lru_cache
import time
import importlib.metadata
import numpy as np
import awkward as ak
import uproot
from .paths import ROOT, JSONS, internal_path
from .io import read_json, write_json, sha256, validate_root
from .reader import resolved_candidates
from .event_selections import select, REGIONS, CATEGORY_ORDER, FixedJetTopologyCache
from .event_selections import _references
from .ids import clean_by_delta_r
from .normalization import dataset_label, norm_vector, data_process_allowed
from .signal_models import signal_topology, signal_mass_key
from .corrections import event_weights, topw_file_input_policy, TopWEvents
from .private_scales import topw_event_variations
from ..gnn4lowdm.inference import Rank005Numpy
from ..gnn4lowdm.region_features import feature_arrays
from .physical_histograms import fill as fill_physical

RECOIL = np.asarray([250, 300, 350, 400, 500, 650, 800, 1000, 1500], dtype=float)
SR_RECOIL = np.asarray([250, 300, 350, 400, 500, 800, 1500], dtype=float)
MLL_EDGES = np.asarray([50., 71., 81., 91., 101., 111., 160., 250., 500.])


@lru_cache(maxsize=1)
def candidate_columns():
    """Read only columns consumed by the stored-TROTA histogram calculation.

    Literal accesses come from the actual consumers; explicitly enumerated
    dynamic accesses retain optional branches without changing fallback rules.
    """
    consumers = ('utils/histogramming.py', 'utils/event_selections.py',
        'utils/region_kinematics.py', 'utils/ids.py', 'utils/weight_inputs.py',
        'utils/private_scales.py', 'utils/normalization.py',
        'utils/physical_histograms.py', 'utils/shape_kinematics.py',
        'utils/corrections.py', 'gnn4lowdm/region_features.py', 'gnn4lowdm/features.py')
    names = set()
    for relative in consumers:
        tree = ast.parse((ROOT / relative).read_text())
        names.update(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str))
    for flavor in ('electron', 'muon', 'photon', 'tau'):
        for field in ('pt', 'mass', 'eta', 'phi', 'eta_sc', 'charge', 'r9',
                      'seed_gain', 'tracker_layers', 'decay_mode', 'genpart_flavour'):
            names.add(flavor + '_' + field + '_all')
        for field in ('pt', 'mass'):
            for kind in ('nanoaod', 'corrected'):
                names.add(flavor + '_' + kind + '_' + field)
    for prefix in ('electron_veto', 'electron_medium', 'muon_loose', 'muon_medium', 'photon_medium'):
        for field in ('pt', 'eta', 'phi', 'eta_sc'):
            names.add(prefix + '_' + field)
    for field in ('candidateIndex', 'sourceFatJetIdx', 'sourceJetIdx0',
                  'sourceJetIdx1', 'sourceJetIdx2', 'QCDDiscriminant'):
        names.add('TopMixed1pct_' + field)
    for region in ('gcr', 'dy2e', 'dy2m'):
        names.update(('recoil_' + region, 'recoil_' + region + '_phi',
                      'pass_' + region + '_open_high', 'pass_' + region + '_ut_250'))
    for index in range(1, 5):
        names.update(('j%d_met_dphi' % index, 'gcr_j%d_recoil_dphi' % index))
    for field in ('pt', 'eta', 'phi', 'btag_upart', 'hadron_flavour', 'b_loose', 'b_medium'):
        names.add('good_jet_' + field)
    for direction in ('up', 'down'):
        names.update(('puppi_met_unclustered_' + direction, 'puppi_met_phi_unclustered_' + direction))
    for prefix in ('photon', 'lepton'):
        for field in ('ht', 'njet', 'nb'):
            names.add(field + '_' + prefix + '_clean')
    return frozenset(names)


def selected_columns(available):
    available = list(available)
    required = {'run', 'luminosityBlock', 'event', 'file_id', 'entry', 'dataset_id',
        'mStop', 'mLSP', 'is_data', 'is_signal', 'jet_corrected_pt', 'jet_eta_all',
        'jet_phi_all', 'met', 'met_phi'}
    if required - set(available):
        raise ValueError('Missing mandatory input columns: ' + str(sorted(required - set(available))))
    return [name for name in available if name in candidate_columns()]


def branch_plan(tree):
    available = list(tree.keys())
    selected = selected_columns(available)
    size = lambda keys: sum(int(getattr(tree[k], 'compressed_bytes', 0)) for k in keys)
    return dict(available=len(available), selected=len(selected), columns=selected,
        excluded=[k for k in available if k not in set(selected)],
        all_compressed_bytes=size(available), selected_compressed_bytes=size(selected))


def endpoint_tuple(endpoints):
    from .corrections import NONJME_ENDPOINTS
    allowed = ('nominal', *NONJME_ENDPOINTS)
    values = tuple(endpoints)
    if not values or len(values) != len(set(values)) or set(values) - set(allowed):
        raise ValueError('Empty, duplicate or unknown endpoint inventory')
    return tuple(x for x in allowed if x in values)


def split_endpoint(payload, endpoint):
    """Project one actually executed endpoint without duplicate nominal counts."""
    endpoint_tuple((endpoint,))
    executed = payload['contract'].get('execution', {}).get('endpoints')
    if executed is None:
        executed = payload['contract'].get('execution_adapter', {}).get('endpoints')
    if not executed or endpoint not in executed:
        raise ValueError('Endpoint was not executed by this file batch')
    result = copy.deepcopy(payload)
    result['contract'].update(object_task_endpoint=endpoint,
        object_endpoints=[] if endpoint == 'nominal' else [endpoint])
    for section in ('histograms', 'physical_histograms'):
        result[section] = {mode: {endpoint: variations[endpoint]}
            for mode, variations in result.get(section, {}).items() if endpoint in variations}
    for section in ('chunks', 'weights'):
        result['audit'][section] = [x for x in result['audit'][section]
            if x.get('endpoint', 'nominal') == endpoint]
    for section in ('shape_migrations', 'object_endpoints', 'object_response', 'object_acceptance'):
        values = result['audit'].get(section, {})
        result['audit'][section] = {endpoint: values[endpoint]} if endpoint != 'nominal' and endpoint in values else {}
    values = result.get('dy_rz_variations', {})
    result['dy_rz_variations'] = {endpoint: values[endpoint]} if endpoint != 'nominal' and endpoint in values else {}
    if endpoint != 'nominal':
        result['dy_rz'] = {}
        result['audit']['migrations'] = {}
    return result


def region_scores(a, selections, model, batch_size=256):
    """Evaluate each region once before splitting datasets and mass points."""
    result = {}
    for region in REGIONS:
        selected = np.asarray(selections[region]['low'], dtype=bool).copy()
        if region in ('DY2E', 'DY2M'):
            mass = np.asarray(a['mee' if region == 'DY2E' else 'mmm'], dtype=float)
            selected &= (mass > 71.) & (mass < 111.)
        score = np.zeros(len(a), dtype=float)
        if model is not None and np.any(selected):
            features = feature_arrays(a, selections[region]['block'], region, selected)
            score[selected] = model.predict(*features[:5], batch_size=batch_size)
        result[region] = score
    return result


def add(target, path, values, weight, edges):
    for key in path:
        target = target.setdefault(str(key), {})
    if len(values) != len(weight) or not np.isfinite(values).all() or not np.isfinite(weight).all():
        raise ValueError('Nonfinite or misaligned histogram input')
    coordinates = np.clip(values, edges[0], np.nextafter(float(edges[-1]), -np.inf))
    for name, weights in [('sumw', weight), ('sumw2', weight**2), ('entries', None)]:
        counts = np.histogram(coordinates, edges, weights=weights)[0]
        target[name] = (np.asarray(target.get(name, np.zeros(len(edges)-1))) + counts).tolist()
    target['edges'] = np.asarray(edges).tolist()


def add2d(target, path, x, y, weight, ex, ey):
    for key in path:
        target = target.setdefault(str(key), {})
    x = np.clip(x, ex[0], np.nextafter(float(ex[-1]), -np.inf))
    y = np.clip(y, ey[0], np.nextafter(float(ey[-1]), -np.inf))
    for name, weights in [('sumw', weight), ('sumw2', weight**2), ('entries', None)]:
        values = np.histogram2d(x, y, bins=(ex, ey), weights=weights)[0]
        target[name] = (np.asarray(target.get(name, np.zeros_like(values))) + values).tolist()
    target['edges'] = [np.asarray(ex).tolist(), np.asarray(ey).tolist()]


def signatures():
    return {str(p.relative_to(ROOT)): sha256(p) for folder in ('utils', 'gnn4lowdm')
            for p in sorted((ROOT / folder).rglob('*.py'))}


def _response_identity(dataset_id, mstop, mlsp, region):
    # Process labels aggregate several datasets, but these arrays belong to
    # one exact dataset/mass subset of the current chunk.
    return int(dataset_id), int(mstop), int(mlsp), region


def nominal_response_arrays(a, selections, model, cached_scores=None):
    """Only the central arrays needed by endpoint response diagnostics.

    An independent endpoint must not repeat central SF/histogram production.
    Selection and GNN are still evaluated, including disjoint DY mass windows.
    Nothing event-level is persisted.
    """
    result = {}
    identity = np.stack([np.asarray(a[x], dtype=int) for x in ('dataset_id', 'mStop', 'mLSP')], axis=1)
    for dataset, ms, ml in np.unique(identity, axis=0):
        chosen = np.all(identity == [dataset, ms, ml], axis=1)
        for region in REGIONS:
            sel = selections[region]
            if not np.any((sel['high'] | sel['low'])[chosen]):
                continue
            high, low = sel['high_search'][chosen], sel['low'][chosen]
            if region in ('DY2E', 'DY2M'):
                mass = np.asarray(a['mee' if region == 'DY2E' else 'mmm'][chosen], dtype=float)
                on = (mass > 71.) & (mass < 111.)
                high, low = high & on, low & on
            score = (cached_scores[region][chosen].copy() if cached_scores is not None
                     else np.zeros(int(chosen.sum()), dtype=float))
            if cached_scores is None and model is not None and np.any(low):
                selected = np.zeros(len(a), dtype=bool)
                selected[chosen] = low
                features = feature_arrays(a, sel['block'], region, selected)
                score[low] = model.predict(*features[:5], batch_size=128)
            result[_response_identity(dataset, ms, ml, region)] = dict(
                high=high, low=low, counts=sel['counts'][chosen],
                recoil=np.asarray(sel['block'].recoil[chosen]), score=score)
    return result


def build(input_path, year, output, modes=('highdm', 'lowdm'), variations=('nominal',), chunk_size=2000,
          shape_mode=None, candidate_budget=150000, object_endpoint=None, object_endpoints=None,
          reference_execution=False, support_only=False):
    if year not in (2024, 2025) or chunk_size <= 0:
        raise ValueError('Only 2024/2025 and positive chunk sizes are supported')
    if not set(modes) <= {'highdm', 'lowdm'}:
        raise ValueError('Invalid histogram mode')
    if reference_execution and shape_mode != 'stored_trota_nonjme':
        raise ValueError('Unoptimized reference execution is only for non-JME equivalence tests')
    if support_only and shape_mode != 'cms_trota_jme':
        raise ValueError('Inference support masking applies only to corrected-central/JME views')
    if object_endpoint is not None:
        from .corrections import NONJME_ENDPOINTS
        if (shape_mode != 'stored_trota_nonjme' or variations != ('nominal',) or
                object_endpoint not in ('nominal', *NONJME_ENDPOINTS)):
            raise ValueError('Independent endpoint requires one known stored-TROTA nominal-weight variation')
    if object_endpoint is not None and object_endpoints is not None:
        raise ValueError('Choose either one endpoint or a file-local endpoint group')
    if object_endpoints is not None and (shape_mode != 'stored_trota_nonjme' or variations != ('nominal',)):
        raise ValueError('Grouped endpoints require stored-TROTA nominal weights')
    endpoints = None
    if shape_mode == 'stored_trota_nonjme':
        from .corrections import NONJME_ENDPOINTS
        endpoints = endpoint_tuple(object_endpoints if object_endpoints is not None else
            (object_endpoint,) if object_endpoint is not None else ('nominal', *NONJME_ENDPOINTS))
    input_path = internal_path(input_path)
    output = internal_path(output)
    fingerprint = dict(size=input_path.stat().st_size, mtime_ns=input_path.stat().st_mtime_ns)
    metadata = validate_root(input_path, year)
    for asset in read_json(JSONS / 'assets.json')['files']:
        if sha256(ROOT / asset['target']) != asset['sha256']:
            raise ValueError('Changed immutable asset: ' + asset['target'])
    configuration = read_json(ROOT / 'gnn4lowdm/config.json')
    normalization_path = ROOT / 'estimations' / ('normalization_%s.json.gz' % year)
    normalization = read_json(normalization_path)
    contract = dict(input=str(input_path), fingerprint=fingerprint, year=year,
                    code=signatures(), normalization_sha256=sha256(normalization_path),
                    assets_sha256=sha256(JSONS / 'assets.json'), modes=list(modes), variations=list(variations),
                    configuration_sha256=sha256(ROOT / 'gnn4lowdm/config.json'),
                    sr_binning_sha256=sha256(JSONS / 'highdm_sr_binning.json'))
    if shape_mode is not None:
        if shape_mode not in ('cms_trota_jme', 'stored_trota_nonjme'):
            raise ValueError('Unknown shape implementation')
        contract['shape_mode'] = shape_mode
        if shape_mode == 'cms_trota_jme':
            contract.update(trota_models_sha256=sha256(ROOT/'models/TROTA/manifest.json'),
                jme_payloads_sha256=sha256(ROOT/'scales/JME/manifest.json'))
            if support_only:
                contract['execution'] = dict(inference_support='all_region_pretopology_union',
                    physics_definitions_changed=False)
        else:
            from .corrections import nonjme_payload_contract, NONJME_ENDPOINTS
            contract['object_payloads'] = nonjme_payload_contract(year)
            contract['object_endpoints'] = list(NONJME_ENDPOINTS)
            if object_endpoint is not None:
                contract['object_task_endpoint'] = object_endpoint
                contract['object_endpoints'] = [] if object_endpoint == 'nominal' else [object_endpoint]
            contract['object_endpoints'] = [e for e in endpoints if e != 'nominal']
            contract['execution'] = dict(schema='trotasr_file_local_v1', endpoints=list(endpoints),
                chunk_size=chunk_size, gnn_batch_size=128 if reference_execution else 256,
                branch_selection='all_reference' if reference_execution else 'consumer_columns',
                physics_definitions_changed=False)
    if Path(output).exists():
        previous = read_json(output)
        if previous.get('status') == 'complete_one_file_test' and previous.get('contract') == contract:
            return previous
        raise ValueError('Existing output has a different contract; select a separate test output')
    model = Rank005Numpy(ROOT / 'gnn4lowdm/diagonal_v3_numpy.npz') if 'lowdm' in modes else None
    start = time.monotonic()
    payload = dict(schema='trotasr_histograms_v1', status='running', contract=contract,
                   category_order=list(CATEGORY_ORDER), input_metadata=metadata.get('datasets', {}),
                   bins_merged=False, sr_data_blinded=True, scope='one_file_validation_not_full_campaign',
                   highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                   mixed_sf='not_provided', overlap_policy='B>M>R; independent W; test only',
                   histograms={}, physical_histograms={}, dy_rz={}, audit={'chunks': [], 'weights': [], 'events_read': 0, 'migrations': {}})
    with uproot.open(input_path, object_cache=None, array_cache=None) as f:
        resolved = resolved_candidates(f) if shape_mode != 'cms_trota_jme' else None
        policy = topw_file_input_policy(f)
        reader = TopWEvents(f, policy)
        # Events is an already reduced intermediate ntuple, not NanoAOD. Only
        # existing columns are read; all inference input and truth stay in ROOT.
        branches = f['Events'].keys()
        if shape_mode == 'stored_trota_nonjme' and not reference_execution:
            payload['branch_read'] = branch_plan(f['Events'])
            branches = payload['branch_read']['columns']
        source_chunks = reader.iterate(branches, step_size=chunk_size)
        if shape_mode == 'stored_trota_nonjme':
            from .corrections import nonjme_event_views
            payload['audit']['object_endpoints'] = {}
            payload['dy_rz_variations'] = {}
            if endpoints == ('nominal',):
                views = ((chunk, resolved, 'nominal') for chunk in source_chunks)
            else:
                selected_endpoints = tuple(e for e in endpoints if e != 'nominal')
                views = nonjme_event_views(source_chunks, year, resolved, payload['audit']['object_endpoints'], selected_endpoints)
        elif shape_mode:
            from .systematic_views import event_views
            from .trota_inference import load_models
            payload['audit']['trota_inference'] = {}
            payload['dy_rz_variations'] = {}
            views = event_views(source_chunks, year, load_models(year), payload['audit']['trota_inference'],
                                candidate_budget, support_only=support_only)
        else:
            views = ((chunk, resolved, 'nominal') for chunk in source_chunks)
        for a, resolved, endpoint in views:
            n = len(a)
            if endpoint == 'nominal':
                payload['audit']['events_read'] += n
                nominal_response = {}
                topology_cache = (FixedJetTopologyCache(a, resolved)
                                  if shape_mode == 'stored_trota_nonjme' else None)
            # One selection/object pass also retains the disjoint off-Z
            # measurement sample. The displayed DY CR remains on-Z only.
            selections, audit = select(a, resolved, dy_mass_window=None, topology_cache=topology_cache)
            cached_scores = region_scores(a, selections, model) if shape_mode == 'stored_trota_nonjme' and not reference_execution else None
            if endpoints is not None and 'nominal' not in endpoints and endpoint == 'nominal':
                if np.any(np.asarray(a['is_data'], bool)):
                    raise ValueError('Independent object variations require MC input')
                nominal_selections = selections
                nominal_response = nominal_response_arrays(a, selections, model, cached_scores)
                continue
            audit['endpoint'] = endpoint
            payload['audit']['chunks'].append(audit)
            if shape_mode == 'stored_trota_nonjme':
                if endpoint == 'nominal':
                    nominal_selections = selections
                elif not np.any(np.asarray(a['is_data'],bool)):
                    for region in REGIONS:
                        now, old = selections[region], nominal_selections[region]
                        target = payload['audit'].setdefault('object_acceptance', {}).setdefault(endpoint, {}).setdefault(region, {})
                        for name in ('high_search','low'):
                            before, after = old[name], now[name]
                            if region in ('DY2E','DY2M'):
                                # Selection audit is explicitly before the mll window,
                                # matching select(..., dy_mass_window=None).
                                pass
                            for label,value in (('gained',after & ~before),('lost',before & ~after)):
                                key=name+'_'+label
                                target[key]=target.get(key,0)+int(np.sum(value))
            identity = np.stack([np.asarray(a[x], dtype=int) for x in ('dataset_id', 'mStop', 'mLSP')], axis=1)
            for dataset_id, ms, ml in np.unique(identity, axis=0):
                chosen = np.all(identity == [dataset_id, ms, ml], axis=1)
                dataset, process, is_data, is_signal = dataset_label(metadata, int(dataset_id))
                if dataset == 'unknown':
                    raise ValueError('Unknown dataset ID')
                sub = {k: a[k][chosen] for k in ak.fields(a)}
                norm = norm_vector(normalization, sub, int(dataset_id), dataset, is_data, is_signal,
                                   require_normalization=True)
                masks = {r: (selections[r]['high'] | selections[r]['low'])[chosen] for r in REGIONS}
                if is_data:
                    base = {'nominal': np.ones(int(chosen.sum()))}; wa = {'data': True}
                else:
                    base, wa = event_weights(sub, dataset, process, year, masks, policy, apply_topw=False,
                                             include_variations=(endpoint == 'nominal'))
                record = dict(dataset=dataset, process=process, audit=wa, topw={}, endpoint=endpoint)
                payload['audit']['weights'].append(record)
                sample = 'data' if is_data else (signal_mass_key(signal_topology(dataset), int(ms), int(ml)) if is_signal else process)
                for region in REGIONS:
                    if is_data and (region == 'SR' or not data_process_allowed(process, region)):
                        continue
                    sel = selections[region]
                    if not np.any((sel['high'] | sel['low'])[chosen]):
                        continue
                    if is_data:
                        tw = {'nominal': np.ones(int(chosen.sum()))}
                        old_tw = tw
                    else:
                        tw, record['topw'][region] = topw_event_variations(ROOT, str(year), dataset, process, sub, policy,
                            cleaned=sel['sf_clean'][chosen], include_variations=(endpoint == 'nominal'))
                        eta, phi = _references(a, region)
                        old_clean = clean_by_delta_r(a['fatjet_eta_all'], a['fatjet_phi_all'], eta, phi, .4)[chosen]
                        old_tw, _ = topw_event_variations(ROOT, str(year), dataset, process, sub, policy,
                                                        cleaned=old_clean, include_variations=False)
                    allowed = set(base) | set(tw)
                    requested = variations if endpoint == 'nominal' else ('nominal',)
                    names = sorted(allowed) if requested == ('all_weights',) else requested
                    if set(names) - allowed:
                        raise ValueError('Requested unevaluated variations: ' + str(set(names) - allowed))
                    b = sel['block']
                    nb = np.where(b.nb[chosen] == 1, 1, 2)
                    counts = sel['counts'][chosen]
                    high = sel['high_search'][chosen]
                    low = sel['low'][chosen]
                    before_high = sel['baseline_search'][chosen]
                    before_low = sel['baseline_low'][chosen]
                    if region in ('DY2E', 'DY2M'):
                        mass = np.asarray(sub['mee' if region == 'DY2E' else 'mmm'], dtype=float)
                        on = (mass > 71.) & (mass < 111.)
                        off = ((mass > 50.) & (mass < 71.)) | (mass > 111.)
                        nominal_weight = base['nominal'] * tw['nominal'] * norm
                        component = ('data' if is_data else 'zll' if process == 'DY' or
                                     any(t in dataset for t in ('TTZ', 'WZ', 'ZZ', 'WWZ', 'WZZ', 'ZZZ', 'WZG'))
                                     else 'other')
                        if not is_signal:
                            dy_target = (payload['dy_rz'] if endpoint == 'nominal' else
                                         payload['dy_rz_variations'].setdefault(endpoint, {}))
                            for mode, acceptance in (('highdm', high), ('lowdm', low)):
                                if mode not in modes:
                                    continue
                                for group, group_mask in (('Nb1', nb == 1), ('Nb2plus', nb == 2)):
                                    mask = acceptance & group_mask & (mass > 50.)
                                    add(dy_target, (mode, 'mll', region, group, component),
                                        mass[mask], nominal_weight[mask], MLL_EDGES)
                                    for window, window_mask in (('on', on), ('off', off)):
                                        mask = acceptance & group_mask & window_mask
                                        add(dy_target, (mode, 'yields', region, group, window, component),
                                            np.full(int(mask.sum()), .5), nominal_weight[mask], np.array([0., 1.]))
                        high, low = high & on, low & on
                        before_high, before_low = before_high & on, before_low & on
                    migration_store = (payload['audit']['migrations'] if endpoint == 'nominal' else
                        payload['audit'].setdefault('shape_migrations', {}).setdefault(endpoint, {}))
                    migration = migration_store.setdefault(sample, {}).setdefault(region, {})
                    for regime, before, after in [('highdm_search', before_high, high),
                                                   ('lowdm', before_low, low)]:
                        item = migration.setdefault(regime, {})
                        old_w = base['nominal'] * old_tw['nominal'] * norm
                        new_w = base['nominal'] * tw['nominal'] * norm
                        for name, value in [('before_entries', int(before.sum())), ('after_entries', int(after.sum())),
                                            ('gained', int((after & ~before).sum())), ('lost', int((before & ~after).sum())),
                                            ('before_sumw', float(old_w[before].sum())), ('after_sumw', float(new_w[after].sum())),
                                            ('after_with_baseline_weight', float(old_w[after].sum()))]:
                            item[name] = item.get(name, 0) + value
                    recoil = np.asarray(b.recoil[chosen])
                    score = (cached_scores[region][chosen].copy() if cached_scores is not None
                             else np.zeros(int(chosen.sum()), dtype=float))
                    if cached_scores is None and model is not None and np.any(low):
                        selected_low = np.zeros(n, dtype=bool)
                        selected_low[chosen] = low
                        features = feature_arrays(a, b, region, selected_low)
                        score[low] = model.predict(*features[:5], batch_size=128)
                    if shape_mode == 'stored_trota_nonjme' and not is_data:
                        key = _response_identity(dataset_id, ms, ml, region)
                        current = dict(high=high, low=low, counts=counts, recoil=recoil, score=score)
                        if endpoint == 'nominal':
                            nominal_response[key] = {k: v.copy() for k,v in current.items()}
                        else:
                            before = nominal_response.get(key)
                            if before is None:
                                before = dict(high=np.zeros_like(high),low=np.zeros_like(low),
                                    counts=np.zeros_like(counts),recoil=np.zeros_like(recoil),score=np.zeros_like(score))
                            store = payload['audit'].setdefault('object_response', {}).setdefault(endpoint, {}).setdefault(region, {})
                            values = dict(selected_recoil_changed=int(np.sum((high|low|before['high']|before['low']) & (recoil!=before['recoil']))),
                                low_score_changed=int(np.sum((low&before['low']) & (score!=before['score']))),
                                topology_changed=int(np.sum(np.any(counts!=before['counts'],axis=1))))
                            for k,v in values.items():store[k]=store.get(k,0)+v
                    for v in names:
                        weights = base.get(v, base['nominal']) * tw.get(v, tw['nominal']) * norm
                        v = v if endpoint == 'nominal' else endpoint
                        if not is_signal:
                            for mode, acceptance in (('highdm',high),('lowdm',low)):
                                if mode in modes:
                                    fill_physical(payload['physical_histograms'], mode, v, region, sample,
                                                  sub, b, chosen, counts, acceptance, weights)
                        if 'highdm' in modes and np.any(high):
                            if region == 'SR':
                                categories = np.column_stack((np.asarray(b.nb[chosen], dtype=int), counts))
                                for cat in np.unique(categories[high], axis=0):
                                    mask = high & np.all(categories == cat, axis=1)
                                    label = ','.join(map(str, cat))
                                    add(payload['histograms'], ('highdm', v, region, label, sample, 'recoil'),
                                        recoil[mask], weights[mask], RECOIL)
                            else:
                                for group in (1, 2):
                                    mask = high & (nb == group)
                                    add(payload['histograms'], ('highdm', v, region, 'Nb' + str(group), sample, 'recoil'),
                                        recoil[mask], weights[mask], RECOIL)
                        if 'lowdm' in modes and np.any(low):
                            if region == 'SR':
                                nisr = np.minimum(np.asarray(b.nisr[chosen], dtype=int), 2)
                                categories = [('Nb%s_NISR%s' % ('1' if g == 1 else '2plus', str(i) if i < 2 else '2plus'),
                                               (nb == g) & (nisr == i)) for g in (1, 2) for i in (0, 1, 2)]
                            else:
                                categories = [('Nb' + str(g), nb == g) for g in (1, 2)]
                            for cat, mask in categories:
                                mask = mask & low
                                edges = np.asarray(configuration['sr_binning']['edges_by_category'][cat] if region == 'SR'
                                                   else configuration['cr_binning']['score_edges'])
                                path = ('lowdm', v, region, cat, sample)
                                add(payload['histograms'], (*path, 'gnn_score'), score[mask], weights[mask], edges)
                                add(payload['histograms'], (*path, 'recoil'), recoil[mask], weights[mask], RECOIL)
                                add2d(payload['histograms'], (*path, 'gnn_recoil'), score[mask], recoil[mask], weights[mask], edges, RECOIL)
            # Never export even raw SR event counts for data.
            if np.any(np.asarray(a['is_data'], dtype=bool)):
                audit['SR'] = {'blinded': True}
    final_fingerprint = dict(size=input_path.stat().st_size, mtime_ns=input_path.stat().st_mtime_ns)
    if final_fingerprint != fingerprint:
        raise ValueError('Input ROOT changed during the read; output not committed')
    if payload['audit']['events_read'] != reader.num_entries:
        raise ValueError('Incomplete Events traversal')
    payload.update(status='complete_one_file_test', seconds=time.monotonic()-start,
                   versions={x: importlib.metadata.version(x) for x in ('numpy', 'awkward', 'uproot', 'coffea', 'correctionlib')})
    write_json(output, payload)
    return payload
