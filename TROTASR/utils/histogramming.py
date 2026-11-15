"""Fill high-/low-dM histograms and their weight and object variations.

Keep sparse SR axes and each control region's own category definition.
"""
from pathlib import Path
import ast
import copy
from functools import lru_cache
from contextlib import ExitStack
import time
import importlib.metadata
import numpy as np
import awkward as ak
import uproot
from .paths import ROOT, JSONS, internal_path, era_year, analysis_era, EARLY_ERAS
from .io import read_json, write_json, validate_root
from .reader import resolved_candidates
from .event_selections import select, REGIONS, CATEGORY_ORDER, FixedJetTopologyCache
from .event_selections import _references
from .region_kinematics import update_veto_leptons
from .ids import clean_by_delta_r
from .normalization import dataset_label, norm_vector, data_process_allowed
from .signal_models import signal_topology, signal_mass_key
from .corrections import event_weights, topw_file_input_policy, TopWEvents
from .private_scales import topw_event_variations
from ..gnn4lowdm.inference import Rank005Numpy
from ..gnn4lowdm.region_features import feature_arrays
from ..gnn4lowdm.configuration import configuration_path, portable_model
from .physical_histograms import fill as fill_physical

RECOIL = np.asarray([250, 300, 350, 400, 500, 650, 800, 1000, 1500], dtype=float)
SR_RECOIL = np.asarray([250, 300, 350, 400, 500, 800, 1500], dtype=float)
MLL_EDGES = np.asarray([50., 71., 81., 91., 101., 111., 160., 250., 500.])


def weight_variations_needed(endpoint, variations):
    """Evaluate weight shifts only on the nominal object collection."""
    return endpoint == 'nominal' and tuple(variations) != ('nominal',)


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
    names.add('jet_btag_pnet_all')
    for field in ('pt', 'eta', 'phi', 'btag_upart', 'btag_pnet', 'hadron_flavour', 'b_loose', 'b_medium'):
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


def region_scores(a, selections, model, batch_size=256, year=2024):
    """Evaluate each region once before splitting datasets and mass points."""
    result = {}
    for region in REGIONS:
        selected = np.asarray(selections[region]['low'], dtype=bool).copy()
        if region in ('DY2E', 'DY2M'):
            mass = np.asarray(a['mee' if region == 'DY2E' else 'mmm'], dtype=float)
            selected &= (mass > 71.) & (mass < 111.)
        score = np.zeros(len(a), dtype=float)
        if model is not None and np.any(selected):
            features = feature_arrays(a, selections[region]['block'], region, selected, year=year)
            score[selected] = model.predict(*features[:5], batch_size=batch_size)
        result[region] = score
    return result


def _signal_sr_selection(a, resolved, year):
    """Project only the canonical SR branch, never evaluate a CR selector.

    The SR branch uses the shared SR kinematic/topology primitives. Its veto
    bookkeeping, tag arbitration and high-dM mTb bin mask match `select`.
    """
    from .event_selections import highdm_core, topology_counts
    from .paths import tagging_definition
    from .region_kinematics import RegionBlock, as_ak_bool, as_bool, as_float, fatjet_kinematics, jet_kinematics

    audit = update_veto_leptons(a, year=year)
    pt, eta, phi = (a[k] for k in ('jet_corrected_pt', 'jet_eta_all', 'jet_phi_all'))
    btag = a[tagging_definition(year)['btag_branch']]
    good = (pt > 30.) & (abs(eta) < 2.4) & as_ak_bool(a['jet_id_all'])
    met, met_phi = as_float(a, 'met'), as_float(a, 'met_phi')
    jets = jet_kinematics(pt, eta, phi, btag, good, met, met_phi, year)
    empty = a['photon_pt_all'][:, :0]
    nisr, nt, nw = fatjet_kinematics(a, empty, empty)
    core = np.logical_and.reduce([as_bool(a, name) for name in (
        'pass_base_common', 'pass_zero_tau', 'pass_signal_trigger', 'pass_no_veto_leptons',
        'pass_met_250', 'pass_ht_300')])
    core &= (jets['njet'] >= 2) & jets['open_pre']
    block = RegionBlock(core=core, recoil=met, recoil_phi=met_phi, ht=jets['ht'], njet=jets['njet'],
        nb=jets['nb'], nt=nt, nw=nw, nisr=nisr, met_sqrt_ht=jets['met_sqrt_ht'], mtb=jets['mtb'], ptb=jets['ptb'])
    counts, clean, topology, baseline = topology_counts(a, resolved, 'SR', year=year)
    high_core = highdm_core(a, {'SR': block}, 'SR', year=year)
    search = np.isfinite(block.mtb) & (block.mtb >= 175.)
    low_core = block.core & (block.nb >= 1)
    ntag, nold = counts.sum(axis=1), baseline.sum(axis=1)
    high, old_high = high_core & (ntag >= 1), high_core & (nold >= 1)
    low, old_low = low_core & (ntag == 0), low_core & (nold == 0)
    if np.any(high & low):
        raise ValueError('High/low overlap: SR')
    high_search, old_search = high & search, old_high & search
    result = dict(high=high, high_search=high_search, low=low, counts=counts, block=block, sf_clean=clean,
                  baseline_counts=baseline, baseline_high=old_high, baseline_search=old_search, baseline_low=old_low)
    audit.update(events=len(a), SR=dict(high=int(high.sum()), high_search=int(high_search.sum()),
        low=int(low.sum()), intersection=0, baseline_high=int(old_high.sum()), baseline_low=int(old_low.sum()),
        high_gained=int((high & ~old_high).sum()), high_lost=int((old_high & ~high).sum()),
        low_gained=int((low & ~old_low).sum()), low_lost=int((old_low & ~low).sum()), **topology))
    return {'SR': result}, audit


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


def _response_identity(dataset_id, mstop, mlsp, region):
    # Process labels aggregate several datasets, but these arrays belong to
    # one exact dataset/mass subset of the current chunk.
    return int(dataset_id), int(mstop), int(mlsp), region


def nominal_response_arrays(a, selections, model, cached_scores=None, year=2024):
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
                features = feature_arrays(a, sel['block'], region, selected, year=year)
                score[low] = model.predict(*features[:5], batch_size=128)
            result[_response_identity(dataset, ms, ml, region)] = dict(
                high=high, low=low, counts=sel['counts'][chosen],
                recoil=np.asarray(sel['block'].recoil[chosen]), score=score)
    return result


def _friend_chunks(chunks, friend):
    """Join by original ROOT row and event identity before any region masks."""
    from .trota_friend import read_friend_slice
    start = 0
    for chunk in chunks:
        stop = start + len(chunk)
        yield read_friend_slice(friend, chunk, start, stop)
        start = stop
    if start != friend.num_entries:
        raise ValueError('Incomplete corrected nominal TROTA friend traversal')


def _signal_normalization_for_histogram(input_path, year, metadata, normalization):
    source = {str(i): r['dataset'] for i, r in metadata.get('datasets', {}).items() if r.get('is_signal')}
    replacement = normalization.get('signal_replacement')
    if not source or not replacement:
        return normalization, None
    current = {str(i): r['dataset'] for i, r in normalization['datasets'].items() if r.get('is_signal')}
    if all(current.get(i) == name for i, name in source.items()):
        return normalization, None
    from .trota_friend import find_friend_task
    if find_friend_task(input_path, year) is None:
        raise ValueError('Signal input is not part of the requested normalization inventory')
    from ..workflows.validation_regions import _registered_signal_normalization
    basis = _registered_signal_normalization(analysis_era(year))
    if any(basis['datasets'].get(i) != name for i, name in source.items()):
        raise ValueError('Signal input is not part of the registered Runs sumw inventory')
    return {**normalization, **basis['norm']}, {k: basis[k] for k in ('basis', 'era')}


def build(input_path, year, output, modes=('highdm', 'lowdm'), variations=('nominal',), chunk_size=2000,
          shape_mode=None, candidate_budget=150000, object_endpoint=None, object_endpoints=None,
          support_only=False, lhe_weights=False, normalization_path=None,
          use_trota_friend=None, nominal_calibrations_only=False, signal_sr_only=False):
    """Build or resume one file under an exclusive output lock."""
    import fcntl
    output = internal_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(input_path, year, output, modes, variations, chunk_size, shape_mode,
                      candidate_budget, object_endpoint, object_endpoints, support_only, lhe_weights,
                      normalization_path, use_trota_friend, nominal_calibrations_only, signal_sr_only)


def _build(input_path, year, output, modes, variations, chunk_size, shape_mode,
           candidate_budget, object_endpoint, object_endpoints,
           support_only, lhe_weights, normalization_path, use_trota_friend, nominal_calibrations_only=False,
           signal_sr_only=False):
    """Build an input's histograms without changing its requested physics view.

    An explicit friend request with no shape mode means corrected CMS nominal
    only. The CMS-JME mode keeps its requested shape endpoints and can reuse
    nominal tags automatically once the accepted MC cache is complete.
    """
    year = era_year(year)
    modes, variations = tuple(modes), tuple(variations)
    input_path = internal_path(input_path)
    signal_source = None
    if lhe_weights or signal_sr_only:
        from .lhe_weights import lhe_input_population
        signal_source = lhe_input_population(input_path, year)
    if lhe_weights and signal_source is not None:
        raise ValueError('Generator nuisances are applied to backgrounds only; keep new signals nominal')
    if signal_sr_only and signal_source is None:
        raise ValueError('SR-only signal production requires the approved new UUID signal inputs')
    # The stored non-JME endpoints are frozen absolute templates. Corrected
    # nominal friends change their tag definition, not just their execution.
    if use_trota_friend and shape_mode == 'stored_trota_nonjme':
        raise ValueError('Corrected TROTA friends cannot replace the frozen stored-TROTA non-JME endpoints')
    friend_nominal_only = use_trota_friend is True and shape_mode is None and not lhe_weights
    if friend_nominal_only:
        shape_mode = 'cms_trota_jme'
    if lhe_weights:
        if variations != ('nominal',) or object_endpoint or object_endpoints:
            raise ValueError('Source LHE weights use the accepted nominal selection only')
        if shape_mode not in (None, 'cms_trota_jme') or (signal_source is not None and use_trota_friend):
            raise ValueError('LHE weights require the canonical corrected nominal tag view for this input')
        if signal_source is not None:
            signal_sr_only, nominal_calibrations_only = True, True
        shape_mode, support_only = 'cms_trota_jme', not signal_sr_only
    if nominal_calibrations_only and (shape_mode != 'cms_trota_jme' or variations != ('nominal',)
            or object_endpoint or object_endpoints):
        raise ValueError('Nominal-only calibration requires corrected CMS nominal without alternate endpoints')
    nominal_only = bool(nominal_calibrations_only or friend_nominal_only or lhe_weights)
    if signal_sr_only and (not nominal_only or support_only or use_trota_friend):
        raise ValueError('SR-only signal inference requires corrected nominal without CR support preselection or a friend')
    if signal_sr_only:
        use_trota_friend = False
    if chunk_size <= 0:
        raise ValueError('Positive chunk sizes are required')
    if not set(modes) <= {'highdm', 'lowdm'}:
        raise ValueError('Invalid histogram mode')
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
    output = internal_path(output)
    friend_task = None
    if shape_mode == 'cms_trota_jme' and use_trota_friend is not False:
        from .trota_friend import find_friend_task, friend_production_complete
        if use_trota_friend or friend_production_complete():
            friend_task = find_friend_task(input_path, year)
        if use_trota_friend and friend_task is None:
            raise ValueError('Corrected nominal TROTA friends require an approved MC input in the friend inventory')
    fingerprint = dict(size=input_path.stat().st_size, mtime_ns=input_path.stat().st_mtime_ns)
    metadata = validate_root(input_path, year)
    signal_lhe = None
    if lhe_weights or signal_sr_only:
        if signal_source is not None:
            from .lhe_weights import signal_lhe_provenance
            provenance = signal_lhe_provenance(signal_source, metadata, year)
            if lhe_weights:
                signal_lhe = provenance
            directory = ROOT / 'production' / 'histograms' if lhe_weights else ROOT / 'estimations'
            required = directory / f'normalization_{year}.json.gz'
            if normalization_path is not None and internal_path(normalization_path) != required:
                raise ValueError('New signal weights require the canonical normalization of their production stage')
            normalization_path = required
        elif not metadata['datasets'] or any(m.get('is_data') or m.get('is_signal')
                                                for m in metadata['datasets'].values()):
            raise ValueError('Background LHE requires an intermediate with only background MC datasets')
    from ..workflows.rare_top_policy import excluded_ids, require_clean
    excluded = excluded_ids(metadata['datasets'])
    excluded_datasets = {int(value) for value in excluded}
    configuration = read_json(configuration_path(year))
    normalization_path = internal_path(normalization_path) if normalization_path is not None else ROOT / 'estimations' / ('normalization_%s.json.gz' % year)
    normalization = read_json(normalization_path)
    normalization, signal_basis = _signal_normalization_for_histogram(input_path, year, metadata, normalization)
    contract = dict(input=str(input_path), fingerprint=fingerprint, year=year,
                    central_object_corrections=('target_era_intermediate' if analysis_era(year) in EARLY_ERAS
                                                else 'muon_scaled_resolution_fallback_v1'),
                    modes=list(modes), variations=list(variations))
    if signal_basis is not None:
        contract['signal_normalization'] = signal_basis
    if signal_sr_only:
        contract['signal_sr_only'] = True
        contract['regions'] = ['SR']
        contract['registered_signal_source'] = provenance
    if lhe_weights:
        from .lhe_weights import SOURCE_AUDIT_SCHEMA
        contract['lhe_weights'] = dict(source='original_nanoaod',
            join='file_id_entry_run_luminosityBlock_event',
            reduction='after_all_files_are_merged', source_audit=SOURCE_AUDIT_SCHEMA,
            signal_uncertainties=signal_lhe is not None,
            normalization='unscaled_selected_yields_and_signed_per_mass_denominators' if signal_lhe
                          else 'unchanged_nominal_gen_sumw')
        if signal_lhe is not None:
            contract['lhe_weights'].update(selection='new_uuid_corrected_trota_nominal',
                join='file_id_entry_run_luminosityBlock_event_GenModel', registered_source=signal_lhe)
    if analysis_era(year) in EARLY_ERAS:
        contract['muon_selection'] = dict(loose_pt_gt=15., medium_pt_gt=15.)
    if shape_mode is not None:
        if shape_mode not in ('cms_trota_jme', 'stored_trota_nonjme'):
            raise ValueError('Unknown shape implementation')
        contract['shape_mode'] = shape_mode
        if shape_mode == 'cms_trota_jme':
            if nominal_calibrations_only:
                contract['nominal_calibrations_only'] = True
            if friend_task is not None:
                from .trota_friend import SCHEMA as FRIEND_SCHEMA
                contract['nominal_trota_friend'] = dict(schema=FRIEND_SCHEMA,
                    central_object_view='cms_trota_jme', nominal_only=nominal_only)
            if support_only:
                contract['execution'] = dict(inference_support='all_region_pretopology_union',
                    physics_definitions_changed=False)
            elif signal_sr_only:
                contract['execution'] = dict(inference_support='all_retained_mc_rows',
                    physics_definitions_changed=False)
        else:
            from .corrections import NONJME_ENDPOINTS
            contract['object_endpoints'] = list(NONJME_ENDPOINTS)
            if object_endpoint is not None:
                contract['object_task_endpoint'] = object_endpoint
                contract['object_endpoints'] = [] if object_endpoint == 'nominal' else [object_endpoint]
            contract['object_endpoints'] = [e for e in endpoints if e != 'nominal']
            contract['execution'] = dict(schema='trotasr_file_local_v1', endpoints=list(endpoints),
                chunk_size=chunk_size, gnn_batch_size=256,
                branch_selection='consumer_columns',
                physics_definitions_changed=False)
    if Path(output).exists():
        from .io import scientific_settings
        previous = read_json(output)
        complete = previous.get('status') in ('complete', 'complete_one_file_test', 'complete_member_sums')
        if complete:
            saved = previous['contract']
            if saved['fingerprint']['size'] != fingerprint['size']:
                raise ValueError('Completed histogram reports a different input size: ' + str(output))
            def settings(record):
                selected = {k: v for k, v in record.items()
                            if k not in {'input', 'fingerprint', 'registered_signal_source'}}
                if 'lhe_weights' in selected:
                    selected['lhe_weights'] = {k: v for k, v in selected['lhe_weights'].items()
                                               if k != 'registered_source'}
                return scientific_settings(selected)
            if settings(saved) != settings(contract):
                raise ValueError('Completed histogram uses different inputs, corrections or binning: ' + str(output))
            require_clean(previous)
            return previous
    model = Rank005Numpy(portable_model(year)) if 'lowdm' in modes else None
    start = time.monotonic()
    payload = dict(schema='trotasr_histograms_v1', status='running', contract=contract,
                   category_order=list(CATEGORY_ORDER), input_metadata=metadata.get('datasets', {}),
                   bins_merged=False, sr_data_blinded=True, scope='single_file_histograms',
                   highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                   mixed_sf='not_provided', overlap_policy='B>M>R; independent W',
                   histograms={}, physical_histograms={}, dy_rz={}, audit={'chunks': [], 'weights': [], 'events_read': 0, 'migrations': {}})
    with ExitStack() as resources:
        f = resources.enter_context(uproot.open(input_path, object_cache=None, array_cache=None))
        if lhe_weights:
            from .lhe_weights import SourceWeights, accumulate, theory_process
            lhe_reader = resources.enter_context(SourceWeights(metadata, signal_provenance=signal_lhe))
            payload['lhe_histograms'] = {}
            payload['lhe_source_audit'] = lhe_reader.audit
            payload['lhe_source_audit']['excluded_dataset_ids'] = sorted(excluded_datasets)
            if signal_lhe is not None:
                payload['lhe_signal_denominators'] = lhe_reader.signal_denominators()
        resolved = resolved_candidates(f, year) if shape_mode != 'cms_trota_jme' else None
        policy = topw_file_input_policy(f)
        reader = TopWEvents(f, policy)
        # Events is an already reduced intermediate ntuple, not NanoAOD. Only
        # existing columns are read; all inference input and truth stay in ROOT.
        branches = f['Events'].keys()
        if friend_task is not None:
            from .trota_friend import open_friend
            friend = resources.enter_context(open_friend(friend_task, input_path))
            branches = [name for name in branches
                        if name != 'nTopMixed1pct' and not name.startswith(('TopMixed1pct_', 'nTopMixed1pct_'))]
        if shape_mode == 'stored_trota_nonjme':
            payload['branch_read'] = branch_plan(f['Events'])
            branches = payload['branch_read']['columns']
        from .corrections import corrected_central_chunks
        payload['audit']['central_object_corrections'] = {}
        source_chunks = corrected_central_chunks(reader.iterate(branches, step_size=chunk_size),
            year, payload['audit']['central_object_corrections'])
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
            if friend_task is not None:
                source_chunks = _friend_chunks(source_chunks, friend)
            models = None if friend_task is not None and nominal_only else load_models(year)
            views = event_views(source_chunks, year, models, payload['audit']['trota_inference'],
                                candidate_budget, support_only=support_only,
                                nominal_only=nominal_only, friend_chunks=friend_task is not None)
        else:
            views = ((chunk, resolved, 'nominal') for chunk in source_chunks)
        for a, resolved, endpoint in views:
            n = len(a)
            if endpoint == 'nominal':
                payload['audit']['events_read'] += n
                nominal_response = {}
                nominal_arrays = a
                unchanged_weights = {}
                topology_cache = (FixedJetTopologyCache(a, resolved, year)
                                  if shape_mode == 'stored_trota_nonjme' else None)
            # One selection/object pass also retains the disjoint off-Z
            # measurement sample. The displayed DY CR remains on-Z only.
            reuse_nominal = (shape_mode == 'stored_trota_nonjme'
                             and endpoint != 'nominal' and a is nominal_arrays)
            cache_nominal = (shape_mode == 'stored_trota_nonjme'
                             and (endpoint == 'nominal' or reuse_nominal))
            include_weight_variations = weight_variations_needed(endpoint, variations)
            if reuse_nominal:
                selections, audit = nominal_selections, copy.deepcopy(nominal_selection_audit)
                # select() updates veto collections in place. Preserve its
                # per-call bookkeeping after the nominal removal has occurred.
                audit.update(update_veto_leptons(a, year=year))
                cached_scores = nominal_scores
            elif signal_sr_only:
                if np.any(np.asarray(a['is_data'], bool)) or not np.all(np.asarray(a['is_signal'], bool)):
                    raise ValueError('Only signal MC rows are allowed in the SR-only histogram scope')
                selections, audit = _signal_sr_selection(a, resolved, year)
                cached_scores = None
            else:
                selections, audit = select(a, resolved, dy_mass_window=None, topology_cache=topology_cache, year=year)
                cached_scores = region_scores(a, selections, model, year=year) if shape_mode == 'stored_trota_nonjme' else None
            if endpoint == 'nominal':
                nominal_selection_audit = copy.deepcopy(audit)
                nominal_scores = cached_scores
            if endpoints is not None and 'nominal' not in endpoints and endpoint == 'nominal':
                if np.any(np.asarray(a['is_data'], bool)):
                    raise ValueError('Independent object variations require MC input')
                nominal_selections = selections
                nominal_response = nominal_response_arrays(a, selections, model, cached_scores, year)
                continue
            audit['endpoint'] = endpoint
            payload['audit']['chunks'].append(audit)
            source_groups = []
            if lhe_weights:
                simulation = np.asarray(a['is_signal'], bool)
                if np.any(np.asarray(a['is_data'], bool)) or np.any(simulation != (signal_lhe is not None)):
                    raise ValueError('Retained rows disagree with the LHE sample population')
                approved = ~np.isin(np.asarray(a['dataset_id']), list(excluded_datasets))
                contributing = np.zeros(len(a), dtype=bool)
                for selection in selections.values():
                    contributing |= selection['high'] | selection['low']
                source_rows = np.flatnonzero(approved & contributing)
                for indices, prescription, ratios in lhe_reader.groups(a[source_rows]):
                    source_groups.append((source_rows[indices], prescription, ratios))
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
                if int(dataset_id) in excluded_datasets:
                    continue
                chosen = np.all(identity == [dataset_id, ms, ml], axis=1)
                dataset, process, is_data, is_signal = dataset_label(metadata, int(dataset_id))
                if dataset == 'unknown':
                    raise ValueError('Unknown dataset ID')
                sub = {k: a[k][chosen] for k in ak.fields(a)}
                norm = norm_vector(normalization, sub, int(dataset_id), dataset, is_data, is_signal,
                                   require_normalization=True)
                masks = {r: ((selections[r]['high'] | selections[r]['low'])[chosen] if r in selections
                             else np.zeros(int(chosen.sum()), dtype=bool)) for r in REGIONS}
                if is_data:
                    base = {'nominal': np.ones(int(chosen.sum()))}; wa = {'data': True}
                else:
                    cache_key = (int(dataset_id), int(ms), int(ml))
                    if reuse_nominal and cache_key in unchanged_weights:
                        base, wa = unchanged_weights[cache_key]
                        wa = copy.deepcopy(wa)
                    else:
                        base, wa = event_weights(sub, dataset, process, year, masks, policy, apply_topw=False,
                                                 include_variations=include_weight_variations)
                        if cache_nominal:
                            unchanged_weights[cache_key] = (base, copy.deepcopy(wa))
                record = dict(dataset=dataset, process=process, audit=wa, topw={}, endpoint=endpoint)
                payload['audit']['weights'].append(record)
                sample = 'data' if is_data else (signal_mass_key(signal_topology(dataset), int(ms), int(ml)) if is_signal else process)
                if sample in ('TTV', 'TTTT'):
                    sample = 'TT'
                for region in selections:
                    if is_data and (region == 'SR' or not data_process_allowed(process, region)):
                        continue
                    sel = selections[region]
                    if not np.any((sel['high'] | sel['low'])[chosen]):
                        continue
                    if is_data:
                        tw = {'nominal': np.ones(int(chosen.sum()))}
                        old_tw = tw
                    else:
                        region_key = (*cache_key, region)
                        if reuse_nominal and region_key in unchanged_weights:
                            tw, topw_audit, old_tw = unchanged_weights[region_key]
                            record['topw'][region] = copy.deepcopy(topw_audit)
                        else:
                            tw, record['topw'][region] = topw_event_variations(ROOT, str(year), dataset, process, sub, policy,
                                cleaned=sel['sf_clean'][chosen], include_variations=include_weight_variations)
                            eta, phi = _references(a, region, year)
                            old_clean = clean_by_delta_r(a['fatjet_eta_all'], a['fatjet_phi_all'], eta, phi, .4)[chosen]
                            old_tw, _ = topw_event_variations(ROOT, str(year), dataset, process, sub, policy,
                                                            cleaned=old_clean, include_variations=False)
                            if cache_nominal:
                                unchanged_weights[region_key] = (tw, copy.deepcopy(record['topw'][region]), old_tw)
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
                        features = feature_arrays(a, b, region, selected_low, year=year)
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
                        def add_lhe(path, coordinates, mask, edges):
                            if not lhe_weights or is_data or v != 'nominal':
                                return
                            chosen_rows = np.flatnonzero(chosen)
                            # Signal member yields carry signed event weights
                            # and the nominal SFs, but no nominal or per-file
                            # variation normalization. Keep each mass point
                            # separate until summing all its original files.
                            hard_process = sample if is_signal else theory_process(dataset, process)
                            member_weights = base['nominal'] * tw['nominal'] if is_signal else weights
                            for source_indices, prescription, ratios in source_groups:
                                belongs = chosen[source_indices]
                                if not np.any(belongs):
                                    continue
                                local = np.searchsorted(chosen_rows, source_indices[belongs])
                                selected = mask[local]
                                if np.any(selected):
                                    accumulate(payload['lhe_histograms'], prescription, hard_process,
                                        path, np.asarray(coordinates)[local[selected]],
                                        member_weights[local[selected]], ratios[belongs][selected], edges)
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
                                    add_lhe(('highdm', region, label, sample, 'recoil'),
                                            recoil, mask, [RECOIL])
                            else:
                                for group in (1, 2):
                                    mask = high & (nb == group)
                                    add(payload['histograms'], ('highdm', v, region, 'Nb' + str(group), sample, 'recoil'),
                                        recoil[mask], weights[mask], RECOIL)
                                    add_lhe(('highdm', region, 'Nb' + str(group), sample, 'recoil'),
                                            recoil, mask, [RECOIL])
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
                                # Reduce the joint histogram first; derive both
                                # marginals from it to preserve their exact sum.
                                add_lhe(('lowdm', region, cat, sample, 'gnn_recoil'),
                                        np.column_stack((score, recoil)), mask, [edges, RECOIL])
            # Never export even raw SR event counts for data.
            if np.any(np.asarray(a['is_data'], dtype=bool)):
                audit['SR'] = {'blinded': True}
    if input_path.stat().st_size != fingerprint['size']:
        raise ValueError('Input ROOT size changed during the read; output not committed')
    if payload['audit']['events_read'] != reader.num_entries:
        raise ValueError('Incomplete Events traversal')
    payload['background_policy'] = dict(excluded_primaries=['TTWH', 'TTZH'],
        excluded_files=[dict(datasets={key: metadata['datasets'][key] for key in sorted(excluded)})])
    payload.update(status='complete', seconds=time.monotonic()-start,
                   versions={x: importlib.metadata.version(x) for x in ('numpy', 'awkward', 'uproot', 'coffea', 'correctionlib')})
    if lhe_weights:
        payload.update(status='complete_member_sums', scope='source_lhe_member_sums_not_reduced')
    write_json(output, payload)
    return payload
