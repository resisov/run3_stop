"""Fill high-/low-dM histograms and their weight and object variations.

Keep sparse SR axes and each control region's own category definition.
"""
from __future__ import annotations

from pathlib import Path
import ast
import copy
from functools import lru_cache
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable
import time
import importlib.metadata
import numpy as np
import awkward as ak
import uproot
from .paths import ROOT, JSONS, internal_path, era_year, analysis_era, EARLY_ERAS
from .selection_contract import require_current_llcr
from . import top_pt
from .io import read_json, scientific_settings, write_json, validate_root
from .reader import gated_event_chunks as _gated_event_chunks, resolved_candidates
from .analysis_regions import highdm_llcr_inference_support, select, select_highdm_llcr
from .event_selections import REGIONS, CATEGORY_ORDER, FixedJetTopologyCache, _references
from .region_kinematics import LEPTON_VETO_PT_MIN, update_veto_leptons
from .ids import clean_by_delta_r, object_masks
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


def histogram_settings(contract):
    """Keep the selected physics view independent of how its tags were computed."""
    settings = {key: value for key, value in contract.items()
                if key not in {'input', 'fingerprint', 'registered_signal_source', 'nominal_trota_friend'}}
    if 'lhe_weights' in settings:
        settings['lhe_weights'] = {key: value for key, value in settings['lhe_weights'].items()
                                   if key != 'registered_source'}
    elif (contract.get('nominal_trota_friend') or {}).get('nominal_only'):
        settings['nominal_calibrations_only'] = True
    return scientific_settings(settings)


def weight_variations_needed(endpoint, variations):
    """Evaluate weight shifts only on the nominal object collection."""
    return endpoint == 'nominal' and tuple(variations) != ('nominal',)


def _top_pt_replay_contract(prescription, primaries):
    if prescription is None:
        return {}
    if (prescription != top_pt.PRESCRIPTION or not isinstance(primaries, (tuple, list))
            or not primaries or not set(primaries) <= set(top_pt.PRIMARIES)):
        raise ValueError('Top-pT replay requires the current prescription and exact inclusive ttbar primaries')
    return {'top_pt_prescription': prescription}


def _weight_requests(endpoint, variations, prescription):
    """Return (ordinary weight shifts, top-pT shifts), on nominal objects only."""
    enabled = weight_variations_needed(endpoint, variations)
    if not enabled or prescription is None:
        return enabled, False
    names = set(variations)
    return bool(names - {'nominal', *top_pt.ENDPOINTS}), bool(names & {'all_weights', *top_pt.ENDPOINTS})


def _base_event_weights(cols, dataset, process, year, masks, policy, *, include_variations,
                        include_top_pt_variations, prescription):
    def evaluate(arrays):
        return event_weights(arrays, dataset, process, year, masks, policy,
                             apply_topw=False, include_variations=include_variations)
    if prescription is None:
        return evaluate(cols)
    if not top_pt.inclusive_ttbar(dataset):
        raise ValueError('Unexpected non-ttbar row in a top-pT contribution')
    return top_pt.replay_event_weights(evaluate, cols, dataset, include_variations=include_top_pt_variations)


def _requested_weights(endpoint, variations, sample, allowed):
    requested = tuple(variations) if endpoint == 'nominal' else ('nominal',)
    names = sorted(allowed) if requested == ('all_weights',) else requested
    if sample != 'TT':
        names = [name for name in names if name not in top_pt.ENDPOINTS]
    if set(names) - allowed:
        raise ValueError('Requested unevaluated variations: ' + str(set(names) - allowed))
    return names


@lru_cache(maxsize=1)
def candidate_columns():
    """Read only columns consumed by the stored-TROTA histogram calculation.

    Literal accesses come from the actual consumers; explicitly enumerated
    dynamic accesses retain optional branches without changing fallback rules.
    """
    consumers = ('utils/histogramming.py', 'utils/analysis_regions.py', 'utils/event_selections.py',
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
        for region, sel in selections.items():
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


@dataclass(frozen=True)
class _SystematicPreselection:
    name: str
    columns: tuple[str, ...]
    accept: Callable[[ak.Array], np.ndarray]
    after_central: Callable[[ak.Array], np.ndarray] | None = None
    inference_support: Callable[[ak.Array], np.ndarray] | None = None


DATASET_COMPLETE = 'complete_dataset_contribution'


def _check_dataset_coverage(metadata, binding, seen):
    if not isinstance(seen, dict) or set(seen) != set(binding['dataset_ids']):
        raise ValueError('Incomplete dataset replay coverage')
    for key, count in seen.items():
        expected = metadata['datasets'][key].get('events_written')
        if type(count) is not int or count < 0 or expected is not None and count != expected:
            raise ValueError('Dataset replay event count differs from the input metadata: ' + key)


def _selected_weights(arrays, keep, evaluate):
    """Evaluate event-local SFs on contributing rows; retain aligned response arrays."""
    if keep is None or np.all(keep):
        return evaluate(arrays, slice(None))
    if not np.any(keep):
        raise ValueError('Cannot evaluate weights on an empty selection')
    weights, audit = evaluate({k: col[keep] for k, col in arrays.items()}, keep)
    result = {}
    for name, value in weights.items():
        value = np.asarray(value)
        if value.shape != (np.count_nonzero(keep),):
            raise ValueError('Selected weight coverage mismatch')
        full = np.zeros(len(keep), dtype=value.dtype)
        full[keep] = value
        result[name] = full
    return result, audit


def _dataset_preselection(metadata, primaries):
    """Bind exact background primaries to file-local IDs, never a grouped Top label."""
    from ..workflows.rare_top_policy import primary
    if (not isinstance(primaries, (list, tuple)) or not primaries
            or any(not isinstance(p, str) or not p or '/' in p or p.strip() != p for p in primaries)
            or len(set(primaries)) != len(primaries)):
        raise ValueError('Supply distinct nonempty background dataset primaries')
    records = metadata['datasets']
    if any(r.get('is_data') or r.get('is_signal') for r in records.values()):
        raise ValueError('Dataset replay requires a background-only MC input')
    bound = {k: r['dataset'] for k, r in records.items() if primary(r) in primaries}
    if not bound:
        raise ValueError('No requested background primary in input metadata')
    selected_ids, known_ids = tuple(map(int, bound)), tuple(map(int, records))
    seen = {str(i): 0 for i in selected_ids}
    def accept(a):
        ids = np.asarray(a['dataset_id'])
        if (np.any(np.asarray(a['is_data'], bool)) or np.any(np.asarray(a['is_signal'], bool))
                or not np.all(np.isin(ids, known_ids))):
            raise ValueError('Invalid background dataset identity during replay')
        target = np.isin(ids, selected_ids)
        for key, count in zip(*np.unique(ids[target], return_counts=True)):
            seen[str(int(key))] += int(count)
        # Base-quality flags are invariant across the canonical JES/JER and
        # lepton/photon/tau/MET views and required by *all* SR/CR/off-Z selections.
        # Do not prefilter on nominal kinematics, tags or lepton multiplicities.
        return target & np.asarray(a['pass_base_common'], bool)
    return (dict(primaries=sorted(primaries), dataset_ids=bound), seen,
            _SystematicPreselection('background_dataset_and_common_quality',
                                    ('dataset_id', 'is_data', 'is_signal', 'pass_base_common'), accept))


def _common_preselection(metadata):
    """Invariant early gate for the full SR/CR/off-Z histogram population.

    Do not apply region-specific lepton, MET, mTb, or tag cuts here. Even for
    rows that fail quality, retain coverage and check dataset/population identity.
    """
    labels = metadata['datasets']
    known = tuple(map(int, labels))
    data = tuple(int(k) for k, m in labels.items() if m.get('is_data'))
    signal = tuple(int(k) for k, m in labels.items() if m.get('is_signal'))
    def accept(a):
        ids = np.asarray(a['dataset_id'])
        if (not np.all(np.isin(ids, known))
                or not np.array_equal(np.asarray(a['is_data'], bool), np.isin(ids, data))
                or not np.array_equal(np.asarray(a['is_signal'], bool), np.isin(ids, signal))):
            raise ValueError('Invalid dataset identity in SR/CR input')
        return np.asarray(a['pass_base_common'], bool)
    return _SystematicPreselection('all_datasets_common_quality',
                                   ('dataset_id', 'is_data', 'is_signal', 'pass_base_common'), accept)


def _all_region_input_gate(base, available, *, unchanged_stored, sr_only=False):
    """A cheap necessary union for the hadronic, photon and (on/off-Z) DY routes.

    The calibrated endpoints can alter pT, veto membership, MET and acceptance,
    but cannot change trigger decisions, add AK4 jets or create new photons or
    leptons. Only use scalar ROOT collection-length counters, never a nominal
    selected count as a bound on a shifted view. If a raw counter is absent,
    leave that bound open; don't unpack a jagged collection just to count it.
    """
    available = set(available)
    flavors = ('signal',) if sr_only else ('signal', 'photon', 'electron', 'muon')
    trig = tuple('pass_' + flavor + '_trigger' for flavor in flavors)
    # Legacy layouts can lack a particular count branch; count names below
    # are uproot's scalar counters for the corresponding *raw* jagged arrays.
    counters = dict(jets=('njet_eta_all', 'njet_source_index_all', 'njet_corrected_pt'),
                    photon=('nphoton_eta_all', 'nphoton_pt_all'),
                    electron=('nelectron_eta_all', 'nelectron_pt_all'),
                    muon=('nmuon_eta_all', 'nmuon_pt_all'))
    raw = {name: next((k for k in candidates if k in available), None)
           for name, candidates in counters.items() if name == 'jets' or name in flavors}
    stable = tuple(name for name in ('pass_zero_tau', 'pass_met_250', 'pass_ht_300')
                   if unchanged_stored and name in available)
    columns = tuple(dict.fromkeys((*base.columns, *trig, *(k for k in raw.values() if k), *stable)))
    def accept(a):
        accepted = base.accept(a)
        signal = np.asarray(a[trig[0]], bool)
        # Nominal-only, unreconstructed views use these stored flags literally.
        # No such requirement is applied upstream of JME/non-JME migrations.
        for name in stable:
            if name == 'pass_zero_tau':
                accepted = accepted & np.asarray(a[name], bool)
            else:
                signal = signal & np.asarray(a[name], bool)
        def at_least(flavor, n):
            key = raw[flavor]
            return True if key is None else np.asarray(a[key]) >= n
        # Full selectors also retain low-dM migration diagnostics when the
        # exported histogram axis is high-dM; keep the common bound at two.
        possible = signal
        for flavor, trigger in zip(flavors[1:], trig[1:]):
            possible = possible | (np.asarray(a[trigger], bool) & at_least(flavor, 1 if flavor == 'photon' else 2))
        return accepted & at_least('jets', 2) & possible
    return _SystematicPreselection(base.name + ('_possible_sr' if sr_only else '_possible_regions'), columns, accept)


def _llcr_mc_preselection(year, shape_mode, labels):
    """Choose the necessary cuts; the shared input and view code is region-agnostic."""
    from .event_selections import (HIGHDM_LLCR_JME_EARLY_COLUMNS, HIGHDM_LLCR_NONJME_INPUT_COLUMNS,
        highdm_llcr_input_support, highdm_llcr_jme_early_support)

    if shape_mode not in ('cms_trota_jme', 'stored_trota_nonjme'):
        return None
    jme = shape_mode == 'cms_trota_jme'
    columns = ('dataset_id', 'is_data', 'is_signal',
               *(HIGHDM_LLCR_JME_EARLY_COLUMNS if jme else HIGHDM_LLCR_NONJME_INPUT_COLUMNS))
    dataset_ids = tuple(labels)

    def accept(a):
        if (np.any(np.asarray(a['is_data'], bool)) or np.any(np.asarray(a['is_signal'], bool))
                or not np.all(np.isin(np.asarray(a['dataset_id']), dataset_ids))):
            raise ValueError('Wrong dataset identity in LLCR background preselection')
        return highdm_llcr_input_support(a, year, shape_mode=shape_mode)

    return _SystematicPreselection('necessary_llcr_conditions_before_heavy_projection', columns, accept,
        after_central=(lambda a: highdm_llcr_jme_early_support(a, year)) if jme else None,
        inference_support=(lambda a: highdm_llcr_inference_support(a, year=year)) if jme else None)




def _gated_systematic_chunks(chunks, year, audit, *, after_central=None, friend=None):
    """Carry absolute ROOT rows through central corrections and friend attachment."""
    from .corrections import corrected_central_chunks
    from .trota_friend import read_friend_slice

    record = audit.setdefault('input_preselection', {})
    for chunk, start, stop, rows in chunks:
        corrected, = corrected_central_chunks((chunk,), year, audit['central_object_corrections'])
        if len(corrected) != len(rows):
            raise ValueError('Central correction changed the number of selected Events')
        if after_central is not None:
            use = np.asarray(after_central(corrected))
            if use.shape != (len(rows),) or use.dtype.kind != 'b':
                raise ValueError('Post-correction preselection must return one Boolean per input row')
            if not np.all(use):
                corrected, rows = corrected[use], rows[use]
            record['rejected_after_central'] = record.get('rejected_after_central', 0) + len(chunk) - len(rows)
        record['passed_to_systematics'] = record.get('passed_to_systematics', 0) + len(rows)
        if len(rows):
            if friend is None:
                yield corrected
            else:
                yield read_friend_slice(friend, corrected, start, stop, row_indices=rows)


def _projected_stored_views(root, chunks, year, endpoints, audit):
    """Project stored resolved candidates once per selected chunk, shared by all endpoints."""
    from .reader import StoredResolvedProjection
    from .corrections import nonjme_event_views
    index = None
    for chunk in chunks:
        if index is None:
            index = StoredResolvedProjection(root['TROTA'], year)
        resolved = index.for_events(chunk)
        if not endpoints or endpoints == ('nominal',):
            yield chunk, resolved, 'nominal'
        else:
            yield from nonjme_event_views((chunk,), year, resolved, audit,
                                          tuple(e for e in endpoints if e != 'nominal'))


def _llcr_observed_chunks(tree, columns, labels, year, shape_mode, step_size, audit):
    """Read only the veto and trigger columns until an observed row is CR-only."""
    from ..workflows.validation_data import _project
    from .paths import muon_pt_min

    ds_ids = [k for k, (_, process, _, _) in labels.items() if data_process_allowed(process, "LLCR")]
    base = ('dataset_id', 'is_data', 'is_signal', 'pass_base_common', 'pass_signal_trigger')
    if shape_mode == 'cms_trota_jme':
        # Canonical data JME rebuild derives these masks from the same stored
        # lepton collections. Only the jet four-momenta are passed to the view.
        veto = (
            'electron_pt_all', 'electron_eta_all', 'electron_cutbased_all', 'electron_mini_iso_all',
            'muon_pt_all', 'muon_eta_all', 'muon_mini_iso_all', 'muon_loose_id_all', 'muon_medium_id_all',
            'photon_pt_all', 'photon_eta_all', 'photon_cutbased_all', 'photon_electron_veto_all',
        )
    else:
        veto = ('electron_veto_pt', 'muon_loose_pt')
    fields = (*base, *(veto if ds_ids else ()))
    for start in range(0, tree.num_entries, step_size):
        stop = min(tree.num_entries, start + step_size)
        gate = tree.arrays(fields, entry_start=start, entry_stop=stop)
        ids = np.asarray(gate['dataset_id'])
        if (len(ids) != stop - start or not np.all(gate['is_data']) or np.any(gate['is_signal'])
                or not np.all(np.isin(ids, tuple(labels)))):
            raise ValueError('Unexpected event range or dataset identity in observed LLCR input')
        audit['events_read'] += len(ids)
        if not ds_ids:
            continue
        if shape_mode == 'cms_trota_jme':
            masks = object_masks(gate, year)
            e, m = masks['electron_veto'], masks['muon_loose']
        else:
            e = gate['electron_veto_pt'] > LEPTON_VETO_PT_MIN
            m = gate['muon_loose_pt'] > muon_pt_min(year)
        use = (np.isin(ids, ds_ids) & np.asarray(gate['pass_base_common'], bool)
               & np.asarray(gate['pass_signal_trigger'], bool)
               & np.asarray((ak.sum(e, axis=1) + ak.sum(m, axis=1)) == 1))
        if np.any(use):
            entries = start + np.flatnonzero(use)
            more = _project(tree, entries, tuple(c for c in columns if c not in fields))
            yield ak.zip({c: (gate[c][use] if c in fields else more[c]) for c in columns}, depth_limit=1)


def _llcr_observed_stored_views(root, chunks, year):
    """Project stored resolved candidates only for events admitted to LLCR."""
    from ..workflows.validation_data import _project

    by_event = None
    for chunk in chunks:
        if by_event is None:
            identity = root['TROTA'].arrays(['file_id', 'entry'], library='np')
            by_event = {}
            for i, (file_id, entry) in enumerate(zip(identity['file_id'], identity['entry'])):
                by_event.setdefault((int(file_id), int(entry)), []).append(i)
        rows = np.asarray(sorted({i for file_id, entry in zip(chunk['file_id'], chunk['entry'])
                                  for i in by_event.get((int(file_id), int(entry)), ())}), dtype=int)
        yield chunk, resolved_candidates(root, year, project=_project, rows=rows), 'nominal'


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
          use_trota_friend=None, nominal_calibrations_only=False, signal_sr_only=False,
          highdm_llcr_only=False, *, dataset_primaries=None, top_pt_prescription=None, nominal_pass_only=False,
          force_recompute=False, expected_entries=None):
    """Build or resume one file under an exclusive output lock."""
    import fcntl
    output = internal_path(output)
    require_current_llcr(output)
    if dataset_primaries is not None and ROOT/'stats/run3/dataset_replay' not in output.parents:
        raise ValueError('Dataset contributions require a separate output in stats/run3/dataset_replay')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(input_path, year, output, modes, variations, chunk_size, shape_mode,
                      candidate_budget, object_endpoint, object_endpoints, support_only, lhe_weights,
                      normalization_path, use_trota_friend, nominal_calibrations_only, signal_sr_only,
                      highdm_llcr_only, dataset_primaries=dataset_primaries, top_pt_prescription=top_pt_prescription,
                      force_recompute=force_recompute, expected_entries=expected_entries, nominal_pass_only=nominal_pass_only)


def _build(input_path, year, output, modes, variations, chunk_size, shape_mode,
           candidate_budget, object_endpoint, object_endpoints,
           support_only, lhe_weights, normalization_path, use_trota_friend, nominal_calibrations_only=False,
           signal_sr_only=False, highdm_llcr_only=False, *, dataset_primaries=None, top_pt_prescription=None,
           force_recompute=False, expected_entries=None, nominal_pass_only=False):
    """Build an input's histograms without changing its requested physics view.

    An explicit friend request with no shape mode means corrected CMS nominal
    only. The CMS-JME mode keeps its requested shape endpoints and can reuse
    nominal tags automatically once the accepted MC cache is complete.
    """
    year = era_year(year)
    modes, variations = tuple(modes), tuple(variations)
    if nominal_pass_only and (not dataset_primaries or shape_mode != 'cms_trota_jme'
            or use_trota_friend is not True or variations != ('all_weights',)
            or set(modes) != {'highdm', 'lowdm'} or nominal_calibrations_only or lhe_weights
            or signal_sr_only or highdm_llcr_only or object_endpoint is not None or object_endpoints is not None):
        raise ValueError('Nominal-pass population requires the all-region background JME + weights replay')
    top_pt_contract = _top_pt_replay_contract(top_pt_prescription, dataset_primaries)
    if top_pt_contract and (shape_mode is None or nominal_calibrations_only) and use_trota_friend is not True:
        raise ValueError('Central top-pT replay requires the accepted corrected nominal TROTA friend')
    if dataset_primaries is not None and (lhe_weights or signal_sr_only or highdm_llcr_only or force_recompute):
        raise ValueError('Dataset replay is an independent all-region MC histogram contribution')
    if highdm_llcr_only and (modes != ('highdm',) or signal_sr_only or lhe_weights):
        raise ValueError('Scoped LLCR needs only highdm histograms, without signal or LHE processing')
    if force_recompute and (not highdm_llcr_only or type(expected_entries) is not int or expected_entries < 0):
        raise ValueError('Replacement requires a scoped LLCR inventory event count')
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
        if use_trota_friend is not True:
            raise ValueError('Background LHE weights require the accepted nominal TROTA friend')
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
    llcr_observed = False
    if highdm_llcr_only:
        labels = {int(k): dataset_label(metadata, int(k)) for k in metadata['datasets']}
        populations = {data for dataset, _, data, _ in labels.values()}
        if (len(populations) != 1
                or any(dataset == 'unknown' or signal for dataset, _, _, signal in labels.values())):
            raise ValueError('Scoped LLCR expects an exclusively observed or background input')
        llcr_observed = populations == {True}
        if llcr_observed and endpoints is not None and endpoints != ('nominal',):
            raise ValueError('Observed LLCR inputs only support the nominal endpoint')
    preselection = (_llcr_mc_preselection(year, shape_mode, labels)
                    if highdm_llcr_only and not llcr_observed else None)
    # Scan all rows; project only the SR/CR numerator. Registered Runs and
    # original generator member denominators are independent of this iterator.
    regional_projection = bool(metadata.get('datasets')) and not highdm_llcr_only
    if regional_projection:
        if signal_sr_only and any(not m.get('is_signal') or m.get('is_data') for m in metadata['datasets'].values()):
            raise ValueError('Only signal MC rows are allowed in the SR-only histogram scope')
        preselection = _common_preselection(metadata)
        if (not signal_sr_only and shape_mode == 'cms_trota_jme'
                and not any(r.get('is_data') for r in metadata['datasets'].values())):
            support_only = True
    if dataset_primaries is not None:
        dataset_binding, seen_dataset_rows, preselection = _dataset_preselection(metadata, dataset_primaries)
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
    if dataset_primaries is not None:
        contract['dataset_replay'] = dataset_binding
        contract.update(top_pt_contract)
    if nominal_pass_only:
        contract['event_population'] = 'corrected_nominal_pass_sr_cr_and_dy_control_v1'
    if highdm_llcr_only:
        contract.update(regions=['LLCR'], highdm_llcr_only=True)
    if signal_basis is not None:
        contract['signal_normalization'] = signal_basis
    if signal_sr_only:
        contract['signal_sr_only'] = True
        contract['regions'] = ['SR']
        contract['registered_signal_source'] = provenance
    if lhe_weights:
        from .lhe_weights import SOURCE_AUDIT_SCHEMA
        contract['lhe_weights'] = dict(source='original_nanoaod',
            object_policy='fixed_nominal_events_and_aligned_friend', jet_recalibration=False,
            trota_inference=False, background_scales='separate_muR_muF',
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
            if nominal_calibrations_only or friend_nominal_only:
                contract['nominal_calibrations_only'] = True
            if friend_task is not None:
                from .trota_friend import SCHEMA as FRIEND_SCHEMA
                contract['nominal_trota_friend'] = dict(schema=FRIEND_SCHEMA,
                    central_object_view='cms_trota_jme', nominal_only=nominal_only)
            if highdm_llcr_only and not llcr_observed:
                contract['execution'] = dict(inference_support='highdm_llcr_pretopology_per_jme_endpoint',
                    physics_definitions_changed=False)
            elif support_only:
                contract['execution'] = dict(inference_support='all_region_pretopology_union',
                    physics_definitions_changed=False)
            elif signal_sr_only:
                contract['execution'] = dict(inference_support=('projected_sr_candidate_rows' if regional_projection
                                                                else 'all_retained_mc_rows'),
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
    if preselection is not None:
        contract.setdefault('execution', {})['input_preselection'] = preselection.name
    if Path(output).exists():
        previous = read_json(output)
        require_current_llcr(record=previous)
        complete = previous.get('status') in ('complete', 'complete_one_file_test', 'complete_member_sums',
                                              DATASET_COMPLETE)
        if complete:
            saved = previous['contract']
            if dataset_primaries is not None or saved.get('dataset_replay') is not None:
                if (previous.get('status') != DATASET_COMPLETE or dataset_primaries is None or
                        internal_path(saved['input']) != input_path or
                        previous.get('scope') != 'background_dataset_contribution'):
                    raise ValueError('Cannot mix full histograms and dataset contributions or change input')
            if force_recompute and internal_path(saved['input']) != input_path:
                raise ValueError('Cannot replace a completed histogram for a different input: ' + str(output))
            if histogram_settings(saved) != histogram_settings(contract):
                raise ValueError('Completed histogram uses different inputs, corrections or binning: ' + str(output))
            if ('events_written' in metadata
                    and previous['audit']['events_read'] != metadata['events_written']):
                raise ValueError('Completed histogram does not cover the current ROOT events: ' + str(output))
            dataset_fields = ('dataset', 'is_data', 'is_signal', 'events_written', 'sumw',
                              'signal_sumw_by_genmodel')
            def dataset_scope(records):
                return {key: tuple(record.get(field) for field in dataset_fields)
                        for key, record in records.items()}
            if dataset_scope(previous['input_metadata']) != dataset_scope(metadata['datasets']):
                raise ValueError('Completed histogram has different input populations: ' + str(output))
            if dataset_primaries is not None:
                _check_dataset_coverage(metadata, dataset_binding, previous['audit'].get('dataset_rows'))
            require_clean(previous)
            if not force_recompute:
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
        if expected_entries is not None and f['Events'].num_entries != expected_entries:
            raise ValueError('Histogram event count differs from the inventory')
        if lhe_weights:
            from .lhe_weights import SourceWeights, accumulate, theory_process
            lhe_reader = resources.enter_context(SourceWeights(metadata, signal_provenance=signal_lhe))
            payload['lhe_histograms'] = {}
            payload['lhe_source_audit'] = lhe_reader.audit
            payload['lhe_source_audit']['excluded_dataset_ids'] = sorted(excluded_datasets)
            if signal_lhe is not None:
                payload['lhe_signal_denominators'] = lhe_reader.signal_denominators()
        project_stored = regional_projection and shape_mode != 'cms_trota_jme'
        resolved = (resolved_candidates(f, year) if shape_mode != 'cms_trota_jme'
                    and not llcr_observed and not project_stored else None)
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
        if preselection is not None:
            if regional_projection:
                unvaried = shape_mode is None or (shape_mode == 'stored_trota_nonjme' and endpoints == ('nominal',))
                no_central_rebuild = (analysis_era(year) in EARLY_ERAS
                                      or all(m.get('is_data') for m in metadata['datasets'].values()))
                preselection = _all_region_input_gate(preselection, f['Events'].keys(),
                    unchanged_stored=unvaried and no_central_rebuild, sr_only=signal_sr_only)
                contract['execution'].update(input_preselection=preselection.name,
                                            gate_columns=list(preselection.columns))
            attached = friend if friend_task is not None else None
            if attached is not None and attached.num_entries != reader.num_entries:
                raise ValueError('Corrected nominal TROTA friend Events length differs')
            chunks = _gated_event_chunks(reader, branches, chunk_size,
                preselection.columns, preselection.accept, payload['audit'])
            source_chunks = _gated_systematic_chunks(chunks, year, payload['audit'],
                after_central=preselection.after_central, friend=attached)
        else:
            chunks = (_llcr_observed_chunks(reader.events, branches, labels, year, shape_mode, chunk_size, payload['audit'])
                      if llcr_observed else reader.iterate(branches, step_size=chunk_size))
            source_chunks = corrected_central_chunks(chunks,
                year, payload['audit']['central_object_corrections'])
        if llcr_observed and shape_mode != 'cms_trota_jme':
            views = _llcr_observed_stored_views(f, source_chunks, year)
        elif project_stored:
            if shape_mode == 'stored_trota_nonjme':
                payload['dy_rz_variations'] = {}
            views = _projected_stored_views(f, source_chunks, year, endpoints,
                                           payload['audit'].setdefault('object_endpoints', {}))
        elif shape_mode == 'stored_trota_nonjme':
            from .corrections import nonjme_event_views
            payload['audit']['object_endpoints'] = {}
            payload['dy_rz_variations'] = {}
            if endpoints == ('nominal',):
                views = ((chunk, resolved, 'nominal') for chunk in source_chunks)
            else:
                selected_endpoints = tuple(e for e in endpoints if e != 'nominal')
                views = nonjme_event_views(source_chunks, year, resolved, payload['audit']['object_endpoints'], selected_endpoints)
        elif lhe_weights:
            from .systematic_views import weight_only_friend_views
            if friend_task is None:
                raise ValueError('Missing corrected nominal friend for background LHE')
            if preselection is None:
                source_chunks = _friend_chunks(source_chunks, friend)
            views = weight_only_friend_views(source_chunks, year,
                payload['audit'].setdefault('trota_inference', {}),
                preselection.inference_support if preselection is not None else None)
        elif shape_mode:
            from .systematic_views import event_views
            from .trota_inference import load_models
            payload['audit']['trota_inference'] = {}
            payload['dy_rz_variations'] = {}
            if friend_task is not None and preselection is None:
                source_chunks = _friend_chunks(source_chunks, friend)
            if nominal_pass_only:
                from .systematic_views import nominal_pass_friend_chunks
                source_chunks = nominal_pass_friend_chunks(source_chunks, year,
                    payload['audit'].setdefault('nominal_pass_selection', {}))
            models = None if friend_task is not None and nominal_only else load_models(year)
            views = event_views(source_chunks, year, models, payload['audit']['trota_inference'],
                                candidate_budget, support_only=support_only,
                                nominal_only=nominal_only, friend_chunks=friend_task is not None,
                                support_mask=preselection.inference_support if preselection is not None else None)
        else:
            views = ((chunk, resolved, 'nominal') for chunk in source_chunks)
        for a, resolved, endpoint in views:
            n = len(a)
            if endpoint == 'nominal':
                if not llcr_observed and preselection is None:
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
            include_weight_variations, include_top_pt_variations = _weight_requests(
                endpoint, variations, top_pt_prescription)
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
            elif highdm_llcr_only:
                selections, audit = select_highdm_llcr(a, resolved, topology_cache=topology_cache, year=year)
                cached_scores = None
            else:
                selections, audit = select(a, resolved, dy_mass_window=None, topology_cache=topology_cache, year=year)
                cached_scores = (region_scores(a, selections, model, year=year)
                                 if lhe_weights or shape_mode == 'stored_trota_nonjme' else None)
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
                    for region in selections:
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
                masks = {r: ((selections[r]['high'] | selections[r]['low'])[chosen] if r in selections
                             else np.zeros(int(chosen.sum()), dtype=bool)) for r in REGIONS}
                weight_rows = None
                if not is_data:
                    if not any(np.any(mask) for mask in masks.values()):
                        continue
                    weight_rows = np.logical_or.reduce(tuple(masks.values()))
                    # Weighted before/after topology audits also consume baseline-only rows.
                    for sel in selections.values():
                        weight_rows |= (sel['baseline_search'] | sel['baseline_low'])[chosen]
                take = slice(None) if chosen.all() else chosen
                sub = {k: a[k][take] for k in ak.fields(a)}
                # The outer loop already partitions on this exact dataset and
                # mass pair. Look up its authoritative factor once, retaining
                # the same strict Runs/mass checks; NumPy broadcasts it on fill.
                normalization_key = {k: sub[k][:1] for k in ('dataset_id', 'mStop', 'mLSP')}
                norm = norm_vector(normalization, normalization_key, int(dataset_id), dataset, is_data, is_signal,
                                   require_normalization=True)
                if is_data:
                    base = {'nominal': np.ones(int(chosen.sum()))}; wa = {'data': True}
                else:
                    cache_key = (int(dataset_id), int(ms), int(ml))
                    if reuse_nominal and cache_key in unchanged_weights:
                        base, wa = unchanged_weights[cache_key]
                        wa = copy.deepcopy(wa)
                    else:
                        base, wa = _selected_weights(sub, weight_rows, lambda cols, ix:
                            _base_event_weights(cols, dataset, process, year, {r: m[ix] for r, m in masks.items()},
                                                policy, include_variations=include_weight_variations,
                                                include_top_pt_variations=include_top_pt_variations,
                                                prescription=top_pt_prescription))
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
                            region_rows = (masks[region] | (sel['baseline_search'] | sel['baseline_low'])[chosen]
                                           if weight_rows is not None else None)
                            clean = sel['sf_clean'][chosen]
                            tw, record['topw'][region] = _selected_weights(sub, region_rows, lambda cols, ix:
                                topw_event_variations(ROOT, str(year), dataset, process, cols, policy,
                                    cleaned=clean[ix], include_variations=include_weight_variations))
                            eta, phi = _references(a, region, year)
                            old_clean = clean_by_delta_r(a['fatjet_eta_all'], a['fatjet_phi_all'], eta, phi, .4)[chosen]
                            old_tw, _ = _selected_weights(sub, region_rows, lambda cols, ix:
                                topw_event_variations(ROOT, str(year), dataset, process, cols, policy,
                                                      cleaned=old_clean[ix], include_variations=False))
                            if cache_nominal:
                                unchanged_weights[region_key] = (tw, copy.deepcopy(record['topw'][region]), old_tw)
                    allowed = set(base) | set(tw)
                    names = _requested_weights(endpoint, variations, sample, allowed)
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
            if not highdm_llcr_only and np.any(np.asarray(a['is_data'], dtype=bool)):
                audit['SR'] = {'blinded': True}
    if input_path.stat().st_size != fingerprint['size']:
        raise ValueError('Input ROOT size changed during the read; output not committed')
    if payload['audit']['events_read'] != reader.num_entries:
        raise ValueError('Incomplete Events traversal')
    if expected_entries is not None and payload['audit']['events_read'] != expected_entries:
        raise ValueError('Histogram event count differs from the inventory')
    payload['background_policy'] = dict(excluded_primaries=['TTWH', 'TTZH'],
        excluded_files=[dict(datasets={key: metadata['datasets'][key] for key in sorted(excluded)})])
    payload.update(status='complete', seconds=time.monotonic()-start,
                   versions={x: importlib.metadata.version(x) for x in ('numpy', 'awkward', 'uproot', 'coffea', 'correctionlib')})
    if lhe_weights:
        payload.update(status='complete_member_sums', scope='source_lhe_member_sums_not_reduced')
    if dataset_primaries is not None:
        _check_dataset_coverage(metadata, dataset_binding, seen_dataset_rows)
        payload['audit']['dataset_rows'] = seen_dataset_rows
        payload.update(status=DATASET_COMPLETE, scope='background_dataset_contribution')
    write_json(output, payload)
    return payload
