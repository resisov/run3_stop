"""Bounded central/JME event views: one ROOT read, no persistent event data."""
import awkward as ak
import numpy as np
from .jme_calibration import calibrated_endpoints, ENDPOINTS
from .trota_inference import (p4_view, P4_FIELDS, MIXED_INTS, MIXED_FLOATS, infer_chunk,
                              bounded_slices, migration_counts)
from .shape_kinematics import varied_view, rebuild


def anchored_endpoints(a, year, audit=None, *, nominal_only=False):
    """Keep stored precision; permit only the adopted sub-10-MeV JER floor."""
    values = calibrated_endpoints(a, year, nominal_only=nominal_only)
    for collection in ('jet', 'fatjet'):
        ptname = collection+'_corrected_pt'
        massname = collection+'_corrected_mass'
        pt, mass = values['nominal'][ptname], values['nominal'][massname]
        eta64 = ak.values_astype(a[collection+'_eta_all'], np.float64)
        energy = np.sqrt((pt*np.cosh(eta64))**2 + mass*abs(mass))
        at_floor = (pt<=.010000001)&(abs(energy-.01)<1e-9)
        stored_pt = ak.values_astype(a[ptname], np.float64)
        stored_mass = ak.values_astype(a[massname], np.float64)
        stored_e2 = (stored_pt*np.cosh(eta64))**2 + stored_mass*abs(stored_mass)
        close_pt = abs(stored_pt-pt) <= 2e-5+1e-5*abs(pt)
        close_mass = abs(stored_mass-mass) <= 2e-5+1e-5*abs(mass)
        # Only extend the former *failure* domain. Previously accepted p4 and
        # histograms retain their exact storage-precision anchoring.
        lhs, rhs = stored_mass*pt, mass*stored_pt
        same_scale = abs(lhs-rhs) <= 1e-15+1e-5*np.maximum(abs(lhs),abs(rhs))
        positive_floor = (at_floor & (stored_pt>0) & (stored_e2>=0)
                          & (stored_e2<.01**2) & same_scale & (~close_pt|~close_mass))
        floor = ((stored_pt==0)&at_floor) | positive_floor
        if audit is not None:
            audit['jme_positive_subfloor_'+collection] = int(ak.sum(positive_floor))
        for name in (ptname, massname):
            nominal = values['nominal'][name]
            stored = a[name]
            close = abs(stored-nominal) <= 2e-5+1e-5*abs(nominal)
            if ak.any(~close & ~floor):
                raise ValueError('Stored calibration does not match internal JME: '+name)
            for shift in values:
                values[shift][name] = ak.where(floor, values[shift][name],
                                               stored+(values[shift][name]-nominal))
    return values


def analysis_support(view, year=2024):
    """Necessary pre-topology selection, including the off-Z measurement.

    This is an execution mask, not a new selection: an event outside this
    union cannot enter any histogram whatever its resolved/mixed decision.
    """
    from .event_selections import REGIONS, highdm_core
    from .region_kinematics import build_region_blocks
    blocks, _ = build_region_blocks(view, dy_mass_window=None, year=year)
    support = np.zeros(len(view), dtype=bool)
    for region in REGIONS:
        support |= highdm_core(view, blocks, region, dy_mass_window=None, year=year)
        support |= blocks[region].core & (blocks[region].nb >= 1)
    return support


def supported_inference(source, models, support):
    """Infer only contributing events, then restore their original row indices."""
    from .trota_inference import MIXED_INTS, MIXED_FLOATS
    indices = np.flatnonzero(support)
    inferred, resolved, counts, states = infer_chunk(source[support], models)
    multiplicity = np.zeros(len(source), dtype=np.int32)
    multiplicity[indices] = np.asarray(inferred['nTopMixed1pct'])
    out = ak.with_field(source, multiplicity, 'nTopMixed1pct')
    for name in (*MIXED_INTS, *MIXED_FLOATS):
        field = 'TopMixed1pct_' + name
        out = ak.with_field(out, ak.unflatten(ak.flatten(inferred[field]), multiplicity), field)
    for state in states.values():
        state['event'] = indices[state['event']]
    counts.update(events=len(source), inference_events=len(indices),
                  outside_analysis_support=len(source)-len(indices))
    return out, resolved, counts, states


def _same_jet_momenta(left, right):
    """Bit-exact values and row boundaries, for a provably unchanged endpoint."""
    if left.keys() != right.keys():
        return False
    for name in left:
        a, b = left[name], right[name]
        if not np.array_equal(np.asarray(ak.num(a)), np.asarray(ak.num(b))):
            return False
        x, y = (np.ascontiguousarray(ak.to_numpy(ak.flatten(v))) for v in (a, b))
        if x.dtype != y.dtype or not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
            return False
    return True


def _inference_support(view, support_only, year, support_mask):
    if support_mask is None:
        return analysis_support(view, year=year) if support_only else np.ones(len(view), dtype=bool)
    support = np.asarray(support_mask(view))
    if support.shape != (len(view),) or support.dtype.kind != 'b':
        raise ValueError('TROTA support_mask must return one Boolean per rebuilt event')
    return support


def _replace_mixed(view, inferred):
    for field in ('nTopMixed1pct', *('TopMixed1pct_' + k for k in (*MIXED_INTS, *MIXED_FLOATS))):
        view = ak.with_field(view, inferred[field], field)
    return view


def _support_migrations(counts, nominal, varied):
    common = nominal & varied
    counts.update(support_migrations_common_events=int(common.sum()),
                  support_migrations_gained_events=int((varied & ~nominal).sum()),
                  support_migrations_lost_events=int((nominal & ~varied).sum()))
    return common


def _restrict_candidates(state, support):
    selected = support[state['event']]
    return {name: state[name][selected] for name in ('event', 'sources', 'passing')}


def _friend_selection(view, candidates, support):
    """Apply the selected execution mask to aligned precomputed tags."""
    if not np.all(support):
        counts = np.where(support, np.asarray(view['nTopMixed1pct']), 0).astype(np.int32)
        view = ak.with_field(view, counts, 'nTopMixed1pct')
        for name in (*MIXED_INTS, *MIXED_FLOATS):
            field = 'TopMixed1pct_' + name
            view = ak.with_field(view, ak.unflatten(ak.flatten(view[field][support]), counts), field)
    if hasattr(candidates, 'mask_rows'):
        # The friend is aligned to Events, so filter by row without building
        # another event-key dictionary or converting all candidates to objects.
        resolved = candidates.mask_rows(support)
        resolved_passing = resolved.candidate_count
    else:
        keys = zip(np.asarray(view['file_id'])[support], np.asarray(view['entry'])[support])
        resolved = {key: candidates[key] for f, e in keys if (key := (int(f), int(e))) in candidates}
        resolved_passing = sum(map(len, resolved.values()))
    counts = dict(events=len(view), inference_events=0, nominal_friend_events=len(view),
                  candidate_inventory_unavailable_events=len(view),
                  outside_analysis_support=len(view) - int(support.sum()),
                  mixed_passing=int(ak.sum(view['nTopMixed1pct'])),
                  resolved_passing=resolved_passing)
    return view, resolved, counts


def weight_only_friend_views(chunks, year, audit, support_mask=None):
    """Fixed Events coordinates + row-aligned nominal tag friend; no JME or inference."""
    for view, candidates in chunks:
        if not len(view):
            continue
        if np.any(np.asarray(view['is_data'])) or np.any(np.asarray(view['is_signal'])):
            raise ValueError('Expected only background MC in the LHE weight pass')
        if hasattr(candidates, 'matches_rows') and not candidates.matches_rows(
                np.asarray(view['file_id']), np.asarray(view['entry'])):
            raise ValueError('Fixed nominal friend is not aligned to LHE selection')
        support = _inference_support(view, True, year, support_mask)
        view, resolved, counts = _friend_selection(view, candidates, support)
        record = audit.setdefault('nominal', {})
        for k, v in counts.items():
            record[k] = record.get(k, 0) + v
        yield view, resolved, 'nominal'


def nominal_pass_friend_chunks(chunks, year, audit):
    """Feed shifted views only original rows accepted by the corrected nominal analysis.

    Read the aligned nominal tags, never infer them. The union spans both dM
    regimes, SR/CR and the existing DY mass measurement. It is a population
    restriction, not a frozen region, tag or bin assignment.
    """
    from .analysis_regions import select
    for field in ('input_rows', 'pretopology_rows', 'accepted_rows'):
        audit.setdefault(field, 0)
    for raw, friend in chunks:
        if not friend.matches_rows(raw['file_id'], raw['entry']):
            raise ValueError('Unaligned nominal-pass TROTA friend')
        if np.any(np.asarray(raw['is_data'], bool)) or np.any(np.asarray(raw['is_signal'], bool)):
            raise ValueError('Nominal-pass TT replay requires background MC')
        # Keep the fixed ROOT chunk; bound the transient central reconstruction.
        for start in range(0, len(raw), 8192):
            block = slice(start, start + 8192)
            source = raw[block]
            audit['input_rows'] += len(source)
            central = anchored_endpoints(source, year, nominal_only=True)['nominal']
            view = varied_view(source, central, year=year)
            eligible = np.flatnonzero(analysis_support(view[:], year=year))
            audit['pretopology_rows'] += len(eligible)
            if not len(eligible):
                continue
            cand = friend.select_rows(block).select_rows(eligible)
            view = view[eligible]
            selections, _ = select(view, cand, dy_mass_window=None, year=year)
            keep = np.zeros(len(view), dtype=bool)
            for region, sel in selections.items():
                passes = sel['high_search'] | sel['low']
                if region in ('DY2E', 'DY2M'):
                    # Includes the existing mll>50 control measurement, not
                    # just the on-Z plot; regular histogram masks remain intact.
                    passes = passes & (np.asarray(view['mee' if region == 'DY2E' else 'mmm']) > 50.)
                keep |= passes
            audit['accepted_rows'] += int(keep.sum())
            if keep.any():
                # Use the original corrected input, not the rebuilt nominal
                # view, so shifted MET and jet corrections are applied once.
                yield source[eligible[keep]], cand.select_rows(keep)


def _friend_event_views(chunks, year, models, audit, candidate_budget, support_only, nominal_only, support_mask):
    for original, friend_resolved in chunks:
        if np.any(np.asarray(original['is_data'], dtype=bool)):
            raise ValueError('Corrected nominal TROTA friends are MC-only')
        if not len(original):
            continue
        if hasattr(friend_resolved, 'matches_rows') and not friend_resolved.matches_rows(
                np.asarray(original['file_id']), np.asarray(original['entry'])):
            raise ValueError('Resolved friend rows are not aligned to JME Events')
        floor_audit = {}
        values = anchored_endpoints(original, year, floor_audit, nominal_only=nominal_only)
        endpoints = ('nominal',) if nominal_only else ENDPOINTS
        bounds = {0, len(original)}
        inference_inputs = {}
        if not nominal_only:
            # Keep the same union of candidate-bounded row ranges as the
            # legacy evaluator, including the bounds set by nominal inputs.
            for shift in endpoints:
                model_input = p4_view(original, {n: values[shift][n.replace('nanoaod', 'corrected')]
                                                 for n in P4_FIELDS})
                for sl in bounded_slices(model_input, candidate_budget, year=year):
                    bounds.update((sl.start, sl.stop))
                if shift != 'nominal':
                    inference_inputs[shift] = model_input
        bounds = sorted(bounds)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            source = original[start:stop]
            candidates = (friend_resolved.select_rows(slice(start, stop))
                          if hasattr(friend_resolved, 'select_rows') else friend_resolved)
            nominal = {k: v[start:stop] for k, v in values['nominal'].items()}
            nominal_support = None
            for shift in endpoints:
                momenta = nominal if shift == 'nominal' else {k: v[start:stop] for k, v in values[shift].items()}
                view = varied_view(source, momenta, year=year)
                support = _inference_support(view, support_only, year, support_mask)
                if shift == 'nominal' or _same_jet_momenta(nominal, momenta):
                    view, resolved, counts = _friend_selection(view, candidates, support)
                else:
                    if models is None:
                        raise ValueError('TROTA models are required for changed JME inputs')
                    model_input = inference_inputs[shift][start:stop]
                    if support_only or support_mask is not None:
                        inferred, resolved, counts, _ = supported_inference(model_input, models, support)
                    else:
                        inferred, resolved, counts, _ = infer_chunk(model_input, models)
                        counts['inference_events'] = len(source)
                    # The rebuild used original aliases for type-1 MET. The
                    # inference returns only a replacement Mixed collection;
                    # transfer that collection without another kinematic pass.
                    view = _replace_mixed(view, inferred)
                record = audit.setdefault(shift, {})
                if support_mask is not None:
                    if shift == 'nominal':
                        nominal_support = support
                    else:
                        _support_migrations(counts, nominal_support, support)
                if shift != 'nominal':
                    # The friend contains WP-passing candidates only, so all
                    # candidate migrations against nominal cannot be audited.
                    counts['candidate_migration_unavailable_events'] = len(source)
                for name, count in counts.items():
                    record[name] = record.get(name, 0) + count
                if start == 0:
                    for name, count in floor_audit.items():
                        record[name] = record.get(name, 0) + count
                yield view, resolved, shift


def event_views(chunks, year, models, audit, candidate_budget=150000, support_only=False,
                nominal_only=False, *, friend_chunks=False, support_mask=None):
    """Yield canonical views, optionally using validated, aligned nominal tags.

    Friend chunks are (Events with attached Mixed fields, Resolved-by-key).
    The friend does not replace the adopted central JME kinematic anchoring.
    An optional support_mask overrides support_only with a necessary selection
    evaluated separately on each fully rebuilt view, before any model inference.
    """
    if friend_chunks:
        yield from _friend_event_views(chunks, year, models, audit, candidate_budget,
                                      support_only, nominal_only, support_mask)
        return
    for original in chunks:
        floor_audit = {}
        data = np.asarray(original['is_data'], dtype=bool)
        if np.any(data) and not np.all(data):
            raise ValueError('Mixed data/MC chunk is not a valid systematic input')
        endpoints = ('nominal',) if np.all(data) or nominal_only else ENDPOINTS
        if np.all(data):
            values = {'nominal': {n.replace('nanoaod','corrected'): original[n.replace('nanoaod','corrected')]
                                  for n in P4_FIELDS}}
        else:
            values = anchored_endpoints(original, year, floor_audit, nominal_only=nominal_only)
        inference_inputs = {s: p4_view(original, {n: values[s][n.replace('nanoaod','corrected')]
                                                 for n in P4_FIELDS}) for s in endpoints}
        bounds = {0, len(original)}
        for view in inference_inputs.values():
            for sl in bounded_slices(view, candidate_budget, year=year):
                bounds.update((sl.start, sl.stop))
        bounds = sorted(bounds)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            source = original[start:stop]
            nominal_states = None
            nominal_support = None
            for shift in endpoints:
                momenta = {k: v[start:stop] for k,v in values[shift].items()}
                model_input = inference_inputs[shift][start:stop]
                if support_only or support_mask is not None:
                    view = varied_view(source, momenta, year=year)
                    support = _inference_support(view, support_only, year, support_mask)
                    inferred, resolved, counts, states = supported_inference(model_input, models, support)
                    # The first view already has the correct Type-1 reference
                    # and every varied kinematic field. Inference changes only
                    # the tag payload; reuse that view for any region mask.
                    view = _replace_mixed(view, inferred)
                else:
                    inferred, resolved, counts, states = infer_chunk(model_input, models)
                    # Inference aliases must never leak into the type-1 MET
                    # reference or into later correction calculations.
                    for name in P4_FIELDS:
                        inferred = ak.with_field(inferred, source[name], name)
                    view = varied_view(inferred, momenta, year=year)
                common = None
                if support_mask is not None:
                    if shift == 'nominal':
                        nominal_support = support
                    else:
                        common = _support_migrations(counts, nominal_support, support)
                record = audit.setdefault(shift, dict(events=0, resolved_candidates=0, resolved_passing=0,
                    mixed_candidates=0, mixed_passing=0, candidate_migrations={}))
                for name, count in counts.items():
                    record[name] = record.get(name,0) + count
                if start==0:
                    for name,count in floor_audit.items():
                        record[name] = record.get(name,0) + count
                if shift=='nominal':
                    nominal_states = states
                elif not np.all(data):
                    for kind in states:
                        before, after = nominal_states[kind], states[kind]
                        if common is not None:
                            # Candidate eligibility is measured only when both
                            # views were inferred; report support crossings separately.
                            before, after = (_restrict_candidates(s, common) for s in (before, after))
                        target = record['candidate_migrations'].setdefault(kind, {})
                        for name, count in migration_counts(before, after).items():
                            target[name] = target.get(name,0)+count
                yield view, resolved, shift
