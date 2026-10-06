"""Bounded central/JME event views: one ROOT read, no persistent event data."""
import awkward as ak
import numpy as np
from .jme_calibration import calibrated_endpoints, ENDPOINTS
from .trota_inference import p4_view, P4_FIELDS, infer_chunk, bounded_slices, migration_counts
from .shape_kinematics import varied_view, rebuild


def anchored_endpoints(a, year, audit=None):
    """Keep stored precision; permit only the adopted sub-10-MeV JER floor."""
    values = calibrated_endpoints(a, year)
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
            for shift in ENDPOINTS:
                values[shift][name] = ak.where(floor, values[shift][name],
                                               stored+(values[shift][name]-nominal))
    return values


def analysis_support(view):
    """Necessary pre-topology selection, including the off-Z measurement.

    This is an execution mask, not a new selection: an event outside this
    union cannot enter any histogram whatever its resolved/mixed decision.
    """
    from .event_selections import REGIONS, highdm_core
    from .region_kinematics import build_region_blocks
    blocks, _ = build_region_blocks(view, dy_mass_window=None)
    support = np.zeros(len(view), dtype=bool)
    for region in REGIONS:
        support |= highdm_core(view, blocks, region, dy_mass_window=None)
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


def event_views(chunks, year, models, audit, candidate_budget=150000, support_only=False):
    for original in chunks:
        floor_audit = {}
        data = np.asarray(original['is_data'], dtype=bool)
        if np.any(data) and not np.all(data):
            raise ValueError('Mixed data/MC chunk is not a valid systematic input')
        endpoints = ('nominal',) if np.all(data) else ENDPOINTS
        if np.all(data):
            values = {'nominal': {n.replace('nanoaod','corrected'): original[n.replace('nanoaod','corrected')]
                                  for n in P4_FIELDS}}
        else:
            values = anchored_endpoints(original, year, floor_audit)
        inference_inputs = {s: p4_view(original, {n: values[s][n.replace('nanoaod','corrected')]
                                                 for n in P4_FIELDS}) for s in endpoints}
        bounds = {0, len(original)}
        for view in inference_inputs.values():
            for sl in bounded_slices(view, candidate_budget):
                bounds.update((sl.start, sl.stop))
        bounds = sorted(bounds)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            source = original[start:stop]
            nominal_states = None
            for shift in endpoints:
                momenta = {k: v[start:stop] for k,v in values[shift].items()}
                if support_only:
                    support = analysis_support(varied_view(source, momenta))
                    inferred, resolved, counts, states = supported_inference(
                        inference_inputs[shift][start:stop], models, support)
                else:
                    inferred, resolved, counts, states = infer_chunk(inference_inputs[shift][start:stop], models)
                # Inference input aliases must never leak into the type-1 MET
                # reference or into later correction calculations.
                for name in P4_FIELDS:
                    inferred = ak.with_field(inferred, source[name], name)
                view = varied_view(inferred, momenta)
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
                        target = record['candidate_migrations'].setdefault(kind, {})
                        for name, count in migration_counts(nominal_states[kind], states[kind]).items():
                            target[name] = target.get(name,0)+count
                yield view, resolved, shift
