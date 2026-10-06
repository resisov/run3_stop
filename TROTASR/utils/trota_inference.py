"""Bounded, in-memory full TROTA inference for nominal-closure and shape tests.

All eligible candidates are rebuilt, not just the stored WP-passing rows.
This module neither chooses a JME momentum convention nor writes event files.
The caller supplies explicit AK4/AK8 p4, leaving integrated ROOTs immutable.
"""
from __future__ import annotations
import numpy as np
import awkward as ak
from .paths import ROOT, internal_path
from .io import read_json, sha256
from .trota_candidates import (RESOLVED_JETS, JETS, FATJETS, _flatten_jets,
                               build_resolved_candidates, build_mixed_candidates,
                               pack, candidate_cost)

RESOLVED_WP = np.float32(0.9433798789978027)
MIXED_WP = np.float32(0.9027690887451172)
MIXED_INTS = ("candidateIndex", "category", "idxFatJet", "idxJet0", "idxJet1", "idxJet2",
              "sourceFatJetIdx", "sourceJetIdx0", "sourceJetIdx1", "sourceJetIdx2")
MIXED_FLOATS = ("pt", "eta", "phi", "mass", "FTScore", "TTScore", "QCDScore", "QCDDiscriminant")
P4_FIELDS = ("jet_nanoaod_pt", "jet_nanoaod_mass", "fatjet_nanoaod_pt", "fatjet_nanoaod_mass")


def load_models(year):
    manifest = read_json(ROOT / 'models/TROTA/manifest.json')
    if year not in manifest['application_years']:
        raise ValueError('Unsupported TROTA application year')
    paths = {}
    for kind, wp in (('resolved', RESOLVED_WP), ('mixed', MIXED_WP)):
        spec = manifest[kind]
        path = internal_path(ROOT / spec['path'])
        if sha256(path) != spec['sha256'] or spec['threshold'] != float(wp):
            raise ValueError('TROTA model hash/WP mismatch: ' + kind)
        paths[kind] = path
    import tensorflow as tf
    # Configure once, before either model initializes the runtime.
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    return {kind: tf.keras.models.load_model(str(path), compile=False) for kind, path in paths.items()}


def p4_view(a, momenta):
    """Replace exactly four inference inputs in a new Awkward record view."""
    if set(momenta) != set(P4_FIELDS):
        raise ValueError('Supply all four explicit TROTA momentum arrays')
    out = ak.Array(a)
    for name in P4_FIELDS:
        values = momenta[name]
        if not np.array_equal(np.asarray(ak.num(a[name])), np.asarray(ak.num(values))):
            raise ValueError('Unaligned TROTA p4: ' + name)
        if not np.isfinite(np.asarray(ak.flatten(values))).all():
            raise ValueError('Nonfinite TROTA p4: ' + name)
        # Finite negative shifted pT fails the existing candidate pT selection.
        # Do not reject the entire file before that selection, take abs(pT),
        # or clip/calibrate the endpoint differently from the event view.
        out = ak.with_field(out, values, name)
    return out


def bounded_slices(a, budget=150000):
    """Budget both models by candidate count, including previously failed WPs."""
    if budget < 1:
        raise ValueError('Candidate budget must be positive')
    good = (a[JETS[6]] != 0) & (a[JETS[0]] > 25) & (abs(a[JETS[1]]) < 2.5)
    nj = np.asarray(ak.sum(good, axis=1), dtype=np.int64)
    costs = candidate_cost(a) + nj * (nj-1) * (nj-2) // 6
    if np.any(costs > budget):
        raise ValueError('One event exceeds TROTA candidate budget; no event silently dropped')
    start, accumulated = 0, 0
    for i, cost in enumerate(costs):
        if accumulated + cost > budget and i > start:
            yield slice(start, i)
            start, accumulated = i, 0
        accumulated += int(cost)
    if start < len(a):
        yield slice(start, len(a))


def probabilities(model, inputs, batch_size=8192):
    n = len(next(iter(inputs.values())))
    if batch_size < 1 or any(len(v) != n or not np.isfinite(v).all() for v in inputs.values()):
        raise ValueError('Invalid model feature buffers/batch size')
    scores = np.empty((n, 3), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        values = np.asarray(model({k: v[start:stop] for k, v in inputs.items()}, training=False), dtype=np.float32)
        if values.shape != (stop-start, 3):
            raise ValueError('Unexpected TROTA output shape')
        scores[start:stop] = values
    if (not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Invalid TROTA probabilities')
    denom = scores[:, 1] + scores[:, 2]
    if np.any(denom <= 0):
        raise ValueError('Undefined TROTA QCD discriminant')
    return scores, scores[:, 1] / denom


def infer_chunk(a, models):
    """Infer one budgeted slice; return new Events view, Resolved map and audit.

    Fourth return value holds candidate states for validation/migration counts
    only, in memory. Constituent identities, not renumbered candidateIndex,
    identify a candidate across endpoints.
    """
    flat = _flatten_jets(a)
    rp = build_resolved_candidates(*flat)
    _, features, rev, rind, _, _, _, r0, r1, r2, rpt, reta, rphi, rmass = rp
    rs, rd = probabilities(models['resolved'], {'jet': features})
    rselected = rd >= RESOLVED_WP
    # The baseline builder returns vector positions. Resolve to original IDs.
    source = np.asarray(ak.flatten(a['jet_source_index_all']), dtype=np.int32)
    rsrc = np.column_stack([source[flat[0][rev] + idx] for idx in (r0, r1, r2)])
    rkin = np.column_stack((rpt, reta, rphi, rmass)).astype(np.float32)
    if not np.isfinite(rkin).all():
        raise ValueError('Nonfinite Resolved candidate p4')
    resolved = {}
    file_id, entry = np.asarray(a['file_id']), np.asarray(a['entry'])
    for i in np.flatnonzero(rselected):
        key = (int(file_id[rev[i]]), int(entry[rev[i]]))
        resolved.setdefault(key, []).append(dict(index=int(rind[i]), jets=rsrc[i].tolist(),
            score=float(rd[i]), eta=float(reta[i]), mass=float(rmass[i])))
    del features, rp
    jf, ff, tf, mi, mk, mev, invalid = build_mixed_candidates(*pack(a, JETS), *pack(a, FATJETS))
    ms, md = probabilities(models['mixed'], {'jet': jf, 'fatjet': ff, 'top': tf})
    mselected = md >= MIXED_WP
    if not np.isfinite(mk).all():
        raise ValueError('Nonfinite Mixed candidate p4')
    counts = np.bincount(mev[mselected], minlength=len(a)).astype(np.int32)
    out = ak.with_field(a, counts, 'nTopMixed1pct')
    mf = np.column_stack((mk, ms, md)).astype(np.float32)
    for names, values in ((MIXED_INTS, mi), (MIXED_FLOATS, mf)):
        for j, name in enumerate(names):
            out = ak.with_field(out, ak.unflatten(values[mselected, j], counts), 'TopMixed1pct_' + name)
    audit = dict(events=len(a), resolved_candidates=len(rev), resolved_passing=int(rselected.sum()),
                 mixed_candidates=len(mev), mixed_passing=int(mselected.sum()))
    for collection in ('jet', 'fatjet'):
        negative = a[collection+'_nanoaod_pt'] < 0
        if ak.any(negative):
            audit['negative_pt_excluded_'+collection+'_objects'] = int(ak.sum(negative))
            audit['negative_pt_excluded_'+collection+'_events'] = int(ak.sum(ak.any(negative, axis=1)))
    for name,count in zip(('zero_candidates','negative_candidates','zero_fatjets',
                           'negative_fatjets','zero_events','negative_events'),invalid):
        audit['mixed_invalid_denominator_'+name] = int(count)
    states = {
        'resolved': dict(event=rev, index=rind, sources=rsrc, p4=rkin, scores=rs, discriminant=rd, passing=rselected),
        'mixed': dict(event=mev, index=mi[:, 0], sources=mi[:, 6:10], p4=mk, scores=ms, discriminant=md, passing=mselected),
    }
    return out, resolved, audit, states


def migration_counts(before, after):
    """Separate eligibility migrations from WP crossings on common candidates."""
    def keyed(s):
        keys = [tuple([int(ev)] + [int(x) for x in sources]) for ev, sources in zip(s['event'], s['sources'])]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate TROTA constituent identity')
        return dict(zip(keys, map(bool, s['passing'])))
    b, a = keyed(before), keyed(after)
    common = b.keys() & a.keys()
    return dict(eligible_before=len(b), eligible_after=len(a),
        eligible_gained=len(a.keys()-b.keys()), eligible_lost=len(b.keys()-a.keys()),
        wp_fail_to_pass=sum(not b[k] and a[k] for k in common),
        wp_pass_to_fail=sum(b[k] and not a[k] for k in common),
        passing_gained_with_eligibility=sum(a[k] for k in a.keys()-b.keys()),
        passing_lost_with_eligibility=sum(b[k] for k in b.keys()-a.keys()))
