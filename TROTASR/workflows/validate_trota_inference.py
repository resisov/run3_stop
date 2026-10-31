"""Read-only ROOT validation of internal in-memory TROTA inference.

Nominal closure is mandatory. A separate explicitly named CMS-central study
quantifies a convention difference; it never replaces the nominal or authorizes
a production mapping. Only aggregate validation JSON is written.
"""
import argparse
from dataclasses import fields
from pathlib import Path
import time
import resource
import numpy as np
import awkward as ak
import uproot
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import validate_root, write_json, sha256
from TROTASR.utils.reader import resolved_candidates
from TROTASR.utils.trota_inference import (load_models, bounded_slices, infer_chunk,
    p4_view, P4_FIELDS, MIXED_INTS, MIXED_FLOATS, migration_counts)


def compare_values(a, b, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(passed=False, shape_a=list(a.shape), shape_b=list(b.shape))
    same = (a == b) if exact else np.isclose(a, b, rtol=2e-5, atol=2e-5, equal_nan=True)
    with np.errstate(invalid='ignore'):
        delta = np.abs(a.astype(float)-b.astype(float))
    return dict(passed=bool(np.all(same)), different=int(np.count_nonzero(~same)),
                max_abs=float(np.max(delta[np.isfinite(delta)], initial=0)))


def compare_selections(a, ra, b, rb):
    from TROTASR.utils.event_selections import select
    before, _ = select(ak.copy(a), ra, dy_mass_window=None)
    after, _ = select(ak.copy(b), rb, dy_mass_window=None)
    checks = {}
    for region in before:
        for name in ('high', 'high_search', 'low', 'counts', 'sf_clean'):
            av, bv = before[region][name], after[region][name]
            if name == 'sf_clean':
                av, bv = ak.flatten(av), ak.flatten(bv)
            checks[region+'/'+name] = compare_values(av, bv, exact=True)
        for field in fields(before[region]['block']):
            av = getattr(before[region]['block'], field.name)
            bv = getattr(after[region]['block'], field.name)
            checks[region+'/block/'+field.name] = compare_values(av, bv)
    return checks


def validate(path, year, models, *, chunk_size=1000, budget=150000, central_study=False, selection_check=False):
    meta = validate_root(path, year)
    initial = path.stat()
    started = time.monotonic()
    checks, sums, study = [], {}, []
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        stored_resolved = resolved_candidates(root)
        stored_columns = root['TROTA'].arrays(library='np')
        selected_keys = list(zip(stored_columns['file_id'], stored_columns['entry']))
        lookup = {}
        for i, key in enumerate(selected_keys):
            lookup.setdefault(tuple(map(int, key)), []).append(i)
        for a in root['Events'].iterate(step_size=chunk_size, library='ak'):
            for sl in bounded_slices(a, budget):
                c = a[sl]
                out, resolved, audit, states = infer_chunk(c, models)
                for key, value in audit.items():
                    sums[key] = sums.get(key, 0) + value
                item = {'first_entry': int(c['entry'][0]), 'checks': {}}
                tests = item['checks']
                for name in ('nTopMixed1pct', *('TopMixed1pct_'+x for x in MIXED_INTS+MIXED_FLOATS)):
                    av, bv = c[name], out[name]
                    if name != 'nTopMixed1pct':
                        # Check lengths before flattening to protect event identity.
                        tests[name+'/lengths'] = compare_values(ak.num(av), ak.num(bv), True)
                        av, bv = ak.flatten(av), ak.flatten(bv)
                    tests[name] = compare_values(av, bv, name == 'nTopMixed1pct' or name[13:] in MIXED_INTS)
                s = states['resolved']; mask = s['passing']
                keys = list(zip(map(int, c['file_id']), map(int, c['entry'])))
                old = np.asarray([i for key in keys for i in lookup.get(key, [])], dtype=np.int64)
                tests['resolved/event'] = compare_values(np.repeat(np.arange(len(c)), [len(lookup.get(k, [])) for k in keys]), s['event'][mask], True)
                tests['resolved/index'] = compare_values(stored_columns['TopResolved1pct_candidateIndex'][old], s['index'][mask], True)
                for leg in range(3):
                    tests['resolved/source'+str(leg)] = compare_values(stored_columns['TopResolved1pct_sourceJetIdx'+str(leg)][old], s['sources'][mask, leg], True)
                for j, name in enumerate(('pt', 'eta', 'phi', 'mass')):
                    tests['resolved/'+name] = compare_values(stored_columns['TopResolved1pct_'+name][old], s['p4'][mask, j])
                for j, name in enumerate(('FTScore', 'TTScore', 'QCDScore')):
                    tests['resolved/'+name] = compare_values(stored_columns['TopResolved1pct_'+name][old], s['scores'][mask, j])
                tests['resolved/QCDDiscriminant'] = compare_values(stored_columns['TopResolved1pct_QCDDiscriminant'][old], s['discriminant'][mask])
                if selection_check:
                    tests.update(compare_selections(c, stored_resolved, out, resolved))
                checks.append(item)
                if central_study:
                    momenta = {name: c[name.replace('nanoaod', 'corrected')] for name in P4_FIELDS}
                    view = p4_view(c, momenta)
                    cout, cr, caudit, cs = infer_chunk(view, models)
                    study.append(dict(first_entry=int(c['entry'][0]), audit=caudit,
                        migrations={kind: migration_counts(states[kind], cs[kind]) for kind in states},
                        selection_comparison=(compare_selections(out, resolved, cout, cr) if selection_check else None)))
    final = path.stat()
    unchanged = (initial.st_ino, initial.st_size, initial.st_mtime_ns) == (final.st_ino, final.st_size, final.st_mtime_ns)
    passed = unchanged and all(x['passed'] for c in checks for x in c['checks'].values())
    return dict(input=str(path.relative_to(ROOT)), year=year, status='passed' if passed else 'failed',
        input_unchanged=unchanged, model_manifest_sha256=sha256(ROOT/'models/TROTA/manifest.json'),
        total=sums, chunks=checks, seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        selection_check=selection_check, cms_central_convention_study=study,
        scope='neutral_inference_closure_only_not_full_systematics_validation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--year', type=int, choices=(2024, 2025), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=1000)
    p.add_argument('--candidate-budget', type=int, default=150000)
    p.add_argument('--cms-central-study', action='store_true')
    p.add_argument('--selections', action='store_true')
    args = p.parse_args()
    path, output = internal_path(args.input), internal_path(args.output)
    if output.exists():
        raise FileExistsError('Validation evidence exists; do not overwrite')
    result = validate(path, args.year, load_models(args.year), chunk_size=args.chunk_size,
        budget=args.candidate_budget, central_study=args.cms_central_study, selection_check=args.selections)
    write_json(output, result)
    print(result['status'], result['total'], 'seconds', result['seconds'], flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
