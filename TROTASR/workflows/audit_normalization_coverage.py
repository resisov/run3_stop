"""Prepare a separate normalization recovery candidate; never edit live assets.

Use existing integrated-file metadata and the preserved Runs-sumw formula.
No event sidecars, ROOT reads, fallback weights, or automatic promotion.
"""
import argparse
import copy
import math
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.nominal_campaign import fingerprint
from TROTASR.workflows.normalization_metadata_legacy import (
    merge_dataset, build_physical, build_signal_mass_points, load_signal_xsec,
    merge_signal_cutflow_histograms)


def missing_tasks(tasks, original):
    names = [Path(r['path']).name for r in original['source_sidecars']]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous original normalization source names')
    by_name = {Path(t['metadata']).name: t for t in tasks}
    if len(by_name) != len(tasks) or set(names) - set(by_name):
        raise ValueError('Normalization sources do not map uniquely onto current inputs')
    missing = [t for t in tasks if Path(t['metadata']).name not in set(names)]
    if any('_signal_shard_' not in t['key'] for t in missing):
        raise ValueError('This recovery is restricted to missing signal shards')
    return missing


def extend(original, additions, xsec):
    candidate = copy.deepcopy(original)
    datasets = candidate['datasets']
    cutflow_missing = []
    source_records = []
    for task, payload in additions:
        if (payload.get('status') != 'complete' or payload.get('bad_files') or
                payload.get('events_written') != task['entries'] or
                payload.get('files_attempted') != payload.get('files_processed')):
            raise ValueError('Incomplete added metadata: ' + task['key'])
        for dsid, rec in payload['datasets'].items():
            if not rec.get('is_signal') or rec.get('is_data'):
                raise ValueError('Added source is not exclusively signal')
            runs = rec.get('signal_sumw_by_genmodel', {})
            if not runs or any(not math.isfinite(float(v)) or float(v) < 0 for v in runs.values()):
                raise ValueError('Missing/invalid Runs mass-point sums')
            event_keys = set(rec.get('signal_event_genweight_sum_by_genmodel', {}))
            if event_keys - set(runs):
                raise ValueError('Selected signal lacks Runs sumw; no fallback allowed')
            if str(dsid) not in datasets:
                datasets[str(dsid)] = copy.deepcopy(rec)
            else:
                target = datasets[str(dsid)]
                for key in ('dataset', 'physical_dataset', 'process', 'is_data', 'is_signal'):
                    if target.get(key) != rec.get(key):
                        raise ValueError('Dataset identity collision: '+str(dsid)+'/'+key)
                merge_dataset(target, rec)
        if payload.get('signal_cutflow_histograms'):
            merge_signal_cutflow_histograms(candidate['signal_cutflow_histograms'], payload['signal_cutflow_histograms'])
        else:
            cutflow_missing.append(task['key'])
        source_records.append(dict(path=task['metadata'], status=payload['status'],
            files_attempted=payload['files_attempted'], files_processed=payload['files_processed'],
            events_written=payload['events_written']))
    points = build_signal_mass_points(datasets, xsec, original['luminosity_pb'])
    old_points = original['signal_mass_points']
    if set(old_points)-set(points):
        raise ValueError('Recovery lost original mass points')
    changed = {key: {'old': old_points[key]['normalization_factor'],
                     'new': points[key]['normalization_factor'],
                     'old_sumw': old_points[key]['sumw_mass_point'],
                     'new_sumw': points[key]['sumw_mass_point']}
               for key in old_points if old_points[key]['normalization_factor'] != points[key]['normalization_factor']}
    # Existing records are not relabelled/recomputed when every number agrees.
    for key in old_points:
        if key not in changed:
            if old_points[key] != points[key]:
                raise ValueError('Existing point provenance changed unexpectedly: '+key)
            points[key] = copy.deepcopy(old_points[key])
    # Do not resummate untouched background floating-point sums in a different
    # JSON key order. Copy those factors bit-for-bit, rebuilding only signal.
    for dsid, rec in original['datasets'].items():
        if not rec.get('is_signal') and datasets.get(dsid) != rec:
            raise ValueError('Background/data metadata changed: '+dsid)
    signal_physical, signal_split, signal_factors = build_physical(
        {k:v for k,v in datasets.items() if v.get('is_signal')}, original['luminosity_pb'])
    physical = copy.deepcopy(original.get('physical_datasets', {}))
    split = copy.deepcopy(original.get('physical_dataset_split_counts', {}))
    factors = copy.deepcopy(original['dataset_factors'])
    physical.update(signal_physical); split.update(signal_split); factors.update(signal_factors)
    blocked = {k:v for k,v in points.items() if v.get('normalization_factor') is None
               or not math.isfinite(float(v['normalization_factor'])) or v['normalization_factor'] <= 0}
    candidate.update(signal_mass_points=points, physical_datasets=physical,
        physical_dataset_split_counts=split, dataset_factors=factors,
        blocked_signal_mass_points=blocked, source_sidecars=original['source_sidecars']+source_records,
        source_sidecar_count=original['source_sidecar_count']+len(source_records))
    candidate['status'] = 'incomplete' if blocked else 'complete'
    candidate['recovery'] = dict(kind='missing_signal_metadata_coverage',
        formula_changed=False, fallback_used=False, changed_existing_points=changed,
        legacy_cutflow_unavailable_for_added_shards=cutflow_missing,
        legacy_cutflow_full_coverage=not cutflow_missing)
    audit = dict(added_shards=len(additions), added_events=sum(t['entries'] for t,p in additions),
        original_points=len(old_points), new_points=len(points),
        added_mass_points=sorted(set(points)-set(old_points)), changed_existing_points=changed,
        blocked_points=sorted(blocked), background_and_data_factors_unchanged=True,
        legacy_cutflow_unavailable_for_added_shards=cutflow_missing)
    return candidate, audit


def run(campaign, output):
    campaign, output = internal_path(campaign), internal_path(output)
    if output.exists():
        raise FileExistsError('Keep prior audit; no implicit rerun or overwrite')
    plan = read_json(campaign/'plan.json')
    xsec_path = ROOT/'stats/stop_xsec_13p6TeV.json'
    xsec = load_signal_xsec(xsec_path)
    yearly = {}
    for year in (2024, 2025):
        original_path = ROOT/'estimations'/('normalization_%s.json.gz'%year)
        original = read_json(original_path)
        if sha256(original_path) != plan['contracts'][str(year)]['normalization_sha256']:
            raise ValueError('Live normalization no longer matches frozen contract')
        tasks = [t for t in plan['tasks'] if t['year'] == year]
        missing = missing_tasks(tasks, original)
        additions, metadata_hashes = [], {}
        for task in missing:
            path = internal_path(ROOT/task['metadata'])
            if fingerprint(path) != task['metadata_fingerprint']:
                raise ValueError('Staged metadata changed: '+task['key'])
            additions.append((task, read_json(path)))
            metadata_hashes[task['key']] = sha256(path)
        candidate, audit = extend(original, additions, xsec)
        audit.update(original_sha256=sha256(original_path), metadata_sha256=metadata_hashes,
                     missing_keys=[t['key'] for t in missing])
        target = output/('normalization_%s.json.gz'%year)
        write_json(target, candidate)
        if read_json(target) != candidate:
            raise ValueError('Candidate readback failed')
        audit.update(candidate=str(target.relative_to(ROOT)), candidate_sha256=sha256(target))
        yearly[str(year)] = audit
    summary = dict(status='candidate_prepared_not_promoted', years=yearly,
        plan_sha256=sha256(campaign/'plan.json'), signal_xsec_sha256=sha256(xsec_path),
        live_assets_modified=False, live_controller_modified=False,
        full_workflow_complete=False)
    write_json(output/'audit.json', summary)
    print(__import__('json').dumps({y:{k:v for k,v in a.items() if k in
        ('added_shards','added_events','original_points','new_points','changed_existing_points','blocked_points')}
        for y,a in yearly.items()}, sort_keys=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    run(args.campaign,args.output)
