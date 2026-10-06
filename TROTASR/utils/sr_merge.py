"""Lossless SR-only projection for the user-authorized category merge.

The event histogram and the original 162-bin definition are immutable sources.
Every source bin occurs exactly once. CR axes, processes and normalization
components are never pooled by this projection.
"""
import copy
import itertools
import numpy as np
from .io import read_json, sha256
from .paths import ROOT, internal_path
from .sr_binning import AXES, expand, matches


def load_merge(reference):
    path = internal_path(ROOT / reference['path'])
    if sha256(path) != reference['sha256']:
        raise ValueError('SR merge definition changed')
    result = read_json(path)
    validate_merge(result)
    return result


def validate_merge(config):
    if (config.get('schema') != 'trotasr_sr_category_merge_v1'
            or config.get('source_binning_sha256') != sha256(ROOT/'jsons/highdm_sr_binning.json')
            or config.get('threshold') != 2 or config.get('threshold_scope') != '2024+2025_category_integral'
            or config.get('cr_binning_changed') is not False
            or config.get('lowdm_bins') != 30):
        raise ValueError('Unrecognized SR-only merge contract')
    original = expand()
    declared = config.get('recoil_bin_merges', [])
    changes = {r['class_1based']: r for r in declared}
    if len(changes) != len(declared) or any(type(i) is not int or not 1 <= i <= len(config['categories']) for i in changes):
        raise ValueError('Invalid or duplicate explicit recoil merge class')
    covered = []
    for class_id, category in enumerate(config['categories'], 1):
        ids = category['source_categories_1based']
        if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or not 1 <= i <= 28 for i in ids):
            raise ValueError('Invalid source category indices')
        covered.extend(ids)
        source_edges = original[ids[0]-1]['edges']
        if any(original[i-1]['edges'] != source_edges for i in ids):
            raise ValueError('Category union requires identical source recoil edges')
        expected_edges = source_edges
        if class_id in changes:
            change = changes[class_id]
            if (change.get('operation') != 'merge_last_into_previous'
                    or change.get('source_categories_1based') != ids
                    or change.get('source_edges') != source_edges
                    or len(source_edges) < 3
                    or change.get('removed_boundary') != source_edges[-2]):
                raise ValueError('Explicit last-bin merge does not match its source class')
            expected_edges = source_edges[:-2] + source_edges[-1:]
        if category['edges'] != expected_edges:
            raise ValueError('Undeclared recoil-bin change or interpolation')
        if list(category['predicates']) != list(AXES):
            raise ValueError('Category axis order changed')
        for values in itertools.product(range(4), repeat=5):
            expected = any(all(matches(v, original[i-1]['predicates'][a])
                               for a,v in zip(AXES,values)) for i in ids)
            actual = all(matches(v,category['predicates'][a]) for a,v in zip(AXES,values))
            if actual != expected:
                raise ValueError('Merged predicate is not the exact source union')
    if sorted(covered) != list(range(1,29)):
        raise ValueError('Source categories omitted or double counted')
    if len(config['categories']) != config['categories_per_year']:
        raise ValueError('Category count mismatch')
    if sum(len(c['edges'])-1 for c in config['categories']) != config['bins_per_year']:
        raise ValueError('Merged bin count mismatch')
    return config


def bin_groups(config):
    """Exact source-bin partition, including explicitly authorized tail merges."""
    offsets = []
    offset = 0
    original = expand()
    for c in original:
        offsets.append(offset)
        offset += len(c['edges'])-1
    groups = [[offsets[k-1]+j for k in c['source_categories_1based']
               for j in range(original[k-1]['edges'].index(lo), original[k-1]['edges'].index(hi))]
              for c in config['categories'] for lo, hi in zip(c['edges'], c['edges'][1:])]
    if sorted(i for group in groups for i in group) != list(range(162)):
        raise ValueError('Not an exact partition of all162 SR bins')
    return groups


def project(values, groups):
    values = np.asarray(values, dtype=float)
    if values.shape != (162,) or not np.isfinite(values).all():
        raise ValueError('Invalid source SR array')
    out = np.asarray([values[g].sum() for g in groups])
    np.testing.assert_allclose(out.sum(), values.sum(), rtol=2e-12, atol=1e-9)
    return out


def project_prediction(record, config):
    if not record['sr_data_blinded'] or 'data' in record or 'data_variance' in record:
        raise ValueError('SR merge does not read observed data')
    groups = bin_groups(config)
    out = copy.deepcopy(record)
    out['labels'] = ['SR_highdm_bin'+str(i) for i in range(len(groups))]
    for field in ('background','variance'):
        out[field] = project(record[field],groups)
    for field in ('groups','group_variances','signals'):
        out[field] = {k:project(v,groups) for k,v in record[field].items()}
    out['nuisance_deltas'] = {k:{d:project(v[d],groups) for d in ('up','down')}
                              for k,v in record['nuisance_deltas'].items()}
    # Sum correlated endpoints first; never sum already-enveloped errors.
    out['uncertainty'] = np.sqrt(out['variance'] + sum(
        np.maximum(abs(v['up']),abs(v['down']))**2 for v in out['nuisance_deltas'].values()))
    return out
