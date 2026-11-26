"""The inclusive ttbar Run 3 top-pT weight and its event-level uncertainty."""
import json

import awkward as ak
import numpy as np

from .paths import JSONS

with (JSONS / 'analysis.json').open(encoding='utf-8') as _config:
    _policy = json.load(_config)['inclusive_ttbar_top_pt']
PRESCRIPTION = _policy['prescription']
PRIMARIES = tuple(_policy['primaries'])
ENDPOINTS = tuple(_policy['endpoints'])
NUISANCE = _policy['nuisance']
CONTRIBUTION_DIRECTORY = _policy['contribution_directory']
_f = _policy['per_top_factor']


def inclusive_ttbar(dataset):
    primary = dataset.lstrip('/').split('/', 1)[0].split('_', 1)[0]
    return primary in PRIMARIES


def per_top_factor(pt):
    return (_f['exp_amplitude'] * np.exp(_f['exp_slope_per_GeV'] * pt)
            + _f['linear_slope_per_GeV'] * pt + _f['offset']) * (
                _f['energy_factor_offset'] + _f['energy_factor_slope_per_GeV'] * pt)


def pair_weight(top_pts):
    """Use the first two hard-process, last-copy tops; keep unpaired events at one."""
    padded = ak.pad_none(top_pts, 2, axis=1, clip=True)
    paired = np.asarray(ak.num(top_pts, axis=1)) >= 2
    result = np.ones(len(top_pts), dtype=float)
    if paired.any():
        a = np.asarray(ak.to_numpy(padded[paired, 0]), dtype=float)
        b = np.asarray(ak.to_numpy(padded[paired, 1]), dtype=float)
        with np.errstate(invalid='ignore'):
            result[paired] = np.sqrt(per_top_factor(a) * per_top_factor(b))
        if not np.isfinite(result[paired]).all():
            raise ValueError('Undefined inclusive ttbar top-pT weight for a generator top pair')
    return result


def variation_components(nominal):
    """Absolute replacement components. Do not mirror already-filled histogram bins."""
    return dict(zip(ENDPOINTS, (np.ones_like(nominal), 2.0 * nominal - 1.0)))


def replay_event_weights(evaluate, arrays, dataset, *, include_variations):
    """Replace exactly the ttbar factor in a scoped intermediate-ROOT replay.

    The stored gen_top_pt column contains only hard-process, last-copy tops;
    weight_inputs uses it solely for the top-pT component. Supplying a typed
    empty list per row to the other-weight evaluator makes its pair factor
    identically one under either prescription, without dividing signed or
    zero weights or changing dataset identity, normalization, or branch data.
    """
    if not inclusive_ttbar(dataset):
        return evaluate(arrays)
    pts = arrays['gen_top_pt']
    central = pair_weight(pts)
    cols = dict(arrays)
    cols['gen_top_pt'] = pts[:, :0]
    base, audit = evaluate(cols)
    weights = {}
    for name, value in base.items():
        if name in ENDPOINTS:
            continue
        value = np.asarray(value)
        if value.shape != central.shape:
            raise ValueError('Event count does not match top-pT component: ' + name)
        weights[name] = value * central
    if include_variations:
        for name, component in variation_components(central).items():
            weights[name] = base['nominal'] * component
    audit = dict(audit, components={**audit['components'], 'top_pt_reweight': {
        'applied': True, 'source': PRESCRIPTION}}, available_variations=sorted(weights))
    return weights, audit
