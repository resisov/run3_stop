"""Current histogram adapter for the existing RZ, Qgamma, Sgamma and TF model.

The mathematical measurement functions live in internal, source-tracked modules.
No event reads, old measured factors, new binning, or guessed normalizations.
"""
import math
import numpy as np
from .dy_estimation import finalize_rz
from .photon_estimation import build_q_sgamma
from .zgamma_estimation import rebin, ratio, records
from .sr_binning import NATIVE_EDGES
from .renderers.background_process_groups import BACKGROUND_GROUP_SPECS

GROUPS = ('Nb1', 'Nb2plus')
EDGES = np.asarray([250., 300., 350., 400., 500., 1500.])
BACKGROUNDS = {s for _, _, sources in BACKGROUND_GROUP_SPECS for s in sources}


def finite_json(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def leaf_arrays(leaf):
    if leaf['edges'] != NATIVE_EDGES:
        raise ValueError('Changed native recoil edges')
    values = [np.asarray(leaf[k], dtype=float) for k in ('sumw', 'sumw2', 'entries')]
    if any(v.shape != (8,) or not np.isfinite(v).all() for v in values):
        raise ValueError('Malformed recoil histogram')
    if any((v < 0).any() for v in values[1:]):
        raise ValueError('Negative recoil variance/entries')
    return values


def recoil_boundary(payload):
    """Adapt both regimes without inferring missing data coverage from zeros."""
    streams = {v.get('process') for v in payload.get('input_metadata', {}).values() if v.get('is_data')}
    if not {'JetMET', 'EGamma', 'Muon'} <= streams:
        raise ValueError('Background measurements require JetMET, EGamma and Muon coverage')
    output = {}
    for mode in ('highdm', 'lowdm'):
        source = payload['histograms'][mode]['nominal']
        target = output[mode] = dict(recoil_edges=EDGES.tolist(), nb_groups=list(GROUPS), recoil={})
        for region in ('GCR', 'DY2E', 'DY2M'):
            by_group = target['recoil'][region] = {}
            for group, category in zip(GROUPS, ('Nb1', 'Nb2')):
                by_sample = source.get(region, {}).get(category, {})
                if 'data' not in by_sample:
                    raise ValueError('Missing measured CR data: ' + '/'.join((mode, region, category)))
                leaves = by_group[group] = {}
                for sample, observables in by_sample.items():
                    if sample not in BACKGROUNDS | {'data'}:
                        continue  # signal contamination convention unchanged
                    value, variance, entries = leaf_arrays(observables['recoil'])
                    leaves['data_obs' if sample == 'data' else sample] = {'nominal': {
                        k: rebin(a, np.asarray(NATIVE_EDGES), EDGES).tolist()
                        for k, a in zip(('sumw', 'sumw2', 'entries'), (value, variance, entries))}}
    return output


def photon_factors(exact):
    measurement = {'gcr_data': {'highdm': {'yields': {}}, 'lowdm': {'yields_by_group': {}}}}
    mc = {}
    for mode in ('highdm', 'lowdm'):
        mc[mode] = dict(recoil_edges=EDGES.tolist(), recoil={'GCR': {}})
        for group in GROUPS:
            source = exact[mode]['recoil']['GCR'][group]
            data = source['data_obs']['nominal']
            key = 'yields' if mode == 'highdm' else 'yields_by_group'
            measurement['gcr_data'][mode][key][group] = {
                str(i): {q: data[q][i] for q in ('sumw', 'sumw2')} for i in range(5)}
            mc[mode]['recoil']['GCR'][group] = {s: v for s, v in source.items() if s != 'data_obs'}
    result = build_q_sgamma(measurement, mc)
    problems, closure = [], {}
    for mode in ('highdm', 'lowdm'):
        closure[mode] = {}
        for group, payload in result[mode].items():
            if payload['Q']['status'] != 'complete' or payload['Q']['value'] <= 0:
                problems.append('/'.join((mode, group, 'Qgamma')))
            residuals = []
            for i, row in enumerate(payload['bins']):
                s = row['Sgamma']
                if s['status'] != 'complete' or s['value'] <= 0:
                    problems.append('/'.join((mode, group, 'Sgamma', str(i))))
                    continue
                predicted = payload['Q']['value'] * s['value'] * row['gamma_mc'] + row['other_mc']
                residual = predicted - row['data']
                if not math.isclose(predicted, row['data'], rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError('Qgamma/Sgamma algebraic closure failed')
                residuals.append(residual)
            closure[mode][group] = residuals
    return dict(status='blocked' if problems else 'complete', highdm=result['highdm'],
                lowdm_families=result['lowdm'], lowdm_Q_groups=result['lowdm_Q_groups'],
                closure_residuals=closure, unavailable=problems, recoil_edges=EDGES.tolist())


def ratio_inputs(exact, mode, regions, target):
    totals = {k: np.zeros(5) for k in ('data', 'data_variance', 'other', 'other_variance', 'target', 'target_variance')}
    for region in regions:
        for by_sample in exact[mode]['recoil'][region].values():
            for sample, node in by_sample.items():
                family = 'data' if sample == 'data_obs' else 'target' if sample == target else 'other'
                totals[family] += np.asarray(node['nominal']['sumw'])
                totals[family + '_variance'] += np.asarray(node['nominal']['sumw2'])
    value, stat, residual, residual_variance = ratio(*(totals[k] for k in
        ('data', 'data_variance', 'other', 'other_variance', 'target', 'target_variance')))
    return dict(totals, value=value, stat=stat, residual=residual,
                residual_variance=residual_variance, edges=EDGES.copy())


def measure(payload):
    if payload.get('status') not in ('complete', 'complete_one_file_test') or not payload.get('sr_data_blinded'):
        raise ValueError('Measurements require completed SR-blinded current histograms')
    exact = recoil_boundary(payload)
    result = {'sgamma': photon_factors(exact)}
    for mode, name, key in (('highdm', 'rz_high', 'rz_high'), ('lowdm', 'rz_low', 'rz_low')):
        raw = payload['dy_rz'][mode]['yields']
        converted = {}
        for channel in ('DY2E', 'DY2M'):
            converted[channel] = {}
            for group in GROUPS:
                converted[channel][group] = {}
                for window in ('on', 'off'):
                    cells = converted[channel][group][window] = {}
                    source = raw.get(channel, {}).get(group, {}).get(window, {})
                    for component in ('data', 'zll', 'other'):
                        leaf = source.get(component)
                        if leaf is None:
                            raise ValueError('Missing on/off-Z measurement component: ' + str((mode, channel, group, window, component)))
                        if leaf['edges'] != [0., 1.]:
                            raise ValueError('Invalid counting histogram')
                        cells[component] = {k: float(leaf[k][0]) for k in ('sumw', 'sumw2', 'entries')}
        rz = finalize_rz(converted)
        issues = [dict(channel=c, group=g, result=row) for c, groups in rz['channels'].items()
                  for g, row in groups.items() if row.get('status') != 'complete' or not row.get('fit_converged')]
        issues += [dict(group=g, result=row) for g, row in rz['combined'].items()
                   if row.get('status') != 'complete' or row.get('RZ', 0) <= 0]
        result[name] = dict(status='blocked' if issues else 'complete', **{key: rz}, issues=issues,
                            **{key+'_raw': converted, 'mll_'+('high' if mode == 'highdm' else 'low'):
                               payload['dy_rz'][mode].get('mll', {})})
    double = {'status': 'complete'}
    for mode in ('highdm', 'lowdm'):
        with np.errstate(divide='ignore', invalid='ignore'):
            rows = records(ratio_inputs(exact, mode, ('DY2E', 'DY2M'), 'DY'),
                           ratio_inputs(exact, mode, ('GCR',), 'GJ'))
        for row in rows:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                row['status'] = 'unavailable'
                double['status'] = 'blocked'
        double[mode] = {'bins': rows}
    result['double_ratio'] = double
    return finite_json(result)
