"""Legacy displayed MC transfer factors, separate from fit rate parameters.

High-dM: process-specific SR/CR in Nb x native recoil, not SR topology bins.
Low-dM: each SR score bin / same-Nb CR integral, inclusive in NISR.
The raw Zinv/photon coefficient contains neither RZ nor Sgamma nor Qgamma.
The equations and signed/empty-bin policy follow the legacy TF producers.
"""
import numpy as np
from .sr_binning import NATIVE_EDGES
from .background_estimation import finite_json

ROUTES = {
    'top_llcr': (('ST', 'TT'), 'LLCR', ('ST', 'TT')),
    'w_llcr': (('WtoLNu',), 'LLCR', ('WtoLNu',)),
    'qcd_qcdcr': (('QCD',), 'QCDCR', ('QCD',)),
    'zinv_gcr': (('Zto2Nu',), 'GCR', ('GJ',)),
}


def summed(samples, names, observable, edges):
    values = np.zeros(len(edges)-1)
    variances = np.zeros_like(values)
    present = False
    for name in names:
        if name not in samples:
            continue
        leaf = samples[name][observable]
        if leaf['edges'] != edges:
            raise ValueError('TF histogram edges differ from the adopted definition')
        value, variance = (np.asarray(leaf[k], dtype=float) for k in ('sumw', 'sumw2'))
        if (value.shape != values.shape or variance.shape != values.shape
                or not np.isfinite(value).all() or not np.isfinite(variance).all()
                or (variance < 0).any()):
            raise ValueError('Malformed TF histogram')
        values += value
        variances += variance
        present = True
    return values, variances, present


def recoil_ratio(num, numvar, den, denvar):
    # Same >0 validity and independent-region variance as calculate_ratio in
    # the legacy plot_recoil_transfer_factors_2024.py.
    valid = (num > 0.) & (den > 0.)
    ratio = np.full(len(num), np.nan)
    variance = np.full(len(num), np.nan)
    ratio[valid] = num[valid] / den[valid]
    variance[valid] = numvar[valid] / den[valid]**2 + num[valid]**2 * denvar[valid] / den[valid]**4
    residual = np.full(len(num), np.nan)
    residual[valid] = (ratio[valid] * den[valid] - num[valid]) / num[valid]
    return dict(status='complete' if valid.all() else 'unsupported_cells',
                numerator=num, numerator_sumw2=numvar, denominator=den, denominator_sumw2=denvar,
                valid=valid, transfer_factor=ratio, mcstat=np.sqrt(np.maximum(variance, 0.)),
                mechanical_relative_residual=residual)


def score_ratio(num, numvar, den, denvar):
    total, totalvar = float(den.sum()), float(denvar.sum())
    if total <= 0.:
        return dict(status='invalid_denominator', denominator_total=total)
    covariance = np.diag(numvar) / total**2 + np.outer(num, num) * totalvar / total**4
    return dict(status='complete' if (num >= 0).all() else 'signed_mc_bins',
                numerator=num, numerator_sumw2=numvar,
                denominator=np.full(len(num), total), denominator_sumw2=np.full(len(num), totalvar),
                denominator_total=total, denominator_total_sumw2=totalvar,
                transfer_factor=num/total, mcstat=np.sqrt(np.diag(covariance)), covariance=covariance,
                valid=num >= 0., cr_score_yield=den, cr_score_sumw2=denvar,
                cr_score_fraction=den/total, mechanical_relative_residual=np.zeros(len(num)))


def build_transfer_factors(payload, config):
    if not payload.get('sr_data_blinded'):
        raise ValueError('TF input must retain SR blinding')
    high, low = (payload['histograms'][m]['nominal'] for m in ('highdm', 'lowdm'))
    if any('data' in samples for source in (high, low) for samples in source.get('SR', {}).values()):
        raise ValueError('SR data are forbidden in TF inputs')
    if payload.get('highdm_sr_source_axes') != 'exact_nb_exact_topology_native_recoil_v2':
        raise ValueError('TF requires exact Nb/native-recoil components')
    categories = config['sr_binning']['category_labels']
    if set(low.get('SR', {})) - set(categories):
        raise ValueError('Unknown low-dM SR category')
    cr_edges = config['cr_binning']['score_edges']
    high_records, low_records, cross_covariance, issues = {}, {}, {}, []
    for route, (numerator_names, region, denominator_names) in ROUTES.items():
        high_records[route], low_records[route], cross_covariance[route] = {}, {}, {}
        for group, control in (('Nb1', 'Nb1'), ('Nb2plus', 'Nb2')):
            num, numvar, have_num = np.zeros(8), np.zeros(8), False
            for category, samples in high.get('SR', {}).items():
                nb, *topology = map(int, category.split(','))
                if len(topology) != 4 or nb < 1:
                    raise ValueError('Invalid exact-Nb category')
                if (nb == 1) != (group == 'Nb1'):
                    continue
                v, v2, present = summed(samples, numerator_names, 'recoil', NATIVE_EDGES)
                num += v; numvar += v2; have_num |= present
            den, denvar, have_den = summed(high.get(region, {}).get(control, {}), denominator_names,
                                         'recoil', NATIVE_EDGES)
            high_records[route][group] = recoil_ratio(num, numvar, den, denvar)
            high_records[route][group]['process_coverage'] = dict(numerator=have_num, denominator=have_den)
            pooled_den, pooled_var, present = summed(low.get(region, {}).get(control, {}), denominator_names,
                                                   'gnn_score', cr_edges)
            selected = [c for c in categories if c.split('_')[0] == group]
            joint_num, joint_var = [], []
            for category in selected:
                edges = config['sr_binning']['edges_by_category'][category]
                v, v2, available = summed(low.get('SR', {}).get(category, {}), numerator_names, 'gnn_score', edges)
                row = score_ratio(v, v2, pooled_den, pooled_var)
                row.update(nb_group=group, parent=group, denominator_region=region,
                           score_edges=edges, cr_score_edges=cr_edges,
                           process_coverage=dict(numerator=available, denominator=present))
                if not available or not present:
                    row['status'] = 'missing_process_coverage'
                low_records[route][category] = row
                joint_num.extend(v); joint_var.extend(v2)
            if pooled_den.sum() > 0:
                cross_covariance[route][group] = dict(categories=selected,
                    bins_per_category=5, covariance=score_ratio(np.asarray(joint_num), np.asarray(joint_var),
                                                               pooled_den, pooled_var)['covariance'])
    for mode, records in (('highdm', high_records), ('lowdm', low_records)):
        for route, rows in records.items():
            for category, row in rows.items():
                if row['status'] != 'complete':
                    issues.append(dict(mode=mode, route=route, category=category, status=row['status']))
    return finite_json(dict(schema='trotasr_transfer_factors_v2', year=payload['contract']['year'],
        status='complete' if not issues else 'unsupported_cells', issues=issues,
        highdm=dict(edges=NATIVE_EDGES, records=high_records),
        lowdm=dict(kind='gnn', records=low_records, sr_binning=config['sr_binning'],
                   cr_binning=config['cr_binning'], shared_parent_covariance=cross_covariance),
        mc_stat_only=True, extra_weight_applied=False,
        definitions=dict(highdm='same-process SR/CR in Nb and native recoil',
                         lowdm='SR score bin / same-process same-Nb CR integral; NISR pooled',
                         zinv_gcr='raw Zinv/GJ MC; excludes RZ, Qgamma, Sgamma')))
