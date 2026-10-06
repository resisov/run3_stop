"""Physical AN control distributions from the same selected event traversal.

Bins, overflow treatment and labels are copied from the legacy producer.
This does not change SR/CR acceptance or create any extra likelihood bins.
"""
import numpy as np
from .observable_specs import LOWDM_VARIABLE_SPECS, HIGHDM_DISTRIBUTION_VARIABLE_SPECS

HIGH = ('nb', 'njet', 'ntop', 'nw', 'met', 'ut')
LOW = {
    'LLCR': ('met', 'njet', 'nb_medium_lowdm', 'ht'),
    'QCDCR': ('met', 'njet', 'nb_medium_lowdm', 'ht'),
    'GCR': ('recoil_gcr', 'njet_photon_clean', 'nb_photon_clean', 'ht_photon_clean'),
    'DY2E': ('recoil_dy2e', 'njet_lepton_clean', 'nb_lepton_clean', 'ht_lepton_clean'),
    'DY2M': ('recoil_dy2m', 'njet_lepton_clean', 'nb_lepton_clean', 'ht_lepton_clean'),
}


def binned(values, weights, edges, overflow_policy='exclude'):
    values, weights, edges = np.asarray(values), np.asarray(weights), np.asarray(edges)
    if values.shape != weights.shape or not np.isfinite(weights).all():
        raise ValueError('Invalid physical-histogram weights')
    good = np.isfinite(values)
    values, weights = values[good], weights[good]
    if overflow_policy == 'fold':
        # Exactly the legacy upper-overflow folding. Never fold underflow.
        values = np.where(values > edges[-1], np.nextafter(float(edges[-1]), float(edges[0])), values)
    elif overflow_policy != 'exclude':
        raise ValueError('Unknown overflow policy')
    return dict(edges=edges.tolist(), **{k: np.histogram(values, edges, weights=w)[0].tolist()
        for k,w in (('sumw',weights),('sumw2',weights**2),('entries',None))})


def fill(target, mode, variation, region, sample, arrays, block, chosen, counts, acceptance, weight):
    if region == 'SR':
        return  # Categorized SR only; no collision-data leakage.
    nb = np.asarray(block.nb[chosen])
    if mode == 'highdm':
        specs = HIGHDM_DISTRIBUTION_VARIABLE_SPECS
        names = HIGH
        values = dict(nb=nb, njet=block.njet[chosen], ntop=counts[:,0], nw=counts[:,3],
                      met=np.asarray(arrays['met']), ut=block.recoil[chosen])
    else:
        specs = LOWDM_VARIABLE_SPECS
        names = LOW[region]
        values = dict(met=np.asarray(arrays['met']), **{name:
            (block.njet[chosen] if name.startswith('njet') else nb if name.startswith('nb_') else
             block.ht[chosen] if name.startswith('ht') else block.recoil[chosen]) for name in names if name != 'met'})
    for name in names:
        if mode == 'highdm' and name == 'met' and region in ('GCR','DY2E','DY2M'):
            continue  # AN uses visible-object recoil in these panels.
        spec = specs[name]
        for label, group in (('Nb1',nb == 1),('Nb2',nb >= 2)):
            mask = acceptance & group
            leaf = binned(np.asarray(values[name])[mask], weight[mask], spec['bins'], spec.get('overflow_policy','exclude'))
            node = target
            for key in (mode, variation, region, label, sample):
                node = node.setdefault(key,{})
            if name not in node:
                node[name] = dict(leaf, xlabel=spec['xlabel'], overflow_policy=spec.get('overflow_policy','exclude'))
            else:
                for key in ('sumw','sumw2','entries'):
                    node[name][key] = (np.asarray(node[name][key]) + leaf[key]).tolist()
