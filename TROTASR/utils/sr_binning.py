"""The user-adopted 162-bin high-dM SR map; no occupancy-driven decisions."""
import copy
import itertools
import numpy as np
from .io import read_json
from .paths import ROOT


AXES = ('Nb', 'Nbst', 'Nmix', 'Nres', 'Nw')
# Histogram overflow is folded into the last interval, whose physical upper
# bound is infinity. This finite storage edge matches the existing histogrammer.
STORAGE_END = 1500.
NATIVE_EDGES = [250., 300., 350., 400., 500., 650., 800., 1000., STORAGE_END]


def matches(value, predicate):
    if set(predicate) == {'eq'}:
        return value == predicate['eq']
    if set(predicate) == {'ge'}:
        return value >= predicate['ge']
    raise ValueError('Invalid count predicate')


def expand(definition=None):
    d = read_json(ROOT / 'jsons/highdm_sr_binning.json') if definition is None else definition
    if (d.get('schema') != 'trotasr_adopted_sr_binning_v1'
            or d.get('category_axis_order') != list(AXES)
            or d.get('bins_per_year') != 162 or d.get('categories_per_year') != 28):
        raise ValueError('Not the adopted 162-bin definition')
    categories = []
    for row in d['topology_rules']:
        for nb in d['nb_partitions'][row['nb_partition']]:
            predicates = {'Nb': nb, **{axis: row[axis] for axis in AXES[1:]}}
            for pred in predicates.values():
                if len(pred) != 1 or next(iter(pred)) not in ('eq', 'ge') or type(next(iter(pred.values()))) is not int:
                    raise ValueError('Invalid adopted predicate')
            key = ','.join(axis + next(iter(pred)) + str(next(iter(pred.values())))
                           for axis, pred in predicates.items())
            met = d['met_binning'][row['met']]
            edges = list(map(float, met['lower_edges_GeV'])) + [STORAGE_END]
            if not met['last_bin_open_ended'] or not np.all(np.diff(edges) > 0):
                raise ValueError('Invalid adopted MET intervals')
            categories.append(dict(key=key, predicates=predicates, edges=edges, rule_id=row['id']))
    # Axis order is fixed. Pooled Nb>=1 precedes the exclusive Nb=1,2,>=3
    # display blocks; topology predicates make these categories disjoint.
    def order(c):
        p = c['predicates']
        nb = p['Nb']
        nborder = 0 if nb == {'ge': 1} else (1 if nb == {'eq': 1} else 2 if nb == {'eq': 2} else 3)
        return (nborder,) + tuple((next(iter(p[a].values())), next(iter(p[a])) == 'ge') for a in AXES[1:])
    categories.sort(key=order)
    if len(categories) != 28 or len({c['key'] for c in categories}) != 28 or sum(len(c['edges']) - 1 for c in categories) != 162:
        raise ValueError('Adopted map does not contain exactly 28 categories / 162 bins')
    # Exhaustive representatives of every equality and threshold partition.
    for values in itertools.product(range(4), repeat=5):
        count = sum(all(matches(v, c['predicates'][a]) for a, v in zip(AXES, values)) for c in categories)
        if count != int(values[0] >= 1 and any(values[1:])):
            raise ValueError('Overlapping or incomplete SR map: ' + str(values))
    return categories


def label(category):
    symbols = ('b', 'bst', 'mix', 'res', 'W')
    terms = []
    for axis, symbol in zip(AXES, symbols):
        pred = category['predicates'][axis]
        op = '=' if 'eq' in pred else r'\geq'
        terms.append(r'N_{%s}%s%d' % (symbol, op, next(iter(pred.values()))))
    return '$' + terms[0] + '$\n$' + ',\\ '.join(terms[1:3]) + '$\n$' + ',\\ '.join(terms[3:]) + '$'


def project(payload, definition):
    """Project exact Nb/topology/native-recoil aggregates, preserving sumw2."""
    if payload.get('highdm_sr_source_axes') != 'exact_nb_exact_topology_native_recoil_v2':
        raise ValueError('162-bin projection requires exact Nb; old Nb>=2-pooled test histograms are incompatible')
    layout = expand(definition)
    output = copy.deepcopy(payload)
    for variation, regions in output['histograms'].get('highdm', {}).items():
        projected = {}
        for key, samples in regions.get('SR', {}).items():
            values = tuple(map(int, key.split(',')))
            if len(values) != 5:
                raise ValueError('Malformed exact-count source key')
            hits = [c for c in layout if all(matches(v, c['predicates'][a]) for a, v in zip(AXES, values))]
            if len(hits) != 1:
                raise ValueError('Uncovered or overlapping SR source: ' + key)
            c = hits[0]
            for sample, observables in samples.items():
                leaf = observables['recoil']
                if leaf['edges'] != NATIVE_EDGES:
                    raise ValueError('SR source lacks the exact native recoil components')
                target = projected.setdefault(c['key'], {}).setdefault(sample, {}).setdefault('recoil',
                          dict(edges=c['edges'], **{f: [0.] * (len(c['edges'])-1) for f in ('sumw', 'sumw2', 'entries')}))
                for field in ('sumw', 'sumw2', 'entries'):
                    arr = np.asarray(leaf[field], dtype=float)
                    if arr.shape != (8,) or not np.isfinite(arr).all() or (field != 'sumw' and np.any(arr < 0)):
                        raise ValueError('Invalid native SR histogram array')
                    for i, (lo, hi) in enumerate(zip(c['edges'][:-1], c['edges'][1:])):
                        start, stop = NATIVE_EDGES.index(lo), NATIVE_EDGES.index(hi)
                        target[field][i] += float(arr[start:stop].sum())
        regions['SR'] = projected
    return output, layout
