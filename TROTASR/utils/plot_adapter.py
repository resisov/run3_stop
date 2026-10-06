"""Data-format adapter to the byte-preserved legacy histogram renderer."""
import numpy as np
from .paths import internal_modules
internal_modules('autonomous_allhad', 'background_process_groups')
from .renderers import plot_control_search_bins_style as renderer
internal_modules('autonomous_allhad', 'background_process_groups')
from .renderers.background_process_groups import background_process_for_sample, BACKGROUND_DISPLAY_LABELS
from .sr_binning import project as project_adopted_sr, label as adopted_label


def category_label(mode, region, category):
    if mode == 'highdm' and region == 'SR':
        nb, b, m, r, w = map(int, category.split(','))
        return (r'$N_b%s$' % ('=1' if nb == 1 else r'\geq2') + '\n' +
                r'$N_{bst}=%d,\ N_{mix}=%d$' % (b, m) + '\n' + r'$N_{res}=%d,\ N_W=%d$' % (r, w))
    if region != 'SR':
        return r'$N_b=1$' if category == 'Nb1' else r'$N_b\geq2$'
    nb, nisr = category.split('_')
    return (r'$N_b%s$' % ('=1' if nb == 'Nb1' else r'\geq2') + '\n' +
            r'$N_{ISR}%s$' % (r'\geq2' if nisr.endswith('2plus') else '=' + nisr[-1]))


def full_sr_layout(bin_map, observable):
    """Require an explicit full layout; never infer it from a test's occupancy.

    This validates a supplied definition, not its physics approval. No default
    category universe, multiplicity cap, merging, or rebinning is invented here.
    """
    if not isinstance(bin_map, dict):
        raise ValueError('High-dM SR requires an explicit full bin map, including empty categories')
    required = dict(schema='trotasr_full_sr_bin_map_v1', scope='full', mode='highdm',
                    region='SR', observable=observable,
                    category_axis_order=['Nb', 'Nbst', 'Nmix', 'Nres', 'Nw'])
    if any(bin_map.get(k) != v for k, v in required.items()):
        raise ValueError('Invalid full high-dM SR bin-map contract')
    definitions = bin_map.get('categories')
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('Full SR bin map must contain category definitions')
    layout = {}
    numeric_order = []
    for record in definitions:
        if not isinstance(record, dict):
            raise ValueError('Invalid SR category definition')
        key = record.get('key')
        try:
            counts = tuple(map(int, key.split(',')))
        except (AttributeError, TypeError, ValueError):
            raise ValueError('Invalid SR category key')
        if (len(counts) != 5 or counts[0] not in (1, 2) or min(counts[1:]) < 0
                or key != ','.join(map(str, counts)) or key in layout):
            raise ValueError('Invalid or duplicate SR category key: ' + str(key))
        try:
            edges = np.asarray(record['edges'], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError('Invalid SR recoil edges: ' + key)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError('Invalid SR recoil edges: ' + key)
        layout[key] = edges.tolist()
        numeric_order.append(counts)
    if numeric_order != sorted(numeric_order):
        raise ValueError('SR category order must be Nb, Nbst, Nmix, Nres, Nw')
    total = sum(len(edges) - 1 for edges in layout.values())
    if type(bin_map.get('total_bins')) is not int or bin_map['total_bins'] != total:
        raise ValueError('Full SR bin-map total_bins mismatch')
    return layout


def blocks(payload, mode, region, observable, bin_map=None):
    adopted = (mode == 'highdm' and region == 'SR' and isinstance(bin_map, dict)
               and bin_map.get('schema') == 'trotasr_adopted_sr_binning_v1')
    adopted_categories = None
    if adopted:
        if observable != 'recoil':
            raise ValueError('The adopted high-dM SR axis is recoil')
        payload, adopted_categories = project_adopted_sr(payload, bin_map)
    categories = payload['histograms'].get(mode, {}).get('nominal', {}).get(region, {})
    fixed_sr = mode == 'highdm' and region == 'SR'
    layout = ({c['key']: c['edges'] for c in adopted_categories} if adopted else
              full_sr_layout(bin_map, observable) if fixed_sr else None)
    adopted_labels = {c['key']: adopted_label(c) for c in adopted_categories} if adopted else {}
    if fixed_sr and set(categories) - set(layout):
        raise ValueError('Observed SR categories outside the full bin map: ' +
                         ', '.join(sorted(set(categories) - set(layout))))
    order = list(layout) if fixed_sr else sorted(categories)
    out = []
    for category in order:
        records = categories.get(category, {})
        if fixed_sr and any(observable not in rec for rec in records.values()):
            raise ValueError('Missing SR observable in a populated category: ' + category)
        samples = {name: rec[observable] for name, rec in records.items() if observable in rec}
        if not samples and not fixed_sr:
            continue
        edges = layout[category] if fixed_sr else next(iter(samples.values()))['edges']
        n = len(edges)-1
        groups = {name: np.zeros(n) for name in renderer.GROUP_ORDER}
        variance = np.zeros(n); data = np.zeros(n); data_variance = np.zeros(n)
        for sample, leaf in samples.items():
            if leaf['edges'] != edges:
                raise ValueError('Inconsistent plotting bin edges; automatic rebinning is forbidden')
            for field in ('sumw', 'sumw2'):
                values = np.asarray(leaf[field], dtype=float)
                if values.shape != (n,) or not np.isfinite(values).all():
                    raise ValueError('Invalid plotting histogram array: ' + field)
            if np.any(np.asarray(leaf['sumw2']) < 0):
                raise ValueError('Negative plotting variance')
            if sample.startswith(('mStop', 'T2tt_', 'T2bW_', 'T2tb_')):
                continue  # Overlay points are not chosen implicitly.
            if sample == 'data':
                if region == 'SR':
                    raise ValueError('SR data present in plot input')
                data += np.asarray(leaf['sumw']); data_variance += np.asarray(leaf['sumw2'])
            else:
                group = BACKGROUND_DISPLAY_LABELS[background_process_for_sample(sample)]
                groups[group] += np.asarray(leaf['sumw']); variance += np.asarray(leaf['sumw2'])
        total = sum(groups.values())
        sr = region == 'SR'
        out.append(dict(nbin=n, label=adopted_labels[category] if adopted else category_label(mode, region, category), groups=groups,
                        background=total, background_unc=np.sqrt(variance), background_stat_unc=np.sqrt(variance),
                        data=data, data_unc=np.sqrt(data_variance), signals={}, signal_specs=[],
                        blind_data=sr or 'data' not in samples, xlabels=[str(i+1) for i in range(n)],
                        label_box=True, label_fontsize=10.2 if sr else 15, label_box_pad=.42 if sr else .28,
                        figure_width=16.4 if sr else 14., category_labels_on_main=True, category_label_y=.72,
                        main_panel_ymax_factor=600., significance_panel=sr, significance_ylim=[0., 1.],
                        main_ylabel='Events', physics_scope=region, unit_area=False))
        if sr and mode == 'highdm':
            # Exact styling values from the existing high-dM SR block builder.
            out[-1].update(xlabels=[], label_fontsize=12., label_box_pad=.18, figure_width=22.,
                           significance_ylim=[0., 5.], significance_mode='s_over_sqrt_b',
                           significance_ylabel=r'$S/\sqrt{B}$')
    return out
