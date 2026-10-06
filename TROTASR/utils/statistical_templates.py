"""Internal native-TH1/card functions, extracted from three approved sources.

No legacy campaign loader, source-code argument, or external import is retained.
The exact source hashes and extraction scope are recorded in statistics_sources.json.
"""
from collections import OrderedDict, defaultdict
import math
import re
import sys
import numpy as np
from .paths import internal_path
from .renderers.signal_theory import add_signal_xsec_lnN, profiled_theory, load_theory as _read_theory

ALIASES = OrderedDict([('Top', 'Top'), ('WtoLNu', 'WJet'), ('Zto2Nu', 'ZJet'),
                       ('QCD', 'QCD'), ('PhotonJet', 'PhotonJet'), ('DY', 'DYJet'),
                       ('VV_VVV', 'VV')])
ORDER = ['signal'] + list(ALIASES.values())


def load_theory(table):
    """Read the internal cross-section table; no external AN-source fallback."""
    return _read_theory(internal_path(table))

def physical(name):
    for old, new in ALIASES.items():
        if name == old or name.startswith(old + '_'):
            return new
    raise ValueError(name)


def aggregate(records, preserve_flat=False):
    """Sum disjoint event cells, including signed variations and binwise lnN."""
    nominal = sum(float(r['nominal'][0]) for r, extras in records)
    variance = sum(float(r['sumw2'][0]) for r, extras in records)
    names = {n for r, extras in records for n in r['variations']}
    names.update(e['name'] for r, extras in records for e in extras)
    pairs = {}
    for name in sorted(names):
        pair = {}
        for direction in ('up', 'down'):
            value = 0.
            for record, extras in records:
                factor = next((float(e[direction]) for e in extras if e['name'] == name), 1.)
                varied = record['variations'].get(name, {}).get(direction, record['nominal'])
                value += float(varied[0]) * factor
            pair[direction] = value
        if preserve_flat or any(not math.isclose(v, nominal, rel_tol=1e-12, abs_tol=1e-15) for v in pair.values()):
            pairs[name] = pair
    return dict(nominal=nominal, sumw2=variance, variations=pairs)


def binding(process, channel):
    extras = tuple(sorted((e['name'], float(e['down']), float(e['up']))
                          for e in channel.get('extra_lnN', {}).get(process, [])))
    return physical(process), channel['rate_params'].get(process, ''), extras


def prepare_bin(channel, year, clipping, preserve_flat=False):
    buckets = defaultdict(list)
    initials = {}
    for process, record in channel['backgrounds'].items():
        key = binding(process, channel)
        buckets[key].append((record, []))
        initial = float(channel['rate_initial'].get(process, 1.))
        if key in initials and not math.isclose(initials[key], initial):
            raise ValueError('inconsistent initial rate')
        initials[key] = initial
    result = {}
    for key, records in buckets.items():
        record = aggregate(records, preserve_flat=preserve_flat)
        record['initial'] = initials[key]
        result[key] = record
    if 'signal' in channel['physical']:
        result['signal', '', ()] = dict(channel['physical']['signal'], initial=1.)
    for key, record in result.items():
        endpoints = [('nominal', record, 'nominal')]
        endpoints += [(n+d, pair, d) for n, pair in record['variations'].items() for d in ('up', 'down')]
        for endpoint, parent, field in endpoints:
            value = float(parent[field])
            if not math.isfinite(value):
                raise ValueError('nonfinite ' + endpoint)
            if value < 0:
                clipping.append(dict(year=year, channel=channel['name'], process=key[0],
                                     parameter=key[1], endpoint=endpoint, before=value, after=0.))
                parent[field] = 0.
        if not math.isfinite(record['sumw2']) or record['sumw2'] < 0:
            raise ValueError('invalid sumw2')
    return dict(name=channel['name'], year=year, region=channel['region'], regime=channel['regime'],
                observation=channel['observation'], records=result,
                control_group=channel.get('control_group'), nb_group=channel.get('nb_group'))


def empty():
    return dict(nominal=0., sumw2=0., variations={}, initial=1.)


def keys_for(bins):
    return sorted({key for b in bins for key in b['records']})


def unsupported(bins):
    """Nonempty components with no nominal support cannot be normalized TH1s."""
    bad = []
    for key in keys_for(bins):
        records = [b['records'].get(key, empty()) for b in bins]
        if sum(r['nominal'] for r in records) == 0 and any(
                r['sumw2'] > 0 or any(v > 0 for pair in r['variations'].values() for v in pair.values())
                for r in records):
            bad.append((key, 'nominal', ''))
        if sum(r['nominal'] for r in records) > 0:
            for name in {n for r in records for n in r['variations']}:
                for direction in ('up','down'):
                    if sum(r['variations'].get(name, {}).get(direction, r['nominal']) for r in records) <= 0:
                        bad.append((key, name, direction))
    return bad


def pack_bins(bins):
    packed = OrderedDict()
    for b in bins:
        if b.get('regime') == 'lowdm':
            signature = b['nb_group'] if b['region'] == 'GCR' else b['control_group']
        else:
            signature = tuple(sorted({parameter_unit(k[1]) for k in b['records'] if k[1]}))
        packed.setdefault(signature, []).append(b)
    groups = list(packed.values())
    # A zero nominal bin with finite variance is supported by autoMCStats,
    # but its process histogram needs positive total normalization. Packing
    # with another channel does not merge any observed/analysis bins.
    while True:
        change = False
        for i, group in enumerate(groups):
            bad = unsupported(group)
            if not bad:
                continue
            def support(other, problem):
                key, name, direction = problem
                records = [b['records'].get(key, empty()) for b in other]
                return sum(r['nominal'] if name == 'nominal' else r['variations'].get(name, {}).get(direction, r['nominal']) for r in records)
            candidates = [(j, other) for j, other in enumerate(groups) if j != i
                          and any(support(other, problem) > 0 for problem in bad)]
            if not candidates:
                continue
            j, other = min(candidates, key=lambda v: (len(v[1]), v[0]))
            combined = group + other
            groups = [g for n, g in enumerate(groups) if n not in (i, j)] + [combined]
            change = True
            break
        if not change:
            return groups


def parameter_unit(parameter):
    if not parameter:
        return 'fixed'
    match = re.search(r'_(?:highdm|lowdm)_(.+)_(bin\d+|inclusive)_202[45]$', parameter)
    if not match:
        raise ValueError(parameter)
    category, unit = match.groups()
    return category + (('_u' + unit[3:]) if unit.startswith('bin') else '')


def process_names(keys):
    result = {}
    for family in ORDER:
        selected = [k for k in keys if k[0] == family]
        if len(selected) == 1:
            result[selected[0]] = family
        else:
            units = [parameter_unit(k[1]) for k in selected]
            categories = [re.sub(r'_u\d+$', '', unit) for unit in units]
            for i, key in enumerate(selected):
                unit = categories[i] if categories.count(categories[i]) == 1 else units[i]
                name = family + '_' + unit
                if name in result.values():
                    name += '_part' + str(i+1)
                result[key] = name
    return result


def assemble(payloads, packing='region', preserve_flat=False):
    seen = set()
    for payload in payloads:
        if str(payload['year']) not in ('2024', '2025'):
            raise ValueError('Only 2024/2025 are supported')
        for original in payload['groups'].values():
            if not original:
                raise ValueError('An empty channel group is not an analysis layout')
            scope = original[0]['region'], original[0]['regime']
            for b in original:
                identity = str(payload['year']), b['name']
                if identity in seen or (b['region'], b['regime']) != scope:
                    raise ValueError('Duplicate analysis bin or mixed channel scope')
                seen.add(identity)
                if b['region'] != 'SR' and (not math.isfinite(float(b['observation'])) or b['observation'] < 0):
                    raise ValueError('Invalid CR observation')
    if packing not in ('region', 'compatible'):
        raise ValueError('Unknown TH1 packing')
    channels, data, clipping, issues = OrderedDict(), {}, [], []
    source_bin_count = 0
    for payload in payloads:
        year = payload['year']
        for block, original in payload['groups'].items():
            bins = [prepare_bin(b, year, clipping, preserve_flat=preserve_flat) for b in original]
            source_bin_count += len(bins)
            packed = [bins] if packing == 'region' else pack_bins(bins)
            for index, group in enumerate(packed, 1):
                region, regime = group[0]['region'], group[0]['regime']
                c = '{}_{}_c{}_{}'.format(region, regime, index, year)
                channels[c] = group
                keys = keys_for(group)
                names = process_names(keys)
                for key in keys:
                    records = [b['records'].get(key, empty()) for b in group]
                    values = np.asarray([r['nominal'] for r in records])
                    variance = np.asarray([r['sumw2'] for r in records])
                    nuisances = sorted({n for r in records for n in r['variations']})
                    pairs = {n: {d: np.asarray([r['variations'].get(n, {}).get(d, r['nominal']) for r in records])
                                 for d in ('up', 'down')} for n in nuisances}
                    active = {n: pair for n, pair in pairs.items() if any(
                        not np.allclose(v, values, rtol=1e-12, atol=1e-15) for v in pair.values())}
                    if values.sum() == 0 and not variance.any() and not active:
                        continue
                    if values.sum() <= 0:
                        issues.append(dict(channel=c, process=names[key], problem='zero_integral_nonempty_component',
                                           parameter=key[1], sumw2=float(variance.sum()), variations=list(active)))
                    for n, pair in active.items():
                        for direction, endpoint in pair.items():
                            if endpoint.sum() <= 0:
                                issues.append(dict(channel=c, process=names[key], problem='zero_integral_variation', nuisance=n, direction=direction))
                    present = [b['records'][key] for b in group if key in b['records']]
                    initial = present[0]['initial']
                    if any(not math.isclose(r['initial'], initial) for r in present):
                        raise ValueError('initial mismatch within TH1')
                    data[c, names[key]] = dict(family=key[0], parameter=key[1], extra_lnN=key[2],
                                               initial=initial, nominal=values, sumw2=variance,
                                               variations=pairs if preserve_flat else active)
    if source_bin_count != sum(map(len, channels.values())):
        raise ValueError('bin loss or duplication')
    return channels, data, clipping, issues


def template_name(channel):
    _, regime, _, year = channel.split('_')
    return 'template_{}_{}.root'.format(regime, year)


def card_text(channels, data, cr_only=False):
    selected = [c for c in channels if not cr_only or not c.startswith('SR_')]
    names = sorted({p for c, p in data if p != 'signal'}, key=lambda p: (ORDER.index(data[next(k for k in data if k[1] == p)]['family']), p))
    ids = {p: i+1 for i, p in enumerate(names)}
    ids['signal'] = 0
    columns = [(c, p) for c in selected for p in ['signal'] + names if (c, p) in data]
    active_names = {p for c, p in columns if p != 'signal'}
    lines = ['imax ' + str(len(selected)), 'jmax *' if cr_only else 'jmax ' + str(len(active_names)), 'kmax *', '------------']
    lines += ['shapes * {} {} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC'.format(c, template_name(c)) for c in selected]
    lines += ['------------', 'bin ' + ' '.join(selected), 'observation ' + ' '.join('-1' for c in selected), '------------',
              'bin ' + ' '.join(c for c, p in columns), 'process ' + ' '.join(p for c, p in columns),
              'process ' + ' '.join(str(ids[p]) for c, p in columns), 'rate ' + ' '.join('-1' for k in columns), '------------']
    # ROOT may retain supplied flat endpoints for complete provenance. They
    # remain inactive in the likelihood, as in the adopted nominal builder.
    active_shapes = {key: {n for n, pair in data[key]['variations'].items()
        if any(not np.allclose(v, data[key]['nominal'], rtol=1e-12, atol=1e-15)
               for v in pair.values())} for key in columns}
    shape_names = sorted({n for key in columns for n in active_shapes[key]})
    extra_names = sorted({e[0] for key in columns for e in data[key]['extra_lnN']})
    if set(shape_names).intersection(extra_names):
        raise ValueError('mixed shape/lnN nuisance type')
    for name in shape_names:
        lines.append(name + ' shape ' + ' '.join('1' if name in active_shapes[k] else '-' for k in columns))
    for name in extra_names:
        factors = {k: next((e for e in data[k]['extra_lnN'] if e[0] == name), None) for k in columns}
        lines.append(name + ' lnN ' + ' '.join('{:.8g}/{:.8g}'.format(factors[k][1], factors[k][2]) if factors[k] else '-' for k in columns))
    for year in ('2024', '2025'):
        lines.append('lumi_13p6TeV_' + year + ' lnN ' + ' '.join(
            '1.016' if c.endswith(year) and (data[c,p]['family'] == 'signal' or
                        (data[c,p]['family'] in ('VV','DYJet','PhotonJet') and not data[c,p]['parameter'])) else '-'
            for c,p in columns))
    for c, p in columns:
        r = data[c, p]
        if r['parameter']:
            lines.append('{} rateParam {} {} {:.8g} [0,10]'.format(r['parameter'], c, p, r['initial']))
    lines.append('* autoMCStats 10 1 1')
    return '\n'.join(lines) + '\n'


def write_templates(output, channels, data):
    output = internal_path(output)
    if sys.platform == 'darwin':
        raise RuntimeError('ROOT outputs are prohibited on the local laptop; use approved remote scratch')
    import ROOT
    ROOT.gROOT.SetBatch(True)
    for filename in sorted({template_name(c) for c in channels}):
        f = ROOT.TFile(str(internal_path(output / filename)), 'CREATE')
        if f.IsZombie():
            raise OSError(filename)
        for c, bins in channels.items():
            if template_name(c) != filename:
                continue
            f.mkdir(c).cd()
            def put(name, values, variances):
                hist = ROOT.TH1D(name, name, len(bins), 0., float(len(bins)))
                hist.Sumw2()
                for i, (value, variance) in enumerate(zip(values, variances), 1):
                    hist.SetBinContent(i, float(value))
                    hist.SetBinError(i, math.sqrt(float(variance)))
                    hist.GetXaxis().SetBinLabel(i, bins[i-1]['name'])
                hist.Write()
                hist.SetDirectory(0)
            for (channel, process), record in data.items():
                if channel != c:
                    continue
                put(process, record['nominal'], record['sumw2'])
                for name, pair in record['variations'].items():
                    for direction, suffix in (('up','Up'), ('down','Down')):
                        put(process + '_' + name + suffix, pair[direction], record['sumw2'])
            observed = [b['observation'] if b['region'] != 'SR' else sum(
                r['nominal'][i] * r['initial'] for (cc,p), r in data.items() if cc == c and r['family'] != 'signal')
                for i, b in enumerate(bins)]
            put('data_obs', observed, observed)
        f.Close()


def verify_templates(output, channels, data):
    """Read back every written TH1 and observation, not just file existence."""
    import uproot
    output = internal_path(output)
    for filename in sorted({template_name(c) for c in channels}):
        with uproot.open(internal_path(output / filename), object_cache=None, array_cache=None) as source:
            for c, bins in channels.items():
                if template_name(c) != filename:
                    continue
                def check(name, values, variances):
                    h = source[c + '/' + name]
                    actual, variance = h.values(), h.variances()
                    if (actual.shape != (len(bins),) or variance is None
                            or not np.allclose(actual, values, rtol=1e-12, atol=1e-12)
                            or not np.allclose(variance, variances, rtol=1e-12, atol=1e-12)
                            or h.axis().labels() != [b['name'] for b in bins]):
                        raise ValueError('TH1 integrity mismatch: ' + c + '/' + name)
                for (channel, process), record in data.items():
                    if channel != c:
                        continue
                    check(process, record['nominal'], record['sumw2'])
                    for name, pair in record['variations'].items():
                        for direction, suffix in (('up', 'Up'), ('down', 'Down')):
                            check(process + '_' + name + suffix, pair[direction], record['sumw2'])
                observed = [b['observation'] if b['region'] != 'SR' else sum(
                    r['nominal'][i] * r['initial'] for (cc, p), r in data.items() if cc == c and r['family'] != 'signal')
                    for i, b in enumerate(bins)]
                check('data_obs', observed, observed)


def json_ready(value):
    """Serialize numerical model records without pickle or an external loader."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
