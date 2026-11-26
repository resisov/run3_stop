"""Combine six eras into eight likelihood channels with shared CR rates."""
from collections import OrderedDict
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import tempfile
from TROTASR.utils.paths import internal_path
from TROTASR.utils.io import read_json, write_json
from TROTASR.utils.cr_binning import N_HIGH_CR, N_RATES, N_BINS


ERAS = ('2022', '2022EE', '2023', '2023BPix', '2024', '2025')
CHANNEL_BINS = {'SR_highdm_c1': 118, 'SR_lowdm_c1': 30,
    'LLCR_highdm_c1': N_HIGH_CR, 'LLCR_lowdm_c1': 10,
    'QCDCR_highdm_c1': N_HIGH_CR, 'QCDCR_lowdm_c1': 10,
    'GCR_highdm_c1': N_HIGH_CR, 'GCR_lowdm_c1': 10}
RATE_NAME = re.compile(r'^(CMS_NPS26012_(?:qcd_norm|ll_norm|sgamma_shape)_.+)_(2022EE|2023BPix|2022|2023|2024|2025)$')
HIGHDM_Z_SHAPE = 'highdm_z_shape_coefficients'
HIGHDM_PHOTON = re.compile(r'CMS_NPS26012_sgamma_shape_highdm_(Nb1|Nb2plus)_bin[0-4]')


def highdm_z_shape_group(parameter):
    match = HIGHDM_PHOTON.fullmatch(parameter)
    return match[1] if match else None


def merged_channel(channel):
    head, separator, era = channel.rpartition('_')
    if not separator or era not in ERAS or head not in CHANNEL_BINS:
        raise ValueError('Unexpected benchmark channel: ' + channel)
    return head, era


def parse_accepted_card(text):
    """Parse the supported Combine card syntax."""
    rows = [line.split() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    bins = [row[1:] for row in rows if row[0] == 'bin']
    processes = [row[1:] for row in rows if row[0] == 'process']
    observations = [row[1:] for row in rows if row[0] == 'observation']
    rates = [row[1:] for row in rows if row[0] == 'rate']
    if len(bins) != 2 or len(processes) != 2 or len(observations) != 1 or len(rates) != 1:
        raise ValueError('Unsupported accepted-card header')
    channels, columns = bins
    names, ids = processes
    if len(channels) != 48 or len(set(channels)) != 48:
        raise ValueError('Benchmark requires the accepted 48-channel six-era card')
    if any(value != '-1' for value in observations[0]) or len(observations[0]) != 48:
        raise ValueError('Expected histogram observations, not inline observed counts')
    if not len(columns) == len(names) == len(ids) == len(rates[0]):
        raise ValueError('Inconsistent card columns')
    if set(columns) != set(channels) or len(set(zip(columns, names))) != len(columns):
        raise ValueError('Missing channel or duplicate process column')
    expected = {head + '_' + era for head in CHANNEL_BINS for era in ERAS}
    if set(channels) != expected:
        raise ValueError('Unexpected source channel coverage')
    if any(value != '-1' for value in rates[0]):
        raise ValueError('Expected native histogram rates')
    ids = [int(value) for value in ids]
    if any((name == 'signal') != (pid <= 0) for name, pid in zip(names, ids)):
        raise ValueError('Unexpected source signal identity')
    shapes = [row for row in rows if row[0] == 'shapes']
    if not shapes or any(len(row) != 6 for row in shapes):
        raise ValueError('Expected explicit nominal/systematic histogram shape rules')
    tail = []
    for row in rows:
        if row[0] in ('imax', 'jmax', 'kmax', 'shapes', 'bin', 'observation', 'process', 'rate') or set(row[0]) == {'-'}:
            continue
        if len(row) < 2 or row[1] not in ('shape', 'lnN', 'rateParam', 'autoMCStats'):
            raise ValueError('Unsupported card directive: ' + row[0])
        if row[1] in ('shape', 'lnN') and len(row) != len(columns) + 2:
            raise ValueError('Malformed nuisance row: ' + row[0])
        tail.append(row)
    auto = [row for row in tail if row[1] == 'autoMCStats']
    if auto != [['*', 'autoMCStats', '10', '1', '1']]:
        raise ValueError('The approved autoMCStats directive must be unchanged')
    return dict(channels=channels, columns=columns, names=names, ids=ids,
                rates=rates[0], shapes=shapes, tail=tail)


@lru_cache(maxsize=16384)
def _shape_file(parent, relative):
    # Delivered template identities are immutable during this card-stage handoff.
    # Resolve each shared path once, not once per process/point column.
    source = internal_path(Path(parent)/relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def source_shape(card, rules, channel, process):
    """Resolve a card rule without opening or hashing its immutable ROOT file."""
    matches = [row for row in rules if row[1] in ('*', process) and row[2] in ('*', channel)]
    if not matches:
        raise ValueError('No source shape binding: ' + channel + '/' + process)
    rank = lambda row: (int(row[1] == process), int(row[2] == channel))
    best = max(rank(row) for row in matches)
    matches = [row for row in matches if rank(row) == best]
    if len(matches) != 1:
        raise ValueError('Ambiguous source shape binding')
    row = matches[0]
    source = _shape_file(str(card.parent), row[3])
    def resolve(pattern):
        value = pattern.replace('$CHANNEL', channel).replace('$PROCESS', process)
        if '$' in value.replace('$SYSTEMATIC', ''):
            raise ValueError('Unrecognized source template substitution')
        return value
    nominal, systematic = resolve(row[4]), resolve(row[5])
    if '$SYSTEMATIC' in nominal or '$SYSTEMATIC' not in systematic:
        raise ValueError('Unexpected template nominal/systematic pattern')
    return source, nominal, systematic


def merged_card_plan(card, text, output):
    parsed = parse_accepted_card(text)
    new_channels = list(OrderedDict.fromkeys(merged_channel(c)[0] for c in parsed['channels']))
    renamed = {(c, p): p + '_' + merged_channel(c)[1]
               for c, p in zip(parsed['columns'], parsed['names'])}
    signal_names = list(OrderedDict.fromkeys(renamed[c, p]
        for c, p, pid in zip(parsed['columns'], parsed['names'], parsed['ids']) if pid <= 0))
    background_names = list(OrderedDict.fromkeys(renamed[c, p]
        for c, p, pid in zip(parsed['columns'], parsed['names'], parsed['ids']) if pid > 0))
    ids = {name: -index for index, name in enumerate(signal_names)}
    ids.update({name: index + 1 for index, name in enumerate(background_names)})
    new_columns = [(merged_channel(c)[0], renamed[c, p])
                   for c, p in zip(parsed['columns'], parsed['names'])]
    if len(set(new_columns)) != len(new_columns):
        raise ValueError('Era-tagged internal process collision')
    data_root = output.parent / 'era_merged_data_obs.root'
    lines = ['imax 8', 'jmax *', 'kmax *', '------------']
    bindings = []
    for c, p in zip(parsed['columns'], parsed['names']):
        root, nominal, systematic = source_shape(card, parsed['shapes'], c, p)
        target_c, target_p = merged_channel(c)[0], renamed[c, p]
        path = os.path.relpath(root, output.parent)
        if any(character.isspace() for character in path):
            raise ValueError('Whitespace in referenced ROOT path')
        lines.append('shapes {} {} {} {} {}'.format(target_p, target_c, path, nominal, systematic))
        bindings.append(dict(source_channel=c, source_process=p,
            target_channel=target_c, target_process=target_p,
            root=str(root), nominal=nominal, systematic=systematic))
    for c in new_channels:
        lines.append('shapes data_obs {} {} {}/data_obs'.format(c, data_root.name, c))
    rate_line_index = len(lines) + 7
    lines += ['------------', 'bin ' + ' '.join(new_channels),
        'observation ' + ' '.join('-1' for c in new_channels), '------------',
        'bin ' + ' '.join(c for c, p in new_columns),
        'process ' + ' '.join(p for c, p in new_columns),
        'process ' + ' '.join(str(ids[p]) for c, p in new_columns),
        'rate ' + ' '.join(parsed['rates']), '------------']
    shared_rates, old_rates, targets = {}, set(), set()
    rate_map, rate_columns, tail = [], {}, []
    source_initials, representatives = {}, {}
    for row in parsed['tail']:
        if row[1] != 'rateParam':
            tail.append(row)
            continue
        if len(row) != 6 or (row[2], row[3]) not in renamed:
            raise ValueError('Unexpected source rateParam binding')
        match = RATE_NAME.fullmatch(row[0])
        if not match or match[2] != merged_channel(row[2])[1]:
            raise ValueError('Unexpected era-qualified normalization name')
        name = match[1]
        initial = float(row[4])
        if not math.isfinite(initial) or not 0.01 <= initial <= 5 or row[5] != '[0.01,5]':
            raise ValueError('Unexpected source normalization initial value/bounds: ' + row[0])
        prior_bounds = shared_rates.get(name, (0., math.inf))
        bounds = max(prior_bounds[0], 0.01 / initial), min(prior_bounds[1], 5. / initial)
        if not bounds[0] < bounds[1] or not bounds[0] <= 1. <= bounds[1]:
            raise ValueError('Shared residual coefficient has no common admissible interval')
        shared_rates[name] = bounds
        if row[0] in source_initials and source_initials[row[0]] != initial:
            raise ValueError('A source normalization has inconsistent initial values')
        source_initials[row[0]] = initial
        if name not in representatives or initial == 1.:
            representatives[name] = initial
        old_rates.add(row[0])
        channel, process = merged_channel(row[2])[0], renamed[row[2], row[3]]
        target = name, channel, process
        if target in targets:
            raise ValueError('Duplicate merged normalization target')
        targets.add(target)
        column = (row[2], row[3])
        if column in rate_columns:
            raise ValueError('Multiple free normalization factors on one column are unsupported')
        rate_columns[column] = dict(parameter=name, original_parameter=row[0],
            original_initial=initial, original_bounds=[0.01, 5.],
            target_channel=channel, target_process=process)
        rate_map.append(dict(source=row[0], target=name, source_channel=row[2],
                             source_process=row[3], target_channel=channel, process=process,
                             original_initial=initial))
    if len(old_rates) != len(ERAS)*N_RATES or len(shared_rates) != N_RATES:
        raise ValueError('Unexpected normalization inventory for the pooled high-dM tail')
    # Declare every floating alpha first; dependent expressions are not new
    # floating parameters. alpha/a_ref is the shared residual beta.
    direct_rows, formula_rows, alpha_bounds = [], [], {}
    for item in rate_map:
        name, initial = item['target'], item['original_initial']
        reference = representatives[name]
        beta_lo, beta_hi = shared_rates[name]
        lo, hi = reference * beta_lo, reference * beta_hi
        alpha_bounds[name] = (lo, hi)
        item.update(reference_initial=reference, residual_initial=1.,
                    alpha_bounds=[lo, hi], residual_bounds=[beta_lo, beta_hi])
        prefix = [name, 'rateParam', item['target_channel'], item['process']]
        if initial == reference:
            direct_rows.append(prefix + ['{:.17g}'.format(reference), '[{:.17g},{:.17g}]'.format(lo, hi)])
            item.update(binding_kind='direct_shared_alpha', effective_name=name)
        else:
            expression_name = name + '__eraScale_' + str(len(formula_rows) + 1)
            ratio = initial / reference
            formula = '(@0*{:.17g})'.format(ratio)
            formula_rows.append([expression_name, 'rateParam', item['target_channel'], item['process'], formula, name])
            item.update(binding_kind='dependent_formula', effective_name=expression_name,
                        formula=formula, arguments=[name], scale_ratio=ratio)
    declared = {row[0] for row in direct_rows}
    if declared != set(shared_rates) or len({row[0] for row in formula_rows}) != len(formula_rows):
        raise ValueError('Missing shared coefficient or duplicate dependent expression')
    if declared.intersection(row[0] for row in formula_rows):
        raise ValueError('Floating/dependent normalization name collision')
    lines += [' '.join(row) for row in tail if row[1] != 'autoMCStats']
    lines += [' '.join(row) for row in direct_rows + formula_rows]
    lines += [' '.join(row) for row in tail if row[1] == 'autoMCStats']
    return dict(lines=lines, rate_line_index=rate_line_index, source=parsed,
        channels=new_channels, columns=new_columns, bindings=bindings,
        data_root=data_root, rate_map=rate_map, source_rate_count=len(old_rates),
        target_rate_count=len(shared_rates), signal_names=signal_names,
        residual_bounds=shared_rates, representative_initials=representatives,
        alpha_bounds=alpha_bounds, formula_count=len(formula_rows))


def highdm_z_shape_coefficients(plan, background):
    """Use the fitted SR columns' nominal, recoil-corrected Z yields as weights."""
    sums = {era: {nb: {} for nb in ('Nb1', 'Nb2plus')} for era in ERAS}
    for item in plan['rate_map']:
        name = item['target']
        nb = highdm_z_shape_group(name)
        if item['target_channel'] != 'SR_highdm_c1' or nb is None:
            continue
        channel, process = item['source_channel'], item['source_process']
        era = merged_channel(channel)[1]
        if not process.startswith(f'ZJet_{nb}_u'):
            raise ValueError('Unexpected high-dM Z transfer target: ' + process)
        value = math.fsum(background[channel, process]['nominal']) * item['original_initial']
        if not math.isfinite(value) or value < 0.:
            raise ValueError('Invalid nominal Z transfer yield: ' + process)
        group = sums[era][nb]
        group[name] = group.get(name, 0.) + value
    coefficients = {}
    for era, groups in sums.items():
        coefficients[era] = {}
        for nb, yields in groups.items():
            total = math.fsum(yields.values())
            if total <= 0.:
                raise ValueError(f'No nominal Z transfer support for {era}/{nb}')
            coefficients[era][nb] = {
                name: value / total / plan['representative_initials'][name]
                for name, value in sorted(yields.items()) if value > 0.
            }
    return coefficients


def highdm_z_shape_formula(parameter, scale, weights):
    arguments = [parameter] + [other for other in sorted(weights) if other != parameter]
    denominator = '+'.join(f'@{i}*{weights[arg]:.17g}'
                           for i, arg in enumerate(arguments) if arg in weights)
    return f'(@0*{scale:.17g}/({denominator}))', arguments


def normalize_highdm_z_rates(plan, coefficients):
    """Apply the fitted recoil ratios with the common SR Z scale divided out."""
    replacements = {}
    for item in plan['rate_map']:
        name, channel, process = item['target'], item['target_channel'], item['process']
        nb = highdm_z_shape_group(name)
        if channel != 'SR_highdm_c1' or nb is None:
            continue
        era = merged_channel(item['source_channel'])[1]
        weights = coefficients[era][nb]
        scale = item['original_initial'] / item['reference_initial']
        formula, arguments = highdm_z_shape_formula(name, scale, weights)
        effective = name + '__zShape_' + process
        replacements[channel, process] = [effective, 'rateParam', channel, process,
                                          formula, ','.join(arguments)]
        item.pop('scale_ratio', None)
        item.update(binding_kind='highdm_z_shape', effective_name=effective,
                    formula=formula, arguments=arguments)

    remaining, replaced, direct = [], set(), set()
    for line in plan['lines']:
        row = line.split()
        if len(row) == 6 and row[1] == 'rateParam':
            key = tuple(row[2:4])
            if key in replacements:
                replaced.add(key)
                plan['formula_count'] += row[5].startswith('[')
                continue
            if row[5].startswith('['):
                direct.add(row[0])
        remaining.append(line)
    arguments = {name for row in replacements.values() for name in row[5].split(',')}
    if not replacements or replaced != replacements.keys() or not arguments <= direct:
        raise ValueError('High-dM Z shape must be linked to the retained GCR parameters')
    remaining[-1:-1] = [' '.join(row) for row in replacements.values()]
    plan['lines'] = remaining


def sum_template_observations(plan, channels, background):
    """Use the template writer's inputs: measured CRs and MC-only SR placeholders."""
    sums, source_labels, by_channel = {}, {}, {}
    for (channel, _), record in background.items():
        if record['family'] != 'signal':
            by_channel.setdefault(channel, []).append(record)
    for channel in plan['source']['channels']:
        head, era = merged_channel(channel)
        bins = channels[channel]
        size = CHANNEL_BINS[head]
        labels = [str(row['name']) for row in bins]
        if len(bins) != size or not all(labels) or len(set(labels)) != size:
            raise ValueError('Unexpected ordinal bins: ' + channel)
        if head in source_labels and source_labels[head] != labels:
            raise ValueError('Era ordinal categories do not match: ' + channel)
        source_labels[head] = labels
        row = sums.setdefault(head, dict(values=[0.] * size, variances=[0.] * size, eras=[]))
        row['eras'].append(era)
        for index, bin_ in enumerate(bins):
            if head.startswith('SR_'):
                value = sum(r['nominal'][index] * r['initial'] for r in by_channel.get(channel, ()))
            else:
                value = bin_['observation']
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid observation template input: ' + channel)
            # The per-era writer sets GetBinError to sqrt(value); retain its
            # variance convention exactly, rather than switching to MC sumw2.
            error = math.sqrt(value)
            row['values'][index] += value
            row['variances'][index] += error * error
    if set(sums) != set(CHANNEL_BINS) or any(set(row['eras']) != set(ERAS) for row in sums.values()):
        raise ValueError('Incomplete era summation')
    return sums, source_labels


def write_observation_sum(path, sums, labels):
    import ROOT as root
    if path.exists():
        raise FileExistsError('Preserve existing benchmark output: ' + str(path))
    output = root.TFile(str(path), 'CREATE')
    if not output or output.IsZombie():
        raise OSError('Cannot create merged data_obs output')
    try:
        for channel, row in sums.items():
            output.mkdir(channel).cd()
            size = len(row['values'])
            hist = root.TH1D('data_obs', 'data_obs', size, 0., float(size))
            hist.Sumw2()
            for index, (value, variance, label) in enumerate(zip(row['values'], row['variances'], labels[channel]), 1):
                hist.SetBinContent(index, value)
                hist.SetBinError(index, math.sqrt(variance))
                hist.GetXaxis().SetBinLabel(index, label)
            if hist.Write() <= 0:
                raise OSError('Could not write merged observation histogram')
            hist.SetDirectory(0)
    finally:
        output.Close()


def cr_only_card(plan):
    """Derive the six observed-CR channels from the same merged likelihood."""
    rows = [line.split() for line in plan['lines'] if line.strip()]
    bins = [row[1:] for row in rows if row[0] == 'bin']
    processes = [row[1:] for row in rows if row[0] == 'process']
    rates = next(row[1:] for row in rows if row[0] == 'rate')
    channels = [c for c in bins[0] if not c.startswith('SR_')]
    if len(channels) != 6:
        raise ValueError('The merged CR card must contain six CR channels and no SR')
    selected = [i for i,(c,p) in enumerate(zip(bins[1],processes[1]))
                if c in channels and int(p) > 0]
    attached = {(bins[1][i],processes[0][i]) for i in selected}
    lines = ['imax 6','jmax *','kmax *','------------']
    lines += [' '.join(r) for r in rows if r[0] == 'shapes' and r[2] in channels]
    lines += ['------------','bin '+' '.join(channels),
              'observation '+' '.join('-1' for _ in channels),'------------',
              'bin '+' '.join(bins[1][i] for i in selected)]
    lines += ['process '+' '.join(row[i] for i in selected) for row in processes]
    lines += ['rate '+' '.join(rates[i] for i in selected),'------------']
    tail = []
    for row in rows:
        if len(row) < 2:
            continue
        if row[1] in ('shape','lnN'):
            values = [row[2+i] for i in selected]
            if any(v != '-' for v in values):
                tail.append(row[:2]+values)
        elif row[1] == 'rateParam' and (row[2],row[3]) in attached:
            tail.append(row)
        elif row[1] == 'autoMCStats':
            tail.append(row)
    declared = {r[0] for r in tail if r[1] == 'rateParam' and '@' not in r[4]}
    for row in tail:
        if row[1] == 'rateParam' and '@' in row[4] and not set(row[5].split(',')) <= declared:
            raise ValueError('CR-only derivation lost a shared normalization declaration')
    return '\n'.join(lines+[' '.join(r) for r in tail])+'\n'


DISPLAY_COUNTS = {'T2tt': 205, 'T2bW': 345, 'T2tb': 345}
SOURCE_COUNTS = {'T2tt': 225, 'T2bW': 365, 'T2tb': 365}
REFERENCE_POINT = ('T2tt', 'mStop1200_mLSP500')


def merge_accepted_grid(manifest_path):
    """Return an accepted era-merged receipt; never regenerate a past model."""
    manifest = read_json(internal_path(manifest_path))
    if (manifest.get('era_merged') is not True or manifest.get('status') != 'cards_ready'
            or manifest.get('sr_data_blinded') is not True):
        raise ValueError('A completed blinded era-merged card grid is required')
    return manifest


def displayed_points(points):
    """Select exactly the authorized mass window without dropping inventory rows."""
    selected, seen = [], set()
    source_counts = dict.fromkeys(SOURCE_COUNTS, 0)
    display_counts = dict.fromkeys(DISPLAY_COUNTS, 0)
    for point in points:
        model, mass = point['model'], point['mass']
        match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', mass)
        if model not in source_counts or not match or (model, mass) in seen:
            raise ValueError('Invalid or duplicate signal-grid point')
        seen.add((model, mass))
        source_counts[model] += 1
        stop, lsp = map(int, match.groups())
        if 600 <= stop <= 1800 and 1 <= lsp <= 1800:
            selected.append(point)
            display_counts[model] += 1
    if source_counts != SOURCE_COUNTS or display_counts != DISPLAY_COUNTS:
        raise ValueError('Keep the authorized 955-point inventory and 895-point display scope')
    return selected


def observation_bindings(card, parsed):
    return tuple((channel, *map(str, source_shape(card, parsed['shapes'], channel, 'data_obs')[:2]))
                 for channel in parsed['channels'])


def verify_observation_sum(path, sums, labels):
    """Validate the small shared observation file when its receipt is missing."""
    import uproot
    with uproot.open(path) as root:
        for channel, row in sums.items():
            hist = root[channel+'/data_obs']
            axis = hist.axis()
            variances = hist.variances()
            if (not hist.classname.startswith('TH1') or axis.labels() != labels[channel]
                    or list(axis.edges()) != list(range(len(row['values'])+1))
                    or list(hist.values()) != row['values'] or variances is None
                    or not all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                               for actual, expected in zip(variances, row['variances']))):
                raise ValueError('Merged observation histogram differs from current inputs: '+channel)


def prepare_merged_cards(path, text, metadata, channels, template_background):
    """Prepare shared observations from today's in-memory card ingredients."""
    if metadata.get('fit_ready') is not True or metadata.get('sr_data_blinded') is not True:
        raise ValueError('Era-merged cards require ready blinded template inputs')
    plan = merged_card_plan(path, text, path)
    z_shape = highdm_z_shape_coefficients(plan, template_background)
    normalize_highdm_z_rates(plan, z_shape)
    bindings = observation_bindings(path, plan['source'])
    target = plan['data_root']
    receipt_path = target.parent.parent/'observation_construction.json'
    receipt = dict(scope='era_merged_template_observations',
        template_inputs=metadata['inputs'], data_obs_root=str(target),
        source_channels=48, target_channels=8, sr_data_blinded=True, observed_sr_used=False,
        construction='sum the CR observations and the native MC-only SR placeholders')
    if receipt_path.is_file():
        saved = read_json(receipt_path)
        fields = ('scope', 'source_channels', 'target_channels',
                  'sr_data_blinded', 'observed_sr_used')
        if any(saved.get(field) != receipt[field] for field in fields):
            raise ValueError('Merged observation layout or blinding differs from the existing receipt')
    elif target.exists():
        sums, labels = sum_template_observations(plan, channels, template_background)
        verify_observation_sum(target, sums, labels)
        write_json(receipt_path, receipt)
    else:
        write_json(receipt_path, receipt)
    if not target.exists():
        sums, labels = sum_template_observations(plan, channels, template_background)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.observations-', dir=target.parent) as tmp:
            pending = Path(tmp)/target.name
            write_observation_sum(pending, sums, labels)
            if target.exists():
                raise FileExistsError('Preserve existing merged observations: ' + str(target))
            pending.rename(target)
    if not target.is_file() or not target.stat().st_size:
        raise ValueError('Merged observation ROOT is missing or empty')
    return dict(observation_bindings=bindings, observation_receipt=str(receipt_path),
                data_obs_root=str(target), reference_plan=plan, **{HIGHDM_Z_SHAPE: z_shape})


def merged_card_metadata(metadata, merge):
    """Common provenance for new and already completed canonical card pairs."""
    return dict(metadata, era_merged=True, official_result=metadata.get('official_result', True),
        comparison_only=False,
        observed_sr_used=False, rate_parameters=N_RATES, floating_normalization_parameters=N_RATES,
        source_channels=48, source_bins=len(ERAS)*N_BINS, source_rate_parameters=len(ERAS)*N_RATES,
        target_channels=8, target_bins=N_BINS, target_rate_parameters=N_RATES,
        normalization_model='shared alpha with era-dependent scales and high-dM Z shape-only transfer',
        **{HIGHDM_Z_SHAPE: merge[HIGHDM_Z_SHAPE]},
        original_histograms_and_nuisance_functions_preserved=True,
        data_obs_root=merge['data_obs_root'], data_obs_construction_receipt=merge['observation_receipt'])


def render_merged_signal_card(path, text, metadata, merge):
    """Transform one native source text without publishing any unmerged card."""
    if metadata.get('fit_ready') is not True or metadata.get('sr_data_blinded') is not True:
        raise ValueError('Only ready blinded cards can be merged')
    plan = merged_card_plan(path, text, path)
    normalize_highdm_z_rates(plan, merge[HIGHDM_Z_SHAPE])
    if observation_bindings(path, plan['source']) != merge['observation_bindings']:
        raise ValueError('Observation references differ; do not assume common data')
    model = merged_card_metadata(metadata, merge)
    model.update(dependent_rateparam_formulas=plan['formula_count'],
        shared_alpha_bounds={n:list(v) for n,v in plan['alpha_bounds'].items()})
    return '\n'.join(plan['lines'])+'\n', model


def publication_metadata(merge):
    return dict(era_merged=True, canonical_model='era_merged', official_result=True,
        comparison_only=False, bins=N_BINS, source_bins=len(ERAS)*N_BINS, likelihood_channels=8,
        source_likelihood_channels=48, rate_parameters=N_RATES, floating_normalization_parameters=N_RATES,
        displayed_model_points=sum(DISPLAY_COUNTS.values()), merged_points_by_model=DISPLAY_COUNTS,
        th1_channels=CHANNEL_BINS, data_obs_construction_receipt=merge['observation_receipt'],
        **{HIGHDM_Z_SHAPE: merge[HIGHDM_Z_SHAPE]})



