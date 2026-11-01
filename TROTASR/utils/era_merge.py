"""Canonical six-era merged-channel likelihood; adopted benchmark algebra unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from collections import OrderedDict
import copy
import fcntl
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import time
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256


ERAS = ('2022', '2022EE', '2023', '2023BPix', '2024', '2025')
CHANNEL_BINS = {'SR_highdm_c1': 118, 'SR_lowdm_c1': 30,
    'LLCR_highdm_c1': 12, 'LLCR_lowdm_c1': 10,
    'QCDCR_highdm_c1': 12, 'QCDCR_lowdm_c1': 10,
    'GCR_highdm_c1': 12, 'GCR_lowdm_c1': 10}
RATE_NAME = re.compile(r'^(CMS_NPS26012_(?:qcd_norm|ll_norm|sgamma_shape)_.+)_(2022EE|2023BPix|2022|2023|2024|2025)$')


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
                             target_channel=channel, process=process, original_initial=initial))
    if len(old_rates) != 288 or len(shared_rates) != 48:
        raise ValueError('Benchmark requires exactly 288 to 48 normalization names')
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


def sum_accepted_observations(card, plan):
    """Sum the 48 source data_obs histograms."""
    import ROOT as root
    root.gROOT.SetBatch(True)
    files, sums, source_labels = {}, {}, {}
    def source_file(path):
        if path not in files:
            files[path] = root.TFile.Open(str(path), 'READ')
            if not files[path] or files[path].IsZombie():
                raise OSError('Cannot open accepted source: ' + str(path))
        return files[path]
    try:
        for channel in plan['source']['channels']:
            head, era = merged_channel(channel)
            path, nominal, unused = source_shape(card, plan['source']['shapes'], channel, 'data_obs')
            hist = source_file(path).Get(nominal)
            if not hist or not hist.InheritsFrom('TH1') or hist.GetDimension() != 1:
                raise ValueError('Missing approved data_obs TH1: ' + channel)
            size = int(hist.GetNbinsX())
            if size != CHANNEL_BINS[head]:
                raise ValueError('Unexpected ordinal-bin count: ' + channel)
            labels = [str(hist.GetXaxis().GetBinLabel(i)) for i in range(1, size + 1)]
            if not all(labels) or len(set(labels)) != size:
                raise ValueError('Missing or duplicate ordinal labels: ' + channel)
            if head in source_labels and source_labels[head] != labels:
                raise ValueError('Era ordinal categories do not match: ' + channel)
            source_labels[head] = labels
            row = sums.setdefault(head, dict(values=[0.] * size, variances=[0.] * size, eras=[]))
            row['eras'].append(era)
            for index in range(size):
                value, error = float(hist.GetBinContent(index + 1)), float(hist.GetBinError(index + 1))
                if not math.isfinite(value) or value < 0 or not math.isfinite(error) or error < 0:
                    raise ValueError('Invalid approved observation histogram')
                row['values'][index] += value
                row['variances'][index] += error * error
        if set(sums) != set(CHANNEL_BINS) or any(set(row['eras']) != set(ERAS) for row in sums.values()):
            raise ValueError('Incomplete era summation')
        return sums, source_labels
    finally:
        for source in files.values():
            if source:
                source.Close()


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


def merge_accepted_grid(manifest_path, lock_held=False):
    """Build the era-merged likelihood grid from the existing signal cards.

    The 895 displayed points are rewritten from delivered cards only. Original
    templates, nuisance functions and blinded observations are not recomputed.
    """
    manifest_path = internal_path(manifest_path)
    grid, analysis = manifest_path.parent, manifest_path.parents[3]
    old = read_json(manifest_path)
    if old.get('era_merged') is True:
        return old
    if old.get('status') != 'cards_ready' or not old.get('run3_systematics') or old.get('sr_data_blinded') is not True:
        raise ValueError('A completed blinded six-era card grid is required')
    history = grid/'history/separate_eras'
    stage = grid/'datacards.next'
    cards = grid/'datacards'
    evidence_path = grid/'observation_construction.json'
    reference_point = next(p for p in old['points']
        if p['model'] == 'T2tt' and p['mass'] == 'mStop1200_mLSP500')
    reference = internal_path(analysis/reference_point['card'])
    reference_text = reference.read_text()
    if hashlib.sha256(reference_text.encode()).hexdigest() != reference_point['card_sha256']:
        raise ValueError('Accepted reference card changed')
    reference_parsed = parse_accepted_card(reference_text)
    reference_plan = merged_card_plan(reference,reference_text,cards/reference.name)
    def observations(card, parsed):
        return tuple((c, str(source_shape(card, parsed['shapes'], c, 'data_obs')[0]),
                      source_shape(card, parsed['shapes'], c, 'data_obs')[1]) for c in parsed['channels'])
    observation_identity = observations(reference, reference_parsed)
    with (nullcontext() if lock_held else (grid/'builder.lock').open('a')) as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (history/'datacards').exists():
            raise RuntimeError('Prior card handoff exists; inspect it rather than overwrite')
        stage.mkdir(exist_ok=True)
        observation_copy = stage/'era_merged_data_obs.root'
        if not observation_copy.exists():
            sums, labels = sum_accepted_observations(reference, reference_plan)
            write_observation_sum(observation_copy, sums, labels)
        write_json(evidence_path,dict(scope='accepted_card_era_merge_observations',
            source_card=str(history/'datacards'/reference.name),
            source_card_sha256=reference_point['card_sha256'],
            data_obs_root=str(cards/'era_merged_data_obs.root'),sr_data_blinded=True,
            observed_sr_used=False,source_channels=48,target_channels=8,
            construction='canonical sum_accepted_observations/write_observation_sum',
            historical_code_or_benchmark_dependency=False))
        selected = []
        for point in old['points']:
            match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', point['mass'])
            stop, lsp = map(int, match.groups())
            if 600 <= stop <= 1800 and 1 <= lsp <= 1800:
                selected.append(point)
        counts = {m:sum(p['model']==m for p in selected) for m in ('T2tt','T2bW','T2tb')}
        if counts != dict(T2tt=205,T2bW=345,T2tb=345):
            raise ValueError('Keep the authorized 895-point display scope')
        def convert(point):
            original = internal_path(analysis/point['card'])
            text = original.read_text()
            if hashlib.sha256(text.encode()).hexdigest() != point['card_sha256']:
                raise ValueError('Delivered card receipt mismatch: '+point['mass'])
            final = cards/original.name
            output = stage/original.name
            meta = read_json(original.with_suffix('.manifest.json'))
            if meta.get('fit_ready') is not True or meta.get('sr_data_blinded') is not True:
                raise ValueError('Only accepted blinded cards can be merged')
            plan = merged_card_plan(original, text, final)
            if observations(original, plan['source']) != observation_identity:
                raise ValueError('Observation references differ; do not assume common data')
            payload = '\n'.join(plan['lines'])+'\n'
            card_hash = hashlib.sha256(payload.encode()).hexdigest()
            if output.exists():
                if sha256(output) != card_hash:
                    raise ValueError('Retained new card differs')
            else:
                with output.open('x') as stream:
                    stream.write(payload)
            meta.update(card_sha256=card_hash, era_merged=True, official_result=True,
                source_card=str(history/'datacards'/original.name), source_card_sha256=point['card_sha256'],
                source_channels=48,target_channels=8,source_bins=1284,target_bins=214,
                source_rate_parameters=288,target_rate_parameters=48,rate_parameters=48,
                floating_normalization_parameters=48,dependent_rateparam_formulas=plan['formula_count'],
                normalization_model='shared alpha with preserved era-dependent rateParam formula scales',
                shared_alpha_bounds={n:list(v) for n,v in plan['alpha_bounds'].items()},
                original_histograms_and_nuisance_functions_preserved=True,
                data_obs_root=str(cards/'era_merged_data_obs.root'),
                data_obs_construction_receipt=str(evidence_path),
                observed_sr_used=False,comparison_only=False,production_model_replaced=True,
                input_check_scope='accepted small card/receipt and exact observation-reference identity; no template ROOT reads/hashes')
            write_json(output.with_suffix('.manifest.json'),meta)
            return dict(point,card=str(final.relative_to(analysis)),card_sha256=card_hash,era_merged=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            converted=list(pool.map(convert,selected))
        # Construct the CR-only card from the merged model, excluding SR
        # observations.
        cr_source=cards/'datacard_cronly.txt'
        if not cr_source.is_file():
            raise FileNotFoundError('Accepted CR-only source card is missing')
        cr_payload=cr_only_card(reference_plan)
        (stage/'datacard_cronly.txt').write_text(cr_payload)
        cr_hash=hashlib.sha256(cr_payload.encode()).hexdigest()
        cr_meta=read_json(cr_source.with_suffix('.manifest.json'))
        cr_meta.update(status='card_written_not_fitted',
            fit_ready=True,cr_only=True,era_merged=True,official_result=True,
            card_sha256=cr_hash,sr_data_blinded=True,observed_sr_used=False,
            source_card=str(cr_source),auto_mc_stats=[10,1,1])
        write_json(stage/'datacard_cronly.manifest.json',cr_meta)
        mapping={p['model']+'/'+p['mass']:p for p in converted}
        updated=[]
        for point in old['points']:
            key=point['model']+'/'+point['mass']
            if key in mapping:
                updated.append(mapping[key])
            else:
                updated.append(dict(point,card=str((history/'datacards'/Path(point['card']).name).relative_to(analysis)),
                                    fit_ready=False,excluded_out_of_plot_range=True))
        final=copy.deepcopy(old)
        final.update(era_merged=True,official_result=True,comparison_only=False,
            production_model_replaced=True,bins=214,source_bins=1284,
            likelihood_channels=8,source_likelihood_channels=48,
            floating_normalization_parameters=48,points=updated,
            canonical_model='era_merged',model_adopted=time.time(),
            displayed_model_points=895,merged_points_by_model=counts,
            th1_channels=CHANNEL_BINS,
            cr_only_card=dict(card=str((cards/'datacard_cronly.txt').relative_to(analysis)),card_sha256=cr_hash),
            previous_model='separate_eras',previous_model_manifest=str(history/'manifest.json'),
            data_obs_construction_receipt=str(evidence_path),
            input_check_scope='accepted card-stage handoff; no upstream production validation or immutable ROOT hashing')
        history.mkdir(parents=True,exist_ok=True)
        shutil.copy2(manifest_path,history/'manifest.json')
        if (grid/'state.json').exists():
            shutil.copy2(grid/'state.json',history/'state.json')
        # Preserve original card-relative template references inside history.
        if not (history/'templates').exists():
            (history/'templates').symlink_to(grid/'templates',target_is_directory=True)
        cards.rename(history/'datacards')
        stage.rename(cards)
        write_json(manifest_path,final)
        write_json(grid/'state.json',{k:v for k,v in final.items() if k!='points'})
        print('canonical_era_merged_cards_ready',len(converted),counts,flush=True)
        return final



