"""Build shared native TH1 templates and the three nominal signal-card grids.

Large histograms are read once per year. Background TH1s are written once;
each signal mass point has a distinct named TH1 in the same four files, as in
the adopted legacy grid builder. All paths and runtime analysis inputs remain
internal. No old fit values or workspaces are consumed.
"""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import re
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.statistical_templates import (assemble, aggregate, card_text as native_card_text, template_name,
    write_templates, verify_templates, json_ready, load_theory, profiled_theory, add_signal_xsec_lnN)
from TROTASR.workflows.signal_grid_inputs import (MODELS, FIELDS, requested_grid,
    mass_pair, extract_year, signal_data, require_fit_support_issues)
from TROTASR.utils.sr_merge import load_merge, bin_groups, project

CODE = ('workflows/build_nominal_grid.py', 'workflows/signal_grid_inputs.py',
    'utils/statistics_inputs.py', 'utils/statistics_model.py', 'utils/statistical_templates.py',
    'utils/sr_binning.py', 'utils/sr_merge.py', 'utils/renderers/signal_theory.py')

CR12_POLICY = dict(id='cr12_shapes_20260929', user_approved=True,
    rate_parameter_range=[0.01, 5], highdm_cr_bins_per_region_year=12,
    native_recoil_groups=[[0], [1], [2], [3], [4, 5], [6, 7]],
    recoil_edges=[250, 300, 350, 400, 500, 800, 1500], last_bin_open_ended=True,
    signal_contamination=False, auto_mc_stats=[10, 1, 1],
    preserve_sr_bins=[118, 30], preserve_existing_uncertainty_correlations=True)


def require_statistical_revision(policy):
    if policy is not None and policy != CR12_POLICY:
        raise ValueError('Not the exact user-approved CR12/shape statistical revision')
    return policy is not None


def cr12_parameter(name):
    match = re.fullmatch(r'(CMS_NPS26012_(?:ll|qcd)_norm_highdm_Nb(?:1|2plus)_bin)([0-7])(_202[45])', name)
    return match[1]+str((0, 1, 2, 3, 4, 4, 5, 5)[int(match[2])])+match[3] if match else name


def cr12_payload(payload):
    """User-adopted signed native-CR sum, before clipping; no SR changes."""
    result = dict(year=payload['year'], groups={})
    for region in ('LLCR', 'QCDCR', 'GCR'):
        old = payload['groups'][region+'_highdm']
        if len(old) != 16:
            raise ValueError('Expected eight native CR recoil cells per Nb')
        rows = []
        for nb in ('Nb1', 'Nb2plus'):
            by_index = {int(b['name'].rsplit('bin', 1)[1]): b for b in old if b['nb_group'] == nb}
            if set(by_index) != set(range(8)):
                raise ValueError('Incomplete native CR cell coverage')
            for index, group in enumerate(CR12_POLICY['native_recoil_groups']):
                members = [by_index[i] for i in group]
                row = copy.deepcopy(members[0])
                row.update(name=f'{region}_highdm_{nb}_bin{index}',
                    observation=sum(b['observation'] for b in members), backgrounds={},
                    rate_params={}, rate_initial={}, extra_lnN={}, physical={})
                for process in sorted({p for b in members for p in b['backgrounds']}):
                    present = [b for b in members if process in b['backgrounds']]
                    bindings = {(cr12_parameter(b['rate_params'].get(process, '')),
                        b['rate_initial'].get(process, 1.),
                        tuple((e['name'], e['down'], e['up']) for e in b['extra_lnN'].get(process, [])))
                        for b in present}
                    if len(bindings) != 1:
                        raise ValueError('Cannot sum different CR normalization/error bindings')
                    par, initial, extras = next(iter(bindings))
                    record = aggregate([(b['backgrounds'][process], []) for b in present], preserve_flat=True)
                    row['backgrounds'][process] = dict(nominal=[record['nominal']], sumw2=[record['sumw2']],
                        variations={n: {d: [v] for d, v in pair.items()} for n, pair in record['variations'].items()})
                    if par:
                        row['rate_params'][process] = par
                        row['rate_initial'][process] = initial
                    if extras:
                        row['extra_lnN'][process] = [dict(name=n, down=d, up=u) for n, d, u in extras]
                rows.append(row)
        result['groups'][region+'_highdm'] = rows
    return result


def card_text(channels, data, cr_only=False, statistical_revision=None):
    text = native_card_text(channels, data, cr_only)
    if not require_statistical_revision(statistical_revision):
        return text
    lines = text.splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) > 1 and fields[1] == 'rateParam':
            if len(fields) != 6 or fields[5] != '[0,10]' or not 0.01 <= float(fields[4]) <= 5:
                raise ValueError('Unexpected free-normalization declaration/initial value')
            fields[5] = '[0.01,5]'
            lines[i] = ' '.join(fields)
    if '* autoMCStats 10 1 1' not in lines:
        raise ValueError('autoMCStats changed')
    return '\n'.join(lines)+'\n'


def background_shape_names():
    """Exact names in the user-adopted JME background-error revision."""
    names = {'CMS_NPS26012_RZstat_'+m+'_'+b+'_'+y
        for m in ('highdm', 'lowdm') for b in ('Nb1', 'Nb2plus') for y in ('2024', '2025')}
    for mode, intervals in (
            ('highdm', ('250to300', '300to350', '350to400', '400to500',
                        '500to650', '650to800', '800to1000', '1000to1500')),
            ('lowdm', ('250to300', '300to350', '350to400', '400to500', '500toInf'))):
        names.update('CMS_NPS26012_zgammaNonclosure_'+mode+'_u'+i+'_'+y
                     for i in intervals for y in ('2024', '2025'))
    return names

def background_error_shapes(background, policy=None):
    """Convert measured factors AFTER assembly/clipping/projection.

    The factor-bearing binding keys must still be used during assembly: changing
    them earlier merges components and changes the adopted rate-parameter model.
    This transformation changes interpolation, not the measured +/-1 endpoints.
    Existing arrays, component identities, zero bins and nominal MC variances
    are retained. The shared writer's endpoint-variance convention is unchanged.
    """
    if policy is None:
        return background, None
    if policy != dict(id='jme_background_shapes_20260929', mode='shape',
                      auto_mc_stats=[10, 1, 1], user_approved=True):
        raise ValueError('Unapproved background shape policy')
    import numpy as np
    expected, seen, audit, result = background_shape_names(), set(), [], {}
    for (channel, process), record in background.items():
        extras = record['extra_lnN']
        names = [r[0] for r in extras]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate background uncertainty')
        converted = []
        variations = dict(record['variations'])
        for name, down, up in extras:
            if name not in expected:
                if name.startswith(('CMS_NPS26012_RZstat_', 'CMS_NPS26012_zgammaNonclosure_')):
                    raise ValueError('Unexpected background-error nuisance: '+name)
                continue
            if (not channel.startswith('SR_') or record['family'] != 'ZJet'
                    or name in variations or not name.endswith(channel[-4:])
                    or '_'+channel.split('_')[1]+'_' not in name
                    or not all(math.isfinite(v) and v > 0 for v in (down, up))):
                raise ValueError('Invalid or mixed-type background shape target: '+name)
            nominal = np.asarray(record['nominal'])
            if not np.isfinite(nominal).all() or (nominal < 0).any():
                raise ValueError('Background shapes require already-clipped finite nominal')
            pair = dict(down=nominal*down, up=nominal*up)
            if not all(np.isfinite(a).all() for a in pair.values()):
                raise ValueError('Nonfinite background shape endpoint')
            variations[name] = pair
            converted.append(name); seen.add(name)
            audit.append(dict(channel=channel, process=process, nuisance=name,
                down=down, up=up, bins=len(nominal), zero_bins=int(np.count_nonzero(nominal == 0))))
        result[channel, process] = dict(record, variations=variations,
            extra_lnN=tuple(r for r in extras if r[0] not in converted))
    if seen != expected:
        raise ValueError('Incomplete background shape coverage: '+str(sorted(expected-seen)))
    return result, dict(policy=policy, nuisance_names=sorted(seen), nuisances=len(seen),
        targets=audit, target_count=len(audit), nominal_and_sumw2_unchanged=True,
        component_and_rate_parameter_bindings_unchanged=True,
        existing_shape_endpoints_unchanged=True, measured_factors_unchanged=True,
        interpolation_changed=True, auto_mc_stats=[10, 1, 1])


def fit_background(background, issues, policy=None):
    """User-adopted preliminary exception: two SR columns, never ROOT content.

    This is a physics-scope restriction, not a zero-integral numerical repair.
    No other process, CR attachment, rate parameter, or endpoint is waived.
    """
    if policy is None:
        return background, issues, []
    expected = {('SR_highdm_c1_'+y, 'QCD_Nb1_u6'): v for y, v in (
        ('2024', .05757514512913886), ('2025', .0253831485651911))}
    if (policy.get('id') != 'sr_qcd_nb1_u6_20260928'
            or policy.get('user_approved') is not True
            or policy.get('preserve_all_root_objects') is not True
            or policy.get('auto_mc_stats') != [10, 1, 1]
            or len(policy.get('targets', [])) != 2
            or {(r['channel'], r['process']) for r in policy['targets']} != set(expected)):
        raise ValueError('Unapproved preliminary fit exclusion policy')
    audit = []
    for (channel, process), up in expected.items():
        record = background[channel, process]
        year = channel[-4:]; nuisance = 'CMS_scale_j_Total_'+year
        parameter = 'CMS_NPS26012_qcd_norm_highdm_Nb1_bin6_'+year
        cr = background['QCDCR_highdm_c1_'+year, process]
        if (record['family'] != 'QCD' or record['parameter'] != parameter
                or any(record['nominal']) or any(record['sumw2'])
                or nuisance not in record['variations']
                or any(any(a) for n,pair in record['variations'].items() if n != nuisance for a in pair.values())
                or any(record['variations'][nuisance]['down'])
                or not math.isclose(sum(record['variations'][nuisance]['up']), up, rel_tol=1e-12)
                or cr['parameter'] != parameter or sum(cr['nominal']) <= 0):
            raise ValueError('Approved zero-total SR/positive CR support changed')
        target_issues = [i for i in issues if (i.get('channel'), i.get('process')) == (channel, process)]
        if ({i['problem'] for i in target_issues} !=
                {'zero_integral_nonempty_component', 'zero_integral_variation'}
                or len(target_issues) != 2
                or any(i['problem'] == 'zero_integral_variation' and
                    (i.get('nuisance') != nuisance or i.get('direction') != 'down') for i in target_issues)):
            raise ValueError('Approved SR fit issue set changed')
        audit.append(dict(channel=channel, process=process, parameter=parameter,
            nominal=0., sumw2=0., omitted_variations={nuisance: dict(up=up, down=0.)},
            cr_nominal_preserved=float(sum(cr['nominal'])), root_objects_preserved=True,
            decision='user_approved_fit_only_exclusion', preliminary=True))
    return ({k: v for k, v in background.items() if k not in expected},
            [i for i in issues if (i.get('channel'), i.get('process')) not in expected], audit)


def original_exclusion_audit(background, issues, policy, combination=None):
    """Check the original JME exception without changing any stored endpoint.

    Only the two already-approved SR columns may carry additional non-JME
    endpoints. Their original nominal/JME/weight subset must still pass the
    unchanged strict exclusion check above. All other issues remain blocking.
    """
    if combination is None:
        return fit_background(background, issues, policy)[2]
    from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    from TROTASR.utils.statistics_model import nps_nuisance_name
    import numpy as np
    if combination != COMBINATION_POLICY or policy is None:
        raise ValueError('Unapproved combined exclusion scope')
    keys = {('SR_highdm_c1_'+y, 'QCD_Nb1_u6') for y in ('2024', '2025')}
    subset, additional, allowed = dict(background), {}, {}
    for key in keys:
        record = background[key]
        names = {nps_nuisance_name(e[:-2], key[0][-4:]) for e in NONJME_ENDPOINTS if e.endswith('Up')}
        allowed[key] = names
        if len(names) != 8:
            raise ValueError('Non-JME nuisance naming changed')
        extra = {}
        for name in sorted(names & set(record['variations'])):
            pair = record['variations'][name]
            if set(pair) != {'up', 'down'}:
                raise ValueError('Incomplete non-JME endpoint pair')
            for values in pair.values():
                a = np.asarray(values)
                if a.shape != np.asarray(record['nominal']).shape or not np.isfinite(a).all() or (a < 0).any():
                    raise ValueError('Invalid preserved non-JME endpoint array')
            extra[name] = {d: dict(integral=float(np.sum(a)), nonzero_bins=int(np.count_nonzero(a)))
                           for d, a in pair.items()}
        additional[key] = extra
        subset[key] = dict(record, variations={n: p for n, p in record['variations'].items() if n not in names})
    original_issues = [i for i in issues if not (
        (i.get('channel'), i.get('process')) in keys and i.get('problem') == 'zero_integral_variation'
        and i.get('nuisance') in allowed[i['channel'], i['process']]
        and i.get('direction') in ('up', 'down'))]
    audit = fit_background(subset, original_issues, policy)[2]
    for row in audit:
        row.update(original_jme_subset_verified=True,
            additional_nonjme_endpoints_preserved=additional[row['channel'], row['process']],
            absent_nonjme_nuisances=sorted(allowed[row['channel'], row['process']]
                - set(additional[row['channel'], row['process']])),
            absent_nonjme_arrays_not_fabricated=True,
            nominal_convention_mismatch_acknowledged=True)
    return audit


def cr12_fit_background(background, issues, original_audit):
    keys = {('SR_highdm_c1_'+y, 'QCD_Nb1_u6') for y in ('2024', '2025')}
    if len(original_audit) != 2 or {(r['channel'], r['process']) for r in original_audit} != keys:
        raise ValueError('Missing strict original JME exclusion audit')
    audit = copy.deepcopy(original_audit)
    for row in audit:
        record = background[row['channel'], row['process']]
        cr = background['QCDCR_highdm_c1_'+row['channel'][-4:], 'QCD_Nb1_u5']
        parameter = cr12_parameter(row['parameter'])
        if (record['family'] != 'QCD' or record['parameter'] != parameter
                or cr['family'] != 'QCD' or cr['parameter'] != parameter
                or any(record['nominal']) or any(record['sumw2']) or sum(cr['nominal']) <= 0):
            raise ValueError('CR12 mapped exception no longer has zero SR/positive CR support')
        row.update(original_parameter=row['parameter'], parameter=parameter,
            cr_nominal_preserved=float(sum(cr['nominal'])), cr_native_bins=[6, 7])
    return ({k: r for k, r in background.items() if k not in keys},
        [i for i in issues if (i.get('channel'), i.get('process')) not in keys], audit)


def audit_continuation(output, contract, config, receipt_path):
    """Strict reuse of written TH1s in a distinct, fit-only metadata directory.

    The receipt pins both source contracts and all cached inputs/ROOT hashes.
    The original builder is not stopped or modified. Only this builder version
    and the explicit fit-only config policy may differ.
    Full numerical/object-inventory readback remains mandatory afterwards.
    """
    receipt = read_json(internal_path(receipt_path))
    old = receipt['prior_contract']
    if receipt['new_contract'] != contract or receipt['output'] != str(output.relative_to(ROOT)):
        raise ValueError('Template continuation target/contract mismatch')
    source = internal_path(ROOT / receipt['source_output'])
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Read-only template reuse needs a distinct output directory')
    if read_json(source/'contract.json') != old:
        raise ValueError('Source template contract changed')
    old_config_path = internal_path(ROOT / receipt['prior_config'])
    if sha256(old_config_path) != old['config_sha256']:
        raise ValueError('Template continuation prior config changed')
    old_config = read_json(old_config_path)
    if {k: v for k, v in config.items() if k != 'preliminary_fit_exclusions'} != old_config:
        raise ValueError('Continuation changed physics inputs beyond approved exclusion')
    allowed = {'code', 'config_sha256'}
    if ({k: v for k, v in old.items() if k not in allowed} !=
            {k: v for k, v in contract.items() if k not in allowed}):
        raise ValueError('Continuation changed template input contract')
    builder = 'workflows/build_nominal_grid.py'
    if {k: v for k, v in old['code'].items() if k != builder} != {
            k: v for k, v in contract['code'].items() if k != builder}:
        raise ValueError('Continuation changed numerical template implementation')
    old_source = internal_path(ROOT / receipt['prior_builder'])
    if sha256(old_source) != old['code'][builder]:
        raise ValueError('Prior builder provenance changed')
    expected = {'template_'+m+'_'+y+'.root' for m in ('highdm','lowdm') for y in ('2024','2025')}
    if set(receipt['files']) != expected or {p.name for p in (source/'templates').glob('*.root')} != expected:
        raise ValueError('Continuation requires exactly the four existing ROOTs')
    for name, digest in receipt['files'].items():
        if sha256(source/'templates'/name) != digest:
            raise ValueError('Written ROOT changed before continuation')
    for year in ('2024', '2025'):
        path = source/'inputs'/(year+'.json.gz')
        cache = read_json(path.with_suffix('.receipt.json'))
        if cache != dict(contract=old, sha256=receipt['cache_sha256'][year]) or sha256(path) != cache['sha256']:
            raise ValueError('Prior cache contract/hash changed')
    return receipt


def grid_contract(config_path, config, inventory, validate_only, preserve_unsupported=False):
    revision = config.get('statistical_revision')
    require_statistical_revision(revision)
    if set(config['years']) != {'2024', '2025'}:
        raise ValueError('Exactly years 2024 and 2025 are required')
    inputs = {}
    for year in ('2024', '2025'):
        if set(config['years'][year]) != set(FIELDS):
            raise ValueError('Incomplete histogram/measurement configuration')
        inputs[year] = {key: dict(path=str(internal_path(ROOT / path).relative_to(ROOT)),
                                  sha256=sha256(ROOT / path))
                        for key, path in config['years'][year].items()}
    contract = dict(config_sha256=sha256(config_path), inputs=inputs,
        grid_sha256=sha256(inventory), theory_sha256=sha256(ROOT / config['theory']),
        code={name: sha256(Path(__file__) if name == 'workflows/build_nominal_grid.py' else ROOT / name) for name in CODE},
        sr_binning_sha256=sha256(ROOT / 'jsons/highdm_sr_binning.json'),
        gnn_sha256=sha256(ROOT / 'gnn4lowdm/config.json'), validate_only=validate_only,
        preserve_unsupported_templates=preserve_unsupported)
    if revision:
        contract['statistical_revision'] = revision
    if config.get('background_estimation_uncertainties'):
        contract['background_estimation_uncertainties'] = config['background_estimation_uncertainties']
    if config.get('systematic_combination'):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        if (config['systematic_combination'] != COMBINATION_POLICY or not revision
                or config.get('fit_campaign') != 'combined_systematics_preliminary'):
            raise ValueError('Missing exact acknowledged preliminary combination policy')
        contract['systematic_combination'] = config['systematic_combination']
        contract['code'].update({name: sha256(ROOT/name) for name in (
            'workflows/merge_nominal.py', 'utils/corrections.py')})
    return contract


def combine_cached(caches, sr_merge=None, fit_issues=None, statistical_revision=None,
                   exclusion_policy=None, systematic_combination=None, exclusion_audit=None):
    revised = require_statistical_revision(statistical_revision)
    channels, data, clipping, issues = assemble([caches[y]['background'] for y in ('2024', '2025')], preserve_flat=True)
    require_fit_support_issues(issues)
    if any(p == 'signal' for c, p in data):
        raise ValueError('Shared background model contains unsupported or signal components')
    if len(channels) != 16 or sum(map(len, channels.values())) != 540:
        raise ValueError('Keep all 540 analysis bins in 16 TH1 channels')
    for c in channels:
        if not any(cc == c for cc, p in data):
            raise ValueError('Channel has no background support: ' + c)
    sources = {(y, name): entries for y, cache in caches.items() for name, entries in cache['source_bins'].items()}
    if sr_merge:
        # Apply the already-adopted exact partition before the first ROOT write:
        # four final files, not a second set of unmerged ROOT templates.
        groups = bin_groups(load_merge(sr_merge))
        original_sources = dict(sources)
        for c, bins in list(channels.items()):
            if not c.startswith('SR_highdm_'):
                continue
            projected = []
            for i, members in enumerate(groups):
                row = copy.deepcopy(bins[members[0]])
                row['name'] = 'SR_highdm_bin'+str(i)
                sources[row['year'], row['name']] = [entry for j in members
                    for entry in original_sources[row['year'], bins[j]['name']]]
                projected.append(row)
            channels[c] = projected
            for (channel, process), record in data.items():
                if channel != c:
                    continue
                for field in ('nominal', 'sumw2'):
                    record[field] = project(record[field], groups)
                record['variations'] = {n: {d: project(a, groups) for d, a in pair.items()}
                                        for n, pair in record['variations'].items()}
    if revised:
        if exclusion_policy is not None:
            if exclusion_audit is None:
                raise ValueError('CR12 exclusion proof must be retained')
            exclusion_audit.extend(original_exclusion_audit(data, issues, exclusion_policy, systematic_combination))
        cr_channels, cr_data, cr_clipping, cr_issues = assemble(
            [cr12_payload(caches[y]['background']) for y in ('2024', '2025')], preserve_flat=True)
        require_fit_support_issues(cr_issues)
        high_cr = set(cr_channels)
        data = {k: r for k, r in data.items() if k[0] not in high_cr}
        data.update(cr_data)
        channels.update(cr_channels)
        for (channel, process), record in data.items():
            if channel.startswith('SR_highdm_'):
                record['parameter'] = cr12_parameter(record['parameter'])
        issues = [i for i in issues if i['channel'] not in high_cr] + cr_issues
        clipping = [i for i in clipping if not any(i['channel'].startswith(r+'_highdm_')
                    for r in ('LLCR', 'QCDCR', 'GCR'))] + cr_clipping
        expected = {f'{reg}_{mode}_c1_{year}': (118 if mode == 'highdm' else 30)
            if reg == 'SR' else (12 if mode == 'highdm' else 10)
            for reg in ('SR', 'LLCR', 'QCDCR', 'GCR')
            for mode in ('highdm', 'lowdm') for year in ('2024', '2025')}
        if {c: len(b) for c, b in channels.items()} != expected:
            raise ValueError('CR12 revision requires exactly 428 bins/16 channels with SR118/30')
    if issues and fit_issues is None:
        raise ValueError('Shared background model contains unsupported components')
    if fit_issues is not None:
        fit_issues.extend(issues)
    return channels, data, clipping, sources


def point_signals(channels, sources, caches, model, mass, keep_empty=False):
    grids = {y: dict(signals=cache['signals'][model]) for y, cache in caches.items()}
    return signal_data(channels, sources, grids, mass, keep_empty=keep_empty)


def point_card(channels, data, model, mass, theory, card_dir, templates, statistical_revision=None):
    """Keep original card/nuisance math; bind only the named signal objects."""
    text = card_text(channels, data, statistical_revision=statistical_revision)
    object_name = 'signal_' + model + '_' + mass
    lines, extra = [], []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'shapes':
            c = fields[2]
            path = internal_path(templates / template_name(c))
            fields[3] = os.path.relpath(path, card_dir)
            if any(ch.isspace() for ch in fields[3]):
                raise ValueError('Whitespace in Combine template path')
            line = ' '.join(fields)
            if (c, 'signal') in data:
                extra.append('shapes signal {} {} $CHANNEL/{} $CHANNEL/{}_$SYSTEMATIC'.format(
                    c, fields[3], object_name, object_name))
        lines.append(line)
    position = next(i for i, line in enumerate(lines) if line.startswith('shapes ')) + len(channels)
    lines[position:position] = extra
    return add_signal_xsec_lnN('\n'.join(lines) + '\n', mass_pair(mass)[0], theory)


def put_signals(files, channels, signals, object_name):
    import ROOT as pyroot
    for (c, process), record in signals.items():
        root = files[template_name(c)]
        directory = root.GetDirectory(c)
        if not directory:
            raise ValueError('Missing shared ROOT channel directory: ' + c)
        directory.cd()
        if directory.Get(object_name):
            raise FileExistsError('Refusing to overwrite an existing signal TH1')
        bins = channels[c]
        def put(name, values):
            h = pyroot.TH1D(name, name, len(bins), 0., float(len(bins)))
            h.Sumw2()
            for i, (value, variance) in enumerate(zip(values, record['sumw2']), 1):
                h.SetBinContent(i, float(value)); h.SetBinError(i, math.sqrt(float(variance)))
                h.GetXaxis().SetBinLabel(i, bins[i - 1]['name'])
            if h.Write() <= 0:
                raise OSError('Failed to write signal TH1 ' + name)
            h.SetDirectory(0)
        put(object_name, record['nominal'])
        for name, pair in record['variations'].items():
            for direction, suffix in (('up', 'Up'), ('down', 'Down')):
                put(object_name + '_' + name + suffix, pair[direction])


def verify_signal_objects(templates, channels, points, caches, sources, preserve_unsupported=False,
                          background=None):
    """Read every signal nominal/endpoint TH1 back after closing all files."""
    import uproot
    coverage = {}
    for filename in sorted({template_name(c) for c in channels}):
        expected = set()
        if background is not None:
            expected.update(c+'/data_obs' for c in channels if template_name(c) == filename)
            for (c, process), record in background.items():
                if template_name(c) != filename:
                    continue
                expected.add(c+'/'+process)
                expected.update(c+'/'+process+'_'+n+suffix for n in record['variations'] for suffix in ('Up', 'Down'))
        background_objects = len(expected)
        with uproot.open(templates / filename, object_cache=None, array_cache=None) as root:
            # Reusing TDirectory handles avoids reparsing their large key lists
            # for every object. No numerical check or object inventory is skipped.
            directories = {c: root[c] for c in channels if template_name(c) == filename}
            def check(c, name, values, variance):
                h = directories[c][name]
                if (h.variances() is None or h.values().shape != (len(channels[c]),)
                        or not np_allclose(h.values(), values)
                        or not np_allclose(h.variances(), variance)
                        or h.axis().labels() != [b['name'] for b in channels[c]]):
                    raise ValueError('TH1 readback mismatch: '+c+'/'+name)
            if background is not None:
                for (c, process), record in background.items():
                    if c not in directories:
                        continue
                    check(c, process, record['nominal'], record['sumw2'])
                    for n, pair in record['variations'].items():
                        for d, suffix in (('up','Up'), ('down','Down')):
                            check(c, process+'_'+n+suffix, pair[d], record['sumw2'])
                for c in directories:
                    observed = [b['observation'] if b['region'] != 'SR' else sum(
                        r['nominal'][i]*r['initial'] for (cc,p), r in background.items()
                        if cc == c and r['family'] != 'signal') for i,b in enumerate(channels[c])]
                    check(c, 'data_obs', observed, observed)
            for point in points:
                signals, issues, _ = point_signals(channels, sources, caches, point['model'], point['mass'],
                                                 keep_empty=preserve_unsupported)
                require_fit_support_issues(issues)
                if issues and not preserve_unsupported:
                    raise ValueError('Previously accepted signal is now unsupported')
                for (c, process), record in signals.items():
                    if template_name(c) != filename:
                        continue
                    name = 'signal_' + point['model'] + '_' + point['mass']
                    objects = [(name, record['nominal'])] + [(name + '_' + n + suffix, pair[d])
                        for n, pair in record['variations'].items() for d, suffix in (('up', 'Up'), ('down', 'Down'))]
                    for obj, values in objects:
                        expected.add(c+'/'+obj)
                        check(c, obj, values, record['sumw2'])
            if background is not None:
                actual = {k for k, cls in root.classnames(recursive=True, cycle=False).items()
                          if not cls.startswith('TDirectory')}
                if actual != expected:
                    raise ValueError('Missing/extra ROOT objects: ' + filename)
        coverage[filename] = dict(objects=len(expected), background_and_observation_objects=background_objects,
            signal_objects=len(expected)-background_objects, all_expected_objects_read_back=True,
            extra_objects_rejected=background is not None)
        print('Verified', filename, len(expected), 'objects', flush=True)
    return coverage


def np_allclose(a, b):
    import numpy as np
    return np.allclose(a, b, rtol=1e-12, atol=1e-12)


def write_card_once(path, text, manifest):
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('Divergent datacard retained: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)
    manifest = dict(manifest, card_sha256=sha256(path))
    side = path.with_suffix('.manifest.json')
    if side.exists() and read_json(side) != manifest:
        raise FileExistsError('Divergent datacard provenance retained')
    if not side.exists():
        write_json(side, manifest)
    return sha256(path)


def build(config_path, output, validate_only=False, preserve_unsupported=False, continuation=None):
    config_path, output = internal_path(config_path), internal_path(output)
    if not validate_only and sys.platform != 'linux':
        raise RuntimeError('Native ROOT templates must stay on hep2, not the local laptop')
    config = read_json(config_path)
    revision = config.get('statistical_revision')
    revised = require_statistical_revision(revision)
    if revised and continuation:
        raise ValueError('CR12 templates require their own immutable output, not old-template continuation')
    if revised and config.get('background_estimation_uncertainties') != dict(
            id='jme_background_shapes_20260929', mode='shape', auto_mc_stats=[10, 1, 1], user_approved=True):
        raise ValueError('CR12 requires the exact adopted measured-background shape policy')
    inventory = internal_path(ROOT / config['grid'])
    grid = requested_grid(inventory)
    contract = grid_contract(config_path, config, inventory, validate_only, preserve_unsupported)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = audit_continuation(output, contract, config, continuation) if continuation else None
        contract_path = output / 'contract.json'
        if contract_path.exists() and read_json(contract_path) != contract and (
                not receipt or read_json(contract_path) != receipt['prior_contract']):
            raise ValueError('Preserve existing grid with a different source/input contract')
        write_json(contract_path, contract)
        caches = {}
        for year in ('2024', '2025'):
            cache_root = ROOT / receipt['source_output'] if receipt else output
            path = cache_root / 'inputs' / (year + '.json.gz')
            cache_receipt = path.with_suffix('.receipt.json')
            if path.exists():
                state = read_json(cache_receipt)
                accepted = [contract] + ([receipt['prior_contract']] if receipt else [])
                if state['contract'] not in accepted or state['sha256'] != sha256(path):
                    raise ValueError('Changed or unverified extracted year inputs')
                caches[year] = read_json(path)
            else:
                write_json(output / 'state.json', dict(status='extracting', year=year,
                    pid=os.getpid(), updated=time.time(), full_workflow_complete=False))
                cache = extract_year(config['years'][year], grid, allow_validation=validate_only,
                                     preserve_unsupported=preserve_unsupported)
                if cache['year'] != year:
                    raise ValueError('Year input mismatch')
                write_json(path, json_ready(cache))
                write_json(cache_receipt, dict(contract=contract, sha256=sha256(path)))
                caches[year] = cache
        background_issues, original_exclusions = [], []
        channels, background, clipping, sources = combine_cached(caches, config.get('sr_merge'),
            fit_issues=background_issues if preserve_unsupported else None,
            statistical_revision=revision, exclusion_policy=config.get('preliminary_fit_exclusions'),
            systematic_combination=config.get('systematic_combination'), exclusion_audit=original_exclusions)
        original_background_issues = list(background_issues)
        background, shape_audit = background_error_shapes(background, config.get('background_estimation_uncertainties'))
        if revised:
            fit_data, background_issues, exclusions = cr12_fit_background(background, background_issues, original_exclusions)
            if len({r['parameter'] for r in fit_data.values() if r['parameter']}) != 96:
                raise ValueError('CR12 requires exactly 96 free background normalization parameters')
        else:
            fit_data, background_issues, exclusions = fit_background(
                background, background_issues, config.get('preliminary_fit_exclusions'))
        total_bins = sum(map(len, channels.values()))
        scopes = {c['scope'] for c in caches.values()}
        if len(scopes) != 1:
            raise ValueError('Mixed validation/production scopes')
        scope = next(iter(scopes))
        combined = scope == 'full_combined_systematic_production'
        if combined != bool(config.get('systematic_combination')):
            raise ValueError('Combined inputs and acknowledged preliminary policy must agree')
        if combined and (not revised or grid['expected_points_by_model'] != {'T2tt':225, 'T2bW':365, 'T2tb':365}):
            raise ValueError('Combined model requires the exact 955-point CR12 grid')
        model_metadata = {}
        if revised:
            model_metadata.update(statistical_revision=revision, rate_parameters=96,
                signal_contamination=False, auto_mc_stats=[10, 1, 1],
                background_estimation_uncertainties=config['background_estimation_uncertainties'])
        if config.get('systematic_combination'):
            model_metadata.update(systematic_combination=config['systematic_combination'],
                preliminary=True, central_conventions_consistent=False)
        points, blocked, signal_clipping = [], [], []
        for model in MODELS:
            for mass in grid['models'][model]:
                signals, issues, audit = point_signals(channels, sources, caches, model, mass,
                                                     keep_empty=preserve_unsupported)
                require_fit_support_issues(issues)
                if revised and any(not c.startswith('SR_') for c, p in signals):
                    raise ValueError('The adopted CR12 model requires SR-only signal')
                signal_clipping.extend(dict(a, model=model) for a in audit)
                record = dict(model=model, mass=mass)
                if issues:
                    blocked.append(dict(record, issues=issues))
                if not issues or preserve_unsupported:
                    points.append(record)
        fit_blocked = bool(background_issues or blocked)
        fit_compatibility = dict(status='blocked' if fit_blocked else 'ready',
            auto_mc_stats=[10, 1, 1], background_issues=background_issues,
            blocked_signal_points=blocked,
            cr_only_ready=not any(not i['channel'].startswith('SR_') for i in background_issues),
            interpolation_changed=bool(shape_audit), endpoints_modified=False)
        if shape_audit:
            fit_compatibility['background_error_shapes'] = shape_audit
        if exclusions:
            fit_compatibility.update(preliminary_fit_exclusions=exclusions,
                original_background_issues=original_background_issues,
                likelihood_scope_restricted=True, all_template_objects_preserved=True)
        write_json(output / 'fit_compatibility.json', fit_compatibility)
        write_json(output / 'negative_to_zero_audit.json', dict(
            background=clipping, signal=signal_clipping, policy='adopted_clip_negative_content_preserve_sumw2'))
        write_json(output / 'blocked_points.json', blocked)
        summary = dict(status='blocked' if fit_blocked else 'validated_inputs_not_root_templates',
            scope=scope, expected_points_by_model=grid['expected_points_by_model'],
            ready_inputs=0 if background_issues else len(points)-len(blocked) if preserve_unsupported else len(points),
            template_points=len(points), blocked=blocked, fit_compatibility=fit_compatibility,
            bins=total_bins, th1_channels=16,
            full_workflow_complete=False, sr_data_blinded=True, root_integrity_checked=False)
        summary.update(model_metadata)
        if config.get('sr_merge'):
            summary['sr_merge'] = config['sr_merge']
        if config.get('systematic_scope'):
            summary['systematic_scope'] = config['systematic_scope']
        if validate_only or (fit_blocked and not preserve_unsupported):
            write_json(output / 'state.json', summary)
            return 2 if fit_blocked else 0
        templates = (ROOT/receipt['source_output'] if receipt else output) / 'templates'
        template_manifest = output / 'templates' / 'model_manifest.json'
        if template_manifest.exists():
            manifest = read_json(template_manifest)
            expected_status = 'templates_verified_fit_blocked' if fit_blocked else 'templates_ready'
            if manifest['contract'] != contract or manifest['status'] != expected_status:
                raise ValueError('Interrupted/divergent template outputs retained; explicit recovery required')
            for name, digest in manifest['files'].items():
                if sha256(templates / name) != digest:
                    raise ValueError('Completed shared template changed')
            coverage = verify_signal_objects(templates, channels, points, caches, sources,
                                              preserve_unsupported, background)
            if manifest.get('object_coverage') != coverage:
                raise ValueError('Changed template endpoint coverage')
        elif templates.exists() and not receipt:
            raise FileExistsError('Incomplete ROOT template directory retained; explicit recovery required')
        else:
            if not receipt:
                templates.mkdir()
                write_json(output / 'state.json', dict(summary, status='writing_templates', pid=os.getpid()))
                write_templates(templates, channels, background)
                import ROOT as pyroot
                pyroot.gROOT.SetBatch(True)
                files = {}
                try:
                    for name in sorted({template_name(c) for c in channels}):
                        f = pyroot.TFile.Open(str(templates / name), 'UPDATE')
                        if not f or f.IsZombie():
                            raise OSError('Cannot open native shared template: ' + name)
                        files[name] = f
                    for point in points:
                        signals, issues, _ = point_signals(channels, sources, caches, point['model'], point['mass'],
                                                         keep_empty=preserve_unsupported)
                        require_fit_support_issues(issues)
                        if issues and not preserve_unsupported:
                            raise ValueError('Signal validation changed while writing')
                        put_signals(files, channels, signals, 'signal_' + point['model'] + '_' + point['mass'])
                finally:
                    for f in files.values():
                        f.Close()
            write_json(output/'state.json', dict(summary, status='verifying_templates', pid=os.getpid(),
                reused_written_templates=bool(receipt)))
            coverage = verify_signal_objects(templates, channels, points, caches, sources,
                                              preserve_unsupported, background)
            manifest = dict(status='templates_verified_fit_blocked' if fit_blocked else 'templates_ready',
                scope=scope, contract=contract, fit_compatibility=fit_compatibility, object_coverage=coverage,
                root_integrity_checked=True, sr_data_blinded=True, auto_mc_stats=[10, 1, 1],
                bins=total_bins, channels={c: [b['name'] for b in bins] for c, bins in channels.items()},
                points=points, files={name: sha256(templates / name) for name in {template_name(c) for c in channels}},
                full_workflow_complete=False)
            manifest.update(model_metadata)
            if receipt:
                if manifest['files'] != receipt['files']:
                    raise ValueError('Read-only continuation modified ROOT files')
                manifest['template_continuation'] = dict(path=str(internal_path(continuation).relative_to(ROOT)),
                    sha256=sha256(continuation))
                manifest['template_directory'] = str(templates.relative_to(ROOT))
            if config.get('sr_merge'):
                manifest['sr_merge'] = config['sr_merge']
            if config.get('systematic_scope'):
                manifest['systematic_scope'] = config['systematic_scope']
            write_json(template_manifest, manifest)
        theory = profiled_theory(load_theory(ROOT / config['theory']))
        write_json(output / 'signal_theory.json', theory)
        origin = [dict(path=str(template_manifest.parent.relative_to(ROOT)), sha256=sha256(template_manifest))]
        for point in points:
            model, mass = point['model'], point['mass']
            signals, issues, _ = point_signals(channels, sources, caches, model, mass,
                                             keep_empty=preserve_unsupported)
            data = dict(fit_data); data.update(signals)
            path = output / 'datacards' / ('datacard_' + model + '_' + mass + '.txt')
            text = point_card(channels, data, model, mass, theory, path.parent, templates, revision)
            stop, lsp = mass_pair(mass)
            point['fit_ready'] = not background_issues and not issues
            point['card_sha256'] = write_card_once(path, text, dict(
                status='card_written_not_fitted' if point['fit_ready'] else 'card_written_fit_blocked',
                fit_ready=point['fit_ready'], fit_issues=background_issues+issues,
                preliminary_fit_exclusions=exclusions,
                inputs=origin, years=['2024', '2025'], topology=model, mStop=stop, mLSP=lsp,
                cr_only=False, sr_data_blinded=True,
                theory_sha256=contract['theory_sha256'], full_workflow_complete=False,
                **dict(dict(auto_mc_stats=[10, 1, 1]), **model_metadata)))
            point['card'] = str(path.relative_to(ROOT))
        # The background-only diagnostic card uses the identical background
        # likelihood. Its observations are exclusively the measured CR data.
        cr_path = output / 'datacards' / 'datacard_cronly.txt'
        lines = []
        for line in card_text(channels, background, cr_only=True, statistical_revision=revision).splitlines():
            f = line.split()
            if f and f[0] == 'shapes':
                f[3] = os.path.relpath(templates / template_name(f[2]), cr_path.parent)
                line = ' '.join(f)
            lines.append(line)
        cr_hash = write_card_once(cr_path, '\n'.join(lines) + '\n', dict(
            status='card_written_not_fitted' if fit_compatibility['cr_only_ready'] else 'card_written_fit_blocked',
            fit_ready=fit_compatibility['cr_only_ready'], inputs=origin, years=['2024', '2025'], cr_only=True,
            sr_data_blinded=True, full_workflow_complete=False,
            **dict(dict(auto_mc_stats=[10, 1, 1]), **model_metadata)))
        final = dict(summary, status='templates_verified_fit_blocked' if fit_blocked else 'cards_ready',
                     points=points, root_integrity_checked=True,
                     templates_manifest=str(template_manifest.relative_to(ROOT)),
                     cr_only_card=dict(card=str(cr_path.relative_to(ROOT)), card_sha256=cr_hash),
                     finished=time.time())
        write_json(output / 'manifest.json', final)
        write_json(output / 'state.json', {k: v for k, v in final.items() if k != 'points'})
        print(final['status'], len(points), 'points; fits and AN plots are not complete')
        return 2 if fit_blocked else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--preserve-unsupported-templates', action='store_true',
        help='Preserve genuine zero-support nominal/endpoint TH1s and report fit incompatibility; never mark them fit-ready')
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--continue-template-audit', type=Path,
        help='Exact old/new contract receipt for reusing written ROOTs; full readback required')
    a = p.parse_args()
    if not a.validate_only and not a.native_runtime:
        # The histogram interpreter is Python 3.8, whereas the existing native
        # PyROOT/Combine software is CMSSW Python 3.9. Never mix extension paths.
        from TROTASR.workflows.combine_limits import runtime
        env = runtime(internal_path(a.output) / 'software_runtime')
        os.execvpe('python3', ['python3', str(Path(__file__).resolve()),
            '--config', str(internal_path(a.config)), '--output', str(internal_path(a.output)),
            '--native-runtime'] + (['--preserve-unsupported-templates'] if a.preserve_unsupported_templates else [])
            + (['--continue-template-audit', str(internal_path(a.continue_template_audit))]
               if a.continue_template_audit else []), env)
    raise SystemExit(build(a.config, a.output, a.validate_only, a.preserve_unsupported_templates, a.continue_template_audit))
