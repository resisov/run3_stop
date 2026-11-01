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
READBACK_RUNTIME = Path('/hep2-scratch/twkim/postprocessing/TROTASR/workflows/run_inference.sh')

CR12_POLICY = dict(id='cr12_shapes_20260929', user_approved=True,
    rate_parameter_range=[0.01, 5], highdm_cr_bins_per_region_year=12,
    native_recoil_groups=[[0], [1], [2], [3], [4, 5], [6, 7]],
    recoil_edges=[250, 300, 350, 400, 500, 800, 1500], last_bin_open_ended=True,
    signal_contamination=False, auto_mc_stats=[10, 1, 1],
    preserve_sr_bins=[118, 30], preserve_existing_uncertainty_correlations=True)

HYBRID_SCOPE = 'full_run3_hybrid_production'
HYBRID_POLICY = dict(id='early_nominal_late_systematics_20261022', user_approved=True,
    nominal_eras=['2022', '2022EE', '2023', '2023BPix'], systematic_eras=['2024', '2025'],
    preserve_existing_nuisance_names=True, retain_background_measurement_uncertainties=True,
    auto_mc_stats=[10, 1, 1])

RUN3_SYSTEMATICS_POLICY = dict(id='six_era_systematics_20261025', user_approved=True,
    systematic_eras=['2022', '2022EE', '2023', '2023BPix', '2024', '2025'],
    preserve_existing_nonluminosity_nuisance_names=True,
    luminosity_policy='cms_lum_run3_r39_20261025', retain_background_measurement_uncertainties=True,
    preserve_per_era_gnn=True, preserve_late_inputs=True, auto_mc_stats=[10, 1, 1])

ZJET_FIT_EXCLUSION = dict(id='bpix_gcr_zjet_fit_only_20261022', user_approved=True,
    preserve_all_root_objects=True, auto_mc_stats=[10, 1, 1],
    targets=[dict(channel='GCR_highdm_c1_2023BPix', process='ZJet')])


def approved_zjet_fit_background(background, issues, policy=None):
    """Apply the fit-only exclusion while retaining stored TH1s."""
    import numpy as np
    if policy is None:
        return background, issues, []
    if policy != ZJET_FIT_EXCLUSION:
        raise ValueError('Unapproved additional fit exclusion')
    key = ('GCR_highdm_c1_2023BPix', 'ZJet')
    row = background[key]
    variance = [0.] * 12
    variance[8] = 0.010207432109671547
    expected_issue = dict(channel=key[0], process=key[1], parameter='',
        problem='zero_integral_nonempty_component', sumw2=variance[8], variations=[])
    target_issues = [i for i in issues if (i.get('channel'), i.get('process')) == key]
    if (row['family'] != 'ZJet' or list(row['nominal']) != [0.] * 12 or list(row['sumw2']) != variance
            or row['parameter'] or row['initial'] != 1.
            or row.get('extra_lnN') or target_issues != [expected_issue]):
        raise ValueError('Approved GCR ZJet support changed')
    # preserve_flat retains provenance-only endpoints in systematic inputs.
    # They are inactive in native_card_text, but must be exactly zero here;
    # no tolerance, new shape support, or incomplete endpoint pair is allowed.
    if not isinstance(row['variations'], dict):
        raise ValueError('Approved GCR ZJet endpoint structure changed')
    omitted_variations = {}
    for name, pair in row['variations'].items():
        if (not isinstance(name, str) or not name or not isinstance(pair, dict)
                or set(pair) != {'up', 'down'}):
            raise ValueError('Approved GCR ZJet endpoint structure changed')
        endpoints = {}
        for direction in ('up', 'down'):
            values = np.asarray(pair[direction])
            if (values.shape != (12,) or values.dtype.kind not in 'fiu'
                    or not np.isfinite(values).all() or np.any(values != 0.)):
                raise ValueError('Approved GCR ZJet endpoint support changed')
            endpoints[direction] = values.tolist()
        omitted_variations[name] = endpoints
    audit = dict(channel=key[0], process=key[1], parameter='', nominal=0.,
        sumw2=variance[8], omitted_sumw2_by_bin=variance, omitted_variations=omitted_variations,
        omitted_variation_names=sorted(omitted_variations),
        omitted_variation_count=len(omitted_variations), omitted_endpoint_count=2*len(omitted_variations),
        decision='user_approved_fit_only_exclusion', preliminary=True,
        root_objects_preserved=True, policy=policy, auto_mc_stats=[10, 1, 1])
    return ({k: v for k, v in background.items() if k != key},
            [i for i in issues if (i.get('channel'), i.get('process')) != key], [audit])


def approved_zjet_config_change(prior, current):
    if (not require_hybrid_policy(prior) or not require_hybrid_policy(current)
            or 'additional_fit_exclusions' in prior
            or current.get('additional_fit_exclusions') != ZJET_FIT_EXCLUSION
            or {k: v for k, v in current.items() if k != 'additional_fit_exclusions'} != prior):
        raise ValueError('Fit-only recovery changed unrelated configuration')
    return True


def require_hybrid_policy(config):
    """Explicit six-era opt-in; never silently mix statistical treatments."""
    policy = config.get('hybrid_systematics')
    if policy is None:
        return False
    from TROTASR.utils.paths import ERAS
    if (policy != HYBRID_POLICY or tuple(config.get('years', ())) != ERAS
            or config.get('systematic_evaluation_deferred') is not True
            or not config.get('systematic_scope')):
        raise ValueError('Exact six-era hybrid systematic policy required')
    return True


def combined_input_scope(caches, config):
    if require_run3_systematics_policy(config):
        from TROTASR.utils.paths import ERAS
        if tuple(caches) != ERAS or any(c['scope'] != 'full_combined_systematic_production'
                                       for c in caches.values()):
            raise ValueError('Six-era systematics require six complete combined input campaigns')
        return 'full_combined_systematic_production'
    if require_hybrid_policy(config):
        from TROTASR.utils.paths import ERAS, EARLY_ERAS
        if tuple(caches) != ERAS:
            raise ValueError('Missing or reordered hybrid input eras')
        for year, cache in caches.items():
            expected = 'full_nominal_production' if year in EARLY_ERAS else 'full_combined_systematic_production'
            if cache['scope'] != expected:
                raise ValueError('Wrong systematic input scope for '+year)
            if year in EARLY_ERAS:
                audit = cache['projection_audit']['conservation']
                if set(audit) != {'highdm', 'lowdm'} or any(set(v) != {'nominal'} for v in audit.values()):
                    raise ValueError('Early-era experimental endpoints are deferred')
        return HYBRID_SCOPE
    scopes = {c['scope'] for c in caches.values()}
    if len(scopes) != 1:
        raise ValueError('Mixed validation/production scopes')
    return next(iter(scopes))


def require_run3_systematics_policy(config):
    """Separate the new six-era endpoint model from the frozen hybrid result."""
    policy = config.get('run3_systematics')
    if policy is None:
        return False
    from TROTASR.utils.paths import ERAS
    if (policy != RUN3_SYSTEMATICS_POLICY or tuple(config.get('years', ())) != ERAS
            or config.get('hybrid_systematics') is not None
            or config.get('systematic_evaluation_deferred') is not False
            or not config.get('systematic_scope')):
        raise ValueError('Exact six-era systematic policy required')
    luminosity = read_json(ROOT / 'jsons/run3_luminosity_20261025_policy.json')
    if (config.get('luminosity_model') != luminosity
            or config.get('luminosity_uncertainties') != {}
            or luminosity.get('id') != RUN3_SYSTEMATICS_POLICY['luminosity_policy']
            or not luminosity.get('user_approved')
            or not luminosity.get('central_normalization_unchanged')):
        raise ValueError('New systematic model must use the exact approved luminosity policy alone')
    return True


def require_statistical_revision(policy):
    if policy is not None and policy != CR12_POLICY:
        raise ValueError('Not the exact user-approved CR12/shape statistical revision')
    return policy is not None


def cr12_parameter(name):
    match = re.fullmatch(r'(CMS_NPS26012_(?:ll|qcd)_norm_highdm_Nb(?:1|2plus)_bin)([0-7])(_(?:2022(?:EE)?|2023(?:BPix)?|202[45]))', name)
    return match[1]+str((0, 1, 2, 3, 4, 4, 5, 5)[int(match[2])])+match[3] if match else name


def cr12_payload(payload):
    """Sum signed native CR yields before clipping; leave SR yields unchanged."""
    result = dict(year=payload['year'], groups={})
    fixed = {(r['channel'], r['process']): r for r in payload.get('cr_only_unity_audit', [])}
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
                    active = {cr12_parameter(b['rate_params'][process]) for b in present
                              if process in b['rate_params']}
                    bindings = set()
                    for b in present:
                        parameter = cr12_parameter(b['rate_params'].get(process, ''))
                        initial = b['rate_initial'].get(process, 1.)
                        audit = fixed.get((b['name'], process))
                        # The native eight-bin adapter fixes CR-only parameters
                        # before this adopted six-bin pooling. A pooled cell is
                        # not CR-only when its other member has SR support.
                        # Reconnect only the audited, identical pooled binding;
                        # genuinely unsupported pooled groups remain fixed.
                        if not parameter and audit and active:
                            restored = cr12_parameter(audit['parameter'])
                            if (active != {restored} or audit.get('fixed') != 1.
                                    or audit.get('previous_initial') != initial):
                                raise ValueError('Inconsistent audited CR-only binding')
                            parameter = restored
                            row.setdefault('cr12_binding_reconciliation', []).append(dict(
                                channel=b['name'], process=process, parameter=restored,
                                reason='adopted_pooled_CR_cell_has_SR_support'))
                        bindings.add((parameter, initial,
                            tuple((e['name'], e['down'], e['up']) for e in b['extra_lnN'].get(process, []))))
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


def correlated_luminosity_rows(channels, data, cr_only, model):
    """Regroup native luminosity columns using the LUM POG correlation matrix.

    The native builder retains its exact signal/unconstrained-background mask;
    only same-name nuisance correlation and luminosity factors change.
    """
    policy=read_json(ROOT/'jsons/run3_luminosity_20261025_policy.json')
    if model != policy or not policy.get('user_approved') or not policy.get('central_normalization_unchanged'):
        raise ValueError('Exact approved luminosity model required')
    rows=[]
    for name,factors in policy['components'].items():
        native=native_card_text(channels,data,cr_only,factors)
        columns=[line.split()[2:] for line in native.splitlines()
                 if line.startswith('lumi_13p6TeV_') and line.split()[1]=='lnN']
        if len(columns)!=len(factors) or len({len(c) for c in columns})!=1:
            raise ValueError('Native luminosity layout changed')
        combined=[]
        for values in zip(*columns):
            active=[v for v in values if v!='-']
            if len(active)>1: raise ValueError('A likelihood column belongs to multiple eras')
            combined.append(active[0] if active else '-')
        if any(v!='-' for v in combined): rows.append(name+' lnN '+' '.join(combined))
    return rows


def card_text(channels, data, cr_only=False, statistical_revision=None, luminosity_uncertainties=None,
              luminosity_model=None):
    if luminosity_model is not None and luminosity_uncertainties not in (None,{}):
        raise ValueError('Do not double count independent and correlated luminosity')
    text = native_card_text(channels, data, cr_only, {} if luminosity_model is not None else luminosity_uncertainties)
    if luminosity_model is not None:
        rows=correlated_luminosity_rows(channels,data,cr_only,luminosity_model)
        text=text.replace('* autoMCStats 10 1 1\n','\n'.join(rows+['* autoMCStats 10 1 1'])+'\n')
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


def background_shape_names(years=('2024', '2025')):
    """Nuisance names for the JME background-error revision."""
    names = {'CMS_NPS26012_RZstat_'+m+'_'+b+'_'+y
        for m in ('highdm', 'lowdm') for b in ('Nb1', 'Nb2plus') for y in years}
    for mode, intervals in (
            ('highdm', ('250to300', '300to350', '350to400', '400to500',
                        '500to650', '650to800', '800to1000', '1000to1500')),
            ('lowdm', ('250to300', '300to350', '350to400', '400to500', '500toInf'))):
        names.update('CMS_NPS26012_zgammaNonclosure_'+mode+'_u'+i+'_'+y
                     for i in intervals for y in years)
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
    years = sorted({c.rsplit('_', 1)[-1] for c, p in background})
    expected, seen, audit, result = background_shape_names(years), set(), [], {}
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
                    or name in variations or not name.endswith('_'+channel.rsplit('_', 1)[-1])
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
    """Preliminary fit-scope exclusion of two SR columns; retain ROOT content.

    This is a physics-scope restriction, not a zero-integral numerical repair.
    No other process, CR attachment, rate parameter, or endpoint is waived.
    """
    if policy is None:
        return background, issues, []
    expected = {('SR_highdm_c1_'+y, 'QCD_Nb1_u6') for y in ('2024', '2025')}
    if (policy.get('id') != 'sr_qcd_nb1_u6_20260928'
            or policy.get('user_approved') is not True
            or policy.get('preserve_all_root_objects') is not True
            or policy.get('auto_mc_stats') != [10, 1, 1]
            or len(policy.get('targets', [])) != 2
            or {(r['channel'], r['process']) for r in policy['targets']} != set(expected)):
        raise ValueError('Unapproved preliminary fit exclusion policy')
    audit = []
    for channel, process in sorted(expected):
        record = background[channel, process]
        year = channel[-4:]; nuisance = 'CMS_scale_j_Total_'+year
        parameter = 'CMS_NPS26012_qcd_norm_highdm_Nb1_bin6_'+year
        cr = background['QCDCR_highdm_c1_'+year, process]
        if (record['family'] != 'QCD' or record['parameter'] != parameter
                or any(record['nominal']) or any(record['sumw2'])
                or nuisance not in record['variations']
                or any(any(a) for n,pair in record['variations'].items() if n != nuisance for a in pair.values())
                or any(record['variations'][nuisance]['down'])
                or not all(math.isfinite(v) and v >= 0 for v in record['variations'][nuisance]['up'])
                or sum(record['variations'][nuisance]['up']) <= 0
                or cr['parameter'] != parameter or sum(cr['nominal']) <= 0):
            raise ValueError('Approved zero-total SR/positive CR support changed')
        target_issues = [i for i in issues if (i.get('channel'), i.get('process')) == (channel, process)]
        if ({i['problem'] for i in target_issues} !=
                {'zero_integral_nonempty_component', 'zero_integral_variation'}
                or len(target_issues) != 2
                or any(i['problem'] == 'zero_integral_variation' and
                    (i.get('nuisance') != nuisance or i.get('direction') != 'down') for i in target_issues)):
            raise ValueError('Approved SR fit issue set changed')
        up = float(sum(record['variations'][nuisance]['up']))
        audit.append(dict(channel=channel, process=process, parameter=parameter,
            nominal=0., sumw2=0., omitted_variations={nuisance: dict(up=up, down=0.)},
            cr_nominal_preserved=float(sum(cr['nominal'])), root_objects_preserved=True,
            decision='user_approved_fit_only_exclusion', preliminary=True))
    return ({k: v for k, v in background.items() if k not in expected},
            [i for i in issues if (i.get('channel'), i.get('process')) not in expected], audit)


def original_exclusion_audit(background, issues, policy, combination=None):
    """Check the original JME exception without changing any stored endpoint.

    Only the two excluded SR columns may carry additional non-JME
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
    from TROTASR.utils.paths import EARLY_ERAS
    early = set(config['years']) == set(EARLY_ERAS)
    hybrid = require_hybrid_policy(config)
    run3 = require_run3_systematics_policy(config)
    if not early and not hybrid and not run3 and set(config['years']) != {'2024', '2025'}:
        raise ValueError('Choose the four early eras or the existing 2024/2025 campaign')
    inputs = {}
    for year in config['years']:
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
    if config.get('early_qcd_fit_exclusions') is not None:
        from TROTASR.workflows.early_qcd_fit_policy import load_early_qcd_fit_policy
        load_early_qcd_fit_policy(config['early_qcd_fit_exclusions'])
        contract['early_qcd_fit_exclusions'] = config['early_qcd_fit_exclusions']
        contract['code']['workflows/early_qcd_fit_policy.py'] = sha256(ROOT/'workflows/early_qcd_fit_policy.py')
    if early or hybrid or run3:
        from TROTASR.gnn4lowdm.configuration import configuration_path
        contract['gnn_sha256'] = sha256(configuration_path('2022'))
        contract['luminosity_uncertainties'] = config['luminosity_uncertainties']
        contract['systematic_evaluation_deferred'] = config.get('systematic_evaluation_deferred', False)
        contract['gnn_sha256_by_era'] = {y: sha256(configuration_path(y)) for y in config['years']}
    if hybrid:
        contract['hybrid_systematics'] = config['hybrid_systematics']
    if run3:
        contract['run3_systematics'] = config['run3_systematics']
        contract['luminosity_model'] = config['luminosity_model']
        contract['luminosity_policy_sha256'] = sha256(ROOT / 'jsons/run3_luminosity_20261025_policy.json')
    if revision:
        contract['statistical_revision'] = revision
    if config.get('background_estimation_uncertainties'):
        contract['background_estimation_uncertainties'] = config['background_estimation_uncertainties']
    if config.get('systematic_combination'):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        if (config['systematic_combination'] != COMBINATION_POLICY or not revision
                or config.get('fit_campaign') not in ('combined_systematics_preliminary', 'run3_hybrid_preliminary', 'run3_systematics_preliminary')
                or config.get('fit_campaign') == 'run3_hybrid_preliminary' and not hybrid
                or config.get('fit_campaign') == 'run3_systematics_preliminary' and not run3):
            raise ValueError('Missing exact acknowledged preliminary combination policy')
        contract['systematic_combination'] = config['systematic_combination']
        contract['code'].update({name: sha256(ROOT/name) for name in (
            'workflows/merge_nominal.py', 'utils/corrections.py')})
    return contract


def combine_cached(caches, sr_merge=None, fit_issues=None, statistical_revision=None,
                   exclusion_policy=None, systematic_combination=None, exclusion_audit=None):
    revised = require_statistical_revision(statistical_revision)
    channels, data, clipping, issues = assemble([cache['background'] for cache in caches.values()], preserve_flat=True)
    require_fit_support_issues(issues)
    if any(p == 'signal' for c, p in data):
        raise ValueError('Shared background model contains unsupported or signal components')
    if len(channels) != 8 * len(caches) or sum(map(len, channels.values())) != 270 * len(caches):
        raise ValueError('Keep all 270 input analysis bins in eight TH1 channels per era')
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
            [cr12_payload(cache['background']) for cache in caches.values()], preserve_flat=True)
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
            for mode in ('highdm', 'lowdm') for year in caches}
        if {c: len(b) for c, b in channels.items()} != expected:
            raise ValueError('CR12 revision requires 214 bins/eight channels per era with SR118/30')
    if issues and fit_issues is None:
        raise ValueError('Shared background model contains unsupported components')
    if fit_issues is not None:
        fit_issues.extend(issues)
    return channels, data, clipping, sources


def point_signals(channels, sources, caches, model, mass, keep_empty=False):
    grids = {y: dict(signals=cache['signals'][model]) for y, cache in caches.items()}
    return signal_data(channels, sources, grids, mass, keep_empty=keep_empty)


def point_card(channels, data, model, mass, theory, card_dir, templates, statistical_revision=None,
               luminosity_uncertainties=None, luminosity_model=None):
    """Keep original card/nuisance math; bind only the named signal objects."""
    text = card_text(channels, data, statistical_revision=statistical_revision,
                     luminosity_uncertainties=luminosity_uncertainties, luminosity_model=luminosity_model)
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
                          background=None, filenames=None):
    """Read every signal nominal/endpoint TH1 back after closing all files."""
    import uproot
    coverage = {}
    inventory = {template_name(c) for c in channels}
    selected = inventory if filenames is None else set(filenames)
    if not selected or not selected <= inventory:
        raise ValueError('Unrecognized ROOT readback file selection')
    for filename in sorted(selected):
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


def resume_written_templates(output, contract, config, dry_run=False, check_mode='exhaustive'):
    """Pin closed, not-yet-verified ROOTs; never infer success from log lines."""
    old = read_json(output/'contract.json')
    state = read_json(output/'state.json')
    prior = old
    approval_path = output/'written_template_resume_approval.json'
    if approval_path.exists():
        approval = read_json(approval_path)
        if approval.get('schema') != 'written_template_resume_approval_v1':
            raise ValueError('Invalid written-template source approval')
        prior = read_json(internal_path(ROOT/approval['prior_contract']))
        before_config = internal_path(ROOT/approval['prior_config'])
        before_builder = internal_path(ROOT/approval['prior_builder'])
        if (sha256(before_config) != prior['config_sha256']
                or sha256(before_builder) != prior['code']['workflows/build_nominal_grid.py']
                or old not in (prior, contract)):
            raise ValueError('Preserved written-template source identity changed')
        normalized = copy.deepcopy(contract)
        if not same_extraction_contract(prior, normalized):
            from TROTASR.workflows.early_qcd_fit_policy import approved_early_qcd_config_change
            approved_early_qcd_config_change(read_json(before_config), config)
            if (normalized.pop('early_qcd_fit_exclusions', None) != config['early_qcd_fit_exclusions']
                    or 'workflows/early_qcd_fit_policy.py' in prior['code']
                    or normalized['code'].pop('workflows/early_qcd_fit_policy.py', None)
                        != sha256(ROOT/'workflows/early_qcd_fit_policy.py')):
                raise ValueError('Unapproved written-template fit-only source change')
            normalized['config_sha256'] = prior['config_sha256']
        if not same_extraction_contract(prior, normalized):
            raise ValueError('Written-template approval changed extraction/physics inputs')
    elif not same_extraction_contract(old, contract):
        raise ValueError('Written-template contract change requires preserved source approval')
    if (state.get('status') != 'verifying_templates'
            or any((output/p).exists() for p in
                   ('manifest.json', 'templates/model_manifest.json', 'cards', 'datacards'))):
        raise ValueError('Written-template resume requires the unchanged pre-card readback stage')
    expected = {'template_'+mode+'_'+year+'.root'
                for mode in ('highdm', 'lowdm') for year in config['years']}
    templates = output/'templates'
    if (not templates.is_dir() or {p.name for p in templates.iterdir()} != expected
            or any(not (templates/name).is_file() for name in expected)):
        raise ValueError('Incomplete or extra written ROOT inventory')
    cache_hashes = {}
    for year in config['years']:
        path = output/'inputs'/(year+'.json.gz')
        proof = read_json(path.with_suffix('.receipt.json'))
        digest = sha256(path)
        if not same_extraction_contract(proof['contract'], prior) or digest != proof['sha256']:
            raise ValueError('Written-template cache contract/hash changed')
        cache_hashes[year] = digest
    files = {name: sha256(templates/name) for name in sorted(expected)}
    history = output/'history'/('written_templates_'+prior['code']['workflows/build_nominal_grid.py'])
    proof = dict(schema='written_template_readback_resume_v1', prior_contract=prior,
        new_contract=contract, cache_sha256=cache_hashes, files=files,
        root_objects_rewritten=False, prior_log_lines_are_not_success_receipts=True,
        mandatory_full_readback=check_mode == 'exhaustive', mandatory_structural_readback=True,
        template_check_mode=check_mode, explicit_user_numeric_waiver=check_mode == 'minimal')
    path = history/'resume_receipt.json'
    if path.exists() and read_json(path) != proof:
        raise ValueError('Written-template snapshot changed')
    if not path.exists() and not dry_run:
        write_json(history/'prior_state.json', state)
        write_json(path, proof)
    return proof


def encode_readback_data(value):
    """Lossless JSON transport for native tuple-key records and NumPy arrays."""
    import numpy as np
    if isinstance(value, dict):
        return dict(type='dict', items=[[encode_readback_data(k), encode_readback_data(v)]
                                       for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(type='tuple', items=[encode_readback_data(v) for v in value])
    if isinstance(value, list):
        return [encode_readback_data(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Unsupported ROOT readback numeric dtype')
        return dict(type='ndarray' if isinstance(value, np.ndarray) else 'numpy_scalar',
                    dtype=value.dtype.str, shape=list(value.shape), values=value.tolist())
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite ROOT readback transport scalar')
        return value
    raise TypeError('Unsupported ROOT readback transport type: '+type(value).__name__)


def decode_readback_data(value):
    """Decode only explicit typed records; never stringify or drop mapping keys."""
    import numpy as np
    if isinstance(value, list):
        return [decode_readback_data(v) for v in value]
    if isinstance(value, dict):
        kind = value.get('type')
        if kind in ('dict', 'tuple') and set(value) == {'type', 'items'} and isinstance(value['items'], list):
            if kind == 'tuple':
                return tuple(decode_readback_data(v) for v in value['items'])
            result = {}
            for pair in value['items']:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Malformed ROOT readback mapping pair')
                key, item = map(decode_readback_data, pair)
                try:
                    if key in result:
                        raise ValueError('Duplicate ROOT readback mapping key')
                    result[key] = item
                except TypeError as error:
                    raise ValueError('Unhashable ROOT readback mapping key') from error
            return result
        if kind in ('ndarray', 'numpy_scalar') and set(value) == {'type', 'dtype', 'shape', 'values'}:
            dtype = np.dtype(value['dtype'])
            if dtype.kind not in 'biuf':
                raise ValueError('Unsupported ROOT readback numeric dtype')
            shape = value['shape']
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('Invalid ROOT readback numeric shape')
            array = np.asarray(value['values'], dtype=dtype)
            if array.size != math.prod(shape) or not np.isfinite(array).all() or (kind == 'numpy_scalar' and shape):
                raise ValueError('Invalid ROOT readback numeric value')
            array = array.reshape(shape)
            return array if kind == 'ndarray' else array[()]
        raise ValueError('Malformed ROOT readback typed transport record')
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite ROOT readback transport scalar')
        return value
    raise ValueError('Malformed ROOT readback transport value')


def minimal_template_inventory(channels, points, caches, sources, background):
    """Expected names only; never inspect bin contents, errors, or labels."""
    inventory = {}
    for filename in sorted({template_name(c) for c in channels}):
        selected = {c: bins for c, bins in channels.items() if template_name(c) == filename}
        names = {c+'/data_obs' for c in selected}
        for (c, process), row in background.items():
            if c not in selected:
                continue
            names.add(c+'/'+process)
            names.update(c+'/'+process+'_'+n+suffix for n in row['variations'] for suffix in ('Up','Down'))
        background_count = len(names)
        for c, bins in selected.items():
            if not c.startswith('SR_'):
                continue
            year = bins[0]['year']
            regimes = {regime for row in bins for regime, index in sources[year, row['name']]}
            if len(regimes) != 1:
                raise ValueError('Minimal ROOT inventory requires an unmixed signal axis')
            regime = next(iter(regimes))
            for point in points:
                row = caches[year]['signals'][point['model']][point['mass']][regime]
                name = c+'/signal_'+point['model']+'_'+point['mass']
                names.add(name)
                names.update(name+'_'+n+suffix for n in row['variations'] for suffix in ('Up','Down'))
        inventory[filename] = dict(names=sorted(names), channels=sorted(selected),
            background_and_observation_objects=background_count, signal_objects=len(names)-background_count)
    return inventory


def verify_minimal_template(path, inventory):
    """Check template structure without reading histogram arrays or labels."""
    import uproot
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        classes = root.classnames(recursive=True, cycle=False)
        actual = {name: cls for name, cls in classes.items() if not cls.startswith('TDirectory')}
        if set(actual) != set(inventory['names']):
            raise ValueError('Missing/extra ROOT objects: '+path.name)
        if any(cls != 'TH1D' for cls in actual.values()):
            raise ValueError('Unexpected ROOT histogram class: '+path.name)
        if any(not classes.get(c, '').startswith('TDirectory') for c in inventory['channels']):
            raise ValueError('Missing ROOT channel directory: '+path.name)
    return dict(objects=len(actual),
        background_and_observation_objects=inventory['background_and_observation_objects'],
        signal_objects=inventory['signal_objects'], check_mode='minimal',
        numeric_readback_performed=False, all_expected_objects_read_back=False,
        all_expected_objects_present=True, extra_objects_rejected=True, required_types_checked=True,
        file_open_header_checked=True)


def readback_worker(request_path, filename, result_path):
    """One read-only ROOT verification, launched through run_inference.sh."""
    from TROTASR.workflows.normalization_recovery import process_identity
    request_path, result_path = map(internal_path, (request_path, result_path))
    request = read_json(request_path)
    mode = request.get('check_mode', 'exhaustive')
    if (request.get('schema') != 'root_file_readback_request_v2'
            or mode not in ('minimal','exhaustive')
            or request.get('data_codec') != ('root_inventory_v1' if mode == 'minimal' else 'typed_numpy_tuple_v1')
            or filename not in request['files'] or Path(filename).name != filename
            or request['contract']['code']['workflows/build_nominal_grid.py'] != sha256(Path(__file__))
            or request.get('runtime') != readback_runtime_identity()):
        raise ValueError('Invalid ROOT readback worker contract')
    for name, digest in request['contract']['code'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('ROOT readback implementation changed: '+name)
    templates = internal_path(ROOT/request['templates'])
    request_hash = sha256(request_path)
    root_hash = request['files'][filename]
    with (request_path.parent/(filename+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256(templates/filename) != root_hash:
            raise ValueError('Written ROOT changed before readback')
        if mode == 'minimal':
            coverage = verify_minimal_template(templates/filename, request['inventory'][filename])
        else:
            caches = {}
            for year, item in request['caches'].items():
                path = internal_path(ROOT/item['path'])
                if sha256(path) != item['sha256']:
                    raise ValueError('Readback cache changed: '+year)
                caches[year] = read_json(path)
            data = decode_readback_data(request['data'])
            if not isinstance(data, dict) or set(data) != {'channels', 'background', 'sources'}:
                raise ValueError('Incomplete ROOT readback transported model')
            coverage = verify_signal_objects(templates, data['channels'], request['points'],
                caches, data['sources'], request['preserve_unsupported'], data['background'], filenames=[filename])[filename]
        if sha256(templates/filename) != root_hash or sha256(request_path) != request_hash:
            raise ValueError('Readback source changed during verification')
        write_json(result_path, dict(status='readback_complete_pending_parent_exit_check',
            request_sha256=request_hash, filename=filename, root_sha256=root_hash,
            coverage=coverage, identity=process_identity(os.getpid()), finished=time.time()))
    return 0


def verified_readback_receipt(path, request_hash, filename, root_hash, check_mode='exhaustive'):
    """Only a parent-observed exit-zero result can skip exhaustive readback."""
    proof = read_json(path)
    coverage = proof.get('coverage', {})
    identity = proof.get('identity') or {}
    command = proof.get('command') or []
    expected_prefix = ['bash', str(READBACK_RUNTIME), str(Path(__file__).resolve()),
        '--verify-template-request', str(path.parent/'request.json.gz'), '--verify-template-file', filename,
        '--verify-template-result']
    counts = [coverage.get(key) for key in ('objects', 'background_and_observation_objects', 'signal_objects')]
    if (proof.get('status') != 'complete' or proof.get('exit_code') != 0
            or proof.get('request_sha256') != request_hash
            or proof.get('filename') != filename or proof.get('root_sha256') != root_hash
            or coverage.get('check_mode', 'exhaustive') != check_mode
            or (check_mode == 'exhaustive' and coverage.get('all_expected_objects_read_back') is not True)
            or (check_mode == 'minimal' and any(coverage.get(k) is not v for k,v in dict(
                numeric_readback_performed=False, all_expected_objects_read_back=False,
                all_expected_objects_present=True, required_types_checked=True, file_open_header_checked=True).items()))
            or coverage.get('extra_objects_rejected') is not True
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts)
            or counts[0] != counts[1]+counts[2]
            or len(command) != 9 or command[:8] != expected_prefix
            or not isinstance(identity.get('pid'), int) or identity['pid'] < 2
            or not isinstance(identity.get('start_ticks'), int) or not identity.get('arguments')
            or not proof.get('result') or command[8] != str(internal_path(ROOT/proof['result']))
            or sha256(ROOT/proof['result']) != proof.get('result_sha256')
            or sha256(ROOT/proof['log']) != proof.get('log_sha256')):
        raise ValueError('Invalid completed ROOT readback receipt: '+filename)
    candidate = read_json(ROOT/proof['result'])
    launch = read_json(ROOT/proof['launch'])
    if (sha256(ROOT/proof['launch']) != proof.get('launch_sha256')
            or candidate.get('status') != 'readback_complete_pending_parent_exit_check'
            or any(candidate.get(k) != proof.get(k) for k in
                   ('request_sha256', 'filename', 'root_sha256', 'coverage', 'identity', 'finished'))
            or launch.get('command') != command or launch.get('request_sha256') != request_hash
            or launch.get('pid') != identity['pid']
            or (launch.get('identity') or {}).get('pid') != identity['pid']
            or (launch.get('identity') or {}).get('start_ticks') != identity['start_ticks']):
        raise ValueError('ROOT readback receipt does not match its exited worker: '+filename)
    return coverage


def readback_runtime_identity():
    # Record the ROOT readback executable separately from analysis-input paths.
    import hashlib
    return dict(path=str(READBACK_RUNTIME), sha256=hashlib.sha256(READBACK_RUNTIME.read_bytes()).hexdigest())


def parallel_verify_signal_objects(templates, channels, points, caches, sources,
        preserve_unsupported, background, output, contract, workers=1, cache_root=None, check_mode='exhaustive'):
    """Bounded file workers, exact serial checks, durable hash-bound successes."""
    import subprocess
    from TROTASR.workflows.nominal_campaign import resources
    from TROTASR.workflows.campaign_resources import memory_usage
    from TROTASR.workflows.normalization_recovery import process_identity
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 12:
        raise ValueError('ROOT readback workers must be an explicit integer from 1 to 12')
    if check_mode not in ('exhaustive','minimal') or (check_mode == 'minimal' and not preserve_unsupported):
        raise ValueError('Minimal structural check requires the preserved native template inventory')
    cache_root = output if cache_root is None else cache_root
    folder = output/('template_readback_minimal' if check_mode == 'minimal' else 'template_readback')
    filenames = sorted({template_name(c) for c in channels})
    payload = dict(schema='root_file_readback_request_v2', contract=contract, runtime=readback_runtime_identity(),
        templates=str(templates.relative_to(ROOT)), check_mode=check_mode,
        points=[dict(model=p['model'], mass=p['mass']) for p in points],
        preserve_unsupported=preserve_unsupported,
        files={name: sha256(templates/name) for name in filenames},
        caches={year: dict(path=str((cache_root/'inputs'/(year+'.json.gz')).relative_to(ROOT)),
                          sha256=sha256(cache_root/'inputs'/(year+'.json.gz'))) for year in caches})
    if check_mode == 'minimal':
        payload.update(data_codec='root_inventory_v1',
            inventory=minimal_template_inventory(channels, points, caches, sources, background))
    else:
        payload.update(data_codec='typed_numpy_tuple_v1',
            data=encode_readback_data(dict(channels=channels, background=background, sources=sources)))
    request = folder/'request.json.gz'
    if request.exists():
        if read_json(request) != payload:
            raise ValueError('Preserve divergent ROOT readback request')
    else:
        write_json(request, payload)
    request_hash = sha256(request)
    coverage, pending, active, failed = {}, [], {}, []
    reused = {}
    old_folder = output/'template_readback'
    old_request = old_folder/'request.json.gz'
    if check_mode == 'minimal' and old_request.exists():
        old = read_json(old_request)
        if (old.get('schema') != 'root_file_readback_request_v2'
                or old.get('check_mode', 'exhaustive') != 'exhaustive'
                or old.get('data_codec') != 'typed_numpy_tuple_v1'
                or not same_extraction_contract(old['contract'], contract)
                or any(old.get(k) != payload[k] for k in ('templates','files','caches','runtime','points','preserve_unsupported'))):
            raise ValueError('Prior exhaustive ROOT request changed inputs')
        old_hash = sha256(old_request)
        for name in filenames:
            proof = old_folder/(name+'.receipt.json')
            if not proof.exists():
                continue
            row = verified_readback_receipt(proof, old_hash, name, payload['files'][name])
            if row['objects'] != len(payload['inventory'][name]['names']):
                raise ValueError('Prior exhaustive ROOT object coverage changed')
            coverage[name] = dict(row, check_mode='exhaustive', numeric_readback_performed=True)
            reused[name] = dict(path=str(proof.relative_to(ROOT)), sha256=sha256(proof))
        write_json(folder/'reused_exhaustive_receipts.json', dict(files=reused,
            source_request_sha256=old_hash, no_original_receipts_modified=True))
    for name in filenames:
        if name in coverage:
            continue
        path = folder/(name+'.receipt.json')
        if path.exists():
            coverage[name] = verified_readback_receipt(path, request_hash, name, payload['files'][name], check_mode)
            print('Reused verified ROOT receipt', name, flush=True)
        else:
            pending.append(name)
    # A previous parent's exit does not imply its read-only children exited.
    # Do not start a duplicate attempt while any orphan still owns its file.
    for name in pending:
        with (folder/(name+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import atexit
    def preserve_active_at_exit():
        if active:
            write_json(folder/'interrupted_parent.json', dict(status='parent_exited_with_active_workers',
                pid=os.getpid(), request_sha256=request_hash, root_objects_rewritten=False,
                active_processes={name: dict(pid=p.pid, identity=process_identity(p.pid),
                    result=str(result.relative_to(ROOT)), command=command,
                    launch=str(result.with_suffix('.launch.json').relative_to(ROOT)))
                    for name, (p, stream, result, command) in active.items()}, updated=time.time()))
    atexit.register(preserve_active_at_exit)
    reserve, per_worker = 32*1024**3, 16*1024**3
    while pending or active:
        external, available = resources()
        growth = sum(max(0, per_worker-memory_usage(p.pid)['VmRSS']) for p, _, _, _ in active.values())
        limit = min(workers, max(0, 120-1-external),
                    len(active)+max(0, (available-reserve-growth)//per_worker))
        while pending and len(active) < limit and not failed:
            name = pending.pop(0)
            attempt = str(time.time_ns())
            result = folder/'attempts'/(name+'.'+attempt+'.result.json')
            log = result.with_suffix('.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            command = ['bash', str(READBACK_RUNTIME), str(Path(__file__).resolve()),
                '--verify-template-request', str(request), '--verify-template-file', name,
                '--verify-template-result', str(result)]
            stream = log.open('x')
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=str(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            active[name] = (process, stream, result, command)
            write_json(result.with_suffix('.launch.json'), dict(filename=name, pid=process.pid,
                identity=process_identity(process.pid), command=command, request_sha256=request_hash,
                result=str(result.relative_to(ROOT)), log=str(log.relative_to(ROOT)), started=time.time()))
        for name, (process, stream, result, command) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); del active[name]
            if code != 0 or not result.exists():
                failed.append(dict(filename=name, exit_code=code, result=str(result.relative_to(ROOT))))
                continue
            candidate = read_json(result)
            log = result.with_suffix('.log')
            launch_path = result.with_suffix('.launch.json')
            launch = read_json(launch_path)
            identity = candidate.get('identity') or {}
            if (candidate.get('status') != 'readback_complete_pending_parent_exit_check'
                    or candidate.get('request_sha256') != request_hash
                    or candidate.get('filename') != name
                    or candidate.get('root_sha256') != payload['files'][name]
                    or identity.get('pid') != process.pid
                    or (launch.get('identity') or {}).get('pid') != process.pid
                    or identity.get('start_ticks') != (launch.get('identity') or {}).get('start_ticks')
                    or sha256(templates/name) != payload['files'][name]):
                failed.append(dict(filename=name, exit_code=code, error='readback result/hash mismatch'))
                continue
            proof = dict(candidate, status='complete', exit_code=code, command=command,
                         result=str(result.relative_to(ROOT)), result_sha256=sha256(result),
                         launch=str(launch_path.relative_to(ROOT)), launch_sha256=sha256(launch_path),
                         log=str(log.relative_to(ROOT)), log_sha256=sha256(log))
            proof_path = folder/(name+'.receipt.json')
            write_json(proof_path, proof)
            coverage[name] = verified_readback_receipt(proof_path, request_hash, name, payload['files'][name], check_mode)
            print('Verified ROOT file receipt', name, coverage[name]['objects'], 'objects', flush=True)
        write_json(folder/'state.json', dict(status='failed' if failed else 'running',
            pid=os.getpid(), request_sha256=request_hash, workers=workers, aggregate_slots=120,
            minimum_available_memory_bytes=reserve, admission_bytes_per_worker=per_worker,
            emergency_memory_bytes=24*1024**3, available_memory_bytes=available,
            external_slots=external, completed=len(coverage), active=len(active), pending=len(pending),
            active_processes={name: dict(pid=p.pid, identity=process_identity(p.pid),
                result=str(result.relative_to(ROOT)), command=command)
                for name, (p, stream, result, command) in active.items()},
            failed=failed, updated=time.time(), root_objects_rewritten=False))
        if failed and not active:
            atexit.unregister(preserve_active_at_exit)
            raise RuntimeError('ROOT readback failed; preserved successes: '+str(failed))
        if pending or active:
            time.sleep(2)
    if set(coverage) != set(filenames):
        raise ValueError('Incomplete ROOT readback receipt coverage')
    write_json(folder/'state.json', dict(status='complete', request_sha256=request_hash,
        completed=len(coverage), active=0, pending=0, failed=[], root_objects_rewritten=False))
    atexit.unregister(preserve_active_at_exit)
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


def resume_card_inputs(output, current, check_mode):
    """Keep immutable template/card provenance; only execution code may change."""
    prior = read_json(output/'contract.json')
    manifest_path = output/'templates/model_manifest.json'
    manifest = read_json(manifest_path)
    if (not current.get('run3_systematics') or not same_extraction_contract(prior, current)
            or manifest.get('contract') != prior or manifest.get('status') != 'templates_ready'
            or manifest.get('scope') != 'full_combined_systematic_production'
            or manifest.get('template_check_mode') != check_mode
            or manifest.get('root_integrity_checked') is not True
            or manifest.get('sr_data_blinded') is not True
            or manifest.get('fit_compatibility', {}).get('status') != 'ready'
            or manifest.get('template_directory')
            or len(manifest.get('points', [])) != 955):
        raise ValueError('Card resume requires unchanged complete six-era templates and inputs')
    # The canonical template_name spelling, not a directory glob, defines scope.
    expected = {template_name('SR_'+mode+'_c1_'+year) for year in prior['inputs']
                for mode in ('highdm','lowdm')}
    if len(prior['inputs']) != 6 or set(manifest.get('files', {})) != expected:
        raise ValueError('Card resume requires the exact twelve saved ROOT files')
    for name, digest in manifest['files'].items():
        if sha256(output/'templates'/name) != digest:
            raise ValueError('Saved card template changed: '+name)
    cache_contract = prior
    written = manifest.get('written_template_resume')
    if written is not None:
        if (written.get('schema') != 'written_template_readback_resume_v1'
                or written.get('new_contract') != prior or written.get('files') != manifest['files']
                or written.get('root_objects_rewritten') is not False
                or written.get('template_check_mode') != check_mode
                or set(written.get('cache_sha256', {})) != set(prior['inputs'])):
            raise ValueError('Saved written-template/cache continuation proof changed')
        history = output/'history'/('written_templates_'+written['prior_contract']['code']['workflows/build_nominal_grid.py'])
        if read_json(history/'resume_receipt.json') != written:
            raise ValueError('Saved template history and embedded proof differ')
        cache_contract = written['prior_contract']
    caches = {}
    for year in prior['inputs']:
        path = output/'inputs'/(year+'.json.gz')
        saved = read_json(path.with_suffix('.receipt.json'))
        if (not same_extraction_contract(saved['contract'], cache_contract)
                or sha256(path) != saved['sha256']
                or written is not None and saved['sha256'] != written['cache_sha256'][year]):
            raise ValueError('Card resume extraction changed: '+year)
        caches[year] = saved['sha256']
    return dict(prior_contract=prior, manifest=manifest,
        manifest_sha256=sha256(manifest_path), cache_sha256=caches,
        cache_contract=cache_contract, execution_contract=current, explicit_user_approved=True)


def point_card_metadata(context, point, issues):
    stop, lsp = mass_pair(point['mass'])
    ready = not context['background_issues'] and not issues
    return dict(status='card_written_not_fitted' if ready else 'card_written_fit_blocked',
        fit_ready=ready, fit_issues=context['background_issues']+issues,
        preliminary_fit_exclusions=context['exclusions'], inputs=context['origin'],
        topology=point['model'], mStop=stop, mLSP=lsp, cr_only=False, sr_data_blinded=True,
        theory_sha256=context['contract']['theory_sha256'], full_workflow_complete=False,
        **dict(dict(auto_mc_stats=[10, 1, 1]), **context['model_metadata']))


def render_point_card(context, point):
    """The original serial card calculation, shared unchanged by fork workers."""
    model, mass = point['model'], point['mass']
    signals, issues, _ = point_signals(context['channels'], context['sources'], context['caches'],
        model, mass, keep_empty=context['preserve_unsupported'])
    require_fit_support_issues(issues)
    if issues != context['point_issues'].get(model+'/'+mass, []):
        raise ValueError('Signal support changed while writing card')
    data = dict(context['fit_data']); data.update(signals)
    path = context['output']/'datacards'/('datacard_'+model+'_'+mass+'.txt')
    text = point_card(context['channels'], data, model, mass, context['theory'], path.parent,
        context['templates'], context['revision'], context['config'].get('luminosity_uncertainties'),
        context['config'].get('luminosity_model'))
    return path, text, point_card_metadata(context, point, issues)


_CARD_CONTEXT = None


def card_worker(point):
    path, text, metadata = render_point_card(_CARD_CONTEXT, point)
    digest = write_card_once(path, text, metadata)
    return dict(point, fit_ready=metadata['fit_ready'], card_sha256=digest,
                card=str(path.relative_to(ROOT)))


def existing_card(context, point):
    """Reuse only a complete hash-bound pair, without regenerating its text."""
    path = context['output']/'datacards'/('datacard_'+point['model']+'_'+point['mass']+'.txt')
    side = path.with_suffix('.manifest.json')
    if not side.exists():
        return None  # An interrupted TXT alone is compared exactly by write_card_once.
    if not path.is_file():
        raise ValueError('Card provenance without its TXT is retained: '+str(path))
    issues = context['point_issues'].get(point['model']+'/'+point['mass'], [])
    expected = point_card_metadata(context, point, issues)
    digest = sha256(path)
    if read_json(side) != dict(expected, card_sha256=digest):
        raise ValueError('Existing card provenance/hash differs: '+str(path))
    return dict(point, fit_ready=expected['fit_ready'], card_sha256=digest,
                card=str(path.relative_to(ROOT)))


def verify_representative_card(context, points):
    """One saved real card only: compare the unchanged calculation byte-for-byte."""
    for point in points:
        if existing_card(context, point) is None:
            continue
        path, text, metadata = render_point_card(context, point)
        digest = sha256(path)
        if path.read_bytes() != text.encode('utf-8') or read_json(path.with_suffix('.manifest.json')) != dict(
                metadata, card_sha256=digest):
            raise ValueError('Representative serial card differs; existing files retained')
        return dict(model=point['model'], mass=point['mass'], card_sha256=digest,
            sidecar_sha256=sha256(path.with_suffix('.manifest.json')),
            text_exact=True, metadata_exact=True, existing_bytes_modified=False)
    raise ValueError('Card resume requires at least one existing pair for representative validation')


def card_capacity(workers, external, memory, active, free_disk):
    if free_disk < 80*1024**3 or memory < 24*1024**3:
        return 0
    return max(0, min(workers, 120-1-external,
        active+max(0, (memory-32*1024**3)//(8*1024**3))))


def write_point_cards(context, points, workers=1, execution=None):
    """Fork the initialized canonical runtime; share large read-only arrays by COW."""
    global _CARD_CONTEXT
    from TROTASR.workflows.normalization_recovery import process_identity
    if not 1 <= workers <= 32:
        raise ValueError('Card workers must be between one and 32')
    results, pending, reused_pairs = {}, [], {}
    for point in points:
        key = point['model']+'/'+point['mass']
        if key in results or any(p['model']+'/'+p['mass'] == key for p in pending):
            raise ValueError('Duplicate card point')
        saved = existing_card(context, point)
        if saved is None: pending.append(point)
        else:
            results[key] = saved
            reused_pairs[key] = dict(card=saved['card'], card_sha256=saved['card_sha256'],
                sidecar_sha256=sha256((ROOT/saved['card']).with_suffix('.manifest.json')))
    path = context['output']/'card_execution.json'
    record = dict(status='writing_cards', pid=os.getpid(), started=time.time(), workers=workers,
        reused=len(results), completed=len(results), total=len(points),
        process_model='fork_inherited_canonical_runtime' if workers > 1 else 'serial_canonical_runtime',
        executable=sys.executable, argv=sys.argv, aggregate_slots=120, reserve_GiB=32,
        emergency_GiB=24, disk_floor_GiB=80, root_and_cache_rewrite=False,
        template_manifest_sha256=sha256(context['output']/'templates/model_manifest.json'),
        execution=execution, worker_pids=[], worker_identities={}, reused_pairs=reused_pairs,
        controller_identity=process_identity(os.getpid()),
        inherited_runtime=readback_runtime_identity() if workers > 1 else None,
        thread_environment={name:os.environ.get(name) for name in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    if path.exists():
        write_json(context['output']/'card_execution_history'/(str(time.time_ns())+'.json'), read_json(path))
    write_json(path, record)
    _CARD_CONTEXT = context
    pool_children = {}
    try:
        if workers == 1:
            for point in pending:
                result = card_worker(point); results[result['model']+'/'+result['mass']] = result
                record.update(completed=len(results)); write_json(path, record)
        elif pending:
            if sys.platform != 'linux':
                raise RuntimeError('Parallel cards require the inherited Linux inference runtime')
            import multiprocessing
            import shutil
            from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
            from TROTASR.workflows.nominal_campaign import resources
            external, memory = resources()
            admitted = card_capacity(workers, external, memory, 0, shutil.disk_usage(context['output']).free)
            if not admitted:
                raise RuntimeError('Insufficient shared slots/memory/disk for card admission')
            record['admitted_workers'] = admitted
            with ProcessPoolExecutor(max_workers=admitted, mp_context=multiprocessing.get_context('fork')) as pool:
                active = {}; cursor = 0
                while active or cursor < len(pending):
                    external, memory = resources()
                    limit = card_capacity(admitted, external, memory, len(active),
                                          shutil.disk_usage(context['output']).free)
                    while len(active) < limit and cursor < len(pending):
                        point = pending[cursor]; cursor += 1
                        active[pool.submit(card_worker, point)] = point
                    pool_children = dict(pool._processes)
                    record.update(worker_pids=sorted(pool._processes),
                        worker_identities={str(pid):process_identity(pid) for pid in pool._processes}, completed=len(results),
                        active=len(active), pending=len(pending)-cursor, external_slots=external,
                        available_memory_bytes=memory, updated=time.time())
                    write_json(path, record)
                    if not active:
                        raise RuntimeError('Shared card capacity exhausted; completed pairs are reusable')
                    done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        active.pop(future); result = future.result()
                        results[result['model']+'/'+result['mass']] = result
            if any(process.exitcode != 0 for process in pool_children.values()):
                raise RuntimeError('Card fork worker did not exit successfully')
        for key, saved in reused_pairs.items():
            card = ROOT/saved['card']
            if sha256(card) != saved['card_sha256'] or sha256(card.with_suffix('.manifest.json')) != saved['sidecar_sha256']:
                raise ValueError('Previously complete card pair changed: '+key)
        record.update(status='cards_complete', completed=len(results), reused_pairs_preserved=True,
                      active=0, pending=0, worker_pids=[])
    except BaseException as error:
        record.update(status='needs_attention', error=repr(error)); raise
    finally:
        _CARD_CONTEXT = None
        record.update(pid=None, finished=time.time(),
            worker_exit_codes={str(pid):process.exitcode for pid,process in pool_children.items()})
        write_json(path, record)
    return [results[p['model']+'/'+p['mass']] for p in points]


def same_extraction_contract(old, current):
    """A downstream builder repair does not invalidate unchanged extracted inputs."""
    builder = 'workflows/build_nominal_grid.py'
    def inputs_only(value):
        return dict(value, code={k: v for k, v in value['code'].items() if k != builder})
    return inputs_only(old) == inputs_only(current)


def recover_verified_fit_scope(output, contract, config, dry_run=False):
    """Reuse completed ROOT templates after the configured card-scope change.

    Original metadata/cards are archived, not deleted. The completed exhaustive
    readback is reused only with identical ROOT/cache hashes and input contracts.
    This runs under the canonical builder lock, after the old controller exits.
    """
    audit = output.parent/'recovery_20261022/approved_zjet_fit_exclusion'
    prior_config = audit/'prior_config.json'
    prior_builder = audit/'prior_build_nominal_grid.py'
    approved_zjet_config_change(read_json(prior_config), config)
    history = output/'history/approved_zjet_fit_only_20261022'
    proof_path = history/'reuse_receipt.json'
    if proof_path.exists():
        proof = read_json(proof_path)
        if proof['new_contract'] != contract:
            raise ValueError('Changed fit-only recovery contract')
        old = proof['prior_contract']
        old_manifest = read_json(history/'model_manifest.json')
        if sha256(history/'model_manifest.json') != proof['verified_manifest_sha256']:
            raise ValueError('Prior exhaustive readback receipt changed')
    else:
        old = read_json(output/'contract.json')
        old_manifest = read_json(output/'templates/model_manifest.json')
        final = read_json(output/'manifest.json')
        support = final['fit_compatibility']
        target = dict(channel='GCR_highdm_c1_2023BPix', process='ZJet', parameter='',
            problem='zero_integral_nonempty_component', sumw2=0.010207432109671547, variations=[])
        if (final['status'] != 'templates_verified_fit_blocked'
                or final.get('root_integrity_checked') is not True
                or final.get('template_points') != 955 or len(final.get('points', [])) != 955
                or final.get('blocked') != [] or support.get('background_issues') != [target]
                or support.get('blocked_signal_points') != []
                or old_manifest.get('root_integrity_checked') is not True
                or old_manifest.get('contract') != old
                or old_manifest.get('fit_compatibility') != support
                or old_manifest.get('status') != 'templates_verified_fit_blocked'):
            raise ValueError('Only the completed exact GCR-blocked grid may be recovered')
        proof = dict(source_output=str(output.relative_to(ROOT)), prior_contract=old,
            new_contract=contract, files=old_manifest['files'],
            object_coverage=old_manifest['object_coverage'],
            verified_manifest_sha256=sha256(output/'templates/model_manifest.json'),
            approved_policy=ZJET_FIT_EXCLUSION, reuse_verified_readback=True,
            full_workflow_complete=False)
    if sha256(prior_config) != old['config_sha256'] or sha256(prior_builder) != old['code']['workflows/build_nominal_grid.py']:
        raise ValueError('Prior fit-only recovery source/config identity changed')
    ignored = {'config_sha256', 'code'}
    if ({k:v for k,v in old.items() if k not in ignored} !=
            {k:v for k,v in contract.items() if k not in ignored}
            or {k:v for k,v in old['code'].items() if k != 'workflows/build_nominal_grid.py'} !=
               {k:v for k,v in contract['code'].items() if k != 'workflows/build_nominal_grid.py'}):
        raise ValueError('Fit-only recovery changed numerical template inputs')
    expected = {'template_'+m+'_'+y+'.root' for m in ('highdm','lowdm') for y in config['years']}
    if (set(proof['files']) != expected or set(proof['object_coverage']) != expected
            or {p.name for p in (output/'templates').glob('*.root')} != expected
            or any(r.get('all_expected_objects_read_back') is not True or
                   r.get('extra_objects_rejected') is not True for r in proof['object_coverage'].values())):
        raise ValueError('Incomplete prior exhaustive ROOT verification')
    for name, digest in proof['files'].items():
        if sha256(output/'templates'/name) != digest:
            raise ValueError('Verified ROOT changed: '+name)
    for year in config['years']:
        path = output/'inputs'/(year+'.json.gz')
        cache = read_json(path.with_suffix('.receipt.json'))
        if not same_extraction_contract(cache['contract'], old) or sha256(path) != cache['sha256']:
            raise ValueError('Verified extracted cache changed')
        proof.setdefault('cache_contracts', {})[year] = cache['contract']
    if dry_run:
        return proof, proof_path
    if not proof_path.exists():
        history.mkdir(parents=True, exist_ok=False)
        for name in ('contract.json','manifest.json','state.json','fit_compatibility.json',
                     'negative_to_zero_audit.json','blocked_points.json','signal_theory.json','datacards'):
            source = output/name
            if source.exists():
                source.rename(history/name)
        (output/'templates/model_manifest.json').rename(history/'model_manifest.json')
        write_json(proof_path, proof)
    return proof, proof_path


def build(config_path, output, validate_only=False, preserve_unsupported=False, continuation=None,
          resume_extracted=False, resume_written=False, template_verify_workers=1, template_check_mode='exhaustive',
          resume_cards=False, card_workers=1):
    output = internal_path(output)
    # A delivered card stage is the downstream input. Do not reconstruct or
    # revalidate its upstream production just to adopt the canonical likelihood.
    accepted = output/'manifest.json'
    if accepted.is_file() and not validate_only:
        delivered = read_json(accepted)
        if delivered.get('status') == 'cards_ready' and delivered.get('run3_systematics'):
            from TROTASR.utils.era_merge import merge_accepted_grid
            merge_accepted_grid(accepted)
            return 0
    if config_path is None:
        raise ValueError('A config is required only before the accepted card stage exists')
    config_path = internal_path(config_path)
    if resume_cards and (resume_extracted or resume_written or continuation or validate_only):
        raise ValueError('Card resume is mutually exclusive with extraction/template recovery')
    if not 1 <= card_workers <= 32:
        raise ValueError('Card workers must be between one and 32')
    if resume_written and (resume_extracted or continuation or validate_only):
        raise ValueError('Written-template resume is a distinct production-only recovery')
    if template_check_mode not in ('exhaustive','minimal') or (
            template_check_mode == 'minimal' and (not (resume_written or resume_cards) or not preserve_unsupported)):
        raise ValueError('Minimal checks require explicit preserved written-template recovery')
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
        card_resume = resume_card_inputs(output, contract, template_check_mode) if resume_cards else None
        execution_contract = contract
        if card_resume:
            contract = card_resume['prior_contract']
        receipt = audit_continuation(output, contract, config, continuation) if continuation else None
        written_receipt = resume_written_templates(output, contract, config,
            check_mode=template_check_mode) if resume_written else None
        contract_path = output / 'contract.json'
        prior_contract = read_json(contract_path) if contract_path.exists() else None
        unwritten_same_extraction = (prior_contract is not None
            and same_extraction_contract(prior_contract, contract)
            and not any((output/name).exists() for name in ('manifest.json', 'templates', 'cards', 'datacards')))
        if (resume_extracted and config.get('additional_fit_exclusions') and not continuation
                and ((prior_contract is not None and prior_contract != contract and not unwritten_same_extraction)
                     or (output/'history/approved_zjet_fit_only_20261022/reuse_receipt.json').exists())):
            receipt, continuation = recover_verified_fit_scope(output, contract, config)
        if not written_receipt and contract_path.exists() and read_json(contract_path) != contract and (
                not receipt or read_json(contract_path) != receipt['prior_contract']):
            old = read_json(contract_path)
            if (not resume_extracted or not same_extraction_contract(old, contract)
                    or (output/'manifest.json').exists() or (output/'templates').exists()
                    or (output/'cards').exists() or (output/'datacards').exists()):
                raise ValueError('Preserve existing grid with a different source/input contract')
            history = output/'history'/('extracted_'+old['code']['workflows/build_nominal_grid.py'])
            write_json(history/'contract.json', old)
            if (output/'state.json').exists():
                write_json(history/'state.json', read_json(output/'state.json'))
        if not card_resume:
            write_json(contract_path, contract)
        caches = {}
        for year in config['years']:
            cache_root = ROOT / receipt['source_output'] if receipt else output
            path = cache_root / 'inputs' / (year + '.json.gz')
            cache_receipt = path.with_suffix('.receipt.json')
            if path.exists():
                state = read_json(cache_receipt)
                accepted = [contract] + ([receipt['prior_contract']] if receipt else [])
                if receipt and year in receipt.get('cache_contracts', {}):
                    accepted.append(receipt['cache_contracts'][year])
                reusable = ((resume_extracted or resume_cards) and same_extraction_contract(state['contract'], contract))
                if card_resume:
                    reusable = (same_extraction_contract(state['contract'], card_resume['cache_contract'])
                        and state['sha256'] == card_resume['cache_sha256'][year])
                if written_receipt:
                    reusable = (same_extraction_contract(state['contract'], written_receipt['prior_contract'])
                        and state['sha256'] == written_receipt['cache_sha256'][year])
                if (state['contract'] not in accepted and not reusable) or state['sha256'] != sha256(path):
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
            if original_exclusions:
                fit_data, background_issues, exclusions = cr12_fit_background(background, background_issues, original_exclusions)
            else:
                # The historical late-era QCD exception is not an early-era
                # physics choice. Keep all supported current components.
                fit_data, exclusions = background, []
            if set(caches) == {'2024', '2025'} and len({r['parameter'] for r in fit_data.values() if r['parameter']}) != 96:
                raise ValueError('Late-era CR12 normalization inventory changed')
        else:
            fit_data, background_issues, exclusions = fit_background(
                background, background_issues, config.get('preliminary_fit_exclusions'))
        fit_data, background_issues, additional_exclusions = approved_zjet_fit_background(
            fit_data, background_issues, config.get('additional_fit_exclusions'))
        exclusions.extend(additional_exclusions)
        if config.get('early_qcd_fit_exclusions') is not None:
            from TROTASR.workflows.early_qcd_fit_policy import early_qcd_fit_background
            fit_data, background_issues, early_exclusions = early_qcd_fit_background(
                fit_data, background_issues, config['early_qcd_fit_exclusions'])
            exclusions.extend(early_exclusions)
        total_bins = sum(map(len, channels.values()))
        scope = combined_input_scope(caches, config)
        combined = scope in ('full_combined_systematic_production', HYBRID_SCOPE)
        if combined != bool(config.get('systematic_combination')):
            raise ValueError('Combined inputs and acknowledged preliminary policy must agree')
        if combined and (not revised or grid['expected_points_by_model'] != {'T2tt':225, 'T2bW':365, 'T2tb':365}):
            raise ValueError('Combined model requires the exact 955-point CR12 grid')
        model_metadata = dict(years=list(caches), requested_grid=dict(
            path=str(internal_path(inventory).relative_to(ROOT)), sha256=sha256(inventory)))
        if config.get('systematic_evaluation_deferred'):
            model_metadata.update(systematic_evaluation_deferred=True, preliminary=True,
                systematic_scope=config['systematic_scope'])
        if require_hybrid_policy(config):
            model_metadata.update(hybrid_systematics=config['hybrid_systematics'],
                input_scopes_by_era={y: c['scope'] for y, c in caches.items()})
        if require_run3_systematics_policy(config):
            model_metadata.update(run3_systematics=config['run3_systematics'],
                systematic_evaluation_deferred=False, systematic_scope=config['systematic_scope'],
                input_scopes_by_era={y: c['scope'] for y, c in caches.items()},
                luminosity_uncertainties=contract['luminosity_uncertainties'],
                luminosity_model=contract['luminosity_model'],
                luminosity_policy_sha256=contract['luminosity_policy_sha256'])
        if revised:
            model_metadata.update(statistical_revision=revision,
                rate_parameters=len({r['parameter'] for r in fit_data.values() if r['parameter']}),
                signal_contamination=False, auto_mc_stats=[10, 1, 1],
                background_estimation_uncertainties=config['background_estimation_uncertainties'])
        if config.get('systematic_combination'):
            model_metadata.update(systematic_combination=config['systematic_combination'],
                preliminary=True, central_conventions_consistent=False)
        points, blocked, signal_clipping = [], [], []
        if card_resume:
            points = copy.deepcopy(card_resume['manifest']['points'])
            expected_points = [dict(model=model, mass=mass) for model in MODELS for mass in grid['models'][model]]
            if points != expected_points:
                raise ValueError('Saved template point inventory changed')
            blocked = copy.deepcopy(card_resume['manifest']['fit_compatibility']['blocked_signal_points'])
            signal_clipping = read_json(output/'negative_to_zero_audit.json')['signal']
        else:
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
        if card_resume and fit_compatibility != card_resume['manifest']['fit_compatibility']:
            raise ValueError('Saved template/card fit compatibility changed')
        write_json(output / 'fit_compatibility.json', fit_compatibility)
        write_json(output / 'negative_to_zero_audit.json', dict(
            background=clipping, signal=signal_clipping, policy='adopted_clip_negative_content_preserve_sumw2'))
        write_json(output / 'blocked_points.json', blocked)
        summary = dict(status='blocked' if fit_blocked else 'validated_inputs_not_root_templates',
            scope=scope, expected_points_by_model=grid['expected_points_by_model'],
            ready_inputs=0 if background_issues else len(points)-len(blocked) if preserve_unsupported else len(points),
            template_points=len(points), blocked=blocked, fit_compatibility=fit_compatibility,
            bins=total_bins, th1_channels=len(channels),
            full_workflow_complete=False, sr_data_blinded=True, root_integrity_checked=False)
        summary.update(model_metadata)
        minimal = template_check_mode == 'minimal'
        validation = dict(mode=template_check_mode, numeric_readback_checked=not minimal,
            explicit_user_waiver=minimal, file_open_header=True, object_inventory=True,
            required_types=minimal, checksums=True,
            bin_values=not minimal, sumw2=not minimal, labels=not minimal)
        summary.update(template_check_mode=template_check_mode, numeric_readback_performed=not minimal,
            numeric_readback_checked=not minimal, template_check_user_approved=minimal,
            root_integrity_scope=('file_open_header_object_inventory_types_and_checksums' if minimal else
                                  'full_bin_values_sumw2_labels_object_inventory_and_checksums'),
            template_validation=validation)
        if config.get('sr_merge'):
            summary['sr_merge'] = config['sr_merge']
        if config.get('systematic_scope'):
            summary['systematic_scope'] = config['systematic_scope']
        if validate_only or (fit_blocked and not preserve_unsupported):
            write_json(output / 'state.json', summary)
            return 2 if fit_blocked else 0
        templates = (ROOT/receipt['source_output'] if receipt else output) / 'templates'
        template_manifest = output / 'templates' / 'model_manifest.json'
        def full_readback():
            return parallel_verify_signal_objects(templates, channels, points, caches, sources,
                preserve_unsupported, background, output, contract, workers=template_verify_workers,
                cache_root=ROOT/receipt['source_output'] if receipt else output, check_mode=template_check_mode)
        if template_manifest.exists():
            manifest = read_json(template_manifest)
            expected_status = 'templates_verified_fit_blocked' if fit_blocked else 'templates_ready'
            if manifest['contract'] != contract or manifest['status'] != expected_status:
                raise ValueError('Interrupted/divergent template outputs retained; explicit recovery required')
            if not card_resume:
                for name, digest in manifest['files'].items():
                    if sha256(templates / name) != digest:
                        raise ValueError('Completed shared template changed')
            coverage = (card_resume['manifest']['object_coverage'] if card_resume else
                receipt['object_coverage'] if receipt and receipt.get('reuse_verified_readback') else full_readback())
            validation['reused_exhaustive_files'] = sorted(name for name,row in coverage.items()
                if minimal and row.get('check_mode') == 'exhaustive' and row.get('numeric_readback_performed') is True)
            if manifest.get('object_coverage') != coverage:
                raise ValueError('Changed template endpoint coverage')
        elif templates.exists() and not receipt and not written_receipt:
            raise FileExistsError('Incomplete ROOT template directory retained; explicit recovery required')
        else:
            if not receipt and not written_receipt:
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
                reused_written_templates=bool(receipt or written_receipt)))
            coverage = (receipt['object_coverage'] if receipt and receipt.get('reuse_verified_readback') else
                full_readback())
            validation['reused_exhaustive_files'] = sorted(name for name,row in coverage.items()
                if minimal and row.get('check_mode') == 'exhaustive' and row.get('numeric_readback_performed') is True)
            manifest = dict(status='templates_verified_fit_blocked' if fit_blocked else 'templates_ready',
                scope=scope, contract=contract, fit_compatibility=fit_compatibility, object_coverage=coverage,
                root_integrity_checked=True, sr_data_blinded=True, auto_mc_stats=[10, 1, 1],
                bins=total_bins, channels={c: [b['name'] for b in bins] for c, bins in channels.items()},
                points=points, files={name: sha256(templates / name) for name in {template_name(c) for c in channels}},
                full_workflow_complete=False)
            manifest.update(model_metadata)
            manifest.update({k:summary[k] for k in ('template_check_mode','numeric_readback_performed',
                'numeric_readback_checked','template_check_user_approved','root_integrity_scope','template_validation')})
            if written_receipt:
                if manifest['files'] != written_receipt['files']:
                    raise ValueError('Read-only written-template resume modified ROOT files')
                manifest['written_template_resume'] = written_receipt
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
        card_context = dict(channels=channels, sources=sources, caches=caches, fit_data=fit_data,
            theory=theory, output=output, templates=templates, revision=revision, config=config,
            preserve_unsupported=preserve_unsupported, background_issues=background_issues,
            exclusions=exclusions, origin=origin, contract=contract, model_metadata=model_metadata,
            point_issues={r['model']+'/'+r['mass']:r['issues'] for r in blocked})
        execution = dict(resume_cards=resume_cards, execution_contract=execution_contract,
            preserved_template_contract=contract, template_manifest_sha256=sha256(template_manifest))
        if card_resume:
            execution['representative_check'] = verify_representative_card(card_context, points)
        points = write_point_cards(card_context, points, card_workers, execution)
        # The background-only diagnostic card uses the identical background
        # likelihood. Its observations are exclusively the measured CR data.
        cr_path = output / 'datacards' / 'datacard_cronly.txt'
        lines = []
        for line in card_text(channels, fit_data, cr_only=True, statistical_revision=revision,
                              luminosity_uncertainties=config.get('luminosity_uncertainties'),
                              luminosity_model=config.get('luminosity_model')).splitlines():
            f = line.split()
            if f and f[0] == 'shapes':
                f[3] = os.path.relpath(templates / template_name(f[2]), cr_path.parent)
                line = ' '.join(f)
            lines.append(line)
        cr_hash = write_card_once(cr_path, '\n'.join(lines) + '\n', dict(
            status='card_written_not_fitted' if fit_compatibility['cr_only_ready'] else 'card_written_fit_blocked',
            fit_ready=fit_compatibility['cr_only_ready'], inputs=origin, cr_only=True,
            preliminary_fit_exclusions=exclusions,
            sr_data_blinded=True, full_workflow_complete=False,
            **dict(dict(auto_mc_stats=[10, 1, 1]), **model_metadata)))
        final = dict(summary, status='templates_verified_fit_blocked' if fit_blocked else 'cards_ready',
                     points=points, root_integrity_checked=True,
                     templates_manifest=str(template_manifest.relative_to(ROOT)),
                     cr_only_card=dict(card=str(cr_path.relative_to(ROOT)), card_sha256=cr_hash),
                     finished=time.time())
        final['card_execution'] = dict(path=str((output/'card_execution.json').relative_to(ROOT)),
                                      sha256=sha256(output/'card_execution.json'))
        write_json(output / 'manifest.json', final)
        write_json(output / 'state.json', {k: v for k, v in final.items() if k != 'points'})
        if not fit_blocked and final.get('run3_systematics') and len(final.get('years', [])) == 6:
            from TROTASR.utils.era_merge import merge_accepted_grid
            merge_accepted_grid(output/'manifest.json', lock_held=True)
        print(final['status'], len(points), 'points; fits and AN plots are not complete')
        return 2 if fit_blocked else 0


COMPARISON_SCOPE = 'sr_only_nominal_mcstat_comparison'
COMPARISON_VARIANTS = ('baseline', 'medium', 'combined', 'low', 'high')
CANDIDATE_SCHEMA = 'medium_dm_validation_candidate_v1'
CANDIDATE_RANKING_POLICY = 'fixed_high_low_combined_expected_cls_v1'
FIXED_BASELINE_GRID = 'stats/topology_transition_nominal/baseline/grid/manifest.json'
FIXED_BASELINE_SHA256 = '8f26fcf36e5569ab0ab058e5085d712d027554e33384775bf87e3ef1e69d50fc'
CANDIDATE_PARTITION = dict(partition='validation', event_hash_modulus=10,
    event_hash_remainder=2, physical_identity_fields=['physical_dataset_id','run','luminosityBlock','event'],
    year_in_event_hash=False, test_events_used=False, train_events_used=False)
CUTBASED_SCHEMA = 'medium_dm_cutbased_candidate_v1'
MET_CATEGORY_SCHEMA = 'medium_dm_met_category_candidate_v1'
MET_CATEGORY_CAVEAT = ('Explicit S0/S1 selection comparison using the same 100% parent MC population, '
    'including historical train/validation/test partitions; not independent validation. '
    'S1 removes events and must account for both selected and rejected signed weights and variances. '
    'SR data remain blinded; propagated systematics and control-region fits are not included. '
    'This comparison does not adopt a new selection.')
EXACT_GLOBAL_CAVEAT = ('Exact mTb/HT/jet1-3 pT global-cut proposals on unchanged S2, ten physical categories '
    'and 46 MET bins, using the same 100% MC. Signed source moments are preserved; the shared main '
    'builder clips negative nominal content to zero and retains sumw2. Empty-background bins are '
    'retained, not vetoed or given invented background. No independent validation, systematics or CR fit; '
    'no automatic adoption or global-optimum claim.')
SHAP_GLOBAL_CAVEAT = ('SHAP-guided global B/T/I veto proposals on unchanged S2 and ten categories/46 MET bins. '
    'Same full MC in both years; no event NN, independent validation, systematics or CR fit. '
    'Shared main negative-to-zero content treatment preserves sumw2; empty-background bins remain. '
    'MC support is diagnostic, not an eligibility veto. No automatic adoption or global-optimum claim.')


def met_category_caveat(candidate):
    if 'shap_guided' in candidate:return SHAP_GLOBAL_CAVEAT
    return EXACT_GLOBAL_CAVEAT if 'global_cuts' in candidate.get('selection',{}) else MET_CATEGORY_CAVEAT

CUTBASED_FEATURES = frozenset(('njet','nb','nisr','met','ht','dphi','met_sqrt_ht','b1_pt',
    'b_mt_min10','b_mt_max10','has_two_b10','mbb','mct_bb','mt2_bb','has_resolved','mjj','mjjj',
    'resolved_top_pt','resolved_chi2','has_isr','isr_pt','isr_dphi','isr_balance','met_over_isr',
    'sphericity','centrality','met_over_meff','jet1_pt','jet2_pt'))
CUTBASED_POPULATION_HASHES = ('event_population_sha256','physical_weights_sha256','feature_cache_sha256')
CUTBASED_CAVEAT = ('Cut-based selections and categories are optimized using 100% of the same MC, '
    'including the historical train/validation/test partitions; this is not independent validation. '
    'Finite-MC fluctuations can affect the optimized sensitivity. SR data remain blinded; '
    'propagated systematics and control-region fits are not included.')
TEST_COMPARISON_SCHEMA = 'medium_dm_frozen_test_candidate_v1'
TEST_COMPARISON_IDENTITIES = ('event_population_sha256', 'physical_weights_sha256')
TEST_COMPARISON_PARTITION = dict(partition='test', event_hash_modulus=10,
    event_hash_remainders=list(range(3,10)),
    physical_identity_fields=['physical_dataset_id','run','luminosityBlock','event'],
    year_in_event_hash=False, test_events_used=True, train_events_used=False,
    validation_events_used=False, historical_test_already_opened=True)


def test_comparison_bin_count(record):
    """Frozen test evaluation is not a validation candidate or a ranking input."""
    import hashlib
    import json
    if (record.get('schema') != TEST_COMPARISON_SCHEMA or record.get('status') != 'complete'
            or record.get('candidate_id') not in ('before','after','selected')
            or record.get('fixed_baseline_manifest_sha256') != FIXED_BASELINE_SHA256
            or not re.fullmatch(r'[0-9a-f]{64}',record.get('model',{}).get('checkpoint_sha256',''))
            or any(record.get(k) is not v for k,v in dict(systematics=False,test_events_used=True,
                historical_test_already_opened=True,reselection_allowed=False,rebinning_allowed=False,
                full_tr_retained=True,all_955_baseline_mass_points_retained=True).items())
            or not re.fullmatch(r'[0-9a-f]{64}',record.get('comparison_freeze_sha256',''))):
        raise ValueError('Invalid frozen test comparison contract')
    definition=record.get('category_definition',{});groups=definition.get('groups',[])
    canonical=dict(groups=[dict(conditions=g.get('conditions'),score_edges=g.get('score_edges')) for g in groups],
                   nbins=definition.get('nbins'))
    digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (not groups or type(canonical['nbins']) is not int or canonical['nbins']!=5*len(groups)
            or definition!=dict(canonical,id='categories_'+digest[:12])
            or record.get('category_definition_sha256')!=digest):
        raise ValueError('Frozen test categories or exactly-five-bin identity changed')
    for group in groups:
        edges=group['score_edges']
        if (not isinstance(group['conditions'],list) or not isinstance(edges,list) or len(edges)!=6
                or edges[0]!=0 or edges[-1]!=1
                or any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in edges)
                or not all(a<b for a,b in zip(edges,edges[1:]))):
            raise ValueError('Invalid fixed test score bins')
    inputs=record.get('inputs',{});support=record.get('support',{})
    if (set(inputs)!={'2024','2025'} or set(support)!=set(inputs)
            or any(inputs[y].get('path')!='medium_'+y+'.json'
                   or not re.fullmatch(r'[0-9a-f]{64}',inputs[y].get('sha256',''))
                   or type(support[y].get('passed')) is not bool
                   or not isinstance(support[y].get('unsupported_bins'),list) for y in inputs)):
        raise ValueError('Both frozen test exports and diagnostic support records are required')
    return canonical['nbins']


def load_medium_test_comparison(path):
    """Read the two-model freeze with its original evaluation-sample labels."""
    import json
    import hashlib
    path=Path(path).resolve()
    study=ROOT.parent/'sidestudy/topology_transition';allowed=study/'training/test_comparison'
    if (allowed.resolve()!=allowed or path.name!='contract.json' or path.parent.parent!=allowed
            or path.parent.name not in ('before','after','selected')):
        raise ValueError('Not an approved frozen test comparison path')
    record=json.loads(path.read_text());test_comparison_bin_count(record)
    selected=record['candidate_id']=='selected'
    frozen=(path.parent if selected else allowed)/'freeze.json'
    if (record['candidate_id']!=path.parent.name or record.get('comparison_freeze_path')!=str(frozen)
            or frozen.resolve()!=frozen
            or hashlib.sha256(frozen.read_bytes()).hexdigest()!=record['comparison_freeze_sha256']):
        raise ValueError('Frozen test comparison path/hash changed')
    freeze=json.loads(frozen.read_text())
    expected=dict(schema='medium_dm_test_comparison_freeze_v1',status='frozen_before_test_comparison',
        mc_partitions={'2024':'test','2025':'test'},split={'train':.2,'validation':.1,'test':.7},
        physical_identity_fields=TEST_COMPARISON_PARTITION['physical_identity_fields'],
        event_hash_modulus=10,event_hash_remainders=list(range(3,10)),year_in_event_hash=False,
        weight_scale=1/.7,variance_scale=1/.49,sr_data_blinded=True,expected_only=True,
        systematics=False,control_region_fit=False,reselection_allowed=False,rebinning_allowed=False,
        historical_test_already_opened=True,full_tr_retained=True,
        fixed_baseline_manifest_sha256=FIXED_BASELINE_SHA256)
    if (any(freeze.get(k)!=v or type(freeze.get(k)) is not type(v) for k,v in expected.items())
            or not freeze.get('authorized') or set(freeze.get('definitions',{}))!=({'selected'} if selected else {'before','after'})):
        raise ValueError('Not the same fixed physical-event test comparison')
    definition=freeze['definitions'][record['candidate_id']]
    for key in ('model','category_definition','category_definition_sha256'):
        if record[key]!=definition[key]:raise ValueError('Test export differs from its frozen definition')
    for definition in freeze['definitions'].values():
        checkpoint=(study/definition['checkpoint']).resolve()
        if (study not in checkpoint.parents
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=definition['model']['checkpoint_sha256']):
            raise ValueError('Frozen test checkpoint changed')
    if not freeze.get('sources'):raise ValueError('Missing frozen comparison source identities')
    for name,digest in freeze['sources'].items():
        source=(ROOT.parent/name).resolve()
        if ROOT.parent not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen comparison source changed: '+name)
    if selected:
        name=freeze.get('selected_validation_contract');digest=freeze.get('selected_validation_contract_sha256')
        if (not isinstance(name,str) or freeze['sources'].get(name)!=digest
                or record.get('selected_validation_contract')!=name
                or record.get('selected_validation_contract_sha256')!=digest):
            raise ValueError('Final test requires the exact validation-selected source contract')
        source=json.loads((ROOT.parent/name).read_text())
        for key in ('model','category_definition','category_definition_sha256'):
            if source.get(key)!=record[key]:raise ValueError('Final test changed the validation-selected '+key)
    baseline=ROOT/FIXED_BASELINE_GRID
    if sha256(baseline)!=FIXED_BASELINE_SHA256:raise ValueError('The fixed high+low baseline manifest changed')
    source=read_json(baseline)
    if (source.get('variant')!='baseline' or source.get('scope')!=COMPARISON_SCOPE
            or sorted(p['model']+'/'+p['mass'] for p in source['points'])!=freeze.get('mass_points')):
        raise ValueError('Frozen test comparison must retain the exact fixed baseline mass grid')
    for name,digest in source['files'].items():
        if sha256(internal_path(ROOT/name))!=digest:raise ValueError('The fixed high+low template changed')
    return dict(record,contract_path=str(path),contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fixed_baseline_files=source['files'])


def read_medium_test_comparison(path, candidate):
    import json
    import hashlib
    import numpy as np
    nbins=test_comparison_bin_count(candidate);path=Path(path).resolve()
    parent=Path(candidate['contract_path']).parent;year=path.stem.rsplit('_',1)[-1]
    if (year not in ('2024','2025') or path!=parent/('medium_'+year+'.json')
            or hashlib.sha256(path.read_bytes()).hexdigest()!=candidate['inputs'][year]['sha256']):
        raise ValueError('Frozen test input path/hash mismatch')
    payload=json.loads(path.read_text())
    if (payload.get('schema')!='medium_dm_frozen_test_templates_v1' or payload.get('status')!='complete'
            or payload.get('candidate_id')!=candidate['candidate_id']
            or payload.get('model')!=candidate['model'] or payload.get('category_definition')!=candidate['category_definition']
            or payload.get('sr_data_blinded') is not True or payload.get('expected_only') is not True
            or payload.get('systematics') is not False or set(payload.get('years',{}))!={year}
            or len(payload.get('bins',[]))!=nbins):
        raise ValueError('Incomplete/different frozen test export')
    provenance=payload.get('provenance',{})
    if (any(provenance.get(k)!=v or type(provenance.get(k)) is not type(v)
            for k,v in TEST_COMPARISON_PARTITION.items())
            or provenance.get('frozen_comparison_sha256')!=candidate['comparison_freeze_sha256']
            or not re.fullmatch(r'[0-9a-f]{64}',provenance.get('test_scores_sha256',''))
            or any(not re.fullmatch(r'[0-9a-f]{64}',provenance.get(k,'')) for k in TEST_COMPARISON_IDENTITIES)
            or not isinstance(provenance.get('score_provenance'),dict) or not provenance['score_provenance']):
        raise ValueError('Same physical-event test partition evidence is required')
    data=payload['years'][year]
    if (data.get('partition')!='test' or data.get('weight_scale')!=1/.7 or data.get('variance_scale')!=1/.49
            or data.get('luminosity_fb')!={'2024':109.82,'2025':110.84}[year]):
        raise ValueError('Test luminosity/subset normalization changed')
    if (payload.get('coverage',{}).get('full_tr_retained') is not True
            or payload.get('statistical_support')!=candidate['support'][year]):
        raise ValueError('Frozen test coverage/support record changed')
    edges=[(a,b) for g in candidate['category_definition']['groups'] for a,b in zip(g['score_edges'][:-1],g['score_edges'][1:])]
    for i,(label,(low,high)) in enumerate(zip(payload['bins'],edges)):
        if (label.get('score_low')!=low or label.get('score_high')!=high
                or label.get('category')!='C'+str(i//5) or label.get('bin')!=i%5+1):
            raise ValueError('Frozen test bin labels changed')
    backgrounds=payload.get('background_processes',[]);signals=payload.get('signal_names',[]);samples=data.get('samples',{})
    if (len(set(backgrounds+signals))!=len(backgrounds)+len(signals)
            or set(samples)!=set(backgrounds+signals) or not signals):
        raise ValueError('Frozen test sample inventory is incomplete or duplicated')
    for row in samples.values():
        for field in ('sumw','sumw2','entries'):
            values=np.asarray(row.get(field,[]),float)
            if values.shape!=(nbins,) or not np.isfinite(values).all() or (field!='sumw' and (values<0).any()):
                raise ValueError('Invalid signed nominal/variance/count test array')
    return payload,dict(path=str(path),sha256=candidate['inputs'][year]['sha256'],
                        **{k:provenance[k] for k in TEST_COMPARISON_IDENTITIES})


def candidate_bin_count(candidate, *, allow_hard_invalid=False):
    """Validate the optimization contract separately from the frozen-test gate."""
    if candidate.get('schema') == MET_CATEGORY_SCHEMA:
        return met_category_candidate_bin_count(candidate,allow_hard_invalid=allow_hard_invalid)
    if allow_hard_invalid:
        raise ValueError('Diagnostic hard-invalid reading applies only to MET-category exports')
    if candidate.get('schema') == CUTBASED_SCHEMA:
        return cutbased_candidate_bin_count(candidate)
    import hashlib
    import json
    if (candidate.get('schema') != CANDIDATE_SCHEMA or candidate.get('status') != 'complete'
            or not re.fullmatch(r'[A-Za-z0-9_-]+', candidate.get('candidate_id', ''))
            or candidate.get('fixed_baseline_manifest_sha256') != FIXED_BASELINE_SHA256
            or candidate.get('ranking_policy') != CANDIDATE_RANKING_POLICY
            or not re.fullmatch(r'[0-9a-f]{64}', candidate.get('model', {}).get('checkpoint_sha256',''))
            or candidate.get('systematics') is not False or candidate.get('test_events_used') is not False
            or candidate.get('full_tr_retained') is not True
            or candidate.get('all_955_baseline_mass_points_retained') is not True):
        raise ValueError('Invalid validation candidate contract')
    definition = candidate.get('category_definition', {})
    groups = definition.get('groups', [])
    nbins = definition.get('nbins')
    if not groups or type(nbins) is not int or nbins != 5*len(groups):
        raise ValueError('Exactly five score bins per candidate category are required')
    definition_hash = hashlib.sha256(json.dumps(
        {k:v for k,v in definition.items() if k != 'id'},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (candidate.get('category_definition_sha256') != definition_hash
            or definition.get('id') != 'categories_'+definition_hash[:12]):
        raise ValueError('Candidate category definition identity changed')
    for group in groups:
        edges = group.get('score_edges', [])
        if (len(edges) != 6 or edges[0] != 0 or edges[-1] != 1
                or not all(isinstance(x, (float,int)) and math.isfinite(x) for x in edges)
                or not all(a < b for a,b in zip(edges,edges[1:]))):
            raise ValueError('Candidate score bins must cover the full score range')
    inputs = candidate.get('inputs', {})
    if set(inputs) != {'2024','2025'} or any(
            inputs[y].get('path') != 'medium_'+y+'.json'
            or not re.fullmatch(r'[0-9a-f]{64}', inputs[y].get('sha256','')) for y in inputs):
        raise ValueError('Both hash-identified validation-year exports are required')
    support = candidate.get('support', {})
    if set(support) != set(inputs) or any(
            support[y].get('passed') is not True or support[y].get('unsupported_bins') != [] for y in inputs):
        raise ValueError('Both validation-year support reports must pass')
    return nbins


def cutbased_candidate_bin_count(candidate):
    """One physical leaf per bin, with no learned-score or checkpoint contract."""
    import hashlib
    import json
    if (candidate.get('schema') != CUTBASED_SCHEMA or candidate.get('status') != 'complete'
            or candidate.get('method') != 'physical_cuts_only' or candidate.get('population') != 'all_mc'
            or candidate.get('historical_partitions_used') != ['train','validation','test']
            or any(candidate.get(k) is not v for k,v in dict(neural_network_used=False,
                independent_validation=False,systematics=False,control_region_fit=False,
                full_tr_retained=True,all_955_baseline_mass_points_retained=True).items())
            or candidate.get('fixed_baseline_manifest_sha256') != FIXED_BASELINE_SHA256
            or candidate.get('ranking_policy') != CANDIDATE_RANKING_POLICY
            or any(k in candidate for k in ('model','checkpoint','checkpoint_sha256','score_edges'))):
        raise ValueError('Invalid full-MC cut-based candidate contract')
    definition = candidate.get('category_definition', {})
    groups = definition.get('groups', []); nbins = definition.get('nbins')
    digest = hashlib.sha256(json.dumps({k:v for k,v in definition.items() if k != 'id'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (definition.get('method') != 'physical_cuts_only' or type(nbins) is not int or nbins < 1
            or nbins != len(groups) or candidate.get('category_definition_sha256') != digest
            or definition.get('id') != 'categories_'+digest[:12]
            or candidate.get('candidate_id') != 'cutbased_'+digest[:12]):
        raise ValueError('Cut-based physical leaf count or definition identity changed')
    features = candidate.get('feature_definitions', {})
    if (not isinstance(features, dict) or not set(features) <= CUTBASED_FEATURES or any(
            not isinstance(record, dict) or not isinstance(record.get('definition'), str)
            or not record['definition'].strip() or not isinstance(record.get('unit'), str)
            for record in features.values())):
        raise ValueError('Explicit deterministic physical feature definitions are required')
    seen = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {'conditions'} or not isinstance(group['conditions'], list):
            raise ValueError('Each cut-based leaf contains physical conditions only, not score bins')
        bounds = {}
        for condition in group['conditions']:
            if (not isinstance(condition,list) or len(condition) != 3 or not isinstance(condition[0],str)
                    or condition[0] not in features
                    or type(condition[1]) not in (int,float) or not math.isfinite(condition[1])
                    or type(condition[2]) is not bool):
                raise ValueError('Invalid cut-based feature/threshold/direction')
            name,threshold,is_ge = condition
            low,high = bounds.get(name,(-math.inf,math.inf))
            bounds[name] = (max(low,threshold),high) if is_ge else (low,min(high,threshold))
            if bounds[name][0] >= bounds[name][1]:raise ValueError('Contradictory cut-based leaf')
        identity = tuple(sorted((name,*pair) for name,pair in bounds.items()))
        if identity in seen:raise ValueError('Duplicate cut-based physical leaf')
        seen.add(identity)
    for section in ('inputs','support','populations'):
        if set(candidate.get(section,{})) != {'2024','2025'}:
            raise ValueError('Both full-MC years and their population identities are required')
    for year in ('2024','2025'):
        record = candidate['inputs'][year]; support = candidate['support'][year]
        if (record.get('path') != 'medium_'+year+'.json'
                or not re.fullmatch(r'[0-9a-f]{64}',record.get('sha256',''))
                or support.get('passed') is not True or support.get('unsupported_bins') != []
                or any(not re.fullmatch(r'[0-9a-f]{64}',candidate['populations'][year].get(k,''))
                       for k in CUTBASED_POPULATION_HASHES)):
            raise ValueError('Missing full-MC input hash, support or population evidence')
    return nbins


def met_category_groups():
    """Define ten physical categories; open MET endpoints use JSON null."""
    groups = []
    for number,nb in ((1,False),(2,True)):
        for suffix,nisr in (('a',False),('b',True)):
            groups.append(dict(id='M'+str(number)+suffix,
                conditions=[['njet',3.5,False],['nb',1.5,nb],['nisr',.5,nisr]],
                high_angle_pass=True,met_edges=[250,350,450,600,800,None]))
    for number,nb,jet in ((3,False,False),(4,True,False),(5,False,True),(6,True,True)):
        conditions=([['njet',6.5,True]] if jet else [['njet',3.5,True],['njet',6.5,False]])
        groups.append(dict(id='M'+str(number),conditions=conditions+[['nb',1.5,nb]],
            high_angle_pass=True,met_edges=[250,350,450,600,800,None]))
    for number,nb in ((7,False),(8,True)):
        groups.append(dict(id='M'+str(number),conditions=[['nb',1.5,nb]],
            high_angle_pass=False,met_edges=[250,400,600,None]))
    return groups


def exact_global_cuts(cuts):
    """Validate five-observable global predicates separately from SR splits."""
    if not isinstance(cuts,list) or not 1<=len(cuts)<=5:
        raise ValueError('Explicit one-to-five global cut predicates are required')
    axes={'mtb','ht','jet1_pt','jet2_pt','jet3_pt'}
    finite=lambda x:type(x) in (int,float) and math.isfinite(x)
    for cut in cuts:
        if not isinstance(cut,dict):raise ValueError('Invalid global predicate')
        kind=cut.get('kind')
        if kind is None:
            fields={'observable','operator','undefined'}
            op=cut.get('operator')
            fields|={'low','high'} if op=='window' else {'threshold'}
            numbers=[cut.get('low'),cut.get('high')] if op=='window' else [cut.get('threshold')]
            if (set(cut)!=fields or cut.get('observable') not in axes or cut.get('undefined')!='pass'
                    or op not in ('window','>=','<') or not all(finite(x) for x in numbers)
                    or (op=='window' and numbers[0]>=numbers[1])):
                raise ValueError('Invalid single-axis global cut')
        else:
            pair=cut.get('axes',[])
            if (not isinstance(pair,list) or len(pair)!=2 or len(set(pair))!=2
                    or not set(pair)<=axes or cut.get('undefined')!='pass_joint'):
                raise ValueError('Invalid global-cut axes or missing-object policy')
            if kind in ('quadrant','veto'):
                if (set(cut)!={'kind','axes','thresholds','upper','undefined'}
                        or len(cut.get('thresholds',[]))!=2 or not all(finite(x) for x in cut['thresholds'])
                        or len(cut.get('upper',[]))!=2 or any(type(x) is not bool for x in cut['upper'])):
                    raise ValueError('Invalid global quadrant or veto')
            elif kind=='linear':
                if (set(cut)!={'kind','axes','scales','slope','threshold','upper','undefined'}
                        or len(cut.get('scales',[]))!=2 or not all(finite(x) and x>0 for x in cut['scales'])
                        or not finite(cut.get('slope')) or not finite(cut.get('threshold'))
                        or type(cut.get('upper')) is not bool):
                    raise ValueError('Invalid global linear cut')
            else:raise ValueError('Unsupported global predicate')
    return cuts


def shap_global_cuts(cuts):
    """Apply three conditional vetoes; missing objects pass, with no NN input."""
    if not isinstance(cuts,list) or len(cuts)>3:raise ValueError('At most three B/T/I blocks')
    seen=set()
    for cut in cuts:
        block=cut.get('block');seen.add(block)
        axes={'B':['b2_pt','mbb'],'T':['resolved_chi2','met_sqrt_ht'],
              'I':['isr_subjet_btag','isr_pt','ht_over_isr']}.get(block)
        fields={'kind','axes','thresholds','undefined','block'}|({'upper'} if block!='I' else set())
        if (axes is None or set(cut)!=fields or cut['axes']!=axes or cut['undefined']!='pass_joint'
                or cut['kind']!=('isr_composition_veto' if block=='I' else 'veto')
                or len(cut['thresholds'])!=len(axes)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in cut['thresholds'])
                or (block!='I' and cut['upper']!=([True,False] if block=='B' else [False,False]))):
            raise ValueError('Invalid approved SHAP-guided global block')
    if len(seen)!=len(cuts):raise ValueError('Duplicate B/T/I block')
    return cuts


def met_category_selection(name, global_cuts=None):
    if global_cuts is not None:
        import hashlib
        import json
        shap=name.startswith('Y')
        cuts=shap_global_cuts(global_cuts) if shap else exact_global_cuts(global_cuts)
        digest=hashlib.sha256(json.dumps(cuts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if name!=('Y' if shap else 'X')+digest[:12]:raise ValueError('Global-cut identity differs from its actual predicates')
        return dict(id=name,conditions=[['jet2_dphi','>=',.4]],full_tr_retained=False,global_cuts=cuts)
    conditions = {'S0': [], 'S1': [['dphi','>',.15],['met_sqrt_ht','>',10.]],
                  'S2': [['jet2_dphi','>=',.4]]}
    if name not in conditions:raise ValueError('Only approved explicit global selections are supported')
    return dict(id=name,conditions=conditions[name],full_tr_retained=name=='S0')


def met_category_candidate_bin_count(candidate, *, allow_hard_invalid=False):
    """Do not relax the all-TR historical cut contract to accommodate S1."""
    import hashlib
    import json
    selection=candidate.get('selection',{})
    name=selection.get('id')
    exact='global_cuts' in selection
    shap=name.startswith('Y') if isinstance(name,str) else False
    expected_selection=met_category_selection(name,selection.get('global_cuts'))
    if (set(selection)!=set(expected_selection)|{'label'}
            or any(selection[k]!=v for k,v in expected_selection.items())
            or type(selection.get('full_tr_retained')) is not bool
            or not isinstance(selection['label'],str) or not selection['label'].strip()):
        raise ValueError('The approved global selection changed')
    if (candidate.get('schema')!=MET_CATEGORY_SCHEMA or candidate.get('status')!='complete'
            or candidate.get('method')!='physical_categories_met' or candidate.get('population')!='all_mc'
            or candidate.get('historical_partitions_used')!=['train','validation','test']
            or any(candidate.get(k) is not v for k,v in dict(neural_network_used=False,
                independent_validation=False,systematics=False,control_region_fit=False,
                full_tr_retained=name=='S0',full_tr_accounted=True,all_955_baseline_mass_points_retained=True).items())
            or candidate.get('fixed_baseline_manifest_sha256')!=FIXED_BASELINE_SHA256
            or candidate.get('ranking_policy')!=CANDIDATE_RANKING_POLICY
            or any(k in candidate for k in ('model','checkpoint','checkpoint_sha256','score_edges','populations'))):
        raise ValueError('Invalid MET-category selection comparison contract')
    definition=candidate.get('category_definition',{})
    expected=dict(method='physical_categories_met',nbins=46,groups=met_category_groups())
    groups=definition.get('groups',[])
    physical=dict(definition)
    physical.pop('id',None)
    physical['groups']=[{k:v for k,v in g.items() if k!='label'} for g in groups]
    digest=hashlib.sha256(json.dumps({k:v for k,v in definition.items() if k!='id'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (physical!=expected or any(not isinstance(g.get('label'),str) or not g['label'].strip() for g in groups)
            or definition.get('id')!='categories_'+digest[:12]
            or candidate.get('category_definition_sha256')!=digest
            or candidate.get('candidate_id')!='met_categories_'+name+'_'+digest[:12]):
        raise ValueError('The approved ten-category MET definition or identity changed')
    features=candidate.get('feature_definitions',{})
    extra_features = ({'jet2_dphi','b2_pt','ht_over_isr','isr_subjet_btag'} if shap else
                      {'jet2_dphi','mtb','jet3_pt'} if exact else {'jet2_dphi'} if name == 'S2' else set())
    if (not {'njet','nb','nisr','met','dphi','met_sqrt_ht','pass_high_angle'}<=set(features)
            or not extra_features<=set(features)
            or not set(features)<=CUTBASED_FEATURES|{'pass_high_angle'}|extra_features
            or any(not isinstance(v,dict) or not isinstance(v.get('definition'),str)
                   or not v['definition'].strip() or not isinstance(v.get('unit'),str) for v in features.values())):
        raise ValueError('Explicit MET-category physical feature definitions are required')
    if shap:
        evidence=candidate.get('shap_guided',{});policy=evidence.get('physics_policy',{})
        if (set(evidence.get('years',{}))!={'2024','2025'} or evidence.get('selection_adopted') is not False
                or not re.fullmatch(r'[0-9a-f]{64}',evidence.get('search_report_sha256',''))
                or policy.get('baseline_selection')!='S2' or policy.get('new_categories') is not False
                or policy.get('systematics') is not False or policy.get('control_region_fit') is not False):
            raise ValueError('Missing SHAP-cut source/population provenance')
    elif exact:
        evidence=candidate.get('exact_kinematics',{});policy=evidence.get('physics_policy',{})
        if (set(evidence.get('years',{}))!={'2024','2025'} or evidence.get('screening_only') is not True
                or evidence.get('selection_adopted') is not False
                or not re.fullmatch(r'[0-9a-f]{64}',evidence.get('search_report_sha256',''))
                or policy.get('baseline_selection')!='S2' or policy.get('new_categories') is not False
                or policy.get('systematics') is not False or policy.get('control_region_fit') is not False
                or set(policy.get('thresholds',{}))!={'mtb','ht','jet1_pt','jet2_pt','jet3_pt'}):
            raise ValueError('Missing exact-observable/global-selection provenance')
        for year,row in evidence['years'].items():
            if (type(row.get('events')) is not int or row['events']<0 or
                    any(not re.fullmatch(r'[0-9a-f]{64}',row.get(k,'')) for k in
                        ('event_population_sha256','physical_weights_sha256','manifest_sha256','contract_sha256'))
                    or not isinstance(row.get('definition'),str) or not row['definition']):
                raise ValueError('Missing exact S2 population/weight evidence: '+year)
    for section in ('inputs','support','parent_populations'):
        if set(candidate.get(section,{}))!={'2024','2025'}:
            raise ValueError('Both MET-category years and parent populations are required')
    for year in ('2024','2025'):
        record=candidate['inputs'][year];support=candidate['support'][year]
        invalid=support.get('hard_invalid_bins')
        if (record.get('path')!='medium_'+year+'.json' or not re.fullmatch(r'[0-9a-f]{64}',record.get('sha256',''))
                or type(support.get('passed')) is not bool or not isinstance(support.get('unsupported_bins'),list)
                or not isinstance(invalid,list) or any(type(i) is not int or not 0<=i<46 for i in invalid)
                or len(invalid)!=len(set(invalid))
                or any(not re.fullmatch(r'[0-9a-f]{64}',candidate['parent_populations'][year].get(k,''))
                       for k in CUTBASED_POPULATION_HASHES)):
            raise ValueError('Missing MET-category input hash, support or parent-population evidence')
        empty=support.get('empty_all_samples_bins',[]) if exact else []
        if (not isinstance(empty,list) or any(type(i) is not int or not 0<=i<46 for i in empty)
                or len(empty)!=len(set(empty)) or not set(empty)<=set(invalid)):
            raise ValueError('Invalid all-sample-empty bin declaration')
        # These bins are retained, not rebinned or populated with an invented
        # background. The reader must verify every sample moment is exactly 0.
        # Signed/empty finite-MC support is a diagnostic, not a CLs veto.
        # The shared assemble() applies the adopted main clipping policy.
    return 46


def load_medium_candidate(path, *, allow_hard_invalid=False):
    """One explicit external export contract; all likelihood products stay internal."""
    import json
    import hashlib
    path = Path(path).resolve()
    allowed = ROOT.parent/'sidestudy/topology_transition/training/combined_optimization/candidates'
    cutbased = ROOT.parent/'sidestudy/topology_transition/cutbased/candidates'
    met_categories = ROOT.parent/'sidestudy/topology_transition/cutbased/met_categories/candidates'
    selected = path.parent.parent
    if (selected not in (allowed,cutbased,met_categories) or selected.resolve() != selected or path.name != 'contract.json'):
        raise ValueError('Not an approved medium validation candidate path')
    record = json.loads(path.read_text())
    if {CANDIDATE_SCHEMA:allowed,CUTBASED_SCHEMA:cutbased,MET_CATEGORY_SCHEMA:met_categories}.get(record.get('schema')) != selected:
        raise ValueError('Cut-based and historical validation contract paths cannot be interchanged')
    candidate_bin_count(record,allow_hard_invalid=allow_hard_invalid)
    if record['candidate_id'] != path.parent.name:
        raise ValueError('Candidate directory and identity differ')
    baseline = ROOT/FIXED_BASELINE_GRID
    if sha256(baseline) != FIXED_BASELINE_SHA256:
        raise ValueError('The fixed high+low baseline manifest changed')
    source = read_json(baseline)
    if source.get('variant') != 'baseline' or source.get('scope') != COMPARISON_SCOPE:
        raise ValueError('Not the fixed nominal high+low baseline')
    for name,digest in source['files'].items():
        if sha256(internal_path(ROOT/name)) != digest:
            raise ValueError('The fixed high+low template changed')
    return dict(record, contract_path=str(path), contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fixed_baseline_files=source['files'])


def read_medium_validation(path, candidate):
    if candidate.get('schema') == MET_CATEGORY_SCHEMA:
        return read_medium_met_categories(path,candidate)
    if candidate.get('schema') == CUTBASED_SCHEMA:
        return read_medium_cutbased(path,candidate)
    import json
    import hashlib
    import numpy as np
    nbins = candidate_bin_count(candidate)
    path = Path(path).resolve()
    parent = Path(candidate['contract_path']).parent
    year = path.stem.rsplit('_',1)[-1]
    if (year not in ('2024','2025') or path != parent/('medium_'+year+'.json')
            or hashlib.sha256(path.read_bytes()).hexdigest() != candidate['inputs'][year]['sha256']):
        raise ValueError('Candidate input path/hash mismatch')
    payload = json.loads(path.read_text())
    if (payload.get('schema') != 'medium_dm_validation_templates_v1'
            or payload.get('status') != 'complete' or payload.get('candidate_id') != candidate['candidate_id']
            or payload.get('sr_data_blinded') is not True or payload.get('expected_only') is not True
            or payload.get('systematics') is not False or payload.get('model') != candidate['model']
            or payload.get('category_definition') != candidate['category_definition']
            or set(payload.get('years', {})) != {year} or len(payload.get('bins', [])) != nbins):
        raise ValueError('Incomplete/different medium validation export')
    provenance = payload.get('provenance', {})
    if (any(provenance.get(k) != v or type(provenance.get(k)) is not type(v)
            for k,v in CANDIDATE_PARTITION.items())
            or not re.fullmatch(r'[0-9a-f]{64}', provenance.get('validation_scores_sha256',''))):
        raise ValueError('Validation-only physical-event partition evidence is required')
    data = payload['years'][year]
    if (data.get('partition') != 'validation' or data.get('weight_scale') != 10.
            or data.get('variance_scale') != 100.
            or data.get('luminosity_fb') != {'2024':109.82,'2025':110.84}[year]):
        raise ValueError('Validation luminosity/subset normalization changed')
    if (payload.get('coverage', {}).get('full_tr_retained') is not True
            or payload.get('statistical_support') != candidate['support'][year]):
        raise ValueError('Candidate coverage/support differs from its contract')
    for index,(label,(low,high)) in enumerate(zip(payload['bins'],
            [(a,b) for group in candidate['category_definition']['groups']
             for a,b in zip(group['score_edges'][:-1],group['score_edges'][1:])])):
        if (label.get('score_low') != low or label.get('score_high') != high
                or label.get('category') != 'C'+str(index//5) or label.get('bin') != index%5+1):
            raise ValueError('Candidate bin labels differ from its score edges')
    backgrounds, signals = payload.get('background_processes', []), payload.get('signal_names', [])
    samples = data.get('samples', {})
    if (len(set(backgrounds+signals)) != len(backgrounds)+len(signals)
            or set(samples) != set(backgrounds+signals) or not signals):
        raise ValueError('Candidate sample inventory is incomplete or duplicated')
    for record in samples.values():
        for field in ('sumw','sumw2','entries'):
            values = np.asarray(record.get(field, []),float)
            if (values.shape != (nbins,) or not np.isfinite(values).all()
                    or (field != 'sumw' and (values < 0).any())):
                raise ValueError('Invalid signed nominal/variance/count candidate array')
    return payload, dict(path=str(path),sha256=candidate['inputs'][year]['sha256'])


def read_medium_cutbased(path,candidate):
    """Authenticate full-MC signed moments and complete physical-leaf assignment."""
    import hashlib
    import json
    import numpy as np
    nbins = cutbased_candidate_bin_count(candidate)
    path = Path(path).resolve(); year = path.stem.rsplit('_',1)[-1]
    if (year not in ('2024','2025') or path != Path(candidate['contract_path']).parent/('medium_'+year+'.json')
            or hashlib.sha256(path.read_bytes()).hexdigest() != candidate['inputs'][year]['sha256']):
        raise ValueError('Cut-based input path/hash mismatch')
    payload = json.loads(path.read_text())
    if (payload.get('schema') != 'medium_dm_cutbased_templates_v1' or payload.get('status') != 'complete'
            or payload.get('candidate_id') != candidate['candidate_id']
            or payload.get('category_definition') != candidate['category_definition']
            or payload.get('feature_definitions') != candidate['feature_definitions']
            or payload.get('sr_data_blinded') is not True or payload.get('expected_only') is not True
            or payload.get('systematics') is not False or set(payload.get('years',{})) != {year}
            or any(k in payload for k in ('model','checkpoint','checkpoint_sha256','score_edges'))):
        raise ValueError('Incomplete/different cut-based MC export')
    provenance = payload.get('provenance', {})
    if (provenance.get('partition') != 'all_mc' or provenance.get('neural_network_used') is not False
            or provenance.get('independent_validation') is not False
            or any(provenance.get(k) != candidate['populations'][year][k] for k in CUTBASED_POPULATION_HASHES)):
        raise ValueError('Cut-based all-MC population/weight identities changed')
    expected_bins = [dict(bin=i+1,conditions=g['conditions'])
                     for i,g in enumerate(candidate['category_definition']['groups'])]
    if payload.get('bins') != expected_bins:
        raise ValueError('Cut-based bin labels differ from physical leaves')
    data = payload['years'][year]; coverage = payload.get('coverage', {})
    if (data.get('partition') != 'all_mc' or data.get('weight_scale') != 1. or data.get('variance_scale') != 1.
            or data.get('luminosity_fb') != {'2024':109.82,'2025':110.84}[year]
            or type(data.get('events')) is not int or data['events'] < 0
            or coverage.get('full_tr_retained') is not True or coverage.get('source_events') != data['events']
            or coverage.get('assigned_events') != data['events'] or coverage.get('unassigned_events') != 0
            or coverage.get('multiply_assigned_events') != 0
            or payload.get('statistical_support') != candidate['support'][year]):
        raise ValueError('Full-MC normalization, support or complete TR assignment changed')
    backgrounds,signals = payload.get('background_processes',[]),payload.get('signal_names',[])
    samples = data.get('samples',{}); totals = coverage.get('sample_totals',{})
    if (not signals or len(set(backgrounds+signals)) != len(backgrounds)+len(signals)
            or set(samples) != set(backgrounds+signals) or set(totals) != set(samples)):
        raise ValueError('Cut-based sample inventory/totals are incomplete or duplicated')
    entries = 0.
    for name,record in samples.items():
        for field in ('sumw','sumw2','entries'):
            values = np.asarray(record.get(field,[]),float); total = totals[name].get(field)
            if (values.shape != (nbins,) or not np.isfinite(values).all()
                    or (field != 'sumw' and (values < 0).any())
                    or type(total) not in (int,float) or not math.isfinite(total)
                    or (field == 'entries' and (np.any(values != np.floor(values)) or total != math.fsum(values)))
                    or not math.isclose(math.fsum(values),total,rel_tol=1e-10,abs_tol=1e-7)):
                raise ValueError('Cut-based signed weight/variance/count conservation failed: '+name+'/'+field)
        entries += totals[name]['entries']
    if entries != data['events']:raise ValueError('Full-MC event population count differs from retained samples')
    return payload,dict(path=str(path),sha256=candidate['inputs'][year]['sha256'])


def read_medium_met_categories(path,candidate, *, allow_hard_invalid=False):
    """Read selected signed moments and account for every rejected parent event."""
    import hashlib
    import json
    import numpy as np
    nbins=met_category_candidate_bin_count(candidate,allow_hard_invalid=allow_hard_invalid)
    path=Path(path).resolve();year=path.stem.rsplit('_',1)[-1]
    if (year not in ('2024','2025') or path!=Path(candidate['contract_path']).parent/('medium_'+year+'.json')
            or hashlib.sha256(path.read_bytes()).hexdigest()!=candidate['inputs'][year]['sha256']):
        raise ValueError('MET-category input path/hash mismatch')
    payload=json.loads(path.read_text())
    if (payload.get('schema')!='medium_dm_met_category_templates_v1' or payload.get('status')!='complete'
            or any(payload.get(k)!=candidate[k] for k in
                   ('candidate_id','category_definition','feature_definitions','selection'))
            or payload.get('sr_data_blinded') is not True or payload.get('expected_only') is not True
            or payload.get('systematics') is not False or set(payload.get('years',{}))!={year}
            or any(k in payload for k in ('model','checkpoint','checkpoint_sha256','score_edges'))):
        raise ValueError('Incomplete/different MET-category export')
    provenance=payload.get('provenance',{})
    if (provenance.get('partition')!='all_mc' or provenance.get('neural_network_used') is not False
            or provenance.get('independent_validation') is not False
            or provenance.get('parent_population')!=candidate['parent_populations'][year]):
        raise ValueError('MET-category parent population/weight identities changed')
    subsets=payload.get('selection_population',{})
    identities=('event_population_sha256','physical_weights_sha256')
    if (set(subsets)!={'selected','rejected'} or any(
            not isinstance(subsets[part],dict) or set(subsets[part])!=set(identities)
            or any(not re.fullmatch(r'[0-9a-f]{64}',subsets[part].get(k,'')) for k in identities)
            for part in subsets)):
        raise ValueError('Missing selected/rejected event and physical-weight identities')
    if candidate['full_tr_retained'] and (
            subsets['selected']!={k:candidate['parent_populations'][year][k] for k in identities}
            or any(subsets['rejected'][k]!=hashlib.sha256(b'').hexdigest() for k in identities)):
        raise ValueError('S0 selected population must equal its parent and have an empty rejected population')
    bins=[dict(category=g['id'],bin=i+1,met_low=low,met_high=high)
          for g in candidate['category_definition']['groups']
          for i,(low,high) in enumerate(zip(g['met_edges'],g['met_edges'][1:]))]
    if payload.get('bins')!=bins:raise ValueError('MET-category bin labels or open endpoint changed')
    data=payload['years'][year];coverage=payload.get('coverage',{})
    counts={k:coverage.get(k) for k in ('source_events','selected_events','rejected_events','assigned_events')}
    if (data.get('partition')!='all_mc' or data.get('weight_scale')!=1. or data.get('variance_scale')!=1.
            or data.get('luminosity_fb')!={'2024':109.82,'2025':110.84}[year]
            or any(type(v) is not int or v<0 for v in counts.values())
            or type(data.get('events')) is not int or data['events']!=counts['selected_events']
            or counts['assigned_events']!=counts['selected_events']
            or counts['source_events']!=counts['selected_events']+counts['rejected_events']
            or coverage.get('full_tr_retained') is not candidate['full_tr_retained']
            or coverage.get('full_tr_accounted') is not True or coverage.get('unassigned_events')!=0
            or coverage.get('multiply_assigned_events')!=0
            or (candidate['full_tr_retained'] and counts['rejected_events']!=0)
            or payload.get('statistical_support')!=candidate['support'][year]):
        raise ValueError('MET-category normalization or selected/rejected event accounting changed')
    backgrounds=payload.get('background_processes',[]);signals=payload.get('signal_names',[])
    samples=data.get('samples',{});totals=coverage.get('sample_totals',{})
    exact='global_cuts' in candidate['selection']
    if exact:
        evidence='shap_guided' if 'shap_guided' in candidate else 'exact_kinematics'
        if payload.get(evidence)!=candidate[evidence]:
            raise ValueError('Exact global-cut provenance differs from its contract')
        stack=np.asarray([[samples[s][k] for k in ('sumw','sumw2','entries')] for s in samples],float)
        if stack.shape!=(len(samples),3,nbins) or not np.isfinite(stack).all():
            raise ValueError('Malformed global-cut template moments')
        empty=np.flatnonzero(np.all(stack==0,axis=(0,1))).tolist()
        if empty!=candidate['support'][year]['empty_all_samples_bins']:
            raise ValueError('A declared constant zero-bin contains MC events or moments')
        b=np.asarray([samples[s]['sumw'] for s in backgrounds],float)
        v=np.asarray([samples[s]['sumw2'] for s in backgrounds],float)
        invalid=np.flatnonzero(np.any(b<-1e-9,axis=0)|(b.sum(axis=0)<=0)|(v.sum(axis=0)<=0)).tolist()
        if invalid!=candidate['support'][year]['hard_invalid_bins']:
            raise ValueError('Global-cut statistical support was not computed from the actual arrays')
    if (not signals or len(set(backgrounds+signals))!=len(backgrounds+signals)
            or set(samples)!=set(backgrounds+signals) or set(totals)!=set(samples)):
        raise ValueError('MET-category sample inventory/totals incomplete or duplicated')
    events=dict(source=0,selected=0,rejected=0)
    for sample,record in samples.items():
        for field in ('entries','sumw','sumw2'):
            values=np.asarray(record.get(field,[]),float)
            pieces={part:totals[sample].get(part,{}).get(field) for part in events}
            if (values.shape!=(nbins,) or not np.isfinite(values).all()
                    or (field!='sumw' and (values<0).any())
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in pieces.values())
                    or (field!='sumw' and any(v<0 for v in pieces.values()))
                    or (field=='entries' and (np.any(values!=np.floor(values))
                        or any(v!=math.floor(v) for v in pieces.values())))
                    or (candidate['full_tr_retained'] and pieces['rejected']!=0)):
                raise ValueError('Invalid MET-category signed moment/count: '+sample+'/'+field)
            matches=(lambda a,b:a==b) if field=='entries' else (lambda a,b:math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7))
            if (not matches(math.fsum(values),pieces['selected'])
                    or not matches(math.fsum((pieces['selected'],pieces['rejected'])),pieces['source'])):
                raise ValueError('MET-category selected+rejected conservation failed: '+sample+'/'+field)
            if field=='entries':
                for part in events:events[part]+=pieces[part]
    if any(events[part]!=counts[part+'_events'] for part in events):
        raise ValueError('MET-category sample event counts differ from parent accounting')
    origin=dict(path=str(path),sha256=candidate['inputs'][year]['sha256'])
    # Retain the legacy metadata field without interpreting it as impossibility.
    # Eligibility is decided by the actual shared template/workspace builders.
    return payload,origin


def comparison_background(cache):
    """Recover corrected nominal MC, removing only the cached data-driven RZ.

    Event SFs stay applied. No Sgamma initial, CR observation, rate parameter,
    or systematic endpoint enters this explicitly SR-only sensitivity study.
    """
    from TROTASR.utils.statistics_model import rz_value
    payload = cache['background']
    result = dict(year=payload['year'], groups={})
    for mode in ('highdm', 'lowdm'):
        rows = []
        for original in payload['groups']['SR_' + mode]:
            row = dict(original, backgrounds={}, rate_params={}, rate_initial={},
                       extra_lnN={}, physical={}, observation=None)
            for name, record in original['backgrounds'].items():
                scale = 1.
                if name.startswith('Zto2Nu_'):
                    group = name.split('_')[1]
                    scale = rz_value(payload['rz_covariance'], mode + '_' + group)
                row['backgrounds'][name] = dict(
                    nominal=[float(record['nominal'][0]) / scale],
                    sumw2=[float(record['sumw2'][0]) / scale**2], variations={})
            rows.append(row)
        result['groups']['SR_' + mode] = rows
    return result


def read_medium_nominal(path):
    """Only the explicitly delegated compact medium-SR export may cross roots."""
    import hashlib
    import json
    allowed = (ROOT.parent / 'sidestudy/topology_transition/statistics/nominal').resolve()
    path = Path(path).resolve()
    if (path.parent != allowed or path.name not in ('medium_2024.json', 'medium_2025.json')
            or allowed != ROOT.parent / 'sidestudy/topology_transition/statistics/nominal'):
        raise ValueError('Not an approved medium nominal input')
    raw = path.read_bytes(); payload = json.loads(raw)
    year = path.stem.rsplit('_', 1)[1]
    if (payload.get('schema') != 'medium_dm_nominal_templates_v1'
            or payload.get('status') != 'complete' or not payload.get('sr_data_blinded')
            or not payload.get('expected_only') or payload.get('reoptimized') is not False
            or set(payload.get('years', {})) != {year} or len(payload.get('bins', [])) != 20
            or payload.get('model', {}).get('checkpoint_sha256') !=
               '1c2e6ff627ff2a54fd715563f4736fd08814640accf6c7dc4ed3a7ebbb19cb16'
            or payload.get('provenance', {}).get('frozen_definition_sha256') !=
               '88f80d6e1ca1acb83829d76473ab4fcbff996de806bd13e82e61e68ab8e09b4e'):
        raise ValueError('Incomplete/different frozen medium input')
    return payload, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def comparison_card(channels, point, directory):
    """A nominal-only card with no CR, rateParam, shape or lnN nuisance."""
    selected = sorted(channels, key=lambda c: (c.rsplit('_', 1)[1], c))
    names = sorted({p for c in channels.values() for p in c['backgrounds']})
    ids = {p: i + 1 for i, p in enumerate(names)}; ids['signal'] = 0
    columns = [(c, p) for c in selected for p in
               (['signal'] if c in point['signal_channels'] else []) + channels[c]['backgrounds']]
    lines = ['imax ' + str(len(selected)), 'jmax *', 'kmax *', '------------']
    for c in selected:
        spec = channels[c]
        path = os.path.relpath(internal_path(ROOT / spec['template']), directory)
        lines.append('shapes * {} {} $CHANNEL/$PROCESS'.format(c, path))
        if c in point['signal_channels']:
            lines.append('shapes signal {} {} $CHANNEL/signal_{}_{}'.format(c, path, point['model'], point['mass']))
    lines += ['------------', 'bin ' + ' '.join(selected),
              'observation ' + ' '.join('-1' for c in selected), '------------',
              'bin ' + ' '.join(c for c, p in columns),
              'process ' + ' '.join(p for c, p in columns),
              'process ' + ' '.join(str(ids[p]) for c, p in columns),
              'rate ' + ' '.join('-1' for c, p in columns), '* autoMCStats 10 1 1']
    return '\n'.join(lines) + '\n'


def single_comparison_channels(component, variant):
    """Select existing high/low SR templates verbatim; never regenerate histograms."""
    if (component.get('variant') != 'baseline' or component.get('scope') != COMPARISON_SCOPE
            or component.get('status') != 'cards_ready'):
        raise ValueError('A single region requires the completed nominal high+low baseline')
    if variant not in ('high','low'): raise ValueError('Expected a high-only or low-only variant')
    mode, bins = ('highdm',118) if variant == 'high' else ('lowdm',30)
    selected = {c:spec for c,spec in component['channels'].items() if c.startswith('SR_'+mode+'_')}
    if {c:s['bins'] for c,s in selected.items()} != {f'SR_{mode}_c1_{y}':bins for y in ('2024','2025')}:
        raise ValueError('Single-region comparison must retain both original SR channels and bins')
    files = {s['template']:component['files'][s['template']] for s in selected.values()}
    return selected, files


def single_comparison_signals(specifications, component):
    import contextlib
    import numpy as np
    import uproot
    information = {}
    with contextlib.ExitStack() as stack:
        handles = {f:stack.enter_context(uproot.open(ROOT/f,object_cache=None,array_cache=None))
                   for f in {s['template'] for s in specifications.values()}}
        background = {}; variance = {}
        for c,spec in specifications.items():
            source = handles[spec['template']]
            background[c] = sum(source[c+'/'+p].values() for p in spec['backgrounds'])
            variance[c] = sum(source[c+'/'+p].variances() for p in spec['backgrounds'])
        for point in component['points']:
            active = sorted(c for c in point['signal_channels'] if c in specifications)
            sensitivity = 0.
            for c in active:
                spec = specifications[c]
                h = handles[spec['template']][c+'/signal_'+point['model']+'_'+point['mass']]
                values, errors = h.values(), h.variances()
                if (values.shape != (spec['bins'],) or errors.shape != (spec['bins'],)
                        or not np.isfinite(values).all() or not np.isfinite(errors).all()
                        or (values < 0).any() or (errors < 0).any()
                        or h.axis().labels() != spec['labels']):
                    raise ValueError('Existing single-SR signal template is invalid')
                sensitivity += float(np.sum(values**2/np.maximum(background[c]+variance[c],1.)))
            issues = [issue for issue in point.get('signal_issues',[])
                      if not issue.get('channel') or issue['channel'] in specifications]
            information[point['model'],point['mass']] = dict(signal_channels=active,signal_issues=issues,
                rmax_seed=min(100000.,max(.01,8./math.sqrt(sensitivity))) if sensitivity > 0 else 100000.)
    return information


def build_comparison(variant, output, parent_grid, medium_inputs=(), component_grids=(), medium_candidate=None,
                     medium_test_comparison=None):
    """Build a separate nominal-MC SR comparison; never alter production inputs."""
    import numpy as np
    if sys.platform != 'linux':
        raise RuntimeError('Comparison ROOT files must remain on hep2')
    output, parent_grid = internal_path(output), internal_path(parent_grid)
    if ROOT / 'stats/topology_transition_nominal' not in output.parents:
        raise ValueError('Comparison outputs belong under stats/topology_transition_nominal')
    parent = read_json(parent_grid)
    from TROTASR.workflows.combine_limits import verify_combined_grid
    verify_combined_grid(parent)
    if parent.get('status') != 'cards_ready' or parent.get('scope') != 'full_combined_systematic_production':
        raise ValueError('Require the actual current 955-point production grid')
    if variant not in COMPARISON_VARIANTS:
        raise ValueError('Unknown comparison variant')
    if medium_candidate and medium_test_comparison:
        raise ValueError('Validation optimization and frozen test evaluation are separate contracts')
    candidate = load_medium_candidate(medium_candidate) if medium_candidate else None
    cutbased = candidate is not None and candidate.get('schema') == CUTBASED_SCHEMA
    met_categories = candidate is not None and candidate.get('schema') == MET_CATEGORY_SCHEMA
    test = load_medium_test_comparison(medium_test_comparison) if medium_test_comparison else None
    medium_bins = candidate_bin_count(candidate) if candidate else test_comparison_bin_count(test) if test else 20
    def medium_input(path):
        return (read_medium_validation(path,candidate) if candidate else
                read_medium_test_comparison(path,test) if test else read_medium_nominal(path))
    if candidate:
        expected_output = ROOT/'stats/topology_transition_nominal/combined_optimization'/candidate['candidate_id']/variant/'grid'
        if variant not in ('medium','combined') or output != expected_output:
            raise ValueError('Validation candidates require their own medium/combined output directory')
    if test:
        expected_output=ROOT/'stats/topology_transition_nominal/test_comparison'/test['candidate_id']/variant/'grid'
        if variant not in ('medium','combined') or output!=expected_output:
            raise ValueError('Frozen tests require their own before/after medium/combined output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'manifest.json').exists():
            old = read_json(output / 'manifest.json')
            if old['variant'] != variant or old['parent_grid_sha256'] != sha256(parent_grid):
                raise ValueError('Preserve divergent comparison')
            if old.get('medium_candidate') != candidate:
                raise ValueError('Preserve comparison with a different validation candidate')
            if old.get('medium_test_comparison') != test:
                raise ValueError('Preserve comparison with a different frozen test definition')
            for origin in old['origins']:
                path = Path(origin['path'])
                actual = (medium_input(path)[1]['sha256']
                          if path.is_absolute() else sha256(ROOT/path))
                if actual != origin['sha256']:
                    raise ValueError('Preserve comparison made from different inputs')
            from TROTASR.workflows.combine_limits import verify_card, IntegrityCache
            digest = IntegrityCache()
            for point in old['points']:
                verify_card(point, digest)
            return 0
        templates = output / 'templates'; channels = {}; data = {}; sources = {}; caches = {}
        origins = []; clipping = []; medium = {}; files = {}; specifications = {}; coverage = {}
        requested = [(p['model'], p['mass']) for p in parent['points']]
        if variant == 'combined':
            components = [read_json(internal_path(p)) for p in component_grids]
            if ({p['variant'] for p in components} != {'baseline', 'medium'} or len(components) != 2
                    or any(p['scope'] != COMPARISON_SCOPE or p['parent_grid_sha256'] != sha256(parent_grid)
                           or [(q['model'], q['mass']) for q in p['points']] != requested for p in components)):
                raise ValueError('Combined comparison requires matching baseline and medium grids')
            medium_component = next(p for p in components if p['variant']=='medium')
            if medium_component.get('medium_candidate') != candidate:
                raise ValueError('Combined candidate must match the exact medium component')
            if medium_component.get('medium_test_comparison') != test:
                raise ValueError('Combined test must match the exact frozen medium component')
            if candidate or test:
                baseline_path = next(p for p,c in zip(component_grids,components) if c['variant']=='baseline')
                if internal_path(baseline_path) != ROOT/FIXED_BASELINE_GRID or sha256(baseline_path) != FIXED_BASELINE_SHA256:
                    raise ValueError('Candidate combination must reuse the fixed high+low baseline')
            for path, component in zip(component_grids, components):
                if set(specifications) & set(component['channels']):
                    raise ValueError('Duplicated SR channel')
                specifications.update(component['channels'])
                origins.append(dict(path=str(internal_path(path).relative_to(ROOT)), sha256=sha256(path)))
                files.update(component['files'])
                coverage.update(component.get('signal_grid_coverage', {}))
            components = {c['variant']: c for c in components}
        elif variant in ('high','low'):
            if len(component_grids) != 1 or medium_inputs:
                raise ValueError('Single-region comparison reuses exactly one completed baseline grid')
            component_path = internal_path(component_grids[0]); component = read_json(component_path)
            if (component.get('parent_grid_sha256') != sha256(parent_grid)
                    or [(p['model'],p['mass']) for p in component['points']] != requested):
                raise ValueError('Single-region parent or common mass grid differs')
            specifications, files = single_comparison_channels(component,variant)
            origins.append(dict(path=str(component_path.relative_to(ROOT)),sha256=sha256(component_path)))
            for path,digest in files.items():
                if sha256(ROOT/path) != digest: raise ValueError('Existing baseline template changed')
        else:
            if variant == 'baseline':
                for year in ('2024', '2025'):
                    path = parent_grid.parent / 'inputs' / (year + '.json.gz')
                    cache = read_json(path); caches[year] = cache
                    if cache['status'] != 'complete' or str(cache['year']) != year or not cache['sr_data_blinded']:
                        raise ValueError('Incomplete baseline extraction')
                    origins.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
                    for masses in cache['signals'].values():
                        for point in masses.values():
                            for record in point.values(): record['variations'] = {}
                channels, data, clipping, issues = assemble([comparison_background(caches[y]) for y in ('2024', '2025')])
                sources = {(y, name): entries for y, cache in caches.items() for name, entries in cache['source_bins'].items()}
                groups = bin_groups(load_merge(parent['sr_merge']))
                original_sources = dict(sources)
                for c, bins in list(channels.items()):
                    if not c.startswith('SR_highdm_'): continue
                    projected = []
                    for i, members in enumerate(groups):
                        row = dict(bins[members[0]], name='SR_highdm_bin' + str(i))
                        sources[row['year'], row['name']] = [entry for j in members
                            for entry in original_sources[row['year'], bins[j]['name']]]
                        projected.append(row)
                    channels[c] = projected
                    for (cc, p), record in data.items():
                        if cc == c:
                            for field in ('nominal', 'sumw2'): record[field] = project(record[field], groups)
            else:
                aliases = dict(TT='Top', ST='Top', VV='VV_VVV', DY='DY', GJ='PhotonJet',
                               WtoLNu='WtoLNu', Zto2Nu='Zto2Nu', QCD='QCD')
                payloads = []
                for path in medium_inputs:
                    payload, origin = medium_input(path)
                    year = next(iter(payload['years']))
                    if year in medium: raise ValueError('Duplicate medium year')
                    medium[year] = payload; origins.append(origin)
                    if set(payload['background_processes']) != set(aliases):
                        raise ValueError('Changed medium background groups')
                    rows = []
                    for i in range(medium_bins):
                        background = {}
                        for sample, family in aliases.items():
                            record = payload['years'][year]['samples'][sample]
                            values, variance = np.asarray(record['sumw']), np.asarray(record['sumw2'])
                            if (values.shape != (medium_bins,) or variance.shape != (medium_bins,) or
                                    not np.isfinite(values).all() or not np.isfinite(variance).all() or (variance < 0).any()):
                                raise ValueError('Invalid medium nominal background')
                            dest = background.setdefault(family, dict(nominal=[0.], sumw2=[0.], variations={}))
                            dest['nominal'][0] += float(values[i]); dest['sumw2'][0] += float(variance[i])
                        rows.append(dict(name='SR_mediumdm_bin' + str(i), region='SR', regime='mediumdm',
                            observation=None, backgrounds=background, rate_params={}, rate_initial={}, extra_lnN={}, physical={}))
                    payloads.append(dict(year=year, groups={'SR_mediumdm': rows}))
                if set(medium) != {'2024', '2025'} or medium['2024']['bins'] != medium['2025']['bins']:
                    raise ValueError('Both completed years with identical declared medium bins are required')
                requested_names = {mass if model == 'T2tt' else model+'_'+mass for model,mass in requested}
                for year, payload in medium.items():
                    declared = set(payload['signal_names'])
                    additional = sorted(declared-requested_names)
                    coverage[year] = dict(parent_comparison_points=len(requested_names),
                        medium_export_points=len(declared), additional_medium_points=additional,
                        additional_medium_mStop_above_1800=[s for s in additional if
                            mass_pair(s[s.index('mStop'):])[0] > 1800],
                        missing_requested_points=sorted(requested_names-declared),
                        policy='Comparison uses the current parent 955-point grid; additional medium points are explicitly retained in this inventory, not claimed fitted')
                    if (cutbased or met_categories) and coverage[year]['missing_requested_points']:
                        raise ValueError('Cut-based export must retain every requested baseline signal')
                channels, data, clipping, issues = assemble(payloads)
            if issues: raise ValueError('Unsupported nominal MC component: ' + str(issues))
            if templates.exists(): raise ValueError('Interrupted templates retained; explicit resume required')
            templates.mkdir()
            write_templates(templates, channels, data)
            verify_templates(templates, channels, data)
            specifications = {c: dict(bins=len(bins), template=str((templates/template_name(c)).relative_to(ROOT)),
                backgrounds=sorted(p for cc, p in data if cc == c), labels=[b['name'] for b in bins]) for c, bins in channels.items()}
        signal_information = {}; unsupported = []; signal_payloads = {}
        if variant in ('high','low'):
            signal_information = single_comparison_signals(specifications,component)
        elif variant == 'combined':
            for i, (model, mass) in enumerate(requested):
                parts = [c['points'][i] for c in components.values()]
                signal_information[model, mass] = dict(signal_channels=[ch for p in parts for ch in p['signal_channels']],
                    rmax_seed=min(p['rmax_seed'] for p in parts), signal_issues=[issue for p in parts for issue in p.get('signal_issues', [])])
        else:
            import ROOT as pyroot
            handles = {name: pyroot.TFile(str(templates/name), 'UPDATE') for name in sorted({template_name(c) for c in channels})}
            if any(h.IsZombie() for h in handles.values()): raise OSError('Cannot update nominal signal templates')
            try:
                for model, mass in requested:
                    if variant == 'baseline':
                        signals, issues, negative = point_signals(channels, sources, caches, model, mass)
                    else:
                        signals = {}; issues = []; negative = []
                        sample = mass if model == 'T2tt' else model + '_' + mass
                        for c in channels:
                            year = c.rsplit('_', 1)[1]
                            if sample not in medium[year]['signal_names']:
                                raise ValueError('Requested signal absent from medium input: ' + sample)
                            r = medium[year]['years'][year]['samples'][sample]
                            values, variance = np.asarray(r['sumw'], float), np.asarray(r['sumw2'], float)
                            if (values.shape != (medium_bins,) or variance.shape != (medium_bins,) or not np.isfinite(values).all()
                                    or not np.isfinite(variance).all() or (variance < 0).any()):
                                raise ValueError('Invalid medium signal')
                            for i in np.flatnonzero(values < 0):
                                negative.append(dict(channel=c, bin=int(i), mass=mass, before=float(values[i]), after=0.))
                            signals[c, 'signal'] = dict(nominal=np.maximum(values, 0), sumw2=variance, variations={})
                    clipping.extend(negative)
                    active = {k:r for k,r in signals.items() if float(np.sum(r['nominal'])) > 0}
                    for (c, p), r in signals.items():
                        if (c, p) not in active and np.any(r['sumw2']):
                            issues.append(dict(channel=c, problem='zero_signal_total_with_variance'))
                    signal_channels = sorted(c for c, p in active)
                    information = 0.
                    for (c, p), r in active.items():
                        background = sum(v['nominal'] for (cc, pp), v in data.items() if cc == c)
                        variance = sum(v['sumw2'] for (cc, pp), v in data.items() if cc == c)
                        information += float(np.sum(r['nominal']**2 / np.maximum(background + variance, 1.)))
                    signal_information[model, mass] = dict(signal_channels=signal_channels, signal_issues=issues,
                        rmax_seed=min(100000., max(.01, 8./math.sqrt(information))) if information > 0 else 100000.)
                    signal_payloads[model, mass] = active
                    put_signals(handles, channels, active, 'signal_' + model + '_' + mass)
            finally:
                for handle in handles.values(): handle.Close()
            # Background content and observations must survive appending named signals.
            verify_templates(templates, channels, data)
            import uproot
            for name in sorted(handles):
                with uproot.open(templates/name, object_cache=None, array_cache=None) as source:
                    for (model, mass), records in signal_payloads.items():
                        for (c, p), record in records.items():
                            if template_name(c) != name: continue
                            h = source[c+'/signal_'+model+'_'+mass]
                            np.testing.assert_allclose(h.values(), record['nominal'], rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(h.variances(), record['sumw2'], rtol=1e-12, atol=1e-12)
                            if h.axis().labels() != specifications[c]['labels']:
                                raise ValueError('Comparison signal labels changed')
            files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(templates.glob('*.root'))}
        for path, digest in files.items():
            if sha256(ROOT/path) != digest: raise ValueError('Comparison template changed')
        model_manifest = dict(scope=COMPARISON_SCOPE, variant=variant, status='templates_ready',
            root_integrity_checked=True, files=files, channels=specifications, origins=origins,
            treatment='Corrected nominal MC; no RZ/Sgamma data-driven central factors, CR observations, rateParams, shape/lnN or signal-theory nuisance; autoMCStats 10 1 1',
            mc_statistics_assumption='Per-year/per-channel autoMCStats as in the current baseline. Shared physical signal-MC reuse across 2024/2025 is not modelled as statistical covariance; 2025 is not claimed an independent held-out test sample.',
            sr_data_blinded=True, parent_grid_sha256=sha256(parent_grid))
        if candidate:
            model_manifest['medium_candidate'] = candidate
        if cutbased:
            model_manifest.update(evaluation_partition='all_mc',neural_network_used=False,
                independent_validation=False,optimization_population=candidate['populations'],
                historical_partitions_used=candidate['historical_partitions_used'],
                statistical_caveat=CUTBASED_CAVEAT)
        if met_categories:
            model_manifest.update(evaluation_partition='all_mc',neural_network_used=False,
                independent_validation=False,parent_population=candidate['parent_populations'],
                selection=candidate['selection'],full_tr_retained=candidate['full_tr_retained'],full_tr_accounted=True,
                historical_partitions_used=candidate['historical_partitions_used'],statistical_caveat=met_category_caveat(candidate),
                nominal_input_policy='adopted_clip_negative_content_preserve_sumw2',
                finite_mc_support_is_diagnostic_only=True)
        if test:
            model_manifest.update(medium_test_comparison=test,evaluation_partition='test',
                test_support_is_diagnostic_only=True,reselection_allowed=False,rebinning_allowed=False,
                medium_test_provenance=(medium_component['medium_test_provenance'] if variant=='combined' else
                    {year:{k:payload['provenance'][k] for k in TEST_COMPARISON_IDENTITIES}
                     for year,payload in medium.items()}))
        template_manifest = output/'model_manifest.json'; write_json(template_manifest, model_manifest)
        card_dir = output/'datacards'; card_dir.mkdir()
        points = []
        for model, mass in requested:
            info = signal_information[model, mass]
            ready = bool(info['signal_channels']) and not info['signal_issues']
            if not ready: unsupported.append(dict(model=model, mass=mass, issues=info['signal_issues']))
            point = dict(model=model, mass=mass, fit_ready=ready, **info)
            card = card_dir/('datacard_'+model+'_'+mass+'.txt')
            meta = dict(scope=COMPARISON_SCOPE, variant=variant, topology=model,
                mStop=mass_pair(mass)[0], mLSP=mass_pair(mass)[1], years=['2024','2025'],
                auto_mc_stats=[10,1,1], cr_only=False, sr_data_blinded=True, signal_channels=info['signal_channels'],
                comparison_manifest=str(template_manifest.relative_to(ROOT)), comparison_manifest_sha256=sha256(template_manifest))
            if test:meta['medium_test_comparison_sha256']=test['contract_sha256']
            if cutbased:meta['medium_cutbased_candidate_sha256']=candidate['contract_sha256']
            if met_categories:meta['medium_met_category_candidate_sha256']=candidate['contract_sha256']
            digest = write_card_once(card, comparison_card(specifications, point, card_dir), meta)
            points.append(dict(point, card=str(card.relative_to(ROOT)), card_sha256=digest))
        result = dict(model_manifest, status='cards_ready', points=points, blocked=unsupported,
            expected_points_by_model=parent['expected_points_by_model'],
            bins=sum(c['bins'] for c in specifications.values()), auto_mc_stats=[10,1,1],
            sr_bins_per_year=({'highdm':118,'lowdm':30,'mediumdm':0} if variant=='baseline' else
                              {'highdm':118,'lowdm':0,'mediumdm':0} if variant=='high' else
                              {'highdm':0,'lowdm':30,'mediumdm':0} if variant=='low' else
                              {'highdm':0,'lowdm':0,'mediumdm':medium_bins} if variant=='medium' else
                              {'highdm':118,'lowdm':30,'mediumdm':medium_bins}),
            sr_merge=parent['sr_merge'], parent_grid=str(parent_grid.relative_to(ROOT)),
            signal_grid_coverage=coverage, negative_to_zero=clipping, code_sha256=sha256(Path(__file__)))
        write_json(output/'manifest.json', result)
        print('cards_ready', variant, len(points), 'requested;', len(unsupported), 'unsupported; no fits submitted', flush=True)
        return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'jsons/analysis.json')
    p.add_argument('--output', type=Path, default=ROOT/'stats/run3/grid')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--preserve-unsupported-templates', action='store_true',
        help='Preserve genuine zero-support nominal/endpoint TH1s and report fit incompatibility; never mark them fit-ready')
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--continue-template-audit', type=Path,
        help='Exact old/new contract receipt for reusing written ROOTs; full readback required')
    p.add_argument('--resume-extracted', action='store_true',
        help='Reuse unchanged year extraction after a downstream builder-only repair, before templates exist')
    p.add_argument('--resume-written-templates', action='store_true',
        help='Verify exact existing closed ROOTs after a builder-only repair; never rewrite them')
    p.add_argument('--resume-cards', action='store_true',
        help='Reuse complete immutable templates and hash-bound cards after a card-writer-only repair')
    p.add_argument('--card-workers', type=int, choices=range(1,33), default=1,
        help='Canonical-runtime fork workers for unfinished cards (default: 1)')
    p.add_argument('--template-verify-workers', type=int, choices=range(1,13), default=1,
        help='Bounded file-level exhaustive ROOT readback workers (default: 1)')
    p.add_argument('--template-check-mode', choices=('exhaustive','minimal'), default='minimal',
        help='Explicit user-approved structural-only check for existing written templates')
    p.add_argument('--verify-template-request', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--verify-template-file', help=argparse.SUPPRESS)
    p.add_argument('--verify-template-result', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--sr-only-comparison', choices=COMPARISON_VARIANTS)
    p.add_argument('--parent-grid', type=Path)
    p.add_argument('--medium-inputs', nargs='*', type=Path, default=[])
    p.add_argument('--component-grids', nargs='*', type=Path, default=[])
    p.add_argument('--medium-candidate', type=Path,
        help='Explicit validation-NN or all-MC cut-based candidate contract for fixed-high+low optimization')
    p.add_argument('--medium-test-comparison', type=Path,
        help='Explicit frozen before/after test-only contract; no test-based selection or rebinning')
    a = p.parse_args()
    if a.verify_template_request:
        if not a.verify_template_file or not a.verify_template_result or a.output or a.config:
            p.error('Readback worker requires only request/file/result')
        raise SystemExit(readback_worker(a.verify_template_request, a.verify_template_file, a.verify_template_result))
    if a.verify_template_file or a.verify_template_result:
        p.error('Incomplete ROOT readback worker invocation')
    if not a.output:
        p.error('--output is required')
    if a.resume_written_templates and (a.resume_extracted or a.continue_template_audit or a.validate_only):
        p.error('--resume-written-templates is mutually exclusive with other recovery/validation modes')
    if a.resume_cards and (a.resume_written_templates or a.resume_extracted or a.continue_template_audit or a.validate_only):
        p.error('--resume-cards is mutually exclusive with other recovery/validation modes')
    if a.sr_only_comparison:
        if not a.parent_grid: p.error('--parent-grid is required for the comparison')
        raise SystemExit(build_comparison(a.sr_only_comparison, a.output, a.parent_grid,
                                         a.medium_inputs, a.component_grids, a.medium_candidate,a.medium_test_comparison))
    if a.medium_candidate: p.error('--medium-candidate requires --sr-only-comparison')
    if a.medium_test_comparison: p.error('--medium-test-comparison requires --sr-only-comparison')
    accepted = internal_path(a.output)/'manifest.json'
    if not a.validate_only and accepted.is_file():
        delivered = read_json(accepted)
        if delivered.get('status') == 'cards_ready' and delivered.get('run3_systematics'):
            raise SystemExit(build(a.config, a.output))
    if not a.config: p.error('--config is required before accepted cards exist')
    if not a.validate_only and not a.native_runtime:
        # The histogram interpreter is Python 3.8, whereas the existing native
        # PyROOT/Combine software is CMSSW Python 3.9. Never mix extension paths.
        from TROTASR.workflows.combine_limits import runtime
        env = runtime(internal_path(a.output) / 'software_runtime')
        os.execvpe('python3', ['python3', str(Path(__file__).resolve()),
            '--config', str(internal_path(a.config)), '--output', str(internal_path(a.output)),
            '--native-runtime'] + (['--preserve-unsupported-templates'] if a.preserve_unsupported_templates else [])
            + (['--continue-template-audit', str(internal_path(a.continue_template_audit))]
               if a.continue_template_audit else [])
            + (['--resume-extracted'] if a.resume_extracted else [])
            + (['--resume-written-templates'] if a.resume_written_templates else [])
            + (['--resume-cards'] if a.resume_cards else [])
            + ['--template-verify-workers', str(a.template_verify_workers),
               '--template-check-mode', a.template_check_mode, '--card-workers', str(a.card_workers)], env)
    raise SystemExit(build(a.config, a.output, a.validate_only, a.preserve_unsupported_templates,
                          a.continue_template_audit, a.resume_extracted,
                          a.resume_written_templates, a.template_verify_workers, a.template_check_mode,
                          a.resume_cards, a.card_workers))
