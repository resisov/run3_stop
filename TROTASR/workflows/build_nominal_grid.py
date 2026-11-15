"""Build shared native TH1 templates and the three nominal signal-card grids.

Large histograms are read once per year. Background TH1s are written once;
each signal mass point has a distinct named TH1 in the same four files, as in
the adopted legacy grid builder. All paths and runtime analysis inputs remain
internal. No old fit values or workspaces are consumed.
"""
import argparse
import copy
import fcntl
from uuid import uuid4
import json
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
from TROTASR.utils.io import read_json, write_json, without_provenance_fields
from TROTASR.utils.statistical_templates import (assemble, aggregate, card_text as native_card_text, template_name,
    write_templates, json_ready, load_theory, profiled_theory, add_signal_xsec_lnN)
from TROTASR.workflows.signal_grid_inputs import (MODELS, FIELDS, requested_grid,
    mass_pair, extract_year, signal_data, require_fit_support_issues)
from TROTASR.utils.sr_merge import load_merge, bin_groups, project

CODE = ('workflows/build_nominal_grid.py', 'workflows/signal_grid_inputs.py',
    'utils/statistics_inputs.py', 'utils/statistics_model.py', 'utils/statistical_templates.py',
    'utils/sr_binning.py', 'utils/sr_merge.py', 'utils/renderers/signal_theory.py')
INFERENCE_RUNTIME = ROOT / 'workflows/run_inference.sh'


def card_scientific_metadata(metadata):
    return without_provenance_fields({
        k: v for k, v in metadata.items()
        if k not in ('inputs', 'baseline_card', 'signal_templates_manifest',
                     'data_obs_construction_receipt', 'data_obs_root')
    })


CR12_POLICY = dict(rate_parameter_range=[0.01, 5], highdm_cr_bins_per_region_year=12,
    native_recoil_groups=[[0], [1], [2], [3], [4, 5], [6, 7]],
    recoil_edges=[250, 300, 350, 400, 500, 800, 1500], last_bin_open_ended=True,
    signal_contamination=False, auto_mc_stats=[10, 1, 1],
    preserve_sr_bins=[118, 30], preserve_existing_uncertainty_correlations=True)

RUN3_SYSTEMATICS_POLICY = dict(systematic_eras=['2022', '2022EE', '2023', '2023BPix', '2024', '2025'],
    preserve_existing_nonluminosity_nuisance_names=True,
    retain_background_measurement_uncertainties=True,
    preserve_per_era_gnn=True, preserve_late_inputs=True, auto_mc_stats=[10, 1, 1])

ZJET_FIT_EXCLUSION = dict(preserve_all_root_objects=True, auto_mc_stats=[10, 1, 1],
    targets=[dict(channel='GCR_highdm_c1_2023BPix', process='ZJet')])


def approved_zjet_fit_background(background, issues, policy=None):
    """Apply the fit-only exclusion while retaining stored TH1s."""
    import numpy as np
    if policy is None:
        return background, issues, []
    if scientific_settings(policy) != scientific_settings(ZJET_FIT_EXCLUSION):
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




def scientific_settings(policy):
    """Receipt labels are provenance; numerical/model settings define compatibility."""
    if policy is None:
        return None
    metadata = {'id', 'user_approved', 'approved_date', 'source', 'source_updated', 'scope',
                'luminosity_policy'}
    return {key: value for key, value in policy.items() if key not in metadata}


def combined_input_scope(caches, config):
    from TROTASR.utils.paths import ERAS
    if require_run3_systematics_policy(config):
        if tuple(caches) != ERAS or any(c['scope'] != 'full_combined_systematic_production'
                                       for c in caches.values()):
            raise ValueError('Six-era systematics require six complete combined era inputs')
        return 'full_combined_systematic_production'
    if set(caches) == set(ERAS):
        raise ValueError('Six-era inputs require the adopted systematic settings')
    scopes = {c['scope'] for c in caches.values()}
    if len(scopes) != 1:
        raise ValueError('Mixed validation/production scopes')
    return next(iter(scopes))


def require_run3_systematics_policy(config):
    """Require the adopted six-era endpoint and luminosity settings."""
    policy = config.get('run3_systematics')
    if policy is None:
        return False
    from TROTASR.utils.paths import ERAS
    if (scientific_settings(policy) != scientific_settings(RUN3_SYSTEMATICS_POLICY) or tuple(config.get('years', ())) != ERAS
            or config.get('systematic_evaluation_deferred') is not False
            or not config.get('systematic_scope')):
        raise ValueError('Exact six-era systematic policy required')
    if config.get('luminosity_uncertainties') != {}:
        raise ValueError('Do not double count independent and correlated luminosity')
    require_luminosity_model(config.get('luminosity_model'))
    return True


def require_luminosity_model(model):
    """Use the luminosity factors of the adopted analysis, not an approval receipt."""
    luminosity = read_json(ROOT / 'jsons/analysis.json')['luminosity_model']
    if (model is None or scientific_settings(model) != scientific_settings(luminosity)
            or model.get('central_normalization_unchanged') is not True
            or model.get('early_normalization_fb') != 61.8971):
        raise ValueError('Luminosity must match the adopted correlations and central normalization')
    return luminosity


def require_statistical_revision(policy):
    if policy is not None and scientific_settings(policy) != scientific_settings(CR12_POLICY):
        raise ValueError('CR channel layout, nuisance treatment or MC-statistical settings differ')
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
    luminosity = require_luminosity_model(model)
    rows=[]
    for name,factors in luminosity['components'].items():
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
    if scientific_settings(policy) != dict(mode='shape', auto_mc_stats=[10, 1, 1]):
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
    if (policy.get('preserve_all_root_objects') is not True
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
    if scientific_settings(combination) != scientific_settings(COMBINATION_POLICY) or policy is None:
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




def grid_contract(config_path, config, inventory, validate_only, preserve_unsupported=False, prior=None):
    revision = config.get('statistical_revision')
    require_statistical_revision(revision)
    from TROTASR.utils.paths import EARLY_ERAS
    early = set(config['years']) == set(EARLY_ERAS)
    run3 = require_run3_systematics_policy(config)
    if not run3 and (not validate_only or not (early or set(config['years']) == {'2024', '2025'})):
        raise ValueError('Card production requires the canonical six-era inputs; subsets are validation-only')
    inputs = {}
    for year in config['years']:
        if set(config['years'][year]) != set(FIELDS):
            raise ValueError('Incomplete histogram/measurement configuration')
        inputs[year] = {}
        for key, source in config['years'][year].items():
            path = internal_path(ROOT / source)
            relative = str(path.relative_to(ROOT))
            inputs[year][key] = dict(path=relative)
    contract = dict(inputs=inputs, validate_only=validate_only,
        preserve_unsupported_templates=preserve_unsupported,
        selection_settings=dict(sr_merge=config.get('sr_merge'),
            preliminary_fit_exclusions=scientific_settings(config.get('preliminary_fit_exclusions')),
            additional_fit_exclusions=scientific_settings(config.get('additional_fit_exclusions'))))
    if config.get('early_qcd_fit_exclusions') is not None:
        from TROTASR.workflows.early_qcd_fit_policy import load_early_qcd_fit_policy
        load_early_qcd_fit_policy(config['early_qcd_fit_exclusions'])
        contract['early_qcd_fit_exclusions'] = config['early_qcd_fit_exclusions']
    if early or run3:
        contract['luminosity_uncertainties'] = config['luminosity_uncertainties']
        contract['systematic_evaluation_deferred'] = config.get('systematic_evaluation_deferred', False)
    if run3:
        contract['run3_systematics'] = config['run3_systematics']
        contract['luminosity_model'] = config['luminosity_model']
    if revision:
        contract['statistical_revision'] = revision
    if config.get('background_estimation_uncertainties'):
        contract['background_estimation_uncertainties'] = config['background_estimation_uncertainties']
    if config.get('systematic_combination'):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        if (scientific_settings(config['systematic_combination']) != scientific_settings(COMBINATION_POLICY)
                or not revision):
            raise ValueError('Combined inputs require the adopted CR layout and endpoint convention')
        contract['systematic_combination'] = config['systematic_combination']
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


def inference_runtime_identity():
    return dict(path=str(INFERENCE_RUNTIME))


def write_card_once(path, text, manifest):
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('Divergent datacard retained: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)
    side = path.with_suffix('.manifest.json')
    if side.exists() and card_scientific_metadata(read_json(side)) != card_scientific_metadata(manifest):
        raise FileExistsError('Divergent datacard scientific metadata retained')
    if not side.exists():
        write_json(side, manifest)


def require_template_files(directory, names):
    for name in names:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Saved card template missing or empty: ' + name)


def resume_written_templates(output, contract):
    """Resume the structural check of closed, not-yet-accepted ROOTs."""
    state = read_json(output/'state.json')
    if (state.get('status') != 'verifying_templates'
            or any((output/path).exists() for path in
                   ('manifest.json', 'templates/model_manifest.json', 'cards', 'datacards'))):
        raise ValueError('Written-template resume requires the completed pre-card write stage')
    expected = {template_name('SR_'+mode+'_c1_'+year) for year in contract['inputs']
                for mode in ('highdm', 'lowdm')}
    directory = output/'templates'
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
        raise ValueError('Incomplete or extra written ROOT inventory')
    require_template_files(directory, expected)


def resume_card_inputs(output, prior):
    """Trust the completed template receipt; never reopen its accepted ROOTs."""
    if prior is None or not prior.get('run3_systematics'):
        raise ValueError('Missing six-era template input contract')
    manifest_path = output/'templates/model_manifest.json'
    manifest = read_json(manifest_path)
    if (not same_extraction_contract(manifest.get('contract') or {}, prior) or manifest.get('status') != 'templates_ready'
            or manifest.get('scope') != 'full_combined_systematic_production'
            or manifest.get('root_integrity_checked') is not True
            or manifest.get('sr_data_blinded') is not True
            or manifest.get('fit_compatibility', {}).get('status') != 'ready'
            or manifest.get('template_directory')
            or len(manifest.get('points', [])) != 955):
        raise ValueError('Card resume requires unchanged complete six-era templates and inputs')
    expected = {template_name('SR_'+mode+'_c1_'+year) for year in prior['inputs']
                for mode in ('highdm', 'lowdm')}
    if (len(prior['inputs']) != 6 or set(manifest.get('files', {})) != expected
            or set(manifest.get('object_coverage', {})) != expected):
        raise ValueError('Card resume requires the exact twelve accepted ROOT files')
    require_template_files(manifest_path.parent, expected)
    return manifest


def point_card_metadata(context, point, issues):
    stop, lsp = mass_pair(point['mass'])
    ready = not context['background_issues'] and not issues
    return dict(status='card_written_not_fitted' if ready else 'card_written_fit_blocked',
        fit_ready=ready, fit_issues=context['background_issues']+issues,
        preliminary_fit_exclusions=context['exclusions'], inputs=context['origin'],
        topology=point['model'], mStop=stop, mLSP=lsp, cr_only=False, sr_data_blinded=True,
        full_workflow_complete=False,
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
    if _CARD_CONTEXT.get('era_merge') is not None:
        from TROTASR.utils.era_merge import render_merged_signal_card
        text, metadata = render_merged_signal_card(path, text, metadata, _CARD_CONTEXT['era_merge'])
    write_card_once(path, text, metadata)
    result = dict(point, fit_ready=metadata['fit_ready'],
                  card=str(path.relative_to(ROOT)))
    if metadata.get('era_merged') is True:
        result['era_merged'] = True
    return result


def existing_card(context, point):
    """Reuse a complete card and its scientific receipt without regenerating its text."""
    path = context['output']/'datacards'/('datacard_'+point['model']+'_'+point['mass']+'.txt')
    side = path.with_suffix('.manifest.json')
    if not side.exists():
        return None  # An interrupted TXT alone is compared exactly by write_card_once.
    if not path.is_file():
        raise ValueError('Card provenance without its TXT is retained: '+str(path))
    issues = context['point_issues'].get(point['model']+'/'+point['mass'], [])
    expected = point_card_metadata(context, point, issues)
    saved = read_json(side)
    if context.get('era_merge') is not None:
        from TROTASR.utils.era_merge import merged_card_metadata
        expected = merged_card_metadata(expected, context['era_merge'])
        for key in ('dependent_rateparam_formulas', 'shared_alpha_bounds'):
            if key not in saved:
                raise ValueError('Merged card provenance is incomplete: '+str(path))
            expected[key] = saved[key]
    if card_scientific_metadata(saved) != card_scientific_metadata(expected):
        raise ValueError('Existing card scientific metadata differs: '+str(path))
    result = dict(point, fit_ready=expected['fit_ready'],
                  card=str(path.relative_to(ROOT)))
    if expected.get('era_merged') is True:
        result['era_merged'] = True
    return result


def card_capacity(workers, external, memory, active, free_disk):
    if free_disk < 80*1024**3 or memory < 24*1024**3:
        return 0
    return max(0, min(workers, 120-1-external,
        active+max(0, (memory-32*1024**3)//(8*1024**3))))


def write_point_cards(context, points, workers=1, execution=None):
    """Fork the initialized canonical runtime; share large read-only arrays by COW."""
    global _CARD_CONTEXT
    from TROTASR.workflows.resources import process_identity, resources
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
            reused_pairs[key] = dict(card=saved['card'])
    path = context['output']/'card_execution.json'
    record = dict(status='writing_cards', pid=os.getpid(), started=time.time(), workers=workers,
        reused=len(results), completed=len(results), total=len(points),
        process_model='fork_inherited_canonical_runtime' if workers > 1 else 'serial_canonical_runtime',
        executable=sys.executable, argv=sys.argv, aggregate_slots=120, reserve_GiB=32,
        emergency_GiB=24, disk_floor_GiB=80, root_and_cache_rewrite=False,
        execution=execution, worker_pids=[], worker_identities={}, reused_pairs=reused_pairs,
        controller_identity=process_identity(os.getpid()),
        inherited_runtime=inference_runtime_identity() if workers > 1 else None,
        thread_environment={name:os.environ.get(name) for name in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
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
            require_template_files(card.parent, (card.name, card.with_suffix('.manifest.json').name))
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
    """Input identity survives implementation cleanup; physics inputs must match."""
    ignore = {'code', 'inputs'}
    policies = {'statistical_revision', 'run3_systematics',
                'background_estimation_uncertainties', 'systematic_combination', 'luminosity_model'}
    if 'selection_settings' not in old or 'selection_settings' not in current:
        # Older receipts do not track selection settings separately.
        ignore.add('selection_settings')
    def science(contract):
        return without_provenance_fields({k: scientific_settings(v) if k in policies else v
                for k, v in contract.items() if k not in ignore})
    def fields(contract):
        return {era: set(inputs) for era, inputs in contract.get('inputs', {}).items()}
    return fields(old) == fields(current) and science(old) == science(current)


def same_year_extraction(old, current, year):
    """Cached arrays depend on source inputs and their projection, not fit-only exclusions."""
    fields = ('validate_only', 'preserve_unsupported_templates')
    return (set(old.get('inputs', {}).get(year, {})) == set(current['inputs'][year])
            and all(old.get(key) == current.get(key) for key in fields))




def build(config_path, output, validate_only=False, preserve_unsupported=False,
          card_workers=1):
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
    if not 1 <= card_workers <= 32:
        raise ValueError('Card workers must be between one and 32')
    if not validate_only and sys.platform != 'linux':
        raise RuntimeError('Native ROOT templates must stay on hep2, not the local laptop')
    config = read_json(config_path)
    if not validate_only and not require_run3_systematics_policy(config):
        raise ValueError('Only the canonical six-era merged production grid may write cards')
    revision = config.get('statistical_revision')
    revised = require_statistical_revision(revision)
    if revised and scientific_settings(config.get('background_estimation_uncertainties')) != dict(
            mode='shape', auto_mc_stats=[10, 1, 1]):
        raise ValueError('CR12 requires the exact adopted measured-background shape policy')
    inventory = internal_path(ROOT / config['grid'])
    grid = requested_grid(inventory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract_path = output / 'contract.json'
        prior_contract = read_json(contract_path) if contract_path.exists() else None
        execution_contract = grid_contract(config_path, config, inventory, validate_only, preserve_unsupported,
                                           prior=prior_contract)
        if prior_contract is not None and not same_extraction_contract(prior_contract, execution_contract):
            raise ValueError('Existing grid has different physics inputs; preserve its products')
        contract = prior_contract if prior_contract is not None else execution_contract
        templates = output/'templates'
        template_manifest = templates/'model_manifest.json'
        card_resume = (resume_card_inputs(output, prior_contract)
                       if not validate_only and template_manifest.is_file() else None)
        state_path = output/'state.json'
        saved = read_json(state_path) if state_path.is_file() else {}
        resume_written = not validate_only and card_resume is None and saved.get('status') == 'verifying_templates'
        if resume_written:
            if prior_contract is None:
                raise ValueError('Written templates require the saved input contract')
            resume_written_templates(output, contract)
        if prior_contract is None:
            write_json(contract_path, contract)
        caches = {}
        for year in config['years']:
            path = output / 'inputs' / (year + '.json.gz')
            cache_receipt = path.with_suffix('.receipt.json')
            if path.exists():
                receipt = read_json(cache_receipt)
                if not same_year_extraction(receipt['contract'], contract, year):
                    raise ValueError('Changed or unverified extracted year inputs')
                cache = read_json(path)
            else:
                if card_resume is not None or resume_written or cache_receipt.exists():
                    raise ValueError('Saved year input is missing: '+year)
                write_json(output / 'state.json', dict(status='extracting', year=year,
                    pid=os.getpid(), updated=time.time(), full_workflow_complete=False))
                cache = extract_year(config['years'][year], grid, allow_validation=validate_only,
                                     preserve_unsupported=preserve_unsupported)
            if (cache['status'] != 'complete' or cache['year'] != year or cache.get('sr_data_blinded') is not True
                    or set(cache['inputs']) != set(contract['inputs'][year])):
                raise ValueError('Year input coverage or blinding does not match: '+year)
            if not path.exists():
                write_json(path, json_ready(cache))
                write_json(cache_receipt, dict(contract=contract))
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
        combined = scope == 'full_combined_systematic_production'
        if combined != bool(config.get('systematic_combination')):
            raise ValueError('Combined inputs and acknowledged preliminary policy must agree')
        if combined and (not revised or grid['expected_points_by_model'] != {'T2tt':225, 'T2bW':365, 'T2tb':365}):
            raise ValueError('Combined model requires the exact 955-point CR12 grid')
        model_metadata = dict(years=list(caches), requested_grid=dict(
            path=str(inventory.relative_to(ROOT))))
        if require_run3_systematics_policy(config):
            model_metadata.update(run3_systematics=config['run3_systematics'],
                systematic_evaluation_deferred=False, systematic_scope=config['systematic_scope'],
                input_scopes_by_era={y: c['scope'] for y, c in caches.items()},
                luminosity_uncertainties=contract['luminosity_uncertainties'],
                luminosity_model=contract['luminosity_model'])
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
            points = copy.deepcopy(card_resume['points'])
            expected_points = [dict(model=model, mass=mass) for model in MODELS for mass in grid['models'][model]]
            if points != expected_points:
                raise ValueError('Saved template point inventory changed')
            blocked = copy.deepcopy(card_resume['fit_compatibility']['blocked_signal_points'])
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
        if card_resume and fit_compatibility != card_resume['fit_compatibility']:
            raise ValueError('Saved template/card fit compatibility changed')
        if not card_resume:
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
        if config.get('sr_merge'):
            summary['sr_merge'] = config['sr_merge']
        if config.get('systematic_scope'):
            summary['systematic_scope'] = config['systematic_scope']
        if validate_only or (fit_blocked and not preserve_unsupported):
            write_json(output / 'state.json', summary)
            return 2 if fit_blocked else 0
        validation_fields = ('template_check_mode', 'numeric_readback_performed', 'numeric_readback_checked',
                             'template_check_user_approved', 'root_integrity_scope', 'template_validation')
        if card_resume:
            summary.update({k:card_resume[k] for k in validation_fields if k in card_resume})
        elif templates.exists() and not resume_written:
            raise FileExistsError('Incomplete ROOT template directory retained; explicit recovery required')
        else:
            if not resume_written:
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
                reused_written_templates=resume_written))
            template_inventory = minimal_template_inventory(channels, points, caches, sources, background)
            coverage = {name: verify_minimal_template(templates/name, expected)
                        for name, expected in template_inventory.items()}
            summary.update(template_check_mode='minimal', numeric_readback_performed=False,
                numeric_readback_checked=False,
                root_integrity_scope='file_open_header_object_inventory_types',
                template_validation=dict(mode='minimal', numeric_readback_checked=False,
                    file_open_header=True, object_inventory=True, required_types=True,
                    bin_values=False, sumw2=False, labels=False))
            manifest = dict(status='templates_verified_fit_blocked' if fit_blocked else 'templates_ready',
                scope=scope, contract=contract, fit_compatibility=fit_compatibility, object_coverage=coverage,
                root_integrity_checked=True, sr_data_blinded=True, auto_mc_stats=[10, 1, 1],
                bins=total_bins, channels={c: [b['name'] for b in bins] for c, bins in channels.items()},
                points=points, files=dict.fromkeys(template_inventory),
                full_workflow_complete=False)
            manifest.update(model_metadata)
            manifest.update({k:summary[k] for k in validation_fields if k in summary})
            if config.get('sr_merge'):
                manifest['sr_merge'] = config['sr_merge']
            if config.get('systematic_scope'):
                manifest['systematic_scope'] = config['systematic_scope']
            write_json(template_manifest, manifest)
        if fit_blocked:
            final = dict(summary, status='templates_verified_fit_blocked', points=points,
                root_integrity_checked=True, templates_manifest=str(template_manifest.relative_to(ROOT)),
                finished=time.time())
            write_json(output/'manifest.json', final)
            write_json(output/'state.json', {k:v for k,v in final.items() if k != 'points'})
            return 2
        theory = profiled_theory(load_theory(ROOT / config['theory']))
        write_json(output / 'signal_theory.json', theory)
        origin = [dict(path=str(template_manifest.parent.relative_to(ROOT)))]
        card_context = dict(channels=channels, sources=sources, caches=caches, fit_data=fit_data,
            theory=theory, output=output, templates=templates, revision=revision, config=config,
            preserve_unsupported=preserve_unsupported, background_issues=background_issues,
            exclusions=exclusions, origin=origin, contract=contract, model_metadata=model_metadata,
            point_issues={r['model']+'/'+r['mass']:r['issues'] for r in blocked})
        execution = dict(resume_cards=card_resume is not None, execution_contract=execution_contract,
            preserved_template_contract=contract)
        from TROTASR.utils.era_merge import (REFERENCE_POINT, cr_only_card, displayed_points,
            merged_card_metadata, prepare_merged_cards, publication_metadata)
        selected = displayed_points(points)
        reference = next(point for point in selected if (point['model'], point['mass']) == REFERENCE_POINT)
        reference_path, reference_text, reference_meta = render_point_card(card_context, reference)
        merge = prepare_merged_cards(reference_path, reference_text, reference_meta, channels, background)
        card_context['era_merge'] = merge
        converted = write_point_cards(card_context, selected, card_workers, execution)
        written = {(point['model'], point['mass']):point for point in converted}
        all_points = [written.get((point['model'], point['mass']),
                          dict(point, fit_ready=False, excluded_out_of_plot_range=True)) for point in points]
        # Derive CR-only from the same model without SR shapes or observations.
        cr_path = output / 'datacards' / 'datacard_cronly.txt'
        cr_meta = dict(
            status='card_written_not_fitted' if fit_compatibility['cr_only_ready'] else 'card_written_fit_blocked',
            fit_ready=fit_compatibility['cr_only_ready'], inputs=origin, cr_only=True,
            preliminary_fit_exclusions=exclusions,
            sr_data_blinded=True, full_workflow_complete=False,
            **dict(dict(auto_mc_stats=[10, 1, 1]), **model_metadata))
        write_card_once(cr_path, cr_only_card(merge['reference_plan']), merged_card_metadata(cr_meta, merge))
        final = dict(summary, status='cards_ready', points=all_points, root_integrity_checked=True,
                     templates_manifest=str(template_manifest.relative_to(ROOT)),
                     cr_only_card=dict(card=str(cr_path.relative_to(ROOT))),
                     finished=time.time())
        final.update(publication_metadata(merge))
        final['card_execution'] = dict(path=str((output/'card_execution.json').relative_to(ROOT)))
        write_json(output / 'manifest.json', final)
        write_json(output / 'state.json', {k: v for k, v in final.items() if k != 'points'})
        print(final['status'], len(converted), 'merged cards; fits and AN plots are not complete')
        return 0


def rewrite_replacement_signal_shapes(text, source, target, point, signal_templates):
    """Rebind only the twelve signal aliases; all other objects keep the same ROOT."""
    from TROTASR.utils.paths import ERAS
    from TROTASR.workflows.signal_grid_inputs import replacement_signal_object
    expected = {(f'SR_{mode}_c1', 'signal_'+year):f'SR_{mode}_c1_{year}'
                for year in ERAS for mode in ('highdm','lowdm')}
    lines = []
    seen = set()
    for line in text.splitlines(keepends=True):
        fields = line.split()
        if not fields or fields[0] != 'shapes':
            lines.append(line)
            continue
        if len(fields) < 5:
            raise ValueError('Malformed accepted shape binding')
        channel, process = fields[2], fields[1]
        key = channel, process
        if process.startswith('signal'):
            if (key not in expected or key in seen or len(fields) != 6
                    or fields[4] != replacement_signal_object(expected[key],point)
                    or fields[5] != fields[4]+'_$SYSTEMATIC'):
                raise ValueError('Unexpected accepted signal shape alias: '+str(key))
            seen.add(key)
            root = signal_templates/template_name(expected[key])
        elif fields[3] != 'FAKE':
            root = internal_path(source.parent/fields[3])
        else:
            lines.append(line)
            continue
        root = internal_path(root)
        if not root.is_file() or not root.stat().st_size:
            raise FileNotFoundError(root)
        path = os.path.relpath(root,target.parent)
        if any(character.isspace() for character in path):
            raise ValueError('Whitespace in datacard template path')
        fields[3] = path
        lines.append(' '.join(fields)+('\n' if line.endswith('\n') else ''))
    if seen != set(expected):
        raise ValueError('Expected exactly twelve SR signal shape aliases')
    rendered = ''.join(lines)
    clauses = [r.split() for r in rendered.splitlines()]
    names, ids = [r[1:] for r in clauses if r and r[0]=='process']
    columns = [r[1:] for r in clauses if r and r[0]=='bin'][-1]
    if len(names) != len(ids) or len(columns) != len(ids):
        raise ValueError('Unexpected accepted datacard column count')
    signal_columns = [(c,p) for c,p,pid in zip(columns,names,ids) if int(pid)<=0]
    if len(signal_columns)!=len(expected) or set(signal_columns)!=set(expected):
        raise ValueError('Unexpected physically active signal columns')
    return rendered


def write_replacement_card(path, text, metadata):
    """Atomically publish a new card, then its completion receipt."""
    import tempfile
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix='.card-',dir=path.parent) as temporary:
            candidate = Path(temporary)/path.name
            candidate.write_text(text)
            if path.exists():
                raise FileExistsError('Preserve existing signal card: '+str(path))
            candidate.rename(path)
    return write_card_once(path,text,metadata)


def write_replacement_era(templates, extracted, year, points, labels, highdm_groups):
    """Materialize only the accepted nominal signal aliases in two new ROOTs."""
    import tempfile
    import ROOT as pyroot
    from TROTASR.workflows.signal_grid_inputs import replacement_channel_records, replacement_signal_object
    pyroot.gROOT.SetBatch(True)
    inventory = {template_name(channel):dict(names=sorted(replacement_signal_object(channel,p) for p in points),
                      channels=[channel], background_and_observation_objects=0, signal_objects=len(points))
                 for channel in labels}
    native = {c:[dict(name=label) for label in names] for c,names in labels.items()}
    support, clipped = [], 0
    with tempfile.TemporaryDirectory(prefix='.era-'+year+'-',dir=templates) as temporary:
        directory = Path(temporary)
        handles = {}
        try:
            for name, spec in inventory.items():
                if (templates/name).exists():
                    continue
                handle = pyroot.TFile(str(directory/name),'CREATE')
                if not handle or handle.IsZombie():
                    raise OSError('Cannot create nominal-only replacement ROOT: '+name)
                handles[name] = handle
                if not handle.mkdir(spec['channels'][0]):
                    raise OSError('Cannot create nominal-only replacement channel: '+name)
            for point in points:
                records, issues, audit = replacement_channel_records(extracted,year,point,labels,highdm_groups)
                require_fit_support_issues(issues)
                if any(record['nominal'].sum()>0 for record in records.values()):
                    support.append(point['model']+'/'+point['mass'])
                clipped += len(audit)
                missing = {key:val for key,val in records.items() if template_name(key[0]) in handles}
                if missing:
                    put_signals(handles,native,missing,'signal_'+point['model']+'_'+point['mass'])
        finally:
            for handle in handles.values():
                handle.Close()
        coverage = {name:verify_minimal_template(directory/name if name in handles else templates/name, spec)
                    for name,spec in inventory.items()}
        for name in handles:
            if (templates/name).exists():
                raise FileExistsError('Preserve existing signal template: '+name)
            (directory/name).rename(templates/name)
    return dict(files=dict.fromkeys(inventory),
                object_coverage=coverage, support=support, clipped_source_bins=clipped)


def build_signal_replacement():
    """Publish a signal-only grid under the authorized Run 3 result namespace."""
    if sys.platform != 'linux':
        raise RuntimeError('Replacement ROOTs must stay in the existing hep2 runtime')
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    from TROTASR.gnn4lowdm.configuration import configuration
    from TROTASR.utils.paths import ERAS
    from TROTASR.workflows.resources import resources
    from TROTASR.workflows.signal_grid_inputs import (extract_replacement_year, read_replacement_model,
        replacement_points, replacement_production_receipts, replacement_source_binding)
    baseline = ROOT/'stats/run3/grid'
    output = ROOT/'stats/run3/new_signal/grid'
    production = ROOT/'stats/run3/new_signal/signal_histograms'
    baseline_manifest = baseline/'manifest.json'
    accepted = read_json(baseline_manifest)
    points = replacement_points(accepted)
    config = read_json(ROOT/'jsons/analysis.json')
    require_run3_systematics_policy(config)
    native_manifest = internal_path(ROOT/accepted['templates_manifest'])
    native = read_json(native_manifest)
    contract = read_json(baseline/'contract.json')
    if (native.get('root_integrity_checked') is not True or native.get('sr_data_blinded') is not True
            or not same_extraction_contract(native.get('contract') or {}, contract)
            or native.get('status') != 'templates_ready'):
        raise ValueError('The accepted native template receipt does not match the frozen card model')
    highdm_groups = bin_groups(load_merge(accepted['sr_merge']))
    cr = accepted['cr_only_card']
    cr_path = internal_path(ROOT/cr['card'])
    require_template_files(cr_path.parent, (cr_path.name,))
    def source_bindings(receipts):
        bindings = {}
        for era in ERAS:
            path = internal_path(ROOT/receipts[era]['output'])
            if production not in path.parents:
                raise ValueError('Signal replacement requires corrected inputs in the current signal_histograms stage')
            bindings[era] = replacement_source_binding(path,era,receipts[era])
        return bindings

    output.mkdir(parents=True,exist_ok=True)
    state_path, manifest_path = output/'state.json', output/'manifest.json'
    source_lock = ROOT/'production/histograms/lock'
    with (output/'builder.lock').open('a') as lock, source_lock.open('rb') as generation_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(generation_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        production_state = read_json(production/'state.json')
        receipts = replacement_production_receipts(production_state)
        sources = source_bindings(receipts)
        origin = dict(accepted_grid=str(baseline_manifest.relative_to(ROOT)),
                      accepted_native_templates=str(native_manifest.relative_to(ROOT)), sources=sources)
        if manifest_path.is_file():
            delivered = read_json(manifest_path)
            replacement_points(delivered)
            return 0
        templates, cards = output/'templates', output/'datacards'
        if state_path.is_file():
            state = read_json(state_path)
            revision = state['signal_input_revision']
        elif templates.exists() or cards.exists():
            raise ValueError('Existing signal replacement products lack their input receipt')
        else:
            revision = str(uuid4())
            state = dict(signal_input_revision=revision, origin=origin, status='writing_templates',
                         eras={}, completed_cards=0, sr_data_blinded=True, official_result=False)
            write_json(state_path,state)
        templates.mkdir(exist_ok=True)
        for era in ERAS:
            channels = {f'SR_{mode}_c1_{era}':native['channels'][f'SR_{mode}_c1_{era}']
                        for mode in ('highdm','lowdm')}
            if era in state['eras']:
                require_template_files(templates,state['eras'][era]['files'])
                continue
            external, memory = resources()
            if external >= 115 or memory < 32*1024**3 or shutil.disk_usage(output).free < 80*1024**3:
                raise RuntimeError('Insufficient account slots, memory or disk; completed eras are preserved')
            model_input = read_replacement_model(baseline/'inputs'/(era+'.json.gz'))
            extracted = extract_replacement_year(ROOT/sources[era]['path'],era,points,model_input,configuration(era),
                production_receipt=receipts[era],accepted_contract=contract)
            state['eras'][era] = write_replacement_era(templates,extracted,era,points,channels,highdm_groups)
            state['eras'][era]['projection_audit'] = extracted['projection_audit']
            write_json(state_path,state)
            print('SIGNAL_TEMPLATES',era,len(state['eras']),len(ERAS),flush=True)
        supported = {name for era in state['eras'].values() for name in era['support']}
        missing = [p['model']+'/'+p['mass'] for p in points if p['model']+'/'+p['mass'] not in supported]
        if missing:
            raise ValueError(f'{len(missing)} displayed mass points have no total nominal SR support; first: {missing[0]}')
        template_manifest = templates/'model_manifest.json'
        template_record = dict(status='templates_ready', scope=accepted['scope'], years=list(ERAS),
            root_integrity_checked=True, sr_data_blinded=True, nominal_signal_only=True,
            signal_input_revision=revision, origin=origin, channels=native['channels'],
            files={k:v for row in state['eras'].values() for k,v in row['files'].items()},
            object_coverage={k:v for row in state['eras'].values() for k,v in row['object_coverage'].items()},
            numeric_readback_performed=False)
        if template_manifest.is_file():
            saved_templates = read_json(template_manifest)
            if (set(saved_templates.get('files', {})) != set(template_record['files'])
                    or without_provenance_fields({k:v for k,v in saved_templates.items() if k not in ('files','origin')})
                    != without_provenance_fields({k:v for k,v in template_record.items() if k not in ('files','origin')})):
                raise ValueError('Existing signal template receipt differs')
        else:
            write_json(template_manifest,template_record)
        cards.mkdir(exist_ok=True)
        state['status'] = 'writing_cards'
        write_json(state_path,state)
        def replace(point):
            source = internal_path(ROOT/point['card'])
            if source.parent != baseline/'datacards':
                raise ValueError('Frozen point card is outside the accepted directory')
            target = cards/source.name
            reference = str(template_manifest.relative_to(ROOT))
            side = target.with_suffix('.manifest.json')
            if side.is_file():
                saved = read_json(side)
                if (saved.get('sr_data_blinded') is not True or saved.get('fit_ready') is not True
                        or saved.get('era_merged') is not True or not target.is_file() or not target.stat().st_size):
                    raise ValueError('Preserve different completed signal card: '+str(target))
                return dict(model=point['model'],mass=point['mass'],fit_ready=True,era_merged=True,
                            signal_input_revision=revision,card=str(target.relative_to(ROOT)))
            metadata = read_json(source.with_suffix('.manifest.json'))
            if (metadata.get('fit_ready') is not True or metadata.get('sr_data_blinded') is not True
                    or metadata.get('era_merged') is not True):
                raise ValueError('Frozen point sidecar identity changed: '+point['mass'])
            text = rewrite_replacement_signal_shapes(source.read_text(),source,target,point,templates)
            new = dict(metadata, official_result=False, signal_input_revision=revision,
                       signal_templates_manifest=reference, baseline_card=str(source.relative_to(ROOT)))
            write_replacement_card(target,text,new)
            return dict(model=point['model'],mass=point['mass'],fit_ready=True,era_merged=True,
                        signal_input_revision=revision,card=str(target.relative_to(ROOT)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            fitted = {(p['model'],p['mass']):p for p in pool.map(replace,points)}
        published = [fitted.get((p['model'],p['mass']),dict(model=p['model'],mass=p['mass'],fit_ready=False,
                        excluded_out_of_plot_range=True,status='excluded_by_approved_range')) for p in accepted['points']]
        final = dict(accepted,points=published,official_result=False,signal_input_revision=revision,origin=origin,
            templates_manifest=str(template_manifest.relative_to(ROOT)), signal_templates_nominal_only=True,
            finished=time.time(), observed_sr_used=False)
        for key in ('card_execution','previous_model','previous_model_manifest'):
            final.pop(key,None)
        write_json(manifest_path,final)
        state.update(status='cards_ready', completed_cards=len(fitted), finished=final['finished'])
        write_json(state_path,state)
        print('signal_replacement_cards_ready',len(fitted),'of',len(published),flush=True)
        return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'jsons/analysis.json')
    p.add_argument('--output', type=Path, default=ROOT/'stats/run3/grid')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--replace-signals', action='store_true',
        help='Publish nominal replacement signals to the isolated current new_signal grid')
    p.add_argument('--preserve-unsupported-templates', action='store_true',
        help='Preserve genuine zero-support nominal/endpoint TH1s and report fit incompatibility; never mark them fit-ready')
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--card-workers', type=int, choices=range(1,33), default=1,
        help='Canonical-runtime fork workers for unfinished cards (default: 1)')
    a = p.parse_args()
    if a.replace_signals:
        allowed_outputs = (ROOT/'stats/run3/grid', ROOT/'stats/run3/new_signal/grid')
        if (a.validate_only or a.preserve_unsupported_templates
                or internal_path(a.output) not in allowed_outputs
                or internal_path(a.config) != ROOT/'jsons/analysis.json'):
            p.error('--replace-signals uses only the accepted config and new_signal grid')
        if not a.native_runtime:
            from TROTASR.workflows.combine_limits import runtime
            env = runtime(ROOT/'stats/run3/new_signal/grid/software_runtime')
            os.execvpe('python3',['python3',str(Path(__file__).resolve()),'--replace-signals','--native-runtime'],env)
        raise SystemExit(build_signal_replacement())
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
            + ['--card-workers', str(a.card_workers)], env)
    raise SystemExit(build(a.config, a.output, a.validate_only, a.preserve_unsupported_templates,
                           card_workers=a.card_workers))
