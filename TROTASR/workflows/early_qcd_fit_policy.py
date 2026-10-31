"""Exact approved 2022 SR-only fit omission; never edit native ROOT inputs.

This is a downstream card-scope policy, not an endpoint or zero-yield repair.
The two omitted records remain in the original background/template mapping.
"""
import copy
import json
import math
import re

import numpy as np

from TROTASR.utils.io import read_json, sha256
from TROTASR.utils.paths import ROOT, internal_path


POLICY_PATH = 'jsons/early_qcd_fit_20261027_policy.json'
POLICY_ID = 'early_2022_sr_qcd_fit_only_20261027'
CONFIG_KEY = 'early_qcd_fit_exclusions'
CHANNEL = 'SR_highdm_c1_2022'
CONTROL_CHANNEL = 'QCDCR_highdm_c1_2022'
JER = 'CMS_res_j_13p6TeV'
JES = 'CMS_scale_j_Total_2022'


def expected_policy():
    """Independent strict schema/value guard for the approval document."""
    return dict(id=POLICY_ID, user_approved=True, fit_only=True,
        preserve_all_root_objects=True, preserve_control_regions=True,
        preserve_rate_parameters=True, auto_mc_stats=[10, 1, 1],
        guard=dict(sr_bins=118, control_bins=12, exact_nonzero_support=True,
                   rtol=1e-12, atol=0.0),
        targets=[
            dict(channel=CHANNEL, process='QCD_Nb1_u5',
                native_parameter='CMS_NPS26012_qcd_norm_highdm_Nb1_bin5_2022',
                parameter='CMS_NPS26012_qcd_norm_highdm_Nb1_bin4_2022',
                nominal=[], sumw2=[], variations={
                    JER: dict(up=[], down=[[54, 0.009758691988466295]]),
                    JES: dict(up=[], down=[[54, 0.009758929269378316]])},
                control_channel=CONTROL_CHANNEL,
                control_process='QCD_Nb1_u4',
                control_nominal=[[4, 10.63255264466289]],
                control_sumw2=[[4, 0.44481395096630316]]),
            dict(channel=CHANNEL, process='QCD_Nb2plus_u5',
                native_parameter='CMS_NPS26012_qcd_norm_highdm_Nb2plus_bin5_2022',
                parameter='CMS_NPS26012_qcd_norm_highdm_Nb2plus_bin4_2022',
                nominal=[[77, 0.009966592479303288]],
                sumw2=[[77, 9.933296564850486e-05]], variations={
                    JER: dict(up=[[77, 0.009948720283384062]], down=[]),
                    JES: dict(up=[[77, 0.009952991528752954]], down=[])},
                control_channel=CONTROL_CHANNEL,
                control_process='QCD_Nb2plus_u4',
                control_nominal=[[10, 7.804977220854109]],
                control_sumw2=[[10, 0.5363446357283476]])])


def load_early_qcd_fit_policy(reference):
    """Require an exact approved in-package document and its actual hash."""
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or reference.get('path') != POLICY_PATH
            or not isinstance(reference.get('sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256'])):
        raise ValueError('Invalid exact early-QCD policy reference')
    path = internal_path(ROOT / POLICY_PATH)
    if sha256(path) != reference['sha256']:
        raise ValueError('Approved early-QCD policy hash changed')
    policy = read_json(path)
    if policy != expected_policy():
        raise ValueError('Unapproved early-QCD fit-only policy')
    return policy


def approved_early_qcd_config_change(prior, current):
    """Only adding this hash-pinned approval is a compatible card-only change."""
    if (CONFIG_KEY in prior or CONFIG_KEY not in current
            or {k: v for k, v in current.items() if k != CONFIG_KEY} != prior):
        raise ValueError('Early-QCD recovery changed unrelated configuration')
    load_early_qcd_fit_policy(current[CONFIG_KEY])
    return True


def _finite_array(values, bins, identity):
    value = np.asarray(values)
    if (value.shape != (bins,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value < 0)):
        raise ValueError('Invalid approved early-QCD array: ' + identity)
    return value


def _guard_array(values, sparse, bins, identity):
    value = _finite_array(values, bins, identity)
    expected = np.zeros(bins)
    for index, number in sparse:
        expected[index] = number
    # Exact zero support is never relaxed. Relative tolerance only covers
    # floating-point TH1 sqrt(variance)**2 round trips of positive cells.
    if (not np.array_equal(value != 0, expected != 0)
            or not np.allclose(value, expected, rtol=1e-12, atol=0.0)):
        raise ValueError('Approved early-QCD support changed: ' + identity)
    return value


def _sparse_audit(values):
    return dict(bins=len(values), integral=float(np.sum(values)),
        nonzero=[[int(i), float(values[i])] for i in np.flatnonzero(values)])


def expected_issues():
    base = dict(channel=CHANNEL, process='QCD_Nb1_u5')
    rows = [dict(base, parameter='CMS_NPS26012_qcd_norm_highdm_Nb1_bin5_2022',
                 problem='zero_integral_nonempty_component', sumw2=0.0,
                 variations=[JER, JES])]
    rows.extend(dict(base, problem='zero_integral_variation', nuisance=n, direction='up')
                for n in (JER, JES))
    rows.extend(dict(channel=CHANNEL, process='QCD_Nb2plus_u5',
                     problem='zero_integral_variation', nuisance=n, direction='down')
                for n in (JER, JES))
    return rows


def early_qcd_fit_background(background, issues, policy_reference=None):
    """Return a fit-only view, remaining issues and a lossless sparse audit.

    No supplied record/array/list is mutated. Non-target records, including
    both positive CR attachments and their rate parameters, retain identity.
    """
    if policy_reference is None:
        return background, issues, []
    policy = load_early_qcd_fit_policy(policy_reference)
    keys = {(row['channel'], row['process']) for row in policy['targets']}
    selected = [row for row in issues if (row.get('channel'), row.get('process')) in keys]
    canonical = lambda rows: sorted(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)
    if canonical(selected) != canonical(expected_issues()):
        raise ValueError('Approved early-QCD five-issue set changed')
    audit = []
    for target in policy['targets']:
        key = target['channel'], target['process']
        # SR process names and support-issue labels retain the native eight-bin
        # identity. The adopted CR12 model maps native bin5 -> pooled bin4;
        # its matching CR process is u4, NOT the same-named CR u5 component.
        control_key = target['control_channel'], target['control_process']
        if key not in background or control_key not in background:
            raise ValueError('Missing exact early-QCD SR/CR attachment')
        row, control = background[key], background[control_key]
        if (row.get('family') != 'QCD' or control.get('family') != 'QCD'
                or row.get('parameter') != target['parameter']
                or control.get('parameter') != target['parameter']
                or not isinstance(row.get('initial'), (int, float))
                or not math.isfinite(row['initial']) or row['initial'] <= 0
                or row['initial'] != control.get('initial')):
            raise ValueError('Approved early-QCD CR/rate-parameter binding changed')
        nominal = _guard_array(row['nominal'], target['nominal'], 118, key[1]+'/nominal')
        variance = _guard_array(row['sumw2'], target['sumw2'], 118, key[1]+'/sumw2')
        cr_nominal = _guard_array(control['nominal'], target['control_nominal'], 12, key[1]+'/CR nominal')
        cr_variance = _guard_array(control['sumw2'], target['control_sumw2'], 12, key[1]+'/CR sumw2')
        variations = row.get('variations')
        if not isinstance(variations, dict) or not set(target['variations']).issubset(variations):
            raise ValueError('Missing approved early-QCD JME endpoint pairs')
        omitted = {}
        for name, pair in variations.items():
            if (not isinstance(name, str) or not name or not isinstance(pair, dict)
                    or set(pair) != {'up', 'down'}):
                raise ValueError('Malformed early-QCD endpoint pair')
            endpoints = {}
            for direction in ('up', 'down'):
                identity = key[1]+'/'+name+'/'+direction
                values = (_guard_array(pair[direction], target['variations'][name][direction], 118, identity)
                          if name in target['variations'] else _finite_array(pair[direction], 118, identity))
                if key[1] == 'QCD_Nb1_u5' and name not in target['variations'] and np.any(values != 0):
                    raise ValueError('New non-JME support in approved zero-nominal early-QCD component')
                endpoints[direction] = _sparse_audit(values)
            omitted[name] = endpoints
        audit.append(dict(channel=key[0], process=key[1], parameter=target['parameter'],
            native_parameter=target['native_parameter'],
            initial=row['initial'], nominal=float(np.sum(nominal)), sumw2=float(np.sum(variance)),
            omitted_nominal=_sparse_audit(nominal), omitted_sumw2=_sparse_audit(variance),
            omitted_variations=omitted, omitted_variation_names=sorted(omitted),
            omitted_variation_count=len(omitted), omitted_endpoint_count=2*len(omitted),
            control_channel=control_key[0], control_process=control_key[1],
            cr_nominal_preserved=float(np.sum(cr_nominal)),
            cr_sumw2_preserved=float(np.sum(cr_variance)), control_record_preserved=True,
            rate_parameter_preserved=True, other_records_unchanged=True,
            removed_issues=copy.deepcopy([i for i in selected if (i['channel'], i['process']) == key]),
            decision='user_approved_fit_only_exclusion', preliminary=True,
            root_objects_preserved=True, policy=copy.deepcopy(policy_reference),
            policy_id=POLICY_ID, auto_mc_stats=[10, 1, 1]))
    return ({key: row for key, row in background.items() if key not in keys},
            [row for row in issues if (row.get('channel'), row.get('process')) not in keys], audit)
