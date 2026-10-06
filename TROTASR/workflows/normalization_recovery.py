"""Strict compatibility for a disjoint extension of signal normalization.

This is not a generic contract bypass. Every old factor and all non-normalization
physics hashes must be identical; original outputs remain byte-for-byte intact.
"""
import copy
import os
from pathlib import Path
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, sha256


def process_identity(pid):
    """Kernel start time prevents adopting an unrelated reused PID."""
    if not isinstance(pid, int) or pid < 2:
        return None
    try:
        folder = Path('/proc')/str(pid)
        fields = (folder/'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    arguments=[x.decode() for x in (folder/'cmdline').read_bytes().split(b'\0') if x])
    except FileNotFoundError:
        return None


def retained_worker(campaign, key, state):
    """Keep an already-running nominal attempt; never resubmit its ROOT."""
    if state.get('status') != 'running' or state.get('attempt') != 1:
        raise ValueError('Only the original running attempt may be retained')
    identity = process_identity(state.get('pid'))
    history = campaign/'history'/(key+'.attempt1.json')
    if identity:
        args = identity['arguments']
        expected = [str(ROOT/'workflows/nominal_campaign.py'), 'worker', '--campaign',
                    str(campaign), '--key', key, '--attempt', '1']
        if args[-len(expected):] != expected:
            raise ValueError('Retained PID is not the exact expected worker: '+key)
    elif not history.exists() or read_json(history).get('status') != 'complete':
        raise ValueError('Exited retained worker has no successful receipt: '+key)
    return dict(pid=state['pid'], attempt=1, identity=identity,
                history=str(history.relative_to(ROOT)))


class RetainedProcess:
    """Read-only polling of an orphaned worker; no signal and no new process."""
    def __init__(self, record):
        self.record = record
        self.pid = record['pid']

    def poll(self):
        actual = process_identity(self.pid)
        if actual is not None:
            if actual != self.record['identity']:
                raise RuntimeError('Retained worker PID reused or command changed')
            return None
        path = internal_path(ROOT/self.record['history'])
        return 0 if path.exists() and read_json(path).get('status') == 'complete' else 1


def adopt_worker(plan, campaign, key, state):
    record = plan.get('normalization_extension_recovery', {}).get('retained_workers', {}).get(key)
    if record is None:
        return None
    if state.get('pid') != record['pid'] or state.get('attempt') != record['attempt']:
        raise ValueError('Retained worker state changed: '+key)
    process = RetainedProcess(record)
    process.poll()  # Reject PID reuse before admitting any new work.
    return process, open(os.devnull, 'w'), record['attempt']


def verify_extension(old, new):
    for key in ('luminosity_pb', 'luminosity_fb', 'normalization_policy',
                'background_xsec_overrides', 'signal_xsec_source'):
        if old.get(key) != new.get(key):
            raise ValueError('Normalization prescription changed: '+key)
    if new.get('status') != 'complete' or new.get('blocked_signal_mass_points'):
        raise ValueError('Incomplete normalization extension')
    for section in ('signal_mass_points', 'dataset_factors', 'datasets'):
        for key, value in old[section].items():
            if new[section].get(key) != value:
                raise ValueError('Existing normalization record changed: '+section+'/'+key)
    if any(not rec.get('is_signal') for key,rec in new['datasets'].items() if key not in old['datasets']):
        raise ValueError('Extension added non-signal dataset')
    added = set(new['signal_mass_points'])-set(old['signal_mass_points'])
    if not added:
        raise ValueError('No missing signal normalization to recover')
    return sorted(added)


def retry_keys(plan, states):
    """Exactly one corrected attempt; initial failed records/logs are retained."""
    spec=plan.get('normalization_extension_recovery')
    if not spec:return []
    result=[]
    for key in spec['retry_keys']:
        state=states.get(key,{})
        if state.get('status')=='complete':continue
        if state.get('status')=='failed' and state.get('attempt',0)>=2:continue
        if state.get('status')!='failed' or state.get('attempt')!=1:
            raise ValueError('Unexpected normalization recovery state: '+key)
        if (state.get('exception_type')!='RuntimeError' or not state.get('error','').startswith(
                'missing, non-finite, or non-positive signal normalization factor for ')):
            raise ValueError('Different failure is not normalization recovery: '+key)
        result.append(key)
    return result


def prior_contracts(plan):
    spec=plan.get('normalization_extension_recovery')
    if not spec:return {}
    audit_path=internal_path(ROOT/spec['audit'])
    if sha256(audit_path)!=spec['audit_sha256']:
        raise ValueError('Changed normalization recovery audit')
    audit=read_json(audit_path)
    if audit['status']!='candidate_prepared_not_promoted':
        raise ValueError('Unexpected normalization audit')
    for name,digest in spec['code_sha256'].items():
        if sha256(internal_path(ROOT/name))!=digest:
            raise ValueError('Changed recovery code: '+name)
    previous=spec['prior_contracts']
    original_assets=internal_path(ROOT/spec['original_assets'])
    current_assets=ROOT/'jsons/assets.json'
    if any(sha256(original_assets)!=v['assets_sha256'] for v in previous.values()):
        raise ValueError('Original asset manifest hash changed')
    if any(sha256(current_assets)!=v['assets_sha256'] for v in plan['contracts'].values()):
        raise ValueError('Current asset manifest hash changed')
    omit={'estimations/normalization_2024.json.gz','estimations/normalization_2025.json.gz'}
    remaining=lambda p:{r['target']:r for r in read_json(p)['files'] if r['target'] not in omit}
    if remaining(original_assets)!=remaining(current_assets):
        raise ValueError('Non-normalization assets changed')
    for year,current in plan['contracts'].items():
        before=previous[year]
        for key in set(current)|set(before):
            if key not in ('normalization_sha256','assets_sha256') and current.get(key)!=before.get(key):
                raise ValueError('Non-normalization physics contract changed')
        old_path=internal_path(ROOT/spec['original_normalizations'][year])
        new_path=ROOT/'estimations'/('normalization_%s.json.gz'%year)
        if sha256(old_path)!=before['normalization_sha256'] or sha256(new_path)!=current['normalization_sha256']:
            raise ValueError('Normalization hash mismatch')
        expected=audit['years'][year]
        if (expected['changed_existing_points'] or expected['blocked_points'] or
                expected['candidate_sha256']!=current['normalization_sha256']):
            raise ValueError('Not a disjoint validated normalization extension')
        if verify_extension(read_json(old_path),read_json(new_path))!=expected['added_mass_points']:
            raise ValueError('Mass-point extension differs from audited candidate')
    return copy.deepcopy(previous)


def payload_contract(payload, current, previous):
    if isinstance(previous, list):
        for candidate in previous:
            if all(payload['contract'].get(k)==v for k,v in candidate.items()):
                return candidate
        return current
    if previous and all(payload['contract'].get(k)==v for k,v in previous.items()):
        return previous
    return current
