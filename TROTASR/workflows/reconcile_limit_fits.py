"""Read-only compatibility for saved limit recovery results.

No controller, runtime, worker or campaign dispatcher is retained here.
New primary fits, finite recovery and collection belong to combine_limits."""
if __name__ == '__main__':
    raise SystemExit("This manual recovery entrypoint is retired. Use python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> for primary fits, finite recovery and collection.")

import hashlib
import json
import re
from pathlib import Path
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, sha256
from TROTASR.workflows.limit_serialization import validate_saved_limit
from TROTASR.workflows.limit_recovery import (
    choose_range, command_for, MAX_RMAX, profile_recovery_configs,
    profile_recovery_command, unfinished_configs)
from TROTASR.workflows.continue_limit_fits import verify_attempt

UNFINISHED_HISTORY = ('numerical_rmax100000', 'numerical_adaptive_nll',
                      'numerical_profile_recovery', 'numerical_profile_calls2m')


def command_key(command, cwd):
    """Historical receipt key only; no execution/admission dependency."""
    return hashlib.sha256(json.dumps([list(map(str, command)), str(cwd)],
        separators=(',', ':')).encode()).hexdigest()[:24]


def attempt_output(point, attempt):
    return ROOT/attempt['log'], (ROOT/attempt['log']).parent/(
        'higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')


def verified_complete(point,record):
    product=ROOT/record['output']
    if sha256(product)!=record['output_sha256']:raise ValueError('Accepted output changed')
    attempts=[a for a in record['attempts'] if attempt_output(point,a)[1]==product]
    if len(attempts)!=1:raise ValueError('Accepted output not linked to exactly one attempt')
    attempt=attempts[0]
    if attempt.get('exit_code')!=0:raise ValueError('Accepted command failed')
    passed,_=validate_saved_limit(product,ROOT/attempt['log'],attempt['tolerance'])
    if not passed:raise ValueError('Accepted output no longer validates')


def unfinished_history(point, output, include_latest=False):
    key = point['model']+'/'+point['mass']
    records = [('primary', output/key/'state.json')]
    sources = UNFINISHED_HISTORY + (('numerical_all_unfinished_20260930',) if include_latest else ())
    records += [(name, output/name/key/'state.json') for name in sources
                if (output/name/key/'state.json').exists()]
    history, hashes = [], {}
    for origin, path in records:
        record = read_json(path)
        if record.get('status') not in ('complete','failed','verified_not_promoted'):
            raise ValueError('Nonterminal predecessor retained: '+str(path))
        if record.get('pid'):
            raise ValueError('Registered predecessor process retained: '+str(path))
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        history.extend((origin, attempt) for attempt in record.get('attempts', []))
    return history, hashes


def unfinished_saved(point, output, history):
    key = point['model']+'/'+point['mass']
    logs = []
    for origin, attempt in history:
        log = ROOT/attempt['log']
        if sha256(log) != attempt['log_sha256']:
            raise ValueError('Historical log changed')
        logs.append((attempt['log'], log.read_text(errors='replace')))
        if attempt.get('exit_code') != 0:
            continue
        if read_json(log.parent/'validation.json') != attempt:
            raise ValueError('Historical validation receipt differs')
        if origin != 'primary':
            directory = output/origin
            if origin in ('numerical_profile_recovery','numerical_profile_calls2m','numerical_all_unfinished_20260930'):
                directory = directory/key
            execution = read_json(directory/'command_receipts/commands'/
                (command_key(attempt['command'], log.parent)+'.json'))
            if (execution['status'] != 'complete' or execution['exit_code'] != 0
                    or execution.get('pid') or execution['command'] != attempt['command']
                    or execution['products'].get(attempt['output']) != attempt['output_sha256']):
                raise ValueError('Historical execution receipt differs')
        passed, evidence = verify_attempt(point, attempt)
        if passed:
            return dict(origin=origin, attempt=attempt, validation=evidence), logs
    return None, logs


def historical_source(output, name, digest):
    current=ROOT/name
    if sha256(current)==digest:return current
    archived=output.parent/'history/recovery_20261009/source'/Path(name).name
    if archived.is_file() and sha256(archived)==digest:return archived
    source=Path(name)
    pattern='*/source/'+source.stem+'_'+digest[:8]+source.suffix
    for candidate in sorted((output.parent/'history').glob(pattern)):
        if candidate.is_file() and sha256(candidate)==digest:return candidate
    raise ValueError('Frozen historical source unavailable: '+name)


def in_plot_range(point):
    match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', point['mass'])
    if not match:
        raise ValueError('Invalid signal mass key: '+point['mass'])
    stop, lsp = map(int, match.groups())
    return 600 <= stop <= 1800 and 1 <= lsp <= 1800


def plot_range_first(points):
    """Stable admission order only; use the existing contour's exact mass window."""
    return sorted(points, key=lambda point: not in_plot_range(point))


def recovery_code_versions(auxiliary, contract):
    """Authenticate explicit admission-only revisions without rewriting old points."""
    versions = [contract['code']]
    for item in contract.get('prior_admission_contracts', []):
        path = internal_path(ROOT/item['path'])
        if auxiliary not in path.parents or sha256(path) != item['sha256']:
            raise ValueError('Admission-history contract changed')
        old = read_json(path)
        for name in ('manifest_sha256', 'primary_contract_sha256'):
            if old[name] != contract[name]:
                raise ValueError('Admission revision changes fit inputs')
        versions.append(old['code'])
    for version in versions:
        for name, want in version.items():
            historical_source(auxiliary.parent, name, want)
    return versions


def _retired_execution(*args, **kwargs):
    raise RuntimeError("This manual recovery entrypoint is retired. Use python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> for primary fits, finite recovery and collection.")


run = run_point = recover_profile = recover_profile_batch = recover_unfinished_point = run_live_unfinished = run_unfinished = join_live_unfinished = join_scoped_remaining = _retired_execution
