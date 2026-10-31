"""Legacy limit-result readers and numerical compatibility exports.

New execution is owned by combine_limits. This module never launches a fit;
its retained readers exist only for already saved historical receipts."""
if __name__ == '__main__':
    raise SystemExit("This manual recovery entrypoint is retired. Use python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> for primary fits, finite recovery and collection.")

from pathlib import Path
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import sha256
from TROTASR.workflows.limit_serialization import validate_saved_limit as validate
from TROTASR.workflows.limit_recovery import (
    MAX_RMAX, QUANTILE_ORDER, adaptive_seed, choose_range, command_for,
    fit_command, target_nll, _verify_bisection_attempt as verify_attempt)


def assist_identity(pid):
    try:
        path = Path('/proc')/str(pid)
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    argv=(path/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except FileNotFoundError:
        return None


def same_primary_failure(current, original):
    # The unchanged primary may encounter an assisted terminal point later.
    # Its no-new-attempt path updates only `finished`; retain both receipts.
    return ({k:v for k,v in current.items() if k != 'finished'} ==
            {k:v for k,v in original.items() if k != 'finished'})


def validate_complete(record):
    product=internal_path(ROOT/record['output'])
    if sha256(product)!=record['output_sha256']:raise ValueError('Completed ROOT changed')
    matches=[a for a in record['attempts'] if a.get('log') and
        (ROOT/a['log']).parent/('higgsCombine_'+record['mass']+'.AsymptoticLimits.mH120.root')==product]
    if len(matches)!=1 or matches[0].get('exit_code')!=0:
        raise ValueError('Saved output not bound to exactly one successful command')
    accepted=matches[0]
    passed,evidence=validate(product,ROOT/accepted['log'],accepted['tolerance'])
    if not passed:raise ValueError('Saved complete point fails numerical validation')


def _retired_execution(*args, **kwargs):
    raise RuntimeError("This manual recovery entrypoint is retired. Use python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> for primary fits, finite recovery and collection.")


run = run_point = run_assist = assist_retry = adaptive_retry = run_shared_pool = run_local_t2tt_pool = _retired_execution
