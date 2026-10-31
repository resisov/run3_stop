"""Legacy numerical-result verifiers, re-exported from limit_recovery.

The former manual recovery controllers are retired. combine_limits owns the
only new fit lifecycle; these exports preserve saved-result reader imports.
"""
if __name__ == '__main__':
    raise SystemExit(
        'This manual recovery entrypoint is retired. Use '
        'python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> '
        'for primary fits, finite recovery and collection.')

from TROTASR.workflows.limit_recovery import (
    measured_start_commands, minimum_commands, refinement_command,
    native_profiled_text, verify_native_attempt, winner_range_commands,
    winner_native_commands, winner_stepping_commands, stepping_median_evidence,
    verify_stepping_attempt, crossing_commands as run3_crossing_commands,
    precision_commands as run3_precision_commands,
    five_quantile_structure as run3_five_quantile_structure,
    native_stepping_proposals as run3_native_stepping_proposals,
    validate_attempt as verify_run3_numerical_attempt)


def _retired_execution(*args, **kwargs):
    raise RuntimeError(
        'This manual recovery entrypoint is retired. Use '
        'python -m TROTASR limits --manifest <grid/manifest.json> --output <limits> '
        'for primary fits, finite recovery and collection.')


run = trial = recover_failed_point = minimum_point = recover_comparison = _retired_execution
recover_minima = refine_candidate_point = step_winner_point = refine_point = _retired_execution
continue_run3_point = run3_extended_pool = _retired_execution
