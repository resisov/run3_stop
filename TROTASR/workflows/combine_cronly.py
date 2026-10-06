"""Existing observed-CR background-only fit, with no observed SR and no new model.

Fits and original pull rendering are separate commands. --plot-only authors a
PDF; perform the PDF authoring/visual-QA procedure before/after that command.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import (native_runtime, grid_card,
    cr_channels, freeze_contract, CommandRunner, require_success, workspace)

SOURCES = ['workflows/combine_cronly.py', 'workflows/diagnostic_common.py',
           'workflows/diagnostic_payload.py', 'workflows/extract_cronly_covariance.py',
           'workflows/renderers/cronly_pulls.py', 'workflows/combine_limits.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py']


def fit_command(workspace_path):
    return ['combine', '-M', 'FitDiagnostics', str(workspace_path), '-m', '120', '-n', '_cronly',
            '--skipSBFit', '--robustFit', '1', '--rMin', '0', '--rMax', '10',
            '--setParameters', 'r=0', '--freezeParameters', 'r',
            '--cminDefaultMinimizerStrategy', '0', '--cminDefaultMinimizerTolerance', '0.1',
            '--cminFallbackAlgo', 'Minuit2,Migrad,1:0.1']


def validate_fit(path):
    import ROOT as root
    source = root.TFile.Open(str(internal_path(path)))
    try:
        fit = source.Get('fit_b') if source and not source.IsZombie() else None
        if not fit:
            raise ValueError('fit_b is missing')
        result = dict(fit_status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()))
        if result['fit_status'] != 0 or result['covariance_quality'] != 3 or not math.isfinite(result['edm']):
            raise ValueError('CR fit did not pass status 0 / covariance quality 3 / finite EDM: ' + str(result))
        return result
    finally:
        if source:
            source.Close()


def plot(runner, output, state, ylim):
    pulls = output / 'pulls.json'
    if state.get('status') not in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
        raise ValueError('Pulls require a successful verified CR fit')
    for name, digest in state['products'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('CR fit/extraction product changed')
    stem = output / ('cronly_nuisance_pulls_ylim' + format(ylim, 'g').replace('.', ''))
    # Same Python/scientific software and same plotting body as the original.
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(output / 'cache'),
               TMPDIR=str(output / 'tmp'))
    command = ['/hep2-scratch/twkim/envs/py38/bin/python', ROOT / 'workflows/diagnostic_payload.py',
               '--operation', 'plot_cronly', '--input', pulls, '--output-stem', stem, '--ylim', str(ylim)]
    figures = [stem.with_suffix('.png'), stem.with_suffix('.pdf')]
    require_success(runner.run(command, output, output / 'plot.log', figures, env=env))
    state.update(status='generated_pending_visual_QA', figures={str(p.relative_to(ROOT)): sha256(p) for p in figures})
    return state


def run(manifest, output, plot_only=False, ylim=6.0):
    if not math.isfinite(ylim) or ylim <= 0:
        raise ValueError('Invalid pull range')
    output = internal_path(output)
    card, origins = grid_card(manifest, 'cronly')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_contract(output, origins, SOURCES, dict(signal_strength=0, workers=1, strategy=0, pull_ylim=ylim))
        runner = CommandRunner(output)
        state_path = output / 'state.json'
        state = read_json(state_path) if state_path.exists() else dict(
            status='running', started=time.time(), origins=origins, sr_observed_data_used=False,
            channels=cr_channels(card.read_text()), workers=1, strategy=0, auto_mc_stats=[10, 1, 1],
            full_workflow_complete=False)
        try:
            state['pid'] = os.getpid()
            if plot_only:
                state = plot(runner, output, state, ylim)
            elif state.get('status') in ('fits_complete_pending_plots', 'generated_pending_visual_QA'):
                for name, digest in state['products'].items():
                    if sha256(ROOT / name) != digest:
                        raise ValueError('Completed CR fit/extraction product changed')
                validate_fit(output / 'fitDiagnostics_cronly.root')
                return 0
            else:
                target = workspace(runner, card, output, cr_only=True)
                fit = output / 'fitDiagnostics_cronly.root'
                state.update(status='running', stage='fit')
                write_json(state_path, state)
                require_success(runner.run(fit_command(target), output, output / 'combine.log', [fit]))
                state.update(validate_fit(fit))
                parameters = output / 'fit_parameters.json'
                require_success(runner.run(['python3', ROOT / 'workflows/diagnostic_payload.py',
                    '--operation', 'extract_cronly', '--fit-diagnostics', fit, '--output', parameters],
                    output, output / 'extract.log', [parameters]))
                payload = read_json(parameters)
                for name, record in payload['parameters'].items():
                    if (not math.isfinite(record['value']) or not math.isfinite(record['error'])
                            or record['error'] < 0 or any(record.get(k) is not None
                            and not math.isfinite(record[k]) for k in ('pull', 'constraint'))):
                        raise ValueError('Nonfinite/invalid extracted fit parameter: ' + name)
                payload.pop('covariance', None)  # full covariance stays in the remote extraction product
                payload.update(sr_observed_data_used=False, source_card=str(card), channels=state['channels'])
                pulls = output / 'pulls.json'
                if pulls.exists() and read_json(pulls) != payload:
                    raise ValueError('Existing pull payload is divergent')
                if not pulls.exists():
                    write_json(pulls, payload)
                state.update(status='fits_complete_pending_plots', stage='fit_complete', finished=time.time(),
                    plotted_nuisances=sum(p['constraint_model'] == 'standard_normal' for p in payload['parameters'].values()),
                    products={str(p.relative_to(ROOT)): sha256(p) for p in (target, fit, parameters, pulls)})
            grid_card(manifest, 'cronly')  # unchanged card and TH1 files after execution
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error), finished=time.time())
            raise
        finally:
            state['pid'] = None
            write_json(state_path, state)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--pull-ylim', type=float, default=6.0)
    p.add_argument('--native-runtime', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    native_runtime(a.output)
    raise SystemExit(run(a.manifest, a.output, a.plot_only, a.pull_ylim))
