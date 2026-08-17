#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path('/eos/user/t/taiwoo/run3_stop/decaf')
PY38 = Path('/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python')
SYS_PY = Path('/usr/bin/python3')
CMSSW_COMBINE = Path('/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4')
CAMPAIGN = REPO / 'autonomous_allhad/workflow/condor_flat_ntuple_20260630_nominal'
OUTPUT = REPO / 'autonomous_allhad/workflow/flat_ntuple_20260630_nominal_outputs'
ARGS_ALL = CAMPAIGN / 'flat_ntuple_args_all.txt'
LOG_PATH = REPO / 'autonomous_allhad/workflow/flat_ntuple_pipeline_watch_20260630.log'
STATE_PATH = REPO / 'autonomous_allhad/workflow/flat_ntuple_pipeline_watch_20260630_state.json'


def now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a') as handle:
        handle.write(f'[{now()}] {message}\n')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def run(cmd: list[str], step: str, env: dict[str, str] | None = None) -> int:
    append_log(f'RUN {step}: ' + ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    append_log(f'END {step}: returncode={proc.returncode}')
    if proc.stdout:
        for line in proc.stdout.splitlines()[-200:]:
            append_log(f'{step}: {line}')
    return int(proc.returncode)


def expected_args() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in ARGS_ALL.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = line.split()[0]
        out[name] = line
    return out


def output_names() -> tuple[set[str], set[str]]:
    roots = {p.stem for p in OUTPUT.glob('*.root')}
    sidecars = {p.stem for p in OUTPUT.glob('*.json') if not p.name.startswith(('validation', 'merged', 'boosted'))}
    return roots, sidecars


def condor_summary() -> str:
    proc = subprocess.run(['condor_q', '-name', 'bigbird24'], cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout[-4000:]


def active_bigbird_jobs(summary: str) -> bool:
    for line in summary.splitlines():
        if line.startswith('taiwoo ID:'):
            parts = line.split()
            try:
                run = parts[-3]
                idle = parts[-2]
            except Exception:
                continue
            if run != '_' or idle != '_':
                return True
    return False


def write_repair_submit(names: list[str], args_map: dict[str, str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    args_path = CAMPAIGN / f'flat_ntuple_args_repair_{stamp}.txt'
    sub_path = CAMPAIGN / f'flat_ntuple_repair_{stamp}.sub'
    args_path.write_text('\n'.join(args_map[name] for name in names if name in args_map) + '\n')
    sub_path.write_text(f'''universe = vanilla
executable = {REPO}/autonomous_allhad/workflow/condor_flat_ntuple_20260630_nominal/run_flat_ntuple_worker.sh
arguments = $(name) $(shard) $(root_out) $(meta_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {REPO}/condor/py38.tgz, /eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757
transfer_output_files = ""
output = {REPO}/autonomous_allhad/workflow/condor_flat_ntuple_20260630_nominal/logs/$(name).repair.out
error = {REPO}/autonomous_allhad/workflow/condor_flat_ntuple_20260630_nominal/logs/$(name).repair.err
log = {REPO}/autonomous_allhad/workflow/condor_flat_ntuple_20260630_nominal/logs/flat_ntuple_repair_{stamp}.log
request_cpus = 4
request_memory = 4500MB
request_disk = 8000MB
+JobFlavour = "workday"
queue name,shard,root_out,meta_out from {args_path}
''')
    return sub_path


def full_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get('validated_all'):
        rc = run([str(PY38), 'autonomous_allhad/workflow/validate_flat_ntuple_outputs.py',
                  '--output-dir', str(OUTPUT),
                  '--args-file', str(ARGS_ALL),
                  '--require-all-expected',
                  '--summary-output', str(OUTPUT / 'validation_all.json')], 'validate_all')
        validation = read_json(OUTPUT / 'validation_all.json', {})
        if rc != 0 or validation.get('status') != 'complete':
            state['validation_all_status'] = validation.get('status', 'failed')
            state['validation_all_rc'] = rc
            bad_names = [r.get('name') for r in validation.get('results', []) if not r.get('ok')]
            missing = validation.get('missing_expected') or []
            state['needs_repair'] = sorted({x for x in bad_names + missing if x})
            return state
        state['validated_all'] = True

    if not state.get('merged_normalization'):
        rc = run([str(PY38), 'autonomous_allhad/workflow/merge_flat_ntuple_metadata.py',
                  '--inputs', str(OUTPUT),
                  '--output', str(OUTPUT / 'merged_normalization.json')], 'merge_normalization')
        if rc != 0:
            state['merge_normalization_rc'] = rc
            return state
        state['merged_normalization'] = True

    if not state.get('built_hists'):
        env = {'PYTHONPATH': str(REPO / 'autonomous_allhad')}
        rc = run([str(PY38), 'autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py',
                  '--repo', str(REPO),
                  '--inputs', str(OUTPUT),
                  '--normalization', str(OUTPUT / 'merged_normalization.json'),
                  '--output', str(OUTPUT / 'boosted_recoil_hists.json'),
                  '--require-weight-components',
                  'met_trigger', 'photon_trigger',
                  'veto_electron_5to10', 'loose_muon_5to10'],
                 'build_hists', env=env)
        if rc != 0:
            state['build_hists_rc'] = rc
            return state
        state['built_hists'] = True

    if not state.get('built_combine_inputs'):
        rc = run([str(SYS_PY), 'autonomous_allhad/workflow/build_flat_boosted_an17_combine_inputs.py',
                  '--hists', str(OUTPUT / 'boosted_recoil_hists.json'),
                  '--output-dir', 'analysis/combine/flat_boosted_an17_20260630',
                  '--data-mode', 'asimov'], 'build_combine_inputs')
        if rc != 0:
            state['build_combine_inputs_rc'] = rc
            return state
        state['built_combine_inputs'] = True

    if not state.get('ran_expected_limits'):
        runner = REPO / 'analysis/combine/flat_boosted_an17_20260630/run_combine_expected.sh'
        shell_cmd = f'cd {CMSSW_COMBINE}/src && eval $(scramv1 runtime -sh) && cd {REPO} && /usr/bin/bash {runner}'
        rc = run(['/usr/bin/bash', '-lc', shell_cmd], 'run_expected_limits')
        if rc != 0:
            state['run_expected_limits_rc'] = rc
            return state
        state['ran_expected_limits'] = True

    state['pipeline_status'] = 'complete_through_expected_limits'
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--submit-repair', action='store_true')
    args = parser.parse_args()
    args_map = expected_args()
    expected = set(args_map)
    state = read_json(STATE_PATH, {})
    append_log(f'watcher_start expected={len(expected)} interval={args.interval}')

    while True:
        roots, sidecars = output_names()
        missing = sorted(expected - (roots & sidecars))
        summary = condor_summary()
        state.update({
            'updated_at': now(),
            'expected': len(expected),
            'roots': len(roots),
            'sidecars': len(sidecars),
            'complete_pairs': len(roots & sidecars & expected),
            'missing_count': len(missing),
            'missing_preview': missing[:20],
            'condor_summary_tail': summary,
        })
        append_log(f"snapshot pairs={state['complete_pairs']}/{len(expected)} roots={len(roots)} sidecars={len(sidecars)} missing={len(missing)}")

        if not missing:
            state = full_pipeline(state)
            write_json(STATE_PATH, state)
            if state.get('pipeline_status') == 'complete_through_combine_inputs':
                append_log('pipeline complete through combine inputs')
                return 0
        elif args.submit_repair and not active_bigbird_jobs(summary):
            repair = missing[:500]
            if repair:
                sub_path = write_repair_submit(repair, args_map)
                rc = run(['condor_submit', '-name', 'bigbird24', str(sub_path)], 'submit_repair')
                state['last_repair_submit'] = {'path': str(sub_path), 'count': len(repair), 'returncode': rc, 'created_at': now()}
                write_json(STATE_PATH, state)
        else:
            write_json(STATE_PATH, state)

        if args.once:
            return 0
        time.sleep(max(args.interval, 30))


if __name__ == '__main__':
    raise SystemExit(main())
