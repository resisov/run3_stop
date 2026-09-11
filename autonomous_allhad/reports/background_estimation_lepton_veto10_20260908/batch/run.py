#!/usr/bin/env python3
"""Campaign orchestration only; all factor calculations use existing commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path('/eos/user/t/taiwoo/run3_stop/decaf')
CAMPAIGN = REPO / 'autonomous_allhad/reports/background_estimation_lepton_veto10_20260908'
PRODUCTS = ('dy_measurement.json', 'sgamma/sgamma_ut.json', 'sgamma/sgamma_ut.csv',
            'zgamma/zgamma_double_ratio.json', 'tf_inputs.json')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def staged_repository(scratch, repo):
    """Keep final relative provenance paths while writing only worker scratch."""
    root = Path(tempfile.mkdtemp(prefix='background_estimation_', dir=str(scratch)))
    (root / 'autonomous_allhad').mkdir()
    (root / 'autonomous_allhad/workflow').symlink_to(repo / 'autonomous_allhad/workflow')
    return root


def validate_products(output, hist_sha):
    for name in PRODUCTS:
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('missing calculated product: ' + name)
        if path.suffix == '.json':
            product = json.loads(path.read_text())
            if product.get('status') != 'complete' or product['provenance']['hist_input_sha256'] != hist_sha:
                raise ValueError('incomplete or stale calculated product: ' + name)
    return {name: sha(output / name) for name in PRODUCTS}


def promote_products(staged, output, products):
    """Verify every staged copy before replacing any canonical product."""
    pending = []
    try:
        for name, expected in products.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name('.' + destination.name + '.pending')
            pending.append((temporary, destination))
            shutil.copyfile(staged / name, temporary)
            if sha(temporary) != expected:
                raise ValueError('product copy checksum mismatch: ' + name)
        for temporary, destination in pending:
            temporary.replace(destination)
    finally:
        for temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('year', choices=('2024', '2025'))
    parser.add_argument('--replace-stale', action='store_true',
                        help='Recalculate changed, promoted inputs in worker scratch before replacing old products.')
    args = parser.parse_args()
    year = args.year
    frozen = CAMPAIGN / 'batch/code'
    code = json.loads((CAMPAIGN / 'batch/code_manifest.json').read_text())
    for relative, expected in code['files'].items():
        if sha(frozen / relative) != expected:
            raise ValueError('frozen source checksum mismatch: ' + relative)
    manifest_path = CAMPAIGN / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'canonical' or manifest['electron_veto_pt_min_gev'] != 10 or manifest['muon_veto_pt_min_gev'] != 10 or manifest['lepton_threshold_operator'] != '>':
        raise ValueError('not the approved strict 10-GeV canonical inputs')
    info = manifest['years'][year]
    update = manifest.get('topw_sf_update', {})
    if update and int(year) not in update.get('promoted_years', []):
        raise ValueError('this year has not been promoted for the Top/W update')
    canonical = json.loads((REPO / 'autonomous_allhad/reports/lepton_veto10_canonical_20260908.json').read_text())
    if canonical['years'][year] != info:
        raise ValueError('campaign input manifest differs from the canonical year')
    inputs = {}
    for name in ('main', 'background_estimation', 'gnn'):
        path = REPO / info[name]['path']
        if sha(path) != info[name]['sha256']:
            raise ValueError('input checksum mismatch: ' + str(path))
        if (REPO / info[name]['canonical_alias']).resolve() != path.resolve():
            raise ValueError('canonical alias mismatch')
        inputs[name] = info[name]['sha256']
    output = CAMPAIGN / year
    state_path = output / 'calculation_state.json'
    previous = None
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('status') == 'complete' and previous.get('inputs') == inputs and previous.get('code_manifest_sha256') == sha(CAMPAIGN / 'batch/code_manifest.json') and all(sha(output / p) == h for p, h in previous['products'].items()):
            print(json.dumps({'year': year, 'status': 'already_complete_valid'}))
            return 0
        if not args.replace_stale:
            raise ValueError('existing calculation state is not reusable; use --replace-stale for the promoted inputs')
        if previous.get('status') == 'running':
            raise ValueError('calculation is already running; diagnose before retrying')
    scratch = Path(os.environ['_CONDOR_SCRATCH_DIR'])
    work_repo = staged_repository(scratch, REPO)
    output.mkdir(parents=True, exist_ok=True)
    state = {'year': year, 'status': 'running', 'inputs': inputs,
             'input_manifest_sha256': sha(manifest_path),
             'code_manifest_sha256': sha(CAMPAIGN / 'batch/code_manifest.json'),
             'intermediate_root_reread': False, 'started_unix': time.time(), 'steps': [],
             'staged_in_worker_scratch': True,
             'replaces_input_hashes': previous.get('inputs') if previous else None}
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(frozen / 'autonomous_allhad'))
    workflow = frozen / 'autonomous_allhad/workflow'
    result = output.relative_to(REPO)
    hist = info['background_estimation']['path']
    commands = [
        ['-m', 'autonomous_allhad.dy_estimation', 'build-measurement', '--hist-input', hist, '--campaign-year', year, '--output', str(result / 'dy_measurement.json')],
        [str(workflow / 'build_sgamma_ut_report_2024.py'), '--hist-input', hist, '--campaign-year', year, '--output-dir', str(result / 'sgamma'), '--no-plots'],
        [str(workflow / 'build_zgamma_double_ratio_2024.py'), '--hist-input', hist, '--campaign-year', year, '--low-ut-min', '250', '--output-dir', str(result / 'zgamma'), '--no-plots'],
        [str(workflow / 'build_histogram_tf_inputs_2024.py'), '--hist-input', hist, '--campaign-year', year,
         '--gnn-input', info['gnn']['path'], '--gnn-config', str(frozen / manifest['lowdm']['configuration']),
         '--sgamma-input', str(result / 'sgamma/sgamma_ut.json'), '--dy-measurement', str(result / 'dy_measurement.json'), '--output', str(result / 'tf_inputs.json')],
    ]
    try:
        for command in commands:
            subprocess.run([sys.executable] + command, cwd=str(work_repo), env=env, check=True)
            state['steps'].append({'command': command, 'exit_code': 0})
            state_path.write_text(json.dumps(state, indent=2) + '\n')
        staged = work_repo / result
        products = validate_products(staged, inputs['background_estimation'])
        canonical = json.loads((REPO / 'autonomous_allhad/reports/lepton_veto10_canonical_20260908.json').read_text())
        if canonical['years'][year] != info:
            raise ValueError('canonical year changed during calculation')
        promote_products(staged, output, products)
        state.update(status='complete', products=products, finished_unix=time.time())
    except Exception as exc:
        state.update(status='failed', exception=type(exc).__name__, error=str(exc), finished_unix=time.time())
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        shutil.rmtree(work_repo)
    print(json.dumps({'year': year, 'status': state['status'], 'products': state.get('products')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
