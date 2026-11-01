"""Recover failed b-tag input files while retaining successful outputs."""
import argparse
import contextlib
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

FIELD = 'btag_zero_denominator_recovery'
KEY = '2022_signal_v15_batch_00517'
CHANGED = {'utils/pog_corrections.py', 'utils/weight_components.py'}
ERROR = 'Out of range float values are not JSON compliant: inf'
POLICY = 'per_jet_zero_probability_unity_20261022'


def check_contracts(before, after):
    if set(before) != set(after):
        raise ValueError('Recovery changed era inventory')
    for year, old in before.items():
        new = after[year]
        if {k:v for k,v in old.items() if k != 'code'} != {k:v for k,v in new.items() if k != 'code'}:
            raise ValueError('Recovery changed non-code physics contract')
        changed = {k for k in set(old['code']) | set(new['code']) if old['code'].get(k) != new['code'].get(k)}
        if changed != CHANGED:
            raise ValueError('Recovery changed unrelated source: '+str(changed))


def prior_contracts(plan):
    spec = plan.get(FIELD)
    if not spec:
        return {}
    before_path = internal_path(ROOT/spec['previous_plan'])
    if sha256(before_path) != spec['previous_plan_sha256'] or spec['policy'] != POLICY or spec['retry_keys'] != [KEY]:
        raise ValueError('Changed b-tag recovery identity')
    before = read_json(before_path)
    if {k:v for k,v in plan.items() if k not in ('contracts',FIELD)} != {k:v for k,v in before.items() if k != 'contracts'}:
        raise ValueError('B-tag recovery changed scheduling or task inventory')
    check_contracts(before['contracts'], plan['contracts'])
    for rel, digest in spec['sources'].items():
        if sha256(ROOT/rel) != digest:
            raise ValueError('Changed approved b-tag recovery source: '+rel)
    gate = ROOT/spec['unit_tests']
    if sha256(gate) != spec['unit_tests_sha256'] or read_json(gate)['status'] != 'passed':
        raise ValueError('B-tag recovery validation changed')
    numeric = ROOT/spec['full_file_validation']
    if sha256(numeric) != spec['full_file_validation_sha256']:
        raise ValueError('B-tag full-file validation changed')
    return copy.deepcopy(before['contracts'])


def retry_keys(plan, states):
    if FIELD not in plan:
        return []
    s = states.get(KEY, {})
    if s.get('status') == 'complete' or s.get('attempt',0) >= 2:
        return []
    if s.get('status') != 'failed' or s.get('attempt') != 1 or s.get('error') != ERROR:
        raise ValueError('Not the diagnosed first-attempt b-tag failure')
    return [KEY]


def reuse_unchanged_merge(campaign, plan, year, receipt):
    """Reuse an exact pre-recovery era merge, never rewrite its provenance."""
    spec = plan.get(FIELD)
    if not spec or str(year) == '2022':
        return None
    prior_contracts(plan)
    saved = spec.get('preserved_merges', {}).get(str(year))
    if not saved:
        return None
    if sha256(receipt) != saved['receipt_sha256']:
        raise ValueError('Preserved rare-top receipt changed')
    record = read_json(receipt)
    output = ROOT/record['output']
    if (sha256(output) != saved['output_sha256'] or record['output_sha256'] != saved['output_sha256']
            or record['contract']['plan_sha256'] != spec['previous_plan_sha256']
            or sha256(ROOT/record['original_output']) != record['original_sha256']):
        raise ValueError('Preserved era merge changed')
    return output


@contextlib.contextmanager
def historical_btag_projection(campaign, original_contract):
    """Subtract saved mixed-file backgrounds with their original evaluator."""
    plan = read_json(campaign/'plan.json')
    if FIELD not in plan:
        yield original_contract, None
        return
    before = prior_contracts(plan)
    year = str(original_contract['year'])
    old_code = before[year]['code']
    if original_contract['code'] == plan['contracts'][year]['code']:
        yield original_contract, None
        return
    if original_contract['code'] != old_code:
        raise ValueError('Unknown saved mixed-file source contract')
    source = ROOT/plan[FIELD]['checkpoint']/'utils/pog_corrections.py'
    if sha256(source) != old_code['utils/pog_corrections.py']:
        raise ValueError('Historical b-tag evaluator changed')
    name = 'TROTASR.utils._historical_btag_20261022'
    mspec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(mspec); sys.modules[name] = module; mspec.loader.exec_module(module)
    from TROTASR.utils import weight_components as wc
    original = wc.get_btag_corrector
    cache = {}
    def factory(repo, era, caller, tagger=None, workingpoint='medium'):
        tagger = tagger or ('UParTAK4' if wc.analysis_year(era) in {'2024','2025'} else 'particleNet')
        key = (tagger, wc.analysis_year(era), workingpoint, caller)
        if key not in cache:
            cache[key] = module.BTagCorrector(*key)
        return cache[key]
    wc.get_btag_corrector = factory
    expected = dict(original_contract, code=plan['contracts'][year]['code'])
    try:
        yield expected, dict(purpose='exact_subtraction_of_saved_background', historical_source_sha256=sha256(source))
    finally:
        wc.get_btag_corrector = original


def promote(campaign):
    from TROTASR.workflows.normalization_recovery import process_identity
    from TROTASR.workflows.nominal_campaign import fingerprint, current_contract
    campaign = internal_path(campaign)
    folder = campaign/'recovery/btag_20261022'
    stage = folder/'staged'; checkpoint = folder/'checkpoint'
    plan = read_json(campaign/'plan.json')
    if FIELD in plan:
        prior_contracts(plan)
        return
    gate = read_json(folder/'unit_tests.json')
    if gate.get('status') != 'passed' or gate.get('tests',0) < 6:
        raise ValueError('Missing b-tag unit-test gate')
    for rel, digest in gate['sources'].items():
        if sha256(stage/rel) != digest:
            raise ValueError('Staged candidate changed after tests')
    numeric = read_json(folder/'full_file_btag_validation.json')
    if (numeric.get('status') != 'passed' or numeric.get('events') != 215262
            or numeric.get('zero_events') != 1 or numeric.get('zero_jets') != 1
            or not numeric.get('supported_events_bitwise_equal') or not numeric.get('all_nine_endpoints_finite')
            or numeric.get('candidate_sha256') != sha256(stage/'utils/pog_corrections.py')):
        raise ValueError('Missing exact full-file b-tag validation')
    with contextlib.ExitStack() as stack:
        for name in ('chain.lock','controller.lock'):
            lock = stack.enter_context((campaign/name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('state.json','chain_state.json'):
            if process_identity(read_json(campaign/name).get('pid')):
                raise RuntimeError('Original controller is still alive')
        state = read_json(campaign/'state.json')
        if state['status'] != 'complete_with_failures' or state['failed'] != [KEY] or state['active'] or state['pending']:
            raise ValueError('Original production has not drained with exactly the diagnosed failure')
        original = {y:current_contract(y,plan.get('shape_mode'),plan.get('variations')) for y in plan['contracts']}
        if original != plan['contracts']:
            raise ValueError('Active source changed before promotion')
        retained = {}
        for task in plan['tasks']:
            s = read_json(campaign/'state'/(task['key']+'.json'))
            if fingerprint(ROOT/task['input']) != task['fingerprint']:
                raise ValueError('Input changed: '+task['key'])
            if task.get('metadata') and fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
                raise ValueError('Metadata changed: '+task['key'])
            if task['key'] == KEY:
                retry_keys({FIELD:True},{KEY:s})
                continue
            if s['status'] != 'complete' or sha256(ROOT/s['output']) != s['output_sha256']:
                raise ValueError('Completed output changed: '+task['key'])
            retained[task['key']] = dict(output=s['output'],sha256=s['output_sha256'])
        if checkpoint.exists():
            raise FileExistsError('Partial promotion checkpoint; inspect before continuing')
        checkpoint.mkdir()
        deploy = sorted(CHANGED | {'workflows/normalization_recovery.py','workflows/rare_top_policy.py',
                                  'workflows/btag_zero_recovery.py'})
        for path in [campaign/'plan.json', campaign/'state.json', campaign/'chain_state.json',
                     *[ROOT/x for x in deploy if (ROOT/x).exists()],
                     *[ROOT/'utils/compiled'/x for x in ('manifest.json','ids.coffea','corrections.coffea')]]:
            dest = checkpoint/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
        write_json(folder/'retained_outputs.json',retained)
        preserved = {}
        for era in plan['contracts']:
            receipt = campaign/'background_policy'/era/'receipt.json'
            if receipt.exists():
                rec=read_json(receipt)
                if rec['status'] != 'complete' or sha256(ROOT/rec['output']) != rec['output_sha256']:
                    raise ValueError('Existing rare-top result not verified')
                preserved[era]=dict(receipt_sha256=sha256(receipt),output_sha256=rec['output_sha256'])
        for rel in deploy:
            shutil.copy2(stage/rel, ROOT/rel)
        subprocess.run([sys.executable,str(ROOT/'workflows/compile_definitions.py')],cwd=ROOT,check=True)
        new = copy.deepcopy(plan)
        new['contracts'] = {y:current_contract(y,plan.get('shape_mode'),plan.get('variations')) for y in plan['contracts']}
        check_contracts(plan['contracts'],new['contracts'])
        previous = checkpoint/(campaign/'plan.json').relative_to(ROOT)
        new[FIELD] = dict(policy=POLICY,retry_keys=[KEY],previous_plan=str(previous.relative_to(ROOT)),
            previous_plan_sha256=sha256(previous),checkpoint=str(checkpoint.relative_to(ROOT)),
            sources={r:sha256(ROOT/r) for r in deploy},unit_tests=str((folder/'unit_tests.json').relative_to(ROOT)),
            unit_tests_sha256=sha256(folder/'unit_tests.json'),preserved_merges=preserved,
            full_file_validation=str((folder/'full_file_btag_validation.json').relative_to(ROOT)),
            full_file_validation_sha256=sha256(folder/'full_file_btag_validation.json'),
            retained_outputs=str((folder/'retained_outputs.json').relative_to(ROOT)),
            retained_outputs_sha256=sha256(folder/'retained_outputs.json'))
        prior_contracts(new)
        write_json(campaign/'plan.json',new)
        write_json(folder/'promotion.json',dict(status='promoted_pending_single_file_retry',finished=time.time(),
            retry_keys=[KEY],preserved_files=len(retained),recomputed_successful_files=0,
            plan_sha256=sha256(campaign/'plan.json'),compiled_manifest_sha256=sha256(ROOT/'utils/compiled/manifest.json')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True,type=Path)
    args=parser.parse_args();promote(args.campaign)
