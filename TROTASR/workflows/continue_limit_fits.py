"""One bounded numerical continuation of failed limits, without model changes.

Wait for the primary controller to finish, acquire its lock, preserve its
receipts, and apply at most two additional numerical configurations per failed
point. No valid point is rerun and no old-campaign fitted value is imported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import fcntl
import math
from pathlib import Path
import os
import re
import statistics
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.combine_limits import IntegrityCache, verify_card, runtime, fit_command, collect
from TROTASR.workflows.diagnostic_common import CommandRunner
from TROTASR.workflows.limit_validation import SCAN
from TROTASR.workflows.limit_serialization import validate_saved_limit as validate

SOURCES = ('workflows/continue_limit_fits.py', 'workflows/combine_limits.py',
           'workflows/limit_validation.py', 'workflows/limit_serialization.py',
           'workflows/diagnostic_common.py',
           'stats/runtime.json', 'stats/runtime/sitecustomize.py')
QUANTILE_ORDER = (.5, .025, .16, .84, .975)
MAX_RMAX = 10000.
# A campaign-specific, immutable input contract, not a general JME scope waiver.
# These are the already verified primary inputs from 2026-09-28; the adapter
# neither rewrites them nor admits stored-TROTA/non-JME inputs to this fit.
JME_SCOPE = 'full_cms_trota_jme_production'
JME_INPUT_HASHES = {
    'manifest': '44fc908e8a956082779b7a5e31ccd79563d8eee020712352aa5fc4115afeff97',
    'primary_contract': '5399342bfafecd880382395b3e74976e9fc10cc7e50feed07297128d75a2a218',
    'templates': 'de19bab6de3961a7cfa9088477f247a88c64e2ce1b9e3267f0f4ddc786e8f205',
}
JME_SYSTEMATICS = ('JME corrected-central nominal + JES Total Up/Down + JER Up/Down + '
                   'existing weights; preliminary, non-JME16 excluded by user choice')


def verify_grid_inputs(manifest, output, grid, workers, digest):
    """Retain nominal behavior; explicitly bind the approved preliminary JME fit.

    Run before any continuation receipt or command is created. Every card is
    checked using the existing validator, including all four actual ROOT hashes.
    No histogram, card, physics definition or numerical policy is modified.
    """
    if grid.get('status') != 'cards_ready':
        raise ValueError('Invalid grid')
    if grid.get('scope') == 'full_nominal_production':
        return None
    if grid.get('scope') != JME_SCOPE:
        raise ValueError('Invalid grid scope')
    if not 1 <= workers <= 10:
        raise ValueError('Approved JME continuation requires at most ten workers')
    if output != manifest.parent.parent/'limits' or manifest.parent.name != 'fit_grid':
        raise ValueError('JME continuation must use its existing primary limits directory')
    if (digest(manifest) != JME_INPUT_HASHES['manifest'] or
            digest(output/'contract.json') != JME_INPUT_HASHES['primary_contract']):
        raise ValueError('Unapproved or changed JME primary input contract')
    points = grid['points']
    counts = {m: sum(p['model'] == m for p in points) for m in ('T2tt', 'T2bW', 'T2tb')}
    keys = [(p['model'], p['mass']) for p in points]
    if (counts != {'T2tt': 205, 'T2bW': 354, 'T2tb': 365}
            or grid.get('expected_points_by_model') != counts
            or len(points) != 924 or len(set(keys)) != 924
            or any(p.get('fit_ready') is not True for p in points)
            or grid.get('blocked') != [] or grid.get('ready_inputs') != 924
            or grid.get('template_points') != 924
            or grid.get('root_integrity_checked') is not True
            or grid.get('sr_data_blinded') is not True
            or grid.get('systematic_scope') != JME_SYSTEMATICS):
        raise ValueError('Incomplete or changed JME grid/blinding/systematic scope')
    template_path = internal_path(ROOT/grid['templates_manifest'])
    if digest(template_path) != JME_INPUT_HASHES['templates']:
        raise ValueError('Changed JME template manifest')
    template = read_json(template_path)
    names = {f'template_{mode}_{year}.root' for mode in ('highdm', 'lowdm') for year in (2024, 2025)}
    channels = template.get('channels', {})
    if (template.get('scope') != JME_SCOPE or template.get('systematic_scope') != JME_SYSTEMATICS
            or template.get('status') != 'templates_ready'
            or template.get('root_integrity_checked') is not True
            or template.get('sr_data_blinded') is not True
            or template.get('auto_mc_stats') != [10, 1, 1]
            or set(template.get('files', {})) != names
            or grid.get('bins') != template.get('bins') or grid.get('bins') != 452
            or grid.get('th1_channels') != len(channels) or len(channels) != 16
            or sum(map(len, channels.values())) != 452
            or any(len(channels.get(f'SR_{mode}_c1_{year}', [])) != count
                   for mode, count in (('highdm', 118), ('lowdm', 30)) for year in (2024, 2025))
            or grid.get('sr_merge') != template.get('sr_merge')):
        raise ValueError('Changed JME template scope, files, binning or statistics')
    merge = grid['sr_merge']
    if digest(internal_path(ROOT/merge['path'])) != merge['sha256']:
        raise ValueError('Changed adopted SR projection')
    support = grid.get('fit_compatibility', {})
    exclusions = support.get('preliminary_fit_exclusions', [])
    expected_exclusions = {(f'SR_highdm_c1_{year}', 'QCD_Nb1_u6') for year in (2024, 2025)}
    if (support != template.get('fit_compatibility') or support.get('status') != 'ready'
            or support.get('auto_mc_stats') != [10, 1, 1]
            or support.get('interpolation_changed') is not False
            or support.get('endpoints_modified') is not False
            or support.get('all_template_objects_preserved') is not True
            or support.get('background_issues') != [] or support.get('blocked_signal_points') != []
            or len(exclusions) != 2
            or {(x['channel'], x['process']) for x in exclusions} != expected_exclusions
            or any(x.get('root_objects_preserved') is not True or x.get('preliminary') is not True
                   or x.get('decision') != 'user_approved_fit_only_exclusion' for x in exclusions)):
        raise ValueError('Changed approved JME fit-only exclusions or likelihood support')
    expected_input = [dict(path=str(template_path.parent.relative_to(ROOT)),
                           sha256=JME_INPUT_HASHES['templates'])]
    card_manifests = {}
    for point in points:
        card = verify_card(point, digest)
        meta_path = card.with_suffix('.manifest.json')
        meta = read_json(meta_path)
        if (meta.get('inputs') != expected_input or meta.get('fit_ready') is not True
                or meta.get('fit_issues') != [] or meta.get('preliminary_fit_exclusions') != exclusions):
            raise ValueError('Changed JME card inputs, support or approved exclusions')
        card_manifests[str(meta_path.relative_to(ROOT))] = digest(meta_path)
    return dict(scope=JME_SCOPE, input_hashes=dict(JME_INPUT_HASHES),
                systematic_scope=JME_SYSTEMATICS, card_manifests_sha256=card_manifests,
                nonjme16_included=False, physics_model_changed=False)


def target_nll(q):
    normal = statistics.NormalDist()
    return .5*(normal.inv_cdf(q)+normal.inv_cdf(1-q*.05))**2


def choose_range(logs, previous_rmax):
    """A search-range seed only, never an accepted/extrapolated limit.

    Require actual positive NLL samples near an exhausted scan boundary.
    A quadratic estimate to the largest expected-quantile target, with a factor
    two safety margin, proposes a finite new bracket. Only measured crossings
    in the subsequent Combine run can establish a limit.
    """
    if not math.isfinite(previous_rmax) or previous_rmax <= 0:
        raise ValueError('Invalid previous numerical range')
    seed = previous_rmax
    evidence = []
    for name, text, old_range in logs:
        if not math.isfinite(old_range) or old_range <= 0:
            raise ValueError('Invalid logged numerical range')
        blocks = text.split('Will search for NLL crossing by bisection')[1:]
        if len(blocks) > 5:
            continue
        # In a truncated search the first block is still unambiguously median.
        # This is the existing range-parser correction, not new quantile values.
        partial = 1 < len(blocks) < 5
        if partial:
            blocks = blocks[:1]
        for q, block in zip(QUANTILE_ORDER, blocks):
            scans = [(float(r),float(n)) for r,n in SCAN.findall(block)]
            scans = [(r,n) for r,n in scans if math.isfinite(r) and math.isfinite(n) and r>0 and n>0]
            if len(scans)<3 or any(n>=target_nll(q) for r,n in scans):
                continue
            r, nll = max(scans)
            if r < .9*old_range:
                continue
            proposal = max(2*old_range, 2*r*math.sqrt(target_nll(.975)/nll))
            seed = max(seed, proposal)
            evidence.append(dict(attempt=name, quantile=q, sampled_r=r, sampled_delta_nll=nll,
                                 previous_rmax=old_range, proposed_rmax=proposal))
            if partial:
                evidence[-1]['search_scope'] = 'median_only_from_incomplete_sequence'
    seed = math.ceil(seed/10.)*10.
    if seed > MAX_RMAX:
        raise ValueError('Required numerical seed exceeds finite10000 cap; inspect explicitly')
    return seed, evidence


def configs(seed):
    return [dict(name='bisection_s0_extended', strategy=0, tolerance=.1, rmax=seed),
            dict(name='bisection_s1', strategy=1, tolerance=.1, rmax=seed)]


def command_for(workspace, mass, spec):
    command = fit_command(workspace, mass, spec['rmax'], spec['tolerance'])
    command[command.index('--cminDefaultMinimizerStrategy')+1] = str(spec['strategy'])
    return command


def make_plan(grid, output):
    planned = []
    unsupported = []
    for point in grid['points']:
        key=point['model']+'/'+point['mass'];path=output/key/'state.json'
        record=read_json(path)
        if record['status']=='complete':continue
        try:
            if (record['status']!='failed' or record.get('pid') or not record.get('workspace_valid')
                    or [a['name'] for a in record['attempts']]!=['bisection_s0','bisection_s0_refined']
                    or record['card_sha256']!=point['card_sha256']):
                raise ValueError('Not a completed two-attempt numerical failure')
            workspace=output/key/'workspace.root'
            if sha256(workspace)!=record['workspace_sha256']:raise ValueError('Workspace changed')
            last_range=float(record['command'][record['command'].index('--rMax')+1])
            logs=[];hashes={}
            for a,scale in zip(record['attempts'],(.5,1.)):
                log=internal_path(ROOT/a['log']);hashes[a['log']]=sha256(log)
                logs.append((a['name'],log.read_text(errors='replace'),last_range*scale))
            seed,evidence=choose_range(logs,last_range)
            planned.append(dict(point=point,key=key,source_state_sha256=sha256(path),
                workspace_sha256=sha256(workspace),logs_sha256=hashes,rmax=seed,
                range_evidence=evidence,attempts=configs(seed)))
        except Exception as error:
            unsupported.append(dict(key=key,error=repr(error)))
    return dict(schema='trotasr_bounded_limit_numerics_v1',points=planned,unsupported=unsupported,
        maximum_extra_attempts_per_point=2,old_fit_values_imported=False,physics_model_changed=False,
        range_estimate_is_not_a_limit=True)


def run_point(plan, output, runner, env, digest):
    point,key=plan['point'],plan['key'];work=output/key
    state_path=work/'state.json';snapshot=output/'history/numerical_20260924'/key/'state.json'
    if plan['attempts'] != configs(plan['rmax']) or not 0 < plan['rmax'] <= MAX_RMAX:
        raise ValueError('Changed finite numerical configurations')
    verify_card(point,digest)
    with (work/'point.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=read_json(state_path)
        if record['status']=='complete':
            validate_complete(record);return record
        if record.get('pid'):raise RuntimeError('A point process is still registered')
        if snapshot.exists():
            if sha256(snapshot)!=plan['source_state_sha256']:raise ValueError('Changed source receipt')
        else:
            if sha256(state_path)!=plan['source_state_sha256']:raise ValueError('Changed failed point')
            write_json(snapshot,record)
            if sha256(snapshot)!=plan['source_state_sha256']:raise ValueError('Snapshot bytes differ')
        workspace=work/'workspace.root'
        if sha256(workspace)!=plan['workspace_sha256']:raise ValueError('Changed workspace')
        for name,expected in plan['logs_sha256'].items():
            if sha256(ROOT/name)!=expected:raise ValueError('Changed initial failure log')
        # Check actual saved outputs with the already-validated float/log
        # representation adapter before scheduling any additional fit.
        for attempt in record['attempts'][:2]:
            if attempt.get('exit_code') != 0:
                continue
            log = internal_path(ROOT/attempt['log'])
            product = log.parent/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            passed, evidence = validate(product, log, attempt['tolerance'])
            if passed:
                record.update(status='complete', pid=None, finished=time.time(),
                    output=str(product.relative_to(ROOT)), output_sha256=sha256(product),
                    validation=evidence, current_attempt=attempt['name'],
                    numerical_continuation=str((output/'numerical_plan.json').relative_to(ROOT)),
                    saved_output_revalidated_without_refit=True)
                record.pop('error',None);write_json(state_path,record);return record
        done={a['name'] for a in record['attempts']}
        for spec in plan['attempts']:
            if spec['name'] in done:continue
            here=work/spec['name'];here.mkdir(exist_ok=True)
            (here/'tmp').mkdir(exist_ok=True)
            result=here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            log=here/'combine.log';command=command_for(workspace,point['mass'],spec)
            receipt=runner.run(command,here,log,[result],env=dict(env,TMPDIR=str(here/'tmp')))
            passed,evidence=validate(result,log,spec['tolerance']) if receipt['status']=='complete' else (
                False,dict(reason='Combine command/output failure',exit_code=receipt['exit_code']))
            verify_card(point,digest)
            attempt=dict(spec,passed=passed,exit_code=receipt['exit_code'],log=str(log.relative_to(ROOT)),
                         validation=evidence,command=command)
            write_json(here/'validation.json',attempt);record['attempts'].append(attempt)
            record.update(pid=None,command=command,current_attempt=spec['name'])
            if passed:
                record.update(status='complete',finished=time.time(),output=str(result.relative_to(ROOT)),
                    output_sha256=sha256(result),validation=evidence,
                    numerical_continuation=str((output/'numerical_plan.json').relative_to(ROOT)))
                record.pop('error',None);write_json(state_path,record);return record
            write_json(state_path,record)
        record.update(status='failed',finished=time.time(),error='finite_numerical_configurations_exhausted',pid=None)
        write_json(state_path,record);return record


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


def run(manifest,output,workers,wait_seconds):
    if not 1<=workers<=47 or not 0<=wait_seconds<=21600:raise ValueError('Unbounded continuation settings')
    manifest,output=internal_path(manifest),internal_path(output)
    grid=read_json(manifest)
    digest=IntegrityCache()
    jme_inputs=verify_grid_inputs(manifest,output,grid,workers,digest)
    base_contract=read_json(output/'contract.json')
    if base_contract['manifest_sha256']!=sha256(manifest):raise ValueError('Wrong primary fit grid')
    for name,expected in base_contract['code'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Active fit code changed')
    path=output/'numerical_state.json'
    with (output/'numerical_controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        primary=(output/'controller.lock').open('a');deadline=time.monotonic()+wait_seconds
        try:
            while True:
                current=read_json(output/'state.json')
                try:fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:acquired=False
                else:acquired=True
                if acquired and current['status'] in ('fits_complete_pending_plots','needs_attention'):break
                if acquired:fcntl.flock(primary,fcntl.LOCK_UN)
                if time.monotonic()>=deadline:raise TimeoutError('Primary controller has not finished; no fits changed')
                write_json(path,dict(status='waiting_for_primary_controller',pid=os.getpid(),updated=time.time(),
                                     full_workflow_complete=False))
                time.sleep(max(0,min(30,deadline-time.monotonic())))
        except BaseException:
            primary.close()
            raise
        try:
            pp=output/'numerical_plan.json'
            cp=output/'numerical_contract.json'
            if jme_inputs is not None:
                # The primary lock may have required a long wait. Recheck all
                # immutable inputs under that lock; unchanged ROOTs are cached.
                if verify_grid_inputs(manifest,output,read_json(manifest),workers,digest)!=jme_inputs:
                    raise ValueError('JME inputs changed while waiting for primary lock')
                for name,expected in base_contract['code'].items():
                    if sha256(ROOT/name)!=expected:raise ValueError('Active fit code changed')
                # Never mint a new contract around an unbound interrupted plan.
                if pp.exists() and not cp.exists():
                    raise ValueError('Unbound JME numerical plan retained; explicit inspection required')
            if pp.exists():plan=read_json(pp)
            else:
                plan=make_plan(grid,output);write_json(pp,plan)
                write_json(output/'history/numerical_20260924/controller_state.json',current)
            contract=dict(grid_sha256=sha256(manifest),plan_sha256=sha256(pp),
                          code={n:sha256(ROOT/n) for n in SOURCES})
            if jme_inputs is not None:
                contract['jme_inputs']=jme_inputs
            if cp.exists() and read_json(cp)!=contract:raise ValueError('Changed numerical continuation')
            write_json(cp,contract)
            env=runtime(output);runner=CommandRunner(output/'numerical_commands')
            state=dict(status='running',pid=os.getpid(),started=time.time(),total=len(plan['points']),
                workers=workers,results={},unsupported=plan['unsupported'],plan_sha256=sha256(pp),
                full_workflow_complete=False)
            write_json(path,state)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                tasks={pool.submit(run_point,p,output,runner,env,digest):p['key'] for p in plan['points']}
                for future in as_completed(tasks):
                    key=tasks[future]
                    try:
                        r=future.result();state['results'][key]=dict(status=r['status'],error=r.get('error'))
                    except Exception as error:state['results'][key]=dict(status='failed',error=repr(error))
                    state['updated']=time.time();write_json(path,state)
            results={}
            for point in grid['points']:
                key=point['model']+'/'+point['mass'];r=read_json(output/key/'state.json')
                verify_card(point,digest)
                if r['status']=='complete':validate_complete(r)
                results[key]=r
            for model in ('T2tt','T2bW','T2tb'):
                write_json(output/model/'expected_limits.json',collect(grid['points'],results,model,grid['scope']))
            failed=[k for k,r in results.items() if r['status']!='complete']
            final=dict(status='needs_attention' if failed else 'fits_complete_pending_plots',total=len(results),
                completed=len(results)-len(failed),failed=failed,results=results,observed_sr_used=False,
                numerical_contract_sha256=sha256(cp),full_workflow_complete=False)
            write_json(output/'state.json',final)
            state.update(status='needs_attention' if failed else 'fits_complete_pending_plots',pid=None,
                finished=time.time(),completed=final['completed'],failed=failed)
            write_json(path,state);return 2 if failed else 0
        finally:primary.close()


COMBINED_MANIFEST_SHA = '751af1e833f07f4b60df6411d986782be641d860a8b05c91145c11a6f66ce1d3'


def user_configs(rmax):
    """2026-09-30 explicit user range; two strategies, never unlimited retries."""
    if rmax != 100000.:
        raise ValueError('This handoff requires the explicitly approved rMax=100000')
    return [dict(name='user_s1', strategy=1, tolerance=.01, rmax=rmax),
            dict(name='user_s2', strategy=2, tolerance=.001, rmax=rmax)]


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


def assist_capacity(workers, external, memory, active, primary_slots, primary_children):
    # The old controller does not share our admission lock. Reserve its entire
    # configured pool, including temporarily empty slots, to prevent a race.
    reserved_external = external + max(0, primary_slots-primary_children)
    # One own controller plus one diagnostic/transfer slot are counted in 100.
    return max(0, min(workers, 100-2-reserved_external,
                      active+max(0, (memory-32*1024**3)//(4*1024**3))))


def verify_assist_inputs(manifest, output, digest):
    from TROTASR.workflows.combine_limits import verify_combined_grid
    if (output != ROOT/'stats/combined_systematics_sr118/limits'
            or manifest != output.parent/'grid/manifest.json'):
        raise ValueError('Not the approved current combined 955-point grid')
    if digest(manifest) != COMBINED_MANIFEST_SHA:
        # The approved 2025 SF refresh changes the actual grid hash. Bind the
        # new inputs explicitly; never pretend it is the historical grid.
        from TROTASR.workflows.run_systematic_campaign import refresh_plan, REFRESH_ID
        from TROTASR.workflows.build_nominal_grid import grid_contract
        from TROTASR.workflows.run_nominal_products import upstream_gate, verify_measurements
        campaign=ROOT/'campaigns/nonjme_20260927'
        for folder in (campaign,ROOT/'campaigns/systematics_20260924'):
            spec=refresh_plan(folder)['payload_refresh']
            if spec['id']!=REFRESH_ID or spec['validation']['status']!='passed':
                raise ValueError('2025 payload refresh not fully verified')
        upstream_gate(campaign,combined=True)
        config_path=ROOT/'jsons/combined_systematics_config.json';config=read_json(config_path)
        verify_measurements(campaign,config_path)
        expected=grid_contract(config_path,config,ROOT/config['grid'],False,True)
        if read_json(manifest.parent/'contract.json')!=expected:
            raise ValueError('Refreshed grid input/code contract changed')
    grid = read_json(manifest)
    if grid.get('scope') != 'full_combined_systematic_production' or grid.get('status') != 'cards_ready':
        raise ValueError('Not a ready combined grid')
    verify_combined_grid(grid)
    primary = read_json(output/'contract.json')
    if primary['manifest_sha256'] != digest(manifest):
        raise ValueError('Primary manifest contract changed')
    for name, expected in primary['code'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen primary source changed: '+name)
    for point in grid['points']:
        verify_card(point, digest)
    return grid, primary


def verify_attempt(point, attempt):
    """Validate a saved exact attempt, never the last/strongest fitted value."""
    if attempt.get('exit_code') != 0:
        return False, {'reason': 'nonzero_command_exit'}
    log = internal_path(ROOT/attempt['log'])
    product = log.parent/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
    if attempt.get('output') and internal_path(ROOT/attempt['output']) != product:
        raise ValueError('Attempt output/log mismatch')
    if attempt.get('log_sha256') != sha256(log):
        raise ValueError('Attempt log hash mismatch')
    if not product.is_file():
        return False, {'reason': 'missing_saved_output'}
    if attempt.get('output_sha256') != sha256(product):
        raise ValueError('Attempt ROOT hash mismatch')
    return validate(product, log, attempt['tolerance'])


def same_primary_failure(current, original):
    # The unchanged primary may encounter an assisted terminal point later.
    # Its no-new-attempt path updates only `finished`; retain both receipts.
    return ({k:v for k,v in current.items() if k != 'finished'} ==
            {k:v for k,v in original.items() if k != 'finished'})


def assist_retry(point, output, auxiliary, env, digest, rmax):
    """Point-locked failure-only retry; primary state stays read-only until handoff."""
    key = point['model']+'/'+point['mass']
    work, target = output/key, auxiliary/key
    target.mkdir(parents=True, exist_ok=True)
    with (work/'point.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'deferred'}
        record = read_json(work/'state.json')
        if record['status'] == 'complete':
            return {'status': 'already_complete'}
        if (record['status'] != 'failed' or record.get('pid')
                or not record.get('workspace_valid')
                or [a['name'] for a in record['attempts']] != ['bisection_s0', 'bisection_s0_refined']):
            raise ValueError('Not a terminal, workspace-valid, two-attempt primary failure')
        verify_card(point, digest)
        workspace = work/'workspace.root'
        if sha256(workspace) != record['workspace_sha256']:
            raise ValueError('Failed-point workspace changed')
        plan = dict(point=point, source_state_sha256=sha256(work/'state.json'),
                    workspace_sha256=record['workspace_sha256'], attempts=user_configs(rmax))
        plan_path = target/'plan.json'
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError('Changed frozen failure plan')
        if not plan_path.exists():
            write_json(plan_path, plan)
            write_json(target/'primary_state.json', record)
        state_path = target/'state.json'
        state = read_json(state_path) if state_path.exists() else dict(status='running', attempts=[])
        if state['status'] in ('complete', 'failed'):
            return state
        def accepted(attempt, evidence, reused):
            state.update(status='complete', point=point, accepted_attempt=attempt,
                output=attempt['output'], output_sha256=attempt['output_sha256'],
                validation=evidence, saved_output_revalidated_without_refit=reused,
                source_state_sha256=plan['source_state_sha256'], finished=time.time())
            write_json(state_path, state)
            return state
        for attempt in record['attempts']:
            passed, evidence = verify_attempt(point, attempt)
            if passed:
                return accepted(attempt, evidence, True)
        runner = CommandRunner(auxiliary/'command_receipts')
        for spec in user_configs(rmax):
            old = next((a for a in state['attempts'] if a['name'] == spec['name']), None)
            if old:
                passed, evidence = verify_attempt(point, old)
                if passed:
                    return accepted(old, evidence, False)
                continue
            here = target/spec['name']; (here/'tmp').mkdir(parents=True, exist_ok=True)
            result = here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            log = here/'combine.log'
            command = command_for(workspace, point['mass'], spec)
            receipt = runner.run(command, here, log, [result], env=dict(env, TMPDIR=str(here/'tmp')))
            attempt = dict(spec, command=command, exit_code=receipt['exit_code'],
                log=str(log.relative_to(ROOT)), log_sha256=sha256(log),
                output=str(result.relative_to(ROOT)),
                output_sha256=sha256(result) if result.is_file() else None)
            passed, evidence = verify_attempt(point, attempt)
            attempt.update(passed=passed, validation=evidence)
            verify_card(point, digest)
            if sha256(work/'state.json') != plan['source_state_sha256']:
                raise ValueError('Primary failed state changed during locked retry')
            write_json(here/'validation.json', attempt)
            state['attempts'].append(attempt)
            write_json(state_path, state)
            if passed:
                return accepted(attempt, evidence, False)
        state.update(status='failed', point=point, finished=time.time(),
                     error='two_user_configurations_exhausted')
        write_json(state_path, state)
        return state


def adaptive_policy():
    return dict(id='measured_nll_range_v1', maximum_rmax=100000.,
                maximum_new_attempts=4, strategies=[1, 2, 1, 2],
                tolerances=[.01, .001, .01, .001],
                seed_is_not_a_limit=True, acceptance='existing_measured_five_crossings',
                asimov_warning_veto=False, physics_model_changed=False)


def adaptive_seed(logs, ceiling=100000.):
    """Use measured profile samples to seed the next full expected-limit fit.

    No fitted limit (including a broken bisection return) enters this estimate.
    Log-log interpolation or a quadratic extrapolation is a proposal only.
    Prefer the attempt sampling nearest the required target, not the one with
    the strongest returned limit. If no profile evaluations exist,
    use the saved Asimov fit error solely as an explicitly unverified seed.
    """
    if ceiling != 100000.:
        raise ValueError('Adaptive ceiling must be the user-approved 100000')
    target = target_nll(.975)
    candidates = []
    for name, text in logs:
        rows = [(float(r), float(n)) for r, n in SCAN.findall(text)]
        rows = sorted(set((r, n) for r, n in rows
                          if math.isfinite(r) and math.isfinite(n) and r > 0 and n > 0))
        if not rows:
            continue
        below = [(r, n) for r, n in rows if n < target]
        above = [(r, n) for r, n in rows if n >= target]
        lo = max(below, key=lambda row: row[1]) if below else None
        hi = min(above, key=lambda row: row[1]) if above else None
        if lo and hi and lo[0] < hi[0] and lo[1] < hi[1]:
            fraction = math.log(target/lo[1])/math.log(hi[1]/lo[1])
            estimate = math.exp(math.log(lo[0])+fraction*math.log(hi[0]/lo[0]))
            evidence = dict(method='log_log_measured_bracket_seed', samples=[list(lo), list(hi)])
        else:
            r, n = min(rows, key=lambda row: abs(math.log(row[1]/target)))
            estimate = r*math.sqrt(target/n)
            evidence = dict(method='quadratic_measured_profile_seed', samples=[[r, n]])
        proposal = max(.0001, min(ceiling, 1.5*estimate))
        score = min(abs(math.log(n/target)) for r,n in rows)
        candidates.append((score, float(format(proposal, '.6g')), dict(evidence, source_log=name,
            target_delta_nll=target, safety_factor=1.5, unbounded_proposal=1.5*estimate,
            ceiling_reached=1.5*estimate >= ceiling, measured_crossing_verified=False)))
    if candidates:
        _, seed, evidence = min(candidates, key=lambda row: row[0])
        return seed, evidence
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    # Original history order avoids taking the huge error/step induced solely
    # by the later uniform rMax100000 run as a new measurement of sensitivity.
    for name, text in logs:
        block = text.split('Fit to asimov dataset:')
        if len(block) != 2:
            continue
        block = block[1].split('Will search for NLL crossing by bisection')[0]
        values = re.findall(r'^\s*r\s+('+number+r')\s+('+number+r')\s+\+/-\s+('+number+r')', block, re.M)
        if len(values) == 1:
            error = float(values[0][2])
            if math.isfinite(error) and error > 0:
                proposal = min(ceiling, max(.0001, 1.5*math.sqrt(2*target)*error))
                return float(format(proposal, '.6g')), dict(method='asimov_error_seed_no_profile_rows',
                    source_log=name, fit_error=error, measured_crossing_verified=False,
                    warning_is_fatal=False)
    raise ValueError('No measured NLL or finite Asimov fit error; explicit diagnosis required')


def adaptive_predecessor(output, digest):
    """Verify a drained predecessor; never mutate its frozen contract or source."""
    old = output/'numerical_rmax100000'
    contract = read_json(old/'contract.json')
    state = read_json(old/'state.json')
    if state.get('pid') and assist_identity(state['pid']):
        raise ValueError('Predecessor still draining; do not duplicate its workers')
    expected = 'f7596f4d1aa261017f913f20144b59ebcebfe33ab32e0cd29f1927453af3e3ef'
    backup = output.parent/'history/limit_assistance/source_before_adaptive/continue_limit_fits.py'
    if contract['code'].get('workflows/continue_limit_fits.py') != expected or digest(backup) != expected:
        raise ValueError('Missing exact predecessor source archive')
    for name, wanted in contract['code'].items():
        if name != 'workflows/continue_limit_fits.py' and digest(ROOT/name) != wanted:
            raise ValueError('Predecessor dependency changed: '+name)
    receipts = {}; commands = {}
    for path in sorted((old/'command_receipts/commands').glob('*.json')):
        row = read_json(path)
        if row['status'] == 'running' or row.get('pid'):
            raise ValueError('Unfinished predecessor command: '+str(path))
        receipts[str(path.relative_to(ROOT))] = digest(path)
        commands[row['log']] = row
    for path in sorted(old.glob('T2*/*/state.json')):
        record = read_json(path)
        if record['status'] not in ('complete', 'failed'):
            raise ValueError('Unfinished predecessor point: '+str(path))
        for attempt in record.get('attempts', []):
            row = commands.get(attempt['log'])
            if (row is None or row['command'] != list(map(str, attempt['command']))
                    or row.get('exit_code') != attempt['exit_code']
                    or digest(ROOT/attempt['log']) != attempt['log_sha256']
                    or row.get('products', {}).get(attempt['output']) != attempt['output_sha256']):
                raise ValueError('Predecessor attempt/command receipt mismatch')
            if attempt['output_sha256'] and digest(ROOT/attempt['output']) != attempt['output_sha256']:
                raise ValueError('Predecessor output changed')
    for key, before in contract['preserved_successes'].items():
        if (digest(output/key/'state.json') != before['state_sha256']
                or digest(ROOT/before['output']) != before['output_sha256']):
            raise ValueError('Predecessor protected success changed: '+key)
    return dict(contract_sha256=digest(old/'contract.json'), source_backup=str(backup.relative_to(ROOT)),
                source_sha256=expected, command_receipts_sha256=receipts,
                original_state_preserved=True)


def adaptive_retry(point, output, auxiliary, env, digest):
    """Failure-only, point-locked and bounded; accept exact saved evidence first."""
    key = point['model']+'/'+point['mass']; work = output/key; target = auxiliary/key
    with (work/'point.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status':'deferred'}
        record = read_json(work/'state.json')
        if record['status'] == 'complete':
            return {'status':'already_complete'}
        if record['status'] != 'failed' or record.get('pid') or not record.get('workspace_valid'):
            raise ValueError('Adaptive retry requires a terminal workspace-valid failure')
        verify_card(point, digest)
        workspace = work/'workspace.root'
        if digest(workspace) != record['workspace_sha256']:
            raise ValueError('Failed-point workspace changed')
        previous = list(record['attempts'])
        old_path = output/'numerical_rmax100000'/key/'state.json'
        if old_path.exists():
            old = read_json(old_path)
            if old['status'] not in ('complete','failed'):
                raise ValueError('Predecessor point not terminal')
            previous.extend(old.get('attempts', []))
        target.mkdir(parents=True, exist_ok=True)
        path = target/'state.json'
        if path.exists():
            saved = read_json(path)
            if saved['status'] in ('complete','failed'):
                return saved
            raise ValueError('Interrupted adaptive attempt retained; no automatic reset')
        write_json(target/'primary_state.json', record)
        source_sha = sha256(work/'state.json')
        state = dict(status='running', point=point, attempts=[], previous_attempts=previous,
                     source_state_sha256=source_sha, workspace_sha256=record['workspace_sha256'])
        write_json(path, state)
        logs = []
        def accept(attempt, evidence, reused):
            state.update(status='complete', accepted_attempt=attempt, output=attempt['output'],
                output_sha256=attempt['output_sha256'], validation=evidence,
                saved_output_revalidated_without_refit=reused, finished=time.time())
            write_json(path, state)
            return state
        for attempt in previous:
            passed, evidence = verify_attempt(point, attempt)
            if passed:
                return accept(attempt, evidence, True)
            log = internal_path(ROOT/attempt['log'])
            if sha256(log) != attempt['log_sha256']:
                raise ValueError('Changed historical NLL log')
            logs.append((attempt['log'], log.read_text(errors='replace')))
        runner = CommandRunner(auxiliary/'command_receipts')
        used = set()
        policy = adaptive_policy()
        for index, (strategy, tolerance) in enumerate(zip(policy['strategies'], policy['tolerances'])):
            try:
                seed, evidence = adaptive_seed(logs)
            except ValueError as error:
                state.update(status='failed', error='no_usable_range_evidence',
                             detail=str(error), finished=time.time())
                write_json(path, state)
                return state
            identity = (seed, strategy, tolerance)
            if identity in used:
                state['stop_reason'] = 'same_measured_seed_and_strategy_already_attempted'
                continue
            used.add(identity)
            spec = dict(name='adaptive_nll_%d' % (index+1), strategy=strategy,
                        tolerance=tolerance, rmax=seed, range_evidence=evidence)
            here = target/spec['name']; (here/'tmp').mkdir(parents=True, exist_ok=False)
            write_json(here/'plan.json', dict(spec, input_log_hashes={n:sha256(ROOT/n) for n,t in logs}))
            result = here/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
            log = here/'combine.log'
            command = command_for(workspace, point['mass'], spec)
            receipt = runner.run(command, here, log, [result], env=dict(env,TMPDIR=str(here/'tmp')))
            attempt = dict(spec, command=command, exit_code=receipt['exit_code'],
                log=str(log.relative_to(ROOT)), log_sha256=sha256(log),
                output=str(result.relative_to(ROOT)), output_sha256=sha256(result) if result.is_file() else None)
            passed, validation = verify_attempt(point, attempt)
            attempt.update(passed=passed, validation=validation)
            verify_card(point, digest)
            if sha256(work/'state.json') != source_sha or digest(workspace) != record['workspace_sha256']:
                raise ValueError('Primary inputs changed during point-locked retry')
            write_json(here/'validation.json', attempt)
            state['attempts'].append(attempt); write_json(path, state)
            if passed:
                return accept(attempt, validation, False)
            logs.append((attempt['log'], log.read_text(errors='replace')))
        state.update(status='failed', error='bounded_adaptive_policy_exhausted', finished=time.time())
        write_json(path, state)
        return state


def run_assist(manifest, output, workers, rmax, preflight_only=False, adaptive_nll=False):
    """Use spare slots without killing or rewriting the live primary controller.

    New points run the unchanged canonical primary worker, in reverse grid
    order, with its existing per-point lock. If the primary meets an assisted
    running point its lock failure is reconciled from actual point receipts
    only after the primary exits. Retry products remain separate until then.
    """
    from TROTASR.workflows.combine_limits import run_point as primary_point
    from TROTASR.workflows.nominal_campaign import resources
    user_configs(rmax)
    if not 1 <= workers <= 99:
        raise ValueError('Worker request exceeds aggregate budget')
    manifest, output = internal_path(manifest), internal_path(output)
    digest = IntegrityCache()
    grid, primary_contract = verify_assist_inputs(manifest, output, digest)
    points = grid['points']
    auxiliary = output/('numerical_adaptive_nll' if adaptive_nll else 'numerical_rmax100000')
    predecessor = adaptive_predecessor(output, digest) if adaptive_nll else None
    primary_state = read_json(output/'state.json')
    identity = assist_identity(primary_state.get('pid')) if primary_state.get('pid') else None
    if identity:
        args = identity['argv']
        if (str(ROOT/'workflows/combine_limits.py') not in args
                or str(manifest) not in args or str(output) not in args or '--workers' not in args):
            raise ValueError('Unexpected primary process identity')
        primary_workers = int(args[args.index('--workers')+1])
    else:
        primary_workers = 0
    baseline = {}
    for point in points:
        key = point['model']+'/'+point['mass']; path = output/key/'state.json'
        if path.exists():
            record = read_json(path)
            if record['status'] == 'complete':
                validate_complete(record)
                baseline[key] = dict(state_sha256=sha256(path), output=record['output'],
                                     output_sha256=record['output_sha256'])
    if preflight_only:
        print({'status': 'preflight_passed', 'points': len(points), 'verified_successes': len(baseline),
               'primary': identity, 'rmax': rmax,
               'maximum_extra_attempts': 4 if adaptive_nll else 2,
               'adaptive_nll': adaptive_nll}, flush=True)
        return 0
    auxiliary.mkdir(exist_ok=True)
    with (output/'numerical_controller.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (auxiliary/'contract.json').exists():
            raise ValueError('Existing assistance run retained; explicit inspected resume required')
        contract = dict(manifest_sha256=sha256(manifest), primary_contract=primary_contract,
            code={n:sha256(ROOT/n) for n in SOURCES},
            policy=adaptive_policy() if adaptive_nll else user_configs(rmax), predecessor=predecessor,
            aggregate_slots=100, reserved_diagnostic_slots=1, workers=workers,
            primary_identity=identity, primary_workers=primary_workers, preserved_successes=baseline,
            physics_model_changed=False, observed_sr_used=False)
        write_json(auxiliary/'contract.json', contract)
        env = runtime(auxiliary)
        errors = {}; active = {}; started = time.time()
        def dispatch(point, kind):
            try:
                if kind == 'pending':
                    return primary_point(point, output, env, digest)
                if adaptive_nll:
                    return adaptive_retry(point, output, auxiliary, env, digest)
                return assist_retry(point, output, auxiliary, env, digest, rmax)
            except BlockingIOError:
                return {'status': 'deferred'}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                for future in list(active):
                    if not future.done():
                        continue
                    key = active.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        errors[key] = repr(error)
                live = assist_identity(identity['pid']) if identity else None
                if live is not None and live != identity:
                    raise ValueError('Primary PID identity changed; no signaling or adoption')
                external, memory = resources()
                children = 0
                if live:
                    for proc in Path('/proc').iterdir():
                        try:
                            fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
                            children += fields[0] != 'Z' and int(fields[1]) == identity['pid']
                        except (OSError, IndexError, ValueError):
                            pass
                limit = assist_capacity(workers, external, memory, len(active),
                                        primary_workers if live else 0, children)
                candidates = []; counts = dict(complete=0, failed=0, running=0, pending=0)
                unfinished = False
                for point in reversed(points):
                    key = point['model']+'/'+point['mass']; path = output/key/'state.json'
                    record = read_json(path) if path.exists() else {'status':'pending'}
                    counts[record['status']] = counts.get(record['status'],0)+1
                    if key in active.values() or key in errors or record['status'] == 'complete':
                        continue
                    retry_path = auxiliary/key/'state.json'
                    if record['status'] == 'failed' and retry_path.exists() and read_json(retry_path)['status'] in ('complete','failed'):
                        continue
                    unfinished = True
                    if record['status'] in ('failed','pending'):
                        candidates.append((point,record['status']))
                # Failed points first, then the opposite end of the pending grid.
                candidates.sort(key=lambda item: item[1] != 'failed')
                for point, kind in candidates[:max(0,limit-len(active))]:
                    active[pool.submit(dispatch, point, kind)] = point['model']+'/'+point['mass']
                write_json(auxiliary/'state.json', dict(status='running', pid=os.getpid(), started=started,
                    updated=time.time(), active=len(active), admission_limit=limit, raw_counts=counts,
                    primary_alive=bool(live), primary_reserved_workers=primary_workers if live else 0,
                    external_slots=external, available_memory_bytes=memory, errors=errors,
                    full_workflow_complete=False))
                if not active and not unfinished and not live:
                    break
                if active:
                    wait(active, timeout=5, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(5)
        # Reconcile only under the now-free original controller lock.
        with (output/'controller.lock').open('a') as primary_lock:
            fcntl.flock(primary_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            verify_assist_inputs(manifest, output, digest)
            for name, expected in contract['code'].items():
                if digest(ROOT/name) != expected:
                    raise ValueError('Assistance source changed: '+name)
            for key, before in baseline.items():
                if (sha256(output/key/'state.json') != before['state_sha256']
                        or sha256(ROOT/before['output']) != before['output_sha256']):
                    raise ValueError('Previously successful point changed: '+key)
            write_json(auxiliary/'primary_controller_final.json', read_json(output/'state.json'))
            results = {}
            for point in points:
                key = point['model']+'/'+point['mass']; path = output/key/'state.json'
                with (path.parent/'point.lock').open('a') as point_lock:
                    fcntl.flock(point_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    record = read_json(path)
                    retry_path = auxiliary/key/'state.json'
                    if record['status'] != 'complete' and retry_path.exists():
                        retry = read_json(retry_path)
                        if retry['status'] == 'complete':
                            original = read_json(auxiliary/key/'primary_state.json')
                            if (sha256(auxiliary/key/'primary_state.json') != retry['source_state_sha256']
                                    or not same_primary_failure(record, original)):
                                raise ValueError('Failed-point receipt changed before handoff')
                            passed, evidence = verify_attempt(point, retry['accepted_attempt'])
                            if not passed:
                                raise ValueError('Retry no longer validates')
                            write_json(auxiliary/key/'primary_state_at_handoff.json', record)
                            known = {a['name'] for a in record['attempts']}
                            for attempt in [*retry.get('previous_attempts', []), *retry['attempts']]:
                                if attempt['name'] not in known:
                                    record['attempts'].append(attempt)
                                    known.add(attempt['name'])
                            record.update(status='complete', pid=None, output=retry['output'],
                                output_sha256=retry['output_sha256'], validation=evidence,
                                numerical_continuation=str((auxiliary/'contract.json').relative_to(ROOT)),
                                finished=retry['finished'])
                            record.pop('error',None)
                            write_json(path,record)
                    if record['status'] == 'complete':
                        validate_complete(record)
                    results[key] = record
            failed = [k for k,r in results.items() if r['status'] != 'complete']
            for model in ('T2tt','T2bW','T2tb'):
                write_json(output/model/'expected_limits.json', collect(points,results,model,grid['scope']))
            final = dict(status='needs_attention' if failed else 'fits_complete_pending_plots',
                total=len(points), completed=len(points)-len(failed), failed=failed, results=results,
                observed_sr_used=False, full_workflow_complete=False,
                numerical_contract_sha256=sha256(auxiliary/'contract.json'))
            write_json(output/'state.json', final)
            write_json(auxiliary/'state.json', dict(final, results={k:r['status'] for k,r in results.items()},
                       pid=None, finished=time.time(), errors=errors))
        return 2 if failed else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=47)
    parser.add_argument('--wait-seconds',type=int,default=21600)
    parser.add_argument('--assist',action='store_true',help='Current combined grid: spare-slot assistance and explicit rMax policy')
    parser.add_argument('--rmax',type=float)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--adaptive-nll',action='store_true',help='Use each failed point measured NLL to seed a bounded search')
    args=parser.parse_args()
    if args.assist:
        raise SystemExit(run_assist(args.manifest,args.output,args.workers,args.rmax,args.preflight_only,args.adaptive_nll))
    if args.rmax is not None or args.preflight_only or args.adaptive_nll:
        parser.error('--rmax/--preflight-only require --assist')
    raise SystemExit(run(args.manifest,args.output,args.workers,args.wait_seconds))
