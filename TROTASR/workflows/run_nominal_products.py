"""Continue the existing full nominal chain; never submit histograms again.

Wait for verified measurements, build new native cards, and run the existing
limits/impact/CR diagnostics and renderers. Exit zero means commands succeeded,
NOT that visual QA or the AN inventory is complete. Failed commands are never
blindly retried by a repeated heartbeat. No physics definitions live here.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.diagnostic_common import live_resources
from TROTASR.workflows.nominal_campaign import THREAD_ENV

FIT_WORKERS = 47
READY = 'measurements_complete_pending_plots_cards_fits'
STAGES = ('histograms', 'merge_2024', 'merge_2025', 'measure_2024', 'measure_2025')


def upstream_gate(campaign, jme=False, combined=False):
    """Queue disappearance and a completed-file count alone are not success."""
    plan = read_json(campaign/'plan.json')
    if combined:
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        for year in ('2024', '2025'):
            path = campaign/'combined/merged'/('systematics_'+year+'.json.gz')
            summary = read_json(path.with_name('systematics_'+year+'.summary.json'))
            provenance = summary.get('combination_provenance', {})
            coverage = summary.get('coverage', {})
            if (summary.get('status') != 'complete'
                    or summary.get('scope') != 'full_combined_systematic_production'
                    or summary.get('sha256') != sha256(path)
                    or provenance.get('policy') != COMBINATION_POLICY
                    or set(provenance.get('sources', {})) != {'jme', 'nonjme'}
                    or coverage.get('full_input_inventory') is not True
                    or coverage.get('failed_files') != 0
                    or coverage.get('completed_files') != coverage.get('expected_files')):
                raise ValueError('Incomplete or changed combined year product')
            for source in provenance['sources'].values():
                original = internal_path(ROOT/source['path'])
                receipt = original.with_name('systematics_'+year+'.summary.json')
                if sha256(original) != source['sha256'] or sha256(receipt) != source['summary_sha256']:
                    raise ValueError('Changed original systematic component')
        return True
    if jme:
        if plan.get('shape_mode')!='cms_trota_jme' or plan.get('scope')!='full_cms_trota_jme_production':
            raise ValueError('Consistent full JME central/weights campaign required')
        state=read_json(campaign/'state.json')
        if state.get('active') or state.get('pending') or state.get('failed') or state.get('completed')!=14474:
            raise ValueError('JME campaign is not fully complete')
        stage=read_json(campaign/'stage_result.json')
        if stage.get('status')!='central_weights_jme_histograms_merged':
            raise ValueError('JME strict merges not complete')
        for year in ('2024','2025'):
            path=campaign/'merged'/('systematics_'+year+'.json.gz')
            summary=read_json(path.with_suffix('').with_suffix('.summary.json'))
            coverage=summary['coverage']
            if (summary['status']!='complete' or summary['sha256']!=sha256(path)
                or not coverage['full_input_inventory'] or coverage['failed_files']
                or coverage['expected_files']!=coverage['completed_files']):
                raise ValueError('Incomplete JME merge coverage')
        return True
    if plan.get('scope') != 'full_nominal_production':
        raise ValueError('Full-input campaign required')
    state = read_json(campaign/'chain_state.json')
    if state.get('status') == 'running':
        pid = state.get('pid')
        if not isinstance(pid, int) or pid < 2:
            raise ValueError('Missing chain controller PID')
        command = Path('/proc')/str(pid)/'cmdline'
        if not command.exists() or b'run_nominal_chain.py' not in command.read_bytes():
            raise RuntimeError('Upstream running marker has no matching live controller')
        return False
    diagnostic_exit = (state.get('status') == 'needs_attention'
        and set(state.get('stages', {})) == set(STAGES)
        and all(state['stages'][n].get('exit_code') == 0 for n in STAGES[:3])
        and all(state['stages'][n].get('exit_code') in (0, 2) for n in STAGES[3:])
        and any(state['stages'][n].get('exit_code') == 2 for n in STAGES[3:]))
    if (state.get('status') != READY and not diagnostic_exit) or set(state.get('stages', {})) != set(STAGES):
        raise RuntimeError('Upstream chain needs attention: '+str(state.get('status')))
    if not diagnostic_exit and any(v.get('exit_code') != 0 for v in state['stages'].values()):
        raise ValueError('An upstream stage failed')
    # Exit2 is NOT success: verify_measurements must independently prove that
    # only a diagnostic TF display is unsupported before any stage is launched.
    return True


def diagnostic_tf_gaps(source):
    """Separate a proved empty SR numerator from failed fit inputs.

    The TF payload stays unsupported, with null values unchanged. Cards do not
    consume these displayed ratios: they use the actual SR/CR MC components and
    their existing, separately audited CR-only parameter policy.
    """
    if (source.get('schema') != 'trotasr_transfer_factors_v2'
            or source.get('status') != 'unsupported_cells' or not source.get('issues')):
        raise ValueError('Unrecognized diagnostic TF status')
    expected = [dict(mode='highdm', route='qcd_qcdcr', category='Nb1', status='unsupported_cells')]
    if source['issues'] != expected:
        raise ValueError('Unreviewed TF support issue')
    row = source['highdm']['records']['qcd_qcdcr']['Nb1']
    if row.get('process_coverage') != dict(numerator=True, denominator=True):
        raise ValueError('Missing TF process coverage')
    columns = ('valid', 'numerator', 'numerator_sumw2', 'denominator',
               'denominator_sumw2', 'transfer_factor', 'mcstat', 'mechanical_relative_residual')
    if any(len(row.get(k, [])) != 8 for k in columns):
        raise ValueError('Invalid diagnostic TF layout')
    gaps = []
    for i, valid in enumerate(row['valid']):
        if valid:
            continue
        if (row['numerator'][i] != 0 or row['numerator_sumw2'][i] != 0
                or not math.isfinite(row['denominator'][i]) or row['denominator'][i] <= 0
                or not math.isfinite(row['denominator_sumw2'][i]) or row['denominator_sumw2'][i] < 0
                or any(row[k][i] is not None for k in ('transfer_factor', 'mcstat', 'mechanical_relative_residual'))):
            raise ValueError('TF issue is not an empty SR numerator with supported CR')
        gaps.append(dict(mode='highdm', route='qcd_qcdcr', category='Nb1',
                         native_bin_1based=i+1, numerator=0, numerator_sumw2=0,
                         denominator=row['denominator'][i], status='unsupported_cells'))
    if [g['native_bin_1based'] for g in gaps] != [7, 8]:
        raise ValueError('Unreviewed empty TF cell set')
    return gaps


def verify_measurements(campaign, config_path, years=('2024', '2025')):
    config = read_json(config_path)
    jme = config.get('fit_campaign')=='jme_weights_preliminary'
    combined = config.get('fit_campaign')=='combined_systematics_preliminary'
    if set(config['years']) != {'2024', '2025'}:
        raise ValueError('Both analysis years are required')
    snapshots = {}
    for year in years:
        fields = config['years'][year]
        hists = ((campaign/'combined/merged') if combined else campaign/'merged')/(
            ('systematics_' if jme or combined else 'nominal_')+year+'.json.gz')
        if internal_path(ROOT/fields['hists']) != hists:
            raise ValueError('Grid uses a different histogram campaign')
        hist_digest = sha256(hists)
        folder = (campaign/'combined/measurements'/year if combined else
            ROOT/'estimations'/('jme_weights_20260928' if jme else campaign.name)/year)
        manifest = read_json(folder/'measurement_manifest.json')
        stages = manifest['stages']
        diagnostic_only = (manifest['status'] == 'blocked'
            and stages.get('transfer_factors') == 'unsupported_cells'
            and all(stages.get(n) == 'complete' for n in ('sgamma', 'rz_high', 'rz_low', 'double_ratio')))
        if (manifest['status'] != 'complete' and not diagnostic_only
                or manifest['provenance']['hist_input_sha256'] != hist_digest
                or str(manifest['provenance']['campaign_year']) != year
                or set(stages) != {'sgamma', 'rz_high', 'rz_low', 'double_ratio', 'transfer_factors'}
                or not diagnostic_only and any(v != 'complete' for v in stages.values())):
            raise ValueError('Incomplete or stale current measurements: '+year)
        for name, digest in manifest['files'].items():
            if sha256(internal_path(folder/name)) != digest:
                raise ValueError('Changed measurement product: '+year+'/'+name)
        for name, digest in manifest['provenance']['code'].items():
            if sha256(internal_path(ROOT/name)) != digest:
                raise ValueError('Changed measurement implementation: '+name)
        gaps = diagnostic_tf_gaps(read_json(folder/'transfer_factors.json.gz')) if diagnostic_only else []
        for name in ('sgamma', 'rz_high', 'rz_low', 'double_ratio'):
            if internal_path(ROOT/fields[name]) != folder/(name+'.json.gz'):
                raise ValueError('Grid uses a different measurement: '+name)
        snapshots[year] = dict(hist_sha256=hist_digest,
            measurement_manifest_sha256=sha256(folder/'measurement_manifest.json'),
            diagnostic_tf_gaps=gaps, measurement_status=manifest['status'])
    return snapshots


def stage_plan(campaign, output, config, worker_budget=None):
    """One auxiliary stage at a time, alongside 47+47 bounded fit workers.

    Peak owned slots: this controller + two fit controllers +94 fits + one
    auxiliary controller/child pair =99. Live admission also counts other jobs.
    """
    campaign, output, config = map(internal_path, (campaign, output, config))
    configuration=read_json(config)
    jme=configuration.get('fit_campaign')=='jme_weights_preliminary'
    combined=configuration.get('fit_campaign')=='combined_systematics_preliminary'
    fit_workers=10 if jme else FIT_WORKERS
    if worker_budget is not None:
        if not 10<=worker_budget<=98:raise ValueError('Invalid shared production slot budget')
        fit_workers=min(fit_workers,(worker_budget-8)//2)
    grid = output/'grid'; manifest = grid/'manifest.json'
    specs = {}
    def add(name, script, args, receipt, statuses, lane='aux', dependencies=()):
        specs[name] = dict(command=[sys.executable, '-u', str(ROOT/'workflows'/script), *map(str, args)],
            receipt=str(receipt.relative_to(ROOT)), statuses=list(statuses), lane=lane,
            dependencies=list(dependencies))
    if combined:
        for year in ('2024', '2025'):
            add('measure_'+year, 'build_background_estimation.py',
                ['--hists', campaign/'combined/merged'/('systematics_'+year+'.json.gz'),
                 '--output', campaign/'combined/measurements'/year],
                campaign/'combined/measurements'/year/'measurement_manifest.json', ['complete'], lane='measure')
            specs['measure_'+year].update(measurement_year=year, campaign=str(campaign), config=str(config))
    add('grid', 'build_nominal_grid.py', ['--config', config, '--output', grid]+(
        ['--preserve-unsupported-templates'] if combined else []),
        manifest, ['cards_ready'], dependencies=('measure_2024', 'measure_2025') if combined else ())
    for name, script, receipt, status in (
            ('limits', 'combine_limits.py', 'state.json', 'fits_complete_pending_plots'),
            ('impacts', 'combine_impacts.py', 'impact_status.json', 'fits_complete')):
        add(name, script, ['--manifest', manifest, '--output', output/name, '--workers', fit_workers],
            output/name/receipt, [status], lane='fit', dependencies=['grid'])
    add('cronly', 'combine_cronly.py', ['--manifest', manifest, '--output', output/'cronly'],
        output/'cronly/state.json', ['fits_complete_pending_plots'], dependencies=['grid'])
    for year in ('2024', '2025'):
        hist = (campaign/'combined/merged'/('systematics_'+year+'.json.gz') if combined else
                campaign/'merged'/('nominal_'+year+'.json.gz'))
        factors = campaign/'combined/measurements'/year if combined else ROOT/'estimations'/campaign.name/year
        plots = output/'plots'/year
        for name, script, args in (
                ('distributions', 'plotting_nominal_distributions.py',
                 ['--hists', hist, '--rz-high', factors/'rz_high.json.gz', '--rz-low', factors/'rz_low.json.gz']),
                ('measurements', 'plotting_background_measurements.py', ['--hists', hist, '--measurements', factors]),
                ('transfer_factors', 'plotting_transfer_factors.py', ['--hists', hist, '--input', factors/'transfer_factors.json.gz']),
                ('gnn_controls', 'plotting_gnn_controls.py', ['--hists', hist, '--rz-low', factors/'rz_low.json.gz'])):
            add(name+'_'+year, script, [*args, '--output', plots/name],
                plots/name/'plot_manifest.json', ['generated_pending_visual_QA'],
                dependencies=('measure_'+year,) if combined else ())
    add('predictions', 'plotting_card_predictions.py', ['--manifest', manifest, '--output', output/'plots/predictions'],
        output/'plots/predictions/plot_manifest.json', ['generated_pending_visual_QA'], dependencies=['grid'])
    add('cronly_plot', 'combine_cronly.py', ['--manifest', manifest, '--output', output/'cronly', '--plot-only'],
        output/'cronly/state.json', ['generated_pending_visual_QA'], dependencies=['cronly'])
    add('contours', 'plotting_limit_contour.py', ['--limits', output/'limits', '--grid', manifest,
        '--output', output/'plots/contours'], output/'plots/contours/plot_manifest.json',
        ['generated_pending_visual_QA'], dependencies=['limits'])
    add('impact_recovery', 'recover_impacts.py', ['--manifest', manifest, '--work', output/'impacts', '--workers', fit_workers],
        output/'impacts/direct_profile/status.json', ['fits_complete'], lane='fit')
    if combined:
        specs['impact_recovery']['command'].append('--bounded-sgamma')
        specs['impact_recovery']['receipt']=str((output/'impacts/bounded_sgamma_20260930/status.json').relative_to(ROOT))
        specs['impact_recovery']['bounded_sgamma']=True
        add('limit_recovery','reconcile_limit_fits.py',['--manifest',manifest,
            '--output',output/'limits','--recover-unfinished','--workers',fit_workers],
            output/'limits/numerical_all_unfinished_20260930/state.json',
            ['all_verified_pending_reconciliation'],lane='fit')
        specs['contours']['dependencies']=['limit_recovery']
        specs['contours']['command'].append('--allow-partial')
    add('impacts_plot', 'combine_impacts.py', ['--manifest', manifest, '--output', output/'impacts', '--plot-only'],
        output/'impacts/impact_status.json', ['generated_pending_visual_QA'])
    if combined:
        for year in ('2024','2025'):
            specs['transfer_factors_'+year]['statuses'].append('partial_generated_pending_visual_QA')
            specs['distributions_'+year]['command'].append('--include-four-jet-bin')
        for model in ('T2bW','T2tb'):
            for mass in ('mStop1200_mLSP500','mStop1000_mLSP800'):
                name='impact_'+model+'_'+mass
                folder=output/'impacts_benchmarks_20261001'/(model+'_'+mass)
                args=['--manifest',manifest,'--output',folder,'--model',model,'--mass',mass]
                add(name,'combine_impacts.py',[*args,'--workers',
                    min(98,worker_budget-8) if worker_budget is not None else fit_workers],
                    folder/'impact_status.json',['fits_complete'],lane='benchmark',dependencies=['grid'])
                add(name+'_plot','combine_impacts.py',[*args,'--plot-only'],
                    folder/'impact_status.json',['generated_pending_visual_QA'],dependencies=[name])
    if jme:
        # Preserve the existing entry points. Start fits with ten workers per
        # pool alongside the active 72-worker non-JME propagation. Rendering
        # follows separately after fit validation and required visual QA.
        specs={n:s for n,s in specs.items() if n in ('grid','limits','impacts','cronly','impact_recovery')}
    return specs


def disposition(name, spec, records):
    """Return ready/wait/blocked/not_needed; no auto-reset of failed commands."""
    if spec.get('execution_blocked_reason'):
        return 'blocked'
    if name=='limit_recovery':
        parent=records.get('limits',{})
        if parent.get('status')=='succeeded':return 'not_needed'
        if parent.get('status')=='failed':
            actual=parent.get('receipt_snapshot',{})
            return 'ready' if actual.get('status')=='needs_attention' and actual.get('results') else 'blocked'
        return 'blocked' if parent.get('status')=='blocked' else 'wait'
    if name=='contours' and spec['dependencies']==['limit_recovery']:
        primary=records.get('limits',{}).get('status')
        recovery=records.get('limit_recovery',{}).get('status')
        if primary=='succeeded' or recovery in ('succeeded','failed'):return 'ready'
        return 'blocked' if recovery=='blocked' else 'wait'
    if name == 'impact_recovery':
        parent = records.get('impacts', {})
        if parent.get('status') == 'succeeded':
            return 'not_needed'
        if parent.get('status') == 'failed':
            if spec.get('recovery_supported') is False:
                return 'blocked'
            actual = parent.get('receipt_snapshot', {})
            names = actual.get('failed_nuisances', [])
            if spec.get('bounded_sgamma'):
                return ('ready' if actual.get('phase')=='finite_fits_exhausted' and names
                    and all(n.startswith('CMS_NPS26012_sgamma_shape_lowdm_') for n in names) else 'blocked')
            return ('ready' if actual.get('phase') == 'finite_fits_exhausted' and names
                    and all('sgamma_shape' in n or 'qcd_norm' in n for n in names) else 'blocked')
        if parent.get('status') == 'blocked':
            return 'blocked'
        return 'wait'
    if name == 'impacts_plot':
        parents = [records.get(k, {}).get('status') for k in ('impacts', 'impact_recovery')]
        if 'succeeded' in parents:
            return 'ready'
        return 'blocked' if all(v in ('failed', 'blocked', 'not_needed') for v in parents) else 'wait'
    parents = [records.get(k, {}).get('status') for k in spec['dependencies']]
    if any(v in ('failed', 'blocked') for v in parents):
        return 'blocked'
    return 'ready' if all(v == 'succeeded' for v in parents) else 'wait'


def verify_receipt(spec, exit_code):
    path = ROOT/spec['receipt']
    value = read_json(path) if path.exists() else {}
    passed = exit_code == 0 and value.get('status') in spec['statuses']
    if spec.get('measurement_year') and exit_code in (0, 2):
        verified = verify_measurements(Path(spec['campaign']), Path(spec['config']),
            years=(spec['measurement_year'],))
        value = dict(value, independently_verified=verified)
        passed = True  # Only the exact diagnostic empty-SR TF case may retain exit2.
    # All renderers publish file hashes. Verify their files, never interpret
    # 'generated_pending_visual_QA' as an AN-delivered or visually passed plot.
    if passed:
        for name, digest in value.get('files', {}).items():
            if sha256(internal_path(path.parent/name)) != digest:
                raise ValueError('Stage output checksum mismatch: '+name)
        for name, digest in value.get('figures', {}).items():
            if sha256(internal_path(ROOT/name)) != digest:
                raise ValueError('Diagnostic figure checksum mismatch: '+name)
    return passed, value


def run(campaign, output, config, refresh_year=None, worker_budget=None):
    if sys.platform != 'linux':
        raise RuntimeError('Full production and native ROOT outputs remain on hep2')
    campaign, output, config = map(internal_path, (campaign, output, config))
    jme=read_json(config).get('fit_campaign')=='jme_weights_preliminary'
    combined=read_json(config).get('fit_campaign')=='combined_systematics_preliminary'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'products.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = stage_plan(campaign, output, config,worker_budget)
        contract = dict(campaign=str(campaign.relative_to(ROOT)), plan_sha256=sha256(campaign/'plan.json'),
            grid_config_sha256=sha256(config), code_sha256=sha256(Path(__file__)), stages=plan)
        path = output/'products_state.json'
        if path.exists():
            previous = read_json(path)
            if previous['contract'] != contract:
                raise ValueError('Divergent products controller contract; preserve prior run')
            if previous.get('stages'):
                if previous.get('status') in ('generated_pending_visual_QA_and_inventory', 'fits_complete_pending_plots', 'needs_attention'):
                    print(previous['status']); return 0 if not previous['failed_stages'] else 2
                raise RuntimeError('Interrupted controller retained; inspect child PIDs before explicit recovery')
        state = dict(status='waiting_for_full_measurements', pid=os.getpid(), started=time.time(),
            contract=contract, stages={}, full_workflow_complete=False, an_inventory_entries_delivered=0)
        active = {}
        def save():
            state['updated'] = time.time(); write_json(path, state)
        try:
            save()
            while not upstream_gate(campaign,jme=jme,combined=combined):
                save(); time.sleep(30)
            state['verified_inputs'] = {} if combined else verify_measurements(campaign, config)
            if refresh_year:
                if refresh_year!=2025 or not combined:raise ValueError('Only current combined 2025 refresh is approved')
                reused=verify_measurements(campaign,config,years=('2024',))
                state['verified_inputs'].update(reused)
                state['stages']['measure_2024']=dict(status='succeeded',reused_unchanged=True,
                    verified=reused,finished=time.time())
                for name,spec in plan.items():
                    if name.endswith('_2024') and name!='measure_2024' and (ROOT/spec['receipt']).exists():
                        passed,receipt=verify_receipt(spec,0)
                        if not passed:raise ValueError('Invalid preserved 2024 plot: '+name)
                        state['stages'][name]=dict(status='succeeded',reused_unchanged=True,receipt_snapshot=receipt)
            for year, snapshot in state['verified_inputs'].items():
                if snapshot['diagnostic_tf_gaps'] and 'transfer_factors_'+year in plan and not combined:
                    state['stages']['transfer_factors_'+year] = dict(status='blocked',
                        reason='diagnostic_empty_SR_cells_not_card_inputs',
                        unavailable=snapshot['diagnostic_tf_gaps'])
            state['status'] = 'running'; save()
            while len(state['stages']) < len(plan) or active:
                for name, (process, stream) in list(active.items()):
                    rc = process.poll()
                    if rc is None:
                        continue
                    stream.close(); del active[name]
                    record = state['stages'][name]
                    try:
                        passed, receipt = verify_receipt(plan[name], rc)
                        record.update(status='succeeded' if passed else 'failed', receipt_snapshot=receipt)
                        if passed and plan[name].get('measurement_year'):
                            verified = receipt['independently_verified']
                            state['verified_inputs'].update(verified)
                            for year, snapshot in verified.items():
                                if snapshot['diagnostic_tf_gaps'] and not combined:
                                    state['stages']['transfer_factors_'+year] = dict(status='blocked',
                                        reason='diagnostic_empty_SR_cells_not_card_inputs',
                                        unavailable=snapshot['diagnostic_tf_gaps'])
                    except Exception as error:
                        record.update(status='failed', error=repr(error))
                    record.update(exit_code=rc, pid=None, finished=time.time())
                    save()
                for name, spec in plan.items():
                    if name in state['stages']:
                        continue
                    choice = disposition(name, spec, state['stages'])
                    if choice in ('blocked', 'not_needed'):
                        state['stages'][name] = dict(status=choice, dependencies=spec['dependencies'],
                            reason=spec.get('execution_blocked_reason')); save()
                        continue
                    if choice != 'ready' or (spec['lane'] in ('aux','benchmark') and any(plan[n]['lane'] == spec['lane'] for n in active)):
                        continue
                    if spec['lane']=='benchmark' and any(s['lane']=='fit' and
                            state['stages'].get(n,{}).get('status') not in ('succeeded','failed','blocked','not_needed')
                            for n,s in plan.items()):
                        continue
                    # Serialize admission with diagnostic controllers; leave
                    # capacity for a native child's own subprocess as well.
                    with (ROOT/'stats/fit_admission.lock').open('a') as admission:
                        fcntl.flock(admission, fcntl.LOCK_EX)
                        slots, memory = live_resources()
                        if slots + 2 > 100 or memory < 36*1024**3:
                            continue
                        logs = output/'logs'; logs.mkdir(exist_ok=True)
                        logfile = logs/(name+'.log')
                        if logfile.exists():
                            raise FileExistsError('Unregistered stage log retained: '+name)
                        stream = logfile.open('x')
                        process = subprocess.Popen(spec['command'], cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                        active[name] = (process, stream)
                        state['stages'][name] = dict(status='running', pid=process.pid, started=time.time(),
                            log=str(logfile.relative_to(ROOT)), command=spec['command'],
                            entrypoint_sha256=sha256(Path(spec['command'][2])))
                        save()
                if active or len(state['stages']) < len(plan):
                    time.sleep(5)
            failed = [n for n, r in state['stages'].items() if r['status'] in ('failed', 'blocked')]
            # Keep original impact failure in history even when bounded recovery
            # succeeded. Exclude only that recovered stage from remaining work.
            unresolved = [n for n in failed if not (
                n == 'impacts' and state['stages']['impact_recovery']['status'] == 'succeeded' or
                n == 'limits' and state['stages'].get('limit_recovery',{}).get('status') == 'succeeded')]
            state.update(status='needs_attention' if unresolved else 'fits_complete_pending_plots' if jme else 'generated_pending_visual_QA_and_inventory',
                failed_stages=unresolved, original_failed_stages=failed, finished=time.time())
            return 2 if unresolved else 0
        except BaseException as error:
            state.update(status='controller_error_children_retained', error=repr(error),
                child_pids={name:proc.pid for name, (proc, _) in active.items()})
            raise
        finally:
            state['pid'] = None
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'stats/nominal_grid_config.json')
    parser.add_argument('--refresh-year',type=int,choices=[2025])
    parser.add_argument('--worker-budget',type=int,help='Aggregate allowance for this products chain, below the shared 100-slot ceiling')
    args = parser.parse_args()
    raise SystemExit(run(args.campaign, args.output, args.config,args.refresh_year,args.worker_budget))
