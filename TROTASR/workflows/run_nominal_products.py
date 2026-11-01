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
import shutil
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
RUN3_PLOT_PRIORITY = 'approved_run3_plot_priority_20261027'
MINIMAL_TEMPLATE_CHECK = 'approved_minimal_root_checks_20261027'
DISPLAY_LIMIT_RANGE = 'approved_display_limit_range_20261027'
PARALLEL_CARD_RESUME = 'approved_parallel_card_resume_20261027'
LIMIT_RANGE_ARGUMENTS = {'limits':['--plot-range-only'],
    'limit_recovery':['--plot-range-only','--assist-active-primary'], 'contours':['--live-snapshot']}
INFERENCE_WRAPPER = '/hep2-scratch/twkim/postprocessing/TROTASR/workflows/run_inference.sh'
RUN3_PLOT_NAMES = tuple('gnn_controls_'+y for y in ('2022','2022EE','2023','2023BPix','2024','2025')) + tuple(
    'predictions_'+y for y in ('2022','2022EE','2023','2023BPix','2024','2025','highdm','lowdm_early','lowdm_late'))


def canonical_limit_output(output):
    """Use the unified fit caller for the six-era products destination."""
    return Path(output) in (ROOT/'stats/run3', ROOT/'stats/run3_systematics_20261025')


def require_canonical_limit_completion(value):
    """A successful command alone is not complete displayed-grid collection."""
    expected = dict(source_total=955, total=895, completed=895, remaining=0,
        excluded_out_of_plot_range=60,
        expected_points_by_model={'T2tt':205, 'T2bW':345, 'T2tb':345})
    if (value.get('status') != 'fits_complete_pending_plots' or value.get('pid')
            or value.get('canonical_limits_pipeline') is not True
            or value.get('plot_range_only') is not True
            or value.get('observed_sr_used') is not False
            or any(value.get(key) != target for key, target in expected.items())):
        raise ValueError('Canonical limits require all 895 collected expected points before final plots')


def require_canonical_plan_handoff(previous, plan):
    """Do not reinterpret a frozen split-stage plan as the new single caller."""
    if not plan.get('limits', {}).get('canonical_limits_pipeline'):
        return
    old = previous.get('contract', {}).get('stages', {})
    if (old.get('limits', {}).get('canonical_limits_pipeline') is not True
            or 'limit_recovery' in old):
        raise ValueError('Legacy split-limit products plan retained unchanged; '
            'an explicit new-plan handoff is required for the canonical limits pipeline')


def products_contract_matches(previous, current):
    """New six-era callers record source hashes without using them as input gates."""
    def canonical(contract):
        stages = contract.get('stages', {})
        limits = stages.get('limits', {})
        return (limits.get('canonical_limits_pipeline') is True
            and 'limit_recovery' not in stages
            and limits.get('receipt') in (
                'stats/run3_systematics_20261025/limits/workflow_state.json',
                'stats/run3/limits/workflow_state.json'))
    if not (canonical(previous) and canonical(current)):
        return previous == current
    provenance_only = {'code_sha256', 'code'}
    return ({key: value for key, value in previous.items() if key not in provenance_only}
        == {key: value for key, value in current.items() if key not in provenance_only})


def command_entrypoint(command):
    """Parse recorded legacy Python or explicit canonical-wrapper commands."""
    if len(command) >= 3 and command[0] == 'bash' and Path(command[1]).name == 'run_inference.sh':
        return Path(command[2]), command[3:]
    if len(command) >= 3 and command[1] == '-u':
        return Path(command[2]), command[3:]
    raise ValueError('Unsupported products stage command')


def worker_command(command):
    script, arguments = command_entrypoint(command)
    if script.parent != ROOT/'workflows':
        raise ValueError('Only canonical internal workflow entrypoints may launch')
    return ['bash', INFERENCE_WRAPPER, str(script), *arguments]


def stage_order(plan):
    """Stable insertion order for defaults; explicit plot priority otherwise."""
    return sorted(plan.items(), key=lambda item:item[1].get('priority', 100))


def lane_available(plan, active, spec):
    count = sum(plan[name]['lane'] == spec['lane'] for name in active)
    return count < (4 if spec['lane'] == 'plot' else 1) if spec['lane'] in ('plot','aux','benchmark') else True


def resource_admission(slots, memory, active_plot_count=0, prioritized=False, free_disk=None):
    if not prioritized:
        return slots + 2 <= 100 and memory >= 36*1024**3
    # Count all live user work via live_resources, not a per-lane slot budget.
    # Pending plot starts reserve 4 GiB each in addition to the 32 GiB floor;
    # this is an admission reservation, not a measured peak-RSS claim.
    return (slots + 2 <= 120 and memory >= (36+4*active_plot_count)*1024**3
            and free_disk is not None and free_disk >= 80*1024**3)


def limit_pair_pending(records, stages=('limits','limit_recovery')):
    terminal={'succeeded','failed','blocked','not_needed'}
    return any(records.get(name,{}).get('status') not in terminal for name in stages)


def limit_priority_admission(name, plan, active, records, worker_budget=98):
    """Reserve the owned limit pool(s) before other work; never kill workers."""
    limit_stages=tuple(stage for stage in ('limits','limit_recovery') if stage in plan)
    if not limit_pair_pending(records, limit_stages):
        return True
    spec=plan[name]
    if name.startswith('impact') or spec['lane']=='benchmark':
        return False
    def slots(stage):
        item=plan[stage];command=item['command']
        if '--workers' in command:
            return 1+int(command[command.index('--workers')+1])
        if stage=='grid' and '--template-verify-workers' in command:
            return 1+max(int(command[command.index(flag)+1]) for flag in
                         ('--template-verify-workers','--card-workers') if flag in command)
        return 1 if item['lane']=='plot' else 2
    reserved=set(active)|{name}
    if records.get('grid',{}).get('status')=='succeeded':
        for stage in limit_stages:
            if records.get(stage,{}).get('status') not in {'succeeded','failed','blocked','not_needed'}:
                reserved.add(stage)
    budget = 120 if plan.get('limits', {}).get('canonical_limits_pipeline') else (worker_budget or 98)
    return 1+sum(slots(stage) for stage in reserved) <= budget


def nominal_stages(years):
    return ('histograms', *('merge_'+y for y in years), *('measure_'+y for y in years))


def upstream_gate(campaign, jme=False, combined=False):
    """Queue disappearance and a completed-file count alone are not success."""
    plan = read_json(campaign/'plan.json')
    if combined:
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        for year in plan['contracts']:
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
    years = tuple(plan.get('contracts', ('2024', '2025')))
    stages = nominal_stages(years)
    nonmeasure = stages[:1+len(years)]
    measures = stages[1+len(years):]
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
        and set(state.get('stages', {})) == set(stages)
        and all(state['stages'][n].get('exit_code') == 0 for n in nonmeasure)
        and all(state['stages'][n].get('exit_code') in (0, 2) for n in measures)
        and any(state['stages'][n].get('exit_code') == 2 for n in measures))
    if (state.get('status') != READY and not diagnostic_exit) or set(state.get('stages', {})) != set(stages):
        raise RuntimeError('Upstream chain needs attention: '+str(state.get('status')))
    if not diagnostic_exit and any(v.get('exit_code') != 0 for v in state['stages'].values()):
        raise ValueError('An upstream stage failed')
    # Exit2 is NOT success: verify_measurements must independently prove that
    # only a diagnostic TF display is unsupported before any stage is launched.
    return True


def diagnostic_tf_gaps(source, year=None):
    """Separate a proved empty SR numerator from failed fit inputs.

    The TF payload stays unsupported, with null values unchanged. Cards do not
    consume these displayed ratios: they use the actual SR/CR MC components and
    their existing, separately audited CR-only parameter policy.
    """
    if (source.get('schema') != 'trotasr_transfer_factors_v2'
            or source.get('status') != 'unsupported_cells' or not source.get('issues')):
        raise ValueError('Unrecognized diagnostic TF status')
    year = str(year or source.get('year', '2024'))
    reviewed = {'2024': {'Nb1': [7, 8]}, '2025': {'Nb1': [7, 8]},
                '2022': {'Nb1': [6, 7, 8]},
                '2023BPix': {'Nb1': [6, 7, 8], 'Nb2plus': [8]}}
    if year not in reviewed or str(source.get('year', year)) != year:
        raise ValueError('Unreviewed diagnostic TF era')
    expected = [dict(mode='highdm', route='qcd_qcdcr', category=category,
                     status='unsupported_cells') for category in reviewed[year]]
    if year == '2023BPix':
        expected.append(dict(mode='lowdm', route='w_llcr', category='Nb2plus_NISR0', status='signed_mc_bins'))
    issue_key = lambda r: (r['mode'], r['route'], r['category'], r['status'])
    if sorted(source['issues'], key=issue_key) != sorted(expected, key=issue_key):
        raise ValueError('Unreviewed TF support issue')
    gaps = []
    for category, bins in reviewed[year].items():
        row = source['highdm']['records']['qcd_qcdcr'][category]
        gaps.extend(empty_tf_cells(row, category, bins))
    if year == '2023BPix':
        # Re-evaluate the existing pooled-CR ratio/covariance equations. This
        # signed diagnostic is not a fit input; preserve its negative value.
        import numpy as np
        from TROTASR.utils.transfer_factors import score_ratio
        row = source['lowdm']['records']['w_llcr']['Nb2plus_NISR0']
        if row.get('process_coverage') != dict(numerator=True, denominator=True):
            raise ValueError('Missing TF process coverage')
        arrays = [np.asarray(row[k], dtype=float) for k in
                  ('numerator', 'numerator_sumw2', 'cr_score_yield', 'cr_score_sumw2')]
        if (any(a.shape != (5,) or not np.isfinite(a).all() for a in arrays)
                or any((a < 0).any() for a in arrays[1::2]) or arrays[2].sum() <= 0
                or np.flatnonzero(arrays[0] < 0).tolist() != [2]):
            raise ValueError('Unreviewed signed TF diagnostic')
        computed = score_ratio(*arrays)
        for key, value in computed.items():
            actual = row.get(key)
            if isinstance(value, (list, np.ndarray, float, int)):
                if not np.allclose(actual, value, rtol=1e-12, atol=1e-14, equal_nan=False):
                    raise ValueError('Signed TF equation/covariance mismatch: '+key)
            elif actual != value:
                raise ValueError('Signed TF diagnostic status mismatch: '+key)
        gaps.append(dict(mode='lowdm', route='w_llcr', category='Nb2plus_NISR0',
            native_bin_1based=3, numerator=float(arrays[0][2]),
            numerator_sumw2=float(arrays[1][2]), denominator=float(arrays[2].sum()),
            status='signed_mc_bins', diagnostic_only=True))
    return gaps


def empty_tf_cells(row, category, reviewed_bins):
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
        gaps.append(dict(mode='highdm', route='qcd_qcdcr', category=category,
                         native_bin_1based=i+1, numerator=0, numerator_sumw2=0,
                         denominator=row['denominator'][i], status='unsupported_cells'))
    if [g['native_bin_1based'] for g in gaps] != reviewed_bins:
        raise ValueError('Unreviewed empty TF cell set')
    return gaps


def verify_measurements(campaign, config_path, years=None):
    config = read_json(config_path)
    from TROTASR.workflows.build_nominal_grid import require_hybrid_policy, require_run3_systematics_policy
    hybrid = require_hybrid_policy(config)
    run3 = require_run3_systematics_policy(config)
    jme = config.get('fit_campaign')=='jme_weights_preliminary'
    combined = config.get('fit_campaign')=='combined_systematics_preliminary' or run3
    from TROTASR.utils.paths import EARLY_ERAS
    if not (hybrid or run3) and set(config['years']) not in ({'2024', '2025'}, set(EARLY_ERAS)):
        raise ValueError('Complete configured analysis-era group is required')
    late_campaign = ROOT/'campaigns/nonjme_20260927'
    if hybrid or run3:
        upstream_gate(late_campaign, combined=True)
    years = tuple(config['years']) if years is None else years
    snapshots = {}
    for year in years:
        fields = config['years'][year]
        reused_late = (hybrid or run3) and year in ('2024', '2025')
        source_campaign = late_campaign if reused_late else campaign
        source_combined = combined or reused_late
        hists = ((source_campaign/'combined/merged') if source_combined else source_campaign/'merged')/(
            ('systematics_' if jme or source_combined else 'nominal_')+year+'.json.gz')
        if internal_path(ROOT/fields['hists']) != hists:
            raise ValueError('Grid uses a different histogram campaign')
        hist_digest = sha256(hists)
        folder = (source_campaign/'combined/measurements'/year if source_combined else
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
        gaps = diagnostic_tf_gaps(read_json(folder/'transfer_factors.json.gz'), year) if diagnostic_only else []
        for name in ('sgamma', 'rz_high', 'rz_low', 'double_ratio'):
            if internal_path(ROOT/fields[name]) != folder/(name+'.json.gz'):
                raise ValueError('Grid uses a different measurement: '+name)
        snapshots[year] = dict(hist_sha256=hist_digest,
            measurement_manifest_sha256=sha256(folder/'measurement_manifest.json'),
            diagnostic_tf_gaps=gaps, measurement_status=manifest['status'])
        if reused_late:
            actual = dict(hist_sha256=hist_digest,
                summary_sha256=sha256(hists.with_name('systematics_'+year+'.summary.json')),
                measurement_manifest_sha256=snapshots[year]['measurement_manifest_sha256'])
            if actual != config['reused_late_inputs'][year]:
                raise ValueError('Frozen reused late-era inputs changed: '+year)
    return snapshots


def stage_plan(campaign, output, config, worker_budget=None, prioritize_run3_plots=False, limit_plot_range_only=False):
    """One auxiliary stage at a time, alongside 47+47 bounded fit workers.

    Peak owned slots: this controller + two fit controllers +94 fits + one
    auxiliary controller/child pair =99. Live admission also counts other jobs.
    """
    campaign, output, config = map(internal_path, (campaign, output, config))
    configuration=read_json(config)
    from TROTASR.workflows.build_nominal_grid import require_hybrid_policy, require_run3_systematics_policy
    hybrid=require_hybrid_policy(configuration)
    run3=require_run3_systematics_policy(configuration)
    canonical_limits = run3 and canonical_limit_output(output)
    if prioritize_run3_plots and not run3:
        raise ValueError('Plot-priority amendment is only approved for the current six-era systematic model')
    if limit_plot_range_only and not run3:
        raise ValueError('Displayed-range limit execution is only approved for the current six-era systematic model')
    jme=configuration.get('fit_campaign')=='jme_weights_preliminary'
    combined=configuration.get('fit_campaign')=='combined_systematics_preliminary' or run3
    years=tuple(configuration['years'])
    measured_years=tuple(y for y in years if not run3 or y not in ('2024','2025'))
    nominal_first=configuration.get('systematic_evaluation_deferred', False)
    fit_workers=10 if jme else FIT_WORKERS
    if worker_budget is not None:
        if not 10<=worker_budget<=98:raise ValueError('Invalid shared production slot budget')
        fit_workers=min(fit_workers,(worker_budget-8)//2)
    if nominal_first:
        # Only one fit pool: the deferred impacts must not reserve half the
        # compute budget. Retain room for controllers and the auxiliary lane;
    # live admission continues to count the account's other compute processes.
        fit_workers=min(92, (worker_budget if worker_budget is not None else 96)-4)
    grid = output/'grid'; manifest = grid/'manifest.json'
    specs = {}
    def add(name, script, args, receipt, statuses, lane='aux', dependencies=()):
        specs[name] = dict(command=[sys.executable, '-u', str(ROOT/'workflows'/script), *map(str, args)],
            receipt=str(receipt.relative_to(ROOT)), statuses=list(statuses), lane=lane,
            dependencies=list(dependencies))
    if combined:
        for year in measured_years:
            add('measure_'+year, 'build_background_estimation.py',
                ['--hists', campaign/'combined/merged'/('systematics_'+year+'.json.gz'),
                 '--output', campaign/'combined/measurements'/year],
                campaign/'combined/measurements'/year/'measurement_manifest.json', ['complete'], lane='measure')
            specs['measure_'+year].update(measurement_year=year, campaign=str(campaign), config=str(config))
    add('grid', 'build_nominal_grid.py', ['--config', config, '--output', grid]+(
        ['--preserve-unsupported-templates'] if combined or hybrid else []),
        manifest, ['cards_ready'], dependencies=tuple('measure_'+y for y in measured_years) if combined else ())
    for name, script, receipt, status in (
            ('limits', 'combine_limits.py', 'state.json', 'fits_complete_pending_plots'),
            ('impacts', 'combine_impacts.py', 'impact_status.json', 'fits_complete')):
        add(name, script, ['--manifest', manifest, '--output', output/name, '--workers', fit_workers],
            output/name/receipt, [status], lane='fit', dependencies=['grid'])
    add('cronly', 'combine_cronly.py', ['--manifest', manifest, '--output', output/'cronly'],
        output/'cronly/state.json', ['fits_complete_pending_plots'], dependencies=['grid'])
    for year in years:
        if (hybrid or run3) and year in ('2024', '2025'):
            # Existing late-era plots remain valid. Only the six-era likelihood
            # and its limits are new; do not repeat completed per-era rendering.
            continue
        hist = ROOT/configuration['years'][year]['hists']
        factors = (ROOT/configuration['years'][year]['sgamma']).parent
        plots = output/'plots'/year
        for name, script, args in (
                ('distributions', 'plotting_nominal_distributions.py',
                 ['--hists', hist, '--rz-high', factors/'rz_high.json.gz', '--rz-low', factors/'rz_low.json.gz']),
                ('measurements', 'plotting_background_measurements.py', ['--hists', hist, '--measurements', factors]),
                ('transfer_factors', 'plotting_transfer_factors.py', ['--hists', hist, '--input', factors/'transfer_factors.json.gz']),
                ('gnn_controls', 'plotting_gnn_controls.py', ['--hists', hist, '--rz-low', factors/'rz_low.json.gz',
                                                          '--manifest', manifest])):
            add(name+'_'+year, script, [*args, '--output', plots/name],
                plots/name/'plot_manifest.json', ['generated_pending_visual_QA'],
                dependencies=('grid',) if name == 'gnn_controls' else ('measure_'+year,) if combined else ())
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
        specs['impact_recovery']['command'].append('--complete-bounded-sgamma')
        specs['impact_recovery']['receipt']=str((output/'impacts/bounded_sgamma_complete/status.json').relative_to(ROOT))
        specs['impact_recovery']['bounded_sgamma']=True
        if not canonical_limits:
            add('limit_recovery','reconcile_limit_fits.py',['--manifest',manifest,
                '--output',output/'limits','--recover-unfinished','--workers',fit_workers],
                output/'limits/numerical_all_unfinished_20260930/state.json',
                ['all_verified_pending_reconciliation'],lane='fit')
            specs['contours']['dependencies']=['limit_recovery']
        specs['contours']['command'].append('--allow-partial')
    add('impacts_plot', 'combine_impacts.py', ['--manifest', manifest, '--output', output/'impacts', '--plot-only'],
        output/'impacts/impact_status.json', ['generated_pending_visual_QA'])
    if combined:
        for year in measured_years:
            specs['transfer_factors_'+year]['statuses'].append('partial_generated_pending_visual_QA')
            specs['transfer_factors_'+year]['command'].append('--allow-partial')
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
                    folder/'impact_status.json',['generated_pending_visual_QA'],dependencies=[name+'_recovery'])
                add(name+'_recovery','recover_impacts.py',['--manifest',manifest,'--work',folder,
                    '--model',model,'--mass',mass,'--complete-bounded-sgamma','--workers',fit_workers],
                    folder/'bounded_sgamma_complete/status.json',['fits_complete'],lane='benchmark')
                specs[name+'_recovery'].update(bounded_sgamma=True,recovery_parent=name)
    if jme:
        # Preserve the existing entry points. Start fits with ten workers per
        # pool alongside the active 72-worker non-JME propagation. Rendering
        # follows separately after fit validation and required visual QA.
        specs={n:s for n,s in specs.items() if n in ('grid','limits','impacts','cronly','impact_recovery')}
    if nominal_first:
        # Nominal limits first. Impact scans and systematic propagation belong
        # to the subsequent campaign, not a parallel lane of this controller.
        specs={n:s for n,s in specs.items() if not n.startswith(('impact', 'impacts'))}
    if hybrid:
        # Early/late GNN boundaries differ. Keep individual-era plots; never
        # sum their score-bin indices into a falsely common SR/CR display.
        specs.pop('predictions')
        for year in years:
            if 'transfer_factors_'+year in specs:
                specs['transfer_factors_'+year]['command'].append('--allow-partial')
                specs['transfer_factors_'+year]['statuses'].append('partial_generated_pending_visual_QA')
                specs['distributions_'+year]['command'].append('--include-four-jet-bin')
        add('limit_recovery', 'reconcile_limit_fits.py', ['--manifest', manifest,
            '--output', output/'limits', '--recover-unfinished', '--workers', fit_workers],
            output/'limits/numerical_all_unfinished_20260930/state.json',
            ['all_verified_pending_reconciliation'], lane='fit')
        specs['contours']['dependencies'] = ['limit_recovery']
        specs['contours']['command'].append('--allow-partial')
    if run3:
        # Render each era's SR groups without pooling unlike early/late GNN bins.
        specs.pop('predictions')
        for year in years:
            folder=output/'plots'/year/'predictions'
            add('predictions_'+year,'plotting_card_predictions.py',
                ['--manifest',manifest,'--year',year,'--sr-only','--output',folder],
                folder/'plot_manifest.json',['generated_pending_visual_QA'],dependencies=['grid'])
        for name,args in (('highdm',['--high-sr-only']),
                          ('lowdm_early',['--low-sr-only','--era-group','early']),
                          ('lowdm_late',['--low-sr-only','--era-group','late'])):
            folder=output/'plots'/'combined_sr'/name
            add('predictions_'+name,'plotting_card_predictions.py',
                ['--manifest',manifest,'--combine-era-sr',*args,'--output',folder],
                folder/'plot_manifest.json',['generated_pending_visual_QA'],dependencies=['grid'])
        if prioritize_run3_plots:
            for year in ('2024','2025'):
                hist=ROOT/configuration['years'][year]['hists']
                factors=(ROOT/configuration['years'][year]['sgamma']).parent
                folder=output/'plots'/year/'gnn_controls'
                add('gnn_controls_'+year,'plotting_gnn_controls.py',
                    ['--hists',hist,'--rz-low',factors/'rz_low.json.gz','--manifest',manifest,'--output',folder],
                    folder/'plot_manifest.json',['generated_pending_visual_QA'],dependencies=['grid'])
            for name in RUN3_PLOT_NAMES:
                specs[name].update(lane='plot',priority=0)
        if canonical_limits:
            # One controller owns primary, finite recovery and collection. Keep
            # the old split-stage plans immutable; this is a new-plan layout.
            specs['limits']['command'].append('--plot-range-only')
            command=specs['limits']['command']
            command[command.index('--workers')+1]='70'
            specs['limits'].update(canonical_limits_pipeline=True,
                receipt=str((output/'limits/workflow_state.json').relative_to(ROOT)))
            specs['contours']['dependencies']=['limits']
        elif limit_plot_range_only:
            for name,args in LIMIT_RANGE_ARGUMENTS.items():
                specs[name]['command'].extend(args)
    return specs


def resolve_products_handoff(campaign, output, config):
    """Redirect an unstarted downstream products destination.

    The live parent and its immutable histogram plan are never restarted.
    Only an unstarted four-era products destination may be redirected.
    """
    source = read_json(config)
    handoff = source.get('products_handoff')
    if handoff is None:
        return output, config, None
    from TROTASR.utils.paths import EARLY_ERAS
    from TROTASR.workflows.build_nominal_grid import require_hybrid_policy
    target = internal_path(ROOT/handoff['config'])
    destination = internal_path(ROOT/handoff['output'])
    chosen = read_json(target)
    if (handoff.get('user_approved') is not True or tuple(source['years']) != EARLY_ERAS
            or not require_hybrid_policy(chosen) or chosen.get('products_handoff')
            or chosen['years'] != {**source['years'], **{y:chosen['years'][y] for y in ('2024','2025')}}
            or any(chosen.get(k) != source.get(k) for k in
                ('grid', 'theory', 'sr_merge', 'statistical_revision', 'background_estimation_uncertainties'))
            or destination == output or (output/'products_state.json').exists()):
        raise ValueError('Invalid or already-started downstream combination handoff')
    return destination, target, dict(source_config=str(config.relative_to(ROOT)),
        source_config_sha256=sha256(config), prior_output=str(output.relative_to(ROOT)),
        target_config=str(target.relative_to(ROOT)), target_config_sha256=sha256(target),
        target_output=str(destination.relative_to(ROOT)), user_approved=True)


def live_primary_ready(spec, parent):
    """Check the live limit controller's identity and displayed-point selection."""
    command=spec['command']
    manifest=internal_path(Path(command[command.index('--manifest')+1]))
    output=internal_path(Path(command[command.index('--output')+1]))
    if not (output/'state.json').is_file() or not (output/'contract.json').is_file():
        return False
    state=read_json(output/'state.json')
    if state.get('status') != 'running' or not state.get('pid'):
        return False
    if state['pid'] != parent.get('pid'):
        raise ValueError('Live assistance primary PID differs from the owned products stage')
    from TROTASR.workflows.continue_limit_fits import assist_identity
    identity=assist_identity(state['pid'])
    if identity is None:
        return False
    if parent.get('start_ticks') is not None and identity['start_ticks'] != parent['start_ticks']:
        raise ValueError('Live primary start-ticks differ from the owned products stage')
    args=identity['argv']
    workers=command[command.index('--workers')+1]
    if (str(ROOT/'workflows/combine_limits.py') not in args or str(manifest) not in args
            or str(output) not in args or '--plot-range-only' not in args or '--workers' not in args
            or args[args.index('--workers')+1] != workers):
        raise ValueError('Unexpected live displayed-only primary command')
    from TROTASR.workflows.combine_limits import selected_fit_points, selection_metadata
    _,selection=selected_fit_points(read_json(manifest),True)
    contract=read_json(output/'contract.json')
    if (contract.get('manifest_sha256') != sha256(manifest) or contract.get('fit_selection') != selection
            or any(state.get(k) != v for k,v in selection_metadata(selection).items())
            or state.get('observed_sr_used') is not False):
        raise ValueError('Live primary manifest/895-point selection contract changed')
    return True


def disposition(name, spec, records):
    """Return ready/wait/blocked/not_needed; no auto-reset of failed commands."""
    if spec.get('execution_blocked_reason'):
        return 'blocked'
    if name=='limit_recovery':
        parent=records.get('limits',{})
        if parent.get('status')=='running' and '--assist-active-primary' in spec['command']:
            return 'ready' if live_primary_ready(spec,parent) else 'wait'
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
    if name == 'impact_recovery' or spec.get('recovery_parent'):
        parent = records.get(spec.get('recovery_parent','impacts'), {})
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
    return 'ready' if all(v in ('succeeded','not_needed') for v in parents) else 'wait'


def verify_receipt(spec, exit_code):
    path = ROOT/spec['receipt']
    value = read_json(path) if path.exists() else {}
    allowed = set(spec['statuses'])
    if allowed.intersection(('fits_complete', 'fits_complete_pending_plots')):
        allowed.add('generated_pending_visual_QA')
    passed = exit_code == 0 and value.get('status') in allowed
    command=spec.get('command',[])
    if passed and spec.get('canonical_limits_pipeline'):
        require_canonical_limit_completion(value)
    if passed and '--template-check-mode' in command:
        expected=dict(template_check_mode='minimal',numeric_readback_performed=False,
            numeric_readback_checked=False,root_integrity_checked=True,
            root_integrity_scope='file_open_header_object_inventory_types_and_checksums',
            template_check_user_approved=True)
        if (command[command.index('--template-check-mode')+1] != 'minimal'
                or any(value.get(key) != target or isinstance(target,bool) and value.get(key) is not target
                       for key,target in expected.items())):
            raise ValueError('Minimal grid receipt must truthfully identify its non-exhaustive check scope')
    if passed and '--resume-cards' in command:
        reference=value.get('card_execution',{})
        execution_path=path.parent/'card_execution.json'
        if (reference.get('path') != str(execution_path.relative_to(ROOT))
                or not execution_path.is_file() or sha256(execution_path) != reference.get('sha256')):
            raise ValueError('Card execution receipt path/checksum mismatch')
        cards=read_json(execution_path);execution=cards.get('execution',{})
        representative=execution.get('representative_check',{})
        counts=dict(completed=955,total=955,workers=16,active=0,pending=0)
        exits=cards.get('worker_exit_codes')
        if (cards.get('status') != 'cards_complete' or cards.get('pid') is not None
                or any(type(cards.get(k)) is not int or cards[k] != v for k,v in counts.items())
                or cards.get('worker_pids') != [] or not isinstance(exits,dict)
                or any(type(code) is not int or code != 0 for code in exits.values())
                or cards.get('root_and_cache_rewrite') is not False
                or cards.get('reused_pairs_preserved') is not True
                or execution.get('resume_cards') is not True
                or any(representative.get(k) is not True for k in ('text_exact','metadata_exact'))
                or representative.get('existing_bytes_modified') is not False):
            raise ValueError('Card execution receipt lacks complete preserved-card/worker-exit evidence')
    if passed and '--plot-range-only' in command:
        expected=dict(source_total=955,total=895,excluded_out_of_plot_range=60,
            expected_points_by_model=dict(T2tt=205,T2bW=345,T2tb=345))
        if value.get('plot_range_only') is not True or any(value.get(k) != v for k,v in expected.items()):
            raise ValueError('Displayed-range fit receipt must preserve the 955-point source and report only 895 selected points')
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


def approved_plot_plan_migration(previous, contract):
    """Exactly two new late GNN stages; no changes to completed work or fits."""
    old, new = previous['contract']['stages'], contract['stages']
    approval = contract.get('approved_plot_priority', {})
    added = {'gnn_controls_2024','gnn_controls_2025'}
    if (approval.get('id') != RUN3_PLOT_PRIORITY or approval.get('user_approved') is not True
            or set(new)-set(old) != added or set(old)-set(new)):
        raise ValueError('Unapproved products stage inventory migration')
    preserved = {kind+'_'+year for kind in ('measure','distributions','measurements','transfer_factors')
                 for year in ('2022','2022EE','2023','2023BPix')}
    if any(previous['stages'].get(name,{}).get('status') != 'succeeded' for name in preserved):
        raise ValueError('The sixteen completed early stages must be present for this exact migration')
    grid_command=new['grid']['command']
    config=read_json(Path(grid_command[grid_command.index('--config')+1]))
    from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
    if not require_run3_systematics_policy(config):
        raise ValueError('Only the approved Run-3 systematic plan may migrate')
    for name in added:
        spec = new[name]
        script, args = command_entrypoint(spec['command'])
        year = name.rsplit('_',1)[1]
        expected_args=list(map(str,('--hists',ROOT/config['years'][year]['hists'],
            '--rz-low',(ROOT/config['years'][year]['sgamma']).parent/'rz_low.json.gz',
            '--manifest',ROOT/new['grid']['receipt'],'--output',(ROOT/spec['receipt']).parent)))
        if (script != ROOT/'workflows/plotting_gnn_controls.py' or spec['lane'] != 'plot'
                or spec.get('priority') != 0 or spec['dependencies'] != ['grid']
                or not spec['receipt'].endswith('/plots/'+year+'/gnn_controls/plot_manifest.json')
                or args != expected_args or spec['statuses'] != ['generated_pending_visual_QA']
                or name in previous['stages']):
            raise ValueError('Invalid additional late GNN stage: '+name)
    for name, before in old.items():
        after, record = new[name], previous['stages'].get(name, {})
        if before == after:
            continue
        if record.get('status') == 'succeeded':
            raise ValueError('Cannot change an already successful stage: '+name)
        if name in RUN3_PLOT_NAMES:
            if (record.get('status') not in (None,'blocked','not_needed')
                    or any(record.get(k) is not None for k in ('pid','started','log','exit_code'))
                    or before.get('lane') != 'aux' or after.get('lane') != 'plot'
                    or after.get('priority') != 0
                    or {k:v for k,v in before.items() if k not in ('lane','priority')}
                    != {k:v for k,v in after.items() if k not in ('lane','priority')}):
                raise ValueError('Only unstarted plot lane/priority may change: '+name)
        elif name == 'grid':
            approved_written_grid_resume(before, after, record)
        else:
            raise ValueError('Plot-priority migration changed an unrelated stage: '+name)


def approved_written_grid_resume(before, after, record):
    if record.get('status') != 'failed' or record.get('pid'):
        raise ValueError('Written-template resume requires a stopped failed grid')
    def base_command(spec, new=False):
        command = list(spec['command'])
        if '--resume-extracted' in command:
            command.remove('--resume-extracted')
        if new:
            if '--resume-written-templates' not in command or '--template-verify-workers' not in command:
                raise ValueError('Explicit written-template parallel readback flags required')
            command.remove('--resume-written-templates')
            index=command.index('--template-verify-workers')
            if command[index+1] != '12':
                raise ValueError('Only the approved twelve-reader handoff may amend this failed grid')
            del command[index:index+2]
        return command
    if (base_command(before) != base_command(after, True)
            or {k:v for k,v in before.items() if k != 'command'} != {k:v for k,v in after.items() if k != 'command'}):
        raise ValueError('Written-template recovery changed grid inputs or options')


def approved_products_early_qcd_config_change(previous, contract, output):
    """Share the builder's frozen prior-config approval; no generic SHA waiver."""
    command=contract['stages']['grid']['command']
    if (previous['stages'].get('grid',{}).get('status') != 'failed'
            or '--resume-written-templates' not in command):
        raise ValueError('Exact early-QCD config migration requires a stopped written-template recovery')
    approval_path=output/'grid/written_template_resume_approval.json'
    approval=read_json(approval_path)
    prior=internal_path(ROOT/approval['prior_config'])
    current=internal_path(Path(command[command.index('--config')+1]))
    if (sha256(prior) != previous['contract']['grid_config_sha256']
            or sha256(current) != contract['grid_config_sha256']):
        raise ValueError('Early-QCD prior/current config snapshot checksum mismatch')
    from TROTASR.workflows.early_qcd_fit_policy import approved_early_qcd_config_change
    approved_early_qcd_config_change(read_json(prior),read_json(current))


def approved_limit_stage_flags(previous, contract):
    """Only the two unstarted limit stages and contours may change arguments."""
    old=previous['contract'];before=old['stages'];after=contract['stages']
    approval=contract.get('approved_limit_plot_range',{})
    if (approval.get('id') != DISPLAY_LIMIT_RANGE or approval.get('user_approved') is not True
            or approval.get('mstop_gev') != [600,1800] or approval.get('mlsp_gev') != [1,1800]
            or approval.get('expected_display_points') != 895 or len(before) != 52 or set(before) != set(after)
            or any(old[k] != contract[k] for k in ('campaign','plan_sha256','grid_config_sha256'))):
        raise ValueError('Displayed-range amendment requires the exact unchanged approved 52-stage inputs')
    preserved={kind+'_'+year for kind in ('measure','distributions','measurements','transfer_factors')
               for year in ('2022','2022EE','2023','2023BPix')}
    if any(previous['stages'].get(n,{}).get('status') != 'succeeded' for n in preserved):
        raise ValueError('Displayed-range amendment must preserve the sixteen completed stages')
    for name,arguments in LIMIT_RANGE_ARGUMENTS.items():
        record=previous['stages'].get(name,{})
        if (record.get('status') not in (None,'blocked','not_needed')
                or any(record.get(k) is not None for k in ('pid','started','log','exit_code'))
                or any(arg in before[name]['command'] for arg in arguments)
                or name=='contours' and '--allow-partial' not in before[name]['command']
                or after[name]['command'] != [*before[name]['command'],*arguments]
                or {k:v for k,v in before[name].items() if k != 'command'}
                != {k:v for k,v in after[name].items() if k != 'command'}):
            raise ValueError('Only exact unstarted displayed-limit/live-recovery/contour arguments may change: '+name)


def approved_limit_plot_range_plan_migration(previous, contract):
    approved_limit_stage_flags(previous,contract)
    if any(spec != contract['stages'][name] for name,spec in previous['contract']['stages'].items()
           if name not in LIMIT_RANGE_ARGUMENTS):
        raise ValueError('Displayed-range amendment changed another stage')


def approved_minimal_template_plan_migration(previous, contract):
    """The stopped 52-stage plan may add only the explicit minimal-check pair."""
    old=previous['contract'];before=old['stages'];after=contract['stages']
    approval=contract.get('approved_template_check_mode',{})
    grid=previous['stages'].get('grid',{})
    if (approval.get('id') != MINIMAL_TEMPLATE_CHECK or approval.get('user_approved') is not True
            or approval.get('check_mode') != 'minimal' or len(before) != 52 or set(before) != set(after)
            or any(old[k] != contract[k] for k in ('campaign','plan_sha256','grid_config_sha256'))
            or grid.get('status') != 'failed' or grid.get('pid')):
        raise ValueError('Minimal-check handoff requires the exact stopped approved 52-stage plan')
    preserved={kind+'_'+year for kind in ('measure','distributions','measurements','transfer_factors')
               for year in ('2022','2022EE','2023','2023BPix')}
    if any(previous['stages'].get(n,{}).get('status') != 'succeeded' for n in preserved):
        raise ValueError('Minimal-check handoff must preserve the sixteen completed stages')
    command=before['grid']['command']
    allowed_other_changes=set()
    if any(before[n] != after[n] for n in LIMIT_RANGE_ARGUMENTS):
        approved_limit_stage_flags(previous,contract)
        allowed_other_changes=set(LIMIT_RANGE_ARGUMENTS)
    if ('--template-check-mode' in command or '--resume-written-templates' not in command
            or '--template-verify-workers' not in command
            or command[command.index('--template-verify-workers')+1] != '12'
            or after['grid']['command'] != [*command,'--template-check-mode','minimal']
            or {k:v for k,v in before['grid'].items() if k != 'command'}
            != {k:v for k,v in after['grid'].items() if k != 'command'}
            or any(before[n] != after[n] for n in before if n not in {'grid',*allowed_other_changes})):
        raise ValueError('Only --template-check-mode minimal may be added to the stopped grid')


def card_resume_command(command, workers=16):
    """Replace only the completed template-check phase with native card resume."""
    if (workers != 16 or command.count('--resume-written-templates') != 1
            or any(flag in command for flag in ('--resume-extracted','--resume-cards','--card-workers'))):
        raise ValueError('Card handoff requires exactly the approved sixteen-worker written-grid replacement')
    result=list(command)
    result[result.index('--resume-written-templates')]='--resume-cards'
    return [*result,'--card-workers','16']


def approved_card_resume_plan_migration(previous, contract):
    """Stopped 52-stage plan: grid-only card resume; all other specs immutable."""
    old=previous['contract'];before=old['stages'];after=contract['stages']
    approval=contract.get('approved_card_resume',{})
    grid=previous['stages'].get('grid',{})
    if (approval.get('id') != PARALLEL_CARD_RESUME or approval.get('user_approved') is not True
            or approval.get('card_workers') != 16 or len(before) != 52 or set(before) != set(after)
            or any(old[k] != contract[k] for k in ('campaign','plan_sha256','grid_config_sha256'))
            or grid.get('status') != 'failed' or grid.get('pid')):
        raise ValueError('Card handoff requires the exact stopped approved 52-stage inputs')
    preserved={kind+'_'+year for kind in ('measure','distributions','measurements','transfer_factors')
               for year in ('2022','2022EE','2023','2023BPix')}
    if any(previous['stages'].get(n,{}).get('status') != 'succeeded' for n in preserved):
        raise ValueError('Card handoff must preserve the sixteen completed stages')
    command=before['grid']['command']
    for flag,value in (('--template-verify-workers','12'),('--template-check-mode','minimal')):
        if command.count(flag) != 1 or command[command.index(flag)+1:command.index(flag)+2] != [value]:
            raise ValueError('Card handoff must preserve the completed minimal template checks')
    for name,arguments in LIMIT_RANGE_ARGUMENTS.items():
        if any(command_flag not in before[name]['command'] for command_flag in arguments):
            raise ValueError('Card handoff must preserve displayed-only limits and live recovery')
    if '--resume-cards' in command:
        # A later explicitly stopped card continuation may reuse its exact plan.
        if (command.count('--resume-cards') != 1 or command[-2:] != ['--card-workers','16']
                or command.count('--card-workers') != 1
                or any(flag in command for flag in ('--resume-written-templates','--resume-extracted'))):
            raise ValueError('Invalid existing native card-resume plan')
        expected=command
    else:
        expected=card_resume_command(command)
    if (after['grid']['command'] != expected
            or {k:v for k,v in before['grid'].items() if k != 'command'}
            != {k:v for k,v in after['grid'].items() if k != 'command'}
            or any(before[n] != after[n] for n in before if n != 'grid')):
        raise ValueError('Card handoff changed inputs, fit flags or another stage')


def resume_records(previous, contract, output, plots_only=False):
    """Explicit pre-fit recovery; keep successful stages and all old receipts."""
    require_canonical_plan_handoff(previous, contract['stages'])
    empty_preflight = (previous.get('status') == 'controller_error_children_retained'
                      and previous.get('stages') == {} and not previous.get('child_pids'))
    if ((previous.get('status') != 'needs_attention' and not empty_preflight) or previous.get('pid')
            or previous.get('child_pids')
            or any(v.get('pid') for v in previous.get('stages', {}).values())):
        raise RuntimeError('Resume requires a stopped products controller and stopped children')
    old = previous['contract']
    new_grid_command=contract['stages'].get('grid',{}).get('command',[])
    old_grid_command=old['stages'].get('grid',{}).get('command',[])
    if '--resume-cards' in new_grid_command:
        approved_card_resume_plan_migration(previous,contract)
    elif '--template-check-mode' in new_grid_command and new_grid_command != old_grid_command:
        approved_minimal_template_plan_migration(previous,contract)
    elif any('--plot-range-only' in contract['stages'].get(n,{}).get('command',[])
             and contract['stages'][n] != old['stages'].get(n) for n in ('limits','limit_recovery')):
        approved_limit_plot_range_plan_migration(previous,contract)
    for key in ('campaign', 'plan_sha256', 'grid_config_sha256'):
        if old[key] != contract[key]:
            if key != 'grid_config_sha256':
                raise ValueError('Resume changed production inputs: '+key)
            command=contract['stages']['grid'].get('command', [])
            if '--resume-written-templates' in command:
                approved_products_early_qcd_config_change(previous,contract,output)
                continue
            from TROTASR.workflows.build_nominal_grid import approved_zjet_config_change
            prior = output/'recovery_20261022/approved_zjet_fit_exclusion/prior_config.json'
            if not prior.is_file():
                raise ValueError('Missing approved fit-only recovery config snapshot')
            command = contract['stages']['grid']['command']
            current = Path(command[command.index('--config')+1])
            if (sha256(prior) != old[key] or sha256(current) != contract[key]
                    or previous['stages'].get('grid', {}).get('exit_code') != 2):
                raise ValueError('Missing stopped fit-only recovery provenance')
            approved_zjet_config_change(read_json(prior), read_json(current))
    if not empty_preflight and set(old['stages']) != set(contract['stages']):
        approved_plot_plan_migration(previous, contract)
    retained = {}
    for name, record in previous['stages'].items():
        spec = contract['stages'][name]
        is_plot = command_entrypoint(spec['command'])[0].name.startswith('plotting_') or '--plot-only' in spec['command'] if plots_only else False
        if plots_only and record['status'] != 'succeeded' and not is_plot:
            retained[name] = dict(record, retained_for_finite_recovery=True)
            continue
        if record['status'] == 'failed' and spec['lane'] in ('fit', 'benchmark'):
            raise ValueError('Use the existing finite fit recovery, not a stage reset: '+name)
        if record['status'] != 'succeeded':
            continue
        if spec != old['stages'][name]:
            raise ValueError('Cannot change an already successful stage: '+name)
        passed, _ = verify_receipt(spec, record.get('exit_code', 0))
        if not passed:
            raise ValueError('Completed stage output no longer matches: '+name)
        retained[name] = dict(record, reused_unchanged=True)
    return retained


def run(campaign, output, config, refresh_year=None, worker_budget=None, resume_failed=False, plots_only=False,
        prioritize_run3_plots=False, template_verify_workers=None, resume_written_templates=False,
        template_check_mode=None, limit_plot_range_only=False, card_workers=None, resume_cards=False):
    if sys.platform != 'linux':
        raise RuntimeError('Full production and native ROOT outputs remain on hep2')
    campaign, output, config = map(internal_path, (campaign, output, config))
    if canonical_limit_output(output) and (output/'products_state.json').exists():
        require_canonical_plan_handoff(read_json(output/'products_state.json'),
            {'limits':dict(canonical_limits_pipeline=True)})
    output, config, handoff = resolve_products_handoff(campaign, output, config)
    jme=read_json(config).get('fit_campaign')=='jme_weights_preliminary'
    from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
    run3=require_run3_systematics_policy(read_json(config))
    canonical_limits = run3 and canonical_limit_output(output)
    if resume_cards or card_workers is not None:
        if not (run3 and resume_failed and prioritize_run3_plots and limit_plot_range_only
                and resume_cards and card_workers == 16 and template_verify_workers == 12
                and template_check_mode == 'minimal' and not plots_only and not resume_written_templates):
            raise ValueError('Card handoff requires the explicit stopped Run-3 sixteen-worker minimal-grid resume')
    if (template_verify_workers is not None or resume_written_templates) and not (run3 and resume_failed):
        raise ValueError('Parallel written-template handoff requires explicit stopped Run-3 recovery')
    if resume_written_templates and (template_verify_workers != 12 or plots_only):
        raise ValueError('Written-template handoff requires twelve readers and is not a plots-only resume')
    if template_check_mode is not None and (template_check_mode != 'minimal' or not (resume_written_templates or resume_cards)):
        raise ValueError('Only explicit minimal checking on the approved written-template recovery is allowed')
    if limit_plot_range_only and not (run3 and (resume_failed or canonical_limits)):
        raise ValueError('Displayed-range amendment requires explicit stopped Run-3 recovery')
    combined=read_json(config).get('fit_campaign')=='combined_systematics_preliminary' or run3
    output.mkdir(parents=True, exist_ok=True)
    with (output/'products.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = stage_plan(campaign, output, config,worker_budget,prioritize_run3_plots,limit_plot_range_only)
        if template_verify_workers is not None:
            if not 1 <= template_verify_workers <= 12:
                raise ValueError('Template readback worker count must be between one and twelve')
            plan['grid']['command'].extend(['--template-verify-workers',str(template_verify_workers)])
        if resume_written_templates or resume_cards:
            plan['grid']['command'].append('--resume-written-templates')
        if template_check_mode is not None:
            plan['grid']['command'].extend(['--template-check-mode',template_check_mode])
        if resume_cards:
            plan['grid']['command']=card_resume_command(plan['grid']['command'],card_workers)
        resume_failed = resume_failed or plots_only
        contract = dict(campaign=str(campaign.relative_to(ROOT)), plan_sha256=sha256(campaign/'plan.json'),
            grid_config_sha256=sha256(config), code_sha256=sha256(Path(__file__)), stages=plan)
        if prioritize_run3_plots:
            contract['approved_plot_priority'] = dict(id=RUN3_PLOT_PRIORITY,user_approved=True,
                authority='Explicit Oct27 user approval to complete the remaining thirty-nine AN figures',
                plot_groups=list(RUN3_PLOT_NAMES),max_parallel_plot_groups=4,
                aggregate_slots=120,memory_reserve_gib=32,emergency_memory_gib=24,
                completed_stage_specs_preserved=True,canonical_worker_wrapper=True)
        if template_check_mode is not None:
            contract['approved_template_check_mode']=dict(id=MINIMAL_TEMPLATE_CHECK,user_approved=True,
                authority='Explicit Oct27 user instruction to replace exhaustive readback with minimal ROOT checks',
                check_mode=template_check_mode,numeric_readback_requested=False,
                preserve_existing_exhaustive_evidence=True)
        if limit_plot_range_only:
            contract['approved_limit_plot_range']=dict(id=DISPLAY_LIMIT_RANGE,user_approved=True,
                authority='Explicit user instruction to fit only the displayed limit window',
                mstop_gev=[600,1800],mlsp_gev=[1,1800],expected_display_points=895,
                native_grid_points=955,excluded_new_fit_points=60,native_grid_preserved=True,
                live_finite_recovery=True,contour_live_snapshot=True,
                impacts_wait_for_limit_pair=True,owned_slot_budget=worker_budget or 98)
        if resume_cards:
            contract['approved_card_resume']=dict(id=PARALLEL_CARD_RESUME,user_approved=True,
                authority='Explicit Oct27 user approval to parallelize only remaining native cards',
                card_workers=16,reuse_existing_cards_required=True,preserve_templates_and_check_evidence=True,
                physics_config_unchanged=True,completed_stage_specs_preserved=True)
        path = output/'products_state.json'
        retained = {}
        if path.exists():
            previous = read_json(path)
            require_canonical_plan_handoff(previous, plan)
            if resume_failed:
                if not plots_only and not (resume_written_templates or resume_cards) and previous['stages'].get('grid', {}).get('status') != 'succeeded':
                    plan['grid']['command'].append('--resume-extracted')
                retained = resume_records(previous, contract, output, plots_only)
                if resume_written_templates and previous['contract']['grid_config_sha256'] != contract['grid_config_sha256']:
                    approval_path=output/'grid/written_template_resume_approval.json'
                    contract['approved_grid_config_change']=dict(
                        id='early_2022_sr_qcd_fit_only_20261027',user_approved=True,
                        approval=str(approval_path.relative_to(ROOT)),approval_sha256=sha256(approval_path),
                        prior_config_sha256=previous['contract']['grid_config_sha256'],
                        current_config_sha256=contract['grid_config_sha256'],
                        only_hash_pinned_policy_reference_added=True)
                history = output/'history'/('products_resume_'+time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
                history.mkdir(parents=True, exist_ok=False)
                write_json(history/'products_state.json', previous)
                for name, record in previous['stages'].items():
                    if name not in retained and record.get('log'):
                        source = internal_path(ROOT/record['log'])
                        if source.exists():
                            source.rename(history/source.name)
            elif not products_contract_matches(previous['contract'], contract):
                raise ValueError('Divergent products controller contract; preserve prior run')
            elif previous.get('stages'):
                if previous.get('status') in ('generated_pending_visual_QA_and_inventory', 'fits_complete_pending_plots', 'needs_attention'):
                    print(previous['status']); return 0 if not previous['failed_stages'] else 2
                raise RuntimeError('Interrupted controller retained; inspect child PIDs before explicit recovery')
        elif resume_failed:
            raise ValueError('No failed products run to resume')
        state = dict(status='waiting_for_full_measurements', pid=os.getpid(), started=time.time(),
            contract=contract, stages=retained, full_workflow_complete=False, an_inventory_entries_delivered=0)
        if handoff:
            state['products_handoff'] = handoff
        if resume_failed:
            state['resumed_from'] = str((history/'products_state.json').relative_to(ROOT))
        active = {}
        def save():
            state['updated'] = time.time(); write_json(path, state)
        try:
            save()
            while not upstream_gate(campaign,jme=jme,combined=combined):
                save(); time.sleep(30)
            state['verified_inputs'] = {} if combined else verify_measurements(campaign, config)
            if run3:
                state['verified_inputs'].update(verify_measurements(campaign,config,years=('2024','2025')))
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
                if (snapshot['diagnostic_tf_gaps'] and 'transfer_factors_'+year in plan
                        and '--allow-partial' not in plan['transfer_factors_'+year]['command']):
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
                for name, spec in stage_order(plan):
                    if name in state['stages']:
                        continue
                    choice = disposition(name, spec, state['stages'])
                    if choice in ('blocked', 'not_needed'):
                        state['stages'][name] = dict(status=choice, dependencies=spec['dependencies'],
                            reason=spec.get('execution_blocked_reason')); save()
                        continue
                    if choice != 'ready' or not lane_available(plan, active, spec):
                        continue
                    if (limit_plot_range_only or canonical_limits) and not limit_priority_admission(name,plan,active,state['stages'],worker_budget):
                        continue
                    if spec['lane']=='benchmark' and any(s['lane']=='fit' and
                            state['stages'].get(n,{}).get('status') not in ('succeeded','failed','blocked','not_needed')
                            for n,s in plan.items()):
                        continue
                    if spec['lane']=='plot' and (ROOT/spec['receipt']).exists():
                        raise FileExistsError('Unregistered plot receipt retained; verify explicit adoption before launch: '+name)
                    # Serialize admission with diagnostic controllers; leave
                    # capacity for a native child's own subprocess as well.
                    with (ROOT/'stats/fit_admission.lock').open('a') as admission:
                        fcntl.flock(admission, fcntl.LOCK_EX)
                        slots, memory = live_resources()
                        active_plots=sum(plan[n]['lane']=='plot' for n in active)
                        prioritized_resources=prioritize_run3_plots or canonical_limits
                        free_disk=shutil.disk_usage(ROOT).free if prioritized_resources else None
                        if not resource_admission(slots,memory,active_plots,prioritized_resources,free_disk):
                            if prioritized_resources:
                                state['admission_wait']=dict(slots=slots,memory_available_bytes=memory,
                                    disk_free_bytes=free_disk,active_plot_groups=active_plots,
                                    emergency_memory=memory < 24*1024**3)
                                save()
                            continue
                        logs = output/'logs'; logs.mkdir(exist_ok=True)
                        logfile = logs/(name+'.log')
                        if logfile.exists():
                            raise FileExistsError('Unregistered stage log retained: '+name)
                        stream = logfile.open('x')
                        command=worker_command(spec['command']) if prioritize_run3_plots or spec.get('canonical_limits_pipeline') else spec['command']
                        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                        active[name] = (process, stream)
                        identity=None
                        if limit_plot_range_only:
                            from TROTASR.workflows.continue_limit_fits import assist_identity
                            identity=assist_identity(process.pid)
                        state['stages'][name] = dict(status='running', pid=process.pid, started=time.time(),
                            log=str(logfile.relative_to(ROOT)), command=command,
                            entrypoint_sha256=sha256(command_entrypoint(command)[0]))
                        if identity is not None:
                            state['stages'][name]['start_ticks']=identity['start_ticks']
                        save()
                if active or len(state['stages']) < len(plan):
                    time.sleep(5)
            failed = [n for n, r in state['stages'].items() if r['status'] in ('failed', 'blocked')]
            # Keep original impact failure in history even when bounded recovery
            # succeeded. Exclude only that recovered stage from remaining work.
            unresolved = [n for n in failed if not (
                n == 'impacts' and state['stages']['impact_recovery']['status'] == 'succeeded' or
                n == 'limits' and state['stages'].get('limit_recovery',{}).get('status') == 'succeeded' or
                state['stages'].get(n+'_recovery',{}).get('status')=='succeeded')]
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


def failed_fit_plan(output, worker_budget=None):
    """Canonical finite recovery only; never rebuild completed inputs or fits."""
    limit_workers, impact_workers = 6, 14
    if worker_budget is not None:
        if not 10 <= worker_budget <= 98:
            raise ValueError('Worker budget must be between 10 and 98')
        available = max(6, worker_budget-8)
        limit_workers = min(6, max(1, available//6))
        impact_workers = min(14, max(1, (available-limit_workers)//5))
    manifest=output/'grid/manifest.json'
    specs={}
    def add(name,script,args,receipt,statuses,after=(),partial=False):
        specs[name]=dict(command=[sys.executable,'-u',str(ROOT/'workflows'/script),*map(str,args)],
            receipt=str(receipt.relative_to(ROOT)),statuses=statuses,dependencies=list(after),partial=partial)
    canonical_limits = canonical_limit_output(output)
    if canonical_limits:
        add('limits','combine_limits.py',['--manifest',manifest,'--output',output/'limits',
            '--plot-range-only','--resume','--workers',115],
            output/'limits/workflow_state.json',['fits_complete_pending_plots'])
        specs['limits']['canonical_limits_pipeline']=True
    else:
        add('limits','reconcile_limit_fits.py',['--manifest',manifest,'--output',output/'limits',
            '--recover-remaining','--workers',limit_workers],output/'limits/numerical_remaining_20261009/state.json',
            ['all_verified_pending_reconciliation'])
    benchmarks=[('T2tt','mStop1200_mLSP500',output/'impacts')]
    benchmarks += [(m,p,output/'impacts_benchmarks_20261001'/(m+'_'+p))
        for m in ('T2bW','T2tb') for p in ('mStop1200_mLSP500','mStop1000_mLSP800')]
    for model,mass,folder in benchmarks:
        name='impacts_'+model+'_'+mass
        add(name,'recover_impacts.py',['--manifest',manifest,'--work',folder,'--model',model,
            '--mass',mass,'--complete-bounded-sgamma','--workers',impact_workers],
            folder/'impact_status.json',['fits_complete','generated_pending_visual_QA'])
        add(name+'_plot','combine_impacts.py',['--manifest',manifest,'--output',folder,
            '--model',model,'--mass',mass,'--plot-only'],folder/'impact_status.json',
            ['generated_pending_visual_QA'],after=[name])
    add('cronly','combine_cronly.py',['--manifest',manifest,'--output',output/'cronly','--recover-failed'],
        output/'cronly/state.json',['fits_complete_pending_plots','generated_pending_visual_QA'])
    add('cronly_plot','combine_cronly.py',['--manifest',manifest,'--output',output/'cronly','--plot-only'],
        output/'cronly/state.json',['generated_pending_visual_QA'],after=['cronly'])
    # The original contour snapshot is kept. This directory is a new fit-recovery snapshot.
    add('contours','plotting_limit_contour.py',['--limits',output/'limits','--grid',manifest,
        '--output',output/'plots/contours_recovery_20261009','--allow-partial'],
        output/'plots/contours_recovery_20261009/plot_manifest.json',['generated_pending_visual_QA'],
        after=['limits'],partial=not canonical_limits)
    return specs


def recover_failed_fits(output, worker_budget=None):
    if sys.platform!='linux':raise RuntimeError('Native fits remain on hep2')
    output=internal_path(output)
    with (output/'products.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=read_json(output/'products_state.json')
        if parent.get('pid') or any(v.get('pid') for v in parent.get('stages',{}).values()):
            raise RuntimeError('Existing products controller or children still active')
        plan=failed_fit_plan(output, worker_budget)
        require_canonical_plan_handoff(parent, plan)
        path=output/'fit_recovery_state.json'
        contract=dict(original_state_sha256=sha256(output/'products_state.json'),
            manifest_sha256=sha256(output/'grid/manifest.json'),stages=plan,
            code={str(p.relative_to(ROOT)):sha256(p) for p in
                  [Path(__file__),*[Path(s['command'][2]) for s in plan.values()]]})
        if path.exists():
            previous=read_json(path)
            if not products_contract_matches(previous['contract'], contract):
                raise ValueError('Recovery contract changed')
            if previous.get('pid') or previous.get('child_pids'):
                raise RuntimeError('Inspect retained recovery children before resuming')
            if previous['status'] in ('needs_attention','generated_pending_visual_QA'):
                return 2 if previous['failed_stages'] else 0
            raise RuntimeError('Interrupted recovery requires explicit inspection')
        state=dict(status='running',pid=os.getpid(),started=time.time(),contract=contract,stages={})
        active={}
        def save():
            state['updated']=time.time();write_json(path,state)
        try:
            save()
            while len(state['stages'])<len(plan) or active:
                for name,(proc,stream) in list(active.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    stream.close();del active[name]
                    record=state['stages'][name]
                    try:
                        passed,receipt=verify_receipt(plan[name],rc)
                        record.update(status='succeeded' if passed else 'failed',receipt_snapshot=receipt)
                    except Exception as error:record.update(status='failed',error=repr(error))
                    record.update(exit_code=rc,pid=None,finished=time.time());save()
                for name,spec in plan.items():
                    if name in state['stages']:continue
                    parents=[state['stages'].get(p,{}).get('status') for p in spec['dependencies']]
                    if any(p not in ('succeeded','failed','blocked') for p in parents):continue
                    if not spec['partial'] and any(p!='succeeded' for p in parents):
                        state['stages'][name]=dict(status='blocked',dependencies=spec['dependencies']);save();continue
                    if not spec['dependencies'] and name!='limits':
                        passed,receipt=verify_receipt(spec,0)
                        if passed:
                            state['stages'][name]=dict(status='succeeded',reused_unchanged=True,receipt_snapshot=receipt)
                            save();continue
                    with (ROOT/'stats/fit_admission.lock').open('a') as admission:
                        fcntl.flock(admission,fcntl.LOCK_EX)
                        slots,memory=live_resources()
                        if slots+2>(120 if canonical_limit_output(output) else 100) or memory<36*1024**3:continue
                        logfile=output/'logs'/('recovery_20261009_'+name+'.log')
                        stream=logfile.open('x')
                        command=worker_command(spec['command']) if spec.get('canonical_limits_pipeline') else spec['command']
                        proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**THREAD_ENV),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        active[name]=(proc,stream)
                        state['stages'][name]=dict(status='running',pid=proc.pid,started=time.time(),
                            command=command,log=str(logfile.relative_to(ROOT)))
                        save()
                if active or len(state['stages'])<len(plan):time.sleep(5)
            failed=[n for n,v in state['stages'].items() if v['status'] in ('failed','blocked')]
            state.update(status='needs_attention' if failed else 'generated_pending_visual_QA',
                failed_stages=failed,finished=time.time())
            return 2 if failed else 0
        except BaseException as error:
            state.update(status='controller_error_children_retained',error=repr(error),
                child_pids={n:p.pid for n,(p,_) in active.items()})
            raise
        finally:
            state['pid']=None;save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'stats/nominal_grid_config.json')
    parser.add_argument('--refresh-year',type=int,choices=[2025])
    parser.add_argument('--worker-budget',type=int,help='Worker allowance for this products chain; shared live admission still applies')
    parser.add_argument('--resume-failed',action='store_true',
        help='Resume stopped pre-fit stages after repair, retaining completed measurements and plots')
    parser.add_argument('--recover-failed-fits',action='store_true',
        help='Recover only unresolved limits, all benchmark impacts and CR fits, then render')
    parser.add_argument('--resume-plots',action='store_true',
        help='Retry failed renderers only; retain all measurements, grids and fit records')
    parser.add_argument('--prioritize-run3-plots',action='store_true',
        help='Explicit Oct27 approved fifteen-group plot-priority amendment, four renderers maximum')
    parser.add_argument('--template-verify-workers',type=int,choices=range(1,13),
        help='Explicit stopped Run-3 grid recovery only: native readback worker count')
    grid_resume=parser.add_mutually_exclusive_group()
    grid_resume.add_argument('--resume-written-templates',action='store_true',
        help='Explicit twelve-reader grid handoff; preserve written ROOT and replace --resume-extracted')
    grid_resume.add_argument('--resume-cards',action='store_true',
        help='Explicit stopped Run-3 card handoff; reuse templates/checks/cards and replace written-template resume')
    parser.add_argument('--card-workers',type=int,choices=(16,),
        help='Explicit approved sixteen-worker native card continuation only')
    parser.add_argument('--template-check-mode',choices=('minimal',),
        help='Explicit approved minimal ROOT checks on a stopped written-template recovery; omit for unchanged default')
    parser.add_argument('--limit-plot-range-only',action='store_true',
        help='Explicit 895-point display-window filter for unstarted Run-3 limits and finite recovery only')
    args = parser.parse_args()
    if args.recover_failed_fits:
        raise SystemExit(recover_failed_fits(args.output,args.worker_budget))
    raise SystemExit(run(args.campaign,args.output,args.config,args.refresh_year,args.worker_budget,
        args.resume_failed,args.resume_plots,args.prioritize_run3_plots,args.template_verify_workers,args.resume_written_templates,
        args.template_check_mode,args.limit_plot_range_only,args.card_workers,args.resume_cards))
