"""Expected-limit group-freeze diagnostics, invoked by combine_limits.py.

Original cards/templates/workspaces and fit states are read-only. These are
conditional, non-additive improvements, not a variance decomposition. Combined
autoMCStats parameters must not be mislabelled as separate signal/background.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import math
import os
from pathlib import Path
import re
import statistics
import time

from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

LABELS = {
    'mc_statistics': 'MC statistics (signal + background)',
    'background_transfer': 'Background transfer / nonclosure',
    'btag': 'b tagging', 'topw_tagging': 'Top / W tagging',
    'lepton_photon': 'Lepton / photon corrections',
    'jet_met': 'Jet / MET corrections', 'trigger': 'Trigger',
    'pileup_lumi': 'Pileup / luminosity', 'signal_theory': 'Signal cross section',
    'cr_normalizations': 'CR-constrained background normalizations',
    'all_constrained': 'All constrained nuisances (CR norms free)',
}
COMPETING = tuple(k for k in LABELS if k != 'all_constrained')
SOURCES = ('workflows/sensitivity_map.py', 'workflows/combine_limits.py',
           'workflows/limit_validation.py', 'workflows/limit_serialization.py',
           'workflows/diagnostic_common.py')


def nuisance_groups(text):
    groups = {key: [] for key in LABELS if key not in ('mc_statistics', 'all_constrained')}
    rates = set()
    for line in text.splitlines():
        t = line.split()
        if len(t) < 2 or t[0].startswith('#'):
            continue
        n, kind = t[:2]
        if kind == 'rateParam':
            rates.add(n)
            continue
        if kind not in ('shape', 'lnN', 'param', 'gmN'):
            continue
        if 'RZstat_' in n or 'zgammaNonclosure_' in n:
            group = 'background_transfer'
        elif 'btag_' in n:
            group = 'btag'
        elif '_eff_top_' in n or '_eff_w_' in n:
            group = 'topw_tagging'
        elif '_trigger_' in n:
            group = 'trigger'
        elif 'cross_section_stop' in n:
            group = 'signal_theory'
        elif n.startswith('lumi_') or '_pileup_' in n:
            group = 'pileup_lumi'
        elif '_scale_j_' in n or '_res_j_' in n or '_scale_met_' in n:
            group = 'jet_met'
        elif any(s in n for s in ('_eff_e_', '_eff_m_', '_eff_g_', '_scale_e_',
                                  '_scale_m_', '_scale_g_', '_scale_t_', '_res_e_',
                                  '_res_m_', '_res_g_')):
            group = 'lepton_photon'
        else:
            raise ValueError('Unclassified nuisance: ' + n)
        groups[group].append(n)
    groups['cr_normalizations'] = sorted(rates)
    return {k: sorted(set(v)) for k, v in groups.items()}


def median_check(path, log, tolerance):
    """Use the existing median crossing tolerance and binary32 reconciliation.

--newExpected 0 changes only unused expected bands, not the median search.
Never expose those approximate bands as validated results.
"""
    import uproot
    from TROTASR.workflows.limit_validation import SCAN, REL_ACC, ABS_ACC, asimov_diagnostics
    from TROTASR.workflows.limit_serialization import rounding_interval
    try:
        with uproot.open(path) as f:
            a = f['limit'].arrays(['limit', 'quantileExpected'], library='np')
        values = [float(v) for q, v in zip(a['quantileExpected'], a['limit']) if abs(float(q)-.5) < 1e-5]
        if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
            raise ValueError('Missing positive finite unique median')
        value = values[0]
        text = log.read_text(errors='replace')
        blocks = text.split('Will search for NLL crossing by bisection')[1:]
        if len(blocks) != 1:
            raise ValueError('Expected one actual median crossing search')
        scans = [(float(r), float(n)) for r, n in SCAN.findall(blocks[0])]
        target = .5 * statistics.NormalDist().inv_cdf(.975)**2
        try:
            lo, hi = rounding_interval(value)
        except ValueError:
            lo, hi = value-2e-6, value+2e-6
        close = [(r, n) for r, n in scans if lo <= r <= hi and abs(n-target) <= .05*tolerance+1e-5]
        below = [r for r, n in scans if n <= target and r <= value+2e-6]
        above = [r for r, n in scans if n >= target and r >= value-2e-6]
        width = min(above)-max(below) if below and above else None
        bracket = width is not None and 0 <= width <= 2*(max(REL_ACC*value, ABS_ACC)+2e-6)
        return bool(close or bracket), dict(expected=value, target_delta_nll=target,
            measured_rows=len(scans), matching_rows=close, bracket_width=width,
            asimov=asimov_diagnostics(text), provided_quantiles=[.5], unused_bands_not_validated=True)
    except Exception as exc:
        return False, dict(reason=str(exc))


def saved_baseline(source, limits):
    """Revalidate actual accepted artifacts, independent of retired driver copies.

Old source archives are useful provenance, but are not a substitute for checking
the saved ROOT, exact attempt log, command exit, and measured crossings today.
"""
    from TROTASR.workflows.combine_limits import IntegrityCache, verify_card
    from TROTASR.workflows.limit_serialization import validate_saved_limit
    from TROTASR.workflows.diagnostic_common import command_key
    cache=IntegrityCache();results={};evidence={}
    recovery_names=('numerical_all_unfinished_20260930','numerical_remaining_20261009',
                    'numerical_targeted_20261009','numerical_minima_20261009')
    for point in source['points']:
        verify_card(point,cache)
        key=point['model']+'/'+point['mass']
        state=read_json(limits/key/'state.json')
        if state.get('pid') or state['status'] not in ('complete','failed'):
            raise ValueError('Baseline not terminal: '+key)
        attempts=state.get('attempts',[])
        if state['status']=='complete':
            attempts=[a for a in attempts if a.get('output')==state['output']]
            if len(attempts)!=1:raise ValueError('Ambiguous accepted baseline')
        histories=[('primary',attempts)]
        if state['status']!='complete':
            for name in recovery_names:
                path=limits/name/key/'state.json'
                if path.exists():histories.append((name,read_json(path).get('attempts',[])))
        accepted=None
        for origin,attempts in histories:
            for attempt in attempts:
                if attempt.get('exit_code')!=0:continue
                log=internal_path(ROOT/attempt['log']);product=internal_path(ROOT/attempt['output'])
                if (product.parent!=log.parent or read_json(log.parent/'validation.json')!=attempt
                        or sha256(log)!=attempt['log_sha256'] or cache(product)!=attempt['output_sha256']):
                    raise ValueError('Stored baseline attempt identity mismatch: '+key)
                if origin!='primary':
                    receipt=read_json(limits/origin/key/'command_receipts/commands'/
                        (command_key(attempt['command'],log.parent)+'.json'))
                    if (receipt['status']!='complete' or receipt['exit_code']!=0 or receipt.get('pid')
                            or receipt['command']!=attempt['command']
                            or receipt['products'].get(attempt['output'])!=attempt['output_sha256']):
                        raise ValueError('Unverified baseline execution receipt')
                passed,validation=validate_saved_limit(product,log,attempt['tolerance'])
                if passed:
                    accepted=dict(status='complete',validation=validation,output=attempt['output'],
                        output_sha256=attempt['output_sha256'],log=attempt['log'],log_sha256=attempt['log_sha256'],
                        origin=origin,command=attempt['command'],exit_code=0)
                    break
            if accepted:break
        if not accepted:raise ValueError('No verified saved baseline: '+key)
        results[key]=accepted;evidence[key]=dict(accepted=accepted)
    return results,evidence


def prepare(manifest, output):
    from TROTASR.workflows.combine_limits import verify_combined_grid, IntegrityCache
    manifest, output = internal_path(manifest), internal_path(output)
    source = read_json(manifest)
    verify_combined_grid(source)
    if source['scope'] != 'full_combined_systematic_production':
        raise ValueError('This diagnostic is explicitly for the full-systematic high+low model')
    if output.parent != manifest.parent.parent or not output.name.startswith('sensitivity_map_'):
        raise ValueError('Keep diagnostics in the source campaign, outside original limits/grid')
    output.mkdir(parents=True, exist_ok=True)
    cp = output/'contract.json'
    code = {name: sha256(ROOT/name) for name in SOURCES}
    if cp.exists():
        old = read_json(cp)
        if old['manifest_sha256'] != sha256(manifest) or old['code'] != code:
            raise ValueError('Existing diagnostic contract changed; do not overwrite')
        return old, read_json(output/'baseline.json')
    limits = manifest.parent.parent/'limits'
    with (limits/'controller.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results, evidence = saved_baseline(source, limits)
    cache = IntegrityCache()
    points = []
    for point in source['points']:
        key = point['model']+'/'+point['mass']
        result = results[key]
        if result['status'] != 'complete':
            raise ValueError('Baseline missing a verified result: '+key)
        state = read_json(limits/key/'state.json')
        workspace = limits/key/'workspace.root'
        digest = cache(workspace)
        if digest != state['workspace_sha256']:
            raise ValueError('Original workspace changed: '+key)
        value = result['validation']['quantiles'].get(.5, result['validation']['quantiles'].get('0.5'))
        if not value or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid baseline median')
        groups = nuisance_groups((ROOT/point['card']).read_text())
        points.append(dict(point, key=key, expected=value, workspace=str(workspace.relative_to(ROOT)),
            workspace_sha256=digest, groups=groups, accepted=evidence[key]['accepted']))
    baseline = dict(points=points, original_states_modified=False, checked_at=time.time())
    write_json(output/'baseline.json', baseline)
    contract = dict(schema='trotasr_sensitivity_map_v1', manifest=str(manifest.relative_to(ROOT)),
        manifest_sha256=sha256(manifest), baseline_sha256=sha256(output/'baseline.json'), code=code,
        labels=LABELS, competing_groups=list(COMPETING), total_points=len(points),
        scope=source['systematic_scope'], sr_bins_per_year={'highdm':118, 'lowdm':30},
        sr_data_blinded=True, asimov='prefit background-only, --run blind', auto_mc_stats=[10,1,1],
        metric='100 * (1 - frozen_group_expected_median / baseline_expected_median)',
        interpretation='One group fixed at workspace initial values; other parameters profiled; effects are non-additive.',
        mc_statistics='Joint signal+background; shared Barlow-Beeston-lite nuisances are not separable by names.',
        cr_normalizations='Diagnostic ideal known-normalization comparison, not removal of a Gaussian systematic.',
        configurations=[dict(strategy=0,tolerance=.1),dict(strategy=1,tolerance=.01),dict(strategy=2,tolerance=.001)],
        rmax_policy='max(0.1,4*baseline median), bounded by 100000; strict bounds',
        unused_bands='newExpected=0 only skips nonmedian profile searches; approximate bands are not published',
        original_results_read_only=True, created=time.time())
    write_json(cp, contract)
    return contract, baseline


def command_for(point, group, config):
    from TROTASR.workflows.combine_limits import fit_command
    rmax = min(100000., max(.1, 4*point['expected']))
    command = fit_command(ROOT/point['workspace'], point['mass'], rmax, config['tolerance'])
    command[command.index('--cminDefaultMinimizerStrategy')+1] = str(config['strategy'])
    command += ['--strictBounds', '--newExpected', '0']
    if group == 'mc_statistics':
        command += ['--freezeNuisanceGroups', 'autoMCStats']
    elif group == 'all_constrained':
        command += ['--freezeParameters', 'allConstrainedNuisances']
    else:
        command += ['--freezeParameters', ','.join(point['groups'][group])]
    return command


def execute(point, group, output, contract, env):
    from TROTASR.workflows.diagnostic_common import CommandRunner
    dest = output/'fits'/point['key']/group
    statepath = dest/'state.json'
    if statepath.exists():
        state = read_json(statepath)
        if state['status'] in ('complete','no_effect'):
            for field in ('output','log'):
                if field in state and sha256(ROOT/state[field]) != state[field+'_sha256']:
                    raise ValueError('Saved diagnostic result changed')
            return state
        if state['status'] == 'failed':
            return state
    dest.mkdir(parents=True, exist_ok=True)
    state = dict(key=point['key'], group=group, status='running', attempts=[])
    if group not in ('mc_statistics','all_constrained') and not point['groups'][group]:
        state.update(status='no_effect', expected=point['expected'], improvement_percent=0.)
        write_json(statepath,state)
        return state
    if sha256(ROOT/point['workspace']) != point['workspace_sha256']:
        raise ValueError('Workspace changed before diagnostic execution')
    write_json(statepath, state)
    runner = CommandRunner(dest/'command_receipts')
    for index, config in enumerate(contract['configurations']):
        attempt = dest/('attempt_'+str(index+1))
        attempt.mkdir(exist_ok=True)
        log = attempt/'combine.log'
        product = attempt/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root')
        command = command_for(point, group, config)
        receipt = runner.run(command, attempt, log, [product], env)
        passed, validation = median_check(product, log, config['tolerance']) if receipt['exit_code']==0 else (False,dict(reason='nonzero_exit'))
        record = dict(config, command=command, exit_code=receipt['exit_code'], validation=validation,
            log=str(log.relative_to(ROOT)), log_sha256=sha256(log), output=str(product.relative_to(ROOT)),
            output_sha256=sha256(product) if product.exists() else None)
        state['attempts'].append(record)
        if passed:
            state.update(status='complete', expected=validation['expected'],
                improvement_percent=100*(1-validation['expected']/point['expected']),
                **{k:record[k] for k in ('log','log_sha256','output','output_sha256')})
            write_json(statepath,state)
            return state
        write_json(statepath,state)
    state.update(status='failed')
    write_json(statepath,state)
    return state


def collect(output, contract, baseline):
    rows = []; counts = dict(complete=0,failed=0,pending=0,running=0)
    for point in baseline['points']:
        values = {}; incomplete=[]
        for group in LABELS:
            path = output/'fits'/point['key']/group/'state.json'
            state = read_json(path) if path.exists() else dict(status='pending')
            status = state['status']
            if status in ('complete','no_effect'):
                counts['complete'] += 1
                values[group] = dict(status=status, expected=state['expected'], improvement_percent=state['improvement_percent'])
            else:
                counts[status] += 1
                values[group] = dict(status=status)
                if group in COMPETING: incomplete.append(group)
        winner = max(COMPETING,key=lambda k:values[k]['improvement_percent']) if not incomplete else None
        ms,ml = map(int,re.fullmatch(r'mStop(\d+)_mLSP(\d+)',point['mass']).groups())
        rows.append(dict(key=point['key'],model=point['model'],mStop=ms,mLSP=ml,
            baseline_expected=point['expected'], groups=values, dominant_group=winner,
            complete_competing_groups=not incomplete))
    result = dict(schema='trotasr_sensitivity_map_results_v1',contract=contract,
        updated=time.time(), counts=counts, total_fits=len(rows)*len(LABELS), points=rows,
        complete_points=sum(p['complete_competing_groups'] for p in rows),
        status='complete' if counts['complete']==len(rows)*len(LABELS) else 'partial')
    write_json(output/'results.json', result)
    return result


def run(manifest, output, workers, prepare_only=False, point_keys=None):
    from TROTASR.workflows.combine_limits import runtime
    output = internal_path(output)
    output.mkdir(parents=True,exist_ok=True)
    if not 1 <= workers <= 96: raise ValueError('Use 1-96 workers; account-wide admission is also enforced')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract,baseline = prepare(manifest,output)
        if prepare_only:
            print('Prepared',len(baseline['points']),'points',flush=True)
            return 0
        env=runtime(output/'runtime')
        points=baseline['points']
        if point_keys:
            unknown=set(point_keys)-{p['key'] for p in points}
            if unknown:raise ValueError('Unknown requested points: '+str(unknown))
            points=[p for p in points if p['key'] in point_keys]
        # Give each topology a useful first result, then interleave mass points.
        points=sorted(points,key=lambda p:(p['mass']!='mStop1200_mLSP500',p['mass'],p['model']))
        tasks=iter((p,g) for p in points for g in LABELS)
        futures={};exhausted=False;last=0.;errors=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while not exhausted or futures:
                while len(futures)<workers and not exhausted:
                    try:p,g=next(tasks)
                    except StopIteration:exhausted=True;break
                    future=pool.submit(execute,p,g,output,contract,env)
                    futures[future]=p['key']+'/'+g
                done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    key=futures.pop(f)
                    try:
                        state=f.result()
                        print(key,state['status'],state.get('expected'),flush=True)
                    except Exception as exc:
                        errors.append(dict(key=key,error=repr(exc)))
                        print(key,'error',repr(exc),flush=True)
                if time.time()-last>30 or not futures:
                    summary=collect(output,contract,baseline)
                    write_json(output/'state.json',dict(status='running' if futures or not exhausted else
                        ('complete' if summary['status']=='complete' else 'partial'),pid=os.getpid() if futures or not exhausted else None,
                        counts=summary['counts'],complete_points=summary['complete_points'],errors=errors,updated=time.time()))
                    last=time.time()
        return 0 if not errors else 1


def plot(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import mplhep as hep
    output=internal_path(output)
    result=collect(output,read_json(output/'contract.json'),read_json(output/'baseline.json'))
    dest=output/'plots';dest.mkdir(exist_ok=True)
    hep.style.use('CMS')
    artifacts={}
    colors=plt.get_cmap('tab10').colors
    for model in ('T2tt','T2bW','T2tb'):
        allrows=[p for p in result['points'] if p['model']==model]
        valid=[p for p in allrows if p['dominant_group'] is not None]
        if not valid:continue
        fig,ax=plt.subplots(figsize=(12,9))
        ax.scatter([p['mStop'] for p in allrows],[p['mLSP'] for p in allrows],c='.88',s=24,marker='s',label='Pending / incomplete')
        c=np.array([COMPETING.index(p['dominant_group']) for p in valid])
        ax.scatter([p['mStop'] for p in valid],[p['mLSP'] for p in valid],c=c,s=32,marker='s',
                   cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,10.5),10))
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],marker='s',linestyle='',color=colors[i],label=LABELS[g]) for i,g in enumerate(COMPETING)]
        handles.append(Line2D([],[],marker='s',linestyle='',color='.88',label='Pending / incomplete'))
        ax.legend(handles=handles,loc='upper left',fontsize=10,frameon=False,ncol=2)
        ax.set(xlabel=r'$m_{\widetilde{t}}$ (GeV)',ylabel=r'$m_{\widetilde{\chi}^0_1}$ (GeV)',
               xlim=(75,1850),ylim=(-25,2100),title=model+' | largest conditional improvement')
        hep.cms.label('Work in progress',data=False,lumi=220.66,com=13.6,ax=ax)
        fig.tight_layout()
        for ext in ('png','pdf'):
            name=dest/(model+'_dominant_group.'+ext);fig.savefig(name,dpi=180);artifacts[name.name]=sha256(name)
        plt.close(fig)
        fig,axes=plt.subplots(3,4,figsize=(19,13),sharex=True,sharey=True)
        for ax,group in zip(axes.flat,LABELS):
            selected=[p for p in allrows if 'improvement_percent' in p['groups'][group]]
            ax.scatter([p['mStop'] for p in allrows],[p['mLSP'] for p in allrows],c='.9',s=10,marker='s')
            if selected:
                sc=ax.scatter([p['mStop'] for p in selected],[p['mLSP'] for p in selected],
                    c=[p['groups'][group]['improvement_percent'] for p in selected],s=15,marker='s',cmap='coolwarm',vmin=-10,vmax=100)
            ax.set_title(LABELS[group],fontsize=11)
            ax.tick_params(labelsize=10)
        axes.flat[-1].axis('off')
        if selected or any('improvement_percent' in p['groups'][g] for p in allrows for g in LABELS):
            fig.colorbar(sc,ax=list(axes.flat),label='Expected median limit improvement (%)',shrink=.7,pad=.02)
        fig.suptitle(model+' | 2024+2025 high-dM + low-dM | conditional group freezes',fontsize=19)
        fig.supxlabel(r'$m_{\widetilde{t}}$ (GeV)');fig.supylabel(r'$m_{\widetilde{\chi}^0_1}$ (GeV)')
        for ext in ('png','pdf'):
            name=dest/(model+'_group_improvements.'+ext);fig.savefig(name,dpi=150);artifacts[name.name]=sha256(name)
        plt.close(fig)
    write_json(dest/'plot_manifest.json',dict(status='generated_pending_visual_QA',artifacts=artifacts,
        results_sha256=sha256(output/'results.json'),coverage=result['complete_points'],requested=955,
        interpolation='none; actual requested mass-point markers only',non_additive=True))
    return 0
