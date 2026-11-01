"""Card and accepted CR-fit projections; no event input or fit execution.

This retains the legacy exporter prescription: apply declared rateParam initial
values once, add independent MC variances, and sum same-name nuisance endpoint
deltas before taking their envelope. Free parameter ranges are NOT errors.
SR observations are never read, including the synthetic data_obs template.
"""
import fnmatch
from pathlib import Path
import numpy as np
from TROTASR.utils.paths import ROOT, internal_path, ERAS
from TROTASR.gnn4lowdm.configuration import configuration, configuration_path
from TROTASR.utils.io import read_json, sha256
from TROTASR.utils.sr_binning import expand, label
from TROTASR.utils.plot_adapter import renderer
from TROTASR.utils.statistical_templates import json_ready
from TROTASR.workflows.diagnostic_common import grid_card

DISPLAY = {'Top':'Top', 'WJet':'W -> lv', 'ZJet':'Z -> vv', 'QCD':'QCD Multijet',
           'PhotonJet':'Photon+jet', 'DYJet':'DY', 'VV':'VV+VVV'}
LOW_PROCESSES = {'VV':'VV', 'Top':'Top', 'DYJet':'DY', 'PhotonJet':'Photon',
                 'WJet':'W', 'ZJet':'Zinv', 'QCD':'QCD'}
LOW_SIGNALS = ('mStop800_mLSP650', 'mStop1000_mLSP850')

PREVIEW_SCOPE = 'single_era_nominal_prefit_preview'
ERA_VIEW_SCOPE = 'single_era_card_prefit_view'


def export_cronly_fit(result, output):
    """Project the accepted CR-only RooFit model without fitting or rebuilding.

    Evaluate the existing CMSHistErrorPropagator totals and official process
    wrappers directly: no repeated per-process histograms for every covariance
    draw, and no duplicate TotalBkg/TotalProcs calculation.
    """
    import fcntl
    import math
    import time
    import ROOT as root
    from TROTASR.utils.io import write_json
    from TROTASR.utils.era_merge import CHANNEL_BINS
    result, output = internal_path(result), internal_path(output)
    published = read_json(result/'publication.json')
    if (published.get('kind') != 'cronly' or published.get('complete') is not True
            or published.get('sr_observed_data_used') is not False
            or published.get('dataset') != 'observed CR-only B-only r0'):
        raise ValueError('Accepted observed CR-only result required')
    card = result.parent/'grid/datacards/datacard_cronly.txt'
    workspace, fit_path = result/'workspace.root', result/'fitDiagnostics_cronly.root'
    identity = {name:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                for name,p in (('workspace',workspace),('fit',fit_path),('card',card))}
    payload_path = output/'prediction_payload.json'
    if payload_path.is_file():
        saved = read_json(payload_path)
        if (saved.get('schema') == 'trotasr_cronly_prefit_postfit_v1'
                and saved.get('input_identity') == identity):
            return saved
        raise ValueError('Changed CR plot inputs; do not silently reuse a cached projection')
    output.mkdir(parents=True,exist_ok=True)
    started = time.time()
    with (result/'controller.lock').open('a') as owner, (output/'plot.lock').open('a') as lock:
        fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files = [root.TFile.Open(str(p)) for p in
                 (workspace,fit_path,card.parent/'era_merged_data_obs.root')]
        try:
            w, fit = files[0].Get('w'), files[1].Get('fit_b')
            if not fit or int(fit.status()) != 0 or not math.isfinite(float(fit.edm())):
                raise ValueError('Missing successful saved CR fit')
            fit_quality = dict(status=int(fit.status()),covariance_quality=int(fit.covQual()),
                               edm=float(fit.edm()),global_covariance_accepted=False)
            channels = [r+'_'+m+'_c1' for m in ('highdm','lowdm')
                        for r in ('LLCR','QCDCR','GCR')]
            category = w.cat('CMS_channel')
            if {str(c.first) for c in category} != set(channels):
                raise ValueError('Only the six observed CR channels may be projected')
            x = w.var('CMS_th1x')
            components, records, slices = {}, {}, {}
            offset = 0
            for channel in channels:
                n = CHANNEL_BINS[channel]
                prop = w.function('prop_bin'+channel)
                if not prop or prop.ClassName() != 'CMSHistErrorPropagator':
                    raise ValueError('Unexpected native CR histogram model')
                wrappers, coefs = prop.wrapperList(), prop.coefList()
                labels = [str(p.getStringAttribute('combine.process')) for p in wrappers]
                if len(labels) != len(coefs) or any(p.split('_',1)[0] not in DISPLAY for p in labels):
                    raise ValueError('Unclassified native CR process')
                components[channel] = (prop,wrappers,coefs,labels)
                data = files[2].Get(channel+'/data_obs')
                if not data or int(data.GetNbinsX()) != n:
                    raise ValueError('Changed approved CR observation bins')
                records[channel] = dict(
                    labels=[str(data.GetXaxis().GetBinLabel(i)) for i in range(1,n+1)],
                    data=[float(data.GetBinContent(i)) for i in range(1,n+1)])
                slices[channel] = slice(offset,offset+n);offset += n
            def totals():
                values = []
                for channel in channels:
                    prop = components[channel][0]
                    for i in range(CHANNEL_BINS[channel]):
                        x.setVal(i+.5);values.append(float(prop.getVal()))
                return np.asarray(values)
            def central(stage):
                total = totals()
                if not np.isfinite(total).all() or np.any(total <= 0):
                    raise ValueError('Invalid plotted CR expectation')
                for channel in channels:
                    prop, wrappers, coefs, labels = components[channel]
                    n = CHANNEL_BINS[channel]
                    groups = {g:np.zeros(n) for g in renderer.GROUP_ORDER}
                    for i in range(n):
                        x.setVal(i+.5)
                        for wrapper,coef,label in zip(wrappers,coefs,labels):
                            groups[DISPLAY[label.split('_',1)[0]]][i] += float(wrapper.getVal()*coef.getVal())
                    if not np.allclose(sum(groups.values()),total[slices[channel]],rtol=2e-9,atol=1e-6):
                        raise ValueError('CR process wrappers do not sum to the native total')
                    records[channel][stage] = dict(groups={k:v.tolist() for k,v in groups.items()},
                                                  background=total[slices[channel]].tolist())
                return total
            central('prefit')
            prior = list(files[1].Get('nuisances_prefit_res').floatParsFinal())
            variance = np.zeros(offset)
            prior_counts = dict(gaussian=0,poisson=0)
            for index,p in enumerate(prior):
                v = w.var(p.GetName());nominal = float(v.getVal())
                if v.hasError() and v.getError() > 0:
                    sigma = float(v.getError());prior_counts['gaussian'] += 1
                elif nominal > 0 and str(p.GetName()).startswith('prop_bin'):
                    # Count-valued autoMCStats auxiliary Poisson constraint:
                    # Var(N)=N. Bounds are never interpreted as uncertainties.
                    sigma = math.sqrt(nominal);prior_counts['poisson'] += 1
                else:
                    raise ValueError('Unspecified prior width: '+str(p.GetName()))
                v.setVal(max(nominal-sigma,float(v.getMin())) if v.hasMin() else nominal-sigma)
                down = totals()
                v.setVal(min(nominal+sigma,float(v.getMax())) if v.hasMax() else nominal+sigma)
                up = totals();v.setVal(nominal)
                variance += ((up-down)/2.)**2
                if (index+1)%64 == 0:
                    print('CR prefit nuisance propagation',index+1,'/',len(prior),flush=True)
            prefit_error = np.sqrt(variance)
            for p in fit.floatParsFinal():
                w.var(p.GetName()).setVal(p.getVal())
            central('postfit')
            root.RooRandom.randomGenerator().SetSeed(12345)
            randomized = list(fit.randomizePars())
            targets = [w.var(p.GetName()) for p in randomized]
            samples = np.empty((500,offset))
            for index in range(500):
                fit.randomizePars()
                for v,p in zip(targets,randomized):v.setVal(p.getVal())
                samples[index] = totals()
                if (index+1)%100 == 0:
                    print('CR postfit correlated covariance samples',index+1,'/ 500',flush=True)
            if not np.isfinite(samples).all():
                raise ValueError('Nonfinite CR covariance propagation')
            postfit_error = samples.std(axis=0,ddof=0)
            for channel in channels:
                records[channel]['prefit']['background_unc'] = prefit_error[slices[channel]].tolist()
                records[channel]['postfit']['background_unc'] = postfit_error[slices[channel]].tolist()
            payload = dict(schema='trotasr_cronly_prefit_postfit_v1',status='complete',
                dataset=published['dataset'],canonical_model='era_merged',years=list(ERAS),
                sr_observations_included=False,channels=records,input_identity=identity,
                fit=fit_quality,extracted=time.time(),fit_recalculated=False,
                uncertainty_method=dict(
                    prefit='native prior up/down propagation; Gaussian stored width, Poisson sqrt(N)',
                    postfit='saved RooFitResult full correlated covariance sampling'),
                prior_nuisances=prior_counts,covariance_samples=500,covariance_seed=12345,
                projection='native CMSHistErrorPropagator and CMSHistFuncWrapper',
                projection_seconds=time.time()-started)
            write_json(payload_path,payload)
            return payload
        finally:
            for f in files:
                if f:f.Close()


def select_era(payload, year):
    """Select exact existing card channels; never combine unlike era axes."""
    from copy import deepcopy
    from TROTASR.workflows.diagnostic_common import require_revised_record, record_years
    if not require_revised_record(payload):
        raise ValueError('Adopted statistical revision required')
    years = record_years(payload)
    if (year not in years or payload.get('status') != 'complete'
            or payload.get('prediction_stage') != 'prefit'
            or payload.get('rate_initials_applied') is not True
            or payload.get('sr_observations_included') is not False):
        raise ValueError('Verified current prefit card export required')
    expected = {year+'/'+region+'/'+mode for mode, regions in
                (('highdm', ('SR',)), ('lowdm', ('SR','LLCR','QCDCR','GCR'))) for region in regions}
    selected = {k:deepcopy(v) for k,v in payload['channels'].items() if k.startswith(year+'/')}
    if set(selected) != expected:
        raise ValueError('Missing/extra single-era card channels')
    for key, record in selected.items():
        if '/SR/' in key and (record.get('sr_data_blinded') is not True or 'data' in record):
            raise ValueError('Observed SR cannot enter a single-era view')
    parent = {k:deepcopy(v) for k,v in payload.items() if k not in ('channels','source_inputs')}
    result = deepcopy(parent)
    result.update(scope=ERA_VIEW_SCOPE, years=[year], bins=214, th1_channels=8,
        channels=selected, source_inputs={year:deepcopy(payload['source_inputs'][year])},
        gnn_sha256=sha256(configuration_path(year)), full_workflow_complete=False,
        single_era_view=dict(plot_only=True, fit_ready=False, source_model=parent))
    prediction_years(result)
    return result


def prediction_years(payload):
    """One-era previews are plot-only, never accepted by the fit-grid reader."""
    if payload.get('scope') == ERA_VIEW_SCOPE:
        from TROTASR.workflows.diagnostic_common import require_revised_record, record_years
        view = payload.get('single_era_view', {})
        parent = view.get('source_model', {})
        if not require_revised_record(parent):
            raise ValueError('Adopted parent statistical revision required')
        years = tuple(payload.get('years', ()))
        if (view.get('plot_only') is not True or view.get('fit_ready') is not False
                or len(years) != 1 or years[0] not in record_years(parent)
                or payload.get('bins') != 214 or payload.get('th1_channels') != 8
                or payload.get('full_workflow_complete') is not False
                or payload.get('gnn_sha256') != sha256(configuration_path(years[0]))
                or any(payload.get(k) != parent.get(k) for k in (
                    'provenance','statistical_revision','auto_mc_stats','signal_contamination',
                    'background_estimation_uncertainties','sr_merge','systematic_combination',
                    'uncertainty_audit','sr_data_blinded','sr_observations_included',
                    'prediction_stage','rate_initials_applied'))
                or payload.get('sr_observations_included') is not False
                or payload.get('sr_data_blinded') is not True):
            raise ValueError('Invalid exact single-era card view')
        return years
    if payload.get('scope') != PREVIEW_SCOPE:
        if 'single_era_preview' in payload or 'single_era_view' in payload:
            raise ValueError('Preview cannot be relabeled as a combined result')
        from TROTASR.workflows.diagnostic_common import record_years
        return record_years(payload)
    from TROTASR.utils.paths import EARLY_ERAS
    from TROTASR.workflows.build_nominal_grid import CR12_POLICY
    years = tuple(payload.get('years', ()))
    if (len(years) != 1 or years[0] not in EARLY_ERAS
            or payload.get('single_era_preview') != dict(plot_only=True, fit_ready=False,
                source_scope='full_nominal_production')
            or payload.get('statistical_revision') != CR12_POLICY
            or payload.get('bins') != 214 or payload.get('th1_channels') != 8
            or payload.get('sr_data_blinded') is not True
            or payload.get('sr_observations_included') is not False
            or payload.get('prediction_stage') != 'prefit'
            or payload.get('signal_contamination') is not False
            or payload.get('auto_mc_stats') != [10, 1, 1]
            or payload.get('full_workflow_complete') is not False
            or payload.get('background_estimation_uncertainties') != dict(
                id='jme_background_shapes_20260929', mode='shape',
                auto_mc_stats=[10, 1, 1], user_approved=True)):
        raise ValueError('Invalid single-era blinded nominal preview contract')
    return years


def export_single_era(config_path, year, output):
    """Exact existing statistical kernels and native TH1s, with no fit/grid claim.

    This does not introduce missing eras, alter SR/CR definitions, or read SR
    observations. Only the four original overlay points and the projection
    validation benchmark are extracted. The full production chain is untouched.
    """
    import sys
    import fcntl
    import ROOT as pyroot
    import uproot
    from TROTASR.utils.paths import EARLY_ERAS
    from TROTASR.utils.io import write_json
    from TROTASR.workflows.signal_grid_inputs import extract_year
    from TROTASR.workflows.build_nominal_grid import (
        CODE, CR12_POLICY, combine_cached, background_error_shapes, card_text,
        point_signals, put_signals, template_name, write_templates, verify_templates)
    from TROTASR.utils.sr_merge import load_merge
    config_path, output = internal_path(config_path), internal_path(output)
    config = read_json(config_path)
    if (sys.platform != 'linux' or year not in EARLY_ERAS
            or config.get('statistical_revision') != CR12_POLICY
            or config.get('systematic_evaluation_deferred') is not True
            or config.get('luminosity_uncertainties') != {}
            or config.get('fit_campaign') != 'early_run3_nominal_first'):
        raise ValueError('Only explicit early-era nominal-first production preview')
    paths = config['years'][year]
    contract = dict(config_sha256=sha256(config_path), year=year,
        inputs={k: dict(path=v, sha256=sha256(internal_path(v))) for k, v in paths.items()},
        code={p: sha256(ROOT/p) for p in (*CODE, 'workflows/prediction_plot_inputs.py')})
    output.mkdir(parents=True, exist_ok=True)
    with (output/'preview.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = output/'preview_manifest.json'
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if (previous['contract'] != contract or
                    any(sha256(output/n) != h for n, h in previous['files'].items())):
                raise ValueError('Changed single-era preview inputs/products')
            return read_json(output/'prediction_payload.json')
        if any(p.name != 'preview.lock' for p in output.iterdir()):
            raise FileExistsError('Retain incomplete native preview for diagnosis')
        required = [('T2tt', s['key']) for s in renderer.SIGNAL_OVERLAYS] + [('T2bW', m) for m in LOW_SIGNALS]
        grid = dict(models={'T2tt': [m for model,m in required if model=='T2tt'] + ['mStop1200_mLSP500'],
                            'T2bW': list(LOW_SIGNALS), 'T2tb': []})
        cache = extract_year(paths, grid, preserve_unsupported=True)
        if cache['year'] != year or cache['scope'] != 'full_nominal_production':
            raise ValueError('Wrong era or scope in full-input histograms')
        issues = []
        channels, background, clipping, sources = combine_cached({year: cache},
            sr_merge=config['sr_merge'], fit_issues=issues, statistical_revision=CR12_POLICY)
        background, shape_audit = background_error_shapes(background, config['background_estimation_uncertainties'])
        text = card_text(channels, background, statistical_revision=CR12_POLICY, luminosity_uncertainties={})
        card = parse_card(text)
        (output/'preview_background_card.txt').write_text(text)
        write_templates(output, channels, background)
        verify_templates(output, channels, background)
        filenames = sorted({template_name(c) for c in channels})
        files = {n: pyroot.TFile(str(output/n), 'UPDATE') for n in filenames}
        expected_signals = {}
        try:
            for model, mass in required:
                signals, signal_issues, signal_clipping = point_signals(
                    channels, sources, {year: cache}, model, mass, keep_empty=True)
                issues.extend(signal_issues); clipping.extend(signal_clipping)
                put_signals(files, channels, signals, 'signal_'+model+'_'+mass)
                expected_signals[model,mass] = signals
        finally:
            for root in files.values(): root.Close()
        merge = load_merge(config['sr_merge'])
        result = {}
        for mode, regions in (('highdm', ('SR',)), ('lowdm', ('SR','LLCR','QCDCR','GCR'))):
            for region in regions:
                channel = f'{region}_{mode}_c1_{year}'
                labels = expected_labels(channel, merge)
                if labels != [b['name'] for b in channels[channel]]:
                    raise ValueError('Preview native bin order changed')
                with uproot.open(output/template_name(channel), object_cache=None, array_cache=None) as root:
                    directory = root[channel]
                    record = channel_prediction(card, channel, directory, labels, (.01, 5))
                    if region == 'SR':
                        for model, mass in required:
                            if (model == 'T2tt') != (mode == 'highdm'): continue
                            values, variance = read_th1(directory, 'signal_'+model+'_'+mass, labels)
                            expected = expected_signals[model,mass][channel,'signal']
                            np.testing.assert_allclose(values, expected['nominal'], rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(variance, expected['sumw2'], rtol=1e-12, atol=1e-12)
                            record['signals'][mass] = values
                    result[year+'/'+region+'/'+mode] = record
        payload = dict(schema='trotasr_card_prediction_plot_v1', status='complete', scope=PREVIEW_SCOPE,
            years=[year], single_era_preview=dict(plot_only=True, fit_ready=False, source_scope=cache['scope']),
            prediction_stage='prefit', sr_observations_included=False, sr_data_blinded=True,
            rate_initials_applied=True, full_workflow_complete=False, signal_contamination=False,
            channels=json_ready(result), source_inputs={year: cache['inputs']}, provenance=contract,
            bins=214, th1_channels=8, auto_mc_stats=[10,1,1], statistical_revision=CR12_POLICY,
            background_estimation_uncertainties=config['background_estimation_uncertainties'],
            sr_merge=config['sr_merge'], sr_binning_sha256=sha256(ROOT/'jsons/highdm_sr_binning.json'),
            gnn_sha256=sha256(configuration_path(year)), fit_support_issues=issues,
            definition='Single-era nominal prefit preview; existing card initials, MC sumw2 and measured background errors. No observed SR or fit.')
        prediction_years(payload)
        write_json(output/'prediction_payload.json', payload)
        write_json(output/'projection_audit.json', json_ready(dict(
            projection=cache['projection_audit'], shapes=shape_audit, clipping=clipping, fit_support_issues=issues)))
        files = {p.name: sha256(p) for p in output.iterdir() if p.name != 'preview.lock'}
        write_json(receipt_path, dict(status='complete', contract=contract, files=files,
            sr_observations_included=False, fit_ready=False))
        return payload


def array(values, size, nonnegative=False):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or (nonnegative and (result < 0).any()):
        raise ValueError('Invalid prediction array')
    return result


def expected_labels(channel, merge=None):
    region, mode, packing, year = channel.split('_')
    if packing != 'c1' or year not in ERAS:
        raise ValueError('Unexpected native channel')
    if region == 'SR' and mode == 'highdm':
        return ['SR_highdm_bin'+str(i) for i in range(merge['bins_per_year'] if merge else 162)]
    if mode == 'lowdm':
        cats = (configuration(year)['sr_binning']['category_labels']
                if region == 'SR' else ['Nb1','Nb2'])
        return [f'{region}_lowdm_gnn_{c}_bin{i}' for c in cats for i in range(5)]
    raise ValueError('Unsupported plot channel')


def parse_card(text):
    rows = [line.split('#',1)[0].split() for line in text.splitlines()]
    rows = [r for r in rows if r]
    processes = [r[1:] for r in rows if r[0] == 'process']
    if len(processes) != 2:
        raise ValueError('Exactly two process rows required')
    names, ids = processes
    channel_rows = [r[1:] for r in rows if r[0] == 'bin']
    if len(channel_rows) != 2 or len(channel_rows[1]) != len(names) or len(ids) != len(names):
        raise ValueError('Invalid card columns')
    columns = list(zip(channel_rows[1], names))
    if len(set(columns)) != len(columns):
        raise ValueError('Duplicate card columns')
    rates = next(r[1:] for r in rows if r[0] == 'rate')
    if rates != ['-1'] * len(columns):
        raise ValueError('Only native TH1 integral rates are supported')
    nuisances = [r for r in rows if len(r)>2 and r[1] in ('shape','lnN')]
    if any(len(r) != len(columns)+2 for r in nuisances):
        raise ValueError('Nuisance column count mismatch')
    if len({r[0] for r in nuisances}) != len(nuisances):
        raise ValueError('Duplicate constrained nuisance')
    allowed = {'imax','jmax','kmax','------------','shapes','bin','observation','process','rate'}
    for row in rows:
        if row[0] not in allowed and not (len(row)>1 and row[1] in ('shape','lnN','rateParam','autoMCStats')):
            raise ValueError('Unsupported card directive: '+row[0])
    if ['*','autoMCStats','10','1','1'] not in rows:
        raise ValueError('Unexpected MC statistics prescription')
    return dict(columns=columns, nuisances=nuisances,
                rateparams=[r for r in rows if len(r)>1 and r[1]=='rateParam'],
                shapes=[r for r in rows if r[0]=='shapes'])


def read_th1(directory, name, labels):
    h = directory[name]
    if not h.classname.startswith('TH1') or h.axis().labels() != labels or h.variances() is None:
        raise ValueError('TH1 type/label/variance mismatch: '+name)
    return array(h.values(),len(labels),True), array(h.variances(),len(labels),True)


def channel_prediction(card, channel, directory, labels, rate_range=(0, 10)):
    if tuple(rate_range) not in ((0, 10), (.01, 5)):
        raise ValueError('Unapproved prediction normalization range')
    range_text = '[0,10]' if tuple(rate_range) == (0, 10) else '[0.01,5]'
    size = len(labels)
    result = dict(labels=list(labels), sr_data_blinded=channel.startswith('SR_'),
        background=np.zeros(size), variance=np.zeros(size), groups={}, group_variances={},
        nuisance_deltas={}, rateparam_initials={}, signals={})
    if not result['sr_data_blinded']:
        result['data'], result['data_variance'] = read_th1(directory,'data_obs',labels)
    for i,(c,process) in enumerate(card['columns']):
        if c != channel or process == 'signal':
            continue
        family = process.split('_',1)[0]
        if family not in DISPLAY:
            raise ValueError('Unclassified native process '+process)
        initials = {}
        for row in card['rateparams']:
            if len(row) != 6 or row[5] != range_text:
                raise ValueError('Unsupported rate parameter expression/range')
            if fnmatch.fnmatch(c,row[2]) and fnmatch.fnmatch(process,row[3]):
                if row[0] in initials:
                    raise ValueError('Duplicate rate parameter application')
                initial = float(row[4])
                if not np.isfinite(initial) or not rate_range[0] <= initial <= rate_range[1]:
                    raise ValueError('Invalid rate initial value')
                initials[row[0]] = initial
                if row[0] in result['rateparam_initials'] and result['rateparam_initials'][row[0]] != initial:
                    raise ValueError('Inconsistent shared initial')
                result['rateparam_initials'][row[0]] = initial
        scale = float(np.prod(list(initials.values())))
        values, variance = read_th1(directory,process,labels)
        nominal = values*scale
        result['background'] += nominal
        result['variance'] += variance*scale**2
        result['groups'].setdefault(family,np.zeros(size))[:] += nominal
        result['group_variances'].setdefault(family,np.zeros(size))[:] += variance*scale**2
        for row in card['nuisances']:
            name,kind = row[:2]; strength = row[2+i]
            if strength in ('-','0'):
                continue
            if kind == 'shape':
                if float(strength) != 1:
                    raise ValueError('Unsupported shape strength')
                up = read_th1(directory,process+'_'+name+'Up',labels)[0]*scale
                down = read_th1(directory,process+'_'+name+'Down',labels)[0]*scale
            else:
                factors = list(map(float,strength.split('/')))
                if not np.isfinite(factors).all() or min(factors)<=0 or len(factors) not in (1,2):
                    raise ValueError('Invalid lnN endpoint')
                low,high = (1/factors[0],factors[0]) if len(factors)==1 else factors
                up,down = nominal*high,nominal*low
            pair = result['nuisance_deltas'].setdefault(name,dict(up=np.zeros(size),down=np.zeros(size)))
            pair['up'] += up-nominal; pair['down'] += down-nominal
    if not result['groups']:
        raise ValueError('No background support')
    return result


def combine(records):
    """Same-name deltas sum before the envelope, including across years."""
    if not records:
        raise ValueError('Empty prediction combination')
    first = records[0]; n=len(first['background'])
    if any(r['labels'] != first['labels'] or r['sr_data_blinded'] != first['sr_data_blinded'] for r in records):
        raise ValueError('Mixed layout or blinding in combination')
    out = dict(labels=list(first['labels']),sr_data_blinded=first['sr_data_blinded'],
               background=np.zeros(n),variance=np.zeros(n),groups={},group_variances={},
               nuisance_deltas={},signals={})
    if not first['sr_data_blinded']:
        out.update(data=np.zeros(n),data_variance=np.zeros(n))
    for r in records:
        for field in ('background','variance') + (() if first['sr_data_blinded'] else ('data','data_variance')):
            out[field] += array(r[field],n,True)
        if first['sr_data_blinded'] and ('data' in r or 'data_variance' in r):
            raise ValueError('SR observation in prediction export')
        for field in ('groups','group_variances','signals'):
            for name,values in r[field].items():
                out[field].setdefault(name,np.zeros(n))[:] += array(values,n,True)
        for name,pair in r['nuisance_deltas'].items():
            dst=out['nuisance_deltas'].setdefault(name,dict(up=np.zeros(n),down=np.zeros(n)))
            for direction in ('up','down'):
                dst[direction] += array(pair[direction],n)
    np.testing.assert_allclose(sum(out['groups'].values()),out['background'],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(sum(out['group_variances'].values()),out['variance'],rtol=1e-12,atol=1e-10)
    out['uncertainty'] = np.sqrt(out['variance'] + sum(
        np.maximum(abs(p['up']),abs(p['down']))**2 for p in out['nuisance_deltas'].values()))
    return out


def _prediction_metadata(grid):
    """Inherit the verified grid policy; never synthesize missing approval fields."""
    return {k:grid[k] for k in ('statistical_revision','rate_parameters','signal_contamination',
        'auto_mc_stats','background_estimation_uncertainties','systematic_combination','preliminary',
        'central_conventions_consistent','systematic_scope','bins','th1_channels','sr_data_blinded',
        'years','systematic_evaluation_deferred','hybrid_systematics','run3_systematics','input_scopes_by_era',
        'luminosity_uncertainties','luminosity_model','luminosity_policy_sha256') if k in grid}


def _template_validation_provenance(grid, templates, provenance):
    """Carry the exact producer check level, without upgrading partial checks."""
    from copy import deepcopy
    field = 'template_validation'
    if field not in grid and field not in templates:
        return provenance  # Preserve legacy exports with no new claim.
    value = grid.get(field)
    if (field not in grid or field not in templates or not isinstance(value, dict)
            or not value or value != templates[field]):
        raise ValueError('Template/grid validation metadata mismatch')
    return dict(provenance, template_validation=deepcopy(value))


def export_grid(manifest_path):
    """Read only verified current templates, with no SR data_obs access."""
    import uproot
    manifest_path = internal_path(manifest_path)
    card_path,provenance = grid_card(manifest_path,'impact')
    grid = read_json(manifest_path)
    from TROTASR.workflows.diagnostic_common import require_revised_record, record_years
    revised = require_revised_record(grid)
    years = record_years(grid)
    rate_range = (.01, 5) if revised else (0, 10)
    from TROTASR.utils.sr_merge import load_merge
    merge = load_merge(grid['sr_merge']) if grid.get('sr_merge') else None
    required = [('T2tt',s['key']) for s in renderer.SIGNAL_OVERLAYS] + [('T2bW',m) for m in LOW_SIGNALS]
    available = {(p['model'],p['mass']) for p in grid['points']}
    if set(required)-available:
        raise ValueError('Missing original SR signal overlay; no nearest-point substitution')
    card = parse_card(card_path.read_text())
    templates = read_json(ROOT/grid['templates_manifest'])
    provenance = _template_validation_provenance(grid, templates, provenance)
    if templates.get('sr_merge') != grid.get('sr_merge'):
        raise ValueError('Template/grid SR merge mismatch')
    for key,path in (('sr_binning_sha256','jsons/highdm_sr_binning.json'),('gnn_sha256',configuration_path(years[0]))):
        if templates['contract'].get(key)!=sha256(ROOT/path):
            raise ValueError('Templates use a different plot/physics layout')
    if grid.get('hybrid_systematics') or grid.get('run3_systematics'):
        if templates['contract'].get('gnn_sha256_by_era') != {
                y: sha256(configuration_path(y)) for y in years}:
            raise ValueError('Mixed-era GNN configuration contract changed')
    result = {}
    for year in years:
        for mode,regions in (('highdm',('SR',)),('lowdm',('SR','LLCR','QCDCR','GCR'))):
            for region in regions:
                channel = f'{region}_{mode}_c1_{year}'
                labels = expected_labels(channel, merge)
                if templates['channels'][channel] != labels:
                    raise ValueError('Changed native bin order')
                shape = [r for r in card['shapes'] if r[1]=='*' and r[2]==channel]
                if len(shape)!=1 or shape[0][4:]!=['$CHANNEL/$PROCESS','$CHANNEL/$PROCESS_$SYSTEMATIC']:
                    raise ValueError('Unsupported shape object mapping')
                path = internal_path(card_path.parent/shape[0][3])
                with uproot.open(path,object_cache=None,array_cache=None) as root:
                    directory = root[channel]
                    record = channel_prediction(card,channel,directory,labels,rate_range)
                    if region=='SR':
                        points = required[:len(renderer.SIGNAL_OVERLAYS)] if mode=='highdm' else required[len(renderer.SIGNAL_OVERLAYS):]
                        for model,mass in points:
                            record['signals'][mass] = read_th1(directory,'signal_'+model+'_'+mass,labels)[0]
                result[year+'/'+region+'/'+mode] = record
    metadata = _prediction_metadata(grid)
    return dict(schema='trotasr_card_prediction_plot_v1',status='complete',scope=grid['scope'],
        provenance=provenance,channels=json_ready(result),sr_observations_included=False,
        source_inputs=templates['contract']['inputs'],
        sr_binning_sha256=templates['contract']['sr_binning_sha256'],gnn_sha256=templates['contract']['gnn_sha256'],
        prediction_stage='prefit',rate_initials_applied=True,
        definition='Declared card initials once; MC sumw2 and constrained shape/lnN endpoints. '
                   'Free rateParam ranges are not uncertainties. No observed SR, no new fit.',
        sr_merge=grid.get('sr_merge'), full_workflow_complete=False,
        uncertainty_audit=dict(nuisance_types={r[0]:r[1] for r in card['nuisances']},
            signal_only_nuisances=[r[0] for r in card['nuisances'] if not any(
                p!='signal' and r[i+2] not in ('-','0') for i,(c,p) in enumerate(card['columns']))],
            normalization_range=list(rate_range),mcstat_counted_once=True,
            correlation_policy='sum same-name absolute endpoint deltas across components and years before envelope',
            free_rate_parameters_are_not_added_as_errors=True), **metadata)


def flat_blocks(record, mode, region, merge=None, *, pool_llcr_display=True):
    """Existing high-SR/pooled-CR style values; full map including empty bins."""
    if region=='SR' and mode=='highdm':
        layout=[(label(c),len(c['edges'])-1) for c in (merge['categories'] if merge else expand())]
        specs=renderer.SIGNAL_OVERLAYS
    elif region=='LLCR' and mode=='lowdm' and pool_llcr_display:
        record=ll_display_projection(record)
        layout=[('',5)];specs=[]
    elif region in ('LLCR','QCDCR','GCR') and mode=='lowdm':
        layout=[(r'$N_b=1$',5),(r'$N_b\geq2$',5)];specs=[]
    else:
        raise ValueError('Use original dedicated low-dM SR renderer')
    if len(record['background'])!=sum(n for _,n in layout):
        raise ValueError('Full plot layout mismatch')
    sr=region=='SR'
    if record['sr_data_blinded'] != sr or (sr and 'data' in record):
        raise ValueError('Plot blinding mismatch')
    out=[];offset=0
    for text,n in layout:
        sl=slice(offset,offset+n);offset+=n
        b=dict(nbin=n,label=text,label_box=True,label_fontsize=15,
            category_labels_on_main=True,category_label_y=.72,main_panel_ymax_factor=600.,
            groups={DISPLAY[k]:v[sl] for k,v in record['groups'].items()},
            background=record['background'][sl],background_unc=record['uncertainty'][sl],
            background_stat_unc=np.sqrt(record['variance'][sl]),
            data=np.zeros(n) if sr else record['data'][sl],
            data_unc=np.zeros(n) if sr else np.sqrt(record['data_variance'][sl]),
            signals={k:v[sl] for k,v in record['signals'].items()},signal_specs=specs,
            xlabels=['1','2','3','4','5'],figure_width=14.,main_ylabel='Events',
            physics_scope=region,unit_area=False,blind_data=sr)
        if sr:
            # Use compact category boxes for the dense high-dM SR layout.
            b.update(xlabels=[],label_fontsize=9.5,label_box_pad=.12,figure_width=22.,
                significance_panel=True,significance_ylim=[0.,5.],
                significance_mode='s_over_sqrt_b',significance_ylabel=r'$S/\sqrt{B}$')
        elif region=='LLCR' and pool_llcr_display:
            # Exact style settings of the AN's render_llcr_inclusive.py.
            b.pop('label_fontsize');b.pop('category_labels_on_main');b.pop('category_label_y')
            b.update(label_box=False,annotation=r'Low-$\Delta m$ LLCR',
                     annotation_x=.035,annotation_y=.80)
        out.append(b)
    return out


def ll_display_projection(record):
    """AN-adopted 10 -> 5 display only; project endpoints before envelopes."""
    if record['sr_data_blinded'] or len(record['background'])!=10 or record['signals']:
        raise ValueError('LL projection requires a ten-bin CR without signals')
    def project(values,nonnegative=False):
        values=array(values,10,nonnegative)
        result=values[:5]+values[5:]
        np.testing.assert_allclose(result.sum(),values.sum(),rtol=1e-12,atol=1e-8)
        return result
    out=dict(labels=['LLCR_display_bin'+str(i) for i in range(5)],
             sr_data_blinded=False,signals={})
    for field in ('background','variance','data','data_variance'):
        out[field]=project(record[field],True)
    for field in ('groups','group_variances'):
        if field in record:
            out[field]={k:project(v,True) for k,v in record[field].items()}
    out['nuisance_deltas']={k:{d:project(p[d]) for d in ('up','down')}
                            for k,p in record['nuisance_deltas'].items()}
    np.testing.assert_allclose(sum(out['groups'].values()),out['background'],rtol=1e-12,atol=1e-8)
    out['uncertainty']=np.sqrt(out['variance']+sum(
        np.maximum(abs(p['up']),abs(p['down']))**2 for p in out['nuisance_deltas'].values()))
    return out


def uncertainty_band(record):
    return dict(status='complete',background=record['background'].tolist(),
        stat_unc=np.sqrt(record['variance']).tolist(),total_unc=record['uncertainty'].tolist(),
        variations={name:{d:(record['background']+pair[d]).tolist() for d in ('up','down')}
                    for name,pair in record['nuisance_deltas'].items()})
