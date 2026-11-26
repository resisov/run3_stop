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
from TROTASR.gnn4lowdm.configuration import configuration
from TROTASR.utils.io import read_json, without_provenance_fields
from TROTASR.utils.sr_binning import expand, label
from TROTASR.utils.plot_adapter import renderer
from TROTASR.utils.statistical_templates import json_ready
from TROTASR.workflows.diagnostic_common import grid_card, provisional_cronly_products

DISPLAY = {'Top':'Top', 'WJet':'W -> lv', 'ZJet':'Z -> vv', 'QCD':'QCD Multijet',
           'PhotonJet':'Photon+jet', 'DYJet':'DY', 'VV':'VV+VVV'}
LOW_PROCESSES = {'VV':'VV', 'Top':'Top', 'DYJet':'DY', 'PhotonJet':'Photon',
                 'WJet':'W', 'ZJet':'Zinv', 'QCD':'QCD'}
LOW_SIGNALS = ('mStop800_mLSP650', 'mStop1000_mLSP850')

ERA_VIEW_SCOPE = 'single_era_card_prefit_view'


def _llcr_prior_band(nuisances, evaluate, *, covariance=None):
    """Sum central finite differences in quadrature; a shared coordinate moves all eras."""
    nominal = np.asarray(evaluate(), dtype=float)
    variance = np.zeros_like(nominal)
    stat_variance = np.zeros_like(nominal)
    if covariance is not None:
        covariance.update(full=np.zeros((nominal.size,nominal.size)),
                          stat=np.zeros((nominal.size,nominal.size)))
    counts = dict(gaussian=0, poisson=0, auto_mc_stats=0)
    for variable, sigma, kind, mc_stat in nuisances:
        center = float(variable.getVal())
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Invalid prefit prior width: '+variable.GetName())
        try:
            variable.setVal(max(center-sigma, float(variable.getMin()))
                            if variable.hasMin() else center-sigma)
            low = np.asarray(evaluate(), dtype=float)
            variable.setVal(min(center+sigma, float(variable.getMax()))
                            if variable.hasMax() else center+sigma)
            high = np.asarray(evaluate(), dtype=float)
        finally:
            variable.setVal(center)
        if (low.shape != nominal.shape or high.shape != nominal.shape
                or not np.isfinite(low).all() or not np.isfinite(high).all()):
            raise ValueError('Nonfinite or changed LLCR prior response: '+variable.GetName())
        shift = (high-low)/2.
        shift2 = shift**2
        if covariance is not None:
            pair = np.outer(shift,shift)
            covariance['full'] += pair
            if mc_stat: covariance['stat'] += pair
        variance += shift2
        counts[kind] += 1
        if mc_stat:
            stat_variance += shift2
            counts['auto_mc_stats'] += 1
    return np.sqrt(variance), np.sqrt(stat_variance), counts


def _llcr_prefit_stage(card_path, workspace_path):
    """Read only the declared high-dM LLCR observation and pristine native workspace."""
    import ROOT as root
    from contextlib import ExitStack
    from TROTASR.utils.era_merge import CHANNEL_BINS

    channel = 'LLCR_highdm_c1'
    card_path, workspace_path = internal_path(card_path), internal_path(workspace_path)
    rows = [line.split('#', 1)[0].split() for line in card_path.read_text().splitlines()]
    channels = next(row[1:] for row in rows if row[:1] == ['bin'])
    if channel not in channels or any(c.startswith('SR_') for c in channels):
        raise ValueError('Only a CR-only workspace may supply the LLCR prefit comparison')
    expected = [f'LLCR_highdm_{nb}_bin{i}' for nb in ('Nb1','Nb2plus') for i in range(6)]
    if len(expected) != CHANNEL_BINS[channel]:
        raise ValueError('Expected six recoil bins for each b multiplicity')
    # Resolve only the high-dM LLCR observation shape. No SR key or observed fit is opened.
    observation = [r for r in rows if r[:3] == ['shapes', 'data_obs', channel]]
    if len(observation) != 1 or len(observation[0]) != 5 or observation[0][4] != channel+'/data_obs':
        raise ValueError('Missing explicit LLCR data_obs shape')
    column_channels = [r[1:] for r in rows if r[:1] == ['bin']][1]
    proc_names, proc_ids = [r[1:] for r in rows if r[:1] == ['process']]
    if len(proc_names) != len(column_channels) or len(proc_names) != len(proc_ids) or any(
            int(pid) <= 0 for pid in proc_ids):
        raise ValueError('Expected a background-only CR card')
    declared = {r[0] for r in rows if len(r)>1 and r[1] in ('lnN', 'shape')}
    free = {r[0]:float(r[4]) for r in rows if len(r)>4 and r[1]=='rateParam' and '@' not in r[4]}
    if not any(len(r) == 5 and r[1:] == ['autoMCStats','10','1','1']
               and fnmatch.fnmatch(channel,r[0]) for r in rows):
        raise ValueError('Expected the canonical LLCR autoMCStats model')
    root.gROOT.SetBatch(True)
    root.gSystem.Load('libHiggsAnalysisCombinedLimit')
    with ExitStack() as stack:
        def load(path):
            obj = root.TFile.Open(str(path),'READ')
            if not obj or obj.IsZombie() or obj.TestBit(root.TFile.kRecovered):
                raise ValueError('Unreadable LLCR prefit input: '+str(path))
            stack.callback(obj.Close)
            return obj
        w = load(workspace_path).Get('w')
        if not w or not w.obj('ModelConfig') or not w.var('CMS_th1x'):
            raise ValueError('Missing native Combine model')
        # Text2workspace writes an unfit workspace. Reject other parameter settings;
        # free-rate bounds are not prior errors and we never apply best-fit values.
        r = w.var('r')
        if r:
            r.setVal(0)
        mc = w.obj('ModelConfig')
        prop = w.function('prop_bin'+channel)
        if not prop or prop.ClassName() != 'CMSHistErrorPropagator':
            raise ValueError('Missing native LLCR histogram propagator')
        for name, initial in free.items():
            var = w.var(name)
            if not var or not np.isclose(float(var.getVal()), initial, rtol=1e-10, atol=1e-12):
                raise ValueError('Workspace does not preserve card rate initial: '+name)
        data_file = load(internal_path(card_path.parent/observation[0][3]))
        obs = data_file.Get(observation[0][4])
        if not obs or obs.GetNbinsX() != len(expected):
            raise ValueError('Wrong LLCR data bin count')
        labels = [str(obs.GetXaxis().GetBinLabel(i)) for i in range(1,len(expected)+1)]
        if labels != expected:
            raise ValueError('Wrong LLCR category or recoil bin ordering: '+str(labels))
        data = np.asarray([obs.GetBinContent(i) for i in range(1,len(expected)+1)])
        if not np.isfinite(data).all() or np.any(data < 0) or np.any(data != np.floor(data)):
            raise ValueError('Invalid observed CR counts')
        x = w.var('CMS_th1x')
        def totals():
            out = []
            for i in range(len(expected)):
                x.setVal(i+0.5)
                out.append(float(prop.getVal()))
            return np.asarray(out)
        background = totals()
        groups = {g:np.zeros_like(background) for g in renderer.GROUP_ORDER}
        wrappers, coefs = prop.wrapperList(), prop.coefList()
        native = [str(wrapper.getStringAttribute('combine.process')) for wrapper in wrappers]
        if (len(native) != len(coefs) or len(set(native)) != len(native)
                or set(native) != {proc for c,proc in zip(column_channels, proc_names) if c == channel}):
            raise ValueError('Workspace and LLCR card have different process lists')
        for i in range(len(expected)):
            x.setVal(i+0.5)
            for name, wrapper, coef in zip(native,wrappers,coefs):
                groups[DISPLAY[name.split('_',1)[0]]][i] += float(wrapper.getVal())*float(coef.getVal())
        if (not np.isfinite(background).all() or np.any(background < 0)
                or not np.allclose(sum(groups.values()),background,rtol=2e-9,atol=1e-6)):
            raise ValueError('Native LLCR process yields do not sum to the prediction')
        nuisances = []
        for p in mc.GetNuisanceParameters():
            if p.isConstant() or not prop.dependsOn(p):
                continue
            name = str(p.GetName())
            if name in free:
                continue
            var = w.var(name)
            nominal = float(var.getVal())
            stat = name.startswith('prop_bin')
            if name not in declared and not stat:
                raise ValueError('Unrecognized LLCR model nuisance: '+name)
            if name in declared and not np.isclose(nominal,0,atol=1e-10):
                raise ValueError('Expected the central LLCR systematic setting: '+name)
            constraint = w.pdf(name+'_Pdf')
            if stat and constraint and constraint.InheritsFrom('RooPoisson') and nominal > 0:
                sigma, kind = np.sqrt(nominal), 'poisson'
            elif var.hasError() and var.getError() > 0:
                sigma, kind = float(var.getError()), 'gaussian'
            elif stat and nominal > 0:
                # Count-valued gamma constraints in Combine can be wrapped; a
                # plain positive count is the native prefit Poisson convention.
                sigma, kind = np.sqrt(nominal), 'poisson'
            elif np.isclose(nominal,0,atol=1e-10):
                # Centered lnN/shape and Gaussian autoMCStats are unit-width pulls
                # even if the virgin workspace has not attached a RooFit error.
                if not constraint or not constraint.InheritsFrom('RooGaussian'):
                    raise ValueError('No native Gaussian prior for '+name)
                sigma, kind = 1., 'gaussian'
            else:
                raise ValueError('Unknown LLCR prior width: '+name)
            nuisances.append((var,sigma,kind,stat))
        if not nuisances:
            raise ValueError('Native CR model has no LLCR priors')
        cov = {}
        unc, stat, counts = _llcr_prior_band(nuisances, totals, covariance=cov)
    def totals_over(sl):
        # Sum the bin shifts of a common nuisance before squaring.
        return dict(data=int(data[sl].sum()),background=float(background[sl].sum()),
                    background_unc=float(np.sqrt(max(0.,cov['full'][sl,sl].sum()))),
                    background_stat_unc=float(np.sqrt(max(0.,cov['stat'][sl,sl].sum()))))
    stage = dict(inclusive=totals_over(slice(None)))
    for j, nb in enumerate(('Nb1','Nb2plus')):
        sl = slice(6*j,6*j+6)
        stage[nb] = dict(labels=labels[sl],data=data[sl].tolist(),
            groups={g:v[sl].tolist() for g,v in groups.items()}, background=background[sl].tolist(),
            background_unc=unc[sl].tolist(),background_stat_unc=stat[sl].tolist(),
            summed_bins=totals_over(sl))
    return stage, counts


def _compare_llcr_before(before, accepted):
    """Verify only the actual before-cut yields/bands, independent of any postfit result."""
    previous = accepted['channels']['LLCR_highdm_c1']
    bands = previous['prefit']
    checks = {}
    for j, nb in enumerate(('Nb1','Nb2plus')):
        sl = slice(6*j,6*j+6)
        row = before[nb]
        if row['labels'] != previous['labels'][sl] or row['data'] != previous['data'][sl]:
            raise ValueError('Before-cut LLCR data/labels differ from the accepted prefit: '+nb)
        pairs = [('total', row['background'], bands['background']),
                 ('uncertainty',row['background_unc'],bands['background_unc'])]
        pairs += [(g, row['groups'][g], v) for g,v in bands['groups'].items()]
        for key, current, old_all in pairs:
            old = np.asarray(old_all[sl],dtype=float)
            checks[nb+'/'+key]=float(np.max(np.abs(np.asarray(current)-old)))
            if not np.allclose(current,old,rtol=2e-4,atol=1e-6):
                raise ValueError('Before-cut LLCR prefit mismatch in '+nb+'/'+key+
                                 ', max absolute delta='+str(checks[nb+'/'+key]))
    return checks


def llcr_prefit_comparison(before_card, before_workspace, after_card, after_workspace):
    from TROTASR.utils.paths import luminosity_fb
    before, after = [(internal_path(c),internal_path(w)) for c,w in
        ((before_card,before_workspace),(after_card,after_workspace))]
    if before[0] == after[0] or before[1] == after[1]:
        raise ValueError('The before/after LLCR models must be distinct')
    stages, counts = {}, {}
    for label, model in (('before',before),('after',after)):
        stages[label], counts[label] = _llcr_prefit_stage(*model)
    old = ROOT/'stats/run3/cronly'
    reference = internal_path(old/read_json(old/'publication.json')['projection'])
    if reference.parent != old:
        raise ValueError('Before-cut prefit projection must belong to the accepted CR results')
    deltas = _compare_llcr_before(stages['before'],read_json(reference))
    if any(np.any(np.asarray(stages['after'][nb]['data']) > stages['before'][nb]['data'])
           for nb in ('Nb1','Nb2plus')):
        raise ValueError('Tightened LLCR data count exceeds original occupancy')
    return dict(schema='trotasr_llcr_cut_prefit_v1',prediction_stage='prefit',stages=stages,
        years=list(ERAS),luminosity_fb=sum(luminosity_fb(y) for y in ERAS),
        models={k:dict(card=str(c.relative_to(ROOT)),workspace=str(w.relative_to(ROOT)))
                for k,(c,w) in [('before',before),('after',after)]},
        prior_nuisances=counts,fit_executed=False,postfit_values_used=False,sr_observations_used=False,
        before_reference=str(reference.relative_to(ROOT)),before_max_abs_deltas=deltas,
        uncertainty_label='Prefit unc.',uncertainty_method='Native model priors: independent half-differences; each common coordinate varied once, including autoMCStats')


def cronly_refit_paths(result, variant):
    if variant not in ('fixed', 'matched'):
        raise ValueError('Unknown saved CR-only refit')
    return (result/f'fitDiagnostics_cronly_refit_{variant}.root',
            result/'commands'/f'cronly_refit_{variant}.json')


def _cr_observation_shapes(card, channels):
    """Resolve only the six explicit data_obs shapes in the blinded CR card."""
    result = {}
    for line in card.read_text().splitlines():
        row = line.split()
        if row[:2] != ['shapes', 'data_obs']:
            continue
        if len(row) != 5 or row[2] not in channels or row[2] in result:
            raise ValueError('Unexpected observed shape in the CR-only card')
        channel, path, key = row[2:]
        if key != channel+'/data_obs':
            raise ValueError('Observed CR shape does not identify exactly its CR channel')
        result[channel] = (internal_path(card.parent/path), key)
    if set(result) != set(channels):
        raise ValueError('Require six explicit observed CR histogram shapes')
    return result


def export_cronly_fit(result, output, *, refit=None, write=True):
    """Project the accepted CR-only RooFit model without fitting or rebuilding.

    Evaluate the existing CMSHistErrorPropagator totals and official process
    wrappers directly: no repeated per-process histograms for every covariance
    draw, and no duplicate TotalBkg/TotalProcs calculation.
    """
    import fcntl
    import math
    import time
    from contextlib import ExitStack
    from TROTASR.workflows.vr_fit_projection import completed_cr_fit_inputs
    from TROTASR.utils.io import write_json
    from TROTASR.utils.era_merge import CHANNEL_BINS
    result, output = internal_path(result), internal_path(output)
    expected = {name: n for name, n in CHANNEL_BINS.items() if not name.startswith('SR_')}
    baseline = ROOT/'stats/run3'
    selected = result.name == 'cronly' and baseline in result.parent.parents
    if selected:
        if refit or (write and result.parent/'plots' not in output.parents):
            raise ValueError('A separate completed CR model uses its own plots, without a refit')
        with (result/'controller.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
            card, native, published = completed_cr_fit_inputs(result.parent)
        fitted_card = card
        current_tt = result == baseline/'top_pt/cronly'
        if current_tt:
            fitted_card = internal_path(ROOT/read_json(result/'state.json')['fitted_card'])
            if fitted_card != result/'linked_card.txt':
                raise ValueError('Expected the fitted Photon-linked CR card')
        observed = _cr_observation_shapes(fitted_card, expected)
    else:
        published = read_json(result/'publication.json')
    required = ('workspace', 'projection') if refit else ('projection',)
    products = provisional_cronly_products(result, published, required=required)
    if products is not None and not refit:
        with (result/'controller.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
            payload = read_json(products['projection'])
            channels = payload.get('channels', {})
            if (payload.get('schema') != 'trotasr_cronly_prefit_postfit_v1'
                    or payload.get('sr_observations_included') is not False
                    or payload.get('fit_recalculated') is not False
                    or payload.get('fit', {}).get('covariance_quality') not in (2, 3)
                    or tuple(payload.get('years', ())) != ERAS
                    or set(channels) != set(expected)
                    or any(len(channels[name].get('data', ())) != n for name, n in expected.items())):
                raise ValueError('Invalid provisional CR-only projection')
            for name, n in expected.items():
                row = channels[name]
                data = array(row['data'], n, nonnegative=True)
                if np.any(data != np.floor(data)):
                    raise ValueError('CR-only observations must be unweighted counts')
                for stage in ('prefit', 'postfit'):
                    array(row[stage]['background'], n)
                    array(row[stage]['background_unc'], n, nonnegative=True)
            cached = output/'prediction_payload.json'
            if write and cached.is_file():
                recorded = read_json(cached)
                if any(recorded.get(k) != payload.get(k) for k in (
                        'schema', 'years', 'sr_observations_included', 'channels')):
                    raise ValueError('Divergent CR projection at output path')
            elif write:
                write_json(cached, payload)
            return payload
    if not selected:
        card = result.parent/'grid/datacards/datacard_cronly.txt'
    if refit:
        if products is None or output == result.parent/'plots/cronly':
            raise ValueError('Refit plots require the current workspace and a separate figure destination')
        fit_path, receipt_path = cronly_refit_paths(result, refit)
        receipt = read_json(receipt_path)
        workspace = products['workspace']
        if (receipt.get('status') != 'complete' or receipt.get('exit_code') != 0
                or receipt.get('fit_status') != 0 or receipt.get('signal_strength') != 0
                or receipt.get('sr_observed_data_used') is not False
                or not receipt.get('nll_cache_recreated')):
            raise ValueError('Normally terminated CR-only refit required')
        if refit == 'matched' and (receipt.get('matched_initialization') is not True
                or receipt.get('float_parameters') != 305
                or receipt.get('invalid_nll') != 0
                or receipt.get('fit_steps') != {'MINIMIZE':0,'HESSE':0}):
            raise ValueError('Verified matched-initialization refit required')
        prefit_source = read_json(products['projection'])
        if (prefit_source.get('schema') != 'trotasr_cronly_prefit_postfit_v1'
                or prefit_source.get('sr_observations_included') is not False
                or tuple(prefit_source.get('years', ())) != ERAS):
            raise ValueError('Missing approved prefit projection')
        sources = (('workspace',workspace),('fit',fit_path),('prefit',products['projection']))
    else:
        if (published.get('kind') != 'cronly' or published.get('complete') is not True
                or published.get('sr_observed_data_used') is not False
                or published.get('dataset') != 'observed CR-only B-only r0'):
            raise ValueError('Accepted observed CR-only result required')
        workspace, fit_path = ((native['workspace'], native['fit_result']) if selected else
                               (result/'workspace.root', result/'fitDiagnostics_cronly.root'))
        sources = (('workspace',workspace),('fit',fit_path),('card',card))
    identity = {name:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for name,p in sources}
    payload_path = output/'prediction_payload.json'
    if payload_path.is_file():
        saved = read_json(payload_path)
        if (saved.get('schema') == 'trotasr_cronly_prefit_postfit_v1'
                and saved.get('sr_observations_included') is False
                and tuple(saved.get('years', ())) == ERAS
                and bool('clipped_parameter_values' in saved) == bool(refit or selected)
                and (not selected or (saved.get('fit_source') == published
                                     and saved.get('projection')
                                     == 'native CMSHistErrorPropagator and CMSHistFuncWrapper with the adopted Photon CR GNN terms'))):
            return saved
        raise ValueError('Cached CR projection has incompatible schema, years or fit mode')
    import ROOT as root
    started = time.time()
    with ExitStack() as stack:
        owner = stack.enter_context((result/'controller.lock').open('rb'))
        fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
        if selected and completed_cr_fit_inputs(result.parent) != (card, native, published):
            raise ValueError('Registered CR fit changed while preparing the projection')
        if write:
            output.mkdir(parents=True,exist_ok=True)
            lock = stack.enter_context((output/'plot.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data_paths = (tuple(sorted({p for p,_ in observed.values()})) if selected
                      else (card.parent/'era_merged_data_obs.root',))
        root_paths = (workspace,fit_path) if refit else (workspace,fit_path,*data_paths)
        files = [root.TFile.Open(str(p),'READ') for p in root_paths]
        try:
            if any(not f or f.IsZombie() or f.TestBit(root.TFile.kRecovered) for f in files):
                raise ValueError('Cannot read the accepted CR-only ROOT inputs')
            w, fit = files[0].Get('w'), files[1].Get('fit_b')
            if (not w or not fit or int(fit.status()) != 0 or fit.covQual() < 2
                    or not math.isfinite(float(fit.edm()))):
                raise ValueError('Missing successful saved CR fit')
            fixed_r = fit.constPars().find('r')
            if fit.floatParsFinal().find('r') or (fixed_r and fixed_r.getVal() != 0):
                raise ValueError('Expected a saved B-only fit with r fixed to zero')
            fit_quality = dict(status=int(fit.status()),covariance_quality=int(fit.covQual()),
                               edm=float(fit.edm()),global_covariance_accepted=False)
            if selected:
                keys = [('fit_status', 'status'), ('covariance_quality', 'covariance_quality'), ('edm', 'edm')]
                if any(fit_quality[v] != published[k] for k,v in keys):
                    raise ValueError('Saved CR fit_b does not match its fit-quality receipt')
                fit_quality.update(process_exit_code=published['process_exit_code'],
                                   source_fit=fit_path.name, source_receipt=published['receipt'])
                if current_tt:
                    cov = read_json(result/'pulls.json')['covariance_validation']
                    fit_quality['global_covariance_accepted'] = bool(
                        cov['finite'] and cov['minimum_eigenvalue']>0 and cov['max_asymmetry']<1.e-6)
            if refit:
                fit_quality['process_exit_code'] = receipt['exit_code']
                fit_quality['source_fit'] = fit_path.name
                fit_quality['source_receipt'] = receipt_path.name
            channels = [r+'_'+m+'_c1' for m in ('highdm','lowdm')
                        for r in ('LLCR','QCDCR','GCR')]
            category = w.cat('CMS_channel')
            if {str(c.first) for c in category} != set(channels):
                raise ValueError('Only the six observed CR channels may be projected')
            x = w.var('CMS_th1x')
            modified_channel = 'GCR_lowdm_c1'
            linked = refit or selected
            modified, delta_terms = None, []
            if linked:
                from TROTASR.workflows.cr_linked_workspace import (
                    GNN_BY_GROUP, PHOTON_GROUPS, RECOIL_BY_GROUP, TOP_PT_CRONLY_REVISION, _photon_pdf,
                )
                modified = _photon_pdf(w)
                groups = PHOTON_GROUPS if modified.funcList().getSize() == 11 else ("Nb1",)
                if selected:
                    both_nb = published.get('model_revision') == TOP_PT_CRONLY_REVISION
                    floats = fit.floatParsFinal()
                    if (both_nb != (groups == PHOTON_GROUPS)
                            or any(not floats.find(n) for nb in groups for n in GNN_BY_GROUP[nb])
                            or any(floats.find(n) for nb in groups for n in RECOIL_BY_GROUP[nb])):
                        raise ValueError('The selected fit did not preserve the adopted Photon CR GNN model')
                delta_terms = list(zip(list(modified.funcList())[1:], list(modified.coefList())[1:]))
            data_files = dict(zip(data_paths, files[2:]))
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
                if refit:
                    row = prefit_source['channels'][channel]
                    if len(row['data']) != n:
                        raise ValueError('Changed approved CR observation bins')
                    records[channel] = {key:row[key] for key in ('labels','data','prefit')}
                else:
                    data_path, key = observed[channel] if selected else (data_paths[0], channel+'/data_obs')
                    data = data_files[data_path].Get(key)
                    if not data or int(data.GetNbinsX()) != n:
                        raise ValueError('Changed approved CR observation bins')
                    records[channel] = dict(
                        labels=[str(data.GetXaxis().GetBinLabel(i)) for i in range(1,n+1)],
                        data=[float(data.GetBinContent(i)) for i in range(1,n+1)])
                    if selected and any(n < 0 or not math.isfinite(n) or n != int(n)
                                        for n in records[channel]['data']):
                        raise ValueError('CR-only observations must be finite unweighted counts')
                slices[channel] = slice(offset,offset+n);offset += n
            def totals():
                values = []
                for channel in channels:
                    prop = modified if linked and channel == modified_channel else components[channel][0]
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
                        if linked and channel == modified_channel:
                            groups[DISPLAY['PhotonJet']][i] += sum(f.getVal()*c.getVal() for f,c in delta_terms)
                    if not np.allclose(sum(groups.values()),total[slices[channel]],rtol=2e-9,atol=1e-6):
                        raise ValueError('CR process wrappers do not sum to the native total')
                    records[channel][stage] = dict(groups={k:v.tolist() for k,v in groups.items()},
                                                  background=total[slices[channel]].tolist())
                return total
            if not refit:
                central('prefit')
            prior = [] if refit else list(files[1].Get('nuisances_prefit_res').floatParsFinal())
            variance = np.zeros(offset)
            prior_counts = prefit_source['prior_nuisances'] if refit else dict(gaussian=0,poisson=0)
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
            if linked:
                w.var('r').setVal(0)
            for p in fit.floatParsFinal():
                w.var(p.GetName()).setVal(p.getVal())
            central('postfit')
            root.RooRandom.randomGenerator().SetSeed(12345)
            randomized = list(fit.randomizePars())
            targets = [w.var(p.GetName()) for p in randomized]
            limits = [(v.getMin(),v.getMax()) for v in targets] if linked else []
            clipped = 0
            samples = np.empty((500,offset))
            for index in range(500):
                fit.randomizePars()
                for j,(v,p) in enumerate(zip(targets,randomized)):
                    value = p.getVal()
                    if linked:
                        low, high = limits[j]
                        bounded = min(high,max(low,value))
                        clipped += bounded != value
                        value = bounded
                    v.setVal(value)
                samples[index] = totals()
                if (index+1)%100 == 0:
                    print('CR postfit correlated covariance samples',index+1,'/ 500',flush=True)
            if not np.isfinite(samples).all():
                raise ValueError('Nonfinite CR covariance propagation')
            postfit_error = samples.std(axis=0,ddof=0)
            for channel in channels:
                if not refit:
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
            if linked:
                payload['projection'] += ' with the adopted Photon CR GNN terms'
                payload['uncertainty_method']['postfit'] += '; workspace-bound projection'
                payload['clipped_parameter_values'] = clipped
            if refit:
                payload['uncertainty_method']['prefit'] = 'reused approved prefit, same model'
            if selected:
                payload['fit_source'] = published
                if current_tt:
                    from TROTASR.workflows.cr_linked_workspace import _cr_observations
                    if _cr_observations(w) != {k:v['data'] for k,v in records.items()}:
                        raise ValueError('Plotted observations differ from the fitted CR workspace')
                    payload['observation_card'] = str(fitted_card.relative_to(ROOT))
            if write:
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
    bins = parent['bins'] if parent.get('era_merged') else parent['bins'] // len(parent['years'])
    result.update(scope=ERA_VIEW_SCOPE, years=[year], bins=bins, th1_channels=8,
        channels=selected, source_inputs={year:deepcopy(payload['source_inputs'][year])},
        full_workflow_complete=False,
        single_era_view=dict(plot_only=True, fit_ready=False, source_model=parent))
    prediction_years(result)
    return result


def prediction_years(payload):
    """A single-era display comes only from an accepted card result."""
    from TROTASR.workflows.diagnostic_common import require_revised_record, record_years

    if payload.get('scope') == ERA_VIEW_SCOPE:
        view = payload.get('single_era_view', {})
        parent = view.get('source_model', {})
        if not require_revised_record(parent):
            raise ValueError('Adopted parent statistical revision required')
        years = tuple(payload.get('years', ()))
        if (view.get('plot_only') is not True or view.get('fit_ready') is not False
                or len(years) != 1 or years[0] not in record_years(parent)
                or payload.get('bins') != parent.get('bins', 0) // (1 if parent.get('era_merged') else len(record_years(parent))) or payload.get('th1_channels') != 8
                or payload.get('full_workflow_complete') is not False
                or any(without_provenance_fields(payload.get(k)) != without_provenance_fields(parent.get(k)) for k in (
                    'statistical_revision','auto_mc_stats','signal_contamination',
                    'background_estimation_uncertainties','sr_merge','systematic_combination',
                    'uncertainty_audit','sr_data_blinded','sr_observations_included',
                    'prediction_stage','rate_initials_applied'))
                or payload.get('sr_observations_included') is not False
                or payload.get('sr_data_blinded') is not True):
            raise ValueError('Invalid exact single-era card view')
        return years
    if 'single_era_preview' in payload or 'single_era_view' in payload:
        raise ValueError('A standalone preview is not a current card result')
    return record_years(payload)


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
        'years','systematic_evaluation_deferred','run3_systematics','input_scopes_by_era',
        'luminosity_uncertainties','luminosity_model') if k in grid}


def _template_validation_provenance(grid, templates, provenance):
    """Record the reported check levels without making them plot prerequisites."""
    from copy import deepcopy
    field = 'template_validation'
    recorded = {name: source[field] for name, source in (('grid', grid), ('templates', templates))
                if isinstance(source.get(field), dict) and source[field]}
    if not recorded:
        return provenance
    result = dict(provenance)
    if 'templates' in recorded:
        result['template_validation'] = deepcopy(recorded['templates'])
    if 'grid' in recorded:
        result['grid_template_validation'] = deepcopy(recorded['grid'])
    return result


def _merged_rate_bindings(rows, z_coeff):
    """Decode only the rate formulas generated by the era-merged card writer."""
    import re
    from TROTASR.utils.era_merge import highdm_z_shape_formula,highdm_z_shape_group
    bindings={}
    for name,kind,ch,proc,value,tail in rows:
        key=(ch,proc)
        if kind!='rateParam' or key in bindings:raise ValueError('Ambiguous rate binding: '+str(key))
        if tail.startswith('['):
            lo,hi=map(float,tail.strip('[]').split(','))
            central=float(value)
            if not lo<=central<=hi:raise ValueError('Rate initial outside bounds')
            bindings[key]=(name,1.,central)
        elif '__zShape_' in name:
            arg=tail.split(',')[0];match=re.match(r'\(@0\*([0-9.eE+-]+)/\(',value)
            if not match or name!=arg+'__zShape_'+proc:raise ValueError('Unrecognized Z shape scale')
            scale=float(match[1]);era=proc.rsplit('_',1)[-1]
            expr,args=highdm_z_shape_formula(arg,scale,z_coeff[era][highdm_z_shape_group(arg)])
            if (ch!='SR_highdm_c1' or expr!=value or ','.join(args)!=tail or scale<=0):
                raise ValueError('Z shape equation/receipt mismatch')
            bindings[key]=(arg,scale,None)
        else:
            match=re.fullmatch(r'\(@0\*([0-9.eE+-]+)\)',value)
            if not match or '__eraScale_' not in name or float(match[1])<=0:
                raise ValueError('Unsupported rate formula: '+name)
            bindings[key]=(tail,float(match[1]),None)
    return bindings


def _merged_sr_export(grid, templates, card_path, provenance, required):
    """Project the merged likelihood by era without opening any observed shape."""
    from contextlib import ExitStack
    import math
    import uproot
    from TROTASR.utils.era_merge import HIGHDM_Z_SHAPE, highdm_z_shape_group
    from TROTASR.utils.sr_merge import load_merge

    card = parse_card(card_path.read_text())
    z_coeff = grid[HIGHDM_Z_SHAPE]
    bindings = _merged_rate_bindings(card['rateparams'],z_coeff)
    initials = {}
    for name, _, value in bindings.values():
        if value is not None and initials.setdefault(name,value) != value:
            raise ValueError('Inconsistent shared normalization initial: '+name)
    if len(initials)!=grid['floating_normalization_parameters']:
        raise ValueError('Missing shared normalization initials')
    native = {p['model']+'/'+p['mass']:p for p in grid['points'] if p.get('fit_ready')}
    shapes = {(r[2],r[1]):r for r in card['shapes'] if r[1]!='data_obs'}
    signal_rules = {}
    for model,mass in required:
        path = internal_path(ROOT/native[model+'/'+mass]['card'])
        rules = [line.split() for line in path.read_text().splitlines() if line.startswith('shapes signal_')]
        signal_rules[model,mass] = {(r[2],r[1]):(path,r) for r in rules}
    from TROTASR.workflows.diagnostic_common import record_years
    channels, signal_sources = {}, {}
    for year in record_years(grid):
        for mode in ('highdm','lowdm'):
            ch='SR_'+mode+'_c1'; origin=ch+'_'+year; key=year+'/SR/'+mode
            labels=expected_labels(origin,load_merge(grid['sr_merge']))
            if templates['channels'][origin] != labels:
                raise ValueError('Changed native template edges/order: '+origin)
            # Project the declared, non-excluded columns back to their native
            # template names. Keep shared nuisance names across eras.
            indices=[i for i,(c,p) in enumerate(card['columns']) if c==ch and
                p.endswith('_'+year) and not p.startswith('signal_')]
            projected=dict(columns=[],rateparams=[],nuisances=[
                row[:2]+[row[2+i] for i in indices] for row in card['nuisances']])
            with ExitStack() as stack:
                opened={}
                def source(path):
                    if path not in opened:
                        opened[path]=stack.enter_context(uproot.open(path,object_cache=None,array_cache=None))
                    return opened[path]
                bg_paths=set()
                for i in indices:
                    c,p=card['columns'][i]; rule=shapes[c,p]; name=p[:-(len(year)+1)]
                    if rule[4:] != [origin+'/'+name,origin+'/'+name+'_$SYSTEMATIC']:
                        raise ValueError('Unrecognized merged shape binding: '+p)
                    projected['columns'].append((origin,name))
                    bg_paths.add(internal_path(card_path.parent/rule[3]))
                    if (c,p) not in bindings:continue
                    parameter,scale,_=bindings[c,p]
                    central=initials[parameter]*scale
                    if '__zShape_' in next(r[0] for r in card['rateparams'] if r[2:4]==[c,p]):
                        w=z_coeff[year][highdm_z_shape_group(parameter)]
                        central/=math.fsum(weight*initials[arg] for arg,weight in w.items())
                    if not math.isfinite(central) or not .01-1e-12<=central<=5+1e-12:
                        raise ValueError('Invalid native central normalization: '+p)
                    projected['rateparams'].append([parameter,'rateParam',origin,name,str(central),'[0.01,5]'])
                if len(bg_paths)!=1:raise ValueError('Expected one native template per era and mode')
                path=bg_paths.pop()
                record=channel_prediction(projected,origin,source(path)[origin],labels,(.01,5))
                for model,mass in required:
                    if (model=='T2tt') != (mode=='highdm'):continue
                    p,r=signal_rules[model,mass][ch,'signal_'+year]
                    obj='signal_'+model+'_'+mass
                    if r[4]!=origin+'/'+obj:raise ValueError('Signal overlay/card binding mismatch')
                    sigpath=internal_path(p.parent/r[3])
                    record['signals'][mass]=read_th1(source(sigpath)[origin],obj,labels)[0]
                    signal_sources.setdefault(model+'/'+mass,[]).append(dict(era=year,
                        card=str(p.relative_to(ROOT)),template=str(sigpath.relative_to(ROOT)),object=r[4]))
                channels[key]=record
    metadata=_prediction_metadata(grid)
    metadata.update({k:grid[k] for k in ('era_merged','likelihood_channels','floating_normalization_parameters',
        'source_bins','source_likelihood_channels','signal_input_revision','tt_card_input') if k in grid})
    return dict(schema='trotasr_card_prediction_plot_v1',status='complete',scope=grid['scope'],
        provenance=dict(provenance,signal_sources=signal_sources),channels=json_ready(channels),
        sr_observations_included=False,prediction_stage='prefit',rate_initials_applied=True,
        sr_merge=grid['sr_merge'],merged_sr_only=True,full_workflow_complete=False,
        uncertainty_audit=dict(nuisance_types={r[0]:r[1] for r in card['nuisances']},mcstat_counted_once=True,
            correlation_policy='sum signed same-name endpoint shifts across bins/components/eras before envelope',
            free_rate_parameters_are_not_added_as_errors=True),**metadata)


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
    if grid.get('era_merged') and grid.get('tt_card_input'):
        templates=read_json(internal_path(ROOT/grid['templates_manifest']))
        if (templates.get('sr_data_blinded') is not True
                or templates.get('tt_card_input') != grid['tt_card_input']):
            raise ValueError('SR templates must be blinded and use the grid ttbar prescription')
        return _merged_sr_export(grid, templates, card_path, provenance, required)
    available = {(p['model'],p['mass']) for p in grid['points']}
    if set(required)-available:
        raise ValueError('Missing original SR signal overlay; no nearest-point substitution')
    card = parse_card(card_path.read_text())
    templates = read_json(ROOT/grid['templates_manifest'])
    provenance = _template_validation_provenance(grid, templates, provenance)
    if without_provenance_fields(templates.get('sr_merge')) != without_provenance_fields(grid.get('sr_merge')):
        raise ValueError('Template/grid SR merge mismatch')
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
