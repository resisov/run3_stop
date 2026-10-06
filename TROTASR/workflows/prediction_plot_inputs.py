"""Exact prefit-card projection for plots; no event input or fit execution.

This retains the legacy exporter prescription: apply declared rateParam initial
values once, add independent MC variances, and sum same-name nuisance endpoint
deltas before taking their envelope. Free parameter ranges are NOT errors.
SR observations are never read, including the synthetic data_obs template.
"""
import fnmatch
from pathlib import Path
import numpy as np
from TROTASR.utils.paths import ROOT, internal_path
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


def array(values, size, nonnegative=False):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or (nonnegative and (result < 0).any()):
        raise ValueError('Invalid prediction array')
    return result


def expected_labels(channel, merge=None):
    region, mode, packing, year = channel.split('_')
    if packing != 'c1' or year not in ('2024', '2025'):
        raise ValueError('Unexpected native channel')
    if region == 'SR' and mode == 'highdm':
        return ['SR_highdm_bin'+str(i) for i in range(merge['bins_per_year'] if merge else 162)]
    if mode == 'lowdm':
        cats = (read_json(ROOT/'gnn4lowdm/config.json')['sr_binning']['category_labels']
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


def export_grid(manifest_path):
    """Read only verified current templates, with no SR data_obs access."""
    import uproot
    manifest_path = internal_path(manifest_path)
    card_path,provenance = grid_card(manifest_path,'impact')
    grid = read_json(manifest_path)
    from TROTASR.workflows.diagnostic_common import require_revised_record
    revised = require_revised_record(grid)
    rate_range = (.01, 5) if revised else (0, 10)
    from TROTASR.utils.sr_merge import load_merge
    merge = load_merge(grid['sr_merge']) if grid.get('sr_merge') else None
    required = [('T2tt',s['key']) for s in renderer.SIGNAL_OVERLAYS] + [('T2bW',m) for m in LOW_SIGNALS]
    available = {(p['model'],p['mass']) for p in grid['points']}
    if set(required)-available:
        raise ValueError('Missing original SR signal overlay; no nearest-point substitution')
    card = parse_card(card_path.read_text())
    templates = read_json(ROOT/grid['templates_manifest'])
    if templates.get('sr_merge') != grid.get('sr_merge'):
        raise ValueError('Template/grid SR merge mismatch')
    for key,path in (('sr_binning_sha256','jsons/highdm_sr_binning.json'),('gnn_sha256','gnn4lowdm/config.json')):
        if templates['contract'].get(key)!=sha256(ROOT/path):
            raise ValueError('Templates use a different plot/physics layout')
    result = {}
    for year in ('2024','2025'):
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
    metadata = {k:grid[k] for k in ('statistical_revision','rate_parameters','signal_contamination',
        'auto_mc_stats','background_estimation_uncertainties','systematic_combination','preliminary',
        'central_conventions_consistent','systematic_scope','bins','th1_channels','sr_data_blinded') if k in grid}
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
            # User-requested high-dM category-box reduction, 2026-09-24.
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
