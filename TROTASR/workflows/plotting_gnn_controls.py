"""Eight yearly GNN controls through the common physical-distribution template.

LL/QCD/photon use current card-prefit yields, as in the adopted AN figures.
Auxiliary DY retains its measured channel/Nb RZ and MC-statistical band.
Every CR retains two Nb groups and ten bins. No unit-area GCR.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
if __package__ in (None,''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.workflows.plotting_nominal_distributions import (
    validate_histograms,checked_leaf,draw_distribution,nominal_renderer,
    systematic_sources,endpoint_container,assert_band_only)
from TROTASR.utils.statistical_templates import json_ready
from TROTASR.workflows.prediction_plot_inputs import export_grid,combine,flat_blocks,renderer,expected_labels

REGIONS=('LLCR','QCDCR','GCR','DYCR')
POOLED=('Nb1_NISRinclusive','Nb2plus_NISRinclusive')
ALLOWED={'data','DY','GJ','QCD','ST','TT','VV','WtoLNu','Zto2Nu'}


def raw_region(payload,region):
    if region not in ('LLCR','QCDCR','GCR','DY2E','DY2M'):
        raise ValueError('Only control regions may expose data')
    source=payload['histograms']['lowdm']['nominal']
    if set(source.get(region,{}))!={'Nb1','Nb2'}:
        raise ValueError('Missing/extra CR Nb stream: '+region)
    edges=read_json(ROOT/'gnn4lowdm/config.json')['cr_binning']['score_edges']
    result={}
    for nb,category in zip(('Nb1','Nb2'),POOLED):
        samples=source[region][nb]
        if 'data' not in samples:
            raise ValueError('Missing actual CR data stream')
        target=result[category]={}
        for sample,observables in samples.items():
            if sample.startswith(('mStop','T2tt_','T2tb_','T2bW_')):
                continue
            if sample not in ALLOWED:
                raise ValueError('Unknown GNN CR process: '+sample)
            target['data_obs' if sample=='data' else sample]={
                'gnn_score':dict(edges=list(edges),**checked_leaf(observables['gnn_score'],edges))}
    return result


def prepare(payload,rz,allow_validation=False):
    year=validate_histograms(payload,allow_validation)
    if not allow_validation:
        cov=payload.get('coverage',{})
        if (not cov.get('full_input_inventory') or cov.get('failed_files')!=0
                or cov.get('expected_files')!=cov.get('completed_files')):
            raise ValueError('Complete full-file coverage required')
    if (rz.get('status')!='complete' or rz.get('scope')!=payload['scope']
            or str(rz.get('provenance',{}).get('campaign_year'))!=str(year)):
        raise ValueError('Incomplete or different-year/scope RZ')
    edges=read_json(ROOT/'gnn4lowdm/config.json')['cr_binning']['score_edges']
    result=dict(status='complete',year=year,scope=payload['scope'],score_edges=edges,
                histograms={'nominal':{},'nominal_rz':{}},rz_application=[])
    sources=systematic_sources(payload)
    if sources:result.update(systematic_sources=sources,dy_endpoint_containers={})
    for region in ('LLCR','QCDCR','GCR','DY2E','DY2M'):
        result['histograms']['nominal'][region]=raw_region(payload,region)
        result['histograms']['nominal_rz'][region]={}
        for nb,category in zip(('Nb1','Nb2'),POOLED):
            target=result['histograms']['nominal'][region][category]
            scaled=deepcopy(target)
            if region in ('DY2E','DY2M'):
                group='Nb1' if nb=='Nb1' else 'Nb2plus'
                fit=rz['rz_low']['channels'][region][group]
                factor=float(fit['RZ'])
                if (fit.get('status')!='complete' or not fit.get('fit_converged')
                        or not np.isfinite(factor) or factor<0):
                    raise ValueError('Invalid current channel/Nb RZ measurement')
                if 'DY' in scaled:
                    leaf=scaled['DY']['gnn_score']
                    leaf['sumw']=(np.asarray(leaf['sumw'])*factor).tolist()
                    leaf['sumw2']=(np.asarray(leaf['sumw2'])*factor**2).tolist()
                result['rz_application'].append(dict(region=region,nb=group,RZ=factor,
                    valid_boundary_fit=factor==0,fit_illustration_not_independent_closure=True))
            result['histograms']['nominal_rz'][region][category]=scaled
            if sources and region in ('DY2E','DY2M'):
                container=endpoint_container(payload,'histograms','lowdm',region,nb,
                    'gnn_score',edges,sources)
                for leaf in container.get('DY',{}).values():
                    leaf['sumw']=(np.asarray(leaf['sumw'])*factor).tolist()
                    leaf['sumw2']=(np.asarray(leaf['sumw2'])*factor**2).tolist()
                result['dy_endpoint_containers'].setdefault(region,{})[category]=container
    return result


def ll_inclusive(payload):
    result=deepcopy(payload)
    for variation,regions in result['histograms'].items():
        old=regions['LLCR']
        if set(old)!=set(POOLED):
            raise ValueError('LL display requires the two unchanged Nb groups')
        merged={}
        for sample in sorted(set(old[POOLED[0]])|set(old[POOLED[1]])):
            leaves=[old[c][sample]['gnn_score'] for c in POOLED if sample in old[c]]
            if any(l['edges']!=payload['score_edges'] for l in leaves):
                raise ValueError('Cannot pool unequal GNN score axes')
            merged[sample]={'gnn_score':dict(edges=payload['score_edges'],**{
                field:sum((np.asarray(l[field]) for l in leaves),np.zeros(5)).tolist()
                for field in ('sumw','sumw2','entries')})}
        regions['LLCR']={'Nb1plus_NISRinclusive':merged}
    return result


def source_hashes():
    provenance=read_json(ROOT/'jsons/prediction_plot_sources.json')
    for source in provenance['sources']:
        if sha256(ROOT/source['target'])!=source['target_sha256']:
            raise ValueError('Preserved prediction renderer changed')
    hashes={s['target']:s['target_sha256'] for s in provenance['sources']}
    name='utils/renderers/plot_control_search_bins_style.py'
    expected=next(s['sha256'] for s in read_json(ROOT/'jsons/sources.json')['files'] if s['target']==name)
    if sha256(ROOT/name)!=expected:
        raise ValueError('Original SR/CR renderer changed')
    hashes[name]=expected
    return hashes


def current_card_controls(export,year,hist_digest,rz_digest):
    from TROTASR.workflows.diagnostic_common import require_revised_record
    require_revised_record(export)
    if (export.get('status')!='complete' or export.get('scope') not in
            ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production')
            or export.get('prediction_stage')!='prefit' or export.get('rate_initials_applied') is not True
            or export.get('sr_observations_included') is not False):
        raise ValueError('Current blinded full-input prefit export required')
    inputs=export['source_inputs'][str(year)]
    if inputs['hists']['sha256']!=hist_digest or inputs['rz_low']['sha256']!=rz_digest:
        raise ValueError('Card predictions and auxiliary histograms/measurements differ')
    result={}
    for region in ('LLCR','QCDCR','GCR'):
        record=export['channels'][str(year)+'/'+region+'/lowdm']
        if record['labels']!=expected_labels(f'{region}_lowdm_c1_{year}'):
            raise ValueError('Changed native CR bin order')
        result[region]=combine([record])
    return result


def score_interval_labels():
    edges=read_json(ROOT/'gnn4lowdm/config.json')['cr_binning']['score_edges']
    return [f'[{low:.3g}, {high:.3g}{"]" if i==len(edges)-2 else ")"}'
            for i,(low,high) in enumerate(zip(edges[:-1],edges[1:]))]


def distribution_blocks(record, region):
    """Native ten-bin prefit inputs; common template and real score labels."""
    fields = ('nbin','label','groups','background','background_unc',
              'background_stat_unc','data','data_unc','signals','signal_specs',
              'xlabels','physics_scope','unit_area','blind_data')
    blocks=[{k:deepcopy(block[k]) for k in fields}
            for block in flat_blocks(record, 'lowdm', region,pool_llcr_display=False)]
    for block in blocks:
        block.update(xlabels=score_interval_labels(),variable='gnn_score')
    return blocks


def dy_distribution_blocks(payload,include_systematics=True):
    """Channel/Nb RZ once; ordinary distribution record builder, no DY draw()."""
    blocks=[]
    for category,label in zip(POOLED,(r'$N_b=1$',r'$N_b\geq2$')):
        parts=[]
        for channel in ('DY2E','DY2M'):
            if include_systematics and payload.get('dy_endpoint_containers'):
                parts.append(payload['dy_endpoint_containers'][channel][category]);continue
            records=payload['histograms']['nominal'][channel][category]
            raw={}
            for sample,axes in records.items():
                source=(payload['histograms']['nominal_rz'][channel][category][sample]
                        if sample=='DY' else axes)
                raw[sample]={'nominal':deepcopy(source['gnn_score'])}
            parts.append(raw)
        raw=renderer.combine_histogram_containers(*parts)
        old_lumi=renderer.LUMINOSITY_RELATIVE_UNCERTAINTY
        try:
            renderer.LUMINOSITY_RELATIVE_UNCERTAINTY=0.
            block=renderer.lowdm_variable_record(
                {'lowdm_variable_specs':{'gnn_score':{'bins':payload['score_edges']}}},
                'DYCR','gnn_score',label,False,raw_override=raw)
        finally:renderer.LUMINOSITY_RELATIVE_UNCERTAINTY=old_lumi
        # The existing auxiliary DY comparison has MC statistics only. A
        # template change must not silently add a luminosity uncertainty.
        if not include_systematics or not payload.get('dy_endpoint_containers'):
            block['background_unc']=block['background_stat_unc'].copy()
        block.update(edges=[],xlabels=score_interval_labels(),blind_data=False)
        blocks.append(block)
    return blocks


def render_input(payload,output):
    if payload.get('schema')!='trotasr_gnn_control_plot_inputs_v1':
        raise ValueError('Unknown compact GNN input')
    contract=payload['contract'];year=contract['year']
    if (contract['code_sha256']!=sha256(Path(__file__)) or contract['renderers']!=source_hashes()
            or contract['distribution_code_sha256']!=sha256(ROOT/'workflows/plotting_nominal_distributions.py')):
        raise ValueError('Changed compact GNN renderer contract')
    receipt=output/'plot_manifest.json'
    if receipt.exists():
        previous=read_json(receipt)
        if previous['contract']!=contract or any(sha256(output/n)!=d for n,d in previous['files'].items()):
            raise ValueError('Changed previous GNN figures')
        return previous
    if output.exists() and any(output.iterdir()):raise FileExistsError('Preserve incomplete GNN output')
    output.mkdir(parents=True,exist_ok=True);plots=[]
    for region,blocks in payload['blocks'].items():
        name='lowdm_'+region.lower()+'_gnn_out_inclusive'
        info=draw_distribution(blocks,output/name,'GNN output',{2024:109.82,2025:110.84}[year],
            ('MC stat+syst unc.' if payload['nuisance_names']['DYCR'] else 'MC stat. unc.')
            if region=='DYCR' else 'Prefit stat+syst unc.')
        if info['bins']!=10 or info.get('unit_area') or info.get('dropped_input_bins_1based'):
            raise ValueError('Changed adopted prefit CR display')
        info.update(status='generated_pending_visual_QA',region=region,
            prediction_stage='auxiliary_RZ_scaled' if region=='DYCR' else 'prefit',
            nuisance_names=payload['nuisance_names'][region],scope=contract['scope'],
            an_reference=f'figures/results_distributions/{year}/{name}.pdf')
        plots.append(info)
    files={f.name:sha256(f) for f in output.iterdir() if f.suffix in ('.png','.pdf')}
    if len(files)!=8:raise ValueError('Missing GNN figures')
    write_json(output/'plot_inputs.json',payload)
    result=dict(status='generated_pending_visual_QA',contract=contract,plots=plots,files=files,
        plot_inputs_sha256=sha256(output/'plot_inputs.json'),uncertainty_audit=payload['uncertainty_audit'],
        rz_application=payload['rz_application'],llcr_display_only_nb_pooling=False,
        likelihood_changed=False,full_workflow_complete=False)
    write_json(receipt,result);return result


def export_loaded(h,rz,export,hist_digest,rz_digest,allow_validation=False):
    """Canonical compact projection; shared reads never imply a new physics input."""
    if rz.get('provenance',{}).get('hist_input_sha256')!=hist_digest:
        raise ValueError('RZ does not belong to these histograms')
    payload=prepare(h,rz,allow_validation)
    predictions=current_card_controls(export,payload['year'],hist_digest,rz_digest)
    contract=dict(hists_sha256=hist_digest,rz_sha256=rz_digest,year=payload['year'],
        scope=payload['scope'],code_sha256=sha256(Path(__file__)),renderers=source_hashes(),
        prediction_provenance=export['provenance'],
        prediction_code_sha256=sha256(ROOT/'workflows/prediction_plot_inputs.py'),
        distribution_code_sha256=sha256(ROOT/'workflows/plotting_nominal_distributions.py'),
        rendering_entry_point='plotting_nominal_distributions.draw_distribution')
    with nominal_renderer(payload['year']):old_dy=dy_distribution_blocks(payload,False)
    with nominal_renderer(payload['year'],payload.get('systematic_sources',[])):
        prepared={region:distribution_blocks(record,region) for region,record in predictions.items()}
        prepared['DYCR']=dy_distribution_blocks(payload)
        for old,new in zip(old_dy,prepared['DYCR']):assert_band_only(old,new)
    return json_ready(dict(schema='trotasr_gnn_control_plot_inputs_v1',contract=contract,blocks=prepared,
        score_edges=payload['score_edges'],rz_application=payload['rz_application'],
        nuisance_names={**{r:sorted(v['nuisance_deltas']) for r,v in predictions.items()},
                        'DYCR':payload.get('systematic_sources',[])},
        uncertainty_audit=dict(card=export.get('uncertainty_audit'),
            auxiliary_dy='Existing RZ central plus stored event endpoints and MC statistics; no new luminosity term',
            auxiliary_dy_rz_fit_uncertainty='Not propagated: no approved correlated display-error mapping',
            same_name_shifts_summed_before_envelope=True,mcstat_counted_once=True,
            central_sumw2_data_labels_axes_exactly_preserved=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--hists',type=Path)
    inputs.add_argument('--input',type=Path)
    p.add_argument('--rz-low',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--predictions',type=Path,help='Verified compact native-card export')
    p.add_argument('--allow-validation',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--manifest',type=Path,
                   help='Explicit current verified native grid; never default to a historical model')
    a=p.parse_args()
    if a.input:
        if a.manifest or a.predictions or a.rz_low or a.export_only or a.allow_validation:
            raise ValueError('Compact GNN input already freezes all sources')
        compact=read_json(internal_path(a.input))
        if a.validate_only:print(compact['contract']);return
        render_input(compact,internal_path(a.output));return
    if not a.rz_low or bool(a.manifest)==bool(a.predictions):
        raise ValueError('RZ and exactly one current native or compact prediction source required')
    hists,rzpath,output=map(internal_path,(a.hists,a.rz_low,a.output))
    h=read_json(hists);rz=read_json(rzpath)
    export=export_grid(internal_path(a.manifest)) if a.manifest else read_json(internal_path(a.predictions))
    compact=export_loaded(h,rz,export,sha256(hists),sha256(rzpath),a.allow_validation)
    if a.validate_only:
        print(dict(status='validated_plot_inputs_not_generated',count=4,contract=compact['contract']));return
    if a.export_only:
        if output.exists() and any(output.iterdir()):raise FileExistsError('Preserve compact GNN export')
        output.mkdir(parents=True,exist_ok=True);write_json(output/'plot_inputs.json',compact)
        print('Exported four compact GNN controls');return
    render_input(compact,output)


if __name__=='__main__':main()
