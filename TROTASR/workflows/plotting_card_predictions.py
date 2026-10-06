"""Two blinded combined SR and three pooled CR figures from current native cards."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
if __package__ in (None,''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.utils.io import read_json,write_json,sha256
from TROTASR.utils.statistical_templates import json_ready
from TROTASR.workflows.prediction_plot_inputs import (
    export_grid,combine,flat_blocks,uncertainty_band,LOW_PROCESSES,LOW_SIGNALS,renderer,expected_labels)
from TROTASR.workflows.plotting_gnn_controls import source_hashes,distribution_blocks
from TROTASR.workflows.plotting_nominal_distributions import draw_distribution
from TROTASR.utils.sr_merge import load_merge


def prepare(payload):
    from TROTASR.workflows.diagnostic_common import require_revised_record
    require_revised_record(payload)
    if (payload.get('schema')!='trotasr_card_prediction_plot_v1' or payload.get('status')!='complete'
            or payload.get('scope') not in ('full_nominal_production','full_cms_trota_jme_production','full_combined_systematic_production')
            or payload.get('prediction_stage')!='prefit'
            or payload.get('sr_observations_included') is not False or payload.get('rate_initials_applied') is not True):
        raise ValueError('Only current full-input blinded prefit predictions')
    expected={f'{y}/{r}/{m}' for y in ('2024','2025') for m,rs in
              (('highdm',('SR',)),('lowdm',('SR','LLCR','QCDCR','GCR'))) for r in rs}
    if set(payload['channels'])!=expected:
        raise ValueError('Missing/extra year or plot channel')
    for key,path in (('sr_binning_sha256','jsons/highdm_sr_binning.json'),('gnn_sha256','gnn4lowdm/config.json')):
        if payload.get(key)!=sha256(ROOT/path):
            raise ValueError('Changed prediction layout contract')
    merge=load_merge(payload['sr_merge']) if payload.get('sr_merge') else None
    for key,record in payload['channels'].items():
        y,r,m=key.split('/')
        if record['labels']!=expected_labels(f'{r}_{m}_c1_{y}', merge):
            raise ValueError('Prediction bins not in the full native order')
    records={}
    for mode,regions in (('highdm',('SR',)),('lowdm',('SR','LLCR','QCDCR','GCR'))):
        for region in regions:
            record=combine([payload['channels'][y+'/'+region+'/'+mode] for y in ('2024','2025')])
            if len(record['background'])!={'highdm':merge['bins_per_year'] if merge else 162,'lowdm':30 if region=='SR' else 10}[mode]:
                raise ValueError('Changed plot bin count')
            records[mode,region]=record
    if set(records['highdm','SR']['signals'])!={s['key'] for s in renderer.SIGNAL_OVERLAYS}:
        raise ValueError('Changed high-dM signal overlays')
    if set(records['lowdm','SR']['signals'])!=set(LOW_SIGNALS):
        raise ValueError('Changed low-dM T2bW signal overlays')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest',type=Path,help='Full verified new card grid; ROOT stays on hep2')
    inputs.add_argument('--input',type=Path,help='Compact export from these verified cards')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--export-only',action='store_true',help='Write compact verified inputs without authoring figures')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--sr-only',action='store_true',help='Produce only the two SR figures; do not redraw CRs')
    selection.add_argument('--high-sr-only',action='store_true',help='Produce only high-dM SR; leave low-dM and CR figures unchanged')
    selection.add_argument('--cr-only',action='store_true',help='Produce only the three combined CRs with the shared distribution template')
    a=p.parse_args();output=internal_path(a.output)
    if a.manifest:
        if sys.platform!='linux':
            raise RuntimeError('Native ROOT exports are hep2-only')
        payload=export_grid(internal_path(a.manifest))
    else:
        payload=read_json(internal_path(a.input))
    records=prepare(payload)
    if a.export_only:
        if not a.manifest or a.validate_only:
            raise ValueError('Compact export requires an explicit verified native manifest')
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Preserve earlier compact export')
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'prediction_payload.json',payload)
        print('Exported verified compact inputs; no figures generated')
        return
    merge=load_merge(payload['sr_merge']) if payload.get('sr_merge') else None
    high_bins=merge['bins_per_year'] if merge else 162
    contract=dict(provenance=payload['provenance'],
        code={name:sha256(ROOT/name) for name in ('workflows/plotting_card_predictions.py',
                'workflows/prediction_plot_inputs.py')},renderers=source_hashes(),
        highdm_map=sha256(ROOT/'jsons/highdm_sr_binning.json'),gnn=sha256(ROOT/'gnn4lowdm/config.json'))
    if merge:contract.update(sr_merge=payload['sr_merge'],merge_code_sha256=sha256(ROOT/'utils/sr_merge.py'))
    if a.sr_only:contract['sr_only']=True
    if a.high_sr_only:contract['high_sr_only']=True
    if a.cr_only:contract['cr_only']=True
    if not (a.sr_only or a.high_sr_only):
        contract['distribution_code']={name:sha256(ROOT/name) for name in
            ('workflows/plotting_nominal_distributions.py','workflows/plotting_gnn_controls.py')}
    if a.input:contract['input_sha256']=sha256(internal_path(a.input))
    if a.validate_only:
        print(dict(status='validated_plot_inputs_not_generated',count=1 if a.high_sr_only else 2 if a.sr_only else 3 if a.cr_only else 5,contract=contract));return
    receipt=output/'plot_manifest.json'
    if receipt.exists():
        previous=read_json(receipt)
        if previous['contract']!=contract or any(sha256(output/n)!=d for n,d in previous['files'].items()):
            raise ValueError('Changed previous card-prediction plots')
        print(previous['status']);return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Retain incomplete card plot outputs')
    output.mkdir(parents=True,exist_ok=True)
    from TROTASR.workflows.renderers import lowdm_sr
    plots=[]
    for (mode,region),record in records.items():
        if a.high_sr_only and (mode,region)!=('highdm','SR'):continue
        if a.sr_only and region!='SR':continue
        if a.cr_only and region=='SR':continue
        name=(f'{mode}{high_bins if mode=="highdm" else 30}_sr_2024_2025' if region=='SR'
              else 'lowdm_'+region.lower()+'_gnn_nisr_merged')
        if mode=='lowdm' and region=='SR':
            n=30
            processes={target:record['groups'].get(source,np.zeros(n)) for source,target in LOW_PROCESSES.items()}
            variances={target:record['group_variances'].get(source,np.zeros(n)) for source,target in LOW_PROCESSES.items()}
            signals={'signal_'+k:v for k,v in record['signals'].items()}
            config=read_json(ROOT/'gnn4lowdm/config.json')['sr_binning']
            if config['category_labels']!=[c[3:] for c in lowdm_sr.CATEGORIES]:
                raise ValueError('Low-dM plot category order mismatch')
            edges={'SR_'+k:np.asarray(v) for k,v in config['edges_by_category'].items()}
            lowdm_sr.draw(SimpleNamespace(output=output/name,luminosity_fb=220.66),
                          processes,variances,signals,edges,uncertainty_band(record))
            info=dict(bins=30,png=str(output/(name+'.png')),pdf=str(output/(name+'.pdf')))
        elif region!='SR':
            info=draw_distribution(distribution_blocks(record,region),output/name,
                'GNN output',220.66,'Prefit stat+syst unc.')
            if info['bins']!=10 or info.get('unit_area'):
                raise ValueError('Changed combined CR yields or bin layout')
        else:
            info=renderer.draw_flat_blocks(flat_blocks(record,mode,region,merge),output/name,
                xlabel='Search bin' if region=='SR' else 'GNN bin',luminosity_fb=220.66,
                uncertainty_label_override='MC stat+syst unc.' if region=='SR' else 'Prefit stat+syst unc.')
            expected_bins=high_bins if region=='SR' else (5 if region=='LLCR' else 10)
            if info['bins']!=expected_bins or info.get('dropped_input_bins_1based'):
                raise ValueError('Renderer changed full bin layout')
        info.update(status='generated_pending_visual_QA',mode=mode,region=region,
            prediction_stage='prefit',rate_initials_applied=True,nuisances=sorted(record['nuisance_deltas']),
            source_scope=payload['scope'])
        plots.append(info)
    files={f.name:sha256(f) for f in output.iterdir() if f.suffix in ('.png','.pdf')}
    if len(files)!=(2 if a.high_sr_only else 4 if a.sr_only else 6 if a.cr_only else 10):raise ValueError('Missing card-prediction figures')
    write_json(output/'prediction_payload.json',payload)
    write_json(receipt,dict(status='generated_pending_visual_QA',contract=contract,plots=plots,files=files,
        payload_sha256=sha256(output/'prediction_payload.json'),sr_observations_included=False,
        no_old_fits_reused=True,full_workflow_complete=False))


if __name__=='__main__':main()
