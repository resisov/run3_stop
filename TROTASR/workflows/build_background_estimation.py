"""Measure the existing background model from current TROTASR histograms."""
import argparse
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.background_estimation import measure
from TROTASR.utils.transfer_factors import build_transfer_factors
from TROTASR.gnn4lowdm.configuration import configuration_path


def merged_zgamma_measurement(payload, previous, policy):
    """Merge measurement inputs using the existing ratio equations."""
    import copy
    import math
    import numpy as np
    from TROTASR.utils.background_estimation import recoil_boundary, ratio_inputs, finite_json
    from TROTASR.utils.zgamma_estimation import rebin, ratio, records
    if (str(payload['contract']['year']) != '2023BPix' or policy.get('year') != '2023BPix'
            or policy.get('id') != 'zgamma_2023BPix_measurement_merge_20261022'
            or policy.get('user_approved') is not True or policy.get('mode') != 'highdm'
            or policy.get('original_edges') != [250,300,350,400,500,1500]
            or policy.get('measurement_edges') != [250,300,400,500,1500]
            or any(policy.get(k) is not True for k in ('analysis_bins_unchanged',
                'gnn_boundaries_unchanged','preserve_other_measurements'))):
        raise ValueError('Exact approved 2023BPix measurement-only merge required')
    old = previous['highdm']['bins']
    if (len(old) != 5 or [(r['low'],r['high']) for r in old] !=
            list(zip(policy['original_edges'][:-1],policy['original_edges'][1:]))
            or old[2]['status'] != 'unavailable'
            or any(old[i]['status'] != 'complete' for i in (0,1,3,4))
            or any(r['status'] != 'complete' for r in previous['lowdm']['bins'])):
        raise ValueError('Unexpected original Z/gamma measurement failure')
    exact = recoil_boundary(payload)
    inputs = [ratio_inputs(exact,'highdm',regions,target)
              for regions,target in ((('DY2E','DY2M'),'DY'),(('GCR',),'GJ'))]
    if not inputs[0]['target'][2] < 0:
        raise ValueError('The diagnosed signed DY cell is not reproduced')
    fields = ('data','data_variance','other','other_variance','target','target_variance')
    edges = np.asarray(policy['measurement_edges'],dtype=float)
    merged=[]
    for source in inputs:
        totals={k:rebin(source[k],source['edges'],edges) for k in fields}
        value,stat,residual,variance=ratio(*(totals[k] for k in fields))
        merged.append(dict(totals,value=value,stat=stat,residual=residual,
                           residual_variance=variance,edges=edges))
    row=records(*merged)[1]
    if row['status'] != 'complete' or any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):
        raise ValueError('Approved merged measurement is still unsupported')
    result=copy.deepcopy(previous)
    result['highdm']['bins']=[copy.deepcopy(old[0]),row,copy.deepcopy(old[3]),copy.deepcopy(old[4])]
    result['status']='complete'
    result['measurement_bin_policy']=copy.deepcopy(policy)
    result['measurement_merge_audit']=finite_json(dict(
        input_totals=[{k:v[k] for k in fields} for v in inputs],
        merged_totals=[{k:v[k] for k in fields} for v in merged],
        valid_nonmerged_rows_preserved=True,lowdm_unchanged=True,
        equations='zgamma_estimation.rebin/ratio/records',
        downstream_mapping='existing native-recoil closure_record overlap mapping unchanged'))
    return finite_json(result)


def recover_zgamma(source, output, payload, policy_path):
    """Finite, locked replacement of only the explicitly failed measurement."""
    import fcntl
    import shutil
    with (output/'zgamma_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        policy=read_json(policy_path)
        receipt_path=output/'zgamma_recovery.json'
        if receipt_path.exists():
            receipt=read_json(receipt_path)
            if (receipt['policy_sha256'] != sha256(policy_path)
                    or receipt['hist_sha256'] != sha256(source)
                    or receipt['double_ratio_sha256'] != sha256(output/'double_ratio.json.gz')):
                raise ValueError('Changed completed Z/gamma recovery')
            print('complete (verified existing Z/gamma recovery)'); return 0
        manifest_path=output/'measurement_manifest.json'
        old=read_json(manifest_path)
        if old['provenance']['hist_input_sha256'] != sha256(source):
            raise ValueError('Measurement belongs to different histograms')
        if any(sha256(output/n)!=h for n,h in old['files'].items()):
            raise ValueError('Changed original measurements')
        if old['stages']['double_ratio'] != 'blocked':
            raise ValueError('Only the diagnosed failed measurement can be resumed')
        result=merged_zgamma_measurement(payload,read_json(output/'double_ratio.json.gz'),policy)
        history=output/'history/zgamma_measurement_merge_20261022'
        history.mkdir(parents=True,exist_ok=False)
        for name in ('double_ratio.json.gz','measurement_manifest.json'):
            shutil.copy2(output/name,history/name)
        write_json(output/'double_ratio.json.gz',result)
        old['stages']['double_ratio']='complete'
        old['files']['double_ratio.json.gz']=sha256(output/'double_ratio.json.gz')
        old['status']='complete' if all(s=='complete' for s in old['stages'].values()) else 'blocked'
        receipt=dict(status='complete',policy=str(policy_path.relative_to(ROOT)),
            policy_sha256=sha256(policy_path),hist_sha256=sha256(source),
            entrypoint_sha256=sha256(Path(__file__)),prior=str(history.relative_to(ROOT)),
            original_double_ratio_sha256=sha256(history/'double_ratio.json.gz'),
            double_ratio_sha256=old['files']['double_ratio.json.gz'],
            preserved={n:h for n,h in old['files'].items() if n!='double_ratio.json.gz'})
        old['measurement_recovery']=receipt
        write_json(manifest_path,old)
        for n,h in receipt['preserved'].items():
            if sha256(output/n)!=h:raise ValueError('An unrelated measurement changed')
        write_json(receipt_path,receipt)
        print('complete (only Z/gamma measurement recovered)',result['highdm']['bins'][1])
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hists', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-zgamma-merge', type=Path,
                   help='Explicit approved measurement-only policy; preserve other outputs')
    a = p.parse_args()
    source, output = internal_path(a.hists), internal_path(a.output)
    payload = read_json(source)
    from TROTASR.workflows.rare_top_policy import require_clean
    require_clean(payload)
    if a.resume_zgamma_merge:
        return recover_zgamma(source,output,payload,internal_path(a.resume_zgamma_merge))
    gnn_path = configuration_path(payload['contract']['year'])
    provenance = dict(campaign_year=str(payload['contract']['year']),
                      hist_input=str(source.relative_to(ROOT)), hist_input_sha256=sha256(source),
                      code={f: sha256(ROOT/f) for f in ('utils/background_estimation.py',
                          'utils/dy_estimation.py','utils/photon_estimation.py','utils/zgamma_estimation.py',
                          'utils/transfer_factors.py', str(gnn_path.relative_to(ROOT)))})
    manifest_path = output/'measurement_manifest.json'
    if output.exists():
        old = read_json(manifest_path)
        if old['provenance'] == provenance and all(sha256(output/n) == h for n,h in old['files'].items()):
            print(old['status'], '(existing measurements validated)')
            return 0 if old['status'] == 'complete' else 2
        raise FileExistsError('Preserve divergent measured payloads; use a new directory')
    products = measure(payload)
    products['transfer_factors'] = build_transfer_factors(payload, read_json(gnn_path))
    output.mkdir(parents=True)
    for name, product in products.items():
        product.update(provenance=provenance, scope=payload.get('scope'), sr_data_blinded=True)
        write_json(output/(name+'.json.gz'), product)
    status = 'complete' if all(v['status']=='complete' for v in products.values()) else 'blocked'
    write_json(manifest_path, dict(status=status, provenance=provenance,
        stages={n:v['status'] for n,v in products.items()},
        files={n+'.json.gz':sha256(output/(n+'.json.gz')) for n in products}, full_workflow_complete=False))
    print(status, {n:v['status'] for n,v in products.items()})
    if (str(payload['contract']['year']) == '2023BPix'
            and products['double_ratio']['status'] == 'blocked'):
        return recover_zgamma(source, output, payload,
            ROOT/'jsons/zgamma_2023BPix_20261022_policy.json')
    return 0 if status=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
