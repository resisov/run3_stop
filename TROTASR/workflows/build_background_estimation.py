"""Measure the existing background model from current TROTASR histograms."""
import argparse
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256, scientific_settings
from TROTASR.utils.background_estimation import measure
from TROTASR.utils.transfer_factors import build_transfer_factors
from TROTASR.utils.zgamma_estimation import ZGamma_BPIX_MERGE
from TROTASR.gnn4lowdm.configuration import configuration_path


def merged_zgamma_measurement(payload, previous, policy=ZGamma_BPIX_MERGE):
    """Merge measurement inputs using the existing ratio equations."""
    import copy
    import math
    import numpy as np
    from TROTASR.utils.background_estimation import recoil_boundary, ratio_inputs, finite_json
    from TROTASR.utils.zgamma_estimation import rebin, ratio, records
    if (str(payload['contract']['year']) != ZGamma_BPIX_MERGE['year']
            or scientific_settings(policy) != ZGamma_BPIX_MERGE):
        raise ValueError('Exact 2023BPix measurement-only merge required')
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
    result['measurement_bin_policy']=copy.deepcopy(ZGamma_BPIX_MERGE)
    result['measurement_merge_audit']=finite_json(dict(
        input_totals=[{k:v[k] for k in fields} for v in inputs],
        merged_totals=[{k:v[k] for k in fields} for v in merged],
        valid_nonmerged_rows_preserved=True,lowdm_unchanged=True,
        equations='zgamma_estimation.rebin/ratio/records',
        downstream_mapping='existing native-recoil closure_record overlap mapping unchanged'))
    return finite_json(result)


def recover_zgamma(output, payload):
    """Complete the diagnosed measurement in place without an earlier snapshot."""
    import fcntl
    with (output/'zgamma_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        name = 'double_ratio.json.gz'
        path = output / name
        manifest_path = output / 'measurement_manifest.json'
        manifest = read_json(manifest_path)
        if (str(payload['contract']['year']) != ZGamma_BPIX_MERGE['year']
                or str(manifest['provenance']['campaign_year']) != ZGamma_BPIX_MERGE['year']):
            raise ValueError('Only the diagnosed 2023BPix measurement can be resumed')
        if any(sha256(output/n) != h for n, h in manifest['files'].items() if n != name):
            raise ValueError('An unrelated measurement changed')
        prior = manifest['files'][name]
        current = sha256(path)
        if manifest['stages']['double_ratio'] != 'blocked':
            if current != prior:
                raise ValueError('Changed accepted Z/gamma measurement')
            return 0 if manifest['status'] == 'complete' else 2
        result = read_json(path)
        if current == prior:
            result = merged_zgamma_measurement(payload, result)
            result['measurement_merge_audit']['replaces_sha256'] = prior
            write_json(path, result)
            current = sha256(path)
        elif (result.get('status') != 'complete'
                or scientific_settings(result.get('measurement_bin_policy')) != ZGamma_BPIX_MERGE
                or result.get('measurement_merge_audit', {}).get('replaces_sha256') != prior):
            raise ValueError('Changed incomplete Z/gamma measurement')
        manifest['stages']['double_ratio'] = 'complete'
        manifest['files'][name] = current
        manifest['status'] = 'complete' if all(s == 'complete' for s in manifest['stages'].values()) else 'blocked'
        write_json(manifest_path, manifest)
        print(manifest['status'], '(only Z/gamma measurement recovered)')
        return 0 if manifest['status'] == 'complete' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hists', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, output = internal_path(a.hists), internal_path(a.output)
    payload = read_json(source)
    from TROTASR.workflows.rare_top_policy import require_clean
    require_clean(payload)
    year = str(payload['contract']['year'])
    manifest_path = output/'measurement_manifest.json'
    if output.exists():
        old = read_json(manifest_path)
        if str(old['provenance'].get('campaign_year')) != year:
            raise ValueError('Existing measurements belong to a different era')
        if year == ZGamma_BPIX_MERGE['year'] and old['stages'].get('double_ratio') == 'blocked':
            return recover_zgamma(output, payload)
        # The measured products, not mutable producer code or merged signal bytes, are inputs.
        if all(sha256(output/n) == h for n,h in old['files'].items()):
            print(old['status'], '(existing measurements validated)')
            return 0 if old['status'] == 'complete' else 2
        raise FileExistsError('Preserve divergent measured payloads; use a new directory')
    gnn_path = configuration_path(year)
    provenance = dict(campaign_year=year, hist_input=str(source.relative_to(ROOT)), hist_input_sha256=sha256(source),
        code={f: sha256(ROOT/f) for f in ('utils/background_estimation.py', 'utils/dy_estimation.py',
              'utils/photon_estimation.py', 'utils/zgamma_estimation.py', 'utils/transfer_factors.py',
              str(gnn_path.relative_to(ROOT)))})
    products = measure(payload)
    if year == ZGamma_BPIX_MERGE['year'] and products['double_ratio']['status'] == 'blocked':
        products['double_ratio'] = merged_zgamma_measurement(payload, products['double_ratio'])
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
    return 0 if status=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
