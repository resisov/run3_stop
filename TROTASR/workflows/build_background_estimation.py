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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hists', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, output = internal_path(a.hists), internal_path(a.output)
    payload = read_json(source)
    provenance = dict(campaign_year=str(payload['contract']['year']),
                      hist_input=str(source.relative_to(ROOT)), hist_input_sha256=sha256(source),
                      code={f: sha256(ROOT/f) for f in ('utils/background_estimation.py',
                          'utils/dy_estimation.py','utils/photon_estimation.py','utils/zgamma_estimation.py',
                          'utils/transfer_factors.py', 'gnn4lowdm/config.json')})
    manifest_path = output/'measurement_manifest.json'
    if output.exists():
        old = read_json(manifest_path)
        if old['provenance'] == provenance and all(sha256(output/n) == h for n,h in old['files'].items()):
            print(old['status'], '(existing measurements validated)')
            return 0 if old['status'] == 'complete' else 2
        raise FileExistsError('Preserve divergent measured payloads; use a new directory')
    products = measure(payload)
    products['transfer_factors'] = build_transfer_factors(payload, read_json(ROOT/'gnn4lowdm/config.json'))
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
