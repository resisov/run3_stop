"""Render current RZ/RT, mll, Q/Sgamma and double-ratio measurements.

Rendering functions are migrated verbatim. No measurement is recomputed here.
The corrected mll panels illustrate their own fit, not an independent closure.
The inherited mll highlight is explicitly audited separately from fit windows.
"""
import argparse
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

PRODUCTS = ('rz_high', 'rz_low', 'sgamma', 'double_ratio')
GROUPS = {'Nb1', 'Nb2plus'}
RECOIL_EDGES = [250., 300., 350., 400., 500., 1500.]


def finite_number(value, nonnegative=False):
    if value is None or not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('Unavailable or malformed measurement number')


def validate_products(products):
    if set(products) != set(PRODUCTS) or any(p.get('status') != 'complete' for p in products.values()):
        raise ValueError('All current background measurements must actually be complete')
    boundaries = []
    for key,suffix in (('rz_high','high'), ('rz_low','low')):
        product = products[key]; fits = product[key]; mll = product['mll_'+suffix]
        if set(fits['channels']) != {'DY2E','DY2M'} or set(fits['combined']) != GROUPS:
            raise ValueError('Unexpected RZ channel/Nb coverage')
        for group in GROUPS:
            row = fits['combined'][group]
            if row['status'] != 'complete':
                raise ValueError('Incomplete combined RZ')
            for n in ('RZ','RZ_stat'): finite_number(row[n],True)
        for channel in ('DY2E','DY2M'):
            if set(fits['channels'][channel]) != GROUPS or set(mll.get(channel,{})) != GROUPS:
                raise ValueError('Missing RZ or mll Nb group')
            for group in GROUPS:
                row = fits['channels'][channel][group]
                if row.get('status') != 'complete' or not row.get('fit_converged'):
                    raise ValueError('RZ channel fit did not converge')
                for name in ('RZ','RT','RZ_stat','RT_stat'):finite_number(row[name],True)
                if row['RZ'] == 0 or row['RT'] == 0:
                    boundaries.append(dict(mode=key,channel=channel,group=group,
                                           RZ=row['RZ'],RT=row['RT']))
                node = mll[channel][group]
                if set(node) != {'data','zll','other'}:
                    raise ValueError('Missing/extra mll component')
                edges = np.asarray(node['data']['edges'],float)
                if edges.ndim != 1 or len(edges)<2 or not np.isfinite(edges).all() or not (np.diff(edges)>0).all():
                    raise ValueError('Malformed mll edges')
                for component,leaf in node.items():
                    if leaf['edges'] != edges.tolist():raise ValueError('Inconsistent mll edges')
                    for field in ('sumw','sumw2'):
                        values = np.asarray(leaf[field],float)
                        if (values.shape != (len(edges)-1,) or not np.isfinite(values).all()
                                or (field=='sumw2' and (values<0).any())):
                            raise ValueError('Malformed mll histogram')
    photon = products['sgamma']
    if photon['recoil_edges'] != RECOIL_EDGES:
        raise ValueError('Changed Sgamma recoil bins')
    if set(photon['lowdm_Q_groups']) != GROUPS:raise ValueError('Missing low-dM Q group')
    for section in ('highdm','lowdm_families'):
        if set(photon[section]) != GROUPS:raise ValueError('Missing Sgamma Nb group')
        for group,row in photon[section].items():
            factors = [row['Q']] + [r['Sgamma'] for r in row['bins']]
            if len(row['bins']) != 5:raise ValueError('Sgamma must have five recoil bins')
            if section=='lowdm_families':factors.append(photon['lowdm_Q_groups'][group])
            for factor in factors:
                if factor['status']!='complete':raise ValueError('Incomplete photon factor')
                finite_number(factor['value'],True); finite_number(factor['stat'],True)
    for mode in ('highdm','lowdm'):
        rows = products['double_ratio'][mode]['bins']
        if len(rows)!=5 or [rows[0]['low']]+[r['high'] for r in rows] != RECOIL_EDGES:
            raise ValueError('Changed double-ratio recoil bins')
        for i,row in enumerate(rows):
            if row['low']!=RECOIL_EDGES[i] or row['status']!='complete':
                raise ValueError('Missing/discontinuous double-ratio bins')
            for key in ('z_data_over_mc','z_stat','photon_data_over_mc','photon_stat',
                        'double_ratio','double_ratio_stat','systematic'):
                finite_number(row[key],True)
    return boundaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('measurements','hists','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-validation',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    a = p.parse_args()
    source,hists,output = (internal_path(v) for v in (a.measurements,a.hists,a.output))
    hist_sha = sha256(hists)
    products = {name:read_json(source/(name+'.json.gz')) for name in PRODUCTS}
    boundaries = validate_products(products)
    year = products['rz_high']['provenance']['campaign_year']
    scope = products['rz_high']['scope']
    if str(year) not in ('2024','2025'):raise ValueError('Unsupported year')
    if scope!='full_nominal_production' and not a.allow_validation:
        raise ValueError('Full production required; validation needs explicit opt-in')
    for product in products.values():
        provenance = product['provenance']
        if (str(provenance['campaign_year']) != str(year) or provenance['hist_input_sha256']!=hist_sha
                or internal_path(ROOT/provenance['hist_input'])!=hists or product['scope']!=scope
                or not product.get('sr_data_blinded')):
            raise ValueError('Measurement year/source/scope/blinding mismatch')
    rows = [r for r in read_json(ROOT/'jsons/nominal_plot_sources.json')['sources']
            if r['target'].endswith(('/dy_measurement.py','/photon_measurement.py','/zgamma_measurement.py'))]
    if len(rows)!=3:raise ValueError('Missing renderer provenance')
    for row in rows:
        if sha256(ROOT/row['target'])!=row['target_sha256']:raise ValueError('Changed legacy renderer')
    contract = dict(year=str(year),scope=scope,hist_input_sha256=hist_sha,
        measurements={name:sha256(source/(name+'.json.gz')) for name in PRODUCTS},
        adapter_sha256=sha256(Path(__file__)),renderers={r['target']:r['target_sha256'] for r in rows})
    manifest_path = output/'plot_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['contract']!=contract or any(sha256(output/n)!=h for n,h in old['files'].items()):
            raise ValueError('Preserve divergent measured-factor figures')
        print(old['status'],'(verified existing outputs)');return
    if a.validate_only:
        print(dict(status='validated_plot_inputs_not_generated',plots=25,boundary_fits=boundaries,contract=contract));return
    if output.exists():raise FileExistsError('Preserve unfinished output; use a new directory')
    from TROTASR.workflows.renderers import dy_measurement as dy, photon_measurement as photon, zgamma_measurement as zgamma
    for module in (dy,photon,zgamma):
        module.CMS_LABEL = dict(module.CMS_LABEL,rlabel=str(year)+' (13.6 TeV)')
    output.mkdir(parents=True)
    paths=[]
    for mode,key,suffix in (('highdm','rz_high','high'),('lowdm','rz_low','low')):
        directory=output/'dy'/mode;directory.mkdir(parents=True)
        result=products[key]
        names=dy.plot_rz_nb(result[key],mode,directory)+dy.plot_rt(result[key],mode,directory)
        for corrected in (False,True):
            names+=dy.plot_mll(result['mll_'+suffix],result[key],mode,directory,corrected=corrected)
        paths.extend(directory/n for n in names)
    directory=output/'photon'
    paths.extend(map(Path,photon.plot_q(products['sgamma'],directory)))
    for mode,key in (('highdm','highdm'),('lowdm','lowdm_families')):
        paths.extend(map(Path,photon.plot_sgamma(products['sgamma'][key],mode,directory,RECOIL_EDGES)))
        paths.extend(map(Path,zgamma.plot(mode,products['double_ratio'][mode]['bins'],output/'double_ratio')))
    if len(set(paths))!=50 or any(not path.is_file() for path in paths):
        raise ValueError('Missing background measurement figures')
    files={str(path.relative_to(output)):sha256(path) for path in paths}
    plots=[]
    for path in paths:
        if path.suffix=='.pdf':
            name=path.stem+('_2025' if str(year)=='2025' else '')
            plots.append(dict(pdf=str(path.relative_to(ROOT)),png=str(path.with_suffix('.png').relative_to(ROOT)),
                an_reference='figures/zinv_measurements/'+name+'.pdf'))
    write_json(manifest_path,dict(status='generated_pending_visual_QA',contract=contract,plots=plots,files=files,
        boundary_fits=boundaries,corrected_mll_is_independent_closure=False,
        inherited_annotation_audit=dict(mll_shaded_interval_gev=[81.,101.],actual_fit_on_z_interval_gev=[71.,111.],
            renderer_changed=False,meaning='Inherited display highlight is not the current fit window; must remain explicit in final QA/report'),
        full_workflow_complete=False))


if __name__=='__main__':main()
