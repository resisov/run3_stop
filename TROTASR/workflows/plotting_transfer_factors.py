"""Plot newly measured MC transfer factors with the verbatim legacy renderer."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.sr_binning import NATIVE_EDGES

HIGH_ROUTES = ('top_llcr', 'w_llcr', 'qcd_qcdcr')
LOW_ROUTES = HIGH_ROUTES + ('zinv_gcr',)


def prepare_factors(source, config):
    """Validate and expose unsupported cells; never replace a factor by unity."""
    if source.get('schema') != 'trotasr_transfer_factors_v2':
        raise ValueError('The current NISR-pooled TF schema is required')
    if not source.get('mc_stat_only') or source.get('extra_weight_applied'):
        raise ValueError('TFs must be raw MC without RZ/Q/Sgamma weights')
    if source['highdm']['edges'] != NATIVE_EDGES:
        raise ValueError('Changed high-dM TF recoil bins')
    if (source['lowdm']['sr_binning'] != config['sr_binning']
            or source['lowdm']['cr_binning'] != config['cr_binning']):
        raise ValueError('Changed low-dM TF binning')
    categories = config['sr_binning']['category_labels']
    result, unavailable = deepcopy(source), []
    for mode,routes,labels,n in (('highdm', HIGH_ROUTES, ['Nb1', 'Nb2plus'], 8),
                                ('lowdm', LOW_ROUTES, categories, 5)):
        for route in routes:
            records = result[mode]['records'][route]
            if set(records) != set(labels):
                raise ValueError('Missing/extra TF categories: '+mode+'/'+route)
            for category,row in records.items():
                coverage = row.get('process_coverage', {})
                if mode == 'lowdm':
                    if (row['score_edges'] != config['sr_binning']['edges_by_category'][category]
                            or row['cr_score_edges'] != config['cr_binning']['score_edges']
                            or row['nb_group'] != category.split('_')[0]):
                        raise ValueError('Low-dM TF mapping does not match category/Nb')
                values = np.asarray([np.nan if x is None else x for x in row.get('transfer_factor', [None]*n)])
                errors = np.asarray([np.nan if x is None else x for x in row.get('mcstat', [None]*n)])
                valid = np.asarray(row.get('valid', [False]*n), dtype=bool)
                if values.shape != (n,) or errors.shape != (n,) or valid.shape != (n,):
                    raise ValueError('Invalid TF array length')
                finite = np.isfinite(values) & np.isfinite(errors)
                if (errors[finite] < 0).any() or (valid & ~finite).any():
                    raise ValueError('Malformed TF uncertainty or valid cell')
                valid &= finite & (values >= 0)
                if not coverage.get('numerator') or not coverage.get('denominator'):
                    valid[:] = False
                if mode == 'lowdm' and 'denominator_total' in row and row['denominator_total'] > 0:
                    denominator = np.asarray(row.get('denominator', []), dtype=float)
                    cr = np.asarray(row.get('cr_score_yield', []), dtype=float)
                    covariance = np.asarray(row.get('covariance', []), dtype=float)
                    if (denominator.shape != (5,) or cr.shape != (5,)
                            or covariance.shape != (5,5) or not np.isfinite(covariance).all()
                            or not np.allclose(denominator, row['denominator_total'], rtol=1e-12)
                            or not np.isclose(cr.sum(), row['denominator_total'], rtol=1e-12)
                            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
                            or not np.allclose(np.diag(covariance), errors**2, rtol=1e-12, atol=1e-14)):
                        raise ValueError('Low-dM TF is not the pooled integral/covariance model')
                # The original renderer skips None cells. Record every omission.
                row['transfer_factor'] = [float(v) if ok else None for v,ok in zip(values, valid)]
                row['mcstat'] = [float(v) if ok else None for v,ok in zip(errors, valid)]
                if not valid.all():
                    unavailable.append(dict(mode=mode, route=route, category=category,
                        bins_1based=(np.flatnonzero(~valid)+1).tolist(), measurement_status=row['status']))
    return result, unavailable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'hists', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--allow-validation', action='store_true')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    source_path, hist_path, output = (internal_path(v) for v in (a.input, a.hists, a.output))
    source = read_json(source_path)
    year = source.get('year')
    if year not in (2024, 2025) or not source.get('sr_data_blinded'):
        raise ValueError('Unsupported TF year or missing blinding contract')
    provenance = source['provenance']
    if (provenance['hist_input_sha256'] != sha256(hist_path)
            or str(provenance['campaign_year']) != str(year)
            or internal_path(ROOT/provenance['hist_input']) != hist_path):
        raise ValueError('TF does not belong to the current histogram input')
    if source.get('scope') != 'full_nominal_production' and not a.allow_validation:
        raise ValueError('Full production required; validation needs explicit opt-in')
    factors, unavailable = prepare_factors(source, read_json(ROOT/'gnn4lowdm/config.json'))
    if unavailable and not a.allow_partial:
        raise ValueError('Unsupported TF cells; explicit diagnostic opt-in required: '+str(unavailable))
    provenance_rows = read_json(ROOT/'jsons/nominal_plot_sources.json')['sources']
    for row in provenance_rows:
        if sha256(ROOT/row['target']) != row['target_sha256']:
            raise ValueError('Preserved TF rendering source changed')
    contract = dict(input_sha256=sha256(source_path), hist_input_sha256=provenance['hist_input_sha256'],
        year=year, scope=source['scope'], adapter_sha256=sha256(Path(__file__)),
        renderer_sources={r['target']:r['target_sha256'] for r in provenance_rows})
    manifest_path = output/'plot_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['contract'] != contract or any(sha256(output/n) != h for n,h in old['files'].items()):
            raise ValueError('Divergent TF plots must be preserved')
        print(old['status'], '(verified existing outputs)')
        return
    if a.validate_only:
        print(dict(status='validated_plot_inputs_not_generated', plots=7, unavailable=unavailable, contract=contract))
        return
    if output.exists():
        raise FileExistsError('Preserve incomplete TF outputs; choose a new directory')
    from TROTASR.workflows.renderers import transfer_factors as renderer
    renderer.CMS_LABEL = dict(renderer.CMS_LABEL, rlabel=str(year)+' (13.6 TeV)')
    output.mkdir(parents=True)
    high, low = renderer.render_factors(output, factors, 'all')
    files = {str(Path(path).relative_to(output)):sha256(path) for path in high+low}
    if len(files) != 14:
        raise ValueError('Missing expected transfer-factor figures')
    plots = []
    for path in high+low:
        if Path(path).suffix != '.pdf':
            continue
        name = Path(path).stem + ('_2025' if year == 2025 else '')
        plots.append(dict(pdf=str(Path(path).relative_to(ROOT)),
                          png=str(Path(path).with_suffix('.png').relative_to(ROOT)),
                          an_reference='figures/transfer_factors/'+name+'.pdf'))
    write_json(manifest_path, dict(status=('partial_generated_pending_visual_QA' if unavailable
        else 'generated_pending_visual_QA'), contract=contract, files=files, plots=plots,
        unavailable=unavailable, definitions=source['definitions'], covariance_preserved=True,
        full_workflow_complete=False))


if __name__ == '__main__':
    main()
