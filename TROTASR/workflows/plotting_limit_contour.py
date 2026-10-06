"""Original limit renderer with internal inputs and explicit coverage checks.

Axes, colors, typography, overlays and topology-aware interpolation are the
adopted legacy settings. Only the input boundary and SR-bin identity change.
Rendering never declares visual QA or the complete AN workflow successful.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.renderers import postprocess_limits as renderer
from TROTASR.utils.renderers.signal_theory import annotate_expected_grid, profiled_theory
from TROTASR.utils.statistical_templates import load_theory, json_ready
from TROTASR.workflows.signal_grid_inputs import MODELS, mass_pair
from TROTASR.workflows.combine_limits import collect
from TROTASR.utils.sr_merge import load_merge


def sr_bin_counts(grid):
    high = load_merge(grid['sr_merge'])['bins_per_year'] if grid.get('sr_merge') else 162
    if grid.get('scope')=='full_combined_systematic_production':
        from TROTASR.workflows.combine_limits import verify_combined_grid
        verify_combined_grid(grid)
        return high,30
    if grid.get('bins') != 540 + 2 * (high - 162):
        raise ValueError('Grid bin count and explicit SR projection differ')
    return high, 30


def validate_collected(payload, masses, allow_partial=False):
    points = payload.get('points', {})
    missing = payload.get('missing_points', [])
    if (set(points) & set(missing) or len(set(missing)) != len(missing)
        or set(points) | set(missing) != set(masses)
        or payload.get('requested_point_count') != len(masses)
        or payload.get('collected_point_count') != len(points)):
        raise ValueError('Contour point accounting does not match the full requested grid')
    status = 'complete' if not missing else 'partial' if points else 'no_combine_outputs'
    if payload.get('status') != status or (missing and not allow_partial):
        raise ValueError('Incomplete limits; a partial diagnostic plot requires an explicit option')
    if payload.get('observed_sr_used') is not False or payload.get('scope') not in ('full_nominal_production','full_combined_systematic_production'):
        raise ValueError('Full-input blinded expected limits are required')
    fields = ('expected_m2', 'expected_m1', 'expected', 'expected_p1', 'expected_p2')
    for mass, record in points.items():
        stop, lsp = mass_pair(mass)
        if (record['mStop'], record['mLSP']) != (stop, lsp) or 'observed' in record:
            raise ValueError('Mass/expected-only limit contract mismatch')
        values = [float(record[f]) for f in fields]
        if not all(math.isfinite(v) and v > 0. for v in values) or not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError('Invalid expected quantiles')


def verified_current_results(grid, limits_dir):
    """Exact saved attempts in execution-history order; no strongest-fit choice."""
    from TROTASR.workflows.combine_limits import IntegrityCache, verify_card
    from TROTASR.workflows.continue_limit_fits import verify_attempt
    from TROTASR.workflows.diagnostic_common import command_key
    cache=IntegrityCache();result={};evidence={}
    recovery=limits_dir/'numerical_all_unfinished_20260930'
    if (recovery/'contract.json').exists():
        contract=read_json(recovery/'contract.json')
        if (contract['primary_contract_sha256']!=sha256(limits_dir/'contract.json') or
                contract['manifest_sha256']!=read_json(limits_dir/'contract.json')['manifest_sha256']):
            raise ValueError('Recovery belongs to a different model')
        for name,digest in contract['code'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Recovery source changed')
        terminal=read_json(recovery/'state.json')
        if terminal.get('pid') or terminal['status'] not in ('all_verified_pending_reconciliation','needs_attention'):
            raise ValueError('Recovery is not terminal')
    for point in grid['points']:
        verify_card(point,cache);key=point['model']+'/'+point['mass']
        path=limits_dir/key/'state.json';state=read_json(path)
        if state.get('pid') or state['status'] not in ('complete','failed'):
            raise ValueError('Primary point not terminal: '+key)
        histories=[('primary',state.get('attempts',[]))]
        if state['status']=='complete':
            histories=[('primary',[a for a in state['attempts'] if a.get('output')==state['output']])]
            if len(histories[0][1])!=1:raise ValueError('Ambiguous accepted primary attempt')
        else:
            retry_path=recovery/key/'state.json'
            if retry_path.exists():
                retry=read_json(retry_path);cp=retry_path.with_name('contract.json');c=read_json(cp)
                if retry.get('pid') or retry['contract_sha256']!=sha256(cp) or c['point']!=point or c['code']!=contract['code']:
                    raise ValueError('Unverified point recovery contract')
                for name,digest in c['protected_states'].items():
                    if sha256(ROOT/name)!=digest:raise ValueError('Protected fit state changed')
                if cache(limits_dir/key/'workspace.root')!=c['workspace_sha256']:
                    raise ValueError('Recovery workspace changed')
                histories.append(('numerical_all_unfinished_20260930',retry['attempts']))
        accepted=None
        for origin,attempts in histories:
            for attempt in attempts:
                if attempt.get('exit_code')!=0:continue
                log=internal_path(ROOT/attempt['log']);product=internal_path(ROOT/attempt['output'])
                if (product!=log.parent/('higgsCombine_'+point['mass']+'.AsymptoticLimits.mH120.root') or
                        read_json(log.parent/'validation.json')!=attempt or sha256(log)!=attempt['log_sha256']):
                    raise ValueError('Saved attempt/log identity changed')
                if not product.exists():continue
                if cache(product)!=attempt['output_sha256']:raise ValueError('Saved ROOT changed')
                if origin!='primary':
                    receipt=read_json(recovery/key/'command_receipts/commands'/
                        (command_key(attempt['command'],log.parent)+'.json'))
                    if (receipt['status']!='complete' or receipt['exit_code']!=0 or receipt.get('pid') or
                            receipt['command']!=attempt['command'] or receipt['products'].get(attempt['output'])!=attempt['output_sha256']):
                        raise ValueError('Recovery execution receipt changed')
                passed,validation=verify_attempt(point,attempt)
                if passed:
                    accepted=dict(status='complete',validation=validation,output=attempt['output'],
                        output_sha256=attempt['output_sha256'],accepted_origin=origin,
                        accepted_log=attempt['log'],log_sha256=attempt['log_sha256'])
                    break
            if accepted:break
        if state['status']=='complete' and (not accepted or accepted['output_sha256']!=state['output_sha256']):
            raise ValueError('Previously accepted fit failed revalidation')
        result[key]=accepted or state
        evidence[key]=dict(state_sha256=sha256(path),accepted=accepted)
    return result,evidence


def support_mask(masses,available):
    import numpy as np
    from scipy.spatial import Delaunay
    coords=np.asarray([mass_pair(m) for m in masses],float)
    tri=Delaunay(coords);known=np.asarray([m in available for m in masses],bool)
    good=known[tri.simplices].all(axis=1)
    def accepted(x,y):
        x,y=np.broadcast_arrays(x,y)
        simplex=tri.find_simplex(np.column_stack((x.ravel(),y.ravel())))
        mask=np.zeros(len(simplex),bool);inside=simplex>=0;mask[inside]=good[simplex[inside]]
        return mask.reshape(x.shape)
    return accepted,dict(requested=len(masses),verified=int(known.sum()),
        triangles=len(good),verified_triangles=int(good.sum()),
        physical_coordinates=['mStop','mLSP'],policy='three verified vertices in full requested mass grid')


@contextlib.contextmanager
def supported_contours(masses,available,diagnostics):
    import numpy as np
    from matplotlib.axes import Axes
    accepted,coverage=support_mask(masses,available)
    originals=(Axes.contour,Axes.contourf)
    def guard(original):
        def draw(self,x,y,z,*args,**kwargs):
            x,y=np.asarray(x),np.asarray(y)
            if x.ndim==1 and y.ndim==1:x,y=np.meshgrid(x,y)
            shown=np.ma.array(z,mask=np.ma.getmaskarray(z)|~accepted(x,y))
            kwargs['corner_mask']=False
            result=original(self,x,y,shown,*args,**kwargs)
            if original.__name__=='contour':
                for level in result.allsegs:
                    for vertices in level:
                        if len(vertices) and not accepted(vertices[:,0],vertices[:,1]).all():
                            raise ValueError('Contour entered an unsupported mass region')
            return result
        return draw
    Axes.contour,Axes.contourf=map(guard,originals)
    try:yield
    finally:
        Axes.contour,Axes.contourf=originals
        diagnostics['support_mask']=coverage


def prepare(limits_dir, grid_manifest, allow_partial=False):
    limits_dir, grid_manifest = internal_path(limits_dir), internal_path(grid_manifest)
    grid = read_json(grid_manifest)
    sr_bin_counts(grid)
    contract = read_json(limits_dir / 'contract.json')
    state = read_json(limits_dir / 'state.json')
    if grid['status'] != 'cards_ready' or grid['scope'] not in ('full_nominal_production','full_combined_systematic_production'):
        raise ValueError('Require a completed full-input card grid')
    if contract['manifest_sha256'] != sha256(grid_manifest):
        raise ValueError('Limits belong to a different card grid')
    if state['status'] not in ('fits_complete_pending_plots', 'needs_attention'):
        raise ValueError('The fit controller is not finished')
    if state['status'] != 'fits_complete_pending_plots' and not allow_partial:
        raise ValueError('Failed fits remain; do not label a partial contour complete')
    expected = next(f['sha256'] for f in read_json(ROOT / 'jsons/sources.json')['files']
                    if f['target'] == 'utils/renderers/postprocess_limits.py')
    if sha256(Path(renderer.__file__)) != expected:
        raise ValueError('Original limit renderer bytes changed')
    records = {}
    combined=grid['scope']=='full_combined_systematic_production'
    results,evidence=verified_current_results(grid,limits_dir) if combined else (state['results'],{})
    for model in MODELS:
        masses = [p['mass'] for p in grid['points'] if p['model'] == model]
        if len(masses) != grid['expected_points_by_model'][model]:
            raise ValueError('Requested signal grid was reduced')
        path = limits_dir / model / 'expected_limits.json'
        payload = collect(grid['points'],results,model,grid['scope']) if combined else read_json(path)
        validate_collected(payload, masses, allow_partial)
        regenerated = collect(grid['points'], results, model, grid['scope'])
        if regenerated != payload:
            raise ValueError('Collected quantiles no longer match fit-controller evidence')
        selected, outside = [], []
        for mass in masses:
            stop, lsp = mass_pair(mass)
            (selected if 600 <= stop <= 1800 and 1 <= lsp <= 1800 else outside).append(mass)
        records[model] = dict(payload=renderer.select_limit_range(payload, selected),
            input_sha256=(hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest() if combined else sha256(path)),
            input=str(path.relative_to(ROOT)),requested_masses=selected,
            full_grid_status=payload['status'], full_grid_missing=payload['missing_points'],
            full_grid_point_count=len(masses), out_of_plot_range=outside)
    return grid, records, expected


def render(limits_dir, grid_manifest, output, allow_partial=False):
    output = internal_path(output)
    grid, records, renderer_hash = prepare(limits_dir, grid_manifest, allow_partial)
    high_bins, low_bins = sr_bin_counts(grid)
    config = read_json(ROOT / 'stats/contour_config.json')
    refs = {name: internal_path(ROOT / spec['path']) for name, spec in config['references'].items()}
    for name, path in refs.items():
        if sha256(path) != config['references'][name]['sha256']:
            raise ValueError('Changed contour reference: ' + name)
    theory_path = internal_path(ROOT / config['theory']['path'])
    if sha256(theory_path) != config['theory']['sha256']:
        raise ValueError('Changed stop cross-section table')
    theory = profiled_theory(load_theory(theory_path))
    cross_sections = renderer.load_stop_pair_xsecs(theory_path)
    thermal = renderer.load_thermal_relic_contour(refs['thermal_relic'])
    runtime = renderer.runtime_versions()  # Same recorded-runtime policy as the legacy runner.
    contract = dict(grid_sha256=sha256(grid_manifest),
        collected_sha256={m: r['input_sha256'] for m, r in records.items()},
        renderer_sha256=renderer_hash, adapter_sha256=sha256(Path(__file__)),
        config_sha256=sha256(ROOT / 'stats/contour_config.json'), runtime=runtime,
        allow_partial=allow_partial, sr_bins_per_year=dict(highdm=high_bins,lowdm=low_bins),
        sr_merge=grid.get('sr_merge'))
    receipt = output / 'plot_manifest.json'
    if output.exists():
        old = read_json(receipt)
        if old['contract'] != contract or not old.get('plots'):
            raise FileExistsError('Preserve divergent/incomplete contour directory')
        for name, digest in old['files'].items():
            if sha256(output / name) != digest:
                raise ValueError('Completed contour artifact changed')
        return old
    output.mkdir(parents=True)
    plots = []
    write_json(receipt, dict(status='rendering', contract=contract, plots=[], full_workflow_complete=False))
    for model in MODELS:
        record = records[model]
        payload = annotate_expected_grid(record['payload'], theory)
        name = 'expected_limit_%s_2024_2025_highdm%d_lowdm%d_xsec_topology_thermal_relic' % (model.lower(),high_bins,low_bins)
        png = output / (name + '.png')
        diagnostics = {}
        if len(payload['points'])<4:continue
        with supported_contours(record['requested_masses'],set(payload['points']),diagnostics):
            complete = renderer.plot_contour(payload, png, run2_contours=refs[model],
                luminosity_label=renderer.LUMINOSITY_LABELS['2024_2025'], analysis_label=None,
                x_min=600., x_max=1800., y_min=1., y_max=1800.,
                decay_label=renderer.DECAY_LABELS[model], color_field='xsec', stop_pair_xsecs=cross_sections,
                mask_offshell=False, run2_compressed_contours=refs.get(model + '_compressed'),
                topology=model, interpolation_mode='topology-aware', interpolation_diagnostics=diagnostics,
                thermal_relic=thermal)
        if not complete or not png.is_file() or not png.with_suffix('.pdf').is_file():
            raise RuntimeError('Original contour renderer failed: ' + model)
        write_json(output / (model + '_expected_limits_with_signal_theory.json'), payload)
        write_json(output / (model + '_interpolation.json'), json_ready(diagnostics))
        plots.append(dict(model=model, png=png.name, pdf=png.with_suffix('.pdf').name,
            plotted_grid_status=payload['status'], plotted_points=payload['collected_point_count'],
            plotted_missing=payload['missing_points'], full_grid_status=record['full_grid_status'],
            full_grid_missing=record['full_grid_missing'], full_grid_point_count=record['full_grid_point_count'],
            out_of_plot_range=record['out_of_plot_range'],
            status='generated_pending_visual_QA', external_theory_shift_applied=False))
    result = dict(status='generated_pending_visual_QA', contract=contract, plots=plots,
        files={p.name: sha256(p) for p in output.iterdir() if p.is_file() and p != receipt},
        sr_data_blinded=True, full_workflow_complete=False)
    write_json(receipt, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limits', required=True, type=Path)
    p.add_argument('--grid', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-partial', action='store_true', help='Audited diagnostic only; never a completed contour grid')
    a = p.parse_args()
    print(render(a.limits, a.grid, a.output, a.allow_partial)['status'])
