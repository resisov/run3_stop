"""Original limit renderer for accepted expected-fit results.

Axes, colors, typography, overlays and topology-aware interpolation are the
adopted settings.
Rendering never declares visual QA or the complete AN workflow successful.
"""
import argparse
import contextlib
import hashlib
import json
import math
import time
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


FULL_SCOPES = ('full_nominal_production', 'full_combined_systematic_production')


def sr_bin_counts(grid):
    saved = grid.get('sr_bins_per_year')
    if isinstance(saved, dict) and 'highdm' in saved and 'lowdm' in saved:
        return saved['highdm'], saved['lowdm']
    saved = grid.get('statistical_revision', {}).get('preserve_sr_bins')
    if isinstance(saved, (list, tuple)) and len(saved) == 2:
        return tuple(saved)
    raise ValueError('Accepted grid is missing its recorded SR-bin metadata')


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
    if payload.get('observed_sr_used') is not False or payload.get('scope') not in FULL_SCOPES:
        raise ValueError('Full-input blinded expected limits are required')
    fields = ('expected_m2', 'expected_m1', 'expected', 'expected_p1', 'expected_p2')
    for mass, record in points.items():
        stop, lsp = mass_pair(mass)
        if (record['mStop'], record['mLSP']) != (stop, lsp) or 'observed' in record:
            raise ValueError('Mass/expected-only limit contract mismatch')
        try:
            values = [float(record[field]) for field in fields]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Invalid expected quantiles') from exc
        if not all(math.isfinite(v) and v > 0. for v in values) or not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError('Invalid expected quantiles')


def canonical_analysis_root(workflow):
    """Use the accepted pipeline's explicit analysis payload, not the code root."""
    value = workflow.get('analysis_root')
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('Canonical workflow must identify its absolute analysis_root')
    return internal_path(value)


def canonical_workflow_results(grid, workflow, state_path, state_hash,
                               allow_partial=False, live_snapshot=False):
    """Read accepted pipeline receipts without replaying upstream validation."""
    required = (.025, .16, .5, .84, .975)
    saved = workflow.get('results', {})
    if not isinstance(saved, dict):
        raise ValueError('Canonical workflow results must be a point mapping')
    analysis_root = canonical_analysis_root(workflow)
    if live_snapshot:
        # Add newly accepted point receipts without reopening ROOT or producer inputs.
        from TROTASR.workflows.combine_limits import accepted_analysis_root, saved_point_result
        saved = dict(saved)
        with accepted_analysis_root(analysis_root/workflow['manifest'], state_path.parent):
            for point in grid['points']:
                key = point['model']+'/'+point['mass']
                if saved.get(key, {}).get('status') != 'complete':
                    accepted = saved_point_result(point, state_path.parent)
                    if accepted is not None:
                        saved[key] = accepted
    result, evidence = {}, {}
    for point in grid['points']:
        key = point['model']+'/'+point['mass']
        record = saved.get(key, saved.get(point['model']+':'+point['mass'], {'status':'pending'}))
        if not isinstance(record, dict):
            raise ValueError('Canonical point receipt is not a mapping: '+key)
        status = record.get('status', 'pending')
        accepted = None
        if status == 'complete':
            raw_quantiles = record.get('validation', {}).get('quantiles', {})
            try:
                quantiles = {float(q):float(value) for q,value in raw_quantiles.items()}
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid accepted quantiles: '+key) from exc
            if (len(raw_quantiles) != 5 or set(quantiles) != set(required)
                    or not all(math.isfinite(quantiles[q]) and quantiles[q] > 0 for q in required)
                    or not all(quantiles[a] < quantiles[b] for a,b in zip(required, required[1:]))):
                raise ValueError('Accepted workflow point lacks five ordered quantiles: '+key)
            if (record.get('model',point['model']) != point['model']
                    or record.get('mass',point['mass']) != point['mass']
                    or record.get('exit_code',0) != 0):
                raise ValueError('Accepted workflow point identity or exit status differs: '+key)
            output = record.get('output', record.get('root'))
            log = record.get('log', record.get('accepted_log'))
            for name, value in (('output',output),('log',log),('workspace',record.get('workspace'))):
                if not isinstance(value,str) or not value or not internal_path(analysis_root/value).is_file():
                    raise ValueError('Accepted workflow '+name+' missing: '+key)
            accepted = dict(record, output=output, accepted_log=log,
                accepted_origin=record.get('accepted_origin','canonical_limits_pipeline'))
            result[key] = accepted
        else:
            if not (allow_partial or live_snapshot) and status in ('pending','running'):
                raise ValueError('Canonical workflow point not terminal: '+key)
            result[key] = dict(record, status=status)
        evidence[key] = dict(state_sha256=None, accepted=accepted,
            workflow_state=str(state_path), workflow_state_sha256=state_hash,
            snapshot_status=status, input_source='canonical_limits_pipeline',
            inspection_scope='accepted_receipt_five_quantile_structure_and_artifact_presence')
    return result, evidence


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




def _workflow(limits_dir):
    path = internal_path(limits_dir) / 'workflow_state.json'
    raw = path.read_bytes()
    workflow = json.loads(raw)
    if workflow.get('canonical_limits_pipeline') is not True or workflow.get('observed_sr_used') is True:
        raise ValueError('Require the canonical blinded limits workflow')
    return path, workflow, hashlib.sha256(raw).hexdigest()


def verified_current_results(grid, limits_dir, allow_partial=False, live_snapshot=False):
    path, workflow, digest = _workflow(limits_dir)
    return canonical_workflow_results(grid, workflow, path, digest, allow_partial, live_snapshot)


def prepare(limits_dir, grid_manifest, allow_partial=False, live_snapshot=False):
    grid = read_json(internal_path(grid_manifest))
    if (grid.get('status') != 'cards_ready' or grid.get('scope') not in FULL_SCOPES
            or grid.get('sr_data_blinded') is not True):
        raise ValueError('Require a completed full-input blinded card grid')
    if live_snapshot and (not allow_partial or grid['scope'] != 'full_combined_systematic_production'
                          or not grid.get('run3_systematics')):
        raise ValueError('Live snapshot requires an explicit partial six-era view')
    workflow_path, workflow, digest = _workflow(limits_dir)
    grid = dict(grid, canonical_limits_pipeline=True,
                _canonical_analysis_root=str(canonical_analysis_root(workflow)))
    grid['points'] = [point for point in grid['points']
                      if 600 <= mass_pair(point['mass'])[0] <= 1800
                      and 1 <= mass_pair(point['mass'])[1] <= 1800]
    grid['expected_points_by_model'] = {
        model: sum(point['model'] == model for point in grid['points']) for model in MODELS}
    sr_bin_counts(grid)
    results, evidence = canonical_workflow_results(
        grid, workflow, workflow_path, digest, allow_partial, live_snapshot)
    evidence_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    records = {}
    for model in MODELS:
        masses = [point['mass'] for point in grid['points'] if point['model'] == model]
        payload = collect(grid['points'], results, model, grid['scope'])
        validate_collected(payload, masses, allow_partial)
        records[model] = dict(
            payload=renderer.select_limit_range(payload, masses), full_payload=payload,
            input_sha256=evidence_digest, input=str(workflow_path.relative_to(ROOT)),
            requested_masses=masses, full_grid_status=payload['status'],
            full_grid_missing=payload['missing_points'], full_grid_point_count=len(masses),
            out_of_plot_range=[])
        if live_snapshot:
            from collections import Counter
            records[model]['snapshot_counts'] = dict(Counter(results[model+'/'+m]['status'] for m in masses))
    return grid, records, sha256(Path(renderer.__file__))


def render(limits_dir, grid_manifest, output, allow_partial=False, live_snapshot=False):
    output = internal_path(output)
    snapshot_started = time.time()
    grid, records, renderer_hash = prepare(limits_dir, grid_manifest, allow_partial, live_snapshot)
    eras = tuple(grid['years'])
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
    contract = dict(grid_sha256=sha256(grid_manifest),
        collected_sha256={m: r['input_sha256'] for m, r in records.items()},
        renderer_sha256=renderer_hash, adapter_sha256=sha256(Path(__file__)),
        config_sha256=sha256(ROOT / 'stats/contour_config.json'), runtime=renderer.runtime_versions(),
        allow_partial=allow_partial, sr_bins_per_year=dict(highdm=high_bins, lowdm=low_bins),
        sr_merge=grid.get('sr_merge'), canonical_limits_pipeline=True,
        analysis_root=grid['_canonical_analysis_root'])
    if grid.get('run3_systematics'):
        contract.update(years=list(eras), run3_systematics=grid['run3_systematics'],
            systematic_scope=grid['systematic_scope'], preliminary=True,
            analysis_label=None, status_annotations='report_and_manifest_only')
    if live_snapshot:
        contract.update(live_snapshot=True,snapshot_started=snapshot_started,snapshot_finished=time.time(),
            snapshot_counts={m:r['snapshot_counts'] for m,r in records.items()},
            claim='In-progress grid; only complete five-quantile fits; missing support is masked',
            fit_or_histogram_recalculation=False)
    receipt = output / 'plot_manifest.json'
    if output.exists():
        old = read_json(receipt)
        # Source checksums record provenance; reuse depends on the plotted inputs.
        exclude = ('adapter_sha256', 'renderer_sha256')
        if ({k: v for k, v in old['contract'].items() if k not in exclude}
                != {k: v for k, v in contract.items() if k not in exclude}
                or not old.get('plots')):
            raise FileExistsError('Preserve divergent/incomplete contour directory')
        for name, digest in old['files'].items():
            if sha256(output / name) != digest:
                raise ValueError('Completed contour artifact changed')
        return old
    output.mkdir(parents=True)
    plots = []
    write_json(receipt, dict(status='rendering', contract=contract, plots=[], full_workflow_complete=False))
    from TROTASR.utils.paths import analysis_era
    analysis_root = Path(grid['_canonical_analysis_root'])
    luminosities = [float(read_json(analysis_root / 'estimations' /
                    ('normalization_'+analysis_era(year)+'.json.gz'))['luminosity_fb'])
                    for year in eras]
    if not all(math.isfinite(value) and value > 0 for value in luminosities):
        raise ValueError('Invalid recorded normalization luminosity')
    luminosity_label = r'%.2f fb$^{-1}$ (13.6 TeV)' % sum(luminosities)
    for model in MODELS:
        record = records[model]
        payload = annotate_expected_grid(record['payload'], theory)
        name = 'expected_limit_%s_%s_highdm%d_lowdm%d_xsec_topology_thermal_relic' % (
            model.lower(), '_'.join(eras), high_bins, low_bins)
        if live_snapshot:
            name += '_in_progress'
        png = output / (name + '.png')
        diagnostics = {}
        if len(payload['points']) < 4:
            continue
        with supported_contours(record['requested_masses'], set(payload['points']), diagnostics):
            complete = renderer.plot_contour(payload, png, run2_contours=refs[model],
                luminosity_label=luminosity_label, analysis_label=None,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limits', required=True, type=Path)
    p.add_argument('--grid', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sensitivity-map', action='store_true',
                   help='Render saved group-freeze results without inference')
    p.add_argument('--allow-partial', action='store_true',
                   help='Diagnostic only; never a completed contour grid')
    p.add_argument('--live-snapshot', action='store_true', help='Read-only view; requires --allow-partial')
    a = p.parse_args()
    if a.live_snapshot and (not a.allow_partial or a.sensitivity_map):
        p.error('--live-snapshot requires a partial expected-limit contour')
    if a.sensitivity_map:
        from TROTASR.workflows.renderers.sensitivity_map import render as render_sensitivity
        result = render_sensitivity(a.limits, a.grid, a.output)
    else:
        result = render(a.limits, a.grid, a.output, a.allow_partial, a.live_snapshot)
    print(result['status'])


if __name__ == '__main__':
    main()
