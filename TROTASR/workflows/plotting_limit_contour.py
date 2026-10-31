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
from TROTASR.utils.sr_merge import load_merge
from TROTASR.workflows.diagnostic_common import record_years, require_revised_record


def sr_bin_counts(grid):
    if grid.get('canonical_limits_pipeline') is True:
        saved = grid.get('sr_bins_per_year')
        if isinstance(saved, dict) and 'highdm' in saved and 'lowdm' in saved:
            return saved['highdm'], saved['lowdm']
        saved = grid.get('statistical_revision', {}).get('preserve_sr_bins')
        if isinstance(saved, (list, tuple)) and len(saved) == 2:
            return tuple(saved)
        raise ValueError('Accepted grid is missing its recorded SR-bin metadata')
    if grid.get('scope') == 'sr_only_nominal_mcstat_comparison':
        expected = {'baseline': dict(highdm=118,lowdm=30,mediumdm=0),
                    'low': dict(highdm=0,lowdm=30,mediumdm=0),
                    'high': dict(highdm=118,lowdm=0,mediumdm=0),
                    'medium': dict(highdm=0,lowdm=0,mediumdm=20),
                    'combined': dict(highdm=118,lowdm=30,mediumdm=20)}
        if grid.get('medium_candidate') is not None:
            from TROTASR.workflows.build_nominal_grid import candidate_bin_count
            if grid.get('variant') not in ('medium','combined'):
                raise ValueError('A medium candidate cannot redefine the fixed high/low plot')
            expected[grid['variant']]['mediumdm'] = candidate_bin_count(grid['medium_candidate'])
        if grid.get('medium_test_comparison') is not None:
            from TROTASR.workflows.build_nominal_grid import test_comparison_bin_count
            if grid.get('medium_candidate') is not None or grid.get('variant') not in ('medium','combined'):
                raise ValueError('Frozen test cannot redefine fixed high/low or validation provenance')
            expected[grid['variant']]['mediumdm'] = test_comparison_bin_count(grid['medium_test_comparison'])
        if (grid.get('sr_bins_per_year') != expected.get(grid.get('variant'))
                or grid['bins'] != 2*sum(grid['sr_bins_per_year'].values())):
            raise ValueError('Comparison SR-bin identity mismatch')
        return grid['sr_bins_per_year']['highdm'],grid['sr_bins_per_year']['lowdm']
    high = load_merge(grid['sr_merge'])['bins_per_year'] if grid.get('sr_merge') else 162
    if grid.get('scope') in ('full_combined_systematic_production', 'full_run3_hybrid_production'):
        from TROTASR.workflows.combine_limits import verify_combined_grid
        verify_combined_grid(grid)
        return high,30
    eras = record_years(grid)
    revised = require_revised_record(grid)
    if grid.get('bins') != len(eras) * ((258 if revised else 270) + high - 162):
        raise ValueError('Grid bin count and explicit SR projection differ')
    return high, 30


def require_same_medium_candidate(grids):
    """Do not overlay a new combined result with another medium definition."""
    definitions = [(grid.get('medium_candidate'),grid.get('medium_test_comparison')) for grid in grids
                   if grid.get('variant') in ('medium', 'combined')]
    if definitions and any(value != definitions[0] for value in definitions[1:]):
        raise ValueError('Medium-only and combined contours use different candidate definitions')


def require_fixed_baseline_comparison(after, before):
    """Explicit historical comparison: only medium inputs may have changed."""
    if any(g.get('variant') != 'combined' or
           g.get('scope') != 'sr_only_nominal_mcstat_comparison' for g in (after, before)):
        raise ValueError('Before/after comparison requires two nominal combined grids')
    for key in ('parent_grid_sha256', 'treatment', 'auto_mc_stats',
                'mc_statistics_assumption', 'sr_merge'):
        if after.get(key) != before.get(key):
            raise ValueError('Before/after statistical definition changed: '+key)
    fixed = []
    for grid in (after, before):
        files = {name:digest for name,digest in grid['files'].items()
                 if Path(name).name in {'template_'+region+'_'+year+'.root'
                     for region in ('highdm','lowdm') for year in ('2024','2025')}}
        if len(files) != 4:
            raise ValueError('Before/after comparison requires four fixed high/low templates')
        fixed.append(files)
    if fixed[0] != fixed[1]:
        raise ValueError('Before/after high/low templates changed')
    if ({(p['model'],p['mass']) for p in after['points']} !=
            {(p['model'],p['mass']) for p in before['points']}):
        raise ValueError('Before/after requested mass grids differ')
    return fixed[0]


def require_fixed_test_comparison(after, before, baseline, baseline_sha256):
    """Same test identities/weights and fixed HL; no test-based ranking."""
    fixed = require_fixed_baseline_comparison(after, before)
    if (baseline.get('variant') != 'baseline'
            or baseline.get('scope') != 'sr_only_nominal_mcstat_comparison'):
        raise ValueError('Every test comparison requires the fixed w/o medium SR baseline')
    baseline_as_combined = dict(baseline, variant='combined')
    require_fixed_baseline_comparison(after, baseline_as_combined)
    provenance = []
    for grid, variant in ((before, 'before'), (after, 'after')):
        spec = grid.get('medium_test_comparison')
        if (not spec or grid.get('medium_candidate') is not None
                or spec.get('candidate_id') != variant
                or spec.get('fixed_baseline_manifest_sha256') != baseline_sha256):
            raise ValueError('Frozen before/after test contract and fixed baseline are required')
        from TROTASR.workflows.build_nominal_grid import test_comparison_bin_count, TEST_COMPARISON_IDENTITIES
        test_comparison_bin_count(spec)
        identity = grid.get('medium_test_provenance', {})
        if (set(identity) != {'2024','2025'} or any(
                not isinstance(identity[y].get(k), str) or len(identity[y][k]) != 64
                for y in identity for k in TEST_COMPARISON_IDENTITIES)):
            raise ValueError('Missing physical test-event and weight identities')
        provenance.append(identity)
    if (provenance[0] != provenance[1] or
            before['medium_test_comparison']['comparison_freeze_sha256'] !=
            after['medium_test_comparison']['comparison_freeze_sha256']):
        raise ValueError('Before/after must use identical physical test events, weights and freeze')
    return dict(fixed_high_low_templates=fixed, medium_test_provenance=provenance[0],
                baseline_manifest_sha256=baseline_sha256)


VALIDATION_REFERENCE_ID = 'tr_sig005_latest_9c05c80dfe80'
VALIDATION_REFERENCE_SHA256 = '7922d1795fd0a3358ec7c9d44e95a6f40cc000e4e9551d697f113e3d3389fd65'


def require_fixed_validation_comparison(after, before, baseline, baseline_sha256):
    """C01 score edges only; neither a different partition nor a different model."""
    from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
    from TROTASR.workflows.combine_limits import verify_comparison_grid
    for grid in (after, before, baseline):
        if grid.get('medium_test_comparison') is not None or grid.get('evaluation_partition', 'validation') != 'validation':
            raise ValueError('Validation score comparison cannot read test inputs')
        verify_comparison_grid(grid)
    if baseline.get('variant') != 'baseline' or baseline_sha256 != FIXED_BASELINE_SHA256:
        raise ValueError('Validation comparison requires the fixed w/o medium SR baseline')
    fixed = require_fixed_baseline_comparison(after, before)
    require_fixed_baseline_comparison(after, dict(baseline, variant='combined'))
    old, new = before.get('medium_candidate'), after.get('medium_candidate')
    if (not old or not new or old.get('candidate_id') != VALIDATION_REFERENCE_ID
            or old.get('contract_sha256') != VALIDATION_REFERENCE_SHA256
            or old.get('fixed_baseline_manifest_sha256') != baseline_sha256
            or new.get('fixed_baseline_manifest_sha256') != baseline_sha256):
        raise ValueError('The authenticated original C01 reference is required')
    if (old['model'] != new['model'] or old['model'].get('id') != 'tr_sig005_latest'
            or old['model'].get('epoch') != 17):
        raise ValueError('Score-only comparison must retain the C01 epoch-17 model')
    categories = [c['category_definition'] for c in (old, new)]
    partitions = [[{k:v for k,v in group.items() if k != 'score_edges'}
                   for group in definition['groups']] for definition in categories]
    if (partitions[0] != partitions[1] or len(partitions[0]) != 4
            or any(d['nbins'] != 20 for d in categories)):
        raise ValueError('Score-only comparison must retain four physical categories and twenty bins')
    return dict(fixed_high_low_templates=fixed, baseline_manifest_sha256=baseline_sha256,
                model=old['model'], physical_partitions=partitions[0])


def load_validation_plot_inputs(spec):
    """Use the builder's authenticated compact exports, never test-event files."""
    from TROTASR.workflows.build_nominal_grid import load_medium_candidate, read_medium_validation, CUTBASED_SCHEMA,MET_CATEGORY_SCHEMA
    if spec.get('schema') in (CUTBASED_SCHEMA,MET_CATEGORY_SCHEMA):
        raise ValueError('All-MC cut-based inputs are not NN-validation exports')
    if load_medium_candidate(spec['contract_path']) != spec:
        raise ValueError('Validation candidate input contract changed')
    return {year:read_medium_validation(Path(spec['contract_path']).parent/record['path'], spec)[0]
            for year,record in spec['inputs'].items()}


def require_same_validation_inputs(before, after):
    """Rebinning may change sums at floating-point precision, not events/weights."""
    from TROTASR.workflows.build_nominal_grid import CANDIDATE_PARTITION
    if set(before) != {'2024','2025'} or set(after) != set(before):
        raise ValueError('Both validation years are required')
    identities = {}
    for year in sorted(before):
        old, new = before[year], after[year]
        provenance = old['provenance']
        score = provenance.get('score_provenance', {})
        if (provenance != new['provenance'] or any(provenance.get(k) != v for k,v in CANDIDATE_PARTITION.items())
                or not isinstance(provenance.get('validation_scores_sha256'), str)
                or len(provenance['validation_scores_sha256']) != 64
                or score.get('partition') != 'validation'
                or score.get('checkpoint_sha256') != old['model']['checkpoint_sha256']
                or any(not isinstance(score.get(k), str) or len(score[k]) != 64
                       for k in ('cache_manifest_sha256','contract_sha256'))):
            raise ValueError('Validation score/event/weight provenance changed')
        data = [p['years'][year] for p in (old, new)]
        metadata = [{k:v for k,v in d.items() if k != 'samples'} for d in data]
        if (metadata[0] != metadata[1] or metadata[0].get('partition') != 'validation'
                or metadata[0].get('weight_scale') != 10. or metadata[0].get('variance_scale') != 100.
                or metadata[0].get('luminosity_fb') != {'2024':109.82,'2025':110.84}[year]
                or old['background_processes'] != new['background_processes']
                or old['signal_names'] != new['signal_names']
                or set(data[0]['samples']) != set(data[1]['samples'])):
            raise ValueError('Validation population, sample inventory or normalization changed')
        maximum_difference = 0.
        for sample in data[0]['samples']:
            for field in ('entries','sumw','sumw2'):
                for group in range(4):
                    totals = [math.fsum(d['samples'][sample][field][5*group:5*group+5]) for d in data]
                    if (not all(math.isfinite(v) for v in totals) or
                            (totals[0] != totals[1] if field == 'entries' else
                             not math.isclose(*totals, rel_tol=1e-10, abs_tol=1e-7))):
                        raise ValueError('Validation physical-category weights changed: '+year+'/'+sample+'/'+field)
                    maximum_difference = max(maximum_difference, abs(totals[0]-totals[1]))
        identities[year] = dict(provenance=provenance, normalization=metadata[0],
            sample_count=len(data[0]['samples']), physical_category_totals_preserved=True,
            maximum_absolute_sum_difference=maximum_difference, relative_tolerance=1e-10, absolute_tolerance=1e-7)
    return identities


def require_fixed_cutbased_comparison(after, baseline, baseline_sha256, *, allow_met_categories=False):
    """Explicit all-MC physical cuts, compared only with their fixed HL baseline."""
    from TROTASR.workflows.build_nominal_grid import CUTBASED_SCHEMA,MET_CATEGORY_SCHEMA,FIXED_BASELINE_SHA256
    from TROTASR.workflows.combine_limits import verify_comparison_grid
    for grid in (after,baseline):verify_comparison_grid(grid)
    candidate=after.get('medium_candidate') or {}
    schemas=(CUTBASED_SCHEMA,MET_CATEGORY_SCHEMA) if allow_met_categories else (CUTBASED_SCHEMA,)
    if (candidate.get('schema') not in schemas or after.get('variant')!='combined'
            or after.get('medium_test_comparison') is not None
            or baseline.get('variant')!='baseline' or baseline_sha256!=FIXED_BASELINE_SHA256
            or candidate.get('fixed_baseline_manifest_sha256')!=baseline_sha256):
        raise ValueError('Cut-based comparison requires all-MC combined and the fixed w/o medium SR baseline')
    fixed=require_fixed_baseline_comparison(after,dict(baseline,variant='combined'))
    population_key='parent_populations' if candidate['schema']==MET_CATEGORY_SCHEMA else 'populations'
    return dict(fixed_high_low_templates=fixed,baseline_manifest_sha256=baseline_sha256,
        **{population_key:candidate[population_key]},feature_definitions=candidate['feature_definitions'])


def load_cutbased_plot_inputs(spec):
    from TROTASR.workflows.build_nominal_grid import load_medium_candidate,read_medium_cutbased,cutbased_candidate_bin_count
    cutbased_candidate_bin_count(spec)
    if load_medium_candidate(spec['contract_path'])!=spec:
        raise ValueError('Cut-based candidate input contract changed')
    return {year:read_medium_cutbased(Path(spec['contract_path']).parent/record['path'],spec)[0]
            for year,record in spec['inputs'].items()}


def require_fixed_cutbased_preview(cut, gnn, baseline, baseline_sha256):
    """Explicit complete candidates with different MC populations, not a ranking."""
    from TROTASR.workflows.build_nominal_grid import CANDIDATE_SCHEMA,TEST_COMPARISON_SCHEMA
    from TROTASR.workflows.combine_limits import verify_comparison_grid
    identities = require_fixed_cutbased_comparison(cut, baseline, baseline_sha256,allow_met_categories=True)
    verify_comparison_grid(gnn)
    for grid in (cut, gnn, baseline):sr_bin_counts(grid)
    test=gnn.get('medium_test_comparison')
    candidate = test or gnn.get('medium_candidate') or {}
    if (candidate.get('schema') != (TEST_COMPARISON_SCHEMA if test else CANDIDATE_SCHEMA)
            or gnn.get('variant') != 'combined'
            or gnn.get('evaluation_partition', 'validation') != ('test' if test else 'validation')
            or candidate.get('fixed_baseline_manifest_sha256') != baseline_sha256):
        raise ValueError('Preview GNN overlay requires an authenticated validation candidate with the fixed baseline')
    require_fixed_baseline_comparison(cut, gnn)
    return identities


def load_test_plot_inputs(spec):
    """Authenticate completed test templates; never select or rebin them."""
    from TROTASR.workflows.build_nominal_grid import load_medium_test_comparison,read_medium_test_comparison
    if load_medium_test_comparison(spec['contract_path'])!=spec:
        raise ValueError('Frozen test input contract changed')
    return {year:read_medium_test_comparison(Path(spec['contract_path']).parent/record['path'],spec)[0]
            for year,record in spec['inputs'].items()}


CUTBASED_PREVIEW_CAVEAT = (
    'Intermediate complete-candidate preview, not a final selection or pool ranking. '
    'The GNN medium SR uses 10% validation MC with weights multiplied by 10 and variances by 100; '
    'the cut-based medium SR uses 100% MC with weights and variances unchanged. '
    'This is not a common-MC, method-only comparison. High/low SR inputs are identical. '
    'SR data remain blinded; only nominal MC and autoMCStats are included.')

TEST_CUTBASED_PREVIEW_CAVEAT = (
    'Final evaluation of the validation-frozen GNN model/categories/score edges on historical 70% test MC; '
    'weights are divided by 0.7 and variances by 0.49. No test selection or rebinning. '
    'The S2 cut-based medium SR uses 100% MC with unchanged weights and variances. '
    'This is not a common-MC method-only comparison or an untouched independent test. '
    'High/low SR inputs are identical; SR data remain blinded; nominal MC and autoMCStats only.')


def require_fixed_met_category_comparison(after,before,baseline,baseline_sha256,global_cuts=False):
    from TROTASR.workflows.build_nominal_grid import MET_CATEGORY_SCHEMA,FIXED_BASELINE_SHA256
    from TROTASR.workflows.combine_limits import verify_comparison_grid
    for grid in (after,before,baseline):
        verify_comparison_grid(grid);sr_bin_counts(grid)
    if baseline.get('variant')!='baseline' or baseline_sha256!=FIXED_BASELINE_SHA256:
        raise ValueError('MET-category comparison requires the fixed w/o medium baseline')
    for grid,name in ((before,'S0'),(after,'S1')):
        spec=grid.get('medium_candidate') or {}
        if (spec.get('schema')!=MET_CATEGORY_SCHEMA or grid.get('medium_test_comparison') is not None
                or (not global_cuts and spec.get('selection',{}).get('id')!=name)
                or (global_cuts and 'global_cuts' not in spec.get('selection',{}))
                or spec.get('fixed_baseline_manifest_sha256')!=baseline_sha256):
            raise ValueError('MET-category comparison requires explicit S0/S1 or two exact global-cut contracts')
    old,new=before['medium_candidate'],after['medium_candidate']
    if any(old[k]!=new[k] for k in ('parent_populations','feature_definitions','category_definition')):
        raise ValueError('S0/S1 parent MC, physical definitions or MET categories differ')
    if global_cuts:
        evidence='shap_guided' if 'shap_guided' in old else 'exact_kinematics'
        if evidence not in new or any(old[evidence][k]!=new[evidence][k]
                                      for k in ('years','physics_policy','search_report_sha256')):
            raise ValueError('Global-cut comparisons require identical exact observables and search provenance')
    fixed=require_fixed_baseline_comparison(after,before)
    require_fixed_baseline_comparison(after,dict(baseline,variant='combined'))
    return dict(fixed_high_low_templates=fixed,baseline_manifest_sha256=baseline_sha256,
                parent_populations=new['parent_populations'])


def load_met_category_plot_inputs(spec):
    from TROTASR.workflows.build_nominal_grid import load_medium_candidate,read_medium_met_categories
    if load_medium_candidate(spec['contract_path'])!=spec:
        raise ValueError('MET-category input contract changed')
    return {year:read_medium_met_categories(Path(spec['contract_path']).parent/record['path'],spec)[0]
            for year,record in spec['inputs'].items()}


def summarize_ranking(report, grid):
    """Compact one complete, explicitly separated ranking without changing fits."""
    from TROTASR.workflows.combine_limits import comparison_candidate_score
    from TROTASR.workflows.build_nominal_grid import CUTBASED_SCHEMA,CUTBASED_CAVEAT,cutbased_candidate_bin_count
    candidate = grid.get('medium_candidate')
    cutbased = (candidate or {}).get('schema')==CUTBASED_SCHEMA
    field = 'cutbased_candidate' if cutbased else 'validation_candidate'
    if cutbased:
        cutbased_candidate_bin_count(candidate)
        if (report.get('schema')!='combined_cutbased_expected_cls_ranking_v1'
                or report.get('evaluation_partition')!='all_mc' or report.get('neural_network_used') is not False
                or report.get('independent_validation') is not False or report.get('statistical_caveat')!=CUTBASED_CAVEAT
                or report.get('populations')!=candidate['populations']
                or report.get('historical_partitions_used')!=candidate['historical_partitions_used']):
            raise ValueError('Cut-based ranking must retain its explicit all-MC statistical provenance')
    elif report.get('schema')=='combined_cutbased_expected_cls_ranking_v1':
        raise ValueError('Cut-based ranking cannot be relabelled as NN-validation')
    rows = report.get('candidates', [])
    if (grid.get('variant') != 'combined' or not candidate or not rows
            or report.get('status') != 'complete' or report.get('baseline_pending')
            or report.get('observed_sr_used') is not False
            or report.get('winner') != candidate['candidate_id']
            or report.get('policy') != candidate['ranking_policy']
            or report.get('baseline_manifest_sha256') != candidate['fixed_baseline_manifest_sha256']):
        raise ValueError('Require the complete ranking for this exact plotted winner')
    keys = {p['model']+'/'+p['mass'] for p in grid['points']}
    summaries = []
    for row in rows:
        spec=row.get(field)
        if (not spec or row.get('candidate_id')!=spec.get('candidate_id')
                or ('validation_candidate' if cutbased else 'cutbased_candidate') in row):
            raise ValueError('Ranking mixes cut-based and NN-validation contracts')
        if cutbased:
            cutbased_candidate_bin_count(spec)
            if (spec['populations']!=candidate['populations'] or spec['feature_definitions']!=candidate['feature_definitions']
                    or spec['fixed_baseline_manifest_sha256']!=candidate['fixed_baseline_manifest_sha256']):
                raise ValueError('Cut-based ranked populations or physical definitions changed')
        points = row['points']
        if (row.get('status') != 'complete' or not row.get('ranking_eligible')
                or row.get('pending') or row.get('missing_points')
                or row['requested'] != len(keys) or row['verified'] != len(keys)
                or set(points) != keys):
            raise ValueError('Ranking candidate has incomplete mass coverage')
        calculated = comparison_candidate_score(grid['points'],
            {k:v['baseline_expected_median'] for k,v in points.items()},
            {k:v['combined_expected_median'] for k,v in points.items()})
        if any(calculated[k] != row[k] for k in
               ('mean_log_ratio','geometric_mean_ratio','cells','points')):
            raise ValueError('Ranking summary differs from its actual fit values')
        summaries.append({k:row[k] for k in
            ('candidate_id','verified','mean_log_ratio','geometric_mean_ratio')})
    summaries.sort(key=lambda r:(r['mean_log_ratio'],r['candidate_id']))
    if (len({r['candidate_id'] for r in summaries}) != len(rows)
            or report['ranking_of_complete_candidates'] != [r['candidate_id'] for r in summaries]
            or summaries[0]['candidate_id'] != report['winner']):
        raise ValueError('Ranking order or winner changed')
    winner = next(r for r in rows if r['candidate_id'] == report['winner'])
    if winner[field] != candidate:
        raise ValueError('Plotted winner model/category contract differs')
    models = {}
    for model in MODELS:
        points = {k:v for k,v in winner['points'].items() if k.startswith(model+'/')}
        cells = [c for c in winner['cells'] if c['model'] == model]
        ratios = [p['ratio'] for p in points.values()]
        models[model] = dict(points=len(points), improved=sum(r<1 for r in ratios),
            equal=sum(r==1 for r in ratios), worsened=sum(r>1 for r in ratios),
            minimum_ratio=min(ratios), maximum_ratio=max(ratios),
            cell_balanced_ratio=math.exp(sum(c['mean_log_ratio'] for c in cells)/len(cells)))
    result = dict(schema='combined_optimization_plot_summary_v1', winner=report['winner'],
        **{field:candidate}, candidate_ranking=summaries,
        objective=report['policy'], baseline_manifest_sha256=report['baseline_manifest_sha256'],
        geometric_mean_ratio=winner['geometric_mean_ratio'], models=models,
        cells=winner['cells'], points=winner['points'], claim=report['claim'],
        scope='Validation-only corrected nominal SR MC + autoMCStats; not an independent test or systematic analysis')
    if cutbased:
        result.update(schema='combined_cutbased_optimization_plot_summary_v1',evaluation_partition='all_mc',
            neural_network_used=False,independent_validation=False,statistical_caveat=CUTBASED_CAVEAT,
            populations=candidate['populations'],historical_partitions_used=candidate['historical_partitions_used'],
            scope='All-MC cut-based corrected nominal SR MC + autoMCStats; not independent validation or systematic analysis')
    return result


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
    if payload.get('observed_sr_used') is not False or payload.get('scope') not in ('full_nominal_production','full_combined_systematic_production','full_run3_hybrid_production','sr_only_nominal_mcstat_comparison'):
        raise ValueError('Full-input blinded expected limits are required')
    fields = ('expected_m2', 'expected_m1', 'expected', 'expected_p1', 'expected_p2')
    if payload.get('quantiles_provided') == [.5]:
        if payload.get('scope') != 'sr_only_nominal_mcstat_comparison':
            raise ValueError('Median-only export is only a nominal contour product')
        fields = ('expected',)
        if any(any(k in p for k in ('expected_m2','expected_m1','expected_p1','expected_p2')) for p in points.values()):
            raise ValueError('Median-only export must not contain unrequested endpoints')
    if payload.get('quantiles_provided') == [.16,.5,.84]:
        if payload.get('scope') != 'sr_only_nominal_mcstat_comparison':
            raise ValueError('Three-quantile payload is only a nominal contour product')
        fields = ('expected_m1','expected','expected_p1')
        if any('expected_m2' in p or 'expected_p2' in p for p in points.values()):
            raise ValueError('A contour-only payload must not export unused two-sigma endpoints')
    for mass, record in points.items():
        stop, lsp = mass_pair(mass)
        if (record['mStop'], record['mLSP']) != (stop, lsp) or 'observed' in record:
            raise ValueError('Mass/expected-only limit contract mismatch')
        values = [float(record[f]) for f in fields]
        if not all(math.isfinite(v) and v > 0. for v in values) or not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError('Invalid expected quantiles')


def recovery_readable(state, allow_partial):
    terminal=not state.get('pid') and state['status'] in ('all_verified_pending_reconciliation','needs_attention')
    return terminal or (allow_partial and state.get('pid') and state['status']=='running')


def run3_numerical_adapter_scope(grid, limits_dir, origin, parent, point_contract):
    """Opt in only the source-bound new six-era finite continuation contract.

    Caller has already verified the point contract SHA, exact point, code
    version, workspace and protected states. This changes only the parser of
    real native/stepping logs, never quantile or measured-crossing acceptance.
    """
    flag=point_contract.get('run3_extended_numerics')
    if flag is None or flag is False:
        return False
    selection=parent.get('fit_selection',{})
    if (flag is not True or limits_dir!=ROOT/'stats/run3_systematics_20261025/limits'
            or origin!='numerical_minima_20261009'
            or grid.get('scope')!='full_combined_systematic_production'
            or parent.get('scope')!='run3_displayed_finite_continuation'
            or parent.get('stage')!=origin or parent.get('plot_range_only') is not True
            or selection.get('plot_range_only') is not True
            or selection.get('source_total')!=955 or selection.get('total')!=895
            or selection.get('expected_points_by_model')!={'T2tt':205,'T2bW':345,'T2tb':345}):
        raise ValueError('Numerical-adapter contract is outside the exact six-era continuation')
    return True


def contour_quantiles(validation, required=(.16,.5,.84)):
    required = set(required)
    crossings=validation.get('crossings',[])
    good={float(c['quantile']) for c in crossings if c['passed']}
    values={float(k):v for k,v in validation.get('quantiles',{}).items()}
    if not required <= good or not required <= set(values):return None
    values={q:values[q] for q in sorted(required)}
    ordered = list(values.values())
    if (not all(math.isfinite(v) and v>0 for v in ordered)
            or not all(a<b for a,b in zip(ordered,ordered[1:]))):return None
    return values


def collect_contour_quantiles(grid, results, evidence, model, median_only=False):
    """Separate plotting product; never promote an incomplete five-quantile fit."""
    payload=collect(grid['points'],results,model,grid['scope'])
    full_missing=list(payload['missing_points'])
    for record in payload['points'].values():
        record.pop('expected_m2');record.pop('expected_p2')
        if median_only:
            record.pop('expected_m1');record.pop('expected_p1')
    supplemental={}
    for point in grid['points']:
        if point['model']!=model:continue
        key=model+'/'+point['mass'];record=evidence[key].get('contour_only')
        if not record:continue
        ms,ml=mass_pair(point['mass']);q={float(k):v for k,v in record['quantiles'].items()}
        payload['points'][point['mass']]=dict(mStop=ms,mLSP=ml,expected=q[.5])
        if not median_only:payload['points'][point['mass']].update(expected_m1=q[.16],expected_p1=q[.84])
        supplemental[point['mass']]=record
    payload.update(quantiles_provided=[.5] if median_only else [.16,.5,.84],
        full_five_quantile_missing=full_missing,
        full_five_quantile_status='partial' if full_missing else 'complete',
        contour_only_evidence=supplemental,
        missing_points=[m for m in full_missing if m not in supplemental])
    payload['collected_point_count']=len(payload['points'])
    payload['status']='partial' if payload['missing_points'] else 'complete'
    return payload


def read_plot_point(path, live_snapshot=False):
    """Read one atomic point receipt; running fits never supply partial quantiles."""
    if live_snapshot and not path.exists():
        return dict(status='pending'), None
    raw=path.read_bytes();state=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if state.get('pid') or state['status'] not in ('complete','failed'):
        if live_snapshot:
            return dict(status='running' if state.get('pid') else 'pending'), digest
        raise ValueError('Primary point not terminal: '+str(path))
    return state,digest


def verified_current_results(grid, limits_dir, allow_partial=False, contour_only=False, median_only=False,
                             live_snapshot=False):
    """Exact saved attempts in execution-history order; no strongest-fit choice."""
    workflow_path = limits_dir/'workflow_state.json'
    if workflow_path.is_file():
        raw = workflow_path.read_bytes()
        workflow = json.loads(raw)
        if workflow.get('canonical_limits_pipeline') is True:
            return canonical_workflow_results(grid, workflow, workflow_path,
                hashlib.sha256(raw).hexdigest(), allow_partial, live_snapshot)
    from TROTASR.workflows.combine_limits import IntegrityCache, verify_card
    from TROTASR.workflows.continue_limit_fits import verify_attempt
    from TROTASR.workflows.diagnostic_common import command_key
    cache=IntegrityCache();result={};evidence={}
    from TROTASR.workflows.reconcile_limit_fits import historical_source, recovery_code_versions
    recoveries={}
    recovery_names = ('numerical_all_unfinished_20260930','numerical_remaining_20261009',
                      'numerical_targeted_20261009','numerical_minima_20261009')
    if grid.get('scope') == 'sr_only_nominal_mcstat_comparison':
        recovery_names = ('numerical_comparison_20261011','numerical_crossing_20261011',
                          'numerical_native_bracket_20261011','numerical_stepping_20261011')
    for recovery_name in recovery_names:
        recovery=limits_dir/recovery_name
        if not (recovery/'contract.json').exists():continue
        contract=read_json(recovery/'contract.json')
        if (contract['primary_contract_sha256']!=sha256(limits_dir/'contract.json') or
                contract['manifest_sha256']!=read_json(limits_dir/'contract.json')['manifest_sha256']):
            raise ValueError('Recovery belongs to a different model')
        code_versions = recovery_code_versions(recovery, contract)
        terminal=read_json(recovery/'state.json')
        if not recovery_readable(terminal,allow_partial):
            raise ValueError('Recovery is not terminal')
        recoveries[recovery_name]=(recovery,contract,code_versions)
    for point in grid['points']:
        verify_card(point,cache);key=point['model']+'/'+point['mass']
        path=limits_dir/key/'state.json';state,state_hash=read_plot_point(path,live_snapshot)
        if state['status'] in ('pending','running'):
            result[key]=state
            evidence[key]=dict(state_sha256=state_hash,accepted=None,snapshot_status=state['status'])
            continue
        histories=[('primary',state.get('attempts',[]))]
        if state['status']=='complete':
            histories=[('primary',[a for a in state['attempts'] if a.get('output')==state['output']])]
            if len(histories[0][1])!=1:raise ValueError('Ambiguous accepted primary attempt')
        else:
            for recovery_name,(recovery,contract,code_versions) in recoveries.items():
                retry_path=recovery/key/'state.json'
                if not retry_path.exists():continue
                retry=read_json(retry_path);cp=retry_path.with_name('contract.json');c=read_json(cp)
                if allow_partial and retry.get('pid') and retry['status']=='running':continue
                if retry.get('pid') or retry['contract_sha256']!=sha256(cp) or c['point']!=point or c['code'] not in code_versions:
                    raise ValueError('Unverified point recovery contract')
                for name,digest in c['protected_states'].items():
                    if sha256(ROOT/name)!=digest:raise ValueError('Protected fit state changed')
                if cache(limits_dir/key/'workspace.root')!=c['workspace_sha256']:
                    raise ValueError('Recovery workspace changed')
                histories.append((recovery_name,retry['attempts']))
        accepted=None; contour_candidate=None
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
                    recovery=recoveries[origin][0]
                    receipt=read_json(recovery/key/'command_receipts/commands'/
                        (command_key(attempt['command'],log.parent)+'.json'))
                    if (receipt['status']!='complete' or receipt['exit_code']!=0 or receipt.get('pid') or
                            receipt['command']!=attempt['command'] or receipt['products'].get(attempt['output'])!=attempt['output_sha256']):
                        raise ValueError('Recovery execution receipt changed')
                if origin=='numerical_native_bracket_20261011':
                    from TROTASR.workflows.validate_limit_precision import verify_native_attempt
                    passed,validation=verify_native_attempt(point,attempt)
                elif origin=='numerical_stepping_20261011':
                    from TROTASR.workflows.validate_limit_precision import verify_stepping_attempt
                    passed,validation=verify_stepping_attempt(point,attempt)
                elif origin=='numerical_minima_20261009':
                    recovery,parent,_=recoveries[origin]
                    point_contract=read_json(recovery/key/'contract.json')
                    if run3_numerical_adapter_scope(grid,limits_dir,origin,parent,point_contract):
                        from TROTASR.workflows.validate_limit_precision import verify_run3_numerical_attempt
                        passed,validation=verify_run3_numerical_attempt(point,attempt)
                    else:
                        passed,validation=verify_attempt(point,attempt)
                else:
                    passed,validation=verify_attempt(point,attempt)
                if ((contour_only or median_only) and contour_candidate is None and not passed
                        and grid.get('scope')=='sr_only_nominal_mcstat_comparison'
                        and ((contour_only and grid.get('variant')=='low' and key=='T2bW/mStop1650_mLSP1350')
                             or (median_only and grid.get('variant')=='medium' and grid.get('medium_candidate')))):
                    q=contour_quantiles(validation, (.5,) if median_only else (.16,.5,.84))
                    if q:
                        contour_candidate=dict(status='contour_quantiles_verified_only',quantiles=q,
                            output=attempt['output'],output_sha256=attempt['output_sha256'],
                            log=attempt['log'],log_sha256=attempt['log_sha256'],origin=origin,
                            execution_receipt_sha256=sha256(recovery/key/'command_receipts/commands'/
                                (command_key(attempt['command'],log.parent)+'.json')) if origin!='primary' else None,
                            full_five_quantile_passed=False)
                if passed:
                    accepted=dict(status='complete',validation=validation,output=attempt['output'],
                        output_sha256=attempt['output_sha256'],accepted_origin=origin,
                        accepted_log=attempt['log'],log_sha256=attempt['log_sha256'])
                    break
            if accepted:break
        if state['status']=='complete' and (not accepted or accepted['output_sha256']!=state['output_sha256']):
            raise ValueError('Previously accepted fit failed revalidation')
        result[key]=accepted or state
        evidence[key]=dict(state_sha256=state_hash,accepted=accepted)
        if not accepted and contour_candidate:evidence[key]['contour_only']=contour_candidate
    return result,evidence


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
        # During a main-only controller handoff the normal aggregate still
        # belongs to the draining owner. Add newly accepted point receipts,
        # never partial quantiles, without reopening ROOT or producer inputs.
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


@contextlib.contextmanager
def capture_expected_paths():
    """Read the original renderer's final red paths, without writing a figure."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.path import Path as FigurePath
    original = Figure.savefig
    captured = []
    inspected = False
    def capture(figure, *args, **kwargs):
        nonlocal inspected
        if inspected: return
        inspected = True
        for collection in figure.axes[0].collections:
            colors = collection.get_edgecolors()
            if len(colors) != 1 or not np.allclose(colors[0], [1.,0.,0.,1.]): continue
            widths = collection.get_linewidths()
            if len(widths) != 1 or float(widths[0]) not in (1.7,3.): continue
            style = '--' if collection.get_linestyles()[0][1] is not None else '-'
            for path in collection.get_paths():
                if len(path.vertices) < 2: continue
                captured.append(dict(path=FigurePath(path.vertices.copy(),
                    None if path.codes is None else path.codes.copy()),
                    linewidth=float(widths[0]),linestyle=style))
    Figure.savefig = capture
    try: yield captured
    finally: Figure.savefig = original


@contextlib.contextmanager
def additional_expected_paths(paths, label, color='#009E73', extra_overlays=(), legend_index=2):
    """Place an extra original-renderer contour before its normal legend/save."""
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    original = Axes.legend
    drawn = set()
    def legend(axes, *args, **kwargs):
        handles = kwargs.get('handles')
        if handles is not None and axes not in drawn:
            drawn.add(axes)
            handles = list(handles)
            for index,(rows,title,shade) in enumerate([(paths,label,color),*extra_overlays]):
                for row in rows:
                    axes.add_patch(PathPatch(row['path'],facecolor='none',edgecolor=shade,
                        linewidth=row['linewidth'],linestyle=row['linestyle'],
                        zorder=7 if row['linestyle']=='-' else 5.5))
                handles.insert(legend_index+index,Line2D([0],[0],color=shade,lw=3.,label=title))
            kwargs = dict(kwargs,handles=handles)
        return original(axes,*args,**kwargs)
    Axes.legend = legend
    try: yield
    finally: Axes.legend = original


def expected_improvement(baseline, combined):
    """A ratio above one means a smaller combined expected-r upper limit."""
    import statistics
    first, second = baseline['points'], combined['points']
    common = sorted(set(first) & set(second))
    ratios = [float(first[m]['expected'])/float(second[m]['expected']) for m in common]
    if any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('Invalid expected-r comparison')
    return dict(definition='baseline expected r / combined expected r; larger than 1 means improvement',
        matched_points=len(common),baseline_points=len(first),combined_points=len(second),
        median=statistics.median(ratios) if ratios else None,
        minimum=min(ratios) if ratios else None,maximum=max(ratios) if ratios else None,
        fraction_improved=sum(r>1 for r in ratios)/len(ratios) if ratios else None)


def load_plot_snapshot(path):
    path=internal_path(path);manifest=read_json(path/'plot_manifest.json')
    if (manifest['status']!='visual_QA_passed' or manifest['contract'].get('variant')!='combined'
            or set(manifest['contract'].get('expected_line_colors',{}))!={'combined','high','medium','low'}):
        raise ValueError('Requires the verified four-region nominal plot snapshot')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or sha256(path/name)!=digest:
            raise ValueError('Snapshot artifact changed: '+name)
    return dict(path=path,manifest=manifest,sha256=sha256(path/'plot_manifest.json'))


def prepare_snapshot(grid_manifest,snapshot):
    """Read compact, already-verified plot exports; never open fit ROOT files."""
    grid_manifest=internal_path(grid_manifest);grid=read_json(grid_manifest);sr_bin_counts(grid)
    prefixes={'combined':'','high':'overlay_','medium':'second_overlay_','low':'third_overlay_'}
    variant=grid.get('variant');contract=snapshot['manifest']['contract']
    if (grid.get('scope')!='sr_only_nominal_mcstat_comparison' or variant not in prefixes
            or sha256(grid_manifest)!=contract[prefixes[variant]+'grid_sha256']):
        raise ValueError('Snapshot and requested grid differ')
    expected=contract['renderer_sha256']
    if (sha256(Path(renderer.__file__))!=expected
            or sha256(ROOT/'stats/contour_config.json')!=contract['config_sha256']):
        raise ValueError('Snapshot renderer/reference configuration changed')
    records={}
    for model in MODELS:
        old=next(p for p in snapshot['manifest']['plots'] if p['model']==model)
        coverage=old['variant_coverage'][variant]
        masses=[p['mass'] for p in grid['points'] if p['model']==model]
        selected=[m for m in masses if 600<=mass_pair(m)[0]<=1800 and 1<=mass_pair(m)[1]<=1800]
        name=(model+'_expected_limits_with_signal_theory.json' if variant=='combined'
              else model+'_'+variant+'_contour_limits.json')
        payload=read_json(snapshot['path']/name)
        validate_collected(dict(payload,scope=grid['scope'],observed_sr_used=False),
                           selected if variant=='combined' else masses)
        records[model]=dict(payload=renderer.select_limit_range(payload,selected),full_payload=payload,
            input_sha256=contract[prefixes[variant]+'collected_sha256'][model],
            input=str((snapshot['path']/name).relative_to(ROOT)),requested_masses=selected,
            full_grid_status=coverage['full_five_quantile_status'],
            full_grid_missing=coverage['full_five_quantile_missing'],full_grid_point_count=len(masses),
            full_contour_point_count=coverage['contour_points'],
            out_of_plot_range=[m for m in masses if m not in selected])
    return grid,records,expected


def contour_artist_signature(axes):
    """Hashes are in physical mass coordinates, before any display transform."""
    import numpy as np
    from matplotlib.colors import to_rgba
    palette={tuple(to_rgba(c)):v for v,c in [('combined','red'),('high','blue'),
             ('medium','#009E73'),('low','#9467BD')]}
    rows=[]
    for artist in [*axes.collections,*axes.patches]:
        colors=artist.get_edgecolors() if hasattr(artist,'get_edgecolors') else [artist.get_edgecolor()]
        if len(colors)!=1 or tuple(colors[0]) not in palette:continue
        variant=palette[tuple(colors[0])]
        dashed=(artist.get_linestyles()[0][1] is not None if hasattr(artist,'get_linestyles')
                else artist.get_linestyle() not in ('-','solid'))
        paths=artist.get_paths() if hasattr(artist,'get_paths') else [artist.get_path()]
        for path in paths:
            digest=hashlib.sha256(np.asarray(path.vertices,dtype='<f8').tobytes())
            if path.codes is not None:digest.update(path.codes.tobytes())
            rows.append(dict(variant=variant,dashed=bool(dashed),sha256=digest.hexdigest()))
    return rows


@contextlib.contextmanager
def contours_only_display(diagnostics, standalone_medians=True, all_medians=False, point_metric=None):
    """Hide only requested display artists; retain the original renderer bytes."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import numpy as np
    original_filled,original_colorbar,original_save=Axes.contourf,Figure.colorbar,Figure.savefig
    filled={};bars={};checked=set()
    def track_filled(axes,*args,**kwargs):
        result=original_filled(axes,*args,**kwargs)
        filled.setdefault(axes.figure,[]).extend(result.collections)
        return result
    def track_colorbar(figure,*args,**kwargs):
        result=original_colorbar(figure,*args,**kwargs)
        bars.setdefault(figure,[]).append(result)
        return result
    def save(figure,*args,**kwargs):
        if figure not in checked:
            checked.add(figure);axes=figure.axes[0]
            before=contour_artist_signature(axes)
            keep=[row for row in before if not row['dashed'] or
                  (not all_medians and (not standalone_medians or row['variant']=='combined'))]
            for artist in filled.get(figure,[]):artist.remove()
            for bar in bars.get(figure,[]):bar.remove()
            for artist in list(axes.collections) if standalone_medians or all_medians else []:
                colors=artist.get_edgecolors()
                if (len(colors)==1 and (np.allclose(colors[0],[0.,0.,1.,1.]) or
                        (all_medians and np.allclose(colors[0],[1.,0.,0.,1.])))
                        and artist.get_linestyles()[0][1] is not None):artist.remove()
            for artist in list(axes.patches) if standalone_medians or all_medians else []:
                if (any(np.allclose(artist.get_edgecolor(),color) for color in
                        ([0.,.6196078431372549,.45098039215686275,1.],
                         [.5803921568627451,.403921568627451,.7411764705882353,1.]))
                        and artist.get_linestyle() not in ('-','solid')):artist.remove()
            after=contour_artist_signature(axes)
            if after!=keep:raise ValueError('Display-only change modified retained contour coordinates')
            diagnostics.update(heatmap=False,colorbar=False,
                standalone='median_only' if standalone_medians else 'not_shown',
                combined='median_only' if all_medians else 'median_and_one_sigma',retained_paths=after,
                retained_coordinates_unchanged=True,removed_paths=len(before)-len(after),
                axes_count=len(figure.axes))
            if point_metric is not None:
                from matplotlib.colors import TwoSlopeNorm
                bound=point_metric['bound']
                dots=axes.scatter(point_metric['mstop'],point_metric['mlsp'],
                    c=point_metric['percent'],s=50,marker='s',edgecolors='none',
                    cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0.,vmax=bound),zorder=1)
                bar=original_colorbar(figure,dots,ax=axes,pad=.04,fraction=.048,aspect=34,extend='both')
                bar.set_label(point_metric['label'],fontsize=20,labelpad=20)
                bar.ax.tick_params(labelsize=20,direction='in')
                diagnostics.update(point_metric=point_metric,colorbar=True,
                    metric_interpolation='none; individual evaluated mass points',axes_count=len(figure.axes))
        return original_save(figure,*args,**kwargs)
    Axes.contourf,Figure.colorbar,Figure.savefig=track_filled,track_colorbar,save
    try:yield
    finally:Axes.contourf,Figure.colorbar,Figure.savefig=original_filled,original_colorbar,original_save


def prepare(limits_dir, grid_manifest, allow_partial=False, contour_only=False,snapshot=None,median_only=False,
            live_snapshot=False):
    if snapshot is not None:return prepare_snapshot(grid_manifest,snapshot)
    limits_dir, grid_manifest = internal_path(limits_dir), internal_path(grid_manifest)
    grid = read_json(grid_manifest)
    workflow_path = limits_dir/'workflow_state.json'
    workflow_raw = workflow_path.read_bytes() if workflow_path.is_file() else None
    workflow = json.loads(workflow_raw) if workflow_raw is not None else {}
    canonical = workflow.get('canonical_limits_pipeline') is True
    if canonical:
        # The accepted manifest supplies display metadata; do not replay its
        # production/card gates or reopen template and policy inputs here.
        grid = dict(grid, canonical_limits_pipeline=True,
                    _canonical_analysis_root=str(canonical_analysis_root(workflow)))
        grid['points'] = [point for point in grid['points']
                          if 600 <= mass_pair(point['mass'])[0] <= 1800
                          and 1 <= mass_pair(point['mass'])[1] <= 1800]
        grid['expected_points_by_model'] = {
            model: sum(point['model'] == model for point in grid['points']) for model in MODELS}
        state = workflow
        run3_systematics = bool(grid.get('run3_systematics'))
    else:
        contract = read_json(limits_dir / 'contract.json')
        state = read_json(limits_dir / 'state.json')
        from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
        run3_systematics = require_run3_systematics_policy(grid)
    sr_bin_counts(grid)
    if live_snapshot and (not allow_partial or not (
                          grid.get('scope')=='full_run3_hybrid_production' or
                          grid.get('scope')=='full_combined_systematic_production' and run3_systematics)
                          or snapshot is not None or contour_only or median_only):
        raise ValueError('Live snapshot requires an explicit six-era grid and five validated quantiles')
    study = grid.get('scope') == 'sr_only_nominal_mcstat_comparison'
    if grid['status'] != 'cards_ready' or grid['scope'] not in ('full_nominal_production','full_combined_systematic_production','full_run3_hybrid_production','sr_only_nominal_mcstat_comparison'):
        raise ValueError('Require a completed full-input card grid')
    if canonical:
        expected = sha256(Path(renderer.__file__))
    else:
        if contract['manifest_sha256'] != sha256(grid_manifest):
            raise ValueError('Limits belong to a different card grid')
        if (state['status'] not in ('fits_complete_pending_plots', 'needs_attention')
                and not (study and allow_partial and state['status'] == 'awaiting_condor_results')
                and not (live_snapshot and state['status']=='running' and state.get('pid'))):
            raise ValueError('The fit controller is not finished')
        if state['status'] != 'fits_complete_pending_plots' and not allow_partial and not study:
            raise ValueError('Failed fits remain; do not label a partial contour complete')
        expected = sha256(Path(renderer.__file__))  # provenance, not a historical code-version gate
    records = {}
    combined=grid['scope'] in ('full_combined_systematic_production', 'full_run3_hybrid_production')
    audited = canonical or combined or (study and state['status'] != 'awaiting_condor_results')
    if canonical:
        results,evidence = canonical_workflow_results(grid, workflow, workflow_path,
            hashlib.sha256(workflow_raw).hexdigest(), allow_partial, live_snapshot)
    else:
        results,evidence=verified_current_results(grid,limits_dir,allow_partial,contour_only,median_only,
                                                 live_snapshot) if audited else (state['results'],{})
    if study and not audited:
        from TROTASR.workflows.combine_limits import IntegrityCache, verify_card
        from TROTASR.workflows.limit_validation import validate
        cache = IntegrityCache()
        for point in grid['points']:
            key = point['model']+'/'+point['mass']
            record = results.get(key,{})
            if record.get('status') != 'complete': continue
            verify_card(point,cache)
            saved = read_json(limits_dir/key/'state.json')
            if saved != record or record.get('pid'):
                raise ValueError('Comparison point is not the saved terminal result')
            attempts = [a for a in record['attempts'] if a.get('output') == record['output']]
            if len(attempts) != 1: raise ValueError('Ambiguous comparison accepted attempt')
            attempt = attempts[0]; product = internal_path(ROOT/record['output']); log = internal_path(ROOT/attempt['log'])
            if (attempt['exit_code'] != 0 or cache(product) != record['output_sha256']
                    or sha256(log) != attempt['log_sha256']
                    or read_json(log.parent/'validation.json') != attempt):
                raise ValueError('Comparison ROOT/log/receipt mismatch')
            passed, check = validate(product,log,attempt['tolerance'])
            if not passed or check != record['validation']:
                # Serialized quantile keys are strings; compare their JSON form.
                if not passed or json.loads(json.dumps(check)) != record['validation']:
                    raise ValueError('Comparison limit did not pass saved-output validation')
    for model in MODELS:
        masses = [p['mass'] for p in grid['points'] if p['model'] == model]
        if len(masses) != grid['expected_points_by_model'][model]:
            raise ValueError('Requested signal grid was reduced')
        path = limits_dir / model / 'expected_limits.json'
        payload = collect(grid['points'],results,model,grid['scope']) if audited else read_json(path)
        if contour_only or median_only:
            expected_variant='medium' if median_only else 'low'
            if (not study or grid.get('variant')!=expected_variant or not audited
                    or (median_only and not grid.get('medium_candidate'))):
                raise ValueError('Explicit contour-only extension requires its exact nominal comparison')
            payload=collect_contour_quantiles(grid,results,evidence,model,median_only)
        validate_collected(payload, masses, allow_partial)
        regenerated = collect(grid['points'], results, model, grid['scope'])
        if contour_only or median_only:regenerated=collect_contour_quantiles(grid,results,evidence,model,median_only)
        if regenerated != payload:
            raise ValueError('Collected quantiles no longer match fit-controller evidence')
        selected, outside = [], []
        for mass in masses:
            stop, lsp = mass_pair(mass)
            (selected if 600 <= stop <= 1800 and 1 <= lsp <= 1800 else outside).append(mass)
        records[model] = dict(payload=renderer.select_limit_range(payload, selected), full_payload=payload,
            input_sha256=(hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest() if audited else sha256(path)),
            input=str((workflow_path if canonical else path).relative_to(ROOT)),requested_masses=selected,
            full_grid_status=payload.get('full_five_quantile_status',payload['status']),
            full_grid_missing=payload.get('full_five_quantile_missing',payload['missing_points']),
            full_grid_point_count=len(masses), out_of_plot_range=outside)
        if live_snapshot:
            from collections import Counter
            records[model]['snapshot_counts']=dict(Counter(results[model+'/'+m]['status'] for m in masses))
    return grid, records, expected


def test_comparison_metrics(baseline, before, after):
    """Report every mass on the common test; this does not select a model."""
    keys=set(baseline['points'])
    if keys != set(before['points']) or keys != set(after['points']):
        raise ValueError('Test metric coverage must match the full fixed baseline')
    rows=[]
    for mass in sorted(keys):
        base,old,new=(float(p['points'][mass]['expected']) for p in (baseline,before,after))
        if any(not math.isfinite(v) or v<=0 for v in (base,old,new)):
            raise ValueError('Nonfinite or nonpositive expected limit in test comparison')
        ms,ml=mass_pair(mass)
        rows.append(dict(mass=mass,mStop=ms,mLSP=ml,baseline_r95=base,before_r95=old,after_r95=new,
            before_reduction_percent=100*(1-old/base),after_reduction_percent=100*(1-new/base),
            after_over_before=new/old))
    return dict(points=rows,test_evaluation_only=True,reselection_allowed=False,
        definition='100*(1 - combined r95 / w/o medium-DeltaM SR r95); positive means improvement',
        before_vs_baseline=expected_improvement(baseline,before),
        after_vs_baseline=expected_improvement(baseline,after),
        after_vs_before=expected_improvement(before,after))


def load_frozen_test_plot_definition(spec):
    """Read only the sidestudy freeze authenticated by the canonical builder."""
    from TROTASR.workflows.build_nominal_grid import load_medium_test_comparison
    if load_medium_test_comparison(spec['contract_path']) != spec:
        raise ValueError('Frozen test comparison input contract changed')
    content=Path(spec['comparison_freeze_path']).read_bytes()
    if hashlib.sha256(content).hexdigest()!=spec['comparison_freeze_sha256']:
        raise ValueError('Frozen test comparison definition changed')
    return json.loads(content)


def render_test_comparison(limits_dir, grid_manifest, output, before_limits, before_grid,
                           baseline_limits, baseline_grid):
    """Frozen, same-test comparison suite through the unchanged original renderer."""
    output=internal_path(output)
    paths=dict(after=(limits_dir,grid_manifest),before=(before_limits,before_grid),
               baseline=(baseline_limits,baseline_grid))
    if any(v is None for pair in paths.values() for v in pair):
        raise ValueError('Test comparison requires before, after and w/o medium SR grid/limit pairs')
    prepared={name:prepare(*pair) for name,pair in paths.items()}
    after,before,baseline=(prepared[name][0] for name in ('after','before','baseline'))
    identities=require_fixed_test_comparison(after,before,baseline,sha256(baseline_grid))
    definitions={name:load_frozen_test_plot_definition(grid['medium_test_comparison'])
                 for name,grid in (('before',before),('after',after))}
    hashes={value[2] for value in prepared.values()}
    if len(hashes)!=1:raise ValueError('Test comparison renderers differ')
    config=read_json(ROOT/'stats/contour_config.json')
    refs={name:internal_path(ROOT/spec['path']) for name,spec in config['references'].items()}
    for name,path in refs.items():
        if sha256(path)!=config['references'][name]['sha256']:raise ValueError('Contour reference changed')
    theory_path=internal_path(ROOT/config['theory']['path'])
    if sha256(theory_path)!=config['theory']['sha256']:raise ValueError('Theory table changed')
    thermal=renderer.load_thermal_relic_contour(refs['thermal_relic'])
    cross_sections=renderer.load_stop_pair_xsecs(theory_path)
    runtime=renderer.runtime_versions()
    frozen=definitions['after']
    source_plots=[ROOT.parent/p for p in frozen['sources'] if p.endswith('/plot_manifest.json')]
    if not source_plots or any(read_json(p)['contract']['runtime']!=runtime for p in source_plots):
        raise ValueError('Reuse the frozen comparison source plots numerical runtime')
    contract=dict(schema='medium_dm_same_test_plot_suite_v1',**identities,
        variants={name:dict(grid_sha256=sha256(pair[1]),
            collected_sha256={m:r['input_sha256'] for m,r in prepared[name][1].items()})
            for name,pair in paths.items()},
        frozen_comparison_sha256=after['medium_test_comparison']['comparison_freeze_sha256'],
        before=before['medium_test_comparison'],after=after['medium_test_comparison'],
        styles=['medians','one_sigma','before_improvement','after_improvement'],
        baseline_label=r'w/o medium-$\Delta M$ SR',
        expected_line_colors=dict(after='red',before='blue',baseline='#009E73'),
        renderer_sha256=next(iter(hashes)),adapter_sha256=sha256(__file__),
        config_sha256=sha256(ROOT/'stats/contour_config.json'),runtime=runtime,
        sr_data_blinded=True,expected_only=True,historical_test_already_opened=True,
        systematics=False,control_region_fit=False,reselection_allowed=False,rebinning_allowed=False,
        claim='Same previously opened historical test MC, frozen models/categories, fixed high+low; nominal MC plus autoMCStats only')
    receipt=output/'plot_manifest.json'
    if output.exists():
        old=read_json(receipt)
        if old['contract']!=contract or len(old.get('plots',[]))!=12:
            raise FileExistsError('Preserve divergent/incomplete test-comparison output')
        for name,digest in old['files'].items():
            if sha256(output/name)!=digest:raise ValueError('Test-comparison artifact changed')
        return old
    output.mkdir(parents=True)
    write_json(receipt,dict(status='rendering',contract=contract,plots=[]))
    plots=[]
    for model in MODELS:
        records={name:values[1][model] for name,values in prepared.items()}
        metrics=test_comparison_metrics(*(records[name]['full_payload'] for name in ('baseline','before','after')))
        write_json(output/(model+'_test_comparison_metrics.json'),metrics)
        for name,record in records.items():
            write_json(output/(model+'_'+name+'_limits.json'),record['full_payload'])
        available=set(records['after']['payload']['points'])
        if any(set(r['payload']['points'])!=available for r in records.values()):
            raise ValueError('Displayed same-test mass coverage differs')
        common=dict(run2_contours=refs[model],luminosity_label=renderer.LUMINOSITY_LABELS['2024_2025'],
            analysis_label=None,x_min=600.,x_max=1800.,y_min=1.,y_max=1800.,
            decay_label=renderer.DECAY_LABELS[model],color_field='xsec',stop_pair_xsecs=cross_sections,
            mask_offshell=False,run2_compressed_contours=refs.get(model+'_compressed'),
            topology=model,interpolation_mode='topology-aware',thermal_relic=thermal)
        requested=records['after']['requested_masses'];baseline_diag={}
        with supported_contours(requested,available,baseline_diag),capture_expected_paths() as captured:
            success=renderer.plot_contour(records['baseline']['payload'],output/'unused.png',
                interpolation_diagnostics=baseline_diag,**common)
        if not success:raise RuntimeError('Cannot render the fixed baseline contour')
        baseline_paths=[p for p in captured if p['linestyle']=='-']
        displayed=[r for r in metrics['points'] if r['mass'] in available]
        bound=max(1.,math.ceil(max(abs(r[key]) for r in displayed
            for key in ('before_reduction_percent','after_reduction_percent'))/5)*5)
        for style in contract['styles']:
            png=output/('expected_limit_'+model.lower()+'_2024_2025_test_'+style+'.png')
            diagnostic={};display={};metric=None
            if style.endswith('_improvement'):
                name=style.split('_',1)[0]
                metric=dict(mstop=[r['mStop'] for r in displayed],mlsp=[r['mLSP'] for r in displayed],
                    percent=[r[name+'_reduction_percent'] for r in displayed],bound=bound,
                    label=name.capitalize()+r': $r_{95}$ reduction vs w/o medium-$\Delta M$ SR (%)')
            sigma=style=='one_sigma'
            suffix=r' $\pm1\sigma$' if sigma else ' median'
            with supported_contours(requested,available,diagnostic), \
                    additional_expected_paths(baseline_paths,contract['baseline_label']), \
                    contours_only_display(display,standalone_medians=False,all_medians=not sigma,point_metric=metric):
                success=renderer.plot_contour(records['after']['payload'],png,**common,
                    interpolation_diagnostics=diagnostic,overlay_limit_payload=records['before']['payload'],
                    primary_legend_label='After reoptimization'+suffix,
                    overlay_legend_label='Before reoptimization'+suffix)
            if not success or not png.exists() or not png.with_suffix('.pdf').exists():
                raise RuntimeError('Incomplete test-comparison rendering')
            diagnostic.update(display_style=display,fixed_baseline=baseline_diag)
            write_json(png.with_suffix('.json'),json_ready(diagnostic))
            plots.append(dict(model=model,style=style,png=png.name,pdf=png.with_suffix('.pdf').name,
                status='generated_pending_visual_QA',w_o_medium_sr_included=True,
                full_grid_points=len(metrics['points']),displayed_points=len(available),
                out_of_plot_range=records['after']['out_of_plot_range']))
    result=dict(status='generated_pending_visual_QA',contract=contract,plots=plots,
        files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p!=receipt},
        sr_data_blinded=True,full_workflow_complete=False)
    write_json(receipt,result)
    return result


def render_validation_comparison(limits_dir, grid_manifest, output, before_limits, before_grid,
                                 baseline_limits, baseline_grid, ranking=None, style_reference=None):
    """Fixed C01 validation score-edge comparison, with only the final one-sigma band."""
    return render_candidate_comparison(limits_dir,grid_manifest,output,before_limits,before_grid,
                                      baseline_limits,baseline_grid,ranking,style_reference)


def render_cutbased_comparison(limits_dir, grid_manifest, output, baseline_limits, baseline_grid,
                              ranking, style_reference):
    """The all-MC cut-only winner and fixed w/o medium baseline, without an NN overlay."""
    return render_candidate_comparison(limits_dir,grid_manifest,output,None,None,
                                      baseline_limits,baseline_grid,ranking,style_reference,cutbased=True)


def render_cutbased_preview(limits_dir, grid_manifest, output, gnn_limits, gnn_grid,
                           baseline_limits, baseline_grid, style_reference):
    """Show explicitly chosen complete cut/GNN candidates without a winner claim."""
    return render_candidate_comparison(limits_dir,grid_manifest,output,gnn_limits,gnn_grid,
        baseline_limits,baseline_grid,None,style_reference,cutbased=True,preview=True)


def render_met_category_comparison(limits_dir,grid_manifest,output,s0_limits,s0_grid,
                                  baseline_limits,baseline_grid,style_reference,gnn_limits=None,gnn_grid=None,
                                  global_cuts=False):
    if global_cuts and (gnn_limits is not None or gnn_grid is not None):
        raise ValueError('Exact global-cut comparison cannot add a different-population NN reference')
    return render_candidate_comparison(limits_dir,grid_manifest,output,s0_limits,s0_grid,
        baseline_limits,baseline_grid,None,style_reference,met_categories=True,
        gnn_reference=(gnn_limits,gnn_grid) if gnn_limits is not None or gnn_grid is not None else None,
        global_cuts=global_cuts)


def render_candidate_comparison(limits_dir, grid_manifest, output, before_limits, before_grid,
                                baseline_limits, baseline_grid, ranking, style_reference, cutbased=False, preview=False,
                                met_categories=False,gnn_reference=None,global_cuts=False):
    """Shared adopted renderer/style; only the explicitly selected input contract differs."""
    if preview and (not cutbased or ranking is not None):
        raise ValueError('Complete-candidate preview is separate from final ranking modes')
    if global_cuts and (not met_categories or gnn_reference is not None):
        raise ValueError('Global-cut plots require the explicit same-MC MET-category mode')
    if met_categories and (cutbased or preview or ranking is not None):
        raise ValueError('S0/S1 comparison cannot claim a ranked cut/NN winner')
    if gnn_reference is not None and not met_categories:
        raise ValueError('An extra GNN reference is only supported in the explicit S0/S1 comparison')
    output = internal_path(output)
    paths = dict(after=(limits_dir,grid_manifest),baseline=(baseline_limits,baseline_grid))
    if not cutbased or preview:paths=dict(after=paths['after'],before=(before_limits,before_grid),baseline=paths['baseline'])
    if gnn_reference is not None:paths['reference']=gnn_reference
    if any(v is None for pair in paths.values() for v in pair) or style_reference is None:
        raise ValueError('Candidate comparison requires its grid/limit pairs and a plot style reference')
    grids = {name:read_json(internal_path(pair[1])) for name,pair in paths.items()}
    after,baseline = (grids[name] for name in ('after','baseline'))
    from TROTASR.workflows.build_nominal_grid import MET_CATEGORY_SCHEMA
    met_preview=preview and after['medium_candidate']['schema']==MET_CATEGORY_SCHEMA
    if met_categories:
        before=grids['before']
        identities=require_fixed_met_category_comparison(after,before,baseline,sha256(baseline_grid),global_cuts)
        exports={name:load_met_category_plot_inputs(grid['medium_candidate'])
                 for name,grid in (('before',before),('after',after))}
        identities['selection_inputs']={name:{y:dict(input_sha256=grids[name]['medium_candidate']['inputs'][y]['sha256'],
            provenance=p['provenance'],selection_population=p.get('selection_population',{}),
            coverage={k:v for k,v in p['coverage'].items() if k!='sample_totals'})
            for y,p in years.items()} for name,years in exports.items()}
        if gnn_reference is not None:
            from TROTASR.workflows.build_nominal_grid import CANDIDATE_SCHEMA
            from TROTASR.workflows.combine_limits import verify_comparison_grid
            reference=grids['reference'];verify_comparison_grid(reference);sr_bin_counts(reference)
            if ((reference.get('medium_candidate') or {}).get('schema')!=CANDIDATE_SCHEMA
                    or reference.get('evaluation_partition','validation')!='validation'
                    or reference.get('medium_test_comparison') is not None):
                raise ValueError('Optional GNN reference must be an explicit validation-MC candidate')
            require_fixed_baseline_comparison(after,reference)
            gnn_exports=load_validation_plot_inputs(reference['medium_candidate'])
            identities['gnn_inputs']={y:dict(input_sha256=reference['medium_candidate']['inputs'][y]['sha256'],
                provenance=p['provenance']) for y,p in gnn_exports.items()}
    elif cutbased:
        if ranking is None and not preview:
            raise ValueError('Cut-based optimized comparison requires its complete all-MC CLs ranking')
        if preview:
            before=grids['before']
            identities=require_fixed_cutbased_preview(after,before,baseline,sha256(baseline_grid))
            gnn_test=before.get('medium_test_comparison') is not None
            gnn_spec=before['medium_test_comparison'] if gnn_test else before['medium_candidate']
            gnn_exports=(load_test_plot_inputs if gnn_test else load_validation_plot_inputs)(gnn_spec)
            identities['gnn_inputs']={y:dict(input_sha256=gnn_spec['inputs'][y]['sha256'],
                provenance=p['provenance'],normalization={k:v for k,v in p['years'][y].items() if k!='samples'})
                for y,p in gnn_exports.items()}
        else:
            identities=require_fixed_cutbased_comparison(after,baseline,sha256(baseline_grid))
        loader=load_met_category_plot_inputs if met_preview else load_cutbased_plot_inputs
        exports=loader(after['medium_candidate'])
        identities['cutbased_inputs']={y:dict(input_sha256=after['medium_candidate']['inputs'][y]['sha256'],
            provenance=p['provenance'],coverage=p['coverage']) for y,p in exports.items()}
    else:
        before=grids['before']
        identities = require_fixed_validation_comparison(after,before,baseline,sha256(baseline_grid))
        exports = {name:load_validation_plot_inputs(grid['medium_candidate'])
                   for name,grid in (('before',before),('after',after))}
        identities['validation_inputs'] = require_same_validation_inputs(exports['before'],exports['after'])
    check_only = ranking is None and not preview and not met_categories
    if check_only and after['medium_candidate'] != before['medium_candidate']:
        raise ValueError('A changed score candidate requires its complete validation CLs ranking')
    ranking_data = summarize_ranking(read_json(internal_path(ranking)),after) if ranking else None
    if ranking_data is not None and not cutbased:
        report = read_json(internal_path(ranking))
        if before['medium_candidate']['candidate_id'] not in report['ranking_of_complete_candidates']:
            raise ValueError('Score-edge ranking must include the original C01 reference')
        for row in report['candidates']:
            require_fixed_validation_comparison(dict(after,medium_candidate=row['validation_candidate']),
                                                before,baseline,sha256(baseline_grid))
    runtime = renderer.runtime_versions()
    style_reference = internal_path(style_reference)
    style_contract = read_json(style_reference)['contract']
    if style_contract['runtime'] != runtime:
        raise ValueError('Candidate comparison must reuse the recorded original-plot runtime')
    prepared = {}
    for name,pair in paths.items():
        key = tuple(str(internal_path(p)) for p in pair)
        reused = next((prepared[n] for n in prepared
                       if tuple(str(internal_path(p)) for p in paths[n]) == key),None)
        prepared[name] = reused if reused is not None else prepare(*pair)
    hashes = {value[2] for value in prepared.values()}
    config = read_json(ROOT/'stats/contour_config.json')
    if (len(hashes) != 1 or style_contract['renderer_sha256'] != next(iter(hashes))
            or style_contract['config_sha256'] != sha256(ROOT/'stats/contour_config.json')):
        raise ValueError('Candidate comparison renderer/reference configuration changed')
    refs = {name:internal_path(ROOT/spec['path']) for name,spec in config['references'].items()}
    for name,path in refs.items():
        if sha256(path) != config['references'][name]['sha256']:raise ValueError('Contour reference changed')
    theory = internal_path(ROOT/config['theory']['path'])
    if sha256(theory) != config['theory']['sha256']:raise ValueError('Theory table changed')
    thermal = renderer.load_thermal_relic_contour(refs['thermal_relic'])
    cross_sections = renderer.load_stop_pair_xsecs(theory)
    contract = dict(schema='medium_dm_validation_score_comparison_v1', **identities,
        variants={name:dict(grid_sha256=sha256(pair[1]),
            collected_sha256={m:r['input_sha256'] for m,r in prepared[name][1].items()})
            for name,pair in paths.items()},
        after=after['medium_candidate'],
        renderer_sha256=next(iter(hashes)), adapter_sha256=sha256(__file__),
        config_sha256=sha256(ROOT/'stats/contour_config.json'), runtime=runtime,
        style_reference=dict(path=str(style_reference.relative_to(ROOT)),sha256=sha256(style_reference)),
        display_style='baseline_and_before_medians_after_one_sigma_no_heatmap',
        expected_line_colors=dict(after='red',before='blue',baseline='#009E73'),
        evaluation_partition='validation',test_events_used=False,systematics=False,control_region_fit=False,
        sr_data_blinded=True,expected_only=True,functional_check_only=check_only,
        selection_status='not_ranked_render_check' if check_only else 'complete_full_grid_validation_ranking',
        ranking=None if ranking is None else dict(path=str(internal_path(ranking).relative_to(ROOT)),
                                             sha256=sha256(ranking),winner=ranking_data['winner']),
        claim='Same validation MC and model; C01 score edges only, fixed high+low; nominal MC plus autoMCStats, not an independent test')
    if met_categories:
        from TROTASR.workflows.build_nominal_grid import MET_CATEGORY_CAVEAT,met_category_caveat
        contract.update(schema='medium_dm_met_category_selection_comparison_v1',before=before['medium_candidate'],
            evaluation_partition='all_mc',display_style='baseline_and_S0_medians_S1_one_sigma_no_heatmap',
            selection_status='explicit_S0_S1_comparison',statistical_caveat=MET_CATEGORY_CAVEAT,
            automatic_selection=False,final_adoption=False,ranking_used=False,independent_validation=False,
            neural_network_used=False,claim=MET_CATEGORY_CAVEAT)
        contract.pop('test_events_used')
        if global_cuts:
            caveat=met_category_caveat(after['medium_candidate'])
            contract.update(schema='medium_dm_exact_global_cut_comparison_v1',
                display_style='baseline_and_cut_B_medians_cut_A_one_sigma_no_heatmap',
                selection_status='explicit_restricted_global_cut_pool',statistical_caveat=caveat,claim=caveat,
                cut_labels=dict(A=after['medium_candidate']['selection'],B=before['medium_candidate']['selection']))
        if gnn_reference is not None:
            contract.update(gnn_reference=grids['reference']['medium_candidate'],neural_network_used=True,
                evaluation_partition='all_mc_with_validation_gnn_reference',
                gnn_reference_caveat='GNN uses 10% validation MC (weight x10, variance x100); S0/S1 use 100% MC (x1/x1). Not a method-only comparison.',
                gnn_method_only_comparison=False)
            contract['expected_line_colors']['reference']='#9467BD'
    elif cutbased:
        from TROTASR.workflows.build_nominal_grid import CUTBASED_CAVEAT
        contract.update(schema='medium_dm_all_mc_cutbased_comparison_v1',
            display_style='baseline_median_cutbased_combined_one_sigma_no_heatmap',
            expected_line_colors=dict(after='red',baseline='#009E73'),evaluation_partition='all_mc',
            neural_network_used=False,independent_validation=False,
            historical_partitions_used=after['medium_candidate']['historical_partitions_used'],
            statistical_caveat=CUTBASED_CAVEAT,selection_status='complete_full_grid_all_mc_cutbased_ranking',
            claim='Best within the declared all-MC physical-cut pool; fixed high+low; nominal MC plus autoMCStats, not independent validation')
        contract.pop('test_events_used')
        if preview:
            gnn_sampling=(dict(partition='test',fraction=.7,weight_scale=1/.7,variance_scale=1/.49) if gnn_test else
                          dict(partition='validation',fraction=.1,weight_scale=10.,variance_scale=100.))
            caveat=TEST_CUTBASED_PREVIEW_CAVEAT if gnn_test else CUTBASED_PREVIEW_CAVEAT
            contract.update(schema='medium_dm_cutbased_gnn_complete_candidate_preview_v1',
                before=gnn_spec,
                display_style='baseline_and_gnn_medians_cutbased_one_sigma_no_heatmap',
                expected_line_colors=dict(after='red',before='blue',baseline='#009E73'),
                evaluation_partition='mixed_explicit_candidate_populations',neural_network_used=True,
                selection_status='explicit_complete_candidate_preview',final_adoption=False,
                automatic_selection=False,ranking_used=False,method_only_comparison=False,
                mc_sampling=dict(before=gnn_sampling,
                                 after=dict(partition='all_mc',fraction=1.,weight_scale=1.,variance_scale=1.)),
                comparison_caveat=caveat,claim=caveat)
            if gnn_test:
                contract.update(gnn_test_evaluation=True,reselection_allowed=False,rebinning_allowed=False,
                                historical_test_already_opened=True)
            if met_preview:
                contract.update(cut_selection=after['medium_candidate']['selection'],
                    category_definition=after['medium_candidate']['category_definition'])
    else:contract['before']=before['medium_candidate']
    receipt = output/'plot_manifest.json'
    if output.exists():
        old = read_json(receipt)
        if old['contract'] != contract or len(old.get('plots',[])) != len(MODELS):
            raise FileExistsError('Preserve divergent/incomplete candidate comparison output')
        for name,digest in old['files'].items():
            if sha256(output/name) != digest:raise ValueError('Candidate comparison artifact changed')
        return old
    output.mkdir(parents=True)
    write_json(receipt,dict(status='rendering',contract=contract,plots=[]))
    if ranking_data is not None:write_json(output/'ranking_summary.json',ranking_data)
    plots = []
    for model in MODELS:
        records = {name:values[1][model] for name,values in prepared.items()}
        if ranking_data is not None:
            for mass,value in records['after']['full_payload']['points'].items():
                ranked = ranking_data['points'][model+'/'+mass]
                if (ranked['combined_expected_median'] != value['expected'] or
                        ranked['baseline_expected_median'] != records['baseline']['full_payload']['points'][mass]['expected']):
                    raise ValueError('Ranked fit values differ from plotted saved-output values')
        for name,record in records.items():write_json(output/(model+'_'+name+'_limits.json'),record['full_payload'])
        available = set(records['after']['payload']['points'])
        if any(set(r['payload']['points']) != available for r in records.values()):
            raise ValueError('Displayed candidate mass coverage differs')
        common = dict(run2_contours=refs[model],luminosity_label=renderer.LUMINOSITY_LABELS['2024_2025'],
            analysis_label=None,x_min=600.,x_max=1800.,y_min=1.,y_max=1800.,
            decay_label=renderer.DECAY_LABELS[model],color_field='xsec',stop_pair_xsecs=cross_sections,
            mask_offshell=False,run2_compressed_contours=refs.get(model+'_compressed'),
            topology=model,interpolation_mode='topology-aware',thermal_relic=thermal)
        requested = records['after']['requested_masses'];baseline_diag = {}
        with supported_contours(requested,available,baseline_diag),capture_expected_paths() as captured:
            success = renderer.plot_contour(records['baseline']['payload'],output/'unused.png',
                interpolation_diagnostics=baseline_diag,**common)
        if not success:raise RuntimeError('Cannot render the fixed baseline contour')
        baseline_paths = [p for p in captured if p['linestyle'] == '-']
        suffix=('exact_global_cuts' if global_cuts else 'met_categories_S0_S1' if met_categories else 'cutbased_gnn_preview' if preview else
                'all_mc_cutbased_comparison' if cutbased else 'validation_score_comparison')
        png = output/('expected_limit_'+model.lower()+'_2024_2025_'+suffix+'.png')
        diagnostic = {};display = {}
        labels=(dict(primary_legend_label=r'All-MC cut-based combined $\pm1\sigma$') if cutbased else
            dict(overlay_limit_payload=records['before']['payload'],
                 primary_legend_label=(r'C01 render check $\pm1\sigma$' if check_only else r'Optimized score bins $\pm1\sigma$'),
                 overlay_legend_label='Original C01 score bins (median)'))
        if preview:
            cut_label=(after['medium_candidate']['selection']['label'].split(' (')[0]+' cut-based'
                       if met_preview else 'Cut-based')
            labels=dict(overlay_limit_payload=records['before']['payload'],
                primary_legend_label=cut_label+r' (100% MC) $\pm1\sigma$',
                overlay_legend_label=('GNN (70% test, scaled) median' if gnn_test else 'GNN (10% MC, scaled) median'))
        if met_categories:
            labels=dict(overlay_limit_payload=records['before']['payload'],
                primary_legend_label=r'S1 (100% MC) $\pm1\sigma$',overlay_legend_label='S0 (100% MC) median')
        if global_cuts:
            labels=dict(overlay_limit_payload=records['before']['payload'],
                primary_legend_label=r'Cut A combined $\pm1\sigma$',overlay_legend_label='Cut B combined (median)')
            if 'shap_guided' in after['medium_candidate']:
                short=lambda grid:grid['medium_candidate']['selection']['label'].split(':')[0]
                labels.update(primary_legend_label=short(after)+r' combined $\pm1\sigma$',
                              overlay_legend_label=short(before)+' combined (median)')
        extra=[]
        if gnn_reference is not None:
            reference_diag={}
            with supported_contours(requested,available,reference_diag),capture_expected_paths() as reference_paths:
                success=renderer.plot_contour(records['reference']['payload'],output/'unused.png',
                    interpolation_diagnostics=reference_diag,**common)
            if not success:raise RuntimeError('Cannot render the optional GNN reference')
            extra=[([p for p in reference_paths if p['linestyle']=='-'],'GNN (10% MC, scaled) median','#9467BD')]
            diagnostic['gnn_reference']=reference_diag
        with supported_contours(requested,available,diagnostic), \
                additional_expected_paths(baseline_paths,r'w/o medium-$\Delta M$ SR (median)',extra_overlays=extra,
                                          legend_index=1 if cutbased and not preview else 2), \
                contours_only_display(display,standalone_medians=True):
            success = renderer.plot_contour(records['after']['payload'],png,**common,
                interpolation_diagnostics=diagnostic,**labels)
        if not success or not png.exists() or not png.with_suffix('.pdf').exists():
            raise RuntimeError('Incomplete candidate-comparison rendering')
        diagnostic.update(display_style=display,fixed_baseline=baseline_diag)
        write_json(png.with_suffix('.json'),json_ready(diagnostic))
        plots.append(dict(model=model,png=png.name,pdf=png.with_suffix('.pdf').name,
            status='generated_pending_visual_QA',w_o_medium_sr_included=True,
            full_grid_points=records['after']['full_grid_point_count'],displayed_points=len(available),
            out_of_plot_range=records['after']['out_of_plot_range']))
    result = dict(status='generated_pending_visual_QA',contract=contract,plots=plots,
        files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p != receipt},
        sr_data_blinded=True,full_workflow_complete=False)
    write_json(receipt,result)
    return result


def render(limits_dir, grid_manifest, output, allow_partial=False, overlay_limits=None, overlay_grid=None,
           second_overlay_limits=None, second_overlay_grid=None,
           third_overlay_limits=None, third_overlay_grid=None, three_quantile_low=False,
           contours_only=False,from_plot_snapshot=None,ranking=None,median_medium=False,
           before_plot_snapshot=None,comparison_note=None,live_snapshot=False):
    output = internal_path(output)
    if live_snapshot and (not allow_partial or any(v is not None for v in
            (overlay_limits,overlay_grid,second_overlay_limits,second_overlay_grid,third_overlay_limits,
             third_overlay_grid,from_plot_snapshot,ranking,before_plot_snapshot,comparison_note))
            or three_quantile_low or contours_only or median_medium):
        raise ValueError('Live snapshot is only a standalone partial-grid view')
    snapshot_started=time.time()
    snapshot=load_plot_snapshot(from_plot_snapshot) if from_plot_snapshot is not None else None
    before_snapshot=load_plot_snapshot(before_plot_snapshot) if before_plot_snapshot is not None else None
    comparison=before_snapshot is not None
    if comparison and (snapshot is None or not contours_only or allow_partial or
            overlay_grid is None or overlay_limits is None or
            any(v is not None for v in (second_overlay_grid,second_overlay_limits,
                                       third_overlay_grid,third_overlay_limits)) or not comparison_note):
        raise ValueError('Before/after requires two snapshots, two grid/limit pairs, contours-only and a provenance note')
    if comparison_note and not comparison:
        raise ValueError('Comparison note requires before/after snapshot mode')
    def inputs(limits,manifest,partial=False,three=False,median=False):
        prepared=prepare(limits,manifest,partial,three,snapshot=snapshot,median_only=median,
                         live_snapshot=live_snapshot)
        if prepared[0].get('canonical_limits_pipeline') is True:
            return prepared
        from TROTASR.workflows.build_nominal_grid import CUTBASED_SCHEMA,MET_CATEGORY_SCHEMA
        if (prepared[0].get('medium_candidate') or {}).get('schema')==MET_CATEGORY_SCHEMA:
            raise ValueError('MET-category plots require --met-category-comparison with explicit S0/S1 and baseline')
        if (prepared[0].get('medium_candidate') or {}).get('schema')==CUTBASED_SCHEMA:
            raise ValueError('All-MC cut-based plots require --cutbased-comparison (ranked) or --cutbased-preview (explicit complete candidates)')
        return prepared
    grid, records, renderer_hash = inputs(limits_dir, grid_manifest, allow_partial)
    eras = (tuple(grid['years']) if grid.get('canonical_limits_pipeline') is True
            else record_years(grid))
    ranking_data = summarize_ranking(read_json(internal_path(ranking)),grid) if ranking is not None else None
    high_bins, low_bins = sr_bin_counts(grid)
    study = grid.get('scope') == 'sr_only_nominal_mcstat_comparison'
    overlay_records = None
    if overlay_limits is not None or overlay_grid is not None:
        if not study or overlay_limits is None or overlay_grid is None:
            raise ValueError('A comparison overlay requires both its grid and limits')
        other, overlay_records, other_renderer = (prepare_snapshot(overlay_grid,before_snapshot)
            if comparison else inputs(overlay_limits,overlay_grid,allow_partial))
        if (other.get('scope') != grid['scope'] or other.get('parent_grid_sha256') != grid['parent_grid_sha256']
                or other_renderer != renderer_hash or other.get('treatment') != grid['treatment']):
            raise ValueError('Overlay statistical treatment/grid differs')
    second_records = None
    four_way = third_overlay_limits is not None or third_overlay_grid is not None
    if median_medium and (not contours_only or not four_way or ranking is None):
        raise ValueError('Median medium export requires the ranked four-way contour-only display')
    if contours_only and not (four_way or comparison):raise ValueError('Contour-only style requires an explicit comparison')
    if three_quantile_low and not four_way:raise ValueError('Contour-only low extension requires four-way comparison')
    if second_overlay_limits is not None or second_overlay_grid is not None:
        if overlay_records is None or second_overlay_limits is None or second_overlay_grid is None:
            raise ValueError('A three-way comparison requires both complete overlay pairs')
        third, second_records, third_renderer = inputs(second_overlay_limits,second_overlay_grid,allow_partial,median=median_medium)
        if (third.get('scope') != grid['scope'] or third.get('parent_grid_sha256') != grid['parent_grid_sha256']
                or third_renderer != renderer_hash or third.get('treatment') != grid['treatment']
                or {grid['variant'],other['variant'],third.get('variant')} !=
                   ({'high','medium','combined'} if four_way else {'baseline','medium','combined'})):
            raise ValueError('Comparison requires the distinct requested variants with identical treatment')
    third_records = None
    if four_way:
        if second_records is None or third_overlay_limits is None or third_overlay_grid is None:
            raise ValueError('The fourth contour requires all three complete overlay pairs')
        fourth,third_records,fourth_renderer = inputs(third_overlay_limits,third_overlay_grid,allow_partial,three_quantile_low)
        if (fourth.get('scope') != grid['scope'] or fourth.get('parent_grid_sha256') != grid['parent_grid_sha256']
                or fourth_renderer != renderer_hash or fourth.get('treatment') != grid['treatment']
                or [grid['variant'],other['variant'],third['variant'],fourth.get('variant')] !=
                   ['combined','high','medium','low']):
            raise ValueError('Four-way comparison must be combined/high/medium/low with identical treatment')
    if comparison:
        fixed_templates=require_fixed_baseline_comparison(grid,other)
        for model in MODELS:
            if set(records[model]['payload']['points']) != set(overlay_records[model]['payload']['points']):
                raise ValueError('Before/after plotted mass coverage differs')
    else:
        require_same_medium_candidate([grid]+([other] if overlay_records is not None else [])
            +([third] if second_records is not None else [])+([fourth] if third_records is not None else []))
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
    if comparison and any(s['manifest']['contract']['runtime'] != runtime
                          for s in (snapshot,before_snapshot)):
        raise ValueError('Before/after plots must reuse the recorded renderer runtime')
    contract = dict(grid_sha256=sha256(grid_manifest),
        collected_sha256={m: r['input_sha256'] for m, r in records.items()},
        renderer_sha256=renderer_hash, adapter_sha256=sha256(Path(__file__)),
        config_sha256=sha256(ROOT / 'stats/contour_config.json'), runtime=runtime,
        allow_partial=allow_partial, sr_bins_per_year=dict(highdm=high_bins,lowdm=low_bins),
        sr_merge=grid.get('sr_merge'))
    if grid.get('canonical_limits_pipeline') is True:
        contract.update(canonical_limits_pipeline=True, analysis_root=grid['_canonical_analysis_root'])
    if grid.get('hybrid_systematics'):
        contract.update(years=list(eras), hybrid_systematics=grid['hybrid_systematics'],
            systematic_scope=grid['systematic_scope'], preliminary=True,
            analysis_label=None, status_annotations='report_and_manifest_only')
    if grid.get('run3_systematics'):
        contract.update(years=list(eras), run3_systematics=grid['run3_systematics'],
            systematic_scope=grid['systematic_scope'], preliminary=True,
            analysis_label=None, status_annotations='report_and_manifest_only')
    if live_snapshot:
        contract.update(live_snapshot=True,snapshot_started=snapshot_started,snapshot_finished=time.time(),
            snapshot_counts={m:r['snapshot_counts'] for m,r in records.items()},
            claim='In-progress grid; only complete five-quantile fits; missing support is masked',
            fit_or_histogram_recalculation=False)
    if ranking_data is not None:
        contract['ranking'] = dict(path=str(internal_path(ranking).relative_to(ROOT)),
            sha256=sha256(ranking), winner=ranking_data['winner'])
    if median_medium:contract['median_only_medium_export']=True
    labels = {'baseline':'High + low SR', 'high':r'High-$\Delta M$ SR only', 'low':r'Low-$\Delta M$ SR only',
              'medium':'Medium SR', 'combined':'High + low + medium SR'}
    if four_way: labels['medium'] = r'Medium-$\Delta M$ SR only'
    if contours_only:
        labels['combined']=r'Combined expected median $\pm1\sigma$'
        for variant in ('high','medium','low'):labels[variant]+=' median'
        contract['display_style']='contours_only_standalone_medians_combined_one_sigma'
    if snapshot is not None:
        contract['source_plot_snapshot']=dict(path=str(snapshot['path'].relative_to(ROOT)),
            manifest_sha256=snapshot['sha256'],files=snapshot['manifest']['files'],
            fit_or_histogram_recalculation=False)
    if comparison:
        contract.update(display_style='combined_before_after_medians_and_one_sigma',
            expected_line_colors=dict(after='red',before='blue'),
            fixed_high_low_templates=fixed_templates,comparison_note=comparison_note,
            before_plot_snapshot=dict(path=str(before_snapshot['path'].relative_to(ROOT)),
                manifest_sha256=before_snapshot['sha256'],files=before_snapshot['manifest']['files']),
            before_medium_candidate=other.get('medium_candidate'),
            claim='Historical result overlay; changes are not attributed solely to optimization when input sampling differs')
    if study:
        contract.update(variant=grid['variant'],statistical_treatment=grid['treatment'],
                        sr_bins_per_year=grid['sr_bins_per_year'])
        if grid.get('medium_candidate') is not None:
            contract['medium_candidate'] = grid['medium_candidate']
    if overlay_records is not None:
        contract.update(overlay_grid_sha256=sha256(overlay_grid),
            overlay_collected_sha256={m:r['input_sha256'] for m,r in overlay_records.items()})
    if second_records is not None:
        contract.update(second_overlay_grid_sha256=sha256(second_overlay_grid),
            second_overlay_collected_sha256={m:r['input_sha256'] for m,r in second_records.items()},
            second_overlay_policy='Original-renderer paths after topology interpolation, support mask and tiny-path suppression; no numerical reinterpolation',
            expected_line_colors={grid['variant']:'red',other['variant']:'blue',third['variant']:'#009E73'})
    if third_records is not None:
        contract.update(third_overlay_grid_sha256=sha256(third_overlay_grid),
            third_overlay_collected_sha256={m:r['input_sha256'] for m,r in third_records.items()})
        contract['expected_line_colors'][fourth['variant']] = '#9467BD'
        contract['three_quantile_low_overlay']=three_quantile_low
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
    if ranking_data is not None:
        write_json(output/'combined_optimization_summary.json',dict(ranking_data,source=contract['ranking']))
    canonical_luminosity_label = None
    if grid.get('canonical_limits_pipeline') is True:
        from TROTASR.utils.paths import analysis_era
        analysis_root = Path(grid['_canonical_analysis_root'])
        luminosities = [float(read_json(analysis_root/'estimations'/
                        ('normalization_'+analysis_era(year)+'.json.gz'))['luminosity_fb'])
                        for year in eras]
        if not all(math.isfinite(value) and value > 0 for value in luminosities):
            raise ValueError('Invalid recorded normalization luminosity')
        canonical_luminosity_label = r'%.2f fb$^{-1}$ (13.6 TeV)' % sum(luminosities)
    for model in MODELS:
        record = records[model]
        payload = (dict(record['payload'], signal_theory_policy='Nominal cross-section reference only; no profiled or external theory uncertainty')
                   if study else annotate_expected_grid(record['payload'], theory))
        era_name = '_'.join(eras)
        if canonical_luminosity_label is not None:
            luminosity_label = canonical_luminosity_label
        elif eras == ('2024', '2025'):
            luminosity_label = renderer.LUMINOSITY_LABELS['2024_2025']
        else:
            from TROTASR.utils.paths import luminosity_fb
            luminosity_label = r'%.2f fb$^{-1}$ (13.6 TeV)' % sum(luminosity_fb(y) for y in eras)
        name = 'expected_limit_%s_%s_highdm%d_lowdm%d_xsec_topology_thermal_relic' % (model.lower(),era_name,high_bins,low_bins)
        if grid.get('hybrid_systematics'):
            name += '_partial_systematics'
        if live_snapshot:name += '_in_progress'
        if study:
            name = 'expected_limit_%s_2024_2025_%s_nominal_mcstat_xsec' % (model.lower(),grid['variant'])
            if overlay_records is not None: name += '_vs_'+other['variant']
            if second_records is not None: name += '_vs_'+third['variant']
            if third_records is not None: name += '_vs_'+fourth['variant']
            if contours_only:name += '_contours_only'
        if comparison:name='expected_limit_%s_2024_2025_combined_before_after_contours_only' % model.lower()
        png = output / (name + '.png')
        diagnostics = {}
        if len(payload['points'])<4:continue
        available = set(payload['points'])
        if overlay_records is not None: available &= set(overlay_records[model]['payload']['points'])
        if second_records is not None: available &= set(second_records[model]['payload']['points'])
        if third_records is not None: available &= set(third_records[model]['payload']['points'])
        extras = []
        if second_records is not None:
            extras.append((third,second_records,'second_overlay','#009E73'))
        if third_records is not None:
            extras.append((fourth,third_records,'third_overlay','#9467BD'))
        extra_paths = []
        for extra_grid,extra_records,diagnostic_key,color in extras:
            extra_diagnostics = {}
            with supported_contours(record['requested_masses'],available,extra_diagnostics), capture_expected_paths() as paths:
                captured = renderer.plot_contour(extra_records[model]['payload'],png,
                    run2_contours=refs[model],luminosity_label=renderer.LUMINOSITY_LABELS['2024_2025'],
                    analysis_label=None,x_min=600.,x_max=1800.,y_min=1.,y_max=1800.,
                    decay_label=renderer.DECAY_LABELS[model],color_field='xsec',stop_pair_xsecs=cross_sections,
                    mask_offshell=False,run2_compressed_contours=refs.get(model+'_compressed'),
                    topology=model,interpolation_mode='topology-aware',
                    interpolation_diagnostics=extra_diagnostics,thermal_relic=thermal)
            if not captured: raise RuntimeError('Original renderer could not prepare extra contour: '+model)
            diagnostics[diagnostic_key] = dict(variant=extra_grid['variant'],path_count=len(paths),
                interpolation=extra_diagnostics)
            extra_paths.append((paths,labels[extra_grid['variant']],color))
        extra = (additional_expected_paths(*extra_paths[0],extra_overlays=extra_paths[1:])
                 if extra_paths else contextlib.nullcontext())
        display_diagnostics={}
        display=contours_only_display(display_diagnostics,standalone_medians=not comparison) if contours_only else contextlib.nullcontext()
        # Progress/provenance belongs in the report and manifest, not the figure.
        with supported_contours(record['requested_masses'],available,diagnostics), extra, display:
            complete = renderer.plot_contour(payload, png, run2_contours=refs[model],
                luminosity_label=luminosity_label, analysis_label=None,
                x_min=600., x_max=1800., y_min=1., y_max=1800.,
                decay_label=renderer.DECAY_LABELS[model], color_field='xsec', stop_pair_xsecs=cross_sections,
                mask_offshell=False, run2_compressed_contours=refs.get(model + '_compressed'),
                topology=model, interpolation_mode='topology-aware', interpolation_diagnostics=diagnostics,
                thermal_relic=thermal,
                overlay_limit_payload=overlay_records[model]['payload'] if overlay_records is not None else None,
                primary_legend_label=(r'After reoptimization: combined $\pm1\sigma$' if comparison
                    else labels[grid['variant']] if study else None),
                overlay_legend_label=(r'Before reoptimization: combined $\pm1\sigma$' if comparison
                    else labels[other['variant']] if overlay_records is not None else None))
        if contours_only:diagnostics['display_style']=display_diagnostics
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
        if comparison:
            write_json(output/(model+'_before_combined_contour_limits.json'),overlay_records[model]['payload'])
            plots[-1]['before_after']=dict(
                plotted_mass_range=expected_improvement(overlay_records[model]['payload'],payload),
                before_full_grid_status=overlay_records[model]['full_grid_status'],
                before_full_grid_point_count=overlay_records[model]['full_grid_point_count'],
                note=comparison_note)
        if four_way:
            variants={grid['variant']:record,other['variant']:overlay_records[model],
                      third['variant']:second_records[model],fourth['variant']:third_records[model]}
            plots[-1]['variant_coverage']={v:dict(full_five_quantile_status=r['full_grid_status'],
                full_five_quantile_missing=r['full_grid_missing'],requested=r['full_grid_point_count'],
                contour_points=r.get('full_contour_point_count',r['full_payload']['collected_point_count'])) for v,r in variants.items()}
            for variant,variant_record in variants.items():
                if variant==grid['variant']:continue
                write_json(output/(model+'_'+variant+'_contour_limits.json'),variant_record['full_payload'])
        if second_records is not None and not four_way:
            variants = {grid['variant']:record,other['variant']:overlay_records[model],
                        third['variant']:second_records[model]}
            plots[-1]['expected_r_improvement'] = dict(
                full_requested_grid=expected_improvement(variants['baseline']['full_payload'],variants['combined']['full_payload']),
                plotted_mass_range=expected_improvement(variants['baseline']['payload'],variants['combined']['payload']))
    result = dict(status='generated_pending_visual_QA', contract=contract, plots=plots,
        files={p.name: sha256(p) for p in output.iterdir() if p.is_file() and p != receipt},
        sr_data_blinded=True, full_workflow_complete=False)
    write_json(receipt, result)
    return result


def live_candidate_ready(state, total):
    """A live contour never bridges missing/unfinished mass points."""
    return (state.get('status') == 'fits_complete_pending_plots'
            and state.get('total') == total and state.get('completed') == total
            and state.get('failed') in (0, []) and state.get('observed_sr_used') is False
            and not state.get('pid'))


def live_candidate_registry(pools, baseline_digest):
    """Union explicit, authenticated comparison pools without silently losing IDs."""
    rows = {}
    for path in pools:
        pool = read_json(path)
        if pool.get('fixed_baseline_manifest_sha256') != baseline_digest:
            raise ValueError('Live pool uses a different high+low baseline')
        for row in pool['candidates']:
            key = row['candidate_id']
            if not key.replace('_', '').isalnum():
                raise ValueError('Unsafe candidate identifier')
            if key in rows and rows[key]['contract_sha256'] != row['contract_sha256']:
                raise ValueError('Conflicting live candidate contracts: '+key)
            rows[key] = dict(rows.get(key, {}), **row, directory=str(Path(path).parent/key/'combined'))
    return list(rows.values())


@contextlib.contextmanager
def live_svg_display(layers, svg, png):
    """Display cached original-renderer paths; never interpolate in the browser."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.path import Path as FigurePath
    from matplotlib.patches import PathPatch
    original = Figure.savefig
    saved = set()
    def save(figure, *args, **kwargs):
        if figure in saved:
            return
        saved.add(figure)
        axes = figure.axes[0]
        # The primary is the fixed baseline, not a new candidate or a winner.
        for collection in axes.collections:
            colors = collection.get_edgecolors()
            if len(colors) == 1 and np.allclose(colors[0], [1.,0.,0.,1.]):
                collection.set_edgecolor('#222222')
                collection.set_linewidth(3.)
        legend = axes.get_legend()
        if legend:
            for line in legend.get_lines():
                if line.get_color() == 'red':
                    line.set_color('#222222')
        for layer in layers:
            for index, row in enumerate(layer['paths']):
                path = FigurePath(np.asarray(row['vertices']),
                                  None if row['codes'] is None else np.asarray(row['codes']))
                artist = PathPatch(path, facecolor='none', edgecolor=layer['color'],
                                   linewidth=1.5, alpha=.8, zorder=7)
                artist.set_gid('candidate-'+layer['id']+'-'+str(index))
                axes.add_patch(artist)
        for path, fmt in ((svg, 'svg'), (png, 'png')):
            pending = path.with_name(path.name+'.pending')
            original(figure, pending, format=fmt, dpi=150)
            pending.replace(path)
    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original


def live_page():
    """Small temporary viewer; scientific paths/axes come only from the renderer."""
    return '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medium-ΔM · live candidate limits</title>
<style>
body{margin:0;background:#f5f6f8;color:#18212f;font:15px system-ui}header{padding:18px 24px;background:white;border-bottom:1px solid #ddd}
h1{font-size:23px;margin:0 0 8px}p{margin:6px 0}.muted{color:#566273;font-size:13px}button{padding:7px 11px;margin:4px;border:1px solid #bbb;border-radius:5px;background:white;cursor:pointer}
main{padding:14px}.plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;background:white;border-radius:6px}.plot{min-width:0}h2{font-size:18px;margin:10px 12px 0}.chart{display:block;width:100%;aspect-ratio:1.04;background:white;border:0}.downloads{margin:0 12px 12px;font-size:12px}aside{margin-top:14px;background:white;padding:12px;border-radius:6px}#rows{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:0 16px}
.row{border-bottom:1px solid #eee;padding:8px 3px}.row:hover{background:#f2f5fa}.row label{cursor:pointer}summary{font-size:12px;cursor:pointer;color:#657183}.swatch{display:inline-block;width:20px;border-top:3px solid;margin-right:7px;vertical-align:middle}.count{font-size:12px;color:#586473}.error{color:#b32318}pre{white-space:pre-wrap;font-size:11px}.status{font-weight:600}
@media(max-width:700px){header{padding:12px}main{padding:6px}.plots{gap:2px}h2{font-size:15px;margin:8px 6px 0}.downloads{margin:0 6px 8px}}
</style><header><h1>Medium-ΔM category comparison · Live</h1>
<p>2024 + 2025 · 220.66 fb⁻¹ · Expected 95% CL median contours</p>
<p class="muted">임시 validation 비교 · nominal MC + autoMCStats · systematic / CR fit 미포함 · 자동 최종 선택 없음</p>
<div><button id="all">완료 후보 모두</button><button id="none">기준선만</button></div>
<p id="status" class="status">완료 결과를 불러오는 중…</p><p id="updated" class="muted"></p></header>
<main><div class="plots">
<section class="plot"><h2>T2tt</h2><object class="chart" data-model="T2tt" aria-label="T2tt limit" type="image/svg+xml"></object><p class="downloads"><a class="svgLink">SVG</a> · <a class="pngLink">PNG</a></p></section>
<section class="plot"><h2>T2bW</h2><object class="chart" data-model="T2bW" aria-label="T2bW limit" type="image/svg+xml"></object><p class="downloads"><a class="svgLink">SVG</a> · <a class="pngLink">PNG</a></p></section>
<section class="plot"><h2>T2tb</h2><object class="chart" data-model="T2tb" aria-label="T2tb limit" type="image/svg+xml"></object><p class="downloads"><a class="svgLink">SVG</a> · <a class="pngLink">PNG</a></p></section>
</div><aside>
<p><span class="swatch" style="color:#222"></span>w/o medium-ΔM SR (항상 표시)</p>
<p class="muted">모든 질량점 완료 후 곡선 추가. 후보 선택과 강조는 세 플롯에 동시에 적용됩니다. 얇은 선은 각각 high+low+해당 medium 후보입니다.</p>
<div id="rows"></div></aside></main>
<script>
let state=null,revision='',selected={},highlight=null;
const charts=[...document.querySelectorAll('.chart')];
function apply(){charts.forEach(chart=>{let doc=chart.contentDocument;if(!doc)return;doc.querySelectorAll('[id^="candidate-"]').forEach(g=>{let id=g.id.slice(10).replace(/-\\d+$/,'');g.style.display=selected[id]===false?'none':'';g.style.opacity=highlight&&highlight!==id?'.12':'1';});});}
function loadCharts(){if(!state)return;let r=state.plot_revision||'0';if(revision===r)return;revision=r;charts.forEach(chart=>{let m=chart.dataset.model,url=m+'.svg?v='+r;chart.data=url;chart.parentElement.querySelector('.svgLink').href=url;chart.parentElement.querySelector('.pngLink').href=m+'.png?v='+r;});}
function describe(row){let d=row.category_definition;if(!d)return '카드 준비 후 정의 표시';return d.groups.map((g,i)=>'C'+(i+1)+': '+g.conditions.map(c=>c[0]+(c[2]?' ≥ ':' < ')+c[1]).join(' & ')+'; score '+g.score_edges.join(', ')).join('\\n');}
function rows(){let box=document.querySelector('#rows');box.replaceChildren();state.candidates.forEach(row=>{let div=document.createElement('div');div.className='row';let label=document.createElement('label'),cb=document.createElement('input');cb.type='checkbox';cb.disabled=row.status!=='plotted';cb.checked=selected[row.id]!==false&&!cb.disabled;cb.onchange=()=>{selected[row.id]=cb.checked;apply();};let sw=document.createElement('span');sw.className='swatch';sw.style.color=row.color;label.append(cb,sw,document.createTextNode(row.label+' · '+row.bins+' bins'));div.append(label);let count=document.createElement('div');count.className='count';count.textContent=row.id.split('_').slice(-1)[0]+' · '+row.completed+'/'+state.points_per_candidate+' · '+row.status;div.append(count);let details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='물리 분할 / GNN 경계';pre.textContent=describe(row);details.append(summary,pre);div.append(details);if(row.error){let err=document.createElement('p');err.className='error';err.textContent=row.error;div.append(err);}div.onmouseenter=()=>{highlight=row.id;apply();};div.onmouseleave=()=>{highlight=null;apply();};box.append(div);});}
async function refresh(){try{let response=await fetch('live_state.json?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error(response.status);state=await response.json();let n=state.candidates.filter(r=>r.status==='plotted').length;document.querySelector('#status').textContent=n+' / '+state.candidates.length+' 후보 표시 · 완료 전 후보는 보간하지 않음';let age=(Date.now()-Date.parse(state.updated_utc))/1000;document.querySelector('#updated').textContent='최근 확인 '+new Date(state.updated_utc).toLocaleString()+' · 1분 주기'+(age>180?' · 갱신 지연: 연결/프로세스 확인 필요':'');rows();loadCharts();apply();}catch(e){document.querySelector('#status').textContent='갱신 대기 / 연결 확인: '+e.message;}}
charts.forEach(chart=>chart.onload=apply);document.querySelector('#all').onclick=()=>{if(!state)return;highlight=null;selected={};rows();apply();};document.querySelector('#none').onclick=()=>{if(!state)return;highlight=null;state.candidates.forEach(r=>selected[r.id]=false);rows();apply();};refresh();setInterval(refresh,15000);
</script></html>'''


def live_candidates(limits_dir, grid_manifest, output, pools, interval=60, port=None, once=False,
                    style_reference=None):
    """Read-only fit monitor with cached, full-grid-authenticated renderer paths."""
    import colorsys
    import fcntl
    import os
    import time
    from datetime import datetime, timezone
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    output = internal_path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'live.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())+'\n');lock.flush()
    baseline_digest = sha256(grid_manifest)
    registry = live_candidate_registry(pools, baseline_digest)
    if (style_reference is None or read_json(style_reference)['contract']['runtime']
            != renderer.runtime_versions()):
        raise ValueError('Live view must use the recorded original-plot numerical runtime')
    baseline, records, renderer_digest = prepare(limits_dir, grid_manifest)
    if baseline.get('variant') != 'baseline':
        raise ValueError('Live comparison requires w/o medium SR as the baseline')
    total = len(baseline['points'])
    config = read_json(ROOT/'stats/contour_config.json')
    refs = {name:internal_path(ROOT/spec['path']) for name,spec in config['references'].items()}
    for name,path in refs.items():
        if sha256(path) != config['references'][name]['sha256']:
            raise ValueError('Live contour reference changed')
    theory = ROOT/config['theory']['path']
    if sha256(theory) != config['theory']['sha256']:
        raise ValueError('Live contour theory changed')
    thermal = renderer.load_thermal_relic_contour(refs['thermal_relic'])
    xsecs = renderer.load_stop_pair_xsecs(theory)
    render_key = dict(adapter_sha256=sha256(__file__), renderer_sha256=renderer_digest,
        config_sha256=sha256(ROOT/'stats/contour_config.json'), runtime=renderer.runtime_versions())
    def common(model):
        return dict(run2_contours=refs[model], luminosity_label=renderer.LUMINOSITY_LABELS['2024_2025'],
            analysis_label=None, x_min=600., x_max=1800., y_min=1., y_max=1800.,
            decay_label=renderer.DECAY_LABELS[model], color_field='xsec', stop_pair_xsecs=xsecs,
            mask_offshell=False, run2_compressed_contours=refs.get(model+'_compressed'),
            topology=model, interpolation_mode='topology-aware', thermal_relic=thermal)
    (output/'index.html').write_text(live_page(), encoding='utf-8')
    server = None
    if port is not None:
        class Handler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Cache-Control', 'no-store')
                super().end_headers()
            def list_directory(self, path):
                self.send_error(403)
                return None
        server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(output)))
        Thread(target=server.serve_forever, daemon=True).start()
    caches = {};previous = None;revision = None
    try:
        while True:
            rows = [];layers = {model:[] for model in MODELS}
            for index,item in enumerate(registry):
                cid = item['candidate_id']; directory = internal_path(item['directory'])
                rgb = colorsys.hsv_to_rgb((index*.61803398875+.57)%1., .78, .77)
                color = '#'+''.join('%02x'%round(c*255) for c in rgb)
                row = dict(id=cid, label='C%02d'%(index+1), color=color, bins=item['medium_bins'],
                    status='queued', completed=0, category_definition=item.get('category_definition'))
                rows.append(row)
                try:
                    state_path = directory/'limits/state.json'
                    if not state_path.exists():
                        continue
                    stamp = (state_path.stat().st_mtime_ns, state_path.stat().st_size)
                    cached = caches.get(cid)
                    if cached is None or cached['stamp'] != stamp:
                        state = read_json(state_path)
                        row.update(completed=state.get('completed',0), status=state['status'])
                        if not live_candidate_ready(state,total):
                            continue
                        manifest = directory/'grid/manifest.json'
                        grid = read_json(manifest)
                        require_fixed_baseline_comparison(grid,dict(baseline,variant='combined'))
                        spec = grid['medium_candidate']
                        if (spec['candidate_id'] != cid or spec['contract_sha256'] != item['contract_sha256']
                                or spec['test_events_used'] is not False):
                            raise ValueError('Live candidate identity/partition differs from pool')
                        key = dict(render_key, grid_sha256=sha256(manifest), state_sha256=sha256(state_path))
                        cache_path = output/'paths'/(cid+'.json')
                        cached = read_json(cache_path) if cache_path.exists() else None
                        if cached is None or cached['key'] != key:
                            _, candidate_records, _ = prepare(directory/'limits',manifest)
                            paths = {};diagnostics = {}
                            for model in MODELS:
                                record = candidate_records[model];diagnostic = {}
                                if set(record['payload']['points']) != set(records[model]['payload']['points']):
                                    raise ValueError('Live displayed mass grids differ')
                                with supported_contours(record['requested_masses'],set(record['payload']['points']),diagnostic), capture_expected_paths() as captured:
                                    success = renderer.plot_contour(record['payload'],output/'unused.png',
                                        interpolation_diagnostics=diagnostic,**common(model))
                                if not success:
                                    raise RuntimeError('Original renderer failed: '+model)
                                paths[model] = [dict(vertices=p['path'].vertices.tolist(),
                                    codes=None if p['path'].codes is None else p['path'].codes.tolist())
                                    for p in captured if p['linestyle']=='-']
                                diagnostics[model] = diagnostic
                            cached = dict(key=key, paths=paths, diagnostics=json_ready(diagnostics),
                                category_definition=spec['category_definition'])
                            write_json(cache_path,cached)
                            print('LIVE_CANDIDATE_READY',cid,total,flush=True)
                        cached['stamp'] = stamp;caches[cid] = cached
                    row.update(status='plotted', completed=total, category_definition=cached['category_definition'])
                    for model in MODELS:
                        layers[model].append(dict(id=cid,color=color,paths=cached['paths'][model]))
                except Exception as error:
                    row.update(status='display_error',error=type(error).__name()+': '+str(error))
                    print('LIVE_DISPLAY_ERROR',cid,row['error'],flush=True)
            signature = [(r['id'],caches[r['id']]['key']) for r in rows if r['status']=='plotted']
            if signature != previous:
                for model in MODELS:
                    record = records[model];diagnostic = {};display = {}
                    with supported_contours(record['requested_masses'],set(record['payload']['points']),diagnostic), live_svg_display(layers[model],output/(model+'.svg'),output/(model+'.png')), contours_only_display(display,all_medians=True):
                        success = renderer.plot_contour(record['payload'],output/'unused.png',
                            interpolation_diagnostics=diagnostic,
                            primary_legend_label=r'w/o medium-$\Delta M$ SR (median)',**common(model))
                    if not success:
                        raise RuntimeError('Live baseline render failed')
                revision = str(time.time_ns());previous = signature
            now = datetime.now(timezone.utc).isoformat()
            write_json(output/'live_state.json',dict(status='live_diagnostic',updated_utc=now,
                plot_revision=revision,pid=os.getpid(),points_per_candidate=total,candidates=rows,
                baseline_manifest_sha256=baseline_digest,pools={str(p):sha256(p) for p in pools},
                render=render_key,expected_only=True,systematics=False,control_region_fit=False,
                evaluation_partition='validation',automatic_selection=False,full_workflow_complete=False))
            if once:
                break
            time.sleep(max(10,interval))
    finally:
        if server is not None:
            server.shutdown()
        lock.close()
    return dict(status='live_snapshot_ready')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limits', required=True, type=Path)
    p.add_argument('--grid', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sensitivity-map', action='store_true', help='Render compact group-freeze results from --limits; no inference or contour interpolation')
    p.add_argument('--allow-partial', action='store_true', help='Audited diagnostic only; never a completed contour grid')
    p.add_argument('--live-snapshot', action='store_true', help='Read-only in-progress six-era view; requires --allow-partial')
    p.add_argument('--overlay-limits', type=Path)
    p.add_argument('--overlay-grid', type=Path)
    p.add_argument('--second-overlay-limits', type=Path)
    p.add_argument('--second-overlay-grid', type=Path)
    p.add_argument('--third-overlay-limits', type=Path)
    p.add_argument('--third-overlay-grid', type=Path)
    p.add_argument('--three-quantile-low',action='store_true',help='Export validated median/one-sigma low contour separately from full five-quantile status')
    p.add_argument('--contours-only',action='store_true',help='No heatmap/colorbar; standalone medians and combined median plus one sigma')
    p.add_argument('--from-plot-snapshot',type=Path,help='Reuse verified compact four-region exports without reopening fit ROOT inputs')
    p.add_argument('--ranking',type=Path,help='Verify plotted candidate against complete CLs ranking and export compact mass/cell improvements')
    p.add_argument('--median-medium',action='store_true',help='Export only verified medium medians; retain separate full-five-quantile failure accounting')
    p.add_argument('--before-plot-snapshot',type=Path,help='Explicit before/after combined overlay; use --from-plot-snapshot for after and --overlay-grid for before')
    p.add_argument('--comparison-note',help='Required provenance/sampling qualification for a historical before/after comparison')
    p.add_argument('--test-comparison',action='store_true',help='Frozen same-test before/after suite; baseline appears in every figure')
    p.add_argument('--validation-comparison',action='store_true',help='C01 validation score edges only; baseline/before medians and after one sigma')
    p.add_argument('--cutbased-comparison',action='store_true',help='All-MC physical-cut winner plus fixed w/o medium SR baseline; no NN-validation overlay')
    p.add_argument('--cutbased-preview',action='store_true',help='Explicit complete cut/GNN/baseline candidates; no ranking or final-adoption claim; MC sampling differs')
    p.add_argument('--met-category-comparison',action='store_true',help='Complete S1 (--grid), S0 (--overlay-grid) and baseline; optional validation GNN in --second-overlay-grid; no adoption')
    p.add_argument('--global-cut-comparison',action='store_true',help='Two complete exact-global-cut combined grids and fixed baseline, identical parent MC and MET bins; explicit proposals, not an optimized-winner claim')
    p.add_argument('--comparison-style-reference',type=Path,help='Existing plot manifest fixing renderer, references and numerical runtime')
    p.add_argument('--baseline-limits',type=Path)
    p.add_argument('--baseline-grid',type=Path)
    p.add_argument('--live-candidates',action='store_true',help='Temporary, auto-updating full-grid candidate median viewer')
    p.add_argument('--refresh-live-page',action='store_true',help='Update the live HTML only without interrupting monitoring or rendering')
    p.add_argument('--candidate-pool',type=Path,action='append',default=[])
    p.add_argument('--live-interval',type=int,default=60)
    p.add_argument('--live-port',type=int,help='Serve only the generated view on loopback (use SSH forwarding)')
    p.add_argument('--live-once',action='store_true')
    p.add_argument('--live-style-reference',type=Path,help='Existing plot manifest fixing the numerical runtime')
    a = p.parse_args()
    if a.live_snapshot and (not a.allow_partial or a.sensitivity_map or a.refresh_live_page or a.live_candidates
            or a.met_category_comparison or a.global_cut_comparison or a.cutbased_preview or a.cutbased_comparison
            or a.validation_comparison or a.test_comparison):
        p.error('Live snapshot requires the standalone --allow-partial mode')
    if a.sensitivity_map:
        from TROTASR.workflows.renderers.sensitivity_map import render as render_sensitivity
        result=render_sensitivity(a.limits,a.grid,a.output)
    elif a.refresh_live_page:
        output=internal_path(a.output)
        if not (output/'live_state.json').is_file() or a.live_candidates or a.test_comparison or a.validation_comparison or a.cutbased_comparison or a.cutbased_preview or a.met_category_comparison or a.global_cut_comparison:
            p.error('HTML refresh requires an existing live view and no inference mode')
        pending=output/'index.html.pending'
        pending.write_text(live_page(),encoding='utf-8')
        pending.replace(output/'index.html')
        result=dict(status='live_page_updated')
    elif a.live_candidates:
        if not a.candidate_pool or not a.live_style_reference or a.test_comparison or a.validation_comparison or a.cutbased_comparison or a.cutbased_preview or a.met_category_comparison or a.global_cut_comparison or a.allow_partial:
            p.error('Live view requires explicit candidate pools and complete validation grids')
        result=live_candidates(a.limits,a.grid,a.output,a.candidate_pool,a.live_interval,a.live_port,a.live_once,a.live_style_reference)
    elif a.met_category_comparison or a.global_cut_comparison:
        if (a.cutbased_comparison or a.cutbased_preview or a.test_comparison or a.validation_comparison or a.ranking
                or a.allow_partial or a.from_plot_snapshot or a.before_plot_snapshot or a.third_overlay_grid
                or a.third_overlay_limits or a.three_quantile_low or a.median_medium or a.comparison_note or a.candidate_pool
                or (a.met_category_comparison and a.global_cut_comparison)):
            p.error('S0/S1 comparison cannot mix ranked-winner, partial, test or snapshot modes')
        result=render_met_category_comparison(a.limits,a.grid,a.output,a.overlay_limits,a.overlay_grid,
            a.baseline_limits,a.baseline_grid,a.comparison_style_reference,a.second_overlay_limits,a.second_overlay_grid,
            global_cuts=a.global_cut_comparison)
    elif a.cutbased_preview:
        if (a.cutbased_comparison or a.test_comparison or a.validation_comparison or a.ranking or a.allow_partial
                or a.from_plot_snapshot or a.before_plot_snapshot or a.second_overlay_grid or a.second_overlay_limits
                or a.third_overlay_grid or a.third_overlay_limits or a.three_quantile_low or a.median_medium
                or a.comparison_note or a.candidate_pool):
            p.error('Complete-candidate preview cannot mix ranking, partial, test or snapshot modes')
        result=render_cutbased_preview(a.limits,a.grid,a.output,a.overlay_limits,a.overlay_grid,
            a.baseline_limits,a.baseline_grid,a.comparison_style_reference)
    elif a.cutbased_comparison:
        if (a.test_comparison or a.validation_comparison or a.allow_partial or a.from_plot_snapshot or a.before_plot_snapshot
                or a.overlay_grid or a.overlay_limits or a.second_overlay_grid or a.second_overlay_limits
                or a.third_overlay_grid or a.third_overlay_limits or a.three_quantile_low or a.median_medium
                or a.comparison_note or a.candidate_pool):
            p.error('All-MC cut-based comparison cannot mix NN-validation, test, partial or historical snapshot modes')
        result=render_cutbased_comparison(a.limits,a.grid,a.output,a.baseline_limits,a.baseline_grid,
                                         a.ranking,a.comparison_style_reference)
    elif a.validation_comparison:
        if (a.test_comparison or a.allow_partial or a.from_plot_snapshot or a.before_plot_snapshot or
                a.second_overlay_grid or a.second_overlay_limits or a.third_overlay_grid or a.third_overlay_limits
                or a.three_quantile_low or a.median_medium or a.comparison_note or a.candidate_pool):
            p.error('Validation score comparison cannot mix test, partial or historical snapshot modes')
        result=render_validation_comparison(a.limits,a.grid,a.output,a.overlay_limits,a.overlay_grid,
            a.baseline_limits,a.baseline_grid,a.ranking,a.comparison_style_reference)
    elif a.test_comparison:
        if (a.allow_partial or a.ranking or a.from_plot_snapshot or a.before_plot_snapshot or
                a.second_overlay_grid or a.second_overlay_limits or a.third_overlay_grid or a.third_overlay_limits
                or a.three_quantile_low or a.median_medium or a.comparison_note or a.comparison_style_reference):
            p.error('Frozen test suite cannot mix validation, partial coverage or historical snapshots')
        result=render_test_comparison(a.limits,a.grid,a.output,a.overlay_limits,a.overlay_grid,
                                      a.baseline_limits,a.baseline_grid)
    else:
        if a.baseline_limits or a.baseline_grid:p.error('Baseline pair requires an explicit test/validation/cutbased/met-category comparison')
        if a.comparison_style_reference:p.error('Comparison style reference requires an explicit validation/cutbased/met-category comparison')
        result=render(a.limits, a.grid, a.output, a.allow_partial,a.overlay_limits,a.overlay_grid,
                     a.second_overlay_limits,a.second_overlay_grid,a.third_overlay_limits,a.third_overlay_grid,
                     a.three_quantile_low,a.contours_only,a.from_plot_snapshot,a.ranking,a.median_medium,
                     a.before_plot_snapshot,a.comparison_note,a.live_snapshot)
    print(result['status'])
