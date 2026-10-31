"""Contour input/accounting tests; no PDF authoring and no fabricated limits."""
import copy
import unittest
from TROTASR.utils.io import read_json, sha256
from TROTASR.utils.paths import ROOT
from TROTASR.workflows.plotting_limit_contour import validate_collected, sr_bin_counts


def fixture():
    return dict(status='complete', scope='full_nominal_production', observed_sr_used=False,
        requested_point_count=1, collected_point_count=1, missing_points=[],
        points={'mStop1200_mLSP500': dict(mStop=1200, mLSP=500, expected_m2=.5,
            expected_m1=.7, expected=1., expected_p1=1.3, expected_p2=1.6)})


class NominalContourTests(unittest.TestCase):
    def validation_comparison_fixture(self):
        import hashlib
        import json
        from TROTASR.tests.test_sr_only_comparison import ValidationCandidateTests
        from TROTASR.workflows.plotting_limit_contour import VALIDATION_REFERENCE_ID, VALIDATION_REFERENCE_SHA256
        helper=ValidationCandidateTests();spec=helper.candidate(4)
        fixed={'templates/template_'+r+'_'+y+'.root':r+y
               for r in ('highdm','lowdm') for y in ('2024','2025')}
        spec.update(candidate_id=VALIDATION_REFERENCE_ID,contract_sha256=VALIDATION_REFERENCE_SHA256,
                    model=dict(id='tr_sig005_latest',epoch=17,checkpoint_sha256='a'*64),
                    fixed_baseline_files=fixed,contract_path='/unused/contract.json')
        baseline=helper.grid();baseline.update(files=fixed,parent_grid_sha256='same',treatment='nominal',
            mc_statistics_assumption='same',sr_bins_per_year=dict(highdm=118,lowdm=30,mediumdm=0))
        before=helper.grid(spec);before.update(parent_grid_sha256='same',treatment='nominal',
            mc_statistics_assumption='same',sr_bins_per_year=dict(highdm=118,lowdm=30,mediumdm=20))
        after=copy.deepcopy(before)
        candidate=after['medium_candidate'];candidate['candidate_id']='changed_score_edges'
        candidate['category_definition']['groups'][0]['score_edges'][1]=.1
        definition=candidate['category_definition']
        digest=hashlib.sha256(json.dumps({k:v for k,v in definition.items() if k!='id'},
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
        definition['id']='categories_'+digest[:12];candidate['category_definition_sha256']=digest
        candidate['contract_sha256']='c'*64
        return baseline,before,after

    def test_validation_score_comparison_requires_fixed_c01_model_and_physical_partitions(self):
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        from TROTASR.workflows.plotting_limit_contour import require_fixed_validation_comparison
        baseline,before,after=self.validation_comparison_fixture()
        result=require_fixed_validation_comparison(after,before,baseline,FIXED_BASELINE_SHA256)
        self.assertEqual(len(result['physical_partitions']),4)
        self.assertEqual(result['model']['epoch'],17)
        for mutate in (lambda g:g.update(medium_test_comparison={}),lambda g:g.update(evaluation_partition='test'),
                       lambda g:g.update(auto_mc_stats=[0,1,1]),lambda g:g['points'].pop(),
                       lambda g:g['medium_candidate']['model'].update(epoch=18),
                       lambda g:g['medium_candidate']['model'].update(checkpoint_sha256='b'*64),
                       lambda g:g['medium_candidate']['category_definition']['groups'][0].update(conditions=[['njet',4.,True]])):
            bad=copy.deepcopy(after);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_validation_comparison(bad,before,baseline,FIXED_BASELINE_SHA256)
        wrong=copy.deepcopy(before);wrong['medium_candidate']['contract_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'C01'):require_fixed_validation_comparison(after,wrong,baseline,FIXED_BASELINE_SHA256)
        with self.assertRaisesRegex(ValueError,'baseline'):require_fixed_validation_comparison(after,before,baseline,'wrong')

    def validation_input_fixture(self):
        from TROTASR.workflows.build_nominal_grid import CANDIDATE_PARTITION
        return {year:dict(model=dict(checkpoint_sha256='a'*64),
            provenance=dict(CANDIDATE_PARTITION,validation_scores_sha256='b'*64,
                score_provenance=dict(partition='validation',checkpoint_sha256='a'*64,
                    cache_manifest_sha256='c'*64,contract_sha256='d'*64)),
            background_processes=['TT'],signal_names=['T2tt_mStop1200_mLSP500'],
            years={year:dict(partition='validation',weight_scale=10.,variance_scale=100.,
                luminosity_fb=lumi,events=40,samples={name:dict(entries=[1.]*20,sumw=[2.]*20,sumw2=[4.]*20)
                    for name in ('TT','T2tt_mStop1200_mLSP500')})})
            for year,lumi in [('2024',109.82),('2025',110.84)]}

    def test_validation_inputs_preserve_population_and_weights_after_rebinning(self):
        from TROTASR.workflows.plotting_limit_contour import require_same_validation_inputs
        before=self.validation_input_fixture();after=copy.deepcopy(before)
        row=after['2024']['years']['2024']['samples']['TT']
        row['sumw'][:2]=[1.,3.+1e-8];row['sumw2'][:2]=[2.,6.]
        identities=require_same_validation_inputs(before,after)
        self.assertTrue(identities['2024']['physical_category_totals_preserved'])
        self.assertLess(identities['2024']['maximum_absolute_sum_difference'],1e-7)
        for mutate in (lambda p:p['2024']['provenance'].update(validation_scores_sha256='f'*64),
                       lambda p:p['2025']['provenance'].update(test_events_used=True),
                       lambda p:p['2024']['provenance']['score_provenance'].update(cache_manifest_sha256='e'*64),
                       lambda p:p['2025']['years']['2025'].update(weight_scale=1.),
                       lambda p:p['2024']['years']['2024']['samples']['TT']['sumw'].__setitem__(0,10.),
                       lambda p:p['2024']['years']['2024']['samples']['TT']['entries'].__setitem__(0,2.)):
            bad=copy.deepcopy(before);mutate(bad)
            with self.assertRaises(ValueError):require_same_validation_inputs(before,bad)
        with self.assertRaises(ValueError):require_same_validation_inputs(before,{'2024':before['2024']})

    def test_validation_loader_uses_only_authenticated_validation_exports(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import load_validation_plot_inputs
        _,before,_=self.validation_comparison_fixture();spec=before['medium_candidate']
        with patch('TROTASR.workflows.build_nominal_grid.load_medium_candidate',return_value=spec), \
                patch('TROTASR.workflows.build_nominal_grid.read_medium_validation',return_value=({'verified':True},{})) as read, \
                patch('TROTASR.workflows.build_nominal_grid.load_medium_test_comparison') as forbidden:
            result=load_validation_plot_inputs(spec)
            self.assertEqual(set(result),{'2024','2025'})
            self.assertEqual([c.args[0].name for c in read.call_args_list],['medium_2024.json','medium_2025.json'])
            forbidden.assert_not_called()
        with patch('TROTASR.workflows.build_nominal_grid.load_medium_candidate',return_value={}):
            with self.assertRaisesRegex(ValueError,'contract changed'):load_validation_plot_inputs(spec)

    def test_validation_changed_candidate_requires_completed_ranking_before_fit_read(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import render_validation_comparison
        baseline,before,after=self.validation_comparison_fixture()
        exports=self.validation_input_fixture()
        with patch('TROTASR.workflows.plotting_limit_contour.read_json',side_effect=[after,before,baseline]), \
                patch('TROTASR.workflows.plotting_limit_contour.sha256',return_value=before['medium_candidate']['fixed_baseline_manifest_sha256']), \
                patch('TROTASR.workflows.plotting_limit_contour.load_validation_plot_inputs',return_value=exports), \
                patch('TROTASR.workflows.plotting_limit_contour.prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'complete validation CLs ranking'):
                render_validation_comparison(ROOT/'after',ROOT/'after-grid',ROOT/'unused',ROOT/'before',ROOT/'before-grid',
                    ROOT/'baseline',ROOT/'baseline-grid',style_reference=ROOT/'style')
            prepare.assert_not_called()

    def cutbased_comparison_fixture(self):
        from TROTASR.tests.test_sr_only_comparison import CutbasedCandidateTests,ValidationCandidateTests
        helper=CutbasedCandidateTests();candidate=helper.candidate(3)
        fixed={'templates/template_'+r+'_'+y+'.root':r+y
               for r in ('highdm','lowdm') for y in ('2024','2025')}
        candidate.update(fixed_baseline_files=fixed,contract_path='/unused/contract.json',contract_sha256='c'*64)
        baseline=ValidationCandidateTests().grid();after=helper.grid(candidate)
        for grid in (baseline,after):
            grid.update(files=dict(fixed),parent_grid_sha256='same',treatment='nominal',mc_statistics_assumption='same')
        baseline['sr_bins_per_year']=dict(highdm=118,lowdm=30,mediumdm=0)
        after['sr_bins_per_year']=dict(highdm=118,lowdm=30,mediumdm=3)
        return baseline,after

    def cutbased_ranking_fixture(self):
        from TROTASR.workflows.build_nominal_grid import CUTBASED_CAVEAT
        from TROTASR.workflows.combine_limits import comparison_candidate_score
        from TROTASR.tests.test_sr_only_comparison import CutbasedCandidateTests
        baseline,after=self.cutbased_comparison_fixture();candidate=after['medium_candidate']
        other=CutbasedCandidateTests().candidate(4)
        other.update(fixed_baseline_files=candidate['fixed_baseline_files'],contract_path='/unused/other.json',contract_sha256='d'*64)
        references={p['model']+'/'+p['mass']:1. for p in baseline['points']}
        rows=[]
        for spec,ratio in ((candidate,.9),(other,1.1)):
            score=comparison_candidate_score(baseline['points'],references,dict.fromkeys(references,ratio))
            rows.append(dict(score,candidate_id=spec['candidate_id'],pending={},cutbased_candidate=spec))
        report=dict(schema='combined_cutbased_expected_cls_ranking_v1',evaluation_partition='all_mc',
            status='complete',baseline_pending={},observed_sr_used=False,winner=candidate['candidate_id'],
            policy=candidate['ranking_policy'],baseline_manifest_sha256=candidate['fixed_baseline_manifest_sha256'],
            candidates=rows,ranking_of_complete_candidates=[s['candidate_id'] for s in (candidate,other)],
            claim='Within the all-MC cut-based pool, not independent validation',neural_network_used=False,
            independent_validation=False,statistical_caveat=CUTBASED_CAVEAT,populations=candidate['populations'],
            historical_partitions_used=['train','validation','test'])
        return baseline,after,report

    def test_cutbased_comparison_keeps_fixed_hl_and_arbitrary_leaf_bins(self):
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        from TROTASR.workflows.plotting_limit_contour import require_fixed_cutbased_comparison
        baseline,after=self.cutbased_comparison_fixture()
        self.assertEqual(sr_bin_counts(after),(118,30))
        result=require_fixed_cutbased_comparison(after,baseline,FIXED_BASELINE_SHA256)
        self.assertEqual(result['populations'],after['medium_candidate']['populations'])
        self.assertEqual(len(result['fixed_high_low_templates']),4)
        mutations=(lambda g:g.update(evaluation_partition='validation'),lambda g:g.update(medium_test_comparison={}),
            lambda g:g.update(independent_validation=True),lambda g:g.update(auto_mc_stats=[0,1,1]),
            lambda g:g['points'].pop(),lambda g:g['files'].update({'templates/template_highdm_2024.root':'changed'}))
        for mutate in mutations:
            bad=copy.deepcopy(after);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_cutbased_comparison(bad,baseline,FIXED_BASELINE_SHA256)
        with self.assertRaises(ValueError):require_fixed_cutbased_comparison(after,baseline,'other')
        _,_,nn=self.validation_comparison_fixture()
        with self.assertRaises(ValueError):require_fixed_cutbased_comparison(nn,baseline,FIXED_BASELINE_SHA256)

    def test_cutbased_loader_authenticates_own_exports_and_validation_rejects_them(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import load_cutbased_plot_inputs,load_validation_plot_inputs
        _,grid=self.cutbased_comparison_fixture();spec=grid['medium_candidate']
        with patch('TROTASR.workflows.build_nominal_grid.load_medium_candidate',return_value=spec), \
                patch('TROTASR.workflows.build_nominal_grid.read_medium_cutbased',return_value=({'verified':True},{})) as read, \
                patch('TROTASR.workflows.build_nominal_grid.read_medium_validation') as forbidden:
            self.assertEqual(set(load_cutbased_plot_inputs(spec)),{'2024','2025'})
            self.assertEqual([c.args[0].name for c in read.call_args_list],['medium_2024.json','medium_2025.json'])
            forbidden.assert_not_called()
            with self.assertRaisesRegex(ValueError,'not NN-validation'):load_validation_plot_inputs(spec)
        with patch('TROTASR.workflows.build_nominal_grid.load_medium_candidate',return_value={}):
            with self.assertRaisesRegex(ValueError,'contract changed'):load_cutbased_plot_inputs(spec)

    def test_cutbased_ranking_summary_is_all_mc_and_preserves_all955_values(self):
        from TROTASR.workflows.plotting_limit_contour import summarize_ranking
        _,grid,report=self.cutbased_ranking_fixture();summary=summarize_ranking(report,grid)
        self.assertEqual(summary['schema'],'combined_cutbased_optimization_plot_summary_v1')
        self.assertEqual(summary['evaluation_partition'],'all_mc');self.assertFalse(summary['independent_validation'])
        self.assertFalse(summary['neural_network_used']);self.assertNotIn('validation_candidate',summary)
        self.assertEqual(summary['cutbased_candidate'],grid['medium_candidate'])
        self.assertEqual(summary['points'],report['candidates'][0]['points']);self.assertEqual(len(summary['points']),955)
        self.assertEqual(sum(s['points'] for s in summary['models'].values()),955)

    def test_cutbased_ranking_rejects_nn_mixture_changed_population_or_incomplete_grid(self):
        from TROTASR.workflows.plotting_limit_contour import summarize_ranking
        _,grid,original=self.cutbased_ranking_fixture()
        mutations=(lambda r:r.update(schema='combined_expected_cls_ranking_v1'),lambda r:r.update(evaluation_partition='validation'),
            lambda r:r.update(independent_validation=True),lambda r:r.update(statistical_caveat=''),
            lambda r:r['populations']['2024'].update(physical_weights_sha256='e'*64),
            lambda r:r['candidates'][1]['cutbased_candidate']['populations']['2025'].update(feature_cache_sha256='e'*64),
            lambda r:r['candidates'][1].update(validation_candidate={}),lambda r:r['candidates'][1].update(verified=954),
            lambda r:r['candidates'][0].update(mean_log_ratio=0.),lambda r:r.update(winner='other'))
        for mutate in mutations:
            report=copy.deepcopy(original);mutate(report)
            with self.assertRaises(ValueError):summarize_ranking(report,grid)
        nn,_=self.ranking_fixture()
        with self.assertRaisesRegex(ValueError,'relabelled'):summarize_ranking(original,nn)

    def test_cutbased_missing_ranking_fails_before_reading_fit_outputs(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import render_cutbased_comparison
        baseline,after=self.cutbased_comparison_fixture()
        with patch('TROTASR.workflows.plotting_limit_contour.read_json',side_effect=[after,baseline]), \
                patch('TROTASR.workflows.plotting_limit_contour.prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'complete all-MC CLs ranking'):
                render_cutbased_comparison(ROOT/'after',ROOT/'after-grid',ROOT/'unused',ROOT/'baseline',ROOT/'baseline-grid',
                    ranking=None,style_reference=ROOT/'style')
            prepare.assert_not_called()

    def test_cutbased_requires_explicit_display_branch(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import render
        _,grid=self.cutbased_comparison_fixture()
        with patch('TROTASR.workflows.plotting_limit_contour.prepare',return_value=(grid,{},'renderer')):
            with self.assertRaisesRegex(ValueError,'--cutbased-comparison'):
                render(ROOT/'limits',ROOT/'grid',ROOT/'unused')

    def test_cutbased_preview_requires_full955_fixed_hl_and_validation_gnn(self):
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        from TROTASR.workflows.plotting_limit_contour import require_fixed_cutbased_preview
        baseline,cut=self.cutbased_comparison_fixture()
        _,_,gnn=self.validation_comparison_fixture()
        result=require_fixed_cutbased_preview(cut,gnn,baseline,FIXED_BASELINE_SHA256)
        self.assertEqual(result['fixed_high_low_templates'],cut['medium_candidate']['fixed_baseline_files'])
        mutations=(lambda g:g.update(evaluation_partition='test'),lambda g:g.update(medium_test_comparison={}),
            lambda g:g.update(auto_mc_stats=[0,1,1]),lambda g:g['points'].pop(),
            lambda g:g['files'].update({'templates/template_lowdm_2025.root':'changed'}),
            lambda g:g.update(treatment='other'),lambda g:g.update(bins=42))
        for mutate in mutations:
            bad=copy.deepcopy(gnn);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_cutbased_preview(cut,bad,baseline,FIXED_BASELINE_SHA256)
        with self.assertRaises(ValueError):require_fixed_cutbased_preview(cut,cut,baseline,FIXED_BASELINE_SHA256)

    def test_cutbased_preview_cannot_claim_ranked_winner(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import render_candidate_comparison
        with patch('TROTASR.workflows.plotting_limit_contour.prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'separate from final ranking'):
                render_candidate_comparison(None,None,None,None,None,None,None,ROOT/'ranking',None,
                                           cutbased=True,preview=True)
            prepare.assert_not_called()

    def test_cutbased_preview_requires_all_three_pairs(self):
        from TROTASR.workflows.plotting_limit_contour import render_cutbased_preview
        with self.assertRaisesRegex(ValueError,'grid/limit pairs'):
            render_cutbased_preview(ROOT/'cut',ROOT/'cut-grid',ROOT/'unused',None,None,
                                    ROOT/'base',ROOT/'base-grid',ROOT/'style')

    def test_cutbased_shared_render_flow_keeps_style_and_waits_for_visual_qa(self):
        self.check_cutbased_render_flow()

    def test_cutbased_preview_three_curves_keep_mc_caveat_and_no_ranking_claim(self):
        self.check_cutbased_render_flow(preview=True)

    def test_met_cutbased_preview_preserves_fixed_baseline_and_sampling(self):
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        from TROTASR.workflows.plotting_limit_contour import require_fixed_cutbased_preview,require_fixed_cutbased_comparison
        baseline,_,cut=self.met_category_comparison_fixture()
        _,_,gnn=self.validation_comparison_fixture()
        result=require_fixed_cutbased_preview(cut,gnn,baseline,FIXED_BASELINE_SHA256)
        self.assertEqual(result['parent_populations'],cut['medium_candidate']['parent_populations'])
        self.assertNotIn('populations',result)
        with self.assertRaises(ValueError):require_fixed_cutbased_comparison(cut,baseline,FIXED_BASELINE_SHA256)
        for mutate in (lambda g:g.update(evaluation_partition='validation'),
                       lambda g:g.update(auto_mc_stats=[0,1,1]),lambda g:g['points'].pop(),
                       lambda g:g['files'].update({'templates/template_lowdm_2025.root':'changed'})):
            bad=copy.deepcopy(cut);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_cutbased_preview(bad,gnn,baseline,FIXED_BASELINE_SHA256)

    def test_met_cutbased_gnn_preview_reuses_renderer_and_retains_selection(self):
        self.check_cutbased_render_flow(preview=True,met_preview=True)

    def test_met_cutbased_final_test_preview_keeps_seventy_percent_normalization(self):
        self.check_cutbased_render_flow(preview=True,met_preview=True,gnn_test=True)

    def met_category_comparison_fixture(self):
        from TROTASR.tests.test_sr_only_comparison import MetCategoryCandidateTests
        baseline,_=self.cutbased_comparison_fixture();helper=MetCategoryCandidateTests();grids=[]
        for name in ('S0','S1'):
            c=helper.candidate(name);c.update(fixed_baseline_files=baseline['files'],contract_path='/unused/'+name+'.json',contract_sha256=name)
            grid=helper.grid(c);grid.update(files=dict(baseline['files']),parent_grid_sha256='same',treatment='nominal',
                mc_statistics_assumption='same',sr_bins_per_year=dict(highdm=118,lowdm=30,mediumdm=46))
            grids.append(grid)
        return baseline,*grids

    def test_met_plot_requires_same_parent_and_exact_s0_s1(self):
        from TROTASR.workflows.plotting_limit_contour import require_fixed_met_category_comparison
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        baseline,s0,s1=self.met_category_comparison_fixture()
        self.assertEqual(require_fixed_met_category_comparison(s1,s0,baseline,FIXED_BASELINE_SHA256)['parent_populations'],
                         s0['medium_candidate']['parent_populations'])
        with self.assertRaises(ValueError):require_fixed_met_category_comparison(s0,s1,baseline,FIXED_BASELINE_SHA256)
        for mutate in (lambda g:g['points'].pop(),lambda g:g.update(auto_mc_stats=[0,1,1]),
                       lambda g:g['medium_candidate']['parent_populations']['2024'].update(physical_weights_sha256='f'*64)):
            bad=copy.deepcopy(s1);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_met_category_comparison(bad,s0,baseline,FIXED_BASELINE_SHA256)

    def global_cut_comparison_fixture(self):
        import hashlib,json
        from TROTASR.tests.test_sr_only_comparison import MetCategoryCandidateTests
        from TROTASR.workflows.build_nominal_grid import met_category_selection,met_category_caveat
        baseline,before,after=self.met_category_comparison_fixture()
        for grid,threshold in ((before,50.),(after,75.)):
            c=MetCategoryCandidateTests().exact_candidate();cuts=c['selection']['global_cuts']
            cuts[0]['threshold']=threshold
            name='X'+hashlib.sha256(json.dumps(cuts,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
            c['selection']=dict(met_category_selection(name,cuts),label='Synthetic exact cut')
            c.update(candidate_id='met_categories_'+name+'_'+c['category_definition_sha256'][:12],
                fixed_baseline_files=baseline['files'],contract_path='/unused/'+name+'.json',contract_sha256='x')
            grid.update(medium_candidate=c,parent_population=c['parent_populations'],selection=c['selection'],
                full_tr_retained=False,statistical_caveat=met_category_caveat(c))
        return baseline,before,after

    def test_exact_global_plot_requires_same_observables_and_explicit_mode(self):
        from TROTASR.workflows.plotting_limit_contour import require_fixed_met_category_comparison
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        baseline,before,after=self.global_cut_comparison_fixture()
        require_fixed_met_category_comparison(after,before,baseline,FIXED_BASELINE_SHA256,True)
        with self.assertRaises(ValueError):require_fixed_met_category_comparison(after,before,baseline,FIXED_BASELINE_SHA256)
        after['medium_candidate']['exact_kinematics']['years']['2024']['manifest_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'exact observables'):
            require_fixed_met_category_comparison(after,before,baseline,FIXED_BASELINE_SHA256,True)

    def test_exact_global_plot_is_same_mc_not_a_winner_claim(self):
        self.check_cutbased_render_flow(met_categories=True,global_cuts=True)

    def test_met_plot_cannot_be_a_ranked_winner(self):
        from TROTASR.workflows.plotting_limit_contour import render_candidate_comparison,render
        from unittest.mock import patch
        with self.assertRaisesRegex(ValueError,'ranked cut/NN winner'):
            render_candidate_comparison(None,None,None,None,None,None,None,ROOT/'rank',None,met_categories=True)
        _,_,g=self.met_category_comparison_fixture()
        with patch('TROTASR.workflows.plotting_limit_contour.prepare',return_value=(g,{},'renderer')):
            with self.assertRaisesRegex(ValueError,'--met-category-comparison'):render(ROOT/'limits',ROOT/'grid',ROOT/'unused')

    def test_met_s0_s1_render_keeps_selected_rejected_accounting(self):
        self.check_cutbased_render_flow(met_categories=True)

    def test_met_s0_s1_optional_gnn_is_only_a_sampling_qualified_reference(self):
        self.check_cutbased_render_flow(met_categories=True,gnn_reference=True)

    def check_cutbased_render_flow(self,preview=False,met_categories=False,gnn_reference=False,global_cuts=False,met_preview=False,gnn_test=False):
        import contextlib
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from TROTASR.workflows import plotting_limit_contour as plots
        from TROTASR.workflows.build_nominal_grid import FIXED_BASELINE_SHA256
        baseline,after,ranking=self.cutbased_ranking_fixture()
        _,_,gnn=self.validation_comparison_fixture()
        if gnn_test:
            from TROTASR.tests.test_sr_only_comparison import FrozenTestComparisonTests
            spec=gnn.pop('medium_candidate')
            spec.update(schema='medium_dm_frozen_test_candidate_v1',candidate_id='selected',
                test_events_used=True,historical_test_already_opened=True,reselection_allowed=False,
                rebinning_allowed=False,comparison_freeze_sha256='a'*64)
            gnn.update(medium_test_comparison=spec,evaluation_partition='test',reselection_allowed=False,
                rebinning_allowed=False,test_support_is_diagnostic_only=True,
                medium_test_provenance=FrozenTestComparisonTests().identities())
        s0=None
        if met_categories:baseline,s0,after=self.met_category_comparison_fixture()
        if met_preview:baseline,_,after=self.met_category_comparison_fixture()
        if global_cuts:baseline,s0,after=self.global_cut_comparison_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            folder=Path(directory);output=folder/'plots';ap=folder/'after.json';bp=folder/'baseline.json'
            gp=folder/'gnn.json';s0p=folder/'s0.json'
            rp=folder/'ranking.json';sp=folder/'style.json';config_path=ROOT/'stats/contour_config.json'
            refs={m:dict(path='tests/fixture_'+m+'.json',sha256='sha') for m in (*plots.MODELS,'thermal_relic')}
            config=dict(references=refs,theory=dict(path='tests/fixture_theory.txt',sha256='sha'))
            runtime=dict(test='mock-render-only')
            mapping={ap:after,gp:gnn,s0p:s0,bp:baseline,rp:ranking,sp:dict(contract=dict(runtime=runtime,renderer_sha256='sha',config_sha256='sha')),
                     config_path:config}
            read_original=plots.read_json
            def read(path):return mapping[Path(path)] if Path(path) in mapping else read_original(path)
            def digest(path):return FIXED_BASELINE_SHA256 if Path(path)==bp else 'sha'
            def prepared(grid,value):
                records={}
                for model in plots.MODELS:
                    masses=[p['mass'] for p in grid['points'] if p['model']==model]
                    payload=dict(points={m:dict(expected=value) for m in masses})
                    records[model]=dict(payload=payload,full_payload=payload,input_sha256='input',
                        requested_masses=masses,full_grid_point_count=len(masses),out_of_plot_range=[])
                return grid,records,'sha'
            def prepare(limits,path):
                grid,value={ap:(after,.9),gp:(gnn,.95),s0p:(s0,.93),bp:(baseline,1.)}[Path(path)]
                return prepared(grid,value)
            calls=[]
            def mock_renderer(payload,path,**kwargs):
                calls.append((Path(path),kwargs))
                if Path(path).name!='unused.png':
                    Path(path).write_bytes(b'unit-test renderer receipt, not an image')
                    Path(path).with_suffix('.pdf').write_bytes(b'unit-test renderer receipt, not a PDF')
                return True
            exports={y:dict(provenance=dict(partition='all_mc'),coverage=dict(full_tr_retained=True)) for y in ('2024','2025')}
            def met_inputs(c):
                from TROTASR.tests.test_sr_only_comparison import MetCategoryCandidateTests
                return {y:MetCategoryCandidateTests().payload(c,y) for y in ('2024','2025')}
            with patch.object(plots,'read_json',side_effect=read),patch.object(plots,'sha256',side_effect=digest), \
                    patch.object(plots,'prepare',side_effect=prepare) as prepare_call, \
                    patch.object(plots,'load_cutbased_plot_inputs',return_value=exports), \
                    patch.object(plots,'load_met_category_plot_inputs',side_effect=met_inputs), \
                    patch.object(plots,'load_validation_plot_inputs',return_value=self.validation_input_fixture()) as gnn_loader, \
                    patch.object(plots,'load_test_plot_inputs',return_value=self.validation_input_fixture()) as test_loader, \
                    patch.object(plots,'summarize_ranking',wraps=plots.summarize_ranking) as ranker, \
                    patch.object(plots.renderer,'runtime_versions',return_value=runtime), \
                    patch.object(plots.renderer,'load_thermal_relic_contour',return_value='thermal'), \
                    patch.object(plots.renderer,'load_stop_pair_xsecs',return_value='xsecs'), \
                    patch.object(plots.renderer,'plot_contour',side_effect=mock_renderer), \
                    patch.object(plots,'supported_contours',side_effect=lambda *a:contextlib.nullcontext()) as masks, \
                    patch.object(plots,'capture_expected_paths',side_effect=lambda:contextlib.nullcontext([])), \
                    patch.object(plots,'additional_expected_paths',side_effect=lambda *a,**k:contextlib.nullcontext()) as baseline_paths, \
                    patch.object(plots,'contours_only_display',side_effect=lambda *a,**k:contextlib.nullcontext()) as display:
                if met_categories:
                    result=plots.render_met_category_comparison(folder/'after_limits',ap,output,folder/'s0_limits',s0p,
                        folder/'baseline_limits',bp,sp,folder/'gnn_limits' if gnn_reference else None,gp if gnn_reference else None,
                        global_cuts=global_cuts)
                elif preview:
                    result=plots.render_cutbased_preview(folder/'after_limits',ap,output,folder/'gnn_limits',gp,
                                                        folder/'baseline_limits',bp,sp)
                else:
                    result=plots.render_cutbased_comparison(folder/'after_limits',ap,output,folder/'baseline_limits',bp,rp,sp)
            self.assertEqual(prepare_call.call_count,(4 if gnn_reference else 3) if met_categories else 3 if preview else 2)
            self.assertEqual(masks.call_count,9 if gnn_reference else 6)
            self.assertTrue(all(len(c.args)==2 and not c.kwargs for c in prepare_call.call_args_list))
            self.assertEqual(display.call_count,3);self.assertEqual(baseline_paths.call_count,3)
            self.assertEqual(result['status'],'generated_pending_visual_QA');self.assertFalse(result['full_workflow_complete'])
            contract=result['contract'];self.assertFalse(contract['independent_validation'])
            self.assertNotIn('test_events_used',contract)
            if global_cuts:
                ranker.assert_not_called();gnn_loader.assert_not_called()
                self.assertEqual(contract['schema'],'medium_dm_exact_global_cut_comparison_v1')
                self.assertEqual(contract['selection_status'],'explicit_restricted_global_cut_pool')
                self.assertEqual(contract['cut_labels']['A'],after['medium_candidate']['selection'])
                self.assertFalse(contract['final_adoption']);self.assertFalse(contract['ranking_used'])
                self.assertTrue(all(c.kwargs['extra_overlays']==[] for c in baseline_paths.call_args_list))
            elif met_categories:
                ranker.assert_not_called()
                self.assertEqual(contract['schema'],'medium_dm_met_category_selection_comparison_v1')
                self.assertEqual(contract['selection_status'],'explicit_S0_S1_comparison')
                self.assertEqual(contract['before']['selection']['id'],'S0');self.assertEqual(contract['after']['selection']['id'],'S1')
                self.assertFalse(contract['after']['full_tr_retained']);self.assertTrue(contract['before']['full_tr_retained'])
                self.assertFalse(contract['final_adoption']);self.assertFalse(contract['ranking_used']);self.assertIsNone(contract['ranking'])
                self.assertFalse(contract['selection_inputs']['after']['2024']['coverage']['full_tr_retained'])
                if gnn_reference:
                    gnn_loader.assert_called_once_with(gnn['medium_candidate'])
                    self.assertFalse(contract['gnn_method_only_comparison'])
                    self.assertIn('10% validation MC',contract['gnn_reference_caveat'])
                    self.assertEqual(set(contract['variants']),{'after','before','baseline','reference'})
                    self.assertTrue(all(c.kwargs['extra_overlays'][0][1]=='GNN (10% MC, scaled) median' for c in baseline_paths.call_args_list))
                else:gnn_loader.assert_not_called()
            elif preview:
                ranker.assert_not_called()
                gnn_spec=gnn['medium_test_comparison'] if gnn_test else gnn['medium_candidate']
                if gnn_test:
                    test_loader.assert_called_once_with(gnn_spec);gnn_loader.assert_not_called()
                    self.assertFalse(contract['reselection_allowed']);self.assertFalse(contract['rebinning_allowed'])
                else:gnn_loader.assert_called_once_with(gnn_spec)
                self.assertEqual(set(contract['variants']),{'after','before','baseline'})
                self.assertEqual(contract['schema'],'medium_dm_cutbased_gnn_complete_candidate_preview_v1')
                self.assertEqual(contract['before'],gnn_spec)
                self.assertEqual(contract['evaluation_partition'],'mixed_explicit_candidate_populations')
                self.assertEqual(contract['selection_status'],'explicit_complete_candidate_preview')
                self.assertIsNone(contract['ranking']);self.assertFalse(contract['ranking_used'])
                self.assertFalse(contract['final_adoption']);self.assertFalse(contract['automatic_selection'])
                self.assertFalse(contract['method_only_comparison']);self.assertFalse(contract['functional_check_only'])
                sampling=(dict(partition='test',fraction=.7,weight_scale=1/.7,variance_scale=1/.49) if gnn_test else
                          dict(partition='validation',fraction=.1,weight_scale=10.,variance_scale=100.))
                self.assertEqual(contract['mc_sampling']['before'],sampling)
                self.assertEqual(contract['mc_sampling']['after'],dict(partition='all_mc',fraction=1.,weight_scale=1.,variance_scale=1.))
                self.assertEqual(contract['comparison_caveat'],plots.TEST_CUTBASED_PREVIEW_CAVEAT if gnn_test else plots.CUTBASED_PREVIEW_CAVEAT)
                self.assertEqual(contract['expected_line_colors'],dict(after='red',before='blue',baseline='#009E73'))
                self.assertFalse((output/'ranking_summary.json').exists())
                if met_preview:
                    self.assertEqual(contract['cut_selection'],after['medium_candidate']['selection'])
                    self.assertEqual(contract['category_definition'],after['medium_candidate']['category_definition'])
                    self.assertEqual(contract['cutbased_inputs']['2024']['provenance']['partition'],'all_mc')
            else:
                ranker.assert_called_once();gnn_loader.assert_not_called()
                self.assertEqual(set(contract['variants']),{'after','baseline'})
                self.assertEqual(contract['schema'],'medium_dm_all_mc_cutbased_comparison_v1')
                self.assertEqual(contract['evaluation_partition'],'all_mc');self.assertNotIn('before',contract)
                self.assertEqual(contract['expected_line_colors'],dict(after='red',baseline='#009E73'))
            self.assertEqual(sum(p['full_grid_points'] for p in result['plots']),955)
            rendered=[kwargs for path,kwargs in calls if path.name!='unused.png']
            self.assertEqual(len(rendered),3)
            for kwargs in rendered:
                if global_cuts:
                    self.assertEqual(kwargs['primary_legend_label'],r'Cut A combined $\pm1\sigma$')
                    self.assertEqual(kwargs['overlay_legend_label'],'Cut B combined (median)')
                elif met_categories:
                    self.assertEqual(kwargs['primary_legend_label'],r'S1 (100% MC) $\pm1\sigma$')
                    self.assertEqual(kwargs['overlay_legend_label'],'S0 (100% MC) median')
                elif preview:
                    cut_label='Synthetic S1 cut-based' if met_preview else 'Cut-based'
                    self.assertEqual(kwargs['primary_legend_label'],cut_label+r' (100% MC) $\pm1\sigma$')
                    self.assertEqual(kwargs['overlay_legend_label'],'GNN (70% test, scaled) median' if gnn_test else 'GNN (10% MC, scaled) median')
                    self.assertIn('overlay_limit_payload',kwargs)
                else:
                    self.assertEqual(kwargs['primary_legend_label'],r'All-MC cut-based combined $\pm1\sigma$')
                    self.assertNotIn('overlay_limit_payload',kwargs)
                self.assertIsNone(kwargs['analysis_label'])
                self.assertEqual(kwargs['thermal_relic'],'thermal');self.assertEqual(kwargs['interpolation_mode'],'topology-aware')
                self.assertEqual((kwargs['x_min'],kwargs['x_max'],kwargs['y_min'],kwargs['y_max']),(600.,1800.,1.,1800.))
            self.assertTrue(all(c.kwargs['standalone_medians'] for c in display.call_args_list))
            self.assertTrue(all(c.args[1]==r'w/o medium-$\Delta M$ SR (median)' for c in baseline_paths.call_args_list))
            self.assertTrue(all(c.kwargs['legend_index']==(2 if preview or met_categories else 1) for c in baseline_paths.call_args_list))

    def test_cutbased_baseline_legend_precedes_unchanged_run2_pair(self):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from TROTASR.workflows.plotting_limit_contour import additional_expected_paths
        for index,expected in ((1,['all-MC','baseline','Run2 observed','Run2 expected']),
                               (2,['all-MC','Run2 observed','baseline','Run2 expected'])):
            fig,ax=plt.subplots()
            try:
                handles=[Line2D([0],[0],label=name) for name in ('all-MC','Run2 observed','Run2 expected')]
                with additional_expected_paths([],'baseline',legend_index=index):
                    legend=ax.legend(handles=handles)
                self.assertEqual([t.get_text() for t in legend.get_texts()],expected)
            finally:plt.close(fig)

    def test_live_view_requires_complete_full_grid(self):
        from TROTASR.workflows.plotting_limit_contour import live_candidate_ready
        ready = dict(status='fits_complete_pending_plots',total=955,completed=955,
                     failed=0,observed_sr_used=False)
        self.assertTrue(live_candidate_ready(ready,955))
        self.assertTrue(live_candidate_ready(dict(ready,failed=[]),955))
        for change in (dict(completed=954),dict(failed=1),dict(total=954),
                       dict(status='running'),dict(pid=123),dict(observed_sr_used=True),
                       dict(failed=['one']),dict(failed=None)):
            self.assertFalse(live_candidate_ready(dict(ready,**change),955))

    def test_live_registry_deduplicates_only_identical_contracts(self):
        from unittest.mock import patch
        from pathlib import Path
        from TROTASR.workflows.plotting_limit_contour import live_candidate_registry
        row = dict(candidate_id='candidate_1',contract_sha256='a')
        pool = dict(fixed_baseline_manifest_sha256='base',candidates=[row])
        with patch('TROTASR.workflows.plotting_limit_contour.read_json',return_value=pool):
            self.assertEqual(len(live_candidate_registry([Path('a.json'),Path('b.json')],'base')),1)
            with self.assertRaisesRegex(ValueError,'different high'):
                live_candidate_registry([Path('a.json')],'wrong')
        other = dict(pool,candidates=[dict(row,contract_sha256='b')])
        with patch('TROTASR.workflows.plotting_limit_contour.read_json',side_effect=[pool,other]):
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                live_candidate_registry([Path('a.json'),Path('b.json')],'base')

    def test_live_svg_adds_identifiable_unchanged_paths(self):
        import tempfile
        from pathlib import Path
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from TROTASR.workflows.plotting_limit_contour import live_svg_display, live_page
        vertices = [[600.,200.],[1000.,500.],[1500.,100.]]
        layer = dict(id='candidate_1',color='#123456',paths=[dict(vertices=vertices,codes=None)])
        with tempfile.TemporaryDirectory() as directory:
            svg = Path(directory)/'view.svg';png = Path(directory)/'view.png'
            fig,ax = plt.subplots();ax.set_xlim(600,1800);ax.set_ylim(1,1800)
            with live_svg_display([layer],svg,png):
                fig.savefig(Path(directory)/'unused.png')
            self.assertTrue(png.is_file())
            self.assertIn('candidate-candidate_1-0',svg.read_text())
            self.assertEqual(ax.patches[0].get_path().vertices.tolist(),vertices)
            self.assertFalse((Path(directory)/'unused.png').exists())
            plt.close(fig)
        self.assertIn('w/o medium-ΔM SR',live_page())
        self.assertIn('setInterval(refresh,15000)',live_page())
        self.assertNotIn('<select',live_page())
        self.assertEqual(live_page().count('<object class="chart"'),3)
        for topology in ('T2tt','T2bW','T2tb'):
            self.assertIn('data-model="'+topology+'"',live_page())

    def test_authenticated_test_freeze_can_be_outside_trotasr(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import load_frozen_test_plot_definition
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'freeze.json'
            content=json.dumps({'sources':{},'status':'frozen_before_test_comparison'}).encode()
            path.write_bytes(content)
            spec=dict(contract_path=str(Path(directory)/'contract.json'),
                      comparison_freeze_path=str(path),
                      comparison_freeze_sha256=hashlib.sha256(content).hexdigest())
            with patch('TROTASR.workflows.build_nominal_grid.load_medium_test_comparison',
                       return_value=spec):
                self.assertEqual(load_frozen_test_plot_definition(spec),json.loads(content))
                path.write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError,'definition changed'):
                    load_frozen_test_plot_definition(spec)
            with patch('TROTASR.workflows.build_nominal_grid.load_medium_test_comparison',
                       return_value={}):
                with self.assertRaisesRegex(ValueError,'input contract changed'):
                    load_frozen_test_plot_definition(spec)

    def test_same_test_comparison_requires_baseline_and_equal_population_weights(self):
        from TROTASR.tests.test_sr_only_comparison import FrozenTestComparisonTests
        from TROTASR.workflows.plotting_limit_contour import require_fixed_test_comparison
        helper=FrozenTestComparisonTests();spec=helper.candidate()
        fixed={'templates/template_'+r+'_'+y+'.root':r+y
               for r in ('highdm','lowdm') for y in ('2024','2025')}
        baseline=dict(variant='baseline',scope='sr_only_nominal_mcstat_comparison',
            parent_grid_sha256='same',treatment='same',auto_mc_stats=[10,1,1],files=fixed,
            points=[dict(model='T2tt',mass='mStop1000_mLSP500')])
        before=dict(baseline,variant='combined',medium_test_comparison=spec,
                    medium_test_provenance=helper.identities())
        after=copy.deepcopy(before);after['medium_test_comparison']['candidate_id']='after'
        baseline_hash=spec['fixed_baseline_manifest_sha256']
        result=require_fixed_test_comparison(after,before,baseline,baseline_hash)
        self.assertEqual(result['medium_test_provenance'],helper.identities())
        for mutate in (lambda g:g['medium_test_provenance']['2024'].update(event_population_sha256='e'*64),
                       lambda g:g['medium_test_provenance']['2025'].update(physical_weights_sha256='e'*64),
                       lambda g:g['medium_test_comparison'].update(comparison_freeze_sha256='e'*64),
                       lambda g:g.update(medium_candidate={'id':'validation'})):
            bad=copy.deepcopy(after);mutate(bad)
            with self.assertRaises(ValueError):require_fixed_test_comparison(bad,before,baseline,baseline_hash)
        with self.assertRaises(ValueError):
            require_fixed_test_comparison(after,before,dict(baseline,variant='high'),baseline_hash)

    def test_same_test_metrics_preserve_worsening_and_full_mass_grid(self):
        from TROTASR.workflows.plotting_limit_contour import test_comparison_metrics
        base=fixture();old=copy.deepcopy(base);new=copy.deepcopy(base)
        mass=next(iter(base['points']))
        old['points'][mass]['expected']=.8;new['points'][mass]['expected']=1.1
        result=test_comparison_metrics(base,old,new)
        self.assertAlmostEqual(result['points'][0]['before_reduction_percent'],20.)
        self.assertAlmostEqual(result['points'][0]['after_reduction_percent'],-10.)
        self.assertAlmostEqual(result['points'][0]['after_over_before'],1.375)
        self.assertFalse(result['reselection_allowed'])
        with self.assertRaises(ValueError):test_comparison_metrics(base,old,dict(new,points={}))

    def test_test_display_medians_and_point_map_preserve_baseline(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.collections import LineCollection
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import contours_only_display,contour_artist_signature
        for metric in (None,dict(mstop=[800],mlsp=[100],percent=[-10.],bound=20.,label='fixture')):
            diagnostics={}
            with patch.object(Figure,'savefig'),contours_only_display(diagnostics,
                    standalone_medians=False,all_medians=True,point_metric=metric):
                fig,ax=plt.subplots()
                fill=ax.contourf([0,1],[0,1],[[0,1],[1,2]])
                fig.colorbar(fill,ax=ax)
                for color in ('red','blue'):
                    for style in ('-','--'):
                        ax.add_collection(LineCollection([[[600,100],[1000,500]]],colors=color,linestyles=style))
                ax.add_patch(PathPatch(Path([[600,100],[1000,500]]),facecolor='none',edgecolor='#009E73'))
                before=contour_artist_signature(ax)
                fig.savefig('not_written.png')
                self.assertEqual(contour_artist_signature(ax),[r for r in before if not r['dashed']])
                self.assertTrue(any(r['variant']=='medium' for r in diagnostics['retained_paths']))
                self.assertEqual(diagnostics['removed_paths'],2)
                self.assertEqual(len(fig.axes),2 if metric else 1)
                plt.close(fig)

    def test_before_after_requires_fixed_high_low_and_matching_mass_grid(self):
        from TROTASR.workflows.plotting_limit_contour import require_fixed_baseline_comparison
        fixed={'templates/template_'+r+'_'+y+'.root':r+y
               for r in ('highdm','lowdm') for y in ('2024','2025')}
        before=dict(variant='combined',scope='sr_only_nominal_mcstat_comparison',
            parent_grid_sha256='same',treatment='same',auto_mc_stats='10 1 1',
            files=dict(fixed,medium='old'),points=[dict(model='T2tt',mass='mStop1000_mLSP500')])
        after=copy.deepcopy(before)
        after['files']['medium']='new'
        after['medium_candidate']={'id':'new'}
        self.assertEqual(require_fixed_baseline_comparison(after,before),fixed)
        for change in (lambda g:g.update(variant='medium'),lambda g:g.update(treatment='different'),
                       lambda g:g['files'].update({next(iter(fixed)):'changed'}),
                       lambda g:g['files'].pop(next(iter(fixed))),lambda g:g.update(points=[])):
            bad=copy.deepcopy(after);change(bad)
            with self.assertRaises(ValueError):require_fixed_baseline_comparison(bad,before)

    def test_before_after_display_retains_both_one_sigma_bands(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.collections import LineCollection
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import contours_only_display,contour_artist_signature
        diagnostics={}
        with patch.object(Figure,'savefig'):
            with contours_only_display(diagnostics,standalone_medians=False):
                fig,ax=plt.subplots()
                fill=ax.contourf([0,1],[0,1],[[0,1],[1,2]])
                fig.colorbar(fill,ax=ax)
                for color in ('red','blue'):
                    for style in ('-','--'):
                        ax.add_collection(LineCollection([[[600,100],[1000,500]]],colors=color,linestyles=style))
                before=contour_artist_signature(ax)
                fig.savefig('not_written.png')
                self.assertEqual(contour_artist_signature(ax),before)
                self.assertEqual(diagnostics['removed_paths'],0)
                self.assertEqual(len(fig.axes),1)
                plt.close(fig)

    def test_medium_median_export_retains_failed_band_accounting(self):
        from TROTASR.workflows import plotting_limit_contour as c
        points=[dict(model='T2tb',mass=m) for m in ('mStop1600_mLSP1450','mStop1500_mLSP1300')]
        keys=['T2tb/'+p['mass'] for p in points]
        q={.025:1.,.16:2.,.5:3.,.84:4.,.975:5.}
        results={keys[0]:dict(status='failed'),keys[1]:dict(status='complete',validation=dict(quantiles=q))}
        evidence={keys[0]:dict(contour_only=dict(quantiles={.5:3.},full_five_quantile_passed=False)),keys[1]:{}}
        grid=dict(points=points,scope='sr_only_nominal_mcstat_comparison')
        payload=c.collect_contour_quantiles(grid,results,evidence,'T2tb',median_only=True)
        self.assertEqual(payload['quantiles_provided'],[.5])
        self.assertEqual(payload['status'],'complete')
        self.assertEqual(payload['full_five_quantile_missing'],[points[0]['mass']])
        self.assertEqual(results[keys[0]]['status'],'failed')
        self.assertTrue(all(set(r)=={'mStop','mLSP','expected'} for r in payload['points'].values()))
        c.validate_collected(payload,[p['mass'] for p in points])
        for change in (lambda x:x.update(scope='full_combined_systematic_production'),
                       lambda x:x['points'][points[0]['mass']].update(expected_p1=4.)):
            bad=copy.deepcopy(payload);change(bad)
            with self.assertRaises(ValueError):c.validate_collected(bad,[p['mass'] for p in points])

    def test_medium_median_requires_actual_validated_crossing(self):
        from TROTASR.workflows.plotting_limit_contour import contour_quantiles
        validation=dict(quantiles={.16:2.,.5:3.,.84:4.},
            crossings=[dict(quantile=q,passed=q==.5) for q in (.16,.5,.84)])
        self.assertIsNone(contour_quantiles(validation))
        self.assertEqual(contour_quantiles(validation,(.5,)),{.5:3.})
        validation['crossings'][1]['passed']=False
        self.assertIsNone(contour_quantiles(validation,(.5,)))
        validation['crossings'][1]['passed']=True
        validation['quantiles'][.5]=float('nan')
        self.assertIsNone(contour_quantiles(validation,(.5,)))

    def ranking_fixture(self):
        from TROTASR.workflows.combine_limits import comparison_candidate_score
        points = [dict(model=m,mass='mStop1000_mLSP500') for m in ('T2tt','T2bW','T2tb')]
        baseline = {p['model']+'/'+p['mass']:2. for p in points}
        rows = []
        for name,values in [('a',[1.,2.,3.]),('b',[2.,3.,4.])]:
            candidate = dict(candidate_id=name,ranking_policy='policy',fixed_baseline_manifest_sha256='baseline')
            score = comparison_candidate_score(points,baseline,
                {p['model']+'/'+p['mass']:v for p,v in zip(points,values)})
            rows.append(dict(score,candidate_id=name,pending={},validation_candidate=candidate))
        grid = dict(variant='combined',points=points,medium_candidate=rows[0]['validation_candidate'])
        report = dict(status='complete',baseline_pending={},observed_sr_used=False,winner='a',
            policy='policy',baseline_manifest_sha256='baseline',candidates=rows,
            ranking_of_complete_candidates=['a','b'],claim='bounded validation pool')
        return grid,report

    def test_ranking_summary_preserves_every_mass_and_regression(self):
        from TROTASR.workflows.plotting_limit_contour import summarize_ranking
        grid,report = self.ranking_fixture()
        summary = summarize_ranking(report,grid)
        self.assertEqual(summary['points'],report['candidates'][0]['points'])
        self.assertEqual(summary['models']['T2tt']['improved'],1)
        self.assertEqual(summary['models']['T2bW']['equal'],1)
        self.assertEqual(summary['models']['T2tb']['worsened'],1)
        self.assertEqual(len(summary['candidate_ranking']),2)

    def test_ranking_summary_refuses_wrong_winner_partial_or_changed_values(self):
        from TROTASR.workflows.plotting_limit_contour import summarize_ranking
        grid,original = self.ranking_fixture()
        changes = [lambda r:r.update(status='provisional'),lambda r:r.update(winner='b'),
            lambda r:r['candidates'][1].update(verified=2),
            lambda r:r['candidates'][0].update(mean_log_ratio=0),
            lambda r:r.update(ranking_of_complete_candidates=['b','a']),
            lambda r:r['candidates'][0]['validation_candidate'].update(extra='changed')]
        for change in changes:
            report=copy.deepcopy(original);change(report)
            with self.assertRaises(ValueError):summarize_ranking(report,grid)

    def test_validation_candidate_bins_preserve_fixed_high_low_and_old_layout(self):
        from TROTASR.tests.test_sr_only_comparison import ValidationCandidateTests
        for categories in (1,4,7,12):
            candidate = ValidationCandidateTests().candidate(categories)
            for variant,high,low in [('medium',0,0),('combined',118,30)]:
                counts = dict(highdm=high,lowdm=low,mediumdm=5*categories)
                grid = dict(scope='sr_only_nominal_mcstat_comparison',variant=variant,
                    medium_candidate=candidate,sr_bins_per_year=counts,bins=2*sum(counts.values()))
                self.assertEqual(sr_bin_counts(grid),(high,low))
                grid['bins'] += 2
                with self.assertRaises(ValueError):sr_bin_counts(grid)
        with self.assertRaises(ValueError):
            sr_bin_counts(dict(scope='sr_only_nominal_mcstat_comparison',variant='high',
                               medium_candidate=candidate))
        for variant,high,low in [('medium',0,0),('combined',118,30)]:
            counts=dict(highdm=high,lowdm=low,mediumdm=20)
            self.assertEqual(sr_bin_counts(dict(scope='sr_only_nominal_mcstat_comparison',
                variant=variant,sr_bins_per_year=counts,bins=2*sum(counts.values()))),(high,low))

    def test_medium_and_combined_overlay_must_use_identical_candidate(self):
        from TROTASR.workflows.plotting_limit_contour import require_same_medium_candidate
        require_same_medium_candidate([dict(variant='combined'),dict(variant='medium')])
        require_same_medium_candidate([dict(variant='combined',medium_candidate={'id':'same'}),
            dict(variant='medium',medium_candidate={'id':'same'}),dict(variant='high')])
        for other in (None,{'id':'different'}):
            with self.assertRaises(ValueError):
                require_same_medium_candidate([dict(variant='combined',medium_candidate={'id':'new'}),
                    dict(variant='medium',medium_candidate=other)])

    def test_contour_only_display_preserves_physical_paths_and_restores_defaults(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import contours_only_display,contour_artist_signature
        original=(Axes.contourf,Figure.colorbar,Figure.savefig)
        diagnostics={}
        with patch.object(Figure,'savefig') as save:
            with contours_only_display(diagnostics):
                fig,ax=plt.subplots()
                filled=ax.contourf([0,1],[0,1],[[0,1],[1,2]])
                fig.colorbar(filled,ax=ax)
                xy=np.array([[600.,100.],[1000.,500.],[1400.,300.]])
                for color in ('red','blue'):
                    for style in ('-','--'):
                        ax.add_collection(LineCollection([xy],colors=color,linestyles=style))
                for color in ('#009E73','#9467BD'):
                    for style in ('-','--'):
                        ax.add_patch(PathPatch(Path(xy),fill=False,edgecolor=color,linestyle=style))
                run2,=ax.plot(xy[:,0],xy[:,1],color='black',ls='--')
                before=contour_artist_signature(ax)
                fig.savefig('not_written.png');fig.savefig('not_written.pdf')
                self.assertEqual(save.call_count,2)
                self.assertEqual(len(fig.axes),1)
                self.assertEqual(contour_artist_signature(ax),
                    [r for r in before if r['variant']=='combined' or not r['dashed']])
                self.assertTrue(np.array_equal(run2.get_xydata(),xy))
                self.assertTrue(diagnostics['retained_coordinates_unchanged'])
                self.assertEqual(diagnostics['removed_paths'],3)
                self.assertEqual(len(diagnostics['retained_paths']),5)
                plt.close(fig)
        self.assertEqual((Axes.contourf,Figure.colorbar,Figure.savefig),original)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with contours_only_display({}):raise RuntimeError('fixture')
        self.assertEqual((Axes.contourf,Figure.colorbar,Figure.savefig),original)

    def test_snapshot_input_uses_compact_exports_without_fit_reader(self):
        from unittest.mock import patch
        from TROTASR.workflows import plotting_limit_contour as c
        grid=dict(scope='sr_only_nominal_mcstat_comparison',variant='low',bins=60,
            sr_bins_per_year=dict(highdm=0,lowdm=30,mediumdm=0),
            points=[dict(model='T2bW',mass='mStop1650_mLSP1350')])
        payload=dict(status='complete',scope=grid['scope'],observed_sr_used=False,
            requested_point_count=1,collected_point_count=1,missing_points=[],quantiles_provided=[.16,.5,.84],
            points={'mStop1650_mLSP1350':dict(mStop=1650,mLSP=1350,expected_m1=97.,expected=148.,expected_p1=256.)})
        snapshot=dict(path=ROOT/'plots/test',manifest=dict(contract=dict(renderer_sha256='hash',
            config_sha256='hash',third_overlay_grid_sha256='hash',third_overlay_collected_sha256={'T2bW':'evidence'}),
            plots=[dict(model='T2bW',variant_coverage={'low':dict(full_five_quantile_status='partial',
                full_five_quantile_missing=['mStop1650_mLSP1350'],contour_points=1)})]))
        with patch.object(c,'MODELS',['T2bW']),patch.object(c,'sha256',return_value='hash'), \
                patch.object(c,'read_json',side_effect=[grid,payload]), \
                patch.object(c,'verified_current_results',side_effect=AssertionError('fit read')):
            _,records,_=c.prepare(ROOT/'limits',ROOT/'grid.json',snapshot=snapshot)
        self.assertEqual(records['T2bW']['payload']['points'],payload['points'])
        self.assertEqual(records['T2bW']['full_grid_status'],'partial')
        self.assertEqual(records['T2bW']['full_contour_point_count'],1)

    def test_improvement_ratio_is_baseline_over_combined_on_matched_points(self):
        from TROTASR.workflows.plotting_limit_contour import expected_improvement
        baseline = dict(points={'a':{'expected':2.},'b':{'expected':3.},'c':{'expected':8.}})
        combined = dict(points={'a':{'expected':1.},'b':{'expected':2.}})
        result = expected_improvement(baseline,combined)
        self.assertEqual(result['matched_points'],2)
        self.assertEqual(result['baseline_points'],3)
        self.assertEqual(result['combined_points'],2)
        self.assertEqual(result['median'],1.75)
        self.assertEqual(result['fraction_improved'],1.)

    def test_extra_contour_uses_final_original_paths_without_saving(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure
        import numpy as np
        from TROTASR.workflows.plotting_limit_contour import capture_expected_paths, additional_expected_paths
        figure, axes = plt.subplots()
        vertices = np.array([[600.,1.],[700.,100.],[800.,100.]])
        red = LineCollection([vertices.copy()],colors='red',linewidths=3.)
        axes.add_collection(red)
        axes.add_collection(LineCollection([vertices+10],colors='blue',linewidths=3.))
        original_save = Figure.savefig
        with capture_expected_paths() as paths:
            figure.savefig('must_not_be_written.png')
            figure.savefig('must_not_be_written.pdf')
        self.assertIs(Figure.savefig,original_save)
        self.assertEqual(len(paths),1)
        self.assertTrue(np.array_equal(paths[0]['path'].vertices,vertices))
        red.get_paths()[0].vertices[:] = 0.
        self.assertTrue(np.array_equal(paths[0]['path'].vertices,vertices))
        from matplotlib.lines import Line2D
        from matplotlib.axes import Axes
        original_legend = Axes.legend
        with additional_expected_paths(paths,'Medium SR'):
            legend = axes.legend(handles=[Line2D([0],[0],label='Baseline'),
                Line2D([0],[0],label='Combined'),Line2D([0],[0],label='Run 2')])
        self.assertIs(Axes.legend,original_legend)
        self.assertEqual([x.get_text() for x in legend.get_texts()],['Baseline','Combined','Medium SR','Run 2'])
        self.assertTrue(np.array_equal(axes.patches[0].get_path().vertices,vertices))
        plt.close(figure)
        figure,axes=plt.subplots()
        with additional_expected_paths(paths,'Medium SR',extra_overlays=[(paths,'Low SR','#9467BD')]):
            legend=axes.legend(handles=[Line2D([0],[0],label='Combined'),
                Line2D([0],[0],label='High SR'),Line2D([0],[0],label='Run 2')])
        self.assertEqual([x.get_text() for x in legend.get_texts()],
                         ['Combined','High SR','Medium SR','Low SR','Run 2'])
        self.assertEqual(len(axes.patches),2)
        self.assertTrue(all(np.array_equal(p.get_path().vertices,vertices) for p in axes.patches))
        from matplotlib.colors import to_rgba
        self.assertEqual(axes.patches[1].get_edgecolor(),to_rgba('#9467BD'))
        plt.close(figure)

    def test_fourth_contour_requires_explicit_complete_overlay_pairs(self):
        from unittest.mock import patch
        from TROTASR.workflows.plotting_limit_contour import render
        grid=dict(scope='sr_only_nominal_mcstat_comparison',variant='combined',bins=336,
                  sr_bins_per_year=dict(highdm=118,lowdm=30,mediumdm=20))
        with patch('TROTASR.workflows.plotting_limit_contour.prepare',return_value=(grid,{},'hash')):
            with self.assertRaisesRegex(ValueError,'fourth contour'):
                render(ROOT/'l',ROOT/'g',ROOT/'tests/not_written',third_overlay_limits=ROOT/'low')

    def test_contour_capture_restores_renderer_after_error(self):
        from matplotlib.figure import Figure
        from TROTASR.workflows.plotting_limit_contour import capture_expected_paths
        original = Figure.savefig
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with capture_expected_paths(): raise RuntimeError('fixture')
        self.assertIs(Figure.savefig,original)

    def test_active_recovery_only_readable_for_explicit_partial_snapshot(self):
        from TROTASR.workflows.plotting_limit_contour import recovery_readable
        self.assertFalse(recovery_readable(dict(status='running',pid=123),False))
        self.assertTrue(recovery_readable(dict(status='running',pid=123),True))
        self.assertFalse(recovery_readable(dict(status='running',pid=None),True))
        self.assertTrue(recovery_readable(dict(status='needs_attention',pid=None),False))
        self.assertFalse(recovery_readable(dict(status='unknown',pid=None),True))

    def test_live_snapshot_never_reads_running_or_missing_point_quantiles(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from TROTASR.workflows.plotting_limit_contour import read_plot_point
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            path=Path(folder)/'state.json'
            self.assertEqual(read_plot_point(path,True),({'status':'pending'},None))
            with self.assertRaises(FileNotFoundError):read_plot_point(path)
            raw=json.dumps(dict(status='running',pid=123,validation={'not':'usable'})).encode()
            path.write_bytes(raw)
            state,digest=read_plot_point(path,True)
            self.assertEqual(state,{'status':'running'})
            self.assertEqual(digest,hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError,'not terminal'):read_plot_point(path)
            for status in ('complete','failed'):
                record=dict(status=status,pid=None,attempts=[])
                path.write_text(json.dumps(record))
                self.assertEqual(read_plot_point(path,True)[0],record)

    def test_live_snapshot_rejects_implicit_partial_and_overlay_modes(self):
        from TROTASR.workflows.plotting_limit_contour import render
        from unittest.mock import patch
        with patch('TROTASR.workflows.plotting_limit_contour.prepare') as prepare:
            for kwargs in ({}, {'allow_partial':True,'overlay_grid':ROOT/'other'},
                           {'allow_partial':True,'contours_only':True}):
                with self.assertRaisesRegex(ValueError,'standalone partial'):
                    render(ROOT/'limits',ROOT/'grid',ROOT/'unused',live_snapshot=True,**kwargs)
            prepare.assert_not_called()

    def test_progress_annotations_stay_out_of_contour_figures(self):
        import ast, inspect
        from TROTASR.workflows import plotting_limit_contour as contour
        self.assertFalse(hasattr(contour, 'live_snapshot_caption'))
        tree=ast.parse(inspect.getsource(contour.render))
        calls=[node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr=='plot_contour']
        self.assertTrue(calls)
        for call in calls:
            for keyword in call.keywords:
                if keyword.arg=='analysis_label':
                    self.assertIsNone(ast.literal_eval(keyword.value))
        self.assertNotIn('IN PROGRESS', inspect.getsource(contour.render))

    def test_mass_support_mask_never_fills_a_missing_vertex(self):
        import numpy as np
        from TROTASR.workflows.plotting_limit_contour import support_mask
        masses=['mStop100_mLSP1','mStop100_mLSP99','mStop200_mLSP1',
                'mStop200_mLSP99','mStop150_mLSP50']
        for missing in (masses[0],masses[-1]):
            mask,stats=support_mask(masses,set(masses)-{missing})
            from TROTASR.workflows.signal_grid_inputs import mass_pair
            x,y=mass_pair(missing)
            self.assertFalse(mask(np.array([x]),np.array([y]))[0])
            self.assertEqual(stats['verified'],4)
        mask,_=support_mask(masses,set(masses))
        self.assertFalse(mask(np.array([110.]),np.array([120.]))[0])

    def test_merged_identity_and_unchanged_original(self):
        self.assertEqual(sr_bin_counts(dict(bins=540)),(162,30))
        reference=dict(path='sr_category_merge.json',sha256=sha256(ROOT/'sr_category_merge.json'))
        self.assertEqual(sr_bin_counts(dict(bins=452,sr_merge=reference)),(118,30))
        with self.assertRaises(ValueError):sr_bin_counts(dict(bins=458,sr_merge=reference))
        with self.assertRaises(ValueError):sr_bin_counts(dict(bins=540,sr_merge=reference))
        with self.assertRaises(ValueError):sr_bin_counts(dict(bins=452))

    def test_complete_accounting(self):
        validate_collected(fixture(), ['mStop1200_mLSP500'])

    def test_a_dropped_failed_point_is_not_complete(self):
        with self.assertRaisesRegex(ValueError, 'accounting'):
            validate_collected(fixture(), ['mStop1200_mLSP500', 'mStop1300_mLSP500'])

    def test_partial_is_explicit_and_retains_missing_points(self):
        payload = fixture()
        payload.update(status='partial', requested_point_count=2, missing_points=['mStop1300_mLSP500'])
        before = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, 'Incomplete limits'):
            validate_collected(payload, ['mStop1200_mLSP500', 'mStop1300_mLSP500'])
        validate_collected(payload, ['mStop1200_mLSP500', 'mStop1300_mLSP500'], allow_partial=True)
        self.assertEqual(payload, before)

    def test_observed_or_validation_input_rejected(self):
        for key, value in (('observed_sr_used', True), ('scope', 'representative')):
            payload = fixture(); payload[key] = value
            with self.assertRaises(ValueError):
                validate_collected(payload, list(payload['points']))

    def test_nonordered_quantiles_rejected(self):
        payload = fixture(); payload['points']['mStop1200_mLSP500']['expected_p2'] = .2
        with self.assertRaisesRegex(ValueError, 'quantiles'):
            validate_collected(payload, list(payload['points']))

    def test_internal_reference_hashes_and_original_renderer(self):
        config = read_json(ROOT / 'stats/contour_config.json')
        for spec in list(config['references'].values()) + [config['theory']]:
            self.assertEqual(sha256(ROOT / spec['path']), spec['sha256'])
            self.assertTrue(spec['physics_values_unchanged'])
        self.assertEqual(sha256(ROOT / 'utils/renderers/postprocess_limits.py'),
            '446dece4eddd330313902260edf1c3ec362e1d531e648ac9867e45ae735e0bd3')


class Run3NumericalAdapterScopeTests(unittest.TestCase):
    def fixture(self):
        return (dict(scope='full_combined_systematic_production'),
            ROOT/'stats/run3_systematics_20261025/limits', 'numerical_minima_20261009',
            dict(scope='run3_displayed_finite_continuation',stage='numerical_minima_20261009',
                plot_range_only=True,fit_selection=dict(plot_range_only=True,source_total=955,total=895,
                    expected_points_by_model={'T2tt':205,'T2bW':345,'T2tb':345})),
            dict(run3_extended_numerics=True))

    def test_exact_source_bound_stage_optin_is_read_only(self):
        from TROTASR.workflows.plotting_limit_contour import run3_numerical_adapter_scope
        args=self.fixture();before=copy.deepcopy(args)
        self.assertTrue(run3_numerical_adapter_scope(*args))
        self.assertEqual(args,before)

    def test_legacy_minima_do_not_opt_in(self):
        from TROTASR.workflows.plotting_limit_contour import run3_numerical_adapter_scope
        args=list(self.fixture())
        for record in ({},dict(run3_extended_numerics=False)):
            args[-1]=record
            self.assertFalse(run3_numerical_adapter_scope(*args))

    def test_other_scope_or_loose_boolean_cannot_use_adapter(self):
        from TROTASR.workflows.plotting_limit_contour import run3_numerical_adapter_scope
        mutations=(lambda a:a[0].update(scope='full_run3_hybrid_production'),
            lambda a:a.__setitem__(1,ROOT/'stats/run3_hybrid_20261022/limits'),
            lambda a:a.__setitem__(2,'numerical_remaining_20261009'),
            lambda a:a[3].update(scope='other'),lambda a:a[3].update(stage='other'),
            lambda a:a[3].update(plot_range_only=False),
            lambda a:a[3]['fit_selection'].update(total=955),
            lambda a:a[3]['fit_selection'].update(source_total=895),
            lambda a:a[3]['fit_selection']['expected_points_by_model'].update(T2tt=225),
            lambda a:a[-1].update(run3_extended_numerics=1))
        for mutate in mutations:
            args=list(copy.deepcopy(self.fixture()));mutate(args)
            with self.assertRaises(ValueError):run3_numerical_adapter_scope(*args)


if __name__ == '__main__':
    unittest.main()
