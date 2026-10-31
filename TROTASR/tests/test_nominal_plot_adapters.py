"""Full-layout synthetic adapter checks; not production plots or physics fits."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.plotting_nominal_distributions import (
    physical_boundary, checked_leaf, dy_display_config, make_records,
    nominal_renderer, validate_histograms, renderer,
    HIGH_VARIABLES, LOW_VARIABLES, SCHEMES,
    HIGHDM_DISTRIBUTION_VARIABLE_SPECS, LOWDM_VARIABLE_SPECS, omitted_physical_cells)
from TROTASR.workflows.plotting_transfer_factors import prepare_factors
from TROTASR.workflows.plotting_background_measurements import validate_products, RECOIL_EDGES


def physical_fixture():
    p = dict(status='complete', scope='synthetic_unit_test', sr_data_blinded=True,
        contract=dict(year=2024, sr_binning_sha256=sha256(ROOT/'highdm_sr_binning.json'),
                      configuration_sha256=sha256(ROOT/'gnn4lowdm/config.json')),
        physical_histograms={})
    for mode,specs in (('highdm',HIGHDM_DISTRIBUTION_VARIABLE_SPECS),('lowdm',LOWDM_VARIABLE_SPECS)):
        source = p['physical_histograms'][mode] = {'nominal':{}}
        for region in ('LLCR','QCDCR','GCR','DY2E','DY2M'):
            source['nominal'][region] = {}
            for nb in ('Nb1','Nb2'):
                samples = source['nominal'][region][nb] = {}
                variables = (HIGH_VARIABLES['DYCR' if region.startswith('DY') else region]
                             if mode == 'highdm' else LOW_VARIABLES[region])
                for sample in ('data','DY','ST','GJ','WtoLNu','T2bW_mStop1200_mLSP500'):
                    samples[sample] = {}
                    for v in variables:
                        edges = specs[v]['bins']; n=len(edges)-1
                        samples[sample][v] = dict(edges=edges, sumw=[1.]*n, sumw2=[2.]*n, entries=[3.]*n)
    return p


def rz_fixture(key):
    product = dict(status='complete', **{key:dict(channels={}),key+'_raw':{}})
    for c in ('DY2E','DY2M'):
        product[key]['channels'][c] = {}
        product[key+'_raw'][c] = {}
        for g,value,yield_ in (('Nb1',2.,1.),('Nb2plus',4.,3.)):
            product[key]['channels'][c][g] = dict(status='complete',fit_converged=True,RZ=value)
            product[key+'_raw'][c][g] = dict(on=dict(zll=dict(sumw=yield_)))
    return product


class PhysicalPlotAdapterTests(unittest.TestCase):
    def test_explicit_four_jet_display_only_extends_old_bins_and_restores_map(self):
        from TROTASR.workflows.plotting_nominal_distributions import assert_four_jet_extension
        h=physical_fixture()
        for categories in h['physical_histograms']['highdm']['nominal'].values():
            for samples in categories.values():
                for record in samples.values():
                    for field in ('sumw','sumw2','entries'):record['njet'][field][:3]=[0,0,0]
        boundary=physical_boundary(h);mapping=deepcopy(renderer.HIGHDM_MULTIPLICITY_PLOT_BINS)
        with nominal_renderer(2024):old=[r for r in make_records(boundary,['highdm']) if r[3]=='njet']
        with nominal_renderer(2024,include_four_jet_bin=True):
            new=[r for r in make_records(boundary,['highdm']) if r[3]=='njet']
        self.assertEqual(mapping,renderer.HIGHDM_MULTIPLICITY_PLOT_BINS)
        reference=dict(schema='trotasr_physical_plot_inputs_v1',contract=dict(inputs={},year=2024),records=old)
        assert_four_jet_extension(reference,new,{},2024)
        self.assertTrue(all(r[-1]['xlabels']==['4','5','6','7-8',r'$\geq9$'] for r in new))
        for field in ('data','background','background_stat_unc'):
            changed=deepcopy(new);changed[0][-1][field][2]+=1
            with self.assertRaises(AssertionError):assert_four_jet_extension(reference,changed,{},2024)

    def test_sparse_shapes_keep_migrations_but_weight_only_population_is_rejected(self):
        from TROTASR.workflows.plotting_nominal_distributions import endpoint_container
        leaf=lambda v:dict(edges=[0,1],sumw=[v],sumw2=[v*v],entries=[int(v!=0)])
        rows={'nominal':{'CR':{'Nb1':{'data':{'x':leaf(1)},'TT':{'x':leaf(2)}}}},
              'jesTotalUp':{'CR':{'Nb1':{'ST':{'x':leaf(3)}}}},'jesTotalDown':{},
              'puUp':{},'puDown':{}}
        payload={'physical_histograms':{'highdm':rows}}
        with patch('TROTASR.workflows.plotting_nominal_distributions.combined_shape_endpoints',
                   return_value={'jesTotalUp','jesTotalDown'}):
            result=endpoint_container(payload,'physical_histograms','highdm','CR','Nb1','x',[0,1],['jesTotal','pu'])
            self.assertEqual(result['ST']['nominal']['sumw'],[0.])
            self.assertEqual(result['ST']['jesTotalUp']['sumw'],[3.])
            self.assertEqual(result['TT']['jesTotalDown']['sumw'],[0.])
            self.assertEqual(result['TT']['puUp']['sumw'],[2.])
            self.assertEqual(result['ST']['puDown']['sumw'],[0.])
            self.assertEqual(set(result['data_obs']),{'nominal'})
            rows['puUp']={'CR':{'Nb1':{'ST':{'x':leaf(1)}}}}
            with self.assertRaisesRegex(ValueError,'non-nominal population'):
                endpoint_container(payload,'physical_histograms','highdm','CR','Nb1','x',[0,1],['jesTotal','pu'])

    def test_band_only_guard_covers_data_variance_style_and_labels(self):
        from TROTASR.workflows.plotting_nominal_distributions import assert_band_only
        row=dict(data=[3.],background=[2.],background_stat_unc=[1.],edges=[0,1],
                 label='unchanged',background_unc=[1.])
        other=deepcopy(row);other.update(background_unc=[2.],background_systematic_totals={'x':{}})
        assert_band_only(row,other)
        for key in ('data','background','background_stat_unc','edges','label'):
            changed=deepcopy(other);changed[key]='changed'
            with self.assertRaises(ValueError):assert_band_only(row,changed)

    def test_both_nb_streams_are_added_and_signal_is_not_overlaid(self):
        b = physical_boundary(physical_fixture())
        leaf = b['highdm_variable_histograms']['LLCR']['met']['ST']['nominal']
        self.assertEqual(leaf['sumw'], [2.]*8)
        self.assertEqual(leaf['sumw2'], [4.]*8)
        self.assertEqual(leaf['entries'], [6.]*8)
        self.assertEqual(set(b['highdm_variable_histograms']['LLCR']['met']),
                         {'data_obs','DY','ST','GJ','WtoLNu'})

    def test_all_36_yearly_physical_plots_use_original_record_functions(self):
        with nominal_renderer(2024):
            rows=make_records(physical_boundary(physical_fixture()))
        self.assertEqual(len(rows),36)
        self.assertEqual(sum(r[1]=='highdm' for r in rows),20)
        self.assertTrue(all(r[-1]['reference_style'] for r in rows))
        self.assertTrue(all(r[-1]['unit_area']==(r[2]=='GCR') for r in rows))
        self.assertTrue(all(not r[-1].get('blind_data',False) for r in rows))
        self.assertTrue(all(not r[-1]['signals'] for r in rows))
        dy=next(r[-1] for r in rows if r[0]=='lowdm_cr_dycr_recoil_dy')
        np.testing.assert_array_equal(dy['data'], np.full(len(dy['data']),4.))

    def test_original_multiplicity_display_rebin_is_not_sr_rebinning(self):
        with nominal_renderer(2024):
            rows=make_records(physical_boundary(physical_fixture()),['highdm'],['LLCR'])
        nb=next(r[-1] for r in rows if r[3]=='nb')
        self.assertEqual(nb['edges'],[.5,1.5,2.5,3.5])
        np.testing.assert_array_equal(nb['data'],[2.,2.,6.])
        self.assertEqual(nb['xlabels'],['1','2',r'$\geq3$'])

    def test_missing_data_invalid_edges_and_unknown_sample_rejected(self):
        p=physical_fixture(); del p['physical_histograms']['highdm']['nominal']['LLCR']['Nb2']['data']
        with self.assertRaisesRegex(ValueError,'Missing collision-data'):
            physical_boundary(p)
        with self.assertRaisesRegex(ValueError,'Changed physical histogram edges'):
            checked_leaf(dict(edges=[0,2]),[0,1])
        p=physical_fixture();p['physical_histograms']['highdm']['nominal']['LLCR']['Nb1']['unknown']={}
        with self.assertRaisesRegex(ValueError,'unmapped background'):
            physical_boundary(p)

    def test_current_effective_dy_policy_matches_legacy_exactly(self):
        config=dy_display_config(rz_fixture('rz_high'),rz_fixture('rz_low'))
        self.assertEqual(config['dy_rz']['channels']['DY2E']['highdm']['effective'],3.5)
        b=physical_boundary(physical_fixture())
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as d:
            path=Path(d)/'factors.json';write_json(path,config)
            renderer.apply_dy_rz(b,path)
        dy=b['highdm_variable_histograms']['DY2E']
        np.testing.assert_array_equal(dy['ut']['DY']['nominal']['sumw'],[7.]*8)
        self.assertEqual(dy['nb']['DY']['nominal']['sumw'][1:3],[4.,8.])
        self.assertEqual(dy['nb']['DY']['nominal']['sumw2'][1:3],[16.,64.])
        self.assertEqual(dy['ut']['ST']['nominal']['sumw'],[2.]*8)

    def test_no_invalid_dy_fit_or_zero_support_unity_fallback(self):
        for change in ('status','support'):
            p=rz_fixture('rz_high')
            if change=='status':p['rz_high']['channels']['DY2E']['Nb1']['fit_converged']=False
            else:
                for g in ('Nb1','Nb2plus'):p['rz_high_raw']['DY2E'][g]['on']['zll']['sumw']=0
            with self.assertRaises(ValueError):dy_display_config(p,rz_fixture('rz_low'))

    def test_histogram_scope_blinding_and_contract_enforced(self):
        p=physical_fixture()
        with self.assertRaisesRegex(ValueError,'Full production'):validate_histograms(p)
        self.assertEqual(validate_histograms(p,True),2024)
        p['histograms']={'highdm':{'nominal':{'SR':{'cat':{'data':{}}}}}}
        with self.assertRaisesRegex(ValueError,'SR data'):validate_histograms(p,True)

    def test_renderer_globals_are_restored(self):
        old=renderer.PLOT_SYSTEMATIC_SOURCES
        lumi=renderer.LUMINOSITY_FB
        with nominal_renderer(2025):
            self.assertEqual(renderer.PLOT_SYSTEMATIC_SOURCES,[])
            self.assertEqual(renderer.LUMINOSITY_FB,110.84)
        self.assertIs(renderer.PLOT_SYSTEMATIC_SOURCES,old)
        self.assertEqual(renderer.LUMINOSITY_FB,lumi)

    def test_four_jet_events_are_flagged_not_silently_lost_or_rebinned(self):
        raw={'data_obs':{'nominal':dict(sumw=[0.]*10,sumw2=[0.]*10,entries=[0.]*10)}}
        for field in ('sumw','sumw2','entries'):raw['data_obs']['nominal'][field][3]=52.
        before=deepcopy(renderer.HIGHDM_MULTIPLICITY_PLOT_BINS)
        audit=omitted_physical_cells(raw,'njet')
        self.assertEqual(audit,[dict(sample='data_obs',source_bin_1based=4,interval=[3.5,4.5],
                                    sumw=52.,sumw2=52.,entries=52.)])
        self.assertEqual(renderer.HIGHDM_MULTIPLICITY_PLOT_BINS,before)


class TransferPlotAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/'gnn4lowdm/config.json')
        # Explicit synthetic positive records, with a shared full-score denominator.
        from TROTASR.utils.sr_binning import NATIVE_EDGES
        self.p=dict(schema='trotasr_transfer_factors_v2',mc_stat_only=True,extra_weight_applied=False,
            highdm=dict(edges=NATIVE_EDGES,records={}),
            lowdm=dict(records={},sr_binning=self.config['sr_binning'],cr_binning=self.config['cr_binning']))
        for route in ('top_llcr','w_llcr','qcd_qcdcr','zinv_gcr'):
            self.p['highdm']['records'][route]={g:dict(status='complete',valid=[True]*8,
                transfer_factor=[.5]*8,mcstat=[np.sqrt(.375)]*8,
                process_coverage=dict(numerator=True,denominator=True)) for g in ('Nb1','Nb2plus')}
            self.p['lowdm']['records'][route]={c:dict(status='complete',valid=[True]*5,
                transfer_factor=[.1]*5,mcstat=[np.sqrt(.011)]*5,
                denominator_total=10.,denominator=[10.]*5,cr_score_yield=[2.]*5,
                covariance=(np.eye(5)*.01+np.ones((5,5))*.001).tolist(),
                process_coverage=dict(numerator=True,denominator=True),nb_group=c.split('_')[0],
                score_edges=self.config['sr_binning']['edges_by_category'][c],
                cr_score_edges=self.config['cr_binning']['score_edges']) for c in self.config['sr_binning']['category_labels']}

    def test_all_valid_values_and_covariance_unchanged(self):
        out,missing=prepare_factors(self.p,self.config)
        self.assertEqual(missing,[])
        self.assertEqual(out,self.p)

    def test_missing_cell_is_audited_not_replaced_with_unity(self):
        r=self.p['highdm']['records']['top_llcr']['Nb1']
        r['transfer_factor'][0]=None;r['mcstat'][0]=None;r['valid'][0]=False;r['status']='unsupported_cells'
        out,missing=prepare_factors(self.p,self.config)
        self.assertEqual(missing[0]['bins_1based'],[1])
        self.assertIsNone(out['highdm']['records']['top_llcr']['Nb1']['transfer_factor'][0])

    def test_score_index_denominator_or_covariance_corruption_fails(self):
        for key in ('denominator','covariance'):
            p=deepcopy(self.p); r=p['lowdm']['records']['top_llcr']['Nb1_NISR0']
            if key=='denominator':r[key][0]=2.
            else:r[key][0][0]*=2.
            with self.assertRaisesRegex(ValueError,'pooled integral/covariance'):
                prepare_factors(p,self.config)

    def test_no_extra_z_normalization_or_missing_category(self):
        self.p['extra_weight_applied']=True
        with self.assertRaisesRegex(ValueError,'raw MC'):prepare_factors(self.p,self.config)
        self.p['extra_weight_applied']=False
        del self.p['lowdm']['records']['zinv_gcr']['Nb1_NISR0']
        with self.assertRaisesRegex(ValueError,'TF categories'):prepare_factors(self.p,self.config)



def measurement_fixture():
    products = {key:rz_fixture(key) for key in ('rz_high','rz_low')}
    for key,suffix in (('rz_high','high'),('rz_low','low')):
        p=products[key]
        p[key]['combined']={g:dict(status='complete',RZ=3.,RZ_stat=.2) for g in ('Nb1','Nb2plus')}
        p['mll_'+suffix]={}
        for channel in ('DY2E','DY2M'):
            p['mll_'+suffix][channel]={}
            for group in ('Nb1','Nb2plus'):
                p[key]['channels'][channel][group].update(RZ_stat=.3,RT=1.,RT_stat=.4)
                p['mll_'+suffix][channel][group]={c:dict(edges=[50.,71.,111.,500.],
                    sumw=[1.,2.,3.],sumw2=[1.,2.,3.]) for c in ('data','zll','other')}
    factor=dict(status='complete',value=1.,stat=.1)
    products['sgamma']=dict(status='complete',recoil_edges=RECOIL_EDGES,
        highdm={g:dict(Q=deepcopy(factor),bins=[dict(Sgamma=deepcopy(factor)) for _ in range(5)]) for g in ('Nb1','Nb2plus')},
        lowdm_families={g:dict(Q=deepcopy(factor),bins=[dict(Sgamma=deepcopy(factor)) for _ in range(5)]) for g in ('Nb1','Nb2plus')},
        lowdm_Q_groups={g:deepcopy(factor) for g in ('Nb1','Nb2plus')})
    products['double_ratio']=dict(status='complete',**{mode:dict(bins=[dict(low=RECOIL_EDGES[i],high=RECOIL_EDGES[i+1],
        status='complete',z_data_over_mc=1.,z_stat=.1,photon_data_over_mc=1.,photon_stat=.1,
        double_ratio=1.,double_ratio_stat=.2,systematic=.2) for i in range(5)]) for mode in ('highdm','lowdm')})
    return products


class MeasurementPlotAdapterTests(unittest.TestCase):
    def test_exact_approved_measurement_merge_only(self):
        p = measurement_fixture()
        ratio = p['double_ratio']
        from TROTASR.workflows.plotting_background_measurements import ROOT as package_root
        ratio['measurement_bin_policy'] = read_json(package_root/'jsons/zgamma_2023BPix_20261022_policy.json')
        ratio['provenance'] = dict(campaign_year='2023BPix')
        ratio['measurement_merge_audit'] = dict(valid_nonmerged_rows_preserved=True, lowdm_unchanged=True)
        ratio['highdm']['bins'][1]['high'] = 400.
        del ratio['highdm']['bins'][2]
        before = deepcopy(p)
        self.assertEqual(validate_products(p), [])
        self.assertEqual(p, before)
        ratio['provenance']['campaign_year'] = '2024'
        with self.assertRaisesRegex(ValueError, 'measurement-only merge'): validate_products(p)

    def test_complete_data_not_refitted_or_mutated(self):
        p=measurement_fixture();before=deepcopy(p)
        self.assertEqual(validate_products(p),[])
        self.assertEqual(p,before)

    def test_actual_boundary_fit_kept_and_reported(self):
        p=measurement_fixture();p['rz_low']['rz_low']['channels']['DY2M']['Nb2plus']['RT']=0.
        out=validate_products(p)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['RT'],0.)

    def test_missing_mll_component_blocks_plot_instead_of_zero_fill(self):
        p=measurement_fixture();del p['rz_high']['mll_high']['DY2E']['Nb1']['other']
        with self.assertRaisesRegex(ValueError,'mll component'):validate_products(p)

    def test_unconverged_fit_and_negative_uncertainty_fail(self):
        p=measurement_fixture();p['rz_high']['rz_high']['channels']['DY2E']['Nb1']['fit_converged']=False
        with self.assertRaisesRegex(ValueError,'did not converge'):validate_products(p)
        p=measurement_fixture();p['sgamma']['highdm']['Nb1']['Q']['stat']=-1.
        with self.assertRaisesRegex(ValueError,'measurement number'):validate_products(p)

    def test_no_implicit_photon_rebinning(self):
        p=measurement_fixture();p['sgamma']['recoil_edges']=[250,350,400,500,800,1500]
        with self.assertRaisesRegex(ValueError,'Sgamma recoil bins'):validate_products(p)
        p=measurement_fixture();p['double_ratio']['highdm']['bins'][1]['low']=351
        with self.assertRaisesRegex(ValueError,'discontinuous'):validate_products(p)


if __name__=='__main__':unittest.main()
