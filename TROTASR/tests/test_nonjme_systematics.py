import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from TROTASR.utils.corrections import NONJME_ENDPOINTS, nonjme_event_views, _valid_scaled_pt
from TROTASR.workflows.nominal_campaign import current_contract


class NonJMETests(unittest.TestCase):
    def test_weight_endpoints_only_when_requested_on_central_objects(self):
        from TROTASR.utils.histogramming import weight_variations_needed
        self.assertTrue(weight_variations_needed('nominal', ('all_weights',)))
        self.assertTrue(weight_variations_needed('nominal', ('pileupUp',)))
        self.assertFalse(weight_variations_needed('nominal', ('nominal',)))
        self.assertTrue(weight_variations_needed('nominal', ('nominal',), reference_execution=True))
        for endpoint in ('jesTotalUp', 'jerDown', *NONJME_ENDPOINTS):
            for requested in (('nominal',), ('all_weights',)):
                self.assertFalse(weight_variations_needed(endpoint, requested))
                self.assertFalse(weight_variations_needed(endpoint, requested, reference_execution=True))

    def test_central_muon_delta_reuses_trota_and_is_idempotent(self):
        from TROTASR.utils.corrections import corrected_central_chunks
        from TROTASR.utils import corrections as c
        original = ak.Array(dict(is_data=[False,False],muon_corrected_pt=[[199.],[40.]],
            muon_corrected_mass=[[.1],[.1]],muon_pt_all=[[199.],[40.]],
            muon_mass_all=[[.1],[.1]],muon_phi_all=[[0.],[1.]],
            met=[300.,350.],met_phi=[0.,0.],nTopMixed1pct=[2,3],
            puppi_met_corrected=[300.,350.],puppi_met_corrected_phi=[0.,0.]))
        def calibration(*args, **kwargs):
            return (ak.Array([[199.],[40.]]) if kwargs.get('preserve_scaled_fallback') is False
                    else ak.Array([[201.],[40.]]),ak.Array([[.1],[.1]]),{})
        with patch.object(c,'nonjme_calibration_inputs',return_value={}), \
             patch.object(c,'_nonjme_cset',return_value={}), \
             patch.object(c,'_calibrate_muons',side_effect=calibration), \
             patch('TROTASR.utils.shape_kinematics.rebuild',side_effect=lambda a,**k:ak.Array(a)):
            audit={}
            after=list(corrected_central_chunks([original],2024,audit))[0]
            self.assertEqual(ak.to_list(after.muon_corrected_pt),[[201.],[40.]])
            self.assertEqual(ak.to_list(after.met),[298.,350.])
            self.assertEqual(ak.to_list(after.nTopMixed1pct),[2,3])
            self.assertEqual(audit['updated_muons'],1)
            again=list(corrected_central_chunks([after],2024,{}))[0]
            self.assertEqual(ak.to_list(after),ak.to_list(again))
            bad=ak.with_field(original,ak.Array([[150.],[40.]]),'muon_corrected_pt')
            with self.assertRaisesRegex(ValueError,'neither retained nor updated'):
                list(corrected_central_chunks([bad],2024,{}))

    def test_central_muon_update_leaves_data_untouched(self):
        from TROTASR.utils.corrections import corrected_central_chunks
        data=ak.Array(dict(is_data=[True],met=[300.]))
        self.assertIs(list(corrected_central_chunks([data],2025,{}))[0],data)

    def test_medium_miniiso_matches_all_six_official_eras(self):
        from TROTASR.utils import pog_corrections as pog
        from TROTASR.utils.paths import ROOT, payload_path
        from TROTASR.utils.weight_components import campaign_year
        import correctionlib
        eta, pt = ak.Array([[-.5, .5], []]), ak.Array([[15., 50.], []])
        for era in ('2022', '2022EE', '2023', '2023BPix', '2024', '2025'):
            with self.subTest(era=era):
                year = campaign_year(era)
                path = payload_path('data/MuonSF/'+year+'/muon_Z.json.gz')
                self.assertEqual(path, ROOT/'scales/MUO'/era/'muon_Z.json.gz')
                official = correctionlib.CorrectionSet.from_file(str(path))
                for function, key in ((pog.get_mu_medium_miniiso_sf, 'NUM_MediumMiniIso_DEN_MediumID'),
                                      (pog.get_mu_loose_miniiso_sf, 'NUM_LooseMiniIso_DEN_LooseID')):
                    for actual, variation in zip(function(year, eta, pt), ('nominal', 'systup', 'systdown')):
                        expected = official[key].evaluate(np.array([-.5,.5]), np.array([15.,50.]), variation)
                        np.testing.assert_allclose(ak.to_numpy(ak.flatten(actual)), expected, rtol=1e-12)
                        self.assertEqual(ak.to_list(ak.num(actual)), [2,0])
        with self.assertRaises(ValueError):
            campaign_year('unknown')

    def test_btag_physical_pt_preserves_official_extrapolation_all_eras(self):
        from TROTASR.utils.pog_corrections import BTagCorrector, _load_pog
        from TROTASR.utils.paths import ROOT
        from types import SimpleNamespace
        for era in ('2022', '2022EE', '2023', '2023BPix', '2024', '2025'):
            with self.subTest(era=era):
                tag = 'UParTAK4' if era in ('2024','2025') else 'particleNet'
                official = _load_pog(str(ROOT/'scales/BTV'/era/'btagging.json.gz'))
                obj = BTagCorrector.__new__(BTagCorrector)
                obj._wp = 'M'
                obj.sf = {'comb':official[tag+'_comb'], 'ligh':official[tag+'_light']}
                seen = []
                def efficiency(flavor, pt, eta):
                    seen.append(np.asarray(pt)); return np.full(len(pt), .2)
                obj.eff = SimpleNamespace(evaluate=efficiency)
                pt=ak.Array([[500.],[1500.],[]]); eta=ak.Array([[.5],[.5],[]])
                weights=obj.btag_weight(pt, eta, ak.Array([[5],[5],[]]), ak.Array([[True],[True],[]]))
                for index, variation in ((0,'central'), (1,'up_correlated'), (2,'down_correlated')):
                    np.testing.assert_allclose(np.asarray(weights[index])[:2],
                        official[tag+'_comb'].evaluate(variation,'M',5,np.array([.5,.5]),np.array([500.,1500.])), rtol=1e-12)
                    self.assertEqual(float(weights[index][2]),1.)
                np.testing.assert_array_equal(seen[0],[500.,999.99])
                with self.assertRaises(ValueError):
                    BTagCorrector('particleNet' if tag=='UParTAK4' else 'UParTAK4',era,'medium','unused')

    def test_muon_resolution_fallback_keeps_scaled_pt_by_default(self):
        from TROTASR.utils import corrections as numerical
        from TROTASR.utils.paths import ROOT
        a=dict(Muon_pt=ak.Array([[35.,199.]]), Muon_mass=ak.Array([[.105,.105]]),
            Muon_eta=ak.Array([[.5,.5]]),Muon_phi=ak.Array([[.1,.1]]),
            Muon_charge=ak.Array([[1,1]]),Muon_nTrackerLayers=ak.Array([[5,5]]),
            event=np.array([1]),luminosityBlock=np.array([2]))
        names=['a_mc','m_mc','poly_params','cb_params','RandomSmearing','k_data','k_mc']
        def evaluate(name,like,*inputs):
            value={'a_mc':0.,'m_mc':.99,'poly_params':.01,'cb_params':1.,
                   'RandomSmearing':.3,'k_data':1.2,'k_mc':1.}[name]
            return ak.ones_like(like)*value
        with patch.object(numerical,'_evaluate_jagged',side_effect=evaluate), \
             patch.object(numerical,'_crystal_ball_invcdf',side_effect=lambda u,*args:ak.ones_like(u)*np.nan):
            p,m,s=numerical._calibrate_muons(a,False,'nominal',ROOT,correction_set={x:x for x in names})
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(p)),[35./.99,199./.99])
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(m)),[.105/.99,.105/.99])
        self.assertEqual(s['unphysical_resolution_fallback'],'scaled_pt')

    def test_pruned_columns_keep_identity_and_optional_physics_inputs(self):
        from TROTASR.utils.histogramming import selected_columns, candidate_columns
        required=['run','luminosityBlock','event','file_id','entry','dataset_id','mStop','mLSP',
            'is_data','is_signal','jet_corrected_pt','jet_eta_all','jet_phi_all','met','met_phi']
        self.assertEqual(selected_columns(required+['unused_heavy_branch']),required)
        self.assertIn('gen_weight',candidate_columns())
        for flavor in ('electron','muon','photon','tau'):
            for name in ('nanoaod_pt','corrected_pt','eta_all','phi_all'):
                self.assertIn(flavor+'_'+name,candidate_columns())
        for name in ('candidateIndex','sourceFatJetIdx','sourceJetIdx0','sourceJetIdx1','sourceJetIdx2','QCDDiscriminant'):
            self.assertIn('TopMixed1pct_'+name,candidate_columns())
        with self.assertRaises(ValueError):selected_columns(['met'])

    def test_endpoint_group_is_canonical_and_rejects_ambiguity(self):
        from TROTASR.utils.histogramming import endpoint_tuple
        self.assertEqual(endpoint_tuple([NONJME_ENDPOINTS[1],'nominal']),('nominal',NONJME_ENDPOINTS[1]))
        for invalid in ([],['invented'],['nominal','nominal']):
            with self.assertRaises(ValueError):endpoint_tuple(invalid)

    def test_region_scores_batch_once_before_mass_split(self):
        from TROTASR.utils.histogramming import region_scores
        from types import SimpleNamespace
        a=ak.Array({'mee':[91.,50.,100.],'mmm':[91.,91.,200.], 'mStop':[100,200,300]})
        selections={r:dict(low=np.array([True,True,True]),block=r) for r in ('SR','DY2E','DY2M')}
        calls=[]
        model=SimpleNamespace(predict=lambda *args,**kw:np.arange(len(args[0]),dtype=float)+.5)
        def features(a,block,region,mask,year=2024):
            self.assertEqual(year, 2024)
            calls.append((region,mask.tolist()))
            return (np.arange(mask.sum()),)*5
        with patch('TROTASR.utils.histogramming.REGIONS',tuple(selections)), \
             patch('TROTASR.utils.histogramming.feature_arrays',side_effect=features):
            scores=region_scores(a,selections,model)
        self.assertEqual(len(calls),3)
        self.assertEqual(scores['DY2E'].tolist(),[.5,0.,1.5])
        self.assertEqual(scores['DY2M'].tolist(),[.5,1.5,0.])

    def test_group_projection_does_not_mutate_or_double_count(self):
        import copy
        from TROTASR.utils.histogramming import split_endpoint
        endpoint=NONJME_ENDPOINTS[0]
        p=dict(contract={'execution':{'endpoints':['nominal',endpoint]}},
            histograms={'highdm':{'nominal':{'x':1},endpoint:{'x':2}}},
            physical_histograms={},dy_rz={'n':1},dy_rz_variations={endpoint:{'s':2}},
            audit={'events_read':10,'migrations':{'nominal':1},'chunks':[{'endpoint':'nominal'},{'endpoint':endpoint}],
                'weights':[{'endpoint':'nominal'},{'endpoint':endpoint}], 'object_endpoints':{endpoint:{'events':10}}})
        before=copy.deepcopy(p);part=split_endpoint(p,endpoint)
        self.assertEqual(p,before)
        self.assertEqual(part['audit']['events_read'],10)
        self.assertEqual(part['dy_rz'],{})
        self.assertEqual(part['audit']['migrations'],{})
        self.assertEqual(list(part['histograms']['highdm']),[endpoint])
        with self.assertRaises(ValueError):split_endpoint(p,NONJME_ENDPOINTS[-1])

    def test_same_process_datasets_have_independent_response_arrays(self):
        from TROTASR.utils.histogramming import _response_identity
        cache={_response_identity(11,0,0,'SR'):np.arange(215),
               _response_identity(22,0,0,'SR'):np.arange(161)}
        for dataset,entries in ((11,215),(22,161)):
            baseline=cache[_response_identity(dataset,0,0,'SR')]
            self.assertEqual(len(baseline),entries)
            np.testing.assert_array_equal(baseline,np.arange(entries))
        self.assertNotEqual(_response_identity(11,500,100,'SR'),_response_identity(11,500,200,'SR'))

    def chunk(self):
        return ak.Array(dict(is_data=[False,False], met=np.array([100.,100.],dtype='float32'),
            met_phi=np.array([0.,0.],dtype='float32'), puppi_met_nanoaod=[90.,90.],
            puppi_met_nanoaod_phi=[0.,0.], puppi_met_unclustered_up=[100.,90.],
            puppi_met_phi_unclustered_up=[0.,0.], puppi_met_corrected=[100.,100.],
            puppi_met_corrected_phi=[0.,0.], electron_corrected_pt=[[25.],[]],
            electron_corrected_mass=[[.0005],[]], electron_pt_all=[[25.],[]],
            electron_mass_all=[[.0005],[]], electron_phi_all=[[0.],[]],
            TopMixed_score=[[.95],[.98]]))

    def test_exact_endpoint_list(self):
        self.assertEqual(len(NONJME_ENDPOINTS),16)
        self.assertEqual(len(set(NONJME_ENDPOINTS)),16)
        self.assertFalse(any(v.startswith(('jes','jer')) for v in NONJME_ENDPOINTS))

    def test_nominal_identity_met_delta_unchanged_row_trota(self):
        a=self.chunk(); resolved={'identity':'stored'}; audit={}
        with patch('TROTASR.utils.shape_kinematics.rebuild',side_effect=lambda out,**kw:ak.Array(out)):
            views=list(nonjme_event_views([a],2024,resolved,audit,('metUnclusteredUp',)))
        self.assertIs(views[0][0],a)
        self.assertIs(views[1][1],resolved)
        v=views[1][0]
        np.testing.assert_array_equal(v['met'],[110.,100.])
        self.assertEqual(ak.to_list(v['TopMixed_score']),ak.to_list(a['TopMixed_score']))
        self.assertEqual(audit['metUnclusteredUp'],dict(events=2,affected_events=1))
        self.assertEqual(np.asarray(v['met']).dtype,np.dtype('float32'))

    def test_object_delta_anchored_to_stored_and_met_opposite(self):
        a=self.chunk(); p=ak.Array([[20.],[]]);m=ak.Array([[.0004],[]])
        def calibrate(raw,year,flavor,shift):
            return (p,m,{}) if shift=='nominal' else (p+2,m*1.1,{})
        with patch('TROTASR.utils.corrections.nonjme_calibration_inputs',return_value={}), \
             patch('TROTASR.utils.corrections.nonjme_calibrate',side_effect=calibrate), \
             patch('TROTASR.utils.shape_kinematics.rebuild',side_effect=lambda out,**kw:ak.Array(out)):
            v=list(nonjme_event_views([a],2025,{}, {},('electronScaleUp',)))[1][0]
        self.assertEqual(ak.to_list(v['electron_pt_all']),[[27.],[]])
        np.testing.assert_array_equal(v['met'],[98.,100.])
        self.assertEqual(ak.to_list(a['electron_pt_all']),[[25.],[]])

    def test_data_has_nominal_only(self):
        a=ak.with_field(self.chunk(),[True,True],'is_data')
        self.assertEqual(len(list(nonjme_event_views([a],2024,{},{}))),1)

    def test_duplicate_or_unknown_endpoint_rejected(self):
        for endpoints in [('invented',),('electronScaleUp','electronScaleUp')]:
            with self.assertRaises(ValueError):list(nonjme_event_views([],2024,{}, {},endpoints))

    def test_existing_out_of_acceptance_policy(self):
        p=ak.Array([[10.,30.,40.]])
        got=_valid_scaled_pt(p,ak.Array([[11.,-1.,41.]]),ak.Array([[False,True,True]]))
        self.assertEqual(ak.to_list(got),[[10.,30.,41.]])

    def test_contract_keeps_stored_trota_and_nominal_weight_only(self):
        with patch('TROTASR.utils.corrections.nonjme_payload_contract',return_value={'payloads':'verified'}):
            c=current_contract(2024,'stored_trota_nonjme',['nominal'])
        self.assertEqual(c['variations'],['nominal'])
        self.assertEqual(c['object_endpoints'],list(NONJME_ENDPOINTS))
        self.assertNotIn('trota_models_sha256',c)


if __name__=='__main__':unittest.main()
