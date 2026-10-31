"""Small synthetic numerical fixtures with the FULL adopted bin layout."""
import copy
import unittest
import numpy as np
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT
from TROTASR.utils.statistics_inputs import build_inputs, HistogramView
from TROTASR.utils.statistics_model import nps_nuisance_name
from TROTASR.utils.statistical_templates import assemble, card_text


class CombinedContractTests(unittest.TestCase):
    def fixture(self):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        from TROTASR.utils.corrections import NONJME_ENDPOINTS
        coverage = dict(full_input_inventory=True, expected_files=2, completed_files=2, failed_files=0)
        return dict(scope='full_combined_systematic_production', status='complete', preliminary=True,
            central_conventions_consistent=False, coverage=coverage,
            audit=dict(nonjme_source_coverage=dict(coverage)),
            combination_provenance=dict(policy=dict(COMBINATION_POLICY),
                sources={name:dict(sha256='a'*64, summary_sha256='b'*64) for name in ('jme','nonjme')}),
            contract=dict(shape_mode='jme_with_stored_nonjme', combination_policy=dict(COMBINATION_POLICY),
                object_endpoints=list(NONJME_ENDPOINTS), component_contracts=dict(
                    jme=dict(shape_mode='cms_trota_jme'), nonjme=dict(shape_mode='stored_trota_nonjme'))))

    def test_exact_twenty_shape_endpoints(self):
        from TROTASR.utils.statistics_inputs import combined_shape_endpoints
        self.assertEqual(len(combined_shape_endpoints(self.fixture())), 20)

    def test_policy_coverage_and_source_identity_cannot_be_waived(self):
        from TROTASR.utils.statistics_inputs import combined_shape_endpoints
        for change in ('approval','reanchor','central','coverage','endpoint','hash','component'):
            p = self.fixture()
            if change == 'approval': p['contract']['combination_policy']['user_approved'] = False
            if change == 'reanchor': p['combination_provenance']['policy']['reanchor'] = True
            if change == 'central': p['central_conventions_consistent'] = True
            if change == 'coverage': p['audit']['nonjme_source_coverage']['completed_files'] = 1
            if change == 'endpoint': p['contract']['object_endpoints'].pop()
            if change == 'hash': p['combination_provenance']['sources']['jme']['sha256'] = 'missing'
            if change == 'component': p['contract']['component_contracts']['nonjme']['shape_mode'] = 'cms_trota_jme'
            with self.subTest(change=change), self.assertRaises(ValueError):
                combined_shape_endpoints(p)

def fixture(year=2024):
    gnn = read_json(ROOT / 'gnn4lowdm/config.json')
    native = [250.,300.,350.,400.,500.,650.,800.,1000.,1500.]
    def recoil(value):
        return {'recoil': dict(edges=native, sumw=[value]*8, sumw2=[value*value]*8, entries=[1]*8)}
    def score(category, region, value):
        edges = gnn['sr_binning']['edges_by_category'][category] if region == 'SR' else gnn['cr_binning']['score_edges']
        matrix = np.full((5,8), value, dtype=float)
        matrix[:,0] = 0.  # The retained low-dM selection has recoil >= 300 GeV.
        return {
            'gnn_score': dict(edges=edges, sumw=matrix.sum(axis=1).tolist(),
                              sumw2=(matrix*matrix).sum(axis=1).tolist(), entries=[7]*5),
            'gnn_recoil': dict(edges=[edges,native], sumw=matrix.tolist(), sumw2=(matrix*matrix).tolist(),
                               entries=(matrix != 0).astype(int).tolist())}
    hist = {'highdm': {'nominal': {}}, 'lowdm': {'nominal': {}}}
    for region in ('LLCR','QCDCR','GCR'):
        hist['highdm']['nominal'][region] = {
            category: {p: recoil(2. if p == 'data' else 1.) for p in ('ST','TT','QCD','GJ','WtoLNu','data')}
            for category in ('Nb1','Nb2')}
        hist['lowdm']['nominal'][region] = {
            category: {p: score(category,region,2. if p == 'data' else 1.) for p in ('ST','TT','QCD','GJ','WtoLNu','data')}
            for category in ('Nb1','Nb2')}
    hist['highdm']['nominal']['SR'] = {
        f'{nb},0,0,1,0': {p: recoil(float(nb)) for p in ('ST','QCD','Zto2Nu','mStop1200_mLSP500')}
        for nb in (1,2,3)}
    hist['lowdm']['nominal']['SR'] = {
        category: {p: score(category,'SR',1.) for p in ('ST','QCD','Zto2Nu','mStop1200_mLSP500')}
        for category in gnn['sr_binning']['category_labels']}
    payload = dict(status='complete', sr_data_blinded=True,
                   highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                   scope='synthetic_unit_test_not_physics_result', contract={'year':year}, histograms=hist,
                   input_metadata={
                       'a': {'is_data':True,'process':'JetMET','dataset':'/JetMET/Fixture/NANOAOD'},
                       'b': {'is_data':True,'process':'EGamma','dataset':'/EGamma/Fixture/NANOAOD'}})
    provenance = {'campaign_year': str(year)}
    def gamma(q, offset):
        return {'Q':dict(status='complete',value=q),
                'bins':[{'Sgamma':dict(status='complete',value=offset+i)} for i in range(5)]}
    sgamma = dict(status='complete', provenance=provenance,
                  highdm={g:gamma(2.,1.) for g in ('Nb1','Nb2plus')},
                  lowdm_families={g:gamma(3.,2.) for g in ('Nb1','Nb2plus')})
    rz_high = dict(status='complete', provenance=provenance,
                   rz_high={'combined':{g:dict(status='complete',RZ=2.,RZ_stat=.2) for g in ('Nb1','Nb2plus')}})
    rz_low = dict(status='complete', provenance=provenance,
                  rz_low={'combined':{g:dict(status='complete',RZ=4.,RZ_stat=.4) for g in ('Nb1','Nb2plus')}})
    bins = [dict(status='complete', low=lo, high=hi, double_ratio=1.1)
            for lo,hi in zip([250.,300.,350.,400.,500.],[300.,350.,400.,500.,1500.])]
    ratio = dict(status='complete', provenance=provenance, highdm={'bins':bins}, lowdm={'bins':bins})
    return payload, sgamma, rz_high, rz_low, ratio


class StatisticsInputTests(unittest.TestCase):
    def build(self, values=None):
        return build_inputs(*(fixture() if values is None else values), 'T2tt', 1200, 500)

    def test_fixed_layout_with_sparse_sr(self):
        out = self.build()
        self.assertEqual(out['status'], 'complete')
        self.assertEqual(out['bin_counts']['SR_highdm'], 162)
        self.assertEqual(out['bin_counts']['SR_lowdm'], 30)
        for region in ('LLCR','QCDCR','GCR'):
            self.assertEqual(out['bin_counts'][region+'_highdm'],16)
            self.assertEqual(out['bin_counts'][region+'_lowdm'],10)
        self.assertEqual(sum(out['bin_counts'].values()),270)
        channels, data, _, issues = assemble([out])
        self.assertFalse(issues)
        self.assertEqual(sum(map(len,channels.values())),270)
        self.assertIn('* autoMCStats 10 1 1',card_text(channels,data))

    def test_shape_sparse_gain_loss_and_flat_pair_retention(self):
        values = fixture()
        payload = values[0]
        payload['contract']['shape_mode'] = 'cms_trota_jme'
        for mode, hist in payload['histograms'].items():
            for name in ('jesTotalUp', 'jesTotalDown', 'jerUp', 'jerDown'):
                hist[name] = copy.deepcopy(hist['nominal'])
        h = payload['histograms']['highdm']
        gained = '1,0,1,0,0'
        h['jesTotalUp']['SR'][gained] = h['jesTotalUp']['SR'].pop('1,0,0,1,0')
        view = HistogramView(payload, 2024, read_json(ROOT/'gnn4lowdm/config.json'))
        self.assertEqual(view._cell('highdm','SR','1,0,0,1,0','ST','jesTotalUp',0), (0.,0.))
        self.assertEqual(view._cell('highdm','SR',gained,'ST','jesTotalUp',0), (1.,1.))
        out = self.build(values)
        channels, data, _, issues = assemble([out], preserve_flat=True)
        self.assertFalse(issues)
        nuisance = nps_nuisance_name('jesTotal','2024')
        top = [(c,p,r) for (c,p),r in data.items() if c.startswith('SR_highdm') and r['family']=='Top']
        self.assertEqual(sum(r['nominal'].sum() for c,p,r in top),48.)
        self.assertEqual(sum(r['variations'][nuisance]['up'].sum() for c,p,r in top),48.)
        self.assertTrue(any(not np.array_equal(r['nominal'],r['variations'][nuisance]['up']) for c,p,r in top))
        flat = nps_nuisance_name('jer','2024')
        self.assertTrue(any(flat in r['variations'] for r in data.values()))
        self.assertNotIn(flat+' shape ',card_text(channels,data))

    def test_shape_endpoints_require_explicit_matching_contract(self):
        values=fixture()
        for hist in values[0]['histograms'].values():
            hist['jesTotalUp']={}; hist['jesTotalDown']={}
        with self.assertRaisesRegex(ValueError,'explicit histogram contract'):
            self.build(values)

    def test_nb_pooled_sr_keeps_two_cr_bindings_and_exact_variance(self):
        out = self.build()
        bins = [b for b in out['groups']['SR_highdm'] if 'Top_Nb1_u0' in b['backgrounds']]
        self.assertEqual(len(bins),1)
        b = bins[0]
        self.assertEqual(b['backgrounds']['Top_Nb1_u0']['nominal'][0],1.)
        self.assertEqual(b['backgrounds']['Top_Nb2plus_u0']['nominal'][0],5.)
        self.assertEqual(b['backgrounds']['Top_Nb2plus_u0']['sumw2'][0],13.)
        self.assertNotEqual(b['rate_params']['Top_Nb1_u0'],b['rate_params']['Top_Nb2plus_u0'])

    def test_high_sr_yields_and_signal_are_conserved(self):
        out = self.build()
        top = sum(r['nominal'][0] for b in out['groups']['SR_highdm']
                  for p,r in b['backgrounds'].items() if p.startswith('Top_'))
        signal = sum(b['physical'].get('signal',{}).get('nominal',0.) for b in out['groups']['SR_highdm'])
        self.assertEqual(top,48.)
        self.assertEqual(signal,48.)

    def test_low_sgamma_uses_actual_recoil_not_score_index(self):
        out = self.build()
        for b in out['groups']['SR_lowdm']:
            z = sum(r['nominal'][0]*b['rate_initial'][p] for p,r in b['backgrounds'].items() if p.startswith('Zto2Nu_'))
            # Seven unit-yield native cells: 1*3 + 1*4 + 1*5 + 4*6; RZ=4.
            self.assertEqual(z,144.)
        parameters = {p for b in out['groups']['SR_lowdm'] for proc,p in b['rate_params'].items() if proc == 'Top'}
        self.assertEqual(len(parameters),2)
        self.assertTrue(all('NISR' not in p for p in parameters))

    def test_high_nonclosure_retains_native_recoil_scope(self):
        out = self.build()
        names = {r['name'] for b in out['groups']['SR_highdm'] for records in b['extra_lnN'].values()
                 for r in records if 'zgammaNonclosure' in r['name']}
        self.assertTrue(any('u500to650' in n for n in names))
        self.assertTrue(any('u1000to1500' in n for n in names))

    def test_unmatched_sr_parameter_blocks_instead_of_changing_binning(self):
        values = fixture()
        for category in values[0]['histograms']['highdm']['nominal']['LLCR'].values():
            category.pop('ST'); category.pop('TT'); category.pop('WtoLNu')
        out = self.build(values)
        self.assertEqual(out['status'],'blocked')
        self.assertTrue(any(x['problem']=='sr_only_parameter' for x in out['issues']))
        self.assertEqual(out['bin_counts']['SR_highdm'],162)

    def test_cr_only_policy_audited(self):
        values = fixture()
        for samples in values[0]['histograms']['highdm']['nominal']['SR'].values():
            samples.pop('QCD')
        out = self.build(values)
        self.assertTrue(out['cr_only_unity_audit'])
        self.assertTrue(all('qcd_norm_highdm' in a['parameter'] and a['fixed']==1. for a in out['cr_only_unity_audit']))

    def test_missing_measured_factors_rejected(self):
        values = fixture(); values[1]['status']='missing'
        with self.assertRaisesRegex(ValueError,'Incomplete measured factor'):
            self.build(values)

    def test_missing_cr_data_rejected(self):
        values = fixture(); values[0]['input_metadata']={}
        with self.assertRaisesRegex(ValueError,'data streams'):
            self.build(values)

    def test_blinding_rejected_if_data_sr_exists(self):
        values = fixture(); h=values[0]['histograms']['highdm']['nominal']['SR']
        h['1,0,0,1,0']['data']=copy.deepcopy(h['1,0,0,1,0']['ST'])
        with self.assertRaisesRegex(ValueError,'SR observations'):
            self.build(values)

    def test_years_do_not_leak_via_global_state(self):
        a=self.build(fixture(2024)); b=self.build(fixture(2025)); c=self.build(fixture(2024))
        parameters=lambda out:{p for bins in out['groups'].values() for bin in bins for p in bin['rate_params'].values()}
        self.assertEqual(parameters(a),parameters(c))
        self.assertTrue(all(p.endswith('_2025') for p in parameters(b)))
        self.assertNotEqual(nps_nuisance_name('jesTotal','2024'),nps_nuisance_name('jesTotal','2025'))
        self.assertEqual(nps_nuisance_name('btagSF_bc_correlated','2024'),nps_nuisance_name('btagSF_bc_correlated','2025'))

    def test_gnn_recoil_marginal_mismatch_rejected(self):
        values=fixture(); leaf=values[0]['histograms']['lowdm']['nominal']['SR']['Nb1_NISR0']['Zto2Nu']
        leaf['gnn_recoil']['sumw'][0][1]+=1.
        with self.assertRaisesRegex(ValueError,'does not conserve'):
            self.build(values)

    def test_unknown_low_category_cannot_be_silently_lost(self):
        values=fixture(); sr=values[0]['histograms']['lowdm']['nominal']['SR']
        sr['not_in_the_map']=copy.deepcopy(sr['Nb1_NISR0'])
        with self.assertRaisesRegex(ValueError,'Unknown category'):
            self.build(values)


if __name__=='__main__': unittest.main()
