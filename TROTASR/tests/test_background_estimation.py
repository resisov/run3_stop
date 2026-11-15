import copy
import json
import unittest
from TROTASR.utils.background_estimation import measure
from TROTASR.utils.photon_estimation import factor
from TROTASR.utils.physical_histograms import binned, fill
from TROTASR.utils.sr_binning import NATIVE_EDGES
import numpy as np
from types import SimpleNamespace


def measurement_fixture():
    def recoil(v):
        return {'recoil': dict(edges=NATIVE_EDGES, sumw=[v]*8, sumw2=[abs(v)]*8, entries=[1]*8)}
    def scalar(v):
        return dict(edges=[0.,1.], sumw=[v], sumw2=[abs(v)], entries=[int(v)])
    hist, dy = {}, {}
    for mode in ('highdm','lowdm'):
        hist[mode] = {'nominal': {}}
        dy[mode] = dict(yields={}, mll={})
        for region in ('GCR','DY2E','DY2M'):
            target = 'GJ' if region == 'GCR' else 'DY'
            hist[mode]['nominal'][region] = {g:{target:recoil(10.), 'ST':recoil(2.), 'data':recoil(22.)}
                                                   for g in ('Nb1','Nb2')}
        for channel in ('DY2E','DY2M'):
            dy[mode]['yields'][channel] = {g:{
                'on':{p:scalar(v) for p,v in [('data',210.),('zll',100.),('other',20.)]},
                'off':{p:scalar(v) for p,v in [('data',60.),('zll',10.),('other',80.)]}}
                for g in ('Nb1','Nb2plus')}
    return dict(status='complete', sr_data_blinded=True, contract={'year':2024}, histograms=hist, dy_rz=dy,
                scope='synthetic_unit_test_not_physics_result',
                input_metadata={s:dict(process=s,is_data=True) for s in ('JetMET','EGamma','Muon')})


def zgamma_policy_fixture():
    return dict(id='zgamma_2023BPix_measurement_merge_20261022', year='2023BPix',
                user_approved=True, mode='highdm', original_edges=[250, 300, 350, 400, 500, 1500],
                measurement_edges=[250, 300, 400, 500, 1500], analysis_bins_unchanged=True,
                gnn_boundaries_unchanged=True, preserve_other_measurements=True)


class BackgroundTests(unittest.TestCase):
    def test_approved_zgamma_merge_preserves_other_results(self):
        from TROTASR.workflows.build_background_estimation import merged_zgamma_measurement
        p=measurement_fixture();p['contract']['year']='2023BPix'
        for region in ('DY2E','DY2M'):
            for group in ('Nb1','Nb2'):
                leaf=p['histograms']['highdm']['nominal'][region][group]['DY']['recoil']
                leaf['sumw'][2]=-1.
        old=measure(p)['double_ratio'];preserved=copy.deepcopy(old)
        policy=zgamma_policy_fixture()
        result=merged_zgamma_measurement(p,old,policy)
        self.assertEqual(old,preserved)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['lowdm'],old['lowdm'])
        for old_i,new_i in ((0,0),(3,2),(4,3)):
            self.assertEqual(result['highdm']['bins'][new_i],old['highdm']['bins'][old_i])
        self.assertEqual((result['highdm']['bins'][1]['low'],result['highdm']['bins'][1]['high']),(300.,400.))
        totals=result['measurement_merge_audit']
        for before,after in zip(totals['input_totals'],totals['merged_totals']):
            for key in before:self.assertAlmostEqual(sum(before[key]),sum(after[key]))
        for key,value in (('year','2024'),('measurement_edges',[250,400,1500]),
                          ('preserve_other_measurements',False)):
            bad=dict(policy,**{key:value})
            with self.assertRaises(ValueError):merged_zgamma_measurement(p,old,bad)
        metadata_only=dict(policy,id='receipt label',user_approved=False)
        self.assertEqual(merged_zgamma_measurement(p,old,metadata_only), result)

    def test_zgamma_recovery_reuses_same_output_without_history(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from TROTASR.utils.io import read_json, write_json
        from TROTASR.utils.paths import ROOT
        from TROTASR.workflows.build_background_estimation import recover_zgamma

        payload = measurement_fixture()
        payload['contract']['year'] = '2023BPix'
        for region in ('DY2E', 'DY2M'):
            for group in ('Nb1', 'Nb2'):
                payload['histograms']['highdm']['nominal'][region][group]['DY']['recoil']['sumw'][2] = -1.
        products = measure(payload)
        with TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            files = {}
            for name, product in products.items():
                path = output / (name+'.json.gz')
                write_json(path, product)
                files[path.name] = path.read_bytes()
            manifest = output / 'measurement_manifest.json'
            write_json(manifest, dict(status='blocked', provenance=dict(campaign_year='2023BPix'),
                stages={n:p['status'] for n, p in products.items()},
                files={name: 'legacy-receipt' for name in files}))
            self.assertEqual(recover_zgamma(output, payload), 0)
            self.assertEqual(recover_zgamma(output, payload), 0)
            self.assertEqual(read_json(manifest)['status'], 'complete')
            self.assertFalse((output/'history').exists())
            self.assertFalse((output/'zgamma_recovery.json').exists())
            for name, contents in files.items():
                if name != 'double_ratio.json.gz':
                    self.assertEqual((output/name).read_bytes(), contents)

    def test_existing_measurement_equations_and_algebraic_closure(self):
        products = measure(measurement_fixture())
        self.assertTrue(all(p['status']=='complete' for p in products.values()))
        for name in ('rz_high','rz_low'):
            for row in products[name][name]['combined'].values():
                self.assertAlmostEqual(row['RZ'],2.)
            for groups in products[name][name]['channels'].values():
                for row in groups.values():
                    self.assertAlmostEqual(row['RT'],.5)
        for groups in (products['sgamma']['highdm'],products['sgamma']['lowdm_families']):
            for row in groups.values():
                self.assertAlmostEqual(row['Q']['value'],2.)
                self.assertTrue(all(abs(b['Sgamma']['value']-1.)<1e-12 for b in row['bins']))
        for mode in ('highdm','lowdm'):
            for row in products['double_ratio'][mode]['bins']:
                self.assertAlmostEqual(row['double_ratio'],1.)
        json.dumps(products, allow_nan=False)

    def test_no_unity_fallback_for_missing_background_measurements(self):
        p = measurement_fixture(); del p['input_metadata']['Muon']
        with self.assertRaisesRegex(ValueError,'coverage'): measure(p)
        p = measurement_fixture(); del p['dy_rz']['lowdm']['yields']['DY2E']['Nb1']['off']['zll']
        with self.assertRaisesRegex(ValueError,'Missing on/off-Z'): measure(p)
        self.assertEqual(factor(2.,2.,0.,0.)['status'],'unavailable')

    def test_nonpositive_photon_measurement_is_not_marked_complete(self):
        p = measurement_fixture()
        p['histograms']['lowdm']['nominal']['GCR']['Nb1']['data']['recoil']['sumw']=[0.]*8
        self.assertEqual(measure(p)['sgamma']['status'],'blocked')

    def test_physical_histogram_preserves_legacy_upper_only_overflow(self):
        r=binned([-1.,.5,1.5,2.,3.],[1.,2.,3.,4.,5.],[0.,1.,2.],'fold')
        self.assertEqual(r['sumw'],[2.,12.])
        self.assertEqual(r['sumw2'],[4.,50.])
        r=binned([-1.,.5,1.5,2.,3.],[1.,2.,3.,4.,5.],[0.,1.,2.],'exclude')
        self.assertEqual(r['sumw'],[2.,7.])

    def test_physical_counts_use_exclusive_w_and_cleaned_jets(self):
        b=SimpleNamespace(nb=np.array([1,2]),njet=np.array([4,7]),ht=np.array([450.,650.]),recoil=np.array([300.,400.]))
        counts=np.array([[0,1,0,0],[1,1,0,1]])
        out={}
        fill(out,'highdm','nominal','GCR','ST',{'met':np.array([150.,180.])},b,np.array([True,True]),
             counts,np.array([True,True]),np.array([2.,3.]))
        first=out['highdm']['nominal']['GCR']['Nb1']['ST']
        self.assertEqual(first['nw']['sumw'][0],2.)
        self.assertNotIn('met',first)
        self.assertEqual(sum(first['njet']['sumw']),2.)
        fill(out,'highdm','nominal','SR','data',{},b,np.array([True,True]),counts,np.ones(2,dtype=bool),np.ones(2))
        self.assertNotIn('SR',out['highdm']['nominal'])


if __name__=='__main__': unittest.main()
