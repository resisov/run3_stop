import copy
import unittest
import numpy as np
from TROTASR.utils.io import read_json
from TROTASR.utils.paths import ROOT
from TROTASR.utils.sr_merge import validate_merge, bin_groups, project, project_prediction
from TROTASR.workflows.prediction_plot_inputs import combine, expected_labels, flat_blocks


SR_LAYOUT = ROOT/'jsons/highdm_sr_binning.json'
SR_MERGE = ROOT/'jsons/sr_category_merge.json'


@unittest.skipUnless(SR_LAYOUT.is_file() and SR_MERGE.is_file(),
    'Canonical high-dM SR layout/merge are not installed locally')
class SRMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=read_json(SR_MERGE)
        cls.groups=bin_groups(cls.config)

    def test_exact_predicate_union_and_partition(self):
        validate_merge(self.config)
        self.assertEqual(len(self.groups),118)
        self.assertEqual(sorted(i for g in self.groups for i in g),list(range(162)))
        self.assertEqual(len(self.config['categories']),21)

    def test_no_duplicate_or_omitted_category(self):
        c=copy.deepcopy(self.config);c['categories'][0]['source_categories_1based']=[1,1]
        with self.assertRaises(ValueError):validate_merge(c)

    def test_no_widened_selection(self):
        c=copy.deepcopy(self.config);c['categories'][0]['predicates']['Nres']={'ge':1}
        with self.assertRaises(ValueError):validate_merge(c)

    def test_no_interpolation(self):
        c=copy.deepcopy(self.config);c['categories'][0]['edges'][1]=310
        with self.assertRaises(ValueError):validate_merge(c)

    def test_only_requested_last_two_bins_are_merged(self):
        before=copy.deepcopy(self.config)
        changes=before.pop('recoil_bin_merges')
        self.assertEqual([c['class_1based'] for c in changes],[2,15,20])
        for c in changes:
            before['categories'][c['class_1based']-1]['edges']=c['source_edges']
        before['bins_per_year']=121
        validate_merge(before)
        old=bin_groups(before);new=self.groups;offset_old=offset_new=0
        for i,(a,b) in enumerate(zip(before['categories'],self.config['categories']),1):
            n,m=len(a['edges'])-1,len(b['edges'])-1
            expected=old[offset_old:offset_old+n]
            if i in (2,15,20):expected=expected[:-2]+[expected[-2]+expected[-1]]
            self.assertEqual(new[offset_new:offset_new+m],expected)
            offset_old+=n;offset_new+=m

    def test_tail_merge_requires_explicit_matching_authorization(self):
        c=copy.deepcopy(self.config);c.pop('recoil_bin_merges')
        with self.assertRaises(ValueError):validate_merge(c)
        c=copy.deepcopy(self.config);c['recoil_bin_merges'][0]['class_1based']=1
        with self.assertRaises(ValueError):validate_merge(c)
        c=copy.deepcopy(self.config);c['recoil_bin_merges'].append(copy.deepcopy(c['recoil_bin_merges'][0]))
        with self.assertRaises(ValueError):validate_merge(c)

    def test_signed_endpoints_and_variance_integrals_conserved(self):
        for a in (np.arange(162.),np.arange(162.)**2,np.arange(162.)-80):
            np.testing.assert_allclose(project(a,self.groups).sum(),a.sum())
        with self.assertRaises(ValueError):project(np.ones(161),self.groups)

    def test_real_prefit_threshold_decision_and_conservation(self):
        p=ROOT/self.config['decision_payload']
        if not p.is_file():
            self.skipTest('Accepted SR merge decision payload is not installed locally')
        payload=read_json(p)
        old=combine([payload['channels'][y+'/SR/highdm'] for y in ('2024','2025')])
        new=project_prediction(old,self.config)
        self.assertEqual(len(new['background']),118)
        offset=0
        for c in self.config['categories']:
            n=len(c['edges'])-1
            self.assertGreaterEqual(sum(new['background'][offset:offset+n]),2)
            np.testing.assert_allclose(sum(new['background'][offset:offset+n]),c['background_total'])
            offset+=n
        for field in ('groups','group_variances','signals'):
            for k in old[field]:np.testing.assert_allclose(new[field][k].sum(),old[field][k].sum())
        self.assertNotIn('data',new)
        self.assertEqual(len(flat_blocks(new,'highdm','SR',self.config)),21)

    def test_correlated_endpoints_summed_before_envelope(self):
        record=dict(sr_data_blinded=True,background=np.ones(162),variance=np.ones(162),
            groups={'Top':np.ones(162)},group_variances={'Top':np.ones(162)},signals={},
            nuisance_deltas={'a':{'up':np.zeros(162),'down':np.zeros(162)}})
        g=next(g for g in self.groups if len(g)==2)
        record['nuisance_deltas']['a']['up'][g]=[2,-2]
        record['nuisance_deltas']['a']['down'][g]=[-3,3]
        out=project_prediction(record,self.config)
        i=self.groups.index(g)
        self.assertEqual(out['nuisance_deltas']['a']['up'][i],0)
        self.assertAlmostEqual(out['uncertainty'][i],np.sqrt(2))
        record['data']=np.zeros(162)
        with self.assertRaises(ValueError):project_prediction(record,self.config)

    def test_legacy_label_contract_unchanged(self):
        self.assertEqual(len(expected_labels('SR_highdm_c1_2024')),162)
        self.assertEqual(len(expected_labels('SR_highdm_c1_2024',self.config)),118)
        self.assertEqual(expected_labels('GCR_lowdm_c1_2024'),expected_labels('GCR_lowdm_c1_2024',self.config))


if __name__=='__main__':unittest.main()
