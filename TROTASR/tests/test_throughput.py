import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from TROTASR.utils import event_selections as selections
from TROTASR.utils.private_scales import topw_pass_fail_triplet, AnalysisScaleFactorUnavailable


class ThroughputTests(unittest.TestCase):
    def test_nominal_topw_preserves_every_efficiency_case(self):
        rng=np.random.default_rng(123)
        efficiency=np.r_[0.,1.,.5,rng.uniform(0,1,100)]
        nominal=rng.uniform(.1,2,len(efficiency))
        tagged=rng.random(len(efficiency))>.5
        denominator=np.ones(len(efficiency));denominator[2]=0
        full=topw_pass_fail_triplet(tagged,efficiency,nominal,nominal*1.2,nominal*.8,denominator=denominator)
        only=topw_pass_fail_triplet(tagged,efficiency,nominal,denominator=denominator,include_variations=False)
        self.assertEqual(len(only),1)
        np.testing.assert_array_equal(full[0],only[0])

    def test_nominal_topw_does_not_ignore_invalid_nominal(self):
        with self.assertRaises(AnalysisScaleFactorUnavailable):
            topw_pass_fail_triplet(np.array([True]),np.array([.5]),np.array([np.nan]),include_variations=False)

    def example(self):
        a={k:ak.Array([[]]) for k in selections.OVERLAP_FIELDS}
        a.update(jet_source_index_all=ak.Array([[10,11]]),jet_eta_all=ak.Array([[0.,2.]]),
                 jet_phi_all=ak.Array([[0.,2.]]),fatjet_source_index_all=ak.Array([[7]]),
                 fatjet_eta_all=ak.Array([[0.]]),fatjet_phi_all=ak.Array([[0.]]),
                 fatjet_subjet_index1_all=ak.Array([[-1]]),fatjet_subjet_index2_all=ak.Array([[-1]]),
                 fatjet_boosted_top_pass_all=ak.Array([[True]]),fatjet_boosted_w_pass_all=ak.Array([[False]]),
                 nTopMixed1pct=np.array([0]),file_id=np.array([1]),entry=np.array([2]))
        for k in selections.MIXED_FIELDS:a['TopMixed1pct_'+k]=ak.Array([[]])
        return ak.Array(a)

    def test_buffers_shared_without_changing_decisions(self):
        a=self.example();context=selections.topology_buffers(a)
        self.assertEqual(context[2],[{7:{10}}])
        for region in ('SR','LLCR','QCDCR'):
            original=selections.topology_counts(a,{},region)
            prepared=selections.topology_counts(a,{},region,context)
            for i in (0,3):np.testing.assert_array_equal(original[i],prepared[i])
            self.assertEqual(ak.to_list(original[1]),ak.to_list(prepared[1]))
            self.assertEqual(original[2],prepared[2])

    def test_buffers_are_rebuilt_for_each_shift(self):
        a=self.example()
        shifted=ak.with_field(a,ak.Array([[False]]),'fatjet_boosted_top_pass_all')
        self.assertEqual(selections.topology_counts(a,{},'SR')[0][0,0],1)
        self.assertEqual(selections.topology_counts(shifted,{},'SR')[0][0,0],0)

    def test_duplicate_identity_still_rejected(self):
        a=ak.with_field(self.example(),ak.Array([[10,10]]),'jet_source_index_all')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            selections.topology_buffers(a)

    def test_fixed_jet_cache_reuses_only_identical_references(self):
        a = self.example(); resolved = {}
        cache = selections.FixedJetTopologyCache(a, resolved)
        cache.validate(a, resolved)
        with patch.object(selections, 'topology_counts', wraps=selections.topology_counts) as count:
            one = cache.get(a, 'SR'); two = cache.get(a, 'SR')
            self.assertIs(one, two)
            self.assertEqual(count.call_count, 1)
        changed = ak.with_field(a, ak.Array([[False]]), 'fatjet_boosted_top_pass_all')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            cache.validate(changed, resolved)
        with self.assertRaisesRegex(ValueError, 'different resolved'):
            cache.validate(a, {})

    def test_reference_threshold_crossing_recomputes_cleaning(self):
        a = self.example(); cache = selections.FixedJetTopologyCache(a, {})
        empty = ak.Array([[]]); reference = ak.Array([[0.]])
        with patch.object(selections, '_references', side_effect=[(empty,empty),(reference,reference)]), \
             patch.object(selections, 'topology_counts', side_effect=['before','after']) as count:
            self.assertEqual(cache.get(a, 'GCR'), 'before')
            self.assertEqual(cache.get(a, 'GCR'), 'after')
            self.assertEqual(count.call_count, 2)

    def test_pog_payload_parsed_once(self):
        from TROTASR.utils import pog_corrections as pog
        pog._load_pog.cache_clear()
        with patch.object(pog.correctionlib.CorrectionSet, 'from_file', return_value=object()) as load:
            self.assertIs(pog._load_pog('test-payload'), pog._load_pog('test-payload'))
            self.assertEqual(load.call_count, 1)
        pog._load_pog.cache_clear()

    def test_reused_selection_preserves_per_call_veto_audit(self):
        from TROTASR.utils.region_kinematics import update_veto_leptons
        a = ak.Array(dict(n_e_veto=[2], n_m_loose=[1],
            electron_veto_pt=[[7.,20.]], electron_veto_eta=[[0.,0.]],
            electron_veto_phi=[[0.,0.]], muon_loose_pt=[[8.]],
            muon_loose_eta=[[0.]], muon_loose_phi=[[0.]], met=[300.], met_phi=[0.]))
        first = update_veto_leptons(a)
        before = ak.to_list(a)
        second = update_veto_leptons(a)
        self.assertEqual(first,dict(events_with_removed_electrons=1,events_with_removed_muons=1))
        self.assertEqual(second,dict(events_with_removed_electrons=0,events_with_removed_muons=0))
        self.assertEqual(ak.to_list(a),before)

    def test_execution_handoff_preserves_completed_hash_only(self):
        from TROTASR.workflows import run_systematic_campaign as campaign
        old, new = {'code':'old'}, {'code':'new'}
        spec = dict(id='revision',contracts={'2024':new})
        spec[campaign.UNCHANGED_VIEW_REUSE] = dict(campaign='unused',
            retained_completed={'task':'exact'},retained_running={},previous_contracts={'2024':old})
        task = dict(key='task',year=2024)
        self.assertEqual(campaign.refresh_result_contract(spec,task,'exact'),old)
        self.assertEqual(campaign.refresh_result_contract(spec,task,'different'),new)
        self.assertEqual(campaign.refresh_result_contract(spec,task,None),new)

    def test_execution_handoff_requires_exact_inflight_receipt(self):
        from TROTASR.workflows import run_systematic_campaign as campaign
        old, new = {'code':'old'}, {'code':'new'}
        identity = dict(pid=123,start_ticks=456,arguments=['worker'])
        spec = dict(id='revision',contracts={'2024':new})
        spec[campaign.UNCHANGED_VIEW_REUSE] = dict(campaign='unused',retained_completed={},
            retained_running={'task':identity},previous_contracts={'2024':old})
        receipt = dict(identity=identity,status='complete',exit_code=0,
            output_sha256='exact',payload_revision='revision')
        task = dict(key='task',year=2024)
        with patch.object(campaign,'read_json',return_value=receipt):
            self.assertEqual(campaign.refresh_result_contract(spec,task,'exact'),old)
        for delta in (dict(status='failed'),dict(exit_code=1),dict(output_sha256='different'),
                      dict(payload_revision='different'),dict(identity=dict(identity,start_ticks=457))):
            with patch.object(campaign,'read_json',return_value=dict(receipt,**delta)):
                self.assertEqual(campaign.refresh_result_contract(spec,task,'exact'),new)


if __name__=='__main__':unittest.main()
