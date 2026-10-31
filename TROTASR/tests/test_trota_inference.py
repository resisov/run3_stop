import unittest
import numpy as np
import awkward as ak
from TROTASR.utils.trota_inference import (infer_chunk, p4_view, bounded_slices, probabilities,
    migration_counts, RESOLVED_WP, MIXED_WP, P4_FIELDS)


def sample():
    return ak.Array(dict(file_id=[3], entry=[12],
        jet_nanoaod_pt=[[60., 50., 40., 24.]], jet_nanoaod_mass=[[5., 5., 5., 5.]],
        jet_eta_all=[[0., .1, -.2, .3]], jet_phi_all=[[0., 1., 2., -1.]],
        jet_area=[[.4]*4], jet_btag_upart_all=[[.6]*4], jet_id_all=[[1]*4],
        jet_source_index_all=[[6, 8, 12, 15]],
        fatjet_nanoaod_pt=[[250.]], fatjet_nanoaod_mass=[[80.]],
        fatjet_eta_all=[[0.]], fatjet_phi_all=[[.5]], fatjet_area=[[1.]],
        fatjet_globalpart3_xbb_all=[[.2]], fatjet_globalpart3_qcd_all=[[.4]],
        fatjet_top_score_all=[[.8]], fatjet_w_score_all=[[.7]],
        fatjet_id_all=[[1]], fatjet_source_index_all=[[4]]))


def constant(discriminant):
    def model(inputs, training=False):
        return np.tile(np.array([0, discriminant, 1-discriminant], np.float32),
                       (len(next(iter(inputs.values()))), 1))
    return model


class TrotaInferenceTests(unittest.TestCase):
    def test_supported_inference_preserves_rows_and_empty_events(self):
        from TROTASR.utils.systematic_views import supported_inference
        a = ak.concatenate([sample(), sample(), sample()])
        a = ak.with_field(a, np.array([10, 11, 12]), 'entry')
        models = {'resolved': constant(.99), 'mixed': constant(.99)}
        for mask in (np.array([True, False, True]), np.zeros(3, bool), np.ones(3, bool)):
            out, resolved, audit, states = supported_inference(a, models, mask)
            self.assertEqual(len(out), 3)
            self.assertEqual(audit['inference_events'], int(mask.sum()))
            self.assertEqual(np.asarray(out.nTopMixed1pct).tolist(), np.where(mask, 4, 0).tolist())
            self.assertEqual(set(resolved), {(3, int(i)) for i in np.array([10,11,12])[mask]})
            for state in states.values():
                self.assertTrue(set(state['event']).issubset(set(np.flatnonzero(mask))))

    def test_inclusive_wp_and_original_constituent_ids(self):
        a = sample()
        out, r, audit, states = infer_chunk(a, {'resolved': constant(RESOLVED_WP), 'mixed': constant(MIXED_WP)})
        self.assertEqual(audit['resolved_candidates'], 1)
        self.assertEqual(audit['resolved_passing'], 1)
        self.assertEqual(audit['mixed_candidates'], 4)
        self.assertEqual(int(out['nTopMixed1pct'][0]), 4)
        self.assertEqual(r[(3, 12)][0]['jets'], [12, 8, 6])
        self.assertTrue(np.all(states['mixed']['sources'][:, 0] == 4))
        self.assertNotIn('nTopMixed1pct', a.fields)

    def test_failing_wp_still_evaluated(self):
        out, r, audit, states = infer_chunk(sample(), {'resolved': constant(.1), 'mixed': constant(.1)})
        self.assertEqual(audit['resolved_candidates'], 1)
        self.assertEqual(audit['mixed_candidates'], 4)
        self.assertEqual(r, {})
        self.assertEqual(int(out['nTopMixed1pct'][0]), 0)
        self.assertEqual(len(states['mixed']['passing']), 4)

    def test_new_eligible_candidates_not_limited_to_stored_collection(self):
        a = sample()
        models = {'resolved': constant(.99), 'mixed': constant(.99)}
        _, _, _, before = infer_chunk(a, models)
        p4 = {name: a[name] for name in P4_FIELDS}
        p4['jet_nanoaod_pt'] = ak.Array([[60., 50., 40., 26.]])
        changed = p4_view(a, p4)
        _, _, audit, after = infer_chunk(changed, models)
        self.assertEqual(audit['resolved_candidates'], 4)
        self.assertEqual(audit['mixed_candidates'], 10)
        self.assertEqual(migration_counts(before['resolved'], after['resolved'])['passing_gained_with_eligibility'], 3)
        self.assertEqual(migration_counts(before['mixed'], after['mixed'])['passing_gained_with_eligibility'], 6)
        self.assertEqual(float(a['jet_nanoaod_pt'][0, 3]), 24.)

    def test_wp_migration_uses_constituents_not_index(self):
        before = dict(event=np.array([0, 0]), sources=np.array([[4, 3, 2], [3, 2, 1]]), passing=[False, True])
        after = dict(event=np.array([0, 0]), sources=np.array([[4, 3, 2], [3, 2, 1]]), passing=[True, False])
        m = migration_counts(before, after)
        self.assertEqual(m['wp_fail_to_pass'], 1)
        self.assertEqual(m['wp_pass_to_fail'], 1)

    def test_candidate_budget_has_no_silent_event_drop(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            list(bounded_slices(sample(), 4))
        a = ak.concatenate([sample(), sample()])
        slices = list(bounded_slices(a, 5))
        self.assertEqual([(x.start, x.stop) for x in slices], [(0, 1), (1, 2)])

    def test_all_four_p4_and_alignment_required(self):
        with self.assertRaises(ValueError):
            p4_view(sample(), {})
        p4 = {name: sample()[name] for name in P4_FIELDS}
        p4['jet_nanoaod_mass'] = ak.Array([[1.]])
        with self.assertRaises(ValueError):
            p4_view(sample(), p4)

    def test_invalid_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            probabilities(constant(np.nan), {'jet': np.zeros((1, 3, 8))})

    def test_negative_shifted_pt_uses_existing_candidate_selection(self):
        a=sample(); models={'resolved':constant(.99),'mixed':constant(.99)}
        _,base_resolved,_,base_states=infer_chunk(a,models)
        for value in (-1e-8,-3.,-100.):
            p4={name:a[name] for name in P4_FIELDS}
            p4['jet_nanoaod_pt']=ak.Array([[60.,50.,40.,value]])
            view=p4_view(a,p4)
            out,resolved,audit,states=infer_chunk(view,models)
            self.assertEqual(len(out),1)
            self.assertEqual(float(view['jet_nanoaod_pt'][0,3]),value)
            self.assertEqual(resolved,base_resolved)
            self.assertEqual(audit['negative_pt_excluded_jet_objects'],1)
            for kind in states:
                for name in states[kind]:
                    self.assertTrue(np.array_equal(states[kind][name],base_states[kind][name]))
            p4['fatjet_nanoaod_pt']=ak.Array([[value]])
            out,resolved,audit,states=infer_chunk(p4_view(a,p4),models)
            self.assertEqual(resolved,base_resolved)
            self.assertEqual(int(out['nTopMixed1pct'][0]),0)
            self.assertEqual(audit['negative_pt_excluded_fatjet_objects'],1)

    def test_nonfinite_pt_is_not_accepted_as_negative(self):
        a=sample()
        for value in (np.nan,np.inf,-np.inf):
            p4={name:a[name] for name in P4_FIELDS}
            p4['jet_nanoaod_pt']=ak.Array([[60.,50.,40.,value]])
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                p4_view(a,p4)

    def test_undefined_mixed_input_is_candidate_local_untagged(self):
        for xbb,qcd,kind in ((0.,0.,'zero'),(-1.,0.,'negative')):
            a=sample()
            a=ak.with_field(a,ak.Array([[xbb]]),'fatjet_globalpart3_xbb_all')
            a=ak.with_field(a,ak.Array([[qcd]]),'fatjet_globalpart3_qcd_all')
            def forbidden_model(*args,**kwargs):
                self.fail('Undefined input must not be sent to the model')
            out,r,audit,states=infer_chunk(a,{'resolved':constant(.99),'mixed':forbidden_model})
            self.assertEqual(len(out),1)
            self.assertEqual(len(r[(3,12)]),1)
            self.assertEqual(int(out['nTopMixed1pct'][0]),0)
            self.assertEqual(audit['mixed_invalid_denominator_'+kind+'_candidates'],4)
            self.assertEqual(audit['mixed_invalid_denominator_'+kind+'_fatjets'],1)
            self.assertEqual(audit['mixed_invalid_denominator_'+kind+'_events'],1)
            self.assertEqual(len(states['mixed']['passing']),0)
            self.assertEqual(ak.to_list(out['fatjet_source_index_all']),[[4]])

    def test_valid_mixed_on_another_ak8_is_preserved(self):
        a=sample()
        for name in [n for n in a.fields if n.startswith('fatjet_')]:
            a=ak.with_field(a,ak.concatenate([a[name],a[name]],axis=1),name)
        a=ak.with_field(a,ak.Array([[0.,.2]]),'fatjet_globalpart3_xbb_all')
        a=ak.with_field(a,ak.Array([[0.,.4]]),'fatjet_globalpart3_qcd_all')
        a=ak.with_field(a,ak.Array([[4,9]]),'fatjet_source_index_all')
        out,_,audit,states=infer_chunk(a,{'resolved':constant(.99),'mixed':constant(.99)})
        self.assertEqual(int(out['nTopMixed1pct'][0]),4)
        self.assertEqual(audit['mixed_invalid_denominator_zero_candidates'],4)
        self.assertTrue(np.all(states['mixed']['sources'][:,0]==9))
        self.assertEqual(ak.to_list(out['TopMixed1pct_idxFatJet']),[[1]*4])


if __name__ == '__main__':
    unittest.main()
