"""Early-era dispatch on the existing JME/non-JME implementation."""
import math
import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from TROTASR.utils.paths import ROOT, EARLY_ERAS, tagging_definition
from TROTASR.utils.io import read_json
from TROTASR.utils import jme_calibration as jme
from TROTASR.utils import trota_candidates as candidates
from TROTASR.utils import trota_inference as inference
from TROTASR.utils.corrections import _nonjme_cset
from TROTASR.tests.test_trota_inference import sample, constant


def early_sample():
    a = sample()
    names = {'jet_btag_pnet_all': 'jet_btag_upart_all',
             'jet_btag_deepflav_all': 'jet_btag_upart_all',
             'jet_nanoaod_id': 'jet_id_all', 'fatjet_nanoaod_id': 'fatjet_id_all',
             'fatjet_btagdeepb_all': 'fatjet_globalpart3_xbb_all',
             'fatjet_pnetwmass_qcd_all': 'fatjet_globalpart3_qcd_all',
             'fatjet_pnet_qcd_all': 'fatjet_globalpart3_qcd_all',
             'fatjet_pnet_xbbvsqcd_all': 'fatjet_globalpart3_xbb_all',
             'fatjet_pnet_xqqvsqcd_all': 'fatjet_globalpart3_xbb_all'}
    for target, source in names.items():
        a = ak.with_field(a, a[source], target)
    return a


class EarlySystematicShapeTests(unittest.TestCase):
    def test_each_era_payload_contract(self):
        jme_specs = read_json(ROOT/'scales/JME/manifest.json')['years']
        obj_specs = read_json(ROOT/'scales/object_shapes.json')['years']
        for era in EARLY_ERAS:
            with self.subTest(era=era):
                self.assertEqual(jme.TAGS[era], (jme_specs[era]['jec_tag'], jme_specs[era]['jer_tag']))
                self.assertEqual(set(obj_specs[era]), {'electron', 'photon', 'muon', 'tau'})
                paths = [v['path'] for v in obj_specs[era].values()]
                paths += [jme_specs[era][k]['path'] for k in ('Jet', 'FatJet')]
                if not all((ROOT/path).is_file() for path in paths):
                    self.skipTest('Early-era official shape payloads not installed locally: '+era)
                for collection in ('Jet', 'FatJet'):
                    payload = jme.correction_set(era, collection)
                    radius = 'AK4PFPuppi' if collection == 'Jet' else 'AK8PFPuppi'
                    jec, jer = jme.TAGS[era]
                    self.assertIn(jec+'_MC_L1L2L3Res_'+radius, payload.compound)
                    self.assertIn(jer+'_MC_ScaleFactor_'+radius, payload)
                for flavor in ('electron', 'photon', 'muon', 'tau'):
                    self.assertGreater(len(_nonjme_cset(era, flavor)), 0)

    def test_early_jer_uses_angular_match_and_zero_floor(self):
        def smear(gen_eta, sf=1.2, gen_pt=95.):
            args = [ak.Array([[x]]) for x in (100., 0., 0., .1, sf, 0, gen_pt, gen_eta, 0.)]
            return float(jme._early_jer_smear(*args, [1], [2], [3], 'Jet', .4)[0,0])
        self.assertAlmostEqual(smear(.1), 1.01)
        with patch.object(jme, '_deterministic_normal', return_value=2.):
            self.assertAlmostEqual(smear(.3), 1+2*.1*math.sqrt(1.2**2-1))
        self.assertEqual(smear(.1, sf=20., gen_pt=120.), 0.)

    def test_early_wp_and_feature_dimensions(self):
        a = early_sample()
        for era in EARLY_ERAS:
            definition = tagging_definition(era)
            widths = []
            def mixed(inputs, training=False):
                widths.append(inputs['fatjet'].shape[1])
                return constant(definition['mixed_wp'])(inputs)
            models = {'_year': era, 'resolved': constant(definition['resolved_wp']), 'mixed': mixed}
            out, resolved, audit, states = inference.infer_chunk(a, models)
            self.assertEqual(widths, [12])
            self.assertEqual(audit['resolved_passing'], 1)
            self.assertEqual(int(out['nTopMixed1pct'][0]), 4)
            _, _, audit, _ = inference.infer_chunk(a, dict(models, resolved=constant(.95), mixed=constant(.92)))
            self.assertEqual(audit['resolved_passing'], 0)
            self.assertEqual(audit['mixed_passing'], 0)

    def test_early_mixed_has_no_late_xbb_ratio(self):
        a = early_sample()
        a = ak.with_field(a, ak.Array([[0.]]), 'fatjet_btagdeepb_all')
        a = ak.with_field(a, ak.Array([[0.]]), 'fatjet_pnetwmass_qcd_all')
        _, _, audit, _ = inference.infer_chunk(a, {'_year':'2022EE',
            'resolved':constant(.99), 'mixed':constant(.99)})
        self.assertEqual(audit['mixed_passing'], 4)
        self.assertEqual(audit['mixed_invalid_denominator_zero_candidates'], 0)

    def test_early_uses_stored_nanoaod_ids(self):
        a = ak.with_field(early_sample(), ak.Array([[0,0,0,0]]), 'jet_nanoaod_id')
        _, _, audit, _ = inference.infer_chunk(a, {'_year':'2023BPix',
            'resolved':constant(.99), 'mixed':constant(.99)})
        self.assertEqual(audit['resolved_candidates'], 0)
        self.assertEqual(audit['mixed_candidates'], 0)
        self.assertEqual(candidates.candidate_cost(a, '2023BPix').tolist(), [0])


if __name__ == '__main__':
    unittest.main()
