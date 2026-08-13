import unittest

import awkward as ak

from workflow.tnp_histograms import (
    _external_reference_pair_mask,
    _invariant_mass,
    _objects,
)


class TnpHistogramTest(unittest.TestCase):
    def test_muon_probe_is_conditional_on_loose_id(self):
        arrays = ak.Array({
            "Muon_pt": [[7.0, 6.0]],
            "Muon_eta": [[0.1, -0.1]],
            "Muon_phi": [[0.0, 0.45]],
            "Muon_mass": [[0.105, 0.105]],
            "Muon_charge": [[1, -1]],
            "Muon_miniPFRelIso_all": [[0.05, 0.25]],
            "Muon_looseId": [[True, True]],
            "Muon_tightId": [[True, False]],
            "TrigObj_eta": [[0.1]],
            "TrigObj_phi": [[0.0]],
            "TrigObj_id": [[13]],
            "TrigObj_filterBits": [[8]],
        })
        tags, probes, _ = _objects(arrays, "muon", 8)
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [1])
        self.assertEqual(ak.to_list(ak.num(probes, axis=1)), [2])
        self.assertEqual(ak.to_list(probes.passing), [[True, False]])

    def test_electron_target_is_veto_id_plus_miniiso(self):
        arrays = ak.Array({
            "Electron_pt": [[7.0, 6.0]],
            "Electron_eta": [[0.1, -0.1]],
            "Electron_phi": [[0.0, 0.45]],
            "Electron_mass": [[0.0005, 0.0005]],
            "Electron_charge": [[1, -1]],
            "Electron_miniPFRelIso_all": [[0.05, 0.15]],
            "Electron_cutBased": [[4, 1]],
            "Electron_convVeto": [[True, True]],
            "Electron_lostHits": [[0, 0]],
            "TrigObj_eta": [[0.1]],
            "TrigObj_phi": [[0.0]],
            "TrigObj_id": [[11]],
            "TrigObj_filterBits": [[2]],
        })
        tags, probes, _ = _objects(arrays, "electron", 2)
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [1])
        self.assertEqual(ak.to_list(probes.passing), [[True, False]])

    def test_electron_id_only_does_not_condition_probe_on_isolation(self):
        arrays = ak.Array({
            "Electron_pt": [[7.0, 6.0]],
            "Electron_eta": [[0.1, -0.1]],
            "Electron_phi": [[0.0, 0.45]],
            "Electron_mass": [[0.0005, 0.0005]],
            "Electron_charge": [[1, -1]],
            "Electron_miniPFRelIso_all": [[0.05, 0.50]],
            "Electron_cutBased": [[4, 1]],
            "Electron_convVeto": [[True, True]],
            "Electron_lostHits": [[0, 0]],
            "TrigObj_eta": [[0.1]],
            "TrigObj_phi": [[0.0]],
            "TrigObj_id": [[11]],
            "TrigObj_filterBits": [[2]],
        })
        tags, probes, _ = _objects(arrays, "electron", 2, "veto_id_only")
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [1])
        self.assertEqual(ak.to_list(probes.passing), [[True, True]])

    def test_muon_id_only_uses_tracker_muon_denominator(self):
        arrays = ak.Array({
            "Muon_pt": [[7.0, 6.0, 6.5]],
            "Muon_eta": [[0.1, -0.1, 0.2]],
            "Muon_phi": [[0.0, 0.45, 1.2]],
            "Muon_mass": [[0.105, 0.105, 0.105]],
            "Muon_charge": [[1, -1, -1]],
            "Muon_miniPFRelIso_all": [[0.05, 0.90, 0.05]],
            "Muon_isTracker": [[True, True, False]],
            "Muon_looseId": [[True, False, True]],
            "Muon_tightId": [[True, False, False]],
            "TrigObj_eta": [[0.1]],
            "TrigObj_phi": [[0.0]],
            "TrigObj_id": [[13]],
            "TrigObj_filterBits": [[8]],
        })
        tags, probes, _ = _objects(arrays, "muon", 8, "loose_id_only")
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [1])
        self.assertEqual(ak.to_list(ak.num(probes, axis=1)), [2])
        self.assertEqual(ak.to_list(probes.passing), [[True, False]])

    def test_tag_plateau_threshold_is_enforced(self):
        arrays = ak.Array({
            "Muon_pt": [[9.5, 6.0]],
            "Muon_eta": [[0.1, -0.1]],
            "Muon_phi": [[0.0, 0.45]],
            "Muon_mass": [[0.105, 0.105]],
            "Muon_charge": [[1, -1]],
            "Muon_miniPFRelIso_all": [[0.05, 0.90]],
            "Muon_isTracker": [[True, True]],
            "Muon_looseId": [[True, False]],
            "Muon_tightId": [[True, False]],
            "TrigObj_eta": [[0.1]],
            "TrigObj_phi": [[0.0]],
            "TrigObj_id": [[13]],
            "TrigObj_filterBits": [[8]],
        })
        tags, _, _ = _objects(arrays, "muon", 8, "loose_id_only", 10.0)
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [0])

    def test_external_muon_reference_does_not_trigger_match_electron_tag(self):
        arrays = ak.Array({
            "Electron_pt": [[12.0, 6.0, 14.0]],
            "Electron_eta": [[0.1, -0.1, 0.2]],
            "Electron_phi": [[0.0, 0.45, 1.2]],
            "Electron_mass": [[0.0005, 0.0005, 0.0005]],
            "Electron_charge": [[1, -1, -1]],
            "Electron_miniPFRelIso_all": [[0.05, 0.90, 0.20]],
            "Electron_cutBased": [[4, 0, 1]],
            "Electron_convVeto": [[True, True, True]],
            "Electron_lostHits": [[0, 0, 0]],
        })
        tags, probes, _ = _objects(
            arrays,
            "electron",
            None,
            "veto_id_only",
            5.0,
            5.0,
            15.0,
            False,
        )
        self.assertEqual(ak.to_list(ak.num(tags, axis=1)), [1])
        self.assertEqual(ak.to_list(ak.num(probes, axis=1)), [3])
        self.assertEqual(ak.to_list(probes.passing), [[True, False, True]])

    def test_parking_reference_muon_is_distinct_from_tnp_pair(self):
        arrays = ak.Array({
            "Muon_pt": [[12.5, 7.0, 6.0], [7.0, 6.0]],
            "Muon_eta": [[0.3, 0.1, -0.1], [0.1, -0.1]],
            "Muon_phi": [[2.0, 0.0, 0.45], [0.0, 0.45]],
            "Muon_mass": [[0.105, 0.105, 0.105], [0.105, 0.105]],
            "Muon_charge": [[1, 1, -1], [1, -1]],
            "Muon_miniPFRelIso_all": [[0.5, 0.2, 0.9], [0.2, 0.9]],
            "Muon_isTracker": [[True, True, True], [True, True]],
            "Muon_looseId": [[True, True, False], [True, False]],
            "Muon_tightId": [[True, True, False], [True, False]],
        })
        tags, probes, _ = _objects(
            arrays,
            "muon",
            None,
            "loose_id_only",
            5.0,
            5.0,
            10.0,
            False,
            None,
        )
        pairs = ak.cartesian({"tag": tags, "probe": probes}, axis=1)
        mask = _external_reference_pair_mask(
            arrays,
            pairs,
            {
                "external_reference_muon": {
                    "enabled": True,
                    "pt_min_gev": 12.0,
                    "abseta_max": 1.5,
                    "require_tight_id": True,
                    "miniiso_max": None,
                }
            },
        )
        selected = pairs[mask]
        self.assertGreater(len(selected[0]), 0)
        self.assertEqual(len(selected[1]), 0)


if __name__ == "__main__":
    unittest.main()
