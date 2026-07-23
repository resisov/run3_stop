from __future__ import annotations

import unittest
from pathlib import Path

import awkward as ak
import numpy as np

from autonomous_allhad.object_corrections_2024 import (
    calibrate_jets_and_met,
    validate_shift,
)
from scripts.prepare_2024_objectcorr_campaign import (
    btag_efficiency_coverage,
    wrapper_text,
)


REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")


def event_arrays(is_data: bool) -> ak.Array:
    values = {
        "run": [380001 if is_data else 1],
        "luminosityBlock": [12],
        "event": [345],
        "Rho_fixedGridRhoFastjetAll": [20.0],
        "Jet_pt": [[100.0]], "Jet_eta": [[0.3]], "Jet_phi": [[0.4]], "Jet_mass": [[10.0]],
        "Jet_rawFactor": [[0.1]], "Jet_area": [[0.5]], "Jet_muonSubtrFactor": [[0.0]],
        "Jet_chEmEF": [[0.1]], "Jet_neEmEF": [[0.1]], "Jet_btagUParTAK4B": [[0.2]],
        "FatJet_pt": [[300.0]], "FatJet_eta": [[0.5]], "FatJet_phi": [[1.0]],
        "FatJet_mass": [[80.0]], "FatJet_rawFactor": [[0.1]], "FatJet_area": [[0.8]],
        "FatJet_msoftdrop": [[75.0]],
        "Electron_pt": [[50.0]], "Electron_eta": [[0.2]], "Electron_deltaEtaSC": [[0.01]],
        "Electron_phi": [[0.7]], "Electron_mass": [[0.000511]], "Electron_charge": [[-1]],
        "Electron_cutBased": [[3]], "Electron_miniPFRelIso_all": [[0.02]],
        "Electron_r9": [[0.95]], "Electron_seedGain": [[12]],
        "Muon_pt": [[50.0]], "Muon_eta": [[0.4]], "Muon_phi": [[-0.8]],
        "Muon_mass": [[0.105]], "Muon_charge": [[1]], "Muon_looseId": [[1]],
        "Muon_mediumId": [[1]], "Muon_miniPFRelIso_all": [[0.03]], "Muon_nTrackerLayers": [[12]],
        "Photon_pt": [[250.0]], "Photon_eta": [[0.6]], "Photon_phi": [[2.0]],
        "Photon_cutBased": [[3]], "Photon_r9": [[0.96]], "Photon_seedGain": [[12]],
        "Photon_electronVeto": [[1]], "Photon_pixelSeed": [[0]],
        "Tau_pt": [[25.0]], "Tau_eta": [[0.5]], "Tau_phi": [[-2.0]], "Tau_mass": [[1.7]],
        "Tau_dz": [[0.01]], "Tau_decayMode": [[0]], "Tau_idDeepTau2018v2p5VSe": [[1]],
        "Tau_idDeepTau2018v2p5VSmu": [[1]], "Tau_idDeepTau2018v2p5VSjet": [[5]],
        "PuppiMET_pt": [300.0], "PuppiMET_phi": [1.5],
        "PuppiMET_ptUnclusteredUp": [305.0], "PuppiMET_phiUnclusteredUp": [1.51],
        "PuppiMET_ptUnclusteredDown": [295.0], "PuppiMET_phiUnclusteredDown": [1.49],
    }
    if not is_data:
        values.update({
            "Jet_hadronFlavour": [[5]], "Jet_genJetIdx": [[0]], "FatJet_genJetAK8Idx": [[0]],
            "GenJet_pt": [[95.0]], "GenJet_eta": [[0.3]], "GenJet_phi": [[0.4]], "GenJet_mass": [[9.5]],
            "GenJetAK8_pt": [[290.0]], "GenJetAK8_eta": [[0.5]],
            "GenJetAK8_phi": [[1.0]], "GenJetAK8_mass": [[78.0]], "Tau_genPartFlav": [[5]],
        })
    return ak.Array(values)


class ObjectCorrections2024Test(unittest.TestCase):
    def test_nominal_data_and_mc_are_finite(self) -> None:
        for is_data in (True, False):
            corrected, status = calibrate_jets_and_met(event_arrays(is_data), is_data, "nominal", REPO)
            self.assertEqual(status["status"], "applied")
            for name in ("Jet_pt", "FatJet_pt", "Electron_pt", "Muon_pt", "Photon_pt", "Tau_pt", "PuppiMET_pt"):
                self.assertTrue(np.all(np.isfinite(ak.to_numpy(ak.flatten(corrected[name], axis=None)))))

    def test_object_shape_variations_execute(self) -> None:
        arrays = event_arrays(False)
        for shift in (
            "electronScaleUp", "electronSmearDown", "photonScaleUp", "photonSmearDown",
            "muonScaleUp", "muonResolutionDown", "tauEnergyScaleUp",
            "jesTotalUp", "jesRegroupedTotalDown", "jerUp", "metUnclusteredDown",
        ):
            corrected, status = calibrate_jets_and_met(arrays, False, shift, REPO)
            self.assertEqual(status["shift"], shift)
            self.assertTrue(np.isfinite(float(corrected["PuppiMET_pt"][0])))

    def test_unknown_shift_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_shift("madeUpShift")

    def test_btag_coverage_uses_physical_dataset_key(self) -> None:
        records = [
            {"dataset": "A____1_", "is_background": True},
            {"dataset": "A____2_", "is_background": True},
            {"dataset": "B____1_", "is_background": True},
        ]
        coverage = btag_efficiency_coverage(records, {"upart_processes": ["A", "B"]})
        self.assertEqual(coverage["status"], "complete")
        self.assertEqual(coverage["matched_unique_physical_datasets"], 2)
        self.assertEqual(coverage["missing_records"], 0)

    def test_condor_wrapper_has_no_afs_or_literal_tmp_path(self) -> None:
        wrapper = wrapper_text("x509up_u123")
        self.assertNotIn("/afs/", wrapper)
        self.assertNotIn("/tmp/", wrapper)
        self.assertIn("_CONDOR_SCRATCH_DIR", wrapper)
        self.assertIn("xrdcp", wrapper)


if __name__ == "__main__":
    unittest.main()
