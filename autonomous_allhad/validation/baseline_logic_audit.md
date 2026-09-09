# Baseline Logic Audit

- AK4 jet ID: baseline calls `ids.isGoodJet`, which uses correctionlib `AK4PUPPI_TightLeptonVeto`; feature validation evaluates the same correctionlib when the NanoAOD composition branches are present, falls back to `Jet_jetId` bits if available, and labels any raw-kinematic fallback explicitly.
- Electron veto/medium IDs: baseline calls `isVetoElectron` and `isMediumElectron` from `ids.py`; feature validation mirrors the documented pt/eta/cutBased/miniIso cuts.
- Muon loose/medium IDs: baseline calls `isLooseMuon` and `isMediumMuon`; feature validation mirrors pt/eta/ID/miniIso cuts.
- Photon selection: the adopted 2024/2025 production definition is pT > 220 GeV, ECAL fiducial/gap, `cutBased >= 2` (Medium), and `electronVeto == 1`, implemented by `real_subset_worker.medium_photon_mask`. The reference processor's legacy `isMediumPhoton` helper is not the authority for this definition.
- Tau veto and isolated-track veto: feature validation mirrors the scalar cuts from `ids.py`.
- B-tag WP: baseline UParTAK4 medium threshold is `0.1272`; feature validation uses the same threshold.
- Object cleaning: baseline uses metric-table cleaning against selected photons/leptons; feature validation does not yet apply full object cleaning for cleaned-jet CRs and records this discrepancy.
- Recoil construction: feature validation computes photon recoil for GCR and dilepton kinematics for DY regions, but full vector behavior is a lightweight mirror, not the coffea processor output.
- Year behavior: validation is fixed to 2024 inputs and correction availability.

The feature-table validation is therefore not a substitute for the actual `stop_processor_v4.py` subprocess.
