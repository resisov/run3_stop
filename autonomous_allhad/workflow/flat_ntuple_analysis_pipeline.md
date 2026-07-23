# Flat Ntuple Analysis Pipeline

Canonical pipeline after the flat preselection ntuple step.

1. Produce intermediate flat ROOT ntuples
   - Apply x-axis / event-selection-changing corrections before the skim.
   - Nominal AK4 JEC and AK8 FJEC are required for nominal ntuples.
   - Keep event-by-event quantities and region flags, not pre-binned histograms.
   - Store raw `gen_weight`; do not apply luminosity, xsec, or scale-factor normalization in the ROOT ntuple.

2. Apply remaining scale factors after the skim
   - Pileup, btag, lepton/photon ID/HLT, top-pT, and related y-axis weights are evaluated downstream.
   - Background MC normalization uses campaign-level `xsec_pb * lumi_pb / physical_dataset_sumw`.
   - Signal normalization uses mass-point `Runs.genEventSumw_T2tt_<mStop>_<mLSP>` denominators.

3. Extract uncertainties
   - Derive SF uncertainty variations from post-skim weight recalculation.
   - Treat event-selection/x-axis variations as separately produced shifted ntuples when needed.
   - Keep nominal MET for nominal skim; MET unclustered shifts are uncertainty variations, not nominal selection inputs.

4. Split analysis regions
   - Use stored flags and event quantities to define SR/CR regions after the flat ntuple stage.
   - Boosted SR studies should use `feature_SR && nboosted_top >= 1` or the stored `feature_SR_Nt1` flag.
   - Recoil_pt binning can be changed downstream without rerunning NanoAOD event extraction.

5. Plotting
   - Build histograms from flat ntuples after region selection and weight application.
   - Produce validation/control plots before datacard production.

6. Datacard and template preparation
   - Convert weighted histograms into combine templates.
   - Preserve normalization provenance and uncertainty definitions in sidecar manifests.

7. Limits
   - Run expected/observed limits from the prepared datacards and templates.
   - Compare against previous binnings and Run-2 reference contours where relevant.

8. Impacts
   - Run impact fits for selected representative mass points after limits are stable.
   - Use impacts to diagnose dominant nuisance parameters before freezing the binning strategy.
