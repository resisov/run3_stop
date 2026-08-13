# Photon trigger measurement

This directory contains only configuration and machine outputs.  All code is
in `../measure_trigger.py`.  The measurement follows AN2019-016 section 5.1.4.
It is a reference-trigger
measurement, not a lepton-style tag-and-probe measurement.  The denominator is
selected by independent JetMET/PFHT paths and the numerator additionally passes
the analysis `HLT_Photon175 || HLT_Photon200` OR.

The baseline-like photon control selection is evaluated without the lepton veto
and MET threshold, then requires the analysis veto-electron veto, loose-muon
veto, and a medium-ID photon.  Efficiencies and SFs are measured versus photon
`pT` and `abs(eta)`.  The unified `export` command refuses to install any result
that is not explicitly marked `adopted`.
