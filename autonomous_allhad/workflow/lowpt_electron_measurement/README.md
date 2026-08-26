# Low-pT veto-electron ID-only measurement

The probe denominator is a fiducial reconstructed GSF electron with
`5 < pT < 10 GeV`, conversion veto, and at most one lost hit.  It has no
cut-based ID or isolation requirement.  The numerator is `cutBased >= Veto`;
mini-isolation is deliberately excluded from the measured efficiency.

ParkingSingleMuon records the event independently of the dielectron pair via
the Mu9/Mu10 displaced-muon OR.  Neither electron leg is trigger matched.  The
tag is a tight electron above 5 GeV with mini-isolation below 0.1.  The 2025
result combines 2025 data with the compatible high-statistics 2024
`SPS-JpsiJpsiToMuMuEE` reference simulation and records that provenance.

Use the unified CLI and full reproducibility guide in
`autonomous_allhad/lowpt_tnp/README.md`.  This directory contains only the
electron-specific configuration and compatibility entry-point wrappers.
