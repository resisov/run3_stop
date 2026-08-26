# Low-pT loose-muon ID-only measurement

The probe denominator is a reconstructed tracker muon with
`5 < pT < 10 GeV` and `abs(eta) < 2.4`.  It has no LooseID or isolation
requirement.  The numerator is `looseId`; mini-isolation is deliberately
excluded from the measured efficiency.

ParkingSingleMuon records the event through the Mu9/Mu10 displaced-muon OR.
The measured tag and probe are not trigger matched.  A third tight barrel muon
above 12 GeV, distinct from both J/psi legs, makes the trigger external to the
measured pair.  The same offline topology is imposed on simulation, while the
HLT itself remains data-only.  The 2025 result uses the compatible
high-statistics 2024 `SPS-JpsiJpsiTo4Mu_Fil-4Mu` reference simulation.

Use the unified CLI and full reproducibility guide in
`autonomous_allhad/lowpt_tnp/README.md`.  This directory contains only the
muon-specific configuration and compatibility entry-point wrappers.
