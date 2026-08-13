# Low-pT veto-electron measurement

The target is the exact analysis veto-electron definition in the uncovered
`5 < pT < 10 GeV` interval: fiducial electron, `cutBased >= Veto`, and
`miniPFRelIso_all < 0.1`.  Following the Run-2 AN object-correction philosophy,
the efficiency is measured with resonance tag-and-probe in data and MC and all
statistical and fit-model variations are propagated.

The measurement uses `J/psi -> ee` with a tight electron tag matched to an
unrestricted single-electron `Ele8/Ele12 + PFJet30` reference path.  PFJet30 is
only the event-recording leg and is never a probe pass/fail requirement.  Data
pass/fail spectra are fit simultaneously with morphed pass/fail signal
templates from the single Summer24 `SPS-JpsiJpsiToMuMuEE` sample and independent
backgrounds.  The pure resonant MC efficiency is obtained by weighted
pass/fail counting; this is not generator-truth efficiency.

Counting starts on a 4 eta x 5 pT grid.  The reduced histograms are merged
exactly, without re-reading NanoAOD, to three eta bins spanning 5--10 GeV.  The
coarser adopted grid was selected because subdivided endcap bins produced
boundary efficiencies or model uncertainties up to O(50%).  Statistical,
signal-template, background, fit-window, alternate-binning, and pileup
variations are propagated to the correctionlib uncertainty.
