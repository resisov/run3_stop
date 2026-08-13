# Low-pT loose-muon measurement

The target is the exact analysis loose-muon definition in the uncovered
`5 < pT < 10 GeV` interval: `looseId` and `miniPFRelIso_all < 0.2` within
`abs(eta) < 2.4`.

The official 2024 `muon_JPsi.json.gz` already covers the LooseID component down
to low pT.  This workflow measures the missing low-pT mini-isolation component
with J/psi tag-and-probe and combines it with the official LooseID term.  The
combined correction and its propagated uncertainty are exported so that the
analysis applies one unambiguous low-pT loose-muon factor.

Data pass/fail spectra are fit simultaneously with morphed J/psi MC signal
templates and independent backgrounds.  The pure resonant MC isolation
efficiency is evaluated with weighted pass/fail counting.  Statistical,
signal-template, background, fit-window, alternate-binning, and pileup
variations are combined with the official LooseID uncertainty in the final
correctionlib payload.
