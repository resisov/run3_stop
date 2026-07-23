# 2025 Data and 2024+2025 Limit Summary

Generated UTC: `2026-07-15T10:57Z`
Status: `complete; independent web page generated; public deployment pending`

## 2025 intermediate ROOT

- Full input set: 7,587 ROOT files.
- Selected events: 9,685,947.
- Zero-entry shards: 1,354; these are valid empty outputs.
- Local recovery: 5 shards total.
- Histogram-stage corrupt ROOT recovery: 3 files, 2,644 selected entries.
- Recovered ROOT validation: 171 branches, 3/3 valid.
- Permanently skipped files: 0.
- Lost selected events: 0.

## Histograms and plots

- Active high-dM signal scheme: `boosted_an17_selected_recoil6_with_nt0_wsplit_SR`.
- Search bins: 54 = 9 category blocks x 6 recoil bins.
- 2025 high-dM 54-bin data yield: 38,664.
- Plot records: 130, including 105 low-dM variable plots.
- Plot uncertainty sources: pileup, electron, muon, photon, and four b-tag nuisance sources.
- b-tag implementation: four nuisance sources with eight Up/Down shape variations.
- Existing plotting code and canvas sizes were preserved.

## Expected limits

- Combine: v10.0.1, AsymptoticLimits, blind expected.
- 2025: 352/352 valid, 0 missing, 0 invalid.
- 2025 expected-excluded grid points: 70.
- 2025 max expected-excluded mStop for mLSP <= 100 GeV: 1,350 GeV.
- 2024+2025: 352/352 valid, 0 missing, 0 invalid.
- 2024+2025 expected-excluded grid points: 87.
- 2024+2025 max expected-excluded mStop for mLSP <= 100 GeV: 1,400 GeV.
- Both limit plots use the existing 12 x 10 inch style and Run-2 SUS-19-010 observed/expected overlays.

## Paths and publication

- Run directory: `autonomous_allhad/workflow/full_run_2025_data_limit_20260715`.
- Independent page: `docs/full_run_2025_data_limit_20260715/index.html`.
- Runtime and outputs used EOS only.
- No AFS or system `/tmp` writes were made by this run.
- `decaf/analysis` was not modified by this run.
- Public GitHub Pages deployment has not yet been confirmed.
