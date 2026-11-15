# Current TROTASR results

Use the frozen 2026-10-31 Run 3 baseline and its delivered figures for review.
The 2026-11-09 provisional CR-only result uses the saved low-dM Photon CR
Nb=1 GNN shape fit. The updated limit calculation is listed separately below.

## Limit and impact update

On 2026-11-10 the user approved projecting the four low-dM Photon CR Nb=1
GNN coefficients onto the five recoil bins. Use the accepted prefit
gamma+jets weights separately in each era, preserve its total, and multiply
the existing Z→νν transfer by the resulting recoil factors. Keep the CR
model and the 2025 gamma+jets CSEV treatment; do not import the CR best-fit
values. Remove signal shape, normalization and MC-statistical uncertainties,
not their background counterparts. Keep nominal signal templates and SR blinding.
SR autoMCStats grouping uses background only, with threshold 10.

The T2tt (1200, 500) workspace reproduced the accepted CR predictions in
the tested variations. Both likelihoods have 30 projected SR coefficients;
the signal shape and yield have no nuisance dependence.

Signal production completed on 2026-11-14 at 10:31 CET. The collection receipt
is `production/collection.json` (1,122 ROOT files on hep2). The six-era
efficiency payloads use 187/187 source files and 48.58M events.
`production/histograms/state.json` records 1,122/1,122 signal-only histograms,
six era merges, 116,542,856 retained events and zero errors.

`inputs/friends/trota_nominal/` contains 17,411 nominal MC friend ROOTs,
completed on 2026-11-14 at 10:25 CET with zero errors. Registered nominal SR/CR
uses these tags and the registered signal Runs weights.

## VR validation

Test the CR-constrained prediction in the approved VRs using independent
observed VR data. Keep the SR blinded and use the saved CR-only fit without
fitting or tuning to VR data. Propagate the approved transfers, era corrections,
full fit covariance, adopted uncertainties and VR MC sumw2. Use VR MC errors
for the VR prediction. Compare yields and shapes by category and bin, record
actual fit quality and quantify signal contamination separately. Report MC
coverage, systematic response, CR-to-VR propagation and data comparison.

## Where to work

The hep2 analysis root is `/hep2-scratch/twkim/postprocessing/TROTASR`.
Use `jsons/analysis.json` for configuration and [`README.md`](README.md) for
commands, runtime paths and recovery. Per-file histogram production and merging
consume the current input inventory. Failed files and fits resume through their
own stage; retired campaign controllers and source-history fallbacks are removed.
Current scientific receipts, numerical acceptance and SR blinding remain in use.

## Fits

The six-era partial-systematics model merges 48 channels into 8, 1,284 bins
into 214, and 288 free CR normalizations into 48. `AGENTS.md` records the adopted
blinding, normalization, systematic scope and fit exclusions.

| Result | hep2 path | Recorded status |
| --- | --- | --- |
| Official expected limits | `stats/references/era_merged_recoil/limits/workflow_state.json` | **895/895 accepted**. T2tt 205, T2bW 345, T2tb 345. The other 60 points remain excluded. |
| Updated expected limits | `stats/run3/limits/workflow_state.json` | **892/895 complete**, `needs_attention`, 0 active on 2026-11-15. Use the official figure delivery for review. |
| Updated cards and templates | `stats/run3/grid/manifest.json` | Grid manifest and inputs for the updated fits. |
| Official impacts | `stats/references/era_merged_recoil/impacts/` | T2tt (1200, 500), S+B Asimov at μ = 1, r ∈ [0, 20]. All 753 profiles passed. |
| Provisional CR-only | `stats/run3/cronly/publication.json` | Observed-CR B-only at r = 0. The receipt selects `current_workspace.root`, `current_fit.root`, `current_pulls.json` and `current_prediction.json`. Saved `fit_b` has status 0, HESSE 0, covQual 3 and zero invalid NLL. Combine exited with SIGABRT (−6) after saving. |

The provisional CR-only fit uses four relative GNN coefficients in low-dM
Photon CR Nb=1. It preserves each era's photon total and the 2025 gamma+jets
CSEV total treatment. The earlier approved CR-only result remains a reference.

The CR-only reference tests used `--forceRecreateNLL` to avoid the Combine
v10.5.1 RooHistFunc overflow. In `stats/run3/cronly/`, the strategy 1 result
`fitDiagnostics_cronly_refit_fixed.root` exited 0 with covQual 2. The matched
strategy 0 result `fitDiagnostics_cronly_refit_matched.root` used the original
initial values and exited 0 with covQual 3, MINIMIZE/HESSE 0 and zero invalid NLL.
Its command is recorded in `commands/cronly_refit_matched.json`.

## Figures

- Official limits: local `output/pdf/run3_limits_current_20261023/` contains
  the T2tt, T2bW and T2tb PNG/PDF pairs and `delivery_manifest.json`. The
  delivery covers 895/895 points as of 2026-10-30 13:11:55 CET with partial
  systematics. Updated plots are under hep2 `stats/run3/plots/limits/`.
- CR comparison: `cronly_prefit_postfit.png` or `.pdf` in hep2
  `stats/run3/plots/cronly/` or local `output/pdf/cronly/`. See
  `prediction_payload.json` and `plot_manifest.json`. The 66-bin comparison
  puts 36 high-dM bins left, 30 low-dM bins right, and prefit above postfit.
  Both show bands and Data/MC. Prefit uses 242 Gaussian and 16 Poisson priors.
  Postfit uses the saved correlated covariance with 500 shared draws and seed
  12345. The delivered PDF was checked on 2026-11-09.
- Official impacts: local `output/pdf/impacts/impacts_T2tt_1200_500.pdf` and
  its two PNG previews of the top 60. The Ghostscript PNG exports preserve
  the native Combine labels and axes. The 26-page PDF was checked on 2026-11-01.
- CR pulls: local `output/pdf/cronly/refit_matched/` contains
  `cronly_nuisance_pulls_ylim6_split2.pdf` and two PNGs (78 pulls per page).
  Errors are from the normally terminated covQual 3 HESSE result. The band
  is ±1σ prefit. The earlier approved CR figures are in
  `output/pdf/era_merged_diagnostics_20261029/cronly/`.

Local `output/pdf/` also holds the delivered six-era distributions, systematic
plots, reference diagrams and GNN figures.

## Analysis note

At GitLab/Overleaf revision `410a466`, AN §11.4 places the cropped
prefit/postfit figure before the nuisance pulls. The PDF placement was
verified. The native impact PDF and PNGs were delivered on 2026-11-01.
The final AN commit receipt is pending.
