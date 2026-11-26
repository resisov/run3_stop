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

Nov 25: apply the same photon GNN link to Nb≥2 for new workspaces. Each Nb
category has four independent GNN coefficients and preserves its own per-era
photon total. CR-only and T2tt (1200,500) workspace checks passed for all
60 SR recoil factors. Running fits use the Nb=1 link.

Signal production completed on 2026-11-14 at 10:31 CET. The collection receipt
is `production/collection.json` (1,122 ROOT files on hep2). The six-era
efficiency payloads use 187/187 source files and 48.58M events.
`production/histograms/state.json` records 1,122/1,122 signal-only histograms,
six era merges, 116,542,856 retained events and zero errors.

`inputs/friends/trota_nominal/` contains 17,411 nominal MC friend ROOTs,
completed on 2026-11-14 at 10:25 CET with zero errors. Registered nominal SR/CR
uses these tags and the registered signal Runs weights.

The corrected nominal UUID signals are complete: 1,122 histograms, six era
merges, and 895 fit-ready signal-only cards in `stats/run3/new_signal/grid`.
They use accepted Runs normalizations; the stored-tag merges are not their input.

Use the high-dM LLCR **without an mTb cut**. The 175 GeV LLCR study was
retired on 2026-11-19. Do not use its merged yields, transfers or cards for
production or fits. Keep the original six-era inputs and nominal signal update.
The high-dM SR threshold and VR split at 175 GeV are unchanged.
Study receipts and the before/after prefit figure remain in
hep2 `stats/run3/new_signal/llcr_mtb/` and local `output/pdf/llcr_mtb/prefit/`
for reference. No fit used the study. On hep2, ordinary SR/CR histograms now
gate heavy reads on the union of possible regions and variations; the inventory
dispatcher uses measured worker memory. See `README.md`.

Six-era inclusive ttbar top-pT (canonical Nov 21): nominal and paired Up/Down
each completed 1,627/1,627 inputs, 118,530,412 events, zero failures (Nov 20,
14:40:20/18:04:32 CET); `stats/run3/dataset_replay/top_pt/{nominal,uncertainty}.state.json`.
Central-only CR/blinded SR figures: `output/pdf/top_pt_nominal/canonical/`.
Both TT detector replays finished (1,850/1,850 each). JES/JER uses nominal
passers only; losses and bin moves are included, entrants omitted. PDF/μR/μF deferred.
Nov 25 CET: DY 1,627/1,627 10:24; cards 895/895 11:43 (208 bins, 44 CR rates).
`stats/run3/top_pt/`, Nov 26 07:05 CET: limits 882/895, 13 running; grid .0025/1e-5.
Impacts 829/833 collected; Sγ bins 2,4 done, bins 1,3 in `impact_workers/`.
Collector PID 250693 adopts both.

## VR validation

Use independent approved VR data for validation only; keep SR blind. Propagate
transfers, era corrections, full CR-fit covariance and VR MC sumw2.

High-dM VR now uses the approved 40-bin Run 2-style layout: two mTb ranges
split at 175 GeV, Nb=1 or ≥2, four exclusive single tags and a ≥2-tag group,
and MET split at 400 GeV. Retain medium Δφ; untagged would overlap low-dM SR.
Keep the existing low-dM VR bins. Resume high-dM recounts with
`histograms --vr-mc --vr-background-only --vr-workers 72` and
`vr-data --workers 12`, reusing low-dM receipts. Read current states in
`stats/run3/vr`; after both complete use
`plot vr-predictions --cr-normalization-only --highdm-only`. No new fit or
systematic production.

## Where to work

The hep2 analysis root is `/hep2-scratch/twkim/postprocessing/TROTASR`.
Use `jsons/analysis.json` for configuration and [`README.md`](README.md) for
commands, runtime paths and recovery. Per-file histogram production and merging
consume the current input inventory. Failed files and fits resume through their
own stage; retired campaign controllers and source-history fallbacks are removed.
Current scientific receipts, numerical acceptance and SR blinding remain in use.

## Fits

Prior fits use 214 bins and 48 CR rates. The new cards use 208/44 with the high-dM
CR tail pooled at ≥500 GeV per Nb. The GCR shapes high-dM Z→νν at fixed nominal
SR totals per era/Nb. DYCR RZ sets their scale. Background PDF/μR/μF use fixed
nominal events if evaluated; they are not included in this update.

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
- Current top-pT plots: local `output/pdf/top_pt_card/{background,physical,gnn,sr,cronly}/`.
  AN Figs. 106–107 use 30+30 CR bins and 161 pulls, fit status 0 / covQual 3.
  The band uses 500 correlated draws (seed 12345). Older 66-bin comparison: `output/pdf/cronly/`.
- Official impacts: local `output/pdf/impacts/impacts_T2tt_1200_500.pdf` and
  its two PNG previews of the top 60. The Ghostscript PNG exports preserve
  the native Combine labels and axes. The 26-page PDF was checked on 2026-11-01.
- Previous CR pulls: `output/pdf/cronly/refit_matched/` (156 pulls, not for the new model).

## Analysis note

Nov 25: AN updated 410 figure PDFs in GitLab/Overleaf (through `7f1db24`).
Tables 19–30 followed in `820490e`; CR signal not evaluated, SR data blinded.
243 control distributions, 162 background/TF panels, 3 blinded SRs, CR comparison and pulls.
Keep the earlier Fig. 108–109 until final 895-point limits and 833-parameter T2tt impacts
are rendered and checked; never use in-progress contours in the AN. Appendix B 2024/25
T2bW/T2tb impact benchmarks are separate reference fits.
