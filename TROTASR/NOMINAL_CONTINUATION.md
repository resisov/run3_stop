# Current TROTASR results

Use the frozen 2026-10-31 baseline and its delivered figures for review. The
updated limit calculation and provisional CR-only result are listed below.

## Where to work

The hep2 analysis root is `/hep2-scratch/twkim/postprocessing/TROTASR`.
Use `jsons/analysis.json` for configuration and [`README.md`](README.md) for
commands and recovery. `WORKSPACE_LAYOUT.md` lists workspace and runtime paths.

## Fits

The six-era partial-systematics model combines 2022, 2022EE, 2023, 2023BPix,
2024 and 2025. It merges 48 channels into 8, 1,284 bins into 214, and 288 free
CR normalizations into 48. The adopted eraScale formulas, era templates, GNN
boundaries, nuisance functions and histogram rates of −1 apply. SR observations
remain blinded. The frozen early-era normalization is 61.8971 fb⁻¹ with LUM
revision 39 and independent 5% for 2025. TTWH and TTZH are excluded, and the
available TTTT belongs to Top. See `AGENTS.md` for the fit exclusions.

| Result | hep2 path | Recorded status |
| --- | --- | --- |
| Official expected limits | `stats/references/era_merged_recoil/limits/workflow_state.json` | **895/895 accepted**. T2tt 205, T2bW 345, T2tb 345. The other 60 points remain excluded. |
| Updated expected limits | `stats/run3/limits/workflow_state.json` | **892/895 complete**, `needs_attention`, 0 active on 2026-11-15. Use the official figure delivery for review. |
| Official impacts | `stats/references/era_merged_recoil/impacts/` | T2tt (1200, 500), S+B Asimov at μ = 1, r ∈ [0, 20]. All 753 profiles passed. The receipt is `publication.json` and the collection is `collection/impacts_mStop1200_mLSP500.json`. |
| Provisional CR-only | `stats/run3/cronly/publication.json` | Observed CR data, B-only at r = 0, with no Asimov data or SR observations. The receipt selects `current_workspace.root`, `current_fit.root`, `current_pulls.json` and `current_prediction.json`. Saved `fit_b` has status 0, HESSE 0, covQual 3 and zero invalid NLL. Combine exited with SIGABRT (−6) after saving. |

The provisional CR-only fit is the 2026-11-09 low-dM Photon CR Nb=1 GNN shape
result. It uses four relative GNN coefficients and preserves each era's
photon total and the 2025 gamma+jets CSEV total treatment. The earlier
approved CR-only result remains a reference. Its recorded covQual is 2,
with `global_covariance_accepted=false`, 156/156 constrained profiles and
306 total parameters.

## Figures

- Official limits: local `output/pdf/run3_limits_current_20261023/` contains
  the T2tt, T2bW and T2tb PNG/PDF pairs and `delivery_manifest.json`. The
  delivery covers 895/895 points as of 2026-10-30 13:11:55 CET with partial
  systematics. Updated plots are under hep2 `stats/run3/plots/limits/`.
- CR comparison: `cronly_prefit_postfit.png` and `.pdf` in hep2
  `stats/run3/plots/cronly/` or local `output/pdf/cronly/`. See
  `prediction_payload.json` and `plot_manifest.json`. The 66-bin comparison
  puts 36 high-dM bins on the left and 30 low-dM bins on the right, with
  prefit above postfit. Both show bands and Data/MC. Prefit uses 242 Gaussian
  and 16 Poisson priors. Postfit uses the saved correlated covariance with
  500 shared draws and seed 12345.
- Official impacts: local `output/pdf/impacts/impacts_T2tt_1200_500.pdf` and
  the two PNG previews of the top 60. The delivery manifest records the
  26-page PDF.
- Earlier approved CR pulls: local
  `output/pdf/era_merged_diagnostics_20261029/cronly/` contains
  `cronly_nuisance_pulls_ylim6_split2.pdf` and its two PNGs (78 pulls per page).

Local `output/pdf/` also holds the delivered six-era distributions, systematic
plots, reference diagrams and GNN figures.

## Analysis note

At GitLab/Overleaf revision `410a466`, AN §11.4 places the cropped
prefit/postfit figure before the nuisance pulls. The PDF placement was verified.
