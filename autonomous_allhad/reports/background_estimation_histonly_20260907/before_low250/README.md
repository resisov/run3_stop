# Histogram-only background estimation — 2024 and 2025

Inputs: canonical DY window 71 < mll < 111 GeV, promoted 2026-09-07. No intermediate ROOT or NanoAOD input was opened. Nominal histograms were not modified or reweighted.

High-dM final template: retained 73 bins. Low-dM final template: frozen GNN30; Nb1/Nb2plus are factor-measurement groups, not new SR categories.

## RZ results

| Year | Region | Nb = 1 | Nb ≥ 2 |
|---|---|---|---|
| 2024 | highdm | 0.7501 ± 0.0469 | 0.7643 ± 0.0804 |
| 2024 | lowdm | 0.6023 ± 0.0114 | 0.6587 ± 0.0306 |
| 2025 | highdm | 0.6434 ± 0.0452 | 0.7836 ± 0.0884 |
| 2025 | lowdm | 0.5635 ± 0.0115 | 0.5836 ± 0.0304 |

Errors above use the existing on/off-Z profile fit (Poisson data, weighted-MC template constraints) and inverse-variance ee/μμ combination. Per-channel RZ–RT covariance is exported; the combined cross-group diagonal covariance retains the existing downstream assumption, not a new measurement of absent cross-region correlations.

## GNN transfer and shape propagation

`lowdm_gnn_backgrounds.json` contains the frozen 6×5 SR bins, 4 CR parent categories, all available weight variations, parent-integrated TF coefficients, and Zinv joint components. `tf_inputs.json` additionally retains the selected background/data-CR joint histograms. No SR data are exported.

- Top = TT + ST. W and QCD are separate measured processes; existing card parameter sharing is not changed.
- GNN coefficient: SR category score-bin yield divided by its existing CR parent's target-process integral. CR score fractions preserve the five-bin CR shape. Equal SR/CR score indices are never divided because their physical edges differ.
- Z prediction: RZ(Nb) × ΣUT H_Zinv(GNN,UT) Sgamma(Nb,UT). Q is not propagated. Sgamma is shape-normalized in the GCR; no extra normalization of the Z template is imposed.
- The shared CR denominator induces covariance across GNN bins and SR children of the same parent. Within-category covariance and the shared denominator total/variance are exported; parent coefficients are not independent measurements.
- U_T TF plots remain Nb-only diagnostics, not Low-dM final templates. GNN TF plots are in each year's `tf/gnn/` directory.

## Double-ratio domain decision (not yet adopted)

The adopted 300-start definition and proposed 250-start definition are both recomputed from the same new inputs. Proposal plots and JSON are isolated in `zgamma_250_proposal/`. `double_ratio_domain_comparison.json` records CR support, normalization changes, every common bin, and per-UT-bin GNN response changes.

| Year | Max common-bin absolute ΔD | Max GNN fractional-response change |
|---|---|---|
| 2024 | 0.004034 | 0.1909% |
| 2025 | 0.012977 | 0.5817% |

Downstream field: `downstream_central_abs_deviation = abs(D−1)`. The historical figure's `systematic = max(abs(D−1), stat)` is a reporting quantity and is not substituted for the current card nuisance. The 300-start result does not constrain 250–300; missing support must not be interpreted as zero uncertainty. No nuisance or card changes were made.

## Limits and validation

- 2025 retained inputs are complete as a retained set (8390/8390), but upstream skips include one data file: luminosity coverage is not complete. Existing normalization bookkeeping was retained.
- Zero QCD GNN numerator bins remain zero and are listed in `campaign_state.json`; no smoothing, floors, or invented events were introduced.
- RZ mll post plots use the fitted data and are not independent closure. Q/Sgamma normalization identities and TF reconstruction checks are mechanical identities, not physics closure.
- High-dM source histograms contain 79 original bins; the main agent must continue its established 1–6 exclusion for the final 73. Legacy Low-dM34 is not a final template here.
- The double-ratio domain change remains a proposal until adopted. Background products are not automatically injected into cards or published to the web.

## Existing plotting entrypoints

`python -m autonomous_allhad.dy_estimation report`, `build_sgamma_ut_report_2024.py`, `build_zgamma_double_ratio_2024.py`, and `plot_recoil_transfer_factors_2024.py` were reused for both years. No independent replacement plotting implementation was created.
