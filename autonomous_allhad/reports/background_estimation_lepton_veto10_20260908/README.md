# Histogram-only background estimation — 2024 and 2025

Canonical input manifest: autonomous_allhad/reports/lepton_veto10_canonical_20260908.json. Selection validity: validated_10gev_inputs.
No intermediate ROOT or NanoAOD input was opened. Nominal histograms were not modified or reweighted.

High-dM: retained 73 bins. Low-dM: frozen GNN30. Nb1/Nb2plus are factor groups, not new SR categories.

## RZ results

| Year | Region | Nb = 1 | Nb ≥ 2 |
|---|---|---|---|
| 2024 | highdm | 0.7539 ± 0.0468 | 0.7719 ± 0.0805 |
| 2024 | lowdm | 0.6019 ± 0.0112 | 0.6569 ± 0.0302 |
| 2025 | highdm | 0.6502 ± 0.0453 | 0.8150 ± 0.0899 |
| 2025 | lowdm | 0.5661 ± 0.0115 | 0.5778 ± 0.0299 |

RZ errors use the existing on/off-Z profile fit and inverse-variance ee/μμ combination. Per-channel RZ–RT covariance is exported. Combined cross-group diagonal covariance is the existing downstream assumption, not a measured absent correlation.

## GNN transfer and shape propagation

- Top = TT + ST. Top/W parameter sharing is unchanged; W and QCD factors remain separately exported.
- GNN TF = SR score-bin MC / mapped CR parent MC integral. Parent score fractions and shared-denominator covariance are retained.
- Z prediction = RZ(Nb) × sum_UT H_Zinv(GNN,UT) Sgamma(Nb,UT). Q is not transferred; no additional Z-integral rescaling is applied.
- RZ/Sgamma factor categories are Nb1/Nb2plus. Final Low-dM templates stay in the frozen six categories × five GNN bins.

## Double-ratio uncertainty

The already-approved 250-GeV lower boundary and merged 500–1500 display tail are used. Only downstream_central_abs_deviation = abs(D−1) is projected into GNN bins, with up=1+delta and down=1/(1+delta). The plotted max(abs(D−1),stat) band is not substituted for the existing card convention. No nuisance names or correlations were changed.

## Coverage and caveats

| Year | Retained inputs | Source bad files | Data bad files | Complete data luminosity coverage | Zero / signed GNN TF numerator bins |
|---|---|---|---|---|---|
| 2024 | 5954 | 0 | 0 | True | 3 / 0 |
| 2025 | 8390 | 11 | 1 | False | 4 / 0 |

No smoothing, floors, fabricated missing templates, SR observations, or VR likelihood channels were introduced. RZ post plots use fitted data and are not independent closure tests; TF reconstruction and Q/Sgamma identities are mechanical checks, not physics closure.

## Reused plotting entrypoints

dy_estimation report, build_sgamma_ut_report_2024.py, build_zgamma_double_ratio_2024.py, and plot_recoil_transfer_factors_2024.py. Plot-only rendering does not change fitted factor JSONs. Cards, limits, impacts and web publication remain with the main task.
