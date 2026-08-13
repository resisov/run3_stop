# Failure of the 2024 Lost-Lepton Transfer-Factor Closure

## Analysis decision

**The current MC-derived LLCR-to-zero-lepton transfer-factor method fails its data validation and must be retired from the nominal lost-lepton background estimation.**

This decision applies to both implementations tested in this study:

1. a single transfer factor derived from the combined \(t\bar t+\mathrm{W+jets}+\mathrm{single\ top}\) sample; and
2. a two-component implementation in which Top (\(t\bar t+\mathrm{single\ top}\)) and W+jets are normalized and transferred independently.

The failure is not a small residual effect that can be covered by the statistical uncertainty of the method. The discrepancies are large, region dependent, and in several low-\(\Delta m\) validation regions the application of the transfer factor makes an initially satisfactory raw-MC prediction substantially worse.

The failed transfer-factor implementation will be retained only as an archived diagnostic study. It must not be used to define the nominal lost-lepton central value, to build nominal datacards, or to claim a validated data-driven background prediction.

## Scope

This report documents the pre-fit 2024 lost-lepton closure study. No post-fit normalization or signal-region data constraint is used.

- Input ROOT files: 1,153
- Events scanned: 361,054,245
- Events retained for the closure and validation selections: 34,735,958
- Lost-lepton target processes: \(t\bar t\), W+jets, and single-top
- Data sample: JetMET
- Subtracted non-target backgrounds: \(Z\to\nu\nu\), DY, \(\gamma+\)jets, diboson, and QCD multijet
- Selection authority: `real_subset_worker.py`
- Nominal intermediate products modified: no
- Event-level A/B-fold duplicates: zero

The quoted uncertainties and hypothesis tests include statistical covariance only. Detector and modeling systematic uncertainties have not been added. This limitation does not rescue the method: the observed failures are coherent, large, and accompanied by direct evidence that the LLCR-derived corrections are not transferable to the zero-lepton validation regions.

## Method under test

### Combined transfer factor

The original estimate uses

\[
T_i^{\mathrm{combined}}
=
\frac{
N_{i,\mathrm{MC}}^{0\ell,\,
t\bar t+\mathrm{W+jets}+\mathrm{ST}}
}{
N_{i,\mathrm{MC}}^{1\ell,\,
t\bar t+\mathrm{W+jets}+\mathrm{ST}}
}
\]

and predicts the lost-lepton contribution from the background-subtracted one-lepton data:

\[
N_{i,\mathrm{pred}}^{\mathrm{LL}}
=
T_i^{\mathrm{combined}}
\left(
N_{i,\mathrm{data}}^{1\ell}
-
N_{i,\mathrm{other\,MC}}^{1\ell}
\right).
\]

The closure observable is the ratio of this prediction to the lost-lepton residual in a zero-lepton validation region:

\[
R_i
=
\frac{
N_{i,\mathrm{pred}}^{\mathrm{LL}}
}{
N_{i,\mathrm{data}}^{0\ell}
-
N_{i,\mathrm{other\,MC}}^{0\ell}
}.
\]

A valid method requires \(R_i\) to be compatible with unity in independent validation regions.

### Top/W-split transfer factor

A second implementation was constructed to test whether the failure arose from combining Top and W+jets into a single transfer factor. Independent one-lepton normalizations were extracted from W-enriched and Top-enriched control categories:

\[
N_{i,\mathrm{pred}}^{\mathrm{LL}}
=
\mu_{\mathrm{Top}}N_{i,\mathrm{MC}}^{0\ell,\mathrm{Top}}
+
\mu_WN_{i,\mathrm{MC}}^{0\ell,\mathrm{W+jets}},
\]

where \(\mathrm{Top}=t\bar t+\mathrm{single\ top}\).

The fitted normalization factors were

| Regime | \(\mu_{\mathrm{Top}}\) | \(\mu_W\) | Correlation |
|---|---:|---:|---:|
| High-\(\Delta m\) | \(0.8823\pm0.0101\) | \(1.0504\pm0.0175\) | -0.659 |
| Low-\(\Delta m\) | \(0.9514\pm0.0178\) | \(0.7300\pm0.0086\) | -0.083 |

The normalization covariance was propagated to all target bins.

## Technical closure

The event-level A/B-fold MC closure succeeds:

| Categorization | Valid bins | \(\chi^2/\mathrm{ndf}\) | Maximum absolute pull |
|---|---:|---:|---:|
| High-\(\Delta m\) \(p_{\mathrm{T}}^{\mathrm{miss}}\) | 6/7 | 0.0029/6 | 0.054 |
| High-\(\Delta m\) search bins | 29/60 | 0.0926/29 | 0.173 |
| Low-\(\Delta m\) search bins | 36/42 | 0.1402/36 | 0.206 |

The full-mixture MC pseudodata closure also succeeds, with maximum absolute pulls below 0.19.

These results establish that the histogram reduction, normalization bookkeeping, bin mapping, fold independence, and non-target background subtraction are internally consistent. They do **not** validate the transfer from simulation to data, because both sides of the MC closure are drawn from the same generator model.

## Failure in data residual closure

The data closure directly compares the predicted lost-lepton component with

\[
N_{\mathrm{data}}^{0\ell}
-
N_{\mathrm{other\,MC}}^{0\ell}.
\]

The results are:

| Validation region | Combined-TF ratio | Top/W-split ratio | Top/W-split p-value | Maximum pull |
|---|---:|---:|---:|---:|
| High-\(\Delta m\), \(N_b=0\) | 0.6526 | \(0.6539\pm0.0211\) | \(1.6\times10^{-75}\) | 14.51 |
| High-\(\Delta m\), \(3\leq N_j\leq4,\ N_b\geq1\) | 0.6757 | \(0.6794\pm0.0078\) | \(4.0\times10^{-268}\) | 19.64 |
| Low-\(\Delta m\), low \(p_{\mathrm{T}}^{\mathrm{miss}}\) | 0.8417 | \(0.7763\pm0.0143\) | \(1.4\times10^{-36}\) | 10.58 |
| Low-\(\Delta m\), low ISR | 0.8966 | \(0.8001\pm0.0128\) | \(4.2\times10^{-42}\) | 10.23 |
| Low-\(\Delta m\), low MET significance | 1.0197 | \(0.9911\pm0.2919\) | 0.969 | 0.42 |

The high-\(\Delta m\) transfer-factor prediction accounts for only about 65--68% of the lost-lepton residual required by data. Separating Top and W+jets changes the result by less than 0.4 percentage points and therefore does not address the failure.

The low-MET-significance validation region is statistically compatible with closure, but it has low lost-lepton purity in the zero-lepton target and a large residual uncertainty. A single compatible sideband cannot validate a method that fails strongly in the other, more constraining validation regions.

## Direct evidence that the transfer factor can be harmful

The most decisive test compares the uncorrected target-process MC, the combined transfer-factor estimate, and the Top/W-split estimate using exactly the same valid bins:

| Validation region | Raw lost-lepton MC | Combined TF | Top/W-split TF |
|---|---:|---:|---:|
| High-\(\Delta m\), \(N_b=0\) | 0.6438 | 0.6526 | 0.6539 |
| High-\(\Delta m\), Top-enriched | 0.7332 | 0.6757 | 0.6794 |
| Low-\(\Delta m\), low ISR | **1.0254** | 0.8966 | 0.8001 |
| Low-\(\Delta m\), low \(p_{\mathrm{T}}^{\mathrm{miss}}\) | **0.9904** | 0.8417 | 0.7763 |
| Low-\(\Delta m\), low MET significance | 1.1326 | 1.0197 | **0.9911** |

In the low-ISR validation region, raw MC is within approximately 3% of the data residual. The combined transfer factor moves the prediction to 0.897, and the Top/W-split implementation moves it further to 0.800.

In the low-\(p_{\mathrm{T}}^{\mathrm{miss}}\) validation region, raw MC is within approximately 1% of the data residual. The combined transfer factor reduces the prediction to 0.842, and the Top/W-split implementation reduces it to 0.776.

These are not failures to improve an already poor prediction. They are examples in which an LLCR-derived correction transfers a control-region-specific discrepancy into a zero-lepton region where that correction is not required.

The low-MET-significance region improves after applying the correction, demonstrating that the direction and magnitude of the transfer-factor effect are region dependent. This lack of portability is itself a reason to reject the method. Selecting only the sideband in which the correction happens to work would constitute an unjustified a posteriori choice.

## Independent control-region diagnostics

The two-component fit does not describe all one-lepton control categories.

### High-\(\Delta m\) W-enriched shape

The integrated W-enriched and Top-enriched anchor yields are matched by construction. Their \(p_{\mathrm{T}}^{\mathrm{miss}}\) shapes remain predictive tests:

| One-lepton category | Integrated prediction/residual | Shape p-value | Maximum pull |
|---|---:|---:|---:|
| \(N_b=0\), W enriched | 1.000 | \(7.95\times10^{-4}\) | 3.55 |
| \(3\leq N_j\leq4,\ N_b\geq1\), Top enriched | 1.000 | 0.483 | 1.47 |

The W-enriched control sample retains a recoil-dependent mismatch after its integrated normalization has been fixed.

### Low-\(\Delta m\) held-out \(N_b=1\) category

The low-\(\Delta m\) \(N_b=1\) one-lepton category was excluded from the Top/W normalization fit and therefore provides an independent control-region validation:

- prediction: \(6814.8\pm81.2\);
- data residual: \(8649.9\pm97.3\);
- prediction/residual: \(0.7878\pm0.0129\);
- pull: -14.48.

The normalizations obtained from the \(N_b=0\) and \(N_b\geq2\) anchors underpredict the held-out \(N_b=1\) control data by approximately 21%. Two global process normalizations cannot describe the observed \(N_b\) migration.

## Interpretation

The failed closure is not explained by a missing overall normalization alone. The evidence instead points to non-transferable differences involving one or more of the following:

1. recoil-dependent modeling in the W-enriched one-lepton sample;
2. \(N_b\) migration and b-tagging dependence;
3. the \(t\bar t\)/single-top composition inside the Top category;
4. electron/muon reconstruction, identification, isolation, or acceptance;
5. correlations between lepton loss, hadronic activity, and the target-region categorization;
6. non-target background subtraction in zero-lepton regions with modest lost-lepton purity.

Process-specific transfer factors are themselves different. In the first high-\(\Delta m\) MET bin, representative transfer factors are approximately 0.62 for single-top, 0.76 for \(t\bar t\), and 0.80 for W+jets. Nevertheless, explicitly separating Top and W does not restore data closure. The problem is therefore more fundamental than the use of a single combined process fraction.

## Decision and consequences

The analysis adopts the following decision:

1. **Retire the current MC-derived LLCR transfer-factor method.**
2. Do not use either the combined or Top/W-split TF prediction as the nominal lost-lepton central value.
3. Do not propagate the failed TF estimate into nominal datacards.
4. Do not convert the observed 20--50% nonclosure into an empirical correction without an independently validated model.
5. Preserve the implementation, JSON outputs, and plots as an archived failed-method study.
6. Use raw lost-lepton MC only as a clearly labeled temporary reference, not as a validated final estimate. Raw MC does not solve the high-\(\Delta m\) discrepancy.
7. Require any replacement method to improve closure relative to raw MC in held-out validation regions before adoption.

## Recommended replacement

The preferred replacement is an event-level lepton-removal or lepton-embedding estimate:

1. start from electron and muon one-lepton data separately;
2. emulate a lost lepton by adding its transverse momentum vector to the missing transverse momentum;
3. recompute recoil, angular selections, high-/low-\(\Delta m\) categorization, and search-bin assignment event by event;
4. weight each event by a data-measured probability for the lepton to fail the veto selection;
5. treat out-of-acceptance leptons and tau contributions as explicit auxiliary components;
6. validate the method in MC truth, dilepton tag-and-probe data, flavor-split control samples, and held-out zero-lepton sidebands.

This construction uses the observed hadronic system, b-jet multiplicity, recoil shape, and Top/W mixture in data rather than transferring a control-region normalization through a single MC yield ratio.

## AN-ready conclusion

> The MC-derived lost-lepton transfer-factor method was tested using independent MC folds and orthogonal data validation regions. Although the implementation closes in pure-MC and full-mixture MC pseudodata tests, it fails the data residual closure in both the high- and low-\(\Delta m\) selections. In the high-\(\Delta m\) validation regions, the method predicts only 65--68% of the lost-lepton residual observed in data. In two low-\(\Delta m\) validation regions, the uncorrected lost-lepton MC agrees with the data residual at the 1--3% level, while the application of the transfer factor degrades the agreement to 78--90%. Separating Top and W+jets does not restore closure and exposes additional inconsistencies in recoil shape and \(N_b\) migration. The transfer-factor method is therefore rejected as the nominal lost-lepton background-estimation strategy and is retained only as a documented failed validation study.

