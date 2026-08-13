# 2024 fake-photon follow-up: full-production same-region template test

> Work in progress. Nominal intermediate histograms were not modified.

## Scope and validated inputs

This follow-up removes the low-$\Delta\phi$ to high-$\Delta\phi$ fake-factor transfer that failed in the primary template study. The shower-shape template and fitted charged-isolation-pass sample are taken from the same event topology but from disjoint charged-isolation states. Event and object selection remain those of `real_subset_worker.py` through `photon_fake_template_2024_worker.py`.

The input production is complete:

| Campaign | Valid shards | Events read | Compact events |
|---|---:|---:|---:|
| Main allowed non-DY | 5456/5456 | 12,103,540,422 | 440,093 |
| DY2E/Mu/Tau replacement | 669/669 | 1,560,462,096 | 10 |
| Rare supplement | 60/60 | 75,003,040 | 3,792 |

All legacy PTLL-binned DY outputs are excluded. The measurement contains 443,895 deduplicated compact events and uses the complete physical-dataset normalization exactly once.

## Method

For each EB/EE and $p_T^\gamma=220$--400 or at least 400 GeV group:

1. A fake $\sigma_{i\eta i\eta}$ template is built from charged-isolation-fail events in the same region after subtracting prompt-photon and electron-origin MC.
2. The charged-isolation-pass $\sigma_{i\eta i\eta}$ distribution is fitted for an integrated fake yield.
3. The fake $U_T$ shape is taken from the tight-shape charged-isolation-fail residual in the same region.
4. A deterministic two-fold MC test prevents the same MC event pool from both training and testing the template.

A second full-MC pass/fail structural diagnostic removes the fold split. Its fake pass and fail samples remain event-disjoint, but prompt-template events are reused; therefore it diagnoses finite-MC instability and template compatibility but is not an independent closure test.

## Results

### Independent two-fold MC closure

| Region | Integral prediction/truth | 250--400 GeV | 400--650 GeV | 650--1500 GeV |
|---|---:|---:|---:|---:|
| $0.30\leq\min\Delta\phi<0.50$ validation | 0.926 | 1.258 | 0.187 | 0.071 |
| Nominal GCR | 1.746 | 1.963 | 0.987 | 1.536 |

The validation integral happens to be close to unity, but its $U_T$ shape does not close. The nominal GCR also fails the predeclared integral closure gate.

### Full-MC structural diagnostic

| Region | Integral prediction/truth | 250--400 GeV | 400--650 GeV | 650--1500 GeV |
|---|---:|---:|---:|---:|
| $0.30\leq\min\Delta\phi<0.50$ validation | 1.225 | 1.612 | 0.378 | 0.060 |
| Nominal GCR | 0.700 | 0.773 | 0.422 | 1.206 |

Removing the fold split substantially changes the integrals but does not repair the $U_T$ shape. Therefore the failure is not only a two-fold statistical fluctuation: the charged-isolation-fail recoil spectrum is not a stable proxy for tight fake photons.

### Prefit data comparison

Within this evaluator, the replacement moves the integral data/prediction ratio from 1.126 to 1.092 in the validation region and from 1.448 to 1.374 in the nominal GCR. This is a modest improvement, but it is insufficient evidence for adoption because the independent MC shape and GCR integral gates fail.

## Diagnosis

- The primary low-to-high-$\Delta\phi$ transfer method failed catastrophically with MC prediction/truth = 55.86.
- The same-region construction removes that catastrophic topology transfer and is clearly better behaved.
- The remaining failure is dominated by charged-isolation-dependent $U_T$ composition. It persists without the fold split.
- The truth-fake MC targets have only about 11 effective events in the validation region and 10 in the GCR. Individual fold/group fits consequently fluctuate strongly, so the exact integral ratios are not precise enough to define a correction.
- All mechanical gates and all fit-component usability checks pass. This is a physics/statistics closure failure, not missing production, wrong DY input, double normalization, or malformed plotting.

## Decision

**Do not adopt the same-region estimator.** Production completeness and prefit integral-improvement gates pass, but the independent integral and coarse-$U_T$ shape closure gates fail. The nominal fake/QCD component remains unchanged.

If the study is resumed, the defensible next candidate is a regularized simultaneous $\sigma_{i\eta i\eta}$ fit in a small number of coarse $U_T$ intervals. Its regularization strength and binning must be frozen using the validation region before the GCR is examined, and it must retain event-level weights if it is to populate every downstream distribution. A freely normalized fit in every final $U_T$ bin would be tautological and is not acceptable.

## Figures

- [Validation two-fold MC closure](plots/mc_crossfit_highvr_ut_coarse.png)
- [GCR two-fold MC closure](plots/mc_crossfit_gcr_ut_coarse.png)
- [Validation structural diagnostic](plots/mc_structural_highvr_ut_coarse.png)
- [GCR structural diagnostic](plots/mc_structural_gcr_ut_coarse.png)
- [Validation data comparison](plots/data_comparison_highvr_ut.png)
- [GCR data comparison](plots/data_comparison_gcr_ut.png)

Machine-readable result: [evaluation.json](evaluation.json).
