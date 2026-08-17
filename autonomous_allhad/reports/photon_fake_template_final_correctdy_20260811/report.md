# 2024 fake-photon background estimation: template-fit study

> Work in progress. This study does not mutate nominal histograms.

## 1. Objective

The study tests whether the hadronic-fake photon component can be constrained from data and improve the prefit high-$\Delta m$ photon control-region agreement, especially the $U_T$ shape. A method is rejected if closure fails or if the GCR Data/MC agreement is worse than the nominal prediction.

## 2. Event and photon definition

All event selection, jet cleaning, b tagging, recoil, trigger, filter, and region decisions come from `real_subset_worker.py`. The selected photon keeps the medium photon requirements except that $\sigma_{i\eta i\eta}$ and charged isolation are opened for the template fit. Exactly one nominal medium photon takes precedence; without one, exactly one relaxed photon candidate is required. Nominal intermediate ROOT files are read-only and unchanged.

### Production validation

| Campaign | Status | Valid jobs | Source files | Events read | Selected compact events |
|---|---:|---:|---:|---:|---:|
| photon_fake_template_run2method_20260810 | complete | 5456/5456 | 42217/42217 | 12103540422 | 440093 |
| photon_fake_template_run2method_20260810_dy2x | complete | 669/669 | 9367/9367 | 1560462096 | 10 |
| photon_fake_template_run2method_20260810_rare | complete | 60/60 | 937/937 | 75003040 | 3792 |

The rare-background `Runs` audit processed 937/937 files across 7 datasets with status `complete`.

## 3. Method

The factor is measured in `GCR_DPhiVR_Low`. The prompt template is obtained from normalized prompt-photon simulation. The fake template is obtained from data failing loose charged isolation after subtracting prompt-photon and electron-origin MC. The electron component is kept as a separate fixed template. Extended binned likelihood fits to $\sigma_{i\eta i\eta}$ determine fake yields in the charged-isolation pass and loose-not-medium samples. Their ratio defines the fake factor. This follows the template-fit logic used in CMS Run-2 W$\gamma$/Z$\gamma$ measurements rather than assuming ABCD independence.

References: [CMS Wγ full Run 2](https://arxiv.org/abs/2102.02283), [CMS electroweak Zγjj](https://arxiv.org/abs/2002.09902).

## 4. Measured factors

| Requested bin | Used measurement bin | Fake factor |
|---|---|---|
| EB_pt220to300 | EB_pt220to300 | 3.677 ± 1.25 (stat.) ± 0.0682 (syst.) |
| EB_pt300to400 | EB_pt300to400 | 2.242 ± 0.657 (stat.) ± 0.42 (syst.) |
| EB_pt400to600 | EB_pt400to600 | 1.535 ± 0.69 (stat.) ± 0.118 (syst.) |
| EB_pt600toinf | EB_pt400toinf | 1.779 ± 0.784 (stat.) ± 0.0889 (syst.) |
| EE_pt220to300 | EE_pt220to300 | 4.949 ± 3.06 (stat.) ± 1.16 (syst.) |
| EE_pt300to400 | EE_pt220to400 | 3.777 ± 1.67 (stat.) ± 1.22 (syst.) |
| EE_pt400to600 | EE_inclusive | 3.542 ± 1.46 (stat.) ± 1.27 (syst.) |
| EE_pt600toinf | EE_inclusive | 3.542 ± 1.46 (stat.) ± 1.27 (syst.) |

Fine bins automatically fall back to a coarser bin only when the predefined data/template statistics requirements fail. Nominal GCR target data do not choose the binning or factor.

## 5. Closure tests

- MC high-$\Delta\phi$ closure: prediction/truth = 55.856536492240735; $\chi^2$/ndf = 1.1234132936693781.
- Independent data high-$\Delta\phi$ closure: prediction/direct fit = 2.0670201614760284; $\chi^2$/ndf = 1.7631593031053927.
- Nominal GCR validation: prediction/direct fit = 0.6486359639720566; $\chi^2$/ndf = 0.0008493130549191129.

The direct data target is itself obtained from a shower-shape template fit; it is not forced to equal the loose-photon prediction. The $U_T$ points are therefore genuine closure tests.

The predeclared closure gate requires both integral ratios to be within $|\log(\mathrm{prediction}/\mathrm{target})|<0.35$, the data-VR integral pull below 2, and $\chi^2$/ndf below 2.5 (simulation) and 2.0 (data VR). The GCR replacement must also improve the integral distance and all three shape metrics: log-ratio RMS, maximum absolute log ratio, and Poisson deviance.

## 6. Uncertainties

Per-factor uncertainties include fit statistics, the low/high charged-isolation fake-template choice, a ±50% electron normalization variation, a ±30% prompt-contamination variation, and a stage-specific prompt-template shape variation. The larger absolute integral nonclosure from simulation and the independent data VR is propagated as a correlated method uncertainty to the nominal GCR prediction. MC normalization is applied exactly once as generator/SF weight times the physical-dataset normalization factor.

## 7. GCR impact and decision

The integral Data/MC ratio is 1.416 for nominal MC and 1.394 after replacing the nominal fake component. All three predeclared shape metrics improve: False.

**Decision: do not adopt: closure or Data/MC gate failed.**

This decision is deliberately based on closure and prefit Data/MC behavior; no postfit normalization is used.

## 8. Figures

- [Fake factor versus photon pT](plots/fake_factor_vs_photon_pt.png)
- [MC template-fit bias](plots/mc_template_fit_bias.png)
- [EB_pt220to300: pass fit](plots/fit_EB_pt220to300_pass.png)
- [EB_pt220to300: loose fit](plots/fit_EB_pt220to300_loose.png)
- [EB_pt300to400: pass fit](plots/fit_EB_pt300to400_pass.png)
- [EB_pt300to400: loose fit](plots/fit_EB_pt300to400_loose.png)
- [EB_pt400to600: pass fit](plots/fit_EB_pt400to600_pass.png)
- [EB_pt400to600: loose fit](plots/fit_EB_pt400to600_loose.png)
- [EB_pt400toinf: pass fit](plots/fit_EB_pt400toinf_pass.png)
- [EB_pt400toinf: loose fit](plots/fit_EB_pt400toinf_loose.png)
- [EE_inclusive: pass fit](plots/fit_EE_inclusive_pass.png)
- [EE_inclusive: loose fit](plots/fit_EE_inclusive_loose.png)
- [EE_pt220to300: pass fit](plots/fit_EE_pt220to300_pass.png)
- [EE_pt220to300: loose fit](plots/fit_EE_pt220to300_loose.png)
- [EE_pt220to400: pass fit](plots/fit_EE_pt220to400_pass.png)
- [EE_pt220to400: loose fit](plots/fit_EE_pt220to400_loose.png)
- [Simulation closure in UT](plots/simulation_closure_ut.png)
- [Independent data closure in UT](plots/data_validation_closure_ut.png)
- [Nominal GCR fake prediction in UT](plots/gcr_fake_prediction_ut.png)
- [Nominal versus fake-replacement GCR](plots/gcr_nominal_vs_fake_replacement_ut.png)
