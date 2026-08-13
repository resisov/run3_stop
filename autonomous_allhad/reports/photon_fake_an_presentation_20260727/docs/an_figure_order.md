# Recommended AN figure order

## Main text

### Figure 1 — Method schematic

Files:

- `../figures/main/fake_photon_background_estimation_schematic.pdf`
- `../figures/main/fake_photon_background_estimation_schematic.png`

Purpose: define the orthogonal A/B/C/D regions, contamination subtraction,
fake-factor measurement, application, closure test, and uncertainty model.

### Figure 2 — Transfer-factor inputs

Files:

- `../figures/main/01_transfer_factor_inputs_EB.pdf`
- `../figures/main/01_transfer_factor_inputs_EE.pdf`

Recommended layout: EB and EE as two columns. The upper panels are the
contamination-subtracted C-region numerator inputs and the lower panels are the
D-region denominator inputs.

### Figure 3 — Fake factors

Files:

- `../figures/main/02_transfer_factors_EB.pdf`
- `../figures/main/02_transfer_factors_EE.pdf`

Recommended layout: EB and EE as two columns. Show the central factors, fake
factor statistical uncertainties, and prompt/electron contamination
variations.

### Figure 4 — Application-region subtraction in \(U_T\)

File: `../figures/main/GCR__ut__application.pdf`

Purpose: show the B-region EGamma data, simulated prompt/electron
contamination, and the residual to which the fake factor is applied.

### Figure 5 — QCD truth-fake closure in \(U_T\)

File: `../figures/main/GCR__ut__qcd_closure.pdf`

Purpose: compare the QCD truth-fake A target with the prediction obtained from
the simulated B/C/D sidebands. Quote the integral closure ratio 0.6806 and the
45.18% nonclosure uncertainty.

### Figure 6 — Fake-background uncertainties in \(U_T\)

File: `../figures/main/GCR__ut__systematics.pdf`

Purpose: show fake-factor statistics, prompt contamination, electron
contamination, and QCD nonclosure variations propagated to the prediction.

### Figure 7 — A-region target validation

File: `../figures/main/GCR__ut__target_validation.pdf`

Purpose: compare validation-only A data with prompt MC, electron MC, and the
data-driven fake estimate. The caption must state that A data do not constrain
the prediction.

### Figure 8 — QCD replacement-policy decision

File: `../figures/main/GCR__ut__qcd_replacement_comparison.pdf`

Purpose: compare nominal, rejected entire-QCD replacement, and conditional
truth-fake-only replacement. This figure explains why the nominal QCD process
cannot be identified with the fake-photon background.

### Figure 9 — Final prefit GCR result

File: `../figures/main/gcr-ut-prefit-fake-only.pdf`

Purpose: show the final prefit diagnostic in which only the QCD truth-fake
component is replaced. Quote Data/MC \(1.4156\rightarrow1.3458\) and the
shape-only \(p\)-value \(0.683\rightarrow0.877\). Do not quote the
prompt-normalization fit.

## Appendix

Include the complete application, target-validation, systematics, and closure
families from `../figures/appendix/`. The primary text should refer to these
plots as checks of hadronic activity, jet and b-jet composition, boosted-object
categories, and \(N_{\mathrm{top}}\)-dependent recoil.

Do not include `GCR/met` until a dedicated below-250-GeV histogram exists.

