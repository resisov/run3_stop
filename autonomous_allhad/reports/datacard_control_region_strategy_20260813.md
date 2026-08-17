# Datacard control-region strategy for the 2024 analysis

## Adopted decision

The statistical model must reproduce the Run-2 control-region logic, with the
2024 measurements and binning substituted for the Run-2 inputs.

1. Include only the LLCR, QCDCR, and GCR as simultaneous-fit control channels.
2. Connect every fitted control bin to its target SR bin through the measured
   MC transfer factor in the corresponding analysis category and `U_T` bin.
3. Do not include the dielectron or dimuon DYCR as Poisson channels.  Measure
   `R_Z` outside the signal fit and insert its central values and complete
   covariance as constrained nuisance parameters.
4. Apply the measured `S_gamma` factors to the nominal `Z -> nu nu` SR shape.
   Retain the GCR in the likelihood so that its observed yield constrains the
   residual photon-to-Z rate parameter.
5. Do not use the same control data twice.  In particular, do not add an
   external Gaussian constraint derived from GCR statistics when the same GCR
   bins are already present as Poisson channels.

The DYCR is sufficiently far from the zero-lepton target phase space that its
use as a simultaneous-fit channel is not adopted.  Its information enters only
through the external `R_Z` measurement, its covariance, and the Z/gamma closure
study.  The GCR is retained because it is the direct shape-control sample for
the `Z -> nu nu` prediction.

## Control-to-signal transfer-factor model

For process `p`, category `c`, and recoil bin `i`, use a control normalization
parameter `mu[p,c,i]` and preserve the measured MC ratio

```text
TF[p,c,i] = N_MC[p,SR,c,i] / N_MC[p,CR,c,i].
```

The expected target and control yields are therefore

```text
lambda[p,CR,c,i] = mu[p,c,i] * N_MC[p,CR,c,i]
lambda[p,SR,c,i] = mu[p,c,i] * TF[p,c,i] * N_MC[p,CR,c,i].
```

An equivalent implementation is to use the nominal MC templates in both
channels and apply the same `rateParam` to the matched SR and CR processes.  In
either implementation, the SR/CR ratio must remain the measured transfer
factor.  A control parameter must never be shared between bins that do not
have an explicit mapping.

Use the following mappings:

- High-Delta-m: six native `U_T` bins, separately for `N_b = 1` and
  `N_b >= 2`.
- Low-Delta-m: all 34 adopted Run-2-derived search-bin categories.  These are
  the `N_b = 1, N_SV = 0` and `N_b >= 2` subset of the Run-2 53-bin scheme,
  with their native `pT_ISR`, `pT_b`, `N_j`, and recoil divisions.
- Top is defined as `TT + ST`.
- QCD uses the QCDCR as its control process.

### Lost-lepton identifiability

One LLCR count constrains the sum of Top and W, not two arbitrary independent
normalizations.  Giving Top and W unconstrained independent `rateParam`s in the
same LLCR bin produces a degeneracy.

The recommended nominal implementation is therefore:

- use one free `ll_norm[c,i]` parameter shared by Top and W in the matched
  LLCR/SR bins;
- retain the separate Top and W transfer factors in their respective nominal
  templates; and
- add a constrained Top/W composition nuisance derived from MC.

Independent Top and W rate parameters are not part of the adopted model.  They
would require additional composition-sensitive control information.

## External `R_Z` constraint from the DYCR

For each adopted `R_Z` category `k`, the nominal invisible-Z yield before the
shape correction is

```text
N_Z_nominal[k,i] = R_Z[k] * N_Z_MC[k,i].
```

Do not reduce the `R_Z` measurement to independent per-bin `lnN` terms when a
covariance is available.  Let `r` be the vector of central values and `V_r` its
covariance.  Implement a positive multiplicative model in log space:

```text
eta = log(r)
V_eta[k,l] ~= V_r[k,l] / (r[k] * r[l])
V_eta = L * L^T
R_Z[k](theta) = r[k] * exp(sum_a L[k,a] * theta[a])
theta[a] ~ Normal(0,1).
```

An eigenvector decomposition of `V_eta` is equivalent and is often easier to
translate into Combine shape/`param` nuisances.  Preserve the correlations
across `N_b`, lepton channels, and High-/Low-Delta-m categories when they arise
from shared detector, normalization, or subtraction sources.

The existing `R_Z` summaries contain combined central values and statistical
errors, plus per-channel `(R_Z, R_T)` covariance matrices.  They do **not** yet
constitute a documented final cross-category covariance matrix for the
datacard.  The main agent must build that matrix before producing the final
card.  A diagonal matrix may be used only as an explicitly labeled
temporary statistical-only model.

Do not include the DYCR as a fit channel after applying this external
constraint.  Doing both would count the same DYCR observations twice.

## Applying `S_gamma` to the Z shape

For category `g` and recoil bin `i`, the measured correction is

```text
Q_gamma[g] = sum_i(Data_i - Other_i) / sum_i(GammaJets_i)
S_gamma[g,i] = (Data_i - Other_i) /
               (Q_gamma[g] * GammaJets_i).
```

Apply it to the nominal invisible-Z template:

```text
N_Z_SR[g,i] = R_Z[k(g)] * S_gamma[g,i] * N_Z_MC_SR[g,i].
```

The category map `k(g)` must be explicit.  In High-Delta-m, map the measured
`R_Z(N_b)` to the corresponding `S_gamma` category, combining `N_b >= 2`
where required by the adopted datacard.  In Low-Delta-m, use the adopted
`N_b = 1` or `N_b >= 2` `R_Z` value for every detailed family in that `N_b`
class while retaining the family-specific `S_gamma` shape.

### Retaining the GCR without double counting

The GCR must remain a Poisson channel.  The safest parameterization is to
pre-scale the nominal templates by the measured central values and fit an
otherwise unconstrained residual rate parameter `rho_gamma[g,i]`:

```text
N_GCR_target[g,i] = rho_gamma[g,i] *
                     Q_gamma[g] * S_gamma[g,i] * N_GammaJets_MC[g,i]

N_Z_SR[g,i] = rho_gamma[g,i] *
              R_Z[k(g)] * S_gamma[g,i] * N_Z_MC_SR[g,i].
```

The GCR Poisson term is the statistical constraint on `rho_gamma`.  The
precomputed `S_gamma` is a central-value reparameterization and starting point,
not an additional observation.  Therefore:

- do not also constrain `rho_gamma` with the diagonal `S_gamma` statistical
  errors from the same GCR data;
- do not add `Q_gamma` statistical errors as independent Gaussian nuisances
  when `Q_gamma` and the same GCR counts are already represented in the
  Poisson model; and
- propagate detector/model variations of GCR target and contamination
  templates coherently through both GCR and Z SR.

If the implementation cannot share `rho_gamma` between the GCR and Z SR, the
alternative is to remove the GCR channel and use the full `S_gamma` covariance
as an external constraint.  That alternative is not the adopted model.

## Z/gamma closure uncertainty

The measured double ratio

```text
D[i] = S_Z[i] / S_gamma[i]
```

tests the portability of the photon-derived shape to Z production.  It is not
another normalization factor and must not replace `R_Z` or `S_gamma`.

Apply an adopted photon-to-Z nonclosure nuisance to the Z SR only; do not apply
it to the GCR.  Because GCR statistics are already present as Poisson terms and
DYCR normalization statistics enter the external `R_Z` covariance, do not
blindly insert the full plotted `max(|D-1|, sigma_D)` band as a new independent
nuisance.  The recommended final treatment is:

1. use the central nonclosure `|D-1|` as the candidate transfer systematic;
2. construct a joint or bootstrap covariance if the statistical sensitivity of
   the double-ratio validation must also be represented;
3. correlate a given `U_T` nonclosure nuisance across all `N_b` and detailed
   categories to which the same inclusive High-/Low-Delta-m measurement is
   mapped; and
4. keep different `U_T` bins independent unless a validated smoothing model or
   source-level correlation justifies a correlation.

The Run-2 `max(|D-1|, sigma_D)` envelope may be retained as a labeled
conservative alternative.  It must be called a closure uncertainty, not a pure
systematic, and it must replace rather than duplicate its overlapping
statistical components.

## Mandatory action for the missing `Zto2Nu` SR histogram

The transfer-factor exact file is intentionally scoped to the Top, W, and QCD
measurements.  It is not a complete datacard-template source and does not
contain the `Zto2Nu` SR histogram.  Do not interpret this as missing MC
production, do not rerun NanoAOD, and do not drop the GCR--Z connection.

Read the already-produced `Zto2Nu` SR histograms from the nominal histogram
intermediate:

```text
/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/plot2024/hists.json
```

For `Zto2Nu`, import the binwise nominal weighted yield, `sumw2`, and all
available systematic up/down histograms into the Combine template input:

- High-Delta-m: use the adopted 60 source bins and apply the existing 60-to-55
  merge map used by the final SR card;
- Low-Delta-m: use the adopted 34 `cat7_SR_lowDeltaM` search bins;
- preserve the explicit `N_b`/category and `U_T` mapping used by `S_gamma`.

Keep the current transfer-factor files unchanged for Top, W, and QCD.  The
datacard builder may either read the nominal `Zto2Nu` histogram source directly
or materialize a small dedicated `zinv_sr_inputs.json`; this is only extraction
from an existing histogram, not new event processing.

Use the photon side already stored in:

```text
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/sgamma_ut_2024_20260813/exact.json
```

For every mapped category and `U_T` bin, the same free parameter must multiply
the photon target in the GCR and the invisible-Z target in the SR:

```text
GCR: PhotonJet -> zg_norm[g,i] * Q_gamma[g] * S_gamma[g,i] * GammaJets_MC[g,i]
SR:  Zto2Nu   -> zg_norm[g,i] * R_Z[k(g)] * S_gamma[g,i] * Zto2Nu_MC[g,i]
```

Do not substitute the normalized Z shape in the double-ratio file for the
absolute `Zto2Nu` SR histogram.  Do not silently create a zero Z template, and
do not remove an unmatched `zg_norm`; supply the missing SR histogram so that
the GCR and SR occurrences of `zg_norm[g,i]` are both present.

## Required machine-readable inputs

Transfer factors and exact High-Delta-m input:

```text
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/transfer_factors_2024_20260813/transfer_factors_2024_nb_recoil.json
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/transfer_factors_2024_20260813/highdm_exact_nb12_final_current.json
```

Nominal `Zto2Nu` SR histogram source:

```text
/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/plot2024/hists.json
```

`R_Z` measurements:

```text
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/dy_rz_nb_run2method_20260731/highdm/summary.json
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/dy_rz_nb_run2method_20260731/lowdm/summary.json
```

`S_gamma`, `Q_gamma`, and double-ratio measurements:

```text
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/sgamma_ut_2024_20260813/sgamma_ut.json
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/zgamma_double_ratio_2024_20260813/zgamma_double_ratio.json
```

The `R_Z` inputs for this implementation are restricted to the adopted
`DYto2E-4Jets`, `DYto2Mu-4Jets`, and `DYto2Tau-4Jets` production.  Forbidden
PTLL-binned DY artifacts are outside the model.

## Suggested implementation order

1. Generate an explicit SR-to-CR bin-map JSON.
2. Build the final `R_Z` covariance artifact.
3. Apply `R_Z` and `S_gamma` to the nominal Z SR templates.
4. Create matched LLCR, QCDCR, and GCR channels and shared `rateParam`s.
5. Add non-overlapping detector, model, transfer-factor, and closure nuisances.
