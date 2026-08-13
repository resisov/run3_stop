# Fake-photon background estimation: AN handoff and adoption audit

## Scope and decision

This report documents the 2024 high-\(\Delta m\) photon-control-region (GCR)
fake-photon measurement and the quantitative decision on whether it should
replace the nominal QCD prediction.

The trusted event and object selection is
`autonomous_allhad/autonomous_allhad/real_subset_worker.py`.  The legacy
`stop_processor_v4.py` and `ids.py` are not used as selection sources in this
measurement.

The full campaign is complete:

- 5,435 validated sidecars from EGamma, G+jets, QCD, DY, single-top,
  \(t\bar t\), diboson, \(W\to\ell\nu\), and \(Z\to\nu\bar{\nu}\);
- 44,935 logical input segments processed out of 44,935 expected;
- 12,869,109,105 input events read and 340,565 sideband events selected;
- no missing process, invalid sidecar, blocked normalization dataset, or
  target `cutBased` mismatch.

The result is:

1. the fake-factor measurement itself is complete;
2. replacing the *entire* nominal QCD process with the measured fake-photon
   yield is rejected because it substantially worsens GCR \(U_{\mathrm T}\)
   Data/MC;
3. replacing only the truth-fake fraction of QCD improves every evaluated
   nonempty distribution, but is not adopted until the overlap between
   prompt photons in inclusive QCD and G+jets is explicitly resolved;
4. the nominal intermediate remains unchanged.

This distinction is essential.  The nominal process labelled QCD is not
equivalent to the fake-photon background.

## 1. Purpose

The GCR is selected with a high-\(p_{\mathrm T}\) photon and otherwise follows
the hadronic-recoil selection used to constrain
\(Z(\nu\bar{\nu})+\)jets.  Its data contain genuine prompt photons, hadronic
objects reconstructed as photons, and a smaller electron-to-photon component.

The purpose of the fake-photon estimate is to measure the reducible hadron-to-
photon component from EGamma data.  This removes dependence on the absolute
QCD multijet rate and on the simulated jet-to-photon misidentification
probability.  The target region is used only for validation and does not
normalize the prediction.

### AN-ready prose

> The reducible background from hadronic objects reconstructed as photons is
> estimated from data using photon-identification sidebands.  Prompt-photon
> and electron-to-photon contributions in the sidebands are subtracted using
> simulation.  The target region is reserved for validation and does not
> constrain the central prediction.

## 2. Methodology

### 2.1 Event and photon selection

The GCR selection is inherited from `real_subset_worker.py`, including trigger,
luminosity mask, MET filters, lepton and tau vetoes, photon-cleaned jets,
\(N_j\), \(N_b\), \(H_{\mathrm T}\), angular requirements, and photon recoil
\(U_{\mathrm T}\).

Photon probes satisfy:

- \(p_{\mathrm T}^{\gamma}>220\) GeV;
- \(|\eta|<1.4442\) or \(1.5660<|\eta|<2.5\);
- the electron veto;
- at least the medium VID level for \(p_{\mathrm T}\), \(\eta\), \(H/E\),
  ECAL isolation, and HCAL isolation.

The two-bit VID level is read from `Photon_vidNestedWPBitmap`.  Medium means a
VID level greater than or equal to 2.

The four mutually exclusive regions are

| Region | \(\sigma_{i\eta i\eta}\) | charged isolation | Role |
|---|---:|---:|---|
| A | pass medium | pass medium | nominal target; validation only |
| B | pass medium | fail medium | application region |
| C | fail medium | pass medium | fake-factor numerator |
| D | fail medium | fail medium | fake-factor denominator |

Region A follows the nominal requirement of exactly one medium photon.
Regions B, C, and D require no A photon and exactly one photon in the B/C/D
sideband union.  The B/C/D definition is therefore orthogonal to A.

### 2.2 Contamination subtraction

For \(X\in\{B,C,D\}\) and photon-\(\eta,p_{\mathrm T}\) stratum \(k\),

\[
N^{\mathrm{fake}}_{X,k} =
N^{\mathrm{data}}_{X,k}
-N^{\mathrm{prompt}\,\gamma,\mathrm{MC}}_{X,k}
-N^{e\rightarrow\gamma,\mathrm{MC}}_{X,k}.
\]

The selected reconstructed photon is classified with `Photon_genPartFlav`:
absolute flavour 1 is prompt, absolute flavour 11 is electron matched, and all
other values are classified as fake.  The subtraction is origin based and
therefore includes prompt/electron contamination from every simulated process,
including QCD.

The full-medium photon-ID scale factor is suppressed because B/C/D probes do
not satisfy the full medium ID.  All other applicable nominal scale factors
are retained.

### 2.3 MC normalization audit

The photon-fake worker stores

\[
w_{\mathrm{sidecar}} =
w_{\mathrm{gen}}\prod w_{\mathrm{nominal\ post\mbox{-}skim\ SF}},
\]

and the measurement multiplies each physical dataset exactly once by

\[
\frac{\sigma_{\mathrm{dataset}}\,\mathcal L}
     {\sum w_{\mathrm{gen,retained}}}.
\]

The independent audit recomputed this factor for all 65 physical MC datasets
using \(\mathcal L=109{,}820~\mathrm{pb}^{-1}\).  All 65 factors agree with the
normalization record to a relative tolerance of \(10^{-13}\); there are zero
missing records, nonfinite factors, cross-section conflicts, or blocked
datasets.

As a second scale sanity check in the common \(U_{\mathrm T}\) range, the
sidecar all-origin target yields differ from nominal by only \(-2.60\%\) for
G+jets and \(-2.16\%\) for QCD.  This small difference is caused by the known
target multi-probe subset, not by a normalization multiplier.

### 2.4 Fake factors

The fake factor is measured in EB and EE in four photon-\(p_{\mathrm T}\)
intervals:

\[
[220,300),\quad[300,400),\quad[400,600),\quad[600,\infty)\ {\rm GeV}.
\]

\[
f_{\mathrm{fake}}(k)=
\frac{N^{\mathrm{fake}}_{C,k}}{N^{\mathrm{fake}}_{D,k}}.
\]

All eight factors are measured directly, with no fallback or cap:

| Region | 220–300 | 300–400 | 400–600 | \(>600\) GeV |
|---|---:|---:|---:|---:|
| EB | \(0.1613\pm0.0117\) | \(0.1034\pm0.0069\) | \(0.0835\pm0.0082\) | \(0.0709\pm0.0194\) |
| EE | \(0.2177\pm0.0225\) | \(0.1141\pm0.0128\) | \(0.0614\pm0.0177\) | \(0.1346\pm0.0776\) |

The prediction in observable bin \(i\) is

\[
N^{\mathrm{fake,pred}}_{A,i} =
\sum_k f_{\mathrm{fake}}(k)
\left[
N^{\mathrm{data}}_{B,i,k}
-N^{\mathrm{prompt}\,\gamma,\mathrm{MC}}_{B,i,k}
-N^{e\rightarrow\gamma,\mathrm{MC}}_{B,i,k}
\right].
\]

The integrated data-driven fake yield is \(849.11\).

### 2.5 Target event audit

The authoritative nominal EGamma target contains 14,497 unique run-lumi-event
keys.  The sidecar target contains 14,468:

- common events: 14,468;
- sidecar-only events: 0;
- nominal-only events: 29, or 0.200%;
- differing \(U_{\mathrm T}\) values among common events: 0.

The 29 events affect A only.  They do not change the B/C/D measurement, so a
multi-hour campaign rerun is not justified.  Nominal `data_obs` is retained
for target validation.

## 3. Closure test

Closure is measured with truth-fake photons in QCD simulation:

1. measure the QCD truth-fake C/D factor;
2. apply it to QCD truth-fake B;
3. compare with QCD truth-fake A;
4. repeat inclusively and in every supported distribution.

For the inclusive GCR recoil,

\[
N_A^{\mathrm{QCD,fake}}=311.09,\qquad
N_A^{\mathrm{QCD,fake,pred}}=457.06,
\]

\[
R_{\mathrm{closure}} =
\frac{N_A^{\mathrm{QCD,fake}}}
     {N_A^{\mathrm{QCD,fake,pred}}}
=0.6806.
\]

The assigned relative nonclosure uncertainty is 45.18%.  It includes the
stored target and prediction statistical variances and does not rescale the
central data-driven prediction.

### AN-ready prose

> Closure is tested in QCD simulation using truth-matched fake photons.  The
> fake factor measured in the simulated sidebands is applied to the simulated
> application region and compared with the truth-fake target.  The observed
> nonclosure is not used to rescale the central value; it is covered by a
> dedicated uncertainty.

## 4. Uncertainties

| Source | Implementation |
|---|---|
| Fake-factor statistics | propagate C and D data/MC `sumw2` through the ratio and application |
| Application statistics | include \(f_k^2\sigma^2_{B,i,k}\) in each predicted bin |
| Prompt contamination | vary prompt MC by \(\pm30\%\), remeasure C/D, and repeat B subtraction |
| Electron contamination | vary electron-matched MC by \(\pm50\%\), remeasure C/D, and repeat B subtraction |
| QCD nonclosure | 45.18%, correlated normalization variation in the present implementation |

The prompt and electron variations are recomputed end to end.  They therefore
change both the fake factors and the application-region subtraction.

The 30% and 50% contamination normalizations and the normalization-only
treatment of nonclosure remain analysis choices that require AN justification.

## 5. GCR Data/MC adoption test

The primary gate is the nominal GCR \(U_{\mathrm T}\) distribution with the
trusted nominal data and identical eight-bin edges.  Four metrics were fixed in
advance: Poisson deviance, \(\chi^2\) using data plus MC statistical variance,
RMS log(Data/MC), and the absolute log integral ratio.

| Prediction | Integral | Data/MC | Poisson deviance | \(\chi^2\) | RMS log ratio |
|---|---:|---:|---:|---:|---:|
| nominal | 10,239.67 | 1.4156 | 1,677.3 | 131.5 | 0.3286 |
| replace entire QCD | 6,782.14 | 2.1372 | 6,618.2 | 3,530.2 | 0.6889 |
| replace truth-fake QCD only | 10,770.55 | 1.3458 | 1,196.5 | 112.2 | 0.2939 |

Replacing the entire QCD process fails all four metrics and is rejected.

The origin audit explains the failure.  In the sidecar QCD A region,

| QCD photon origin | Yield | Fraction |
|---|---:|---:|
| prompt | 3,902.54 | 92.62% |
| electron | 0.03 | \(<0.01\%\) |
| fake | 311.09 | 7.38% |
| all | 4,213.66 | 100% |

Thus, an entire-QCD replacement removes about 3,903 prompt-photon events when
only the 311-event truth-fake component is the object of the data-driven
estimate.

The truth-fake-only diagnostic retains the nominal QCD yield multiplied
bin-by-bin by the sidecar prompt-plus-electron fraction and replaces only its
fake fraction.  It improves all four metrics in \(U_{\mathrm T}\), and all four
metrics also improve in every other nonempty evaluated GCR distribution.
Nevertheless, it remains conditional because inclusive QCD and NLO G+jets
both contain prompt photons and the current nominal production has no explicit
generator-level overlap-removal record.  Adopting this candidate without that
decision could preserve a prompt-photon double count.

### Adoption statement

> The full-QCD replacement is not used.  The nominal GCR prediction is retained
> as the analysis baseline.  The truth-fake-only result is an R&D candidate,
> not an adopted background model, until the QCD/G+jets prompt-photon overlap
> and the target-origin fraction are validated.

## 6. Distribution reasonability audit

The application, target, closure, and systematic plots were inspected.
\(U_{\mathrm T}\), \(H_{\mathrm T}\), leading-jet and fat-jet
\(p_{\mathrm T}\), b-jet \(p_{\mathrm T}\), \(N_j\), \(N_b\), \(N_{\mathrm
fj}\), \(N_{\mathrm top}\), and \(N_W\) have populated, finite distributions.
No QCD-origin fraction is clipped outside \([0,1]\).

The GCR \(p_{\mathrm T}^{\mathrm{miss}}\) histogram is structurally empty and is
excluded from the plot page.  This is a binning error: `real_subset_worker.py`
requires \(p_{\mathrm T}^{\mathrm{miss}}<250\) GeV in the GCR, while the common
high-\(\Delta m\) histogram bins start at 250 GeV.  All selected events
therefore enter underflow.  This is not a zero-yield physics result and cannot
be repaired from the aggregated sidecars without producing a dedicated
below-250-GeV histogram.

## 7. Reproducible outputs

Primary EOS outputs:

- frozen complete snapshot:
  `workflow/photon_fake_2024_snapshot_complete_20260726T1917Z`;
- measurement:
  `measurement_all_background_contamination.json`;
- normalization audit:
  `normalization_audit_all_background_contamination.json`;
- Data/MC evaluation:
  `datamc_evaluation_all_background_contamination.json`;
- origin-aware plot page:
  `workflow/photon_fake_plots_complete_originaware_20260726T1935Z/index.html`.

The plot page uses `mplhep`, square figures, no plot titles,
`hep.cms.label(llabel="Work in progress", rlabel="2024 (13.6 TeV)")`,
the established CR/SR axis labels, and zero horizontal histogram margins.

## Remaining requirements before adoption

1. define and validate generator-level QCD/G+jets prompt-photon overlap
   removal;
2. regenerate the A-region origin fraction with the exact nominal one-medium-
   photon policy if the truth-fake-only candidate is pursued;
3. provide a dedicated GCR \(p_{\mathrm T}^{\mathrm{miss}}<250\) GeV binning
   if that distribution is required;
4. justify the prompt/electron normalization variations and decide whether
   nonclosure must be shape dependent;
5. repeat the predeclared U_T adoption gate after items 1–2.
