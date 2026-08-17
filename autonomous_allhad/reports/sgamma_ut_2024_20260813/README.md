# 2024 Sgamma versus UT

Status: complete, preliminary statistical measurement.

## Definition

For each adopted category `g`, the integrated photon normalization and the
bin-wise shape factor are

```text
Q_g          = sum_i (Data_i - OtherMC_i) / sum_i GJ_i
Sgamma_g,i   = (Data_i - OtherMC_i) / (Q_g * GJ_i)
```

The data source is EGamma0/EGamma1 only. `OtherMC` is every normalized GCR MC
process except GJ. The event and object selections are the trusted nominal
features produced by `real_subset_worker.py`.

## Inputs and audit

- Exact GCR build: 3231/3231 allowed ROOT inputs, zero missing roots.
- Five obsolete repair ROOTs were explicitly excluded: three legacy GJ PTG
  records and two legacy QCD PT records superseded by the adopted GJ-4Jets
  HTxPTG and QCD-4Jets HT replacements.
- GJ: 16 GJ-4Jets HTxPTG physical datasets with dRGJ > 0.25.
- QCD: 11 QCD-4Jets HT physical datasets.
- DY: DYto2E-4Jets, DYto2Mu-4Jets, and DYto2Tau-4Jets only; zero PTLL records.
- GCR data: EGamma0 and EGamma1 only; no JetMET or Muon data.
- Normalization SHA256:
  `f4acec64255c61c77eb9007da1b4fe1e747ae696372112ac5f134948c063d974`.

## Integrated Q values

| Regime | Category | Q | Statistical uncertainty |
|---|---:|---:|---:|
| High-dM | Nb = 1 | 1.8432 | 0.0254 |
| High-dM | Nb = 2 | 2.1418 | 0.0513 |
| High-dM | Nb >= 3 | 3.2505 | 0.1964 |
| Low-dM | Nb = 1 | 1.8650 | 0.0216 |
| Low-dM | Nb >= 2 | 1.7980 | 0.0528 |

## Interpretation

- Because Q is derived from the same bins, the GJ-weighted average Sgamma is
  one by construction within each Q category. Sgamma therefore diagnoses the
  residual UT shape, not the overall GCR normalization.
- High-dM is close to one through 800 GeV. The 800--1500 GeV bin is lower:
  0.736 +/- 0.076 (Nb=1), 0.862 +/- 0.161 (Nb=2), and
  0.249 +/- 0.203 (Nb>=3). The last number contains only two data events before
  the non-GJ subtraction and is statistically fragile.
- Low-dM Nb=1 with pTISR >= 500 is category-dependent: the pTb=20--40 GeV
  family is systematically below one, while pTb=40--70 GeV is closer to one.
- The four Nb>=2 families with pTb<140 GeV are broadly compatible with one.
  The two pTb>140 GeV, Nj>=7 families are statistically weak; one bin is
  unavailable and the remaining uncertainties are too large for a useful
  shape constraint.
- The displayed errors are the current diagonal statistical propagation,
  including Q and MC statistical terms. Correlations and systematic
  uncertainties are not included in these preliminary error bars.
- The internal sum identity closes to 1.9e-16 relative precision.

The Low-dM plot preserves all ten pTb/Nj category families and all 34 search
bins; no category yields are combined for plotting. The last plotted UT bin in
each family includes overflow and is displayed with a visual upper cap of
1500 GeV. Full inputs and all 52 High- and Low-dM bin values are in
`exact.json`, `sgamma_ut.json`, and `sgamma_ut.csv`.
