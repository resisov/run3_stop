# Higgsino-stop relic-density contours

This directory reproduces the numerical thermal relic-density contours behind
Fig. 4.2 of Bhattiprolu, Martin, and Wells, *Chasing Higgsino dark matter at
colliders in the neutrino fog era*, Phys. Rev. D 113, 095029 (2026),
[arXiv:2512.12457](https://arxiv.org/abs/2512.12457).

## Inputs and assumptions

- Published numerical data: [Zenodo 19666737](https://doi.org/10.5281/zenodo.19666737),
  archive `FigureData_arXiv2512.12457.zip`, directory `Figure5`.
- Archive MD5: `d0d68bbf2238d7ca0050a1e3a8a9273b`, matching the Zenodo record.
- Relic-density calculation: micrOMEGAs 6.0.
- Spectrum and one-loop masses/couplings: SuSpect 2.41.
- Model point: negative mu, tan(beta) = 1.6.
- Gaugino mass parameters and all scalar masses other than the light Higgs are
  set to 10 TeV; the light Higgs mass is 125.1 GeV.
- Cosmology: standard thermal freeze-out.

The six fixed reference contours are Omega(LSP) h^2 = 0.0045, 0.01, 0.03,
0.06, 0.09, and 0.12.  The hook near the stop-LSP diagonal is the stop-coannihilation branch.
Away from the stop-coannihilation region, the Omega h^2 = 0.12 contour tends
to a higgsino-like LSP mass near 1.1 TeV.

## Collider-overlay caveat

The bundled right panel includes the 2026-09-01 snapshot of the 2024+2025 T2tt
expected 95% CL contour as a reference only, not the latest analysis result.
It is not a valid exclusion of the paper benchmark.  T2tt
assumes BR(stop -> top neutralino1) = 100%, while the higgsino benchmark also
contains a nearly degenerate chargino and second neutralino.  For an
unsuppressed top channel the paper quotes approximate stop branching fractions
of 50% to bottom+chargino and 25% to each top+neutralino state.  Near or below
the on-shell-top threshold the bottom+chargino mode dominates.  A physical
limit therefore requires dedicated signal samples and a combined acceptance
for the bottom+chargino and top+neutralino final states.

## Reproduction

```bash
MPLCONFIGDIR=.workspace-local/matplotlib python3 \
  autonomous_allhad/workflow/plot_higgsino_relic_contours.py \
  --source-dir autonomous_allhad/reports/higgsino_relic_contours_20260902/source \
  --output-dir autonomous_allhad/reports/higgsino_relic_contours_20260902
```

This command draws the fixed relic-density contours without a collider overlay.
Use `--expected-limits <validated-expected_limits.json>` to add a reference overlay.

The JSON output preserves all published contour points and records the model
assumptions and source identifiers.
