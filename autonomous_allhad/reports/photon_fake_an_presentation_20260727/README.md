# 2024 fake-photon AN and presentation plot package

This directory is the local, read-only presentation package for the completed
2024 high-\(\Delta m\) GCR fake-photon measurement.

The trusted event and object selection is
`autonomous_allhad/autonomous_allhad/real_subset_worker.py`. No plot in this
package uses the legacy `decaf/analysis/stop_processor*` or `ids.py` as its
selection source.

## Physics status

- Measurement: complete.
- Validated sidecars: 5,435 / 5,435.
- Logical input segments: 44,935 / 44,935.
- Data-driven fake yield: 849.108 events.
- QCD truth-fake closure: \(311.090/457.057=0.6806\).
- Assigned QCD nonclosure uncertainty: 45.18%.
- Prefit nominal GCR \(U_T\) Data/MC: 1.4156.
- Prefit truth-fake-only replacement Data/MC: 1.3458.
- Entire-QCD replacement Data/MC: 2.1372; rejected.
- Nominal intermediate: unchanged.
- Truth-fake-only replacement: conditional R&D result, not adopted, pending
  the QCD/\(\gamma+\)jets prompt-photon overlap decision.

The prompt-normalization fit result near Data/MC \(=1\) is not part of the
prefit AN/presentation result.

## Directory layout

```text
photon_fake_an_presentation_20260727/
├── README.md
├── docs/
│   ├── an_figure_order.md
│   ├── presentation_slide_order.md
│   └── finalization_checklist.md
├── figures/
│   ├── main/
│   └── appendix/
│       ├── application/
│       ├── target_validation/
│       ├── systematics/
│       └── qcd_closure/
├── data/
│   ├── prefit/
│   └── excluded_postfit_context/
```

### `figures/main`

Eleven selected figures, each stored as PNG and PDF:

1. A/B/C/D method schematic;
2. EB transfer-factor inputs;
3. EE transfer-factor inputs;
4. EB fake factors;
5. EE fake factors;
6. GCR \(U_T\) application-region composition;
7. GCR \(U_T\) QCD truth-fake closure;
8. GCR \(U_T\) fake-background systematics;
9. GCR \(U_T\) target validation;
10. GCR \(U_T\) QCD replacement-policy comparison;
11. final prefit GCR \(U_T\) with the truth-fake component replaced.

### `figures/appendix`

Four validation plot families for each of the 13 supported nonempty
distributions:

- GCR \(U_T\);
- GCR recoil;
- \(H_T\);
- leading-jet \(p_T\);
- b-jet \(p_T\);
- fat-jet \(p_T\);
- \(N_j\);
- \(N_b\);
- \(N_{\mathrm{fatjet}}\);
- \(N_{\mathrm{top}}\);
- \(N_W\);
- \(N_{\mathrm{top}}=0\) recoil;
- \(N_{\mathrm{top}}\geq1\) recoil.

Each appendix directory contains 13 PNG and 13 PDF files.

### `data/prefit`

Machine-readable measurement, normalization, origin, Data/MC evaluation, and
plot-summary records used for the prefit figures.

### `data/excluded_postfit_context`

Auxiliary records that include the prompt-normalization fit study. They are
kept for provenance but must not be used for a prefit AN or presentation
claim.

## Known exclusions

`GCR/met` must not be shown as a physics result. The trusted GCR selection
requires \(p_T^{miss}<250\) GeV while the supplied histogram bins begin at
250 GeV, so all selected events enter underflow.
