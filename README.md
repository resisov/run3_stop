# TROTASR - final canonical analysis

The current inputs, results, analysis code, plotting code and fitting code were
approved by the user as the final frozen baseline on **2026-10-31**.

The sole analysis implementation is [TROTASR](TROTASR/README.md), on the
`run3_full_analysis` branch. Its actual hep2 root is:

```text
/hep2-scratch/twkim/postprocessing/TROTASR
```

Use `python -m TROTASR <stage>` through the existing runtime documented in the
[execution guide](TROTASR/README.md). Current configuration is
`TROTASR/jsons/analysis.json`; current statistical products are `stats/run3`
inside the hep2 TROTASR root. No separate campaign, sidecar, old analysis copy
or replacement environment is part of this baseline.

The official model combines 2022, 2022EE, 2023, 2023BPix, 2024 and 2025 in
8 era-merged channels, 214 likelihood bins and 48 shared CR normalizations.
Era templates, GNN boundaries and nuisance functions are retained. SR data
remain blinded. The adopted systematic scope is unchanged by finalization.

Completed products: 895 expected-limit points, T2tt(1200,500) impacts,
observed CR-only pulls, prefit/postfit predictions and the approved figures.
See [the current baseline handoff](TROTASR/NOMINAL_CONTINUATION.md) for exact paths.

Review work reuses these final artifacts. New source/input/model changes,
reprocessing, refitting or rerendering need a new explicit user instruction.
Monitoring remains stopped.

Object, region and correction definitions are documented alongside the code:
[ids.adl](TROTASR/utils/ids.adl),
[event_selections.adl](TROTASR/utils/event_selections.adl), and
[corrections.adl](TROTASR/utils/corrections.adl).

The user explicitly retained hep2 `Stop_Object_Review` as review material,
not an alternative analysis runtime. Current scientific receipts, calibration
inputs and published physics references are preserved. Retired environments,
earlier model outputs, source backups and operational histories are not retained
as restoration mechanisms.
