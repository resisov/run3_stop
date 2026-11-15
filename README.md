# TROTASR - final canonical analysis

The current inputs, results, analysis code, plotting code and fitting code were
approved by the user as the final frozen baseline on **2026-10-31**.

The sole analysis implementation is [TROTASR](TROTASR/README.md), on the
`run3_full_analysis` branch. Its actual hep2 root is:

```text
/hep2-scratch/twkim/postprocessing/TROTASR
```

Use `python -m TROTASR <stage>` through the existing runtime documented in the
[workflow and datacard review examples](TROTASR/README.md). Current configuration is
`TROTASR/jsons/analysis.json`; current statistical products are `stats/run3`
inside the hep2 TROTASR root.

The official model combines 2022, 2022EE, 2023, 2023BPix, 2024 and 2025 in
8 era-merged channels, 214 likelihood bins and 48 shared CR normalizations.
Era templates, GNN boundaries and nuisance functions are retained. SR data
remain blinded. The adopted systematic scope is unchanged by finalization.

Completed products: 895 expected-limit points, T2tt(1200,500) impacts,
observed CR-only pulls, prefit/postfit predictions and the approved figures.
See [the current baseline handoff](TROTASR/NOMINAL_CONTINUATION.md) for exact paths.

See [AGENTS.md](AGENTS.md) for review and execution policy.

Object, region and correction definitions are documented alongside the code:
[ids.adl](TROTASR/utils/ids.adl),
[event_selections.adl](TROTASR/utils/event_selections.adl), and
[corrections.adl](TROTASR/utils/corrections.adl).

The [workspace map](WORKSPACE_LAYOUT.md) includes the retained hep2
`Stop_Object_Review` material.
