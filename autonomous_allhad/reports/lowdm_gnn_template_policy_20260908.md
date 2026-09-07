# Low-dM: GNN-only final background templates

User instruction adopted on 2026-09-08: all Low-dM background estimations
use GNN score bins as their final templates. This enforces the existing
frozen GNN30 layout; it does not define a new event selection or likelihood.

## Enforced boundary

- Top (TT+ST), W, QCD, and Zinv: all final SR/CR distributions and transfer
  coefficients use their existing GNN score bins and CR parent mapping.
- DY, GJ, VV: the process-separated GNN histograms also supply these smaller
  simulated backgrounds. Missing nominal process records are errors, not an
  invitation to substitute a UT distribution or silently assume zero.
- RZ is still measured in Nb categories. Sgamma and the Z/gamma double ratio
  are still measured versus UT and propagated using the actual GNN×UT joint
  histogram. Their measurement axis is not the final template axis.
- The six SR categories, five score bins in each, four CR parents, and all
  existing physical score edges are unchanged. SR NISR0 maps to CR NISR0;
  SR NISR1/2plus map to CR NISR1plus within the same Nb group.
- GNN TFs divide an SR score-bin yield by the mapped CR parent integral, not
  an equal-index CR score bin. Correlations/rate parameters are unchanged.

## Existing code changes

`workflow/build_histogram_tf_inputs_2024.py` now requires `--gnn-input`,
`--gnn-config`, `--sgamma-input`, and `--dy-measurement`. Both years use the
same entrypoint. TF input schema v4 reserves `lowdm` as a GNN-only pointer to
`lowdm_gnn`; it no longer exposes Low-dM recoil bins as a final template.
The old UT counts are isolated under `diagnostics.lowdm_ut`, explicitly
`template_eligible: false`. Sgamma/RZ input hashes must match the compact
histogram source.

`workflow/gnn_background_histograms.py` enforces complete GNN mappings for
all four controlled processes, unchanged score edges and parent mapping,
and the presence of the RZ×Sgamma Zinv GNN projection. The exported
`lowdm_gnn.template_contract` identifies all process-template sources.

`workflow/plot_recoil_transfer_factors_2024.py` reuses its existing plotting
function with the GNN axis. Low-dM recoil/search34 fallbacks are removed;
default Low-dM plots are the four routes × six GNN categories. The Zinv/GJ TF
plot is explicitly raw, before RZ/Sgamma corrections. `--regime highdm` does
not also redraw Low-dM. The historical output filename
`transfer_factors_YEAR_nb_recoil.json` is retained for caller compatibility;
its schema is now `template_transfer_factors_YEAR_v3` and its
`factors.lowdm.kind` is `gnn`. High-dM calculations are unchanged.

No replacement plotting implementation, event-level reader, ROOT fallback,
new nuisance correlation, score redistribution, or extra Z renormalization
has been introduced.

Regression result: **26 tests passed in 21.18 seconds**. Both years were
exercised end to end with synthetic compact/GNN histograms and the existing
plotter (48 GNN plot files per year, zero Low-dM UT plot files). Checks include
unchanged High-dM TF values, missing-GNN rejection, incomplete/mis-binned
mapping rejection, missing minor-background rejection, and mismatched
RZ/Sgamma histogram-hash rejection. All 254 existing measurement artifacts
retain their recorded SHA256 values.

## Execution after new canonical inputs are validated

Run from the repository root. First calculate RZ and Sgamma from the same
year's validated compact histogram using the existing entrypoints. Then:

```bash
python3 autonomous_allhad/workflow/build_histogram_tf_inputs_2024.py \
  --campaign-year YEAR \
  --hist-input HIST_BACKGROUND_JSON \
  --gnn-input GNN_HIST_JSON \
  --gnn-config autonomous_allhad/gnn_lowdm/config.json \
  --sgamma-input RESULT/sgamma/sgamma_ut.json \
  --dy-measurement RESULT/dy_measurement.json \
  --output RESULT/tf_inputs.json

python3 autonomous_allhad/workflow/plot_recoil_transfer_factors_2024.py \
  --campaign-year YEAR --input RESULT/tf_inputs.json --output-dir RESULT/tf
```

The joint mapping remains at `lowdm_gnn.histograms`; the corrected nominal Z
template remains at `lowdm_gnn.zinv_projection[category].rz_sgamma`. A card
consumer must use these GNN products and must never fall back to
`diagnostics.lowdm_ut`. The main agent owns card integration and was notified
of this schema change. Legacy event-level score-fraction reconstruction is
not part of this workflow.

## Input status

This change is code-only, tested with synthetic histogram fixtures for both
years. The stored 2026-09-07 measurements remain exact, recorded-commit
results for 5-GeV-veto inputs. They were not rewritten to claim a 10-GeV
selection. Production remeasurement waits for the main agent's validated
10-GeV canonical histograms and hashes. No cards, batch jobs, or limits were
run as part of this change.
