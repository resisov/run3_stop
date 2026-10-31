# TROTASR

CMS Run-3 all-hadronic stop analysis for 2022, 2022EE, 2023, 2023BPix,
2024 and 2025. The package provides event histograms, background estimates,
statistical templates, expected limits and the analysis figures.

## Final canonical baseline

The user approved the current inputs, results, analysis, plotting and fitting
implementation as the final frozen TROTASR on 2026-10-31. Use this top-level
package only. Further changes, regeneration or fitting require a new explicit
instruction; review preparation reuses the completed products. The declaration
does not change the adopted six-era partial-systematic physics model.

## One command interface

Run from the repository root in the analysis environment:

```sh
python -m TROTASR --help
python -m TROTASR <stage> --help
```

On hep2 use the installed numerical environment:

```sh
cd /hep2-scratch/twkim/postprocessing
bash TROTASR/workflows/run_inference.sh -m TROTASR <stage> --help
```

| Stage | Input | Output | Implementation |
|---|---|---|---|
| `histograms` | Integrated event ROOT, target era | High-/low-dM histograms | `workflows/build_highdm_histogram.py` |
| `merge` | Per-file campaign histograms | Per-era merged histograms | `workflows/merge_nominal.py` |
| `background` | Per-era merged histograms | RZ, Sgamma, double ratios and transfer factors | `workflows/build_background_estimation.py` |
| `cards` | Analysis configuration, histograms and background estimates | Template ROOT, cards and grid manifest | `workflows/build_nominal_grid.py` |
| `limits` | Grid manifest and cards | Expected CLs results | `workflows/combine_limits.py` |
| `impacts` | Grid manifest and cards | Nuisance impacts | `workflows/combine_impacts.py` |
| `cronly` | Grid manifest and cards | Control-region fit | `workflows/combine_cronly.py` |
| `plot` | Existing fit results or compact plot inputs | PNG/PDF figures | `workflows/plotting_*.py` |

The stage modules share the physics implementations in `utils/` and
`gnn4lowdm/`. Changing a command does not introduce a second physics pipeline.
Use the stage whose inputs are already available; plotting does not require
repeating event processing or fitting.

## Approved analysis

- The official likelihood merges corresponding six-era channels: 48 channels
  become 8, 1284 likelihood bins become 214, and 288 free CR normalizations become
  48 shared factors. Each era keeps its original templates, low-dM GNN boundaries
  and nuisance functions. The approved `eraScale` rateParam formulas preserve
  era-specific central corrections and bounds; histogram rates remain `-1`.
- High-dM contains 118 merged likelihood SR bins. Low-dM has six Nb x NISR
  categories with five score bins each. This is the adopted model, not an
  optional alternate campaign or command-line model switch.
- SR observations are blinded. Limit calculations are expected-only.
- Retain the approved CR constraints, background errors and
  `autoMCStats 10 1 1`. The current six-era result has partial systematics.
- Fit only the displayed mass window: mStop 600--1800 GeV and mLSP 1--1800 GeV,
  containing 895 points. The input grid retains all 955 points.
- Preserve the approved early-era central luminosity, 61.8971 fb^-1.
  A different luminosity label must not substitute for recalculating yields.

The current statistical configuration on hep2 is
`jsons/analysis.json`. It specifies the actual inputs,
era channels, nuisance model and the approved limited exclusions. The file is
an analysis input; a source-only checkout is not a complete production payload.
The exact active result locations are in `NOMINAL_CONTINUATION.md`.

## Run from completed inputs

Use the single top-level package and its existing product locations. Do not
create a dated/request-specific analysis checkout or execution script.
Current products live in `stats/run3`; retired execution trees and launchers
have been removed permanently. Do not recreate them from historical records.

```sh
python -m TROTASR histograms \
  --input TROTASR/inputs/2024/events.root --year 2024 --mode both \
  --output TROTASR/results/histograms/2024.json.gz

python -m TROTASR merge \
  --campaign TROTASR/results --year 2024

python -m TROTASR background \
  --hists TROTASR/results/merged/nominal_2024.json.gz \
  --output TROTASR/results/measurements/2024

python -m TROTASR cards
python -m TROTASR limits
python -m TROTASR impacts
python -m TROTASR cronly

python -m TROTASR plot limits \
  --grid TROTASR/stats/run3/grid/manifest.json \
  --limits TROTASR/stats/run3/limits \
  --output TROTASR/stats/run3/plots/limits

python -m TROTASR plot predictions \
  --cronly-result TROTASR/stats/run3/cronly \
  --output TROTASR/stats/run3/plots/cronly
```

The CR-only prefit/postfit comparison reuses the accepted observed-CR workspace
and saved background-only fit. It does not refit or rebuild inputs. The saved
post-fit covariance is propagated with correlations using the saved RooFitResult
(500 shared samples). The native CMS histogram totals and process wrappers are
evaluated directly, without repeated process-histogram construction. The prefit
band uses the model's prior uncertainties: stored Gaussian widths and the
Poisson count variance for the low-statistics MC auxiliaries.
Both stages include uncertainty bands; actual fit
quality is recorded without imposing a covariance-quality-only plotting veto.
One 24-inch-wide comparison joins high-dM on the left and low-dM on the right,
with prefit above postfit. It calls the existing canonical distribution renderer
with the existing stack colors and process labels. Each box includes
its CR, regime and Nb category; only this comparison omits legend yield counts.
Its legend is one horizontal row, category dividers span the full panel, boxes
are enlarged, and Prefit/Postfit titles are larger and bold. The compact
24-by-11-inch base canvas uses shorter main/ratio axes and a tighter main y limit.
Export trims only horizontal whitespace to 0.10-inch outer padding, preserving
the plot artists' sizes and vertical placement.
Category boxes sit inside the main axis immediately above its x axis, even
where they cover the background stack; they are not placed below the ratio axis.

The worker count is an admission ceiling, not a separate allocation for each
command. Production and fitting share the account's current CPU and memory
budget. Reuse completed inputs and results; never run a second controller over
an active output directory. Current completion receipts and paths are listed
in `NOMINAL_CONTINUATION.md`.

For the six-era systematic grid, `limits` runs the primary fits, the finite
recovery sequence and result collection in one workflow. Failed points take
priority when capacity becomes available. Completed results and running point
locks are reused; restarting the same command continues the saved recovery
plan rather than creating another retry sequence. Same-point successful prior
numerical recipes are tried first on the current workspace, within the existing
finite budget; historical fit values are never reused as current results. No
sidecar or separate manual recovery command is required.

The default commands use `stats/run3/grid/manifest.json` and its sibling product
directories. Completed results are reused directly. Historical readers never
authorize executing an old analysis copy or restoring a retired launcher.

`plot --help` lists limits, predictions, distributions, GNN controls,
background measurements, transfer factors and region schematics. Partial limits require
`plot limits --allow-partial`; missing points must not be filled by
extrapolation. Figures use the existing analysis renderers.

## Physics and data locations

| Location | Purpose |
|---|---|
| `utils/ids.py`, `utils/nanoaod_ids.py` | Object definitions |
| `utils/event_selections.py`, `utils/region_kinematics.py` | Regions and kinematics |
| `utils/corrections.py`, `utils/private_scales.py`, `utils/weight_components.py` | Calibration and event weights |
| `utils/statistics_inputs.py`, `utils/statistics_model.py`, `utils/statistical_templates.py` | Likelihood construction |
| `gnn4lowdm/` | Classifier, features and era-specific bin definitions |
| `jsons/`, `scales/`, `estimations/`, `models/` | Configuration, calibration, normalization and model inputs |
| `results/` | Current histogram/measurement products; no request-specific analysis roots |
| `stats/` | Templates, cards, fit state and results |
| `stats/run3/` | The one current six-era statistical model and its diagnostics |
| `workflows/renderers/`, `utils/renderers/` | Common figure implementations |

Source NanoAOD/MiniAOD remains read-only. ROOT inputs, templates and fit scratch
normally stay on the server. Compact plot inputs and figures can be delivered
to the laptop. Historical results are not runtime dependencies of a new
analysis. Do not import code from a result folder or an old analysis checkout.

The companion `utils/ids.adl`, `utils/event_selections.adl` and
`utils/corrections.adl` describe the adopted physics definitions. Runtime state
and completed-result records belong with their products, not in source-code
update histories.
