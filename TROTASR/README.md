# TROTASR — all-hadronic stop analysis

TROTASR turns integrated event ROOT files into high-/low-dM histograms,
background estimates, ROOT templates, Combine datacards, expected limits,
impacts, pulls, and plots. The implemented histogram and statistical workflow
covers **2024 and 2025**.

Start at the stage for which you already have inputs:

- **Integrated ROOT files:** [histograms and systematics](#3-produce-histograms).
- **Completed histograms:** [merge and background estimates](#4-merge-and-measure-backgrounds).
- **Completed combined histograms:** [templates through final plots](#5-run-templates-limits-impacts-and-plots).
- **Existing cards or fit results:** [individual downstream commands](#6-run-one-downstream-stage).
- **Analysis definitions:** [ADL and Python definitions](#analysis-definitions).

This is the current analysis package. The older `autonomous_allhad/` analysis
drivers are not alternative TROTASR histogram or datacard entry points.

## What is included

The repository contains the analysis code, configurations, calibration
payloads, efficiency inputs, normalization files, and portable low-dM GNN.
It does **not** contain event ROOT files, TROTA HDF5 models, campaign output
directories, fit workspaces, or generated figures. A Git clone alone cannot
reproduce a full result without those inputs.

| Era | State of this package |
|---|---|
| 2024, 2025 | Implemented histogram, background, template and statistical workflow |
| 2022, 2022EE, 2023, 2023BPix | POG payloads and Top/W reference payloads are staged; the TROTASR histogram/statistical year interfaces are not yet integrated |

The 2022/2023 inputs use NanoAODv12, PNet AK4 b tagging and PNetWithMass AK8
Top/W tagging. Do not feed them to the 2024/2025 UParT/GlobalParT3 workflow
by changing a year label. Approved 2023 signal reuse from the 2022 campaign
does not change that interface requirement.

## Current analysis model

| Item | Definition |
|---|---|
| High-dM SR | 21 classes, 118 bins per year after the fixed category merge |
| Low-dM SR | Six `Nb × NISR` classes, five GNN bins each: 30 bins per year |
| High-dM LLCR, QCDCR, GCR | 12 bins each per year: two Nb groups × six recoil intervals |
| High-dM CR recoil | 250–300, 300–350, 350–400, 400–500, 500–800, ≥800 GeV |
| Low-dM LLCR, QCDCR, GCR | 10 bins each per year: two Nb groups × five common GNN bins, inclusive in NISR |
| Combined likelihood | 16 multibin channels, 428 bins, 96 free background normalization parameters |
| Normalization range | `[0.01, 5]` |
| MC statistics | `autoMCStats 10 1 1` |
| Background-estimate uncertainties | Eight RZ statistical and 26 nonclosure shape nuisances |
| Signal | SR only; CR signal contamination omitted by the adopted model |
| Data | Observed SR remains blinded |

The histogram source map has 162 high-dM SR bins.
[`jsons/sr_category_merge.json`](jsons/sr_category_merge.json) maps them to
the final 118-bin model; it is not an occupancy-dependent merge.
[`jsons/combined_systematics_config.json`](jsons/combined_systematics_config.json)
selects the complete current model and the 955-point signal grid.

The combined systematic prescription uses the corrected-central JME nominal,
weight variations and JES/JER, together with the stored-TROTA non-JME
**absolute** endpoints. These endpoint sets have different nominal
conventions; the adopted combination does not recenter them. The prescription
is explicit in `systematic_combination` in that configuration.

## Directory layout

| Directory | Contents |
|---|---|
| `workflows/` | Commands listed below; plotting implementations in `workflows/renderers/` |
| `utils/` | Object selection, corrections, histogramming and statistical functions |
| `jsons/` | Analysis configuration and input/payload inventories |
| `scales/<POG>/<year>/` | Official JME, BTV, EGM, MUO, TAU and LUM payloads |
| `scales/Private/<year>/` | Measured trigger and Top/W scale factors |
| `estimations/` | Normalization and tagging-efficiency inputs; campaign background measurements |
| `gnn4lowdm/` | Frozen low-dM classifier, feature construction and bin definitions |
| `models/TROTA/` | Model inventory; HDF5 model files must be provisioned on the server |
| `inputs/<year>/` | Integrated ROOT inputs and their existing metadata, server only |
| `campaigns/<name>/` | Input plan, per-file histograms, state, logs and merged histograms |
| `stats/<name>/` | Templates, cards, fit results, plots and downstream state |

Generated files stay in their campaign or result directory. Internal tests,
validation products, large histogram exports, ROOT files, PNGs and PDFs are
not committed.

## 1. Set up the runtime

The commands below are for the existing **hep2** installation:

```sh
cd /hep2-scratch/twkim/postprocessing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Use the package's runtime launcher for the commands in this guide:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_highdm_histogram.py --help
```

[`workflows/run_inference.sh`](workflows/run_inference.sh) selects the
installed CMSSW 14_1_0_pre4 / Python 3.9 numerical environment and its
TROTASR dependencies. Required Python packages include NumPy, Awkward,
Uproot, Coffea, correctionlib, hist, SciPy and Matplotlib; JME/TROTA inference
also needs TensorFlow. ROOT, Combine and CombineHarvester are required for
templates and fits.

The installed Python package versions checked on 2026-10-06 are:

| Package | Version |
|---|---|
| Python | 3.9.14 |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| Awkward / Uproot | 1.10.3 / 4.3.7 |
| Coffea / correctionlib | 0.7.21 / 2.7.0 |
| hist / Matplotlib | 2.6.2 / 3.7.1 |
| tensorflow-cpu | 2.12.0 |

The launcher and `stats/runtime.json` refer to the provisioned hep2 software
installation. The fit runtime also checks that installation explicitly.
**This is not a portable environment installer**: on another host those
software paths and runtime checks must be ported together before running.
Do not add an older analysis checkout to `PYTHONPATH`.

Build the local correction/ID artifacts once after provisioning, and again
after changing their Python sources:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/compile_definitions.py
```

Outputs: `TROTASR/utils/compiled/ids.coffea`,
`corrections.coffea`, and `manifest.json`.
These are generated locally, not downloaded as opaque substitutes for source.

## 2. Prepare the required inputs

All runtime analysis paths must resolve **inside TROTASR**. An external
symlink is not an accepted input path.

| Required input | Expected location / content |
|---|---|
| Event files | `inputs/2024/*.root`, `inputs/2025/*.root` |
| Event trees | `Events`, `TROTA`, their completion metadata, and `TopMixedResults_metadata` |
| Top/W truth | `TopWTruth` and matching original object identities, when present |
| Normalization | `estimations/normalization_2024.json.gz`, `normalization_2025.json.gz` |
| Measured b/Top/W efficiencies | `estimations/btageff<year>.merged`, `topwtageff<year>.merged` |
| Calibration payloads | Files identified by `jsons/assets.json`, `scales/JME/manifest.json` and `scales/object_shapes.json` |
| Resolved/mixed models | Exact HDF5 filenames and checksums in `models/TROTA/manifest.json` |
| Low-dM model | `gnn4lowdm/diagonal_v3_numpy.npz`, `config.json`, `selection.json` |
| Signal grid and cross sections | `jsons/signal_grid_955.json`, `stats/stop_xsec_13p6TeV.json` |

The ROOT files are the integrated intermediate products, **not raw NanoAOD**.
Events and reconstructed candidates must belong to the same file and join
through `file_id` and `entry`. Input metadata may be stored inside
`TROTASR_metadata`; older inputs use their existing adjacent JSON metadata.

Intermediate production is an upstream stage, not performed by the histogram
commands below. Provision completed products from the production campaign;
do not reconstruct Events/TROTA merely to redraw a plot or replace a weight SF.

For a file missing only Top/W correction inputs, the adopted treatment is
Top/W SF nominal/Up/Down = 1 for that file, recorded in its histogram audit.
This does not excuse missing Events/TROTA or invalid normalization.

## 3. Produce histograms

### One integrated file

This example processes both branches in one input traversal:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_highdm_histogram.py \
  --input TROTASR/inputs/2024/mc_shard_00000.root \
  --year 2024 --mode both \
  --output TROTASR/campaigns/example_2024/outputs/mc_shard_00000.json.gz
```

The output contains `highdm` and `lowdm` histograms, sumw/sumw2,
physical distributions, background-measurement inputs and the input/weight
audit. A successful single-file output has status
`complete_one_file_test`; that historical status name does not mean a
whole campaign is complete.

### Systematic variations for one file

Choose **one** of these commands for its stated purpose; do not add the
three nominal histograms together.

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_systematics.py \
  --input TROTASR/inputs/2024/mc_shard_00000.root --year 2024 \
  --variation all_weights \
  --output TROTASR/campaigns/example_weights/outputs/mc_shard_00000.json.gz

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_systematics.py \
  --input TROTASR/inputs/2024/mc_shard_00000.root --year 2024 \
  --variation all_weights --cms-trota-jme \
  --output TROTASR/campaigns/example_jme/outputs/mc_shard_00000.json.gz

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_systematics.py \
  --input TROTASR/inputs/2024/mc_shard_00000.root --year 2024 \
  --variation nominal --stored-trota-nonjme \
  --output TROTASR/campaigns/example_nonjme/outputs/mc_shard_00000.json.gz
```

| Command | Work performed |
|---|---|
| `--variation all_weights` | Nominal and all adopted weight Up/Down variations; stored candidates reused |
| `--cms-trota-jme` | Corrected-central nominal, JES/JER and weight variations; jet-dependent TROTA reconstruction |
| `--stored-trota-nonjme` | Electron/photon scale and smearing, muon scale and resolution, tau energy scale and unclustered MET: 16 endpoints; stored jet/TROTA reconstruction retained |

The non-JME path reevaluates object selections, cleaning, recoil, GNN and SFs.
It does not create shifted event ROOT files.

### A full campaign

A full run requires an input inventory, not a directory glob. The inventory
contains one task per integrated ROOT file: `key`, `year`, internal
`input` path, `entries`, and the file's `fingerprint` (inode, size,
mtime_ns); existing adjacent metadata has its own path and fingerprint.
It also records the complete file counts by year, scope and the representative
input check used to prepare that campaign. Use the inventory delivered with
the production inputs; do not invent file counts or reuse another campaign's
fingerprints.

For a **new** campaign, replace the inventory filename and new campaign name:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/nominal_campaign.py prepare \
  --manifest TROTASR/jsons/my_input_inventory.json \
  --campaign TROTASR/campaigns/my_campaign \
  --workers 20 --chunk-size 2000
```

The inventory's `scope`, `shape_mode` and `variations` select what is run:

| Run | `scope` | `shape_mode` | `variations` |
|---|---|---|---|
| Nominal | `full_nominal_production` | omitted | `["nominal"]` |
| Stored nominal + weights | `full_weight_systematic_production` | omitted | `["all_weights"]` |
| JME central + JES/JER + weights | `full_cms_trota_jme_production` | `cms_trota_jme` | `["all_weights"]` |
| Non-JME shapes | `full_nonjme_systematic_production` | `stored_trota_nonjme` | `["nominal"]` |

The full-campaign inventory must also reference its existing representative
check record (`validation_gate`), with matching source/payload contracts and
coverage of all three signal topologies. A successful one-file command alone
does not create that record. Keep it with the campaign, not in a separate
top-level validation directory.

Then select the controller matching the inventory's scope:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/nominal_campaign.py run \
  --campaign TROTASR/campaigns/my_campaign
```

For a systematic campaign, use this controller instead; it also merges both
years after histogram completion:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/run_systematic_campaign.py \
  --campaign TROTASR/campaigns/my_campaign
```

For nominal-only histogram → merge → background measurement, use
`workflows/run_nominal_chain.py --campaign ...` through the same launcher
instead. Do not launch two controllers for the same campaign.

There is no command in the published package that discovers raw NanoAOD and
creates a complete production inventory automatically. Input provisioning and
the campaign inventory/check record must be supplied before this full-campaign
entry point can run. The single-file examples above do not require that
full-campaign inventory.

## 4. Merge and measure backgrounds

For an existing campaign, merge one year at a time if the controller has not
already done so:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/merge_nominal.py \
  --campaign TROTASR/campaigns/my_campaign --year 2024
```

Repeat for 2025. Outputs are `merged/nominal_<year>.json.gz` for nominal
campaigns or `merged/systematics_<year>.json.gz` for systematic campaigns,
with adjacent merge summaries.

The existing combined-systematics example uses
`campaigns/systematics_20260924` for JME and
`campaigns/nonjme_20260927` for non-JME. Once both components are complete:

```sh
for year in 2024 2025; do
  bash TROTASR/workflows/run_inference.sh \
    TROTASR/workflows/merge_nominal.py \
    --campaign TROTASR/campaigns/systematics_20260924 \
    --combine-with TROTASR/campaigns/nonjme_20260927 \
    --year "$year" \
    --config TROTASR/jsons/combined_systematics_config.json \
    --output "TROTASR/campaigns/nonjme_20260927/combined/merged/systematics_$year.json.gz"
done
```

Measure background factors from the resulting histograms:

```sh
for year in 2024 2025; do
  bash TROTASR/workflows/run_inference.sh \
    TROTASR/workflows/build_background_estimation.py \
    --hists "TROTASR/campaigns/nonjme_20260927/combined/merged/systematics_$year.json.gz" \
    --output "TROTASR/campaigns/nonjme_20260927/combined/measurements/$year"
done
```

Outputs include `rz_high.json.gz`, `rz_low.json.gz`, `sgamma.json.gz`,
`double_ratio.json.gz`, transfer-factor inputs and
`measurement_manifest.json`. These are computed from the current
histograms; this stage does not reread NanoAOD or repeat TROTA.

## 5. Run templates, limits, impacts and plots

For the existing combined configuration, this is the downstream controller:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/run_nominal_products.py \
  --campaign TROTASR/campaigns/nonjme_20260927 \
  --config TROTASR/jsons/combined_systematics_config.json \
  --output TROTASR/stats/combined_systematics_sr118 \
  --worker-budget 40
```

It follows the dependencies: background measurements → grid/templates/cards
→ limits, impacts and CR-only fit → numerical recovery where configured →
prediction plots, contours, impacts and pulls. Existing valid outputs are
reused.

`--worker-budget` is this chain's shared allowance, not a separate limit
for each fit pool. On hep2, all of this user's simultaneous compute,
controllers and transfer workers share the 100-slot limit.

For a new result directory, use a copied configuration under `jsons/` with
explicit histogram and measurement paths for both years. Preserve the adopted
`statistical_revision`, `sr_merge`, `systematic_combination`, signal
grid and exclusions unless intentionally changing the analysis model.
Do not point new input histograms at an old finished fit directory.

| Output | Location under `stats/combined_systematics_sr118/` |
|---|---|
| Downstream progress | `products_state.json` |
| Four template ROOT files, cards and grid inventory | `grid/`, including `grid/manifest.json` |
| Limits, point states and command histories | `limits/` |
| Default T2tt(1200,500) impacts | `impacts/` |
| T2bW/T2tb(1200,500), (1000,800) impacts | `impacts_benchmarks_20261001/` |
| CR-only fit and pulls | `cronly/` |
| Yearly CR distributions and background measurements | `plots/<year>/` |
| Combined SR/CR predictions | `plots/predictions/` |
| T2tt, T2bW and T2tb contours | `plots/contours/` |

## 6. Run one downstream stage

Use these when the preceding products already exist; running the whole chain
is unnecessary.

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/build_nominal_grid.py \
  --config TROTASR/jsons/combined_systematics_config.json \
  --output TROTASR/stats/combined_systematics_sr118/grid \
  --preserve-unsupported-templates

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/combine_limits.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/limits --workers 20

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/combine_impacts.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/impacts \
  --model T2tt --mass mStop1200_mLSP500 --workers 20

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/combine_cronly.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/cronly
```

The grid builder writes all mass-point cards and four template ROOT files.
`--preserve-unsupported-templates` retains genuine zero-support
histograms and reports their fit compatibility; it does not manufacture
nonzero bins.

To draw existing results without repeating event processing or fits:

```sh
bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/plotting_card_predictions.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/plots/predictions

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/plotting_limit_contour.py \
  --limits TROTASR/stats/combined_systematics_sr118/limits \
  --grid TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/plots/contours

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/combine_impacts.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/impacts --plot-only

bash TROTASR/workflows/run_inference.sh \
  TROTASR/workflows/combine_cronly.py \
  --manifest TROTASR/stats/combined_systematics_sr118/grid/manifest.json \
  --output TROTASR/stats/combined_systematics_sr118/cronly --plot-only
```

For an explicitly partial contour, add `--allow-partial`. Only triangles
supported by three verified mass points are drawn; holes are not filled by
extrapolation. Keep ROOT files on the server. For laptop plotting,
`plotting_card_predictions.py --export-only` produces compact inputs;
render them with the same command's `--input` option.

## Resume and diagnose

Read the stage that actually owns the work:

| Stage | State / useful evidence |
|---|---|
| Histograms | `campaigns/<name>/state.json`, `state/`, `logs/`, `plan.json` |
| Merge | `campaigns/<name>/merged/*summary.json` |
| Background measurement | `measurement_manifest.json` in its output directory |
| Downstream chain | `stats/<name>/products_state.json` |
| Limit point | `limits/` point state, accepted ROOT/log and command history |
| Impacts | `impact_status.json` and individual nuisance fit logs |
| CR-only | `cronly/state.json` |
| Figures | `plot_manifest.json`; rendered pages still need visual inspection |

The histogram controller can resume unchanged work with the same campaign
path; `prepare` is for a new campaign and refuses an existing directory.
The downstream products controller is different: a retained interrupted stage
requires inspection of its child processes before resumption. Simply rerunning
`run_nominal_products.py` does not reset failed or interrupted stages.
Do not delete a plan, contract or successful point to make a retry run.
A payload/source change needs a recorded input change, not a renamed hash.
The `--refresh-year 2025` options mean **apply the approved 2025 payload
update to the existing histogram campaign**; they are not generic rerun flags.

An empty process queue is not completion. Check the stage result, expected
outputs and accepted fit record. Numerical warnings are diagnostic information,
not by themselves failed limit results.

## Analysis definitions

| File | Contents |
|---|---|
| [`utils/ids.adl`](utils/ids.adl) | Leptons, photons, taus, isolated tracks, AK4/AK8 jets and top/W candidates |
| [`utils/event_selections.adl`](utils/event_selections.adl) | Triggers, data quality, SR/CR selections, 21 high-dM classes, six low-dM classes and likelihood CR bins |
| [`utils/corrections.adl`](utils/corrections.adl) | Calibration keys, SF working points, application domains, weight formulas and systematic endpoints |

The ADLs use the same object/region notation as the AN. They describe the
analysis; the production commands execute Python, not a separately installed
CutLang engine. NanoAOD object names refer to the corresponding calibrated
columns in the intermediate event view.

The externally calculated quantities have specific implementations:

| ADL quantity | Python source |
|---|---|
| `tightLeptonVetoID` | Stored `jet_id_all` / `fatjet_id_all`; correctionlib AK4/AK8 TightLeptonVeto ID |
| `jetVetoMap`, certified luminosity, valid MET | Input production flags and `utils/shape_kinematics.py` event-view rebuilding |
| `TopMixed1pct`, `TopResolved1pct` | Events branches and same-file TROTA tree; `utils/reader.py` |
| `SR/GCR/DY2E/DY2M_Nbst,Nmix,Nres,Nw` | `utils/event_selections.py:topology_counts`, with `arbitrate` and constituent footprints |
| `highdm_mtb` | `utils/region_kinematics.py:jet_kinematics` |
| `gnn_score` | `gnn4lowdm/region_features.py` and `gnn4lowdm/inference.py` |
| Event-weight factors | `utils/weight_components.py`, `utils/private_scales.py`, `utils/corrections.py` |

Object momenta, region cleaning and the GNN score are evaluated separately
for each applicable systematic event view. The numerical bin definitions
are in `jsons/highdm_sr_binning.json`, `jsons/sr_category_merge.json`
and `gnn4lowdm/config.json`.

## Updating a payload

1. Install the approved payload in its existing `scales/<POG>/<year>/`
   location and update the matching inventory entry.
2. Recompile only if ID/correction Python sources changed.
3. Recalculate the affected event weights or kinematics through the existing
   histogram entry points. A photon CSEV or b-tag SF change does not by itself
   require new NanoAOD production or a new TROTA model.
4. Replace the affected central histogram products through their campaign's
   update path, then merge, remeasure affected background factors and regenerate
   the dependent templates/cards/fits. Unaffected valid products remain reusable.

Do not publish a sidecar as though it had updated the central histogram.
