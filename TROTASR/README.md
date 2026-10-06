# TROTASR: isolated clean-pipeline reconstruction

## Current directory contract (2026-09-29)

Root-level configuration JSONs have moved to `jsons/`. Do not recreate copies at
the package root. The explicit sixteen-name compatibility map in `utils/paths.py`
resolves old frozen configuration references without changing their contents;
it does not redirect arbitrary missing inputs or outputs.

Systematic execution uses `workflows/run_systematic_campaign.py`; per-file
histogram calculation uses `workflows/build_systematics.py` and the shared
`utils/histogramming.py`. The implementation directly performs file-local
endpoints, region-level GNN reuse and consumer-branch reading. It does not load
production code from a dated validation adapter or transform source with `exec`.
Execution state, outputs, receipts and evidence belong inside the selected
campaign, not in `validation/` or the package root. The hep2 `validation/` and
`tests/` directories were removed by the user and must not be recreated by a
production command. Local development tests are not production entry points.

The non-JME canonical controller started on2026-09-29 at15:19UTC after18 real-file
comparisons and51 remote regression tests passed. New canonical worker execution
was verified at15:22UTC against actual PID/startticks/argv and thread settings;
this is not a completed campaign. Refer to the newest section of
`NOMINAL_CONTINUATION.md` and the actual campaign state for execution status.
The older dated notes below describe historical checkpoints, not current status.

## Historical implementation notes

Status (2026-09-23): **partial implementation with real-file validation**.
The current authorized endpoint is all AN-required nominal plots and new
2024+2025 combined expected T2tt/T2tb/T2bW contours, not just histogram code.
See `completion_contract.json`, `an_figure_inventory.json`, and
`NOMINAL_CONTINUATION.md` for the completion gate and latest execution details.
This is not a completed end-to-end pipeline or an adopted production result.
The user authorized a trial/comparison of Boosted > Mixed > Resolved, followed
by independent W candidates. Full isolated nominal histogram production started
on hep2 at 2026-09-23 15:44 UTC after the representative validation gate passed.
No systematic campaign or actual fit has started; this workflow does not change
original ROOT files or the main analysis. Prior TopMixed integration finished
14,474/14,474 files with zero failures before full histogram dispatch.

The machine-readable requirements are in `selection_contract.json`.

Latest execution issue (2026-09-23 20:24 UTC):8,234/14,474 histograms succeeded;
65 signal files failed because the original normalization omitted the final65
signal shards per year. Normal jobs continue. A separate verified candidate
adds31 disjoint mass points per year without changing any existing factor;
11 focused recovery tests pass. **It is not promoted yet.** The finite recovery
waits for existing controllers to exit, preserves prior results, then retries
only the130 affected shards. See the first continuation section and
`validation/normalization_recovery_20260923/implementation_status.json`.
No full-input fit/contour has been completed.

## Runtime dependency boundary (user instruction, 2026-09-23)

Analysis source, configuration, calibration, efficiency, normalization and model
files must resolve inside this TROTASR directory. Development permission to read
an external source does **not** authorize importing it at runtime. Do not add a
parent repository to `sys.path`, automatically copy from an old campaign, or use
an external symlink. Shared file I/O rejects external paths before opening them;
payload/model loaders and renderer import aliases also have boundary checks.
Origin paths in `sources.json` and `assets.json` are provenance strings only,
not runtime lookup locations.

Workflow entry points bind this package by its internal `__init__.py` without
adding the parent repository to `sys.path`. The old development migration tools
now accept only sources already staged **inside** TROTASR; they do not fetch
anything, and normal workflows never invoke them. Missing assets fail explicitly.
The later 2026-09-23 instruction authorizes necessary legacy inspection and
inward staging while the user is away, without repeated questions. It does not
authorize external analysis imports: all runtime inputs still resolve internally.
Completed hep2 ROOT inputs may be staged as read-only-use hardlinks within
TROTASR after their completion markers/fingerprints are verified. No laptop ROOT.
Ordinary Python/numerical libraries are software dependencies, not a license to
import another analysis checkout.

Full hep2 unit suite: **178 passed** on 2026-09-23. These include
path and symlink escapes, preloaded external renderer-module aliases, the fixed
162-bin map, unchanged renderer bytes and independent Nb normalization bindings.
Yearly and combined LL/QCD/GCR figures now share the AN-adopted native-card
prefit inputs; LL is five bins only in display, while its likelihood stays ten.
GCR event yields are not unit-area. Real legacy LL projection arrays are used
only for regression tests. Reference figure QA is partial:67 copied/inspected,
10 fully verified,47 awaiting provenance checks and10 inherited visual issues.
The approved native-TH1/card functions have been extracted into
`utils/statistical_templates.py`; see `statistics_sources.json` for exact scope.
The legacy campaign reader and its external imports were not copied. The new
`statistics_model.py` contains only approved mathematical rules with explicit
year arguments, and `statistics_inputs.py` connects the current histogram axes.
The tests also execute the template command in `--validate-only` mode, verify
idempotent compact outputs, and reject datacard creation from unverified ROOT
templates. These synthetic tests are not an end-to-end ROOT/Combine validation.

The production Python sources were synchronized and tested on hep2, and
`compile_definitions.py` rebuilt and reloaded IDs/corrections there successfully.
Completed TopMixed integration is unaffected. The laptop bundled Python lacks
Awkward/SciPy; its full-suite import failures are not represented as a pass.

On 2026-09-23 the user adopted **162 high-dM SR bins per year**, comprising
28 categories with topology-dependent Nb pooling/splitting. The exact adopted
18-row definition is `highdm_sr_binning.json`. This is physics-definition
approval, **not a claim of completed histogram/plot implementation or production**.
Low-dM SR remains 30 bins per year; the existing CR binning is unchanged.

## Working and tested now

- `workflows/combine_cronly.py`, `combine_impacts.py`, and `recover_impacts.py`
  now connect the current full-input cards to the existing fit prescriptions.
  CR-only uses exactly twelve channels/156 CR bins and no SR observations;
  impact uses T2tt(1200,500), mu=1 S+B Asimov. Numerical kernels and recovery
  provenance are recorded in `diagnostic_sources.json`. Command attempts,
  failures, output hashes and retry limits persist across restarts. Successful
  outputs are reused; interrupted outputs are retained for explicit recovery.
  Original pull rendering and `plotImpacts.py` are separate `--plot-only` steps;
  PDF generation never claims visual QA. Twenty-three diagnostic tests passed
  locally; the full server suite passed153 tests and the native software-only
  smoke passed (ROOT6.30/07, Python3.9.14, four help commands). Real full-input
  fits have not started. Exact validation status is separately recorded in
  `validation/fit_diagnostics_20260923/implementation_status.json`; implementing
  these runners is not evidence that physical fits have completed.
- `workflows/plotting_nominal_distributions.py`, `plotting_transfer_factors.py`
  and `plotting_background_measurements.py` connect current products to the
  preserved rendering functions. Planned coverage is 72 physical CR, 14 TF and
  50 background-measurement figures over both years, not yet generated on full
  inputs. Nineteen new adapter tests pass locally; the full remote suite has
  130 passes. Three real representative-input physical plots were rendered and
  all three PDF pages visually checked via Poppler. These validation-only
  figures are not AN deliverables or closure results.
  A genuine inherited-display problem is blocked explicitly: the high-dM Njet
  display starts at five, but approved selection includes four jets. The test
  found 52 data events and 29 ST entries excluded by that display map. The source
  histograms retain them. The affected figures are not published, and neither
  the selection nor the immutable renderer was changed. Details and the
  separate inherited mll-highlight/fit-window mismatch are in the continuation.
- `workflows/nominal_campaign.py`, `merge_nominal.py` and
  `validate_nominal_campaign.py`: 16 real ROOT inputs / 428,436 events passed,
  including all three signal topologies and all data streams in both years.
  The SR162 projection preserves yield/variance and no high/low overlap was found.
  Both representative year merges completed. Full production now uses the same
  physics contract, one subprocess per ROOT and aggregate resource admission;
  `run_nominal_chain.py` continues into yearly merge/background measurements.
- `workflows/combine_limits.py` and `limit_validation.py`: adopted finite
  strategy0 and measured-crossing checks, with internal runtime/card guards.
  Both Combine/text2workspace help commands passed; **no real fits yet**.
- `workflows/build_nominal_grid.py` and `signal_grid_inputs.py`: new shared
  native-template grid builder, using current histograms/measurements and all
  924 requested masses (205 T2tt, 354 T2bW, 365 T2tb). Only mass keys, including
  previously failed points, were migrated; no prior fit values are used.
  Full bin-layout, single-point equivalence, negative-bin audit, exact signal
  naming and resumable validation-only tests pass locally. The native writer
  uses CMSSW's Python/PyROOT runtime rather than Python 3.8 extension paths.
- `workflows/plotting_limit_contour.py`: internal boundary adapter to the
  unchanged original renderer. All reference coordinates and cross sections
  are staged internally with hashes. It rejects unaccounted missing points,
  stale fit/grid pairs and unblinded/representative inputs. New contours have
  **not** been produced. The focused local downstream suite has 49 passing tests;
  this is not the full remote physics suite or fit validation.
  The subsequent full remote suite passed 111 tests, and the synthetic native
  ROOT readback test passed for four files, 540 analysis bins and 16 channels
  with three named signal mass points. These are technical validation results,
  not production templates, background measurements, or fits.
- `utils/ids.py`, `corrections.py`, `event_selections.py` and their `.adl`
  companions. ADL files document the definitions; they are not advertised as
  an independently tested CutLang execution engine.
- Source functions and calibration paths are migrated into this package.
  Normal execution does not import an old campaign controller. The development
  migration script is not called by workers. `sources.json` records origins and
  hashes; `assets.json` records 50 copied payload/model/efficiency assets.
- `scales/{JME,BTV,EGM,MUO,LUM,Private}/{2024,2025}/`, `estimations/`, and
  `gnn4lowdm/` contain the inputs. Existing normalization values are unchanged;
  missing signal factors fail rather than being replaced by unity.
- `workflows/compile_definitions.py` compiles and reloads both `.coffea`
  artifacts, with source/artifact hashes checked before correction evaluation.
- High-/low-dM histogram entry points share one bounded ROOT traversal via
  `--mode both`. ROOT Events, TROTA and TopWTruth are read from the same input.
  Histograms contain sumw, sumw2 and entries, not event-level sidecars.
- `workflows/build_systematics.py` supports explicit weight endpoints. Object
  shapes are not implemented/validated and are rejected before execution.
- `workflows/plotting_histogram.py` requires explicit regions. The two original
  renderer source files are byte-identical to their recorded sources. The SR
  adapter uses the original SR style parameters and includes all five count
  axes, including Nw=0. The nominal-only test labels its band MC statistical
  uncertainty; no systematic uncertainty is invented.
- `workflows/build_template.py` and `build_datacard.py` now provide internal
  entry points, not wrappers around a legacy campaign. The template command
  requires histogram-matched, year-matched measured RZ/Qgamma/Sgamma/closure
  payloads. It preserves exact Nb/recoil normalization bindings under the
  162-bin high-dM SR map; low-dM Sgamma uses GNN x recoil cells, never score-bin
  index matching. Unsupported SR-only parameters block output rather than
  changing binning; the adopted CR-only unity policy is separately audited.
  The template writer includes full TH1 read-back checks. Actual remote ROOT
  creation and Combine execution have **not** been run for these new commands.

## Background measurement stage required by the user

`workflows/build_background_estimation.py` is implemented as a real
measurement producer, not an importer of old numerical factors. It uses
histograms made with the current TROTASR selections, including the required
on/off-Z and Nb/recoil measurement inputs, compute RZ, Qgamma, Sgamma and
Z/gamma double ratios, and keep the compact per-estimate products in
`estimations/`. Templates/cards consume these internal products only. Failed
or unsupported measurements must remain explicit; no unity placeholders are
authorized for missing background factors.

Its mathematical measurement functions were migrated from the existing legacy
producer and verified verbatim; `background_sources.json` records the sources.
Synthetic RZ/Q/Sgamma/double-ratio and TF covariance tests pass. Actual full
data/MC measurements have **not run**. The AN/frozen-code difference in the
auxiliary low-dM RZ selection is documented in `NOMINAL_CONTINUATION.md`; the
current implementation preserves the verified frozen producer definition.
No background-measurement results have been invented or marked complete.

## Actual test evidence

On hep2, in a single-threaded process, using immutable completed 2024 inputs:

| Input | Events traversed | Wall time | Result |
| --- | ---: | ---: | --- |
| W+jets `mc_shard_02023.root` | 1,534 | 13.30 s | Corrected low-dM GNN histograms; no SR acceptance in this input |
| Single Top `mc_shard_00078.root` | 14,137 | 78.35 s | High/low histograms and before/after weighted comparison |

For the Single Top input, comparing topology
definitions on **the same non-topology cores** (not comparing old merged bins):

| Selection | B/W-before-Resolved baseline | B>Mixed>Resolved trial |
| --- | ---: | ---: |
| High-dM SR before search mTb cut | 35 | 55 |
| High-dM SR with mTb >= 175 | 10 | 15 |
| Low-dM SR | 785 | 683 |

High/low intersections are zero for all six regions. Four events in this file
retain both a selected Mixed and an independent W before region cuts. This is
not evidence of T2bW sensitivity. This earlier check predated the now-completed
16-file test; current three-signal acceptance comparisons are recorded in
`validation/nominal_campaign_20260923/validation_gate.json`. The older compact
report is `validation/comparison_summary.json`.
The earlier SR rendering in `validation/plots/one_st_2024/` is **withdrawn as
bin-layout evidence**: it displayed only five occupied categories (30 displayed
bins), not a full SR bin map. Its manifest marks it noncanonical. This did not
merge the underlying sparse histogram, but the occupancy-dependent display is
not allowed even for a one-file test. The files remain only as a marked test
artifact, not a physics prediction or an adopted 30-bin definition.

## Fixed high-dM SR layout, including in tests

Reduce the input file count for a test, **never the category/bin definition**.
High-dM SR plotting requires an explicit `--bin-map` for the entire layout. All
categories and recoil bins in that map are retained, including zero-yield bins.
Unknown input categories, changed recoil edges, malformed arrays, duplicate
categories and a mismatched total bin count fail rather than being discarded,
merged or automatically rebinned. The legacy renderer is unchanged.

The 162-bin definition in `highdm_sr_binning.json` is now connected to the
histogram/plot adapter. High-dM SR retains exact Nb and native recoil components
before the fixed projection, while CR Nb pooling is unchanged. The earlier
two-file trial aggregates cannot recover Nb=2 versus Nb>=3 and are not 162-bin
validation outputs. A new 2024 ST file test traversed all 14,137 events in
96.07 seconds with exact-Nb source axes. Its plotting command exited successfully;
visual QA remains pending. Representative multi-file validation subsequently
passed for both years; full production is now running, not completed.

The earlier exact-key test map uses schema `trotasr_full_sr_bin_map_v1`, `scope: full`,
`mode: highdm`, `region: SR`, `observable: recoil`,
`category_axis_order: [Nb, Nbst, Nmix, Nres, Nw]`, and a `categories` list. Each
category has a `key` (the five comma-separated counts, with Nb=2 denoting >=2)
and the exact `edges` array. `total_bins` must equal the sum of the category bin
counts. Categories follow the stated numeric axis order. This format does not
confer physics approval on a map. The smaller fixtures in
`tests/test_full_sr_layout.py` are synthetic unit-test inputs only and are never
used to produce analysis figures. That exact-key schema cannot express the
new topology-dependent Nb partitions or >= multiplicity predicates. The adopted
definition uses a different schema handled by `utils/sr_binning.py`.

## Still required; do not mark these complete

- A clean NanoAOD -> Events + TROTA + TopMixed producer in one main job.
- Validate the implemented background-estimation histograms and
  RZ/Qgamma/Sgamma/double-ratio/TF adapters on sufficient real data/MC coverage.
- Real-data/MC validation of the implemented exact Nb/native-recoil projection
  and template/datacard commands. Synthetic layout/conservation tests are passed;
  these are not fit or physics-performance evidence.
- Independent CR-only/impact runners; real native-template/card and fit tests
  of the implemented shared grid builder and contour boundary adapter.
- Object-shape propagation is outside this nominal stage; do not restart it.
- End-to-end validation of all workflows. A single background file cannot
  provide measured CR constraints or a physical signal limit; no dummy fit is
  being passed off as this validation.

## Reproduce the implemented test on hep2

Use the py38 runtime and numerical dependencies already installed on hep2.
Only numerical libraries are external software dependencies. Stage completed
ROOT inputs inside TROTASR first; the histogrammer rejects outside paths.
Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and the
documented vendor PYTHONPATH. From the parent of TROTASR, run:

```sh
python TROTASR/workflows/compile_definitions.py
python -m unittest discover -s TROTASR/tests -v
python TROTASR/workflows/build_highdm_histogram.py --year 2024 --mode both \
  --input TROTASR/inputs/approved_input.root --output TROTASR/validation/test.json.gz
```

The histogrammer will reject incomplete/mismatched ROOT markers, missing
normalization, stale compiled corrections, changed payloads, nonfinite weights,
or an existing output with a different contract. It does not submit a campaign.

## User-directed changes

1. SR category axes: **Nb, Nbst, Nmix, Nres, Nw**, in that order.
2. Selected Mixed and counted W objects must not overlap. Select the Mixed
   collection, remove overlapping W candidates, and count the remaining W
   candidates. An independent W may coexist with a Mixed candidate; do not set
   Nw to zero merely because Nmix is positive. A shared original AK8 identity
   is unambiguously an overlap. The complete constituent/geometric predicate
   must be documented and tested using the adopted cleaning rules before
   production; no new angular threshold is adopted here.
3. High-dM tag union becomes
   `Nbst >= 1 || Nmix >= 1 || Nres >= 1 || Nw >= 1`, with the existing `Nj >= 4`
   and other region-specific cuts retained.
4. Low-dM topology veto becomes
   `Nbst == 0 && Nmix == 0 && Nres == 0 && Nw == 0`, with the existing
   SR/CR-specific non-topology cuts retained.
5. Histogramming may use the **100-core aggregate hep2 budget**, including
   other campaigns and controller/merge/transfer reservations. With no other
   work, this permits 99 single-threaded histogram workers plus one controller
   slot, subject to memory and I/O safety. This does not increase the running
   inference campaign from 50 workers.

The new high-dM union replaces the old union; it must not merely be ANDed
onto a precomputed region flag that already rejects Mixed-only events. Apply
the new counts consistently to SR and the corresponding CR/preselection/VR
topology decisions. Region-specific lepton/photon cleaning must agree between
high- and low-dM masks. Audit both intersections and acceptance migrations.

## Unchanged workflow boundaries

- Read nominal Mixed candidates from the original Events tree: no repeated
  nominal Mixed inference and no event-level sidecars. `nTopMixed1pct` is a raw
  passing-candidate count, not the exclusive analysis Nmix.
- Keep CR normalization streams at Nb=1 and Nb>=2 with the adopted year/recoil
  scopes. Do not replicate SR topology categories as CR categories.
- Retain the low-dM 30-bin SR and the NISR-pooled 10-bin LLCR/QCDCR/GCR axes.
  Recompute yields for the new selection; do not reuse old numerical templates
  as though they represented the Mixed veto.
- Read each nominal input in bounded chunks, preserve unmerged sparse SR
  category/recoil histograms, and do later merges/plotting from those aggregates.
- Use the user-adopted 162-bin high-dM SR map without occupancy-dependent
  merging. Inspect MC statistics and all three signal topologies; report any
  required physics change for approval rather than silently changing the map.
- Reuse existing plotting and statistical functions; retain SR blinding and
  adopted correction policies. No Mixed-specific SF is invented.
- Deliver nominal histograms and SR/CR figures before any future systematic
  restart. Heavy downstream steps remain separate from this design update.

## Physics gates before production

Boosted/Mixed/Resolved candidate arbitration is implemented for the approved
trial. Representative signal acceptance and fixed-map conservation passed in
both years; full-grid sensitivity and fit validation are still pending. Tests cover
a shared Mixed/W AK8, an independent W in a Mixed event, multiple competing
Mixed candidates, region-specific object cleaning, Mixed-only high-dM
acceptance, low-dM Mixed veto, zero high/low intersections, and process-wise
acceptance changes including T2bW. Missing/invalid Mixed inputs must not be
silently treated as Nmix=0.

## Nominal continuation and plot validation (2026-09-23)

The live full campaign and current blockers are recorded in
`NOMINAL_CONTINUATION.md`; older design status strings are not a live dashboard.
`workflows/run_nominal_products.py --campaign campaigns/nominal_20260923
--output stats/nominal_20260923` is the downstream controller. It waits for the
existing histogram/merge/measurement chain, then runs the new native grid,
limits/impact, CR-only diagnostics and original-renderer adapters. It never
resubmits histograms. Do not start another copy while its lock/PID is alive.
Its47+47 fit-worker split plus controllers and one auxiliary stage is bounded
by the global100-slot cap. Failed stages and interrupted children require an
explicit audited recovery, not automatic repeated submission.

New prediction adapters are `plotting_gnn_controls.py` (yearly CR shapes) and
`plotting_card_predictions.py` (combined blinded SR/prefit pooled CR). The latter
reads the full new native-card grid, applies declared rate initials once and
retains correlated nuisance endpoints. SR observations are never exported.
The original lowSR and GNN drawing bodies and source hashes are preserved under
`workflows/renderers` and `prediction_plot_sources.json`. All175 remote regression
tests passed. This is implementation validation, not full-input fit completion.

HighSR162 currently fails visual QA because the unchanged legacy layout cannot
legibly fit the detailed28-category labels/bin ticks. Do not shrink the bin map
or silently alter style. The inherited highNjet display also omits selectedNj4
events and is explicitly blocked. Unaffected work continues. Two updated region
schematics include the approvedNmix condition with all other TikZ source
byte-preserved; their localPDF/PNG pages passed Poppler visualQA. The corrected
AN inventory contains156 current-analysis items and67 unchanged references,
223 total. No figure is automatically inserted into the AN.
