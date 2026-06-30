# Run-3 All-Hadronic Stop Analysis

This repository contains the CMS Run-3 all-hadronic stop analysis workflow.
The active analysis path is the `autonomous_allhad` workflow. The older
`analysis` and `automation` directories are still present because the active
workflow reuses the Coffea processor, metadata, corrections, and legacy
production utilities.

Main working copy:

```bash
cd /eos/user/t/taiwoo/run3_stop/decaf
```

Main Python environment:

```text
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
```

For autonomous workflow commands, use:

```bash
PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf:/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad \
  ./autonomous_allhad/analysisctl <command> --config autonomous_allhad/configs/run3_2024.yaml
```

## Repository Layout

```text
analysis/
  processors/stop_processor_v4.py      Legacy Coffea processor source.
  metadata/KNU_2024_v4.json.gz         2024 dataset metadata.
  datasets/                            Dataset lists used by legacy and production paths.
  data/, utils/, helpers/              Corrections, IDs, helper functions, local payloads.
  hists/, plots/                       Legacy histogram and plot outputs.
  combine/                             Combine outputs from completed interpretations.

automation/
  cli.py                               Legacy wrapper for validation, local run, Condor submit,
                                       reduce/merge/scale.
  config.py, paths.py, validation.py   Legacy workflow plumbing.

autonomous_allhad/
  analysisctl                          Main command entry point for the active workflow.
  configs/run3_2024.yaml               Active analysis configuration.
  autonomous_allhad/
    real_subset_worker.py              Event loop, object definitions, weights, shifts,
                                       boosted taggers, feature rows.
    full_production_worker.py          Shard worker and search-bin bookkeeping.
    pipeline.py                        Production, signal processing, merging, plotting,
                                       datacards, limits, publication orchestration.
    report_pages.py                    HTML/report helpers.
  spec/                                Analysis, object, region, trigger, recoil, background,
                                       and combine model YAML specs.
  outputs/                             Compact JSON/CSV/NPY products consumed by later stages.
  workflow/                            Condor submit areas, shard manifests, merge previews,
                                       repair scripts, campaign monitors.
  datacards/, limits/, cards/          Statistical model products.
  plots/, reports/, studies/           Validation plots, reports, and category studies.

docs/
  GitHub Pages payload. Public plots and JSON summaries are copied here before publication.

condor/, condor_log/
  Legacy Condor submit files and shared Condor logs.
```

## Active Analysis Configuration

The active 2024 configuration is:

```text
autonomous_allhad/configs/run3_2024.yaml
```

It defines:

```text
analysis.name       run3_2024_allhad_stop
analysis.year       2024
luminosity          109.82 fb^-1
metadata            analysis/metadata/KNU_2024_v4.json.gz
baseline config     configs/stop_2024.yaml
tree                Events
```

The active command surface is:

```bash
./autonomous_allhad/analysisctl --help
```

Important stages:

```text
run-production                 Build data/background feature shards.
full-production-normalization  Normalize full production outputs.
discover-signals-from-das      Build signal manifests from DAS.
parse-signal-xsec              Parse stop cross sections.
process-signals                Process SMS FastSim signal yields.
make-feature-yields            Build feature yield summaries.
make-systematic-yields         Build systematic yield summaries.
make-datacards                 Build Combine datacards.
expected-limits                Run expected limits.
publish-github-pages           Publish docs/ to GitHub Pages.
monitor                        Print workflow status.
```

## Current Physics Path

The current interpretation is boosted-only. Resolved categories are not part of
the active category set until the resolved development is finished.

Boosted taggers:

```text
Top tagger branch  FatJet_globalParT3_withMassTopvsQCD
Top WP             > 0.5078
W tagger branch    FatJet_globalParT3_withMassWvsQCD
W WP               > 0.9385
```

Boosted object requirements in `real_subset_worker.py`:

```text
Top: pt > 400, |eta| < 2.0, msoftdrop > 105, top score > 0.5078
W:   pt > 200, |eta| < 2.0, 60 < msoftdrop < 105, W score > 0.9385
```

The active search-bin scheme is `boosted_an_17`, with 17 categories built from
`nb_medium`, `nboosted_top`, `nboosted_w`, and `nboosted_total`:

```text
B0_Nb1
B0_Nb2plus
Nb1_T1plus_W0
Nb1_T0_W1plus
Nb1_T1plus_W1plus
Nb2_T1_W0
Nb2_T0_W1
Nb2_T1_W1
Nb2_T2_W0
Nb2_T0_W2
Nb2_TW_ge3
Nb3plus_T1_W0
Nb3plus_T0_W1
Nb3plus_T1_W1
Nb3plus_T2_W0
Nb3plus_T0_W2
Nb3plus_TW_ge3
```

Any SR/CR plot, categorized plot, datacard, limit, contour, or impact result
must contain both DATA and background MC where the region requires them. A
data-only product is not a valid analysis product.

## Systematics Policy

The active boosted AN17 production uses:

```text
nominal               DATA + background MC
jesTotalUp            background MC
jesTotalDown          background MC
metUnclusteredUp      background MC
metUnclusteredDown    background MC
```

MET unclustered branches are loaded explicitly in `real_subset_worker.py` for
`PuppiMET`, `PFMET`, and `MET`:

```text
*_ptUnclusteredUp, *_phiUnclusteredUp
*_ptUnclusteredDown, *_phiUnclusteredDown
```

Final plots, datacards, limits, contours, and impacts must be made only after
the nominal and all required shifted payloads are present and merged.

## Signal Policy

Signal interpretation uses SMS FastSim NanoAODv15.

Do not mix FullSim anchor samples into FastSim signal yields. The old 140X SMS
files do not contain the required `FatJet_globalParT3_withMassTopvsQCD` and
`FatJet_globalParT3_withMassWvsQCD` branches. FastSim NanoAODv15 files do
contain both branches and are the signal input for the boosted AN17 path.

Signal processing command:

```bash
PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf:/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad \
AUTONOMOUS_ALLHAD_SIGNAL_FULL=1 \
AUTONOMOUS_ALLHAD_SIGNAL_CHUNK=50000 \
AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=1 \
AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=1800 \
XRD_NETWORKSTACK=IPv4 \
  /eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  ./autonomous_allhad/analysisctl process-signals \
  --config autonomous_allhad/configs/run3_2024.yaml
```

Key signal outputs:

```text
autonomous_allhad/outputs/signal_searchbin_yields.json
autonomous_allhad/outputs/signal_yields_by_mass.json
autonomous_allhad/outputs/signal_cutflows.json
docs/data/signal_searchbin_yields.json
```

`signal_searchbin_yields.json` must contain the `boosted_an_17` scheme before
final datacards, limits, contours, or impacts are regenerated.

## Production

Boosted AN17 campaign preparation script:

```bash
bash autonomous_allhad/workflow/prepare_boosted_all_systs_20260629.sh
```

For real Condor submission:

```bash
AUTONOMOUS_ALLHAD_SUBMIT_CONDOR=1 \
  bash autonomous_allhad/workflow/prepare_boosted_all_systs_20260629.sh
```

The boosted AN17 campaign uses:

```text
Data shards: 5 files/shard
MC shards:   25 files/shard
Schedd:      bigbird24
Tag base:    boosted_an17_20260629
```

Monitor Condor:

```bash
condor_q -name bigbird24
```

Monitor autonomous status:

```bash
./autonomous_allhad/analysisctl monitor \
  --config autonomous_allhad/configs/run3_2024.yaml \
  --json
```

Campaign output layout:

```text
autonomous_allhad/workflow/production_shards_<tag>_<shift>/
autonomous_allhad/workflow/production_outputs_<tag>_<shift>/
autonomous_allhad/workflow/condor_<tag>_<shift>/
```

For the current boosted campaign:

```text
autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_nominal/
autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_jesTotalUp/
autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_jesTotalDown/
autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_metUnclusteredUp/
autonomous_allhad/workflow/production_outputs_boosted_an17_20260629_metUnclusteredDown/
```

## Merge, Plots, Cards, Limits, Impacts, Web

The post-production chain is:

```text
1. Verify every required nominal/shift shard is valid.
2. Merge DATA and MC payloads.
3. Build CR and SR plots.
4. Build categorized `boosted_an_17` plots.
5. Build systematic yield summaries.
6. Build datacards.
7. Run expected limits and contour production.
8. Run impacts.
9. Copy products into docs/.
10. Publish GitHub Pages.
```

Primary merge/preview helper:

```text
autonomous_allhad/workflow/build_partial_merge_preview.py
```

Boosted campaign watcher/deploy helper:

```text
autonomous_allhad/workflow/boosted_an17_watch_merge_deploy_20260629.sh
```

The intended public boosted AN17 preview path is:

```text
https://resisov.github.io/run3_stop/partial_merge_preview_boosted_an17_20260629_final/
```

Do not publish a final boosted AN17 result unless:

```text
background nominal and required systematic shifts are merged
FastSim v15 signal yields contain boosted_an_17
SR/CR plots include MC
categorized plots use boosted_an_17
datacards include the required shape/normalization uncertainties
limits, contours, and impacts were regenerated from those datacards
```

## Legacy Coffea Workflow

The legacy workflow is still useful for processor-level checks and old-style
Condor production. Its main config is:

```text
configs/stop_2024.yaml
```

Validation:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli validate \
  --config configs/stop_2024.yaml
```

Local dry run:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli run \
  --config configs/stop_2024.yaml \
  --shift nominal \
  --dataset-prefix TTto2L2Nu_ \
  --max-datasets 1 \
  --max-files 1 \
  --dry-run
```

Legacy postprocessing sequence:

```text
analysis/reduce.py -> analysis/merge.py -> analysis/scale.py
```

The active boosted AN17 analysis should use `autonomous_allhad` unless the task
is explicitly about the legacy Coffea path.

## Git Policy

Do not commit large runtime artifacts:

```text
*.root
*.futures
*.reduced
*.merged
*.scaled
*.processor
*.tgz
condor_log/
```

For documentation-only updates, stage only the intended documentation files:

```bash
git add README.md
git diff --cached --stat
git commit -m "Update analysis workflow README"
git push github master
```
