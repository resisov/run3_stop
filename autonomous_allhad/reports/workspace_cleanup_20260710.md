# Workspace Cleanup Record

Generated: 2026-07-10

## User Boundary

The user explicitly instructed: do not touch files under `decaf/analysis`.
This cleanup pass did not delete, move, or edit any path under:

```text
/eos/user/t/taiwoo/run3_stop/decaf/analysis
```

## Pipeline Record

The current 2024 pipeline state was recorded in:

```text
autonomous_allhad/reports/run3_2024_pipeline_record_20260710.md
```

## Preserved

Current 2024/high-low-dM pipeline artifacts were preserved, including:

- `autonomous_allhad/autonomous_allhad/flat_ntuple_worker.py`
- `autonomous_allhad/workflow/flat_ntuple_highlowdm_20260704_nominal_outputs`
- `autonomous_allhad/workflow/flat_ntuple_highlowdm_recovery10_20260705_nominal_outputs`
- `autonomous_allhad/workflow/condor_flat_ntuple_highlowdm_20260704_nominal`
- `autonomous_allhad/workflow/condor_flat_ntuple_highlowdm_recovery10_20260705_nominal`
- `autonomous_allhad/workflow/highlowdm_full_20260705_flat_hists.json`
- `autonomous_allhad/workflow/highlowdm_full_20260705_root_inputs.txt`
- `autonomous_allhad/workflow/highlowdm_full_20260707_flat_hists_selected_recoil6_with_nt0_lowdm_variables.json`
- `autonomous_allhad/workflow/highlowdm_full_20260707_flat_hists_selected_recoil6_with_nt0_wsplit_lowdm_variables.json`
- `docs/highlowdm_full_20260705`
- `.github/workflows/pages.yml`
- all `analysis/` paths

## Removed

Removed categories:

- duplicate root-level generated HTML files superseded by `docs/`
- root-level `combine_logger.out`
- root-level `taiwoo.cc`, which was a binary Kerberos/XRootD credential cache, not source code
- Python `__pycache__` directories under `autonomous_allhad`
- stale workflow logs, pid files, tmux logs, and nohup logs
- obsolete 2025 probe records outside `analysis/`
- stale local repair/recovery scratch directories from 20260701
- stale low-dM variable histogram work directories whose merged JSON outputs are preserved
- stale partial-merge preview scratch directories under `autonomous_allhad/workflow`
- stale partial-merge preview static pages under `docs`
- obsolete non-`analysis` direct 2025 Condor helper files under `condor/`
- old handoff/monitoring scratch notes and previous cleanup scratch manifests

Second workflow cleanup pass after explicit user approval:

- removed old `boosted_an17` Condor and production-shard directories from 20260629
- removed old `sf_unc` and shape production-shard directories from 20260622-20260627
- removed old flat-ntuple shape fragment-pilot Condor/output directories from 20260703-20260704
- removed old 20260630 flat-ntuple micro/Condor scratch directories
- removed high/low-dM histogram work scratch directories after preserving the final `highlowdm_full_*.json` outputs
- removed local recovery scratch directories and SR topology scratch plot directories under `workflow/`
- removed old non-code watcher, recovery, retry, probe, monitoring, and partial-merge scratch records

Remaining top-level `workflow/` directories after this pass:

- `condor_flat_ntuple_highlowdm_20260704_nominal`
- `condor_flat_ntuple_highlowdm_recovery10_20260705_nominal`
- `current_available_pipeline_20260702`
- `data_signal_only_20260701`
- `flat_ntuple_highlowdm_20260704_nominal_outputs`
- `flat_ntuple_highlowdm_recovery10_20260705_nominal_outputs`
- `production_shards_highlowdm_recovery10_20260705_nominal`

## Bad-File Manifest

The standard bad-file manifest files were restored as explicit empty manifests:

```text
autonomous_allhad/workflow/bad_files.json
autonomous_allhad/workflow/bad_files.txt
```

This records that this cleanup pass did not identify new bad ROOT files.

## Not Cleaned Automatically

The following were intentionally not removed because they may be source,
reference, or current reproducibility material:

- `AGENTS.md`
- `AN2019_016_v9.pdf`
- `fast_analysis/`
- current high/low-dM and official `analysisctl` reports/data
- current docs dashboard and high/low-dM web output
- all `analysis/` contents
- old `workflow/*.py` and `workflow/*.sh` files that may still be source code; deleting those requires a separate explicit approval for the exact script list
