# Workspace cleanup record — 2026-07-23

## Policy

The repository keeps source code, configuration, tests, validation/reporting
logic, compact audit metadata, and the currently published `docs/` website.
Date-stamped Condor campaign directories and reproducible runtime products stay
on EOS and are excluded from Git.

## Preserved active work

- `workflow/intermediate_2024_fullselection_v2_20260723`
  - nominal cluster `962609`
- `workflow/shape_hists_2024_fullselection_v4_20260723`
  - all-40 variation cluster `962612`
- `workflow/intermediate_2025_data_objectcorr_v1_20260721`
- `workflow/nominal_plots_2024_current_20260723`

## Recoverable quarantine

The following superseded or reproducible runtime products were moved, not
deleted, to:

`/eos/user/t/taiwoo/run3_stop/decaf_cleanup_quarantine_20260723`

- duplicated `_codex_fullselection_stage_20260723`
- Python workflow cache
- superseded variation campaigns v1, v2, and v3
- old JES/MET balanced work directories, copied runtimes, and render cache
- completed 2026-07-10 missing-histogram recovery workspace

The quarantine occupies approximately 2.5 GB. Removing it later requires an
explicit deletion decision; until then every moved item is recoverable.

## Git exclusions

The root `.gitignore` excludes:

- credentials and proxies;
- ROOT, NumPy, pickle, archive, and Condor runtime products;
- generated scale-factor and trigger-efficiency galleries;
- date-stamped campaign workspaces;
- the local analysis-note PDF, which is not published.

Selected public plots are maintained under `docs/`.
