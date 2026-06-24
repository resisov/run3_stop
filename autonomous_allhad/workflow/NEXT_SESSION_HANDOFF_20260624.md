# Next Session Handoff - 2026-06-24

## Current context
- Working dir: `/eos/user/t/taiwoo/run3_stop/decaf`
- GitHub remote to use: `github git@github.com:resisov/run3_stop.git`; do not push to CERN GitLab `origin` unless explicitly requested.
- Worktree is dirty with many generated outputs; do not revert unrelated changes.

## Recent code changes
- Added shape-production support in:
  - `autonomous_allhad/autonomous_allhad/real_subset_worker.py`
  - `autonomous_allhad/autonomous_allhad/full_production_worker.py`
  - `autonomous_allhad/autonomous_allhad/pipeline.py`
- Nominal JEC correction is applied to DATA and MC.
- `jesTotalUp/Down` JEC uncertainty is MC-only.
- `metUnclusteredUp/Down` is MC-only.
- DATA in a shift job remains nominal for JEC uncertainty and MET unclustered uncertainty.
- Shift production is separate by `--shift` / `AUTONOMOUS_ALLHAD_PRODUCTION_SHIFT`; non-nominal shifts are not merged into nominal yields.

## Validation already run
- `python3 -m py_compile` on the three changed code files: passed.
- py38 `py_compile`: passed.
- `full_production_worker --help`: shows `--shift`.
- In-memory MET shift smoke test: DATA remains nominal, MC uses unclustered branch; passed.
- `git diff --check`: passed.

## Submitted Condor jobs
### MC shape campaigns
All are background MC only, 1567 jobs each:
- `jesTotalUp`: cluster `900143`, outputs `autonomous_allhad/workflow/production_outputs_shape_jesTotalUp_20260624`
- `jesTotalDown`: cluster `900144`, outputs `autonomous_allhad/workflow/production_outputs_shape_jesTotalDown_20260624`
- `metUnclusteredUp`: cluster `900145`, outputs `autonomous_allhad/workflow/production_outputs_shape_metUnclusteredUp_20260624`
- `metUnclusteredDown`: cluster `900146`, outputs `autonomous_allhad/workflow/production_outputs_shape_metUnclusteredDown_20260624`
At submit check: total queried shape jobs 6268, running 319, idle 5949, held 0.

### DATA bad-only rerun
- Original DATA campaign: `sf_unc_v2_data2_20260622`, original cluster `892908` no longer active.
- Existing diagnostic: `autonomous_allhad/workflow/data2_bad_files_by_data_type.json`
- Found 1151 DATA file read failures, all `external_access_blocker=True`, `permanently_skipped=0`, across 975 shards.
- Targeted rerun submitted only for those 975 shards.
- New cluster: `900248`
- At submit check: 975 idle, 0 held.
- Submit dir: `autonomous_allhad/workflow/condor_sf_unc_v2_data2_badonly_20260624`
- Args: `autonomous_allhad/workflow/condor_sf_unc_v2_data2_badonly_20260624/full_production_args.txt`
- Backup dir: `autonomous_allhad/workflow/production_outputs_sf_unc_v2_data2_20260622_resubmit_backup_20260624`
- Wrapper moves pre-existing shard output/checkpoint into backup just before rerun. 156 outputs were already backed up before switching to per-job backup.
- Manifest: `autonomous_allhad/workflow/data_badonly_resubmit_20260624_manifest.json`

## Useful follow-up commands
```bash
cd /eos/user/t/taiwoo/run3_stop/decaf
module load lxbatch/eossubmit
condor_q 900143 900144 900145 900146 900248 -totals
condor_q -constraint '(ClusterId == 900143) || (ClusterId == 900144) || (ClusterId == 900145) || (ClusterId == 900146) || (ClusterId == 900248)' -totals
```

## Next tasks
1. Monitor clusters `900143`, `900144`, `900145`, `900146`, `900248` until outputs complete.
2. Re-check invalid/missing/held outputs for the DATA rerun and shape campaigns.
3. Merge/normalize updated DATA and MC shape outputs once complete.
4. Build systematic yields/templates for datacards.
5. Include fixed `Lumi_2024` lnN = 1.016 in datacards.
6. Build template ROOT files, datacards, run Combine, and produce limit contour.

## User preferences from this session
- User wants concise, proactive action in Korean.
- Use `mplhep` style for contours; no off-shell extrapolation, but use all generated points.
- `hep.cms.label` should only use `llabel`, `rlabel`, `ax=ax` options.
- xlim/ylim for contour match provided example: x 600-1500, y 0-1500.
