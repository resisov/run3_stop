# Monitoring Status - 2026-06-24 13:25 CEST

## Code fix
- Fixed 2024 JEC uncertainty key in `analysis/utils/corrections.py`:
  - old: `Summer24Prompt24_V2_MC_Total_AK4PFPuppi`
  - new: `Summer24Prompt24_V3_MC_Total_AK4PFPuppi`
- Direct correctionlib check confirmed the V3 key exists and evaluates.
- py_compile passed with system python and py38 for the edited correction helper.

## Condor action
- Global `jesTotalUp/Down` failures were traced to the stale V2 JEC uncertainty key.
- Removed affected clusters: `900143`, `900144`.
- Resubmitted fixed campaigns:
  - `jesTotalUp`: cluster `906067`, 1567 jobs
  - `jesTotalDown`: cluster `906066`, 1567 jobs
- Existing failed `jesTotal` shard JSONs remain on disk but are invalid under strict validation and will be overwritten by fixed jobs.

## Current queue snapshot
Snapshot time: `2026-06-24 13:25 CEST`.

Relevant clusters queried: `906067 906066 900145 900146 900248`.

- Total queued: `7232`
- Idle: `6372`
- Running: `860`
- Held: `0`

Per campaign:
- `jesTotalUp` cluster `906067`: 1567 idle, 0 running, 0 held
- `jesTotalDown` cluster `906066`: 1567 idle, 0 running, 0 held
- `metUnclusteredUp` cluster `900145`: running and producing outputs
- `metUnclusteredDown` cluster `900146`: still idle/not started
- DATA bad-only cluster `900248`: still idle/not started

## Output integrity snapshot
Strict validation requires final JSON status `complete` or `complete_with_bad_files`, matching shard digest, matching record count, complete attempt accounting, expected shape shift, and `completed_at`.

- `metUnclusteredUp`: 12 final shards, 12 strict-valid; statuses `complete=4`, `complete_with_bad_files=8`; bad files 28, all external access blockers, no permanent skips.
- `metUnclusteredDown`: 0 final shards.
- `jesTotalUp`: old final shards 1389, all `failed`, strict-valid 0.
- `jesTotalDown`: old final shards 742, all `failed`, strict-valid 0.
- DATA bad-only target shards: 975 planned; existing finals 819, missing 156 moved to backup; current existing target finals still contain old external-access bad files and await rerun.

## Proxy
Condor transfer proxy: `/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757`.

- CMS VOMS proxy is valid.
- Time left at check: about 69 hours.
- Valid for 48h check passed; 72h check failed.

## Notes
- Do not merge old `jesTotal` outputs until fixed resubmitted jobs replace them with strict-valid outputs.
- Large generated output directories are intentionally not part of the GitHub status commit.
