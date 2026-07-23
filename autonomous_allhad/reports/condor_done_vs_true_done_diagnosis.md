# Condor Done vs True Shard Done Diagnosis

Generated: `2026-06-18T11:52:27Z`

## Root Cause

Condor queue completion and shard-output completion were being mixed. The worker used atomic writes, but it wrote `status: running` checkpoints directly to the final `shard_XXXXX.json` path at job start and after each file. A retry could therefore replace a previously valid final JSON with a running checkpoint. The scheduler could then show the job as gone, removed, or done while the final JSON was still missing, zero-byte, running, failed, stale, or inconsistent.

## Worker Write Semantics Found

- Before the patch, `write_json()` wrote through a hidden temp file and `os.replace()`, so individual writes were atomic.
- Before the patch, the initial running payload and per-file running checkpoints were written to the final output path.
- `completed_at` was set only at the end.
- Final statuses were `complete` when all attempted files processed, `complete_with_bad_files` when at least one file processed but not all, and `failed` when no file processed.
- If a retry started against an already complete shard, the old worker did not preserve that final output before writing a running checkpoint.

## Patch Summary

- full_production_worker.py now writes progress to shard_XXXXX.json.running and reserves shard_XXXXX.json for final complete/complete_with_bad_files/failed payloads.
- full_production_worker.py skips an existing valid final output before processing and rechecks before final replace, preventing future retries from touching completed shards.
- full_production_worker.py writes through `<target>.tmp.<pid>` and then `os.replace()`; final `shard_XXXXX.json` is replaced only after the JSON is fully written, flushed, fsynced, and closed.
- pipeline.py strict validation now checks status, digest, records_in_shard, files_attempted/files_processed, completed_at, and parser success before skip/merge acceptance.
- pipeline monitor renders condor_done_vs_true_done_diagnosis.json separately from normal pipeline status.

## Current Strict State

- Total shards: `2110`
- Truly complete shards: `0`
- Incomplete shards: `2110`
- Active Condor jobs: `689`
- Active-job-covered shards: `689`
- Terminal-invalid shards not covered by active jobs: `1421`
- Merge can start: `no`

## Queue Totals

```json
{
  "784515": {
    "active": 353,
    "held": 0,
    "idle": 0,
    "in_queue": 353,
    "other": 0,
    "removing": 0,
    "running": 353,
    "suspended": 0,
    "transferring": 0
  },
  "784518": {
    "active": 334,
    "held": 0,
    "idle": 0,
    "in_queue": 334,
    "other": 0,
    "removing": 0,
    "running": 334,
    "suspended": 0,
    "transferring": 0
  },
  "784519": {
    "active": 0,
    "held": 0,
    "idle": 0,
    "in_queue": 0,
    "other": 0,
    "removing": 0,
    "running": 0,
    "suspended": 0,
    "transferring": 0
  },
  "784521": {
    "active": 0,
    "held": 0,
    "idle": 0,
    "in_queue": 0,
    "other": 0,
    "removing": 0,
    "running": 0,
    "suspended": 0,
    "transferring": 0
  },
  "784522": {
    "active": 2,
    "held": 0,
    "idle": 0,
    "in_queue": 2,
    "other": 0,
    "removing": 0,
    "running": 2,
    "suspended": 0,
    "transferring": 0
  },
  "784535": {
    "active": 0,
    "held": 0,
    "idle": 0,
    "in_queue": 0,
    "other": 0,
    "removing": 0,
    "running": 0,
    "suspended": 0,
    "transferring": 0
  }
}
```

## Classification Counts

```json
{
  "active_condor_job_wait": 689,
  "condor_done_zero_byte": 1421
}
```

## Affected Shard Examples

- running checkpoint in final path: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] ...
- zero-byte final JSON: [52, 54, 112, 125, 126, 157, 166, 211, 213, 227, 231, 232, 234, 236, 238, 239, 248, 262, 289, 299] ...
- terminal invalid uncovered: [52, 54, 112, 125, 126, 157, 166, 211, 213, 227, 231, 232, 234, 236, 238, 239, 248, 262, 289, 299] ...
- valid final included in 784535 args: []
- active retry covering valid final output: []

## Regression Snapshot

- Current snapshot: `autonomous_allhad/workflow/shard_status_snapshot_1781783546.json`
- Previous snapshot compared: `None`
- If no previous snapshot is listed, exact historical complete-to-running regressions cannot be proven from the current filesystem alone; the code path and retry coverage explain how the count could decrease.

## Remaining Active Clusters

Active clusters are the clusters with nonzero `active` jobs in the queue totals above. Removed retry `784535` is terminal removed by user according to `condor_history`.

## Safe Next Monitoring Command

```bash
module load lxbatch/eossubmit && cd /eos/home-t/taiwoo/run3_stop/decaf && condor_q 784515 784518 784519 784521 784522 784535 -totals
```
