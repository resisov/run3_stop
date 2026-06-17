# Full Production Status

Status: `blocked_not_submitted`

Full production was not submitted; file-level manifests were built and the exact production blockers were recorded.

Configured datasets: 5331
DATA files: 13576
Background files: 39172
FastSim signal files: 61
FullSim signal anchor files skipped: 47
Planned jobs: 52809

## Blockers

- execution.allow_condor_submit is false in autonomous_allhad/configs/run3_2024.yaml
- no autonomous Condor execute/monitor/retry/merge implementation exists for the 52k-file feature-table production campaign

## Exact unblock steps

- Enable a reviewed autonomous Condor campaign configuration before submitting full DATA/background production.
- Implement and validate an autonomous Condor shard runner that writes per-file feature/yield/hist shards, plus merge and retry stages, before submitting the 52k-file campaign.
