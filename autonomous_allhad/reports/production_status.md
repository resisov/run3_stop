# Full Production Status

Status: `blocked`

Full production was not submitted. This stage only built the production manifest and blocker report.

Configured datasets: 5331
Files in metadata: 52772
Planned jobs: 52772

## Blockers

- execution.allow_condor_submit is false in autonomous_allhad/configs/run3_2024.yaml
- no explicit bounded local full-production fallback is configured

## Exact unblock steps

- Set execution.allow_condor_submit: true only after confirming a valid CERN proxy and intended HTCondor production campaign, then rerun ./autonomous_allhad/analysisctl run-production --config autonomous_allhad/configs/run3_2024.yaml
- Alternatively add an explicit production.local_max_files/local output policy and rerun; do not treat the 14-file validation subset as full production.
