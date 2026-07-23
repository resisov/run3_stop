#!/usr/bin/env bash
set -euo pipefail
module load lxbatch/eossubmit
cd /eos/user/t/taiwoo/run3_stop/decaf
export PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad:${PYTHONPATH:-}
export AUTONOMOUS_ALLHAD_ALLOW_CONDOR=1
export AUTONOMOUS_ALLHAD_SUBMIT_CONDOR=1
export AUTONOMOUS_ALLHAD_FULL_SHARD_SIZE=25
export AUTONOMOUS_ALLHAD_PRODUCTION_TAG=sf_unc_v2_20260621
exec /eos/user/t/taiwoo/miniconda3/envs/py38/bin/python ./autonomous_allhad/analysisctl run-production --config autonomous_allhad/configs/run3_2024.yaml
