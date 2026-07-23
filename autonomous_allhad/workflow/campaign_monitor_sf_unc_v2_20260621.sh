#!/usr/bin/env bash
set +e
set +u
set +o pipefail 2>/dev/null || true

REPO=/eos/user/t/taiwoo/run3_stop/decaf
PY=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
TAG=sf_unc_v2_20260621
LOCK=/tmp/campaign_monitor_${TAG}.lock
LOG=${REPO}/autonomous_allhad/workflow/campaign_monitor_${TAG}.tmux.log

module load lxbatch/eossubmit 2>/dev/null || true
export _condor_CONDOR_HOST="tweetybird04.cern.ch, tweetybird03.cern.ch"
export _myschedd_POOL=eossubmit

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitor already running for ${TAG}" >> "${LOG}"
  exit 0
fi

cd "${REPO}" || exit 1
while true; do
  "${PY}" autonomous_allhad/workflow/campaign_monitor_${TAG}.py >> "${LOG}" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitor sleeping 10800 seconds" >> "${LOG}"
  sleep 10800
done
