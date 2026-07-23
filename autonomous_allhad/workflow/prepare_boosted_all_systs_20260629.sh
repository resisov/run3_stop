#!/usr/bin/env bash
set -euo pipefail

REPO="/eos/user/t/taiwoo/run3_stop/decaf"
CONFIG="autonomous_allhad/configs/run3_2024.yaml"
TAG_BASE="${AUTONOMOUS_ALLHAD_PRODUCTION_TAG_BASE:-boosted_an17_20260629}"
SUBMIT="${AUTONOMOUS_ALLHAD_SUBMIT_CONDOR:-0}"

cd "$REPO"
module load lxbatch/eossubmit >/dev/null 2>&1 || true

export AUTONOMOUS_ALLHAD_ALLOW_CONDOR=1
export AUTONOMOUS_ALLHAD_SUBMIT_CONDOR="$SUBMIT"
export AUTONOMOUS_ALLHAD_DATA_SHARD_SIZE=5
export AUTONOMOUS_ALLHAD_MC_SHARD_SIZE=25

run_campaign() {
  local shift="$1"
  local scope="$2"
  local tag="$3"
  echo "[boosted-run] shift=${shift} scope=${scope} tag=${tag} submit=${AUTONOMOUS_ALLHAD_SUBMIT_CONDOR}"
  AUTONOMOUS_ALLHAD_PRODUCTION_SHIFT="$shift"   AUTONOMOUS_ALLHAD_FULL_RECORD_SCOPE="$scope"   AUTONOMOUS_ALLHAD_PRODUCTION_TAG="$tag"   ./autonomous_allhad/analysisctl run-production --config "$CONFIG"
}

# Nominal carries DATA + background MC. Non-nominal shape campaigns are MC/background-only.
run_campaign nominal all "${TAG_BASE}_nominal"
run_campaign jesTotalUp background "${TAG_BASE}_jesTotalUp"
run_campaign jesTotalDown background "${TAG_BASE}_jesTotalDown"
run_campaign metUnclusteredUp background "${TAG_BASE}_metUnclusteredUp"
run_campaign metUnclusteredDown background "${TAG_BASE}_metUnclusteredDown"
