#!/usr/bin/env bash
set -u
set +e
set +o pipefail 2>/dev/null || true

REPO=${REPO:-/eos/user/t/taiwoo/run3_stop/decaf}
WORKFLOW="${REPO}/autonomous_allhad/workflow"
DATA_LOG_DIR="${WORKFLOW}/local_data2_shard_recovery_20260627_logs"
DATA_OUTPUT_DIR="${WORKFLOW}/production_outputs_sf_unc_v3_dyfix_data2_20260625"
RUNNER="${WORKFLOW}/local_data2_shard_recovery_20260627.sh"
LOG_DIR="${WORKFLOW}/autonomous_recover_merge_deploy_20260627_logs"
LOG_FILE="${LOG_DIR}/data_failed_retry_sidecar.log"
FAILED_LIST="/tmp/taiwoo/data2_failed_retry_20260627.txt"
LOCK="/tmp/data_failed_retry_sidecar_20260627.lock"
DATA_TOTAL=${DATA_TOTAL:-426}
MAX_TOTAL_WORKERS=${MAX_TOTAL_WORKERS:-8}
WORKER_HEADROOM=${WORKER_HEADROOM:-1}
MAX_RETRY_JOBS=${MAX_RETRY_JOBS:-1}
POLL_SLEEP=${POLL_SLEEP:-60}
RUNTIME=${RUNTIME:-/tmp/taiwoo/recover_data2_00016}

mkdir -p "${LOG_DIR}" /tmp/taiwoo
exec >> "${LOG_FILE}" 2>&1

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

active_data_workers() {
  ps -eo comm,args | awk '$1 ~ /python/ && $0 ~ /autonomous_allhad.full_production_worker --repo/ && $0 ~ /production_outputs_sf_unc_v3_dyfix_data2_20260625/ {n++} END{print n+0}'
}

data_counts() {
  local done bad active
  done=$(rg -l "done_readable|skip_readable|skip_complete" "${DATA_LOG_DIR}"/shard_*.log 2>/dev/null | wc -l)
  bad=$(rg -l "done_unreadable|worker_failed" "${DATA_LOG_DIR}"/shard_*.log 2>/dev/null | wc -l)
  active=$(active_data_workers)
  echo "${done} ${bad} ${active}"
}

build_failed_list() {
  local tmp="${FAILED_LIST}.$$"
  : > "${tmp}"
  for f in $(rg -l "done_unreadable|worker_failed" "${DATA_LOG_DIR}"/shard_*.log 2>/dev/null | sort); do
    local name output
    name=$(basename "${f}" .log)
    output="${DATA_OUTPUT_DIR}/${name}.json"
    python3 - "${output}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
if payload.get("status") == "complete" and not (payload.get("bad_files") or []):
    raise SystemExit(0)
raise SystemExit(1)
PY
    if [ $? -ne 0 ]; then
      echo "${output}" >> "${tmp}"
    fi
  done
  mv "${tmp}" "${FAILED_LIST}"
  wc -l < "${FAILED_LIST}"
}

main() {
  exec 8>"${LOCK}"
  if ! flock -n 8; then
    log "already running"
    return 0
  fi

  log "start max_total_workers=${MAX_TOTAL_WORKERS}"
  while true; do
    read -r done bad active <<EOF
$(data_counts)
EOF
    pending=$(build_failed_list)
    log "status done=${done} bad=${bad} active=${active} pending_failed=${pending}"

    if [ "${done}" -ge "${DATA_TOTAL}" ] && [ "${bad}" -eq 0 ] && [ "${active}" -eq 0 ]; then
      log "complete; exiting"
      return 0
    fi

    if [ "${pending}" -gt 0 ]; then
      active=$(active_data_workers)
      if [ "${active}" -eq "$((MAX_TOTAL_WORKERS - 1))" ]; then
        log "candidate worker slot active=${active}; rechecking before retry"
        sleep 20
        active=$(active_data_workers)
      fi
      free=$((MAX_TOTAL_WORKERS - active))
      if [ "${free}" -gt "${MAX_RETRY_JOBS}" ]; then
        free="${MAX_RETRY_JOBS}"
      fi
      if [ "${free}" -gt 0 ]; then
        retry_list="${FAILED_LIST}.one"
        head -n "${free}" "${FAILED_LIST}" > "${retry_list}"
        log "retry failed shards free_slots=${free} list=${retry_list} active=${active}"
        MAX_JOBS="${free}" \
        RUNTIME="${RUNTIME}" \
        SOURCE_RUNTIME="${RUNTIME}" \
        AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=1 \
        AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=300 \
        LOG_DIR="${DATA_LOG_DIR}" \
          "${RUNNER}" "${retry_list}"
        log "retry runner rc=$?"
      else
        log "waiting for worker slot active=${active}"
      fi
    fi
    sleep "${POLL_SLEEP}"
  done
}

main "$@"
