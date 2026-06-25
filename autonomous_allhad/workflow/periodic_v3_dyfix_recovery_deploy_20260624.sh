#!/usr/bin/env bash
set +e
set +u
set +o pipefail 2>/dev/null || true

REPO=/eos/user/t/taiwoo/run3_stop/decaf
PY=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
TAG=sf_unc_v3_dyfix_20260624
DATA2_TAG=sf_unc_v3_dyfix_data2_20260625
DOCS_REL=docs/partial_merge_preview_${TAG}
PREVIEW_REL=autonomous_allhad/workflow/partial_merge_preview_${TAG}
OUTPUT_REL=autonomous_allhad/workflow/production_outputs_${TAG}
SHARD_REL=autonomous_allhad/workflow/production_shards_${TAG}
DATA2_OUTPUT_REL=autonomous_allhad/workflow/production_outputs_${DATA2_TAG}
DATA2_SHARD_REL=autonomous_allhad/workflow/production_shards_${DATA2_TAG}
DATA2_RECORDS_REL=autonomous_allhad/workflow/condor_${DATA2_TAG}/${DATA2_TAG}_data_records.json
LABEL="Partial Merge Preview ${TAG} (v3 DY recoil fix)"
SOURCE_CLUSTER_ID=923877
CLUSTERS="923877 923878 923880 923881 923882 924704"
RECOVERY_INTERVAL=${RECOVERY_INTERVAL:-3600}
DEPLOY_INTERVAL=${DEPLOY_INTERVAL:-10800}
SLEEP_SECONDS=${PERIODIC_SLEEP_SECONDS:-300}
LOCK=/tmp/periodic_v3_dyfix_recovery_deploy_20260624.lock
LOG_DIR=${REPO}/autonomous_allhad/workflow/periodic_v3_dyfix_logs
STATE_JSON=${REPO}/autonomous_allhad/workflow/periodic_v3_dyfix_state.json
STATUS_DOC=${REPO}/docs/data/v3_dyfix_periodic_status.json
CONDOR_DOC=${REPO}/docs/data/v3_dyfix_condor_totals.txt

export PYTHONPATH="${REPO}:${REPO}/autonomous_allhad:${PYTHONPATH:-}"
export X509_USER_PROXY=${X509_USER_PROXY:-/tmp/x509up_u147757}
export XRD_NETWORKSTACK=${XRD_NETWORKSTACK:-IPv4}
export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=${AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT:-1800}
export AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=${AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE:-1}
export AUTONOMOUS_ALLHAD_FULL_CHUNK=${AUTONOMOUS_ALLHAD_FULL_CHUNK:-50000}
export _condor_CONDOR_HOST=${_condor_CONDOR_HOST:-"tweetybird04.cern.ch, tweetybird03.cern.ch"}
export _myschedd_POOL=${_myschedd_POOL:-eossubmit}

mkdir -p "${LOG_DIR}" "${REPO}/docs/data"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) periodic runner already active"
  exit 0
fi

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG_DIR}/periodic.log"
}

write_state() {
  local phase="$1"
  local status="$2"
  PHASE="${phase}" STATUS="${status}" CLUSTERS="${CLUSTERS}" TAG="${TAG}" STATE_JSON="${STATE_JSON}" STATUS_DOC="${STATUS_DOC}" "${PY}" - <<'STATEPY'
import json
import os
import subprocess
import time
from pathlib import Path
repo = Path('/eos/user/t/taiwoo/run3_stop/decaf')
state_path = Path(os.environ['STATE_JSON'])
doc_path = Path(os.environ['STATUS_DOC'])
old = {}
try:
    old = json.loads(state_path.read_text())
except Exception:
    pass
clusters = os.environ.get('CLUSTERS', '').split()
condor = {'query_status': 'not_run', 'clusters': clusters}
try:
    proc = subprocess.run(['condor_q', '-name', 'bigbird24', *clusters, '-totals'], cwd=repo, text=True, capture_output=True, timeout=120)
    condor = {'query_status': 'ok' if proc.returncode == 0 else 'failed', 'returncode': proc.returncode, 'stdout_tail': proc.stdout[-4000:], 'stderr_tail': proc.stderr[-1000:], 'clusters': clusters}
except Exception as exc:
    condor = {'query_status': 'exception', 'error': f'{type(exc).__name__}: {exc}', 'clusters': clusters}
now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
old.update({
    'updated_at': now,
    'tag': os.environ.get('TAG'),
    'phase': os.environ.get('PHASE'),
    'status': os.environ.get('STATUS'),
    'clusters': clusters,
    'condor': condor,
    'github_pages_url': 'https://resisov.github.io/run3_stop/',
    'partial_preview_url': f"https://resisov.github.io/run3_stop/partial_merge_preview_{os.environ.get('TAG')}/",
})
state_path.write_text(json.dumps(old, indent=2, sort_keys=True) + '\n')
doc_path.parent.mkdir(parents=True, exist_ok=True)
doc_path.write_text(json.dumps(old, indent=2, sort_keys=True) + '\n')
STATEPY
}

run_recovery_campaign() {
  local tag="$1"
  local shift="$2"
  local kinds="$3"
  local output_rel="$4"
  local shard_rel="$5"
  if [ ! -d "${REPO}/${shard_rel}" ]; then
    log "recovery skip ${tag}: missing shard dir ${shard_rel}"
    return 0
  fi
  mkdir -p "${REPO}/${output_rel}"
  log "recovery start tag=${tag} shift=${shift} kinds=${kinds}"
  "${PY}" "${REPO}/autonomous_allhad/workflow/local_bad_file_recovery_generic.py" \
    --repo "${REPO}" \
    --output-dir "${output_rel}" \
    --shard-dir "${shard_rel}" \
    --tag "${tag}" \
    --shift "${shift}" \
    --kinds "${kinds}" \
    --workers 4 \
    --retry-permanent \
    --all-reasons >> "${LOG_DIR}/recovery_${tag}.log" 2>&1
  local rc=$?
  log "recovery done tag=${tag} rc=${rc}"
  return ${rc}
}

materialize_manifest_bad_shards() {
  local tag="$1"
  local records_rel="$2"
  local output_rel="$3"
  local shard_rel="$4"
  if [ ! -f "${REPO}/${records_rel}" ]; then
    log "manifest shard materialize skip ${tag}: missing records manifest ${records_rel}"
    return 0
  fi
  mkdir -p "${REPO}/${shard_rel}"
  TAG="${tag}" RECORDS_JSON="${REPO}/${records_rel}" OUTPUT_DIR="${REPO}/${output_rel}" SHARD_DIR="${REPO}/${shard_rel}" REPO="${REPO}" "${PY}" - <<'MANIFESTPY'
import hashlib
import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO"])
sys.path.insert(0, str(repo / "autonomous_allhad/workflow"))
from local_bad_file_recovery_generic import extract_top_level_array

records_json = Path(os.environ["RECORDS_JSON"])
output_dir = Path(os.environ["OUTPUT_DIR"])
shard_dir = Path(os.environ["SHARD_DIR"])
payload = json.loads(records_json.read_text())
records = payload.get("records") or []
created = 0
checked = 0
for source in sorted(output_dir.glob("shard_*.json")) + sorted(output_dir.glob("shard_*.json.running")):
    try:
        bad_files = extract_top_level_array(source, "bad_files")
    except Exception:
        continue
    if not bad_files:
        continue
    shard_id = source.name.replace(".json.running", "").replace(".json", "")
    try:
        index = int(shard_id.split("_", 1)[1])
    except Exception:
        continue
    chunk = records[index * 2:index * 2 + 2]
    if not chunk:
        continue
    checked += 1
    digest = hashlib.sha256(json.dumps(chunk, sort_keys=True).encode()).hexdigest()[:16]
    shard_path = shard_dir / f"{shard_id}.json"
    if shard_path.exists():
        try:
            old = json.loads(shard_path.read_text())
            if old.get("record_digest") == digest:
                continue
        except Exception:
            pass
    shard_path.write_text(json.dumps({
        "schema_version": "full_production_shard_spec_v1",
        "shard_id": shard_id,
        "record_digest": digest,
        "records": chunk,
    }, separators=(",", ":"), sort_keys=True) + "\n")
    created += 1
print(json.dumps({"tag": os.environ["TAG"], "bad_output_shards_checked": checked, "shards_materialized": created}, sort_keys=True))
MANIFESTPY
  local rc=$?
  log "manifest shard materialize tag=${tag} rc=${rc}"
  return ${rc}
}

run_recovery_once() {
  log "hourly recovery sweep start"
  write_state recovery running
  materialize_manifest_bad_shards "${DATA2_TAG}" "${DATA2_RECORDS_REL}" "${DATA2_OUTPUT_REL}" "${DATA2_SHARD_REL}"
  run_recovery_campaign "${DATA2_TAG}" "nominal" "data" \
    "${DATA2_OUTPUT_REL}" \
    "${DATA2_SHARD_REL}"
  run_recovery_campaign "sf_unc_v3_dyfix_20260624" "nominal" "mc" \
    "autonomous_allhad/workflow/production_outputs_sf_unc_v3_dyfix_20260624" \
    "autonomous_allhad/workflow/production_shards_sf_unc_v3_dyfix_20260624"
  run_recovery_campaign "shape_v3_dyfix_jesTotalUp_20260624" "jesTotalUp" "mc" \
    "autonomous_allhad/workflow/production_outputs_shape_v3_dyfix_jesTotalUp_20260624" \
    "autonomous_allhad/workflow/production_shards_shape_v3_dyfix_jesTotalUp_20260624"
  run_recovery_campaign "shape_v3_dyfix_jesTotalDown_20260624" "jesTotalDown" "mc" \
    "autonomous_allhad/workflow/production_outputs_shape_v3_dyfix_jesTotalDown_20260624" \
    "autonomous_allhad/workflow/production_shards_shape_v3_dyfix_jesTotalDown_20260624"
  run_recovery_campaign "shape_v3_dyfix_metUnclusteredUp_20260624" "metUnclusteredUp" "mc" \
    "autonomous_allhad/workflow/production_outputs_shape_v3_dyfix_metUnclusteredUp_20260624" \
    "autonomous_allhad/workflow/production_shards_shape_v3_dyfix_metUnclusteredUp_20260624"
  run_recovery_campaign "shape_v3_dyfix_metUnclusteredDown_20260624" "metUnclusteredDown" "mc" \
    "autonomous_allhad/workflow/production_outputs_shape_v3_dyfix_metUnclusteredDown_20260624" \
    "autonomous_allhad/workflow/production_shards_shape_v3_dyfix_metUnclusteredDown_20260624"
  write_state recovery complete
  log "hourly recovery sweep done"
}

run_deploy_once() {
  cd "${REPO}" || return 1
  log "3-hour deploy start"
  write_state deploy running
  condor_q -name bigbird24 ${CLUSTERS} -totals > "${CONDOR_DOC}" 2>&1

  "${PY}" "${REPO}/autonomous_allhad/workflow/build_partial_merge_preview.py" \
    --repo "${REPO}" \
    --preview-dir "${PREVIEW_REL}" \
    --docs-dir "${DOCS_REL}" \
    --output-dir "${OUTPUT_REL}" \
    --shard-dir "${SHARD_REL}" \
    --label "${LABEL}" \
    --source-cluster-id "${SOURCE_CLUSTER_ID}" >> "${LOG_DIR}/partial_preview_${TAG}.log" 2>&1
  local preview_rc=$?
  if [ ${preview_rc} -ne 0 ]; then
    log "partial preview failed rc=${preview_rc}; continuing with monitor publish"
  fi

  "${PY}" ./autonomous_allhad/analysisctl monitor --config autonomous_allhad/configs/run3_2024.yaml --json > "${LOG_DIR}/monitor_latest.json" 2>> "${LOG_DIR}/deploy.log"
  "${PY}" ./autonomous_allhad/analysisctl publish-github-pages --config autonomous_allhad/configs/run3_2024.yaml > "${LOG_DIR}/publish_latest.json" 2>> "${LOG_DIR}/deploy.log"
  write_state deploy prepared

  git add docs .github/workflows/pages.yml
  if git diff --cached --quiet -- docs .github/workflows/pages.yml; then
    log "no docs changes staged after heartbeat; skipping git commit"
    write_state deploy no_changes
    return 0
  fi
  local stamp
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  git commit -m "Update v3 DY recoil monitoring ${stamp}" -- docs .github/workflows/pages.yml >> "${LOG_DIR}/git_deploy.log" 2>&1
  local commit_rc=$?
  if [ ${commit_rc} -ne 0 ]; then
    log "git commit failed rc=${commit_rc}; will retry next deploy"
    write_state deploy commit_failed
    return ${commit_rc}
  fi
  git push github HEAD >> "${LOG_DIR}/git_deploy.log" 2>&1
  local push_rc=$?
  if [ ${push_rc} -ne 0 ]; then
    log "git push failed rc=${push_rc}; will retry next deploy"
    write_state deploy push_failed
    return ${push_rc}
  fi
  write_state deploy pushed
  log "3-hour deploy pushed"
}

last_recovery=0
last_deploy=0
log "periodic runner start recovery_interval=${RECOVERY_INTERVAL} deploy_interval=${DEPLOY_INTERVAL} clusters=${CLUSTERS}"
while true; do
  now=$(date +%s)
  if [ $((now - last_recovery)) -ge ${RECOVERY_INTERVAL} ]; then
    run_recovery_once
    last_recovery=$(date +%s)
  fi
  now=$(date +%s)
  if [ $((now - last_deploy)) -ge ${DEPLOY_INTERVAL} ]; then
    run_deploy_once
    last_deploy=$(date +%s)
  fi
  if [ "${RUN_ONCE:-0}" = "1" ]; then
    log "RUN_ONCE complete"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done
