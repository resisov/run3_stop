#!/usr/bin/env bash
set -uo pipefail

REPO=${REPO:-/eos/user/t/taiwoo/run3_stop/decaf}
WORKFLOW="$REPO/autonomous_allhad/workflow"
LIST=${1:-/tmp/taiwoo/data2_unreadable_after_proxy.txt}
TAG=sf_unc_v3_dyfix_data2_20260625
CONDOR_DIR="$WORKFLOW/condor_${TAG}"
OUTPUT_DIR="$WORKFLOW/production_outputs_${TAG}"
MANIFEST="$CONDOR_DIR/${TAG}_data_records.json"
RUNTIME=${RUNTIME:-/tmp/taiwoo/recover_data2_runtime}
SOURCE_RUNTIME=${SOURCE_RUNTIME:-/tmp/taiwoo/recover_data2_00016}
SPEC_DIR=${SPEC_DIR:-/tmp/taiwoo/data2_recovery_specs}
LOG_DIR=${LOG_DIR:-$WORKFLOW/local_data2_shard_recovery_20260627_logs}
PROXY=${X509_USER_PROXY:-/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757}
MAX_JOBS=${MAX_JOBS:-2}
WORKER_TIMEOUT=${WORKER_TIMEOUT:-7200}

mkdir -p "$RUNTIME" "$SPEC_DIR" "$LOG_DIR"

if [[ ! -x "$RUNTIME/bin/python" ]]; then
  if [[ -x "$SOURCE_RUNTIME/bin/python" ]]; then
    RUNTIME="$SOURCE_RUNTIME"
  else
    cp "$REPO/condor/py38.tgz" "$RUNTIME/"
    tar -xzf "$RUNTIME/py38.tgz" -C "$RUNTIME"
  fi
fi

cp "$PROXY" "$RUNTIME/x509up_u147757"
chmod 600 "$RUNTIME/x509up_u147757" || true

PYTHON="$RUNTIME/bin/python"
export PYTHONNOUSERSITE=1
export PATH="$RUNTIME/bin:$PATH"
export LD_LIBRARY_PATH="$RUNTIME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/autonomous_allhad:${PYTHONPATH:-}"
export X509_USER_PROXY="$RUNTIME/x509up_u147757"
export AUTONOMOUS_ALLHAD_FULL_CHUNK=${AUTONOMOUS_ALLHAD_FULL_CHUNK:-50000}
export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=${AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT:-300}
if [[ "${AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT}" =~ ^[0-9]+$ ]] && (( AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT < 300 )); then
  export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=300
fi
export AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE=${AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE:-0}
export XRD_NETWORKSTACK=${XRD_NETWORKSTACK:-IPv4}

make_spec() {
  local name=$1
  local start=$2
  local spec=$3
  python3 - "$MANIFEST" "$name" "$start" "$spec" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
name = sys.argv[2]
start = int(sys.argv[3])
spec = Path(sys.argv[4])
payload = json.loads(manifest_path.read_text())
chunk = payload["records"][start:start + 2]
if not chunk:
    raise SystemExit(f"empty DATA chunk for {name} start={start}")
digest = hashlib.sha256(json.dumps(chunk, sort_keys=True).encode()).hexdigest()[:16]
spec.write_text(json.dumps({
    "schema_version": "full_production_shard_spec_v1",
    "shard_id": name,
    "record_digest": digest,
    "records": chunk,
}, separators=(",", ":"), sort_keys=True) + "\n")
PY
}

recover_one() {
  local output=$1
  local base name raw_id start spec state prefer_cache
  base=$(basename "$output")
  name=${base%.json}
  raw_id=${name#shard_}
  start=$((10#$raw_id * 2))
  spec="$SPEC_DIR/${name}.json"

  state=$("$PYTHON" - "$output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception:
    print("retry")
    raise SystemExit

bad_files = data.get("bad_files") or []
if data.get("status") == "complete" and not bad_files:
    print("complete")
elif data.get("status") == "failed" or bad_files:
    print("failed")
else:
    print("retry")
PY
)
  if [[ "$state" == "complete" ]]; then
    echo "$(date -u +%FT%TZ) skip_complete $name"
    return 0
  fi
  prefer_cache="${AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE:-0}"
  backup=""
  if [[ "$state" == "failed" ]]; then
    prefer_cache=1
    echo "$(date -u +%FT%TZ) retry_failed_json $name prefer_cache=1"
    if [[ -s "$output" ]]; then
      backup="${output}.retry_backup.$(date -u +%Y%m%dT%H%M%SZ).$$"
      mv "$output" "$backup"
      rm -f "${output}.running"
      echo "$(date -u +%FT%TZ) moved_existing_bad_json $name backup=$backup"
    fi
  fi

  echo "$(date -u +%FT%TZ) start $name start=$start output=$output"
  make_spec "$name" "$start" "$spec"
  AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="$prefer_cache" timeout "$WORKER_TIMEOUT" "$PYTHON" -m autonomous_allhad.full_production_worker \
    --repo "$REPO" \
    --shard "$spec" \
    --output "$output" \
    --shift nominal
  worker_rc=$?
  if [[ $worker_rc -ne 0 ]]; then
    if [[ -n "$backup" && ! -s "$output" && -s "$backup" ]]; then
      mv "$backup" "$output"
      echo "$(date -u +%FT%TZ) restored_bad_json $name backup=$backup"
    fi
    echo "$(date -u +%FT%TZ) worker_failed $name rc=$worker_rc"
    return $worker_rc
  fi
  state=$("$PYTHON" - "$output" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    print("unreadable")
    raise SystemExit
if data.get("status") == "complete" and not (data.get("bad_files") or []):
    print("complete")
else:
    print("failed")
PY
)
  if [[ -n "$backup" && -s "$backup" ]]; then
    rm -f "$backup"
  fi
  if [[ "$state" == "complete" ]]; then
    echo "$(date -u +%FT%TZ) done_readable $name"
  else
    echo "$(date -u +%FT%TZ) done_unreadable $name"
    return 1
  fi
}

echo "$(date -u +%FT%TZ) data2 shard recovery start list=$LIST max_jobs=$MAX_JOBS"
while IFS= read -r output; do
  if [[ -z "$output" ]]; then
    continue
  fi
  case "$output" in
    /*) ;;
    *) output="$OUTPUT_DIR/$(basename "$output")" ;;
  esac
  name=$(basename "${output%.json}")
  while [[ $(jobs -rp | wc -l) -ge $MAX_JOBS ]]; do
    wait -n || true
  done
  recover_one "$output" >"$LOG_DIR/${name}.log" 2>&1 &
done < "$LIST"

while [[ $(jobs -rp | wc -l) -gt 0 ]]; do
  wait -n || true
done
echo "$(date -u +%FT%TZ) data2 shard recovery done"
