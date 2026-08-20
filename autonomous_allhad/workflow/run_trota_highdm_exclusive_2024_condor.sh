#!/usr/bin/env bash
set -euo pipefail

CHUNK_LIST="${1:?missing chunk list}"
DEST="${2:?missing destination JSON}"
WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$WORKDIR"

case "$DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS destination: $DEST" >&2; exit 64 ;;
esac

mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig inputs
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export MPLCONFIGDIR="$WORKDIR/runtime_mplconfig"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4
export XRD_REQUESTTIMEOUT=180
export XRD_REDIRECTLIMIT=10
export X509_USER_PROXY="$WORKDIR/x509up_u147757"
chmod 600 "$X509_USER_PROXY"

tar -xzf py38.tgz
tar -xzf hist2024_nres_code.tgz
test -s "$WORKDIR/btageff2024.merged"
mkdir -p "$WORKDIR/analysis/hists"
ln -s "$WORKDIR/btageff2024.merged" "$WORKDIR/analysis/hists/btageff2024.merged"
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR/autonomous_allhad:$WORKDIR"

XRDCOPY="$(command -v xrdcp)"
XRDFS="$(command -v xrdfs)"
LOCAL_LIST="$WORKDIR/local_inputs.tsv"
: > "$LOCAL_LIST"

copy_with_retry() {
  local source="$1"
  local target="$2"
  local copied=0
  for attempt in 1 2 3 4 5; do
    if "$XRDCOPY" -f --nopbar --streams 4 "$source" "$target"; then
      copied=1
      break
    fi
    sleep "$((attempt * 10))"
  done
  test "$copied" -eq 1
}

while IFS= read -r SOURCE_ROOT; do
  [ -n "$SOURCE_ROOT" ] || continue
  case "$SOURCE_ROOT" in
    /eos/user/t/taiwoo/*) ;;
    *) echo "refusing input outside scoped EOS area: $SOURCE_ROOT" >&2; exit 65 ;;
  esac
  BASE="$(basename "$SOURCE_ROOT")"
  LOCAL_ROOT="$WORKDIR/inputs/$BASE"
  SOURCE_JSON="${SOURCE_ROOT%.root}.json"
  LOCAL_JSON="${LOCAL_ROOT%.root}.json"
  copy_with_retry "root://eosuser.cern.ch/$SOURCE_ROOT" "$LOCAL_ROOT"
  copy_with_retry "root://eosuser.cern.ch/$SOURCE_JSON" "$LOCAL_JSON"
  test -s "$LOCAL_ROOT"
  test -s "$LOCAL_JSON"
  printf '%s\t%s\n' "$LOCAL_ROOT" "$SOURCE_ROOT" >> "$LOCAL_LIST"
done < "$CHUNK_LIST"

"$PY" -u autonomous_allhad/workflow/build_trota_highdm_exclusive_2024.py \
  --repo "$WORKDIR" \
  --input-list "$LOCAL_LIST" \
  --normalization "$WORKDIR/normalization.json" \
  --output "$WORKDIR/result.json"
test -s "$WORKDIR/result.json"

DEST_URL="root://eosuser.cern.ch/$DEST"
copy_with_retry "$WORKDIR/result.json" "$DEST_URL"
"$XRDFS" eosuser.cern.ch stat "$DEST" >/dev/null
sha256sum "$WORKDIR/result.json"
echo "completed $CHUNK_LIST -> $DEST"
