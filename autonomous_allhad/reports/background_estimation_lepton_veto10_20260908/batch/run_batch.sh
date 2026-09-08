#!/bin/bash
# Reuse the established transferred py38 runtime and saved-argv launcher.
set -euo pipefail
ARGV_FILE="${1:?missing saved argv}"
case "$ARGV_FILE" in
  /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/reports/background_estimation_lepton_veto10_20260908/batch/*.argv) ;;
  *) exit 64 ;;
esac
WORKDIR="${_CONDOR_SCRATCH_DIR:?Condor worker scratch is required}"
cd "$WORKDIR"
mkdir -p runtime_tmp runtime_cache runtime_mplconfig
export TMPDIR="$WORKDIR/runtime_tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export MPLCONFIGDIR="$WORKDIR/runtime_mplconfig"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
tar -xzf py38.tgz
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${LD_LIBRARY_PATH:-}"
mapfile -t ARGS < "$ARGV_FILE"
ARGS[0]="$PY"
exec "${ARGS[@]}"
