#!/bin/bash
set -euo pipefail

ARGV_FILE="${1:?missing saved argv file}"
case "$ARGV_FILE" in
  /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/studies/object_definition_veto_pt_20260905/recovery/*.argv) ;;
  /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/histograms/dy_window20_20260907/*/*.argv) ;;
  /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/histograms/lepton_veto10_20260908/*/*.argv) ;;
  *)
    echo "refusing argv file outside the approved histogram campaigns: $ARGV_FILE" >&2
    exit 64
    ;;
esac

WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$WORKDIR"
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig
export TMPDIR="$WORKDIR/runtime_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export MPLCONFIGDIR="$WORKDIR/runtime_mplconfig"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export AUTONOMOUS_ALLHAD_ANALYSIS_CACHE_DIR="$WORKDIR/runtime_cache/analysis"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

tar -xzf py38.tgz
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${LD_LIBRARY_PATH:-}"

MT2_WHEEL="$WORKDIR/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl"
if [ -s "$MT2_WHEEL" ]; then
  "$PY" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$MT2_WHEEL" "$WORKDIR/vendor"
  export PYTHONPATH="$WORKDIR/vendor:${PYTHONPATH:-}"
  "$PY" -c 'import mt2; assert mt2.__version__ == "1.2.0"'
fi

mapfile -t ARGS < "$ARGV_FILE"
[ "${#ARGS[@]}" -gt 1 ]
ARGS[0]="$PY"
for ((index=1; index<${#ARGS[@]}; index++)); do
  if [ "${ARGS[$index]}" = "--python" ]; then
    ARGS[$((index+1))]="$PY"
  fi
done
exec "${ARGS[@]}"
