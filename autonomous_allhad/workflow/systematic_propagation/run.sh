#!/bin/bash
set -euo pipefail
DRIVER="${1:?missing driver}"
shift
case "$DRIVER" in
  /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/systematic_propagation/run.py) ;;
  *) exit 64 ;;
esac
cd "${_CONDOR_SCRATCH_DIR:?batch scratch required}"
mkdir -p runtime_cache runtime_tmp runtime_mpl
export TMPDIR="$PWD/runtime_tmp" TMP="$PWD/runtime_tmp" TEMP="$PWD/runtime_tmp"
export XDG_CACHE_HOME="$PWD/runtime_cache" MPLCONFIGDIR="$PWD/runtime_mpl"
export NUMBA_CACHE_DIR="$PWD/runtime_cache/numba"
export KERAS_HOME="$PWD/runtime_cache/keras"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
tar -xzf py38.tgz
PY="$PWD/bin/python"
test -x "$PY"
export PATH="$PWD/bin:$PATH" LD_LIBRARY_PATH="$PWD/lib"
"$PY" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
  mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl "$PWD/vendor"
export PYTHONPATH="$PWD/vendor"
export SHAPE_VENDOR="$PWD/vendor"
"$PY" -c 'import mt2; assert mt2.__version__ == "1.2.0"'
if [ "${1:-}" = "object-worker" ]; then
  export SHAPE_BUNDLE="$PWD/object_code.tgz"
  export SHAPE_CODE_ROOT="$PWD/object_code"
  mkdir -p "$SHAPE_CODE_ROOT"
  tar -xzf "$SHAPE_BUNDLE" -C "$SHAPE_CODE_ROOT"
  DRIVER="$SHAPE_CODE_ROOT/run.py"
fi
exec "$PY" -u "$DRIVER" "$@"
