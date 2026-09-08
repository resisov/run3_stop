#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--topw" ]; then
    processor="${2:?Missing processor}"
    metadata="${3:?Missing metadata}"
    workdir="${_CONDOR_SCRATCH_DIR:-$PWD}"
    cd "$workdir"
    mkdir -p runtime_tmp runtime_cache
    export TMPDIR="$workdir/runtime_tmp"
    export XDG_CACHE_HOME="$workdir/runtime_cache"
    export NUMBA_CACHE_DIR="$workdir/runtime_cache"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export X509_USER_PROXY="$workdir/x509up_u147757"
    export XRD_NETWORKSTACK=IPv4 XRD_REQUESTTIMEOUT=120
    unset PYTHONPATH LD_LIBRARY_PATH
    tar -xzf py38.tgz
    pyroot="$workdir"
    if [ ! -x "$pyroot/bin/python" ]; then pyroot="$workdir/py38"; fi
    test -x "$pyroot/bin/python"
    export PATH="$pyroot/bin:$PATH" LD_LIBRARY_PATH="$pyroot/lib"
    tar -xzf analysis.tgz
    cd analysis
    export PYTHONPATH="$workdir/analysis"
    exec "$pyroot/bin/python" -u run.py -p "$processor" \
        --metadata-path "$workdir/$metadata" --output "$workdir/out.futures" \
        --file-report "$workdir/report.json" --chunk-size 10000 -w 1
fi

name="$1"
shard="$2"
# Results are created in the Condor execute sandbox and transferred to EOS.
output="${name}.npz"
metadata="${name}.json"

export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=1800
export AUTONOMOUS_ALLHAD_XRD_STREAMS=4
export XRD_NETWORKSTACK=IPv4
export PYTHONNOUSERSITE=1
export X509_USER_PROXY="$PWD/x509up_u147757"
chmod 600 "$X509_USER_PROXY" || true

tar -xzf py38.tgz
PYROOT="$PWD/py38"
if [ ! -x "$PYROOT/bin/python" ]; then
    PYBIN=$(find "$PWD" -maxdepth 3 -type f -path "*/bin/python" | head -1)
    PYROOT=$(dirname "$(dirname "$PYBIN")")
fi
PYTHON="$PYROOT/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "local python not found after unpacking py38.tgz" >&2
    exit 66
fi

export PATH="$PYROOT/bin:$PATH"
export LD_LIBRARY_PATH="$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad:${PYTHONPATH:-}

exec "$PYTHON" -m autonomous_allhad.toptag_eff_worker \
    --repo /eos/user/t/taiwoo/run3_stop/decaf \
    --shard "$shard" \
    --output "$output" \
    --metadata-output "$metadata"
