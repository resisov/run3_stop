#!/usr/bin/env bash
set -euo pipefail
echo CONDOR_SCRATCH=$(pwd)
name="$1"
shard="$2"
result_json="$3"
export AUTONOMOUS_ALLHAD_FULL_CHUNK=50000
export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=300
export PYTHONNOUSERSITE=1
export X509_USER_PROXY="$PWD/x509up_u147757"
chmod 600 "$X509_USER_PROXY" || true
tar -xzf py38.tgz
PYROOT="$PWD/py38"
if [ ! -x "$PYROOT/bin/python" ]; then PYBIN=$(find "$PWD" -maxdepth 3 -type f -path "*/bin/python" | head -1); PYROOT=$(dirname "$(dirname "$PYBIN")"); fi
PYTHON="$PYROOT/bin/python"
if [ ! -x "$PYTHON" ]; then echo "local python not found after unpacking py38.tgz" >&2; exit 66; fi
export PATH="$PYROOT/bin:$PATH"
export LD_LIBRARY_PATH="$PYROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/eos/home-t/taiwoo/run3_stop/decaf/autonomous_allhad:${PYTHONPATH:-}
"$PYTHON" - <<'PY'
import sys
import numpy, awkward, uproot
print('sys.executable', sys.executable)
print('numpy.__file__', numpy.__file__)
print('awkward.__file__', awkward.__file__)
print('uproot.__file__', uproot.__file__)
PY
exec "$PYTHON" -m autonomous_allhad.full_production_worker --repo /eos/home-t/taiwoo/run3_stop/decaf --shard "$shard" --output "$result_json" --shift metUnclusteredDown
