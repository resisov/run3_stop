#!/usr/bin/env bash
set -euo pipefail
echo CONDOR_SCRATCH=$(pwd)
name="$1"
shard="$2"
result_json="$3"
backup_dir="/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/production_outputs_sf_unc_v2_data2_20260622_resubmit_backup_20260624"
mkdir -p "$backup_dir"
if [ -e "$result_json" ]; then
  base=$(basename "$result_json")
  backup="$backup_dir/${base}.pre_resubmit_${name}_$(date +%s)"
  mv "$result_json" "$backup"
  echo "moved existing result_json to $backup"
fi
if [ -e "${result_json}.running" ]; then
  base=$(basename "$result_json")
  backup="$backup_dir/${base}.running.pre_resubmit_${name}_$(date +%s)"
  mv "${result_json}.running" "$backup"
  echo "moved existing running checkpoint to $backup"
fi
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
export PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad:${PYTHONPATH:-}
"$PYTHON" - <<'PYSMOKE'
import sys
import numpy, awkward, uproot
print('sys.executable', sys.executable)
print('numpy.__file__', numpy.__file__)
print('awkward.__file__', awkward.__file__)
print('uproot.__file__', uproot.__file__)
PYSMOKE
exec "$PYTHON" -m autonomous_allhad.full_production_worker --repo /eos/user/t/taiwoo/run3_stop/decaf --shard "$shard" --output "$result_json" --shift nominal
