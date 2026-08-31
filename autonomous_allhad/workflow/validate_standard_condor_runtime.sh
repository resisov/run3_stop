#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    exit 2
fi

STATUS=$1
: "${_CONDOR_SCRATCH_DIR:?Condor scratch directory is required}"
SCRATCH=$_CONDOR_SCRATCH_DIR
exec > "${STATUS%.json}.out" 2> "${STATUS%.json}.err"
cd "$SCRATCH"
export HOME="$SCRATCH"
export TMPDIR="$SCRATCH"
export XDG_CACHE_HOME="$SCRATCH/cache"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
BASE_PATH=$PATH
BASE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
mkdir -p "$XDG_CACHE_HOME"

PYTHON_SHA=$(awk '$2 ~ /py38[.]tgz$/ {print $1}' SHA256SUMS)
COMBINE_SHA=$(awk '$2 ~ /combine_cmssw_14_1_0_pre4[.]tgz$/ {print $1}' SHA256SUMS)
test -n "$PYTHON_SHA"
test -n "$COMBINE_SHA"
echo "$PYTHON_SHA  py38.tgz" | sha256sum -c -
echo "$COMBINE_SHA  combine_cmssw_14_1_0_pre4.tgz" | sha256sum -c -

mkdir python
tar -xzf py38.tgz -C python
export PATH="$SCRATCH/python/bin:$PATH"
export LD_LIBRARY_PATH="$SCRATCH/python/lib:${LD_LIBRARY_PATH:-}"
python/bin/conda-unpack
python/bin/python -c 'import awkward,coffea,correctionlib,numpy,uproot'
export PATH=$BASE_PATH
export LD_LIBRARY_PATH=$BASE_LD_LIBRARY_PATH

tar -xzf combine_cmssw_14_1_0_pre4.tgz
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd CMSSW_14_1_0_pre4/src
scramv1 b ProjectRename >/dev/null
eval "$(scramv1 runtime -sh)"
command -v combine
command -v text2workspace.py
case "$(command -v combine)" in "$SCRATCH"/*) ;; *) exit 70 ;; esac
case "$(command -v text2workspace.py)" in "$SCRATCH"/*) ;; *) exit 70 ;; esac
combine --help >/dev/null
text2workspace.py --help >/dev/null
python3 -c 'from HiggsAnalysis.CombinedLimit.PhysicsModel import defaultModel'

mkdir -p "$(dirname "$STATUS")"
printf '{"status":"ready","python":"3.8.20","cmssw":"CMSSW_14_1_0_pre4"}\n' \
    > "$STATUS"
