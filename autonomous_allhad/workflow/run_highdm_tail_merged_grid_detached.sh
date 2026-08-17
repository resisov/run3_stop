#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:?usage: run_highdm_tail_merged_grid_detached.sh OUTPUT_DIR [MAX_JOBS]}
MAX_JOBS=${2:-4}

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4/src
eval "$(scram runtime -sh)"
cd /eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad

exec bash workflow/run_highdm_tail_merged_grid.sh "$OUTPUT_DIR" "$MAX_JOBS"
