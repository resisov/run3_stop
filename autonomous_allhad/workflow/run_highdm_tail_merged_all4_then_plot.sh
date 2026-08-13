#!/usr/bin/env bash
set -euo pipefail

WORKFLOW=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/nominal_plots_2024_fullselection_v6_dyto2x4jets_t2models_freebkg_20260729
DIAGNOSTIC=${WORKFLOW}/diagnostics/highdm_tail2_merged_fullgrid_fast_20260730
REPO=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad
PY38=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
LOCK=${DIAGNOSTIC}/run_all4.lock

mkdir -p "$DIAGNOSTIC"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] another all4 controller holds $LOCK"
    exit 2
fi

wait_until_account_idle() {
    while true; do
        local zinv_count combine_count heavy_count
        zinv_count=$(pgrep -u "$(id -u)" -fc \
            'autonomous_allhad.dy_estimation build-features' || true)
        combine_count=$(pgrep -u "$(id -u)" -fc '^combine ' || true)
        heavy_count=$(
            ps -u "$(id -u)" -o pcpu=,comm= |
                awk '$1 >= 50.0 && $2 !~ /^(ps|awk)$/ {count++} END {print count+0}'
        )
        if [[ "$zinv_count" -eq 0 && "$combine_count" -eq 0 && "$heavy_count" -eq 0 ]]; then
            echo "[$(date --iso-8601=seconds)] account idle; starting controlled Combine"
            return
        fi
        echo "[$(date --iso-8601=seconds)] waiting: zinv=${zinv_count} combine=${combine_count} heavy=${heavy_count}"
        sleep 60
    done
}

validate_topology() {
    local topology=$1
    local output_dir=${DIAGNOSTIC}/${topology}
    local report=${output_dir}/validation_final.json
    "$PY38" "$REPO/workflow/validate_highdm_tail_merged_grid.py" \
        "$output_dir" >"$report"
    "$PY38" -c '
import json
import sys
with open(sys.argv[1]) as source:
    report = json.load(source)
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report["status"] == "complete" else 1)
' "$report"
}

wait_until_account_idle

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4/src
eval "$(scram runtime -sh)"
cd "$REPO"

for topology in T2tt T2bW T2tb; do
    echo "[$(date --iso-8601=seconds)] begin ${topology}, max total Combine=4"
    bash workflow/run_highdm_tail_merged_grid.sh \
        "${DIAGNOSTIC}/${topology}" 4 || true
    if ! validate_topology "$topology"; then
        echo "[$(date --iso-8601=seconds)] ${topology} incomplete; stopping before plotting"
        exit 3
    fi
    echo "[$(date --iso-8601=seconds)] ${topology} complete"
done

for topology in T2tt T2bW T2tb; do
    lower=$(printf '%s' "$topology" | tr '[:upper:]' '[:lower:]')
    python3 workflow/postprocess_highdm_tail_merged_grid.py \
        --input-dir "${DIAGNOSTIC}/${topology}" \
        --baseline-limits "${WORKFLOW}/free_background_${lower}/expected_limits.json" \
        --topology "$topology" \
        --max-mstop 1800
done

echo "[$(date --iso-8601=seconds)] all grids validated and plots generated"
