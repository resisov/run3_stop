#!/usr/bin/env bash
set -uo pipefail

BASE=${1:?usage: run_signal_tail_bin_merge_diagnostics.sh DIAGNOSTIC_BASE}
COMBINE=${COMBINE:-/eos/home-t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4/bin/el9_amd64_gcc12/combine}
POINT_TIMEOUT=${POINT_TIMEOUT:-1800}

fail=0
while IFS= read -r card; do
    point_dir=$(dirname "$card")
    limit_dir="${point_dir}/limits"
    output="${limit_dir}/higgsCombine_diagnostic.AsymptoticLimits.mH120.root"
    log="${limit_dir}/combine.log"
    mkdir -p "$limit_dir"
    if [[ -s "$output" ]]; then
        echo "[diagnostic-skip] ${point_dir}"
        continue
    fi
    echo "[diagnostic-start] ${point_dir}"
    (
        cd "$limit_dir" &&
        nice -n 5 timeout "$POINT_TIMEOUT" "$COMBINE" \
            -M AsymptoticLimits --run blind -n _diagnostic "$card"
    ) >"$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -s "$output" ]]; then
        echo "[diagnostic-fail] ${point_dir} rc=${rc}"
        fail=1
    else
        echo "[diagnostic-complete] ${point_dir}"
    fi
done < <(find "$BASE" -type f -name 'datacard_mStop*_mLSP*.txt' | sort)

exit "$fail"
