#!/usr/bin/env bash
set -uo pipefail

# Keep CMSSW's pinned Python stack isolated from an intermittently unavailable
# AFS user-site directory (for example ~/.local/lib/python3.9/site-packages).
export PYTHONNOUSERSITE=1

BASE=${1:?usage: run_highdm_tail_merged_grid.sh OUTPUT_DIR [MAX_JOBS]}
MAX_JOBS=${2:-4}
COMBINE=${COMBINE:-combine}
POINT_TIMEOUT=${POINT_TIMEOUT:-1800}
CARD_DIR="${BASE}/datacards"
LIMIT_DIR="${BASE}/limits"
VALIDATION_PYTHON=${VALIDATION_PYTHON:-/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python}

mkdir -p "$LIMIT_DIR"

valid_limit_output() {
    local output="$1"
    [[ -s "$output" ]] || return 1
    "$VALIDATION_PYTHON" -c '
import math
import sys
import uproot
with uproot.open(sys.argv[1]) as root_file:
    tree = root_file["limit"]
    values = tree["limit"].array(library="np")
    quantiles = tree["quantileExpected"].array(library="np")
expected_quantiles = (0.025, 0.16, 0.5, 0.84, 0.975)
if len(values) != 5 or len(quantiles) != 5:
    raise SystemExit(1)
if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
    raise SystemExit(1)
if not all(
    math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-5)
    for value, expected in zip(quantiles, expected_quantiles)
):
    raise SystemExit(1)
' "$output" >/dev/null 2>&1
}

exclude_failed_point() {
    local card="$1"
    local mass="$2"
    local output="$3"
    local log="$4"
    local rc="$5"
    local topology
    topology=$(basename "$BASE")
    case "$topology" in
        T2bW|T2tb) ;;
        *) return 0 ;;
    esac

    local exclusion_root="${BASE}/excluded_points"
    local artifact_dir="${exclusion_root}/${mass}"
    local record="${exclusion_root}/limit_point_exclusions_${mass}.json"
    mkdir -p "$artifact_dir"
    [[ -e "$card" ]] && mv "$card" "$artifact_dir/"
    [[ -e "$output" ]] && mv "$output" "$artifact_dir/"
    [[ -e "$log" ]] && mv "$log" "$artifact_dir/"
    "$VALIDATION_PYTHON" -c '
import datetime
import json
import re
import sys
from pathlib import Path

path, topology, mass, rc, timeout = sys.argv[1:]
match = re.fullmatch(r"mStop(\d+)_mLSP(\d+)", mass)
if match is None:
    raise SystemExit(f"invalid mass point: {mass}")
payload = {
    "campaign": "highdm_tail2_merged_fullgrid_fast_20260730",
    "exclusions": [{
        "model": topology,
        "mStop_GeV": int(match.group(1)),
        "mLSP_GeV": int(match.group(2)),
        "status": "excluded_by_user_policy",
        "reason": f"Combine AsymptoticLimits failed with rc={rc}; user directed failed T2bW/T2tb points to be dropped.",
        "point_timeout_seconds": int(timeout),
        "excluded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts_policy": "Failed datacard, output, and log preserved in the point-specific excluded_points directory."
    }]
}
destination = Path(path)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
' "$record" "$topology" "$mass" "$rc" "$POINT_TIMEOUT"
    echo "[combine-exclude] ${topology} ${mass} rc=${rc}"
}

run_one() {
    local card="$1"
    card=$(readlink -f "$card")
    local mass
    mass=$(basename "$card" .txt)
    mass=${mass#datacard_}
    local output="${LIMIT_DIR}/higgsCombine_${mass}.AsymptoticLimits.mH120.root"
    local log="${LIMIT_DIR}/log_${mass}.txt"
    if valid_limit_output "$output"; then
        echo "[combine-skip] ${mass}"
        return 0
    fi
    echo "[combine-start] ${mass}"
    (
        cd "$LIMIT_DIR" &&
        nice -n 15 timeout "$POINT_TIMEOUT" "$COMBINE" \
            -M AsymptoticLimits --run blind -n "_${mass}" "$card"
    ) >"$log" 2>&1
    local rc=$?
    if [[ "$rc" -eq 0 ]] && ! valid_limit_output "$output"; then
        rc=65
        echo "[combine-invalid] ${mass} rc=${rc}"
    fi
    echo "[combine-rc] ${mass} rc=${rc}"
    if [[ "$rc" -ne 0 ]]; then
        exclude_failed_point "$card" "$mass" "$output" "$log" "$rc"
    fi
    return "$rc"
}

fail=0
running=0
while IFS= read -r card; do
    run_one "$card" || fail=1 &
    running=$((running + 1))
    if [[ "$running" -ge "$MAX_JOBS" ]]; then
        wait -n || fail=1
        running=$((running - 1))
    fi
done < <(find "$CARD_DIR" -maxdepth 1 -type f \
    -name 'datacard_mStop*_mLSP*.txt' | sort)

while [[ "$running" -gt 0 ]]; do
    wait -n || fail=1
    running=$((running - 1))
done

exit "$fail"
