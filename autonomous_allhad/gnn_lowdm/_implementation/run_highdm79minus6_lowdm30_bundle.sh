#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 CAMPAIGN HIGH_BASE LOW_BASE TOPOLOGY BUNDLE" >&2
    exit 2
fi

CAMPAIGN=$1
HIGH_BASE=$2
LOW_BASE=$3
TOPOLOGY=$4
BUNDLE=$5
CMSSW=/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4
BUNDLE_NAME=$(basename "$BUNDLE" .txt)
TARGET=$CAMPAIGN/$TOPOLOGY
CARDDIR=$TARGET/cards
OUTDIR=$TARGET/limits
STATUSDIR=$TARGET/status
RUNTIME_HOME=$CAMPAIGN/runtime_home/$TOPOLOGY/$BUNDLE_NAME

case "$CAMPAIGN:$HIGH_BASE:$LOW_BASE:$BUNDLE:$CMSSW" in
    /eos/*:/eos/*:/eos/*:/eos/*:/eos/*) ;;
    *) echo "campaign, inputs, bundle, and CMSSW must all be on EOS" >&2; exit 72 ;;
esac
case "$TOPOLOGY" in
    T2tt|T2bW|T2tb) ;;
    *) echo "unknown topology: $TOPOLOGY" >&2; exit 2 ;;
esac

mkdir -p "$CARDDIR" "$OUTDIR" "$STATUSDIR" "$RUNTIME_HOME/cache"
export XDG_CACHE_HOME=$RUNTIME_HOME/cache
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src"
eval "$(scramv1 runtime -sh)"
case "$(command -v combine):$(command -v combineCards.py)" in
    /eos/*:/eos/*) ;;
    *) echo "Combine runtime is not on EOS" >&2; exit 70 ;;
esac

validate_card() {
    python3 - "$1" "$2" <<'PY'
import re
import sys

path, mode = sys.argv[1:]
lines = open(path).read().splitlines()
observation = next((line for line in lines if line.startswith("bin ")), None)
if observation is None:
    raise SystemExit(1)
channels = observation.split()[1:]
high_sr = [channel for channel in channels if "SR_highdm_bin" in channel]
low_sr = [channel for channel in channels if re.search(r"(^|_)SR_Nb", channel)]
forbidden = [
    channel for channel in high_sr
    if re.search(r"SR_highdm_bin[0-5]$", channel)
]
expected_channels = 127 if mode == "combined" else 109
valid = (
    len(channels) == expected_channels
    and len(channels) == len(set(channels))
    and len(high_sr) == 73
    and len(low_sr) == (6 if mode == "combined" else 0)
    and not forbidden
    and not any("lowdm_" in channel for channel in channels)
)
raise SystemExit(0 if valid else 1)
PY
}

validate_limit() {
    python3 - "$1" <<'PY'
import math
import sys
import ROOT

root_file = ROOT.TFile.Open(sys.argv[1])
tree = root_file.Get("limit") if root_file else None
values = [(float(row.quantileExpected), float(row.limit)) for row in tree] if tree else []
if root_file:
    root_file.Close()
expected = (0.025, 0.16, 0.5, 0.84, 0.975)
valid = len(values) == 5 and all(
    any(
        abs(quantile - target) < 1.0e-5
        and math.isfinite(value)
        and value >= 0.0
        for quantile, value in values
    )
    for target in expected
)
raise SystemExit(0 if valid else 1)
PY
}

fail=0
completed=0
skipped=0
while read -r MASS MODE; do
    [[ -n "${MASS:-}" ]] || continue
    case "$MODE" in
        combined|highonly) ;;
        *) echo "invalid mode for $MASS: $MODE" >&2; fail=1; continue ;;
    esac
    HIGH_CARD=$HIGH_BASE/$TOPOLOGY/cards/datacard_$MASS.txt
    LOW_CARD=$LOW_BASE/$TOPOLOGY/cards/datacard_$MASS.txt
    CARD=$CARDDIR/datacard_$MASS.txt
    OUTPUT=$OUTDIR/higgsCombine_$MASS.AsymptoticLimits.mH120.root
    LOG=$OUTDIR/log_$MASS.txt

    if [[ -s "$OUTPUT" ]] && validate_limit "$OUTPUT"; then
        skipped=$((skipped + 1))
        continue
    fi
    if [[ ! -s "$HIGH_CARD" ]]; then
        echo "missing High-dM card: $HIGH_CARD" >&2
        fail=1
        continue
    fi
    if [[ "$MODE" == combined && ! -s "$LOW_CARD" ]]; then
        echo "missing Low-dM card: $LOW_CARD" >&2
        fail=1
        continue
    fi

    if [[ ! -s "$CARD" ]] || ! validate_card "$CARD" "$MODE"; then
        HIGH_ONLY=$CARDDIR/.highonly_${MASS}_${BUNDLE_NAME}.txt
        combineCards.py \
            --ic '.*highdm.*' \
            --xc '^SR_highdm_bin[0-5]$' \
            "$HIGH_CARD" > "$HIGH_ONLY"
        if [[ "$MODE" == combined ]]; then
            combineCards.py high="$HIGH_ONLY" low="$LOW_CARD" > "$CARD"
        else
            mv "$HIGH_ONLY" "$CARD"
        fi
        if [[ -e "$HIGH_ONLY" ]]; then
            unlink "$HIGH_ONLY"
        fi
        if ! validate_card "$CARD" "$MODE"; then
            echo "invalid constructed card: $CARD" >&2
            fail=1
            continue
        fi
    fi

    RUNDIR=$RUNTIME_HOME/work/$MASS
    RUN_OUTPUT=$RUNDIR/higgsCombine_$MASS.AsymptoticLimits.mH120.root
    mkdir -p "$RUNDIR"
    point_ok=0
    for attempt in 1 2 3; do
        if [[ -e "$RUN_OUTPUT" ]]; then
            unlink "$RUN_OUTPUT"
        fi
        cd "$RUNDIR"
        combine_timeout=2400
        minimizer_args=(--cminDefaultMinimizerStrategy 0)
        range_args=()
        # The very high-yield T2bW(200,100) point produces an empty limit tree
        # with Combine's default absolute r precision.  Keep this point-specific
        # numerical treatment out of the rest of the grid.
        if [[ "$TOPOLOGY:$MASS" == "T2bW:mStop200_mLSP100" ]]; then
            range_args=(
                --rMin 0
                --rMax 0.1
                --rAbsAcc 0.000001
                --rRelAcc 0.001
            )
        fi
        if [[ "$attempt" -eq 2 ]]; then
            combine_timeout=3600
            minimizer_args+=(--cminFallbackAlgo Minuit2,Migrad,1:0.1)
        elif [[ "$attempt" -eq 3 ]]; then
            combine_timeout=3600
            minimizer_args=(
                --cminDefaultMinimizerStrategy 1
                --cminFallbackAlgo Minuit2,Migrad,0:0.1
            )
        fi
        combine_status=0
        if timeout "$combine_timeout" combine \
            -M AsymptoticLimits \
            --run blind \
            "${minimizer_args[@]}" \
            "${range_args[@]}" \
            -n "_$MASS" \
            "$CARD" > "$LOG" 2>&1
        then
            if [[ -s "$RUN_OUTPUT" ]] && validate_limit "$RUN_OUTPUT"; then
                mv -f "$RUN_OUTPUT" "$OUTPUT"
                point_ok=1
                break
            fi
            combine_status=65
        else
            combine_status=$?
        fi
        printf 'attempt %d failed for %s (exit=%d, timeout=%ds)\n' \
            "$attempt" "$MASS" "$combine_status" "$combine_timeout" >> "$LOG"
        sleep 5
    done
    if [[ "$point_ok" -eq 1 ]] && validate_limit "$OUTPUT"; then
        completed=$((completed + 1))
    else
        echo "Combine failed after three attempts: $MASS" >&2
        fail=1
    fi
done < "$BUNDLE"

printf '{"bundle":"%s","topology":"%s","completed":%d,"skipped":%d,"exit_code":%d}\n' \
    "$BUNDLE_NAME" "$TOPOLOGY" "$completed" "$skipped" "$fail" \
    > "$STATUSDIR/$BUNDLE_NAME.json"
exit "$fail"
