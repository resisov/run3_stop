#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 DATACARD OUTPUT_DIR MASS" >&2
    exit 2
fi

CARD=$1
OUTDIR=$2
MASS=$3
: "${COMBINE_RUNTIME_SHA256:?Combine runtime checksum is required}"
: "${COMBINE_RUNTIME_ARCHIVE:?Combine runtime archive name is required}"
IMPACT_PARALLEL=${IMPACT_PARALLEL:-4}
IMPACT_MINIMIZER_STRATEGY=${IMPACT_MINIMIZER_STRATEGY:-0}
IMPACT_EXPECT_SIGNAL=${IMPACT_EXPECT_SIGNAL:-1}
IMPACT_R_MIN=${IMPACT_R_MIN:-0}
IMPACT_R_MAX=${IMPACT_R_MAX:-20}
IMPACT_PLOT=${IMPACT_PLOT:-1}
IMPACT_RESUME_DIR=${IMPACT_RESUME_DIR:-}

if ! [[ "$IMPACT_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMPACT_PARALLEL must be a positive integer" >&2
    exit 2
fi
if ! [[ "$IMPACT_MINIMIZER_STRATEGY" =~ ^[012]$ ]]; then
    echo "IMPACT_MINIMIZER_STRATEGY must be 0, 1 or 2" >&2
    exit 2
fi
if ! [[ "$IMPACT_EXPECT_SIGNAL" =~ ^[01]$ ]]; then
    echo "IMPACT_EXPECT_SIGNAL must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$IMPACT_PLOT" =~ ^[01]$ ]]; then
    echo "IMPACT_PLOT must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$IMPACT_R_MIN" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || \
   ! [[ "$IMPACT_R_MAX" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v lo="$IMPACT_R_MIN" -v hi="$IMPACT_R_MAX" 'BEGIN { exit !(lo < hi) }'; then
    echo "IMPACT_R_MIN and IMPACT_R_MAX must be numbers with IMPACT_R_MIN < IMPACT_R_MAX" >&2
    exit 2
fi
RANGE_ARGS=(--setParameterRanges "r=${IMPACT_R_MIN},${IMPACT_R_MAX}")
if [[ -n "$IMPACT_RESUME_DIR" ]]; then
    case "$IMPACT_RESUME_DIR" in
        /eos/*) ;;
        *) echo "resume directory must be on EOS" >&2; exit 2 ;;
    esac
    for checksum in "${IMPACT_RESUME_CARD_SHA256:-}" \
        "${IMPACT_RESUME_WORKSPACE_SHA256:-}" "${IMPACT_RESUME_INITIAL_SHA256:-}"; do
        if ! [[ "$checksum" =~ ^[0-9a-f]{64}$ ]]; then
            echo "resume card, workspace and initial-fit checksums are required" >&2
            exit 2
        fi
    done
    printf '%s  %s\n' "$IMPACT_RESUME_CARD_SHA256" "$CARD" | sha256sum -c -
fi

case "$OUTDIR" in
    /eos/*) ;;
    *) echo "output directory must be on EOS: $OUTDIR" >&2; exit 2 ;;
esac
case "$OUTDIR" in
    /tmp/*|/afs/*) echo "system /tmp and AFS outputs are forbidden" >&2; exit 2 ;;
esac

: "${_CONDOR_SCRATCH_DIR:?Condor scratch directory is required}"
SCRATCH_BASE="$_CONDOR_SCRATCH_DIR"
RUNTIME_ARCHIVE="$SCRATCH_BASE/$COMBINE_RUNTIME_ARCHIVE"
WORKDIR="$SCRATCH_BASE/work"
CMSSW="$SCRATCH_BASE/CMSSW_14_1_0_pre4"
RESULTDIR="$OUTDIR/work"
mkdir -p "$WORKDIR" "$SCRATCH_BASE/cache" "$RESULTDIR"
export TMPDIR="$SCRATCH_BASE"
export XDG_CACHE_HOME="$SCRATCH_BASE/cache"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export CVS_RSH="${CVS_RSH:-ssh}"

echo "$COMBINE_RUNTIME_SHA256  $RUNTIME_ARCHIVE" | sha256sum -c -
tar -xzf "$RUNTIME_ARCHIVE" -C "$SCRATCH_BASE"
rm -f "$RUNTIME_ARCHIVE"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src"
scramv1 b ProjectRename >/dev/null
eval "$(scramv1 runtime -sh)"
command -v text2workspace.py >/dev/null
command -v combineTool.py >/dev/null
case "$(command -v combineTool.py)" in "$CMSSW"/*) ;; *) exit 70 ;; esac
cd "$WORKDIR"

WORKSPACE="workspace_mStop${MASS}_mLSP500.root"
IMPACT_BASE="impacts_mStop${MASS}_mLSP500"
INITIAL_FIT="higgsCombine_initialFit_Test.MultiDimFit.mH${MASS}.root"

preserve_results() {
    local result=$?
    trap - EXIT
    set +e
    if [[ "$result" -ne 0 ]]; then
        printf '{"status":"failed","exit_code":%s,"asimov_expect_signal":%s}\n' \
            "$result" "$IMPACT_EXPECT_SIGNAL" > impact_status.json
    fi
    cp -a "$WORKDIR/." "$RESULTDIR/"
    local copy_result=$?
    if [[ "$copy_result" -ne 0 ]]; then
        echo "impact result preservation failed: $RESULTDIR" >&2
        exit 74
    fi
    exit "$result"
}
trap preserve_results EXIT

if [[ -n "$IMPACT_RESUME_DIR" ]]; then
    printf '%s  %s\n' "$IMPACT_RESUME_WORKSPACE_SHA256" "$IMPACT_RESUME_DIR/$WORKSPACE" \
        "$IMPACT_RESUME_INITIAL_SHA256" "$IMPACT_RESUME_DIR/$INITIAL_FIT" | sha256sum -c -
    cp -a "$IMPACT_RESUME_DIR/." "$WORKDIR/"
    printf '%s  %s\n' "$IMPACT_RESUME_WORKSPACE_SHA256" "$WORKSPACE" \
        "$IMPACT_RESUME_INITIAL_SHA256" "$INITIAL_FIT" | sha256sum -c -
    echo "resuming existing workspace and initial fit; only missing nuisance fits will run"
else
    text2workspace.py "$CARD" -m "$MASS" -o "$WORKSPACE" > text2workspace.log 2>&1
    if ! combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
        --robustFit 1 --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
        -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" > impacts_initial.log 2>&1; then
        combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
            --robustFit 1 --cminDefaultMinimizerStrategy 1 \
            -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" >> impacts_initial.log 2>&1
    fi
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
        --robustFit 1 --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
        -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" --parallel "$IMPACT_PARALLEL" \
        > impacts_fits.log 2>&1 || true
fi
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
    -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1 || true

MISSING=$(sed -n "s/^Missing inputs: //p" impacts_collect.log \
    | tr -d "' \"[]" | tail -n 1)
if [[ -n "$MISSING" ]]; then
    RETRY_STRATEGY=1
    [[ -z "$IMPACT_RESUME_DIR" ]] || RETRY_STRATEGY=$IMPACT_MINIMIZER_STRATEGY
    printf 'Retrying missing nuisance fits with strategy %s: %s\n' "$RETRY_STRATEGY" "$MISSING" \
        >> impacts_fits.log
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
        --named "$MISSING" --robustFit 1 --cminDefaultMinimizerStrategy "$RETRY_STRATEGY" \
        -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" --parallel "$IMPACT_PARALLEL" \
        >> impacts_fits.log 2>&1 || true
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
        -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1
fi
if grep -q '^Missing inputs:' impacts_collect.log; then
    echo "impact collection is incomplete; see $OUTDIR/work/impacts_collect.log" >&2
    exit 1
fi
test -s "$IMPACT_BASE.json"
if [[ "$IMPACT_PLOT" = 1 ]]; then
    plotImpacts.py -i "$IMPACT_BASE.json" -o "$IMPACT_BASE" > impacts_plot.log 2>&1
    if command -v pdftoppm >/dev/null 2>&1; then
        pdftoppm -f 1 -singlefile -png -r 160 "$IMPACT_BASE.pdf" "$IMPACT_BASE" > pdftoppm.log 2>&1
    fi
    printf '{\n  "status": "complete",\n  "benchmark": "mStop%s_mLSP500",\n  "asimov_expect_signal": %s,\n  "signal_strength_range": [%s, %s],\n  "workspace": "%s",\n  "json": "%s.json",\n  "pdf": "%s.pdf"\n}\n' \
        "$MASS" "$IMPACT_EXPECT_SIGNAL" "$IMPACT_R_MIN" "$IMPACT_R_MAX" "$WORKSPACE" "$IMPACT_BASE" "$IMPACT_BASE" > impact_status.json
else
    printf '{\n  "status": "fits_complete",\n  "plots_status": "pending_local",\n  "benchmark": "mStop%s_mLSP500",\n  "asimov_expect_signal": %s,\n  "signal_strength_range": [%s, %s],\n  "workspace": "%s",\n  "json": "%s.json"\n}\n' \
        "$MASS" "$IMPACT_EXPECT_SIGNAL" "$IMPACT_R_MIN" "$IMPACT_R_MAX" "$WORKSPACE" "$IMPACT_BASE" > impact_status.json
fi

echo "impact fits complete: $RESULTDIR/$IMPACT_BASE.json"
