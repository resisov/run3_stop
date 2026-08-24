#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 DATACARD OUTPUT_DIR MASS" >&2
    exit 2
fi

CARD=$1
OUTDIR=$2
MASS=$3
CMSSW=/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4
IMPACT_PARALLEL=${IMPACT_PARALLEL:-4}
IMPACT_MINIMIZER_STRATEGY=${IMPACT_MINIMIZER_STRATEGY:-0}
IMPACT_EXPECT_SIGNAL=${IMPACT_EXPECT_SIGNAL:-1}
IMPACT_R_MIN=${IMPACT_R_MIN:-0}
IMPACT_R_MAX=${IMPACT_R_MAX:-20}

if ! [[ "$IMPACT_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMPACT_PARALLEL must be a positive integer" >&2
    exit 2
fi
if ! [[ "$IMPACT_MINIMIZER_STRATEGY" =~ ^[01]$ ]]; then
    echo "IMPACT_MINIMIZER_STRATEGY must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$IMPACT_EXPECT_SIGNAL" =~ ^[01]$ ]]; then
    echo "IMPACT_EXPECT_SIGNAL must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$IMPACT_R_MIN" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || \
   ! [[ "$IMPACT_R_MAX" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v lo="$IMPACT_R_MIN" -v hi="$IMPACT_R_MAX" 'BEGIN { exit !(lo < hi) }'; then
    echo "IMPACT_R_MIN and IMPACT_R_MAX must be numbers with IMPACT_R_MIN < IMPACT_R_MAX" >&2
    exit 2
fi
RANGE_ARGS=(--setParameterRanges "r=${IMPACT_R_MIN},${IMPACT_R_MAX}")

case "$OUTDIR" in
    /eos/*) ;;
    *) echo "output directory must be on EOS: $OUTDIR" >&2; exit 2 ;;
esac
case "$OUTDIR" in
    /tmp/*|/afs/*) echo "system /tmp and AFS outputs are forbidden" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR"/{tmp,cache,work}
export TMPDIR="$OUTDIR/tmp"
export XDG_CACHE_HOME="$OUTDIR/cache"
export PYTHONNOUSERSITE=1
export CVS_RSH="${CVS_RSH:-ssh}"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src"
eval "$(scramv1 runtime -sh)"
cd "$OUTDIR/work"

WORKSPACE="workspace_mStop${MASS}_mLSP500.root"
IMPACT_BASE="impacts_mStop${MASS}_mLSP500"

text2workspace.py "$CARD" -m "$MASS" -o "$WORKSPACE" > text2workspace.log 2>&1
if ! combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
    --robustFit 1 --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
    -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" > impacts_initial.log 2>&1; then
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
        --robustFit 1 --cminDefaultMinimizerStrategy 1 \
        -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" >> impacts_initial.log 2>&1
fi

# Strategy 0 is substantially faster for the large autoMCStats model.  Keep
# robustFit enabled, collect the successful results, and rerun only missing
# nuisance fits with strategy 1 instead of repeating the complete campaign.
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
    --robustFit 1 --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
    -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" --parallel "$IMPACT_PARALLEL" \
    > impacts_fits.log 2>&1 || true
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
    -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1 || true

MISSING=$(sed -n "s/^Missing inputs: //p" impacts_collect.log \
    | tr -d "' \"[]" | tail -n 1)
if [[ -n "$MISSING" ]]; then
    printf 'Retrying missing nuisance fits with strategy 1: %s\n' "$MISSING" \
        >> impacts_fits.log
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
        --named "$MISSING" --robustFit 1 --cminDefaultMinimizerStrategy 1 \
        -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" --parallel "$IMPACT_PARALLEL" \
        >> impacts_fits.log 2>&1
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
        -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1
fi
if grep -q '^Missing inputs:' impacts_collect.log; then
    echo "impact collection is incomplete; see $OUTDIR/work/impacts_collect.log" >&2
    exit 1
fi
plotImpacts.py -i "$IMPACT_BASE.json" -o "$IMPACT_BASE" > impacts_plot.log 2>&1
if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -f 1 -singlefile -png -r 160 "$IMPACT_BASE.pdf" "$IMPACT_BASE" > pdftoppm.log 2>&1
else
    printf 'pdftoppm unavailable on worker; recover PNG from the validated PDF on LXPLUS\n' > pdftoppm.log
fi

printf '{\n  "status": "complete",\n  "benchmark": "mStop%s_mLSP500",\n  "asimov_expect_signal": %s,\n  "signal_strength_range": [%s, %s],\n  "workspace": "%s",\n  "json": "%s.json",\n  "pdf": "%s.pdf",\n  "png": "%s.png"\n}\n' \
    "$MASS" "$IMPACT_EXPECT_SIGNAL" "$IMPACT_R_MIN" "$IMPACT_R_MAX" "$WORKSPACE" "$IMPACT_BASE" "$IMPACT_BASE" "$IMPACT_BASE" > impact_status.json

echo "impact complete: $OUTDIR/work/$IMPACT_BASE.pdf"
