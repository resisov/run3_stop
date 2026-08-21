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

if ! [[ "$IMPACT_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMPACT_PARALLEL must be a positive integer" >&2
    exit 2
fi
if ! [[ "$IMPACT_MINIMIZER_STRATEGY" =~ ^[01]$ ]]; then
    echo "IMPACT_MINIMIZER_STRATEGY must be 0 or 1" >&2
    exit 2
fi

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
    -t -1 --expectSignal 1 > impacts_initial.log 2>&1; then
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
        --robustFit 1 --cminDefaultMinimizerStrategy 1 \
        -t -1 --expectSignal 1 >> impacts_initial.log 2>&1
fi

# Strategy 0 is substantially faster for the large autoMCStats model.  Keep
# robustFit enabled, collect the successful results, and rerun only missing
# nuisance fits with strategy 1 instead of repeating the complete campaign.
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
    --robustFit 1 --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
    -t -1 --expectSignal 1 --parallel "$IMPACT_PARALLEL" \
    > impacts_fits.log 2>&1 || true
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
    -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1 || true

MISSING=$(sed -n "s/^Missing inputs: \[\(.*\)\]$/\1/p" impacts_collect.log \
    | tr -d "' " | tail -n 1)
if [[ -n "$MISSING" ]]; then
    printf 'Retrying missing nuisance fits with strategy 1: %s\n' "$MISSING" \
        >> impacts_fits.log
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
        --named "$MISSING" --robustFit 1 --cminDefaultMinimizerStrategy 1 \
        -t -1 --expectSignal 1 --parallel "$IMPACT_PARALLEL" \
        >> impacts_fits.log 2>&1
    combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
        -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1
fi
if grep -q '^Missing inputs:' impacts_collect.log; then
    echo "impact collection is incomplete; see $OUTDIR/work/impacts_collect.log" >&2
    exit 1
fi
plotImpacts.py -i "$IMPACT_BASE.json" -o "$IMPACT_BASE" > impacts_plot.log 2>&1
pdftoppm -f 1 -singlefile -png -r 160 "$IMPACT_BASE.pdf" "$IMPACT_BASE" > pdftoppm.log 2>&1

printf '{\n  "status": "complete",\n  "benchmark": "mStop%s_mLSP500",\n  "asimov_expect_signal": 1,\n  "workspace": "%s",\n  "json": "%s.json",\n  "pdf": "%s.pdf",\n  "png": "%s.png"\n}\n' \
    "$MASS" "$WORKSPACE" "$IMPACT_BASE" "$IMPACT_BASE" "$IMPACT_BASE" > impact_status.json

echo "impact complete: $OUTDIR/work/$IMPACT_BASE.pdf"
