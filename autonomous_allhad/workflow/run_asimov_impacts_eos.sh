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

case "$OUTDIR" in
    /eos/*) ;;
    *) echo "output directory must be on EOS: $OUTDIR" >&2; exit 2 ;;
esac
case "$OUTDIR" in
    /tmp/*|/afs/*) echo "system /tmp and AFS outputs are forbidden" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR"/{home,tmp,cache,work}
export HOME="$OUTDIR/home"
export TMPDIR="$OUTDIR/tmp"
export XDG_CACHE_HOME="$OUTDIR/cache"
export PYTHONNOUSERSITE=1

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src"
eval "$(scramv1 runtime -sh)"
cd "$OUTDIR/work"

WORKSPACE="workspace_mStop${MASS}_mLSP500.root"
IMPACT_BASE="impacts_mStop${MASS}_mLSP500"

text2workspace.py "$CARD" -m "$MASS" -o "$WORKSPACE" > text2workspace.log 2>&1
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doInitialFit \
    --robustFit 1 -t -1 --expectSignal 1 > impacts_initial.log 2>&1
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
    --robustFit 1 -t -1 --expectSignal 1 --parallel 16 > impacts_fits.log 2>&1
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
    -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1
if grep -q '^Missing inputs:' impacts_collect.log; then
    echo "impact collection is incomplete; see $OUTDIR/work/impacts_collect.log" >&2
    exit 1
fi
plotImpacts.py -i "$IMPACT_BASE.json" -o "$IMPACT_BASE" > impacts_plot.log 2>&1
pdftoppm -f 1 -singlefile -png -r 160 "$IMPACT_BASE.pdf" "$IMPACT_BASE" > pdftoppm.log 2>&1

printf '{\n  "status": "complete",\n  "benchmark": "mStop%s_mLSP500",\n  "asimov_expect_signal": 1,\n  "workspace": "%s",\n  "json": "%s.json",\n  "pdf": "%s.pdf",\n  "png": "%s.png"\n}\n' \
    "$MASS" "$WORKSPACE" "$IMPACT_BASE" "$IMPACT_BASE" "$IMPACT_BASE" > impact_status.json

echo "impact complete: $OUTDIR/work/$IMPACT_BASE.pdf"
