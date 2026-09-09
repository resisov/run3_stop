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
IMPACT_VERBOSITY=${IMPACT_VERBOSITY:-0}
IMPACT_MINIMIZER_PRECISION=${IMPACT_MINIMIZER_PRECISION:-}
IMPACT_STEP_SIZE=${IMPACT_STEP_SIZE:-}
IMPACT_ROBUST_STRATEGY=${IMPACT_ROBUST_STRATEGY:-}
IMPACT_SUBSET_ONLY=${IMPACT_SUBSET_ONLY:-0}
IMPACT_EXPECT_SIGNAL=${IMPACT_EXPECT_SIGNAL:-1}
IMPACT_R_MIN=${IMPACT_R_MIN:-0}
IMPACT_R_MAX=${IMPACT_R_MAX:-20}
IMPACT_PLOT=${IMPACT_PLOT:-1}
IMPACT_RESUME_DIR=${IMPACT_RESUME_DIR:-}
IMPACT_RESUME_NUISANCES=${IMPACT_RESUME_NUISANCES:-}

if ! [[ "$IMPACT_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMPACT_PARALLEL must be a positive integer" >&2
    exit 2
fi
if ! [[ "$IMPACT_MINIMIZER_STRATEGY" =~ ^[012]$ ]]; then
    echo "IMPACT_MINIMIZER_STRATEGY must be 0, 1 or 2" >&2
    exit 2
fi
if ! [[ "$IMPACT_VERBOSITY" =~ ^[0-3]$ ]]; then
    echo "IMPACT_VERBOSITY must be 0, 1, 2 or 3" >&2
    exit 2
fi
if [[ -n "$IMPACT_MINIMIZER_PRECISION" ]] && {
    ! [[ "$IMPACT_MINIMIZER_PRECISION" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    ! awk -v p="$IMPACT_MINIMIZER_PRECISION" 'BEGIN { exit !(p > 0 && p < 1) }';
}; then
    echo "IMPACT_MINIMIZER_PRECISION must be between 0 and 1" >&2
    exit 2
fi
if [[ -n "$IMPACT_STEP_SIZE" ]] && {
    ! [[ "$IMPACT_STEP_SIZE" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    ! awk -v p="$IMPACT_STEP_SIZE" 'BEGIN { exit !(p > 0 && p < 1) }';
}; then
    echo "IMPACT_STEP_SIZE must be between 0 and 1" >&2
    exit 2
fi
if [[ -n "$IMPACT_ROBUST_STRATEGY" ]] && ! [[ "$IMPACT_ROBUST_STRATEGY" =~ ^[012]$ ]]; then
    echo "IMPACT_ROBUST_STRATEGY must be 0, 1 or 2" >&2
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
if [[ -n "$IMPACT_RESUME_NUISANCES" ]]; then
    if [[ -z "$IMPACT_RESUME_DIR" ]] || \
        ! [[ "$IMPACT_RESUME_NUISANCES" =~ ^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ ]]; then
        echo "resume nuisance names require a resume directory and a comma-separated list" >&2
        exit 2
    fi
fi
if ! [[ "$IMPACT_SUBSET_ONLY" =~ ^[01]$ ]] || \
    [[ "$IMPACT_SUBSET_ONLY" = 1 && -z "$IMPACT_RESUME_NUISANCES" ]]; then
    echo "IMPACT_SUBSET_ONLY requires an explicitly named resume subset" >&2
    exit 2
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
if [[ -n "${IMPACT_MINIMIZER_PRECISION:-}" ]]; then
    RANGE_ARGS+=(--cminDefaultMinimizerPrecision "$IMPACT_MINIMIZER_PRECISION")
fi
if [[ -n "${IMPACT_STEP_SIZE:-}" ]]; then
    RANGE_ARGS+=(--stepSize "$IMPACT_STEP_SIZE")
fi
if [[ -n "${IMPACT_ROBUST_STRATEGY:-}" ]]; then
    RANGE_ARGS+=(--setRobustFitStrategy "$IMPACT_ROBUST_STRATEGY")
fi

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
    if [[ -n "$IMPACT_RESUME_NUISANCES" ]]; then
        IFS=, read -r -a RESUME_NAMES <<< "$IMPACT_RESUME_NUISANCES"
        for nuisance in "${RESUME_NAMES[@]}"; do
            if [[ -e "higgsCombine_paramFit_Test_${nuisance}.MultiDimFit.mH${MASS}.root" ]]; then
                echo "resume target already has a ROOT file: $nuisance; use a validated-only checkpoint" >&2
                exit 2
            fi
        done
        printf 'Resuming named nuisance fits with strategy %s: %s\n' \
            "$IMPACT_MINIMIZER_STRATEGY" "$IMPACT_RESUME_NUISANCES" >> impacts_fits.log
        combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" --doFits \
            -v "${IMPACT_VERBOSITY:-0}" \
            --named "$IMPACT_RESUME_NUISANCES" --robustFit 1 \
            --cminDefaultMinimizerStrategy "$IMPACT_MINIMIZER_STRATEGY" \
            -t -1 --expectSignal "$IMPACT_EXPECT_SIGNAL" "${RANGE_ARGS[@]}" \
            --parallel "$IMPACT_PARALLEL" >> impacts_fits.log 2>&1 || true
    fi
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
validate_fit_roots() {
    python3 - "$1" "$MASS" "$IMPACT_BASE.json" <<'PY'
import json
import math
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
names = sys.argv[1].split(",") if sys.argv[1] else [
    item["name"] for item in json.load(open(sys.argv[3]))["params"]
]
results = []
workspace_source = None
for name in names:
    path = "higgsCombine_paramFit_Test_{}.MultiDimFit.mH{}.root".format(name, sys.argv[2])
    source = None
    item = {"name": name, "file": path, "valid": False}
    try:
        source = ROOT.TFile.Open(path)
        if not source or source.IsZombie():
            raise ValueError("unreadable ROOT")
        tree = source.Get("limit")
        if not tree:
            raise ValueError("missing limit tree")
        item["entries"] = int(tree.GetEntries())
        values = [[float(event.r), float(getattr(event, name)), float(event.deltaNLL)] for event in tree]
        if len(values) != 3 or not all(math.isfinite(v) for row in values for v in row):
            raise ValueError("expected three finite entries")
        nominal, lower, upper = [row[1] for row in values]
        item["parameter_values"] = [nominal, lower, upper]
        tolerance = 1e-6 * max(1.0, abs(nominal), abs(lower), abs(upper))
        if nominal == lower == upper:
            raise ValueError("identical parameter endpoints")
        if not lower <= nominal + tolerance or not nominal <= upper + tolerance:
            raise ValueError("parameter endpoints do not bracket nominal")
        if lower == nominal or upper == nominal:
            if workspace_source is None:
                workspace_source = ROOT.TFile.Open("workspace_mStop{}_mLSP500.root".format(sys.argv[2]))
            if not workspace_source or workspace_source.IsZombie():
                raise ValueError("cannot verify parameter bounds: unreadable workspace")
            workspace = workspace_source.Get("w")
            parameter = workspace.var(name) if workspace else None
            if not parameter:
                raise ValueError("cannot verify parameter bounds: missing workspace parameter")
            bounds = [float(parameter.getMin()), float(parameter.getMax())]
            item["parameter_bounds"] = bounds
            if lower == nominal and not math.isclose(nominal, bounds[0], rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("zero-width lower endpoint away from lower bound")
            if upper == nominal and not math.isclose(nominal, bounds[1], rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("zero-width upper endpoint away from upper bound")
            item["nominal_at_boundary"] = True
        item["valid"] = True
    except Exception as error:
        item["error"] = str(error)
    finally:
        if source:
            source.Close()
    results.append(item)
if workspace_source:
    workspace_source.Close()
valid = bool(results) and all(item["valid"] for item in results)
print(json.dumps({"valid": valid, "scope": "finite bracketed endpoints; zero-width sides require a verified parameter boundary; convergence review pending", "results": results}))
sys.exit(0 if valid else 1)
PY
}

if [[ "${IMPACT_SUBSET_ONLY:-0}" = 1 ]]; then
    if ! validate_fit_roots "$IMPACT_RESUME_NUISANCES" > subset_fit_validation.json
    then
        echo "named impact subset is incomplete; see subset_fit_validation.json" >&2
        exit 1
    fi
    printf '{"status":"subset_fits_complete","full_collection_pending":true,"asimov_expect_signal":%s,"subset_validation":"subset_fit_validation.json"}\n' \
        "$IMPACT_EXPECT_SIGNAL" > impact_status.json
    exit 0
fi
combineTool.py -M Impacts -d "$WORKSPACE" -m "$MASS" \
    -o "$IMPACT_BASE.json" > impacts_collect.log 2>&1 || true

MISSING=$(sed -n "s/^Missing inputs: //p" impacts_collect.log \
    | tr -d "' \"[]" | tail -n 1)
if [[ -n "$MISSING" && -z "$IMPACT_RESUME_DIR" ]]; then
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
if ! validate_fit_roots "" > fit_endpoint_validation.json; then
    echo "impact endpoints failed validation; see $RESULTDIR/fit_endpoint_validation.json" >&2
    exit 1
fi
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
