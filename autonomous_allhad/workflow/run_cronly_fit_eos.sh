#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CARD_2024 CARD_2025 OUTPUT_DIR" >&2
    exit 2
fi

CARD_2024=$1
CARD_2025=$2
OUTPUT_DIR=$3
CMSSW=/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4
FIT_STRATEGY=${FIT_STRATEGY:-0}

case "$FIT_STRATEGY" in
    0|1) ;;
    *) echo "FIT_STRATEGY must be 0 or 1" >&2; exit 2 ;;
esac

case "$OUTPUT_DIR" in
    /eos/*) ;;
    *) echo "output directory must be on EOS: $OUTPUT_DIR" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
export PYTHONNOUSERSITE=1
export CVS_RSH="${CVS_RSH:-ssh}"
export TMPDIR="$OUTPUT_DIR/tmp"
export XDG_CACHE_HOME="$OUTPUT_DIR/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src"
eval "$(scramv1 runtime -sh)"
cd "$OUTPUT_DIR"

CR_CARD="$OUTPUT_DIR/datacard_cronly_2024_2025.txt"
WORKSPACE="$OUTPUT_DIR/workspace_cronly_2024_2025.root"

combineCards.py --ic='^(LLCR|QCDCR|GCR)_' \
    --ic='^SR_highdm_bin0$' \
    "y2024=$CARD_2024" "y2025=$CARD_2025" > "$CR_CARD"

# The fit workspace must contain control regions only.  Validate channel names
# from every shapes directive before constructing the RooWorkspace.  One SR
# bin per year is retained solely to define the signal POI and is hard-masked.
if awk '$1 == "shapes" && $3 !~ /^y20(24|25)_(LLCR|QCDCR|GCR)_/ && $3 !~ /^y20(24|25)_SR_highdm_bin0$/ {bad=1} END {exit bad}' "$CR_CARD"; then
    :
else
    echo "CR-only card contains a non-control likelihood channel" >&2
    exit 1
fi

text2workspace.py --channel-masks "$CR_CARD" -m 120 -o "$WORKSPACE" \
    > text2workspace.log 2>&1

combine -M FitDiagnostics "$WORKSPACE" -m 120 \
    --robustFit 1 \
    --cminDefaultMinimizerStrategy "$FIT_STRATEGY" \
    --skipSBFit \
    --saveWorkspace \
    --saveShapes \
    --saveWithUncertainties \
    --saveNormalizations \
    --setParameters mask_y2024_SR_highdm_bin0=1,mask_y2025_SR_highdm_bin0=1,r=0 \
    --freezeParameters mask_y2024_SR_highdm_bin0,mask_y2025_SR_highdm_bin0,r \
    -n _cronly_2024_2025 \
    > fitdiagnostics.log 2>&1

python3 - "$CR_CARD" "$WORKSPACE" "$OUTPUT_DIR/fitDiagnostics_cronly_2024_2025.root" \
    "$OUTPUT_DIR/fit_status.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import ROOT

card, workspace, fit_path, output = map(Path, sys.argv[1:])
root_file = ROOT.TFile.Open(str(fit_path))
if not root_file or root_file.IsZombie():
    raise SystemExit(f"unreadable FitDiagnostics output: {fit_path}")
fit = root_file.Get("fit_b")
if fit is None:
    raise SystemExit("FitDiagnostics output does not contain fit_b")

channels = []
masked_auxiliary_channels = []
control_channels = []
for line in card.read_text().splitlines():
    fields = line.split()
    if len(fields) >= 4 and fields[0] == "shapes":
        channels.append(fields[2])
channels = sorted(set(channels))
for channel in channels:
    if channel in {"y2024_SR_highdm_bin0", "y2025_SR_highdm_bin0"}:
        masked_auxiliary_channels.append(channel)
    elif not channel.startswith(("y2024_LLCR_", "y2024_QCDCR_", "y2024_GCR_", "y2025_LLCR_", "y2025_QCDCR_", "y2025_GCR_")):
        raise SystemExit(f"non-control channel survived: {channel}")
    else:
        control_channels.append(channel)

workspace_file = ROOT.TFile.Open(str(workspace))
if not workspace_file or workspace_file.IsZombie():
    raise SystemExit(f"unreadable input RooWorkspace file: {workspace}")
roo_workspace = workspace_file.Get("w")
if roo_workspace is None:
    raise SystemExit("input RooWorkspace w is absent")
mask_validation = {}
fit_log = output.with_name("fitdiagnostics.log").read_text()
if ">>> 2 out of 276 channels masked" not in fit_log:
    raise SystemExit("Combine did not report exactly two masked channels")
for channel in masked_auxiliary_channels:
    variable = roo_workspace.var(f"mask_{channel}")
    if variable is None:
        raise SystemExit(f"channel mask is absent for {channel}")
    expected = f"Set Default Value of Parameter mask_{channel} To : 1"
    if expected not in fit_log:
        raise SystemExit(f"runtime mask evidence is absent for {channel}")
    mask_validation[channel] = {
        "workspace_default": float(variable.getVal()),
        "runtime_value": 1.0,
        "runtime_frozen": True,
        "runtime_evidence": [
            expected,
            ">>> 2 out of 276 channels masked",
            "--freezeParameters mask_y2024_SR_highdm_bin0,mask_y2025_SR_highdm_bin0,r",
        ],
    }

payload = {
    "status": "complete" if int(fit.status()) == 0 else "fit_failed",
    "covariance_quality": int(fit.covQual()),
    "edm": float(fit.edm()),
    "fit_status": int(fit.status()),
    "card_channels": channels,
    "likelihood_channels": control_channels,
    "likelihood_channel_count": len(control_channels),
    "masked_auxiliary_channels": masked_auxiliary_channels,
    "mask_validation": mask_validation,
    "fit_input_regions": ["LLCR", "QCDCR", "GCR"],
    "vr_observation_in_likelihood": False,
    "sr_observation_in_likelihood": False,
    "workspace": str(workspace),
    "fit_diagnostics": str(fit_path),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if payload["status"] != "complete" or payload["covariance_quality"] < 2:
    raise SystemExit(json.dumps(payload, sort_keys=True))

initial = fit.floatParsInit()
final = fit.floatParsFinal()
names = [str(final.at(index).GetName()) for index in range(final.getSize())]
covariance_root = fit.covarianceMatrix()
covariance = np.asarray(
    [
        [float(covariance_root[row][column]) for column in range(len(names))]
        for row in range(len(names))
    ],
    dtype=float,
)
if not np.all(np.isfinite(covariance)):
    raise SystemExit("fit covariance contains a nonfinite entry")
max_asymmetry = float(np.max(np.abs(covariance - covariance.T))) if names else 0.0
eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
parameters = {}
for index, name in enumerate(names):
    fitted = final.at(index)
    before = initial.find(name)
    initial_value = float(before.getVal()) if before is not None else None
    initial_error = float(before.getError()) if before is not None else None
    fitted_error = float(fitted.getError())
    constrained = bool(initial_error is not None and initial_error > 0.0)
    parameters[name] = {
        "initial": initial_value,
        "initial_error": initial_error,
        "value": float(fitted.getVal()),
        "error": fitted_error,
        "pull": (
            (float(fitted.getVal()) - initial_value) / initial_error
            if constrained
            else None
        ),
        "constraint": fitted_error / initial_error if constrained else None,
        "bounded": bool(fitted.hasMin() or fitted.hasMax()),
    }
fit_parameters = {
    "status": "complete",
    "fit": "background-only observed CR data",
    "parameter_order": names,
    "parameters": parameters,
    "covariance": covariance.tolist(),
    "covariance_validation": {
        "dimension": len(names),
        "finite": True,
        "max_asymmetry": max_asymmetry,
        "minimum_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else None,
        "maximum_eigenvalue": float(eigenvalues[-1]) if len(eigenvalues) else None,
    },
}
output.with_name("fit_parameters.json").write_text(
    json.dumps(fit_parameters, indent=2, sort_keys=True) + "\n"
)
PY

echo "CR-only fit complete: $OUTPUT_DIR/fit_status.json"
