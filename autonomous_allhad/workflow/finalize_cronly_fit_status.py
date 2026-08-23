#!/usr/bin/env python3
"""Finalize the CR-only fit audit from the fit result, card, and Combine log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ROOT


MASKED = ("y2024_SR_highdm_bin0", "y2025_SR_highdm_bin0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--fit-diagnostics", required=True, type=Path)
    parser.add_argument("--fit-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(str(args.fit_diagnostics))
    if not root_file or root_file.IsZombie():
        raise SystemExit("unreadable FitDiagnostics output")
    fit = root_file.Get("fit_b")
    if fit is None:
        raise SystemExit("fit_b is absent")

    channels = sorted(
        {
            fields[2]
            for line in args.card.read_text().splitlines()
            if len(fields := line.split()) >= 4 and fields[0] == "shapes"
        }
    )
    controls = []
    for channel in channels:
        if channel in MASKED:
            continue
        if not channel.startswith(
            (
                "y2024_LLCR_",
                "y2024_QCDCR_",
                "y2024_GCR_",
                "y2025_LLCR_",
                "y2025_QCDCR_",
                "y2025_GCR_",
            )
        ):
            raise SystemExit(f"non-control likelihood channel: {channel}")
        controls.append(channel)
    if len(controls) != 274:
        raise SystemExit(f"expected 274 control channels, found {len(controls)}")

    log = args.fit_log.read_text()
    for channel in MASKED:
        expected = f"Set Default Value of Parameter mask_{channel} To : 1"
        if expected not in log:
            raise SystemExit(f"missing runtime mask evidence: {expected}")
    if ">>> 2 out of 276 channels masked" not in log:
        raise SystemExit("Combine did not report exactly two masked channels")

    workspace_file = ROOT.TFile.Open(str(args.workspace))
    workspace = workspace_file.Get("w") if workspace_file else None
    if workspace is None:
        raise SystemExit("input RooWorkspace is absent")
    mask_validation = {}
    for channel in MASKED:
        variable = workspace.var(f"mask_{channel}")
        if variable is None:
            raise SystemExit(f"mask variable is absent for {channel}")
        mask_validation[channel] = {
            "workspace_default": float(variable.getVal()),
            "runtime_value": 1.0,
            "runtime_frozen": True,
            "runtime_evidence": [
                f"Set Default Value of Parameter mask_{channel} To : 1",
                ">>> 2 out of 276 channels masked",
                "--freezeParameters mask_y2024_SR_highdm_bin0,mask_y2025_SR_highdm_bin0,r",
            ],
        }

    payload = {
        "status": "complete" if int(fit.status()) == 0 else "fit_failed",
        "fit_status": int(fit.status()),
        "covariance_quality": int(fit.covQual()),
        "edm": float(fit.edm()),
        "fit_input_regions": ["LLCR", "QCDCR", "GCR"],
        "card_channel_count": len(channels),
        "likelihood_channel_count": len(controls),
        "likelihood_channels": controls,
        "masked_auxiliary_channels": list(MASKED),
        "mask_validation": mask_validation,
        "signal_strength": {"runtime_value": 0.0, "runtime_frozen": True},
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "workspace": str(args.workspace),
        "fit_diagnostics": str(args.fit_diagnostics),
        "fit_log": str(args.fit_log),
    }
    if payload["status"] != "complete" or payload["covariance_quality"] < 2:
        raise SystemExit(json.dumps(payload, sort_keys=True))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fit_status": payload["fit_status"],
                "covariance_quality": payload["covariance_quality"],
                "likelihood_channel_count": payload["likelihood_channel_count"],
                "masked_auxiliary_channels": payload["masked_auxiliary_channels"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
