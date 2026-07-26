#!/usr/bin/env python3
"""Build concise labels for the full 2024 High-dM + Low-dM impact plot."""

import argparse
import json
import re
from pathlib import Path


CHANNEL_LABELS = {
    "cat2_LLCR_highDeltaM": "High-dM lost-lepton CR",
    "cat3_QCDCR_highDeltaM": "High-dM QCD CR",
    "cat4_GCR_highDeltaM": "High-dM photon CR",
    "cat5_DY2E_highDeltaM": "High-dM ee CR",
    "cat6_DY2M_highDeltaM": "High-dM mumu CR",
    "cat7_SR_selected_recoil60_nb2_nt2plus_w0": "High-dM SR",
    "cat2_LLCR_lowDeltaM": "Low-dM lost-lepton CR",
    "cat3_QCDCR_lowDeltaM": "Low-dM QCD CR",
    "cat4_GCR_lowDeltaM": "Low-dM photon CR",
    "cat5_DY2E_lowDeltaM": "Low-dM ee CR",
    "cat6_DY2M_lowDeltaM": "Low-dM mumu CR",
    "cat7_SR_lowDeltaM": "Low-dM SR",
}

SYSTEMATIC_LABELS = {
    "Lumi_2024": "2024 luminosity",
    "btagSF_bc_correlated": "b tag heavy flavor, correlated",
    "btagSF_bc_uncorrelated": "b tag heavy flavor, 2024",
    "btagSF_light_correlated": "b tag light flavor, correlated",
    "btagSF_light_uncorrelated": "b tag light flavor, 2024",
    "electron_hlt": "Electron trigger",
    "electron_id": "Electron identification",
    "muon_hlt": "Muon trigger",
    "muon_id": "Muon identification",
    "photon_id": "Photon identification",
    "pileup": "Pileup",
}


def label(name):
    if name in SYSTEMATIC_LABELS:
        return SYSTEMATIC_LABELS[name]
    match = re.fullmatch(r"prop_bin(.+)_bin(\d+)(?:_(.+))?", name)
    if not match:
        return name
    channel, zero_based_bin, process = match.groups()
    channel_label = CHANNEL_LABELS.get(channel, channel)
    suffix = ""
    if process == "background":
        suffix = ", background"
    elif process == "sig_mStop1200_mLSP500":
        suffix = ", signal (1200, 500)"
    elif process:
        suffix = ", " + process
    return f"MC stat: {channel_label} bin {int(zero_based_bin) + 1}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    impacts = json.loads(args.impacts.read_text())
    translations = {
        parameter["name"]: label(parameter["name"])
        for parameter in impacts["params"]
    }
    args.output.write_text(
        json.dumps(translations, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "labels": len(translations),
                "unchanged": sum(
                    name == translated
                    for name, translated in translations.items()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
