#!/usr/bin/env python3
"""Plot all constrained nuisances from a CR-only background-only fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-parameters", required=True, type=Path)
    parser.add_argument("--fit-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    parameters_payload = json.loads(args.fit_parameters.read_text())
    status_payload = json.loads(args.fit_status.read_text())
    if parameters_payload.get("status") != "complete":
        raise SystemExit("fit-parameter extraction is not complete")
    if status_payload.get("status") != "complete":
        raise SystemExit("CR-only fit is not complete")
    if int(status_payload.get("fit_status", -1)) != 0:
        raise SystemExit("CR-only fit status is nonzero")
    if int(status_payload.get("covariance_quality", -1)) < 3:
        raise SystemExit("CR-only fit covariance quality is below 3")
    if status_payload.get("vr_observation_in_likelihood") is not False:
        raise SystemExit("VR observation entered the likelihood")
    if status_payload.get("sr_observation_in_likelihood") is not False:
        raise SystemExit("SR observation entered the likelihood")

    raw_parameters = parameters_payload.get("parameters") or {}
    constrained = []
    rate_parameters = []
    for name, record in raw_parameters.items():
        if bool(record.get("bounded")):
            rate_parameters.append(name)
            continue
        value = float(record["value"])
        error = float(record["error"])
        if not math.isfinite(value) or not math.isfinite(error) or error <= 0.0:
            raise SystemExit(f"invalid nuisance result for {name}")
        constrained.append({"name": name, "value": value, "error": error})

    if not constrained:
        raise SystemExit("no constrained nuisances found")
    constrained.sort(key=lambda item: (-abs(item["value"]), item["name"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    columns = 3
    rows = math.ceil(len(constrained) / columns)
    fig, axes = plt.subplots(1, columns, figsize=(24, 17.5), sharex=True)
    for column, axis in enumerate(axes):
        start = column * rows
        records = constrained[start : start + rows]
        y = np.arange(len(records), dtype=float)
        axis.axvspan(-2.0, 2.0, color="#fde68a", alpha=0.45, zorder=0)
        axis.axvspan(-1.0, 1.0, color="#bbf7d0", alpha=0.75, zorder=1)
        axis.axvline(0.0, color="black", linewidth=1.0, zorder=2)
        axis.errorbar(
            [record["value"] for record in records],
            y,
            xerr=[record["error"] for record in records],
            fmt="o",
            color="#dc2626",
            ecolor="#1f2937",
            elinewidth=1.15,
            capsize=2.2,
            markersize=4.8,
            zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels([record["name"] for record in records], fontsize=7.0)
        axis.invert_yaxis()
        axis.set_xlim(-2.25, 2.25)
        axis.set_xticks([-2, -1, 0, 1, 2])
        axis.grid(axis="y", color="#d1d5db", linewidth=0.45, alpha=0.65)
        axis.tick_params(axis="x", labelsize=13)
        axis.set_xlabel(r"Postfit $\hat{\theta}\;\pm\;\sigma_{\mathrm{postfit}}$", fontsize=15)
        axis.set_title(f"Nuisances {start + 1}–{start + len(records)}", fontsize=15)

    fig.suptitle("2024+2025 CR-only background-only fit: nuisance pulls and constraints", fontsize=24, y=0.985)
    fig.text(0.03, 0.96, "CMS", fontsize=22, fontweight="bold", ha="left")
    fig.text(0.082, 0.96, "Work in progress", fontsize=16, style="italic", ha="left")
    fig.text(
        0.97,
        0.96,
        (
            rf"220.66 fb$^{{-1}}$ (13.6 TeV)  ·  fit status {status_payload['fit_status']}  "
            f"·  covQual {status_payload['covariance_quality']}  "
            f"·  {status_payload['likelihood_channel_count']} CR channels"
        ),
        fontsize=14,
        ha="right",
    )
    fig.text(
        0.5,
        0.018,
        "Green/yellow bands show ±1σ/±2σ prefit ranges. Bounded CR normalization and Sγ rate parameters are excluded.",
        ha="center",
        fontsize=13,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.925, bottom=0.065, wspace=0.61)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

    digest = hashlib.sha256()
    with args.fit_parameters.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "status": "complete",
        "fit": parameters_payload.get("fit"),
        "fit_status": int(status_payload["fit_status"]),
        "covariance_quality": int(status_payload["covariance_quality"]),
        "likelihood_channel_count": int(status_payload["likelihood_channel_count"]),
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "constrained_nuisance_count": len(constrained),
        "excluded_bounded_rate_parameter_count": len(rate_parameters),
        "fit_parameters_sha256": digest.hexdigest(),
        "ordering": "descending absolute postfit shift, then name",
        "pull_definition": "Gaussian nuisance postfit value in nominal prefit-sigma units",
        "points": constrained,
        "excluded_bounded_rate_parameters": sorted(rate_parameters),
        "png": str(args.output.with_suffix(".png").relative_to(args.summary.parent)),
        "pdf": str(args.output.with_suffix(".pdf").relative_to(args.summary.parent)),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "constrained_nuisances": len(constrained),
                "excluded_rate_parameters": len(rate_parameters),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
