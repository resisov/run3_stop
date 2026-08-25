#!/usr/bin/env python3
"""Build the CR-only web-page audit from validated machine summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_complete(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise SystemExit(f"summary is not complete: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fit = read_complete(args.page_dir / "fit_status.json")
    cr = read_complete(args.page_dir / "cr_met_postfit_summary.json")
    pulls = read_complete(args.page_dir / "pull_summary.json")
    vr = read_complete(args.page_dir / "validation_summary.json")
    parameters = read_complete(args.page_dir / "fit_parameters.json")

    expected = {
        "fit_status": 0,
        "covariance_quality": 3,
        "likelihood_channel_count": 156,
    }
    for label, payload in (("fit", fit), ("cr", cr), ("pulls", pulls), ("vr", vr)):
        for key, value in expected.items():
            if int(payload.get(key, -1)) != value:
                raise SystemExit(f"{label}: unexpected {key}={payload.get(key)}")
        if payload.get("vr_observation_in_likelihood") is not False:
            raise SystemExit(f"{label}: VR observation entered the likelihood")
        if payload.get("sr_observation_in_likelihood") is not False:
            raise SystemExit(f"{label}: SR observation entered the likelihood")

    pngs = sorted((args.page_dir / "plots").rglob("*.png"))
    pdfs = sorted((args.page_dir / "plots").rglob("*.pdf"))
    png_stems = {path.relative_to(args.page_dir).with_suffix("") for path in pngs}
    pdf_stems = {path.relative_to(args.page_dir).with_suffix("") for path in pdfs}
    if png_stems != pdf_stems:
        raise SystemExit("PNG/PDF plot pairs do not match")

    cr_count = int(cr["plot_count"])
    if int(cr.get("highdm_cr_channel_count", -1)) != 72:
        raise SystemExit("unexpected High-dM CR channel count")
    if int(cr.get("lowdm_cr_channel_count", -1)) != 84:
        raise SystemExit("unexpected Low-dM CR channel count")
    if int(cr.get("lowdm_control_group_count_per_year_region", -1)) != 14:
        raise SystemExit("unexpected Low-dM control-group count")
    vr_count = int(vr["plot_count"])
    pull_count = 1
    expected_pairs = cr_count + vr_count + pull_count
    if len(png_stems) != expected_pairs:
        raise SystemExit(
            f"expected {expected_pairs} plot pairs from summaries, found {len(png_stems)}"
        )

    output = {
        "status": "complete",
        "fit": parameters.get("fit"),
        "fit_status": int(fit["fit_status"]),
        "covariance_quality": int(fit["covariance_quality"]),
        "likelihood_channel_count": int(fit["likelihood_channel_count"]),
        "fit_parameter_count": len(parameters.get("parameters") or {}),
        "constrained_nuisance_count": int(pulls["constrained_nuisance_count"]),
        "bounded_rate_parameter_count": int(
            pulls["excluded_bounded_rate_parameter_count"]
        ),
        "highdm_cr_channel_count": int(cr["highdm_cr_channel_count"]),
        "lowdm_cr_channel_count": int(cr["lowdm_cr_channel_count"]),
        "cr_met_postfit_plot_count": cr_count,
        "highdm_cr_postfit_plot_count": 18,
        "lowdm_cr_postfit_plot_count": 36,
        "vr_met_postfit_plot_count": vr_count,
        "nuisance_pull_plot_count": pull_count,
        "total_png_pdf_plot_pairs": len(png_stems),
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "fit_diagnostics_sha256": cr["fit_diagnostics_sha256"],
        "fit_parameters_sha256": pulls["fit_parameters_sha256"],
        "machine_summaries": {
            "fit_status": "fit_status.json",
            "cr_met_postfit": "cr_met_postfit_summary.json",
            "nuisance_pulls": "pull_summary.json",
            "vr_validation": "validation_summary.json",
            "vr_met_postfit": "vr_met_postfit_summary.json",
            "full_parameters_and_covariance": "fit_parameters.json",
        },
        "plot_pairs": [str(path) for path in sorted(png_stems, key=str)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "plot_pairs": len(png_stems),
                "fit_parameters": output["fit_parameter_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
