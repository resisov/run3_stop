#!/usr/bin/env python3
"""Validate CR-only fit provenance and the nine High-dM VR MET products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REGIONS = ("HighDMVR_Nb1", "HighDMVR_Nb2", "HighDMVR_Nb3plus")
EDGES = np.asarray([250, 300, 350, 400, 500, 650, 800, 1000, 1500], dtype=float)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_record(label: str, record: dict, nbin: int) -> dict:
    total = np.asarray(record["total"], dtype=float)
    uncertainty = np.asarray(record["uncertainty"], dtype=float)
    covariance = np.asarray(record["covariance"], dtype=float)
    data = np.asarray(record["data"], dtype=float)
    arrays = [total, uncertainty, covariance, data]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise SystemExit(f"{label}: nonfinite prediction content")
    if len(total) != nbin or len(uncertainty) != nbin or len(data) != nbin:
        raise SystemExit(f"{label}: wrong bin count")
    if covariance.shape != (nbin, nbin):
        raise SystemExit(f"{label}: wrong covariance shape {covariance.shape}")
    if np.any(total < 0.0) or np.any(uncertainty < 0.0) or np.any(data < 0.0):
        raise SystemExit(f"{label}: negative yield or uncertainty")
    process_sum = sum(
        (np.asarray(values, dtype=float) for values in record["processes"].values()),
        np.zeros(nbin),
    )
    if not np.allclose(process_sum, total, rtol=1.0e-10, atol=1.0e-10):
        raise SystemExit(f"{label}: process stack does not close")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-9):
        raise SystemExit(f"{label}: covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    tolerance = max(float(np.max(np.diag(covariance))), 1.0) * 1.0e-9
    if float(eigenvalues[0]) < -tolerance:
        raise SystemExit(f"{label}: covariance is not positive semidefinite")
    if not np.allclose(
        uncertainty,
        np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise SystemExit(f"{label}: uncertainty does not match covariance diagonal")
    return {
        "yield": float(np.sum(total)),
        "data": float(np.sum(data)),
        "minimum_covariance_eigenvalue": float(eigenvalues[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-status", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--page-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fit = load(args.fit_status)
    if fit.get("status") != "complete" or int(fit.get("fit_status", -1)) != 0:
        raise SystemExit("CR-only fit did not complete successfully")
    if int(fit.get("covariance_quality", -1)) < 2:
        raise SystemExit("CR-only fit covariance quality is below 2")
    if int(fit.get("likelihood_channel_count", -1)) != 310:
        raise SystemExit(
            "exact-Nb CR-only likelihood does not contain exactly 310 channels"
        )
    if fit.get("vr_observation_in_likelihood") is not False:
        raise SystemExit("VR observation entered the fit")
    if fit.get("sr_observation_in_likelihood") is not False:
        raise SystemExit("SR observation entered the fit")
    masks = fit.get("mask_validation") or {}
    if sorted(masks) != ["y2024_SR_highdm_bin0", "y2025_SR_highdm_bin0"]:
        raise SystemExit("unexpected auxiliary SR mask set")
    for name, record in masks.items():
        if (
            not np.isclose(float(record["runtime_value"]), 1.0)
            or not record["runtime_frozen"]
        ):
            raise SystemExit(f"{name} was not hard-masked")

    summary = load(args.summary)
    if summary.get("status") != "complete" or summary.get("template_observable") != "met":
        raise SystemExit("VR postfit summary is incomplete or is not the MET template")
    if summary.get("vr_observation_in_likelihood") is not False:
        raise SystemExit("VR observation entered the postfit likelihood")
    if summary.get("sr_observation_in_likelihood") is not False:
        raise SystemExit("SR observation entered the postfit likelihood")
    if summary.get("highdm_control_grouping") != "exact":
        raise SystemExit("VR prediction did not use exact Nb CR-to-VR transfer factors")
    if "denominator sumw2" not in summary.get("transfer_factor_mc_statistics", ""):
        raise SystemExit("CR-to-VR transfer-factor MC-statistics audit is absent")
    edges = np.asarray(summary["edges"], dtype=float)
    if not np.array_equal(edges, EDGES):
        raise SystemExit(f"unexpected MET template edges: {edges.tolist()}")

    checked = {}
    for scope in ("2024", "2025", "combined"):
        records = summary["combined"] if scope == "combined" else summary["years"][scope]
        if tuple(records) != REGIONS:
            raise SystemExit(f"{scope}: unexpected VR region set {list(records)}")
        checked[scope] = {
            region: validate_record(f"{scope}/{region}", records[region], len(edges) - 1)
            for region in REGIONS
        }

    page = load(args.page_summary)
    if page.get("status") != "complete" or int(page.get("plot_count", -1)) != 9:
        raise SystemExit("plot summary does not contain exactly nine plots")
    if page.get("non_template_variables_excluded") != ["njet", "nb", "ntop", "nw"]:
        raise SystemExit("non-template variable exclusion audit is absent")
    pairs = []
    for plot in page["plots"]:
        png = Path(plot["png"])
        pdf = Path(plot["pdf"])
        if not png.is_file() or not pdf.is_file() or png.stat().st_size == 0 or pdf.stat().st_size == 0:
            raise SystemExit(f"missing/nonempty PNG/PDF pair for {plot.get('name')}")
        if "_met_postfit" not in png.stem or "_met_postfit" not in pdf.stem:
            raise SystemExit(f"non-template plot survived: {plot.get('name')}")
        pairs.append({"png": str(png), "pdf": str(pdf)})

    result = {
        "status": "complete",
        "fit_status": int(fit["fit_status"]),
        "covariance_quality": int(fit["covariance_quality"]),
        "edm": float(fit["edm"]),
        "likelihood_channel_count": int(fit["likelihood_channel_count"]),
        "fit_parameter_count": int(summary["fit_parameter_count"]),
        "highdm_control_grouping": summary["highdm_control_grouping"],
        "cr_to_vr_transfer_factor_definition": summary[
            "cr_to_vr_transfer_factor_definition"
        ],
        "transfer_factor_mc_statistics": summary[
            "transfer_factor_mc_statistics"
        ],
        "template_observable": "met",
        "plot_count": len(pairs),
        "png_pdf_pairs": pairs,
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "checked_predictions": checked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "plot_count", "covariance_quality")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
