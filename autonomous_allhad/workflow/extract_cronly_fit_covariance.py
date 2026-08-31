#!/usr/bin/env python3
"""Extract CR-only FitDiagnostics pulls, constraints, and full covariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = ROOT.TFile.Open(str(args.fit_diagnostics))
    if not source or source.IsZombie():
        raise SystemExit(f"unreadable FitDiagnostics file: {args.fit_diagnostics}")
    fit = source.Get("fit_b")
    if fit is None:
        raise SystemExit("fit_b is absent")
    if int(fit.status()) != 0 or int(fit.covQual()) < 2:
        raise SystemExit(
            f"invalid fit: status={fit.status()} covQual={fit.covQual()}"
        )

    minimizer_initial = fit.floatParsInit()
    prefit = source.Get("nuisances_prefit")
    if prefit is None:
        raise SystemExit("nuisances_prefit is absent")
    final = fit.floatParsFinal()
    names = [str(final.at(index).GetName()) for index in range(final.getSize())]
    root_covariance = fit.covarianceMatrix()
    covariance = np.asarray(
        [
            [float(root_covariance[row][column]) for column in range(len(names))]
            for row in range(len(names))
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(covariance)):
        raise SystemExit("fit covariance contains nonfinite entries")
    symmetrized = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetrized)
    parameters = {}
    for index, name in enumerate(names):
        fitted = final.at(index)
        before = prefit.find(name)
        fit_start = minimizer_initial.find(name)
        is_auto_mc_stat = name.startswith("prop_bin")
        has_prefit_record = bool(before)
        prior_constrained = has_prefit_record and not is_auto_mc_stat
        prefit_value = (
            1.0
            if is_auto_mc_stat
            else 0.0
            if has_prefit_record
            else None
        )
        # Combine stores lnN/shape nuisance coordinates as standard-normal
        # theta parameters.  In FitDiagnostics, nuisances_prefit can carry a
        # zero RooRealVar error even though the prior width is one.  Do not use
        # floatParsInit here: robustFit updates those values before the final
        # minimization, so they are minimizer seeds rather than physics prefit
        # values.
        prefit_error = 1.0 if prior_constrained else None
        parameters[name] = {
            "prior_constrained": prior_constrained,
            "constraint_model": (
                "auto_mc_stat"
                if is_auto_mc_stat
                else "standard_normal"
                if prior_constrained
                else "unconstrained_rate_parameter"
            ),
            "prefit": prefit_value,
            "prefit_error": prefit_error,
            "minimizer_initial": (
                float(fit_start.getVal()) if fit_start is not None else None
            ),
            "minimizer_initial_error": (
                float(fit_start.getError()) if fit_start is not None else None
            ),
            "value": float(fitted.getVal()),
            "error": float(fitted.getError()),
            "pull": (
                (float(fitted.getVal()) - prefit_value) / prefit_error
                if prior_constrained
                else None
            ),
            "constraint": (
                float(fitted.getError()) / prefit_error
                if prior_constrained
                else None
            ),
            "bounded": bool(fitted.hasMin() or fitted.hasMax()),
        }

    output = {
        "status": "complete",
        "fit": "background-only observed CR data",
        "fit_status": int(fit.status()),
        "covariance_quality": int(fit.covQual()),
        "edm": float(fit.edm()),
        "parameter_order": names,
        "parameters": parameters,
        "covariance": covariance.tolist(),
        "covariance_validation": {
            "dimension": len(names),
            "finite": True,
            "max_asymmetry": float(np.max(np.abs(covariance - covariance.T))) if names else 0.0,
            "minimum_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else None,
            "maximum_eigenvalue": float(eigenvalues[-1]) if len(eigenvalues) else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "parameters": len(names),
                "covariance_quality": output["covariance_quality"],
                "minimum_eigenvalue": output["covariance_validation"]["minimum_eigenvalue"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
