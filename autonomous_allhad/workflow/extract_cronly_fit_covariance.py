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

    initial = fit.floatParsInit()
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
        before = initial.find(name)
        initial_value = float(before.getVal()) if before is not None else None
        initial_error = float(before.getError()) if before is not None else None
        constrained = bool(initial_error is not None and initial_error > 0.0)
        parameters[name] = {
            "initial": initial_value,
            "initial_error": initial_error,
            "value": float(fitted.getVal()),
            "error": float(fitted.getError()),
            "pull": (
                (float(fitted.getVal()) - initial_value) / initial_error
                if constrained
                else None
            ),
            "constraint": (
                float(fitted.getError()) / initial_error if constrained else None
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
