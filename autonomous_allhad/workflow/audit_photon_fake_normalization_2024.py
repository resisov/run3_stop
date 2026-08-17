#!/usr/bin/env python3
"""Verify every MC normalization factor used by the photon-fake measurement."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_STATUS = "normalized_with_xsec_lumi_physical_dataset_sumw"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    measurement = read_json(args.measurement)
    normalization = read_json(args.normalization)
    luminosity_pb = float(normalization["luminosity_pb"])
    physical = normalization.get("physical_datasets") or {}
    errors: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    blocked_by_component: dict[str, int] = {}

    for component, audit in (measurement.get("component_audits") or {}).items():
        blocked = list((audit or {}).get("blocked_datasets") or [])
        blocked_by_component[component] = len(blocked)
        for item in blocked:
            errors.append(
                {
                    "type": "blocked_dataset",
                    "component": component,
                    "record": item,
                }
            )
        for used in (audit or {}).get("used_datasets") or []:
            name = str(used["physical_dataset"])
            norm = physical.get(name)
            if norm is None:
                errors.append(
                    {
                        "type": "missing_normalization_record",
                        "component": component,
                        "physical_dataset": name,
                    }
                )
                continue
            factor = float(used["normalization_factor"])
            xsec = norm.get("xsec_pb")
            sumw = norm.get("sumw")
            status = str(norm.get("normalization_status") or "")
            recomputed = (
                luminosity_pb * float(xsec) / float(sumw)
                if xsec is not None and sumw is not None and float(sumw) != 0.0
                else None
            )
            checks = {
                "status": status == EXPECTED_STATUS,
                "finite_positive_factor": math.isfinite(factor) and factor > 0.0,
                "measurement_matches_normalization": math.isclose(
                    factor,
                    float(norm.get("normalization_factor")),
                    rel_tol=1.0e-13,
                    abs_tol=0.0,
                ),
                "formula_matches": (
                    recomputed is not None
                    and math.isclose(
                        factor,
                        recomputed,
                        rel_tol=1.0e-13,
                        abs_tol=0.0,
                    )
                ),
                "process_matches": str(used["process"]) == str(norm.get("process")),
                "no_xsec_conflict": not list(norm.get("xsec_conflicts") or []),
                "retained_files_nonzero": int(norm.get("files_processed") or 0) > 0,
            }
            if not all(checks.values()):
                errors.append(
                    {
                        "type": "factor_check_failed",
                        "component": component,
                        "physical_dataset": name,
                        "checks": checks,
                    }
                )
            record = records.setdefault(
                name,
                {
                    "physical_dataset": name,
                    "process": norm.get("process"),
                    "xsec_pb": xsec,
                    "sumw": sumw,
                    "luminosity_pb": luminosity_pb,
                    "normalization_factor": factor,
                    "recomputed_factor": recomputed,
                    "normalization_status": status,
                    "files_attempted": norm.get("files_attempted"),
                    "files_processed": norm.get("files_processed"),
                    "sumw_source_counts": norm.get("sumw_source_counts"),
                    "components": [],
                    "checks": checks,
                },
            )
            record["components"].append(component)
            if not math.isclose(
                float(record["normalization_factor"]),
                factor,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                errors.append(
                    {
                        "type": "component_factor_disagreement",
                        "component": component,
                        "physical_dataset": name,
                    }
                )

    expected_formula = (
        (normalization.get("normalization_policy") or {}).get(
            "background_formula"
        )
    )
    if expected_formula != (
        "gen_weight * post_skim_sf_weight * xsec_pb * lumi_pb / "
        "physical_dataset_sumw"
    ):
        errors.append(
            {
                "type": "unexpected_normalization_policy",
                "background_formula": expected_formula,
            }
        )
    if normalization.get("status") != "complete":
        errors.append(
            {
                "type": "normalization_not_complete",
                "status": normalization.get("status"),
            }
        )
    status = "pass" if not errors else "fail"
    payload = {
        "schema_version": "photon_fake_2024_normalization_audit_v1",
        "status": status,
        "measurement": str(args.measurement),
        "normalization": str(args.normalization),
        "normalization_status": normalization.get("status"),
        "luminosity_pb": luminosity_pb,
        "background_formula": expected_formula,
        "factor_application": (
            "photon_fake_2024_worker stores gen_weight × nominal post-skim "
            "scale factors; measure_photon_fakes_2024 multiplies each physical "
            "dataset once by xsec × luminosity / retained physical-dataset sumw"
        ),
        "unique_datasets_checked": len(records),
        "blocked_by_component": blocked_by_component,
        "errors": errors,
        "datasets": sorted(records.values(), key=lambda item: item["physical_dataset"]),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "unique_datasets_checked": len(records),
                "blocked_by_component": blocked_by_component,
                "errors": len(errors),
                "luminosity_pb": luminosity_pb,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
