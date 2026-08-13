#!/usr/bin/env python3
"""Create a derived nominal payload with the v2 photon-fake estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_photon_fake_datamc_2024_v2 import origin_fraction


MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v2"
DERIVED_SCHEMA = "flat_boosted_recoil_histograms_with_photon_fake_v2"
BACKGROUND_PROCESSES = (
    "DY",
    "GJ",
    "QCD",
    "ST",
    "TT",
    "VV",
    "WtoLNu",
    "Zto2Nu",
)
RECOIL_REGIONS = ("GCR", "GCR_Nt0", "GCR_Nt1")
ORIGIN_UP = "photonFakeOriginFractionUp"
ORIGIN_DOWN = "photonFakeOriginFractionDown"
REQUIRED_FAKE_VARIATIONS = {
    "nominal",
    "photonFakeStatUp",
    "photonFakeStatDown",
    "photonFakePromptUp",
    "photonFakePromptDown",
    "photonFakeElectronUp",
    "photonFakeElectronDown",
    "photonFakePLJShapeUp",
    "photonFakePLJShapeDown",
    "photonFakeClosureUp",
    "photonFakeClosureDown",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("w") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_leaf(
    leaf: dict[str, Any],
    scale: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(leaf["sumw"], dtype=float)
    variances = np.asarray(leaf["sumw2"], dtype=float)
    if len(values) != len(scale):
        raise RuntimeError("origin-fraction bin count differs from nominal")
    return {
        "sumw": (values * scale).tolist(),
        "sumw2": (variances * np.square(scale)).tolist(),
        "entries": [int(value) for value in leaf.get("entries") or []],
    }


def converted_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
    return {
        "sumw": [float(value) for value in leaf["sumw"]],
        "sumw2": [float(value) for value in leaf["sumw2"]],
        "entries": [int(value) for value in leaf.get("entries") or []],
    }


def add_leaves(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    first_values = np.asarray(first["sumw"], dtype=float)
    second_values = np.asarray(second["sumw"], dtype=float)
    first_variances = np.asarray(first["sumw2"], dtype=float)
    second_variances = np.asarray(second["sumw2"], dtype=float)
    if len(first_values) != len(second_values):
        raise RuntimeError("cannot add histogram leaves with different bin counts")
    first_entries = np.asarray(
        first.get("entries") or np.zeros(len(first_values)),
        dtype=int,
    )
    second_entries = np.asarray(
        second.get("entries") or np.zeros(len(second_values)),
        dtype=int,
    )
    return {
        "sumw": (first_values + second_values).tolist(),
        "sumw2": (first_variances + second_variances).tolist(),
        "entries": (first_entries + second_entries).astype(int).tolist(),
    }


def fake_variations(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    fake = measurement["fake_prediction"]
    if variable == "recoil":
        return fake["histograms"][region]
    return fake["highdm_variable_histograms"][region][variable]


def origin_histograms(
    measurement: dict[str, Any],
    process: str,
    region: str,
    variable: str,
) -> dict[str, Any]:
    return measurement["mc_target_origin_histograms"][process][region][variable]


def inject_variable(
    samples: dict[str, Any],
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    measured_fake = fake_variations(measurement, region, variable)
    missing = sorted(REQUIRED_FAKE_VARIATIONS - set(measured_fake))
    if missing:
        raise RuntimeError(
            f"{region}/{variable}: missing fake variations {missing}"
        )
    converted_fake = {
        variation: converted_leaf(leaf)
        for variation, leaf in measured_fake.items()
    }
    process_audits: dict[str, Any] = {}
    for process in BACKGROUND_PROCESSES:
        if process not in samples:
            raise RuntimeError(f"{region}/{variable}: missing process {process}")
        fraction, fraction_variance, audit = origin_fraction(
            origin_histograms(
                measurement,
                process,
                region,
                variable,
            )
        )
        uncertainty = np.sqrt(np.maximum(fraction_variance, 0.0))
        fraction_up = np.clip(fraction + uncertainty, 0.0, 1.0)
        fraction_down = np.clip(fraction - uncertainty, 0.0, 1.0)
        original = samples[process]
        retained = {
            variation: scaled_leaf(leaf, fraction)
            for variation, leaf in original.items()
        }
        retained[ORIGIN_UP] = scaled_leaf(original["nominal"], fraction_up)
        retained[ORIGIN_DOWN] = scaled_leaf(original["nominal"], fraction_down)
        if process == "QCD":
            fake_nominal = converted_fake["nominal"]
            retained = {
                variation: add_leaves(leaf, fake_nominal)
                for variation, leaf in retained.items()
            }
            for variation, leaf in converted_fake.items():
                retained[variation] = add_leaves(
                    scaled_leaf(original["nominal"], fraction),
                    leaf,
                )
        samples[process] = retained
        process_audits[process] = {
            "fallback_bins": audit["fallback_bins"],
            "fraction_clipped_bins": audit["fraction_clipped_bins"],
            "nonfake_fraction": fraction.tolist(),
            "nonfake_fraction_uncertainty": uncertainty.tolist(),
        }
    return process_audits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    nominal_path = args.nominal.absolute()
    output_path = args.output.absolute()
    if nominal_path == output_path:
        raise RuntimeError("derived output must differ from nominal input")
    if output_path.exists() and output_path.samefile(nominal_path):
        raise RuntimeError("derived output aliases nominal input")
    measurement = read_json(args.measurement)
    if measurement.get("schema_version") != MEASUREMENT_SCHEMA:
        raise RuntimeError("unexpected photon-fake measurement schema")
    if measurement.get("status") != "complete" and not args.allow_partial:
        raise RuntimeError("refusing to inject an incomplete measurement")

    payload = read_json(nominal_path)
    audits: dict[str, Any] = {}
    for region in RECOIL_REGIONS:
        samples = (payload.get("histograms") or {}).get(region)
        if not isinstance(samples, dict):
            raise RuntimeError(f"nominal payload is missing {region}/recoil")
        audits[f"{region}/recoil"] = inject_variable(
            samples,
            measurement,
            region,
            "recoil",
        )
    highdm = (payload.get("highdm_variable_histograms") or {}).get("GCR")
    if not isinstance(highdm, dict):
        raise RuntimeError("nominal payload has no high-dM GCR distributions")
    expected_variables = sorted(
        (payload.get("highdm_distribution_variable_specs") or {}).keys()
    )
    for variable in expected_variables:
        samples = highdm.get(variable)
        if not isinstance(samples, dict):
            raise RuntimeError(f"nominal payload is missing GCR/{variable}")
        audits[f"GCR/{variable}"] = inject_variable(
            samples,
            measurement,
            "GCR",
            variable,
        )

    payload["schema_version"] = DERIVED_SCHEMA
    payload.setdefault("summary", {})["photon_fake_measurement_v2"] = {
        "measurement_file": str(args.measurement.absolute()),
        "measurement_sha256": sha256(args.measurement.absolute()),
        "measurement_status": measurement.get("status"),
        "nominal_source_file": str(nominal_path),
        "nominal_source_sha256": sha256(nominal_path),
        "nominal_intermediate_modified": False,
        "target_data_source": "trusted nominal real_subset_worker.py data_obs",
        "replacement_policy": (
            "retain the prompt-photon and electron-matched fraction of every "
            "nominal MC process, remove its truth-fake fraction, and add the "
            "data-driven fake prediction exactly once through the QCD carrier"
        ),
        "origin_fraction_variations": [ORIGIN_UP, ORIGIN_DOWN],
        "injected_paths": sorted(audits),
        "origin_fraction_audits": audits,
    }
    write_json(output_path, payload)
    result = {
        "status": (
            "complete"
            if measurement.get("status") == "complete"
            else "partial_diagnostic"
        ),
        "output": str(output_path),
        "output_size": output_path.stat().st_size,
        "output_sha256": sha256(output_path),
        "nominal_source": str(nominal_path),
        "nominal_source_sha256": payload["summary"][
            "photon_fake_measurement_v2"
        ]["nominal_source_sha256"],
        "variables_injected": len(audits),
        "nominal_intermediate_modified": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
