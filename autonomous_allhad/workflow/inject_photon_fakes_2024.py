#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_MEASUREMENT_SCHEMA = "photon_fake_2024_measurement_v1"
EXPECTED_GCR_AUDIT_SCHEMA = "photon_fake_gcr_event_key_audit_v1"
DERIVED_SCHEMA = "flat_boosted_recoil_histograms_with_data_driven_photon_fake_v1"
FAKE_PROCESS = "QCD"
DATA_PROCESS = "data_obs"
RECOIL_REGIONS = ("GCR", "GCR_Nt0", "GCR_Nt1")
MAX_NOMINAL_TARGET_SUBSET_LOSS_FRACTION = 0.005


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
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_leaf(measured: dict[str, Any]) -> dict[str, Any]:
    return {
        "sumw": [float(value) for value in measured["sumw"]],
        "sumw2": [float(value) for value in measured["sumw2"]],
        "entries": [int(value) for value in measured["entries"]],
    }


def reference_bin_count(samples: dict[str, Any]) -> int:
    for variations in samples.values():
        for leaf in variations.values():
            if isinstance(leaf, dict) and "sumw" in leaf:
                return len(leaf["sumw"])
    raise RuntimeError("histogram has no reference leaf")


def validated_variations(
    measured: dict[str, Any],
    expected_bins: int,
    context: str,
) -> dict[str, Any]:
    required = {
        "nominal",
        "photonFakeTFUp",
        "photonFakeTFDown",
        "photonFakePromptUp",
        "photonFakePromptDown",
        "photonFakeElectronUp",
        "photonFakeElectronDown",
        "photonFakeClosureUp",
        "photonFakeClosureDown",
    }
    missing = sorted(required - set(measured))
    if missing:
        raise RuntimeError(f"{context}: missing fake variations {missing}")
    output: dict[str, Any] = {}
    for variation, leaf in measured.items():
        converted = payload_leaf(leaf)
        if len(converted["sumw"]) != expected_bins:
            raise RuntimeError(
                f"{context}/{variation}: measured bins={len(converted['sumw'])}, "
                f"nominal bins={expected_bins}"
            )
        output[variation] = converted
    return output


def nominal_target_subset_is_safe(audit: dict[str, Any]) -> tuple[bool, float]:
    comparison = audit.get("comparison") or {}
    nominal_only = int(comparison.get("nominal_only_event_keys") or 0)
    fake_only = int(comparison.get("fake_only_event_keys") or 0)
    differing_ut = int(comparison.get("differing_ut_event_keys") or 0)
    nominal_events = int((audit.get("nominal") or {}).get("unique_event_keys") or 0)
    fraction = nominal_only / nominal_events if nominal_events > 0 else float("inf")
    safe = (
        fake_only == 0
        and differing_ut == 0
        and fraction <= MAX_NOMINAL_TARGET_SUBSET_LOSS_FRACTION
    )
    return safe, fraction


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived nominal histogram payload in which high-dM GCR "
            "QCD MC is replaced by the measured data-driven photon fake."
        )
    )
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--gcr-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--target-data-source",
        choices=("nominal", "measurement"),
        default="nominal",
        help=(
            "Use the trusted nominal real_subset_worker.py GCR data by default. "
            "The measurement target may be used only after exact event-key equality."
        ),
    )
    args = parser.parse_args()

    nominal = args.nominal.absolute()
    output = args.output.absolute()
    if nominal == output:
        raise RuntimeError("derived output must differ from the nominal input")
    if output.exists() and output.samefile(nominal):
        raise RuntimeError("derived output aliases the nominal input")

    measurement = read_json(args.measurement)
    if measurement.get("schema_version") != EXPECTED_MEASUREMENT_SCHEMA:
        raise RuntimeError(
            f"unexpected measurement schema: {measurement.get('schema_version')}"
        )
    if measurement.get("status") != "complete" and not args.allow_partial:
        raise RuntimeError(
            f"measurement status is {measurement.get('status')}; "
            "use --allow-partial only for a diagnostic derived payload"
        )
    gcr_audit = None
    if args.gcr_audit is not None:
        gcr_audit = read_json(args.gcr_audit.absolute())
        if gcr_audit.get("schema_version") != EXPECTED_GCR_AUDIT_SCHEMA:
            raise RuntimeError(
                f"unexpected GCR audit schema: {gcr_audit.get('schema_version')}"
            )
    target_subset_fraction = None
    if measurement.get("status") == "complete":
        if gcr_audit is None:
            raise RuntimeError(
                "refusing to inject a complete measurement without a GCR "
                "event-key audit"
            )
        if args.target_data_source == "measurement":
            if gcr_audit.get("status") != "equal":
                raise RuntimeError(
                    "measurement target data require an equal nominal/fake "
                    "GCR event-key audit"
                )
        elif gcr_audit.get("status") != "equal":
            safe_subset, target_subset_fraction = nominal_target_subset_is_safe(
                gcr_audit
            )
            if not safe_subset:
                raise RuntimeError(
                    "fake target is not a sufficiently close subset of the "
                    "nominal GCR; nominal target substitution is unsafe"
                )
    payload = read_json(nominal)
    fake = measurement["fake_prediction"]
    replaced: list[str] = []
    corrected_data: list[str] = []

    recoil = payload.get("histograms") or {}
    for region in RECOIL_REGIONS:
        samples = recoil.get(region)
        if not isinstance(samples, dict):
            raise RuntimeError(f"nominal payload is missing recoil region {region}")
        expected_bins = reference_bin_count(samples)
        measured = (fake.get("histograms") or {}).get(region)
        if not isinstance(measured, dict):
            raise RuntimeError(f"measurement is missing recoil region {region}")
        samples[FAKE_PROCESS] = validated_variations(
            measured,
            expected_bins,
            f"histograms/{region}",
        )
        replaced.append(f"histograms/{region}/{FAKE_PROCESS}")
        if args.target_data_source == "measurement":
            target_data = (
                measurement["diagnostic_histograms"][region]["recoil"]["target"][
                    "data"
                ]
            )
            data_leaf = payload_leaf(target_data)
            if len(data_leaf["sumw"]) != expected_bins:
                raise RuntimeError(
                    f"histograms/{region}/data_obs: measured bins="
                    f"{len(data_leaf['sumw'])}, nominal bins={expected_bins}"
                )
            samples[DATA_PROCESS] = {"nominal": data_leaf}
            corrected_data.append(f"histograms/{region}/{DATA_PROCESS}")

    highdm = payload.get("highdm_variable_histograms") or {}
    measured_highdm = (fake.get("highdm_variable_histograms") or {}).get("GCR") or {}
    nominal_gcr = highdm.get("GCR")
    if not isinstance(nominal_gcr, dict):
        raise RuntimeError("nominal payload is missing highdm_variable_histograms/GCR")
    expected_variables = set(
        (payload.get("highdm_distribution_variable_specs") or {}).keys()
    )
    if not expected_variables:
        expected_variables = set(nominal_gcr)
    missing_variables = sorted(expected_variables - set(measured_highdm))
    if missing_variables:
        raise RuntimeError(
            f"measurement is missing high-dM GCR variables: {missing_variables}"
        )
    for variable in sorted(expected_variables):
        samples = nominal_gcr.get(variable)
        if not isinstance(samples, dict):
            raise RuntimeError(f"nominal payload is missing high-dM GCR {variable}")
        expected_bins = reference_bin_count(samples)
        samples[FAKE_PROCESS] = validated_variations(
            measured_highdm[variable],
            expected_bins,
            f"highdm_variable_histograms/GCR/{variable}",
        )
        replaced.append(
            f"highdm_variable_histograms/GCR/{variable}/{FAKE_PROCESS}"
        )
        if args.target_data_source == "measurement":
            target_data = measurement["diagnostic_histograms"]["GCR"][variable][
                "target"
            ]["data"]
            data_leaf = payload_leaf(target_data)
            if len(data_leaf["sumw"]) != expected_bins:
                raise RuntimeError(
                    f"highdm_variable_histograms/GCR/{variable}/data_obs: "
                    f"measured bins={len(data_leaf['sumw'])}, "
                    f"nominal bins={expected_bins}"
                )
            samples[DATA_PROCESS] = {"nominal": data_leaf}
            corrected_data.append(
                f"highdm_variable_histograms/GCR/{variable}/{DATA_PROCESS}"
            )

    payload["schema_version"] = DERIVED_SCHEMA
    summary = payload.setdefault("summary", {})
    summary["photon_fake_measurement"] = {
        "status": measurement.get("status"),
        "measurement_file": str(args.measurement.absolute()),
        "measurement_sha256": sha256(args.measurement),
        "nominal_source_file": str(nominal),
        "nominal_source_sha256": sha256(nominal),
        "central_value": measurement.get("central_value"),
        "method": measurement.get("method"),
        "target_validation": measurement.get("target_validation"),
        "closure": {
            key: value
            for key, value in (measurement.get("closure") or {}).items()
            if key != "distributions"
        },
        "replacement_policy": (
            "replace QCD MC only in high-dM GCR recoil and high-dM GCR "
            "distributions; retain the trusted nominal real_subset_worker.py "
            "GCR data_obs unless target-data-source=measurement is explicitly "
            "selected; retain every other nominal region and process unchanged "
            "to avoid double counting"
        ),
        "target_data_source": args.target_data_source,
        "nominal_target_subset_loss_fraction": target_subset_fraction,
        "nominal_target_subset_loss_threshold": (
            MAX_NOMINAL_TARGET_SUBSET_LOSS_FRACTION
        ),
        "replaced_paths": replaced,
        "deduplicated_data_paths": corrected_data,
        "gcr_event_key_audit": (
            None
            if args.gcr_audit is None
            else {
                "file": str(args.gcr_audit.absolute()),
                "sha256": sha256(args.gcr_audit.absolute()),
                "status": gcr_audit.get("status"),
                "comparison": gcr_audit.get("comparison"),
            }
        ),
        "nominal_intermediate_modified": False,
    }
    write_json(output, payload)
    result = {
        "status": "complete",
        "output": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": sha256(output),
        "nominal": str(nominal),
        "nominal_size": nominal.stat().st_size,
        "nominal_sha256": summary["photon_fake_measurement"][
            "nominal_source_sha256"
        ],
        "measurement": str(args.measurement),
        "measurement_status": measurement.get("status"),
        "replaced_paths": replaced,
        "deduplicated_data_paths": corrected_data,
        "gcr_event_key_audit_status": (
            None if gcr_audit is None else gcr_audit.get("status")
        ),
        "target_data_source": args.target_data_source,
        "nominal_target_subset_loss_fraction": target_subset_fraction,
        "nominal_intermediate_modified": False,
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
