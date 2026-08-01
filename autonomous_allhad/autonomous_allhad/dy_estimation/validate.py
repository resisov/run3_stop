"""Regression and accounting checks for a complete DY measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import CHANNELS, GROUPS, finalize_rz, merge_tree


FIELDS = ("RZ", "RT", "RZ_stat", "RT_stat", "correlation")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def compare_fit(
    actual: dict[str, Any], expected: dict[str, Any], tolerance: float, regime: str
) -> None:
    for channel in CHANNELS:
        for group in GROUPS:
            observed = actual["channels"][channel][group]
            reference = expected["channels"][channel][group]
            if observed.get("status") != "complete":
                raise RuntimeError(f"{regime}/{channel}/{group}: fit incomplete")
            for field in FIELDS:
                close(
                    float(observed[field]),
                    float(reference[field]),
                    tolerance,
                    f"{regime}/{channel}/{group}/{field}",
                )
    for group in GROUPS:
        for field in ("RZ", "RZ_stat"):
            close(
                float(actual["combined"][group][field]),
                float(expected["combined"][group][field]),
                tolerance,
                f"{regime}/combined/{group}/{field}",
            )


def validate_feature(
    path: Path, payload: dict[str, Any], channel: str, reference: dict[str, Any]
) -> None:
    if payload.get("status") != "feature_stage_complete":
        raise RuntimeError(f"{path}: feature stage is not complete")
    summary = payload.get("summary") or {}
    if summary.get("missing_roots"):
        raise RuntimeError(f"{path}: missing ROOT inputs")
    completed = int(summary.get("completed_roots", -1))
    if completed != int(summary.get("input_roots", -2)):
        raise RuntimeError(f"{path}: ROOT accounting does not close")
    if completed != int(reference["completed_roots"]):
        raise RuntimeError(f"{path}: completed ROOT count changed")
    if int(summary.get("candidate_events", -1)) != int(
        reference["candidate_events"]
    ):
        raise RuntimeError(f"{path}: sparse candidate count changed")
    channels = (payload.get("provenance") or {}).get("channels") or []
    if channels != [channel]:
        raise RuntimeError(f"{path}: expected only {channel}, got {channels}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee", type=Path, required=True)
    parser.add_argument("--mumu", type=Path, required=True)
    parser.add_argument("--low-exact", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("reference_2024.json"),
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reference = read_json(args.reference)
    ee = read_json(args.ee)
    mumu = read_json(args.mumu)
    exact = read_json(args.low_exact)
    validate_feature(args.ee, ee, "DY2E", reference["inputs"]["DY2E"])
    validate_feature(args.mumu, mumu, "DY2M", reference["inputs"]["DY2M"])

    high_raw: dict[str, Any] = {}
    for payload, channel in ((ee, "DY2E"), (mumu, "DY2M")):
        merge_tree(high_raw, {channel: payload["rz_high_raw"][channel]})
    high = finalize_rz(high_raw)
    compare_fit(high, reference["highdm"], args.tolerance, "highdm")

    if exact.get("status") != "complete":
        raise RuntimeError(f"{args.low_exact}: exact recovery is incomplete")
    exact_summary = exact.get("summary") or {}
    expected_exact = reference["inputs"]["lowdm_exact"]
    for field in (
        "candidate_events",
        "candidate_files",
        "completed_partitions",
        "matched_events",
        "selected_events",
    ):
        if int(exact_summary.get(field, -1)) != int(expected_exact[field]):
            raise RuntimeError(f"lowdm exact/{field}: accounting changed")
    if exact_summary.get("failures"):
        raise RuntimeError("lowdm exact: non-empty failure list")
    candidate_total = sum(
        int(item["candidate_events"])
        for item in (reference["inputs"]["DY2E"], reference["inputs"]["DY2M"])
    )
    if candidate_total != int(exact_summary["candidate_events"]):
        raise RuntimeError("feature/exact sparse-candidate accounting differs")
    low = finalize_rz(exact["rz_low_raw"])
    compare_fit(low, reference["lowdm"], args.tolerance, "lowdm")

    hashes = {
        "DY2E": sha256(args.ee),
        "DY2M": sha256(args.mumu),
        "lowdm_exact": sha256(args.low_exact),
    }
    if args.verify_hashes:
        for name, value in hashes.items():
            expected = reference["inputs"][name]["sha256"]
            if value != expected:
                raise RuntimeError(f"{name}: SHA-256 changed")
    result = {
        "schema_version": "dy_estimation_validation_2024_v1",
        "status": "complete",
        "checks": {
            "feature_accounting": "closed",
            "lowdm_exact_accounting": "closed",
            "highdm_fit_regression": "identical",
            "lowdm_fit_regression": "identical",
            "hashes_verified": bool(args.verify_hashes),
        },
        "inputs_sha256": hashes,
        "highdm": high,
        "lowdm": low,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "checks": result["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
