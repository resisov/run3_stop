#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import uproot
from XRootD import client


TREE_NAME = "TROTA"
MARKER_NAME = "TROTA_metadata"
SCHEMA_VERSION = "trota_topresolved_2024_inplace_sparse_v1"
MODEL_SHA256 = "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
TROTA_COMMIT = "38fb282d5c3479d2eec96cc57d60fac7fd412d7f"
WP_NAME = "1pct_qcd_mistag"
WP_THRESHOLD = np.float32(0.9433798789978027)

OUTPUT_DTYPES = {
    "run": np.dtype(np.uint32),
    "luminosityBlock": np.dtype(np.uint32),
    "event": np.dtype(np.uint64),
    "entry": np.dtype(np.int64),
    "dataset_id": np.dtype(np.int64),
    "file_id": np.dtype(np.int64),
    "TopResolved1pct_candidateIndex": np.dtype(np.int32),
    "TopResolved1pct_idxJet0": np.dtype(np.int32),
    "TopResolved1pct_idxJet1": np.dtype(np.int32),
    "TopResolved1pct_idxJet2": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx0": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx1": np.dtype(np.int32),
    "TopResolved1pct_sourceJetIdx2": np.dtype(np.int32),
    "TopResolved1pct_pt": np.dtype(np.float32),
    "TopResolved1pct_eta": np.dtype(np.float32),
    "TopResolved1pct_phi": np.dtype(np.float32),
    "TopResolved1pct_mass": np.dtype(np.float32),
    "TopResolved1pct_FTScore": np.dtype(np.float32),
    "TopResolved1pct_TTScore": np.dtype(np.float32),
    "TopResolved1pct_QCDScore": np.dtype(np.float32),
    "TopResolved1pct_QCDDiscriminant": np.dtype(np.float32),
}
FLOAT_BRANCHES = tuple(
    name for name, dtype in OUTPUT_DTYPES.items() if dtype == np.dtype(np.float32)
)
COMPLETE_METADATA_STATUSES = {"complete", "already_complete", "recovered_complete"}


def schema_version_for_year(target_year: int) -> str:
    if target_year not in (2024, 2025):
        raise ValueError(f"unsupported TROTA target year: {target_year}")
    return f"trota_topresolved_{target_year}_inplace_sparse_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def events_schema_digest(tree: Any) -> str:
    payload = json.dumps(tree.typenames(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def branch_dtype(branch: Any) -> np.dtype[Any]:
    dtype = getattr(branch.interpretation, "numpy_dtype", None)
    if dtype is None:
        raise RuntimeError(f"branch {branch.name} has no NumPy dtype")
    return np.dtype(dtype)


def read_xrootd_json(path: str) -> dict[str, Any]:
    handle = client.File()
    status, _ = handle.open(f"root://eosuser.cern.ch/{path}")
    if not status.ok:
        raise OSError(f"XRootD metadata open failed: {status.message}")
    try:
        status, stat = handle.stat()
        if not status.ok or stat is None:
            raise OSError(f"XRootD metadata stat failed: {status.message}")
        status, payload = handle.read(0, int(stat.size))
        if not status.ok:
            raise OSError(f"XRootD metadata read failed: {status.message}")
    finally:
        handle.close()
    return json.loads(bytes(payload).decode("utf-8"))


def validate_one(
    item: dict[str, Any],
    *,
    full_float_scan: bool,
    target_year: int,
) -> dict[str, Any]:
    expected_schema_version = schema_version_for_year(target_year)
    input_path = str(item["input_root"])
    url = f"root://eosuser.cern.ch/{input_path}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            filesystem = client.FileSystem("root://eosuser.cern.ch")
            status, stat = filesystem.stat(input_path)
            if not status.ok or stat is None:
                raise OSError(f"XRootD stat failed: {status.message}")
            current_bytes = int(stat.size)
            with uproot.open(url, object_cache=None, array_cache=None) as root_file:
                key_names = {str(key).split(";", 1)[0] for key in root_file.keys()}
                required_keys = {"Events", TREE_NAME, MARKER_NAME}
                if not required_keys.issubset(key_names):
                    raise RuntimeError(
                        f"missing ROOT keys: {sorted(required_keys - key_names)}"
                    )
                events = root_file["Events"]
                trota = root_file[TREE_NAME]
                marker = json.loads(str(root_file[MARKER_NAME]))
                event_entries = int(events.num_entries)
                trota_rows = int(trota.num_entries)
                schema_digest = events_schema_digest(events)

                if marker.get("schema_version") != expected_schema_version:
                    raise RuntimeError("unexpected marker schema version")
                if target_year != 2024 and marker.get("application_year") != target_year:
                    raise RuntimeError("application year mismatch")
                if marker.get("status") != "complete":
                    raise RuntimeError("completion marker is not complete")
                if marker.get("model_sha256") != MODEL_SHA256:
                    raise RuntimeError("model SHA256 mismatch")
                if marker.get("trota_commit") != TROTA_COMMIT:
                    raise RuntimeError("TROTA commit mismatch")
                if marker.get("selected_working_point") != WP_NAME:
                    raise RuntimeError("working-point name mismatch")
                if not math.isclose(
                    float(marker.get("selected_threshold")),
                    float(WP_THRESHOLD),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError("working-point threshold mismatch")
                if int(marker.get("events_entries", -1)) != event_entries:
                    raise RuntimeError("Events entry count differs from marker")
                if int(marker.get("selected_candidates", -1)) != trota_rows:
                    raise RuntimeError("TROTA row count differs from marker")
                if marker.get("events_schema_digest") != schema_digest:
                    raise RuntimeError("Events schema digest differs from marker")

                actual_branches = set(trota.keys())
                expected_branches = set(OUTPUT_DTYPES)
                if actual_branches != expected_branches:
                    raise RuntimeError(
                        "TROTA branch set mismatch: "
                        f"missing={sorted(expected_branches - actual_branches)}, "
                        f"extra={sorted(actual_branches - expected_branches)}"
                    )
                for name, expected_dtype in OUTPUT_DTYPES.items():
                    actual_dtype = branch_dtype(trota[name])
                    if actual_dtype != expected_dtype:
                        raise RuntimeError(
                            f"branch {name} dtype {actual_dtype} != {expected_dtype}"
                        )

                minimum_discriminant: float | None = None
                maximum_probability_sum_deviation = 0.0
                maximum_discriminant_difference = 0.0
                rows_scanned = 0
                if full_float_scan:
                    for arrays in trota.iterate(
                        FLOAT_BRANCHES,
                        step_size="32 MB",
                        library="np",
                    ):
                        values = [np.asarray(arrays[name]) for name in FLOAT_BRANCHES]
                        if any(not np.all(np.isfinite(value)) for value in values):
                            raise RuntimeError("non-finite value in float branches")
                        ft = np.asarray(arrays["TopResolved1pct_FTScore"])
                        tt = np.asarray(arrays["TopResolved1pct_TTScore"])
                        qcd = np.asarray(arrays["TopResolved1pct_QCDScore"])
                        disc = np.asarray(arrays["TopResolved1pct_QCDDiscriminant"])
                        if disc.size:
                            chunk_minimum = float(np.min(disc))
                            minimum_discriminant = (
                                chunk_minimum
                                if minimum_discriminant is None
                                else min(minimum_discriminant, chunk_minimum)
                            )
                            if np.any(disc < WP_THRESHOLD):
                                raise RuntimeError("stored candidate is below the 1% WP")
                            probability_deviation = float(
                                np.max(np.abs(ft + tt + qcd - np.float32(1.0)))
                            )
                            maximum_probability_sum_deviation = max(
                                maximum_probability_sum_deviation,
                                probability_deviation,
                            )
                            denominator = tt + qcd
                            if np.any(denominator <= 0):
                                raise RuntimeError("non-positive TTScore + QCDScore")
                            recalculated = tt / denominator
                            disc_difference = float(np.max(np.abs(recalculated - disc)))
                            maximum_discriminant_difference = max(
                                maximum_discriminant_difference,
                                disc_difference,
                            )
                            if not np.allclose(
                                recalculated,
                                disc,
                                rtol=2e-6,
                                atol=2e-7,
                            ):
                                raise RuntimeError("stored discriminator is inconsistent")
                        rows_scanned += int(disc.size)
                    if rows_scanned != trota_rows:
                        raise RuntimeError("full float scan did not read every TROTA row")

            metadata = read_xrootd_json(str(item["job_metadata"]))
            if metadata.get("status") not in COMPLETE_METADATA_STATUSES:
                raise RuntimeError("job metadata is not complete")
            return {
                "name": item["name"],
                "kind": item["kind"],
                "input": input_path,
                "status": "valid",
                "events": event_entries,
                "candidates_evaluated": int(marker["candidates_evaluated"]),
                "selected_candidates": trota_rows,
                "current_bytes": current_bytes,
                "input_bytes_before": int(item["input_bytes_before"]),
                "bytes_added": current_bytes - int(item["input_bytes_before"]),
                "rows_float_scanned": rows_scanned,
                "minimum_discriminant": minimum_discriminant,
                "maximum_probability_sum_deviation": maximum_probability_sum_deviation,
                "maximum_discriminant_difference": maximum_discriminant_difference,
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    assert last_error is not None
    return {
        "name": item["name"],
        "kind": item["kind"],
        "input": input_path,
        "status": "invalid",
        "error_type": type(last_error).__name__,
        "error": str(last_error)[:2000],
        "attempts": 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit every 2024/2025 TROTA ROOT directly through XRootD."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-float-scan", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="validate only this manifest item name; may be supplied repeatedly",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-year", type=int, choices=(2024, 2025), default=2024)
    args = parser.parse_args()

    campaign = args.campaign_dir.absolute()
    manifest = json.loads((campaign / "input_manifest.json").read_text())
    inputs = list(manifest["inputs"])
    if args.name:
        requested = set(args.name)
        inputs = [item for item in inputs if item["name"] in requested]
        found = {item["name"] for item in inputs}
        missing_names = sorted(requested - found)
        if missing_names:
            raise RuntimeError(f"requested manifest names not found: {missing_names}")
    if args.limit is not None:
        inputs = inputs[: args.limit]
    output = args.output or campaign / "xrootd_validation_summary.json"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                validate_one,
                item,
                full_float_scan=args.full_float_scan,
                target_year=args.target_year,
            ): item
            for item in inputs
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if completed % 100 == 0 or completed == len(inputs):
                invalid = sum(result["status"] != "valid" for result in results)
                print(
                    f"validated={completed}/{len(inputs)} invalid={invalid}",
                    flush=True,
                )

    results.sort(key=lambda result: result["name"])
    valid = [result for result in results if result["status"] == "valid"]
    invalid = [result for result in results if result["status"] != "valid"]
    minimum_values = [
        result["minimum_discriminant"]
        for result in valid
        if result["minimum_discriminant"] is not None
    ]
    summary = {
        "schema_version": f"trota_topresolved_{args.target_year}_xrootd_audit_v1",
        "application_year": args.target_year,
        "model_release_year": 2024,
        "updated_at": now(),
        "status": "complete" if not invalid and len(valid) == len(inputs) else "failed",
        "xrootd_endpoint": "root://eosuser.cern.ch",
        "input_digest": manifest["input_digest"],
        "input_files": len(inputs),
        "valid": len(valid),
        "invalid": len(invalid),
        "full_float_scan": args.full_float_scan,
        "events": sum(result["events"] for result in valid),
        "candidates_evaluated": sum(
            result["candidates_evaluated"] for result in valid
        ),
        "selected_candidates": sum(
            result["selected_candidates"] for result in valid
        ),
        "rows_float_scanned": sum(result["rows_float_scanned"] for result in valid),
        "input_bytes_before": sum(result["input_bytes_before"] for result in valid),
        "current_bytes": sum(result["current_bytes"] for result in valid),
        "bytes_added": sum(result["bytes_added"] for result in valid),
        "minimum_discriminant": min(minimum_values) if minimum_values else None,
        "maximum_probability_sum_deviation": max(
            (result["maximum_probability_sum_deviation"] for result in valid),
            default=0.0,
        ),
        "maximum_discriminant_difference": max(
            (result["maximum_discriminant_difference"] for result in valid),
            default=0.0,
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "failures": invalid,
        "files": results,
    }
    atomic_write_json(output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "files"}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
