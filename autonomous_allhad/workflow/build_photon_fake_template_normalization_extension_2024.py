#!/usr/bin/env python3
"""Extend the validated 2024 normalization with current rare backgrounds.

Only the NanoAOD ``Runs`` tree is read.  The program refuses to publish an
extended normalization if any selected file cannot provide
``genEventSumw`` and ``genEventCount`` after alternate-endpoint retries.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import uproot


DEFAULT_DATASET_REGEX = r"^(TTWW|TTWZ|TTZZ|WWW|WWZ|WZZ|ZZZ)"


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def endpoints(path: str) -> list[str]:
    values = [path]
    marker = "//store/"
    if marker in path:
        lfn = "/store/" + path.split(marker, 1)[1]
        values.extend(
            [
                "root://cmsxrootd.fnal.gov/" + lfn,
                "root://xrootd-cms.infn.it/" + lfn,
            ]
        )
    return list(dict.fromkeys(values))


def read_runs(source: tuple[str, str]) -> dict[str, Any]:
    dataset, path = source
    failures: list[dict[str, str]] = []
    for attempt, endpoint in enumerate(endpoints(path), start=1):
        try:
            with uproot.open(endpoint, timeout=120) as root_file:
                runs = root_file["Runs"]
                available = set(runs.keys())
                required = {"genEventSumw", "genEventCount"}
                missing = sorted(required - available)
                if missing:
                    raise RuntimeError(f"Runs branches missing: {missing}")
                requested = ["genEventSumw", "genEventCount"]
                has_sumw2 = "genEventSumw2" in available
                if has_sumw2:
                    requested.append("genEventSumw2")
                arrays = runs.arrays(requested, library="np")
            sumw = float(np.sum(arrays["genEventSumw"], dtype=np.float64))
            events = int(np.sum(arrays["genEventCount"], dtype=np.int64))
            sumw2 = (
                float(np.sum(arrays["genEventSumw2"], dtype=np.float64))
                if has_sumw2
                else None
            )
            if not math.isfinite(sumw) or sumw == 0.0 or events <= 0:
                raise RuntimeError(f"invalid Runs totals: sumw={sumw}, events={events}")
            return {
                "dataset": dataset,
                "file_path": path,
                "endpoint": endpoint,
                "endpoint_attempt": attempt,
                "alternate_access_attempted": attempt > 1,
                "events": events,
                "sumw": sumw,
                "sumw2": sumw2,
                "status": "complete",
                "failures_before_success": failures,
            }
        except Exception as exc:
            failures.append(
                {
                    "endpoint": endpoint,
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )
    return {
        "dataset": dataset,
        "file_path": path,
        "status": "failed",
        "alternate_access_attempted": len(endpoints(path)) > 1,
        "failures": failures,
    }


def process_for(dataset: str) -> str:
    return "TT" if dataset.startswith(("TTWW", "TTWZ", "TTZZ")) else "VV"


def dataset_id(dataset: str, occupied: set[str]) -> str:
    candidate = str(int(hashlib.sha256(dataset.encode()).hexdigest()[:15], 16))
    while candidate in occupied:
        candidate = str(int(candidate) + 1)
    occupied.add(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--dataset-regex", default=DEFAULT_DATASET_REGEX)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    metadata = read_json(args.metadata)
    normalization = read_json(args.base_normalization)
    if normalization.get("status") != "complete":
        raise RuntimeError("base normalization is not complete")
    pattern = re.compile(args.dataset_regex)
    selected = {
        str(dataset): record
        for dataset, record in metadata.items()
        if pattern.search(str(dataset)) is not None
    }
    if not selected:
        raise RuntimeError("dataset regex selected no metadata records")
    sources = [
        (dataset, str(path))
        for dataset, record in sorted(selected.items())
        for path in (record.get("files") or [])
    ]
    started = time.time()
    file_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(read_runs, source): source for source in sources}
        for index, future in enumerate(as_completed(futures), start=1):
            file_records.append(future.result())
            if index % 50 == 0 or index == len(futures):
                print(json.dumps({"completed": index, "total": len(futures)}, sort_keys=True), flush=True)
    file_records.sort(key=lambda item: (item["dataset"], item["file_path"]))
    failed = [item for item in file_records if item["status"] != "complete"]
    audit = {
        "schema_version": "photon_fake_template_normalization_runs_audit_2024_v1",
        "status": "failed" if failed else "complete",
        "dataset_regex": args.dataset_regex,
        "datasets": len(selected),
        "files_attempted": len(file_records),
        "files_processed": len(file_records) - len(failed),
        "failed_files": failed,
        "file_records": file_records,
        "wall_time_s": time.time() - started,
    }
    write_json(args.audit_output, audit)
    if failed:
        raise RuntimeError(f"{len(failed)} Runs files failed; see {args.audit_output}")

    occupied = set((normalization.get("datasets") or {}).keys())
    luminosity_pb = float(normalization["luminosity_pb"])
    extension_records: list[dict[str, Any]] = []
    for dataset, metadata_record in sorted(selected.items()):
        records = [item for item in file_records if item["dataset"] == dataset]
        sumw = float(sum(item["sumw"] for item in records))
        sumw2_values = [item["sumw2"] for item in records]
        sumw2 = (
            float(sum(value for value in sumw2_values if value is not None))
            if all(value is not None for value in sumw2_values)
            else None
        )
        events = int(sum(item["events"] for item in records))
        xsec = float(metadata_record["xs"])
        process = process_for(dataset)
        factor = luminosity_pb * xsec / sumw
        did = dataset_id(dataset, occupied)
        dataset_record = {
            "dataset": dataset,
            "dataset_id": did,
            "events_read": events,
            "events_written": 0,
            "files_attempted": len(records),
            "files_processed": len(records),
            "is_background": True,
            "is_data": False,
            "is_signal": False,
            "physical_dataset": dataset,
            "process": process,
            "sumw": sumw,
            "sumw2": sumw2,
            "sumw_source_counts": {"Runs.genEventSumw": len(records)},
            "xsec_pb": xsec,
        }
        factor_record = {
            "dataset": dataset,
            "dataset_id": did,
            "dataset_sumw": sumw,
            "is_data": False,
            "is_signal": False,
            "normalization_factor": factor,
            "normalization_status": "normalized_with_xsec_lumi_physical_dataset_sumw",
            "physical_dataset": dataset,
            "physical_dataset_sumw": sumw,
            "process": process,
            "xsec_pb": xsec,
        }
        physical_record = {
            "dataset_splits": [dataset],
            "events_read": events,
            "files_attempted": len(records),
            "files_processed": len(records),
            "is_background": True,
            "is_data": False,
            "is_signal": False,
            "normalization_factor": factor,
            "physical_dataset": dataset,
            "process": process,
            "sumw": sumw,
            "sumw2": sumw2,
            "sumw_source_counts": {"Runs.genEventSumw": len(records)},
            "xsec_pb": xsec,
        }
        normalization.setdefault("datasets", {})[did] = dataset_record
        normalization.setdefault("dataset_factors", {})[did] = factor_record
        normalization.setdefault("physical_datasets", {})[dataset] = physical_record
        normalization.setdefault("physical_dataset_split_counts", {})[dataset] = 1
        extension_records.append(physical_record)

    normalization["schema_version"] = "flat_ntuple_campaign_normalization_v1_with_photon_template_rare_extension"
    normalization["photon_template_rare_extension"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_normalization": str(args.base_normalization),
        "metadata": str(args.metadata),
        "runs_audit": str(args.audit_output),
        "dataset_regex": args.dataset_regex,
        "datasets": extension_records,
    }
    write_json(args.output, normalization)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "audit": str(args.audit_output),
                "datasets": len(extension_records),
                "files": len(file_records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
