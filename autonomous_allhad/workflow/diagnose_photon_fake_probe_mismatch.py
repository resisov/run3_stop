#!/usr/bin/env python3
"""Inspect the NanoAOD photon multiplicities of nominal-only GCR events."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from autonomous_allhad import photon_fake_2024_worker as fake_worker
from autonomous_allhad import real_subset_worker as baseline


KEY_FIELDS = ("run", "luminosityBlock", "event")
ROOT_FIELDS = (*KEY_FIELDS, "file_id", "entry", "feature_GCR")
PHOTON_FIELDS = (
    "Photon_pt",
    "Photon_eta",
    "Photon_electronVeto",
    "Photon_vidNestedWPBitmap",
    "Photon_cutBased",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--nominal-data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    missing_records = list(
        (audit.get("comparison") or {}).get("nominal_only_examples") or []
    )
    missing_keys = {
        tuple(int(record[field]) for field in KEY_FIELDS) for record in missing_records
    }
    expected_count = int(
        (audit.get("comparison") or {}).get("nominal_only_event_keys") or 0
    )
    if len(missing_keys) != expected_count:
        raise RuntimeError(
            "audit does not contain every nominal-only key: "
            f"{len(missing_keys)} vs {expected_count}"
        )

    file_map: dict[int, dict[str, str]] = {}
    for metadata_path in sorted(args.nominal_data_dir.glob("EGamma*.json")):
        metadata = json.loads(metadata_path.read_text())
        for record in metadata.get("files") or []:
            identifier = int(record["file_id"])
            value = {
                "file_path": str(record["file_path"]),
                "dataset": str(record["dataset"]),
            }
            previous = file_map.get(identifier)
            if previous is not None and previous != value:
                raise RuntimeError(f"file-id collision for {identifier}")
            file_map[identifier] = value

    locations: dict[tuple[int, int, int], dict[str, Any]] = {}
    for root_path in sorted(args.nominal_data_dir.glob("EGamma*.root")):
        with uproot.open(root_path) as source:
            arrays = source["Events"].arrays(ROOT_FIELDS, library="np")
        gcr = np.asarray(arrays["feature_GCR"], dtype=bool)
        for index in np.flatnonzero(gcr):
            key = tuple(int(arrays[field][index]) for field in KEY_FIELDS)
            if key not in missing_keys:
                continue
            file_id = int(arrays["file_id"][index])
            source_record = file_map.get(file_id)
            if source_record is None:
                raise RuntimeError(f"unknown nominal file_id {file_id}")
            locations[key] = {
                **source_record,
                "file_id": file_id,
                "entry": int(arrays["entry"][index]),
                "nominal_root": str(root_path),
            }

    absent_locations = sorted(missing_keys - set(locations))
    if absent_locations:
        raise RuntimeError(f"nominal-only keys not located: {absent_locations}")

    by_file: dict[str, list[tuple[tuple[int, int, int], dict[str, Any]]]] = defaultdict(list)
    for key, location in locations.items():
        by_file[str(location["file_path"])].append((key, location))

    diagnostics: list[dict[str, Any]] = []
    for file_path, records in sorted(by_file.items()):
        with uproot.open(file_path, timeout=120) as source:
            tree = source["Events"]
            for key, location in sorted(records, key=lambda item: item[1]["entry"]):
                entry = int(location["entry"])
                arrays = tree.arrays(
                    PHOTON_FIELDS,
                    entry_start=entry,
                    entry_stop=entry + 1,
                    library="ak",
                )
                masks = fake_worker.probe_masks(arrays)
                target_count = int(ak.sum(masks["target"], axis=1)[0])
                pass_count = int(ak.sum(masks["measurement_pass"], axis=1)[0])
                fail_count = int(ak.sum(masks["measurement_fail"], axis=1)[0])
                application_count = int(ak.sum(masks["application"], axis=1)[0])
                union_count = target_count + pass_count + fail_count + application_count
                nominal_mask = baseline.medium_photon_mask(
                    arrays["Photon_pt"],
                    arrays["Photon_eta"],
                    arrays["Photon_cutBased"],
                    arrays["Photon_electronVeto"],
                )
                nominal_medium_count = int(ak.sum(nominal_mask, axis=1)[0])
                _masks, selected, codes, candidates = fake_worker._probe_assignment(
                    arrays
                )
                diagnostics.append(
                    {
                        "run": key[0],
                        "luminosityBlock": key[1],
                        "event": key[2],
                        **location,
                        "photon_pt": ak.to_list(arrays["Photon_pt"][0]),
                        "target_count": target_count,
                        "measurement_pass_count": pass_count,
                        "measurement_fail_count": fail_count,
                        "application_count": application_count,
                        "union_count": union_count,
                        "nominal_medium_count": nominal_medium_count,
                        "old_exact_union_candidate": union_count == 1,
                        "corrected_candidate": bool(candidates[0]),
                        "corrected_probe_code": int(codes[0]),
                        "corrected_selected_count": int(ak.sum(selected, axis=1)[0]),
                    }
                )

    summary = {
        "events": len(diagnostics),
        "one_target_plus_sideband": sum(
            int(
                item["target_count"] == 1
                and item["union_count"] > item["target_count"]
            )
            for item in diagnostics
        ),
        "old_rejected": sum(
            int(not item["old_exact_union_candidate"]) for item in diagnostics
        ),
        "corrected_accepted": sum(
            int(item["corrected_candidate"]) for item in diagnostics
        ),
        "nominal_medium_exactly_one": sum(
            int(item["nominal_medium_count"] == 1) for item in diagnostics
        ),
    }
    payload = {
        "schema_version": "photon_fake_probe_mismatch_diagnostic_v1",
        "selection_source": "real_subset_worker.py",
        "summary": summary,
        "events": diagnostics,
    }
    write_json(args.output, payload)
    print(json.dumps(summary, sort_keys=True))
    if (
        summary["events"] != expected_count
        or summary["one_target_plus_sideband"] != expected_count
        or summary["old_rejected"] != expected_count
        or summary["corrected_accepted"] != expected_count
        or summary["nominal_medium_exactly_one"] != expected_count
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
