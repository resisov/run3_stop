#!/usr/bin/env python3
"""Extract one process from a mixed flat-ntuple ROOT/JSON pair.

The output keeps every event branch unchanged, filters ``Events`` by the
dataset identifiers advertised for the requested process, and writes a
matching normalization sidecar.  The parent artifacts remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import awkward as ak
import uproot


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--keep-process", required=True)
    args = parser.parse_args()

    metadata = json.loads(args.input_json.read_text())
    if metadata.get("status") not in {"complete", "complete_with_bad_files"}:
        raise RuntimeError(f"parent metadata is not complete: {metadata.get('status')}")

    kept_datasets = {
        str(dataset_id): copy.deepcopy(record)
        for dataset_id, record in (metadata.get("datasets") or {}).items()
        if record.get("process") == args.keep_process
    }
    if not kept_datasets:
        raise RuntimeError(
            f"parent metadata contains no {args.keep_process!r} datasets"
        )
    kept_ids = {int(dataset_id) for dataset_id in kept_datasets}

    with uproot.open(args.input_root) as source:
        keys = {str(key).split(";", 1)[0] for key in source.keys()}
        if keys != {"Events"}:
            raise RuntimeError(f"unexpected parent ROOT keys: {sorted(keys)}")
        tree = source["Events"]
        if "dataset_id" not in tree.keys():
            raise RuntimeError("parent Events tree has no dataset_id branch")
        arrays = tree.arrays(library="ak")
        mask = ak.zeros_like(arrays["dataset_id"], dtype=bool)
        for dataset_id in sorted(kept_ids):
            mask = mask | (arrays["dataset_id"] == dataset_id)
        filtered = arrays[mask]
        parent_entries = int(tree.num_entries)

    output_entries = int(len(filtered))
    expected_entries = sum(
        int(record.get("events_written") or 0)
        for record in kept_datasets.values()
    )
    if output_entries != expected_entries:
        raise RuntimeError(
            "filtered ROOT entries do not match kept metadata: "
            f"{output_entries} != {expected_entries}"
        )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output_root) as target:
        target.mktree(
            "Events",
            {field: filtered[field] for field in filtered.fields},
        )

    with uproot.open(args.output_root) as check:
        output_tree = check["Events"]
        if int(output_tree.num_entries) != output_entries:
            raise RuntimeError("output ROOT entry validation failed")
        if set(output_tree.keys()) != set(filtered.fields):
            raise RuntimeError("output ROOT branch schema differs from parent selection")
        output_ids = {
            int(value)
            for value in output_tree["dataset_id"].array(library="np")
        }
        if output_ids != kept_ids:
            raise RuntimeError(
                f"output dataset identifiers differ: {output_ids} != {kept_ids}"
            )

    kept_physical_names = {
        str(record.get("physical_dataset") or record.get("dataset"))
        for record in kept_datasets.values()
    }
    kept_files = [
        copy.deepcopy(record)
        for record in (metadata.get("files") or [])
        if record.get("process") == args.keep_process
    ]
    output = copy.deepcopy(metadata)
    output["datasets"] = kept_datasets
    output["files"] = kept_files
    output["physical_datasets"] = {
        name: copy.deepcopy(record)
        for name, record in (metadata.get("physical_datasets") or {}).items()
        if name in kept_physical_names
    }
    output["physical_dataset_split_counts"] = {
        name: int(value)
        for name, value in (
            metadata.get("physical_dataset_split_counts") or {}
        ).items()
        if name in kept_physical_names
    }
    output["files_attempted"] = sum(
        int(record.get("files_attempted") or 0)
        for record in kept_datasets.values()
    )
    output["files_processed"] = sum(
        int(record.get("files_processed") or 0)
        for record in kept_datasets.values()
    )
    output["events_read"] = sum(
        int(record.get("events_read") or 0)
        for record in kept_datasets.values()
    )
    output["events_written"] = output_entries
    output["records_in_shard"] = len(kept_datasets)
    output["root_file"] = str(args.output_root)
    output["completed_at"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    output["record_digest"] = hashlib.sha256(
        (
            str(metadata.get("record_digest"))
            + "\0"
            + args.keep_process
            + "\0"
            + str(output_entries)
        ).encode()
    ).hexdigest()[:16]
    output["process_split"] = {
        "schema_version": "flat_ntuple_process_split_v1",
        "parent_root": str(args.input_root),
        "parent_json": str(args.input_json),
        "parent_root_sha256": sha256(args.input_root),
        "parent_json_sha256": sha256(args.input_json),
        "parent_entries": parent_entries,
        "kept_process": args.keep_process,
        "kept_dataset_ids": sorted(kept_ids),
        "kept_entries": output_entries,
        "removed_entries": parent_entries - output_entries,
        "policy": "exact Events.dataset_id membership from parent sidecar",
    }
    write_json(args.output_json, output)

    report = {
        "status": "complete",
        "parent_entries": parent_entries,
        "kept_entries": output_entries,
        "removed_entries": parent_entries - output_entries,
        "kept_process": args.keep_process,
        "kept_dataset_ids": sorted(kept_ids),
        "output_root": str(args.output_root),
        "output_json": str(args.output_json),
        "output_root_sha256": sha256(args.output_root),
        "output_json_sha256": sha256(args.output_json),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
