#!/usr/bin/env python3
"""Build low-pT tag-and-probe file records from DAS datasets."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _das(query: str) -> list[str]:
    completed = subprocess.run(
        ["dasgoclient", f"--query={query}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})


def build_records(config: dict[str, Any]) -> dict[str, Any]:
    campaign = config["campaign_inputs"]
    queries = campaign.get("data_dataset_queries")
    if queries is None:
        queries = [campaign["data_dataset_query"]]
    data_datasets = sorted({
        dataset
        for query in queries
        for dataset in _das(str(query))
    })
    excluded = tuple(str(value) for value in campaign.get("data_dataset_exclude_contains", []))
    data_datasets = [dataset for dataset in data_datasets if not any(token in dataset for token in excluded)]
    mc_datasets = [str(value) for value in campaign["mc_datasets"]]
    if not data_datasets or not mc_datasets:
        raise RuntimeError("TnP data or MC dataset selection is empty")
    records = []
    dataset_audit = {}
    seen_files: set[str] = set()
    for sample, datasets in (("data", data_datasets), ("mc", mc_datasets)):
        for dataset in datasets:
            files = _das(f"file dataset={dataset}")
            if not files:
                raise RuntimeError(f"DAS returned no files for {dataset}")
            dataset_audit[dataset] = {"sample": sample, "files": len(files)}
            for index, lfn in enumerate(files):
                file_path = lfn if lfn.startswith("root://") else f"root://cms-xrd-global.cern.ch/{lfn}"
                if file_path in seen_files:
                    raise RuntimeError(f"duplicate TnP ROOT file: {file_path}")
                seen_files.add(file_path)
                records.append({
                    "dataset": dataset,
                    "sample": sample,
                    "is_data": sample == "data",
                    "year": str(config["year"]),
                    "file_index": index,
                    "file_path": file_path,
                })
    records.sort(key=lambda item: (item["sample"] == "mc", item["dataset"], item["file_path"]))
    return {
        "schema_version": 1,
        "measurement": config["measurement"],
        "probe_definition": config.get("probe_definition"),
        "tag_pt_min_gev": config.get("tag_pt_min_gev"),
        "reference_paths": list(config.get("reference_paths") or []),
        "files_per_condor_shard": int(campaign["files_per_condor_shard"]),
        "dataset_audit": dataset_audit,
        "data_files": sum(item["sample"] == "data" for item in records),
        "mc_files": sum(item["sample"] == "mc" for item in records),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_records(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"records", "dataset_audit"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
