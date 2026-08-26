#!/usr/bin/env python3
"""Build low-pT tag-and-probe file records from DAS datasets."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _das(
    query: str,
    *,
    dasgoclient: Path | str = "dasgoclient",
    dasmaps: Path | None = None,
) -> list[str]:
    command = [str(dasgoclient)]
    if dasmaps is not None:
        command.extend(["--dasmaps", str(dasmaps)])
    command.append(f"--query={query}")
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})


def build_records(
    config: dict[str, Any],
    *,
    dasgoclient: Path | str = "dasgoclient",
    dasmaps: Path | None = None,
    samples: set[str] | None = None,
) -> dict[str, Any]:
    selected_samples = set(samples or {"data", "mc"})
    if not selected_samples or not selected_samples <= {"data", "mc"}:
        raise ValueError(f"samples must be a non-empty subset of data/mc: {selected_samples}")
    campaign = config["campaign_inputs"]
    data_datasets: list[str] = []
    if "data" in selected_samples:
        queries = campaign.get("data_dataset_queries")
        if queries is None:
            queries = [campaign["data_dataset_query"]]
        data_datasets = sorted({
            dataset
            for query in queries
            for dataset in _das(str(query), dasgoclient=dasgoclient, dasmaps=dasmaps)
        })
        excluded = tuple(str(value) for value in campaign.get("data_dataset_exclude_contains", []))
        data_datasets = [
            dataset for dataset in data_datasets
            if not any(token in dataset for token in excluded)
        ]
        if not data_datasets:
            raise RuntimeError("TnP data dataset selection is empty")
    mc_datasets = (
        [str(value) for value in campaign["mc_datasets"]]
        if "mc" in selected_samples
        else []
    )
    if "mc" in selected_samples and not mc_datasets:
        raise RuntimeError("TnP MC dataset selection is empty")
    records = []
    dataset_audit = {}
    seen_files: set[str] = set()
    for sample, datasets in (("data", data_datasets), ("mc", mc_datasets)):
        for dataset in datasets:
            files = _das(
                f"file dataset={dataset}",
                dasgoclient=dasgoclient,
                dasmaps=dasmaps,
            )
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
        "year": str(config["year"]),
        "probe_definition": config.get("probe_definition"),
        "tag_pt_min_gev": config.get("tag_pt_min_gev"),
        "reference_paths": list(config.get("reference_paths") or []),
        "files_per_condor_shard": int(campaign["files_per_condor_shard"]),
        "dataset_audit": dataset_audit,
        "data_files": sum(item["sample"] == "data" for item in records),
        "mc_files": sum(item["sample"] == "mc" for item in records),
        "selected_samples": sorted(selected_samples),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dasgoclient", type=Path, default=Path("dasgoclient"))
    parser.add_argument("--dasmaps", type=Path)
    parser.add_argument(
        "--sample",
        choices=("data", "mc"),
        action="append",
        help="sample to freeze; repeat for both (default: data and mc)",
    )
    args = parser.parse_args(argv)
    payload = build_records(
        json.loads(args.config.read_text()),
        dasgoclient=args.dasgoclient,
        dasmaps=args.dasmaps,
        samples=set(args.sample) if args.sample else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"records", "dataset_audit"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
