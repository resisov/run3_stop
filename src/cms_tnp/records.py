"""DAS file discovery and deterministic shard manifests."""

from __future__ import annotations

import subprocess
from typing import Any, Mapping


def _das(query: str, executable: str = "dasgoclient") -> list[str]:
    result = subprocess.run(
        [executable, f"--query={query}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def discover(
    config: Mapping[str, Any], sample: str, executable: str = "dasgoclient"
) -> dict[str, Any]:
    if sample not in {"data", "mc"}:
        raise ValueError("sample must be data or mc")
    sources = [str(value) for value in config.get("samples", {}).get(sample, [])]
    if not sources:
        raise ValueError(f"samples.{sample} is empty")
    datasets: list[str] = []
    for source in sources:
        if source.startswith("/"):
            datasets.append(source)
        else:
            datasets.extend(_das(source, executable))
    datasets = sorted(set(datasets))
    records = []
    for dataset in datasets:
        for file_path in _das(f"file dataset={dataset}", executable):
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "file_path": file_path
                    if file_path.startswith("root://")
                    else f"root://cms-xrd-global.cern.ch/{file_path}",
                }
            )
    records.sort(key=lambda item: (item["dataset"], item["file_path"]))
    return {
        "schema_version": 1,
        "measurement": config["measurement"],
        "year": str(config["year"]),
        "sample": sample,
        "datasets": datasets,
        "records": records,
    }


def shards(records: Mapping[str, Any], files_per_shard: int) -> list[dict[str, Any]]:
    if files_per_shard <= 0:
        raise ValueError("files_per_shard must be positive")
    items = list(records.get("records", []))
    return [
        {
            "schema_version": 1,
            "measurement": records["measurement"],
            "year": records["year"],
            "sample": records["sample"],
            "shard_id": index // files_per_shard,
            "records": items[index : index + files_per_shard],
        }
        for index in range(0, len(items), files_per_shard)
    ]


def shard_files(shard: Mapping[str, Any]) -> list[str]:
    return [str(item["file_path"]) for item in shard.get("records", [])]
