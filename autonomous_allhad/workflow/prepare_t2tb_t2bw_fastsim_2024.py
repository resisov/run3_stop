#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


TOPOLOGIES = ("T2tb", "T2bW")
DATASET_PATTERN = (
    "/SMS-2Stop-{topology}_Par-mStop-*_TuneCP5_13p6TeV_madgraphMLM-pythia8/"
    "RunIII2024Summer24NanoAODv15-FSMiniv6_FSNanov15_150X_mcRun3_2024_realistic_v2-v1/"
    "NANOAODSIM"
)


def run_das(dasgoclient: Path, dasmaps: Path, query: str, *, as_json: bool = False) -> Any:
    command = [
        str(dasgoclient),
        "-dasmaps",
        str(dasmaps),
        "-query",
        query,
    ]
    if as_json:
        command.append("-json")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    if as_json:
        return json.loads(result.stdout)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def dataset_summary(payload: list[dict[str, Any]]) -> dict[str, int]:
    for record in payload:
        values = record.get("summary") or []
        if values:
            value = values[0]
            return {
                "files": int(value.get("nfiles") or value.get("num_file") or 0),
                "events": int(value.get("nevents") or value.get("num_event") or 0),
                "bytes": int(value.get("file_size") or 0),
            }
    return {"files": 0, "events": 0, "bytes": 0}


def inspect_dataset(
    dasgoclient: Path,
    dasmaps: Path,
    topology: str,
    dataset: str,
) -> dict[str, Any]:
    files = run_das(
        dasgoclient,
        dasmaps,
        f"file dataset={dataset} instance=prod/global",
    )
    summary = dataset_summary(
        run_das(
            dasgoclient,
            dasmaps,
            f"summary dataset={dataset} instance=prod/global",
            as_json=True,
        )
    )
    primary = dataset.strip("/").split("/", 1)[0]
    mstop = int(primary.split("Par-mStop-", 1)[1].split("_", 1)[0])
    if len(files) != summary["files"]:
        raise RuntimeError(
            f"{dataset}: DAS file count {len(files)} != summary {summary['files']}"
        )
    return {
        "topology": topology,
        "mStop_dataset": mstop,
        "dataset": dataset,
        "summary": summary,
        "files": files,
    }


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--dasgoclient",
        default="/cvmfs/cms.cern.ch/common/dasgoclient",
    )
    parser.add_argument("--dasmaps", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    campaign = Path(args.campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "shards").mkdir(exist_ok=True)
    dasgoclient = Path(args.dasgoclient)
    dasmaps = Path(args.dasmaps)

    discovered: list[tuple[str, str]] = []
    for topology in TOPOLOGIES:
        datasets = run_das(
            dasgoclient,
            dasmaps,
            (
                f"dataset={DATASET_PATTERN.format(topology=topology)} "
                "instance=prod/global"
            ),
        )
        discovered.extend((topology, dataset) for dataset in datasets)
    if not discovered:
        raise RuntimeError("no T2tb/T2bW FastSim datasets discovered")

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        futures = {
            pool.submit(
                inspect_dataset,
                dasgoclient,
                dasmaps,
                topology,
                dataset,
            ): (topology, dataset)
            for topology, dataset in discovered
        }
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: (item["topology"], item["mStop_dataset"]))

    shards: list[dict[str, Any]] = []
    btag_metadata: dict[str, Any] = {}
    all_files = 0
    all_events = 0
    for index, item in enumerate(records):
        dataset = item["dataset"]
        topology = item["topology"]
        files = item["files"]
        all_files += len(files)
        all_events += int(item["summary"]["events"])
        shard_id = f"signal_{topology.lower()}_shard_{index:03d}"
        shard_records = [
            {
                "sample_name": dataset,
                "dataset": dataset,
                "process_group": "SMS",
                "year": "2024",
                "file_index": file_index,
                "file_path": f"root://cms-xrd-global.cern.ch/{lfn}",
                "xsec_pb": None,
                "is_data": False,
                "is_background": False,
                "is_signal": True,
                "simulation_type": "FastSim signal dataset",
                "signal_topology": topology,
                "sumw_source": (
                    f"Runs.genEventSumw_{topology}_<mStop>_<mLSP>"
                ),
            }
            for file_index, lfn in enumerate(files)
        ]
        shard_payload = {
            "schema_version": "full_production_shard_spec_v5_fullselection_2024",
            "shard_id": shard_id,
            "record_group": "signal",
            "records_per_shard": len(shard_records),
            "record_digest": digest(shard_records)[:16],
            "records": shard_records,
        }
        shard_path = campaign / "shards" / f"{shard_id}.json"
        shard_path.write_text(json.dumps(shard_payload, indent=2, sort_keys=True) + "\n")
        shards.append(
            {
                "name": shard_id,
                "topology": topology,
                "dataset": dataset,
                "mStop_dataset": item["mStop_dataset"],
                "records": len(shard_records),
                "events": int(item["summary"]["events"]),
                "record_digest": shard_payload["record_digest"],
                "shard": str(shard_path),
                "root": str(campaign / "outputs/nominal" / f"{shard_id}.root"),
                "json": str(campaign / "outputs/nominal" / f"{shard_id}.json"),
            }
        )
        parts = dataset.strip("/").split("/")
        btag_key = f"{parts[0]}-{parts[1]}"
        btag_metadata[btag_key] = {
            "files": [
                f"root://xrootd-cms.infn.it/{lfn}" for lfn in files
            ],
            "xs": -1,
        }

    (campaign / "KNU_2024_t2tb_t2bw.json").write_text(
        json.dumps(btag_metadata, indent=2, sort_keys=True) + "\n"
    )
    expected_mass_grid = sorted({item["mStop_dataset"] for item in records})
    missing_mass_datasets = {
        topology: sorted(
            set(expected_mass_grid)
            - {
                item["mStop_dataset"]
                for item in records
                if item["topology"] == topology
            }
        )
        for topology in TOPOLOGIES
    }
    manifest = {
        "schema_version": "t2tb_t2bw_fastsim_2024_campaign_v1",
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "topologies": list(TOPOLOGIES),
        "datasets": len(records),
        "datasets_by_topology": {
            topology: sum(item["topology"] == topology for item in records)
            for topology in TOPOLOGIES
        },
        "expected_mass_grid_from_union": expected_mass_grid,
        "missing_mass_datasets_by_topology": missing_mass_datasets,
        "files": all_files,
        "events": all_events,
        "input_fingerprint": digest(records),
        "fullsim_files": 0,
        "fastsim_trigger_bypass_required": True,
        "sumw_policy": "Runs.genEventSumw_<topology>_<mStop>_<mLSP>",
        "datasets_detail": records,
        "shards": shards,
        "btag_metadata": str(campaign / "KNU_2024_t2tb_t2bw.json"),
        "representative_files": {
            topology: next(
                (
                    f"root://cms-xrd-global.cern.ch/{item['files'][0]}"
                    for item in records
                    if item["topology"] == topology and item["files"]
                ),
                None,
            )
            for topology in TOPOLOGIES
        },
    }
    (campaign / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "datasets": manifest["datasets"],
                "files": all_files,
                "events": all_events,
                "campaign": str(campaign),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
