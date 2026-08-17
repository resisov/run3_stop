#!/usr/bin/env python3
"""Prepare a small, normalization-independent DYto2X-4Jets validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, Any]] = {
    "DYto2E": {
        "dataset": (
            "DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8"
            "-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3"
        ),
        "files": [
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/0022201b-32d6-48e8-8dba-1644b64aed9e.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/00314b9f-dbd8-4473-9126-2dbbcf466018.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/0032f8b6-3d32-46cd-babc-8117bbc2fbed.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/0096f457-f104-40ec-97b9-ea53b49df6a1.root",
        ],
    },
    "DYto2Mu": {
        "dataset": (
            "DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8"
            "-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3"
        ),
        "files": [
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/01e7af71-8db6-4af9-a793-d1ccb7b3a34b.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/05db2f6c-18bb-402f-8826-06ecf9d2516a.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/087ccd2b-cf4f-467d-8ecd-64dd4a7d598e.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/09ad2035-0043-4701-849e-586f05b2bdd7.root",
        ],
    },
    "DYto2Tau": {
        "dataset": (
            "DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8"
            "-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v5"
        ),
        "files": [
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v5/120000/02144a2c-4d54-4ba9-9e7f-60a1876190ad.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v5/120000/0231374f-c186-4c75-b2a1-fc5d6f7bd7fe.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v5/120000/024be0f3-79a2-468d-a6ce-5ef41f9e5b53.root",
            "/store/mc/RunIII2024Summer24NanoAODv15/DYto2Tau-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v5/120000/0257df6b-c54b-4c1c-8a1f-02b8b8c4eb5d.root",
        ],
    },
}


def stable_id(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shards = args.output_dir / "shards"
    outputs = args.output_dir / "outputs"
    logs = args.output_dir / "logs"
    for directory in (shards, outputs, logs):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_datasets: dict[str, Any] = {}
    normalization = {"dataset_factors": {}}
    for flavor, specification in DATASETS.items():
        dataset = str(specification["dataset"])
        records = []
        for index, path in enumerate(specification["files"]):
            records.append(
                {
                    "dataset": dataset,
                    "file_index": index,
                    "file_path": f"root://cms-xrd-global.cern.ch/{path}",
                    "is_background": True,
                    "is_data": False,
                    "is_signal": False,
                    "process_group": "DY",
                    "sample_name": dataset,
                    "sumw_source": (
                        "Runs.genEventSumw preferred; Events.genWeight fallback"
                    ),
                    # Deliberately diagnostic-only. Absolute normalization is
                    # not used until an authoritative cross section is adopted.
                    "xsec_pb": 1.0,
                    "year": "2024",
                }
            )
        shard_id = f"{flavor.lower()}_4jets_subset"
        write_json(
            shards / f"{shard_id}.json",
            {
                "record_digest": hashlib.sha256(
                    json.dumps(records, sort_keys=True).encode()
                ).hexdigest()[:16],
                "record_group": "mc",
                "records": records,
                "records_per_shard": len(records),
                "schema_version": (
                    "full_production_shard_spec_v5_"
                    "fullselection_2024_dyto2x4jets_subset"
                ),
                "shard_id": shard_id,
            },
        )
        dataset_id = str(stable_id(dataset))
        normalization["dataset_factors"][dataset_id] = {
            "normalization_factor": 1.0,
            "policy": "diagnostic_gen_weight_only",
        }
        manifest_datasets[flavor] = {
            "dataset": dataset,
            "dataset_id": int(dataset_id),
            "files": len(records),
            "shard_id": shard_id,
        }

    write_json(args.output_dir / "unit_genweight_normalization.json", normalization)
    write_json(
        args.output_dir / "manifest.json",
        {
            "status": "prepared",
            "purpose": (
                "Compare selection efficiencies and recoil shapes against the "
                "adopted PTLL DY family; no absolute normalization claim."
            ),
            "datasets": manifest_datasets,
            "files_total": sum(item["files"] for item in manifest_datasets.values()),
            "normalization_policy": "unit factor times gen_weight",
            "xsec_status": "not_adopted",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
