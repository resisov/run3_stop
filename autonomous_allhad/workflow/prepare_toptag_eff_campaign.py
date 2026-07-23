#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "toptag_eff_campaign_v1"
DATA_GROUPS = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def efficiency_key_from_item(item: dict[str, Any]) -> str:
    key = str(item.get("dataset") or item.get("dataset_key") or "unknown")
    return re.sub(r"____\d+_$", "", key)


def record_from_file(item: dict[str, Any], source: Path) -> dict[str, Any] | None:
    process = str(item.get("process") or item.get("process_group") or "unknown")
    dataset = efficiency_key_from_item(item)
    file_path = str(item.get("file_path") or item.get("physical_file_path") or "")
    if not file_path or process in DATA_GROUPS:
        return None
    if item.get("is_data"):
        return None
    if item.get("read_status") not in (None, "success"):
        return None
    return {
        "dataset": dataset,
        "sample_name": dataset,
        "file_path": file_path,
        "process_group": process,
        "year": str(item.get("year") or "2024"),
        "is_data": False,
        "is_background": bool(item.get("is_background", process != "SMS")),
        "is_signal": bool(item.get("is_signal", process == "SMS")),
        "xsec_pb": item.get("xsec_pb"),
        "number_of_entries": item.get("number_of_entries"),
        "source_metadata": str(source),
    }


def collect_records(metadata_dirs: list[Path], include_signals: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, str]] = []
    metadata_files = 0
    metadata_bad = 0
    for directory in metadata_dirs:
        for path in sorted(directory.glob("mc_shard_*.json")):
            metadata_files += 1
            try:
                payload = json.loads(path.read_text())
            except Exception:
                metadata_bad += 1
                continue
            for item in payload.get("files", []):
                rec = record_from_file(item, path)
                if rec is None or (rec["is_signal"] and not include_signals):
                    continue
                old = by_path.get(rec["file_path"])
                if old and old["process_group"] != rec["process_group"]:
                    conflicts.append(
                        {
                            "file_path": rec["file_path"],
                            "first_process": old["process_group"],
                            "second_process": rec["process_group"],
                        }
                    )
                    continue
                by_path.setdefault(rec["file_path"], rec)
    records = sorted(by_path.values(), key=lambda x: (x["dataset"], x["file_path"]))
    return records, {
        "metadata_files_scanned": metadata_files,
        "metadata_files_unreadable": metadata_bad,
        "deduplication_conflicts": conflicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Top-tag efficiency campaign from flat-output metadata")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, action="append", required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--files-per-shard", type=int, default=25)
    parser.add_argument("--include-signals", action="store_true")
    args = parser.parse_args()

    repo = args.repo.absolute()
    campaign = args.campaign_dir.absolute()
    shards_dir = campaign / "shards"
    outputs_dir = campaign / "outputs"
    logs_dir = campaign / "logs"
    for directory in (campaign, shards_dir, outputs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records, audit = collect_records(
        [path.absolute() for path in args.metadata_dir],
        include_signals=args.include_signals,
    )
    if not records:
        raise RuntimeError("no MC file records were found")

    by_process: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    for rec in records:
        by_process[rec["process_group"]] = by_process.get(rec["process_group"], 0) + 1
        by_dataset[rec["dataset"]] = by_dataset.get(rec["dataset"], 0) + 1

    shard_paths: list[Path] = []
    digest = hashlib.sha256("\n".join(x["file_path"] for x in records).encode()).hexdigest()
    for index, start in enumerate(range(0, len(records), args.files_per_shard)):
        subset = records[start : start + args.files_per_shard]
        path = shards_dir / f"toptageff_shard_{index:05d}.json"
        write_json(
            path,
            {
                "schema_version": "toptag_eff_input_shard_v1",
                "shard_id": path.stem,
                "created_at": now(),
                "record_start": start,
                "record_stop": start + len(subset),
                "records": subset,
            },
        )
        shard_paths.append(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "status": "prepared",
        "repo": str(repo),
        "metadata_dirs": [str(path.absolute()) for path in args.metadata_dir],
        "records": len(records),
        "files_per_shard": args.files_per_shard,
        "shards": len(shard_paths),
        "process_file_counts": dict(sorted(by_process.items())),
        "dataset_file_counts": dict(sorted(by_dataset.items())),
        "efficiency_grouping_policy": "dataset key / generator-bin key, matching btageff*.merged",
        "input_digest": digest,
        "audit": audit,
    }
    write_json(campaign / "input_manifest.json", {**manifest, "files": records})
    write_json(campaign / "summary.json", manifest)

    arguments = []
    for path in shard_paths:
        name = path.stem
        arguments.append(
            " ".join(
                [
                    name,
                    str(path),
                    str(outputs_dir / f"{name}.npz"),
                    str(outputs_dir / f"{name}.json"),
                ]
            )
        )
    arguments_path = campaign / "arguments.txt"
    arguments_path.write_text("\n".join(arguments) + "\n")

    wrapper = repo / "autonomous_allhad" / "workflow" / "run_toptag_eff_worker.sh"
    proxy = repo / "analysis" / "proxy" / "x509up_u147757"
    pyenv = repo / "condor" / "py38.tgz"
    submit = f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shard) $(npz_dest) $(json_dest)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {pyenv}, {proxy}
transfer_output_files = $(name).npz, $(name).json
transfer_output_remaps = "$(name).npz=$(npz_dest); $(name).json=$(json_dest)"
output = {logs_dir}/$(name).out
error = {logs_dir}/$(name).err
log = {logs_dir}/campaign.log
request_cpus = 1
request_memory = 3000MB
request_disk = 8000MB
+JobFlavour = "workday"
queue name,shard,npz_dest,json_dest from {arguments_path}
"""
    (campaign / "toptageff.sub").write_text(submit)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
