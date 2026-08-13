#!/usr/bin/env python3
"""Build one low-pT lepton pass/fail histogram JSON from up to 20 ROOT files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from workflow.reference_trigger_counts import json_safe
from workflow.tnp_histograms import build_histograms


def count_tnp_shard(
    *,
    kind: str,
    shard: dict[str, Any],
    config: dict[str, Any],
    repo: Path,
    step_size: int,
) -> dict[str, Any]:
    records = list(shard.get("records") or [])
    if not records:
        raise ValueError("empty TnP shard")
    if len(records) > 20:
        raise ValueError(f"TnP shard has {len(records)} files; maximum is 20")
    data_files = []
    mc_files = []
    input_records = []
    for record in records:
        file_path = str(record.get("file_path") or "")
        sample = str(record.get("sample") or ("data" if record.get("is_data") else "mc"))
        if sample not in {"data", "mc"}:
            raise ValueError(f"invalid TnP sample {sample!r}")
        if not file_path:
            raise ValueError("empty TnP file_path")
        (data_files if sample == "data" else mc_files).append(file_path)
        input_records.append({
            "file_path": file_path,
            "dataset": record.get("dataset"),
            "sample": sample,
        })
    payload = build_histograms(
        kind=kind,
        data_files=data_files,
        mc_files=mc_files,
        config=config,
        repo=repo,
        step_size=step_size,
    )
    failed = sum(len(item["files_failed"]) for item in payload["processing"].values())
    processed = sum(int(item["files_processed"]) for item in payload["processing"].values())
    payload.update({
        "status": "success" if processed and not failed else ("incomplete" if processed else "failed"),
        "shard_id": shard.get("shard_id"),
        "files_expected": len(records),
        "files_processed": processed,
        "files_failed": failed,
        "input_records": input_records,
        "input_model": "up to 20 NanoAOD ROOT files reduced directly to pass/fail mass histograms",
    })
    return json_safe(payload)


def cli(argv: list[str] | None = None, *, default_kind: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("electron", "muon"), default=default_kind, required=default_kind is None)
    parser.add_argument("--step-size", type=int, default=100_000)
    args = parser.parse_args(argv)
    payload = count_tnp_shard(
        kind=args.kind,
        shard=json.loads(args.shard.read_text()),
        config=json.loads(args.config.read_text()),
        repo=args.repo.resolve(),
        step_size=args.step_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if payload["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(cli())
