#!/usr/bin/env python3
"""Split a prepared Condor queue into schedd-safe submission parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--max-jobs", type=int, required=True)
    args = parser.parse_args()

    if args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive")

    campaign = args.campaign
    if not campaign.is_absolute():
        raise ValueError("campaign must be an absolute path")
    manifest_path = campaign / "manifest.json"
    condor_dir = campaign / "condor"
    arguments_path = condor_dir / "arguments.txt"
    submit_path = condor_dir / "submit.sub"

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "prepared_not_submitted":
        raise RuntimeError(f"refusing campaign in state {manifest.get('status')!r}")
    if manifest.get("submission", {}).get("clusters"):
        raise RuntimeError("refusing campaign with recorded submitted clusters")

    lines = [line for line in arguments_path.read_text().splitlines() if line.strip()]
    if len(lines) != manifest["jobs"]:
        raise RuntimeError(
            f"arguments/manifest mismatch: {len(lines)} != {manifest['jobs']}"
        )

    submit_text = submit_path.read_text()
    original_queue = f"queue name,shift,shard_name,root_out from {arguments_path}"
    if submit_text.count(original_queue) != 1:
        raise RuntimeError("could not identify the unique queue statement")

    parts = []
    for index, start in enumerate(range(0, len(lines), args.max_jobs)):
        part_lines = lines[start : start + args.max_jobs]
        arguments_part = condor_dir / f"arguments_part{index:03d}.txt"
        submit_part = condor_dir / f"submit_part{index:03d}.sub"
        arguments_part.write_text("\n".join(part_lines) + "\n")
        part_queue = (
            "queue name,shift,shard_name,root_out from "
            f"{arguments_part}"
        )
        submit_part.write_text(submit_text.replace(original_queue, part_queue))
        parts.append(
            {
                "index": index,
                "jobs": len(part_lines),
                "arguments": str(arguments_part),
                "submit_file": str(submit_part),
                "status": "prepared_not_submitted",
            }
        )

    manifest["submission"]["partitioning"] = {
        "reason": "MAX_JOBS_PER_SUBMISSION=20000 on the eossubmit schedd",
        "max_jobs_per_part": args.max_jobs,
        "parts": parts,
        "total_jobs": sum(part["jobs"] for part in parts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "campaign": str(campaign),
                "parts": len(parts),
                "jobs_per_part": [part["jobs"] for part in parts],
                "total_jobs": sum(part["jobs"] for part in parts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
