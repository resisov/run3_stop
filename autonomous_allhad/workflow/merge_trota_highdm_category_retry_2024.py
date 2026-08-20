#!/usr/bin/env python3
"""Merge a failed-file-only TROTA retry into the full compact study result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, record in source.items():
        output = target.setdefault(key, {"entries": 0, "sumw": 0.0, "sumw2": 0.0})
        output["entries"] = int(output["entries"]) + int(record["entries"])
        output["sumw"] = float(output["sumw"]) + float(record["sumw"])
        output["sumw2"] = float(output["sumw2"]) + float(record["sumw2"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--retry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--study-code", required=True, type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    retry = json.loads(args.retry.read_text())
    failed_roots = sorted(record["input_root"] for record in base.get("failures") or [])
    if not failed_roots:
        raise RuntimeError("base result has no failures to repair")
    if base.get("files_completed") != base.get("files_expected"):
        raise RuntimeError("base scan did not inspect every expected file")
    if retry.get("status") != "complete" or retry.get("failures"):
        raise RuntimeError("retry result is not complete and failure-free")
    if retry.get("files_expected") != len(failed_roots):
        raise RuntimeError("retry file count differs from base failure count")
    if base.get("input_manifest_sha256") != retry.get("input_manifest_sha256"):
        raise RuntimeError("base and retry input manifest hashes differ")
    if base.get("normalization_sha256") != retry.get("normalization_sha256"):
        raise RuntimeError("base and retry normalization hashes differ")
    fallback_files = int((retry.get("totals") or {}).get("identity_fallback_files", 0))
    if fallback_files != len(failed_roots):
        raise RuntimeError(
            "every retried failure must explicitly record use of the identity fallback"
        )

    merge_stats(base["stats"], retry["stats"])
    for key, value in (retry.get("totals") or {}).items():
        base["totals"][key] = int(base["totals"].get(key, 0)) + int(value)
    base["status"] = "complete"
    base["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    base["failures"] = []
    base["recovery"] = {
        "status": "complete",
        "retried_files": failed_roots,
        "reason": "malformed primary TROTA entry basket; recovered with unique run/lumi/event join",
        "identity_fallback_files": fallback_files,
        "base_result_sha256": sha256(args.base),
        "retry_result_sha256": sha256(args.retry),
        "study_code_sha256": sha256(args.study_code),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": base["status"],
                "files": base["files_completed"],
                "retries": len(failed_roots),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
