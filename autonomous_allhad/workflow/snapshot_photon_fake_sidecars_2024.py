#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SIDECAR_SCHEMA = "photon_fake_2024_sidecar_shard_v1"
SNAPSHOT_SCHEMA = "photon_fake_2024_sidecar_snapshot_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze validated, complete photon-fake sidecars from a live campaign."
        )
    )
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    campaign = args.campaign.absolute()
    output_dir = args.output_dir.absolute()
    inputs_dir = output_dir / "inputs"
    manifest_path = output_dir / "snapshot_manifest.json"
    if manifest_path.exists() or inputs_dir.exists():
        raise RuntimeError(f"snapshot destination already exists: {output_dir}")

    metadata_paths = sorted((campaign / "metadata").glob("*/*.json"))
    output_paths = sorted((campaign / "outputs").glob("*/*.json.gz"))
    output_set = set(output_paths)
    paired_outputs: set[Path] = set()
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for metadata_path in metadata_paths:
        process = metadata_path.parent.name
        output_path = (
            campaign
            / "outputs"
            / process
            / f"{metadata_path.name[:-5]}.json.gz"
        )
        reasons: list[str] = []
        metadata: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            reasons.append(f"metadata_read:{type(exc).__name__}:{exc}")
        if metadata and str(metadata.get("status") or "") != "complete":
            reasons.append(f"metadata_status:{metadata.get('status')}")
        if not output_path.is_file():
            reasons.append("output_missing")
        else:
            paired_outputs.add(output_path)
            expected_size = metadata.get("histogram_size")
            if expected_size is not None and output_path.stat().st_size != int(
                expected_size
            ):
                reasons.append(
                    "size_mismatch:"
                    f"{output_path.stat().st_size}!={int(expected_size)}"
                )
            expected_sha = str(metadata.get("histogram_sha256") or "")
            actual_sha = sha256(output_path)
            if not expected_sha:
                reasons.append("metadata_sha256_missing")
            elif actual_sha != expected_sha:
                reasons.append(f"sha256_mismatch:{actual_sha}!={expected_sha}")
            try:
                payload = read_gzip_json(output_path)
            except Exception as exc:
                reasons.append(f"sidecar_read:{type(exc).__name__}:{exc}")
            if payload:
                if payload.get("schema_version") != SIDECAR_SCHEMA:
                    reasons.append(
                        f"sidecar_schema:{payload.get('schema_version')}"
                    )
                if str(payload.get("status") or "") != "complete":
                    reasons.append(f"sidecar_status:{payload.get('status')}")
                metadata_digest = str(metadata.get("source_record_digest") or "")
                payload_digest = str(
                    (payload.get("summary") or {}).get("source_record_digest") or ""
                )
                if metadata_digest != payload_digest:
                    reasons.append(
                        f"source_digest_mismatch:{metadata_digest}!={payload_digest}"
                    )

        record = {
            "process": process,
            "metadata": str(metadata_path),
            "output": str(output_path),
            "reasons": reasons,
        }
        if reasons:
            invalid.append(record)
            continue

        destination = inputs_dir / process / output_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(output_path)
        counts[process] += 1
        valid.append(
            {
                "process": process,
                "metadata": str(metadata_path),
                "source": str(output_path),
                "snapshot": str(destination),
                "size": output_path.stat().st_size,
                "sha256": str(metadata["histogram_sha256"]),
                "source_record_digest": str(
                    metadata.get("source_record_digest") or ""
                ),
            }
        )

    unpaired_outputs = sorted(str(path) for path in output_set - paired_outputs)
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "status": "complete" if not invalid and not unpaired_outputs else "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "output_dir": str(output_dir),
        "policy": (
            "read-only snapshot of sidecars with complete metadata and payload "
            "status, matching size and SHA-256, readable gzip JSON, matching "
            "schema, and matching source-record digest"
        ),
        "metadata_count": len(metadata_paths),
        "campaign_output_count": len(output_paths),
        "valid_sidecar_count": len(valid),
        "valid_counts_by_process": dict(sorted(counts.items())),
        "invalid_sidecar_count": len(invalid),
        "invalid": invalid,
        "unpaired_output_count": len(unpaired_outputs),
        "unpaired_outputs": unpaired_outputs,
        "sidecars": valid,
    }
    write_json(manifest_path, snapshot)
    print(
        json.dumps(
            {
                "status": snapshot["status"],
                "manifest": str(manifest_path),
                "metadata_count": len(metadata_paths),
                "campaign_output_count": len(output_paths),
                "valid_sidecar_count": len(valid),
                "valid_counts_by_process": dict(sorted(counts.items())),
                "invalid_sidecar_count": len(invalid),
                "unpaired_output_count": len(unpaired_outputs),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
