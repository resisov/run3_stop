#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ROOTS = 5954


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def chunks(values: list[Path], size: int) -> list[list[Path]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--expected-roots", type=int, default=EXPECTED_ROOTS)
    args = parser.parse_args()

    roots = sorted(path for path in args.source_dir.glob("*.root") if path.stat().st_size > 0)
    if len(roots) != args.expected_roots:
        raise RuntimeError(f"expected {args.expected_roots} ROOTs, found {len(roots)}")
    missing_sidecars = [path for path in roots if not path.with_suffix(".json").is_file() or path.with_suffix(".json").stat().st_size == 0]
    if missing_sidecars:
        raise RuntimeError(f"missing sidecars: {missing_sidecars[:10]}")
    if len({path.name for path in roots}) != len(roots):
        raise RuntimeError("duplicate ROOT basenames")

    campaign = args.campaign_dir
    chunk_dir = campaign / "manifests" / "chunks"
    pilot_dir = campaign / "manifests" / "pilot"
    for directory in (chunk_dir, pilot_dir, campaign / "chunks", campaign / "pilot", campaign / "logs", campaign / "condor"):
        directory.mkdir(parents=True, exist_ok=True)

    master = campaign / "manifests" / "roots.txt"
    write_text(master, "".join(f"{path}\n" for path in roots))
    argument_lines = []
    for index, group in enumerate(chunks(roots, args.chunk_size)):
        name = f"chunk_{index:03d}"
        chunk_path = chunk_dir / f"{name}.txt"
        write_text(chunk_path, "".join(f"{path}\n" for path in group))
        argument_lines.append(
            f"{name} {chunk_path} {campaign / 'chunks' / (name + '.json')}\n"
        )
    arguments = campaign / "condor" / "arguments.txt"
    write_text(arguments, "".join(argument_lines))

    by_prefix = {
        "data": [path for path in roots if path.name.startswith("data_")],
        "mc": [path for path in roots if path.name.startswith("mc_")],
        "signal": [path for path in roots if path.name.startswith("signal_")],
    }
    if any(not values for values in by_prefix.values()):
        raise RuntimeError(f"pilot population missing: {by_prefix}")
    pilot_lines = []
    for label, values in by_prefix.items():
        pilot_path = pilot_dir / f"pilot_{label}.txt"
        write_text(pilot_path, f"{values[0]}\n")
        pilot_lines.append(
            f"pilot_{label} {pilot_path} {campaign / 'pilot' / ('pilot_' + label + '.json')}\n"
        )
    pilot_arguments = campaign / "condor" / "pilot_arguments.txt"
    write_text(pilot_arguments, "".join(pilot_lines))

    manifest = {
        "schema_version": "trota_highdm_exclusive_2024_campaign_manifest_v1",
        "status": "prepared",
        "source_dir": str(args.source_dir),
        "campaign_dir": str(campaign),
        "root_count": len(roots),
        "sidecar_count": len(roots),
        "chunk_size": args.chunk_size,
        "chunk_count": len(argument_lines),
        "process_counts": {key: len(values) for key, values in by_prefix.items()},
        "root_manifest": str(master),
        "root_manifest_sha256": sha256(master),
        "arguments": str(arguments),
        "arguments_sha256": sha256(arguments),
        "pilot_arguments": str(pilot_arguments),
        "pilot_arguments_sha256": sha256(pilot_arguments),
    }
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
