"""Publish only the validated DY-report assets to a static-page directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CHANNELS = ("dy2e", "dy2m")
GROUPS = ("nb1", "nb2plus")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_names(selection: str) -> set[str]:
    names = {"index.html", "summary.json"}
    for stem in (f"rz_{selection}", f"rt_{selection}"):
        names.update({f"{stem}.png", f"{stem}.pdf"})
    for channel in CHANNELS:
        for group in GROUPS:
            stem = f"mll_{selection}_{channel}_{group}"
            for suffix in ("", "_post"):
                names.update({f"{stem}{suffix}.png", f"{stem}{suffix}.pdf"})
    return names


def publish_report(source: Path, destination: Path, selection: str) -> dict[str, str]:
    summary = read_json(source / "summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError(f"{source}: report is incomplete")
    method = summary.get("method") or {}
    if method.get("ut_dependent_rz") is not False:
        raise RuntimeError(f"{source}: only RZ(Nb) may be published")
    required = expected_names(selection)
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise RuntimeError(f"{source}: missing assets: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.iterdir():
        if existing.is_file() and existing.name not in required:
            existing.unlink()
    hashes: dict[str, str] = {}
    for name in sorted(required):
        target = destination / name
        shutil.copy2(source / name, target)
        hashes[name] = sha256(target)
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)

    high_hashes = publish_report(args.high, args.destination / "highdm", "highdm")
    low_hashes = publish_report(args.low, args.destination / "lowdm", "lowdm")
    index = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024 DY normalization</title>
<style>body{font-family:Arial,sans-serif;max-width:900px;margin:48px auto;padding:0 20px}a{display:block;font-size:1.3rem;margin:18px 0}</style>
</head><body><h1>2024 DY normalization</h1>
<p>Run-2-style on/off-Z measurement of R<sub>Z</sub>(N<sub>b</sub>) and R<sub>T</sub>(N<sub>b</sub>).</p>
<a href="highdm/">High-&Delta;m results</a>
<a href="lowdm/">Low-&Delta;m results</a>
</body></html>
"""
    (args.destination / "index.html").write_text(index)
    manifest = {
        "schema_version": "dy_estimation_publication_2024_v1",
        "status": "complete",
        "highdm": high_hashes,
        "lowdm": low_hashes,
    }
    (args.destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "destination": str(args.destination),
                "assets": len(high_hashes) + len(low_hashes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
