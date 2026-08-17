#!/usr/bin/env python3
"""Create a photon-fake template shard with one known bad file excluded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--bad-file", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.shard.open() as handle:
        shard = json.load(handle)
    records = list(shard.get("records") or [])
    observed_digest = canonical_digest(records)
    expected_digest = str(shard.get("record_digest") or "")
    if observed_digest != expected_digest:
        raise ValueError(
            f"source shard digest mismatch: {observed_digest} != {expected_digest}"
        )

    removed = [item for item in records if item.get("file_path") == args.bad_file]
    retained = [item for item in records if item.get("file_path") != args.bad_file]
    if len(removed) != 1:
        raise ValueError(f"expected exactly one matching bad file, found {len(removed)}")
    if not retained:
        raise ValueError("bad-file exclusion would leave an empty shard")

    updated = dict(shard)
    updated["records"] = retained
    updated["record_digest"] = canonical_digest(retained)
    write_json_atomic(args.output, updated)

    print(
        json.dumps(
            {
                "bad_file": args.bad_file,
                "original_digest": expected_digest,
                "original_records": len(records),
                "output": str(args.output),
                "retained_digest": updated["record_digest"],
                "retained_records": len(retained),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
