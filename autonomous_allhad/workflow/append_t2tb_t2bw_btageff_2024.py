#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from coffea.util import load, save


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def histogram_sum(histogram: Any) -> float:
    values = np.asarray(histogram.values(flow=True), dtype=float)
    return float(np.sum(values[np.isfinite(values)]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    repo = Path(args.repo)
    campaign = Path(args.campaign)
    manifest = json.loads((campaign / "manifest.json").read_text())
    payload_path = repo / "analysis/hists/btageff2024.merged"
    output_dir = repo / "analysis/hists/btageff2024"
    expected_keys: dict[str, str] = {}
    for item in manifest["datasets_detail"]:
        parts = item["dataset"].strip("/").split("/")
        expected_keys[f"{parts[0]}-{parts[1]}"] = item["topology"]
    missing_files = [
        key for key in expected_keys if not (output_dir / f"{key}.futures").exists()
    ]
    if missing_files:
        raise RuntimeError(
            f"missing {len(missing_files)} b-tag efficiency outputs: {missing_files[:5]}"
        )

    payload = load(payload_path)
    target = payload.setdefault("UParTAK4", {})
    collisions = sorted(set(expected_keys) & set(target))
    if collisions:
        raise RuntimeError(
            "refusing to overwrite existing b-tag efficiency keys: "
            + ", ".join(collisions[:10])
        )
    sums: dict[str, float] = {}
    for key in sorted(expected_keys):
        item = load(output_dir / f"{key}.futures")
        histogram = item.get("UParTAK4")
        if histogram is None:
            raise RuntimeError(f"{key}: UParTAK4 histogram missing")
        total = histogram_sum(histogram)
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"{key}: empty or invalid efficiency histogram sum {total}")
        target[key] = histogram
        sums[key] = total

    before = sha256(payload_path)
    backup = campaign / "btageff2024.merged.before_t2tb_t2bw"
    if not backup.exists():
        shutil.copy2(payload_path, backup)
    temporary = payload_path.with_name(
        f"{payload_path.name}.tmp.t2tb_t2bw.{os.getpid()}"
    )
    save(payload, temporary)
    os.replace(temporary, payload_path)
    after = sha256(payload_path)
    reloaded = load(payload_path)
    reloaded_keys = set((reloaded.get("UParTAK4") or {}).keys())
    absent_after = sorted(set(expected_keys) - reloaded_keys)
    if absent_after:
        raise RuntimeError(f"post-write keys missing: {absent_after}")
    audit = {
        "schema": "btageff2024_t2tb_t2bw_append_audit_v1",
        "status": "complete",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "payload": str(payload_path),
        "backup": str(backup),
        "sha256_before": before,
        "sha256_after": after,
        "added_keys": sorted(expected_keys),
        "added_keys_by_topology": {
            topology: sorted(
                key for key, value in expected_keys.items() if value == topology
            )
            for topology in sorted(set(expected_keys.values()))
        },
        "histogram_sums": sums,
        "missing_keys_after": absent_after,
        "fullsim_included": False,
    }
    audit_path = campaign / "btageff_append_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "added_keys": len(expected_keys),
                "sha256_after": after,
                "audit": str(audit_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
