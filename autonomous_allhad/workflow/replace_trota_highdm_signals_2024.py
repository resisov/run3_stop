#!/usr/bin/env python3
"""Replace only signal samples in a validated TROTA histogram payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SCHEMES = {
    "highdm55_mtb175_inclusive_nres_SR": 55,
    "highdm85_mtb175_exclusive_nres_SR": 85,
    "highdm80_mtb175_exclusive_nres_tailmerged_SR": 80,
}
EXPECTED_SIGNALS = {
    "T2tt_mStop1000_mLSP1",
    "T2tt_mStop1200_mLSP1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_record(sample: str, scheme: str, record: dict[str, Any], bins: int) -> None:
    if not record:
        raise RuntimeError(f"empty sample record: {scheme}/{sample}")
    if "nominal" not in record:
        raise RuntimeError(f"nominal variation missing: {scheme}/{sample}")
    for variation, values in record.items():
        for field in ("sumw", "sumw2", "entries"):
            sequence = list(values.get(field) or [])
            if len(sequence) != bins:
                raise RuntimeError(
                    f"bin count mismatch: {scheme}/{sample}/{variation}/{field}"
                )
            if field != "entries" and any(not math.isfinite(float(value)) for value in sequence):
                raise RuntimeError(
                    f"non-finite content: {scheme}/{sample}/{variation}/{field}"
                )
        if any(float(value) < 0 for value in values["sumw2"]):
            raise RuntimeError(f"negative sumw2: {scheme}/{sample}/{variation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--signal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    signal = json.loads(args.signal.read_text())
    if base.get("status") != "complete":
        raise RuntimeError("base histogram is not complete")
    if signal.get("status") != "complete" or signal.get("process_scope") != "signal":
        raise RuntimeError("signal rebuild is not a complete signal-only payload")
    if int(signal.get("files_completed") or 0) != 517:
        raise RuntimeError("signal rebuild does not cover exactly 517 signal ROOTs")

    base_histograms = base.get("histograms") or {}
    signal_histograms = signal.get("histograms") or {}
    merged_histograms: dict[str, Any] = {}
    replacement_audit: dict[str, Any] = {}
    for scheme, bins in EXPECTED_SCHEMES.items():
        base_samples = base_histograms.get(scheme) or {}
        new_samples = signal_histograms.get(scheme) or {}
        if set(new_samples) != EXPECTED_SIGNALS:
            raise RuntimeError(
                f"unexpected signal samples for {scheme}: {sorted(new_samples)}"
            )
        old_signal_samples = {
            name for name in base_samples if name.startswith(("T2tt_", "T2tb_", "T2bW_"))
        }
        if old_signal_samples != EXPECTED_SIGNALS:
            raise RuntimeError(
                f"unexpected base signal samples for {scheme}: {sorted(old_signal_samples)}"
            )
        combined = {
            name: values
            for name, values in base_samples.items()
            if name not in old_signal_samples
        }
        for sample, record in new_samples.items():
            validate_record(sample, scheme, record, bins)
            combined[sample] = record
        merged_histograms[scheme] = combined
        replacement_audit[scheme] = {
            "removed": sorted(old_signal_samples),
            "inserted": sorted(new_samples),
            "background_and_data_sample_count": len(combined) - len(new_samples),
        }

    output = dict(base)
    output["schema_version"] = "trota_highdm_exclusive_2024_signal_replaced_v1"
    output["histograms"] = merged_histograms
    output["signal_replacement"] = {
        "policy": "background/data copied unchanged; all existing signal samples removed and replaced",
        "base": {"path": str(args.base), "sha256": sha256(args.base)},
        "signal": {
            "path": str(args.signal),
            "sha256": sha256(args.signal),
            "normalization_sha256": signal.get("normalization_sha256"),
            "files_completed": signal.get("files_completed"),
            "samples": sorted(EXPECTED_SIGNALS),
        },
        "schemes": replacement_audit,
    }
    write_json(args.output, output)
    output_sha = sha256(args.output)
    summary = {
        "schema_version": "trota_highdm_exclusive_2024_signal_replacement_summary_v1",
        "status": "complete",
        "output": str(args.output),
        "output_sha256": output_sha,
        "base_sha256": sha256(args.base),
        "signal_sha256": sha256(args.signal),
        "signal_roots": int(signal["files_completed"]),
        "signal_samples": sorted(EXPECTED_SIGNALS),
        "schemes": EXPECTED_SCHEMES,
        "background_data_policy": "copied unchanged from base",
    }
    write_json(args.summary, summary)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{output_sha}  {args.output.name}\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
