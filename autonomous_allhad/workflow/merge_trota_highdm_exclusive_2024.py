#!/usr/bin/env python3
"""Merge and strictly validate the bounded 2024 TROTA histogram chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import build_flat_boosted_recoil_hists as base
from build_trota_highdm_exclusive_2024 import (
    BASELINE_SCHEME,
    EXCLUSIVE85_SCHEME,
    REQUIRED_COMPONENTS,
    SCHEMA_VERSION as CHUNK_SCHEMA_VERSION,
    TAILMERGED80_SCHEME,
    merge_histograms,
    validate_conservation,
)


SCHEMA_VERSION = "trota_highdm_exclusive_2024_merged_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_noncomment_lines(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    campaign = args.campaign_dir
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    process_scope = str(manifest.get("process_scope") or "all")
    if process_scope not in {"all", "signal"}:
        raise RuntimeError(f"unsupported process scope: {process_scope}")
    expected_roots = read_noncomment_lines(Path(manifest["root_manifest"]))
    argument_rows = read_noncomment_lines(Path(manifest["arguments"]))
    expected_chunks = [Path(row.split()[2]) for row in argument_rows]
    if len(expected_roots) != int(manifest["root_count"]):
        raise RuntimeError("campaign root manifest count drift")
    if len(expected_chunks) != int(manifest["chunk_count"]):
        raise RuntimeError("campaign chunk manifest count drift")

    histograms: dict[str, Any] = {}
    totals: defaultdict[str, int] = defaultdict(int)
    audit: dict[str, dict[str, int | str]] = {}
    completed_roots: list[str] = []
    input_root_sha256: dict[str, str] = {}
    common: dict[str, Any] | None = None
    for index, chunk_path in enumerate(expected_chunks):
        if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
            raise RuntimeError(f"missing nonempty chunk {index}: {chunk_path}")
        chunk = json.loads(chunk_path.read_text())
        if chunk.get("schema_version") != CHUNK_SCHEMA_VERSION or chunk.get("status") != "complete":
            raise RuntimeError(f"invalid chunk status/schema: {chunk_path}")
        roots = list(chunk.get("completed_input_roots") or [])
        if len(roots) != int(chunk.get("files_expected") or -1) or len(roots) != int(chunk.get("files_completed") or -1):
            raise RuntimeError(f"chunk input count mismatch: {chunk_path}")
        if len(set(roots)) != len(roots):
            raise RuntimeError(f"duplicate ROOT inside chunk: {chunk_path}")
        root_hashes = chunk.get("input_root_sha256") or {}
        if set(root_hashes) != set(roots) or any(not value for value in root_hashes.values()):
            raise RuntimeError(f"incomplete ROOT checksum map: {chunk_path}")
        signature = {
            key: chunk.get(key)
            for key in (
                "normalization_sha256", "builder_sha256", "physics_source_sha256",
                "category_source_sha256", "btag_efficiency_sha256",
                "trota_model_sha256", "input_schema", "required_components",
                "analysis_sf_components", "resolved_top_definition", "schemes",
            )
        }
        if common is None:
            common = signature
        elif signature != common:
            raise RuntimeError(f"code/configuration provenance drift: {chunk_path}")
        completed_roots.extend(roots)
        for root, digest in root_hashes.items():
            if root in input_root_sha256:
                raise RuntimeError(f"ROOT processed more than once: {root}")
            input_root_sha256[root] = str(digest)
        for key, value in (chunk.get("totals") or {}).items():
            totals[str(key)] += int(value)
        merge_histograms(histograms, chunk.get("histograms") or {})
        for component, state in (chunk.get("component_audit") or {}).items():
            merged = audit.setdefault(
                component,
                {"applied_events": 0, "failed_events": 0, "source": state.get("source")},
            )
            if merged.get("source") != state.get("source"):
                raise RuntimeError(f"component source drift for {component}: {chunk_path}")
            merged["applied_events"] = int(merged["applied_events"]) + int(state.get("applied_events") or 0)
            merged["failed_events"] = int(merged["failed_events"]) + int(state.get("failed_events") or 0)

    if len(completed_roots) != len(expected_roots):
        raise RuntimeError(f"completed ROOT count {len(completed_roots)} != {len(expected_roots)}")
    if len(set(completed_roots)) != len(completed_roots):
        raise RuntimeError("duplicate ROOT across chunks")
    if set(completed_roots) != set(expected_roots):
        missing = sorted(set(expected_roots) - set(completed_roots))
        extra = sorted(set(completed_roots) - set(expected_roots))
        raise RuntimeError(f"ROOT coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    if int(totals.get("identity_fallback_files") or 0) != 0:
        raise RuntimeError(f"unexpected TROTA identity fallback files: {totals['identity_fallback_files']}")
    for component in REQUIRED_COMPONENTS:
        state = audit.get(component) or {}
        if int(state.get("failed_events") or 0) != 0 or int(state.get("applied_events") or 0) <= 0:
            raise RuntimeError(f"required component audit failed: {component}: {state}")
    for scheme, expected_bins in (
        (BASELINE_SCHEME, 55), (EXCLUSIVE85_SCHEME, 85), (TAILMERGED80_SCHEME, 80),
    ):
        if scheme not in histograms:
            raise RuntimeError(f"missing merged scheme: {scheme}")
        for sample, variations in histograms[scheme].items():
            for variation, record in variations.items():
                if any(len(record[field]) != expected_bins for field in ("sumw", "sumw2", "entries")):
                    raise RuntimeError(f"bin-count mismatch: {scheme}/{sample}/{variation}")
                if not all(np.all(np.isfinite(np.asarray(record[field], dtype=float))) for field in ("sumw", "sumw2")):
                    raise RuntimeError(f"non-finite histogram: {scheme}/{sample}/{variation}")
                if np.any(np.asarray(record["sumw2"], dtype=float) < 0):
                    raise RuntimeError(f"negative sumw2: {scheme}/{sample}/{variation}")
    conservation = validate_conservation(histograms)
    signal_samples = {"T2tt_mStop1000_mLSP1", "T2tt_mStop1200_mLSP1"}
    required_samples = (
        signal_samples
        if process_scope == "signal"
        else {
            "data_obs", "QCD", "Zto2Nu", "WtoLNu", "ST", "TT", "DY", "GJ", "VV",
            *signal_samples,
        }
    )
    missing_samples = sorted(required_samples - set(histograms[BASELINE_SCHEME]))
    if missing_samples:
        raise RuntimeError(f"missing required samples: {missing_samples}")

    normalization = json.loads((campaign / "normalization.json").read_text())
    selected_signal_normalization = {}
    for key in ("mStop1000_mLSP1", "mStop1200_mLSP1"):
        record = (normalization.get("signal_mass_points") or {}).get(key) or {}
        factor = record.get("normalization_factor")
        if factor is None or not math.isfinite(float(factor)) or float(factor) <= 0:
            raise RuntimeError(f"selected signal normalization invalid: {key}: {record}")
        selected_signal_normalization[key] = {
            "normalization_factor": float(factor),
            "normalization_status": record.get("normalization_status"),
            "sumw": record.get("sumw"),
            "xsec_pb": record.get("xsec_pb"),
        }

    assert common is not None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": now(),
        "campaign_manifest": str(manifest_path),
        "campaign_manifest_sha256": sha256(manifest_path),
        "process_scope": process_scope,
        "normalization_sha256": sha256(campaign / "normalization.json"),
        "chunk_count": len(expected_chunks),
        "files_expected": len(expected_roots),
        "files_completed": len(completed_roots),
        "unique_input_roots": len(set(completed_roots)),
        "completed_input_roots": completed_roots,
        "input_root_sha256": input_root_sha256,
        "totals": dict(totals),
        "component_audit": audit,
        "selected_signal_normalization": selected_signal_normalization,
        "provenance": common,
        "conservation": conservation,
        "histograms": histograms,
        "physics_status": "exploratory category proposal; not adopted for datacards",
        "run2_reference": "AN2019_016_v9 Table 16 (Nt, NW, Nres High-dM topology lattice)",
    }
    write_json(args.output, payload)
    output_sha256 = sha256(args.output)
    summary = {
        "schema_version": "trota_highdm_exclusive_2024_summary_v1",
        "status": "complete",
        "process_scope": process_scope,
        "output": str(args.output),
        "output_sha256": output_sha256,
        "chunks": len(expected_chunks),
        "unique_input_roots": len(set(completed_roots)),
        "events": int(totals.get("events") or 0),
        "baseline_population": int(totals.get("eligible_population") or 0),
        "nres_positive_events": int(totals.get("nres_positive_events") or 0),
        "schemes": {BASELINE_SCHEME: 55, EXCLUSIVE85_SCHEME: 85, TAILMERGED80_SCHEME: 80},
        "required_samples": sorted(required_samples),
        "required_components": list(REQUIRED_COMPONENTS),
        "component_failures": 0,
        "conservation_checks": len(conservation),
    }
    write_json(args.summary, summary)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{output_sha256}  {args.output.name}\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
