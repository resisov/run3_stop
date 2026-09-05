#!/usr/bin/env python3
"""Merge diagonal-v3 CR NN-output partials with complete file accounting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .build_diagonal_v3_cr_nnout_partial import merge as merge_histograms
except ImportError:
    from build_diagonal_v3_cr_nnout_partial import merge as merge_histograms


PARTIAL_SCHEMAS = {
    "gnn_lowdm_diagonal_v3_cr_nnout_partial_v1",
    "gnn_lowdm_diagonal_v3_year_nnout_partial_v1",
    "gnn_lowdm_diagonal_v3_srcr_nnout_partial_v2",
}
MERGED_SCHEMA = "gnn_lowdm_diagonal_v3_cr_nnout_merged_v1"
YEAR_MERGED_SCHEMA = "gnn_lowdm_diagonal_v3_year_nnout_merged_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def add_nested_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            add_nested_counts(target.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials", required=True, type=Path)
    parser.add_argument("--retry-partials", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional full-campaign manifest used for exact input-set validation.",
    )
    args = parser.parse_args()
    paths = sorted(args.partials.glob("*.json"))
    if args.retry_partials:
        paths.extend(sorted(args.retry_partials.glob("*.json")))
    if not paths:
        raise RuntimeError("no diagonal-v3 CR partials found")

    # Campaign directories can retain successful one-file retry partials after
    # the corresponding grouped job is later recovered.  Prefer the grouped
    # result in that case so the same ROOT input is never counted twice.
    primary_valid_inputs: set[str] = set()
    for path in paths:
        if "_retry_" in path.name:
            continue
        payload = json.loads(path.read_text())
        primary_valid_inputs.update(map(str, payload.get("input_files", [])))
    filtered_paths = []
    superseded_retry_partials = []
    for path in paths:
        if "_retry_" not in path.name:
            filtered_paths.append(path)
            continue
        payload = json.loads(path.read_text())
        retry_inputs = set(map(str, payload.get("input_files", [])))
        if retry_inputs and retry_inputs <= primary_valid_inputs:
            superseded_retry_partials.append(str(path))
        else:
            filtered_paths.append(path)
    paths = filtered_paths

    histograms: dict[str, Any] = {}
    requested: set[str] = set()
    valid: set[str] = set()
    failures: dict[str, dict[str, Any]] = {}
    score_edges = selection_contract = checkpoint = histogram_specs = None
    status_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    aggregate_audit: dict[str, Any] = {}
    weight_component_audit: dict[str, dict[str, Any]] = {}
    partial_schema = year = None

    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") not in PARTIAL_SCHEMAS:
            raise RuntimeError(f"unexpected partial schema: {path}")
        if partial_schema is None:
            partial_schema = payload["schema_version"]
            year = payload.get("year")
        elif (payload["schema_version"], payload.get("year")) != (
            partial_schema,
            year,
        ):
            raise RuntimeError(f"inconsistent partial schema/year: {path}")
        contract = (
            payload.get("score_edges"),
            payload.get("selection_contract"),
            payload.get("checkpoint"),
        )
        if score_edges is None:
            score_edges, selection_contract, checkpoint = contract
        elif contract != (score_edges, selection_contract, checkpoint):
            raise RuntimeError(f"inconsistent frozen contract: {path}")
        incoming_specs = payload.get("histogram_specs")
        if histogram_specs is None:
            histogram_specs = incoming_specs
        elif incoming_specs != histogram_specs:
            raise RuntimeError(f"inconsistent histogram specifications: {path}")
        status = str(payload.get("status"))
        kind = str(payload.get("kind"))
        status_counts[status] = status_counts.get(status, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        for source in map(str, payload.get("input_files", [])):
            requested.add(source)
            if source in valid:
                raise RuntimeError(f"duplicate valid input: {source}")
            valid.add(source)
            failures.pop(source, None)
        for record in payload.get("bad_files", []):
            source = str(record["file"])
            requested.add(source)
            if source not in valid:
                failures[source] = record
        for audit in payload.get("source_audits", []):
            add_nested_counts(
                aggregate_audit,
                {
                    "events": audit.get("events", 0),
                    "selected_union_events": audit.get("selected_union_events", 0),
                    "selected_by_region": audit.get("selected_by_region", {}),
                    "feature_reconstruction": audit.get("feature_reconstruction", {}),
                    "data_stream_exclusions": audit.get("data_stream_exclusions", {}),
                    "blinded_sr_data_events": audit.get("blinded_sr_data_events", 0),
                },
            )
            for weight_status in (audit.get("weight_status") or {}).values():
                known_unavailable = set(
                    (weight_status.get("known_unavailable_unity_components") or {})
                )
                for component, component_status in (
                    weight_status.get("components") or {}
                ).items():
                    summary = weight_component_audit.setdefault(
                        component,
                        {
                            "applied_weight_groups": 0,
                            "unapplied_weight_groups": 0,
                            "known_unavailable_unity_weight_groups": 0,
                            "sources": set(),
                            "errors": set(),
                        },
                    )
                    applied = bool(component_status.get("applied"))
                    counter = (
                        "applied_weight_groups"
                        if applied
                        else "unapplied_weight_groups"
                    )
                    summary[counter] += 1
                    if component in known_unavailable:
                        summary["known_unavailable_unity_weight_groups"] += 1
                    if component_status.get("source"):
                        summary["sources"].add(str(component_status["source"]))
                    if component_status.get("error"):
                        summary["errors"].add(str(component_status["error"]))
        merge_histograms(histograms, payload.get("histograms", {}), 5)

    serial_weight_component_audit = {
        component: {
            **{
                key: value
                for key, value in summary.items()
                if key not in {"sources", "errors"}
            },
            "sources": sorted(summary["sources"]),
            "errors": sorted(summary["errors"]),
        }
        for component, summary in sorted(weight_component_audit.items())
    }

    unresolved = [failures[key] for key in sorted(failures) if key not in valid]
    manifest_audit: dict[str, Any] | None = None
    exact_manifest_match = True
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if int(manifest.get("year", -1)) != int(year):
            raise RuntimeError("manifest year does not match partial year")
        expected = {str(record["root"]) for record in manifest.get("shards", [])}
        missing_expected = sorted(expected - valid)
        unexpected_requested = sorted(requested - expected)
        exact_manifest_match = not missing_expected and not unexpected_requested
        source_bad_files = (manifest.get("audit") or {}).get("source_bad_files", [])
        data_bad_files = [
            record
            for record in source_bad_files
            if record.get("process") in {"JetMET", "EGamma", "Muon"}
        ]
        manifest_audit = {
            "manifest": str(args.manifest.resolve()),
            "manifest_status": manifest.get("status"),
            "expected_input_files": len(expected),
            "missing_expected_input_files": missing_expected,
            "unexpected_requested_input_files": unexpected_requested,
            "exact_input_set_match": exact_manifest_match,
            "expected_by_kind": (manifest.get("inventory") or {}).get("by_kind", {}),
            "source_bad_file_count": int(
                (manifest.get("audit") or {}).get("source_bad_file_count", 0)
            ),
            "data_source_bad_file_count": len(data_bad_files),
            "data_luminosity_coverage_complete": not data_bad_files,
        }

    merged = {
        "schema_version": (
            YEAR_MERGED_SCHEMA
            if partial_schema == "gnn_lowdm_diagonal_v3_year_nnout_partial_v1"
            else MERGED_SCHEMA
        ),
        "status": (
            "complete"
            if requested == valid and not unresolved and exact_manifest_match
            else "incomplete"
        ),
        "year": year,
        "checkpoint": checkpoint,
        "selection_contract": selection_contract,
        "score_edges": score_edges,
        "histogram_specs": histogram_specs or {},
        "partials": len(paths),
        "superseded_retry_partials": superseded_retry_partials,
        "partial_status_counts": status_counts,
        "partial_kind_counts": kind_counts,
        "input_files_requested": len(requested),
        "input_files_valid": len(valid),
        "bad_files": unresolved,
        "aggregate_audit": aggregate_audit,
        "weight_component_audit": serial_weight_component_audit,
        "manifest_audit": manifest_audit,
        "histograms": histograms,
    }
    write_json(args.output, merged)
    print(json.dumps({key: merged[key] for key in (
        "status", "partials", "input_files_requested", "input_files_valid", "bad_files"
    )}, indent=2, sort_keys=True))
    return 0 if merged["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
