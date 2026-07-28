#!/usr/bin/env python3
"""Merge validated flat-histogram chunks without holding all sections in RAM."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from run_flat_hists_chunked import (
    compatible_build_options,
    file_sha256,
    merge_dy_ptll_exclusions,
    merge_exclusions,
    merge_nested_numeric_counts,
    merge_numeric_counts,
    merge_status,
    merge_tree,
    read_json,
    summary_has_strict_warnings,
    write_json,
)


HISTOGRAM_KEYS = (
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
)


def update_summary(
    summary: dict[str, Any],
    payload: dict[str, Any],
    path: Path,
    normalization: Path,
    dy_ptll_policy: str,
    expected_build_options: dict[str, Any] | None,
) -> dict[str, Any]:
    if payload.get("status") != "complete":
        raise RuntimeError(f"{path}: chunk status is not complete")
    src_summary = payload.get("summary") or {}
    if summary_has_strict_warnings(src_summary):
        raise RuntimeError(f"{path}: strict chunk warnings are present")
    recorded_normalization = payload.get("normalization")
    if not recorded_normalization:
        raise RuntimeError(f"{path}: chunk normalization provenance is missing")
    if Path(recorded_normalization).resolve() != normalization.resolve():
        raise RuntimeError(
            f"{path}: chunk normalization does not match requested normalization"
        )
    chunk_policy = str(src_summary.get("dy_ptll_policy", "all"))
    if chunk_policy != dy_ptll_policy:
        raise RuntimeError(
            f"{path}: DY policy {chunk_policy!r} does not match "
            f"{dy_ptll_policy!r}"
        )
    chunk_build_options = src_summary.get("build_options")
    if expected_build_options is None:
        expected_build_options = chunk_build_options
        summary["build_options"] = chunk_build_options
    elif not compatible_build_options(
        chunk_build_options,
        expected_build_options,
    ):
        raise RuntimeError(f"{path}: chunk build options do not match")

    summary["events_processed"] += int(
        src_summary.get("events_processed") or 0
    )
    summary["input_roots"].extend(src_summary.get("input_roots") or [])
    chunk_status = str(payload.get("status") or "missing")
    statuses = summary.setdefault("chunk_statuses", {})
    statuses[chunk_status] = int(statuses.get(chunk_status, 0)) + 1
    if src_summary.get("region_filter"):
        summary["region_filter"] = src_summary["region_filter"]
    if src_summary.get("variable_filter"):
        summary["variable_filter"] = src_summary["variable_filter"]
    for key in (
        "weight_failures",
        "missing_input_roots",
        "missing_sidecars",
        "zero_entry_roots",
    ):
        if src_summary.get(key):
            summary.setdefault(key, []).extend(src_summary.get(key) or [])
    merge_status(
        summary["scale_factor_status"],
        src_summary.get("scale_factor_status") or {},
    )
    merge_exclusions(
        summary.setdefault("data_stream_exclusions", {}),
        src_summary.get("data_stream_exclusions") or {},
    )
    merge_dy_ptll_exclusions(
        summary.setdefault("dy_ptll_dataset_exclusions", {}),
        src_summary.get("dy_ptll_dataset_exclusions") or {},
    )
    merge_numeric_counts(
        summary.setdefault("dy_ptll_prefilter", {}),
        src_summary.get("dy_ptll_prefilter") or {},
    )
    for key in (
        "input_sidecar_schema_versions",
        "electron_eta_sources",
        "weight_rejections",
        "histogram_range_exclusions",
        "histogram_folded_flow",
        "lowdm_search_bin_entry_accounting",
        "scale_factor_status_audit",
        "gcr_prefilter",
        "gcr_photon_selection_audit",
    ):
        merge_nested_numeric_counts(
            summary.setdefault(key, {}),
            src_summary.get(key) or {},
        )
    return expected_build_options


def dump_member(
    handle: Any,
    key: str,
    value: Any,
    first: bool,
) -> bool:
    if not first:
        handle.write(",")
    json.dump(key, handle, ensure_ascii=False)
    handle.write(":")
    json.dump(
        value,
        handle,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dy-ptll-policy", default="all")
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=HISTOGRAM_KEYS,
        default=list(HISTOGRAM_KEYS),
    )
    args = parser.parse_args()

    chunks = sorted(args.chunk_dir.glob("chunk_*.json"))
    if len(chunks) != args.expected_chunks:
        raise SystemExit(
            f"expected {args.expected_chunks} chunks, found {len(chunks)}"
        )
    normalization = args.normalization.resolve()
    first_payload = read_json(chunks[0])
    metadata = {
        key: value
        for key, value in first_payload.items()
        if key
        not in {
            *HISTOGRAM_KEYS,
            "summary",
            "status",
            "normalization",
        }
    }
    del first_payload
    gc.collect()

    summary: dict[str, Any] = {
        "events_processed": 0,
        "input_roots": [],
        "chunk_outputs": [str(path) for path in chunks],
        "scale_factor_status": {},
        "dy_ptll_policy": args.dy_ptll_policy,
        "streaming_merge": True,
    }
    expected_build_options: dict[str, Any] | None = None
    for index, path in enumerate(chunks, start=1):
        payload = read_json(path)
        expected_build_options = update_summary(
            summary,
            payload,
            path,
            normalization,
            args.dy_ptll_policy,
            expected_build_options,
        )
        del payload
        if index % 25 == 0 or index == len(chunks):
            print(
                json.dumps(
                    {
                        "stage": "summary_validation",
                        "chunks": index,
                        "total": len(chunks),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    duplicate_roots = sorted(
        root
        for root, count in Counter(summary["input_roots"]).items()
        if count > 1
    )
    if duplicate_roots:
        raise RuntimeError(
            "duplicate input ROOTs across chunk payloads: "
            + ", ".join(duplicate_roots)
        )
    summary["input_roots"] = sorted(summary["input_roots"])
    status = (
        "complete"
        if set(summary.get("chunk_statuses") or {}) <= {"complete"}
        and not summary_has_strict_warnings(summary)
        else "complete_with_warnings"
    )
    if status != "complete":
        raise RuntimeError(f"strict merged status is {status}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".streaming.tmp")
    with temporary.open("w") as handle:
        handle.write("{")
        first = True
        for key, value in metadata.items():
            first = dump_member(handle, key, value, first)
        for histogram_key in args.sections:
            merged_section: dict[str, Any] = {}
            for index, path in enumerate(chunks, start=1):
                payload = read_json(path)
                merge_tree(
                    merged_section,
                    payload.get(histogram_key) or {},
                )
                del payload
                if index % 25 == 0:
                    gc.collect()
            first = dump_member(
                handle,
                histogram_key,
                merged_section,
                first,
            )
            del merged_section
            gc.collect()
            print(
                json.dumps(
                    {
                        "stage": "section_merged",
                        "section": histogram_key,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        first = dump_member(
            handle,
            "normalization",
            str(normalization),
            first,
        )
        first = dump_member(handle, "summary", summary, first)
        dump_member(handle, "status", status, first)
        handle.write("}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    write_json(
        args.results,
        {
            "status": status,
            "dy_ptll_policy": args.dy_ptll_policy,
            "output": str(args.output),
            "output_sha256": file_sha256(args.output),
            "chunks": [str(path) for path in chunks],
            "streaming_merge": True,
            "sections": list(args.sections),
        },
    )
    print(
        json.dumps(
            {
                "stage": "streaming_merge_done",
                "status": status,
                "events_processed": summary["events_processed"],
                "input_roots": len(summary["input_roots"]),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
