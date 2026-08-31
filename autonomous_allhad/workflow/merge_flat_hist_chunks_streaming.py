#!/usr/bin/env python3
"""Merge validated flat-histogram chunks without holding all sections in RAM."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from run_flat_hists_chunked import (
    REPAIRABLE_CODE_PATHS,
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
    validate_highdm_control_components,
    validate_highdm_search_bin_components,
    validate_search_bin_payload,
    write_json,
)


HISTOGRAM_KEYS = (
    "histograms",
    "highdm_control_components",
    "search_bin_histograms",
    "highdm_search_bin_components",
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
    allow_hist_builder_repair: bool,
    allow_zero_entry_roots: bool,
) -> dict[str, Any]:
    recorded_status = str(payload.get("status") or "")
    allowed_statuses = {"complete"}
    if allow_zero_entry_roots:
        allowed_statuses.add("complete_with_warnings")
    if recorded_status not in allowed_statuses:
        raise RuntimeError(f"{path}: chunk status is not complete")
    src_summary = payload.get("summary") or {}
    strict_warning_keys = (
        "weight_failures",
        "missing_input_roots",
        "missing_sidecars",
        "weight_rejections",
    )
    if any(bool(src_summary.get(key)) for key in strict_warning_keys):
        raise RuntimeError(f"{path}: strict chunk warnings are present")
    if src_summary.get("zero_entry_roots") and not allow_zero_entry_roots:
        raise RuntimeError(f"{path}: zero-entry ROOT warnings are present")
    if recorded_status == "complete_with_warnings" and not src_summary.get(
        "zero_entry_roots"
    ):
        raise RuntimeError(
            f"{path}: warning status is not explained by zero-entry ROOTs"
        )
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
        allow_hist_builder_repair,
    ):
        raise RuntimeError(f"{path}: chunk build options do not match")
    validate_search_bin_payload(
        payload,
        (chunk_build_options or {}).get("search_bins"),
        require_histogram=False,
    )
    validate_highdm_search_bin_components(
        payload,
        (chunk_build_options or {}).get("search_bins"),
        require_components=False,
    )
    validate_highdm_control_components(payload, require_components=False)

    for code_path in REPAIRABLE_CODE_PATHS:
        code_sha = (chunk_build_options or {}).get("code_sha256", {}).get(code_path)
        if code_sha:
            variants = summary.setdefault("repair_code_sha256_variants", {}).setdefault(
                code_path,
                [],
            )
            if code_sha not in variants:
                variants.append(code_sha)

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
        "highdm_search_bin_entry_accounting",
        "trota_resolved_top_audit",
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


def write_compact_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    control_component_reference: dict[str, Any] | None = None
    search_component_reference: dict[str, Any] | None = None
    with temporary.open("w") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        handle.write("\n")
    os.replace(temporary, path)


def merge_section_partition(
    histogram_key: str,
    chunks: list[str],
    output: str,
) -> str:
    merged_section: dict[str, Any] = {}
    for path_string in chunks:
        payload = read_json(Path(path_string))
        merge_tree(
            merged_section,
            payload.get(histogram_key) or {},
        )
        del payload
    output_path = Path(output)
    write_compact_json(output_path, merged_section)
    return str(output_path)


def split_paths(paths: list[Path], parts: int) -> list[list[Path]]:
    count = max(1, min(int(parts), len(paths)))
    return [paths[index::count] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dy-ptll-policy", default="all")
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--allow-hist-builder-repair", action="store_true")
    parser.add_argument(
        "--allow-zero-entry-roots",
        action="store_true",
        help=(
            "Allow complete_with_warnings chunks only when their sole strict "
            "warning is a recorded zero-entry ROOT."
        ),
    )
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
            args.allow_hist_builder_repair,
            args.allow_zero_entry_roots,
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
    allowed_chunk_statuses = {"complete"}
    if args.allow_zero_entry_roots:
        allowed_chunk_statuses.add("complete_with_warnings")
    status = (
        "complete"
        if set(summary.get("chunk_statuses") or {}) <= allowed_chunk_statuses
        and not summary_has_strict_warnings(
            summary,
            args.allow_zero_entry_roots,
        )
        else "complete_with_warnings"
    )
    if status != "complete":
        raise RuntimeError(f"strict merged status is {status}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    partials_by_section: dict[str, list[Path]] = {}
    partial_work_dir: Path | None = None
    if workers > 1:
        partial_work_dir = (
            args.work_dir.resolve()
            if args.work_dir is not None
            else (args.results.parent / "merge_parts").resolve()
        )
        if partial_work_dir.exists():
            shutil.rmtree(partial_work_dir)
        partial_work_dir.mkdir(parents=True)
        partitions_per_section = max(1, workers // len(args.sections))
        tasks: list[tuple[str, list[Path], Path]] = []
        for histogram_key in args.sections:
            for part_index, partition in enumerate(
                split_paths(chunks, partitions_per_section)
            ):
                tasks.append(
                    (
                        histogram_key,
                        partition,
                        partial_work_dir
                        / f"{histogram_key}.part{part_index:03d}.json",
                    )
                )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks))
        ) as executor:
            future_to_task = {
                executor.submit(
                    merge_section_partition,
                    histogram_key,
                    [str(path) for path in partition],
                    str(partial_path),
                ): (histogram_key, partial_path)
                for histogram_key, partition, partial_path in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                histogram_key, partial_path = future_to_task[future]
                completed_path = Path(future.result())
                partials_by_section.setdefault(histogram_key, []).append(
                    completed_path
                )
                print(
                    json.dumps(
                        {
                            "stage": "section_partition_merged",
                            "section": histogram_key,
                            "output": str(completed_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    temporary = args.output.with_suffix(args.output.suffix + ".streaming.tmp")
    with temporary.open("w") as handle:
        handle.write("{")
        first = True
        for key, value in metadata.items():
            first = dump_member(handle, key, value, first)
        for histogram_key in args.sections:
            merged_section: dict[str, Any] = {}
            section_inputs = (
                sorted(partials_by_section[histogram_key])
                if workers > 1
                else chunks
            )
            for index, path in enumerate(section_inputs, start=1):
                payload = read_json(path)
                merge_tree(merged_section, payload if workers > 1 else payload.get(histogram_key) or {})
                del payload
                if index % 25 == 0:
                    gc.collect()
            if histogram_key == "histograms":
                control_component_reference = {
                    region: merged_section.get(region) or {}
                    for region in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
                }
            if histogram_key == "search_bin_histograms":
                validate_search_bin_payload(
                    {
                        "search_bin_schemes": metadata.get(
                            "search_bin_schemes", {}
                        ),
                        "search_bin_histograms": merged_section,
                        "summary": summary,
                    },
                    (expected_build_options or {}).get("search_bins"),
                    require_histogram=True,
                )
                scheme = str(
                    ((expected_build_options or {}).get("search_bins") or {}).get(
                        "scheme_name", ""
                    )
                )
                if scheme:
                    search_component_reference = merged_section.get(scheme)
            if histogram_key == "highdm_control_components":
                control_contract = (expected_build_options or {}).get(
                    "search_bins"
                )
                validate_highdm_control_components(
                    {
                        "histograms": control_component_reference or {},
                        "highdm_control_components": merged_section,
                    },
                    require_components=bool(control_contract),
                )
            if histogram_key == "highdm_search_bin_components":
                contract = (expected_build_options or {}).get("search_bins")
                scheme = str((contract or {}).get("scheme_name", ""))
                validate_highdm_search_bin_components(
                    {
                        "search_bin_histograms": {
                            scheme: search_component_reference or {}
                        },
                        "highdm_search_bin_components": {
                            scheme: merged_section.get(scheme) or {}
                        },
                    },
                    contract,
                    require_components=True,
                )
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
    if partial_work_dir is not None:
        shutil.rmtree(partial_work_dir)
    write_json(
        args.results,
        {
            "status": status,
            "dy_ptll_policy": args.dy_ptll_policy,
            "output": str(args.output),
            "output_sha256": file_sha256(args.output),
            "chunks": [str(path) for path in chunks],
            "streaming_merge": True,
            "merge_workers": workers,
            "sections": list(args.sections),
            "validated_chunk_count": len(chunks),
            "unique_input_root_count": len(summary["input_roots"]),
            "chunk_statuses": summary.get("chunk_statuses") or {},
            "zero_entry_root_count": len(summary.get("zero_entry_roots") or []),
            "strict_warning_counts": {
                key: len(summary.get(key) or [])
                if isinstance(summary.get(key), list)
                else int(bool(summary.get(key)))
                for key in (
                    "weight_failures",
                    "missing_input_roots",
                    "missing_sidecars",
                    "weight_rejections",
                )
            },
            "search_bin_contract": (
                expected_build_options or {}
            ).get("search_bins"),
            "component_validations": {
                "highdm_control_components": True,
                "highdm_search_bin_components": True,
                "search_bin_histograms": True,
                "finite_histogram_content": True,
            },
            "build_options": expected_build_options,
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
