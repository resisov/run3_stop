from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .shape_histogram_2024_worker import (
    FINAL_JES_PUBLIC_SOURCES,
    FINAL_SHAPE_NUISANCES,
    FINAL_SHAPE_VARIATIONS,
    OUTPUT_SCHEMA,
)


MERGED_SCHEMA = "shape_histogram_2024_merged_v2"
SECTIONS = (
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
)
ADOPTED_SEARCH_BIN_SCHEMES = {
    "boosted_an17_selected_recoil6_with_nt0_wsplit_SR",
    "cat2_LLCR_lowDeltaM",
    "cat3_QCDCR_lowDeltaM",
    "cat4_GCR_lowDeltaM",
    "cat5_DY2E_lowDeltaM",
    "cat6_DY2M_lowDeltaM",
    "cat7_SR_lowDeltaM",
}
DEFINITION_FIELDS = (
    "recoil_pt_bins",
    "regions",
    "ntop_split_policy",
    "search_bin_schemes",
    "lowdm_region_policy",
    "highdm_distribution_variable_specs",
    "highdm_distribution_regions",
    "lowdm_variable_specs",
    "lowdm_region_variables",
)


def read_payload(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        if path.suffix == ".gz":
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
                handle.write("\n")
        else:
            partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, fill: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fill
    return out if math.isfinite(out) else fill


def hist_leaf(value: Any) -> bool:
    return isinstance(value, dict) and all(name in value for name in ("sumw", "sumw2", "entries"))


def empty_like(source: dict[str, Any]) -> dict[str, Any]:
    nbin = len(source.get("sumw") or [])
    return {"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin}


def merge_scaled_leaf(target: dict[str, Any], source: dict[str, Any], scale: float) -> None:
    nbin = len(source.get("sumw") or [])
    if not target:
        target.update({"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin})
    if len(target.get("sumw") or []) != nbin:
        raise ValueError("incompatible histogram bin counts during shape merge")
    source_sumw = source.get("sumw") or []
    source_sumw2 = source.get("sumw2") or []
    source_entries = source.get("entries") or []
    for index in range(nbin):
        target["sumw"][index] = finite(target["sumw"][index]) + scale * finite(source_sumw[index])
        target["sumw2"][index] = finite(target["sumw2"][index]) + scale * scale * finite(source_sumw2[index])
        target["entries"][index] = int(target["entries"][index]) + int(source_entries[index])


def merge_variations(
    target: dict[str, Any],
    source: dict[str, Any],
    scale: float,
    coverage: dict[str, int],
) -> None:
    for variation, leaf in source.items():
        if not hist_leaf(leaf):
            raise ValueError(f"invalid shape histogram leaf for {variation}")
        merge_scaled_leaf(target.setdefault(variation, {}), leaf, scale)
        coverage[variation] = int(coverage.get(variation, 0)) + 1


def physical_normalization(norm: dict[str, Any], physical_dataset: str) -> tuple[float | None, str]:
    record = (norm.get("physical_datasets") or {}).get(physical_dataset) or {}
    factor = record.get("normalization_factor")
    status = str(record.get("normalization_status") or "missing_physical_dataset")
    if factor is None:
        return None, status
    value = finite(factor, float("nan"))
    return (value, status) if math.isfinite(value) else (None, "nonfinite_normalization_factor")


def sidecar_path(histogram: Path) -> Path:
    name = histogram.name
    if name.endswith(".json.gz"):
        return histogram.with_name(name[:-8] + ".meta.json")
    return histogram.with_suffix(".meta.json")


def verify_histogram_file(path: Path) -> dict[str, Any]:
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"shape histogram metadata sidecar is missing: {sidecar}")
    metadata = read_payload(sidecar)
    expected = str(metadata.get("histogram_sha256") or "")
    actual = sha256(path)
    if expected != actual:
        raise RuntimeError(f"shape histogram checksum mismatch for {path}: {actual} != {expected}")
    if metadata.get("status") not in {"complete", "complete_with_bad_files"}:
        raise RuntimeError(f"shape histogram sidecar is not successful: {sidecar}")
    return metadata


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json.gz")))
        else:
            paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def merge_histograms(
    paths: list[Path],
    normalization: dict[str, Any],
    expected_shards: int | None,
) -> dict[str, Any]:
    merged = {section: {} for section in SECTIONS}
    coverage: dict[str, int] = {}
    blocked: dict[str, Any] = {}
    processed_digests: set[str] = set()
    bad_files: list[dict[str, Any]] = []
    process_datasets: dict[str, list[str]] = {}
    files_processed = 0
    events_read = 0
    definitions: dict[str, Any] | None = None

    for path in paths:
        metadata = verify_histogram_file(path)
        payload = read_payload(path)
        if payload.get("schema_version") != OUTPUT_SCHEMA:
            raise RuntimeError(f"unexpected shape histogram schema in {path}: {payload.get('schema_version')}")
        digest = str((payload.get("summary") or {}).get("source_record_digest") or metadata.get("source_record_digest") or "")
        if not digest:
            raise RuntimeError(f"missing source record digest in {path}")
        if digest in processed_digests:
            raise RuntimeError(f"duplicate source record digest {digest} in {path}")
        processed_digests.add(digest)
        current_definitions = {field: payload.get(field) for field in DEFINITION_FIELDS}
        if definitions is None:
            definitions = current_definitions
        elif current_definitions != definitions:
            raise RuntimeError(f"inconsistent nominal histogram definitions in {path}")
        bad_files.extend((payload.get("summary") or {}).get("bad_files") or [])
        files_processed += int((payload.get("summary") or {}).get("files_processed") or 0)
        events_read += int((payload.get("summary") or {}).get("events_read") or 0)

        for physical_dataset, dataset_record in (payload.get("datasets") or {}).items():
            factor, norm_status = physical_normalization(normalization, physical_dataset)
            if factor is None:
                blocked[physical_dataset] = {
                    "normalization_status": norm_status,
                    "source": str(path),
                }
                continue
            process = str(dataset_record.get("process") or "unknown")
            process_datasets.setdefault(process, []).append(physical_dataset)

            for region, by_sample in (dataset_record.get("histograms") or {}).items():
                source = (by_sample or {}).get(physical_dataset) or {}
                target = merged["histograms"].setdefault(region, {}).setdefault(process, {})
                merge_variations(target, source, factor, coverage)

            for scheme, by_sample in (dataset_record.get("search_bin_histograms") or {}).items():
                source = (by_sample or {}).get(physical_dataset) or {}
                target = merged["search_bin_histograms"].setdefault(scheme, {}).setdefault(process, {})
                merge_variations(target, source, factor, coverage)

            for region, by_variable in (dataset_record.get("lowdm_variable_histograms") or {}).items():
                for variable, by_sample in (by_variable or {}).items():
                    source = (by_sample or {}).get(physical_dataset) or {}
                    target = (
                        merged["lowdm_variable_histograms"]
                        .setdefault(region, {})
                        .setdefault(variable, {})
                        .setdefault(process, {})
                    )
                    merge_variations(target, source, factor, coverage)

            for region, by_variable in (dataset_record.get("highdm_variable_histograms") or {}).items():
                for variable, by_sample in (by_variable or {}).items():
                    source = (by_sample or {}).get(physical_dataset) or {}
                    target = (
                        merged["highdm_variable_histograms"]
                        .setdefault(region, {})
                        .setdefault(variable, {})
                        .setdefault(process, {})
                    )
                    merge_variations(target, source, factor, coverage)

    missing_shards = None if expected_shards is None else max(0, expected_shards - len(processed_digests))
    status = "complete"
    if blocked or bad_files or (missing_shards not in {None, 0}):
        status = "incomplete"
    return {
        "schema_version": MERGED_SCHEMA,
        "status": status,
        "year": 2024,
        "jes_source_policy": {
            "status": "adopted_final_11_sources",
            "public_sources": list(FINAL_JES_PUBLIC_SOURCES),
            "excluded_validation_envelopes": ["jesTotal", "jesRegroupedTotal"],
        },
        "shape_nuisances": list(FINAL_SHAPE_NUISANCES),
        "variations": list(FINAL_SHAPE_VARIATIONS),
        "normalization_source_schema": normalization.get("schema_version"),
        **(definitions or {}),
        "summary": {
            "input_shards": len(processed_digests),
            "expected_shards": expected_shards,
            "missing_shards": missing_shards,
            "files_processed": files_processed,
            "events_read": events_read,
            "blocked_normalization": blocked,
            "bad_files": bad_files,
            "variation_leaf_coverage": dict(sorted(coverage.items())),
            "process_datasets": {
                process: sorted(set(datasets))
                for process, datasets in sorted(process_datasets.items())
            },
        },
        **merged,
    }


def zero_missing_variations(
    nominal_by_sample: dict[str, Any],
    shape_by_sample: dict[str, Any],
    background_processes: set[str],
) -> tuple[int, int]:
    attached = 0
    zeroed = 0
    for process, nominal_variations in nominal_by_sample.items():
        if process not in background_processes or not isinstance(nominal_variations, dict):
            continue
        nominal_leaf = nominal_variations.get("nominal")
        if not hist_leaf(nominal_leaf):
            continue
        source_variations = shape_by_sample.get(process) or {}
        for variation in FINAL_SHAPE_VARIATIONS:
            source = source_variations.get(variation)
            if hist_leaf(source):
                nominal_variations[variation] = source
                attached += 1
            else:
                nominal_variations[variation] = empty_like(nominal_leaf)
                zeroed += 1
    return attached, zeroed


def attach_to_nominal(nominal: dict[str, Any], shapes: dict[str, Any]) -> dict[str, Any]:
    background_processes = set((shapes.get("summary") or {}).get("process_datasets") or {})
    attached = 0
    zeroed = 0

    for region, nominal_by_sample in (nominal.get("histograms") or {}).items():
        shape_by_sample = (shapes.get("histograms") or {}).get(region) or {}
        a, z = zero_missing_variations(nominal_by_sample, shape_by_sample, background_processes)
        attached += a
        zeroed += z

    for scheme, nominal_by_sample in (nominal.get("search_bin_histograms") or {}).items():
        if scheme not in ADOPTED_SEARCH_BIN_SCHEMES:
            continue
        shape_by_sample = (shapes.get("search_bin_histograms") or {}).get(scheme) or {}
        a, z = zero_missing_variations(nominal_by_sample, shape_by_sample, background_processes)
        attached += a
        zeroed += z

    for region, nominal_by_variable in (nominal.get("lowdm_variable_histograms") or {}).items():
        shape_by_variable = (shapes.get("lowdm_variable_histograms") or {}).get(region) or {}
        for variable, nominal_by_sample in (nominal_by_variable or {}).items():
            shape_by_sample = (shape_by_variable or {}).get(variable) or {}
            a, z = zero_missing_variations(nominal_by_sample, shape_by_sample, background_processes)
            attached += a
            zeroed += z

    for region, nominal_by_variable in (nominal.get("highdm_variable_histograms") or {}).items():
        shape_by_variable = (shapes.get("highdm_variable_histograms") or {}).get(region) or {}
        for variable, nominal_by_sample in (nominal_by_variable or {}).items():
            shape_by_sample = (shape_by_variable or {}).get(variable) or {}
            a, z = zero_missing_variations(nominal_by_sample, shape_by_sample, background_processes)
            attached += a
            zeroed += z

    nominal.setdefault("summary", {})["shape_histogram_2024"] = {
        "status": shapes.get("status"),
        "schema_version": shapes.get("schema_version"),
        "jes_source_policy": shapes.get("jes_source_policy"),
        "shape_nuisances": shapes.get("shape_nuisances"),
        "attached_variation_leaves": attached,
        "zero_filled_variation_leaves": zeroed,
        "signal_shape_status": "not_included_fastsim_deferred_by_user",
    }
    return nominal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify, normalize, and merge compact 2024 shape histogram shards."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--nominal-hists")
    parser.add_argument("--attached-output")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs)
    if not paths:
        raise RuntimeError("no shape histogram shard files found")
    normalization = read_payload(Path(args.normalization))
    if normalization.get("schema_version") != "flat_ntuple_campaign_normalization_v1":
        raise RuntimeError(f"unexpected normalization schema: {normalization.get('schema_version')}")
    merged = merge_histograms(paths, normalization, args.expected_shards)
    write_payload(Path(args.output), merged)

    attached_output = None
    if args.nominal_hists:
        if not args.attached_output:
            raise ValueError("--attached-output is required with --nominal-hists")
        nominal = read_payload(Path(args.nominal_hists))
        combined = attach_to_nominal(nominal, merged)
        attached_output = Path(args.attached_output)
        write_payload(attached_output, combined)

    print(
        json.dumps(
            {
                "status": merged["status"],
                "input_shards": merged["summary"]["input_shards"],
                "missing_shards": merged["summary"]["missing_shards"],
                "blocked_normalization": len(merged["summary"]["blocked_normalization"]),
                "bad_files": len(merged["summary"]["bad_files"]),
                "output": args.output,
                "attached_output": str(attached_output) if attached_output else None,
            },
            sort_keys=True,
        )
    )
    return 0 if merged["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
