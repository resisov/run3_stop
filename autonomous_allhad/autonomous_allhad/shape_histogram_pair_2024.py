from __future__ import annotations

import copy
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .shape_histogram_2024_worker import (
    FINAL_SHAPE_NUISANCES,
    FINAL_SHAPE_VARIATIONS,
    OUTPUT_SCHEMA,
    OUTPUT_SECTIONS,
    _merge_dataset_records,
    _merge_hist_tree,
)


PAIR_SCHEMA = "shape_histogram_2024_nuisance_pair_v1"


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
                json.dump(
                    payload,
                    handle,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
        else:
            partial.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_variations(nuisance: str) -> tuple[str, str]:
    if nuisance not in FINAL_SHAPE_NUISANCES:
        raise ValueError(f"unknown adopted nuisance: {nuisance}")
    return f"{nuisance}Up", f"{nuisance}Down"


def validate_single_source_pair(
    histogram: Path,
    metadata: Path,
    nuisance: str,
    source_record_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not histogram.is_file() or histogram.stat().st_size == 0:
        raise RuntimeError(f"pair histogram is missing or empty: {histogram}")
    if not metadata.is_file() or metadata.stat().st_size == 0:
        raise RuntimeError(f"pair metadata is missing or empty: {metadata}")
    payload = read_payload(histogram)
    sidecar = read_payload(metadata)
    pair = list(expected_variations(nuisance))
    summary = sidecar.get("summary") or {}
    errors: list[str] = []
    if payload.get("schema_version") != OUTPUT_SCHEMA:
        errors.append("schema")
    if payload.get("status") != "complete" or sidecar.get("status") != "complete":
        errors.append("status")
    if list(payload.get("variations") or []) != pair:
        errors.append("payload_variations")
    if list(summary.get("variations") or []) != pair:
        errors.append("summary_variations")
    if int(sidecar.get("variation_count") or 0) != 2:
        errors.append("variation_count")
    if set((payload.get("output_policy") or {}).get("sections") or []) != set(
        OUTPUT_SECTIONS
    ):
        errors.append("sections")
    if int(summary.get("files_attempted") or 0) != 1:
        errors.append("files_attempted")
    if int(summary.get("files_processed") or 0) != 1:
        errors.append("files_processed")
    if summary.get("bad_files"):
        errors.append("bad_files")
    observed_digest = str(
        summary.get("source_record_digest")
        or sidecar.get("source_record_digest")
        or ""
    )
    if observed_digest != source_record_digest:
        errors.append("source_record_digest")
    if sha256(histogram) != sidecar.get("histogram_sha256"):
        errors.append("checksum")
    btag = summary.get("btag_sf_status") or {}
    if set(btag) != set(pair) or any(
        not (record or {}).get("applied") for record in btag.values()
    ):
        errors.append("btag")
    if errors:
        raise RuntimeError(
            f"invalid {nuisance} pair output for {source_record_digest}: "
            + ",".join(errors)
        )
    return payload, sidecar


def new_pair_accumulator(template: dict[str, Any], nuisance: str) -> dict[str, Any]:
    pair = list(expected_variations(nuisance))
    if list(template.get("variations") or []) != pair:
        raise RuntimeError(
            f"template variations do not match {nuisance}: "
            f"{template.get('variations')}"
        )
    accumulator = {
        key: copy.deepcopy(value)
        for key, value in template.items()
        if key not in {"datasets", "summary", "status", "schema_version"}
    }
    accumulator.update(
        {
            "schema_version": PAIR_SCHEMA,
            "status": "partial",
            "nuisance": nuisance,
            "variations": pair,
            "datasets": {},
            "summary": {
                "source_record_digests": [],
                "files_attempted": 0,
                "files_processed": 0,
                "events_read": 0,
                "variation_event_evaluations": 0,
                "bad_files": [],
                "btag_sf_status": {},
                "variation_calibration": {},
                "variation_region_counts": {},
            },
        }
    )
    return accumulator


def merge_single_source_pair(
    accumulator: dict[str, Any] | None,
    source: dict[str, Any],
    nuisance: str,
) -> dict[str, Any]:
    pair = list(expected_variations(nuisance))
    if list(source.get("variations") or []) != pair:
        raise RuntimeError(
            f"source variations do not match {nuisance}: {source.get('variations')}"
        )
    if accumulator is None:
        accumulator = new_pair_accumulator(source, nuisance)
    if accumulator.get("nuisance") != nuisance:
        raise RuntimeError("cannot merge different nuisance pairs")
    for key in (
        "recoil_pt_bins",
        "regions",
        "ntop_split_policy",
        "search_bin_schemes",
        "lowdm_region_policy",
        "highdm_distribution_variable_specs",
        "highdm_distribution_regions",
        "lowdm_variable_specs",
        "lowdm_region_variables",
        "output_policy",
    ):
        if accumulator.get(key) != source.get(key):
            raise RuntimeError(f"inconsistent pair histogram definition: {key}")

    source_summary = source.get("summary") or {}
    digest = str(source_summary.get("source_record_digest") or "")
    if not digest:
        raise RuntimeError("single-source pair is missing source_record_digest")
    processed = accumulator["summary"]["source_record_digests"]
    if digest in processed:
        raise RuntimeError(f"duplicate source record in {nuisance}: {digest}")

    _merge_dataset_records(
        accumulator.setdefault("datasets", {}),
        source.get("datasets") or {},
    )
    target_summary = accumulator["summary"]
    processed.append(digest)
    for field in (
        "files_attempted",
        "files_processed",
        "events_read",
        "variation_event_evaluations",
    ):
        target_summary[field] = int(target_summary.get(field) or 0) + int(
            source_summary.get(field) or 0
        )
    target_summary["bad_files"].extend(source_summary.get("bad_files") or [])
    for field in ("btag_sf_status", "variation_calibration"):
        for variation, value in (source_summary.get(field) or {}).items():
            target_summary[field].setdefault(variation, value)
    for variation, by_region in (
        source_summary.get("variation_region_counts") or {}
    ).items():
        merged = target_summary["variation_region_counts"].setdefault(
            variation, {}
        )
        for region, count in by_region.items():
            merged[region] = int(merged.get(region) or 0) + int(count)
    return accumulator


def finalize_pair_accumulator(
    accumulator: dict[str, Any],
    expected_sources: int,
) -> dict[str, Any]:
    result = copy.deepcopy(accumulator)
    summary = result.get("summary") or {}
    digests = list(summary.get("source_record_digests") or [])
    summary["source_record_digests"] = sorted(digests)
    summary["source_record_count"] = len(digests)
    summary["expected_source_records"] = int(expected_sources)
    summary["missing_source_records"] = max(0, int(expected_sources) - len(digests))
    complete = (
        len(digests) == int(expected_sources)
        and int(summary.get("files_processed") or 0) == int(expected_sources)
        and not summary.get("bad_files")
    )
    result["status"] = "complete" if complete else "partial"
    return result


def write_pair_with_sidecar(
    histogram: Path,
    metadata: Path,
    accumulator: dict[str, Any],
) -> None:
    write_payload(histogram, accumulator)
    write_payload(
        metadata,
        {
            "schema_version": f"{PAIR_SCHEMA}_metadata",
            "status": accumulator.get("status"),
            "nuisance": accumulator.get("nuisance"),
            "variations": accumulator.get("variations"),
            "variation_count": len(accumulator.get("variations") or []),
            "histogram_file": str(histogram),
            "histogram_size": histogram.stat().st_size,
            "histogram_sha256": sha256(histogram),
            "summary": accumulator.get("summary"),
        },
    )


def combine_pair_payloads(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(FINAL_SHAPE_NUISANCES):
        raise RuntimeError(
            f"expected 20 nuisance-pair payloads, received {len(paths)}"
        )
    by_nuisance: dict[str, dict[str, Any]] = {}
    reference_digests: set[str] | None = None
    reference_definitions: dict[str, Any] | None = None
    combined_datasets: dict[str, Any] = {}
    combined_btag: dict[str, Any] = {}
    combined_calibration: dict[str, Any] = {}
    combined_region_counts: dict[str, Any] = {}
    total_evaluations = 0
    reference_summary: dict[str, Any] | None = None
    definition_fields = (
        "recoil_pt_bins",
        "regions",
        "ntop_split_policy",
        "search_bin_schemes",
        "lowdm_region_policy",
        "highdm_distribution_variable_specs",
        "highdm_distribution_regions",
        "lowdm_variable_specs",
        "lowdm_region_variables",
        "output_policy",
        "jes_source_policy",
    )

    for path in paths:
        payload = read_payload(path)
        if payload.get("schema_version") != PAIR_SCHEMA:
            raise RuntimeError(f"unexpected pair schema in {path}")
        if payload.get("status") != "complete":
            raise RuntimeError(f"nuisance pair is incomplete: {path}")
        nuisance = str(payload.get("nuisance") or "")
        if nuisance not in FINAL_SHAPE_NUISANCES or nuisance in by_nuisance:
            raise RuntimeError(f"invalid or duplicate nuisance in {path}: {nuisance}")
        if list(payload.get("variations") or []) != list(
            expected_variations(nuisance)
        ):
            raise RuntimeError(f"directional variation mismatch in {path}")
        by_nuisance[nuisance] = payload
        summary = payload.get("summary") or {}
        digests = set(summary.get("source_record_digests") or [])
        if reference_digests is None:
            reference_digests = digests
        elif digests != reference_digests:
            raise RuntimeError(
                f"source coverage differs for nuisance pair {nuisance}"
            )
        definitions = {field: payload.get(field) for field in definition_fields}
        if reference_definitions is None:
            reference_definitions = definitions
        elif definitions != reference_definitions:
            raise RuntimeError(
                f"histogram definitions differ for nuisance pair {nuisance}"
            )
        if reference_summary is None:
            reference_summary = summary
        else:
            for field in (
                "files_attempted",
                "files_processed",
                "events_read",
                "expected_source_records",
                "missing_source_records",
            ):
                if int(summary.get(field) or 0) != int(
                    reference_summary.get(field) or 0
                ):
                    raise RuntimeError(
                        f"summary field {field} differs for {nuisance}"
                    )
        total_evaluations += int(
            summary.get("variation_event_evaluations") or 0
        )
        combined_btag.update(summary.get("btag_sf_status") or {})
        combined_calibration.update(summary.get("variation_calibration") or {})
        combined_region_counts.update(
            summary.get("variation_region_counts") or {}
        )

        incoming_datasets = payload.get("datasets") or {}
        if not combined_datasets:
            combined_datasets = copy.deepcopy(incoming_datasets)
            continue
        if set(incoming_datasets) != set(combined_datasets):
            raise RuntimeError(f"physical dataset coverage differs for {nuisance}")
        for physical, incoming in incoming_datasets.items():
            current = combined_datasets[physical]
            for field in (
                "physical_dataset",
                "process",
                "xsec_pb",
                "dataset_splits",
                "files_processed",
                "events_read",
            ):
                if current.get(field) != incoming.get(field):
                    raise RuntimeError(
                        f"dataset field {field} differs for {physical}/{nuisance}"
                    )
            current["variation_event_evaluations"] = int(
                current.get("variation_event_evaluations") or 0
            ) + int(incoming.get("variation_event_evaluations") or 0)
            for section in OUTPUT_SECTIONS:
                _merge_hist_tree(
                    current.setdefault(section, {}),
                    incoming.get(section) or {},
                )

    if set(by_nuisance) != set(FINAL_SHAPE_NUISANCES):
        missing = sorted(set(FINAL_SHAPE_NUISANCES) - set(by_nuisance))
        raise RuntimeError(f"missing nuisance-pair payloads: {missing}")
    if set(combined_btag) != set(FINAL_SHAPE_VARIATIONS):
        raise RuntimeError("combined btag coverage is not exactly 40 variations")
    reference_summary = reference_summary or {}
    output = copy.deepcopy(next(iter(by_nuisance.values())))
    output["schema_version"] = OUTPUT_SCHEMA
    output["status"] = "complete"
    output.pop("nuisance", None)
    output["shape_nuisances"] = list(FINAL_SHAPE_NUISANCES)
    output["variations"] = list(FINAL_SHAPE_VARIATIONS)
    output["datasets"] = combined_datasets
    output["summary"] = {
        "status": "complete",
        "source_record_digests": sorted(reference_digests or set()),
        "source_record_count": len(reference_digests or set()),
        "expected_source_records": int(
            reference_summary.get("expected_source_records") or 0
        ),
        "missing_source_records": int(
            reference_summary.get("missing_source_records") or 0
        ),
        "files_attempted": int(reference_summary.get("files_attempted") or 0),
        "files_processed": int(reference_summary.get("files_processed") or 0),
        "events_read": int(reference_summary.get("events_read") or 0),
        "variation_event_evaluations": total_evaluations,
        "variations": list(FINAL_SHAPE_VARIATIONS),
        "bad_files": [],
        "btag_sf_status": combined_btag,
        "variation_calibration": combined_calibration,
        "variation_region_counts": combined_region_counts,
        "pair_merge_policy": "20 independently checkpointed Up/Down nuisance payloads",
    }
    return output


def write_combined_with_sidecar(
    histogram: Path,
    metadata: Path,
    payload: dict[str, Any],
) -> None:
    write_payload(histogram, payload)
    write_payload(
        metadata,
        {
            "schema_version": f"{OUTPUT_SCHEMA}_metadata",
            "status": payload.get("status"),
            "variation_count": len(payload.get("variations") or []),
            "histogram_file": str(histogram),
            "histogram_size": histogram.stat().st_size,
            "histogram_sha256": sha256(histogram),
            "summary": payload.get("summary"),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine 20 complete 2024 nuisance-pair histogram payloads."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args(argv)
    payload = combine_pair_payloads([Path(item) for item in args.inputs])
    write_combined_with_sidecar(
        Path(args.output),
        Path(args.metadata_output),
        payload,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source_record_count": payload["summary"][
                    "source_record_count"
                ],
                "variation_count": len(payload["variations"]),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
