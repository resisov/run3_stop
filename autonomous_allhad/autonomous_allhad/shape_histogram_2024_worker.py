from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from . import flat_ntuple_worker as flat
from . import intermediate_2024_worker as intermediate
from . import real_subset_worker as baseline
from .analysis_scale_factors import REQUIRED_ANALYSIS_SF_COMPONENTS
from .object_corrections_2024 import (
    JES_SOURCES,
    branch_audit,
    iter_calibrated_variations,
    validate_payloads,
)


FINAL_JES_PUBLIC_SOURCES = (
    "jesFlavorQCD",
    "jesRelativeBal",
    "jesHF",
    "jesBBEC1",
    "jesEC2",
    "jesAbsolute",
    "jesAbsolute2024",
    "jesHF2024",
    "jesEC22024",
    "jesRelativeSample2024",
    "jesBBEC12024",
)
FINAL_JES_CORRECTION_SOURCES = tuple(JES_SOURCES[name] for name in FINAL_JES_PUBLIC_SOURCES)
NON_JES_SHAPE_NUISANCES = (
    "jer",
    "metUnclustered",
    "electronScale",
    "electronSmear",
    "photonScale",
    "photonSmear",
    "muonScale",
    "muonResolution",
    "tauEnergyScale",
)
FINAL_SHAPE_NUISANCES = FINAL_JES_PUBLIC_SOURCES + NON_JES_SHAPE_NUISANCES
FINAL_SHAPE_VARIATIONS = tuple(
    f"{name}{direction}"
    for name in FINAL_SHAPE_NUISANCES
    for direction in ("Up", "Down")
)
VARIATION_GROUPS = {
    "all": FINAL_SHAPE_VARIATIONS,
    "jes_final11": tuple(
        f"{name}{direction}"
        for name in FINAL_JES_PUBLIC_SOURCES
        for direction in ("Up", "Down")
    ),
    "jer_met": ("jerUp", "jerDown", "metUnclusteredUp", "metUnclusteredDown"),
    "egamma": (
        "electronScaleUp",
        "electronScaleDown",
        "electronSmearUp",
        "electronSmearDown",
        "photonScaleUp",
        "photonScaleDown",
        "photonSmearUp",
        "photonSmearDown",
    ),
    "muon_tau": (
        "muonScaleUp",
        "muonScaleDown",
        "muonResolutionUp",
        "muonResolutionDown",
        "tauEnergyScaleUp",
        "tauEnergyScaleDown",
    ),
}
VARIATION_GROUPS.update(
    {
        nuisance: (f"{nuisance}Up", f"{nuisance}Down")
        for nuisance in FINAL_SHAPE_NUISANCES
    }
)
OUTPUT_SCHEMA = "shape_histogram_2024_shard_v4_streamed_parallel_fullselection"
OUTPUT_SECTIONS = (
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_histogram_builder() -> Any:
    path = Path(__file__).resolve().parents[1] / "workflow" / "build_flat_boosted_recoil_hists.py"
    if not path.is_file():
        raise FileNotFoundError(f"current histogram builder is missing: {path}")
    spec = importlib.util.spec_from_file_location("_current_flat_histogram_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load histogram builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_dataset(dataset: str) -> str:
    return str(dataset or "unknown").split("____", 1)[0]


def rows_to_chunk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    array = ak.Array(rows)
    return {name: array[name] for name in ak.fields(array)}


def new_dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_dataset": physical_dataset(str(record.get("dataset") or "")),
        "process": str(record.get("process_group") or "unknown"),
        "xsec_pb": record.get("xsec_pb"),
        "dataset_splits": [],
        "files_processed": 0,
        "events_read": 0,
        "variation_event_evaluations": 0,
        "histograms": {},
        "search_bin_histograms": {},
        "lowdm_variable_histograms": {},
        "highdm_variable_histograms": {},
    }


def register_dataset_record(target: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "unknown")
    key = physical_dataset(dataset)
    current = target.setdefault(key, new_dataset_record(record))
    if str(current.get("process")) != str(record.get("process_group") or "unknown"):
        raise RuntimeError(f"process conflict for {key}: {current.get('process')} vs {record.get('process_group')}")
    old_xsec = current.get("xsec_pb")
    new_xsec = record.get("xsec_pb")
    if old_xsec is not None and new_xsec is not None and abs(float(old_xsec) - float(new_xsec)) > 1.0e-12:
        raise RuntimeError(f"xsec conflict for {key}: {old_xsec} vs {new_xsec}")
    if dataset not in current["dataset_splits"]:
        current["dataset_splits"].append(dataset)
    return current


def fill_shape_histograms(
    builder: Any,
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    variation: str,
    record: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    if not rows:
        return
    chunk = rows_to_chunk(rows)
    n = len(rows)
    weights = np.asarray([float(row.get("nominal_weight") or 0.0) for row in rows], dtype=float)
    variations = {variation: weights}
    normv = np.ones(n, dtype=float)
    label = physical_dataset(str(record.get("dataset") or "unknown"))
    process = str(record.get("process_group") or "unknown")

    for region, (flag, variable) in builder.REGION_VARIABLES.items():
        mask = builder.region_mask(chunk, region, flag, n)
        values = builder.finite_array(chunk[variable], n, 0.0)
        leaf = (
            target["histograms"]
            .setdefault(region, {})
            .setdefault(label, {})
            .setdefault(variation, builder.empty_hist())
        )
        builder.add_hist(leaf, values, weights, mask)

    builder.fill_highdm_distribution_histograms(
        chunk,
        variations,
        normv,
        label,
        process,
        False,
        target["highdm_variable_histograms"],
        summary,
    )

    sr_mask = builder.as_bool(chunk["feature_SR"], n)
    indices54 = builder.selected_an17_recoil54_indices(chunk, n, sr_mask)
    leaf54 = (
        target["search_bin_histograms"]
        .setdefault(builder.SELECTED_RECOIL54_SCHEME, {})
        .setdefault(label, {})
        .setdefault(variation, builder.empty_index_hist(len(builder.selected_an17_recoil54_labels())))
    )
    builder.add_index_hist(leaf54, indices54, weights)
    indices60 = builder.selected_an17_recoil60_indices(chunk, n, sr_mask)
    leaf60 = (
        target["search_bin_histograms"]
        .setdefault(builder.EXTENDED_RECOIL60_SCHEME, {})
        .setdefault(label, {})
        .setdefault(
            variation,
            builder.empty_index_hist(len(builder.selected_an17_recoil60_labels())),
        )
    )
    builder.add_index_hist(leaf60, indices60, weights)

    for lowdm_region, lowdm_channel in builder.LOWDM_REGION_MAP.items():
        mask = builder.lowdm_region_mask(chunk, lowdm_region, n)
        if not np.any(mask):
            continue
        indices = builder.int_field(
            chunk,
            f"lowdm_search_bin_{lowdm_region}",
            n,
            -1,
        )
        indices = builder.lowdm_nbge1_indices(np.where(mask, indices, -1))
        leaf = (
            target["search_bin_histograms"]
            .setdefault(lowdm_channel, {})
            .setdefault(label, {})
            .setdefault(variation, builder.empty_index_hist(len(builder.LOWDM_34BIN_LABELS)))
        )
        builder.add_index_hist(leaf, indices, weights)
        lowdm_values = {
            variable: builder.lowdm_variable_values(
                chunk,
                builder.LOWDM_VARIABLE_SPECS[variable],
                n,
            )
            for variable in builder.LOWDM_REGION_VARIABLES.get(lowdm_region, [])
        }
        for variable, values in lowdm_values.items():
            spec = builder.LOWDM_VARIABLE_SPECS[variable]
            variable_leaf = (
                target["lowdm_variable_histograms"]
                .setdefault(lowdm_channel, {})
                .setdefault(variable, {})
                .setdefault(label, {})
                .setdefault(variation, builder.empty_binned_hist(spec["bins"]))
            )
            builder.add_binned_hist(variable_leaf, values, weights, mask, spec["bins"])


def required_read_branches(tree: Any) -> list[str]:
    present = set(tree.keys())
    genmodel = [name for name in present if str(name).startswith("GenModel_T2tt_")]
    requested = set(
        flat.CORE_BRANCHES
        + baseline.FILTERS
        + baseline.SIGNAL_HLT
        + baseline.PHOTON_HLT
        + baseline.ELECTRON_HLT
        + baseline.MUON_HLT
        + genmodel
    )
    return sorted(requested & present)


def weight_fallbacks(chunk_summary: dict[str, Any]) -> dict[str, Any]:
    status = chunk_summary.get("scale_factor_status") or {}
    return {
        name: item
        for name, item in (status.get("components") or {}).items()
        if not item.get("applied") and not str(item.get("source") or "").startswith("unity_fallback_not_")
    }


def record_entry_bounds(record: dict[str, Any], tree_entries: int) -> tuple[int, int]:
    """Return the validated half-open Events range assigned to one record."""
    entry_start = int(record.get("entry_start", 0))
    entry_stop = int(record.get("entry_stop", tree_entries))
    if entry_start < 0 or entry_stop < entry_start or entry_stop > int(tree_entries):
        raise RuntimeError(
            "invalid Events entry range "
            f"[{entry_start}, {entry_stop}) for tree with {tree_entries} entries"
        )
    expected_events = record.get("segment_events")
    if expected_events is not None and int(expected_events) != entry_stop - entry_start:
        raise RuntimeError(
            "segment event count does not match its entry range: "
            f"{expected_events} vs {entry_stop - entry_start}"
        )
    return entry_start, entry_stop


def process_record(
    builder: Any,
    record: dict[str, Any],
    variations: tuple[str, ...],
    datasets: dict[str, Any],
    summary: dict[str, Any],
    chunk_size: int,
    max_chunks_per_file: int | None,
) -> bool:
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process_group") or "unknown")
    file_path = str(record.get("file_path") or "")
    if record.get("is_data") or not record.get("is_background"):
        raise RuntimeError(f"shape histogram campaign accepts background MC only: {dataset}")
    target = register_dataset_record(datasets, record)
    file_id = flat.stable_id(file_path)
    root_file = None
    access_info: dict[str, Any] = {}
    start_time = time.time()
    try:
        root_file, access_info = baseline.open_root_with_xrd_fallback(file_path, timeout=60)
        keys = flat.split_keys(root_file)
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        tree = root_file["Events"]
        branches = required_read_branches(tree)
        assigned_start, assigned_stop = record_entry_bounds(
            record,
            int(tree.num_entries),
        )
        chunks_seen = 0
        file_events = 0
        file_evaluations = 0
        for entry_start in range(assigned_start, assigned_stop, chunk_size):
            if max_chunks_per_file is not None and chunks_seen >= max_chunks_per_file:
                break
            entry_stop = min(entry_start + chunk_size, assigned_stop)
            arrays = tree.arrays(branches, entry_start=entry_start, entry_stop=entry_stop, library="ak")
            raw_entries = len(arrays)
            file_events += raw_entries
            validation_context = {
                "object_branch_audit": branch_audit(set(arrays.fields), is_data=False),
                "payload_status": summary["payload_status"],
            }
            if validation_context["object_branch_audit"]["status"] != "valid":
                raise RuntimeError(
                    "required 2024 object branches missing: "
                    f"{validation_context['object_branch_audit']['missing_required']}"
                )
            for variation, precalibrated in iter_calibrated_variations(
                arrays,
                is_data=False,
                shifts=variations,
                root=Path.cwd(),
            ):
                rows, chunk_summary = intermediate.extract_chunk_2024(
                    arrays,
                    dataset,
                    process,
                    None,
                    str(record.get("year") or "2024"),
                    file_path,
                    entry_start,
                    entry_stop,
                    fastsim_trigger_bypass=False,
                    shift_name=variation,
                    compute_weights=True,
                    decorate_rows=False,
                    materialize_skim_flag="feature_flat_preselection",
                    precalibrated=precalibrated,
                    validation_context=validation_context,
                )
                fallback = weight_fallbacks(chunk_summary)
                btag = ((chunk_summary.get("scale_factor_status") or {}).get("components") or {}).get("btagSF") or {}
                if not btag.get("applied"):
                    raise RuntimeError(f"btagSF unavailable for {dataset}: {btag}")
                for component in REQUIRED_ANALYSIS_SF_COMPONENTS:
                    component_status = (
                        (chunk_summary.get("scale_factor_status") or {})
                        .get("components", {})
                        .get(component, {})
                    )
                    if not component_status.get("applied"):
                        raise RuntimeError(
                            f"required analysis SF {component} unavailable for "
                            f"{dataset}: {component_status}"
                        )
                summary.setdefault("btag_sf_status", {}).setdefault(variation, btag)
                if fallback:
                    summary.setdefault("weight_fallbacks", {}).setdefault(variation, {}).update(fallback)
                rows = [flat.decorate_row(row, record, file_id) for row in rows]
                fill_shape_histograms(builder, target, rows, variation, record, summary)
                file_evaluations += raw_entries
                summary.setdefault("variation_region_counts", {}).setdefault(variation, {})
                for region, count in (chunk_summary.get("regions") or {}).items():
                    by_region = summary["variation_region_counts"][variation]
                    by_region[region] = int(by_region.get(region, 0)) + int(count)
                summary.setdefault("variation_calibration", {}).setdefault(
                    variation, chunk_summary.get("object_corrections_2024")
                )
            chunks_seen += 1
        if (
            max_chunks_per_file is None
            and file_events != assigned_stop - assigned_start
        ):
            raise RuntimeError(
                "processed Events entries do not match the assigned range: "
                f"{file_events} vs {assigned_stop - assigned_start}"
            )
        target["files_processed"] += 1
        target["events_read"] += file_events
        target["variation_event_evaluations"] += file_evaluations
        summary["files_processed"] += 1
        summary["events_read"] += file_events
        summary["variation_event_evaluations"] += file_evaluations
        summary.setdefault("file_records", []).append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "source_record_digest": record.get("source_record_digest"),
                "segment_id": record.get("segment_id"),
                "entry_start": assigned_start,
                "entry_stop": assigned_stop,
                "tree_entries": int(tree.num_entries),
                "events_read": file_events,
                "variation_event_evaluations": file_evaluations,
                "wall_time_s": round(time.time() - start_time, 3),
                "access": access_info,
                "status": "complete",
            }
        )
        return True
    except Exception as exc:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary.setdefault("bad_files", []).append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "failure_stage": "shape_histogram_open_read_or_evaluate",
                "exception_type": type(exc).__name__,
                "concise_error": str(exc)[:500],
                "first_failure_time": now,
                "last_failure_time": now,
                "alternate_access_attempted": bool(access_info.get("alternate_access_attempted")),
                "permanently_skipped": False,
            }
        )
        return False
    finally:
        try:
            if root_file is not None:
                root_file.close()
        except Exception:
            pass
        baseline.cleanup_xrd_cache(access_info)


def parse_variations(group: str, explicit: list[str] | None) -> tuple[str, ...]:
    selected = tuple(explicit) if explicit else tuple(VARIATION_GROUPS[group])
    unknown = sorted(set(selected) - set(FINAL_SHAPE_VARIATIONS))
    if unknown:
        raise ValueError(f"variations are outside the adopted final set: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate shape variations requested")
    return selected


def _hist_leaf(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(name in value for name in ("sumw", "sumw2", "entries"))
    )


def _merge_hist_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if _hist_leaf(value):
            leaf = target.setdefault(
                key,
                {
                    "sumw": [0.0] * len(value["sumw"]),
                    "sumw2": [0.0] * len(value["sumw2"]),
                    "entries": [0] * len(value["entries"]),
                },
            )
            if len(leaf["sumw"]) != len(value["sumw"]):
                raise RuntimeError(f"incompatible histogram bin count while merging {key}")
            for index in range(len(value["sumw"])):
                leaf["sumw"][index] += float(value["sumw"][index])
                leaf["sumw2"][index] += float(value["sumw2"][index])
                leaf["entries"][index] += int(value["entries"][index])
        elif isinstance(value, dict):
            _merge_hist_tree(target.setdefault(key, {}), value)
        else:
            raise RuntimeError(f"unexpected non-histogram value while merging key {key}")


def _merge_dataset_records(target: dict[str, Any], source: dict[str, Any]) -> None:
    for physical, incoming in source.items():
        current = target.get(physical)
        if current is None:
            target[physical] = incoming
            continue
        for field in ("physical_dataset", "process", "xsec_pb"):
            if current.get(field) != incoming.get(field):
                raise RuntimeError(
                    f"dataset metadata conflict for {physical} field {field}: "
                    f"{current.get(field)} vs {incoming.get(field)}"
                )
        current["dataset_splits"] = sorted(
            set(current.get("dataset_splits") or [])
            | set(incoming.get("dataset_splits") or [])
        )
        for field in ("files_processed", "events_read", "variation_event_evaluations"):
            current[field] = int(current.get(field) or 0) + int(incoming.get(field) or 0)
        for section in OUTPUT_SECTIONS:
            _merge_hist_tree(
                current.setdefault(section, {}),
                incoming.get(section) or {},
            )


def _merge_record_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ("files_processed", "events_read", "variation_event_evaluations"):
        target[field] = int(target.get(field) or 0) + int(source.get(field) or 0)
    target.setdefault("bad_files", []).extend(source.get("bad_files") or [])
    target.setdefault("file_records", []).extend(source.get("file_records") or [])
    for field in ("btag_sf_status", "variation_calibration"):
        for variation, value in (source.get(field) or {}).items():
            target.setdefault(field, {}).setdefault(variation, value)
    for variation, by_name in (source.get("weight_fallbacks") or {}).items():
        target.setdefault("weight_fallbacks", {}).setdefault(variation, {}).update(by_name)
    for variation, by_region in (source.get("variation_region_counts") or {}).items():
        merged = target.setdefault("variation_region_counts", {}).setdefault(variation, {})
        for region, count in by_region.items():
            merged[region] = int(merged.get(region) or 0) + int(count)


def _process_record_isolated(
    record_index: int,
    record: dict[str, Any],
    variations: tuple[str, ...],
    chunk_size: int,
    max_chunks_per_file: int | None,
) -> dict[str, Any]:
    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload_status = validate_payloads(Path.cwd())
    if payload_status.get("status") != "valid":
        raise RuntimeError(
            f"2024 correction payload validation failed: {payload_status.get('errors')}"
        )
    datasets: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "files_processed": 0,
        "events_read": 0,
        "variation_event_evaluations": 0,
        "payload_status": payload_status,
        "bad_files": [],
    }
    process_record(
        load_histogram_builder(),
        record,
        variations,
        datasets,
        summary,
        chunk_size,
        max_chunks_per_file,
    )
    return {
        "record_index": record_index,
        "datasets": datasets,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill the nominal 2024 histogram schema for object-shape variations directly from background NanoAOD."
    )
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--variation-group", choices=sorted(VARIATION_GROUPS), default="all")
    parser.add_argument("--variations", nargs="+")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--record-workers", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-chunks-per-file", type=int)
    args = parser.parse_args(argv)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.record_workers <= 0:
        raise ValueError("--record-workers must be positive")
    variations = parse_variations(args.variation_group, args.variations)
    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload_status = validate_payloads(Path.cwd())
    if payload_status.get("status") != "valid":
        raise RuntimeError(f"2024 correction payload validation failed: {payload_status.get('errors')}")

    builder = load_histogram_builder()
    shard_path = Path(args.shard)
    shard = read_json(shard_path)
    records = list(shard.get("records") or [])
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    if not records:
        raise RuntimeError("shape histogram shard has no records")

    start = time.time()
    summary: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_shard": str(shard_path),
        "source_record_digest": shard.get("record_digest"),
        "files_attempted": len(records),
        "segments_attempted": len(records),
        "physical_files_attempted": len(
            {str(record.get("file_path") or "") for record in records}
        ),
        "files_processed": 0,
        "events_read": 0,
        "variation_event_evaluations": 0,
        "variations": list(variations),
        "single_read_policy": "each NanoAOD chunk is read once; nominal correction, branch audit, and unaffected object-family corrections are cached; shifted chunks are streamed one at a time while all selections and x-axis values are reevaluated",
        "payload_status": payload_status,
        "record_workers": min(args.record_workers, len(records)),
        "normalization_policy": "raw genWeight times nominal central SF; xsec*lumi/sumw is applied only after nominal metadata merge",
        "bad_files": [],
    }
    datasets: dict[str, Any] = {}
    workers = min(args.record_workers, len(records))
    if workers == 1:
        for record in records:
            process_record(
                builder,
                record,
                variations,
                datasets,
                summary,
                args.chunk_size,
                args.max_chunks_per_file,
            )
    else:
        results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_record_isolated,
                    index,
                    record,
                    variations,
                    args.chunk_size,
                    args.max_chunks_per_file,
                ): (index, record)
                for index, record in enumerate(records)
            }
            for future in concurrent.futures.as_completed(futures):
                index, record = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    summary["bad_files"].append(
                        {
                            "dataset": str(record.get("dataset") or "unknown"),
                            "file_path": str(record.get("file_path") or ""),
                            "failure_stage": "parallel_record_worker",
                            "exception_type": type(exc).__name__,
                            "concise_error": str(exc)[:500],
                            "permanently_skipped": False,
                        }
                    )
                    results.append(
                        {
                            "record_index": index,
                            "datasets": {},
                            "summary": {
                                "files_processed": 0,
                                "events_read": 0,
                                "variation_event_evaluations": 0,
                                "bad_files": [],
                            },
                        }
                    )
        for result in sorted(results, key=lambda item: int(item["record_index"])):
            _merge_dataset_records(datasets, result["datasets"])
            _merge_record_summary(summary, result["summary"])

    summary["segments_processed"] = int(summary["files_processed"])
    summary["physical_files_processed"] = len(
        {
            str(record.get("file_path") or "")
            for record in summary.get("file_records") or []
            if record.get("status") == "complete"
        }
    )
    if summary["files_processed"] == summary["files_attempted"]:
        status = "complete"
    elif summary["files_processed"] > 0:
        status = "complete_with_bad_files"
    else:
        status = "failed"
    summary["status"] = status
    summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["wall_time_s"] = round(time.time() - start, 3)

    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": status,
        "year": 2024,
        "histogram_definition_source": str(
            Path(__file__).resolve().parents[1] / "workflow" / "build_flat_boosted_recoil_hists.py"
        ),
        "recoil_pt_bins": builder.RECOIL_PT_BINS,
        "regions": builder.REGION_VARIABLES,
        "ntop_split_policy": {
            "status": "included",
            "axis": "nboosted_top",
            "split_regions": {
                region: {"Nt1": f"{region}_Nt1", "Nt0": f"{region}_Nt0"}
                for region in builder.NTOP_SPLIT_BASE_REGIONS
            },
        },
        "search_bin_schemes": {
            builder.SELECTED_RECOIL54_SCHEME: {
                "bin_labels": builder.selected_an17_recoil54_labels(),
                "selection": "feature_SR with Nt0 categories and selected AN17 bins split into recoil/MET bins",
                "selected_an17_bins_1based": builder.SELECTED_AN17_RECOIL_BINS_1BASED,
                "recoil_pt_bins": builder.RECOIL_PT_BINS,
            },
            builder.EXTENDED_RECOIL60_SCHEME: {
                "bin_labels": builder.selected_an17_recoil60_labels(),
                "selection": "the adopted feature_SR categorization with Nb=2,Nt>=2,NW=0 inserted as bins 37--42, all classes split into six recoil/MET bins",
                "base_scheme": builder.SELECTED_RECOIL54_SCHEME,
                "extra_category": builder.EXTENDED_RECOIL60_CATEGORY_KEY,
                "recoil_pt_bins": builder.RECOIL_PT_BINS,
            },
            **{
                channel: {
                    "bin_labels": builder.LOWDM_34BIN_LABELS,
                    "selection": f"feature_lowdm_{region} && Nb>=1",
                    "delta_m": "low",
                    "region": region,
                    "category_sizes": builder.LOWDM_NBGE1_CATEGORY_SIZES,
                    "removed_categories": builder.LOWDM_REMOVED_NB0_CATEGORY_SIZES,
                }
                for region, channel in builder.LOWDM_REGION_MAP.items()
            },
        },
        "lowdm_region_policy": {
            "status": "same_as_nominal_builder",
            "search_bins": "34-bin Nsv-inclusive Low-dM search scheme per region with explicit Nb>=1",
            "isr_subjet_bveto": "diagnostic only; not applied",
            "mtb_requirement": "diagnostic only; mTb<175 is not applied",
            "regions": builder.LOWDM_REGION_MAP,
        },
        "highdm_distribution_variable_specs": builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS,
        "highdm_distribution_regions": {
            "control": builder.HIGHDM_CR_REGIONS,
            "validation": builder.HIGHDM_VR_REGIONS,
            "signal_categories": builder.HIGHDM_SR_CATEGORY_KEYS,
        },
        "lowdm_variable_specs": builder.LOWDM_VARIABLE_SPECS,
        "lowdm_region_variables": builder.LOWDM_REGION_VARIABLES,
        "jes_source_policy": {
            "status": "adopted_final_11_sources",
            "public_sources": list(FINAL_JES_PUBLIC_SOURCES),
            "correctionlib_sources": list(FINAL_JES_CORRECTION_SOURCES),
            "excluded_validation_envelopes": ["jesTotal", "jesRegroupedTotal"],
        },
        "shape_nuisances": list(FINAL_SHAPE_NUISANCES),
        "variations": list(variations),
        "search_bin_scheme": {
            "name": builder.EXTENDED_RECOIL60_SCHEME,
            "bins": len(builder.selected_an17_recoil60_labels()),
            "labels": builder.selected_an17_recoil60_labels(),
        },
        "output_policy": {
            "event_rows_stored": False,
            "shifted_root_stored": False,
            "compressed_histogram_only": True,
            "sections": list(OUTPUT_SECTIONS),
        },
        "summary": summary,
        "datasets": datasets,
    }
    write_json_gz(output, payload)
    metadata = {
        "schema_version": f"{OUTPUT_SCHEMA}_metadata",
        "status": status,
        "histogram_file": str(output),
        "histogram_size": output.stat().st_size,
        "histogram_sha256": sha256(output),
        "source_shard": str(shard_path),
        "source_record_digest": shard.get("record_digest"),
        "summary": summary,
        "dataset_count": len(datasets),
        "jes_source_count": len(FINAL_JES_PUBLIC_SOURCES),
        "shape_nuisance_count": len(FINAL_SHAPE_NUISANCES),
        "variation_count": len(variations),
    }
    write_json(metadata_output, metadata)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "metadata": str(metadata_output),
                "files_processed": summary["files_processed"],
                "files_attempted": summary["files_attempted"],
                "events_read": summary["events_read"],
                "variation_count": len(variations),
                "histogram_size": output.stat().st_size,
            },
            sort_keys=True,
        )
    )
    return 0 if summary["files_processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
