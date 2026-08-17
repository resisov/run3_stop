"""Build compact 2024 photon-template events without changing nominal outputs.

The event and non-photon object selection is delegated to
``real_subset_worker.py`` through ``intermediate_2024_worker.py``.  This worker
only replaces the selected photon's cutBased value in memory so that the
trusted GCR event selection can be evaluated for a photon candidate in which
the shower-shape and charged-isolation requirements are left open.  Continuous
photon observables are persisted for the downstream template fit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from . import intermediate_2024_worker as intermediate
from . import photon_fake_2024_worker as common
from . import real_subset_worker as baseline
from .object_corrections_2024 import branch_audit, calibrate_jets_and_met, validate_payloads
from .shape_histogram_2024_worker import (
    load_histogram_builder,
    physical_dataset,
    record_entry_bounds,
    required_read_branches,
)


OUTPUT_SCHEMA = "photon_fake_template_events_2024_v1"
TEMPLATE_REQUIRED_BRANCHES = (
    "Photon_vidNestedWPBitmap",
    "Photon_sieie",
    "Photon_pfRelIso03_chg_quadratic",
)
_PARALLEL_BUILDER: Any | None = None
_PARALLEL_PAYLOAD_STATUS: dict[str, Any] | None = None


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


def template_candidate_mask(arrays: Any) -> Any:
    """Photon N-2 mask: leave sigma_ieta_ieta and charged isolation open."""
    fields = set(arrays.fields)
    missing = {"Photon_vidNestedWPBitmap"} - fields
    if missing:
        raise RuntimeError(f"template photon branches missing: {sorted(missing)}")
    bitmap = arrays["Photon_vidNestedWPBitmap"]
    pt = arrays["Photon_pt"]
    eta = arrays["Photon_eta"]
    electron_veto = ak.values_astype(arrays["Photon_electronVeto"], np.bool_)
    fiducial = (abs(eta) < 1.4442) | ((abs(eta) > 1.5660) & (abs(eta) < 2.5))
    return (
        (pt > 220.0)
        & fiducial
        & electron_veto
        & (common.vid_level(bitmap, "pt") >= 2)
        & (common.vid_level(bitmap, "eta") >= 2)
        & (common.vid_level(bitmap, "hoe") >= 2)
        & (common.vid_level(bitmap, "ecal_iso") >= 2)
        & (common.vid_level(bitmap, "hcal_iso") >= 2)
    )


def template_probe_assignment(arrays: Any) -> tuple[Any, np.ndarray]:
    """Choose the unique photon while preserving nominal target multiplicity.

    Exactly one nominal medium target wins even if extra relaxed candidates are
    present.  With no target, exactly one N-2 candidate is required.  Multiple
    nominal targets or ambiguous all-relaxed events are rejected.
    """
    candidate = template_candidate_mask(arrays)
    bitmap = arrays["Photon_vidNestedWPBitmap"]
    target = (
        candidate
        & (common.vid_level(bitmap, "sieie") >= 2)
        & (common.vid_level(bitmap, "charged_iso") >= 2)
    )
    target_count = ak.sum(target, axis=1)
    candidate_count = ak.sum(candidate, axis=1)
    selected = (target & (target_count == 1)) | (
        candidate & (target_count == 0) & (candidate_count == 1)
    )
    has_selected = np.asarray(ak.to_numpy(ak.sum(selected, axis=1) == 1), dtype=bool)
    return selected, has_selected


def _selected(values: Any, mask: Any, *, fill: float = np.nan) -> np.ndarray:
    first = ak.fill_none(ak.firsts(values[mask]), fill)
    return np.asarray(ak.to_numpy(first))


def selected_probe_observables(arrays: Any, mask: Any, is_data: bool) -> dict[str, np.ndarray]:
    bitmap = arrays["Photon_vidNestedWPBitmap"]
    result = {
        "pt": _selected(arrays["Photon_pt"], mask).astype(float),
        "eta": _selected(arrays["Photon_eta"], mask).astype(float),
        "sieie": _selected(arrays["Photon_sieie"], mask).astype(float),
        "charged_iso": _selected(
            arrays["Photon_pfRelIso03_chg_quadratic"], mask
        ).astype(float),
        "shape_level": _selected(common.vid_level(bitmap, "sieie"), mask, fill=-1).astype(int),
        "charged_iso_level": _selected(
            common.vid_level(bitmap, "charged_iso"), mask, fill=-1
        ).astype(int),
    }
    if is_data:
        result["gen_part_flavour"] = np.zeros(len(arrays), dtype=int)
    else:
        result["gen_part_flavour"] = _selected(
            arrays["Photon_genPartFlav"], mask, fill=0
        ).astype(int)
    return result


def photon_category(shape_level: int, charged_iso_level: int) -> str:
    if shape_level >= 2 and charged_iso_level >= 2:
        return "tight"
    if shape_level >= 2 and charged_iso_level == 1:
        return "loose_charged_iso"
    if charged_iso_level == 0:
        return "fail_loose_charged_iso"
    return "fit_only"


def _row_value(row: dict[str, Any], spec: dict[str, Any]) -> float:
    branch = (spec.get("branch_by_region") or {}).get("GCR", spec["branch"])
    if spec.get("source", "scalar") == "masked_first":
        for value, keep in zip(row.get(branch) or [], row.get(str(spec["mask_branch"])) or []):
            if bool(keep):
                return float(value)
        return float(spec.get("fill", -99.0))
    value = row.get(branch, spec.get("fill", -99.0))
    if isinstance(value, list):
        return float(value[0]) if value else float(spec.get("fill", -99.0))
    return float(value)


def compact_event(
    row: dict[str, Any],
    probe: dict[str, np.ndarray],
    local: int,
    record: dict[str, Any],
    builder: Any,
) -> dict[str, Any]:
    is_data = bool(record.get("is_data"))
    flavour = 0 if is_data else int(probe["gen_part_flavour"][local])
    shape_level = int(probe["shape_level"][local])
    charged_level = int(probe["charged_iso_level"][local])
    values = {
        "ut": float(row["recoil_gcr"]),
        **{
            variable: _row_value(row, spec)
            for variable, spec in builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
        },
    }
    return {
        "run": int(row["run"]),
        "luminosityBlock": int(row["luminosityBlock"]),
        "event": int(row["event"]),
        "entry": int(row["entry"]),
        "source_dataset": str(record.get("dataset") or "unknown"),
        "physical_dataset": physical_dataset(str(record.get("dataset") or "unknown")),
        "process": str(record.get("process_group") or "unknown"),
        "is_data": is_data,
        "nominal_weight_without_photon_id_sf": float(row["nominal_weight"]),
        "probe": {
            "pt": float(probe["pt"][local]),
            "eta": float(probe["eta"][local]),
            "sieie": float(probe["sieie"][local]),
            "charged_iso": float(probe["charged_iso"][local]),
            "shape_level": shape_level,
            "charged_iso_level": charged_level,
            "gen_part_flavour": flavour,
            "origin": "data" if is_data else common.photon_origin(flavour),
            "category": photon_category(shape_level, charged_level),
        },
        "regions": common.row_regions(row),
        "values": values,
    }


def _dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_dataset": physical_dataset(str(record.get("dataset") or "unknown")),
        "process": str(record.get("process_group") or "unknown"),
        "xsec_pb": record.get("xsec_pb"),
        "dataset_splits": [str(record.get("dataset") or "unknown")],
        "files_processed": 0,
        "events_read": 0,
        "selected_events": 0,
    }


def process_record(
    record: dict[str, Any],
    builder: Any,
    payload_status: dict[str, Any],
    chunk_size: int,
    prefilter_block_size: int,
    max_chunks_per_file: int | None,
) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process_group") or "unknown")
    file_path = str(record.get("file_path") or "")
    is_data = bool(record.get("is_data"))
    if is_data and process != "EGamma":
        raise RuntimeError(f"template data must come from EGamma, got {process}")
    root_file = None
    access_info: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    summary = {
        "files_attempted": 1,
        "files_processed": 0,
        "events_read": 0,
        "photon_prefilter_candidate_events": 0,
        "prefilter_candidate_events": 0,
        "fully_evaluated_events": 0,
        "prefilter_blocks_evaluated": 0,
        "selected_events": 0,
        "category_counts": {},
        "bad_files": [],
        "file_records": [],
    }
    dataset_record = _dataset_record(record)
    started = time.time()
    try:
        root_file, access_info = common._open_root_for_record(file_path, timeout=60)
        tree = root_file["Events"]
        present = set(tree.keys())
        required = set(TEMPLATE_REQUIRED_BRANCHES)
        if not is_data:
            required.add("Photon_genPartFlav")
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(f"template photon branches missing: {missing}")
        audit = branch_audit(present, is_data=is_data)
        if audit.get("status") != "valid":
            raise RuntimeError(f"required 2024 object branches missing: {audit.get('missing_required')}")
        branches = sorted(set(required_read_branches(tree)) | required)
        prefilter_branches = common._prefilter_read_branches(present)
        assigned_start, assigned_stop = record_entry_bounds(record, int(tree.num_entries))
        chunks_seen = 0
        for entry_start in range(assigned_start, assigned_stop, chunk_size):
            if max_chunks_per_file is not None and chunks_seen >= max_chunks_per_file:
                break
            entry_stop = min(entry_start + chunk_size, assigned_stop)
            prefilter = tree.arrays(
                prefilter_branches, entry_start=entry_start, entry_stop=entry_stop, library="ak"
            )
            summary["events_read"] += len(prefilter)
            _, photon_candidate = template_probe_assignment(prefilter)
            summary["photon_prefilter_candidate_events"] += int(np.count_nonzero(photon_candidate))
            candidate, _ = common._necessary_gcr_event_mask(
                prefilter, photon_candidate, is_data, process
            )
            summary["prefilter_candidate_events"] += int(np.count_nonzero(candidate))
            offsets = np.flatnonzero(candidate)
            block_offsets = np.unique((offsets // prefilter_block_size) * prefilter_block_size)
            for block_offset in block_offsets:
                block_start = entry_start + int(block_offset)
                block_stop = min(block_start + prefilter_block_size, entry_stop)
                arrays = tree.arrays(
                    branches, entry_start=block_start, entry_stop=block_stop, library="ak"
                )
                summary["fully_evaluated_events"] += len(arrays)
                summary["prefilter_blocks_evaluated"] += 1
                selected_mask, full_candidate = template_probe_assignment(arrays)
                expected = photon_candidate[int(block_offset) : int(block_offset) + len(arrays)]
                if not np.array_equal(full_candidate, expected):
                    raise RuntimeError("minimal/full template photon prefilter disagreement")
                probe = selected_probe_observables(arrays, selected_mask, is_data)
                corrected, calibration = calibrate_jets_and_met(
                    arrays, is_data=is_data, shift="nominal", root=Path.cwd()
                )
                modified = common._modified_cutbased(corrected, selected_mask)
                validation_context = {"object_branch_audit": audit, "payload_status": payload_status}
                with common.suppress_full_medium_photon_id_sf():
                    rows, chunk_summary = intermediate.extract_chunk_2024(
                        arrays,
                        dataset,
                        process,
                        None,
                        str(record.get("year") or "2024"),
                        file_path,
                        block_start,
                        block_stop,
                        fastsim_trigger_bypass=False,
                        shift_name="nominal",
                        compute_weights=True,
                        decorate_rows=False,
                        materialize_skim_flag="feature_GCR_fake_validation",
                        precalibrated=(modified, calibration),
                        validation_context=validation_context,
                    )
                for row in rows:
                    if not common.row_regions(row):
                        continue
                    local = int(row["entry"]) - block_start
                    item = compact_event(row, probe, local, record, builder)
                    events.append(item)
                    category = str(item["probe"]["category"])
                    summary["category_counts"][category] = summary["category_counts"].get(category, 0) + 1
                if not is_data:
                    btag = (((chunk_summary.get("scale_factor_status") or {}).get("components") or {}).get("btagSF") or {})
                    if not btag.get("applied"):
                        raise RuntimeError(f"btagSF unavailable for {dataset}: {btag}")
            chunks_seen += 1
        if max_chunks_per_file is None and summary["events_read"] != assigned_stop - assigned_start:
            raise RuntimeError(
                f"processed Events entries differ from assigned range: {summary['events_read']} vs {assigned_stop - assigned_start}"
            )
        summary["files_processed"] = 1
        summary["selected_events"] = len(events)
        dataset_record["files_processed"] = 1
        dataset_record["events_read"] = summary["events_read"]
        dataset_record["selected_events"] = len(events)
        summary["file_records"].append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "events_read": summary["events_read"],
                "selected_events": len(events),
                "wall_time_s": round(time.time() - started, 3),
                "status": "complete",
                "access": access_info,
            }
        )
    except Exception as exc:
        if isinstance(exc, baseline.RootOpenFailure):
            access_info = dict(exc.access_info)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["bad_files"].append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "failure_stage": "photon_template_open_read_or_evaluate",
                "exception_type": type(exc).__name__,
                "concise_error": str(exc)[:500],
                "first_failure_time": now,
                "last_failure_time": now,
                "alternate_access_attempted": bool(access_info.get("alternate_access_attempted")),
                "permanently_skipped": False,
            }
        )
    finally:
        try:
            if root_file is not None:
                root_file.close()
        except Exception:
            pass
        baseline.cleanup_xrd_cache(access_info)
    return {"dataset": dataset_record, "events": events, "summary": summary}


def _parallel_initializer() -> None:
    global _PARALLEL_BUILDER, _PARALLEL_PAYLOAD_STATUS
    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload = validate_payloads(Path.cwd())
    if payload.get("status") != "valid":
        raise RuntimeError(f"2024 correction payload validation failed: {payload.get('errors')}")
    _PARALLEL_PAYLOAD_STATUS = payload
    _PARALLEL_BUILDER = load_histogram_builder()


def _process_isolated(task: tuple[dict[str, Any], int, int, int | None]) -> dict[str, Any]:
    if _PARALLEL_BUILDER is None or _PARALLEL_PAYLOAD_STATUS is None:
        raise RuntimeError("template record worker not initialized")
    return process_record(task[0], _PARALLEL_BUILDER, _PARALLEL_PAYLOAD_STATUS, *task[1:])


def _merge_dataset(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    key = str(incoming["physical_dataset"])
    if key not in target:
        target[key] = copy.deepcopy(incoming)
        return
    record = target[key]
    if record["process"] != incoming["process"]:
        raise RuntimeError(f"process mismatch for {key}")
    if record.get("xsec_pb") != incoming.get("xsec_pb"):
        raise RuntimeError(f"cross-section mismatch for {key}")
    record["dataset_splits"] = sorted(set(record["dataset_splits"]) | set(incoming["dataset_splits"]))
    for field in ("files_processed", "events_read", "selected_events"):
        record[field] += int(incoming[field])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--prefilter-block-size", type=int, default=512)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-chunks-per-file", type=int)
    parser.add_argument("--record-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or args.prefilter_block_size <= 0 or args.record_workers <= 0:
        parser.error("chunk, prefilter block, and worker counts must be positive")
    if args.prefilter_block_size > args.chunk_size:
        parser.error("--prefilter-block-size cannot exceed --chunk-size")

    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload_status = validate_payloads(Path.cwd())
    if payload_status.get("status") != "valid":
        raise RuntimeError(f"2024 correction payload validation failed: {payload_status.get('errors')}")
    builder = load_histogram_builder()
    shard_path = Path(args.shard)
    shard = json.loads(shard_path.read_text())
    records = list(shard.get("records") or [])
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    if not records:
        raise RuntimeError("template shard has no records")

    tasks = [(record, args.chunk_size, args.prefilter_block_size, args.max_chunks_per_file) for record in records]
    if args.record_workers == 1 or len(tasks) == 1:
        results = [
            process_record(record, builder, payload_status, chunk, block, maximum)
            for record, chunk, block, maximum in tasks
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.record_workers, len(tasks)), initializer=_parallel_initializer
        ) as executor:
            results = list(executor.map(_process_isolated, tasks))

    datasets: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    summary = {
        "status": "running",
        "source_shard": str(shard_path),
        "source_record_digest": shard.get("record_digest"),
        "files_attempted": len(records),
        "files_processed": 0,
        "events_read": 0,
        "photon_prefilter_candidate_events": 0,
        "prefilter_candidate_events": 0,
        "fully_evaluated_events": 0,
        "prefilter_blocks_evaluated": 0,
        "selected_events": 0,
        "category_counts": {},
        "bad_files": [],
        "file_records": [],
        "selection_source": "real_subset_worker.py via intermediate_2024_worker.py",
    }
    for result in results:
        _merge_dataset(datasets, result["dataset"])
        events.extend(result["events"])
        child = result["summary"]
        for field in (
            "files_processed", "events_read", "photon_prefilter_candidate_events",
            "prefilter_candidate_events", "fully_evaluated_events",
            "prefilter_blocks_evaluated", "selected_events",
        ):
            summary[field] += int(child.get(field) or 0)
        for category, count in (child.get("category_counts") or {}).items():
            summary["category_counts"][category] = summary["category_counts"].get(category, 0) + int(count)
        summary["bad_files"].extend(child.get("bad_files") or [])
        summary["file_records"].extend(child.get("file_records") or [])
    summary["status"] = (
        "complete" if summary["files_processed"] == summary["files_attempted"]
        else "complete_with_bad_files" if summary["files_processed"] else "failed"
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": summary["status"],
        "year": 2024,
        "scope": "high-dM GCR and adjacent delta-phi validation regions",
        "selection": {
            "event_and_object_selection": "real_subset_worker.py",
            "photon_candidate": "medium VID with sieie and charged isolation requirements removed",
            "multiplicity": "one nominal target takes precedence; otherwise exactly one N-2 candidate",
            "photon_sf_policy": "full-medium photon ID SF suppressed; all other nominal central weights retained",
        },
        "summary": summary,
        "datasets": datasets,
        "events": events,
    }
    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    write_json_gz(output, payload)
    write_json(
        metadata_output,
        {
            "schema_version": f"{OUTPUT_SCHEMA}_metadata",
            "status": summary["status"],
            "event_file": str(output),
            "event_file_size": output.stat().st_size,
            "event_file_sha256": sha256(output),
            "source_shard": str(shard_path),
            "source_record_digest": shard.get("record_digest"),
            "summary": summary,
            "dataset_count": len(datasets),
            "event_count": len(events),
        },
    )
    print(json.dumps({"status": summary["status"], "output": str(output), "events": len(events)}, sort_keys=True))
    return 0 if summary["files_processed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
