#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DATA_PREFIXES = ("EGamma", "JetMET", "Muon")
ACCESS_TOKENS = (
    "operation expired",
    "timed out",
    "timeout",
    "redirect limit",
    "xrd",
    "xrootd",
    "certificate",
    "proxy",
    "auth",
)
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def add_package_path(repo: Path) -> None:
    package_dir = repo / "autonomous_allhad"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


def display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def item_file_path(item: dict[str, Any]) -> str:
    return str(item.get("file_path") or item.get("path") or item.get("file") or "")


def source_manifest_name(source_json: str) -> str:
    return source_json.replace(".running", "")


def successful_summary(summary: dict[str, Any]) -> bool:
    return summary.get("read_status") == "success" or summary.get("processing_status") == "processed_full_file"


def classify_bad_item(item: dict[str, Any]) -> str:
    dataset = str(item.get("dataset") or "")
    file_path = item_file_path(item)
    if "/store/data/" in file_path or dataset.startswith(DATA_PREFIXES):
        return "data"
    if "/store/mc/" in file_path:
        return "mc"
    return "other"


def infer_process(item: dict[str, Any]) -> str:
    for key in ("process", "process_group"):
        value = str(item.get(key) or "")
        if value:
            return value
    text = f"{item.get('dataset') or ''} {item_file_path(item)}"
    if "GJets" in text or "Gamma" in text:
        return "GJ"
    if "QCD" in text:
        return "QCD"
    if "DY" in text:
        return "DY"
    if "WtoLNu" in text or "WJets" in text:
        return "WtoLNu"
    if "TT" in text or "ST_" in text or "TBbar" in text or "Tbar" in text:
        return "TT/ST"
    return "other"


def reason_bucket(item: dict[str, Any]) -> str:
    blob = " ".join(
        str(item.get(key) or "")
        for key in ("concise_error", "error", "exception_type", "direct_open_error", "xrdcp_stderr_tail")
    ).lower()
    if "operation expired" in blob:
        return "operation_expired"
    if "map::at" in blob or "summer24prompt24_v2" in blob:
        return "jec_key_error"
    if any(token in blob for token in ACCESS_TOKENS):
        return "external_access"
    return "other"


def xrd_source_candidates(file_path: str) -> list[str]:
    if not str(file_path).startswith("root://"):
        return [file_path]
    candidates = [file_path]
    idx = str(file_path).find("/store/")
    if idx >= 0:
        lfn = str(file_path)[idx:]
        for host in ["cmsxrootd.fnal.gov", "xrootd-cms.infn.it", "cms-xrd-global.cern.ch"]:
            candidates.append(f"root://{host}/{lfn}")
    out: list[str] = []
    for item in candidates:
        if item not in out:
            out.append(item)
    return out


def should_retry_alternate_source(bad_files: list[dict[str, Any]], file_summary: dict[str, Any]) -> bool:
    blob = " ".join(
        [str(file_summary.get("error") or "")]
        + [str(b.get("concise_error") or b.get("exception_type") or "") for b in bad_files if isinstance(b, dict)]
    ).lower()
    return any(token in blob for token in ["deserialization", "tbasket", "outside expected range", "while reading"])


def extract_top_level_array(path: Path, key: str) -> list[Any]:
    prefix = f'"{key}"'
    collecting = False
    value_chars: list[str] = []
    depth = 0
    in_string = False
    escape = False
    started = False
    with path.open(errors="replace") as handle:
        for line in handle:
            if not collecting:
                stripped = line.lstrip()
                if not stripped.startswith(prefix):
                    continue
                colon = line.find(":")
                if colon < 0:
                    return []
                line = line[colon + 1 :]
                collecting = True
            for char in line:
                if not started:
                    if char.isspace():
                        continue
                    if char != "[":
                        raise ValueError(f"{path}: top-level {key} is not an array")
                    started = True
                    depth = 1
                    value_chars.append(char)
                    continue
                value_chars.append(char)
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                    if depth == 0:
                        return json.loads("".join(value_chars))
    return []


def find_record(shard_path: Path, file_path: str) -> dict[str, Any] | None:
    if not shard_path.exists():
        return None
    shard = read_json(shard_path)
    if not isinstance(shard, dict):
        return None
    for record in shard.get("records") or []:
        if str(record.get("file_path") or "") == file_path:
            return record
    return None


def cached_find_record(cache: dict[Path, dict[str, dict[str, Any]] | None], shard_path: Path, file_path: str) -> dict[str, Any] | None:
    if shard_path not in cache:
        records: dict[str, dict[str, Any]] = {}
        try:
            shard = read_json(shard_path)
        except Exception:
            cache[shard_path] = None
        else:
            if isinstance(shard, dict):
                for record in shard.get("records") or []:
                    if isinstance(record, dict):
                        path = str(record.get("file_path") or "")
                        if path:
                            records[path] = record
            cache[shard_path] = records
    bucket = cache.get(shard_path)
    if not bucket:
        return None
    return bucket.get(file_path)


def recovered_paths(summary: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for result in summary.get("results") or []:
        if result.get("recovery_status") == "recovered" and result.get("applied"):
            paths.add(str(result.get("file_path") or ""))
    return paths


def load_history(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        payload = read_json(summary_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_candidates(
    output_dir: Path,
    shard_dir: Path,
    kinds: set[str],
    include_running: bool,
    retry_permanent: bool,
    access_only: bool,
    recovered: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[tuple[str, Path]] = [("final", p) for p in sorted(output_dir.glob("shard_*.json"))]
    if include_running:
        sources.extend(("running_checkpoint", p) for p in sorted(output_dir.glob("shard_*.json.running")))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stats: dict[str, Any] = {
        "source_files": len(sources),
        "source_read_errors": {},
        "bad_kind_counts": Counter(),
        "bad_reason_counts": Counter(),
        "bad_process_counts": Counter(),
        "skipped_recovered": 0,
        "skipped_permanent": 0,
        "skipped_non_access": 0,
    }
    record_cache: dict[Path, dict[str, dict[str, Any]] | None] = {}
    for source_kind, source_path in sources:
        try:
            bad_files = extract_top_level_array(source_path, "bad_files")
        except Exception as exc:
            stats["source_read_errors"][source_path.name] = f"{type(exc).__name__}: {exc}"
            continue
        shard_id = source_path.name.replace(".json.running", "").replace(".json", "")
        shard_path = shard_dir / source_manifest_name(source_path.name)
        for item in bad_files:
            if not isinstance(item, dict):
                continue
            kind = classify_bad_item(item)
            reason = reason_bucket(item)
            file_path = item_file_path(item)
            record = cached_find_record(record_cache, shard_path, file_path) if file_path else None
            process = str((record or {}).get("process_group") or infer_process(item))
            stats["bad_kind_counts"][kind] += 1
            stats["bad_reason_counts"][reason] += 1
            if kind == "mc":
                stats["bad_process_counts"][process] += 1
            if kind not in kinds:
                continue
            if not file_path:
                continue
            if file_path in recovered:
                stats["skipped_recovered"] += 1
                continue
            if item.get("permanently_skipped") and not retry_permanent:
                stats["skipped_permanent"] += 1
                continue
            if access_only and reason not in {"operation_expired", "external_access"}:
                stats["skipped_non_access"] += 1
                continue
            key = (source_path.name, file_path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "kind": kind,
                    "process": process,
                    "source_kind": source_kind,
                    "candidate_reason": reason,
                    "source_json": source_path.name,
                    "shard_id": shard_id,
                    "dataset": item.get("dataset") or "",
                    "file_path": file_path,
                    "failure_stage": item.get("failure_stage") or "",
                    "exception_type": item.get("exception_type") or "",
                    "concise_error": item.get("concise_error") or "",
                    "permanently_skipped": bool(item.get("permanently_skipped")),
                }
            )
    for key in ("bad_kind_counts", "bad_reason_counts", "bad_process_counts"):
        stats[key] = dict(stats[key])
    return candidates, stats


def process_candidate(repo_text: str, shard_dir_text: str, tag: str, shift_name: str, chunk_size: int, candidate: dict[str, Any]) -> dict[str, Any]:
    for key in THREAD_ENV:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("XRD_NETWORKSTACK", "IPv4")
    os.environ.setdefault("AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT", "1800")
    os.environ.setdefault("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", "1")
    os.environ.setdefault("AUTONOMOUS_ALLHAD_XRD_CACHE", f"/tmp/taiwoo/autonomous_allhad_xrd_cache_{tag}")

    repo = Path(repo_text)
    add_package_path(repo)
    from autonomous_allhad.full_production_worker import process_file

    shard_dir = Path(shard_dir_text)
    workflow = repo / "autonomous_allhad/workflow"
    shard_path = shard_dir / source_manifest_name(str(candidate["source_json"]))
    result = dict(candidate)
    result.update(
        {
            "tag": tag,
            "shape_shift": shift_name,
            "started_at": utc_now(),
            "finished_at": None,
            "recovery_status": "not_started",
            "events_read": 0,
            "regions": {},
            "error": "",
            "recovered_payload_json": "",
            "failed_file_summary": {},
            "bad_files": [],
            "applied": False,
            "apply_status": "",
        }
    )
    try:
        record = find_record(shard_path, str(candidate["file_path"]))
        if record is None:
            result["recovery_status"] = "skipped_record_not_found"
            result["error"] = str(shard_path)
        else:
            file_summary, file_payload, bad = process_file(record, repo, chunk_size, shift_name=shift_name)
            alternate_retry_attempts: list[dict[str, Any]] = []
            original_file_path = str(record.get("file_path") or candidate["file_path"])
            if file_payload is None and should_retry_alternate_source(bad, file_summary):
                for alternate_source in xrd_source_candidates(original_file_path)[1:]:
                    retry_record = dict(record)
                    retry_record["file_path"] = alternate_source
                    retry_summary, retry_payload, retry_bad = process_file(retry_record, repo, chunk_size, shift_name=shift_name)
                    alternate_retry_attempts.append(
                        {
                            "source": alternate_source,
                            "recovery_status": "recovered" if retry_payload is not None else "still_bad",
                            "events_read": int(retry_summary.get("events_read") or 0),
                            "error": str(retry_summary.get("error") or "; ".join(str(b.get("concise_error") or b.get("exception_type") or b) for b in retry_bad))[:800],
                        }
                    )
                    if retry_payload is not None:
                        retry_summary["file_path"] = original_file_path
                        retry_summary["alternate_recovery_source"] = alternate_source
                        retry_summary["alternate_recovery_original_file_path"] = original_file_path
                        if isinstance(retry_summary.get("file_access"), dict):
                            retry_summary["file_access"]["alternate_recovery_source"] = alternate_source
                            retry_summary["file_access"]["alternate_recovery_original_file_path"] = original_file_path
                        file_summary, file_payload, bad = retry_summary, retry_payload, []
                        break
                    file_summary, bad = retry_summary, retry_bad
            if alternate_retry_attempts:
                result["alternate_retry_attempts"] = alternate_retry_attempts
            result["events_read"] = int(file_summary.get("events_read") or 0)
            result["regions"] = file_summary.get("region_counts") or {}
            if file_payload is not None:
                result["recovery_status"] = "recovered"
                payload_dir = workflow / f"local_bad_file_recovery_{tag}" / "recovered_file_payloads"
                payload_path = payload_dir / f"{candidate['shard_id']}__{int(candidate['index']):06d}.json"
                write_json(
                    payload_path,
                    {
                        "tag": tag,
                        "created_at": utc_now(),
                        "source_kind": candidate["source_kind"],
                        "candidate_reason": candidate.get("candidate_reason"),
                        "source_json": candidate["source_json"],
                        "shard_id": candidate["shard_id"],
                        "dataset": candidate["dataset"],
                        "file_path": candidate["file_path"],
                        "shape_shift": shift_name,
                        "file_summary": file_summary,
                        "file_payload": file_payload,
                    },
                )
                result["recovered_payload_json"] = display_path(payload_path, repo)
            else:
                result["recovery_status"] = "still_bad"
                result["failed_file_summary"] = file_summary
                result["bad_files"] = bad
                result["error"] = "; ".join(str(b.get("concise_error") or b.get("exception_type") or b) for b in bad)[:800]
    except Exception as exc:
        result["recovery_status"] = "local_exception"
        result["error"] = f"{type(exc).__name__}: {exc}"[:800]
    result["finished_at"] = utc_now()
    return result


def fallback_bad_entry(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": result.get("dataset") or "",
        "file_path": result.get("file_path") or "",
        "failure_stage": "local_bad_file_recovery",
        "exception_type": "RecoveryFailed",
        "concise_error": result.get("error") or "local bad-file recovery failed",
        "first_failure_time": utc_now(),
        "last_failure_time": utc_now(),
        "alternate_access_attempted": True,
        "external_access_blocker": True,
        "permanently_skipped": False,
    }


def finalize_source(source_path: Path, source: dict[str, Any]) -> tuple[bool, str]:
    attempted = int(source.get("files_attempted") or 0)
    processed = int(source.get("files_processed") or 0)
    bad_files = source.get("bad_files") or []
    if processed == attempted and not bad_files:
        source["status"] = "complete"
    elif processed > 0:
        source["status"] = "complete_with_bad_files"
    else:
        source["status"] = "failed"
    source["completed_at"] = source.get("completed_at") or utc_now()
    write_json(source_path, source)
    return True, "applied_to_final"


def apply_recovery(repo: Path, output_dir: Path, shard_dir: Path, result: dict[str, Any]) -> tuple[bool, str]:
    add_package_path(repo)
    from autonomous_allhad.full_production_worker import ensure_dataset, merge_file_payload

    status = result.get("recovery_status")
    if status not in {"recovered", "still_bad"}:
        return False, "not_recovered"
    source_path = output_dir / str(result["source_json"])
    payload_path = repo / str(result.get("recovered_payload_json") or "")
    if not source_path.exists():
        return False, "missing_source_json"
    if status == "recovered" and not payload_path.exists():
        return False, "missing_recovered_payload"

    source = read_json(source_path)
    recovered = read_json(payload_path) if status == "recovered" else {}
    if not isinstance(source, dict) or not isinstance(recovered, dict):
        return False, "invalid_json"

    file_path = str(result["file_path"])
    bad_files = source.get("bad_files") or []
    remaining_bad = []
    removed_bad = 0
    for bad in bad_files:
        if isinstance(bad, dict) and item_file_path(bad) == file_path:
            removed_bad += 1
            continue
        remaining_bad.append(bad)
    source["bad_files"] = remaining_bad

    file_summary = recovered.get("file_summary") if status == "recovered" else result.get("failed_file_summary")
    file_payload = recovered.get("file_payload") if status == "recovered" else None
    if not isinstance(file_summary, dict) or not file_summary:
        return False, "missing_file_summary"

    summaries = source.setdefault("file_summaries", [])
    existing_success = False
    had_summary = False
    replaced_summary = False
    for idx, summary in enumerate(summaries):
        if isinstance(summary, dict) and item_file_path(summary) == file_path:
            had_summary = True
            existing_success = successful_summary(summary)
            if existing_success and status == "recovered" and removed_bad == 0:
                return False, "already_applied"
            if not existing_success:
                summaries[idx] = file_summary
                replaced_summary = True
            break
    if not had_summary:
        summaries.append(file_summary)
    elif existing_success and status == "still_bad":
        return False, "already_processed"

    record = find_record(shard_dir / source_manifest_name(str(result["source_json"])), file_path) or {
        "dataset": result.get("dataset"),
        "process_group": file_summary.get("process") or result.get("process"),
        "xsec_pb": None,
        "is_data": result.get("kind") == "data",
        "is_background": result.get("kind") == "mc",
    }
    dataset_key = str(record.get("dataset") or result.get("dataset") or "")
    datasets = source.setdefault("datasets", {})
    dataset_was_known = dataset_key in datasets
    dataset_rec = ensure_dataset(source, record)

    had_attempt = had_summary or removed_bad > 0
    if not had_attempt:
        source["files_attempted"] = int(source.get("files_attempted") or 0) + 1
    if not dataset_was_known or not had_attempt:
        dataset_rec["files_attempted"] = int(dataset_rec.get("files_attempted") or 0) + 1

    if status == "recovered":
        if not existing_success:
            dataset_rec["files_processed"] = int(dataset_rec.get("files_processed") or 0) + 1
            source["files_processed"] = int(source.get("files_processed") or 0) + 1
            merge_file_payload(dataset_rec, file_payload or {})
    else:
        new_bad = result.get("bad_files") or [fallback_bad_entry(result)]
        existing_bad_paths = {item_file_path(b) for b in source.get("bad_files") or [] if isinstance(b, dict)}
        for bad in new_bad:
            if not isinstance(bad, dict):
                continue
            bad = dict(bad)
            bad.setdefault("dataset", result.get("dataset") or "")
            bad.setdefault("file_path", file_path)
            if item_file_path(bad) in existing_bad_paths:
                continue
            source.setdefault("bad_files", []).append(bad)
            existing_bad_paths.add(item_file_path(bad))

    source.setdefault("local_recovery_overlays", []).append(
        {
            "timestamp_utc": utc_now(),
            "source": "local_bad_file_recovery_generic.py",
            "tag": result.get("tag"),
            "kind": result.get("kind"),
            "process": result.get("process"),
            "shape_shift": result.get("shape_shift"),
            "candidate_reason": result.get("candidate_reason"),
            "recovery_status": status,
            "file_path": file_path,
            "recovered_payload_json": result.get("recovered_payload_json"),
            "removed_bad_file_entries": removed_bad,
            "had_attempt_in_source": had_attempt,
            "replaced_summary": replaced_summary,
        }
    )
    return finalize_source(source_path, source)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "kind",
        "process",
        "source_kind",
        "candidate_reason",
        "source_json",
        "shard_id",
        "dataset",
        "file_path",
        "permanently_skipped",
        "recovery_status",
        "events_read",
        "applied",
        "apply_status",
        "recovered_payload_json",
        "error",
    ]
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/eos/user/t/taiwoo/run3_stop/decaf")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--shift", default="nominal")
    parser.add_argument("--kinds", default="mc")
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--retry-permanent", action="store_true")
    parser.add_argument("--all-reasons", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("AUTONOMOUS_ALLHAD_FULL_CHUNK", "50000")))
    parser.add_argument("--no-apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    add_package_path(repo)
    output_dir = Path(args.output_dir)
    shard_dir = Path(args.shard_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if not shard_dir.is_absolute():
        shard_dir = repo / shard_dir
    workflow = repo / "autonomous_allhad/workflow"
    summary_path = workflow / f"local_bad_file_recovery_{args.tag}_current.json"
    tsv_path = workflow / f"local_bad_file_recovery_{args.tag}_current.tsv"
    history = load_history(summary_path)
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    workers = max(1, min(int(args.workers), 4))
    for key in THREAD_ENV:
        os.environ.setdefault(key, "1")

    candidates, collect_stats = collect_candidates(
        output_dir,
        shard_dir,
        kinds,
        include_running=bool(args.include_running),
        retry_permanent=bool(args.retry_permanent),
        access_only=not bool(args.all_reasons),
        recovered=recovered_paths(history),
    )
    start_index = max([int(r.get("index") or 0) for r in history.get("results") or []] or [0])
    for offset, candidate in enumerate(candidates, start=1):
        candidate["index"] = start_index + offset
    if args.max_files > 0:
        candidates = candidates[: args.max_files]

    previous_results = list(history.get("results") or [])
    summary: dict[str, Any] = {
        "tag": args.tag,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "output_dir": display_path(output_dir, repo),
        "shard_dir": display_path(shard_dir, repo),
        "shape_shift": args.shift,
        "kinds": sorted(kinds),
        "workers_requested": args.workers,
        "workers": workers,
        "chunk_size": args.chunk_size,
        "include_running": bool(args.include_running),
        "retry_permanent": bool(args.retry_permanent),
        "access_only": not bool(args.all_reasons),
        "dry_run": bool(args.dry_run),
        "xrd_environment": {
            "XRD_NETWORKSTACK": os.environ.get("XRD_NETWORKSTACK", "IPv4"),
            "AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT": os.environ.get("AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT", "1800"),
            "AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE": os.environ.get("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", "1"),
        },
        "collect_stats": collect_stats,
        "candidate_files": len(candidates),
        "previous_results": len(previous_results),
        "total_results_expected": len(previous_results) + len(candidates),
        "candidate_kind_counts": dict(Counter(c["kind"] for c in candidates)),
        "candidate_process_counts": dict(Counter(c.get("process", "") for c in candidates)),
        "candidate_reason_counts": dict(Counter(c.get("candidate_reason", "") for c in candidates)),
        "status_counts": dict(Counter(r.get("recovery_status") for r in previous_results)),
        "apply_status_counts": dict(Counter(r.get("apply_status", "") for r in previous_results)),
        "results": previous_results,
    }
    write_json(summary_path, summary)
    write_tsv(tsv_path, previous_results)
    if args.dry_run:
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        print(json.dumps({k: summary[k] for k in ["tag", "candidate_files", "candidate_kind_counts", "candidate_process_counts", "candidate_reason_counts", "finished_at"]}, sort_keys=True))
        return 0

    results: list[dict[str, Any]] = list(previous_results)
    if candidates:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(process_candidate, str(repo), str(shard_dir), args.tag, args.shift, args.chunk_size, candidate)
                for candidate in candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                if not args.no_apply and result.get("recovery_status") in {"recovered", "still_bad"}:
                    applied, apply_status = apply_recovery(repo, output_dir, shard_dir, result)
                    result["applied"] = bool(applied)
                    result["apply_status"] = apply_status
                results.append(result)
                results.sort(key=lambda row: int(row.get("index") or 0))
                summary["updated_at"] = utc_now()
                summary["results"] = results
                summary["status_counts"] = dict(Counter(r.get("recovery_status") for r in results))
                summary["apply_status_counts"] = dict(Counter(r.get("apply_status", "") for r in results))
                write_json(summary_path, summary)
                write_tsv(tsv_path, results)

    summary["finished_at"] = utc_now()
    summary["results"] = results
    summary["status_counts"] = dict(Counter(r.get("recovery_status") for r in results))
    summary["apply_status_counts"] = dict(Counter(r.get("apply_status", "") for r in results))
    write_json(summary_path, summary)
    write_tsv(tsv_path, results)
    print(json.dumps({k: summary[k] for k in ["tag", "candidate_files", "candidate_kind_counts", "status_counts", "apply_status_counts", "finished_at"]}, sort_keys=True))
    return 0 if not any(r.get("recovery_status") == "local_exception" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
