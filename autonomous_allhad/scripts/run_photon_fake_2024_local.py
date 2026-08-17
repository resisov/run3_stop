#!/usr/bin/env python3
"""Run the photon-fake sidecar campaign locally on lxplus with 12 workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/photon_fake_2024_local2400k_v3_20260726"
)
DEFAULT_PYTHON = Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python")
PROCESS_TIERS = (
    ("EGamma", "QCD"),
    ("GJ",),
    ("DY", "TT", "WtoLNu", "ST", "VV", "Zto2Nu"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
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


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
            ):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def prepare_runtime(campaign: Path, manifest: dict[str, Any]) -> Path:
    runtime = campaign / "local_runtime"
    worker = Path(manifest["bundles"]["worker"]["path"])
    payload = Path(manifest["bundles"]["payload"]["path"])
    expected = {
        "worker_sha256": str(manifest["bundles"]["worker"]["sha256"]),
        "payload_sha256": str(manifest["bundles"]["payload"]["sha256"]),
    }
    marker = runtime / "runtime_manifest.json"
    if marker.is_file():
        observed = read_json(marker)
        if all(observed.get(key) == value for key, value in expected.items()):
            return runtime
        raise RuntimeError("existing photon-fake runtime checksum differs")
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"refusing non-empty unvalidated runtime: {runtime}")
    for path, checksum in (
        (worker, expected["worker_sha256"]),
        (payload, expected["payload_sha256"]),
    ):
        if not path.is_file() or sha256(path) != checksum:
            raise RuntimeError(f"runtime bundle missing or checksum mismatch: {path}")
    safe_extract(worker, runtime)
    safe_extract(payload, runtime)
    write_json(
        marker,
        {
            **expected,
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return runtime


def read_jobs(campaign: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    local_bundle_info = manifest.get("local_job_bundle")
    if local_bundle_info:
        bundle = Path(str(local_bundle_info["path"]))
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        if sha256(bundle) != str(local_bundle_info["sha256"]):
            raise RuntimeError("local job bundle checksum mismatch")
        with gzip.open(bundle, "rt", encoding="utf-8") as handle:
            packed = json.load(handle)
        if packed.get("schema_version") != "photon_fake_2024_local_jobs_v1":
            raise RuntimeError("unsupported photon-fake local job bundle")
        jobs: list[dict[str, Any]] = []
        for raw in packed.get("jobs") or []:
            shard = dict(raw.get("shard") or {})
            process = str(shard.get("process_group") or "")
            records = list(shard.get("records") or [])
            if process != str(raw.get("process") or "") or not records:
                raise RuntimeError("invalid local photon-fake job")
            jobs.append(
                {
                    "name": str(raw["name"]),
                    "process": process,
                    "shard": None,
                    "shard_payload": shard,
                    "shard_basename": str(raw["shard_basename"]),
                    "histogram": Path(str(raw["histogram"])),
                    "metadata": Path(str(raw["metadata"])),
                    "log_dir": Path(str(raw["log_dir"])),
                    "record_digest": str(shard.get("record_digest") or ""),
                    "expected_files": len(records),
                    "expected_events": int(shard.get("expected_events") or 0),
                    "expected_paths": sorted(
                        str(record["file_path"]) for record in records
                    ),
                    "records": records,
                }
            )
        if len(jobs) != int(manifest.get("jobs") or -1):
            raise RuntimeError(
                f"local bundle/manifest job count mismatch: {len(jobs)} vs "
                f"{manifest.get('jobs')}"
            )
        if len({job["name"] for job in jobs}) != len(jobs):
            raise RuntimeError("duplicate photon-fake local job names")
        return jobs

    jobs: list[dict[str, Any]] = []
    arguments = Path(manifest["condor"]["arguments"])
    for line_number, raw in enumerate(arguments.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 6:
            raise RuntimeError(f"invalid arguments line {line_number}: {raw}")
        name, shard_path, shard_basename, histogram, metadata, log_dir = fields
        shard = read_json(Path(shard_path))
        process = str(shard.get("process_group") or "")
        records = list(shard.get("records") or [])
        if not process or not records:
            raise RuntimeError(f"invalid balanced shard: {shard_path}")
        jobs.append(
            {
                "name": name,
                "process": process,
                "shard": Path(shard_path),
                "shard_basename": shard_basename,
                "histogram": Path(histogram),
                "metadata": Path(metadata),
                "log_dir": Path(log_dir),
                "record_digest": str(shard.get("record_digest") or ""),
                "expected_files": len(records),
                "expected_events": int(shard.get("expected_events") or 0),
                "expected_paths": sorted(
                    str(record["file_path"]) for record in records
                ),
                "records": records,
            }
        )
    if len(jobs) != int(manifest.get("jobs") or -1):
        raise RuntimeError(
            f"argument/manifest job count mismatch: {len(jobs)} vs "
            f"{manifest.get('jobs')}"
        )
    if len({job["name"] for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate photon-fake job names")
    return jobs


def validate_output(job: dict[str, Any]) -> tuple[bool, str]:
    histogram = job["histogram"]
    metadata = job["metadata"]
    if not histogram.is_file() or histogram.stat().st_size <= 0:
        return False, "histogram_missing_or_empty"
    if not metadata.is_file() or metadata.stat().st_size <= 0:
        return False, "metadata_missing_or_empty"
    try:
        meta = read_json(metadata)
        with gzip.open(histogram, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return False, f"decode_failed:{type(exc).__name__}:{str(exc)[:160]}"
    summary = meta.get("summary") or {}
    observed_paths = sorted(
        str(record.get("file_path") or "")
        for record in summary.get("file_records") or []
    )
    errors = []
    if meta.get("status") != "complete" or payload.get("status") != "complete":
        errors.append("status")
    if str(meta.get("source_record_digest") or "") != job["record_digest"]:
        errors.append("record_digest")
    if int(summary.get("files_attempted") or 0) != job["expected_files"]:
        errors.append("files_attempted")
    if int(summary.get("files_processed") or 0) != job["expected_files"]:
        errors.append("files_processed")
    if int(summary.get("events_read") or 0) != job["expected_events"]:
        errors.append("events_read")
    if observed_paths != job["expected_paths"]:
        errors.append("file_coverage")
    if summary.get("bad_files"):
        errors.append("bad_files")
    if int(summary.get("target_cutbased_mismatch_objects") or 0) != 0:
        errors.append("photon_id_mismatch")
    if sha256(histogram) != str(meta.get("histogram_sha256") or ""):
        errors.append("checksum")
    return (not errors), ",".join(errors) if errors else "complete"


def publish(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial.{os.getpid()}")
    try:
        shutil.copy2(source, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def run_one(
    job: dict[str, Any],
    campaign: Path,
    repo: Path,
    runtime: Path,
    python: Path,
    chunk_size: int,
    prefilter_block_size: int,
    scratch_root: Path,
    shared_xrd_cache: Path,
    record_workers: int = 1,
) -> dict[str, Any]:
    valid, reason = validate_output(job)
    if valid:
        return {
            "name": job["name"],
            "process": job["process"],
            "status": "skipped_valid",
            "validation": reason,
            "events_read": job["expected_events"],
        }
    task = scratch_root / job["name"]
    if task.exists():
        shutil.rmtree(task)
    for directory in ("tmp", "cache", "mpl", "xrd"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    temporary_histogram = task / "out.json.gz"
    temporary_metadata = task / "out.meta.json"
    shard_path = job["shard"]
    if shard_path is None:
        shard_path = task / job["shard_basename"]
        write_json(shard_path, job["shard_payload"])
    stdout_path = job["log_dir"] / f"{job['name']}.local.out"
    stderr_path = job["log_dir"] / f"{job['name']}.local.err"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-u",
        "-m",
        "autonomous_allhad.photon_fake_2024_worker",
        "--shard",
        str(shard_path),
        "--output",
        str(temporary_histogram),
        "--metadata-output",
        str(temporary_metadata),
        "--chunk-size",
        str(chunk_size),
        "--prefilter-block-size",
        str(prefilter_block_size),
        "--record-workers",
        str(record_workers),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(runtime),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "XRD_NETWORKSTACK": "IPv4",
            "XRD_REQUESTTIMEOUT": "180",
            "XRD_REDIRECTLIMIT": "10",
            "X509_USER_PROXY": str(
                repo / f"analysis/proxy/x509up_u{os.getuid()}"
            ),
            "AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0",
            "AUTONOMOUS_ALLHAD_XRD_CACHE": str(shared_xrd_cache),
            "AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE": "1",
            "AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE": "0",
            "TMPDIR": str(task / "tmp"),
            "TMP": str(task / "tmp"),
            "TEMP": str(task / "tmp"),
            "XDG_CACHE_HOME": str(task / "cache"),
            "MPLCONFIGDIR": str(task / "mpl"),
            "NUMBA_CACHE_DIR": str(task / "cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(task / "cache" / "pycache"),
        }
    )
    started = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run(
            command,
            cwd=runtime,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    if temporary_histogram.is_file():
        publish(temporary_histogram, job["histogram"])
    if temporary_metadata.is_file():
        publish(temporary_metadata, job["metadata"])
    valid, reason = validate_output(job)
    result = {
        "name": job["name"],
        "process": job["process"],
        "status": "complete" if process.returncode == 0 and valid else "failed",
        "returncode": process.returncode,
        "validation": reason,
        "wall_time_s": round(time.time() - started, 3),
        "events_read": job["expected_events"] if valid else 0,
    }
    if valid:
        shutil.rmtree(task)
    return result


def xrd_cache_paths(file_path: str, shared_xrd_cache: Path) -> tuple[Path, Path]:
    raw_name = file_path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    raw_name = raw_name or "cached.root"
    if not raw_name.endswith(".root"):
        raw_name += ".root"
    digest = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    cached = shared_xrd_cache / f"{digest}_{raw_name}"
    return cached, cached.with_suffix(f"{cached.suffix}.lock")


def release_job_caches(
    job: dict[str, Any],
    remaining_uses: Counter[str],
    shared_xrd_cache: Path,
) -> None:
    for record in job["records"]:
        file_path = str(record["file_path"])
        remaining_uses[file_path] -= 1
        if remaining_uses[file_path] < 0:
            raise RuntimeError(f"negative shared-cache reference count: {file_path}")
        if remaining_uses[file_path] == 0:
            cached, lock = xrd_cache_paths(file_path, shared_xrd_cache)
            cached.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


def interleave(jobs: list[dict[str, Any]], processes: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped = {
        process: [job for job in jobs if job["process"] == process]
        for process in processes
    }
    output: list[dict[str, Any]] = []
    index = 0
    while any(index < len(grouped[process]) for process in processes):
        for process in processes:
            if index < len(grouped[process]):
                output.append(grouped[process][index])
        index += 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--prefilter-block-size", type=int, default=5_000)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.workers != 12:
        raise ValueError("this adopted local campaign must use exactly 12 workers")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.prefilter_block_size <= 0:
        raise ValueError("--prefilter-block-size must be positive")
    if args.prefilter_block_size > args.chunk_size:
        raise ValueError("--prefilter-block-size cannot exceed --chunk-size")
    campaign = args.campaign.absolute()
    repo = args.repo.absolute()
    if not str(campaign).startswith("/eos/user/"):
        raise ValueError("campaign must be under /eos/user")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    if not proxy.is_file():
        raise FileNotFoundError(proxy)

    lock_path = campaign / "local_runner.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another photon-fake local runner owns this campaign") from exc
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()

    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    runtime = prepare_runtime(campaign, manifest)
    jobs = read_jobs(campaign, manifest)
    valid_at_start = []
    pending = []
    for job in jobs:
        valid, _ = validate_output(job)
        (valid_at_start if valid else pending).append(job)
    if args.max_jobs is not None:
        pending = pending[: max(0, args.max_jobs)]

    scratch_root = Path("/tmp") / f"{campaign.name}_local12"
    scratch_root.mkdir(parents=True, exist_ok=True)
    shared_xrd_cache = scratch_root / "shared_xrd"
    shared_xrd_cache.mkdir(parents=True, exist_ok=True)
    remaining_uses: Counter[str] = Counter(
        str(record["file_path"])
        for job in pending
        for record in job["records"]
    )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    results: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    status_path = campaign / "local_status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def snapshot(active: list[str], tier: list[str], queued: int) -> None:
        completed = Counter(
            item["process"]
            for item in results
            if item["status"] in {"complete", "skipped_valid"}
        )
        failed = Counter(
            item["process"] for item in results if item["status"] == "failed"
        )
        write_json(
            status_path,
            {
                "schema_version": "photon_fake_2024_local12_status_v1",
                "status": (
                    "stopping"
                    if stop.is_set()
                    else "running"
                    if active
                    else "complete_with_failures"
                    if failed
                    else "running"
                ),
                "campaign": str(campaign),
                "controller_pid": os.getpid(),
                "host": os.uname().nodename,
                "python": str(args.python),
                "workers": args.workers,
                "chunk_size": args.chunk_size,
                "prefilter_block_size": args.prefilter_block_size,
                "current_tier": tier,
                "active": active,
                "queued_in_current_tier": queued,
                "valid_at_start": len(valid_at_start),
                "completed_by_process": dict(sorted(completed.items())),
                "failed_by_process": dict(sorted(failed.items())),
                "events_read_this_run": sum(
                    int(item.get("events_read") or 0) for item in results
                ),
                "started_at": started_at,
                "updated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "failures": [
                    item for item in results if item["status"] == "failed"
                ][-50:],
            },
        )

    manifest["status"] = "running_local"
    manifest["execution"] = {
        "mode": "lxplus_miniconda_py38_local12",
        "python": str(args.python),
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "prefilter_block_size": args.prefilter_block_size,
        "process_tiers": [list(tier) for tier in PROCESS_TIERS],
        "condor_cluster_removed": 16331223,
        "started_at": started_at,
        "status_file": str(status_path),
    }
    write_json(manifest_path, manifest)

    for tier in PROCESS_TIERS:
        if stop.is_set():
            break
        phase = interleave(pending, tier)
        if not phase:
            continue
        active: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_map: dict[
                concurrent.futures.Future[dict[str, Any]],
                dict[str, Any],
            ] = {}
            iterator = iter(phase)
            phase_done = 0

            def submit_next() -> bool:
                if stop.is_set():
                    return False
                try:
                    job = next(iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    run_one,
                    job,
                    campaign,
                    repo,
                    runtime,
                    args.python,
                    args.chunk_size,
                    args.prefilter_block_size,
                    scratch_root,
                    shared_xrd_cache,
                )
                future_map[future] = job
                active.add(job["name"])
                return True

            for _ in range(min(args.workers, len(phase))):
                submit_next()
            with state_lock:
                snapshot(
                    sorted(active),
                    list(tier),
                    max(0, len(phase) - len(active)),
                )
            monitor_stop = threading.Event()

            def monitor_phase() -> None:
                while not monitor_stop.wait(30):
                    with state_lock:
                        snapshot(
                            sorted(active),
                            list(tier),
                            max(0, len(phase) - phase_done - len(active)),
                        )

            monitor = threading.Thread(
                target=monitor_phase,
                name="photon-fake-local-monitor",
                daemon=True,
            )
            monitor.start()
            while future_map:
                done, _ = concurrent.futures.wait(
                    future_map,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    job = future_map.pop(future)
                    active.discard(job["name"])
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "name": job["name"],
                            "process": job["process"],
                            "status": "failed",
                            "validation": (
                                f"runner_exception:{type(exc).__name__}:"
                                f"{str(exc)[:300]}"
                            ),
                            "events_read": 0,
                        }
                    with state_lock:
                        results.append(result)
                        release_job_caches(
                            job,
                            remaining_uses,
                            shared_xrd_cache,
                        )
                    phase_done += 1
                    submit_next()
                    with state_lock:
                        snapshot(
                            sorted(active),
                            list(tier),
                            max(0, len(phase) - phase_done - len(active)),
                        )
            monitor_stop.set()
            monitor.join()
        if any(
            item["status"] == "failed" and item["process"] in tier
            for item in results
        ):
            break

    expected_names = {job["name"] for job in jobs}
    valid_final = 0
    for job in jobs:
        valid, _ = validate_output(job)
        valid_final += int(valid)
    failures = [item for item in results if item["status"] == "failed"]
    final_status = (
        "complete"
        if valid_final == len(expected_names)
        else "stopped"
        if stop.is_set()
        else "complete_with_failures"
        if failures
        else "partial_priority_complete"
    )
    final = read_json(status_path) if status_path.is_file() else {}
    final.update(
        {
            "status": final_status,
            "valid_outputs": valid_final,
            "expected_outputs": len(expected_names),
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
    )
    write_json(status_path, final)
    manifest = read_json(manifest_path)
    manifest["status"] = final_status
    manifest["execution"]["completed_at"] = final["completed_at"]
    manifest["execution"]["result"] = final
    write_json(manifest_path, manifest)
    if not any(shared_xrd_cache.iterdir()):
        shared_xrd_cache.rmdir()
    if scratch_root.is_dir() and not any(scratch_root.iterdir()):
        scratch_root.rmdir()
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final_status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
