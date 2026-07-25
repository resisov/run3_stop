#!/usr/bin/env python3
"""Run one-file 2024 shape-histogram shards concurrently on lxplus."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_PYTHON = Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python")
EXPECTED_SECTIONS = {
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
    marker = runtime / "runtime_manifest.json"
    expected = {
        "worker_bundle_sha256": manifest["worker_bundle"]["sha256"],
        "payload_bundle_sha256": manifest["payload_bundle"]["sha256"],
        "shard_bundle_sha256": manifest["shard_bundle"]["sha256"],
    }
    if marker.is_file():
        observed = read_json(marker)
        if all(observed.get(key) == value for key, value in expected.items()):
            return runtime
        raise RuntimeError("existing local runtime does not match campaign bundle checksums")
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"refusing non-empty unvalidated runtime: {runtime}")
    for key in ("worker_bundle", "payload_bundle", "shard_bundle"):
        entry = manifest[key]
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"{key} is missing or has a checksum mismatch: {path}")
    for key in ("worker_bundle", "payload_bundle"):
        entry = manifest[key]
        path = Path(entry["path"])
        safe_extract(path, runtime)
    write_json(
        marker,
        {
            **expected,
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return runtime


def read_jobs(campaign: Path, manifest: dict[str, Any], runtime: Path) -> list[dict[str, Any]]:
    shard_payloads: dict[str, dict[str, Any]] = {}
    shard_bundle = Path(manifest["shard_bundle"]["path"])
    with tarfile.open(shard_bundle, "r:gz") as archive:
        for member in archive:
            basename = PurePosixPath(member.name).name
            if not member.isfile() or not basename.startswith("shape_input_"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member.name} from shard bundle")
            payload = json.load(extracted)
            records = list(payload.get("records") or [])
            if len(records) != 1 or int(payload.get("records_per_shard") or 0) != 1:
                raise RuntimeError(f"{basename} is not a one-file shard")
            if basename in shard_payloads:
                raise RuntimeError(f"duplicate shard basename in bundle: {basename}")
            shard_payloads[basename] = payload

    arguments = campaign / "condor/arguments.txt"
    jobs = []
    for line_number, line in enumerate(arguments.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(f"invalid arguments line {line_number}: {line}")
        name, shard_name, histogram, metadata = fields
        payload = shard_payloads.pop(shard_name, None)
        if payload is None:
            raise RuntimeError(f"arguments reference missing shard: {shard_name}")
        records = list(payload.get("records") or [])
        jobs.append(
            {
                "name": name,
                "shard_name": shard_name,
                "shard_payload": payload,
                "record_digest": payload.get("record_digest"),
                "histogram": Path(histogram),
                "metadata": Path(metadata),
                "input_file": records[0].get("file_path"),
                "dataset": records[0].get("dataset"),
            }
        )
    if shard_payloads:
        raise RuntimeError(f"{len(shard_payloads)} shard bundle entries are not referenced")
    expected = int((manifest.get("jobs") or {}).get("total") or 0)
    if not jobs or len(jobs) != expected:
        raise RuntimeError(f"job count mismatch: arguments={len(jobs)} manifest={expected}")
    if len({str(job["input_file"]) for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate source files in one-file campaign")
    return jobs


def validate_output(job: dict[str, Any]) -> tuple[bool, str]:
    histogram = job["histogram"]
    metadata = job["metadata"]
    if not histogram.is_file() or histogram.stat().st_size == 0:
        return False, "histogram_missing_or_empty"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        return False, "metadata_missing_or_empty"
    try:
        meta = read_json(metadata)
        with gzip.open(histogram, "rt", encoding="utf-8") as source:
            payload = json.load(source)
    except Exception as exc:
        return False, f"decode_failed:{type(exc).__name__}:{str(exc)[:160]}"
    summary = meta.get("summary") or {}
    errors = []
    if meta.get("status") != "complete" or payload.get("status") != "complete":
        errors.append("status")
    if meta.get("source_record_digest") != job["record_digest"]:
        errors.append("record_digest")
    if int(summary.get("files_attempted") or 0) != 1:
        errors.append("files_attempted")
    if int(summary.get("files_processed") or 0) != 1:
        errors.append("files_processed")
    if summary.get("bad_files"):
        errors.append("bad_files")
    if int(meta.get("variation_count") or 0) != 40:
        errors.append("variation_count")
    if len(payload.get("variations") or []) != 40:
        errors.append("payload_variations")
    if set((payload.get("output_policy") or {}).get("sections") or []) != EXPECTED_SECTIONS:
        errors.append("sections")
    if sha256(histogram) != meta.get("histogram_sha256"):
        errors.append("checksum")
    btag = summary.get("btag_sf_status") or {}
    if len(btag) != 40 or any(not (item or {}).get("applied") for item in btag.values()):
        errors.append("btag")
    lowdm = payload.get("lowdm_region_policy") or {}
    if "42-bin" not in str(lowdm.get("search_bins") or ""):
        errors.append("lowdm_42bin")
    return (not errors), ",".join(errors) if errors else "complete"


def run_one(
    job: dict[str, Any],
    campaign: Path,
    runtime: Path,
    python: Path,
    chunk_size: int,
    proxy: Path,
) -> dict[str, Any]:
    valid, reason = validate_output(job)
    if valid:
        return {"name": job["name"], "status": "skipped_valid", "reason": reason}

    task = runtime / "tasks" / job["name"]
    if task.exists():
        shutil.rmtree(task)
    task.mkdir(parents=True)
    shard_path = task / job["shard_name"]
    write_json(shard_path, job["shard_payload"])
    temporary_histogram = task / "out.json.gz"
    temporary_metadata = task / "out.meta.json"
    stdout_path = campaign / "logs" / f"{job['name']}.local.out"
    stderr_path = campaign / "logs" / f"{job['name']}.local.err"
    command = [
        str(python),
        "-u",
        "-m",
        "autonomous_allhad.shape_histogram_2024_worker",
        "--shard",
        str(shard_path),
        "--output",
        str(temporary_histogram),
        "--metadata-output",
        str(temporary_metadata),
        "--variation-group",
        "all",
        "--chunk-size",
        str(chunk_size),
        "--record-workers",
        "1",
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
            "X509_USER_PROXY": str(proxy),
            "AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0",
            "AUTONOMOUS_ALLHAD_XRD_CACHE": str(task / "xrd"),
            "TMPDIR": str(task / "tmp"),
            "TMP": str(task / "tmp"),
            "TEMP": str(task / "tmp"),
            "XDG_CACHE_HOME": str(task / "cache"),
            "MPLCONFIGDIR": str(task / "mpl"),
            "NUMBA_CACHE_DIR": str(task / "cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(task / "cache" / "pycache"),
        }
    )
    for directory in ("xrd", "tmp", "cache", "mpl"):
        (task / directory).mkdir(parents=True, exist_ok=True)
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
    result = {
        "name": job["name"],
        "dataset": job["dataset"],
        "input_file": job["input_file"],
        "returncode": process.returncode,
        "wall_time_s": round(time.time() - started, 3),
    }
    if temporary_histogram.is_file():
        job["histogram"].parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_histogram, job["histogram"])
    if temporary_metadata.is_file():
        job["metadata"].parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_metadata, job["metadata"])
    valid, reason = validate_output(job)
    result["status"] = "complete" if process.returncode == 0 and valid else "failed"
    result["validation"] = reason
    if valid:
        shutil.rmtree(task)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    campaign = args.campaign
    if not campaign.is_absolute() or not str(campaign).startswith("/eos/user/"):
        raise ValueError("campaign must be an absolute EOS path")
    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("year") != 2024 or int(manifest.get("variation_count") or 0) != 40:
        raise RuntimeError("campaign is not the adopted 2024 all-40 variation campaign")
    if int((manifest.get("jobs") or {}).get("source_files_per_job") or 0) != 1:
        raise RuntimeError("campaign is not one source file per job")
    if "v3_lowdm_relaxed" not in str(manifest.get("source_nominal_campaign") or ""):
        raise RuntimeError("campaign is not based on the new nominal v3 selection")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    proxy = args.repo / f"analysis/proxy/x509up_u{os.getuid()}"
    if not proxy.is_file():
        raise FileNotFoundError(proxy)

    lock_path = campaign / "local_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another local runner already owns this campaign") from exc
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()

    runtime = prepare_runtime(campaign, manifest)
    jobs = read_jobs(campaign, manifest, runtime)
    if args.max_jobs is not None:
        jobs = jobs[: max(0, args.max_jobs)]
    valid_at_start = []
    pending = []
    outputs_dir = campaign / "outputs"
    existing_output_names = (
        {path.name for path in outputs_dir.iterdir() if path.is_file()}
        if outputs_dir.is_dir()
        else set()
    )
    for job in jobs:
        has_pair = (
            job["histogram"].name in existing_output_names
            and job["metadata"].name in existing_output_names
        )
        valid, _reason = validate_output(job) if has_pair else (False, "missing_pair")
        (valid_at_start if valid else pending).append(job)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    work: queue.Queue[dict[str, Any]] = queue.Queue()
    for job in pending:
        work.put(job)
    results: list[dict[str, Any]] = []
    active: set[str] = set()
    state_lock = threading.Lock()
    status_path = campaign / "local_status.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    last_write = 0.0

    def snapshot(force: bool = False) -> None:
        nonlocal last_write
        now = time.time()
        if not force and now - last_write < 30:
            return
        completed = sum(item["status"] == "complete" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        write_json(
            status_path,
            {
                "schema_version": "shape_histogram_2024_local_parallel_status_v1",
                "status": (
                    "stopping"
                    if stop.is_set()
                    else "running"
                    if active or not work.empty()
                    else "complete_with_failures"
                    if failed
                    else "complete"
                ),
                "campaign": str(campaign),
                "controller_pid": os.getpid(),
                "host": os.uname().nodename,
                "workers": min(args.workers, max(1, len(pending))),
                "source_files_per_task": 1,
                "variation_count": 40,
                "jobs_selected": len(jobs),
                "valid_at_start": len(valid_at_start),
                "completed_this_run": completed,
                "failed_this_run": failed,
                "active": sorted(active),
                "queued": work.qsize(),
                "started_at": started_at,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "failures": [item for item in results if item["status"] == "failed"],
            },
        )
        last_write = now

    def worker_loop() -> None:
        while not stop.is_set():
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            with state_lock:
                active.add(job["name"])
                snapshot()
            try:
                result = run_one(
                    job,
                    campaign,
                    runtime,
                    args.python,
                    args.chunk_size,
                    proxy,
                )
            except Exception as exc:
                result = {
                    "name": job["name"],
                    "dataset": job["dataset"],
                    "input_file": job["input_file"],
                    "status": "failed",
                    "validation": f"runner_exception:{type(exc).__name__}:{str(exc)[:300]}",
                }
            with state_lock:
                active.discard(job["name"])
                results.append(result)
                snapshot(force=result["status"] == "failed")
            work.task_done()

    manifest["execution"] = {
        "mode": "lxplus_local_parallel",
        "condor_submission": False,
        "source_files_per_task": 1,
        "workers": min(args.workers, max(1, len(pending))),
        "controller_pid": os.getpid(),
        "host": os.uname().nodename,
        "started_at": started_at,
        "status_file": str(status_path),
    }
    manifest["status"] = "running_local"
    write_json(manifest_path, manifest)
    snapshot(force=True)

    threads = [
        threading.Thread(target=worker_loop, name=f"shape-local-{index:02d}")
        for index in range(min(args.workers, max(1, len(pending))))
    ]
    monitor_done = threading.Event()

    def monitor_loop() -> None:
        while not monitor_done.wait(30):
            with state_lock:
                snapshot(force=True)

    monitor = threading.Thread(
        target=monitor_loop,
        name="shape-local-monitor",
        daemon=True,
    )
    monitor.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    monitor_done.set()
    monitor.join()
    snapshot(force=True)
    final = read_json(status_path)
    manifest = read_json(manifest_path)
    manifest["status"] = final["status"]
    manifest["execution"]["completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    manifest["execution"]["result"] = final
    write_json(manifest_path, manifest)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
