#!/usr/bin/env python3
"""Run one local recovery attempt for each adopted 2024 bad MC file."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uproot


TARGETS = ("01952", "02841", "05312", "07593")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def source_shard(archive: tarfile.TarFile, shard_id: str) -> dict[str, Any]:
    member = archive.getmember(f"shards/mc_shard_{shard_id}.json")
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member.name)
    payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{member.name}: expected JSON object")
    return payload


def validate_output(root_path: Path, json_path: Path) -> dict[str, Any]:
    if not root_path.exists() or root_path.stat().st_size <= 0:
        raise ValueError("recovery ROOT is missing or empty")
    if not json_path.exists() or json_path.stat().st_size <= 0:
        raise ValueError("recovery JSON is missing or empty")
    metadata = read_json(json_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"recovery status={metadata.get('status')!r}")
    if int(metadata.get("files_attempted") or 0) != 1:
        raise ValueError("recovery files_attempted is not 1")
    if int(metadata.get("files_processed") or 0) != 1:
        raise ValueError("recovery files_processed is not 1")
    if metadata.get("bad_files"):
        raise ValueError("recovery bad_files is nonempty")
    with uproot.open(root_path) as root_file:
        if "Events" not in root_file:
            raise ValueError("recovery Events tree is missing")
        entries = int(root_file["Events"].num_entries)
    if entries != int(metadata.get("events_written") or 0):
        raise ValueError(
            f"recovery ROOT entries={entries}, metadata events_written={metadata.get('events_written')}"
        )
    return {
        "status": "complete",
        "events_read": int(metadata.get("events_read") or 0),
        "events_written": entries,
        "root_bytes": root_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
        "record_digest": metadata.get("record_digest"),
    }


def run_one(
    target: dict[str, Any],
    python: Path,
    runtime: Path,
    output_dir: Path,
    proxy: Path,
) -> dict[str, Any]:
    shard_id = target["shard_id"]
    shard_path = Path(target["recovery_shard"])
    root_path = output_dir / f"{shard_id}.root"
    json_path = output_dir / f"{shard_id}.json"
    stdout_path = output_dir / f"{shard_id}.out"
    stderr_path = output_dir / f"{shard_id}.err"
    task_runtime = output_dir / "runtime" / shard_id
    task_runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": str(task_runtime / "tmp"),
            "TMP": str(task_runtime / "tmp"),
            "TEMP": str(task_runtime / "tmp"),
            "XDG_CACHE_HOME": str(task_runtime / "cache"),
            "MPLCONFIGDIR": str(task_runtime / "mplconfig"),
            "NUMBA_CACHE_DIR": str(task_runtime / "cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(task_runtime / "cache" / "pycache"),
            "AUTONOMOUS_ALLHAD_XRD_CACHE": str(task_runtime / "xrd"),
            "AUTONOMOUS_ALLHAD_FRAGMENT_DIR": str(task_runtime / "fragments"),
            "AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0",
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
            "PYTHONPATH": str(runtime),
        }
    )
    for path in (
        Path(env["TMPDIR"]),
        Path(env["XDG_CACHE_HOME"]),
        Path(env["MPLCONFIGDIR"]),
        Path(env["AUTONOMOUS_ALLHAD_XRD_CACHE"]),
        Path(env["AUTONOMOUS_ALLHAD_FRAGMENT_DIR"]),
    ):
        path.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-u",
        "-m",
        "autonomous_allhad.intermediate_2024_worker",
        "--repo",
        str(runtime),
        "--shard",
        str(shard_path),
        "--output",
        str(root_path),
        "--metadata-output",
        str(json_path),
        "--shift",
        "nominal",
        "--record-workers",
        "1",
    ]
    started = utc_now()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(
            command,
            cwd=runtime,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    result = {
        "shard_id": shard_id,
        "parent_shard": target["parent_shard"],
        "bad_file": target["bad_file"],
        "dataset": target["dataset"],
        "process": target["process"],
        "attempt_number": 1,
        "started_at": started,
        "completed_at": utc_now(),
        "return_code": proc.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "root_output": str(root_path),
        "json_output": str(json_path),
    }
    if proc.returncode == 0:
        try:
            result["validation"] = validate_output(root_path, json_path)
            result["status"] = "complete"
        except Exception as exc:
            result["status"] = "failed_validation"
            result["error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["status"] = "failed_execution"
        try:
            result["stderr_tail"] = stderr_path.read_text(errors="replace")[-4000:]
        except Exception:
            result["stderr_tail"] = ""
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--proxy", required=True, type=Path)
    parser.add_argument("--setup-failure-reference", default=None)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS),
        help="Five-digit mc_shard suffixes to attempt exactly once (default: adopted original four).",
    )
    args = parser.parse_args()
    args.campaign = args.campaign.absolute()
    args.output_dir = args.output_dir.absolute()
    args.python = args.python.absolute()
    args.proxy = args.proxy.absolute()
    target_suffixes = list(dict.fromkeys(args.targets))
    if not target_suffixes or any(len(item) != 5 or not item.isdigit() for item in target_suffixes):
        raise ValueError("--targets entries must be unique five-digit mc_shard suffixes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "attempt_manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        raise RuntimeError(
            f"one-attempt recovery already recorded with status={previous.get('status')!r}; refusing rerun"
        )
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    if not args.proxy.exists():
        raise FileNotFoundError(args.proxy)

    runtime = args.output_dir / "runtime_bundle"
    if runtime.exists():
        raise RuntimeError(f"recovery runtime already exists: {runtime}; refusing to overwrite")
    safe_extract(args.campaign / "bundles/objectcorr_2024_worker.tgz", runtime)
    safe_extract(args.campaign / "bundles/objectcorr_2024_payloads.tgz", runtime)

    targets: list[dict[str, Any]] = []
    shard_archive_path = args.campaign / "bundles/fullselection_shards.tgz"
    with tarfile.open(shard_archive_path, "r:gz") as archive:
        for suffix in target_suffixes:
            parent_name = f"mc_shard_{suffix}"
            metadata = read_json(args.campaign / "outputs" / "nominal" / f"{parent_name}.json")
            bad_files = metadata.get("bad_files") or []
            if len(bad_files) != 1:
                raise ValueError(f"{parent_name}: expected exactly one bad file, found {len(bad_files)}")
            bad = bad_files[0]
            original = source_shard(archive, suffix)
            matches = [
                record
                for record in original.get("records") or []
                if record.get("file_path") == bad.get("file_path")
            ]
            if len(matches) != 1:
                raise ValueError(f"{parent_name}: bad file does not uniquely match source record")
            recovery_id = f"mc_badfile_recovery_{suffix}"
            records = matches
            recovery_payload = {
                "schema_version": original.get("schema_version"),
                "shard_id": recovery_id,
                "record_digest": record_digest(records),
                "record_group": "mc",
                "records_per_shard": 1,
                "parent_shard": parent_name,
                "recovery_policy": "one local attempt for the single recorded bad file",
                "records": records,
            }
            shard_path = args.output_dir / "shards" / f"{recovery_id}.json"
            write_json_atomic(shard_path, recovery_payload)
            targets.append(
                {
                    "shard_id": recovery_id,
                    "parent_shard": parent_name,
                    "recovery_shard": str(shard_path),
                    "bad_file": bad["file_path"],
                    "dataset": bad.get("dataset"),
                    "process": bad.get("process"),
                    "original_error": bad.get("concise_error"),
                    "original_exception_type": bad.get("exception_type"),
                }
            )

    manifest = {
        "schema_version": "bad_file_local_recovery_once_2024_v1",
        "status": "running",
        "attempt_count_per_bad_file": 1,
        "started_at": utc_now(),
        "campaign": str(args.campaign),
        "python": str(args.python),
        "proxy": str(args.proxy),
        "setup_failure_reference_not_counted_as_input_attempt": args.setup_failure_reference,
        "requested_target_suffixes": target_suffixes,
        "targets": targets,
        "results": [],
    }
    write_json_atomic(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = [
            pool.submit(run_one, target, args.python, runtime, args.output_dir, args.proxy)
            for target in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['shard_id']}: status={result['status']} rc={result['return_code']}",
                flush=True,
            )
    results.sort(key=lambda item: item["shard_id"])
    completed = sum(item.get("status") == "complete" for item in results)
    manifest.update(
        {
            "status": "complete" if completed == len(targets) else "complete_with_failures",
            "completed_at": utc_now(),
            "successful_recoveries": completed,
            "failed_recoveries": len(targets) - completed,
            "results": results,
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "successful_recoveries": manifest["successful_recoveries"],
                "failed_recoveries": manifest["failed_recoveries"],
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
