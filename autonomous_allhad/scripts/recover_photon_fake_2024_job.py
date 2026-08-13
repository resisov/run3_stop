#!/usr/bin/env python3
"""Recover one photon-fake sidecar locally without mutating campaign state."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from collections import Counter
from pathlib import Path

from run_photon_fake_2024_local import (
    DEFAULT_CAMPAIGN,
    DEFAULT_PYTHON,
    DEFAULT_REPO,
    prepare_runtime,
    read_jobs,
    read_json,
    release_job_caches,
    run_one,
    validate_output,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--prefilter-block-size", type=int, default=5_000)
    parser.add_argument("--record-workers", type=int, default=2)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_name):
        raise ValueError(f"unsafe job name: {args.job_name}")
    if args.chunk_size <= 0 or args.prefilter_block_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.prefilter_block_size > args.chunk_size:
        raise ValueError("--prefilter-block-size cannot exceed --chunk-size")
    if args.record_workers <= 0:
        raise ValueError("--record-workers must be positive")

    campaign = args.campaign.absolute()
    repo = args.repo.absolute()
    python = args.python.absolute()
    if not str(campaign).startswith("/eos/user/"):
        raise ValueError("campaign must be under /eos/user")
    if not python.is_file():
        raise FileNotFoundError(python)
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    if not proxy.is_file():
        raise FileNotFoundError(proxy)

    lock_path = campaign / f"recovery_{args.job_name}.lock"
    with lock_path.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another recovery owns {args.job_name}"
            ) from exc
        lock_handle.write(str(os.getpid()) + "\n")
        lock_handle.flush()

        manifest = read_json(campaign / "manifest.json")
        runtime = prepare_runtime(campaign, manifest)
        matching = [
            job
            for job in read_jobs(campaign, manifest)
            if job["name"] == args.job_name
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"expected exactly one job named {args.job_name}, found "
                f"{len(matching)}"
            )
        job = matching[0]
        valid, validation = validate_output(job)
        if valid:
            print(
                json.dumps(
                    {
                        "name": job["name"],
                        "process": job["process"],
                        "status": "skipped_valid",
                        "validation": validation,
                    },
                    sort_keys=True,
                )
            )
            return 0

        scratch_root = (
            Path("/tmp")
            / f"{campaign.name}_recovery_{args.job_name}"
        )
        scratch_root.mkdir(parents=True, exist_ok=True)
        shared_xrd_cache = scratch_root / "shared_xrd"
        shared_xrd_cache.mkdir(parents=True, exist_ok=True)
        remaining_uses = Counter(
            str(record["file_path"]) for record in job["records"]
        )
        result = run_one(
            job,
            campaign,
            repo,
            runtime,
            python,
            args.chunk_size,
            args.prefilter_block_size,
            scratch_root,
            shared_xrd_cache,
            args.record_workers,
        )
        release_job_caches(job, remaining_uses, shared_xrd_cache)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
