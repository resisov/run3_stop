from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import real_subset_worker as baseline
from .shape_histogram_2024_worker import FINAL_SHAPE_NUISANCES
from .shape_histogram_pair_2024 import (
    combine_pair_payloads,
    validate_single_source_pair,
    write_combined_with_sidecar,
    write_payload,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pair_environment(workdir: Path, pairdir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(workdir),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0",
            "AUTONOMOUS_ALLHAD_XRD_CACHE": str(workdir / "runtime_xrd"),
            "AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE": "1",
            "TMPDIR": str(pairdir / "tmp"),
            "TMP": str(pairdir / "tmp"),
            "TEMP": str(pairdir / "tmp"),
            "XDG_CACHE_HOME": str(pairdir / "cache"),
            "MPLCONFIGDIR": str(pairdir / "mpl"),
            "NUMBA_CACHE_DIR": str(pairdir / "cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(pairdir / "cache" / "pycache"),
        }
    )
    return environment


def run_pair(
    nuisance: str,
    shard: Path,
    record_digest: str,
    workdir: Path,
    chunk_size: int,
) -> dict[str, Any]:
    pairdir = workdir / "pair_outputs" / nuisance
    for name in ("tmp", "cache", "mpl"):
        (pairdir / name).mkdir(parents=True, exist_ok=True)
    histogram = pairdir / "out.json.gz"
    metadata = pairdir / "out.meta.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        "autonomous_allhad.shape_histogram_2024_worker",
        "--shard",
        str(shard),
        "--output",
        str(histogram),
        "--metadata-output",
        str(metadata),
        "--variation-group",
        nuisance,
        "--chunk-size",
        str(chunk_size),
        "--record-workers",
        "1",
    ]
    started = time.time()
    process = subprocess.run(
        command,
        cwd=workdir,
        env=pair_environment(workdir, pairdir),
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        (pairdir / "worker.stdout.txt").write_text(process.stdout[-20000:])
        (pairdir / "worker.stderr.txt").write_text(process.stderr[-20000:])
        raise RuntimeError(
            f"{nuisance} pair worker failed ({process.returncode}): "
            f"{process.stderr[-1200:]}"
        )
    validate_single_source_pair(
        histogram,
        metadata,
        nuisance,
        record_digest,
    )
    return {
        "nuisance": nuisance,
        "histogram": str(histogram),
        "metadata": str(metadata),
        "wall_time_s": round(time.time() - started, 3),
        "runtime_warning_count": process.stderr.count("RuntimeWarning"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache one 2024 background NanoAOD source once, evaluate 20 "
            "independent Up/Down nuisance pairs, and combine them."
        )
    )
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.workers > len(FINAL_SHAPE_NUISANCES):
        raise ValueError("--workers must be between 1 and 20")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    shard = Path(args.shard).resolve()
    workdir = Path.cwd().resolve()
    report_path = Path(args.report_output)
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": "shape_histogram_2024_pair_source_report_v1",
        "status": "running",
        "shard": str(shard),
        "workers": args.workers,
        "nuisance_pairs": list(FINAL_SHAPE_NUISANCES),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        shard_payload = read_json(shard)
        records = list(shard_payload.get("records") or [])
        if len(records) != 1:
            raise RuntimeError("pairwise source worker requires exactly one record")
        record_digest = str(shard_payload.get("record_digest") or "")
        input_file = str(records[0].get("file_path") or "")
        if not record_digest or not input_file:
            raise RuntimeError("source shard is missing record digest or file path")

        xrd_cache = workdir / "runtime_xrd"
        xrd_cache.mkdir(parents=True, exist_ok=True)
        os.environ["AUTONOMOUS_ALLHAD_XRD_CACHE"] = str(xrd_cache)
        os.environ["AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE"] = "1"
        source, access = baseline.open_root_with_xrd_fallback(
            input_file,
            timeout=60,
        )
        source.close()
        cache_path = Path(str(access.get("cache_path") or ""))
        if not cache_path.is_file() or cache_path.stat().st_size == 0:
            raise RuntimeError(f"prefetched source cache is invalid: {cache_path}")

        results: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    run_pair,
                    nuisance,
                    shard,
                    record_digest,
                    workdir,
                    args.chunk_size,
                ): nuisance
                for nuisance in FINAL_SHAPE_NUISANCES
            }
            for future in concurrent.futures.as_completed(futures):
                nuisance = futures[future]
                results[nuisance] = future.result()
        if set(results) != set(FINAL_SHAPE_NUISANCES):
            raise RuntimeError("not all 20 nuisance pairs completed")

        pair_paths = [
            Path(results[nuisance]["histogram"])
            for nuisance in FINAL_SHAPE_NUISANCES
        ]
        combined = combine_pair_payloads(pair_paths)
        combined["summary"]["pair_source_execution"] = {
            "workers": args.workers,
            "network_source_reads": 1,
            "cache_path": str(cache_path),
            "cache_size": cache_path.stat().st_size,
            "pair_wall_times_s": {
                nuisance: results[nuisance]["wall_time_s"]
                for nuisance in FINAL_SHAPE_NUISANCES
            },
            "runtime_warning_counts": {
                nuisance: results[nuisance]["runtime_warning_count"]
                for nuisance in FINAL_SHAPE_NUISANCES
            },
        }
        write_combined_with_sidecar(
            Path(args.output),
            Path(args.metadata_output),
            combined,
        )
        report.update(
            {
                "status": "complete",
                "source_record_digest": record_digest,
                "input_file": input_file,
                "access": access,
                "variation_count": 40,
                "pair_results": results,
                "output": args.output,
                "metadata_output": args.metadata_output,
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "error": str(exc)[:4000],
            }
        )
    report["completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    report["wall_time_s"] = round(time.time() - started, 3)
    write_payload(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
