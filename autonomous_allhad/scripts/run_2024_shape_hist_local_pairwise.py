#!/usr/bin/env python3
"""Run 2024 object-shape nuisances as 20 independently checkpointed Up/Down pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from autonomous_allhad.shape_histogram_2024_worker import FINAL_SHAPE_NUISANCES
from autonomous_allhad.shape_histogram_pair_2024 import (
    finalize_pair_accumulator,
    merge_single_source_pair,
    read_payload,
    sha256,
    validate_single_source_pair,
    write_pair_with_sidecar,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_PYTHON = Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python")
DEFAULT_SOURCE_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v5_local1_20260725"
)
DEFAULT_OUTPUT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v6_localpairs_20260725"
)


def load_parallel_helpers() -> Any:
    path = Path(__file__).with_name("run_2024_shape_hist_local_parallel.py")
    spec = importlib.util.spec_from_file_location("_shape_local_parallel_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local-runner helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_pair_runtime(
    helpers: Any,
    source_manifest: dict[str, Any],
    repo: Path,
    campaign: Path,
) -> Path:
    runtime = campaign / "local_runtime/runtime"
    package_source = repo / "autonomous_allhad/autonomous_allhad"
    builder_source = (
        repo
        / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    )
    payload_bundle = Path(source_manifest["payload_bundle"]["path"])
    code_sources = sorted(package_source.glob("*.py")) + [builder_source]
    code_hashes = {
        str(path.relative_to(repo)): helpers.sha256(path) for path in code_sources
    }
    expected = {
        "payload_bundle_sha256": source_manifest["payload_bundle"]["sha256"],
        "source_shard_bundle_sha256": source_manifest["shard_bundle"]["sha256"],
        "code_sha256": code_hashes,
    }
    marker = runtime / "runtime_manifest.json"
    if marker.is_file():
        observed = json.loads(marker.read_text())
        if all(observed.get(key) == value for key, value in expected.items()):
            return runtime
        raise RuntimeError("existing pair runtime does not match code/payload hashes")
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"refusing non-empty unvalidated pair runtime: {runtime}")
    if (
        not payload_bundle.is_file()
        or helpers.sha256(payload_bundle)
        != source_manifest["payload_bundle"]["sha256"]
    ):
        raise RuntimeError("source payload bundle is missing or has a bad checksum")
    helpers.safe_extract(payload_bundle, runtime)
    package_target = runtime / "autonomous_allhad"
    package_target.mkdir(parents=True, exist_ok=True)
    for source in sorted(package_source.glob("*.py")):
        shutil.copy2(source, package_target / source.name)
    builder_target = runtime / "workflow/build_flat_boosted_recoil_hists.py"
    builder_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(builder_source, builder_target)
    write_json(
        marker,
        {
            **expected,
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return runtime


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


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def base_environment(
    runtime: Path,
    proxy: Path,
    source_task: Path,
) -> dict[str, str]:
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
            "AUTONOMOUS_ALLHAD_XRD_CACHE": str(source_task / "xrd"),
            "AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE": "1",
        }
    )
    return environment


def prefetch_source(
    python: Path,
    runtime: Path,
    proxy: Path,
    source_task: Path,
    input_file: str,
) -> dict[str, Any]:
    source_task.mkdir(parents=True, exist_ok=True)
    for name in ("xrd", "prefetch_tmp", "prefetch_cache"):
        (source_task / name).mkdir(parents=True, exist_ok=True)
    environment = base_environment(runtime, proxy, source_task)
    environment.update(
        {
            "TMPDIR": str(source_task / "prefetch_tmp"),
            "TMP": str(source_task / "prefetch_tmp"),
            "TEMP": str(source_task / "prefetch_tmp"),
            "XDG_CACHE_HOME": str(source_task / "prefetch_cache"),
            "NUMBA_CACHE_DIR": str(source_task / "prefetch_cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(
                source_task / "prefetch_cache" / "pycache"
            ),
        }
    )
    code = (
        "import json,sys;"
        "from autonomous_allhad.real_subset_worker import "
        "open_root_with_xrd_fallback;"
        "root,info=open_root_with_xrd_fallback(sys.argv[1],timeout=60);"
        "root.close();"
        "print(json.dumps(info,sort_keys=True))"
    )
    process = subprocess.run(
        [str(python), "-c", code, input_file],
        cwd=runtime,
        env=environment,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"source prefetch failed ({process.returncode}): "
            f"{process.stderr[-1000:]}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("source prefetch produced no access record")
    info = json.loads(lines[-1])
    cache = Path(str(info.get("cache_path") or ""))
    if not cache.is_file() or cache.stat().st_size == 0:
        raise RuntimeError(f"source cache is missing after prefetch: {cache}")
    return info


def run_pair(
    nuisance: str,
    job: dict[str, Any],
    python: Path,
    runtime: Path,
    proxy: Path,
    source_task: Path,
    chunk_size: int,
) -> dict[str, Any]:
    pair_task = source_task / nuisance
    pair_task.mkdir(parents=True, exist_ok=True)
    shard_path = pair_task / job["shard_name"]
    write_json(shard_path, job["shard_payload"])
    histogram = pair_task / "out.json.gz"
    metadata = pair_task / "out.meta.json"
    for name in ("tmp", "cache", "mpl"):
        (pair_task / name).mkdir(parents=True, exist_ok=True)
    environment = base_environment(runtime, proxy, source_task)
    environment.update(
        {
            "TMPDIR": str(pair_task / "tmp"),
            "TMP": str(pair_task / "tmp"),
            "TEMP": str(pair_task / "tmp"),
            "XDG_CACHE_HOME": str(pair_task / "cache"),
            "MPLCONFIGDIR": str(pair_task / "mpl"),
            "NUMBA_CACHE_DIR": str(pair_task / "cache" / "numba"),
            "PYTHONPYCACHEPREFIX": str(pair_task / "cache" / "pycache"),
        }
    )
    command = [
        str(python),
        "-u",
        "-m",
        "autonomous_allhad.shape_histogram_2024_worker",
        "--shard",
        str(shard_path),
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
        cwd=runtime,
        env=environment,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{nuisance} worker failed ({process.returncode}): "
            f"{process.stderr[-1200:]}"
        )
    payload, sidecar = validate_single_source_pair(
        histogram,
        metadata,
        nuisance,
        str(job["record_digest"]),
    )
    return {
        "nuisance": nuisance,
        "payload": payload,
        "metadata": sidecar,
        "wall_time_s": round(time.time() - started, 3),
        "warning_count": process.stderr.count("RuntimeWarning"),
    }


def checkpoint(
    campaign: Path,
    accumulators: dict[str, dict[str, Any]],
    expected_sources: int,
    completed_sources: int,
    previous: str | None,
) -> str:
    generation = f"generation_{completed_sources:08d}"
    target = campaign / "checkpoints" / generation
    temporary = target.with_name(f"{target.name}.partial.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for nuisance in FINAL_SHAPE_NUISANCES:
        payload = finalize_pair_accumulator(
            accumulators[nuisance],
            expected_sources,
        )
        write_pair_with_sidecar(
            temporary / f"{nuisance}.json.gz",
            temporary / f"{nuisance}.meta.json",
            payload,
        )
    os.replace(temporary, target)
    write_json(
        campaign / "checkpoint.json",
        {
            "schema_version": "shape_histogram_2024_pair_checkpoint_v1",
            "generation": generation,
            "completed_sources": completed_sources,
            "expected_sources": expected_sources,
            "nuisances": list(FINAL_SHAPE_NUISANCES),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    if previous and previous != generation:
        old = campaign / "checkpoints" / previous
        if old.is_dir():
            shutil.rmtree(old)
    return generation


def load_checkpoint(
    campaign: Path,
) -> tuple[dict[str, dict[str, Any]], int, str | None]:
    pointer = campaign / "checkpoint.json"
    if not pointer.is_file():
        return {}, 0, None
    record = json.loads(pointer.read_text())
    generation = str(record["generation"])
    directory = campaign / "checkpoints" / generation
    accumulators: dict[str, dict[str, Any]] = {}
    digest_sets: list[set[str]] = []
    for nuisance in FINAL_SHAPE_NUISANCES:
        histogram = directory / f"{nuisance}.json.gz"
        metadata = directory / f"{nuisance}.meta.json"
        payload = read_payload(histogram)
        sidecar = read_payload(metadata)
        if sha256(histogram) != sidecar.get("histogram_sha256"):
            raise RuntimeError(f"checkpoint checksum mismatch: {histogram}")
        if payload.get("nuisance") != nuisance:
            raise RuntimeError(f"checkpoint nuisance mismatch: {histogram}")
        accumulators[nuisance] = payload
        digest_sets.append(
            set((payload.get("summary") or {}).get("source_record_digests") or [])
        )
    if any(items != digest_sets[0] for items in digest_sets[1:]):
        raise RuntimeError("pair checkpoint source coverage is inconsistent")
    completed = len(digest_sets[0])
    if completed != int(record.get("completed_sources") or 0):
        raise RuntimeError("pair checkpoint source count is inconsistent")
    return accumulators, completed, generation


def publish_final_outputs(
    campaign: Path,
    accumulators: dict[str, dict[str, Any]],
    expected_sources: int,
) -> None:
    outputs = campaign / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    for nuisance in FINAL_SHAPE_NUISANCES:
        payload = finalize_pair_accumulator(
            accumulators[nuisance],
            expected_sources,
        )
        if payload.get("status") != "complete":
            raise RuntimeError(f"cannot publish incomplete nuisance pair: {nuisance}")
        write_pair_with_sidecar(
            outputs / f"{nuisance}.json.gz",
            outputs / f"{nuisance}.meta.json",
            payload,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-campaign", type=Path, default=DEFAULT_SOURCE_CAMPAIGN)
    parser.add_argument("--output-campaign", type=Path, default=DEFAULT_OUTPUT_CAMPAIGN)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or args.workers > 20:
        raise ValueError("--workers must be between 1 and 20")
    if args.chunk_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("chunk-size and checkpoint-every must be positive")
    for path in (args.source_campaign, args.output_campaign):
        if not path.is_absolute() or not str(path).startswith("/eos/user/"):
            raise ValueError(f"campaign path must be under /eos/user: {path}")
    if args.source_campaign == args.output_campaign:
        raise ValueError("source and output campaigns must differ")

    helpers = load_parallel_helpers()
    source_manifest = helpers.read_json(args.source_campaign / "manifest.json")
    if int(source_manifest.get("variation_count") or 0) != 40:
        raise RuntimeError("source campaign is not the adopted all-40 definition")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    proxy = args.repo / f"analysis/proxy/x509up_u{os.getuid()}"
    if not proxy.is_file():
        raise FileNotFoundError(proxy)

    args.output_campaign.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_campaign / "local_runner.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another pairwise local runner owns the campaign") from exc
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()

    runtime = prepare_pair_runtime(
        helpers,
        source_manifest,
        args.repo,
        args.output_campaign,
    )
    jobs = helpers.read_jobs(args.source_campaign, source_manifest, runtime)
    if args.max_sources is not None:
        jobs = jobs[: max(0, args.max_sources)]
    expected_sources = len(jobs)
    accumulators, completed_sources, generation = load_checkpoint(
        args.output_campaign
    )
    if completed_sources > expected_sources:
        raise RuntimeError("checkpoint has more sources than selected campaign")
    processed_digests = (
        set(
            (
                next(iter(accumulators.values())).get("summary") or {}
            ).get("source_record_digests")
            or []
        )
        if accumulators
        else set()
    )
    pending = [
        job for job in jobs if str(job["record_digest"]) not in processed_digests
    ]

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "schema_version": "shape_histogram_2024_pair_campaign_v1",
        "status": "running",
        "year": 2024,
        "source_campaign": str(args.source_campaign),
        "source_manifest_sha256": helpers.sha256(
            args.source_campaign / "manifest.json"
        ),
        "execution": {
            "mode": "lxplus_local_source_cached_nuisance_pairs",
            "controller_pid": os.getpid(),
            "host": os.uname().nodename,
            "workers": args.workers,
            "source_files_per_stage": 1,
            "nuisance_pairs_per_source": 20,
            "network_reads_per_source": 1,
            "started_at": started_at,
        },
        "nuisances": list(FINAL_SHAPE_NUISANCES),
        "directional_variations": 40,
        "expected_source_records": expected_sources,
        "output_policy": {
            "histogram_payloads": 20,
            "metadata_sidecars": 20,
            "one_payload_per_nuisance": True,
            "checkpointed": True,
            "merge_after_completion": True,
        },
    }
    write_json(args.output_campaign / "manifest.json", manifest)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status_path = args.output_campaign / "local_status.json"
    history_path = args.output_campaign / "history.jsonl"

    def snapshot(status: str, active: str | None = None, failure: Any = None) -> None:
        write_json(
            status_path,
            {
                "schema_version": "shape_histogram_2024_pair_status_v1",
                "status": status,
                "controller_pid": os.getpid(),
                "host": os.uname().nodename,
                "workers": args.workers,
                "expected_sources": expected_sources,
                "completed_sources": completed_sources,
                "remaining_sources": expected_sources - completed_sources,
                "completed_pair_evaluations": completed_sources * 20,
                "expected_pair_evaluations": expected_sources * 20,
                "active_source": active,
                "checkpoint_generation": generation,
                "failure": failure,
                "started_at": started_at,
                "updated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
        )

    snapshot("running")
    failure: dict[str, Any] | None = None
    for job in pending:
        if stop.is_set():
            break
        source_task = (
            args.output_campaign / "local_runtime/tasks" / str(job["name"])
        )
        if source_task.exists():
            shutil.rmtree(source_task)
        snapshot("running", str(job["name"]))
        source_started = time.time()
        try:
            access = prefetch_source(
                args.python,
                runtime,
                proxy,
                source_task,
                str(job["input_file"]),
            )
            results: dict[str, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures = {
                    executor.submit(
                        run_pair,
                        nuisance,
                        job,
                        args.python,
                        runtime,
                        proxy,
                        source_task,
                        args.chunk_size,
                    ): nuisance
                    for nuisance in FINAL_SHAPE_NUISANCES
                }
                for future in concurrent.futures.as_completed(futures):
                    nuisance = futures[future]
                    results[nuisance] = future.result()
            if set(results) != set(FINAL_SHAPE_NUISANCES):
                raise RuntimeError("not all 20 nuisance pairs returned")
            for nuisance in FINAL_SHAPE_NUISANCES:
                accumulators[nuisance] = merge_single_source_pair(
                    accumulators.get(nuisance),
                    results[nuisance]["payload"],
                    nuisance,
                )
            completed_sources += 1
            append_jsonl(
                history_path,
                {
                    "status": "complete",
                    "source_name": job["name"],
                    "source_record_digest": job["record_digest"],
                    "dataset": job["dataset"],
                    "input_file": job["input_file"],
                    "access_method": access.get("access_method"),
                    "cache_reused": access.get("cache_reused"),
                    "pair_wall_times_s": {
                        nuisance: results[nuisance]["wall_time_s"]
                        for nuisance in FINAL_SHAPE_NUISANCES
                    },
                    "runtime_warning_counts": {
                        nuisance: results[nuisance]["warning_count"]
                        for nuisance in FINAL_SHAPE_NUISANCES
                    },
                    "wall_time_s": round(time.time() - source_started, 3),
                    "completed_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
            )
            if (
                completed_sources % args.checkpoint_every == 0
                or completed_sources == expected_sources
            ):
                generation = checkpoint(
                    args.output_campaign,
                    accumulators,
                    expected_sources,
                    completed_sources,
                    generation,
                )
        except Exception as exc:
            failure = {
                "source_name": job.get("name"),
                "source_record_digest": job.get("record_digest"),
                "dataset": job.get("dataset"),
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            append_jsonl(
                history_path,
                {
                    **failure,
                    "status": "failed",
                    "failed_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
            )
            break
        finally:
            if source_task.exists():
                shutil.rmtree(source_task)

    if accumulators and (
        generation is None
        or int(
            json.loads((args.output_campaign / "checkpoint.json").read_text()).get(
                "completed_sources", -1
            )
        )
        != completed_sources
    ):
        generation = checkpoint(
            args.output_campaign,
            accumulators,
            expected_sources,
            completed_sources,
            generation,
        )

    if failure:
        final_status = "failed"
    elif stop.is_set():
        final_status = "stopped"
    elif completed_sources == expected_sources:
        publish_final_outputs(
            args.output_campaign,
            accumulators,
            expected_sources,
        )
        final_status = "complete"
    else:
        final_status = "incomplete"
    snapshot(final_status, failure=failure)
    manifest = json.loads((args.output_campaign / "manifest.json").read_text())
    manifest["status"] = final_status
    manifest["execution"]["completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    manifest["execution"]["completed_sources"] = completed_sources
    manifest["execution"]["failure"] = failure
    write_json(args.output_campaign / "manifest.json", manifest)
    return 0 if final_status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
