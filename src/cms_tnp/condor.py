"""Self-contained HTCondor campaigns for histogram counting."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config

STATUS_NAMES = {
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring_output",
    7: "suspended",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _reject_afs(paths: Sequence[Path]) -> None:
    rejected = [str(path) for path in paths if str(path.resolve()).startswith("/afs/")]
    if rejected:
        raise ValueError(f"AFS paths are not supported: {rejected}")


def _safe_jdl_path(path: Path) -> str:
    value = str(path.resolve())
    if any(character in value for character in ('"', "\n", "\r", ",")):
        raise ValueError(f"path cannot be represented safely in HTCondor JDL: {value}")
    return f'"{value}"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_payload(
    value: str, config_dir: Path, staging: Path, name: str
) -> str:
    source = Path(value)
    if not source.is_absolute():
        source = config_dir / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    _reject_afs([source])
    target = staging / "payloads" / f"{name}_{source.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target.relative_to(staging))


def _bundle_inputs(
    config_path: Path,
    data_shards: Path,
    mc_shards: Path,
    archive: Path,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    jobs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cms_tnp_condor_") as temporary:
        staging = Path(temporary)
        lumimask = config.get("lumimask")
        if not lumimask:
            raise ValueError("Condor production requires a data luminosity mask")
        config["lumimask"] = _copy_payload(
            str(lumimask), config_path.parent, staging, "lumimask"
        )
        for index, correction in enumerate(
            config.get("weights", {}).get("corrections", [])
        ):
            correction["file"] = _copy_payload(
                str(correction["file"]),
                config_path.parent,
                staging,
                f"correction_{index:03d}",
            )
        _write(staging / "config.json", config)
        for sample, directory in (("data", data_shards), ("mc", mc_shards)):
            sources = sorted(directory.glob("shard_*.json"))
            if not sources:
                raise ValueError(f"no shard JSON files found in {directory}")
            for source in sources:
                shard = _read(source)
                if shard.get("sample") != sample:
                    raise ValueError(f"{source} is not a {sample} shard")
                if shard.get("measurement") != config["measurement"]:
                    raise ValueError(f"{source} belongs to another measurement")
                relative = Path("shards") / f"{sample}_{source.name}"
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                jobs.append(
                    {
                        "job_id": len(jobs),
                        "sample": sample,
                        "shard": str(relative),
                        "result": f"{sample}_{source.name}",
                        "files_expected": len(shard.get("records", [])),
                    }
                )
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as target:
            for source in sorted(staging.rglob("*")):
                target.add(
                    source,
                    arcname=str(source.relative_to(staging)),
                    recursive=False,
                )
    return jobs


RUNNER = """#!/usr/bin/env bash
set -euo pipefail

sample="$1"
shard="$2"
result="$3"

mkdir runtime job_inputs
tar -xzf cms_tnp_environment.tar.gz -C runtime
tar -xzf cms_tnp_inputs.tar.gz -C job_inputs

if [[ -x runtime/bin/conda-unpack ]]; then
  runtime/bin/conda-unpack
fi
if [[ ! -x runtime/bin/python ]]; then
  echo "worker archive does not contain runtime/bin/python" >&2
  exit 64
fi
if [[ -f x509up ]]; then
  export X509_USER_PROXY="$PWD/x509up"
fi
export MPLBACKEND=Agg

runtime/bin/python -m cms_tnp count \
  --config job_inputs/config.json \
  --sample "$sample" \
  --shard "job_inputs/$shard" \
  --output "$result"

runtime/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); q=p["processing"]; ok=p.get("status")=="complete" and q["files_expected"]==q["files_processed"] and not q["files_failed"]; raise SystemExit(0 if ok else 2)' "$result"
"""


def _submit_file(
    campaign_dir: Path,
    environment: Path,
    bundle: Path,
    proxy: Path | None,
    request_cpus: int,
    request_memory_mb: int,
    request_disk_mb: int,
    job_flavour: str | None,
) -> str:
    inputs = [environment, bundle]
    if proxy:
        inputs.append(proxy)
    lines = [
        "universe = vanilla",
        "executable = run_job.sh",
        "arguments = $(sample) $(shard) $(result)",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "transfer_input_files = " + ",".join(_safe_jdl_path(path) for path in inputs),
        "transfer_output_files = $(result)",
        (
            'transfer_output_remaps = "$(result)='
            + str((campaign_dir / "outputs").resolve())
            + '/$(result)"'
        ),
        f"request_cpus = {request_cpus}",
        f"request_memory = {request_memory_mb}MB",
        f"request_disk = {request_disk_mb}MB",
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "output = logs/$(job_id).out",
        "error = logs/$(job_id).err",
        "log = logs/cluster.log",
    ]
    if job_flavour:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_flavour):
            raise ValueError(f"invalid job flavour: {job_flavour!r}")
        lines.append(f'+JobFlavour = "{job_flavour}"')
    lines.append("queue job_id,sample,shard,result from jobs.tsv")
    return "\n".join(lines) + "\n"


def prepare_campaign(
    *,
    config_path: Path,
    data_shards: Path,
    mc_shards: Path,
    environment_path: Path,
    campaign_dir: Path,
    proxy_path: Path | None = None,
    request_cpus: int = 1,
    request_memory_mb: int = 4000,
    request_disk_mb: int = 4000,
    job_flavour: str | None = None,
) -> dict[str, Any]:
    sources = [config_path, data_shards, mc_shards, environment_path, campaign_dir]
    if proxy_path:
        sources.append(proxy_path)
    _reject_afs(sources)
    for path in (config_path, environment_path):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if proxy_path and not proxy_path.resolve().is_file():
        raise FileNotFoundError(proxy_path)
    if min(request_cpus, request_memory_mb, request_disk_mb) <= 0:
        raise ValueError("requested Condor resources must be positive")

    campaign_dir = campaign_dir.resolve()
    input_dir = campaign_dir / "input"
    for directory in (input_dir, campaign_dir / "logs", campaign_dir / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)
    environment = input_dir / "cms_tnp_environment.tar.gz"
    bundle = input_dir / "cms_tnp_inputs.tar.gz"
    if environment_path.resolve() != environment:
        shutil.copy2(environment_path, environment)
    jobs = _bundle_inputs(
        config_path.resolve(), data_shards.resolve(), mc_shards.resolve(), bundle
    )
    proxy = None
    if proxy_path:
        proxy = input_dir / "x509up"
        if proxy_path.resolve() != proxy:
            shutil.copy2(proxy_path, proxy)
        proxy.chmod(stat.S_IRUSR | stat.S_IWUSR)

    runner = campaign_dir / "run_job.sh"
    runner.write_text(RUNNER)
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    (campaign_dir / "jobs.tsv").write_text(
        "".join(
            f"{job['job_id']}\t{job['sample']}\t{job['shard']}\t{job['result']}\n"
            for job in jobs
        )
    )
    (campaign_dir / "submit.sub").write_text(
        _submit_file(
            campaign_dir,
            environment,
            bundle,
            proxy,
            request_cpus,
            request_memory_mb,
            request_disk_mb,
            job_flavour,
        )
    )
    manifest = {
        "schema_version": 1,
        "campaign_dir": str(campaign_dir),
        "config_sha256": _sha256(config_path.resolve()),
        "environment_sha256": _sha256(environment),
        "input_bundle_sha256": _sha256(bundle),
        "proxy_included": proxy is not None,
        "jobs": jobs,
    }
    _write(campaign_dir / "campaign.json", manifest)
    return {
        "campaign_dir": str(campaign_dir),
        "jobs": len(jobs),
        "data_jobs": sum(job["sample"] == "data" for job in jobs),
        "mc_jobs": sum(job["sample"] == "mc" for job in jobs),
    }


def submit_campaign(
    campaign_dir: Path,
    *,
    submit_command: str = "condor_submit",
    resubmit: bool = False,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    _reject_afs([campaign_dir])
    submission_path = campaign_dir / "submission.json"
    if submission_path.exists() and not resubmit:
        raise FileExistsError(
            f"{submission_path} already exists; use --resubmit deliberately"
        )
    result = subprocess.run(
        [submit_command, "submit.sub"],
        cwd=campaign_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"cluster\s+(\d+)", result.stdout, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"cannot parse cluster id from {result.stdout!r}")
    submission = {
        "cluster_id": int(match.group(1)),
        "command": submit_command,
        "stdout": result.stdout,
    }
    _write(submission_path, submission)
    return submission


def _validate_result(path: Path, job: Mapping[str, Any]) -> str | None:
    try:
        result = _read(path)
        processing = result["processing"]
        expected = int(job["files_expected"])
        if result.get("sample") != job["sample"]:
            return "sample mismatch"
        if int(processing["files_expected"]) != expected:
            return "file-count mismatch"
        if int(processing["files_processed"]) != expected:
            return "incomplete ROOT coverage"
        if processing["files_failed"]:
            return "ROOT failures are present"
        if result.get("status") != "complete":
            return "result is not complete"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def _queue_status(campaign_dir: Path, query_command: str) -> dict[str, Any]:
    submission_path = campaign_dir / "submission.json"
    if not submission_path.exists():
        return {"submitted": False, "jobs": {}, "held_reasons": {}}
    cluster_id = int(_read(submission_path)["cluster_id"])
    try:
        result = subprocess.run(
            [query_command, str(cluster_id), "-json"],
            check=False,
            capture_output=True,
            text=True,
        )
        records = json.loads(result.stdout or "[]") if result.returncode == 0 else []
        counts = Counter(
            STATUS_NAMES.get(int(record.get("JobStatus", 0)), "unknown")
            for record in records
        )
        held_reasons = Counter(
            str(record.get("HoldReason", "unknown"))
            for record in records
            if int(record.get("JobStatus", 0)) == 5
        )
        return {
            "submitted": True,
            "cluster_id": cluster_id,
            "jobs": dict(sorted(counts.items())),
            "held_reasons": dict(sorted(held_reasons.items())),
            "query_error": None if result.returncode == 0 else result.stderr.strip(),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {
            "submitted": True,
            "cluster_id": cluster_id,
            "jobs": {},
            "held_reasons": {},
            "query_error": f"{type(error).__name__}: {error}",
        }


def campaign_status(
    campaign_dir: Path, *, query_command: str = "condor_q"
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    _reject_afs([campaign_dir])
    manifest = _read(campaign_dir / "campaign.json")
    missing: list[str] = []
    invalid: dict[str, str] = {}
    valid: list[str] = []
    for job in manifest["jobs"]:
        path = campaign_dir / "outputs" / str(job["result"])
        if not path.exists():
            missing.append(str(job["result"]))
            continue
        error = _validate_result(path, job)
        if error:
            invalid[str(job["result"])] = error
        else:
            valid.append(str(job["result"]))
    return {
        "status": "complete" if len(valid) == len(manifest["jobs"]) else "incomplete",
        "jobs_expected": len(manifest["jobs"]),
        "outputs_valid": len(valid),
        "outputs_missing": missing,
        "outputs_invalid": invalid,
        "queue": _queue_status(campaign_dir, query_command),
    }


def finalize_campaign(campaign_dir: Path, output_dir: Path) -> dict[str, Any]:
    from .fit import fit_payload
    from .payload import build_payload, write_payload
    from .plot import plot_result
    from .reduce import merge

    campaign_dir = campaign_dir.resolve()
    output_dir = output_dir.resolve()
    _reject_afs([campaign_dir, output_dir])
    status = campaign_status(campaign_dir)
    if status["status"] != "complete":
        raise RuntimeError(
            f"campaign has {status['outputs_valid']}/{status['jobs_expected']} valid outputs"
        )
    manifest = _read(campaign_dir / "campaign.json")
    shards = [
        _read(campaign_dir / "outputs" / str(job["result"]))
        for job in manifest["jobs"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    histograms = merge(shards)
    _write(output_dir / "histograms.json", histograms)
    result = fit_payload(histograms)
    _write(output_dir / "fit_result.json", result)
    if not all(item.get("valid") for item in result["bins"]):
        raise RuntimeError("one or more fitted bins are invalid")
    digest = write_payload(
        output_dir / "scale_factors.json.gz", build_payload(result)
    )
    plots = plot_result(result, output_dir / "plots")
    summary = {
        "status": "complete",
        "jobs": len(manifest["jobs"]),
        "correction": str(output_dir / "scale_factors.json.gz"),
        "correction_sha256": digest,
        "plots": plots,
    }
    _write(output_dir / "summary.json", summary)
    return summary
