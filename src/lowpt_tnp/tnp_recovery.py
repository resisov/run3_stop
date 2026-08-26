#!/usr/bin/env python3
"""Build residual manifests and recover low-pT TnP ROOT files on lxplus scratch."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .reference_trigger_counts import json_safe
from .tnp_measurement_shard import count_tnp_shard


def _xrootd_candidates(file_path: str) -> list[str]:
    if "/store/" not in file_path:
        return [file_path]
    lfn = "/store/" + file_path.split("/store/", 1)[1]
    return list(dict.fromkeys([
        f"root://eoscms.cern.ch//eos/cms{lfn}",
        file_path,
        f"root://xrootd-cms.infn.it/{lfn}",
        f"root://cmsxrootd.fnal.gov/{lfn}",
    ]))


def _successful_paths(payload: dict[str, Any]) -> set[str]:
    return {
        str(path)
        for stats in payload.get("processing", {}).values()
        for path in (stats.get("files_successful") or [])
    }


def build_residual_manifest(
    records_path: Path,
    workdir: Path,
    recovery_dirs: list[Path],
) -> dict[str, Any]:
    campaign = json.loads(records_path.read_text())
    records = list(campaign.get("records") or [])
    by_path = {str(record["file_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("duplicate logical ROOT file in TnP campaign records")

    primary_paths = sorted((workdir / "shard_outputs").glob("shard_*.json"))
    recovery_paths = sorted(
        path for directory in recovery_dirs for path in directory.glob("shard_recovery_*.json")
    )
    successful: set[str] = set()
    failure_by_path: dict[str, dict[str, Any]] = {}
    invalid_outputs: list[dict[str, str]] = []
    for path in [*primary_paths, *recovery_paths]:
        try:
            payload = json.loads(path.read_text())
            if payload.get("measurement") != campaign["measurement"]:
                raise ValueError("measurement mismatch")
            successful.update(_successful_paths(payload))
            for stats in payload.get("processing", {}).values():
                for failure in stats.get("files_failed") or []:
                    logical = str(failure.get("path") or "")
                    if logical:
                        failure_by_path[logical] = failure
        except Exception as exc:
            invalid_outputs.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    unexpected_successes = sorted(successful - set(by_path))
    if unexpected_successes:
        raise RuntimeError(f"successful recovery paths are absent from campaign records: {unexpected_successes[:5]}")
    unresolved = [by_path[path] for path in sorted(set(by_path) - successful)]
    expected_shards = len(list((workdir / "manifests").glob("tnp_*.json")))
    return {
        "schema_version": 1,
        "measurement": campaign["measurement"],
        "status": "complete" if not unresolved and not invalid_outputs else "recovery_required",
        "campaign_records": str(records_path),
        "primary_workdir": str(workdir),
        "primary_shards_expected": expected_shards,
        "primary_outputs_present": len(primary_paths),
        "recovery_outputs_present": len(recovery_paths),
        "files_expected": len(records),
        "files_successful": len(successful),
        "files_unresolved": len(unresolved),
        "invalid_outputs": invalid_outputs,
        "failure_diagnostics": {
            path: failure_by_path[path] for path in sorted(failure_by_path) if path not in successful
        },
        "records": unresolved,
        "created_unix": time.time(),
    }


def _normalise_logical_path(payload: dict[str, Any], local_path: str, record: dict[str, Any]) -> None:
    logical = str(record["file_path"])
    payload["input_records"] = [{
        "file_path": logical,
        "dataset": record.get("dataset"),
        "sample": record.get("sample"),
    }]
    for stats in payload.get("processing", {}).values():
        stats["files_successful"] = [
            logical if str(path) == local_path else str(path)
            for path in stats.get("files_successful") or []
        ]
        for failure in stats.get("files_failed") or []:
            if str(failure.get("path")) == local_path:
                failure["path"] = logical
        for access in stats.get("file_access") or []:
            if str(access.get("path")) == local_path:
                access["path"] = logical


def recover(
    *,
    kind: str,
    manifest_path: Path,
    config_path: Path,
    repo: Path,
    output_dir: Path,
    scratch_dir: Path,
    step_size: int,
    worker_index: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1 or worker_index < 0 or worker_index >= workers:
        raise ValueError(
            f"invalid worker partition: worker_index={worker_index}, workers={workers}"
        )
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    campaign_records = list(manifest.get("records") or [])
    indexed_records = list(enumerate(campaign_records))[worker_index::workers]
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    successful = 0
    failed = []
    xrdcp_environment = dict(os.environ)
    # The packed Python runtime supplies its own C++ libraries through
    # LD_LIBRARY_PATH.  Let the host xrdcp resolve against the matching host
    # XRootD libraries while retaining the proxy and all other credentials.
    xrdcp_environment.pop("LD_LIBRARY_PATH", None)
    for index, record in indexed_records:
        logical = str(record["file_path"])
        digest = hashlib.sha256(logical.encode()).hexdigest()[:16]
        local_path = scratch_dir / f"tnp_{kind}_{index:05d}_{digest}.root"
        attempts = []
        copied_from = None
        try:
            for candidate in _xrootd_candidates(logical):
                try:
                    completed = subprocess.run(
                        ["xrdcp", "-f", "--nopbar", candidate, str(local_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=xrdcp_environment,
                        # A dead XRootD endpoint often accepts the connection but
                        # transfers zero bytes.  Move on to the next replica
                        # promptly; healthy CERN/AAA copies finish well inside
                        # this per-endpoint ceiling for the recovery files.
                        timeout=180,
                    )
                except subprocess.TimeoutExpired as exc:
                    attempts.append({
                        "source": candidate,
                        "exit_code": None,
                        "error": f"xrdcp timed out after {exc.timeout} seconds",
                    })
                    local_path.unlink(missing_ok=True)
                    continue
                attempts.append({
                    "source": candidate,
                    "exit_code": completed.returncode,
                    "error": completed.stderr[-1000:],
                })
                if completed.returncode == 0 and local_path.is_file() and local_path.stat().st_size > 0:
                    copied_from = candidate
                    break
                local_path.unlink(missing_ok=True)
            if copied_from is None:
                raise RuntimeError("all xrdcp access routes failed")
            local_record = dict(record)
            local_record["file_path"] = str(local_path)
            payload = count_tnp_shard(
                kind=kind,
                shard={
                    "schema_version": 1,
                    "shard_id": f"recovery_{kind}_{index:05d}",
                    "records": [local_record],
                },
                config=config,
                repo=repo,
                step_size=step_size,
            )
            _normalise_logical_path(payload, str(local_path), record)
            payload["recovery"] = {
                "mode": "lxplus local scratch after xrdcp",
                "logical_file": logical,
                "copied_from": copied_from,
                "xrdcp_attempts": attempts,
            }
            destination = output_dir / f"shard_recovery_{index:05d}.json"
            destination.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
            if payload["status"] != "success":
                raise RuntimeError(f"local TnP count status is {payload['status']}")
            successful += 1
            print(json.dumps({
                "index": index,
                "status": "recovered",
                "logical_file": logical,
                "copied_from": copied_from,
                "files_processed": payload["files_processed"],
                "files_failed": payload["files_failed"],
            }, sort_keys=True), flush=True)
        except Exception as exc:
            failed.append({
                "record": record,
                "error": f"{type(exc).__name__}: {exc}",
                "xrdcp_attempts": attempts,
            })
            print(json.dumps({
                "index": index,
                "status": "failed",
                "logical_file": logical,
                "error": f"{type(exc).__name__}: {exc}",
            }, sort_keys=True), flush=True)
        finally:
            local_path.unlink(missing_ok=True)
    summary = {
        "schema_version": 1,
        "measurement": manifest["measurement"],
        "kind": kind,
        "campaign_unresolved_records": len(campaign_records),
        "records_requested": len(indexed_records),
        "records_recovered": successful,
        "records_failed": len(failed),
        "worker_partition": {"worker_index": worker_index, "workers": workers},
        "failures": failed,
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch_dir),
        "created_unix": time.time(),
    }
    (output_dir / "recovery_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary


def finalize_permanent_skips(
    *,
    manifest_path: Path,
    output_records: Path,
    output_skips: Path,
    dataset_incomplete_threshold: float = 0.01,
) -> dict[str, Any]:
    """Freeze an explicitly audited campaign after irrecoverable files are skipped."""

    manifest = json.loads(manifest_path.read_text())
    campaign_path = Path(str(manifest["campaign_records"]))
    if not campaign_path.is_absolute():
        campaign_path = Path.cwd() / campaign_path
    campaign = json.loads(campaign_path.read_text())
    skipped_records = list(manifest.get("records") or [])
    skipped_paths = {str(record["file_path"]) for record in skipped_records}
    all_records = list(campaign.get("records") or [])
    retained_records = [
        record for record in all_records
        if str(record["file_path"]) not in skipped_paths
    ]
    if len(retained_records) + len(skipped_paths) != len(all_records):
        raise RuntimeError("permanent-skip accounting does not match campaign records")
    dataset_totals = Counter(str(record.get("dataset") or "unknown") for record in all_records)
    dataset_skips = Counter(str(record.get("dataset") or "unknown") for record in skipped_records)
    dataset_loss_summary = []
    for dataset in sorted(dataset_skips):
        total_files = dataset_totals[dataset]
        skipped_files = dataset_skips[dataset]
        fraction = skipped_files / total_files if total_files else 0.0
        dataset_loss_summary.append({
            "dataset": dataset,
            "files_total": total_files,
            "files_permanently_skipped": skipped_files,
            "file_loss_fraction": fraction,
            "incomplete": fraction > dataset_incomplete_threshold,
        })
    frozen = dict(campaign)
    frozen["records"] = retained_records
    frozen["status"] = "complete_with_permanent_skips"
    frozen["source_campaign_records"] = str(campaign_path)
    frozen["permanently_skipped_files"] = len(skipped_paths)
    frozen["files_before_permanent_skips"] = len(all_records)
    frozen["file_loss_fraction"] = len(skipped_paths) / len(all_records) if all_records else 0.0
    frozen["dataset_incomplete_threshold"] = dataset_incomplete_threshold
    frozen["incomplete_datasets"] = [
        item["dataset"] for item in dataset_loss_summary if item["incomplete"]
    ]
    frozen["created_unix"] = time.time()
    diagnostics = manifest.get("failure_diagnostics") or {}
    skip_payload = {
        "schema_version": 1,
        "measurement": manifest["measurement"],
        "status": "permanently_skipped",
        "files_before_permanent_skips": len(all_records),
        "files_retained": len(retained_records),
        "files_permanently_skipped": len(skipped_paths),
        "file_loss_fraction": len(skipped_paths) / len(all_records) if all_records else 0.0,
        "data_lumi_coverage_complete": not any(record.get("sample") == "data" for record in skipped_records),
        "dataset_incomplete_threshold": dataset_incomplete_threshold,
        "dataset_loss_summary": dataset_loss_summary,
        "incomplete_datasets": [
            item["dataset"] for item in dataset_loss_summary if item["incomplete"]
        ],
        "records": [
            {
                **record,
                "failure_diagnostic": diagnostics.get(str(record["file_path"])),
                "alternate_access_attempted": True,
                "permanently_skipped": True,
            }
            for record in skipped_records
        ],
        "created_unix": time.time(),
    }
    output_records.parent.mkdir(parents=True, exist_ok=True)
    output_skips.parent.mkdir(parents=True, exist_ok=True)
    output_records.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    output_skips.write_text(json.dumps(skip_payload, indent=2, sort_keys=True) + "\n")
    return {
        "measurement": manifest["measurement"],
        "files_before_permanent_skips": len(all_records),
        "files_retained": len(retained_records),
        "files_permanently_skipped": len(skipped_paths),
        "file_loss_fraction": skip_payload["file_loss_fraction"],
        "data_lumi_coverage_complete": skip_payload["data_lumi_coverage_complete"],
        "incomplete_datasets": skip_payload["incomplete_datasets"],
        "output_records": str(output_records),
        "output_skips": str(output_skips),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--records", type=Path, required=True)
    manifest_parser.add_argument("--workdir", type=Path, required=True)
    manifest_parser.add_argument("--recovery-dir", type=Path, action="append", default=[])
    manifest_parser.add_argument("--output", type=Path, required=True)
    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--kind", choices=("electron", "muon"), required=True)
    recover_parser.add_argument("--manifest", type=Path, required=True)
    recover_parser.add_argument("--config", type=Path, required=True)
    recover_parser.add_argument("--repo", type=Path, required=True)
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--scratch-dir", type=Path, required=True)
    recover_parser.add_argument("--step-size", type=int, default=100_000)
    recover_parser.add_argument("--worker-index", type=int, default=0)
    recover_parser.add_argument("--workers", type=int, default=1)
    skip_parser = subparsers.add_parser("finalize-skips")
    skip_parser.add_argument("--manifest", type=Path, required=True)
    skip_parser.add_argument("--output-records", type=Path, required=True)
    skip_parser.add_argument("--output-skips", type=Path, required=True)
    skip_parser.add_argument("--dataset-incomplete-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        result = build_residual_manifest(args.records, args.workdir, args.recovery_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif args.command == "recover":
        result = recover(
            kind=args.kind,
            manifest_path=args.manifest,
            config_path=args.config,
            repo=args.repo,
            output_dir=args.output_dir,
            scratch_dir=args.scratch_dir,
            step_size=args.step_size,
            worker_index=args.worker_index,
            workers=args.workers,
        )
    else:
        result = finalize_permanent_skips(
            manifest_path=args.manifest,
            output_records=args.output_records,
            output_skips=args.output_skips,
            dataset_incomplete_threshold=args.dataset_incomplete_threshold,
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "failures", "failure_diagnostics"}}, indent=2, sort_keys=True))
    return 0 if not result.get("files_unresolved", result.get("records_failed", 0)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
