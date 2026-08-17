#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN = Path(
    "/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/"
    "shape_hists_2024_fullselection_v8_condorpairs_20260725"
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


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, text=True, capture_output=True)
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"stdout: {process.stdout[-4000:]}\n"
            f"stderr: {process.stderr[-4000:]}"
        )
    return process


def classad_constraint(fingerprint: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid campaign fingerprint")
    return f'CampaignFingerprint == "{fingerprint}"'


def query_queue(schedd: str, owner: str, fingerprint: str) -> list[dict[str, Any]]:
    process = run(
        [
            "condor_q",
            "-name",
            schedd,
            owner,
            "-constraint",
            classad_constraint(fingerprint),
            "-af",
            "ClusterId",
            "ProcId",
            "JobStatus",
            "QDate",
            "JobCurrentStartDate",
            "NuisancePair",
        ]
    )
    rows = []
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        start_time = 0 if fields[4].lower() == "undefined" else int(fields[4])
        rows.append(
            {
                "cluster_id": int(fields[0]),
                "proc_id": int(fields[1]),
                "job_status": int(fields[2]),
                "qdate": int(fields[3]),
                "job_current_start_date": start_time,
                "nuisance": fields[5],
            }
        )
    return rows


def query_history(schedd: str, fingerprint: str) -> dict[str, Any]:
    completed_since = int(time.time()) - 2 * 24 * 60 * 60
    command = [
            "condor_history",
            "-name",
            schedd,
            "-completedsince",
            str(completed_since),
            "-constraint",
            classad_constraint(fingerprint),
            "-limit",
            "100",
            "-af",
            "ClusterId",
            "ProcId",
            "JobStatus",
            "ExitCode",
            "NuisancePair",
        ]
    try:
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "bounded_timeout",
            "timeout_seconds": 10,
            "completed_since": completed_since,
            "rows": [],
        }
    if process.returncode != 0:
        raise RuntimeError(
            f"recent condor_history query failed ({process.returncode}): "
            f"{process.stderr[-2000:]}"
        )
    rows = []
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        rows.append(
            {
                "cluster_id": int(fields[0]),
                "proc_id": int(fields[1]),
                "job_status": int(fields[2]),
                "exit_code": int(fields[3]),
                "nuisance": fields[4],
            }
        )
    return {
        "status": "complete",
        "timeout_seconds": 10,
        "completed_since": completed_since,
        "rows": rows,
    }


def existing_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def check_equivalent_campaigns(
    campaign: Path,
    manifest: dict[str, Any],
    schedd: str,
    owner: str,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    fingerprint = str(manifest["campaign_fingerprint"])
    workflow = campaign.parent
    matches = []
    for path in workflow.glob("shape_hists_*/manifest.json"):
        try:
            candidate = read_json(path)
        except Exception:
            continue
        if str(candidate.get("campaign_fingerprint") or "") != fingerprint:
            continue
        matches.append(
            {
                "manifest": str(path),
                "same_campaign": path.parent == campaign,
                "status": candidate.get("status"),
                "cluster_ids": (
                    candidate.get("submission") or {}
                ).get("cluster_ids")
                or [],
            }
        )
    queue = query_queue(schedd, owner, fingerprint)
    history_query = query_history(schedd, fingerprint)
    history = list(history_query["rows"])
    receipt_clusters = {int(item["cluster_id"]) for item in receipts}
    external_queue = [
        row for row in queue if int(row["cluster_id"]) not in receipt_clusters
    ]
    external_history = [
        row for row in history if int(row["cluster_id"]) not in receipt_clusters
    ]
    other_manifests = [item for item in matches if not item["same_campaign"]]
    passed = not other_manifests and not external_queue and not external_history
    return {
        "schema_version": "shape_histogram_2024_equivalent_campaign_check_v1",
        "status": "complete" if passed else "equivalent_campaign_found",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign_fingerprint": fingerprint,
        "manifest_matches": matches,
        "queue_rows": queue[:100],
        "queue_row_count": len(queue),
        "history_rows": history[:100],
        "history_row_count": len(history),
        "history_query": history_query,
        "external_queue_rows": external_queue[:100],
        "external_history_rows": external_history[:100],
        "passed": passed,
        "decision_basis": (
            "exact fingerprint manifest scan, live queue scan, receipts, and "
            "a bounded recent-history query"
        ),
    }


def check_proxy(proxy: Path, minimum_seconds: int) -> dict[str, Any]:
    if not proxy.is_file():
        return {
            "status": "renewal_required",
            "reason": "proxy file is missing",
            "path": str(proxy),
            "timeleft_seconds": 0,
        }
    process = run(
        ["voms-proxy-info", "-file", str(proxy), "-timeleft"],
        check=False,
    )
    try:
        timeleft = int(process.stdout.strip().splitlines()[-1])
    except Exception:
        timeleft = 0
    valid = process.returncode == 0 and timeleft >= minimum_seconds
    return {
        "status": "valid" if valid else "renewal_required",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": str(proxy),
        "timeleft_seconds": timeleft,
        "minimum_required_seconds": minimum_seconds,
        "command_exit_status": process.returncode,
        "stderr_tail": process.stderr[-1000:],
    }


def dry_run(
    campaign: Path,
    submit_file: Path,
    *,
    request_cpus: int,
    request_memory_mb: int,
    request_disk_mb: int,
) -> dict[str, Any]:
    target = campaign / "reports" / "condor_dry_run.ads"
    reused = (
        target.is_file()
        and target.stat().st_size > 0
        and target.stat().st_mtime >= submit_file.stat().st_mtime
    )
    if reused:
        stdout = "reused existing dry-run artifact"
        stderr = ""
    else:
        process = run(
            [
                "condor_submit",
                "-dry-run",
                str(target),
                str(submit_file),
            ]
        )
        stdout = process.stdout
        stderr = process.stderr
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("condor dry-run produced no ClassAd artifact")
    text = target.read_text(errors="replace")
    if "/afs/" in text:
        raise RuntimeError("Condor dry-run contains an AFS path")
    required = [
        "CampaignFingerprint",
        "NuisancePair",
        f"RequestCpus={request_cpus}",
        f"RequestMemory={request_memory_mb}",
        f"RequestDisk={request_disk_mb * 1024}",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Condor dry-run is missing required fields: {missing}")
    return {
        "schema_version": "shape_histogram_2024_condor_dry_run_v1",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submit_file": str(submit_file),
        "artifact": str(target),
        "artifact_size": target.stat().st_size,
        "artifact_sha256": sha256(target),
        "condor_submit_stdout": stdout[-4000:],
        "condor_submit_stderr": stderr[-4000:],
        "reused_existing_artifact": reused,
        "required_fields": required,
        "afs_paths": 0,
    }


def parse_cluster_id(text: str) -> int:
    match = re.search(r"(?m)^(\d+)\.\d+\s+-\s+\1\.\d+\s*$", text.strip())
    if not match:
        match = re.search(r"cluster\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"cannot parse cluster id from condor_submit output: {text}")
    return int(match.group(1))


def submit_file_with_quota_fallback(
    submit_file: Path,
    *,
    max_materialize: int = 350,
) -> tuple[subprocess.CompletedProcess[str], str]:
    process = run(
        ["condor_submit", "-terse", str(submit_file)],
        check=False,
    )
    combined = f"{process.stdout}\n{process.stderr}"
    if process.returncode == 0:
        return process, "eager"
    if "MAX_JOBS_PER_OWNER" not in combined:
        raise RuntimeError(
            f"condor_submit failed ({process.returncode}): {combined[-4000:]}"
        )
    process = run(
        [
            "condor_submit",
            "-terse",
            "-append",
            f"max_materialize = {max_materialize}",
            str(submit_file),
        ]
    )
    return process, f"late_materialization_max_{max_materialize}"


def queue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    now = int(time.time())
    states = Counter(int(row["job_status"]) for row in rows)
    nuisances = Counter(str(row["nuisance"]) for row in rows)
    waits = [
        max(0, now - int(row["qdate"]))
        for row in rows
        if int(row["job_status"]) == 1
    ]
    return {
        "rows": len(rows),
        "by_job_status": {str(key): value for key, value in sorted(states.items())},
        "by_nuisance": dict(sorted(nuisances.items())),
        "idle_wait_seconds": {
            "count": len(waits),
            "min": min(waits) if waits else None,
            "max": max(waits) if waits else None,
            "mean": (sum(waits) / len(waits)) if waits else None,
        },
        "cluster_ids": sorted({int(row["cluster_id"]) for row in rows}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit and submit the 2024 nuisance-pair campaign idempotently."
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--schedd", default="bigbird24")
    parser.add_argument("--owner", default=os.environ.get("USER") or "taiwoo")
    parser.add_argument("--minimum-proxy-seconds", type=int, default=43_200)
    parser.add_argument(
        "--max-new-submissions",
        type=int,
        default=0,
        help="Stop successfully after this many new receipts; zero means no limit.",
    )
    args = parser.parse_args(argv)
    if args.max_new_submissions < 0:
        raise ValueError("--max-new-submissions cannot be negative")
    campaign = args.campaign.absolute()
    if not str(campaign).startswith("/eos/user/t/taiwoo/"):
        raise ValueError("campaign must be under the approved EOS user path")
    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {
        "prepared_not_submitted",
        "submitting",
        "submitted",
    }:
        raise RuntimeError(f"campaign is not submit-ready: {manifest.get('status')}")

    static_audit_path = campaign / "reports/static_audit.json"
    benchmark_path = Path(
        str(
            (manifest.get("benchmark") or {}).get(
                "representative_condor_layout_report"
            )
            or campaign / "benchmarks/submission_qualification/report.json"
        )
    )
    if not benchmark_path.is_file():
        candidates = sorted((campaign / "benchmarks").glob("*/report.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                "cannot resolve a unique representative Condor-layout benchmark"
            )
        benchmark_path = candidates[0]
    static_audit = read_json(static_audit_path)
    benchmark = read_json(benchmark_path)
    if static_audit.get("status") != "complete":
        raise RuntimeError("static campaign audit is incomplete")
    if benchmark.get("status") != "complete":
        raise RuntimeError("representative Condor-layout benchmark is incomplete")
    if float(benchmark.get("worker_wall_time_s") or 1e99) > 3600:
        raise RuntimeError("representative worker walltime exceeds one hour")
    peak_total_rss_kb = int(
        benchmark.get("peak_total_rss_kb")
        or benchmark.get("maximum_resident_set_kb")
        or 1e99
    )
    if peak_total_rss_kb > 7_500_000:
        raise RuntimeError("representative worker memory is too close to the request")

    receipts_path = campaign / "submission_receipts.jsonl"
    receipts = existing_receipts(receipts_path)
    equivalent = check_equivalent_campaigns(
        campaign,
        manifest,
        args.schedd,
        args.owner,
        receipts,
    )
    write_json(campaign / "reports/equivalent_campaign_check.json", equivalent)
    if not equivalent["passed"]:
        raise RuntimeError("equivalent variation campaign already exists")

    first_submit = Path(manifest["submit_files"][0]["submit_file"])
    runtime_policy = manifest["runtime_policy"]
    dryrun = dry_run(
        campaign,
        first_submit,
        request_cpus=int(runtime_policy["request_cpus"]),
        request_memory_mb=int(runtime_policy["request_memory_mb"]),
        request_disk_mb=int(runtime_policy["request_disk_mb"]),
    )
    write_json(campaign / "reports/condor_dry_run.json", dryrun)

    proxy = Path(manifest["bundles"]["python"]["path"]).parents[1] / (
        f"analysis/proxy/x509up_u{os.getuid()}"
    )
    # The Python archive lives in <repo>/condor; resolve the proxy from the
    # exact transferred path recorded in the first submit file.
    transfer_line = next(
        line
        for line in first_submit.read_text().splitlines()
        if line.startswith("transfer_input_files =")
    )
    transferred = [Path(item.strip()) for item in transfer_line.split("=", 1)[1].split(",")]
    proxy_candidates = [item for item in transferred if item.name.startswith("x509up_u")]
    if len(proxy_candidates) != 1:
        raise RuntimeError("cannot resolve the exact transferred proxy")
    proxy = proxy_candidates[0]
    proxy_check = check_proxy(proxy, args.minimum_proxy_seconds)
    write_json(campaign / "reports/proxy_check_before_submission.json", proxy_check)
    if proxy_check["status"] != "valid":
        print(json.dumps(proxy_check, sort_keys=True))
        return 3

    received = {str(item["nuisance"]): item for item in receipts}
    manifest["status"] = "submitting"
    manifest["submission"]["status"] = "submitting"
    manifest["submission"]["equivalent_campaign_check"] = equivalent
    manifest["submission"]["proxy_check"] = proxy_check
    manifest["submission"]["dry_run"] = dryrun
    write_json(manifest_path, manifest)

    new_submissions = 0
    for item in manifest["submit_files"]:
        nuisance = str(item["nuisance"])
        if nuisance in received:
            continue
        submit_file = Path(item["submit_file"])
        if sha256(submit_file) != item["submit_file_sha256"]:
            raise RuntimeError(f"submit file checksum changed: {submit_file}")
        process = run(["condor_submit", "-terse", str(submit_file)])
        materialization_mode = "eager"
        cluster_id = parse_cluster_id(process.stdout)
        receipt = {
            "schema_version": "shape_histogram_2024_submission_receipt_v1",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "campaign_fingerprint": manifest["campaign_fingerprint"],
            "nuisance": nuisance,
            "cluster_id": cluster_id,
            "jobs": int(item["jobs"]),
            "submit_file": str(submit_file),
            "submit_file_sha256": item["submit_file_sha256"],
            "arguments_file_sha256": item["arguments_file_sha256"],
            "condor_submit_stdout": process.stdout.strip(),
            "condor_submit_stderr": process.stderr.strip(),
            "materialization_mode": materialization_mode,
        }
        append_jsonl(receipts_path, receipt)
        received[nuisance] = receipt
        new_submissions += 1
        manifest["submission"]["cluster_ids"] = [
            int(received[name]["cluster_id"])
            for name in manifest["physics_policy"]["nuisance_sets"]
            if name in received
        ]
        manifest["submission"]["receipts"] = [
            received[name]
            for name in manifest["physics_policy"]["nuisance_sets"]
            if name in received
        ]
        write_json(manifest_path, manifest)
        if (
            args.max_new_submissions
            and new_submissions >= args.max_new_submissions
        ):
            break

    missing_nuisances = [
        name
        for name in manifest["physics_policy"]["nuisance_sets"]
        if name not in received
    ]
    if missing_nuisances and args.max_new_submissions:
        print(
            json.dumps(
                {
                    "status": "submitting",
                    "new_submissions": new_submissions,
                    "receipts": len(received),
                    "cluster_ids": [
                        int(received[name]["cluster_id"])
                        for name in manifest["physics_policy"]["nuisance_sets"]
                        if name in received
                    ],
                    "missing_nuisances": missing_nuisances,
                },
                sort_keys=True,
            )
        )
        return 0
    if missing_nuisances:
        raise RuntimeError("not all 20 nuisance clusters were submitted")
    rows = query_queue(
        args.schedd,
        args.owner,
        str(manifest["campaign_fingerprint"]),
    )
    initial = queue_summary(rows)
    expected_jobs = int(manifest["jobs"]["total"])
    expected_clusters = {
        int(received[name]["cluster_id"])
        for name in manifest["physics_policy"]["nuisance_sets"]
    }
    if not (0 < initial["rows"] <= expected_jobs):
        raise RuntimeError(
            f"invalid initial materialized queue row count: "
            f"{initial['rows']} vs submitted {expected_jobs}"
        )
    if set(initial["cluster_ids"]) != expected_clusters:
        raise RuntimeError("initial queue cluster IDs do not match receipts")
    if set(initial["by_nuisance"]) != set(manifest["physics_policy"]["nuisance_sets"]):
        raise RuntimeError("initial queue nuisance coverage is incomplete")

    submitted_at = min(
        str(item["submitted_at"]) for item in received.values()
    )
    manifest["status"] = "submitted"
    manifest["submission"].update(
        {
            "status": "submitted",
            "submitted_at": submitted_at,
            "cluster_ids": sorted(expected_clusters),
            "initial_queue": initial,
            "submitted_jobs": expected_jobs,
            "materialized_queue_rows": initial["rows"],
        }
    )
    write_json(manifest_path, manifest)
    state = {
        "schema_version": "shape_histogram_2024_variation_monitoring_state_v1",
        "status": "submitted_active",
        "campaign": str(campaign),
        "campaign_fingerprint": manifest["campaign_fingerprint"],
        "submission_time": submitted_at,
        "cluster_ids": sorted(expected_clusters),
        "last_queue_state": {
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **initial,
        },
        "last_email_time": None,
        "monitoring_automation": "disabled_by_user",
    }
    write_json(campaign / "monitoring_state.json", state)
    print(
        json.dumps(
            {
                "status": "submitted",
                "campaign_fingerprint": manifest["campaign_fingerprint"],
                "cluster_ids": sorted(expected_clusters),
                "jobs": expected_jobs,
                "initial_queue": initial,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
