#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN = Path(
    "/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/"
    "shape_hists_2024_jesFlavorQCD_highdm60_20260726"
)
SCHEdd = "bigbird24"
OWNER = "taiwoo"
OWNER_JOB_LIMIT = 100_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
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


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result


def parse_cluster(output: str) -> int:
    match = re.search(r"(?m)^(\d+)\.\d+\s+-\s+\1\.\d+\s*$", output.strip())
    if not match:
        raise RuntimeError(f"cannot parse cluster from condor_submit: {output}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign.absolute()
    if not str(campaign).startswith("/eos/user/t/taiwoo/"):
        raise ValueError("campaign must be on the approved EOS user path")
    manifest_path = campaign / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] == "submitted":
        print(
            json.dumps(
                {
                    "status": "already_submitted",
                    "cluster_ids": manifest["submission"]["cluster_ids"],
                },
                sort_keys=True,
            )
        )
        return 0
    if manifest["status"] != "prepared_not_submitted":
        raise RuntimeError(f"campaign is not submit-ready: {manifest['status']}")

    jobs = int(manifest["submit"]["jobs"])
    fingerprint = str(manifest["campaign_fingerprint"])
    constraint = f'CampaignFingerprint == "{fingerprint}"'
    existing = run(
        [
            "condor_q",
            "-name",
            SCHEdd,
            OWNER,
            "-constraint",
            constraint,
            "-af",
            "ClusterId",
            "ProcId",
        ]
    )
    if existing.stdout.strip():
        raise RuntimeError("an equivalent live campaign already exists")
    receipts = campaign / "submission_receipts.jsonl"
    if receipts.is_file() and receipts.read_text().strip():
        raise RuntimeError("a submission receipt already exists")

    proxy = next(
        Path(item.strip())
        for line in Path(manifest["submit"]["file"]).read_text().splitlines()
        if line.startswith("transfer_input_files =")
        for item in line.split("=", 1)[1].split(",")
        if Path(item.strip()).name.startswith("x509up_u")
    )
    proxy_check = run(
        ["voms-proxy-info", "-file", str(proxy), "-timeleft"],
        check=False,
    )
    timeleft = int(proxy_check.stdout.strip() or 0)
    if proxy_check.returncode != 0 or timeleft < 43_200:
        print(
            json.dumps(
                {
                    "status": "proxy_renewal_required",
                    "timeleft_seconds": timeleft,
                },
                sort_keys=True,
            )
        )
        return 3

    live = run(
        ["condor_q", "-name", SCHEdd, OWNER, "-af", "ClusterId", "ProcId"]
    )
    live_rows = sum(1 for line in live.stdout.splitlines() if line.strip())
    quota_report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owner_live_materialized_jobs": live_rows,
        "new_jobs": jobs,
        "owner_job_limit": OWNER_JOB_LIMIT,
        "fits_eagerly": live_rows + jobs <= OWNER_JOB_LIMIT,
    }
    write_json(campaign / "reports" / "owner_quota_before_submission.json", quota_report)
    if not quota_report["fits_eagerly"]:
        print(json.dumps({"status": "wait_for_owner_quota", **quota_report}, sort_keys=True))
        return 0

    worker = Path(manifest["bundles"]["worker"]["path"])
    if sha256(worker) != manifest["bundles"]["worker"]["sha256"]:
        raise RuntimeError("worker bundle checksum mismatch")
    with tarfile.open(worker, "r:gz") as archive:
        source = archive.extractfile("workflow/build_flat_boosted_recoil_hists.py")
        if source is None:
            raise RuntimeError("histogram builder is absent from worker bundle")
        text = source.read().decode()
    if "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR" not in text:
        raise RuntimeError("worker bundle does not contain the adopted 60-bin scheme")

    submit_file = Path(manifest["submit"]["file"])
    if sha256(submit_file) != manifest["submit"]["file_sha256"]:
        raise RuntimeError("submit file checksum mismatch")
    dryrun = campaign / "reports" / "condor_dry_run.ads"
    result = run(["condor_submit", "-dry-run", str(dryrun), str(submit_file)])
    dry_text = dryrun.read_text(errors="replace")
    if "/afs/" in dry_text or fingerprint not in dry_text:
        raise RuntimeError("Condor dry-run path/fingerprint audit failed")

    submitted = run(["condor_submit", "-terse", str(submit_file)])
    cluster = parse_cluster(submitted.stdout)
    submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = {
        "schema_version": "jesFlavorQCD_highdm60_submission_receipt_v1",
        "submitted_at": submitted_at,
        "campaign_fingerprint": fingerprint,
        "nuisance": "jesFlavorQCD",
        "cluster_id": cluster,
        "jobs": jobs,
        "materialization_mode": "eager",
        "submit_file": str(submit_file),
        "submit_file_sha256": manifest["submit"]["file_sha256"],
        "arguments_sha256": manifest["submit"]["arguments_sha256"],
        "condor_submit_stdout": submitted.stdout.strip(),
        "condor_submit_stderr": submitted.stderr.strip(),
        "proxy_timeleft_seconds": timeleft,
        "owner_quota_before_submission": quota_report,
    }
    receipts.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")

    queue = run(
        [
            "condor_q",
            "-name",
            SCHEdd,
            OWNER,
            "-constraint",
            constraint,
            "-af",
            "ClusterId",
            "ProcId",
            "JobStatus",
        ]
    )
    rows = [line.split() for line in queue.stdout.splitlines() if line.strip()]
    if not rows or {int(row[0]) for row in rows} != {cluster}:
        raise RuntimeError("submitted cluster is absent from the initial queue")
    state_counts: dict[str, int] = {}
    for row in rows:
        state_counts[row[2]] = state_counts.get(row[2], 0) + 1

    manifest["status"] = "submitted"
    manifest["submission"] = {
        "status": "submitted_active",
        "submitted_at": submitted_at,
        "cluster_ids": [cluster],
        "submitted_jobs": jobs,
        "initial_materialized_rows": len(rows),
        "initial_queue_by_job_status": state_counts,
        "receipt": receipt,
        "proxy_check": {
            "status": "valid",
            "timeleft_seconds": timeleft,
        },
        "dry_run": {
            "path": str(dryrun),
            "sha256": sha256(dryrun),
            "size": dryrun.stat().st_size,
            "condor_submit_stdout": result.stdout[-2000:],
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        campaign / "monitoring_state.json",
        {
            "schema_version": "jesFlavorQCD_highdm60_monitoring_state_v1",
            "status": "submitted_active",
            "campaign_fingerprint": fingerprint,
            "submission_time": submitted_at,
            "cluster_ids": [cluster],
            "last_queue_state": {
                "checked_at": submitted_at,
                "materialized_rows": len(rows),
                "by_job_status": state_counts,
            },
            "completed_valid_outputs": 0,
            "invalid_outputs": 0,
            "accepted_unrecovered_failures": manifest["coverage"]["excluded_partitions"],
            "recovery_actions": [],
        },
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "cluster_id": cluster,
                "jobs": jobs,
                "initial_materialized_rows": len(rows),
                "initial_queue_by_job_status": state_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
