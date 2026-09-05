#!/usr/bin/env python3
"""Prepare an EOS-only HTCondor campaign for the diagonal-v3 feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ALLOWED_EOS_PREFIX = "/eos/user/t/taiwoo/"
DEFAULT_PROXY = "/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def require_eos(path: str, label: str) -> str:
    if not path.startswith(ALLOWED_EOS_PREFIX):
        raise ValueError(f"{label} must be under {ALLOWED_EOS_PREFIX}: {path}")
    lowered = path.lower()
    if "/afs/" in lowered or "/tmp/" in lowered:
        raise ValueError(f"{label} contains a forbidden non-EOS path: {path}")
    return path.rstrip("/")


def request_rows(
    manifest: dict[str, Any],
    *,
    eos_campaign: str,
    files_per_job: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("mc", "signal"):
        records = [
            {
                "root": item["root"],
                "sidecar": item["sidecar"],
                "events_read": int(item.get("events_read", 0)),
                "events_written": int(item.get("events_written", 0)),
                "root_size_bytes": int(item.get("root_size_bytes", 0)),
                "physical_dataset_ids": item.get("physical_dataset_ids", []),
                "processes": item.get("processes", []),
            }
            for item in manifest["shards"]
            if item["kind"] == kind
        ]
        for record in records:
            require_eos(str(record["root"]), "input ROOT")
            require_eos(str(record["sidecar"]), "input sidecar")
        for batch, start in enumerate(range(0, len(records), files_per_job)):
            name = f"{kind}_{batch:04d}"
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "batch": batch,
                    "request_name": f"{name}.json",
                    "request_eos": f"{eos_campaign}/requests/{name}.json",
                    "output_eos": f"{eos_campaign}/outputs/{kind}_cache_{batch:04d}.root",
                    "inputs": records[start : start + files_per_job],
                }
            )
    return rows


def worker_text(eos_campaign: str) -> str:
    return """#!/usr/bin/env bash
set -eo pipefail
REQUEST="${1:?missing absolute EOS request path}"
case "$REQUEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS request path: $REQUEST" >&2; exit 64 ;;
esac
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_swan/x86_64-el9-gcc13-opt/setup.sh
set -u
export X509_USER_PROXY="/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757"
test -s "$X509_USER_PROXY"
chmod 600 "$X509_USER_PROXY"
export PYTHONDONTWRITEBYTECODE=1
python3 -u __EOS_CAMPAIGN__/payload/build_expanded_feature_cache.py --worker "$REQUEST"
""".replace("__EOS_CAMPAIGN__", eos_campaign)


def submit_text(eos_campaign: str, queue_name: str, flavour: str) -> str:
    queue_path = f"{eos_campaign}/{queue_name}"
    return f"""universe = vanilla
initialdir = {eos_campaign}/condor
executable = {eos_campaign}/condor/run_diagonal_v3_cache_worker.sh
arguments = $(request)

should_transfer_files = NO

output = {eos_campaign}/logs/stdout.$(ClusterId).$(ProcId)
error = {eos_campaign}/logs/stderr.$(ClusterId).$(ProcId)
log = {eos_campaign}/logs/cluster.log

request_cpus = 1
request_memory = 4000MB
request_disk = 1000MB
+JobFlavour = "{flavour}"

queue request from {queue_path}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--eos-campaign", required=True)
    parser.add_argument("--files-per-job", type=int, default=50)
    opts = parser.parse_args()
    if opts.files_per_job < 1:
        raise ValueError("files-per-job must be positive")
    eos_campaign = require_eos(opts.eos_campaign, "EOS campaign")
    manifest = json.loads(opts.manifest.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("full-campaign manifest is incomplete")

    source_dir = Path(__file__).resolve().parent
    cache_worker = source_dir / "build_expanded_feature_cache.py"
    finalizer = source_dir / "finalize_diagonal_v3_cache.py"
    resolved_worker = (
        source_dir.parent.parent / "autonomous_allhad/highdm_resolved_categories.py"
    )
    if not all(path.is_file() for path in (cache_worker, finalizer, resolved_worker)):
        raise FileNotFoundError("required cache-worker source is missing")

    staging = opts.staging
    staging.mkdir(parents=True, exist_ok=True)
    for directory in (
        "requests",
        "pilot_requests",
        "pilot_outputs",
        "outputs",
        "logs",
        "condor",
        "payload",
    ):
        (staging / directory).mkdir(exist_ok=True)
    shutil.copy2(cache_worker, staging / "payload/build_expanded_feature_cache.py")
    shutil.copy2(finalizer, staging / "payload/finalize_diagonal_v3_cache.py")
    shutil.copy2(resolved_worker, staging / "payload/highdm_resolved_categories.py")

    rows = request_rows(
        manifest,
        eos_campaign=eos_campaign,
        files_per_job=opts.files_per_job,
    )
    for row in rows:
        write_json(
            staging / "requests" / str(row["request_name"]),
            {
                "schema_version": "gnn_lowdm_diagonal_v3_feature_cache_request_v1",
                "selection_mode": "diagonal_v3",
                "kind": row["kind"],
                "batch": row["batch"],
                "inputs": row["inputs"],
                "output": row["output_eos"],
            },
        )

    pilot = []
    for kind in ("mc", "signal"):
        source = next(row for row in rows if row["kind"] == kind)
        request_name = f"pilot_{kind}.json"
        request_eos = f"{eos_campaign}/pilot_requests/{request_name}"
        output_eos = f"{eos_campaign}/pilot_outputs/{kind}_pilot.root"
        write_json(
            staging / "pilot_requests" / request_name,
            {
                "schema_version": "gnn_lowdm_diagonal_v3_feature_cache_request_v1",
                "selection_mode": "diagonal_v3",
                "kind": kind,
                "batch": -1,
                "inputs": source["inputs"][:1],
                "output": output_eos,
            },
        )
        pilot.append({"request_eos": request_eos})
    remaining = rows
    (staging / "pilot_requests.txt").write_text(
        "".join(f"{row['request_eos']}\n" for row in pilot)
    )
    (staging / "remaining_requests.txt").write_text(
        "".join(f"{row['request_eos']}\n" for row in remaining)
    )
    (staging / "all_requests.txt").write_text(
        "".join(f"{row['request_eos']}\n" for row in rows)
    )
    wrapper = staging / "condor/run_diagonal_v3_cache_worker.sh"
    wrapper.write_text(worker_text(eos_campaign))
    wrapper.chmod(0o755)
    (staging / "condor/pilot.sub").write_text(
        submit_text(eos_campaign, "pilot_requests.txt", "microcentury")
    )
    (staging / "condor/remaining.sub").write_text(
        submit_text(eos_campaign, "remaining_requests.txt", "workday")
    )
    (staging / "condor/all.sub").write_text(
        submit_text(eos_campaign, "all_requests.txt", "workday")
    )

    payloads = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted((staging / "payload").iterdir())
    }
    production_requests = sorted((staging / "requests").glob("*.json"))
    pilot_request_files = sorted((staging / "pilot_requests").glob("*.json"))
    audit = {
        "status": "prepared",
        "schema_version": "gnn_lowdm_diagonal_v3_condor_campaign_v1",
        "source_manifest": str(opts.manifest),
        "source_manifest_sha256": sha256(opts.manifest),
        "eos_campaign": eos_campaign,
        "selection": (
            "feature_lowdm_preselection && Nt=0 && NW=0 && Nres=0 && Nb>=1; "
            "no MET/sqrt(HT), NISR, ISR-dphi, or !feature_SR requirement"
        ),
        "input_policy": "only mc and signal nominal intermediate ROOT shards",
        "input_files": sum(len(row["inputs"]) for row in rows),
        "jobs": len(rows),
        "pilot_jobs": len(pilot),
        "remaining_jobs": len(remaining),
        "pilot_input_files": 2,
        "files_per_job": opts.files_per_job,
        "job_counts": {
            kind: sum(row["kind"] == kind for row in rows)
            for kind in ("mc", "signal")
        },
        "input_file_counts": {
            kind: sum(
                len(row["inputs"]) for row in rows if row["kind"] == kind
            )
            for kind in ("mc", "signal")
        },
        "payloads": payloads,
        "request_sets": {
            "production": {
                "count": len(production_requests),
                "aggregate_sha256": aggregate_sha256(production_requests),
                "queue_sha256": sha256(staging / "remaining_requests.txt"),
            },
            "pilot": {
                "count": len(pilot_request_files),
                "aggregate_sha256": aggregate_sha256(pilot_request_files),
                "queue_sha256": sha256(staging / "pilot_requests.txt"),
            },
        },
        "test_partition_touched": False,
        "submission": {
            "schedd": "bigbird24.cern.ch",
            "pool": "eossubmit",
            "pilot_command": (
                f"module load lxbatch/eossubmit && condor_submit -name bigbird24 "
                f"{eos_campaign}/condor/pilot.sub"
            ),
            "remaining_command_after_pilot_validation": (
                f"module load lxbatch/eossubmit && condor_submit -name bigbird24 "
                f"{eos_campaign}/condor/remaining.sub"
            ),
        },
    }
    write_json(staging / "campaign_manifest.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
