#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_2024_shape_hist_campaign_fullselection import (  # noqa: E402
    build_worker_bundle,
    require_eos,
    sha256,
    write_json,
)
from prepare_2024_shape_hist_condor_pairs import (  # noqa: E402
    canonical_digest,
    submit_text,
    wrapper_text,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_SOURCE = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_fullselection_v7_condorpairs_20260725"
)
DEFAULT_CAMPAIGN = (
    DEFAULT_REPO
    / "autonomous_allhad/workflow/shape_hists_2024_jesFlavorQCD_highdm60_20260726"
)
NUISANCE = "jesFlavorQCD"
EXCLUDED_PARTITIONS = {
    "GJ_pairpart_00066": "accepted unrecovered failure from the completed 54-bin campaign",
    "WtoLNu_pairpart_00039": "accepted unrecovered failure from the completed 54-bin campaign",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the targeted jesFlavorQCD NanoAOD reread needed for the "
            "extra High-dM 60-bin category while reusing frozen v7 partitions."
        )
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--source-campaign", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()

    repo = args.repo.absolute()
    source = args.source_campaign.absolute()
    campaign = args.campaign.absolute()
    for label, path in (
        ("repository", repo),
        ("source campaign", source),
        ("target campaign", campaign),
    ):
        require_eos(path, label)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty campaign: {campaign}")

    source_manifest = read_json(source / "manifest.json")
    source_partitions = read_json(source / "partitions.json")
    partitions = [
        item
        for item in source_partitions.get("partitions") or []
        if str(item["partition_id"]) not in EXCLUDED_PARTITIONS
    ]
    observed_exclusions = {
        str(item["partition_id"])
        for item in source_partitions.get("partitions") or []
        if str(item["partition_id"]) in EXCLUDED_PARTITIONS
    }
    if observed_exclusions != set(EXCLUDED_PARTITIONS):
        raise RuntimeError("the exact accepted failure partitions were not found")
    if len(partitions) + len(observed_exclusions) != 6479:
        raise RuntimeError("unexpected frozen partition count")

    for directory in ("bundles", "condor", "logs", "outputs", "reports"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)

    py38 = repo / "condor/py38.tgz"
    proxy = repo / "analysis" / "proxy" / "x509up_u147757"
    btag = repo / "analysis" / "hists" / "btageff2024.merged"
    source_payload = Path(
        ((source_manifest.get("bundles") or {}).get("payload") or {}).get("path")
        or ""
    )
    for label, path in (
        ("Python archive", py38),
        ("proxy", proxy),
        ("b-tag efficiency", btag),
        ("source payload bundle", source_payload),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    worker_path = campaign / "bundles" / "shape_hist_2024_worker.tgz"
    payload_path = campaign / "bundles" / "shape_hist_2024_payloads.tgz"
    worker = build_worker_bundle(repo, worker_path)
    shutil.copy2(source_payload, payload_path)
    payload = {
        "path": str(payload_path),
        "source": str(source_payload),
        "size": payload_path.stat().st_size,
        "sha256": sha256(payload_path),
    }
    btag_sha256 = sha256(btag)

    fingerprint_payload = {
        "purpose": "exact_jesFlavorQCD_propagation_to_highdm60_extra_category",
        "source_campaign_fingerprint": source_manifest["campaign_fingerprint"],
        "partition_digests": [item["record_digest"] for item in partitions],
        "excluded_partitions": EXCLUDED_PARTITIONS,
        "nuisance": NUISANCE,
        "highdm_scheme": "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR",
        "worker_bundle_sha256": worker["sha256"],
        "payload_bundle_sha256": payload["sha256"],
        "python_bundle_sha256": sha256(py38),
        "btag_efficiency_sha256": btag_sha256,
        "request": {
            "cpus": 8,
            "memory_mb": 8000,
            "disk_mb": 10000,
            "job_flavour": "longlunch",
        },
    }
    fingerprint = canonical_digest(fingerprint_payload)

    wrapper = campaign / "condor" / "run_jesFlavorQCD_highdm60.sh"
    wrapper.write_text(wrapper_text(proxy.name, btag_sha256))
    wrapper.chmod(0o755)
    rows = []
    source_records: set[str] = set()
    expected_events = 0
    expected_segments = 0
    for partition in partitions:
        process = str(partition["process_group"])
        partition_id = str(partition["partition_id"])
        output_dir = campaign / "outputs" / NUISANCE / process
        log_dir = campaign / "logs" / NUISANCE / process
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        shard = read_json(Path(partition["path"]))
        source_records.update(str(item) for item in shard["source_record_digests"])
        expected_events += int(partition["expected_events"])
        expected_segments += int(partition["expected_segments"])
        rows.append(
            " ".join(
                [
                    f"{NUISANCE}_{partition_id}",
                    str(partition["path"]),
                    str(partition["sha256"]),
                    NUISANCE,
                    str(output_dir / f"{partition_id}.json.gz"),
                    str(output_dir / f"{partition_id}.meta.json"),
                    str(log_dir),
                    str(partition["expected_events"]),
                    str(partition["expected_segments"]),
                ]
            )
        )

    arguments = campaign / "condor" / f"arguments_{NUISANCE}.txt"
    arguments.write_text("\n".join(rows) + "\n")
    submit = campaign / "condor" / f"submit_{NUISANCE}.sub"
    submit.write_text(
        submit_text(
            wrapper=wrapper,
            arguments=arguments,
            logs=campaign / "logs",
            py38=py38,
            worker_bundle=worker_path,
            payload_bundle=payload_path,
            proxy=proxy,
            initialdir=campaign / "condor",
            request_memory_mb=8000,
            request_disk_mb=10000,
            request_cpus=8,
            job_flavour="longlunch",
            campaign_name=campaign.name,
            campaign_fingerprint=fingerprint,
            segment_events=250_000,
        )
    )

    manifest = {
        "schema_version": "jesFlavorQCD_highdm60_targeted_campaign_v1",
        "status": "prepared_not_submitted",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": str(campaign),
        "campaign_fingerprint": fingerprint,
        "source_campaign": str(source),
        "source_campaign_fingerprint": source_manifest["campaign_fingerprint"],
        "physics_policy": {
            "bjet_pt_min_gev": 30.0,
            "nuisance": NUISANCE,
            "directions": [f"{NUISANCE}Up", f"{NUISANCE}Down"],
            "object_variations_are_weight_only": False,
            "selection_category_search_bin_and_xaxis_migration": True,
            "highdm_scheme": "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR",
            "highdm_bins": 60,
            "lowdm_bins": 42,
        },
        "coverage": {
            "partition_count": len(partitions),
            "source_record_count": len(source_records),
            "expected_events": expected_events,
            "expected_segments": expected_segments,
            "excluded_partitions": [
                {"partition_id": key, "reason": value}
                for key, value in sorted(EXCLUDED_PARTITIONS.items())
            ],
            "frozen_local_seed_source_records": 30,
            "frozen_local_seed_policy": (
                "preserved and not rerun; its exact existing 54-bin variation "
                "is retained, while the new six-bin extension records this "
                "small missing variation coverage explicitly"
            ),
            "recovery_policy": "none, per user instruction",
        },
        "runtime_policy": {
            "pool": "bigbird24",
            "persistent_paths": "EOS only",
            "python_archive": str(py38),
            "request_cpus": 8,
            "request_memory_mb": 8000,
            "request_disk_mb": 10000,
            "job_flavour": "longlunch",
        },
        "bundles": {"worker": worker, "payload": payload},
        "submit": {
            "file": str(submit),
            "file_sha256": sha256(submit),
            "arguments": str(arguments),
            "arguments_sha256": sha256(arguments),
            "jobs": len(rows),
        },
        "submission": {
            "status": "not_submitted",
            "cluster_ids": [],
            "submitted_at": None,
        },
        "fingerprint_payload": fingerprint_payload,
    }
    write_json(campaign / "manifest.json", manifest)
    write_json(
        campaign / "partitions.json",
        {
            "schema_version": "jesFlavorQCD_highdm60_targeted_partitions_v1",
            "status": "complete_with_explicit_exclusions",
            "campaign_fingerprint": fingerprint,
            "partitions": partitions,
            "excluded_partitions": sorted(EXCLUDED_PARTITIONS),
        },
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "fingerprint": fingerprint,
                "jobs": len(rows),
                "source_records": len(source_records),
                "expected_events": expected_events,
                "worker_bundle_sha256": worker["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
