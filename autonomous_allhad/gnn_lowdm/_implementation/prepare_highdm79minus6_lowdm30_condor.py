#!/usr/bin/env python3
"""Prepare the orthogonal 2024 High-dM(79-6) + Low-dM(30) limit campaign.

The first six High-dM bins are the Nt=Nw=Nres=0 recoil bins and overlap the
new Low-dM selection.  They are removed at card-construction time and replaced
by the six five-bin GNN categories (30 Low-dM SR shape bins).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TOPOLOGIES = ("T2tt", "T2bW", "T2tb")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-base", type=Path, required=True)
    parser.add_argument("--low-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-size", type=int, default=4)
    parser.add_argument("--job-flavour", default="longlunch")
    args = parser.parse_args()

    for path in (args.high_base, args.low_base, args.output):
        if not str(path).startswith("/eos/"):
            raise RuntimeError(f"campaign input/output must be on EOS: {path}")
    if args.bundle_size < 1:
        raise ValueError("bundle size must be positive")

    condor = args.output / "condor"
    bundles = condor / "bundles"
    logs = condor / "logs"
    for path in (args.output, condor, bundles, logs):
        path.mkdir(parents=True, exist_ok=True)

    queue: list[str] = []
    jobs: list[dict] = []
    topology_summary: dict[str, dict] = {}
    all_points = 0
    all_combined = 0

    for topology in TOPOLOGIES:
        high_manifest = read_json(args.high_base / topology / "manifest.json")
        low_manifest = read_json(args.low_base / topology / "manifest.json")
        high_points = list(high_manifest["mass_points"])
        low_points = set(low_manifest["mass_points"])
        missing_low = sorted(set(high_points) - low_points)
        if missing_low:
            raise RuntimeError(
                f"{topology}: High-dM grid points absent from Low-dM signal: {missing_low}"
            )
        extra_low = sorted(low_points - set(high_points))
        records = [
            (mass, "combined" if mass in low_points else "highonly")
            for mass in high_points
        ]
        for start in range(0, len(records), args.bundle_size):
            chunk = records[start : start + args.bundle_size]
            name = f"{topology}_{start // args.bundle_size:03d}.txt"
            bundle = bundles / name
            bundle.write_text(
                "".join(f"{mass} {mode}\n" for mass, mode in chunk)
            )
            queue.append(f"{topology} {bundle}")
            jobs.append(
                {
                    "topology": topology,
                    "bundle": str(bundle),
                    "points": [
                        {"mass": mass, "mode": mode} for mass, mode in chunk
                    ],
                }
            )
        topology_summary[topology] = {
            "point_count": len(records),
            "combined_point_count": len(records),
            "high_only_point_count": 0,
            "all_lowdm_point_count": len(low_points),
            "lowdm_points_outside_highdm_grid": extra_low,
            "mass_points": high_points,
        }
        all_points += len(records)
        all_combined += len(records)

    queue_path = condor / "queue.txt"
    pilot_queue_path = condor / "pilot_queue.txt"
    remaining_queue_path = condor / "remaining_queue.txt"
    queue_path.write_text("\n".join(queue) + "\n")
    pilot_queue_path.write_text(queue[0] + "\n")
    remaining_queue_path.write_text("\n".join(queue[1:]) + "\n")
    executable = condor / "run_highdm79minus6_lowdm30_bundle.sh"
    worker_source = Path(__file__).resolve().parent / executable.name
    if not worker_source.is_file():
        raise FileNotFoundError(f"missing worker payload: {worker_source}")
    shutil.copy2(worker_source, executable)
    executable.chmod(0o755)
    submit_path = condor / "submit.sub"
    submit_path.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {executable}",
                (
                    f"arguments = {args.output} {args.high_base} {args.low_base} "
                    "$(topology) $(bundle)"
                ),
                f"output = {logs}/$(ClusterId).$(ProcId).out",
                f"error = {logs}/$(ClusterId).$(ProcId).err",
                f"log = {logs}/campaign.log",
                f"initialdir = {args.output}",
                "should_transfer_files = NO",
                "request_cpus = 1",
                "request_memory = 4500MB",
                f'+JobFlavour = "{args.job_flavour}"',
                "getenv = False",
                f"queue topology,bundle from {condor / 'queue.txt'}",
                "",
            ]
        )
    )
    submit_text = submit_path.read_text()
    (condor / "pilot.sub").write_text(
        submit_text.replace(str(queue_path), str(pilot_queue_path))
    )
    (condor / "remaining.sub").write_text(
        submit_text.replace(str(queue_path), str(remaining_queue_path))
    )
    payload = {
        "schema_version": "highdm79minus6_lowdm30_limit_campaign_v1",
        "status": "ready",
        "year": "2024",
        "schedd": "bigbird24",
        "high_source": str(args.high_base),
        "low_source": str(args.low_base),
        "output": str(args.output),
        "orthogonality": {
            "source_highdm_sr_bins": 79,
            "removed_highdm_channels": [
                f"SR_highdm_bin{index}" for index in range(6)
            ],
            "retained_highdm_sr_bins": 73,
            "lowdm_sr_categories": 6,
            "lowdm_shape_bins_per_category": 5,
            "lowdm_sr_bins": 30,
            "combined_sr_bins": 103,
            "reason": "remove overlapping Nt=Nw=Nres=0 High-dM recoil category",
        },
        "bundle_size": args.bundle_size,
        "job_flavour": args.job_flavour,
        "job_count": len(jobs),
        "point_count": all_points,
        "combined_point_count": all_combined,
        "high_only_point_count": all_points - all_combined,
        "topologies": topology_summary,
        "jobs": jobs,
    }
    write_json(args.output / "campaign_manifest.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "job_count": payload["job_count"],
                "point_count": payload["point_count"],
                "combined_point_count": payload["combined_point_count"],
                "high_only_point_count": payload["high_only_point_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
