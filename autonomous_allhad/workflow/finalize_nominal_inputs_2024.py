#!/usr/bin/env python3
"""Freeze a validated, non-overlapping input view for the 2024 nominal analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RESIDUAL_MC = {
    "mc_shard_01359",
    "mc_shard_01952",
    "mc_shard_02841",
    "mc_shard_05312",
    "mc_shard_07593",
    "mc_shard_08028",
    "mc_shard_10457",
    "mc_shard_18683",
}
MIXED_PROCESS_MC = {"mc_shard_01359", "mc_shard_08028", "mc_shard_10457"}
PARTIAL_MC = EXPECTED_RESIDUAL_MC - MIXED_PROCESS_MC


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty file: {path}")


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def validate_groups(directory: Path, expected_status: str) -> tuple[list[dict[str, Any]], set[str]]:
    manifest = read_json(directory / "manifest.json")
    plan = read_json(directory / "merge_plan.json")
    if manifest.get("status") != expected_status or manifest.get("failures"):
        raise RuntimeError(f"{directory}: incomplete manifest")
    groups: list[dict[str, Any]] = []
    covered: set[str] = set()
    for planned in plan.get("groups") or []:
        name = str(planned["group"])
        root = directory / f"{name}.root"
        sidecar = directory / f"{name}.json"
        require_file(root)
        require_file(sidecar)
        meta = read_json(sidecar)
        sources = [str(value) for value in meta.get("source_shards") or []]
        if meta.get("status") != "complete" or meta.get("bad_files"):
            raise RuntimeError(f"{sidecar}: invalid status or bad_files")
        if meta.get("source_fingerprint") != planned.get("source_fingerprint"):
            raise RuntimeError(f"{sidecar}: source fingerprint mismatch")
        if sorted(sources) != sorted(str(value) for value in planned.get("source_shards") or []):
            raise RuntimeError(f"{sidecar}: source shard mismatch")
        if covered.intersection(sources):
            raise RuntimeError(f"{sidecar}: duplicate source shard")
        covered.update(sources)
        groups.append(
            {
                "name": name,
                "root": str(root.absolute()),
                "sidecar": str(sidecar.absolute()),
                "source_count": len(sources),
                "source_fingerprint": meta.get("source_fingerprint"),
                "events_written": int(meta.get("events_written") or 0),
            }
        )
    if len(groups) != int(manifest.get("group_count_valid") or -1):
        raise RuntimeError(f"{directory}: group count mismatch")
    return groups, covered


def recovery_results(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    successful: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    for path in paths:
        manifest = read_json(path)
        if manifest.get("status") not in {"complete", "complete_with_failures"}:
            raise RuntimeError(f"{path}: recovery is not terminal")
        for result in manifest.get("results") or []:
            parent = str(result["parent_shard"])
            if result.get("status") == "complete":
                root = Path(str(result["root_output"]))
                sidecar = Path(str(result["json_output"]))
                require_file(root)
                require_file(sidecar)
                meta = read_json(sidecar)
                if (
                    meta.get("status") != "complete"
                    or int(meta.get("files_attempted") or 0) != 1
                    or int(meta.get("files_processed") or 0) != 1
                    or meta.get("bad_files")
                ):
                    raise RuntimeError(f"{sidecar}: invalid recovery output")
                successful[parent] = {
                    "root": str(root),
                    "sidecar": str(sidecar),
                    "events_written": int(meta.get("events_written") or 0),
                    "bad_file": result.get("bad_file"),
                }
            else:
                failed[parent] = {
                    "status": result.get("status"),
                    "bad_file": result.get("bad_file"),
                    "error": result.get("error") or result.get("stderr_tail"),
                }
    return successful, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-data", type=int, default=13_576)
    parser.add_argument("--expected-mc", type=int, default=19_586)
    parser.add_argument("--expected-signal", type=int, default=61)
    args = parser.parse_args()

    campaign = args.campaign.absolute()
    output_dir = args.output_dir.absolute()
    data_dir = campaign / "merged/data_balanced20"
    mc_dirs = [
        campaign / "merged/mc_balanced20_snapshot_20260725",
        campaign / "merged/mc_balanced20_postsnapshot_20260725",
    ]

    data_groups, data_covered = validate_groups(data_dir, "complete_with_missing_inputs")
    data_manifest = read_json(data_dir / "manifest.json")
    data_missing = sorted(str(value) for value in data_manifest.get("missing_or_invalid_shards") or [])
    expected_data = {f"data_shard_{idx:05d}" for idx in range(args.expected_data)}
    if data_covered | set(data_missing) != expected_data or data_covered & set(data_missing):
        raise RuntimeError("data source coverage is not an exact partition")

    mc_groups: list[dict[str, Any]] = []
    mc_covered: set[str] = set()
    for directory in mc_dirs:
        groups, covered = validate_groups(directory, "snapshot_complete_with_pending_inputs")
        if mc_covered & covered:
            raise RuntimeError(f"{directory}: overlaps a prior MC snapshot")
        mc_groups.extend(groups)
        mc_covered.update(covered)

    expected_mc = {f"mc_shard_{idx:05d}" for idx in range(args.expected_mc)}
    residual_mc = expected_mc - mc_covered
    if residual_mc != EXPECTED_RESIDUAL_MC:
        raise RuntimeError(f"unexpected residual MC shards: {sorted(residual_mc)}")

    residual_inputs: list[dict[str, Any]] = []
    nominal_dir = campaign / "outputs/nominal"
    for shard in sorted(residual_mc):
        root = nominal_dir / f"{shard}.root"
        sidecar = nominal_dir / f"{shard}.json"
        require_file(root)
        require_file(sidecar)
        meta = read_json(sidecar)
        expected_status = "complete" if shard in MIXED_PROCESS_MC else "complete_with_bad_files"
        if meta.get("status") != expected_status:
            raise RuntimeError(f"{sidecar}: status={meta.get('status')}, expected={expected_status}")
        if shard in PARTIAL_MC and len(meta.get("bad_files") or []) != 1:
            raise RuntimeError(f"{sidecar}: expected exactly one bad file")
        residual_inputs.append(
            {
                "shard": shard,
                "root": str(root),
                "sidecar": str(sidecar),
                "status": expected_status,
                "events_written": int(meta.get("events_written") or 0),
                "datasets": sorted(
                    {str(rec.get("process")) for rec in (meta.get("datasets") or {}).values()}
                ),
            }
        )

    recovery_manifests = [
        campaign / "recovery/badfile_local_actual_once_20260725/attempt_manifest.json",
        campaign / "recovery/badfile_18683_local_actual_once_20260725/attempt_manifest.json",
    ]
    recovered, failed_recoveries = recovery_results(recovery_manifests)
    expected_recovered = PARTIAL_MC - {"mc_shard_01952"}
    if set(recovered) != expected_recovered or set(failed_recoveries) != {"mc_shard_01952"}:
        raise RuntimeError("unexpected bad-file recovery partition")

    signal_report_path = campaign / "reports/signal_validation_20260725.json"
    signal_report = read_json(signal_report_path)
    if (
        signal_report.get("status") != "complete"
        or int(signal_report.get("ok") or 0) != args.expected_signal
        or int(signal_report.get("bad") or 0) != 0
    ):
        raise RuntimeError("signal validation report is incomplete")
    signal_inputs: list[dict[str, Any]] = []
    for idx in range(args.expected_signal):
        name = f"signal_shard_{idx:05d}"
        root = nominal_dir / f"{name}.root"
        sidecar = nominal_dir / f"{name}.json"
        require_file(root)
        require_file(sidecar)
        meta = read_json(sidecar)
        if meta.get("status") != "complete" or meta.get("bad_files"):
            raise RuntimeError(f"{sidecar}: invalid signal output")
        signal_inputs.append(
            {
                "shard": name,
                "root": str(root),
                "sidecar": str(sidecar),
                "events_written": int(meta.get("events_written") or 0),
            }
        )

    campaign_manifest = read_json(campaign / "manifest.json")
    btag = campaign_manifest.get("btag_efficiency") or {}
    if btag.get("status") != "valid" or (btag.get("coverage") or {}).get("status") != "complete":
        raise RuntimeError("campaign btag efficiency audit is not valid")

    roots = (
        [record["root"] for record in data_groups]
        + [record["root"] for record in mc_groups]
        + [record["root"] for record in residual_inputs]
        + [record["root"] for _, record in sorted(recovered.items())]
        + [record["root"] for record in signal_inputs]
    )
    sidecars = (
        [record["sidecar"] for record in data_groups]
        + [record["sidecar"] for record in mc_groups]
        + [record["sidecar"] for record in residual_inputs]
        + [record["sidecar"] for _, record in sorted(recovered.items())]
        + [record["sidecar"] for record in signal_inputs]
    )
    if len(roots) != len(set(roots)) or len(sidecars) != len(set(sidecars)):
        raise RuntimeError("duplicate final input paths")
    for path in map(Path, roots + sidecars):
        require_file(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    roots_path = output_dir / "nominal_input_roots.txt"
    sidecars_path = output_dir / "nominal_input_sidecars.txt"
    roots_path.write_text("\n".join(roots) + "\n")
    sidecars_path.write_text("\n".join(sidecars) + "\n")
    manifest = {
        "schema_version": "final_nominal_input_coverage_2024_v1",
        "status": "complete_with_known_missing_inputs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "coverage": {
            "data": {
                "expected_shards": args.expected_data,
                "represented_shards": len(data_covered),
                "missing_shards": data_missing,
                "merged_groups": len(data_groups),
            },
            "background_mc": {
                "expected_shards": args.expected_mc,
                "represented_shards": len(mc_covered) + len(residual_mc),
                "merged_groups": len(mc_groups),
                "residual_parent_shards": sorted(residual_mc),
                "successful_bad_file_recoveries": sorted(recovered),
                "irrecoverable_bad_files": failed_recoveries,
            },
            "signal": {
                "expected_shards": args.expected_signal,
                "validated_shards": len(signal_inputs),
                "validation_report": str(signal_report_path),
            },
        },
        "input_counts": {
            "roots": len(roots),
            "sidecars": len(sidecars),
            "data_merged": len(data_groups),
            "mc_merged": len(mc_groups),
            "mc_residual_parents": len(residual_inputs),
            "mc_recoveries": len(recovered),
            "signal": len(signal_inputs),
        },
        "input_lists": {
            "roots": str(roots_path),
            "roots_sha256": sha256_lines(roots),
            "sidecars": str(sidecars_path),
            "sidecars_sha256": sha256_lines(sidecars),
        },
        "btag_efficiency": {
            "status": btag.get("status"),
            "coverage_status": (btag.get("coverage") or {}).get("status"),
            "sha256": btag.get("sha256"),
        },
        "normalization_policy": {
            "scope": "aggregate the listed merged, residual-parent, recovery, and signal sidecars exactly once",
            "known_loss": "data shards 04737/06758 absent; one GJ file in mc_shard_01952 reproducibly corrupt",
        },
        "data_groups": data_groups,
        "mc_groups": mc_groups,
        "mc_residual_inputs": residual_inputs,
        "mc_recovery_inputs": recovered,
        "signal_inputs": signal_inputs,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "roots": len(roots),
                "data_groups": len(data_groups),
                "mc_groups": len(mc_groups),
                "mc_residual_parents": len(residual_inputs),
                "mc_recoveries": len(recovered),
                "signal": len(signal_inputs),
                "output": str(output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
