from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

from .config.defaults import DEFAULTS
from .paths import PathKind, PathPolicy, canonicalize
from .workflow.manifest import load_benchmark_manifest
from .validation.reference import validate_scaled_reference


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def choose_output_format():
    if _has_module("pyarrow"):
        return "parquet", "PyArrow is available in the fixed EOS py38 environment."
    return "root", "PyArrow is unavailable in the fixed EOS py38 environment; using flat ROOT with uproot, which is already required."


def benchmark_manifest(manifest_path, dry_run=False, max_events=10000):
    policy = PathPolicy.default()
    manifest = load_benchmark_manifest(manifest_path)
    samples = manifest.get("samples", [])
    fmt, reason = choose_output_format()
    reference_gate = validate_scaled_reference(DEFAULTS.legacy_scaled_reference, "recoilpt", "cat2_LLCR_highDeltaM")
    result = {
        "status": "dry-run" if dry_run else "benchmark",
        "reference_gate": {
            "variable": "recoilpt",
            "region": "cat2_LLCR_highDeltaM",
            "global_classification": reference_gate.get("global_classification"),
            "region_scoped_classification": reference_gate.get("region_scoped_classification"),
            "ok": reference_gate.get("ok"),
            "message": "Legacy regression is currently validated only for cat2_LLCR_highDeltaM. Other regions, including cat1_preselection, are not yet accepted as valid references.",
        },
        "format_choice": fmt,
        "format_reason": reason,
        "implemented_logic": {
            "object_selection": False,
            "event_cleaning": False,
            "triggers": False,
            "recoil": "primitive MET-as-recoil only for benchmark unless role-specific region logic is implemented",
            "nominal_weights": False,
            "region_selections": False,
            "process_normalization": False,
            "signal_mass_extraction": "manifest metadata only",
        },
        "recommended_job_granularity": "target 5-15 minutes per chunk after real selection benchmark; first benchmark is limited to one file per role",
        "maturity": list(DEFAULTS.maturity),
        "samples": [],
    }
    if not reference_gate.get("ok"):
        result["status"] = "blocked"
        result["blocked_reason"] = "cat2_LLCR_highDeltaM/recoilpt/nominal legacy reference validation failed"
        return result
    if dry_run:
        for sample in samples:
            result["samples"].append({"role": sample.get("role"), "dataset": sample.get("dataset"), "files": sample.get("files", [])})
        return result
    try:
        import uproot
    except Exception as exc:
        result["status"] = "blocked"
        result["blocked_reason"] = "uproot is unavailable in the fixed environment: %s" % exc
        return result
    for sample in samples:
        file_results = []
        for input_path in sample.get("files", [])[:1]:
            if str(input_path).startswith(("root://", "xrootd://")):
                size = None
                resolved = input_path
            else:
                resolved_path = policy.resolve(input_path, PathKind.INPUT)
                size = os.path.getsize(resolved_path)
                resolved = str(resolved_path)
            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            events = None
            branches = []
            try:
                with uproot.open("%s:Events" % resolved) as tree:
                    events = tree.num_entries if max_events is None else min(tree.num_entries, max_events)
                    branches = list(tree.keys())[:40]
            except Exception as exc:
                file_results.append({"path": canonicalize(resolved), "size_bytes": size, "status": "failed", "error": str(exc)})
                continue
            wall = max(time.perf_counter() - started_wall, 1e-9)
            cpu = max(time.process_time() - started_cpu, 0.0)
            file_results.append({
                "path": canonicalize(resolved),
                "size_bytes": size,
                "events_inspected": events,
                "branches_preview": branches,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "events_per_second_metadata": events / wall if events is not None else None,
                "status": "metadata-only",
            })
        result["samples"].append({"role": sample.get("role"), "dataset": sample.get("dataset"), "files": file_results})
    return result


def benchmark_json(manifest_path, dry_run=False, max_events=10000):
    return json.dumps(benchmark_manifest(manifest_path, dry_run=dry_run, max_events=max_events), indent=2, sort_keys=True)
