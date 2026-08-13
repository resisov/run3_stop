#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EXECUTION_CONTRACT_PATHS = (
    "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
    "autonomous_allhad/workflow/run_flat_hists_chunked.py",
    "autonomous_allhad/autonomous_allhad/analysis_scale_factors.py",
    "autonomous_allhad/autonomous_allhad/real_subset_worker.py",
    "autonomous_allhad/autonomous_allhad/dy_ptll_policy.py",
    "analysis/utils/corrections.py",
    "analysis/data/corrections.coffea",
    "analysis/data/PUweight/2024/puWeights.json.gz",
    "analysis/data/BTVSF/2024/btagging.json.gz",
    "analysis/data/EGammaSF/2024/electron.json.gz",
    "analysis/data/EGammaSF/2024/electronHlt.json.gz",
    "analysis/data/EGammaSF/2024/photon.json.gz",
    "analysis/data/MuonSF/2024/muon_Z.json.gz",
    "analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz",
    "analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz",
    "analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz",
    "analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz",
)
REQUIRED_ANALYSIS_SF_COMPONENTS = [
    "met_trigger",
    "photon_trigger",
    "veto_electron_5to10",
    "loose_muon_5to10",
]
EXPECTED_BTAG_EFFICIENCY_SHA256_2024 = (
    "03524e9ae28110814f336eafc887e60d54b495a7b8dec7cda59bd792f56feaf4"
)
BTAG_EFFICIENCY_RELATIVE_PATH = "analysis/hists/btageff2024.merged"


def execution_code_sha256(repo: Path) -> dict[str, str]:
    return {
        relative_path: file_sha256(repo / relative_path)
        for relative_path in EXECUTION_CONTRACT_PATHS
    }


def btag_efficiency_contract(
    repo: Path,
    expected_sha256: str,
    required: bool,
) -> dict[str, Any]:
    path = repo / BTAG_EFFICIENCY_RELATIVE_PATH
    if not path.exists():
        if required:
            raise RuntimeError(f"required b-tag efficiency payload is missing: {path}")
        return {
            "path": BTAG_EFFICIENCY_RELATIVE_PATH,
            "exists": False,
            "expected_sha256": expected_sha256,
        }
    actual_sha256 = file_sha256(path)
    matches = not expected_sha256 or actual_sha256 == expected_sha256
    if required and not matches:
        raise RuntimeError(
            f"b-tag efficiency SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return {
        "path": BTAG_EFFICIENCY_RELATIVE_PATH,
        "exists": True,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "matches_expected": matches,
    }


def hist_leaf(obj: Any) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in ("sumw", "sumw2", "entries"))


def finite_sum(left: float, right: float) -> float:
    value = float(left) + float(right)
    if value == float("inf"):
        return 1.0e300
    if value == float("-inf"):
        return -1.0e300
    if value != value:
        return 0.0
    return value


def merge_hist(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sumw", "sumw2", "entries"):
        left = target.setdefault(key, [0] * len(source.get(key, [])))
        right = source.get(key, [])
        if len(left) < len(right):
            left.extend([0] * (len(right) - len(left)))
        for idx, value in enumerate(right):
            if key == "entries":
                left[idx] += value
            else:
                left[idx] = finite_sum(left[idx], value)


def merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    if hist_leaf(source):
        merge_hist(target, source)
        return
    for key, value in source.items():
        if hist_leaf(value):
            merge_hist(target.setdefault(key, {}), value)
        elif isinstance(value, dict):
            merge_tree(target.setdefault(key, {}), value)
        else:
            target[key] = value


def merge_status(target: dict[str, Any], source: dict[str, Any]) -> None:
    for label, status in source.items():
        current = target.get(label)
        if current is None:
            target[label] = status
            continue
        current_components = current.get("components") or {}
        source_components = status.get("components") or {}
        for name, item in source_components.items():
            if name not in current_components or item.get("applied"):
                current_components[name] = item
        current["components"] = current_components
        variations = sorted(set(current.get("available_variations") or []) | set(status.get("available_variations") or []))
        if variations:
            current["available_variations"] = variations
        current["applied"] = bool(current.get("applied")) or bool(status.get("applied"))


def merge_exclusions(target: dict[str, Any], source: dict[str, Any]) -> None:
    for region, by_process in source.items():
        out_region = target.setdefault(region, {})
        for process, rec in (by_process or {}).items():
            out = out_region.setdefault(process, {"entries": 0})
            out["entries"] = int(out.get("entries", 0)) + int((rec or {}).get("entries", 0))


def merge_dy_ptll_exclusions(target: dict[str, Any], source: dict[str, Any]) -> None:
    for dataset, record in source.items():
        out = target.setdefault(
            dataset,
            {
                "dataset_id": int((record or {}).get("dataset_id", -1)),
                "entries": 0,
                "policy": (record or {}).get("policy"),
            },
        )
        out["entries"] = int(out.get("entries", 0)) + int((record or {}).get("entries", 0))


def merge_numeric_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value or 0)


def merge_nested_numeric_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(
                    f"cannot merge nested counter {key!r} into scalar {child!r}"
                )
            merge_nested_numeric_counts(child, value)
        else:
            if isinstance(target.get(key), dict):
                raise TypeError(
                    f"cannot merge scalar counter {key!r} into nested mapping"
                )
            target[key] = int(target.get(key, 0)) + int(value or 0)


STRICT_WARNING_KEYS = (
    "weight_failures",
    "missing_input_roots",
    "missing_sidecars",
    "zero_entry_roots",
    "weight_rejections",
)


def summary_has_strict_warnings(summary: dict[str, Any]) -> bool:
    return any(bool(summary.get(key)) for key in STRICT_WARNING_KEYS)


REPAIRABLE_CODE_PATHS = (
    "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
    "autonomous_allhad/workflow/run_flat_hists_chunked.py",
)


def compatible_build_options(
    recorded: dict[str, Any] | None,
    expected: dict[str, Any] | None,
    allow_hist_builder_repair: bool = False,
) -> bool:
    if recorded == expected:
        return True
    if not allow_hist_builder_repair or recorded is None or expected is None:
        return False
    recorded_copy = json.loads(json.dumps(recorded))
    expected_copy = json.loads(json.dumps(expected))
    recorded_copy.setdefault("gcr_only", False)
    recorded_copy.setdefault("gcr_photon_policy", "nominal")
    expected_copy.setdefault("gcr_only", False)
    expected_copy.setdefault("gcr_photon_policy", "nominal")
    recorded_code = recorded_copy.get("code_sha256") or {}
    expected_code = expected_copy.get("code_sha256") or {}
    for code_path in REPAIRABLE_CODE_PATHS:
        recorded_code.pop(code_path, None)
        expected_code.pop(code_path, None)
    return recorded_copy == expected_copy


def merge_payloads(
    chunks: list[Path],
    output: Path,
    normalization: Path,
    dy_ptll_policy: str = "all",
    expected_build_options: dict[str, Any] | None = None,
    allow_hist_builder_repair: bool = False,
) -> dict[str, Any]:
    merged: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "events_processed": 0,
        "input_roots": [],
        "chunk_outputs": [str(p) for p in chunks],
        "scale_factor_status": {},
        "dy_ptll_policy": dy_ptll_policy,
    }
    for path in chunks:
        payload = read_json(path)
        if expected_build_options is not None:
            recorded_normalization = payload.get("normalization")
            if not recorded_normalization:
                raise RuntimeError(f"{path}: chunk normalization provenance is missing")
            if Path(recorded_normalization).resolve() != normalization.resolve():
                raise RuntimeError(
                    f"{path}: chunk normalization {recorded_normalization!r} does not "
                    f"match requested normalization {str(normalization)!r}"
                )
        if merged is None:
            merged = {
                key: value
                for key, value in payload.items()
                if key not in {"histograms", "search_bin_histograms", "lowdm_variable_histograms", "highdm_variable_histograms", "summary", "status", "normalization"}
            }
            merged["histograms"] = {}
            merged["search_bin_histograms"] = {}
            merged["lowdm_variable_histograms"] = {}
            merged["highdm_variable_histograms"] = {}
        merge_tree(merged["histograms"], payload.get("histograms") or {})
        merge_tree(merged["search_bin_histograms"], payload.get("search_bin_histograms") or {})
        merge_tree(merged["lowdm_variable_histograms"], payload.get("lowdm_variable_histograms") or {})
        merge_tree(merged["highdm_variable_histograms"], payload.get("highdm_variable_histograms") or {})
        src_summary = payload.get("summary") or {}
        chunk_policy = str(src_summary.get("dy_ptll_policy", "all"))
        if chunk_policy != dy_ptll_policy:
            raise RuntimeError(
                f"{path}: DY policy {chunk_policy!r} does not match "
                f"requested merge policy {dy_ptll_policy!r}"
            )
        chunk_build_options = src_summary.get("build_options")
        if expected_build_options is not None and not compatible_build_options(
            chunk_build_options,
            expected_build_options,
            allow_hist_builder_repair,
        ):
            raise RuntimeError(
                f"{path}: chunk build options do not match the requested merge contract"
            )
        if chunk_build_options is not None:
            recorded_build_options = summary.setdefault("build_options", expected_build_options or chunk_build_options)
            if not compatible_build_options(
                chunk_build_options,
                recorded_build_options,
                allow_hist_builder_repair,
            ):
                raise RuntimeError(
                    f"{path}: chunk build options do not match the other chunks"
                )
            for code_path in REPAIRABLE_CODE_PATHS:
                code_sha = (chunk_build_options.get("code_sha256") or {}).get(code_path)
                if code_sha:
                    variants = summary.setdefault("repair_code_sha256_variants", {}).setdefault(
                        code_path,
                        [],
                    )
                    if code_sha not in variants:
                        variants.append(code_sha)
        chunk_status = str(payload.get("status") or "missing")
        status_counts = summary.setdefault("chunk_statuses", {})
        status_counts[chunk_status] = int(status_counts.get(chunk_status, 0)) + 1
        if src_summary.get("region_filter"):
            summary["region_filter"] = src_summary["region_filter"]
        if src_summary.get("variable_filter"):
            summary["variable_filter"] = src_summary["variable_filter"]
        summary["events_processed"] += int(src_summary.get("events_processed") or 0)
        summary["input_roots"].extend(src_summary.get("input_roots") or [])
        for key in (
            "weight_failures",
            "missing_input_roots",
            "missing_sidecars",
            "zero_entry_roots",
        ):
            if src_summary.get(key):
                summary.setdefault(key, []).extend(src_summary.get(key) or [])
        merge_status(summary["scale_factor_status"], src_summary.get("scale_factor_status") or {})
        merge_exclusions(summary.setdefault("data_stream_exclusions", {}), src_summary.get("data_stream_exclusions") or {})
        merge_dy_ptll_exclusions(
            summary.setdefault("dy_ptll_dataset_exclusions", {}),
            src_summary.get("dy_ptll_dataset_exclusions") or {},
        )
        merge_numeric_counts(
            summary.setdefault("dy_ptll_prefilter", {}),
            src_summary.get("dy_ptll_prefilter") or {},
        )
        for key in (
            "input_sidecar_schema_versions",
            "electron_eta_sources",
            "weight_rejections",
            "histogram_range_exclusions",
            "histogram_folded_flow",
            "lowdm_search_bin_entry_accounting",
            "scale_factor_status_audit",
            "gcr_prefilter",
            "gcr_photon_selection_audit",
        ):
            merge_nested_numeric_counts(
                summary.setdefault(key, {}),
                src_summary.get(key) or {},
            )
    if merged is None:
        raise RuntimeError("no chunk payloads to merge")
    duplicate_merged_roots = sorted(
        root
        for root, count in Counter(summary["input_roots"]).items()
        if count > 1
    )
    if duplicate_merged_roots:
        raise RuntimeError(
            "duplicate input ROOTs across chunk payloads: "
            + ", ".join(duplicate_merged_roots)
        )
    summary["input_roots"] = sorted(summary["input_roots"])
    merged["normalization"] = str(normalization)
    merged["summary"] = summary
    all_chunks_clean = set(summary.get("chunk_statuses") or {}) <= {"complete"}
    merged["status"] = (
        "complete"
        if all_chunks_clean and not summary_has_strict_warnings(summary)
        else "complete_with_warnings"
    )
    write_json(output, merged)
    return merged


def split_roots(paths: list[str], jobs: int) -> list[list[str]]:
    chunks = [[] for _ in range(jobs)]
    for idx, path in enumerate(paths):
        chunks[idx % jobs].append(path)
    return [chunk for chunk in chunks if chunk]


def split_roots_by_size(paths: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [paths[start:start + size] for start in range(0, len(paths), size)]


def completed_chunk_matches(
    path: Path,
    expected_roots: list[str],
    expected_dy_ptll_policy: str = "all",
    require_clean_status: bool = False,
    expected_build_options: dict[str, Any] | None = None,
    expected_normalization: Path | None = None,
    allow_hist_builder_repair: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    recorded_status = str(payload.get("status") or "")
    if require_clean_status:
        if recorded_status != "complete":
            return False
    elif not recorded_status.startswith("complete"):
        return False
    summary = payload.get("summary") or {}
    if require_clean_status and summary_has_strict_warnings(summary):
        return False
    if summary.get("dy_ptll_policy", "all") != expected_dy_ptll_policy:
        return False
    if expected_build_options is not None and not compatible_build_options(
        summary.get("build_options"),
        expected_build_options,
        allow_hist_builder_repair,
    ):
        return False
    if expected_normalization is not None:
        recorded_normalization = payload.get("normalization")
        if not recorded_normalization:
            return False
        if Path(recorded_normalization).resolve() != expected_normalization.resolve():
            return False
    recorded = list(summary.get("input_roots") or [])
    return len(recorded) == len(expected_roots) and set(recorded) == set(expected_roots)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run flat boosted histogram builder in local chunks and merge outputs.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--jobs", type=int, default=8, help="Number of chunks for the legacy splitter, and default max parallelism.")
    parser.add_argument("--chunk-size", type=int, default=0, help="If positive, split input ROOT files into chunks of this many files.")
    parser.add_argument("--max-parallel", type=int, default=0, help="Maximum number of chunk builders to run at once. Defaults to --jobs.")
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=1,
        help="Use the existing streaming merger with this many worker processes.",
    )
    parser.add_argument("--step-size", type=int, default=50000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--local-analysis-data", choices=["0", "1"], default="0")
    parser.add_argument("--only-regions", nargs="+", choices=["GCR", "HighDMVR_Nb1", "HighDMVR_Nb2", "HighDMVR_Nb3plus"])
    parser.add_argument(
        "--only-variables",
        nargs="+",
        choices=[
            "nb",
            "njet",
            "nfatjet",
            "ntop",
            "nw",
            "ht",
            "ut",
            "ptll",
            "met",
            "jet_pt",
            "fatjet_pt",
            "bjet_pt",
        ],
    )
    parser.add_argument("--require-btag", action="store_true")
    parser.add_argument(
        "--expected-btag-efficiency-sha256",
        default=EXPECTED_BTAG_EFFICIENCY_SHA256_2024,
        help="Expected SHA256 for analysis/hists/btageff2024.merged.",
    )
    parser.add_argument(
        "--require-weight-components",
        nargs="+",
        default=REQUIRED_ANALYSIS_SF_COMPONENTS,
        help="Weight components that every non-data dataset must apply.",
    )
    parser.add_argument(
        "--analysis-sf-components",
        nargs="+",
        choices=REQUIRED_ANALYSIS_SF_COMPONENTS,
        default=REQUIRED_ANALYSIS_SF_COMPONENTS,
        help="Analysis-owned SF components included in nominal and Up/Down weights.",
    )
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help=(
            "Do not resume warning-bearing chunks and require the merged payload "
            "to have exact status 'complete'."
        ),
    )
    parser.add_argument(
        "--require-branches",
        action="store_true",
        help="Require every non-forward-optional branch in the histogram schema.",
    )
    parser.add_argument(
        "--require-normalization",
        action="store_true",
        help="Require every MC event group to have a finite, positive normalization factor.",
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help=(
            "Compute and validate the full weight bundle, but fill histogram "
            "payloads with nominal weights only."
        ),
    )
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--only-signal-mass", nargs=2, type=int, metavar=("MSTOP", "MLSP"))
    parser.add_argument("--only-lowdm-sr-nsv-inclusive", action="store_true")
    parser.add_argument("--only-lowdm-nsv-repair", action="store_true")
    parser.add_argument("--gcr-only", action="store_true")
    parser.add_argument(
        "--gcr-photon-policy",
        choices=("nominal", "tight_eb"),
        default="nominal",
    )
    parser.add_argument(
        "--dy-ptll-policy",
        choices=("all", "ptll100_only", "ptll200_only", "ptll100_200"),
        default="all",
        help="DY pT(ll) sample-family policy passed to every histogram-builder chunk.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-hist-builder-repair",
        action="store_true",
        help=(
            "Reuse clean chunks when the only build-contract difference is the "
            "histogram-builder hash, and record every builder hash in the merge. "
            "Use only for a scoped repair whose affected chunks are rerun."
        ),
    )
    args = parser.parse_args()
    if args.only_lowdm_nsv_repair and not args.only_lowdm_sr_nsv_inclusive:
        parser.error("--only-lowdm-nsv-repair requires --only-lowdm-sr-nsv-inclusive")
    if args.gcr_only and args.only_regions:
        parser.error("--gcr-only cannot be combined with --only-regions")
    if args.gcr_photon_policy != "nominal" and not args.gcr_only:
        parser.error("--gcr-photon-policy requires --gcr-only")
    unavailable_required = (
        set(args.require_weight_components)
        & set(REQUIRED_ANALYSIS_SF_COMPONENTS)
    ) - set(args.analysis_sf_components)
    if unavailable_required:
        parser.error(
            "--require-weight-components contains disabled analysis SFs: "
            + ", ".join(sorted(unavailable_required))
        )

    repo = Path(args.repo).resolve()
    input_list = Path(args.input_list).resolve()
    normalization = Path(args.normalization).resolve()
    output_path = Path(args.output).resolve()
    roots = [
        str(Path(line.strip()).resolve())
        for line in input_list.read_text().splitlines()
        if line.strip()
    ]
    if not roots:
        raise SystemExit("empty input list")
    duplicate_roots = sorted(
        root
        for root, count in Counter(roots).items()
        if count > 1
    )
    if duplicate_roots:
        raise SystemExit(
            "duplicate input ROOTs are not allowed: "
            + ", ".join(duplicate_roots)
        )
    work_dir = Path(args.work_dir).resolve()
    chunk_dir = work_dir / "hist_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    script = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    env = os.environ.copy()
    env["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = args.local_analysis_data
    env["PYTHONPATH"] = str(repo / "autonomous_allhad") + os.pathsep + env.get("PYTHONPATH", "")
    chunks = split_roots_by_size(roots, args.chunk_size) if args.chunk_size > 0 else split_roots(roots, max(1, args.jobs))
    max_parallel = max(1, args.max_parallel or args.jobs)
    btag_payload_required = bool(
        args.require_btag or "btagSF" in args.require_weight_components
    )
    expected_build_options = {
        "step_size": int(args.step_size),
        "only_regions": list(args.only_regions) if args.only_regions else None,
        "only_variables": list(args.only_variables) if args.only_variables else None,
        "require_btag": bool(args.require_btag),
        "require_weight_components": list(args.require_weight_components),
        "analysis_sf_components": list(args.analysis_sf_components),
        "require_branches": bool(args.require_branches),
        "require_normalization": bool(args.require_normalization),
        "nominal_only": bool(args.nominal_only),
        "distribution_only": bool(args.distribution_only),
        "only_signal_mass": list(args.only_signal_mass) if args.only_signal_mass else None,
        "only_lowdm_sr_nsv_inclusive": bool(args.only_lowdm_sr_nsv_inclusive),
        "only_lowdm_nsv_repair": bool(args.only_lowdm_nsv_repair),
        "dy_ptll_policy": str(args.dy_ptll_policy),
        "gcr_only": bool(args.gcr_only),
        "gcr_photon_policy": str(args.gcr_photon_policy),
        "local_analysis_data": str(args.local_analysis_data),
        "normalization_sha256": file_sha256(normalization),
        "code_sha256": execution_code_sha256(repo),
        "btag_efficiency": btag_efficiency_contract(
            repo,
            str(args.expected_btag_efficiency_sha256),
            btag_payload_required,
        ),
    }
    print(json.dumps({"stage": "chunked_hists_start", "roots": len(roots), "chunks": len(chunks), "chunk_size": args.chunk_size or None, "max_parallel": max_parallel, "dy_ptll_policy": args.dy_ptll_policy, "work_dir": str(work_dir)}, sort_keys=True), flush=True)

    def launch(idx: int) -> tuple[int, Path, Path, subprocess.Popen[Any], Any]:
        chunk = chunks[idx]
        output = chunk_dir / f"chunk_{idx:03d}.json"
        log = chunk_dir / f"chunk_{idx:03d}.log"
        handle = log.open("w")
        cmd = [
            args.python,
            str(script),
            "--repo",
            str(repo),
            "--inputs",
            *chunk,
            "--normalization",
            str(normalization),
            "--output",
            str(output),
            "--step-size",
            str(args.step_size),
        ]
        if args.only_regions:
            cmd.extend(["--only-regions", *args.only_regions])
        if args.only_variables:
            cmd.extend(["--only-variables", *args.only_variables])
        if args.distribution_only:
            cmd.append("--distribution-only")
        if args.require_btag:
            cmd.append("--require-btag")
        cmd.extend(
            [
                "--expected-btag-efficiency-sha256",
                str(args.expected_btag_efficiency_sha256),
            ]
        )
        if args.require_weight_components:
            cmd.extend(["--require-weight-components", *args.require_weight_components])
        if args.analysis_sf_components:
            cmd.extend(["--analysis-sf-components", *args.analysis_sf_components])
        if args.require_branches:
            cmd.append("--require-branches")
        if args.require_normalization:
            cmd.append("--require-normalization")
        if args.nominal_only:
            cmd.append("--nominal-only")
        if args.only_signal_mass:
            cmd.extend(["--only-signal-mass", *(str(value) for value in args.only_signal_mass)])
        if args.only_lowdm_sr_nsv_inclusive:
            cmd.append("--only-lowdm-sr-nsv-inclusive")
        if args.only_lowdm_nsv_repair:
            cmd.append("--only-lowdm-nsv-repair")
        if args.gcr_only:
            cmd.append("--gcr-only")
        cmd.extend(["--gcr-photon-policy", args.gcr_photon_policy])
        cmd.extend(["--dy-ptll-policy", args.dy_ptll_policy])
        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, env=env)
        print(json.dumps({"stage": "chunk_started", "chunk": idx, "roots": len(chunk), "output": str(output), "log": str(log)}, sort_keys=True), flush=True)
        return idx, output, log, proc, handle

    ok = True
    finished: list[Path] = []
    todo: list[int] = []
    for idx, chunk in enumerate(chunks):
        output = chunk_dir / f"chunk_{idx:03d}.json"
        if args.resume and completed_chunk_matches(
            output,
            chunk,
            args.dy_ptll_policy,
            args.strict_complete,
            expected_build_options,
            normalization,
            args.allow_hist_builder_repair,
        ):
            finished.append(output)
        else:
            todo.append(idx)
    print(json.dumps({"stage": "chunk_resume_scan", "reused": len(finished), "remaining": len(todo)}, sort_keys=True), flush=True)
    pending: list[tuple[int, Path, Path, subprocess.Popen[Any], Any]] = []
    next_todo = 0
    while pending or next_todo < len(todo):
        while next_todo < len(todo) and len(pending) < max_parallel:
            pending.append(launch(todo[next_todo]))
            next_todo += 1
        still: list[tuple[int, Path, Path, subprocess.Popen[Any], Any]] = []
        for idx, output, log, proc, handle in pending:
            rc = proc.poll()
            if rc is None:
                still.append((idx, output, log, proc, handle))
                continue
            handle.close()
            print(json.dumps({"stage": "chunk_finished", "chunk": idx, "returncode": rc, "roots": len(chunks[idx]), "output": str(output), "log": str(log)}, sort_keys=True), flush=True)
            if rc == 0 and output.exists():
                finished.append(output)
            else:
                ok = False
        pending = still
        if pending or next_todo < len(todo):
            time.sleep(5)

    if not ok:
        return 2
    if args.merge_workers > 1:
        merge_command = [
            args.python,
            str(repo / "autonomous_allhad/workflow/merge_flat_hist_chunks_streaming.py"),
            "--chunk-dir",
            str(chunk_dir),
            "--normalization",
            str(normalization),
            "--output",
            str(output_path),
            "--results",
            str(work_dir / "chunked_hist_results.json"),
            "--dy-ptll-policy",
            args.dy_ptll_policy,
            "--expected-chunks",
            str(len(chunks)),
            "--workers",
            str(args.merge_workers),
            "--work-dir",
            str(work_dir / "merge"),
        ]
        if args.allow_hist_builder_repair:
            merge_command.append("--allow-hist-builder-repair")
        print(
            json.dumps(
                {
                    "stage": "parallel_streaming_merge_started",
                    "workers": args.merge_workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return subprocess.run(merge_command, cwd=str(repo), env=env).returncode
    merged = merge_payloads(
        sorted(finished),
        output_path,
        normalization,
        args.dy_ptll_policy,
        expected_build_options,
        args.allow_hist_builder_repair,
    )
    if args.strict_complete and merged["status"] != "complete":
        print(
            json.dumps(
                {
                    "stage": "chunked_hists_strict_validation_failed",
                    "status": merged["status"],
                    "weight_failures": len(merged["summary"].get("weight_failures") or []),
                    "missing_input_roots": len(
                        merged["summary"].get("missing_input_roots") or []
                    ),
                    "missing_sidecars": len(merged["summary"].get("missing_sidecars") or []),
                    "zero_entry_roots": len(
                        merged["summary"].get("zero_entry_roots") or []
                    ),
                    "weight_rejection_groups": len(
                        merged["summary"].get("weight_rejections") or {}
                    ),
                    "output": str(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 3
    write_json(
        work_dir / "chunked_hist_results.json",
        {
            "status": merged["status"],
            "dy_ptll_policy": args.dy_ptll_policy,
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "chunks": [str(p) for p in sorted(finished)],
        },
    )
    print(json.dumps({"stage": "chunked_hists_done", "status": merged["status"], "events_processed": merged["summary"]["events_processed"], "input_roots": len(merged["summary"]["input_roots"]), "output": str(output_path)}, sort_keys=True), flush=True)
    return 0 if merged["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
