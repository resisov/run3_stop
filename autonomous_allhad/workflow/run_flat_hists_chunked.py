#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def merge_payloads(chunks: list[Path], output: Path, normalization: Path) -> dict[str, Any]:
    merged: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "events_processed": 0,
        "input_roots": [],
        "chunk_outputs": [str(p) for p in chunks],
        "scale_factor_status": {},
    }
    for path in chunks:
        payload = read_json(path)
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
        if src_summary.get("region_filter"):
            summary["region_filter"] = src_summary["region_filter"]
        summary["events_processed"] += int(src_summary.get("events_processed") or 0)
        summary["input_roots"].extend(src_summary.get("input_roots") or [])
        for key in ("weight_failures", "missing_sidecars", "zero_entry_roots"):
            if src_summary.get(key):
                summary.setdefault(key, []).extend(src_summary.get(key) or [])
        merge_status(summary["scale_factor_status"], src_summary.get("scale_factor_status") or {})
        merge_exclusions(summary.setdefault("data_stream_exclusions", {}), src_summary.get("data_stream_exclusions") or {})
    if merged is None:
        raise RuntimeError("no chunk payloads to merge")
    summary["input_roots"] = sorted(dict.fromkeys(summary["input_roots"]))
    merged["normalization"] = str(normalization)
    merged["summary"] = summary
    merged["status"] = "complete" if not summary.get("weight_failures") and not summary.get("missing_sidecars") else "complete_with_warnings"
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


def completed_chunk_matches(path: Path, expected_roots: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if not str(payload.get("status") or "").startswith("complete"):
        return False
    recorded = list((payload.get("summary") or {}).get("input_roots") or [])
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
    parser.add_argument("--step-size", type=int, default=50000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--local-analysis-data", choices=["0", "1"], default="0")
    parser.add_argument("--only-regions", nargs="+", choices=["HighDMVR_Nb1", "HighDMVR_Nb2", "HighDMVR_Nb3plus"])
    parser.add_argument("--only-variables", nargs="+", choices=["nb", "njet", "nfatjet", "ntop", "nw", "ht", "ut", "met", "jet_pt", "fatjet_pt", "bjet_pt"])
    parser.add_argument("--require-btag", action="store_true")
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--only-signal-mass", nargs=2, type=int, metavar=("MSTOP", "MLSP"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    roots = [line.strip() for line in Path(args.input_list).read_text().splitlines() if line.strip()]
    if not roots:
        raise SystemExit("empty input list")
    work_dir = Path(args.work_dir)
    chunk_dir = work_dir / "hist_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    script = repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"
    env = os.environ.copy()
    env["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = args.local_analysis_data
    env["PYTHONPATH"] = str(repo / "autonomous_allhad") + os.pathsep + env.get("PYTHONPATH", "")
    chunks = split_roots_by_size(roots, args.chunk_size) if args.chunk_size > 0 else split_roots(roots, max(1, args.jobs))
    max_parallel = max(1, args.max_parallel or args.jobs)
    print(json.dumps({"stage": "chunked_hists_start", "roots": len(roots), "chunks": len(chunks), "chunk_size": args.chunk_size or None, "max_parallel": max_parallel, "work_dir": str(work_dir)}, sort_keys=True), flush=True)

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
            str(Path(args.normalization).resolve()),
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
        if args.only_signal_mass:
            cmd.extend(["--only-signal-mass", *(str(value) for value in args.only_signal_mass)])
        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, env=env)
        print(json.dumps({"stage": "chunk_started", "chunk": idx, "roots": len(chunk), "output": str(output), "log": str(log)}, sort_keys=True), flush=True)
        return idx, output, log, proc, handle

    ok = True
    finished: list[Path] = []
    todo: list[int] = []
    for idx, chunk in enumerate(chunks):
        output = chunk_dir / f"chunk_{idx:03d}.json"
        if args.resume and completed_chunk_matches(output, chunk):
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
    merged = merge_payloads(sorted(finished), Path(args.output), Path(args.normalization).resolve())
    write_json(work_dir / "chunked_hist_results.json", {"status": merged["status"], "output": str(args.output), "chunks": [str(p) for p in sorted(finished)]})
    print(json.dumps({"stage": "chunked_hists_done", "status": merged["status"], "events_processed": merged["summary"]["events_processed"], "input_roots": len(merged["summary"]["input_roots"]), "output": str(args.output)}, sort_keys=True), flush=True)
    return 0 if merged["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
