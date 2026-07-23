#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any


HISTOGRAM_TREES = (
    "histograms",
    "search_bin_histograms",
    "lowdm_variable_histograms",
    "highdm_variable_histograms",
)

RETIRED_DATA_PATHS = {
    "search_bin_histograms/boosted_an17_selected_recoil6_with_nt0_SR/data_obs",
}


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_histogram_leaf(node: Any) -> bool:
    return isinstance(node, dict) and all(key in node for key in ("sumw", "sumw2", "entries"))


def count_histogram_leaves(node: Any) -> int:
    if is_histogram_leaf(node):
        return 1
    if not isinstance(node, dict):
        return 0
    return sum(count_histogram_leaves(value) for value in node.values())


def scale_histogram_leaf(node: dict[str, Any], scale: float) -> dict[str, Any]:
    out = copy.deepcopy(node)
    out["sumw"] = [float(value) * scale for value in node.get("sumw") or []]
    out["sumw2"] = [float(value) * scale * scale for value in node.get("sumw2") or []]
    out["entries"] = [int(value) for value in node.get("entries") or []]
    return out


def zero_histogram_tree(node: Any) -> Any:
    if is_histogram_leaf(node):
        out = copy.deepcopy(node)
        out["sumw"] = [0.0] * len(node.get("sumw") or [])
        out["sumw2"] = [0.0] * len(node.get("sumw2") or [])
        out["entries"] = [0] * len(node.get("entries") or [])
        return out
    if isinstance(node, dict):
        return {key: zero_histogram_tree(value) for key, value in node.items()}
    return copy.deepcopy(node)


def compose_histogram_tree(
    base: Any,
    data: Any,
    scale: float,
    stats: dict[str, Any],
    path: tuple[str, ...] = (),
) -> Any:
    if is_histogram_leaf(base):
        stats["scaled_nondata_leaves"] += 1
        return scale_histogram_leaf(base, scale)
    if not isinstance(base, dict):
        return copy.deepcopy(base)

    data_dict = data if isinstance(data, dict) else {}
    out: dict[str, Any] = {}
    for key, value in base.items():
        if key == "data_obs":
            if key in data_dict:
                out[key] = copy.deepcopy(data_dict[key])
                stats["replaced_data_leaves"] += count_histogram_leaves(data_dict[key])
            else:
                out[key] = zero_histogram_tree(value)
                leaf_count = count_histogram_leaves(value)
                data_path = "/".join(path + (key,))
                if data_path in RETIRED_DATA_PATHS:
                    stats["zeroed_retired_data_leaves"] += leaf_count
                    stats["retired_data_paths"].append(data_path)
                else:
                    stats["zeroed_missing_data_leaves"] += leaf_count
                    stats["missing_data_paths"].append(data_path)
            continue
        out[key] = compose_histogram_tree(value, data_dict.get(key), scale, stats, path + (key,))

    if "data_obs" in data_dict and "data_obs" not in out:
        out["data_obs"] = copy.deepcopy(data_dict["data_obs"])
        stats["inserted_data_leaves"] += count_histogram_leaves(data_dict["data_obs"])
    return out


def build_payload(
    base: dict[str, Any],
    data: dict[str, Any],
    source_lumi_fb: float,
    target_lumi_fb: float,
    normalization: str,
    base_path: str,
    data_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_lumi_fb <= 0 or target_lumi_fb <= 0:
        raise ValueError("luminosities must be positive")
    scale = target_lumi_fb / source_lumi_fb
    out = {key: copy.deepcopy(value) for key, value in base.items() if key not in HISTOGRAM_TREES + ("summary",)}
    stats: dict[str, Any] = {
        "source_luminosity_fb": source_lumi_fb,
        "target_luminosity_fb": target_lumi_fb,
        "mc_signal_scale_factor": scale,
        "scaled_nondata_leaves": 0,
        "replaced_data_leaves": 0,
        "inserted_data_leaves": 0,
        "zeroed_retired_data_leaves": 0,
        "retired_data_paths": [],
        "zeroed_missing_data_leaves": 0,
        "missing_data_paths": [],
    }
    for tree_name in HISTOGRAM_TREES:
        out[tree_name] = compose_histogram_tree(
            base.get(tree_name) or {},
            data.get(tree_name) or {},
            scale,
            stats,
            (tree_name,),
        )

    data_summary = copy.deepcopy(data.get("summary") or {})
    out["summary"] = data_summary
    out["summary"].update(
        {
            "base_histogram_payload": base_path,
            "data_histogram_payload": data_path,
            "source_luminosity_fb": source_lumi_fb,
            "target_luminosity_fb": target_lumi_fb,
            "mc_signal_scale_factor": scale,
            "base_events_processed": int((base.get("summary") or {}).get("events_processed") or 0),
            "data_events_processed": int(data_summary.get("events_processed") or 0),
            "composition_stats": stats,
        }
    )
    out["normalization"] = normalization
    out["schema_version"] = "flat_boosted_recoil_hists_2025_data_v1"
    if data.get("status") != "complete":
        out["status"] = str(data.get("status") or "incomplete")
    elif stats["missing_data_paths"]:
        out["status"] = "complete_with_missing_data"
    else:
        out["status"] = "complete"
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose the 2025 analysis histogram payload.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source-lumi-fb", required=True, type=float)
    parser.add_argument("--target-lumi-fb", required=True, type=float)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    base = read_json(args.base)
    data = read_json(args.data)
    payload, stats = build_payload(
        base,
        data,
        args.source_lumi_fb,
        args.target_lumi_fb,
        args.normalization,
        str(args.base),
        str(args.data),
    )
    write_json(args.output, payload)
    summary = {
        "status": payload["status"],
        "output": str(args.output),
        "base": str(args.base),
        "data": str(args.data),
        **stats,
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
