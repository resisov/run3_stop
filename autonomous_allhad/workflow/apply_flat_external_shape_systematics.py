#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REGION_TO_SOURCE = {
    "LLCR": ("LLCR", "recoil_pt"),
    "QCDCR": ("QCDCR", "recoil_pt"),
    "GCR": ("GCR", "recoil_pt"),
    "DY2E": ("DY2E", "recoil_pt"),
    "DY2M": ("DY2M", "recoil_pt"),
    "SR": ("SR", "recoil_pt"),
}
SHIFTS = {
    "jesTotalUp": "jesTotal",
    "jesTotalDown": "jesTotal",
    "metUnclusteredUp": "metUnclustered",
    "metUnclusteredDown": "metUnclustered",
}
DATA_PROCESSES = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}
SEARCH_SOURCE_SCHEME = "boosted_an_17"
SEARCH_TARGET_SCHEME = "boosted_an_17_SR_Nt1"
SELECTED_RECOIL54_SCHEME = "boosted_an17_selected_recoil6_with_nt0_wsplit_SR"
SELECTED_RECOIL54_REGIONS = [
    "SR_Nb1plus_T0_W0",
    "SR_Nb1plus_T0_W1plus",
    "SR_Nb1_T1plus_W0",
    "SR_Nb1_T1plus_W1plus",
    "SR_Nb2_T1_W0",
    "SR_Nb2_T1_W1",
    "SR_Nb3plus_T1_W0",
    "SR_Nb3plus_T1_W1",
    "SR_Nb3plus_T2_W0",
]
RECOIL6_MET_GROUPS = ((0,), (1,), (2,), (3,), (4, 5), (6, 7))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def canonical_process(process: str, dataset: str = "") -> str:
    if process in DATA_PROCESSES:
        return process
    if process == "VV" or dataset.startswith(("WW", "WZ", "ZZ")):
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "ST"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "TT"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "GJ"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "WtoLNu"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Zto2Nu"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD"
    return process or "other"


def sample_is_signal(name: str) -> bool:
    return name.startswith("T2tt_")


def dataset_factor_map(norm: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in (norm.get("dataset_factors") or {}).values():
        dataset = rec.get("dataset")
        factor = rec.get("normalization_factor")
        if dataset is None or factor is None:
            continue
        try:
            out[str(dataset)] = float(factor)
        except Exception:
            continue
    return out


def add_arr(target: dict[tuple[str, str], np.ndarray], key: tuple[str, str], values: Any, factor: float, nbin: int) -> None:
    vals = np.asarray(values or [], dtype=float)
    arr = target.setdefault(key, np.zeros(nbin, dtype=float))
    n = min(nbin, len(vals))
    if n:
        arr[:n] += vals[:n] * factor


def add_search_arr(target: dict[str, np.ndarray], process: str, rec: dict[str, Any], labels: list[str], factor: float) -> None:
    arr = target.setdefault(process, np.zeros(len(labels), dtype=float))
    bins = ((rec.get("search_bins") or {}).get(SEARCH_SOURCE_SCHEME) or {})
    for idx, label in enumerate(labels):
        item = bins.get(label) or {}
        arr[idx] += float(item.get("raw_weighted") or 0.0) * factor


def aggregate_source_dir(source_dir: Path, factors: dict[str, float], labels: list[str]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    region_hists: dict[tuple[str, str], np.ndarray] = {}
    search_hists: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "path": str(source_dir),
        "files": 0,
        "datasets": 0,
        "missing_factor": 0,
        "missing_histograms": defaultdict(int),
        "processes": defaultdict(int),
    }
    for path in sorted(source_dir.glob("*.json")):
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        meta["files"] += 1
        for _dataset_key, rec in (payload.get("datasets") or {}).items():
            dataset = str(rec.get("dataset") or "")
            process = canonical_process(str(rec.get("process") or ""), dataset)
            if rec.get("is_data") or rec.get("is_signal") or process in DATA_PROCESSES:
                continue
            factor = factors.get(dataset)
            if factor is None:
                meta["missing_factor"] += 1
                continue
            meta["datasets"] += 1
            meta["processes"][process] += 1
            for target_region, (source_region, variable) in REGION_TO_SOURCE.items():
                hist = (((rec.get("histograms") or {}).get(source_region) or {}).get(variable) or {})
                values = hist.get("raw_values")
                if values is None:
                    meta["missing_histograms"][f"{source_region}/{variable}"] += 1
                    continue
                add_arr(region_hists, (target_region, process), values, factor, 6)
            add_search_arr(search_hists, process, rec, labels, factor)
    meta["missing_histograms"] = dict(meta["missing_histograms"])
    meta["processes"] = dict(sorted(meta["processes"].items()))
    return region_hists, search_hists, meta


def current_nominal(rec: dict[str, Any], nbin: int) -> np.ndarray:
    vals = np.asarray(((rec or {}).get("nominal") or {}).get("sumw") or [], dtype=float)
    out = np.zeros(nbin, dtype=float)
    n = min(nbin, len(vals))
    if n:
        out[:n] = vals[:n]
    return out


def hist_from_values(values: np.ndarray, nominal_rec: dict[str, Any]) -> dict[str, Any]:
    nominal = nominal_rec.get("nominal") or {}
    entries = list(nominal.get("entries") or [0] * len(values))
    sumw2 = list(nominal.get("sumw2") or [0.0] * len(values))
    return {"sumw": [float(x) for x in values], "sumw2": sumw2[: len(values)], "entries": entries[: len(values)]}


def scaled_shift(current: np.ndarray, old_nominal: np.ndarray | None, old_shifted: np.ndarray | None) -> tuple[np.ndarray, dict[str, float | str]]:
    if old_nominal is None or old_shifted is None:
        return current.copy(), {"status": "missing_source"}
    delta = old_shifted - old_nominal
    old_total = float(np.sum(old_nominal))
    current_total = float(np.sum(current))
    scale = 1.0
    status = "absolute_delta"
    if math.isfinite(old_total) and math.isfinite(current_total) and abs(old_total) > 1.0e-9:
        scale = current_total / old_total
        if not math.isfinite(scale) or abs(scale) > 10.0:
            scale = 1.0
            status = "absolute_delta_scale_protected"
        else:
            status = "delta_scaled_by_process_yield"
    varied = current + delta * scale
    return varied, {"status": status, "scale": float(scale), "old_total": old_total, "current_total": current_total}


def apply_region_shapes(flat: dict[str, Any], old_nom: dict[tuple[str, str], np.ndarray], shifted: dict[str, dict[tuple[str, str], np.ndarray]], meta: dict[str, Any]) -> None:
    applied: dict[str, Any] = {}
    for region in REGION_TO_SOURCE:
        by_sample = (flat.get("histograms") or {}).get(region) or {}
        for sample, variations in by_sample.items():
            if sample == "data_obs" or sample_is_signal(sample):
                continue
            current = current_nominal(variations, 6)
            process = canonical_process(sample)
            source_key = (region, process)
            for shift_name, shifted_hists in shifted.items():
                values, info = scaled_shift(current, old_nom.get(source_key), shifted_hists.get(source_key))
                variations[shift_name] = hist_from_values(values, variations)
                applied.setdefault(region, {}).setdefault(sample, {})[shift_name] = info
    meta["applied_region_shapes"] = applied


def apply_search_shapes(flat: dict[str, Any], old_nom: dict[str, np.ndarray], shifted: dict[str, dict[str, np.ndarray]], meta: dict[str, Any]) -> None:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(SEARCH_TARGET_SCHEME) or {})
    labels = (((flat.get("search_bin_schemes") or {}).get(SEARCH_TARGET_SCHEME) or {}).get("bin_labels") or [])
    applied: dict[str, Any] = {}
    for sample, variations in by_sample.items():
        if sample == "data_obs" or sample_is_signal(sample):
            continue
        current = current_nominal(variations, len(labels))
        process = canonical_process(sample)
        for shift_name, shifted_hists in shifted.items():
            values, info = scaled_shift(current, old_nom.get(process), shifted_hists.get(process))
            variations[shift_name] = hist_from_values(values, variations)
            applied.setdefault(sample, {})[shift_name] = info
    meta["applied_search_shapes"] = applied


def parse_direct_shape_hists(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --shape-hist value {item!r}; expected SHIFT=PATH")
        shift, raw_path = item.split("=", 1)
        if shift not in SHIFTS:
            raise ValueError(f"unsupported shape shift {shift!r}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        out[shift] = path
    missing = sorted(set(SHIFTS) - set(out))
    if missing:
        raise ValueError(f"missing direct shape histograms: {missing}")
    return out


def scaled_hist_leaf(source: dict[str, Any], scale: float) -> dict[str, Any]:
    sumw = np.asarray(source.get("sumw") or [], dtype=float)
    sumw2 = np.asarray(source.get("sumw2") or [], dtype=float)
    entries = np.asarray(source.get("entries") or [], dtype=int)
    return {
        "sumw": (sumw * scale).tolist(),
        "sumw2": (sumw2 * scale * scale).tolist(),
        "entries": entries.astype(int).tolist(),
    }


def scaled_hist_tree(source: Any, scale: float) -> Any:
    if isinstance(source, dict) and all(key in source for key in ("sumw", "sumw2", "entries")):
        return scaled_hist_leaf(source, scale)
    if isinstance(source, dict):
        return {key: scaled_hist_tree(value, scale) for key, value in source.items()}
    return source


def replace_signal_nodes(target: dict[str, Any], source: dict[str, Any], levels_before_sample: int, scale: float) -> int:
    if levels_before_sample == 0:
        replaced = 0
        for sample, variations in source.items():
            if not sample_is_signal(sample):
                continue
            target[sample] = scaled_hist_tree(variations, scale)
            replaced += 1
        return replaced
    replaced = 0
    for key, subtree in source.items():
        if not isinstance(subtree, dict):
            continue
        replaced += replace_signal_nodes(
            target.setdefault(key, {}),
            subtree,
            levels_before_sample - 1,
            scale,
        )
    return replaced


def replace_signal_histograms(flat: dict[str, Any], signal: dict[str, Any], scale: float, meta: dict[str, Any]) -> None:
    if not str(signal.get("status") or "").startswith("complete"):
        raise RuntimeError(f"nominal signal histogram status is {signal.get('status')!r}")
    replaced = {
        "histograms": replace_signal_nodes(
            flat.setdefault("histograms", {}), signal.get("histograms") or {}, 1, scale
        ),
        "search_bin_histograms": replace_signal_nodes(
            flat.setdefault("search_bin_histograms", {}),
            signal.get("search_bin_histograms") or {},
            1,
            scale,
        ),
        "lowdm_variable_histograms": replace_signal_nodes(
            flat.setdefault("lowdm_variable_histograms", {}),
            signal.get("lowdm_variable_histograms") or {},
            2,
            scale,
        ),
        "highdm_variable_histograms": replace_signal_nodes(
            flat.setdefault("highdm_variable_histograms", {}),
            signal.get("highdm_variable_histograms") or {},
            2,
            scale,
        ),
    }
    meta["nominal_signal_replacements"] = replaced
    meta["nominal_signal_status"] = {
        "status": signal.get("status"),
        "events_processed": int((signal.get("summary") or {}).get("events_processed") or 0),
        "input_roots": len((signal.get("summary") or {}).get("input_roots") or []),
        "scale": float(scale),
    }


def zero_hist_like(target: dict[str, Any]) -> dict[str, Any]:
    nominal = target.get("nominal") or {}
    nbin = len(nominal.get("sumw") or [])
    return {"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin}


def shape_samples(payload: dict[str, Any]) -> set[str]:
    samples: set[str] = set()
    for by_sample in (payload.get("histograms") or {}).values():
        samples.update(by_sample)
    for variables in (payload.get("highdm_variable_histograms") or {}).values():
        for by_sample in variables.values():
            samples.update(by_sample)
    return samples


def zero_leaf(nbin: int) -> dict[str, Any]:
    return {"sumw": [0.0] * nbin, "sumw2": [0.0] * nbin, "entries": [0] * nbin}


def recoil6_from_met(source: dict[str, Any]) -> dict[str, Any]:
    arrays = {
        "sumw": np.asarray(source.get("sumw") or [], dtype=float),
        "sumw2": np.asarray(source.get("sumw2") or [], dtype=float),
        "entries": np.asarray(source.get("entries") or [], dtype=int),
    }
    if any(len(values) != 8 for values in arrays.values()):
        raise RuntimeError(f"expected eight MET bins, got {[len(values) for values in arrays.values()]}")
    return {
        key: [float(np.sum(values[list(group)])) if key != "entries" else int(np.sum(values[list(group)])) for group in RECOIL6_MET_GROUPS]
        for key, values in arrays.items()
    }


def selected_recoil54_leaf(payload: dict[str, Any], sample: str, known_samples: set[str]) -> dict[str, Any] | None:
    if sample not in known_samples:
        return None
    leaves: list[dict[str, Any]] = []
    tree = payload.get("highdm_variable_histograms") or {}
    for region in SELECTED_RECOIL54_REGIONS:
        source = ((((tree.get(region) or {}).get("met") or {}).get(sample) or {}).get("nominal"))
        leaves.append(recoil6_from_met(source) if source is not None else zero_leaf(6))
    return {
        key: [value for leaf in leaves for value in leaf[key]]
        for key in ("sumw", "sumw2", "entries")
    }


def apply_direct_highdm_shapes(
    flat: dict[str, Any],
    shape_payloads: dict[str, dict[str, Any]],
    shape_paths: dict[str, Path],
    scale: float,
    meta: dict[str, Any],
) -> None:
    target_tree = flat.get("highdm_variable_histograms") or {}
    samples_by_shift: dict[str, set[str]] = {}
    for shift, payload in shape_payloads.items():
        if not str(payload.get("status") or "").startswith("complete"):
            raise RuntimeError(f"{shift} shape histogram status is {payload.get('status')!r}")
        samples_by_shift[shift] = shape_samples(payload)

    attached = 0
    missing_as_zero: list[dict[str, str]] = []
    missing_processes: list[dict[str, str]] = []
    for region, variables in target_tree.items():
        for variable, by_sample in variables.items():
            for sample, target_variations in by_sample.items():
                if sample == "data_obs" or sample_is_signal(sample):
                    continue
                target_nbin = len(((target_variations.get("nominal") or {}).get("sumw") or []))
                for shift, payload in shape_payloads.items():
                    source_variations = (
                        (((payload.get("highdm_variable_histograms") or {}).get(region) or {}).get(variable) or {}).get(sample)
                        or {}
                    )
                    source = source_variations.get("nominal")
                    if source is None:
                        if sample not in samples_by_shift[shift]:
                            missing_processes.append(
                                {"shift": shift, "region": region, "variable": variable, "sample": sample}
                            )
                            continue
                        target_variations[shift] = zero_hist_like(target_variations)
                        missing_as_zero.append(
                            {"shift": shift, "region": region, "variable": variable, "sample": sample}
                        )
                        attached += 1
                        continue
                    source_nbin = len(source.get("sumw") or [])
                    if source_nbin != target_nbin:
                        raise RuntimeError(
                            f"bin mismatch for {shift}/{region}/{variable}/{sample}: "
                            f"source={source_nbin}, target={target_nbin}"
                        )
                    target_variations[shift] = scaled_hist_leaf(source, scale)
                    attached += 1
    if missing_processes:
        raise RuntimeError(f"shape histograms are missing target processes: {missing_processes[:20]}")
    meta["attached_highdm_variations"] = attached
    meta["missing_region_sample_histograms_treated_as_zero"] = missing_as_zero
    meta["shape_histogram_status"] = {
        shift: {
            "path": str(shape_paths[shift]),
            "status": shape_payloads[shift].get("status"),
            "events_processed": int((shape_payloads[shift].get("summary") or {}).get("events_processed") or 0),
            "input_roots": len((shape_payloads[shift].get("summary") or {}).get("input_roots") or []),
        }
        for shift in sorted(shape_payloads)
    }


def apply_direct_region_shapes(
    flat: dict[str, Any],
    shape_payloads: dict[str, dict[str, Any]],
    scale: float,
    meta: dict[str, Any],
) -> None:
    samples_by_shift = {shift: shape_samples(payload) for shift, payload in shape_payloads.items()}
    attached = 0
    zeroed = 0
    skipped_signal_processes: set[str] = set()
    for region, by_sample in (flat.get("histograms") or {}).items():
        for sample, target_variations in by_sample.items():
            if sample == "data_obs":
                continue
            target_nbin = len(((target_variations.get("nominal") or {}).get("sumw") or []))
            for shift, payload in shape_payloads.items():
                source = (((payload.get("histograms") or {}).get(region) or {}).get(sample) or {}).get("nominal")
                if source is None:
                    if sample not in samples_by_shift[shift]:
                        if sample_is_signal(sample):
                            skipped_signal_processes.add(sample)
                            continue
                        raise RuntimeError(f"{shift} region shapes are missing background process {sample!r}")
                    target_variations[shift] = zero_leaf(target_nbin)
                    zeroed += 1
                    attached += 1
                    continue
                if len(source.get("sumw") or []) != target_nbin:
                    raise RuntimeError(
                        f"bin mismatch for {shift}/{region}/{sample}: "
                        f"source={len(source.get('sumw') or [])}, target={target_nbin}"
                    )
                target_variations[shift] = scaled_hist_leaf(source, scale)
                attached += 1
    meta["attached_region_variations"] = attached
    meta["region_histograms_treated_as_zero"] = zeroed
    meta["region_signal_processes_without_shifted_input"] = sorted(skipped_signal_processes)


def apply_direct_selected_recoil54_shapes(
    flat: dict[str, Any],
    shape_payloads: dict[str, dict[str, Any]],
    scale: float,
    meta: dict[str, Any],
) -> None:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(SELECTED_RECOIL54_SCHEME) or {})
    if not by_sample:
        meta["attached_selected_recoil54_variations"] = 0
        meta["selected_recoil54_status"] = "target_scheme_absent"
        return
    labels = (((flat.get("search_bin_schemes") or {}).get(SELECTED_RECOIL54_SCHEME) or {}).get("bin_labels") or [])
    if len(labels) != 54:
        raise RuntimeError(f"{SELECTED_RECOIL54_SCHEME} has {len(labels)} labels, expected 54")
    samples_by_shift = {shift: shape_samples(payload) for shift, payload in shape_payloads.items()}
    attached = 0
    skipped_signal_processes: set[str] = set()
    for sample, target_variations in by_sample.items():
        if sample == "data_obs":
            continue
        for shift, payload in shape_payloads.items():
            source = selected_recoil54_leaf(payload, sample, samples_by_shift[shift])
            if source is None:
                if sample_is_signal(sample):
                    skipped_signal_processes.add(sample)
                    continue
                raise RuntimeError(f"{shift} selected-recoil shapes are missing background process {sample!r}")
            target_variations[shift] = scaled_hist_leaf(source, scale)
            attached += 1
    meta["attached_selected_recoil54_variations"] = attached
    meta["selected_recoil54_status"] = "complete"
    meta["selected_recoil54_signal_processes_without_shifted_input"] = sorted(skipped_signal_processes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach existing JES/MET shifted production-output shapes to current flat histograms.")
    parser.add_argument("--flat-hists", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--nominal-dir")
    parser.add_argument("--shape-base-dir")
    parser.add_argument("--shape-prefix", default="production_outputs_boosted_an17_20260629")
    parser.add_argument(
        "--shape-hist",
        action="append",
        default=[],
        help="Attach a reduced shape histogram directly, as SHIFT=PATH. Provide all four shifts.",
    )
    parser.add_argument(
        "--shape-scale",
        type=float,
        default=1.0,
        help="Scale direct shape sumw by this factor and sumw2 by its square.",
    )
    parser.add_argument(
        "--signal-hists",
        help="Replace signal histogram nodes with this corrected nominal signal payload before attaching shapes.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    flat = read_json(Path(args.flat_hists))
    if args.shape_hist:
        shape_paths = parse_direct_shape_hists(args.shape_hist)
        shape_payloads = {shift: read_json(path) for shift, path in shape_paths.items()}
        meta: dict[str, Any] = {
            "status": "complete",
            "method": "direct reduced shifted-flat histograms with nominal event weights",
            "normalization": str(args.normalization),
            "shape_names": sorted(shape_paths),
            "shape_scale": float(args.shape_scale),
        }
        if args.signal_hists:
            signal_path = Path(args.signal_hists)
            if not signal_path.exists():
                raise FileNotFoundError(signal_path)
            replace_signal_histograms(flat, read_json(signal_path), float(args.shape_scale), meta)
            meta["nominal_signal_path"] = str(signal_path)
        apply_direct_highdm_shapes(flat, shape_payloads, shape_paths, float(args.shape_scale), meta)
        apply_direct_region_shapes(flat, shape_payloads, float(args.shape_scale), meta)
        apply_direct_selected_recoil54_shapes(flat, shape_payloads, float(args.shape_scale), meta)
        flat.setdefault("summary", {})["external_shape_systematics"] = meta
        flat["status"] = "complete"
        write_json(Path(args.output), flat)
        print(json.dumps({
            "status": "complete",
            "output": args.output,
            "shape_names": sorted(shape_paths),
            "attached_highdm_variations": meta["attached_highdm_variations"],
            "attached_region_variations": meta["attached_region_variations"],
            "attached_selected_recoil54_variations": meta["attached_selected_recoil54_variations"],
            "missing_as_zero": len(meta["missing_region_sample_histograms_treated_as_zero"]),
        }, sort_keys=True))
        return 0

    if not args.nominal_dir or not args.shape_base_dir:
        raise SystemExit("--nominal-dir and --shape-base-dir are required without --shape-hist")
    norm = read_json(Path(args.normalization))
    factors = dataset_factor_map(norm)
    labels = list((((flat.get("search_bin_schemes") or {}).get(SEARCH_TARGET_SCHEME) or {}).get("bin_labels") or []))
    if not labels:
        raise SystemExit(f"missing {SEARCH_TARGET_SCHEME} labels in flat hists")

    nominal_dir = Path(args.nominal_dir)
    old_region_nominal, old_search_nominal, nominal_meta = aggregate_source_dir(nominal_dir, factors, labels)
    shifted_regions: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    shifted_search: dict[str, dict[str, np.ndarray]] = {}
    shape_meta: dict[str, Any] = {}
    base = Path(args.shape_base_dir)
    for shift_name in SHIFTS:
        source_dir = base / f"{args.shape_prefix}_{shift_name}"
        if not source_dir.exists():
            shape_meta[shift_name] = {"status": "missing", "path": str(source_dir)}
            continue
        reg, search, meta = aggregate_source_dir(source_dir, factors, labels)
        shifted_regions[shift_name] = reg
        shifted_search[shift_name] = search
        shape_meta[shift_name] = meta

    meta: dict[str, Any] = {
        "status": "complete",
        "method": "normalize 20260629 shifted production-output raw histograms with current dataset factors; apply process-yield-scaled deltas to current flat nominal histograms",
        "normalization": str(args.normalization),
        "source_nominal": nominal_meta,
        "source_shapes": shape_meta,
        "shape_names": sorted(SHIFTS),
    }
    apply_region_shapes(flat, old_region_nominal, shifted_regions, meta)
    apply_search_shapes(flat, old_search_nominal, shifted_search, meta)
    flat.setdefault("summary", {})["external_shape_systematics"] = meta
    flat["status"] = "complete"
    write_json(Path(args.output), flat)
    print(json.dumps({"status": "complete", "output": args.output, "shape_names": sorted(SHIFTS), "source_nominal_files": nominal_meta.get("files")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
