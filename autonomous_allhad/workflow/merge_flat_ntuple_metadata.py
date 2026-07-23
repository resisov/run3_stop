#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

LUMI_FB_DEFAULT = 109.82
LUMI_PB_DEFAULT = LUMI_FB_DEFAULT * 1000.0
FLAT_SCHEMAS = {
    "flat_ntuple_shard_v1",
    "flat_ntuple_shard_v2_lowdm",
    "flat_ntuple_shard_v4_objectcorr_2024",
    "flat_ntuple_shard_v4_objectcorr_2025_data",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value: Any, fill: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fill
    return out if math.isfinite(out) else fill


def positive(value: Any) -> float | None:
    out = finite(value, float("nan"))
    return out if math.isfinite(out) and out > 0.0 else None


def add_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, val in (source or {}).items():
        target[str(key)] = int(target.get(str(key), 0)) + int(val or 0)


def add_float_map(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, val in (source or {}).items():
        target[str(key)] = finite(target.get(str(key), 0.0)) + finite(val)


def parse_genmodel(genmodel: str) -> tuple[int | None, int | None, str]:
    nums = re.findall(r"(\d+)", str(genmodel))
    if len(nums) >= 2:
        mstop, mlsp = int(nums[-2]), int(nums[-1])
        return mstop, mlsp, f"mStop{mstop}_mLSP{mlsp}"
    return None, None, str(genmodel)


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            matches = [Path(x) for x in glob.glob(item)]
            paths.extend(matches or [p])
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_signal_xsec(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for key, rec in (payload.get("mass_points") or {}).items():
        xsec = positive(rec.get("xsec_pb"))
        if xsec is None:
            continue
        out[str(key)] = {
            "xsec_pb": xsec,
            "xsec_uncertainty_relative": rec.get("xsec_uncertainty_relative"),
            "xsec_uncertainty_percent": rec.get("xsec_uncertainty_percent"),
        }
    return out


def merge_dataset(target: dict[str, Any], rec: dict[str, Any]) -> None:
    for key in ["files_attempted", "files_processed", "events_read", "events_written"]:
        target[key] = int(target.get(key, 0)) + int(rec.get(key) or 0)
    for key in ["sumw", "sumw2"]:
        target[key] = finite(target.get(key, 0.0)) + finite(rec.get(key))
    add_counts(target.setdefault("sumw_source_counts", {}), rec.get("sumw_source_counts") or {})
    add_counts(target.setdefault("signal_runs_sumw_source_counts", {}), rec.get("signal_runs_sumw_source_counts") or {})
    add_float_map(target.setdefault("signal_sumw_by_genmodel", {}), rec.get("signal_sumw_by_genmodel") or {})
    add_float_map(target.setdefault("signal_event_genweight_sum_by_genmodel", {}), rec.get("signal_event_genweight_sum_by_genmodel") or {})


def build_physical(datasets: dict[str, Any], lumi_pb: float) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    physical: dict[str, Any] = {}
    split_counts: dict[str, int] = {}
    factors: dict[str, Any] = {}
    for dsid, rec in datasets.items():
        phys = str(rec.get("physical_dataset") or rec.get("dataset") or dsid)
        split_counts[phys] = split_counts.get(phys, 0) + 1
        prec = physical.setdefault(phys, {
            "physical_dataset": phys,
            "process": rec.get("process"),
            "is_data": bool(rec.get("is_data")),
            "is_signal": bool(rec.get("is_signal")),
            "is_background": bool(rec.get("is_background")),
            "xsec_pb": rec.get("xsec_pb"),
            "sumw": 0.0,
            "sumw2": 0.0,
            "files_attempted": 0,
            "files_processed": 0,
            "events_written": 0,
            "split_dataset_ids": [],
            "sumw_source_counts": {},
            "xsec_conflicts": [],
        })
        xs = rec.get("xsec_pb")
        if not rec.get("is_data") and xs is not None and prec.get("xsec_pb") is not None and abs(finite(xs) - finite(prec.get("xsec_pb"))) > 1.0e-12:
            prec["xsec_conflicts"].append({"dataset_id": dsid, "xsec_pb": xs})
        elif prec.get("xsec_pb") is None:
            prec["xsec_pb"] = xs
        prec["sumw"] += finite(rec.get("sumw"))
        prec["sumw2"] += finite(rec.get("sumw2"))
        prec["files_attempted"] += int(rec.get("files_attempted") or 0)
        prec["files_processed"] += int(rec.get("files_processed") or 0)
        prec["events_written"] += int(rec.get("events_written") or 0)
        prec["split_dataset_ids"].append(dsid)
        add_counts(prec["sumw_source_counts"], rec.get("sumw_source_counts") or {})
    for phys, prec in physical.items():
        xs = positive(prec.get("xsec_pb"))
        sumw = finite(prec.get("sumw"))
        if prec.get("is_data"):
            factor = 1.0
            status = "data_unscaled"
        elif prec.get("is_signal"):
            factor = None
            status = "signal_uses_mass_point_sumw_not_physical_dataset_sumw"
        elif prec.get("xsec_conflicts"):
            factor = None
            status = "blocked_inconsistent_xsec_across_split_datasets"
        elif xs is None:
            factor = None
            status = "blocked_missing_positive_xsec"
        elif sumw == 0.0:
            factor = None
            status = "blocked_zero_sumw"
        else:
            factor = xs * lumi_pb / sumw
            status = "normalized_with_xsec_lumi_physical_dataset_sumw"
        prec["normalization_factor"] = factor
        prec["normalization_status"] = status
        for dsid in prec["split_dataset_ids"]:
            rec = datasets[dsid]
            factors[dsid] = {
                "dataset": rec.get("dataset"),
                "dataset_id": dsid,
                "physical_dataset": phys,
                "process": rec.get("process"),
                "is_data": rec.get("is_data"),
                "is_signal": rec.get("is_signal"),
                "xsec_pb": rec.get("xsec_pb"),
                "dataset_sumw": rec.get("sumw"),
                "physical_dataset_sumw": prec.get("sumw"),
                "normalization_factor": factor,
                "normalization_status": status,
            }
    return physical, split_counts, factors


def build_signal_mass_points(datasets: dict[str, Any], signal_xsec: dict[str, dict[str, Any]], lumi_pb: float) -> dict[str, Any]:
    points: dict[str, Any] = {}
    for rec in datasets.values():
        if not rec.get("is_signal"):
            continue
        for genmodel, sumw in (rec.get("signal_sumw_by_genmodel") or {}).items():
            mstop, mlsp, key = parse_genmodel(genmodel)
            item = points.setdefault(key, {
                "mass_key": key,
                "mStop": mstop,
                "mLSP": mlsp,
                "genmodel_branch": genmodel,
                "sumw_mass_point": 0.0,
                "event_genweight_sum_fallback": 0.0,
                "sumw_source": "Runs.genEventSumw_T2tt_<mStop>_<mLSP>",
                "datasets": [],
            })
            item["sumw_mass_point"] += finite(sumw)
            item["datasets"].append(rec.get("dataset"))
        for genmodel, sumw in (rec.get("signal_event_genweight_sum_by_genmodel") or {}).items():
            _mstop, _mlsp, key = parse_genmodel(genmodel)
            item = points.setdefault(key, {
                "mass_key": key,
                "mStop": _mstop,
                "mLSP": _mlsp,
                "genmodel_branch": genmodel,
                "sumw_mass_point": 0.0,
                "event_genweight_sum_fallback": 0.0,
                "sumw_source": "missing_runs_sumw_uses_event_genweight_fallback_only_if_explicitly_approved",
                "datasets": [],
            })
            item["event_genweight_sum_fallback"] += finite(sumw)
    for key, item in points.items():
        xrec = signal_xsec.get(key) or {}
        xsec = positive(xrec.get("xsec_pb"))
        sumw = finite(item.get("sumw_mass_point"))
        item["xsec_pb"] = xsec
        item["xsec_uncertainty_relative"] = xrec.get("xsec_uncertainty_relative")
        if xsec is not None and sumw != 0.0:
            item["normalization_factor"] = xsec * lumi_pb / sumw
            item["normalization_status"] = "normalized_with_signal_xsec_and_runs_mass_point_sumw"
        elif sumw == 0.0:
            item["normalization_factor"] = None
            item["normalization_status"] = "blocked_zero_or_missing_runs_mass_point_sumw"
        else:
            item["normalization_factor"] = None
            item["normalization_status"] = "blocked_missing_signal_xsec"
        item["datasets"] = sorted(set(str(x) for x in item.get("datasets") or []))
    return dict(sorted(points.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge flat ntuple shard sidecars and compute campaign-level normalization factors.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Sidecar JSON files, directories, or glob patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--signal-yields", default="autonomous_allhad/outputs/signal_yields_by_mass.json")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    datasets: dict[str, Any] = {}
    source_records = []
    lumi_pb = LUMI_PB_DEFAULT
    for path in paths:
        payload = read_json(path)
        if payload.get("schema_version") not in FLAT_SCHEMAS:
            continue
        policy = payload.get("normalization_policy") or {}
        lumi_pb = finite(policy.get("luminosity_pb"), lumi_pb)
        source_records.append({
            "path": str(path),
            "status": payload.get("status"),
            "files_processed": payload.get("files_processed"),
            "files_attempted": payload.get("files_attempted"),
            "events_written": payload.get("events_written"),
        })
        for dsid, rec in (payload.get("datasets") or {}).items():
            target = datasets.setdefault(str(dsid), {
                "dataset": rec.get("dataset"),
                "dataset_id": str(dsid),
                "physical_dataset": rec.get("physical_dataset"),
                "process": rec.get("process"),
                "xsec_pb": rec.get("xsec_pb"),
                "is_data": bool(rec.get("is_data")),
                "is_signal": bool(rec.get("is_signal")),
                "is_background": bool(rec.get("is_background")),
                "files_attempted": 0,
                "files_processed": 0,
                "events_read": 0,
                "events_written": 0,
                "sumw": 0.0,
                "sumw2": 0.0,
                "sumw_source_counts": {},
                "signal_runs_sumw_source_counts": {},
                "signal_sumw_by_genmodel": {},
                "signal_event_genweight_sum_by_genmodel": {},
            })
            merge_dataset(target, rec)
    signal_xsec = load_signal_xsec(Path(args.signal_yields) if args.signal_yields else None)
    physical, split_counts, factors = build_physical(datasets, lumi_pb)
    signal_points = build_signal_mass_points(datasets, signal_xsec, lumi_pb)
    output = {
        "schema_version": "flat_ntuple_campaign_normalization_v1",
        "status": "complete" if source_records else "empty",
        "source_sidecars": source_records,
        "source_sidecar_count": len(source_records),
        "luminosity_pb": lumi_pb,
        "luminosity_fb": lumi_pb / 1000.0,
        "normalization_policy": {
            "flat_root_event_weight_status": "raw gen_weight only; no xsec/lumi or scale factor applied in ROOT ntuples",
            "background_formula": "gen_weight * post_skim_sf_weight * xsec_pb * lumi_pb / physical_dataset_sumw",
            "signal_formula": "gen_weight * post_skim_sf_weight * xsec_pb(mStop) * lumi_pb / Runs.genEventSumw_T2tt_<mStop>_<mLSP>",
            "data_formula": "1",
        },
        "datasets": datasets,
        "dataset_factors": factors,
        "physical_datasets": physical,
        "physical_dataset_split_counts": split_counts,
        "signal_mass_points": signal_points,
        "blocked_dataset_factors": {k: v for k, v in factors.items() if v.get("normalization_factor") is None and not v.get("is_signal")},
        "blocked_signal_mass_points": {k: v for k, v in signal_points.items() if v.get("normalization_factor") is None},
    }
    write_json(Path(args.output), output)
    print(json.dumps({
        "status": output["status"],
        "source_sidecars": len(source_records),
        "datasets": len(datasets),
        "physical_datasets": len(physical),
        "signal_mass_points": len(signal_points),
        "blocked_dataset_factors": len(output["blocked_dataset_factors"]),
        "blocked_signal_mass_points": len(output["blocked_signal_mass_points"]),
        "output": args.output,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
