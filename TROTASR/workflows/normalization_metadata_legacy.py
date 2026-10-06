"""Preserved normalization kernels; internal boundary only, no alternate physics."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from TROTASR.utils.io import read_json
from TROTASR.utils.signal_models import signal_mass_from_genmodel, signal_mass_key
SIGNAL_XSEC_SCHEMA = "stop_pair_xsec_13p6tev_v1"

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


def merge_signal_cutflow_histograms(target: dict[str, Any], source: dict[str, Any]) -> None:
    if not source:
        return
    if source.get("schema_version") != "signal_cutflow_histograms_v1":
        raise RuntimeError(f"unexpected signal cutflow schema: {source.get('schema_version')}")
    if not target:
        target.update(
            {
                "schema_version": source["schema_version"],
                "sample_scope": "signal_only",
                "weight_scope": source.get("weight_scope"),
                "topologies": {},
            }
        )
    for topology, regions in (source.get("topologies") or {}).items():
        target_regions = target["topologies"].setdefault(str(topology), {})
        for region, histogram in (regions or {}).items():
            labels = [str(value) for value in histogram.get("labels") or []]
            edges = [finite(value) for value in histogram.get("bin_edges") or []]
            entries = [int(value) for value in histogram.get("entries") or []]
            raw_values = [finite(value) for value in histogram.get("raw_values") or []]
            raw_sumw2 = [finite(value) for value in histogram.get("raw_sumw2") or []]
            if not (
                len(labels) == len(entries) == len(raw_values) == len(raw_sumw2)
                and len(edges) == len(labels) + 1
            ):
                raise RuntimeError(f"malformed signal cutflow histogram: {topology}/{region}")
            existing = target_regions.get(str(region))
            if existing is None:
                target_regions[str(region)] = {
                    "schema_version": "cumulative_cutflow_histogram_v1",
                    "cumulative": True,
                    "labels": labels,
                    "bin_edges": edges,
                    "entries": entries,
                    "raw_values": raw_values,
                    "raw_sumw2": raw_sumw2,
                }
                continue
            if existing["labels"] != labels or existing["bin_edges"] != edges:
                raise RuntimeError(f"incompatible signal cutflow binning: {topology}/{region}")
            for key, values in (
                ("entries", entries),
                ("raw_values", raw_values),
                ("raw_sumw2", raw_sumw2),
            ):
                existing[key] = [
                    current + value for current, value in zip(existing[key], values)
                ]


def parse_genmodel(genmodel: str) -> tuple[int | None, int | None, str]:
    topology, mstop, mlsp = signal_mass_from_genmodel(genmodel)
    if topology and mstop is not None and mlsp is not None:
        return mstop, mlsp, signal_mass_key(topology, mstop, mlsp)
    return None, None, str(genmodel)


def load_signal_xsec(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"authoritative signal xsec JSON does not exist: {path}")
    payload = read_json(path)
    if payload.get("schema_version") != SIGNAL_XSEC_SCHEMA:
        raise RuntimeError(
            "refusing non-authoritative signal xsec input: expected schema "
            f"{SIGNAL_XSEC_SCHEMA!r}, got {payload.get('schema_version')!r} from {path}"
        )
    if "mass_points" in payload:
        raise RuntimeError(
            f"refusing yield-shaped JSON as a signal xsec source: {path}"
        )
    if (
        payload.get("source_file") != "signal_xsec.txt"
        or payload.get("xsec_table_status") != "parsed"
        or payload.get("parsed") is not True
    ):
        raise RuntimeError(f"signal xsec provenance is incomplete or invalid: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"signal xsec records are missing: {path}")
    if int(payload.get("records_parsed") or -1) != len(records):
        raise RuntimeError(f"signal xsec record count is inconsistent: {path}")
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        mstop = rec.get("mStop")
        xsec = positive(rec.get("xsec_pb"))
        if mstop is None or xsec is None or rec.get("parsing_status") != "parsed":
            raise RuntimeError(f"invalid signal xsec record in {path}: {rec!r}")
        key = f"mStop{int(mstop)}"
        if key in out:
            raise RuntimeError(f"duplicate signal xsec record {key} in {path}")
        out[key] = {
            "xsec_pb": xsec,
            "xsec_uncertainty_relative": rec.get(
                "uncertainty_relative"
            ),
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
                "topology": signal_mass_from_genmodel(genmodel)[0],
                "genmodel_branch": genmodel,
                "sumw_mass_point": 0.0,
                "event_genweight_sum_fallback": 0.0,
                "sumw_source": "Runs.genEventSumw_<topology>_<mStop>_<mLSP>",
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
                "topology": signal_mass_from_genmodel(genmodel)[0],
                "genmodel_branch": genmodel,
                "sumw_mass_point": 0.0,
                "event_genweight_sum_fallback": 0.0,
                "sumw_source": "missing_runs_sumw_uses_event_genweight_fallback_only_if_explicitly_approved",
                "datasets": [],
            })
            item["event_genweight_sum_fallback"] += finite(sumw)
    for key, item in points.items():
        compatibility_key = f"mStop{item['mStop']}_mLSP{item['mLSP']}"
        xrec = (
            signal_xsec.get(key)
            or signal_xsec.get(compatibility_key)
            or signal_xsec.get(f"mStop{item['mStop']}")
            or {}
        )
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
