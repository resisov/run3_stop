from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from coffea.util import load

from ..config.defaults import DEFAULTS
from ..paths import PathKind, PathPolicy, canonicalize, configure_eos_runtime_env
from .reference import EXPECTED_REGIONS, validate_scaled_reference


def _region_projection(hist_obj, region):
    try:
        return hist_obj[{"region": region}]
    except Exception:
        return None


def _nominal_projection(hist_obj):
    try:
        if "systematic" in [axis.name for axis in hist_obj.axes]:
            return hist_obj[{"systematic": "nominal"}]
    except Exception:
        pass
    return hist_obj


def _bin_edges(hist_obj, variable):
    try:
        return [float(x) for x in hist_obj.axes[variable].edges]
    except Exception:
        return []


def _values_variances(hist_obj):
    values = np.asarray(hist_obj.values(flow=False), dtype=float)
    variances = np.asarray(hist_obj.variances(flow=False), dtype=float)
    return values, variances


def _summarize_hist(hist_obj, variable, region):
    reg = _region_projection(hist_obj, region)
    if reg is None:
        return None
    reg = _nominal_projection(reg)
    values, variances = _values_variances(reg)
    return {
        "bin_edges": _bin_edges(reg, variable),
        "values": values.tolist(),
        "variances": variances.tolist(),
        "total_yield": float(np.sum(values)),
        "total_variance": float(np.sum(variances)),
    }


def _compare_arrays(fast, ref):
    f = np.asarray(fast, dtype=float)
    r = np.asarray(ref, dtype=float)
    if f.shape != r.shape:
        return {"shape_mismatch": [list(f.shape), list(r.shape)]}
    diff = f - r
    rel = np.divide(diff, r, out=np.full_like(diff, np.nan, dtype=float), where=(r != 0))
    return {
        "absolute_difference": diff.tolist(),
        "relative_difference": rel.tolist(),
        "max_abs_difference": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
        "max_abs_relative_difference": float(np.nanmax(np.abs(rel))) if rel.size else 0.0,
    }


def compare_scaled_files(fast_output, reference_scaled=None, variable="recoilpt", output_json=None, dry_run=False, region=None):
    policy = PathPolicy.default()
    reference_scaled = Path(reference_scaled or DEFAULTS.legacy_scaled_reference)
    regions = [region] if region else list(EXPECTED_REGIONS)
    ref_validation = validate_scaled_reference(reference_scaled, variable, region)
    result = {
        "status": "blocked",
        "variable": variable,
        "regions": regions,
        "systematic": "nominal",
        "reference_validation": ref_validation,
        "fast_output": str(fast_output),
        "reference_scaled": str(reference_scaled),
        "comparisons": [],
        "unmatched_processes": [],
        "missing_regions": ref_validation.get("missing_regions", []),
    }
    if not ref_validation.get("ok"):
        result["blocked_reason"] = "legacy reference validation failed for requested scope"
        return result
    fast_path = Path(fast_output)
    if not fast_path.is_file():
        result["blocked_reason"] = "fast output does not exist: %s" % fast_output
        return result
    try:
        ref_payload = load(str(reference_scaled))
        fast_payload = load(str(fast_path))
    except Exception as exc:
        result["blocked_reason"] = "load failed: %s: %s" % (type(exc).__name__, exc)
        return result
    result["status"] = "ok"
    for group_name in ("bkg", "data", "sig"):
        ref_group = ref_payload.get(group_name, {}).get(variable, {})
        fast_group = fast_payload.get(group_name, {}).get(variable, {}) if isinstance(fast_payload, dict) else {}
        for process in sorted(set(ref_group.keys()).intersection(set(fast_group.keys()))):
            ref_hist = ref_group[process]
            fast_hist = fast_group[process]
            for one_region in regions:
                ref_summary = _summarize_hist(ref_hist, variable, one_region)
                fast_summary = _summarize_hist(fast_hist, variable, one_region)
                if ref_summary is None or fast_summary is None:
                    continue
                result["comparisons"].append({
                    "group": group_name,
                    "process": str(process),
                    "region": one_region,
                    "systematic": "nominal",
                    "bin_edges_reference": ref_summary["bin_edges"],
                    "bin_edges_fast": fast_summary["bin_edges"],
                    "total_yield_reference": ref_summary["total_yield"],
                    "total_yield_fast": fast_summary["total_yield"],
                    "total_variance_reference": ref_summary["total_variance"],
                    "total_variance_fast": fast_summary["total_variance"],
                    "yield_difference": fast_summary["total_yield"] - ref_summary["total_yield"],
                    "variance_difference": fast_summary["total_variance"] - ref_summary["total_variance"],
                    "per_bin_yield": _compare_arrays(fast_summary["values"], ref_summary["values"]),
                    "per_bin_variance": _compare_arrays(fast_summary["variances"], ref_summary["variances"]),
                })
        result["unmatched_processes"].append({
            "group": group_name,
            "reference_only": sorted(str(p) for p in set(ref_group.keys()) - set(fast_group.keys())),
            "fast_only": sorted(str(p) for p in set(fast_group.keys()) - set(ref_group.keys())),
        })
    if output_json is None:
        suffix = "%s_%s" % (variable, region) if region else variable
        output_json = DEFAULTS.output_root / "validation" / ("comparison_%s.json" % suffix)
    output_path = policy.resolve(output_json, PathKind.OUTPUT)
    result["output_json"] = canonicalize(str(output_path))
    if not dry_run:
        configure_eos_runtime_env(DEFAULTS.output_root, dry_run=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
        tmp.replace(output_path)
    return result


def compare_scaled_files_json(fast_output, reference_scaled=None, variable="recoilpt", output_json=None, dry_run=False, region=None):
    return json.dumps(compare_scaled_files(fast_output, reference_scaled, variable, output_json, dry_run, region), indent=2, sort_keys=True)
