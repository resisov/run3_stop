from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from coffea.util import load

from ..config.defaults import DEFAULTS
from ..paths import canonicalize

EXPECTED_REGIONS = DEFAULTS.target_regions
PROCESS_HINTS = {
    "tt": ("TT", "tt"),
    "single_top": ("ST", "single", "Single"),
    "wjets": ("W (lnu)", "W+", "W"),
    "z_to_neutrinos": ("Z (inv)", "Nu", "nunu"),
    "qcd": ("QCD Multijet", "QCD"),
    "dy": ("DY",),
    "gamma_jets": ("Gamma + Jets", "GJ", "Gamma", "gamma"),
    "diboson": ("VV", "WW", "WZ", "ZZ"),
}
REQUIRED_DATA_HINTS = ("JetMET",)


def _axis_names(hist_obj):
    try:
        return [axis.name for axis in hist_obj.axes]
    except Exception:
        return []


def _axis_values(hist_obj, axis_name):
    try:
        return [str(x) for x in hist_obj.axes[axis_name]]
    except Exception:
        return []


def _edges(hist_obj, axis_name):
    try:
        axis = hist_obj.axes[axis_name]
        return [float(x) for x in axis.edges]
    except Exception:
        return []


def _project(hist_obj, selectors):
    try:
        return hist_obj[selectors]
    except Exception:
        return None


def _nominal_exists(hist_obj):
    names = _axis_names(hist_obj)
    if "systematic" not in names:
        return False
    return "nominal" in _axis_values(hist_obj, "systematic")


def _region_exists(hist_obj, region):
    names = _axis_names(hist_obj)
    if "region" not in names:
        return False
    return region in _axis_values(hist_obj, "region")


def _regular_status(hist_obj, variable):
    try:
        values = np.asarray(hist_obj.values(flow=False), dtype=float)
        variances = np.asarray(hist_obj.variances(flow=False), dtype=float)
    except Exception as exc:
        return {"readable": False, "values_finite": False, "variances_finite": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    edges = _edges(hist_obj, variable)
    edges_array = np.asarray(edges, dtype=float) if edges else np.asarray([], dtype=float)
    finite_edges = bool(edges and np.all(np.isfinite(edges_array)))
    ordered_edges = bool(edges and np.all(np.diff(edges_array) > 0))
    return {
        "readable": True,
        "values_finite": bool(np.all(np.isfinite(values))),
        "variances_finite": bool(np.all(np.isfinite(variances))),
        "bin_edges_finite": finite_edges,
        "bin_edges_ordered": ordered_edges,
        "bin_edges": edges,
        "total_yield": float(np.sum(values)) if values.size else 0.0,
        "total_variance": float(np.sum(variances)) if variances.size else 0.0,
        "nonfinite_value_indices": np.argwhere(~np.isfinite(values))[:10].tolist(),
        "nonfinite_variance_indices": np.argwhere(~np.isfinite(variances))[:10].tolist(),
    }


def _flow_status(hist_obj):
    try:
        values = np.asarray(hist_obj.values(flow=True), dtype=float)
        variances = np.asarray(hist_obj.variances(flow=True), dtype=float)
    except Exception as exc:
        return {"readable": False, "values_finite": False, "variances_finite": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return {
        "readable": True,
        "values_finite": bool(np.all(np.isfinite(values))),
        "variances_finite": bool(np.all(np.isfinite(variances))),
        "nonfinite_value_indices": np.argwhere(~np.isfinite(values))[:10].tolist(),
        "nonfinite_variance_indices": np.argwhere(~np.isfinite(variances))[:10].tolist(),
    }


def _finite_hist(hist_obj):
    try:
        values = np.asarray(hist_obj.values(flow=True), dtype=float)
        variances = np.asarray(hist_obj.variances(flow=True), dtype=float)
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)
    bad_messages = []
    if not np.all(np.isfinite(values)):
        bad_messages.append("non-finite values at flow indices %s" % np.argwhere(~np.isfinite(values))[:10].tolist())
    if variances is not None and not np.all(np.isfinite(variances)):
        bad_messages.append("non-finite variances at flow indices %s" % np.argwhere(~np.isfinite(variances))[:10].tolist())
    if bad_messages:
        return False, "; ".join(bad_messages)
    return True, None


def _match_keys(keys, hints):
    matched = []
    for key in keys:
        if any(hint in key for hint in hints):
            matched.append(key)
    return sorted(matched)


def _scan_invalid_elsewhere(payload, variable, selected_region):
    invalid = []
    excluded = set()
    for top_name in ("bkg", "data", "sig"):
        group = payload.get(top_name, {})
        if not isinstance(group, dict) or variable not in group or not isinstance(group[variable], dict):
            continue
        for proc, hist_obj in group[variable].items():
            for region in _axis_values(hist_obj, "region"):
                if selected_region and region == selected_region:
                    continue
                if selected_region:
                    excluded.add(region)
                scoped = _project(hist_obj, {"region": region, "systematic": "nominal"})
                if scoped is None:
                    continue
                flow = _flow_status(scoped)
                regular = _regular_status(scoped, variable)
                if not flow.get("values_finite") or not flow.get("variances_finite") or not regular.get("values_finite") or not regular.get("variances_finite"):
                    invalid.append({
                        "group": top_name,
                        "process": str(proc),
                        "region": region,
                        "regular_values_finite": regular.get("values_finite"),
                        "regular_variances_finite": regular.get("variances_finite"),
                        "flow_values_finite": flow.get("values_finite"),
                        "flow_variances_finite": flow.get("variances_finite"),
                        "regular_nonfinite_value_indices": regular.get("nonfinite_value_indices", []),
                        "regular_nonfinite_variance_indices": regular.get("nonfinite_variance_indices", []),
                        "flow_nonfinite_value_indices": flow.get("nonfinite_value_indices", []),
                        "flow_nonfinite_variance_indices": flow.get("nonfinite_variance_indices", []),
                    })
    return sorted(excluded), invalid


def validate_scaled_reference(path=None, variable="recoilpt", region=None):
    reference = Path(path or DEFAULTS.legacy_scaled_reference)
    resolved = reference.expanduser().resolve()
    result = {
        "path": str(reference),
        "resolved_path": canonicalize(str(resolved)),
        "exists": resolved.is_file(),
        "readable": False,
        "loadable": False,
        "ok": False,
        "validation_scope": "region-scoped" if region else "global",
        "requested_region": region,
        "requested_variable": variable,
        "requested_systematic": "nominal",
        "global_classification": "unknown",
        "region_scoped_classification": None,
        "top_level_keys": [],
        "background_keys": [],
        "data_keys": [],
        "signal_keys": [],
        "regions": [],
        "variables": [],
        "missing_regions": [],
        "excluded_regions": [],
        "invalid_bins_outside_scope": [],
        "process_coverage": {},
        "checked_processes": {},
        "process_checks": [],
        "cat2_data_key_used": None,
        "bin_edges": [],
        "errors": [],
    }
    if not result["exists"]:
        result["errors"].append("reference file does not exist")
        result["global_classification"] = "invalid"
        return result
    result["readable"] = True
    try:
        payload = load(str(resolved))
        result["loadable"] = True
    except Exception as exc:
        result["errors"].append("coffea load failed: %s: %s" % (type(exc).__name__, exc))
        result["global_classification"] = "invalid"
        return result
    if not isinstance(payload, dict):
        result["errors"].append("top-level object is not a dict: %s" % type(payload).__name__)
        result["global_classification"] = "invalid"
        return result
    result["top_level_keys"] = sorted(str(k) for k in payload.keys())
    for top_name, out_key in (("bkg", "background_keys"), ("data", "data_keys"), ("sig", "signal_keys")):
        group = payload.get(top_name, {})
        if isinstance(group, dict) and variable in group and isinstance(group[variable], dict):
            result[out_key] = sorted(str(k) for k in group[variable].keys())
    variables = set()
    regions = set()
    finite_errors = []
    for top_name in ("bkg", "data", "sig"):
        group = payload.get(top_name, {})
        if not isinstance(group, dict):
            continue
        variables.update(str(k) for k in group.keys())
        if variable in group and isinstance(group[variable], dict):
            for proc, hist_obj in group[variable].items():
                regions.update(_axis_values(hist_obj, "region"))
                ok, err = _finite_hist(hist_obj)
                if not ok:
                    finite_errors.append("%s/%s/%s: %s" % (top_name, variable, proc, err))
    result["variables"] = sorted(variables)
    result["regions"] = sorted(regions)
    global_missing_regions = [item for item in EXPECTED_REGIONS if item not in regions]
    for label, hints in PROCESS_HINTS.items():
        result["process_coverage"][label] = _match_keys(result["background_keys"], hints)
    missing_major = [label for label, keys in result["process_coverage"].items() if not keys]
    result["global_classification"] = "valid" if (variable in variables and not global_missing_regions and not missing_major and not finite_errors) else "partially valid"

    if variable not in variables:
        result["errors"].append("missing variable %s" % variable)
        result["global_classification"] = "invalid"
        return result

    if not region:
        result["missing_regions"] = global_missing_regions
        if global_missing_regions:
            result["errors"].append("missing regions: %s" % ", ".join(global_missing_regions))
        if missing_major:
            result["errors"].append("missing major process coverage: %s" % ", ".join(missing_major))
        if finite_errors:
            result["errors"].extend(finite_errors[:20])
        result["ok"] = result["global_classification"] == "valid"
        return result

    result["excluded_regions"], result["invalid_bins_outside_scope"] = _scan_invalid_elsewhere(payload, variable, region)
    if region not in regions:
        result["missing_regions"] = [region]
        result["errors"].append("requested region is missing: %s" % region)
        result["region_scoped_classification"] = "invalid for %s" % region
        return result

    bkg_group = payload.get("bkg", {}).get(variable, {})
    data_group = payload.get("data", {}).get(variable, {})
    required_bkg = {}
    for label, hints in PROCESS_HINTS.items():
        matches = _match_keys([str(k) for k in bkg_group.keys()], hints)
        if not matches:
            result["errors"].append("missing required process for %s" % label)
        required_bkg[label] = matches[0] if matches else None
    data_matches = _match_keys([str(k) for k in data_group.keys()], REQUIRED_DATA_HINTS)
    if not data_matches:
        result["errors"].append("missing required JetMET data process")
    result["cat2_data_key_used"] = data_matches[0] if data_matches else None
    result["checked_processes"] = {"backgrounds": required_bkg, "data": result["cat2_data_key_used"]}

    process_items = []
    for label, key in required_bkg.items():
        if key is not None:
            process_items.append(("bkg", label, key, bkg_group.get(key)))
    if result["cat2_data_key_used"]:
        process_items.append(("data", "jetmet", result["cat2_data_key_used"], data_group.get(result["cat2_data_key_used"])))

    for group_name, label, key, hist_obj in process_items:
        check = {"group": group_name, "label": label, "process": key, "region_exists": False, "nominal_systematic_exists": False}
        if hist_obj is None:
            check["error"] = "histogram missing"
            result["errors"].append("missing histogram for %s/%s" % (group_name, key))
            result["process_checks"].append(check)
            continue
        check["region_exists"] = _region_exists(hist_obj, region)
        check["nominal_systematic_exists"] = _nominal_exists(hist_obj)
        if not check["region_exists"]:
            result["errors"].append("region %s missing for %s" % (region, key))
        if not check["nominal_systematic_exists"]:
            result["errors"].append("nominal systematic missing for %s" % key)
        scoped = _project(hist_obj, {"region": region, "systematic": "nominal"})
        if scoped is None:
            result["errors"].append("could not project %s to %s/nominal" % (key, region))
            check["regular"] = {"readable": False}
            check["flow"] = {"readable": False}
        else:
            check["regular"] = _regular_status(scoped, variable)
            check["flow"] = _flow_status(scoped)
            if not result["bin_edges"] and check["regular"].get("bin_edges"):
                result["bin_edges"] = check["regular"]["bin_edges"]
            regular = check["regular"]
            for field in ("readable", "values_finite", "variances_finite", "bin_edges_finite", "bin_edges_ordered"):
                if not regular.get(field):
                    result["errors"].append("%s failed regular-bin check %s" % (key, field))
        result["process_checks"].append(check)

    result["ok"] = not result["errors"]
    result["region_scoped_classification"] = "valid for %s" % region if result["ok"] else "invalid for %s" % region
    return result


def validate_scaled_reference_json(path=None, variable="recoilpt", region=None):
    return json.dumps(validate_scaled_reference(path, variable, region), indent=2, sort_keys=True)
