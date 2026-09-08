"""GNN background mapping from frozen, process-separated JSON histograms.

No event inputs are supported. The reader selects small background objects
from the canonical indented JSON without materializing signal templates.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import re
from pathlib import Path
from typing import Any

import numpy as np

SAMPLES = {"DY", "GJ", "QCD", "ST", "TT", "VV", "WtoLNu", "Zto2Nu", "data_obs"}
FIELDS = ("sumw", "sumw2", "entries")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_backgrounds(path: Path) -> dict[str, Any]:
    """Read only complete background objects from canonical indent=2 JSON.

    This intentionally rejects other serialization layouts. Each selected
    object is parsed by json.loads; frozen input hashes are checked by the
    campaign runner. SR observations are never deserialized or exported.
    """
    headers = re.compile(rb'^( {2}| {4}| {6}| {8}| {10})"([^"\\\n]+)": \{', re.M)
    output: dict[str, Any] = {}
    context: dict[int, str] = {}
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[:2] != b"{\n":
            raise ValueError("GNN JSON must use the canonical indent=2 serialization")
        for match in headers.finditer(data):
            depth = len(match[1]) // 2
            context[depth] = match[2].decode()
            for level in list(context):
                if level > depth:
                    del context[level]
            if depth != 5 or context.get(1) != "histograms":
                continue
            variation, region, category, sample = (context[level] for level in (2, 3, 4, 5))
            if sample not in SAMPLES or (region == "SR" and sample == "data_obs"):
                continue
            start = match.end() - 1
            end = data.find(b"\n          }", start)
            if end < 0:
                raise ValueError(f"unclosed histogram object: {region}/{category}/{sample}")
            record = json.loads(data[start:end + len(b"\n          }")])
            selected = {key: record[key] for key in ("gnn_score", "gnn_score_ut")}
            destination = output.setdefault(variation, {}).setdefault(region, {}).setdefault(category, {})
            if sample in destination:
                raise ValueError(f"duplicate GNN histogram: {variation}/{region}/{category}/{sample}")
            destination[sample] = selected
    if "nominal" not in output or set(output["nominal"]) != {"SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M"}:
        raise ValueError("missing canonical GNN nominal regions")
    return output


def nb_group(category: str) -> str:
    group = category.split("_")[0]
    if group not in {"Nb1", "Nb2plus"}:
        raise ValueError(f"unknown Nb category {category}")
    return group


def parent(category: str) -> str:
    return nb_group(category) + ("_NISR0" if category.endswith("_NISR0") else "_NISR1plus")


def require_gnn_mapping(source: dict[str, Any]) -> dict[str, Any]:
    """Reject missing/legacy Low-dM templates before any final output is made."""
    mapping = source.get("lowdm_gnn") or {}
    if mapping.get("status") != "complete" or mapping.get("axis") != "GNN output":
        raise ValueError("Low-dM requires complete GNN templates; UT/search-bin fallback is forbidden")
    sr, cr = mapping["sr_binning"], mapping["cr_binning"]
    categories = sr["category_labels"]
    if sr["total_bins"] != 30 or sr["bins_per_category"] != 5 or len(categories) != 6:
        raise ValueError("Low-dM must preserve the frozen GNN30 layout")
    if cr["bins_per_category"] != 5 or len(cr["category_labels"]) != 4:
        raise ValueError("Low-dM must preserve the four GNN CR parents")
    nominal = mapping["transfer_factors"]["nominal"]
    for route in ("top_llcr", "w_llcr", "qcd_qcdcr", "zinv_gcr"):
        if set(nominal.get(route, {})) != set(categories):
            raise ValueError(f"missing GNN transfer templates for {route}")
        for category, record in nominal[route].items():
            if record.get("status") not in {"complete", "signed_mc_bins"}:
                raise ValueError(f"invalid GNN transfer template: {route}/{category}")
            if record["score_edges"] != sr["edges_by_category"][category] or record["cr_score_edges"] != cr["score_edges"]:
                raise ValueError("GNN template edges disagree with frozen SR/CR configuration")
            if record["parent"] != parent(category) or record["parent"] not in cr["category_labels"]:
                raise ValueError("GNN template parent mapping changed")
            for field in ("numerator", "numerator_sumw2", "transfer_factor", "mcstat", "cr_score_fraction"):
                values = np.asarray(record[field])
                if values.shape != (5,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid GNN template array: {route}/{category}/{field}")
    if set(mapping["zinv_projection"]) != set(categories):
        raise ValueError("missing RZ/Sgamma-corrected GNN Zinv templates")
    for category, record in mapping["zinv_projection"].items():
        for field in ("rz_sgamma", "rz_sgamma_mc_sumw2"):
            values = np.asarray(record[field])
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError(f"invalid GNN Zinv template: {category}/{field}")
    return mapping


def template_contract(source: dict[str, Any]) -> dict[str, Any]:
    """Describe the sole final template source, without changing fit parameters."""
    mapping = require_gnn_mapping(source)
    return {
        "axis": "GNN output", "bins": 30,
        "controlled_backgrounds": {"Top": "top_llcr", "WtoLNu": "w_llcr", "QCD": "qcd_qcdcr", "Zto2Nu": "zinv_gcr"},
        "all_process_templates": "lowdm_gnn.histograms[variation][region][category][sample].gnn_score",
        "zinv_nominal": "lowdm_gnn.zinv_projection[category].rz_sgamma",
        "minor_backgrounds": ["DY", "GJ", "VV"],
        "top_components": ["TT", "ST"],
        "factor_axes": "RZ measured in Nb; Sgamma/double ratio measured in UT and projected via actual GNN x UT components",
        "sr_to_cr_parent": {category: parent(category) for category in mapping["sr_binning"]["category_labels"]},
        "ut_fallback_allowed": False, "legacy_search_bin_fallback_allowed": False,
        "rate_parameters_changed": False,
    }


def summed(samples: dict[str, Any], names: tuple[str, ...], variable: str, field: str) -> np.ndarray:
    available = [np.asarray(samples[name][variable][field], dtype=float) for name in names if name in samples]
    if not available:
        raise ValueError(f"all required processes are absent: {names}")
    return np.sum(available, axis=0)


def validate(histograms: dict[str, Any], config: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    nominal = histograms["nominal"]
    if set(nominal["SR"]) != set(config["sr_binning"]["category_labels"]):
        raise ValueError("GNN SR categories differ from frozen config")
    for region in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M"):
        if set(nominal[region]) != set(config["cr_binning"]["category_labels"]):
            raise ValueError(f"GNN {region} categories differ from frozen config")
    sparse_records = []
    for region, categories in nominal.items():
        required = SAMPLES - {"data_obs"} if region == "SR" else SAMPLES
        for category, samples in categories.items():
            missing = required - set(samples)
            if missing:
                independent = compact.get("lowdm", {}).get("recoil", {}).get(region, {}).get(nb_group(category), {})
                if compact.get("status") != "complete" or not missing <= set(independent):
                    raise ValueError(f"missing GNN process templates without independent accounting: {region}/{category}/{sorted(missing)}")
                sparse_records.extend({"region": region, "category": category, "sample": sample}
                                      for sample in sorted(missing))
    checked = 0
    for variation, regions in histograms.items():
        for region, categories in regions.items():
            for category, samples in categories.items():
                for sample, record in samples.items():
                    for field in FIELDS:
                        joint = np.asarray(record["gnn_score_ut"][field], dtype=float)
                        projection = np.asarray(record["gnn_score"][field], dtype=float)
                        if joint.shape != (5, 8) or projection.shape != (5,):
                            raise ValueError(f"wrong GNN dimensions: {variation}/{region}/{category}/{sample}")
                        if not np.isfinite(joint).all() or not np.allclose(joint.sum(axis=1), projection, rtol=1e-10, atol=1e-8):
                            raise ValueError("joint does not reproduce GNN projection")
                        if field != "sumw" and np.any(joint < 0):
                            raise ValueError("negative entries or variance")
                    checked += 1
    # Independent products must agree after dropping only the frozen GNN axes.
    comparisons = []
    for region, categories in nominal.items():
        for group in ("Nb1", "Nb2plus"):
            for sample, source in compact["lowdm"]["recoil"][region][group].items():
                if region == "SR" and sample == "data_obs":
                    continue
                selected = [s[sample] for cat, s in categories.items() if nb_group(cat) == group and sample in s]
                for field in FIELDS:
                    expected = np.asarray(source["nominal"][field], dtype=float)
                    observed = np.sum([np.asarray(r["gnn_score_ut"][field]).sum(axis=0) for r in selected], axis=0) if selected else np.zeros(8)
                    if expected.shape != (8,) or not np.isfinite(expected).all():
                        raise ValueError(f"invalid compact projection: {region}/{group}/{sample}/{field}")
                    if field != "sumw" and np.any(expected < 0):
                        raise ValueError(f"negative compact entries or variance: {region}/{group}/{sample}/{field}")
                    # The producer omits empty categories. Exact event accounting
                    # proves their absence is zero, rather than a lost template.
                    agrees = (np.array_equal(observed, expected) if field == "entries"
                              else np.allclose(observed, expected, rtol=1e-7, atol=1e-6))
                    if not agrees:
                        raise ValueError(f"GNN/compact mismatch {region}/{group}/{sample}/{field}: max={np.max(np.abs(observed-expected))}")
                comparisons.append(f"{region}/{group}/{sample}")
    return {"joint_projection_records": checked, "compact_projection_comparisons": len(comparisons),
            "verified_sparse_zero_records": sparse_records, "frozen_GNN30": True, "SR_blinded": True}


def transfer_records(histograms: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """SR GNN-bin coefficients and CR shapes for the existing parent norm.

    SR and CR score edges differ: never divide equal bin indices. A fitted
    parent CR target yield multiplies these coefficients; CR score fractions
    are provided separately. No new rate parameters are introduced here.
    """
    paths = {"top_llcr": (("ST", "TT"), "LLCR", ("ST", "TT")),
             "w_llcr": (("WtoLNu",), "LLCR", ("WtoLNu",)),
             "qcd_qcdcr": (("QCD",), "QCDCR", ("QCD",)),
             "zinv_gcr": (("Zto2Nu",), "GCR", ("GJ",))}
    output: dict[str, Any] = {}
    for variation, source in histograms.items():
        paths_out = {}
        for key, (num_processes, region, den_processes) in paths.items():
            by_category = {}
            for category in config["sr_binning"]["category_labels"]:
                control = parent(category)
                num = summed(source["SR"][category], num_processes, "gnn_score", "sumw")
                numvar = summed(source["SR"][category], num_processes, "gnn_score", "sumw2")
                den = summed(source[region][control], den_processes, "gnn_score", "sumw")
                denvar = summed(source[region][control], den_processes, "gnn_score", "sumw2")
                total, totalvar = float(den.sum()), float(denvar.sum())
                if total <= 0:
                    by_category[category] = {"status": "invalid_denominator", "denominator_total": total}
                    continue
                cov = np.diag(numvar) / total**2 + np.outer(num, num) * totalvar / total**4
                by_category[category] = {
                    "status": "complete" if np.all(num >= 0) else "signed_mc_bins",
                    "nb_group": nb_group(category), "parent": control, "denominator_region": region,
                    "score_edges": config["sr_binning"]["edges_by_category"][category],
                    "numerator": num.tolist(), "numerator_sumw2": numvar.tolist(),
                    "denominator": [total] * 5, "denominator_sumw2": [totalvar] * 5,
                    "denominator_total": total, "denominator_total_sumw2": totalvar,
                    "transfer_factor": (num / total).tolist(), "mcstat": np.sqrt(np.diag(cov)).tolist(),
                    "valid": (num >= 0).tolist(), "covariance": cov.tolist(),
                    "cr_score_edges": config["cr_binning"]["score_edges"],
                    "cr_score_yield": den.tolist(), "cr_score_sumw2": denvar.tolist(),
                    "cr_score_fraction": (den / total).tolist(),
                    "mechanical_relative_residual": np.zeros(5).tolist(),
                }
            paths_out[key] = by_category
        output[variation] = paths_out
    return output


def project_zinv(histograms: dict[str, Any], sgamma: dict[str, Any], rz: dict[str, Any]) -> dict[str, Any]:
    """Transfer only Sgamma (not Q) using actual Zinv GNN x U_T components."""
    output = {}
    ut_to_shape = [0, 1, 2, 3, 4, 4, 4, 4]
    for category, samples in histograms["nominal"]["SR"].items():
        group = nb_group(category)
        joint = np.asarray(samples["Zto2Nu"]["gnn_score_ut"]["sumw"], dtype=float)
        jointvar = np.asarray(samples["Zto2Nu"]["gnn_score_ut"]["sumw2"], dtype=float)
        shape = np.asarray([item["Sgamma"]["value"] for item in sgamma["lowdm_families"][group]["bins"]], dtype=float)
        rz_value = float(rz["rz_low"]["combined"][group]["RZ"])
        if len(shape) != 5 or not np.isfinite(shape).all() or np.any(shape <= 0):
            raise ValueError(f"invalid Sgamma for {group}")
        components = np.stack([joint[:, np.asarray(ut_to_shape) == i].sum(axis=1) for i in range(5)], axis=1)
        componentvar = np.stack([jointvar[:, np.asarray(ut_to_shape) == i].sum(axis=1) for i in range(5)], axis=1)
        prediction = rz_value * (components @ shape)
        output[category] = {
            "nb_group": group, "parent": parent(category), "RZ": rz_value,
            "raw": joint.sum(axis=1).tolist(), "raw_sumw2": jointvar.sum(axis=1).tolist(),
            "rz_only": (rz_value * joint.sum(axis=1)).tolist(),
            "rz_sgamma": prediction.tolist(),
            "rz_sgamma_mc_sumw2": (rz_value**2 * (componentvar @ shape**2)).tolist(),
            "shape_components": components.tolist(), "shape_components_sumw2": componentvar.tolist(),
            "Sgamma": shape.tolist(),
        }
    return output


def project_double_ratio_uncertainty(mapping: dict[str, Any], factors: dict[str, Any]) -> dict[str, Any]:
    """Export the existing central-nonclosure response basis in GNN bins.

    This product does not define nuisance names, their correlations, or a
    likelihood. The main card writer retains ownership of those definitions.
    """
    rows = factors["lowdm"]["bins"]
    expected = [250.0, 300.0, 350.0, 400.0, 500.0, 1500.0]
    if factors.get("adoption_status") != "adopted" or factors["lowdm"]["edges"] != expected:
        raise ValueError("GNN uncertainty propagation requires the adopted full 250-GeV domain")
    deltas = np.asarray([row["downstream_central_abs_deviation"] for row in rows])
    if len(rows) != 5 or not np.isfinite(deltas).all() or np.any(deltas < 0):
        raise ValueError("invalid central double-ratio deviation")
    for row, delta in zip(rows, deltas):
        if row["status"] != "complete" or not np.isclose(delta, abs(row["double_ratio"] - 1), rtol=1e-12, atol=1e-15):
            raise ValueError("downstream deviation is not abs(D-1)")
    result = {}
    for category, node in mapping["zinv_projection"].items():
        components = np.asarray(node["shape_components"], dtype=float)
        component_var = np.asarray(node["shape_components_sumw2"], dtype=float)
        shape = np.asarray(node["Sgamma"], dtype=float)
        nominal = np.asarray(node["rz_sgamma"], dtype=float)
        rz_value = float(node["RZ"])
        if components.shape != (5, 5) or component_var.shape != (5, 5):
            raise ValueError("GNN x shape components must have 5 x 5 dimensions")
        weighted = rz_value * components * shape
        if not np.allclose(weighted.sum(axis=1), nominal, rtol=1e-12, atol=1e-10):
            raise ValueError("stored nominal prediction disagrees with its UT components")
        responses = []
        for index, (row, delta) in enumerate(zip(rows, deltas)):
            up_weights, down_weights = shape.copy(), shape.copy()
            up_weights[index] *= 1 + delta
            down_weights[index] /= 1 + delta
            up = nominal + weighted[:, index] * delta
            down = nominal + weighted[:, index] * (1 / (1 + delta) - 1)
            if not np.allclose(up, rz_value * (components @ up_weights), rtol=1e-12, atol=1e-10):
                raise ValueError("GNN up response failed component reconstruction")
            if not np.allclose(down, rz_value * (components @ down_weights), rtol=1e-12, atol=1e-10):
                raise ValueError("GNN down response failed component reconstruction")
            responses.append({
                "ut_bin": index, "ut_low": row["low"], "ut_high_display": row["high"],
                "open_ended": index == 4, "delta": float(delta),
                "reporting_band_not_used": row["systematic"],
                "up_multiplier": float(1 + delta), "down_multiplier": float(1 / (1 + delta)),
                "up": up.tolist(), "down": down.tolist(),
                "up_mc_sumw2": (rz_value**2 * (component_var @ up_weights**2)).tolist(),
                "down_mc_sumw2": (rz_value**2 * (component_var @ down_weights**2)).tolist(),
                "fractional_up_response": np.divide(up - nominal, nominal, out=np.zeros(5), where=nominal != 0).tolist(),
            })
        result[category] = {
            "nb_group": node["nb_group"], "parent": node["parent"],
            "score_edges": mapping["sr_binning"]["edges_by_category"][category],
            "nominal": nominal.tolist(), "nominal_sumw2": node["rz_sgamma_mc_sumw2"],
            "responses": responses,
        }
    return {"status": "complete", "schema_version": "gnn_zgamma_uncertainty_v1",
            "adoption_status": "adopted", "ut_edges": expected, "axis": "GNN output",
            "definition": "Central abs(D-1) response in each UT component; up=1+delta, down=1/(1+delta)",
            "correlation_policy": "Response basis only. No new nuisance names, parameters, or correlations; downstream card structure unchanged.",
            "nominal_changed": False, "statistical_error_used_as_nuisance": False,
            "categories": result}
