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
                    if not np.allclose(observed, expected, rtol=1e-7, atol=1e-6):
                        raise ValueError(f"GNN/compact mismatch {region}/{group}/{sample}/{field}: max={np.max(np.abs(observed-expected))}")
                comparisons.append(f"{region}/{group}/{sample}")
    return {"joint_projection_records": checked, "compact_projection_comparisons": len(comparisons), "frozen_GNN30": True, "SR_blinded": True}


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
