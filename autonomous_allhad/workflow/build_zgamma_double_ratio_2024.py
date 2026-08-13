#!/usr/bin/env python3
"""Build the Run-2-style Z/gamma double ratio from current 2024 histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")
CMS_LABEL = {"llabel": "Work in progress", "rlabel": "2024 (13.6 TeV)"}
BACKGROUND_SAMPLES = {"DY", "GJ", "QCD", "ST", "TT", "VV", "WtoLNu", "Zto2Nu"}
HIGH_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0])
LOW_FULL_EDGES = np.asarray(
    [0.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0]
)
LOW_EDGES = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1500.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_stream(path: Path) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            keys, value = json.loads(line)
        except Exception as error:
            raise ValueError(f"invalid stream record {path}:{line_number}") from error
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(str(key), {})
        node[str(keys[-1])] = float(value)
    return tree


def indexed_array(node: dict[str, Any], quantity: str, nbin: int) -> np.ndarray:
    values = (node or {}).get(quantity) or {}
    result = np.zeros(nbin, dtype=float)
    for index, value in values.items():
        result[int(index)] = float(value)
    return result


def exact_leaf(node: dict[str, Any], nbin: int) -> tuple[np.ndarray, np.ndarray]:
    nominal = (node or {}).get("nominal") or {}
    values = np.asarray(nominal.get("sumw") or [0.0] * nbin, dtype=float)
    variances = np.asarray(nominal.get("sumw2") or [0.0] * nbin, dtype=float)
    if len(values) != nbin or len(variances) != nbin:
        raise ValueError(f"expected {nbin} bins, found {len(values)}/{len(variances)}")
    return values, variances


def rebin(values: np.ndarray, source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Sum source bins into target bins with exactly aligned boundaries."""
    if np.array_equal(source_edges, target_edges):
        return np.asarray(values, dtype=float).copy()
    if not set(target_edges).issubset(set(source_edges)):
        raise ValueError(f"target edges {target_edges} are not aligned with {source_edges}")
    output = np.zeros(len(target_edges) - 1, dtype=float)
    for index, (low, high) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        mask = (source_edges[:-1] >= low) & (source_edges[1:] <= high)
        output[index] = float(np.sum(values[mask]))
    return output


def ratio(
    data: np.ndarray,
    data_variance: np.ndarray,
    other: np.ndarray,
    other_variance: np.ndarray,
    target: np.ndarray,
    target_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = data - other
    residual_variance = data_variance + other_variance
    value = np.full_like(residual, np.nan)
    variance = np.full_like(residual, np.nan)
    valid = (target > 0.0) & (residual >= 0.0)
    value[valid] = residual[valid] / target[valid]
    variance[valid] = (
        residual_variance[valid] / target[valid] ** 2
        + residual[valid] ** 2 * target_variance[valid] / target[valid] ** 4
    )
    return value, np.sqrt(np.maximum(variance, 0.0)), residual, residual_variance


def normalized_shape(result: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    normalization_value, normalization_stat, _, _ = ratio(
        np.asarray([np.sum(result["data"])]),
        np.asarray([np.sum(result["data_variance"])]),
        np.asarray([np.sum(result["other"])]),
        np.asarray([np.sum(result["other_variance"])]),
        np.asarray([np.sum(result["target"])]),
        np.asarray([np.sum(result["target_variance"])]),
    )
    norm = float(normalization_value[0])
    norm_stat = float(normalization_stat[0])
    residual = np.asarray(result["residual"], dtype=float)
    residual_variance = np.asarray(result["residual_variance"], dtype=float)
    target = np.asarray(result["target"], dtype=float)
    target_variance = np.asarray(result["target_variance"], dtype=float)
    residual_total = float(np.sum(residual))
    residual_variance_total = float(np.sum(residual_variance))
    target_total = float(np.sum(target))
    target_variance_total = float(np.sum(target_variance))

    value = np.full_like(residual, np.nan)
    stat = np.full_like(residual, np.nan)
    valid = (
        (residual > 0.0)
        & (target > 0.0)
        & np.isfinite(result["value"])
        & (residual_total > 0.0)
        & (target_total > 0.0)
        & np.isfinite(norm)
        & (norm > 0.0)
    )
    value[valid] = result["value"][valid] / norm

    # The inclusive normalization contains the bin being normalized.  Keep
    # that covariance instead of adding the inclusive uncertainty as if it
    # came from an independent sample.  For S_i=(A_i/B_i)/(A_tot/B_tot),
    # propagate the derivatives with respect to every independent bin of A
    # and B.  This is the Run-2 normalized-shape comparison with a correct
    # first-order statistical uncertainty.
    residual_term = (
        (1.0 / residual[valid] - 1.0 / residual_total) ** 2
        * residual_variance[valid]
        + np.maximum(residual_variance_total - residual_variance[valid], 0.0)
        / residual_total**2
    )
    target_term = (
        (-1.0 / target[valid] + 1.0 / target_total) ** 2
        * target_variance[valid]
        + np.maximum(target_variance_total - target_variance[valid], 0.0)
        / target_total**2
    )
    stat[valid] = np.abs(value[valid]) * np.sqrt(
        np.maximum(residual_term + target_term, 0.0)
    )
    return {
        "value": value,
        "stat": stat,
        "normalization": norm,
        "normalization_stat": norm_stat,
    }


def photon_ratio(exact: dict[str, Any], regime: str) -> dict[str, np.ndarray]:
    source_edges = np.asarray(exact[regime]["recoil_edges"], dtype=float)
    edges = source_edges if regime == "highdm" else LOW_EDGES
    nbin = len(source_edges) - 1
    source = exact[regime]["recoil"]["GCR"]
    totals = {
        name: np.zeros(nbin, dtype=float)
        for name in ("data", "data_variance", "target", "target_variance", "other", "other_variance")
    }
    for group in exact[regime]["nb_groups"]:
        by_sample = source[group]
        for sample, leaf in by_sample.items():
            values, variances = exact_leaf(leaf, nbin)
            if sample == "data_obs":
                totals["data"] += values
                totals["data_variance"] += variances
            elif sample == "GJ":
                totals["target"] += values
                totals["target_variance"] += variances
            elif sample in BACKGROUND_SAMPLES:
                totals["other"] += values
                totals["other_variance"] += variances
    if not np.array_equal(source_edges, edges):
        totals = {
            name: rebin(values, source_edges, edges)
            for name, values in totals.items()
        }
    value, stat, residual, residual_variance = ratio(
        totals["data"], totals["data_variance"], totals["other"],
        totals["other_variance"], totals["target"], totals["target_variance"]
    )
    return {**totals, "value": value, "stat": stat, "residual": residual,
            "residual_variance": residual_variance, "edges": edges}


def dy_ratio(stream: dict[str, Any], regime: str) -> dict[str, np.ndarray]:
    if regime == "highdm":
        channels = (("DY2E", None), ("DY2M", None))
        source_edges = HIGH_EDGES
        edges = HIGH_EDGES
    else:
        channels = (
            ("cat5_DY2E_lowDeltaM", "recoil_dy2e"),
            ("cat6_DY2M_lowDeltaM", "recoil_dy2m"),
        )
        source_edges = LOW_FULL_EDGES
        edges = LOW_EDGES
    nbin = len(source_edges) - 1
    totals = {
        name: np.zeros(nbin, dtype=float)
        for name in ("data", "data_variance", "target", "target_variance", "other", "other_variance")
    }
    top_key = "histograms" if regime == "highdm" else "lowdm_variable_histograms"
    for channel, variable in channels:
        by_sample = stream[top_key][channel]
        if variable is not None:
            by_sample = by_sample[variable]
        for sample, leaf in by_sample.items():
            nominal = leaf.get("nominal") or {}
            values = indexed_array(nominal, "sumw", nbin)
            variances = indexed_array(nominal, "sumw2", nbin)
            if sample == "data_obs":
                totals["data"] += values
                totals["data_variance"] += variances
            elif sample == "DY":
                totals["target"] += values
                totals["target_variance"] += variances
            elif sample in BACKGROUND_SAMPLES:
                totals["other"] += values
                totals["other_variance"] += variances
    if not np.array_equal(source_edges, edges):
        totals = {
            name: rebin(values, source_edges, edges)
            for name, values in totals.items()
        }
    value, stat, residual, residual_variance = ratio(
        totals["data"], totals["data_variance"], totals["other"],
        totals["other_variance"], totals["target"], totals["target_variance"]
    )
    return {**totals, "value": value, "stat": stat, "residual": residual,
            "residual_variance": residual_variance, "edges": edges}


def records(dy: dict[str, np.ndarray], photon: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if not np.array_equal(dy["edges"], photon["edges"]):
        raise ValueError("DY and photon recoil edges differ")
    dy_shape = normalized_shape(dy)
    photon_shape = normalized_shape(photon)
    value = dy_shape["value"] / photon_shape["value"]
    stat = np.abs(value) * np.sqrt(
        (dy_shape["stat"] / dy_shape["value"]) ** 2
        + (photon_shape["stat"] / photon_shape["value"]) ** 2
    )
    systematic = np.maximum(np.abs(value - 1.0), stat)
    output = []
    for index in range(len(value)):
        output.append(
            {
                "low": float(dy["edges"][index]),
                "high": float(dy["edges"][index + 1]),
                "z_data_over_mc_raw": float(dy["value"][index]),
                "photon_data_over_mc_raw": float(photon["value"][index]),
                "z_data_over_mc": float(dy_shape["value"][index]),
                "z_stat": float(dy_shape["stat"][index]),
                "photon_data_over_mc": float(photon_shape["value"][index]),
                "photon_stat": float(photon_shape["stat"][index]),
                "z_normalization": float(dy_shape["normalization"]),
                "z_normalization_stat": float(dy_shape["normalization_stat"]),
                "photon_normalization": float(photon_shape["normalization"]),
                "photon_normalization_stat": float(photon_shape["normalization_stat"]),
                "double_ratio": float(value[index]),
                "double_ratio_stat": float(stat[index]),
                "systematic": float(systematic[index]),
                "status": "complete" if np.isfinite(value[index]) else "unavailable",
            }
        )
    return output


def plot(regime: str, rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    edges = np.asarray([rows[0]["low"]] + [row["high"] for row in rows])
    x = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)
    z = np.asarray([row["z_data_over_mc"] for row in rows])
    zerr = np.asarray([row["z_stat"] for row in rows])
    gamma = np.asarray([row["photon_data_over_mc"] for row in rows])
    gamma_err = np.asarray([row["photon_stat"] for row in rows])
    double = np.asarray([row["double_ratio"] for row in rows])
    double_err = np.asarray([row["double_ratio_stat"] for row in rows])
    systematic = np.asarray([row["systematic"] for row in rows])

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.25], "hspace": 0.05},
    )
    ax.errorbar(x, z, xerr=xerr, yerr=zerr, fmt="o", ms=9, color="#ff0000",
                capsize=3, label=r"$Z(\ell\ell)$ CR")
    ax.errorbar(x, gamma, xerr=xerr, yerr=gamma_err, fmt="s", ms=8,
                color="#0000ff", capsize=3, label=r"$\gamma$ CR")
    ax.axhline(1.0, color="0.35", lw=1.5)
    ax.set_ylabel("Normalized Data/MC")
    ax.legend(fontsize=18, loc="best")
    ax.text(0.04, 0.08, r"High-$\Delta m$" if regime == "highdm" else r"Low-$\Delta m$",
            transform=ax.transAxes, fontsize=20)
    hep.cms.label(ax=ax, loc=0, **CMS_LABEL)

    band_low = 1.0 - systematic
    band_high = 1.0 + systematic
    rax.stairs(band_low, edges, baseline=band_high, fill=True, color="#66bb66",
               alpha=0.4, label="Assigned systematic")
    rax.errorbar(x, double, xerr=xerr, yerr=double_err, fmt="o", ms=9,
                 color="black", capsize=3, label=r"$Z/\gamma$ double ratio")
    rax.axhline(1.0, color="0.25", lw=1.5)
    rax.set_ylabel(r"$Z/\gamma$")
    rax.set_xlabel(r"$U_T$ (GeV)")
    rax.legend(fontsize=15, loc="best")
    for axis in (ax, rax):
        axis.set_xlim(edges[0], edges[-1])
        axis.margins(x=0)
        axis.grid(axis="y", alpha=0.22, linestyle=":")
    finite = np.concatenate((z[np.isfinite(z)], gamma[np.isfinite(gamma)]))
    if len(finite):
        ax.set_ylim(0.0, max(1.6, float(np.max(finite)) * 1.35))
    finite_double = double[np.isfinite(double)]
    finite_span = (double_err + systematic)[np.isfinite(double)]
    if len(finite_double):
        rax.set_ylim(
            max(0.0, float(np.min(finite_double - finite_span)) - 0.1),
            max(1.5, float(np.max(finite_double + finite_span)) + 0.1),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"zgamma_double_ratio_{regime}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcr-exact", type=Path, required=True)
    parser.add_argument("--dy-high-stream", type=Path, required=True)
    parser.add_argument("--dy-low-stream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exact = json.loads(args.gcr_exact.read_text())
    high_stream = read_stream(args.dy_high_stream)
    low_stream = read_stream(args.dy_low_stream)
    payload: dict[str, Any] = {
        "schema_version": "zgamma_double_ratio_2024_v1",
        "status": "complete",
        "definition": {
            "z_ratio_raw": "(DYCR data - non-DY MC) / DY MC",
            "photon_ratio_raw": "(GCR data - non-GJ MC) / GJ MC = Q * Sgamma",
            "z_shape": "z_ratio_raw / inclusive z_ratio_raw",
            "photon_shape": "photon_ratio_raw / inclusive photon_ratio_raw",
            "double_ratio": "z_shape / photon_shape",
            "systematic": "max(abs(double_ratio - 1), double_ratio_stat)",
            "category_policy": "inclusive within High-dM and Low-dM, following Run-2 AN Sec. 7.5",
            "low_dm_tail_policy": "merge raw 800-1000 and 1000-1500 GeV yields into 800-1500 GeV before ratios",
            "statistical_policy": (
                "first-order propagation including the covariance between each bin "
                "and its inclusive normalization"
            ),
        },
        "provenance": {
            "gcr_exact": str(args.gcr_exact),
            "gcr_exact_sha256": sha256(args.gcr_exact),
            "dy_high_stream": str(args.dy_high_stream),
            "dy_high_stream_sha256": sha256(args.dy_high_stream),
            "dy_low_stream": str(args.dy_low_stream),
            "dy_low_stream_sha256": sha256(args.dy_low_stream),
            "dy_channels": ["DY2E", "DY2M"],
            "dy_samples": "DYto2E/Mu/Tau-4Jets current merged DY process; PTLL excluded upstream",
        },
        "plots": [],
    }
    for regime, stream in (("highdm", high_stream), ("lowdm", low_stream)):
        photon = photon_ratio(exact, regime)
        dy = dy_ratio(stream, regime)
        current = records(dy, photon)
        payload[regime] = {"edges": dy["edges"].tolist(), "bins": current}
        payload["plots"].extend(plot(regime, current, args.output_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "zgamma_double_ratio.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), "plots": payload["plots"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
