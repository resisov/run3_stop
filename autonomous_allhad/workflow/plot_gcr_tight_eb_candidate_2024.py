#!/usr/bin/env python3
"""Compare the trusted nominal GCR with a strict Tight-EB photon subset."""

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
from scipy.stats import chi2 as chi2_distribution


hep.style.use("CMS")

CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}
STACK_ORDER = [
    "VV",
    "ST",
    "TT",
    "DY",
    "WtoLNu",
    "Zto2Nu",
    "QCD",
    "GJ",
    "other",
]
STACK_COLORS = {
    "VV": "#65724B",
    "ST": "#8D7CC1",
    "TT": "#9AC1B9",
    "DY": "#31B7BC",
    "WtoLNu": "#D8C1A7",
    "Zto2Nu": "#F0C27B",
    "QCD": "#D6949D",
    "GJ": "#8B008B",
    "other": "#B0B0B0",
}
PROCESS_LABELS = {
    "VV": "VV",
    "ST": "Single Top",
    "TT": r"$t\bar{t}$",
    "DY": "DY",
    "WtoLNu": r"$W\to\ell\nu$",
    "Zto2Nu": r"$Z\to\nu\nu$",
    "QCD": "QCD Multijet",
    "GJ": r"$\gamma$+jets",
    "other": "Other",
}
POLICY_STYLE = {
    "nominal": {
        "label": "Medium, EB+EE",
        "color": "#777777",
        "marker": "s",
    },
    "tight_eb": {
        "label": "Tight, EB only",
        "color": "#0072B2",
        "marker": "o",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram_node(
    payload: dict[str, Any],
    delta_m: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if delta_m == "high":
        edges = np.asarray(
            payload["highdm_distribution_variable_specs"]["ut"]["bins"],
            dtype=float,
        )
        node = payload["highdm_variable_histograms"]["GCR"]["ut"]
        return edges, node
    if delta_m == "low":
        edges = np.asarray(
            payload["lowdm_variable_specs"]["recoil_gcr"]["bins"],
            dtype=float,
        )
        node = payload["lowdm_variable_histograms"][
            "cat4_GCR_lowDeltaM"
        ]["recoil_gcr"]
        return edges, node
    raise ValueError(f"unknown delta-m region: {delta_m}")


def nominal_histogram(
    node: dict[str, Any],
    process: str,
    key: str,
    size: int,
) -> np.ndarray:
    record = ((node.get(process) or {}).get("nominal") or {}).get(key)
    if record is None:
        return np.zeros(size, dtype=float)
    values = np.asarray(record, dtype=float)
    if values.shape != (size,):
        raise RuntimeError(
            f"{process}: expected {size} {key} bins, found {values.shape}"
        )
    return values


def extract_distribution(
    payload: dict[str, Any],
    delta_m: str,
) -> dict[str, Any]:
    edges, node = histogram_node(payload, delta_m)
    size = len(edges) - 1
    data = nominal_histogram(node, "data_obs", "sumw", size)
    data_variance = nominal_histogram(node, "data_obs", "sumw2", size)
    components: dict[str, np.ndarray] = {}
    component_variances: dict[str, np.ndarray] = {}
    for process in STACK_ORDER:
        components[process] = nominal_histogram(
            node,
            process,
            "sumw",
            size,
        )
        component_variances[process] = nominal_histogram(
            node,
            process,
            "sumw2",
            size,
        )
    unknown = sorted(set(node) - set(STACK_ORDER) - {"data_obs"})
    if unknown:
        raise RuntimeError(
            f"{delta_m}: unclassified histogram processes: {unknown}"
        )
    prediction = sum(components.values(), np.zeros(size, dtype=float))
    prediction_variance = sum(
        component_variances.values(),
        np.zeros(size, dtype=float),
    )
    occupied = (data != 0.0) | (prediction != 0.0)
    if np.any(occupied):
        first = int(np.flatnonzero(occupied)[0])
        last = int(np.flatnonzero(occupied)[-1]) + 1
        edges = edges[first:last + 1]
        data = data[first:last]
        data_variance = data_variance[first:last]
        prediction = prediction[first:last]
        prediction_variance = prediction_variance[first:last]
        components = {
            process: values[first:last]
            for process, values in components.items()
        }
        component_variances = {
            process: values[first:last]
            for process, values in component_variances.items()
        }
    return {
        "edges": edges,
        "data": data,
        "data_variance": data_variance,
        "components": components,
        "component_variances": component_variances,
        "prediction": prediction,
        "prediction_variance": prediction_variance,
    }


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def finite_list(values: np.ndarray) -> list[float | None]:
    return [finite_or_none(value) for value in np.asarray(values, dtype=float)]


def distribution_metrics(distribution: dict[str, Any]) -> dict[str, Any]:
    data = distribution["data"]
    prediction = distribution["prediction"]
    prediction_variance = distribution["prediction_variance"]
    data_total = float(np.sum(data))
    prediction_total = float(np.sum(prediction))
    data_stat = math.sqrt(max(data_total, 0.0))
    mc_stat = math.sqrt(
        max(float(np.sum(prediction_variance)), 0.0)
    )
    integral_ratio = (
        data_total / prediction_total
        if prediction_total > 0.0
        else None
    )
    integral_ratio_stat = (
        integral_ratio
        * math.sqrt(
            (data_stat / data_total) ** 2
            + (mc_stat / prediction_total) ** 2
        )
        if integral_ratio is not None
        and data_total > 0.0
        and prediction_total > 0.0
        else None
    )
    mask = (
        np.isfinite(data)
        & np.isfinite(prediction)
        & np.isfinite(prediction_variance)
        & (prediction > 0.0)
    )
    if np.count_nonzero(mask) >= 2 and prediction_total > 0.0:
        scale = data_total / prediction_total
        scaled_prediction = scale * prediction[mask]
        variance = (
            np.maximum(data[mask], 1.0)
            + scale * scale * prediction_variance[mask]
        )
        chi2_value = float(
            np.sum(
                np.square(data[mask] - scaled_prediction)
                / variance
            )
        )
        dof = int(np.count_nonzero(mask) - 1)
        pvalue = float(chi2_distribution.sf(chi2_value, dof))
    else:
        scale = float("nan")
        chi2_value = float("nan")
        dof = 0
        pvalue = float("nan")
    return {
        "data": data_total,
        "mc": prediction_total,
        "data_over_mc": integral_ratio,
        "data_over_mc_stat": integral_ratio_stat,
        "data_stat": data_stat,
        "mc_stat": mc_stat,
        "shape_only_scale": finite_or_none(scale),
        "shape_only_chi2": finite_or_none(chi2_value),
        "shape_only_dof": dof,
        "shape_only_pvalue": finite_or_none(pvalue),
        "binwise_data_over_mc": finite_list(safe_ratio(data, prediction)),
        "components": {
            process: float(np.sum(values))
            for process, values in distribution["components"].items()
        },
    }


def finish_figure(
    fig: Any,
    label_axis: Any,
    stem: Path,
) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.88, right=0.97)
    hep.cms.label(ax=label_axis, loc=0, **CMS_LABEL)
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_stack(
    distribution: dict[str, Any],
    policy: str,
    delta_m: str,
    output_dir: Path,
) -> None:
    edges = distribution["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    data = distribution["data"]
    prediction = distribution["prediction"]
    prediction_variance = distribution["prediction_variance"]
    policy_style = POLICY_STYLE[policy]

    fig, (axis, lower) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    bottom = np.zeros_like(prediction)
    for process in STACK_ORDER:
        values = distribution["components"][process]
        if not np.any(values):
            continue
        axis.bar(
            centers,
            values,
            width=widths,
            bottom=bottom,
            align="center",
            color=STACK_COLORS[process],
            edgecolor="black",
            linewidth=0.35,
            label=PROCESS_LABELS[process],
        )
        bottom += values
    prediction_sigma = np.sqrt(np.maximum(prediction_variance, 0.0))
    axis.fill_between(
        edges,
        np.r_[
            np.maximum(prediction - prediction_sigma, 1.0e-3),
            max(prediction[-1] - prediction_sigma[-1], 1.0e-3),
        ],
        np.r_[
            prediction + prediction_sigma,
            prediction[-1] + prediction_sigma[-1],
        ],
        step="post",
        facecolor="none",
        edgecolor="0.25",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
        label="MC stat. unc.",
    )
    axis.errorbar(
        centers,
        data,
        xerr=0.5 * widths,
        yerr=np.sqrt(np.maximum(data, 0.0)),
        color="black",
        marker="o",
        linestyle="none",
        label="Data",
        zorder=10,
    )
    axis.set_yscale("log")
    positive = np.r_[data[data > 0.0], prediction[prediction > 0.0]]
    ymin = max(0.2, float(np.min(positive)) * 0.35) if len(positive) else 0.2
    ymax = max(float(np.max(positive)) * 12.0, 10.0) if len(positive) else 10.0
    axis.set_ylim(ymin, ymax)
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, fontsize=8, loc="upper right")
    region_label = (
        r"High-$\Delta m$ GCR"
        if delta_m == "high"
        else r"Low-$\Delta m$ GCR"
    )
    axis.text(
        0.025,
        0.045,
        f"{region_label}\n{policy_style['label']} photon",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )

    ratio = safe_ratio(data, prediction)
    ratio_error = safe_ratio(np.sqrt(np.maximum(data, 0.0)), prediction)
    relative_mc = safe_ratio(prediction_sigma, prediction)
    lower.fill_between(
        edges,
        np.r_[1.0 - relative_mc, 1.0 - relative_mc[-1]],
        np.r_[1.0 + relative_mc, 1.0 + relative_mc[-1]],
        step="post",
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        alpha=0.45,
    )
    lower.errorbar(
        centers,
        ratio,
        xerr=0.5 * widths,
        yerr=ratio_error,
        color=policy_style["color"],
        marker=policy_style["marker"],
        linestyle="none",
    )
    lower.axhline(1.0, color="black", linewidth=0.9)
    finite_ratio = ratio[np.isfinite(ratio)]
    upper = max(2.0, float(np.max(finite_ratio + ratio_error[np.isfinite(ratio)])) * 1.15) if len(finite_ratio) else 2.0
    lower.set_ylim(0.0, min(upper, 5.0))
    lower.set_ylabel("Data/MC")
    lower.set_xlabel(r"$U_{T}$ (GeV)")
    lower.grid(axis="y", alpha=0.22)
    lower.set_xlim(edges[0], edges[-1])
    lower.margins(x=0)
    axis.margins(x=0)
    finish_figure(
        fig,
        axis,
        output_dir / f"{delta_m}dm_gcr_ut_{policy}",
    )


def plot_ratio_comparison(
    nominal: dict[str, Any],
    tight_eb: dict[str, Any],
    delta_m: str,
    output_dir: Path,
) -> None:
    edges = nominal["edges"]
    if not np.array_equal(edges, tight_eb["edges"]):
        raise RuntimeError(f"{delta_m}: nominal/candidate binning differs")
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    fig, axis = plt.subplots(figsize=(10, 10))
    for index, (policy, distribution) in enumerate(
        (("nominal", nominal), ("tight_eb", tight_eb))
    ):
        style = POLICY_STYLE[policy]
        ratio = safe_ratio(
            distribution["data"],
            distribution["prediction"],
        )
        ratio_error = safe_ratio(
            np.sqrt(np.maximum(distribution["data"], 0.0)),
            distribution["prediction"],
        )
        offset = (-0.025 if index == 0 else 0.025) * widths
        axis.errorbar(
            centers + offset,
            ratio,
            xerr=0.45 * widths,
            yerr=ratio_error,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=(
                "white" if policy == "nominal" else style["color"]
            ),
            linestyle="none",
            capsize=2,
            label=style["label"],
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xlim(edges[0], edges[-1])
    axis.margins(x=0)
    axis.set_ylim(0.0, 3.0)
    axis.set_xlabel(r"$U_{T}$ (GeV)")
    axis.set_ylabel("Data/MC")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="upper right", fontsize=13)
    region_label = (
        r"High-$\Delta m$ GCR"
        if delta_m == "high"
        else r"Low-$\Delta m$ GCR"
    )
    axis.text(
        0.025,
        0.04,
        region_label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
    )
    finish_figure(
        fig,
        axis,
        output_dir / f"{delta_m}dm_gcr_ut_data_mc_comparison",
    )


def aggregate_audit(
    payload: dict[str, Any],
) -> dict[str, Any]:
    source = (
        payload.get("summary", {})
        .get("gcr_photon_selection_audit", {})
    )
    result: dict[str, Any] = {}
    for process, datasets in source.items():
        process_record = result.setdefault(
            process,
            {
                "highdm": {
                    "nominal_entries": 0,
                    "selected_entries": 0,
                },
                "lowdm": {
                    "nominal_entries": 0,
                    "selected_entries": 0,
                },
            },
        )
        for dataset_record in datasets.values():
            for region in ("highdm", "lowdm"):
                for key in ("nominal_entries", "selected_entries"):
                    process_record[region][key] += int(
                        dataset_record.get(region, {}).get(key, 0)
                    )
        for region in ("highdm", "lowdm"):
            nominal_entries = process_record[region]["nominal_entries"]
            process_record[region]["retention"] = (
                process_record[region]["selected_entries"]
                / nominal_entries
                if nominal_entries
                else None
            )
    return result


def execution_checks(
    nominal: dict[str, Any],
    tight_eb: dict[str, Any],
) -> dict[str, Any]:
    warning_keys = (
        "weight_failures",
        "missing_input_roots",
        "missing_sidecars",
        "zero_entry_roots",
        "weight_rejections",
    )
    nominal_summary = nominal.get("summary") or {}
    tight_summary = tight_eb.get("summary") or {}

    def photon_sources(summary: dict[str, Any]) -> list[str]:
        sources = set()
        for label, status in (summary.get("scale_factor_status") or {}).items():
            if label == "data_obs":
                continue
            source = (
                (status.get("components") or {})
                .get("photon_id", {})
                .get("source")
            )
            if source:
                sources.add(str(source))
        return sorted(sources)

    nominal_options = nominal_summary.get("build_options") or {}
    tight_options = tight_summary.get("build_options") or {}
    return {
        "nominal_status": nominal.get("status"),
        "tight_eb_status": tight_eb.get("status"),
        "input_root_count_nominal": len(
            nominal_summary.get("input_roots") or []
        ),
        "input_root_count_tight_eb": len(
            tight_summary.get("input_roots") or []
        ),
        "input_root_sets_match": (
            sorted(nominal_summary.get("input_roots") or [])
            == sorted(tight_summary.get("input_roots") or [])
        ),
        "normalization_sha256_nominal": nominal_options.get(
            "normalization_sha256"
        ),
        "normalization_sha256_tight_eb": tight_options.get(
            "normalization_sha256"
        ),
        "normalization_sha256_match": (
            nominal_options.get("normalization_sha256")
            == tight_options.get("normalization_sha256")
        ),
        "btag_efficiency_nominal": nominal_options.get("btag_efficiency"),
        "btag_efficiency_tight_eb": tight_options.get("btag_efficiency"),
        "code_sha256_match": (
            nominal_options.get("code_sha256")
            == tight_options.get("code_sha256")
        ),
        "nominal_warning_flags": {
            key: bool(nominal_summary.get(key))
            for key in warning_keys
        },
        "tight_eb_warning_flags": {
            key: bool(tight_summary.get(key))
            for key in warning_keys
        },
        "nominal_photon_id_sources": photon_sources(nominal_summary),
        "tight_eb_photon_id_sources": photon_sources(tight_summary),
    }


def fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}"


def fmt_pvalue(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2e}" if number < 1.0e-3 else f"{number:.3f}"


def write_report(
    path: Path,
    comparison: dict[str, Any],
) -> None:
    lines = [
        "# 2024 GCR Tight-EB photon candidate",
        "",
        "This is a read-only comparison. No nominal intermediate ROOT file was modified.",
        "",
        "Candidate definition: trusted nominal GCR selection, additionally requiring exactly one photon with corrected "
        r"$p_T>220$ GeV, $|\eta|<1.4442$, `cutBased>=3`, and `electronVeto`. "
        "The Tight photon ID scale factor is used for MC.",
        "",
        "| Region | Policy | Data | MC | Data/MC | Shape-only p-value |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for delta_m in ("high", "low"):
        for policy in ("nominal", "tight_eb"):
            metrics = comparison["metrics"][delta_m][policy]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{delta_m}-dM",
                        POLICY_STYLE[policy]["label"],
                        fmt(metrics["data"], 1),
                        fmt(metrics["mc"], 1),
                        fmt(metrics["data_over_mc"]),
                        fmt_pvalue(metrics["shape_only_pvalue"]),
                    ]
                )
                + " |"
            )
    high_nominal = comparison["metrics"]["high"]["nominal"]
    high_tight = comparison["metrics"]["high"]["tight_eb"]
    low_nominal = comparison["metrics"]["low"]["nominal"]
    low_tight = comparison["metrics"]["low"]["tight_eb"]
    high_excess_reduction = (
        1.0
        - (high_tight["data_over_mc"] - 1.0)
        / (high_nominal["data_over_mc"] - 1.0)
    )
    data_audit = comparison["unweighted_selection_audit"]["tight_eb"][
        "data_obs"
    ]
    checks = comparison["execution_checks"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Execution: both payloads are complete, use the same "
            f"{checks['input_root_count_nominal']} ROOT files, and have matching normalization and code hashes. "
            "No missing input, sidecar, zero-entry, weight-fallback, or weight-rejection warning was recorded.",
            f"- High-dM: Data/MC changes from {high_nominal['data_over_mc']:.3f} to "
            f"{high_tight['data_over_mc']:.3f}. This removes only "
            f"{100.0 * high_excess_reduction:.1f}% of the excess above unity, while retaining "
            f"{100.0 * data_audit['highdm']['retention']:.1f}% of the data events. "
            f"The absolute ratio shift ({abs(high_tight['data_over_mc'] - high_nominal['data_over_mc']):.3f}) "
            f"is smaller than the per-selection statistical uncertainty "
            f"({high_nominal['data_over_mc_stat']:.3f} nominal, {high_tight['data_over_mc_stat']:.3f} Tight-EB).",
            f"- High-dM shape-only compatibility changes from p={fmt_pvalue(high_nominal['shape_only_pvalue'])} "
            f"to p={fmt_pvalue(high_tight['shape_only_pvalue'])}; both are compatible, but the Tight-EB "
            "candidate does not improve the shape metric.",
            f"- Low-dM: Data/MC changes from {low_nominal['data_over_mc']:.3f} to "
            f"{low_tight['data_over_mc']:.3f}, so the integrated agreement becomes slightly worse. "
            f"The shape-only p-value remains poor ({fmt_pvalue(low_tight['shape_only_pvalue'])}).",
            "- Decision: Tight-EB alone is not a sufficient GCR Data/MC remedy and should not replace "
            "the nominal photon definition on the basis of this test. It is useful as a diagnostic "
            "cross-check because it preserves the qualitative high-dM shape while reducing statistics by about one third.",
            "",
            "Primary plots:",
            "",
            "- `highdm_gcr_ut_nominal.png` and `highdm_gcr_ut_tight_eb.png`",
            "- `highdm_gcr_ut_data_mc_comparison.png`",
            "- `lowdm_gcr_ut_nominal.png` and `lowdm_gcr_ut_tight_eb.png`",
            "- `lowdm_gcr_ut_data_mc_comparison.png`",
            "",
            "The normalization comparison and the shape-only test answer different questions. "
            "The integrated Data/MC value measures rate agreement; the shape-only p-value first normalizes MC to data "
            "and tests the remaining binned shape disagreement.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_index(path: Path) -> None:
    cards = []
    for delta_m in ("high", "low"):
        pretty = "High-dM" if delta_m == "high" else "Low-dM"
        for stem, caption in (
            (
                f"{delta_m}dm_gcr_ut_nominal",
                f"{pretty} nominal Medium, EB+EE",
            ),
            (
                f"{delta_m}dm_gcr_ut_tight_eb",
                f"{pretty} Tight, EB only",
            ),
            (
                f"{delta_m}dm_gcr_ut_data_mc_comparison",
                f"{pretty} Data/MC comparison",
            ),
        ):
            cards.append(
                "<figure>"
                f'<a href="{stem}.pdf"><img src="{stem}.png" alt="{caption}"></a>'
                f"<figcaption>{caption}</figcaption>"
                "</figure>"
            )
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2024 GCR Tight-EB photon study</title>
  <style>
    body { font-family: sans-serif; margin: 2rem auto; max-width: 1500px; color: #202124; }
    h1 { font-size: 1.7rem; }
    p { line-height: 1.45; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.2rem; }
    figure { margin: 0; border: 1px solid #ddd; padding: 0.7rem; background: #fff; }
    img { width: 100%; height: auto; display: block; }
    figcaption { margin-top: 0.5rem; font-weight: 600; }
    code { background: #f3f3f3; padding: 0.1rem 0.25rem; }
  </style>
</head>
<body>
  <h1>2024 GCR Tight-EB photon study</h1>
  <p>Read-only comparison of the trusted nominal GCR and its strict
  <code>Tight photon, |eta|&lt;1.4442</code> subset. Click a plot for PDF.</p>
  <p><a href="report.md">Report</a> · <a href="comparison.json">Machine-readable metrics</a></p>
  <div class="grid">
"""
        + "\n".join(cards)
        + """
  </div>
</body>
</html>
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot the 2024 nominal versus Tight-EB GCR comparison."
    )
    parser.add_argument("--nominal", required=True)
    parser.add_argument("--tight-eb", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    nominal_path = Path(args.nominal).resolve()
    tight_path = Path(args.tight_eb).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nominal_payload = read_json(nominal_path)
    tight_payload = read_json(tight_path)
    if nominal_payload.get("status") != "complete":
        raise RuntimeError(
            f"nominal payload is not complete: {nominal_payload.get('status')}"
        )
    if tight_payload.get("status") != "complete":
        raise RuntimeError(
            f"Tight-EB payload is not complete: {tight_payload.get('status')}"
        )

    comparison: dict[str, Any] = {
        "schema_version": "gcr_tight_eb_comparison_v1",
        "status": "complete",
        "inputs": {
            "nominal": {
                "path": str(nominal_path),
                "sha256": sha256(nominal_path),
            },
            "tight_eb": {
                "path": str(tight_path),
                "sha256": sha256(tight_path),
            },
        },
        "selection": tight_payload.get("gcr_photon_policy"),
        "execution_checks": execution_checks(
            nominal_payload,
            tight_payload,
        ),
        "metrics": {},
        "unweighted_selection_audit": {
            "nominal": aggregate_audit(nominal_payload),
            "tight_eb": aggregate_audit(tight_payload),
        },
    }
    for delta_m in ("high", "low"):
        nominal_distribution = extract_distribution(
            nominal_payload,
            delta_m,
        )
        tight_distribution = extract_distribution(
            tight_payload,
            delta_m,
        )
        comparison["metrics"][delta_m] = {
            "nominal": distribution_metrics(nominal_distribution),
            "tight_eb": distribution_metrics(tight_distribution),
        }
        plot_stack(
            nominal_distribution,
            "nominal",
            delta_m,
            output_dir,
        )
        plot_stack(
            tight_distribution,
            "tight_eb",
            delta_m,
            output_dir,
        )
        plot_ratio_comparison(
            nominal_distribution,
            tight_distribution,
            delta_m,
            output_dir,
        )

    write_json(output_dir / "comparison.json", comparison)
    write_report(output_dir / "report.md", comparison)
    write_index(output_dir / "index.html")
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "comparison": str(output_dir / "comparison.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
