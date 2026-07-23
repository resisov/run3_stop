#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import correctionlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


COM_ENERGY_TEV = 13.6
FIGURE_SIZE_INCHES = (8.0, 8.0)
FIGURE_DPI = 180
CVMFS_BASE = Path("/cvmfs/cms-griddata.cern.ch/cat/metadata")


@dataclass(frozen=True)
class YearConfig:
    year: str
    era: str
    jec_tag: str
    jer_tag: str
    pileup_filename: str
    pileup_correction: str
    egamma_year: str
    has_electron_hlt: bool

    def payload(self, group: str, filename: str) -> Path:
        return CVMFS_BASE / group / self.era / "latest" / filename


YEAR_CONFIGS = {
    "2024": YearConfig(
        year="2024",
        era="Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
        jec_tag="Summer24Prompt24_V5",
        jer_tag="Summer24Prompt24_JRV2",
        pileup_filename="puWeights_BCDEFGHI.json.gz",
        pileup_correction="Collisions24_BCDEFGHI_goldenJSON",
        egamma_year="2024Prompt",
        has_electron_hlt=True,
    ),
    "2025": YearConfig(
        year="2025",
        era="Run3-25Prompt-Summer24-NanoAODv15",
        jec_tag="Summer24Prompt25_V3",
        jer_tag="Summer24Prompt25_JRV2",
        pileup_filename="puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
        pileup_correction="Collisions25_goldenJSON",
        egamma_year="2025Prompt",
        has_electron_hlt=False,
    ),
}


@dataclass(frozen=True)
class GridSpec:
    slug: str
    title: str
    source: str
    correction: str
    x_input: str
    y_input: str
    fixed: dict[str, Any]
    variations: tuple[tuple[str, str], ...]
    x_label: str
    y_label: str
    y_min: float | None = None
    y_max: float | None = None
    y_log: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_reference(repo: Path, source: Path) -> str:
    try:
        return str(source.relative_to(repo))
    except ValueError:
        return str(source)


def correction_payload(raw: dict[str, Any], name: str) -> dict[str, Any]:
    for correction in raw.get("corrections", []):
        if correction.get("name") == name:
            return correction
    raise KeyError(f"correction {name!r} is not present")


def collect_edges(node: Any, input_name: str, output: list[float]) -> None:
    if isinstance(node, dict):
        nodetype = node.get("nodetype")
        if nodetype == "binning" and node.get("input") == input_name:
            output.extend(float(value) for value in node.get("edges", []) if isinstance(value, (int, float)))
        elif nodetype == "multibinning" and input_name in node.get("inputs", []):
            index = node["inputs"].index(input_name)
            output.extend(float(value) for value in node.get("edges", [])[index] if isinstance(value, (int, float)))
        for value in node.values():
            collect_edges(value, input_name, output)
    elif isinstance(node, list):
        for value in node:
            collect_edges(value, input_name, output)


def input_domain(raw: dict[str, Any], correction: str, input_name: str) -> tuple[float, float]:
    edges: list[float] = []
    collect_edges(correction_payload(raw, correction).get("data"), input_name, edges)
    finite = np.asarray([value for value in edges if np.isfinite(value)], dtype=float)
    if len(finite) < 2:
        raise RuntimeError(f"could not determine {input_name} domain for {correction}")
    return float(np.min(finite)), float(np.max(finite))


def centers(edges: np.ndarray, logarithmic: bool = False) -> np.ndarray:
    if logarithmic:
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def collect_selected_edges(
    node: Any,
    input_name: str,
    fixed: dict[str, Any],
    output: list[float],
) -> None:
    if isinstance(node, list):
        for value in node:
            collect_selected_edges(value, input_name, fixed, output)
        return
    if not isinstance(node, dict):
        return

    nodetype = node.get("nodetype")
    if nodetype == "category" and node.get("input") in fixed:
        requested = fixed[node["input"]]
        for item in node.get("content", []):
            if item.get("key") == requested:
                collect_selected_edges(item.get("value"), input_name, fixed, output)
                return
        default = node.get("default")
        if default is not None:
            collect_selected_edges(default, input_name, fixed, output)
            return
        raise KeyError(f"category {node['input']} has no key {requested!r}")

    if nodetype == "binning" and node.get("input") == input_name:
        output.extend(float(value) for value in node.get("edges", []) if isinstance(value, (int, float)))
    elif nodetype == "multibinning" and input_name in node.get("inputs", []):
        index = node["inputs"].index(input_name)
        output.extend(float(value) for value in node.get("edges", [])[index] if isinstance(value, (int, float)))
    for value in node.values():
        collect_selected_edges(value, input_name, fixed, output)


def selected_input_edges(
    raw: dict[str, Any],
    correction: str,
    input_name: str,
    selections: list[dict[str, Any]],
) -> np.ndarray:
    found: list[float] = []
    data = correction_payload(raw, correction).get("data")
    for fixed in selections:
        collect_selected_edges(data, input_name, fixed, found)
    finite = sorted(set(value for value in found if np.isfinite(value)))
    if len(finite) < 2:
        raise RuntimeError(f"could not determine selected {input_name} edges for {correction}")
    return np.asarray(finite, dtype=float)


def bounded_edges(edges: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    lower = float(edges[0] if low is None else max(edges[0], low))
    upper = float(edges[-1] if high is None else high)
    if upper <= lower:
        raise ValueError(f"invalid bounded edge range: {lower} to {upper}")
    interior = edges[(edges > lower) & (edges < upper)]
    return np.concatenate(([lower], interior, [upper]))


def evaluate_curve(
    correction: correctionlib.highlevel.Correction,
    eta_input: str,
    pt_input: str,
    eta_value: float,
    pt_values: np.ndarray,
    fixed: dict[str, Any],
) -> np.ndarray:
    supplied = dict(fixed)
    supplied[eta_input] = np.full(len(pt_values), eta_value, dtype=float)
    supplied[pt_input] = pt_values
    missing = [item.name for item in correction.inputs if item.name not in supplied]
    if missing:
        raise KeyError(f"missing inputs for {correction.name}: {missing}")
    args = [supplied[item.name] for item in correction.inputs]
    return np.asarray(correction.evaluate(*args), dtype=float)


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan, math.nan
    return float(np.min(finite)), float(np.max(finite))


def cms_label(ax: plt.Axes, year: str) -> None:
    hep.cms.label(
        label="Preliminary",
        data=True,
        rlabel=f"{year} (13.6 TeV)",
        ax=ax,
        fontsize=23,
    )


def eta_range_label(spec: GridSpec, low: float, high: float) -> str:
    if spec.x_input == "abseta":
        eta_symbol = r"|\eta|"
    elif spec.slug.startswith("electron_id"):
        eta_symbol = r"\eta_{\mathrm{SC}}"
    else:
        eta_symbol = r"\eta"
    return rf"${low:g} < {eta_symbol} < {high:g}$"


def eta_plot_region(spec: GridSpec, low: float, high: float) -> Optional[str]:
    if spec.x_input == "abseta":
        return "Inclusive"

    abs_near = 0.0 if low <= 0.0 <= high else min(abs(low), abs(high))
    abs_far = max(abs(low), abs(high))
    if spec.slug.startswith(("electron_", "photon_")):
        if abs_far <= 1.444 + 1e-6:
            return "Barrel"
        if abs_near >= 1.566 - 1e-6:
            return "Endcap"
        return None
    if spec.slug.startswith("muon_"):
        return "Barrel" if abs(0.5 * (low + high)) < 1.2 else "Endcap"
    raise KeyError(f"no eta plotting region is defined for {spec.slug}")


def curve_y_limits(*curves: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([curve[np.isfinite(curve)] for curve in curves])
    low = min(float(np.min(finite)), 1.0)
    high = max(float(np.max(finite)), 1.0)
    span = max(high - low, 0.04)
    padding = 0.12 * span
    return max(0.0, low - padding), high + padding


def plain_pt_ticks(low: float, high: float) -> list[float]:
    candidates = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, 3000]
    ticks = [float(value) for value in candidates if low <= value <= high]
    if low not in ticks:
        ticks.insert(0, float(low))
    if high not in ticks:
        ticks.append(float(high))
    return ticks


def plot_sf_group(
    output_dir: Path,
    slug: str,
    pt_label: str,
    pt_edges: np.ndarray,
    curves: list[dict[str, Any]],
    year: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    vivid_colors = (
        "#e41a1c",
        "#0057ff",
        "#00a33c",
        "#ff7f00",
        "#7a1fa2",
        "#00a6d6",
        "#d0008f",
    )
    colors = [vivid_colors[index % len(vivid_colors)] for index in range(len(curves))]
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*", "d", "8")
    pt_values = centers(pt_edges, logarithmic=True)
    limit_curves: list[np.ndarray] = []
    for index, (color, curve) in enumerate(zip(colors, curves)):
        nominal = curve["nominal"]
        up = curve["up"]
        down = curve["down"]
        limit_curves.extend((nominal, up, down))
        ax.stairs(
            nominal,
            pt_edges,
            color=color,
            lw=2.5,
            ls="-",
            label="_nolegend_",
            zorder=2,
        )
        ax.errorbar(
            pt_values,
            nominal,
            yerr=np.vstack((np.abs(nominal - down), np.abs(up - nominal))),
            color=color,
            fmt=markers[index % len(markers)],
            ms=5.8,
            lw=1.3,
            capsize=4.0,
            capthick=1.3,
            alpha=0.88,
            label=curve["eta_label"],
            zorder=3,
        )
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(pt_edges[0], pt_edges[-1])
    ax.set_ylim(*curve_y_limits(*limit_curves))
    ticks = plain_pt_ticks(float(pt_edges[0]), float(pt_edges[-1]))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1000" if abs(value - 999.99) < 0.1 else f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.set_xlabel(pt_label, fontsize=22)
    ax.set_ylabel("Scale Factor", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=16, length=8, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(axis="y", color="#c7c7c7", lw=0.7, alpha=0.65)
    ax.legend(
        frameon=False,
        ncol=2 if len(curves) > 4 else 1,
        loc="best",
        fontsize=20.0,
    )
    cms_label(ax, year)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.88)
    return save_figure(fig, output_dir, slug)


def save_figure(fig: plt.Figure, output_dir: Path, slug: str) -> list[str]:
    fig.set_size_inches(*FIGURE_SIZE_INCHES, forward=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{slug}.{suffix}"
        fig.savefig(path, dpi=FIGURE_DPI if suffix == "png" else None)
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_grid_spec(repo: Path, output_dir: Path, spec: GridSpec, year: str) -> dict[str, Any]:
    source = Path(spec.source)
    if not source.is_absolute():
        source = repo / source
    raw = read_json(source)
    cset = correctionlib.CorrectionSet.from_file(str(source))
    correction = cset[spec.correction]
    variation_input = next(
        (name for name in ("ValType", "scale_factors", "systematic") if name in [item.name for item in correction.inputs]),
        None,
    )
    if variation_input is None:
        raise RuntimeError(f"no variation input found for {spec.correction}")

    eta_bounds = {
        "electron_id_veto": (-2.0, 2.0),
        "electron_id_medium": (-2.0, 2.0),
        "electron_hlt_ele30": (-2.0, 2.0),
        "photon_id_medium": (-2.0, 2.0),
        "muon_id_loose": (-2.4, 2.4),
        "muon_id_medium": (-2.4, 2.4),
        "muon_hlt_highpt": (-2.4, 2.4),
        "btag_light_medium": (0.0, 2.5),
        "btag_charm_medium": (0.0, 2.5),
        "btag_bottom_medium": (0.0, 2.5),
    }[spec.slug]
    pt_bounds = {
        "electron_id_veto": (10.0, 1000.0),
        "electron_id_medium": (10.0, 1000.0),
        "electron_hlt_ele30": (25.0, 200.0),
        "muon_id_loose": (10.0, 500.0),
        "muon_id_medium": (10.0, 500.0),
        "muon_hlt_highpt": (52.0, 500.0),
        "photon_id_medium": (20.0, 500.0),
        "btag_light_medium": (20.0, 999.99),
        "btag_charm_medium": (20.0, 999.99),
        "btag_bottom_medium": (20.0, 999.99),
    }[spec.slug]
    selections = []
    for _, variation in spec.variations:
        fixed = dict(spec.fixed)
        fixed[variation_input] = variation
        selections.append(fixed)
    eta_edges = bounded_edges(
        selected_input_edges(raw, spec.correction, spec.x_input, selections),
        eta_bounds[0],
        eta_bounds[1],
    )
    pt_edges = bounded_edges(
        selected_input_edges(raw, spec.correction, spec.y_input, selections),
        pt_bounds[0],
        pt_bounds[1],
    )
    eta_values = centers(eta_edges)
    pt_values = centers(pt_edges, logarithmic=spec.y_log)
    plot_curves_by_region: dict[str, list[dict[str, Any]]]
    if spec.x_input == "abseta":
        plot_curves_by_region = {"Inclusive": []}
    else:
        plot_curves_by_region = {"Barrel": [], "Endcap": []}
    eta_curves: list[dict[str, Any]] = []
    accumulated: dict[str, list[np.ndarray]] = {label: [] for label, _ in spec.variations}
    display_nominal: list[np.ndarray] = []
    display_up_values: list[np.ndarray] = []
    display_down_values: list[np.ndarray] = []
    display_policy = (
        "direct JSON up/down"
        if len(spec.variations) == 3
        else "correlated and uncorrelated components combined in quadrature"
    )

    for index, eta_value in enumerate(eta_values):
        curves: dict[str, np.ndarray] = {}
        for label, variation in spec.variations:
            fixed = dict(spec.fixed)
            fixed[variation_input] = variation
            curves[label] = evaluate_curve(
                correction,
                spec.x_input,
                spec.y_input,
                float(eta_value),
                pt_values,
                fixed,
            )
            accumulated[label].append(curves[label])

        nominal = curves[spec.variations[0][0]]
        if len(spec.variations) == 3:
            display_up = curves[spec.variations[1][0]]
            display_down = curves[spec.variations[2][0]]
        else:
            display_up = nominal + np.sqrt(
                np.square(curves["Correlated up"] - nominal)
                + np.square(curves["Uncorrelated up"] - nominal)
            )
            display_down = nominal - np.sqrt(
                np.square(nominal - curves["Correlated down"])
                + np.square(nominal - curves["Uncorrelated down"])
            )
        display_nominal.append(nominal)
        display_up_values.append(display_up)
        display_down_values.append(display_down)

        eta_low = float(eta_edges[index])
        eta_high = float(eta_edges[index + 1])
        label = eta_range_label(spec, eta_low, eta_high)
        plot_region = eta_plot_region(spec, eta_low, eta_high)
        if plot_region is not None:
            plot_curves_by_region[plot_region].append({
                "eta_label": label,
                "nominal": nominal,
                "up": display_up,
                "down": display_down,
            })
        eta_curves.append(
            {
                "eta_range": [eta_low, eta_high],
                "eta_label": label,
                "plot_region": plot_region,
                "pt_edges": pt_edges.tolist(),
                "nominal": nominal.tolist(),
                "display_up": display_up.tolist(),
                "display_down": display_down.tolist(),
                "variation_values": {key: value.tolist() for key, value in curves.items()},
            }
        )
    files: list[str] = []
    plot_regions: list[dict[str, Any]] = []
    for region, region_curves in plot_curves_by_region.items():
        if not region_curves:
            continue
        region_slug = spec.slug if region == "Inclusive" else f"{spec.slug}_{region.lower()}"
        region_files = plot_sf_group(
            output_dir,
            region_slug,
            spec.y_label,
            pt_edges,
            region_curves,
            year,
        )
        files.extend(region_files)
        plot_regions.append(
            {
                "name": region,
                "slug": region_slug,
                "curve_count": len(region_curves),
                "outputs": region_files,
            }
        )

    return {
        "slug": spec.slug,
        "title": spec.title,
        "source_json": source_reference(repo, source),
        "source_sha256": sha256(source),
        "correction": spec.correction,
        "inputs": [{"name": item.name, "type": item.type} for item in correction.inputs],
        "fixed_inputs": spec.fixed,
        "variation_values": dict(spec.variations),
        "display_policy": display_policy,
        "eta_edges": eta_edges.tolist(),
        "pt_edges": pt_edges.tolist(),
        "nominal_range": list(finite_range(np.concatenate(display_nominal))),
        "display_up_range": list(finite_range(np.concatenate(display_up_values))),
        "display_down_range": list(finite_range(np.concatenate(display_down_values))),
        "variation_ranges": {
            label: list(finite_range(np.concatenate(values)))
            for label, values in accumulated.items()
        },
        "eta_curves": eta_curves,
        "plot_regions": plot_regions,
        "outputs": files,
    }


def plot_pileup(repo: Path, output_dir: Path, config: YearConfig) -> dict[str, Any]:
    source = config.payload("LUM", config.pileup_filename)
    correction_name = config.pileup_correction
    raw = read_json(source)
    correction = correctionlib.CorrectionSet.from_file(str(source))[correction_name]
    low, high = input_domain(raw, correction_name, "NumTrueInteractions")
    x = np.linspace(low + 1e-5, high - 1e-5, 600)
    values = {
        label: np.asarray(correction.evaluate(x, variation), dtype=float)
        for label, variation in (("Nominal", "nominal"), ("Up", "up"), ("Down", "down"))
    }
    sample_x = np.asarray([10.0, 30.0, 50.0, 70.0, 90.0])
    sample_x = sample_x[(sample_x > low) & (sample_x < high)]
    sample_values = {
        label: np.asarray(correction.evaluate(sample_x, variation), dtype=float)
        for label, variation in (("Nominal", "nominal"), ("Up", "up"), ("Down", "down"))
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    ax.plot(x, values["Nominal"], color="#0057ff", lw=2.2, label="Nominal")
    ax.plot(x, values["Up"], color="#ff1f1f", lw=1.6, label="Up")
    ax.plot(x, values["Down"], color="#00a33c", lw=1.6, ls="-", label="Down")
    ax.scatter(sample_x, sample_values["Nominal"], color="#0057ff", s=22, zorder=4)
    for index, x_value in enumerate(sample_x):
        nominal_value = sample_values["Nominal"][index]
        up_value = sample_values["Up"][index]
        down_value = sample_values["Down"][index]
        label = f"{nominal_value:.3g}" + chr(10) + f"{up_value:.3g} / {down_value:.3g}"
        ax.annotate(
            label,
            (x_value, sample_values["Nominal"][index]),
            xytext=(0, 13 if index % 2 == 0 else -27),
            textcoords="offset points",
            ha="center",
            va="bottom" if index % 2 == 0 else "top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
    ax.axhline(1.0, color="black", lw=0.9, ls=":")
    ax.set_xlabel("Number of true interactions", fontsize=22)
    ax.set_ylabel("Pileup weight", fontsize=22)
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.tick_params(axis="both", which="major", labelsize=16, length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.legend(frameon=False, ncol=3, loc="best", fontsize=20)
    cms_label(ax, config.year)
    fig.tight_layout()
    files = save_figure(fig, output_dir, "pileup")
    return {
        "slug": "pileup",
        "title": "Pileup SF",
        "source_json": source_reference(repo, source),
        "source_sha256": sha256(source),
        "correction": correction_name,
        "inputs": [{"name": item.name, "type": item.type} for item in correction.inputs],
        "x_domain": [low, high],
        "annotation_points": sample_x.tolist(),
        "variation_ranges": {key: list(finite_range(value)) for key, value in values.items()},
        "outputs": files,
    }


def plot_jer(repo: Path, output_dir: Path, radius: str, config: YearConfig) -> dict[str, Any]:
    token = "AK4PFPuppi" if radius == "AK4" else "AK8PFPuppi"
    filename = "jet_jerc.json.gz" if radius == "AK4" else "fatJet_jerc.json.gz"
    source = config.payload("JME", filename)
    nominal_name = f"{config.jer_tag}_MC_ScaleFactor_{token}"
    uncertainty_name = f"{config.jer_tag}_MC_SFUncertainty_{token}"
    raw = read_json(source)
    cset = correctionlib.CorrectionSet.from_file(str(source))
    nominal_corr = cset[nominal_name]
    uncertainty_corr = cset[uncertainty_name]
    eta_edges = np.asarray([-5.191, -3.0, -2.5, -1.5, 0.0, 1.5, 2.5, 3.0, 5.191])
    pt_edges = np.asarray([15.0, 30.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0])
    eta_values = centers(eta_edges)
    pt_values = centers(pt_edges, logarithmic=True)
    nominal_values: list[np.ndarray] = []
    uncertainty_values: list[np.ndarray] = []
    up_values: list[np.ndarray] = []
    down_values: list[np.ndarray] = []
    plot_curves_by_region: dict[str, list[dict[str, Any]]] = {
        "Central": [],
        "Forward": [],
    }
    eta_curves: list[dict[str, Any]] = []
    for index, eta_value in enumerate(eta_values):
        nominal = evaluate_curve(nominal_corr, "JetEta", "JetPt", float(eta_value), pt_values, {})
        uncertainty = evaluate_curve(uncertainty_corr, "JetEta", "JetPt", float(eta_value), pt_values, {})
        up = nominal + uncertainty
        down = nominal - uncertainty
        nominal_values.append(nominal)
        uncertainty_values.append(uncertainty)
        up_values.append(up)
        down_values.append(down)
        eta_low = float(eta_edges[index])
        eta_high = float(eta_edges[index + 1])
        label = rf"${eta_low:g} < \eta < {eta_high:g}$"
        plot_region = "Central" if abs(float(eta_value)) < 1.5 else "Forward"
        plot_curves_by_region[plot_region].append({
                "eta_label": label,
                "nominal": nominal,
                "up": up,
                "down": down,
            })
        eta_curves.append(
            {
                "eta_range": [eta_low, eta_high],
                "eta_label": label,
                "plot_region": plot_region,
                "pt_edges": pt_edges.tolist(),
                "nominal": nominal.tolist(),
                "uncertainty": uncertainty.tolist(),
                "display_up": up.tolist(),
                "display_down": down.tolist(),
            }
        )
    files: list[str] = []
    plot_regions: list[dict[str, Any]] = []
    for region, region_curves in plot_curves_by_region.items():
        region_slug = f"jer_{radius.lower()}_{region.lower()}"
        region_files = plot_sf_group(
            output_dir,
            region_slug,
            r"Jet $p_{\mathrm{T}}$ (GeV)",
            pt_edges,
            region_curves,
            config.year,
        )
        files.extend(region_files)
        plot_regions.append(
            {
                "name": region,
                "slug": region_slug,
                "curve_count": len(region_curves),
                "outputs": region_files,
            }
        )

    return {
        "slug": f"jer_{radius.lower()}",
        "title": f"{radius} JER SF",
        "source_json": source_reference(repo, source),
        "source_sha256": sha256(source),
        "corrections": [nominal_name, uncertainty_name],
        "eta_edges": eta_edges.tolist(),
        "pt_edges": pt_edges.tolist(),
        "nominal_range": list(finite_range(np.concatenate(nominal_values))),
        "uncertainty_range": list(finite_range(np.concatenate(uncertainty_values))),
        "up_range": list(finite_range(np.concatenate(up_values))),
        "down_range": list(finite_range(np.concatenate(down_values))),
        "eta_curves": eta_curves,
        "plot_regions": plot_regions,
        "outputs": files,
    }


def grid_specs(config: YearConfig) -> list[GridSpec]:
    electron = str(config.payload("EGM", "electron.json.gz"))
    photon = str(config.payload("EGM", "photon.json.gz"))
    muon = str(config.payload("MUO", "muon_Z.json.gz"))
    btag = str(config.payload("BTV", "btagging.json.gz"))
    specs = [
        GridSpec("electron_id_veto", "Electron Veto ID SF", electron, "Electron-ID-SF", "eta", "pt", {"year": config.egamma_year, "WorkingPoint": "Veto"}, (("Nominal", "sf"), ("Up", "sfup"), ("Down", "sfdown")), r"Electron $\eta_{\mathrm{SC}}$", r"Electron $p_{\mathrm{T}}$ (GeV)", 10.0, None),
        GridSpec("electron_id_medium", "Electron Medium ID SF", electron, "Electron-ID-SF", "eta", "pt", {"year": config.egamma_year, "WorkingPoint": "Medium"}, (("Nominal", "sf"), ("Up", "sfup"), ("Down", "sfdown")), r"Electron $\eta_{\mathrm{SC}}$", r"Electron $p_{\mathrm{T}}$ (GeV)", 10.0, None),
        GridSpec("muon_id_loose", "Muon Loose ID+miniIso SF", muon, "NUM_LooseMiniIso_DEN_LooseID", "eta", "pt", {}, (("Nominal", "nominal"), ("Up", "systup"), ("Down", "systdown")), r"Muon $\eta$", r"Muon $p_{\mathrm{T}}$ (GeV)", 10.0, None),
        GridSpec("muon_id_medium", "Muon Medium ID+miniIso SF", muon, "NUM_LooseMiniIso_DEN_MediumID", "eta", "pt", {}, (("Nominal", "nominal"), ("Up", "systup"), ("Down", "systdown")), r"Muon $\eta$", r"Muon $p_{\mathrm{T}}$ (GeV)", 10.0, None),
        GridSpec("muon_hlt_highpt", "Muon HLT SF", muon, "NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose", "eta", "pt", {}, (("Nominal", "nominal"), ("Up", "systup"), ("Down", "systdown")), r"Muon $\eta$", r"Muon $p_{\mathrm{T}}$ (GeV)", 52.0, None),
        GridSpec("photon_id_medium", "Photon Medium ID SF", photon, "Photon-ID-SF", "eta", "pt", {"year": config.egamma_year, "WorkingPoint": "Medium"}, (("Nominal", "sf"), ("Up", "sfup"), ("Down", "sfdown")), r"Photon $\eta$", r"Photon $p_{\mathrm{T}}$ (GeV)", 20.0, None),
        GridSpec("btag_light_medium", "UParT b-tag SF (light, M)", btag, "UParTAK4_light", "abseta", "pt", {"working_point": "M", "flavor": 0}, (("Nominal", "central"), ("Correlated up", "up_correlated"), ("Correlated down", "down_correlated"), ("Uncorrelated up", "up_uncorrelated"), ("Uncorrelated down", "down_uncorrelated")), r"Jet $|\eta|$", r"Jet $p_{\mathrm{T}}$ (GeV)", 20.0, 999.99),
        GridSpec("btag_charm_medium", "UParT b-tag SF (c, M)", btag, "UParTAK4_comb", "abseta", "pt", {"working_point": "M", "flavor": 4}, (("Nominal", "central"), ("Correlated up", "up_correlated"), ("Correlated down", "down_correlated"), ("Uncorrelated up", "up_uncorrelated"), ("Uncorrelated down", "down_uncorrelated")), r"Jet $|\eta|$", r"Jet $p_{\mathrm{T}}$ (GeV)", 20.0, 999.99),
        GridSpec("btag_bottom_medium", "UParT b-tag SF (b, M)", btag, "UParTAK4_comb", "abseta", "pt", {"working_point": "M", "flavor": 5}, (("Nominal", "central"), ("Correlated up", "up_correlated"), ("Correlated down", "down_correlated"), ("Uncorrelated up", "up_uncorrelated"), ("Uncorrelated down", "down_uncorrelated")), r"Jet $|\eta|$", r"Jet $p_{\mathrm{T}}$ (GeV)", 20.0, 999.99),
    ]
    if config.has_electron_hlt:
        specs.insert(
            2,
            GridSpec("electron_hlt_ele30", "Electron HLT SF", str(config.payload("EGM", "electronHlt.json.gz")), "Electron-HLT-SF", "eta", "pt", {"year": config.egamma_year, "Path": "HLT_SF_Ele30_TightID"}, (("Nominal", "sf"), ("Up", "sfup"), ("Down", "sfdown")), r"Electron $\eta$", r"Electron $p_{\mathrm{T}}$ (GeV)", 25.0, None),
        )
    return specs


def write_gallery(output_dir: Path, entries: list[dict[str, Any]], config: YearConfig) -> None:
    sections = []
    for entry in entries:
        plot_regions = entry.get("plot_regions") or [
            {"name": None, "slug": entry["slug"], "outputs": entry["outputs"]}
        ]
        articles = []
        for region in plot_regions:
            png = next(path for path in region["outputs"] if path.endswith(".png"))
            pdf = next(path for path in region["outputs"] if path.endswith(".pdf"))
            region_name = region.get("name")
            heading = (
                f'<h3>{html.escape(region_name)}</h3>'
                if region_name not in (None, "Inclusive")
                else ""
            )
            alt = entry["title"] if region_name is None else f'{entry["title"]} {region_name}'
            articles.append(
                f'<article>{heading}<a href="{html.escape(pdf)}"><img src="{html.escape(png)}" '
                f'loading="lazy" alt="{html.escape(alt)}"></a>'
                f'<p><a href="{html.escape(pdf)}">PDF</a></p></article>'
            )
        sections.append(
            f'<section id="{html.escape(entry["slug"])}"><h2>{html.escape(entry["title"])}</h2>'
            f'<div class="plots">{"".join(articles)}</div></section>'
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{config.year} scale-factor plots</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f5f5;color:#111}}header{{background:#fff;border-bottom:1px solid #ddd;padding:1.4rem 2rem}}main{{max-width:1800px;margin:0 auto;padding:1.5rem 2rem}}
section{{min-width:0;margin-bottom:2rem}}.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.2rem}}article{{background:white;padding:.8rem;border:1px solid #ddd;border-radius:6px}}img{{display:block;width:100%;height:auto}}h1{{margin:0 0 .2rem}}h2{{font-size:1.15rem}}h3{{margin:.2rem 0 .7rem}}p{{margin:.6rem 0 0}}
@media(max-width:520px){{header,main{{padding:1rem}}.plots{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>CMS Run-3 {config.year} scale factors</h1><div>{html.escape(config.era)}</div></header><main>{"".join(sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot JSON-backed Run-3 scale factors from the official correction-era payloads")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--year", choices=sorted(YEAR_CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = YEAR_CONFIGS[args.year]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for stale in output_dir.glob(pattern):
            stale.unlink()
    hep.style.use("CMS")

    entries = [plot_pileup(repo, output_dir, config)]
    for spec in grid_specs(config):
        entries.append(plot_grid_spec(repo, output_dir, spec, config.year))
    entries.append(plot_jer(repo, output_dir, "AK4", config))
    entries.append(plot_jer(repo, output_dir, "AK8", config))
    figure_count = sum(
        1
        for entry in entries
        for output in entry["outputs"]
        if output.endswith(".png")
    )

    excluded = [
        {"name": "JEC", "reason": "kinematic calibration, not a scale factor"},
        {"name": "MET XY", "reason": "kinematic correction, not a scale factor"},
        {"name": "top-pT reweight", "reason": "analytic event weight, not JSON-backed"},
        {"name": "MET trigger efficiency ratio", "reason": "no standalone correction-era JSON is applied"},
        {"name": "photon trigger efficiency ratio", "reason": "no standalone correction-era JSON is applied"},
        {"name": "top-tag SF", "reason": "no Run-3 data/MC SF JSON and decorrelation prescription have been supplied"},
    ]
    if not config.has_electron_hlt:
        excluded.append({
            "name": "electron HLT SF",
            "reason": f"electronHlt.json.gz is not published in {config.era}",
        })

    manifest = {
        "schema_version": "official_scale_factor_plots_v6",
        "created_at": utc_now(),
        "status": "complete",
        "correction_year": config.year,
        "correction_era": config.era,
        "center_of_mass_energy_tev": COM_ENERGY_TEV,
        "scope": f"Scale factors versus pT from official {config.year} correction-era JSON payloads",
        "plot_policy": "8x8-inch pT plots split into detector regions; eta ranges are overlaid as solid vivid-color curves",
        "plots": entries,
        "excluded": excluded,
        "output_count": {"png": figure_count, "pdf": figure_count},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_gallery(output_dir, entries, config)
    print(json.dumps({"status": "complete", "year": config.year, "plots": figure_count, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
