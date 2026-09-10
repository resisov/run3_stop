#!/usr/bin/env python3
"""Reproduce and contextualize the higgsino-stop relic-density contours.

The relic contours are the numerical data behind Fig. 4.2 of
arXiv:2512.12457 / Phys. Rev. D 113, 095029 (2026).  The optional collider
overlay is the current 2024+2025 expected T2tt contour and is intentionally
labelled as a reference: the paper benchmark has a higgsino triplet and mixed
stop branching fractions, rather than the 100% stop -> top neutralino decay
assumed by T2tt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RELIC_LEVELS = (0.0045, 0.01, 0.03, 0.06, 0.09, 0.12)
TOP_MASS_GEV = 172.5
W_PLUS_B_MASS_GEV = 80.4 + 4.8
PAPER_DOI = "https://doi.org/10.1103/m7qf-j8j6"
DATA_DOI = "https://doi.org/10.5281/zenodo.19666737"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the six Figure5_Omegah2_*.dat files.",
    )
    parser.add_argument(
        "--expected-limits",
        type=Path,
        help="Optional current T2tt expected_limits.json for a reference overlay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Optional in-conversation HTML-fragment output.",
    )
    return parser.parse_args()


def level_token(level: float) -> str:
    return format(level, "g")


def read_relic_curve(path: Path) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 2:
            raise RuntimeError(f"unexpected row in {path}: {raw_line!r}")
        points.append([float(columns[0]), float(columns[1])])
    if len(points) < 2:
        raise RuntimeError(f"relic contour has too few points: {path}")
    return points


def load_relic_curves(source_dir: Path) -> dict[str, list[list[float]]]:
    curves: dict[str, list[list[float]]] = {}
    for level in RELIC_LEVELS:
        token = level_token(level)
        path = source_dir / f"Figure5_Omegah2_{token}.dat"
        if not path.is_file():
            raise FileNotFoundError(path)
        curves[token] = read_relic_curve(path)
    return curves


def signed_two_body_phase_space(
    parent_mass: np.ndarray,
    visible_mass: float,
    invisible_mass: np.ndarray,
) -> np.ndarray:
    parent = np.asarray(parent_mass, dtype=float)
    invisible = np.asarray(invisible_mass, dtype=float)
    parent, invisible = np.broadcast_arrays(parent, invisible)
    coordinate = np.full(parent.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(parent)
        & np.isfinite(invisible)
        & (parent > 0.0)
        & (invisible >= 0.0)
    )
    first = parent[valid] ** 2 - (visible_mass + invisible[valid]) ** 2
    second = parent[valid] ** 2 - (visible_mass - invisible[valid]) ** 2
    kallen = first * second
    coordinate[valid] = np.sign(kallen) * np.sqrt(np.abs(kallen)) / parent[valid] ** 2
    return coordinate


def _has_two_dimensional_support(first: np.ndarray, second: np.ndarray) -> bool:
    if first.size < 3:
        return False
    coordinates = np.column_stack((first, second))
    return bool(
        np.linalg.matrix_rank(coordinates - np.mean(coordinates, axis=0)) >= 2
    )


def extract_t2tt_expected_contours(
    expected_limits_path: Path,
    *,
    x_domain: tuple[float, float] = (600.0, 1800.0),
    y_domain: tuple[float, float] = (0.0, 1300.0),
) -> list[list[list[float]]]:
    """Match the canonical topology-aware T2tt interpolation and extract r=1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    payload = json.loads(expected_limits_path.read_text())
    records = [
        record
        for record in (payload.get("points") or {}).values()
        if float(record.get("expected", 0.0)) > 0.0
    ]
    xs = np.asarray([float(record["mStop"]) for record in records])
    ys = np.asarray([float(record["mLSP"]) for record in records])
    zs = np.log10(
        np.asarray([float(record["expected"]) for record in records])
    )

    input_valid = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(zs)
        & ((xs - ys) > W_PLUS_B_MASS_GEV)
    )
    input_first = xs[input_valid]
    input_second = signed_two_body_phase_space(
        xs[input_valid], TOP_MASS_GEV, ys[input_valid]
    )
    input_z = zs[input_valid]
    if not _has_two_dimensional_support(input_first, input_second):
        raise RuntimeError("T2tt expected-limit grid lacks 2D support")

    xi = np.linspace(*x_domain, 260)
    yi = np.linspace(*y_domain, 260)
    xx, yy = np.meshgrid(xi, yi)

    # Match the lower-boundary anchoring used by the canonical limit plotter.
    minimum_lsp = float(np.min(ys[input_valid]))
    lower_boundary = input_valid & np.isclose(ys, minimum_lsp)
    boundary_x = xs[lower_boundary]
    boundary_z = zs[lower_boundary]
    order = np.argsort(boundary_x)
    boundary_x = boundary_x[order]
    boundary_z = boundary_z[order]
    boundary_x, unique_indices = np.unique(boundary_x, return_index=True)
    boundary_z = boundary_z[unique_indices]
    if boundary_x.size >= 2:
        dense_x = xi[(xi >= boundary_x[0]) & (xi <= boundary_x[-1])]
        dense_y = np.full_like(dense_x, minimum_lsp)
        dense_z = np.interp(dense_x, boundary_x, boundary_z)
        dense_valid = (dense_x - dense_y) > W_PLUS_B_MASS_GEV
        input_first = np.concatenate((input_first, dense_x[dense_valid]))
        input_second = np.concatenate(
            (
                input_second,
                signed_two_body_phase_space(
                    dense_x[dense_valid], TOP_MASS_GEV, dense_y[dense_valid]
                ),
            )
        )
        input_z = np.concatenate((input_z, dense_z[dense_valid]))

    target_second = signed_two_body_phase_space(xx, TOP_MASS_GEV, yy)
    linear = griddata(
        (input_first, input_second),
        input_z,
        (xx, target_second),
        method="linear",
        rescale=True,
    )
    target_valid = (
        np.isfinite(linear)
        & np.isfinite(target_second)
        & ((xx - yy) > W_PLUS_B_MASS_GEV)
    )
    grid = np.ma.array(linear, mask=~target_valid)

    figure, axis = plt.subplots()
    contour_set = axis.contour(xx, yy, grid, levels=[0.0])
    paths: list[list[list[float]]] = []
    for contour_path in contour_set.get_paths():
        vertices = contour_path.vertices
        codes = contour_path.codes
        if codes is None:
            if len(vertices) >= 2:
                paths.append(vertices[:, :2].round(4).tolist())
            continue
        starts = np.flatnonzero(codes == 1)
        for index, start in enumerate(starts):
            stop = int(starts[index + 1]) if index + 1 < len(starts) else len(vertices)
            segment = vertices[start:stop, :2]
            if len(segment) >= 2:
                paths.append(segment.round(4).tolist())
    plt.close(figure)
    return paths


def write_machine_data(
    output_path: Path,
    curves: dict[str, list[list[float]]],
    collider_paths: list[list[list[float]]],
    expected_limits_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "higgsino_stop_relic_contours_v1",
        "axes": {
            "x": "m_stop1_GeV",
            "y": "m_neutralino1_GeV",
        },
        "model": {
            "mu_sign": "negative",
            "tan_beta": 1.6,
            "gaugino_mass_parameters_GeV": 10000.0,
            "other_scalar_masses_except_light_higgs_GeV": 10000.0,
            "light_higgs_mass_GeV": 125.1,
            "relic_calculator": "micrOMEGAs 6.0",
            "spectrum_and_couplings": "SuSpect 2.41 (one loop)",
            "cosmology": "standard thermal freeze-out",
        },
        "sources": {
            "paper": PAPER_DOI,
            "data": DATA_DOI,
            "paper_figure": "Fig. 4.2",
            "zenodo_directory": "Figure5",
        },
        "relic_contours": [
            {"omega_h2": float(level), "points": curves[level]}
            for level in map(level_token, RELIC_LEVELS)
        ],
        "t2tt_expected_reference": {
            "present": bool(collider_paths),
            "interpretation": (
                "reference only; T2tt assumes BR(stop->top neutralino1)=100%, "
                "unlike the higgsino-triplet benchmark"
            ),
            "source": str(expected_limits_path) if expected_limits_path else None,
            "paths": collider_paths,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def plot_static(
    payload: dict[str, Any], output_png: Path, output_pdf: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(RELIC_LEVELS)))
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.20,
        top=0.84,
        wspace=0.17,
    )
    panel_settings = (
        ("Published contour plane", (0.0, 7000.0), (100.0, 1500.0)),
        ("Run-3 stop-search window", (600.0, 1800.0), (0.0, 1300.0)),
    )

    relic_entries = payload["relic_contours"]
    for axis, (title, xlim, ylim) in zip(axes, panel_settings):
        for color, entry in zip(colors, relic_entries):
            points = np.asarray(entry["points"], dtype=float)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=2.2,
                label=rf"$\Omega_{{\mathrm{{LSP}}}}h^2={entry['omega_h2']:g}$",
                zorder=4,
            )
        diagonal_x = np.linspace(xlim[0], xlim[1], 400)
        axis.plot(
            diagonal_x,
            diagonal_x,
            color="0.55",
            linewidth=1.2,
            linestyle=":",
            label=r"$m_{\tilde t_1}=m_{\tilde\chi_1^0}$",
            zorder=2,
        )
        axis.plot(
            diagonal_x,
            diagonal_x - TOP_MASS_GEV,
            color="0.35",
            linewidth=1.4,
            linestyle="--",
            label=r"$m_{\tilde t_1}=m_{\tilde\chi_1^0}+m_t$",
            zorder=2,
        )
        if title.startswith("Run-3"):
            for collider_path in payload["t2tt_expected_reference"]["paths"]:
                points = np.asarray(collider_path, dtype=float)
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    color="#d62728",
                    linewidth=3.0,
                    zorder=6,
                )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_title(title, loc="left", fontweight="normal", pad=8)
        axis.set_xlabel(r"$m_{\tilde t_1}$ [GeV]")
        axis.set_ylabel(r"$m_{\tilde\chi_1^0}$ [GeV]")
        axis.grid(color="0.88", linewidth=0.6, zorder=0)
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].xaxis.set_major_locator(MultipleLocator(1000))
    axes[0].yaxis.set_major_locator(MultipleLocator(200))
    axes[1].xaxis.set_major_locator(MultipleLocator(200))
    axes[1].yaxis.set_major_locator(MultipleLocator(200))

    handles = [
        Line2D([0], [0], color=color, lw=2.2, label=rf"$\Omega h^2={level:g}$")
        for color, level in zip(colors, RELIC_LEVELS)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="0.35", lw=1.4, ls="--", label="on-shell top threshold"),
            Line2D([0], [0], color="#d62728", lw=3.0, label="2024+2025 T2tt expected (reference only)"),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(
        r"Higgsino-stop thermal relic-density contours ($\mu<0$, $\tan\beta=1.6$)",
        fontsize=17,
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        "Gauginos and other scalars at 10 TeV; published micrOMEGAs 6.0 data",
        ha="center",
        va="top",
        fontsize=11,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def write_html_fragment(payload: dict[str, Any], output_path: Path) -> None:
    compact = json.dumps(payload, separators=(",", ":"))
    html = f'''<div id="higgsino-relic-contours">
  <h2>Higgsino-stop thermal relic-density contours</h2>
  <p class="text-muted text-small">Published micrOMEGAs 6.0 data; μ &lt; 0, tan β = 1.6, gauginos and other scalars at 10 TeV</p>
  <div class="legend viz-row" aria-label="Contour legend"></div>
  <div class="plots">
    <section><h3>Published contour plane</h3><div class="chart" data-panel="full"></div></section>
    <section><h3>Run-3 stop-search window</h3><div class="chart" data-panel="zoom"></div></section>
  </div>
  <div class="tooltip" role="tooltip" hidden></div>
  <p class="text-muted text-small">The T2tt expected contour is a collider-sensitivity reference only: the paper benchmark contains a quasi-degenerate higgsino triplet and mixed stop decay modes.</p>
</div>
<style>
#higgsino-relic-contours {{ color: var(--foreground); position: relative; width: 100%; }}
#higgsino-relic-contours h2 {{ margin-bottom: 0; }}
#higgsino-relic-contours h3 {{ margin-bottom: 0; font-weight: 500; }}
#higgsino-relic-contours .legend {{ margin: 0.5rem 0; gap: 0.7rem; align-items: center; }}
#higgsino-relic-contours .legend button {{ display: inline-flex; gap: 0.35rem; align-items: center; padding: 0; border: 0; background: transparent; color: var(--foreground); }}
#higgsino-relic-contours .legend button[aria-pressed="false"] {{ opacity: 0.42; }}
#higgsino-relic-contours .swatch {{ display: inline-block; width: 1.4rem; height: 0.18rem; background: var(--swatch); }}
#higgsino-relic-contours .swatch.cms {{ height: 0; border-top: 0.18rem dashed var(--foreground); background: transparent; }}
#higgsino-relic-contours .plots {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
#higgsino-relic-contours section {{ min-width: 0; }}
#higgsino-relic-contours .chart {{ width: 100%; min-height: 390px; }}
#higgsino-relic-contours .chart-svg {{ width: 100%; display: block; overflow: visible; }}
#higgsino-relic-contours .axis text {{ fill: var(--foreground); font-size: 12px; }}
#higgsino-relic-contours .axis path, #higgsino-relic-contours .axis line {{ stroke: var(--border); }}
#higgsino-relic-contours .grid line {{ stroke: var(--border); stroke-opacity: 0.45; }}
#higgsino-relic-contours .grid path {{ display: none; }}
#higgsino-relic-contours rect[data-chart-frame] {{ fill: none; stroke: var(--border); }}
#higgsino-relic-contours text.axis-title {{ fill: var(--foreground); font-size: 12px; }}
#higgsino-relic-contours .threshold {{ fill: none; stroke: var(--muted-foreground); stroke-width: 1.2; stroke-dasharray: 6 5; }}
#higgsino-relic-contours .degenerate {{ fill: none; stroke: var(--muted-foreground); stroke-width: 1; stroke-dasharray: 2 5; }}
#higgsino-relic-contours .threshold-label {{ fill: var(--muted-foreground); font-size: 11px; }}
#higgsino-relic-contours .relic-line {{ fill: none; stroke-width: 2.2; }}
#higgsino-relic-contours .cms-line {{ fill: none; stroke: var(--foreground); stroke-width: 3; stroke-dasharray: 9 5; }}
#higgsino-relic-contours .hover-guide {{ stroke: var(--muted-foreground); stroke-width: 1; pointer-events: none; }}
#higgsino-relic-contours .hover-marker {{ stroke: var(--background); stroke-width: 1.5; pointer-events: none; }}
#higgsino-relic-contours .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 0.45rem 0.55rem; z-index: 3; }}
#higgsino-relic-contours .tooltip-row {{ display: flex; align-items: center; gap: 0.35rem; white-space: nowrap; }}
@media (max-width: 760px) {{
  #higgsino-relic-contours .plots {{ grid-template-columns: 1fr; }}
  #higgsino-relic-contours .chart {{ min-height: 360px; }}
}}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("higgsino-relic-contours");
  const payload = {compact};
  const colors = ["var(--viz-series-1)", "var(--viz-series-2)", "var(--viz-series-3)", "var(--viz-series-4)", "var(--viz-series-5)", "var(--viz-series-6)"];
  const relic = payload.relic_contours.map((series, index) => ({{
    id: `omega-${{series.omega_h2}}`,
    label: `Ωh² = ${{series.omega_h2}}`,
    color: colors[index],
    points: series.points.map(d => ({{x: d[0], y: d[1]}})),
    visible: true,
    kind: "relic"
  }}));
  const cms = {{
    id: "cms-reference",
    label: "T2tt expected (reference)",
    color: "var(--foreground)",
    segments: payload.t2tt_expected_reference.paths.map(path => path.map(d => ({{x: d[0], y: d[1]}}))),
    visible: true,
    kind: "cms"
  }};
  const series = [...relic, cms];
  const legend = d3.select(root).select(".legend");
  series.forEach(item => {{
    const button = legend.append("button")
      .attr("type", "button")
      .attr("aria-pressed", "true")
      .attr("aria-label", `Toggle ${{item.label}}`)
      .on("click", function() {{
        item.visible = !item.visible;
        d3.select(this).attr("aria-pressed", String(item.visible));
        drawAll();
      }});
    button.append("span")
      .attr("class", item.kind === "cms" ? "swatch cms" : "swatch")
      .style("--swatch", item.color);
    button.append("span").text(item.label);
  }});

  const panels = [
    {{name: "full", xDomain: [0, 7000], yDomain: [100, 1500], showCms: false}},
    {{name: "zoom", xDomain: [600, 1800], yDomain: [0, 1300], showCms: true}}
  ];
  const tooltip = d3.select(root).select(".tooltip");

  function intersectionsAtX(item, xValue, yValue, panel) {{
    const paths = item.kind === "cms" ? item.segments : [item.points];
    const candidates = [];
    paths.forEach(points => {{
      for (let i = 1; i < points.length; i += 1) {{
        const a = points[i - 1], b = points[i];
        if ((xValue < Math.min(a.x, b.x)) || (xValue > Math.max(a.x, b.x)) || a.x === b.x) continue;
        const fraction = (xValue - a.x) / (b.x - a.x);
        const y = a.y + fraction * (b.y - a.y);
        if (y >= panel.yDomain[0] && y <= panel.yDomain[1]) candidates.push(y);
      }}
    }});
    if (!candidates.length) return null;
    return candidates.reduce((best, value) => Math.abs(value - yValue) < Math.abs(best - yValue) ? value : best);
  }}

  function drawPanel(panel) {{
    const container = root.querySelector(`[data-panel="${{panel.name}}"]`);
    const width = Math.max(280, Math.floor(container.getBoundingClientRect().width));
    const height = width < 430 ? 360 : 420;
    const margin = {{top: 14, right: 18, bottom: 58, left: 72}};
    d3.select(container).selectAll("*").remove();
    const svg = d3.select(container).append("svg")
      .attr("class", "chart-svg")
      .attr("viewBox", `0 0 ${{width}} ${{height}}`)
      .attr("role", "img")
      .attr("aria-label", `${{panel.name === "full" ? "Full" : "Run-3 zoom"}} stop versus higgsino LSP relic-density contour plot`);
    svg.append("title").text("Higgsino-stop relic-density contours");
    svg.append("desc").text("Equal thermal relic-density curves in the stop and lightest-neutralino mass plane, with the top-decay threshold and a reference collider limit in the zoom panel.");
    const x = d3.scaleLinear().domain(panel.xDomain).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain(panel.yDomain).range([height - margin.bottom, margin.top]);
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const clipId = `clip-${{panel.name}}-${{Math.round(width)}}`;
    svg.append("defs").append("clipPath").attr("id", clipId)
      .append("rect").attr("x", margin.left).attr("y", margin.top).attr("width", plotWidth).attr("height", plotHeight);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${{height - margin.bottom}})`)
      .call(d3.axisBottom(x).ticks(width < 430 ? 4 : 6).tickSize(-plotHeight).tickFormat(""));
    svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`)
      .call(d3.axisLeft(y).ticks(6).tickSize(-plotWidth).tickFormat(""));
    svg.append("rect").attr("data-chart-frame", "").attr("x", margin.left).attr("y", margin.top).attr("width", plotWidth).attr("height", plotHeight);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`)
      .call(d3.axisBottom(x).ticks(width < 430 ? 4 : 6));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
      .attr("x", margin.left + plotWidth / 2).attr("y", height - 12).attr("text-anchor", "middle").text("m(t̃₁) [GeV]");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
      .attr("transform", `translate(18,${{margin.top + plotHeight / 2}}) rotate(-90)`).attr("text-anchor", "middle").text("m(χ̃⁰₁) [GeV]");

    const line = d3.line().x(d => x(d.x)).y(d => y(d.y));
    const clipped = svg.append("g").attr("clip-path", `url(#${{clipId}})`);
    const referenceX = d3.range(panel.xDomain[0], panel.xDomain[1] + 1, (panel.xDomain[1] - panel.xDomain[0]) / 200);
    clipped.append("path").datum(referenceX.map(value => ({{x: value, y: value}}))).attr("class", "degenerate").attr("d", line);
    clipped.append("path").datum(referenceX.map(value => ({{x: value, y: value - 172.5}}))).attr("class", "threshold").attr("d", line);
    const labelX = panel.xDomain[0] + 0.17 * (panel.xDomain[1] - panel.xDomain[0]);
    const labelY = labelX - 172.5;
    if (labelY > panel.yDomain[0] && labelY < panel.yDomain[1]) {{
      svg.append("text").attr("class", "threshold-label").attr("x", x(labelX)).attr("y", y(labelY) - 7).text("on-shell t threshold");
    }}
    relic.filter(item => item.visible).forEach(item => {{
      clipped.append("path").datum(item.points).attr("class", "relic-line").attr("stroke", item.color).attr("d", line);
    }});
    if (panel.showCms && cms.visible) {{
      cms.segments.forEach(segment => clipped.append("path").datum(segment).attr("class", "cms-line").attr("d", line));
    }}

    const guide = svg.append("line").attr("class", "hover-guide").attr("data-chart-hover-guide", "").attr("y1", margin.top).attr("y2", height - margin.bottom).style("display", "none");
    const markerLayer = svg.append("g");
    svg.append("rect")
      .attr("data-chart-hit", "")
      .attr("data-chart-hover-overlay", "cross-series")
      .attr("x", margin.left).attr("y", margin.top).attr("width", plotWidth).attr("height", plotHeight)
      .attr("fill", "transparent")
      .on("pointermove", event => {{
        const [px, py] = d3.pointer(event, svg.node());
        const xValue = x.invert(Math.max(margin.left, Math.min(width - margin.right, px)));
        const yValue = y.invert(Math.max(margin.top, Math.min(height - margin.bottom, py)));
        const active = series.filter(item => item.visible && (panel.showCms || item.kind !== "cms"));
        const rows = active.map(item => ({{item, y: intersectionsAtX(item, xValue, yValue, panel)}})).filter(row => row.y !== null);
        guide.attr("x1", x(xValue)).attr("x2", x(xValue)).style("display", null);
        markerLayer.selectAll("circle").data(rows, row => row.item.id).join("circle")
          .attr("class", "hover-marker").attr("data-chart-hover-marker", "")
          .attr("cx", x(xValue)).attr("cy", row => y(row.y)).attr("r", 4).attr("fill", row => row.item.color);
        tooltip.html(`<div><strong>m(t̃₁) = ${{xValue.toFixed(1)}} GeV</strong></div>` + rows.map(row => `<div class="tooltip-row"><span class="swatch${{row.item.kind === "cms" ? " cms" : ""}}" style="--swatch:${{row.item.color}}"></span><span>${{row.item.label}}: ${{row.y.toFixed(1)}} GeV</span></div>`).join(""))
          .attr("hidden", null)
          .style("left", `${{Math.min(px + 14, root.clientWidth - 260)}}px`)
          .style("top", `${{Math.max(0, container.offsetTop + py - 26)}}px`);
      }})
      .on("pointerleave", () => {{
        guide.style("display", "none");
        markerLayer.selectAll("circle").remove();
        tooltip.attr("hidden", true);
      }});
  }}

  function drawAll() {{ panels.forEach(drawPanel); }}
  drawAll();
  let resizeTimer;
  new ResizeObserver(() => {{ clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 80); }}).observe(root);
}})();
</script>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> int:
    args = parse_args()
    curves = load_relic_curves(args.source_dir)
    collider_paths = (
        extract_t2tt_expected_contours(args.expected_limits)
        if args.expected_limits
        else []
    )
    output_dir = args.output_dir.resolve()
    machine_path = output_dir / "higgsino_stop_relic_contours.json"
    payload = write_machine_data(
        machine_path, curves, collider_paths, args.expected_limits
    )
    plot_static(
        payload,
        output_dir / "higgsino_stop_relic_contours_full_and_run3.png",
        output_dir / "higgsino_stop_relic_contours_full_and_run3.pdf",
    )
    if args.html_output:
        write_html_fragment(payload, args.html_output.resolve())
    print(json.dumps({
        "machine_data": str(machine_path),
        "png": str(output_dir / "higgsino_stop_relic_contours_full_and_run3.png"),
        "pdf": str(output_dir / "higgsino_stop_relic_contours_full_and_run3.pdf"),
        "html": str(args.html_output.resolve()) if args.html_output else None,
        "relic_contour_count": len(curves),
        "t2tt_reference_path_count": len(collider_paths),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
