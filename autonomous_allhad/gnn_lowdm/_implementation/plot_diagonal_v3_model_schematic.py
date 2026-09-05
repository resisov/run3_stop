#!/usr/bin/env python3
"""Draw the publication schematic for the diagonal-v3 Low-dM GNN.

The figure is intentionally generated from vector primitives so the PDF and
SVG remain editable and sharp at journal typesetting scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#17202A"
MUTED = "#5B6573"
LINE = "#AAB4C0"
BLUE = "#0072B2"
BLUE_LIGHT = "#E7F2F8"
TEAL = "#009E73"
TEAL_LIGHT = "#E5F5EF"
ORANGE = "#E69F00"
ORANGE_LIGHT = "#FFF3D6"
RED = "#D55E00"
RED_LIGHT = "#FCE9DF"
PURPLE = "#7B61A8"
PURPLE_LIGHT = "#F0EBF7"
GRAY_LIGHT = "#F4F6F8"


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = LINE,
    linewidth: float = 1.2,
    radius: float = 0.012,
    zorder: float = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.35,
    style: str = "-|>",
    mutation_scale: float = 12.0,
    connectionstyle: str = "arc3",
    zorder: float = 4,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 9.0,
    weight: str = "normal",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
    zorder: float = 8,
    linespacing: float = 1.18,
) -> None:
    axis.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
        linespacing=linespacing,
    )


def draw_event_graph(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor="white", edgecolor=BLUE, linewidth=1.4)
    label(axis, x + 0.014, y + height - 0.028, "AK4 jet graph", size=10.2, weight="bold", ha="left")
    label(axis, x + width - 0.014, y + height - 0.028, r"$N_{\mathrm{jet}}\leq 10$", size=8.3, color=MUTED, ha="right")

    positions = [
        (0.16, 0.25),
        (0.38, 0.15),
        (0.62, 0.27),
        (0.83, 0.19),
        (0.23, 0.62),
        (0.49, 0.53),
        (0.77, 0.66),
    ]
    points = [(x + width * px, y + height * (0.16 + 0.62 * py)) for px, py in positions]
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            axis.plot(
                [points[first][0], points[second][0]],
                [points[first][1], points[second][1]],
                color="#CCD7E0",
                linewidth=0.55,
                zorder=2.5,
            )
    for index, (px, py) in enumerate(points):
        fill = ORANGE if index in (1, 5) else BLUE
        axis.add_patch(Circle((px, py), 0.0088, facecolor=fill, edgecolor="white", linewidth=1.0, zorder=5))
        axis.add_patch(Circle((px, py), 0.0108, facecolor="none", edgecolor=INK, linewidth=0.55, zorder=4))

    met_start = (x + 0.22 * width, y + 0.18 * height)
    met_end = (x + 0.72 * width, y + 0.18 * height)
    arrow(axis, met_start, met_end, color=TEAL, linewidth=2.1, mutation_scale=11)
    label(axis, met_end[0] + 0.006, met_end[1], r"$\vec{p}_{\mathrm{T}}^{\mathrm{miss}}$", size=8.2, color=TEAL, ha="left")

    label(
        axis,
        x + width / 2,
        y + 0.032,
        r"$\mathbf{x}_i=(\log p_{\mathrm{T}},\eta,\sin\Delta\phi,\cos\Delta\phi,\log m,\mathrm{b\ score})$",
        size=7.7,
    )


def draw_vector(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.25)
    label(axis, x + 0.014, y + height - 0.027, "Event-level vector", size=10.0, weight="bold", ha="left")
    label(axis, x + width - 0.014, y + height - 0.027, "40 features", size=8.3, color=MUTED, ha="right")
    entries = (
        r"Event: $p_{\mathrm{T}}^{\mathrm{miss}}, H_{\mathrm{T}}, N_j, N_b, \min\Delta\phi$",
        r"b system: $m_{\mathrm{T}}$, $m_{bb}$, $m_{\mathrm{CT}}$, $m_{\mathrm{T2}}^{bb}$",
        r"Resolved: $m_{jj}$, $m_{jjj}$, $\chi^2$, event shapes",
        r"ISR/recoil: $N_{\mathrm{ISR}}$, $p_{\mathrm{T}}^{\mathrm{ISR}}$, balance",
    )
    for index, entry in enumerate(entries):
        cy = y + height - 0.060 - 0.030 * index
        axis.add_patch(Circle((x + 0.019, cy), 0.0034, facecolor=TEAL, edgecolor="none", zorder=5))
        label(axis, x + 0.029, cy, entry, size=7.75, ha="left")
    label(axis, x + 0.015, y + 0.020, "Presence masks encode missing ISR and composite candidates", size=7.15, color=MUTED, ha="left")


def draw_encoder(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.35)
    label(axis, x + width / 2, y + height - 0.032, "Input encoders", size=10.4, weight="bold")
    rounded_box(axis, x + 0.015, y + 0.088, width - 0.030, 0.074, facecolor="white", edgecolor=BLUE, linewidth=1.0, radius=0.007)
    label(axis, x + width / 2, y + 0.125, r"Jet MLP: $6\rightarrow h\rightarrow h$", size=8.8, weight="bold")
    rounded_box(axis, x + 0.015, y + 0.014, width - 0.030, 0.057, facecolor="white", edgecolor=TEAL, linewidth=1.0, radius=0.007)
    label(axis, x + width / 2, y + 0.043, r"Global MLP: $40\rightarrow h\rightarrow h$", size=8.55, weight="bold")
    label(axis, x + width / 2, y + height - 0.060, r"$\mathbf{h}_i^{(0)}=f_{\mathrm{jet}}(\mathbf{x}_i)+f_{\mathrm{global}}(\mathbf{g})$", size=7.6)


def draw_message_block(axis: plt.Axes, x: float, y: float, width: float, height: float, index: int) -> None:
    rounded_box(axis, x, y, width, height, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.3)
    label(axis, x + width / 2, y + height - 0.030, f"Message block {index}", size=9.8, weight="bold")
    label(axis, x + width / 2, y + height - 0.067, r"Dense jet pairs $i\ne j$", size=7.65, color=MUTED)

    nodes = [(x + 0.034, y + 0.075), (x + 0.070, y + 0.105), (x + 0.106, y + 0.071)]
    for first in range(len(nodes)):
        for second in range(len(nodes)):
            if first == second:
                continue
            arrow(
                axis,
                nodes[first],
                nodes[second],
                color="#C58B26",
                linewidth=0.55,
                mutation_scale=5.0,
                connectionstyle="arc3,rad=0.12",
                zorder=3,
            )
    for px, py in nodes:
        axis.add_patch(Circle((px, py), 0.0063, facecolor=ORANGE, edgecolor="white", linewidth=0.8, zorder=5))

    label(axis, x + width / 2, y + 0.038, r"$\mathbf{e}_{ij}=(\Delta\eta,\sin\Delta\phi,\cos\Delta\phi,\Delta R)$", size=7.2)
    label(axis, x + width / 2, y + 0.016, "message MLP + mean/max aggregation", size=7.05, color=MUTED)


def draw_pool(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.3)
    label(axis, x + width / 2, y + height - 0.033, "Masked event pooling", size=9.8, weight="bold")
    for row, symbol in enumerate((r"$\langle\mathbf{h}_i^{(L)}\rangle_i$", r"$\max_i\mathbf{h}_i^{(L)}$")):
        by = y + height - 0.078 - 0.050 * row
        rounded_box(axis, x + 0.027, by - 0.017, width - 0.054, 0.034, facecolor="white", edgecolor=PURPLE, linewidth=0.9, radius=0.005)
        label(axis, x + width / 2, by, symbol, size=8.8)
    label(axis, x + width / 2, y + 0.023, r"concatenate with $f_{\mathrm{global}}(\mathbf{g})$ and $\mathbf{g}$", size=7.0, color=MUTED)


def draw_head(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor=RED_LIGHT, edgecolor=RED, linewidth=1.35)
    label(axis, x + width / 2, y + height - 0.034, "Event classifier", size=10.1, weight="bold")
    layers_y = [y + 0.112, y + 0.082, y + 0.052]
    widths = [5, 4, 2]
    x_groups = [x + 0.036, x + 0.077, x + 0.111]
    previous = []
    for group_x, count, cy in zip(x_groups, widths, layers_y, strict=True):
        current = []
        for offset in range(count):
            py = cy + (offset - (count - 1) / 2) * 0.017
            current.append((group_x, py))
            axis.add_patch(Circle((group_x, py), 0.0046, facecolor="white", edgecolor=RED, linewidth=0.85, zorder=5))
        for p0 in previous:
            for p1 in current:
                axis.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#DAB4A3", linewidth=0.38, zorder=3)
        previous = current
    score = (x + width - 0.025, y + 0.082)
    for p0 in previous:
        axis.plot([p0[0], score[0]], [p0[1], score[1]], color="#DAB4A3", linewidth=0.48, zorder=3)
    axis.add_patch(Circle(score, 0.0085, facecolor=RED, edgecolor="white", linewidth=1.0, zorder=5))
    label(axis, x + width / 2, y + 0.022, r"MLP + sigmoid $\rightarrow s_{\mathrm{GNN}}\in[0,1]$", size=7.8, weight="bold")


def draw_binning(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(axis, x, y, width, height, facecolor=GRAY_LIGHT, edgecolor=INK, linewidth=1.25)
    label(axis, x + width / 2, y + height - 0.030, r"Low-$\Delta m$ search bins", size=9.9, weight="bold")
    categories = (
        r"$N_b=1$, $N_{\mathrm{ISR}}=0$",
        r"$N_b=1$, $N_{\mathrm{ISR}}=1$",
        r"$N_b=1$, $N_{\mathrm{ISR}}\geq2$",
        r"$N_b\geq2$, $N_{\mathrm{ISR}}=0$",
        r"$N_b\geq2$, $N_{\mathrm{ISR}}=1$",
        r"$N_b\geq2$, $N_{\mathrm{ISR}}\geq2$",
    )
    colors = (BLUE, TEAL, ORANGE, PURPLE, RED, MUTED)
    for index, (entry, color) in enumerate(zip(categories, colors, strict=True)):
        by = y + height - 0.059 - 0.025 * index
        axis.add_patch(Rectangle((x + 0.016, by - 0.006), 0.010, 0.012, facecolor=color, edgecolor="none", zorder=5))
        label(axis, x + 0.032, by, entry, size=6.8, ha="left")
    label(axis, x + width / 2, y + 0.020, "5 validation-optimized\nscore bins per category", size=6.4, color=MUTED)


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(16.0, 7.5), facecolor="white")
    axis = figure.add_axes((0.012, 0.025, 0.976, 0.95))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    # Stage headings and panel letters.
    headings = (
        (0.022, 0.945, "(a) Event representation"),
        (0.335, 0.945, "(b) Common jet GNN"),
        (0.735, 0.945, "(c) Classifier and analysis output"),
    )
    for x, y, text in headings:
        label(axis, x, y, text, size=12.0, weight="bold", ha="left")
    axis.plot([0.318, 0.318], [0.23, 0.925], color="#D7DDE3", linewidth=0.9)
    axis.plot([0.718, 0.718], [0.23, 0.925], color="#D7DDE3", linewidth=0.9)

    # Event selection capsule.
    rounded_box(axis, 0.025, 0.868, 0.278, 0.046, facecolor=GRAY_LIGHT, edgecolor=LINE, linewidth=0.9, radius=0.008)
    label(
        axis,
        0.164,
        0.891,
        r"$0\ell$, $N_b\geq1$, $N_t=N_W=N_{\mathrm{res}}=0$; no $p_{\mathrm{T}}^{\mathrm{miss}}/\sqrt{H_{\mathrm{T}}}$ or $N_{\mathrm{ISR}}$ cut",
        size=7.65,
    )

    draw_event_graph(axis, 0.025, 0.555, 0.278, 0.285)
    draw_vector(axis, 0.025, 0.315, 0.278, 0.205)

    # GNN core.
    draw_encoder(axis, 0.335, 0.602, 0.125, 0.238)
    draw_message_block(axis, 0.483, 0.602, 0.103, 0.238, 1)
    draw_message_block(axis, 0.610, 0.602, 0.103, 0.238, r"$L$")
    arrow(axis, (0.460, 0.721), (0.483, 0.721), color=BLUE, linewidth=1.7)
    arrow(axis, (0.586, 0.721), (0.610, 0.721), color=ORANGE, linewidth=1.7)
    label(axis, 0.598, 0.757, r"$\cdots$", size=13.0, color=MUTED)
    rounded_box(axis, 0.475, 0.526, 0.246, 0.044, facecolor="white", edgecolor=ORANGE, linewidth=0.9, radius=0.006)
    label(axis, 0.598, 0.548, r"Residual update + LayerNorm; $L=2$ or $3$ in the scan", size=7.35)

    # Input arrows into both encoders.
    arrow(axis, (0.303, 0.695), (0.335, 0.695), color=BLUE, linewidth=1.6)
    arrow(axis, (0.303, 0.418), (0.325, 0.418), color=TEAL, linewidth=1.5)
    arrow(axis, (0.325, 0.418), (0.325, 0.635), color=TEAL, linewidth=1.5, style="-")
    arrow(axis, (0.325, 0.635), (0.335, 0.635), color=TEAL, linewidth=1.5)

    draw_pool(axis, 0.505, 0.300, 0.180, 0.182)
    arrow(axis, (0.676, 0.602), (0.676, 0.498), color=PURPLE, linewidth=1.6)
    arrow(axis, (0.676, 0.498), (0.595, 0.482), color=PURPLE, linewidth=1.6)

    # Global skip into pooled event representation.
    arrow(axis, (0.303, 0.382), (0.470, 0.382), color=TEAL, linewidth=1.0, style="-", zorder=1)
    arrow(axis, (0.470, 0.382), (0.505, 0.382), color=TEAL, linewidth=1.35)
    label(axis, 0.405, 0.397, "global skip", size=6.9, color=TEAL)

    # Output blocks.
    draw_head(axis, 0.742, 0.560, 0.125, 0.238)
    draw_binning(axis, 0.882, 0.540, 0.106, 0.278)
    arrow(axis, (0.685, 0.391), (0.728, 0.391), color=PURPLE, linewidth=1.5, style="-")
    arrow(axis, (0.728, 0.391), (0.728, 0.679), color=PURPLE, linewidth=1.5, style="-")
    arrow(axis, (0.728, 0.679), (0.742, 0.679), color=PURPLE, linewidth=1.5)
    arrow(axis, (0.867, 0.679), (0.882, 0.679), color=RED, linewidth=1.7)

    # Forward path.
    arrow(axis, (0.335, 0.885), (0.986, 0.885), color=BLUE, linewidth=2.0, mutation_scale=15)
    label(axis, 0.660, 0.907, "forward inference", size=8.0, weight="bold", color=BLUE)

    # Training and validation band.
    rounded_box(axis, 0.025, 0.070, 0.963, 0.125, facecolor="#F8FAFB", edgecolor=LINE, linewidth=1.05, radius=0.010)
    label(axis, 0.044, 0.166, "Training and model selection", size=10.1, weight="bold", ha="left")
    label(axis, 0.045, 0.127, "Deterministic event split", size=7.7, weight="bold", ha="left", color=MUTED)
    # Split bar.
    split_x, split_y, split_w, split_h = 0.160, 0.111, 0.183, 0.031
    axis.add_patch(Rectangle((split_x, split_y), split_w * 0.2, split_h, facecolor=BLUE, edgecolor="white", linewidth=0.8, zorder=4))
    axis.add_patch(Rectangle((split_x + split_w * 0.2, split_y), split_w * 0.1, split_h, facecolor=ORANGE, edgecolor="white", linewidth=0.8, zorder=4))
    axis.add_patch(Rectangle((split_x + split_w * 0.3, split_y), split_w * 0.7, split_h, facecolor="#AEB8C2", edgecolor="white", linewidth=0.8, zorder=4))
    label(axis, split_x + split_w * 0.10, split_y + split_h / 2, "train 20%", size=6.8, color="white", weight="bold")
    label(axis, split_x + split_w * 0.25, split_y + split_h / 2, "val. 10%", size=6.2, color="white", weight="bold")
    label(axis, split_x + split_w * 0.65, split_y + split_h / 2, "sealed test 70%", size=7.0, color="white", weight="bold")

    rounded_box(axis, 0.375, 0.097, 0.280, 0.062, facecolor="white", edgecolor=RED, linewidth=1.0, radius=0.006)
    label(
        axis,
        0.515,
        0.128,
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\lambda\,\left[-\left\langle\log\left(S/\sqrt{B}\right)\right\rangle_{c,t}\right]$",
        size=9.2,
        weight="bold",
    )
    rounded_box(axis, 0.680, 0.097, 0.288, 0.062, facecolor="white", edgecolor=TEAL, linewidth=1.0, radius=0.006)
    label(axis, 0.824, 0.139, "Validation only", size=8.2, weight="bold", color=TEAL)
    label(axis, 0.824, 0.115, "select hyperparameters, epoch, and score-bin edges", size=7.5)
    label(axis, 0.824, 0.091, "objective: worst-tail then median mass-point sensitivity", size=7.0, color=MUTED)

    arrow(axis, (0.950, 0.071), (0.095, 0.071), color=RED, linewidth=1.75, mutation_scale=14)
    label(axis, 0.523, 0.046, "back-propagation through the common classifier", size=8.0, weight="bold", color=RED)

    label(axis, 0.987, 0.023, r"CMS Run 3 Low-$\Delta m$ stop search - diagonal-v3 architecture", size=6.9, color=MUTED, ha="right")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("output/pdf/lowdm_diagonal_v3_gnn_schematic.pdf"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("output/figures/lowdm_diagonal_v3_gnn_schematic.svg"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("output/figures/lowdm_diagonal_v3_gnn_schematic.png"),
    )
    options = parser.parse_args()
    for path in (options.pdf, options.svg, options.png):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.savefig(options.pdf, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(options.svg, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(options.png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


if __name__ == "__main__":
    main()
