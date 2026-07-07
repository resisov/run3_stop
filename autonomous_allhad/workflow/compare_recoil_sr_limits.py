#!/usr/bin/env python3
"""Compare recoil-bin SR and recoil-bin SR+nTop expected-limit outputs.

This script intentionally does not draw new limit contours.  It compares the
machine-readable Combine outputs and reuses the already produced contour images
for a small side-by-side HTML summary.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def load_points(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload.get("points") or {}


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = fraction * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def build_rows(sr_points: dict[str, dict[str, Any]], nt1_points: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    common = sorted(set(sr_points) & set(nt1_points), key=lambda key: (sr_points[key]["mStop"], sr_points[key]["mLSP"]))
    rows: list[dict[str, Any]] = []
    for key in common:
        sr_expected = float(sr_points[key].get("expected", float("nan")))
        nt1_expected = float(nt1_points[key].get("expected", float("nan")))
        if sr_expected <= 0 or nt1_expected <= 0:
            continue
        if not (math.isfinite(sr_expected) and math.isfinite(nt1_expected)):
            continue
        rows.append(
            {
                "mass_key": key,
                "mStop": int(sr_points[key]["mStop"]),
                "mLSP": int(sr_points[key]["mLSP"]),
                "expected_SR": sr_expected,
                "expected_SR_Nt1": nt1_expected,
                "ratio_Nt1_over_SR": nt1_expected / sr_expected,
                "better": "SR_Nt1" if nt1_expected < sr_expected else "SR" if sr_expected < nt1_expected else "tie",
                "excluded_SR": sr_expected <= 1.0,
                "excluded_SR_Nt1": nt1_expected <= 1.0,
            }
        )
    return rows


def lsp_slice_reach(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | None]]:
    output: dict[str, dict[str, int | None]] = {}
    for mlsp in sorted({int(row["mLSP"]) for row in rows}):
        selected = [row for row in rows if int(row["mLSP"]) == mlsp]
        sr_reach = max([int(row["mStop"]) for row in selected if row["excluded_SR"]], default=None)
        nt1_reach = max([int(row["mStop"]) for row in selected if row["excluded_SR_Nt1"]], default=None)
        output[str(mlsp)] = {
            "reach_SR": sr_reach,
            "reach_SR_Nt1": nt1_reach,
            "delta_Nt1_minus_SR": None if sr_reach is None or nt1_reach is None else nt1_reach - sr_reach,
        }
    return output


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    ratio = summary["ratio_summary"]
    counts = summary["point_counts"]
    lines = [
        "# Recoil SR vs Recoil SR + nTop Expected Limits\n",
        f"Common mass points: {summary['common_points']}; valid ratio points: {summary['valid_ratio_points']}.\n",
        (
            f"Median ratio SR_Nt1/SR: {ratio['median']:.3g}; "
            f"q16-q84: {ratio['q16']:.3g}-{ratio['q84']:.3g}. "
            "Ratio < 1 means SR_Nt1 is stronger.\n"
        ),
        (
            f"Point counts: SR_Nt1 better {counts['SR_Nt1_better']}, "
            f"SR better {counts['SR_better']}, ties {counts['tie']}. "
            f"Expected-excluded grid points: SR {counts['excluded_SR']}, "
            f"SR_Nt1 {counts['excluded_SR_Nt1']}.\n"
        ),
        "\n## LSP-Slice Expected Excluded Stop Reach\n",
        "| mLSP | SR reach | SR_Nt1 reach | delta |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for mlsp, rec in summary["lsp_slice_reach"].items():
        lines.append(f"| {mlsp} | {rec['reach_SR']} | {rec['reach_SR_Nt1']} | {rec['delta_Nt1_minus_SR']} |\n")
    lines.extend(
        [
            "\n## Strongest SR_Nt1 Relative Gains\n",
            "| point | SR | SR_Nt1 | ratio |\n",
            "|---|---:|---:|---:|\n",
        ]
    )
    for row in summary["largest_SR_Nt1_gains"][:10]:
        lines.append(
            f"| {row['mass_key']} | {row['expected_SR']:.4g} | "
            f"{row['expected_SR_Nt1']:.4g} | {row['ratio_Nt1_over_SR']:.3g} |\n"
        )
    lines.extend(
        [
            "\n## Largest SR_Nt1 Relative Losses\n",
            "| point | SR | SR_Nt1 | ratio |\n",
            "|---|---:|---:|---:|\n",
        ]
    )
    for row in summary["largest_SR_losses"][:10]:
        lines.append(
            f"| {row['mass_key']} | {row['expected_SR']:.4g} | "
            f"{row['expected_SR_Nt1']:.4g} | {row['ratio_Nt1_over_SR']:.3g} |\n"
        )
    output_path.write_text("".join(lines))


def write_html(summary: dict[str, Any], output_path: Path) -> None:
    ratio = summary["ratio_summary"]
    counts = summary["point_counts"]
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Recoil SR Limit Comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
img {{ max-width: 100%; border: 1px solid #ccc; }}
</style>
<h1>Recoil SR vs Recoil SR + nTop Expected Limits</h1>
<p>
Common mass points: {summary["common_points"]}.
Median SR_Nt1/SR expected-limit ratio: {ratio["median"]:.3g}.
Ratio &lt; 1 means SR_Nt1 is stronger.
Point counts: SR_Nt1 better {counts["SR_Nt1_better"]}, SR better {counts["SR_better"]}.
Expected-excluded grid points: SR {counts["excluded_SR"]}, SR_Nt1 {counts["excluded_SR_Nt1"]}.
</p>
<div class="grid">
  <figure>
    <img src="expected_limit_contour_recoil_sr.png" alt="Recoil-bin SR expected limit contour">
    <figcaption>Recoil-bin SR</figcaption>
  </figure>
  <figure>
    <img src="expected_limit_contour_recoil_sr_nt1.png" alt="Recoil-bin SR plus nTop condition expected limit contour">
    <figcaption>Recoil-bin SR + nTop condition</figcaption>
  </figure>
</div>
<p><a href="limit_comparison.json">limit_comparison.json</a> · <a href="summary.md">summary.md</a></p>
"""
    output_path.write_text(html)



def contour_grid(points: dict[str, dict[str, Any]], quantity: str = "expected") -> tuple[Any, Any, Any] | None:
    rows = []
    for rec in points.values():
        value = rec.get(quantity)
        if value is None:
            continue
        value = float(value)
        if value <= 0 or not math.isfinite(value):
            continue
        rows.append((float(rec["mStop"]), float(rec["mLSP"]), math.log10(value)))
    if len(rows) < 4:
        return None
    from scipy.interpolate import griddata

    xmin, xmax = 600.0, 1500.0
    ymin, ymax = 0.0, 1500.0
    xi = np.linspace(xmin, xmax, 260)
    yi = np.linspace(ymin, ymax, 260)
    xx, yy = np.meshgrid(xi, yi)
    xs = np.asarray([row[0] for row in rows])
    ys = np.asarray([row[1] for row in rows])
    zs = np.asarray([row[2] for row in rows])
    zz = griddata((xs, ys), zs, (xx, yy), method="linear")
    mask = np.isnan(zz) | (yy > (xx - 172.5))
    return xx, yy, np.ma.array(zz, mask=mask)


def load_run2_contours(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def draw_multi_overlay(variants: list[dict[str, str]], run2_path: Path, output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    hep.style.use("CMS")
    colors = ["#d7191c", "#2c7bb6", "#1a9641", "#984ea3", "#ff7f00", "#c51b8a", "#17becf", "#8c564b"]
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.90)

    diag_x = np.linspace(600.0, 1500.0, 400)
    diag_y = diag_x - 172.5
    keep = (diag_y >= 0.0) & (diag_y <= 1500.0)
    ax.plot(diag_x[keep], diag_y[keep], color="0.55", linestyle=":", linewidth=1.1, zorder=2)

    rows = []
    run3_handles: list[Any] = []
    reference_handles: list[Any] = []

    def plot_label(label: str) -> str:
        mapping = {
            "Inclusive SR": r"Inclusive high-$\Delta m$ SR",
            "SR, N_t >= 1": r"High-$\Delta m$ SR, $N_t\geq1$",
            "SR, N_t = 0": r"High-$\Delta m$ SR, $N_t=0$",
            "CR/SR N_t split": r"CR/SR split by $N_t$",
            "CR/SR N_t split + low-dM": r"CR/SR split by $N_t$ + low-$\Delta m$ SR",
            "Selected 7x6 + low-dM": r"Selected 7$\times$6 recoil categories + low-$\Delta m$ SR",
            "Selected 8x6 + low-dM": r"Selected 8$\times$6 recoil categories + low-$\Delta m$ SR",
        }
        return mapping.get(label, label)
    for idx, spec in enumerate(variants):
        points = load_points(Path(spec["dir"]) / "expected_limits.json")
        grid = contour_grid(points)
        if grid is None:
            rows.append({"label": spec["label"], "status": "missing_grid", "points": len(points)})
            continue
        xx, yy, zz = grid
        color = colors[idx % len(colors)]
        ax.contour(xx, yy, zz, levels=[0.0], colors=color, linewidths=3.0, zorder=5)
        run3_handles.append(Line2D([0], [0], color=color, lw=3.0, label=plot_label(spec["label"])))
        rows.append({"label": spec["label"], "status": "plotted", "points": len(points)})

    run2 = load_run2_contours(run2_path)
    for key, style, label in [
        ("observed", "-", "Run-2 SUS-19-010 observed"),
        ("expected", "--", "Run-2 SUS-19-010 expected"),
    ]:
        arr = np.asarray(run2.get(key) or [], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="black", linestyle=style, linewidth=2.4, zorder=6)
            reference_handles.append(Line2D([0], [0], color="black", lw=2.4, linestyle=style, label=label))

    reference_handles.append(Line2D([0], [0], color="0.55", lw=1.2, linestyle=":", label=r"$m_{\tilde{\chi}_1^0}=m_{\tilde{t}}-m_t$"))
    ax.set_xlim(600.0, 1500.0)
    ax.set_ylim(0.0, 1500.0)
    ax.set_xlabel(r"$m_{\tilde{t}}$ (GeV)", fontsize=30, loc="right")
    ax.set_ylabel(r"$m_{\tilde{\chi}_1^0}$ (GeV)", fontsize=30)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=21, length=9)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    hep.cms.label(llabel="Work in progress", rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)", ax=ax)
    ax.text(0.14, 0.95, r"$pp\rightarrow \tilde{t}\tilde{t},\ \tilde{t}\rightarrow t\tilde{\chi}_1^0$", transform=ax.transAxes, fontsize=15, va="top")
    ref_legend = ax.legend(
        handles=reference_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.855),
        frameon=False,
        fontsize=13.5,
        handlelength=3.0,
        labelspacing=0.45,
    )
    ax.add_artist(ref_legend)
    ax.legend(
        handles=run3_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        frameon=True,
        facecolor="white",
        edgecolor="0.72",
        framealpha=0.92,
        fontsize=11.5,
        title="Run-3 expected contours",
        title_fontsize=13.0,
        handlelength=3.0,
        borderpad=0.65,
        labelspacing=0.45,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return {"status": "complete", "output_png": str(output), "output_pdf": str(output.with_suffix(".pdf")), "variants": rows, "run2": str(run2_path)}


def run_multi_overlay(args: argparse.Namespace) -> int:
    labels = args.label or []
    dirs = args.variant_dir or []
    if len(labels) != len(dirs):
        raise SystemExit("--label and --variant-dir counts must match")
    variants = [{"label": label, "dir": directory} for label, directory in zip(labels, dirs)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = draw_multi_overlay(variants, Path(args.run2_contours), output_dir / "expected_limit_overlay_run2.png")
    (output_dir / "limit_overlay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr-dir")
    parser.add_argument("--nt1-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--multi-overlay", action="store_true")
    parser.add_argument("--variant-dir", action="append")
    parser.add_argument("--label", action="append")
    parser.add_argument("--run2-contours", default="/eos/user/t/taiwoo/run2_sus19010_contours.json")
    args = parser.parse_args()

    if args.multi_overlay:
        return run_multi_overlay(args)
    if not args.sr_dir or not args.nt1_dir:
        raise SystemExit("--sr-dir and --nt1-dir are required unless --multi-overlay is used")

    sr_dir = Path(args.sr_dir)
    nt1_dir = Path(args.nt1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_points = load_points(sr_dir / "expected_limits.json")
    nt1_points = load_points(nt1_dir / "expected_limits.json")
    rows = build_rows(sr_points, nt1_points)
    ratios = [float(row["ratio_Nt1_over_SR"]) for row in rows]
    summary = {
        "status": "complete",
        "common_points": len(set(sr_points) & set(nt1_points)),
        "valid_ratio_points": len(rows),
        "inputs": {"SR": str(sr_dir), "SR_Nt1": str(nt1_dir)},
        "comparison_definition": "ratio_Nt1_over_SR < 1 means SR_Nt1 gives a stronger expected limit at that mass point.",
        "ratio_summary": {
            "min": min(ratios) if ratios else None,
            "q16": quantile(ratios, 0.16),
            "median": quantile(ratios, 0.5),
            "q84": quantile(ratios, 0.84),
            "max": max(ratios) if ratios else None,
        },
        "point_counts": {
            "SR_Nt1_better": sum(row["better"] == "SR_Nt1" for row in rows),
            "SR_better": sum(row["better"] == "SR" for row in rows),
            "tie": sum(row["better"] == "tie" for row in rows),
            "excluded_SR": sum(bool(row["excluded_SR"]) for row in rows),
            "excluded_SR_Nt1": sum(bool(row["excluded_SR_Nt1"]) for row in rows),
        },
        "lsp_slice_reach": lsp_slice_reach(rows),
        "largest_SR_Nt1_gains": sorted(rows, key=lambda row: float(row["ratio_Nt1_over_SR"]))[:15],
        "largest_SR_losses": sorted(rows, key=lambda row: float(row["ratio_Nt1_over_SR"]), reverse=True)[:15],
    }

    (output_dir / "limit_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_markdown(summary, output_dir / "summary.md")

    for src, name in [
        (sr_dir / "expected_limit_contour.png", "expected_limit_contour_recoil_sr.png"),
        (nt1_dir / "expected_limit_contour.png", "expected_limit_contour_recoil_sr_nt1.png"),
        (sr_dir / "expected_limit_contour.pdf", "expected_limit_contour_recoil_sr.pdf"),
        (nt1_dir / "expected_limit_contour.pdf", "expected_limit_contour_recoil_sr_nt1.pdf"),
    ]:
        if src.exists():
            shutil.copy2(src, output_dir / name)
    write_html(summary, output_dir / "index.html")
    print(json.dumps({"output": str(output_dir), **summary["ratio_summary"], **summary["point_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
