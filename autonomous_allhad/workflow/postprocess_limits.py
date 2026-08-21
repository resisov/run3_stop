#!/usr/bin/env python3
"""Collect and plot a canonical High-dM + Low-dM limit grid."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


RUN2_CONTOURS = {
    "T2tt": Path("/eos/user/t/taiwoo/run2_sus19010_contours.json"),
    "T2tb": Path("/eos/user/t/taiwoo/run2_sus19010_contours_t2tb.json"),
    "T2bW": Path("/eos/user/t/taiwoo/run2_sus19010_contours_t2bw.json"),
}

DECAY_LABELS = {
    "T2tt": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow t\tilde{\chi}_1^0$"
    ),
    "T2tb": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^{+*}\tilde{\chi}_1^0\ (50\%),\ "
        r"\bar{\tilde{t}}_1\rightarrow\bar{t}\tilde{\chi}_1^0\ (50\%)"
        r"$"
    ),
    "T2bW": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^+\tilde{\chi}_1^0$"
    ),
}

LUMINOSITY_LABELS = {
    "2024": r"109.82 fb$^{-1}$ (13.6 TeV)",
    "2025": r"110.84 fb$^{-1}$ (13.6 TeV)",
    "2024_2025": r"220.66 fb$^{-1}$ (13.6 TeV)",
}


def load_excluded_points(input_dir: Path) -> set[str]:
    """Read only explicit user-policy exclusions attached to this model."""

    excluded_points: set[str] = set()
    exclusion_dir = input_dir / "excluded_points"
    for path in sorted(exclusion_dir.glob("limit_point_exclusions_*.json")):
        payload = json.loads(path.read_text())
        for record in payload.get("exclusions", []):
            if record.get("status") not in {
                "excluded_by_user",
                "excluded_by_user_policy",
            }:
                continue
            if record.get("model") != input_dir.name:
                continue
            excluded_points.add(
                f"mStop{int(record['mStop_GeV'])}_mLSP{int(record['mLSP_GeV'])}"
            )
    return excluded_points


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"mStop(\d+)_mLSP(\d+)", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def parse_limit_file(path: Path) -> dict[str, float] | None:
    rows = []
    try:
        import uproot

        with uproot.open(path) as root_file:
            tree = root_file.get("limit")
            if tree is not None:
                quantiles = tree["quantileExpected"].array(library="np")
                limits = tree["limit"].array(library="np")
                rows = [
                    (float(quantile), float(value))
                    for quantile, value in zip(quantiles, limits)
                ]
    except Exception:
        rows = []
    if not rows:
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(path))
            tree = root_file.Get("limit") if root_file else None
            if not tree:
                return None
            for entry in tree:
                rows.append(
                    (float(entry.quantileExpected), float(entry.limit))
                )
            root_file.Close()
        except Exception:
            return None
    if not rows:
        return None
    labels = {
        0.025: "expected_m2",
        0.16: "expected_m1",
        0.5: "expected",
        0.84: "expected_p1",
        0.975: "expected_p2",
    }
    if len(rows) != len(labels):
        return None
    output = {}
    for expected_quantile, label in labels.items():
        matches = [
            value
            for quantile, value in rows
            if math.isclose(
                quantile,
                expected_quantile,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
        ]
        if (
            len(matches) != 1
            or not math.isfinite(matches[0])
            or matches[0] < 0.0
        ):
            return None
        output[label] = matches[0]
    return output


def collect_limits(
    limit_dir: Path, mass_keys: list[str], output_json: Path
) -> dict[str, Any]:
    results = {}
    for mass_key in mass_keys:
        candidates = (
            sorted(
                limit_dir.glob(
                    f"higgsCombine_{mass_key}.AsymptoticLimits*.root"
                )
            )
            + sorted(limit_dir.glob(f"higgsCombine_{mass_key}.root"))
            + sorted(limit_dir.glob(f"higgsCombine_{mass_key}*.root"))
        )
        parsed = None
        for path in candidates:
            parsed = parse_limit_file(path)
            if parsed:
                break
        if parsed:
            mstop, mlsp = parse_mass_key(mass_key)
            parsed.update({"mStop": mstop, "mLSP": mlsp})
            results[mass_key] = parsed
    status = (
        "complete"
        if len(results) == len(mass_keys)
        else "partial"
        if results
        else "no_combine_outputs"
    )
    payload = {
        "status": status,
        "points": results,
        "requested_point_count": len(mass_keys),
        "collected_point_count": len(results),
        "missing_points": [key for key in mass_keys if key not in results],
    }
    write_json(output_json, payload)
    return payload


def plot_contour(
    limit_payload: dict[str, Any],
    output_png: Path,
    run2_contours: Path | None,
    luminosity_label: str,
    analysis_label: str | None,
    x_max: float,
    y_min: float,
    y_max: float,
    decay_label: str | None,
) -> bool:
    records = list((limit_payload.get("points") or {}).values())
    points = [
        record
        for record in records
        if "expected" in record and float(record["expected"]) > 0
    ]
    unique_points = {
        (float(record["mStop"]), float(record["mLSP"]))
        for record in points
    }
    if len(unique_points) < 4:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    from scipy.interpolate import griddata

    xmin, xmax = 600.0, float(x_max)
    ymin, ymax = float(y_min), float(y_max)
    top_mass = 172.5
    xi = np.linspace(xmin, xmax, 260)
    yi = np.linspace(ymin, ymax, 260)
    xx, yy = np.meshgrid(xi, yi)
    offshell_mask = yy > (xx - top_mass)

    def interpolated_log_grid(quantity: str) -> np.ma.MaskedArray | None:
        values = []
        for record in records:
            value = record.get(quantity)
            if value is None or float(value) <= 0:
                continue
            values.append(
                (
                    float(record["mStop"]),
                    float(record["mLSP"]),
                    math.log10(float(value)),
                )
            )
        if not values:
            return None
        xs = np.asarray([value[0] for value in values])
        ys = np.asarray([value[1] for value in values])
        zs = np.asarray([value[2] for value in values])
        linear = griddata((xs, ys), zs, (xx, yy), method="linear")
        return np.ma.array(linear, mask=np.isnan(linear) | offshell_mask)

    expected_grid = interpolated_log_grid("expected")
    if expected_grid is None or expected_grid.count() == 0:
        return False
    minus1_grid = interpolated_log_grid("expected_m1")
    plus1_grid = interpolated_log_grid("expected_p1")

    hep.style.use("CMS")
    figure, axes = plt.subplots(figsize=(12.0, 10.0))
    figure.subplots_adjust(left=0.13, right=0.84, bottom=0.11, top=0.90)
    color_min, color_max = -1.5, 1.5
    limit_cmap = LinearSegmentedColormap.from_list(
        "cms_limit_reference",
        [
            "#5965f2",
            "#62a9ff",
            "#55d7f2",
            "#7ef0c9",
            "#d7fb80",
            "#fff176",
            "#ffb45e",
            "#ff6f6f",
        ],
        N=256,
    )
    plot_grid = np.ma.clip(expected_grid, color_min, color_max)
    filled = axes.contourf(
        xx,
        yy,
        plot_grid,
        levels=np.linspace(color_min, color_max, 121),
        cmap=limit_cmap,
    )
    colorbar = figure.colorbar(
        filled, ax=axes, pad=0.04, fraction=0.048, aspect=34
    )
    colorbar.set_label(
        r"$\log_{10}$ (expected 95% CL limit on $\sigma/\sigma_{\mathrm{theory}}$)",
        fontsize=27,
        rotation=90,
        labelpad=22,
    )
    colorbar.set_ticks(np.arange(color_min, color_max + 0.001, 0.5))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    colorbar.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    colorbar.ax.tick_params(
        which="major", labelsize=23, direction="in", length=12, width=1.4
    )
    colorbar.ax.tick_params(
        which="minor", direction="in", length=7, width=1.1
    )
    colorbar.outline.set_linewidth(1.8)

    diagonal_x = np.linspace(xmin, xmax, 400)
    diagonal_y = diagonal_x - top_mass
    keep = (diagonal_y >= ymin) & (diagonal_y <= ymax)
    axes.plot(
        diagonal_x[keep],
        diagonal_y[keep],
        color="0.45",
        linestyle=":",
        linewidth=1.1,
        zorder=4,
    )
    axes.contour(
        xx,
        yy,
        expected_grid,
        levels=[0.0],
        colors="red",
        linewidths=3.0,
        zorder=6,
    )
    for band_grid in (minus1_grid, plus1_grid):
        if band_grid is not None and band_grid.count() > 0:
            axes.contour(
                xx,
                yy,
                band_grid,
                levels=[0.0],
                colors="red",
                linewidths=1.7,
                linestyles="--",
                zorder=5,
            )

    run2_handles: list[Any] = []
    if run2_contours is not None and Path(run2_contours).exists():
        try:
            run2_payload = read_json(Path(run2_contours))
        except Exception:
            run2_payload = {}
        for key, style, label in (
            ("observed", "-", "SUS-19-010 obs."),
            ("expected", "--", "SUS-19-010 exp."),
        ):
            coordinates = np.asarray(run2_payload.get(key) or [], dtype=float)
            if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
                axes.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color="black",
                    linestyle=style,
                    linewidth=2.4,
                    zorder=7,
                )
                run2_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2.4,
                        linestyle=style,
                        label=label,
                    )
                )

    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_xlabel(r"$m_{\tilde{t}}$ (GeV)", fontsize=34, loc="right")
    axes.set_ylabel(r"$m_{\tilde{\chi}_1^0}$ (GeV)", fontsize=34)
    axes.xaxis.set_major_locator(MultipleLocator(200))
    axes.yaxis.set_major_locator(MultipleLocator(200))
    axes.xaxis.set_minor_locator(MultipleLocator(50))
    axes.yaxis.set_minor_locator(MultipleLocator(50))
    axes.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=24,
        length=9,
    )
    axes.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=5,
    )
    for spine in axes.spines.values():
        spine.set_linewidth(1.8)
    with plt.rc_context({"font.size": 22}):
        hep.cms.label(
            llabel="Work in progress",
            rlabel=luminosity_label,
            fontsize=27,
            ax=axes,
        )
    multiline_decay_label = bool(decay_label and "\n" in decay_label)
    if decay_label:
        axes.text(
            0.06,
            0.95,
            decay_label,
            transform=axes.transAxes,
            fontsize=20 if multiline_decay_label else 23,
            va="top",
            linespacing=1.15,
        )
    if analysis_label:
        axes.text(
            0.14,
            0.905,
            analysis_label,
            transform=axes.transAxes,
            fontsize=16,
            va="top",
        )
    axes.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="red",
                lw=3.0,
                label=r"Expected $\pm 1\sigma$",
            ),
            *run2_handles,
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.84 if multiline_decay_label else 0.90),
        frameon=False,
        fontsize=19,
        handlelength=2.8,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    figure.savefig(output_png.with_suffix(".pdf"))
    plt.close(figure)
    return True


def comparison(
    merged: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    ratios = []
    absolute = []
    for mass_key, point in merged.get("points", {}).items():
        reference = baseline.get("points", {}).get(mass_key)
        if not reference:
            continue
        new_value = float(point["expected"])
        old_value = float(reference["expected"])
        if old_value <= 0.0:
            continue
        ratios.append(new_value / old_value)
        absolute.append(new_value - old_value)
    if not ratios:
        return {"matched_points": 0}
    ratio_array = np.asarray(ratios, dtype=float)
    absolute_array = np.asarray(absolute, dtype=float)
    return {
        "matched_points": int(ratio_array.size),
        "expected_limit_ratio": {
            "minimum": float(np.min(ratio_array)),
            "median": float(np.median(ratio_array)),
            "maximum": float(np.max(ratio_array)),
        },
        "expected_limit_difference": {
            "minimum": float(np.min(absolute_array)),
            "median": float(np.median(absolute_array)),
            "maximum": float(np.max(absolute_array)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-limits", type=Path)
    parser.add_argument(
        "--campaign-year", choices=("2024", "2025", "2024_2025"), required=True
    )
    parser.add_argument(
        "--topology",
        choices=("T2tt", "T2bW", "T2tb"),
        required=True,
    )
    parser.add_argument("--max-mstop", type=float, default=1800.0)
    parser.add_argument("--min-mlsp", type=float, default=1.0)
    parser.add_argument("--max-mlsp", type=float, default=1200.0)
    args = parser.parse_args()

    build_manifest = json.loads(
        (args.input_dir / "manifest.json").read_text()
    )
    excluded_points = load_excluded_points(args.input_dir)
    original_mass_keys = list(build_manifest["mass_points"])
    mass_keys = [
        mass_key for mass_key in original_mass_keys
        if mass_key not in excluded_points
    ]
    limits = collect_limits(
        args.input_dir / "limits",
        mass_keys,
        args.input_dir / "expected_limits.json",
    )
    highdm_bins = int(build_manifest["model"]["highdm_bins"])
    lowdm_bins = int(build_manifest["model"]["lowdm_bins"])
    baseline = (
        json.loads(args.baseline_limits.read_text())
        if args.baseline_limits
        else {"points": {}}
    )
    output_png = args.input_dir / (
        f"expected_limit_{args.topology.lower()}_"
        f"{args.campaign_year}_highdm{highdm_bins}_lowdm{lowdm_bins}.png"
    )
    contour_complete = False
    if limits["status"] in {"complete", "partial"}:
        contour_complete = plot_contour(
            limits,
            output_png,
            run2_contours=RUN2_CONTOURS[args.topology],
            luminosity_label=LUMINOSITY_LABELS[args.campaign_year],
            analysis_label=None,
            x_max=args.max_mstop,
            y_min=args.min_mlsp,
            y_max=args.max_mlsp,
            decay_label=DECAY_LABELS[args.topology],
        )

    result = {
        "status": (
            "complete"
            if contour_complete and limits["status"] == "complete"
            else "partial"
            if contour_complete
            else "failed"
        ),
        "schema_version": (
            f"canonical_{args.campaign_year}_highdm{highdm_bins}_"
            f"lowdm{lowdm_bins}_limit_v1"
        ),
        "campaign_year": args.campaign_year,
        "topology": args.topology,
        "highdm_bins": highdm_bins,
        "lowdm_bins": lowdm_bins,
        "original_mass_point_count": len(original_mass_keys),
        "mass_point_count": len(mass_keys),
        "excluded_point_count": len(excluded_points),
        "excluded_points": sorted(excluded_points),
        "limits": limits,
        "baseline_limits": (
            str(args.baseline_limits) if args.baseline_limits else None
        ),
        "baseline_comparison": (
            comparison(limits, baseline) if args.baseline_limits else None
        ),
        "contour_png": str(output_png) if contour_complete else None,
        "contour_pdf": (
            str(output_png.with_suffix(".pdf"))
            if contour_complete
            else None
        ),
        "run2_overlay": True,
        "data_mode": "asimov",
    }
    write_json(args.input_dir / "limit_manifest.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "topology": args.topology,
                "collected": limits["collected_point_count"],
                "missing": len(limits["missing_points"]),
                "comparison": result["baseline_comparison"],
                "contour_png": result["contour_png"],
            },
            sort_keys=True,
        )
    )
    return 0 if contour_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
