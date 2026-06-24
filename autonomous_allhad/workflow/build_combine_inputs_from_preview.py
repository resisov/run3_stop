#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from array import array
from pathlib import Path
from typing import Any

import numpy as np

REGION_TO_SIGNAL_REGION = {
    "cat2_LLCR_highDeltaM": "LLCR",
    "cat3_QCDCR_highDeltaM": "QCDCR",
    "cat7_SR_highDeltaM": "SR",
}
VARIABLE_TO_SIGNAL_VARIABLE = {
    "metpt": "met",
    "ht": "ht",
    "njet": "njet",
    "nb": "nb_medium",
    "min_dphi": "min_dphi4",
}
LUMI_NAME = "Lumi_2024"
LUMI_LNN = 1.016
LUMI_FRAC = 0.016
BACKGROUND_NAME = "background"
BKG_SYST_NAME = "BkgSyst"
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sanitize(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return out or "unnamed"


def signal_process_name(key: str) -> str:
    return "sig_" + sanitize(key)


def stable_path(path: Path) -> str:
    return str(path.absolute()).replace("/eos/home-t/taiwoo", "/eos/user/t/taiwoo")


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.match(r"mStop(\d+)_mLSP(\d+)$", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def finite_edges(edges: list[Any]) -> np.ndarray:
    vals = []
    for item in edges:
        if item == "inf":
            vals.append(float("inf"))
        else:
            vals.append(float(item))
    finite = []
    for idx, val in enumerate(vals):
        if math.isinf(val):
            if len(finite) >= 2:
                finite.append(finite[-1] + (finite[-1] - finite[-2]))
            else:
                finite.append(finite[-1] + 1.0 if finite else 1.0)
        else:
            finite.append(val)
    return np.asarray(finite, dtype=float)


def collapse_to_template(raw_values: list[float], raw_sumw2: list[float], raw_edges: list[float], template_edges: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    out_edges = finite_edges(template_edges)
    nbin = len(out_edges) - 1
    values = np.zeros(nbin, dtype=float)
    sumw2 = np.zeros(nbin, dtype=float)
    if not raw_edges or not raw_values:
        return values, sumw2
    raw_edges_np = np.asarray([float(x) for x in raw_edges], dtype=float)
    raw_vals = np.asarray(raw_values, dtype=float)
    raw_s2 = np.asarray(raw_sumw2 or [0.0] * len(raw_vals), dtype=float)
    for idx, val in enumerate(raw_vals):
        if idx >= len(raw_edges_np) - 1:
            break
        low = raw_edges_np[idx]
        target = None
        for j in range(nbin):
            hi = out_edges[j + 1]
            if low >= out_edges[j] and (low < hi or (j == nbin - 1 and low >= out_edges[j])):
                target = j
                break
        if target is None and low >= out_edges[-2]:
            target = nbin - 1
        if target is not None:
            values[target] += float(val)
            if idx < len(raw_s2):
                sumw2[target] += float(raw_s2[idx])
    return values, sumw2


def make_hist(name: str, values: np.ndarray, sumw2: np.ndarray, edges: np.ndarray):
    import ROOT

    hist = ROOT.TH1D(name, name, len(values), array("d", [float(x) for x in edges]))
    hist.Sumw2()
    for idx, val in enumerate(values, start=1):
        hist.SetBinContent(idx, float(max(val, 0.0)))
        err2 = float(sumw2[idx - 1]) if idx - 1 < len(sumw2) else 0.0
        hist.SetBinError(idx, math.sqrt(max(err2, 0.0)))
    return hist


def write_hist(directory, name: str, values: np.ndarray, sumw2: np.ndarray, edges: np.ndarray) -> None:
    directory.cd()
    hist = make_hist(name, values, sumw2, edges)
    hist.Write(name, 1)


def signal_histogram(signal_histograms: dict[str, Any], signal_meta: dict[str, Any], mass_key: str, region: str, variable: str, template_edges: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    sig_region = REGION_TO_SIGNAL_REGION.get(region)
    sig_variable = VARIABLE_TO_SIGNAL_VARIABLE.get(variable)
    nbin = len(template_edges) - 1
    if not sig_region or not sig_variable:
        return np.zeros(nbin), np.zeros(nbin)
    raw = (((signal_histograms.get(mass_key) or {}).get(sig_region) or {}).get(sig_variable) or {})
    factor = float((signal_meta.get(mass_key) or {}).get("normalization_factor") or 0.0)
    vals, s2 = collapse_to_template(raw.get("raw_weighted") or [], raw.get("raw_sumw2") or [], raw.get("bins") or [], template_edges)
    return vals * factor, s2 * factor * factor


def load_inputs(fit_summary_path: Path, signal_cutflows_path: Path, signal_yields_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fit = read_json(fit_summary_path)
    sig_cut = read_json(signal_cutflows_path)
    sig_yields = read_json(signal_yields_path)
    signal_histograms = (((sig_cut.get("signal_histograms") or {}).get("histograms")) or {})
    signal_meta = sig_yields.get("mass_points") or {}
    return fit, signal_histograms, signal_meta


def build_root(fit: dict[str, Any], signal_histograms: dict[str, Any], signal_meta: dict[str, Any], mass_keys: list[str], output_root: Path, data_mode: str) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    templates = fit.get("templates") or {}
    summary: dict[str, Any] = {"channels": {}, "signals": {}}
    try:
        for region, rec in templates.items():
            if rec.get("status") != "complete":
                continue
            directory = root_file.mkdir(region)
            edges = finite_edges(rec.get("plot_bin_edges") or [])
            bkg = np.asarray(rec.get("background_total") or [], dtype=float)
            stat = np.asarray(rec.get("background_stat_unc") or [], dtype=float)
            syst = np.asarray(rec.get("background_syst_unc") or [], dtype=float)
            lumi = LUMI_FRAC * bkg
            non_lumi2 = np.maximum(syst * syst - lumi * lumi, 0.0)
            non_lumi = np.sqrt(non_lumi2)
            data = bkg if data_mode == "asimov" else np.asarray(rec.get("data") or [], dtype=float)
            write_hist(directory, "data_obs", data, np.maximum(data, 0.0), edges)
            write_hist(directory, BACKGROUND_NAME, bkg, stat * stat, edges)
            write_hist(directory, f"{BACKGROUND_NAME}_{BKG_SYST_NAME}Up", bkg + non_lumi, stat * stat, edges)
            write_hist(directory, f"{BACKGROUND_NAME}_{BKG_SYST_NAME}Down", np.maximum(bkg - non_lumi, MIN_BIN), stat * stat, edges)
            summary["channels"][region] = {
                "variable": rec.get("variable"),
                "background_yield": float(np.sum(bkg)),
                "data_mode": data_mode,
                "data_yield": float(np.sum(data)),
                "non_lumi_background_syst_sum": float(np.sum(non_lumi)),
                "lumi_background_syst_sum": float(np.sum(lumi)),
            }
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                sig, sig_s2 = signal_histogram(signal_histograms, signal_meta, mass_key, region, str(rec.get("variable")), rec.get("plot_bin_edges") or [])
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"channels": {}, "process": proc})["channels"][region] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def datacard_text(template_root: Path, fit: dict[str, Any], mass_key: str, root_summary: dict[str, Any], signal_meta: dict[str, Any], auto_mc_stats: int) -> str:
    channels = [region for region, rec in (fit.get("templates") or {}).items() if rec.get("status") == "complete"]
    proc = signal_process_name(mass_key)
    columns: list[tuple[str, str, int]] = []
    for region in channels:
        sig_yield = float(((root_summary.get("signals") or {}).get(mass_key) or {}).get("channels", {}).get(region, 0.0))
        if sig_yield > 0.0:
            columns.append((region, proc, 0))
        columns.append((region, BACKGROUND_NAME, 1))
    rel_root = stable_path(template_root)
    lines = [
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {rel_root} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
        "------------",
        "bin " + " ".join(channels),
        "observation " + " ".join(["-1"] * len(channels)),
        "------------",
        "bin " + " ".join(c[0] for c in columns),
        "process " + " ".join(c[1] for c in columns),
        "process " + " ".join(str(c[2]) for c in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
        BKG_SYST_NAME + " shape " + " ".join("1" if c[1] == BACKGROUND_NAME else "-" for c in columns),
        LUMI_NAME + " lnN " + " ".join(f"{LUMI_LNN:.3f}" for _ in columns),
    ]
    xsec_unc = (signal_meta.get(mass_key) or {}).get("xsec_uncertainty_relative")
    if xsec_unc is not None:
        factor = 1.0 + float(xsec_unc)
        lines.append("SignalTheory lnN " + " ".join(f"{factor:.4f}" if c[1] == proc else "-" for c in columns))
    lines.extend([
        f"* autoMCStats {auto_mc_stats}",
        "# Expected-limit card generated from partial preview fit templates.",
        "# BkgSyst is the aggregate non-lumi background shape envelope from fit_template_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def write_datacards(fit: dict[str, Any], signal_meta: dict[str, Any], root_summary: dict[str, Any], mass_keys: list[str], template_root: Path, output_dir: Path, auto_mc_stats: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(datacard_text(template_root, fit, mass_key, root_summary, signal_meta, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def find_combine() -> str | None:
    from shutil import which
    return which("combine")


def write_limit_runner(cards: dict[str, str], output_dir: Path, runner_path: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "COMBINE=${COMBINE:-combine}", f"OUTDIR={output_dir}", "mkdir -p \"$OUTDIR\""]
    for mass_key, card in sorted(cards.items()):
        lines.append(f"echo '[combine] {mass_key}'")
        lines.append(f"(cd \"$OUTDIR\" && \"$COMBINE\" -M AsymptoticLimits --run blind -n _{mass_key} \"{stable_path(Path.cwd() / Path(card))}\") | tee \"$OUTDIR/log_{mass_key}.txt\"")
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("\n".join(lines) + "\n")
    runner_path.chmod(0o755)


def parse_limit_file(path: Path) -> dict[str, float] | None:
    rows = []
    try:
        import uproot

        with uproot.open(path) as f:
            tree = f.get("limit")
            if tree is not None:
                quantiles = tree["quantileExpected"].array(library="np")
                limits = tree["limit"].array(library="np")
                rows = [(float(q), float(v)) for q, v in zip(quantiles, limits)]
    except Exception:
        rows = []
    if not rows:
        try:
            import ROOT

            f = ROOT.TFile.Open(str(path))
            tree = f.Get("limit") if f else None
            if not tree:
                return None
            for entry in tree:
                rows.append((float(entry.quantileExpected), float(entry.limit)))
            f.Close()
        except Exception:
            return None
    if not rows:
        return None
    labels = {0.025: "expected_m2", 0.16: "expected_m1", 0.5: "expected", 0.84: "expected_p1", 0.975: "expected_p2"}
    out = {}
    for q, val in rows:
        best = min(labels, key=lambda x: abs(x - q))
        if abs(best - q) < 0.02:
            out[labels[best]] = val
    return out or None


def collect_limits(limit_dir: Path, mass_keys: list[str], output_json: Path) -> dict[str, Any]:
    results = {}
    for mass_key in mass_keys:
        candidates = sorted(limit_dir.glob(f"higgsCombine_{mass_key}.AsymptoticLimits*.root")) + sorted(limit_dir.glob(f"higgsCombine_{mass_key}.root")) + sorted(limit_dir.glob(f"higgsCombine_{mass_key}*.root"))
        parsed = None
        for path in candidates:
            parsed = parse_limit_file(path)
            if parsed:
                break
        if parsed:
            mstop, mlsp = parse_mass_key(mass_key)
            parsed.update({"mStop": mstop, "mLSP": mlsp})
            results[mass_key] = parsed
    payload = {"status": "complete" if results else "no_combine_outputs", "points": results}
    write_json(output_json, payload)
    return payload


def plot_contour(limit_payload: dict[str, Any], output_png: Path) -> bool:
    records = list((limit_payload.get("points") or {}).values())
    points = [rec for rec in records if "expected" in rec and float(rec["expected"]) > 0]
    if not points:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    from scipy.interpolate import griddata

    # Match the reference view, use all generated mass points, and mask unsupported/off-shell regions.
    xmin, xmax = 600.0, 1500.0
    ymin, ymax = 0.0, 1500.0
    top_mass = 172.5
    xi = np.linspace(xmin, xmax, 260)
    yi = np.linspace(ymin, ymax, 260)
    xx, yy = np.meshgrid(xi, yi)
    offshell_mask = yy > (xx - top_mass)

    def interpolated_log_grid(quantity: str) -> np.ma.MaskedArray | None:
        vals = []
        for rec in records:
            val = rec.get(quantity)
            if val is None or float(val) <= 0:
                continue
            x = float(rec["mStop"])
            y = float(rec["mLSP"])
            vals.append((x, y, math.log10(float(val))))
        if not vals:
            return None
        xs = np.asarray([v[0] for v in vals])
        ys = np.asarray([v[1] for v in vals])
        zs = np.asarray([v[2] for v in vals])
        linear = griddata((xs, ys), zs, (xx, yy), method="linear")
        return np.ma.array(linear, mask=np.isnan(linear) | offshell_mask)

    expected_grid = interpolated_log_grid("expected")
    if expected_grid is None or expected_grid.count() == 0:
        return False
    minus1_grid = interpolated_log_grid("expected_m1")
    plus1_grid = interpolated_log_grid("expected_p1")

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.12, top=0.89)

    color_min, color_max = -1.5, 1.5
    plot_grid = np.ma.clip(expected_grid, color_min, color_max)
    filled = ax.contourf(
        xx,
        yy,
        plot_grid,
        levels=np.linspace(color_min, color_max, 121),
        cmap="viridis",
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.055, fraction=0.055)
    cbar.set_label(
        r"$\log_{10}$ (expected 95% CL limit on $\sigma/\sigma_{\mathrm{theory}}$)",
        fontsize=15,
        rotation=90,
        labelpad=18,
    )
    cbar.set_ticks(np.arange(color_min, color_max + 0.001, 0.5))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=20, direction="in", length=7, width=1.2)
    cbar.outline.set_linewidth(1.8)

    xs = np.asarray([float(p["mStop"]) for p in points])
    ys = np.asarray([float(p["mLSP"]) for p in points])
    ax.scatter(xs, ys, s=9, c="black", alpha=0.35, linewidths=0, zorder=3)

    diag_x = np.linspace(xmin, xmax, 400)
    diag_y = diag_x - top_mass
    keep = (diag_y >= ymin) & (diag_y <= ymax)
    ax.plot(diag_x[keep], diag_y[keep], color="0.45", linestyle=":", linewidth=1.1, zorder=4)

    ax.contour(xx, yy, expected_grid, levels=[0.0], colors="red", linewidths=3.0, zorder=6)
    for band_grid in (minus1_grid, plus1_grid):
        if band_grid is not None and band_grid.count() > 0:
            ax.contour(xx, yy, band_grid, levels=[0.0], colors="red", linewidths=1.7, linestyles="--", zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$m_{\tilde{t}}$ (GeV)", fontsize=25, loc="right")
    ax.set_ylabel(r"$m_{\tilde{\chi}_1^0}$ (GeV)", fontsize=25)
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

    legend_handles = [
        Line2D([0], [0], color="red", lw=3.0, label="Expected"),
        Line2D([0], [0], color="red", lw=1.8, linestyle="--", label=r"Expected $\pm1\sigma_{\mathrm{exp}}$"),
        Line2D([0], [0], color="0.45", lw=1.1, linestyle=":", label=r"$m_{\tilde{\chi}_1^0}=m_{\tilde{t}}-m_t$"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.90), frameon=False, fontsize=14, handlelength=2.8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def select_mass_points(signal_meta: dict[str, Any], max_points: int | None, only: list[str] | None) -> list[str]:
    keys = []
    for key, rec in signal_meta.items():
        if only and key not in only:
            continue
        try:
            mstop, mlsp = parse_mass_key(key)
        except ValueError:
            continue
        if rec.get("normalization_status") and "normalized" not in str(rec.get("normalization_status")):
            continue
        if mlsp >= mstop:
            continue
        keys.append((mstop, mlsp, key))
    keys.sort()
    selected = [key for _, _, key in keys]
    if max_points is not None:
        selected = selected[:max_points]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-template-summary", default="docs/partial_merge_preview_sf_unc_v2_combined_data2_v2_20260622/fit_template_summary.json")
    parser.add_argument("--signal-cutflows", default="autonomous_allhad/outputs/signal_cutflows.json")
    parser.add_argument("--signal-yields", default="autonomous_allhad/outputs/signal_yields_by_mass.json")
    parser.add_argument("--output-dir", default="analysis/combine/partial_merge_sf_unc_v2_combined_data2_v2_20260622")
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_stop_2024_partial.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    manifest_path = outdir / "combine_input_manifest.json"
    limit_json = outdir / "expected_limits.json"
    contour_png = outdir / "expected_limit_contour.png"

    fit, signal_histograms, signal_meta = load_inputs(Path(args.fit_template_summary), Path(args.signal_cutflows), Path(args.signal_yields))
    mass_keys = select_mass_points(signal_meta, args.max_points, args.only)
    if not mass_keys:
        raise SystemExit("no normalized signal mass points selected")

    if not args.collect_only:
        root_summary = build_root(fit, signal_histograms, signal_meta, mass_keys, template_root, args.data_mode)
        cards = write_datacards(fit, signal_meta, root_summary, mass_keys, template_root, datacard_dir, args.auto_mc_stats)
        write_limit_runner(cards, limit_dir, runner)
    else:
        cards = {key: str(datacard_dir / f"datacard_{key}.txt") for key in mass_keys}
        root_summary = {}

    limit_payload = collect_limits(limit_dir, mass_keys, limit_json)
    contour_written = plot_contour(limit_payload, contour_png)
    manifest = {
        "status": "combine_ready" if find_combine() is None else "combine_available",
        "combine_path": find_combine(),
        "fit_template_summary": str(Path(args.fit_template_summary)),
        "signal_cutflows": str(Path(args.signal_cutflows)),
        "signal_yields": str(Path(args.signal_yields)),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "mass_points": mass_keys,
        "mass_point_count": len(mass_keys),
        "data_mode": args.data_mode,
        "lumi_uncertainty": {"name": LUMI_NAME, "lnN": LUMI_LNN, "fraction": LUMI_FRAC},
        "background_syst_policy": "BkgSyst shape uses aggregate non-lumi background uncertainty from fit_template_summary.json; Lumi_2024 is separate lnN.",
        "limit_collection_status": limit_payload.get("status"),
        "contour_png": str(contour_png) if contour_written else None,
        "root_summary": root_summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "mass_points": len(mass_keys), "template_root": str(template_root), "datacards": len(cards), "combine_path": manifest["combine_path"], "limit_status": manifest["limit_collection_status"], "contour": manifest["contour_png"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
