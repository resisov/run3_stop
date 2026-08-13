#!/usr/bin/env python3
"""Plot and summarize the full selected-entry 2024 GCR overlap scan."""

from __future__ import annotations

import argparse
import hashlib
import html
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def json_values(array: np.ndarray) -> Any:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return [
            float(item) if math.isfinite(float(item)) else None
            for item in values
        ]
    return [json_values(row) for row in values]


def save(fig: Any, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def cms_label(ax: Any) -> None:
    hep.cms.label(ax=ax, loc=0, **CMS_LABEL)


def plot_dr(scan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = {"GJ": "#3b75af", "QCD": "#d55e00"}
    summary = {}
    for process, label in (("GJ", r"$\gamma$+jets"), ("QCD", "QCD")):
        payload = scan["scan"]["process"][process]["dr_distribution"]
        edges = np.asarray(payload["edges"], dtype=float)
        values = np.asarray(payload["sum_abs_w"], dtype=float)
        integral = float(np.sum(values))
        density = values / integral if integral > 0 else values
        ax.stairs(
            density,
            edges,
            label=label,
            linewidth=2.2,
            color=colors[process],
        )
        summary[process] = {
            "minimum": payload["minimum"],
            "median": payload["median"],
            "maximum": payload["maximum"],
            "abs_weight_integral": integral,
        }
    for radius in (0.15, 0.30, 0.40):
        ax.axvline(radius, color="0.45", linestyle="--", linewidth=1.2)
        ax.text(
            radius + 0.008,
            0.97,
            f"$R={radius:.2f}$",
            rotation=90,
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=14,
        )
    ax.set_xlabel(
        r"$\min\,\Delta R(\gamma_{\mathrm{gen}},\,q/g_{\mathrm{status\,23}})$"
    )
    ax.set_ylabel(r"Fraction of $\sum |w|$")
    ax.set_xlim(0.0, 1.5)
    ax.margins(x=0)
    ax.legend(frameon=False)
    cms_label(ax)
    save(fig, output_dir / "hardparton-dr-distribution")
    return summary


def radius_values(
    scan: dict[str, Any], process: str, field_path: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    radii_payload = scan["scan"]["process"][process]["radii"]
    radii = np.asarray(sorted(float(key) for key in radii_payload), dtype=float)
    values = []
    for radius in radii:
        leaf: Any = radii_payload[f"{radius:.2f}"]["inclusive"]
        for field in field_path:
            leaf = leaf[field]
        values.append(finite(leaf))
    return radii, np.asarray(values, dtype=float)


def plot_radius(scan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08},
    )
    ax, lower = axes
    for process, label, color in (
        ("GJ", r"$\gamma$+jets direct keep", "#3b75af"),
        ("QCD", "QCD fragmentation keep", "#d55e00"),
    ):
        radii, unweighted = radius_values(
            scan, process, ["survival_unweighted"]
        )
        _, abs_weighted = radius_values(
            scan, process, ["survival_abs_weighted"]
        )
        ax.plot(
            radii,
            unweighted,
            marker="o",
            color=color,
            label=f"{label}, events",
        )
        ax.plot(
            radii,
            abs_weighted,
            marker="s",
            linestyle="--",
            color=color,
            label=rf"{label}, $\sum|w|$",
        )
    radii, qcd_neff = radius_values(
        scan, "QCD", ["surviving", "neff_abs"]
    )
    _, qcd_max = radius_values(
        scan,
        "QCD",
        ["surviving", "max_event_fraction_abs_sumw"],
    )
    lower.plot(
        radii,
        qcd_neff,
        marker="o",
        color="#009e73",
        label=r"QCD surviving $N_{\mathrm{eff}}^{|w|}$",
    )
    lower.set_ylabel(r"$N_{\mathrm{eff}}^{|w|}$")
    lower2 = lower.twinx()
    lower2.plot(
        radii,
        qcd_max,
        marker="s",
        linestyle="--",
        color="#cc79a7",
        label="Largest-event fraction",
    )
    lower2.set_ylabel(r"largest event / $\sum|w|$", color="#cc79a7")
    lower2.tick_params(axis="y", colors="#cc79a7")
    ax.set_ylabel("Keep fraction")
    ax.set_ylim(0.0, 1.08)
    lower.set_xlabel("Partition radius $R$")
    ax.set_xlim(float(radii[0]), float(radii[-1]))
    ax.margins(x=0)
    lower.margins(x=0)
    ax.legend(frameon=False, fontsize=13, ncol=2)
    lines1, labels1 = lower.get_legend_handles_labels()
    lines2, labels2 = lower2.get_legend_handles_labels()
    lower.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=13)
    cms_label(ax)
    save(fig, output_dir / "radius-survival-and-stability")
    return {
        "radii": radii.tolist(),
        "qcd_surviving_neff_abs": json_values(qcd_neff),
        "qcd_largest_event_fraction_abs_sumw": json_values(qcd_max),
    }


def plot_qcd_bin_stability(
    scan: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    groups = scan["scan"]["generator_dataset_bin"]
    labels = sorted(
        [key for key, value in groups.items() if value["process"] == "QCD"],
        key=lambda text: (
            float(text.split("_")[-1].split("to")[0].replace("3000", "3000"))
        ),
    )
    radii = np.asarray(
        sorted(float(key) for key in groups[labels[0]]["radii"]), dtype=float
    )
    neff = np.full((len(labels), len(radii)), np.nan)
    max_fraction = np.full_like(neff, np.nan)
    for row_index, label in enumerate(labels):
        for column_index, radius in enumerate(radii):
            leaf = groups[label]["radii"][f"{radius:.2f}"]["surviving"]
            neff[row_index, column_index] = finite(leaf.get("neff_abs"))
            max_fraction[row_index, column_index] = finite(
                leaf.get("max_event_fraction_abs_sumw")
            )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    im0 = axes[0].imshow(
        neff,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            radii[0] - 0.025,
            radii[-1] + 0.025,
            -0.5,
            len(labels) - 0.5,
        ),
        cmap="viridis",
    )
    im1 = axes[1].imshow(
        max_fraction,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            radii[0] - 0.025,
            radii[-1] + 0.025,
            -0.5,
            len(labels) - 0.5,
        ),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    pretty = [
        label.replace("QCD_PT_", "").replace("to", "–") for label in labels
    ]
    for ax in axes:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(pretty)
        ax.set_xlim(radii[0] - 0.025, radii[-1] + 0.025)
        ax.margins(x=0)
    axes[0].set_ylabel(r"QCD generator $\hat{p}_{T}$ bin (GeV)")
    axes[1].set_ylabel(r"QCD generator $\hat{p}_{T}$ bin (GeV)")
    axes[1].set_xlabel("Partition radius $R$")
    fig.colorbar(im0, ax=axes[0], label=r"Surviving $N_{\mathrm{eff}}^{|w|}$")
    fig.colorbar(
        im1,
        ax=axes[1],
        label=r"Largest event / surviving $\sum|w|$",
    )
    cms_label(axes[0])
    save(fig, output_dir / "qcd-stability-by-generator-bin")
    return {
        "generator_bins": labels,
        "radii": radii.tolist(),
        "neff_abs": json_values(neff),
        "max_event_fraction_abs_sumw": json_values(max_fraction),
    }


def plot_qcd_ut_stability(
    scan: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    radii_payload = scan["scan"]["process"]["QCD"]["radii"]
    radii = np.asarray(sorted(float(key) for key in radii_payload), dtype=float)
    labels = list(radii_payload[f"{radii[0]:.2f}"]["by_ut"])
    labels = [label for label in labels if label != "overflow"]
    neff = np.full((len(labels), len(radii)), np.nan)
    max_fraction = np.full_like(neff, np.nan)
    for column, radius in enumerate(radii):
        by_ut = radii_payload[f"{radius:.2f}"]["by_ut"]
        for row, label in enumerate(labels):
            leaf = by_ut[label]["surviving"]
            neff[row, column] = finite(leaf.get("neff_abs"))
            max_fraction[row, column] = finite(
                leaf.get("max_event_fraction_abs_sumw")
            )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    im0 = axes[0].imshow(
        neff,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            radii[0] - 0.025,
            radii[-1] + 0.025,
            -0.5,
            len(labels) - 0.5,
        ),
        cmap="viridis",
    )
    im1 = axes[1].imshow(
        max_fraction,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            radii[0] - 0.025,
            radii[-1] + 0.025,
            -0.5,
            len(labels) - 0.5,
        ),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    pretty = [label.replace("to", "–") for label in labels]
    for ax in axes:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(pretty)
        ax.set_xlim(radii[0] - 0.025, radii[-1] + 0.025)
        ax.margins(x=0)
    axes[0].set_ylabel(r"$U_{T}$ bin (GeV)")
    axes[1].set_ylabel(r"$U_{T}$ bin (GeV)")
    axes[1].set_xlabel("Partition radius $R$")
    fig.colorbar(im0, ax=axes[0], label=r"Surviving $N_{\mathrm{eff}}^{|w|}$")
    fig.colorbar(
        im1,
        ax=axes[1],
        label=r"Largest event / surviving $\sum|w|$",
    )
    cms_label(axes[0])
    save(fig, output_dir / "qcd-stability-by-ut")
    return {
        "ut_bins": labels,
        "radii": radii.tolist(),
        "neff_abs": json_values(neff),
        "max_event_fraction_abs_sumw": json_values(max_fraction),
    }


def format_number(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}g}"


def radius_table(scan: dict[str, Any]) -> str:
    rows = []
    for key, qcd_leaf in scan["scan"]["process"]["QCD"]["radii"].items():
        gj_leaf = scan["scan"]["process"]["GJ"]["radii"][key]["inclusive"]
        qcd = qcd_leaf["inclusive"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{format_number(gj_leaf['survival_abs_weighted'])}</td>"
            f"<td>{format_number(qcd['survival_unweighted'])}</td>"
            f"<td>{format_number(qcd['survival_abs_weighted'])}</td>"
            f"<td>{format_number(qcd['surviving']['neff_abs'])}</td>"
            f"<td>{format_number(qcd['surviving']['max_event_fraction_abs_sumw'])}</td>"
            f"<td>{format_number(qcd['surviving']['file_jackknife_relative_sigma'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def top_event_table(scan: dict[str, Any]) -> str:
    events = scan["scan"]["process"]["QCD"]["pre_policy_stability"][
        "top_abs_weight_events"
    ][:10]
    rows = []
    for item in events:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['generator_bin']))}</td>"
            f"<td>{format_number(item['weight'], 5)}</td>"
            f"<td>{format_number(item['min_dr_status23_parton'], 4)}</td>"
            f"<td>{format_number(item['photon_pt'], 5)}</td>"
            f"<td>{format_number(item['ut'], 5)}</td>"
            f"<td>{format_number(item['ht_photon_clean'], 5)}</td>"
            f"<td>{html.escape(str(item['event']))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_report(
    scan: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    plot_summary: dict[str, Any],
) -> None:
    complete = scan["completeness"]["pass"]
    counts = scan["event_counts"]
    gj = scan["scan"]["process"]["GJ"]["pre_policy_stability"]
    qcd = scan["scan"]["process"]["QCD"]["pre_policy_stability"]
    status_text = (
        "전수 join 완전성 PASS" if complete else "전수 join 완전성 FAIL"
    )
    html_text = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024 GCR prompt overlap full selected-entry scan</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;line-height:1.5}}
.status{{padding:14px;border-left:6px solid {'#16833b' if complete else '#b42318'};background:#f5f5f5}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}
img{{width:100%;height:auto;border:1px solid #ddd}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ccc;padding:7px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
code{{background:#eee;padding:2px 5px}} .warn{{background:#fff6df;padding:12px}}
</style></head><body>
<h1>2024 GCR prompt-photon overlap: full selected-entry scan</h1>
<p class="status"><strong>{status_text}</strong><br>
Selection은 기존 <code>real_subset_worker.py</code>가 저장한 exact
<code>feature_GCR</code> 행이며, NanoAOD selection은 재실행하지 않았다.</p>
<h2>Coverage</h2>
<ul>
<li>Exact GCR rows: {counts['exact_gcr_rows']}</li>
<li>Generator diagnostic complete: {counts['gen_diagnostic_complete']}</li>
<li>Eligible prompt photon: {counts['eligible_primary_dr']}</li>
<li>GJets eligible: {counts['by_process']['GJ']['eligible_primary_dr']};
QCD eligible: {counts['by_process']['QCD']['eligible_primary_dr']}</li>
<li>Source files with selected rows: {scan['gen_cache_summary']['source_files_with_selected_rows']}</li>
<li>Unmapped / failed Gen files: {len(scan['unmapped_candidate_paths'])} /
{len(scan['failed_gen_files'])}</li>
</ul>
<h2>Pre-policy weighted stability</h2>
<table><thead><tr><th>Sample</th><th>sumw</th><th>sumw2</th>
<th>N_eff signed</th><th>N_eff |w|</th><th>largest event / sum|w|</th>
<th>file jackknife / yield</th></tr></thead><tbody>
<tr><td>GJets prompt</td><td>{format_number(gj['sumw'],5)}</td>
<td>{format_number(gj['sumw2'],5)}</td><td>{format_number(gj['neff_signed'])}</td>
<td>{format_number(gj['neff_abs'])}</td>
<td>{format_number(gj['max_event_fraction_abs_sumw'])}</td>
<td>{format_number(gj['file_jackknife_relative_sigma'])}</td></tr>
<tr><td>QCD prompt</td><td>{format_number(qcd['sumw'],5)}</td>
<td>{format_number(qcd['sumw2'],5)}</td><td>{format_number(qcd['neff_signed'])}</td>
<td>{format_number(qcd['neff_abs'])}</td>
<td>{format_number(qcd['max_event_fraction_abs_sumw'])}</td>
<td>{format_number(qcd['file_jackknife_relative_sigma'])}</td></tr>
</tbody></table>
<h2>Radius scan</h2>
<table><thead><tr><th>R</th><th>GJets direct keep Σ|w|</th>
<th>QCD frag events</th><th>QCD frag Σ|w|</th><th>QCD surviving N_eff</th>
<th>largest event fraction</th><th>file jackknife / yield</th></tr></thead>
<tbody>{radius_table(scan)}</tbody></table>
<div class="grid">
<p><a href="hardparton-dr-distribution.pdf"><img src="hardparton-dr-distribution.png"></a></p>
<p><a href="radius-survival-and-stability.pdf"><img src="radius-survival-and-stability.png"></a></p>
<p><a href="qcd-stability-by-generator-bin.pdf"><img src="qcd-stability-by-generator-bin.png"></a></p>
<p><a href="qcd-stability-by-ut.pdf"><img src="qcd-stability-by-ut.png"></a></p>
</div>
<h2>Largest QCD prompt events before partition</h2>
<table><thead><tr><th>generator bin</th><th>weight</th><th>min ΔR</th>
<th>photon pT</th><th>U_T</th><th>photon-cleaned H_T</th><th>event</th>
</tr></thead><tbody>{top_event_table(scan)}</tbody></table>
<h2>Decision guardrail</h2>
<p class="warn">이 페이지는 overlap radius를 Data/MC에 맞추어 선택하지 않는다.
완전성, generator 정의, 반경 안정성, generator-bin 및 U_T별
N_eff와 leave-one-out 결과를 함께 만족할 때만 stitching 후보를 제시한다.
QCD prompt 제거는 현재 prefit MC를 낮추므로 GCR rate deficit 자체의
해결책이 아니다.</p>
<p>Input SHA-256: <code>{sha256(input_path)}</code></p>
<p><a href="full-selected-overlap.json">Full machine-readable audit</a> ·
<a href="plot-summary.json">Plot summary</a></p>
</body></html>"""
    (output_dir / "index.html").write_text(html_text)

    report_lines = [
        "# 2024 GCR prompt-photon overlap: full selected-entry scan",
        "",
        f"- Completeness: **{'PASS' if complete else 'FAIL'}**",
        f"- Exact GCR rows: {counts['exact_gcr_rows']}",
        f"- Gen diagnostic complete: {counts['gen_diagnostic_complete']}",
        f"- Eligible prompt photon: {counts['eligible_primary_dr']}",
        (
            "- Eligible GJets/QCD: "
            f"{counts['by_process']['GJ']['eligible_primary_dr']} / "
            f"{counts['by_process']['QCD']['eligible_primary_dr']}"
        ),
        (
            "- Source files with selected rows: "
            f"{scan['gen_cache_summary']['source_files_with_selected_rows']}"
        ),
        "",
        "The event set is the exact materialized `feature_GCR` selection from "
        "`real_subset_worker.py`; no NanoAOD selection was rerun.",
        "",
        "No radius is adopted automatically. The decision must use generator "
        "motivation together with the per-bin Neff and leave-one-out results.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    summary = {
        "schema_version": "gcr_prompt_overlap_full_plot_summary_v1",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "completeness": scan["completeness"],
        "event_counts": counts,
        "plots": plot_summary,
    }
    (output_dir / "plot-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scan = read_json(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = args.output_dir / "full-selected-overlap.json"
    copied.write_bytes(args.input.read_bytes())
    plot_summary = {
        "hardparton_dr": plot_dr(scan, args.output_dir),
        "radius": plot_radius(scan, args.output_dir),
        "qcd_generator_bin": plot_qcd_bin_stability(
            scan, args.output_dir
        ),
        "qcd_ut": plot_qcd_ut_stability(scan, args.output_dir),
    }
    write_report(scan, args.input, args.output_dir, plot_summary)
    print(args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
