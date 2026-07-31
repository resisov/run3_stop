#!/usr/bin/env python3
"""Publish the validated DY2E-only 2024 DYCR result to GitHub Pages.

This deliberately excludes DY2M and channel-combined AN results.  The current
DY2M AN input reconstruction is being rerun after correcting an integer-versus-
boolean Awkward selection issue; DY2E is independent of that code path.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")

CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def dy2e_summary(measurement: dict[str, Any], factors: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"regimes": {}, "status": "preliminary_dy2e_only"}
    for regime in ("highdm", "lowdm"):
        direct = measurement["factors"][regime]
        matrix = factors["RZ"][regime]["channels"]["DY2E"]
        result["regimes"][regime] = {
            "inclusive": {
                key: finite_number(direct["inclusive"]["DY2E"][key], f"{regime}.inclusive.{key}")
                for key in (
                    "value",
                    "stat",
                    "btag",
                    "other_experimental",
                    "total",
                    "prefit_data_mc",
                    "dy_purity",
                )
            },
            "by_nb": {},
            "ut_gof": measurement["goodness_of_fit"]["ut"][f"{regime}_ut_dy2e"],
        }
        for nb in ("Nb1", "Nb2plus"):
            direct_nb = direct["by_nb"][nb]["DY2E"]
            matrix_nb = matrix[nb]
            result["regimes"][regime]["by_nb"][nb] = {
                "direct": {
                    key: finite_number(direct_nb[key], f"{regime}.{nb}.direct.{key}")
                    for key in ("value", "stat", "total", "prefit_data_mc")
                },
                "matrix": {
                    key: finite_number(matrix_nb[key], f"{regime}.{nb}.matrix.{key}")
                    for key in ("RZ", "RZ_stat", "RT", "RT_stat")
                },
                "matrix_inputs": matrix_nb["inputs"],
            }
    result["lowdm_search_bin_gof"] = measurement["goodness_of_fit"]["search_bins"][
        "lowdm_search_bins_dy2e"
    ]
    return result


def plot_factors(summary: dict[str, Any], output: Path) -> None:
    records = [
        ("highdm", "Nb1", "High-$\\Delta m$\n$N_b=1$"),
        ("highdm", "Nb2plus", "High-$\\Delta m$\n$N_b\\geq2$"),
        ("lowdm", "Nb1", "Low-$\\Delta m$\n$N_b=1$"),
        ("lowdm", "Nb2plus", "Low-$\\Delta m$\n$N_b\\geq2$"),
    ]
    x = np.arange(len(records), dtype=float)
    direct = np.asarray(
        [summary["regimes"][regime]["by_nb"][nb]["direct"]["value"] for regime, nb, _ in records]
    )
    direct_error = np.asarray(
        [summary["regimes"][regime]["by_nb"][nb]["direct"]["total"] for regime, nb, _ in records]
    )
    matrix = np.asarray(
        [summary["regimes"][regime]["by_nb"][nb]["matrix"]["RZ"] for regime, nb, _ in records]
    )
    matrix_error = np.asarray(
        [summary["regimes"][regime]["by_nb"][nb]["matrix"]["RZ_stat"] for regime, nb, _ in records]
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axhline(1.0, color="black", lw=1.3, ls="--")
    ax.errorbar(
        x - 0.08,
        direct,
        yerr=direct_error,
        fmt="o",
        color="#0072B2",
        ms=7,
        capsize=3,
        label=r"Direct residual $R_Z$ (total unc.)",
    )
    ax.errorbar(
        x + 0.08,
        matrix,
        yerr=matrix_error,
        fmt="s",
        color="#D55E00",
        ms=7,
        capsize=3,
        label=r"On/off-$Z$ fit $R_Z$ (stat. unc.)",
    )
    ax.set_xlim(-0.5, len(records) - 0.5)
    ax.set_ylim(0.35, 1.15)
    ax.set_xticks(x, [label for _, _, label in records])
    ax.tick_params(axis="x", labelsize=19)
    ax.set_ylabel(r"DY normalization factor $R_Z$")
    ax.legend(loc="lower left", fontsize=15)
    ax.grid(axis="y", color="0.85", lw=0.8)
    hep.cms.label(ax=ax, **CMS_LABEL)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_html(summary: dict[str, Any], generated: str) -> str:
    rows = []
    labels = {
        ("highdm", "Nb1"): r"High-&Delta;m, N<sub>b</sub> = 1",
        ("highdm", "Nb2plus"): r"High-&Delta;m, N<sub>b</sub> &ge; 2",
        ("lowdm", "Nb1"): r"Low-&Delta;m, N<sub>b</sub> = 1",
        ("lowdm", "Nb2plus"): r"Low-&Delta;m, N<sub>b</sub> &ge; 2",
    }
    for regime in ("highdm", "lowdm"):
        for nb in ("Nb1", "Nb2plus"):
            cell = summary["regimes"][regime]["by_nb"][nb]
            direct = cell["direct"]
            matrix = cell["matrix"]
            rows.append(
                "<tr>"
                f"<td>{labels[(regime, nb)]}</td>"
                f"<td>{fmt(direct['prefit_data_mc'])}</td>"
                f"<td>{fmt(direct['value'])} &plusmn; {fmt(direct['total'])}</td>"
                f"<td>{fmt(matrix['RZ'])} &plusmn; {fmt(matrix['RZ_stat'])}</td>"
                f"<td>{fmt(matrix['RT'])} &plusmn; {fmt(matrix['RT_stat'])}</td>"
                "</tr>"
            )

    high = summary["regimes"]["highdm"]
    low = summary["regimes"]["lowdm"]
    search = summary["lowdm_search_bin_gof"]
    cards = [
        ("plots/dy2e_rz_comparison.png", "plots/dy2e_rz_comparison.pdf", "DY2E normalization factors", "Direct background-subtracted factors and the simultaneous on/off-Z solution agree within their uncertainties in every category."),
        ("plots/highdm_ut_dy2e.png", "plots/highdm_ut_dy2e.pdf", "High-Δm DYCR: U<sub>T</sub>", "The DY-only correction lowers the prefit prediction. The U<sub>T</sub> goodness-of-fit improves from χ²/ndof = 2.52 to 1.09."),
        ("plots/lowdm_ut_dy2e.png", "plots/lowdm_ut_dy2e.pdf", "Low-Δm DYCR: U<sub>T</sub>", "The low-Δm U<sub>T</sub> agreement improves from χ²/ndof = 4.76 to 0.57. The last populated bins remain statistically limited."),
        ("plots/lowdm_search_bins_dy2e.png", "plots/lowdm_search_bins_dy2e.pdf", "Low-Δm DYCR: search bins", "Across 34 bins the correction improves χ²/ndof from 1.98 to 1.35. Individual sparse bins can still fluctuate strongly."),
        ("plots/mll_highdm_dy2e_nb1.png", "plots/mll_highdm_dy2e_nb1.pdf", "High-Δm, N<sub>b</sub> = 1: m<sub>ee</sub>", "On-Z and off-Z yields used by the simultaneous R<sub>Z</sub>–R<sub>T</sub> extraction."),
        ("plots/mll_highdm_dy2e_nb2plus.png", "plots/mll_highdm_dy2e_nb2plus.pdf", "High-Δm, N<sub>b</sub> ≥ 2: m<sub>ee</sub>", "The non-DY fraction grows in the off-Z sideband and is fitted through R<sub>T</sub>."),
        ("plots/mll_lowdm_dy2e_nb1.png", "plots/mll_lowdm_dy2e_nb1.pdf", "Low-Δm, N<sub>b</sub> = 1: m<sub>ee</sub>", "The Z window is shown explicitly; no postfit result is used."),
        ("plots/mll_lowdm_dy2e_nb2plus.png", "plots/mll_lowdm_dy2e_nb2plus.pdf", "Low-Δm, N<sub>b</sub> ≥ 2: m<sub>ee</sub>", "This is the statistically weakest DY2E b-tag category and has the largest R<sub>Z</sub> uncertainty."),
    ]
    card_html = "\n".join(
        f'''<article class="plot-card"><a href="{pdf}"><img loading="lazy" src="{png}" alt="{html.escape(caption)}"></a><div class="plot-copy"><h3>{caption}</h3><p>{description}</p><a class="download" href="{pdf}">PDF</a></div></article>'''
        for png, pdf, caption, description in cards
    )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2024 DY2E-only DYCR result</title>
  <style>
    :root {{ --ink:#18202c; --muted:#5b6674; --line:#dbe1e8; --blue:#0b6fad; --orange:#d75a16; --paper:#fff; --wash:#f3f6f9; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--wash); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.55; }}
    header {{ color:white; background:linear-gradient(120deg,#111c2e,#0b547d); padding:56px 24px 46px; }}
    header .inner, main {{ width:min(1220px,calc(100% - 36px)); margin:auto; }}
    .eyebrow {{ margin:0 0 10px; letter-spacing:.12em; text-transform:uppercase; color:#bfe6fb; font-weight:700; font-size:.82rem; }}
    h1 {{ margin:0; font-size:clamp(2rem,5vw,4.25rem); line-height:1.04; letter-spacing:-.045em; }}
    header p {{ max-width:850px; color:#e3eff7; font-size:1.08rem; margin:18px 0 0; }}
    .status {{ display:inline-block; margin-top:22px; padding:7px 12px; border:1px solid #73c6eb; border-radius:999px; color:#dff4ff; font-weight:700; }}
    main {{ padding:28px 0 72px; }}
    .callout, section {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:22px; margin:0 0 22px; box-shadow:0 8px 24px rgba(17,28,46,.045); }}
    .callout {{ border-left:5px solid var(--orange); }}
    .callout strong {{ color:#973c0e; }}
    h2 {{ margin:0 0 12px; font-size:1.45rem; letter-spacing:-.02em; }}
    p {{ margin:7px 0; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:0 0 22px; }}
    .kpi {{ background:white; border:1px solid var(--line); border-radius:12px; padding:18px; }}
    .kpi b {{ display:block; font-size:1.65rem; color:var(--blue); }}
    .kpi span {{ color:var(--muted); font-size:.92rem; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#eef4f7; font-size:.9rem; }}
    .plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .plot-card {{ background:white; border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:0 8px 24px rgba(17,28,46,.045); }}
    .plot-card img {{ display:block; width:100%; aspect-ratio:1/1; object-fit:contain; background:white; }}
    .plot-copy {{ padding:16px 18px 18px; border-top:1px solid var(--line); }}
    .plot-copy h3 {{ margin:0 0 6px; }}
    .plot-copy p {{ color:var(--muted); }}
    .download {{ display:inline-block; margin-top:7px; color:var(--blue); font-weight:700; text-decoration:none; }}
    code {{ overflow-wrap:anywhere; }}
    footer {{ color:var(--muted); padding:18px 0 0; font-size:.88rem; }}
    @media(max-width:850px) {{ .kpis,.plots {{ grid-template-columns:1fr; }} header {{ padding-top:38px; }} }}
  </style>
</head>
<body>
<header><div class="inner">
  <p class="eyebrow">CMS Run 3 · all-hadronic stop search</p>
  <h1>2024 DY2E-only DYCR result</h1>
  <p>Background-subtracted DY normalization and shape diagnostics using the new DYto2E-4Jets inclusive sample with the measured 2024 b-tag scale factors. These are prefit control-region results.</p>
  <span class="status">Preliminary · DY2E validated</span>
</div></header>
<main>
  <div class="callout"><strong>Scope boundary:</strong> DY2M and electron–muon combined factors are intentionally not shown. A muon-pair reconstruction selector was corrected and that channel is being rerun. The DY2E stream and all results on this page do not use the affected selector.</div>
  <div class="kpis">
    <div class="kpi"><b>{fmt(high['inclusive']['value'])} ± {fmt(high['inclusive']['total'])}</b><span>High-Δm inclusive DY2E factor</span></div>
    <div class="kpi"><b>{fmt(low['inclusive']['value'])} ± {fmt(low['inclusive']['total'])}</b><span>Low-Δm inclusive DY2E factor</span></div>
    <div class="kpi"><b>2.52 → 1.09</b><span>High-Δm U<sub>T</sub> χ²/ndof</span></div>
    <div class="kpi"><b>4.76 → 0.57</b><span>Low-Δm U<sub>T</sub> χ²/ndof</span></div>
  </div>
  <section>
    <h2>What the DY2E result says</h2>
    <p>The prefit DY prediction is high: the inclusive data/MC values are {fmt(high['inclusive']['prefit_data_mc'])} in High-Δm and {fmt(low['inclusive']['prefit_data_mc'])} in Low-Δm. After subtracting non-DY MC, the measured DY scale factors are {fmt(high['inclusive']['value'])} ± {fmt(high['inclusive']['total'])} and {fmt(low['inclusive']['value'])} ± {fmt(low['inclusive']['total'])}, respectively.</p>
    <p>The normalization correction materially improves the U<sub>T</sub> shape agreement. It is not a postfit result. Fine search-bin closure also improves, from χ²/ndof = {fmt(search['prefit']['chi2_per_ndof'],2)} to {fmt(search['corrected']['chi2_per_ndof'],2)}, but sparse bins remain statistically unstable and should not drive a stronger claim.</p>
  </section>
  <section>
    <h2>Normalization cross-check</h2>
    <p>The direct residual method fixes non-DY MC to its nominal prediction. The on/off-Z matrix method simultaneously solves for DY normalization R<sub>Z</sub> and an effective non-DY normalization R<sub>T</sub>. Their R<sub>Z</sub> results agree within uncertainties.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Category</th><th>Prefit data/MC</th><th>Direct R<sub>Z</sub> (total)</th><th>Matrix R<sub>Z</sub> (stat.)</th><th>Matrix R<sub>T</sub> (stat.)</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
  </section>
  <div class="plots">{card_html}</div>
  <section style="margin-top:22px">
    <h2>Inputs and interpretation</h2>
    <p>DY MC is the new <code>DYto2E-4Jets_Bin-MLL-50</code> inclusive sample. No legacy DY p<sub>T</sub><sup>ll</sup>-binned dataset is included. Non-DY contamination is taken from the normalized 2024 MC components, with b-tag, pileup, electron trigger, and electron identification variations propagated to the direct factor.</p>
    <p>Same-region corrected plots are goodness-of-fit diagnostics, not independent closure tests. The page therefore reports both the prefit and corrected distributions and does not claim a blinded-SR validation.</p>
  </section>
  <footer>Generated {html.escape(generated)} · machine-readable values: <a href="page_summary.json">page_summary.json</a></footer>
</main>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve()
    measurement_path = report_dir / "measurement.json"
    factors_path = report_dir / "an_zinv_factors_2024.json"
    measurement = read_json(measurement_path)
    factors = read_json(factors_path)
    if measurement.get("status") != "complete":
        raise ValueError("direct DY measurement is not complete")
    if factors.get("status") != "complete":
        raise ValueError("AN factor payload is not complete")

    summary = dy2e_summary(measurement, factors)
    summary["provenance"] = {
        "measurement_sha256": sha256_file(measurement_path),
        "an_factor_sha256": sha256_file(factors_path),
        "new_dy_datasets": measurement["input_audit"]["new_dy_datasets"],
        "ptll_dataset_count": measurement["input_audit"]["ptll_dataset_count"],
        "dy2m_and_combined_published": False,
    }

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_factors(summary, plot_dir / "dy2e_rz_comparison")

    sources = {
        "highdm_ut_dy2e": report_dir / "plots" / "highdm_ut_dy2e",
        "lowdm_ut_dy2e": report_dir / "plots" / "lowdm_ut_dy2e",
        "lowdm_search_bins_dy2e": report_dir / "plots" / "lowdm_search_bins_dy2e",
        "mll_highdm_dy2e_nb1": report_dir / "plots" / "an_zinv" / "mll_highdm_dy2e_nb1",
        "mll_highdm_dy2e_nb2plus": report_dir / "plots" / "an_zinv" / "mll_highdm_dy2e_nb2plus",
        "mll_lowdm_dy2e_nb1": report_dir / "plots" / "an_zinv" / "mll_lowdm_dy2e_nb1",
        "mll_lowdm_dy2e_nb2plus": report_dir / "plots" / "an_zinv" / "mll_lowdm_dy2e_nb2plus",
    }
    copied = []
    for name, stem in sources.items():
        for suffix in (".png", ".pdf"):
            source = stem.with_suffix(suffix)
            if not source.is_file():
                raise FileNotFoundError(source)
            target = plot_dir / f"{name}{suffix}"
            shutil.copy2(source, target)
            copied.append(target.relative_to(output_dir).as_posix())
    summary["published_files"] = sorted(
        copied + ["plots/dy2e_rz_comparison.png", "plots/dy2e_rz_comparison.pdf"]
    )
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary["generated_utc"] = generated
    write_json(output_dir / "page_summary.json", summary)
    (output_dir / "index.html").write_text(build_html(summary, generated))
    print(json.dumps({"output_dir": str(output_dir), "files": len(summary["published_files"]) + 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
