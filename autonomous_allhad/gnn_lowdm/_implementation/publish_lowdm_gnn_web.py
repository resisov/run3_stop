#!/usr/bin/env python3
"""Publish the validated 2024/2025 Low-dM GNN SR/CR plots as a static page."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any


REGIONS = ("llcr", "qcdcr", "gcr", "dycr")
REGION_LABELS = {
    "llcr": "Lost-lepton control region",
    "qcdcr": "QCD control region",
    "gcr": "Photon control region · unit area",
    "dycr": "Dilepton control region · RZ applied",
}

PHYSICS_PLOTS = (
    ("lowdm_cr_llcr_met", "LLCR", "MET"),
    ("lowdm_cr_qcdcr_met", "QCDCR", "MET"),
    ("lowdm_cr_gcr_recoil_gcr", "GCR", "Recoil"),
    ("lowdm_cr_dy2e_recoil_dy2e", "DY2E", "Recoil"),
    ("lowdm_cr_dy2m_recoil_dy2m", "DY2M", "Recoil"),
    ("lowdm_cr_llcr_njet", "LLCR", "Njet"),
    ("lowdm_cr_qcdcr_njet", "QCDCR", "Njet"),
    ("lowdm_cr_gcr_njet_photon_clean", "GCR", "Njet"),
    ("lowdm_cr_dy2e_njet_lepton_clean", "DY2E", "Njet"),
    ("lowdm_cr_dy2m_njet_lepton_clean", "DY2M", "Njet"),
    ("lowdm_cr_llcr_nb_medium_lowdm", "LLCR", "Nb"),
    ("lowdm_cr_qcdcr_nb_medium_lowdm", "QCDCR", "Nb"),
    ("lowdm_cr_gcr_nb_photon_clean", "GCR", "Nb"),
    ("lowdm_cr_dy2e_nb_lepton_clean", "DY2E", "Nb"),
    ("lowdm_cr_dy2m_nb_lepton_clean", "DY2M", "Nb"),
    ("lowdm_cr_llcr_ht", "LLCR", "HT"),
    ("lowdm_cr_qcdcr_ht", "QCDCR", "HT"),
    ("lowdm_cr_gcr_ht_photon_clean", "GCR", "HT"),
    ("lowdm_cr_dy2e_ht_lepton_clean", "DY2E", "HT"),
    ("lowdm_cr_dy2m_ht_lepton_clean", "DY2M", "HT"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def copy_pair(source: Path, destination: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = {}
    for suffix in (".png", ".pdf"):
        source_file = source.with_suffix(suffix)
        destination_file = destination.with_suffix(suffix)
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        shutil.copy2(source_file, destination_file)
        output[suffix.removeprefix(".")] = sha256(destination_file)
    return output


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def year_cards(year: int) -> str:
    rows = [
        (
            f"plots/{year}/lowdm_sr_gnn_out_30bin",
            "Signal region",
            "30 bins · blinded Asimov background",
            "wide",
        )
    ]
    rows.extend(
        (
            f"plots/{year}/lowdm_{region}_gnn_out_inclusive",
            REGION_LABELS[region],
            "Inclusive · 5 GNN bins",
            "",
        )
        for region in REGIONS
    )
    return "\n".join(
        f"""<a class="card {kind}" href="{path}.pdf">
          <img src="{path}.png" loading="lazy" alt="{html.escape(title)}">
          <div class="caption"><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span></div>
        </a>"""
        for path, title, note, kind in rows
    )


def physics_cards(year: int) -> str:
    return "\n".join(
        f"""<a class="card" href="plots/{year}/physics/{stem}.pdf">
          <img src="plots/{year}/physics/{stem}.png" loading="lazy" alt="{html.escape(region + ' ' + variable)}">
          <div class="caption"><strong>{html.escape(region + ' · ' + variable)}</strong><span>New Low-Δm selection</span></div>
        </a>"""
        for stem, region, variable in PHYSICS_PLOTS
    )


def page_html(summary: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Low-dM GNN SR and control regions</title>
<style>
:root{{--ink:#14212b;--muted:#60717d;--line:#dce5e9;--wash:#f4f8f9;--blue:#176b87}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--wash);font:15px/1.5 Inter,system-ui,sans-serif}}
header{{color:#fff;background:linear-gradient(120deg,#102c3a,#176b87 70%,#17999c);padding:38px 24px}}
.wrap{{width:min(1500px,calc(100% - 36px));margin:auto}}h1{{margin:0;font-size:clamp(30px,4vw,48px)}}header p{{color:#d8edf1;max-width:920px}}
nav{{position:sticky;top:0;z-index:5;background:#fffffff0;border-bottom:1px solid var(--line)}}nav .wrap{{display:flex;gap:8px;padding:10px 0}}nav a{{color:var(--ink);font-weight:750;text-decoration:none;padding:7px 12px;border-radius:999px}}nav a:hover{{color:#fff;background:var(--blue)}}
main{{padding:28px 0 48px}}section{{margin-bottom:42px;scroll-margin-top:70px}}h2{{font-size:27px;margin:0 0 6px}}.lead{{color:var(--muted);margin:0 0 16px}}
.notice{{background:#fff;border:1px solid var(--line);border-left:5px solid #d9992b;border-radius:10px;padding:13px 16px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.card{{display:block;color:inherit;text-decoration:none;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 24px #1c37430e}}
.card:hover{{transform:translateY(-2px)}}.card img{{display:block;width:100%;aspect-ratio:1.71/1;object-fit:contain;background:#fff}}.card.wide{{grid-column:1/-1}}.card.wide img{{aspect-ratio:1.95/1}}
.caption{{display:flex;justify-content:space-between;gap:14px;padding:13px 15px;border-top:1px solid var(--line)}}.caption strong{{font-size:16px}}.caption span{{color:var(--muted)}}
.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}}.fact{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px}}.fact b{{display:block;font-size:21px;color:var(--blue)}}
footer{{text-align:center;color:var(--muted);padding:0 0 30px}}@media(max-width:850px){{.grid,.facts{{grid-template-columns:1fr}}.card.wide{{grid-column:auto}}.caption{{align-items:start;flex-direction:column}}}}
</style></head><body>
<header><div class="wrap"><h1>Low-Δm GNN signal and control regions</h1><p>Frozen diagonal-v3 GNN evaluated independently on the full 2024 and 2025 intermediate ROOT campaigns. Both years use the adopted 2024 score boundaries.</p></div></header>
<nav><div class="wrap"><a href="../">Overview</a><a href="#y2024">2024</a><a href="#y2025">2025</a><a href="page_summary.json">Machine summary</a></div></nav>
<main class="wrap">
<div class="facts"><div class="fact"><b>30</b>SR bins (6×5)</div><div class="fact"><b>5</b>inclusive GNN bins per CR</div><div class="fact"><b>8,390</b>2025 intermediate ROOTs</div><div class="fact"><b>1,347</b>2025 signal mass points</div></div>
<section id="y2024"><h2>2024 · 109.82 fb<sup>−1</sup></h2><p class="lead">Reference evaluation used to freeze the SR and CR score boundaries.</p><h3>GNN templates</h3><div class="grid">{year_cards(2024)}</div><h3>Low-Δm control-region physics distributions</h3><p class="lead">MET/recoil, Njet, medium-WP Nb, and HT rebuilt from the full intermediate ROOT campaign with the same new Low-Δm selection used by the GNN templates.</p><div class="grid">{physics_cards(2024)}</div></section>
<section id="y2025"><h2>2025 · 110.84 fb<sup>−1</sup></h2><p class="lead">The same model, selection contract, categories, and bin edges are applied without retraining.</p>
<div class="notice"><strong>Preliminary calibration status.</strong> The published Prompt-2025 EGM payload does not yet provide the electron-HLT and photon-CSEV working-point scale factors, so their central factors are explicitly recorded as unity. The 2025 b-tag, trigger, lepton/photon ID, pileup, and low-pT analysis scale factors are applied. One upstream JetMET source file remains permanently skipped, so complete 2025 luminosity coverage is not claimed.</div>
<h3>GNN templates</h3><div class="grid">{year_cards(2025)}</div><h3>Low-Δm control-region physics distributions</h3><p class="lead">The same observables, new Low-Δm selection, region-specific data streams, and 2024 bin edges are applied to 2025.</p><div class="grid">{physics_cards(2025)}</div></section>
</main><footer>CMS Work in progress · generated from machine-readable outputs</footer></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, type=Path)
    parser.add_argument("--sr-2024", required=True, type=Path)
    parser.add_argument("--cr-2024", required=True, type=Path)
    parser.add_argument("--result-2025", required=True, type=Path)
    parser.add_argument("--highdm73", required=True, type=Path)
    parser.add_argument(
        "--lowdm-cr-physics-plots",
        "--an-lowdm-plots",
        dest="lowdm_cr_physics_plots",
        required=True,
        type=Path,
    )
    args = parser.parse_args()
    page = args.docs / "lowdm_gnn_20260901"
    artifacts: dict[str, Any] = {}

    year_sources = {
        2024: {"sr": args.sr_2024, "cr": args.cr_2024},
        2025: {"sr": args.result_2025, "cr": args.result_2025},
    }
    for year, sources in year_sources.items():
        artifacts[str(year)] = {}
        sr = sources["sr"] / "plots/sr/lowdm_sr_gnn_out_30bin"
        artifacts[str(year)]["SR"] = copy_pair(
            sr, page / f"plots/{year}/lowdm_sr_gnn_out_30bin"
        )
        for region in REGIONS:
            stem = f"lowdm_{region}_gnn_out_inclusive"
            artifacts[str(year)][region.upper()] = copy_pair(
                sources["cr"] / f"plots/cr/{stem}", page / f"plots/{year}/{stem}"
            )
        artifacts[str(year)]["physics_distributions"] = {}
        for stem, region, variable in PHYSICS_PLOTS:
            artifacts[str(year)]["physics_distributions"][stem] = {
                "region": region,
                "variable": variable,
                "hashes": copy_pair(
                    args.lowdm_cr_physics_plots / str(year) / stem,
                    page / f"plots/{year}/physics/{stem}",
                ),
            }

    root_assets = args.docs / "plots/search_bins"
    artifacts["root_cards"] = {
        "2024_highdm73": copy_pair(
            args.highdm73 / "highdm73_search_bins",
            root_assets / "2024/highdm73_search_bins",
        ),
        "2024_lowdm30": copy_pair(
            args.sr_2024 / "plots/sr/lowdm_sr_gnn_out_30bin",
            root_assets / "2024/lowdm30_gnn_search_bins",
        ),
        "2025_lowdm30": copy_pair(
            args.result_2025 / "plots/sr/lowdm_sr_gnn_out_30bin",
            root_assets / "2025/lowdm30_gnn_search_bins",
        ),
    }

    validation_2025 = load(args.result_2025 / "validation_summary.json")
    cr_2024 = load(args.cr_2024 / "plots/cr/lowdm_cr_nnout_inclusive_plot_summary.json")
    cr_2025 = load(args.result_2025 / "plots/cr/lowdm_cr_nnout_inclusive_plot_summary.json")
    summary = {
        "schema_version": "lowdm_gnn_web_page_v1",
        "status": "complete_with_known_2025_calibration_gaps",
        "years": {
            "2024": {
                "luminosity_fb": cr_2024.get("luminosity_fb", 109.82),
                "input_files": cr_2024["input_files"],
                "regions": cr_2024["regions"],
            },
            "2025": {
                "luminosity_fb": cr_2025["luminosity_fb"],
                "input_files": cr_2025["input_files"],
                "regions": cr_2025["regions"],
                "validation_status": validation_2025["status"],
                "input_audit": validation_2025["input_audit"],
                "correction_audit": validation_2025["correction_audit"],
            },
        },
        "binning": validation_2025["binning_audit"],
        "physics_distribution_source": {
            "reference": "Run-3 AN observables and the adopted analysis plotting style",
            "policy": "Rebuilt from the full 2024/2025 intermediate ROOT campaigns with the frozen new Low-dM CR selection; AN outputs are not reused.",
            "variables": ["MET/recoil", "Njet", "Nb (medium WP)", "HT"],
            "regions": ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"],
        },
        "artifacts": artifacts,
    }
    write_text(page / "page_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_text(page / "index.html", page_html(summary))
    print(json.dumps({"status": summary["status"], "page": str(page.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
