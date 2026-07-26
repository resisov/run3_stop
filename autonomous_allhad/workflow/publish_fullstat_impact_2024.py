#!/usr/bin/env python3
"""Publish the validated 2024 High-dM + Low-dM full-stat impact result."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


NOTICE_START = "<!-- full-stat-impact-update:start -->"
NOTICE_END = "<!-- full-stat-impact-update:end -->"
CARD_START = "<!-- nominal-stat-addendum:start -->"
CARD_END = "<!-- nominal-stat-addendum:end -->"
STEM = "impacts_2024_highdm60_lowdm42_mStop1200_mLSP500_full_mcstat"


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    args = parser.parse_args()

    page_dir = args.page_dir
    bundle = args.bundle_dir
    validation = json.loads((bundle / "impact_validation.json").read_text())
    scan = validation["impact_scan"]
    analysis = validation["analysis"]

    required = {
        "status": validation.get("status") == "complete",
        "year": analysis.get("year") == 2024,
        "highdm_bins": analysis.get("highdm_signal_bins") == 60,
        "lowdm_bins": analysis.get("lowdm_signal_bins") == 42,
        "benchmark": analysis.get("benchmark")
        == {"mStop_GeV": 1200, "mLSP_GeV": 500},
        "all_fits": scan.get("parameter_count") == 369
        and scan.get("missing_fit_count") == 0
        and scan.get("invalid_fit_count") == 0,
        "statistical_nuisances": scan.get("statistical_parameter_count")
        == 358,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError("publication validation failed: " + ", ".join(failed))

    plot_dir = page_dir / "plots/impacts"
    data_dir = page_dir / "data"
    sources = {
        f"{STEM}.png": bundle / f"{STEM}.png",
        f"{STEM}.pdf": bundle / f"{STEM}.pdf",
        f"{STEM}_summary.png": bundle / f"{STEM}_summary.png",
        f"{STEM}_summary.pdf": bundle / f"{STEM}_summary.pdf",
    }
    for name, source in sources.items():
        copy(source, plot_dir / name)

    data_sources = {
        f"{STEM}.json": bundle / "impacts_mStop1200_mLSP500.json",
        f"{STEM}_validation.json": bundle / "impact_validation.json",
        f"{STEM}_labels.json": bundle / "impact_labels_2024.json",
    }
    for name, source in data_sources.items():
        copy(source, data_dir / name)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    poi_fit = scan["poi"]["fit"]
    notice = (
        f"{NOTICE_START}"
        "<section class='update' style='max-width:1500px;margin:14px auto 0;"
        "padding:12px 18px;background:#fff;border:1px solid #d6dcdf;"
        "border-radius:6px'>"
        "<strong>2024-only High-dM + Low-dM nuisance impacts "
        "for (mStop, mLSP)=(1200, 500) GeV</strong>"
        "<p>The expected Asimov fit uses the High-dM 60-bin and Low-dM "
        "42-bin signal regions with their control-region constraints "
        "(12 channels, 342 analysis bins). The Poisson likelihood includes "
        "data-counting statistics, and 358 individual autoMCStats nuisance "
        "parameters are scanned alongside 11 luminosity/rate/weight "
        "nuisances. All 369 fits are present and valid. "
        f"The fitted signal strength is r={poi_fit[1]:.2f}"
        f"<sup>+{poi_fit[2] - poi_fit[1]:.2f}</sup>"
        f"<sub>−{poi_fit[1] - poi_fit[0]:.2f}</sub>. "
        "The current card does not contain object-shape nuisance templates. "
        f"<a href='plots/impacts/{STEM}.pdf'>All 15 impact pages</a> · "
        f"<a href='data/{STEM}.json'>Combine impact JSON</a> · "
        f"<a href='data/{STEM}_validation.json'>validation report</a>.</p>"
        "</section>"
        f"{NOTICE_END}"
    )
    card = (
        f"{CARD_START}"
        "<a class='plot' data-family='impacts' data-kind='Overview' "
        "data-search='2024 only full statistical autoMCStats impact "
        "high-dm 60 low-dm 42 mstop1200 mlsp500' "
        f"href='plots/impacts/{STEM}.pdf'>"
        f"<img src='plots/impacts/{STEM}.png' loading='lazy' "
        "alt='2024-only High-dM plus Low-dM nuisance impacts including "
        "individual autoMCStats parameters'>"
        "<span>2024-only impact · High-dM 60 + Low-dM 42 · "
        "(1200, 500) GeV · 358 autoMCStats + 11 other nuisances</span></a>"
        f"{CARD_END}"
    )

    index_path = page_dir / "index.html"
    html = index_path.read_text()
    html = re.sub(
        re.escape(NOTICE_START) + r".*?" + re.escape(NOTICE_END),
        "",
        html,
        flags=re.DOTALL,
    )
    html = html.replace("</header>", "</header>" + notice, 1)
    replacement = re.escape(CARD_START) + r".*?" + re.escape(CARD_END)
    if not re.search(replacement, html, flags=re.DOTALL):
        raise RuntimeError("impact card marker is missing from index.html")
    html = re.sub(replacement, card, html, flags=re.DOTALL)
    html = re.sub(
        r"Frozen 2024 nominal snapshot · \d+ plots · generated [^<]+",
        f"Frozen 2024 nominal snapshot · 293 plots · generated {now}",
        html,
        count=1,
    )
    index_path.write_text(html)

    summary_path = page_dir / "page_summary.json"
    page_summary = json.loads(summary_path.read_text())
    records = [
        record
        for record in page_summary["records"]
        if record.get("family") != "impacts"
    ]
    records.append(
        {
            "family": "impacts",
            "family_label": "Nuisance impacts",
            "kind": "Overview",
            "name": STEM,
            "pdf": f"plots/impacts/{STEM}.pdf",
            "png": f"plots/impacts/{STEM}.png",
            "region": "2024 High-dM 60 + Low-dM 42",
            "variable": "mStop1200 mLSP500 full statistics",
        }
    )
    page_summary["records"] = records
    page_summary["generated_at"] = now
    page_summary["impact_update"] = {
        "status": "complete_current_2024_model",
        "updated_at": now,
        "year": 2024,
        "data_mode": "asimov",
        "benchmark": {"mStop_GeV": 1200, "mLSP_GeV": 500},
        "channels": 12,
        "total_analysis_bins": 342,
        "highdm_signal_bins": 60,
        "lowdm_signal_bins": 42,
        "poi": scan["poi"],
        "parameter_count": 369,
        "statistical_parameter_count": 358,
        "nonstatistical_parameter_count": 11,
        "missing_fit_count": 0,
        "invalid_fit_count": 0,
        "statistical_model": {
            "data_counting": "Poisson likelihood",
            "mc_statistics": "individual autoMCStats nuisance fits",
            "autoMCStats_threshold": analysis["autoMCStats_threshold"],
        },
        "current_model_scope": (
            "2024 luminosity plus ten rate/weight shape nuisances and "
            "individual MC-statistical nuisances; object-shape templates "
            "are absent from this card"
        ),
        "plot": f"plots/impacts/{STEM}.png",
        "multipage_pdf": f"plots/impacts/{STEM}.pdf",
        "summary_pdf": f"plots/impacts/{STEM}_summary.pdf",
        "combine_json": f"data/{STEM}.json",
        "validation": f"data/{STEM}_validation.json",
        "labels": f"data/{STEM}_labels.json",
    }
    statistical = page_summary.setdefault("statistical_results", {})
    statistical.update(
        {
            "combined": (
                "plots/limits/expected_limit_highdm60_lowdm42_x1600.png"
            ),
            "impact": f"plots/impacts/{STEM}.png",
            "impact_scope": page_summary["impact_update"][
                "current_model_scope"
            ],
            "impact_benchmark": {
                "mStop_GeV": 1200,
                "mLSP_GeV": 500,
            },
            "impact_parameters": {
                "total": 369,
                "autoMCStats": 358,
                "other": 11,
            },
            "status": "complete_current_2024_model_full_statistics",
            "updated_at": now,
        }
    )
    summary_path.write_text(
        json.dumps(page_summary, indent=2, sort_keys=True) + "\n"
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "page": str(index_path),
                "record_count": len(records),
                "impact_record_count": sum(
                    record.get("family") == "impacts"
                    for record in records
                ),
                "assets": [
                    str(path)
                    for path in (
                        list(plot_dir / name for name in sources)
                        + list(data_dir / name for name in data_sources)
                    )
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
