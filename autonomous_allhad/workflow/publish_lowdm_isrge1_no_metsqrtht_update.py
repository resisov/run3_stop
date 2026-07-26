#!/usr/bin/env python3
"""Publish the Low-dM ISR>=1, no-MET/sqrt(HT) alternative-selection study."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


MARKER_START = "<!-- lowdm-isrge1-no-metsqrtht-update:start -->"
MARKER_END = "<!-- lowdm-isrge1-no-metsqrtht-update:end -->"


def copy_pair(source_png: Path, source_pdf: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_png, destination.with_suffix(".png"))
    shutil.copy2(source_pdf, destination.with_suffix(".pdf"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", required=True, type=Path)
    parser.add_argument("--category-png", required=True, type=Path)
    parser.add_argument("--category-pdf", required=True, type=Path)
    parser.add_argument("--limit-png", required=True, type=Path)
    parser.add_argument("--limit-pdf", required=True, type=Path)
    parser.add_argument("--analysis-summary", required=True, type=Path)
    args = parser.parse_args()

    page_dir = args.page_dir
    category = page_dir / "plots/categories/lowdm_sr_42bin_isrge1_no_metsqrtht"
    limit = page_dir / "plots/limits/expected_limit_lowdm42_isrge1_no_metsqrtht"
    summary_name = "lowdm_isrge1_no_metsqrtht_summary.json"

    copy_pair(args.category_png, args.category_pdf, category)
    copy_pair(args.limit_png, args.limit_pdf, limit)
    shutil.copy2(args.analysis_summary, page_dir / summary_name)

    index = page_dir / "index.html"
    html = index.read_text()
    html = re.sub(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        "",
        html,
        flags=re.DOTALL,
    )
    notice = (
        f"{MARKER_START}"
        "<section class='update' style='max-width:1500px;margin:14px auto 0;padding:12px 18px;"
        "background:#fff;border:1px solid #d6dcdf;border-radius:6px'>"
        "<strong>Low-dM SR alternative-selection study (2026-07-26)</strong>"
        "<p>The SR requires at least one ISR AK8 jet instead of exactly one and removes "
        "MET/&radic;HT&ge;10. The five Low-dM CR templates remain frozen at their current "
        "definitions. The Low-dM-only expected model has six channels and 42 bins per channel. "
        "At (mStop, mLSP)=(900,700) GeV, the median expected r changes from 3.3281 to 3.2656 "
        "(1.88% improvement). This is a proposed alternative, not an adopted selection. "
        f"<a href='{summary_name}'>Machine-readable summary</a>.</p>"
        "</section>"
        f"{MARKER_END}"
    )
    html = html.replace("</header>", "</header>" + notice, 1)
    cards = (
        f"{MARKER_START}"
        "<a class='plot' data-family='categories' data-kind='SR' "
        "data-search='low-dm sr 42-bin alternative isr ak8 ge1 no met sqrt ht' "
        "href='plots/categories/lowdm_sr_42bin_isrge1_no_metsqrtht.pdf'>"
        "<img src='plots/categories/lowdm_sr_42bin_isrge1_no_metsqrtht.png' loading='lazy' "
        "alt='Low-dM SR 42-bin categories with at least one ISR AK8 jet and no MET over square-root HT cut'>"
        "<span>Low-dM SR 42 bins · ISR AK8 ≥1 · no MET/√HT cut</span></a>"
        "<a class='plot' data-family='limits' data-kind='SR' "
        "data-search='limit low-dm only 42-bin alternative isr ak8 ge1 no met sqrt ht' "
        "href='plots/limits/expected_limit_lowdm42_isrge1_no_metsqrtht.pdf'>"
        "<img src='plots/limits/expected_limit_lowdm42_isrge1_no_metsqrtht.png' loading='lazy' "
        "alt='Low-dM-only expected limit with at least one ISR AK8 jet and no MET over square-root HT cut'>"
        "<span>2024 expected limit · Low-dM-only alternative selection</span></a>"
        f"{MARKER_END}"
    )
    html = html.replace("</div></main>", cards + "</div></main>", 1)
    index.write_text(html)

    summary_path = page_dir / "page_summary.json"
    page_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    page_summary["lowdm_isrge1_no_metsqrtht_update"] = {
        "status": "complete",
        "adoption_status": "proposed_alternative",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selection_change": [
            "ISR AK8 multiplicity changed from exactly 1 to at least 1",
            "MET/sqrt(HT) >= 10 removed",
        ],
        "category_plot": str(category.with_suffix(".png").relative_to(page_dir)),
        "lowdm_only_limit": str(limit.with_suffix(".png").relative_to(page_dir)),
        "analysis_summary": summary_name,
    }
    summary_path.write_text(json.dumps(page_summary, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "complete",
                "page": str(index),
                "assets": [
                    str(category.with_suffix(".png")),
                    str(category.with_suffix(".pdf")),
                    str(limit.with_suffix(".png")),
                    str(limit.with_suffix(".pdf")),
                    str(page_dir / summary_name),
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
