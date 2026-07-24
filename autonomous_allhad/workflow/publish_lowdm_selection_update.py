#!/usr/bin/env python3
"""Publish the adopted Low-dM SR selection study into the current 2024 page."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


MARKER_START = "<!-- lowdm-selection-update:start -->"
MARKER_END = "<!-- lowdm-selection-update:end -->"


def copy_pair(source_png: Path, source_pdf: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_png, destination.with_suffix(".png"))
    shutil.copy2(source_pdf, destination.with_suffix(".pdf"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", required=True, type=Path)
    parser.add_argument("--category-png", required=True, type=Path)
    parser.add_argument("--category-pdf", required=True, type=Path)
    parser.add_argument("--lowdm-limit-png", required=True, type=Path)
    parser.add_argument("--lowdm-limit-pdf", required=True, type=Path)
    parser.add_argument("--combined-limit-png", required=True, type=Path)
    parser.add_argument("--combined-limit-pdf", required=True, type=Path)
    parser.add_argument("--analysis-summary", required=True, type=Path)
    args = parser.parse_args()

    page_dir = args.page_dir
    category = page_dir / "plots/categories/lowdm_sr_42bin_no_bveto_mtb"
    lowdm_limit = page_dir / "plots/limits/expected_limit_lowdm42_no_bveto_mtb_run2_x1600"
    combined_limit = page_dir / "plots/limits/expected_limit_highdm54_lowdm42_no_bveto_mtb_run2_x1600"
    copy_pair(args.category_png, args.category_pdf, category)
    copy_pair(args.lowdm_limit_png, args.lowdm_limit_pdf, lowdm_limit)
    copy_pair(args.combined_limit_png, args.combined_limit_pdf, combined_limit)
    shutil.copy2(args.analysis_summary, page_dir / "lowdm_selection_update_summary.json")

    index = page_dir / "index.html"
    html = index.read_text()
    html = re.sub(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"<a class='plot'[^>]*href='plots/categories/lowdm_[^']*onebin\.pdf'.*?</a>",
        "",
        html,
        flags=re.DOTALL,
    )
    if "data-value='limits'" not in html:
        html = html.replace(
            "<button data-value='categories'>Categories</button>",
            "<button data-value='categories'>Categories</button><button data-value='limits'>Limits</button>",
        )
    notice = (
        f"{MARKER_START}"
        "<section class='update' style='max-width:1500px;margin:14px auto 0;padding:12px 18px;"
        "background:#fff;border:1px solid #d6dcdf;border-radius:6px'>"
        "<strong>Low-dM SR selection update (2026-07-24)</strong>"
        "<p>ISR-subjet b veto and mTb&lt;175 GeV are removed from the rebuilt 42-bin SR. "
        "The five Low-dM CR templates remain at their current nominal-intermediate definitions. "
        "Expected contours use 2024 Asimov data, mStop≤1600 GeV, and show the Run-2 SUS-19-010 overlay. "
        "They include nominal-payload weight shapes, luminosity, and autoMCStats; object-shape "
        "variations are not included.</p>"
        "</section>"
        f"{MARKER_END}"
    )
    html = html.replace("</header>", "</header>" + notice, 1)
    cards = (
        f"{MARKER_START}"
        "<a class='plot' data-family='categories' data-kind='SR' data-search='low-dm sr 42-bin categories no isr subjet b veto no mtb' "
        "href='plots/categories/lowdm_sr_42bin_no_bveto_mtb.pdf'>"
        "<img src='plots/categories/lowdm_sr_42bin_no_bveto_mtb.png' loading='lazy' "
        "alt='Low-dM SR 42-bin categories after removing ISR-subjet b veto and mTb requirement'>"
        "<span>Low-dM SR 42-bin categories · updated selection</span></a>"
        "<a class='plot' data-family='limits' data-kind='SR' data-search='limit low-dm only 42-bin no isr subjet b veto no mtb run2' "
        "href='plots/limits/expected_limit_lowdm42_no_bveto_mtb_run2_x1600.pdf'>"
        "<img src='plots/limits/expected_limit_lowdm42_no_bveto_mtb_run2_x1600.png' loading='lazy' "
        "alt='Low-dM-only expected limit after removing ISR-subjet b veto and mTb requirement'>"
        "<span>2024 expected limit · Low-dM 42-bin CR+SR only · updated SR selection</span></a>"
        "<a class='plot' data-family='limits' data-kind='SR' data-search='limit high-dm 54-bin low-dm 42-bin combined no isr subjet b veto no mtb run2' "
        "href='plots/limits/expected_limit_highdm54_lowdm42_no_bveto_mtb_run2_x1600.pdf'>"
        "<img src='plots/limits/expected_limit_highdm54_lowdm42_no_bveto_mtb_run2_x1600.png' loading='lazy' "
        "alt='Combined High-dM and Low-dM expected limit after updating the Low-dM SR selection'>"
        "<span>2024 expected limit · High-dM 54-bin + Low-dM 42-bin · updated SR selection</span></a>"
        f"{MARKER_END}"
    )
    html = html.replace("</div></main>", cards + "</div></main>", 1)
    index.write_text(html)

    summary_path = page_dir / "page_summary.json"
    page_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    page_summary["lowdm_selection_update"] = {
        "status": "complete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selection_change": ["ISR-subjet b veto removed", "mTb < 175 GeV requirement removed"],
        "category_plot": str(category.with_suffix(".png").relative_to(page_dir)),
        "lowdm_only_limit": str(lowdm_limit.with_suffix(".png").relative_to(page_dir)),
        "combined_limit": str(combined_limit.with_suffix(".png").relative_to(page_dir)),
        "analysis_summary": "lowdm_selection_update_summary.json",
    }
    summary_path.write_text(json.dumps(page_summary, indent=2, sort_keys=True) + "\n")
    result = {
        "status": "complete",
        "page": str(index),
        "assets": [
            str(category.with_suffix(".png")),
            str(lowdm_limit.with_suffix(".png")),
            str(combined_limit.with_suffix(".png")),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
