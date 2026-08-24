#!/usr/bin/env python3
"""Publish validated expected limits and optional impacts to a plot page."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


NOTICE_START = "<!-- free-background-limits:start -->"
NOTICE_END = "<!-- free-background-limits:end -->"
CARDS_START = "<!-- free-background-limit-cards:start -->"
CARDS_END = "<!-- free-background-limit-cards:end -->"
TOPOLOGIES = ("T2tt", "T2tb", "T2bW")


def copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def public_value(value):
    """Remove private filesystem locations from published provenance."""
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str) and (
        "/eos/" in value
        or "/afs/" in value
        or value.startswith("workflow/")
    ):
        return Path(value).name
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("free_background", "tailmerged", "canonical_combined"),
        default="free_background",
    )
    parser.add_argument("--highdm-bins", type=int, default=60)
    parser.add_argument("--lowdm-bins", type=int, default=34)
    parser.add_argument("--campaign-label", default="2024")
    parser.add_argument("--impact-r1-dir", type=Path)
    parser.add_argument("--impact-r0-dir", type=Path)
    parser.add_argument("--an-category-plot", type=Path)
    parser.add_argument("--an-category-summary", type=Path)
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=TOPOLOGIES,
        default=list(TOPOLOGIES),
    )
    args = parser.parse_args()
    topologies = tuple(args.topologies)

    page_dir = args.page_dir
    plot_dir = page_dir / "plots/limits"
    data_dir = page_dir / "data"
    records = []
    manifests = {}
    for topology in topologies:
        source_dir = (
            args.results_dir / f"free_background_{topology.lower()}"
            if args.layout == "free_background"
            else args.results_dir / topology
        )
        manifest_path = source_dir / (
            "combine_input_manifest.json"
            if args.layout == "free_background"
            else "limit_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        valid_statuses = (
            {"combine_outputs_complete", "combine_outputs_partial"}
            if args.layout == "free_background"
            else {"complete", "partial"}
        )
        if manifest.get("status") not in valid_statuses:
            raise RuntimeError(
                f"{topology} result is incomplete: {manifest.get('status')}"
            )
        limits = manifest.get("limits") or {}
        if limits.get("status") not in {"complete", "partial"}:
            raise RuntimeError(
                f"{topology} limit collection is invalid: "
                f"{limits.get('status')}"
            )
        if (
            int(limits.get("collected_point_count", 0))
            + len(limits.get("missing_points") or [])
            != int(limits.get("requested_point_count", 0))
        ):
            raise RuntimeError(
                f"{topology} limit coverage does not reconcile"
            )
        if not bool(manifest.get("run2_overlay")):
            raise RuntimeError(f"wrong Run-2 overlay policy for {topology}")
        png = source_dir / Path(manifest["contour_png"]).name
        pdf = source_dir / Path(manifest["contour_pdf"]).name
        stem = png.stem
        copy(png, plot_dir / png.name)
        copy(pdf, plot_dir / pdf.name)
        copy(source_dir / "expected_limits.json", data_dir / f"{stem}.json")
        public_manifest_path = data_dir / f"{stem}_manifest.json"
        public_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        public_manifest_path.write_text(
            json.dumps(public_value(manifest), indent=2, sort_keys=True) + "\n"
        )
        records.append(
            {
                "family": "limits",
                "family_label": "Expected limits",
                "kind": "Overview",
                "name": stem,
                "pdf": f"plots/limits/{pdf.name}",
                "png": f"plots/limits/{png.name}",
                "region": (
                    f"{args.campaign_label} High-dM {args.highdm_bins} + "
                    f"Low-dM {args.lowdm_bins}"
                ),
                "variable": (
                    f"{topology} "
                    + (
                        "combined expected limit"
                        if args.layout == "canonical_combined"
                        else "free-background expected limit"
                    )
                ),
            }
        )
        manifests[topology] = manifest

    links = []
    cards = []
    coverage_notes = []
    for record, topology in zip(records, topologies):
        limits = manifests[topology]["limits"]
        missing = limits.get("missing_points") or []
        coverage_notes.append(
            f"{topology}: {limits['collected_point_count']}/"
            f"{limits['requested_point_count']} points"
            + (
                " (missing " + ", ".join(missing) + ")"
                if missing
                else ""
            )
        )
        links.append(
            f"<a href='{html.escape(record['pdf'])}'>{topology} PDF</a>"
        )
        cards.append(
            "<a class='plot' data-family='limits' data-kind='Overview' "
            f"data-search='{args.campaign_label} expected limit "
            f"{topology.lower()} high-dm {args.highdm_bins} "
            f"low-dm {args.lowdm_bins}' "
            f"href='{html.escape(record['pdf'])}'>"
            f"<img src='{html.escape(record['png'])}' loading='lazy' "
            f"alt='{args.campaign_label} {topology} expected limit'>"
            f"<span>{args.campaign_label} expected limit · {topology} · "
            f"High-dM {args.highdm_bins} + Low-dM {args.lowdm_bins}"
            + (
                "</span></a>"
                if args.layout == "canonical_combined"
                else " · free background normalizations</span></a>"
            )
        )
    category_record = None
    if args.an_category_plot:
        if not args.an_category_summary:
            raise RuntimeError(
                "--an-category-summary is required with --an-category-plot"
            )
        category_summary = json.loads(args.an_category_summary.read_text())
        if category_summary.get("status") != "complete":
            raise RuntimeError("AN category plot summary is incomplete")
        if int(category_summary.get("highdm_search_bins", 0)) != args.highdm_bins:
            raise RuntimeError("AN category plot High-dM bin count mismatch")
        if "background stack" not in str(category_summary.get("signal_policy", "")):
            raise RuntimeError(
                "AN category plot does not document the background stack"
            )
        source_png = args.an_category_plot
        source_pdf = source_png.with_suffix(".pdf")
        copy(source_png, page_dir / "plots/categories" / source_png.name)
        copy(source_pdf, page_dir / "plots/categories" / source_pdf.name)
        category_record = {
            "family": "categories",
            "family_label": "Category/search-bin plots",
            "kind": "SR",
            "name": source_png.stem,
            "pdf": f"plots/categories/{source_pdf.name}",
            "png": f"plots/categories/{source_png.name}",
            "region": f"High-dM {args.highdm_bins}-bin SR categories",
            "variable": "background stack and signal overlays",
        }
        cards.insert(
            0,
            "<a class='plot' data-family='categories' data-kind='SR' "
            f"data-search='high-dm {args.highdm_bins} an signal categories "
            f"background stack' href='{category_record['pdf']}'>"
            f"<img src='{category_record['png']}' loading='lazy' "
            f"alt='High-dM {args.highdm_bins}-bin SR categories with "
            "background stack and signal overlays'><span>AN signal "
            f"categories · High-dM {args.highdm_bins} bins · background "
            "stack + signal overlays</span></a>"
        )
    impact_records = []
    impact_links = []
    for fit_label, impact_dir, expected_signal in (
        ("r1", args.impact_r1_dir, 1),
        ("r0", args.impact_r0_dir, 0),
    ):
        if impact_dir is None:
            continue
        source_dir = impact_dir / "work"
        status = json.loads((source_dir / "impact_status.json").read_text())
        if status.get("status") != "complete":
            raise RuntimeError(f"{fit_label} impact is incomplete")
        if int(status.get("asimov_expect_signal", -1)) != expected_signal:
            raise RuntimeError(f"{fit_label} impact expectation mismatch")
        source_stem = "impacts_mStop1200_mLSP500"
        stem = (
            f"impacts_t2tt_2024_2025_highdm{args.highdm_bins}_"
            f"lowdm{args.lowdm_bins}_mStop1200_mLSP500_{fit_label}"
        )
        copy(source_dir / f"{source_stem}.png", page_dir / "plots/impacts" / f"{stem}.png")
        copy(source_dir / f"{source_stem}.pdf", page_dir / "plots/impacts" / f"{stem}.pdf")
        copy(source_dir / f"{source_stem}.json", data_dir / f"{stem}.json")
        public_status = public_value(status)
        (data_dir / f"{stem}_status.json").write_text(
            json.dumps(public_status, indent=2, sort_keys=True) + "\n"
        )
        record = {
            "family": "impacts",
            "family_label": "Nuisance impacts",
            "kind": "Overview",
            "name": stem,
            "pdf": f"plots/impacts/{stem}.pdf",
            "png": f"plots/impacts/{stem}.png",
            "region": (
                f"2024+2025 High-dM {args.highdm_bins} + "
                f"Low-dM {args.lowdm_bins}"
            ),
            "variable": f"T2tt mStop1200 mLSP500 Asimov {fit_label}",
        }
        impact_records.append(record)
        impact_links.append(
            f"<a href='{html.escape(record['pdf'])}'>Impact {fit_label} PDF</a>"
        )
        cards.append(
            "<a class='plot' data-family='impacts' data-kind='Overview' "
            f"data-search='2024 2025 combined t2tt impact {fit_label} "
            f"high-dm {args.highdm_bins} low-dm {args.lowdm_bins} "
            "mstop1200 mlsp500' "
            f"href='{html.escape(record['pdf'])}'>"
            f"<img src='{html.escape(record['png'])}' loading='lazy' "
            f"alt='Combined 2024+2025 T2tt nuisance impacts ({fit_label})'>"
            f"<span>2024+2025 impact · T2tt (1200, 500) GeV · "
            f"Asimov {fit_label}</span></a>"
        )

    if args.layout == "canonical_combined":
        notice_body = (
            "<strong>2024+2025 combined statistical results</strong>"
            f"<p>High-dM {args.highdm_bins} + Low-dM {args.lowdm_bins}. "
            "Lost-lepton and QCD are constrained by shared control-region "
            "transfer factors. Z→νν uses the measured RZ normalization and "
            "the Qγ-normalized photon control sample for Sγ shape only. "
            "Signal-region observations remain blinded. Coverage: "
            + "; ".join(coverage_notes)
            + ". "
            + " · ".join(links + impact_links)
            + ".</p>"
        )
    else:
        notice_body = (
            "<strong>2024 expected limits with provisional free-background "
            "model</strong><p>The corrected exclusive Drell–Yan stitching is used. "
            + ", ".join(topologies)
            + " "
            + ("are independent signal hypotheses. " if len(topologies) > 1 else "is shown. ")
            + "The previous transfer-factor and RZ/Sγ background-estimation "
            "constraints are absent. Seven canonical background normalizations "
            "are unconstrained global rate parameters, while shape/weight "
            "nuisances and autoMCStats remain. The evaluated grid has "
            "mStop≤1800 GeV. "
            + (
                "The High-dM signal model has 55 bins: categories 1, 2, 3, "
                "5, and 8 retain six bins, while the final two bins are merged "
                "in the other five categories. "
                if args.layout == "tailmerged" and args.highdm_bins == 55
                else ""
            )
            + "Official topology-matched CMS-SUS-19-010 observed and expected "
            "Run-2 contours are overlaid for every displayed signal model. "
            "Coverage: "
            + "; ".join(coverage_notes)
            + ". "
            + " · ".join(links)
            + ".</p>"
        )
    notice = (
        NOTICE_START
        + "<section class='update' style='max-width:1500px;margin:14px auto 0;"
        "padding:12px 18px;background:#fff;border:1px solid #d6dcdf;"
        "border-radius:6px'>"
        + notice_body
        + "</section>"
        + NOTICE_END
    )
    card_html = CARDS_START + "".join(cards) + CARDS_END

    index_path = page_dir / "index.html"
    page = index_path.read_text()
    if not args.an_category_plot:
        preserved_search_cards = re.findall(
            r"<a class='plot'[^>]*href='plots/categories/"
            r"highdm_sr_selected_recoil[^']*'[^>]*>.*?</a>",
            page,
            flags=re.DOTALL,
        )
        if preserved_search_cards:
            card_html = (
                CARDS_START
                + "".join(preserved_search_cards)
                + "".join(cards)
                + CARDS_END
            )
    if args.an_category_plot:
        page = re.sub(
            r"<a class='plot'[^>]*href='plots/categories/"
            r"highdm_sr_selected_recoil[^']*'[^>]*>.*?</a>",
            "",
            page,
            flags=re.DOTALL,
        )
    page = re.sub(
        re.escape(NOTICE_START) + r".*?" + re.escape(NOTICE_END),
        "",
        page,
        flags=re.DOTALL,
    )
    if impact_records:
        page = re.sub(
            r"<!-- full-stat-impact-update:start -->.*?"
            r"<!-- full-stat-impact-update:end -->",
            "",
            page,
            flags=re.DOTALL,
        )
        page = re.sub(
            r"<!-- nominal-stat-addendum:start -->.*?"
            r"<!-- nominal-stat-addendum:end -->",
            "",
            page,
            flags=re.DOTALL,
        )
    page = page.replace("</header>", "</header>" + notice, 1)
    pattern = re.escape(CARDS_START) + r".*?" + re.escape(CARDS_END)
    if re.search(pattern, page, flags=re.DOTALL):
        page = re.sub(pattern, card_html, page, flags=re.DOTALL)
    elif "</div></main>" in page:
        page = page.replace("</div></main>", card_html + "</div></main>", 1)
    else:
        raise RuntimeError("plot-card insertion point is missing")
    index_path.write_text(page)

    summary_path = page_dir / "page_summary.json"
    summary = json.loads(summary_path.read_text())
    replaced_names = {
        record["name"] for record in records + impact_records
    }
    if category_record:
        replaced_names.add(category_record["name"])
    current = [
        record
        for record in summary.get("records", [])
        if not (
            record.get("name") in replaced_names
            or (
                record.get("family") == "limits"
                and str(record.get("name", "")).startswith("expected_limit_t2")
            )
            or (impact_records and record.get("family") == "impacts")
            or (
                category_record
                and record.get("family") == "categories"
                and str(record.get("name", "")).startswith(
                    "highdm_sr_selected_recoil"
                )
            )
        )
    ]
    summary["records"] = (
        current
        + ([category_record] if category_record else [])
        + records
        + impact_records
    )
    summary["generated_at"] = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    prior_signals = (
        (summary.get("statistical_results") or {}).get("signals") or {}
    )
    prior_signals.update(
        {
            topology: {
                "plot": record["png"],
                "limits": f"data/{record['name']}.json",
                "manifest": f"data/{record['name']}_manifest.json",
                "mass_point_count": manifests[topology]["mass_point_count"],
                "collected_mass_point_count": manifests[topology][
                    "limits"
                ]["collected_point_count"],
                "missing_mass_points": manifests[topology]["limits"][
                    "missing_points"
                ],
                "run2_overlay": manifests[topology]["run2_overlay"],
            }
            for topology, record in zip(topologies, records)
        }
    )
    if args.layout == "canonical_combined":
        summary["statistical_results"] = {
            "status": "complete_2024_2025_combined",
            "model": "shared_control_region_transfer_factors_and_rz_sgamma",
            "campaign": "2024+2025",
            "highdm_signal_bins": args.highdm_bins,
            "lowdm_signal_bins": args.lowdm_bins,
            "signal_regions_blinded": True,
            "max_mstop_GeV": 1800,
            "signals": prior_signals,
            "impacts": {
                record["variable"].rsplit(" ", 1)[-1]: {
                    "plot": record["png"],
                    "json": f"data/{record['name']}.json",
                    "status": f"data/{record['name']}_status.json",
                }
                for record in impact_records
            },
        }
        if impact_records:
            summary["impact_update"] = {
                "status": "complete_2024_2025_combined_r1_r0",
                "benchmark": {"mStop_GeV": 1200, "mLSP_GeV": 500},
                "analysis": "NPS26012",
                "fits": [record["variable"] for record in impact_records],
            }
    else:
        summary["statistical_results"] = {
            "status": "complete_free_background_2024",
            "model": "free_background_global_process_normalizations",
            "background_rate_parameters": 7,
            "external_background_constraints": [],
            "autoMCStats": 10,
            "max_mstop_GeV": 1800,
            "signals": prior_signals,
        }
    summary["plot_counts"] = {
        **(summary.get("plot_counts") or {}),
        "limits": len(records),
        "impacts": len(impact_records),
        "total": len(summary["records"]),
    }
    if category_record:
        summary["an_signal_category_update"] = {
            "status": "complete",
            "highdm_bins": args.highdm_bins,
            "background_stack": True,
            "signal_overlays": True,
            "plot": category_record["png"],
        }
    summary = public_value(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "page": str(index_path),
                "limits": {
                    topology: manifests[topology]["mass_point_count"]
                    for topology in topologies
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
