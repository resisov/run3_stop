#!/usr/bin/env python3
"""Add the three validated 2024 free-background limits to a plot page."""

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
        source_dir = args.results_dir / f"free_background_{topology.lower()}"
        manifest_path = source_dir / "combine_input_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {
            "combine_outputs_complete",
            "combine_outputs_partial",
        }:
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
                "region": "2024 High-dM 60 + Low-dM 34",
                "variable": f"{topology} free-background expected limit",
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
            f"data-search='2024 expected limit free background {topology.lower()} "
            "high-dm 60 low-dm 34' "
            f"href='{html.escape(record['pdf'])}'>"
            f"<img src='{html.escape(record['png'])}' loading='lazy' "
            f"alt='2024 {topology} expected limit with free background "
            "normalizations'>"
            f"<span>2024 expected limit · {topology} · High-dM 60 + "
            "Low-dM 34 · free background normalizations</span></a>"
        )
    notice = (
        NOTICE_START
        + "<section class='update' style='max-width:1500px;margin:14px auto 0;"
        "padding:12px 18px;background:#fff;border:1px solid #d6dcdf;"
        "border-radius:6px'><strong>2024 expected limits with provisional "
        "free-background model</strong><p>The corrected exclusive Drell–Yan "
        "stitching is used. "
        + ", ".join(topologies)
        + " "
        + ("are independent signal hypotheses. " if len(topologies) > 1 else "is shown. ")
        + "The previous transfer-factor and "
        "RZ/Sγ background-estimation constraints are absent. Seven canonical "
        "background normalizations are unconstrained global rate parameters, "
        "while shape/weight nuisances and autoMCStats remain. The evaluated "
        "grid has mStop≤1800 GeV. "
        + "Official topology-matched CMS-SUS-19-010 observed and expected "
        "Run-2 contours are overlaid for every displayed signal model. "
        + "Coverage: "
        + "; ".join(coverage_notes)
        + ". "
        + " · ".join(links)
        + ".</p></section>"
        + NOTICE_END
    )
    card_html = CARDS_START + "".join(cards) + CARDS_END

    index_path = page_dir / "index.html"
    page = index_path.read_text()
    page = re.sub(
        re.escape(NOTICE_START) + r".*?" + re.escape(NOTICE_END),
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
    replaced_names = {record["name"] for record in records}
    current = [
        record
        for record in summary.get("records", [])
        if record.get("family") != "limits"
        or record.get("name") not in replaced_names
    ]
    summary["records"] = current + records
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
    summary["statistical_results"] = {
        "status": "complete_free_background_2024",
        "model": "free_background_global_process_normalizations",
        "background_rate_parameters": 7,
        "external_background_constraints": [],
        "autoMCStats": 10,
        "max_mstop_GeV": 1800,
        "signals": prior_signals,
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
