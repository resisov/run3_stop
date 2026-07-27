#!/usr/bin/env python3
"""Publish a validated 2024 High-dM + Low-dM expected-limit result."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


NOTICE_START = "<!-- combined-limit-update:start -->"
NOTICE_END = "<!-- combined-limit-update:end -->"
CARD_START = "<!-- combined-limit-card:start -->"
CARD_END = "<!-- combined-limit-card:end -->"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()

    page_dir = args.page_dir
    result_dir = args.result_dir
    manifest = json.loads(
        (result_dir / "combine_input_manifest.json").read_text()
    )
    schema = manifest.get("schema_version") or manifest.get("schema") or ""
    transfer_factor_model = "nb_recoil_tf" in schema
    an_zinv_model = "an_zinv" in schema
    if transfer_factor_model or an_zinv_model:
        highdm_bins = int(manifest["channels"]["highdm_signal"])
        lowdm_bins = int(manifest["channels"]["lowdm_signal"])
        expected_total = int(manifest["channels"]["total"])
        collection = manifest["limits"]
        max_mstop = int(manifest["max_mstop"])
    else:
        highdm_bins = int(manifest["highdm"]["signal_bins"])
        lowdm_bins = int(manifest["lowdm"]["signal_bins"])
        expected_total = 5 * 6 + highdm_bins + 6 * lowdm_bins
        collection = manifest["limit_collection"]
        max_mstop = int(manifest["max_mstop_inclusive"])
    requested_points = int(collection.get("requested_point_count", 0))
    collected_points = int(collection.get("collected_point_count", 0))
    missing_points = list(collection.get("missing_points") or [])
    collection_complete = (
        collection.get("status") == "complete"
        and collected_points == requested_points
    )
    documented_boundary_partial = (
        collection.get("status") == "partial"
        and collected_points == requested_points - 1
        and missing_points == ["mStop1800_mLSP1600"]
    )
    checks = {
        "status": (
            manifest.get("status") == "combine_outputs_complete"
            or documented_boundary_partial
        ),
        "year_model": "2024" in schema,
        "highdm_60": highdm_bins == 60,
        "lowdm_positive": lowdm_bins > 0,
        "mass_grid_valid": collection_complete or documented_boundary_partial,
        "mstop_cap": max_mstop == 1800,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "limit publication validation failed: " + ", ".join(failed)
        )

    if transfer_factor_model:
        source_png = (
            result_dir
            / "expected_limit_contour_highdm60_lowdm34_nb_recoil_tf.png"
        )
        source_pdf = source_png.with_suffix(".pdf")
        source_json = result_dir / "expected_limits.json"
    elif an_zinv_model:
        source_png = (
            result_dir
            / "expected_limit_highdm60_lowdm34_an_zinv_x1800.png"
        )
        source_pdf = source_png.with_suffix(".pdf")
        source_json = result_dir / "expected_limits.json"
    else:
        source_png = Path(manifest["contour_png"])
        source_pdf = Path(manifest["contour_pdf"])
        source_json = Path(manifest["expected_limits"])
    for source in (source_png, source_pdf, source_json):
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty limit artifact: {source}")

    stem = (
        f"expected_limit_highdm{highdm_bins}_lowdm{lowdm_bins}_"
        + (
            "an_zinv_"
            if an_zinv_model
            else ("nb_recoil_tf_" if transfer_factor_model else "")
        )
        + "x1800"
    )
    plot_dir = page_dir / "plots/limits"
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_png, plot_dir / f"{stem}.png")
    shutil.copy2(source_pdf, plot_dir / f"{stem}.pdf")
    data_name = (
        f"highdm{highdm_bins}_lowdm{lowdm_bins}_"
        + (
            "an_zinv_"
            if an_zinv_model
            else ("nb_recoil_tf_" if transfer_factor_model else "")
        )
        + "expected_limits.json"
    )
    shutil.copy2(source_json, page_dir / data_name)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    transfer_notice = (
        "The background prediction is constrained by the published "
        f"{manifest['transfer_parameter_count']} matched "
        "N<sub>b</sub>×recoil transfer parameters. "
        if transfer_factor_model
        else (
            "The Z→νν prediction follows the AN model: RZ is measured with "
            "the on/off-Z dilepton matrix, while the Q-normalized photon CR "
            "constrains the bin-wise Sγ shape. Lost-lepton and QCD retain "
            "direct CR/SR rate constraints. "
            if an_zinv_model
            else ""
        )
    )
    grid_notice = (
        f"The expected grid contains all {collected_points} requested points "
        if collection_complete
        else (
            f"The expected grid contains {collected_points}/{requested_points} "
            "requested points; the timed-out near-diagonal boundary point "
            "(mStop,mLSP)=(1800,1600) GeV is explicitly omitted. "
        )
    )
    notice = (
        f"{NOTICE_START}<section class='update' style='max-width:1500px;"
        "margin:14px auto 0;padding:12px 18px;background:#fff;"
        "border:1px solid #d6dcdf;border-radius:6px'>"
        f"<strong>Adopted 2024 model: High-dM {highdm_bins} + Low-dM "
        f"{lowdm_bins} bins</strong><p>DY PTLL-400 and PTLL-600 are excluded. "
        "Every Low-dM CR and SR explicitly requires Nb≥1; the two leading "
        "Nb=0 categories were removed. "
        f"{transfer_notice}"
        f"{grid_notice}"
        "The evaluated grid has mStop≤1800 GeV and includes the established "
        "Run-2 contour. "
        f"<a href='plots/limits/{stem}.pdf'>Expected-limit PDF</a> · "
        f"<a href='{data_name}'>machine-readable limits</a>.</p></section>"
        f"{NOTICE_END}"
    )
    card = (
        f"{CARD_START}<a class='plot' data-family='limits' "
        "data-kind='Overview' data-search='2024 expected limit high-dm "
        f"{highdm_bins} low-dm {lowdm_bins} nbge1 dy ptll 100 200' "
        f"href='plots/limits/{stem}.pdf'><img "
        f"src='plots/limits/{stem}.png' loading='lazy' "
        "alt='2024 expected stop limit'><span>2024 expected limit · "
        f"High-dM {highdm_bins} + Low-dM {lowdm_bins} · mStop≤1800 GeV"
        f"</span></a>{CARD_END}"
    )

    summary_path = page_dir / "page_summary.json"
    summary = json.loads(summary_path.read_text())
    records = [
        record for record in summary["records"]
        if record.get("family") != "limits"
    ]
    records.append(
        {
            "family": "limits",
            "family_label": "Expected limits",
            "kind": "Overview",
            "name": stem,
            "pdf": f"plots/limits/{stem}.pdf",
            "png": f"plots/limits/{stem}.png",
            "region": (
                f"2024 High-dM {highdm_bins} + Low-dM {lowdm_bins}"
            ),
            "variable": "mStop mLSP expected 95% CL",
        }
    )
    summary["records"] = records
    summary["generated_at"] = now
    summary["limit_update"] = {
        "status": (
            "complete" if collection_complete else "documented_boundary_partial"
        ),
        "updated_at": now,
        "highdm_signal_bins": highdm_bins,
        "lowdm_signal_bins": lowdm_bins,
        "total_analysis_bins": expected_total,
        "mass_point_count": manifest["mass_point_count"],
        "max_mstop_inclusive": 1800,
        "dy_ptll_policy": "ptll100_200",
        "lowdm_selection": "all CR/SR require Nb>=1",
        "removed_lowdm_categories": [
            "Nb0_Nj2to5_PISR500plus",
            "Nb0_Nj6plus_PISR500plus",
        ],
        "transfer_factor_model": transfer_factor_model,
        "an_zinv_model": an_zinv_model,
        "transfer_parameter_count": manifest.get("transfer_parameter_count", 0),
        "rate_parameter_count": manifest.get("rate_parameter_count", 0),
        "limit_grid_status": (
            "complete" if collection_complete else "documented_boundary_partial"
        ),
        "requested_mass_point_count": requested_points,
        "collected_mass_point_count": collected_points,
        "missing_mass_points": missing_points,
        "plot": f"plots/limits/{stem}.png",
        "expected_limits": data_name,
    }
    summary["plot_counts"] = {
        **(summary.get("plot_counts") or {}),
        "limits": 1,
        "total": len(records),
    }
    summary.setdefault("statistical_results", {}).update(
        {
            "combined": f"plots/limits/{stem}.png",
            "status": (
                "expected_limit_complete"
                if collection_complete
                else "expected_limit_documented_boundary_partial"
            ),
            "updated_at": now,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
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
    pattern = re.escape(CARD_START) + r".*?" + re.escape(CARD_END)
    if re.search(pattern, html, flags=re.DOTALL):
        html = re.sub(pattern, card, html, flags=re.DOTALL)
    elif "</div></main>" in html:
        html = html.replace("</div></main>", card + "</div></main>", 1)
    else:
        raise RuntimeError("limit card insertion point is missing")
    html = re.sub(
        r"Frozen 2024 nominal snapshot · \d+ plots · generated [^<]+",
        f"Frozen 2024 nominal snapshot · {len(records)} plots · generated {now}",
        html,
        count=1,
    )
    index_path.write_text(html)
    print(
        json.dumps(
            {
                "status": (
                    "complete"
                    if collection_complete
                    else "documented_boundary_partial"
                ),
                "page": str(index_path),
                "limit": str(plot_dir / f"{stem}.png"),
                "records": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
