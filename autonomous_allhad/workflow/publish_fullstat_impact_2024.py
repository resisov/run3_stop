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
    highdm_bins = int(analysis["highdm_signal_bins"])
    lowdm_bins = int(analysis["lowdm_signal_bins"])
    transfer_factor_model = analysis.get("model") == "nb_recoil_transfer_factor"
    an_zinv_model = analysis.get("model") == "an_zinv"
    free_background_model = analysis.get("model") == "free_background"
    lowdm_sgamma_sharing = (
        ((analysis.get("model_details") or {}).get("lowdm_sgamma_sharing"))
        or {}
    )
    stem = (
        f"impacts_2024_highdm{highdm_bins}_lowdm{lowdm_bins}_"
        + (
            "an_zinv_"
            if an_zinv_model
            else ("nb_recoil_tf_" if transfer_factor_model else "")
        )
        + "mStop1200_mLSP500_full_mcstat"
    )

    required = {
        "status": validation.get("status") == "complete",
        "year": analysis.get("year") == 2024,
        "highdm_bins": highdm_bins > 0,
        "lowdm_bins": lowdm_bins > 0,
        "benchmark": analysis.get("benchmark")
        == {"mStop_GeV": 1200, "mLSP_GeV": 500},
        "all_fits": scan.get("parameter_count", 0) > 0
        and scan.get("missing_fit_count") == 0
        and scan.get("invalid_fit_count") == 0,
        "statistical_nuisances": scan.get("statistical_parameter_count", 0)
        > 0,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError("publication validation failed: " + ", ".join(failed))

    plot_dir = page_dir / "plots/impacts"
    data_dir = page_dir / "data"
    sources = {
        f"{stem}.png": bundle / f"{stem}.png",
        f"{stem}.pdf": bundle / f"{stem}.pdf",
        f"{stem}_summary.png": bundle / f"{stem}_summary.png",
        f"{stem}_summary.pdf": bundle / f"{stem}_summary.pdf",
    }
    for name, source in sources.items():
        copy(source, plot_dir / name)

    data_sources = {
        f"{stem}.json": bundle / "impacts_mStop1200_mLSP500.json",
        f"{stem}_validation.json": bundle / "impact_validation.json",
        f"{stem}_labels.json": bundle / "impact_labels_2024.json",
    }
    for name, source in data_sources.items():
        copy(source, data_dir / name)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    poi_fit = scan["poi"]["fit"]
    stat_count = int(scan["statistical_parameter_count"])
    nonstat_count = int(scan["nonstatistical_parameter_count"])
    parameter_count = int(scan["parameter_count"])
    channel_count = int(analysis["channels"])
    total_bins = int(analysis["total_analysis_bins"])
    object_shapes = (
        (analysis.get("systematics") or {}).get("object_shape_variations")
        or []
    )
    object_scope = (
        " Object-shape templates included: "
        + ", ".join(str(name) for name in object_shapes)
        + "."
        if isinstance(object_shapes, list) and object_shapes
        else " Object-shape templates are absent from this card."
    )
    transfer_scope = (
        " The background model uses the published bin-by-bin "
        "N_b×recoil transfer-factor constraints."
        if transfer_factor_model
        else (
            " The Z→νν background uses the AN structure: RZ from the "
            "on/off-Z dilepton matrix and Sγ from the Q-normalized photon CR; "
            + (
                f"Low-dM Sγ is represented by "
                f"{int(lowdm_sgamma_sharing['recoil_shape_parameter_count'])} "
                "recoil-shape parameters shared within four Nb×ISR-pT groups; "
                if lowdm_sgamma_sharing
                else ""
            )
            + "lost-lepton and QCD retain direct CR/SR rate constraints."
            if an_zinv_model
            else (
                " The seven canonical background process normalizations are "
                "global unconstrained rate parameters; external background-"
                "estimation constraints are intentionally absent."
                if free_background_model
                else ""
            )
        )
    )
    region_scope = (
        "and the corresponding control-region observations"
        if free_background_model
        else "with their control-region constraints"
    )
    notice = (
        f"{NOTICE_START}"
        "<section class='update' style='max-width:1500px;margin:14px auto 0;"
        "padding:12px 18px;background:#fff;border:1px solid #d6dcdf;"
        "border-radius:6px'>"
        "<strong>2024-only High-dM + Low-dM nuisance impacts "
        "for (mStop, mLSP)=(1200, 500) GeV</strong>"
        f"<p>The expected Asimov fit uses the High-dM {highdm_bins}-bin and Low-dM "
        f"{lowdm_bins}-bin signal regions {region_scope} "
        f"({channel_count} channels, {total_bins} analysis bins). The Poisson likelihood includes "
        f"data-counting statistics, and {stat_count} individual autoMCStats nuisance "
        f"parameters are scanned alongside {nonstat_count} luminosity/rate/weight "
        f"nuisances. All {parameter_count} fits are present and valid. "
        f"The fitted signal strength is r={poi_fit[1]:.2f}"
        f"<sup>+{poi_fit[2] - poi_fit[1]:.2f}</sup>"
        f"<sub>−{poi_fit[1] - poi_fit[0]:.2f}</sub>. "
        f"{transfer_scope}{object_scope} "
        f"<a href='plots/impacts/{stem}.pdf'>Multipage impact plot</a> · "
        f"<a href='data/{stem}.json'>Combine impact JSON</a> · "
        f"<a href='data/{stem}_validation.json'>validation report</a>.</p>"
        "</section>"
        f"{NOTICE_END}"
    )
    card = (
        f"{CARD_START}"
        "<a class='plot' data-family='impacts' data-kind='Overview' "
        "data-search='2024 only full statistical autoMCStats impact "
        f"high-dm {highdm_bins} low-dm {lowdm_bins} mstop1200 mlsp500' "
        f"href='plots/impacts/{stem}.pdf'>"
        f"<img src='plots/impacts/{stem}.png' loading='lazy' "
        "alt='2024-only High-dM plus Low-dM nuisance impacts including "
        "individual autoMCStats parameters'>"
        f"<span>2024-only impact · High-dM {highdm_bins} + Low-dM {lowdm_bins} · "
        f"(1200, 500) GeV · {stat_count} autoMCStats + {nonstat_count} other nuisances</span></a>"
        f"{CARD_END}"
    )

    summary_path = page_dir / "page_summary.json"
    page_summary = json.loads(summary_path.read_text())
    replaced_families = {"impacts"}
    if transfer_factor_model:
        replaced_families.add("transfer_factors")
    base_records = [
        record
        for record in page_summary["records"]
        if record.get("family") not in replaced_families
    ]
    transfer_records = []
    if transfer_factor_model:
        transfer_specs = (
            ("top_llcr", "Top: SR / lost-lepton CR"),
            ("w_llcr", "W+jets: SR / lost-lepton CR"),
            ("qcd_qcdcr", "QCD: SR / QCD CR"),
            ("zinv_gcr", "Z→νν / Photon+jet GCR"),
            ("zinv_dy2e", "Z→νν / dielectron CR"),
            ("zinv_dy2m", "Z→νν / dimuon CR"),
        )
        for key, label in transfer_specs:
            name = f"transfer_factor_{key}_vs_recoil"
            transfer_records.append(
                {
                    "family": "transfer_factors",
                    "family_label": "Transfer factors",
                    "kind": "Overview",
                    "name": name,
                    "pdf": f"plots/transfer_factors/{name}.pdf",
                    "png": f"plots/transfer_factors/{name}.png",
                    "region": "2024 High-dM and Low-dM",
                    "variable": label,
                }
            )
    publication_plot_count = len(base_records) + len(transfer_records) + 1

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
    if re.search(replacement, html, flags=re.DOTALL):
        html = re.sub(replacement, card, html, flags=re.DOTALL)
    elif "</div></main>" in html:
        html = html.replace("</div></main>", card + "</div></main>", 1)
    else:
        raise RuntimeError("impact card insertion point is missing from index.html")
    html = re.sub(
        r"Frozen 2024 nominal snapshot · \d+ plots · generated [^<]+",
        (
            "Frozen 2024 nominal snapshot · "
            f"{publication_plot_count} plots · generated {now}"
        ),
        html,
        count=1,
    )
    index_path.write_text(html)

    records = base_records + transfer_records
    records.append(
        {
            "family": "impacts",
            "family_label": "Nuisance impacts",
            "kind": "Overview",
            "name": stem,
            "pdf": f"plots/impacts/{stem}.pdf",
            "png": f"plots/impacts/{stem}.png",
            "region": (
                f"2024 High-dM {highdm_bins} + Low-dM {lowdm_bins}"
            ),
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
        "channels": channel_count,
        "total_analysis_bins": total_bins,
        "highdm_signal_bins": highdm_bins,
        "lowdm_signal_bins": lowdm_bins,
        "poi": scan["poi"],
        "parameter_count": parameter_count,
        "statistical_parameter_count": stat_count,
        "nonstatistical_parameter_count": nonstat_count,
        "missing_fit_count": 0,
        "invalid_fit_count": 0,
        "statistical_model": {
            "data_counting": "Poisson likelihood",
            "mc_statistics": "individual autoMCStats nuisance fits",
            "autoMCStats_threshold": analysis["autoMCStats_threshold"],
            "background_normalization": (
                "seven global unconstrained process rate parameters"
                if free_background_model
                else "control-region-constrained background model"
            ),
        },
        "current_model_scope": (
            f"2024 luminosity plus {max(0, nonstat_count - 1)} "
            "rate/weight or object-shape nuisances and individual "
            f"MC-statistical nuisances."
            + (
                f" Low-dM Sγ uses "
                f"{int(lowdm_sgamma_sharing['recoil_shape_parameter_count'])} "
                "shared Nb×ISR-pT×recoil parameters."
                if lowdm_sgamma_sharing
                else ""
            )
            + object_scope
        ),
        "plot": f"plots/impacts/{stem}.png",
        "multipage_pdf": f"plots/impacts/{stem}.pdf",
        "summary_pdf": f"plots/impacts/{stem}_summary.pdf",
        "combine_json": f"data/{stem}.json",
        "validation": f"data/{stem}_validation.json",
        "labels": f"data/{stem}_labels.json",
    }
    if transfer_factor_model:
        page_summary["transfer_factor_update"] = {
            "status": "complete",
            "definition": "N_SR/N_CR",
            "binning": "matched Nb and pTmiss/hadronic-recoil bins",
            "highdm_nb": ["1", "2", ">=3"],
            "lowdm_nb": ["1", ">=2"],
            "plot_count": len(transfer_records),
            "data": "data/transfer_factors_2024_nb_recoil.json",
            "used_in_datacard": True,
        }
    statistical = page_summary.setdefault("statistical_results", {})
    limit_stem = (
        (
            f"expected_limit_t2tt_highdm{highdm_bins}_lowdm{lowdm_bins}_"
            "free_background_x1800"
        )
        if free_background_model
        else (
            f"expected_limit_highdm{highdm_bins}_lowdm{lowdm_bins}_"
            + (
                "an_zinv_"
                if an_zinv_model
                else ("nb_recoil_tf_" if transfer_factor_model else "")
            )
            + "x1800"
        )
    )
    statistical.update(
        {
            "combined": f"plots/limits/{limit_stem}.png",
            "impact": f"plots/impacts/{stem}.png",
            "impact_scope": page_summary["impact_update"][
                "current_model_scope"
            ],
            "impact_benchmark": {
                "mStop_GeV": 1200,
                "mLSP_GeV": 500,
            },
            "impact_parameters": {
                "total": parameter_count,
                "autoMCStats": stat_count,
                "other": nonstat_count,
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
