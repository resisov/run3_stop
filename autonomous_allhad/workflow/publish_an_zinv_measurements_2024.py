#!/usr/bin/env python3
"""Publish validated AN-style Z->nunu measurement plots on the 2024 page."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


SECTION_START = "<!-- transfer-factor-update:start -->"
SECTION_END = "<!-- transfer-factor-update:end -->"


def copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def caption(name: str) -> str:
    if name == "rz_highdm":
        return r"High-dM \(R_Z\) from the on/off-Z dilepton matrix."
    if name == "rz_lowdm":
        return r"Low-dM \(R_Z\) from the on/off-Z dilepton matrix."
    if name == "photon_q_normalization":
        return (
            r"Photon-CR normalization \(Q\), split into the adopted "
            r"High- and Low-dM \(N_b\) groups."
        )
    if name == "sgamma_highdm":
        return r"High-dM photon-CR \(S_{\gamma,i}\) versus recoil."
    if name == "zgamma_double_ratio_highdm":
        return r"High-dM normalized \(Z/\gamma\) shape double ratio."
    if name == "sgamma_lowdm_nb_isr_shared":
        return (
            r"Low-dM photon-CR \(S_{\gamma,i}\), shared within the four "
            r"\(N_b\times p_T^{ISR}\) groups."
        )
    if name == "zgamma_double_ratio_lowdm_nb_isr_shared":
        return (
            r"Low-dM normalized \(Z/\gamma\) shape double ratio in the four "
            r"\(N_b\times p_T^{ISR}\) groups."
        )
    if name.startswith("sgamma_lowdm_"):
        return r"Low-dM photon-CR \(S_{\gamma,i}\) in one analysis family."
    if name.startswith("zgamma_double_ratio_lowdm_"):
        return r"Low-dM normalized \(Z/\gamma\) shape double ratio."
    if name.startswith("mll_"):
        return "Dilepton-mass input to the on/off-Z matrix."
    return name.replace("_", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-dir", type=Path, required=True)
    parser.add_argument("--factor-json", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()

    factors = json.loads(args.factor_json.read_text())
    if factors.get("status") != "complete":
        raise RuntimeError(f"factors incomplete: {factors.get('status')}")
    plot_sources = sorted(args.plot_dir.glob("*.png"))
    stems = [path.stem for path in plot_sources]
    required = {
        "rz_highdm",
        "rz_lowdm",
        "photon_q_normalization",
        "sgamma_highdm",
        "sgamma_lowdm_nb_isr_shared",
        "zgamma_double_ratio_highdm",
        "zgamma_double_ratio_lowdm_nb_isr_shared",
    }
    missing = sorted(required - set(stems))
    if missing:
        raise RuntimeError("required Zinv plots missing: " + ", ".join(missing))
    target_plot_dir = args.page_dir / "plots/zinv_measurements"
    expected_assets = {
        path.with_suffix(suffix).name
        for path in plot_sources
        for suffix in (".png", ".pdf")
    }
    if target_plot_dir.exists():
        for existing in target_plot_dir.iterdir():
            if (
                existing.is_file()
                and existing.suffix in {".png", ".pdf"}
                and existing.name not in expected_assets
            ):
                existing.unlink()
    records = []
    for png in plot_sources:
        pdf = png.with_suffix(".pdf")
        copy(png, target_plot_dir / png.name)
        copy(pdf, target_plot_dir / pdf.name)
        records.append(
            {
                "family": "zinv_measurements",
                "family_label": "Z invisible measurements",
                "kind": "Measurement",
                "name": png.stem,
                "pdf": f"plots/zinv_measurements/{pdf.name}",
                "png": f"plots/zinv_measurements/{png.name}",
                "region": (
                    "2024 Low-dM"
                    if "lowdm" in png.stem
                    else "2024 High-dM"
                ),
                "variable": caption(png.stem),
            }
        )
    data_target = args.page_dir / "data/an_zinv_factors_2024.json"
    copy(args.factor_json, data_target)

    cards = []
    for record in records:
        cards.append(
            "<a class='tf-card' "
            f"href='{record['pdf']}'><figure><img "
            f"src='{record['png']}' loading='lazy' "
            f"alt='{record['name']}'><figcaption>{record['variable']}"
            "</figcaption></figure></a>"
        )
    retained = """
<a class='tf-card' href='plots/transfer_factors/transfer_factor_top_llcr_vs_recoil.pdf'><figure><img src='plots/transfer_factors/transfer_factor_top_llcr_vs_recoil.png' loading='lazy' alt='Top component of lost-lepton transfer factors'><figcaption>Top component of the shared lost-lepton CR constraint.</figcaption></figure></a>
<a class='tf-card' href='plots/transfer_factors/transfer_factor_w_llcr_vs_recoil.pdf'><figure><img src='plots/transfer_factors/transfer_factor_w_llcr_vs_recoil.png' loading='lazy' alt='W component of lost-lepton transfer factors'><figcaption>W+jets component of the shared lost-lepton CR constraint.</figcaption></figure></a>
<a class='tf-card' href='plots/transfer_factors/transfer_factor_qcd_qcdcr_vs_recoil.pdf'><figure><img src='plots/transfer_factors/transfer_factor_qcd_qcdcr_vs_recoil.png' loading='lazy' alt='QCD control-region transfer factors'><figcaption>QCD SR/QCD-CR transfer-factor diagnostic.</figcaption></figure></a>
""".strip()
    section = (
        f"{SECTION_START}<section class='tf-section'>"
        "<h2>2024 background constraints</h2>"
        "<p>Lost-lepton and QCD use direct CR/SR rate constraints. "
        "The Z→νν estimate follows the analysis-note construction "
        "N<sub>pred</sub>=R<sub>Z</sub>S<sub>γ,i</sub>N<sub>MC,i</sub>: "
        "R<sub>Z</sub> comes from the on/off-Z dielectron and dimuon matrix, "
        "and S<sub>γ,i</sub> comes from the photon CR after Q normalization "
        "within the adopted N<sub>b</sub> and p<sub>T</sub><sup>ISR</sup> "
        "groups. Dilepton CRs are "
        "measurement-only and are not direct Z transfer-factor likelihood "
        "channels. <a href='data/an_zinv_factors_2024.json'>Machine-readable "
        "measurement</a>.</p><div class='tf-grid'>"
        + retained
        + "".join(cards)
        + f"</div></section>{SECTION_END}"
    )
    index_path = args.page_dir / "index.html"
    html = index_path.read_text()
    pattern = re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END)
    if not re.search(pattern, html, flags=re.DOTALL):
        raise RuntimeError("background-constraint section marker missing")
    index_path.write_text(
        re.sub(pattern, lambda _match: section, html, count=1, flags=re.DOTALL)
    )

    summary_path = args.page_dir / "page_summary.json"
    summary = json.loads(summary_path.read_text())
    retained_records = [
        record
        for record in summary["records"]
        if record.get("family") != "zinv_measurements"
        and record.get("name")
        not in {
            "transfer_factor_zinv_gcr_vs_recoil",
            "transfer_factor_zinv_dy2e_vs_recoil",
            "transfer_factor_zinv_dy2m_vs_recoil",
        }
    ]
    summary["records"] = retained_records + records
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary["generated_at"] = now
    summary["transfer_factor_update"] = {
        "status": "complete",
        "used_in_datacard": ["shared lost-lepton RLL", "QCD RQCD"],
        "zinv_direct_transfer_factors_removed": True,
    }
    summary["zinv_measurement_update"] = {
        "status": "complete",
        "updated_at": now,
        "prediction": "N_Zinv_pred = RZ * Sgamma_i * N_Zinv_MC_i",
        "RZ_source": "on/off-Z dielectron+dimuon 2x2 matrix",
        "Sgamma_source": "Q-normalized photon CR",
        "dilepton_likelihood_channels": False,
        "plot_count": len(records),
        "data": "data/an_zinv_factors_2024.json",
    }
    summary.setdefault("plot_counts", {})["zinv_measurements"] = len(records)
    summary["plot_counts"]["total"] = len(summary["records"])
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "plots": len(records),
                "page": str(index_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
