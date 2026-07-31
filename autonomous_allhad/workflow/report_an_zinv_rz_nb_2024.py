#!/usr/bin/env python3
"""Build the Run-2-style R_Z(N_b) dilepton normalization report.

The report intentionally excludes bin-by-bin post-fit reconstructions of the
same on/off-Z counts used to determine R_Z and R_T.  Such reconstructions are
algebraically saturated and are not closure tests.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from build_an_zinv_factors_2024 import plot_mll
from build_an_zinv_measurement_inputs_2024 import finalize_rz, merge_tree


hep.style.use("CMS")

CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}
CHANNELS = ("DY2E", "DY2M")
GROUPS = ("Nb1", "Nb2plus")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def validate_channel(payload: dict[str, Any], channel: str, path: Path) -> None:
    if payload.get("status") != "feature_stage_complete":
        raise RuntimeError(f"{path}: incomplete status {payload.get('status')}")
    summary = payload.get("summary") or {}
    if summary.get("missing_roots"):
        raise RuntimeError(f"{path}: missing ROOT inputs")
    if int(summary.get("input_roots", -1)) != int(
        summary.get("completed_roots", -2)
    ):
        raise RuntimeError(f"{path}: ROOT accounting does not close")
    available = set((payload.get("rz_high_raw") or {}).keys())
    if channel not in available:
        raise RuntimeError(f"{path}: channel {channel} is absent")


def combine_highdm(
    ee: dict[str, Any], mumu: dict[str, Any]
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    mll: dict[str, Any] = {}
    for payload, channel in ((ee, "DY2E"), (mumu, "DY2M")):
        merge_tree(
            raw,
            {channel: (payload.get("rz_high_raw") or {}).get(channel, {})},
        )
        merge_tree(
            mll,
            {channel: (payload.get("mll_high") or {}).get(channel, {})},
        )
    return {
        "rz_high_raw": raw,
        "rz_high": finalize_rz(raw),
        "mll_high": mll,
    }


def plot_rt(
    rz: dict[str, Any], selection: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    x = np.arange(len(GROUPS), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4"}
    offsets = {"DY2E": -0.10, "DY2M": 0.10}
    labels = {"DY2E": r"$ee$", "DY2M": r"$\mu\mu$"}
    for channel in CHANNELS:
        values: list[float] = []
        errors: list[float] = []
        positions: list[float] = []
        for index, group in enumerate(GROUPS):
            record = ((rz.get("channels") or {}).get(channel) or {}).get(
                group, {}
            )
            if record.get("status") != "complete":
                continue
            positions.append(float(x[index] + offsets[channel]))
            values.append(float(record["RT"]))
            errors.append(float(record["RT_stat"]))
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="o",
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=labels[channel],
        )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_T$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=20)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rt_{selection}")


def plot_rz_nb(
    rz: dict[str, Any], selection: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    x = np.arange(len(GROUPS), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4", "combined": "#111111"}
    offsets = {"DY2E": -0.16, "DY2M": 0.0, "combined": 0.16}
    labels = {"DY2E": r"$ee$", "DY2M": r"$\mu\mu$"}
    for channel in CHANNELS:
        records = [rz["channels"][channel][group] for group in GROUPS]
        ax.errorbar(
            x + offsets[channel],
            [record["RZ"] for record in records],
            yerr=[record["RZ_stat"] for record in records],
            fmt="o",
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=labels[channel],
        )
    combined = [rz["combined"][group] for group in GROUPS]
    ax.errorbar(
        x + offsets["combined"],
        [record["RZ"] for record in combined],
        yerr=[record["RZ_stat"] for record in combined],
        fmt="s",
        color=colors["combined"],
        lw=2.2,
        capsize=4,
        label="Combined",
    )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_Z$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=20)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rz_{selection}")


def value_cell(record: dict[str, Any], key: str) -> str:
    if record.get("status") != "complete":
        return "unavailable"
    return f"{float(record[key]):.3f} &plusmn; {float(record[key + '_stat']):.3f}"


def write_html(
    output_dir: Path, result: dict[str, Any], selection: str
) -> None:
    suffix = "highdm" if selection == "highdm" else "lowdm"
    key = "rz_high" if selection == "highdm" else "rz_low"
    mll_key = "mll_high" if selection == "highdm" else "mll_low"
    rz = result[key]
    rows: list[str] = []
    for group, group_label in (("Nb1", "N<sub>b</sub> = 1"), ("Nb2plus", "N<sub>b</sub> &ge; 2")):
        ee = rz["channels"]["DY2E"][group]
        mm = rz["channels"]["DY2M"][group]
        combined = rz["combined"][group]
        rows.append(
            "<tr>"
            f"<td>{group_label}</td>"
            f"<td>{value_cell(ee, 'RZ')}</td>"
            f"<td>{value_cell(mm, 'RZ')}</td>"
            f"<td>{value_cell(combined, 'RZ')}</td>"
            f"<td>{value_cell(ee, 'RT')}</td>"
            f"<td>{value_cell(mm, 'RT')}</td>"
            "</tr>"
        )
    cards = [
        (f"rz_{suffix}.png", "R<sub>Z</sub>(N<sub>b</sub>)"),
        (f"rt_{suffix}.png", "R<sub>T</sub>(N<sub>b</sub>)"),
        (f"mll_{suffix}_dy2e_nb1.png", "Dielectron, N<sub>b</sub> = 1"),
        (f"mll_{suffix}_dy2e_nb2plus.png", "Dielectron, N<sub>b</sub> &ge; 2"),
        (f"mll_{suffix}_dy2m_nb1.png", "Dimuon, N<sub>b</sub> = 1"),
        (f"mll_{suffix}_dy2m_nb2plus.png", "Dimuon, N<sub>b</sub> &ge; 2"),
    ]
    card_html = "\n".join(
        f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{label}"></a><figcaption>{label}</figcaption></figure>'
        for name, label in cards
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024 RZ(Nb) measurement</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1320px;margin:0 auto;padding:24px;color:#161616}}
h1{{font-size:2rem}} .note{{background:#f2f5f8;border-left:5px solid #4178be;padding:14px 18px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:22px 0}} th,td{{border:1px solid #bbb;padding:10px;text-align:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}} figure{{margin:0}} img{{width:100%;height:auto}} figcaption{{font-size:1rem;padding:6px 0 18px}}
</style></head><body>
<h1>2024 {"high" if selection == "highdm" else "low"}-&Delta;m R<sub>Z</sub>(N<sub>b</sub>) measurement</h1>
<div class="note">Run-2-style on/off-Z matrix measurement. R<sub>Z</sub> and R<sub>T</sub> are determined independently in ee and &mu;&mu; for N<sub>b</sub>=1 and N<sub>b</sub>&ge;2. The combined R<sub>Z</sub> uses inverse-variance weighting. No U<sub>T</sub>-dependent R<sub>Z</sub> and no saturated post-fit reconstruction are shown.</div>
<table><thead><tr><th>Category</th><th>R<sub>Z</sub>(ee)</th><th>R<sub>Z</sub>(&mu;&mu;)</th><th>Combined R<sub>Z</sub></th><th>R<sub>T</sub>(ee)</th><th>R<sub>T</sub>(&mu;&mu;)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid">{card_html}</div>
</body></html>"""
    (output_dir / "index.html").write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee", type=Path, required=True)
    parser.add_argument("--mumu", type=Path, required=True)
    parser.add_argument("--selection", choices=("highdm", "lowdm"), default="highdm")
    parser.add_argument("--low-exact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ee = read_json(args.ee)
    mumu = read_json(args.mumu)
    validate_channel(ee, "DY2E", args.ee)
    validate_channel(mumu, "DY2M", args.mumu)
    if args.selection == "highdm":
        result = combine_highdm(ee, mumu)
        rz_key = "rz_high"
        mll_key = "mll_high"
    else:
        if args.low_exact is None:
            raise SystemExit("--low-exact is required for lowdm")
        low = read_json(args.low_exact)
        if low.get("status") != "complete":
            raise SystemExit(f"{args.low_exact}: exact Low-dM input is incomplete")
        result = {"rz_low": low["rz_low"], "mll_low": low["mll_low"]}
        rz_key = "rz_low"
        mll_key = "mll_low"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "rz": plot_rz_nb(result[rz_key], args.selection, output_dir),
        "rt": plot_rt(result[rz_key], args.selection, output_dir),
        "mll": plot_mll(result, mll_key, args.selection, output_dir),
    }
    payload = {
        "schema_version": "an_zinv_rz_nb_2024_v1",
        "status": "complete",
        "method": {
            "mass_windows": {
                "on": "81 < mll < 101 GeV",
                "off": "50 < mll < 81 GeV or mll > 101 GeV",
            },
            "categories": ["Nb1", "Nb2plus"],
            "channels": list(CHANNELS),
            "combination": "inverse-variance weighted RZ across ee and mumu",
            "ut_dependent_rz": False,
        },
        "inputs": {"ee": str(args.ee), "mumu": str(args.mumu)},
        rz_key: result[rz_key],
        "plots": plots,
    }
    write_json(output_dir / "summary.json", payload)
    write_html(output_dir, result, args.selection)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "combined": result[rz_key]["combined"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
