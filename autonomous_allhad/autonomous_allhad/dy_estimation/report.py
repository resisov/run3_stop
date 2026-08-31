#!/usr/bin/env python3
"""Build the Run-2-style R_Z(N_b) dilepton normalization report.

The report intentionally excludes bin-by-bin post-fit reconstructions of the
same on/off-Z counts used to determine R_Z and R_T.  Such reconstructions are
algebraically saturated and are not closure tests.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .model import finalize_rz, merge_tree


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_provenance(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "input_roots": int(summary.get("input_roots", 0)),
        "completed_roots": int(summary.get("completed_roots", 0)),
        "candidate_events": int(summary.get("candidate_events", 0)),
    }


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.name)
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
    markers = {"DY2E": "o", "DY2M": "^"}
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
            positions.append(float(x[index]))
            values.append(float(record["RT"]))
            errors.append(float(record["RT_stat"]))
        ax.errorbar(
            positions,
            values,
            xerr=np.full(len(positions), 0.5, dtype=float),
            yerr=errors,
            fmt=markers[channel],
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
    ax.legend(frameon=False, fontsize=24)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rt_{selection}")


def plot_rz_nb(
    rz: dict[str, Any], selection: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    x = np.arange(len(GROUPS), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4"}
    markers = {"DY2E": "o", "DY2M": "^"}
    labels = {"DY2E": r"$ee$", "DY2M": r"$\mu\mu$"}
    for channel in CHANNELS:
        records = [rz["channels"][channel][group] for group in GROUPS]
        ax.errorbar(
            x,
            [record["RZ"] for record in records],
            xerr=np.full(len(records), 0.5, dtype=float),
            yerr=[record["RZ_stat"] for record in records],
            fmt=markers[channel],
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=labels[channel],
        )
    combined = [rz["combined"][group] for group in GROUPS]
    ax.errorbar(
        x,
        [record["RZ"] for record in combined],
        xerr=np.full(len(combined), 0.5, dtype=float),
        yerr=[record["RZ_stat"] for record in combined],
        fmt="D",
        color="#111111",
        markerfacecolor="white",
        markeredgewidth=2.0,
        markersize=10,
        lw=2.4,
        capsize=5,
        zorder=5,
        label="Combined",
    )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_Z$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=24)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rz_{selection}")


def plot_mll(
    source: dict[str, Any],
    rz: dict[str, Any],
    selection: str,
    output_dir: Path,
    *,
    corrected: bool,
) -> list[str]:
    """Plot the matrix input, optionally after the fitted RZ/RT scaling."""

    paths: list[str] = []
    for channel in CHANNELS:
        for group in GROUPS:
            node = ((source.get(channel) or {}).get(group) or {})
            if not node:
                continue
            first = next(iter(node.values()))
            edges = np.asarray(first["edges"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            data = np.asarray(
                (node.get("data") or {}).get("sumw", []), dtype=float
            )
            data2 = np.asarray(
                (node.get("data") or {}).get("sumw2", []), dtype=float
            )
            zll = np.asarray(
                (node.get("zll") or {}).get("sumw", []), dtype=float
            )
            zll2 = np.asarray(
                (node.get("zll") or {}).get("sumw2", []), dtype=float
            )
            other = np.asarray(
                (node.get("other") or {}).get("sumw", []), dtype=float
            )
            other2 = np.asarray(
                (node.get("other") or {}).get("sumw2", []), dtype=float
            )
            if not len(data):
                continue

            if corrected:
                factors = ((rz.get("channels") or {}).get(channel) or {}).get(
                    group, {}
                )
                if factors.get("status") != "complete":
                    continue
                z_scale = float(factors["RZ"])
                other_scale = float(factors["RT"])
                zll = zll * z_scale
                zll2 = zll2 * z_scale * z_scale
                other = other * other_scale
                other2 = other2 * other_scale * other_scale

            total = zll + other
            total_error = np.sqrt(np.maximum(zll2 + other2, 0.0))
            data_error = np.sqrt(np.maximum(data2, 0.0))
            valid_ratio = total > 0.0
            ratio = np.full_like(data, np.nan)
            ratio_error = np.full_like(data, np.nan)
            ratio[valid_ratio] = data[valid_ratio] / total[valid_ratio]
            ratio_error[valid_ratio] = (
                data_error[valid_ratio] / total[valid_ratio]
            )
            relative_mc_error = np.zeros_like(total)
            relative_mc_error[valid_ratio] = (
                total_error[valid_ratio] / total[valid_ratio]
            )

            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(10.2, 10.2),
                sharex=True,
                gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
            )
            ax.stairs(
                zll,
                edges,
                fill=True,
                baseline=0.0,
                color="#35B6B4",
                edgecolor="black",
                linewidth=0.7,
                label="DY",
                zorder=1,
            )
            ax.stairs(
                total,
                edges,
                fill=True,
                baseline=zll,
                color="#6A625F",
                edgecolor="black",
                linewidth=0.7,
                label="Others",
                zorder=2,
            )
            ax.stairs(
                total + total_error,
                edges,
                baseline=np.maximum(total - total_error, 0.0),
                fill=True,
                facecolor="none",
                edgecolor="0.35",
                hatch="////",
                linewidth=0.0,
                label="Stat. unc.",
                zorder=3,
            )
            ax.errorbar(
                centers,
                data,
                yerr=data_error,
                fmt="o",
                color="black",
                ms=6,
                lw=2.0,
                capsize=2,
                label="Data",
                zorder=4,
            )
            ax.axvspan(81.0, 101.0, color="#FFD166", alpha=0.18)
            rax.axvspan(81.0, 101.0, color="#FFD166", alpha=0.18)
            rax.stairs(
                1.0 + relative_mc_error,
                edges,
                baseline=1.0 - relative_mc_error,
                fill=True,
                facecolor="0.75",
                edgecolor="0.55",
                alpha=0.55,
                linewidth=0.0,
            )
            rax.errorbar(
                centers[valid_ratio],
                ratio[valid_ratio],
                yerr=ratio_error[valid_ratio],
                fmt="o",
                color="black",
                ms=6,
                lw=2.0,
                capsize=2,
            )
            rax.axhline(1.0, color="black", lw=1.5)
            ax.set_ylabel("Events / bin", fontsize=30)
            rax.set_ylabel("Data/MC", fontsize=28)
            rax.set_xlabel(r"$m_{\ell\ell}$ (GeV)", fontsize=30, loc="right")
            ax.set_yscale("log")
            ax.set_ylim(1.0e-1, 1.0e3)
            rax.set_ylim(0.0, 2.0)
            for axis in (ax, rax):
                axis.set_xlim(float(edges[0]), float(edges[-1]))
                axis.set_xmargin(0)
                axis.tick_params(
                    which="major",
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=24,
                    length=9,
                )
                axis.tick_params(
                    which="minor",
                    direction="in",
                    top=True,
                    right=True,
                    length=5,
                )
                axis.minorticks_on()
            handles, labels = ax.get_legend_handles_labels()
            order = ["Stat. unc.", "Others", "DY", "Data"]
            ordered = [
                (handles[labels.index(label)], label)
                for label in order
                if label in labels
            ]
            ax.legend(
                [item[0] for item in ordered],
                [item[1] for item in ordered],
                frameon=False,
                fontsize=22,
                ncol=2,
                loc="upper right",
            )
            hep.cms.label(**CMS_LABEL, ax=ax)
            suffix = "_post" if corrected else ""
            paths.extend(
                save_figure(
                    fig,
                    output_dir
                    / f"mll_{selection}_{channel.lower()}_{group.lower()}{suffix}",
                )
            )
    return paths


def value_cell(record: dict[str, Any], key: str) -> str:
    if record.get("status") != "complete":
        return "unavailable"
    return f"{float(record[key]):.3f} &plusmn; {float(record[key + '_stat']):.3f}"


def write_html(
    output_dir: Path,
    result: dict[str, Any],
    selection: str,
    campaign_year: str = "2024",
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
        (f"mll_{suffix}_dy2e_nb1.png", "Dielectron, N<sub>b</sub> = 1, matrix input"),
        (f"mll_{suffix}_dy2e_nb1_post.png", "Dielectron, N<sub>b</sub> = 1, after R<sub>Z</sub>/R<sub>T</sub>"),
        (f"mll_{suffix}_dy2e_nb2plus.png", "Dielectron, N<sub>b</sub> &ge; 2, matrix input"),
        (f"mll_{suffix}_dy2e_nb2plus_post.png", "Dielectron, N<sub>b</sub> &ge; 2, after R<sub>Z</sub>/R<sub>T</sub>"),
        (f"mll_{suffix}_dy2m_nb1.png", "Dimuon, N<sub>b</sub> = 1, matrix input"),
        (f"mll_{suffix}_dy2m_nb1_post.png", "Dimuon, N<sub>b</sub> = 1, after R<sub>Z</sub>/R<sub>T</sub>"),
        (f"mll_{suffix}_dy2m_nb2plus.png", "Dimuon, N<sub>b</sub> &ge; 2, matrix input"),
        (f"mll_{suffix}_dy2m_nb2plus_post.png", "Dimuon, N<sub>b</sub> &ge; 2, after R<sub>Z</sub>/R<sub>T</sub>"),
    ]
    card_html = "\n".join(
        f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{label}"></a><figcaption>{label}</figcaption></figure>'
        for name, label in cards
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{campaign_year} RZ(Nb) measurement</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1320px;margin:0 auto;padding:24px;color:#161616}}
h1{{font-size:2rem}} .note{{background:#f2f5f8;border-left:5px solid #4178be;padding:14px 18px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:22px 0}} th,td{{border:1px solid #bbb;padding:10px;text-align:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}} figure{{margin:0}} img{{width:100%;height:auto}} figcaption{{font-size:1rem;padding:6px 0 18px}}
</style></head><body>
<h1>{campaign_year} {"high" if selection == "highdm" else "low"}-&Delta;m R<sub>Z</sub>(N<sub>b</sub>) measurement</h1>
<div class="note">Run-2-style on/off-Z matrix measurement. R<sub>Z</sub> and R<sub>T</sub> are determined independently in ee and &mu;&mu; for N<sub>b</sub>=1 and N<sub>b</sub>&ge;2. The combined R<sub>Z</sub> uses inverse-variance weighting. Both the matrix input and the distribution after applying the fitted channel-specific R<sub>Z</sub>/R<sub>T</sub> factors are provided; the latter is a fit visualization, not a closure test. No U<sub>T</sub>-dependent R<sub>Z</sub> is used.</div>
<table><thead><tr><th>Category</th><th>R<sub>Z</sub>(ee)</th><th>R<sub>Z</sub>(&mu;&mu;)</th><th>Combined R<sub>Z</sub></th><th>R<sub>T</sub>(ee)</th><th>R<sub>T</sub>(&mu;&mu;)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid">{card_html}</div>
</body></html>"""
    (output_dir / "index.html").write_text(document)


def measurement_provenance(
    path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "summary": payload.get("summary") or {},
        "provenance": payload.get("provenance") or {},
    }


def validate_measurement(payload: dict[str, Any], path: Path) -> None:
    if payload.get("status") != "complete":
        raise RuntimeError(f"{path}: incomplete status {payload.get('status')}")
    required = ("rz_high", "mll_high", "rz_low", "mll_low")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise RuntimeError(
            f"{path}: unified measurement is missing {','.join(missing)}"
        )
    summary = payload.get("summary") or {}
    if summary.get("failures"):
        raise RuntimeError(f"{path}: exact-refinement failures are non-empty")
    if int(summary.get("expected_partitions", -1)) != int(
        summary.get("completed_partitions", -2)
    ):
        raise RuntimeError(f"{path}: exact-refinement accounting does not close")
    if int(summary.get("candidate_events", -1)) != int(
        summary.get("matched_events", -2)
    ):
        raise RuntimeError(f"{path}: exact-refinement event matching does not close")


def build_selection_report(
    result: dict[str, Any],
    selection: str,
    output_dir: Path,
    campaign_year: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    rz_key = "rz_high" if selection == "highdm" else "rz_low"
    mll_key = "mll_high" if selection == "highdm" else "mll_low"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "rz": plot_rz_nb(result[rz_key], selection, output_dir),
        "rt": plot_rt(result[rz_key], selection, output_dir),
        "mll_input": plot_mll(
            result[mll_key],
            result[rz_key],
            selection,
            output_dir,
            corrected=False,
        ),
        "mll_post": plot_mll(
            result[mll_key],
            result[rz_key],
            selection,
            output_dir,
            corrected=True,
        ),
    }
    payload = {
        "schema_version": f"dy_estimation_report_{campaign_year}_v1",
        "status": "complete",
        "method": {
            "campaign_year": campaign_year,
            "mass_windows": {
                "on": "81 < mll < 101 GeV",
                "off": "50 < mll < 81 GeV or mll > 101 GeV",
            },
            "categories": ["Nb1", "Nb2plus"],
            "channels": list(CHANNELS),
            "combination": "inverse-variance weighted RZ across ee and mumu",
            "ut_dependent_rz": False,
            "post_mll_scaling": "zll *= channel RZ(Nb); other *= channel RT(Nb)",
        },
        "inputs": inputs,
        rz_key: result[rz_key],
        "plots": plots,
    }
    write_json(output_dir / "summary.json", payload)
    write_html(output_dir, result, selection, campaign_year)
    return payload


def write_combined_index(
    output_dir: Path, campaign_year: str, payloads: dict[str, Any]
) -> None:
    links = []
    for selection, label in (
        ("highdm", "High-&Delta;m"),
        ("lowdm", "Low-&Delta;m"),
    ):
        if selection in payloads:
            links.append(
                f'<li><a href="{selection}/">{label} R<sub>Z</sub>/R<sub>T</sub></a></li>'
            )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{campaign_year} unified DY normalization</title>
<style>body{{font-family:Arial,sans-serif;max-width:960px;margin:0 auto;padding:28px}}
li{{font-size:1.25rem;margin:14px 0}}</style></head><body>
<h1>{campaign_year} unified DY normalization measurement</h1>
<p>High- and Low-&Delta;m are produced from one audited measurement artifact.
The sparse NanoAOD stage is only an exact refinement of topology-ambiguous
Low-&Delta;m events, not an independent normalization workflow.</p>
<ul>{''.join(links)}</ul>
</body></html>"""
    (output_dir / "index.html").write_text(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--ee", type=Path)
    parser.add_argument("--mumu", type=Path)
    parser.add_argument(
        "--selection",
        choices=("highdm", "lowdm", "both"),
        default="both",
    )
    parser.add_argument("--low-exact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-year", choices=("2024", "2025"), default="2024"
    )
    args = parser.parse_args(argv)
    CMS_LABEL["rlabel"] = f"{args.campaign_year} (13.6 TeV)"

    if args.measurement is not None:
        result = read_json(args.measurement)
        validate_measurement(result, args.measurement)
        inputs = {
            "measurement": measurement_provenance(
                args.measurement, result
            )
        }
        selections = (
            ("highdm", "lowdm")
            if args.selection == "both"
            else (args.selection,)
        )
    else:
        if args.ee is None or args.mumu is None:
            raise SystemExit(
                "use --measurement for the unified workflow, or provide both --ee and --mumu for a legacy single-selection report"
            )
        if args.selection == "both":
            raise SystemExit(
                "--selection both requires the unified --measurement artifact"
            )
        ee = read_json(args.ee)
        mumu = read_json(args.mumu)
        validate_channel(ee, "DY2E", args.ee)
        validate_channel(mumu, "DY2M", args.mumu)
        inputs = {
            "ee": feature_provenance(args.ee, ee),
            "mumu": feature_provenance(args.mumu, mumu),
        }
        if args.selection == "highdm":
            result = combine_highdm(ee, mumu)
        else:
            if args.low_exact is None:
                raise SystemExit("--low-exact is required for lowdm")
            low = read_json(args.low_exact)
            if low.get("status") != "complete":
                raise SystemExit(
                    f"{args.low_exact}: exact Low-dM input is incomplete"
                )
            result = {
                "rz_low": finalize_rz(low["rz_low_raw"]),
                "mll_low": low["mll_low"],
            }
            inputs["low_exact"] = {
                "file": args.low_exact.name,
                "sha256": sha256_file(args.low_exact),
                "summary": low.get("summary") or {},
            }
        selections = (args.selection,)

    payloads: dict[str, Any] = {}
    for selection in selections:
        target = (
            args.output_dir / selection
            if len(selections) > 1
            else args.output_dir
        )
        payloads[selection] = build_selection_report(
            result,
            selection,
            target,
            args.campaign_year,
            inputs,
        )
    if len(selections) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        combined = {
            selection: payloads[selection][
                "rz_high" if selection == "highdm" else "rz_low"
            ]["combined"]
            for selection in selections
        }
        write_json(
            args.output_dir / "summary.json",
            {
                "schema_version": f"dy_estimation_combined_report_{args.campaign_year}_v1",
                "status": "complete",
                "inputs": inputs,
                "reports": {
                    selection: f"{selection}/summary.json"
                    for selection in selections
                },
                "combined": combined,
            },
        )
        write_combined_index(
            args.output_dir, args.campaign_year, payloads
        )
    else:
        selection = selections[0]
        combined = payloads[selection][
            "rz_high" if selection == "highdm" else "rz_low"
        ]["combined"]
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "selections": list(selections),
                "combined": combined,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
