#!/usr/bin/env python3
"""Evaluate truth-only charged-isolation transfer closure for photon fakes."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import measure_photon_fake_template_2024 as core


COARSE_UT_EDGES = np.asarray([250.0, 400.0, 650.0, 1500.0])
MIN_EFFECTIVE_EVENTS = 10.0
CMS_LEFT = "Work in progress"
CMS_RIGHT = "2024 (13.6 TeV)"
FIGSIZE = (10, 10)


def weighted_record(table: core.EventTable, mask: np.ndarray) -> dict[str, Any]:
    weights = table.weight[mask]
    sumw = float(np.sum(weights))
    sumw2 = float(np.sum(weights * weights))
    return {
        "entries": int(np.count_nonzero(mask)),
        "sumw": sumw,
        "sumw2": sumw2,
        "effective_events": sumw * sumw / sumw2 if sumw2 > 0.0 else 0.0,
    }


def sideband_mask(table: core.EventTable, sideband: str) -> np.ndarray:
    if sideband == "loose_not_medium":
        return table.charged_level == 1
    if sideband == "fail_medium":
        return (table.charged_level >= 0) & (table.charged_level < 2)
    raise KeyError(sideband)


def factor_record(
    table: core.EventTable,
    mask: np.ndarray,
    sideband: str,
) -> dict[str, Any]:
    common = mask & ~table.is_data & (table.origin == "fake") & (table.shape_level >= 2)
    tight = weighted_record(table, common & (table.charged_level >= 2))
    denominator = weighted_record(table, common & sideband_mask(table, sideband))
    valid = bool(tight["sumw"] >= 0.0 and denominator["sumw"] > 0.0)
    if valid:
        value = float(tight["sumw"] / denominator["sumw"])
        variance = value * value * (
            tight["sumw2"] / (tight["sumw"] * tight["sumw"])
            if tight["sumw"] > 0.0
            else 0.0
        )
        variance += value * value * denominator["sumw2"] / (
            denominator["sumw"] * denominator["sumw"]
        )
    else:
        value = None
        variance = None
    stable = bool(
        valid
        and tight["effective_events"] >= MIN_EFFECTIVE_EVENTS
        and denominator["effective_events"] >= MIN_EFFECTIVE_EVENTS
    )
    return {
        "value": value,
        "variance": variance,
        "uncertainty": math.sqrt(variance) if variance is not None else None,
        "valid": valid,
        "stable": stable,
        "minimum_effective_events": MIN_EFFECTIVE_EVENTS,
        "tight": tight,
        "sideband": sideband,
        "sideband_yield": denominator,
    }


def fine_name(table: core.EventTable, index: int) -> str:
    eta = "EB" if abs(float(table.eta[index])) < 1.4442 else "EE"
    position = int(np.searchsorted(np.asarray(core.FINE_PT_EDGES), table.pt[index], side="right") - 1)
    position = max(0, min(position, len(core.FINE_PT_EDGES) - 2))
    low = core.FINE_PT_EDGES[position]
    high = core.FINE_PT_EDGES[position + 1]
    upper = "inf" if high >= 1_000_000.0 else f"{high:g}"
    return f"{eta}_pt{low:g}to{upper}"


def stable_mapping(factors: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for eta in ("EB", "EE"):
        for low, high in zip(core.FINE_PT_EDGES[:-1], core.FINE_PT_EDGES[1:]):
            upper = "inf" if high >= 1_000_000.0 else f"{high:g}"
            fine = f"{eta}_pt{low:g}to{upper}"
            coarse = f"{eta}_pt220to400" if high <= 400.0 else f"{eta}_pt400toinf"
            candidates = (fine, coarse, f"{eta}_inclusive")
            mapping[fine] = next(
                (name for name in candidates if factors[name]["stable"]),
                None,
            )
    return mapping


def closure(
    table: core.EventTable,
    factors: dict[str, dict[str, Any]],
    mapping: dict[str, str | None],
    edges: np.ndarray,
    sideband: str,
) -> dict[str, Any]:
    bins = len(edges) - 1
    truth = np.zeros(bins)
    truth_variance = np.zeros(bins)
    prediction = np.zeros(bins)
    prediction_stat_variance = np.zeros(bins)
    loose_total = np.zeros(bins)
    loose_used = np.zeros(bins)
    signed_by_factor = {
        name: np.zeros(bins) for name in sorted({name for name in mapping.values() if name})
    }
    selected = (
        table.region_masks[core.VALIDATION_REGION]
        & ~table.is_data
        & (table.origin == "fake")
        & (table.shape_level >= 2)
    )
    sideband_selection = sideband_mask(table, sideband)
    for index in np.flatnonzero(selected):
        bin_index = int(np.searchsorted(edges, table.ut[index], side="right") - 1)
        if bin_index < 0 or bin_index >= bins:
            continue
        weight = float(table.weight[index])
        if table.charged_level[index] >= 2:
            truth[bin_index] += weight
            truth_variance[bin_index] += weight * weight
        elif bool(sideband_selection[index]):
            loose_total[bin_index] += weight
            name = mapping.get(fine_name(table, int(index)))
            factor = factors.get(str(name)) if name else None
            if not factor or not factor["valid"]:
                continue
            value = float(factor["value"])
            predicted_weight = weight * value
            prediction[bin_index] += predicted_weight
            prediction_stat_variance[bin_index] += predicted_weight * predicted_weight
            loose_used[bin_index] += weight
            signed_by_factor[str(name)][bin_index] += weight
    prediction_factor_variance = np.zeros(bins)
    for name, loose in signed_by_factor.items():
        variance = factors[name]["variance"]
        if variance is not None:
            prediction_factor_variance += loose * loose * float(variance)
    prediction_variance = prediction_stat_variance + prediction_factor_variance
    total_truth = float(np.sum(truth))
    total_prediction = float(np.sum(prediction))
    total_loose = float(np.sum(loose_total))
    used_loose = float(np.sum(loose_used))
    active = (truth_variance + prediction_variance) > 0.0
    chi2 = float(
        np.sum(
            np.divide(
                (prediction - truth) ** 2,
                prediction_variance + truth_variance,
                out=np.zeros_like(truth),
                where=active,
            )
        )
    )
    return {
        "edges": edges.tolist(),
        "truth": truth.tolist(),
        "truth_variance": truth_variance.tolist(),
        "prediction": prediction.tolist(),
        "prediction_statistical_variance": prediction_stat_variance.tolist(),
        "prediction_factor_variance": prediction_factor_variance.tolist(),
        "prediction_variance": prediction_variance.tolist(),
        "loose_total": loose_total.tolist(),
        "loose_used": loose_used.tolist(),
        "total_truth": total_truth,
        "total_truth_effective_events": (
            total_truth * total_truth / float(np.sum(truth_variance))
            if float(np.sum(truth_variance)) > 0.0
            else 0.0
        ),
        "total_prediction": total_prediction,
        "prediction_over_truth": total_prediction / total_truth if total_truth > 0.0 else None,
        "loose_weight_coverage": used_loose / total_loose if total_loose != 0.0 else None,
        "sideband": sideband,
        "chi2": chi2,
        "ndf": int(np.count_nonzero(active)),
        "chi2_over_ndf": chi2 / int(np.count_nonzero(active)) if np.any(active) else None,
    }


def plot_closure(records: dict[str, dict[str, Any]], output: Path, annotation: str) -> None:
    first = next(iter(records.values()))
    edges = np.asarray(first["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * (edges[1:] - edges[:-1])
    truth = np.asarray(first["truth"], dtype=float)
    truth_variance = np.asarray(first["truth_variance"], dtype=float)
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    labels = {
        "current_mapping": "Current bin mapping",
        "eta_inclusive": r"$\eta$-inclusive $p_T^\gamma$",
        "global_inclusive": "Global inclusive",
        "strict_stable": r"Strict $N_{\mathrm{eff}}$ mapping",
    }
    ax.errorbar(
        centers,
        truth,
        xerr=widths,
        yerr=np.sqrt(np.maximum(truth_variance, 0.0)),
        fmt="o",
        color="black",
        label="Truth tight-fake MC",
        zorder=10,
    )
    positive = list(truth[truth > 0.0])
    for (name, record), color in zip(records.items(), colors):
        prediction = np.asarray(record["prediction"], dtype=float)
        hep.histplot(
            prediction,
            bins=edges,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=labels[name],
            ax=ax,
        )
        positive.extend(prediction[prediction > 0.0])
        ratio = np.divide(
            prediction,
            truth,
            out=np.full_like(prediction, np.nan),
            where=truth > 0.0,
        )
        ratio_ax.errorbar(
            centers,
            ratio,
            xerr=widths,
            fmt="o-",
            markersize=5,
            color=color,
        )
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(1.0e-3, 0.3 * min(positive)), 4.0 * max(positive))
    ax.set_ylabel("Fake-photon yield")
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(fontsize=13, frameon=False)
    ax.text(0.04, 0.05, annotation, transform=ax.transAxes, fontsize=14, va="bottom")
    hep.cms.label(llabel=CMS_LEFT, rlabel=CMS_RIGHT, ax=ax)
    ratios = []
    for record in records.values():
        values = np.divide(
            np.asarray(record["prediction"], dtype=float),
            truth,
            out=np.full_like(truth, np.nan),
            where=truth > 0.0,
        )
        ratios.extend(values[np.isfinite(values) & (values > 0.0)])
    ratio_ax.axhline(1.0, color="0.35", linewidth=1.2)
    if ratios and max(ratios) / min(ratios) > 20.0:
        ratio_ax.set_yscale("log")
        ratio_ax.set_ylim(max(1.0e-3, 0.3 * min(ratios)), 4.0 * max(ratios))
    else:
        ratio_ax.set_ylim(0.0, 2.0)
    ratio_ax.set_ylabel("Pred./truth")
    ratio_ax.set_xlabel(r"$U_T$ (GeV)")
    ratio_ax.set_xlim(edges[0], edges[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_factors(
    factors: dict[str, dict[str, Any]],
    mapping: dict[str, str],
    output: Path,
    sideband: str,
) -> None:
    display_edges = np.asarray([220.0, 300.0, 400.0, 600.0, 1000.0])
    centers = 0.5 * (display_edges[:-1] + display_edges[1:])
    widths = 0.5 * (display_edges[1:] - display_edges[:-1])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for eta, marker, color in (("EB", "o", "tab:blue"), ("EE", "^", "tab:orange")):
        values = []
        errors = []
        for low, high in zip(core.FINE_PT_EDGES[:-1], core.FINE_PT_EDGES[1:]):
            upper = "inf" if high >= 1_000_000.0 else f"{high:g}"
            used = mapping[f"{eta}_pt{low:g}to{upper}"]
            factor = factors[used]
            values.append(factor["value"] if factor["valid"] else np.nan)
            errors.append(factor["uncertainty"] if factor["valid"] else 0.0)
        values_array = np.asarray(values, dtype=float)
        errors_array = np.asarray(errors, dtype=float)
        valid = np.isfinite(values_array) & (values_array > 0.0)
        ax.errorbar(
            centers[valid],
            values_array[valid],
            xerr=widths[valid],
            yerr=errors_array[valid],
            fmt=marker,
            markersize=8,
            capsize=3,
            color=color,
            label=eta,
        )
    ax.set_yscale("log")
    ax.set_xlim(display_edges[0], display_edges[-1])
    ax.set_xlabel(r"$p_T^\gamma$ (GeV)")
    denominator_label = (
        "loose-not-medium" if sideband == "loose_not_medium" else "fail-medium"
    )
    ax.set_ylabel(f"Truth tight/{denominator_label} fake factor")
    ax.legend(fontsize=16, frameon=False)
    ax.text(
        0.04,
        0.05,
        r"Low-$\Delta\phi$ measurement region" "\nCurrent bin mapping",
        transform=ax.transAxes,
        fontsize=14,
        va="bottom",
    )
    hep.cms.label(llabel=CMS_LEFT, rlabel=CMS_RIGHT, ax=ax)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_report(output: Path, payload: dict[str, Any]) -> None:
    rows = []
    for name, record in payload["nominal_ut_closure"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{record['total_prediction']:.6g}</td>"
            f"<td>{record['total_truth']:.6g}</td>"
            f"<td>{record['prediction_over_truth']}</td>"
            f"<td>{record['loose_weight_coverage']}</td>"
            "</tr>"
        )
    factor_rows = []
    for name, factor in sorted(payload["truth_factors"].items()):
        factor_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{factor['value']}</td>"
            f"<td>{factor['tight']['effective_events']:.3g}</td>"
            f"<td>{factor['sideband_yield']['effective_events']:.3g}</td>"
            f"<td>{factor['stable']}</td>"
            "</tr>"
        )
    style = """
body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5;color:#202124}
img{width:min(100%,900px);display:block;margin:1rem auto;border:1px solid #ddd}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccc;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}
.warning{background:#fff3cd;border-left:5px solid #d39e00;padding:1rem}
"""
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Photon fake truth-transfer closure</title><style>{style}</style></head><body>
<h1>Photon fake truth-transfer closure</h1>
<p class="warning">Diagnostic only. Template fits are not used. Nominal intermediates are unchanged.</p>
<p>The fake factor is measured from truth-fake MC in <code>{core.MEASUREMENT_REGION}</code> and applied to truth-fake <code>{payload['method']['sideband']}</code> photons in <code>{core.VALIDATION_REGION}</code>.</p>
<h2>Integral closure</h2><table><thead><tr><th>Mapping</th><th>Prediction</th><th>Truth</th><th>Pred./truth</th><th>Loose coverage</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Figures</h2>
<img src="plots/truth_transfer_closure_ut.png" alt="Nominal UT closure">
<img src="plots/truth_transfer_closure_ut_coarse.png" alt="Coarse UT closure">
<img src="plots/truth_factor_vs_photon_pt.png" alt="Truth fake factors">
<h2>Factor statistics</h2><table><thead><tr><th>Group</th><th>Factor</th><th>Tight Neff</th><th>Sideband Neff</th><th>Stable</th></tr></thead><tbody>{''.join(factor_rows)}</tbody></table>
</body></html>"""
    (output / "index.html").write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sideband",
        choices=("loose_not_medium", "fail_medium"),
        default="loose_not_medium",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    events, input_audit = core.load_inputs(sorted(set(paths)))
    events, deduplication = core.deduplicate_data(events)
    normalization = core.read_payload(args.normalization)
    table = core.EventTable(events, normalization)
    measurement = core.read_payload(args.measurement)
    current_mapping = {
        str(key): str(value) for key, value in measurement["fine_to_used_group"].items()
    }

    truth_factors = {
        group.name: factor_record(
            table,
            core.base_mask(table, core.MEASUREMENT_REGION, group),
            args.sideband,
        )
        for group in core.groups()
    }
    all_mask = table.region_masks[core.MEASUREMENT_REGION] & (table.pt >= 220.0)
    truth_factors["ALL_inclusive"] = factor_record(table, all_mask, args.sideband)
    eta_mapping = {
        fine: ("EB_inclusive" if fine.startswith("EB_") else "EE_inclusive")
        for fine in current_mapping
    }
    global_mapping = {fine: "ALL_inclusive" for fine in current_mapping}
    strict_mapping = stable_mapping(truth_factors)
    mappings: dict[str, dict[str, str | None]] = {
        "current_mapping": current_mapping,
        "eta_inclusive": eta_mapping,
        "global_inclusive": global_mapping,
        "strict_stable": strict_mapping,
    }
    nominal = {
        name: closure(table, truth_factors, mapping, core.UT_EDGES, args.sideband)
        for name, mapping in mappings.items()
    }
    coarse = {
        name: closure(table, truth_factors, mapping, COARSE_UT_EDGES, args.sideband)
        for name, mapping in mappings.items()
    }
    payload = {
        "schema_version": "photon_fake_truth_transfer_closure_2024_v1",
        "status": "complete",
        "method": {
            "measurement_region": core.MEASUREMENT_REGION,
            "validation_region": core.VALIDATION_REGION,
            "factor": f"truth-fake tight divided by truth-fake {args.sideband}",
            "sideband": args.sideband,
            "template_fit_used": False,
            "selection_source": "real_subset_worker.py via photon_fake_template_2024_worker.py",
            "minimum_effective_events_for_strict_mapping": MIN_EFFECTIVE_EVENTS,
            "nominal_intermediate_mutation": False,
        },
        "input_audit": input_audit,
        "deduplication": deduplication,
        "event_count_after_deduplication": table.n,
        "truth_factors": truth_factors,
        "mappings": mappings,
        "nominal_ut_closure": nominal,
        "coarse_ut_closure": coarse,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    core.write_payload(output / "evaluation.json", payload)
    plt.style.use(hep.style.CMS)
    plot_closure(
        nominal,
        output / "plots/truth_transfer_closure_ut",
        r"High-$\Delta\phi$ truth-transfer closure" "\n" r"factor from low-$\Delta\phi$",
    )
    plot_closure(
        coarse,
        output / "plots/truth_transfer_closure_ut_coarse",
        r"High-$\Delta\phi$ truth-transfer closure" "\nCoarse $U_T$ bins",
    )
    plot_factors(
        truth_factors,
        current_mapping,
        output / "plots/truth_factor_vs_photon_pt",
        args.sideband,
    )
    render_report(output, payload)
    print(
        json.dumps(
            {
                "event_count": table.n,
                "output": str(output),
                "ratios": {
                    name: record["prediction_over_truth"]
                    for name, record in nominal.items()
                },
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
