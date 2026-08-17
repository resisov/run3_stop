#!/usr/bin/env python3
"""Create the review report for the 2024 photon-template fake measurement."""

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


plt.style.use(hep.style.CMS)
CMS_LEFT = "Work in progress"
CMS_RIGHT = "2024 (13.6 TeV)"
FIGSIZE = (10, 10)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def cms_label(ax: Any) -> None:
    hep.cms.label(llabel=CMS_LEFT, rlabel=CMS_RIGHT, ax=ax)


def save(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def ratio_summary(prediction: np.ndarray, target: np.ndarray, pred_var: np.ndarray, target_var: np.ndarray) -> dict[str, Any]:
    valid = (prediction > 0.0) & (target > 0.0)
    total_prediction = float(np.sum(prediction))
    total_target = float(np.sum(target))
    total_variance = float(np.sum(pred_var) + np.sum(target_var))
    difference = total_prediction - total_target
    chi2 = 0.0
    bins = 0
    for pred, obs, variance in zip(prediction, target, pred_var + target_var):
        if variance > 0.0 and np.isfinite(variance):
            chi2 += float((pred - obs) ** 2 / variance)
            bins += 1
    ratios = prediction[valid] / target[valid]
    return {
        "prediction": total_prediction,
        "target": total_target,
        "prediction_over_target": total_prediction / total_target if total_target > 0.0 else None,
        "integral_pull": difference / math.sqrt(total_variance) if total_variance > 0.0 else None,
        "chi2": chi2,
        "ndf": bins,
        "chi2_over_ndf": chi2 / bins if bins else None,
        "maximum_absolute_log_ratio": float(np.max(np.abs(np.log(ratios)))) if len(ratios) else None,
    }


def plot_fit(record: dict[str, Any], stage: str, output: Path) -> None:
    fit = record[stage]
    edges = np.asarray(record["fake_template"]["edges"], dtype=float)
    observation = np.asarray(fit["observation"], dtype=float)
    prompt = np.asarray(fit["prompt_component"], dtype=float)
    fake = np.asarray(fit["fake_component"], dtype=float)
    electron = np.asarray(fit["electron_component"], dtype=float)
    expectation = prompt + fake + electron
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * (edges[1:] - edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    hep.histplot([fake, electron, prompt], bins=edges, stack=True, histtype="fill", label=["Fake photon", "Electron", "Prompt photon"], ax=ax)
    ax.errorbar(centers, observation, xerr=widths, yerr=np.sqrt(np.maximum(observation, 0.0)), fmt="o", color="black", label="Data")
    ax.set_ylabel("Events")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=14, frameon=False)
    cms_label(ax)
    label = "pass charged isolation" if stage == "pass_fit" else "loose charged isolation"
    group = record["group"]
    pt_label = (
        f"$p_T^\\gamma > {group['pt_low']:g}$ GeV"
        if float(group["pt_high"]) >= 1_000_000.0
        else f"{group['pt_low']:g} < $p_T^\\gamma$ < {group['pt_high']:g} GeV"
    )
    ax.text(0.04, 0.05, f"{group['eta']}, {pt_label}\n{label}", transform=ax.transAxes, fontsize=14, va="bottom")
    ratio = np.divide(observation, expectation, out=np.full_like(observation, np.nan), where=expectation > 0.0)
    ratio_error = np.divide(np.sqrt(np.maximum(observation, 0.0)), expectation, out=np.zeros_like(observation), where=expectation > 0.0)
    ratio_ax.axhline(1.0, color="0.35", linewidth=1.2)
    ratio_ax.errorbar(centers, ratio, xerr=widths, yerr=ratio_error, fmt="o", color="black")
    ratio_ax.set_ylabel("Data/Fit")
    ratio_ax.set_xlabel(r"Photon $\sigma_{i\eta i\eta}$")
    ratio_ax.set_ylim(0.0, 2.0)
    ratio_ax.set_xlim(edges[0], edges[-1])
    save(fig, output)


def plot_factors(measurement: dict[str, Any], output: Path) -> None:
    groups = measurement["groups"]
    mapping = measurement["fine_to_used_group"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    display_edges = np.asarray([220.0, 300.0, 400.0, 600.0, 1000.0])
    centers = 0.5 * (display_edges[:-1] + display_edges[1:])
    xerr = 0.5 * (display_edges[1:] - display_edges[:-1])
    for eta, marker, color in (("EB", "o", "tab:blue"), ("EE", "^", "tab:orange")):
        values = []
        errors = []
        for low, high in zip((220.0, 300.0, 400.0, 600.0), (300.0, 400.0, 600.0, 1_000_000.0)):
            fine = f"{eta}_pt{low:g}to{'inf' if high >= 1_000_000 else f'{high:g}'}"
            factor = groups[mapping[fine]]["fake_factor"]
            values.append(float(factor["value"]) if factor["valid"] else np.nan)
            errors.append(
                float(factor.get("total_uncertainty", factor["uncertainty"]))
                if factor["valid"]
                else 0.0
            )
        values_array = np.asarray(values, dtype=float)
        errors_array = np.asarray(errors, dtype=float)
        valid = np.isfinite(values_array) & np.isfinite(errors_array)
        if np.any(valid):
            ax.errorbar(
                centers[valid],
                values_array[valid],
                xerr=xerr[valid],
                yerr=errors_array[valid],
                fmt=marker,
                markersize=8,
                capsize=3,
                color=color,
                label=eta,
            )
    ax.set_xlim(display_edges[0], display_edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$p_T^\gamma$ (GeV)")
    ax.set_ylabel("Fake factor")
    ax.legend(fontsize=16, frameon=False)
    ax.text(0.04, 0.05, "Low-$\\Delta\\phi$ measurement region", transform=ax.transAxes, fontsize=15, va="bottom")
    cms_label(ax)
    save(fig, output)


def plot_factor_mc_bias(measurement: dict[str, Any], output: Path) -> None:
    groups = measurement["simulation_template_fit_groups"]
    mapping = measurement["fine_to_used_group"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    display_edges = np.asarray([220.0, 300.0, 400.0, 600.0, 1000.0])
    centers = 0.5 * (display_edges[:-1] + display_edges[1:])
    xerr = 0.5 * (display_edges[1:] - display_edges[:-1])
    for eta, marker, color in (("EB", "o", "tab:blue"), ("EE", "^", "tab:orange")):
        values = []
        for low, high in zip((220.0, 300.0, 400.0, 600.0), (300.0, 400.0, 600.0, 1_000_000.0)):
            fine = f"{eta}_pt{low:g}to{'inf' if high >= 1_000_000 else f'{high:g}'}"
            value = finite(groups[mapping[fine]].get("fitted_over_truth_factor"))
            values.append(np.nan if value is None else value)
        values_array = np.asarray(values, dtype=float)
        valid = np.isfinite(values_array)
        if np.any(valid):
            ax.errorbar(
                centers[valid],
                values_array[valid],
                xerr=xerr[valid],
                fmt=marker,
                markersize=8,
                color=color,
                label=eta,
            )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.4)
    ax.set_xlim(display_edges[0], display_edges[-1])
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$p_T^\gamma$ (GeV)")
    ax.set_ylabel("Fitted / truth fake factor")
    ax.legend(fontsize=16, frameon=False)
    ax.text(0.04, 0.05, "MC template-fit bias", transform=ax.transAxes, fontsize=15, va="bottom")
    cms_label(ax)
    save(fig, output)


def plot_ut_comparison(
    edges: np.ndarray,
    target: np.ndarray,
    target_variance: np.ndarray,
    prediction: np.ndarray,
    prediction_variance: np.ndarray,
    target_label: str,
    annotation: str,
    output: Path,
) -> dict[str, Any]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * (edges[1:] - edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    hep.histplot(prediction, bins=edges, yerr=np.sqrt(np.maximum(prediction_variance, 0.0)), histtype="step", linewidth=2.0, label="Loose-photon prediction", ax=ax)
    ax.errorbar(centers, target, xerr=widths, yerr=np.sqrt(np.maximum(target_variance, 0.0)), fmt="o", color="black", label=target_label)
    positive = np.concatenate([target[target > 0.0], prediction[prediction > 0.0]])
    if len(positive) and np.max(positive) / max(np.min(positive), 1.0e-12) > 80.0:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.05, 0.3 * np.min(positive)))
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Fake-photon yield")
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(fontsize=14, frameon=False)
    ax.text(0.04, 0.05, annotation, transform=ax.transAxes, fontsize=15, va="bottom")
    cms_label(ax)
    ratio = np.divide(prediction, target, out=np.full_like(prediction, np.nan), where=target > 0.0)
    ratio_variance = np.divide(prediction_variance, target * target, out=np.zeros_like(prediction), where=target > 0.0)
    ratio_variance += np.divide(prediction * prediction * target_variance, target**4, out=np.zeros_like(prediction), where=target > 0.0)
    ratio_ax.axhline(1.0, color="0.35", linewidth=1.2)
    ratio_ax.errorbar(centers, ratio, xerr=widths, yerr=np.sqrt(np.maximum(ratio_variance, 0.0)), fmt="o", color="black")
    ratio_ax.set_ylim(0.0, 2.0)
    ratio_ax.set_ylabel("Pred./target")
    ratio_ax.set_xlabel(r"$U_T$ (GeV)")
    ratio_ax.set_xlim(edges[0], edges[-1])
    save(fig, output)
    return ratio_summary(prediction, target, prediction_variance, target_variance)


def baseline_gcr_plot(measurement: dict[str, Any], baseline: dict[str, Any], output: Path) -> dict[str, Any]:
    record = baseline["results"]["GCR/ut"]
    old_edges = np.asarray(record["bin_edges"], dtype=float)
    new = measurement["nominal_gcr_validation"]["loose_prediction"]
    new_edges = np.asarray(new["edges"], dtype=float)
    if not np.array_equal(old_edges, new_edges):
        raise RuntimeError(f"baseline/new GCR UT binning differs: {old_edges} vs {new_edges}")
    data = np.asarray(record["data"], dtype=float)
    nominal = np.asarray(record["nominal_prediction"], dtype=float)
    retained = np.asarray(record["retained_prompt_plus_electron_mc"], dtype=float)
    fake = np.asarray(new["prediction"], dtype=float)
    candidate = retained + fake
    centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    widths = 0.5 * (old_edges[1:] - old_edges[:-1])
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05})
    hep.histplot(nominal, bins=old_edges, histtype="step", linewidth=2.0, label="Nominal MC", ax=ax)
    hep.histplot(candidate, bins=old_edges, histtype="step", linewidth=2.0, label="Template fake replacement", ax=ax)
    ax.errorbar(centers, data, xerr=widths, yerr=np.sqrt(np.maximum(data, 0.0)), fmt="o", color="black", label="Data")
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.set_xlim(old_edges[0], old_edges[-1])
    ax.legend(fontsize=14, frameon=False)
    ax.text(0.04, 0.05, "High-$\\Delta m$ GCR", transform=ax.transAxes, fontsize=15, va="bottom")
    cms_label(ax)
    nominal_ratio = np.divide(data, nominal, out=np.full_like(data, np.nan), where=nominal > 0.0)
    candidate_ratio = np.divide(data, candidate, out=np.full_like(data, np.nan), where=candidate > 0.0)
    ratio_ax.axhline(1.0, color="0.35", linewidth=1.2)
    ratio_ax.errorbar(centers, nominal_ratio, xerr=widths, fmt="o", label="Nominal", color="tab:blue")
    ratio_ax.errorbar(centers, candidate_ratio, xerr=widths, fmt="s", label="Replacement", color="tab:orange")
    ratio_ax.set_ylabel("Data/pred.")
    ratio_ax.set_xlabel(r"$U_T$ (GeV)")
    ratio_ax.set_ylim(0.0, 3.0)
    ratio_ax.set_xlim(old_edges[0], old_edges[-1])
    save(fig, output)
    def agreement_metrics(prediction: np.ndarray) -> dict[str, float | None]:
        active = (data > 0.0) & (prediction > 0.0)
        if not np.any(active) or float(np.sum(prediction)) <= 0.0:
            return {
                "integral_data_over_prediction": None,
                "abs_log_integral_ratio": None,
                "log_ratio_rms": None,
                "max_abs_log_ratio": None,
                "poisson_deviance": None,
            }
        log_ratios = np.log(data[active] / prediction[active])
        deviance_terms = prediction[active] - data[active] + data[active] * np.log(
            data[active] / prediction[active]
        )
        integral_ratio = float(np.sum(data) / np.sum(prediction))
        return {
            "integral_data_over_prediction": integral_ratio,
            "abs_log_integral_ratio": abs(math.log(integral_ratio)),
            "log_ratio_rms": float(np.sqrt(np.mean(log_ratios * log_ratios))),
            "max_abs_log_ratio": float(np.max(np.abs(log_ratios))),
            "poisson_deviance": float(2.0 * np.sum(deviance_terms)),
        }

    nominal_metrics = agreement_metrics(nominal)
    replacement_metrics = agreement_metrics(candidate)
    shape_metric_names = ("log_ratio_rms", "max_abs_log_ratio", "poisson_deviance")
    shape_improves = all(
        nominal_metrics[name] is not None
        and replacement_metrics[name] is not None
        and float(replacement_metrics[name]) < float(nominal_metrics[name])
        for name in shape_metric_names
    )
    return {
        "nominal_data_over_mc": float(np.sum(data) / np.sum(nominal)) if np.sum(nominal) > 0.0 else None,
        "replacement_data_over_prediction": float(np.sum(data) / np.sum(candidate)) if np.sum(candidate) > 0.0 else None,
        "nominal_absolute_integral_distance": float(abs(np.sum(data) / np.sum(nominal) - 1.0)) if np.sum(nominal) > 0.0 else None,
        "replacement_absolute_integral_distance": float(abs(np.sum(data) / np.sum(candidate) - 1.0)) if np.sum(candidate) > 0.0 else None,
        "nominal_metrics": nominal_metrics,
        "replacement_metrics": replacement_metrics,
        "replacement_shape_metrics_improve": shape_improves,
    }


def html_report(output: Path, summary: dict[str, Any], figures: list[tuple[str, str]], measurement: dict[str, Any]) -> None:
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    cards = "".join(
        f'<section><h2>{html.escape(label)}</h2><a href="{html.escape(path)}"><img src="{html.escape(path)}" alt="{html.escape(label)}"></a></section>'
        for label, path in figures
    )
    group_rows = []
    for fine, used in sorted(measurement["fine_to_used_group"].items()):
        factor = measurement["groups"][used]["fake_factor"]
        value = factor.get("value")
        uncertainty = factor.get("total_uncertainty", factor.get("uncertainty"))
        rendered = "invalid" if value is None else f"{value:.4g} ± {uncertainty:.3g}"
        group_rows.append(f"<tr><td>{html.escape(fine)}</td><td>{html.escape(used)}</td><td>{rendered}</td></tr>")
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Photon fake template fit</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "img{width:100%;max-width:820px}section{margin:3rem 0}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem;text-align:left}"
        ".warning{background:#fff3cd;padding:1rem;border-left:5px solid #d39e00}</style></head><body>"
        "<h1>2024 fake-photon template-fit measurement</h1>"
        "<p class='warning'>Work in progress. Nominal histograms were not modified. Adoption is gated by MC and independent-data closure.</p>"
        f"<table>{rows}</table><h2>Factor mapping</h2><table><tr><th>Fine bin</th><th>Used measurement</th><th>Fake factor</th></tr>{''.join(group_rows)}</table>"
        f"{cards}</body></html>"
    )


def markdown_report(
    output: Path,
    measurement: dict[str, Any],
    machine: dict[str, Any],
    figures: list[tuple[str, str]],
    production_audits: list[dict[str, Any]],
    normalization_audit: dict[str, Any] | None,
) -> None:
    factor_rows = []
    for fine, used in sorted(measurement["fine_to_used_group"].items()):
        factor = measurement["groups"][used]["fake_factor"]
        if factor["valid"]:
            value = float(factor["value"])
            stat = float(factor.get("statistical_uncertainty", factor["uncertainty"]))
            syst = float(factor.get("systematic_uncertainty", 0.0))
            rendered = f"{value:.4g} ± {stat:.3g} (stat.) ± {syst:.3g} (syst.)"
        else:
            rendered = "invalid"
        factor_rows.append(f"| {fine} | {used} | {rendered} |")
    figure_lines = [f"- [{label}]({path})" for label, path in figures]
    production_rows = [
        "| {campaign} | {status} | {valid}/{expected} | {files_processed}/{files_attempted} | {events_read} | {selected_events} |".format(
            **audit
        )
        for audit in production_audits
    ]
    normalization_text = "No supplemental normalization audit was supplied."
    if normalization_audit is not None:
        normalization_text = (
            f"The rare-background `Runs` audit processed "
            f"{normalization_audit['files_processed']}/{normalization_audit['files_attempted']} files "
            f"across {normalization_audit['datasets']} datasets with status "
            f"`{normalization_audit['status']}`."
        )
    simulation = machine["simulation_closure"]
    validation = machine["data_validation_closure"]
    application = machine["nominal_gcr_validation"]
    baseline = machine.get("baseline_comparison")
    baseline_text = "Baseline comparison was not supplied."
    if baseline is not None:
        baseline_text = (
            f"The integral Data/MC ratio is {baseline['nominal_data_over_mc']:.4g} "
            f"for nominal MC and {baseline['replacement_data_over_prediction']:.4g} "
            "after replacing the nominal fake component. "
            f"All three predeclared shape metrics improve: "
            f"{baseline['replacement_shape_metrics_improve']}."
        )
    output.write_text(
        "# 2024 fake-photon background estimation: template-fit study\n\n"
        "> Work in progress. This study does not mutate nominal histograms.\n\n"
        "## 1. Objective\n\n"
        "The study tests whether the hadronic-fake photon component can be constrained from data "
        "and improve the prefit high-$\\Delta m$ photon control-region agreement, especially the "
        "$U_T$ shape. A method is rejected if closure fails or if the GCR Data/MC agreement is "
        "worse than the nominal prediction.\n\n"
        "## 2. Event and photon definition\n\n"
        "All event selection, jet cleaning, b tagging, recoil, trigger, filter, and region decisions "
        "come from `real_subset_worker.py`. The selected photon keeps the medium photon requirements "
        "except that $\\sigma_{i\\eta i\\eta}$ and charged isolation are opened for the template fit. "
        "Exactly one nominal medium photon takes precedence; without one, exactly one relaxed photon "
        "candidate is required. Nominal intermediate ROOT files are read-only and unchanged.\n\n"
        "### Production validation\n\n"
        "| Campaign | Status | Valid jobs | Source files | Events read | Selected compact events |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + ("\n".join(production_rows) if production_rows else "| not supplied | — | — | — | — | — |")
        + "\n\n"
        + normalization_text
        + "\n\n"
        "## 3. Method\n\n"
        "The factor is measured in `GCR_DPhiVR_Low`. The prompt template is obtained from normalized "
        "prompt-photon simulation. The fake template is obtained from data failing loose charged "
        "isolation after subtracting prompt-photon and electron-origin MC. The electron component is "
        "kept as a separate fixed template. Extended binned likelihood fits to "
        "$\\sigma_{i\\eta i\\eta}$ determine fake yields in the charged-isolation pass and "
        "loose-not-medium samples. Their ratio defines the fake factor. This follows the template-fit "
        "logic used in CMS Run-2 W$\\gamma$/Z$\\gamma$ measurements rather than assuming ABCD "
        "independence.\n\n"
        "References: [CMS Wγ full Run 2](https://arxiv.org/abs/2102.02283), "
        "[CMS electroweak Zγjj](https://arxiv.org/abs/2002.09902).\n\n"
        "## 4. Measured factors\n\n"
        "| Requested bin | Used measurement bin | Fake factor |\n"
        "|---|---|---|\n"
        + "\n".join(factor_rows)
        + "\n\n"
        "Fine bins automatically fall back to a coarser bin only when the predefined data/template "
        "statistics requirements fail. Nominal GCR target data do not choose the binning or factor.\n\n"
        "## 5. Closure tests\n\n"
        f"- MC high-$\\Delta\\phi$ closure: prediction/truth = {simulation['prediction_over_target']}; "
        f"$\\chi^2$/ndf = {simulation['chi2_over_ndf']}.\n"
        f"- Independent data high-$\\Delta\\phi$ closure: prediction/direct fit = {validation['prediction_over_target']}; "
        f"$\\chi^2$/ndf = {validation['chi2_over_ndf']}.\n"
        f"- Nominal GCR validation: prediction/direct fit = {application['prediction_over_target']}; "
        f"$\\chi^2$/ndf = {application['chi2_over_ndf']}.\n\n"
        "The direct data target is itself obtained from a shower-shape template fit; it is not forced "
        "to equal the loose-photon prediction. The $U_T$ points are therefore genuine closure tests.\n\n"
        "The predeclared closure gate requires both integral ratios to be within "
        "$|\\log(\\mathrm{prediction}/\\mathrm{target})|<0.35$, the data-VR integral pull below 2, "
        "and $\\chi^2$/ndf below 2.5 (simulation) and 2.0 (data VR). The GCR replacement must also "
        "improve the integral distance and all three shape metrics: log-ratio RMS, maximum absolute "
        "log ratio, and Poisson deviance.\n\n"
        "## 6. Uncertainties\n\n"
        "Per-factor uncertainties include fit statistics, the low/high charged-isolation fake-template "
        "choice, a ±50% electron normalization variation, a ±30% prompt-contamination variation, "
        "and a stage-specific prompt-template shape variation. The larger absolute integral "
        "nonclosure from simulation and the independent data "
        "VR is propagated as a correlated method uncertainty to the nominal GCR prediction. MC "
        "normalization is applied exactly once as generator/SF weight times the physical-dataset "
        "normalization factor.\n\n"
        "## 7. GCR impact and decision\n\n"
        + baseline_text
        + "\n\n"
        f"**Decision: {machine['decision']}.**\n\n"
        "This decision is deliberately based on closure and prefit Data/MC behavior; no postfit "
        "normalization is used.\n\n"
        "## 8. Figures\n\n"
        + "\n".join(figure_lines)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--production-audits", type=Path, nargs="*", default=[])
    parser.add_argument("--normalization-audit", type=Path)
    args = parser.parse_args()
    measurement = read_json(args.measurement)
    if measurement.get("status") != "complete":
        raise RuntimeError("measurement is not complete")
    output = args.output_dir
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []
    production_audits = []
    for path in args.production_audits:
        audit = read_json(path)
        production_audits.append(
            {
                "campaign": Path(str(audit.get("campaign") or path.parent)).name,
                "status": str(audit.get("status")),
                "valid": int(audit.get("valid_outputs") or 0),
                "expected": int(audit.get("expected_jobs") or 0),
                "files_processed": int(audit.get("files_processed") or 0),
                "files_attempted": int(audit.get("files_attempted") or 0),
                "events_read": int(audit.get("events_read") or 0),
                "selected_events": int(audit.get("selected_events") or 0),
            }
        )
    normalization_audit = None
    if args.normalization_audit is not None:
        audit = read_json(args.normalization_audit)
        normalization_audit = {
            key: audit.get(key)
            for key in (
                "status",
                "datasets",
                "files_attempted",
                "files_processed",
                "wall_time_s",
            )
        }
    plot_factors(measurement, plots / "fake_factor_vs_photon_pt")
    figures.append(("Fake factor versus photon pT", "plots/fake_factor_vs_photon_pt.png"))
    plot_factor_mc_bias(measurement, plots / "mc_template_fit_bias")
    figures.append(("MC template-fit bias", "plots/mc_template_fit_bias.png"))
    used_groups = sorted(set(measurement["fine_to_used_group"].values()))
    for name in used_groups:
        record = measurement["groups"][name]
        for stage, label in (("pass_fit", "pass"), ("loose_fit", "loose")):
            plot_fit(record, stage, plots / f"fit_{name}_{label}")
            figures.append((f"{name}: {label} fit", f"plots/fit_{name}_{label}.png"))

    simulation = measurement["simulation_validation_region_closure"]
    simulation_summary = plot_ut_comparison(
        np.asarray(simulation["edges"], dtype=float),
        np.asarray(simulation["truth"], dtype=float),
        np.asarray(simulation["truth_variance"], dtype=float),
        np.asarray(simulation["prediction"], dtype=float),
        np.asarray(simulation["prediction_variance"], dtype=float),
        "Truth fake MC",
        "High-$\\Delta\\phi$ simulation closure",
        plots / "simulation_closure_ut",
    )
    figures.append(("Simulation closure in UT", "plots/simulation_closure_ut.png"))

    validation = measurement["data_validation_region"]
    validation_direct = validation["direct_template_fit"]
    validation_prediction = validation["loose_prediction"]
    validation_summary = plot_ut_comparison(
        np.asarray(validation_direct["edges"], dtype=float),
        np.asarray(validation_direct["yield"], dtype=float),
        np.asarray(validation_direct["variance"], dtype=float),
        np.asarray(validation_prediction["prediction"], dtype=float),
        np.asarray(validation_prediction["total_variance"], dtype=float),
        "Direct template fit",
        "High-$\\Delta\\phi$ data closure",
        plots / "data_validation_closure_ut",
    )
    figures.append(("Independent data closure in UT", "plots/data_validation_closure_ut.png"))

    application = measurement["nominal_gcr_validation"]
    application_direct = application["direct_template_fit"]
    application_prediction = application["loose_prediction"]
    application_summary = plot_ut_comparison(
        np.asarray(application_direct["edges"], dtype=float),
        np.asarray(application_direct["yield"], dtype=float),
        np.asarray(application_direct["variance"], dtype=float),
        np.asarray(application_prediction["prediction"], dtype=float),
        np.asarray(
            application_prediction.get(
                "total_variance_with_closure",
                application_prediction["total_variance"],
            ),
            dtype=float,
        ),
        "Direct template fit",
        "Nominal GCR validation",
        plots / "gcr_fake_prediction_ut",
    )
    figures.append(("Nominal GCR fake prediction in UT", "plots/gcr_fake_prediction_ut.png"))

    summary: dict[str, Any] = {
        "measurement region": measurement["method"]["measurement_region"],
        "validation region": measurement["method"]["validation_region"],
        "selected compact events": measurement["event_count_after_deduplication"],
        "simulation closure prediction/truth": simulation_summary["prediction_over_target"],
        "simulation closure chi2/ndf": simulation_summary["chi2_over_ndf"],
        "data VR prediction/direct": validation_summary["prediction_over_target"],
        "data VR chi2/ndf": validation_summary["chi2_over_ndf"],
        "GCR prediction/direct": application_summary["prediction_over_target"],
        "GCR chi2/ndf": application_summary["chi2_over_ndf"],
    }
    for index, audit in enumerate(production_audits, start=1):
        summary[f"production audit {index}"] = (
            f"{audit['status']}: {audit['valid']}/{audit['expected']} valid jobs"
        )
    if normalization_audit is not None:
        summary["rare normalization audit"] = (
            f"{normalization_audit['status']}: "
            f"{normalization_audit['files_processed']}/{normalization_audit['files_attempted']} files"
        )
    baseline_summary = None
    if args.baseline_evaluation is not None:
        baseline_summary = baseline_gcr_plot(
            measurement,
            read_json(args.baseline_evaluation),
            plots / "gcr_nominal_vs_fake_replacement_ut",
        )
        summary.update(baseline_summary)
        figures.append(("Nominal versus fake-replacement GCR", "plots/gcr_nominal_vs_fake_replacement_ut.png"))

    usable = all(record["statistically_usable"] for name, record in measurement["groups"].items() if name in used_groups)
    sim_ratio = finite(simulation_summary["prediction_over_target"])
    data_ratio = finite(validation_summary["prediction_over_target"])
    closure_pass = bool(
        usable
        and sim_ratio is not None
        and data_ratio is not None
        and abs(math.log(sim_ratio)) < 0.35
        and abs(math.log(data_ratio)) < 0.35
        and (finite(validation_summary["integral_pull"]) is None or abs(float(validation_summary["integral_pull"])) < 2.0)
        and finite(simulation_summary["chi2_over_ndf"]) is not None
        and float(simulation_summary["chi2_over_ndf"]) < 2.5
        and finite(validation_summary["chi2_over_ndf"]) is not None
        and float(validation_summary["chi2_over_ndf"]) < 2.0
    )
    improves = None
    if baseline_summary is not None:
        improves = bool(
            baseline_summary["replacement_absolute_integral_distance"]
            < baseline_summary["nominal_absolute_integral_distance"]
            and baseline_summary["replacement_shape_metrics_improve"]
        )
    production_complete = bool(production_audits) and all(
        audit["status"] == "complete" and audit["valid"] == audit["expected"]
        for audit in production_audits
    )
    if not production_complete:
        decision = "incomplete production: adoption decision deferred"
    elif closure_pass and improves is not False:
        decision = "candidate passes preliminary closure gates"
    else:
        decision = "do not adopt: closure or Data/MC gate failed"
    summary["decision"] = decision
    machine = {
        "status": "complete",
        "decision": decision,
        "production_complete": production_complete,
        "closure_pass": closure_pass,
        "gcr_integral_improves": improves,
        "predeclared_decision_thresholds": {
            "maximum_absolute_log_integral_ratio": 0.35,
            "maximum_absolute_data_validation_integral_pull": 2.0,
            "maximum_simulation_chi2_over_ndf": 2.5,
            "maximum_data_validation_chi2_over_ndf": 2.0,
            "gcr_require_integral_distance_improvement": True,
            "gcr_required_shape_metric_improvements": [
                "log_ratio_rms",
                "max_abs_log_ratio",
                "poisson_deviance",
            ],
        },
        "simulation_closure": simulation_summary,
        "data_validation_closure": validation_summary,
        "nominal_gcr_validation": application_summary,
        "baseline_comparison": baseline_summary,
        "production_audits": production_audits,
        "normalization_audit": normalization_audit,
        "figures": [path for _, path in figures],
    }
    (output / "report_summary.json").write_text(json.dumps(machine, indent=2, sort_keys=True, allow_nan=False) + "\n")
    html_report(output / "index.html", summary, figures, measurement)
    markdown_report(
        output / "report.md",
        measurement,
        machine,
        figures,
        production_audits,
        normalization_audit,
    )
    print(json.dumps({"status": "complete", "output": str(output), "decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
