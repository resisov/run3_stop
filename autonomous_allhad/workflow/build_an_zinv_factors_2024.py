#!/usr/bin/env python3
"""Build and plot the AN-style R_Z, Q, and S_gamma measurements for 2024."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")

CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
    "fontsize": 24,
}
HIGH_GROUPS = ("Nb1", "Nb2", "Nb3plus")
HIGH_RZ_GROUP = {"Nb1": "Nb1", "Nb2": "Nb2plus", "Nb3plus": "Nb2plus"}
LOW_GROUPS = ("Nb1", "Nb2plus")
LOW_SHARED_GROUPS = (
    (
        "Nb1",
        "PISR300to500",
        r"$N_b=1,\ 300\leq p_T^{ISR}<500$",
        "#D62728",
        "o",
    ),
    (
        "Nb1",
        "PISR500plus",
        r"$N_b=1,\ p_T^{ISR}\geq500$",
        "#FF8C00",
        "s",
    ),
    (
        "Nb2plus",
        "PISR300to500",
        r"$N_b\geq2,\ 300\leq p_T^{ISR}<500$",
        "#1F77B4",
        "^",
    ),
    (
        "Nb2plus",
        "PISR500plus",
        r"$N_b\geq2,\ p_T^{ISR}\geq500$",
        "#7B2CBF",
        "D",
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def leaf_array(
    payload: dict[str, Any], nbin: int
) -> tuple[np.ndarray, np.ndarray]:
    nominal = (payload or {}).get("nominal") or {}
    values = np.asarray(nominal.get("sumw") or [0.0] * nbin, dtype=float)
    variances = np.asarray(
        nominal.get("sumw2") or [0.0] * nbin, dtype=float
    )
    if len(values) != nbin or len(variances) != nbin:
        raise ValueError(
            f"expected {nbin} bins, got {len(values)}/{len(variances)}"
        )
    return values, variances


def sum_samples(
    by_sample: dict[str, Any],
    nbin: int,
    include: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(nbin, dtype=float)
    variances = np.zeros(nbin, dtype=float)
    for sample, payload in by_sample.items():
        if include is not None and sample not in include:
            continue
        current, current_variance = leaf_array(payload, nbin)
        values += current
        variances += current_variance
    return values, variances


def data_leaf(payload: dict[str, Any]) -> tuple[float, float]:
    return (
        float((payload or {}).get("sumw", 0.0)),
        float((payload or {}).get("sumw2", 0.0)),
    )


def factor(
    numerator: float,
    numerator_variance: float,
    denominator: float,
    denominator_variance: float,
) -> dict[str, Any]:
    if denominator <= 0.0 or numerator < 0.0:
        return {
            "status": "unavailable",
            "value": None,
            "stat": None,
            "numerator": numerator,
            "denominator": denominator,
        }
    value = numerator / denominator
    variance = (
        numerator_variance / denominator**2
        + numerator**2 * denominator_variance / denominator**4
    )
    return {
        "status": "complete",
        "value": float(value),
        "stat": float(math.sqrt(max(variance, 0.0))),
        "numerator": float(numerator),
        "numerator_variance": float(numerator_variance),
        "denominator": float(denominator),
        "denominator_variance": float(denominator_variance),
    }


def low_family(label: str) -> str:
    return re.sub(r"_recoil_[0-9]+$", "", label)


def low_ut_tick_labels(family: str, nbin: int) -> list[str]:
    if "PISR300to500" in family:
        thresholds = [300, 400, 500, 600] if nbin == 4 else [300, 400, 500]
    elif "PISR500plus" in family:
        thresholds = [450, 550, 650, 750] if nbin == 4 else [450, 550, 650]
    else:
        raise ValueError(f"cannot infer Low-dM U_T bins for {family}")
    if len(thresholds) != nbin:
        raise ValueError(
            f"Low-dM U_T bin mismatch for {family}: {nbin} bins"
        )
    return [
        (
            f"{thresholds[index]}–{thresholds[index + 1]}"
            if index + 1 < nbin
            else rf"$\geq {thresholds[index]}$"
        )
        for index in range(nbin)
    ]


def low_ut_geometry(isr_group: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    if isr_group == "PISR300to500":
        lower = 300.0
    elif isr_group == "PISR500plus":
        lower = 450.0
    else:
        raise ValueError(f"unknown Low-dM ISR group: {isr_group}")
    edges = lower + 100.0 * np.arange(nbin + 1, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)


def aggregate_low_sgamma(
    factors: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for group, isr_group, label, color, marker in LOW_SHARED_GROUPS:
        selected = [
            payload
            for family, payload in factors.items()
            if payload["group"] == group and isr_group in family
        ]
        if not selected:
            raise ValueError(f"no Low-dM Sgamma inputs for {group}/{isr_group}")
        nbin = len(selected[0]["bins"])
        if any(len(payload["bins"]) != nbin for payload in selected):
            raise ValueError(
                f"inconsistent Low-dM Sgamma bins for {group}/{isr_group}"
            )
        q = selected[0]["Q"]
        records = []
        for offset in range(nbin):
            data = sum(payload["bins"][offset]["data"] for payload in selected)
            data2 = sum(
                payload["bins"][offset]["data_variance"]
                for payload in selected
            )
            gamma = sum(
                payload["bins"][offset]["gamma_mc"] for payload in selected
            )
            gamma2 = sum(
                payload["bins"][offset]["gamma_mc_variance"]
                for payload in selected
            )
            other = sum(
                payload["bins"][offset]["other_mc"] for payload in selected
            )
            other2 = sum(
                payload["bins"][offset]["other_mc_variance"]
                for payload in selected
            )
            denominator = (
                float(q["value"]) * gamma
                if q["status"] == "complete"
                else 0.0
            )
            denominator_variance = (
                gamma**2 * float(q["stat"]) ** 2
                + float(q["value"]) ** 2 * gamma2
                if q["status"] == "complete"
                else 0.0
            )
            records.append(
                factor(
                    data - other,
                    data2 + other2,
                    denominator,
                    denominator_variance,
                )
            )
        output[f"{group}_{isr_group}"] = {
            "group": group,
            "isr_group": isr_group,
            "label": label,
            "color": color,
            "marker": marker,
            "source_family_count": len(selected),
            "bins": records,
        }
    return output


def normalized_double_ratio(
    z: np.ndarray,
    z2: np.ndarray,
    gamma: np.ndarray,
    gamma2: np.ndarray,
) -> tuple[list[float | None], list[float | None]]:
    z_total = float(np.sum(z))
    gamma_total = float(np.sum(gamma))
    z2_total = float(np.sum(z2))
    gamma2_total = float(np.sum(gamma2))
    values: list[float | None] = []
    errors: list[float | None] = []
    for index in range(len(z)):
        if z_total <= 0.0 or gamma_total <= 0.0 or gamma[index] <= 0.0:
            values.append(None)
            errors.append(None)
            continue
        z_fraction = float(z[index] / z_total)
        gamma_fraction = float(gamma[index] / gamma_total)
        value = z_fraction / gamma_fraction
        z_fraction_variance = (
            (z_total - z[index]) ** 2 * z2[index]
            + z[index] ** 2 * max(z2_total - z2[index], 0.0)
        ) / z_total**4
        gamma_fraction_variance = (
            (gamma_total - gamma[index]) ** 2 * gamma2[index]
            + gamma[index] ** 2
            * max(gamma2_total - gamma2[index], 0.0)
        ) / gamma_total**4
        variance = (
            z_fraction_variance / gamma_fraction**2
            + z_fraction**2
            * gamma_fraction_variance
            / gamma_fraction**4
        )
        values.append(float(value))
        errors.append(float(math.sqrt(max(variance, 0.0))))
    return values, errors


def aggregate_low_double_ratios(
    ratios: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for group, isr_group, label, color, marker in LOW_SHARED_GROUPS:
        selected = [
            payload
            for family, payload in ratios.items()
            if payload["group"] == group and isr_group in family
        ]
        if not selected:
            raise ValueError(
                f"no Low-dM double-ratio inputs for {group}/{isr_group}"
            )
        nbin = len(selected[0]["double_ratio"])
        if any(len(payload["double_ratio"]) != nbin for payload in selected):
            raise ValueError(
                f"inconsistent Low-dM double-ratio bins for "
                f"{group}/{isr_group}"
            )
        z = np.sum(
            [np.asarray(payload["Z_shape"], dtype=float) for payload in selected],
            axis=0,
        )
        z2 = np.sum(
            [
                np.asarray(payload["Z_shape_sumw2"], dtype=float)
                for payload in selected
            ],
            axis=0,
        )
        gamma = np.sum(
            [
                np.asarray(payload["gamma_shape"], dtype=float)
                for payload in selected
            ],
            axis=0,
        )
        gamma2 = np.sum(
            [
                np.asarray(payload["gamma_shape_sumw2"], dtype=float)
                for payload in selected
            ],
            axis=0,
        )
        values, errors = normalized_double_ratio(z, z2, gamma, gamma2)
        output[f"{group}_{isr_group}"] = {
            "group": group,
            "isr_group": isr_group,
            "label": label,
            "color": color,
            "marker": marker,
            "source_family_count": len(selected),
            "double_ratio": values,
            "double_ratio_stat": errors,
        }
    return output


def build_q_sgamma(
    measurement: dict[str, Any], exact: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "highdm": {},
        "lowdm": {},
        "lowdm_Q_groups": {},
    }
    high_nbin = len(exact["highdm"]["recoil_edges"]) - 1
    high_data = measurement["gcr_data"]["highdm"]["yields"]
    for group in HIGH_GROUPS:
        by_sample = exact["highdm"]["recoil"]["GCR"][group]
        total_mc, total_mc2 = sum_samples(by_sample, high_nbin)
        gamma, gamma2 = sum_samples(by_sample, high_nbin, {"GJ"})
        other = total_mc - gamma
        other2 = np.maximum(total_mc2 - gamma2, 0.0)
        data = np.asarray(
            [
                data_leaf((high_data.get(group) or {}).get(str(index), {}))[0]
                for index in range(high_nbin)
            ],
            dtype=float,
        )
        data2 = np.asarray(
            [
                data_leaf((high_data.get(group) or {}).get(str(index), {}))[1]
                for index in range(high_nbin)
            ],
            dtype=float,
        )
        q = factor(
            float(np.sum(data - other)),
            float(np.sum(data2 + other2)),
            float(np.sum(gamma)),
            float(np.sum(gamma2)),
        )
        bins = []
        for index in range(high_nbin):
            denominator = (
                float(q["value"]) * float(gamma[index])
                if q["status"] == "complete"
                else 0.0
            )
            denominator_variance = 0.0
            if q["status"] == "complete":
                denominator_variance = (
                    gamma[index] ** 2 * q["stat"] ** 2
                    + q["value"] ** 2 * gamma2[index]
                )
            bins.append(
                {
                    "index": index,
                    "data": float(data[index]),
                    "data_variance": float(data2[index]),
                    "gamma_mc": float(gamma[index]),
                    "gamma_mc_variance": float(gamma2[index]),
                    "other_mc": float(other[index]),
                    "other_mc_variance": float(other2[index]),
                    "Sgamma": factor(
                        float(data[index] - other[index]),
                        float(data2[index] + other2[index]),
                        denominator,
                        denominator_variance,
                    ),
                }
            )
        output["highdm"][group] = {"Q": q, "bins": bins}

    labels = exact["lowdm"]["search_bin_labels"]
    low_nbin = len(labels)
    low_data = measurement["gcr_data"]["lowdm"]["yields"]
    low_q_by_group: dict[str, dict[str, Any]] = {}
    for group, group_indices in {
        "Nb1": list(range(0, 16)),
        "Nb2plus": list(range(16, low_nbin)),
    }.items():
        by_sample = exact["lowdm"]["search_components"]["GCR"][group]
        total_mc, total_mc2 = sum_samples(by_sample, low_nbin)
        gamma, gamma2 = sum_samples(by_sample, low_nbin, {"GJ"})
        other = total_mc - gamma
        other2 = np.maximum(total_mc2 - gamma2, 0.0)
        data = np.asarray(
            [
                data_leaf(low_data.get(str(index), {}))[0]
                for index in group_indices
            ],
            dtype=float,
        )
        data2 = np.asarray(
            [
                data_leaf(low_data.get(str(index), {}))[1]
                for index in group_indices
            ],
            dtype=float,
        )
        low_q_by_group[group] = factor(
            float(np.sum(data - other[group_indices])),
            float(np.sum(data2 + other2[group_indices])),
            float(np.sum(gamma[group_indices])),
            float(np.sum(gamma2[group_indices])),
        )
    output["lowdm_Q_groups"] = low_q_by_group
    families: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        families.setdefault(low_family(label), []).append(index)
    for family, indices in families.items():
        group = "Nb1" if indices[0] < 16 else "Nb2plus"
        by_sample = exact["lowdm"]["search_components"]["GCR"][group]
        total_mc_all, total_mc2_all = sum_samples(by_sample, low_nbin)
        gamma_all, gamma2_all = sum_samples(by_sample, low_nbin, {"GJ"})
        other_all = total_mc_all - gamma_all
        other2_all = np.maximum(total_mc2_all - gamma2_all, 0.0)
        data = np.asarray(
            [data_leaf(low_data.get(str(index), {}))[0] for index in indices],
            dtype=float,
        )
        data2 = np.asarray(
            [data_leaf(low_data.get(str(index), {}))[1] for index in indices],
            dtype=float,
        )
        gamma = gamma_all[indices]
        gamma2 = gamma2_all[indices]
        other = other_all[indices]
        other2 = other2_all[indices]
        q = low_q_by_group[group]
        bins = []
        for offset, index in enumerate(indices):
            denominator = (
                float(q["value"]) * float(gamma[offset])
                if q["status"] == "complete"
                else 0.0
            )
            denominator_variance = 0.0
            if q["status"] == "complete":
                denominator_variance = (
                    gamma[offset] ** 2 * q["stat"] ** 2
                    + q["value"] ** 2 * gamma2[offset]
                )
            bins.append(
                {
                    "index": index,
                    "label": labels[index],
                    "data": float(data[offset]),
                    "data_variance": float(data2[offset]),
                    "gamma_mc": float(gamma[offset]),
                    "gamma_mc_variance": float(gamma2[offset]),
                    "other_mc": float(other[offset]),
                    "other_mc_variance": float(other2[offset]),
                    "Sgamma": factor(
                        float(data[offset] - other[offset]),
                        float(data2[offset] + other2[offset]),
                        denominator,
                        denominator_variance,
                    ),
                }
            )
        output["lowdm"][family] = {
            "group": group,
            "indices": indices,
            "Q": q,
            "bins": bins,
        }
    return output


def build_double_ratios(exact: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"highdm": {}, "lowdm": {}}
    high_nbin = len(exact["highdm"]["recoil_edges"]) - 1
    for group in HIGH_GROUPS:
        gcr, gcr2 = sum_samples(
            exact["highdm"]["recoil"]["GCR"][group],
            high_nbin,
            {"GJ"},
        )
        sr_source = exact["highdm"]["sr_components"][group]
        sr, sr2 = sum_samples(sr_source, 60, {"Zto2Nu"})
        z = np.asarray(
            [np.sum(sr[index::high_nbin]) for index in range(high_nbin)]
        )
        z2 = np.asarray(
            [np.sum(sr2[index::high_nbin]) for index in range(high_nbin)]
        )
        ratio, ratio_stat = normalized_double_ratio(z, z2, gcr, gcr2)
        output["highdm"][group] = {
            "Z_shape": z.tolist(),
            "Z_shape_sumw2": z2.tolist(),
            "gamma_shape": gcr.tolist(),
            "gamma_shape_sumw2": gcr2.tolist(),
            "double_ratio": ratio,
            "double_ratio_stat": ratio_stat,
        }

    labels = exact["lowdm"]["search_bin_labels"]
    families: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        families.setdefault(low_family(label), []).append(index)
    for family, indices in families.items():
        group = "Nb1" if indices[0] < 16 else "Nb2plus"
        gcr, gcr2 = sum_samples(
            exact["lowdm"]["search_components"]["GCR"][group],
            len(labels),
            {"GJ"},
        )
        sr, sr2 = sum_samples(
            exact["lowdm"]["search_components"]["SR"][group],
            len(labels),
            {"Zto2Nu"},
        )
        z = sr[indices]
        z2 = sr2[indices]
        g = gcr[indices]
        g2 = gcr2[indices]
        ratio, ratio_stat = normalized_double_ratio(z, z2, g, g2)
        output["lowdm"][family] = {
            "group": group,
            "indices": indices,
            "Z_shape": z.tolist(),
            "Z_shape_sumw2": z2.tolist(),
            "gamma_shape": g.tolist(),
            "gamma_shape_sumw2": g2.tolist(),
            "double_ratio": ratio,
            "double_ratio_stat": ratio_stat,
        }
    return output


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_rz(
    rz: dict[str, Any], regime: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    groups = ["Nb1", "Nb2plus"]
    x = np.arange(len(groups), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4", "combined": "#111111"}
    offsets = {"DY2E": -0.16, "DY2M": 0.0, "combined": 0.16}
    for channel in ("DY2E", "DY2M"):
        values, errors, positions = [], [], []
        for index, group in enumerate(groups):
            record = ((rz.get("channels") or {}).get(channel) or {}).get(
                group, {}
            )
            if record.get("status") != "complete":
                continue
            positions.append(x[index] + offsets[channel])
            values.append(record["RZ"])
            errors.append(record["RZ_stat"])
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="o",
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=channel.replace("DY2", ""),
        )
    values, errors, positions = [], [], []
    for index, group in enumerate(groups):
        record = (rz.get("combined") or {}).get(group, {})
        if record.get("status") != "complete":
            continue
        positions.append(x[index] + offsets["combined"])
        values.append(record["RZ"])
        errors.append(record["RZ_stat"])
    ax.errorbar(
        positions,
        values,
        yerr=errors,
        fmt="s",
        color=colors["combined"],
        lw=2.2,
        capsize=4,
        label="combined",
    )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_Z$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rz_{regime}")


def plot_q(factors: dict[str, Any], output_dir: Path) -> list[str]:
    categories = [
        ("highdm", "Nb1", r"High $N_b=1$"),
        ("highdm", "Nb2", r"High $N_b=2$"),
        ("highdm", "Nb3plus", r"High $N_b\geq3$"),
        ("lowdm_Q_groups", "Nb1", r"Low $N_b=1$"),
        ("lowdm_Q_groups", "Nb2plus", r"Low $N_b\geq2$"),
    ]
    values, errors, labels = [], [], []
    for section, group, label in categories:
        record = (
            factors[section][group]["Q"]
            if section == "highdm"
            else factors[section][group]
        )
        labels.append(label)
        if record.get("status") == "complete":
            values.append(float(record["value"]))
            errors.append(float(record["stat"]))
        else:
            values.append(float("nan"))
            errors.append(float("nan"))
    x = np.arange(len(labels), dtype=float)
    colors = ["#D62728"] * 3 + ["#7B2CBF"] * 2
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    for index in range(len(labels)):
        ax.errorbar(
            x[index],
            values[index],
            yerr=errors[index],
            fmt="o",
            color=colors[index],
            lw=2.4,
            ms=6,
            capsize=4,
        )
    ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
    ax.axvline(2.5, color="0.75", lw=1.2)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=24)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$Q=(N_{\mathrm{data}}-N_{\mathrm{other}})"
                  r"/N_{\gamma,\mathrm{MC}}$", fontsize=28)
    ax.tick_params(axis="y", labelsize=24)
    ax.grid(axis="y", alpha=0.16)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / "photon_q_normalization")


def plot_sgamma(
    factors: dict[str, Any], regime: str, output_dir: Path
) -> list[str]:
    paths: list[str] = []
    if regime == "highdm":
        edges = np.asarray([250, 300, 350, 400, 500, 800, 1500], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = 0.5 * (edges[1:] - edges[:-1])
        fig, ax = plt.subplots(figsize=(10.2, 10.2))
        styles = {
            "Nb1": ("o", r"$N_b=1$"),
            "Nb2": ("s", r"$N_b=2$"),
            "Nb3plus": ("^", r"$N_b\geq3$"),
        }
        for group in HIGH_GROUPS:
            records = factors[group]["bins"]
            values = np.asarray(
                [
                    item["Sgamma"]["value"]
                    if item["Sgamma"]["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            errors = np.asarray(
                [
                    item["Sgamma"]["stat"]
                    if item["Sgamma"]["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            marker, label = styles[group]
            ax.errorbar(
                centers,
                values,
                xerr=widths,
                yerr=errors,
                fmt=marker,
                ls="none",
                lw=2.6,
                ms=4.5,
                capsize=3,
                label=label,
            )
        ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
        ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_xmargin(0)
        ax.set_ylabel(r"$S_{\gamma,i}$", fontsize=30)
        ax.tick_params(labelsize=24)
        ax.legend(frameon=False, fontsize=12)
        ax.grid(alpha=0.16)
        hep.cms.label(**CMS_LABEL, ax=ax)
        paths.extend(save_figure(fig, output_dir / "sgamma_highdm"))
    else:
        shared = aggregate_low_sgamma(factors)
        fig, ax = plt.subplots(figsize=(10.2, 10.2))
        for payload in shared.values():
            records = payload["bins"]
            centers, widths = low_ut_geometry(
                payload["isr_group"], len(records)
            )
            values = np.asarray(
                [
                    item["value"]
                    if item["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            errors = np.asarray(
                [
                    item["stat"]
                    if item["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            ax.errorbar(
                centers,
                values,
                xerr=widths,
                yerr=errors,
                fmt=payload["marker"],
                ls="none",
                lw=2.6,
                ms=6,
                capsize=3,
                color=payload["color"],
                label=payload["label"],
            )
        ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
        ax.set_xlim(300.0, 850.0)
        ax.set_xmargin(0)
        ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
        ax.set_ylabel(r"$S_{\gamma,i}$", fontsize=30)
        ax.tick_params(labelsize=24)
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=16, ncol=1)
        hep.cms.label(**CMS_LABEL, ax=ax)
        paths.extend(
            save_figure(fig, output_dir / "sgamma_lowdm_nb_isr_shared")
        )
    return paths


def plot_double_ratios(
    ratios: dict[str, Any], regime: str, output_dir: Path
) -> list[str]:
    paths: list[str] = []
    if regime == "highdm":
        edges = np.asarray([250, 300, 350, 400, 500, 800, 1500], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = 0.5 * (edges[1:] - edges[:-1])
        styles = {
            "Nb1": ("o", r"$N_b=1$"),
            "Nb2": ("s", r"$N_b=2$"),
            "Nb3plus": ("^", r"$N_b\geq3$"),
        }
        fig, ax = plt.subplots(figsize=(10.2, 10.2))
        for group in HIGH_GROUPS:
            values = np.asarray(
                [
                    np.nan if value is None else value
                    for value in ratios[group]["double_ratio"]
                ],
                dtype=float,
            )
            errors = np.asarray(
                [
                    np.nan if value is None else value
                    for value in ratios[group]["double_ratio_stat"]
                ],
                dtype=float,
            )
            marker, label = styles[group]
            ax.errorbar(
                centers,
                values,
                xerr=widths,
                yerr=errors,
                fmt=marker,
                ls="none",
                lw=2.6,
                ms=4.5,
                capsize=3,
                label=label,
            )
        ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
        ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_xmargin(0)
        ax.set_ylabel(
            r"$(Z_i/\sum Z)/(\gamma_i/\sum\gamma)$", fontsize=28
        )
        ax.tick_params(labelsize=24)
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=12)
        hep.cms.label(**CMS_LABEL, ax=ax)
        paths.extend(
            save_figure(fig, output_dir / "zgamma_double_ratio_highdm")
        )
    else:
        shared = aggregate_low_double_ratios(ratios)
        fig, ax = plt.subplots(figsize=(10.2, 10.2))
        for payload in shared.values():
            values = np.asarray(
                [
                    np.nan if value is None else value
                    for value in payload["double_ratio"]
                ],
                dtype=float,
            )
            errors = np.asarray(
                [
                    np.nan if value is None else value
                    for value in payload["double_ratio_stat"]
                ],
                dtype=float,
            )
            centers, widths = low_ut_geometry(
                payload["isr_group"], len(values)
            )
            ax.errorbar(
                centers,
                values,
                xerr=widths,
                yerr=errors,
                fmt=payload["marker"],
                ls="none",
                lw=2.6,
                ms=6,
                capsize=3,
                color=payload["color"],
                label=payload["label"],
            )
        ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
        ax.set_xlim(300.0, 850.0)
        ax.set_xmargin(0)
        ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
        ax.set_ylabel(
            r"$(Z_i/\sum Z)/(\gamma_i/\sum\gamma)$", fontsize=28
        )
        ax.tick_params(labelsize=24)
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=16, ncol=1)
        hep.cms.label(**CMS_LABEL, ax=ax)
        paths.extend(
            save_figure(
                fig, output_dir / "zgamma_double_ratio_lowdm_nb_isr_shared"
            )
        )
    return paths


def plot_mll(
    measurement: dict[str, Any],
    key: str,
    regime: str,
    output_dir: Path,
) -> list[str]:
    paths: list[str] = []
    source = measurement.get(key) or {}
    for channel in ("DY2E", "DY2M"):
        for group in ("Nb1", "Nb2plus"):
            node = ((source.get(channel) or {}).get(group) or {})
            if not node:
                continue
            first = next(iter(node.values()))
            edges = np.asarray(first["edges"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            data = np.asarray((node.get("data") or {}).get("sumw", []), dtype=float)
            data2 = np.asarray(
                (node.get("data") or {}).get("sumw2", []), dtype=float
            )
            zll = np.asarray((node.get("zll") or {}).get("sumw", []), dtype=float)
            zll2 = np.asarray(
                (node.get("zll") or {}).get("sumw2", []), dtype=float
            )
            other = np.asarray((node.get("other") or {}).get("sumw", []), dtype=float)
            other2 = np.asarray(
                (node.get("other") or {}).get("sumw2", []), dtype=float
            )
            if not len(data):
                continue
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
                gridspec_kw={
                    "height_ratios": [3.2, 1.1],
                    "hspace": 0.04,
                },
            )
            ax.stairs(
                other,
                edges,
                fill=True,
                baseline=0.0,
                color="#6A625F",
                edgecolor="black",
                linewidth=0.7,
                label="Others",
            )
            ax.stairs(
                total,
                edges,
                fill=True,
                baseline=other,
                color="#35B6B4",
                edgecolor="black",
                linewidth=0.7,
                label="DY",
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
            rax.set_xlabel(
                r"$m_{\ell\ell}$ (GeV)", fontsize=30, loc="right"
            )
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.2)
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
            order = ["Stat. unc.", "DY", "Others", "Data"]
            ordered = [
                (handles[labels.index(label)], label)
                for label in order
                if label in labels
            ]
            ax.legend(
                [item[0] for item in ordered],
                [item[1] for item in ordered],
                frameon=False,
                fontsize=18,
                ncol=2,
                loc="upper right",
            )
            hep.cms.label(**CMS_LABEL, ax=ax)
            paths.extend(
                save_figure(
                    fig,
                    output_dir
                    / f"mll_{regime}_{channel.lower()}_{group.lower()}",
                )
            )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--low-sparse", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()

    measurement = read_json(args.measurement)
    exact = read_json(args.exact_inputs)
    if not str(measurement.get("status", "")).startswith(
        "feature_stage_complete"
    ):
        raise SystemExit(f"measurement is incomplete: {measurement.get('status')}")
    if exact.get("status") != "complete":
        raise SystemExit(f"exact inputs are incomplete: {exact.get('status')}")
    low_rz = measurement["rz_low_feature"]
    low_mll_key = "mll_low_feature"
    if args.low_sparse and args.low_sparse.exists():
        sparse = read_json(args.low_sparse)
        if sparse.get("status") != "complete":
            raise SystemExit(f"sparse finalizer incomplete: {sparse.get('status')}")
        low_rz = sparse["rz_low"]
        if sparse.get("mll_low"):
            measurement["mll_low_final"] = sparse["mll_low"]
            low_mll_key = "mll_low_final"

    factors = build_q_sgamma(measurement, exact)
    double_ratios = build_double_ratios(exact)
    factors["lowdm_nb_isr_shared"] = aggregate_low_sgamma(
        factors["lowdm"]
    )
    double_ratios["lowdm_nb_isr_shared"] = (
        aggregate_low_double_ratios(double_ratios["lowdm"])
    )
    plot_dir = args.plot_dir
    plots = {
        "rz_high": plot_rz(measurement["rz_high"], "highdm", plot_dir),
        "rz_low": plot_rz(low_rz, "lowdm", plot_dir),
        "photon_q": plot_q(factors, plot_dir),
        "sgamma_high": plot_sgamma(factors["highdm"], "highdm", plot_dir),
        "sgamma_low": plot_sgamma(factors["lowdm"], "lowdm", plot_dir),
        "zgamma_double_ratio_high": plot_double_ratios(
            double_ratios["highdm"], "highdm", plot_dir
        ),
        "zgamma_double_ratio_low": plot_double_ratios(
            double_ratios["lowdm"], "lowdm", plot_dir
        ),
        "mll_high": plot_mll(
            measurement, "mll_high", "highdm", plot_dir
        ),
        "mll_low": plot_mll(
            measurement, low_mll_key, "lowdm", plot_dir
        ),
    }
    payload = {
        "schema_version": "an_zinv_factors_2024_v1",
        "status": "complete",
        "definition": {
            "prediction": "N_Zinv_pred = RZ * Sgamma_i * N_Zinv_MC_i",
            "Q": "(N_data_GCR - N_other_GCR) / N_gamma_MC within the adopted photon-CR category",
            "Sgamma_i": "(N_data_GCR_i - N_other_GCR_i) / (Q * N_gamma_MC_i)",
            "RZ": "on/off-Z 2x2 matrix solution, combined ee+mumu",
            "high_Q_categories": ["Nb1", "Nb2", "Nb3plus"],
            "low_Q_categories": ["Nb1", "Nb2plus"],
            "low_Sgamma_shared_categories": [
                "Nb1_PISR300to500",
                "Nb1_PISR500plus",
                "Nb2plus_PISR300to500",
                "Nb2plus_PISR500plus",
            ],
        },
        "RZ": {"highdm": measurement["rz_high"], "lowdm": low_rz},
        "photon": factors,
        "z_gamma_double_ratio": double_ratios,
        "plots": plots,
        "inputs": {
            "measurement": str(args.measurement),
            "exact_inputs": str(args.exact_inputs),
            "low_sparse": str(args.low_sparse) if args.low_sparse else None,
        },
    }
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "high_Q": len(factors["highdm"]),
                "low_Q": len(factors["lowdm"]),
                "plot_files": sum(len(items) for items in plots.values()),
                "output": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
