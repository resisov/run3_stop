#!/usr/bin/env python3
"""Evaluate the 2024 High-dM SR event-selection figure of merit.

The calculation intentionally mirrors the earlier Low-dM study: event weights
are the intermediate ``gen_weight`` multiplied by the campaign normalization,
the FoM is S/sqrt(B), and cut correlations use absolute-weighted binary phi
coefficients.  The final 54-bin step is category coverage, not an additional
physics-object selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


LUMINOSITY_FB = 109.82
RECOIL_PT_BINS = np.asarray([250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0])
SIGNAL_BENCHMARKS = ((1000, 1), (1500, 1))
_WORKER_NORMALIZATION: dict[str, Any] | None = None
_WORKER_BENCHMARKS: tuple[tuple[int, int], ...] = SIGNAL_BENCHMARKS

BRANCHES = [
    "dataset_id",
    "gen_weight",
    "mStop",
    "mLSP",
    "pass_base_common",
    "pass_signal_trigger",
    "pass_no_veto_leptons",
    "pass_zero_tau",
    "njet",
    "nb_medium",
    "pass_met_250",
    "pass_ht_300",
    "pass_open_high",
    "feature_SR",
    "met",
    "nboosted_top",
    "nboosted_w",
]

SELECTIONS = [
    ("base_event_quality", "Base quality"),
    ("signal_trigger", "Signal HLT"),
    ("lepton_veto", "e/μ veto"),
    ("tau_veto", "τ veto"),
    ("njet_ge5", "Njet(pT>30) ≥ 5"),
    ("nb_ge1", "Nb(pT>30) ≥ 1"),
    ("met_gt250", "MET > 250 GeV"),
    ("ht_gt300", "HT > 300 GeV"),
    ("high_dphi", "Δφ(j1–4,MET) > 0.5"),
    ("valid_54bin", "Valid 54-bin"),
]


def canonical_process(process: str, dataset: str) -> str:
    if process == "VV":
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "ST"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "TT"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "GJ"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "WtoLNu"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Zto2Nu"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD"
    return process or "other"


def empty_stats() -> dict[str, Any]:
    count = len(SELECTIONS)
    return {
        "sumw": np.zeros(count + 1, dtype=float),
        "sumw2": np.zeros(count + 1, dtype=float),
        "entries": np.zeros(count + 1, dtype=np.int64),
        "corr_w": 0.0,
        "corr_x": np.zeros(count, dtype=float),
        "corr_xx": np.zeros((count, count), dtype=float),
    }


def add_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["sumw"] += source["sumw"]
    target["sumw2"] += source["sumw2"]
    target["entries"] += source["entries"]
    target["corr_w"] += source["corr_w"]
    target["corr_x"] += source["corr_x"]
    target["corr_xx"] += source["corr_xx"]


def valid_54bin_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return coverage of the adopted 54-bin High-dM SR map.

    This is kept explicit and is regression-tested against
    ``selected_an17_recoil54_indices``.
    """

    sr = np.asarray(arrays["feature_SR"], dtype=bool)
    met = np.asarray(arrays["met"], dtype=float)
    nb = np.asarray(arrays["nb_medium"], dtype=int)
    nt = np.asarray(arrays["nboosted_top"], dtype=int)
    nw = np.asarray(arrays["nboosted_w"], dtype=int)
    valid_recoil = np.isfinite(met) & (met >= RECOIL_PT_BINS[0]) & (met < RECOIL_PT_BINS[-1])

    nt0 = (nb >= 1) & (nt == 0)
    nb1_selected = (nb == 1) & (nt >= 1)
    nb2_selected = (nb == 2) & (nt == 1) & ((nw == 0) | (nw == 1))
    nb3_selected = (nb >= 3) & (
        ((nt == 1) & ((nw == 0) | (nw == 1))) | ((nt == 2) & (nw == 0))
    )
    return sr & valid_recoil & (nt0 | nb1_selected | nb2_selected | nb3_selected)


def mask_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(arrays["pass_base_common"], dtype=bool),
            np.asarray(arrays["pass_signal_trigger"], dtype=bool),
            np.asarray(arrays["pass_no_veto_leptons"], dtype=bool),
            np.asarray(arrays["pass_zero_tau"], dtype=bool),
            np.asarray(arrays["njet"], dtype=int) >= 5,
            np.asarray(arrays["nb_medium"], dtype=int) >= 1,
            np.asarray(arrays["pass_met_250"], dtype=bool),
            np.asarray(arrays["pass_ht_300"], dtype=bool),
            np.asarray(arrays["pass_open_high"], dtype=bool),
            valid_54bin_mask(arrays),
        ]
    )


def stats_for_mask(
    arrays: dict[str, np.ndarray], event_mask: np.ndarray, weight: np.ndarray
) -> dict[str, Any]:
    decisions = mask_matrix(arrays)[event_mask]
    selected_weight = np.asarray(weight, dtype=float)[event_mask]
    good = np.isfinite(selected_weight)
    decisions = decisions[good]
    selected_weight = selected_weight[good]
    output = empty_stats()
    if len(selected_weight) == 0:
        return output

    cumulative = np.ones(len(selected_weight), dtype=bool)
    output["sumw"][0] = np.sum(selected_weight)
    output["sumw2"][0] = np.sum(selected_weight * selected_weight)
    output["entries"][0] = len(selected_weight)
    for index in range(len(SELECTIONS)):
        cumulative &= decisions[:, index]
        weights_after_cut = selected_weight[cumulative]
        output["sumw"][index + 1] = np.sum(weights_after_cut)
        output["sumw2"][index + 1] = np.sum(weights_after_cut * weights_after_cut)
        output["entries"][index + 1] = np.count_nonzero(cumulative)

    correlation_weight = np.abs(selected_weight)
    output["corr_w"] = float(np.sum(correlation_weight))
    numeric_decisions = decisions.astype(float)
    output["corr_x"] = np.einsum("n,ni->i", correlation_weight, numeric_decisions)
    output["corr_xx"] = np.einsum(
        "n,ni,nj->ij", correlation_weight, numeric_decisions, numeric_decisions
    )
    return output


def initialize_worker(
    normalization_name: str, benchmarks: tuple[tuple[int, int], ...]
) -> None:
    global _WORKER_NORMALIZATION, _WORKER_BENCHMARKS
    _WORKER_NORMALIZATION = json.loads(Path(normalization_name).read_text())
    _WORKER_BENCHMARKS = benchmarks


def process_root(root_name: str) -> dict[str, Any]:
    root_path = Path(root_name)
    if _WORKER_NORMALIZATION is None:
        raise RuntimeError("worker normalization was not initialized")
    normalization = _WORKER_NORMALIZATION
    benchmarks = _WORKER_BENCHMARKS
    metadata_path = root_path.with_suffix(".json")
    if not metadata_path.is_file():
        return {"error": f"{root_path}: missing metadata sidecar"}
    metadata = json.loads(metadata_path.read_text())

    try:
        with uproot.open(root_path) as root_file:
            if "Events" not in root_file:
                return {"error": f"{root_path}: missing Events tree"}
            tree = root_file["Events"]
            present = set(tree.keys())
            missing = [name for name in BRANCHES if name not in present]
            if missing:
                return {"error": f"{root_path}: missing branches {missing}"}
            arrays = tree.arrays(BRANCHES, library="np", how=dict)
    except Exception as exc:
        return {"error": f"{root_path}: {type(exc).__name__}: {exc}"}

    dataset_ids = np.asarray(arrays["dataset_id"], dtype=np.int64)
    gen_weight = np.asarray(arrays["gen_weight"], dtype=float)
    mstops = np.asarray(arrays["mStop"], dtype=int)
    mlsps = np.asarray(arrays["mLSP"], dtype=int)
    result: dict[str, Any] = {
        "background": {},
        "signals": {},
        "missing_norm": [],
        "root": str(root_path),
    }

    for dataset_id in np.unique(dataset_ids):
        dataset_mask = dataset_ids == dataset_id
        record = (metadata.get("datasets") or {}).get(str(int(dataset_id))) or {}
        if bool(record.get("is_signal")):
            for mstop, mlsp in benchmarks:
                event_mask = dataset_mask & (mstops == mstop) & (mlsps == mlsp)
                if not np.any(event_mask):
                    continue
                mass_key = f"mStop{mstop}_mLSP{mlsp}"
                factor = (
                    (normalization.get("signal_mass_points") or {}).get(mass_key) or {}
                ).get("normalization_factor")
                if factor is None:
                    result["missing_norm"].append(mass_key)
                    continue
                result["signals"][mass_key] = stats_for_mask(
                    arrays, event_mask, gen_weight * float(factor)
                )
            continue

        factor = (
            (normalization.get("dataset_factors") or {}).get(str(int(dataset_id)))
            or {}
        ).get("normalization_factor")
        if factor is None:
            result["missing_norm"].append(str(int(dataset_id)))
            continue
        process = canonical_process(
            str(record.get("process") or "other"), str(record.get("dataset") or "")
        )
        result["background"][process] = stats_for_mask(
            arrays, dataset_mask, gen_weight * float(factor)
        )
    return result


def correlation(stats: dict[str, Any]) -> np.ndarray:
    total = float(stats["corr_w"])
    if total <= 0:
        return np.zeros((len(SELECTIONS), len(SELECTIONS)), dtype=float)
    mean = stats["corr_x"] / total
    second = stats["corr_xx"] / total
    covariance = second - np.outer(mean, mean)
    variance = np.clip(np.diag(covariance), 0.0, None)
    denominator = np.sqrt(np.outer(variance, variance))
    output = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    np.fill_diagonal(output, np.where(variance > 0, 1.0, 0.0))
    return np.clip(output, -1.0, 1.0)


def serializable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "sumw": stats["sumw"].tolist(),
        "sumw2": stats["sumw2"].tolist(),
        "entries": stats["entries"].tolist(),
        "correlation": correlation(stats).tolist(),
    }


def intended_roots(campaign: Path) -> tuple[list[Path], list[Path]]:
    background: list[Path] = []
    signal: list[Path] = []
    argument_files = sorted((campaign / "condor").glob("arguments_part*.txt"))
    if not argument_files:
        argument_files = sorted((campaign / "condor").glob("arguments*.txt"))
    for argument_file in argument_files:
        for line in argument_file.read_text().splitlines():
            fields = line.split()
            if len(fields) < 4 or fields[1] != "nominal":
                continue
            root_path = Path(fields[3])
            if fields[0].startswith("mc_shard_"):
                background.append(root_path)
            elif fields[0].startswith("signal_shard_"):
                signal.append(root_path)
    return background, signal


def finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_fom(
    output_dir: Path,
    labels: list[str],
    background: dict[str, Any],
    signals: dict[str, dict[str, Any]],
) -> None:
    x = np.arange(len(labels))
    fig, (axis_fom, axis_gain) = plt.subplots(
        2, 1, figsize=(13.2, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    colors = ["#1479ff", "#ff8c1a"]
    background_yield = background["sumw"]
    for color, (mass_key, stats) in zip(colors, sorted(signals.items())):
        signal_yield = stats["sumw"]
        fom = np.divide(
            signal_yield,
            np.sqrt(background_yield),
            out=np.full_like(signal_yield, np.nan),
            where=background_yield > 0,
        )
        gain = finite_ratio(fom, np.roll(fom, 1))
        gain[0] = np.nan
        display = mass_key.replace("mStop", "mStop=").replace("_mLSP", " GeV, mLSP=") + " GeV"
        axis_fom.plot(x, fom, marker="o", linewidth=2.1, color=color, label=display)
        axis_gain.plot(x, gain, marker="o", linewidth=1.8, color=color, label=display)
    axis_fom.set_ylabel(r"$S/\sqrt{B}$")
    axis_fom.grid(axis="y", alpha=0.25)
    axis_fom.legend(frameon=False, ncol=2)
    axis_fom.set_title("2024 High-$\\Delta m$ SR cumulative event-selection FoM")
    axis_gain.axhline(1.0, color="0.35", linewidth=1.0)
    axis_gain.set_ylabel("FoM / previous")
    axis_gain.grid(axis="y", alpha=0.25)
    axis_gain.set_xticks(x)
    axis_gain.set_xticklabels(labels, rotation=38, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir / "highdm_selection_fom_2024")


def plot_efficiency(
    output_dir: Path,
    labels: list[str],
    background: dict[str, Any],
    signals: dict[str, dict[str, Any]],
) -> None:
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(13.2, 6.5))
    series: list[tuple[str, np.ndarray, str, str]] = [
        ("Total background", background["sumw"], "#333333", "--")
    ]
    colors = ["#1479ff", "#ff8c1a"]
    for color, (mass_key, stats) in zip(colors, sorted(signals.items())):
        display = mass_key.replace("mStop", "mStop=").replace("_mLSP", " GeV, mLSP=") + " GeV"
        series.append((display, stats["sumw"], color, "-"))
    for label, values, color, style in series:
        efficiency = finite_ratio(values, np.full_like(values, values[0]))
        axis.plot(x, efficiency, marker="o", linewidth=2.0, linestyle=style, color=color, label=label)
    axis.set_yscale("log")
    axis.set_ylim(1e-5, 1.4)
    axis.set_ylabel("Cumulative weighted efficiency")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=38, ha="right")
    axis.grid(axis="y", which="both", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    axis.set_title("2024 High-$\\Delta m$ SR cumulative selection efficiency")
    fig.tight_layout()
    save_figure(fig, output_dir / "highdm_selection_efficiency_2024")


def plot_correlation(
    output_dir: Path, selection_labels: list[str], name: str, stats: dict[str, Any], title: str
) -> None:
    matrix = correlation(stats)
    fig, axis = plt.subplots(figsize=(10.8, 9.4))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(np.arange(len(selection_labels)))
    axis.set_yticks(np.arange(len(selection_labels)))
    axis.set_xticklabels(selection_labels, rotation=48, ha="right")
    axis.set_yticklabels(selection_labels)
    for row in range(len(selection_labels)):
        for column in range(len(selection_labels)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "black",
            )
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Weighted binary correlation")
    fig.tight_layout()
    save_figure(fig, output_dir / name)


def write_tables(
    output_dir: Path,
    stage_keys: list[str],
    stage_labels: list[str],
    background: dict[str, Any],
    signals: dict[str, dict[str, Any]],
) -> None:
    rows = []
    background_yield = background["sumw"]
    for index, (key, label) in enumerate(zip(stage_keys, stage_labels)):
        row: dict[str, Any] = {
            "stage": key,
            "label": label,
            "background_sumw": float(background_yield[index]),
            "background_entries": int(background["entries"][index]),
            "background_efficiency": (
                float(background_yield[index] / background_yield[0]) if background_yield[0] else None
            ),
        }
        for mass_key, stats in sorted(signals.items()):
            signal_yield = stats["sumw"]
            row[f"{mass_key}_sumw"] = float(signal_yield[index])
            row[f"{mass_key}_entries"] = int(stats["entries"][index])
            row[f"{mass_key}_efficiency"] = (
                float(signal_yield[index] / signal_yield[0]) if signal_yield[0] else None
            )
            row[f"{mass_key}_s_over_sqrt_b"] = (
                float(signal_yield[index] / math.sqrt(background_yield[index]))
                if background_yield[index] > 0
                else None
            )
        rows.append(row)
    with (output_dir / "highdm_selection_fom_2024.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 2024 High-dM event-selection figure of merit",
        "",
        "FoM: `S/sqrt(B)`. The 54-bin entry is category coverage, not a new object selection.",
        "",
        "| Selection | Background | "
        + " | ".join(f"{mass} S/sqrt(B)" for mass in sorted(signals))
        + " |",
        "|---|---:|" + "---:|" * len(signals),
    ]
    for row in rows:
        values = []
        for mass_key in sorted(signals):
            value = row[f"{mass_key}_s_over_sqrt_b"]
            values.append("n/a" if value is None else f"{value:.5g}")
        lines.append(
            f"| {row['label']} | {row['background_sumw']:.6g} | " + " | ".join(values) + " |"
        )
    (output_dir / "highdm_selection_fom_2024.md").write_text("\n".join(lines) + "\n")


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    intended_background, intended_signal = intended_roots(args.campaign)
    if not intended_background or not intended_signal:
        raise RuntimeError("campaign argument files contain no MC or signal jobs")
    available_background = [path for path in intended_background if path.is_file() and path.stat().st_size > 0]
    available_signal = [path for path in intended_signal if path.is_file() and path.stat().st_size > 0]
    missing_background = sorted(set(intended_background) - set(available_background))
    missing_signal = sorted(set(intended_signal) - set(available_signal))
    if (missing_background or missing_signal) and not args.allow_partial:
        raise RuntimeError(
            "nominal intermediate is incomplete: "
            f"background {len(available_background)}/{len(intended_background)}, "
            f"signal {len(available_signal)}/{len(intended_signal)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_tuple = tuple(SIGNAL_BENCHMARKS)
    root_names = [str(path) for path in available_background + available_signal]
    background_by_process: dict[str, dict[str, Any]] = {}
    signal_stats = {
        f"mStop{mstop}_mLSP{mlsp}": empty_stats() for mstop, mlsp in SIGNAL_BENCHMARKS
    }
    errors: list[str] = []
    missing_normalization: set[str] = set()
    processed_roots = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.normalization), benchmark_tuple),
    ) as pool:
        futures = [pool.submit(process_root, root_name) for root_name in root_names]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if "error" in result:
                errors.append(result["error"])
                continue
            processed_roots += 1
            missing_normalization.update(result.get("missing_norm") or [])
            for process, stats in result["background"].items():
                add_stats(background_by_process.setdefault(process, empty_stats()), stats)
            for mass_key, stats in result["signals"].items():
                add_stats(signal_stats[mass_key], stats)
            if done % 250 == 0 or done == len(futures):
                print(f"processed {done}/{len(futures)} roots", flush=True)

    background_total = empty_stats()
    for stats in background_by_process.values():
        add_stats(background_total, stats)

    stage_keys = ["intermediate_rows"] + [item[0] for item in SELECTIONS]
    stage_labels = ["Intermediate rows"] + [item[1] for item in SELECTIONS]
    fom: dict[str, Any] = {}
    for mass_key, stats in signal_stats.items():
        values = np.divide(
            stats["sumw"],
            np.sqrt(background_total["sumw"]),
            out=np.full_like(stats["sumw"], np.nan),
            where=background_total["sumw"] > 0,
        )
        gain = finite_ratio(values, np.roll(values, 1))
        gain[0] = np.nan
        fom[mass_key] = {
            "s_over_sqrt_b": [None if not math.isfinite(value) else float(value) for value in values],
            "relative_to_previous": [
                None if not math.isfinite(value) else float(value) for value in gain
            ],
        }

    signals_present = {
        mass_key: int(stats["entries"][0]) > 0 for mass_key, stats in signal_stats.items()
    }
    complete = not missing_background and not missing_signal and not errors and not missing_normalization
    complete = complete and all(signals_present.values())
    payload = {
        "schema_version": "highdm_selection_fom_v1",
        "status": "complete" if complete else "partial",
        "classification": "event-selection diagnostic; adopted High-dM SR definition",
        "definition": {
            "fom": "S/sqrt(B)",
            "correlation": "absolute-normalized-weighted Pearson/phi correlation of individual binary selections, conditioned on rows present in the broad intermediate",
            "event_weight": "gen_weight * 2024 xsec*lumi/sumw normalization; deferred pileup/btag/lepton/photon/top-pT scale factors are not included",
            "luminosity_fb": LUMINOSITY_FB,
            "selection_order": stage_keys,
            "selection_labels": stage_labels,
            "valid_54bin": "category-coverage diagnostic for the adopted 54-bin High-dM SR map; not an independent object/event selection",
            "signal_benchmarks": [
                {"mStop": mstop, "mLSP": mlsp} for mstop, mlsp in SIGNAL_BENCHMARKS
            ],
        },
        "inputs": {
            "campaign": str(args.campaign),
            "normalization": str(args.normalization),
            "intended_background_roots": len(intended_background),
            "available_background_roots": len(available_background),
            "intended_signal_roots": len(intended_signal),
            "available_signal_roots": len(available_signal),
            "processed_roots": processed_roots,
            "allow_partial": bool(args.allow_partial),
            "git_commit": git_commit(args.repo),
        },
        "background_total": serializable_stats(background_total),
        "background_by_process": {
            key: serializable_stats(value) for key, value in sorted(background_by_process.items())
        },
        "signals": {key: serializable_stats(value) for key, value in signal_stats.items()},
        "signals_present": signals_present,
        "fom": fom,
        "missing_outputs": {
            "background": [str(path) for path in missing_background],
            "signal": [str(path) for path in missing_signal],
        },
        "missing_normalization": sorted(missing_normalization),
        "errors": errors,
    }
    json_path = args.output_dir / "highdm_selection_fom_2024.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if not all(signals_present.values()) or background_total["entries"][0] == 0:
        print(f"wrote partial diagnostics to {json_path}", flush=True)
        return 2

    plot_fom(args.output_dir, stage_labels, background_total, signal_stats)
    plot_efficiency(args.output_dir, stage_labels, background_total, signal_stats)
    plot_correlation(
        args.output_dir,
        [item[1] for item in SELECTIONS],
        "highdm_selection_correlation_background_2024",
        background_total,
        "2024 High-$\\Delta m$ background selection correlation",
    )
    for mass_key, stats in sorted(signal_stats.items()):
        plot_correlation(
            args.output_dir,
            [item[1] for item in SELECTIONS],
            f"highdm_selection_correlation_{mass_key}_2024",
            stats,
            f"2024 High-$\\Delta m$ signal selection correlation: {mass_key}",
        )
    write_tables(args.output_dir, stage_keys, stage_labels, background_total, signal_stats)
    print(f"wrote {json_path}; status={payload['status']}", flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
