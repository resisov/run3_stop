from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_lowdm_category_nn_out import CATEGORIES, signed_stack


PROCESS_ORDER = ("DY", "GJ", "QCD", "VV", "ST", "WtoLNu", "TT", "Zto2Nu")
PROCESS_LABELS = {
    "DY": r"DY $\to \ell\ell$",
    "GJ": r"$\gamma$+jets",
    "QCD": "QCD multijet",
    "VV": "Diboson",
    "ST": "Single top",
    "WtoLNu": r"W $\to \ell\nu$",
    "TT": r"$t\bar{t}$",
    "Zto2Nu": r"Z $\to \nu\nu$",
}
PROCESS_COLORS = {
    "DY": "#CC79A7",
    "GJ": "#F4D35E",
    "QCD": "#9E77B5",
    "VV": "#78C6A3",
    "ST": "#A5C8E1",
    "WtoLNu": "#D9C6A5",
    "TT": "#7A9FC2",
    "Zto2Nu": "#E6A84F",
}
TOPOLOGY_IDS = {"T2tt": 1, "T2bW": 2, "T2tb": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_benchmark(
    mstop: np.ndarray,
    mlsp: np.ndarray,
    signal: np.ndarray,
    requested_mstop: int,
    requested_mlsp: int,
) -> tuple[int, int]:
    pairs = np.unique(np.stack((mstop[signal], mlsp[signal]), axis=1), axis=0)
    if not len(pairs):
        raise RuntimeError("locked test sample contains no signal events")
    distance = (pairs[:, 0] - requested_mstop) ** 2 + (pairs[:, 1] - requested_mlsp) ** 2
    selected = pairs[int(np.argmin(distance))]
    return int(selected[0]), int(selected[1])


def grouped_process(name: str) -> str:
    if name in PROCESS_ORDER:
        return name
    # These are not simulated-background histogram components.  Preserve
    # their explicit identities for callers that first map all events and
    # subsequently select signal or the appropriate collision-data stream.
    if name in {"SMS", "signal", "EGamma", "JetMET", "Muon"}:
        return name
    raise RuntimeError(f"unregistered process cannot be plotted: {name}")


def histogram_by_process(
    process: np.ndarray,
    coordinate: np.ndarray,
    weight: np.ndarray,
    edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    histograms: dict[str, np.ndarray] = {}
    sumw2: dict[str, np.ndarray] = {}
    for name in PROCESS_ORDER:
        selected = process == name
        histograms[name] = np.histogram(coordinate[selected], bins=edges, weights=weight[selected])[0]
        sumw2[name] = np.histogram(coordinate[selected], bins=edges, weights=weight[selected] ** 2)[0]
    return histograms, sumw2


def relative_stat(background: np.ndarray, sumw2: np.ndarray) -> np.ndarray:
    uncertainty = np.sqrt(sumw2)
    return np.divide(
        uncertainty,
        np.abs(background),
        out=np.full_like(background, np.nan, dtype=float),
        where=background != 0.0,
    )


def decorate_stat_axis(axis: plt.Axes, relative: np.ndarray, edges: np.ndarray) -> None:
    axis.stairs(relative, edges, color="black", linewidth=1.4)
    axis.fill_between(
        edges,
        np.zeros(len(edges)),
        np.r_[relative, relative[-1]],
        step="post",
        color="0.82",
        hatch="////",
        alpha=0.65,
        linewidth=0.0,
    )
    axis.axhline(0.30, color="#b2182b", linestyle=":", linewidth=1.3)
    axis.set_ylim(0.0, 2.05)
    axis.set_ylabel(r"MC stat. / $|B|$", fontsize=18)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot the locked-test GNN output and adopted Low-dM 30-bin templates."
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-mstop", type=int, default=1050)
    parser.add_argument("--benchmark-mlsp", type=int, default=900)
    parser.add_argument(
        "--benchmark-topology", choices=tuple(TOPOLOGY_IDS), default="T2tt"
    )
    parser.add_argument("--signal-scale", type=float, default=100.0)
    parser.add_argument("--luminosity-fb", type=float, default=109.82)
    opts = parser.parse_args()
    output = opts.output or opts.result
    output.mkdir(parents=True, exist_ok=True)
    scores_path = opts.result / "test_scores.root"
    templates_path = opts.result / "lowdm_30bin_test_templates.root"
    test_summary_path = opts.result / "test_summary.json"
    for path in (scores_path, templates_path, test_summary_path, opts.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(opts.manifest.read_text())
    test_summary = json.loads(test_summary_path.read_text())
    selected_score_edges = np.asarray(
        test_summary["thirty_bin_model"]["score_edges"], dtype=float
    )
    if len(selected_score_edges) != 4 or not np.all(np.diff(selected_score_edges) > 0.0):
        raise RuntimeError("test summary contains invalid validation-selected score edges")
    by_physical = manifest["normalization"]["by_physical_dataset_id"]
    with uproot.open(scores_path) as root_file:
        arrays = root_file["Events"].arrays(library="np")
    signal = np.asarray(arrays["is_signal"], dtype=bool)
    physical = np.asarray(arrays["physical_dataset_id"], dtype=np.int64)
    score = np.asarray(arrays["gnn_score"], dtype=float)
    weight = np.asarray(arrays["signed_normalized_weight"], dtype=float)
    mstop = np.asarray(arrays["mStop"], dtype=np.int32)
    mlsp = np.asarray(arrays["mLSP"], dtype=np.int32)
    if np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
        raise RuntimeError("invalid GNN score in locked test output")
    process = np.full(len(score), "signal", dtype=object)
    for physical_id in np.unique(physical[~signal]):
        record = by_physical.get(str(int(physical_id)))
        if record is None:
            raise RuntimeError(f"missing process mapping for physical_dataset_id={physical_id}")
        process[(physical == physical_id) & ~signal] = grouped_process(str(record["process"]))

    topology_signal = signal.copy()
    if "signal_topology_id" in arrays:
        topology_signal &= (
            np.asarray(arrays["signal_topology_id"], dtype=np.int32)
            == TOPOLOGY_IDS[opts.benchmark_topology]
        )
    elif opts.benchmark_topology != "T2tt":
        raise RuntimeError("legacy score table only supports a T2tt benchmark")
    benchmark = choose_benchmark(
        mstop, mlsp, topology_signal, opts.benchmark_mstop, opts.benchmark_mlsp
    )
    benchmark_name = f"{opts.benchmark_topology}_{benchmark[0]}_{benchmark[1]}"
    benchmark_mask = topology_signal & (mstop == benchmark[0]) & (mlsp == benchmark[1])
    signal_label = (
        rf"{opts.benchmark_topology} $({benchmark[0]},{benchmark[1]})$ GeV "
        rf"$\times {opts.signal_scale:g}$"
    )

    import mplhep as hep

    hep.style.use("CMS")
    plt.rcParams["hatch.linewidth"] = 1.2

    score_edges = np.linspace(0.0, 1.0, 26)
    background_process = process[~signal]
    background_score = score[~signal]
    background_weight = weight[~signal]
    grouped, grouped_sumw2 = histogram_by_process(
        background_process, background_score, background_weight, score_edges
    )
    background = sum(grouped.values(), np.zeros(len(score_edges) - 1))
    background_sumw2 = sum(grouped_sumw2.values(), np.zeros(len(score_edges) - 1))
    signal_hist = np.histogram(score[benchmark_mask], bins=score_edges, weights=weight[benchmark_mask])[0]
    relative = relative_stat(background, background_sumw2)

    fig, (axis, stat_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.04},
    )
    signed_stack(
        axis,
        [grouped[name] for name in PROCESS_ORDER],
        score_edges,
        [PROCESS_LABELS[name] for name in PROCESS_ORDER],
        [PROCESS_COLORS[name] for name in PROCESS_ORDER],
    )
    axis.stairs(background, score_edges, color="black", linewidth=1.35, zorder=7)
    uncertainty = np.sqrt(background_sumw2)
    axis.fill_between(
        score_edges,
        np.r_[background - uncertainty, (background - uncertainty)[-1]],
        np.r_[background + uncertainty, (background + uncertainty)[-1]],
        step="post",
        facecolor="0.82",
        edgecolor="0.15",
        hatch="////",
        linewidth=0.0,
        alpha=0.65,
        label="MC stat. unc.",
    )
    axis.stairs(
        signal_hist * opts.signal_scale,
        score_edges,
        color="#e31a1c",
        linewidth=2.4,
        linestyle="--",
        label=signal_label,
        zorder=9,
    )
    decorate_stat_axis(stat_axis, relative, score_edges)
    axis.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    axis.set_ylabel("Events / 0.04", fontsize=25)
    stat_axis.set_xlabel("GNN score", fontsize=25, loc="right")
    stat_axis.set_xlim(0.0, 1.0)
    axis.text(
        0.03,
        0.77,
        r"Adopted Low-$\Delta m$ SR, High-$\Delta m$ SR veto" + "\n"
        + r"Locked 70% test sample; $1/0.7$ luminosity normalization" + "\n"
        + "All simulated background processes",
        transform=axis.transAxes,
        fontsize=14,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel=rf"{opts.luminosity_fb:.2f} fb$^{{-1}}$ (13.6 TeV)",
        ax=axis,
    )
    axis.legend(fontsize=11, ncol=3, frameon=False, loc="upper center")
    score_base = output / "locked_test_gnn_score"
    fig.savefig(score_base.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(score_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with uproot.open(templates_path) as root_file:
        template_background = np.asarray(root_file["background_total"].values(), dtype=float)
        template_sumw2 = np.asarray(root_file["background_sumw2"].values(), dtype=float)
        template_signal = np.asarray(root_file[f"signal/{benchmark_name}"].values(), dtype=float)
        available_processes = {
            key.split("/", 1)[1].split(";", 1)[0]
            for key in root_file.keys(recursive=True)
            if key.startswith("background/") and "/" in key
        }
        template_grouped = {name: np.zeros(30, dtype=float) for name in PROCESS_ORDER}
        for raw_name in available_processes:
            group = grouped_process(raw_name)
            template_grouped[group] += np.asarray(
                root_file[f"background/{raw_name}"].values(), dtype=float
            )
    flat_edges = np.arange(31, dtype=float)
    template_relative = relative_stat(template_background, template_sumw2)
    fig, (axis, stat_axis) = plt.subplots(
        2,
        1,
        figsize=(19.0, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.04},
    )
    signed_stack(
        axis,
        [template_grouped[name] for name in PROCESS_ORDER],
        flat_edges,
        [PROCESS_LABELS[name] for name in PROCESS_ORDER],
        [PROCESS_COLORS[name] for name in PROCESS_ORDER],
    )
    axis.stairs(template_background, flat_edges, color="black", linewidth=1.35, zorder=7)
    template_uncertainty = np.sqrt(template_sumw2)
    axis.fill_between(
        flat_edges,
        np.r_[template_background - template_uncertainty, (template_background - template_uncertainty)[-1]],
        np.r_[template_background + template_uncertainty, (template_background + template_uncertainty)[-1]],
        step="post",
        facecolor="0.82",
        edgecolor="0.15",
        hatch="////",
        linewidth=0.0,
        alpha=0.65,
        label="MC stat. unc.",
    )
    axis.stairs(
        template_signal * opts.signal_scale,
        flat_edges,
        color="#e31a1c",
        linewidth=2.4,
        linestyle="--",
        label=signal_label,
        zorder=9,
    )
    decorate_stat_axis(stat_axis, template_relative, flat_edges)
    nscore = len(selected_score_edges) - 1
    for boundary in range(nscore, 30, nscore):
        axis.axvline(boundary, color="0.20", linewidth=1.1)
        stat_axis.axvline(boundary, color="0.20", linewidth=1.1)
    for index, (_, _, label) in enumerate(CATEGORIES):
        center = index * nscore + nscore / 2.0
        axis.text(
            center,
            0.88,
            f"C{index + 1}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14,
            weight="bold",
        )
        stat_axis.text(
            center,
            -0.43,
            label.replace(r"\n", "\n"),
            transform=stat_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
        )
    stat_axis.set_xticks(
        np.arange(30) + 0.5,
        [
            f"{selected_score_edges[index]:.2f}-{selected_score_edges[index + 1]:.2f}"
            for index in range(nscore)
        ]
        * len(CATEGORIES),
        rotation=90,
        fontsize=9,
    )
    axis.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    axis.set_ylabel("Events / GNN-score bin", fontsize=25)
    stat_axis.set_xlabel(
        r"GNN score repeated inside each adopted Low-$\Delta m$ category",
        fontsize=18,
        loc="right",
        labelpad=70,
    )
    stat_axis.set_xlim(0.0, 30.0)
    hep.cms.label(
        llabel="Work in progress",
        rlabel=rf"{opts.luminosity_fb:.2f} fb$^{{-1}}$ (13.6 TeV)",
        ax=axis,
    )
    axis.legend(fontsize=11, ncol=5, frameon=False, loc="upper center")
    template_base = output / "lowdm_30bin_locked_test_gnn"
    fig.savefig(template_base.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(template_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    selection_path = opts.result / "selection.json"
    selection = json.loads(selection_path.read_text())
    ranking = selection["ranking"]
    fig, axis = plt.subplots(figsize=(10.5, max(4.0, 0.65 * len(ranking) + 1.8)))
    names = [row["name"] for row in ranking][::-1]
    macro_auc = np.asarray([row["macro_mass_auc"] for row in ranking][::-1], dtype=float)
    minimum_auc = np.asarray(
        [row.get("minimum_mass_auc", np.nan) for row in ranking][::-1], dtype=float
    )
    positions = np.arange(len(names))
    axis.hlines(positions, np.nanmin(macro_auc) - 0.01, macro_auc, color="#4C78A8", linewidth=3)
    axis.scatter(macro_auc, positions, color="#4C78A8", s=65, zorder=5, label="Macro mass-point AUC")
    finite_minimum = np.isfinite(minimum_auc)
    if np.any(finite_minimum):
        axis.scatter(
            minimum_auc[finite_minimum],
            positions[finite_minimum],
            color="#E45756",
            marker="D",
            s=48,
            zorder=5,
            label="Minimum mass-point AUC",
        )
    axis.set_yticks(positions, names, fontsize=10)
    finite_values = np.r_[macro_auc, minimum_auc[finite_minimum]]
    lower = max(0.45, float(np.min(finite_values)) - 0.03)
    axis.set_xlim(lower, min(1.0, float(np.max(macro_auc)) + 0.03))
    axis.set_xlabel("Validation AUC", fontsize=17)
    axis.set_title("50-epoch GNN hyperparameter ranking (validation only)", fontsize=17)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    ranking_base = output / "validation_hyperparameter_ranking"
    fig.savefig(ranking_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(ranking_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    per_mass_auc = test_summary["metrics"]["per_mass_auc"]
    mass_rows = []
    for name, auc in per_mass_auc.items():
        topology, mstop_text, mlsp_text = name.split("_")
        if topology != opts.benchmark_topology:
            continue
        current_mstop = int(mstop_text)
        current_mlsp = int(mlsp_text)
        mass_rows.append((current_mstop, current_mlsp, current_mstop - current_mlsp, float(auc)))
    fig, axis = plt.subplots(figsize=(11.0, 7.0))
    for delta_m in sorted({row[2] for row in mass_rows}):
        local = sorted((row for row in mass_rows if row[2] == delta_m), key=lambda row: row[0])
        axis.plot(
            [row[0] for row in local],
            [row[3] for row in local],
            marker="o",
            linewidth=1.8,
            label=rf"$\Delta m={delta_m}$ GeV",
        )
    axis.axhline(0.5, color="0.35", linestyle=":", linewidth=1.2)
    axis.set_xlabel(r"$m_{\widetilde{t}}$ [GeV]", fontsize=18)
    axis.set_ylabel("Locked-test weighted AUC", fontsize=18)
    axis.set_ylim(max(0.45, min(row[3] for row in mass_rows) - 0.04), 1.0)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    axis.set_title(
        f"Mass-by-mass {opts.benchmark_topology} discrimination in the diagonal domain",
        fontsize=17,
    )
    mass_auc_base = output / "locked_test_auc_by_mass"
    fig.savefig(mass_auc_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(mass_auc_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema_version": "gnn_lowdm_full_plot_summary_v1",
        "status": "complete",
        "benchmark": {
            "topology": opts.benchmark_topology,
            "mStop": benchmark[0],
            "mLSP": benchmark[1],
            "display_scale": opts.signal_scale,
        },
        "normalization": (
            "Only the untouched 70% test partition is plotted; weights are scaled by 1/0.7 "
            "to the full 109.82/fb expectation."
        ),
        "events": {
            "test": len(score),
            "background": int(np.count_nonzero(~signal)),
            "signal_all_masses": int(np.count_nonzero(signal)),
            "benchmark_signal": int(np.count_nonzero(benchmark_mask)),
        },
        "yields": {
            "background": float(np.sum(background)),
            "benchmark_signal": float(np.sum(signal_hist)),
        },
        "thirty_bin_mc_stat": {
            "nonpositive_background_bins": int(np.count_nonzero(template_background <= 0.0)),
            "bins_above_30pct": int(np.count_nonzero(template_relative > 0.30)),
            "bins_above_50pct": int(np.count_nonzero(template_relative > 0.50)),
        },
        "no_nuisance_sensitivity_diagnostic": {
            "warning": "Asimov Z comparison only; this is not a nuisance-aware expected limit.",
            "score30_over_raw34_median": float(
                np.median(
                    [
                        row.get("score30_over_raw34")
                        if row.get("score30_over_raw34") is not None
                        else row["asimov_z_no_nuisances_score30"]
                        / row["asimov_z_no_nuisances_raw34"]
                        for row in test_summary["thirty_bin_model"]["signals"].values()
                        if row.get("score30_over_raw34") is not None
                        or row.get("asimov_z_no_nuisances_raw34", 0.0) > 0.0
                    ]
                )
            ),
            "mass_points_score30_above_raw34": int(
                np.count_nonzero(
                    [
                        (
                            row.get("score30_over_raw34")
                            if row.get("score30_over_raw34") is not None
                            else row["asimov_z_no_nuisances_score30"]
                            / row["asimov_z_no_nuisances_raw34"]
                        )
                        > 1.0
                        for row in test_summary["thirty_bin_model"]["signals"].values()
                        if row.get("score30_over_raw34") is not None
                        or row.get("asimov_z_no_nuisances_raw34", 0.0) > 0.0
                    ]
                )
            ),
            "mass_points_compared": int(
                np.count_nonzero(
                    [
                        row.get("score30_over_raw34") is not None
                        or row.get("asimov_z_no_nuisances_raw34", 0.0) > 0.0
                        for row in test_summary["thirty_bin_model"]["signals"].values()
                    ]
                )
            ),
        },
        "provenance": {
            "scores": {"path": str(scores_path), "sha256": sha256(scores_path)},
            "templates": {"path": str(templates_path), "sha256": sha256(templates_path)},
            "test_summary": {"path": str(test_summary_path), "sha256": sha256(test_summary_path)},
            "selection": {"path": str(selection_path), "sha256": sha256(selection_path)},
            "manifest": {"path": str(opts.manifest), "sha256": sha256(opts.manifest)},
        },
        "artifacts": {
            "score_png": score_base.with_suffix(".png").name,
            "score_pdf": score_base.with_suffix(".pdf").name,
            "thirty_bin_png": template_base.with_suffix(".png").name,
            "thirty_bin_pdf": template_base.with_suffix(".pdf").name,
            "ranking_png": ranking_base.with_suffix(".png").name,
            "ranking_pdf": ranking_base.with_suffix(".pdf").name,
            "mass_auc_png": mass_auc_base.with_suffix(".png").name,
            "mass_auc_pdf": mass_auc_base.with_suffix(".pdf").name,
        },
    }
    (output / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
