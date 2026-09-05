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


KEY_BRANCHES = ("physical_dataset_id", "run", "luminosityBlock", "event")
PROCESS_ORDER = ("TT", "WtoLNu", "Zto2Nu")
PROCESS_LABELS = {
    "TT": "Top",
    "WtoLNu": r"W $\to \ell\nu$",
    "Zto2Nu": r"Z $\to \nu\nu$",
}
PROCESS_COLORS = {"TT": "#7A9FC2", "WtoLNu": "#D9C6A5", "Zto2Nu": "#E6A84F"}

# Exact order and raw-bin sizes from LOWDM_NBGE1_CATEGORY_SIZES in
# workflow/build_flat_boosted_recoil_hists.py.  Raw bins 0--7 are the two
# removed Nb=0 categories; the adopted SR categories begin at raw bin 8.
CATEGORIES = (
    ("Nb1_PISR300to500_PTb20to40", 4, r"$N_b=1$\n$300\leq p_T^{ISR}<500$\n$20<p_T^b<40$"),
    ("Nb1_PISR300to500_PTb40to70", 4, r"$N_b=1$\n$300\leq p_T^{ISR}<500$\n$40<p_T^b<70$"),
    ("Nb1_PISR500plus_PTb20to40", 4, r"$N_b=1$\n$p_T^{ISR}\geq500$\n$20<p_T^b<40$"),
    ("Nb1_PISR500plus_PTb40to70", 4, r"$N_b=1$\n$p_T^{ISR}\geq500$\n$40<p_T^b<70$"),
    ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", 3, r"$N_b\geq2$\n$300\leq p_T^{ISR}<500$\n$40<p_T^b<80$"),
    ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", 3, r"$N_b\geq2$\n$300\leq p_T^{ISR}<500$\n$80<p_T^b<140$"),
    ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", 3, r"$N_b\geq2, N_j\geq7$\n$300\leq p_T^{ISR}<500$\n$p_T^b>140$"),
    ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", 3, r"$N_b\geq2$\n$p_T^{ISR}\geq500$\n$40<p_T^b<80$"),
    ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", 3, r"$N_b\geq2$\n$p_T^{ISR}\geq500$\n$80<p_T^b<140$"),
    ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", 3, r"$N_b\geq2, N_j\geq7$\n$p_T^{ISR}\geq500$\n$p_T^b>140$"),
)
SCORE_EDGES = np.asarray([0.0, 0.75, 0.92, 1.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_to_category(raw: np.ndarray) -> np.ndarray:
    boundaries = 8 + np.cumsum([size for _, size, _ in CATEGORIES])
    category = np.searchsorted(boundaries, raw, side="right")
    category[(raw < 8) | (raw > 41)] = -1
    return category


def signed_stack(
    axis: plt.Axes,
    values: list[np.ndarray],
    edges: np.ndarray,
    labels: list[str],
    colors: list[str],
) -> None:
    positive_bottom = np.zeros(len(edges) - 1, dtype=float)
    negative_bottom = np.zeros(len(edges) - 1, dtype=float)
    for current, label, color in zip(values, labels, colors):
        positive = np.clip(current, 0.0, None)
        negative = np.clip(current, None, 0.0)
        axis.stairs(
            positive_bottom + positive,
            edges,
            baseline=positive_bottom,
            fill=True,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            label=label,
        )
        axis.stairs(
            negative_bottom + negative,
            edges,
            baseline=negative_bottom,
            fill=True,
            color=color,
            edgecolor="black",
            linewidth=0.55,
        )
        positive_bottom += positive
        negative_bottom += negative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace MET by GNN score inside each adopted Low-dM SR category."
    )
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--category-map", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("dnn", "gnn", "transformer"), default="gnn")
    parser.add_argument("--luminosity-fb", type=float, default=109.82)
    parser.add_argument("--signal-xsec-pb", type=float, default=0.006426)
    parser.add_argument("--signal-sumw", type=float, default=43107.0)
    parser.add_argument("--signal-scale", type=float, default=100.0)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)

    category_payload = json.loads(opts.category_map.read_text())
    if category_payload.get("status") != "complete":
        raise RuntimeError("category map is incomplete")
    category_by_key = {
        tuple(int(value) for value in row["key"]): int(row["lowdm_search_bin_SR"])
        for row in category_payload["rows"]
    }
    normalization = json.loads(opts.normalization.read_text())
    by_physical_id = normalization["by_physical_dataset_id"]
    with uproot.open(opts.scores) as root_file:
        arrays = root_file["Events"].arrays(
            (*KEY_BRANCHES, "is_signal", "gen_weight", "signed_normalized_weight", f"{opts.model}_score"),
            library="np",
        )
    keys = list(zip(*(np.asarray(arrays[name], dtype=np.int64) for name in KEY_BRANCHES)))
    missing = [key for key in keys if key not in category_by_key]
    if missing:
        raise RuntimeError(f"category map is missing {len(missing)} scored events")
    raw_bin = np.asarray([category_by_key[key] for key in keys], dtype=np.int32)
    category = raw_to_category(raw_bin)
    adopted = category >= 0
    is_signal = np.asarray(arrays["is_signal"], dtype=bool)
    score = np.asarray(arrays[f"{opts.model}_score"], dtype=float)
    if np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
        raise RuntimeError("invalid NN score")

    process = np.full(len(score), "signal", dtype=object)
    physical = np.asarray(arrays["physical_dataset_id"], dtype=np.int64)
    for physical_id in np.unique(physical[~is_signal]):
        record = by_physical_id.get(str(int(physical_id)))
        if record is None:
            raise RuntimeError(f"missing normalization for physical_dataset_id={physical_id}")
        process[(physical == physical_id) & ~is_signal] = record["process"]
    unexpected = sorted(set(process[~is_signal]) - set(PROCESS_ORDER))
    if unexpected:
        raise RuntimeError(f"unexpected background processes: {unexpected}")
    signal_factor = opts.signal_xsec_pb * opts.luminosity_fb * 1000.0 / opts.signal_sumw
    weight = np.where(
        is_signal,
        np.asarray(arrays["gen_weight"], dtype=float) * signal_factor,
        np.asarray(arrays["signed_normalized_weight"], dtype=float),
    )

    score_bin = np.clip(
        np.searchsorted(SCORE_EDGES, score, side="right") - 1,
        0,
        len(SCORE_EDGES) - 2,
    )
    nscore = len(SCORE_EDGES) - 1
    flat_index = category * nscore + score_bin
    nflat = len(CATEGORIES) * nscore
    flat_edges = np.arange(nflat + 1, dtype=float)
    grouped: dict[str, np.ndarray] = {}
    grouped_sumw2: dict[str, np.ndarray] = {}
    for name in PROCESS_ORDER:
        mask = adopted & ~is_signal & (process == name)
        grouped[name] = np.bincount(flat_index[mask], weights=weight[mask], minlength=nflat)
        grouped_sumw2[name] = np.bincount(
            flat_index[mask], weights=weight[mask] ** 2, minlength=nflat
        )
    background = sum(grouped.values(), np.zeros(nflat, dtype=float))
    background_sumw2 = sum(grouped_sumw2.values(), np.zeros(nflat, dtype=float))
    background_stat = np.sqrt(background_sumw2)
    signal_mask = adopted & is_signal
    signal = np.bincount(
        flat_index[signal_mask], weights=weight[signal_mask], minlength=nflat
    )

    import mplhep as hep

    hep.style.use("CMS")
    plt.rcParams["hatch.linewidth"] = 1.2
    fig, (axis, stat_axis) = plt.subplots(
        2, 1, figsize=(19.0, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.04},
    )
    signed_stack(
        axis,
        [grouped[name] for name in PROCESS_ORDER],
        flat_edges,
        [PROCESS_LABELS[name] for name in PROCESS_ORDER],
        [PROCESS_COLORS[name] for name in PROCESS_ORDER],
    )
    axis.stairs(background, flat_edges, color="black", linewidth=1.35, zorder=7)
    positive = background > 0.0
    lower = np.where(positive, np.maximum(background - background_stat, 1.0e-4), np.nan)
    upper = np.where(positive, background + background_stat, np.nan)
    axis.fill_between(
        flat_edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post",
        facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0.0,
        alpha=0.65, label="MC stat. unc.",
    )
    signal_label = rf"T2tt $(1050,900)$ GeV $\times {opts.signal_scale:g}$"
    axis.stairs(
        signal * opts.signal_scale, flat_edges, color="#ff0000", linewidth=2.4,
        linestyle="--", label=signal_label, zorder=9,
    )
    relative = np.divide(
        background_stat, np.abs(background), out=np.full_like(background, np.nan),
        where=background != 0.0,
    )
    stat_axis.stairs(relative, flat_edges, color="black", linewidth=1.5)
    stat_axis.fill_between(
        flat_edges, np.zeros(len(flat_edges)), np.r_[relative, relative[-1]], step="post",
        color="0.82", hatch="////", alpha=0.65, linewidth=0.0,
    )
    stat_axis.axhline(0.30, color="#b2182b", linestyle=":", linewidth=1.4)
    nonpositive = background <= 0.0
    if np.any(nonpositive):
        stat_axis.scatter(
            np.flatnonzero(nonpositive) + 0.5,
            np.full(np.count_nonzero(nonpositive), 1.95),
            marker="x", s=35, color="#b2182b", zorder=10, label=r"$B\leq0$",
        )
    for boundary in range(nscore, nflat, nscore):
        axis.axvline(boundary, color="0.20", linewidth=1.2)
        stat_axis.axvline(boundary, color="0.20", linewidth=1.2)
    for index, (_, _, label) in enumerate(CATEGORIES):
        center = index * nscore + nscore / 2.0
        display_label = label.replace(r"\n", "\n")
        axis.text(
            center, 0.88, f"C{index + 1}", transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=15, weight="bold",
        )
        stat_axis.text(
            center, -0.43, display_label, transform=stat_axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=9,
        )
    tick_centers = np.arange(nflat) + 0.5
    score_labels = ["0-.75", ".75-.92", ".92-1"] * len(CATEGORIES)
    stat_axis.set_xticks(tick_centers, score_labels, rotation=90, fontsize=10)
    for current in (axis, stat_axis):
        current.set_xlim(0.0, float(nflat))
        current.set_xmargin(0)
        current.tick_params(which="major", direction="in", top=True, right=True)
    axis.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    positive_scale = max(float(np.max(background + background_stat)), float(np.max(signal * opts.signal_scale)))
    negative_scale = max(1.0, abs(float(np.min(background - background_stat))))
    axis.set_ylim(-negative_scale * 2.0, positive_scale * 35.0)
    axis.set_ylabel("Events / GNN-score bin", fontsize=28)
    stat_axis.set_ylabel(r"MC stat. / $|B|$", fontsize=21)
    stat_axis.set_ylim(0.0, 2.05)
    stat_axis.set_xlabel(
        "GNN score bin repeated inside each adopted Low-$\\Delta m$ category",
        fontsize=20, loc="right", labelpad=70,
    )
    axis.text(
        0.015, 0.73,
        r"Adopted Low-$\Delta m$ SR: $N_b\geq1$, High-$\Delta m$ SR veto" + "\n"
        + r"Exact original category assignment; $\sigma\mathcal{L}/\Sigma w$ normalized" + "\n"
        + "Dominant backgrounds only; post-skim SF not applied",
        transform=axis.transAxes, ha="left", va="top", fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel=rf"{opts.luminosity_fb:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=axis,
    )
    handles, labels = axis.get_legend_handles_labels()
    desired = ["MC stat. unc.", *[PROCESS_LABELS[name] for name in PROCESS_ORDER], signal_label]
    ordered = [(handles[labels.index(label)], label) for label in desired if label in labels]
    axis.legend(
        [item[0] for item in ordered], [item[1] for item in ordered],
        fontsize=15, ncol=5, frameon=False, loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
    )
    flat_base = opts.output / f"lowdm_categories_{opts.model}_searchbins"
    fig.savefig(flat_base.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(flat_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # A panel view makes the repeated score axis directly readable without
    # changing any selection or normalization.
    fig, axes = plt.subplots(2, 5, figsize=(20.0, 8.5), sharex=True, sharey=True)
    for category_index, (axis, (_, _, label)) in enumerate(zip(axes.flat, CATEGORIES)):
        start = category_index * nscore
        stop = start + nscore
        local_grouped = [grouped[name][start:stop] for name in PROCESS_ORDER]
        signed_stack(
            axis, local_grouped, SCORE_EDGES,
            [PROCESS_LABELS[name] for name in PROCESS_ORDER],
            [PROCESS_COLORS[name] for name in PROCESS_ORDER],
        )
        axis.stairs(background[start:stop], SCORE_EDGES, color="black", linewidth=1.2)
        axis.stairs(
            signal[start:stop] * opts.signal_scale, SCORE_EDGES,
            color="#ff0000", linewidth=2.0, linestyle="--",
            label=signal_label if category_index == 0 else None,
        )
        axis.set_yscale("symlog", linthresh=0.5, linscale=0.8)
        axis.set_xlim(0.0, 1.0)
        axis.text(
            0.04, 0.96,
            f"C{category_index + 1}\n{label.replace(r'\n', chr(10))}",
            transform=axis.transAxes, ha="left", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        axis.grid(axis="y", alpha=0.16)
    for axis in axes[1, :]:
        axis.set_xlabel("GNN score", fontsize=18)
    for axis in axes[:, 0]:
        axis.set_ylabel("Events", fontsize=19)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=4, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.91), fontsize=14,
    )
    fig.suptitle(
        r"CMS Work in progress   109.82 fb$^{-1}$ (13.6 TeV)"
        + "\n"
        + r"Adopted Low-$\Delta m$ categories with MET binning replaced by GNN score; "
        + signal_label,
        fontsize=18, y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    panel_base = opts.output / f"lowdm_categories_{opts.model}_score_panels"
    fig.savefig(panel_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(panel_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    category_rows = []
    for category_index, (name, raw_size, label) in enumerate(CATEGORIES):
        start = category_index * nscore
        stop = start + nscore
        category_rows.append(
            {
                "index": category_index + 1,
                "name": name,
                "label": label,
                "raw_met_bins": [
                    8 + sum(size for _, size, _ in CATEGORIES[:category_index]),
                    8 + sum(size for _, size, _ in CATEGORIES[:category_index]) + raw_size - 1,
                ],
                "events": {
                    "background": int(np.count_nonzero(adopted & ~is_signal & (category == category_index))),
                    "signal": int(np.count_nonzero(adopted & is_signal & (category == category_index))),
                },
                "yield": {
                    "background": float(np.sum(background[start:stop])),
                    "signal": float(np.sum(signal[start:stop])),
                },
                "score_bins": [
                    {
                        "low": float(SCORE_EDGES[score_index]),
                        "high": float(SCORE_EDGES[score_index + 1]),
                        "background": float(background[start + score_index]),
                        "background_stat": float(background_stat[start + score_index]),
                        "signal": float(signal[start + score_index]),
                    }
                    for score_index in range(nscore)
                ],
            }
        )

    dropped = ~adopted
    summary = {
        "schema_version": "lowdm_category_gnn_score_v1",
        "status": "complete",
        "physics_proposal": (
            "Keep the ten adopted Low-dM topology categories and replace their internal MET/recoil "
            "bins by three common GNN-score bins."
        ),
        "score_edges": SCORE_EDGES.tolist(),
        "categories": category_rows,
        "coverage": {
            "scored_events": len(score),
            "category_map_matches": len(raw_bin),
            "adopted_background_events": int(np.count_nonzero(adopted & ~is_signal)),
            "adopted_signal_events": int(np.count_nonzero(adopted & is_signal)),
            "excluded_nb0_background_events": int(np.count_nonzero(dropped & ~is_signal)),
            "excluded_nb0_signal_events": int(np.count_nonzero(dropped & is_signal)),
            "adopted_background_yield": float(np.sum(weight[adopted & ~is_signal])),
            "adopted_signal_yield": float(np.sum(weight[adopted & is_signal])),
            "excluded_nb0_background_yield": float(np.sum(weight[dropped & ~is_signal])),
            "excluded_nb0_signal_yield": float(np.sum(weight[dropped & is_signal])),
        },
        "mc_stat": {
            "candidate_bins": nflat,
            "nonpositive_background_bins": int(np.count_nonzero(background <= 0.0)),
            "bins_above_30pct": int(np.count_nonzero(relative > 0.30)),
            "bins_above_50pct": int(np.count_nonzero(relative > 0.50)),
        },
        "normalization": {
            "luminosity_fb": opts.luminosity_fb,
            "signal_xsec_pb": opts.signal_xsec_pb,
            "signal_sumw": opts.signal_sumw,
            "signal_display_scale": opts.signal_scale,
            "background_formula": "signed_normalized_weight from external source-shard-disjoint evaluation",
            "signal_formula": "gen_weight * xsec_pb * luminosity_pb / mass_point_sumw",
        },
        "limitations": [
            "The current external sample is too small for a statistical-model decision after ten-category splitting.",
            "Only TT, Zto2Nu, and WtoLNu are included.",
            "Post-skim scale factors, systematics, and GNN-score migrations are not included.",
            "Nonpositive signed-MC bins are displayed on a symmetric-log scale and marked in the lower panel.",
        ],
        "provenance": {
            "scores": {"path": str(opts.scores), "sha256": sha256(opts.scores)},
            "category_map": {"path": str(opts.category_map), "sha256": sha256(opts.category_map)},
            "normalization": {"path": str(opts.normalization), "sha256": sha256(opts.normalization)},
        },
        "artifacts": {
            "flat_png": flat_base.name + ".png",
            "flat_pdf": flat_base.name + ".pdf",
            "panels_png": panel_base.name + ".png",
            "panels_pdf": panel_base.name + ".pdf",
            "root": "lowdm_category_gnn_score.root",
        },
    }
    summary_path = opts.output / "lowdm_category_gnn_score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with uproot.recreate(opts.output / "lowdm_category_gnn_score.root") as root_file:
        root_file["candidate/background_total"] = (background, flat_edges)
        root_file["candidate/background_sumw2"] = (background_sumw2, flat_edges)
        root_file["candidate/signal_T2tt_1050_900"] = (signal, flat_edges)
        for name in PROCESS_ORDER:
            root_file[f"candidate/background_{name}"] = (grouped[name], flat_edges)
        for category_index, (name, _, _) in enumerate(CATEGORIES):
            start = category_index * nscore
            stop = start + nscore
            root_file[f"categories/{name}/background_total"] = (background[start:stop], SCORE_EDGES)
            root_file[f"categories/{name}/background_sumw2"] = (background_sumw2[start:stop], SCORE_EDGES)
            root_file[f"categories/{name}/signal_T2tt_1050_900"] = (signal[start:stop], SCORE_EDGES)
            for process_name in PROCESS_ORDER:
                root_file[f"categories/{name}/background_{process_name}"] = (
                    grouped[process_name][start:stop], SCORE_EDGES
                )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_bins": nflat,
                "adopted_background_events": summary["coverage"]["adopted_background_events"],
                "adopted_signal_events": summary["coverage"]["adopted_signal_events"],
                "nonpositive_bins": summary["mc_stat"]["nonpositive_background_bins"],
                "output": str(opts.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
