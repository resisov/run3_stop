#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_outputs(outputs_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray, list[Path]]:
    combined: dict[str, dict[str, np.ndarray]] = {}
    pt_edges: np.ndarray | None = None
    eta_edges: np.ndarray | None = None
    used: list[Path] = []
    for path in sorted(outputs_dir.glob("toptageff_shard_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                this_pt = np.asarray(data["pt_edges"], dtype=float)
                this_eta = np.asarray(data["abseta_edges"], dtype=float)
                if pt_edges is None:
                    pt_edges, eta_edges = this_pt, this_eta
                elif not np.array_equal(pt_edges, this_pt) or not np.array_equal(eta_edges, this_eta):
                    raise RuntimeError(f"incompatible binning in {path}")
                for index, process in enumerate(data["process"].astype(str)):
                    target = combined.setdefault(
                        process,
                        {
                            "total": np.zeros_like(data["total"][index], dtype=float),
                            "passed": np.zeros_like(data["passed"][index], dtype=float),
                            "total_signed": np.zeros_like(data["total_signed"][index], dtype=float),
                            "passed_signed": np.zeros_like(data["passed_signed"][index], dtype=float),
                        },
                    )
                    for name in target:
                        target[name] += np.asarray(data[name][index], dtype=float)
            used.append(path)
        except Exception as exc:
            print(f"skip {path}: {type(exc).__name__}: {exc}")
    if pt_edges is None or eta_edges is None:
        raise RuntimeError(f"no valid shard outputs found in {outputs_dir}")
    return combined, pt_edges, eta_edges, used


def efficiency_and_error(passed: np.ndarray, total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    efficiency = np.full_like(total, np.nan, dtype=float)
    error = np.full_like(total, np.nan, dtype=float)
    valid = total > 0
    efficiency[valid] = passed[valid] / total[valid]
    error[valid] = np.sqrt(efficiency[valid] * (1.0 - efficiency[valid]) / total[valid])
    return efficiency, error


def display_process_label(process: str) -> str:
    label = process.split("_Tune", 1)[0]
    label = label.split("-RunIII", 1)[0]
    return label.replace("_", " ")


def draw_map(
    process: str,
    efficiency: np.ndarray,
    pt_edges: np.ndarray,
    eta_edges: np.ndarray,
    outdir: Path,
) -> None:
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    mesh = ax.pcolormesh(
        eta_edges,
        pt_edges,
        efficiency.T,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.05)
    colorbar.set_label("Efficiency")

    for ieta in range(len(eta_edges) - 1):
        x = 0.5 * (eta_edges[ieta] + eta_edges[ieta + 1])
        for ipt in range(len(pt_edges) - 1):
            value = efficiency[ieta, ipt]
            if not np.isfinite(value):
                label = "--"
                color = "black"
            else:
                label = f"{value:.3f}"
                color = "white" if value < 0.45 else "black"
            y = np.sqrt(pt_edges[ipt] * pt_edges[ipt + 1])
            ax.text(x, y, label, ha="center", va="center", fontsize=13, fontweight="bold", color=color)

    ax.set_xlim(eta_edges[0], eta_edges[-1])
    ax.set_ylim(pt_edges[0], pt_edges[-1])
    ax.set_yscale("log")
    y_ticks = [400, 600, 1000, 2000, 3000]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(value) for value in y_ticks])
    ax.set_xlabel(r"FatJet $|\eta|$")
    ax.set_ylabel(r"FatJet $p_{\mathrm{T}}$ (GeV)")
    hep.cms.label(data=False, com=13.6, ax=ax)
    ax.text(0.98, 0.97, display_process_label(process), transform=ax.transAxes, ha="right", va="top", fontsize=16)
    fig.tight_layout()
    safe = process.replace("/", "_").replace(" ", "_")
    fig.savefig(outdir / f"toptageff_2024_{safe}.png", dpi=180)
    fig.savefig(outdir / f"toptageff_2024_{safe}.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and plot per-process Top-tag MC efficiencies")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()

    campaign = args.campaign_dir.resolve()
    plots = campaign / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    combined, pt_edges, eta_edges, used = load_outputs(campaign / "outputs")

    processes = sorted(combined)
    stack = lambda name: np.stack([combined[p][name] for p in processes], axis=0)
    merged_path = campaign / "toptageff_2024_merged.npz"
    np.savez_compressed(
        merged_path,
        process=np.asarray(processes),
        pt_edges=pt_edges,
        abseta_edges=eta_edges,
        total=stack("total"),
        passed=stack("passed"),
        total_signed=stack("total_signed"),
        passed_signed=stack("passed_signed"),
    )

    payload = {
        "status": "complete",
        "shard_outputs_used": len(used),
        "processes": {},
        "pt_edges_gev": pt_edges.tolist(),
        "abseta_edges": eta_edges.tolist(),
        "merged_output": str(merged_path),
    }
    for process in processes:
        efficiency, error = efficiency_and_error(combined[process]["passed"], combined[process]["total"])
        draw_map(process, efficiency, pt_edges, eta_edges, plots)
        payload["processes"][process] = {
            "total_jets": int(np.sum(combined[process]["total"])),
            "passed_jets": int(np.sum(combined[process]["passed"])),
            "efficiency": np.where(np.isfinite(efficiency), efficiency, -1.0).tolist(),
            "stat_error": np.where(np.isfinite(error), error, -1.0).tolist(),
        }
    write_json(campaign / "toptageff_2024_summary.json", payload)
    print(json.dumps({"processes": processes, "shards": len(used), "plots": str(plots)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
