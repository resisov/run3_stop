"""One plotting implementation for every supported tag-and-probe profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SQUARE = (8, 8)
COLORBAR = (12, 10)


def _label(ax: Any, year: str) -> None:
    import mplhep as hep

    hep.cms.label(
        llabel="Work in progress", rlabel=f"{year} (13.6 TeV)", ax=ax, fontsize=18
    )


def _save(fig: Any, path: Path) -> list[str]:
    outputs = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        outputs.append(str(target))
    return outputs


def _event_label(centers: np.ndarray) -> str:
    width = float(np.median(np.diff(centers)))
    return (
        f"Events / {1000.0 * width:g} MeV" if width < 0.1 else f"Events / {width:g} GeV"
    )


def plot_result(result: Mapping[str, Any], output_dir: Path | str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eta_edges = np.asarray(result["probe_abseta_edges"], dtype=float)
    pt_edges = np.asarray(result["probe_pt_edges_gev"], dtype=float)
    n_eta = len(eta_edges) - 1
    n_pt = len(pt_edges) - 1
    sf = np.asarray(
        [item.get("scale_factor", np.nan) for item in result["bins"]]
    ).reshape(n_eta, n_pt)
    uncertainty = np.asarray(
        [item.get("scale_factor_uncertainty", np.nan) for item in result["bins"]]
    ).reshape(n_eta, n_pt)
    collection = str(result["probe_collection"])
    outputs: list[str] = []

    fig, ax = plt.subplots(figsize=SQUARE)
    centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    errors = 0.5 * np.diff(pt_edges)
    for index in range(n_eta):
        ax.errorbar(
            centers,
            sf[index],
            xerr=errors,
            yerr=uncertainty[index],
            marker="o",
            linestyle="none",
            capsize=2,
            label=rf"${eta_edges[index]:g}<|\eta|<{eta_edges[index + 1]:g}$",
        )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel(rf"{collection} $p_{{\mathrm{{T}}}}$ (GeV)")
    ax.set_ylabel("Data/MC scale factor")
    ax.set_xlim(pt_edges[0], pt_edges[-1])
    ax.legend(frameon=False)
    _label(ax, str(result["year"]))
    outputs += _save(fig, output_dir / "scale_factor")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=COLORBAR)
    image = ax.pcolormesh(pt_edges, eta_edges, sf, shading="flat", cmap="viridis")
    for eta_index in range(n_eta):
        for pt_index in range(n_pt):
            ax.text(
                centers[pt_index],
                0.5 * (eta_edges[eta_index] + eta_edges[eta_index + 1]),
                f"{sf[eta_index, pt_index]:.3f}\n$\\pm${uncertainty[eta_index, pt_index]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, label="Data/MC scale factor")
    ax.set_xlabel(rf"{collection} $p_{{\mathrm{{T}}}}$ (GeV)")
    ax.set_ylabel(rf"{collection} $|\eta|$")
    _label(ax, str(result["year"]))
    outputs += _save(fig, output_dir / "scale_factor_heatmap")
    plt.close(fig)

    for item in result["bins"]:
        nominal = item.get("fits", {}).get("nominal", {})
        if not nominal:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
        handles = []
        labels = []
        for column, sample in enumerate(("data", "mc")):
            fitted = nominal[sample]
            mass = np.asarray(fitted["mass_centers_gev"])
            for row, state in enumerate(("pass", "fail")):
                ax = axes[row, column]
                observed = np.asarray(fitted[f"{state}_observed"])
                model = np.asarray(fitted[f"{state}_model"])
                points = ax.errorbar(
                    mass,
                    observed,
                    yerr=np.sqrt(np.maximum(np.abs(observed), 1.0)),
                    fmt="o",
                    color="black",
                    markersize=3,
                )
                (line,) = ax.plot(mass, model, color="#0000ff", linewidth=1.5)
                ax.set_box_aspect(1)
                if column == 0:
                    ax.set_ylabel(_event_label(mass))
                if row == 1:
                    ax.set_xlabel("Tag-probe mass (GeV)")
                if row == 0 and column == 0:
                    handles.extend([points, line])
                    labels.extend(["Observed", "Fit"])
        _label(axes[0, 0], str(result["year"]))
        fig.legend(
            handles, labels, loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=False
        )
        fig.subplots_adjust(
            left=0.10, right=0.86, bottom=0.09, top=0.88, wspace=0.20, hspace=0.12
        )
        outputs += _save(
            fig, output_dir / f"mass_fit_bin_{int(item['flat_index']):03d}"
        )
        plt.close(fig)
    manifest = {"measurement": result["measurement"], "outputs": outputs}
    (output_dir / "plots.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
