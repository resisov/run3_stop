import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import loc
from coffea.util import load
from matplotlib.ticker import NullFormatter

plt.style.use(hep.style.CMS)

SAMPLES = {
    "TTto2L2Nu": "TTto2L2Nu_",
    "TTtoLNu2Q": "TTtoLNu2Q_",
    "TTto4Q": "TTto4Q_",
    "WtoLNu": "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_",
    "Zto2Nu": "Zto2Nu-2Jets_Bin-1J-PTNuNu-200to400_",
}

flavours = {
    5: "b",
    4: "c",
    0: "lf",
}

wps = ["loose", "medium", "tight"]


def plot_btag(merged, outdir):
    tagger = "PNetUParT"
    mcs = merged[tagger].keys()
    os.makedirs(outdir, exist_ok=True)
    for mc in mcs:
        h = merged[tagger][mc]
        for wp in wps:
            for flav, flav_name in flavours.items():
                h_pass = h[{"wp": loc(wp), "btag": loc("pass"), "flavor": loc(flav)}]
                h_fail = h[{"wp": loc(wp), "btag": loc("fail"), "flavor": loc(flav)}]
                num = h_pass.values(flow=False)
                den = num + h_fail.values(flow=False)
                eff = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)
                pt_edges = h_pass.axes[0].edges
                eta_edges = h_pass.axes[1].edges
                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(eta_edges, pt_edges, eff, shading="auto", vmin=0, vmax=1)
                fig.colorbar(mesh, ax=ax).set_label("Efficiency")
                hep.cms.label(llabel="Simulation", rlabel="(13.6 TeV)", loc=0, ax=ax)
                ax.set_xlabel(r"Jet |$\eta$|")
                ax.set_ylabel(r"Jet $p_{T}$ (GeV)")
                ax.set_ylim(30, 1000)
                ax.set_yscale("log")
                annotate(ax, mesh, eff, pt_edges, eta_edges, fontsize=14)
                fig.tight_layout()
                fig.savefig(f"{outdir}/{mc}_{flav_name}_{wp}.png", dpi=200)
                plt.close(fig)
                print(f"saved: {outdir}/{mc}_{flav_name}_{wp}.png")


def annotate(ax, mesh, values, pt_edges, eta_edges, fontsize):
    for ipt, pt in enumerate(np.sqrt(pt_edges[:-1] * pt_edges[1:])):
        for ieta, eta in enumerate(0.5 * (eta_edges[:-1] + eta_edges[1:])):
            value = values[ipt, ieta]
            if not np.isfinite(value):
                continue
            r, g, b, _ = mesh.cmap(mesh.norm(value))
            color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
            ax.text(eta, pt, f"{value:.3f}", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")


def rebin_pt(values, source_edges, display_edges):
    indices = []
    for edge in display_edges:
        matches = np.flatnonzero(source_edges == edge)
        if len(matches) != 1:
            raise ValueError(f"Display edge {edge} is not a source bin boundary")
        indices.append(int(matches[0]))
    rebinned = np.stack([values[lo:hi].sum(axis=0) for lo, hi in zip(indices[:-1], indices[1:])])
    if not np.array_equal(rebinned.sum(axis=0), values[indices[0]:indices[-1]].sum(axis=0)):
        raise ValueError("Display rebinning changed the jet counts")
    return rebinned


def draw_topw(efficiency, pt_edges, eta_edges, output):
    with plt.rc_context({"axes.labelsize": 32, "xtick.labelsize": 26,
                         "ytick.labelsize": 26, "savefig.bbox": None}):
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.13, top=0.90)
        mesh = ax.pcolormesh(eta_edges, pt_edges, np.ma.masked_invalid(efficiency),
                             cmap="viridis", shading="flat", vmin=0, vmax=1)
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.045)
        colorbar.set_label("Efficiency", fontsize=32)
        ax.set_xlim(eta_edges[0], eta_edges[-1])
        ax.set_ylim(pt_edges[0], pt_edges[-1])
        ax.set_yscale("log")
        ticks = [value for value in (200, 400, 600, 1000, 2000, 3000)
                 if pt_edges[0] <= value <= pt_edges[-1]]
        ax.set_yticks(ticks, labels=[str(value) for value in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_box_aspect(1)
        ax.set_xlabel(r"Jet $|\eta|$")
        ax.set_ylabel(r"Jet $p_{\mathrm{T}}$ (GeV)")
        ax.yaxis.set_label_coords(-0.18, 1)
        hep.cms.label(llabel="Simulation", rlabel="(13.6 TeV)", loc=0, ax=ax, fontsize=28)
        annotate(ax, mesh, efficiency, pt_edges, eta_edges, fontsize=21)
        for suffix in (".pdf", ".png"):
            fig.savefig(output.with_suffix(suffix), dpi=200)
        plt.close(fig)


def plot_topw(merged, source, outdir, samples, selection):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": selection,
        "flavor": "inclusive over all five decay-containment categories",
        "weight": "unweighted jet counts",
        "scale_factors_applied": False,
        "figure_size_inches": [12, 10],
        "display_rebinning": "Sum pass and fail counts before taking their ratio; source histograms unchanged.",
        "outside_display_range": "Retained in the source; not included in displayed cells.",
        "panels": [],
    }
    for name in samples:
        for tagger, wp in (("top", 0.5078), ("w", 0.9385)):
            datasets = merged["GlobalParT3_" + ("Top" if tagger == "top" else "W")]
            matches = [key for key in datasets if key.startswith(SAMPLES[name])]
            if len(matches) != 1:
                raise ValueError(f"Expected one dataset for {name}; found {matches}")
            h = datasets[matches[0]]
            passed = h[{"selection": loc(selection), "tag": loc("pass")}].project("pt", "abseta")
            failed = h[{"selection": loc(selection), "tag": loc("fail")}].project("pt", "abseta")
            pt_edges = np.asarray([400, 500, 600, 800, 1000, 1500, 2000, 3000], dtype=float)
            if tagger == "w" or selection == "score_only":
                pt_edges = np.concatenate(([200, 300], pt_edges))
            eta_edges = passed.axes["abseta"].edges
            num = rebin_pt(passed.values(), passed.axes["pt"].edges, pt_edges)
            fail = rebin_pt(failed.values(), failed.axes["pt"].edges, pt_edges)
            den = num + fail
            if not np.all(np.isfinite(den)) or np.any(num < 0) or np.any(fail < 0):
                raise ValueError(f"Invalid unweighted counts for {name}/{tagger}")
            efficiency = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            stem = f"{name}_{tagger}"
            draw_topw(efficiency, pt_edges, eta_edges, outdir / stem)
            payload["panels"].append({
                "sample": name, "dataset": matches[0], "tagger": tagger, "wp": wp,
                "pt_edges_gev": pt_edges.tolist(), "abseta_edges": eta_edges.tolist(),
                "passed": num.tolist(), "failed": fail.tolist(),
                "efficiency": [[float(v) if np.isfinite(v) else None for v in row] for row in efficiency],
                "empty_cells": int(np.sum(den == 0)),
                "passed_outside_display": float(passed.values().sum() - num.sum()),
                "failed_outside_display": float(failed.values().sum() - fail.sum()),
                "pdf": stem + ".pdf", "png": stem + ".png",
            })
    (outdir / "plot_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"panels": len(payload["panels"]), "output_dir": str(outdir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Draw b-tag or Top/W MC efficiency maps")
    parser.add_argument("--tagger", choices=("b", "topw"), default="b")
    parser.add_argument("--merged", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample", choices=tuple(SAMPLES), action="append")
    parser.add_argument("--selection", choices=("analysis", "score_only"), default="analysis")
    args = parser.parse_args()
    source = args.merged or Path("hists/btageff2022EE.merged" if args.tagger == "b" else "hists/topwtageff2024.merged")
    outdir = args.output_dir or Path("btageff2022EE" if args.tagger == "b" else "topwtageff")
    merged = load(source)
    if args.tagger == "b":
        plot_btag(merged, outdir)
    else:
        plot_topw(merged, source, outdir, args.sample or list(SAMPLES), args.selection)


if __name__ == "__main__":
    main()
