#!/usr/bin/env python3
"""Create machine-readable efficiencies and plots from flat trigger counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.special import betaincinv


def cp_interval(passed, total, confidence=0.682689492):
    alpha = 1.0 - confidence
    low = np.where(passed == 0, 0.0, betaincinv(passed, total - passed + 1, alpha / 2))
    high = np.where(passed == total, 1.0, betaincinv(passed + 1, total - passed, 1 - alpha / 2))
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("counts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", default="2024")
    parser.add_argument("--data-label", default="2024 EGamma (C,D,E,F,G,H,I)")
    parser.add_argument("--ratio-ymin", type=float, default=0.95)
    parser.add_argument("--ratio-ymax", type=float, default=1.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.counts.read_text())
    edges = np.asarray(payload["bin_edges_gev"], dtype=float)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) / 2
    dt = np.asarray(payload["data"]["total"], dtype=int)
    dp = np.asarray(payload["data"]["passed"], dtype=int)
    mt = np.asarray(payload["mc"]["sumw_total"], dtype=float)
    mp = np.asarray(payload["mc"]["sumw_passed"], dtype=float)
    mt2 = np.asarray(payload["mc"]["sumw2_total"], dtype=float)
    de = np.divide(dp, dt, out=np.full(len(dt), np.nan), where=dt > 0)
    me = np.divide(mp, mt, out=np.full(len(mt), np.nan), where=mt != 0)
    dl, dh = cp_interval(dp, dt)
    neff = np.divide(mt * mt, mt2, out=np.zeros(len(mt)), where=mt2 > 0)
    # Symmetric effective-entry uncertainty is diagnostic, not an adoption covariance.
    me_unc = np.sqrt(np.divide(me * (1 - me), neff, out=np.zeros(len(me)), where=neff > 0))
    sf = np.divide(de, me, out=np.full(len(de), np.nan), where=me > 0)
    de_unc = np.maximum(de - dl, dh - de)
    sf_unc = sf * np.sqrt(np.divide(de_unc, de, out=np.zeros(len(de)), where=de > 0) ** 2
                          + np.divide(me_unc, me, out=np.zeros(len(me)), where=me > 0) ** 2)
    candidates = []
    for threshold in [250, 260, 270, 280, 290, 300]:
        mask = edges[:-1] >= threshold
        valid = mask & (dt >= 100)
        passed_gate = bool(np.any(valid) and np.all(de[valid] >= 0.98) and np.all((de[valid] - dl[valid]) <= 0.02))
        candidates.append({"threshold_gev": threshold, "passes": passed_gate,
                           "bins_tested": int(np.sum(valid)), "minimum_efficiency": float(np.min(de[valid])) if np.any(valid) else None,
                           "minimum_lower_interval": float(np.min(dl[valid])) if np.any(valid) else None})
    chosen = next((x["threshold_gev"] for x in candidates if x["passes"]), None)
    bins = []
    for i in range(len(dt)):
        bins.append({"low_gev": float(edges[i]), "high_gev": float(edges[i + 1]),
                     "data_total": int(dt[i]), "data_passed": int(dp[i]), "data_efficiency": float(de[i]),
                     "data_interval": [float(dl[i]), float(dh[i])], "mc_effective_entries": float(neff[i]),
                     "mc_efficiency": float(me[i]), "mc_uncertainty_diagnostic": float(me_unc[i]),
                     "data_mc_scale_factor": float(sf[i]), "scale_factor_uncertainty_diagnostic": float(sf_unc[i])})
    result = {"status": "preliminary_not_for_adoption", "measurement": payload["measurement"],
              "plateau_threshold_candidate_gev": chosen, "plateau_tests": candidates, "bins": bins,
              "processing": payload["processing"], "adoption_gates": payload["adoption_gates"]}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.style.use(hep.style.CMS)
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    hep.cms.label("Work in progress", data=True, year=args.year, com=13.6, ax=ax)
    ax.errorbar(centres, de, xerr=widths, yerr=[de - dl, dh - de], fmt="o", color="#0000FF", label=args.data_label)
    ax.errorbar(centres, me, xerr=widths, yerr=me_unc, fmt="s", color="#FF0000", label=r"$t\bar{t}\to\ell\nu qq$ MC")
    ax.set_ylabel("MET trigger efficiency")
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0.1, 1.01, 0.1))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ratio.errorbar(centres, sf, xerr=widths, yerr=sf_unc, fmt="o", color="black")
    ratio.axhline(1, color="tab:red", linestyle="--")
    ratio.set_xlabel(r"$p_T^{miss}$ (GeV)")
    ratio.set_ylabel("Data/MC")
    ratio.set_ylim(args.ratio_ymin, args.ratio_ymax)
    ratio.set_xlim(100, 800)
    ratio.grid(alpha=0.3)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.12, hspace=0.05)
    fig.savefig(args.output_dir / "efficiency.png", dpi=160)
    fig.savefig(args.output_dir / "efficiency.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
