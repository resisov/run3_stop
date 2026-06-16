#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from coffea.util import load
import mplhep
import matplotlib.pyplot as plt
import hist

# ----------------------------------------------------------------------
# Arguments / Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--year",
    default="2024",
    choices=["2022pre", "2022post", "2023pre", "2023post", "2024", "2025"],
    help="Year to plot",
)
parser.add_argument(
    "--hist-dir",
    default="hists",
    help="Directory containing .scaled files",
)
parser.add_argument(
    "--plot-dir",
    default=None,
    help="Output plot directory. Default: plots_<year>",
)
parser.add_argument(
    "--nominal-only",
    action="store_true",
    help="Skip external shifted .scaled files and draw nominal-only plots.",
)
parser.add_argument(
    "--input-scaled",
    default=None,
    help="Nominal .scaled input. Default: <hist-dir>/stop_<year>_nominal.scaled",
)
parser.add_argument(
    "--met-unclustered-up",
    default=None,
    help="metUnclusteredUp .scaled input. Default: <hist-dir>/stop_<year>_metUnclusteredUp.scaled",
)
parser.add_argument(
    "--met-unclustered-down",
    default=None,
    help="metUnclusteredDown .scaled input. Default: <hist-dir>/stop_<year>_metUnclusteredDown.scaled",
)
parser.add_argument(
    "--jes-total-up",
    default=None,
    help="jesTotalUp .scaled input. Default: <hist-dir>/stop_<year>_jesTotalUp.scaled",
)
parser.add_argument(
    "--jes-total-down",
    default=None,
    help="jesTotalDown .scaled input. Default: <hist-dir>/stop_<year>_jesTotalDown.scaled",
)
parser.add_argument(
    "--luminosity-fb",
    type=float,
    default=None,
    help="Luminosity in fb^-1 for the CMS label. Default: script year table.",
)
args = parser.parse_args()

YEAR = args.year
HIST_DIR = args.hist_dir
PLOT_DIR = args.plot_dir if args.plot_dir is not None else f"plots_{YEAR}"
NOMINAL_ONLY = args.nominal_only

NOMINAL_FILE = args.input_scaled if args.input_scaled is not None else f"{HIST_DIR}/stop_{YEAR}_nominal.scaled"
#NOMINAL_FILE = f"hists/Legacy_0514/stop_2024_nominal.scaled"

METUNCL_UP_FILE = args.met_unclustered_up if args.met_unclustered_up is not None else f"{HIST_DIR}/stop_{YEAR}_metUnclusteredUp.scaled"
METUNCL_DOWN_FILE = args.met_unclustered_down if args.met_unclustered_down is not None else f"{HIST_DIR}/stop_{YEAR}_metUnclusteredDown.scaled"

JES_UP_FILE = args.jes_total_up if args.jes_total_up is not None else f"{HIST_DIR}/stop_{YEAR}_jesTotalUp.scaled"
JES_DOWN_FILE = args.jes_total_down if args.jes_total_down is not None else f"{HIST_DIR}/stop_{YEAR}_jesTotalDown.scaled"

LUMI_LABEL = {
    "2022pre": "7.98 fb$^{-1}$ (13.6 TeV)",
    "2022post": "26.67 fb$^{-1}$ (13.6 TeV)",
    "2023pre": "17.79 fb$^{-1}$ (13.6 TeV)",
    "2023post": "9.45 fb$^{-1}$ (13.6 TeV)",
    "2024": "109.82 fb$^{-1}$ (13.6 TeV)",
    "2025": "110.58 fb$^{-1}$ (13.6 TeV)",
}
if args.luminosity_fb is not None:
    LUMI_LABEL[YEAR] = f"{args.luminosity_fb:g} fb$^{{-1}}$ (13.6 TeV)"

INCLUDE_OVERFLOW_IN_LAST = True
INCLUDE_UNDERFLOW_IN_FIRST = False

error_opts = {
    "step": "post",
    "label": "Stat. + syst. unc",
    "hatch": "///",
    "facecolor": "black",
    "alpha": 0.3,
    "edgecolor": (0, 0, 0, 0.3),
    "linewidth": 0,
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def reduce_to_1d(vals, vars_):
    """
    마지막 축을 x축으로 두고, 나머지 축은 모두 합쳐 1D 배열로 만든다.
    """
    while vals.ndim > 1:
        vals = vals.sum(axis=0)
        if vars_ is not None:
            vars_ = vars_.sum(axis=0)
    return vals, vars_


def get_rebinned_hist(h, key):
    """변수별 rebin 규칙 적용"""
    if key == "j1pt":
        h = h[{"j1pt": hist.rebin(5)}]
    elif key == "j2pt":
        h = h[{"j2pt": hist.rebin(5)}]
    elif key == "fj1pt":
        h = h[{"fj1pt": hist.rebin(5)}]
    elif key == "fj1TvsQCD":
        h = h[{"fj1TvsQCD": hist.rebin(10)}]
    elif key == "metpt_10GeVbins":
        h = h[{"metpt_10GeVbins": hist.rebin(5)}]
    return h


def get_hist_safe(container, key, process, region, systematic):
    try:
        h = container[key][process][{"region": region, "systematic": systematic}]
        h = get_rebinned_hist(h, key)
        return h
    except Exception:
        return None


def get_axis_edges_centers(h):
    edges = h.axes[-1].edges
    centers = h.axes[-1].centers
    return edges, centers


def merge_flow_bins_last_axis(vals_flow, vars_flow, nbins):
    vals_flow = np.asarray(vals_flow)

    if vars_flow is not None:
        vars_flow = np.asarray(vars_flow)

    if vals_flow.shape[-1] == nbins:
        return vals_flow, vars_flow

    if vals_flow.shape[-1] == nbins + 2:
        vals = vals_flow[..., 1:-1].copy()
        vars_ = vars_flow[..., 1:-1].copy() if vars_flow is not None else None

        if INCLUDE_UNDERFLOW_IN_FIRST:
            vals[..., 0] += vals_flow[..., 0]
            if vars_ is not None:
                vars_[..., 0] += vars_flow[..., 0]

        if INCLUDE_OVERFLOW_IN_LAST:
            vals[..., -1] += vals_flow[..., -1]
            if vars_ is not None:
                vars_[..., -1] += vars_flow[..., -1]

        return vals, vars_

    print(
        f"[WARNING] Unexpected flow shape: last axis length={vals_flow.shape[-1]}, "
        f"expected nbins={nbins} or nbins+2={nbins + 2}. "
        f"Using first {nbins} bins without explicit flow merging."
    )

    vals = vals_flow[..., :nbins].copy()
    vars_ = vars_flow[..., :nbins].copy() if vars_flow is not None else None
    return vals, vars_


def hist_values_variances_with_overflow(h):
    edges, _ = get_axis_edges_centers(h)
    nbins = len(edges) - 1

    try:
        vals_flow = h.values(flow=True)
        vars_flow = h.variances(flow=True)
        vals, vars_ = merge_flow_bins_last_axis(vals_flow, vars_flow, nbins)
    except Exception:
        vals = h.values()
        vars_ = h.variances()

    return vals, vars_


def hist_values_with_overflow(h):
    vals, _ = hist_values_variances_with_overflow(h)
    return vals


def get_total_bkg_1d_from_container(container, key, region, systematic, stacks):
    """
    Separate .scaled 파일에 저장된 external variation을 읽어서,
    total background 1D histogram으로 합산한다.
    """
    vals_list = []
    bins = None

    if container is None:
        return None, None

    if key not in container:
        return None, None

    for mc in stacks:
        if mc not in container[key]:
            continue

        h = get_hist_safe(container, key, mc, region, systematic)
        if h is None:
            continue

        edges, _ = get_axis_edges_centers(h)
        if bins is None:
            bins = edges

        vals = hist_values_with_overflow(h)
        vals_1d, _ = reduce_to_1d(vals, None)
        vals_list.append(vals_1d)

    if len(vals_list) == 0:
        return None, bins

    return np.array(vals_list).sum(axis=0), bins


def get_data_process(region):
    if "GCR" in region or "DY2E" in region:
        return "EGamma"
    elif "DY2M" in region:
        return "Muon"
    else:
        return "JetMET"


def get_pretty_label(mc):
    if mc == "Z (inv)":
        return r"Z$\rightarrow$$\nu$$\bar{\nu}$"
    elif mc == "W (lnu)":
        return r"W$\rightarrow$$\ell$$\nu$"
    elif mc == "TT":
        return r"t$\bar{t}$"
    else:
        return mc


def same_binning(bins_a, bins_b):
    if bins_a is None or bins_b is None:
        return False
    if len(bins_a) != len(bins_b):
        return False
    return np.allclose(bins_a, bins_b, rtol=0, atol=1e-9)


def load_scaled_bkg(path, label):
    """
    external systematic file을 안전하게 읽는다.
    없으면 None을 반환하고 nominal fallback 없이 systematic 자체를 skip한다.
    """
    if not os.path.exists(path):
        print(f"[INFO] Missing {label} file: {path}")
        return None

    obj = load(path)
    if "bkg" not in obj:
        print(f"[WARNING] File has no 'bkg' key: {path}")
        return None

    return obj["bkg"]


# ----------------------------------------------------------------------
# Load histograms
# ----------------------------------------------------------------------
if not os.path.exists(NOMINAL_FILE):
    raise FileNotFoundError(f"Nominal histogram file not found: {NOMINAL_FILE}")

print(f"[INFO] Loading nominal file: {NOMINAL_FILE}")
myhist = load(NOMINAL_FILE)

bkg = myhist["bkg"]
data = myhist["data"]
sig = myhist["bkg"]

if NOMINAL_ONLY:
    print("[WARNING] nominal-only mode: external shifted uncertainties are not included")
    bkg_metuncl_up = None
    bkg_metuncl_down = None
    bkg_jes_up = None
    bkg_jes_down = None
else:
    bkg_metuncl_up = load_scaled_bkg(METUNCL_UP_FILE, "metUnclusteredUp")
    bkg_metuncl_down = load_scaled_bkg(METUNCL_DOWN_FILE, "metUnclusteredDown")

    bkg_jes_up = load_scaled_bkg(JES_UP_FILE, "jesTotalUp")
    bkg_jes_down = load_scaled_bkg(JES_DOWN_FILE, "jesTotalDown")

# ----------------------------------------------------------------------
# Plot config
# ----------------------------------------------------------------------
nominal_syst = "nominal"

syst_pairs = {
    "pileup": ("pileupUp", "pileupDown"),
    "electron_id": ("electron_idUp", "electron_idDown"),
    "electron_hlt": ("electron_hltUp", "electron_hltDown"),
    "muon_id": ("muon_idUp", "muon_idDown"),
    "muon_hlt": ("muon_hltUp", "muon_hltDown"),
    "btagSF_bc_correlated": ("btagSF_bc_correlatedUp", "btagSF_bc_correlatedDown"),
    "btagSF_bc_uncorrelated": ("btagSF_bc_uncorrelatedUp", "btagSF_bc_uncorrelatedDown"),
    "btagSF_light_correlated": ("btagSF_light_correlatedUp", "btagSF_light_correlatedDown"),
    "btagSF_light_uncorrelated": ("btagSF_light_uncorrelatedUp", "btagSF_light_uncorrelatedDown"),
}

external_syst_pairs = {}

if bkg_metuncl_up is not None and bkg_metuncl_down is not None:
    external_syst_pairs["metUnclustered"] = (
        bkg_metuncl_up,
        "metUnclusteredUp",
        bkg_metuncl_down,
        "metUnclusteredDown",
    )
else:
    print("[INFO] metUnclustered systematic files are missing. Skipping metUnclustered uncertainty.")

if bkg_jes_up is not None and bkg_jes_down is not None:
    external_syst_pairs["jesTotal"] = (
        bkg_jes_up,
        "jesTotalUp",
        bkg_jes_down,
        "jesTotalDown",
    )
else:
    print("[INFO] jesTotal systematic files are missing. Skipping jesTotal uncertainty.")

all_syst_names = list(syst_pairs.keys()) + list(external_syst_pairs.keys())

lumi_unc = 0.016

stacks = [
    "VV",
    "Single Top",
    "TT",
    "DY",
    "Gamma + Jets",
    "W (lnu)",
    "Z (inv)",
    "QCD Multijet",
]

colors = [
    "#6b705c",
    "#8e7dbe",
    "#99c1b9",
    "cyan",
    "purple",
    "#f1e3d3",
    "#f2d0a9",
    "#d88c9a",
]

color_map = dict(zip(stacks, colors))

signals = [
    "SMS-2Stop-Par-mStop-1000",
    #"SMS-2Stop-Par-mStop-1500",
    #"SMS-2Stop-Par-mStop-600",
]

signal_labels = [
    r"$m_{\tilde{t}}$ = 1000 GeV",
    #r"$m_{\tilde{t}}$ = 1500 GeV",
    #r"$m_{\tilde{t}}$ = 600 GeV",
]

signal_ltypes = ["solid", "dashed", "dotted"]
signal_colors = ["red", "darkred", "maroon"]

regions_to_plot = [
    "cat1_preselection",
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
    "cat7_SR_highDeltaM",
]

os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Print yields for checking
# ----------------------------------------------------------------------
for region in regions_to_plot:
    print("\n" + "=" * 80)
    print(f"Yield check for region: {region}")
    print("=" * 80)

    total_bkg_yield = 0.0

    for process in stacks:
        if "nMuon" in bkg and process in bkg["nMuon"]:
            h = get_hist_safe(
                bkg,
                "nMuon",
                process,
                region,
                "nominal",
            )
            if h is None:
                continue

            vals, _ = hist_values_variances_with_overflow(h)
            yield_ = vals.sum()
            total_bkg_yield += yield_
            print(f"Yield for {process}: {yield_}")

    print(f"Total background yield: {total_bkg_yield}")

    try:
        data_process = get_data_process(region)
        h_data = data["nMuon"][data_process][{
            "region": region,
            "systematic": "nominal",
        }]
        vals_data_check = hist_values_with_overflow(h_data)
        data_yield = vals_data_check.sum()
        print(f"Data process: {data_process}")
        print(f"Data yield: {data_yield}")

        if total_bkg_yield > 0:
            print(f"Data/MC: {data_yield / total_bkg_yield}")

    except Exception as e:
        print(f"[WARNING] Could not print data yield for {region}: {e}")
try:
    print("Systematics keys in bkg['metpt']['W (lnu)']:")
    print(bkg["metpt"]["W (lnu)"])
except Exception as e:
    print(f"[WARNING] Could not print metpt W(lnu) systematic keys: {e}")

print("\n[CHECK] External systematic axis examples:")

for label, container, syst_name in [
    ("metUnclusteredUp", bkg_metuncl_up, "metUnclusteredUp"),
    ("metUnclusteredDown", bkg_metuncl_down, "metUnclusteredDown"),
    ("jesTotalUp", bkg_jes_up, "jesTotalUp"),
    ("jesTotalDown", bkg_jes_down, "jesTotalDown"),
]:
    try:
        if container is None:
            print(f"  {label}: container missing")
        else:
            print(f"  {label} recoilpt W(lnu):")
            print(container["recoilpt"]["W (lnu)"].axes["systematic"])
    except Exception as e:
        print(f"  Could not print {label} axis: {e}")


# ----------------------------------------------------------------------
# Main plotting loop
# ----------------------------------------------------------------------
for region in regions_to_plot:
    print(f"Plotting for region: {region}")

    for key in bkg.keys():
        if "sumw" in key:
            continue
        elif "template" in key:
            continue
        elif "nPV" in key:
            continue

        plt.style.use(mplhep.style.CMS)

        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9, 9),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.07)

        mplhep.cms.label(
            ax=ax,
            llabel="Work in progress",
            rlabel=LUMI_LABEL.get(YEAR, "13.6 TeV"),
            fontsize=24,
        )

        # --------------------------------------------------------------
        # 1) MC hist collection
        # --------------------------------------------------------------
        mc_vals_list = []
        mc_vars_list = []
        mc_labels = []
        mc_colors = []
        bins = None

        mc_vals_syst_up = {name: [] for name in syst_pairs}
        mc_vals_syst_down = {name: [] for name in syst_pairs}

        for mc in stacks:
            if mc not in bkg[key]:
                continue

            h_nom = get_hist_safe(bkg, key, mc, region, nominal_syst)
            if h_nom is None:
                continue

            edges, _ = get_axis_edges_centers(h_nom)
            if bins is None:
                bins = edges

            vals_nom, vars_nom = hist_values_variances_with_overflow(h_nom)
            vals_nom_1d, vars_nom_1d = reduce_to_1d(vals_nom, vars_nom)

            mc_vals_list.append(vals_nom_1d)
            mc_vars_list.append(vars_nom_1d)
            mc_labels.append(get_pretty_label(mc))
            mc_colors.append(color_map.get(mc, "gray"))

            for syst_name, (syst_up_name, syst_down_name) in syst_pairs.items():
                h_up = get_hist_safe(bkg, key, mc, region, syst_up_name)
                if h_up is not None:
                    vals_up = hist_values_with_overflow(h_up)
                    vals_up_1d, _ = reduce_to_1d(vals_up, None)
                else:
                    vals_up_1d = vals_nom_1d.copy()

                h_down = get_hist_safe(bkg, key, mc, region, syst_down_name)
                if h_down is not None:
                    vals_down = hist_values_with_overflow(h_down)
                    vals_down_1d, _ = reduce_to_1d(vals_down, None)
                else:
                    vals_down_1d = vals_nom_1d.copy()

                mc_vals_syst_up[syst_name].append(vals_up_1d)
                mc_vals_syst_down[syst_name].append(vals_down_1d)

        if len(mc_vals_list) == 0 or bins is None:
            plt.close()
            continue

        mc_vals_arr = np.array(mc_vals_list)
        mc_vars_arr = np.array(mc_vars_list)

        # --------------------------------------------------------------
        # 2) Total background + uncertainties
        # --------------------------------------------------------------
        total_bkg = mc_vals_arr.sum(axis=0)
        total_bkg_var = mc_vars_arr.sum(axis=0)
        total_bkg_err = np.sqrt(total_bkg_var)

        sys_up_sq = np.zeros_like(total_bkg, dtype=float)
        sys_down_sq = np.zeros_like(total_bkg, dtype=float)

        total_bkg_syst_up = {}
        total_bkg_syst_down = {}
        per_syst_bin_up = {}
        per_syst_bin_down = {}

        # --------------------------------------------------------------
        # 2a) Internal systematics
        # --------------------------------------------------------------
        for syst_name in syst_pairs:
            mc_vals_up_arr = np.array(mc_vals_syst_up[syst_name])
            mc_vals_down_arr = np.array(mc_vals_syst_down[syst_name])

            total_up = mc_vals_up_arr.sum(axis=0)
            total_down = mc_vals_down_arr.sum(axis=0)

            total_bkg_syst_up[syst_name] = total_up
            total_bkg_syst_down[syst_name] = total_down

            delta_up_var = total_up - total_bkg
            delta_down_var = total_down - total_bkg

            this_sys_up = np.maximum.reduce([
                delta_up_var,
                delta_down_var,
                np.zeros_like(total_bkg),
            ])

            this_sys_down = np.maximum.reduce([
                -delta_up_var,
                -delta_down_var,
                np.zeros_like(total_bkg),
            ])

            per_syst_bin_up[syst_name] = this_sys_up
            per_syst_bin_down[syst_name] = this_sys_down

            sys_up_sq += this_sys_up**2
            sys_down_sq += this_sys_down**2

        # --------------------------------------------------------------
        # 2b) External systematics from separate files
        # --------------------------------------------------------------
        for syst_name, (
            container_up,
            syst_up_name,
            container_down,
            syst_down_name,
        ) in external_syst_pairs.items():

            total_up, bins_up = get_total_bkg_1d_from_container(
                container_up,
                key,
                region,
                syst_up_name,
                stacks,
            )

            total_down, bins_down = get_total_bkg_1d_from_container(
                container_down,
                key,
                region,
                syst_down_name,
                stacks,
            )

            if total_up is None or not same_binning(bins, bins_up):
                print(
                    f"[WARNING] Missing or mismatched {syst_name} Up for "
                    f"region={region}, key={key}. Falling back to nominal."
                )
                total_up = total_bkg.copy()

            if total_down is None or not same_binning(bins, bins_down):
                print(
                    f"[WARNING] Missing or mismatched {syst_name} Down for "
                    f"region={region}, key={key}. Falling back to nominal."
                )
                total_down = total_bkg.copy()

            if total_up.shape != total_bkg.shape:
                print(
                    f"[WARNING] Skip {syst_name} Up for "
                    f"region={region}, key={key}: shape mismatch "
                    f"{total_up.shape} vs {total_bkg.shape}"
                )
                total_up = total_bkg.copy()

            if total_down.shape != total_bkg.shape:
                print(
                    f"[WARNING] Skip {syst_name} Down for "
                    f"region={region}, key={key}: shape mismatch "
                    f"{total_down.shape} vs {total_bkg.shape}"
                )
                total_down = total_bkg.copy()

            total_bkg_syst_up[syst_name] = total_up
            total_bkg_syst_down[syst_name] = total_down

            delta_up_var = total_up - total_bkg
            delta_down_var = total_down - total_bkg

            this_sys_up = np.maximum.reduce([
                delta_up_var,
                delta_down_var,
                np.zeros_like(total_bkg),
            ])

            this_sys_down = np.maximum.reduce([
                -delta_up_var,
                -delta_down_var,
                np.zeros_like(total_bkg),
            ])

            per_syst_bin_up[syst_name] = this_sys_up
            per_syst_bin_down[syst_name] = this_sys_down

            sys_up_sq += this_sys_up**2
            sys_down_sq += this_sys_down**2

        sys_up = np.sqrt(sys_up_sq)
        sys_down = np.sqrt(sys_down_sq)

        lumi_err = total_bkg * lumi_unc

        tot_err_up = np.sqrt(total_bkg_err**2 + sys_up**2 + lumi_err**2)
        tot_err_down = np.sqrt(total_bkg_err**2 + sys_down**2 + lumi_err**2)

        unc_low = total_bkg - tot_err_down
        unc_up = total_bkg + tot_err_up
        unc_low = np.maximum(unc_low, 1e-10)

        total_bkg_forratio = np.append(total_bkg, total_bkg[-1])
        unc_low_ratio = np.append(unc_low, unc_low[-1])
        unc_up_ratio = np.append(unc_up, unc_up[-1])

        # --------------------------------------------------------------
        # Debug print
        # --------------------------------------------------------------
        if key == "recoilpt":
            print(f"\n[DEBUG] region={region}, variable={key}")

            nom_yield = total_bkg.sum()
            print(f"  Total yield nominal = {nom_yield:.6f}")

            for syst_name in all_syst_names:
                up_yield = total_bkg_syst_up[syst_name].sum()
                down_yield = total_bkg_syst_down[syst_name].sum()

                rel_up_yield = (
                    100.0 * (up_yield - nom_yield) / nom_yield
                    if nom_yield != 0 else 0.0
                )
                rel_down_yield = (
                    100.0 * (down_yield - nom_yield) / nom_yield
                    if nom_yield != 0 else 0.0
                )

                print(f"  {syst_name:28s} Up   = {up_yield:.6f}  ({rel_up_yield:+.3f}%)")
                print(f"  {syst_name:28s} Down = {down_yield:.6f}  ({rel_down_yield:+.3f}%)")

                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_diff_up = np.divide(
                        total_bkg_syst_up[syst_name] - total_bkg,
                        total_bkg,
                        out=np.zeros_like(total_bkg, dtype=float),
                        where=(total_bkg != 0),
                    ) * 100.0

                    rel_diff_down = np.divide(
                        total_bkg_syst_down[syst_name] - total_bkg,
                        total_bkg,
                        out=np.zeros_like(total_bkg, dtype=float),
                        where=(total_bkg != 0),
                    ) * 100.0

                max_abs_up = np.max(np.abs(total_bkg_syst_up[syst_name] - total_bkg))
                max_abs_down = np.max(np.abs(total_bkg_syst_down[syst_name] - total_bkg))
                max_rel_up = np.max(np.abs(rel_diff_up))
                max_rel_down = np.max(np.abs(rel_diff_down))

                print(f"    Max |Up - Nominal| per bin              = {max_abs_up:.6f}")
                print(f"    Max |Down - Nominal| per bin            = {max_abs_down:.6f}")
                print(f"    Max |Up - Nominal| / Nominal per bin    = {max_rel_up:.3f}%")
                print(f"    Max |Down - Nominal| / Nominal per bin  = {max_rel_down:.3f}%")

            print(f"  Lumi uncertainty = {100.0 * lumi_unc:.3f}%")

            yield_sys_up_sq = 0.0
            yield_sys_down_sq = 0.0

            for syst_name in all_syst_names:
                up_yield = total_bkg_syst_up[syst_name].sum()
                down_yield = total_bkg_syst_down[syst_name].sum()

                delta_up = up_yield - nom_yield
                delta_down = down_yield - nom_yield

                this_up = max(delta_up, delta_down, 0.0)
                this_down = max(-delta_up, -delta_down, 0.0)

                yield_sys_up_sq += this_up**2
                yield_sys_down_sq += this_down**2

            stat_yield_err = np.sqrt(np.sum(total_bkg_var))
            lumi_yield_err = nom_yield * lumi_unc

            total_yield_err_up = np.sqrt(
                stat_yield_err**2 + yield_sys_up_sq + lumi_yield_err**2
            )
            total_yield_err_down = np.sqrt(
                stat_yield_err**2 + yield_sys_down_sq + lumi_yield_err**2
            )

            rel_total_yield_err_up = (
                100.0 * total_yield_err_up / nom_yield
                if nom_yield != 0 else 0.0
            )
            rel_total_yield_err_down = (
                100.0 * total_yield_err_down / nom_yield
                if nom_yield != 0 else 0.0
            )

            print(
                f"  Total yield uncertainty       = "
                f"+{total_yield_err_up:.6f} ({rel_total_yield_err_up:.3f}%), "
                f"-{total_yield_err_down:.6f} ({rel_total_yield_err_down:.3f}%)"
            )

            total_err_up_yield = np.sqrt(np.sum(tot_err_up**2))
            total_err_down_yield = np.sqrt(np.sum(tot_err_down**2))

            rel_total_err_up_yield = (
                100.0 * total_err_up_yield / nom_yield
                if nom_yield != 0 else 0.0
            )
            rel_total_err_down_yield = (
                100.0 * total_err_down_yield / nom_yield
                if nom_yield != 0 else 0.0
            )

            print(
                f"  Band-like combined uncertainty= "
                f"+{total_err_up_yield:.6f} ({rel_total_err_up_yield:.3f}%), "
                f"-{total_err_down_yield:.6f} ({rel_total_err_down_yield:.3f}%)"
            )

        # --------------------------------------------------------------
        # 3) Stacked MC
        # --------------------------------------------------------------
        bottom = np.zeros_like(bins[:-1], dtype=float)
        stack_tops = []

        for vals_1d, lab, col in zip(mc_vals_list, mc_labels, mc_colors):
            top = bottom + vals_1d

            ax.bar(
                bins[:-1],
                vals_1d,
                width=np.diff(bins),
                bottom=bottom,
                align="edge",
                label=lab,
                color=col,
                edgecolor="none",
                linewidth=0,
            )

            stack_tops.append(top.copy())
            bottom = top

        for top in stack_tops:
            y = np.append(top, top[-1])
            ax.step(
                bins,
                y,
                where="post",
                color="black",
                linewidth=0.8,
            )

        # --------------------------------------------------------------
        # 4) Total uncertainty band
        # --------------------------------------------------------------
        ax.fill_between(bins, unc_low_ratio, unc_up_ratio, **error_opts)

        # --------------------------------------------------------------
        # 5) Signal draw
        # --------------------------------------------------------------
        if "CR" not in region and "DY" not in region:
            for i, signal in enumerate(signals):
                signal_container = sig

                if key not in signal_container:
                    continue
                if signal not in signal_container[key]:
                    continue

                h_sig = get_hist_safe(signal_container, key, signal, region, nominal_syst)
                if h_sig is None:
                    continue

                vals_sig = hist_values_with_overflow(h_sig)
                vals_sig_1d, _ = reduce_to_1d(vals_sig, None)
                sig_bins, _ = get_axis_edges_centers(h_sig)

                ax.step(
                    sig_bins,
                    np.append(vals_sig_1d, vals_sig_1d[-1]),
                    where="post",
                    label=signal_labels[i],
                    color=signal_colors[i],
                    linestyle=signal_ltypes[i],
                    linewidth=2.5,
                )

        # --------------------------------------------------------------
        # 6) Data
        # --------------------------------------------------------------
        data_process = get_data_process(region)

        try:
            h_data = data[key][data_process][{
                "region": region,
                "systematic": nominal_syst,
            }]
            h_data = get_rebinned_hist(h_data, key)
        except Exception:
            plt.close()
            continue

        bins_data, centers = get_axis_edges_centers(h_data)

        vals_data = hist_values_with_overflow(h_data)
        vals_data_1d, _ = reduce_to_1d(vals_data, None)

        bins = bins_data
        yerr_data = np.sqrt(vals_data_1d)

        if "SR" not in region:
            ax.errorbar(
                centers,
                vals_data_1d,
                xerr=np.diff(bins) / 2,
                yerr=yerr_data,
                fmt="o",
                label="Data",
                markersize=8,
                capsize=5,
                capthick=1,
                color="black",
            )

        ax.set_ylabel("Events")
        ax.legend(ncol=3, fontsize=14, loc="upper right")
        ax.set_yscale("log")
        ax.set_yticks([
            1,
            10,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6,
            1e7,
            1e8,
            1e9,
            1e10,
        ])

        if "CR" in region or "DY" in region:
            ax.set_ylim(0.1, 1e7)
            ax.set_yticks([0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
        elif "SR" in region:
            ax.set_ylim(0.1, 1e8)
        else:
            ax.set_ylim(1, 1e10)

        if "metpt" in key:
            ax.set_xlim(250, 800)
        if "recoilpt" in key:
            ax.set_xlim(250, 800)
        if "eta" in key:
            ax.set_xlim(-3.0, 3.0)
        if "phi" in key:
            ax.set_xlim(-3.2, 3.2)
        if "nElectron" in key or "nMuon" in key:
            ax.set_xlim(0, 4)
        if "nJet" in key:
            ax.set_xlim(0, 10)
        if "nfj" in key:
            ax.set_xlim(0, 6)
        if "vs" in key:
            ax.set_xlim(0, 1)
        if key == "fj1pt":
            ax.set_xlim(200, 1000)
        if "mll" in key:
            ax.set_xlim(50, 250)
        if "pll" in key:
            ax.set_xlim(200, 1000)

        ax.grid(True, which="both", axis="y", ls="--", lw=0.5)

        # --------------------------------------------------------------
        # 7) Ratio plot
        # --------------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                vals_data_1d,
                total_bkg,
                out=np.zeros_like(vals_data_1d, dtype=float),
                where=(total_bkg != 0),
            )
            ratio_err = np.divide(
                yerr_data,
                total_bkg,
                out=np.zeros_like(yerr_data, dtype=float),
                where=(total_bkg != 0),
            )
            ratio_err = np.where(ratio_err < 0, 0, ratio_err)

        if "SR" not in region:
            rax.errorbar(
                centers,
                ratio,
                xerr=np.diff(bins) / 2,
                yerr=ratio_err,
                fmt="o",
                color="black",
                markersize=8,
                capsize=5,
                capthick=1,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            band_low = np.divide(
                unc_low_ratio,
                total_bkg_forratio,
                out=np.ones_like(unc_low_ratio, dtype=float),
                where=(total_bkg_forratio != 0),
            )
            band_up = np.divide(
                unc_up_ratio,
                total_bkg_forratio,
                out=np.ones_like(unc_up_ratio, dtype=float),
                where=(total_bkg_forratio != 0),
            )

        rax.fill_between(
            bins,
            band_low,
            band_up,
            **error_opts,
        )

        rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        rax.set_xlabel(key)

        if "metpt" in key:
            rax.set_xlabel(r"$E\!\!\!/_{T}\ \mathrm{(GeV)}$")
        elif "recoilpt" in key:
            rax.set_xlabel(r"$U\!\!\!/_{T}\ \mathrm{(GeV)}$")
        elif "ht" in key and "tight" not in key:
            rax.set_xlabel("H$_{T}$ (GeV)")
            ax.set_xlim(300, 1500)
            rax.set_xlim(300, 1500)
        elif "nJet" in key:
            rax.set_xlabel("Number of Jets")
            ax.set_xlim(2, 10)
            rax.set_xlim(2, 10)
        elif "nElectron" in key:
            rax.set_xlabel("Number of Electrons")
        elif "nMuon" in key:
            rax.set_xlabel("Number of Muons")
        elif "nb" in key:
            rax.set_xlabel("Number of b jets")
            ax.set_xlim(0, 5)
            rax.set_xlim(0, 5)
        elif key == "j1pt":
            rax.set_xlabel("Leading jet $p_{T}$ [GeV]")
            ax.set_xlim(100, 600)
            rax.set_xlim(100, 600)
        elif key == "j2pt":
            rax.set_xlabel("Subleading jet $p_{T}$ [GeV]")
            ax.set_xlim(50, 600)
            rax.set_xlim(50, 600)

        xmin, xmax = ax.get_xlim()
        rax.set_xlim(xmin, xmax)

        rax.set_ylabel("Data/MC")
        rax.set_ylim(0, 2)
        rax.set_yticks([0, 0.5, 1, 1.5, 2])
        rax.grid(True, which="both", axis="y", ls="--", lw=0.5)

        plt.savefig(f"{PLOT_DIR}/{key}_{region}.png")
        plt.close()

print("Done.")