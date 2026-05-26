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
# Options
# ----------------------------------------------------------------------
INCLUDE_OVERFLOW_IN_LAST = True
INCLUDE_UNDERFLOW_IN_FIRST = False

DEFAULT_YEARS = ["2024", "2025"]

# Nominal scaled files
NOMINAL_PATTERN = "hists/stop_{year}_nominal.scaled"

# External systematic scaled files
EXTERNAL_FILE_PATTERNS = {
    "metUnclustered": {
        "up": "hists/stop_{year}_metUnclusteredUp.scaled",
        "down": "hists/stop_{year}_metUnclusteredDown.scaled",
        "up_syst": "metUnclusteredUp",
        "down_syst": "metUnclusteredDown",
    },
    "jesTotal": {
        "up": "hists/stop_{year}_jesTotalUp.scaled",
        "down": "hists/stop_{year}_jesTotalDown.scaled",
        "up_syst": "jesTotalUp",
        "down_syst": "jesTotalDown",
    },
}

# Per-year lumi uncertainty.
# 지금은 둘 다 1.6%로 둠. 필요하면 2025 값만 바꾸면 됨.
YEAR_LUMI_UNC = {
    "2024": 0.016,
    "2025": 0.016,
}

YEAR_LUMI_LABEL = {
    "2024": "108.95",
    "2025": "110.58",
}

error_opts = {
    "step": "post",
    "label": "Stat. ⊕ syst. unc",
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
        if container is None:
            return None
        if key not in container:
            return None
        if process not in container[key]:
            return None

        h = container[key][process][{"region": region, "systematic": systematic}]
        h = get_rebinned_hist(h, key)
        return h
    except Exception:
        return None


def get_axis_edges_centers(h):
    edges = h.axes[-1].edges
    centers = h.axes[-1].centers
    return edges, centers


def same_binning(bins_a, bins_b):
    if bins_a is None or bins_b is None:
        return False
    if len(bins_a) != len(bins_b):
        return False
    return np.allclose(bins_a, bins_b, rtol=0, atol=1e-9)


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


def add_array(acc, arr):
    if arr is None:
        return acc
    arr = np.asarray(arr, dtype=float)
    if acc is None:
        return arr.copy()
    return acc + arr


def add_variance(acc, arr):
    if arr is None:
        return acc
    arr = np.asarray(arr, dtype=float)
    if acc is None:
        return arr.copy()
    return acc + arr


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


def load_scaled_file(path, label, required=False):
    if not os.path.exists(path):
        msg = f"[WARNING] Missing {label} file: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    print(f"[INFO] Loading {label}: {path}")
    return load(path)


def load_nominal_years(years):
    payload = {}

    for year in years:
        path = NOMINAL_PATTERN.format(year=year)
        obj = load_scaled_file(path, f"{year} nominal", required=False)

        if obj is None:
            print(f"[WARNING] Skip year={year} because nominal file is missing.")
            continue

        payload[year] = {
            "nominal": obj,
            "bkg": obj.get("bkg", None),
            "data": obj.get("data", None),
            "sig": obj.get("sig", None),
            "external": {},
        }

    if len(payload) == 0:
        raise RuntimeError("No nominal hist files were loaded.")

    return payload


def load_external_years(year_payload):
    for year, yp in year_payload.items():
        yp["external"] = {}

        for syst_name, cfg in EXTERNAL_FILE_PATTERNS.items():
            up_path = cfg["up"].format(year=year)
            down_path = cfg["down"].format(year=year)

            up_obj = load_scaled_file(up_path, f"{year} {syst_name}Up", required=False)
            down_obj = load_scaled_file(down_path, f"{year} {syst_name}Down", required=False)

            yp["external"][syst_name] = {
                "up_bkg": up_obj.get("bkg", None) if up_obj is not None else None,
                "down_bkg": down_obj.get("bkg", None) if down_obj is not None else None,
                "up_sig": up_obj.get("sig", None) if up_obj is not None else None,
                "down_sig": down_obj.get("sig", None) if down_obj is not None else None,
                "up_syst": cfg["up_syst"],
                "down_syst": cfg["down_syst"],
                "has_up": up_obj is not None,
                "has_down": down_obj is not None,
            }


def get_all_plot_keys(year_payload):
    keys = set()

    for yp in year_payload.values():
        bkg = yp["bkg"]
        if bkg is None:
            continue
        for key in bkg.keys():
            if key == "sumw":
                continue
            if "template" in key:
                continue
            if "nPV" in key:
                continue
            keys.add(key)

    return sorted(keys)


def get_combined_data_1d(year_payload, key, region):
    data_process = get_data_process(region)

    bins_ref = None
    vals_sum = None

    for year, yp in year_payload.items():
        h = get_hist_safe(yp["data"], key, data_process, region, "nominal")
        if h is None:
            continue

        bins, _ = get_axis_edges_centers(h)
        if bins_ref is None:
            bins_ref = bins
        elif not same_binning(bins_ref, bins):
            print(
                f"[WARNING] Data binning mismatch for year={year}, "
                f"region={region}, key={key}. Skip this year."
            )
            continue

        vals = hist_values_with_overflow(h)
        vals_1d, _ = reduce_to_1d(vals, None)
        vals_sum = add_array(vals_sum, vals_1d)

    return vals_sum, bins_ref


def get_combined_signal_1d(year_payload, key, signal, region):
    bins_ref = None
    vals_sum = None

    for year, yp in year_payload.items():
        h = get_hist_safe(yp["bkg"], key, signal, region, "nominal")
        if h is None:
            continue

        bins, _ = get_axis_edges_centers(h)
        if bins_ref is None:
            bins_ref = bins
        elif not same_binning(bins_ref, bins):
            print(
                f"[WARNING] Signal binning mismatch for year={year}, "
                f"signal={signal}, region={region}, key={key}. Skip this year."
            )
            continue

        vals = hist_values_with_overflow(h)
        vals_1d, _ = reduce_to_1d(vals, None)
        vals_sum = add_array(vals_sum, vals_1d)

    return vals_sum, bins_ref


def get_combined_process_nominal(year_payload, key, process, region):
    bins_ref = None
    vals_sum = None
    vars_sum = None
    per_year_nominal = {}

    for year, yp in year_payload.items():
        h = get_hist_safe(yp["bkg"], key, process, region, "nominal")
        if h is None:
            continue

        bins, _ = get_axis_edges_centers(h)
        if bins_ref is None:
            bins_ref = bins
        elif not same_binning(bins_ref, bins):
            print(
                f"[WARNING] Nominal binning mismatch for year={year}, "
                f"process={process}, region={region}, key={key}. Skip this year."
            )
            continue

        vals, vars_ = hist_values_variances_with_overflow(h)
        vals_1d, vars_1d = reduce_to_1d(vals, vars_)

        if vars_1d is None:
            vars_1d = np.zeros_like(vals_1d, dtype=float)

        vals_sum = add_array(vals_sum, vals_1d)
        vars_sum = add_variance(vars_sum, vars_1d)

        per_year_nominal[year] = {
            "bins": bins,
            "vals": vals_1d,
            "vars": vars_1d,
        }

    return vals_sum, vars_sum, bins_ref, per_year_nominal


def get_combined_process_internal_syst(
    year_payload,
    key,
    process,
    region,
    syst_up_name,
    syst_down_name,
    per_year_nominal,
):
    vals_up_sum = None
    vals_down_sum = None

    for year, nominal_info in per_year_nominal.items():
        yp = year_payload[year]
        nominal_vals = nominal_info["vals"]
        nominal_bins = nominal_info["bins"]

        h_up = get_hist_safe(yp["bkg"], key, process, region, syst_up_name)
        if h_up is not None:
            bins_up, _ = get_axis_edges_centers(h_up)
            if same_binning(nominal_bins, bins_up):
                vals_up = hist_values_with_overflow(h_up)
                vals_up, _ = reduce_to_1d(vals_up, None)
            else:
                print(
                    f"[WARNING] Binning mismatch for {syst_up_name}: "
                    f"year={year}, process={process}, region={region}, key={key}. "
                    f"Use nominal."
                )
                vals_up = nominal_vals.copy()
        else:
            vals_up = nominal_vals.copy()

        h_down = get_hist_safe(yp["bkg"], key, process, region, syst_down_name)
        if h_down is not None:
            bins_down, _ = get_axis_edges_centers(h_down)
            if same_binning(nominal_bins, bins_down):
                vals_down = hist_values_with_overflow(h_down)
                vals_down, _ = reduce_to_1d(vals_down, None)
            else:
                print(
                    f"[WARNING] Binning mismatch for {syst_down_name}: "
                    f"year={year}, process={process}, region={region}, key={key}. "
                    f"Use nominal."
                )
                vals_down = nominal_vals.copy()
        else:
            vals_down = nominal_vals.copy()

        vals_up_sum = add_array(vals_up_sum, vals_up)
        vals_down_sum = add_array(vals_down_sum, vals_down)

    return vals_up_sum, vals_down_sum


def get_year_total_nominal_by_bkg(year_payload, key, region, stacks):
    """
    Lumi uncertainty와 external systematic fallback 계산용.
    year별 total background nominal을 저장한다.
    """
    out = {}

    for year, yp in year_payload.items():
        bins_ref = None
        vals_sum = None

        for process in stacks:
            h = get_hist_safe(yp["bkg"], key, process, region, "nominal")
            if h is None:
                continue

            bins, _ = get_axis_edges_centers(h)
            if bins_ref is None:
                bins_ref = bins
            elif not same_binning(bins_ref, bins):
                print(
                    f"[WARNING] Year-total nominal binning mismatch: "
                    f"year={year}, process={process}, region={region}, key={key}. Skip."
                )
                continue

            vals = hist_values_with_overflow(h)
            vals_1d, _ = reduce_to_1d(vals, None)
            vals_sum = add_array(vals_sum, vals_1d)

        if vals_sum is not None:
            out[year] = {
                "bins": bins_ref,
                "vals": vals_sum,
            }

    return out


def get_combined_external_total(
    year_payload,
    key,
    region,
    stacks,
    syst_name,
    direction,
    year_total_nominal,
):
    """
    direction: "up" or "down"

    External systematic 파일이 없는 year는 nominal fallback.
    즉 그 year는 해당 systematic uncertainty에 기여하지 않음.
    """
    vals_total = None
    bins_ref = None

    for year, yp in year_payload.items():
        if year not in year_total_nominal:
            continue

        nominal_vals = year_total_nominal[year]["vals"]
        nominal_bins = year_total_nominal[year]["bins"]

        ext = yp["external"].get(syst_name, None)
        if ext is None:
            year_vals = nominal_vals.copy()
            year_bins = nominal_bins
        else:
            if direction == "up":
                container = ext["up_bkg"]
                syst_axis_name = ext["up_syst"]
                has_file = ext["has_up"]
            else:
                container = ext["down_bkg"]
                syst_axis_name = ext["down_syst"]
                has_file = ext["has_down"]

            if not has_file or container is None:
                year_vals = nominal_vals.copy()
                year_bins = nominal_bins
            else:
                year_vals = None
                year_bins = None

                for process in stacks:
                    h = get_hist_safe(container, key, process, region, syst_axis_name)
                    if h is None:
                        # process별 missing은 nominal process로 보완
                        h_nom = get_hist_safe(yp["bkg"], key, process, region, "nominal")
                        h = h_nom

                    if h is None:
                        continue

                    bins, _ = get_axis_edges_centers(h)
                    if year_bins is None:
                        year_bins = bins
                    elif not same_binning(year_bins, bins):
                        print(
                            f"[WARNING] External {syst_name} {direction} binning mismatch: "
                            f"year={year}, process={process}, region={region}, key={key}. Skip process."
                        )
                        continue

                    vals = hist_values_with_overflow(h)
                    vals_1d, _ = reduce_to_1d(vals, None)
                    year_vals = add_array(year_vals, vals_1d)

                if year_vals is None or not same_binning(year_bins, nominal_bins):
                    print(
                        f"[WARNING] External {syst_name} {direction} unusable for "
                        f"year={year}, region={region}, key={key}. Use nominal."
                    )
                    year_vals = nominal_vals.copy()
                    year_bins = nominal_bins

        if bins_ref is None:
            bins_ref = year_bins
        elif not same_binning(bins_ref, year_bins):
            print(
                f"[WARNING] Combined external binning mismatch for {syst_name} {direction}: "
                f"year={year}, region={region}, key={key}. Skip year."
            )
            continue

        vals_total = add_array(vals_total, year_vals)

    return vals_total, bins_ref


def make_lumi_error(year_total_nominal, total_bkg):
    lumi_err_sq = np.zeros_like(total_bkg, dtype=float)

    for year, info in year_total_nominal.items():
        unc = YEAR_LUMI_UNC.get(year, 0.016)
        lumi_err_sq += (info["vals"] * unc) ** 2

    return np.sqrt(lumi_err_sq)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years to combine, e.g. --years 2024 2025",
    )
    parser.add_argument(
        "--outdir",
        default="plots_2024_2025",
        help="Output directory",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Right CMS label. If not given, uses sum of configured lumi labels.",
    )
    args = parser.parse_args()

    years = args.years
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load histograms
    # ------------------------------------------------------------------
    year_payload = load_nominal_years(years)
    load_external_years(year_payload)

    active_years = list(year_payload.keys())
    print(f"[INFO] Active years: {active_years}")

    if args.label is not None:
        cms_rlabel = args.label
    else:
        lumi_sum = 0.0
        for y in active_years:
            try:
                lumi_sum += float(YEAR_LUMI_LABEL.get(y, "0"))
            except Exception:
                pass
        cms_rlabel = f"{lumi_sum:.2f} fb$^{{-1}}$ (13.6 TeV)"

    # ------------------------------------------------------------------
    # Plot config
    # ------------------------------------------------------------------
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

    external_syst_names = list(EXTERNAL_FILE_PATTERNS.keys())
    all_syst_names = list(syst_pairs.keys()) + external_syst_names

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
        #"cat1_preselection",
        #"cat2_LLCR_highDeltaM",
        #"cat3_QCDCR_highDeltaM",
        #"cat4_GCR_highDeltaM",
        #"cat5_DY2E_highDeltaM",
        #"cat6_DY2M_highDeltaM",
        "cat7_SR_highDeltaM",
    ]

    keys_to_plot = get_all_plot_keys(year_payload)

    # ------------------------------------------------------------------
    # Print quick yield check
    # ------------------------------------------------------------------
    total_bkg_yield = 0.0
    for process in stacks:
        process_yield = 0.0

        for year, yp in year_payload.items():
            h = get_hist_safe(
                yp["bkg"],
                "nMuon",
                process,
                "cat1_preselection",
                nominal_syst,
            )
            if h is None:
                continue

            vals, _ = hist_values_variances_with_overflow(h)
            process_yield += vals.sum()

        total_bkg_yield += process_yield
        print(f"Yield for {process}: {process_yield}")

    print(f"Total background yield: {total_bkg_yield}")

    data_yield = 0.0
    data_vals, _ = get_combined_data_1d(year_payload, "nMuon", "cat1_preselection")
    if data_vals is not None:
        data_yield = data_vals.sum()
    print(f"Data yield: {data_yield}")

    # ------------------------------------------------------------------
    # Main plotting loop
    # ------------------------------------------------------------------
    for region in regions_to_plot:
        print(f"Plotting for region: {region}")

        for key in keys_to_plot:
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
                rlabel=cms_rlabel,
                fontsize=24,
            )

            # ----------------------------------------------------------
            # 1) MC hist collection, combined over years
            # ----------------------------------------------------------
            mc_vals_list = []
            mc_vars_list = []
            mc_labels = []
            mc_colors = []
            bins = None

            mc_vals_syst_up = {name: [] for name in syst_pairs}
            mc_vals_syst_down = {name: [] for name in syst_pairs}

            for mc in stacks:
                vals_nom, vars_nom, bins_proc, per_year_nominal = get_combined_process_nominal(
                    year_payload,
                    key,
                    mc,
                    region,
                )

                if vals_nom is None or bins_proc is None:
                    continue

                if bins is None:
                    bins = bins_proc
                elif not same_binning(bins, bins_proc):
                    print(
                        f"[WARNING] Combined nominal binning mismatch: "
                        f"process={mc}, region={region}, key={key}. Skip process."
                    )
                    continue

                if vars_nom is None:
                    vars_nom = np.zeros_like(vals_nom, dtype=float)

                mc_vals_list.append(vals_nom)
                mc_vars_list.append(vars_nom)
                mc_labels.append(get_pretty_label(mc))
                mc_colors.append(color_map.get(mc, "gray"))

                for syst_name, (syst_up_name, syst_down_name) in syst_pairs.items():
                    vals_up, vals_down = get_combined_process_internal_syst(
                        year_payload,
                        key,
                        mc,
                        region,
                        syst_up_name,
                        syst_down_name,
                        per_year_nominal,
                    )

                    if vals_up is None:
                        vals_up = vals_nom.copy()
                    if vals_down is None:
                        vals_down = vals_nom.copy()

                    mc_vals_syst_up[syst_name].append(vals_up)
                    mc_vals_syst_down[syst_name].append(vals_down)

            if len(mc_vals_list) == 0 or bins is None:
                plt.close()
                continue

            mc_vals_arr = np.array(mc_vals_list)
            mc_vars_arr = np.array(mc_vars_list)

            # ----------------------------------------------------------
            # 2) Total background + uncertainties
            # ----------------------------------------------------------
            total_bkg = mc_vals_arr.sum(axis=0)
            total_bkg_var = mc_vars_arr.sum(axis=0)
            total_bkg_err = np.sqrt(total_bkg_var)

            year_total_nominal = get_year_total_nominal_by_bkg(
                year_payload,
                key,
                region,
                stacks,
            )

            sys_up_sq = np.zeros_like(total_bkg, dtype=float)
            sys_down_sq = np.zeros_like(total_bkg, dtype=float)

            total_bkg_syst_up = {}
            total_bkg_syst_down = {}
            per_syst_bin_up = {}
            per_syst_bin_down = {}

            # ----------------------------------------------------------
            # 2a) Internal systematics
            # ----------------------------------------------------------
            for syst_name in syst_pairs:
                mc_vals_up_arr = np.array(mc_vals_syst_up[syst_name])
                mc_vals_down_arr = np.array(mc_vals_syst_down[syst_name])

                if mc_vals_up_arr.size == 0 or mc_vals_down_arr.size == 0:
                    total_up = total_bkg.copy()
                    total_down = total_bkg.copy()
                else:
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

            # ----------------------------------------------------------
            # 2b) External systematics from separate files
            #     Missing file -> nominal fallback for that year
            # ----------------------------------------------------------
            for syst_name in external_syst_names:
                total_up, bins_up = get_combined_external_total(
                    year_payload,
                    key,
                    region,
                    stacks,
                    syst_name,
                    "up",
                    year_total_nominal,
                )

                total_down, bins_down = get_combined_external_total(
                    year_payload,
                    key,
                    region,
                    stacks,
                    syst_name,
                    "down",
                    year_total_nominal,
                )

                if total_up is None or not same_binning(bins, bins_up):
                    total_up = total_bkg.copy()

                if total_down is None or not same_binning(bins, bins_down):
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

            lumi_err = make_lumi_error(year_total_nominal, total_bkg)

            tot_err_up = np.sqrt(total_bkg_err**2 + sys_up**2 + lumi_err**2)
            tot_err_down = np.sqrt(total_bkg_err**2 + sys_down**2 + lumi_err**2)

            unc_low = total_bkg - tot_err_down
            unc_up = total_bkg + tot_err_up
            unc_low = np.maximum(unc_low, 1e-10)

            total_bkg_forratio = np.append(total_bkg, total_bkg[-1])
            unc_low_ratio = np.append(unc_low, unc_low[-1])
            unc_up_ratio = np.append(unc_up, unc_up[-1])

            # ----------------------------------------------------------
            # Debug print
            # ----------------------------------------------------------
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

                print("  Lumi uncertainties by year:")
                for y in active_years:
                    print(f"    {y}: {100.0 * YEAR_LUMI_UNC.get(y, 0.016):.3f}%")

            # ----------------------------------------------------------
            # 3) Stacked MC
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # 4) Total uncertainty band
            # ----------------------------------------------------------
            ax.fill_between(bins, unc_low_ratio, unc_up_ratio, **error_opts)

            # ----------------------------------------------------------
            # 5) Signal draw
            # ----------------------------------------------------------
            if "CR" not in region and "DY" not in region:
                for i, signal in enumerate(signals):
                    vals_sig_1d, sig_bins = get_combined_signal_1d(
                        year_payload,
                        key,
                        signal,
                        region,
                    )

                    if vals_sig_1d is None or sig_bins is None:
                        continue

                    ax.step(
                        sig_bins,
                        np.append(vals_sig_1d, vals_sig_1d[-1]),
                        where="post",
                        label=signal_labels[i],
                        color=signal_colors[i],
                        linestyle=signal_ltypes[i],
                        linewidth=2.5,
                    )

            # ----------------------------------------------------------
            # 6) Data
            # ----------------------------------------------------------
            vals_data_1d, bins_data = get_combined_data_1d(year_payload, key, region)

            if vals_data_1d is None or bins_data is None:
                plt.close()
                continue

            if not same_binning(bins, bins_data):
                print(
                    f"[WARNING] Data/MC binning mismatch for region={region}, key={key}. "
                    f"Skip plot."
                )
                plt.close()
                continue

            centers = 0.5 * (bins_data[:-1] + bins_data[1:])
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

            # ----------------------------------------------------------
            # 7) Ratio plot
            # ----------------------------------------------------------
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

            outname = f"{key}_{region}_combined_{'_'.join(active_years)}.png"
            plt.savefig(os.path.join(outdir, outname))
            plt.close()

    print("Done.")


if __name__ == "__main__":
    main()