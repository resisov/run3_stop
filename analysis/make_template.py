#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import argparse
import numpy as np
import uproot
import hist
from coffea.util import load

# ============================================================
# Config
# ============================================================
INPUT_NOMINAL = "hists/stop_2024_nominal.scaled"

INPUT_METUNCL_UP = "hists/stop_2024_metUnclusteredUp.scaled"
INPUT_METUNCL_DOWN = "hists/stop_2024_metUnclusteredDown.scaled"

INPUT_JES_UP = "hists/stop_2024_jesTotalUp.scaled"
INPUT_JES_DOWN = "hists/stop_2024_jesTotalDown.scaled"

INPUT_JER_UP = "hists/stop_2024_jerUp.scaled"
INPUT_JER_DOWN = "hists/stop_2024_jerDown.scaled"

OUTPUT = "templates_metpt.root"

VARIABLE = "recoilpt"

INCLUDE_OVERFLOW_IN_LAST = True
INCLUDE_UNDERFLOW_IN_FIRST = False

REGIONS = [
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
    "cat7_SR_highDeltaM",
]

BKG_PROCESSES = [
    "VV",
    "Single Top",
    "TT",
    "DY",
    "Gamma + Jets",
    "W (lnu)",
    "Z (inv)",
    "QCD Multijet",
]

SIG_PROCESSES = [
    "SMS-2Stop-Par-mStop-1000",
    "SMS-2Stop-Par-mStop-1500",
    "SMS-2Stop-Par-mStop-600",
]

# Systematics stored inside stop_2024_nominal.scaled
SHAPE_SYSTS = {
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

# External systematics stored in separate .scaled files.
# The actual systematic-axis names inside these files must match up_syst/down_syst.
EXTERNAL_SHAPE_SYSTS = {
    "metUnclustered": {
        "up_file": INPUT_METUNCL_UP,
        "up_syst": "metUnclusteredUp",
        "down_file": INPUT_METUNCL_DOWN,
        "down_syst": "metUnclusteredDown",
    },
    "jesTotal": {
        "up_file": INPUT_JES_UP,
        "up_syst": "jesTotalUp",
        "down_file": INPUT_JES_DOWN,
        "down_syst": "jesTotalDown",
    },
    "jer": {
        "up_file": INPUT_JER_UP,
        "up_syst": "jerUp",
        "down_file": INPUT_JER_DOWN,
        "down_syst": "jerDown",
    },
}

PROCESS_NAME_MAP = {
    "VV": "VV",
    "Single Top": "SingleTop",
    "TT": "TT",
    "DY": "DY",
    "Gamma + Jets": "GammaJets",
    "W (lnu)": "Wlnu",
    "Z (inv)": "Zinv",
    "QCD Multijet": "QCD",
    "SMS-2Stop-Par-mStop-1000": "SMS_2Stop_mStop1000",
    "SMS-2Stop-Par-mStop-1500": "SMS_2Stop_mStop1500",
    "SMS-2Stop-Par-mStop-600": "SMS_2Stop_mStop600",
}


# ============================================================
# Helpers
# ============================================================
def get_data_process(region):
    if "GCR" in region or "DY2E" in region:
        return "EGamma"
    elif "DY2M" in region:
        return "Muon"
    else:
        return "JetMET"


def sanitize_name(name):
    if name in PROCESS_NAME_MAP:
        return PROCESS_NAME_MAP[name]
    return (
        name.replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "")
            .replace("-", "_")
            .replace("/", "_")
    )


def get_hist_safe(container, variable, process, region, systematic):
    try:
        return container[variable][process][{"region": region, "systematic": systematic}]
    except Exception:
        return None


def ensure_1d_hist(h):
    """
    Region/systematic slicing 뒤에도 추가 axis가 남아 있으면 모두 합쳐서 1D로 만든다.
    마지막 axis가 template axis라고 가정한다.
    """
    if h is None:
        return None

    while len(h.axes) > 1:
        h = h[{h.axes[0].name: sum}]

    return h


def make_empty_like(template_hist):
    ax = template_hist.axes[0]

    if isinstance(ax, hist.axis.Variable):
        new_h = hist.Hist(
            hist.axis.Variable(ax.edges, name=ax.name, label=ax.label),
            storage=hist.storage.Weight(),
        )
    elif isinstance(ax, hist.axis.Regular):
        new_h = hist.Hist(
            hist.axis.Regular(
                ax.size,
                ax.edges[0],
                ax.edges[-1],
                name=ax.name,
                label=ax.label,
            ),
            storage=hist.storage.Weight(),
        )
    else:
        raise RuntimeError(f"Unsupported axis type for template writing: {type(ax)}")

    return new_h


def clone_hist(h):
    return copy.deepcopy(h)


def make_hist_from_values_like(template_hist, values, variances):
    """
    template_hist와 같은 axis를 가진 Weight hist를 만들고,
    values/variances를 채운다.
    """
    out = make_empty_like(template_hist)

    values = np.asarray(values, dtype=float)

    if variances is None:
        variances = np.zeros_like(values, dtype=float)
    else:
        variances = np.asarray(variances, dtype=float)

    view = out.view(flow=False)
    view.value = values
    view.variance = variances

    return out


def merge_flow_into_visible_bins(h):
    """
    1D hist에 대해 overflow를 마지막 visible bin에 산입한다.
    underflow는 기본적으로 버린다.
    """
    h = ensure_1d_hist(h)
    if h is None:
        return None

    ax = h.axes[0]
    nbins = ax.size

    vals_flow = np.asarray(h.values(flow=True), dtype=float)
    vars_flow = h.variances(flow=True)

    if vars_flow is not None:
        vars_flow = np.asarray(vars_flow, dtype=float)

    if vals_flow.shape[0] == nbins:
        vals = vals_flow.copy()
        vars_ = vars_flow.copy() if vars_flow is not None else None
        return make_hist_from_values_like(h, vals, vars_)

    if vals_flow.shape[0] != nbins + 2:
        print(
            f"[WARN] Unexpected flow shape for hist axis={ax.name}: "
            f"len(values(flow=True))={vals_flow.shape[0]}, expected {nbins + 2}. "
            f"Using first {nbins} bins."
        )
        vals = vals_flow[:nbins].copy()
        vars_ = vars_flow[:nbins].copy() if vars_flow is not None else None
        return make_hist_from_values_like(h, vals, vars_)

    vals = vals_flow[1:-1].copy()
    vars_ = vars_flow[1:-1].copy() if vars_flow is not None else None

    if INCLUDE_UNDERFLOW_IN_FIRST:
        vals[0] += vals_flow[0]
        if vars_ is not None:
            vars_[0] += vars_flow[0]

    if INCLUDE_OVERFLOW_IN_LAST:
        vals[-1] += vals_flow[-1]
        if vars_ is not None:
            vars_[-1] += vars_flow[-1]

    return make_hist_from_values_like(h, vals, vars_)


def set_negative_bins_to_zero(h):
    vals = np.array(h.values(), dtype=float)
    vars_ = h.variances()

    if vars_ is None:
        vars_ = np.zeros_like(vals, dtype=float)
    else:
        vars_ = np.array(vars_, dtype=float)

    mask = vals < 0
    vals[mask] = 0.0
    vars_[mask] = 0.0

    return make_hist_from_values_like(h, vals, vars_)


def prepare_template_hist(h, clip_negative=True, merge_overflow=True):
    h = ensure_1d_hist(h)
    if h is None:
        return None

    if merge_overflow:
        h = merge_flow_into_visible_bins(h)

    if clip_negative:
        h = set_negative_bins_to_zero(h)

    return h


def find_region_reference_hist(data, bkg, sig, variable, region):
    """
    같은 region의 binning reference로 쓸 1D hist를 하나 찾는다.
    우선순위: data -> backgrounds -> signals
    """
    data_proc = get_data_process(region)

    h = get_hist_safe(data, variable, data_proc, region, "nominal")
    h = prepare_template_hist(h, clip_negative=False)
    if h is not None:
        return h

    for proc in BKG_PROCESSES:
        source = bkg if variable in bkg and proc in bkg[variable] else sig
        h = get_hist_safe(source, variable, proc, region, "nominal")
        h = prepare_template_hist(h, clip_negative=True)
        if h is not None:
            return h

    for proc in SIG_PROCESSES:
        source = bkg if variable in bkg and proc in bkg[variable] else sig
        h = get_hist_safe(source, variable, proc, region, "nominal")
        h = prepare_template_hist(h, clip_negative=True)
        if h is not None:
            return h

    raise RuntimeError(f"Could not find any reference histogram for region={region}")


def choose_source(container_dict, variable, proc):
    """
    container_dict = {"bkg": ..., "sig": ...}
    """
    bkg = container_dict["bkg"]
    sig = container_dict["sig"]

    if variable in bkg and proc in bkg[variable]:
        return bkg
    if variable in sig and proc in sig[variable]:
        return sig
    return bkg


def integral(h):
    if h is None:
        return 0.0
    return float(np.sum(h.values()))


def check_external_axis(container, variable, process, syst_name, label):
    try:
        if variable not in container["bkg"]:
            print(f"[WARN] {label}: variable {variable} not in bkg")
            return

        source = choose_source(container, variable, process)
        if variable not in source or process not in source[variable]:
            print(f"[WARN] {label}: missing process={process}")
            return

        axis = source[variable][process].axes["systematic"]
        print(f"[CHECK] {label}: systematic axis for {variable}/{process} = {axis}")

        # Probe actual histogram existence
        h = get_hist_safe(source, variable, process, REGIONS[0], syst_name)
        if h is None:
            print(
                f"[WARN] {label}: could not find syst={syst_name} "
                f"for region={REGIONS[0]}, process={process}"
            )
        else:
            h = prepare_template_hist(h, clip_negative=True)
            print(
                f"[CHECK] {label}: example integral "
                f"{REGIONS[0]}/{process}/{syst_name} = {integral(h):.6f}"
            )
    except Exception as exc:
        print(f"[WARN] {label}: external axis check failed: {exc}")


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-scaled", default=INPUT_NOMINAL)
    parser.add_argument("--output-root", default=OUTPUT)
    parser.add_argument("--met-unclustered-up", default=INPUT_METUNCL_UP)
    parser.add_argument("--met-unclustered-down", default=INPUT_METUNCL_DOWN)
    parser.add_argument("--jes-total-up", default=INPUT_JES_UP)
    parser.add_argument("--jes-total-down", default=INPUT_JES_DOWN)
    parser.add_argument("--jer-up", default=INPUT_JER_UP)
    parser.add_argument("--jer-down", default=INPUT_JER_DOWN)
    return parser.parse_args()


def main():
    args = parse_args()
    input_nominal = args.nominal_scaled
    output = args.output_root
    external_shape_systs = copy.deepcopy(EXTERNAL_SHAPE_SYSTS)
    external_shape_systs["metUnclustered"]["up_file"] = args.met_unclustered_up
    external_shape_systs["metUnclustered"]["down_file"] = args.met_unclustered_down
    external_shape_systs["jesTotal"]["up_file"] = args.jes_total_up
    external_shape_systs["jesTotal"]["down_file"] = args.jes_total_down
    external_shape_systs["jer"]["up_file"] = args.jer_up
    external_shape_systs["jer"]["down_file"] = args.jer_down

    if not os.path.exists(input_nominal):
        raise FileNotFoundError(f"Input file not found: {input_nominal}")

    print(f"[INFO] Loading nominal: {input_nominal}")
    myhist = load(input_nominal)

    bkg = myhist["bkg"]
    data = myhist["data"]
    sig = myhist["sig"] if "sig" in myhist else myhist["bkg"]

    nominal_container = {
        "bkg": bkg,
        "sig": sig,
    }

    external_loaded = {}

    for syst_name, cfg in external_shape_systs.items():
        up_file = cfg["up_file"]
        down_file = cfg["down_file"]

        if not os.path.exists(up_file):
            raise FileNotFoundError(f"External syst file not found: {up_file}")
        if not os.path.exists(down_file):
            raise FileNotFoundError(f"External syst file not found: {down_file}")

        print(f"[INFO] Loading external syst {syst_name} Up: {up_file}")
        h_up = load(up_file)

        print(f"[INFO] Loading external syst {syst_name} Down: {down_file}")
        h_down = load(down_file)

        external_loaded[syst_name] = {
            "up": {
                "bkg": h_up["bkg"],
                "sig": h_up["sig"] if "sig" in h_up else h_up["bkg"],
                "syst": cfg["up_syst"],
            },
            "down": {
                "bkg": h_down["bkg"],
                "sig": h_down["sig"] if "sig" in h_down else h_down["bkg"],
                "syst": cfg["down_syst"],
            },
        }

        print(f"[INFO] Checking external syst axes for {syst_name}")
        up_container = {
            "bkg": external_loaded[syst_name]["up"]["bkg"],
            "sig": external_loaded[syst_name]["up"]["sig"],
        }
        down_container = {
            "bkg": external_loaded[syst_name]["down"]["bkg"],
            "sig": external_loaded[syst_name]["down"]["sig"],
        }

        check_external_axis(
            up_container,
            VARIABLE,
            "W (lnu)",
            cfg["up_syst"],
            f"{syst_name}Up",
        )
        check_external_axis(
            down_container,
            VARIABLE,
            "W (lnu)",
            cfg["down_syst"],
            f"{syst_name}Down",
        )

    all_mc_processes = BKG_PROCESSES + SIG_PROCESSES

    with uproot.recreate(output) as fout:
        for region in REGIONS:
            print(f"\n[INFO] Writing region: {region}")

            # --------------------------------------------------------
            # Region reference hist for empty templates
            # --------------------------------------------------------
            ref_hist = find_region_reference_hist(data, bkg, sig, VARIABLE, region)

            # --------------------------------------------------------
            # data_obs
            # --------------------------------------------------------
            data_proc = get_data_process(region)
            h_data = get_hist_safe(data, VARIABLE, data_proc, region, "nominal")
            h_data = prepare_template_hist(h_data, clip_negative=False)

            if h_data is None:
                h_data = make_empty_like(ref_hist)
                print(
                    f"  [WARN] missing data for region={region}, "
                    f"process={data_proc} -> writing empty data_obs"
                )

            fout[f"{region}/data_obs"] = h_data
            print(f"  wrote {region}/data_obs  integral={integral(h_data):.6f}")

            # --------------------------------------------------------
            # MC nominal
            # --------------------------------------------------------
            nominal_cache = {}

            for proc in all_mc_processes:
                source = choose_source(nominal_container, VARIABLE, proc)

                h_nom = get_hist_safe(source, VARIABLE, proc, region, "nominal")
                h_nom = prepare_template_hist(h_nom, clip_negative=True)

                if h_nom is None:
                    h_nom = make_empty_like(ref_hist)
                    print(
                        f"  [WARN] missing nominal: region={region}, "
                        f"process={proc} -> writing empty hist"
                    )

                proc_name = sanitize_name(proc)
                nominal_cache[proc] = h_nom

                fout[f"{region}/{proc_name}"] = h_nom
                print(f"  wrote {region}/{proc_name}  integral={integral(h_nom):.6f}")

            # --------------------------------------------------------
            # Internal shape systematics
            # --------------------------------------------------------
            for proc in all_mc_processes:
                source = choose_source(nominal_container, VARIABLE, proc)
                proc_name = sanitize_name(proc)
                h_nom = nominal_cache[proc]

                for syst_name, (syst_up, syst_down) in SHAPE_SYSTS.items():
                    h_up = get_hist_safe(source, VARIABLE, proc, region, syst_up)
                    h_down = get_hist_safe(source, VARIABLE, proc, region, syst_down)

                    h_up = prepare_template_hist(h_up, clip_negative=True)
                    h_down = prepare_template_hist(h_down, clip_negative=True)

                    # Missing systematic means "same as nominal", not zero.
                    if h_up is None:
                        h_up = clone_hist(h_nom)
                        print(
                            f"  [WARN] missing {syst_name}Up: "
                            f"region={region}, process={proc} -> using nominal"
                        )

                    if h_down is None:
                        h_down = clone_hist(h_nom)
                        print(
                            f"  [WARN] missing {syst_name}Down: "
                            f"region={region}, process={proc} -> using nominal"
                        )

                    fout[f"{region}/{proc_name}_{syst_name}Up"] = h_up
                    fout[f"{region}/{proc_name}_{syst_name}Down"] = h_down

                    print(
                        f"  wrote {region}/{proc_name}_{syst_name}Up"
                        f"  integral={integral(h_up):.6f}"
                    )
                    print(
                        f"  wrote {region}/{proc_name}_{syst_name}Down"
                        f"  integral={integral(h_down):.6f}"
                    )

            # --------------------------------------------------------
            # External shape systematics:
            #   metUnclustered, jesTotal, ...
            # --------------------------------------------------------
            for proc in all_mc_processes:
                proc_name = sanitize_name(proc)
                h_nom = nominal_cache[proc]

                for syst_name, payload in external_loaded.items():
                    up_container = {
                        "bkg": payload["up"]["bkg"],
                        "sig": payload["up"]["sig"],
                    }
                    down_container = {
                        "bkg": payload["down"]["bkg"],
                        "sig": payload["down"]["sig"],
                    }

                    up_syst = payload["up"]["syst"]
                    down_syst = payload["down"]["syst"]

                    up_source = choose_source(up_container, VARIABLE, proc)
                    down_source = choose_source(down_container, VARIABLE, proc)

                    h_up = get_hist_safe(up_source, VARIABLE, proc, region, up_syst)
                    h_down = get_hist_safe(down_source, VARIABLE, proc, region, down_syst)

                    h_up = prepare_template_hist(h_up, clip_negative=True)
                    h_down = prepare_template_hist(h_down, clip_negative=True)

                    if h_up is None:
                        h_up = clone_hist(h_nom)
                        print(
                            f"  [WARN] missing external {syst_name}Up: "
                            f"region={region}, process={proc}, syst={up_syst} "
                            f"-> using nominal"
                        )

                    if h_down is None:
                        h_down = clone_hist(h_nom)
                        print(
                            f"  [WARN] missing external {syst_name}Down: "
                            f"region={region}, process={proc}, syst={down_syst} "
                            f"-> using nominal"
                        )

                    fout[f"{region}/{proc_name}_{syst_name}Up"] = h_up
                    fout[f"{region}/{proc_name}_{syst_name}Down"] = h_down

                    print(
                        f"  wrote {region}/{proc_name}_{syst_name}Up"
                        f"  integral={integral(h_up):.6f}"
                    )
                    print(
                        f"  wrote {region}/{proc_name}_{syst_name}Down"
                        f"  integral={integral(h_down):.6f}"
                    )

    print(f"\n[INFO] Done. Output written to: {output}")
    print("\nExample datacard shapes line:")
    print(f"shapes * * {output} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC")

    print("\nDatacard shape nuisance examples:")
    print("pileup                  shape  ...")
    print("electron_id             shape  ...")
    print("electron_hlt            shape  ...")
    print("muon_id                 shape  ...")
    print("muon_hlt                shape  ...")
    print("btagSF_bc_correlated    shape  ...")
    print("btagSF_bc_uncorrelated  shape  ...")
    print("btagSF_light_correlated shape  ...")
    print("btagSF_light_uncorrelated shape ...")
    print("metUnclustered          shape  ...")
    print("jesTotal                shape  ...")
    print("jer                     shape  ...")


if __name__ == "__main__":
    main()