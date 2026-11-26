#!/usr/bin/env python3
"""Draw canonical Run-3 control, signal, and search-bin distributions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .background_process_groups import (
    BACKGROUND_DISPLAY_LABELS,
    BACKGROUND_PROCESS_ORDER,
    background_process_for_sample,
)


REGION_ORDER = [
    "cat2_LLCR_highDeltaM",
    "cat3_QCDCR_highDeltaM",
    "cat4_GCR_highDeltaM",
    "cat5_DY2E_highDeltaM",
    "cat6_DY2M_highDeltaM",
]
SR_REGION = "cat7_SR_highDeltaM"
SEARCH_BIN_SCHEME = "boosted_an_17"

REGION_LABELS = {
    "cat2_LLCR_highDeltaM": "LLCR",
    "cat3_QCDCR_highDeltaM": "QCDCR",
    "cat4_GCR_highDeltaM": "GCR",
    "cat5_DY2E_highDeltaM": "DY2E",
    "cat6_DY2M_highDeltaM": "DY2M",
}

# Keep this process order fixed across control-region and search-bin plots.
# Raw histogram process names stay unchanged; this is the adopted presentation
# and statistical-model grouping contract.
GROUP_ORDER = [
    BACKGROUND_DISPLAY_LABELS[process] for process in BACKGROUND_PROCESS_ORDER
]
GROUP_COLORS = {
    "VV+VVV": "#6F7661",
    "Top": "#7A9FC2",
    "DY": "#35B6B4",
    "Photon+jet": "#8E3B9E",
    "W -> lv": "#D9C6A5",
    "Z -> vv": "#E6A84F",
    "QCD Multijet": "#C995A2",
    "Others": "#6a625f",
}

SIGNAL_OVERLAYS = [
    {"key": "mStop1000_mLSP1", "label": '$m_{\\tilde{t}}=1000$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=1$ GeV', "color": "#ff0000"},
    {"key": "mStop1200_mLSP1", "label": '$m_{\\tilde{t}}=1200$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=1$ GeV', "color": "#00ff00"},
]
LOWDM_SIGNAL_OVERLAYS = [
    {
        "topology": "T2bW",
        "key": "mStop1000_mLSP800",
        "label": '$m_{\\tilde{t}}=1000$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=800$ GeV',
        "color": "#FFD500",
    },
]
LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES = [
    ("Nb0_Nj2to5_PISR500plus", 4),
    ("Nb0_Nj6plus_PISR500plus", 4),
    ("Nb1_PISR300to500_PTb20to40", 4),
    ("Nb1_PISR300to500_PTb40to70", 4),
    ("Nb1_PISR500plus_PTb20to40", 4),
    ("Nb1_PISR500plus_PTb40to70", 4),
    ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", 3),
    ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", 3),
    ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", 3),
]
LOWDM_NSV_INCLUSIVE_CATEGORY_LABELS = {
    "Nb0_Nj2to5_PISR500plus": '$N_{b}=0$\n$2\\leq N_{j}\\leq5$',
    "Nb0_Nj6plus_PISR500plus": '$N_{b}=0$\n$N_{j}\\geq6$',
    "Nb1_PISR300to500_PTb20to40": '$N_{b}=1$\n$300\\leq p_{T}^{ISR}<500$\n$20<p_{T}^{b}<40$',
    "Nb1_PISR300to500_PTb40to70": '$N_{b}=1$\n$300\\leq p_{T}^{ISR}<500$\n$40<p_{T}^{b}<70$',
    "Nb1_PISR500plus_PTb20to40": '$N_{b}=1$\n$p_{T}^{ISR}\\geq500$\n$20<p_{T}^{b}<40$',
    "Nb1_PISR500plus_PTb40to70": '$N_{b}=1$\n$p_{T}^{ISR}\\geq500$\n$40<p_{T}^{b}<70$',
    "Nb2plus_PISR300to500_PTb40to80_Nj2plus": '$N_{b}\\geq2$\n$300\\leq p_{T}^{ISR}<500$\n$40<p_{T}^{b}<80$',
    "Nb2plus_PISR300to500_PTb80to140_Nj2plus": '$N_{b}\\geq2$\n$300\\leq p_{T}^{ISR}<500$\n$80<p_{T}^{b}<140$',
    "Nb2plus_PISR300to500_PTb140plus_Nj7plus": '$N_{b}\\geq2$, $N_{j}\\geq7$\n$300\\leq p_{T}^{ISR}<500$\n$p_{T}^{b}>140$',
    "Nb2plus_PISR500plus_PTb40to80_Nj2plus": '$N_{b}\\geq2$\n$p_{T}^{ISR}\\geq500$\n$40<p_{T}^{b}<80$',
    "Nb2plus_PISR500plus_PTb80to140_Nj2plus": '$N_{b}\\geq2$\n$p_{T}^{ISR}\\geq500$\n$80<p_{T}^{b}<140$',
    "Nb2plus_PISR500plus_PTb140plus_Nj7plus": '$N_{b}\\geq2$, $N_{j}\\geq7$\n$p_{T}^{ISR}\\geq500$\n$p_{T}^{b}>140$',
}
PARTIAL_AN17_SPLIT_BINS = [4, 5, 8, 9, 14, 15, 16]
EXTENDED_AN17_RECOIL_SCHEME = "highdm_search_bins"
HIGHDM_PLOT_DROPPED_LEADING_BINS = 6
LUMINOSITY_FB = 109.82
LUMINOSITY_RELATIVE_UNCERTAINTY = 0.016
PLOT_SYSTEMATIC_SOURCES = [
    "pileup",
    "electron_id",
    "electron_reco",
    "electron_hlt",
    "veto_electron_5to10",
    "muon_id",
    "muon_iso",
    "muon_hlt",
    "loose_muon_5to10",
    "photon_id",
    "photon_csev",
    "photon_trigger",
    "met_trigger",
    "btagSF_bc_correlated",
    "btagSF_bc_uncorrelated",
    "btagSF_light_correlated",
    "btagSF_light_uncorrelated",
    "jesFlavorQCD",
    "jesTotal",
    "metUnclustered",
]
SELECTED_AN17_CATEGORY_LABELS = {
    'Nb1plus_T0_W0': '$N_{b}\\geq1$\n$N_{t}=0$, $N_{W}=0$\n$N_{res}=0$',
    'Nb1plus_T0_W1plus': '$N_{b}\\geq1$\n$N_{t}=0$, $N_{W}\\geq1$\n$N_{res}=0$',
    'Nb1_T1plus_W0': '$N_{b}=1$\n$N_{t}\\geq1$, $N_{W}=0$\n$N_{res}=0$',
    'Nb1_T1plus_W1plus': '$N_{b}=1$\n$N_{t}\\geq1$, $N_{W}\\geq1$\n$N_{res}=0$',
    'Nb2_T1_W0': '$N_{b}=2$\n$N_{t}=1$, $N_{W}=0$\n$N_{res}=0$',
    'Nb2_T1_W1': '$N_{b}=2$\n$N_{t}=1$, $N_{W}=1$\n$N_{res}=0$',
    'Nb3plus_T1_W0': '$N_{b}\\geq3$\n$N_{t}=1$, $N_{W}=0$\n$N_{res}=0$',
    'Nb3plus_T1_W1': '$N_{b}\\geq3$\n$N_{t}=1$, $N_{W}=1$\n$N_{res}=0$',
    'Nb3plus_T2_W0': '$N_{b}\\geq3$\n$N_{t}=2$, $N_{W}=0$\n$N_{res}=0$',
    'Nb2_Nt2plus_W0': '$N_{b}=2$\n$N_{t}\\geq2$, $N_{W}=0$\n$N_{res}=0$',
    'merged_high_nt': '$N_{b}\\geq3$\n$N_{t}=1,2$\n$N_{W}=1,0$; $N_{res}=0$',
}
RESOLVED_CATEGORY_LABELS = {
    "resolved1_only": "$N_{b}\\geq1$\n$N_{t}=0$, $N_{W}=0$\n$N_{res}=1$",
    "resolved2plus_only": "$N_{b}\\geq1$\n$N_{t}=0$, $N_{W}=0$\n$N_{res}\\geq2$",
    "w_resolved": "$N_{b}\\geq1$\n$N_{t}=0$, $N_{W}\\geq1$\n$N_{res}\\geq1$",
    "top_resolved": "$N_{b}\\geq1$\n$N_{t}\\geq1$, $N_{W}=0$\n$N_{res}\\geq1$",
    "top_w_resolved": "$N_{b}\\geq1$\n$N_{t}\\geq1$, $N_{W}\\geq1$\n$N_{res}\\geq1$",
    "nb2_nt0_nw2_nres0": "$N_{b}=2$\n$N_{t}=0$, $N_{W}=2$\n$N_{res}=0$",
}
SELECTED_AN17_CATEGORY_ORDER = [
    "Nb1plus_T0_W0",
    "Nb1plus_T0_W1plus",
    "Nb1_T1plus_W0",
    "Nb1_T1plus_W1plus",
    "Nb2_T1_W0",
    "Nb2_T1_W1",
    "Nb2_Nt2plus_W0",
    "Nb3plus_T1_W0",
    "Nb3plus_T1_W1",
    "Nb3plus_T2_W0",
]



def load_json(path: Path) -> dict:
    return json.loads(path.read_text())




















def as_array(values: list[float] | None, nbin: int) -> np.ndarray:
    out = np.zeros(nbin, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float)
    out[: min(nbin, arr.size)] = arr[:nbin]
    return out


def validated_plot_uncertainty(background, statistical_variance, band=None):
    """Accept an exported total band without changing any nominal plot input."""
    statistical = np.sqrt(np.clip(statistical_variance, 0.0, None))
    if band is None:
        return statistical
    if band.get("status") != "complete":
        raise ValueError("incomplete plot uncertainty export")
    np.testing.assert_allclose(background, band["background"], rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(statistical, band["stat_unc"], rtol=1e-10, atol=1e-7)
    total = np.asarray(band["total_unc"], dtype=float)
    if total.shape != statistical.shape or not np.isfinite(total).all() or np.any(total + 1e-10 < statistical):
        raise ValueError("invalid total plot uncertainty")
    return total


def process_to_group(process: str) -> str:
    return BACKGROUND_DISPLAY_LABELS[background_process_for_sample(process)]


def apply_dy_rz(payload: dict, manifest_path: Path) -> dict:
    """Apply the adopted DY2E/DY2M RZ factors in the main render payload."""
    manifest = load_json(manifest_path)
    config = manifest.get("dy_rz") or {}
    if config.get("status") != "complete":
        raise ValueError(f"missing complete dy_rz configuration in {manifest_path}")
    channels = config.get("channels") or {}
    audit = {
        "status": "complete",
        "source": config.get("source"),
        "factor_policy": config.get("factor_policy"),
        "legend_label": "DY",
        "containers": 0,
        "records": 0,
        "channels": channels,
    }

    def factors(channel: str, regime: str, nbin: int, variable: str | None) -> np.ndarray:
        record = ((channels.get(channel) or {}).get(regime) or {})
        effective = float(record["effective"])
        result = np.full(nbin, effective, dtype=float)
        if variable and variable.startswith("nb"):
            specs = (
                payload.get("highdm_distribution_variable_specs")
                if regime == "highdm"
                else payload.get("lowdm_variable_specs")
            ) or {}
            edges = np.asarray((specs.get(variable) or {}).get("bins") or [], dtype=float)
            if len(edges) == nbin + 1:
                centers = 0.5 * (edges[:-1] + edges[1:])
                result[centers == 1] = float(record["Nb1"])
                result[centers >= 2] = float(record["Nb2plus"])
        return result

    def scale_leaf(leaf: dict, scale: np.ndarray) -> None:
        if "sumw" in leaf:
            values = as_array(leaf.get("sumw"), len(scale))
            leaf["sumw"] = (values * scale).tolist()
        if "sumw2" in leaf:
            values = as_array(leaf.get("sumw2"), len(scale))
            leaf["sumw2"] = (values * scale * scale).tolist()

    def scale_container(raw: dict, channel: str, regime: str, variable: str | None = None) -> None:
        if not raw:
            return
        nbin = 0
        for record in raw.values():
            nominal = record.get("nominal") or record
            nbin = max(nbin, len(nominal.get("sumw") or []))
        if nbin <= 0:
            return
        scale = factors(channel, regime, nbin, variable)
        changed = 0
        for sample, record in raw.items():
            if sample == "data_obs" or is_signal_sample(sample):
                continue
            try:
                is_dy = process_to_group(sample) == "DY"
            except (KeyError, ValueError):
                is_dy = False
            if not is_dy:
                continue
            if "sumw" in record:
                scale_leaf(record, scale)
            else:
                for leaf in record.values():
                    if isinstance(leaf, dict) and "sumw" in leaf:
                        scale_leaf(leaf, scale)
            changed += 1
        if changed:
            audit["containers"] += 1
            audit["records"] += changed

    for region, raw in (payload.get("histograms") or {}).items():
        for channel in ("DY2E", "DY2M"):
            if region == channel or region.startswith(channel + "_"):
                scale_container(raw, channel, "highdm")
    for channel in ("DY2E", "DY2M"):
        variable_map = (payload.get("highdm_variable_histograms") or {}).get(channel) or {}
        for variable, raw in variable_map.items():
            scale_container(raw, channel, "highdm", variable)
    lowdm_channels = {
        "cat5_DY2E_lowDeltaM": "DY2E",
        "cat6_DY2M_lowDeltaM": "DY2M",
    }
    for scheme, channel in lowdm_channels.items():
        scale_container(
            (payload.get("search_bin_histograms") or {}).get(scheme) or {},
            channel,
            "lowdm",
        )
        for variable, raw in (
            (payload.get("lowdm_variable_histograms") or {}).get(scheme) or {}
        ).items():
            scale_container(raw, channel, "lowdm", variable)
    payload["_dy_rz_application"] = audit
    return audit


def poisson_unc(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(data, 0.0))













FLAT_REGION_LABELS = {
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
    "GCR": "GCR",
    "DY2E": "DY2E",
    "DY2M": "DY2M",
    "DYCR": r"DYCR ($ee+\mu\mu$)",
    "SR": "SR",
    "LLCR_Nt0": r"LLCR\n$N_{t}=0$",
    "QCDCR_Nt0": r"QCDCR\n$N_{t}=0$",
    "GCR_Nt0": r"GCR\n$N_{t}=0$",
    "DY2E_Nt0": r"DY2E\n$N_{t}=0$",
    "DY2M_Nt0": r"DY2M\n$N_{t}=0$",
    "SR_Nt0": r"SR\n$N_{t}=0$",
    "LLCR_Nt1": r"LLCR\n$N_{t}\geq1$",
    "QCDCR_Nt1": r"QCDCR\n$N_{t}\geq1$",
    "GCR_Nt1": r"GCR\n$N_{t}\geq1$",
    "DY2E_Nt1": r"DY2E\n$N_{t}\geq1$",
    "DY2M_Nt1": r"DY2M\n$N_{t}\geq1$",
    "SR_Nt1": r"SR\n$N_{t}\geq1$",
}


def is_signal_sample(sample: str) -> bool:
    return sample.startswith(("T2tt", "T2tb", "T2bW"))


def flat_values(rec: dict, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    nominal = rec.get("nominal") or rec
    return as_array(nominal.get("sumw"), nbin), as_array(nominal.get("sumw2"), nbin)


def combine_histogram_containers(*containers: dict) -> dict:
    """Sum same-binned histogram containers before grouping or uncertainty building.

    A systematic shape that is absent from one channel is represented by that
    channel's nominal histogram. This preserves the intended correlation when
    the electron and muon DY control regions are combined.
    """
    sources = [container for container in containers if container]
    if not sources:
        return {}

    def variation_map(record: dict) -> dict[str, dict]:
        if "sumw" in record:
            return {"nominal": record}
        return {
            str(name): leaf
            for name, leaf in record.items()
            if isinstance(leaf, dict) and "sumw" in leaf
        }

    nominal_lengths = {
        len((record.get("nominal") or record).get("sumw") or [])
        for container in sources
        for record in container.values()
        if len((record.get("nominal") or record).get("sumw") or []) > 0
    }
    if len(nominal_lengths) != 1:
        raise ValueError(
            f"cannot combine histogram containers with bin counts {sorted(nominal_lengths)}"
        )

    def sum_leaves(leaves: list[dict]) -> dict:
        nbin = max((len(leaf.get("sumw") or []) for leaf in leaves), default=0)
        result = {}
        for key in ("entries", "sumw", "sumw2"):
            if any(key in leaf for leaf in leaves):
                total = np.zeros(nbin, dtype=float)
                for leaf in leaves:
                    total += as_array(leaf.get(key), nbin)
                result[key] = total.tolist()
        return result

    combined = {}
    samples = sorted({sample for container in sources for sample in container})
    for sample in samples:
        records = [container[sample] for container in sources if sample in container]
        mapped = [variation_map(record) for record in records]
        if not mapped:
            continue
        variations = {"nominal"}
        for record in mapped:
            variations.update(record)
        merged_variations = {}
        for variation in sorted(variations, key=lambda name: (name != "nominal", name)):
            leaves = []
            for record in mapped:
                leaf = record.get(variation) or record.get("nominal")
                if leaf:
                    leaves.append(leaf)
            if leaves:
                merged_variations[variation] = sum_leaves(leaves)
        if merged_variations:
            combined[sample] = merged_variations
    return combined


def background_systematic_variance(raw: dict, nbin: int) -> np.ndarray:
    syst2 = np.zeros(nbin, dtype=float)
    for source in PLOT_SYSTEMATIC_SOURCES:
        up_total = np.zeros(nbin, dtype=float)
        down_total = np.zeros(nbin, dtype=float)
        have = False
        for sample, rec in raw.items():
            if sample == "data_obs" or is_signal_sample(sample):
                continue
            nominal, _ = flat_values(rec, nbin)
            up_rec = rec.get(source + "Up")
            down_rec = rec.get(source + "Down")
            if up_rec:
                up_total += as_array(up_rec.get("sumw"), nbin) - nominal
                have = True
            if down_rec:
                down_total += as_array(down_rec.get("sumw"), nbin) - nominal
                have = True
        if have:
            syst2 += np.maximum(np.abs(up_total), np.abs(down_total)) ** 2
    return syst2


def background_systematic_totals(raw: dict, nbin: int) -> dict[str, dict[str, np.ndarray]]:
    """Return absolute total-background Up/Down shapes for each stored source."""
    nominal_total = np.zeros(nbin, dtype=float)
    for sample, rec in raw.items():
        if sample == "data_obs" or is_signal_sample(sample):
            continue
        nominal, _ = flat_values(rec, nbin)
        nominal_total += nominal
    totals: dict[str, dict[str, np.ndarray]] = {}
    for source in PLOT_SYSTEMATIC_SOURCES:
        up_total = nominal_total.copy()
        down_total = nominal_total.copy()
        have = False
        for sample, rec in raw.items():
            if sample == "data_obs" or is_signal_sample(sample):
                continue
            nominal, _ = flat_values(rec, nbin)
            up_rec = rec.get(source + "Up")
            down_rec = rec.get(source + "Down")
            if up_rec:
                up_total += as_array(up_rec.get("sumw"), nbin) - nominal
                have = True
            if down_rec:
                down_total += as_array(down_rec.get("sumw"), nbin) - nominal
                have = True
        if have:
            totals[source] = {"up": up_total, "down": down_total}
    return totals






VARIABLE_XLABELS = {
    "met": r"$p_{T}^{miss}$ (GeV)",
    "ht": r"$H_{T}$ (GeV)",
    "njet": r"$N_{j}$",
    "nb_medium_lowdm": r"$N_{b}$",
    "nb_loose_lowdm": r"$N_{b}$ (loose WP)",
    "n_e_veto": r"$N_{e}^{\mathrm{veto}}$",
    "n_m_loose": r"$N_{\mu}^{\mathrm{loose}}$",
    "lowdm_mtb": r"$m_{T}^{b}$ (GeV)",
    "recoil_gcr": r"$U_{T}$ (GeV)",
    "recoil_dy2e": r"$U_{T}$ (GeV)",
    "recoil_dy2m": r"$U_{T}$ (GeV)",
    "recoil_dy": r"$U_{T}$ (GeV)",
    "lowdm_met_sqrt_ht": r"$p_{T}^{miss}/\sqrt{H_{T}}$",
    "lowdm_isr_pt": r"$p_{T}^{\mathrm{ISR}}$ (GeV)",
    "lowdm_isr_dphi": r"$\Delta\phi(\mathrm{ISR},p_{T}^{miss})$",
    "lowdm_ptb": r"$p_{T}^{b}$ (GeV)",
    "n_lowdm_isr": r"$N_{\mathrm{ISR}}$",
    "mee": r"$m_{ee}$ (GeV)",
    "mmm": r"$m_{\mu\mu}$ (GeV)",
    "mll": r"$m_{\ell\ell}$ (GeV)",
    "n_photon_medium": r"$N_{\gamma}$",
    "njet_photon_clean": r"$N_{j}$",
    "nb_photon_clean": r"$N_{b}$",
    "ht_photon_clean": r"$H_{T}$ (GeV)",
    "njet_lepton_clean": r"$N_{j}$",
    "nb_lepton_clean": r"$N_{b}$",
    "ht_lepton_clean": r"$H_{T}$ (GeV)",
    "leading_lowdm_fatjet_pt": r"Leading AK8 jet $p_{T}$ (GeV)",
    "leading_lowdm_fatjet_msd": r"Leading AK8 jet $m_{\mathrm{SD}}$ (GeV)",
}


HIGHDM_VARIABLE_XLABELS = {
    "nb": r"$N_{b}$",
    "njet": r"$N_{j}$",
    "nfatjet": r"$N_{\mathrm{AK8}}$",
    "ntop": r"$N_{t}$",
    "nw": r"$N_{W}$",
    "ht": r"$H_{T}$ (GeV)",
    "ut": r"$U_{T}$ (GeV)",
    "ptll": r"$p_{T}(\ell\ell)$ (GeV)",
    "met": r"$p_{T}^{miss}$ (GeV)",
    "jet_pt": r"Leading jet $p_{T}$ (GeV)",
    "fatjet_pt": r"Leading AK8 jet $p_{T}$ (GeV)",
    "bjet_pt": r"Leading b-jet $p_{T}$ (GeV)",
}


def lowdm_variable_record(
    payload: dict,
    scheme: str,
    variable: str,
    label: str,
    allow_signal: bool,
    *,
    raw_override: dict | None = None,
    spec_variable: str | None = None,
    display_variable: str | None = None,
) -> dict | None:
    raw = (
        raw_override
        if raw_override is not None
        else (((payload.get("lowdm_variable_histograms") or {}).get(scheme) or {}).get(variable) or {})
    )
    if not raw:
        return None
    spec_key = spec_variable or variable
    output_variable = display_variable or variable
    spec = ((payload.get("lowdm_variable_specs") or {}).get(spec_key) or {})
    edges = spec.get("bins") or []
    nbin = max(0, len(edges) - 1)
    if nbin <= 0:
        for rec in raw.values():
            nominal = rec.get("nominal") or rec
            nbin = max(nbin, len(nominal.get("sumw") or []))
        edges = []
    if nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {sig["key"]: np.zeros(nbin, dtype=float) for sig in LOWDM_SIGNAL_OVERLAYS}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for sig in LOWDM_SIGNAL_OVERLAYS:
                    if sample == sig.get("topology", "T2tt") + "_" + sig["key"]:
                        signals[sig["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    visible = bkg if allow_signal else bkg + data
    first_visible_bin = next((index for index, value in enumerate(visible) if value > 0), 0)
    xlim_left = float(edges[first_visible_bin]) if len(edges) == nbin + 1 else None
    result = {
        "groups": groups,
        "background": bkg,
        "background_unc": np.sqrt(stat2 + syst2),
        "background_stat_unc": np.sqrt(stat2),
        "background_systematic_totals": background_systematic_totals(raw, nbin),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "signal_specs": LOWDM_SIGNAL_OVERLAYS,
        "label": label,
        "nbin": nbin,
        "edges": edges,
        "xlabel": VARIABLE_XLABELS.get(output_variable, spec.get("xlabel") or output_variable),
        "variable": output_variable,
        "reference_style": True,
        "xlim_left": xlim_left,
        "unit_area": scheme == "cat4_GCR_lowDeltaM",
        "physics_scope": "GCR" if scheme == "cat4_GCR_lowDeltaM" else scheme,
    }
    return result



HIGHDM_DISTRIBUTION_REGION_LABELS = {
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
    "GCR": "GCR",
    "DY2E": "DY2E",
    "DY2M": "DY2M",
    "DYCR": r"DYCR ($ee+\mu\mu$)",
    "SR_Nb1plus_T0_W0": r"SR, $N_{b}\geq1$, $N_{top}=0$, $N_{W}=0$",
    "SR_Nb1plus_T0_W1plus": r"SR, $N_{b}\geq1$, $N_{top}=0$, $N_{W}\geq1$",
    "SR_Nb1_T1plus_W0": r"SR, $N_{b}=1$, $N_{top}\geq1$, $N_{W}=0$",
    "SR_Nb1_T1plus_W1plus": r"SR, $N_{b}=1$, $N_{top}\geq1$, $N_{W}\geq1$",
    "SR_Nb2_T1_W0": r"SR, $N_{b}=2$, $N_{top}=1$, $N_{W}=0$",
    "SR_Nb2_T1_W1": r"SR, $N_{b}=2$, $N_{top}=1$, $N_{W}=1$",
    "SR_Nb3plus_T1_W0": r"SR, $N_{b}\geq3$, $N_{top}=1$, $N_{W}=0$",
    "SR_Nb3plus_T1_W1": r"SR, $N_{b}\geq3$, $N_{top}=1$, $N_{W}=1$",
    "SR_Nb3plus_T2_W0": r"SR, $N_{b}\geq3$, $N_{top}=2$, $N_{W}=0$",
}

GROUP_DISPLAY_LABELS = {
    "VV+VVV": "VV+VVV",
    "Top": "Top",
    "DY": "DY",
    "Photon+jet": "Photon+jet",
    "W -> lv": r"$\mathrm{W}\to\ell\nu$",
    "Z -> vv": r"$\mathrm{Z}\to\nu\bar{\nu}$",
    "QCD Multijet": "QCD Multijet",
    "Others": "Others",
}


HIGHDM_MULTIPLICITY_PLOT_BINS = {
    "nb": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "groups": [[1], [2], [3, 4, 5]],
        "edges": [0.5, 1.5, 2.5, 3.5],
        "labels": ["1", "2", r"$\geq3$"],
    },
    "njet": {
        "source_edges": [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 10.5, 12.5, 16.5],
        "groups": [[4], [5], [6], [7, 8, 9]],
        "edges": [4.5, 5.5, 6.5, 7.5, 8.5],
        "labels": ["5", "6", "7-8", r"$\geq9$"],
    },
    "nfatjet": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5],
        "groups": [[0], [1], [2], [3, 4, 5]],
        "edges": [-0.5, 0.5, 1.5, 2.5, 3.5],
        "labels": ["0", "1", "2", r"$\geq3$"],
    },
    "ntop": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5],
        "groups": [[0], [1], [2, 3, 4]],
        "edges": [-0.5, 0.5, 1.5, 2.5],
        "labels": ["0", "1", r"$\geq2$"],
    },
    "nw": {
        "source_edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5],
        "groups": [[0], [1], [2, 3, 4]],
        "edges": [-0.5, 0.5, 1.5, 2.5],
        "labels": ["0", "1", r"$\geq2$"],
    },
}


def rebin_highdm_multiplicity(
    raw: dict, variable: str, source_edges: list[float]
) -> tuple[dict, list[float], list[str]]:
    config = HIGHDM_MULTIPLICITY_PLOT_BINS.get(variable)
    if config is None:
        return raw, source_edges, []
    expected_edges = np.asarray(config["source_edges"], dtype=float)
    if len(source_edges) != len(expected_edges) or not np.allclose(source_edges, expected_edges):
        raise ValueError(f"unexpected source bins for {variable}: {source_edges}")

    def rebin_leaf(leaf: dict) -> dict:
        result = dict(leaf)
        for key in ("entries", "sumw", "sumw2"):
            if key not in leaf:
                continue
            values = as_array(leaf.get(key), len(expected_edges) - 1)
            result[key] = [float(np.sum(values[group])) for group in config["groups"]]
        return result

    rebinned = {}
    for sample, record in raw.items():
        if "sumw" in record:
            rebinned[sample] = rebin_leaf(record)
            continue
        rebinned[sample] = {
            name: rebin_leaf(value) if isinstance(value, dict) and "sumw" in value else value
            for name, value in record.items()
        }
    return rebinned, list(config["edges"]), list(config["labels"])


def highdm_variable_record(
    payload: dict,
    region: str,
    variable: str,
    *,
    raw_override: dict | None = None,
) -> dict | None:
    raw = (
        raw_override
        if raw_override is not None
        else ((((payload.get("highdm_variable_histograms") or {}).get(region) or {}).get(variable)) or {})
    )
    spec = ((payload.get("highdm_distribution_variable_specs") or {}).get(variable) or {})
    source_edges = spec.get("bins") or []
    raw, edges, xlabels = rebin_highdm_multiplicity(raw, variable, source_edges)
    nbin = max(0, len(edges) - 1)
    if not raw or nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif not is_signal_sample(sample):
            groups[process_to_group(sample)] += vals
            stat2 += s2
    background = sum(groups.values(), np.zeros(nbin, dtype=float))
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * background) ** 2
    if not np.any(background > 0) and not np.any(data > 0):
        return None
    visible = background if region.startswith("SR_") else background + data
    first_visible_bin = next((index for index, value in enumerate(visible) if value > 0), 0)
    xlim_left = float(edges[0] if xlabels else edges[first_visible_bin])
    result = {
        "groups": groups,
        "background": background,
        "background_unc": np.sqrt(stat2 + syst2),
        "background_stat_unc": np.sqrt(stat2),
        "background_systematic_totals": background_systematic_totals(raw, nbin),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": {},
        "label": HIGHDM_DISTRIBUTION_REGION_LABELS.get(region, region),
        "nbin": nbin,
        "edges": edges,
        "xlabels": xlabels,
        "xlim_left": xlim_left,
        "xlabel": HIGHDM_VARIABLE_XLABELS.get(variable, spec.get("xlabel") or variable),
        "variable": variable,
        "region": region,
        "blind_data": region.startswith("SR_"),
        "annotation": HIGHDM_DISTRIBUTION_REGION_LABELS.get(region, region),
        "reference_style": True,
        "unit_area": region == "GCR",
        "physics_scope": "GCR" if region == "GCR" else region,
    }
    return result










def draw_cronly_prefit_postfit(payload, outbase, luminosity_fb):
    """Draw prefit and postfit CR comparisons with the distribution renderer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    if (payload.get("schema") != "trotasr_cronly_prefit_postfit_v1"
            or payload.get("sr_observations_included") is not False):
        raise ValueError("Observed CR-only plot input required")
    hep.style.use("CMS")
    # Use a 24-inch canvas with high-dM left and low-dM right in each fit panel.
    frame_height = 5.5
    canvas_height = 2*frame_height
    fig = plt.figure(figsize=(24.0, canvas_height))
    panels = []
    for index, (stage, title) in enumerate((("prefit", "Prefit"), ("postfit", "Postfit"))):
        offset = frame_height if index == 0 else 0.0
        grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.1], hspace=.04,
            left=matplotlib.rcParams["figure.subplot.left"],
            right=matplotlib.rcParams["figure.subplot.right"],
            bottom=(offset+1.00)/canvas_height,
            top=(offset+5.00)/canvas_height)
        ax = fig.add_subplot(grid[0])
        rax = fig.add_subplot(grid[1],sharex=ax)
        blocks = []
        for mode in ("highdm", "lowdm"):
            mode_label = r"high $\Delta m$" if mode == "highdm" else r"low $\Delta m$"
            for region in ("LLCR", "QCDCR", "GCR"):
                row = payload["channels"][region+"_"+mode+"_c1"]
                current = payload.get("fit_source", {}).get("manifest") == "stats/run3/top_pt/grid/manifest.json"
                count = 5 if current or mode == "lowdm" else 6
                if len(row["data"]) != count*2:
                    raise ValueError("Do not rebin the approved CR likelihood")
                for nb, label in enumerate((r"$N_b=1$", r"$N_b\geq2$")):
                    sl = slice(nb*count,(nb+1)*count)
                    values = row[stage]
                    data = np.asarray(row["data"])[sl]
                    blocks.append(dict(nbin=count,label=region+"\n"+mode_label+"\n"+label,
                        label_box=True,label_fontsize=17,label_box_pad=.30,label_va="bottom",
                        category_labels_on_main=True,category_label_y=.04,
                        main_panel_ymax_factor=4.,main_annotation=title,
                        main_annotation_y=.90,main_annotation_fontsize=24,
                        main_annotation_fontweight="bold",
                        groups={k:np.asarray(v)[sl] for k,v in values["groups"].items()},
                        background=np.asarray(values["background"])[sl],
                        background_unc=np.asarray(values["background_unc"])[sl],
                        data=data,data_unc=poisson_unc(data),
                        signals={},signal_specs=[],xlabels=[str(i+1) for i in range(count)],
                        physics_scope=region,unit_area=False,blind_data=False))
        info = draw_flat_blocks(blocks,outbase,
            xlabel="CR bin",
            reference_style=True,show_yields=False,luminosity_fb=luminosity_fb,
            figure_axes=(fig,ax,rax),legend_ncol=9)
        display_peak = max(float(np.max(np.maximum(block["data"],
            block["background"]+block["background_unc"]))) for block in blocks)
        ax.set_ylim(top=11.5*display_peak)
        info["main_panel_ylim"] = [float(v) for v in ax.get_ylim()]
        ax.tick_params(axis="x",labelbottom=False)
        rax.yaxis.label.set_fontsize(24)
        panels.append(info)
    # Crop horizontal whitespace only; keep every artist's size and vertical layout.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    output_box = Bbox.from_extents(tight.x0-.10, 0., tight.x1+.10, canvas_height)
    fig.savefig(outbase.with_suffix(".png"),dpi=180,bbox_inches=output_box,pad_inches=0)
    fig.savefig(outbase.with_suffix(".pdf"),bbox_inches=output_box,pad_inches=0)
    plt.close(fig)
    return dict(modes=["highdm","lowdm"],bins=sum(len(row["data"]) for row in payload["channels"].values()),
        stages=["prefit","postfit"],uncertainty_bands=True,fit_recalculated=False,
        style="canonical draw_flat_blocks(reference_style=True, show_yields=False)",
        figure_size_inches=[float(output_box.width),canvas_height],
        canvas_size_inches=[24.,canvas_height],horizontal_outer_padding_inches=.10,
        legend_yields_displayed=False,
        category_boxes="inside main ax along its x axis",panel_height_ratios=[3.0,1.1],
        nb_in_each_region_box=True,panels=panels,
        png=str(outbase.with_suffix(".png")),pdf=str(outbase.with_suffix(".pdf")))


def draw_flat_blocks(
    blocks: list[dict],
    outbase: Path,
    xlabel: str = "Bin",
    reference_style: bool = False,
    show_yields: bool = False,
    ratio_ylabel: str | None = None,
    uncertainty_label_override: str | None = None,
    luminosity_fb: float | None = None,
    x_tick_rotation: float = 0.0,
    figure_axes=None,
    legend_ncol: int | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["hatch.linewidth"] = 1.4
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    reference_style = reference_style or any(bool(block.get("reference_style")) for block in blocks)
    nbin = sum(int(block["nbin"]) for block in blocks)
    physical_edges = None
    if len(blocks) == 1:
        candidate_edges = np.asarray(blocks[0].get("edges") or [], dtype=float)
        if len(candidate_edges) == int(blocks[0]["nbin"]) + 1 and np.all(np.diff(candidate_edges) > 0):
            physical_edges = candidate_edges
    if physical_edges is not None:
        edges = physical_edges
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        centers = np.arange(1, nbin + 1, dtype=float)
        edges = np.arange(0.5, nbin + 1.5, 1.0)
    xerr = 0.5 * np.diff(edges)
    boundaries = [0]
    labels = []
    for block in blocks:
        boundaries.append(boundaries[-1] + int(block["nbin"]))
        labels.append(block["label"])
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    bkg = np.zeros(nbin, dtype=float)
    unc = np.zeros(nbin, dtype=float)
    stat_unc = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_unc = np.zeros(nbin, dtype=float)
    data_mask = np.ones(nbin, dtype=bool)
    signal_specs = []
    seen_signal_keys = set()
    for block in blocks:
        for spec in block.get("signal_specs") or SIGNAL_OVERLAYS:
            if spec["key"] not in seen_signal_keys:
                signal_specs.append(spec)
                seen_signal_keys.add(spec["key"])
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in signal_specs}
    group_labels = {}
    for block in blocks:
        group_labels.update(block.get("group_labels") or {})
    systematic_totals = {
        source: {
            "up": np.zeros(nbin, dtype=float),
            "down": np.zeros(nbin, dtype=float),
        }
        for source in PLOT_SYSTEMATIC_SOURCES
    }
    available_systematic_sources = set()
    offset = 0
    for block in blocks:
        n = int(block["nbin"])
        slc = slice(offset, offset + n)
        for group in GROUP_ORDER:
            groups[group][slc] = block["groups"].get(group, np.zeros(n))
        bkg[slc] = block["background"]
        unc[slc] = block["background_unc"]
        stat_unc[slc] = block.get("background_stat_unc", block["background_unc"])
        data[slc] = block["data"]
        data_unc[slc] = block["data_unc"]
        block_systematics = block.get("background_systematic_totals") or {}
        for source in PLOT_SYSTEMATIC_SOURCES:
            varied = block_systematics.get(source)
            if varied:
                systematic_totals[source]["up"][slc] = varied["up"]
                systematic_totals[source]["down"][slc] = varied["down"]
                available_systematic_sources.add(source)
            else:
                systematic_totals[source]["up"][slc] = block["background"]
                systematic_totals[source]["down"][slc] = block["background"]
        if block.get("blind_data"):
            data_mask[slc] = False
        for key, vals in block.get("signals", {}).items():
            signals[key][slc] = vals
        offset += n
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    significance_flags = {bool(block.get("significance_panel")) for block in blocks}
    if len(significance_flags) != 1:
        raise RuntimeError("cannot mix significance and Data/MC lower panels")
    significance_panel = significance_flags.pop()
    significance_ylims = {
        tuple(float(value) for value in block.get("significance_ylim", [0.0, 5.0]))
        for block in blocks
    }
    if significance_panel and len(significance_ylims) != 1:
        raise RuntimeError("cannot mix significance y-axis ranges")
    lower_panel_ylim = significance_ylims.pop() if significance_panel else (0.0, 2.0)
    significance_modes = {
        str(block.get("significance_mode", "s_over_sqrt_b_plus_unc2"))
        for block in blocks
    }
    if significance_panel and len(significance_modes) != 1:
        raise RuntimeError("cannot mix significance definitions")
    significance_mode = (
        significance_modes.pop() if significance_panel else None
    )
    main_ylabels = {
        str(block["main_ylabel"])
        for block in blocks
        if block.get("main_ylabel")
    }
    if len(main_ylabels) > 1:
        raise RuntimeError("cannot mix main-panel y-axis labels")
    main_ylabel = next(iter(main_ylabels), None)
    significance_ylabels = {
        str(block["significance_ylabel"])
        for block in blocks
        if block.get("significance_ylabel")
    }
    if significance_panel and len(significance_ylabels) > 1:
        raise RuntimeError("cannot mix significance y-axis labels")
    significance_ylabel = next(iter(significance_ylabels), "Significance")

    unit_area = bool(blocks) and all(
        bool(block.get("unit_area")) and block.get("physics_scope") == "GCR"
        for block in blocks
    )
    unit_area_audit = None
    uncertainty_label = uncertainty_label_override or (
        "Stat. syst. unc" if reference_style else "MC stat+syst unc."
    )
    raw_legend_yields = {
        group: float(np.sum(values)) for group, values in groups.items()
    }
    raw_legend_yields["Data"] = float(np.sum(data[data_mask]))
    raw_signal_yields = {
        key: float(np.sum(values)) for key, values in signals.items()
    }
    if unit_area:
        raw_groups = {group: values.copy() for group, values in groups.items()}
        raw_bkg = bkg.copy()
        raw_data = data.copy()
        raw_stat_unc = stat_unc.copy()
        data_integral = float(np.sum(raw_data))
        mc_integral = float(np.sum(raw_bkg))
        if not np.isfinite(data_integral) or data_integral <= 0:
            raise RuntimeError(f"GCR data integral is not positive and finite: {data_integral}")
        if not np.isfinite(mc_integral) or mc_integral <= 0:
            raise RuntimeError(f"GCR MC integral is not positive and finite: {mc_integral}")
        raw_legend_yields["Data"] = data_integral

        for group in groups:
            groups[group] = raw_groups[group] / mc_integral
        bkg = raw_bkg / mc_integral
        data = raw_data / data_integral
        data_unc = data_unc / data_integral
        stat_unc = raw_stat_unc / mc_integral
        signals = {key: values / mc_integral for key, values in signals.items()}

        shape_syst2 = np.zeros(nbin, dtype=float)
        normalized_systematic_integrals = {}
        for source in sorted(available_systematic_sources):
            up_raw = systematic_totals[source]["up"]
            down_raw = systematic_totals[source]["down"]
            up_integral = float(np.sum(up_raw))
            down_integral = float(np.sum(down_raw))
            if not np.isfinite(up_integral) or up_integral <= 0:
                raise RuntimeError(
                    f"GCR {source}Up integral is not positive and finite: {up_integral}"
                )
            if not np.isfinite(down_integral) or down_integral <= 0:
                raise RuntimeError(
                    f"GCR {source}Down integral is not positive and finite: {down_integral}"
                )
            up = up_raw / up_integral
            down = down_raw / down_integral
            up_sum = float(np.sum(up))
            down_sum = float(np.sum(down))
            if not np.isclose(up_sum, 1.0, rtol=0.0, atol=1.0e-10):
                raise RuntimeError(f"GCR {source}Up normalized sum is {up_sum}")
            if not np.isclose(down_sum, 1.0, rtol=0.0, atol=1.0e-10):
                raise RuntimeError(f"GCR {source}Down normalized sum is {down_sum}")
            envelope = np.maximum(np.abs(up - bkg), np.abs(down - bkg))
            shape_syst2 += envelope * envelope
            normalized_systematic_integrals[source] = {
                "up": up_sum,
                "down": down_sum,
            }
        unc = np.sqrt(stat_unc * stat_unc + shape_syst2)
        uncertainty_label = (
            "MC stat.+shape syst. unc."
            if available_systematic_sources
            else "MC stat. unc."
        )

        normalized_data_sum = float(np.sum(data))
        normalized_mc_sum = float(np.sum(bkg))
        group_sum = np.zeros(nbin, dtype=float)
        for values in groups.values():
            group_sum += values
        group_match = bool(np.allclose(group_sum, bkg, rtol=0.0, atol=1.0e-12))
        if not np.isclose(normalized_data_sum, 1.0, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"GCR normalized data sum is {normalized_data_sum}")
        if not np.isclose(normalized_mc_sum, 1.0, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"GCR normalized MC sum is {normalized_mc_sum}")
        if not group_match:
            raise RuntimeError("GCR normalized process-group sum does not match total MC")

        raw_ratio = np.divide(
            raw_data,
            raw_bkg,
            out=np.full_like(raw_data, np.nan),
            where=raw_bkg > 0,
        )
        normalized_ratio = np.divide(
            data,
            bkg,
            out=np.full_like(data, np.nan),
            where=bkg > 0,
        )
        ratio_expected = raw_ratio * (mc_integral / data_integral)
        ratio_mask = np.isfinite(normalized_ratio) & np.isfinite(ratio_expected)
        ratio_match = bool(
            np.allclose(
                normalized_ratio[ratio_mask],
                ratio_expected[ratio_mask],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        if not ratio_match:
            raise RuntimeError("GCR normalized Data/MC ratio fails the global-factor check")

        process_fractions = {}
        process_fraction_match = True
        for group in GROUP_ORDER:
            raw_fraction = float(np.sum(raw_groups[group]) / mc_integral)
            normalized_fraction = float(np.sum(groups[group]))
            matched = bool(
                np.isclose(raw_fraction, normalized_fraction, rtol=0.0, atol=1.0e-12)
            )
            process_fraction_match = process_fraction_match and matched
            process_fractions[group] = {
                "raw_fraction": raw_fraction,
                "normalized_fraction": normalized_fraction,
                "matched": matched,
            }
        if not process_fraction_match:
            raise RuntimeError("GCR process fractions changed under normalization")

        unit_area_audit = {
            "status": "complete",
            "scope": "GCR only",
            "normalization": "data and total MC independently normalized over the entire plotted figure",
            "data_source": "EGamma",
            "raw_integrals": {
                "data": data_integral,
                "mc": mc_integral,
                "data_over_mc": data_integral / mc_integral,
                "process_groups": raw_legend_yields,
            },
            "normalized_integrals": {
                "data": normalized_data_sum,
                "mc": normalized_mc_sum,
            },
            "process_fractions": process_fractions,
            "systematic_shape_integrals": normalized_systematic_integrals,
            "checks": {
                "process_group_sum_matches_total_mc": group_match,
                "process_fractions_preserved": process_fraction_match,
                "ratio_global_factor_relation": ratio_match,
                "luminosity_uncertainty_omitted": True,
            },
            "uncertainty_model": "MC statistical uncertainty plus unit-area-normalized Up/Down shape envelopes in quadrature; luminosity omitted",
        }

    requested_width = max((float(block.get("figure_width", 0.0)) for block in blocks), default=0.0)
    figure_size = (12.0, 10.0) if reference_style else (max(12.0, nbin * 0.26, requested_width), 8.4)
    if figure_axes is None:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={"height_ratios": [3.2, 1.05 if reference_style else 1.1], "hspace": 0.04}, sharex=True)
    else:
        fig, ax, rax = figure_axes
    stack_inputs = []
    stack_weights = []
    stack_colors = []
    stack_labels = []
    legend_group_labels = {}

    def compact_yield(value: float) -> str:
        magnitude = abs(value)
        if magnitude >= 10.0:
            return f"{value:.0f}"
        if magnitude >= 1.0:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.2g}"

    for group in GROUP_ORDER:
        vals = groups[group]
        if np.any(vals > 0):
            stack_inputs.append(centers.copy())
            stack_weights.append(vals)
            stack_colors.append(GROUP_COLORS.get(group, "0.7"))
            base_label = group_labels.get(
                group,
                GROUP_DISPLAY_LABELS.get(group, group) if reference_style else group,
            )
            legend_group_labels[group] = (
                f"{base_label} ({compact_yield(raw_legend_yields[group])})"
                if show_yields
                else base_label
            )
            stack_labels.append(legend_group_labels[group])
    if stack_inputs:
        ax.hist(stack_inputs, bins=edges, weights=stack_weights, stacked=True, histtype="stepfilled", color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.7)
    lower = np.maximum(bkg - unc, 1.0e-12)
    upper = np.maximum(bkg + unc, 1.0e-12)
    if np.any(bkg > 0):
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0.0, alpha=0.65, label=uncertainty_label)
    if reference_style and np.any(bkg > 0):
        ax.stairs(bkg, edges, color="black", linewidth=1.4, zorder=6)
    legend_signal_labels = {}
    for spec in signal_specs:
        vals = signals.get(spec["key"])
        if vals is not None:
            signal_label = (
                f"{spec['label']} ({compact_yield(raw_signal_yields[spec['key']])})"
                if show_yields
                else spec["label"]
            )
            legend_signal_labels[spec["key"]] = signal_label
            outline_color = spec.get("outline_color")
            if outline_color:
                ax.hist(
                    centers,
                    bins=edges,
                    weights=vals,
                    histtype="step",
                    linewidth=float(spec.get("outline_linewidth", 4.4)),
                    linestyle="--",
                    color=outline_color,
                    alpha=float(spec.get("outline_alpha", 0.45)),
                    zorder=7,
                )
            ax.hist(
                centers,
                bins=edges,
                weights=vals,
                histtype="step",
                linewidth=2.8,
                linestyle="--",
                color=spec["color"],
                label=signal_label,
                zorder=8,
            )
    mask = data_mask & (data > 0)
    data_base_label = "Data" if reference_style else "DATA"
    if show_yields:
        data_legend_label = (
            f"{data_base_label} ({compact_yield(raw_legend_yields['Data'])})"
            if np.any(data_mask)
            else f"{data_base_label} (blinded)"
        )
    else:
        data_legend_label = data_base_label
    if not np.any(data_mask):
        data_legend_label = "_nolegend_"
    ax.errorbar(
        centers[mask],
        data[mask],
        xerr=xerr[mask],
        yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])),
        fmt="o",
        color="black",
        markersize=8.0 if reference_style else 5.5,
        capsize=4.5,
        capthick=1.4,
        elinewidth=1.4,
        label=data_legend_label,
        zorder=10,
    )
    if significance_panel:
        if significance_mode == "s_over_sqrt_b":
            significance_denominator = np.sqrt(np.maximum(bkg, 0.0))
        elif significance_mode == "s_over_sqrt_b_plus_unc2":
            significance_denominator = np.sqrt(np.maximum(bkg, 0.0) + unc**2)
        else:
            raise RuntimeError(
                f"unsupported significance definition: {significance_mode}"
            )
        for spec in signal_specs:
            vals = signals.get(spec["key"])
            if vals is None:
                continue
            significance = np.divide(
                vals,
                significance_denominator,
                out=(np.full_like(vals, np.nan) if any(block.get('undefined_significance_nan', False) for block in blocks)
                     else np.zeros_like(vals)),
                where=significance_denominator > 0.0,
            )
            rax.stairs(
                significance,
                edges,
                color=spec["color"],
                linewidth=2.6,
                linestyle="--",
            )
        rax.axhline(0.0, color="0.45", linewidth=1)
    else:
        ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
        ratio_err = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & data_mask)
        rmask = np.isfinite(ratio)
        rax.errorbar(
            centers[rmask],
            ratio[rmask],
            xerr=xerr[rmask],
            yerr=ratio_err[rmask],
            fmt="o",
            color="black",
            markersize=7.0 if reference_style else 4.5,
            capsize=4.5,
            capthick=1.4,
            elinewidth=1.4,
        )
        rel = np.divide(unc, bkg, out=np.zeros_like(unc), where=bkg > 0)
        rax.fill_between(edges, np.r_[1.0 - rel, 1.0 - rel[-1]], np.r_[1.0 + rel, 1.0 + rel[-1]], step="post", facecolor="0.82", edgecolor="0.15", hatch="////", linewidth=0.0, alpha=0.65)
        rax.axhline(1.0, color="0.45", linewidth=1)
    for axis in (ax, rax):
        axis.set_xmargin(0)
        if physical_edges is None:
            for boundary in boundaries[1:-1]:
                axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
            axis.set_xlim(0.5, nbin + 0.5)
        else:
            requested_left = blocks[0].get("xlim_left") if len(blocks) == 1 else None
            requested_right = blocks[0].get("xlim_right") if len(blocks) == 1 else None
            axis.set_xlim(
                float(edges[0] if requested_left is None else requested_left),
                float(edges[-1] if requested_right is None else requested_right),
            )
        axis.tick_params(which="major", direction="in", top=True, right=True, labelsize=22 if reference_style else 20, length=9)
        axis.tick_params(which="minor", direction="in", top=True, right=True, length=5)
        axis.minorticks_on()
    for start, end, label, block in zip(boundaries[:-1], boundaries[1:], labels, blocks):
        center = 0.5 * (start + end) + 0.5 if physical_edges is None else 0.5 * (float(edges[0]) + float(edges[-1]))
        if block.get("label_box"):
            label_axis = ax if block.get("category_labels_on_main") else rax
            label_axis.text(
                center,
                float(block.get("category_label_y", 0.5)),
                label,
                transform=label_axis.get_xaxis_transform(),
                ha="center",
                va=str(block.get("label_va", "center")),
                fontsize=float(block.get("label_fontsize", 15)),
                bbox={
                    "boxstyle": f"round,pad={float(block.get('label_box_pad', 0.28))}",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.96,
                },
                zorder=20,
            )
    main_annotations = {block['main_annotation'] for block in blocks if block.get('main_annotation')}
    if len(main_annotations) > 1:
        raise ValueError('Cannot mix global selection annotations')
    if main_annotations:
        annotation_style = next(block for block in blocks if block.get('main_annotation'))
        ax.text(float(annotation_style.get('main_annotation_x',.01)),
                float(annotation_style.get('main_annotation_y',.81)),
                next(iter(main_annotations)), transform=ax.transAxes,
                fontsize=float(annotation_style.get('main_annotation_fontsize',14)),
                fontweight=annotation_style.get('main_annotation_fontweight','normal'),
                ha='left', va='center', zorder=21)
    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *signals.values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    main_panel_ymax_factor = max(
        (float(block.get("main_panel_ymax_factor", 60.0)) for block in blocks),
        default=60.0,
    )
    if positive:
        if reference_style and not unit_area:
            ymax = 10.0 ** np.ceil(np.log10(max(max(positive) * main_panel_ymax_factor, 1.0)))
            ax.set_ylim(1.0e-1, ymax)
        else:
            floor = 1.0e-5 if unit_area else 0.03
            ax.set_ylim(
                max(floor, min(positive) * 0.1),
                max(max(positive) * main_panel_ymax_factor, 1.0),
            )
    else:
        ax.set_ylim(0.03, 1.0)
    ax.set_ylabel(
        "Normalized events"
        if unit_area
        else (main_ylabel or ("Events" if reference_style else "Events / bin")),
        fontsize=32 if reference_style else 30,
    )
    rax.set_ylabel(
        significance_ylabel if significance_panel else (ratio_ylabel or "Data/MC"),
        fontsize=30 if reference_style else 26,
    )
    rax.set_ylim(*lower_panel_ylim)
    rax.set_xlabel(xlabel, fontsize=32 if reference_style else 30, loc="right")
    annotations = [str(block.get("annotation") or "") for block in blocks if block.get("annotation")]
    show_reference_annotation = any(
        bool(block.get("show_annotation")) for block in blocks
    )
    if len(annotations) == 1 and (not reference_style or show_reference_annotation):
        annotation_block = next(
            block for block in blocks if block.get("annotation")
        )
        ax.text(
            float(annotation_block.get("annotation_x", 0.035)),
            float(annotation_block.get("annotation_y", 0.72)),
            annotations[0],
            transform=ax.transAxes,
            ha=str(annotation_block.get("annotation_ha", "left")),
            va="top",
            fontsize=20,
        )
    if physical_edges is not None and len(blocks) == 1:
        xlabels = blocks[0].get("xlabels") or []
        if len(xlabels) == nbin:
            rax.set_xticks(centers)
            rax.set_xticklabels(xlabels, fontsize=22 if reference_style else 16)
    elif physical_edges is None:
        xlabels = []
        for block in blocks:
            block_labels = block.get("xlabels") or []
            if len(block_labels) == int(block["nbin"]):
                xlabels.extend(block_labels)
            else:
                start = len(xlabels) + 1
                xlabels.extend(str(i) for i in range(start, start + int(block["nbin"])))
        rax.set_xticks(centers)
        label_fontsize = 12 if any("\n" in lab for lab in xlabels) else (13 if nbin > 24 else 16)
        rax.set_xticklabels(xlabels, fontsize=label_fontsize)
    shown_luminosity = LUMINOSITY_FB if luminosity_fb is None else luminosity_fb
    if x_tick_rotation:
        for tick in rax.get_xticklabels():
            tick.set_rotation(x_tick_rotation)
            tick.set_horizontalalignment('right')
    hep.cms.label(llabel="Work in progress", rlabel=rf"{shown_luminosity:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=ax)
    if reference_style:
        handles, legend_labels = ax.get_legend_handles_labels()
        desired_groups = [
            "VV+VVV",
            "Top",
            "DY",
            "Photon+jet",
            "W -> lv",
            "Z -> vv",
            "QCD Multijet",
            "Others",
        ]
        desired = [uncertainty_label]
        desired.extend(
            legend_group_labels[group]
            for group in desired_groups
            if group in legend_group_labels
        )
        desired.extend(
            legend_signal_labels[spec["key"]]
            for spec in signal_specs
            if spec["key"] in legend_signal_labels
        )
        desired.append(data_legend_label)
        ordered = [(handles[legend_labels.index(label)], label) for label in desired if label in legend_labels]
        if ordered:
            legend_fontsize = 13 if len(ordered) > 10 else 15
            ax.legend([item[0] for item in ordered], [item[1] for item in ordered], fontsize=legend_fontsize, ncol=legend_ncol or 3, frameon=False, columnspacing=1.2, handlelength=1.8, loc="upper center", bbox_to_anchor=(0.52, 0.995))
    else:
        ax.legend(fontsize=12, ncol=4, frameon=False, columnspacing=1.05, handlelength=2.0, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    outbase.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if reference_style else {"bbox_inches": "tight"}
    if figure_axes is None:
        fig.savefig(outbase.with_suffix(".png"), dpi=180, **save_kwargs)
        fig.savefig(outbase.with_suffix(".pdf"), **save_kwargs)
        plt.close(fig)
    return {
        "status": "complete",
        "name": outbase.name,
        "png": str(outbase.with_suffix(".png")),
        "pdf": str(outbase.with_suffix(".pdf")),
        "bins": nbin,
        "labels": labels,
        "signals": list(signals),
        "legend_yields_displayed": show_yields,
        "unit_area": unit_area,
        "unit_area_audit": unit_area_audit,
        "lower_panel": "signal_significance" if significance_panel else "data_over_mc",
        "lower_panel_ylim": list(lower_panel_ylim),
        "main_panel_ylim": [float(value) for value in ax.get_ylim()],
        "significance_definition": (
            (
                "S/sqrt(B)"
                if significance_mode == "s_over_sqrt_b"
                else "S/sqrt(B+sigma_B^2), with sigma_B equal to the plotted background uncertainty"
            )
            if significance_panel else None
        ),
        "dropped_input_bins_1based": list(
            blocks[0].get("dropped_input_bins_1based") or []
        ),
    }
