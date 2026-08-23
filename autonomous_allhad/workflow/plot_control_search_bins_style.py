#!/usr/bin/env python3
"""Draw canonical Run-3 control, signal, and search-bin distributions."""

from __future__ import annotations

import argparse
import json
import mmap
import shutil
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from background_process_groups import (
    BACKGROUND_DISPLAY_LABELS,
    BACKGROUND_PROCESS_ORDER,
    background_process_for_sample,
)
from autonomous_allhad.search_bin_categorization import (
    configured_bin_position_groups,
    configured_exclusive_mapping,
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
        "key": "mStop600_mLSP400",
        "label": '$m_{\\tilde{t}}=600$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=400$ GeV',
        "color": "#54FAFD",
    },
    {
        "key": "mStop900_mLSP700",
        "label": '$m_{\\tilde{t}}=900$ GeV, $m_{\\tilde{\\chi}^{0}_{1}}=700$ GeV',
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
LUMINOSITY_FB = 109.82
LUMINOSITY_RELATIVE_UNCERTAINTY = 0.016
PLOT_SYSTEMATIC_SOURCES = [
    "pileup",
    "electron_id",
    "electron_hlt",
    "muon_id",
    "muon_hlt",
    "photon_id",
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


def _json_value_end(buffer: mmap.mmap, start: int, limit: int) -> int:
    opening = buffer[start]
    if opening in (ord("{"), ord("[")):
        closing = ord("}") if opening == ord("{") else ord("]")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, limit):
            value = buffer[index]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
                continue
            if value == ord('"'):
                in_string = True
            elif value == opening:
                depth += 1
            elif value == closing:
                depth -= 1
                if depth == 0:
                    return index + 1
        raise ValueError(f"unterminated JSON value at byte {start}")
    if opening == ord('"'):
        escaped = False
        for index in range(start + 1, limit):
            value = buffer[index]
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                return index + 1
        raise ValueError(f"unterminated JSON string at byte {start}")
    index = start
    while index < limit and buffer[index] not in b",}]\r\n":
        index += 1
    return index


def _decode_slice(
    buffer: mmap.mmap, bounds: tuple[int, int]
) -> object:
    return json.loads(buffer[bounds[0] : bounds[1]])


def _value_start(
    buffer: mmap.mmap,
    name: str,
    start: int,
    end: int,
) -> int | None:
    marker = json.dumps(name).encode() + b":"
    member = buffer.find(marker, start, end)
    if member < 0:
        return None
    value_start = member + len(marker)
    while value_start < end and buffer[value_start] in b" \t\r\n":
        value_start += 1
    return value_start


def _section_bounds(
    buffer: mmap.mmap,
    name: str,
) -> tuple[int, int] | None:
    start = _value_start(buffer, name, 0, len(buffer))
    if start is None:
        return None
    boundary_keys = {
        "histograms": (
            "highdm_control_components",
            "search_bin_histograms",
        ),
        "search_bin_histograms": (
            "highdm_search_bin_components",
            "lowdm_variable_histograms",
        ),
        "lowdm_variable_histograms": ("highdm_variable_histograms",),
        "highdm_variable_histograms": (
            "normalization",
            "summary",
            "status",
        ),
    }.get(name, ("normalization", "summary", "status"))
    boundaries = []
    for key in boundary_keys:
        marker = b"," + json.dumps(key).encode() + b":"
        position = buffer.find(marker, start, len(buffer))
        if position >= 0:
            boundaries.append(position)
    end = min(boundaries) if boundaries else len(buffer) - 1
    return start, end


def _child_bounds_by_names(
    buffer: mmap.mmap,
    parent: tuple[int, int],
    names: list[str] | tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    located = []
    for name in names:
        marker = json.dumps(name).encode() + b":"
        position = buffer.find(marker, parent[0], parent[1])
        if position < 0:
            continue
        start = position + len(marker)
        while start < parent[1] and buffer[start] in b" \t\r\n":
            start += 1
        located.append((position, name, start))
    located.sort()
    output = {}
    for index, (_position, name, start) in enumerate(located):
        end = located[index + 1][0] if index + 1 < len(located) else parent[1]
        while end > start and buffer[end - 1] in b" \t\r\n,":
            end -= 1
        output[name] = (start, end)
    return output


def _kept_variations(variations: dict, *, signal: bool) -> dict:
    if signal:
        return {"nominal": variations.get("nominal") or {}}
    allowed = {"nominal"}
    for source in PLOT_SYSTEMATIC_SOURCES:
        allowed.update((source + "Up", source + "Down"))
    return {
        name: record
        for name, record in variations.items()
        if name in allowed
    }


def _sample_object(
    buffer: mmap.mmap,
    bounds: tuple[int, int],
    *,
    allow_signals: bool,
) -> dict:
    selected_signals = {
        "T2tt_" + spec["key"]
        for spec in (*SIGNAL_OVERLAYS, *LOWDM_SIGNAL_OVERLAYS)
    }
    selected_samples = (
        "data_obs",
        "VV",
        "ST",
        "TT",
        "DY",
        "GJ",
        "WtoLNu",
        "Zto2Nu",
        "QCD",
        *sorted(selected_signals),
    )
    output = {}
    for sample in selected_samples:
        signal = sample.startswith(("T2tt_", "T2bW_", "T2tb_"))
        if signal and (not allow_signals or sample not in selected_signals):
            continue
        sample_start = _value_start(buffer, sample, bounds[0], bounds[1])
        if sample_start is None:
            continue
        sample_bounds = (
            sample_start,
            _json_value_end(buffer, sample_start, bounds[1]),
        )
        variations = _decode_slice(buffer, sample_bounds)
        if not isinstance(variations, dict):
            raise ValueError(f"invalid sample histogram payload for {sample}")
        output[sample] = _kept_variations(variations, signal=signal)
    return output


def load_canonical_plot_payload(path: Path) -> dict:
    """Build a bounded-memory plotting projection from canonical hists.json."""
    payload: dict = {}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            for name in (
                "recoil_pt_bins",
                "lowdm_region_policy",
                "lowdm_region_variables",
                "lowdm_variable_specs",
                "highdm_distribution_regions",
                "highdm_distribution_variable_specs",
                "search_bin_schemes",
            ):
                start = _value_start(buffer, name, 0, len(buffer))
                if start is not None:
                    payload[name] = _decode_slice(
                        buffer,
                        (start, _json_value_end(buffer, start, len(buffer))),
                    )

            histogram_regions = (
                "LLCR",
                "QCDCR",
                "GCR",
                "DY2E",
                "DY2M",
                "LLCR_Nt0",
                "LLCR_Nt1",
                "QCDCR_Nt0",
                "QCDCR_Nt1",
                "GCR_Nt0",
                "GCR_Nt1",
                "DY2E_Nt0",
                "DY2E_Nt1",
                "DY2M_Nt0",
                "DY2M_Nt1",
                "HighDMVR_Nb1",
                "HighDMVR_Nb2",
                "HighDMVR_Nb3plus",
                "SR",
                "SR_Nt0",
                "SR_Nt1",
            )
            payload["histograms"] = {}
            histogram_bounds = _section_bounds(buffer, "histograms")
            if histogram_bounds is not None:
                histogram_members = _child_bounds_by_names(
                    buffer, histogram_bounds, histogram_regions
                )
                for region in histogram_regions:
                    bounds = histogram_members.get(region)
                    if bounds is not None:
                        payload["histograms"][region] = _sample_object(
                            buffer,
                            bounds,
                            allow_signals=region.startswith("SR"),
                        )

            schemes = (
                EXTENDED_AN17_RECOIL_SCHEME,
                "cat2_LLCR_lowDeltaM",
                "cat3_QCDCR_lowDeltaM",
                "cat4_GCR_lowDeltaM",
                "cat5_DY2E_lowDeltaM",
                "cat6_DY2M_lowDeltaM",
                "cat7_SR_lowDeltaM",
            )
            payload["search_bin_histograms"] = {}
            search_bounds = _section_bounds(buffer, "search_bin_histograms")
            if search_bounds is not None:
                search_members = _child_bounds_by_names(
                    buffer,
                    search_bounds,
                    list((payload.get("search_bin_schemes") or {}).keys()),
                )
                for scheme in schemes:
                    bounds = search_members.get(scheme)
                    if bounds is not None:
                        payload["search_bin_histograms"][
                            scheme
                        ] = _sample_object(
                            buffer,
                            bounds,
                            allow_signals=scheme
                            in {
                                EXTENDED_AN17_RECOIL_SCHEME,
                                "cat7_SR_lowDeltaM",
                            },
                        )

            payload["lowdm_variable_histograms"] = {}
            lowdm_variables = payload.get("lowdm_region_variables") or {}
            lowdm_scheme_by_region = {
                "LLCR": "cat2_LLCR_lowDeltaM",
                "QCDCR": "cat3_QCDCR_lowDeltaM",
                "GCR": "cat4_GCR_lowDeltaM",
                "DY2E": "cat5_DY2E_lowDeltaM",
                "DY2M": "cat6_DY2M_lowDeltaM",
                "SR": "cat7_SR_lowDeltaM",
            }
            lowdm_bounds = _section_bounds(
                buffer, "lowdm_variable_histograms"
            )
            if lowdm_bounds is not None:
                lowdm_scheme_members = _child_bounds_by_names(
                    buffer,
                    lowdm_bounds,
                    list(lowdm_scheme_by_region.values()),
                )
                for region, scheme in lowdm_scheme_by_region.items():
                    scheme_bounds = lowdm_scheme_members.get(scheme)
                    if scheme_bounds is None:
                        continue
                    variable_members = _child_bounds_by_names(
                        buffer,
                        scheme_bounds,
                        list(lowdm_variables.get(region) or []),
                    )
                    for variable in lowdm_variables.get(region) or []:
                        bounds = variable_members.get(variable)
                        if bounds is None:
                            continue
                        payload["lowdm_variable_histograms"].setdefault(
                            scheme, {}
                        )[variable] = _sample_object(
                            buffer,
                            bounds,
                            allow_signals=region == "SR",
                        )

            payload["highdm_variable_histograms"] = {}
            highdm_regions = payload.get("highdm_distribution_regions") or {}
            all_highdm_regions = {
                region for regions in highdm_regions.values() for region in regions
            }
            highdm_variables = list(
                (
                    payload.get("highdm_distribution_variable_specs") or {}
                ).keys()
            )
            highdm_bounds = _section_bounds(
                buffer, "highdm_variable_histograms"
            )
            if highdm_bounds is not None:
                highdm_region_members = _child_bounds_by_names(
                    buffer, highdm_bounds, sorted(all_highdm_regions)
                )
                for region in sorted(all_highdm_regions):
                    region_bounds = highdm_region_members.get(region)
                    if region_bounds is None:
                        continue
                    variable_members = _child_bounds_by_names(
                        buffer, region_bounds, highdm_variables
                    )
                    for variable in highdm_variables:
                        bounds = variable_members.get(variable)
                        if bounds is None:
                            continue
                        payload["highdm_variable_histograms"].setdefault(
                            region, {}
                        )[variable] = _sample_object(
                            buffer, bounds, allow_signals=False
                        )
    return payload


def load_plot_payload(path: Path) -> dict:
    if path.name == "hists.json":
        return load_canonical_plot_payload(path)
    return load_json(path)


def as_array(values: list[float] | None, nbin: int) -> np.ndarray:
    out = np.zeros(nbin, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float)
    out[: min(nbin, arr.size)] = arr[:nbin]
    return out


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


def recoil_record_from_payload(payload: dict, region: str) -> tuple[dict, int] | None:
    raw_bkg = (((payload.get("histograms") or {}).get("background") or {}).get("recoil_pt") or {}).get(region) or {}
    raw_data = (((payload.get("histograms") or {}).get("data") or {}).get("recoil_pt") or {}).get(region) or {}
    ref = next(iter(raw_bkg.values()), None) or next(iter(raw_data.values()), None)
    if not ref:
        return None
    nbin = max(0, len(ref.get("bin_edges") or []) - 1)
    if nbin <= 0:
        return None
    groups = {group: {"values": [0.0] * nbin, "sumw2": [0.0] * nbin} for group in GROUP_ORDER}
    bkg_total = np.zeros(nbin, dtype=float)
    bkg_stat2 = np.zeros(nbin, dtype=float)
    for proc, hist in raw_bkg.items():
        group = process_to_group(proc)
        vals = as_array(hist.get("values"), nbin)
        s2 = as_array(hist.get("sumw2"), nbin)
        bkg_total += vals
        bkg_stat2 += s2
        groups[group]["values"] = (np.asarray(groups[group]["values"], dtype=float) + vals).tolist()
        groups[group]["sumw2"] = (np.asarray(groups[group]["sumw2"], dtype=float) + s2).tolist()
    data = np.zeros(nbin, dtype=float)
    data_s2 = np.zeros(nbin, dtype=float)
    for hist in raw_data.values():
        data += as_array(hist.get("values"), nbin)
        data_s2 += as_array(hist.get("sumw2"), nbin)
    syst2 = np.zeros(nbin, dtype=float)
    variations = ((((payload.get("histogram_systematic_variations") or {}).get("background") or {}).get("recoil_pt") or {}).get(region) or {})
    for var in variations.values():
        up = as_array(var.get("up_delta"), nbin)
        down = as_array(var.get("down_delta"), nbin)
        syst2 += np.maximum(np.abs(up), np.abs(down)) ** 2
    syst2 += (0.016 * bkg_total) ** 2
    rec = {
        "status": "complete",
        "variable": "recoil_pt",
        "region_short": REGION_LABELS.get(region, region),
        "plot_bin_edges": ref.get("bin_edges") or [],
        "physics_bin_edges": ref.get("bin_edges") or [],
        "background_total": bkg_total.tolist(),
        "background_stat_unc": np.sqrt(bkg_stat2).tolist(),
        "background_syst_unc": np.sqrt(syst2).tolist(),
        "background_total_unc": np.sqrt(bkg_stat2 + syst2).tolist(),
        "background_by_group": {k: v for k, v in groups.items() if any(abs(x) > 0 for x in v["values"])},
        "data": data.tolist(),
        "data_stat_unc": np.sqrt(data_s2).tolist(),
        "data_blinded_in_plots": False,
    }
    return rec, nbin


def flatten_cr_templates(fit: dict, payload: dict) -> dict:
    templates = fit.get("templates") or {}
    records = []
    boundaries = [0]
    labels = []
    for region in REGION_ORDER:
        if region in {"cat2_LLCR_highDeltaM", "cat3_QCDCR_highDeltaM"}:
            built = recoil_record_from_payload(payload, region)
            if not built:
                continue
            rec, nbin = built
        else:
            rec = templates.get(region) or {}
            values = rec.get("background_total") or []
            nbin = len(values)
            if rec.get("status") != "complete" or nbin == 0:
                continue
        records.append((region, rec, nbin))
        boundaries.append(boundaries[-1] + nbin)
        labels.append(REGION_LABELS.get(region, rec.get("region_short") or region))

    nbin_total = boundaries[-1]
    groups = {group: np.zeros(nbin_total, dtype=float) for group in GROUP_ORDER}
    bkg_total = np.zeros(nbin_total, dtype=float)
    bkg_unc = np.zeros(nbin_total, dtype=float)
    data = np.zeros(nbin_total, dtype=float)
    data_unc = np.zeros(nbin_total, dtype=float)
    offset = 0
    for _, rec, nbin in records:
        slc = slice(offset, offset + nbin)
        bkg_total[slc] = as_array(rec.get("background_total"), nbin)
        bkg_unc[slc] = as_array(rec.get("background_total_unc"), nbin)
        data[slc] = as_array(rec.get("data"), nbin)
        data_unc[slc] = as_array(rec.get("data_stat_unc"), nbin)
        for group in GROUP_ORDER:
            group_rec = (rec.get("background_by_group") or {}).get(group) or {}
            groups[group][slc] = as_array(group_rec.get("values"), nbin)
        offset += nbin

    return {
        "records": records,
        "boundaries": boundaries,
        "labels": labels,
        "groups": groups,
        "background": bkg_total,
        "background_unc": bkg_unc,
        "data": data,
        "data_unc": data_unc,
    }


def boosted_search_bins(payload: dict, signal_payload: dict) -> dict:
    bins = (payload.get("search_bins") or {}).get(SEARCH_BIN_SCHEME) or {}
    signal_bins = (signal_payload.get("yields") or {}).get(SEARCH_BIN_SCHEME) or {}
    names = list(bins)
    nbin = len(names)
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_unc2 = np.zeros(nbin, dtype=float)
    for idx, name in enumerate(names):
        for proc, rec in (bins.get(name) or {}).items():
            val = float(rec.get("normalized_weighted") or 0.0)
            s2 = float(rec.get("normalized_sumw2") or 0.0)
            if rec.get("kind") == "data":
                data[idx] += val
                data_unc2[idx] += s2
            elif rec.get("kind") == "background":
                group = process_to_group(proc)
                groups[group][idx] += val
                stat2[idx] += s2
    signals = {}
    for spec in SIGNAL_OVERLAYS:
        vals = np.zeros(nbin, dtype=float)
        for idx, name in enumerate(names):
            vals[idx] = float(((signal_bins.get(name) or {}).get(spec["key"]) or {}).get("normalized_weighted") or 0.0)
        if np.any(vals > 0):
            signals[spec["key"]] = vals
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    lumi_unc = 0.016 * bkg
    # The current preview stores nominal boosted search-bin yields, but not per-search-bin shape variations.
    # Use MC stat plus lumi here; fit-template CR bins retain the full stored stat+syst band.
    unc = np.sqrt(stat2 + lumi_unc * lumi_unc)
    return {
        "names": names,
        "groups": groups,
        "background": bkg,
        "background_unc": unc,
        "data": data,
        "data_unc": np.sqrt(data_unc2),
        "signals": signals,
        "uncertainty_note": "SR boosted_an_17 search-bin band uses MC stat + Lumi_2024 only; search-bin shape variations are not stored in this partial preview payload.",
    }


def concat(cr: dict, sr: dict) -> dict:
    n_cr = len(cr["background"])
    n_sr = len(sr["background"])
    groups = {group: np.r_[cr["groups"].get(group, np.zeros(n_cr)), sr["groups"].get(group, np.zeros(n_sr))] for group in GROUP_ORDER}
    data_mask = np.r_[np.ones(n_cr, dtype=bool), np.zeros(n_sr, dtype=bool)]
    signals = {}
    for spec in SIGNAL_OVERLAYS:
        key = spec["key"]
        sr_vals = sr["signals"].get(key)
        if sr_vals is not None:
            signals[key] = np.r_[np.zeros(n_cr, dtype=float), sr_vals]
    return {
        "groups": groups,
        "background": np.r_[cr["background"], sr["background"]],
        "background_unc": np.r_[cr["background_unc"], sr["background_unc"]],
        "data": np.r_[cr["data"], sr["data"]],
        "data_unc": np.r_[cr["data_unc"], sr["data_unc"]],
        "data_mask": data_mask,
        "signals": signals,
        "boundaries": cr["boundaries"] + [cr["boundaries"][-1] + n_sr],
        "labels": cr["labels"] + ["SR - BLIND"],
        "sr_search_bins": sr["names"],
        "sr_uncertainty_note": sr["uncertainty_note"],
    }


def draw(fit_path: Path, payload_path: Path, signal_searchbin_path: Path, outbase: Path) -> dict:
    reference_style = False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    fit = load_json(fit_path)
    payload = load_json(payload_path)
    signal_payload = load_json(signal_searchbin_path) if signal_searchbin_path.exists() else {}
    cr = flatten_cr_templates(fit, payload)
    sr = boosted_search_bins(payload, signal_payload)
    flat = concat(cr, sr)

    nbin = len(flat["background"])
    if nbin <= 0:
        raise RuntimeError("No complete bins found.")
    centers = np.arange(1, nbin + 1, dtype=float)
    edges = np.arange(0.5, nbin + 1.5, 1.0)

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(18, 8.7), gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04}, sharex=True)

    stack_inputs = []
    stack_weights = []
    stack_colors = []
    stack_labels = []
    for group in GROUP_ORDER:
        vals = flat["groups"].get(group)
        if vals is None or not np.any(vals > 0):
            continue
        stack_inputs.append(centers.copy())
        stack_weights.append(vals)
        stack_colors.append(GROUP_COLORS.get(group, "0.7"))
        stack_labels.append(group)
    if stack_inputs:
        ax.hist(stack_inputs, bins=edges, weights=stack_weights, stacked=True, histtype="stepfilled", color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.7)

    bkg = flat["background"]
    unc = flat["background_unc"]
    lower = np.maximum(bkg - unc, 1.0e-12)
    upper = np.maximum(bkg + unc, 1.0e-12)
    if np.any(bkg > 0):
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", facecolor="0.85", edgecolor="0.35", hatch="////", linewidth=0.0, alpha=0.35, label="MC stat+syst unc.")

    for spec in SIGNAL_OVERLAYS:
        vals = flat["signals"].get(spec["key"])
        if vals is None:
            continue
        ax.hist(centers, bins=edges, weights=vals, histtype="step", linewidth=2.0, color=spec["color"], label=spec["label"])

    data = flat["data"]
    data_unc = flat["data_unc"]
    mask = flat["data_mask"] & (data > 0)
    if np.any(mask):
        ax.errorbar(centers[mask], data[mask], yerr=np.where(data_unc[mask] > 0, data_unc[mask], poisson_unc(data[mask])), fmt="o", color="black", markersize=4, label="Data" if reference_style else "DATA", zorder=10)

    ratio = np.divide(data, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    ratio_err = np.divide(data_unc, bkg, out=np.full_like(data, np.nan), where=(bkg > 0) & flat["data_mask"])
    rmask = np.isfinite(ratio)
    rax.errorbar(centers[rmask], ratio[rmask], yerr=ratio_err[rmask], fmt="o", color="black", markersize=3)
    rel = np.divide(unc, bkg, out=np.full_like(unc, np.nan), where=bkg > 0)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    rax.fill_between(edges, np.r_[1.0 - rel, 1.0 - rel[-1]], np.r_[1.0 + rel, 1.0 + rel[-1]], step="post", facecolor="0.85", edgecolor="none", alpha=0.6)
    rax.axhline(1.0, color="0.45", linewidth=1)

    for axis in (ax, rax):
        for boundary in flat["boundaries"][1:-1]:
            axis.axvline(boundary + 0.5, color="black", linewidth=1.2)
        for boundary in range(1, nbin):
            if boundary not in flat["boundaries"]:
                axis.axvline(boundary + 0.5, color="0.65", linestyle=":", linewidth=0.8, zorder=0)
        axis.set_xlim(0.5, nbin + 0.5)
        axis.tick_params(which="major", direction="in", top=True, right=True, labelsize=20, length=9)
        axis.tick_params(which="minor", direction="in", top=True, right=True, length=5)
        axis.minorticks_on()


    positive = []
    for arr in [bkg + unc, data[mask] if np.any(mask) else np.array([]), *flat["signals"].values()]:
        arr = np.asarray(arr, dtype=float)
        positive.extend(arr[arr > 0].tolist())
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(0.03, min(positive) * 0.1), max(max(positive) * 60, 1.0))
    else:
        ax.set_ylim(0.03, 1.0)
    ax.set_ylabel("Events / bin", fontsize=30)
    rax.set_ylabel("Data/MC", fontsize=26)
    rax.set_ylim(0, 2)
    rax.set_xlabel("Bin", fontsize=30, loc="right")
    rax.set_xticks(centers)
    rax.set_xticklabels([str(i) for i in range(1, nbin + 1)], fontsize=13)
    hep.cms.label(llabel="Work in progress", rlabel=rf"{LUMINOSITY_FB:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=ax)
    ax.legend(fontsize=12, ncol=4, frameon=False, columnspacing=1.05, handlelength=2.0, loc="upper center", bbox_to_anchor=(0.5, 0.995))

    outbase.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if reference_style else {"bbox_inches": "tight"}
    fig.savefig(outbase.with_suffix(".png"), dpi=180, **save_kwargs)
    fig.savefig(outbase.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
    return {
        "status": "complete",
        "name": outbase.name,
        "png": str(outbase.with_suffix(".png")),
        "pdf": str(outbase.with_suffix(".pdf")),
        "bins": nbin,
        "control_bins": int(cr["boundaries"][-1]),
        "sr_search_bins": len(sr["names"]),
        "sr_search_bin_names": sr["names"],
        "signals": list(flat["signals"]),
        "sr_uncertainty_note": flat["sr_uncertainty_note"],
    }



FLAT_REGION_LABELS = {
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
    "GCR": "GCR",
    "DY2E": "DY2E",
    "DY2M": "DY2M",
    "HighDMVR_Nb1": "High-dM VR\n" + r"$N_{b}=1$",
    "HighDMVR_Nb2": "High-dM VR\n" + r"$N_{b}=2$",
    "HighDMVR_Nb3plus": "High-dM VR\n" + r"$N_{b}\geq3$",
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


def flat_hist_record(payload: dict, region: str, allow_signal: bool) -> dict | None:
    raw = (payload.get("histograms") or {}).get(region) or {}
    if not raw:
        return None
    nbin = 0
    for rec in raw.values():
        nominal = rec.get("nominal") or rec
        nbin = max(nbin, len(nominal.get("sumw") or []))
    if nbin <= 0:
        return None
    recoil_edges = payload.get("recoil_pt_bins") or []
    if len(recoil_edges) != nbin + 1:
        recoil_edges = []
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in SIGNAL_OVERLAYS:
                    flat_key = "T2tt_" + spec["key"].replace("mStop", "mStop").replace("_mLSP", "_mLSP")
                    if sample == flat_key:
                        signals[spec["key"]] += vals
        else:
            group = process_to_group(sample)
            groups[group] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    syst2 += (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    result = {
        "groups": groups,
        "background": bkg,
        "background_unc": np.sqrt(stat2 + syst2),
        "background_stat_unc": np.sqrt(stat2),
        "background_systematic_totals": background_systematic_totals(raw, nbin),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "label": FLAT_REGION_LABELS.get(region, region),
        "nbin": nbin,
        "edges": recoil_edges,
        "unit_area": region == "GCR" or region.startswith("GCR_"),
        "physics_scope": "GCR" if region == "GCR" or region.startswith("GCR_") else region,
        "reference_style": region == "GCR" or region.startswith("GCR_"),
    }
    return result


def flat_search_record(payload: dict, scheme: str, label: str, allow_signal: bool, signal_overlays: list[dict] | None = None) -> dict | None:
    raw = (payload.get("search_bin_histograms") or {}).get(scheme) or {}
    if not raw:
        return None
    nbin = 0
    for rec in raw.values():
        nominal = rec.get("nominal") or rec
        nbin = max(nbin, len(nominal.get("sumw") or []))
    if nbin <= 0:
        return None
    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signal_overlays = signal_overlays or SIGNAL_OVERLAYS
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in signal_overlays}
    for sample, rec in raw.items():
        vals, s2 = flat_values(rec, nbin)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in signal_overlays:
                    if sample == "T2tt_" + spec["key"]:
                        signals[spec["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    syst2 = background_systematic_variance(raw, nbin)
    unc = np.sqrt(stat2 + syst2 + (LUMINOSITY_RELATIVE_UNCERTAINTY * bkg) ** 2)
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    is_gcr = scheme == "cat4_GCR_lowDeltaM"
    result = {
        "groups": groups,
        "background": bkg,
        "background_unc": unc,
        "background_stat_unc": np.sqrt(stat2),
        "background_systematic_totals": background_systematic_totals(raw, nbin),
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "signal_specs": signal_overlays,
        "label": label,
        "nbin": nbin,
        "unit_area": is_gcr,
        "physics_scope": "GCR" if is_gcr else scheme,
        "reference_style": is_gcr,
    }
    return result


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
    "lowdm_met_sqrt_ht": r"$p_{T}^{miss}/\sqrt{H_{T}}$",
    "lowdm_isr_pt": r"$p_{T}^{\mathrm{ISR}}$ (GeV)",
    "lowdm_isr_dphi": r"$\Delta\phi(\mathrm{ISR},p_{T}^{miss})$",
    "lowdm_ptb": r"$p_{T}^{b}$ (GeV)",
    "n_lowdm_isr": r"$N_{\mathrm{ISR}}$",
    "mee": r"$m_{ee}$ (GeV)",
    "mmm": r"$m_{\mu\mu}$ (GeV)",
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


def lowdm_variable_record(payload: dict, scheme: str, variable: str, label: str, allow_signal: bool) -> dict | None:
    raw = (((payload.get("lowdm_variable_histograms") or {}).get(scheme) or {}).get(variable) or {})
    if not raw:
        return None
    spec = ((payload.get("lowdm_variable_specs") or {}).get(variable) or {})
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
                    if sample == "T2tt_" + sig["key"]:
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
        "xlabel": VARIABLE_XLABELS.get(variable, spec.get("xlabel") or variable),
        "variable": variable,
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
    "HighDMVR_Nb1": r"High-$\Delta m$ VR, $N_{b}=1$",
    "HighDMVR_Nb2": r"High-$\Delta m$ VR, $N_{b}=2$",
    "HighDMVR_Nb3plus": r"High-$\Delta m$ VR, $N_{b}\geq3$",
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


def highdm_variable_record(payload: dict, region: str, variable: str) -> dict | None:
    raw = ((((payload.get("highdm_variable_histograms") or {}).get(region) or {}).get(variable)) or {})
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


def partial_an17_search_record(payload: dict, label: str, split_bins: list[int], allow_signal: bool) -> dict | None:
    raw_inc = (payload.get("search_bin_histograms") or {}).get("boosted_an_17_SR") or {}
    raw_nt1 = (payload.get("search_bin_histograms") or {}).get("boosted_an_17_SR_Nt1") or {}
    if not raw_inc or not raw_nt1:
        return None
    labels = (((payload.get("search_bin_schemes") or {}).get("boosted_an_17_SR") or {}).get("bin_labels") or [])
    nbin_in = len(labels)
    if nbin_in <= 0:
        for rec in raw_inc.values():
            nominal = rec.get("nominal") or rec
            nbin_in = max(nbin_in, len(nominal.get("sumw") or []))
    if nbin_in <= 0:
        return None
    split = set(split_bins)
    nbin = nbin_in + sum(1 for idx in range(1, nbin_in + 1) if idx in split)

    def expanded(sample: str) -> tuple[np.ndarray, np.ndarray]:
        inc_vals, inc_s2 = flat_values(raw_inc.get(sample) or {}, nbin_in)
        nt1_vals, nt1_s2 = flat_values(raw_nt1.get(sample) or {}, nbin_in)
        nt0_vals = np.maximum(inc_vals - nt1_vals, 0.0)
        nt0_s2 = np.maximum(inc_s2 - nt1_s2, 0.0)
        vals = []
        s2 = []
        for idx in range(nbin_in):
            if idx + 1 in split:
                vals.extend([float(nt0_vals[idx]), float(nt1_vals[idx])])
                s2.extend([float(nt0_s2[idx]), float(nt1_s2[idx])])
            else:
                vals.append(float(inc_vals[idx]))
                s2.append(float(inc_s2[idx]))
        return np.asarray(vals, dtype=float), np.asarray(s2, dtype=float)

    groups = {group: np.zeros(nbin, dtype=float) for group in GROUP_ORDER}
    stat2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data2 = np.zeros(nbin, dtype=float)
    signals = {spec["key"]: np.zeros(nbin, dtype=float) for spec in SIGNAL_OVERLAYS}
    for sample in sorted(raw_inc):
        vals, s2 = expanded(sample)
        if sample == "data_obs":
            data += vals
            data2 += s2
        elif is_signal_sample(sample):
            if allow_signal:
                for spec in SIGNAL_OVERLAYS:
                    if sample == "T2tt_" + spec["key"]:
                        signals[spec["key"]] += vals
        else:
            groups[process_to_group(sample)] += vals
            stat2 += s2
    bkg = np.zeros(nbin, dtype=float)
    for vals in groups.values():
        bkg += vals
    unc = np.sqrt(stat2 + (0.016 * bkg) ** 2)
    signals = {key: vals for key, vals in signals.items() if np.any(vals > 0)}
    xlabels = []
    for idx in range(1, nbin_in + 1):
        if idx in split:
            xlabels.extend([rf"{idx}\n$N_{{t}}=0$", rf"{idx}\n$N_{{t}}\geq1$"])
        else:
            xlabels.append(str(idx))
    return {
        "groups": groups,
        "background": bkg,
        "background_unc": unc,
        "data": data,
        "data_unc": np.sqrt(data2),
        "signals": signals,
        "label": label,
        "nbin": nbin,
        "split_bins_1based": split_bins,
        "xlabels": xlabels,
    }


def apply_configured_search_bin_merges(
    payload: dict,
    scheme_name: str,
    configuration: dict,
) -> dict:
    """Apply configured final-bin merges to the bounded plotting projection.

    This operates only on the canonical histogram projection already loaded in
    memory.  It preserves the correlation of each systematic variation by
    adding the source bins before plot uncertainties are evaluated.
    """
    scheme = (payload.get("search_bin_schemes") or {}).get(scheme_name) or {}
    raw_labels = [str(value) for value in (scheme.get("bin_labels") or [])]
    source_count = len(configured_exclusive_mapping(configuration))
    if len(raw_labels) != source_count:
        raise RuntimeError(
            f"{scheme_name} configuration/source mismatch: "
            f"{source_count} configured source bins but {len(raw_labels)} labels"
        )
    position_groups = configured_bin_position_groups(configuration)
    if len(position_groups) == source_count:
        return {
            "source_bin_count": source_count,
            "final_bin_count": source_count,
            "bin_merges_1based": [],
        }

    def rebin_leaf(leaf: dict) -> dict:
        rebinned = dict(leaf)
        for field in ("entries", "sumw", "sumw2"):
            if field not in leaf:
                continue
            values = leaf.get(field) or []
            if len(values) != source_count:
                raise RuntimeError(
                    f"{scheme_name} {field} has {len(values)} bins; "
                    f"expected {source_count}"
                )
            rebinned[field] = [
                float(sum(float(values[position]) for position in positions))
                for positions in position_groups
            ]
        return rebinned

    raw_histograms = (payload.get("search_bin_histograms") or {}).get(scheme_name) or {}
    rebinned_histograms = {}
    for sample, record in raw_histograms.items():
        if isinstance(record, dict) and "sumw" in record:
            rebinned_histograms[sample] = rebin_leaf(record)
            continue
        rebinned_histograms[sample] = {
            name: rebin_leaf(leaf)
            if isinstance(leaf, dict) and "sumw" in leaf
            else leaf
            for name, leaf in record.items()
        }
    payload.setdefault("search_bin_histograms", {})[scheme_name] = rebinned_histograms

    rebinned_scheme = dict(scheme)
    rebinned_scheme["bin_labels"] = [
        "__plus__".join(raw_labels[position] for position in positions)
        for positions in position_groups
    ]
    rebinned_scheme["plot_bin_merges"] = {
        "source_bin_count": source_count,
        "final_bin_count": len(position_groups),
        "bin_merges_1based": list(configuration.get("bin_merges_1based") or []),
    }
    payload.setdefault("search_bin_schemes", {})[scheme_name] = rebinned_scheme
    return rebinned_scheme["plot_bin_merges"]


def selected_an17_recoil_blocks(payload: dict, scheme_name: str) -> list[dict]:
    rec = flat_search_record(payload, scheme_name, "selected AN17 recoil", allow_signal=True)
    if not rec:
        return []
    scheme = (payload.get("search_bin_schemes") or {}).get(scheme_name) or {}
    raw_labels = scheme.get("bin_labels") or []
    if len(raw_labels) != int(rec["nbin"]):
        raise RuntimeError(
            f"{scheme_name} label/bin mismatch: "
            f"{len(raw_labels)} != {int(rec['nbin'])}"
        )

    def category_key(raw_label: str) -> str:
        if "Nb3plus_T1_W1" in raw_label and "Nb3plus_T2_W0" in raw_label:
            return "merged_high_nt"
        if "__recoil_" in raw_label:
            return raw_label.split("__recoil_", 1)[0]
        first = raw_label.split("__plus__", 1)[0]
        category = first.split("_recoil_", 1)[0]
        if category.startswith("NT0_"):
            category = category[len("NT0_") :]
        elif category.startswith("AN17_"):
            category = category.split("_", 2)[2]
        return category

    category_layout: list[tuple[str, int]] = []
    for raw_label in raw_labels:
        category = category_key(str(raw_label))
        if category_layout and category_layout[-1][0] == category:
            previous, size = category_layout[-1]
            category_layout[-1] = (previous, size + 1)
        else:
            category_layout.append((category, 1))

    blocks = []
    offset = 0
    for category, n_recoil in category_layout:
        slc = slice(offset, offset + n_recoil)
        if slc.stop > int(rec["nbin"]):
            break
        label = RESOLVED_CATEGORY_LABELS.get(
            category,
            SELECTED_AN17_CATEGORY_LABELS.get(category, category),
        )
        raw_block_labels = [str(value) for value in raw_labels[slc]]
        if (
            raw_block_labels
            and all("__Nres0" in value for value in raw_block_labels)
            and label.count("\n") < 2
        ):
            label = label + "\n" + r"$N_{res}=0$"
        block = {
            "groups": {group: vals[slc] for group, vals in rec["groups"].items()},
            "background": rec["background"][slc],
            "background_unc": rec["background_unc"][slc],
            "data": rec["data"][slc],
            "data_unc": rec["data_unc"][slc],
            "signals": {key: vals[slc] for key, vals in rec.get("signals", {}).items()},
            "label": label,
            "nbin": n_recoil,
            "xlabels": [],
            "blind_data": True,
            "label_box": True,
            "label_fontsize": 12.0,
            "label_box_pad": 0.18,
            "category_labels_on_main": True,
            "category_label_y": 0.72,
            "main_panel_ymax_factor": 600.0,
            "significance_panel": True,
            "figure_width": 22.0,
            "category_key": category,
        }
        blocks.append(block)
        offset += n_recoil
    return blocks


def lowdm_nsv_inclusive_blocks(payload: dict, scheme_name: str) -> list[dict]:
    rec = flat_search_record(
        payload,
        scheme_name,
        "Low-dM SR",
        allow_signal=True,
        signal_overlays=LOWDM_SIGNAL_OVERLAYS,
    )
    if not rec:
        return []
    scheme = (payload.get("search_bin_schemes") or {}).get(scheme_name) or {}
    category_sizes = scheme.get("category_sizes") or LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES
    normalized_sizes = [(str(category), int(size)) for category, size in category_sizes]
    if sum(size for _, size in normalized_sizes) != int(rec["nbin"]):
        return []
    blocks = []
    offset = 0
    for category, size in normalized_sizes:
        slc = slice(offset, offset + size)
        blocks.append({
            "groups": {group: vals[slc] for group, vals in rec["groups"].items()},
            "background": rec["background"][slc],
            "background_unc": rec["background_unc"][slc],
            "data": rec["data"][slc],
            "data_unc": rec["data_unc"][slc],
            "signals": {key: vals[slc] for key, vals in rec.get("signals", {}).items()},
            "signal_specs": LOWDM_SIGNAL_OVERLAYS,
            "label": LOWDM_NSV_INCLUSIVE_CATEGORY_LABELS.get(category, category),
            "nbin": size,
            "xlabels": [],
            "blind_data": True,
            "label_box": True,
            "label_fontsize": 10.2,
            "label_box_pad": 0.42,
            "category_labels_on_main": True,
            "category_label_y": 0.72,
            "main_panel_ymax_factor": 600.0,
            "significance_panel": True,
            "figure_width": 16.4,
        })
        offset += size
    return blocks


def draw_flat_blocks(
    blocks: list[dict],
    outbase: Path,
    xlabel: str = "Bin",
    reference_style: bool = False,
    show_yields: bool = False,
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

    unit_area = bool(blocks) and all(
        bool(block.get("unit_area")) and block.get("physics_scope") == "GCR"
        for block in blocks
    )
    unit_area_audit = None
    uncertainty_label = "Stat. syst. unc" if reference_style else "MC stat+syst unc."
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
            if reference_style
            else "MC stat.+shape syst. unc."
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
    fig, (ax, rax) = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={"height_ratios": [3.2, 1.05 if reference_style else 1.1], "hspace": 0.04}, sharex=True)
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
    significance_by_signal = {}
    if significance_panel:
        significance_denominator = np.sqrt(np.maximum(bkg, 0.0) + unc**2)
        for spec in signal_specs:
            vals = signals.get(spec["key"])
            if vals is None:
                continue
            significance = np.divide(
                vals,
                significance_denominator,
                out=np.zeros_like(vals),
                where=significance_denominator > 0.0,
            )
            significance_by_signal[spec["key"]] = significance
            rax.stairs(
                significance,
                edges,
                color=spec["color"],
                linewidth=2.6,
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
                va="center",
                fontsize=float(block.get("label_fontsize", 15)),
                bbox={
                    "boxstyle": f"round,pad={float(block.get('label_box_pad', 0.28))}",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.96,
                },
                zorder=20,
            )
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
            ymax = 10.0 ** np.ceil(
                np.log10(max(max(positive) * main_panel_ymax_factor, 1.0))
            )
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
        "Normalized events" if unit_area else ("Events" if reference_style else "Events / bin"),
        fontsize=32 if reference_style else 30,
    )
    rax.set_ylabel("Significance" if significance_panel else "Data/MC", fontsize=30 if reference_style else 26)
    if significance_panel:
        max_significance = max(
            (float(np.max(values)) for values in significance_by_signal.values()),
            default=0.0,
        )
        rax.set_ylim(0.0, max(0.05, 1.18 * max_significance))
    else:
        rax.set_ylim(0, 2)
    rax.set_xlabel(xlabel, fontsize=32 if reference_style else 30, loc="right")
    annotations = [str(block.get("annotation") or "") for block in blocks if block.get("annotation")]
    if len(annotations) == 1 and not reference_style:
        ax.text(0.035, 0.72, annotations[0], transform=ax.transAxes, ha="left", va="top", fontsize=20)
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
    hep.cms.label(llabel="Work in progress", rlabel=rf"{LUMINOSITY_FB:.2f} fb$^{{-1}}$ (13.6 TeV)", ax=ax)
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
            ax.legend([item[0] for item in ordered], [item[1] for item in ordered], fontsize=legend_fontsize, ncol=3, frameon=False, columnspacing=1.2, handlelength=1.8, loc="upper center", bbox_to_anchor=(0.52, 0.995))
    else:
        ax.legend(fontsize=12, ncol=4, frameon=False, columnspacing=1.05, handlelength=2.0, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    outbase.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if reference_style else {"bbox_inches": "tight"}
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
        "lower_panel": "signal_significance" if significance_panel else "data_over_mc",
        "significance_definition": (
            "S/sqrt(B+sigma_B^2), with sigma_B equal to the plotted background uncertainty"
            if significance_panel
            else None
        ),
        "unit_area": unit_area,
        "unit_area_audit": unit_area_audit,
    }


def draw_highdm_distribution_report(
    payload_path: Path,
    output_dir: Path,
    year: str,
    dy_rz_manifest: Path | None = None,
    only_region: str | None = None,
    only_variable: str | None = None,
) -> dict:
    payload = load_plot_payload(payload_path)
    # High-dM one-dimensional distribution plots intentionally do not draw
    # signal overlays.  Drop those records once, before the repeated region /
    # variable aggregation, so large signal grids do not dominate memory and
    # I/O while leaving every plotted background and data value unchanged.
    signal_records_pruned = 0
    for variable_map in (payload.get("highdm_variable_histograms") or {}).values():
        for raw in variable_map.values():
            signal_samples = [sample for sample in raw if is_signal_sample(sample)]
            for sample in signal_samples:
                del raw[sample]
            signal_records_pruned += len(signal_samples)
    dy_rz_application = (
        apply_dy_rz(payload, dy_rz_manifest) if dy_rz_manifest else None
    )
    region_groups = payload.get("highdm_distribution_regions") or {}
    variable_specs = payload.get("highdm_distribution_variable_specs") or {}
    plots = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, metadata_key in [("CR", "control"), ("VR", "validation"), ("SR", "signal_categories")]:
        for region in region_groups.get(metadata_key) or []:
            if only_region and region != only_region:
                continue
            for variable in variable_specs:
                if only_variable and variable != only_variable:
                    continue
                record = highdm_variable_record(payload, region, variable)
                if not record:
                    continue
                slug = region.lower().replace("highdm", "highdm_").replace("__", "_")
                name = f"{kind.lower()}_{slug}_{variable}"
                plot = draw_flat_blocks(
                    [record],
                    output_dir / name,
                    xlabel=record["xlabel"],
                    reference_style=True,
                    show_yields=True,
                )
                plot.update({
                    "year": year,
                    "kind": kind,
                    "region": region,
                    "region_label": record["label"],
                    "variable": variable,
                    "xlabel": record["xlabel"],
                    "blind_data": bool(record.get("blind_data")),
                    "figure_size_inches": [12.0, 10.0],
                })
                plots.append(plot)
    summary = {
        "status": "complete",
        "year": year,
        "luminosity_fb": LUMINOSITY_FB,
        "luminosity_relative_uncertainty": LUMINOSITY_RELATIVE_UNCERTAINTY,
        "background_systematic_sources": list(PLOT_SYSTEMATIC_SOURCES),
        "uncertainty_model": (
            "MC stat plus unit-area-normalized Up/Down shape envelopes in quadrature; luminosity omitted"
            if plots and all(plot.get("unit_area") for plot in plots)
            else "MC stat plus luminosity plus per-source max(abs(Up-Nominal), abs(Down-Nominal)) envelopes in quadrature"
        ),
        "luminosity_uncertainty_applied": not (
            plots and all(plot.get("unit_area") for plot in plots)
        ),
        "source": str(payload_path),
        "output_dir": str(output_dir),
        "plot_count": len(plots),
        "plots": plots,
        "signal_records_pruned_before_render": signal_records_pruned,
        "dy_rz_application": dy_rz_application,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_highdm_distribution_webpage(
    summary_2024: Path,
    summary_2025: Path | None,
    docs_dir: Path,
    flat_summary_2024: Path | None = None,
    flat_summary_2025: Path | None = None,
    impact_png: Path | None = None,
    impact_pdf: Path | None = None,
    impact_json: Path | None = None,
    result_manifest: Path | None = None,
) -> dict:
    import html

    summary_paths = [summary_2024]
    if summary_2025 is not None:
        summary_paths.append(summary_2025)
    summaries = [load_json(path) for path in summary_paths]
    flat_paths = [flat_summary_2024]
    if summary_2025 is not None:
        flat_paths.append(flat_summary_2025)
    for summary, flat_path in zip(
        summaries,
        flat_paths,
    ):
        if flat_path is None:
            continue
        flat = load_json(flat_path)
        if flat.get("status") != "complete":
            raise ValueError(f"flat plot summary is incomplete: {flat_path}")
        summary.setdefault("plots", []).extend(flat.get("plots") or [])
    docs_dir.mkdir(parents=True, exist_ok=True)
    variable_names = {
        "nb": "Nb", "njet": "Nj", "nfatjet": "Nfj", "ntop": "Ntop", "nw": "NW",
        "ht": "HT", "ut": "UT", "met": "pTmiss", "jet_pt": "Jet pT",
        "fatjet_pt": "FatJet pT", "bjet_pt": "b-jet pT",
        "search_bins": "Search bins", "recoil": "Recoil",
        "limit": "Expected limits", "impact": "Nuisance impacts",
    }
    options = ["<option value='all'>All variables</option>"]
    variables = []
    cards = []
    for summary in summaries:
        year = str(summary["year"])
        for plot in summary.get("plots") or []:
            name = str(plot["name"])
            variable = str(
                plot.get("variable")
                or ("search_bins" if "search_bins" in name else "recoil")
            )
            if variable not in variables:
                variables.append(variable)
            kind = str(
                plot.get("kind")
                or ("VR" if "_vr_" in name else "SR" if "_sr_" in name else "CR")
            )
            region = str(plot.get("region") or name)
            title = f"{year} · {kind} · {region} · {variable_names.get(variable, variable)}"
            cards.append(
                f"<a class='plot' data-year='{html.escape(year)}' data-kind='{html.escape(kind)}' "
                f"data-variable='{html.escape(variable)}' href='plots/{html.escape(year)}/{html.escape(name)}.pdf'>"
                f"<img src='plots/{html.escape(year)}/{html.escape(name)}.png' loading='lazy' alt='{html.escape(title)}'>"
                f"<span>{html.escape(title)}</span></a>"
            )
    impact_record = None
    if impact_png or impact_pdf or impact_json:
        if not impact_png or not impact_pdf or not impact_json:
            raise ValueError("impact PNG, PDF and JSON must be supplied together")
        impact_dir = docs_dir / "impacts"
        impact_dir.mkdir(parents=True, exist_ok=True)
        for source in (impact_png, impact_pdf, impact_json):
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, impact_dir / source.name)
        title = "T2tt · Asimov r=1 impacts · mStop 1200 GeV, mLSP 500 GeV"
        cards.append(
            f"<a class='plot' data-year='Combined' data-kind='Impact' data-variable='impact' "
            f"href='impacts/{html.escape(impact_pdf.name)}'>"
            f"<img src='impacts/{html.escape(impact_png.name)}' loading='lazy' alt='{html.escape(title)}'>"
            f"<span>{html.escape(title)}</span></a>"
        )
        impact_record = {
            "status": "complete",
            "benchmark": "mStop1200_mLSP500",
            "asimov_expect_signal": 1,
            "png": f"impacts/{impact_png.name}",
            "pdf": f"impacts/{impact_pdf.name}",
            "json": f"impacts/{impact_json.name}",
        }
        if "impact" not in variables:
            variables.append("impact")
    published_results = []
    if result_manifest is not None:
        manifest = load_json(result_manifest)
        if manifest.get("status") != "complete":
            raise ValueError(f"result manifest is incomplete: {result_manifest}")
        for record in manifest.get("results") or []:
            year = str(record["year"])
            kind = str(record["kind"])
            title = str(record["title"])
            slug = str(record["slug"])
            destination = docs_dir / "results" / year / slug
            destination.mkdir(parents=True, exist_ok=True)
            copied = {}
            for extension in ("png", "pdf", "json"):
                source_value = record.get(extension)
                if not source_value:
                    continue
                source = Path(source_value)
                if not source.exists():
                    raise FileNotFoundError(source)
                target = destination / source.name
                shutil.copy2(source, target)
                copied[extension] = str(target.relative_to(docs_dir))
            if "png" not in copied or "pdf" not in copied:
                raise ValueError(f"result is missing PNG/PDF pair: {record}")
            variable = "limit" if kind == "Limit" else "impact"
            if variable not in variables:
                variables.append(variable)
            cards.append(
                f"<a class='plot' data-year='{html.escape(year)}' data-kind='{html.escape(kind)}' "
                f"data-variable='{variable}' href='{html.escape(copied['pdf'])}'>"
                f"<img src='{html.escape(copied['png'])}' loading='lazy' alt='{html.escape(title)}'>"
                f"<span>{html.escape(title)}</span></a>"
            )
            published_results.append(
                {**record, "published": copied}
            )
    for variable in variables:
        options.append(f"<option value='{html.escape(variable)}'>{html.escape(variable_names.get(variable, variable))}</option>")
    years = [str(summary["year"]) for summary in summaries]
    year_buttons = "".join(
        f"<button data-value='{html.escape(year)}'"
        + (" class='active'" if index == 0 else "")
        + f">{html.escape(year)}</button>"
        for index, year in enumerate(years)
    )
    initial_year = years[0]
    page = """<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Run-3 all-hadronic stop analysis results</title>
<style>
:root{--ink:#171b1d;--muted:#5f686d;--line:#d6dcdf;--bg:#f3f5f6;--panel:#fff;--accent:#087f5b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:22px 24px}header div,main{max-width:1500px;margin:0 auto}
h1{font-size:26px;margin:0;letter-spacing:0}.toolbar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:11px 18px}
.toolbar-inner{max-width:1500px;margin:0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.segments{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
button,select{font:inherit;background:#fff;color:var(--ink);border:0;min-height:38px;padding:7px 12px}button{border-right:1px solid var(--line);cursor:pointer}button:last-child{border-right:0}
button.active{background:var(--ink);color:#fff}select{border:1px solid var(--line);border-radius:6px;min-width:150px}
main{padding:18px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:13px}.plot{display:block;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;color:var(--ink);text-decoration:none}
.plot[hidden]{display:none}.plot img{display:block;width:100%;height:auto}.plot span{display:block;padding:8px 10px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
@media(max-width:540px){header{padding:18px}h1{font-size:22px}.plots{grid-template-columns:1fr}.toolbar{padding:9px}.toolbar-inner{gap:7px}button{padding:6px 9px}}
</style></head><body>
<header><div><h1>Run-3 all-hadronic stop analysis results</h1></div></header>
<div class='toolbar'><div class='toolbar-inner'>
<div class='segments' id='years'>""" + year_buttons + ("<button data-value='Combined'>Combined</button>" if impact_record else "") + """</div>
<div class='segments' id='kinds'><button data-value='all' class='active'>All</button><button data-value='CR'>CR</button><button data-value='SR'>SR</button><button data-value='VR'>VR</button>""" + ("<button data-value='Limit'>Limit</button>" if any(item.get("kind") == "Limit" for item in published_results) else "") + ("<button data-value='Impact'>Impact</button>" if impact_record or any(item.get("kind") == "Impact" for item in published_results) else "") + """</div>
<select id='variables'>""" + "".join(options) + """</select>
</div></div><main><div class='plots'>""" + "".join(cards) + """</div></main>
<script>
let year='""" + initial_year + """',kind='all',variable='all';
function apply(){document.querySelectorAll('.plot').forEach(card=>{card.hidden=!(card.dataset.year===year&&(kind==='all'||card.dataset.kind===kind)&&(variable==='all'||card.dataset.variable===variable));});}
function bind(id,setter){document.querySelectorAll('#'+id+' button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.remove('active'));button.classList.add('active');setter(button.dataset.value);apply();}));}
bind('years',value=>year=value);bind('kinds',value=>kind=value);document.getElementById('variables').addEventListener('change',event=>{variable=event.target.value;apply();});apply();
</script></body></html>"""
    (docs_dir / "index.html").write_text(page)
    result = {
        "status": "complete",
        "page": str(docs_dir / "index.html"),
        "plot_count": len(cards),
        "years": years,
        "impact": impact_record,
        "results": published_results,
    }
    (docs_dir / "page_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def draw_flat_report(
    flat_hists: Path,
    output_dir: Path,
    selected_highdm_sr_only: bool = False,
    selected_sr_search_bins_only: bool = False,
    dy_rz_manifest: Path | None = None,
    gcr_only: bool = False,
    search_bin_config: Path | None = None,
) -> dict:
    payload = load_plot_payload(flat_hists)
    search_bin_merge_summary = None
    if search_bin_config is not None:
        search_bin_configuration = load_json(search_bin_config)
        if search_bin_configuration.get("schema_version") != "search_bin_scheme_v1":
            raise RuntimeError(
                f"unsupported search-bin configuration: {search_bin_config}"
            )
        configured_scheme = str(search_bin_configuration.get("scheme_name") or "")
        if configured_scheme != EXTENDED_AN17_RECOIL_SCHEME:
            raise RuntimeError(
                f"search-bin configuration names {configured_scheme!r}; "
                f"expected {EXTENDED_AN17_RECOIL_SCHEME!r}"
            )
        search_bin_merge_summary = apply_configured_search_bin_merges(
            payload,
            configured_scheme,
            search_bin_configuration,
        )
    lowdm_nres_zero = bool(
        (((payload.get("lowdm_region_policy") or {}).get("resolved_top_veto") or {}).get("applied"))
    )
    lowdm_nres_suffix = r", $N_{res}=0$" if lowdm_nres_zero else ""
    dy_rz_application = (
        apply_dy_rz(payload, dy_rz_manifest) if dy_rz_manifest else None
    )
    plots = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if gcr_only:
        highdm = flat_hist_record(payload, "GCR", allow_signal=False)
        if highdm:
            plots.append(
                draw_flat_blocks(
                    [highdm],
                    output_dir / "highdm_cr_gcr_recoil",
                    xlabel=r"$U_{T}$ (GeV)",
                )
            )
        highdm_split = []
        for suffix in ("Nt0", "Nt1"):
            record = flat_hist_record(payload, f"GCR_{suffix}", allow_signal=False)
            if record:
                highdm_split.append(record)
        if highdm_split:
            plots.append(
                draw_flat_blocks(
                    highdm_split,
                    output_dir / "highdm_cr_gcr_recoil_ntop_split",
                    xlabel=r"$U_{T}$ bin",
                )
            )

        lowdm_scheme = "cat4_GCR_lowDeltaM"
        lowdm_label = r"GCR low $\Delta m$" + lowdm_nres_suffix
        lowdm = flat_search_record(
            payload,
            lowdm_scheme,
            lowdm_label,
            allow_signal=False,
        )
        if lowdm:
            plots.append(
                draw_flat_blocks(
                    [lowdm],
                    output_dir / "lowdm_cr_gcr_onebin",
                    xlabel="Bin",
                )
            )
        lowdm_region_variables = payload.get("lowdm_region_variables") or {}
        available = (
            (payload.get("lowdm_variable_histograms") or {}).get(lowdm_scheme) or {}
        )
        variables = lowdm_region_variables.get("GCR") or sorted(available)
        for variable in variables:
            record = lowdm_variable_record(
                payload,
                lowdm_scheme,
                variable,
                lowdm_label,
                allow_signal=False,
            )
            if not record:
                continue
            plot = draw_flat_blocks(
                [record],
                output_dir / f"lowdm_cr_gcr_{variable}",
                xlabel=record.get("xlabel", variable),
                show_yields=True,
            )
            plot["variable"] = variable
            plot["region"] = "GCR"
            plots.append(plot)
        if not plots:
            raise RuntimeError("no flat GCR histograms were available")
        if not all(plot.get("unit_area") for plot in plots):
            raise RuntimeError("a GCR-only plot was rendered without unit-area normalization")
        summary = {
            "status": "complete",
            "source": str(flat_hists),
            "output_dir": str(output_dir),
            "gcr_only": True,
            "plots": plots,
            "plot_count": len(plots),
            "uncertainty_model": "MC stat plus unit-area-normalized Up/Down shape envelopes in quadrature; luminosity omitted",
            "luminosity_uncertainty_applied": False,
            "background_systematic_sources": PLOT_SYSTEMATIC_SOURCES,
            "dy_rz_application": dy_rz_application,
        }
        (output_dir / "flat_plot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary
    if selected_highdm_sr_only or selected_sr_search_bins_only:
        available_schemes = payload.get("search_bin_schemes") or {}
        if EXTENDED_AN17_RECOIL_SCHEME not in available_schemes:
            raise RuntimeError(
                f"missing High-dM search-bin scheme: {EXTENDED_AN17_RECOIL_SCHEME}"
            )
        blocks = selected_an17_recoil_blocks(
            payload, EXTENDED_AN17_RECOIL_SCHEME
        )
        if not blocks:
            raise RuntimeError("High-dM selected SR blocks are empty")
        bin_count = sum(int(block["nbin"]) for block in blocks)
        name = f"highdm{bin_count}_search_bins"
        plots.append(
            draw_flat_blocks(
                blocks,
                output_dir / name,
                xlabel="Search bin",
            )
        )
        lowdm_search_bins = None
        if selected_sr_search_bins_only:
            low_sr_blocks = lowdm_nsv_inclusive_blocks(
                payload, "cat7_SR_lowDeltaM"
            )
            if not low_sr_blocks:
                raise RuntimeError("Low-dM selected SR blocks are empty")
            lowdm_search_bins = sum(int(block["nbin"]) for block in low_sr_blocks)
            plots.append(
                draw_flat_blocks(
                    low_sr_blocks,
                    output_dir / "lowdm_sr_onebin",
                    xlabel="Search bin",
                )
            )
        summary = {
            "status": "complete",
            "source": str(flat_hists),
            "output_dir": str(output_dir),
            "plots": plots,
            "signal_policy": "Signals are overlaid on the full background stack in the blinded SR.",
            "highdm_search_bins": bin_count,
            "lowdm_search_bins": lowdm_search_bins,
            "search_bin_merges": search_bin_merge_summary,
            "search_bin_configuration": (
                str(search_bin_config) if search_bin_config is not None else None
            ),
            "luminosity_fb": LUMINOSITY_FB,
            "luminosity_relative_uncertainty": LUMINOSITY_RELATIVE_UNCERTAINTY,
            "background_systematic_sources": PLOT_SYSTEMATIC_SOURCES,
        }
        (output_dir / "flat_plot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary
    cr_regions = ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"]

    cr_inclusive = []
    for base in cr_regions:
        rec = flat_hist_record(payload, base, allow_signal=False)
        if rec:
            cr_inclusive.append(rec)
            plots.append(draw_flat_blocks([rec], output_dir / f"highdm_cr_{base.lower()}_recoil", xlabel=r"$U_{T}$ (GeV)"))
    if cr_inclusive:
        plots.append(draw_flat_blocks(cr_inclusive, output_dir / "highdm_cr_recoil_inclusive", xlabel=r"$U_{T}$ bin"))

    cr_split = []
    for base in cr_regions:
        split_blocks = []
        for suffix in ["Nt0", "Nt1"]:
            rec = flat_hist_record(payload, f"{base}_{suffix}", allow_signal=False)
            if rec:
                split_blocks.append(rec)
                cr_split.append(rec)
        if split_blocks:
            plots.append(draw_flat_blocks(split_blocks, output_dir / f"highdm_cr_{base.lower()}_recoil_ntop_split", xlabel=r"$U_{T}$ bin"))
    if cr_split:
        plots.append(draw_flat_blocks(cr_split, output_dir / "highdm_cr_recoil_ntop_split"))
    for region, slug in [
        ("HighDMVR_Nb1", "nb1"),
        ("HighDMVR_Nb2", "nb2"),
        ("HighDMVR_Nb3plus", "nb3plus"),
    ]:
        highdm_vr = flat_hist_record(payload, region, allow_signal=False)
        if highdm_vr:
            highdm_vr["annotation"] = highdm_vr["label"]
            plots.append(
                draw_flat_blocks(
                    [highdm_vr],
                    output_dir / f"highdm_vr_{slug}_met",
                    xlabel=r"$p_{T}^{miss}$ (GeV)",
                )
            )
    sr_inc = flat_hist_record(payload, "SR", allow_signal=True)
    if sr_inc:
        sr_inc["blind_data"] = True
        plots.append(draw_flat_blocks([sr_inc], output_dir / "highdm_sr_recoil_inclusive", xlabel=r"$U_{T}$ (GeV)"))
    sr_split = []
    for region in ["SR_Nt0", "SR_Nt1"]:
        rec = flat_hist_record(payload, region, allow_signal=True)
        if rec:
            rec["blind_data"] = True
            sr_split.append(rec)
    if sr_split:
        plots.append(draw_flat_blocks(sr_split, output_dir / "highdm_sr_recoil_ntop_split", xlabel=r"$U_{T}$ bin"))
    an17 = flat_search_record(payload, "boosted_an_17_SR", "SR", allow_signal=True)
    if an17:
        an17["blind_data"] = True
        plots.append(draw_flat_blocks([an17], output_dir / "highdm_sr_an17_search_bins", xlabel="Search bin"))
    an17_nt1 = flat_search_record(payload, "boosted_an_17_SR_Nt1", r"SR\n$N_{t}\geq1$", allow_signal=True)
    if an17_nt1:
        an17_nt1["blind_data"] = True
        plots.append(draw_flat_blocks([an17_nt1], output_dir / "highdm_sr_nt1_an17_search_bins", xlabel="Search bin"))
    available_schemes = payload.get("search_bin_schemes") or {}
    if EXTENDED_AN17_RECOIL_SCHEME not in available_schemes:
        raise RuntimeError(
            "canonical High-dM search-bin scheme is missing: "
            + EXTENDED_AN17_RECOIL_SCHEME
        )
    selected_scheme = EXTENDED_AN17_RECOIL_SCHEME
    selected_recoil_blocks = selected_an17_recoil_blocks(payload, selected_scheme)
    if selected_recoil_blocks:
        selected_bin_count = sum(
            len(block["background"]) for block in selected_recoil_blocks
        )
        selected_name = f"highdm{selected_bin_count}_search_bins"
        plots.append(draw_flat_blocks(selected_recoil_blocks, output_dir / selected_name, xlabel="Search bin"))
    low_cr_blocks = []
    low_blocks = []
    low_map = [
        ("cat2_LLCR_lowDeltaM", r"LLCR low $\Delta m$" + lowdm_nres_suffix, False, "LLCR"),
        ("cat3_QCDCR_lowDeltaM", r"QCDCR low $\Delta m$" + lowdm_nres_suffix, False, "QCDCR"),
        ("cat4_GCR_lowDeltaM", r"GCR low $\Delta m$" + lowdm_nres_suffix, False, "GCR"),
        ("cat5_DY2E_lowDeltaM", r"DY2E low $\Delta m$" + lowdm_nres_suffix, False, "DY2E"),
        ("cat6_DY2M_lowDeltaM", r"DY2M low $\Delta m$" + lowdm_nres_suffix, False, "DY2M"),
        ("cat7_SR_lowDeltaM", r"SR low $\Delta m$" + lowdm_nres_suffix, True, "SR"),
    ]
    for scheme, label, is_sr, base_region in low_map:
        rec = flat_search_record(payload, scheme, label, allow_signal=False)
        if rec:
            rec["blind_data"] = is_sr
            if not is_sr:
                short = scheme.replace("_lowDeltaM", "").split("_", 1)[1].lower()
                low_cr_blocks.append(rec)
                plots.append(draw_flat_blocks([rec], output_dir / f"lowdm_cr_{short}_onebin", xlabel="Bin"))
            low_blocks.append(rec)
    lowdm_variable_plots = []
    lowdm_region_variables = payload.get("lowdm_region_variables") or {}
    for scheme, label, is_sr, base_region in low_map:
        available = ((payload.get("lowdm_variable_histograms") or {}).get(scheme) or {})
        variables = lowdm_region_variables.get(base_region) or sorted(available)
        short = scheme.replace("_lowDeltaM", "").split("_", 1)[1].lower()
        kind = "sr" if is_sr else "cr"
        for variable in variables:
            rec = lowdm_variable_record(payload, scheme, variable, label, allow_signal=is_sr)
            if not rec:
                continue
            rec["blind_data"] = is_sr
            outname = f"lowdm_{kind}_{short}_{variable}"
            plot = draw_flat_blocks(
                [rec],
                output_dir / outname,
                xlabel=rec.get("xlabel", variable),
                show_yields=True,
            )
            plot["variable"] = variable
            plot["region"] = base_region
            lowdm_variable_plots.append(plot)
            plots.append(plot)
    if low_cr_blocks:
        plots.append(draw_flat_blocks(low_cr_blocks, output_dir / "lowdm_cr_onebin", xlabel="Bin"))
    if low_blocks:
        plots.append(draw_flat_blocks(low_blocks, output_dir / "lowdm_cr_sr_onebin", xlabel="Bin"))
    low_sr_blocks = lowdm_nsv_inclusive_blocks(payload, "cat7_SR_lowDeltaM")
    if low_sr_blocks:
        plots.append(draw_flat_blocks(low_sr_blocks, output_dir / "lowdm_sr_onebin", xlabel="Search bin"))
    highdm_search_bins = len(
        (((payload.get("search_bin_schemes") or {}).get(EXTENDED_AN17_RECOIL_SCHEME) or {}).get("bin_labels") or [])
    )
    summary = {"status": "complete", "source": str(flat_hists), "output_dir": str(output_dir), "plots": plots, "lowdm_variable_plot_count": len([p for p in plots if str(p.get("name", "")).startswith("lowdm_") and p.get("variable")]), "lowdm_resolved_top_veto": {"applied": lowdm_nres_zero, "requirement": "Nres=0" if lowdm_nres_zero else None}, "signal_policy": "Signals are drawn only in SR plots; CR blocks exclude T2tt overlays.", "cr_plot_policy": "High-dM and low-dM CRs are drawn both as combined overview plots and as individual region plots.", "ntop_order": "N_t = 0 blocks are placed left of N_t >= 1 blocks.", "highdm_search_bins": highdm_search_bins, "luminosity_fb": LUMINOSITY_FB, "luminosity_relative_uncertainty": LUMINOSITY_RELATIVE_UNCERTAINTY, "background_systematic_sources": PLOT_SYSTEMATIC_SOURCES, "dy_rz_application": dy_rz_application}
    (output_dir / "flat_plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary

def add_to_index(docs_dir: Path, plot_name: str) -> None:
    index = docs_dir / "index.html"
    if not index.exists():
        return
    html = index.read_text()
    stem = f"plots/{plot_name}.png"
    if stem in html:
        return
    token = "</div>"
    card = f"<a class='plot' href='plots/{plot_name}.png'><img src='plots/{plot_name}.png' loading='lazy'><span>{plot_name}</span></a>"
    html = html.replace(token, card + token, 1)
    index.write_text(html)


def main() -> int:
    global LUMINOSITY_FB, LUMINOSITY_RELATIVE_UNCERTAINTY
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", required=False, type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--signal-searchbin-yields", default="docs/data/signal_searchbin_yields.json", type=Path)
    parser.add_argument("--name", default="partial_control_search_bins_style")
    parser.add_argument("--highdm-distributions", type=Path)
    parser.add_argument("--year", choices=["2024", "2025"])
    parser.add_argument("--summary-2024", type=Path)
    parser.add_argument("--summary-2025", type=Path)
    parser.add_argument("--flat-summary-2024", type=Path)
    parser.add_argument("--flat-summary-2025", type=Path)
    parser.add_argument("--web-dir", type=Path)
    parser.add_argument("--impact-png", type=Path)
    parser.add_argument("--impact-pdf", type=Path)
    parser.add_argument("--impact-json", type=Path)
    parser.add_argument("--result-manifest", type=Path)
    parser.add_argument("--flat-hists", type=Path)
    parser.add_argument("--flat-output-dir", type=Path)
    parser.add_argument("--dy-rz-manifest", type=Path)
    parser.add_argument("--only-region")
    parser.add_argument("--only-variable")
    parser.add_argument(
        "--gcr-only",
        action="store_true",
        help="Draw only GCR plots with the adopted unit-area shape comparison",
    )
    parser.add_argument(
        "--selected-sr-search-bins-only",
        action="store_true",
        help="Draw only the adopted High-dM and Low-dM SR search-bin plots",
    )
    parser.add_argument(
        "--search-bin-config",
        type=Path,
        help="Apply the configured final High-dM bin merges before plotting",
    )
    parser.add_argument(
        "--selected-highdm-sr-only",
        action="store_true",
        help=(
            "Draw only the adopted High-dM SR category/search-bin plot "
            "with the standard background stack and signal overlays"
        ),
    )
    parser.add_argument("--luminosity-fb", type=float, default=LUMINOSITY_FB)
    parser.add_argument("--luminosity-relative-uncertainty", type=float, default=LUMINOSITY_RELATIVE_UNCERTAINTY)
    args = parser.parse_args()
    LUMINOSITY_FB = args.luminosity_fb
    LUMINOSITY_RELATIVE_UNCERTAINTY = args.luminosity_relative_uncertainty

    if args.summary_2024:
        if not args.web_dir:
            parser.error("--web-dir is required with year summaries")
        print(json.dumps(write_highdm_distribution_webpage(
            args.summary_2024,
            args.summary_2025,
            args.web_dir,
            flat_summary_2024=args.flat_summary_2024,
            flat_summary_2025=args.flat_summary_2025,
            impact_png=args.impact_png,
            impact_pdf=args.impact_pdf,
            impact_json=args.impact_json,
            result_manifest=args.result_manifest,
        ), sort_keys=True))
        return 0

    if args.highdm_distributions:
        if not args.year:
            parser.error("--year is required with --highdm-distributions")
        outdir = args.flat_output_dir or Path(args.docs_dir or ".") / "plots" / args.year
        print(json.dumps(draw_highdm_distribution_report(
            args.highdm_distributions,
            outdir,
            args.year,
            dy_rz_manifest=args.dy_rz_manifest,
            only_region=args.only_region,
            only_variable=args.only_variable,
        ), sort_keys=True))
        return 0

    if args.flat_hists:
        outdir = args.flat_output_dir or Path(args.docs_dir or ".") / "plots"
        print(json.dumps(draw_flat_report(
            args.flat_hists,
            outdir,
            selected_highdm_sr_only=args.selected_highdm_sr_only,
            selected_sr_search_bins_only=args.selected_sr_search_bins_only,
            dy_rz_manifest=args.dy_rz_manifest,
            gcr_only=args.gcr_only,
            search_bin_config=args.search_bin_config,
        ), sort_keys=True))
        return 0

    if not args.preview_dir:
        parser.error("--preview-dir is required for the legacy preview plot")
    fit = args.preview_dir / "fit_template_summary.json"
    payload = args.preview_dir / "partial_normalized_yields.json"
    outbase = args.preview_dir / "plots" / args.name
    summary = draw(fit, payload, args.signal_searchbin_yields, outbase)
    if args.docs_dir:
        plot_dst = args.docs_dir / "plots"
        plot_dst.mkdir(parents=True, exist_ok=True)
        for suffix in [".png", ".pdf"]:
            shutil.copy2(outbase.with_suffix(suffix), plot_dst / outbase.with_suffix(suffix).name)
        add_to_index(args.docs_dir, args.name)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
