#!/usr/bin/env python3
"""Canonical background grouping for plots and Combine models.

Histogram production intentionally preserves the raw process labels.  This
module defines the single downstream grouping contract used when materializing
plot stacks and statistical-model templates.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np


BACKGROUND_GROUP_SPECS = (
    ("VV_VVV", "VV+VVV", ("VV",)),
    ("Top", "Top", ("ST", "TT")),
    ("DY", "DY", ("DY",)),
    ("PhotonJet", "Photon+jet", ("GJ",)),
    ("WtoLNu", "W -> lv", ("WtoLNu",)),
    ("Zto2Nu", "Z -> vv", ("Zto2Nu",)),
    ("QCD", "QCD Multijet", ("QCD",)),
    ("Other", "Others", ()),
)
BACKGROUND_PROCESS_ORDER = tuple(spec[0] for spec in BACKGROUND_GROUP_SPECS)
BACKGROUND_DISPLAY_LABELS = {spec[0]: spec[1] for spec in BACKGROUND_GROUP_SPECS}
RAW_TO_BACKGROUND_PROCESS = {
    raw: process
    for process, _display, raw_processes in BACKGROUND_GROUP_SPECS
    for raw in raw_processes
}


def background_process_for_sample(sample: str) -> str:
    raw_sample = str(sample)
    if raw_sample not in RAW_TO_BACKGROUND_PROCESS:
        raise ValueError(
            f"unmapped background sample {raw_sample!r}; explicitly classify it "
            "instead of silently publishing it as Others"
        )
    return RAW_TO_BACKGROUND_PROCESS[raw_sample]


def _pair_variations(
    nominal: np.ndarray,
    variation_values: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    paired: dict[str, dict[str, np.ndarray]] = {}
    for name, values in variation_values.items():
        if name.endswith("Up"):
            paired.setdefault(name[:-2], {})["up"] = values
        elif name.endswith("Down"):
            paired.setdefault(name[:-4], {})["down"] = values
    return {
        nuisance: {
            "up": pair.get("up", nominal.copy()),
            "down": pair.get("down", nominal.copy()),
        }
        for nuisance, pair in sorted(paired.items())
        if pair
    }


def aggregate_background_processes(
    by_sample: dict[str, Any],
    nbin: int,
    hist_arrays: Callable[[dict[str, Any] | None, int], tuple[np.ndarray, np.ndarray]],
    *,
    signal_prefix: str | tuple[str, ...] = ("T2tt_", "T2tb_", "T2bW_"),
    data_samples: tuple[str, ...] = ("data_obs",),
) -> OrderedDict[str, dict[str, Any]]:
    """Aggregate raw background samples into the adopted process groups."""

    configured_signal_prefixes = (
        (signal_prefix,) if isinstance(signal_prefix, str) else tuple(signal_prefix)
    )
    signal_prefixes = tuple(
        dict.fromkeys(
            (*configured_signal_prefixes, "T2tt_", "T2tb_", "T2bW_")
        )
    )

    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for process in BACKGROUND_PROCESS_ORDER:
        records[process] = {
            "display_label": BACKGROUND_DISPLAY_LABELS[process],
            "sumw": np.zeros(nbin, dtype=float),
            "sumw2": np.zeros(nbin, dtype=float),
            "source_samples": [],
            "variations": {},
        }

    for sample, sample_variations in sorted(by_sample.items()):
        if sample in data_samples or str(sample).startswith(signal_prefixes):
            continue
        process = background_process_for_sample(sample)
        values, sumw2 = hist_arrays((sample_variations or {}).get("nominal"), nbin)
        records[process]["sumw"] += values
        records[process]["sumw2"] += sumw2
        records[process]["source_samples"].append(sample)

    for process, record in records.items():
        sources = record["source_samples"]
        nuisance_variations = sorted(
            {
                variation
                for sample in sources
                for variation in ((by_sample.get(sample) or {}).keys())
                if variation != "nominal"
            }
        )
        varied: dict[str, np.ndarray] = {}
        for variation in nuisance_variations:
            values = np.zeros(nbin, dtype=float)
            for sample in sources:
                sample_variations = by_sample.get(sample) or {}
                shifted, _ = hist_arrays(
                    sample_variations.get(variation)
                    or sample_variations.get("nominal"),
                    nbin,
                )
                values += shifted
            varied[variation] = values
        record["variations"] = _pair_variations(record["sumw"], varied)

    return records


def materialize_grouped_background_templates(
    directory: Any,
    grouped: dict[str, dict[str, Any]],
    background: np.ndarray,
    background_sumw2: np.ndarray,
    edges: np.ndarray,
    write_hist: Callable[..., Any],
    *,
    min_bin: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Write grouped ROOT templates and prove they reproduce the raw total."""

    background = np.asarray(background, dtype=float)
    background_sumw2 = np.asarray(background_sumw2, dtype=float)
    grouped_sumw = np.zeros_like(background)
    grouped_sumw2 = np.zeros_like(background_sumw2)
    summary: dict[str, dict[str, Any]] = {}

    def root_values(values: np.ndarray) -> np.ndarray:
        if min_bin is None:
            return values
        return np.maximum(values, min_bin)

    for process in BACKGROUND_PROCESS_ORDER:
        record = grouped.get(process) or {}
        raw_values = record.get("sumw")
        raw_sumw2 = record.get("sumw2")
        values = np.asarray([] if raw_values is None else raw_values, dtype=float)
        sumw2 = np.asarray([] if raw_sumw2 is None else raw_sumw2, dtype=float)
        if len(values) != len(background) or len(sumw2) != len(background_sumw2):
            raise ValueError(f"{process}: grouped template has the wrong bin count")
        grouped_sumw += values
        grouped_sumw2 += sumw2
        sources = list(record.get("source_samples") or [])
        if not sources:
            continue

        write_hist(directory, process, root_values(values), sumw2, edges)
        nuisances = record.get("variations") or {}
        for nuisance, pair in sorted(nuisances.items()):
            up = np.asarray(pair.get("up", values), dtype=float)
            down = np.asarray(pair.get("down", values), dtype=float)
            if len(up) != len(values) or len(down) != len(values):
                raise ValueError(f"{process}/{nuisance}: grouped variation has the wrong bin count")
            write_hist(
                directory,
                f"{process}_{nuisance}Up",
                root_values(up),
                sumw2,
                edges,
            )
            write_hist(
                directory,
                f"{process}_{nuisance}Down",
                root_values(down),
                sumw2,
                edges,
            )
        summary[process] = {
            "display_label": record.get("display_label", process),
            "yield": float(np.sum(values)),
            "source_samples": sources,
            "shape_nuisances": sorted(nuisances),
        }

    if not np.allclose(grouped_sumw, background, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("grouped yields do not reproduce the raw background total")
    if not np.allclose(
        grouped_sumw2,
        background_sumw2,
        rtol=1.0e-10,
        atol=1.0e-8,
    ):
        raise ValueError("grouped sumw2 does not reproduce the raw background total")
    return summary


def background_grouping_contract() -> dict[str, Any]:
    return {
        "version": 1,
        "raw_histograms_preserved": True,
        "process_order": list(BACKGROUND_PROCESS_ORDER),
        "groups": {
            process: {
                "display_label": display,
                "raw_processes": list(raw_processes),
                "fallback": process == "Other",
            }
            for process, display, raw_processes in BACKGROUND_GROUP_SPECS
        },
    }
