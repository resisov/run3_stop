from __future__ import annotations
import math
import json
import os
import contextlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import awkward as ak
from .signal_models import signal_topology, signal_mass_key


DATA_PROCESSES = {"JetMET", "EGamma", "Muon"}


DATA_PROCESS_BY_REGION = {
    "LLCR": "JetMET",
    "QCDCR": "JetMET",
    "GCR": "EGamma",
    "DY2E": "EGamma",
    "DY2M": "Muon",
    "HighDMVR_Nb1": "JetMET",
    "HighDMVR_Nb2": "JetMET",
    "HighDMVR_Nb3plus": "JetMET",
    "SR": "JetMET",
    "SR_Nt1": "JetMET",
}


def process_to_group(process: str, dataset: str = "") -> str:
    if process in DATA_PROCESSES:
        return "Data"
    if process == "VV":
        return "VV"
    if process == "ST" or dataset.startswith(("TW", "TbarW", "TBbar", "TbarB")):
        return "Single Top"
    if process == "TT" or dataset.startswith("TT") or "TTto" in dataset:
        return "ttbar"
    if process == "DY" or dataset.startswith("DY") or "DYto" in dataset:
        return "DY"
    if process == "GJ" or "GJ" in dataset or "GJets" in dataset:
        return "Gamma + Jets"
    if process == "WtoLNu" or "WtoLNu" in dataset:
        return "W -> lv"
    if process == "Zto2Nu" or "Zto2Nu" in dataset:
        return "Z -> vv"
    if process == "QCD" or dataset.startswith("QCD"):
        return "QCD Multijet"
    return "others"


def canonical_process(process: str, dataset: str = "") -> str:
    group = process_to_group(process, dataset)
    return {
        "VV": "VV",
        "Single Top": "ST",
        "ttbar": "TT",
        "DY": "DY",
        "Gamma + Jets": "GJ",
        "W -> lv": "WtoLNu",
        "Z -> vv": "Zto2Nu",
        "QCD Multijet": "QCD",
    }.get(group, process or "other")


def data_process_allowed(process: str, region: str) -> bool:
    expected = DATA_PROCESS_BY_REGION.get(region)
    return expected is None or process == expected


def dataset_label(meta: dict[str, Any], dataset_id: int) -> tuple[str, str, bool, bool]:
    rec = (meta.get("datasets") or {}).get(str(int(dataset_id))) or {}
    process = str(rec.get("process") or "unknown")
    dataset = str(rec.get("dataset") or "unknown")
    is_data = bool(rec.get("is_data"))
    is_signal = bool(rec.get("is_signal"))
    if not is_data and not is_signal:
        process = canonical_process(process, dataset)
    return dataset, process, is_data, is_signal


def norm_vector(
    norm: dict[str, Any],
    chunk: dict[str, Any],
    dataset_id: int,
    dataset: str,
    is_data: bool,
    is_signal: bool,
    require_normalization: bool = False,
) -> np.ndarray:
    n = len(chunk["dataset_id"])
    if is_data:
        return np.ones(n, dtype=float)
    if is_signal:
        out = np.zeros(n, dtype=float)
        mstops = np.asarray(chunk["mStop"], dtype=int)
        mlsps = np.asarray(chunk["mLSP"], dtype=int)
        topology = signal_topology(dataset)
        for i, (ms, ml) in enumerate(zip(mstops, mlsps)):
            key = signal_mass_key(topology or "T2tt", int(ms), int(ml))
            fac = ((norm.get("signal_mass_points") or {}).get(key) or {}).get("normalization_factor")
            try:
                finite_factor = fac is not None and math.isfinite(float(fac))
            except (TypeError, ValueError, OverflowError):
                finite_factor = False
            positive_factor = finite_factor and float(fac) > 0.0
            if not finite_factor or (require_normalization and not positive_factor):
                if require_normalization:
                    raise RuntimeError(
                        "missing, non-finite, or non-positive signal "
                        f"normalization factor for {key}"
                    )
                continue
            out[i] = float(fac)
        return out
    fac = ((norm.get("dataset_factors") or {}).get(str(int(dataset_id))) or {}).get("normalization_factor")
    try:
        finite_factor = fac is not None and math.isfinite(float(fac))
    except (TypeError, ValueError, OverflowError):
        finite_factor = False
    positive_factor = finite_factor and float(fac) > 0.0
    if not finite_factor or (require_normalization and not positive_factor):
        if require_normalization:
            raise RuntimeError(
                "missing, non-finite, or non-positive background "
                "normalization factor for "
                f"dataset_id={int(dataset_id)}"
            )
        return np.zeros(n, dtype=float)
    return np.full(n, float(fac), dtype=float)

