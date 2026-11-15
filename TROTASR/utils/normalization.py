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


def prepare_signal_replacement(era, sidecars, output):
    """Replace signal metadata and factors, retaining current background inputs."""
    import copy
    from .paths import ROOT
    from .io import read_json, write_json
    from .signal_models import signal_mass_from_genmodel

    current = read_json(ROOT / 'estimations' / f'normalization_{era}.json.gz')
    datasets = {}
    summed = ('events_read', 'events_written', 'files_attempted', 'files_processed', 'sumw', 'sumw2')
    maps = ('signal_sumw_by_genmodel', 'signal_event_genweight_sum_by_genmodel',
            'signal_runs_sumw_source_counts', 'sumw_source_counts')
    receipts = []
    for path in sidecars:
        metadata = read_json(path)
        if metadata.get('status') != 'complete' or str(metadata['analysis_era']) != str(era):
            raise ValueError('Incomplete or wrong-era signal intermediate: ' + str(path))
        receipts.append(dict(path=str(path), status='complete',
            events_written=metadata['events_written'], files_processed=1, files_attempted=1))
        for key, record in metadata['datasets'].items():
            if not record['is_signal']:
                raise ValueError('Signal replacement received a background or data file')
            if key not in datasets:
                datasets[key] = copy.deepcopy(record)
                continue
            target = datasets[key]
            if target['dataset'] != record['dataset']:
                raise ValueError('Signal dataset ID collision')
            for field in summed:
                target[field] += record[field]
            for field in maps:
                for name, value in record.get(field, {}).items():
                    target.setdefault(field, {})[name] = target.get(field, {}).get(name, 0) + value
    xsecs = {int(r['mStop']): r for r in read_json(ROOT / 'stats/stop_xsec_13p6TeV.json')['records']}
    points = {}
    for record in datasets.values():
        for branch, sumw in record['signal_sumw_by_genmodel'].items():
            topology, ms, ml = signal_mass_from_genmodel(branch)
            key = signal_mass_key(topology, ms, ml)
            point = points.setdefault(key, dict(mass_key=key, topology=topology, mStop=ms, mLSP=ml,
                genmodel_branch=branch, sumw_mass_point=0., event_genweight_sum_fallback=0.,
                datasets=[], sumw_source='Runs.genEventSumw_<topology>_<mStop>_<mLSP>'))
            point['sumw_mass_point'] += float(sumw)
            point['event_genweight_sum_fallback'] += float(record.get('signal_event_genweight_sum_by_genmodel', {}).get(branch, 0))
            point['datasets'].append(record['dataset'])
    for point in points.values():
        cross_section = xsecs[point['mStop']]
        sumw = point['sumw_mass_point']
        xsec = float(cross_section['xsec_pb'])
        if not math.isfinite(sumw) or not math.isfinite(xsec) or xsec <= 0:
            raise ValueError('Invalid signal normalization: ' + point['mass_key'])
        point.update(xsec_pb=xsec, xsec_uncertainty_relative=cross_section.get('uncertainty_relative'),
            normalization_factor=xsec * current['luminosity_pb'] / sumw if sumw else None,
            normalization_status='normalized_with_signal_xsec_and_runs_mass_point_sumw' if sumw else 'zero_generated_sumw')
        point['datasets'] = sorted(set(point['datasets']))
    for section in ('datasets', 'physical_datasets', 'dataset_factors'):
        current[section] = {k: v for k, v in current[section].items() if not v.get('is_signal')}
    current['datasets'].update(datasets)
    for key, record in datasets.items():
        physical = record['dataset']
        current['physical_datasets'][physical] = dict(record, physical_dataset=physical,
            split_dataset_ids=[key], normalization_factor=None,
            normalization_status='signal_uses_mass_point_sumw_not_physical_dataset_sumw')
        current['dataset_factors'][key] = dict(dataset=physical, dataset_id=key, physical_dataset=physical,
            is_data=False, is_signal=True, process='SMS', normalization_factor=None,
            normalization_status='signal_uses_mass_point_sumw_not_physical_dataset_sumw',
            dataset_sumw=record['sumw'], physical_dataset_sumw=record['sumw'], xsec_pb=None)
    current['physical_dataset_split_counts'] = {k: len(v.get('split_dataset_ids', ()))
        for k, v in current['physical_datasets'].items()}
    current['signal_mass_points'] = points
    current['blocked_signal_mass_points'] = {k: v for k, v in points.items() if v['normalization_factor'] is None}
    current['signal_cutflow_histograms'] = {}
    current['source_sidecars'] = [s for s in current.get('source_sidecars', [])
        if 'signal' not in Path(s['path']).name] + receipts
    current['source_sidecar_count'] = len(current['source_sidecars'])
    current['signal_replacement'] = dict(datasets=len(datasets), files=len(receipts),
        mass_points=len(points), source_nanoaod_version=15, luminosity_changed=False,
        background_factors_changed=False, sumw_source='full original Runs, per GenModel')
    if not Path(output).exists() or read_json(output) != current:
        write_json(output, current)
    return current['signal_replacement']


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

