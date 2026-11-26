"""MC-only validation of the current SR template binning."""

from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import json
import math
from math import expm1
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import awkward as ak
import numpy as np
import uproot

from TROTASR.gnn4lowdm.configuration import configuration, portable_model
from TROTASR.gnn4lowdm.inference import Rank005Numpy
from TROTASR.gnn4lowdm.region_features import feature_arrays
from TROTASR.utils.corrections import NONJME_ENDPOINTS, corrected_central_chunks, event_weights, nonjme_event_views
from TROTASR.utils.event_selections import REGIONS, highdm_core, select
from TROTASR.utils.io import read_json, root_metadata, validate_root, write_json
from TROTASR.utils.normalization import dataset_label, norm_vector
from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT, era_year, internal_path
from TROTASR.utils.private_scales import TopWEvents, topw_event_variations, topw_file_input_policy
from TROTASR.utils.shape_kinematics import varied_view
from TROTASR.utils.signal_models import signal_mass_key, signal_topology
from TROTASR.utils.sr_binning import AXES, NATIVE_EDGES
from TROTASR.utils.sr_merge import load_merge
from TROTASR.utils.statistics_model import nps_nuisance_name
from TROTASR.utils.systematic_views import anchored_endpoints
from TROTASR.utils.trota_friend import friend_production_complete, open_friend, read_friend_slice
from TROTASR.utils.vr_binning import (
    HIGH_VR_CATEGORIES, HIGH_VR_DEFINITION, HIGH_VR_RATE_EDGES, N_HIGH_BINS, N_LOW_BINS, N_VR_BINS,
)
from TROTASR.utils.trota_inference import (
    MIXED_FLOATS, MIXED_INTS, P4_FIELDS, bounded_slices, infer_chunk, load_models, p4_view,
)
from TROTASR.workflows.rare_top_policy import POLICY, excluded_ids, primary
from TROTASR.workflows.resources import process_identity


SCHEMA = "trotasr_vr_mc_nominal_v1"
RESPONSE_SCHEMA = "trotasr_vr_background_response_v2"
TIGHT_MIXED_SCHEMA = "trotasr_vr_tight_mixed_mc_v1"
SIGNAL_NORM_BASIS = "registered_mc_runs_sumw_v1"
REPLACEMENT_SIGNAL_POPULATION = "regenerated_signal_uuid"
OUTPUT = ROOT / "stats/run3/vr"
GRID = ROOT / "stats/run3/grid/manifest.json"
MC_NAME = re.compile(r"(?:mc_shard|signal_shard|signal_v15(?:_batch)?)_\d+\.root")
DATA_NAME = re.compile(r"data_(?:shard|completion_MET2022C)_\d+\.root")
FIELDS = ("sumw", "sumw2", "entries")
GIB = 2**30
MC_RESERVE_BYTES = 16 * GIB
DATA_WORKER_BYTES = 6 * GIB
MC_SHARDS_PER_SLOT = 8
MC_SIGNAL_WORKERS = 10
MC_MAX_WORKERS = 24
MC_BACKGROUND_PHYSICAL_WORKERS = 48
MC_BACKGROUND_MAX_WORKERS = 72
RESPONSE_MAX_WORKERS = 4
TIGHT_MIXED_MAX_WORKERS = 48
RESPONSE_CHUNK = 4_000
FRIEND_CHUNK = 8_192
RESPONSE_ADMISSION_BYTES = 6 * GIB
RECOIL_BINS = len(NATIVE_EDGES) - 1
RESPONSE_CELLS = N_VR_BINS * 2 * RECOIL_BINS
LOW_RESPONSE_CELLS = N_LOW_BINS * 2 * RECOIL_BINS
HIGH_RATE_CELLS = N_HIGH_BINS * len(HIGH_VR_RATE_EDGES)
LEGACY_VR_BINS = 148
CARD_SAMPLES = dict(DYJet="DY", PhotonJet="GJ", QCD="QCD", Top="Top", VV="VV",
                    WJet="WtoLNu", ZJet="Zto2Nu")
VR_SR_CHANNELS = dict(SR_lowdm_c1="lowdm", SR_highdm_c1="highdm")
JME_RESPONSE_ENDPOINTS = ("jesTotalUp", "jesTotalDown", "jerUp", "jerDown")
PROJECTION_TERMS = ("CMS_NPS26012_RZstat_", "CMS_NPS26012_zgammaNonclosure_")
TROTA_RESULT_FIELDS = ("nTopMixed1pct", *("TopMixed1pct_" + k for k in (*MIXED_INTS, *MIXED_FLOATS)))


@lru_cache(maxsize=6)
def geometry(era):
    grid = read_json(GRID)
    if (grid.get("era_merged") is not True or tuple(grid.get("years", ())) != ERAS
            or grid.get("sr_data_blinded") is not True
            or grid.get("systematic_combination", {}).get("central") != "corrected_central_jme"):
        raise ValueError("Expected the adopted blinded era-merged grid and corrected central MC")
    low = configuration(era)["sr_binning"]["edges_by_category"]
    expected = [f"Nb{nb}_NISR{isr}" for nb in ("1", "2plus") for isr in ("0", "1", "2plus")]
    if set(low) != set(expected) or any(len(e) != 6 or not np.all(np.diff(e) > 0)
                                       or (e[0], e[-1]) != (0, 1) for e in low.values()):
        raise ValueError("Expected the six SR GNN categories with five bins each")
    return {key: low[key] for key in expected}, HIGH_VR_CATEGORIES


@lru_cache(maxsize=1)
def _tight_mixed_categories():
    high = load_merge(read_json(GRID)["sr_merge"])
    if high["bins_per_year"] != 118:
        raise ValueError("Expected the adopted 118-bin high-dM SR projection")
    return high["categories"]


@lru_cache(maxsize=6)
def layout(era):
    low, high = geometry(era)
    grid = read_json(GRID)
    signals = {signal_mass_key(row["model"], *map(int, re.findall(r"\d+", row["mass"])))
               for row in grid["points"] if row.get("fit_ready") is True}
    if len(signals) != 895:
        raise ValueError("Expected exactly the displayed 895 signal points")
    return low, high, signals


def angular_candidates(a):
    """Only the approved intermediate-angle selections; no topology yet."""
    d1, d2, d3, d4 = (np.asarray(a[f"j{i}_met_dphi"]) for i in range(1, 5))
    nj = np.asarray(a["njet"])
    base = np.logical_and.reduce([np.asarray(a[k], bool) for k in (
        "pass_base_common", "pass_signal_trigger", "pass_no_veto_leptons",
        "pass_zero_tau", "pass_met_250", "pass_ht_300")])
    base &= np.asarray(a["nb_medium"]) >= 1
    low = base & (nj >= 2) & (d1 > .15) & (d1 < .5) & (d2 > .15) & ((nj < 3) | (d3 > .15))
    high = (base & (nj >= 5) & (d1 > .5) & (d2 > .15) & (d3 > .15)
            & (np.minimum.reduce((d2, d3, d4)) < .5))
    return low, high


def template_indices(low, high, nb, nisr, tags, met, score, low_layout, high_layout, *, mtb=None):
    """Assign exclusive VR bins and put finite MET overflow in the last bin."""
    if np.any(low & high) or np.any(~np.isfinite(score[low])) or np.any((score[low] < 0) | (score[low] > 1)):
        raise ValueError("Overlapping VRs or invalid GNN scores")
    if np.any(~np.isfinite(met[high])) or np.any(met[high] <= 250):
        raise ValueError("Invalid high-dM VR MET")
    index = np.full(len(low), -1, dtype=int)
    offset = 0

    def fill(mask, values, edges):
        if np.any(index[mask] != -1):
            raise ValueError("VR event mapped to two SR bins")
        index[mask] = offset + np.searchsorted(edges[1:-1], values[mask], side="right")

    for key, edges in low_layout.items():
        b, i = key.split("_")
        bmask = nb == 1 if b == "Nb1" else nb >= 2
        imask = nisr >= 2 if i == "NISR2plus" else nisr == int(i[-1])
        fill(low & bmask & imask, score, edges)
        offset += len(edges) - 1
    def matches(values, predicate):
        return values == predicate["eq"] if "eq" in predicate else values >= predicate["ge"]

    axes = dict(zip(AXES, (nb, *tags.T)))
    axes["Ntag"] = tags.sum(axis=1)
    for category in high_layout:
        mask = high.copy()
        if "mtb_min" in category:
            if mtb is None:
                raise ValueError("Run 3 high-dM VR requires corrected mTb")
            mask &= (mtb >= category["mtb_min"])
            if category["mtb_max"] is not None:
                mask &= (mtb < category["mtb_max"])
        for axis, predicate in category["predicates"].items():
            mask &= matches(axes[axis], predicate)
        fill(mask, met, category["edges"])
        offset += len(category["edges"]) - 1
    if not np.array_equal(index >= 0, low | high):
        raise ValueError("A selected VR event has no canonical template bin")
    return index


@lru_cache(maxsize=1)
def _gnn(era):
    return Rank005Numpy(portable_model(era))


@lru_cache(maxsize=1)
def _tag_models(era):
    return load_models(era_year(era))


def _models(era):
    return _tag_models(era), _gnn(era)


def _nominal_slices(original, era, *, cached_resolved=None, high_only=False):
    year = era_year(era)
    momenta = anchored_endpoints(original, year, nominal_only=True)["nominal"]
    if cached_resolved is None:
        yield from _inferred_slices(original, era, momenta, high_only=high_only)
    else:
        yield from _friend_slices(original, era, momenta, cached_resolved, high_only=high_only)


def _friend_slices(original, era, momenta, resolved, *, high_only=False):
    # Mixed branches were already attached in source row order. The keyed
    # Resolved map remains valid when applying later angular/object masks.
    view = varied_view(original, momenta, year=era_year(era))
    low, high = angular_candidates(view)
    if high_only:
        low[:] = False
    support = low | high
    if np.any(support):
        yield view[support], resolved, low[support], high[support], _gnn(era) if np.any(low) else None


def _inferred_slices(original, era, momenta, *, reuse=(), remember=None, high_only=False):
    year = era_year(era)
    view = varied_view(original, momenta, year=year)
    low, high = angular_candidates(view)
    if high_only:
        low[:] = False
    support = low | high
    if not np.any(support):
        return
    # Infer only events that can enter either VR in this view.
    inputs = p4_view(original[support], {k: momenta[k.replace("nanoaod", "corrected")][support]
                                        for k in P4_FIELDS})
    models, gnn = (_tag_models(era), None) if high_only or not np.any(low) else _models(era)
    rows = np.flatnonzero(support) if reuse or remember is not None else None
    retained = view[support]
    low, high = low[support], high[support]
    for sl in bounded_slices(inputs, year=year):
        key = rows[sl].tobytes() if rows is not None else None
        batch = next((cache[key] for cache in reuse if key in cache), None)
        if batch is None:
            inferred, resolved, _, _ = infer_chunk(inputs[sl], models)
            batch = tuple(inferred[field] for field in TROTA_RESULT_FIELDS), resolved
        if remember is not None:
            remember[key] = batch
        fields, resolved = batch
        a = retained[sl]
        for field, values in zip(TROTA_RESULT_FIELDS, fields):
            a = ak.with_field(a, values, field)
        yield a, resolved, low[sl], high[sl], gnn


def _select_and_bin(a, resolved, low_angle, high_angle, model, era, layouts, *, subdivide=False,
                    with_rate_cells=False):
    year = era_year(era)
    regions, _ = select(a, resolved, dy_mass_window=None, year=year)
    # "SR" supplies only the canonical object cleaning and input features.
    # Its selection never fills any histogram here.
    objects = regions["SR"]
    block, tags = objects["block"], objects["counts"]
    total = tags.sum(axis=1)
    low = low_angle & (total == 0)
    high = high_angle & (total >= 1) & np.isfinite(block.mtb) & (block.mtb >= 0.)
    for name, region in regions.items():
        if np.any((low | high) & (region["low"] | region["high_search"])):
            raise ValueError("VR overlaps existing region: " + name)
    score = np.zeros(len(a), dtype=float)
    if low.any():
        score[low] = model.predict(*feature_arrays(a, block, "SR", low, year=year)[:5], batch_size=256)
    indices = template_indices(low, high, block.nb, block.nisr, tags, block.recoil, score, *layouts, mtb=block.mtb)
    if subdivide:
        return _response_indices(indices, block.nb, block.recoil), objects["sf_clean"]
    if with_rate_cells:
        cells = np.full(len(indices), -1, dtype=int)
        slots = np.searchsorted(HIGH_VR_RATE_EDGES[1:], block.recoil[high], side="right")
        cells[high] = (indices[high] - N_LOW_BINS) * len(HIGH_VR_RATE_EDGES) + slots
        return indices, objects["sf_clean"], cells
    return indices, objects["sf_clean"]


def _tight_mixed_indices(a, resolved, era, layouts):
    """MC-only scout; the nominal VR/observed overlap guards do not change."""
    if np.any(np.asarray(a["is_data"], bool)):
        raise ValueError("Observed data are forbidden in the tight-Mixed MC scout")
    regions, _ = select(a, resolved, dy_mass_window=None, year=era_year(era))
    if set(regions) != set(REGIONS):
        raise ValueError("Expected every canonical SR and CR mask")
    sr = regions["SR"]
    block, tags = sr["block"], sr["counts"]
    mtb = block.mtb
    candidate = (sr["high"] & np.isfinite(mtb) & (mtb >= 0.) & (mtb < 175.)
                 & np.all(tags == (0, 1, 0, 0), axis=1))
    overlap = {name: int(np.count_nonzero(candidate & (region["low"] | region["high_search"])))
               for name, region in regions.items()}
    if any(overlap.values()):
        raise ValueError(f"Tight-Mixed candidate overlaps a final SR/CR mask: {overlap}")
    overlap["SR_preselection"] = int(np.count_nonzero(candidate & sr["high"]))
    bins = template_indices(np.zeros_like(candidate), candidate, block.nb, block.nisr, tags,
                            block.recoil, np.zeros(len(candidate)), *layouts)
    if int(np.count_nonzero(bins >= 0)) != overlap["SR_preselection"]:
        raise ValueError("Tight-Mixed candidate coverage disagrees with its bin assignment")
    return bins, sr["sf_clean"], overlap


def _tight_mixed_slices(original, era, layouts, *, cached_resolved=None):
    year = era_year(era)
    momenta = anchored_endpoints(original, year, nominal_only=True)["nominal"]
    view = varied_view(original, momenta, year=year)
    # The canonical rebuild supplies mTb from the same jet/MET coordinates
    # as select(); the final selector independently checks the full masks.
    mtb = np.asarray(view["lowdm_mtb"])
    support = highdm_core(view, {}, "SR", year=year) & np.isfinite(mtb) & (mtb >= 0.) & (mtb < 175.)
    if not np.any(support):
        return
    if cached_resolved is not None:
        a = view[support]
        bins, clean, overlap = _tight_mixed_indices(a, cached_resolved, era, layouts)
        yield a, bins, clean, overlap
        return
    inputs = p4_view(original[support], {k: momenta[k.replace("nanoaod", "corrected")][support]
                                        for k in P4_FIELDS})
    retained = view[support]
    models = _tag_models(era)
    for sl in bounded_slices(inputs, year=year):
        inferred, resolved, _, _ = infer_chunk(inputs[sl], models)
        a = retained[sl]
        for field in ("nTopMixed1pct", *("TopMixed1pct_" + k for k in (*MIXED_INTS, *MIXED_FLOATS))):
            a = ak.with_field(a, inferred[field], field)
        bins, clean, overlap = _tight_mixed_indices(a, resolved, era, layouts)
        yield a, bins, clean, overlap


def _response_indices(indices, nb, recoil):
    out = np.full(len(indices), -1, dtype=int)
    selected = indices >= 0
    if (np.any(indices[selected] >= N_VR_BINS) or np.any(nb[selected] < 1) or np.any(~np.isfinite(recoil[selected]))
            or np.any(recoil[selected] <= NATIVE_EDGES[0])):
        raise ValueError("Invalid Nb or recoil in selected VR event")
    u = np.searchsorted(NATIVE_EDGES[1:-1], recoil[selected], side="right")
    out[selected] = (indices[selected] * 2 + (nb[selected] > 1)) * RECOIL_BINS + u
    return out


def _task_source(task):
    if task.get("signal_population") == REPLACEMENT_SIGNAL_POPULATION:
        from TROTASR.workflows.produce import _replacement_task_source
        return _replacement_task_source(task)
    return _accepted_input_path(task)


def _accepted_input_path(task):
    era = str(task["year"])
    relative = Path(task["input"])
    if era not in ERAS or relative.parent != Path("inputs") / era or relative.suffix != ".root":
        raise ValueError("Expected a ROOT shard from an accepted era inventory")
    path = internal_path(ROOT / relative)
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered VR shard: {relative}")
    return path


def _input_kind(task):
    if task.get("signal_population") == REPLACEMENT_SIGNAL_POPULATION:
        return "signal"
    if "vr_kind" in task:
        return task["vr_kind"]
    name = Path(task["input"]).name
    if DATA_NAME.fullmatch(name):
        return "data"
    if MC_NAME.fullmatch(name):
        return "background" if name.startswith("mc_") else "signal"
    path = _accepted_input_path(task)
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        datasets = root_metadata(path, root).get("datasets", {})
    kinds = {(row.get("is_data") is True, row.get("is_signal") is True) for row in datasets.values()}
    for flags, kind in (((True, False), "data"), ((False, True), "signal"), ((False, False), "background")):
        if kinds == {flags}:
            return kind
    raise ValueError(f"Unknown or mixed dataset type in {path.name}")


def _event_branches(root, friend):
    branches = root["Events"].keys()
    if friend is None:
        return branches
    # The friend supplies all corrected Mixed fields. Avoid reading the old
    # Mixed candidates (and their per-branch jagged length helpers) twice.
    return [name for name in branches if name != "nTopMixed1pct"
            and not name.startswith(("TopMixed1pct_", "nTopMixed1pct_"))]


def _project_mc_samples(reader, branches, step_size, friend, metadata, kind, signals, excluded,
                        seen, era, normalization, audit):
    """Project MC before central correction, without using nominal kinematics to veto a shift."""
    from TROTASR.utils.reader import gated_event_chunks

    light = ("dataset_id", "mStop", "mLSP", "is_signal", "is_data", "pass_base_common", "pass_signal_trigger")

    def eligible(chunk):
        if np.any(np.asarray(chunk["is_data"], bool)):
            raise ValueError("Observed data are forbidden in VR MC")
        allowed = np.zeros(len(chunk), dtype=bool)
        # Observe eligible signal mass points even if none pass the quality/trigger cuts.
        for mask, *_ in _row_samples(chunk, metadata, kind, signals, excluded, seen, era, normalization):
            allowed |= mask
        return allowed & np.asarray(chunk["pass_base_common"], bool) & np.asarray(chunk["pass_signal_trigger"], bool)

    for selected, start, stop, rows in gated_event_chunks(reader, branches, step_size, light, eligible, audit):
        if friend is None:
            yield selected, None
        else:
            yield read_friend_slice(friend, selected, start, stop, row_indices=rows)


@lru_cache(maxsize=6)
def _normalization(era):
    return read_json(ROOT / "estimations" / f"normalization_{era}.json.gz")


@lru_cache(maxsize=6)
def _registered_signal_normalization(era):
    """Bind weighted VR receipts to the accepted inventory, not the UUID campaign."""
    from TROTASR.workflows.normalization_metadata import (
        build_signal_mass_points, load_signal_xsec, merge_dataset,
    )

    current = _normalization(era)
    lumi = float(current["luminosity_pb"])
    if not np.isfinite(lumi) or lumi <= 0:
        raise ValueError("Invalid adopted luminosity for " + era)
    tasks = sorted((t for t in inventory()[era] if _input_kind(t) == "signal"),
                   key=lambda task: task["input"])
    if not tasks:
        raise ValueError("No registered signal inputs for " + era)
    datasets = {}
    for task in tasks:
        relative = Path(task["metadata"])
        if relative != Path(task["input"]).with_suffix(".json"):
            raise ValueError("Unexpected registered signal metadata path: " + str(relative))
        path = internal_path(ROOT / relative)
        sidecar = read_json(path)
        if (sidecar.get("status") not in ("complete", "complete_with_bad_files")
                or sidecar.get("events_written") != task["entries"]
                or (era in EARLY_ERAS and sidecar.get("campaign_year") != era)
                or not sidecar.get("datasets")):
            raise ValueError("Unexpected registered signal metadata coverage: " + str(relative))
        for dsid, record in sidecar["datasets"].items():
            runs = record.get("signal_sumw_by_genmodel", {})
            if (not record.get("is_signal") or record.get("is_data")
                    or set(record.get("signal_event_genweight_sum_by_genmodel", {})) - set(runs)
                    or any(not np.isfinite(float(v)) for v in runs.values())):
                raise ValueError("Registered signal must use Runs sumw: " + str(relative))
            if dsid not in datasets:
                datasets[dsid] = record
            else:
                target = datasets[dsid]
                if any(target.get(key) != record.get(key)
                       for key in ("dataset", "physical_dataset", "process", "is_data", "is_signal")):
                    raise ValueError("Registered signal dataset identity changed: " + str(dsid))
                merge_dataset(target, record)

    xsecs = load_signal_xsec(ROOT / "stats/stop_xsec_13p6TeV.json")
    points = build_signal_mass_points(datasets, xsecs, lumi)
    factors = {name: float(item["normalization_factor"])
               for name, item in points.items() if item["normalization_factor"] is not None}
    shown = layout(era)[2]
    if any(name not in factors or not np.isfinite(factors[name]) or factors[name] <= 0 for name in shown):
        raise ValueError("Registered signal Runs sums do not cover the adopted grid for " + era)
    names = {str(dsid): r["dataset"] for dsid, r in datasets.items()}

    if not current.get("signal_replacement"):
        live_points = current.get("signal_mass_points", {})
        old_names = {str(dsid): r["dataset"] for dsid, r in current.get("datasets", {}).items()
                     if r.get("is_signal")}
        if names != old_names or any(not np.isclose(
                factors[name], float(live_points[name]["normalization_factor"]), rtol=2e-12, atol=0.)
                for name in shown):
            raise ValueError("Registered signal Runs sums disagree with the accepted normalization for " + era)
    return dict(basis=SIGNAL_NORM_BASIS, era=era,
                datasets=names,
                norm={"signal_mass_points": {k: {"normalization_factor": v} for k, v in factors.items()}})


def _row_samples(a, metadata, sample_kind, signals, excluded, seen, era, normalization=None):
    dataset_ids, stop, lsp = (np.asarray(a[k], dtype=int) for k in ("dataset_id", "mStop", "mLSP"))
    signal = np.asarray(a["is_signal"], dtype=bool)
    for dataset_id, ms, ml in np.unique(np.stack((dataset_ids, stop, lsp), axis=1), axis=0):
        dataset, process, is_data, is_signal = dataset_label(metadata, dataset_id)
        mask = (dataset_ids == dataset_id) & (stop == ms) & (lsp == ml)
        if (dataset == "unknown" or is_data or np.any(signal[mask] != is_signal)
                or is_signal != (sample_kind == "signal")):
            raise ValueError("Input row is not a known MC sample of the expected kind")
        if str(dataset_id) in excluded:
            continue
        if is_signal:
            topology = signal_topology(dataset)
            if not topology:
                raise ValueError("Missing signal topology")
            sample = signal_mass_key(topology, ms, ml)
            if sample not in signals:
                continue
            if normalization is None:
                raise ValueError("Registered signal normalization is required")
            factor = normalization.get("signal_mass_points", {}).get(sample, {}).get("normalization_factor")
            if factor is None or not np.isfinite(float(factor)) or float(factor) <= 0:
                raise ValueError("Missing valid signal normalization for " + sample)
            seen.add(sample)
        else:
            sample = ("Top" if process in ("TT", "ST") or primary(metadata["datasets"][str(dataset_id)])
                      in POLICY["top_primaries"] else process)
        yield mask, dataset_id, dataset, process, is_signal, sample


def _histogram(indices, weights, size=N_VR_BINS):
    if not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite MC event weight")
    return np.stack((np.bincount(indices, weights, minlength=size),
                     np.bincount(indices, weights**2, minlength=size),
                     np.bincount(indices, minlength=size)), axis=1)


def _response_card_scopes(card):
    """Read authorized background applicability; never inspect a template or SR observation."""
    scopes = {era: {} for era in ERAS}
    columns = []
    for (channel, process) in card["columns"]:
        regime = VR_SR_CHANNELS.get(channel)
        if regime is None:
            columns.append(None)
            continue
        stem, _, era = process.rpartition("_")
        if era not in ERAS:
            raise ValueError("Accepted SR process has no era: " + process)
        family = stem.split("_", 1)[0]
        if family == "signal":
            columns.append(None)
            continue
        if family not in CARD_SAMPLES:
            raise ValueError("Unmapped accepted SR background: " + process)
        columns.append((era, regime, CARD_SAMPLES[family]))
    for n, kind, *values in card["nuisances"]:
        if kind != "shape":
            continue
        for column, value in zip(columns, values):
            if column is None or value == "-":
                continue
            if float(value) != 1.:
                raise ValueError("VR response requires unit background endpoint coefficient: " + n)
            era, regime, sample = column
            scopes[era].setdefault(sample, {}).setdefault(regime, set()).add(n)
    return {era: {s: {m: sorted(ns) for m, ns in modes.items()} for s, modes in samples.items()}
            for era, samples in scopes.items()}


def _fit_only_response_support(sr, cr, scopes):
    """The two omitted 2022 SR cells still have canonical QCD CR bindings."""
    omitted, control = "SR_highdm_c1", "QCDCR_highdm_c1"
    sr_columns, cr_columns = set(sr["columns"]), set(cr["columns"])
    evidence = []
    for nb in ("Nb1", "Nb2plus"):
        process = f"QCD_{nb}_u5_2022"
        if (omitted, process) in sr_columns:
            continue
        retained = (control, f"QCD_{nb}_u4_2022")
        matching = [row for row in cr["rateparams"] if tuple(row[2:4]) == retained]
        expected = f"CMS_NPS26012_qcd_norm_highdm_{nb}_bin4"
        if (retained not in cr_columns or len(matching) != 1
                or matching[0][0].split("__eraScale_")[0] != expected):
            raise ValueError("Expected preserved CR binding for fit-only 2022 QCD omission: " + process)
        shapes = {name for name in (nps_nuisance_name("jer", "2022"), nps_nuisance_name("jesTotal", "2022"))}
        values = scopes["2022"].setdefault("QCD", {}).setdefault("highdm", [])
        values[:] = sorted(set(values) | shapes)
        evidence.append(dict(omitted_sr_column=[omitted, process], retained_cr_column=list(retained),
                             parameter=expected, response_shapes=sorted(shapes)))
    return evidence


def _response_contract():
    from TROTASR.workflows.diagnostic_common import grid_card
    from TROTASR.workflows.prediction_plot_inputs import parse_card

    grid = read_json(GRID)
    if grid.get("sr_data_blinded") is not True or grid.get("era_merged") is not True:
        raise ValueError("Require the accepted blinded six-era VR model")
    nominal = read_json(OUTPUT / "mc_summary.json.gz")
    background_only = nominal.get("background_only") is True
    expected = _background_inventory(inventory())
    if (nominal.get("schema") != SCHEMA or nominal.get("status") not in ("complete", "incomplete")
            or nominal.get("high_definition") != HIGH_VR_DEFINITION
            or nominal.get("high_categories") != HIGH_VR_CATEGORIES
            or nominal.get("pilot") or nominal.get("sr_observations_read") is not False
            or nominal.get("vr_observations_read") is not False
            or not _has_background_coverage(nominal, expected)
            or set(nominal.get("era_samples", {})) != set(expected)
            or (background_only and (nominal.get("full_era_signal_points") != 0
                or any(nominal["coverage"][e].get("signal") != [0, 0] for e in expected)))):
        raise ValueError("Complete the accepted nominal VR backgrounds before producing responses")
    card, card_id = grid_card(GRID, "impact", require_grid_ready=False)
    cr, cr_id = grid_card(GRID, "cronly", require_grid_ready=False)
    sr_payload, cr_payload = parse_card(card.read_text()), parse_card(cr.read_text())
    scopes = _response_card_scopes(sr_payload)
    support = _fit_only_response_support(sr_payload, cr_payload, scopes)
    if any(not scope for scope in scopes.values()):
        raise ValueError("Accepted SR background nuisance coverage is incomplete")
    return dict(card=card_id["card"], cr_card=cr_id["card"],
                cr_card_shapes=sorted(row[0] for row in cr_payload["nuisances"] if row[1] == "shape"),
                fit_only_qcd_support=support,
                nominal_era_samples=nominal["era_samples"], scopes=scopes)


def _response_histogram(indices, weights):
    if not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite VR response weight")
    return np.stack((np.bincount(indices, weights, minlength=RESPONSE_CELLS),
                     np.bincount(indices, weights**2, minlength=RESPONSE_CELLS),
                     np.bincount(indices, minlength=RESPONSE_CELLS)), axis=1)


def _response_empty():
    return np.zeros((RESPONSE_CELLS, len(FIELDS)))


def _response_sparse(rows):
    return {str(i): rows[i].tolist() for i in np.flatnonzero(np.any(rows != 0., axis=1))}


def _endpoint_name(endpoint, era):
    if endpoint.endswith("Up"):
        return nps_nuisance_name(endpoint[:-2], era), "up"
    if endpoint.endswith("Down"):
        return nps_nuisance_name(endpoint[:-4], era), "down"
    raise ValueError("Unpaired VR endpoint: " + endpoint)


def _shape_endpoints(era, scopes):
    declared = {n for modes in scopes.values() for nuisances in modes.values() for n in nuisances}
    return tuple(e for e in (*JME_RESPONSE_ENDPOINTS, *NONJME_ENDPOINTS)
                 if _endpoint_name(e, era)[0] in declared)


def _weight_endpoints(base, topw, era):
    # Replace one weight at a time; never form a systematic cross-product.
    if set(base) & set(topw) != {"nominal"}:
        raise ValueError("Duplicate canonical VR weight endpoints")
    raw = [(key, value, topw["nominal"]) for key, value in base.items() if key != "nominal"]
    raw.extend((key, base["nominal"], value) for key, value in topw.items() if key != "nominal")
    paired = defaultdict(dict)
    for key, base_values, topw_values in raw:
        name, direction = _endpoint_name(key, era)
        if direction in paired[name]:
            raise ValueError("Ambiguous canonical VR weight nuisance: " + name)
        paired[name][direction] = (base_values, topw_values)
    if any(set(pair) != {"up", "down"} for pair in paired.values()):
        raise ValueError("Missing direction in canonical VR weight pair")
    return paired


def _response_mode_slice(mode):
    return slice(0, LOW_RESPONSE_CELLS) if mode == "lowdm" else slice(LOW_RESPONSE_CELLS, RESPONSE_CELLS)


def _check_nominal_response(accepted, computed):
    for sample in set(accepted) | set(computed):
        expected = np.zeros((N_VR_BINS, len(FIELDS)))
        for i, row in accepted.get(sample, {}).items():
            expected[int(i)] = row
        actual = computed.get(sample, _response_empty()).reshape(N_VR_BINS, 2, RECOIL_BINS, len(FIELDS)).sum(axis=(1, 2))
        # Independent chunking is allowed. A change in selected counts or
        # beyond floating-point accumulation in the weights is not.
        scale = np.sqrt(expected[:, 1] * expected[:, 2])
        if (not np.array_equal(actual[:, 2], expected[:, 2])
                or np.any(np.abs(actual[:, 0] - expected[:, 0]) > 1e-9 * scale + 1e-10)
                or not np.allclose(actual[:, 1], expected[:, 1], rtol=1e-9, atol=1e-10)):
            raise ValueError("VR response nominal does not reproduce the accepted receipt: " + sample)


def _response_event_views(original, era, shape_endpoints, *, cached_resolved=None):
    year = era_year(era)
    jme = tuple(e for e in shape_endpoints if e in JME_RESPONSE_ENDPOINTS)
    nonjme = tuple(e for e in shape_endpoints if e in NONJME_ENDPOINTS)
    momenta = anchored_endpoints(original, year, nominal_only=not jme)
    if cached_resolved is not None:
        yield "nominal", _friend_slices(original, era, momenta["nominal"], cached_resolved)
        for endpoint in jme:
            yield endpoint, _inferred_slices(original, era, momenta[endpoint])
        if nonjme:
            for shifted, resolved, endpoint in nonjme_event_views((original,), year, cached_resolved, {}, nonjme):
                if endpoint == "nominal":
                    continue
                # These object endpoints do not vary any TROTA model input.
                # Rebuild the selection, but reuse nominal tags even for rows
                # entering VR only after the object shift.
                yield endpoint, (None if shifted is original else
                                 _friend_slices(shifted, era, momenta["nominal"], resolved))
        return
    nominal_batches = {} if nonjme else None
    yield "nominal", _inferred_slices(original, era, momenta["nominal"], remember=nominal_batches)
    for endpoint in jme:
        yield endpoint, _inferred_slices(original, era, momenta[endpoint])
    if nonjme:
        # Non-JME changes object/MET selections but not the jet features or
        # corrected p4 passed to TROTA. Reuse only identical ordered inference
        # batches, so migrations keep their original model batch boundaries.
        previous_batches = {}
        for shifted, _, endpoint in nonjme_event_views((original,), year, None, {}, nonjme):
            if endpoint == "nominal":
                continue
            if shifted is original:
                # This endpoint did not change any event in the whole chunk.
                yield endpoint, None
                continue
            endpoint_batches = {}
            yield endpoint, _inferred_slices(shifted, era, momenta["nominal"],
                                             reuse=(nominal_batches, previous_batches), remember=endpoint_batches)
            previous_batches = endpoint_batches


def _bins_for_response(a, era, policy, dataset_id, dataset, process, mask, clean, variation):
    sub = {key: a[key][mask] for key in ak.fields(a)}
    norm = norm_vector(_normalization(era), sub, int(dataset_id), dataset, False, False,
                       require_normalization=True)
    # The SR flag only selects the MET-trigger SF for the blinded, MC-only VR.
    masks = {key: np.full(int(mask.sum()), key == "SR", bool) for key in REGIONS}
    base, audit = event_weights(sub, dataset, process, era_year(era), masks, policy,
                            apply_topw=False, include_variations=variation)
    protected = audit.get("nonfinite_weight_protection", {})
    if protected:
        raise ValueError(f"Nonfinite weight response in {era}/{dataset}: {protected}")
    topw, _ = topw_event_variations(ROOT, era, dataset, process, sub, policy,
                                 cleaned=clean[mask], include_variations=variation)
    weights = norm * base["nominal"] * topw["nominal"]
    return weights, norm, base, topw


def _weight_response_deltas(cells, nominal_weights, norm, base, topw, era, scopes, shape_names,
                            deltas, coverage):
    paired = _weight_endpoints(base, topw, era)
    for mode, names in scopes.items():
        low = cells < LOW_RESPONSE_CELLS
        selected = low if mode == "lowdm" else ~low
        n = int(selected.sum())
        if not n:
            continue
        nominal = _response_histogram(cells[selected], nominal_weights[selected])
        for name in names:
            if name in shape_names or name.startswith(PROJECTION_TERMS):
                continue
            record = coverage.setdefault(mode, {}).setdefault(name, [0, 0])
            if name not in paired:
                # An absent applicable pair is not silently accepted as a
                # zero. Keep partial numeric deltas for diagnosis only.
                record[1] += n
                continue
            record[0] += n
            saved = deltas.setdefault(name, {})
            for direction, (base_values, topw_values) in paired[name].items():
                varied = norm[selected] * base_values[selected] * topw_values[selected]
                target = saved.setdefault(direction, _response_empty())
                target += _response_histogram(cells[selected], varied) - nominal


def _record_shape_endpoint(target, cells, weights, name, direction, scopes):
    selected = np.zeros(len(cells), dtype=bool)
    for mode, names in scopes.items():
        if name in names:
            low = cells < LOW_RESPONSE_CELLS
            selected |= low if mode == "lowdm" else ~low
    if np.any(selected):
        target.setdefault(name, {}).setdefault(direction, _response_empty())[:] += _response_histogram(
            cells[selected], weights[selected])


def compute_response(task, contract, *, use_friend=True):
    if _input_kind(task) != "background":
        raise ValueError("VR response applies only to background MC")
    path = _task_source(task)
    accepted = _read_receipt(task)
    if accepted is None or accepted.get("kind") != "background":
        raise ValueError("Missing accepted background-only VR nominal receipt")
    era = str(task["year"])
    year = era_year(era)
    low, high = geometry(era)
    metadata = validate_root(path, year)
    datasets = metadata.get("datasets", {})
    if not datasets or any(r.get("is_data") or r.get("is_signal") for r in datasets.values()):
        raise ValueError("VR response input is not exclusively background MC")
    excluded = excluded_ids(datasets)
    scopes = contract["scopes"][era]
    shape_endpoints = _shape_endpoints(era, scopes)
    shape_names = {_endpoint_name(e, era)[0] for e in shape_endpoints}
    nominal, weights, shapes, coverage = {}, defaultdict(dict), defaultdict(dict), defaultdict(dict)
    io_audit = {"events_read": 0}
    friend_context = open_friend(task, path) if use_friend else nullcontext()
    with uproot.open(path, object_cache=None, array_cache=None) as root, friend_context as friend:
        policy = topw_file_input_policy(root)
        reader = TopWEvents(root, policy)
        for selected, resolved in _project_mc_samples(reader, _event_branches(root, friend), RESPONSE_CHUNK,
                friend, metadata, "background", (), excluded, set(), era, None, io_audit):
            for original in corrected_central_chunks((selected,), year, {}):
                central_rows = defaultdict(list)
                for endpoint, views in _response_event_views(original, era, shape_endpoints, cached_resolved=resolved):
                    central = endpoint == "nominal"
                    name, direction = (None, None) if central else _endpoint_name(endpoint, era)
                    if views is None:
                        for sample, rows in central_rows.items():
                            for index, vals in rows:
                                _record_shape_endpoint(shapes[sample], index, vals, name, direction,
                                                       scopes.get(sample, {}))
                        continue
                    for a, resolved, lo, hi, gnn in views:
                        cells, clean = _select_and_bin(a, resolved, lo, hi, gnn, era, (low, high), subdivide=True)
                        for mask, dataset_id, dataset, process, _, sample in _row_samples(
                                a, metadata, "background", (), excluded, set(), era):
                            mask &= cells >= 0
                            if not mask.any():
                                continue
                            scope = scopes.get(sample, {})
                            if not central and not any(name in names for names in scope.values()):
                                continue
                            vals, norm, base, topw = _bins_for_response(
                                a, era, policy, dataset_id, dataset, process, mask, clean, central and bool(scope))
                            index = cells[mask]
                            if central:
                                central_rows[sample].append((index, vals))
                                nominal.setdefault(sample, _response_empty())[:] += _response_histogram(index, vals)
                                _weight_response_deltas(index, vals, norm, base, topw, era, scope, shape_names,
                                                        weights[sample], coverage[sample])
                            else:
                                _record_shape_endpoint(shapes[sample], index, vals, name, direction, scope)
    events = io_audit["events_read"]
    if events != task["entries"]:
        raise ValueError("Read event coverage differs from the accepted MC inventory")
    _check_nominal_response(accepted["bins"], nominal)

    def pairs(values):
        return {n: {d: _response_sparse(h) for d, h in pair.items()} for n, pair in values.items()}

    return dict(schema=RESPONSE_SCHEMA, high_definition=HIGH_VR_DEFINITION, input=task["input"], era=era,
                card=contract["card"], cr_card=contract["cr_card"],
                events=events, input_preselection=io_audit["input_preselection"], sr_observations_read=False,
                nominal={sample: _response_sparse(rows) for sample, rows in nominal.items()},
                weight_deltas={sample: pairs(values) for sample, values in weights.items()},
                shape_endpoints={sample: pairs(values) for sample, values in shapes.items()},
                weight_coverage=dict(coverage), nominal_receipt_reused=True)


def compute(task, *, tight_mixed=False, use_friend=True, signal_normalization=None):
    era = str(task["year"])
    kind = _input_kind(task)
    if kind not in ("background", "signal"):
        raise ValueError("Expected an accepted MC shard")
    path = _task_source(task)
    if tight_mixed and path != ROOT / task["input"]:
        raise ValueError("Tight-Mixed MC inputs cannot redirect through symlinks")
    year = era_year(era)
    metadata = validate_root(path, year)
    if kind == "signal":
        low_layout, high_layout, signals = layout(era)
    else:
        low_layout, high_layout = geometry(era)
        signals = ()
    if tight_mixed:
        high_layout = _tight_mixed_categories()
    replacement = task.get("signal_population") == REPLACEMENT_SIGNAL_POPULATION
    if replacement and (tight_mixed or signal_normalization is None):
        raise ValueError("Replacement signals require the approved nominal VR and their own Runs normalization")
    datasets = metadata.get("datasets", {})
    if not datasets or any(r.get("is_data") or bool(r.get("is_signal")) != (kind == "signal")
                           for r in datasets.values()):
        raise ValueError("Input metadata is not exclusively the expected MC kind")
    signal_basis = None
    if kind == "signal":
        signal_basis = (signal_normalization if signal_normalization is not None
                        else _registered_signal_normalization(era))
        if (signal_basis["era"] != era or signal_basis["basis"] != SIGNAL_NORM_BASIS
                or (replacement and signal_basis.get("signal_population") != REPLACEMENT_SIGNAL_POPULATION)
                or any(signal_basis["datasets"].get(str(key)) != record["dataset"]
                       for key, record in datasets.items())):
            raise ValueError("Signal shard does not belong to the registered normalization inventory")
    normalization = signal_basis["norm"] if signal_basis is not None else _normalization(era)
    excluded = excluded_ids(datasets)
    sums, high_recoil, seen, io_audit = {}, {}, set(), {"events_read": 0}
    legacy = (None if tight_mixed or kind != "background" else
              _read_receipt(task, legacy=True))
    if legacy is not None:
        for sample, rows in legacy["bins"].items():
            fresh = np.zeros((N_VR_BINS, len(FIELDS)))
            for i, row in rows.items():
                if int(i) < N_LOW_BINS:
                    fresh[int(i)] = row
            if np.any(fresh[:, 2]):
                sums[sample] = fresh
    overlap = Counter(dict.fromkeys((*REGIONS, "SR_preselection"), 0))
    def selections(original, resolved):
        if tight_mixed:
            for a, bins, clean, intersections in _tight_mixed_slices(
                    original, era, (low_layout, high_layout), cached_resolved=resolved):
                yield a, bins, clean, intersections, None
        else:
            for a, resolved, low_angle, high_angle, gnn in _nominal_slices(
                    original, era, cached_resolved=resolved, high_only=legacy is not None):
                bins, clean, cells = _select_and_bin(a, resolved, low_angle, high_angle, gnn, era,
                                                    (low_layout, high_layout), with_rate_cells=True)
                yield a, bins, clean, {}, cells

    friend_context = open_friend(task, path) if use_friend and not replacement else nullcontext()
    with uproot.open(path, object_cache=None, array_cache=None) as root, friend_context as friend:
        policy = topw_file_input_policy(root)
        reader = TopWEvents(root, policy)
        step_size = FRIEND_CHUNK if friend is not None or replacement else 1_000_000
        branches = _event_branches(root, friend)
        if replacement:
            # These stored NanoAOD-input tags are not used by the approved
            # corrected-nominal inference, so avoid decompressing them.
            branches = [name for name in branches if name not in TROTA_RESULT_FIELDS
                        and not name.startswith("nTopMixed1pct_")]
        for selected, resolved in _project_mc_samples(reader, branches, step_size, friend, metadata, kind,
                signals, excluded, seen, era, normalization, io_audit):
            for original in corrected_central_chunks((selected,), year, {}):
                for a, bins, clean, intersections, cells in selections(original, resolved):
                    overlap.update(intersections)
                    for mask, dataset_id, dataset, process, is_signal, sample in _row_samples(
                            a, metadata, kind, signals, excluded, seen, era, normalization):
                        mask &= bins >= 0
                        if not mask.any():
                            continue
                        sub = {key: a[key][mask] for key in ak.fields(a)}
                        norm = norm_vector(normalization, sub, int(dataset_id), dataset, False,
                                           is_signal, require_normalization=True)
                        # The correction API's SR key only routes the MET
                        # trigger SF; no SR observation or selection is used.
                        masks = {key: np.full(int(mask.sum()), key == "SR", bool) for key in REGIONS}
                        base, _ = event_weights(sub, dataset, process, year, masks, policy,
                                                apply_topw=False, include_variations=False)
                        tw, _ = topw_event_variations(ROOT, era, dataset, process, sub, policy,
                                                     cleaned=clean[mask], include_variations=False)
                        weights = norm * base["nominal"] * tw["nominal"]
                        value = _histogram(bins[mask], weights, LEGACY_VR_BINS if tight_mixed else N_VR_BINS)
                        sums[sample] = sums.get(sample, 0) + value
                        if cells is not None and kind == "background":
                            subcells = cells[mask]
                            high = subcells >= 0
                            if high.any():
                                value = _histogram(subcells[high], weights[high], HIGH_RATE_CELLS)
                                high_recoil[sample] = high_recoil.get(sample, 0) + value
    events = io_audit["events_read"]
    if events != task["entries"]:
        raise ValueError("Read event coverage differs from the accepted MC inventory")
    if tight_mixed and sum(int(rows[:, 2].sum()) for rows in sums.values()) != overlap["SR_preselection"]:
        raise ValueError("Tight-Mixed weighted MC coverage disagrees with its raw selection")
    return dict(schema=TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA,
                era=era, input=task["input"],
                kind=kind, events=events, input_preselection=io_audit["input_preselection"],
                signal_points_seen=sorted(seen),
                bins={k: {str(i): x[i].tolist() for i in np.flatnonzero(x[:, 2])}
                      for k, x in sums.items()}, sr_observations_read=False,
                **({"signal_normalization": {k: signal_basis[k] for k in ("basis", "era")}}
                   if signal_basis is not None else {}),
                **({"signal_population": REPLACEMENT_SIGNAL_POPULATION} if replacement else {}),
                **({"overlap": dict(overlap), "mc_only": True} if tight_mixed else
                   {"high_definition": HIGH_VR_DEFINITION, "sr_overlap": 0,
                    "high_recoil_bins": {k: {str(i): x[i].tolist() for i in np.flatnonzero(x[:, 2])}
                                         for k, x in high_recoil.items()}}))


@lru_cache(maxsize=1)
def _input_inventory():
    """Read the accepted six-era input population and classify each listed shard."""
    found = {era: [] for era in ERAS}
    seen = set()
    plans = (("early_nominal_20261022_full.json", EARLY_ERAS),
             ("full_nominal_20260923.json", ("2024", "2025")))
    for name, eras in plans:
        source = read_json(ROOT / "inputs/manifests" / name)
        if source.get("integrated_input_inventory_complete") is not True:
            raise ValueError("Incomplete integrated input inventory")
        counts = Counter(str(t["year"]) for t in source["tasks"])
        if dict(counts) != source["expected_files_by_year"] or set(counts) != set(eras):
            raise ValueError("Unexpected input inventory coverage")
        for task in source["tasks"]:
            era, shard = str(task["year"]), Path(task["input"]).stem
            if (era, shard) in seen:
                raise ValueError(f"Duplicate accepted VR shard in {era}: {shard}")
            seen.add((era, shard))
            found[era].append(dict(task, vr_kind=_input_kind(task)))
    return found


@lru_cache(maxsize=1)
def inventory():
    return {era: [task for task in tasks if _input_kind(task) in ("background", "signal")]
            for era, tasks in _input_inventory().items()}


def _pilot(tasks):
    # Only the selected MC shards are opened. Prefer modest-size shards so
    # this exercises both kinds and all target eras without a full scan.
    return [min((t for t in tasks if _input_kind(t) == kind),
                key=lambda t: (abs(t["entries"] - 12000), t["input"]))
            for kind in ("background", "signal")]


def _background_inventory(tasks_by_era):
    """Exclude all signal inputs before resolving paths or normalization."""
    selected = {era: [t for t in tasks if _input_kind(t) == "background"]
                for era, tasks in tasks_by_era.items()}
    if set(selected) != set(ERAS) or not all(selected.values()):
        raise ValueError("Missing a background VR era")
    return selected


def _has_background_coverage(summary, inventory):
    coverage = summary.get("coverage", {})
    return set(coverage) == set(inventory) and all(
        coverage[era].get("background") == [len(tasks)] * 2 for era, tasks in inventory.items())


def _receipt_path(task, response=False, tight_mixed=False):
    return OUTPUT / ("tight_mixed_shards" if tight_mixed else "response_shards" if response else "shards") / str(task["year"]) / (
        Path(task["input"]).stem + ".json.gz")


def _read_receipt(task, *, tight_mixed=False, legacy=False):
    path = _receipt_path(task, tight_mixed=tight_mixed)
    if not path.is_file():
        return None
    result = read_json(path)
    era = str(task["year"])
    if (result.get("schema") != (TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA)
            or result.get("era", era) != era
            or ("kind" in result and result["kind"] != _input_kind(task))
            or result.get("events") != task["entries"] or result.get("sr_observations_read") is not False):
        raise ValueError("Divergent VR MC receipt: " + str(path))
    if tight_mixed:
        counts = result.get("overlap", {})
        entries = sum(x[2] for rows in result["bins"].values() for x in rows.values())
        if (result.get("mc_only") is not True or set(counts) != {*REGIONS, "SR_preselection"}
                or any(counts[k] != 0 for k in REGIONS) or counts["SR_preselection"] != entries):
            raise ValueError("Invalid tight-Mixed overlap or coverage receipt: " + str(path))
    else:
        definition = result.get("high_definition")
        if definition not in (None, HIGH_VR_DEFINITION):
            raise ValueError("Unrecognized high-dM VR receipt: " + str(path))
        if (definition is None) != legacy:
            return None
        if not legacy and (result.get("sr_overlap") != 0 or not isinstance(result.get("high_recoil_bins"), dict)):
            raise ValueError("Missing VR overlap audit or normalization recoil rows: " + str(path))
        size = LEGACY_VR_BINS if legacy else N_VR_BINS
        for rows in result["bins"].values():
            for i, values in rows.items():
                if (not 0 <= int(i) < size or len(values) != len(FIELDS)
                        or not all(math.isfinite(x) for x in values) or values[1] < 0
                        or not 0 <= values[2] <= result["events"] or int(values[2]) != values[2]):
                    raise ValueError("Invalid VR nominal bin receipt: " + str(path))
    return result


def _read_response_receipt(task, contract):
    path = _receipt_path(task, response=True)
    if not path.is_file():
        return None
    result = read_json(path)
    era = str(task["year"])
    if result.get("schema") == "trotasr_vr_background_response_v1":
        return None
    if (result.get("schema") != RESPONSE_SCHEMA or result.get("era", era) != era
            or result.get("high_definition") != HIGH_VR_DEFINITION
            or result.get("events") != task["entries"]
            or result.get("nominal_receipt_reused") is not True or result.get("sr_observations_read") is not False):
        raise ValueError("Divergent VR response receipt: " + str(path))
    return result


def _work(task, contract=None, tight_mixed=False, signal_normalization=None):
    result = (compute(task, tight_mixed=tight_mixed, signal_normalization=signal_normalization) if contract is None
              else compute_response(task, contract))
    write_json(_receipt_path(task, response=contract is not None, tight_mixed=tight_mixed), result)
    return task["input"]


def _available_memory():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024


def _active_controller(state, lock_path, *, pid_field="pid", ticks_field="start_ticks", legacy_args=()):
    """Match the current controller and its held lock, including old active runs."""
    if state.get("status") not in ("running", "draining"):
        return None
    pid = state.get(pid_field)
    live = process_identity(pid)
    if live is None:
        return None
    ticks = state.get(ticks_field)
    if ticks is None:
        if not any(arg in legacy_args or Path(arg).name in legacy_args for arg in live["arguments"]):
            return None
    elif live["start_ticks"] != ticks:
        return None
    try:
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return pid
    except FileNotFoundError:
        pass
    return None


def _resource_room(*, reserve_vr_data=False, reserve_bytes=32 * GIB):
    """Account for all admitted fit futures, including unmaterialized ones."""
    rows = {}
    ps = subprocess.check_output(["ps", "-u", str(os.getuid()),
                                  "-o", "pid=,ppid=,stat=,rss=,comm="], text=True)
    for line in ps.splitlines():
        pid, ppid, stat, rss, command = line.split(None, 4)
        if not stat.startswith("Z"):
            rows[int(pid)] = (int(ppid), int(rss) * 1024, command)
    run_root = OUTPUT.parent
    reserved = credit = unmaterialized = 0
    for name, file, lock, pid_key, ticks_key, workers_key, module in (
            ("limits", "workflow_state.json", "controller.lock", "pid", "start_ticks",
             "requested_workers", "combine_limits.py"),
            ("impacts", "impact_status.json", "impact.lock", "controller_pid", "controller_pid_start_ticks",
             "workers", "combine_impacts.py")):
        path = run_root / name / file
        if not path.is_file():
            continue
        state = read_json(path)
        pid = _active_controller(state, path.with_name(lock), pid_field=pid_key, ticks_field=ticks_key,
                                 legacy_args=(name, module))
        if pid not in rows:
            continue
        family = {pid}
        while True:
            new = {p for p, (parent, _, _) in rows.items() if parent in family} - family
            if not new:
                break
            family |= new
        native = [rows[p][1] for p in family - {pid} if rows[p][2] == "combine"]
        futures = max(int(state.get(workers_key, 0)), len(family) - 1)
        reserved += futures
        credit += sum(min(rss, 3 * GIB) for rss in native)
        unmaterialized += futures - (len(family) - 1)
    memory = _available_memory()
    if memory < 24 * GIB or shutil.disk_usage(OUTPUT).free < 80 * GIB:
        return 0, 0
    remaining_data_bytes = 0
    path = OUTPUT / "data_state.json"
    if reserve_vr_data and path.is_file():
        data = read_json(path)
        pid = _active_controller(data, OUTPUT / "data_controller.lock",
                                 legacy_args=("vr-data", "validation_data.py"))
        if pid in rows:
            materialized = [rss for parent, rss, _ in rows.values() if parent == pid]
            missing = max(0, int(data["workers"]) - len(materialized))
            unmaterialized += missing
            remaining_data_bytes = (missing * DATA_WORKER_BYTES
                                    + sum(max(0, DATA_WORKER_BYTES - rss) for rss in materialized))
    slots = 116 - len(rows) - unmaterialized
    budget = memory + credit - reserve_bytes - 3 * reserved * GIB - remaining_data_bytes
    return slots, budget


def _headroom(worker_bytes=3 * GIB):
    slots, budget = _resource_room()
    return max(0, min(slots, budget // worker_bytes))


@dataclass(eq=False)
class _MCWorker:
    process: object
    pipe: object
    task: dict = None
    completed: int = 0
    retiring: bool = False


def _mc_worker_main(pipe, contract=None, tight_mixed=False, signal_normalizations=None):
    with pipe:
        for _ in range(MC_SHARDS_PER_SLOT):
            try:
                task = pipe.recv()
            except EOFError:
                return
            if task is None:
                return
            try:
                _work(task, contract, tight_mixed=tight_mixed,
                      signal_normalization=(signal_normalizations or {}).get(str(task["year"])))
            except Exception as exc:
                pipe.send(str(exc))
            else:
                pipe.send(None)


def _start_mc_worker(context, contract=None, tight_mixed=False, signal_normalizations=None):
    pipe, child = context.Pipe()
    process = context.Process(target=_mc_worker_main, args=(child, contract, tight_mixed, signal_normalizations))
    try:
        process.start()
    except BaseException:
        pipe.close()
        raise
    finally:
        child.close()
    return _MCWorker(process, pipe)


def _retire_mc_worker(worker):
    if not worker.retiring:
        worker.retiring = True
        try:
            worker.pipe.send(None)
        except (EOFError, OSError):
            pass


def _reap_mc_worker(worker):
    worker.process.join()
    worker.process.close()
    worker.pipe.close()


def _mc_pool_high_water(workers):
    peak = 0
    for worker in workers:
        try:
            status = Path(f"/proc/{worker.process.pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                peak = max(peak, int(line.split()[1]) * 1024)
                break
    return peak


def _mc_batch_kind(task):
    return "signal" if _input_kind(task) == "signal" else "standard"


def _mc_cpu_slots(own_worker_load=0.0, *, allow_smt=False):
    # The explicit larger background pool may use both hardware threads.
    # Charge external CPU load in either mode.
    logical_cpus = len(os.sched_getaffinity(0))
    external_load = max(0.0, os.getloadavg()[0] - own_worker_load)
    cpu_budget = logical_cpus if allow_smt else logical_cpus // 2
    return max(0, int(cpu_budget - external_load))


def _worker_cpu_ticks(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return int(stat[19]), int(stat[11]) + int(stat[12])


class _MCWorkerCPULoad:
    """Estimate this pool's CPU load on the host loadavg's one-minute timescale."""

    def __init__(self):
        self._hz = os.sysconf("SC_CLK_TCK")
        self._at = None
        self._ticks = {}
        self._load = 0.0

    def sample(self, pids):
        now = time.monotonic()
        if self._at is not None and now - self._at < 2.0:
            return self._load
        ticks = {pid: reading for pid in pids if (reading := _worker_cpu_ticks(pid)) is not None}
        if self._at is not None:
            elapsed = now - self._at
            busy = sum(max(0, end - previous[1]) for pid, (identity, end) in ticks.items()
                       if (previous := self._ticks.get(pid)) is not None and previous[0] == identity)
            # Credit CPU time only, not idle workers or I/O wait. Smooth the
            # credit to avoid borrowing against unrelated load during ramp-up.
            self._load += (busy / (self._hz * elapsed) - self._load) * -expm1(-elapsed / 60.0)
        self._at, self._ticks = now, ticks
        return self._load


def _collect(all_tasks, pilot, *, tight_mixed=False, background_only=False, replacement_normalizations=None):
    if background_only:
        all_tasks = _background_inventory(all_tasks)
    coverage = {era: dict(background=[0, 0], signal=[0, 0]) for era in ERAS}
    signal_eras = defaultdict(set)
    samples = {}
    per_era = {}
    recoil_by_era = {}
    overlaps = {}
    n_bins = LEGACY_VR_BINS if tight_mixed else N_VR_BINS
    for era, tasks in all_tasks.items():
        sums, recoil = {}, {}
        per_era[era] = sums
        recoil_by_era[era] = recoil
        overlap = Counter(dict.fromkeys((*REGIONS, "SR_preselection"), 0))
        overlaps[era] = overlap
        for task in tasks:
            kind = _input_kind(task)
            coverage[era][kind][1] += 1
            result = _read_receipt(task, tight_mixed=True) if tight_mixed else _read_receipt(task)
            if result is None:
                continue
            coverage[era][kind][0] += 1
            if tight_mixed:
                overlap.update(result["overlap"])
            for point in result["signal_points_seen"]:
                signal_eras[point].add(era)
            for sample, entries in result["bins"].items():
                target = sums.setdefault(sample, np.zeros((n_bins, 3)))
                for i, values in entries.items():
                    target[int(i)] += np.asarray(values, dtype=float)
            for sample, entries in result.get("high_recoil_bins", {}).items():
                target = recoil.setdefault(sample, np.zeros((HIGH_RATE_CELLS, 3)))
                for i, values in entries.items():
                    if not 0 <= int(i) < HIGH_RATE_CELLS:
                        raise ValueError("Invalid normalization recoil cell")
                    target[int(i)] += np.asarray(values, dtype=float)
        if not tight_mixed:
            for sample, values in sums.items():
                if sample in signal_eras:
                    continue
                cells = recoil.setdefault(sample, np.zeros((HIGH_RATE_CELLS, 3)))
                high = values[N_LOW_BINS:]
                total = cells.reshape(N_HIGH_BINS, len(HIGH_VR_RATE_EDGES), 3).sum(axis=1)
                if (not np.array_equal(total[:, 2], high[:, 2])
                        or not np.allclose(total[:, 1], high[:, 1], rtol=1e-9, atol=1e-10)
                        or np.any(np.abs(total[:, 0] - high[:, 0]) > 1e-9 * np.sqrt(high[:, 1] * high[:, 2]) + 1e-10)):
                    raise ValueError("Normalization recoil cells disagree with nominal VR: " + sample)
        for sample, values in sums.items():
            samples[sample] = samples.get(sample, 0) + values
    complete = all(a == b for era in coverage.values() for a, b in era.values())
    if background_only:
        low, high = geometry("2024")
        signals = ()
    else:
        low, high, signals = layout("2024")
    if tight_mixed:
        high = _tight_mixed_categories()
    bkg = sum((x for k, x in samples.items() if k not in signals), np.zeros((n_bins, 3)))
    full_signals = sorted(k for k in signals if signal_eras[k] == set(ERAS))
    definitions = [(mode, key, j, lo, hi, predicates) for mode, groups in (
        ("lowdm", [(key, edges, {}) for key, edges in low.items()]),
        ("highdm", [(f"class{i}", c["edges"], c["predicates"]) for i, c in enumerate(high, 1)]))
        for key, edges, predicates in groups for j, (lo, hi) in enumerate(zip(edges, edges[1:]), 1)]
    rows = []
    zero = np.zeros((n_bins, 3))
    for i, (mode, key, j, low_edge, high_edge, predicates) in enumerate(definitions):
        if tight_mixed and (mode != "highdm" or predicates["Nmix"] != {"eq": 1}
                            or any(predicates[k] != {"eq": 0} for k in ("Nbst", "Nres", "Nw"))):
            if any(x[i, 2] != 0 for x in samples.values()):
                raise ValueError("Tight-Mixed result fills a non-Mixed canonical bin")
            continue
        b, var, n = bkg[i]
        worst = max(full_signals, key=lambda k: samples.get(k, zero)[i, 0], default=None) if complete else None
        s = float(samples.get(worst, zero)[i, 0])
        axis = (dict(gnn_bin_1based=j) if mode == "lowdm" else
                dict(met_low_GeV=low_edge, met_high_GeV=None if high_edge == 1500 else high_edge))
        rows.append(dict(mode=mode, category=key, index_1based=i+1 if mode == "lowdm" else i-29, **axis,
                         background=float(b), background_mc_stat=float(np.sqrt(var)), raw_background_events=int(n),
                         effective_background_events=float(b*b/var) if var > 0 else None,
                         largest_signal=dict(point=worst, yield_nominal_xsec=s,
                                             fraction=float(s/(b+s)) if b > 0 else None)
                         if worst is not None and s > 0 else None))
    extra = {}
    if tight_mixed:
        if len(rows) != 18 or any(v[k] != 0 for v in overlaps.values() for k in REGIONS):
            raise ValueError("Invalid tight-Mixed bin count or nonzero final SR/CR overlap")
        worst = max(full_signals, key=lambda k: float(samples.get(k, zero)[:, 0].sum()),
                    default=None) if complete else None
        y, var, n = bkg.sum(axis=0)
        s = float(samples.get(worst, zero)[:, 0].sum())
        extra = dict(overlap_by_era={e: dict(v) for e, v in overlaps.items()},
                     definition="SR-high preselection; one Mixed only; finite 0 <= mTb < 175 GeV",
                     background_totals={k: x.sum(axis=0).tolist() for k, x in samples.items()
                                        if k not in signals},
                     total=dict(background=float(y), background_mc_stat=float(np.sqrt(var)),
                                raw_background_events=int(n),
                                effective_background_events=float(y*y/var) if var > 0 else None,
                                largest_signal=dict(point=worst, yield_nominal_xsec=s,
                                                    fraction=float(s/(y+s)) if y > 0. else None)
                                if worst is not None and s > 0. else None))
    if background_only:
        allowed_backgrounds = set(CARD_SAMPLES) | set(CARD_SAMPLES.values())
        if any(signal_eras.values()) or set(samples) - allowed_backgrounds:
            raise ValueError("Signal or unknown process in a background-only VR receipt")
        extra["background_only"] = True
    if replacement_normalizations is not None:
        extra.update(signal_population=REPLACEMENT_SIGNAL_POPULATION,
                     signal_normalization={era: {key: basis[key] for key in ("basis", "era")}
                                           for era, basis in replacement_normalizations.items()})
    if not tight_mixed:
        extra.update(high_definition=HIGH_VR_DEFINITION, sr_overlap=0,
                     high_recoil_edges=list(HIGH_VR_RATE_EDGES),
                     era_high_recoil_samples={era: {sample: cells.reshape(
                         N_HIGH_BINS, len(HIGH_VR_RATE_EDGES), 3).tolist() for sample, cells in recoil.items()}
                         for era, recoil in recoil_by_era.items()})
    return dict(schema=TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA,
                status="complete" if complete and (background_only or len(full_signals) == len(signals))
                else "incomplete", pilot=pilot, sr_observations_read=False, vr_observations_read=False,
                mc_only=True, fit_or_transfer_factors_applied=False, fields=list(FIELDS),
                coverage=coverage, full_era_signal_points=len(full_signals),
                missing_era_signal_points=None if background_only else {
                    k: sorted(set(ERAS) - signal_eras[k]) for k in sorted(signals) if k not in full_signals},
                low_edges_by_era={e: geometry(e)[0] for e in ERAS}, high_categories=high,
                rows=rows, samples={k: v.tolist() for k, v in samples.items()},
                era_samples={era: {k: v.tolist() for k, v in sums.items()}
                             for era, sums in per_era.items()}, **extra)


def _accumulate_sparse(dest, rows):
    for i, value in rows.items():
        cell = int(i)
        if not 0 <= cell < RESPONSE_CELLS:
            raise ValueError("VR response cell out of range")
        dest[cell] += value


def _accumulate_response_pairs(dest, variations):
    for name, pair in variations.items():
        for direction, rows in pair.items():
            _accumulate_sparse(dest.setdefault(name, {}).setdefault(direction, _response_empty()), rows)


def _finish_response_sample(nominal, weight_deltas, shapes, scopes, shape_names, counts):
    result, supported, unsupported = {}, {}, {}
    for name in sorted({n for terms in scopes.values() for n in terms}):
        applied_modes = [m for m, terms in scopes.items() if name in terms]
        if name.startswith(PROJECTION_TERMS):
            continue
        missing = [m for m in applied_modes if counts.get(m, {}).get(name, [0, 0])[1]]
        if missing:
            unsupported[name] = dict(modes=missing, reason="applicable_weight_pair_missing", coverage={
                m: counts[m][name] for m in missing})
            continue
        if name in shape_names:
            # Shifted populations are absolute. An empty evaluated endpoint
            # is zero only in the modes where this nuisance applies.
            pair = {}
            for direction in ("up", "down"):
                endpoint = nominal.copy()
                varied = shapes.get(name, {}).get(direction, _response_empty())
                for mode in applied_modes:
                    part = _response_mode_slice(mode)
                    endpoint[part] = varied[part]
                pair[direction] = _response_sparse(endpoint)
            result[name] = pair
            supported[name] = dict(kind="object", modes=applied_modes)
            continue
        if name.startswith(("CMS_scale_j_", "CMS_res_j_", "pdf_", "QCDscale_")):
            unsupported[name] = dict(modes=applied_modes, reason="unavailable_canonical_VR_endpoint")
            continue
        result[name] = {direction: _response_sparse(nominal + weight_deltas.get(name, {}).get(
            direction, _response_empty())) for direction in ("up", "down")}
        supported[name] = dict(kind="weight", modes=applied_modes,
                               coverage={m: counts.get(m, {}).get(name, [0, 0]) for m in applied_modes})
    return result, supported, unsupported


def _collect_response(all_tasks, contract):
    coverage, era_samples, evaluated, unavailable, accepted = {}, {}, {}, {}, {}
    for era, tasks in all_tasks.items():
        nominal, deltas, shapes, weights = {}, defaultdict(dict), defaultdict(dict), defaultdict(dict)
        done = 0
        for task in tasks:
            receipt = _read_response_receipt(task, contract)
            if receipt is None:
                continue
            done += 1
            for sample, rows in receipt["nominal"].items():
                _accumulate_sparse(nominal.setdefault(sample, _response_empty()), rows)
            for source, destination in (("weight_deltas", deltas), ("shape_endpoints", shapes)):
                for sample, terms in receipt[source].items():
                    _accumulate_response_pairs(destination[sample], terms)
            for sample, modes in receipt["weight_coverage"].items():
                for mode, terms in modes.items():
                    for name, (successes, missing) in terms.items():
                        counts = weights[sample].setdefault(mode, {}).setdefault(name, [0, 0])
                        counts[0] += successes
                        counts[1] += missing
        coverage[era] = [done, len(tasks)]
        era_samples[era], evaluated[era], unavailable[era] = {}, {}, {}
        scopes = contract["scopes"][era]
        shape_names = {_endpoint_name(e, era)[0] for e in _shape_endpoints(era, scopes)}
        used_samples = set(nominal) | set(shapes)
        samples = used_samples | set(scopes)
        accepted[era] = {s: contract["nominal_era_samples"][era][s] for s in samples
                         if s in contract["nominal_era_samples"][era]}
        for sample in sorted(samples):
            nom = nominal.get(sample, _response_empty())
            variations, supported, unsupported = _finish_response_sample(
                nom, deltas[sample], shapes[sample], scopes.get(sample, {}), shape_names, weights[sample])
            era_samples[era][sample] = dict(nominal=_response_sparse(nom), variations=variations)
            if supported:
                evaluated[era][sample] = supported
            if unsupported:
                unavailable[era][sample] = unsupported
            if sample in used_samples and sample not in scopes:
                unavailable[era][sample] = dict(model_scope="no_applicable_SR_background_shape_columns")
    complete = all(done == total for done, total in coverage.values())
    if complete:
        for era, samples in era_samples.items():
            cubes = {}
            for sample, row in samples.items():
                cubes[sample] = _response_empty()
                _accumulate_sparse(cubes[sample], row["nominal"])
            accepted_bins = {sample: {i: row for i, row in enumerate(y) if row[2]}
                             for sample, y in accepted[era].items()}
            _check_nominal_response(accepted_bins, cubes)
    return dict(schema=RESPONSE_SCHEMA, high_definition=HIGH_VR_DEFINITION, high_categories=HIGH_VR_CATEGORIES,
                status="complete" if complete else "incomplete",
                coverage=coverage, source_nominal_reused=True,
                axes=dict(vr_bin=N_VR_BINS, nb_groups=["Nb1", "Nb2plus"], recoil_edges=NATIVE_EDGES,
                          recoil_last_open=True, flat_index="(vr_bin*2+nb_index)*8+recoil_index; zero-based"),
                fields=list(FIELDS), era_samples=era_samples, accepted_nominal_era_samples=accepted,
                expected_shape_nuisances=contract["scopes"], evaluated=evaluated, unsupported=unavailable,
                all_applicable_responses_available=complete and not any(unavailable.values()),
                parent_cards={k: contract[k] for k in ("card", "cr_card")},
                cr_card_shapes=contract["cr_card_shapes"], fit_only_qcd_support=contract["fit_only_qcd_support"],
                projection_owned=list(PROJECTION_TERMS),
                endpoint_scope="accepted background shape names only; unavailable terms are explicitly listed; "
                               "measured transfer shapes and normalization terms are projection-owned",
                object_method=dict(jme="canonical anchored Total JES/JER, VR reinference",
                                   nonjme="canonical event shift with unchanged nominal TROTA inputs; "
                                          "new VR response, not reused absolute SR templates"),
                no_fit_or_transfer_factors_applied=True, sr_observations_read=False, mc_only=True)


def run(pilot=False, workers=4, response=False, *, tight_mixed=False, background_only=False,
        replacement_signals=False):
    maximum = TIGHT_MIXED_MAX_WORKERS if tight_mixed else (RESPONSE_MAX_WORKERS if response else MC_MAX_WORKERS)
    if background_only:
        maximum = MC_BACKGROUND_MAX_WORKERS
    if ROOT != Path("/hep2-scratch/twkim/postprocessing/TROTASR") or not 1 <= workers <= maximum:
        raise ValueError(f"VR MC runs only in the canonical hep2 runtime with 1–{maximum} workers")
    if response and pilot:
        raise ValueError("VR response production requires full accepted nominal coverage; no pilot mode")
    if response and tight_mixed:
        raise ValueError("Tight-Mixed is a nominal MC-only evaluation")
    if background_only and (pilot or response or tight_mixed):
        raise ValueError("The background-only VR option builds the complete ordinary nominal background")
    if replacement_signals and (pilot or response or tight_mixed or background_only):
        raise ValueError("Replacement signals can only complete the ordinary nominal VR")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for era in ERAS:
            (geometry if response or background_only else layout)(era)
        contract = _response_contract() if response else None
        worker_contract = ({k: v for k, v in contract.items() if k != "nominal_era_samples"}
                           if response else None)
        tag = "tight_mixed" if tight_mixed else "response" if response else "mc"
        schema = TIGHT_MIXED_SCHEMA if tight_mixed else RESPONSE_SCHEMA if response else SCHEMA
        all_tasks = inventory()
        selected = {era: _pilot(tasks) if pilot else tasks for era, tasks in all_tasks.items()}
        replacement_normalizations = None
        if response or background_only:
            selected = _background_inventory(selected)
        nominal = OUTPUT / "mc_summary.json.gz"
        if not (response or tight_mixed or replacement_signals) and nominal.is_file():
            previous = read_json(nominal)
            if (background_only and previous.get("high_definition") == HIGH_VR_DEFINITION
                    and previous.get("status") == "complete" and not previous.get("background_only")):
                if (previous.get("schema") != SCHEMA or previous.get("pilot")
                        or previous.get("full_era_signal_points") != 895
                        or previous.get("sr_observations_read") is not False
                        or previous.get("vr_observations_read") is not False
                        or not _has_background_coverage(previous, selected)):
                    raise ValueError("Existing complete VR summary has incompatible coverage; preserve it")
                return 0
            if previous.get("signal_population") == REPLACEMENT_SIGNAL_POPULATION:
                raise ValueError("Use --vr-replacement-signals to preserve the accepted replacement VR summary")
        if replacement_signals:
            from TROTASR.workflows.produce import _replacement_signal_inputs
            backgrounds = _background_inventory(all_tasks)
            if any(_read_receipt(task) is None for group in backgrounds.values() for task in group):
                raise ValueError("Finish --vr-background-only before the replacement signal scan")
            selected, replacement_normalizations = _replacement_signal_inputs()
            all_tasks = {era: backgrounds[era] + selected[era] for era in ERAS}

        def receipt(task):
            if response:
                return _read_response_receipt(task, contract)
            if tight_mixed:
                return _read_receipt(task, tight_mixed=True)
            return _read_receipt(task)

        pending = {era: [t for t in tasks if receipt(t) is None] for era, tasks in selected.items()}
        total = sum(map(len, selected.values()))
        done = total - sum(map(len, pending.values()))
        # When replacing only a controller, let its last assigned shards
        # finish under their original workers rather than scheduling them twice.
        finishing = json.loads(os.environ.get("TROTASR_VR_FINISHING_BACKGROUND", "{}"))
        if finishing and (not background_only or response or tight_mixed):
            raise ValueError("Only a background MC restart can inherit finishing shards")
        candidates = {t["input"]: t for group in pending.values() for t in group} if finishing else {}
        waiting = {}
        for name, (pid, ticks) in finishing.items():
            live_identity = process_identity(pid)
            if name not in candidates or live_identity is None or live_identity["start_ticks"] != ticks:
                continue
            if not any("multiprocessing.spawn" in arg for arg in live_identity["arguments"]):
                raise ValueError("Finishing MC worker is not a spawned shard worker")
            waiting[name] = (candidates[name], pid, ticks)
        signal_normalizations = (replacement_normalizations if replacement_signals else
                                 {era: _registered_signal_normalization(era) for era, tasks in pending.items()
                                  if any(_mc_batch_kind(task) == "signal" for task in tasks)})
        failures = []
        reservation = response or tight_mixed or background_only or replacement_signals
        reserve = 32 * GIB if reservation else MC_RESERVE_BYTES
        emergency = 24 * GIB if reservation else MC_RESERVE_BYTES
        signal_limit = workers if tight_mixed else MC_SIGNAL_WORKERS
        controller = process_identity(os.getpid())

        def state(status, active=0, pool_workers=0, kind=None, peak=0, cpu=None):
            write_json(OUTPUT / f"{tag}_state.json", dict(
                schema=schema, status=status, pilot=pilot, pid=os.getpid(), workers=workers,
                start_ticks=controller["start_ticks"],
                total=total, completed=done, active=active, pool_workers=pool_workers, failures=failures,
                batch_kind=kind, pool_peak_rss_gib=peak / GIB,
                memory_reserve_gib=reserve / GIB,
                sr_observations_read=False, vr_observations_read=False, updated=time.time(),
                **(cpu or {}),
                **({} if tight_mixed else {"high_definition": HIGH_VR_DEFINITION}),
                **({"background_only": True} if background_only else {}),
                **({"signal_population": REPLACEMENT_SIGNAL_POPULATION,
                    "scope": "replacement_signal_shards", "background_receipts_reused": True}
                   if replacement_signals else {})))

        def run_pending():
            nonlocal done
            tasks = deque(task for era in pending.values() for task in era if task["input"] not in waiting)
            context = mp.get_context("spawn")
            live = []
            cpu_load = _MCWorkerCPULoad() if background_only else None
            last_state = 0
            peak = 0
            kind = None
            memory_stopped = False
            try:
                while tasks or live or waiting:
                    for name, (task, pid, ticks) in list(waiting.items()):
                        identity = process_identity(pid)
                        if identity is not None and identity["start_ticks"] == ticks:
                            continue
                        if receipt(task) is None:
                            tasks.appendleft(task)
                        else:
                            done += 1
                        del waiting[name]
                        last_state = 0
                    peak = max(peak, _mc_pool_high_water(live))
                    if live and _available_memory() < emergency:
                        memory_stopped = True
                        for worker in live:
                            worker.retiring = True
                            if worker.process.is_alive():
                                worker.process.kill()
                        return False

                    active = sum(w.task is not None for w in live)
                    own_load = cpu_load.sample(w.process.pid for w in live) if cpu_load is not None else 0.0
                    target = min(max(0, workers - len(waiting)), max(1, _mc_cpu_slots(
                        own_load, allow_smt=background_only and workers > MC_BACKGROUND_PHYSICAL_WORKERS)))
                    if tasks and active < workers:
                        slots, budget = _resource_room(reserve_vr_data=True, reserve_bytes=reserve)
                        if slots < 0:
                            budget = 0
                        signal_active = sum(w.task is not None and _mc_batch_kind(w.task) == "signal" for w in live)
                        while tasks and budget > 0 and active < target:
                            if reservation and budget < RESPONSE_ADMISSION_BYTES:
                                break
                            signal = _mc_batch_kind(tasks[0]) == "signal"
                            if signal and signal_active >= signal_limit:
                                break
                            worker = next((w for w in live if w.task is None and not w.retiring), None)
                            if worker is None:
                                if slots <= 0 or len(live) >= target:
                                    break
                                worker = _start_mc_worker(context, worker_contract, tight_mixed=tight_mixed,
                                                          signal_normalizations=signal_normalizations)
                                live.append(worker)
                                slots -= 1
                            worker.task = tasks.popleft()
                            try:
                                worker.pipe.send(worker.task)
                            except (EOFError, OSError):
                                worker.retiring = True
                            active += 1
                            if reservation:
                                budget -= RESPONSE_ADMISSION_BYTES
                            signal_active += signal
                            last_state = 0
                        if budget <= 0:
                            for worker in live:
                                if worker.task is None:
                                    _retire_mc_worker(worker)
                    elif not tasks:
                        for worker in live:
                            if worker.task is None:
                                _retire_mc_worker(worker)

                    kinds = {_mc_batch_kind(w.task) for w in live if w.task is not None}
                    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed" if kinds else (
                        _mc_batch_kind(tasks[0]) if tasks else None)
                    now = time.monotonic()
                    if last_state == 0 or now - last_state >= 15:
                        cpu = {"cpu_target": target + len(waiting), "own_cpu_load": round(own_load, 2),
                               "finishing_previous": len(waiting)} if cpu_load is not None else None
                        state("running", active + len(waiting), len(live) + len(waiting), kind, peak, cpu)
                        last_state = now
                    if not live:
                        time.sleep(2)
                        continue

                    watched = [w.process.sentinel for w in live]
                    watched.extend(w.pipe for w in live if w.task is not None and not w.retiring)
                    ready = set(wait(watched, timeout=2))
                    for worker in live[:]:
                        exited = worker.process.sentinel in ready
                        if worker.task is not None and (worker.pipe in ready or (exited and worker.pipe.poll())):
                            try:
                                error = worker.pipe.recv()
                            except (EOFError, OSError):
                                worker.retiring = True
                            else:
                                if error is None:
                                    done += 1
                                else:
                                    failures.append(dict(input=worker.task["input"], error=error))
                                worker.task = None
                                worker.completed += 1
                                if worker.completed == MC_SHARDS_PER_SLOT:
                                    worker.retiring = True
                                last_state = 0
                        if exited:
                            if worker.task is not None:
                                try:
                                    received = receipt(worker.task) is not None
                                except Exception as exc:
                                    received = False
                                    error = str(exc)
                                else:
                                    error = f"VR MC worker exited {worker.process.exitcode} without a receipt"
                                if received:
                                    done += 1
                                else:
                                    failures.append(dict(input=worker.task["input"], error=error))
                            _reap_mc_worker(worker)
                            live.remove(worker)
                            last_state = 0
                return True
            finally:
                for worker in live:
                    _retire_mc_worker(worker)
                for worker in live:
                    _reap_mc_worker(worker)
                if memory_stopped:
                    done = sum(receipt(task) is not None
                               for batch in selected.values() for task in batch)
                    state("memory_stopped", kind=kind, peak=peak)

        state("running")
        try:
            if not run_pending():
                return 1
            report = (_collect_response(selected, contract) if response else
                      _collect(all_tasks, pilot, tight_mixed=tight_mixed, background_only=background_only,
                               replacement_normalizations=replacement_normalizations))
            # The complete background-only result stays readable while a new
            # signal scan runs or if signal coverage does not close.
            summary = OUTPUT / f"{tag}_summary.json.gz"
            preserve = summary.is_file() and read_json(summary).get("status") == "complete"
            if (not replacement_signals and not preserve) or (not failures and report["status"] == "complete"):
                write_json(summary, report)
            state("needs_attention" if failures or (not pilot and report["status"] != "complete")
                  else "pilot_complete" if pilot else "complete")
        except BaseException:
            state("needs_attention")
            raise
        return 1 if failures or (not pilot and report["status"] != "complete") else 0
