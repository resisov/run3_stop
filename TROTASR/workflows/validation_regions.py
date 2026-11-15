"""MC-only validation of the current SR template binning."""

from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import hashlib
import json
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
from TROTASR.utils.io import read_json, validate_root, write_json
from TROTASR.utils.normalization import dataset_label, norm_vector
from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT, era_year, internal_path
from TROTASR.utils.private_scales import TopWEvents, topw_event_variations, topw_file_input_policy
from TROTASR.utils.shape_kinematics import varied_view
from TROTASR.utils.signal_models import signal_mass_key, signal_topology
from TROTASR.utils.sr_binning import AXES, NATIVE_EDGES
from TROTASR.utils.sr_merge import load_merge
from TROTASR.utils.statistics_model import nps_nuisance_name
from TROTASR.utils.systematic_views import anchored_endpoints
from TROTASR.utils.trota_inference import (
    MIXED_FLOATS, MIXED_INTS, P4_FIELDS, bounded_slices, infer_chunk, load_models, p4_view,
)
from TROTASR.utils.trota_friend import friend_production_complete, open_friend, read_friend_slice
from TROTASR.workflows.rare_top_policy import POLICY, excluded_ids, primary


SCHEMA = "trotasr_vr_mc_nominal_v1"
RESPONSE_SCHEMA = "trotasr_vr_background_response_v1"
TIGHT_MIXED_SCHEMA = "trotasr_vr_tight_mixed_mc_v1"
SIGNAL_NORM_BASIS = "registered_mc_runs_sumw_v1"
OUTPUT = ROOT / "stats/run3/vr"
GRID = ROOT / "stats/run3/grid/manifest.json"
MC_NAME = re.compile(r"(?:mc_shard|signal_shard|signal_v15(?:_batch)?)_\d+\.root")
FIELDS = ("sumw", "sumw2", "entries")
GIB = 2**30
MC_RESERVE_BYTES = 16 * GIB
DATA_WORKER_BYTES = 6 * GIB
MC_SHARDS_PER_SLOT = 8
MC_SIGNAL_WORKERS = 10
MC_MAX_WORKERS = 24
RESPONSE_MAX_WORKERS = 4
TIGHT_MIXED_MAX_WORKERS = 48
RESPONSE_CHUNK = 4_000
FRIEND_CHUNK = 8_192
RESPONSE_ADMISSION_BYTES = 6 * GIB
RECOIL_BINS = len(NATIVE_EDGES) - 1
RESPONSE_CELLS = 148 * 2 * RECOIL_BINS
LOW_RESPONSE_CELLS = 30 * 2 * RECOIL_BINS
CARD_SAMPLES = dict(DYJet="DY", PhotonJet="GJ", QCD="QCD", Top="Top", VV="VV",
                    WJet="WtoLNu", ZJet="Zto2Nu")
VR_SR_CHANNELS = dict(SR_lowdm_c1="lowdm", SR_highdm_c1="highdm")
JME_RESPONSE_ENDPOINTS = ("jesTotalUp", "jesTotalDown", "jerUp", "jerDown")
PROJECTION_TERMS = ("CMS_NPS26012_RZstat_", "CMS_NPS26012_zgammaNonclosure_")


@lru_cache(maxsize=6)
def layout(era):
    grid = read_json(GRID)
    if (grid.get("status") != "cards_ready" or grid.get("era_merged") is not True
            or grid.get("sr_data_blinded") is not True
            or grid.get("systematic_combination", {}).get("central") != "corrected_central_jme"):
        raise ValueError("Expected the adopted blinded era-merged grid and corrected central MC")
    low = configuration(era)["sr_binning"]["edges_by_category"]
    expected = [f"Nb{nb}_NISR{isr}" for nb in ("1", "2plus") for isr in ("0", "1", "2plus")]
    if set(low) != set(expected) or any(len(e) != 6 or not np.all(np.diff(e) > 0)
                                       or (e[0], e[-1]) != (0, 1) for e in low.values()):
        raise ValueError("Expected the six SR GNN categories with five bins each")
    high = load_merge(grid["sr_merge"])
    if high["bins_per_year"] != 118:
        raise ValueError("Expected the adopted 118-bin high-dM SR projection")
    signals = {signal_mass_key(row["model"], *map(int, re.findall(r"\d+", row["mass"])))
               for row in grid["points"] if row.get("fit_ready") is True}
    if len(signals) != 895:
        raise ValueError("Expected exactly the displayed 895 signal points")
    return {key: low[key] for key in expected}, high["categories"], signals


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


def template_indices(low, high, nb, nisr, tags, met, score, low_layout, high_layout):
    """Use the SR bin order and put finite MET overflow in the open last bin."""
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
    for category in high_layout:
        mask = high.copy()
        for axis, value in zip(AXES, (nb, *tags.T)):
            pred = category["predicates"][axis]
            mask &= value == pred["eq"] if "eq" in pred else value >= pred["ge"]
        fill(mask, met, category["edges"])
        offset += len(category["edges"]) - 1
    if not np.array_equal(index >= 0, low | high):
        raise ValueError("A selected VR event has no canonical template bin")
    return index


@lru_cache(maxsize=1)
def _gnn(era):
    return Rank005Numpy(portable_model(era))



@lru_cache(maxsize=1)
def _models(era):
    return load_models(era_year(era)), _gnn(era)


def _nominal_slices(original, era, *, cached_resolved=None):
    year = era_year(era)
    momenta = anchored_endpoints(original, year, nominal_only=True)["nominal"]
    if cached_resolved is None:
        yield from _inferred_slices(original, era, momenta)
    else:
        yield from _friend_slices(original, era, momenta, cached_resolved)



def _friend_slices(original, era, momenta, resolved):
    # Mixed branches were already attached in source row order. The keyed
    # Resolved map remains valid when applying later angular/object masks.
    view = varied_view(original, momenta, year=era_year(era))
    low, high = angular_candidates(view)
    support = low | high
    if np.any(support):
        yield view[support], resolved, low[support], high[support], _gnn(era) if np.any(low) else None


def _inferred_slices(original, era, momenta):
    year = era_year(era)
    view = varied_view(original, momenta, year=year)
    low, high = angular_candidates(view)
    support = low | high
    if not np.any(support):
        return
    # Infer only events that can enter either VR in this view.
    inputs = p4_view(original[support], {k: momenta[k.replace("nanoaod", "corrected")][support]
                                        for k in P4_FIELDS})
    models, gnn = _models(era)
    for sl in bounded_slices(inputs, year=year):
        inferred, resolved, _, _ = infer_chunk(inputs[sl], models)
        a = view[support][sl]
        for key in ("nTopMixed1pct", *("TopMixed1pct_" + k for k in (*MIXED_INTS, *MIXED_FLOATS))):
            a = ak.with_field(a, inferred[key], key)
        yield a, resolved, low[support][sl], high[support][sl], gnn


def _select_and_bin(a, resolved, low_angle, high_angle, model, era, layouts, *, subdivide=False):
    year = era_year(era)
    regions, _ = select(a, resolved, dy_mass_window=None, year=year)
    # "SR" supplies only the canonical object cleaning and input features.
    # Its selection never fills any histogram here.
    objects = regions["SR"]
    block, tags = objects["block"], objects["counts"]
    total = tags.sum(axis=1)
    low = low_angle & (total == 0)
    high = high_angle & (total >= 1) & np.isfinite(block.mtb) & (block.mtb >= 175.)
    for name, region in regions.items():
        if np.any((low | high) & (region["low"] | region["high"])):
            raise ValueError("VR overlaps existing region: " + name)
    score = np.zeros(len(a), dtype=float)
    if low.any():
        score[low] = model.predict(*feature_arrays(a, block, "SR", low, year=year)[:5], batch_size=256)
    indices = template_indices(low, high, block.nb, block.nisr, tags, block.recoil, score, *layouts)
    if subdivide:
        return _response_indices(indices, block.nb, block.recoil), objects["sf_clean"]
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
    models = load_models(year)
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
    if (np.any(nb[selected] < 1) or np.any(~np.isfinite(recoil[selected]))
            or np.any(recoil[selected] <= NATIVE_EDGES[0])):
        raise ValueError("Invalid Nb or recoil in selected VR event")
    u = np.searchsorted(NATIVE_EDGES[1:-1], recoil[selected], side="right")
    out[selected] = (indices[selected] * 2 + (nb[selected] > 1)) * RECOIL_BINS + u
    return out


def _task_source(task):
    era = str(task["year"])
    relative = Path(task["input"])
    if relative.parent != Path("inputs") / era or not MC_NAME.fullmatch(relative.name):
        raise ValueError("Only accepted background or signal MC files are allowed")
    path = internal_path(ROOT / relative)
    st = path.stat()
    for name, value in (("size", st.st_size), ("mtime_ns", st.st_mtime_ns), ("inode", st.st_ino)):
        if name in task["fingerprint"] and value != task["fingerprint"][name]:
            raise ValueError(f"Accepted input identity changed ({name}): {relative}")
    return path


@lru_cache(maxsize=6)
def _normalization(era):
    return read_json(ROOT / "estimations" / f"normalization_{era}.json.gz")


@lru_cache(maxsize=6)
def _registered_signal_normalization(era):
    """Bind weighted VR receipts to the accepted inventory, not the UUID campaign."""
    from TROTASR.workflows.normalization_metadata import (
        build_signal_mass_points, load_signal_xsec, merge_dataset,
    )

    table = ROOT / "estimations" / f"normalization_{era}.json.gz"
    current = _normalization(era)
    lumi = float(current["luminosity_pb"])
    if not np.isfinite(lumi) or lumi <= 0:
        raise ValueError("Invalid adopted luminosity for " + era)
    tasks = sorted((t for t in inventory()[era] if Path(t["input"]).name.startswith("signal_")),
                   key=lambda task: task["input"])
    if not tasks:
        raise ValueError("No registered signal inputs for " + era)
    datasets = {}
    for task in tasks:
        relative = Path(task["metadata"])
        if relative != Path(task["input"]).with_suffix(".json"):
            raise ValueError("Unexpected registered signal metadata path: " + str(relative))
        path = internal_path(ROOT / relative)
        stamp = path.stat()
        if task["metadata_fingerprint"] != dict(inode=stamp.st_ino, size=stamp.st_size,
                                               mtime_ns=stamp.st_mtime_ns):
            raise ValueError("Registered signal metadata changed: " + str(relative))
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
    digest = hashlib.sha256(json.dumps([era, lumi, names, factors], sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    # Untagged weighted receipts are reusable only if they precede the
    # recorded publication, or the current table is the same registered one.
    cutoff_ns = None
    replacement = current.get("signal_replacement")
    if replacement:
        published = read_json(ROOT / "results/merged" / f"systematics_{era}.summary.json")
        record = published.get("signal_replacement", {})
        started = float(record.get("time", 0))
        finished = float(published.get("completed", 0))
        stamped = table.stat().st_mtime
        if not (record.get("background_and_data_unchanged") is True and 0 < started <= stamped <= finished):
            raise ValueError("Cannot establish the previous signal table publication boundary for " + era)
        cutoff_ns = int(started * 1_000_000_000)
    else:
        live_points = current.get("signal_mass_points", {})
        old_names = {str(dsid): r["dataset"] for dsid, r in current.get("datasets", {}).items()
                     if r.get("is_signal")}
        if names != old_names or any(not np.isclose(
                factors[name], float(live_points[name]["normalization_factor"]), rtol=2e-12, atol=0.)
                for name in shown):
            raise ValueError("Registered signal Runs sums disagree with the accepted normalization for " + era)
    return dict(basis=SIGNAL_NORM_BASIS, era=era, sha256=digest, legacy_before_ns=cutoff_ns,
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


def _histogram(indices, weights):
    if not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite MC event weight")
    return np.stack((np.bincount(indices, weights, minlength=148),
                     np.bincount(indices, weights**2, minlength=148),
                     np.bincount(indices, minlength=148)), axis=1)


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
    if (nominal.get("schema") != SCHEMA or nominal.get("status") != "complete"
            or nominal.get("pilot") or nominal.get("full_era_signal_points") != 895
            or nominal.get("sr_observations_read") is not False):
        raise ValueError("Complete the accepted nominal VR production before producing responses")
    card, card_id = grid_card(GRID, "impact")
    cr, cr_id = grid_card(GRID, "cronly")
    sr_payload, cr_payload = parse_card(card.read_text()), parse_card(cr.read_text())
    scopes = _response_card_scopes(sr_payload)
    support = _fit_only_response_support(sr_payload, cr_payload, scopes)
    if any(not scope for scope in scopes.values()):
        raise ValueError("Accepted SR background nuisance coverage is incomplete")
    return dict(card=card_id["card"], card_sha256=card_id["card_sha256"],
                cr_card=cr_id["card"], cr_card_sha256=cr_id["card_sha256"],
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
        expected = np.zeros((148, len(FIELDS)))
        for i, row in accepted.get(sample, {}).items():
            expected[int(i)] = row
        actual = computed.get(sample, _response_empty()).reshape(148, 2, RECOIL_BINS, len(FIELDS)).sum(axis=(1, 2))
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
    yield "nominal", _inferred_slices(original, era, momenta["nominal"])
    for endpoint in jme:
        yield endpoint, _inferred_slices(original, era, momenta[endpoint])
    nonjme = tuple(e for e in shape_endpoints if e in NONJME_ENDPOINTS)
    if nonjme:
        # Apply the canonical non-JME object shift, then the same corrected
        # JME p4/reinference used by VR nominal. This is a new VR response,
        # not a re-anchoring or reuse of the stored SR template response.
        for shifted, _, endpoint in nonjme_event_views((original,), year, None, {}, nonjme):
            if endpoint != "nominal":
                # The canonical shift returns the identical record only when
                # the endpoint did not change any event in the whole chunk.
                yield endpoint, (None if shifted is original else
                                 _inferred_slices(shifted, era, momenta["nominal"]))


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


def _event_branches(root, friend):
    branches = root["Events"].keys()
    if friend is None:
        return branches
    # The friend supplies all corrected Mixed fields. Avoid reading the old
    # Mixed candidates (and their per-branch jagged length helpers) twice.
    return [name for name in branches if name != "nTopMixed1pct"
            and not name.startswith(("TopMixed1pct_", "nTopMixed1pct_"))]



def _attach_friend(friend, chunk, start, allowed):
    if friend is None:
        return chunk[allowed], None
    return read_friend_slice(friend, chunk, start, start + len(chunk), selection=allowed)



def compute_response(task, contract, *, use_friend=True):
    path = _task_source(task)
    if _mc_batch_kind(task) != "standard":
        raise ValueError("VR response applies only to background MC")
    accepted = _read_receipt(task)
    if accepted is None or accepted.get("kind") != "background":
        raise ValueError("Missing accepted background-only VR nominal receipt")
    era = str(task["year"])
    year = era_year(era)
    low, high, signals = layout(era)
    metadata = validate_root(path, year)
    datasets = metadata.get("datasets", {})
    if not datasets or any(r.get("is_data") or r.get("is_signal") for r in datasets.values()):
        raise ValueError("VR response input is not exclusively background MC")
    excluded = excluded_ids(datasets)
    scopes = contract["scopes"][era]
    shape_endpoints = _shape_endpoints(era, scopes)
    shape_names = {_endpoint_name(e, era)[0] for e in shape_endpoints}
    nominal, weights, shapes, coverage = {}, defaultdict(dict), defaultdict(dict), defaultdict(dict)
    events = 0
    friend_context = open_friend(task, path) if use_friend else nullcontext()
    with uproot.open(path, object_cache=None, array_cache=None) as root, friend_context as friend:
        policy = topw_file_input_policy(root)
        reader = TopWEvents(root, policy)
        for chunk in reader.iterate(_event_branches(root, friend), step_size=RESPONSE_CHUNK):
            start = events
            events += len(chunk)
            if np.any(np.asarray(chunk["is_data"], bool)):
                raise ValueError("Observed data are forbidden in VR responses")
            allowed = np.zeros(len(chunk), dtype=bool)
            for mask, *_ in _row_samples(chunk, metadata, "background", signals, excluded, set(), era):
                allowed |= mask
            allowed &= np.asarray(chunk["pass_base_common"], bool) & np.asarray(chunk["pass_signal_trigger"], bool)
            selected, resolved = _attach_friend(friend, chunk, start, allowed)
            if not len(selected):
                continue
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
                                a, metadata, "background", signals, excluded, set(), era):
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
    if events != task["entries"]:
        raise ValueError("Read event coverage differs from the accepted MC inventory")
    _check_nominal_response(accepted["bins"], nominal)

    def pairs(values):
        return {n: {d: _response_sparse(h) for d, h in pair.items()} for n, pair in values.items()}

    return dict(schema=RESPONSE_SCHEMA, input=task["input"], era=era, fingerprint=task["fingerprint"],
                card_sha256=contract["card_sha256"], cr_card_sha256=contract["cr_card_sha256"],
                events=events, sr_observations_read=False,
                nominal={sample: _response_sparse(rows) for sample, rows in nominal.items()},
                weight_deltas={sample: pairs(values) for sample, values in weights.items()},
                shape_endpoints={sample: pairs(values) for sample, values in shapes.items()},
                weight_coverage=dict(coverage), nominal_receipt_reused=True)


def compute(task, *, tight_mixed=False, use_friend=True, signal_normalization=None):
    era = str(task["year"])
    path = _task_source(task)
    if tight_mixed and path != ROOT / task["input"]:
        raise ValueError("Tight-Mixed MC inputs cannot redirect through symlinks")
    low_layout, high_layout, signals = layout(era)
    year = era_year(era)
    metadata = validate_root(path, year)
    kind = "background" if path.name.startswith("mc_") else "signal"
    datasets = metadata.get("datasets", {})
    if not datasets or any(r.get("is_data") or bool(r.get("is_signal")) != (kind == "signal")
                           for r in datasets.values()):
        raise ValueError("Input metadata is not exclusively the expected MC kind")
    signal_basis = None
    if kind == "signal":
        signal_basis = (signal_normalization if signal_normalization is not None
                        else _registered_signal_normalization(era))
        if (signal_basis["era"] != era or signal_basis["basis"] != SIGNAL_NORM_BASIS
                or any(signal_basis["datasets"].get(str(key)) != record["dataset"]
                       for key, record in datasets.items())):
            raise ValueError("Signal shard does not belong to the registered normalization inventory")
    normalization = signal_basis["norm"] if signal_basis is not None else _normalization(era)
    excluded = excluded_ids(datasets)
    sums, seen, events = {}, set(), 0
    overlap = Counter(dict.fromkeys((*REGIONS, "SR_preselection"), 0))
    def selections(original, resolved):
        if tight_mixed:
            yield from _tight_mixed_slices(original, era, (low_layout, high_layout), cached_resolved=resolved)
        else:
            for a, resolved, low_angle, high_angle, gnn in _nominal_slices(original, era, cached_resolved=resolved):
                bins, clean = _select_and_bin(a, resolved, low_angle, high_angle, gnn, era, (low_layout, high_layout))
                yield a, bins, clean, {}

    friend_context = open_friend(task, path) if use_friend else nullcontext()
    with uproot.open(path, object_cache=None, array_cache=None) as root, friend_context as friend:
        policy = topw_file_input_policy(root)
        reader = TopWEvents(root, policy)
        step_size = FRIEND_CHUNK if friend is not None else 1_000_000
        for chunk in reader.iterate(_event_branches(root, friend), step_size=step_size):
            start = events
            events += len(chunk)
            if np.any(np.asarray(chunk["is_data"], bool)):
                raise ValueError("Observed data are forbidden")
            allowed = np.zeros(len(chunk), dtype=bool)
            for mask, *_ in _row_samples(chunk, metadata, kind, signals, excluded, seen, era, normalization):
                allowed |= mask
            # These two flags are unchanged by the adopted central rebuild.
            allowed &= np.asarray(chunk["pass_base_common"], bool) & np.asarray(chunk["pass_signal_trigger"], bool)
            selected, resolved = _attach_friend(friend, chunk, start, allowed)
            if not len(selected):
                continue
            for original in corrected_central_chunks((selected,), year, {}):
                for a, bins, clean, intersections in selections(original, resolved):
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
                        value = _histogram(bins[mask], norm * base["nominal"] * tw["nominal"])
                        sums[sample] = sums.get(sample, 0) + value
    if events != task["entries"]:
        raise ValueError("Read event coverage differs from the accepted MC inventory")
    if tight_mixed and sum(int(rows[:, 2].sum()) for rows in sums.values()) != overlap["SR_preselection"]:
        raise ValueError("Tight-Mixed weighted MC coverage disagrees with its raw selection")
    return dict(schema=TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA,
                era=era, input=task["input"], fingerprint=task["fingerprint"],
                kind=kind, events=events, signal_points_seen=sorted(seen),
                bins={k: {str(i): x[i].tolist() for i in np.flatnonzero(x[:, 2])}
                      for k, x in sums.items()}, sr_observations_read=False,
                **({"signal_normalization": {k: signal_basis[k] for k in ("basis", "era", "sha256")}}
                   if signal_basis is not None else {}),
                **({"overlap": dict(overlap), "mc_only": True} if tight_mixed else {}))


@lru_cache(maxsize=1)
def inventory():
    """Use the two accepted input inventories without reopening production gates."""
    found = {era: [] for era in ERAS}
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
            name = Path(task["input"]).name
            if name.startswith("data_"):
                continue
            if not MC_NAME.fullmatch(name):
                raise ValueError("Unrecognized MC input name: " + name)
            found[str(task["year"])].append(task)
    paths = [t["input"] for tasks in found.values() for t in tasks]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate MC inputs")
    return found


def _pilot(tasks):
    # Only the selected MC shards are opened. Prefer modest-size shards so
    # this exercises both kinds and all target eras without a full scan.
    return [min((t for t in tasks if Path(t["input"]).name.startswith(prefix)),
                key=lambda t: (abs(t["entries"] - 12000), t["input"]))
            for prefix in ("mc_", "signal_")]


def _receipt_path(task, response=False, tight_mixed=False):
    return OUTPUT / ("tight_mixed_shards" if tight_mixed else "response_shards" if response else "shards") / str(task["year"]) / (
        Path(task["input"]).stem + ".json.gz")


def _read_receipt(task, *, tight_mixed=False):
    path = _receipt_path(task, tight_mixed=tight_mixed)
    if not path.is_file():
        return None
    result = read_json(path)
    if (result.get("schema") != (TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA) or result.get("input") != task["input"]
            or result.get("fingerprint") != task["fingerprint"]
            or result.get("events") != task["entries"] or result.get("sr_observations_read") is not False):
        raise ValueError("Divergent VR MC receipt: " + str(path))
    if Path(task["input"]).name.startswith("signal_"):
        expected = _registered_signal_normalization(str(task["year"]))
        basis = result.get("signal_normalization")
        cutoff = expected["legacy_before_ns"]
        if basis is not None and basis != {k: expected[k] for k in ("basis", "era", "sha256")}:
            raise ValueError("Divergent signal normalization in VR MC receipt: " + str(path))
        if basis is None and result["bins"] and cutoff is not None and path.stat().st_mtime_ns >= cutoff:
            raise ValueError("Cannot establish the signal normalization of VR MC receipt: " + str(path))
    if tight_mixed:
        counts = result.get("overlap", {})
        entries = sum(x[2] for rows in result["bins"].values() for x in rows.values())
        if (result.get("mc_only") is not True or set(counts) != {*REGIONS, "SR_preselection"}
                or any(counts[k] != 0 for k in REGIONS) or counts["SR_preselection"] != entries):
            raise ValueError("Invalid tight-Mixed overlap or coverage receipt: " + str(path))
    return result


def _read_response_receipt(task, contract):
    path = _receipt_path(task, response=True)
    if not path.is_file():
        return None
    result = read_json(path)
    if (result.get("schema") != RESPONSE_SCHEMA or result.get("input") != task["input"]
            or result.get("fingerprint") != task["fingerprint"] or result.get("events") != task["entries"]
            or result.get("card_sha256") != contract["card_sha256"]
            or result.get("cr_card_sha256") != contract["cr_card_sha256"]
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
    for name, file, pid_key, workers_key in (
            ("limits", "workflow_state.json", "pid", "requested_workers"),
            ("impacts", "impact_status.json", "controller_pid", "workers")):
        path = run_root / name / file
        if not path.is_file():
            continue
        state = read_json(path)
        pid = state.get(pid_key)
        if pid not in rows:
            continue
        family = {pid}
        while True:
            new = {p for p, (parent, _, _) in rows.items() if parent in family} - family
            if not new:
                break
            family |= new
        native = [rows[p][1] for p in family - {pid} if rows[p][2] == "combine"]
        futures = max(int(state[workers_key]), len(family) - 1)
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
        if data.get("status") == "running" and data.get("pid") in rows:
            materialized = [rss for parent, rss, _ in rows.values() if parent == data["pid"]]
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
    return "signal" if Path(task["input"]).name.startswith("signal_") else "standard"


def _mc_cpu_slots():
    # Half the logical CPUs is a conservative physical-core estimate on hep2.
    return max(0, int(len(os.sched_getaffinity(0)) // 2 - os.getloadavg()[0]))


def _collect(all_tasks, pilot, *, tight_mixed=False):
    coverage = {era: dict(background=[0, 0], signal=[0, 0]) for era in ERAS}
    signal_eras = defaultdict(set)
    samples = {}
    per_era = {}
    overlaps = {}
    for era, tasks in all_tasks.items():
        sums = {}
        per_era[era] = sums
        overlap = Counter(dict.fromkeys((*REGIONS, "SR_preselection"), 0))
        overlaps[era] = overlap
        for task in tasks:
            kind = "background" if Path(task["input"]).name.startswith("mc_") else "signal"
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
                target = sums.setdefault(sample, np.zeros((148, 3)))
                for i, values in entries.items():
                    target[int(i)] += np.asarray(values, dtype=float)
        for sample, values in sums.items():
            samples[sample] = samples.get(sample, 0) + values
    complete = all(a == b for era in coverage.values() for a, b in era.values())
    _, high, signals = layout("2024")
    bkg = sum((x for k, x in samples.items() if k not in signals), np.zeros((148, 3)))
    full_signals = sorted(k for k in signals if signal_eras[k] == set(ERAS))
    definitions = [(mode, key, j, lo, hi, predicates) for mode, groups in (
        ("lowdm", [(key, edges, {}) for key, edges in layout("2024")[0].items()]),
        ("highdm", [(f"class{i}", c["edges"], c["predicates"]) for i, c in enumerate(high, 1)]))
        for key, edges, predicates in groups for j, (lo, hi) in enumerate(zip(edges, edges[1:]), 1)]
    rows = []
    zero = np.zeros((148, 3))
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
    return dict(schema=TIGHT_MIXED_SCHEMA if tight_mixed else SCHEMA,
                status="complete" if complete and len(full_signals) == len(signals)
                else "incomplete", pilot=pilot, sr_observations_read=False, vr_observations_read=False,
                mc_only=True, fit_or_transfer_factors_applied=False, fields=list(FIELDS),
                coverage=coverage, full_era_signal_points=len(full_signals),
                missing_era_signal_points={k: sorted(set(ERAS) - signal_eras[k])
                                           for k in sorted(signals) if k not in full_signals},
                low_edges_by_era={e: layout(e)[0] for e in ERAS}, high_categories=high,
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
    return dict(schema=RESPONSE_SCHEMA, status="complete" if complete else "incomplete",
                coverage=coverage, source_nominal_reused=True,
                axes=dict(vr_bin=148, nb_groups=["Nb1", "Nb2plus"], recoil_edges=NATIVE_EDGES,
                          recoil_last_open=True, flat_index="(vr_bin*2+nb_index)*8+recoil_index; zero-based"),
                fields=list(FIELDS), era_samples=era_samples, accepted_nominal_era_samples=accepted,
                expected_shape_nuisances=contract["scopes"], evaluated=evaluated, unsupported=unavailable,
                all_applicable_responses_available=complete and not any(unavailable.values()),
                parent_cards={k: contract[k] for k in ("card", "card_sha256", "cr_card", "cr_card_sha256")},
                cr_card_shapes=contract["cr_card_shapes"], fit_only_qcd_support=contract["fit_only_qcd_support"],
                projection_owned=list(PROJECTION_TERMS),
                endpoint_scope="accepted background shape names only; unavailable terms are explicitly listed; "
                               "measured transfer shapes and normalization terms are projection-owned",
                object_method=dict(jme="canonical anchored Total JES/JER, VR reinference",
                                   nonjme="canonical event shift with unchanged nominal TROTA inputs; "
                                          "new VR response, not reused absolute SR templates"),
                no_fit_or_transfer_factors_applied=True, sr_observations_read=False, mc_only=True)


def run(pilot=False, workers=4, response=False, *, tight_mixed=False):
    maximum = TIGHT_MIXED_MAX_WORKERS if tight_mixed else (RESPONSE_MAX_WORKERS if response else MC_MAX_WORKERS)
    if ROOT != Path("/hep2-scratch/twkim/postprocessing/TROTASR") or not 1 <= workers <= maximum:
        raise ValueError(f"VR MC runs only in the canonical hep2 runtime with 1–{maximum} workers")
    if response and pilot:
        raise ValueError("VR response production requires full accepted nominal coverage; no pilot mode")
    if response and tight_mixed:
        raise ValueError("Tight-Mixed is a nominal MC-only evaluation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for era in ERAS:
            layout(era)
        contract = _response_contract() if response else None
        worker_contract = ({k: v for k, v in contract.items() if k != "nominal_era_samples"}
                           if response else None)
        receipt = ((lambda task: _read_response_receipt(task, contract)) if response else
                   (lambda task: _read_receipt(task, tight_mixed=True)) if tight_mixed else _read_receipt)
        tag = "tight_mixed" if tight_mixed else "response" if response else "mc"
        schema = TIGHT_MIXED_SCHEMA if tight_mixed else RESPONSE_SCHEMA if response else SCHEMA
        all_tasks = inventory()
        selected = {era: _pilot(tasks) if pilot else tasks for era, tasks in all_tasks.items()}
        if response:
            selected = {era: [task for task in tasks if _mc_batch_kind(task) == "standard"]
                        for era, tasks in selected.items()}
            if any(not tasks for tasks in selected.values()):
                raise ValueError("Missing a background VR era")
        pending = {era: [t for t in tasks if receipt(t) is None] for era, tasks in selected.items()}
        total = sum(map(len, selected.values()))
        done = total - sum(map(len, pending.values()))
        signal_normalizations = {era: _registered_signal_normalization(era) for era, tasks in pending.items()
                                 if any(_mc_batch_kind(task) == "signal" for task in tasks)}
        failures = []
        reserve = 32 * GIB if response or tight_mixed else MC_RESERVE_BYTES
        emergency = 24 * GIB if response or tight_mixed else MC_RESERVE_BYTES
        signal_limit = workers if tight_mixed else MC_SIGNAL_WORKERS

        def state(status, active=0, pool_workers=0, kind=None, peak=0):
            write_json(OUTPUT / f"{tag}_state.json", dict(
                schema=schema, status=status, pilot=pilot, pid=os.getpid(), workers=workers,
                total=total, completed=done, active=active, pool_workers=pool_workers, failures=failures,
                batch_kind=kind, pool_peak_rss_gib=peak / GIB,
                memory_reserve_gib=reserve / GIB,
                sr_observations_read=False, vr_observations_read=False, updated=time.time()))

        def run_pending():
            nonlocal done
            tasks = deque(task for era in pending.values() for task in era)
            context = mp.get_context("spawn")
            live = []
            last_state = 0
            peak = 0
            kind = None
            memory_stopped = False
            try:
                while tasks or live:
                    peak = max(peak, _mc_pool_high_water(live))
                    if live and _available_memory() < emergency:
                        memory_stopped = True
                        for worker in live:
                            worker.retiring = True
                            if worker.process.is_alive():
                                worker.process.kill()
                        return False

                    active = sum(w.task is not None for w in live)
                    if tasks and active < workers:
                        slots, budget = _resource_room(reserve_vr_data=True, reserve_bytes=reserve)
                        if slots < 0:
                            budget = 0
                        target = min(workers, max(1, _mc_cpu_slots()))
                        signal_active = sum(w.task is not None and _mc_batch_kind(w.task) == "signal" for w in live)
                        while tasks and budget > 0 and active < target:
                            if (response or tight_mixed) and budget < RESPONSE_ADMISSION_BYTES:
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
                            if response or tight_mixed:
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
                        state("running", active, len(live), kind, peak)
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
                    done = sum(_receipt_path(task, response=response, tight_mixed=tight_mixed).is_file()
                               for batch in selected.values() for task in batch)
                    state("memory_stopped", kind=kind, peak=peak)

        state("running")
        try:
            if not run_pending():
                return 1
            report = (_collect_response(selected, contract) if response else
                      _collect(all_tasks, pilot, tight_mixed=True) if tight_mixed else _collect(all_tasks, pilot))
            write_json(OUTPUT / f"{tag}_summary.json.gz", report)
            state("needs_attention" if failures or (not pilot and report["status"] != "complete")
                  else "pilot_complete" if pilot else "complete")
        except BaseException:
            state("needs_attention")
            raise
        return 1 if failures or (not pilot and report["status"] != "complete") else 0
