"""Observed JetMET validation-region histograms with the SR kept blinded."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import os
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from TROTASR.utils.event_selections import OVERLAP_FIELDS, topology_counts
from TROTASR.utils.ids import object_masks
from TROTASR.utils.io import read_json, validate_root, write_json
from TROTASR.utils.normalization import dataset_label
from TROTASR.utils.paths import ERAS, ROOT, era_year, tagging_definition
from TROTASR.utils.region_kinematics import jet_kinematics
from TROTASR.utils.shape_kinematics import jet_block, met_contribution, varied_view
from TROTASR.utils.trota_candidates import input_fields
from TROTASR.utils.trota_inference import MIXED_FLOATS, MIXED_INTS, P4_FIELDS, bounded_slices, infer_chunk, p4_view
from TROTASR.workflows.validation_regions import (
    OUTPUT, _accepted_input_path, _active_controller, _headroom, _input_inventory, _input_kind,
    _models, _select_and_bin, angular_candidates, geometry,
)
from TROTASR.workflows.resources import process_identity


SCHEMA = "trotasr_vr_jetmet_v1"
MAX_WORKERS = 12
IDENTITY = ("dataset_id", "is_data", "is_signal")
JETMET_SELECTION = ("pass_base_common", "pass_signal_trigger")
JETS = ("jet_corrected_pt", "jet_eta_all", "jet_phi_all", "jet_id_all", "met", "met_phi")
TYPE1 = ("jet_nanoaod_pt", "jet_ch_em_ef", "jet_ne_em_ef", "jet_muon_subtr_factor")
VETO = (
    "electron_pt_all", "electron_eta_all", "electron_cutbased_all", "electron_mini_iso_all",
    "muon_pt_all", "muon_eta_all", "muon_mini_iso_all", "muon_loose_id_all", "muon_medium_id_all",
    "photon_pt_all", "photon_eta_all", "photon_cutbased_all", "photon_electron_veto_all",
    "tau_pt_all", "tau_phi_all", "tau_eta_all", "tau_dz_all", "tau_decay_mode_all", "tau_deeptau_vsjet_all",
)
MIXED = ("nTopMixed1pct", *("TopMixed1pct_" + x for x in (*MIXED_INTS, *MIXED_FLOATS)))


def _project(tree, entries, fields):
    """Project selection columns; group reads by each branch's basket."""
    cols = {}
    for name in fields:
        b = tree[name]
        offsets = np.asarray(b.entry_offsets)
        basket = np.searchsorted(offsets[1:], entries, side="right")
        pieces = []
        for indices in np.split(entries, np.flatnonzero(np.diff(basket)) + 1):
            start, stop = int(indices[0]), int(indices[-1]) + 1
            pieces.append(b.array(entry_start=start, entry_stop=stop)[indices - start])
        cols[name] = pieces[0] if len(pieces) == 1 else ak.concatenate(pieces)
    return ak.zip(cols, depth_limit=1)


def _exact_rows(tree, entries):
    """Request full fields only for consecutive entries already proven to be VR."""
    pieces = [tree.arrays(entry_start=int(i[0]), entry_stop=int(i[-1]) + 1)
              for i in np.split(entries, np.flatnonzero(np.diff(entries) != 1) + 1)]
    return pieces[0] if len(pieces) == 1 else ak.concatenate(pieces)


def _central_met(a):
    met, phi = np.asarray(a["met"]), np.asarray(a["met_phi"])
    dx, dy = met_contribution(a, a["jet_corrected_pt"])
    x, y = (met * np.cos(phi) + dx) - dx, (met * np.sin(phi) + dy) - dy
    return np.hypot(x, y), np.arctan2(y, x)


def _jets(a, year):
    tag = tagging_definition(year)
    pt, eta, phi, btag = (a[k] for k in (*JETS[:3], tag["btag_branch"]))
    good = (pt > 30) & (abs(eta) < 2.4) & (a["jet_id_all"] != 0)
    return pt, eta, phi, btag, good


def _angles(a, year):
    met, met_phi = _central_met(a)
    pt, eta, phi, btag, good = _jets(a, year)
    j = jet_block(pt, eta, phi, good, good & (btag > tagging_definition(year)["medium"]), met_phi)
    return angular_candidates(dict(
        pass_base_common=np.ones(len(a), bool), pass_signal_trigger=np.ones(len(a), bool),
        pass_zero_tau=np.ones(len(a), bool), pass_no_veto_leptons=np.ones(len(a), bool),
        pass_met_250=met > 250, pass_ht_300=j["ht"] > 300, nb_medium=j["nb"], njet=j["njet"],
        **{f"j{i}_met_dphi": v for i, v in enumerate(j["angles"], 1)},
    ))


def _veto(a, year, met, phi):
    masks = object_masks(a, year)
    tau_mt = np.sqrt(2 * a["tau_pt_all"] * met * (1 - np.cos(a["tau_phi_all"] - phi)))
    tau = ((a["tau_pt_all"] > 20) & (abs(a["tau_eta_all"]) < 2.5) & (abs(a["tau_dz_all"]) < .2)
           & (a["tau_decay_mode_all"] != 5) & (a["tau_decay_mode_all"] != 6)
           & (a["tau_deeptau_vsjet_all"] >= 5) & (tau_mt < 100))
    return ~np.asarray(ak.any(tau, axis=1) | ak.any(masks["electron_veto"], axis=1)
                       | ak.any(masks["muon_loose"], axis=1))


def _boosted(a, year):
    tag = tagging_definition(year)
    pt, eta, sd = (a[k] for k in ("fatjet_corrected_pt", "fatjet_eta_all", "fatjet_msoftdrop_all"))
    fid = (a["fatjet_id_all"] != 0) & (abs(eta) < 2)
    a = ak.with_field(a, fid & (pt > 400) & (sd > 105) & (a["fatjet_top_score_all"] > tag["top"]),
                      "fatjet_boosted_top_pass_all")
    return ak.with_field(a, fid & (pt > 200) & (sd > 60) & (sd < 105) & (a["fatjet_w_score_all"] > tag["w"]),
                         "fatjet_boosted_w_pass_all")


def _bin_chunk(tree, entries, year, era, bins):
    if not len(entries):
        return
    tag = tagging_definition(year)
    jets = {*JETS, *TYPE1, tag["btag_branch"]}
    low, high = _angles(_project(tree, entries, sorted(jets)), year)
    support = low | high
    if not support.any():
        return
    entries, low, high = entries[support], low[support], high[support]
    fields = (jets | set(VETO) | {n for group in input_fields(year) for n in group}
              | set(OVERLAP_FIELDS) | set(P4_FIELDS) | {k.replace("nanoaod", "corrected") for k in P4_FIELDS}
              | {"file_id", "entry", "fatjet_id_all", "fatjet_msoftdrop_all"})
    fields -= {"fatjet_boosted_top_pass_all", "fatjet_boosted_w_pass_all"}
    a = _project(tree, entries, sorted(fields))
    met, phi = _central_met(a)
    pt, eta, jphi, btag, good = _jets(a, year)
    mtb = jet_kinematics(pt, eta, jphi, btag, good, met.astype(float), phi.astype(float), year)["mtb"]
    high &= np.isfinite(mtb) & (mtb >= 175.)
    keep = (low | high) & _veto(a, year, met, phi)
    if not keep.any():
        return
    a, entries, low, high = a[keep], entries[keep], low[keep], high[keep]
    a = p4_view(_boosted(a, year), {k: a[k.replace("nanoaod", "corrected")] for k in P4_FIELDS})
    trota, gnn = _models(era)
    low_bins, high_bins = geometry(era)
    for sl in bounded_slices(a, year=year):
        inferred, resolved, _, _ = infer_chunk(a[sl], trota)
        tags, _, _, _ = topology_counts(inferred, resolved, "SR", year=year)
        ntag = tags.sum(axis=1)
        accepted = (low[sl] & (ntag == 0)) | (high[sl] & (ntag >= 1))
        if not accepted.any():
            continue
        # Before this read, topology has excluded low-dM SR candidates
        # from the high-dM angular sideband. High-dM SR fails its angle cut.
        final = _exact_rows(tree, entries[sl][accepted])
        for k in MIXED:
            final = ak.with_field(final, inferred[k][accepted], k)
        momenta = {k.replace("nanoaod", "corrected"): final[k.replace("nanoaod", "corrected")] for k in P4_FIELDS}
        final = varied_view(final, momenta, year=year)
        flo, fhi = angular_candidates(final)
        index, _ = _select_and_bin(final, resolved, flo, fhi, gnn, era, (low_bins, high_bins))
        if np.any(index < 0) or not np.array_equal(index < 30, low[sl][accepted]):
            raise ValueError("VR selection projection and canonical final selection disagree")
        bins += np.bincount(index, minlength=len(bins))


def _source(task):
    return _accepted_input_path(task)


def _work(task):
    era = str(task["year"])
    year, path = era_year(era), _source(task)
    meta = validate_root(path, year)
    labels = {int(k): dataset_label(meta, int(k)) for k in meta["datasets"]}
    if not labels or any(not is_data or is_sig or dataset == "unknown"
                         for dataset, _, is_data, is_sig in labels.values()):
        raise ValueError("Observed shard metadata is not data-only")
    jetmet = [k for k, (_, process, _, _) in labels.items() if process == "JetMET"]
    bins = np.zeros(148, np.int64)
    fields = IDENTITY + JETMET_SELECTION if jetmet else IDENTITY
    events = 0
    with uproot.open(path, object_cache=None, array_cache=None) as f:
        tree = f["Events"]
        if tree.num_entries != task["entries"]:
            raise ValueError("Observed Events count differs from the inventory")
        for start in range(0, tree.num_entries, 12000):
            stop = min(tree.num_entries, start + 12000)
            a = tree.arrays(fields, entry_start=start, entry_stop=stop)
            dataset = np.asarray(a["dataset_id"])
            if (len(dataset) != stop - start or not np.all(a["is_data"])
                    or np.any(a["is_signal"]) or not np.all(np.isin(dataset, tuple(labels)))):
                raise ValueError("Unexpected event range or dataset identity in observed shard")
            events += len(dataset)
            if jetmet:
                use = (np.isin(dataset, jetmet) & np.asarray(a["pass_base_common"], bool)
                       & np.asarray(a["pass_signal_trigger"], bool))
                _bin_chunk(tree, np.flatnonzero(use) + start, year, era, bins)
    write_json(_receipt_path(task), dict(schema=SCHEMA, input=task["input"], era=era,
               events=events, event_range=[0, events],
               bins=bins.tolist(), sr_histograms_filled=False, sr_observations_used=False))


def _inventory():
    tasks = {era: [t for t in group if _input_kind(t) == "data"]
             for era, group in _input_inventory().items()}
    if not all(tasks.values()):
        raise ValueError("Missing observed shards for an accepted era")
    return tasks


def _receipt_path(task):
    return OUTPUT / "data_shards" / str(task["year"]) / (Path(task["input"]).stem + ".json.gz")


def _receipt(task):
    path = _receipt_path(task)
    if not path.is_file():
        return None
    d = read_json(path)
    era, expected = str(task["year"]), task["entries"]
    bins = d.get("bins", ())
    if (d.get("schema") != SCHEMA or d.get("era", era) != era
            or d.get("sr_observations_used") is not False or d.get("sr_histograms_filled") is not False
            or d.get("events", expected) != expected or d.get("event_range", [0, expected]) != [0, expected]
            or len(bins) != 148 or any(type(n) is not int or n < 0 for n in bins)
            or sum(bins) > expected):
        raise ValueError("Divergent VR observed receipt")
    return bins


def _capacity():
    state = OUTPUT / "mc_state.json"
    mc_reserve = 0
    if state.is_file():
        d = read_json(state)
        pid = _active_controller(d, OUTPUT / "controller.lock", legacy_args=("--vr-mc",))
        if pid is not None:
            p = Path("/proc") / str(pid)
            try:
                active = len((p / "task" / str(pid) / "children").read_text().split())
            except FileNotFoundError:
                active = 0
            mc_reserve = max(0, int(d["workers"]) - active)
    return max(0, _headroom() - mc_reserve)


def run(workers):
    if ROOT != Path("/hep2-scratch/twkim/postprocessing/TROTASR") or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"VR data runs only in the canonical hep2 runtime with 1–{MAX_WORKERS} workers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "data_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = _inventory()
        pending = {e: [t for t in group if _receipt(t) is None] for e, group in tasks.items()}
        total = sum(map(len, tasks.values()))
        done = total - sum(map(len, pending.values()))
        failures = []
        controller = process_identity(os.getpid())

        def state(status, active=0):
            write_json(OUTPUT / "data_state.json", dict(schema=SCHEMA, status=status, pid=os.getpid(),
                       start_ticks=controller["start_ticks"],
                       workers=workers, total=total, completed=done, active=active, failures=failures,
                       sr_observations_used=False, sr_histograms_filled=False, updated=time.time()))

        if done < total and _capacity() < 1:
            raise RuntimeError("Insufficient hep2 headroom for observed VR")
        state("running")
        try:
            for group in pending.values():
                if not group:
                    continue
                capacity = min(workers, _capacity())
                if capacity < 1:
                    raise RuntimeError("hep2 headroom changed; resume from saved VR receipts")
                queue = iter(group)
                with ProcessPoolExecutor(max_workers=capacity) as pool:
                    active, drained = {}, False
                    while active or not drained:
                        while not drained and len(active) < min(capacity, _capacity()):
                            t = next(queue, None)
                            if t is None:
                                drained = True
                            else:
                                active[pool.submit(_work, t)] = t
                        state("running", len(active))
                        if not active:
                            if not drained:
                                time.sleep(10)
                            continue
                        finished, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                        for future in finished:
                            t = active.pop(future)
                            try:
                                future.result()
                            except Exception as exc:
                                failures.append(dict(input=t["input"], error_type=type(exc).__name__,
                                                     error=str(exc) if isinstance(exc, (ValueError, KeyError)) else "worker failed"))
                            else:
                                done += 1
            counts, coverage = {}, {}
            for era, group in tasks.items():
                per_era = np.zeros(148, np.int64)
                n = 0
                for t in group:
                    rows = _receipt(t)
                    if rows is not None:
                        per_era += rows
                        n += 1
                counts[era], coverage[era] = per_era.tolist(), [n, len(group)]
            status = "complete" if done == total and not failures else "needs_attention"
            write_json(OUTPUT / "data_summary.json.gz", dict(schema=SCHEMA, status=status, coverage=coverage,
                       low_bins=30, high_bins=118, observed_by_era=counts,
                       observed_total=np.sum(list(counts.values()), axis=0).tolist(),
                       sr_observations_used=False, sr_histograms_filled=False))
            state(status)
        except BaseException:
            state("needs_attention")
            raise
        return int(status != "complete")


if __name__ == "__main__":
    from TROTASR.workflows.validation_data import run as run_stage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    raise SystemExit(run_stage(parser.parse_args().workers))
