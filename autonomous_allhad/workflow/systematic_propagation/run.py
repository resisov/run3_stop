#!/usr/bin/env python3
"""Separate kinematic-variation jobs using the existing analysis executables."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).absolute().parent
SHIFTS = ("nominal", "metUnclusteredUp", "metUnclusteredDown")
OBJECT_SHIFTS = tuple(name + direction for name in (
    "electronScale", "electronSmear", "photonScale", "photonSmear",
    "muonScale", "muonResolution", "tauEnergyScale",
) for direction in ("Up", "Down"))
TROTA_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh"
WEIGHTS = ("pileup", "btagSF", "electron_id", "electron_reco", "muon_id",
           "muon_iso", "muon_hlt", "photon_id", "met_trigger", "photon_trigger")


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def load_config(path):
    config = read(path)
    if "shared_config" in config:
        shared_path = eos(config["shared_config"])
        if sha(shared_path) != config["shared_config_sha256"]:
            raise RuntimeError("shared configuration changed")
        shared = read(shared_path)
        pins = {**shared.pop("pins"), **config.get("pins", {})}
        shared.update(config)
        config = dict(shared, pins=pins)
    return config


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eos(path):
    path = Path(path).absolute()
    if not str(path).startswith("/eos/") or any(x in ("afs", "tmp") for x in path.parts):
        raise ValueError("expected EOS path: " + str(path))
    return path


def execute(argv, env, cwd):
    print(json.dumps({"command": list(map(str, argv))}), flush=True)
    subprocess.run(list(map(str, argv)), env=env, cwd=cwd, check=True)


def xrootd(arguments, allow_missing=False):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    executable = shutil.which(arguments[0], path="/usr/bin:/bin")
    if executable is None:
        raise RuntimeError("missing XRootD client: " + arguments[0])
    command = [executable, *map(str, arguments[1:])]
    for attempt in range(3):
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        if allow_missing and "[3011]" in result.stderr and "No such file or directory" in result.stderr:
            return None
        if attempt < 2:
            print(json.dumps(dict(transfer_retry=attempt + 1, command=command,
                                  error=result.stderr[-2000:])), flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"XRootD failed ({result.returncode}): {command}: {result.stderr[-2000:]}")


def stage_input(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    xrootd(["xrdcp", "--force", "--nopbar", "--cksum", "adler32",
            "root://eosuser.cern.ch/" + str(eos(source)), partial])
    partial.replace(destination)
    return destination


def publish_product(source, destination):
    destination = eos(destination)
    xrootd(["xrdfs", "root://eosuser.cern.ch", "mkdir", "-p", destination.parent])
    xrootd(["xrdcp", "--force", "--nopbar", "--posc", "--cksum", "adler32",
            source, "root://eosuser.cern.ch/" + str(destination)])


def checkpoint(path, value, work):
    local = work / "checkpoint.json"
    write(local, value)
    publish_product(local, path)


def write_tree(target, name, values):
    import awkward as ak

    if len(next(iter(values.values()))):
        target[name] = values
    else:
        # uproot 4 cannot extend an empty jagged counter; create only its schema.
        target.mktree(name, {key: ak.type(value).type for key, value in values.items()})


def prepare_payload(config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(config["object_payload_bundle"]) as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("unsupported payload member: " + member.name)
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    return destination


def source_records(sidecar, year, metadata=None):
    metadata = read(sidecar) if metadata is None else metadata
    if metadata["status"] not in ("complete", "complete_with_bad_files"):
        raise ValueError("source shard is not complete: " + str(sidecar))
    datasets = {v["dataset"]: v for v in metadata["datasets"].values()}
    records = []
    for index, item in enumerate(metadata["files"]):
        info = datasets[item["dataset"]]
        if info["is_data"]:
            raise ValueError("shape production requires MC sources")
        if item["read_status"] != "success":
            continue
        record = dict(dataset=item["dataset"], process_group=item["process"],
                      file_path=item["file_path"], file_index=index, year=str(year),
                      is_data=False, is_signal=info["is_signal"],
                      is_background=info["is_background"], xsec_pb=info["xsec_pb"])
        if item.get("fastsim_trigger_bypass"):
            record["simulation_type"] = "FastSim signal dataset"
        records.append(record)
    if len(records) != metadata["files_processed"]:
        raise ValueError("source file accounting mismatch")
    return records


def shift_objects(chunk, year, shift, payload):
    import awkward as ak
    import numpy as np

    if shift not in ("nominal", *OBJECT_SHIFTS):
        raise ValueError("unsupported intermediate-object variation: " + shift)
    output = dict(chunk)
    if shift == "nominal":
        return output
    calibration = importlib.import_module(f"autonomous_allhad.object_corrections_{year}")
    flavor = next(name for name in ("electron", "photon", "muon", "tau") if shift.startswith(name))
    prefix = flavor.capitalize()
    arrays = {key: chunk[key] for key in ("run", "luminosityBlock", "event")}
    for field in ("pt", "mass", "eta", "phi"):
        if flavor == "photon" and field == "mass":
            continue
        branch = f"{flavor}_nanoaod_{field}" if field in ("pt", "mass") else f"{flavor}_{field}_all"
        arrays[f"{prefix}_{field}"] = chunk[branch]
    fields = {
        "electron": {"r9": "r9", "seedGain": "seed_gain"},
        "photon": {"r9": "r9", "seedGain": "seed_gain"},
        "muon": {"charge": "charge", "nTrackerLayers": "tracker_layers"},
        "tau": {"decayMode": "decay_mode", "genPartFlav": "genpart_flavour"},
    }
    for target, source in fields[flavor].items():
        arrays[f"{prefix}_{target}"] = chunk[f"{flavor}_{source}_all"]
    if flavor == "electron":
        arrays["Electron_deltaEtaSC"] = chunk["electron_eta_sc_all"] - chunk["electron_eta_all"]
    if flavor in ("electron", "photon"):
        evaluate = lambda variation: calibration._calibrate_egm_collection(arrays, prefix, False, variation, payload)
    else:
        evaluate = lambda variation: getattr(calibration, f"_calibrate_{flavor}s")(arrays, False, variation, payload)
    nominal = evaluate("nominal")
    varied = evaluate(shift)
    for index, field in enumerate(("pt", "mass")):
        if nominal[index] is None:
            continue
        stored = chunk[f"{flavor}_corrected_{field}"]
        a, b = (np.asarray(ak.flatten(value), dtype=float) for value in (stored, nominal[index]))
        if not np.allclose(a, b, rtol=2e-6, atol=1e-6):
            raise RuntimeError(f"{flavor} stored nominal calibration does not match {year} payload")
        value = stored + (varied[index] - nominal[index])
        if not np.all(np.isfinite(ak.to_numpy(ak.flatten(value)))):
            raise RuntimeError(f"nonfinite {shift} {field}")
        output[f"{flavor}_corrected_{field}"] = value
        output[f"{flavor}_{field}_all"] = value
    delta = output[f"{flavor}_corrected_pt"] - chunk[f"{flavor}_corrected_pt"]
    phi = chunk[f"{flavor}_phi_all"]
    met, met_phi = (np.asarray(chunk[key], dtype=float) for key in ("met", "met_phi"))
    px = met * np.cos(met_phi) - ak.sum(delta * np.cos(phi), axis=1)
    py = met * np.sin(met_phi) - ak.sum(delta * np.sin(phi), axis=1)
    affected = ak.any(delta != 0, axis=1)
    output["met"] = ak.where(affected, np.hypot(px, py), chunk["met"])
    output["met_phi"] = ak.where(affected, np.arctan2(py, px), chunk["met_phi"])
    output["puppi_met_corrected"] = output["met"]
    output["puppi_met_corrected_phi"] = output["met_phi"]
    return output


def rebuild_kinematics(chunk):
    import awkward as ak
    import numpy as np
    from autonomous_allhad import real_subset_worker as physics
    from gnn_lowdm._implementation import region_io as regions
    from build_flat_boosted_recoil_hists import apply_highdm_veto_pt_thresholds, region_mask

    output = dict(chunk)
    masks = regions.object_masks(output)
    counts = {"electron_veto": "n_e_veto", "electron_medium": "n_e_medium",
              "muon_loose": "n_m_loose", "muon_medium": "n_m_medium",
              "photon_medium": "n_photon_medium"}
    for collection, mask in masks.items():
        flavor = collection.split("_")[0]
        for field in ("pt", "eta", "phi", "eta_sc"):
            source = f"{flavor}_{field}_all"
            if source in output:
                output[f"{collection}_{field}"] = output[source][mask]
        output[counts[collection]] = ak.sum(mask, axis=1)
    met = np.asarray(output["met"], dtype=float)
    met_phi = np.asarray(output["met_phi"], dtype=float)
    for flavor, mass_name, pt_name, region in (
        ("electron", "mee", "pee", "dy2e"), ("muon", "mmm", "pmm", "dy2m"),
    ):
        selected = masks[flavor + "_medium"]
        pairs = []
        for index in (0, 1):
            pair = [physics.nth_or(-99 if field == "pt" else 0,
                    output[f"{flavor}_{field}_all"][selected], index)
                    for field in ("pt", "eta", "phi", "mass")]
            pairs.append(pair)
        output[mass_name] = physics.invariant_mass(*pairs[0], *pairs[1])
        pt1, _, phi1, _ = pairs[0]
        pt2, _, phi2, _ = pairs[1]
        output[pt_name] = np.sqrt(np.maximum(0, pt1**2 + pt2**2 + 2 * pt1 * pt2 * np.cos(phi1-phi2)))
        output[f"recoil_{region}"], output[f"recoil_{region}_phi"] = physics.transverse_vector_sum(
            (met, met_phi), (pt1, phi1), (pt2, phi2))
    output["recoil_gcr"], output["recoil_gcr_phi"] = physics.transverse_vector_sum(
        (met, met_phi), (physics.first_or(0, output["photon_medium_pt"]),
                         physics.first_or(0, output["photon_medium_phi"])))
    tau_mt = physics.transverse_mass(output["tau_pt_all"], output["tau_phi_all"], met, met_phi)
    tau = ((output["tau_pt_all"] > 20) & (abs(output["tau_eta_all"]) < 2.5)
           & (abs(output["tau_dz_all"]) < .2) & (output["tau_decay_mode_all"] != 5)
           & (output["tau_decay_mode_all"] != 6) & (output["tau_deeptau_vsjet_all"] >= 5)
           & (tau_mt < 100))
    output["pass_zero_tau"] = ak.sum(tau, axis=1) == 0
    output["pass_met_250"] = met > 250
    jet_pt, jet_eta, jet_phi = (output[f"jet_{field}"] for field in ("corrected_pt", "eta_all", "phi_all"))
    good = (jet_pt > 30) & (abs(jet_eta) < 2.4) & ak.values_astype(output["jet_id_all"], np.bool_)
    medium = good & (output["jet_btag_upart_all"] > physics.UPART_AK4_MEDIUM_WP)
    jets = physics.jet_feature_block(jet_pt, jet_eta, jet_phi, good, medium, met_phi)
    for flag in ("open_pre", "open_high", "qcd_open", "dphi123_0p1"):
        output["pass_" + flag] = jets[flag]
    for index in range(1, 5):
        output[f"j{index}_met_dphi"] = jets[f"j{index}dphi"]
    output["min_dphi4"] = jets["min_dphi4"]
    clean_masks = {}
    for region, flavor in (("gcr", "photon"), ("dy2e", "electron"), ("dy2m", "muon")):
        clean = physics.clean_by_delta_r(jet_eta, jet_phi, output[flavor + "_medium_eta"],
                                         output[flavor + "_medium_phi"], .2)
        clean_masks[flavor] = clean
        block = physics.jet_feature_block(jet_pt, jet_eta, jet_phi, good & clean, medium & clean,
                                           output[f"recoil_{region}_phi"])
        output[f"pass_{region}_open_high"] = block["open_high"]
        if region == "gcr":
            for field, key in (("ht", "ht"), ("njet", "njet"), ("nb", "nb")):
                output[field + "_photon_clean"] = block[key]
            output["pass_ht_photon_300"] = block["ht"] > 300
            for index in range(1, 5):
                output[f"gcr_j{index}_recoil_dphi"] = block[f"j{index}dphi"]
            output["gcr_min_recoil_dphi4"] = block["min_dphi4"]
        else:
            output[f"pass_{region}_ut_250"] = output[f"recoil_{region}"] > 250
    clean = clean_masks["electron"] & clean_masks["muon"]
    leptons = physics.jet_feature_block(jet_pt, jet_eta, jet_phi, good & clean, medium & clean, met_phi)
    for field in ("ht", "njet", "nb"):
        output[field + "_lepton_clean"] = leptons[field]
    output["pass_ht_lepton_300"] = leptons["ht"] > 300
    block = regions.jet_kinematics(jet_pt, jet_eta, jet_phi, output["jet_btag_upart_all"], good, met, met_phi)
    for field in ("mtb", "ptb", "met_sqrt_ht"):
        output["lowdm_" + field] = block[field]
    output["lowdm_isr_dphi"] = np.where(output["lowdm_isr_pt"] > 0,
        physics.delta_phi(output["lowdm_isr_phi"], met_phi), -99.)
    apply_highdm_veto_pt_thresholds(output, 10., 10.)
    for region in ("DY2E", "DY2M"):
        output["feature_" + region] = region_mask(output, region, "feature_" + region, len(met))
    blocks, _ = regions.build_region_blocks(ak.zip(output, depth_limit=1))
    for region, block in blocks.items():
        clean = ak.ones_like(output["fatjet_corrected_pt"], dtype=np.bool_)
        if region in ("GCR", "DY2E", "DY2M"):
            flavor = {"GCR": "photon", "DY2E": "electron", "DY2M": "muon"}[region]
            clean = physics.clean_by_delta_r(output["fatjet_eta_all"], output["fatjet_phi_all"],
                output[flavor + "_medium_eta"], output[flavor + "_medium_phi"], .4)
        selected = (clean & ak.values_astype(output["fatjet_id_all"], np.bool_)
                    & (output["fatjet_corrected_pt"] > 200) & (abs(output["fatjet_eta_all"]) < 2.4))
        isr_pt = physics.first_or(-99, output["fatjet_corrected_pt"][selected])
        isr_phi = physics.first_or(-99, output["fatjet_phi_all"][selected])
        quality = ((block.nt == 0) & (block.nw == 0) & (block.nisr == 1)
                   & (physics.delta_phi(isr_phi, block.recoil_phi) > 2) & (block.met_sqrt_ht >= 10))
        core = block.core & quality
        bins = np.asarray([physics.assign_lowdm_search_bin(int(block.njet[i]), int(block.nb[i]),
            int(output["n_sv_softb"][i]), float(isr_pt[i]), float(block.ptb[i]),
            float(block.recoil[i]), float(block.mtb[i])) if core[i] else -1 for i in range(len(met))])
        output["lowdm_search_bin_" + region] = bins
        output["feature_lowdm_" + region] = core & (bins >= 0)
    output["lowdm_search_bin"] = output["lowdm_search_bin_SR"]
    output["feature_lowdm_sr_base"] = output["feature_lowdm_SR"]
    output["pass_lowdm_isr"] = (output["n_lowdm_isr"] == 1) & (output["lowdm_isr_dphi"] > 2)
    output["pass_lowdm_met_sqrt_ht"] = output["lowdm_met_sqrt_ht"] >= 10
    output["pass_lowdm_mtb"] = (output["nb_medium_lowdm"] == 0) | (output["lowdm_mtb"] < 175)
    output["feature_lowdm_preselection"] = (output["pass_base_common"] & output["pass_signal_trigger"]
        & output["pass_no_veto_leptons"] & output["pass_zero_tau"] & (output["njet"] >= 2)
        & output["pass_met_250"] & output["pass_open_pre"] & output["pass_ht_300"])
    if "feature_preselection" in output:
        output["feature_preselection"] = output["feature_lowdm_preselection"]
    return output


def write_intermediate_parts(config, shift, destination, payload):
    import awkward as ak
    import numpy as np
    import uproot
    from autonomous_allhad.analysis_scale_factors import topw_file_input_policy
    from autonomous_allhad.highdm_resolved_categories import map_candidates_to_events
    from gnn_lowdm._implementation.region_io import validate_trota_provenance

    source = Path(config["source_root"])
    metadata = read(config["source_sidecar"])
    validate_trota_provenance(metadata)
    if metadata["status"] not in ("complete", "complete_with_bad_files"):
        raise RuntimeError("incomplete intermediate input")
    destination.mkdir(parents=True, exist_ok=True)
    products = []
    with uproot.open(source) as root:
        policy = topw_file_input_policy(root)
        tree = root["Events"]
        if tree.num_entries != metadata["events_written"]:
            raise RuntimeError("intermediate Events/sidecar count mismatch")
        trota = root["TROTA"].arrays(library="ak")
        identity = tree.arrays(["file_id", "entry"], library="np")
        candidate_event = map_candidates_to_events(identity["file_id"], identity["entry"],
            np.asarray(trota["file_id"]), np.asarray(trota["entry"]))
        step = int(config.get("chunk_events", 25000))
        for start in range(0, max(1, tree.num_entries), step):
            stop = min(start + step, tree.num_entries)
            original = tree.arrays(entry_start=start, entry_stop=stop, library="ak", how=dict)
            changed = original
            if stop > start and shift != "nominal":
                calibrated = shift_objects(original, config["year"], shift, payload)
                flavor = next(name for name in ("electron", "photon", "muon", "tau") if shift.startswith(name))
                affected = ak.any(calibrated[f"{flavor}_pt_all"] != original[f"{flavor}_pt_all"], axis=1)
                changed = rebuild_kinematics(calibrated)
                selected_rows = np.arange(stop-start) + np.where(affected, 0, stop-start)
                changed = {name: ak.concatenate([ak.Array(value), original[name]], axis=0)[selected_rows] if name in original else value
                           for name, value in changed.items()}
            # Preserve the nominal storage precision and let uproot regenerate counters.
            counters = {branch.count_branch.name for branch in tree.values() if branch.count_branch is not None}
            values = {}
            for name, value in changed.items():
                if name in counters:
                    continue
                if name in original:
                    flat = ak.to_numpy(ak.flatten(original[name], axis=None))
                    value = ak.values_astype(ak.Array(value), flat.dtype)
                values[name] = value
            path = destination / f"part_{start:09d}.root"
            with uproot.recreate(path) as target:
                write_tree(target, "Events", values)
                selected = (candidate_event >= start) & (candidate_event < stop)
                write_tree(target, "TROTA", {name: trota[name][selected] for name in ak.fields(trota)})
                target["TROTA_metadata"] = str(root["TROTA_metadata"])
                if "TopWTruth" in root:
                    truth = root["TopWTruth"].arrays(entry_start=start, entry_stop=stop, library="ak", how=dict)
                    truth_counters = {branch.count_branch.name for branch in root["TopWTruth"].values() if branch.count_branch is not None}
                    write_tree(target, "TopWTruth", {name: value for name, value in truth.items() if name not in truth_counters})
                    if "TopWTruth_metadata" in root:
                        marker = json.loads(str(root["TopWTruth_metadata"]))
                        marker.update(events_entries=stop-start, parent_events_entries=tree.num_entries,
                                      reuse="unchanged jets; retained-event slice")
                        target["TopWTruth_metadata"] = json.dumps(marker)
            sidecar = dict(metadata, events_written=stop-start, root_sha256=sha(path),
                shape_shift=shift, intermediate_parent=config.get("canonical_source_root", str(source)),
                intermediate_entry_range=[start, stop],
                systematic_scope="retained events; nominal base_common fixed; no skim migration study")
            write(path.with_suffix(".json"), sidecar)
            with uproot.open(path) as check:
                if check["Events"].num_entries != stop-start or topw_file_input_policy(check) != policy:
                    raise RuntimeError("intermediate slice integrity mismatch")
            products.append(path)
    return products


def histogram_parts(config, roots, work):
    main_repo, gnn_repo = (Path(config[key]) for key in ("main_repo", "gnn_repo"))
    env = dict(os.environ, AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0")
    vendor = os.environ.get("SHAPE_VENDOR", "")
    env["PYTHONPATH"] = f"{main_repo}/autonomous_allhad:{main_repo}/autonomous_allhad/workflow:{main_repo}:{vendor}"
    execute([sys.executable, "-u", main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
        "--repo", main_repo, "--inputs", *roots, "--normalization", config["normalization"],
        "--output", work / "main.json", "--campaign-year", str(config["year"]),
        "--nominal-only", "--step-size", "5000",
        "--require-btag", "--expected-btag-efficiency-sha256", config["btag_efficiency_sha256"],
        "--require-weight-components", *WEIGHTS, "--analysis-sf-components", "met_trigger", "photon_trigger", "topw_tagging",
        "--require-branches", "--require-normalization", "--allow-zero-entry-roots",
        "--electron-veto-pt-min", "10", "--muon-veto-pt-min", "10",
        "--search-bin-config", config["search_config"], "--dy-ptll-policy", "all"], env, work)
    if read(work / "main.json")["status"] != "complete":
        raise RuntimeError("High-dM histogram production failed")
    env["PYTHONPATH"] = f"{gnn_repo}/autonomous_allhad:{gnn_repo}/autonomous_allhad/workflow:{gnn_repo}:{vendor}"
    request = dict(kind="mc", batch=0, manifest=config["gnn_manifest"], repository=str(gnn_repo),
        stop_xsec=config["stop_xsec"], inputs=[dict(root=str(path), sidecar=str(path.with_suffix(".json"))) for path in roots])
    write(work / "gnn_request.json", request)
    execute([sys.executable, "-u", "-m", "gnn_lowdm.eval", "cr-partial",
        "--request", work / "gnn_request.json", "--output", work / "gnn.json",
        "--model", config["gnn_model"], "--selection", config["gnn_selection"],
        "--configuration", config["gnn_configuration"], "--raw-dy",
        "--regions", "SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M"], env, work)
    result = read(work / "gnn.json")
    if result["status"] != "complete" or result["bad_files"] or result["input_files_valid"] != len(roots):
        raise RuntimeError("GNN histogram production failed: " + str(result["bad_files"]))


def compact_product(source, destination):
    temporary = source.with_name(source.name + ".gz")
    with source.open("rb") as src, temporary.open("wb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0) as archive:
            shutil.copyfileobj(src, archive)
    digest = hashlib.sha256()
    with gzip.open(temporary, "rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha(source):
        raise RuntimeError("compressed product checksum mismatch")
    checksum = 1
    with temporary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.adler32(block, checksum)
    product = dict(path=str(destination), sha256=sha(temporary),
                   adler32=f"{checksum & 0xffffffff:08x}",
                   uncompressed_sha256=digest.hexdigest(), bytes=temporary.stat().st_size)
    publish_product(temporary, destination)
    return product


def root_content_digests(path):
    """Verify the existing TopWTruth marker's flat_ntuple_worker digest format."""
    import awkward as ak
    import numpy as np
    import uproot

    result = {}
    with uproot.open(path) as root:
        for name in root.keys(cycle=False):
            obj = root[name]
            digest = hashlib.sha256()
            if hasattr(obj, "iterate"):
                digest.update(json.dumps(obj.typenames(), sort_keys=True).encode())
                for arrays in obj.iterate(step_size=1000, library="ak"):
                    form, length, buffers = ak.to_buffers(arrays)
                    digest.update(str(length).encode())
                    digest.update(str(form).encode())
                    for key, values in sorted(buffers.items()):
                        digest.update(key.encode())
                        digest.update(np.asarray(values).tobytes())
                result[name] = dict(entries=int(obj.num_entries), sha256=digest.hexdigest())
            elif obj.classname == "TObjString":
                result[name] = dict(sha256=hashlib.sha256(str(obj).encode()).hexdigest())
            else:
                raise RuntimeError("unsupported ROOT object: " + name)
    return result


def object_worker(args):
    work = Path(args.work) if args.work else Path(os.environ["_CONDOR_SCRATCH_DIR"]) / "objects"
    work.mkdir(parents=True, exist_ok=True)
    local_config = stage_input(args.config, work / "config.json")
    config_hash = sha(local_config)
    config = load_config(local_config)
    if config.get("input_mode") != "intermediate_root_only":
        raise ValueError("object worker requires the intermediate-only input contract")
    destination = eos(args.output)
    bundle = Path(os.environ["SHAPE_CODE_ROOT"])
    if sha(os.environ["SHAPE_BUNDLE"]) != config["source_bundle_sha256"]:
        raise RuntimeError("source bundle changed")
    for name, expected in read(bundle / "hashes.json").items():
        if sha(bundle / name) != expected:
            raise RuntimeError("staged dependency checksum mismatch: " + name)
    original = dict(config)
    config.update(read(bundle / f"runtime_{config['year']}.json"))
    for key in ("main_repo", "gnn_repo", "normalization", "gnn_manifest", "gnn_model", "gnn_selection",
                "gnn_configuration", "search_config", "stop_xsec", "object_payload_bundle"):
        config[key] = str(bundle / config[key])
    sys.path[:0] = [str(Path(config["main_repo"]) / "autonomous_allhad"),
                    str(Path(config["main_repo"]) / "autonomous_allhad/workflow")]
    sidecar = stage_input(config["source_sidecar"], work / "source.json")
    if sha(sidecar) != config["source_sidecar_sha256"]:
        raise RuntimeError("intermediate sidecar changed")
    config["source_sidecar"] = str(sidecar)
    source = eos(config["source_root"])
    cached = work / "source.root"
    if not cached.exists():
        stage_input(source, cached)
    source_hash = sha(cached)
    source_validation = "exact_root_sha256"
    if source_hash != config["source_root_sha256"]:
        import uproot

        with uproot.open(cached) as root:
            marker = json.loads(str(root["TopWTruth_metadata"]))
        if (marker.get("input_sha256") != config["source_root_sha256"]
            or marker.get("status") != "complete" or marker.get("application_year") != config["year"]):
            raise RuntimeError("intermediate ROOT checksum mismatch without matching TopWTruth provenance")
        contents = root_content_digests(cached)
        if ({key: value for key, value in contents.items() if key not in ("TopWTruth", "TopWTruth_metadata")}
            != marker["original_contents"] or contents["TopWTruth"] != marker["truth_content"]):
            raise RuntimeError("intermediate content changed after TopWTruth augmentation")
        source_validation = "validated_topw_augmentation_of_sidecar_root"
    config["source_root"] = str(cached)
    config["canonical_source_root"] = str(source)
    payload = prepare_payload(config, work / "payload")
    for shift in args.shifts:
        target = destination / shift
        state_path = target / "result.json"
        if xrootd(["xrdfs", "root://eosuser.cern.ch", "stat", state_path], allow_missing=True) is not None:
            existing = read(stage_input(state_path, work / "existing.json"))
            if existing.get("status") == "complete":
                if (existing.get("config_sha256") != config_hash
                    or set(existing.get("products", {})) != {"main.json", "gnn.json"}):
                    raise RuntimeError("existing result does not match pinned inputs")
                for product in existing["products"].values():
                    checksum = xrootd(["xrdfs", "root://eosuser.cern.ch", "query", "checksum", eos(product["path"])])
                    if checksum.split() != ["adler32", product["adler32"]]:
                        raise RuntimeError("existing product checksum mismatch: " + product["path"])
                continue
        report = dict(status="running", year=config["year"], shift=shift, started=time.time(),
            input_mode=config["input_mode"], source_root=str(source), source_root_sha256=source_hash,
            sidecar_root_sha256=config["source_root_sha256"], source_validation=source_validation,
            config_sha256=config_hash, source_bundle_sha256=original["source_bundle_sha256"],
            migration_study="deferred_by_user", nominal_base_common="fixed",
            trota="reused; jet inputs unchanged", topw_truth="reused; no NanoAOD access",
            normalization="unchanged frozen nominal normalization")
        print(json.dumps(report), flush=True)
        stage = work / shift
        try:
            parts = write_intermediate_parts(config, shift, stage / "parts", payload)
            histogram_parts(config, parts, stage)
            products = {name: compact_product(stage / name, target / (name + ".gz"))
                        for name in ("main.json", "gnn.json")}
            report.update(status="complete", products=products, parts=len(parts),
                events=sum(read(path.with_suffix(".json"))["events_written"] for path in parts),
                finished=time.time(), canonical_promotion=False)
            checkpoint(state_path, report, work)
            shutil.rmtree(stage)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}", finished=time.time())
            try:
                checkpoint(state_path, report, work)
            except Exception as transfer_error:
                print(json.dumps(dict(failed_report=report, checkpoint_error=str(transfer_error))), flush=True)
            raise


def prepare_objects(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    target = eos(campaign / args.label)
    target.mkdir(parents=True, exist_ok=True)
    bundle = target / "bundles/object_code.tgz"
    bundle.parent.mkdir(exist_ok=True)
    if bundle.exists() and not args.refresh_unsubmitted:
        raise FileExistsError("frozen object campaign already exists: " + str(bundle))
    if bundle.exists():
        state_path = target / "campaign_state.json"
        if state_path.exists() and read(state_path)["status"] not in ("prepared_not_submitted", "outputs_discarded"):
            raise RuntimeError("cannot refresh a submitted campaign")
    baseline = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    files, runtime = {}, {}
    for label, folder in (("main", baseline / "code_main_v2"), ("gnn", baseline / "gnn_code")):
        for subtree in ("autonomous_allhad/autonomous_allhad", "autonomous_allhad/workflow",
                        "autonomous_allhad/gnn_lowdm", "autonomous_allhad/configs",
                        "autonomous_allhad/signals", "analysis/utils", "analysis/data", "analysis/hists"):
            for parent, directories, names in os.walk(folder / subtree, followlinks=True):
                directories[:] = [name for name in directories if name not in ("__pycache__", "results", "tests", ".git")]
                for name in names:
                    source = Path(parent) / name
                    if source.suffix not in (".py", ".json", ".gz", ".coffea", ".merged", ".npz"):
                        continue
                    files[str(Path(label) / source.relative_to(folder))] = source
    files["run.py"] = HERE / "run.py"
    for year in args.years:
        common = read(HERE / "full_met_20260911" / str(year) / "common.json")
        for key in ("main_repo", "gnn_repo", "normalization", "gnn_manifest", "gnn_model", "gnn_selection",
                    "gnn_configuration", "search_config", "stop_xsec"):
            path = Path(common[key])
            if key in ("main_repo", "gnn_repo"):
                common[key] = "main" if key == "main_repo" else "gnn"
            elif str(path).startswith(str(baseline / "code_main_v2") + "/"):
                common[key] = str(Path("main") / path.relative_to(baseline / "code_main_v2"))
            elif str(path).startswith(str(baseline / "gnn_code") + "/"):
                common[key] = str(Path("gnn") / path.relative_to(baseline / "gnn_code"))
            else:
                common[key] = f"inputs/{year}/{path.name}"
                files[common[key]] = path
        object_bundle = repo / f"autonomous_allhad/workflow/flat{year}_v8/bundles" / (
            "objectcorr_2024_payloads.tgz" if year == 2024 else "objectcorr_2025_data_payloads.tgz")
        common["object_payload_bundle"] = f"payload_{year}.tgz"
        files[common["object_payload_bundle"]] = object_bundle
        for path, expected in common["pins"].items():
            if Path(path) != HERE / "run.py" and Path(path) in files.values() and sha(path) != expected:
                raise RuntimeError("baseline dependency changed: " + path)
        keep = ("main_repo", "gnn_repo", "normalization", "gnn_manifest", "gnn_model", "gnn_selection",
                "gnn_configuration", "search_config", "stop_xsec", "object_payload_bundle", "btag_efficiency_sha256")
        runtime[year] = {key: common[key] for key in keep}
        runtime_file = target / "bundles" / f"runtime_{year}.json"
        write(runtime_file, runtime[year])
        files[runtime_file.name] = runtime_file
    hashes = {name: sha(path) for name, path in sorted(files.items())}
    write(target / "bundles/hashes.json", hashes)
    write(target / "bundles/origins.json", {name: dict(path=str(path), sha256=hashes[name]) for name, path in files.items()})
    with tarfile.open(bundle, "w:gz", dereference=True) as archive:
        for name, source in sorted(files.items()):
            archive.add(source, arcname=name, recursive=False)
        archive.add(target / "bundles/hashes.json", arcname="hashes.json", recursive=False)
    bundle_hash = sha(bundle)
    summary = dict(status="prepared_not_submitted", input_mode="intermediate_root_only",
        shifts=list(OBJECT_SHIFTS), job_flavour="workday", migration_study="deferred_by_user",
        source_bundle_sha256=bundle_hash, nominal_modified=False, canonical_promotion=False, years={})
    for year in args.years:
        base = target / str(year)
        sources_file = baseline / str(year) / "inputs.txt"
        sources = [eos(line.strip()) for line in sources_file.read_text().splitlines()
                   if line.strip() and not Path(line.strip()).name.startswith("data_")]
        if len(sources) != len(set(sources)):
            raise RuntimeError("duplicate intermediate ROOT paths")
        jobs, inputs = [], []
        previous = read(base / "campaign_state.json") if (base / "campaign_state.json").exists() else {}
        reusable = {item["root"]: item for item in previous.get("inputs", [])}
        if reusable and previous["input_list_sha256"] != sha(sources_file):
            raise RuntimeError("frozen nominal input list changed")

        def prepare_source(source):
            config_path = base / "configs" / (source.stem + ".json")
            output = base / "outputs" / source.stem
            if args.refresh_unsubmitted and str(source) in reusable and config_path.exists():
                config = read(config_path)
                config["source_bundle_sha256"] = bundle_hash
                write(config_path, config)
                return f"{source.stem} {config_path} {output}", reusable[str(source)]
            sidecar = source.with_suffix(".json")
            metadata = read(sidecar)
            if metadata["status"] not in ("complete", "complete_with_bad_files") or not source.is_file():
                raise RuntimeError("invalid intermediate ROOT: " + str(source))
            config = dict(input_mode="intermediate_root_only", year=year, source_root=str(source),
                source_sidecar=str(sidecar), source_sidecar_sha256=sha(sidecar),
                source_root_sha256=metadata["root_sha256"], source_bundle_sha256=bundle_hash,
                chunk_events=25000)
            write(config_path, config)
            output.mkdir(parents=True, exist_ok=True)
            return (f"{source.stem} {config_path} {output}",
                    dict(root=str(source), events=metadata["events_written"],
                         root_sha256=config["source_root_sha256"]))

        with ThreadPoolExecutor(max_workers=8) as pool:
            for row, item in pool.map(prepare_source, sources):
                jobs.append(row)
                inputs.append(item)
        logs = base / "logs"
        logs.mkdir(exist_ok=True)
        queue = base / "jobs.queue"
        queue.write_text("\n".join(jobs) + "\n")
        runtime_dir = repo.parent / "runtime"
        proxy = eos(args.proxy)
        for path in (runtime_dir / "py38.tgz", runtime_dir / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
            if not path.is_file():
                raise FileNotFoundError(path)
        submit = base / "jobs.sub"
        submit.write_text(f'''universe = vanilla
initialdir = {base}
executable = {campaign}/run.sh
arguments = {campaign}/run.py object-worker --config $(config) --output $(resultdir)
getenv = False
output = {logs}/$(shard).$(ClusterId).out
error = {logs}/$(shard).$(ClusterId).err
log = {logs}/jobs.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {runtime_dir}/py38.tgz, {runtime_dir}/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl, {bundle}
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 2
request_memory = 10000MB
request_disk = 25000MB
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "NPS26012_objects_intermediate_{year}"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue shard,config,resultdir from {queue}
''')
        state = dict(status="prepared_not_submitted", jobs=len(jobs), input_list_sha256=sha(sources_file),
            inputs=inputs, submit=str(submit), submit_sha256=sha(submit), queue_sha256=sha(queue))
        write(base / "campaign_state.json", state)
        summary["years"][str(year)] = {key: value for key, value in state.items() if key != "inputs"}
    write(target / "campaign_state.json", summary)
    print(json.dumps(summary), flush=True)


def prepare_full(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    state = read(campaign / "campaign_state.json")
    target = eos(campaign / args.label)
    if (target / "campaign_state.json").exists():
        raise FileExistsError("campaign already prepared: " + str(target))
    summary = dict(status="prepared", scope="all canonical background and signal MC",
                   shifts=list(SHIFTS[1:]), job_flavour="workday", years={},
                   canonical_modified=False, canonical_promotion=False, submitted_clusters=[])
    for year in args.years:
        if state["pilot_validation"][str(year)]["validation_cli"]["status"] != "passed":
            raise RuntimeError("nominal pilot closure is required")
        pilot_config = campaign / str(year) / ("campaign.json" if year == 2024 else "pilot_retry1.json")
        pilot = read(pilot_config)
        shared = dict(pilot)
        for path, expected in shared["pins"].items():
            if Path(path).name in ("run.py", "run.sh"):
                continue
            if sha(path) != expected:
                raise RuntimeError("validated dependency changed: " + path)
        for key in ("source_root", "source_sidecar", "shard"):
            shared.pop(key)
        shared["pins"] = {p: h for p, h in shared["pins"].items()
                          if p not in (pilot["source_sidecar"], pilot["shard"])
                          and Path(p).name not in ("run.py", "run.sh")}
        for name in ("run.py", "run.sh"):
            shared["pins"][str(campaign / name)] = sha(campaign / name)
        shared.update(status="full_mc_prepared", shifts=list(SHIFTS[1:]),
                      keep_input_cache=False, compact_output=True,
                      root_retention="batch scratch only; compressed histograms and metadata on EOS")
        base = target / str(year)
        shared_path = base / "common.json"
        write(shared_path, shared)
        shared_hash = sha(shared_path)
        source_list = repo / f"autonomous_allhad/workflow/histograms/lepton_veto10_20260908/{year}/inputs.txt"
        sources = [eos(line) for line in source_list.read_text().splitlines()
                   if line and not Path(line).name.startswith("data_")]
        seen, names, jobs, reused, input_failures = set(), set(), [], [], []
        counts = Counter()
        process_files = Counter()

        def inspect(source):
            metadata_path = source.with_suffix(".json")
            metadata = read(metadata_path)
            records = source_records(metadata_path, year, metadata)
            if not records or not source.is_file():
                raise RuntimeError("missing or empty canonical source: " + str(source))
            compact = {key: metadata.get(key) for key in ("events_read", "bad_files")}
            return source, metadata_path, compact, records

        def prepare_source(source, metadata_path, records):
            name = source.stem
            shard_path = base / "shards" / (name + ".json")
            write(shard_path, dict(schema_version="full_production_shard_spec_v2_boosted",
                shard_id=name, record_group="mc", records=records, records_per_shard=len(records),
                record_digest=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]))
            config_path = base / "configs" / (name + ".json")
            write(config_path, dict(shared_config=str(shared_path), shared_config_sha256=shared_hash,
                source_root=str(source), source_sidecar=str(metadata_path), shard=str(shard_path),
                pins={str(metadata_path): sha(metadata_path), str(shard_path): sha(shard_path)}))
            memory = 12000 if any(r["is_signal"] for r in records) else 6000
            rows = []
            for shift in SHIFTS[1:]:
                output = base / "outputs" / name / shift
                output.mkdir(parents=True, exist_ok=True)
                rows.append(f"{name} {shift} {config_path} {output} {memory}")
            return rows

        prepared = []
        with ThreadPoolExecutor(max_workers=4) as pool, ThreadPoolExecutor(max_workers=8) as writers:
            for source, metadata_path, metadata, records in pool.map(inspect, sources):
                name = source.stem
                if name in names:
                    raise RuntimeError("duplicate output shard name: " + name)
                names.add(name)
                for record in records:
                    key = record["file_path"].split("/store/", 1)[-1]
                    if key in seen:
                        raise RuntimeError("duplicate canonical NanoAOD: " + key)
                    seen.add(key)
                    process_files[record["process_group"]] += 1
                counts["signal_shards" if any(r["is_signal"] for r in records) else "background_shards"] += 1
                counts["files"] += len(records)
                counts["events_read"] += metadata["events_read"]
                input_failures.extend(metadata.get("bad_files", []))
                if str(source) == pilot["source_root"]:
                    old_base = campaign / str(year) / ("pilot" if year == 2024 else "pilot_retry1")
                    for shift in SHIFTS[1:]:
                        result = read(old_base / shift / "result.json")
                        if result["status"] != "complete" or any(
                                sha(p["path"]) != p["sha256"] for p in result["products"].values()):
                            raise RuntimeError("validated pilot product changed")
                        reused.append(dict(shard=name, shift=shift, output=str(old_base / shift)))
                    continue
                prepared.append(writers.submit(prepare_source, source, metadata_path, records))
                if len(names) % 500 == 0:
                    print(json.dumps(dict(year=year, inspected_shards=len(names))), flush=True)
            for future in prepared:
                jobs.extend(future.result())
        logs = base / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        queue = base / "jobs.queue"
        queue.write_text("\n".join(jobs) + "\n")
        runtime = repo.parent / "runtime"
        proxy = eos(args.proxy)
        for path in (runtime / "py38.tgz", runtime / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
            if not path.is_file():
                raise FileNotFoundError(path)
        submit_path = base / "jobs.sub"
        submit_path.write_text(f'''universe = vanilla
initialdir = {base}
executable = {campaign}/run.sh
arguments = {campaign}/run.py worker --config $(config) --shift $(shift) --output $(resultdir)
getenv = False
output = {logs}/$(shard)_$(shift).$(ClusterId).out
error = {logs}/$(shard)_$(shift).$(ClusterId).err
log = {logs}/jobs.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {runtime}/py38.tgz, {runtime}/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 2
request_memory = $(memory)
request_disk = 25000MB
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "NPS26012_shape_{args.label}_{year}"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue shard,shift,config,resultdir,memory from {queue}
''')
        year_state = dict(status="prepared", jobs=len(jobs), reused=reused, counts=dict(counts),
                          process_files=dict(process_files), unique_files=len(seen), duplicate_files=0,
                          source_bad_files=input_failures, input_list_sha256=sha(source_list),
                          common_sha256=shared_hash, queue_sha256=sha(queue), submit=str(submit_path))
        write(base / "campaign_state.json", year_state)
        summary["years"][str(year)] = year_state
        print(json.dumps(dict(year=year, jobs=len(jobs), files=len(seen), reused=len(reused))), flush=True)
    write(target / "campaign_state.json", summary)
    state.setdefault("full_campaigns", {})[args.label] = dict(path=str(target), status="prepared",
        jobs=sum(y["jobs"] for y in summary["years"].values()), shifts=list(SHIFTS[1:]))
    write(campaign / "campaign_state.json", state)


def prepare(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    canonical_path = repo / "autonomous_allhad/reports/lepton_veto10_canonical_20260908.json"
    canonical = read(canonical_path)
    nominal = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    main_repo = repo / canonical["production"]["frozen_main_code"]
    gnn_repo = repo / canonical["production"]["frozen_gnn_code"]
    all_jobs = []
    for year in args.years:
        base = campaign / str(year)
        source = repo / f"autonomous_allhad/workflow/flat{year}_v8/outputs/nominal/mc_shard_00000.root"
        source_list = nominal / str(year) / "inputs.txt"
        if str(source) not in source_list.read_text().splitlines():
            raise ValueError("pilot ROOT is not in canonical input list")
        records = source_records(source.with_suffix(".json"), year)
        shard = dict(schema_version="full_production_shard_spec_v2_boosted",
                     shard_id="mc_shard_00000", record_group="mc", records=records,
                     records_per_shard=len(records),
                     record_digest=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16])
        shard_path = base / args.label / "shard.json"
        write(shard_path, shard)
        model = repo / f"autonomous_allhad/workflow/flat{year}_v8/bundles/model_TopResolved_2024_TROTA2D_ptcut.h5"
        payload_name = "objectcorr_2024_payloads.tgz" if year == 2024 else "objectcorr_2025_data_payloads.tgz"
        object_payload = model.parent / payload_name
        norm = nominal / str(year) / "norm.json"
        search = main_repo / f"autonomous_allhad/configs/search_bins_{year}.json"
        if sha(norm) != canonical["years"][str(year)]["normalization_sha256"]:
            raise ValueError("canonical normalization changed")
        if sha(search) != canonical["years"][str(year)]["search_bin_config_sha256"]:
            raise ValueError("canonical search configuration changed")
        gnn_model = gnn_repo / "autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010"
        config = dict(year=year, status="pilot_prepared", repo=str(repo),
                      main_repo=str(main_repo), gnn_repo=str(gnn_repo),
                      source_root=str(source), source_sidecar=str(source.with_suffix(".json")),
                      shard=str(shard_path), normalization=str(norm), search_config=str(search),
                      trota_model=str(model), trota_setup=TROTA_SETUP,
                      object_payload_bundle=str(object_payload),
                      gnn_manifest=str(nominal / str(year) / "gnn/full_manifest.json"),
                      gnn_model=str(gnn_model / "diagonal_v3_numpy.npz"),
                      gnn_selection=str(gnn_model / "selection.json"),
                      gnn_configuration=str(gnn_repo / "autonomous_allhad/gnn_lowdm/config.json"),
                      stop_xsec=str(gnn_repo / "autonomous_allhad/signals/stop_xsec_13p6TeV.json"),
                      btag_efficiency_sha256="9326454608126467219a60f8d5763a6c92465127ca6ea32ac415ace8ed0ec004",
                      shifts=list(SHIFTS), root_retention="pilot only, on EOS",
                      canonical_report_sha256=sha(canonical_path), pins={})
        paths = [HERE / "run.py", HERE / "run.sh", canonical_path, source_list,
                 source.with_suffix(".json"), shard_path, norm, search, model, object_payload,
                 Path(config["gnn_manifest"]), Path(config["gnn_model"]),
                 Path(config["gnn_selection"]), Path(config["gnn_configuration"]),
                 main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"]
        for directory in (repo / "autonomous_allhad/autonomous_allhad",
                          gnn_repo / "autonomous_allhad/gnn_lowdm/_implementation"):
            paths.extend(sorted(directory.glob("*.py")))
        paths.extend([repo / "analysis/data/ids.coffea", repo / "analysis/data/corrections.coffea"])
        for path in paths:
            config["pins"][str(path)] = sha(path)
        config_path = base / ("campaign.json" if args.label == "pilot" else args.label + ".json")
        write(config_path, config)
        for shift in SHIFTS:
            out = base / args.label / shift
            out.mkdir(parents=True, exist_ok=True)
            all_jobs.append(f"{year} {shift} {config_path} {out}")
    logs = campaign / "logs"
    logs.mkdir(exist_ok=True)
    (campaign / (args.label + ".queue")).write_text("\n".join(all_jobs) + "\n")
    runtime = repo.parent / "runtime"
    proxy = Path("/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757")
    for path in (runtime / "py38.tgz", runtime / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
        if not path.is_file():
            raise FileNotFoundError(path)
    submit = f'''universe = vanilla
initialdir = {campaign}
executable = {campaign}/run.sh
arguments = {campaign}/run.py worker --config $(config) --shift $(shift) --output $(resultdir)
getenv = False
output = {logs}/pilot_$(year)_$(shift).$(ClusterId).out
error = {logs}/pilot_$(year)_$(shift).$(ClusterId).err
log = {logs}/pilot.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {runtime}/py38.tgz, {runtime}/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 2
request_memory = 6000MB
request_disk = 18000MB
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "NPS26012_separate_shape_pilot"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue year,shift,config,resultdir from {campaign}/{args.label}.queue
'''
    (campaign / (args.label + ".sub")).write_text(submit)
    if (campaign / "campaign_state.json").exists():
        state = read(campaign / "campaign_state.json")
        state.setdefault("preparations", {})[args.label] = dict(jobs=len(all_jobs), years=args.years)
    else:
        state = dict(status="pilot_prepared", jobs=len(all_jobs),
          canonical_modified=False, runtime="py38; TROTA-only LCG_104", job_flavour="workday",
          submitted_clusters=[], pending=["pilot execution", "nominal closure", "full MC propagation",
          "JES/JER TROTA-input propagation", "EGM", "MUO/TAU decomposition", "JMS/JMR prescription"])
    write(campaign / "campaign_state.json", state)
    print(json.dumps({"status": "pilot_prepared", "jobs": len(all_jobs), "submit": str(campaign / (args.label + ".sub"))}))


def preflight(args):
    import importlib
    config = read(args.config)
    repo = eos(config["repo"])
    work = eos(args.output)
    payload = prepare_payload(config, work / "payload")
    os.environ.update(AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0",
                      AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="0",
                      AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"))
    sys.path.insert(0, str(repo / "autonomous_allhad"))
    year = config["year"]
    correction = importlib.import_module(f"autonomous_allhad.object_corrections_{year}")
    validation = correction.validate_payloads(payload)
    if validation["status"] != "valid":
        raise RuntimeError(str(validation))
    module = ("autonomous_allhad.intermediate_2024_worker" if year == 2024
              else "autonomous_allhad.intermediate_2025_data_worker")
    importlib.import_module(module).install_backend()
    from autonomous_allhad import flat_ntuple_worker as flat
    record = read(config["shard"])["records"][0]
    rows, summary, bad = flat.process_record(record, payload, 500, "nominal",
                                             "feature_flat_preselection", False, 1, True)
    write(work / "result.json", dict(status="failed" if bad else "complete", summary=summary, bad_files=bad))
    print(json.dumps(dict(rows=len(rows), events_read=summary["events_read"], error=summary.get("error"))))
    if bad:
        raise RuntimeError(str(bad))


def worker(args):
    config = load_config(args.config)
    if args.shift not in SHIFTS:
        raise ValueError("shift not validated by this propagation stage")
    for path, expected in config["pins"].items():
        if sha(path) != expected:
            raise RuntimeError("pinned input changed: " + path)
    destination = eos(args.output)
    if (destination / "result.json").exists():
        existing = read(destination / "result.json")
        if existing.get("status") == "complete":
            if (existing.get("config_sha256") != sha(args.config) or any(
                    sha(item["path"]) != item["sha256"] for item in existing["products"].values())):
                raise RuntimeError("existing completed output changed")
            print(json.dumps({"status": "already_complete", "output": str(destination)}))
            return
    work = Path(os.environ["_CONDOR_SCRATCH_DIR"]) / "shape"
    work.mkdir()
    repo = Path(config["repo"])
    year = config["year"]
    env = dict(os.environ)
    vendor = os.environ.get("PYTHONPATH", "")
    env.update(PYTHONPATH=f"{repo}/autonomous_allhad:{repo}:{vendor}",
               AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0",
               AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="1",
               AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE="1" if config.get("keep_input_cache", True) else "0",
               AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"),
               AUTONOMOUS_ALLHAD_FRAGMENT_DIR=str(work / "fragments"))
    root = work / Path(config["source_root"]).name
    metadata_path = root.with_suffix(".json")
    report = dict(status="running", year=year, shift=args.shift, stages={}, started=time.time())

    def checkpoint():
        write(destination / "result.json", report)

    try:
        checkpoint()
        payload_repo = prepare_payload(config, work / "payload") if config.get("object_payload_bundle") else repo
        module = ("autonomous_allhad.intermediate_2024_worker" if year == 2024
                  else "autonomous_allhad.intermediate_2025_data_worker")
        execute([sys.executable, "-u", "-m", module, "--repo", payload_repo,
                 "--shard", config["shard"], "--output", root, "--metadata-output", metadata_path,
                 "--shift", args.shift, "--record-workers", "1", "--chunk-size", "25000"], env, work)
        metadata = read(metadata_path)
        if metadata["status"] not in ("complete", "complete_with_bad_files"):
            raise RuntimeError("incomplete shifted Events production")
        report["bad_files"] = metadata["bad_files"]
        report["stages"]["events"] = dict(events_read=metadata["events_read"],
            events_written=metadata["events_written"], files=metadata["files_processed"])
        checkpoint()
        trota_env = dict(env)
        for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
            trota_env.pop(key, None)
        script = ('set -e; set +u; source "$1"; shift; '
                  'export PYTHONPATH="$1/autonomous_allhad:$1${PYTHONPATH:+:$PYTHONPATH}"; shift; '
                  'exec python3 -u -m autonomous_allhad.trota_resolved_2024_inplace "$@"')
        execute(["/bin/bash", "-c", script, "trota", config["trota_setup"], repo,
                 "--input", root, "--model", config["trota_model"],
                 "--metadata-output", work / "trota.json", "--target-year", str(year),
                 "--chunk-events", "20000", "--batch-size", "8192", "--allow-hadd-repair"], trota_env, work)
        trota = read(work / "trota.json")
        if trota["status"] != "complete" or trota["marker"]["status"] != "complete":
            raise RuntimeError("TROTA inference did not complete")
        metadata.update(root_sha256=sha(root), root_trees=["Events", "TROTA"])
        metadata[f"trota_topresolved_{year}"] = trota
        write(metadata_path, metadata)
        report["stages"]["trota"] = trota["counts"]
        checkpoint()
        execute([sys.executable, "-u", "-m", "autonomous_allhad.flat_ntuple_worker",
                 "--repo", repo, "--append-topw-truth", root, "--output", root,
                 "--metadata-output", work / "truth.json", "--truth-year", str(year),
                 "--truth-work-dir", work], env, work)
        report["stages"]["topw"] = read(work / "truth.json")["status"]
        checkpoint()
        main_repo = Path(config["main_repo"])
        hist_env = dict(env, PYTHONPATH=f"{main_repo}/autonomous_allhad:{main_repo}/autonomous_allhad/workflow:{main_repo}:{vendor}")
        execute([sys.executable, "-u", main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
                 "--repo", main_repo, "--inputs", root, "--normalization", config["normalization"],
                 "--output", work / "main.json", "--campaign-year", str(year),
                 "--nominal-only", "--require-btag", "--expected-btag-efficiency-sha256", config["btag_efficiency_sha256"],
                 "--require-weight-components", *WEIGHTS, "--analysis-sf-components", "met_trigger", "photon_trigger", "topw_tagging",
                 "--require-branches", "--require-normalization", "--allow-zero-entry-roots",
                 "--electron-veto-pt-min", "10", "--muon-veto-pt-min", "10",
                 "--search-bin-config", config["search_config"], "--dy-ptll-policy", "all"], hist_env, work)
        main = read(work / "main.json")
        if main["status"] != "complete":
            raise RuntimeError("main histogram validation failed")
        report["stages"]["main"] = main["status"]
        del main
        checkpoint()
        gnn_repo = Path(config["gnn_repo"])
        gnn_env = dict(env, PYTHONPATH=f"{gnn_repo}/autonomous_allhad:{gnn_repo}/autonomous_allhad/workflow:{gnn_repo}:{vendor}")
        request = dict(kind="mc", batch=0, manifest=config["gnn_manifest"], repository=str(gnn_repo),
                       stop_xsec=config["stop_xsec"], inputs=[dict(root=str(root), sidecar=str(metadata_path))])
        write(work / "gnn_request.json", request)
        execute([sys.executable, "-u", "-m", "gnn_lowdm.eval", "cr-partial",
                 "--request", work / "gnn_request.json", "--output", work / "gnn.json",
                 "--model", config["gnn_model"], "--selection", config["gnn_selection"],
                 "--configuration", config["gnn_configuration"], "--raw-dy",
                 "--regions", "SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M"], gnn_env, work)
        gnn = read(work / "gnn.json")
        if gnn["status"] != "complete" or gnn["bad_files"] or gnn["input_files_valid"] != 1:
            raise RuntimeError("GNN histogram validation failed")
        report["stages"]["gnn"] = gnn["status"]
        del gnn
        products = {}
        compact = config.get("compact_output", False)
        names = [metadata_path.name, "main.json", "gnn.json", "trota.json", "truth.json"]
        if not compact:
            names.insert(0, root.name)
        report["root_retained"] = not compact
        report["validated_root_sha256"] = sha(root)
        report["validated_root_bytes"] = root.stat().st_size
        for name in names:
            output_name = name + ".gz" if compact else name
            target = destination / output_name
            if target.exists():
                raise FileExistsError(target)
            temporary = target.with_name(output_name + ".partial")
            source = work / name
            uncompressed_sha = sha(source)
            if compact:
                compressed = work / output_name
                with source.open("rb") as src, compressed.open("wb") as dst:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0) as archive:
                        shutil.copyfileobj(src, archive)
                digest = hashlib.sha256()
                with gzip.open(compressed, "rb") as archive:
                    for block in iter(lambda: archive.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != uncompressed_sha:
                    raise RuntimeError("compressed product round-trip mismatch")
                source = compressed
            shutil.copyfile(source, temporary)
            expected = sha(source)
            if sha(temporary) != expected:
                raise RuntimeError("stage-out checksum mismatch")
            temporary.replace(target)
            products[output_name] = dict(path=str(target), sha256=expected,
                                        uncompressed_sha256=uncompressed_sha, bytes=target.stat().st_size)
        report.update(status="complete_with_bad_files" if report["bad_files"] else "complete",
                      products=products, finished=time.time(),
                      validation="individual shifted job complete; nominal closure pending",
                      config_sha256=sha(args.config))
        checkpoint()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", finished=time.time())
        checkpoint()
        raise


def histogram_leaves(value, path=()):
    if not isinstance(value, dict):
        return {}
    if "sumw" in value and "sumw2" in value:
        return {"/".join(path): value}
    result = {}
    for name, item in value.items():
        result.update(histogram_leaves(item, path + (name,)))
    return result


def compare_histograms(reference, current):
    import numpy as np
    left, right = histogram_leaves(read(reference)), histogram_leaves(read(current))
    if not left or left.keys() != right.keys():
        raise RuntimeError("histogram structure mismatch: " + str(current))
    checked, largest = 0, 0.0
    for path in left:
        for field in ("sumw", "sumw2", "entries"):
            if field not in left[path] and field not in right[path]:
                continue
            a, b = np.asarray(left[path][field]), np.asarray(right[path][field])
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise RuntimeError("invalid histogram array: " + path + "/" + field)
            checked += a.size
            difference = float(np.max(np.abs(a - b))) if a.size else 0.0
            largest = max(largest, difference)
            matches = np.array_equal(a, b) if field == "entries" else np.allclose(a, b, rtol=1e-10, atol=1e-10)
            if not matches:
                raise RuntimeError("nominal histogram mismatch: " + path + "/" + field)
    return dict(status="passed", histogram_leaves=len(left), checked_bin_values=checked,
                maximum_absolute_difference=largest, reference_sha256=sha(reference),
                current_sha256=sha(current))


def tree_identities(root, name):
    import uproot
    names = ["file_id", "entry", "run", "luminosityBlock", "event"]
    if name == "TROTA":
        names.extend(f"TopResolved1pct_sourceJetIdx{i}" for i in range(3))
    with uproot.open(root) as source:
        arrays = source[name].arrays(names, library="np")
        rows = list(zip(*(arrays[field].tolist() for field in names)))
    if len(set(rows)) != len(rows):
        raise RuntimeError("duplicate identities in " + str(root) + ":" + name)
    return set(rows)


def validate(args):
    import math
    import uproot
    config = read(args.config)
    output, reference = eos(args.output), eos(args.reference)
    sys.path.insert(0, str(Path(config["repo"]) / "autonomous_allhad"))
    from autonomous_allhad.flat_ntuple_worker import _root_content_digests
    from autonomous_allhad.trota_resolved_2024_inplace import verify_complete_root

    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return True

    campaign = Path(config["repo"]) / "autonomous_allhad/workflow/systematic_propagation"
    events = (campaign / f"logs/pilot.{args.cluster}.log").read_text().split("\n...\n")
    source_metadata = read(config["source_sidecar"])
    report = dict(status="passed", scope="single complete MC-shard pilot", year=config["year"],
                  config_sha256=sha(args.config), validator_sha256=sha(Path(__file__)),
                  reference=str(reference), shifts={})
    identities = {}
    for proc, shift in enumerate(SHIFTS):
        terminations = [event for event in events if event.startswith(f"005 ({args.cluster}.{proc:03d}.000)")]
        if not terminations or "Normal termination (return value 0)" not in terminations[-1]:
            raise RuntimeError(f"successful termination not recorded: {args.cluster}.{proc}")
        directory = output / shift
        result = read(directory / "result.json")
        if result["status"] != "complete" or result["config_sha256"] != sha(args.config):
            raise RuntimeError("incomplete or changed job: " + shift)
        expected_products = {"mc_shard_00000.root", "mc_shard_00000.json", "main.json", "gnn.json", "trota.json", "truth.json"}
        if set(result["products"]) != expected_products:
            raise RuntimeError("missing expected products: " + shift)
        for name, product in result["products"].items():
            path = directory / name
            if str(path) != product["path"] or sha(path) != product["sha256"] or path.stat().st_size != product["bytes"]:
                raise RuntimeError("product integrity mismatch: " + str(path))
            if name.endswith(".json") and not finite(read(path)):
                raise RuntimeError("non-finite JSON value: " + str(path))
        metadata = read(directory / "mc_shard_00000.json")
        if (metadata["status"] != "complete" or metadata["bad_files"]
                or metadata["files_processed"] != source_metadata["files_processed"]
                or metadata["events_read"] != source_metadata["events_read"]):
            raise RuntimeError("input coverage mismatch: " + shift)
        root = directory / "mc_shard_00000.root"
        if metadata["root_sha256"] != sha(root):
            raise RuntimeError("metadata ROOT checksum mismatch: " + shift)
        trota = verify_complete_root(root, target_year=config["year"])
        with uproot.open(root) as source:
            marker = json.loads(str(source["TopWTruth_metadata"]))
            if (marker["status"] != "complete" or marker["application_year"] != config["year"]
                    or source["TopWTruth"].num_entries != source["Events"].num_entries
                    or marker["events_entries"] != source["Events"].num_entries
                    or metadata["events_written"] != source["Events"].num_entries):
                raise RuntimeError("Top/W truth coverage mismatch: " + shift)
        contents = _root_content_digests(root)
        original = {key: value for key, value in contents.items() if key not in ("TopWTruth", "TopWTruth_metadata")}
        if contents["TopWTruth"] != marker["truth_content"] or original != marker["original_contents"]:
            raise RuntimeError("Top/W truth content mismatch: " + shift)
        for kind in ("main", "gnn"):
            histograms = read(directory / (kind + ".json"))
            if histograms["status"] != "complete" or histograms.get("bad_files"):
                raise RuntimeError("incomplete histogram output: " + shift + "/" + kind)
        identities[shift] = tree_identities(root, "Events")
        report["shifts"][shift] = dict(exit_code=0, files=metadata["files_processed"],
            events_read=metadata["events_read"], events_written=metadata["events_written"],
            trota=trota["counts"], root_sha256=sha(root), truth_integrity="passed")
    for shift in SHIFTS:
        report["shifts"][shift].update(entering=len(identities[shift] - identities["nominal"]),
                                      leaving=len(identities["nominal"] - identities[shift]))
    for tree in ("Events", "TROTA", "TopWTruth"):
        if tree_identities(Path(config["source_root"]), tree) != tree_identities(output / "nominal/mc_shard_00000.root", tree):
            raise RuntimeError("nominal identity mismatch: " + tree)
    report["histogram_closure"] = {
        kind: compare_histograms(reference / (kind + ".json"), output / "nominal" / (kind + ".json"))
        for kind in ("main", "gnn")}
    state_path = campaign / "campaign_state.json"
    state = read(state_path)
    validation = state.setdefault("pilot_validation", {})
    validation.setdefault(str(config["year"]), {})["validation_cli"] = report
    if all(validation.get(str(year), {}).get("validation_cli", {}).get("status") == "passed" for year in (2024, 2025)):
        state["status"] = "met_unclustered_pilot_validated"
        state["pending"] = [item for item in state["pending"] if item not in ("pilot execution", "nominal closure")]
    write(state_path, state)
    print(json.dumps(report, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    objects = commands.add_parser("prepare-objects")
    objects.add_argument("--repo", required=True, type=Path)
    objects.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    objects.add_argument("--label", default="objects_intermediate_20260912")
    objects.add_argument("--proxy", required=True, type=Path)
    objects.add_argument("--refresh-unsubmitted", action="store_true")
    object_job = commands.add_parser("object-worker")
    object_job.add_argument("--config", required=True, type=Path)
    object_job.add_argument("--output", required=True, type=Path)
    object_job.add_argument("--work", type=Path)
    object_job.add_argument("--shifts", nargs="+", choices=("nominal", *OBJECT_SHIFTS), default=list(OBJECT_SHIFTS))
    prep = commands.add_parser("prepare")
    prep.add_argument("--repo", required=True, type=Path)
    prep.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    prep.add_argument("--label", default="pilot", choices=("pilot", "pilot_retry1"))
    full = commands.add_parser("prepare-full")
    full.add_argument("--repo", required=True, type=Path)
    full.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    full.add_argument("--label", required=True, choices=("full_met_20260911",))
    full.add_argument("--proxy", required=True, type=Path)
    check = commands.add_parser("preflight")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    job = commands.add_parser("worker")
    job.add_argument("--config", required=True, type=Path)
    job.add_argument("--shift", required=True, choices=SHIFTS)
    job.add_argument("--output", required=True, type=Path)
    validation = commands.add_parser("validate")
    validation.add_argument("--config", required=True, type=Path)
    validation.add_argument("--output", required=True, type=Path)
    validation.add_argument("--reference", required=True, type=Path)
    validation.add_argument("--cluster", required=True, type=int)
    args = parser.parse_args()
    return {"prepare-objects": prepare_objects, "object-worker": object_worker,
            "prepare": prepare, "prepare-full": prepare_full, "preflight": preflight,
            "worker": worker, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
