"""Atomic compact artifacts, strict provenance, and ROOT-only event inputs."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
from .paths import internal_path, analysis_era, tagging_definition, EARLY_ERAS


def sha256(path):
    h = hashlib.sha256()
    with internal_path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    p = internal_path(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as stream:
        return json.load(stream)


def era_assets(year):
    """Return the target era's registered inputs and their contract digest."""
    from .paths import ROOT
    era = analysis_era(year)
    if era not in EARLY_ERAS:
        path = ROOT / 'jsons/assets.json'
        return sha256(path), read_json(path)['files']
    registry = read_json(ROOT / 'jsons/early_run3_assets.json')
    records = [r for r in registry['files'] if r['era'] == era]
    records.extend(r for r in registry.get('analysis_inputs', {}).values()
                   if r['era'] == era)
    files = sorted((dict(target=r['target'], sha256=r['sha256']) for r in records),
                   key=lambda r: r['target'])
    if not files or len({r['target'] for r in files}) != len(files):
        raise ValueError('Missing or duplicate era assets: ' + era)
    digest = hashlib.sha256(json.dumps(dict(era=era, files=files),
                                      sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, files


def load_efficiency(path):
    """Read existing Coffea measurements across NumPy's private-module rename.

    No array, axis, count or uncertainty is transformed. Restrict this loader
    to the installed analysis-owned efficiency files, not arbitrary pickles.
    """
    from .paths import ROOT
    p = internal_path(path)
    if p.parent != ROOT / 'estimations' or p.suffix != '.merged':
        raise ValueError('Efficiency input must be installed in estimations')
    return decode_efficiency(p.read_bytes())


def decode_efficiency(data):
    """Decode trusted, checksum-verified analysis efficiency bytes only."""
    import io
    import pickle
    import numpy as np
    import lz4.frame

    class EfficiencyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if int(np.__version__.split('.')[0]) < 2 and (
                    module == 'numpy._core' or module.startswith('numpy._core.')):
                module = 'numpy.core' + module[len('numpy._core'):]
            return super().find_class(module, name)

    with lz4.frame.open(io.BytesIO(data), 'rb') as stream:
        return EfficiencyUnpickler(stream).load()


def write_json(path, value):
    p = internal_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = internal_path(p.with_name(p.name + ".pending." + str(os.getpid())))
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(temporary, "wt") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
    os.replace(temporary, p)


def root_metadata(root_path, root_file):
    """New ntuples are self-describing; old adjacent metadata is read-only input."""
    root_path = internal_path(root_path)
    if "TROTASR_metadata" in root_file:
        return json.loads(str(root_file["TROTASR_metadata"]))
    p = Path(root_path).with_suffix(".json")
    if not p.is_file():
        raise ValueError("Input has neither in-ROOT nor legacy file metadata: " + str(root_path))
    return read_json(p)


def validate_root(root_path, year, require_mixed=True):
    root_path = internal_path(root_path)
    import uproot
    with uproot.open(root_path, object_cache=None, array_cache=None) as f:
        required = {"Events", "TROTA", "TROTA_metadata"}
        if require_mixed:
            required.add("TopMixedResults_metadata")
        missing = required - set(f.keys(cycle=False))
        if missing:
            raise ValueError("Missing ROOT objects: " + ",".join(sorted(missing)))
        era = analysis_era(year)
        tag = tagging_definition(era)
        early = era in EARLY_ERAS
        meta = root_metadata(root_path, f)
        if early and meta.get('campaign_year') != era:
            raise ValueError('Intermediate target era differs from requested analysis era')
        trota = json.loads(str(f["TROTA_metadata"]))
        if trota.get("status") != "complete":
            raise ValueError("Incomplete TROTA")
        if (str(trota.get("application_year")) != era[:4] or trota.get("events_entries") != f["Events"].num_entries
                or trota.get("model_sha256") != tag['resolved_sha256']):
            raise ValueError("Resolved year/model/coverage contract mismatch")
        if meta.get("status") not in ("complete", "complete_with_bad_files"):
            raise ValueError("Incomplete input production")
        if require_mixed:
            mixed = json.loads(str(f["TopMixedResults_metadata"]))
            if (mixed.get("status") != "complete" or str(mixed.get("year")) != era[:4]
                    or mixed.get("operator") != ">="
                    or mixed.get("threshold") != tag['mixed_wp']
                    or (early and mixed.get('analysis_era') != era)):
                raise ValueError("Mixed model/year/WP contract mismatch")
            if mixed.get("entries") != f["Events"].num_entries:
                raise ValueError("Incomplete Mixed event coverage")
            if mixed.get("model_sha256") != tag['mixed_sha256']:
                raise ValueError("Mixed model hash mismatch")
        return meta
