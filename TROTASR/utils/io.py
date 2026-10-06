"""Atomic compact artifacts, strict provenance, and ROOT-only event inputs."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
from .paths import internal_path


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
        trota = json.loads(str(f["TROTA_metadata"]))
        if trota.get("status") != "complete":
            raise ValueError("Incomplete TROTA")
        if (trota.get("application_year") != year or trota.get("events_entries") != f["Events"].num_entries
                or trota.get("model_sha256") != "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"):
            raise ValueError("Resolved year/model/coverage contract mismatch")
        meta = root_metadata(root_path, f)
        if meta.get("status") not in ("complete", "complete_with_bad_files"):
            raise ValueError("Incomplete input production")
        if require_mixed:
            mixed = json.loads(str(f["TopMixedResults_metadata"]))
            if (mixed.get("status") != "complete" or mixed.get("year") != year
                    or mixed.get("operator") != ">="
                    or mixed.get("threshold") != 0.9027690887451172):
                raise ValueError("Mixed model/year/WP contract mismatch")
            if mixed.get("entries") != f["Events"].num_entries:
                raise ValueError("Incomplete Mixed event coverage")
            if mixed.get("model_sha256") != "40469a1a62f64b08b4225682a2df6ac2538f49a1e7eef5ef4bdff857d0b875a4":
                raise ValueError("Mixed model hash mismatch")
        return meta
