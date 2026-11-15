"""Read analysis inputs and write JSON products atomically."""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path
from .paths import internal_path, analysis_era, tagging_definition, EARLY_ERAS


def file_identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


_HASH_MARKERS = ('sha256', 'sha1', 'sha512', 'md5', 'checksum', 'adler32', 'digest')
_PROVENANCE_FIELDS = {
    'approval', 'approved_date', 'adopted', 'code', 'execution',
    'input_identity', 'object_payloads', 'origin', 'output_identity',
    'provenance', 'source_provenance', 'source_receipt', 'source_updated',
    'user_approved',
}


def _hash_field(key):
    name = str(key).lower()
    return 'hash' in name or any(marker in name for marker in _HASH_MARKERS)


def _without_fields(value, excluded):
    if isinstance(value, dict):
        return {key: _without_fields(item, excluded) for key, item in value.items()
                if not excluded(key)}
    if isinstance(value, list):
        return [_without_fields(item, excluded) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_fields(item, excluded) for item in value)
    return value


def without_hash_fields(value):
    """Remove stored hash fields from comparisons of current and older records."""
    return _without_fields(value, _hash_field)


def without_provenance_fields(value):
    """Compare scientific fields independently of execution and receipt history."""
    return _without_fields(value, lambda key: key in _PROVENANCE_FIELDS or _hash_field(key))


def scientific_settings(value):
    """Extract scientific settings from configuration and execution records."""
    return _without_fields(value, lambda key: key == 'id' or key in _PROVENANCE_FIELDS or _hash_field(key))


def read_json(path):
    p = internal_path(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as stream:
        return json.load(stream)


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
    """Decode trusted, installed analysis efficiency bytes."""
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
        if (str(trota.get("application_year")) != era[:4]
                or trota.get("events_entries") != f["Events"].num_entries):
            raise ValueError("Resolved year or event coverage mismatch")
        if "resolved_passing" in trota and trota["resolved_passing"] != f["TROTA"].num_entries:
            raise ValueError("Resolved candidate count mismatch")
        if meta.get("status") not in ("complete", "complete_with_bad_files"):
            raise ValueError("Incomplete input production")
        if require_mixed:
            mixed = json.loads(str(f["TopMixedResults_metadata"]))
            if (mixed.get("status") != "complete" or str(mixed.get("year")) != era[:4]
                    or mixed.get("operator") != ">="
                    or mixed.get("threshold") != tag['mixed_wp']
                    or (early and mixed.get('analysis_era') != era)):
                raise ValueError("Mixed year or working point mismatch")
            if mixed.get("entries") != f["Events"].num_entries:
                raise ValueError("Incomplete Mixed event coverage")
        return meta
