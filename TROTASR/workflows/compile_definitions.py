"""Compile the local public ID/correction definitions and record exact hashes."""
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import sha256, write_json


def main():
    from coffea.util import save, load
    from TROTASR.utils import ids, pog_corrections
    folder = internal_path(ROOT / "utils/compiled")
    folder.mkdir(parents=True, exist_ok=True)
    save(pog_corrections.corrections, str(internal_path(folder / "corrections.coffea")))
    save({n: getattr(ids, n) for n in dir(ids) if n.startswith("is") or n == "object_masks"},
         str(internal_path(folder / "ids.coffea")))
    for name in ("corrections", "ids"):
        if not load(str(folder / (name + ".coffea"))):
            raise RuntimeError("Compiled artifact is empty")
    sources = ["utils/" + n + ".py" for n in ("corrections", "pog_corrections", "private_scales",
               "weight_components", "weight_inputs", "paths", "ids", "nanoaod_ids")]
    write_json(folder / "manifest.json", dict(status="complete", sources={p: sha256(ROOT / p) for p in sources},
        artifacts={name: sha256(folder / name) for name in ("ids.coffea", "corrections.coffea")}))
    print("Local IDs and corrections compiled and reloaded")


if __name__ == "__main__":
    main()
