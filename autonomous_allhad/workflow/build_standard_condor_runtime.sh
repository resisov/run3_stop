#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR=/eos/user/t/taiwoo/run3_stop/runtime
PYTHON_PREFIX=/eos/user/t/taiwoo/miniconda3/envs/py38
COMBINE_PARENT=/eos/user/t/taiwoo/decaf/analysis/CombinedArea
COMBINE_RELEASE=CMSSW_14_1_0_pre4
CONDA_PACK_TOOLS=$RUNTIME_DIR/tools

PYTHON_ARCHIVE=$RUNTIME_DIR/py38.tgz
COMBINE_ARCHIVE=$RUNTIME_DIR/combine_cmssw_14_1_0_pre4.tgz
PYTHON_PARTIAL=$RUNTIME_DIR/py38.partial.tgz
COMBINE_PARTIAL=$RUNTIME_DIR/combine_cmssw_14_1_0_pre4.partial.tgz

mkdir -p "$RUNTIME_DIR/build"
cd "$RUNTIME_DIR"
export TMPDIR="$RUNTIME_DIR/build"
export PIP_CACHE_DIR="$RUNTIME_DIR/build/pip-cache"

test -x "$PYTHON_PREFIX/bin/python"
test -x "$COMBINE_PARENT/$COMBINE_RELEASE/bin/el9_amd64_gcc12/combine"
test -f "$COMBINE_PARENT/$COMBINE_RELEASE/src/HiggsAnalysis/CombinedLimit/scripts/text2workspace.py"
test -d "$CONDA_PACK_TOOLS/conda_pack"

if [[ ! -s "$PYTHON_ARCHIVE" ]]; then
    PYTHONPATH="$CONDA_PACK_TOOLS" /eos/user/t/taiwoo/miniconda3/bin/python - \
        "$PYTHON_PREFIX" "$PYTHON_PARTIAL" <<'PY'
import sys
from conda_pack import pack

pack(
    prefix=sys.argv[1],
    output=sys.argv[2],
    force=True,
    compress_level=4,
    n_threads=4,
)
PY
    mv -f "$PYTHON_PARTIAL" "$PYTHON_ARCHIVE"
fi

if [[ ! -s "$COMBINE_ARCHIVE" ]]; then
    tar -C "$COMBINE_PARENT" -cf - "$COMBINE_RELEASE" \
        | pigz -4 > "$COMBINE_PARTIAL"
    mv -f "$COMBINE_PARTIAL" "$COMBINE_ARCHIVE"
fi

tar -tzf "$PYTHON_ARCHIVE" > "$RUNTIME_DIR/build/py38_contents.txt"
tar -tzf "$COMBINE_ARCHIVE" > "$RUNTIME_DIR/build/combine_contents.txt"
grep -qx 'bin/python' "$RUNTIME_DIR/build/py38_contents.txt"
grep -qx 'bin/conda-unpack' "$RUNTIME_DIR/build/py38_contents.txt"
grep -qx "$COMBINE_RELEASE/bin/el9_amd64_gcc12/combine" \
    "$RUNTIME_DIR/build/combine_contents.txt"
grep -qx "$COMBINE_RELEASE/src/HiggsAnalysis/CombinedLimit/scripts/text2workspace.py" \
    "$RUNTIME_DIR/build/combine_contents.txt"

sha256sum "$PYTHON_ARCHIVE" "$COMBINE_ARCHIVE" > "$RUNTIME_DIR/SHA256SUMS"

PYTHON_VERSION=$(
    "$PYTHON_PREFIX/bin/python" -c 'import platform; print(platform.python_version())'
)
PACKAGE_VERSIONS=$(
    "$PYTHON_PREFIX/bin/python" -c \
        'import awkward,coffea,correctionlib,numpy,uproot; print("|".join([awkward.__version__,coffea.__version__,correctionlib.__version__,numpy.__version__,uproot.__version__]))'
)
COMBINE_COMMIT=$(git -C "$COMBINE_PARENT/$COMBINE_RELEASE/src/HiggsAnalysis/CombinedLimit" rev-parse HEAD)
HARVESTER_COMMIT=$(git -C "$COMBINE_PARENT/$COMBINE_RELEASE/src/CombineHarvester/CombineTools" rev-parse HEAD)

/eos/user/t/taiwoo/miniconda3/bin/python - \
    "$RUNTIME_DIR" "$PYTHON_VERSION" "$PACKAGE_VERSIONS" \
    "$COMBINE_RELEASE" "$COMBINE_COMMIT" "$HARVESTER_COMMIT" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

runtime = Path(sys.argv[1])
python_version = sys.argv[2]
awkward, coffea, correctionlib, numpy, uproot = sys.argv[3].split("|")

def record(name):
    path = runtime / name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}

payload = {
    "schema_version": 1,
    "status": "ready",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": {
        **record("py38.tgz"),
        "version": python_version,
        "packages": {
            "awkward": awkward,
            "coffea": coffea,
            "correctionlib": correctionlib,
            "numpy": numpy,
            "uproot": uproot,
        },
    },
    "combine": {
        **record("combine_cmssw_14_1_0_pre4.tgz"),
        "cmssw": sys.argv[4],
        "combined_limit_commit": sys.argv[5],
        "combine_harvester_commit": sys.argv[6],
    },
}
temporary = runtime / f"runtime_manifest.json.partial.{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, runtime / "runtime_manifest.json")
print(json.dumps(payload, sort_keys=True))
PY
