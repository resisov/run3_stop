"""One-time, auditable mechanical extraction of existing physics functions.

This is a development tool, not a runtime dependency. It never copies campaign
controllers. Renderer files are copied byte-for-byte. Extraction records source
hashes and selected symbols; all normal execution imports TROTASR only.
Sources must first be explicitly approved and staged in TROTASR/import_staging.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import internal_path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "import_staging"
PACKAGE = REPO / "autonomous_allhad/autonomous_allhad"
WF = REPO / "autonomous_allhad/workflow"
GNN = REPO / "autonomous_allhad/gnn_lowdm"
RECORDS = []
HEADER = '''from __future__ import annotations
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
'''


def symbols(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]
    return []


def extract(source, names, destination, imports="", exclude=(), transform=None):
    source, destination = internal_path(source), internal_path(destination)
    raw = source.read_text()
    tree = ast.parse(raw)
    by_name = {name: node for node in tree.body for name in symbols(node)}
    selected = set(names)
    while True:
        before = selected.copy()
        for name in before:
            if name not in by_name:
                raise ValueError(f"Missing migration symbol {source}:{name}")
            selected.update(n.id for n in ast.walk(by_name[name])
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                            and n.id in by_name and n.id not in exclude)
        if before == selected:
            break
    nodes = {id(by_name[n]) for n in selected}
    lines = raw.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        if id(node) not in nodes:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        chunks.append("".join(lines[start - 1:node.end_lineno]))
    code = HEADER + imports + "\n\n" + "\n\n".join(chunks) + "\n"
    if transform:
        code = transform(code)
    compile(code, str(destination), "exec")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(code)
    RECORDS.append(dict(source=str(source.relative_to(REPO)),
                        source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        target=str(destination.relative_to(ROOT)), symbols=sorted(selected),
                        sha256=hashlib.sha256(code.encode()).hexdigest()))


def copied(source, destination):
    source, destination = internal_path(source), internal_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    destination.write_bytes(raw)
    RECORDS.append(dict(source=str(source.relative_to(REPO)), target=str(destination.relative_to(ROOT)),
                        byte_identical=True, sha256=hashlib.sha256(raw).hexdigest()))


def relocate_payload_calls(code):
    """Wrap only file arguments, retaining calibration formulae unchanged."""
    tree = ast.parse(code)
    edits = []
    lines = code.splitlines(keepends=True)
    offsets, current = [], 0
    for line in lines:
        offsets.append(current)
        current += len(line)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = ast.unparse(node.func)
        if func not in ("correctionlib.CorrectionSet.from_file", "load"):
            continue
        arg = node.args[0]
        fragment = ast.get_source_segment(code, arg)
        a = offsets[arg.lineno - 1] + arg.col_offset
        b = offsets[arg.end_lineno - 1] + arg.end_col_offset
        edits.append((a, b, "str(resolve_payload(" + fragment + "))"))
    for a, b, value in sorted(edits, reverse=True):
        code = code[:a] + value + code[b:]
    return code


def main():
    if not REPO.is_dir():
        raise ValueError('No approved sources staged inside TROTASR/import_staging')
    for path in REPO.rglob('*'):
        internal_path(path)
    region = GNN / "_implementation/region_io.py"
    extract(region, ["object_masks", "clean_by_delta_r", "as_ak_bool"], ROOT / "utils/ids.py",
            imports="from .nanoaod_ids import isTrackElectron, isTrackMuon, isTrackPion, isVetoElectron, isMediumElectron, isLooseMuon, isMediumMuon, isMediumTau, isGoodJet, isGoodFatJet, isJetVeto\n")
    extract(region, ["update_veto_leptons", "fatjet_kinematics", "jet_kinematics", "build_region_blocks",
                     "SELECTION_BRANCHES", "DATA_STREAM", "HISTOGRAMS", "REGIONS"],
            ROOT / "utils/region_kinematics.py", exclude=("object_masks", "clean_by_delta_r", "as_ak_bool"),
            imports="from .ids import object_masks, clean_by_delta_r, as_ak_bool\n")
    extract(PACKAGE / "highdm_resolved_categories.py",
            ["boosted_overlap_vetoed_ak4_indices", "select_exclusive_resolved_candidates",
             "map_candidates_to_events", "map_candidates_to_events_rle"], ROOT / "utils/constituents.py")
    copied(PACKAGE / "signal_models.py", ROOT / "utils/signal_models.py")
    extract(REPO / "analysis/utils/ids.py",
            ["isTrackElectron", "isTrackMuon", "isTrackPion", "isVetoElectron", "isMediumElectron",
             "isLooseMuon", "isMediumMuon", "isMediumTau", "isGoodJet", "isGoodFatJet", "isJetVeto"],
            ROOT / "utils/nanoaod_ids.py",
            imports="import correctionlib\nfrom .paths import payload_path as resolve_payload\n",
            transform=relocate_payload_calls)
    extract(REPO / "analysis/utils/corrections.py", ["corrections"], ROOT / "utils/pog_corrections.py",
            imports="import correctionlib\nfrom correctionlib import convert\nimport hist\nfrom coffea.util import load\nfrom coffea.lookup_tools.dense_lookup import dense_lookup\nfrom .paths import payload_path as resolve_payload\n",
            transform=relocate_payload_calls)
    # Private SFs keep the adopted file/bin fallback and efficiency saturation.
    private = (PACKAGE / "analysis_scale_factors.py").read_text()
    private = private.replace('repo / "analysis/hists"', 'repo / "estimations"')
    private = private.replace('/ "analysis/data/AnalysisSF"', '/ "scales/Private"')
    (ROOT / "utils/private_scales.py").write_text(private)
    RECORDS.append(dict(source="autonomous_allhad/autonomous_allhad/analysis_scale_factors.py",
                        target="utils/private_scales.py", changes="payload paths only",
                        source_sha256=hashlib.sha256((PACKAGE / "analysis_scale_factors.py").read_bytes()).hexdigest(),
                        sha256=hashlib.sha256(private.encode()).hexdigest()))
    extract(PACKAGE / "real_subset_worker.py", ["compute_weight_bundle"], ROOT / "utils/weight_components.py",
            imports="from .corrections import load_analysis_corrections, analysis_workdir\nfrom .private_scales import (DEFAULT_ANALYSIS_SF_COMPONENTS, REQUIRED_ANALYSIS_SF_COMPONENTS, loose_muon_lowpt_triplet, met_trigger_triplet, photon_trigger_triplet, veto_electron_lowpt_triplet)\n",
            exclude=("load_analysis_corrections", "analysis_workdir"))
    extract(WF / "build_flat_boosted_recoil_hists.py", ["flat_arrays_for_weights", "signal_btag_efficiency_dataset", "WEIGHT_BRANCHES"],
            ROOT / "utils/weight_inputs.py",
            exclude=("region_mask",),
            imports="from .event_selections import legacy_vr_mask as region_mask\nfrom .signal_models import signal_topology\n")
    extract(WF / "build_flat_boosted_recoil_hists.py", ["norm_vector", "dataset_label", "data_process_allowed"],
            ROOT / "utils/normalization.py",
            imports="from .signal_models import signal_topology, signal_mass_key\n")
    extract(GNN / "data.py", ["_diagonal_v3_features", "_pad", "DIAGONAL_V3_GLOBAL_FEATURE_NAMES"],
            ROOT / "gnn4lowdm/features.py")
    extract(GNN / "_implementation/diagonal_v3_region_features.py", ["feature_arrays"],
            ROOT / "gnn4lowdm/region_features.py",
            imports="from TROTASR.utils import region_kinematics as base\nfrom .features import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _diagonal_v3_features, _pad\n")
    copied(GNN / "_implementation/rank005_numpy.py", ROOT / "gnn4lowdm/inference.py")
    copied(GNN / "config.json", ROOT / "gnn4lowdm/config.json")
    # Untouched renderer bytes and their pure data-format dependencies. These
    # modules cannot see the parent analysis repository at runtime.
    for name in ("plot_control_search_bins_style.py", "postprocess_limits.py",
                 "background_process_groups.py", "signal_theory.py"):
        copied(WF / name, ROOT / "utils/renderers" / name)
    copied(PACKAGE / "search_bin_categorization.py",
           ROOT / "utils/renderers/autonomous_allhad/search_bin_categorization.py")
    copied(PACKAGE / "highdm_resolved_categories.py",
           ROOT / "utils/renderers/autonomous_allhad/highdm_resolved_categories.py")
    (ROOT / "utils/renderers/__init__.py").write_text('"""Byte-preserved legacy renderers; see sources.json."""\n')
    (ROOT / "utils/renderers/autonomous_allhad/__init__.py").write_text('')
    (ROOT / "jsons/sources.json").write_text(json.dumps(dict(schema="trotasr_source_migration_v1", files=RECORDS), indent=2) + "\n")
    print(json.dumps({"migrated": len(RECORDS), "root": str(ROOT)}))


if __name__ == "__main__":
    main()
