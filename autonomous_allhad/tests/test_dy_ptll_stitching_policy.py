from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "autonomous_allhad" / "autonomous_allhad" / "dy_ptll_policy.py"
SPEC = importlib.util.spec_from_file_location("dy_ptll_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ptll200_only_keeps_exactly_the_requested_dy_threshold() -> None:
    allowed = MODULE.dy_ptll_dataset_allowed
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-200_TuneCP5", "DY", "ptll200_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5", "DY", "ptll200_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-400_TuneCP5", "DY", "ptll200_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5", "DY", "ptll200_only")


def test_ptll100_only_keeps_broad_inclusive_sample_for_migration_check() -> None:
    allowed = MODULE.dy_ptll_dataset_allowed
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5", "DY", "ptll100_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-200_TuneCP5", "DY", "ptll100_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-400_TuneCP5", "DY", "ptll100_only")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5", "DY", "ptll100_only")


def test_ptll100_200_keeps_exactly_two_requested_production_bins() -> None:
    allowed = MODULE.dy_ptll_dataset_allowed
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5", "DY", "ptll100_200")
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-200_TuneCP5", "DY", "ptll100_200")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-400_TuneCP5", "DY", "ptll100_200")
    assert not allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5", "DY", "ptll100_200")


def test_ptll_policy_does_not_remove_non_dy_samples() -> None:
    allowed = MODULE.dy_ptll_dataset_allowed
    assert allowed("TTto2L2Nu", "TT", "ptll200_only")
    assert allowed("EGamma0-Run2024G", "EGamma", "ptll200_only")


def test_default_policy_preserves_nominal_inputs() -> None:
    allowed = MODULE.dy_ptll_dataset_allowed
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5", "DY", "all")
    assert allowed("DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5", "DY", "all")


def test_true_ranges_returns_half_open_contiguous_spans() -> None:
    assert MODULE.true_ranges([False, True, True, False, True]) == [
        (1, 3),
        (4, 5),
    ]
    assert MODULE.true_ranges([]) == []


def test_ptll_prefilter_plan_reads_only_allowed_spans_and_audits_exclusions() -> None:
    plan = MODULE.dataset_id_prefilter_plan(
        [100, 100, 200, 200, 400, 200, 600, 600],
        {200},
    )
    assert plan == {
        "ranges": [(2, 4), (5, 6)],
        "excluded_counts": {100: 2, 400: 1, 600: 2},
        "entries_scanned": 8,
        "entries_loaded": 3,
    }
