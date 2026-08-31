from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "autonomous_allhad" / "workflow" / "run_flat_hists_chunked.py"
SPEC = importlib.util.spec_from_file_location("run_flat_hists_chunked", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


BUILD_OPTIONS = {
    "step_size": 50000,
    "only_regions": None,
    "only_variables": None,
    "require_btag": True,
    "require_weight_components": ["pileup", "btagSF"],
    "require_branches": True,
    "require_normalization": True,
    "nominal_only": True,
    "distribution_only": False,
    "only_signal_mass": None,
    "only_lowdm_sr_nsv_inclusive": False,
    "only_lowdm_nsv_repair": False,
    "dy_ptll_policy": "ptll200_only",
    "local_analysis_data": "0",
    "normalization_sha256": "test-normalization-sha256",
    "code_sha256": {
        "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py": "builder",
        "autonomous_allhad/workflow/run_flat_hists_chunked.py": "runner",
        "autonomous_allhad/autonomous_allhad/real_subset_worker.py": "worker",
        "autonomous_allhad/autonomous_allhad/dy_ptll_policy.py": "policy",
        "analysis/utils/corrections.py": "corrections",
        "analysis/data/corrections.coffea": "coffea",
        "analysis/data/PUweight/2024/puWeights.json.gz": "pileup",
        "analysis/data/BTVSF/2024/btagging.json.gz": "btagging",
        "analysis/data/EGammaSF/2024/electron.json.gz": "electron",
        "analysis/data/EGammaSF/2024/electronHlt.json.gz": "electron-hlt",
        "analysis/data/EGammaSF/2024/photon.json.gz": "photon",
        "analysis/data/MuonSF/2024/muon_Z.json.gz": "muon",
    },
    "btag_efficiency": {
        "path": "analysis/hists/btageff2024.merged",
        "exists": True,
        "sha256": "btag-efficiency",
        "expected_sha256": "btag-efficiency",
        "matches_expected": True,
    },
}


def chunk_payload(
    root: str,
    excluded_entries: int,
    normalization: str | None = None,
) -> dict:
    return {
        "schema_version": "flat_boosted_recoil_hists_v1",
        "status": "complete",
        "normalization": normalization,
        "summary": {
            "events_processed": excluded_entries,
            "input_roots": [root],
            "dy_ptll_policy": "ptll200_only",
            "build_options": BUILD_OPTIONS,
            "dy_ptll_dataset_exclusions": {
                "DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5": {
                    "dataset_id": 17,
                    "entries": excluded_entries,
                    "policy": "ptll200_only",
                }
            },
            "dy_ptll_prefilter": {
                "entries_scanned": excluded_entries + 2,
                "entries_loaded": 2,
                "read_ranges": 1,
            },
        },
        "histograms": {},
        "search_bin_histograms": {},
        "lowdm_variable_histograms": {},
        "highdm_variable_histograms": {},
    }


def test_resume_requires_matching_dy_policy(tmp_path: Path) -> None:
    path = tmp_path / "chunk.json"
    write_json(path, chunk_payload("/tmp/a.root", 3))
    assert MODULE.completed_chunk_matches(path, ["/tmp/a.root"], "ptll200_only")
    assert not MODULE.completed_chunk_matches(path, ["/tmp/a.root"], "all")


def test_strict_resume_rejects_warning_bearing_chunk(tmp_path: Path) -> None:
    path = tmp_path / "chunk.json"
    payload = chunk_payload("/tmp/a.root", 3)
    payload["status"] = "complete_with_weight_fallbacks"
    write_json(path, payload)
    assert MODULE.completed_chunk_matches(path, ["/tmp/a.root"], "ptll200_only")
    assert not MODULE.completed_chunk_matches(
        path,
        ["/tmp/a.root"],
        "ptll200_only",
        require_clean_status=True,
    )


def test_strict_resume_rejects_complete_status_with_warning_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunk.json"
    payload = chunk_payload("/tmp/a.root", 3)
    payload["summary"]["zero_entry_roots"] = ["/tmp/empty.root"]
    write_json(path, payload)
    assert payload["status"] == "complete"
    assert not MODULE.completed_chunk_matches(
        path,
        ["/tmp/a.root"],
        "ptll200_only",
        require_clean_status=True,
    )


def test_resume_requires_matching_build_options_and_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunk.json"
    normalization = tmp_path / "normalization.json"
    other_normalization = tmp_path / "other_normalization.json"
    write_json(normalization, {})
    write_json(other_normalization, {})
    write_json(
        path,
        chunk_payload("/tmp/a.root", 3, str(normalization)),
    )
    assert MODULE.completed_chunk_matches(
        path,
        ["/tmp/a.root"],
        "ptll200_only",
        expected_build_options=BUILD_OPTIONS,
        expected_normalization=normalization,
    )
    changed_options = {**BUILD_OPTIONS, "require_branches": False}
    assert not MODULE.completed_chunk_matches(
        path,
        ["/tmp/a.root"],
        "ptll200_only",
        expected_build_options=changed_options,
        expected_normalization=normalization,
    )
    assert not MODULE.completed_chunk_matches(
        path,
        ["/tmp/a.root"],
        "ptll200_only",
        expected_build_options=BUILD_OPTIONS,
        expected_normalization=other_normalization,
    )


def test_zero_entry_acceptance_is_not_a_histogram_build_difference() -> None:
    recorded = {**BUILD_OPTIONS, "allow_zero_entry_roots": False}
    expected = {**BUILD_OPTIONS, "allow_zero_entry_roots": True}
    assert MODULE.compatible_build_options(recorded, expected)

    changed_physics = {**expected, "require_branches": False}
    assert not MODULE.compatible_build_options(recorded, changed_physics)


def test_merge_preserves_policy_and_sums_exclusions(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    write_json(first, chunk_payload("/tmp/a.root", 3))
    write_json(second, chunk_payload("/tmp/b.root", 5))
    write_json(normalization, {})

    merged = MODULE.merge_payloads(
        [first, second],
        output,
        normalization,
        "ptll200_only",
    )

    summary = merged["summary"]
    assert summary["dy_ptll_policy"] == "ptll200_only"
    assert summary["events_processed"] == 8
    record = summary["dy_ptll_dataset_exclusions"][
        "DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5"
    ]
    assert record["entries"] == 8
    assert record["dataset_id"] == 17
    assert summary["dy_ptll_prefilter"] == {
        "entries_scanned": 12,
        "entries_loaded": 4,
        "read_ranges": 2,
    }


def test_merge_sums_nested_audit_counters(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    first_payload = chunk_payload("/tmp/a.root", 3)
    second_payload = chunk_payload("/tmp/b.root", 5)
    first_payload["summary"]["histogram_range_exclusions"] = {
        "DY2E": {"ut": {"DY": {"below": 2, "above_or_equal": 1}}}
    }
    second_payload["summary"]["histogram_range_exclusions"] = {
        "DY2E": {"ut": {"DY": {"below": 3, "above_or_equal": 4}}}
    }
    first_payload["summary"]["lowdm_search_bin_entry_accounting"] = {
        "cat5_DY2E_lowDeltaM": {
            "DY": {
                "selected_entries": 4,
                "assigned_entries": 4,
                "unassigned_entries": 0,
            }
        }
    }
    second_payload["summary"]["lowdm_search_bin_entry_accounting"] = {
        "cat5_DY2E_lowDeltaM": {
            "DY": {
                "selected_entries": 6,
                "assigned_entries": 5,
                "unassigned_entries": 1,
            }
        }
    }
    write_json(first, first_payload)
    write_json(second, second_payload)
    write_json(normalization, {})

    merged = MODULE.merge_payloads(
        [first, second],
        output,
        normalization,
        "ptll200_only",
    )

    assert merged["summary"]["histogram_range_exclusions"] == {
        "DY2E": {"ut": {"DY": {"below": 5, "above_or_equal": 5}}}
    }
    assert merged["summary"]["lowdm_search_bin_entry_accounting"] == {
        "cat5_DY2E_lowDeltaM": {
            "DY": {
                "selected_entries": 10,
                "assigned_entries": 9,
                "unassigned_entries": 1,
            }
        }
    }


def test_merge_marks_zero_entry_chunk_as_warning(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    payload = chunk_payload("/tmp/a.root", 0)
    payload["summary"]["zero_entry_roots"] = ["/tmp/a.root"]
    write_json(chunk, payload)
    write_json(normalization, {})

    merged = MODULE.merge_payloads(
        [chunk],
        output,
        normalization,
        "ptll200_only",
    )

    assert merged["status"] == "complete_with_warnings"


def test_merge_accepts_valid_empty_intermediate_root_when_explicit(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "chunk.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    payload = chunk_payload("/tmp/a.root", 0)
    payload["summary"]["zero_entry_roots"] = ["/tmp/a.root"]
    write_json(chunk, payload)
    write_json(normalization, {})

    merged = MODULE.merge_payloads(
        [chunk],
        output,
        normalization,
        "ptll200_only",
        allow_zero_entry_roots=True,
    )

    assert merged["status"] == "complete"


def test_merge_rejects_mixed_execution_contracts(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    first_payload = chunk_payload("/tmp/a.root", 3)
    second_payload = chunk_payload("/tmp/b.root", 5)
    second_payload["summary"]["build_options"] = {
        **BUILD_OPTIONS,
        "require_btag": False,
    }
    write_json(first, first_payload)
    write_json(second, second_payload)
    write_json(normalization, {})

    try:
        MODULE.merge_payloads(
            [first, second],
            output,
            normalization,
            "ptll200_only",
        )
    except RuntimeError as exc:
        assert "build options" in str(exc)
    else:
        raise AssertionError("mixed execution contracts were accepted")


def test_merge_rejects_duplicate_roots_across_chunks(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    normalization = tmp_path / "normalization.json"
    write_json(first, chunk_payload("/tmp/a.root", 3))
    write_json(second, chunk_payload("/tmp/a.root", 5))
    write_json(normalization, {})

    try:
        MODULE.merge_payloads(
            [first, second],
            output,
            normalization,
            "ptll200_only",
        )
    except RuntimeError as exc:
        assert "duplicate input ROOTs" in str(exc)
    else:
        raise AssertionError("duplicate input ROOTs were silently de-duplicated")


def test_runner_rejects_duplicate_input_list_before_launch(tmp_path: Path) -> None:
    root = tmp_path / "a.root"
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{root}\n{root}\n")
    old_argv = sys.argv
    sys.argv = [
        str(MODULE_PATH),
        "--repo",
        str(REPO),
        "--input-list",
        str(input_list),
        "--normalization",
        str(tmp_path / "normalization.json"),
        "--output",
        str(tmp_path / "output.json"),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    try:
        try:
            MODULE.main()
        except SystemExit as exc:
            assert "duplicate input ROOTs" in str(exc)
        else:
            raise AssertionError("duplicate input list was accepted")
    finally:
        sys.argv = old_argv


def test_required_btag_payload_preflight_checks_sha256(tmp_path: Path) -> None:
    expected = MODULE.file_sha256(
        Path(__file__)
    )
    try:
        MODULE.btag_efficiency_contract(tmp_path, expected, required=True)
    except RuntimeError as exc:
        assert "payload is missing" in str(exc)
    else:
        raise AssertionError("missing required b-tag payload was accepted")

    payload = tmp_path / MODULE.BTAG_EFFICIENCY_RELATIVE_PATH
    payload.parent.mkdir(parents=True)
    payload.write_bytes(Path(__file__).read_bytes())
    contract = MODULE.btag_efficiency_contract(tmp_path, expected, required=True)
    assert contract["exists"]
    assert contract["matches_expected"]

    try:
        MODULE.btag_efficiency_contract(tmp_path, "0" * 64, required=True)
    except RuntimeError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("mismatched required b-tag payload was accepted")


def test_runner_forwards_strict_candidate_arguments(tmp_path: Path) -> None:
    root = tmp_path / "a.root"
    root.touch()
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{root}\n")
    normalization = tmp_path / "normalization.json"
    write_json(normalization, {})
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            commands.append(list(cmd))
            environments.append(dict(kwargs["env"]))
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text("{}")

        def poll(self):
            return 0

    def fake_merge(
        chunks,
        output,
        normalization_path,
        policy,
        build_options,
        allow_hist_builder_repair,
    ):
        assert allow_hist_builder_repair is False
        payload = {
            "status": "complete",
            "summary": {
                "events_processed": 1,
                "input_roots": [str(root)],
            },
        }
        write_json(output, payload)
        return payload

    old_argv = sys.argv
    old_popen = MODULE.subprocess.Popen
    old_merge = MODULE.merge_payloads
    sys.argv = [
        str(MODULE_PATH),
        "--repo",
        str(REPO),
        "--input-list",
        str(input_list),
        "--normalization",
        str(normalization),
        "--output",
        str(tmp_path / "output.json"),
        "--work-dir",
        str(tmp_path / "work"),
        "--local-analysis-data",
        "1",
        "--only-variables",
        "ut",
        "ptll",
        "--require-weight-components",
        "pileup",
        "electron_id",
        "--require-branches",
        "--require-normalization",
        "--nominal-only",
        "--strict-complete",
        "--dy-ptll-policy",
        "ptll200_only",
    ]
    MODULE.subprocess.Popen = FakePopen
    MODULE.merge_payloads = fake_merge
    try:
        assert MODULE.main() == 0
    finally:
        MODULE.merge_payloads = old_merge
        MODULE.subprocess.Popen = old_popen
        sys.argv = old_argv

    assert len(commands) == 1
    command = commands[0]
    variable_index = command.index("--only-variables")
    assert command[variable_index + 1 : variable_index + 3] == ["ut", "ptll"]
    required_index = command.index("--require-weight-components")
    analysis_sf_index = command.index("--analysis-sf-components")
    assert command[required_index + 1 : analysis_sf_index] == [
        "pileup",
        "electron_id",
    ]
    require_branches_index = command.index("--require-branches")
    assert command[analysis_sf_index + 1 : require_branches_index] == (
        MODULE.REQUIRED_ANALYSIS_SF_COMPONENTS
    )
    assert "--require-normalization" in command
    assert "--nominal-only" in command
    assert command[command.index("--dy-ptll-policy") + 1] == "ptll200_only"
    expected_btag_index = command.index("--expected-btag-efficiency-sha256")
    assert command[expected_btag_index + 1] == (
        MODULE.EXPECTED_BTAG_EFFICIENCY_SHA256_2024
    )
    assert environments[0]["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] == "1"
    completion = json.loads(
        (tmp_path / "work" / "chunked_hist_results.json").read_text()
    )
    assert completion["output_sha256"] == MODULE.file_sha256(
        tmp_path / "output.json"
    )
