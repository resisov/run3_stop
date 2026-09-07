from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "autonomous_allhad" / "workflow"
PACKAGE_ROOT = REPO / "autonomous_allhad"
ESTIMATOR_SOURCES = (
    WORKFLOW / "build_histogram_tf_inputs_2024.py",
    WORKFLOW / "build_sgamma_ut_report_2024.py",
    WORKFLOW / "build_zgamma_double_ratio_2024.py",
    PACKAGE_ROOT / "autonomous_allhad" / "dy_estimation" / "build_measurement.py",
    PACKAGE_ROOT / "autonomous_allhad" / "dy_estimation" / "report.py",
)
GROUPS = ("Nb1", "Nb2plus")
REGIONS = ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
SAMPLES = ("data_obs", "DY", "GJ", "QCD", "ST", "TT", "VV", "WtoLNu", "Zto2Nu")


def leaf(values: list[float]) -> dict[str, object]:
    return {
        "nominal": {
            "sumw": values,
            "sumw2": [abs(value) for value in values],
            "entries": [int(value > 0.0) for value in values],
        }
    }


def compact_input() -> dict[str, object]:
    edges = [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0]
    mll_edges = [50.0, 71.0, 81.0, 91.0, 101.0, 111.0, 160.0, 250.0, 500.0]
    output: dict[str, object] = {
        "schema_version": "background_estimation_histograms_v1",
        "status": "complete",
        "category_policy": {"highdm": list(GROUPS), "lowdm": list(GROUPS)},
        "summary": {
            "datasets": {
                "DYto2E-4Jets_MLL-50": 1,
                "DYto2Mu-4Jets_MLL-50": 1,
                "DYto2Tau-4Jets_MLL-50": 1,
                "GJ-4Jets_Bin-HT-400to600": 1,
                "QCD-4Jets_Bin-HT-400to600": 1,
                "EGamma0-Run2024C": 1,
                "Muon0-Run2024C": 1,
                "JetMET0-Run2024C": 1,
            }
        },
        "provenance": {
            "intermediate_root_reread": False,
            "regions": list(REGIONS),
        },
        "dy_rz": {
            "mass_windows": {
                "on": [71.0, 111.0],
                "off": [[50.0, 71.0], [111.0, None]],
            },
            "mll_edges": mll_edges,
            "highdm": {"yields": {}, "mll": {}},
            "lowdm": {"yields": {}, "mll": {}},
        },
    }
    for regime in ("highdm", "lowdm"):
        recoil: dict[str, object] = {}
        for region in REGIONS:
            recoil[region] = {}
            for group in GROUPS:
                recoil[region][group] = {
                    sample: leaf([0.0] * 8) for sample in SAMPLES
                }
                if region == "GCR":
                    recoil[region][group]["data_obs"] = leaf([20.0] * 8)
                    recoil[region][group]["GJ"] = leaf([10.0] * 8)
                    recoil[region][group]["TT"] = leaf([2.0] * 8)
                elif region in {"DY2E", "DY2M"}:
                    recoil[region][group]["data_obs"] = leaf([12.0] * 8)
                    recoil[region][group]["DY"] = leaf([10.0] * 8)
                    recoil[region][group]["TT"] = leaf([1.0] * 8)
                elif region == "SR":
                    recoil[region][group]["Zto2Nu"] = leaf([10.0] * 8)
        output[regime] = {
            "recoil_edges": edges,
            "nb_groups": list(GROUPS),
            "recoil": recoil,
        }
        rz = output["dy_rz"][regime]
        for channel in ("DY2E", "DY2M"):
            rz["yields"][channel] = {}
            rz["mll"][channel] = {}
            for group in GROUPS:
                rz["yields"][channel][group] = {
                    "on": {
                        "data": leaf([100.0]),
                        "zll": leaf([80.0]),
                        "other": leaf([10.0]),
                    },
                    "off": {
                        "data": leaf([80.0]),
                        "zll": leaf([20.0]),
                        "other": leaf([50.0]),
                    },
                }
                rz["mll"][channel][group] = {
                    "data": leaf([10.0] * 8),
                    "zll": leaf([8.0] * 8),
                    "other": leaf([2.0] * 8),
                }
    return output


def test_active_estimators_reject_event_level_inputs() -> None:
    for path in ESTIMATOR_SOURCES:
        source = path.read_text()
        assert "import uproot" not in source
        assert "uproot.open" not in source
        assert "--exact" not in source
        assert "--low-sparse" not in source
    for path in ESTIMATOR_SOURCES[:3]:
        assert "--hist-input" in path.read_text()


def test_histogram_boundary_drives_tf_and_rz(tmp_path: Path) -> None:
    hist_input = tmp_path / "nominal_background_estimation.json"
    hist_input.write_text(json.dumps(compact_input()))
    tf_output = tmp_path / "tf.json"
    subprocess.run(
        [
            sys.executable,
            str(WORKFLOW / "build_histogram_tf_inputs_2024.py"),
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            "2024",
            "--output",
            str(tf_output),
        ],
        check=True,
        cwd=WORKFLOW,
    )
    tf = json.loads(tf_output.read_text())
    assert tf["status"] == "complete"
    assert tf["provenance"]["intermediate_root_reread"] is False
    assert tf["highdm"]["nb_groups"] == list(GROUPS)
    assert tf["lowdm"]["nb_groups"] == list(GROUPS)

    sgamma_dir = tmp_path / "sgamma"
    subprocess.run(
        [
            sys.executable,
            str(WORKFLOW / "build_sgamma_ut_report_2024.py"),
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            "2024",
            "--output-dir",
            str(sgamma_dir),
        ],
        check=True,
        cwd=WORKFLOW,
    )
    sgamma = json.loads((sgamma_dir / "sgamma_ut.json").read_text())
    assert sgamma["status"] == "complete"
    assert set(sgamma["highdm"]) == set(GROUPS)
    assert set(sgamma["lowdm_families"]) == set(GROUPS)
    assert sgamma["provenance"]["intermediate_root_reread"] is False

    double_ratio_dir = tmp_path / "double_ratio"
    subprocess.run(
        [
            sys.executable,
            str(WORKFLOW / "build_zgamma_double_ratio_2024.py"),
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            "2024",
            "--output-dir",
            str(double_ratio_dir),
        ],
        check=True,
        cwd=WORKFLOW,
    )
    double_ratio = json.loads(
        (double_ratio_dir / "zgamma_double_ratio.json").read_text()
    )
    assert double_ratio["status"] == "complete"
    assert double_ratio["provenance"]["intermediate_root_reread"] is False
    assert len(double_ratio["highdm"]["bins"]) == 5
    assert len(double_ratio["lowdm"]["bins"]) == 4

    measurement = tmp_path / "dy_measurement.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "autonomous_allhad.dy_estimation",
            "build-measurement",
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            "2024",
            "--output",
            str(measurement),
        ],
        check=True,
        cwd=PACKAGE_ROOT,
    )
    dy = json.loads(measurement.read_text())
    assert dy["status"] == "complete"
    assert dy["provenance"]["intermediate_root_reread"] is False
    assert dy["provenance"]["mass_windows"]["on"] == [71.0, 111.0]
    assert set(dy["rz_high"]["combined"]) == set(GROUPS)
    assert set(dy["rz_low"]["combined"]) == set(GROUPS)
