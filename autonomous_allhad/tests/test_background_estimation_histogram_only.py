from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_tf_overlay_keeps_each_category_edges_values_and_errors(monkeypatch, tmp_path):
    import importlib
    import numpy as np

    monkeypatch.syspath_prepend(str(WORKFLOW))
    plotter = importlib.import_module("plot_recoil_transfer_factors_2024")
    config = json.loads((PACKAGE_ROOT / "gnn_lowdm/config.json").read_text())
    edges = {c: np.asarray(e) for c, e in config["sr_binning"]["edges_by_category"].items()}
    records = {c: {"transfer_factor": [1, 2, 3, 4, 5], "mcstat": [0.1] * 5} for c in edges}
    captured = []
    original = plotter.plt.Axes.errorbar

    def capture(axis, x, y, **kwargs):
        captured.append((np.asarray(x), np.asarray(y), kwargs))
        return original(axis, x, y, **kwargs)

    def close(fig, stem):
        assert fig.axes[0].get_xlim() == (0, 1)
        plotter.plt.close(fig)
        return []

    monkeypatch.setattr(plotter.plt.Axes, "errorbar", capture)
    monkeypatch.setattr(plotter, "save_figure", close)
    plotter.plot_highdm(tmp_path, plotter.GNN_PATHS[0], edges, records,
                       regime="lowdm", styles=plotter.GNN_STYLES)
    assert len(captured) == 6
    for category, (x, y, kw) in zip(plotter.GNN_STYLES, captured):
        np.testing.assert_allclose(x, (edges[category][1:] + edges[category][:-1]) / 2)
        np.testing.assert_allclose(kw["xerr"], np.diff(edges[category]) / 2)
        np.testing.assert_array_equal(y, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(kw["yerr"], [0.1] * 5)
        marker, color, label = plotter.GNN_STYLES[category]
        assert (kw["fmt"], kw["color"], kw["label"]) == (marker, color, label)


def test_mll_plot_keeps_large_z_peak_visible(monkeypatch, tmp_path):
    from autonomous_allhad.dy_estimation import report

    source = {"DY2E": {"Nb1": {
        "data": {"edges": [50, 71, 111, 160], "sumw": [100, 20000, 100], "sumw2": [100, 20000, 100]},
        "zll": {"sumw": [20, 18000, 20], "sumw2": [20, 18000, 20]},
        "other": {"sumw": [80, 2000, 80], "sumw2": [80, 2000, 80]},
    }}}
    limits = []

    def capture(fig, path):
        limits.append(fig.axes[0].get_ylim())
        report.plt.close(fig)
        return []

    monkeypatch.setattr(report, "save_figure", capture)
    report.plot_mll(source, {}, "lowdm", tmp_path, corrected=False)
    assert len(limits) == 1
    assert limits[0][0] == 0.1
    assert limits[0][1] > 60000


def leaf(values: list[float]) -> dict[str, object]:
    return {
        "nominal": {
            "sumw": values,
            "sumw2": [abs(value) for value in values],
            "entries": [30 if value > 0.0 else 0 for value in values],
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
                    sample: leaf([0.0 if sample == "data_obs" else 1.0] * 8) for sample in SAMPLES
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


@pytest.mark.parametrize("corruption", [None, "threshold", "operator", "domain", "sf", "status"])
def test_audit_uses_canonical_selection_policy(monkeypatch, corruption):
    monkeypatch.syspath_prepend(str(WORKFLOW))
    from audit_background_histogram_products import input_policy
    manifest = {"status": "canonical", "electron_veto_pt_min_gev": 10,
                "muon_veto_pt_min_gev": 10, "lepton_threshold_operator": ">",
                "lowdm_double_ratio_min_gev": 250,
                "excluded_sf_components": ["veto_electron_5to10", "loose_muon_5to10"]}
    if corruption is None:
        assert input_policy(manifest)["status"] == "validated_10gev_inputs"
        assert input_policy({}, historical=True)["status"] == "blocked"
        return
    updates = {"threshold": ("muon_veto_pt_min_gev", 5), "operator": ("lepton_threshold_operator", ">="),
               "domain": ("lowdm_double_ratio_min_gev", 300), "sf": ("excluded_sf_components", []),
               "status": ("status", "incomplete")}
    key, value = updates[corruption]
    manifest[key] = value
    with pytest.raises(ValueError):
        input_policy(manifest)


def test_tf_builder_rejects_missing_gnn_inputs(tmp_path: Path) -> None:
    hist_input = tmp_path / "nominal_background_estimation.json"
    hist_input.write_text(json.dumps(compact_input()))
    tf_output = tmp_path / "tf.json"
    result = subprocess.run(
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
        capture_output=True, text=True,
        cwd=WORKFLOW,
    )
    assert result.returncode != 0
    assert "--gnn-input" in result.stderr
    assert not tf_output.exists()


@pytest.mark.parametrize("year", ["2024", "2025"])
def test_histogram_boundary_drives_tf_and_rz(tmp_path: Path, year: str) -> None:
    hist_input = tmp_path / "nominal_background_estimation.json"
    compact = compact_input()
    compact["summary"]["datasets"] = {name.replace("2024", year): value for name, value in compact["summary"]["datasets"].items()}
    hist_input.write_text(json.dumps(compact))
    tf_output = tmp_path / "tf.json"

    sgamma_dir = tmp_path / "sgamma"
    subprocess.run(
        [
            sys.executable,
            str(WORKFLOW / "build_sgamma_ut_report_2024.py"),
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            year,
            "--output-dir",
            str(sgamma_dir),
            "--no-plots",
        ],
        check=True,
        cwd=WORKFLOW,
    )
    sgamma = json.loads((sgamma_dir / "sgamma_ut.json").read_text())
    assert sgamma["status"] == "complete"
    assert set(sgamma["highdm"]) == set(GROUPS)
    assert set(sgamma["lowdm_families"]) == set(GROUPS)
    assert sgamma["provenance"]["intermediate_root_reread"] is False
    assert not sgamma["plots"]
    original_sgamma = (sgamma_dir / "sgamma_ut.json").read_bytes()
    subprocess.run([sys.executable, str(WORKFLOW / "build_sgamma_ut_report_2024.py"),
                    "--hist-input", str(hist_input), "--campaign-year", year,
                    "--output-dir", str(sgamma_dir), "--plot-only"], check=True, cwd=WORKFLOW)
    assert (sgamma_dir / "sgamma_ut.json").read_bytes() == original_sgamma
    assert len(json.loads((sgamma_dir / "plot_manifest.json").read_text())["plots"]) == 6

    double_ratio_dir = tmp_path / "double_ratio"
    subprocess.run(
        [
            sys.executable,
            str(WORKFLOW / "build_zgamma_double_ratio_2024.py"),
            "--hist-input",
            str(hist_input),
            "--campaign-year",
            year,
            "--output-dir",
            str(double_ratio_dir),
            "--no-plots",
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
    assert len(double_ratio["lowdm"]["bins"]) == 5
    assert double_ratio["lowdm"]["edges"] == [250, 300, 350, 400, 500, 1500]
    assert double_ratio["adoption_status"] == "adopted"
    assert not double_ratio["plots"]
    original_double_ratio = (double_ratio_dir / "zgamma_double_ratio.json").read_bytes()
    subprocess.run([sys.executable, str(WORKFLOW / "build_zgamma_double_ratio_2024.py"),
                    "--hist-input", str(hist_input), "--campaign-year", year,
                    "--output-dir", str(double_ratio_dir), "--plot-only"], check=True, cwd=WORKFLOW)
    assert (double_ratio_dir / "zgamma_double_ratio.json").read_bytes() == original_double_ratio
    assert len(json.loads((double_ratio_dir / "plot_manifest.json").read_text())["plots"]) == 4

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
            year,
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

    config_path = PACKAGE_ROOT / "gnn_lowdm/config.json"
    config = json.loads(config_path.read_text())
    histograms = {}
    for region in REGIONS:
        labels = config["sr_binning" if region == "SR" else "cr_binning"]["category_labels"]
        histograms[region] = {}
        for category in labels:
            group = category.split("_")[0]
            multiplicity = sum(label.startswith(group + "_") for label in labels)
            histograms[region][category] = {}
            for sample, values in compact["lowdm"]["recoil"][region][group].items():
                if region == "SR" and sample == "data_obs":
                    continue
                joint = {field: [[value / multiplicity / 5 for value in array] for _ in range(5)]
                         for field, array in values["nominal"].items()}
                histograms[region][category][sample] = {
                    "gnn_score_ut": joint,
                    "gnn_score": {field: [sum(row) for row in matrix] for field, matrix in joint.items()},
                }
    gnn_input = tmp_path / "gnn.json"
    gnn_input.write_text(json.dumps({"histograms": {"nominal": histograms}}, indent=2))
    tf_command = [
        sys.executable, str(WORKFLOW / "build_histogram_tf_inputs_2024.py"),
        "--hist-input", str(hist_input), "--campaign-year", year, "--output", str(tf_output),
        "--gnn-input", str(gnn_input), "--gnn-config", str(config_path),
        "--sgamma-input", str(sgamma_dir / "sgamma_ut.json"), "--dy-measurement", str(measurement),
    ]
    subprocess.run(tf_command, check=True, capture_output=True, text=True)
    tf = json.loads(tf_output.read_text())
    assert tf["status"] == "complete"
    assert tf["provenance"]["intermediate_root_reread"] is False
    assert tf["highdm"]["nb_groups"] == list(GROUPS)
    assert tf["lowdm"]["kind"] == "gnn" and "recoil" not in tf["lowdm"]
    assert tf["diagnostics"]["lowdm_ut"]["template_eligible"] is False
    assert tf["lowdm_gnn"]["template_contract"]["ut_fallback_allowed"] is False
    assert set(tf["lowdm_gnn"]["transfer_factors"]["nominal"]) == {"top_llcr", "w_llcr", "qcd_qcdcr", "zinv_gcr"}
    for samples in tf["lowdm_gnn"]["histograms"]["nominal"]["SR"].values():
        assert set(samples) == set(SAMPLES) - {"data_obs"}

    plots = tmp_path / "tf_plots"
    subprocess.run([
        sys.executable, str(WORKFLOW / "plot_recoil_transfer_factors_2024.py"),
        "--input", str(tf_output), "--campaign-year", year,
        "--regime", "lowdm", "--output-dir", str(plots),
    ], check=True, capture_output=True, text=True)
    plotted = json.loads((plots / f"transfer_factors_{year}_nb_recoil.json").read_text())
    assert plotted["factors"]["lowdm"]["kind"] == "gnn"
    assert plotted["factors"]["lowdm"]["records"] == tf["lowdm_gnn"]["transfer_factors"]["nominal"]
    assert len(plotted["plots"]) == 4 * 2
    assert plotted["provenance"]["lowdm_plot_layout"] == "categories_overlaid"
    assert all(Path(path).parent == plots / "gnn" for path in plotted["plots"])
    stored_factors = plots / f"transfer_factors_{year}_nb_recoil.json"
    stored_bytes = stored_factors.read_bytes()
    subprocess.run([
        sys.executable, str(WORKFLOW / "plot_recoil_transfer_factors_2024.py"),
        "--input", str(stored_factors), "--campaign-year", year, "--plot-only",
        "--regime", "lowdm", "--output-dir", str(plots / "replot"),
    ], check=True, capture_output=True, text=True)
    assert stored_factors.read_bytes() == stored_bytes
    receipt = json.loads((plots / "replot/plot_manifest.json").read_text())
    assert receipt["factor_values_changed"] is False
    assert len(receipt["plots"]) == 8
    for groups in plotted["factors"]["highdm"]["records"].values():
        for record in groups.values():
            assert record["transfer_factor"] == [1.0] * 8

    # A factor derived from a different selection/hash must never replace the
    # successfully built template (e.g. mixing 5-GeV and 10-GeV veto products).
    original = tf_output.read_bytes()
    sgamma["provenance"]["hist_input_sha256"] = "wrong-selection-hash"
    (sgamma_dir / "sgamma_ut.json").write_text(json.dumps(sgamma))
    rejected = subprocess.run(tf_command, capture_output=True, text=True)
    assert rejected.returncode != 0 and "different histogram inputs" in rejected.stderr
    assert tf_output.read_bytes() == original
