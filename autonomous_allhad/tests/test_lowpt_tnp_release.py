from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import correctionlib


REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "autonomous_allhad"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflow.build_tnp_das_records import build_records
from workflow.measure_lowpt_lepton import doctor, render, verify_release
from workflow.prepare_tnp_measurement_condor import prepare
from workflow.tnp_fit import fit_histogram_payload
from workflow.validate_tnp_adoption import FIT_VARIATIONS, validate


class LowPtTnpReleaseTest(unittest.TestCase):
    def _json(self, relative: str) -> dict:
        return json.loads((REPO / relative).read_text())

    def _gz_json(self, relative: str) -> dict:
        with gzip.open(REPO / relative, "rt") as source:
            return json.load(source)

    @staticmethod
    def _audit(config: dict, *, data: bool) -> dict:
        paths = list(config["reference_paths"])
        return {
            "files_processed": 1,
            "file_failures": [],
            "paths_present_by_file": {"audit.root": paths if data else []},
            "event_counts": {
                path: (100 if data else 0) for path in paths
            },
            "matched_trigger_objects": {},
            "individual_bit_index_counts_by_path": {},
            "created_unix": 1.0,
        }

    def test_doctor_and_config_semantics(self):
        result = doctor(REPO)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["local_histogram_fit_plot_ready"])
        for item in result["configs"].values():
            self.assertTrue(item["valid"])
            self.assertTrue(item["probe_definition"].endswith("_id_only"))

        electron = self._json(
            "autonomous_allhad/workflow/lowpt_electron_measurement/"
            "config_2025_id_only_parking_singlemuon.json"
        )
        muon = self._json(
            "autonomous_allhad/workflow/lowpt_muon_measurement/"
            "config_2025_id_only_parking_external.json"
        )
        for config in (electron, muon):
            self.assertFalse(config["tag_trigger_match_required"])
            self.assertFalse(config["apply_reference_trigger_to_mc"])
            self.assertEqual(
                config["reference_paths"],
                ["HLT_Mu9_Barrel_L1HP10_IP6", "HLT_Mu10_Barrel_L1HP11_IP6"],
            )
            self.assertEqual(
                config["systematic_variations"],
                [
                    "alternate_signal",
                    "alternate_background",
                    "pass_fail_shape_independent",
                    "mass_window_narrow",
                    "alternate_binning",
                    "pileup",
                ],
            )
            self.assertIn("not", config["target_selection"])
        self.assertEqual(
            electron["target_selection"],
            "cutBased >= 1; miniPFRelIso_all is not an additional probe pass/fail requirement",
        )
        self.assertEqual(muon["target_selection"], "looseId; miniPFRelIso_all is not part of the probe pass/fail definition")
        self.assertIsNone(muon["tag_miniiso_max"])
        self.assertEqual(muon["external_reference_muon"]["pt_min_gev"], 12.0)

    def test_frozen_records_are_exact_and_sample_pure(self):
        expected = {
            "electron_2025_data.json.gz": (37320, 0, "data"),
            "muon_2025_data.json.gz": (37320, 0, "data"),
            "electron_2024_mc.json.gz": (0, 52, "mc"),
            "muon_2024_mc.json.gz": (0, 671, "mc"),
        }
        for name, (data_files, mc_files, sample) in expected.items():
            payload = self._gz_json(f"autonomous_allhad/lowpt_tnp/records/{name}")
            self.assertEqual(payload["data_files"], data_files)
            self.assertEqual(payload["mc_files"], mc_files)
            self.assertEqual(len(payload["records"]), data_files + mc_files)
            self.assertEqual(payload["selected_samples"], [sample])
            self.assertEqual({item["sample"] for item in payload["records"]}, {sample})
            self.assertEqual(len({item["file_path"] for item in payload["records"]}), len(payload["records"]))
            self.assertEqual(payload["year"], "2025" if "2025" in name else "2024")
            self.assertTrue(payload["measurement"].endswith("parking_singlemuon") or payload["measurement"].endswith("parking_external"))
            self.assertIn("release_config_sha256", payload)

    @mock.patch(
        "workflow.prepare_tnp_measurement_condor._eos",
        side_effect=lambda path, _label: Path(path),
    )
    def test_frozen_records_pass_condor_preflight(self, _eos):
        campaigns = (
            ("electron", "electron_2025_data.json.gz", "config_2025_id_only_parking_singlemuon.json"),
            ("electron", "electron_2024_mc.json.gz", "config_2024_id_only_parking_singlemuon.json"),
            ("muon", "muon_2025_data.json.gz", "config_2025_id_only_parking_external.json"),
            ("muon", "muon_2024_mc.json.gz", "config_2024_id_only_parking_external.json"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            python_archive = root / "python38.tar.gz"
            runtime_archive = root / "repository.tar.gz"
            proxy = root / "x509up"
            for path in (python_archive, runtime_archive, proxy):
                path.touch()
            for index, (kind, records_name, config_name) in enumerate(campaigns):
                records = self._gz_json(f"autonomous_allhad/lowpt_tnp/records/{records_name}")
                records["records"] = records["records"][:2]
                records["data_files"] = sum(item["sample"] == "data" for item in records["records"])
                records["mc_files"] = sum(item["sample"] == "mc" for item in records["records"])
                records_path = root / f"records-{index}.json"
                records_path.write_text(json.dumps(records))
                config_path = REPO / f"autonomous_allhad/workflow/lowpt_{kind}_measurement/{config_name}"
                summary = prepare(
                    records_path=records_path,
                    repo=repository,
                    workdir=root / f"campaign-{index}",
                    python_archive=python_archive,
                    runtime_archive=runtime_archive,
                    proxy=proxy,
                    config=config_path,
                    kind=kind,
                    files_per_shard=20,
                )
                self.assertEqual(summary["records"], 2)
                self.assertEqual(summary["shards"], 1)
                self.assertEqual(summary["afs_reference_check"], "passed")
                wrapper = (root / f"campaign-{index}" / "run_tnp_measurement.sh").read_text()
                self.assertIn("measure_lowpt_lepton.py\" count --kind", wrapper)

    @mock.patch("workflow.build_tnp_das_records._das")
    def test_das_builder_can_freeze_data_and_mc_independently(self, das):
        das.side_effect = lambda query, **_: (
            ["/ParkingSingleMuon0/Run2025B-PromptReco-v1/NANOAOD"]
            if query.startswith("dataset=")
            else ["/store/test.root"]
        )
        config = {
            "measurement": "test",
            "year": "2025",
            "probe_definition": "veto_id_only",
            "tag_pt_min_gev": 5,
            "reference_paths": ["HLT_Test"],
            "campaign_inputs": {
                "data_dataset_query": "dataset=/ParkingSingleMuon*/Run2025*/NANOAOD",
                "mc_datasets": ["/MC/Test/NANOAODSIM"],
                "files_per_condor_shard": 20,
            },
        }
        data = build_records(config, samples={"data"})
        self.assertEqual((data["data_files"], data["mc_files"]), (1, 0))
        mc = build_records(config, samples={"mc"})
        self.assertEqual((mc["data_files"], mc["mc_files"]), (0, 1))

    def test_current_fit_variations_and_adoption_policy(self):
        self.assertEqual(
            FIT_VARIATIONS,
            (
                "nominal",
                "alternate_signal",
                "alternate_background",
                "pass_fail_shape_independent",
                "mass_window_narrow",
                "alternate_binning",
            ),
        )
        for kind, fallback in (("electron", True), ("muon", False)):
            result = self._json(
                f"autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results/{kind}/fit_result.json"
            )
            histograms = self._json(
                f"autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results/{kind}/histograms.json"
            )
            config_name = (
                "config_2025_id_only_parking_singlemuon.json"
                if kind == "electron"
                else "config_2025_id_only_parking_external.json"
            )
            config = self._json(
                f"autonomous_allhad/workflow/lowpt_{kind}_measurement/{config_name}"
            )
            checked = validate(
                result=result,
                histograms=histograms,
                config=config,
                data_trigger_audit=self._audit(config, data=True),
                mc_trigger_audit=self._audit(config, data=False),
                max_chi2_ndf=12.0,
                adopt_after_visual_review=False,
                visual_review_note=None,
                electron_endcap_unity_fallback=fallback,
            )
            if kind == "electron":
                self.assertEqual(
                    checked["status"],
                    "adoption_ready",
                    checked["validation"]["blockers"],
                )
                self.assertEqual(checked["validation"]["electron_endcap_unity_bins"], [4, 5])
            else:
                self.assertEqual(checked["status"], "validation_blocked")
                self.assertTrue(
                    any(
                        "alternate_signal/mc chi2/ndf" in blocker
                        for blocker in checked["validation"]["blockers"]
                    )
                )

    def test_payload_and_plot_reproduction(self):
        expected = {
            "electron": ("veto_electron_id_5to10_sf", 9, True),
            "muon": ("loose_muon_id_5to10_sf", 6, False),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, (correction_name, plots_per_format, fallback) in expected.items():
                inputs = REPO / "autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results" / kind
                output = root / kind
                manifest = render(
                    inputs / "fit_result.json",
                    inputs / "histograms.json",
                    output,
                    kind=kind,
                    electron_endcap_unity_fallback=fallback,
                    repo=REPO,
                )
                self.assertEqual(manifest["plots"]["style"]["standard_figure_inches"], [8.0, 8.0])
                self.assertEqual(manifest["plots"]["style"]["colorbar_figure_inches"], [12.0, 10.0])
                self.assertFalse(manifest["plots"]["style"]["titles"])
                self.assertEqual(
                    manifest["plots"]["style"]["electron_endcap_unity_fallback"],
                    fallback,
                )
                png = list((output / "plots").glob("*.png"))
                pdf = list((output / "plots").glob("*.pdf"))
                self.assertEqual((len(png), len(pdf)), (plots_per_format, plots_per_format))
                payload = correctionlib.CorrectionSet.from_file(manifest["payload"]["output"])
                self.assertEqual(list(payload.keys()), [correction_name])
                correction = payload[correction_name]
                eta = 2.0 if kind == "electron" else 1.8
                pt = 8.5 if kind == "electron" else 7.5
                values = [correction.evaluate(variation, eta, pt) for variation in ("nominal", "up", "down")]
                self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in values))
                self.assertGreaterEqual(values[1], values[0])
                self.assertGreaterEqual(values[0], values[2])
                if kind == "electron":
                    self.assertEqual(values[0], 1.0)
                    self.assertIn(
                        "released unity-central policy",
                        manifest["plots"]["captions"]["scale_factor"],
                    )

    def test_fit_numerical_reproduction(self):
        for kind in ("electron", "muon"):
            source = self._json(
                f"autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results/{kind}/histograms.json"
            )
            reference = self._json(
                f"autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results/{kind}/fit_result.json"
            )
            reproduced = fit_histogram_payload(source)
            self.assertEqual(reproduced["probe_definition"], reference["probe_definition"])
            for expected, actual in zip(reference["bins"], reproduced["bins"]):
                self.assertEqual(actual["valid"], expected["valid"])
                if not expected["valid"]:
                    continue
                self.assertTrue(
                    math.isclose(
                        actual["scale_factor"],
                        expected["scale_factor"],
                        rel_tol=5.0e-4,
                        abs_tol=1.0e-6,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        actual["scale_factor_uncertainty"],
                        expected["scale_factor_uncertainty"],
                        rel_tol=5.0e-4,
                        abs_tol=1.0e-6,
                    )
                )

    def test_release_hashes(self):
        manifest = REPO / "autonomous_allhad/lowpt_tnp/release.json"
        if not manifest.is_file():
            self.skipTest("release.json is generated after source assembly")
        checked = verify_release(REPO, manifest)
        self.assertEqual(checked["status"], "passed", checked["failures"])


if __name__ == "__main__":
    unittest.main()
