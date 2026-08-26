from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import correctionlib
import awkward as ak
import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[1]

from lowpt_tnp.build_tnp_das_records import build_records
from lowpt_tnp.cli import build_runtime_archive, doctor, render, verify_release
from lowpt_tnp.prepare_tnp_measurement_condor import prepare
from lowpt_tnp.tnp_fit import fit_histogram_payload
from lowpt_tnp.validate_tnp_adoption import FIT_VARIATIONS, validate


class LowPtTnpReleaseTest(unittest.TestCase):
    def _json(self, relative: str) -> dict:
        path = REPO / relative
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as source:
                return json.load(source)
        return json.loads(path.read_text())

    def _gz_json(self, relative: str) -> dict:
        with gzip.open(REPO / relative, "rt") as source:
            return json.load(source)

    @staticmethod
    def _write_synthetic_electron_nanoaod(path: Path) -> None:
        branches = {
            "run": np.array([391668], dtype=np.uint32),
            "luminosityBlock": np.array([151], dtype=np.uint32),
            "event": np.array([1], dtype=np.uint64),
            "HLT_Mu9_Barrel_L1HP10_IP6": np.array([True]),
            "Electron_pt": ak.Array([[6.2, 6.0]]),
            "Electron_eta": ak.Array([[0.0, 0.0]]),
            "Electron_phi": ak.Array([[0.0, 0.52]]),
            "Electron_mass": ak.Array([[0.000511, 0.000511]]),
            "Electron_charge": ak.Array([[1, -1]]),
            "Electron_miniPFRelIso_all": ak.Array([[0.05, 0.5]]),
            "Electron_cutBased": ak.Array([[4, 1]]),
            "Electron_convVeto": ak.Array([[True, True]]),
            "Electron_lostHits": ak.Array([[0, 0]]),
        }
        for flag in (
            "Flag_goodVertices",
            "Flag_globalSuperTightHalo2016Filter",
            "Flag_HBHENoiseFilter",
            "Flag_HBHENoiseIsoFilter",
            "Flag_EcalDeadCellTriggerPrimitiveFilter",
            "Flag_BadPFMuonFilter",
            "Flag_BadPFMuonDzFilter",
            "Flag_eeBadScFilter",
            "Flag_ecalBadCalibFilter",
        ):
            branches[flag] = np.array([True])
        with uproot.recreate(path) as output:
            output["Events"] = branches

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
            "configs/config_2025_id_only_parking_singlemuon.json"
        )
        muon = self._json(
            "configs/config_2025_id_only_parking_external.json"
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
            payload = self._gz_json(f"records/{name}")
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
        "lowpt_tnp.prepare_tnp_measurement_condor._eos",
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
            python_archive = root / "python38.tar.gz"
            runtime_archive = root / "repository.tar.gz"
            proxy = root / "x509up"
            for path in (python_archive, runtime_archive, proxy):
                path.touch()
            for index, (kind, records_name, config_name) in enumerate(campaigns):
                records = self._gz_json(f"records/{records_name}")
                records["records"] = records["records"][:2]
                records["data_files"] = sum(item["sample"] == "data" for item in records["records"])
                records["mc_files"] = sum(item["sample"] == "mc" for item in records["records"])
                records_path = root / f"records-{index}.json"
                records_path.write_text(json.dumps(records))
                config_path = REPO / "configs" / config_name
                summary = prepare(
                    records_path=records_path,
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
                self.assertIn("-m lowpt_tnp count --kind", wrapper)
                self.assertIn('PYTHONPATH="$WORKDIR/src"', wrapper)

    def test_runtime_archive_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runtime.tar.gz"
            result = build_runtime_archive(REPO, output)
            self.assertEqual(result["status"], "created")
            self.assertTrue(output.is_file())
            import tarfile
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("src/lowpt_tnp/cli.py", names)
            self.assertIn("data/pileup/puWeights_2024.json.gz", names)
            self.assertIn("configs/config_2025_id_only_parking_external.json", names)

    def test_extracted_runtime_executes_count_on_synthetic_nanoaod(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "runtime.tar.gz"
            build_runtime_archive(REPO, archive)
            extracted = root / "worker"
            extracted.mkdir()
            import tarfile
            with tarfile.open(archive, "r:gz") as source:
                if sys.version_info >= (3, 12):
                    source.extractall(extracted, filter="data")
                else:
                    source.extractall(extracted)
            nanoaod = extracted / "fixture.root"
            self._write_synthetic_electron_nanoaod(nanoaod)
            shard = extracted / "shard.json"
            shard.write_text(json.dumps({
                "shard_id": "synthetic",
                "records": [{
                    "file_path": str(nanoaod),
                    "dataset": "/Synthetic/Fixture/NANOAOD",
                    "sample": "data",
                }],
            }))
            result_path = extracted / "result.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(extracted / "src")
            command = [
                sys.executable,
                "-m",
                "lowpt_tnp",
                "count",
                "--kind",
                "electron",
                "--project-root",
                str(extracted),
                "--shard",
                str(shard),
                "--config",
                str(extracted / "configs/config_2025_id_only_parking_singlemuon.json"),
                "--output",
                str(result_path),
            ]
            completed = subprocess.run(
                command,
                cwd=extracted,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(result_path.read_text())
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["files_processed"], 1)
            self.assertGreater(payload["processing"]["data"]["pairs_selected"], 0)

    @mock.patch("lowpt_tnp.build_tnp_das_records._das")
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
                f"reference/results/{kind}/fit_result.json.gz"
            )
            histograms = self._json(
                f"reference/results/{kind}/histograms.json.gz"
            )
            config_name = (
                "config_2025_id_only_parking_singlemuon.json"
                if kind == "electron"
                else "config_2025_id_only_parking_external.json"
            )
            config = self._json(f"configs/{config_name}")
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
                inputs = REPO / "reference/results" / kind
                output = root / kind
                manifest = render(
                    inputs / "fit_result.json.gz",
                    inputs / "histograms.json.gz",
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
                f"reference/results/{kind}/histograms.json.gz"
            )
            reference = self._json(
                f"reference/results/{kind}/fit_result.json.gz"
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
        manifest = REPO / "release.json"
        if not manifest.is_file():
            self.skipTest("release.json is generated after source assembly")
        checked = verify_release(REPO, manifest)
        self.assertEqual(checked["status"], "passed", checked["failures"])

    def test_release_has_no_private_path_coupling(self):
        forbidden = (
            "/" + "Users" + "/",
            "/eos/user/t/" + "tai" + "woo",
            "/afs/cern.ch/" + "user/",
            "tai" + "woo",
            "autonomous_" + "allhad",
            "analysis/" + "data",
            "run3_stop/" + "decaf",
        )
        paths = [
            *REPO.glob("src/lowpt_tnp/*.py"),
            *REPO.glob("configs/*.json"),
            REPO / "README.md",
            REPO / "pyproject.toml",
            REPO / "environment.yml",
        ]
        for path in paths:
            text = path.read_text(errors="replace")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} leaked into {path}")
        for path in REPO.glob("records/*.json.gz"):
            with gzip.open(path, "rt") as source:
                text = source.read()
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} leaked into {path}")
        for path in REPO.glob("reference/results/*/*.json.gz"):
            with gzip.open(path, "rt") as source:
                text = source.read()
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} leaked into {path}")


if __name__ == "__main__":
    unittest.main()
