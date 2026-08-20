import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import awkward as ak
import correctionlib
import numpy as np

from autonomous_allhad.analysis_scale_factors import (
    AnalysisScaleFactorUnavailable,
    loose_muon_lowpt_triplet,
    met_trigger_triplet,
    photon_trigger_triplet,
    veto_electron_lowpt_triplet,
)
from workflow.sf_payload import correction, correction_set, install_adopted_result, write_json_gz


class AnalysisScaleFactorTest(unittest.TestCase):
    @staticmethod
    def _install_constant_payloads(repo: Path, year: str = "2024") -> Path:
        target = repo / "analysis/data/AnalysisSF" / year
        target.mkdir(parents=True)
        write_json_gz(
            target / "met_trigger_sf.json.gz",
            correction_set(
                "met",
                [
                    correction(name="met_trigger_sf_genuine", description="genuine", axes=[("met", [100, 300, 800])], nominal=[0.9, 1.0], uncertainty=[0.1, 0.02]),
                    correction(name="met_trigger_sf_qcd", description="legacy sentinel", axes=[("met", [100, 300, 800])], nominal=[0.7, 0.8], uncertainty=[0.1, 0.02]),
                ],
            ),
        )
        write_json_gz(
            target / "photon_trigger_sf.json.gz",
            correction_set("photon", [correction(name="photon_trigger_sf", description="ph", axes=[("abseta", [0, 2.5]), ("pt", [220, 1000])], nominal=[0.98], uncertainty=[0.01])]),
        )
        write_json_gz(
            target / "veto_electron_5to10_sf.json.gz",
            correction_set("electron", [correction(name="veto_electron_id_5to10_sf", description="e ID only", axes=[("abseta", [0, 2.5]), ("pt", [5, 10])], nominal=[1.05], uncertainty=[0.03])]),
        )
        write_json_gz(
            target / "loose_muon_5to10_sf.json.gz",
            correction_set("muon", [correction(name="loose_muon_id_5to10_sf", description="m ID only", axes=[("abseta", [0, 2.4]), ("pt", [5, 10])], nominal=[0.97], uncertainty=[0.02])]),
        )
        return target

    def test_correctionlib_roundtrip_and_variations(self):
        payload = correction_set(
            "test",
            [
                correction(
                    name="test_sf",
                    description="test",
                    axes=[("abseta", [0.0, 1.0, 2.5]), ("pt", [5.0, 7.0, 10.0])],
                    nominal=[0.9, 1.0, 1.1, 1.2],
                    uncertainty=[0.1, 0.2, 0.05, 0.1],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.gz"
            digest = write_json_gz(path, payload)
            self.assertEqual(len(digest), 64)
            evaluator = correctionlib.CorrectionSet.from_file(str(path))["test_sf"]
            self.assertAlmostEqual(evaluator.evaluate("nominal", 0.5, 6.0), 0.9)
            self.assertAlmostEqual(evaluator.evaluate("up", 0.5, 6.0), 1.0)
            self.assertAlmostEqual(evaluator.evaluate("down", 0.5, 6.0), 0.8)

    def test_install_refuses_preliminary_result(self):
        payload = correction_set(
            "test",
            [correction(name="x", description="x", axes=[("pt", [1, 2])], nominal=[1], uncertainty=[0.1])],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result.json"
            result.write_text(json.dumps({"status": "validation_pending"}))
            with self.assertRaises(RuntimeError):
                install_adopted_result(result, root / "x.json.gz", payload)

    def test_analysis_helpers_evaluate_installed_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            self.assertTrue(np.allclose(met_trigger_triplet(repo, [250], qcd=False)[0], [0.9]))
            self.assertTrue(np.allclose(met_trigger_triplet(repo, [250], qcd=True)[0], [0.9]))
            self.assertAlmostEqual(float(photon_trigger_triplet(repo, ak.Array([[0.2]]), ak.Array([[300.0]]))[0][0][0]), 0.98)
            self.assertAlmostEqual(float(veto_electron_lowpt_triplet(repo, ak.Array([[0.2]]), ak.Array([[7.0]]))[0][0][0]), 1.05)
            self.assertAlmostEqual(float(loose_muon_lowpt_triplet(repo, ak.Array([[0.2]]), ak.Array([[7.0]]))[0][0][0]), 0.97)

    def test_missing_payload_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AnalysisScaleFactorUnavailable):
                met_trigger_triplet(Path(tmp), [300.0], qcd=False)

    def test_2025_helpers_select_2025_payload_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo, year="2025")
            self.assertTrue(
                np.allclose(
                    met_trigger_triplet(repo, [250], qcd=False, year="2025")[0],
                    [0.9],
                )
            )
            self.assertAlmostEqual(
                float(
                    veto_electron_lowpt_triplet(
                        repo,
                        ak.Array([[0.2]]),
                        ak.Array([[7.0]]),
                        year="2025",
                    )[0][0][0]
                ),
                1.05,
            )

    def test_installed_lowpt_payloads_are_id_only(self):
        repo = Path(__file__).resolve().parents[2]
        electron = correctionlib.CorrectionSet.from_file(
            str(repo / "analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz")
        )
        muon = correctionlib.CorrectionSet.from_file(
            str(repo / "analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz")
        )
        self.assertIn("veto_electron_id_5to10_sf", electron)
        self.assertIn("loose_muon_id_5to10_sf", muon)
        self.assertIn("excluding isolation", electron["veto_electron_id_5to10_sf"].description)
        self.assertIn("excluding isolation", muon["loose_muon_id_5to10_sf"].description)

    def test_adopted_lowpt_payloads_reach_histogram_weight_variations(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        repo = Path(__file__).resolve().parents[2]
        ones = lambda values: ak.ones_like(values, dtype=float)
        corrections = {
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_mu_loose_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_medium_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_hlt_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
        }
        empty_float = ak.Array([[], []])
        empty_bool = ak.Array([[], []])
        with mock.patch(
            "autonomous_allhad.real_subset_worker.load_analysis_corrections",
            return_value=corrections,
        ):
            _generator, variations, status = compute_weight_bundle(
                {"genWeight": np.ones(2)},
                repo,
                "WJetsToLNu",
                "WJets",
                "2024",
                2,
                empty_float,
                empty_float,
                empty_float,
                empty_bool,
                ak.Array([[0.4], []]),
                ak.Array([[0.0], []]),
                ak.Array([[7.5], []]),
                ak.Array([[0.0], []]),
                ak.Array([[True], []]),
                ak.Array([[False], []]),
                np.asarray([1, 0]),
                np.asarray([0, 0]),
                ak.Array([[], [1.0]]),
                ak.Array([[], [7.5]]),
                ak.Array([[], [0.0]]),
                ak.Array([[], [True]]),
                ak.Array([[], [False]]),
                np.asarray([0, 1]),
                np.asarray([0, 0]),
                empty_float,
                empty_float,
                empty_float,
                empty_bool,
                np.asarray([False, False]),
                met_pt=np.asarray([300.0, 300.0]),
                met_trigger_mask=np.asarray([False, False]),
            )
        electron = status["components"]["veto_electron_5to10"]
        muon = status["components"]["loose_muon_5to10"]
        self.assertTrue(electron["applied"])
        self.assertTrue(muon["applied"])
        for name in (
            "veto_electron_5to10Up",
            "veto_electron_5to10Down",
            "loose_muon_5to10Up",
            "loose_muon_5to10Down",
        ):
            self.assertIn(name, variations)
        self.assertNotEqual(variations["veto_electron_5to10Up"][0], variations["nominal"][0])
        self.assertNotEqual(variations["veto_electron_5to10Down"][0], variations["nominal"][0])
        self.assertNotEqual(variations["loose_muon_5to10Up"][1], variations["nominal"][1])
        self.assertNotEqual(variations["loose_muon_5to10Down"][1], variations["nominal"][1])

    def test_lowpt_edge_clipping_is_replaced_and_all_analysis_sf_shapes_fill(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
            from workflow.build_flat_boosted_recoil_hists import (
                add_hist,
                empty_hist,
                histogram_variations,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        # Probe both open edges of the measured interval and the exact handoff
        # point.  7.5 GeV is not a threshold; the production domain is 5<pT<10.
        n = 8
        empty_float = ak.Array([[] for _ in range(n)])
        empty_bool = ak.Array([[] for _ in range(n)])

        def triplet(pt, nominal, up, down):
            return tuple(ak.ones_like(pt, dtype=float) * value for value in (nominal, up, down))

        corrections = {
            # Distinct non-unity edge values prove that the old 10 GeV clamp is
            # removed below 10 GeV instead of being multiplied by the new SF.
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: triplet(pt, 1.4, 1.5, 1.3),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
            "get_mu_loose_id_sf": lambda year, eta, pt: triplet(pt, 1.3, 1.4, 1.2),
            "get_mu_medium_id_sf": lambda year, eta, pt: triplet(pt, 1.0, 1.0, 1.0),
            "get_mu_hlt_sf": lambda year, eta, pt: triplet(pt, 1.0, 1.0, 1.0),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: triplet(pt, 1.0, 1.0, 1.0),
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            with mock.patch(
                "autonomous_allhad.real_subset_worker.load_analysis_corrections",
                return_value=corrections,
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.analysis_workdir",
                side_effect=lambda repo: nullcontext(),
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.veto_electron_lowpt_triplet",
                wraps=veto_electron_lowpt_triplet,
            ) as electron_lowpt_evaluator:
                _generator, variations, status = compute_weight_bundle(
                    {"genWeight": np.ones(n)},
                    repo,
                    "WJetsToLNu",
                    "WJets",
                    "2024",
                    n,
                    empty_float,
                    empty_float,
                    empty_float,
                    empty_bool,
                    ak.Array([[0.4], [0.4], [0.4], [], [], [], [], []]),
                    ak.Array([[0.2], [0.2], [0.2], [], [], [], [], []]),
                    ak.Array([[5.001], [9.999], [10.0], [], [], [], [], []]),
                    ak.Array([[0.0], [0.0], [0.0], [], [], [], [], []]),
                    ak.Array([[True], [True], [True], [], [], [], [], []]),
                    ak.Array([[False], [False], [False], [], [], [], [], []]),
                    np.asarray([1, 1, 1, 0, 0, 0, 0, 0]),
                    np.zeros(n, dtype=int),
                    ak.Array([[], [], [], [0.5], [0.5], [0.5], [], []]),
                    ak.Array([[], [], [], [5.001], [9.999], [10.0], [], []]),
                    ak.Array([[], [], [], [0.0], [0.0], [0.0], [], []]),
                    ak.Array([[], [], [], [True], [True], [True], [], []]),
                    ak.Array([[], [], [], [False], [False], [False], [], []]),
                    np.asarray([0, 0, 0, 1, 1, 1, 0, 0]),
                    np.zeros(n, dtype=int),
                    ak.Array([[], [], [], [], [], [], [0.2], []]),
                    ak.Array([[], [], [], [], [], [], [300.0], []]),
                    ak.Array([[], [], [], [], [], [], [0.0], []]),
                    ak.Array([[], [], [], [], [], [], [True], []]),
                    np.asarray([False, False, False, False, False, False, True, False]),
                    met_pt=np.asarray([250.0] * n),
                    met_trigger_mask=np.asarray([False, False, False, False, False, False, False, True]),
                )

        # The adopted low-pT measurement is parameterized in reconstructed
        # Electron_eta, while the official EGM ID payload uses etaSC.
        lowpt_eta = electron_lowpt_evaluator.call_args.args[1]
        np.testing.assert_allclose(
            np.asarray(ak.to_list(lowpt_eta[:3]), dtype=float).reshape(-1),
            np.asarray([0.4, 0.4, 0.4]),
        )

        self.assertTrue(
            status["components"]["electron_id"]["applied"],
            status["components"]["electron_id"],
        )
        self.assertTrue(
            status["components"]["muon_id"]["applied"],
            status["components"]["muon_id"],
        )
        np.testing.assert_allclose(
            variations["nominal"],
            np.asarray([1.05, 1.05, 1.4, 0.97, 0.97, 1.3, 0.98, 0.9]),
            rtol=0.0,
            atol=1.0e-12,
        )
        expected_shapes = (
            "veto_electron_5to10Up",
            "veto_electron_5to10Down",
            "loose_muon_5to10Up",
            "loose_muon_5to10Down",
            "photon_triggerUp",
            "photon_triggerDown",
            "met_triggerUp",
            "met_triggerDown",
        )
        histogram_weights = histogram_variations(variations, nominal_only=False)
        for name in expected_shapes:
            self.assertIn(name, histogram_weights)
        for component, event_index in (
            ("veto_electron_5to10", 0),
            ("veto_electron_5to10", 1),
            ("loose_muon_5to10", 3),
            ("loose_muon_5to10", 4),
            ("photon_trigger", 6),
            ("met_trigger", 7),
        ):
            for direction in ("Up", "Down"):
                name = f"{component}{direction}"
                self.assertNotEqual(
                    histogram_weights[name][event_index],
                    variations["nominal"][event_index],
                )

        # Exercise the same leaf filler used by the production histogrammer.
        for name in ("nominal", *expected_shapes):
            leaf = empty_hist()
            add_hist(
                leaf,
                np.full(n, 350.0),
                histogram_weights[name],
                np.ones(n, dtype=bool),
            )
            self.assertAlmostEqual(sum(leaf["sumw"]), float(np.sum(histogram_weights[name])))

        for component in (
            "met_trigger",
            "photon_trigger",
            "veto_electron_5to10",
            "loose_muon_5to10",
        ):
            self.assertTrue(status["components"][component]["applied"])

    def test_trigger_only_configuration_excludes_lowpt_lepton_nominal_and_shapes(self):
        try:
            from autonomous_allhad.real_subset_worker import compute_weight_bundle
        except ImportError as exc:
            raise unittest.SkipTest(f"full analysis runtime unavailable: {exc}") from exc

        n = 4
        empty_float = ak.Array([[] for _ in range(n)])
        empty_bool = ak.Array([[] for _ in range(n)])
        ones = lambda values: ak.ones_like(values, dtype=float)
        corrections = {
            "get_ele_veto_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_medium_id_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_ele_hlt_sf": lambda year, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
            "get_mu_loose_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_medium_id_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_mu_hlt_sf": lambda year, eta, pt: (ones(pt), ones(pt), ones(pt)),
            "get_photon_id_sf": lambda year, wp, eta, pt, phi: (ones(pt), ones(pt), ones(pt)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._install_constant_payloads(repo)
            with mock.patch(
                "autonomous_allhad.real_subset_worker.load_analysis_corrections",
                return_value=corrections,
            ), mock.patch(
                "autonomous_allhad.real_subset_worker.analysis_workdir",
                side_effect=lambda repo: nullcontext(),
            ):
                _generator, variations, status = compute_weight_bundle(
                    {"genWeight": np.ones(n)},
                    repo, "WJetsToLNu", "WJets", "2024", n,
                    empty_float, empty_float, empty_float, empty_bool,
                    ak.Array([[0.4], [], [], []]),
                    ak.Array([[0.0], [], [], []]),
                    ak.Array([[7.5], [], [], []]),
                    ak.Array([[0.0], [], [], []]),
                    ak.Array([[True], [], [], []]),
                    ak.Array([[False], [], [], []]),
                    np.asarray([1, 0, 0, 0]), np.zeros(n, dtype=int),
                    ak.Array([[], [0.5], [], []]),
                    ak.Array([[], [7.5], [], []]),
                    ak.Array([[], [0.0], [], []]),
                    ak.Array([[], [True], [], []]),
                    ak.Array([[], [False], [], []]),
                    np.asarray([0, 1, 0, 0]), np.zeros(n, dtype=int),
                    ak.Array([[], [], [0.2], []]),
                    ak.Array([[], [], [300.0], []]),
                    ak.Array([[], [], [0.0], []]),
                    ak.Array([[], [], [True], []]),
                    np.asarray([False, False, True, False]),
                    met_pt=np.asarray([250.0] * n),
                    met_trigger_mask=np.asarray([False, False, False, True]),
                    analysis_sf_components=("met_trigger", "photon_trigger"),
                )

        np.testing.assert_allclose(
            variations["nominal"],
            np.asarray([1.0, 1.0, 0.98, 0.9]),
            rtol=0.0,
            atol=1.0e-12,
        )
        for component in ("veto_electron_5to10", "loose_muon_5to10"):
            self.assertFalse(status["components"][component]["applied"])
            self.assertEqual(
                status["components"][component]["source"],
                "disabled_by_analysis_sf_configuration",
            )
            self.assertNotIn(f"{component}Up", variations)
            self.assertNotIn(f"{component}Down", variations)
        for component in ("met_trigger", "photon_trigger"):
            self.assertTrue(status["components"][component]["applied"])
            self.assertIn(f"{component}Up", variations)
            self.assertIn(f"{component}Down", variations)


if __name__ == "__main__":
    unittest.main()
