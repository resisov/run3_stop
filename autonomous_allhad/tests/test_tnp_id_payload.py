import tempfile
import unittest
from pathlib import Path

from workflow.export_tnp_id_correctionlib import build_payload
from workflow.sf_payload import write_json_gz


class TnpIdPayloadTest(unittest.TestCase):
    def result(self, definition):
        return {
            "probe_definition": definition,
            "probe_abseta_edges": [0.0, 2.4],
            "probe_pt_edges_gev": [5.0, 10.0],
            "bins": [{
                "valid": True,
                "scale_factor": 0.98,
                "scale_factor_uncertainty": 0.02,
            }],
        }

    def test_electron_id_payload(self):
        payload = build_payload(self.result("veto_id_only"), "electron")
        self.assertEqual(payload["corrections"][0]["name"], "veto_electron_id_5to10_sf")
        with tempfile.TemporaryDirectory() as directory:
            digest = write_json_gz(Path(directory) / "electron.json.gz", payload)
        self.assertEqual(len(digest), 64)

    def test_electron_wider_validation_range_is_trimmed_to_5to10(self):
        result = self.result("veto_id_only")
        result["probe_pt_edges_gev"] = [5.0, 7.0, 10.0, 15.0]
        result["bins"] = [
            {"valid": True, "scale_factor": value, "scale_factor_uncertainty": 0.02}
            for value in (0.91, 0.92, 9.99)
        ]
        correction = build_payload(result, "electron")["corrections"][0]
        nominal = correction["data"]["content"][0]["value"]
        self.assertEqual(nominal["edges"][1], [5.0, 7.0, 10.0])
        self.assertEqual(nominal["content"], [0.91, 0.92])

    def test_muon_id_payload(self):
        payload = build_payload(self.result("loose_id_only"), "muon")
        self.assertEqual(payload["corrections"][0]["name"], "loose_muon_id_5to10_sf")

    def test_definition_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_payload(self.result("miniiso_given_loose_id"), "muon")

    def test_electron_endcap_unity_fallback(self):
        result = {
            "probe_definition": "veto_id_only",
            "probe_abseta_edges": [0.0, 1.44, 2.5],
            "probe_pt_edges_gev": [5.0, 7.0, 10.0],
            "bins": [
                {"valid": True, "scale_factor": 0.91, "scale_factor_uncertainty": 0.02},
                {"valid": True, "scale_factor": 0.93, "scale_factor_uncertainty": 0.03},
                self.invalid_fit_bin(0.96, 0.08, 0.80, 0.04),
                self.invalid_fit_bin(0.90, 0.03, 0.90, 0.02),
            ],
        }
        correction = build_payload(
            result,
            "electron",
            electron_endcap_unity_fallback=True,
        )["corrections"][0]
        nominal = correction["data"]["content"][0]["value"]["content"]
        up = correction["data"]["content"][1]["value"]["content"]
        down = correction["data"]["content"][2]["value"]["content"]
        self.assertEqual(nominal[:2], [0.91, 0.93])
        self.assertEqual(nominal[2:], [1.0, 1.0])
        self.assertAlmostEqual(up[2] - 1.0, 1.0 - down[2])
        self.assertAlmostEqual(up[3] - 1.0, 1.0 - down[3])
        self.assertIn("unity central value", correction["description"])

    def test_endcap_unity_fallback_does_not_hide_invalid_barrel_bin(self):
        result = {
            "probe_definition": "veto_id_only",
            "probe_abseta_edges": [0.0, 1.44, 2.5],
            "probe_pt_edges_gev": [5.0, 10.0],
            "bins": [
                self.invalid_fit_bin(0.8, 0.1, 0.9, 0.1),
                self.invalid_fit_bin(0.8, 0.1, 0.9, 0.1),
            ],
        }
        with self.assertRaisesRegex(ValueError, "invalid barrel"):
            build_payload(result, "electron", electron_endcap_unity_fallback=True)

    def test_endcap_unity_fallback_is_electron_only(self):
        with self.assertRaisesRegex(ValueError, "only for electron"):
            build_payload(
                self.result("loose_id_only"),
                "muon",
                electron_endcap_unity_fallback=True,
            )

    @staticmethod
    def invalid_fit_bin(data_efficiency, data_uncertainty, mc_efficiency, mc_uncertainty):
        return {
            "valid": False,
            "fits": {
                "nominal": {
                    "data": {
                        "efficiency": data_efficiency,
                        "efficiency_stat_uncertainty": data_uncertainty,
                    },
                    "mc": {
                        "efficiency": mc_efficiency,
                        "efficiency_stat_uncertainty": mc_uncertainty,
                    },
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
