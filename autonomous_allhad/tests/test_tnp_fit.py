import unittest

import numpy as np

from workflow.tnp_fit import fit_histogram_payload


def spectrum(edges, signal_yield, efficiency, background_pass, background_fail):
    centres = 0.5 * (edges[:-1] + edges[1:])
    signal = np.exp(-0.5 * ((centres - 3.1) / 0.045) ** 2)
    signal /= np.sum(signal)
    background = np.exp(-0.4 * (centres - np.mean(centres)))
    background /= np.sum(background)
    passed = signal_yield * efficiency * signal + background_pass * background
    failed = signal_yield * (1.0 - efficiency) * signal + background_fail * background
    return passed, failed


class TnpFitTest(unittest.TestCase):
    def test_simultaneous_pass_fail_recovers_scale_factor(self):
        edges = np.linspace(2.5, 3.7, 61)
        data_pass, data_fail = spectrum(edges, 30000, 0.82, 3000, 4000)
        mc_pass, mc_fail = spectrum(edges, 50000, 0.78, 2500, 3500)
        payload = {
            "measurement": "test",
            "mass_edges_gev": edges.tolist(),
            "fit_window_gev": [2.6, 3.6],
            "probe_abseta_edges": [0.0, 2.4],
            "probe_pt_edges_gev": [5.0, 10.0],
            "samples": {
                "data": {
                    "pass_sumw": [data_pass.tolist()],
                    "pass_sumw2": [np.maximum(data_pass, 1).tolist()],
                    "fail_sumw": [data_fail.tolist()],
                    "fail_sumw2": [np.maximum(data_fail, 1).tolist()],
                },
                "mc": {
                    "pass_sumw": [mc_pass.tolist()],
                    "pass_sumw2": [np.maximum(mc_pass, 1).tolist()],
                    "fail_sumw": [mc_fail.tolist()],
                    "fail_sumw2": [np.maximum(mc_fail, 1).tolist()],
                },
            },
        }
        result = fit_histogram_payload(payload)
        fitted = result["bins"][0]
        self.assertTrue(fitted["valid"])
        self.assertAlmostEqual(fitted["scale_factor"], 0.82 / 0.78, delta=0.03)
        self.assertGreater(fitted["scale_factor_uncertainty"], 0.0)
        self.assertGreaterEqual(
            fitted["fits"]["nominal"]["data"]["fail_fraction_significance"],
            3.0,
        )
        self.assertEqual(fitted["fits"]["nominal"]["data"]["signal_model"], "double_crystal_ball")
        self.assertEqual(fitted["fits"]["nominal"]["data"]["background_model"], "chebyshev")
        self.assertEqual(fitted["fits"]["nominal"]["data"]["pass_fail_signal_shape"], "shared")
        self.assertEqual(
            fitted["fits"]["pass_fail_shape_independent"]["data"]["pass_fail_signal_shape"],
            "independent",
        )
        self.assertEqual(
            fitted["fits"]["alternate_signal"]["data"]["signal_model"],
            "gaussian_exponential",
        )
        self.assertEqual(
            fitted["fits"]["alternate_background"]["data"]["background_model"],
            "exponential",
        )

    def test_40_mev_nominal_uses_20_mev_as_alternate(self):
        edges = np.linspace(2.5, 3.7, 61)
        data_pass, data_fail = spectrum(edges, 30000, 0.82, 3000, 4000)
        mc_pass, mc_fail = spectrum(edges, 50000, 0.78, 2500, 3500)
        payload = {
            "measurement": "test_40mev",
            "nominal_mass_rebin_factor": 2,
            "mass_edges_gev": edges.tolist(),
            "fit_window_gev": [2.6, 3.6],
            "probe_abseta_edges": [0.0, 2.4],
            "probe_pt_edges_gev": [5.0, 10.0],
            "samples": {
                "data": {
                    "pass_sumw": [data_pass.tolist()],
                    "pass_sumw2": [np.maximum(data_pass, 1).tolist()],
                    "fail_sumw": [data_fail.tolist()],
                    "fail_sumw2": [np.maximum(data_fail, 1).tolist()],
                },
                "mc": {
                    "pass_sumw": [mc_pass.tolist()],
                    "pass_sumw2": [np.maximum(mc_pass, 1).tolist()],
                    "fail_sumw": [mc_fail.tolist()],
                    "fail_sumw2": [np.maximum(mc_fail, 1).tolist()],
                },
            },
        }
        result = fit_histogram_payload(payload)
        fitted = result["bins"][0]
        self.assertTrue(fitted["valid"])
        self.assertAlmostEqual(result["nominal_mass_bin_width_mev"], 40.0)
        self.assertEqual(fitted["fits"]["nominal"]["data"]["rebin_factor"], 2)
        self.assertEqual(fitted["fits"]["alternate_binning"]["data"]["rebin_factor"], 1)


if __name__ == "__main__":
    unittest.main()
