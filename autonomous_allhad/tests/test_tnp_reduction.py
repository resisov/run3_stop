import json
import tempfile
import unittest
from pathlib import Path

from workflow.tnp_measurement_reduce import merge_tnp_shards, rebin_probe_histograms
from workflow.tnp_recovery import finalize_permanent_skips


def payload(path, *, successful, value):
    histogram = {
        "pass_sumw": [[value, 0.0]],
        "pass_sumw2": [[value, 0.0]],
        "fail_sumw": [[0.0, value]],
        "fail_sumw2": [[0.0, value]],
    }
    return {
        "measurement": "test_tnp",
        "kind": "electron",
        "status": "success" if successful else "failed",
        "probe_abseta_edges": [0.0, 2.5],
        "probe_pt_edges_gev": [5.0, 10.0],
        "mass_edges_gev": [2.6, 3.1, 3.6],
        "fit_window_gev": [2.6, 3.6],
        "samples": {name: dict(histogram) for name in ("data", "mc", "mc_pileup_up", "mc_pileup_down")},
        "input_records": [{"file_path": path, "dataset": "/Data/Test/NANOAOD", "sample": "data"}],
        "processing": {
            "data": {
                "files_successful": [path] if successful else [],
                "files_failed": [] if successful else [{"path": path, "error": "read failed"}],
            }
        },
        "files_expected": 1,
        "files_processed": int(successful),
        "files_failed": int(not successful),
        "adoption_blockers": [],
    }


class TnpReductionTest(unittest.TestCase):
    def test_failed_primary_then_successful_recovery_is_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "shard_00000.json"
            recovery = root / "recovery_00000.json"
            primary.write_text(json.dumps(payload("root://example/file.root", successful=False, value=0.0)))
            recovery.write_text(json.dumps(payload("root://example/file.root", successful=True, value=1.0)))
            result = merge_tnp_shards(
                [primary, recovery],
                kind="electron",
                config={"measurement": "test_tnp"},
            )
        self.assertEqual(result["files_expected"], 1)
        self.assertEqual(result["files_processed"], 1)
        self.assertEqual(result["files_failed"], [])
        self.assertEqual(result["samples"]["data"]["pass_sumw"], [[1.0, 0.0]])

    def test_duplicate_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            item = payload("root://example/file.root", successful=True, value=1.0)
            left.write_text(json.dumps(item))
            right.write_text(json.dumps(item))
            with self.assertRaises(RuntimeError):
                merge_tnp_shards([left, right], kind="electron", config={"measurement": "test_tnp"})

    def test_finalize_permanent_skips_freezes_retained_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign.json"
            manifest = root / "residual.json"
            retained = {"file_path": "root://example/good.root", "sample": "data"}
            skipped = {"file_path": "root://example/bad.root", "sample": "data"}
            campaign.write_text(json.dumps({
                "measurement": "test_tnp",
                "records": [retained, skipped],
            }))
            manifest.write_text(json.dumps({
                "measurement": "test_tnp",
                "campaign_records": str(campaign),
                "records": [skipped],
                "failure_diagnostics": {
                    skipped["file_path"]: {"path": skipped["file_path"], "error": "missing replica"}
                },
            }))
            output_records = root / "retained.json"
            output_skips = root / "skips.json"
            summary = finalize_permanent_skips(
                manifest_path=manifest,
                output_records=output_records,
                output_skips=output_skips,
            )
            frozen = json.loads(output_records.read_text())
            audit = json.loads(output_skips.read_text())
        self.assertEqual(summary["files_retained"], 1)
        self.assertEqual(frozen["records"], [retained])
        self.assertEqual(audit["files_permanently_skipped"], 1)
        self.assertFalse(audit["data_lumi_coverage_complete"])

    def test_explicit_records_exclude_audited_failed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.json"
            bad = root / "bad.json"
            good.write_text(json.dumps(payload("root://example/good.root", successful=True, value=1.0)))
            bad.write_text(json.dumps(payload("root://example/bad.root", successful=False, value=0.0)))
            result = merge_tnp_shards(
                [good, bad],
                kind="electron",
                config={"measurement": "test_tnp"},
                expected_records=[{"file_path": "root://example/good.root"}],
            )
        self.assertEqual(result["files_expected"], 1)
        self.assertEqual(result["files_processed"], 1)
        self.assertEqual(result["files_failed"], [])
        self.assertEqual(result["adoption_blockers"], [])

    def test_probe_histograms_are_exactly_merged(self):
        sample = {
            key: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
            for key in ("pass_sumw", "pass_sumw2", "fail_sumw", "fail_sumw2")
        }
        result = rebin_probe_histograms(
            {
                "probe_abseta_edges": [0.0, 1.0, 2.0],
                "probe_pt_edges_gev": [5.0, 7.0, 10.0],
                "samples": {name: sample for name in ("data", "mc", "mc_pileup_up", "mc_pileup_down")},
            },
            target_eta_edges=[0.0, 2.0],
            target_pt_edges=[5.0, 10.0],
        )
        self.assertEqual(result["samples"]["data"]["pass_sumw"], [[16.0, 20.0]])
        self.assertEqual(result["probe_abseta_edges"], [0.0, 2.0])
        self.assertEqual(result["probe_pt_edges_gev"], [5.0, 10.0])

    def test_probe_histograms_can_drop_an_out_of_scope_high_pt_bin(self):
        sample = {
            key: [[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]
            for key in ("pass_sumw", "pass_sumw2", "fail_sumw", "fail_sumw2")
        }
        result = rebin_probe_histograms(
            {
                "probe_abseta_edges": [0.0, 2.5],
                "probe_pt_edges_gev": [5.0, 7.0, 10.0, 15.0],
                "samples": {name: sample for name in ("data", "mc", "mc_pileup_up", "mc_pileup_down")},
            },
            target_eta_edges=[0.0, 2.5],
            target_pt_edges=[5.0, 7.0, 10.0],
        )
        self.assertEqual(result["samples"]["data"]["pass_sumw"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(result["probe_pt_edges_gev"], [5.0, 7.0, 10.0])


if __name__ == "__main__":
    unittest.main()
