from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from prepare_2025_data_campaign import render_submit, render_wrapper  # noqa: E402
from autonomous_allhad.real_subset_worker import (  # noqa: E402
    analysis_year,
    campaign_year,
    golden_lumi_mask,
    lumimask_path,
)


class Data2025PolicyTest(unittest.TestCase):
    def test_2025_uses_its_own_validated_corrections(self) -> None:
        self.assertEqual(campaign_year("2025"), "2025")
        self.assertEqual(analysis_year("2025"), "2025")
        self.assertEqual(analysis_year("2024"), "2024")

    @mock.patch.dict(os.environ, {"AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0"})
    def test_2025_lumimask_is_the_2025_golden_json(self) -> None:
        path = lumimask_path(REPO, "2025")
        self.assertEqual(path.name, "Cert_Collisions2025_391658_398903_Golden.json")
        self.assertTrue(path.is_file())

    @mock.patch.dict(os.environ, {"AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA": "0"})
    def test_2025_data_events_are_filtered_by_2025_good_runs(self) -> None:
        path = lumimask_path(REPO, "2025")
        payload = json.loads(path.read_text())
        run_text = sorted(payload, key=int)[0]
        valid_lumi = int(payload[run_text][0][0])
        arrays = {
            "run": np.asarray([int(run_text), int(run_text)], dtype=np.int64),
            "luminosityBlock": np.asarray([valid_lumi, 0], dtype=np.int64),
        }

        mask, source = golden_lumi_mask(arrays, "JetMET", REPO, 2, "2025")

        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(Path(source), path)

    def test_mc_is_not_lumimasked(self) -> None:
        arrays = {
            "run": np.asarray([1, 2], dtype=np.int64),
            "luminosityBlock": np.asarray([1, 1], dtype=np.int64),
        }

        mask, source = golden_lumi_mask(arrays, "TT", REPO, 2, "2025")

        self.assertEqual(mask.tolist(), [True, True])
        self.assertEqual(source, "not_applicable_mc")

    def test_condor_uses_transferred_python_environment_in_worker_scratch(self) -> None:
        campaign = Path("/eos/user/t/taiwoo/run3_stop/decaf/campaign")
        python_env = Path("/eos/user/t/taiwoo/run3_stop/decaf/condor/py38.tgz")
        proxy = Path(
            "/eos/user/t/taiwoo/run3_stop/decaf/analysis/proxy/x509up_u147757"
        )

        wrapper = render_wrapper(REPO, python_env, proxy)
        submit = render_submit(campaign, python_env, proxy)

        self.assertIn('job_runtime="$PWD/runtime"', wrapper)
        self.assertIn('tar -xzf "$python_env"', wrapper)
        self.assertIn('[ -x "$PWD/bin/python" ]', wrapper)
        self.assertIn('python="$pyroot/bin/python"', wrapper)
        self.assertIn('export X509_USER_PROXY="$proxy"', wrapper)
        self.assertNotIn("miniconda3/envs/py38/bin/python", wrapper)
        self.assertNotIn("/tmp/", wrapper)
        self.assertIn("should_transfer_files = YES", submit)
        self.assertIn("when_to_transfer_output = ON_EXIT", submit)
        self.assertIn(f"transfer_input_files = {python_env}, {proxy}", submit)
        self.assertIn('transfer_output_files = ""', submit)


if __name__ == "__main__":
    unittest.main()
