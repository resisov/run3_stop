"""Canonical account admission: no ROOT or actual child launch."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel
from TROTASR.workflows import diagnostic_common as common

GIB = 1024**3

class FitAdmissionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=common.ROOT / 'tests')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'stats').mkdir()
        for replacement in (patch.object(common, 'ROOT', self.root),
                            patch.object(common.sys, 'platform', 'linux')):
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_account_cap_and_memory_reserve(self):
        observations = [(120, 100*GIB), (119, 36*GIB-1), (119, 36*GIB)]
        with patch.object(common, 'live_resources', side_effect=observations), \
             patch.object(common.time, 'sleep') as sleep, \
             patch.object(common.subprocess, 'Popen', return_value=sentinel.child) as popen:
            self.assertIs(common.admitted_popen(['combine']), sentinel.child)
            self.assertEqual(sleep.call_count, 2)
            popen.assert_called_once_with(['combine'])

    def test_concurrent_admissions_share_lock(self):
        launched = []
        def resources():
            return 104+len(launched), 100*GIB
        def popen(command, **kwargs):
            launched.append(command)
            self.assertLessEqual(104+len(launched), 120)
            return sentinel.child
        with patch.object(common, 'live_resources', side_effect=resources), \
             patch.object(common.subprocess, 'Popen', side_effect=popen):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda i: common.admitted_popen(['combine', str(i)]), range(16)))
        self.assertEqual(results, [sentinel.child]*16)

    def test_local_fits_remain_disabled(self):
        with patch.object(common.sys, 'platform', 'darwin'), self.assertRaises(RuntimeError):
            common.admitted_popen(['combine'])
