from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fast_analysis.paths import PathKind, PathPolicy, PathPolicyError, configure_eos_runtime_env
from fast_analysis.workflow.manifest import ChunkRecord, is_chunk_complete


class PathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path("/eos/user/t/taiwoo/run3_stop/decaf").resolve()
        self.policy = PathPolicy.from_strings(self.repo, ("/eos/user/t/taiwoo/run3_stop/decaf/fast_outputs",))

    def test_rejects_non_eos_paths(self) -> None:
        for bad in ("/tmp/example", "/var/tmp/example", "/afs/example"):
            with self.assertRaises(PathPolicyError):
                self.policy.resolve(bad, PathKind.INPUT)

    def test_rejects_home_and_escape_paths(self) -> None:
        with self.assertRaises(PathPolicyError):
            self.policy.resolve("~/example", PathKind.INPUT)
        with self.assertRaises(PathPolicyError):
            self.policy.resolve("../path_that_escapes_repo", PathKind.INPUT)

    def test_accepts_approved_eos_paths(self) -> None:
        accepted = self.policy.resolve("/eos/user/t/taiwoo/run3_stop/decaf/fast_outputs/example", PathKind.OUTPUT)
        self.assertTrue(str(accepted).startswith("/eos/"))
        self.assertTrue(str(accepted).endswith("/run3_stop/decaf/fast_outputs/example"))
        other = PathPolicy.from_strings(self.repo, ("/eos/user/t/taiwoo/run3_stop/decaf/fast_outputs",))
        resolved = other.resolve("/eos/user/t/taiwoo/run3_stop/decaf/fast_outputs/other", PathKind.CACHE)
        self.assertTrue(str(resolved).startswith("/eos/"))

    def test_symlink_escape_rejected_when_available(self) -> None:
        link = self.repo / "fast_outputs" / "tmp" / "test_escape_link"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        target = Path(tempfile.gettempdir()) / "outside_eos_target"
        target.write_text("outside")
        try:
            os.symlink(target, link)
            with self.assertRaises(PathPolicyError):
                self.policy.resolve(link, PathKind.INPUT)
        finally:
            if link.exists() or link.is_symlink():
                link.unlink()
            if target.exists():
                target.unlink()

    def test_dry_run_writes_nothing_and_env_is_eos(self) -> None:
        marker = self.repo / "fast_outputs" / "tmp" / "dry_run_marker"
        if marker.exists():
            marker.unlink()
        env = configure_eos_runtime_env(self.repo / "fast_outputs", dry_run=True)
        self.assertFalse(marker.exists())
        self.assertTrue(env["TMPDIR"].startswith("/eos/"))
        self.assertTrue(env["XDG_CACHE_HOME"].startswith("/eos/"))

    def test_stale_manifest_record_is_not_complete(self) -> None:
        record = ChunkRecord(
            chunk_id="stale",
            dataset="dummy",
            input_files=[],
            state="validated",
            output_path=str(self.repo / "fast_outputs" / "skims" / "missing.parquet"),
            validation_status="passed",
        )
        self.assertFalse(is_chunk_complete(record, self.policy))


if __name__ == "__main__":
    unittest.main()
