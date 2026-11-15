"""Runtime analysis paths are internal; these are not physics-validation tests."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from TROTASR.utils.paths import ROOT, internal_path, payload_path
from TROTASR.utils.io import read_json, write_json, sha256


class DependencyBoundaryTests(unittest.TestCase):
    def test_external_io_rejected_before_open(self):
        from TROTASR.utils.statistical_templates import load_theory
        outside = ROOT.parent / 'not_approved.json'
        for operation in (read_json, sha256, load_theory, lambda p: write_json(p, {})):
            with self.assertRaisesRegex(ValueError, 'escapes TROTASR'):
                operation(outside)

    def test_parent_traversal_rejected(self):
        with self.assertRaises(ValueError):
            internal_path(ROOT / '..' / 'external.json')

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            link = Path(folder) / 'escape'
            link.symlink_to(ROOT.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                internal_path(link / 'external.json')

    def test_external_payload_root_rejected(self):
        with self.assertRaises(ValueError):
            payload_path('scales/JME/2024/jet_jerc.json.gz', root=ROOT.parent)

    def test_internal_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            path = Path(folder) / 'test.json.gz'
            write_json(path, {'internal': True})
            self.assertEqual(read_json(path), {'internal': True})

    def test_retired_execution_paths_rejected(self):
        for path in ('campaigns/run.json', 'validation/old_runner.py', 'history/old_runner.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                internal_path(ROOT / path)

    def test_workflows_do_not_extend_analysis_import_paths(self):
        native_bindings = {
            ('combine_impacts.py', 'sys.path.insert(0, cmssw_python)'),
            ('prepare_datacard_review.py', "sys.path.insert(0, str(Path(os.environ['CMSSW_BASE']) / 'python'))"),
        }
        for path in (ROOT / 'workflows').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                if ast.unparse(node.func) not in ('sys.path.insert', 'sys.path.append'):
                    continue
                # Impacts can add only the installed CMSSW native Python
                # binding, not an external analysis implementation.
                self.assertIn((path.name, ast.unparse(node)), native_bindings)

    def test_canonical_command_selects_histogram_stage(self):
        result = subprocess.run(
            [sys.executable, '-E', '-s', '-m', 'TROTASR', 'histograms', '--help'],
            cwd=str(ROOT.parent), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--inventory', result.stdout)

    def test_no_internal_file_symlinks_escape(self):
        for path in ROOT.rglob('*'):
            if path.is_symlink():
                internal_path(path)

    def test_legacy_sources_not_runtime_imports(self):
        for path in ROOT.rglob('*.py'):
            if 'renderers' in path.parts:
                continue  # Frozen renderers are checked at the adapter boundary.
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(n.startswith(('autonomous_allhad', 'build_combine_inputs',
                                                  'object_systematics_overlay')) for n in names), path)


if __name__ == '__main__':
    unittest.main()
