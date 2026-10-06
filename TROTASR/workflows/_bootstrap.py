"""Bind this package by its exact internal file, never add its parent to sys.path."""
import importlib.util
from pathlib import Path
import sys


def bootstrap():
    root = Path(__file__).resolve().parents[1]
    package_file = root / '__init__.py'
    if 'TROTASR' in sys.modules:
        if Path(sys.modules['TROTASR'].__file__).resolve() != package_file:
            raise ImportError('A different TROTASR package is already loaded')
        return
    spec = importlib.util.spec_from_file_location(
        'TROTASR', package_file, submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['TROTASR'] = module
    spec.loader.exec_module(module)
