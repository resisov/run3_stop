# EOS py38 Environment Inventory

Fixed executable: `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python`
Actual executable: `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python3.8`
Python version: `3.8.20 (default, Oct  3 2024, 15:24:27) `
sys.prefix: `/eos/user/t/taiwoo/miniconda3/envs/py38`
Validation status: `passed`

Removed sys.path entries before core imports:
- `/afs/cern.ch/user/t/taiwoo/.local/lib/python3.8/site-packages`

Core packages:
- `numpy` (required): version `1.23.5`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/numpy/__init__.py`, error `None`
- `awkward` (required): version `1.10.3`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/awkward/__init__.py`, error `None`
- `uproot` (required): version `4.3.7`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/uproot/__init__.py`, error `None`
- `correctionlib` (required): version `2.6.4`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/correctionlib/__init__.py`, error `None`
- `coffea` (required): version `0.7.22`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/coffea/__init__.py`, error `None`
- `hist` (required): version `2.7.2`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/hist/__init__.py`, error `None`
- `boost_histogram` (required): version `1.4.1`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/boost_histogram/__init__.py`, error `None`
- `pyarrow` (optional): version `17.0.0`, ok `True`, path `/eos/user/t/taiwoo/miniconda3/envs/py38/lib/python3.8/site-packages/pyarrow/__init__.py`, error `None`
- `ROOT` (optional): version `None`, ok `True`, path `None`, error `ModuleNotFoundError: No module named 'ROOT'`

Architecture constraints:
- No new Python environment is created.
- No package installation is performed.
- PyArrow is available, so Parquet can be benchmarked in the fixed environment.
- ROOT is not importable as a Python module; ROOT output should use uproot unless this changes.
- `PYTHONNOUSERSITE=1` is set, and AFS/user-site entries are removed before package imports.
