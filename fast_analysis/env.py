from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass

from .config.defaults import DEFAULTS
from .paths import PathKind, PathPolicy, canonicalize, configure_eos_runtime_env

REQUIRED_PACKAGES = ("numpy", "awkward", "uproot", "correctionlib", "coffea", "hist", "boost_histogram")
OPTIONAL_PACKAGES = ("pyarrow", "ROOT")
FORBIDDEN_PREFIXES = ("/afs", "/usr/lib", "/usr/local/lib")


@dataclass
class PackageInfo:
    name: str
    version: object
    path: object
    ok: bool
    required: bool
    error: object = None


def _same_file(path_a, path_b):
    try:
        return os.path.samefile(str(path_a), str(path_b))
    except OSError:
        return False


def _is_forbidden_package_path(path):
    if not path:
        return False
    text = str(path)
    if "/.local/" in text or text.startswith(FORBIDDEN_PREFIXES):
        return True
    return False


def _package_ok(path):
    if not path:
        return False
    if _is_forbidden_package_path(path):
        return False
    try:
        return os.path.realpath(path).startswith(os.path.realpath(str(DEFAULTS.environment_path)) + os.sep)
    except OSError:
        return False


def _package_info(name, required):
    try:
        module = importlib.import_module(name)
        module_path = getattr(module, "__file__", None)
        version = getattr(module, "__version__", None)
        if name == "ROOT":
            version = getattr(module, "gROOT", None).GetVersion() if getattr(module, "gROOT", None) else version
        return PackageInfo(name, version, canonicalize(module_path) if module_path else module_path, _package_ok(module_path), required)
    except Exception as exc:
        return PackageInfo(name, None, None, not required, required, "%s: %s" % (type(exc).__name__, exc))


def _sanitize_sys_path():
    clean = []
    removed = []
    for item in sys.path:
        if item and (_is_forbidden_package_path(item) or "/.local/" in item):
            removed.append(item)
            continue
        clean.append(item)
    sys.path[:] = clean
    return removed


def collect_environment(dry_run=False):
    os.environ["PYTHONNOUSERSITE"] = "1"
    policy = PathPolicy.default()
    env_vars = configure_eos_runtime_env(DEFAULTS.output_root, dry_run=dry_run)
    removed_sys_path = _sanitize_sys_path()
    executable = policy.resolve(sys.executable, PathKind.EXECUTABLE)
    fixed_python_ok = _same_file(executable, DEFAULTS.fixed_python)
    packages = [_package_info(name, True) for name in REQUIRED_PACKAGES]
    packages.extend(_package_info(name, False) for name in OPTIONAL_PACKAGES)
    package_dicts = [asdict(pkg) for pkg in packages]
    required_ok = all(pkg["ok"] for pkg in package_dicts if pkg["required"])
    forbidden_loaded = [pkg for pkg in package_dicts if pkg.get("path") and _is_forbidden_package_path(pkg["path"])]
    return {
        "python_executable": str(DEFAULTS.fixed_python),
        "actual_python_executable": canonicalize(str(executable)),
        "fixed_python_ok": fixed_python_ok,
        "python_version": sys.version,
        "platform": platform.platform(),
        "sys_prefix": canonicalize(sys.prefix),
        "sys_path": [canonicalize(item) for item in sys.path],
        "removed_sys_path_entries": [canonicalize(item) for item in removed_sys_path],
        "required_environment": str(DEFAULTS.environment_path),
        "environment_variables": env_vars,
        "packages": package_dicts,
        "unavailable_optional_modules": [pkg["name"] for pkg in package_dicts if (not pkg["required"] and pkg.get("error"))],
        "ok": fixed_python_ok and required_ok and not forbidden_loaded,
    }


def environment_json(dry_run=False):
    return json.dumps(collect_environment(dry_run=dry_run), indent=2, sort_keys=True)
