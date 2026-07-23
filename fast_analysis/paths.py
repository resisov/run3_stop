from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from .config.defaults import CANONICAL_REPO_ROOT

PathLike = Union[str, Path]


class PathKind(str, Enum):
    INPUT = "project input path"
    OUTPUT = "project output path"
    TEMP = "temporary path"
    CACHE = "cache path"
    EXECUTABLE = "executable path"
    REMOTE_URL = "remote URL"


class PathPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PathPolicy:
    repo_root: Path
    approved_roots: Tuple[Path, ...]
    display_root: Path = CANONICAL_REPO_ROOT

    @classmethod
    def default(cls):
        repo = CANONICAL_REPO_ROOT
        return cls(repo_root=repo, approved_roots=(_resolve(repo),), display_root=repo)

    @classmethod
    def from_strings(cls, repo_root, extra_roots=()):
        repo_display = Path(str(repo_root))
        roots = (_resolve(repo_display),) + tuple(_resolve(root) for root in extra_roots)
        for root in roots:
            if not str(root).startswith("/eos/"):
                raise PathPolicyError("approved root is not under /eos: %s" % root)
        return cls(repo_root=repo_display, approved_roots=roots, display_root=repo_display)

    def resolve(self, path, kind):
        if kind == PathKind.REMOTE_URL:
            raise PathPolicyError("remote URLs must be validated with validate_remote_url")
        if kind == PathKind.EXECUTABLE:
            return self.resolve_executable(str(path))
        if str(path).startswith("~"):
            raise PathPolicyError("%s uses a home-relative path, which is forbidden: %s" % (kind.value, path))
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = _resolve(candidate)
        if not str(resolved).startswith("/eos/"):
            raise PathPolicyError("%s is outside EOS: %s" % (kind.value, resolved))
        if not self._under_approved_root(resolved):
            raise PathPolicyError("%s is outside approved roots: %s" % (kind.value, resolved))
        return resolved

    def canonical(self, path):
        resolved = self.resolve(path, PathKind.OUTPUT if not Path(path).exists() else PathKind.INPUT)
        text = str(resolved)
        canonical_root = str(self.display_root)
        for root in self.approved_roots:
            root_text = str(root)
            if text == root_text or text.startswith(root_text + "/"):
                return canonical_root + text[len(root_text):]
        return text

    def resolve_executable(self, name):
        found = shutil.which(name)
        if found is None:
            raise PathPolicyError("executable not found: %s" % name)
        return Path(found).resolve()

    def validate_remote_url(self, url, listed_urls):
        parsed = urlparse(url)
        if parsed.scheme not in {"root", "xrootd"}:
            raise PathPolicyError("unsupported remote URL scheme: %s" % url)
        if url not in listed_urls:
            raise PathPolicyError("remote URL was not explicitly listed in an EOS manifest: %s" % url)
        return url

    def _under_approved_root(self, resolved):
        for root in self.approved_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False


def _resolve(path):
    return Path(path).expanduser().resolve()


def canonicalize(path):
    if path is None:
        return None
    text = str(path)
    if not text.startswith("/"):
        return text
    resolved = str(_resolve(text))
    aliases = {
        str(_resolve(CANONICAL_REPO_ROOT)): str(CANONICAL_REPO_ROOT),
        str(_resolve("/eos/user/t/taiwoo/miniconda3/envs/py38")): "/eos/user/t/taiwoo/miniconda3/envs/py38",
        str(_resolve("/eos/user/t/taiwoo/decaf")): "/eos/user/t/taiwoo/decaf",
    }
    for physical, canonical in aliases.items():
        if resolved == physical or resolved.startswith(physical + "/"):
            return canonical + resolved[len(physical):]
    return text


def configure_eos_runtime_env(output_root, dry_run=False):
    policy = PathPolicy.default()
    root = policy.resolve(output_root, PathKind.OUTPUT)
    tmp = root / "tmp"
    cache = root / "cache"
    env = {
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "XDG_CACHE_HOME": str(cache),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "PYTHONNOUSERSITE": "1",
    }
    if not dry_run:
        for value in env.values():
            if value.startswith(str(root)):
                Path(value).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
    return {key: canonicalize(value) for key, value in env.items()}
