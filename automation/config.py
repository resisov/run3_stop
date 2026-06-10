from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisConfig:
    repo_root: Path
    path: Path
    data: Dict[str, Any]

    @property
    def year(self) -> str:
        return str(self.data["analysis"]["year"])

    @property
    def metadata_name(self) -> str:
        return str(self.data["metadata"]["name"])

    @property
    def python(self) -> str:
        return str(self.data["execution"].get("python", "python3"))

    @property
    def workers(self) -> int:
        return int(self.data["execution"].get("workers", 8))

    @property
    def shifts(self) -> Dict[str, Dict[str, Any]]:
        return self.data["shifts"]

    def shift(self, name: str) -> Dict[str, Any]:
        try:
            return self.shifts[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.shifts))
            raise ConfigError(f"Unknown shift '{name}'. Available shifts: {available}") from exc


def find_repo_root(start: Optional[Path] = None) -> Path:
    here = _normalize_eos_user_path((start or Path.cwd()).resolve())
    for candidate in [here] + list(here.parents):
        if (candidate / "analysis").is_dir():
            return candidate
    raise ConfigError(f"Could not find repository root from {here}")


def resolve_config_path(config_path: Union[str, Path]) -> Path:
    raw_path = Path(config_path).expanduser()
    if raw_path.is_absolute():
        return _normalize_eos_user_path(raw_path.resolve())

    cwd_path = _normalize_eos_user_path((Path.cwd() / raw_path).resolve())
    if cwd_path.exists():
        return cwd_path

    repo_root = find_repo_root()
    return _normalize_eos_user_path((repo_root / raw_path).resolve())


def _normalize_eos_user_path(path: Path) -> Path:
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "eos" and parts[2].startswith("home-"):
        initial = parts[2].split("-", 1)[1]
        suffix = Path(*parts[4:]) if len(parts) > 4 else Path()
        return Path("/eos/user") / initial / parts[3] / suffix
    return path


def load_config(config_path: Union[str, Path]) -> AnalysisConfig:
    path = resolve_config_path(config_path)
    repo_root = find_repo_root(path.parent)

    try:
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ConfigError(f"Could not read config {path}: {exc}") from exc

    try:
        data = _load_yaml_text(text)
    except Exception as exc:
        raise ConfigError(f"YAML parse failed for {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")

    _require_mapping(data, "analysis")
    _require_mapping(data, "paths")
    _require_mapping(data, "metadata")
    _require_mapping(data, "processor")
    _require_mapping(data, "execution")
    _require_mapping(data, "shifts")
    _require_mapping(data, "artifacts")

    return AnalysisConfig(repo_root=repo_root, path=path, data=data)


def _load_yaml_text(text: str) -> Dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml_mapping(text)


def _load_simple_yaml_mapping(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.lstrip() != raw_line and "	" in raw_line[:len(raw_line) - len(raw_line.lstrip())]:
            raise ConfigError("Tabs are not supported in indentation")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ConfigError(f"Expected key/value line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError(f"Empty key in line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"Invalid indentation near line: {raw_line}")
        parent = stack[-1][1]

        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _require_mapping(data: Dict[str, Any], key: str) -> None:
    if key not in data or not isinstance(data[key], dict):
        raise ConfigError(f"Missing required mapping: {key}")
