from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AnalysisConfig, ConfigError


@dataclass(frozen=True)
class ResolvedPaths:
    analysis_dir: Path
    metadata_file: Path
    processor_source: Path
    processor_output: Path
    dataset_list_file: Path
    futures_dir: Path
    merged_file: Path
    scaled_file: Path
    run_py: Path
    reduce_py: Path
    merge_py: Path
    scale_py: Path
    corrections_file: Path
    ids_file: Path
    common_file: Path
    condor_dir: Path
    condor_executable: Path
    condor_submit_template: Path
    condor_setup_script: Path
    condor_generated_dir: Path
    condor_analysis_tarball: Path
    condor_python_tarball: Path
    condor_proxy: Path
    condor_initialdir: Path
    condor_shared_log: str


def resolve_paths(config: AnalysisConfig, shift_name: str) -> ResolvedPaths:
    paths_cfg = config.data["paths"]
    shift = config.shift(shift_name)
    condor_cfg = config.data.get("condor", {})

    analysis_dir = _repo_path(config, paths_cfg["analysis_dir"])
    metadata_dir = _repo_path(config, paths_cfg["metadata_dir"])
    data_dir = _repo_path(config, paths_cfg["data_dir"])
    hist_dir = _repo_path(config, paths_cfg["hist_dir"])
    dataset_dir = _repo_path(config, paths_cfg["dataset_dir"])
    condor_dir = _repo_path(config, paths_cfg.get("condor_dir", "condor"))

    processor_source = analysis_dir / config.data["processor"]["source"]
    metadata_file = metadata_dir / config.data["metadata"]["file"]
    dataset_list_file = dataset_dir / shift["datasets_file"]

    format_values = {
        "data_dir": data_dir,
        "hist_dir": hist_dir,
        "processor_name": shift["processor_name"],
    }

    artifacts = config.data["artifacts"]
    processor_output = _format_artifact(config, artifacts["processor_file"], format_values)
    futures_dir = _format_artifact(config, artifacts["futures_dir"], format_values)
    merged_file = _format_artifact(config, artifacts["merged_file"], format_values)
    scaled_file = _format_artifact(config, artifacts["scaled_file"], format_values)

    return ResolvedPaths(
        analysis_dir=analysis_dir,
        metadata_file=metadata_file,
        processor_source=processor_source,
        processor_output=processor_output,
        dataset_list_file=dataset_list_file,
        futures_dir=futures_dir,
        merged_file=merged_file,
        scaled_file=scaled_file,
        run_py=analysis_dir / "run.py",
        reduce_py=analysis_dir / "reduce.py",
        merge_py=analysis_dir / "merge.py",
        scale_py=analysis_dir / "scale.py",
        corrections_file=data_dir / "corrections.coffea",
        ids_file=data_dir / "ids.coffea",
        common_file=data_dir / "common.coffea",
        condor_dir=condor_dir,
        condor_executable=_repo_path(config, str(condor_cfg.get("executable", "condor/run_condor_2024.sh"))),
        condor_submit_template=_repo_path(config, str(condor_cfg.get("submit_template", "condor/run_condor_2024.sub"))),
        condor_setup_script=_repo_path(config, str(condor_cfg.get("setup_script", "setup_condor.sh"))),
        condor_generated_dir=analysis_dir / "condor" / "generated",
        condor_analysis_tarball=_repo_path(config, str(condor_cfg.get("analysis_tarball", ""))),
        condor_python_tarball=_repo_path(config, str(condor_cfg.get("python_tarball", ""))),
        condor_proxy=_repo_path(config, str(condor_cfg.get("proxy", ""))),
        condor_initialdir=_repo_path(config, str(condor_cfg.get("initialdir", "condor_log"))),
        condor_shared_log=str(condor_cfg.get("shared_log", "job.$(ClusterId).log")),
    )


def load_metadata_datasets(metadata_file: Path) -> List[str]:
    with gzip.open(metadata_file, "rt") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        return []
    return sorted(str(key) for key in metadata)


def load_dataset_list(dataset_list_file: Path) -> List[str]:
    try:
        text = dataset_list_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read dataset list {dataset_list_file}: {exc}") from exc
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def resolve_dataset_names(
    metadata_file: Path,
    dataset_filter: Optional[str] = None,
    dataset_key: Optional[str] = None,
    dataset_prefix: Optional[str] = None,
    max_datasets: Optional[int] = None,
) -> List[str]:
    datasets = load_metadata_datasets(metadata_file)

    if max_datasets is not None and max_datasets <= 0:
        raise ConfigError("--max-datasets must be a positive integer")

    modes = [dataset_filter is not None, dataset_key is not None, dataset_prefix is not None]
    if sum(1 for mode in modes if mode) > 1:
        raise ConfigError("--dataset, --dataset-key, and --dataset-prefix are mutually exclusive")

    if dataset_key is not None:
        if dataset_key not in datasets:
            raise ConfigError(f"Metadata key does not exist: {dataset_key}")
        resolved = [dataset_key]
    elif dataset_prefix is not None:
        resolved = [dataset for dataset in datasets if dataset.startswith(dataset_prefix)]
    elif dataset_filter is not None:
        tokens = [token.strip() for token in dataset_filter.split(",") if token.strip()]
        resolved = [dataset for dataset in datasets if any(token in dataset for token in tokens)]
    else:
        resolved = datasets

    if max_datasets is not None:
        resolved = resolved[:max_datasets]

    return resolved


def _repo_path(config: AnalysisConfig, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (config.repo_root / path).absolute()


def _format_artifact(config: AnalysisConfig, template: str, values: Dict[str, Any]) -> Path:
    formatted = template.format(**{key: str(value) for key, value in values.items()})
    path = Path(formatted)
    if path.is_absolute():
        return path.absolute()
    return (config.repo_root / path).absolute()
