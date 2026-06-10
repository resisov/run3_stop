from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import AnalysisConfig, ConfigError
from .paths import ResolvedPaths, resolve_paths


@dataclass(frozen=True)
class ValidationItem:
    status: str
    message: str


def validate_config(config: AnalysisConfig, require_compiled: bool = False) -> List[ValidationItem]:
    items: List[ValidationItem] = []

    _check_value(items, "analysis year", config.year, required=True)
    _check_value(items, "shift names", ", ".join(sorted(config.shifts)), required=True)

    for shift_name in sorted(config.shifts):
        try:
            resolved = resolve_paths(config, shift_name)
        except ConfigError as exc:
            items.append(ValidationItem("ERROR", f"{shift_name}: {exc}"))
            continue

        validate_common_inputs(items, resolved, shift_name, require_compiled=require_compiled)

    return items


def validate_common_inputs(
    items: List[ValidationItem],
    resolved: ResolvedPaths,
    shift_name: str,
    require_compiled: bool = False,
) -> None:
    _check_file(items, "processor source", resolved.processor_source)
    _check_file(items, "run.py", resolved.run_py)
    _check_file(items, "reduce.py", resolved.reduce_py)
    _check_file(items, "merge.py", resolved.merge_py)
    _check_file(items, "scale.py", resolved.scale_py)
    _check_file(items, "metadata JSON", resolved.metadata_file)
    _check_file(items, "corrections.coffea", resolved.corrections_file)
    _check_file(items, "ids.coffea", resolved.ids_file)
    _check_file(items, "common.coffea", resolved.common_file)
    _check_file(items, f"{shift_name} dataset list", resolved.dataset_list_file)
    _check_parent(items, f"{shift_name} processor output parent", resolved.processor_output)
    _check_parent(items, f"{shift_name} histogram output parent", resolved.futures_dir)
    _check_parent(items, f"{shift_name} merged output parent", resolved.merged_file)
    _check_parent(items, f"{shift_name} scaled output parent", resolved.scaled_file)

    if resolved.processor_output.exists():
        items.append(ValidationItem("OK", f"{shift_name} processor pickle exists: {resolved.processor_output}"))
    elif require_compiled:
        items.append(ValidationItem("ERROR", f"{shift_name} processor pickle missing: {resolved.processor_output}"))
    else:
        items.append(ValidationItem("WARNING", f"{shift_name} processor pickle missing; run/compile will create it: {resolved.processor_output}"))


def validate_condor_inputs(config: AnalysisConfig, resolved: ResolvedPaths, shift_name: str, dry_run: bool = False) -> List[ValidationItem]:
    items: List[ValidationItem] = []
    try:
        config.shift(shift_name)
        items.append(ValidationItem("OK", f"configured shift exists: {shift_name}"))
    except ConfigError as exc:
        items.append(ValidationItem("ERROR", str(exc)))
        return items

    _check_file(items, "Condor worker script", resolved.condor_executable)
    _check_file(items, "Condor submit template", resolved.condor_submit_template)
    _check_file(items, "Condor setup script", resolved.condor_setup_script)
    _check_file(items, "Condor dataset-list file", resolved.dataset_list_file)
    dataset_dir = resolved.dataset_list_file.parent
    _check_file(items, "Condor nominal dataset-list file", dataset_dir / "datasets_2024.txt")
    _check_file(items, "Condor MC-only dataset-list file", dataset_dir / "datasets_2024_onlymc.txt")
    _check_file(items, "Condor analysis tarball", resolved.condor_analysis_tarball, dry_run=dry_run)
    _check_file(items, "Condor Python tarball", resolved.condor_python_tarball, dry_run=dry_run)
    _check_file(items, "Condor proxy", resolved.condor_proxy, dry_run=dry_run)
    _check_directory_status(items, "Condor shared log directory", resolved.condor_initialdir, dry_run=dry_run)
    _check_directory_status(items, "Condor destination histogram directory", resolved.futures_dir, dry_run=dry_run)
    _check_parent(items, "Condor processor output parent", resolved.processor_output)
    return items


def has_errors(items: List[ValidationItem]) -> bool:
    return any(item.status == "ERROR" for item in items)


def print_validation(items: List[ValidationItem]) -> None:
    for item in items:
        print(f"[{item.status}] {item.message}")


def _check_file(items: List[ValidationItem], label: str, path: Path, dry_run: bool = False) -> None:
    if path.is_file():
        items.append(ValidationItem("OK", f"{label}: {path}"))
    elif dry_run:
        items.append(ValidationItem("WARNING", f"{label} missing for dry-run: {path}"))
    else:
        items.append(ValidationItem("ERROR", f"{label} missing: {path}"))


def _check_directory_status(items: List[ValidationItem], label: str, path: Path, dry_run: bool = False) -> None:
    if path.is_dir():
        items.append(ValidationItem("OK", f"{label} exists: {path}"))
    elif dry_run:
        items.append(ValidationItem("WARNING", f"{label} missing; dry-run would create it if this were a real submission: {path}"))
    else:
        items.append(ValidationItem("WARNING", f"{label} missing; real submission will create it: {path}"))


def _check_parent(items: List[ValidationItem], label: str, path: Path) -> None:
    parent = path.parent
    if parent.is_dir():
        items.append(ValidationItem("OK", f"{label}: {parent}"))
    else:
        items.append(ValidationItem("ERROR", f"{label} missing: {parent}"))


def _check_value(items: List[ValidationItem], label: str, value: str, required: bool = False) -> None:
    if value:
        items.append(ValidationItem("OK", f"{label}: {value}"))
    elif required:
        items.append(ValidationItem("ERROR", f"{label} is missing"))
    else:
        items.append(ValidationItem("WARNING", f"{label} is empty"))
