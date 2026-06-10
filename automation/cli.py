from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import AnalysisConfig, ConfigError, load_config
from .paths import ResolvedPaths, load_dataset_list, resolve_dataset_names, resolve_paths
from .validation import has_errors, print_validation, validate_condor_inputs, validate_config

SMALL_TEST_EXAMPLE = """
Safe small-test dry-run example:
  python -m automation.cli run \
      --config configs/stop_2024.yaml \
      --shift nominal \
      --dataset-prefix TTto2L2Nu_ \
      --max-datasets 1 \
      --max-files 1 \
      --dry-run
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1/2 automation wrapper for the existing analysis scripts",
        epilog=SMALL_TEST_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    _add_config(validate_parser)
    validate_parser.add_argument("--require-compiled", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    _add_config(compile_parser)
    compile_parser.add_argument("--shift", default="nominal")
    compile_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        epilog=SMALL_TEST_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config(run_parser)
    run_parser.add_argument("--shift", default="nominal")
    dataset_group = run_parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", default=None, help="Legacy substring metadata-key match")
    dataset_group.add_argument("--dataset-key", default=None, help="Exact metadata key")
    dataset_group.add_argument("--dataset-prefix", default=None, help="Metadata-key prefix")
    run_parser.add_argument("--max-datasets", type=int, default=None)
    run_parser.add_argument("--max-files", type=int, default=None)
    run_parser.add_argument("--dry-run", action="store_true")

    submit_parser = subparsers.add_parser("submit")
    _add_config(submit_parser)
    submit_parser.add_argument("--shift", default="nominal")
    submit_parser.add_argument("--max-jobs", type=int, default=None)
    submit_parser.add_argument("--dry-run", action="store_true")

    post_parser = subparsers.add_parser("postprocess")
    _add_config(post_parser)
    post_parser.add_argument("--shift", default="nominal")
    post_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "validate":
            return command_validate(config, require_compiled=args.require_compiled)
        if args.command == "compile":
            return command_compile(config, args.shift, dry_run=args.dry_run)
        if args.command == "run":
            return command_run(
                config,
                args.shift,
                args.dataset,
                args.dataset_key,
                args.dataset_prefix,
                args.max_datasets,
                args.max_files,
                dry_run=args.dry_run,
            )
        if args.command == "submit":
            return command_submit(config, args.shift, args.max_jobs, dry_run=args.dry_run)
        if args.command == "postprocess":
            return command_postprocess(config, args.shift, dry_run=args.dry_run)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unhandled command: {args.command}")
    return 2


def command_validate(config: AnalysisConfig, require_compiled: bool = False) -> int:
    print(f"repository root: {config.repo_root}")
    print(f"configured Python executable: {config.python}")
    items = validate_config(config, require_compiled=require_compiled)
    print_validation(items)
    return 1 if has_errors(items) else 0


def command_compile(config: AnalysisConfig, shift_name: str, dry_run: bool = False) -> int:
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    resolved = resolve_paths(config, shift_name)
    shift = config.shift(shift_name)
    compile_cmd = build_compile_command(config, shift)
    print_context(config, resolved, shift_name, None, None, None, [compile_cmd])
    if dry_run:
        return 0
    return run_command(compile_cmd, cwd=resolved.analysis_dir)


def command_run(
    config: AnalysisConfig,
    shift_name: str,
    dataset_filter: Optional[str],
    dataset_key: Optional[str],
    dataset_prefix: Optional[str],
    max_datasets: Optional[int],
    max_files: Optional[int],
    dry_run: bool = False,
) -> int:
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    if max_files is not None and max_files < 1:
        print("[ERROR] --max-files must be a positive integer", file=sys.stderr)
        return 2

    resolved = resolve_paths(config, shift_name)
    shift = config.shift(shift_name)
    resolved_datasets = resolve_dataset_names(
        resolved.metadata_file,
        dataset_filter=dataset_filter,
        dataset_key=dataset_key,
        dataset_prefix=dataset_prefix,
        max_datasets=max_datasets,
    )

    if dataset_filter is not None:
        print("[WARNING] --dataset uses legacy substring matching and may select unintended metadata keys.")

    if not resolved_datasets:
        print("[ERROR] No metadata keys resolved; aborting before processor compilation.", file=sys.stderr)
        return 2

    dataset_argument = ",".join(resolved_datasets)
    compile_cmd = build_compile_command(config, shift)
    run_cmd = build_run_command(config, shift, dataset_argument, max_files)
    print_context(config, resolved, shift_name, dataset_argument, resolved_datasets, max_files, [compile_cmd, run_cmd])

    if dry_run:
        return 0

    rc = run_command(compile_cmd, cwd=resolved.analysis_dir)
    if rc != 0:
        return rc
    return run_command(run_cmd, cwd=resolved.analysis_dir)


def command_submit(config: AnalysisConfig, shift_name: str, max_jobs: Optional[int], dry_run: bool = False) -> int:
    if max_jobs is not None and max_jobs <= 0:
        print("[ERROR] --max-jobs must be a positive integer", file=sys.stderr)
        return 2

    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc

    resolved = resolve_paths(config, shift_name)
    shift = config.shift(shift_name)
    condor_items = validate_condor_inputs(config, resolved, shift_name, dry_run=dry_run)
    print_validation(condor_items)
    if has_errors(condor_items):
        return 1

    jobs = load_dataset_list(resolved.dataset_list_file)
    total_jobs = len(jobs)
    selected_jobs = jobs[:max_jobs] if max_jobs is not None else jobs
    if not selected_jobs:
        print("[ERROR] No Condor queue entries resolved; aborting before processor compilation.", file=sys.stderr)
        return 2

    compile_cmd = build_compile_command(config, shift)
    submit_file = choose_submit_file(config, resolved, shift_name, selected_jobs, max_jobs, dry_run=dry_run)
    submit_cmd = [str(config.data["condor"].get("submit_command", "condor_submit")), str(submit_file)]

    print_submit_context(config, resolved, shift_name, compile_cmd, submit_cmd, total_jobs, selected_jobs, submit_file)

    if dry_run:
        return 0

    rc = run_command(compile_cmd, cwd=resolved.analysis_dir)
    if rc != 0:
        return rc

    prepare_condor_output_dirs(resolved)
    if submit_file != resolved.condor_submit_template:
        write_generated_submit_file(resolved, shift_name, selected_jobs, submit_file)
    return run_command(submit_cmd, cwd=resolved.condor_dir)


def command_postprocess(config: AnalysisConfig, shift_name: str, dry_run: bool = False) -> int:
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    resolved = resolve_paths(config, shift_name)
    commands = build_postprocess_commands(config, resolved)
    print_context(config, resolved, shift_name, None, None, None, commands)
    if dry_run:
        return 0

    for command in commands:
        rc = run_command(command, cwd=resolved.analysis_dir)
        if rc != 0:
            return rc
    return 0


def validate_before_execution(config: AnalysisConfig) -> int:
    items = validate_config(config, require_compiled=False)
    errors = [item for item in items if item.status == "ERROR"]
    if not errors:
        return 0
    print_validation(items)
    return 1


def build_compile_command(config: AnalysisConfig, shift: Dict[str, object]) -> List[str]:
    command = [
        config.python,
        config.data["processor"]["source"],
        "-m",
        config.metadata_name,
        "-y",
        config.year,
        "-n",
        str(shift["compile_name"]),
    ]
    if shift.get("shift_arg"):
        command.extend(["--shift", str(shift["shift_arg"])])
    return command


def build_run_command(
    config: AnalysisConfig,
    shift: Dict[str, object],
    dataset_argument: str,
    max_files: Optional[int],
) -> List[str]:
    command = [
        config.python,
        "run.py",
        "-p",
        str(shift["processor_name"]),
        "-m",
        config.metadata_name,
        "-w",
        str(config.workers),
        "-d",
        dataset_argument,
    ]
    if max_files is not None:
        command.extend(["--max-files", str(max_files)])
    return command


def build_postprocess_commands(config: AnalysisConfig, resolved: ResolvedPaths) -> List[List[str]]:
    futures_dir = _relative_to_analysis(resolved.futures_dir, resolved.analysis_dir)
    merged_file = _relative_to_analysis(resolved.merged_file, resolved.analysis_dir)
    return [
        [config.python, "reduce.py", "-f", futures_dir],
        [config.python, "merge.py", "-f", futures_dir],
        [config.python, "scale.py", "-f", merged_file],
    ]


def choose_submit_file(
    config: AnalysisConfig,
    resolved: ResolvedPaths,
    shift_name: str,
    selected_jobs: List[str],
    max_jobs: Optional[int],
    dry_run: bool = False,
) -> Path:
    use_template = shift_name == "nominal" and max_jobs is None
    if use_template:
        return resolved.condor_submit_template
    generated = resolved.condor_generated_dir / f"{Path(resolved.condor_submit_template).stem}_{shift_name}"
    if max_jobs is not None:
        generated = Path(str(generated) + f"_max{len(selected_jobs)}")
    generated = Path(str(generated) + ".sub")
    return generated


def build_generated_submit_text(resolved: ResolvedPaths, shift_name: str, selected_jobs: List[str]) -> str:
    template_text = resolved.condor_submit_template.read_text(encoding="utf-8")
    prefix = template_text.split("\nshift =", 1)[0].rstrip()
    if prefix.endswith("# nominal"):
        prefix = prefix.rsplit("\n", 1)[0].rstrip()
    if not prefix:
        raise ConfigError(f"Could not parse Condor submit template: {resolved.condor_submit_template}")

    lines = [prefix, "", f"# generated by automation.cli for shift {shift_name}", f"shift = {shift_name}", "queue dataset from (" ]
    lines.extend(selected_jobs)
    lines.append(")")
    lines.append("")
    return "\n".join(lines)

def write_generated_submit_file(resolved: ResolvedPaths, shift_name: str, selected_jobs: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_generated_submit_text(resolved, shift_name, selected_jobs), encoding="utf-8")


def prepare_condor_output_dirs(resolved: ResolvedPaths) -> None:
    resolved.condor_initialdir.mkdir(parents=True, exist_ok=True)
    resolved.futures_dir.mkdir(parents=True, exist_ok=True)


def print_context(
    config: AnalysisConfig,
    resolved: ResolvedPaths,
    shift_name: str,
    dataset_argument: Optional[str],
    resolved_datasets: Optional[List[str]],
    max_files: Optional[int],
    commands: List[List[str]],
) -> None:
    print(f"repository root: {config.repo_root}")
    print(f"analysis directory: {resolved.analysis_dir}")
    print(f"configured Python executable: {config.python}")
    print(f"metadata file: {resolved.metadata_file}")
    print(f"processor source: {resolved.processor_source}")
    print(f"processor output file: {resolved.processor_output}")
    print(f"selected shift: {shift_name}")
    print(f"requested dataset argument passed to run.py: {dataset_argument if dataset_argument else '<not applicable>'}")
    if resolved_datasets is None:
        print("resolved metadata keys: <not applicable>")
    else:
        print(f"resolved metadata key count: {len(resolved_datasets)}")
        print("resolved metadata keys:")
        for dataset in resolved_datasets:
            print(f"  - {dataset}")
    if max_files is None:
        print("max-files scope: no file limit configured")
    else:
        print(f"max-files scope: up to {max_files} file per resolved metadata key")
    print(f"histogram output directory: {resolved.futures_dir}")
    print("exact subprocess commands:")
    for command in commands:
        print(f"  cwd: {resolved.analysis_dir}")
        print(f"  argv: {command}")


def print_submit_context(
    config: AnalysisConfig,
    resolved: ResolvedPaths,
    shift_name: str,
    compile_cmd: List[str],
    submit_cmd: List[str],
    total_jobs: int,
    selected_jobs: List[str],
    submit_file: Path,
) -> None:
    print(f"repository root: {config.repo_root}")
    print(f"analysis directory: {resolved.analysis_dir}")
    print(f"condor directory: {resolved.condor_dir}")
    print(f"configured Python executable: {config.python}")
    print(f"selected shift: {shift_name}")
    print("current Condor job unit: one shift/dataset-list entry pair; the worker receives $(dataset) and $(shift)")
    print(f"worker script: {resolved.condor_executable}")
    print(f"submit template: {resolved.condor_submit_template}")
    print(f"setup script: {resolved.condor_setup_script}")
    print(f"dataset-list file: {resolved.dataset_list_file}")
    print(f"analysis tarball: {resolved.condor_analysis_tarball}")
    print(f"Python tarball: {resolved.condor_python_tarball}")
    print(f"proxy: {resolved.condor_proxy}")
    print(f"processor output file: {resolved.processor_output}")
    print(f"histogram output directory: {resolved.futures_dir}")
    print(f"condor shared log directory: {resolved.condor_initialdir}")
    print(f"condor shared log exists: {resolved.condor_initialdir.is_dir()}")
    if not resolved.condor_initialdir.is_dir():
        print("condor shared log directory action: would create before a real submission")
    print(f"resolved shared cluster log path: {resolved.condor_initialdir / resolved.condor_shared_log}")
    print(f"submit file to use: {submit_file}")
    for message in condor_path_mismatches(config, resolved):
        print(f"[WARNING] {message}")
    print(f"total available jobs: {total_jobs}")
    print(f"jobs selected after limiting: {len(selected_jobs)}")
    print("selected queue entries:")
    for job in selected_jobs:
        print(f"  - {job}")
    if submit_file != resolved.condor_submit_template:
        print("generated submit description:")
        print(build_generated_submit_text(resolved, shift_name, selected_jobs))
    print("exact subprocess commands:")
    print(f"  compile cwd: {resolved.analysis_dir}")
    print(f"  compile argv: {compile_cmd}")
    print(f"  condor cwd: {resolved.condor_dir}")
    print(f"  condor argv: {submit_cmd}")


def condor_path_mismatches(config: AnalysisConfig, resolved: ResolvedPaths) -> List[str]:
    old_root = "/eos/user/t/taiwoo/decaf"
    messages: List[str] = []
    paths = [
        ("analysis tarball", resolved.condor_analysis_tarball),
        ("Python tarball", resolved.condor_python_tarball),
        ("proxy", resolved.condor_proxy),
        ("initialdir", resolved.condor_initialdir),
    ]
    template_text = resolved.condor_submit_template.read_text(encoding="utf-8") if resolved.condor_submit_template.is_file() else ""
    for label, path in paths:
        path_text = str(path)
        if path_text.startswith(old_root):
            messages.append(f"{label} still points to obsolete path: {path_text}")
    if old_root in template_text:
        messages.append(f"submit template still contains obsolete paths under {old_root}")
    return messages


def run_command(command: List[str], cwd: Path) -> int:
    print(f"running: {command}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def _relative_to_analysis(path: Path, analysis_dir: Path) -> str:
    try:
        return str(path.relative_to(analysis_dir))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
