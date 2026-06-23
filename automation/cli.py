from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import AnalysisConfig, ConfigError, load_config
from .paths import ResolvedPaths, load_dataset_list, resolve_dataset_names, resolve_paths
from .validation import has_errors, print_validation, validate_condor_inputs, validate_config

LUMI_UNCERTAINTY_NAME = "Lumi_2024"
LUMI_UNCERTAINTY_LNN = 1.016

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
        description="Phase 1/2/3 automation wrapper for the existing analysis scripts",
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
    compile_parser.add_argument("--force", action="store_true")
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
    post_parser.add_argument("--allow-partial", action="store_true")
    post_parser.add_argument("--force", action="store_true")
    post_parser.add_argument("--dry-run", action="store_true")

    plot_parser = subparsers.add_parser("plot")
    _add_config(plot_parser)
    plot_parser.add_argument("--nominal-only", action="store_true")
    plot_parser.add_argument("--dry-run", action="store_true")

    template_parser = subparsers.add_parser("template")
    _add_config(template_parser)
    template_parser.add_argument("--dry-run", action="store_true")

    validate_template_parser = subparsers.add_parser("validate-template")
    _add_config(validate_template_parser)

    status_parser = subparsers.add_parser("status")
    _add_config(status_parser)
    status_parser.add_argument("--shift", default="nominal")

    missing_parser = subparsers.add_parser("missing")
    _add_config(missing_parser)
    missing_parser.add_argument("--shift", default="nominal")

    retry_parser = subparsers.add_parser("retry")
    _add_config(retry_parser)
    retry_parser.add_argument("--shift", default="nominal")
    retry_parser.add_argument("--max-jobs", type=int, default=None)
    retry_parser.add_argument("--dry-run", action="store_true")

    datacard_parser = subparsers.add_parser("datacard")
    _add_config(datacard_parser)
    datacard_parser.add_argument("--signal", default=None)
    datacard_parser.add_argument("--dry-run", action="store_true")

    limit_parser = subparsers.add_parser("limit")
    _add_config(limit_parser)
    limit_parser.add_argument("--signal", required=True)
    limit_parser.add_argument("--dry-run", action="store_true")

    report_parser = subparsers.add_parser("report")
    _add_config(report_parser)
    report_parser.add_argument("--dry-run", action="store_true")

    pipeline_parser = subparsers.add_parser("pipeline")
    _add_config(pipeline_parser)
    pipeline_parser.add_argument("--shift", default="nominal")
    pipeline_parser.add_argument("--all-shifts", action="store_true")
    pipeline_parser.add_argument("--from-stage", default=None)
    pipeline_parser.add_argument("--to-stage", default=None)
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.add_argument("--allow-partial", action="store_true")
    pipeline_parser.add_argument("--signal", default=None)
    pipeline_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "validate":
            return command_validate(config, require_compiled=args.require_compiled)
        if args.command == "compile":
            return command_compile(config, args.shift, dry_run=args.dry_run, force=args.force)
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
            return command_postprocess(config, args.shift, dry_run=args.dry_run, allow_partial=args.allow_partial, force=args.force)
        if args.command == "plot":
            return command_plot(config, nominal_only=args.nominal_only, dry_run=args.dry_run)
        if args.command == "template":
            return command_template(config, dry_run=args.dry_run)
        if args.command == "validate-template":
            return command_validate_template(config)
        if args.command == "status":
            return command_status(config, args.shift)
        if args.command == "missing":
            return command_missing(config, args.shift)
        if args.command == "retry":
            return command_retry(config, args.shift, args.max_jobs, dry_run=args.dry_run)
        if args.command == "datacard":
            return command_datacard(config, signal=args.signal, dry_run=args.dry_run)
        if args.command == "limit":
            return command_limit(config, signal=args.signal, dry_run=args.dry_run)
        if args.command == "report":
            return command_report(config, dry_run=args.dry_run)
        if args.command == "pipeline":
            return command_pipeline(config, shift_name=args.shift, all_shifts=args.all_shifts, from_stage=args.from_stage, to_stage=args.to_stage, force=args.force, allow_partial=args.allow_partial, signal=args.signal, dry_run=args.dry_run)
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


def command_compile(config: AnalysisConfig, shift_name: str, dry_run: bool = False, force: bool = False) -> int:
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    resolved = resolve_paths(config, shift_name)
    shift = config.shift(shift_name)
    compile_cmd = build_compile_command(config, shift)
    if resolved.processor_output.is_file() and validate_coffea_file(resolved.processor_output) and not force:
        print(f"[OK] processor pickle already exists and is loadable: {resolved.processor_output}")
        return 0
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


def command_postprocess(config: AnalysisConfig, shift_name: str, dry_run: bool = False, allow_partial: bool = False, force: bool = False) -> int:
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    resolved = resolve_paths(config, shift_name)
    if not allow_partial:
        status = inspect_production_status(config, shift_name)
        if status["missing_count"] or status["unreadable_count"] or status["zero_size_count"]:
            print("[ERROR] Production is incomplete; use --allow-partial only for explicitly labeled partial work.", file=sys.stderr)
            print_status_summary(status)
            return 1
    commands = build_postprocess_commands(config, resolved)
    print_context(config, resolved, shift_name, None, None, None, commands)
    if dry_run:
        return 0
    if force:
        quarantine_stale_postprocess_outputs(resolved)

    # reduce -> validate reduced
    rc = run_command(commands[0], cwd=resolved.analysis_dir)
    if rc != 0:
        return rc
    if not validate_reduced_outputs(resolved):
        return 1
    # merge -> validate merged
    rc = run_command(commands[1], cwd=resolved.analysis_dir)
    if rc != 0:
        return rc
    if not validate_coffea_file(resolved.merged_file):
        print(f"[ERROR] merged output is not loadable: {resolved.merged_file}", file=sys.stderr)
        return 1
    # scale -> validate scaled
    rc = run_command(commands[2], cwd=resolved.analysis_dir)
    if rc != 0:
        return rc
    if not validate_coffea_file(resolved.scaled_file):
        print(f"[ERROR] scaled output is not loadable: {resolved.scaled_file}", file=sys.stderr)
        return 1
    return 0



def command_plot(config: AnalysisConfig, nominal_only: bool = False, dry_run: bool = False) -> int:
    plotting = _require_section(config, "plotting")
    script = _config_path(config, plotting.get("script", "analysis/distribution_draw_v5.py"))
    input_scaled = _config_path(config, plotting.get("input_scaled", "analysis/hists/stop_2024_nominal.scaled"))
    output_dir = _config_path(config, plotting.get("output_dir", "analysis/plots/stop_2024"))
    external = plotting.get("external_shifts", {})
    if not isinstance(external, dict):
        raise ConfigError("plotting.external_shifts must be a mapping")

    required = {
        "nominal": input_scaled,
        "metUnclusteredUp": _config_path(config, external.get("metUnclusteredUp", "analysis/hists/stop_2024_metUnclusteredUp.scaled")),
        "metUnclusteredDown": _config_path(config, external.get("metUnclusteredDown", "analysis/hists/stop_2024_metUnclusteredDown.scaled")),
        "jesTotalUp": _config_path(config, external.get("jesTotalUp", "analysis/hists/stop_2024_jesTotalUp.scaled")),
        "jesTotalDown": _config_path(config, external.get("jesTotalDown", "analysis/hists/stop_2024_jesTotalDown.scaled")),
    }
    required_for_mode = {"nominal": input_scaled} if nominal_only else required
    missing = _missing_files({"plot script": script, **required_for_mode})
    variables, regions = _inspect_scaled_axes(input_scaled) if input_scaled.is_file() else ([], [])
    cmd = [
        config.python,
        str(script),
        "--year",
        str(plotting.get("year", config.year)),
        "--input-scaled",
        str(input_scaled),
        "--plot-dir",
        str(output_dir),
    ]
    if nominal_only:
        cmd.append("--nominal-only")
    else:
        cmd.extend([
            "--met-unclustered-up",
            str(required["metUnclusteredUp"]),
            "--met-unclustered-down",
            str(required["metUnclusteredDown"]),
            "--jes-total-up",
            str(required["jesTotalUp"]),
            "--jes-total-down",
            str(required["jesTotalDown"]),
        ])
    if plotting.get("luminosity_fb") is not None:
        cmd.extend(["--luminosity-fb", str(plotting["luminosity_fb"])])

    print("plot automation context:")
    print(f"  script: {script}")
    print(f"  nominal scaled: {input_scaled}")
    if nominal_only:
        print("  external shifted inputs: ignored in nominal-only mode")
    else:
        for name in ["metUnclusteredUp", "metUnclusteredDown", "jesTotalUp", "jesTotalDown"]:
            print(f"  external {name}: {required[name]}")
    print(f"  output directory: {output_dir}")
    print(f"  year: {plotting.get('year', config.year)}")
    print(f"  luminosity_fb: {plotting.get('luminosity_fb', '<script default>')}")
    print(f"  analysis label: {config.data['analysis'].get('label', '<unset>')}")
    print(f"  mask_signal_region_data: {plotting.get('mask_signal_region_data', True)}")
    print(f"  nominal_only: {nominal_only}")
    if nominal_only:
        print("  external shifted uncertainties: disabled")
        print("[WARNING] nominal-only mode: external shifted uncertainties are not included")
    print("  variables:")
    for variable in variables:
        print(f"    - {variable}")
    print("  regions:")
    for region in regions:
        print(f"    - {region}")
    print("exact subprocess command:")
    print(f"  cwd: {config.repo_root}")
    print(f"  argv: {cmd}")

    if missing:
        _print_missing(missing)
        return 1
    if dry_run:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_command(cmd, cwd=config.repo_root)


def command_template(config: AnalysisConfig, dry_run: bool = False) -> int:
    template_cfg = _require_section(config, "template")
    script = _config_path(config, template_cfg.get("script", "analysis/make_template.py"))
    nominal = _config_path(config, template_cfg.get("nominal_scaled", "analysis/hists/stop_2024_nominal.scaled"))
    output_root = _config_path(config, template_cfg.get("output_root", "analysis/templates/templates_metpt.root"))
    external_cfg = template_cfg.get("external_shifts", {})
    if not isinstance(external_cfg, dict):
        raise ConfigError("template.external_shifts must be a mapping")
    external = {
        "jesTotalUp": _config_path(config, external_cfg.get("jesTotalUp", "analysis/hists/stop_2024_jesTotalUp.scaled")),
        "jesTotalDown": _config_path(config, external_cfg.get("jesTotalDown", "analysis/hists/stop_2024_jesTotalDown.scaled")),
        "metUnclusteredUp": _config_path(config, external_cfg.get("metUnclusteredUp", "analysis/hists/stop_2024_metUnclusteredUp.scaled")),
        "metUnclusteredDown": _config_path(config, external_cfg.get("metUnclusteredDown", "analysis/hists/stop_2024_metUnclusteredDown.scaled")),
        "jerUp": _config_path(config, external_cfg.get("jerUp", "analysis/hists/stop_2024_jerUp.scaled")),
        "jerDown": _config_path(config, external_cfg.get("jerDown", "analysis/hists/stop_2024_jerDown.scaled")),
    }
    missing = _missing_files({"template script": script, "nominal": nominal, **external})
    meta = _load_template_module(script) if script.is_file() else {
        "variable": "<unavailable>",
        "regions": [],
        "processes": [],
        "sanitized_processes": [],
        "systematics": [],
    }
    cmd = [
        config.python,
        str(script),
        "--nominal-scaled",
        str(nominal),
        "--output-root",
        str(output_root),
        "--met-unclustered-up",
        str(external["metUnclusteredUp"]),
        "--met-unclustered-down",
        str(external["metUnclusteredDown"]),
        "--jes-total-up",
        str(external["jesTotalUp"]),
        "--jes-total-down",
        str(external["jesTotalDown"]),
        "--jer-up",
        str(external["jerUp"]),
        "--jer-down",
        str(external["jerDown"]),
    ]

    print("template automation context:")
    print(f"  script: {script}")
    print(f"  nominal scaled: {nominal}")
    for name, path in external.items():
        print(f"  external {name}: {path}")
    print(f"  output ROOT: {output_root}")
    print(f"  variable: {meta['variable']}")
    print("  regions:")
    for region in meta["regions"]:
        print(f"    - {region}")
    print("  processes:")
    for process in meta["processes"]:
        print(f"    - {process}")
    print("  systematic names:")
    for systematic in meta["systematics"]:
        print(f"    - {systematic}")
    print("  ROOT key pattern: <region>/<sanitized_process> and <region>/<sanitized_process>_<systematic>Up/Down")
    print("exact subprocess command:")
    print(f"  cwd: {config.repo_root}")
    print(f"  argv: {cmd}")

    if missing:
        _print_missing(missing)
        return 1
    if dry_run:
        return 0
    output_root.parent.mkdir(parents=True, exist_ok=True)
    rc = run_command(cmd, cwd=config.repo_root)
    if rc != 0:
        return rc
    return command_validate_template(config)


def command_validate_template(config: AnalysisConfig) -> int:
    template_cfg = _require_section(config, "template")
    script = _config_path(config, template_cfg.get("script", "analysis/make_template.py"))
    output_root = _config_path(config, template_cfg.get("output_root", "analysis/templates/templates_metpt.root"))
    if not script.is_file():
        print(f"[ERROR] template script missing: {script}", file=sys.stderr)
        return 1
    if not output_root.is_file():
        print(f"[ERROR] template ROOT missing: {output_root}", file=sys.stderr)
        return 1

    meta = _load_template_module(script)
    issues: List[Tuple[str, str]] = []
    warnings: List[str] = []

    try:
        import numpy as np
        import uproot
        root_file = uproot.open(output_root)
    except Exception as exc:
        print(f"[ERROR] Could not open ROOT file {output_root}: {exc}", file=sys.stderr)
        return 1

    with root_file as handle:
        for region in meta["regions"]:
            if region not in handle:
                issues.append(("ERROR", f"missing region directory: {region}"))
                continue
            directory = handle[region]
            keys = set(directory.keys(cycle=False))
            for name in [region] + list(keys):
                if not _valid_datacard_name(name):
                    issues.append(("ERROR", f"unsupported datacard characters in object name: {region}/{name}"))

            for process in meta["sanitized_processes"]:
                if process not in keys:
                    issues.append(("ERROR", f"missing nominal object: {region}/{process}"))
                    continue
                nominal_values, nominal_variances, nominal_edges = _root_hist_arrays(directory[process])
                _check_hist_arrays(issues, warnings, f"{region}/{process}", nominal_values, nominal_variances)
                total = float(np.sum(nominal_values))
                if total < 0:
                    warnings.append(f"negative total yield: {region}/{process} = {total}")

                for systematic in meta["systematics"]:
                    for direction in ("Up", "Down"):
                        shape_name = f"{process}_{systematic}{direction}"
                        if shape_name not in keys:
                            issues.append(("ERROR", f"missing systematic object: {region}/{shape_name}"))
                            continue
                        values, variances, edges = _root_hist_arrays(directory[shape_name])
                        _check_hist_arrays(issues, warnings, f"{region}/{shape_name}", values, variances)
                        if len(edges) != len(nominal_edges) or not np.allclose(edges, nominal_edges, rtol=0, atol=1e-9):
                            issues.append(("ERROR", f"binning mismatch: {region}/{shape_name}"))

    for warning in warnings:
        print(f"[WARNING] {warning}")
    for status, message in issues:
        print(f"[{status}] {message}")
    if any(status == "ERROR" for status, _ in issues):
        return 1
    print(f"[OK] template ROOT validation passed: {output_root}")
    return 0




def command_status(config: AnalysisConfig, shift_name: str) -> int:
    status = inspect_production_status(config, shift_name)
    print_status_summary(status)
    path = write_status_json(config, shift_name, status)
    print(f"status JSON: {path}")
    return 0 if status["missing_count"] == 0 and status["unreadable_count"] == 0 and status["zero_size_count"] == 0 else 1


def command_missing(config: AnalysisConfig, shift_name: str) -> int:
    status = inspect_production_status(config, shift_name)
    failed = failed_status_entries(status)
    print_status_summary(status)
    print("failed entries:")
    for entry in failed:
        print(f"  - {entry['dataset']} [{entry['state']}] {entry.get('path', '')}")
    write_status_json(config, shift_name, status)
    return 0


def command_retry(config: AnalysisConfig, shift_name: str, max_jobs: Optional[int], dry_run: bool = False) -> int:
    if max_jobs is not None and max_jobs <= 0:
        print("[ERROR] --max-jobs must be a positive integer", file=sys.stderr)
        return 2
    validation_rc = validate_before_execution(config)
    if validation_rc != 0:
        return validation_rc
    resolved = resolve_paths(config, shift_name)
    status = inspect_production_status(config, shift_name)
    failed = failed_status_entries(status)
    selected = [entry["dataset"] for entry in failed]
    if max_jobs is not None:
        selected = selected[:max_jobs]
    print_status_summary(status)
    if not selected:
        print("[OK] No missing/unreadable/failed entries require retry.")
        return 0
    submit_file = choose_submit_file(config, resolved, shift_name, selected, len(selected), dry_run=dry_run)
    submit_cmd = [str(config.data["condor"].get("submit_command", "condor_submit")), str(submit_file)]
    print(f"retry queue entries: {len(selected)}")
    for item in selected:
        print(f"  - {item}")
    print("generated retry submit description:")
    print(build_generated_submit_text(resolved, shift_name, selected))
    print(f"exact subprocess command: {submit_cmd}")
    if dry_run:
        return 0
    prepare_condor_output_dirs(resolved)
    write_generated_submit_file(resolved, shift_name, selected, submit_file)
    return run_command(submit_cmd, cwd=resolved.condor_dir)


def command_datacard(config: AnalysisConfig, signal: Optional[str], dry_run: bool = False) -> int:
    datacard_cfg = config.data.get("datacard", {})
    template_cfg = _require_section(config, "template")
    template_root = _config_path(config, template_cfg.get("output_root", "analysis/templates/templates_metpt.root"))
    output_dir = _config_path(config, datacard_cfg.get("output_dir", "analysis/datacards"))
    output_path = output_dir / datacard_cfg.get("output_name", "stop_2024_shapes.txt")
    if signal is None:
        signal = datacard_cfg.get("signal")
    print("datacard automation context:")
    print(f"  template ROOT: {template_root}")
    print(f"  output datacard: {output_path}")
    print(f"  selected signal: {signal if signal else '<unset>'}")
    print("  model policy: minimal shape card only; no inferred rateParams/transfer factors/nuisance correlations")
    if not template_root.is_file():
        print(f"[ERROR] template ROOT missing: {template_root}", file=sys.stderr)
        return 1
    if signal is None:
        print("[ERROR] datacard.signal is not configured and --signal was not supplied; refusing to guess a signal.", file=sys.stderr)
        return 1
    rc = command_validate_template(config)
    if rc != 0:
        return rc
    if dry_run:
        print("[DRY-RUN] would write minimal shape datacard")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    text = build_minimal_datacard(config, template_root, signal)
    output_path.write_text(text, encoding="utf-8")
    print(f"[OK] wrote datacard: {output_path}")
    return 0


def command_limit(config: AnalysisConfig, signal: str, dry_run: bool = False) -> int:
    limit_cfg = config.data.get("limit", {})
    datacard_cfg = config.data.get("datacard", {})
    card = _config_path(config, limit_cfg.get("datacard", Path(datacard_cfg.get("output_dir", "analysis/datacards")) / datacard_cfg.get("output_name", "stop_2024_shapes.txt")))
    output_dir = _config_path(config, limit_cfg.get("output_dir", "analysis/limits"))
    combine = str(limit_cfg.get("combine", "combine"))
    cmd = [combine, "-M", "AsymptoticLimits", "--run", "blind", "-n", f"_{signal}", str(card)]
    print("limit automation context:")
    print(f"  datacard: {card}")
    print(f"  signal: {signal}")
    print(f"  output dir: {output_dir}")
    print(f"  argv: {cmd}")
    if not card.is_file():
        print(f"[ERROR] datacard missing: {card}", file=sys.stderr)
        return 1
    if dry_run:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=str(output_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    payload = {
        "signal": signal,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "expected": parse_combine_expected(completed.stdout),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out = output_dir / f"limit_{signal}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"limit JSON: {out}")
    return int(completed.returncode)


def command_report(config: AnalysisConfig, dry_run: bool = False) -> int:
    report_cfg = config.data.get("report", {})
    output = _config_path(config, report_cfg.get("output", "analysis/report/index.html"))
    statuses = []
    for shift in sorted(config.shifts):
        cached = load_cached_status(config, shift)
        if cached is not None:
            statuses.append(cached)
        else:
            statuses.append({"shift": shift, "error": "status JSON missing; run automation.cli status for this shift"})
    html = build_static_report(config, statuses)
    print(f"report output: {output}")
    if dry_run:
        print(html[:2000])
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"[OK] wrote report: {output}")
    return 0


def command_pipeline(config: AnalysisConfig, shift_name: str, all_shifts: bool, from_stage: Optional[str], to_stage: Optional[str], force: bool, allow_partial: bool, signal: Optional[str], dry_run: bool = False) -> int:
    shifts = sorted(config.shifts) if all_shifts else [shift_name]
    stages = ["validate", "compile", "status", "submit", "postprocess", "plot", "template", "validate-template", "datacard", "limit", "report"]
    selected = slice_stages(stages, from_stage, to_stage)
    print(f"pipeline stages: {selected}")
    for shift in shifts:
        print(f"=== shift {shift} ===")
        for stage in selected:
            if stage == "validate":
                rc = command_validate(config)
            elif stage == "compile":
                rc = command_compile(config, shift, dry_run=dry_run, force=force)
            elif stage == "status":
                command_status(config, shift)
                rc = 0
            elif stage == "submit":
                status = inspect_production_status(config, shift)
                if status.get("active_condor_jobs", 0) or status.get("held_condor_jobs", 0):
                    print("[INFO] Condor jobs remain active; exiting in resumable state.")
                    print_status_summary(status)
                    return 0
                if failed_status_entries(status):
                    rc = command_retry(config, shift, None, dry_run=True if dry_run else False)
                else:
                    print("[OK] no submit/retry needed")
                    rc = 0
            elif stage == "postprocess":
                rc = command_postprocess(config, shift, dry_run=dry_run, allow_partial=allow_partial, force=force)
            else:
                # Run global stages only once after the requested shift loop.
                rc = 0
            if rc != 0:
                return rc
    global_stages = [stage for stage in selected if stage in {"plot", "template", "validate-template", "datacard", "limit", "report"}]
    for stage in global_stages:
        if stage == "plot":
            rc = command_plot(config, nominal_only=True, dry_run=dry_run)
        elif stage == "template":
            rc = command_template(config, dry_run=dry_run)
        elif stage == "validate-template":
            rc = 0 if dry_run else command_validate_template(config)
        elif stage == "datacard":
            rc = command_datacard(config, signal=signal, dry_run=dry_run)
        elif stage == "limit":
            if not signal:
                print("[INFO] skipping limit: --signal not supplied")
                rc = 0
            else:
                rc = command_limit(config, signal=signal, dry_run=dry_run)
        elif stage == "report":
            rc = command_report(config, dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def validate_coffea_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from coffea.util import load
        load(str(path))
        return True
    except Exception:
        return False


def classify_futures(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "unreadable"
    if path.stat().st_size == 0:
        return "zero-size"
    return "complete" if validate_coffea_file(path) else "unreadable"


def inspect_production_status(config: AnalysisConfig, shift_name: str) -> Dict[str, Any]:
    resolved = resolve_paths(config, shift_name)
    expected = load_dataset_list(resolved.dataset_list_file)
    entries = []
    seen = set()
    duplicate_expected = sorted({name for name in expected if name in seen or seen.add(name)})
    for dataset in expected:
        path = resolved.futures_dir / f"{dataset}.futures"
        state = classify_futures(path)
        entries.append({"dataset": dataset, "path": str(path), "state": state})
    expected_set = set(expected)
    produced_files = sorted(resolved.futures_dir.glob("*.futures")) if resolved.futures_dir.is_dir() else []
    unexpected = [path.name[:-8] for path in produced_files if path.name.endswith(".futures") and path.name[:-8] not in expected_set]
    counts = {state: sum(1 for entry in entries if entry["state"] == state) for state in ("complete", "missing", "unreadable", "zero-size")}
    condor = inspect_condor_jobs(shift_name)
    completion = 100.0 * counts["complete"] / len(expected) if expected else 0.0
    return {
        "shift": shift_name,
        "processor_name": str(config.shift(shift_name)["processor_name"]),
        "dataset_list": str(resolved.dataset_list_file),
        "futures_dir": str(resolved.futures_dir),
        "expected_count": len(expected),
        "readable_output_count": counts["complete"],
        "missing_count": counts["missing"],
        "unreadable_count": counts["unreadable"],
        "zero_size_count": counts["zero-size"],
        "completion_percentage": completion,
        "active_condor_jobs": condor["active"],
        "held_condor_jobs": condor["held"],
        "condor_error": condor.get("error"),
        "duplicate_expected": duplicate_expected,
        "unexpected": unexpected,
        "entries": entries,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def inspect_condor_jobs(shift_name: str) -> Dict[str, Any]:
    if shutil.which("condor_q") is None:
        return {"active": 0, "held": 0, "error": "condor_q not found"}
    cmd = ["condor_q", "-json"]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return {"active": 0, "held": 0, "error": completed.stderr.strip()}
    try:
        jobs = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return {"active": 0, "held": 0, "error": f"condor_q JSON parse failed: {exc}"}
    active = 0
    held = 0
    for job in jobs:
        args = str(job.get("Args", ""))
        if shift_name not in args:
            continue
        status = int(job.get("JobStatus", 0) or 0)
        if status == 5:
            held += 1
        elif status in (1, 2):
            active += 1
    return {"active": active, "held": held}


def failed_status_entries(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [entry for entry in status.get("entries", []) if entry.get("state") in {"missing", "unreadable", "zero-size"}]


def print_status_summary(status: Dict[str, Any]) -> None:
    print(f"shift: {status['shift']}")
    print(f"expected count: {status['expected_count']}")
    print(f"readable output count: {status['readable_output_count']}")
    print(f"missing count: {status['missing_count']}")
    print(f"unreadable count: {status['unreadable_count']}")
    print(f"zero-size count: {status['zero_size_count']}")
    print(f"completion percentage: {status['completion_percentage']:.2f}%")
    print(f"active Condor jobs: {status['active_condor_jobs']}")
    print(f"held Condor jobs: {status['held_condor_jobs']}")
    if status.get("condor_error"):
        print(f"condor status warning: {status['condor_error']}")
    if status.get("unexpected"):
        print("unexpected outputs:")
        for item in status["unexpected"][:50]:
            print(f"  - {item}")
    failed = failed_status_entries(status)
    if failed:
        print("failed entries:")
        for entry in failed[:50]:
            print(f"  - {entry['dataset']} [{entry['state']}]")




def load_cached_status(config: AnalysisConfig, shift_name: str) -> Optional[Dict[str, Any]]:
    path = config.repo_root / "analysis" / "hists" / "status" / f"stop_2024_{shift_name}_status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"shift": shift_name, "error": f"could not read cached status {path}: {exc}"}

def write_status_json(config: AnalysisConfig, shift_name: str, status: Dict[str, Any]) -> Path:
    outdir = config.repo_root / "analysis" / "hists" / "status"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"stop_2024_{shift_name}_status.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_reduced_outputs(resolved: ResolvedPaths) -> bool:
    files = sorted(resolved.futures_dir.glob("*.reduced")) if resolved.futures_dir.is_dir() else []
    if not files:
        print(f"[ERROR] no reduced outputs found in {resolved.futures_dir}", file=sys.stderr)
        return False
    bad = [path for path in files if not validate_coffea_file(path)]
    for path in bad:
        print(f"[ERROR] unreadable reduced output: {path}", file=sys.stderr)
    return not bad


def quarantine_stale_postprocess_outputs(resolved: ResolvedPaths) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = list(resolved.futures_dir.glob("*.reduced")) + list(resolved.futures_dir.glob("*.merged")) + [resolved.merged_file, resolved.scaled_file]
    qdir = resolved.futures_dir / f"stale_{stamp}"
    for path in candidates:
        if path.exists():
            qdir.mkdir(parents=True, exist_ok=True)
            target = qdir / path.name
            print(f"quarantine stale artifact: {path} -> {target}")
            path.rename(target)


def build_minimal_datacard(config: AnalysisConfig, template_root: Path, signal: str) -> str:
    template_cfg = _require_section(config, "template")
    meta = _load_template_module(_config_path(config, template_cfg.get("script", "analysis/make_template.py")))
    regions = meta["regions"]
    processes = [p for p in meta["sanitized_processes"] if p != "data_obs"]
    if signal not in processes:
        raise ConfigError(f"Configured signal '{signal}' is not present in template process list: {processes}")
    backgrounds = [p for p in processes if p != signal and not p.startswith("SMS")]
    lines = [
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {template_root} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
        "------------",
        "bin " + " ".join(regions),
        "observation " + " ".join(["-1"] * len(regions)),
        "------------",
    ]
    bin_cols=[]; proc_cols=[]; idx_cols=[]; rate_cols=[]
    for region in regions:
        ordered=[signal]+backgrounds
        for idx, proc in enumerate(ordered):
            bin_cols.append(region); proc_cols.append(proc); idx_cols.append(str(-idx if idx==0 else idx)); rate_cols.append("-1")
    lines.append("bin " + " ".join(bin_cols))
    lines.append("process " + " ".join(proc_cols))
    lines.append("process " + " ".join(idx_cols))
    lines.append("rate " + " ".join(rate_cols))
    lines.append("------------")
    lines.append(f"{LUMI_UNCERTAINTY_NAME} lnN " + " ".join([f"{LUMI_UNCERTAINTY_LNN:.3f}"] * len(bin_cols)))
    lines.append("* autoMCStats 10")
    return "\n".join(lines) + "\n"


def parse_combine_expected(stdout: str) -> Dict[str, float]:
    out = {}
    for line in stdout.splitlines():
        m = re.search(r"Expected\s+(\d+\.\d+)%:\s+r\s+<\s+([0-9.eE+-]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def build_static_report(config: AnalysisConfig, statuses: List[Dict[str, Any]]) -> str:
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(config.repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout.strip()
    except Exception:
        git = "unavailable"
    rows = []
    for st in statuses:
        if "error" in st:
            rows.append(f"<tr><td>{st['shift']}</td><td colspan='5'>{st['error']}</td></tr>")
        else:
            rows.append("<tr><td>{shift}</td><td>{expected_count}</td><td>{readable_output_count}</td><td>{missing_count}</td><td>{unreadable_count}</td><td>{completion_percentage:.2f}%</td></tr>".format(**st))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Stop 2024 Automation Report</title></head><body>
<h1>Stop 2024 Automation Report</h1>
<p>Year: {config.year}</p><p>Luminosity: {config.data.get('plotting',{}).get('luminosity_fb','unknown')} fb-1</p><p>Git commit: {git}</p>
<h2>Production</h2><table border='1'><tr><th>shift</th><th>expected</th><th>readable</th><th>missing</th><th>unreadable</th><th>completion</th></tr>{''.join(rows)}</table>
<h2>Warnings</h2><p>Report generated from existing artifacts only. No observed limit is run by automation.</p>
</body></html>"""


def slice_stages(stages: List[str], from_stage: Optional[str], to_stage: Optional[str]) -> List[str]:
    start = stages.index(from_stage) if from_stage else 0
    end = stages.index(to_stage) + 1 if to_stage else len(stages)
    return stages[start:end]

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



def _require_section(config: AnalysisConfig, section: str) -> Dict[str, Any]:
    value = config.data.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required mapping: {section}")
    return value


def _config_path(config: AnalysisConfig, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (config.repo_root / path).absolute()


def _missing_files(paths: Dict[str, Path]) -> Dict[str, Path]:
    return {label: path for label, path in paths.items() if not path.is_file()}


def _print_missing(missing: Dict[str, Path]) -> None:
    print("[ERROR] Missing required input files before execution:", file=sys.stderr)
    for label, path in missing.items():
        print(f"  - {label}: {path}", file=sys.stderr)


def _inspect_scaled_axes(path: Path) -> Tuple[List[str], List[str]]:
    try:
        from coffea.util import load
        obj = load(path)
        bkg = obj["bkg"]
    except Exception as exc:
        raise ConfigError(f"Could not inspect scaled file {path}: {exc}") from exc

    variables = [key for key in bkg.keys() if "sumw" not in key and "template" not in key and "nPV" not in key]
    regions: List[str] = []
    for variable in bkg.keys():
        for process in bkg[variable].keys():
            try:
                axis = bkg[variable][process].axes["region"]
                regions = [str(item) for item in axis]
                return variables, regions
            except Exception:
                continue
    return variables, regions


def _load_template_module(script: Path) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("phase3_make_template", script)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Could not load template script metadata: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processes = list(module.BKG_PROCESSES) + list(module.SIG_PROCESSES)
    systematics = list(module.SHAPE_SYSTS.keys()) + list(module.EXTERNAL_SHAPE_SYSTS.keys())
    return {
        "variable": module.VARIABLE,
        "regions": list(module.REGIONS),
        "processes": processes,
        "sanitized_processes": [module.sanitize_name(process) for process in processes],
        "systematics": systematics,
    }


def _valid_datacard_name(name: str) -> bool:
    return re.match(r"^[A-Za-z0-9_]+$", name) is not None


def _root_hist_arrays(obj: Any) -> Tuple[Any, Any, Any]:
    import numpy as np
    try:
        values = np.asarray(obj.values(flow=False), dtype=float)
    except TypeError:
        values = np.asarray(obj.values(), dtype=float)
    try:
        variances = obj.variances(flow=False)
    except TypeError:
        variances = obj.variances()
    except Exception:
        variances = None
    if variances is not None:
        variances = np.asarray(variances, dtype=float)
    else:
        variances = np.zeros_like(values, dtype=float)
    try:
        edges = np.asarray(obj.axis().edges(), dtype=float)
    except Exception:
        _, edges = obj.to_numpy()
        edges = np.asarray(edges, dtype=float)
    return values, variances, edges


def _check_hist_arrays(issues: List[Tuple[str, str]], warnings: List[str], name: str, values: Any, variances: Any) -> None:
    import numpy as np
    if not np.all(np.isfinite(values)):
        issues.append(("ERROR", f"non-finite histogram contents: {name}"))
    if not np.all(np.isfinite(variances)):
        issues.append(("ERROR", f"non-finite histogram variances: {name}"))
    if np.any(variances < 0):
        warnings.append(f"negative bin variances: {name}")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def _relative_to_analysis(path: Path, analysis_dir: Path) -> str:
    try:
        return str(path.relative_to(analysis_dir))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
