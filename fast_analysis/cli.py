from __future__ import annotations

import argparse
import json
import sys

from .benchmark import benchmark_json
from .config.defaults import DEFAULTS
from .env import environment_json
from .paths import PathKind, PathPolicy, configure_eos_runtime_env
from .reporting.static import render_index
from .validation.compare import compare_scaled_files_json
from .validation.reference import validate_scaled_reference_json
from .workflow.manifest import initialize


def _add_common(parser):
    parser.add_argument("--dry-run", action="store_true", help="print resolved actions without writing files")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python -m fast_analysis.cli")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    inspect_p = sub.add_parser("inspect")
    _add_common(inspect_p)

    env_p = sub.add_parser("env-check")
    _add_common(env_p)

    ref_p = sub.add_parser("validate-reference")
    _add_common(ref_p)
    ref_p.add_argument("--reference-scaled", default=str(DEFAULTS.legacy_scaled_reference))
    ref_p.add_argument("--variable", default="recoilpt")
    ref_p.add_argument("--region", default=None)

    bench_p = sub.add_parser("benchmark")
    _add_common(bench_p)
    bench_p.add_argument("--manifest", default=str(DEFAULTS.benchmark_manifest))
    bench_p.add_argument("--max-events", type=int, default=10000)

    plan_p = sub.add_parser("plan")
    _add_common(plan_p)

    status_p = sub.add_parser("status")
    _add_common(status_p)

    validate_p = sub.add_parser("validate")
    _add_common(validate_p)
    validate_p.add_argument("--manifest-db", default=str(DEFAULTS.manifest_path))

    report_p = sub.add_parser("report")
    _add_common(report_p)
    report_p.add_argument("--output", default=str(DEFAULTS.report_path))

    compare_p = sub.add_parser("compare")
    _add_common(compare_p)
    compare_p.add_argument("--fast-output", required=True)
    compare_p.add_argument("--reference-scaled", default=str(DEFAULTS.legacy_scaled_reference))
    compare_p.add_argument("--variable", default="recoilpt")
    compare_p.add_argument("--region", default=None)
    compare_p.add_argument("--output-json", default=None)

    for name in ("submit", "retry", "yields", "plot", "template", "card", "limit"):
        p = sub.add_parser(name)
        _add_common(p)

    args = parser.parse_args(argv)
    policy = PathPolicy.default()
    print("approved EOS root: %s" % DEFAULTS.repo_root)
    print("fixed python: %s" % DEFAULTS.fixed_python)
    print("temporary directory: %s" % (DEFAULTS.output_root / "tmp"))
    print("cache directory: %s" % (DEFAULTS.output_root / "cache"))
    print("manifest path: %s" % DEFAULTS.manifest_path)
    print("log path: %s" % (DEFAULTS.output_root / "logs"))

    if args.command == "inspect":
        resolved = {kind.value: str(DEFAULTS.output_root) for kind in (PathKind.OUTPUT,)}
        print(json.dumps({"defaults": _defaults_dict(), "resolved": resolved}, indent=2, sort_keys=True))
        return 0
    if args.command == "env-check":
        payload = json.loads(environment_json(dry_run=args.dry_run))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["ok"] else 1
    if args.command == "validate-reference":
        payload = json.loads(validate_scaled_reference_json(args.reference_scaled, args.variable, args.region))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["ok"] else 1
    if args.command == "benchmark":
        print(benchmark_json(args.manifest, dry_run=args.dry_run, max_events=args.max_events))
        return 0
    if args.command == "compare":
        payload = json.loads(compare_scaled_files_json(args.fast_output, args.reference_scaled, args.variable, args.output_json, args.dry_run, args.region))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("status") == "ok" else 1
    if args.command == "plan":
        print(json.dumps(_plan(args.dry_run), indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        configure_eos_runtime_env(DEFAULTS.output_root, dry_run=args.dry_run)
        db_path = initialize(args.manifest_db, dry_run=args.dry_run)
        print(json.dumps({"status": "dry-run" if args.dry_run else "ok", "manifest_db": str(db_path)}, indent=2))
        return 0
    if args.command == "report":
        output = render_index(args.output, dry_run=args.dry_run)
        print(json.dumps({"status": "dry-run" if args.dry_run else "ok", "report": str(output)}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps({"status": "prototype", "full_campaign_submitted": False, "observed_limit_run": False}, indent=2))
        return 0
    print("%s: scaffolded command; physics implementation is gated behind benchmark/regression validation." % args.command)
    return 2


def _defaults_dict():
    return {
        "year": DEFAULTS.year,
        "luminosity_fb": DEFAULTS.luminosity_fb,
        "maturity": list(DEFAULTS.maturity),
        "repo_root": str(DEFAULTS.repo_root),
        "output_root": str(DEFAULTS.output_root),
        "fixed_python": str(DEFAULTS.fixed_python),
        "environment_path": str(DEFAULTS.environment_path),
        "legacy_scaled_reference": str(DEFAULTS.legacy_scaled_reference),
        "recoil_bins": list(DEFAULTS.recoil_bins),
        "target_regions": list(DEFAULTS.target_regions),
    }


def _plan(dry_run):
    return {
        "dry_run": dry_run,
        "chosen_environment": "fixed existing EOS py38 environment",
        "environment_path": str(DEFAULTS.environment_path),
        "chosen_output_format": "selected at runtime from installed modules: Parquet if PyArrow exists, otherwise flat ROOT with uproot",
        "chunking_strategy": "first benchmark is one file per role; later target 5-15 minutes/job",
        "no_full_campaign": True,
        "next_command": "%s -m fast_analysis.cli env-check" % DEFAULTS.fixed_python,
    }


if __name__ == "__main__":
    sys.exit(main())
