from __future__ import annotations

import argparse
import json

from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysisctl")
    sub = parser.add_subparsers(dest="command", required=True)
    config_commands = (
        "all", "resume", "real-subset", "validate-real-subset", "validate-feature-subset",
        "run-production", "full-production-normalization", "select-search-bins",
        "discover-signals-from-das", "process-signals", "make-hists-npy", "plot-from-npy",
        "publish-github-pages", "monitor", "design-search-bins", "normalization-audit",
        "normalize-feature-yields", "make-feature-yields", "make-systematic-yields",
        "make-datacards", "expected-limits",
    )
    for name in config_commands:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "monitor":
            p.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    pipeline = Pipeline.from_config(args.config, resume=args.command == "resume")
    if args.command in {"real-subset", "validate-real-subset", "validate-feature-subset"}:
        pipeline.run_real_subset()
    elif args.command == "monitor":
        payload = pipeline.monitor(json_output=args.json_output)
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(pipeline.monitor_text(payload))
    elif args.command == "publish-github-pages":
        payload = pipeline.publish_github_pages()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "design-search-bins":
        payload = pipeline.design_search_bins()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "normalization-audit":
        payload = pipeline.normalization_audit()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "run-production":
        payload = pipeline.run_production()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "full-production-normalization":
        payload = pipeline.full_production_normalization()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "select-search-bins":
        payload = pipeline.select_search_bins()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "discover-signals-from-das":
        payload = pipeline.discover_signals_from_das()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "process-signals":
        payload = pipeline.process_signals()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "make-hists-npy":
        payload = pipeline.make_hists_npy()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plot-from-npy":
        payload = pipeline.plot_from_npy()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "normalize-feature-yields":
        payload = pipeline.normalize_feature_yields()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "make-feature-yields":
        payload = pipeline.make_feature_yields()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "make-systematic-yields":
        payload = pipeline.make_systematic_yields()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "make-datacards":
        payload = pipeline.make_datacards_stage()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "expected-limits":
        payload = pipeline.expected_limits_stage()
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        pipeline.run_all()
    return 0
