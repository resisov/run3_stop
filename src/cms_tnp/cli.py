"""Command-line interface for the complete tag-and-probe workflow."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .config import load_config, validate_config
from .profiles import PROFILES


def _read(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def _write(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as target:
            target.write(encoded.encode())
    else:
        path.write_text(encoded)


def _template(profile: str) -> dict[str, Any]:
    resolved = PROFILES[profile]
    return {
        "schema_version": 1,
        "profile": profile,
        "measurement": f"private_{profile}_sf",
        "year": "2025",
        "id": {
            "fields": [],
            "denominator": resolved["probe"]["selection"],
            "pass": resolved["probe"]["pass"],
        },
        "pt_edges_gev": resolved["axes"]["pt_edges_gev"],
        "abseta_edges": resolved["axes"]["abseta_edges"],
        "samples": {"data": [], "mc": []},
        "lumimask": "golden.json",
        "correction": {
            "name": f"private_{profile}_sf",
            "description": f"Data/MC scale factor for private_{profile}",
            "flow": "clamp",
        },
    }


def _dependencies() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "numpy",
            "awkward",
            "uproot",
            "scipy",
            "correctionlib",
            "matplotlib",
            "mplhep",
        )
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("profiles")
    command = commands.add_parser("init")
    command.add_argument(
        "--profile", choices=sorted(PROFILES), default="electron_jpsi_lowpt"
    )
    command.add_argument("--output", type=Path, default=Path("measurement.json"))

    command = commands.add_parser("resolve")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("doctor")
    command.add_argument("--config", type=Path)

    command = commands.add_parser("discover")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--sample", choices=("data", "mc"), required=True)
    command.add_argument("--dasgoclient", default="dasgoclient")
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("make-shards")
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--files-per-shard", type=int, default=20)
    command.add_argument("--output-dir", type=Path, required=True)

    command = commands.add_parser("count")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--sample", choices=("data", "mc"), required=True)
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", type=Path)
    group.add_argument("--shard", type=Path)
    command.add_argument("--step-size", type=int, default=100_000)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("reduce")
    command.add_argument("inputs", nargs="+", type=Path)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("fit")
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("export")
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("plot")
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)

    command = commands.add_parser("reproduce")
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)

    command = commands.add_parser("run-local")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--data-files", type=Path, required=True)
    command.add_argument("--mc-files", type=Path, required=True)
    command.add_argument("--step-size", type=int, default=100_000)
    command.add_argument("--output-dir", type=Path, required=True)

    command = commands.add_parser("condor-prepare")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--data-shards", type=Path, required=True)
    command.add_argument("--mc-shards", type=Path, required=True)
    command.add_argument("--environment", type=Path, required=True)
    command.add_argument("--proxy", type=Path)
    command.add_argument("--campaign-dir", type=Path, required=True)
    command.add_argument("--request-cpus", type=int, default=1)
    command.add_argument("--request-memory-mb", type=int, default=4000)
    command.add_argument("--request-disk-mb", type=int, default=4000)
    command.add_argument("--job-flavour")

    command = commands.add_parser("condor-submit")
    command.add_argument("--campaign-dir", type=Path, required=True)
    command.add_argument("--submit-command", default="condor_submit")
    command.add_argument("--resubmit", action="store_true")

    command = commands.add_parser("condor-status")
    command.add_argument("--campaign-dir", type=Path, required=True)
    command.add_argument("--query-command", default="condor_q")

    command = commands.add_parser("condor-finalize")
    command.add_argument("--campaign-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "profiles":
        print("\n".join(sorted(PROFILES)))
        return 0
    if args.command == "init":
        if args.output.exists():
            raise FileExistsError(args.output)
        _write(args.output, _template(args.profile))
        print(args.output)
        return 0
    if args.command == "resolve":
        config = load_config(args.config)
        _write(args.output, config)
        return 0
    if args.command == "doctor":
        config_status: Any = None
        if args.config:
            try:
                config = load_config(args.config)
                validate_config(config)
                config_status = {
                    "valid": True,
                    "measurement": config["measurement"],
                    "profile": config.get("profile"),
                }
            except (OSError, ValueError, KeyError, TypeError) as error:
                config_status = {
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dependencies = _dependencies()
        output = {
            "python": sys.version,
            "profiles": sorted(PROFILES),
            "dependencies": dependencies,
            "config": config_status,
            "ready": all(dependencies.values())
            and (config_status is None or config_status["valid"]),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["ready"] else 2
    if args.command == "discover":
        from .records import discover

        _write(
            args.output,
            discover(load_config(args.config), args.sample, args.dasgoclient),
        )
        return 0
    if args.command == "make-shards":
        from .records import shards

        args.output_dir.mkdir(parents=True, exist_ok=True)
        items = shards(_read(args.records), args.files_per_shard)
        for item in items:
            _write(args.output_dir / f"shard_{int(item['shard_id']):05d}.json", item)
        print(
            json.dumps(
                {"shards": len(items), "output_dir": str(args.output_dir)}, indent=2
            )
        )
        return 0
    if args.command == "count":
        from .count import count_files, read_file_list
        from .records import shard_files

        files = (
            read_file_list(args.files) if args.files else shard_files(_read(args.shard))
        )
        output = count_files(
            load_config(args.config),
            files,
            args.sample,
            step_size=args.step_size,
            base_dir=args.config.parent,
        )
        _write(args.output, output)
        return 0 if output["processing"]["files_processed"] else 2
    if args.command == "reduce":
        from .reduce import merge

        output = merge([_read(path) for path in args.inputs])
        _write(args.output, output)
        return 0 if output["status"] == "complete" else 2
    if args.command == "fit":
        from .fit import fit_payload

        output = fit_payload(_read(args.histograms))
        _write(args.output, output)
        return 0 if all(item.get("valid") for item in output["bins"]) else 2
    if args.command == "export":
        from .payload import build_payload, write_payload

        payload = build_payload(_read(args.result))
        digest = write_payload(args.output, payload)
        print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))
        return 0
    if args.command == "plot":
        from .plot import plot_result

        print(json.dumps(plot_result(_read(args.result), args.output_dir), indent=2))
        return 0
    if args.command == "reproduce":
        from .fit import fit_payload
        from .payload import build_payload, write_payload
        from .plot import plot_result

        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = fit_payload(_read(args.histograms))
        if not all(item.get("valid") for item in result["bins"]):
            _write(args.output_dir / "fit_result.json", result)
            return 2
        _write(args.output_dir / "fit_result.json", result)
        write_payload(args.output_dir / "scale_factors.json.gz", build_payload(result))
        plot_result(result, args.output_dir / "plots")
        return 0
    if args.command == "run-local":
        from .count import count_files, read_file_list
        from .fit import fit_payload
        from .payload import build_payload, write_payload
        from .plot import plot_result
        from .reduce import merge

        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = load_config(args.config)
        data = count_files(
            config,
            read_file_list(args.data_files),
            "data",
            step_size=args.step_size,
            base_dir=args.config.parent,
        )
        mc = count_files(
            config,
            read_file_list(args.mc_files),
            "mc",
            step_size=args.step_size,
            base_dir=args.config.parent,
        )
        _write(args.output_dir / "data.json", data)
        _write(args.output_dir / "mc.json", mc)
        histograms = merge([data, mc])
        _write(args.output_dir / "histograms.json", histograms)
        result = fit_payload(histograms)
        _write(args.output_dir / "fit_result.json", result)
        if not all(item.get("valid") for item in result["bins"]):
            return 2
        write_payload(args.output_dir / "scale_factors.json.gz", build_payload(result))
        plot_result(result, args.output_dir / "plots")
        return 0
    if args.command == "condor-prepare":
        from .condor import prepare_campaign

        output = prepare_campaign(
            config_path=args.config,
            data_shards=args.data_shards,
            mc_shards=args.mc_shards,
            environment_path=args.environment,
            proxy_path=args.proxy,
            campaign_dir=args.campaign_dir,
            request_cpus=args.request_cpus,
            request_memory_mb=args.request_memory_mb,
            request_disk_mb=args.request_disk_mb,
            job_flavour=args.job_flavour,
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "condor-submit":
        from .condor import submit_campaign

        output = submit_campaign(
            args.campaign_dir,
            submit_command=args.submit_command,
            resubmit=args.resubmit,
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "condor-status":
        from .condor import campaign_status

        output = campaign_status(
            args.campaign_dir, query_command=args.query_command
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if output["status"] == "complete" else 2
    if args.command == "condor-finalize":
        from .condor import finalize_campaign

        output = finalize_campaign(args.campaign_dir, args.output_dir)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)
