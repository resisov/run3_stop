#!/usr/bin/env python3
"""Unified, reproducible electron/muon low-pT tag-and-probe workflow."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .build_tnp_das_records import build_records
from .audit_lowpt_reference_triggers import audit
from .export_tnp_id_correctionlib import build_payload
from .prepare_tnp_measurement_condor import prepare
from .reference_trigger_counts import json_safe
from .sf_payload import ensure_adopted, write_json_gz
from .tnp_fit import fit_histogram_payload
from .tnp_histograms import build_histograms, read_file_list
from .tnp_measurement_reduce import (
    merge_tnp_shards,
    rebin_probe_histograms,
    use_mc_reference,
)
from .tnp_measurement_shard import count_tnp_shard
from .tnp_recovery import (
    build_residual_manifest,
    finalize_permanent_skips,
    recover,
)
from .validate_tnp_adoption import validate

import correctionlib


KINDS = ("electron", "muon")
VERSIONS = ("numpy", "scipy", "awkward", "uproot", "correctionlib", "matplotlib", "mplhep")
DEFAULT_CONFIGS = {
    ("electron", "2024"): Path("configs/config_2024_id_only_parking_singlemuon.json"),
    ("electron", "2025"): Path("configs/config_2025_id_only_parking_singlemuon.json"),
    ("muon", "2024"): Path("configs/config_2024_id_only_parking_external.json"),
    ("muon", "2025"): Path("configs/config_2025_id_only_parking_external.json"),
}
REQUIRED_ASSETS = (
    Path("data/lumimasks/Cert_Collisions2024_378981_386951_Golden.json"),
    Path("data/lumimasks/Cert_Collisions2025_391658_398903_Golden.json"),
    Path("data/pileup/puWeights_2024.json.gz"),
    Path("data/pileup/puWeights_2025.json.gz"),
)
RELEASE_GLOBS = (
    "src/lowpt_tnp/*.py",
    "configs/*.json",
    "records/*.json.gz",
    "data/lumimasks/*.json",
    "data/pileup/*.json.gz",
    "reference/results/*/*.json.gz",
    "reference/payloads/*.json.gz",
    "tests/*.py",
    ".github/workflows/*.yml",
    "README.md",
    "pyproject.toml",
    "environment.yml",
    "requirements-*.txt",
)


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _release_files(repo: Path) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for pattern in RELEASE_GLOBS:
        expanded.extend(path.relative_to(repo) for path in sorted(repo.glob(pattern)))
    return tuple(dict.fromkeys(expanded))


def _plot_tnp_result(
    result_path: Path,
    histograms_path: Path,
    output_dir: Path,
    *,
    electron_endcap_unity_fallback: bool,
) -> dict[str, Any]:
    """Import matplotlib only for commands that actually render figures."""
    from .plot_measurement import plot_tnp_result

    return plot_tnp_result(
        result_path,
        histograms_path,
        output_dir,
        electron_endcap_unity_fallback=electron_endcap_unity_fallback,
    )


def _config_path(repo: Path, kind: str, year: str) -> Path:
    return repo / DEFAULT_CONFIGS[(kind, year)]


def _electron_unity_indices(result: dict[str, Any]) -> set[int]:
    n_eta = len(result["probe_abseta_edges"]) - 1
    n_pt = len(result["probe_pt_edges_gev"]) - 1
    return set(range((n_eta - 1) * n_pt, n_eta * n_pt))


def _fit_exit_code(
    result: dict[str, Any],
    *,
    kind: str,
    electron_endcap_unity_fallback: bool,
) -> int:
    invalid = {
        int(item.get("flat_index", index))
        for index, item in enumerate(result.get("bins") or [])
        if not item.get("valid")
    }
    if not invalid:
        return 0
    if (
        kind == "electron"
        and electron_endcap_unity_fallback
        and invalid <= _electron_unity_indices(result)
    ):
        return 0
    return 2


def doctor(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    versions = _dependency_versions()
    assets = {str(path): (repo / path).is_file() for path in REQUIRED_ASSETS}
    configs = {
        f"{kind}_{year}": str(_config_path(repo, kind, year))
        for kind in KINDS
        for year in ("2024", "2025")
    }
    config_checks: dict[str, Any] = {}
    for label, value in configs.items():
        path = Path(value)
        if not path.is_file():
            config_checks[label] = {"valid": False, "error": "missing"}
            continue
        payload = _read_json(path)
        expected = "veto_id_only" if label.startswith("electron") else "loose_id_only"
        config_checks[label] = {
            "valid": payload.get("probe_definition") == expected,
            "measurement": payload.get("measurement"),
            "probe_definition": payload.get("probe_definition"),
        }
    commands = {
        command: shutil.which(command)
        for command in ("dasgoclient", "condor_submit", "xrdcp", "xrdfs", "conda-pack")
    }
    python_supported = sys.version_info >= (3, 8)
    local_ready = python_supported and all(versions.values()) and all(assets.values()) and all(
        item["valid"] for item in config_checks.values()
    )
    return {
        "schema_version": 1,
        "status": "ready" if local_ready else "incomplete",
        "python": sys.version,
        "python_supported": python_supported,
        "repository": str(repo),
        "git_commit": _git_commit(repo),
        "dependencies": versions,
        "assets": assets,
        "configs": config_checks,
        "external_commands": commands,
        "local_histogram_fit_plot_ready": bool(local_ready),
        "cern_full_campaign_ready": bool(
            local_ready and all(commands[name] for name in ("dasgoclient", "condor_submit", "xrdcp", "xrdfs"))
        ),
        "created_unix": time.time(),
    }


def build_release_manifest(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    release_files = _release_files(repo)
    missing = [str(path) for path in release_files if not (repo / path).is_file()]
    if missing:
        raise FileNotFoundError(f"release inputs are missing: {missing}")
    files = [
        {
            "path": str(path),
            "bytes": (repo / path).stat().st_size,
            "sha256": _sha256(repo / path),
        }
        for path in release_files
    ]
    records: dict[str, Any] = {}
    for path in release_files:
        if path.parent != Path("records"):
            continue
        payload = _read_json(repo / path)
        records[path.name] = {
            "measurement": payload.get("measurement"),
            "year": payload.get("year"),
            "selected_samples": payload.get("selected_samples"),
            "data_files": payload.get("data_files"),
            "mc_files": payload.get("mc_files"),
            "records": len(payload.get("records") or []),
            "datasets": sorted((payload.get("dataset_audit") or {}).keys()),
        }
    results: dict[str, Any] = {}
    for kind in KINDS:
        base = repo / "reference/results" / kind
        histograms = _read_json(base / "histograms.json.gz")
        fitted = _read_json(base / "fit_result.json.gz")
        results[kind] = {
            "measurement": fitted["measurement"],
            "year": fitted["year"],
            "status": fitted.get("status"),
            "probe_definition": fitted["probe_definition"],
            "probe_abseta_edges": fitted["probe_abseta_edges"],
            "probe_pt_edges_gev": fitted["probe_pt_edges_gev"],
            "valid_bins": sum(bool(item.get("valid")) for item in fitted["bins"]),
            "bins": len(fitted["bins"]),
            "files_expected": histograms.get("files_expected"),
            "files_processed": histograms.get("files_processed"),
            "files_failed": len(histograms.get("files_failed") or []),
            "mc_reference_year": (histograms.get("mc_reference") or {}).get("year"),
            "mc_reference_datasets": (histograms.get("mc_reference") or {}).get("mc_physical_datasets"),
        }
    payloads: dict[str, Any] = {}
    for filename in ("veto_electron_5to10_sf.json.gz", "loose_muon_5to10_sf.json.gz"):
        path = repo / "reference/payloads" / filename
        correction_set = correctionlib.CorrectionSet.from_file(str(path))
        payloads[filename] = {
            "corrections": list(correction_set.keys()),
            "sha256": _sha256(path),
        }
    return {
        "schema_version": 1,
        "release": "lowpt-tnp-2025-v1",
        "scope": "2025 electron veto-ID and muon LooseID scale factors for 5 < pT < 10 GeV",
        "entrypoint": "lowpt-tnp",
        "python_reference": "3.8.20",
        "source_tree": "the git commit containing this manifest",
        "records": records,
        "results": results,
        "payloads": payloads,
        "files": files,
    }


def verify_release(repo: Path, manifest_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    expected = _read_json(manifest_path)
    failures = []
    for item in expected.get("files") or []:
        path = repo / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "reason": "missing"})
            continue
        actual = _sha256(path)
        if actual != item["sha256"]:
            failures.append({
                "path": item["path"],
                "reason": "sha256 mismatch",
                "expected": item["sha256"],
                "actual": actual,
            })
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "manifest": str(manifest_path),
        "files_checked": len(expected.get("files") or []),
        "failures": failures,
    }


def build_runtime_archive(project_root: Path, output: Path) -> dict[str, Any]:
    """Create the self-contained source/data archive transferred to Condor workers."""
    project_root = project_root.resolve()
    required = (Path("src/lowpt_tnp"), Path("configs"), Path("data"), Path("pyproject.toml"))
    missing = [str(path) for path in required if not (project_root / path).exists()]
    if missing:
        raise FileNotFoundError(f"runtime archive inputs are missing: {missing}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in required:
            archive.add(project_root / relative, arcname=str(relative), recursive=True)
    return {
        "schema_version": 1,
        "status": "created",
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "contents": [str(path) for path in required],
    }


def select_records(
    source: Path,
    output: Path,
    samples: set[str],
    *,
    source_label: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if not samples or not samples <= {"data", "mc"}:
        raise ValueError("samples must be a non-empty subset of data/mc")
    payload = _read_json(source)
    records = [item for item in payload.get("records") or [] if item.get("sample") in samples]
    if not records:
        raise ValueError(f"record selection {sorted(samples)} is empty")
    selected = dict(payload)
    selected["records"] = records
    selected["data_files"] = sum(item.get("sample") == "data" for item in records)
    selected["mc_files"] = sum(item.get("sample") == "mc" for item in records)
    selected["selected_samples"] = sorted(samples)
    selected["source_records"] = source_label or str(source)
    selected["source_sha256"] = _sha256(source)
    if config_path is not None:
        config = _read_json(config_path)
        selected["source_metadata"] = {
            key: payload.get(key)
            for key in ("measurement", "year", "probe_definition", "reference_paths")
        }
        for key in ("measurement", "year", "probe_definition", "reference_paths"):
            selected[key] = config.get(key)
        selected["release_config"] = str(config_path)
        selected["release_config_sha256"] = _sha256(config_path)
        expected_mc = set(config.get("campaign_inputs", {}).get("mc_datasets") or [])
        selected_mc = {
            str(item.get("dataset"))
            for item in records
            if item.get("sample") == "mc"
        }
        if selected_mc and selected_mc != expected_mc:
            raise ValueError(
                "selected MC datasets do not exactly match the release config: "
                f"{sorted(selected_mc)} != {sorted(expected_mc)}"
            )
    datasets = {str(item.get("dataset")) for item in records}
    selected["dataset_audit"] = {
        name: value
        for name, value in (payload.get("dataset_audit") or {}).items()
        if name in datasets
    }
    _write_json(output, selected)
    return {
        "measurement": selected.get("measurement"),
        "samples": sorted(samples),
        "data_files": selected["data_files"],
        "mc_files": selected["mc_files"],
        "output": str(output),
        "sha256": _sha256(output),
    }


def fit_result(
    histograms: Path,
    output: Path,
    *,
    kind: str,
    electron_endcap_unity_fallback: bool,
) -> tuple[dict[str, Any], int]:
    result = fit_histogram_payload(_read_json(histograms))
    _write_json(output, result)
    return result, _fit_exit_code(
        result,
        kind=kind,
        electron_endcap_unity_fallback=electron_endcap_unity_fallback,
    )


def export_result(
    result_path: Path,
    output: Path,
    *,
    kind: str,
    candidate: bool,
    electron_endcap_unity_fallback: bool,
) -> dict[str, Any]:
    result = _read_json(result_path)
    if not candidate:
        ensure_adopted(result, result_path)
    payload = build_payload(
        result,
        kind,
        electron_endcap_unity_fallback=electron_endcap_unity_fallback,
    )
    digest = write_json_gz(output, payload)
    return {
        "status": "candidate" if candidate else "installed",
        "kind": kind,
        "source_result": str(result_path),
        "output": str(output),
        "sha256": digest,
        "electron_endcap_unity_fallback": electron_endcap_unity_fallback,
    }


def render(
    result_path: Path,
    histograms_path: Path,
    output_dir: Path,
    *,
    kind: str,
    electron_endcap_unity_fallback: bool,
    repo: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / (
        "veto_electron_5to10_sf.json.gz" if kind == "electron" else "loose_muon_5to10_sf.json.gz"
    )
    exported = export_result(
        result_path,
        payload_path,
        kind=kind,
        candidate=True,
        electron_endcap_unity_fallback=electron_endcap_unity_fallback,
    )
    plot_dir = output_dir / "plots"
    plots = _plot_tnp_result(
        result_path,
        histograms_path,
        plot_dir,
        electron_endcap_unity_fallback=electron_endcap_unity_fallback,
    )
    manifest = {
        "schema_version": 1,
        "status": "rendered",
        "kind": kind,
        "git_commit": _git_commit(repo.resolve()),
        "dependencies": _dependency_versions(),
        "inputs": {
            "result": str(result_path),
            "result_sha256": _sha256(result_path),
            "histograms": str(histograms_path),
            "histograms_sha256": _sha256(histograms_path),
        },
        "payload": exported,
        "plots": plots,
        "created_unix": time.time(),
    }
    _write_json(output_dir / "reproduction_manifest.json", manifest)
    return manifest


def _collect_shards(input_dirs: list[Path], globs: list[str] | None) -> list[Path]:
    patterns = globs or ["shard_*.json", "shard_recovery_*.json"]
    return sorted({
        path
        for directory in input_dirs
        for pattern in patterns
        for path in directory.glob(pattern)
    })


def _add_kind(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kind", choices=KINDS, required=True)


def _add_fallback(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--electron-endcap-unity-fallback", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("doctor", help="validate Python, dependencies, configs, and data assets")
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--output", type=Path)

    command = commands.add_parser("select-records", help="select data or MC from a frozen JSON(.gz) record manifest")
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--sample", choices=("data", "mc"), action="append", required=True)
    command.add_argument("-o", "--output", type=Path, required=True)
    command.add_argument("--source-label")
    command.add_argument(
        "--config",
        type=Path,
        help="bind the selected records to an exact campaign config and normalize its metadata",
    )

    command = commands.add_parser("release-manifest", help="hash every released source, config, record, result, and payload")
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("verify-release", help="verify every hash in a release manifest")
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--manifest", type=Path, required=True)

    command = commands.add_parser("runtime-archive", help="build a self-contained Condor source/data archive")
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("-o", "--output", type=Path, required=True)

    command = commands.add_parser("build-records", help="freeze current DAS file records")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--sample", choices=("data", "mc"), action="append")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dasgoclient", type=Path, default=Path("dasgoclient"))
    command.add_argument("--dasmaps", type=Path)

    command = commands.add_parser("prepare", help="prepare an EOS-only 20-file/shard Condor campaign")
    _add_kind(command)
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--workdir", type=Path, required=True)
    command.add_argument("--python-archive", type=Path, required=True)
    command.add_argument("--runtime-archive", type=Path, required=True)
    command.add_argument("--proxy", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--files-per-shard", type=int, default=20)

    command = commands.add_parser("count", help="count one shard to pass/fail histogram JSON")
    _add_kind(command)
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--shard", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--step-size", type=int, default=100_000)

    command = commands.add_parser("recovery-manifest", help="build the unresolved ROOT-file manifest")
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--workdir", type=Path, required=True)
    command.add_argument("--recovery-dir", type=Path, action="append", default=[])
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("recover", help="recover unresolved files with local xrdcp scratch")
    _add_kind(command)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--scratch-dir", type=Path, required=True)
    command.add_argument("--step-size", type=int, default=100_000)
    command.add_argument("--worker-index", type=int, default=0)
    command.add_argument("--workers", type=int, default=1)

    command = commands.add_parser("finalize-skips", help="freeze an audited campaign after permanent skips")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output-records", type=Path, required=True)
    command.add_argument("--output-skips", type=Path, required=True)
    command.add_argument("--dataset-incomplete-threshold", type=float, default=0.01)

    command = commands.add_parser("reduce", help="merge shards, rebin exactly, and optionally substitute reference MC")
    _add_kind(command)
    command.add_argument("--input-dir", type=Path, action="append", required=True)
    command.add_argument("--glob", dest="globs", action="append")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--records", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--target-eta-edges", type=float, nargs="+")
    command.add_argument("--target-pt-edges", type=float, nargs="+")
    command.add_argument("--mc-reference-histograms", type=Path)
    command.add_argument("--mc-reference-year")

    command = commands.add_parser("fit", help="fit every pass/fail mass bin and propagate variations")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    command = commands.add_parser("audit-trigger", help="audit config-driven reference-HLT presence and rates")
    _add_kind(command)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--sample", choices=("data", "mc"), required=True)
    command.add_argument("--file", dest="files", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--step-size", type=int, default=100_000)
    command.add_argument("--max-events", type=int)

    command = commands.add_parser("validate", help="apply file, fit, trigger, and visual-review adoption gates")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--data-trigger-audit", type=Path, required=True)
    command.add_argument("--mc-trigger-audit", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--max-chi2-ndf", type=float, default=12.0)
    command.add_argument("--adopt-after-visual-review", action="store_true")
    command.add_argument("--visual-review-note")

    command = commands.add_parser("export", help="write correctionlib JSON.GZ")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--candidate", action="store_true")

    command = commands.add_parser("plot", help="draw all standard square mplhep plots")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)

    command = commands.add_parser("render", help="export a candidate payload and regenerate plots from a fit result")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    command = commands.add_parser("reproduce", help="refit frozen histograms, export, and plot")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--histograms", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    command = commands.add_parser("run-local", help="run NanoAOD counting, fitting, export, and plotting locally")
    _add_kind(command)
    _add_fallback(command)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--data-files", type=Path, required=True)
    command.add_argument("--mc-files", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    command.add_argument("--step-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor(args.project_root)
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ready" else 2
    if args.command == "select-records":
        result = select_records(
            args.records,
            args.output,
            set(args.sample),
            source_label=args.source_label,
            config_path=args.config,
        )
    elif args.command == "release-manifest":
        result = build_release_manifest(args.project_root)
        _write_json(args.output, result)
    elif args.command == "verify-release":
        result = verify_release(args.project_root, args.manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 2
    elif args.command == "runtime-archive":
        result = build_runtime_archive(args.project_root, args.output)
    elif args.command == "build-records":
        config = _read_json(args.config)
        payload = build_records(
            config,
            dasgoclient=args.dasgoclient,
            dasmaps=args.dasmaps,
            samples=set(args.sample) if args.sample else None,
        )
        _write_json(args.output, payload)
        result = {key: value for key, value in payload.items() if key not in {"records", "dataset_audit"}}
    elif args.command == "prepare":
        result = prepare(
            records_path=args.records,
            workdir=args.workdir,
            python_archive=args.python_archive,
            runtime_archive=args.runtime_archive,
            proxy=args.proxy,
            config=args.config,
            kind=args.kind,
            files_per_shard=args.files_per_shard,
        )
    elif args.command == "count":
        payload = count_tnp_shard(
            kind=args.kind,
            shard=_read_json(args.shard),
            config=_read_json(args.config),
            repo=args.project_root.resolve(),
            step_size=args.step_size,
        )
        _write_json(args.output, payload)
        result = {key: value for key, value in payload.items() if key not in {"samples", "processing"}}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if payload["status"] != "failed" else 2
    elif args.command == "recovery-manifest":
        payload = build_residual_manifest(args.records, args.workdir, args.recovery_dir)
        _write_json(args.output, payload)
        result = {key: value for key, value in payload.items() if key not in {"records", "failure_diagnostics"}}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not payload["files_unresolved"] else 2
    elif args.command == "recover":
        result = recover(
            kind=args.kind,
            manifest_path=args.manifest,
            config_path=args.config,
            repo=args.project_root,
            output_dir=args.output_dir,
            scratch_dir=args.scratch_dir,
            step_size=args.step_size,
            worker_index=args.worker_index,
            workers=args.workers,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "failures"}, indent=2, sort_keys=True))
        return 0 if not result["records_failed"] else 2
    elif args.command == "finalize-skips":
        result = finalize_permanent_skips(
            manifest_path=args.manifest,
            output_records=args.output_records,
            output_skips=args.output_skips,
            dataset_incomplete_threshold=args.dataset_incomplete_threshold,
        )
    elif args.command == "reduce":
        config = _read_json(args.config)
        paths = _collect_shards(args.input_dir, args.globs)
        if not paths:
            raise FileNotFoundError("no shard JSONs matched the requested inputs")
        expected_records = list(_read_json(args.records).get("records") or []) if args.records else None
        payload = merge_tnp_shards(paths, kind=args.kind, config=config, expected_records=expected_records)
        target_eta = args.target_eta_edges or config["probe_abseta_edges"]
        target_pt = args.target_pt_edges or config["probe_pt_edges_gev"]
        if payload["probe_abseta_edges"] != target_eta or payload["probe_pt_edges_gev"] != target_pt:
            payload = rebin_probe_histograms(payload, target_eta_edges=target_eta, target_pt_edges=target_pt)
        if args.mc_reference_year and not args.mc_reference_histograms:
            raise ValueError("--mc-reference-year requires --mc-reference-histograms")
        if args.mc_reference_histograms:
            payload = use_mc_reference(
                payload,
                _read_json(args.mc_reference_histograms),
                source=args.mc_reference_histograms,
                reference_year=args.mc_reference_year,
            )
        _write_json(args.output, payload)
        result = {
            "status": payload["status"],
            "files_expected": payload["files_expected"],
            "files_processed": payload["files_processed"],
            "files_failed": len(payload["files_failed"]),
            "output": str(args.output),
        }
    elif args.command == "fit":
        payload, exit_code = fit_result(
            args.histograms,
            args.output,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
        result = {
            "status": payload["status"],
            "valid_bins": sum(bool(item.get("valid")) for item in payload["bins"]),
            "bins": len(payload["bins"]),
            "output": str(args.output),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return exit_code
    elif args.command == "audit-trigger":
        config = _read_json(args.config)
        tag_miniiso_value = config.get("tag_miniiso_max", 0.1)
        payload = audit(
            kind=args.kind,
            files=args.files,
            paths=list(config["reference_paths"]),
            step_size=args.step_size,
            max_events=args.max_events,
            tag_pt_min_gev=float(config.get("tag_pt_min_gev", 5.0)),
            tag_miniiso_max=(
                None if tag_miniiso_value is None else float(tag_miniiso_value)
            ),
            tag_trigger_match_required=bool(config.get("tag_trigger_match_required", True)),
            require_reference_paths=bool(
                args.sample == "data" or config.get("apply_reference_trigger_to_mc", True)
            ),
        )
        _write_json(args.output, payload)
        result = {
            "status": payload["status"],
            "files_processed": payload["files_processed"],
            "file_failures": payload["file_failures"],
            "events_read": payload["events_read"],
            "output": str(args.output),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if payload["files_processed"] and not payload["file_failures"] else 2
    elif args.command == "validate":
        payload = validate(
            result=_read_json(args.result),
            histograms=_read_json(args.histograms),
            config=_read_json(args.config),
            data_trigger_audit=_read_json(args.data_trigger_audit),
            mc_trigger_audit=_read_json(args.mc_trigger_audit),
            max_chi2_ndf=args.max_chi2_ndf,
            adopt_after_visual_review=args.adopt_after_visual_review,
            visual_review_note=args.visual_review_note,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
        _write_json(args.output, payload)
        result = {
            "status": payload["status"],
            "blockers": payload["validation"]["blockers"],
            "valid_bins": payload["validation"]["valid_bins"],
            "expected_bins": payload["validation"]["expected_bins"],
            "electron_endcap_unity_bins": payload["validation"]["electron_endcap_unity_bins"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if payload["status"] != "validation_blocked" else 2
    elif args.command == "export":
        result = export_result(
            args.result,
            args.output,
            kind=args.kind,
            candidate=args.candidate,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
    elif args.command == "plot":
        result = _plot_tnp_result(
            args.result,
            args.histograms,
            args.output_dir,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
    elif args.command == "render":
        result = render(
            args.result,
            args.histograms,
            args.output_dir,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
            repo=args.project_root,
        )
    elif args.command == "reproduce":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = args.output_dir / "fit_result.json"
        _, exit_code = fit_result(
            args.histograms,
            result_path,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
        if exit_code:
            return exit_code
        result = render(
            result_path,
            args.histograms,
            args.output_dir,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
            repo=args.project_root,
        )
    elif args.command == "run-local":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        histograms_path = args.output_dir / "histograms.json"
        config = _read_json(args.config)
        histograms = build_histograms(
            kind=args.kind,
            data_files=read_file_list(args.data_files),
            mc_files=read_file_list(args.mc_files),
            config=config,
            repo=args.project_root.resolve(),
            step_size=args.step_size,
        )
        _write_json(histograms_path, histograms)
        result_path = args.output_dir / "fit_result.json"
        _, exit_code = fit_result(
            histograms_path,
            result_path,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
        )
        if exit_code:
            return exit_code
        result = render(
            result_path,
            histograms_path,
            args.output_dir,
            kind=args.kind,
            electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
            repo=args.project_root,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
