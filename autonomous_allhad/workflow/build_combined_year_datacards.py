#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from build_combine_inputs import stable_path, write_parallel_runner


DEFAULT_CMSSW = Path(
    "/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4"
)
DEFAULT_RUNTIME_MANIFEST = Path(
    "/eos/user/t/taiwoo/run3_stop/runtime/runtime_manifest.json"
)
DEFAULT_X509_PROXY = Path(
    "/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757"
)


def cards_by_mass(directory: Path) -> dict[str, Path]:
    prefix = "datacard_"
    cards = {}
    for path in sorted(directory.glob(f"{prefix}*.txt")):
        mass = path.stem[len(prefix):]
        if mass:
            cards[mass] = path.absolute()
    return cards


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_source_manifest(card_dir: Path) -> dict:
    path = card_dir.parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"source card manifest is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "combine_inputs_ready":
        raise ValueError(f"source card manifest is not ready: {path}")
    model = payload.get("model") or {}
    if model.get("zinv_free_normalization_rate_parameter") is not False:
        raise ValueError(
            f"source card manifest lacks the RZ-normalization gate: {path}"
        )
    if model.get("sgamma_role") != "shape only, shared between matched GCR and Z SR":
        raise ValueError(f"source card manifest has the wrong Sgamma role: {path}")
    return payload


def read_combine_runtime(path: Path) -> tuple[Path, str]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ready":
        raise ValueError(f"runtime is not ready: {path}")
    combine = payload.get("combine") or {}
    archive = Path(str(combine.get("path", "")))
    checksum = str(combine.get("sha256", ""))
    if not archive.is_file() or archive.name != "combine_cmssw_14_1_0_pre4.tgz":
        raise ValueError(f"invalid Combine runtime archive: {archive}")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError(f"invalid Combine runtime checksum in {path}")
    return archive, checksum


def write_condor_limit_submission(
    cards: dict[str, str],
    output_dir: Path,
    runtime_archive: Path,
    runtime_checksum: str,
    point_timeout: int,
    batch_name: str,
) -> tuple[Path, Path]:
    wrapper = output_dir / "run_condor_limit_point.sh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "MASS=$1",
                "CARD=$2",
                "OUTDIR=$3",
                f"POINT_TIMEOUT={int(point_timeout)}",
                "WORKSPACE_TIMEOUT=900",
                "export PYTHONNOUSERSITE=1",
                "unset PYTHONPATH PYTHONHOME",
                ': "${_CONDOR_SCRATCH_DIR:?Condor scratch directory is required}"',
                'SCRATCH_BASE="$_CONDOR_SCRATCH_DIR"',
                f'RUNTIME_ARCHIVE="$SCRATCH_BASE/{runtime_archive.name}"',
                'export HOME="$SCRATCH_BASE/home"',
                'export TMPDIR="$SCRATCH_BASE"',
                'export XDG_CACHE_HOME="$SCRATCH_BASE/cache"',
                'WORKDIR="$SCRATCH_BASE/work"',
                'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$WORKDIR"',
                f'echo "{runtime_checksum}  $RUNTIME_ARCHIVE" | sha256sum -c -',
                'tar -xzf "$RUNTIME_ARCHIVE" -C "$SCRATCH_BASE"',
                'rm -f "$RUNTIME_ARCHIVE"',
                'CMSSW="$SCRATCH_BASE/CMSSW_14_1_0_pre4"',
                "source /cvmfs/cms.cern.ch/cmsset_default.sh",
                'cd "$CMSSW/src"',
                'scramv1 b ProjectRename >/dev/null',
                'eval "$(scramv1 runtime -sh)"',
                'command -v text2workspace.py >/dev/null',
                'command -v combine >/dev/null',
                'case "$(command -v combine)" in "$CMSSW"/*) ;; *) exit 70 ;; esac',
                'mkdir -p "$OUTDIR"',
                'cd "$WORKDIR"',
                'WORKSPACE="workspace_${MASS}.root"',
                'timeout "$WORKSPACE_TIMEOUT" text2workspace.py "$CARD" -m 120 -o "$WORKSPACE"',
                'timeout "$POINT_TIMEOUT" combine -M AsymptoticLimits --run blind -m 120 -n "_${MASS}" "$WORKSPACE"',
                'shopt -s nullglob',
                'RESULTS=("higgsCombine_${MASS}.AsymptoticLimits.mH"*.root)',
                '[[ ${#RESULTS[@]} -eq 1 ]]',
                'mv -f "${RESULTS[0]}" "$OUTDIR/"',
            ]
        )
        + "\n"
    )
    wrapper.chmod(0o755)
    logs = output_dir / "condor_logs"
    logs.mkdir(parents=True, exist_ok=True)
    submit = output_dir / "limits_eossubmit.sub"
    rows = "\n".join(
        f"{mass} {stable_path(Path(card))}"
        for mass, card in sorted(cards.items())
        if not (
            output_dir
            / "limits"
            / f"higgsCombine_{mass}.AsymptoticLimits.mH120.root"
        ).is_file()
        or (
            output_dir
            / "limits"
            / f"higgsCombine_{mass}.AsymptoticLimits.mH120.root"
        ).stat().st_size == 0
    )
    stable_output = stable_path(output_dir)
    stable_logs = stable_path(logs)
    submit.write_text(
        f"""universe = vanilla
executable = {stable_path(wrapper)}
initialdir = {stable_output}
arguments = $(mass) $(card) {stable_output}/limits
output = {stable_logs}/$(mass).out
error = {stable_logs}/$(mass).err
log = {stable_logs}/cluster.log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {stable_path(runtime_archive)}
transfer_output_files = ""
use_x509userproxy = true
x509userproxy = {stable_path(DEFAULT_X509_PROXY)}
request_cpus = 1
request_memory = 6000MB
request_disk = 4000MB
+MaxRuntime = {int(point_timeout) + 600}
+JobBatchName = \"{batch_name}\"
queue mass,card from (
{rows}
)
"""
    )
    return wrapper, submit


def write_condor_impact_submission(
    card: str,
    output_dir: Path,
    impact_runner: Path,
    mass_key: str,
    batch_name: str,
    expect_signal: int,
    runtime_archive: Path,
    runtime_checksum: str,
) -> Path:
    match = re.fullmatch(r"mStop([0-9]+)_mLSP([0-9]+)", mass_key)
    if not match or int(match.group(2)) != 500:
        raise ValueError(
            "the existing impact runner supports the requested mLSP500 "
            f"benchmark only, got {mass_key}"
        )
    if not impact_runner.is_file():
        raise FileNotFoundError(f"impact runner is missing: {impact_runner}")
    if expect_signal not in {0, 1}:
        raise ValueError(f"expect_signal must be 0 or 1, got {expect_signal}")
    fit_label = f"r{expect_signal}"
    impact_dir = output_dir / f"impact_{mass_key}_{fit_label}"
    logs = impact_dir / "condor_logs"
    logs.mkdir(parents=True, exist_ok=True)
    submit = output_dir / f"impact_{fit_label}_eossubmit.sub"
    stable_impact_dir = stable_path(impact_dir)
    stable_logs = stable_path(logs)
    submit.write_text(
        f"""universe = vanilla
executable = {stable_path(impact_runner)}
initialdir = {stable_impact_dir}
arguments = {stable_path(Path(card))} {stable_impact_dir} {match.group(1)}
output = {stable_logs}/impact.out
error = {stable_logs}/impact.err
log = {stable_logs}/cluster.log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {stable_path(runtime_archive)}
transfer_output_files = ""
use_x509userproxy = true
x509userproxy = {stable_path(DEFAULT_X509_PROXY)}
environment = "IMPACT_EXPECT_SIGNAL={expect_signal} IMPACT_R_MIN={'-20' if expect_signal == 0 else '0'} IMPACT_R_MAX=20 COMBINE_RUNTIME_SHA256={runtime_checksum} COMBINE_RUNTIME_ARCHIVE={runtime_archive.name}"
request_cpus = 4
request_memory = 16000MB
request_disk = 8000MB
+MaxRuntime = 43200
+JobBatchName = \"{batch_name}\"
queue 1
"""
    )
    return submit


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine two years of matching mass-point datacards.")
    parser.add_argument("--left-dir", required=True, type=Path)
    parser.add_argument("--left-label", default="y2024")
    parser.add_argument("--left-lumi-name", default="lumi_13p6TeV_2024")
    parser.add_argument("--right-dir", required=True, type=Path)
    parser.add_argument("--right-label", default="y2025")
    parser.add_argument("--right-lumi-name", default="lumi_13p6TeV_2025")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--combine-cards", default="combineCards.py")
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=7200)
    parser.add_argument("--submission-only", action="store_true")
    parser.add_argument("--cmssw", type=Path, default=DEFAULT_CMSSW)
    parser.add_argument(
        "--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST
    )
    parser.add_argument(
        "--condor-batch-name", default="NPS26012_2024_2025_combined_limits"
    )
    parser.add_argument("--impact-mass-key")
    parser.add_argument(
        "--impact-runner",
        type=Path,
        default=Path(__file__).with_name("run_asimov_impacts_eos.sh"),
    )
    args = parser.parse_args()
    runtime_archive, runtime_checksum = read_combine_runtime(
        args.runtime_manifest
    )

    left = cards_by_mass(args.left_dir)
    right = cards_by_mass(args.right_dir)
    if not left or not right:
        raise SystemExit("both source datacard directories must be non-empty")
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if left_only or right_only:
        raise SystemExit(
            f"mass grids differ: left_only={len(left_only)}, right_only={len(right_only)}"
        )

    left_manifest = read_source_manifest(args.left_dir)
    right_manifest = read_source_manifest(args.right_dir)
    left_model = left_manifest["model"]
    right_model = right_manifest["model"]
    bin_signature = (
        int(left_model["highdm_bins"]),
        int(left_model["lowdm_bins"]),
    )
    if bin_signature != (
        int(right_model["highdm_bins"]),
        int(right_model["lowdm_bins"]),
    ):
        raise SystemExit("source years use different search-bin accounting")
    if left_model["signal_topology"] != right_model["signal_topology"]:
        raise SystemExit("source years use different signal topologies")

    output_dir = args.output_dir.absolute()
    datacard_dir = output_dir / "datacards"
    limit_dir = output_dir / "limits"
    datacard_dir.mkdir(parents=True, exist_ok=True)
    combined_cards: dict[str, str] = {}
    warnings: list[dict[str, str]] = []
    if args.submission_only:
        existing = cards_by_mass(datacard_dir)
        if set(existing) != set(left):
            raise SystemExit(
                "existing combined-card grid does not match the source grids"
            )
        combined_cards = {mass: str(path) for mass, path in existing.items()}
    else:
        for mass in sorted(left):
            output = datacard_dir / f"datacard_{mass}.txt"
            temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
            try:
                with temporary.open("w") as handle:
                    result = subprocess.run(
                        [
                            args.combine_cards,
                            f"{args.left_label}={left[mass]}",
                            f"{args.right_label}={right[mass]}",
                        ],
                        stdout=handle,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"combineCards.py failed for {mass}: {result.stderr.strip()[:1000]}"
                    )
                text = temporary.read_text()
                required = (args.left_lumi_name, args.right_lumi_name)
                missing_lumi = [name for name in required if name not in text]
                if missing_lumi:
                    raise RuntimeError(
                        f"combined card {mass} is missing luminosity nuisances: {missing_lumi}"
                    )
                if result.stderr.strip():
                    warnings.append({"mass_point": mass, "stderr": result.stderr.strip()[:1000]})
                forbidden = [
                    token
                    for token in ("CMS_SUS26090", "zg_norm_")
                    if token in text
                ]
                if forbidden:
                    raise RuntimeError(
                        f"combined card {mass} contains retired names: {forbidden}"
                    )
                for required_token in (
                    "CMS_NPS26012_",
                    "sgamma_shape_",
                ):
                    if required_token not in text:
                        raise RuntimeError(
                            f"combined card {mass} is missing {required_token!r}"
                        )
                for year in ("2024", "2025"):
                    if not re.search(
                        rf"^[A-Za-z0-9_]+_{year}\s+rateParam\s+",
                        text,
                        flags=re.MULTILINE,
                    ):
                        raise RuntimeError(
                            f"combined card {mass} is missing year-specific "
                            f"{year} rate parameters"
                        )
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            combined_cards[mass] = str(output)

    runner = output_dir / "run_combine_expected.sh"
    write_parallel_runner(
        combined_cards,
        limit_dir,
        runner,
        args.runner_jobs,
        args.point_timeout,
    )
    condor_wrapper, condor_submit = write_condor_limit_submission(
        combined_cards,
        output_dir,
        runtime_archive,
        runtime_checksum,
        args.point_timeout,
        args.condor_batch_name,
    )
    impact_submits: dict[str, str] = {}
    if args.impact_mass_key:
        impact_card = combined_cards.get(args.impact_mass_key)
        if impact_card is None:
            raise SystemExit(
                f"impact benchmark is absent from the common grid: {args.impact_mass_key}"
            )
        for expect_signal in (1, 0):
            fit_label = f"r{expect_signal}"
            impact_submits[fit_label] = str(
                write_condor_impact_submission(
                    impact_card,
                    output_dir,
                    args.impact_runner,
                    args.impact_mass_key,
                    f"NPS26012_2024_2025_T2tt_impact_{fit_label}",
                    expect_signal,
                    runtime_archive,
                    runtime_checksum,
                )
            )
    manifest = {
        "status": "combine_inputs_ready",
        "method": "combineCards.py with year-labelled channels",
        "mass_point_count": len(combined_cards),
        "mass_points": sorted(combined_cards),
        "model": {
            "highdm_bins": bin_signature[0],
            "lowdm_bins": bin_signature[1],
            "zinv_normalization": "external RZ only",
            "sgamma_role": "shape only, shared between matched GCR and Z SR",
            "zinv_free_normalization_rate_parameter": False,
            "signal_topology": left_model["signal_topology"],
        },
        "left": {
            "label": args.left_label,
            "datacard_dir": str(args.left_dir.absolute()),
            "luminosity_nuisance": args.left_lumi_name,
        },
        "right": {
            "label": args.right_label,
            "datacard_dir": str(args.right_dir.absolute()),
            "luminosity_nuisance": args.right_lumi_name,
        },
        "nuisance_correlation_policy": (
            "Identical nuisance names remain correlated across years; "
            "year-specific luminosity nuisance names remain uncorrelated."
        ),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "condor_wrapper": str(condor_wrapper),
        "condor_submit": str(condor_submit),
        "condor_backend": "EOS schedd via module load lxbatch/eossubmit",
        "runtime": {
            "manifest": str(args.runtime_manifest),
            "combine_archive": str(runtime_archive),
            "sha256": runtime_checksum,
        },
        "impact_submit": impact_submits.get("r1"),
        "impact_submits": impact_submits,
        "impact_mass_key": args.impact_mass_key,
        "combine_cards_warnings": warnings,
    }
    write_json(output_dir / "combine_input_manifest.json", manifest)
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "mass_points": len(combined_cards),
                "warnings": len(warnings),
                "datacard_dir": str(datacard_dir),
                "runner": str(runner),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
