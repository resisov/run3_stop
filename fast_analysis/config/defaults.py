from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CANONICAL_REPO_ROOT = Path("/eos/user/t/taiwoo/run3_stop/decaf")
FIXED_PYTHON = Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python")
FIXED_ENV_PREFIX = Path("/eos/user/t/taiwoo/miniconda3/envs/py38")
LEGACY_SCALED_REFERENCE = Path("/eos/user/t/taiwoo/decaf/analysis/hists/stop_2024_nominal.scaled")
FAST_OUTPUTS = CANONICAL_REPO_ROOT / "fast_outputs"


@dataclass(frozen=True)
class FastDefaults:
    year: str = "2024"
    luminosity_fb: float = 109.82
    maturity: tuple = ("prototype", "expected-only", "2024-only", "high-DeltaM-only", "nominal-only")
    repo_root: Path = CANONICAL_REPO_ROOT
    output_root: Path = FAST_OUTPUTS
    manifest_path: Path = FAST_OUTPUTS / "manifests" / "campaign.sqlite"
    benchmark_manifest: Path = FAST_OUTPUTS / "manifests" / "benchmark_inputs.json"
    report_path: Path = FAST_OUTPUTS / "reports" / "index.html"
    fixed_python: Path = FIXED_PYTHON
    environment_path: Path = FIXED_ENV_PREFIX
    legacy_scaled_reference: Path = LEGACY_SCALED_REFERENCE
    recoil_bins: tuple = (250, 300, 350, 400, 500, 800, 1500)
    target_regions: tuple = (
        "cat2_LLCR_highDeltaM",
        "cat3_QCDCR_highDeltaM",
        "cat4_GCR_highDeltaM",
        "cat5_DY2E_highDeltaM",
        "cat6_DY2M_highDeltaM",
        "cat7_SR_highDeltaM",
    )


DEFAULTS = FastDefaults()
