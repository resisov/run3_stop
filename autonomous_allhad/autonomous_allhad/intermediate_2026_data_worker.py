from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import flat_ntuple_worker as flat
from . import intermediate_2025_data_worker as worker_2025
from . import real_subset_worker as baseline


DATA_YEAR = "2026"
CORRECTION_YEAR = "2025"
BLOCKED_REASON = (
    "2026 data processing is disabled: the 2025 data JEC run-binned payload does not "
    "accept 2026 run numbers, and substituting a 2025 proxy run is explicitly forbidden."
)
INTERMEDIATE_SCHEMA = "flat_ntuple_shard_v8_float32_fullselection_2026_data_2025corr"
LUMIMASK_NAME = "Cert_Collisions2026_401624_403937_golden.json"
LUMIMASK_SHA256 = "5c16911a0a03735d21c99f470afa737da0e20c6f213e3beebc79f29e02dcbd6f"
LUMIMASK_RELATIVE_PATH = Path("analysis/data/lumiMask") / LUMIMASK_NAME
JET_ID_2025 = Path("analysis/data/JMESF/2025/jetid.json.gz")
JET_VETO_2025 = Path("analysis/data/JMESF/2025/jetvetomaps.json.gz")
JET_VETO_CORRECTION_2025 = "Summer24Prompt25_RunCDEFG_V1"


_ORIGINAL_CAMPAIGN_YEAR = baseline.campaign_year


def campaign_year_2026(year: str) -> str:
    value = str(year)
    return DATA_YEAR if value == DATA_YEAR else _ORIGINAL_CAMPAIGN_YEAR(value)


def analysis_year_2026(year: str) -> str:
    data_year = campaign_year_2026(str(year))
    return CORRECTION_YEAR if data_year in {DATA_YEAR, CORRECTION_YEAR} else data_year


def extract_chunk_2026(
    arrays: Any,
    dataset: str,
    process: str,
    sp: str | None,
    year: str,
    file_path: str,
    entry_start: int,
    entry_stop: int,
    fastsim_trigger_bypass: bool = False,
    shift_name: str | None = None,
    compute_weights: bool = True,
    materialize_skim_flag: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(year) != DATA_YEAR:
        raise RuntimeError(
            f"intermediate_2026_data_worker only accepts data year {DATA_YEAR}, got {year!r}"
        )
    if process not in flat.DATA_PROCESSES:
        raise RuntimeError(
            f"intermediate_2026_data_worker is data-only; rejected process {process!r}"
        )

    # The 2025 implementation evaluates all object corrections.  The installed
    # year routers below retain data year 2026 for the golden JSON and output,
    # while explicitly routing correction lookups to 2025.
    rows, summary = worker_2025.extract_chunk_2025(
        arrays,
        dataset,
        process,
        sp,
        DATA_YEAR,
        file_path,
        entry_start,
        entry_stop,
        fastsim_trigger_bypass=fastsim_trigger_bypass,
        shift_name=shift_name,
        compute_weights=compute_weights,
        materialize_skim_flag=materialize_skim_flag,
    )
    summary["intermediate_schema"] = INTERMEDIATE_SCHEMA
    summary["year_policy"] = {
        "data_year": DATA_YEAR,
        "selection_year": CORRECTION_YEAR,
        "correction_year": CORRECTION_YEAR,
        "scale_factor_year": CORRECTION_YEAR,
        "lumimask_year": DATA_YEAR,
        "lumimask": str(LUMIMASK_RELATIVE_PATH),
        "lumimask_sha256": LUMIMASK_SHA256,
        "mc_scale_factors_applied": False,
        "reason": "User-approved 2026 data extension of the 2025 analysis calibration policy.",
    }
    return rows, summary


def install_backend() -> None:
    worker_2025.install_backend()

    # Keep the collision-data identity (and therefore the 2026 golden JSON)
    # distinct from the explicitly aliased analysis/calibration identity.
    baseline.campaign_year = campaign_year_2026
    baseline.analysis_year = analysis_year_2026
    baseline.LUMIMASK_RELATIVE_PATHS[DATA_YEAR] = LUMIMASK_RELATIVE_PATH
    baseline.JET_ID_RELATIVE_PATHS[DATA_YEAR] = JET_ID_2025
    baseline.JET_VETO_MAPS[DATA_YEAR] = (JET_VETO_2025, JET_VETO_CORRECTION_2025)
    baseline.FATJET_ID_RELATIVE_PATH = JET_ID_2025
    baseline._LUMIMASK_CACHE.clear()
    baseline._CORRECTION_CACHE.clear()

    flat.SCHEMA_VERSION = INTERMEDIATE_SCHEMA
    flat.CORRECTION_YEAR = CORRECTION_YEAR
    flat.extract_chunk = extract_chunk_2026


def main(argv: list[str] | None = None) -> int:
    raise RuntimeError(BLOCKED_REASON)


if __name__ == "__main__":
    raise SystemExit(main())
