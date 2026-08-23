#!/usr/bin/env python3
"""Seed exact-Nb High-dM CR rates from a converged Nb>=2 CR-only fit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RATE_KINDS = ("ll_norm", "qcd_norm", "sgamma_shape")
EXACT_GROUP_SOURCE = {
    "Nb1": "Nb1",
    "Nb2": "Nb2plus",
    "Nb3plus": "Nb2plus",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_parameters", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.fit_parameters.read_text())
    if payload.get("status") != "complete":
        raise SystemExit("seed fit is not complete")
    parameters = payload["parameters"]
    assignments = []
    for year in ("2024", "2025"):
        for kind in RATE_KINDS:
            for exact_group, source_group in EXACT_GROUP_SOURCE.items():
                for recoil_bin in range(6):
                    source = (
                        f"{kind}_highdm_{source_group}_bin{recoil_bin}_{year}"
                    )
                    target = (
                        f"{kind}_highdm_{exact_group}_bin{recoil_bin}_{year}"
                    )
                    if source not in parameters:
                        raise SystemExit(f"seed parameter is absent: {source}")
                    value = float(parameters[source]["value"])
                    if not math.isfinite(value) or value <= 0.0:
                        raise SystemExit(f"invalid seed {source}={value}")
                    assignments.append(f"{target}={value:.12g}")
    if len(assignments) != 108:
        raise SystemExit(f"expected 108 exact-Nb seeds, built {len(assignments)}")
    print(",".join(assignments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
