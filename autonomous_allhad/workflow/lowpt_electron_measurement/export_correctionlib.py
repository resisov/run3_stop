#!/usr/bin/env python3
"""Export an adopted 5--10 GeV veto-electron result to correctionlib."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.sf_payload import correction, correction_set, ensure_adopted, install_adopted_result


def build_payload(result: dict) -> dict:
    item = correction(
        name="veto_electron_5to10_sf",
        description="2024 data/MC SF for the analysis veto-electron ID plus mini-isolation, 5 < pT < 10 GeV",
        axes=[("abseta", result["probe_abseta_edges"]), ("pt", result["probe_pt_edges_gev"])],
        nominal=[entry["scale_factor"] for entry in result["bins"]],
        uncertainty=[entry["scale_factor_uncertainty"] for entry in result["bins"]],
    )
    return correction_set("Run-3 all-hadronic stop low-pT veto-electron scale factor", [item])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    ensure_adopted(result, args.result)
    print(json.dumps(install_adopted_result(args.result, args.output, build_payload(result)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
