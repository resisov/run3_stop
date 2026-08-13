#!/usr/bin/env python3
"""Export an adopted combined 5--10 GeV loose-muon result to correctionlib."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import correctionlib

from workflow.sf_payload import correction, correction_set, ensure_adopted, install_adopted_result


def combined_values(result: dict, id_payload: Path) -> tuple[list[float], list[float]]:
    evaluator = correctionlib.CorrectionSet.from_file(str(id_payload))["NUM_LooseID_DEN_TrackerMuons"]
    eta_edges = result["probe_abseta_edges"]
    pt_edges = result["probe_pt_edges_gev"]
    nominal: list[float] = []
    uncertainty: list[float] = []
    for eta_index in range(len(eta_edges) - 1):
        eta = 0.5 * (float(eta_edges[eta_index]) + float(eta_edges[eta_index + 1]))
        for pt_index in range(len(pt_edges) - 1):
            pt = 0.5 * (float(pt_edges[pt_index]) + float(pt_edges[pt_index + 1]))
            flat_index = eta_index * (len(pt_edges) - 1) + pt_index
            fitted = result["bins"][flat_index]
            iso_sf = float(fitted["scale_factor"])
            iso_unc = float(fitted["scale_factor_uncertainty"])
            id_sf = float(evaluator.evaluate(eta, pt, "nominal"))
            id_up = float(evaluator.evaluate(eta, pt, "systup"))
            id_down = float(evaluator.evaluate(eta, pt, "systdown"))
            id_unc = max(abs(id_up - id_sf), abs(id_down - id_sf))
            nominal.append(iso_sf * id_sf)
            uncertainty.append(math.hypot(id_sf * iso_unc, iso_sf * id_unc))
    return nominal, uncertainty


def build_payload(
    result: dict,
    id_payload: Path = Path("analysis/data/MuonSF/2024/muon_JPsi.json.gz"),
) -> dict:
    nominal, uncertainty = combined_values(result, id_payload)
    item = correction(
        name="loose_muon_5to10_sf",
        description="2024 combined LooseID and mini-isolation data/MC SF for analysis loose muons, 5 < pT < 10 GeV",
        axes=[("abseta", result["probe_abseta_edges"]), ("pt", result["probe_pt_edges_gev"])],
        nominal=nominal,
        uncertainty=uncertainty,
    )
    return correction_set("Run-3 all-hadronic stop low-pT loose-muon scale factor", [item])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz"))
    parser.add_argument("--loose-id-payload", type=Path, default=Path("analysis/data/MuonSF/2024/muon_JPsi.json.gz"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    ensure_adopted(result, args.result)
    print(json.dumps(install_adopted_result(args.result, args.output, build_payload(result, args.loose_id_payload)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
