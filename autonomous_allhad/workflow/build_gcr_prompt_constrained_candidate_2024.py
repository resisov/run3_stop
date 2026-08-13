#!/usr/bin/env python3
"""Build a conditional GCR prompt-normalized candidate payload.

The source nominal payload is read-only.  In the derived payload only:
  * GJ is multiplied by one GCR prompt normalization;
  * nominal QCD is reduced to its sidecar truth-prompt/electron fraction and
    multiplied by the same normalization;
  * the measured data-driven photon-fake component is added separately.

This is an R&D control-region constraint, not an adopted prefit correction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


FAKE_PROCESS = "PhotonFake"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_leaf(
    leaf: dict[str, Any],
    scale: np.ndarray | float,
) -> dict[str, Any]:
    result = copy.deepcopy(leaf)
    values = np.asarray(leaf["sumw"], dtype=float)
    variances = np.asarray(leaf["sumw2"], dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    result["sumw"] = (values * scale_array).tolist()
    result["sumw2"] = (
        variances * np.square(scale_array)
    ).tolist()
    return result


def payload_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in leaf.items()
        if key in {"sumw", "sumw2", "entries"}
    }


def qcd_nonfake_fraction(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> np.ndarray:
    origins = measurement["qcd_target_origin_histograms"][region][variable]
    all_values = np.asarray(origins["all"]["sumw"], dtype=float)
    prompt = np.asarray(origins["prompt"]["sumw"], dtype=float)
    electron = np.asarray(origins["electron"]["sumw"], dtype=float)
    raw = np.divide(
        prompt + electron,
        all_values,
        out=np.zeros_like(all_values),
        where=all_values != 0.0,
    )
    if np.any(~np.isfinite(raw)):
        raise RuntimeError(f"nonfinite QCD origin fraction in {region}/{variable}")
    return np.clip(raw, 0.0, 1.0)


def fake_variations(
    measurement: dict[str, Any],
    region: str,
    variable: str,
) -> dict[str, Any]:
    if variable == "recoil":
        return measurement["fake_prediction"]["histograms"][region]
    return measurement["fake_prediction"]["highdm_variable_histograms"][region][
        variable
    ]


def transform_samples(
    samples: dict[str, Any],
    measurement: dict[str, Any],
    region: str,
    variable: str,
    alpha: float,
) -> dict[str, Any]:
    fraction = qcd_nonfake_fraction(measurement, region, variable)
    gj = samples["GJ"]
    qcd = samples["QCD"]
    for variation, leaf in list(gj.items()):
        gj[variation] = scaled_leaf(leaf, alpha)
    for variation, leaf in list(qcd.items()):
        if len(leaf["sumw"]) != len(fraction):
            raise RuntimeError(
                f"QCD bin mismatch in {region}/{variable}/{variation}"
            )
        qcd[variation] = scaled_leaf(leaf, alpha * fraction)
    measured = fake_variations(measurement, region, variable)
    samples[FAKE_PROCESS] = {
        variation: payload_leaf(leaf)
        for variation, leaf in measured.items()
    }
    return {
        "region": region,
        "variable": variable,
        "alpha": alpha,
        "qcd_nonfake_fraction": fraction.tolist(),
        "fake_variations": sorted(measured),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    nominal_path = args.nominal.absolute()
    output_path = args.output.absolute()
    if nominal_path == output_path:
        raise RuntimeError("output must differ from the nominal payload")
    measurement = read_json(args.measurement)
    study = read_json(args.study)
    if measurement.get("status") != "complete":
        raise RuntimeError("measurement is not complete")
    if study.get("status") != "complete":
        raise RuntimeError("Data/MC study is not complete")
    fit = study["fits"]["fit_prompt_pool_with_data_fake"]["inclusive"]
    if fit.get("status") != "fit":
        raise RuntimeError("primary prompt-pool fit is unavailable")
    alpha = float(fit["alpha"])
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise RuntimeError(f"invalid fitted alpha: {alpha}")

    payload = read_json(nominal_path)
    transformed: list[dict[str, Any]] = []
    for region in ("GCR", "GCR_Nt0", "GCR_Nt1"):
        transformed.append(
            transform_samples(
                payload["histograms"][region],
                measurement,
                region,
                "recoil",
                alpha,
            )
        )
    specs = payload.get("highdm_distribution_variable_specs") or {}
    highdm_gcr = payload["highdm_variable_histograms"]["GCR"]
    for variable in sorted(specs):
        if variable == "met":
            continue
        transformed.append(
            transform_samples(
                highdm_gcr[variable],
                measurement,
                "GCR",
                variable,
                alpha,
            )
        )

    payload["schema_version"] = (
        "flat_boosted_recoil_histograms_gcr_prompt_constrained_candidate_v1"
    )
    payload.setdefault("summary", {})["gcr_prompt_constrained_candidate"] = {
        "status": "conditional_r_and_d_not_adopted",
        "selection_source": "real_subset_worker.py",
        "nominal_source": str(nominal_path),
        "nominal_sha256": sha256(nominal_path),
        "measurement": str(args.measurement.absolute()),
        "measurement_sha256": sha256(args.measurement.absolute()),
        "study": str(args.study.absolute()),
        "study_sha256": sha256(args.study.absolute()),
        "prompt_normalization": alpha,
        "prompt_normalization_data_stat_sigma": fit.get("data_stat_sigma"),
        "fit_distribution": "GCR/ut",
        "policy": (
            "scale GJ and the nominal QCD truth-prompt/electron fraction by one "
            "GCR prompt normalization; replace the QCD truth-fake fraction "
            "with the measured PhotonFake process; retain all other processes "
            "and trusted nominal data unchanged"
        ),
        "guardrails": [
            "This is a control-region-constrained R&D payload, not a prefit MC correction.",
            "Adoption requires a generator-level QCD/GJets prompt-overlap policy.",
            "QCD origin fractions are taken from the strict-subset sidecar target.",
            "The structurally empty GCR pTmiss histogram is left unchanged.",
            "The source nominal payload is not modified.",
        ],
        "transformed_distributions": transformed,
    }
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path),
                "prompt_normalization": alpha,
                "transformed_distributions": len(transformed),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
