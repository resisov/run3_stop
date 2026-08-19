#!/usr/bin/env python3
"""Render the full-2024 TROTA High-dM category study as Markdown.

The input is the compact accumulator written by
``study_trota_highdm_categories_2024.py``.  This reporter intentionally keeps
the statistical comparison narrow: only the six recoil bins of the existing
Nt=0, Nw=0, Nb>=1 High-dM SR block are split by the reconstructed-top
multiplicity.  It does not interpret the result as a limit or an adopted
category definition.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEME = "trota_disjoint"
FIDUCIAL_SCHEME = "trota_disjoint_run2_kinematic"
RECOILS = ("250-300", "300-350", "350-400", "400-500", "500-800", "800-1500")
NRES = ("0", "1", "2plus")
BACKGROUND_ORDER = ("TT", "ST", "WtoLNu", "Zto2Nu", "QCD", "DY", "GJ", "VV")


def load_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, values in payload["stats"].items():
        scheme, region, sample, nb, njet, recoil, nres = key.split("|")
        rows.append(
            {
                "scheme": scheme,
                "region": region,
                "sample": sample,
                "nb": nb,
                "njet": njet,
                "recoil": recoil,
                "nres": nres,
                "entries": int(values["entries"]),
                "sumw": float(values["sumw"]),
                "sumw2": float(values["sumw2"]),
            }
        )
    return rows


def aggregate(
    rows: Iterable[dict[str, Any]],
    *,
    scheme: str | None = None,
    region: str | None = None,
    samples: set[str] | None = None,
    group_by: tuple[str, ...] = (),
) -> dict[tuple[str, ...], dict[str, float]]:
    out: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"entries": 0.0, "sumw": 0.0, "sumw2": 0.0}
    )
    for row in rows:
        if scheme is not None and row["scheme"] != scheme:
            continue
        if region is not None and row["region"] != region:
            continue
        if samples is not None and row["sample"] not in samples:
            continue
        key = tuple(str(row[field]) for field in group_by)
        target = out[key]
        target["entries"] += row["entries"]
        target["sumw"] += row["sumw"]
        target["sumw2"] += row["sumw2"]
    return dict(out)


def finite(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:,.1f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.3g}"


def percent(numerator: float, denominator: float) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"


def neff(record: dict[str, float]) -> float:
    return 0.0 if record["sumw2"] <= 0 else record["sumw"] ** 2 / record["sumw2"]


def asimov_z2(signal: float, background: float) -> float:
    if not math.isfinite(signal) or not math.isfinite(background):
        return 0.0
    if signal <= 0 or background <= 0:
        return 0.0
    return 2.0 * ((signal + background) * math.log1p(signal / background) - signal)


def sensitivity_gain(
    rows: list[dict[str, Any]],
    backgrounds: set[str],
    signal: str,
    split: str,
    scheme: str = SCHEME,
) -> tuple[float, float, float]:
    background = aggregate(
        rows,
        scheme=scheme,
        region="SR",
        samples=backgrounds,
        group_by=("recoil", "nres"),
    )
    signal_rows = aggregate(
        rows,
        scheme=scheme,
        region="SR",
        samples={signal},
        group_by=("recoil", "nres"),
    )
    current_z2 = 0.0
    split_z2 = 0.0
    for recoil in RECOILS:
        b_values = {n: background.get((recoil, n), {}).get("sumw", 0.0) for n in NRES}
        s_values = {n: signal_rows.get((recoil, n), {}).get("sumw", 0.0) for n in NRES}
        current_z2 += asimov_z2(sum(s_values.values()), sum(b_values.values()))
        groups = (("0",), ("1", "2plus")) if split == "two" else (("0",), ("1",), ("2plus",))
        for group in groups:
            split_z2 += asimov_z2(
                sum(s_values[n] for n in group), sum(b_values[n] for n in group)
            )
    current_z = math.sqrt(max(current_z2, 0.0))
    split_z = math.sqrt(max(split_z2, 0.0))
    gain = split_z / current_z if current_z > 0 else 0.0
    return current_z, split_z, gain


def table(headers: tuple[str, ...], body: Iterable[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def render(
    payload: dict[str, Any],
    plot_summary: dict[str, Any] | None = None,
) -> str:
    if payload.get("status") != "complete":
        raise RuntimeError(f"study is not complete: {payload.get('status')}")
    if payload.get("files_completed") != payload.get("files_expected"):
        raise RuntimeError("completed/expected file counts differ")
    if payload.get("failures"):
        raise RuntimeError("study contains file failures")

    rows = load_rows(payload)
    samples = {row["sample"] for row in rows}
    signals = sorted(sample for sample in samples if sample.startswith(("T2tt_", "T2tb_", "T2bW_")))
    backgrounds = set(samples) - set(signals) - {"data_obs"}
    ordered_backgrounds = [sample for sample in BACKGROUND_ORDER if sample in backgrounds]
    ordered_backgrounds.extend(sorted(backgrounds - set(ordered_backgrounds)))

    sr_by_nres = aggregate(
        rows, scheme=SCHEME, region="SR", samples=backgrounds, group_by=("nres",)
    )
    sr_total = sum((sr_by_nres.get((n,), {}).get("sumw", 0.0) for n in NRES), 0.0)
    sr_neff_total_record = aggregate(
        rows, scheme=SCHEME, region="SR", samples=backgrounds
    ).get((), {"sumw": 0.0, "sumw2": 0.0})

    totals = payload["totals"]
    full_weight_lines: list[str] = []
    if plot_summary is not None:
        validation = plot_summary.get("validation") or {}
        if (
            plot_summary.get("status") != "complete"
            or int(plot_summary.get("bin_count") or 0) != 61
            or not validation.get("split_recombines_to_canonical_first_six_bins")
            or int(validation.get("unchanged_adopted_bins") or 0) != 49
            or not validation.get("finite_histogram_contents")
        ):
            raise RuntimeError("full-weight 61-bin plot summary is incomplete")
        rendered_plot = plot_summary.get("plot") or {}
        full_weight_lines = [
            "",
            "## Full-AnalysisSF 61-bin visual validation",
            "",
            "A dedicated refill of the category-1 events applies the same full 2024 weight bundle used by the canonical histogram. Its two Nres blocks recombine to the canonical first six bins for every stored variation; the other 49 adopted bins are copied unchanged. This closes the event-weight/provenance gap in the earlier normalized-genweight diagnostic, but it does not supply a TROTA efficiency SF.",
            "",
            *table(
                ("Check", "Result"),
                (
                    ("Output search bins", "61"),
                    ("Canonical first-six-bin recombination", "passed for all samples/variations"),
                    ("Unchanged adopted bins", "49"),
                    ("Finite histogram contents", "passed"),
                    ("Canonical hists SHA256", f"`{plot_summary['canonical_histogram_sha256']}`"),
                    ("Full-weight Nres component SHA256", f"`{plot_summary['component_sha256']}`"),
                    ("PNG", f"`{Path(str(rendered_plot.get('png') or '')).name}`"),
                    ("PDF", f"`{Path(str(rendered_plot.get('pdf') or '')).name}`"),
                    ("TROTA SF", "unavailable; not applied"),
                ),
            ),
        ]
    raw_events = int(totals["events_with_raw_candidate"])
    disjoint_events = int(totals["events_with_disjoint_candidate"])
    fiducial_events = int(totals["events_with_run2_kinematic_candidate"])
    lines = [
        "# 2024 TROTA TopResolved impact on the High-dM categories",
        "",
        f"Status: **complete** ({payload['files_completed']:,}/{payload['files_expected']:,} intermediate ROOT files, 0 failures).",
        "",
        "## Executive conclusion",
        "",
        "The 2024 intermediate format now supports a physically meaningful resolved-top split in the current High-dM `Nt=0, Nw=0, Nb>=1` block. The adopted baseline is the **55-bin tail-merged scheme**. The recommended next validation target is an **exploratory 61-bin scheme**: replace each of the six inclusive `Nres` recoil bins retained in category 1 by `Nres=0` and `Nres>=1`. This is a physics proposal, not an adopted category definition.",
        "",
        "Do not yet split boosted-tag categories by `Nres`: the intermediate format does not retain the AK8-subjet identity needed to reproduce the Run-2 AK4/AK8 cross-cleaning. Also do not adopt the separate `Nres>=2` **67-bin** alternative before TROTA data/MC scale factors, systematic variations, control-region closure, and a full expected-limit comparison exist.",
        "",
        "## Provenance and scope",
        "",
        *table(
            ("Quantity", "Value"),
            (
                ("Intermediate ROOT files", f"{payload['files_completed']:,}"),
                ("Events scanned", f"{int(totals['events']):,}"),
                ("Events in the studied High-dM regions/block", f"{int(totals['study_events']):,}"),
                ("Sparse TROTA rows scanned", f"{int(totals['trota_rows']):,}"),
                ("TROTA rows attached to studied events", f"{int(totals['study_candidate_rows']):,}"),
                ("Files failed", "0"),
                ("Input manifest SHA256", f"`{payload['input_manifest_sha256']}`"),
                ("Normalization SHA256", f"`{payload['normalization_sha256']}`"),
            ),
        ),
        "",
        "The study is restricted to events already assigned to a current High-dM region with `Nb>=1`, `Nt=0`, `Nw=0`, and `nboosted_total=0`. Event weights are generator weights times the validated cross-section/luminosity/sum-of-weights normalization. They do **not** include post-skim AnalysisSF weights or a TROTA SF.",
        "",
        "## Adopted 55-bin baseline and proposed mapping",
        "",
        "The adopted 55-bin layout is obtained from the 60-bin precursor by merging the last two recoil bins in five categories: zero-based source-bin pairs `(22,23)`, `(34,35)`, `(40,41)`, `(52,53)`, and `(58,59)`. Categories 1, 2, 3, 5, and 8 retain six recoil bins. Category 1 is `NT0_Nb1plus_T0_W0`, so it is untouched by the five tail merges.",
        "",
        "Therefore the two-way resolved-top proposal is `55 - 6 + 12 = 61` bins. The three-way diagnostic is `55 - 6 + 18 = 67` bins. The full-file TROTA scan studied exactly the six unmerged category-1 recoil bins, so correcting the baseline from the obsolete 60-bin label to the adopted 55-bin label does not require reprocessing events and does not change any yield or sensitivity number below.",
        *full_weight_lines,
        "",
        "## Run-2 mapping and candidate arbitration",
        "",
        "The Run-2 analysis used a resolved trijet top tag (`Nres`) together with boosted-top and boosted-W multiplicities. Candidate triplets were ranked by discriminator and overlapping candidates were removed. This study mirrors that candidate-candidate arbitration by sorting on the TROTA QCD discriminator and greedily rejecting candidates that share an AK4 jet. The primary definition is the TROTA-native 1% WP plus this overlap removal. A separate robustness variant additionally requires `100 <= m(trijet) <= 250 GeV` and `|eta(trijet)| < 2`; those legacy DeepResolved cuts are not silently made part of the new TROTA definition.",
        "",
        *table(
            ("Candidate definition", "Events with >=1 candidate", "Fraction of raw"),
            (
                ("Raw sparse passing triplets", f"{raw_events:,}", "100.0%"),
                ("Jet-disjoint TROTA candidates", f"{disjoint_events:,}", percent(disjoint_events, raw_events)),
                ("Jet-disjoint + Run-2 mass/eta", f"{fiducial_events:,}", percent(fiducial_events, raw_events)),
            ),
        ),
        "",
        "## High-dM SR total background versus Nres",
        "",
        *table(
            ("Nres", "Background yield", "Fraction", "Effective MC events"),
            tuple(
                (
                    n.replace("2plus", ">=2"),
                    finite(sr_by_nres.get((n,), {}).get("sumw", 0.0)),
                    percent(sr_by_nres.get((n,), {}).get("sumw", 0.0), sr_total),
                    finite(neff(sr_by_nres.get((n,), {"sumw": 0.0, "sumw2": 0.0}))),
                )
                for n in NRES
            )
            + (("Total", finite(sr_total), "100.0%", finite(neff(sr_neff_total_record))),),
        ),
        "",
        "The validated TROTA production uses `TTScore/(TTScore+QCDScore) >= 0.9433798789978027`, input jets with stored JetID, `pT>25 GeV`, and `|eta|<2.5`. The model SHA256 is `ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3`.",
        "",
        "### Definition robustness in the High-dM SR",
        "",
    ]

    robustness_rows = []
    for scheme, label in (
        ("raw_pass_triplets", "Raw sparse triplets"),
        (SCHEME, "Jet-disjoint TROTA (primary)"),
        (FIDUCIAL_SCHEME, "Jet-disjoint + Run-2 mass/eta"),
    ):
        values = aggregate(
            rows, scheme=scheme, region="SR", samples=backgrounds, group_by=("nres",)
        )
        yields = [values.get((n,), {}).get("sumw", 0.0) for n in NRES]
        robustness_rows.append(
            (
                label,
                finite(yields[0]),
                finite(yields[1]),
                finite(yields[2]),
                percent(yields[1] + yields[2], sum(yields)),
            )
        )
    lines.extend(
        table(
            ("Definition", "Nres=0", "Nres=1", "Nres>=2", "Nres>=1 fraction"),
            robustness_rows,
        )
    )
    lines.extend(["", "## High-dM SR background-process composition", ""])

    process_rows = aggregate(
        rows, scheme=SCHEME, region="SR", samples=backgrounds, group_by=("sample", "nres")
    )
    lines.extend(
        table(
            ("Process", "Nres=0", "Nres=1", "Nres>=2", "Nres>=1 fraction"),
            (
                (
                    process,
                    finite(process_rows.get((process, "0"), {}).get("sumw", 0.0)),
                    finite(process_rows.get((process, "1"), {}).get("sumw", 0.0)),
                    finite(process_rows.get((process, "2plus"), {}).get("sumw", 0.0)),
                    percent(
                        sum(process_rows.get((process, n), {}).get("sumw", 0.0) for n in ("1", "2plus")),
                        sum(process_rows.get((process, n), {}).get("sumw", 0.0) for n in NRES),
                    ),
                )
                for process in ordered_backgrounds
            ),
        )
    )
    lines.extend(["", "## Recoil dependence", ""])
    recoil_rows = aggregate(
        rows, scheme=SCHEME, region="SR", samples=backgrounds, group_by=("recoil", "nres")
    )
    lines.extend(
        table(
            ("Recoil (GeV)", "Nres=0", "Nres=1", "Nres>=2", "Nres>=1 fraction"),
            (
                (
                    recoil,
                    finite(recoil_rows.get((recoil, "0"), {}).get("sumw", 0.0)),
                    finite(recoil_rows.get((recoil, "1"), {}).get("sumw", 0.0)),
                    finite(recoil_rows.get((recoil, "2plus"), {}).get("sumw", 0.0)),
                    percent(
                        sum(recoil_rows.get((recoil, n), {}).get("sumw", 0.0) for n in ("1", "2plus")),
                        sum(recoil_rows.get((recoil, n), {}).get("sumw", 0.0) for n in NRES),
                    ),
                )
                for recoil in RECOILS
            ),
        )
    )

    lines.extend(["", "## Signal separation diagnostic", ""])
    lines.append(
        "The table below is a normalized-yield, statistical-only Asimov diagnostic for this one six-bin block. It is **not** an expected limit: AnalysisSF weights, TROTA SF/uncertainty, other High-dM categories, control-region constraints, and systematics are absent."
    )
    lines.append("")
    signal_table = []
    for signal in signals:
        current_z, split2_z, gain2 = sensitivity_gain(rows, backgrounds, signal, "two")
        _, split3_z, gain3 = sensitivity_gain(rows, backgrounds, signal, "three")
        _, _, fiducial_gain2 = sensitivity_gain(
            rows, backgrounds, signal, "two", scheme=FIDUCIAL_SCHEME
        )
        signal_table.append(
            (
                signal,
                finite(current_z),
                finite(split2_z),
                f"{gain2:.3f}x",
                finite(split3_z),
                f"{gain3:.3f}x",
                f"{fiducial_gain2:.3f}x",
            )
        )
    lines.extend(
        table(
            ("Signal", "Current category-1 Z", "61-bin-block Z", "Gain", "67-bin-block Z", "Gain", "61-bin Run-2-cut gain"),
            signal_table,
        )
    )

    lines.extend(["", "## Nb and jet-multiplicity diagnostics", ""])
    topology_rows = aggregate(
        rows, scheme=SCHEME, region="SR", samples=backgrounds, group_by=("nb", "njet", "nres")
    )
    diagnostic_rows = []
    for nb in ("Nb1", "Nb2", "Nb3plus"):
        for njet in ("Nj5to6", "Nj7plus"):
            values = [topology_rows.get((nb, njet, n), {}).get("sumw", 0.0) for n in NRES]
            diagnostic_rows.append(
                (
                    nb,
                    njet,
                    finite(values[0]),
                    finite(values[1] + values[2]),
                    percent(values[1] + values[2], sum(values)),
                )
            )
    lines.extend(table(("Nb", "Njet", "Nres=0", "Nres>=1", "Nres>=1 fraction"), diagnostic_rows))

    lines.extend(
        [
            "",
            "## Recommendation and required validation",
            "",
            "1. Implement an **exploratory `highdm61`** definition on top of the adopted 55-bin tail-merged baseline by replacing its six category-1 `NT0_Nb1plus_T0_W0` recoil bins with twelve bins: the same recoil edges crossed with `Nres=0` and `Nres>=1`, where `Nres` is the jet-disjoint TROTA 1% WP multiplicity.",
            "2. Keep `Nres=1` and `Nres>=2` merged initially. The three-way split is only a diagnostic until per-bin MC effective statistics and background closure are demonstrated.",
            "3. Do not add `Nres` to categories with a selected boosted top/W until exact AK4/AK8 subjet cross-cleaning can be reconstructed and validated.",
            (
                "4. The SR category-1 component has now been rebuilt with the full AnalysisSF weight bundle and checked against the canonical histogram. Before adoption, still obtain/apply the TROTA data/MC SF and uncertainty, validate LLCR/QCDCR/GCR/DY closure and transfer factors, and compare the full nuisance-aware expected limit against the adopted 55-bin baseline."
                if plot_summary is not None
                else "4. Before adoption, obtain/apply the TROTA data/MC SF and uncertainty, rebuild the full AnalysisSF-weighted histograms, validate LLCR/QCDCR/GCR/DY closure and transfer factors, and compare the full nuisance-aware expected limit against the adopted 55-bin baseline."
            ),
            "5. Preserve both the adopted 55-bin baseline and the exploratory 61-bin output so the category change remains auditable and reversible. The obsolete 60-bin precursor must not be used as the comparison baseline.",
            "",
            "## Machine-readable source",
            "",
            f"- Study schema: `{payload['schema_version']}`",
            f"- Study finished/updated: `{payload['updated_at']}`",
            f"- Wall time: `{float(payload['wall_time_seconds']):.1f} s`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot-summary", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    plot_summary = (
        json.loads(args.plot_summary.read_text()) if args.plot_summary else None
    )
    report = render(payload, plot_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
