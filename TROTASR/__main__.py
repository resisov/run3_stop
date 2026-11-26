"""Command line for the all-hadronic stop analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


STAGES = {
    "inference": ("tt_card_model", "Run the current six-era cards and their three statistical inference stages"),
    "dy-top-subtraction": ("tt_dy_subtraction", "Collect the inclusive ttbar change in the on/off-Z control measurement"),
    "compile": ("compile_definitions", "Serialize the current object IDs and corrections"),
    "produce": ("produce", "Produce current signal intermediate ROOTs, one file and era per job"),
    "trota-friends": ("trota_friends", "Save corrected nominal TROTA tags for accepted MC inputs"),
    "histograms": ("build_highdm_histogram", "Build high-/low-dM histograms from integrated events"),
    "cr-mtb": ("llcr_mtb", "Render a saved historical CR cut diagnostic; no recount or fit"),
    "vr-data": ("validation_data", "Count observed events only in the validation regions"),
    "merge": ("merge_nominal", "Merge per-file histograms for one analysis era"),
    "background": ("build_background_estimation", "Measure the adopted background estimates"),
    "cards": ("build_nominal_grid", "Build templates and datacards for the signal grid"),
    "limits": ("combine_limits", "Calculate expected limits from existing datacards"),
    "impacts": ("combine_impacts", "Calculate nuisance impacts"),
    "cronly": ("combine_cronly", "Fit control regions"),
}

PLOTS = {
    "limits": ("plotting_limit_contour", "Expected-limit contours"),
    "vr-predictions": ("plotting_vr_predictions", "VR prefit and CR-only postfit comparisons"),
    "predictions": ("plotting_card_predictions", "SR and CR predictions from the statistical model"),
    "distributions": ("plotting_nominal_distributions", "Control-region distributions"),
    "gnn": ("plotting_gnn_controls", "Low-dM GNN control-region distributions"),
    "background": ("plotting_background_measurements", "Background measurements"),
    "transfer": ("plotting_transfer_factors", "Transfer factors"),
    "schematics": ("plotting_region_schematics", "Region-definition diagrams"),
}


def main(argv=None):
    """Select a stage without importing other stages or their dependencies."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m TROTASR",
        description=__doc__,
        epilog="Use <stage> --help for its inputs and options.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="stage", required=True)
    for name, (_, description) in STAGES.items():
        commands.add_parser(name, help=description, add_help=False, allow_abbrev=False)
    plot = commands.add_parser("plot", help="Draw existing analysis products", allow_abbrev=False)
    plots = plot.add_subparsers(dest="plot", required=True)
    for name, (_, description) in PLOTS.items():
        plots.add_parser(name, help=description, add_help=False, allow_abbrev=False)
    if not arguments:
        parser.print_help()
        return 0
    # Only parse the command prefix here. The selected implementation owns its
    # physics options, including negative values and paths containing spaces.
    prefix = 2 if arguments[0] == "plot" and len(arguments) > 1 else 1
    selected = parser.parse_args(arguments[:prefix])
    module = PLOTS[selected.plot][0] if selected.stage == "plot" else STAGES[selected.stage][0]
    previous = sys.argv
    # Native stages re-exec the selected implementation in the installed
    # numerical runtime. Give that handoff a real in-package file, not a
    # human-facing command label that cannot be resolved as a source path.
    sys.argv = [str(Path(__file__).resolve().parent/'workflows'/(module+'.py')),
                *arguments[prefix:]]
    try:
        runpy.run_module(f"TROTASR.workflows.{module}", run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
