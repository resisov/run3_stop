#!/usr/bin/env python3
"""Package topology-specific templates and cards into one year-level ROOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TOPOLOGIES = ("T2tt", "T2bW", "T2tb")
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def canonical_lowdm_layout() -> tuple[list[str], int]:
    """Read and validate the one supported Low-dM SR layout."""
    definition = json.loads(CONFIG_PATH.read_text())["sr_binning"]
    labels = list(definition["category_labels"])
    bins_per_category = int(definition["bins_per_category"])
    total = sum(
        len(definition["edges_by_category"][label]) - 1 for label in labels
    )
    if len(labels) != 6 or bins_per_category != 5 or total != 30:
        raise RuntimeError("config.json is not the adopted 6 x 5 = 30-bin SR")
    return labels, bins_per_category


def validate_lowdm_template(root_file, path: Path) -> None:
    """Reject legacy 20-bin or otherwise incompatible Low-dM templates."""
    labels, bins_per_category = canonical_lowdm_layout()
    for label in labels:
        directory = root_file.GetDirectory(f"SR_{label}")
        if not directory:
            raise KeyError(f"{path} is missing SR_{label}")
        for key in directory.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TH1") and obj.GetNbinsX() != bins_per_category:
                raise ValueError(
                    f"{path}: SR_{label}/{key.GetName()} has "
                    f"{obj.GetNbinsX()} bins, expected {bins_per_category}"
                )


def assignment(value: str) -> tuple[str, Path]:
    topology, separator, raw_path = value.partition("=")
    if separator != "=" or topology not in TOPOLOGIES or not raw_path:
        raise argparse.ArgumentTypeError("expected TOPOLOGY=/absolute/path")
    return topology, Path(raw_path)


def copy_directory(source, target, *, signal_only: bool = False) -> None:
    import ROOT

    for key in source.GetListOfKeys():
        obj = key.ReadObj()
        name = key.GetName()
        if obj.InheritsFrom("TDirectory"):
            # Update templates intentionally contain the same channel
            # directories as the baseline template.  Reuse those directories
            # and overwrite/add their objects instead of attempting mkdir on
            # an existing name (which returns a null TDirectory pointer).
            child = target.GetDirectory(name)
            if not child:
                child = target.mkdir(name)
            if not child:
                raise OSError(f"cannot create or open ROOT directory {name}")
            copy_directory(obj, child, signal_only=signal_only)
        else:
            if signal_only and not name.startswith("signal_"):
                continue
            target.cd()
            obj.Write(name, ROOT.TObject.kOverwrite)


def rewrite_card(source: Path, target: Path, topology: str, root_path: Path) -> None:
    lines = []
    for raw_line in source.read_text().splitlines():
        fields = raw_line.split()
        if fields and fields[0] == "shapes" and len(fields) >= 5:
            fields[3] = str(root_path)
            fields[4] = f"{topology}/{fields[4]}"
            if len(fields) >= 6:
                fields[5] = f"{topology}/{fields[5]}"
            raw_line = " ".join(fields)
        lines.append(raw_line)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, choices=("2024", "2025"))
    parser.add_argument(
        "--region", required=True, choices=("highdm", "lowdm")
    )
    parser.add_argument(
        "--template", action="append", required=True, type=assignment
    )
    parser.add_argument(
        "--update-template", action="append", default=[], type=assignment
    )
    parser.add_argument(
        "--cards", action="append", required=True, type=assignment
    )
    parser.add_argument(
        "--update-cards", action="append", default=[], type=assignment
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import ROOT

    # Objects are written explicitly. Avoid ROOT's global directory list,
    # which makes closing large template files unnecessarily expensive.
    ROOT.TH1.AddDirectory(False)

    templates = dict(args.template)
    template_updates = dict(args.update_template)
    card_directories = dict(args.cards)
    card_updates = dict(args.update_cards)
    if templates.keys() != card_directories.keys():
        raise RuntimeError("template and card topology sets differ")

    args.output.mkdir(parents=True, exist_ok=True)
    output_root = args.output / f"{args.region}_{args.year}.root"
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"cannot create {output_root}")
    try:
        for topology in TOPOLOGIES:
            if topology not in templates:
                continue
            source = ROOT.TFile.Open(str(templates[topology]), "READ")
            if not source or source.IsZombie():
                raise OSError(f"cannot read {templates[topology]}")
            try:
                if args.region == "lowdm":
                    validate_lowdm_template(source, templates[topology])
                target = root_file.mkdir(topology)
                copy_directory(source, target)
            finally:
                source.Close()
            update_path = template_updates.get(topology)
            if update_path is not None:
                update = ROOT.TFile.Open(str(update_path), "READ")
                if not update or update.IsZombie():
                    raise OSError(f"cannot read {update_path}")
                try:
                    if args.region == "lowdm":
                        validate_lowdm_template(update, update_path)
                    # An update template contributes newly produced mass
                    # points only.  Data and background objects remain exactly
                    # those of the audited baseline template.
                    copy_directory(update, target, signal_only=True)
                finally:
                    update.Close()
    finally:
        root_file.Close()

    cards_written = {}
    for topology, card_directory in card_directories.items():
        target_directory = args.output / "cards" / topology
        written = []
        for source in sorted(card_directory.glob("datacard_mStop*_mLSP*.txt")):
            target = target_directory / source.name
            rewrite_card(source, target, topology, output_root)
            written.append(str(target))
        for source in sorted(
            card_updates.get(topology, Path("/__missing__")).glob(
                "datacard_mStop*_mLSP*.txt"
            )
        ):
            target = target_directory / source.name
            rewrite_card(source, target, topology, output_root)
            if str(target) not in written:
                written.append(str(target))
        cards_written[topology] = written

    manifest = {
        "status": "combine_inputs_ready",
        "year": args.year,
        "region": args.region,
        "template_root": str(output_root),
        "topology_templates": {
            topology: str(path) for topology, path in templates.items()
        },
        "topology_template_updates": {
            topology: str(path) for topology, path in template_updates.items()
        },
        "cards": cards_written,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "template_root": str(output_root),
        "card_counts": {
            topology: len(paths) for topology, paths in cards_written.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
