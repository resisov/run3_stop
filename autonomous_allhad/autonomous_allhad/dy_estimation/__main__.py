"""Single command-line entry point for the DY-estimation workflow."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "prepare-features": "prepare_features",
    "build-features": "feature_stage",
    "merge-features": "merge_features",
    "prepare-lowdm": "prepare_lowdm",
    "run-lowdm-partition": "run_lowdm_partition",
    "merge-lowdm": "merge_lowdm",
    "report": "report",
    "publish": "publish",
    "validate": "validate",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return (
        "Usage: python -m autonomous_allhad.dy_estimation <command> [options]\n\n"
        f"Commands:\n{commands}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(usage())
        return 0
    command, *remainder = arguments
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"{__package__}.{module_name}")
    return int(module.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())
