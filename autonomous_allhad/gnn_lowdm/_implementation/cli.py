"""Small dispatcher shared by the stable Low-dM command-line entry points."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import TypeAlias


Command: TypeAlias = str | tuple[str, str]


def dispatch(commands: Mapping[str, Command], description: str) -> int:
    """Dispatch ``<command> [arguments]`` to an internal module's ``main``."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(description)
        print("\nCommands:")
        width = max(map(len, commands))
        for name, command in commands.items():
            _, purpose = command if isinstance(command, tuple) else (command, command)
            print(f"  {name:<{width}}  {purpose}")
        print("\nUse '<command> --help' for stage-specific arguments.")
        return 0
    command = sys.argv[1]
    if command not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"unknown command {command!r}; choose one of: {choices}")
    target = commands[command]
    target = target[0] if isinstance(target, tuple) else target
    module_name, separator, function_name = target.partition(":")
    module = importlib.import_module(module_name)
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    result = getattr(module, function_name if separator else "main")()
    return 0 if result is None else int(result)
