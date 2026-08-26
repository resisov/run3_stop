"""Configuration validation and a small, non-eval array-expression language."""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .profiles import resolve_profile

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
}


class Expression:
    """Evaluate arithmetic and boolean array expressions without Python eval."""

    def __init__(self, source: str):
        self.source = str(source)
        self.tree = ast.parse(self.source, mode="eval").body
        self._validate(self.tree)
        self.names = frozenset(self._names(self.tree))

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            self._validate(node.operand)
            return
        if isinstance(node, ast.Compare) and all(
            type(item) in _COMPARE for item in node.ops
        ):
            self._validate(node.left)
            for item in node.comparators:
                self._validate(item)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            for item in node.args:
                self._validate(item)
            return
        raise ValueError(
            f"syntax is not allowed in expression {self.source!r}: {ast.dump(node)}"
        )

    def _names(self, node: ast.AST) -> set[str]:
        names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        return names - set(_FUNCTIONS)

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        missing = self.names - set(values)
        if missing:
            raise KeyError(
                f"expression {self.source!r} is missing names: {sorted(missing)}"
            )
        return self._evaluate(self.tree, values)

    def _evaluate(self, node: ast.AST, values: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](
                self._evaluate(node.left, values), self._evaluate(node.right, values)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self._evaluate(node.operand, values))
        if isinstance(node, ast.Compare):
            left = self._evaluate(node.left, values)
            result = True
            for operation, comparator in zip(node.ops, node.comparators):
                if type(operation) not in _COMPARE:
                    raise ValueError(f"comparison is not allowed in {self.source!r}")
                right = self._evaluate(comparator, values)
                result = operator.and_(result, _COMPARE[type(operation)](left, right))
                left = right
            return result
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](
                *[self._evaluate(arg, values) for arg in node.args]
            )
        raise ValueError(
            f"syntax is not allowed in expression {self.source!r}: {ast.dump(node)}"
        )


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    payload = resolve_profile(json.loads(path.read_text()))
    validate_config(payload)
    return payload


def _edges(config: Mapping[str, Any], name: str) -> list[float]:
    values = [float(value) for value in config["axes"][name]]
    if len(values) < 2 or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"axes.{name} must be strictly increasing")
    return values


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "measurement",
        "year",
        "tag",
        "probe",
        "pair",
        "axes",
        "fit",
        "correction",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing: {sorted(missing)}")
    if int(config["schema_version"]) != 1:
        raise ValueError("only schema_version=1 is supported")
    for role in ("tag", "probe", "spectator"):
        if role not in config:
            continue
        item = config[role]
        if (
            not item.get("collection")
            or not item.get("fields")
            or not item.get("selection")
        ):
            raise ValueError(f"{role} requires collection, fields, and selection")
        selection = Expression(str(item["selection"]))
        unknown = selection.names - set(map(str, item["fields"]))
        if unknown:
            raise ValueError(
                f"{role}.selection uses fields absent from {role}.fields: {sorted(unknown)}"
            )
    probe = config["probe"]
    if not probe.get("pass") or not probe.get("pt") or not probe.get("eta"):
        raise ValueError("probe requires pass, pt, and eta expressions")
    for key in ("pass", "pt", "eta"):
        expression = Expression(str(probe[key]))
        unknown = expression.names - set(map(str, probe["fields"]))
        if unknown:
            raise ValueError(
                f"probe.{key} uses fields absent from probe.fields: {sorted(unknown)}"
            )
    Expression(str(config["pair"].get("selection", "True")))
    _edges(config, "pt_edges_gev")
    _edges(config, "abseta_edges")
    window = [float(value) for value in config["pair"]["mass_window_gev"]]
    peak = [float(value) for value in config["fit"]["peak_bounds_gev"]]
    if (
        len(window) != 2
        or len(peak) != 2
        or not window[0] < peak[0] < peak[1] < window[1]
    ):
        raise ValueError("mass window must strictly contain fit.peak_bounds_gev")
    if int(config["fit"].get("mass_bins", 0)) < 10:
        raise ValueError("fit.mass_bins must be at least 10")
    if config["fit"].get("signal_model", "gaussian") not in {
        "gaussian",
        "double_gaussian",
        "crystal_ball",
        "voigt",
    }:
        raise ValueError("unsupported signal model")
    if config["fit"].get("background_model", "exponential") not in {
        "exponential",
        "linear",
        "chebyshev2",
    }:
        raise ValueError("unsupported background model")
    trigger = config.get("reference_trigger", {})
    if trigger.get("match_tag") and not trigger.get("object_id"):
        raise ValueError("tag trigger matching requires reference_trigger.object_id")


def expression_names(source: str) -> set[str]:
    return set(Expression(source).names)
