"""Configurable event weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import Expression


def required_fields(config: Mapping[str, Any]) -> set[str]:
    fields: set[str] = set()
    expressions = [
        config.get("mc_nominal", "1.0"),
        *config.get("mc_variations", {}).values(),
    ]
    for source in expressions:
        fields.update(Expression(str(source)).names)
    for correction in config.get("corrections", []):
        for item in correction.get("inputs", []):
            if isinstance(item, Mapping) and "field" in item:
                fields.add(str(item["field"]))
    return fields


class WeightSet:
    def __init__(self, config: Mapping[str, Any], base_dir: Path | str):
        self.config = config
        self.corrections = []
        if config.get("corrections"):
            import correctionlib

            for item in config["corrections"]:
                path = Path(base_dir) / str(item["file"])
                evaluator = correctionlib.CorrectionSet.from_file(str(path))[
                    str(item["name"])
                ]
                self.corrections.append((item, evaluator))

    @staticmethod
    def _values(arrays: Any) -> dict[str, Any]:
        return {name: arrays[name] for name in arrays.fields}

    @staticmethod
    def _arguments(
        specification: Mapping[str, Any], arrays: Any, variation: Any
    ) -> list[Any]:
        arguments = []
        for item in specification.get("inputs", []):
            if not isinstance(item, Mapping):
                arguments.append(item)
            elif "field" in item:
                arguments.append(np.asarray(arrays[str(item["field"])]))
            elif "value" in item:
                arguments.append(item["value"])
            elif item.get("variation"):
                arguments.append(variation)
            else:
                raise ValueError(f"invalid correction input: {item}")
        return arguments

    def evaluate(self, arrays: Any) -> dict[str, np.ndarray]:
        values = self._values(arrays)
        base = np.asarray(
            Expression(str(self.config.get("mc_nominal", "1.0"))).evaluate(values),
            dtype=float,
        )
        if base.ndim == 0:
            base = np.full(len(arrays), float(base))
        nominal = base.copy()
        nominal_factors = []
        for specification, evaluator in self.corrections:
            token = specification.get("nominal", "nominal")
            factor = np.asarray(
                evaluator.evaluate(*self._arguments(specification, arrays, token)),
                dtype=float,
            )
            nominal_factors.append(factor)
            nominal *= factor
        output = {"nominal": nominal}
        for name, source in self.config.get("mc_variations", {}).items():
            varied = np.asarray(Expression(str(source)).evaluate(values), dtype=float)
            if varied.ndim == 0:
                varied = np.full(len(arrays), float(varied))
            for factor in nominal_factors:
                varied *= factor
            output[str(name)] = varied
        for index, (specification, evaluator) in enumerate(self.corrections):
            for name, token in specification.get("variations", {}).items():
                varied = base.copy()
                for factor_index, factor in enumerate(nominal_factors):
                    if factor_index != index:
                        varied *= factor
                varied *= np.asarray(
                    evaluator.evaluate(*self._arguments(specification, arrays, token)),
                    dtype=float,
                )
                if name in output:
                    raise ValueError(f"duplicate weight variation: {name}")
                output[str(name)] = varied
        return output
