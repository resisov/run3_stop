"""Canonical validation commands for the frozen Low-dM GNN."""

from __future__ import annotations

from ._implementation.cli import dispatch


COMMANDS = {
    "numpy-parity": ("autonomous_allhad.gnn_lowdm._implementation.validate_diagonal_v3_numpy_inference", "compare PyTorch and portable NumPy inference"),
    "feature-parity": ("autonomous_allhad.gnn_lowdm._implementation.validate_diagonal_v3_region_feature_parity", "compare cache and SR/CR feature reconstruction"),
    "templates": ("autonomous_allhad.gnn_lowdm._implementation.validate_diagonal_v3_srcr_inputs", "validate SR and control templates"),
}


def main() -> int:
    return dispatch(COMMANDS, __doc__ or "Low-dM validation")


if __name__ == "__main__":
    raise SystemExit(main())
