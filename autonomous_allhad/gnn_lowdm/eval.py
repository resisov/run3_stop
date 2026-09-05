"""Canonical inference, template, and limit-campaign commands."""

from __future__ import annotations

from ._implementation.cli import dispatch


COMMANDS = {
    "test": ("autonomous_allhad.gnn_lowdm._implementation.evaluate_diagonal_v3_test", "evaluate the frozen 70% test partition"),
    "cr-partial": ("autonomous_allhad.gnn_lowdm._implementation.build_diagonal_v3_cr_nnout_partial", "evaluate one control-region ROOT group"),
    "cr-merge": ("autonomous_allhad.gnn_lowdm._implementation.merge_diagonal_v3_cr_nnout_partials", "merge and audit control-region partials"),
    "signal-prepare": ("autonomous_allhad.gnn_lowdm._implementation.prepare_all_signal_template_condor", "prepare all-signal EOS inference"),
    "signal-partial": ("autonomous_allhad.gnn_lowdm._implementation.evaluate_all_signal_template_partial", "evaluate one signal-cache shard"),
    "signal-merge": ("autonomous_allhad.gnn_lowdm._implementation.merge_all_signal_templates", "merge final 30-bin signal templates"),
    "limits-prepare": ("autonomous_allhad.gnn_lowdm._implementation.prepare_highdm79minus6_lowdm30_condor", "prepare the 73+30-bin limit campaign"),
}


def main() -> int:
    return dispatch(COMMANDS, __doc__ or "Low-dM evaluation")


if __name__ == "__main__":
    raise SystemExit(main())
