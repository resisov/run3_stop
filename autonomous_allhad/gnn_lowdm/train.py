"""Canonical data preparation and training commands for the Low-dM GNN."""

from __future__ import annotations

from ._implementation.cli import dispatch


COMMANDS = {
    "manifest": ("autonomous_allhad.gnn_lowdm._implementation.build_full_campaign_manifest", "audit intermediate ROOT inputs"),
    "prepare-cache": ("autonomous_allhad.gnn_lowdm._implementation.prepare_diagonal_v3_cache_condor", "prepare the EOS-only cache campaign"),
    "cache-worker": ("autonomous_allhad.gnn_lowdm._implementation.build_expanded_feature_cache", "build one feature-cache shard"),
    "finalize-cache": ("autonomous_allhad.gnn_lowdm._implementation.finalize_diagonal_v3_cache", "validate and finalize the cache"),
    "fit": ("autonomous_allhad.gnn_lowdm._implementation.train_diagonal_v3_significance", "train and select the GNN on train/validation"),
}


def main() -> int:
    return dispatch(COMMANDS, __doc__ or "Low-dM training")


if __name__ == "__main__":
    raise SystemExit(main())
