"""Canonical plotting interface, including all supplementary GNN figures."""

from __future__ import annotations

from ._implementation.cli import dispatch


COMMANDS = {
    "training-curves": ("autonomous_allhad.gnn_lowdm._implementation.supplementary_plots:main_training_curves", "plot train/validation loss, accuracy, and AUC"),
    "roc": ("autonomous_allhad.gnn_lowdm._implementation.supplementary_plots:main_roc", "plot independent-test ROC curves and AUC"),
    "shap": ("autonomous_allhad.gnn_lowdm._implementation.supplementary_plots:main_shap", "plot global-feature SHAP importance and beeswarm"),
    "cr": ("autonomous_allhad.gnn_lowdm._implementation.plot_control_regions", "plot inclusive Low-dM control regions"),
    "limit": ("autonomous_allhad.gnn_lowdm._implementation.collect_plot_diagonal_v3_limits", "plot Low-dM-only expected limits"),
    "combined-limit": ("autonomous_allhad.gnn_lowdm._implementation.collect_plot_highdm79minus6_lowdm30", "plot combined 73+30-bin expected limits"),
    "schematic": ("autonomous_allhad.gnn_lowdm._implementation.plot_diagonal_v3_model_schematic", "draw the hybrid-GNN model schematic"),
    "publish": ("autonomous_allhad.gnn_lowdm._implementation.publish_lowdm_gnn_web", "publish the generated plot gallery"),
}


def main() -> int:
    return dispatch(COMMANDS, __doc__ or "Low-dM plotting")


if __name__ == "__main__":
    raise SystemExit(main())
