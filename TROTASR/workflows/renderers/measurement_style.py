"""Verbatim legacy rendering fragments; no data or analysis path dependencies."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep


CMS_LABEL_FONT_SIZE = 17


FIGURE_SIZE = (8.0, 8.0)


def _apply_style() -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "axes.linewidth": 1.5,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )
