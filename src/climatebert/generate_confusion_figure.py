#!/usr/bin/env python3
"""Regenerate 3-class nihilism-focus confusion matrix for poster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import CLIMATEBERT_V2_OUT
from common.taxonomy import NIHILISM_FOCUS

OUT = CLIMATEBERT_V2_OUT / "poster_figures" / "confusion_nihilism_focus_3class.png"
METRICS = CLIMATEBERT_V2_OUT / "climatebert_metrics.json"

FOCUS_LABELS = [
    NIHILISM_FOCUS,
    "climate anxiety",
    "Climate nihilism critique",
]

TITLE = "climate bert confusion matrix"


def main() -> None:
    meta = json.loads(METRICS.read_text(encoding="utf-8"))
    mc = meta["approaches"]["embedding_lr_multiclass"]
    labels = mc["labels"]
    cm = np.array(mc["confusion_matrix"])
    idx = [labels.index(lab) for lab in FOCUS_LABELS]
    sub = cm[np.ix_(idx, idx)]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        sub,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=FOCUS_LABELS,
        yticklabels=FOCUS_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(TITLE)
    plt.xticks(rotation=22, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
