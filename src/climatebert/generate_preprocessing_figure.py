#!/usr/bin/env python3
"""Poster figure: text preprocessing — compact side-by-side before/after layout."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import CLIMATEBERT_V2_OUT

OUT = CLIMATEBERT_V2_OUT / "poster_figures" / "preprocessing_example_poster.png"

NAVY = "#1d3557"
GRAY_BG = "#f1f3f5"
BLUE_BG = "#e8f4f8"

RAW = (
    "Climate change isn't real! Imagine thinking\n"
    "that it is [emoji]\n"
    "\n"
    "&gt;you're completely wrong. Alright let's\n"
    "compare your source to mine.\n"
    "[here's one]\n"
    "(https://www.forbes.com/\n"
    ".../voters-dont/)"
)

CLEAN = (
    "Climate change isn't real! Imagine thinking\n"
    "that it is :rolling_on_the_floor_laughing:\n"
    "\n"
    ">you're completely wrong. Alright let's\n"
    "compare your source to mine.\n"
    "[here's one]([[URL]])"
)

CAPTION = "Normalize emoji, HTML entities, and URLs before labeling and modeling"


def _side_box(ax, x, y, w, h, facecolor, title, body, mono: bool = True):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=facecolor,
        edgecolor=NAVY,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h - 0.22, title, fontsize=10, fontweight="bold", color=NAVY, va="top")
    txt = ax.text(
        x + 0.12,
        y + h - 0.48,
        body,
        fontsize=8.2,
        color="#212529",
        va="top",
        ha="left",
        family="DejaVu Sans Mono" if mono else "DejaVu Sans",
        linespacing=1.22,
        clip_on=True,
    )
    txt.set_clip_path(patch)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Compact layout sized to content
    box_w = 4.35
    box_h = 2.05
    left_x = 0.55
    gap = 0.45
    right_x = left_x + box_w + gap
    box_y = 0.62

    fig_w = right_x + box_w + 0.55
    fig_h = box_y + box_h + 1.05

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2,
        fig_h - 0.28,
        "Text preprocessing (example)",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )

    _side_box(ax, left_x, box_y, box_w, box_h, GRAY_BG, "Raw comment", RAW, mono=False)
    _side_box(ax, right_x, box_y, box_w, box_h, BLUE_BG, "Cleaned text", CLEAN, mono=True)

    mid_y = box_y + box_h / 2
    ax.add_patch(
        FancyArrowPatch(
            (left_x + box_w + 0.06, mid_y),
            (right_x - 0.06, mid_y),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.5,
            color=NAVY,
        )
    )

    ax.text(
        fig_w / 2,
        0.18,
        CAPTION,
        ha="center",
        fontsize=9,
        color="#457b9d",
        style="italic",
    )

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
