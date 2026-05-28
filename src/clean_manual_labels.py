#!/usr/bin/env python3
"""Normalize label spelling in the 2000-comment manual labeling file."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_utils import MANUAL_LABELED, PROCESSED
from taxonomy import CANONICAL_LABELS, normalize_label

# Source file with long duplicate name (if present)
SOURCE_LONG = (
    PROCESSED
    / "preprocessed_comments_2000_to_label - preprocessed_comments_2000_to_label.csv"
)


def clean_file(source: Path, dest: Path) -> pd.DataFrame:
    df = pd.read_csv(source, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    df["label_original"] = df["label"].astype(str)
    df["label"] = df["label"].map(normalize_label)
    df["label_needs_review"] = df["label"].isna() & (
        df["label_original"].fillna("").astype(str).str.strip() != ""
    )
    return df


def main() -> None:
    source = SOURCE_LONG if SOURCE_LONG.exists() else MANUAL_LABELED
    if not source.exists():
        raise FileNotFoundError(f"No manual label file at {source}")

    df = clean_file(source, MANUAL_LABELED)
    dest = MANUAL_LABELED
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    # Keep long-named copy in sync
    if SOURCE_LONG.exists() and SOURCE_LONG.resolve() != dest.resolve():
        df.to_csv(SOURCE_LONG, index=False)

    labeled = df[df["label"].notna()]
    print(f"Wrote {len(df)} rows -> {dest}")
    print(f"  Canonical labels: {len(labeled)}")
    print(f"  Needs review: {df['label_needs_review'].sum()}")
    print(f"  Empty (unlabeled): {(df['label_original'].fillna('').str.strip() == '').sum()}")
    print("\nLabel distribution (canonical):")
    print(labeled["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
