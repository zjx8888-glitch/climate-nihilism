#!/usr/bin/env python3
"""
Build final_training_dataset_v2.csv and splits_v2.json from cleaned_data_2.csv.

Usage:
  python src/climatebert/prepare_v2_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import body_hash
from common.paths import (
    CLEANED_DATA_V2,
    DATA_LABELED,
    DATASET_V2_INSPECTION,
    FINAL_TRAINING_V2,
    RANDOM_STATE,
    SPLITS_V2_JSON,
    ensure_project_dirs,
)
from common.taxonomy import CANONICAL_LABELS, NIHILISM_FOCUS, normalize_label

REMOVED_PATTERN = re.compile(r"\[removed\]|\[deleted\]", re.I)


def inspect_raw(df: pd.DataFrame) -> Dict[str, Any]:
    lc = df["label_clean"]
    bodies = df["body"].fillna("").astype(str)
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "missing_label_clean": int(lc.isna().sum()),
        "empty_label_clean": int(lc.fillna("").astype(str).str.strip().eq("").sum()),
        "label_clean_distribution_raw": dict(
            Counter(lc.dropna().astype(str).str.strip().str.lower())
        ),
        "duplicate_bodies": int(bodies.duplicated().sum()),
        "short_bodies_le10": int((bodies.str.len() <= 10).sum()),
        "removed_deleted_bodies": int(bodies.str.contains(REMOVED_PATTERN).sum()),
    }


def load_and_clean_v2() -> pd.DataFrame:
    if not CLEANED_DATA_V2.exists():
        raise FileNotFoundError(
            f"Missing {CLEANED_DATA_V2}. Copy cleaned_data-2.csv to data/labeled/cleaned_data_2.csv"
        )
    df = pd.read_csv(CLEANED_DATA_V2, encoding="utf-8", low_memory=False)
    raw_inspection = inspect_raw(df)

    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label_clean"].map(normalize_label)
    df = df[df["label"].notna()].copy()
    df = df[df["body"].str.len() > 10].copy()
    df = df[~df["body"].str.contains(REMOVED_PATTERN, regex=True)].copy()
    df["body_hash"] = df["body"].map(body_hash)
    df = df.drop_duplicates(subset=["body_hash"], keep="first")

    out_cols = [
        "body_hash",
        "body",
        "label",
        "id",
        "subreddit.name",
        "created_utc",
        "sentiment",
        "label_clean",
        "source",
    ]
    df["source"] = "cleaned_data_v2"
    for c in out_cols:
        if c not in df.columns and c != "source":
            pass
    df = df[[c for c in out_cols if c in df.columns]]

    inspection = {
        **raw_inspection,
        "rows_after_cleaning": len(df),
        "label_distribution_canonical": dict(Counter(df["label"])),
        "nihilism_count": int((df["label"] == NIHILISM_FOCUS).sum()),
        "unmapped_label_clean_samples": [],
    }
    return df, inspection


def stratified_splits(df: pd.DataFrame) -> Dict[str, Any]:
    work = df.copy()
    counts = work["label"].value_counts()
    rare = counts[counts < 2].index.tolist()
    for lab in rare:
        work.loc[work["label"] == lab, "label"] = "climate opinion critique"

    train_val, test = train_test_split(
        work,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=work["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )

    meta = {
        "random_state": RANDOM_STATE,
        "test_size": 0.15,
        "val_size": 0.15,
        "dataset_version": "v2",
        "source_file": str(CLEANED_DATA_V2),
        "n_total": len(work),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "nihilism_total": int((work["label"] == NIHILISM_FOCUS).sum()),
        "train_hashes": train["body_hash"].tolist(),
        "val_hashes": val["body_hash"].tolist(),
        "test_hashes": test["body_hash"].tolist(),
    }
    return meta, train, val, test


def main() -> None:
    ensure_project_dirs()
    DATA_LABELED.mkdir(parents=True, exist_ok=True)

    df, inspection = load_and_clean_v2()
    df.to_csv(FINAL_TRAINING_V2, index=False)

    meta, train, val, test = stratified_splits(df)
    SPLITS_V2_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    inspection["split_sizes"] = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    DATASET_V2_INSPECTION.write_text(
        json.dumps(inspection, indent=2), encoding="utf-8"
    )

    print("=== Dataset v2 inspection ===")
    print(f"Raw rows: {inspection['row_count']}")
    print(f"After cleaning: {inspection['rows_after_cleaning']}")
    print(f"Missing label_clean (raw): {inspection['missing_label_clean']}")
    print(f"Duplicate bodies (raw): {inspection['duplicate_bodies']}")
    print(f"Climate nihilism: {inspection['nihilism_count']}")
    print(f"\nSaved: {FINAL_TRAINING_V2}")
    print(f"Saved: {SPLITS_V2_JSON}")
    print(f"Saved: {DATASET_V2_INSPECTION}")
    print(f"Splits — train {len(train)}, val {len(val)}, test {len(test)}")


if __name__ == "__main__":
    main()
