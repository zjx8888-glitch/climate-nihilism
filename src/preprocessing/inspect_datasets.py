#!/usr/bin/env python3
"""Print exact row counts by source and label for all training datasets.

# TODO(Madeleine): improve preprocessing pipeline
"""

import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import (
    AUTO_LABELED,
    AUGMENTED_TRAINING,
    FINAL_TRAINING,
    HIGH_CONFIDENCE_TRAINING,
    HUMAN_VERIFIED,
    MANUAL_LABELED,
    dataset_summary,
    load_high_confidence_weak,
    load_human_verified,
    load_manual_labeled,
    build_final_training_dataset,
    build_augmented_training_dataset,
)


def main() -> None:
    manual = load_manual_labeled()
    verified = load_human_verified()
    high_conf = load_high_confidence_weak()

    print("=" * 60)
    print("1. ORIGINAL MANUAL LABELS")
    print(f"   File: {MANUAL_LABELED}")
    print(f"   Rows with label: {len(manual)}")
    print(manual["label"].value_counts().to_string())

    print("\n2. WEAK LABELS (auto_labeled_comments.csv)")
    print(f"   File: {AUTO_LABELED}")
    auto_n = sum(1 for _ in open(AUTO_LABELED, encoding="utf-8")) - 1
    print(f"   Total weak-labeled rows: {auto_n}")
    print("   (Not used in final_training_dataset unless high-confidence selected)")

    print("\n3. HUMAN-REVIEWED AUTO LABELS")
    print(f"   File: {HUMAN_VERIFIED}")
    if HUMAN_VERIFIED.exists():
        raw = pd.read_csv(HUMAN_VERIFIED)
        print(f"   Total rows in file: {len(raw)}")
        if len(raw) and "decision" in raw.columns:
            print(raw["decision"].value_counts().to_string())
    print(f"   Accepted/changed (used in training): {len(verified)}")

    print("\n4. HIGH-CONFIDENCE WEAK LABELS")
    print(f"   File: {HIGH_CONFIDENCE_TRAINING}")
    print(f"   Rows: {len(high_conf)}")
    if len(high_conf):
        print(high_conf["label"].value_counts().to_string())

    print("\n5. FINAL TRAINING (manual + human-verified only)")
    final_df = build_final_training_dataset()
    print(f"   File: {FINAL_TRAINING}")
    print(f"   Rows after dedupe/filter: {len(final_df)}")
    print(json.dumps(dataset_summary(final_df), indent=2))

    print("\n6. AUGMENTED TRAINING (+ high-confidence weak)")
    if HIGH_CONFIDENCE_TRAINING.exists() and len(high_conf):
        aug_df = build_augmented_training_dataset()
        print(f"   File: {AUGMENTED_TRAINING}")
        print(f"   Rows: {len(aug_df)}")
        print(json.dumps(dataset_summary(aug_df), indent=2))
    else:
        print("   (Run: python src/build_high_confidence_dataset.py first)")


if __name__ == "__main__":
    main()
