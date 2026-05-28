#!/usr/bin/env python3
"""Build data/labeled/final_training_dataset.csv from recovered labels."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import (
    FINAL_TRAINING,
    SPLITS_JSON,
    build_final_training_dataset,
    dataset_summary,
    stratified_train_val_test,
)


def main() -> None:
    df = build_final_training_dataset()
    summary = dataset_summary(df)
    train, val, test = stratified_train_val_test(df)

    print("=== Final training dataset ===")
    print(f"Path: {FINAL_TRAINING}")
    print(f"Total rows: {summary['total_rows']}")
    print(f"\nLabel distribution:")
    for lab, cnt in sorted(summary["by_label"].items(), key=lambda x: -x[1]):
        print(f"  {cnt:4d}  {lab}")

    print(f"\nSplits (saved {SPLITS_JSON}):")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    nih = summary["by_label"].get("Climate nihilism", 0)
    print(f"\nClimate nihilism examples: {nih}")


if __name__ == "__main__":
    main()
