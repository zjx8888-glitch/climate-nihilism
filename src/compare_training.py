#!/usr/bin/env python3
# TODO(Liu): add TF-IDF baseline comparison results here (separate from ClimateBERT outputs)
"""
DEPRECATED for primary reporting — use train_and_evaluate.py on recovered data.

Legacy: compare baseline vs augmented weak labels.
See outputs/recovered_vs_old_results.md for 335 vs 1844 comparison.

Saves:
  outputs/model_comparison_baseline_vs_augmented.json
  outputs/figures/training_baseline_vs_augmented.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_utils import (
    AUGMENTED_TRAINING,
    FIGURES,
    FINAL_TRAINING,
    OUTPUTS,
    build_augmented_training_dataset,
    build_final_training_dataset,
    dataset_summary,
)
from train_and_evaluate import (
    RANDOM_STATE,
    evaluate,
    predict_sklearn_pipe,
    stratified_split,
    train_tfidf_lr,
    train_tfidf_svm,
)
from taxonomy import NIHILISM_FOCUS


def train_eval_split(df: pd.DataFrame) -> Dict[str, Any]:
    train_df, test_df = stratified_split(df)
    X_train = train_df["body"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["body"].tolist()
    y_test = test_df["label"].tolist()
    labels_present = sorted(set(y_train) | set(y_test))

    results = {}
    lr = train_tfidf_lr(X_train, y_train)
    results["TF-IDF + LR"] = evaluate(y_test, predict_sklearn_pipe(lr, X_test), labels_present)

    svm = train_tfidf_svm(X_train, y_train)
    results["TF-IDF + SVM"] = evaluate(
        y_test, predict_sklearn_pipe(svm, X_test), labels_present
    )
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "models": results,
        "best_model": max(
            results.keys(), key=lambda k: results[k]["macro_f1"]
        ),
    }


def plot_comparison(baseline: dict, augmented: dict, path: Path) -> None:
    models = list(baseline["models"].keys())
    x = range(len(models))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    base_macro = [baseline["models"][m]["macro_f1"] for m in models]
    aug_macro = [augmented["models"][m]["macro_f1"] for m in models]
    axes[0].bar([i - w / 2 for i in x], base_macro, w, label="Baseline (335 manual)")
    axes[0].bar([i + w / 2 for i in x], aug_macro, w, label="Augmented (+ high-conf weak)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Macro F1 — test set")
    axes[0].legend()

    base_nih = [baseline["models"][m]["nihilism_f1"] for m in models]
    aug_nih = [augmented["models"][m]["nihilism_f1"] for m in models]
    axes[1].bar([i - w / 2 for i in x], base_nih, w, label="Baseline")
    axes[1].bar([i + w / 2 for i in x], aug_nih, w, label="Augmented")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1")
    axes[1].set_title(f"{NIHILISM_FOCUS} — test set")
    axes[1].legend()

    fig.suptitle(
        f"Train n={baseline['n_train']} vs {augmented['n_train']} "
        f"(test n={baseline['n_test']})",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== Dataset summaries ===\n")
    baseline_df = build_final_training_dataset()
    print("Baseline (final_training_dataset.csv):")
    print(json.dumps(dataset_summary(baseline_df), indent=2))

    augmented_df = build_augmented_training_dataset()
    print("\nAugmented (augmented_training_dataset.csv):")
    print(json.dumps(dataset_summary(augmented_df), indent=2))

    print("\n=== Training baseline ===")
    baseline_results = train_eval_split(baseline_df)
    print(
        f"Best: {baseline_results['best_model']} "
        f"macro_f1={baseline_results['models'][baseline_results['best_model']]['macro_f1']:.3f}"
    )

    print("\n=== Training augmented ===")
    augmented_results = train_eval_split(augmented_df)
    print(
        f"Best: {augmented_results['best_model']} "
        f"macro_f1={augmented_results['models'][augmented_results['best_model']]['macro_f1']:.3f}"
    )

    out = {
        "baseline": baseline_results,
        "augmented": augmented_results,
        "baseline_summary": dataset_summary(baseline_df),
        "augmented_summary": dataset_summary(augmented_df),
    }
    out_path = OUTPUTS / "model_comparison_baseline_vs_augmented.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    plot_comparison(
        baseline_results,
        augmented_results,
        FIGURES / "training_baseline_vs_augmented.png",
    )

    print(f"\nSaved {out_path}")
    print(f"Saved {FIGURES / 'training_baseline_vs_augmented.png'}")

    print("\n=== Delta (augmented - baseline) best LR ===")
    b = baseline_results["models"]["TF-IDF + LR"]
    a = augmented_results["models"]["TF-IDF + LR"]
    print(f"  accuracy: {a['accuracy'] - b['accuracy']:+.3f}")
    print(f"  macro_f1: {a['macro_f1'] - b['macro_f1']:+.3f}")
    print(f"  nihilism_f1: {a['nihilism_f1'] - b['nihilism_f1']:+.3f}")


if __name__ == "__main__":
    main()
