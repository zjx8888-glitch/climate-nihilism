#!/usr/bin/env python3
"""
Legacy combined training script (pre team split).

  - ClimateBERT (Jinxi): src/climatebert/train.py
  - TF-IDF baseline (Liu): src/tfidf/train.py
  - Demo (Josh): app/streamlit_app.py

# TODO(Liu): move TF-IDF training logic into src/tfidf/train.py and retire this script.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import (
    build_final_training_dataset,
    stratified_train_val_test,
)
from common.paths import FIGURES, FINAL_TRAINING, RANDOM_STATE, TFIDF_MODELS, TFIDF_OUT, AUTO_LABELED
from common.taxonomy import CANONICAL_LABELS, NIHILISM_FOCUS, PRIORITY_LABELS

MODELS = TFIDF_MODELS
OUTPUTS = TFIDF_OUT

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)
DATASET_ROWS = 1845


def load_training_data() -> pd.DataFrame:
    if not FINAL_TRAINING.exists():
        build_final_training_dataset()
    df = pd.read_csv(FINAL_TRAINING, encoding="utf-8")
    df = df[df["label"].isin(CANONICAL_LABELS)].copy()
    return df


def evaluate_multiclass(y_true, y_pred, labels: List[str]) -> Dict[str, Any]:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    per_class = {}
    for lab in labels:
        if lab in report:
            per_class[lab] = {
                "precision": report[lab]["precision"],
                "recall": report[lab]["recall"],
                "f1": report[lab]["f1-score"],
                "support": report[lab]["support"],
            }
    nih = per_class.get(
        NIHILISM_FOCUS, {"precision": 0, "recall": 0, "f1": 0, "support": 0}
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": per_class,
        "nihilism_precision": float(nih["precision"]),
        "nihilism_recall": float(nih["recall"]),
        "nihilism_f1": float(nih["f1"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def evaluate_binary_nihilism(y_true_labels: List[str], y_pred_labels: List[str]) -> Dict[str, float]:
    y_true = [1 if y == NIHILISM_FOCUS else 0 for y in y_true_labels]
    y_pred = [1 if y == NIHILISM_FOCUS else 0 for y in y_pred_labels]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "binary_nihilism_precision": float(p),
        "binary_nihilism_recall": float(r),
        "binary_nihilism_f1": float(f1),
        "binary_nihilism_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_tfidf_lr(X_train, y_train) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=40_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ).fit(X_train, y_train)


def train_tfidf_svm(X_train, y_train) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=40_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    ).fit(X_train, y_train)


def train_embedding_lr(X_train, y_train) -> Optional[Dict]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed; skipping embedding+LR.")
        return None

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        embedder = SentenceTransformer(
            "climatebert/distilbert-base-uncased-finetuned-climatebert"
        )
        model_name = "climatebert/distilbert-base-uncased-finetuned-climatebert"
    except Exception:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    X_emb = embedder.encode(list(X_train), show_progress_bar=True, batch_size=32)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    clf.fit(X_emb, y_train)
    return {
        "type": "embedding_lr",
        "embedder_name": model_name,
        "embedder": embedder,
        "classifier": clf,
    }


def train_binary_nihilism_tfidf(X_train, y_train_labels) -> Pipeline:
    y_bin = [1 if y == NIHILISM_FOCUS else 0 for y in y_train_labels]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ).fit(X_train, y_bin)


def predict_sklearn_pipe(pipe: Pipeline, texts: List[str]) -> List[str]:
    return list(pipe.predict(texts))


def save_bundle(pipe: Pipeline, name: str) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {
        "type": "tfidf_sklearn",
        "vectorizer": pipe.named_steps["tfidf"],
        "classifier": pipe.named_steps["clf"],
    }
    path = MODELS / f"{name}.joblib"
    joblib.dump(bundle, path)
    return path


def plot_label_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["darkred" if l == NIHILISM_FOCUS else "steelblue" for l in counts.index]
    counts.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Count")
    ax.set_title(f"Training dataset label distribution (n={len(df)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: Dict[str, Dict], path: Path) -> None:
    names = [k for k in results if not k.startswith("Binary")]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, [results[n]["macro_f1"] for n in names], w, label="Macro F1")
    ax.bar(x, [results[n]["weighted_f1"] for n in names], w, label="Weighted F1")
    ax.bar(x + w, [results[n]["nihilism_f1"] for n in names], w, label="Nihilism F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model comparison (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nihilism_pr(results: Dict[str, Dict], path: Path) -> None:
    names = [k for k in results if not k.startswith("Binary")]
    p = [results[n]["nihilism_precision"] for n in names]
    r = [results[n]["nihilism_recall"] for n in names]
    f1 = [results[n]["nihilism_f1"] for n in names]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, p, w, label="Precision")
    ax.bar(x, r, w, label="Recall")
    ax.bar(x + w, f1, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{NIHILISM_FOCUS} — precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels: List[str], path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence_distribution(path: Path) -> None:
    if not AUTO_LABELED.exists():
        return
    sample = pd.read_csv(
        AUTO_LABELED, usecols=["confidence", "predicted_label"], nrows=200_000
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sample["confidence"].hist(bins=40, ax=axes[0], color="gray", edgecolor="white")
    axes[0].set_title("Weak-label confidence (auto-labeled sample)")
    for lab in PRIORITY_LABELS:
        sub = sample[sample["predicted_label"] == lab]["confidence"]
        if len(sub):
            axes[1].hist(sub, bins=30, alpha=0.5, label=lab)
    axes[1].set_title("Confidence by priority label")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nihilism_keywords(pipe: Pipeline, path: Path) -> None:
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    if NIHILISM_FOCUS not in list(clf.classes_):
        return
    idx = list(clf.classes_).index(NIHILISM_FOCUS)
    coef = clf.coef_[idx]
    names = vec.get_feature_names_out()
    top = np.argsort(coef)[-20:][::-1]
    terms = [names[i] for i in top if coef[i] > 0]
    scores = [coef[i] for i in top if coef[i] > 0][:20]
    if not terms:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(terms[::-1], scores[::-1], color="darkred")
    ax.set_title(f"Top TF-IDF terms for {NIHILISM_FOCUS}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clear_stale_outputs() -> None:
    for p in FIGURES.glob("*.png"):
        p.unlink()
    for p in MODELS.glob("*.joblib"):
        p.unlink()
    for name in ("model_metrics.json", "best_model.json"):
        p = OUTPUTS / name
        if p.name == "best_model.json":
            p = MODELS / name
        if p.exists():
            p.unlink()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs()

    df = load_training_data()
    print(f"Training dataset: {len(df)} rows (recovered labels)")
    plot_label_distribution(df, FIGURES / "label_distribution.png")

    train_df, val_df, test_df = stratified_train_val_test(df)
    print(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train = train_df["body"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["body"].tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["body"].tolist()
    y_test = test_df["label"].tolist()
    labels_present = sorted(set(y_train) | set(y_val) | set(y_test))

    results: Dict[str, Dict] = {}
    artifacts: Dict[str, str] = {}
    predictions: Dict[str, List] = {}

    # TODO(Liu): add TF-IDF baseline results here — own module: src/train_tfidf.py
    print("Training TF-IDF + Logistic Regression…")
    lr_pipe = train_tfidf_lr(X_train, y_train)
    y_pred_lr = predict_sklearn_pipe(lr_pipe, X_test)
    results["TF-IDF + LR"] = evaluate_multiclass(y_test, y_pred_lr, labels_present)
    results["TF-IDF + LR"].update(evaluate_binary_nihilism(y_test, y_pred_lr))
    artifacts["TF-IDF + LR"] = str(save_bundle(lr_pipe, "tfidf_lr"))
    predictions["TF-IDF + LR"] = y_pred_lr

    print("Training TF-IDF + Linear SVM…")
    svm_pipe = train_tfidf_svm(X_train, y_train)
    y_pred_svm = list(svm_pipe.predict(X_test))
    results["TF-IDF + SVM"] = evaluate_multiclass(y_test, y_pred_svm, labels_present)
    results["TF-IDF + SVM"].update(evaluate_binary_nihilism(y_test, y_pred_svm))
    artifacts["TF-IDF + SVM"] = str(save_bundle(svm_pipe, "tfidf_svm"))
    predictions["TF-IDF + SVM"] = y_pred_svm

    plot_nihilism_keywords(lr_pipe, FIGURES / "nihilism_top_keywords.png")

    emb_bundle = train_embedding_lr(X_train, y_train)
    if emb_bundle:
        print("Training ClimateBERT/MiniLM embeddings + Logistic Regression…")
        X_test_emb = emb_bundle["embedder"].encode(X_test, show_progress_bar=False)
        y_pred_emb = emb_bundle["classifier"].predict(X_test_emb)
        results["Embedding + LR"] = evaluate_multiclass(
            y_test, list(y_pred_emb), labels_present
        )
        results["Embedding + LR"].update(
            evaluate_binary_nihilism(y_test, list(y_pred_emb))
        )
        emb_path = MODELS / "embedding_lr.joblib"
        joblib.dump(emb_bundle, emb_path)
        artifacts["Embedding + LR"] = str(emb_path)
        predictions["Embedding + LR"] = list(y_pred_emb)

    print("Training binary Climate nihilism classifier (TF-IDF + LR)…")
    bin_pipe = train_binary_nihilism_tfidf(X_train, y_train)
    y_bin_pred = bin_pipe.predict(X_test)
    y_true_bin = [1 if y == NIHILISM_FOCUS else 0 for y in y_test]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_bin_pred, average="binary", zero_division=0
    )
    results["Binary nihilism (TF-IDF+LR)"] = {
        "accuracy": float(accuracy_score(y_true_bin, y_bin_pred)),
        "macro_f1": float(f1),
        "weighted_f1": float(f1),
        "nihilism_precision": float(p),
        "nihilism_recall": float(r),
        "nihilism_f1": float(f1),
        "binary_nihilism_precision": float(p),
        "binary_nihilism_recall": float(r),
        "binary_nihilism_f1": float(f1),
    }
    joblib.dump(
        {
            "type": "binary_nihilism",
            "vectorizer": bin_pipe.named_steps["tfidf"],
            "classifier": bin_pipe.named_steps["clf"],
        },
        MODELS / "binary_nihilism.joblib",
    )

    plot_model_comparison(results, FIGURES / "model_comparison.png")
    plot_nihilism_pr(results, FIGURES / "nihilism_pr_by_model.png")
    plot_confidence_distribution(FIGURES / "confidence_distribution.png")

    multiclass_names = [k for k in results if not k.startswith("Binary")]
    best_name = max(multiclass_names, key=lambda k: results[k]["macro_f1"])
    best_pred = predictions[best_name]
    plot_confusion(
        y_test,
        best_pred,
        labels_present,
        FIGURES / "confusion_matrix_best.png",
        f"Confusion matrix — {best_name} (test n={len(test_df)})",
    )

    metrics_path = OUTPUTS / "model_metrics.json"
    meta = {
        "dataset_rows": len(df),
        "nihilism_train_count": int((df["label"] == NIHILISM_FOCUS).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "models": results,
    }
    metrics_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = results[best_name]
    best_meta = {
        "model_name": best_name,
        "artifact": Path(artifacts[best_name]).name,
        "dataset_rows": len(df),
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "nihilism_f1": best["nihilism_f1"],
    }
    (MODELS / "best_model.json").write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

    print("\n=== Test set results ===")
    for name in multiclass_names + ["Binary nihilism (TF-IDF+LR)"]:
        m = results[name]
        print(
            f"{name}: acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
            f"weighted_f1={m['weighted_f1']:.3f} nihilism_f1={m['nihilism_f1']:.3f}"
        )
    print(f"\nBest multiclass model: {best_name}")
    print(f"Metrics -> {metrics_path}")
    print(f"Figures -> {FIGURES}")

    return meta, best_name, best


if __name__ == "__main__":
    main()
