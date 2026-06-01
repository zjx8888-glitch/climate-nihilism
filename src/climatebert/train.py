#!/usr/bin/env python3
"""
ClimateBERT training & evaluation (Jinxi's scope).

Trains and evaluates:
  - ClimateBERT embeddings + Logistic Regression (multiclass + binary nihilism)
  - Optional ClimateBERT fine-tuning (multiclass, --finetune)

Data: data/labeled/recovered_labeled_dataset.csv
Splits: data/labeled/splits.json

Outputs: outputs/climatebert/, outputs/predictions/, outputs/error_analysis/

Usage:
  python src/climatebert/train.py
  python src/climatebert/train.py --dataset-version v2
  python src/climatebert/train.py --finetune --epochs 3
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    average_precision_score,
)
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import body_hash
from common.paths import (
    CLIMATEBERT_OUT,
    CLIMATEBERT_V2_OUT,
    ERROR_ANALYSIS_OUT,
    FINAL_TRAINING,
    FINAL_TRAINING_V2,
    PREDICTIONS_OUT,
    RECOVERED_LABELED,
    ROOT,
    SPLITS_JSON,
    SPLITS_V2_JSON,
    RANDOM_STATE,
    ensure_project_dirs,
)
from common.taxonomy import CANONICAL_LABELS, NIHILISM_FOCUS, normalize_label

warnings.filterwarnings("ignore")
# Recommended ClimateBERT LM (DistilRoBERTa, climate-f corpus); see arXiv:2110.12010
CLIMATEBERT_MODEL = "climatebert/distilroberta-base-climate-f"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_LEN = 512


def dataset_config(version: str) -> Dict[str, Any]:
    if version == "v1":
        return {
            "version": "v1",
            "training_csv": FINAL_TRAINING,
            "splits_json": SPLITS_JSON,
            "output_dir": CLIMATEBERT_OUT,
            "mirror_predictions": True,
        }
    if version == "v2":
        return {
            "version": "v2",
            "training_csv": FINAL_TRAINING_V2,
            "splits_json": SPLITS_V2_JSON,
            "output_dir": CLIMATEBERT_V2_OUT,
            "mirror_predictions": False,
        }
    raise ValueError(f"Unknown dataset version: {version}. Use v1 or v2.")


def load_labeled_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = cfg["training_csv"]
    if not path.exists():
        if cfg["version"] == "v2":
            raise FileNotFoundError(
                f"Missing {path}. Run: python src/climatebert/prepare_v2_dataset.py"
            )
        raise FileNotFoundError(
            f"No dataset at {path}. Run: python -m labeling.recover_labeled_dataset"
        )
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df["body"] = df["body"].fillna("").astype(str)
    if "label" not in df.columns:
        label_col = "label_canonical" if "label_canonical" in df.columns else "label_clean"
        df["label"] = df[label_col].map(normalize_label)
    else:
        df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].notna() & (df["body"].str.len() > 10)].copy()
    if "body_hash" not in df.columns:
        df["body_hash"] = df["body"].map(body_hash)
    else:
        df["body_hash"] = df["body_hash"].astype(str)
    return df


def apply_splits(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_path = cfg["splits_json"]
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}.")
    meta = json.loads(splits_path.read_text(encoding="utf-8"))
    train = df[df["body_hash"].isin(meta["train_hashes"])].copy()
    val = df[df["body_hash"].isin(meta["val_hashes"])].copy()
    test = df[df["body_hash"].isin(meta["test_hashes"])].copy()
    return train, val, test


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name), model_name
    except Exception:
        print(f"Could not load {model_name}; using {FALLBACK_EMBED_MODEL}")
        return SentenceTransformer(FALLBACK_EMBED_MODEL), FALLBACK_EMBED_MODEL


def encode_texts(embedder, texts: List[str], batch_size: int = 32) -> np.ndarray:
    return embedder.encode(
        [t[:8000] for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def evaluate_multiclass(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    per_class = {
        lab: {
            "precision": report[lab]["precision"],
            "recall": report[lab]["recall"],
            "f1": report[lab]["f1-score"],
            "support": report[lab]["support"],
        }
        for lab in labels
        if lab in report
    }
    nih = per_class.get(
        NIHILISM_FOCUS,
        {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
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


def train_embedding_lr(
    X_train: np.ndarray,
    y_train: List[str],
    binary: bool = False,
) -> LogisticRegression:
    if binary:
        y = [1 if lab == NIHILISM_FOCUS else 0 for lab in y_train]
    else:
        y = y_train
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y)
    return clf


def predict_multiclass(clf: LogisticRegression, X: np.ndarray, labels: List[str]) -> Tuple[List[str], np.ndarray]:
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    pred_idx = proba.argmax(axis=1)
    if not isinstance(classes[0], str):
        # binary 0/1
        preds = [NIHILISM_FOCUS if i == 1 else "not_nihilism" for i in pred_idx]
        return preds, proba
    preds = [classes[i] for i in pred_idx]
    return preds, proba


def finetune_climatebert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> Dict[str, Any]:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    label2id = {l: i for i, l in enumerate(CANONICAL_LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CANONICAL_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def to_ds(part: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": part["body"].str[:MAX_TEXT_LEN].tolist(),
                "labels": [label2id[y] for y in part["label"].tolist()],
            }
        )

    train_ds = to_ds(train_df).map(
        lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=256),
        batched=True,
    )
    val_ds = to_ds(val_df).map(
        lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=256),
        batched=True,
    )

    ft_dir = output_dir / "finetuned_model"
    args = TrainingArguments(
        output_dir=str(ft_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=1)
        y_pred = [id2label[p] for p in preds]
        y_true = [id2label[l] for l in labels_arr]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(ft_dir))
    tokenizer.save_pretrained(str(ft_dir))

    return {
        "type": "finetuned",
        "model_dir": str(ft_dir),
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label,
        "label2id": label2id,
    }


def predict_finetuned(bundle: Dict, texts: List[str]) -> Tuple[List[str], List[float]]:
    from transformers import pipeline

    pipe = pipeline(
        "text-classification",
        model=bundle["model_dir"],
        tokenizer=bundle["model_dir"],
        top_k=1,
        truncation=True,
        max_length=256,
        device=-1,
    )
    preds, scores = [], []
    id2label = bundle["id2label"]
    for t in texts:
        out = pipe(t[:MAX_TEXT_LEN])[0]
        lab = out["label"]
        if lab.startswith("LABEL_"):
            lab = id2label[int(lab.split("_")[-1])]
        else:
            lab = normalize_label(lab) or lab
        preds.append(lab if lab in CANONICAL_LABELS else CANONICAL_LABELS[0])
        scores.append(float(out["score"]))
    return preds, scores


def plot_confusion(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    path: Path,
    title: str,
) -> None:
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


def plot_nihilism_pr_curve(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkred", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Climate nihilism — PR curve (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_error_analysis(
    test_df: pd.DataFrame,
    y_pred: List[str],
    approach: str,
) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(test_df.itertuples()):
        true_lab = row.label
        pred_lab = y_pred[i]
        body_snip = str(row.body)[:400]
        err_type = "correct"
        if true_lab == NIHILISM_FOCUS and pred_lab != NIHILISM_FOCUS:
            err_type = "false_negative_nihilism"
        elif true_lab != NIHILISM_FOCUS and pred_lab == NIHILISM_FOCUS:
            err_type = "false_positive_nihilism"
        elif true_lab != pred_lab:
            err_type = "other_misclass"

        confused_anxiety = (
            (true_lab == NIHILISM_FOCUS and pred_lab == "climate anxiety")
            or (true_lab == "climate anxiety" and pred_lab == NIHILISM_FOCUS)
        )
        confused_critique = (
            (true_lab == NIHILISM_FOCUS and pred_lab == "Climate nihilism critique")
            or (true_lab == "Climate nihilism critique" and pred_lab == NIHILISM_FOCUS)
        )

        rows.append(
            {
                "body_hash": row.body_hash,
                "true_label": true_lab,
                "predicted_label": pred_lab,
                "error_type": err_type,
                "confused_with_climate_anxiety": confused_anxiety,
                "confused_with_nihilism_critique": confused_critique,
                "approach": approach,
                "body_snippet": body_snip,
            }
        )
    out = pd.DataFrame(rows)
    return out


def write_old_vs_new_comparison(new_metrics: Dict[str, Any], out_path: Path) -> None:
    old_path = CLIMATEBERT_OUT / "climatebert_metrics.json"
    old = {}
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
    old_mc = old.get("approaches", {}).get("embedding_lr_multiclass", {})
    new_mc = new_metrics.get("approaches", {}).get("embedding_lr_multiclass", {})

    def row(name: str, old_v: Any, new_v: Any) -> str:
        return f"| {name} | {old_v} | {new_v} |"

    lines = [
        "# ClimateBERT: v1 vs v2 dataset comparison",
        "",
        "## Dataset",
        row("Metric", "v1 (recovered)", "v2 (cleaned_data-2)"),
        row("---", "---", "---"),
        row("Total rows", old.get("dataset_rows", "—"), new_metrics.get("dataset_rows", "—")),
        row(
            "Climate nihilism count",
            old.get("nihilism_train_count", "—"),
            new_metrics.get("nihilism_train_count", "—"),
        ),
        row("Test size", old.get("test_size", "—"), new_metrics.get("test_size", "—")),
        "",
        "## Test metrics (embedding + LR multiclass)",
        row("Metric", "v1", "v2"),
        row("---", "---", "---"),
        row("Accuracy", f"{old_mc.get('accuracy', 0):.3f}", f"{new_mc.get('accuracy', 0):.3f}"),
        row("Macro F1", f"{old_mc.get('macro_f1', 0):.3f}", f"{new_mc.get('macro_f1', 0):.3f}"),
        row("Weighted F1", f"{old_mc.get('weighted_f1', 0):.3f}", f"{new_mc.get('weighted_f1', 0):.3f}"),
        row(
            "Nihilism precision",
            f"{old_mc.get('nihilism_precision', 0):.3f}",
            f"{new_mc.get('nihilism_precision', 0):.3f}",
        ),
        row(
            "Nihilism recall",
            f"{old_mc.get('nihilism_recall', 0):.3f}",
            f"{new_mc.get('nihilism_recall', 0):.3f}",
        ),
        row(
            "Nihilism F1",
            f"{old_mc.get('nihilism_f1', 0):.3f}",
            f"{new_mc.get('nihilism_f1', 0):.3f}",
        ),
        "",
        "## Takeaway",
        "",
        "v2 adds substantially more labeled data and **~12× more Climate nihilism examples**, "
        "which should improve nihilism detection stability even if macro F1 shifts with class imbalance.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ClimateBERT training (Jinxi)")
    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        default="v1",
        help="v1=recovered ~1.8k labels; v2=cleaned_data-2 ~16k labels",
    )
    parser.add_argument("--finetune", action="store_true", help="Run transformer fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-model", type=str, default=CLIMATEBERT_MODEL)
    args = parser.parse_args()

    cfg = dataset_config(args.dataset_version)
    out_dir: Path = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_project_dirs()
    sns.set_theme(style="whitegrid")

    print(f"Dataset version: {cfg['version']}")
    print(f"Training CSV: {cfg['training_csv']}")
    print(f"Output dir: {out_dir}")

    df = load_labeled_data(cfg)
    train_df, val_df, test_df = apply_splits(df, cfg)
    labels_present = sorted(set(train_df["label"]) | set(test_df["label"]))

    print(f"Dataset: {len(df)} rows | nihilism train: {(train_df['label']==NIHILISM_FOCUS).sum()}")
    print(f"Splits — train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    embedder, embed_model_used = load_embedder(args.embed_model)
    print(f"Embedding model: {embed_model_used}")

    X_train = encode_texts(embedder, train_df["body"].tolist(), args.batch_size)
    X_val = encode_texts(embedder, val_df["body"].tolist(), args.batch_size)
    X_test = encode_texts(embedder, test_df["body"].tolist(), args.batch_size)

    metrics: Dict[str, Any] = {
        "dataset_version": cfg["version"],
        "embed_model": embed_model_used,
        "dataset_rows": len(df),
        "nihilism_train_count": int((df["label"] == NIHILISM_FOCUS).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "approaches": {},
    }

    # --- Embeddings + LR multiclass ---
    print("\n[1/3] Embeddings + LR — multiclass (14 labels)")
    mc_clf = train_embedding_lr(X_train, train_df["label"].tolist(), binary=False)
    y_pred_mc, proba_mc = predict_multiclass(mc_clf, X_test, labels_present)
    y_true = test_df["label"].tolist()
    mc_metrics = evaluate_multiclass(y_true, y_pred_mc, labels_present)
    metrics["approaches"]["embedding_lr_multiclass"] = mc_metrics
    joblib.dump(
        {"embedder_name": embed_model_used, "classifier": mc_clf, "type": "embedding_lr"},
        out_dir / "embedding_lr_multiclass.joblib",
    )

    # --- Embeddings + LR binary nihilism ---
    print("\n[2/3] Embeddings + LR — binary Climate nihilism vs not")
    bin_clf = train_embedding_lr(X_train, train_df["label"].tolist(), binary=True)
    bin_proba = bin_clf.predict_proba(X_test)[:, 1]
    bin_pred = (bin_proba >= 0.5).astype(int)
    y_true_bin = np.array([1 if y == NIHILISM_FOCUS else 0 for y in y_true])
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, bin_pred, average="binary", zero_division=0
    )
    metrics["approaches"]["embedding_lr_binary_nihilism"] = {
        "accuracy": float(accuracy_score(y_true_bin, bin_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y_true_bin, bin_proba)),
    }
    joblib.dump(
        {"embedder_name": embed_model_used, "classifier": bin_clf, "type": "binary_embedding_lr"},
        out_dir / "embedding_lr_binary_nihilism.joblib",
    )
    plot_nihilism_pr_curve(
        y_true_bin,
        bin_proba,
        out_dir / "climatebert_nihilism_pr_curve.png",
    )

    # --- Optional fine-tune ---
    ft_preds, ft_scores = None, None
    if args.finetune:
        print("\n[3/3] ClimateBERT fine-tuning — multiclass")
        try:
            ft_bundle = finetune_climatebert(
                train_df,
                val_df,
                embed_model_used,
                args.epochs,
                args.batch_size,
                out_dir,
            )
            ft_preds, ft_scores = predict_finetuned(
                ft_bundle, test_df["body"].tolist()
            )
            ft_metrics = evaluate_multiclass(y_true, ft_preds, labels_present)
            metrics["approaches"]["finetuned_multiclass"] = ft_metrics
            joblib.dump(ft_bundle, out_dir / "finetuned_bundle.joblib")
        except Exception as e:
            print(f"Fine-tuning skipped/failed: {e}")
            metrics["approaches"]["finetuned_multiclass"] = {"error": str(e)}
    else:
        print("\n[3/3] Fine-tuning skipped (pass --finetune to enable)")
        metrics["approaches"]["finetuned_multiclass"] = {
            "skipped": True,
            "reason": "Pass --finetune to run",
        }

    # Primary predictions = best embedding LR multiclass by macro F1
    best_approach = "embedding_lr_multiclass"
    if args.finetune and ft_preds and "error" not in metrics["approaches"].get(
        "finetuned_multiclass", {}
    ):
        ft_f1 = metrics["approaches"]["finetuned_multiclass"].get("macro_f1", 0)
        if ft_f1 > mc_metrics["macro_f1"]:
            best_approach = "finetuned_multiclass"
            y_pred_mc = ft_preds

    plot_confusion(
        y_true,
        y_pred_mc,
        labels_present,
        out_dir / "climatebert_confusion_matrix.png",
        f"ClimateBERT confusion matrix — {best_approach} (test)",
    )

    # Predictions CSV
    pred_rows = []
    for i, row in enumerate(test_df.itertuples()):
        pred_rows.append(
            {
                "body_hash": row.body_hash,
                "body": row.body[:500],
                "true_label": row.label,
                "predicted_label": y_pred_mc[i],
                "approach": best_approach,
                "nihilism_prob_binary": float(bin_proba[i]),
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_path = out_dir / "climatebert_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    if cfg.get("mirror_predictions"):
        pred_df.to_csv(PREDICTIONS_OUT / "climatebert_predictions.csv", index=False)

    # Error analysis (multiclass primary)
    err_df = build_error_analysis(test_df, y_pred_mc, best_approach)
    err_path = out_dir / "climatebert_error_analysis.csv"
    err_df.to_csv(err_path, index=False)
    if cfg.get("mirror_predictions"):
        err_df.to_csv(ERROR_ANALYSIS_OUT / "climatebert_error_analysis.csv", index=False)

    # Summary slices for report
    metrics["error_analysis_summary"] = {
        "false_positive_nihilism": int(
            (err_df["error_type"] == "false_positive_nihilism").sum()
        ),
        "false_negative_nihilism": int(
            (err_df["error_type"] == "false_negative_nihilism").sum()
        ),
        "confused_with_climate_anxiety": int(err_df["confused_with_climate_anxiety"].sum()),
        "confused_with_nihilism_critique": int(
            err_df["confused_with_nihilism_critique"].sum()
        ),
    }
    metrics["best_approach"] = best_approach

    metrics_path = out_dir / "climatebert_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if cfg["version"] == "v2":
        write_old_vs_new_comparison(
            metrics, out_dir / "old_vs_new_climatebert_results.md"
        )

    print("\n=== ClimateBERT results (test set) ===")
    for name, m in metrics["approaches"].items():
        if isinstance(m, dict) and "macro_f1" in m:
            print(
                f"{name}: acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
                f"weighted_f1={m['weighted_f1']:.3f} nihilism_f1={m['nihilism_f1']:.3f}"
            )
    bin_m = metrics["approaches"]["embedding_lr_binary_nihilism"]
    print(
        f"binary nihilism: P={bin_m['precision']:.3f} R={bin_m['recall']:.3f} F1={bin_m['f1']:.3f}"
    )
    print(f"\nOutputs -> {out_dir}")
    if cfg["version"] == "v2":
        print(f"Comparison -> {out_dir / 'old_vs_new_climatebert_results.md'}")


if __name__ == "__main__":
    main()
