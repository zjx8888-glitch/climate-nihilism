"""Shared helpers for paths, deduplication, and dataset I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from common.paths import (
    AUGMENTED_TRAINING,
    AUTO_LABELED,
    DATA_LABELED,
    FINAL_TRAINING,
    HIGH_CONFIDENCE_TRAINING,
    HUMAN_VERIFIED,
    MANUAL_LABELED,
    RECOVERED_LABELED,
    SPLITS_JSON,
    RANDOM_STATE,
    ensure_project_dirs,
)
from common.taxonomy import CANONICAL_LABELS, normalize_label

# Re-export paths used by other modules
OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "tfidf" / "models"  # TODO(Liu): TF-IDF models live under outputs/tfidf/


def body_hash(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def load_recovered_labeled() -> pd.DataFrame:
    """Primary supervised dataset (~1,845 recovered labels)."""
    path = RECOVERED_LABELED if RECOVERED_LABELED.exists() else MANUAL_LABELED
    if not path.exists():
        raise FileNotFoundError(
            f"No recovered labels at {RECOVERED_LABELED} or {MANUAL_LABELED}. "
            "Run: python -m labeling.recover_labeled_dataset"
        )
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    df["body"] = df["body"].fillna("").astype(str)
    label_col = "label" if "label" in df.columns else "label_canonical"
    df["label"] = df[label_col].map(normalize_label)
    df = df[df["label"].notna() & (df["body"].str.len() > 10)].copy()
    if "body_hash" in df.columns:
        df["body_hash"] = df["body_hash"].astype(str)
    else:
        df["body_hash"] = df["body"].map(body_hash)
    df["source"] = "recovered_manual"
    return df


def load_manual_labeled() -> pd.DataFrame:
    return load_recovered_labeled()


def load_human_verified() -> pd.DataFrame:
    if not HUMAN_VERIFIED.exists():
        return pd.DataFrame(
            columns=[
                "body_hash",
                "body",
                "predicted_label",
                "verified_label",
                "decision",
                "confidence",
                "reviewed_at",
            ]
        )
    df = pd.read_csv(HUMAN_VERIFIED, encoding="utf-8")
    if df.empty:
        return df
    df["verified_label"] = df["verified_label"].map(normalize_label)
    df = df[df["decision"].isin(["accept", "change"])].copy()
    df["body_hash"] = df["body_hash"].fillna(df["body"].map(body_hash))
    df["source"] = "human_verified"
    return df


def load_auto_labeled(
    predicted_label: Optional[str] = None,
    max_confidence: Optional[float] = None,
    needs_review_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(AUTO_LABELED, encoding="utf-8", on_bad_lines="skip")
    df["body"] = df["body"].fillna("").astype(str)
    df["predicted_label"] = df["predicted_label"].map(normalize_label)
    if predicted_label:
        df = df[df["predicted_label"] == predicted_label]
    if max_confidence is not None:
        df = df[df["confidence"] <= max_confidence]
    if needs_review_only and "needs_human_review" in df.columns:
        df = df[df["needs_human_review"].astype(str).str.lower().isin(["true", "1", "yes"])]
    df["body_hash"] = df["body"].map(body_hash)
    return df


def load_high_confidence_weak() -> pd.DataFrame:
    if not HIGH_CONFIDENCE_TRAINING.exists():
        return pd.DataFrame(columns=["body_hash", "body", "label", "source"])
    df = pd.read_csv(HIGH_CONFIDENCE_TRAINING, encoding="utf-8")
    if df.empty:
        return df
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(CANONICAL_LABELS)].copy()
    df["body_hash"] = df["body_hash"].fillna(df["body"].map(body_hash))
    df["source"] = "high_confidence_weak"
    return df


def _merge_sources(include_high_confidence: bool = False) -> pd.DataFrame:
    recovered = load_recovered_labeled()
    verified = load_human_verified()
    high_conf = load_high_confidence_weak() if include_high_confidence else pd.DataFrame()

    frames = []
    for part, label_col in (
        (recovered, "label"),
        (verified, "verified_label"),
        (high_conf, "label"),
    ):
        if part.empty:
            continue
        col = label_col if label_col in part.columns else "label"
        frames.append(
            part[["body_hash", "body", col, "source"]].rename(columns={col: "label"})
        )

    if not frames:
        raise ValueError("No labeled data available.")

    order = {"recovered_manual": 0, "manual": 0, "human_verified": 1, "high_confidence_weak": 2}
    merged = pd.concat(frames, ignore_index=True)
    merged["_prio"] = merged["source"].map(lambda s: order.get(s, 9))
    merged = merged.sort_values("_prio").drop(columns=["_prio"])
    merged["label"] = merged["label"].map(normalize_label)
    merged = merged[merged["label"].isin(CANONICAL_LABELS)]
    merged = merged.drop_duplicates(subset=["body_hash"], keep="first")
    merged = merged[merged["body"].str.len() > 10].reset_index(drop=True)
    return merged


def build_final_training_dataset() -> pd.DataFrame:
    merged = _merge_sources(include_high_confidence=False)
    ensure_project_dirs()
    DATA_LABELED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FINAL_TRAINING, index=False)
    return merged


def build_augmented_training_dataset() -> pd.DataFrame:
    merged = _merge_sources(include_high_confidence=True)
    ensure_project_dirs()
    merged.to_csv(AUGMENTED_TRAINING, index=False)
    return merged


def stratified_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    counts = work["label"].value_counts()
    rare = counts[counts < 2].index.tolist()
    for lab in rare:
        work.loc[work["label"] == lab, "label"] = "climate opinion critique"

    train_val, test = train_test_split(
        work,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=work["label"],
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )

    split_meta = {
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "val_size": val_size,
        "n_total": len(work),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_hashes": train["body_hash"].tolist(),
        "val_hashes": val["body_hash"].tolist(),
        "test_hashes": test["body_hash"].tolist(),
    }
    DATA_LABELED.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "total_rows": len(df),
        "by_source": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "by_label": df["label"].value_counts().to_dict(),
    }
