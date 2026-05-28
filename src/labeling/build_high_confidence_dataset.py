#!/usr/bin/env python3
"""
Select high-quality weak labels for training augmentation.

Criteria (all must pass):
  - Auto-labeler did not flag needs_human_review
  - Reported confidence >= threshold
  - Keyword evidence for predicted class (keyword_score > 0)
  - Semantic score for predicted class >= floor
  - Recomputed label margin (semantic + keyword fusion) is large
  - Recomputed top label matches predicted_label (agreement)
  - Not already in manual / human-verified set

Output: data/weak_labels/high_confidence_training_dataset.csv

# TODO(Madeleine): improve weak-label quality filters
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from labeling.auto_label_comments import (
    SemanticMatcher,
    combine_scores,
    compile_keyword_patterns,
    keyword_scores,
    normalize_score_dict,
)
from common.label_utils import body_hash, load_human_verified, load_manual_labeled
from common.paths import AUTO_LABELED, HIGH_CONFIDENCE_TRAINING
from common.taxonomy import CANONICAL_LABELS, PRIORITY_LABELS, normalize_label

# Per-class caps — denial is over-predicted by the weak labeler; cap it hard
DEFAULT_CLASS_CAP = 120
PRIORITY_CLASS_CAP = 150
CLASS_CAP_OVERRIDES = {
    "Climate denial": 80,  # manual set already has 24; avoid 400:1 flood
}
MIN_CONFIDENCE = 0.55
MIN_SEMANTIC = 0.09
MIN_MARGIN = 0.10
WEIGHTS = (0.30, 0.70, 0.0, 0.0)  # keyword, semantic


def keywords_support_label(matched: str, label: str) -> bool:
    if not isinstance(matched, str) or not matched.strip():
        return False
    label_l = label.lower()
    for part in matched.split("|"):
        if part.lower().startswith(label_l + ":"):
            return True
    return False


def recompute_scores(
    text: str,
    compiled,
    sem_row: np.ndarray,
) -> Tuple[str, float, float, Dict[str, float]]:
    kw_raw, _ = keyword_scores(text, compiled)
    combined = combine_scores(kw_raw, sem_row, None, None, WEIGHTS)
    sorted_raw = sorted(combined.items(), key=lambda x: -x[1])
    top_label, top_raw = sorted_raw[0]
    second_raw = sorted_raw[1][1] if len(sorted_raw) > 1 else 0.0
    margin = (top_raw - second_raw) / (top_raw + 1e-9)
    prob = normalize_score_dict(combined, temperature=0.30)
    return top_label, float(margin), float(prob[top_label]), combined


def existing_hashes() -> Set[str]:
    manual = load_manual_labeled()
    verified = load_human_verified()
    hashes = set(manual["body_hash"])
    if not verified.empty:
        hashes |= set(verified["body_hash"])
    return hashes


def apply_per_class_cap(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label in CANONICAL_LABELS:
        sub = df[df["label"] == label].sort_values(
            ["selection_margin", "confidence"], ascending=False
        )
        if label in CLASS_CAP_OVERRIDES:
            cap = CLASS_CAP_OVERRIDES[label]
        elif label in PRIORITY_LABELS:
            cap = PRIORITY_CLASS_CAP
        else:
            cap = DEFAULT_CLASS_CAP
        if len(sub):
            parts.append(sub.head(cap))
    return pd.concat(parts, ignore_index=True)


def build_high_confidence(
    min_confidence: float = MIN_CONFIDENCE,
    min_semantic: float = MIN_SEMANTIC,
    min_margin: float = MIN_MARGIN,
    batch_size: int = 4000,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    print(f"Loading manual reference for semantic matcher…")
    manual = load_manual_labeled()
    manual = manual.rename(columns={"label": "canonical_label"})
    semantic = SemanticMatcher(manual)
    compiled = compile_keyword_patterns()
    exclude = existing_hashes()
    print(f"Excluding {len(exclude)} already-labeled comment hashes")

    usecols = [
        "body",
        "predicted_label",
        "confidence",
        "matched_keywords",
        "needs_human_review",
        "keyword_score",
        "semantic_score",
    ]
    print(f"Scanning {AUTO_LABELED}…")
    auto = pd.read_csv(AUTO_LABELED, usecols=usecols, encoding="utf-8", low_memory=False)
    auto["predicted_label"] = auto["predicted_label"].map(normalize_label)
    auto["body"] = auto["body"].fillna("").astype(str)
    auto["body_hash"] = auto["body"].map(body_hash)

    auto["kw_support"] = [
        keywords_support_label(m, l)
        for m, l in zip(auto["matched_keywords"], auto["predicted_label"])
    ]

    # Standard pool: strict keyword + confidence
    base_mask = (
        (auto["needs_human_review"].astype(str).str.lower().isin(["false", "0", "no"]))
        & (auto["confidence"] >= min_confidence)
        & (auto["keyword_score"] > 0)
        & (auto["semantic_score"] >= min_semantic)
        & (auto["predicted_label"].isin(CANONICAL_LABELS))
        & (auto["body"].str.len() > 10)
        & (~auto["body_hash"].isin(exclude))
        & auto["kw_support"]
    )
    # Priority pool: slightly relaxed for nihilism / anxiety / denial
    priority_mask = (
        (auto["needs_human_review"].astype(str).str.lower().isin(["false", "0", "no"]))
        & (auto["predicted_label"].isin(PRIORITY_LABELS))
        & (auto["confidence"] >= max(0.50, min_confidence - 0.05))
        & (auto["semantic_score"] >= min_semantic - 0.01)
        & (auto["body"].str.len() > 10)
        & (~auto["body_hash"].isin(exclude))
        & (auto["kw_support"] | (auto["confidence"] >= 0.65))
    )
    candidates = auto[base_mask | priority_mask].drop_duplicates(subset=["body_hash"]).copy()
    print(f"Stage-1 candidates (base + priority relaxed): {len(candidates)}")

    if max_candidates and len(candidates) > max_candidates:
        candidates = candidates.nlargest(max_candidates, "confidence")
        print(f"  capped to top {max_candidates} by confidence for re-scoring")

    # Stage 2: recompute semantic margin in batches
    rows: List[dict] = []
    bodies = candidates["body"].tolist()
    preds = candidates["predicted_label"].tolist()
    confs = candidates["confidence"].tolist()
    kw_scores = candidates["keyword_score"].tolist()
    sem_scores = candidates["semantic_score"].tolist()
    hashes = candidates["body_hash"].tolist()
    matched_kws = candidates["matched_keywords"].tolist()

    for start in range(0, len(bodies), batch_size):
        end = min(start + batch_size, len(bodies))
        batch_texts = bodies[start:end]
        sem_matrix, _ = semantic.score_batch(batch_texts)
        for i, text in enumerate(batch_texts):
            top_label, margin, top_prob, _ = recompute_scores(
                text, compiled, sem_matrix[i]
            )
            pred = preds[start + i]
            row_margin = min_margin - 0.03 if pred in PRIORITY_LABELS else min_margin
            if top_label != pred:
                continue
            if margin < row_margin:
                continue
            rows.append(
                {
                    "body_hash": hashes[start + i],
                    "body": text,
                    "label": pred,
                    "source": "high_confidence_weak",
                    "confidence": confs[start + i],
                    "keyword_score": kw_scores[start + i],
                    "semantic_score": sem_scores[start + i],
                    "selection_margin": margin,
                    "selection_prob": top_prob,
                    "matched_keywords": matched_kws[start + i],
                    "selection_reason": (
                        f"conf={confs[start + i]:.3f}, margin={margin:.3f}, "
                        f"kw+sem agree on {pred}"
                    ),
                }
            )
        if start % (batch_size * 5) == 0:
            print(f"  re-scored {end}/{len(bodies)}…")

    print(f"Stage-2 passed (margin + label agreement): {len(rows)}")
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = apply_per_class_cap(out)
    out = out.drop_duplicates(subset=["body_hash"], keep="first")
    out = out.sort_values(
        ["label", "selection_margin"], ascending=[True, False]
    ).reset_index(drop=True)

    HIGH_CONFIDENCE_TRAINING.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(HIGH_CONFIDENCE_TRAINING, index=False)
    print(f"Wrote {len(out)} rows -> {HIGH_CONFIDENCE_TRAINING}")
    print(out["label"].value_counts())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    p.add_argument("--min-semantic", type=float, default=MIN_SEMANTIC)
    p.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    p.add_argument("--max-candidates", type=int, default=None)
    args = p.parse_args()
    build_high_confidence(
        min_confidence=args.min_confidence,
        min_semantic=args.min_semantic,
        min_margin=args.min_margin,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
