#!/usr/bin/env python3
"""
Weak-label / pre-label Reddit comments using taxonomy definitions,
keyword heuristics, and semantic similarity to existing labeled examples.

Usage:
  python src/auto_label_comments.py
  python src/auto_label_comments.py --max-rows 5000
  python src/auto_label_comments.py --use-zero-shot --zero-shot-cap 2000
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths — see src/common/paths.py
# TODO(Madeleine): improve preprocessing pipeline integration
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import AUTO_LABELED, MANUAL_LABELED, reddit_unlabeled_path

DEFAULT_LABELED = MANUAL_LABELED
DEFAULT_UNLABELED = reddit_unlabeled_path()
DEFAULT_OUTPUT = AUTO_LABELED

PRIORITY_LABELS = frozenset(
    {"Climate nihilism", "climate anxiety", "Climate denial"}
)

# Official taxonomy from data labels.pdf
TAXONOMY: Dict[str, str] = {
    "Not climate opinion": (
        "Posts that do not relate to climate change."
    ),
    "climate activism": (
        "Posts that urge people to help with climate change, ask how individuals "
        "can help, or promote climate-related projects."
    ),
    "climate anxiety": (
        "Fear of climate change without necessarily being hopeless about stopping it."
    ),
    "climate change importance": (
        "Expresses that climate change is important; may not urge specific action."
    ),
    "Climate apathy": (
        "The author does not care about climate change."
    ),
    "Climate information": (
        "Informational post about climate change: news, disasters, facts without "
        "strong personal opinion."
    ),
    "Climate nihilism": (
        "Belief that climate change is inevitable and irreversible; hopeless about "
        "stopping it (stronger than anxiety)."
    ),
    "Climate denial": (
        "Opinion that climate change does not exist or is not real."
    ),
    "Climate optimism": (
        "Climate is improving or hope that leaders/people can improve the climate."
    ),
    "climate policy critique": (
        "Critique of political policy on climate: laws, politicians, government action."
    ),
    "Climate action critique": (
        "Critique of climate action broadly: companies, greenwashing, AI data centers, etc."
    ),
    "Climate denial critique": (
        "Critique of people who deny climate change exists."
    ),
    "Climate nihilism critique": (
        "Critique of nihilistic or doomist views on climate."
    ),
    "climate opinion critique": (
        "Critique of other climate opinions not covered above."
    ),
}

CANONICAL_LABELS: List[str] = list(TAXONOMY.keys())

# Map messy annotator strings -> canonical
LABEL_ALIASES: Dict[str, str] = {
    "climate information": "Climate information",
    "climate policy critique": "climate policy critique",
    "Climate policy critique": "climate policy critique",
    "climate action critique": "Climate action critique",
    "Climate action critique": "Climate action critique",
    "climate anxiety": "climate anxiety",
    "Climate anxiety": "climate anxiety",
    "climate optimism": "Climate optimism",
    "climate activism critique": "climate opinion critique",
    "climate opinion": "climate opinion critique",
    "Climate opinion": "climate opinion critique",
}

# Critique labels checked before base labels to reduce false positives
CRITIQUE_LABELS = [
    "Climate denial critique",
    "Climate nihilism critique",
    "climate opinion critique",
    "climate policy critique",
    "Climate action critique",
]

BASE_LABELS = [l for l in CANONICAL_LABELS if l not in CRITIQUE_LABELS]

# Phrase heuristics derived from definitions (not single-word only)
KEYWORD_RULES: Dict[str, List[Tuple[str, float]]] = {
    "Climate denial critique": [
        (r"\bclimate\s+denier", 2.0),
        (r"\bdeniers?\b", 1.2),
        (r"\banti[- ]?science\b", 1.5),
        (r"\bdenialism\b", 1.8),
        (r"\bdo not believe in climate", 2.0),
        (r"\brefuse to accept climate", 1.8),
    ],
    "Climate nihilism critique": [
        (r"\bclimate\s+doom", 1.8),
        (r"\bdoomer", 1.8),
        (r"\bdefeatist", 1.6),
        (r"\bstop being hopeless", 1.5),
        (r"\bnot too late", 1.2),
        (r"\bstill time to act", 1.4),
    ],
    "climate policy critique": [
        (r"\bgovernment\b", 0.6),
        (r"\bpolicy\b", 1.2),
        (r"\bpolitician", 1.2),
        (r"\blegislation\b", 1.2),
        (r"\bcarbon tax\b", 1.4),
        (r"\bgreen new deal\b", 1.3),
        (r"\bparis agreement\b", 1.2),
        (r"\bcongress\b", 0.9),
        (r"\btrump\b", 0.5),
        (r"\bbiden\b", 0.5),
    ],
    "Climate action critique": [
        (r"\bgreenwash", 2.0),
        (r"\bcorporate\b", 0.8),
        (r"\bdata center", 1.5),
        (r"\bnet zero\b", 0.9),
        (r"\bvirtue signal", 1.6),
        (r"\bperformative\b", 1.2),
        (r"\belectric vehicle", 0.7),
        (r"\brecycling\b", 0.6),
    ],
    "climate opinion critique": [
        (r"\bstop saying\b", 1.0),
        (r"\byou are wrong\b", 1.0),
        (r"\bstrawman\b", 1.2),
        (r"\bthat take\b", 0.8),
    ],
    "Climate denial": [
        (r"\bclimate change (is a )?hoax\b", 2.5),
        (r"\bglobal warming (is a )?hoax\b", 2.5),
        (r"\bclimate change (isn't|is not|not) real\b", 2.2),
        (r"\bno evidence (for|of) climate\b", 2.0),
        (r"\bnatural cycle\b", 1.2),
        (r"\bco2 is good\b", 1.5),
        (r"\bclimate scam\b", 2.2),
        (r"\bfake science\b", 1.5),
    ],
    "Climate nihilism": [
        (r"\btoo late\b", 1.8),
        (r"\bpointless to\b", 1.6),
        (r"\bno point in\b", 1.5),
        (r"\bwe are (already )?doomed\b", 2.2),
        (r"\bwe're doomed\b", 2.2),
        (r"\bcan't be stopped\b", 1.8),
        (r"\bcannot be stopped\b", 1.8),
        (r"\bno hope\b", 1.6),
        (r"\bhopeless\b", 1.4),
        (r"\binevitable\b", 1.0),
        (r"\birreversible\b", 1.2),
        (r"\bend of the world\b", 1.3),
        (r"\bhumanity (is|are) doomed\b", 2.0),
        (r"\bnothing (we|anyone) can do\b", 1.8),
    ],
    "climate anxiety": [
        (r"\bclimate anxiety\b", 2.5),
        (r"\beco[- ]?anxiety\b", 2.5),
        (r"\bworried about (the )?climate\b", 2.2),
        (r"\bscared (about|of) climate\b", 2.0),
        (r"\bterrified (about|of) climate\b", 2.0),
        (r"\banxious about climate\b", 2.2),
        (r"\bfear (for|of) (our|the) (climate|planet|future)\b", 1.6),
        (r"\bexistential (dread|fear).{0,40}climate\b", 1.8),
    ],
    "climate activism": [
        (r"\bwe (must|need to) act\b", 1.4),
        (r"\btake action\b", 1.2),
        (r"\bjoin (the )?fight\b", 1.2),
        (r"\bprotest\b", 1.0),
        (r"\bpetition\b", 1.0),
        (r"\bvolunteer\b", 0.9),
        (r"\breduce (your )?carbon\b", 1.3),
        (r"\bplant trees\b", 1.0),
        (r"\bcall your representative\b", 1.5),
    ],
    "climate change importance": [
        (r"\bmost important (issue|problem)\b", 1.5),
        (r"\bexistential threat\b", 1.3),
        (r"\bserious (issue|problem|threat)\b", 1.0),
        (r"\bclimate change matters\b", 2.0),
        (r"\bcannot ignore climate\b", 1.5),
    ],
    "Climate apathy": [
        (r"\bdon't care about climate\b", 2.5),
        (r"\bdo not care about climate\b", 2.5),
        (r"\bnot interested in climate\b", 2.0),
        (r"\bdoesn't matter to me\b", 1.2),
        (r"\bwho cares about climate\b", 2.0),
    ],
    "Climate information": [
        (r"\baccording to (the )?study\b", 1.2),
        (r"\breport (says|found)\b", 1.0),
        (r"\bscientists (say|found)\b", 1.0),
        (r"\bipcc\b", 1.3),
        (r"\bdegrees celsius\b", 1.0),
        (r"\bsea level\b", 0.8),
        (r"\bwildfire\b", 0.7),
        (r"\bhurricane\b", 0.7),
    ],
    "Climate optimism": [
        (r"\breason for hope\b", 1.6),
        (r"\bwe can still fix\b", 1.5),
        (r"\boptimistic about climate\b", 2.2),
        (r"\bclimate (is|getting) better\b", 1.8),
        (r"\bsolutions (are|exist)\b", 1.0),
        (r"\brenewable(s)? (will|can)\b", 0.9),
    ],
    "Not climate opinion": [
        (r"\boff topic\b", 1.5),
    ],
}

CLIMATE_RELEVANCE = re.compile(
    r"\b(climate|global warming|greenhouse|co2|carbon|emissions?|"
    r"fossil fuel|renewable|ipcc|net zero|temperature|warming)\b",
    re.I,
)


@dataclass
class Prediction:
    label: str
    confidence: float
    keyword_score: float
    semantic_score: float
    zero_shot_score: float
    matched_keywords: List[str]
    similar_example: str
    reason: str
    needs_human_review: bool


def normalize_label(raw: str) -> Optional[str]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().strip('"').strip()
    if s in TAXONOMY:
        return s
    return LABEL_ALIASES.get(s)


def preprocess_text(text: str, max_len: int = 8000) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if len(t) > max_len:
        t = t[:max_len]
    return t


def load_labeled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    if "body" not in df.columns:
        raise ValueError(f"Labeled file missing 'body' column: {path}")
    df["body"] = df["body"].fillna("").astype(str)
    df["canonical_label"] = df.get("label", pd.Series(dtype=str)).map(
        lambda x: normalize_label(x) if pd.notna(x) else None
    )
    labeled = df[df["canonical_label"].notna()].copy()
    labeled = labeled[labeled["body"].str.len() > 10]
    return labeled


def load_unlabeled(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", nrows=max_rows)
    if "body" not in df.columns:
        raise ValueError(f"Unlabeled file missing 'body' column: {path}")
    df["body"] = df["body"].fillna("").astype(str)
    df = df[df["body"].str.len() > 10].copy()
    return df


def compile_keyword_patterns() -> Dict[str, List[Tuple[re.Pattern, float]]]:
    compiled: Dict[str, List[Tuple[re.Pattern, float]]] = {}
    for label, rules in KEYWORD_RULES.items():
        compiled[label] = [
            (re.compile(pat, re.I), weight) for pat, weight in rules
        ]
    return compiled


def keyword_scores(
    text: str,
    compiled: Dict[str, List[Tuple[re.Pattern, float]]],
) -> Tuple[Dict[str, float], List[str]]:
    scores: Dict[str, float] = {lbl: 0.0 for lbl in CANONICAL_LABELS}
    matched: List[str] = []
    has_climate = bool(CLIMATE_RELEVANCE.search(text))

    # Order: critiques first, then priority bases, then rest
    order = CRITIQUE_LABELS + list(PRIORITY_LABELS) + [
        l for l in BASE_LABELS if l not in PRIORITY_LABELS
    ]

    for label in order:
        for pattern, weight in compiled.get(label, []):
            if pattern.search(text):
                scores[label] += weight
                matched.append(f"{label}:{pattern.pattern}")

    if not has_climate:
        # Weak climate signal -> lean Not climate opinion unless strong keyword hit
        non_trivial = max(scores.values())
        if non_trivial < 1.5:
            scores["Not climate opinion"] = max(scores["Not climate opinion"], 2.0)
        else:
            scores["Not climate opinion"] *= 0.3
        # Suppress climate-opinion labels without climate vocabulary
        for lbl in CANONICAL_LABELS:
            if lbl != "Not climate opinion":
                scores[lbl] *= 0.15
    else:
        scores["Not climate opinion"] *= 0.2

    # Disambiguate nihilism vs anxiety: hopelessness boosts nihilism
    if scores["Climate nihilism"] > 0 and scores["climate anxiety"] > 0:
        if scores["Climate nihilism"] >= scores["climate anxiety"]:
            scores["climate anxiety"] *= 0.65
        else:
            scores["Climate nihilism"] *= 0.75

    return scores, matched


def normalize_score_dict(scores: Dict[str, float], temperature: float = 0.35) -> Dict[str, float]:
    arr = np.array([scores[l] for l in CANONICAL_LABELS], dtype=float)
    if arr.max() <= 0:
        return {l: 1.0 / len(CANONICAL_LABELS) for l in CANONICAL_LABELS}
    exp = np.exp((arr - arr.max()) / max(temperature, 1e-6))
    probs = exp / exp.sum()
    return {l: float(probs[i]) for i, l in enumerate(CANONICAL_LABELS)}


class SemanticMatcher:
    """TF-IDF similarity to labeled examples, aggregated per class."""

    def __init__(self, labeled_df: pd.DataFrame):
        self.examples = labeled_df.reset_index(drop=True)
        self.labels = self.examples["canonical_label"].tolist()
        self.bodies = [preprocess_text(b) for b in self.examples["body"].tolist()]

        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.labeled_matrix = self.vectorizer.fit_transform(self.bodies)
        self._class_indices: Dict[str, List[int]] = {}
        for i, lab in enumerate(self.labels):
            self._class_indices.setdefault(lab, []).append(i)

    def score_batch(self, texts: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        """Returns (n_samples, n_labels) semantic probability matrix + similar snippets."""
        matrix = self.vectorizer.transform([preprocess_text(t) for t in texts])
        sims = cosine_similarity(matrix, self.labeled_matrix)  # (n, n_labeled)

        n = len(texts)
        n_labels = len(CANONICAL_LABELS)
        label_to_idx = {l: i for i, l in enumerate(CANONICAL_LABELS)}
        class_scores = np.zeros((n, n_labels), dtype=float)
        similar: List[str] = []

        for row in range(n):
            best_j = int(np.argmax(sims[row]))
            best_body = self.bodies[best_j]
            similar.append(
                best_body[:240] + ("…" if len(best_body) > 240 else "")
            )
            for label, indices in self._class_indices.items():
                if label not in label_to_idx:
                    continue
                # Top-3 neighbor mean within class
                class_sims = sims[row, indices]
                top = np.sort(class_sims)[-min(3, len(class_sims)) :]
                class_scores[row, label_to_idx[label]] = float(top.mean())

        # row-wise softmax
        for row in range(n):
            r = class_scores[row]
            if r.max() <= 0:
                class_scores[row] = 1.0 / n_labels
            else:
                e = np.exp((r - r.max()) / 0.5)
                class_scores[row] = e / e.sum()

        return class_scores, similar


class OptionalClimateBERT:
    """Optional sentence embeddings; falls back silently if unavailable."""

    def __init__(self):
        self.model = None
        self.label_embeddings: Optional[np.ndarray] = None
        try:
            from sentence_transformers import SentenceTransformer

            # Common public checkpoint; user can swap if team uses another
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def build_label_prototypes(self, labeled_df: pd.DataFrame) -> None:
        if not self.available:
            return
        texts_by_label: Dict[str, List[str]] = {}
        for _, row in labeled_df.iterrows():
            lab = row["canonical_label"]
            texts_by_label.setdefault(lab, []).append(preprocess_text(row["body"])[:512])
        protos = []
        for label in CANONICAL_LABELS:
            examples = texts_by_label.get(label, [TAXONOMY[label]])
            emb = self.model.encode(examples, show_progress_bar=False)
            protos.append(emb.mean(axis=0))
        self.label_embeddings = np.vstack(protos)

    def score_batch(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.available or self.label_embeddings is None:
            return None
        emb = self.model.encode(
            [preprocess_text(t)[:512] for t in texts],
            show_progress_bar=False,
            batch_size=64,
        )
        sims = cosine_similarity(emb, self.label_embeddings)
        probs = np.exp(sims - sims.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class OptionalZeroShot:
    def __init__(self, device: int = -1):
        self.pipe = None
        try:
            from transformers import pipeline

            self.pipe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
        except Exception:
            self.pipe = None

    @property
    def available(self) -> bool:
        return self.pipe is not None

    def classify(self, text: str) -> Dict[str, float]:
        if not self.available:
            return {l: 0.0 for l in CANONICAL_LABELS}
        # Hypothesis templates help disambiguate
        candidate_labels = [
            f"This comment expresses: {lbl}. {TAXONOMY[lbl][:120]}"
            for lbl in CANONICAL_LABELS
        ]
        result = self.pipe(
            preprocess_text(text)[:512],
            candidate_labels=candidate_labels,
            multi_label=False,
        )
        out = {l: 0.0 for l in CANONICAL_LABELS}
        for lab, score in zip(result["labels"], result["scores"]):
            # Map back from template prefix
            for canonical in CANONICAL_LABELS:
                if canonical in lab:
                    out[canonical] = float(score)
                    break
        return out


def combine_scores(
    kw: Dict[str, float],
    sem_row: np.ndarray,
    bert_row: Optional[np.ndarray],
    zs: Optional[Dict[str, float]],
    weights: Tuple[float, float, float, float],
) -> Dict[str, float]:
    w_kw, w_sem, w_bert, w_zs = weights
    kw_norm = normalize_score_dict(kw)
    combined = {}
    for i, label in enumerate(CANONICAL_LABELS):
        v = w_kw * kw_norm[label] + w_sem * float(sem_row[i])
        if bert_row is not None:
            v += w_bert * float(bert_row[i])
        if zs is not None:
            v += w_zs * zs.get(label, 0.0)
        # Priority boost (modest — semantic still dominates)
        if label in PRIORITY_LABELS:
            v *= 1.08
        combined[label] = v
    return combined


def build_reason(
    label: str,
    kw_norm: Dict[str, float],
    matched: List[str],
    similar: str,
    margin: float,
) -> str:
    parts = []
    label_matches = [m for m in matched if m.startswith(label + ":")]
    if label_matches:
        patterns = [m.split(":", 1)[1] for m in label_matches[:3]]
        parts.append(f"keyword patterns: {', '.join(patterns)}")
    if similar:
        parts.append(f"similar to labeled example: \"{similar[:120]}…\"")
    parts.append(f"definition: {TAXONOMY[label][:100]}…")
    if margin < 0.12:
        parts.append("close runner-up label; review recommended")
    return "; ".join(parts)


def predict_one(
    text: str,
    compiled_keywords: Dict[str, List[Tuple[re.Pattern, float]]],
    sem_row: np.ndarray,
    bert_row: Optional[np.ndarray],
    similar: str,
    zero_shot: Optional[OptionalZeroShot],
    use_zero_shot: bool,
    weights: Tuple[float, float, float, float],
    review_threshold: float,
    margin_threshold: float,
) -> Prediction:
    kw_raw, matched = keyword_scores(text, compiled_keywords)
    kw_norm = normalize_score_dict(kw_raw)

    zs_dict = None
    if use_zero_shot and zero_shot and zero_shot.available:
        zs_dict = zero_shot.classify(text)

    combined = combine_scores(kw_raw, sem_row, bert_row, zs_dict, weights)
    sorted_raw = sorted(combined.items(), key=lambda x: -x[1])
    top_label, top_raw = sorted_raw[0]
    second_raw = sorted_raw[1][1] if len(sorted_raw) > 1 else 0.0
    relative_margin = (top_raw - second_raw) / (top_raw + 1e-9)

    kw_top = max(kw_norm.items(), key=lambda x: x[1])[0]
    sem_top = CANONICAL_LABELS[int(np.argmax(sem_row))]
    kw_agree = kw_norm.get(top_label, 0.0)
    sem_agree = float(sem_row[CANONICAL_LABELS.index(top_label)])
    agreement_bonus = 0.12 if kw_top == top_label else 0.0
    agreement_bonus += 0.12 if sem_top == top_label else 0.0

    # Sharper distribution for reporting only
    prob = normalize_score_dict(combined, temperature=0.30)
    top_prob = prob[top_label]

    confidence = float(
        min(
            0.99,
            0.20
            + relative_margin * 0.55
            + top_prob * 0.35
            + kw_agree * 0.15
            + sem_agree * 0.10
            + agreement_bonus,
        )
    )
    margin = top_prob - (sorted(prob.values(), reverse=True)[1] if len(prob) > 1 else 0.0)

    needs_review = (
        confidence < review_threshold
        or relative_margin < margin_threshold
        or (top_label in PRIORITY_LABELS and confidence < 0.42)
    )

    reason = build_reason(top_label, kw_norm, matched, similar, margin)

    return Prediction(
        label=top_label,
        confidence=confidence,
        keyword_score=float(kw_raw.get(top_label, 0.0)),
        semantic_score=float(sem_row[list(CANONICAL_LABELS).index(top_label)]),
        zero_shot_score=float(zs_dict.get(top_label, 0.0)) if zs_dict else 0.0,
        matched_keywords=matched[:15],
        similar_example=similar,
        reason=reason,
        needs_human_review=needs_review,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    labeled_path = Path(args.labeled)
    unlabeled_path = Path(args.unlabeled)
    output_path = Path(args.output)

    print(f"Loading labeled data: {labeled_path}")
    labeled_df = load_labeled(labeled_path)
    print(f"  {len(labeled_df)} labeled examples")

    print(f"Loading unlabeled data: {unlabeled_path}")
    unlabeled_df = load_unlabeled(unlabeled_path, args.max_rows)
    print(f"  {len(unlabeled_df)} unlabeled comments")

    compiled = compile_keyword_patterns()
    semantic = SemanticMatcher(labeled_df)

    climatebert = OptionalClimateBERT()
    if args.use_climatebert and climatebert.available:
        print("Building optional embedding prototypes…")
        climatebert.build_label_prototypes(labeled_df)
    elif args.use_climatebert:
        print("ClimateBERT/embeddings unavailable; using TF-IDF only.")

    zero_shot = OptionalZeroShot() if args.use_zero_shot else None
    if args.use_zero_shot and (not zero_shot or not zero_shot.available):
        print("Zero-shot model unavailable; skipping.")

    w_kw = args.weight_keyword
    w_sem = args.weight_semantic
    w_bert = args.weight_bert if climatebert.available else 0.0
    w_zs = args.weight_zero_shot if (zero_shot and zero_shot.available) else 0.0
    # Renormalize
    total = w_kw + w_sem + w_bert + w_zs
    weights = (w_kw / total, w_sem / total, w_bert / total, w_zs / total)

    bodies = unlabeled_df["body"].tolist()
    batch_size = args.batch_size
    rows_out: List[dict] = []
    zs_budget = args.zero_shot_cap

    for start in range(0, len(bodies), batch_size):
        end = min(start + batch_size, len(bodies))
        batch_texts = bodies[start:end]
        sem_matrix, similars = semantic.score_batch(batch_texts)
        bert_matrix = (
            climatebert.score_batch(batch_texts)
            if climatebert.available and climatebert.label_embeddings is not None
            else None
        )

        for i, text in enumerate(batch_texts):
            global_i = start + i
            use_zs_this = (
                args.use_zero_shot
                and zero_shot
                and zero_shot.available
                and zs_budget > 0
                and (
                    global_i < args.zero_shot_cap
                    or (global_i % max(1, len(bodies) // max(1, zs_budget)) == 0)
                )
            )
            if use_zs_this:
                zs_budget -= 1

            bert_row = bert_matrix[i] if bert_matrix is not None else None
            pred = predict_one(
                text,
                compiled,
                sem_matrix[i],
                bert_row,
                similars[i],
                zero_shot,
                use_zs_this,
                weights,
                args.review_threshold,
                args.margin_threshold,
            )
            rows_out.append(
                {
                    "body": text,
                    "predicted_label": pred.label,
                    "confidence": round(pred.confidence, 4),
                    "matched_keywords": "|".join(pred.matched_keywords),
                    "similar_example": pred.similar_example,
                    "needs_human_review": pred.needs_human_review,
                    "reason": pred.reason,
                    "keyword_score": round(pred.keyword_score, 4),
                    "semantic_score": round(pred.semantic_score, 4),
                }
            )

        if (start // batch_size) % 10 == 0:
            print(f"  processed {end}/{len(bodies)}…")

    out_df = pd.DataFrame(rows_out)

    # Prioritize review queue: priority labels + low confidence first
    out_df["_priority_sort"] = (
        out_df["predicted_label"].isin(PRIORITY_LABELS).astype(int) * 2
        + out_df["needs_human_review"].astype(int)
    )
    out_df = out_df.sort_values(
        ["_priority_sort", "confidence"], ascending=[False, True]
    ).drop(columns=["_priority_sort"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out_df)} rows -> {output_path}")

    review = out_df[out_df["needs_human_review"]]
    priority = out_df[out_df["predicted_label"].isin(PRIORITY_LABELS)]
    print(f"  needs_human_review: {len(review)} ({100*len(review)/len(out_df):.1f}%)")
    print(f"  priority-label predictions: {len(priority)}")
    print("  label distribution (top 10):")
    for lab, cnt in out_df["predicted_label"].value_counts().head(10).items():
        print(f"    {cnt:6d}  {lab}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label Reddit comments (weak labels).")
    p.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    p.add_argument("--unlabeled", type=Path, default=DEFAULT_UNLABELED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-rows", type=int, default=None, help="Limit unlabeled rows (for testing).")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--review-threshold", type=float, default=0.42)
    p.add_argument("--margin-threshold", type=float, default=0.12)
    p.add_argument("--use-climatebert", action="store_true", help="Use sentence embeddings if installed.")
    p.add_argument("--use-zero-shot", action="store_true", help="Use BART-MNLI on a capped subset.")
    p.add_argument("--zero-shot-cap", type=int, default=500, help="Max zero-shot calls (expensive).")
    p.add_argument("--weight-keyword", type=float, default=0.25)
    p.add_argument("--weight-semantic", type=float, default=0.65)
    p.add_argument("--weight-bert", type=float, default=0.15)
    p.add_argument("--weight-zero-shot", type=float, default=0.20)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
