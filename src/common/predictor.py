"""Load trained models and produce predictions with explanations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from common.label_utils import MODELS, body_hash
from common.taxonomy import CANONICAL_LABELS, TAXONOMY, normalize_label

BEST_MODEL_META = MODELS / "best_model.json"


def load_best_model() -> Tuple[Any, str, Dict]:
    if not BEST_MODEL_META.exists():
        raise FileNotFoundError(
            f"No trained model at {BEST_MODEL_META}. Run: python src/tfidf/legacy_train_evaluate.py"
        )
    meta = json.loads(BEST_MODEL_META.read_text(encoding="utf-8"))
    name = meta["model_name"]
    path = MODELS / meta["artifact"]
    bundle = joblib.load(path)
    return bundle, name, meta


def predict(text: str, bundle: dict) -> Dict[str, Any]:
    model_type = bundle["type"]
    text = str(text).strip()
    if not text:
        return {"label": None, "confidence": 0.0, "probs": {}, "explanation": "Empty text."}

    if model_type == "tfidf_sklearn":
        vec = bundle["vectorizer"]
        clf = bundle["classifier"]
        X = vec.transform([text])
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
        elif hasattr(clf, "decision_function"):
            dec = clf.decision_function(X)[0]
            e = np.exp(dec - dec.max())
            proba = e / e.sum()
        else:
            pred = clf.predict(X)[0]
            proba = np.array([1.0 if c == pred else 0.0 for c in clf.classes_])
        classes = list(clf.classes_)
        idx = int(np.argmax(proba))
        label = classes[idx]
        probs = {c: float(proba[i]) for i, c in enumerate(classes)}
        keywords = _top_tfidf_terms(vec, clf, label, top_n=8)
        return {
            "label": label,
            "confidence": float(proba[idx]),
            "probs": probs,
            "explanation": _format_explanation(label, probs, keywords),
            "keywords": keywords,
        }

    if model_type == "embedding_lr":
        emb = bundle["embedder"]
        clf = bundle["classifier"]
        vec_text = emb.encode([text], show_progress_bar=False)
        proba = clf.predict_proba(vec_text)[0]
        classes = list(clf.classes_)
        idx = int(np.argmax(proba))
        label = classes[idx]
        probs = {c: float(proba[i]) for i, c in enumerate(classes)}
        return {
            "label": label,
            "confidence": float(proba[idx]),
            "probs": probs,
            "explanation": _format_explanation(label, probs, []),
            "keywords": [],
        }

    if model_type == "transformer":
        pipe = bundle["pipeline"]
        out = pipe(text[:512])[0]
        label = out["label"]
        confidence = float(out["score"])
        return {
            "label": normalize_label(label) or label,
            "confidence": confidence,
            "probs": {label: confidence},
            "explanation": f"Fine-tuned transformer prediction ({confidence:.2f}).",
            "keywords": [],
        }

    raise ValueError(f"Unknown bundle type: {model_type}")


def _top_tfidf_terms(vectorizer, classifier, label: str, top_n: int = 8) -> List[str]:
    if not hasattr(classifier, "coef_"):
        return []
    try:
        idx = list(classifier.classes_).index(label)
    except ValueError:
        return []
    coef = classifier.coef_[idx]
    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(coef)[-top_n:][::-1]
    return [str(feature_names[i]) for i in top_idx if coef[i] > 0]


def _format_explanation(label: str, probs: Dict[str, float], keywords: List[str]) -> str:
    parts = [f"Predicted **{label}**: {TAXONOMY.get(label, '')}"]
    sorted_p = sorted(probs.items(), key=lambda x: -x[1])[:3]
    parts.append(
        "Top alternatives: "
        + ", ".join(f"{l} ({p:.2f})" for l, p in sorted_p)
    )
    if keywords:
        parts.append("Important terms: " + ", ".join(keywords))
    return " ".join(parts)
