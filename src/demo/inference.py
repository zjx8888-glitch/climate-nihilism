#!/usr/bin/env python3
"""
ClimateBERT inference for the Streamlit demo (Jinxi).

Usage:
  python -m src.demo.inference --text "We are past the point of no return."
  PYTHONPATH=src python -m demo.inference --text "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import CLIMATEBERT_OUT
from common.taxonomy import NIHILISM_FOCUS, TAXONOMY

MULTICLASS_ARTIFACT = CLIMATEBERT_OUT / "embedding_lr_multiclass.joblib"
BINARY_ARTIFACT = CLIMATEBERT_OUT / "embedding_lr_binary_nihilism.joblib"
METRICS_FILE = CLIMATEBERT_OUT / "climatebert_metrics.json"

MISSING_MSG = "Run `python src/climatebert/train.py` first."

# In-process cache (safe to reuse across calls in CLI / Streamlit worker)
_MODEL_CACHE: Optional[Dict[str, Any]] = None
_METRICS_CACHE: Optional[Dict[str, Any]] = None


def _require_multiclass_artifact() -> Path:
    if not MULTICLASS_ARTIFACT.exists():
        raise FileNotFoundError(MISSING_MSG)
    return MULTICLASS_ARTIFACT


def load_climatebert_metrics() -> Dict[str, Any]:
    """Load test-set metrics from climatebert_metrics.json (cached)."""
    global _METRICS_CACHE
    if _METRICS_CACHE is not None:
        return _METRICS_CACHE
    if not METRICS_FILE.exists():
        raise FileNotFoundError(MISSING_MSG)
    _METRICS_CACHE = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return _METRICS_CACHE


def load_climatebert_model() -> Dict[str, Any]:
    """
    Load SentenceTransformer + multiclass LR (+ binary nihilism LR if present).

    Cached in-process; use ``st.cache_resource(load_climatebert_model)`` in Streamlit.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    _require_multiclass_artifact()
    mc_bundle = joblib.load(MULTICLASS_ARTIFACT)
    embedder_name = mc_bundle.get(
        "embedder_name", "climatebert/distilroberta-base-climate-f"
    )

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedder_name)
    binary_clf = None
    if BINARY_ARTIFACT.exists():
        bin_bundle = joblib.load(BINARY_ARTIFACT)
        binary_clf = bin_bundle.get("classifier")

    _MODEL_CACHE = {
        "embedder": embedder,
        "embedder_name": embedder_name,
        "multiclass_clf": mc_bundle["classifier"],
        "binary_clf": binary_clf,
    }
    return _MODEL_CACHE


def _encode(model: Dict[str, Any], text: str) -> np.ndarray:
    return model["embedder"].encode(
        [text[:8000]],
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def _top_k(
    classes: List[str], proba: np.ndarray, k: int = 3
) -> List[Tuple[str, float]]:
    k = min(k, len(classes))
    order = np.argsort(proba)[-k:][::-1]
    return [(classes[i], float(proba[i])) for i in order]


def _binary_nihilism_prob(model: Dict[str, Any], embedding: np.ndarray) -> Optional[float]:
    clf = model.get("binary_clf")
    if clf is None:
        return None
    proba = clf.predict_proba(embedding)[0]
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    if NIHILISM_FOCUS in classes:
        return float(proba[classes.index(NIHILISM_FOCUS)])
    return float(proba.max())


def predict_climatebert(
    text: str,
    model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
  Run ClimateBERT embedding + classifiers on a single comment.

  Returns dict with prediction fields, or ``{"error": "<message>"}`` on failure.
    """
    text = str(text).strip()
    if not text:
        return {"error": "Empty text."}

    try:
        bundle = model if model is not None else load_climatebert_model()
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    try:
        embedding = _encode(bundle, text)
        mc_clf = bundle["multiclass_clf"]
        proba = mc_clf.predict_proba(embedding)[0]
        classes = [str(c) for c in mc_clf.classes_]
        idx = int(np.argmax(proba))
        predicted_label = classes[idx]
        multiclass_confidence = float(proba[idx])
        top_3_labels = _top_k(classes, proba, k=3)
        nihilism_probability = _binary_nihilism_prob(bundle, embedding)

        try:
            metrics = load_climatebert_metrics()
            mc_metrics = metrics.get("approaches", {}).get(
                "embedding_lr_multiclass", {}
            )
            nihilism_f1_test = mc_metrics.get("nihilism_f1")
            embed_model = metrics.get("embed_model", bundle.get("embedder_name"))
        except FileNotFoundError:
            nihilism_f1_test = None
            embed_model = bundle.get("embedder_name")

        return {
            "error": None,
            "predicted_label": predicted_label,
            "multiclass_confidence": multiclass_confidence,
            "nihilism_probability": nihilism_probability,
            "top_3_labels": top_3_labels,
            "taxonomy_definition": TAXONOMY.get(predicted_label),
            "nihilism_f1_test": nihilism_f1_test,
            "embed_model": embed_model,
        }
    except Exception as exc:
        return {"error": f"ClimateBERT inference failed: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="ClimateBERT demo inference")
    parser.add_argument("--text", required=True, help="Comment text to classify")
    args = parser.parse_args()

    out = predict_climatebert(args.text)
    if out.get("error"):
        print(f"Error: {out['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Predicted label:      {out['predicted_label']}")
    print(f"Multiclass confidence: {out['multiclass_confidence']:.3f}")
    if out["nihilism_probability"] is not None:
        print(f"Nihilism probability:  {out['nihilism_probability']:.3f}")
    else:
        print("Nihilism probability:  (binary head not available)")
    print("Top 3 labels:")
    for lab, p in out["top_3_labels"]:
        print(f"  - {lab}: {p:.3f}")
    if out.get("taxonomy_definition"):
        print(f"Definition: {out['taxonomy_definition'][:120]}...")


if __name__ == "__main__":
    main()
