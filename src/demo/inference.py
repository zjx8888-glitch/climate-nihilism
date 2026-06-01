#!/usr/bin/env python3
"""
ClimateBERT inference for the Streamlit demo (Jinxi).

Loads **pretrained** artifacts from ``outputs/climatebert/`` or
``outputs/climatebert_v2/`` (no retraining required for new sentences).

Train once:
  python src/climatebert/train.py --dataset-version v2

Then classify new text:
  PYTHONPATH=src python -m demo.inference --text "Your comment here..."
  streamlit run app/streamlit_app.py

Environment:
  CLIMATEBERT_MODEL_VERSION=v2   # or v1 (optional; auto-detects if unset)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import climatebert_output_dir, resolve_climatebert_version
from common.taxonomy import NIHILISM_FOCUS, TAXONOMY

# In-process cache keyed by model version
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}


def _artifact_paths(version: str) -> Dict[str, Path]:
    out = climatebert_output_dir(version)
    return {
        "out_dir": out,
        "multiclass": out / "embedding_lr_multiclass.joblib",
        "binary": out / "embedding_lr_binary_nihilism.joblib",
        "metrics": out / "climatebert_metrics.json",
    }


def _train_command(version: str) -> str:
    return f"python src/climatebert/train.py --dataset-version {version}"


def missing_model_message(version: str) -> str:
    paths = _artifact_paths(version)
    return (
        f"No pretrained model at {paths['multiclass']}. "
        f"Train once: `{_train_command(version)}`"
    )


def load_climatebert_metrics(version: Optional[str] = None) -> Dict[str, Any]:
    """Load test-set metrics from climatebert_metrics.json (cached)."""
    ver = resolve_climatebert_version(version or os.environ.get("CLIMATEBERT_MODEL_VERSION"))
    if ver in _METRICS_CACHE:
        return _METRICS_CACHE[ver]
    paths = _artifact_paths(ver)
    if not paths["metrics"].exists():
        raise FileNotFoundError(missing_model_message(ver))
    data = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    data["_model_version"] = ver
    _METRICS_CACHE[ver] = data
    return data


def load_climatebert_model(version: Optional[str] = None) -> Dict[str, Any]:
    """
    Load SentenceTransformer + saved logistic-regression heads (pretrained).

    Cached in-process per version; use ``st.cache_resource`` in Streamlit.
    """
    ver = resolve_climatebert_version(version or os.environ.get("CLIMATEBERT_MODEL_VERSION"))
    if ver in _MODEL_CACHE:
        return _MODEL_CACHE[ver]

    paths = _artifact_paths(ver)
    if not paths["multiclass"].exists():
        raise FileNotFoundError(missing_model_message(ver))

    mc_bundle = joblib.load(paths["multiclass"])
    embedder_name = mc_bundle.get(
        "embedder_name", "climatebert/distilroberta-base-climate-f"
    )

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedder_name)
    binary_clf = None
    if paths["binary"].exists():
        bin_bundle = joblib.load(paths["binary"])
        binary_clf = bin_bundle.get("classifier")

    bundle = {
        "model_version": ver,
        "embedder": embedder,
        "embedder_name": embedder_name,
        "multiclass_clf": mc_bundle["classifier"],
        "binary_clf": binary_clf,
        "artifact_dir": str(paths["out_dir"]),
    }
    _MODEL_CACHE[ver] = bundle
    return bundle


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
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify one comment: 14-class label + nihilism probability.

    Returns dict with prediction fields, or ``{"error": "<message>"}`` on failure.
    """
    text = str(text).strip()
    if not text:
        return {"error": "Empty text."}

    try:
        bundle = model if model is not None else load_climatebert_model(version=version)
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
            metrics = load_climatebert_metrics(version=bundle.get("model_version"))
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
            "model_version": bundle.get("model_version"),
            "predicted_label": predicted_label,
            "multiclass_confidence": multiclass_confidence,
            "nihilism_probability": nihilism_probability,
            "top_3_labels": top_3_labels,
            "taxonomy_definition": TAXONOMY.get(predicted_label),
            "nihilism_f1_test": nihilism_f1_test,
            "embed_model": embed_model,
            "is_climate_nihilism": predicted_label == NIHILISM_FOCUS,
        }
    except Exception as exc:
        return {"error": f"ClimateBERT inference failed: {exc}"}


# Backward-compatible alias for streamlit_app.py
MISSING_MSG = (
    "No pretrained ClimateBERT model found. "
    "Train once: `python src/climatebert/train.py --dataset-version v2`"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClimateBERT demo inference")
    parser.add_argument("--text", required=True, help="Comment text to classify")
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default=None,
        help="Model checkpoint (default: auto-detect v2 if present)",
    )
    args = parser.parse_args()

    out = predict_climatebert(args.text, version=args.version)
    if out.get("error"):
        print(f"Error: {out['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Model version:        {out.get('model_version', '?')}")
    print(f"Predicted label:      {out['predicted_label']}")
    print(f"Climate nihilism?     {out.get('is_climate_nihilism')}")
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
