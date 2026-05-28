"""Central path constants for the collaborative project layout."""

from __future__ import annotations

from pathlib import Path

# Project root (…/496finalproj)
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"
APP = ROOT / "app"

# Data
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_LABELED = DATA / "labeled"
DATA_WEAK = DATA / "weak_labels"

# Outputs (by owner / artifact type)
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
CLIMATEBERT_OUT = OUTPUTS / "climatebert"
TFIDF_OUT = OUTPUTS / "tfidf"
TFIDF_MODELS = TFIDF_OUT / "models"
PREDICTIONS_OUT = OUTPUTS / "predictions"
REPORTS_OUT = OUTPUTS / "reports"
ERROR_ANALYSIS_OUT = OUTPUTS / "error_analysis"

# Labeled / training data
RECOVERED_LABELED = DATA_LABELED / "recovered_labeled_dataset.csv"
FINAL_TRAINING = DATA_LABELED / "final_training_dataset.csv"
SPLITS_JSON = DATA_LABELED / "splits.json"
BROKEN_LABEL_ROWS = DATA_LABELED / "broken_label_rows.csv"

# Processed manual labels
MANUAL_LABELED = DATA_PROCESSED / "preprocessed_comments_2000_to_label.csv"
MANUAL_LABELED_ALT = (
    DATA_PROCESSED
    / "preprocessed_comments_2000_to_label - preprocessed_comments_2000_to_label.csv"
)

# Large unlabeled Reddit dump (place file here; not committed)
REDDIT_UNLABELED_400K = DATA_RAW / "preprocessed_comments_400000.csv"
# Legacy location if data not yet moved to data/raw/
REDDIT_UNLABELED_400K_LEGACY = DATA_PROCESSED / "preprocessed_comments_400000.csv"

# Weak / silver labels
AUTO_LABELED = DATA_WEAK / "auto_labeled_comments.csv"
HUMAN_VERIFIED = DATA_WEAK / "human_verified_labels.csv"
HIGH_CONFIDENCE_TRAINING = DATA_WEAK / "high_confidence_training_dataset.csv"
AUGMENTED_TRAINING = DATA_WEAK / "augmented_training_dataset.csv"
LLM_LABELED = DATA_WEAK / "llm_labeled_comments.csv"
LLM_CHECKPOINT = DATA_WEAK / "llm_label_checkpoint.jsonl"
REVIEW_PROGRESS = DATA_WEAK / "review_progress.json"

# Taxonomy reference
TAXONOMY_PDF = ROOT / "data labels.pdf"

RANDOM_STATE = 42


def ensure_project_dirs() -> None:
    """Create standard folders if missing."""
    for path in (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_LABELED,
        DATA_WEAK,
        FIGURES,
        CLIMATEBERT_OUT,
        TFIDF_OUT,
        TFIDF_MODELS,
        PREDICTIONS_OUT,
        REPORTS_OUT,
        ERROR_ANALYSIS_OUT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def reddit_unlabeled_path() -> Path:
    """Return path to large unlabeled CSV (raw preferred)."""
    if REDDIT_UNLABELED_400K.exists():
        return REDDIT_UNLABELED_400K
    if REDDIT_UNLABELED_400K_LEGACY.exists():
        return REDDIT_UNLABELED_400K_LEGACY
    return REDDIT_UNLABELED_400K
