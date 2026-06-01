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
CLIMATEBERT_V2_OUT = OUTPUTS / "climatebert_v2"
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

# Dataset v2 (cleaned_data-2)
CLEANED_DATA_V2 = DATA_LABELED / "cleaned_data_2.csv"
FINAL_TRAINING_V2 = DATA_LABELED / "final_training_dataset_v2.csv"
SPLITS_V2_JSON = DATA_LABELED / "splits_v2.json"
DATASET_V2_INSPECTION = DATA_LABELED / "dataset_v2_inspection.json"

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

# Default ClimateBERT checkpoint for demo / inference (v2 if artifacts exist)
DEFAULT_CLIMATEBERT_VERSION = "v2"


def climatebert_output_dir(version: str = DEFAULT_CLIMATEBERT_VERSION) -> Path:
    """Directory containing embedding_lr_*.joblib and climatebert_metrics.json."""
    if version == "v1":
        return CLIMATEBERT_OUT
    if version == "v2":
        return CLIMATEBERT_V2_OUT
    raise ValueError(f"Unknown ClimateBERT version: {version!r}. Use 'v1' or 'v2'.")


def resolve_climatebert_version(preferred: str | None = None) -> str:
    """
    Pick v1 or v2 for inference.

    Uses ``preferred`` if set; else ``DEFAULT_CLIMATEBERT_VERSION`` when artifacts
    exist; else whichever of v2/v1 has ``embedding_lr_multiclass.joblib``.
    """
    if preferred in ("v1", "v2"):
        return preferred
    for ver in (DEFAULT_CLIMATEBERT_VERSION, "v1", "v2"):
        d = climatebert_output_dir(ver)
        if (d / "embedding_lr_multiclass.joblib").exists():
            return ver
    return DEFAULT_CLIMATEBERT_VERSION


def ensure_project_dirs() -> None:
    """Create standard folders if missing."""
    for path in (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_LABELED,
        DATA_WEAK,
        FIGURES,
        CLIMATEBERT_OUT,
        CLIMATEBERT_V2_OUT,
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
