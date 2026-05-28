# Team Responsibilities

## Jinxi Zhang — ClimateBERT

| | |
|--|--|
| **Primary files** | `src/climatebert/train.py`, `notebooks/climatebert_experiment.ipynb` |
| **Outputs owned** | `outputs/climatebert/`, `outputs/predictions/climatebert_predictions.csv`, `outputs/error_analysis/climatebert_error_analysis.csv` |
| **Documentation** | `docs/climatebert_results.md` |
| **Responsibilities** | Embeddings + logistic regression, optional fine-tuning, binary nihilism head, metrics, confusion matrix, PR curve, error analysis |
| **Integration** | Consumes `data/labeled/` and `data/labeled/splits.json`; Josh loads `outputs/predictions/`; Liu compares against TF-IDF metrics |

**Run:** `python src/climatebert/train.py`

---

## Liu — TF-IDF & classical baselines

| | |
|--|--|
| **Primary files** | `src/tfidf/train.py`, `notebooks/tfidf_experiment.ipynb` |
| **Outputs owned** | `outputs/tfidf/` (metrics, models), TF-IDF figures in `outputs/figures/` |
| **Documentation** | `docs/tfidf_results.md` |
| **Responsibilities** | TF-IDF + Logistic Regression, TF-IDF + Linear SVM, binary nihilism baseline, comparison to ClimateBERT |
| **Integration** | Same splits as Jinxi; reference implementation in `src/tfidf/legacy_train_evaluate.py` until migrated |

**Run:** `python src/tfidf/train.py` (TODO) or `python src/tfidf/legacy_train_evaluate.py`

---

## Josh — Streamlit demo

| | |
|--|--|
| **Primary files** | `app/streamlit_app.py` |
| **Outputs consumed** | `outputs/climatebert/`, `outputs/tfidf/`, `outputs/predictions/`, `data/weak_labels/` |
| **Responsibilities** | Interactive classification UI, visualization, wiring model artifacts |
| **Integration** | `# TODO(Josh)` in `app/streamlit_app.py` for ClimateBERT and TF-IDF outputs |

**Run:** `streamlit run app/streamlit_app.py`

---

## Madeleine Worrall — Data & labeling pipeline

| | |
|--|--|
| **Primary files** | `src/preprocessing/`, `src/labeling/` |
| **Outputs owned** | `data/processed/`, `data/labeled/`, `data/weak_labels/` |
| **Responsibilities** | Cleaning, taxonomy normalization (`src/common/taxonomy.py`), label recovery, weak/LLM labeling, final dataset & splits |
| **Integration** | Feeds all trainers via `data/labeled/final_training_dataset.csv` and `splits.json` |

**Run:** `python src/labeling/recover_labeled_dataset.py` → `python src/labeling/build_final_dataset.py`

---

## Shared (read-only for most teammates)

| Path | Purpose |
|------|---------|
| `src/common/paths.py` | Canonical paths — update here when adding folders |
| `src/common/taxonomy.py` | 14-class label definitions |
| `src/common/label_utils.py` | Loaders, merges, stratified splits |

Do not edit another teammate’s training script without coordination; use TODO comments and pull requests.
