# Climate Nihilism Detection on Reddit

**CS 496 / Northwestern** — NLP pipeline to detect climate-related opinions on Reddit, with emphasis on **Climate nihilism** vs **climate anxiety** vs **Climate nihilism critique** (14-class taxonomy).

---

## Project Overview

We classify Reddit comments into a fixed taxonomy of climate opinions. The research focus is detecting **Climate nihilism** (hopelessness / futility about stopping climate change) and distinguishing it from emotionally similar classes. See [docs/project_overview.md](docs/project_overview.md) for motivation, data choices, and workflow.

---

## Dataset Description

> **Git:** Large CSVs and model outputs are not committed. See [data/README.md](data/README.md) for what to place locally.

| Item | Location | Notes |
|------|----------|--------|
| Processed manual labels | `data/processed/preprocessed_comments_2000_to_label.csv` | ~2k rows, Excel recovery applied |
| Recovered gold labels | `data/labeled/recovered_labeled_dataset.csv` | **1,845** rows, canonical labels |
| Training merge | `data/labeled/final_training_dataset.csv` | **1,844** rows (deduped) |
| Train/val/test splits | `data/labeled/splits.json` | 70 / 15 / 15, seed `42` |
| Weak / silver labels | `data/weak_labels/auto_labeled_comments.csv` | Large-scale auto labels |
| Raw unlabeled Reddit | `data/raw/preprocessed_comments_400000.csv` | Place file locally (see `data/raw/README.md`) |

**Climate nihilism** examples in gold set: **66** (sparse — see limitations in reports).

---

## Taxonomy Labels

14 canonical labels defined in `src/common/taxonomy.py` (from `data labels.pdf`).

| Label | Meaning (short) |
|-------|------------------|
| **climate anxiety** | Worry/fear; action still possible |
| **Climate nihilism** | Hopelessness; action seen as futile |
| **Climate nihilism critique** | Pushback against doomism |
| **Climate denial** | Rejects climate science/consensus |
| … | See taxonomy module for all 14 |

---

## Team Responsibilities

| Owner | Scope | Entry points |
|-------|--------|----------------|
| **Jinxi Zhang** | ClimateBERT training, evaluation, report, predictions | `src/climatebert/` |
| **Liu** | TF-IDF & classical baselines | `src/tfidf/` |
| **Josh** | Streamlit demo & visualization | `app/streamlit_app.py` |
| **Madeleine Worrall** | Preprocessing, cleaning, taxonomy, weak labeling | `src/preprocessing/`, `src/labeling/` |

Full matrix: [docs/team_responsibilities.md](docs/team_responsibilities.md)

---

## Project Structure

```
project-root/
├── README.md
├── requirements.txt
├── data labels.pdf
├── docs/
│   ├── project_overview.md
│   ├── methodology.md
│   ├── team_responsibilities.md
│   ├── climatebert_results.md      # Jinxi
│   ├── tfidf_results.md            # Liu (TODO)
│   └── final_results_summary.md
├── data/
│   ├── raw/                        # Large unlabeled dumps
│   ├── processed/                  # Cleaned CSVs (Madeleine)
│   ├── labeled/                    # Gold labels + splits
│   └── weak_labels/                # Auto / LLM / review
├── notebooks/
│   ├── climatebert_experiment.ipynb
│   ├── tfidf_experiment.ipynb
│   └── data_analysis.ipynb
├── src/
│   ├── common/                     # paths, taxonomy, label_utils
│   ├── preprocessing/              # Madeleine
│   ├── labeling/                   # Madeleine
│   ├── climatebert/                # Jinxi
│   ├── tfidf/                      # Liu
│   ├── evaluation/
│   └── demo/                       # helpers (app is in /app)
├── outputs/
│   ├── climatebert/                # Jinxi
│   ├── tfidf/                      # Liu
│   ├── figures/
│   ├── predictions/
│   ├── reports/
│   └── error_analysis/
└── app/
    └── streamlit_app.py            # Josh
```

Legacy shims at `src/train_climatebert.py`, `src/demo_app.py`, etc. forward to the new paths.

---

**Teammates:** see [docs/teammate_app_setup.md](docs/teammate_app_setup.md) to run the app with the pretrained model (`models/climatebert_v2_pretrained_demo.zip`).

---

## Setup & Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Core:** pandas, scikit-learn, matplotlib, seaborn, joblib  
**ClimateBERT (Jinxi):** sentence-transformers, transformers, torch  
**Demo (Josh):** streamlit  
**Weak labeling (Madeleine):** optional openai for LLM script  

Set `PYTHONPATH=src` if using module imports: `python -m climatebert.train` (from repo root).

---

## How to Run

### 1. Data pipeline (Madeleine)

```bash
python src/labeling/recover_labeled_dataset.py
python src/labeling/build_final_dataset.py
# Optional:
python src/labeling/auto_label_comments.py
python src/preprocessing/inspect_datasets.py
```

### 2. ClimateBERT (Jinxi)

**Train once** (saves pretrained classifiers to `outputs/climatebert_v2/`):

```bash
python src/climatebert/prepare_v2_dataset.py   # if v2 CSV not built yet
python src/climatebert/train.py --dataset-version v2
```

Artifacts used for inference (keep locally; gitignored):

- `outputs/climatebert_v2/embedding_lr_multiclass.joblib` — 14-class head  
- `outputs/climatebert_v2/embedding_lr_binary_nihilism.joblib` — nihilism vs not  
- `outputs/climatebert_v2/climatebert_metrics.json` — test metrics for demo captions  

**Classify new text** (no retraining):

```bash
PYTHONPATH=src python -m demo.inference --text "Your sentence here..." --version v2
```

Optional: `export CLIMATEBERT_MODEL_VERSION=v2` (default auto-detects v2 if artifacts exist).

```bash
python src/climatebert/train.py --finetune --epochs 3   # optional
```

### 3. TF-IDF baseline (Liu)

```bash
python src/tfidf/train.py                    # TODO(Liu): implement
python src/tfidf/legacy_train_evaluate.py      # legacy reference
```

### 4. Demo (Josh)

```bash
streamlit run app/streamlit_app.py
```

#### ClimateBERT inference in demo

The **ClimateBERT** panel loads **pretrained** `.joblib` weights (train once, then classify any new sentence). TF-IDF remains a placeholder until Liu connects it.

**1. Train once** (recommended: v2):

```bash
python src/climatebert/train.py --dataset-version v2
```

**2. Required files** (under `outputs/climatebert_v2/`):

| File | Purpose |
|------|---------|
| `embedding_lr_multiclass.joblib` | 14-class classifier on ClimateBERT embeddings |
| `embedding_lr_binary_nihilism.joblib` | Nihilism vs not (probability score) |
| `climatebert_metrics.json` | Test metrics shown in the app |

**3. Test a sentence (CLI):**

```bash
PYTHONPATH=src python -m demo.inference --text "We are past the point of no return." --version v2
```

**4. Run the app** — sidebar lets you pick v1/v2 checkpoint:

```bash
streamlit run app/streamlit_app.py
```

Inference: `src/demo/inference.py` (`predict_climatebert`, `load_climatebert_model`).

---

## Training Pipelines

| Pipeline | Input | Output |
|----------|--------|--------|
| Label recovery | `data/processed/` | `data/labeled/recovered_labeled_dataset.csv` |
| Final dataset | Recovered + human verified | `data/labeled/final_training_dataset.csv`, `splits.json` |
| ClimateBERT | Labeled + splits | `outputs/climatebert/`, `outputs/predictions/`, `outputs/error_analysis/` |
| TF-IDF | Same splits | `outputs/tfidf/` (TODO Liu) |

All trainers share **`data/labeled/splits.json`** — do not regenerate splits independently.

---

## Demo Instructions

1. Train at least one model (ClimateBERT or TF-IDF legacy).
2. Run `streamlit run app/streamlit_app.py`.
3. # TODO(Josh): wire `outputs/climatebert/` and `outputs/tfidf/` into the UI.

---

## Results Summary

| Model | Macro F1 (test) | Nihilism F1 | Doc |
|-------|-----------------|-------------|-----|
| ClimateBERT + LR | 0.230 | 0.300 | [climatebert_results.md](docs/climatebert_results.md) |
| TF-IDF + LR (legacy) | 0.185 | 0.182 | [tfidf_results.md](docs/tfidf_results.md) |

Full comparison: [docs/final_results_summary.md](docs/final_results_summary.md), [outputs/reports/recovered_vs_old_results.md](outputs/reports/recovered_vs_old_results.md).

---

## Future Work

- More **Climate nihilism** gold labels (class is still small)
- Liu: migrate TF-IDF fully into `src/tfidf/train.py`
- Josh: integrate predictions + team visualizations in Streamlit
- Madeleine: human review loop, quality filters on weak labels
- Optional: ensemble or calibration across ClimateBERT + TF-IDF

---

## Reproducibility

- Random seed: **42**
- Splits: `data/labeled/splits.json`
- Path constants: `src/common/paths.py`

---

## Files intentionally not committed

These are listed in `.gitignore` — clone the repo, then regenerate or download locally:

| Category | Examples | How to obtain |
|----------|----------|----------------|
| Virtual environment | `.venv/` | `python3 -m venv .venv && pip install -r requirements.txt` |
| Secrets | `.env`, API keys | Create locally (LLM labeling only) |
| Hugging Face cache | `~/.cache/huggingface/` | Downloaded on first `train.py` / demo run |
| Large weak labels | `data/weak_labels/auto_labeled_comments.csv` (~437 MB) | `python src/labeling/auto_label_comments.py` |
| Raw Reddit dump | `data/raw/preprocessed_comments_400000.csv` | Place per `data/raw/README.md` |
| Model weights (joblib) | `outputs/climatebert/*.joblib`, `outputs/tfidf/models/*.joblib` | `python src/climatebert/train.py`, `python src/tfidf/legacy_train_evaluate.py` |
| LLM checkpoints | `data/weak_labels/llm_label_checkpoint.jsonl` | `python src/labeling/llm_label_comments.py` |

**Committed for reproducibility:** gold labels (`data/labeled/`, `data/processed/`), metrics JSON, figures (PNG), prediction/error-analysis CSVs, and source code.

---

## Pushing to GitHub

After creating an empty repository on GitHub:

```bash
git remote add origin <GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

Replace `<GITHUB_REPO_URL>` with your repo URL (e.g. `https://github.com/your-org/climate-nihilism-nlp.git`).
