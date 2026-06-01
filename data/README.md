# Data files (not in Git)

Large CSVs are **gitignored**. Each teammate places files locally as below.

## Required for ClimateBERT v2 training

| File | Location | Source |
|------|----------|--------|
| Labeled corpus | `data/processed/cleaned_data-2.csv` | Team export / shared drive |
| (auto) Training merge | `data/labeled/final_training_dataset_v2.csv` | `python src/climatebert/prepare_v2_dataset.py` |
| (auto) Splits | `data/labeled/splits_v2.json` | same script — **committed** (small JSON) |

Copy or symlink: `cleaned_data-2.csv` can also live at `data/labeled/cleaned_data_2.csv` (prepare script uses labeled copy).

## Required for ClimateBERT v1 / recovery pipeline

| File | Location |
|------|----------|
| Recovered labels | `data/labeled/recovered_labeled_dataset.csv` |
| Final v1 training | `data/labeled/final_training_dataset.csv` |
| Splits | `data/labeled/splits.json` — **committed** |

## Optional / large

| File | Location | Notes |
|------|----------|--------|
| ~400k unlabeled Reddit | `data/raw/preprocessed_comments_400000.csv` | Weak labeling only |
| Auto-labeled dump | `data/weak_labels/auto_labeled_comments.csv` | ~450MB |

## What *is* in Git

- `data/labeled/splits.json`, `splits_v2.json`, `dataset_v2_inspection.json`
- `data/raw/README.md`, this file
- All of `src/`, `app/`, `docs/`, `requirements.txt`
