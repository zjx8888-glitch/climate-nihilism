# Methodology

## Task

Multi-class classification of Reddit comments into **14 climate-opinion labels** (see taxonomy in README), with emphasis on **Climate nihilism** detection and binary nihilism vs not.

## Data

- **Gold labels:** ~1,844 manually labeled comments after recovery (`data/labeled/recovered_labeled_dataset.csv`).
- **Splits:** Stratified 70% / 15% / 15% train/val/test, `random_state=42`, stored in `data/labeled/splits.json`.
- **Weak labels:** Optional silver labels in `data/weak_labels/` (auto-label, LLM, human review) — not merged into gold training by default.

## Models

| Approach | Owner | Description |
|----------|--------|-------------|
| ClimateBERT + LR | Jinxi | Mean-pooled `climatebert/distilroberta-base-climate-f` embeddings → balanced logistic regression |
| ClimateBERT fine-tune | Jinxi | Optional `AutoModelForSequenceClassification` on 14 labels |
| TF-IDF + LR / SVM | Liu | Bag-of-words baselines with class weighting |
| Weak labeling | Madeleine | Keywords + TF-IDF similarity + optional zero-shot |

## Evaluation metrics

- Accuracy, macro F1, weighted F1, per-class F1
- Climate nihilism precision / recall / F1
- Confusion matrix, PR curve (binary nihilism)
- Error analysis: FP/FN nihilism, confusion with anxiety and nihilism critique

## Reproducibility

- Fixed random seed: `42`
- Central paths: `src/common/paths.py`
- Do not change `splits.json` without team agreement
