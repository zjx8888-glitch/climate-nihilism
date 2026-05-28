# Final Results Summary (Recovered Dataset)

**Date:** Rebuilt pipeline after label recovery  
**Primary data:** `data/labeled/recovered_labeled_dataset.csv` → `data/labeled/final_training_dataset.csv`

## Dataset

| Item | Value |
|------|------:|
| Labeled rows | **1,844** (1 duplicate body removed) |
| Climate nihilism (train pool) | **66** |
| Train / val / test | **1,290 / 277 / 277** (70/15/15 stratified) |
| Taxonomy | 14 classes |

Labels were recovered from CSV after Excel `#NAME?` issues; 504 spellings normalized to canonical taxonomy.

## Best model (multiclass test set)

| Metric | TF-IDF + Logistic Regression |
|--------|------------------------------:|
| Accuracy | **0.332** |
| Macro F1 | **0.185** |
| Weighted F1 | **0.325** |
| Climate nihilism precision | 0.167 |
| Climate nihilism recall | 0.200 |
| **Climate nihilism F1** | **0.182** |

## All models (test set)

| Model | Accuracy | Macro F1 | Weighted F1 | Nihilism F1 |
|-------|----------|----------|-------------|-------------|
| TF-IDF + LR | 0.332 | 0.185 | 0.325 | 0.182 |
| TF-IDF + SVM | 0.336 | 0.185 | 0.315 | 0.222 |
| Embedding + LR | 0.256 | 0.181 | 0.279 | **0.267** |
| Binary nihilism (TF-IDF+LR) | 0.968* | 0.182 | 0.182 | 0.182 |

\*High accuracy on binary task reflects class imbalance (~10 nihilism vs ~267 not in test).

## Strongest per-class performance (TF-IDF + LR)

- **Climate denial critique:** F1 ≈ 0.50 (largest class)
- **Climate information:** F1 ≈ 0.48
- **Climate denial:** F1 ≈ 0.22
- **climate anxiety:** F1 ≈ 0.33

Rare classes (apathy, optimism, nihilism critique) often score F1 = 0 on test due to few examples.

## Figures

See `outputs/figures/`:

- `label_distribution.png`
- `model_comparison.png`
- `confusion_matrix_best.png`
- `nihilism_pr_by_model.png`
- `nihilism_top_keywords.png`
- `confidence_distribution.png`

## Artifacts

- `outputs/tfidf/model_metrics.json` — full per-class metrics (TF-IDF legacy)
- `data/labeled/splits.json` — reproducible split hashes
- `outputs/tfidf/models/best_model.json` — best multiclass checkpoint metadata
- `outputs/climatebert/climatebert_metrics.json` — ClimateBERT metrics (Jinxi)

## Limitations

1. **14-way classification** on ~1.8k labels remains difficult; macro F1 ~0.18–0.19.
2. **Climate nihilism** improved vs 335-row era (was 0 F1) but still modest (F1 ~0.18–0.27).
3. **Class imbalance** — denial critique dominates; rare classes underperform.
4. **Label noise** — some recovered labels may still be debatable; human audit on priority classes recommended.

## Next steps

- Human-review priority classes via `streamlit run src/review_app.py`
- Optional LLM silver labels (`src/llm_label_comments.py`) with audit
- Fine-tune ClimateBERT when ready to invest GPU time
- Use **Embedding + LR** if optimizing specifically for nihilism F1 (0.267 on test)
