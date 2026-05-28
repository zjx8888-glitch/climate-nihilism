# Recovered (1,844 rows) vs Old (335 rows) Results

Archived old metrics: `outputs/archive/old_335_model_metrics.json`

## Why the old pipeline under-counted labels

The previous pipeline reported **335** training rows because `normalize_label()` did not match case variants (`Climate Denial Critique` vs `Climate denial critique`). ~1,500 valid labels existed in the file but were dropped silently.

After `src/recover_labeled_dataset.py`:

- **1,845** rows recovered from CSV text  
- **1,844** used in training (one duplicate `body_hash` removed)

## Headline comparison (TF-IDF + Logistic Regression, test set)

| Metric | Old (335 rows, 80/20 split) | New (1,844 rows, 70/15/15) | Change |
|--------|----------------------------:|---------------------------:|-------:|
| Test size | ~67 | **277** | +210 |
| Accuracy | 0.209 | **0.332** | **+0.12** |
| Macro F1 | 0.143 | **0.185** | **+0.04** |
| Weighted F1 | — | **0.325** | — |
| **Climate nihilism F1** | **0.000** | **0.182** | **+0.18** |

## Climate nihilism

| | Old | New |
|---|-----|-----|
| Training examples | 8 | **66** |
| Test support (approx.) | 2 | **10** |
| Best multiclass nihilism F1 | 0.00 | **0.18** (LR), **0.27** (Embedding+LR) |
| Binary nihilism classifier | Not trained | F1 **0.18** (high accuracy due to imbalance) |

**Conclusion:** Nihilism detection moved from **non-functional** (0 F1) to **weak but measurable**. Still limited by ~66 training examples and confusion with anxiety/denial critique.

## Why performance changed

### Improvements

1. **5.5× more labeled data** — models see more vocabulary and class patterns.
2. **66 nihilism examples** — enables learning nihilism-specific phrases.
3. **Consistent taxonomy** — no silent label drops from casing.
4. **Larger test set (277)** — metrics are more stable than 67-test estimates.

### Remaining challenges

1. **Macro F1 still low (~0.19)** — 14 similar classes on ~1.8k rows is hard.
2. **Class imbalance** — denial critique (473) vs nihilism critique (16).
3. **Semantic overlap** — anxiety vs nihilism vs nihilism critique.
4. **Weak labels** — not all 1,844 rows are perfect gold standard.

## Model ranking shift

| Model | Old best macro F1 | New macro F1 | New nihilism F1 |
|-------|------------------:|-------------:|----------------:|
| TF-IDF + LR | 0.143 | 0.185 | 0.182 |
| TF-IDF + SVM | 0.138 | 0.185 | 0.222 |
| Embedding + LR | 0.113 | 0.181 | **0.267** |

For **nihilism-specific** performance, **Embedding + LR** is now the best multiclass option (F1 0.267 vs 0.182 LR).

## Recommendation

- Use **recovered dataset** as the only supervised source going forward.
- Report both **macro F1** (overall taxonomy) and **nihilism F1** (project focus).
- Continue human review on nihilism/anxiety border cases.
- Do **not** compare fairly to old 335-row numbers without noting the label-count bug.
