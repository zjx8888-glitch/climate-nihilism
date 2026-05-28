# Dataset Report (Recovered Labels)

**Generated after label recovery from `Processed Data/preprocessed_comments_2000_to_label.csv`**

## Summary

| Metric | Value |
|--------|------:|
| Source file | `outputs/recovered_labeled_dataset.csv` |
| Total labeled rows (recovered file) | **1,845** |
| Rows in training pipeline (deduped) | **1,844** |
| Broken / unlabeled (excluded) | 2 |
| **Climate nihilism** examples | **66** |
| Taxonomy classes | 14 |

## Label distribution

| Label | Count |
|-------|------:|
| Climate denial critique | 473 |
| Climate information | 317 |
| climate policy critique | 199 |
| climate activism | 183 |
| Not climate opinion | 149 |
| Climate denial | 129 |
| climate opinion critique | 95 |
| **Climate nihilism** | **66** |
| Climate action critique | 63 |
| climate anxiety | 61 |
| climate change importance | 36 |
| Climate optimism | 30 |
| Climate apathy | 28 |
| Climate nihilism critique | 16 |

## Train / validation / test split

Stratified **70% / 15% / 15%** (random seed `42`). Indices saved to `outputs/splits.json`.

| Split | Approx. size |
|-------|-------------:|
| Train | **1,290** |
| Validation | **277** |
| Test | **277** |

## Data quality notes

- Labels were recovered from CSV text after Excel `#NAME?` display issues (formula misinterpretation).
- 504 rows had spelling/casing normalized to the official taxonomy.
- `Climate opinion` mapped to `climate opinion critique`.
- Class imbalance remains (denial critique is largest); use `class_weight='balanced'` in models.

## Files

- `outputs/recovered_labeled_dataset.csv` — canonical supervised export
- `outputs/final_training_dataset.csv` — pipeline training file (same rows + human-verified if added)
- `outputs/broken_label_rows.csv` — 2 rows needing manual labels
