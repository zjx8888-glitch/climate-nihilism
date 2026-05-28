# ClimateBERT — Results & Methods (Jinxi)

> Regenerate numbers after training: `python src/climatebert/train.py`  
> Metrics file: `outputs/climatebert/climatebert_metrics.json`

## Model

We use **[ClimateBERT](https://huggingface.co/climatebert/distilroberta-base-climate-f)** (`climatebert/distilroberta-base-climate-f`), the recommended DistilRoBERTa language model from Webersinke et al. (2021), loaded via Sentence Transformers for embeddings. If the checkpoint is unavailable, the script falls back to `sentence-transformers/all-MiniLM-L6-v2` (logged in metrics).

### Why ClimateBERT for this task

Reddit climate-opinion comments mix scientific vocabulary, denial framing, anxiety, and fatalism. General-purpose embeddings under-represent climate-specific semantics. ClimateBERT was pretrained/fine-tuned on climate corpora, so it should better separate **Climate nihilism** from nearby classes (**climate anxiety**, **Climate nihilism critique**, denial variants) than generic sentence encoders.

## Data & splits

| Item | Value |
|------|--------|
| Source | `data/labeled/recovered_labeled_dataset.csv` (or `data/labeled/final_training_dataset.csv`) |
| Labeled rows | ~1,844 (after dedup) |
| Climate nihilism (full set) | ~66 |
| Split | 70% / 15% / 15% train/val/test via `data/labeled/splits.json`, `random_state=42` |

## Training setup

| Approach | Description |
|----------|-------------|
| **Embeddings + LR (multiclass)** | Mean-pooled ClimateBERT embeddings → balanced `LogisticRegression`, 14 canonical labels |
| **Embeddings + LR (binary)** | Same embeddings → nihilism vs not; PR curve on test set |
| **Fine-tuning (optional)** | `AutoModelForSequenceClassification` on 14 labels; run with `--finetune` |

Command:

```bash
python src/climatebert/train.py
python src/climatebert/train.py --finetune --epochs 3
```

## Results (test set, n=277)

Latest run (`climatebert/distilroberta-base-climate-f`, embeddings + LR):

| Approach | Accuracy | Macro F1 | Weighted F1 | Nihilism F1 |
|----------|----------|----------|-------------|-------------|
| Embedding + LR (multiclass) | 0.329 | 0.230 | 0.350 | 0.300 |
| Embedding + LR (binary nihilism) | — | — | — | F1=0.286 (P=0.222, R=0.400) |
| Fine-tuned (optional) | — | — | — | run with `--finetune` |

**Binary nihilism:** precision 0.222, recall 0.400, F1 0.286, average precision in `embedding_lr_binary_nihilism` in metrics JSON.

**Error analysis (test):** 7 false-positive nihilism, 7 false-negative nihilism; 5 pairs confused with climate anxiety; 0 with Climate nihilism critique (see `error_analysis_summary` in metrics).

**Per-class F1:** see `per_class` in metrics JSON.

**Figures:**

- `outputs/climatebert/climatebert_confusion_matrix.png`
- `outputs/climatebert/climatebert_nihilism_pr_curve.png`

## Error analysis

Saved to `outputs/climatebert/climatebert_error_analysis.csv`:

- False positives for Climate nihilism (predicted nihilism, true otherwise)
- False negatives (true nihilism, missed)
- Pairs confused with **climate anxiety**
- Pairs confused with **Climate nihilism critique**

Summary counts: `error_analysis_summary` in metrics JSON.

## Limitations

1. **Sparse nihilism class** (~66 examples total, few in test) — high variance on nihilism precision/recall.
2. **14-way imbalance** — rare labels (e.g. climate doomism) get low support; macro F1 is dominated by frequent classes.
3. **Reddit style** — sarcasm, quotes, and thread context are not modeled; single-comment classification only.
4. **Fine-tuning** — optional and data-hungry; may overfit without more labels.
5. **Fallback encoder** — if ClimateBERT HF weights fail to load, results are not true ClimateBERT performance.

## Why more labeled nihilism data is needed

Climate nihilism is the project focus but remains the **smallest** high-priority class. Confusions with anxiety (“overwhelmed but engaged”) and nihilism critique (“pushing back on fatalism”) need more boundary examples. Additional human labels on ambiguous posts would improve both embedding+LR and fine-tuned heads more than tuning hyperparameters alone.

## Team boundaries

- **Liu:** TF-IDF baseline → `src/tfidf/train.py` (TODO)
- **Josh:** Streamlit demo → `app/streamlit_app.py` (TODO: load `outputs/predictions/`)
